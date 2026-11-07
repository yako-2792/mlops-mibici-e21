"""Tests for the API, against small models saved in the format `make export-model` writes."""

import json

from fastapi.testclient import TestClient
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from api.main import create_app
from api.model import REGISTRY_FILE, load_model
from api.schemas import MAX_INSTANCES

LINEAGE = {"name": "mibici-equipo21-p1-demanda", "version": "7", "alias": "champion"}


class ConstantRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, value: float = 0.0):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def _training_frame() -> pd.DataFrame:
    """Every hour and weekday of one station, with the dtypes of the real signature."""
    hours, days = np.meshgrid(np.arange(24), np.arange(7))
    frame = pd.DataFrame(
        {
            "station_id": 42,
            "hour": hours.ravel(),
            "day_of_week": days.ravel(),
        }
    ).astype("int32")
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype("int64")
    return frame


def _save(estimator, path, columns=None):
    """Save `estimator` like an exported model, fitted so it predicts hour + 10 * weekend."""
    frame = _training_frame()
    if columns is not None:
        frame = frame[columns]
    target = frame["hour"] + 10 * frame["is_weekend"]
    estimator.fit(frame, target)
    mlflow.sklearn.save_model(
        estimator,
        str(path),
        signature=infer_signature(frame, estimator.predict(frame)),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    (path / REGISTRY_FILE).write_text(json.dumps({**LINEAGE, "run_id": "abc123"}))
    return path


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return _save(LinearRegression(), tmp_path_factory.mktemp("models") / "p1-demanda")


@pytest.fixture
def client(model_dir):
    with TestClient(create_app(lambda: load_model(str(model_dir)))) as test_client:
        yield test_client


def _instance(**overrides):
    return {"station_id": 42, "hour": 8, "day_of_week": 0, **overrides}


def test_predict_returns_one_prediction_per_instance_with_the_model_version(client):
    response = client.post(
        "/predict", json={"instances": [_instance(), _instance(hour=18, day_of_week=2)]}
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["model_name"], body["model_version"]) == (LINEAGE["name"], "7")
    assert [p["hour"] for p in body["predictions"]] == [8, 18]
    assert [p["predicted_departures"] for p in body["predictions"]] == pytest.approx([8, 18])


def test_weekend_is_derived_from_the_day_of_week(client):
    response = client.post(
        "/predict", json={"instances": [_instance(day_of_week=4), _instance(day_of_week=5)]}
    )

    predictions = [p["predicted_departures"] for p in response.json()["predictions"]]
    assert predictions == pytest.approx([8, 18])


@pytest.mark.parametrize(
    "instance",
    [
        _instance(station_id=0),
        _instance(hour=-1),
        _instance(hour=24),
        _instance(day_of_week=7),
        _instance(hour="8"),
        _instance(hour=8.5),
        _instance(hour=True),
        _instance(is_weekend=1),
        {"station_id": 42, "hour": 8},
    ],
    ids=[
        "station_zero",
        "hour_negative",
        "hour_24",
        "day_7",
        "hour_string",
        "hour_float",
        "hour_bool",
        "extra_field",
        "missing_field",
    ],
)
def test_predict_rejects_an_invalid_instance(client, instance):
    response = client.post("/predict", json={"instances": [instance]})

    assert response.status_code == 422


@pytest.mark.parametrize("size", [0, MAX_INSTANCES + 1])
def test_predict_rejects_an_empty_or_oversized_batch(client, size):
    response = client.post("/predict", json={"instances": [_instance()] * size})

    assert response.status_code == 422


def test_predict_rejects_an_invalid_model_output(tmp_path):
    model_dir = _save(ConstantRegressor(-1.0), tmp_path / "negative")

    with TestClient(create_app(lambda: load_model(str(model_dir)))) as client:
        response = client.post("/predict", json={"instances": [_instance()]})

    assert response.status_code == 500


def test_model_and_health_report_the_registry_version(client):
    assert client.get("/model").json() == {**LINEAGE, "run_id": "abc123"}
    assert client.get("/health").json() == {"status": "ok", "model_version": "7"}


def test_load_model_rejects_a_model_with_other_columns(tmp_path):
    model_dir = _save(LinearRegression(), tmp_path / "other", columns=["hour", "is_weekend"])

    with pytest.raises(ValueError, match="must take the columns"):
        load_model(str(model_dir))


def test_load_model_reads_the_uri_from_the_environment(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_URI", str(model_dir))

    assert load_model().info.version == "7"
