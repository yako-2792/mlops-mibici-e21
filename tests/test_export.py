"""Tests for exporting a registered model from a throwaway SQLite registry."""

import json

import mlflow
from mlflow import MlflowClient
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from mibici_e21.modeling.export import REGISTRY_FILE, export_model, parse_args
from mibici_e21.modeling.questions import QUESTIONS

QUESTION = QUESTIONS["p1"]


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Register two versions of the P1 model and point `champion` at the first one."""
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    features = pd.DataFrame({"hour": [7, 8, 9]})
    for constant in (1.0, 2.0):
        model = DummyRegressor(strategy="constant", constant=constant).fit(features, [0, 0, 0])
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                model, name="model", registered_model_name=QUESTION.registered_model_name
            )
    client = MlflowClient()
    client.set_registered_model_alias(QUESTION.registered_model_name, "champion", "1")
    yield client
    mlflow.set_tracking_uri(None)


def test_export_resolves_the_alias_and_records_the_version(registry, tmp_path):
    target = tmp_path / "export"

    exported = export_model(QUESTION, alias="champion", target=target)

    assert (exported.name, exported.version, exported.alias) == (
        QUESTION.registered_model_name,
        "1",
        "champion",
    )
    assert exported.run_id == registry.get_model_version(exported.name, "1").run_id
    assert json.loads((target / REGISTRY_FILE).read_text())["version"] == "1"
    model = mlflow.pyfunc.load_model(str(target))
    assert model.predict(pd.DataFrame({"hour": [8]})).tolist() == [1.0]


def test_export_of_a_pinned_version_ignores_the_alias(registry, tmp_path):
    exported = export_model(QUESTION, alias="champion", version="2", target=tmp_path / "export")

    assert (exported.version, exported.alias) == ("2", None)


def test_export_replaces_the_files_of_a_previous_export(registry, tmp_path):
    target = tmp_path / "export"
    target.mkdir()
    (target / "stale.pkl").write_text("old model")

    export_model(QUESTION, target=target)

    assert not (target / "stale.pkl").exists()
    assert (target / "MLmodel").exists()


def test_parse_args_rejects_an_alias_and_a_version_together():
    with pytest.raises(SystemExit):
        parse_args(["--question", "p1", "--alias", "champion", "--version", "3"])
