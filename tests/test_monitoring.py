"""Tests for the monitoring periods, the drift report, the retraining, and the A/B test."""

from dataclasses import replace
from datetime import date

from mlflow import MlflowClient
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from mibici_e21 import tracking
from mibici_e21.config import TRACKING_SETTINGS
from mibici_e21.modeling.demand import DemandQuestion
from mibici_e21.monitoring import drift, retrain
from mibici_e21.monitoring.ab_test import (
    Bootstrap,
    decide,
    metrics_table,
    paired_bootstrap,
    segments,
    traffic_group,
)
from mibici_e21.monitoring.periods import Period, unseen_station_share
from mibici_e21.schemas.columns import GoldColumns

QUESTION = DemandQuestion()


def _demand(days: int, stations: list[int], seed: int = 0, start: str = "2023-01-01"):
    """Build gold demand with every station at every hour of `days` consecutive days."""
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product(
        [pd.date_range(start, periods=days, freq="D"), range(24), stations],
        names=[GoldColumns.DATE, GoldColumns.HOUR, GoldColumns.STATION_ID],
    )
    frame = index.to_frame(index=False)
    frame[GoldColumns.DAY_OF_WEEK] = frame[GoldColumns.DATE].dt.dayofweek
    frame[GoldColumns.IS_WEEKEND] = (frame[GoldColumns.DAY_OF_WEEK] >= 5).astype(int)
    frame[GoldColumns.DEPARTURES_COUNT] = rng.poisson(2 + frame[GoldColumns.HOUR] % 5)
    return frame


# Periods


def test_period_reports_its_years_label_and_rows():
    period = Period("x", date(2023, 11, 1), date(2024, 2, 1))
    dates = pd.Series(pd.to_datetime(["2023-10-31", "2023-11-01", "2024-01-31", "2024-02-01"]))

    assert period.years == [2023, 2024]
    assert period.label == "2023-11-01 to 2024-01-31"
    assert period.contains(dates).tolist() == [False, True, True, False]


def test_period_rejects_an_end_before_its_start():
    with pytest.raises(ValueError, match="before it starts"):
        Period("x", date(2024, 1, 1), date(2024, 1, 1))


def test_quarters_are_clipped_to_the_period():
    quarters = Period("x", date(2024, 2, 15), date(2024, 8, 1)).quarters()

    assert [(q.name, q.start, q.end) for q in quarters] == [
        ("2024Q1", date(2024, 2, 15), date(2024, 4, 1)),
        ("2024Q2", date(2024, 4, 1), date(2024, 7, 1)),
        ("2024Q3", date(2024, 7, 1), date(2024, 8, 1)),
    ]


def test_unseen_station_share_counts_rows_not_stations():
    reference = pd.DataFrame({GoldColumns.STATION_ID: [1, 2]})
    current = pd.DataFrame({GoldColumns.STATION_ID: [1, 1, 1, 3]})

    assert unseen_station_share(reference, current) == 0.25


# Drift


def test_drift_flags_a_new_mix_of_stations_and_not_the_calendar():
    """New stations move station_id past the threshold while hour and weekday stay put."""
    reference = _demand(14, stations=list(range(10)))
    current = _demand(14, stations=list(range(5, 20)), seed=1, start="2024-01-01")

    _, result = drift.compare(reference, current, with_summary=False)

    scores = result.scores.set_index("column")
    assert set(scores.index) == {str(column) for column in drift.COLUMNS}
    assert result.drifted_columns == ["station_id"]
    assert scores.loc["station_id", "method"] == drift.CATEGORICAL_METHOD
    assert scores.loc["hour", "score"] < 0.01
    assert result.unseen_station_share == pytest.approx(10 / 15)


def test_drift_by_period_has_one_row_per_period_with_data():
    reference = _demand(7, stations=[1, 2])
    current = _demand(10, stations=[1, 2, 3], start="2024-03-25")
    periods = Period("current", date(2024, 1, 1), date(2024, 7, 1)).quarters()

    table = drift.drift_by_period(reference, current, periods)

    assert table["period"].tolist() == ["2024Q1", "2024Q2"]
    assert table["rows"].sum() == len(current)
    assert {f"drift_{column}" for column in drift.COLUMNS} <= set(table.columns)


# A/B test


def test_paired_bootstrap_of_identical_arms_is_zero():
    dates = pd.Series(pd.date_range("2024-10-01", periods=30, freq="D").repeat(10))
    errors = np.random.default_rng(0).exponential(1.0, len(dates))

    result, deltas = paired_bootstrap(dates, errors, errors, samples=200)

    assert result.delta == 0
    assert result.low == result.high == 0
    assert len(deltas) == 200


def test_paired_bootstrap_finds_a_consistently_better_arm():
    dates = pd.Series(pd.date_range("2024-10-01", periods=30, freq="D").repeat(10))
    errors_a = np.random.default_rng(0).exponential(1.0, len(dates))

    result, _ = paired_bootstrap(dates, errors_a, errors_a * 0.9, samples=200)

    assert result.delta < 0
    assert result.high < 0
    assert result.share_below_zero == 1
    assert result.days == 30


def test_traffic_group_is_stable_and_uses_both_groups():
    stations = pd.Series(range(200))

    first, second = traffic_group(stations), traffic_group(stations)

    assert first.equals(second)
    assert set(first) == {"A", "B"}
    assert traffic_group(pd.Series([7, 7, 7])).nunique() == 1


def test_metrics_table_splits_known_and_new_stations():
    data = _demand(2, stations=[1, 2])
    target = data[GoldColumns.DEPARTURES_COUNT].to_numpy().astype(float)
    new = (data[GoldColumns.STATION_ID] == 2).to_numpy()
    predictions = {"A": target + 1, "B": np.where(new, target, target + 1)}

    table = metrics_table(QUESTION, data, predictions, segments(data, np.array([1])))

    mae = table[table["metric"] == "MAE"].set_index("segment")
    assert mae.loc["known_stations", "delta"] == 0
    assert mae.loc["new_stations", "A"] == 1
    assert mae.loc["new_stations", "B"] == 0
    assert mae.loc["all", "delta"] == pytest.approx(-0.5)


def _table(known_delta: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"segment": ["known_stations"], "metric": ["MAE"], "A": [1.0], "B": [1.0 + known_delta]}
    ).assign(delta=known_delta)


def _bootstrap(low: float, high: float) -> Bootstrap:
    return Bootstrap((low + high) / 2, low, high, 0.5, 100, 0.95, 92)


@pytest.mark.parametrize(
    ("low", "high", "known_delta", "winner"),
    [
        (-0.05, -0.01, 0.0, "B"),
        (-0.05, 0.01, 0.0, "A"),
        (-0.05, -0.01, 0.05, "A"),
        (-0.05, -0.01, 0.02, "B"),
    ],
)
def test_decide_needs_a_significant_and_safe_improvement(low, high, known_delta, winner):
    decision = decide(_table(known_delta), _bootstrap(low, high), tolerance=0.02)

    assert decision.winner == winner
    assert len(decision.reasons) == 2


# Retraining


@pytest.fixture
def tracking_uri(tmp_path, monkeypatch):
    """Point the project at a SQLite store in a temporary directory."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(
        tracking, "TRACKING_SETTINGS", replace(TRACKING_SETTINGS, TRACKING_URI=uri)
    )
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    tracking.configure_tracking()
    return uri


def _register_champion(data: pd.DataFrame) -> str:
    """Log and register a small random forest as the P1 champion. Returns its version."""
    features = list(QUESTION.feature_columns)
    estimator = RandomForestRegressor(n_estimators=3, max_depth=4, random_state=0)
    estimator.fit(data[features], data[QUESTION.target_column])
    with mlflow.start_run():
        mlflow.log_params({"model": "rf", "n_estimators": 3, "max_depth": 4})
        info = mlflow.sklearn.log_model(
            estimator,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    name = QUESTION.registered_model_name
    version = mlflow.register_model(info.model_uri, name)
    MlflowClient().set_registered_model_alias(name, "champion", version.version)
    return str(version.version)


def test_resolve_recipe_copies_the_champion_and_applies_overrides(tracking_uri):
    champion = _register_champion(_demand(2, stations=[1, 2]))

    copied = retrain.resolve_recipe(QUESTION, None, {"n_estimators": None, "max_depth": None})
    changed = retrain.resolve_recipe(QUESTION, None, {"n_estimators": None, "max_depth": 6})

    assert copied == retrain.Recipe("rf", {"n_estimators": 3, "max_depth": 4}, champion)
    assert changed == retrain.Recipe("rf", {"n_estimators": 3, "max_depth": 6}, None)


def test_resolve_recipe_with_another_model_needs_its_hyperparameters():
    with pytest.raises(ValueError, match="needs max_depth"):
        retrain.resolve_recipe(QUESTION, "hgb", {"n_estimators": 10, "max_depth": None})


def test_evaluate_window_rejects_an_empty_validation():
    data = _demand(3, stations=[1])
    estimator = QUESTION.build_estimator("rf", {"n_estimators": 2, "max_depth": 2})

    with pytest.raises(ValueError, match="empty train or val"):
        retrain.evaluate_window(QUESTION, data, estimator, date(2030, 1, 1))


def test_retrain_registers_a_challenger_and_keeps_the_champion(tracking_uri):
    """The new version gets challenger, its window as tags, and champion does not move."""
    data = _demand(6, stations=[1, 2, 3])
    champion = _register_champion(data)
    recipe = retrain.Recipe("rf", {"n_estimators": 3, "max_depth": 4}, champion)
    period = Period("retrain", date(2023, 1, 1), date(2023, 1, 7))
    manifest = {"year": 2023, "dataset_revision": "abc123", "files": {}}

    run_id = retrain.retrain_run(QUESTION, data, recipe, period, date(2023, 1, 5), [manifest])
    version = retrain.register_challenger(QUESTION, run_id)

    client = MlflowClient(tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["stage_of_work"] == "retrain"
    assert run.data.tags["train_end"] == "2023-01-06"
    assert "val_MAE" in run.data.metrics
    model = client.get_registered_model(QUESTION.registered_model_name)
    assert {alias: str(v) for alias, v in model.aliases.items()} == {
        "champion": champion,
        "challenger": str(version.version),
    }
    assert version.tags["retrained_from_version"] == champion
    assert version.tags["train_start"] == "2023-01-01"
