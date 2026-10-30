"""Integration tests against a throwaway SQLite tracking store."""

from dataclasses import replace
import json

from mlflow import MlflowClient
import pandas as pd
import pytest

from mibici_e21 import tracking
from mibici_e21.config import TRACKING_SETTINGS
from mibici_e21.modeling import runs
from mibici_e21.modeling.clusters import ClusterQuestion, build_station_profiles
from mibici_e21.schemas.columns import GoldColumns, StationColumns


@pytest.fixture
def tracking_uri(tmp_path, monkeypatch):
    """Point the project at a SQLite store in a temporary directory."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(
        tracking, "TRACKING_SETTINGS", replace(TRACKING_SETTINGS, TRACKING_URI=uri)
    )
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    return uri


def test_configure_tracking_rejects_a_project_run_in_another_experiment(tracking_uri, monkeypatch):
    """A run that `mlflow run` created in `Default` stops the script with the export to run."""
    run = MlflowClient(tracking_uri).create_run(experiment_id="0")
    monkeypatch.setenv("MLFLOW_RUN_ID", run.info.run_id)

    with pytest.raises(RuntimeError, match="export MLFLOW_EXPERIMENT_NAME="):
        tracking.configure_tracking()


def test_configure_tracking_accepts_a_project_run_in_the_project_experiment(
    tracking_uri, monkeypatch
):
    client = MlflowClient(tracking_uri)
    experiment_id = client.create_experiment(TRACKING_SETTINGS.EXPERIMENT_NAME)
    run = client.create_run(experiment_id=experiment_id)
    monkeypatch.setenv("MLFLOW_RUN_ID", run.info.run_id)

    assert tracking.configure_tracking() == experiment_id


def test_train_run_logs_model_data_plots_and_config(tracking_uri, tmp_path, monkeypatch):
    """A baseline run carries its tags, metrics, model, and organized artifacts."""
    manifest_path = tmp_path / "data_manifest_2023.json"
    manifest = {"year": 2023, "dataset_revision": "abc123", "files": {}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runs, "data_manifest_path", lambda year: manifest_path)

    gold = pd.DataFrame(
        {
            GoldColumns.DATE: pd.Timestamp("2023-01-01"),
            GoldColumns.HOUR: 8,
            GoldColumns.STATION_ID: range(12),
            GoldColumns.DEPARTURES_COUNT: [1, 2, 3, 40, 41, 42, 90, 91, 92, 5, 45, 95],
            GoldColumns.ARRIVALS_COUNT: [2, 1, 3, 39, 44, 40, 88, 95, 90, 4, 43, 99],
        }
    )
    data = build_station_profiles(gold).assign(
        **{StationColumns.LATITUDE: 20.67, StationColumns.LONGITUDE: -103.35}
    )

    tracking.configure_tracking()
    run_id = runs.train_run(ClusterQuestion(), data, "baseline", {}, manifest)

    client = MlflowClient(tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["stage_of_work"] == "baseline"
    assert run.data.tags["question"] == "p3"
    assert run.data.tags["dataset_revision"] == "abc123"
    assert run.data.tags["model_uri"].startswith("models:/")
    assert "silhouette" in run.data.metrics
    artifacts = {
        folder.path: {item.path for item in client.list_artifacts(run_id, folder.path)}
        for folder in client.list_artifacts(run_id)
    }
    assert artifacts["data"] == {"data/data_manifest_2023.json"}
    assert {"config/run_config.json", "config/MLproject"} <= artifacts["config"]
    assert "plots/station_map.png" in artifacts["plots"]
