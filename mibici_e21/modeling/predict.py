"""Score the MiBici baseline models from their MLflow artifacts.

`MLFLOW_ALLOW_FILE_STORE` is set before MLflow is imported so the local
`mlruns` directory is accepted as a tracking store. Importing this module
points the tracking URI at `mlruns` under `PROJ_ROOT`.
"""

import os
import pickle

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from mibici_e21.config import MODELING_SETTINGS, PREDICT_SETTINGS
from mibici_e21.schemas.columns import GoldColumns

mlflow.set_tracking_uri(MODELING_SETTINGS.TRACKING_URI)


def get_latest_run_id(problem_tag: str) -> str:
    """Return the latest finished MLflow run for a problem tag.

    - `problem_tag`: tag stored as `problem` (`P1_regression`,
      `P2_classification`, or `P3_clustering`).

    The search uses `MODELING_SETTINGS.EXPERIMENT_NAME`. Raises `ValueError`
    when that experiment or tag has no run.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MODELING_SETTINGS.EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(
            f"Experiment '{MODELING_SETTINGS.EXPERIMENT_NAME}' not found in MLflow store."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.problem = '{problem_tag}' and tags.stage_of_work = 'baseline'",
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No MLflow runs found for problem tag: '{problem_tag}'")

    run_id = runs[0].info.run_id
    logger.info(f"Using latest {problem_tag} run {run_id}")
    return run_id


def load_p1_model(run_id: str | None = None) -> pd.DataFrame:
    """Load the P1 station-hour mean table from an MLflow run.

    - `run_id`: run that holds `p1_model.parquet`. Defaults to the latest
      finished run tagged `P1_regression`.

    Raises `ValueError` when that experiment or run does not exist.
    """
    if run_id is None:
        run_id = get_latest_run_id("P1_regression")

    artifact_uri = f"runs:/{run_id}/p1_model.parquet"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
    lookup = pd.read_parquet(local_path)
    logger.info(f"Loaded P1 model from {artifact_uri}: {len(lookup)} station-hour means")
    return lookup


def predict_p1(df: pd.DataFrame, run_id: str | None = None) -> pd.DataFrame:
    """Predict departures with the P1 station-hour mean.

    - `df`: rows that include `station_id` and `hour`.
    - `run_id`: run that holds `p1_model.parquet`. Defaults to the latest
      finished run tagged `P1_regression`.

    An unseen station-hour pair uses the mean of the lookup table.
    The returned frame adds `predicted_departures`.
    """
    p1_lookup = load_p1_model(run_id)
    out = df.copy()

    out = out.merge(
        p1_lookup,
        on=[GoldColumns.STATION_ID, GoldColumns.HOUR],
        how="left",
    )

    global_mean = p1_lookup["prediction"].mean() if "prediction" in p1_lookup.columns else 0.0
    out["predicted_departures"] = out["prediction"].fillna(global_mean)
    out.drop(columns=["prediction"], inplace=True, errors="ignore")

    logger.info(f"P1 predictions: {len(out)} rows")
    return out


def load_p2_model(run_id: str | None = None) -> pd.DataFrame:
    """Load the P2 station-class table from an MLflow run.

    - `run_id`: run that holds `p2_model.parquet`. Defaults to the latest
      finished run tagged `P2_classification`.

    Raises `ValueError` when that experiment or run does not exist.
    """
    if run_id is None:
        run_id = get_latest_run_id("P2_classification")

    artifact_uri = f"runs:/{run_id}/p2_model.parquet"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
    lookup = pd.read_parquet(local_path)
    logger.info(f"Loaded P2 model from {artifact_uri}: {len(lookup)} stations")
    return lookup


def predict_p2(df: pd.DataFrame, run_id: str | None = None) -> pd.DataFrame:
    """Predict imbalance risk with the P2 station mode.

    - `df`: rows that include `station_id`.
    - `run_id`: run that holds `p2_model.parquet`. Defaults to the latest
      finished run tagged `P2_classification`.

    An unseen station uses the mode of the lookup table, or `balanced` when
    that table has no class. The returned frame adds `predicted_risk`.
    """
    p2_lookup = load_p2_model(run_id)
    out = df.copy()

    out = out.merge(p2_lookup, on=GoldColumns.STATION_ID, how="left")

    general_mode = (
        p2_lookup["risk"].mode().iloc[0]
        if "risk" in p2_lookup.columns and not p2_lookup["risk"].empty
        else "balanced"
    )
    out["predicted_risk"] = out["risk"].fillna(general_mode)
    out.drop(columns=["risk"], inplace=True, errors="ignore")

    logger.info(f"P2 predictions: {len(out)} rows")
    return out


def load_p3_model(run_id: str | None = None) -> dict:
    """Load the P3 scaler and KMeans pipeline from an MLflow run.

    - `run_id`: run that holds `p3_pipeline.pkl`. Defaults to the latest
      finished run tagged `P3_clustering`.

    Returns a dict with `scaler` and `kmeans`. Raises `ValueError` when that
    experiment or run does not exist.
    """
    if run_id is None:
        run_id = get_latest_run_id("P3_clustering")

    artifact_uri = f"runs:/{run_id}/p3_pipeline.pkl"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
    logger.info(f"Loaded P3 pipeline from {artifact_uri}")

    with open(local_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def predict_p3(df: pd.DataFrame, run_id: str | None = None) -> pd.DataFrame:
    """Assign stations to a P3 cluster.

    - `df`: one row per station, with `departures`, `arrivals`, and
      `peak_departures`.
    - `run_id`: run that holds `p3_pipeline.pkl`. Defaults to the latest
      finished run tagged `P3_clustering`.

    Features are scaled with the saved `StandardScaler` and scored with the
    saved `KMeans`. Raises `KeyError` when a feature column is missing.
    The returned frame adds `predicted_cluster`.
    """
    pipeline = load_p3_model(run_id)
    scaler = pipeline["scaler"]
    kmeans = pipeline["kmeans"]

    features = ["departures", "arrivals", "peak_departures"]
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Input DataFrame missing required features for P3: {missing_cols}")

    out = df.copy()
    X_scaled = scaler.transform(out[features])
    out["predicted_cluster"] = kmeans.predict(X_scaled)

    logger.info(f"P3 predictions: {len(out)} stations")
    return out


def run_batch_inference(year: int = 2024, save_csv: bool = False) -> None:
    """Score one year of gold demand with P1, P2, and P3 and write the predictions.

    - `year`: demand year used in `demand_aggregated_{year}.parquet`.
      Defaults to 2024.
    - `save_csv`: when true, also write a CSV next to each Parquet file.
      Defaults to false.

    P1 and P2 are scored on every gold row. P3 is scored on station totals
    from that year. Files are `predictions_p1_{year}.parquet`,
    `predictions_p2_{year}.parquet`, and `predictions_p3_{year}.parquet`,
    written to `PREDICT_SETTINGS.P1_OUTPUT_DIR`, `P2_OUTPUT_DIR`, and
    `P3_OUTPUT_DIR` (`data/processed`). Raises `FileNotFoundError` when the
    gold file is missing.
    """
    gold_path = MODELING_SETTINGS.PROCESSED_DIR / f"demand_aggregated_{year}.parquet"
    if not gold_path.exists():
        raise FileNotFoundError(
            f"No parquet file for demand for {year} in: {gold_path}. "
            f"Execute 'make materialize-features-range START={year} END={year}' first."
        )

    logger.info(f"Running inference for {year}")
    logger.info(f"Loading gold demand for {year} from {gold_path}")
    df_gold = pd.read_parquet(gold_path, engine="pyarrow")
    logger.info(f"Loaded gold demand: {len(df_gold)} rows")

    p1_predictions = predict_p1(df_gold)
    p1_out_path = PREDICT_SETTINGS.P1_OUTPUT_DIR / f"predictions_p1_{year}.parquet"
    p1_predictions.to_parquet(p1_out_path, index=False)
    logger.info(f"Wrote P1 predictions to {p1_out_path}")

    p2_predictions = predict_p2(df_gold)
    p2_out_path = PREDICT_SETTINGS.P2_OUTPUT_DIR / f"predictions_p2_{year}.parquet"
    p2_predictions.to_parquet(p2_out_path, index=False)
    logger.info(f"Wrote P2 predictions to {p2_out_path}")

    df_peak = df_gold[df_gold[GoldColumns.HOUR].isin(MODELING_SETTINGS.PICO_HOURS)]
    stations_df = (
        df_gold.groupby(GoldColumns.STATION_ID)
        .agg(
            departures=(GoldColumns.DEPARTURES_COUNT, "sum"),
            arrivals=(GoldColumns.ARRIVALS_COUNT, "sum"),
        )
        .reset_index()
    )
    peak_deps = (
        df_peak.groupby(GoldColumns.STATION_ID)[GoldColumns.DEPARTURES_COUNT]
        .sum()
        .rename("peak_departures")
    )
    stations_df = stations_df.merge(peak_deps, on=GoldColumns.STATION_ID, how="left").fillna(0)

    p3_predictions = predict_p3(stations_df)
    p3_out_path = PREDICT_SETTINGS.P3_OUTPUT_DIR / f"predictions_p3_{year}.parquet"
    p3_predictions.to_parquet(p3_out_path, index=False)
    logger.info(f"Wrote P3 predictions to {p3_out_path}")

    if save_csv:
        p1_csv_path = PREDICT_SETTINGS.P1_OUTPUT_DIR / f"predictions_p1_{year}.csv"
        p2_csv_path = PREDICT_SETTINGS.P2_OUTPUT_DIR / f"predictions_p2_{year}.csv"
        p3_csv_path = PREDICT_SETTINGS.P3_OUTPUT_DIR / f"predictions_p3_{year}.csv"
        p1_predictions.to_csv(p1_csv_path, index=False)
        p2_predictions.to_csv(p2_csv_path, index=False)
        p3_predictions.to_csv(p3_csv_path, index=False)
        logger.info(f"Wrote P1 predictions to {p1_csv_path}")
        logger.info(f"Wrote P2 predictions to {p2_csv_path}")
        logger.info(f"Wrote P3 predictions to {p3_csv_path}")

    logger.info(f"Inference complete for {year}")
