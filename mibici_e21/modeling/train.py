"""Train the MiBici baseline models and log each run to MLflow.

`MLFLOW_ALLOW_FILE_STORE` is set before MLflow is imported so the local
`mlruns` directory is accepted as a tracking store. Importing this module
points the tracking URI at `mlruns` under `PROJ_ROOT` and selects
`MODELING_SETTINGS.EXPERIMENT_NAME`.
"""

import os
from pathlib import Path
import pickle

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from mibici_e21.config import MODELING_SETTINGS, PROJ_ROOT, RAW_DATA_DIR
from mibici_e21.schemas.columns import GoldColumns

mlflow.set_tracking_uri((PROJ_ROOT / "mlruns").as_uri())
mlflow.set_experiment(MODELING_SETTINGS.EXPERIMENT_NAME)


def load_gold_demand(year: int = 2023) -> pd.DataFrame:
    """Load `demand_aggregated_{year}.parquet` from the gold feature store.

    - `year`: demand year used in the file name. Defaults to 2023.

    `date` is parsed to datetime. Raises `FileNotFoundError` when that file
    is not in `MODELING_SETTINGS.PROCESSED_DIR` (`data/processed`).
    """
    demand_path = MODELING_SETTINGS.PROCESSED_DIR / f"demand_aggregated_{year}.parquet"
    if not demand_path.exists():
        raise FileNotFoundError(
            f"No parquet file for demand for {year} in: {demand_path}. "
            f"Execute 'make materialize-features-range START={year} END={year}' first."
        )

    logger.info(f"Loading gold demand for {year} from {demand_path}")
    df = pd.read_parquet(demand_path, engine="pyarrow")
    df[GoldColumns.DATE] = pd.to_datetime(df[GoldColumns.DATE])
    logger.info(f"Loaded gold demand: {len(df)} rows")
    return df


def log_common_info(problem: str, year: int) -> None:
    """Log the shared MLflow params and tags on the active run.

    - `problem`: tag that names the modeling problem (`P1_regression`,
      `P2_classification`, or `P3_clustering`).
    - `year`: demand year recorded as `period` and in the `dataset` tag.

    `dataset_version` is the text of `data-version.txt` in `RAW_DATA_DIR`
    (`data/raw`). When that file is missing, the version is `v1.0.0`.
    The other param is `seed`. Tags are `team`, `problem`, and `dataset`.
    """
    version_file = RAW_DATA_DIR / "data-version.txt"
    if version_file.exists():
        version = version_file.read_text().strip()
    else:
        version = "v1.0.0"
        logger.warning(f"Dataset version file not found: {version_file}")
    logger.info(f"Dataset version: {version}")

    mlflow.log_params(
        {
            "period": str(year),
            "dataset_version": version,
            "seed": MODELING_SETTINGS.SEED,
        }
    )
    mlflow.set_tags(
        {
            "team": "21",
            "problem": problem,
            "dataset": f"MiBici {year} Gold",
        }
    )


def train_p1(df: pd.DataFrame, year: int) -> None:
    """Train the P1 demand baseline: historical mean by station and hour.

    - `df`: gold demand, one row per station, date, and hour.
    - `year`: demand year logged on the MLflow run.

    Rows before `MODELING_SETTINGS.TEST_SPLIT_DATE` are the training set.
    The prediction for a station and hour is the mean departure count of
    that pair. An unseen pair uses the global training mean. The run
    `P1_Historical_Mean` logs MAE, RMSE, and R², a scatter figure, and
    `p1_model.parquet`.
    """
    logger.info(f"Training P1 historical mean (cutoff {MODELING_SETTINGS.TEST_SPLIT_DATE})")

    train = df[df[GoldColumns.DATE] < MODELING_SETTINGS.TEST_SPLIT_DATE].copy()
    test = df[df[GoldColumns.DATE] >= MODELING_SETTINGS.TEST_SPLIT_DATE].copy()

    historical = (
        train.groupby([GoldColumns.STATION_ID, GoldColumns.HOUR])[GoldColumns.DEPARTURES_COUNT]
        .mean()
        .reset_index(name="prediction")
    )
    global_mean = train[GoldColumns.DEPARTURES_COUNT].mean()
    logger.info(
        f"P1 split: {len(train)} train rows, {len(test)} test rows, "
        f"{len(historical)} station-hour means"
    )

    test_eval = test.merge(historical, on=[GoldColumns.STATION_ID, GoldColumns.HOUR], how="left")
    test_eval["prediction"] = test_eval["prediction"].fillna(global_mean)

    mae = mean_absolute_error(test_eval[GoldColumns.DEPARTURES_COUNT], test_eval["prediction"])
    rmse = np.sqrt(
        mean_squared_error(test_eval[GoldColumns.DEPARTURES_COUNT], test_eval["prediction"])
    )
    r2 = r2_score(test_eval[GoldColumns.DEPARTURES_COUNT], test_eval["prediction"])

    with mlflow.start_run(run_name="P1_Historical_Mean"):
        log_common_info("P1_regression", year)
        mlflow.log_params(
            {
                "model": "historical_mean",
                "cutoff_date": MODELING_SETTINGS.TEST_SPLIT_DATE,
            }
        )
        mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})

        fig, ax = plt.subplots(figsize=(8, 5))
        sample = test_eval.sample(n=min(300, len(test_eval)), random_state=MODELING_SETTINGS.SEED)
        ax.scatter(sample[GoldColumns.DEPARTURES_COUNT], sample["prediction"], alpha=0.5)
        ax.set_xlabel("Actual Trips")
        ax.set_ylabel("Estimated Trips")
        ax.set_title(f"P1 - Demand Regression Baseline ({year})")
        fig.tight_layout()
        mlflow.log_figure(fig, "P1_regression.png")
        plt.close(fig)

        artifact_path = Path("p1_model.parquet")
        historical.to_parquet(artifact_path, index=False)
        mlflow.log_artifact(str(artifact_path))
        artifact_path.unlink()

    logger.info(f"P1 complete: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")


def train_p2(df: pd.DataFrame, year: int) -> None:
    """Train the P2 imbalance baseline: historical mode by station.

    - `df`: gold demand, one row per station, date, and hour.
    - `year`: demand year logged on the MLflow run.

    Departures and arrivals are summed by station and date. Balance is
    arrivals minus departures. A station-day is `possible_empty` at or below
    `-BALANCE_THRESHOLD`, `possible_full` at or above `BALANCE_THRESHOLD`,
    and `balanced` otherwise. The predicted class is that station's training
    mode, or the global training mode when the station is new. The run
    `P2_Historical_Mode` logs macro precision, recall, and F1, a class-count
    figure, and `p2_model.parquet`.
    """
    logger.info(
        "Training P2 historical mode "
        f"(cutoff {MODELING_SETTINGS.TEST_SPLIT_DATE}, "
        f"threshold {MODELING_SETTINGS.BALANCE_THRESHOLD})"
    )

    daily = (
        df.groupby([GoldColumns.DATE, GoldColumns.STATION_ID])[
            [GoldColumns.DEPARTURES_COUNT, GoldColumns.ARRIVALS_COUNT]
        ]
        .sum()
        .reset_index()
    )
    daily["balance"] = daily[GoldColumns.ARRIVALS_COUNT] - daily[GoldColumns.DEPARTURES_COUNT]

    threshold = MODELING_SETTINGS.BALANCE_THRESHOLD
    daily["risk"] = np.select(
        [daily["balance"] <= -threshold, daily["balance"] >= threshold],
        ["possible_empty", "possible_full"],
        default="balanced",
    )

    train = daily[daily[GoldColumns.DATE] < MODELING_SETTINGS.TEST_SPLIT_DATE].copy()
    test = daily[daily[GoldColumns.DATE] >= MODELING_SETTINGS.TEST_SPLIT_DATE].copy()

    classes = (
        train.groupby(GoldColumns.STATION_ID)["risk"].agg(lambda x: x.mode().iloc[0]).reset_index()
    )
    general_class = train["risk"].mode().iloc[0]
    logger.info(
        f"P2 split: {len(train)} train station-days, {len(test)} test station-days, "
        f"{len(classes)} stations"
    )

    test_eval = test.merge(classes, on=GoldColumns.STATION_ID, how="left", suffixes=("", "_pred"))
    test_eval["risk_pred"] = test_eval["risk_pred"].fillna(general_class)

    precision = precision_score(
        test_eval["risk"], test_eval["risk_pred"], average="macro", zero_division=0
    )
    recall = recall_score(
        test_eval["risk"], test_eval["risk_pred"], average="macro", zero_division=0
    )
    f1 = f1_score(test_eval["risk"], test_eval["risk_pred"], average="macro", zero_division=0)

    with mlflow.start_run(run_name="P2_Historical_Mode"):
        log_common_info("P2_classification", year)
        mlflow.log_params(
            {
                "model": "historical_mode_by_station",
                "balance_threshold": threshold,
                "cutoff_date": MODELING_SETTINGS.TEST_SPLIT_DATE,
            }
        )
        mlflow.log_metrics(
            {
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1,
            }
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        test_eval["risk"].value_counts().plot.bar(ax=ax)
        ax.set_title("P2 - Proxy Risk Classes Distribution")
        ax.set_xlabel("Classification Category")
        ax.set_ylabel("Records Count")
        fig.tight_layout()
        mlflow.log_figure(fig, "P2_classification.png")
        plt.close(fig)

        artifact_path = Path("p2_model.parquet")
        classes.to_parquet(artifact_path, index=False)
        mlflow.log_artifact(str(artifact_path))
        artifact_path.unlink()

    logger.info(f"P2 complete: precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}")


def train_p3(df: pd.DataFrame, year: int) -> None:
    """Train the P3 station baseline: scaled KMeans on station totals.

    - `df`: gold demand, one row per station, date, and hour.
    - `year`: demand year logged on the MLflow run.

    Each station is described by total departures, total arrivals, and
    departures during `MODELING_SETTINGS.PICO_HOURS`. Features are scaled
    with `StandardScaler` and clustered with `KMeans`. The run
    `P3_KMeans_3_Clusters` logs the silhouette score, a scatter figure, and
    `p3_pipeline.pkl` with the scaler and the fitted model.
    """
    logger.info(f"Training P3 KMeans ({MODELING_SETTINGS.KMEANS_CLUSTERS} clusters)")

    df_peak = df[df[GoldColumns.HOUR].isin(MODELING_SETTINGS.PICO_HOURS)]

    stations = df.groupby(GoldColumns.STATION_ID).agg(
        departures=(GoldColumns.DEPARTURES_COUNT, "sum"),
        arrivals=(GoldColumns.ARRIVALS_COUNT, "sum"),
    )
    peak_departures = (
        df_peak.groupby(GoldColumns.STATION_ID)[GoldColumns.DEPARTURES_COUNT]
        .sum()
        .rename("peak_departures")
    )
    stations = stations.join(peak_departures).fillna(0)

    features = ["departures", "arrivals", "peak_departures"]
    scaler = StandardScaler()
    X = scaler.fit_transform(stations[features])

    kmeans = KMeans(
        n_clusters=MODELING_SETTINGS.KMEANS_CLUSTERS,
        random_state=MODELING_SETTINGS.SEED,
        n_init=10,
    )
    stations["cluster"] = kmeans.fit_predict(X)
    logger.info(f"Clustered {len(stations)} stations")

    silhouette = silhouette_score(X, stations["cluster"])

    with mlflow.start_run(run_name="P3_KMeans_3_Clusters"):
        log_common_info("P3_clustering", year)
        mlflow.log_params(
            {
                "model": "KMeans",
                "n_clusters": MODELING_SETTINGS.KMEANS_CLUSTERS,
                "n_init": 10,
                "features": ",".join(features),
            }
        )
        mlflow.log_metric("silhouette", silhouette)

        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(
            stations["departures"], stations["arrivals"], c=stations["cluster"], alpha=0.7
        )
        ax.set_xlabel("Total Departures")
        ax.set_ylabel("Total Arrivals")
        ax.set_title("P3 - MiBici Station Clusters")
        fig.colorbar(scatter, ax=ax, label="Cluster")
        fig.tight_layout()
        mlflow.log_figure(fig, "P3_clustering.png")
        plt.close(fig)

        pipeline = {"scaler": scaler, "kmeans": kmeans}
        artifact_path = Path("p3_pipeline.pkl")
        with open(artifact_path, "wb") as f:
            pickle.dump(pipeline, f)
        mlflow.log_artifact(str(artifact_path))
        artifact_path.unlink()

    logger.info(f"P3 complete: silhouette={silhouette:.3f}")


def run_training_pipeline(year: int = 2023) -> None:
    """Train P1, P2, and P3 on the gold demand of one year.

    - `year`: demand year loaded and recorded on each MLflow run. Defaults to 2023.

    The three baselines share that frame and record `year` on each MLflow run.
    """
    logger.info(f"Training baselines for {year}")
    df_gold = load_gold_demand(year)
    train_p1(df_gold, year)
    train_p2(df_gold, year)
    train_p3(df_gold, year)
    logger.info(f"Baseline training complete for {year}")
