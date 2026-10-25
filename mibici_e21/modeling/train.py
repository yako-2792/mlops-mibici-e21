"""Train the MiBici baseline and candidate models and log each run to MLflow."""

import argparse
import os
from pathlib import Path
import pickle
import subprocess

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

from mibici_e21.config import MODELING_SETTINGS, RAW_DATA_DIR
from mibici_e21.schemas.columns import GoldColumns

os.environ.pop("MLFLOW_RUN_ID", None)
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
mlflow.set_tracking_uri(MODELING_SETTINGS.TRACKING_URI)
mlflow.set_experiment(MODELING_SETTINGS.EXPERIMENT_NAME)


def get_git_commit() -> str:
    """Return short git commit SHA for traceability."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def load_gold_demand(year: int = MODELING_SETTINGS.DEFAULT_YEAR) -> pd.DataFrame:
    """Load `demand_aggregated_{year}.parquet` from the gold feature store."""
    demand_path = MODELING_SETTINGS.PROCESSED_DIR / f"demand_aggregated_{year}.parquet"
    if not demand_path.exists():
        raise FileNotFoundError(
            f"No parquet file for demand for {year} in: {demand_path}. "
            f"Execute 'make materialize-yearly-features YEAR={year}' first."
        )

    logger.info(f"Loading gold demand for {year} from {demand_path}")
    df = pd.read_parquet(demand_path, engine="pyarrow")
    df[GoldColumns.DATE] = pd.to_datetime(df[GoldColumns.DATE])
    logger.info(f"Loaded gold demand: {len(df)} rows")
    return df


def log_common_info(problem: str, year: int) -> None:
    """Log the shared MLflow params and tags on the active run."""
    version_file = RAW_DATA_DIR / "data-version.txt"
    if version_file.exists():
        version = version_file.read_text().strip()
    else:
        version = "v1.0.0"
        logger.warning(f"Dataset version file not found: {version_file}")

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
            "git_commit": get_git_commit(),
        }
    )


def train_p1(
    df: pd.DataFrame,
    year: int = MODELING_SETTINGS.DEFAULT_YEAR,
    model_type: str = "baseline",
    n_estimators: int = MODELING_SETTINGS.DEFAULT_N_ESTIMATORS,
    max_depth: int = MODELING_SETTINGS.DEFAULT_MAX_DEPTH,
) -> None:
    """Train P1 demand model (`baseline` historical mean or `rf` RandomForestRegressor)."""
    logger.info(f"Training P1 ({model_type}) with cutoff {MODELING_SETTINGS.TEST_SPLIT_DATE}")

    train = df[df[GoldColumns.DATE] < MODELING_SETTINGS.TEST_SPLIT_DATE].copy()
    test = df[df[GoldColumns.DATE] >= MODELING_SETTINGS.TEST_SPLIT_DATE].copy()

    feature_cols = [
        GoldColumns.STATION_ID,
        GoldColumns.HOUR,
        GoldColumns.DAY_OF_WEEK,
        GoldColumns.IS_WEEKEND,
    ]

    if model_type == "rf":
        run_name = f"P1_RF_est{n_estimators}_depth{max_depth}"
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=MODELING_SETTINGS.SEED,
            n_jobs=-1,
        )
        model.fit(train[feature_cols], train[GoldColumns.DEPARTURES_COUNT])
        test_eval = test.copy()
        test_eval["prediction"] = model.predict(test[feature_cols])
    else:
        run_name = "P1_Historical_Mean"
        historical = (
            train.groupby([GoldColumns.STATION_ID, GoldColumns.HOUR])[GoldColumns.DEPARTURES_COUNT]
            .mean()
            .reset_index(name="prediction")
        )
        global_mean = train[GoldColumns.DEPARTURES_COUNT].mean()
        test_eval = test.merge(
            historical, on=[GoldColumns.STATION_ID, GoldColumns.HOUR], how="left"
        )
        test_eval["prediction"] = test_eval["prediction"].fillna(global_mean)

    mae = mean_absolute_error(test_eval[GoldColumns.DEPARTURES_COUNT], test_eval["prediction"])
    rmse = float(
        np.sqrt(
            mean_squared_error(test_eval[GoldColumns.DEPARTURES_COUNT], test_eval["prediction"])
        )
    )
    r2 = r2_score(test_eval[GoldColumns.DEPARTURES_COUNT], test_eval["prediction"])

    with mlflow.start_run(run_name=run_name):
        log_common_info("P1_regression", year)
        mlflow.log_params(
            {
                "model": model_type,
                "cutoff_date": MODELING_SETTINGS.TEST_SPLIT_DATE,
                "n_estimators": n_estimators if model_type == "rf" else None,
                "max_depth": max_depth if model_type == "rf" else None,
            }
        )
        mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})

        fig, ax = plt.subplots(figsize=(8, 5))
        sample = test_eval.sample(n=min(300, len(test_eval)), random_state=MODELING_SETTINGS.SEED)
        ax.scatter(sample[GoldColumns.DEPARTURES_COUNT], sample["prediction"], alpha=0.5)
        ax.set_xlabel("Actual Trips")
        ax.set_ylabel("Estimated Trips")
        ax.set_title(f"P1 - Demand Regression ({model_type} - {year})")
        fig.tight_layout()
        mlflow.log_figure(fig, "P1_regression.png")
        plt.close(fig)

        if model_type == "rf":
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="mibici_p1_demand",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
        else:
            artifact_path = Path("p1_model.parquet")
            historical.to_parquet(artifact_path, index=False)
            mlflow.log_artifact(str(artifact_path))
            artifact_path.unlink()

    logger.info(f"P1 ({model_type}) complete: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")


def train_p2(
    df: pd.DataFrame,
    year: int = MODELING_SETTINGS.DEFAULT_YEAR,
    model_type: str = "baseline",
    n_estimators: int = MODELING_SETTINGS.DEFAULT_N_ESTIMATORS,
    max_depth: int = MODELING_SETTINGS.DEFAULT_MAX_DEPTH,
) -> None:
    """Train P2 station risk model (`baseline` historical mode or `rf` RandomForestClassifier)."""
    logger.info(
        f"Training P2 ({model_type}) "
        f"(cutoff {MODELING_SETTINGS.TEST_SPLIT_DATE}, threshold {MODELING_SETTINGS.BALANCE_THRESHOLD})"
    )

    daily = (
        df.groupby([GoldColumns.DATE, GoldColumns.STATION_ID])[
            [GoldColumns.DEPARTURES_COUNT, GoldColumns.ARRIVALS_COUNT]
        ]
        .sum()
        .reset_index()
    )
    daily["dayofweek"] = daily[GoldColumns.DATE].dt.dayofweek
    daily["balance"] = daily[GoldColumns.ARRIVALS_COUNT] - daily[GoldColumns.DEPARTURES_COUNT]

    threshold = MODELING_SETTINGS.BALANCE_THRESHOLD
    daily["risk"] = np.select(
        [daily["balance"] <= -threshold, daily["balance"] >= threshold],
        ["possible_empty", "possible_full"],
        default="balanced",
    )

    train = daily[daily[GoldColumns.DATE] < MODELING_SETTINGS.TEST_SPLIT_DATE].copy()
    test = daily[daily[GoldColumns.DATE] >= MODELING_SETTINGS.TEST_SPLIT_DATE].copy()

    if model_type == "rf":
        run_name = f"P2_RF_est{n_estimators}_depth{max_depth}"
        feature_cols = [GoldColumns.STATION_ID, "dayofweek", GoldColumns.DEPARTURES_COUNT]
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=MODELING_SETTINGS.SEED,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(train[feature_cols], train["risk"])
        test_eval = test.copy()
        test_eval["risk_pred"] = model.predict(test[feature_cols])
    else:
        run_name = "P2_Historical_Mode"
        classes = (
            train.groupby(GoldColumns.STATION_ID)["risk"]
            .agg(lambda x: x.mode().iloc[0])
            .reset_index()
        )
        general_class = train["risk"].mode().iloc[0]
        test_eval = test.merge(
            classes, on=GoldColumns.STATION_ID, how="left", suffixes=("", "_pred")
        )
        test_eval["risk_pred"] = test_eval["risk_pred"].fillna(general_class)

    precision = precision_score(
        test_eval["risk"], test_eval["risk_pred"], average="macro", zero_division=0
    )
    recall = recall_score(
        test_eval["risk"], test_eval["risk_pred"], average="macro", zero_division=0
    )
    f1 = f1_score(test_eval["risk"], test_eval["risk_pred"], average="macro", zero_division=0)

    with mlflow.start_run(run_name=run_name):
        log_common_info("P2_classification", year)
        mlflow.log_params(
            {
                "model": model_type,
                "balance_threshold": threshold,
                "cutoff_date": MODELING_SETTINGS.TEST_SPLIT_DATE,
                "n_estimators": n_estimators if model_type == "rf" else None,
                "max_depth": max_depth if model_type == "rf" else None,
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
        ax.set_title(f"P2 - Proxy Risk Classes Distribution ({model_type})")
        ax.set_xlabel("Classification Category")
        ax.set_ylabel("Records Count")
        fig.tight_layout()
        mlflow.log_figure(fig, "P2_classification.png")
        plt.close(fig)

        if model_type == "rf":
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="mibici_p2_risk",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
        else:
            artifact_path = Path("p2_model.parquet")
            classes.to_parquet(artifact_path, index=False)
            mlflow.log_artifact(str(artifact_path))
            artifact_path.unlink()

    logger.info(
        f"P2 ({model_type}) complete: precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}"
    )


def train_p3(
    df: pd.DataFrame,
    year: int = MODELING_SETTINGS.DEFAULT_YEAR,
    n_clusters: int = MODELING_SETTINGS.KMEANS_CLUSTERS,
) -> None:
    """Train the P3 station clustering model with scaled KMeans."""
    logger.info(f"Training P3 KMeans ({n_clusters} clusters)")

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
        n_clusters=n_clusters,
        random_state=MODELING_SETTINGS.SEED,
        n_init=10,
    )
    stations["cluster"] = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, stations["cluster"])

    with mlflow.start_run(run_name=f"P3_KMeans_{n_clusters}_Clusters"):
        log_common_info("P3_clustering", year)
        mlflow.log_params(
            {
                "model": "kmeans",
                "n_clusters": n_clusters,
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
        ax.set_title(f"P3 - MiBici Station Clusters (k={n_clusters})")
        fig.colorbar(scatter, ax=ax, label="Cluster")
        fig.tight_layout()
        mlflow.log_figure(fig, "P3_clustering.png")
        plt.close(fig)

        mlflow.sklearn.log_model(kmeans, "model", registered_model_name="mibici_p3_clusters")

        pipeline = {"scaler": scaler, "kmeans": kmeans}
        artifact_path = Path("p3_pipeline.pkl")
        with open(artifact_path, "wb") as f:
            pickle.dump(pipeline, f)
        mlflow.log_artifact(str(artifact_path))
        artifact_path.unlink()

    logger.info(f"P3 complete: silhouette={silhouette:.3f}")


def run_training_pipeline(year: int = MODELING_SETTINGS.DEFAULT_YEAR) -> None:
    """Train P1, P2, and P3 baselines on the gold demand of one year."""
    logger.info(f"Training baselines for {year}")
    df_gold = load_gold_demand(year)
    train_p1(df_gold, year, model_type="baseline")
    train_p2(df_gold, year, model_type="baseline")
    train_p3(df_gold, year, n_clusters=MODELING_SETTINGS.KMEANS_CLUSTERS)
    logger.info(f"Baseline training complete for {year}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiBici models via MLflow Project.")
    parser.add_argument("--question", type=str, default="all", choices=["p1", "p2", "p3", "all"])
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--year", type=int, default=MODELING_SETTINGS.DEFAULT_YEAR)
    parser.add_argument("--n_estimators", type=int, default=MODELING_SETTINGS.DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max_depth", type=int, default=MODELING_SETTINGS.DEFAULT_MAX_DEPTH)
    parser.add_argument("--n_clusters", type=int, default=MODELING_SETTINGS.KMEANS_CLUSTERS)
    args = parser.parse_args()

    df_gold = load_gold_demand(args.year)

    if args.question == "p1":
        train_p1(
            df_gold,
            args.year,
            model_type=args.model,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
    elif args.question == "p2":
        train_p2(
            df_gold,
            args.year,
            model_type=args.model,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
    elif args.question == "p3":
        train_p3(df_gold, args.year, n_clusters=args.n_clusters)
    else:
        run_training_pipeline(args.year)


if __name__ == "__main__":
    main()
