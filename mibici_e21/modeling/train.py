"""Train the MiBici baselines and candidate models and log each run to MLflow.

Every run is launched through the MLflow Project (`mlflow run . -e train ...`).
Importing this module points the tracking URI at `MODELING_SETTINGS.TRACKING_URI`
and selects `MODELING_SETTINGS.EXPERIMENT_NAME`.

Data protocol, shared by P1 and P2 so every run is comparable with its baseline:

- Validation: fit on dates before `VAL_SPLIT_DATE`, score on the dates up to
  `TEST_SPLIT_DATE`. Metrics are logged with the `val_` prefix.
- Test: refit on every date before `TEST_SPLIT_DATE`, score on the dates from
  `TEST_SPLIT_DATE`. Metrics are logged with the `test_` prefix.

P3 clusters stations on the whole year, as the baseline does, so it has no
temporal split and logs unsupervised metrics instead.

Every run carries the tags `question`, `model_family`, and `stage_of_work`
(`baseline`, `search`, or `candidate`).
"""

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
import itertools
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
import time

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici_e21.config import MODELING_SETTINGS, RAW_DATA_DIR
from mibici_e21.schemas.columns import GoldColumns

os.environ.pop("MLFLOW_RUN_ID", None)
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
mlflow.set_tracking_uri(MODELING_SETTINGS.TRACKING_URI)
mlflow.set_experiment(MODELING_SETTINGS.EXPERIMENT_NAME)

SEED = MODELING_SETTINGS.SEED

P1_FEATURES = [
    GoldColumns.STATION_ID,
    GoldColumns.HOUR,
    GoldColumns.DAY_OF_WEEK,
    GoldColumns.IS_WEEKEND,
]
P2_LAGS = (1, 7)
P2_FEATURES = (
    [GoldColumns.STATION_ID, "dayofweek"]
    + [f"balance_lag{k}" for k in P2_LAGS]
    + [f"departures_lag{k}" for k in P2_LAGS]
)
P3_FEATURES = ["departures", "arrivals", "peak_departures"]
RISK_CLASSES = ["possible_empty", "balanced", "possible_full"]
PROBLEM_TAGS = {"p1": "P1_regression", "p2": "P2_classification", "p3": "P3_clustering"}
MODEL_FAMILIES = {
    "baseline": "baseline",
    "rf": "random_forest",
    "hgb": "hist_gradient_boosting",
    "kmeans": "kmeans",
}


@dataclass
class Fitted:
    """A fitted model, whatever its kind.

    - `predict`: function that maps a frame to predictions.
    - `model`: scikit-learn estimator of a candidate. `None` for a baseline.
    - `table`: lookup table of a baseline. `None` for a candidate.
    - `proba`: function that maps a frame to class probabilities, when the model has them.
    - `classes`: class names in the column order of `proba`.
    """

    predict: Callable[[pd.DataFrame], np.ndarray]
    model: object | None = None
    table: pd.DataFrame | None = None
    proba: Callable[[pd.DataFrame], np.ndarray] | None = None
    classes: list[str] | None = None


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


def get_dataset_version() -> str:
    """Return the dataset version written by `ingest` in `data/raw/data-version.txt`.

    Falls back to `v1.0.0` with a warning when the file is missing.
    """
    version_file = RAW_DATA_DIR / "data-version.txt"
    if version_file.exists():
        return version_file.read_text().strip()
    logger.warning(f"Dataset version file not found: {version_file}")
    return "v1.0.0"


def load_gold_demand(year: int = MODELING_SETTINGS.DEFAULT_YEAR) -> pd.DataFrame:
    """Load `demand_aggregated_{year}.parquet` from the gold feature store.

    - `year`: demand year used in the file name. Defaults to `DEFAULT_YEAR`.

    `date` is parsed to datetime. Raises `FileNotFoundError` when that file is not in
    `MODELING_SETTINGS.PROCESSED_DIR` (`data/processed`).
    """
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


def split_by_dates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a frame by date into the validation and the test protocols.

    - `df`: frame with the `GoldColumns.DATE` column as datetime.

    Returns `(train, val, train_full, test)`:

    - `train`: dates before `VAL_SPLIT_DATE`.
    - `val`: dates from `VAL_SPLIT_DATE` up to `TEST_SPLIT_DATE`.
    - `train_full`: dates before `TEST_SPLIT_DATE`, that is `train` plus `val`.
    - `test`: dates from `TEST_SPLIT_DATE`.
    """
    dates = df[GoldColumns.DATE]
    val_start = pd.Timestamp(MODELING_SETTINGS.VAL_SPLIT_DATE)
    test_start = pd.Timestamp(MODELING_SETTINGS.TEST_SPLIT_DATE)
    train = df[dates < val_start]
    val = df[(dates >= val_start) & (dates < test_start)]
    train_full = df[dates < test_start]
    test = df[dates >= test_start]
    return train, val, train_full, test


def set_run_tags(
    question: str,
    model_family: str,
    stage_of_work: str,
    year: int,
    extra: dict[str, str] | None = None,
) -> None:
    """Set the tags every run carries on the active run.

    - `question`: `p1`, `p2`, or `p3`.
    - `model_family`: family of the model, such as `random_forest`.
    - `stage_of_work`: `baseline`, `search`, or `candidate`.
    - `year`: demand year recorded in the `dataset` tag.
    - `extra`: more tags, such as `search_run_id`.
    """
    tags = {
        "team": "21",
        "problem": PROBLEM_TAGS[question],
        "question": question,
        "model_family": model_family,
        "stage_of_work": stage_of_work,
        "dataset": f"MiBici {year} Gold",
        "git_commit": get_git_commit(),
    }
    tags.update(extra or {})
    mlflow.set_tags(tags)


def log_common_params(year: int) -> None:
    """Log the params shared by every run: period, dataset version, seed, and split dates."""
    mlflow.log_params(
        {
            "period": str(year),
            "dataset_version": get_dataset_version(),
            "seed": SEED,
            "val_split_date": MODELING_SETTINGS.VAL_SPLIT_DATE,
            "test_split_date": MODELING_SETTINGS.TEST_SPLIT_DATE,
        }
    )


def log_support_artifacts(year: int, params: dict, summary: dict) -> None:
    """Log the data manifest and the run configuration as artifacts.

    - `year`: demand year of the gold file.
    - `params`: model params of the run.
    - `summary`: row counts of the splits.

    Writes `data/data-version.txt`, `data/manifest.json`, and `config/run_config.json`.
    Paths are left out of the files.
    """
    version_file = RAW_DATA_DIR / "data-version.txt"
    if version_file.exists():
        mlflow.log_artifact(str(version_file), "data")

    manifest = {
        "dataset_version": get_dataset_version(),
        "year": year,
        "gold_file": f"demand_aggregated_{year}.parquet",
        **summary,
    }
    mlflow.log_dict(manifest, "data/manifest.json")

    settings = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in asdict(MODELING_SETTINGS).items()
        if not isinstance(value, Path)
    }
    clean_params = {key: value for key, value in params.items() if value is not None}
    run_config = {"modeling_settings": settings, "run_params": clean_params}
    mlflow.log_dict(run_config, "config/run_config.json")


def log_training_dataset(frame: pd.DataFrame, name: str, year: int) -> None:
    """Log the training frame as a run input, so MLflow shows the dataset of the run.

    - `frame`: training data.
    - `name`: name of the dataset in MLflow.
    - `year`: demand year used to build the source path.

    A failure only logs a warning, because this input is optional.
    """
    source = str(MODELING_SETTINGS.PROCESSED_DIR / f"demand_aggregated_{year}.parquet")
    try:
        dataset = mlflow.data.from_pandas(frame, source=source, name=name)
        mlflow.log_input(dataset, context="training")
    except Exception as error:
        logger.warning(f"Dataset input not logged: {error}")


def log_candidate_model(model: object, example: pd.DataFrame) -> None:
    """Log a scikit-learn model with its signature and an input example.

    - `model`: fitted estimator or pipeline with a `predict` method.
    - `example`: frame with the model input columns. Its first rows become the input example.

    The model URI goes into the `model_uri` tag, which `register.py` reads.
    """
    example = example.head(5)
    signature = infer_signature(example, model.predict(example))
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=example,
        serialization_format="cloudpickle",
    )
    mlflow.set_tag("model_uri", info.model_uri)


def log_table_artifact(table: pd.DataFrame, file_name: str) -> None:
    """Log a lookup table of a baseline as a Parquet artifact.

    - `table`: lookup table.
    - `file_name`: name of the artifact. `predict.py` reads `p1_model.parquet` and
      `p2_model.parquet`.
    """
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / file_name
        table.to_parquet(path, index=False)
        mlflow.log_artifact(str(path))


def run_search(
    question: str,
    model_family: str,
    year: int,
    grid: list[dict],
    evaluate: Callable[[dict], dict],
    primary_metric: str,
    minimize: bool,
) -> tuple[dict, str]:
    """Run a hyperparameter search as one parent run with one nested run per combination.

    - `question`: `p1`, `p2`, or `p3`.
    - `model_family`: family of the searched model.
    - `year`: demand year recorded on the runs.
    - `grid`: list of param dicts, one per combination.
    - `evaluate`: function that maps a param dict to a metrics dict.
    - `primary_metric`: metric that picks the best combination.
    - `minimize`: `True` when a lower `primary_metric` is better.

    Child runs log params, metrics, and tags only. The best params and the best value go on
    the parent run. Returns `(best_params, parent_run_id)`.
    """
    with mlflow.start_run(run_name=f"{question.upper()}_{model_family}_search") as parent:
        log_common_params(year)
        set_run_tags(question, model_family, "search", year)
        mlflow.log_param("n_combinations", len(grid))

        best_params, best_value = None, None
        for params in grid:
            name = "_".join(f"{key}{value}" for key, value in params.items())
            with mlflow.start_run(run_name=name, nested=True):
                set_run_tags(question, model_family, "search", year)
                mlflow.log_params(params)
                metrics = evaluate(params)
                mlflow.log_metrics(metrics)

            value = metrics[primary_metric]
            better = best_value is None or (value < best_value if minimize else value > best_value)
            if better:
                best_params, best_value = params, value

        mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})
        mlflow.log_metric(f"best_{primary_metric}", best_value)
        logger.info(
            f"{question.upper()} search best: {best_params} ({primary_metric}={best_value:.4f})"
        )
        return best_params, parent.info.run_id


# ---------------------------------------------------------------------------
# P1: demand regression
# ---------------------------------------------------------------------------


def build_p1_model(model_type: str, n_estimators: int, max_depth: int) -> object:
    """Build the scikit-learn regressor of a P1 candidate.

    - `model_type`: `rf` for `RandomForestRegressor`, `hgb` for `HistGradientBoostingRegressor`.
    - `n_estimators`: trees of `rf`, or `max_iter` of `hgb`.
    - `max_depth`: maximum depth of the trees.
    """
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=SEED, n_jobs=-1
        )
    if model_type == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=n_estimators, max_depth=max_depth, random_state=SEED
        )
    raise ValueError(f"Unknown P1 model '{model_type}'. Use baseline, rf, or hgb.")


def fit_p1(model_type: str, train: pd.DataFrame, n_estimators: int, max_depth: int) -> Fitted:
    """Fit a P1 model on the training frame.

    - `model_type`: `baseline` (historical mean by station and hour), `rf`, or `hgb`.
    - `train`: gold demand, one row per station, date, and hour.
    - `n_estimators`: trees of `rf`, or `max_iter` of `hgb`.
    - `max_depth`: maximum depth of the trees.

    The baseline predicts the training mean of the station and hour, and the global
    training mean for an unseen pair.
    """
    target = GoldColumns.DEPARTURES_COUNT
    if model_type == "baseline":
        keys = [GoldColumns.STATION_ID, GoldColumns.HOUR]
        historical = train.groupby(keys)[target].mean().reset_index(name="prediction")
        global_mean = float(train[target].mean())

        def predict(frame: pd.DataFrame) -> np.ndarray:
            merged = frame[keys].merge(historical, on=keys, how="left")
            return merged["prediction"].fillna(global_mean).to_numpy()

        return Fitted(predict=predict, table=historical)

    model = build_p1_model(model_type, n_estimators, max_depth)
    model.fit(train[P1_FEATURES], train[target])
    return Fitted(predict=lambda frame: model.predict(frame[P1_FEATURES]), model=model)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray, prefix: str) -> dict:
    """Return MAE, RMSE, and R2 with a `val` or `test` prefix."""
    return {
        f"{prefix}_MAE": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_R2": float(r2_score(y_true, y_pred)),
    }


def log_p1_plots(y_true: pd.Series, y_pred: np.ndarray, label: str, year: int) -> None:
    """Log the P1 figures under `plots/`: actual against predicted, and residuals.

    - `y_true`: actual departures of the test set.
    - `y_pred`: predicted departures.
    - `label`: model name shown in the titles.
    - `year`: demand year shown in the titles.
    """
    frame = pd.DataFrame({"actual": y_true.to_numpy(), "predicted": y_pred})
    sample = frame.sample(n=min(2000, len(frame)), random_state=SEED)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sample["actual"], sample["predicted"], alpha=0.4, s=10)
    top = float(max(sample["actual"].max(), sample["predicted"].max()))
    ax.plot([0, top], [0, top], color="gray", linestyle="--")
    ax.set_xlabel("Actual trips")
    ax.set_ylabel("Predicted trips")
    ax.set_title(f"P1 - Actual vs predicted ({label}, {year})")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/P1_actual_vs_predicted.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(frame["actual"] - frame["predicted"], bins=50)
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Rows")
    ax.set_title(f"P1 - Residuals ({label}, {year})")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/P1_residuals.png")
    plt.close(fig)


def train_p1(
    df: pd.DataFrame,
    year: int = MODELING_SETTINGS.DEFAULT_YEAR,
    model_type: str = "baseline",
    n_estimators: int = MODELING_SETTINGS.DEFAULT_N_ESTIMATORS,
    max_depth: int = MODELING_SETTINGS.DEFAULT_MAX_DEPTH,
    search_run_id: str | None = None,
) -> str:
    """Train P1 under the validation and test protocols and log one run.

    - `df`: gold demand, one row per station, date, and hour.
    - `year`: demand year recorded on the run.
    - `model_type`: `baseline`, `rf`, or `hgb`.
    - `n_estimators`: trees of `rf`, or `max_iter` of `hgb`.
    - `max_depth`: maximum depth of the trees.
    - `search_run_id`: parent run of the search that chose these params, if any.

    The run logs `val_` and `test_` MAE, RMSE, and R2, plus `train_seconds`. A baseline
    logs `p1_model.parquet`, as `predict.py` expects. A candidate logs the model with its
    signature and input example. Returns the run id.
    """
    logger.info(f"Training P1 ({model_type})")
    train, val, train_full, test = split_by_dates(df)
    target = GoldColumns.DEPARTURES_COUNT

    val_fit = fit_p1(model_type, train, n_estimators, max_depth)
    val_metrics = regression_metrics(val[target], val_fit.predict(val), "val")

    started = time.perf_counter()
    final = fit_p1(model_type, train_full, n_estimators, max_depth)
    train_seconds = time.perf_counter() - started
    test_pred = final.predict(test)
    test_metrics = regression_metrics(test[target], test_pred, "test")

    if model_type == "baseline":
        run_name, stage = "P1_Historical_Mean", "baseline"
        params = {"model": model_type}
    else:
        run_name = f"P1_{model_type.upper()}_est{n_estimators}_depth{max_depth}"
        stage = "candidate"
        params = {"model": model_type, "n_estimators": n_estimators, "max_depth": max_depth}
    extra = {"search_run_id": search_run_id} if search_run_id else None
    summary = {"train_rows": len(train), "val_rows": len(val), "test_rows": len(test)}

    with mlflow.start_run(run_name=run_name) as run:
        log_common_params(year)
        mlflow.log_params(params)
        set_run_tags("p1", MODEL_FAMILIES[model_type], stage, year, extra)
        mlflow.log_metrics({**val_metrics, **test_metrics, "train_seconds": train_seconds})
        log_support_artifacts(year, params, summary)
        log_training_dataset(train_full, f"P1 train {year}", year)
        log_p1_plots(test[target], test_pred, model_type, year)
        if final.model is not None:
            log_candidate_model(final.model, train_full[P1_FEATURES])
        else:
            log_table_artifact(final.table, "p1_model.parquet")

    logger.info(
        f"P1 ({model_type}) complete: val_MAE={val_metrics['val_MAE']:.3f}, "
        f"test_MAE={test_metrics['test_MAE']:.3f}, test_R2={test_metrics['test_R2']:.3f}"
    )
    return run.info.run_id


def search_p1(df: pd.DataFrame, year: int, model_type: str) -> tuple[dict, str]:
    """Search `n_estimators` and `max_depth` for P1 with nested runs.

    - `df`: gold demand.
    - `year`: demand year recorded on the runs.
    - `model_type`: `rf` or `hgb`.

    Each combination is fitted on the training dates and scored on validation. The best one
    has the lowest `val_MAE`. Returns `(best_params, parent_run_id)`.
    """
    train, val, _, _ = split_by_dates(df)
    target = GoldColumns.DEPARTURES_COUNT
    grid = [
        {"n_estimators": n, "max_depth": d}
        for n, d in itertools.product(
            MODELING_SETTINGS.SEARCH_N_ESTIMATORS, MODELING_SETTINGS.SEARCH_MAX_DEPTH
        )
    ]

    def evaluate(params: dict) -> dict:
        started = time.perf_counter()
        fitted = fit_p1(model_type, train, params["n_estimators"], params["max_depth"])
        seconds = time.perf_counter() - started
        metrics = regression_metrics(val[target], fitted.predict(val), "val")
        return {**metrics, "train_seconds": seconds}

    return run_search("p1", MODEL_FAMILIES[model_type], year, grid, evaluate, "val_MAE", True)


# ---------------------------------------------------------------------------
# P2: station risk classification
# ---------------------------------------------------------------------------


def build_p2_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Build the station-day frame of P2 with its label and lag features.

    - `df`: gold demand, one row per station, date, and hour.

    Departures and arrivals are summed by station and date. `balance` is arrivals minus
    departures. `risk` is `possible_empty` at or below `-BALANCE_THRESHOLD`, `possible_full`
    at or above `BALANCE_THRESHOLD`, and `balanced` otherwise.

    Features only use earlier days: `balance_lag{k}` and `departures_lag{k}` hold the value
    of the same station `k` days before, for `k` in `P2_LAGS`, and 0 when that station had
    no trips that day. The same-day departures are not a feature, because `risk` is built
    from them.
    """
    keys = [GoldColumns.STATION_ID, GoldColumns.DATE]
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

    for lag in P2_LAGS:
        lagged = daily[[*keys, "balance", GoldColumns.DEPARTURES_COUNT]].copy()
        lagged[GoldColumns.DATE] = lagged[GoldColumns.DATE] + pd.Timedelta(days=lag)
        lagged = lagged.rename(
            columns={
                "balance": f"balance_lag{lag}",
                GoldColumns.DEPARTURES_COUNT: f"departures_lag{lag}",
            }
        )
        daily = daily.merge(lagged, on=keys, how="left")

    lag_columns = [column for column in P2_FEATURES if "_lag" in column]
    daily[lag_columns] = daily[lag_columns].fillna(0)
    return daily


def build_p2_model(model_type: str, n_estimators: int, max_depth: int) -> object:
    """Build the scikit-learn classifier of a P2 candidate.

    - `model_type`: `rf` for `RandomForestClassifier`, `hgb` for
      `HistGradientBoostingClassifier`.
    - `n_estimators`: trees of `rf`, or `max_iter` of `hgb`.
    - `max_depth`: maximum depth of the trees.

    Both weight the classes as balanced, because `balanced` dominates.
    """
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=-1,
        )
    if model_type == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=n_estimators, max_depth=max_depth, random_state=SEED, class_weight="balanced"
        )
    raise ValueError(f"Unknown P2 model '{model_type}'. Use baseline, rf, or hgb.")


def fit_p2(model_type: str, train: pd.DataFrame, n_estimators: int, max_depth: int) -> Fitted:
    """Fit a P2 model on the station-day training frame.

    - `model_type`: `baseline` (historical mode by station), `rf`, or `hgb`.
    - `train`: output of `build_p2_daily`, restricted to the training dates.
    - `n_estimators`: trees of `rf`, or `max_iter` of `hgb`.
    - `max_depth`: maximum depth of the trees.

    The baseline predicts the most frequent class of the station, and the most frequent
    class overall for an unseen station.
    """
    if model_type == "baseline":
        station = GoldColumns.STATION_ID
        classes = train.groupby(station)["risk"].agg(lambda x: x.mode().iloc[0]).reset_index()
        general_class = train["risk"].mode().iloc[0]
        lookup = classes.rename(columns={"risk": "risk_pred"})

        def predict(frame: pd.DataFrame) -> np.ndarray:
            merged = frame[[station]].merge(lookup, on=station, how="left")
            return merged["risk_pred"].fillna(general_class).to_numpy()

        return Fitted(predict=predict, table=classes)

    model = build_p2_model(model_type, n_estimators, max_depth)
    model.fit(train[P2_FEATURES], train["risk"])
    return Fitted(
        predict=lambda frame: model.predict(frame[P2_FEATURES]),
        model=model,
        proba=lambda frame: model.predict_proba(frame[P2_FEATURES]),
        classes=[str(label) for label in model.classes_],
    )


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray, prefix: str) -> dict:
    """Return macro precision, recall, and F1, plus the recall of the two risk classes.

    The recall of `possible_empty` and `possible_full` is the share of real shortages and
    saturations the model catches. The keys carry a `val` or `test` prefix.
    """
    recalls = recall_score(y_true, y_pred, labels=RISK_CLASSES, average=None, zero_division=0)
    return {
        f"{prefix}_precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        f"{prefix}_recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        f"{prefix}_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}_recall_possible_empty": float(recalls[0]),
        f"{prefix}_recall_possible_full": float(recalls[2]),
    }


def log_p2_plots(test: pd.DataFrame, y_pred: np.ndarray, fitted: Fitted, label: str) -> None:
    """Log the P2 figures under `plots/`: confusion matrix and precision-recall curves.

    - `test`: station-day test frame with the `risk` label.
    - `y_pred`: predicted classes.
    - `fitted`: fitted model. The curves need `fitted.proba`, so a baseline gets none.
    - `label`: model name shown in the titles.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        test["risk"], y_pred, labels=RISK_CLASSES, ax=ax, xticks_rotation=20, colorbar=False
    )
    ax.set_title(f"P2 - Confusion matrix ({label})")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/P2_confusion_matrix.png")
    plt.close(fig)

    if fitted.proba is None:
        return
    proba = fitted.proba(test)
    fig, ax = plt.subplots(figsize=(6, 5))
    for index, name in enumerate(fitted.classes):
        precision, recall, _ = precision_recall_curve(
            (test["risk"] == name).astype(int), proba[:, index]
        )
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"P2 - Precision-recall by class ({label})")
    ax.legend()
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/P2_precision_recall.png")
    plt.close(fig)


def train_p2(
    df: pd.DataFrame,
    year: int = MODELING_SETTINGS.DEFAULT_YEAR,
    model_type: str = "baseline",
    n_estimators: int = MODELING_SETTINGS.DEFAULT_N_ESTIMATORS,
    max_depth: int = MODELING_SETTINGS.DEFAULT_MAX_DEPTH,
    search_run_id: str | None = None,
) -> str:
    """Train P2 under the validation and test protocols and log one run.

    - `df`: gold demand, one row per station, date, and hour.
    - `year`: demand year recorded on the run.
    - `model_type`: `baseline`, `rf`, or `hgb`.
    - `n_estimators`: trees of `rf`, or `max_iter` of `hgb`.
    - `max_depth`: maximum depth of the trees.
    - `search_run_id`: parent run of the search that chose these params, if any.

    The run logs `val_` and `test_` macro precision, recall, and F1, the recall of each risk
    class, and `train_seconds`. A baseline logs `p2_model.parquet`, as `predict.py` expects.
    A candidate logs the model with its signature and input example. Returns the run id.
    """
    logger.info(
        f"Training P2 ({model_type}) "
        f"(val {MODELING_SETTINGS.VAL_SPLIT_DATE}, test {MODELING_SETTINGS.TEST_SPLIT_DATE}, "
        f"threshold {MODELING_SETTINGS.BALANCE_THRESHOLD})"
    )
    daily = build_p2_daily(df)
    train, val, train_full, test = split_by_dates(daily)

    val_fit = fit_p2(model_type, train, n_estimators, max_depth)
    val_metrics = classification_metrics(val["risk"], val_fit.predict(val), "val")

    started = time.perf_counter()
    final = fit_p2(model_type, train_full, n_estimators, max_depth)
    train_seconds = time.perf_counter() - started
    test_pred = final.predict(test)
    test_metrics = classification_metrics(test["risk"], test_pred, "test")

    if model_type == "baseline":
        run_name, stage = "P2_Historical_Mode", "baseline"
        params = {"model": model_type, "balance_threshold": MODELING_SETTINGS.BALANCE_THRESHOLD}
    else:
        run_name = f"P2_{model_type.upper()}_est{n_estimators}_depth{max_depth}"
        stage = "candidate"
        params = {
            "model": model_type,
            "balance_threshold": MODELING_SETTINGS.BALANCE_THRESHOLD,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "features": ",".join(P2_FEATURES),
        }
    extra = {"search_run_id": search_run_id} if search_run_id else None
    summary = {"train_rows": len(train), "val_rows": len(val), "test_rows": len(test)}

    with mlflow.start_run(run_name=run_name) as run:
        log_common_params(year)
        mlflow.log_params(params)
        set_run_tags("p2", MODEL_FAMILIES[model_type], stage, year, extra)
        mlflow.log_metrics({**val_metrics, **test_metrics, "train_seconds": train_seconds})
        log_support_artifacts(year, params, summary)
        log_training_dataset(train_full, f"P2 train {year}", year)
        log_p2_plots(test, test_pred, final, model_type)
        if final.model is not None:
            log_candidate_model(final.model, train_full[P2_FEATURES])
        else:
            log_table_artifact(final.table, "p2_model.parquet")

    logger.info(
        f"P2 ({model_type}) complete: val_f1={val_metrics['val_f1_macro']:.3f}, "
        f"test_f1={test_metrics['test_f1_macro']:.3f}"
    )
    return run.info.run_id


def search_p2(df: pd.DataFrame, year: int, model_type: str) -> tuple[dict, str]:
    """Search `n_estimators` and `max_depth` for P2 with nested runs.

    - `df`: gold demand.
    - `year`: demand year recorded on the runs.
    - `model_type`: `rf` or `hgb`.

    Each combination is fitted on the training dates and scored on validation. The best one
    has the highest `val_f1_macro`. Returns `(best_params, parent_run_id)`.
    """
    train, val, _, _ = split_by_dates(build_p2_daily(df))
    grid = [
        {"n_estimators": n, "max_depth": d}
        for n, d in itertools.product(
            MODELING_SETTINGS.SEARCH_N_ESTIMATORS, MODELING_SETTINGS.SEARCH_MAX_DEPTH
        )
    ]

    def evaluate(params: dict) -> dict:
        started = time.perf_counter()
        fitted = fit_p2(model_type, train, params["n_estimators"], params["max_depth"])
        seconds = time.perf_counter() - started
        metrics = classification_metrics(val["risk"], fitted.predict(val), "val")
        return {**metrics, "train_seconds": seconds}

    family = MODEL_FAMILIES[model_type]
    return run_search("p2", family, year, grid, evaluate, "val_f1_macro", False)


# ---------------------------------------------------------------------------
# P3: station clustering
# ---------------------------------------------------------------------------


def build_p3_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Describe each station by its total departures, total arrivals, and peak departures.

    - `df`: gold demand, one row per station, date, and hour.

    Peak departures are the departures during `MODELING_SETTINGS.PICO_HOURS`. Returns one
    row per station with the columns of `P3_FEATURES`.
    """
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
    return stations[P3_FEATURES]


def fit_p3(stations: pd.DataFrame, n_clusters: int) -> Pipeline:
    """Fit a scaler and KMeans as one pipeline.

    - `stations`: output of `build_p3_stations`.
    - `n_clusters`: number of groups.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)),
        ]
    )
    return pipeline.fit(stations)


def clustering_metrics(pipeline: Pipeline, stations: pd.DataFrame) -> dict:
    """Return the quality and the stability of a station clustering.

    - `pipeline`: fitted pipeline from `fit_p3`.
    - `stations`: the frame the pipeline was fitted on.

    Metrics: `silhouette` (higher is better), `davies_bouldin` (lower is better),
    `calinski_harabasz` (higher is better), and `stability_ari`, the mean adjusted Rand index
    between the clustering and three KMeans fits with other seeds.
    """
    scaled = pipeline[:-1].transform(stations)
    labels = pipeline.predict(stations)
    k = pipeline.named_steps["kmeans"].n_clusters
    scores = [
        adjusted_rand_score(
            labels, KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(scaled)
        )
        for seed in range(SEED + 1, SEED + 4)
    ]
    return {
        "silhouette": float(silhouette_score(scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(scaled, labels)),
        "stability_ari": float(np.mean(scores)),
    }


def log_p3_plots(stations: pd.DataFrame, labels: np.ndarray, n_clusters: int) -> None:
    """Log the P3 figures under `plots/`: the stations by cluster and the cluster profiles.

    - `stations`: output of `build_p3_stations`.
    - `labels`: cluster of each station.
    - `n_clusters`: number of groups shown in the titles.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(stations["departures"], stations["arrivals"], c=labels, alpha=0.7)
    ax.set_xlabel("Total departures")
    ax.set_ylabel("Total arrivals")
    ax.set_title(f"P3 - MiBici station clusters (k={n_clusters})")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/P3_clusters.png")
    plt.close(fig)

    profile = stations.assign(cluster=labels).groupby("cluster")[P3_FEATURES].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    profile.plot.bar(ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean per station")
    ax.set_title(f"P3 - Cluster profiles (k={n_clusters})")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/P3_cluster_profiles.png")
    plt.close(fig)


def log_pipeline_pickle(pipeline: Pipeline) -> None:
    """Log `p3_pipeline.pkl` with the scaler and KMeans, which `predict.py` loads.

    - `pipeline`: fitted pipeline from `fit_p3`.

    Only baselines log it, so candidates keep every artifact in a subfolder.
    """
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "p3_pipeline.pkl"
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "scaler": pipeline.named_steps["scaler"],
                    "kmeans": pipeline.named_steps["kmeans"],
                },
                file,
            )
        mlflow.log_artifact(str(path))


def train_p3(
    df: pd.DataFrame,
    year: int = MODELING_SETTINGS.DEFAULT_YEAR,
    model_type: str = "baseline",
    n_clusters: int = MODELING_SETTINGS.KMEANS_CLUSTERS,
    search_run_id: str | None = None,
) -> str:
    """Train the P3 station clustering and log one run.

    - `df`: gold demand, one row per station, date, and hour.
    - `year`: demand year recorded on the run.
    - `model_type`: `baseline` uses `KMEANS_CLUSTERS` groups and ignores `n_clusters`.
      `kmeans` uses `n_clusters` and is logged as a candidate.
    - `n_clusters`: number of groups of a candidate.
    - `search_run_id`: parent run of the search that chose `n_clusters`, if any.

    Stations are described by `P3_FEATURES` over the whole year, scaled, and clustered with
    KMeans. The run logs the metrics of `clustering_metrics` and the pipeline as a model with
    its signature and input example. A baseline also logs `p3_pipeline.pkl`, as `predict.py`
    expects. Returns the run id.
    """
    if model_type not in ("baseline", "kmeans"):
        raise ValueError(f"Unknown P3 model '{model_type}'. Use baseline or kmeans.")
    if model_type == "baseline":
        n_clusters = MODELING_SETTINGS.KMEANS_CLUSTERS
        run_name, stage = f"P3_KMeans_{n_clusters}_Clusters", "baseline"
    else:
        run_name, stage = f"P3_KMeans_k{n_clusters}", "candidate"
    logger.info(f"Training P3 KMeans ({n_clusters} clusters, {stage})")

    stations = build_p3_stations(df)
    started = time.perf_counter()
    pipeline = fit_p3(stations, n_clusters)
    train_seconds = time.perf_counter() - started
    metrics = clustering_metrics(pipeline, stations)
    labels = pipeline.predict(stations)

    params = {"model": "kmeans", "n_clusters": n_clusters, "n_init": 10}
    extra = {"search_run_id": search_run_id} if search_run_id else None
    summary = {"stations": len(stations), "features": P3_FEATURES}

    with mlflow.start_run(run_name=run_name) as run:
        log_common_params(year)
        mlflow.log_params({**params, "features": ",".join(P3_FEATURES)})
        set_run_tags("p3", MODEL_FAMILIES["kmeans"], stage, year, extra)
        mlflow.log_metrics({**metrics, "train_seconds": train_seconds})
        log_support_artifacts(year, params, summary)
        log_training_dataset(stations.reset_index(), f"P3 stations {year}", year)
        log_p3_plots(stations, labels, n_clusters)
        log_candidate_model(pipeline, stations)
        if stage == "baseline":
            log_pipeline_pickle(pipeline)

    logger.info(f"P3 complete: silhouette={metrics['silhouette']:.3f}")
    return run.info.run_id


def search_p3(df: pd.DataFrame, year: int) -> tuple[dict, str]:
    """Search `n_clusters` for P3 with nested runs.

    - `df`: gold demand.
    - `year`: demand year recorded on the runs.

    Each value of `SEARCH_N_CLUSTERS` is fitted and scored with `clustering_metrics`. The
    best one has the highest `silhouette`. Returns `(best_params, parent_run_id)`.
    """
    stations = build_p3_stations(df)
    grid = [{"n_clusters": k} for k in MODELING_SETTINGS.SEARCH_N_CLUSTERS]

    def evaluate(params: dict) -> dict:
        started = time.perf_counter()
        pipeline = fit_p3(stations, params["n_clusters"])
        seconds = time.perf_counter() - started
        return {**clustering_metrics(pipeline, stations), "train_seconds": seconds}

    return run_search("p3", MODEL_FAMILIES["kmeans"], year, grid, evaluate, "silhouette", False)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_training_pipeline(year: int = MODELING_SETTINGS.DEFAULT_YEAR) -> None:
    """Train the P1, P2, and P3 baselines on the gold demand of one year."""
    logger.info(f"Training baselines for {year}")
    df_gold = load_gold_demand(year)
    train_p1(df_gold, year, model_type="baseline")
    train_p2(df_gold, year, model_type="baseline")
    train_p3(df_gold, year, model_type="baseline")
    logger.info(f"Baseline training complete for {year}")


def parse_bool(value: str) -> bool:
    """Read `true` or `false` from the command line, as `mlflow run` passes strings."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got '{value}'.")


def main() -> None:
    """Train from the command line, which is how the MLflow Project calls it.

    - `--question`: `p1`, `p2`, `p3`, or `all` (the three baselines).
    - `--model`: `baseline`, `rf`, or `hgb` for P1 and P2. `baseline` or `kmeans` for P3.
    - `--year`: demand year. Defaults to `DEFAULT_YEAR`.
    - `--n_estimators`, `--max_depth`: params of `rf` and `hgb`.
    - `--n_clusters`: groups of a P3 `kmeans` candidate.
    - `--search`: `true` runs a search with nested runs for that model, then trains the best
      params as a candidate linked to the search through the `search_run_id` tag.
    """
    parser = argparse.ArgumentParser(description="Train MiBici models via MLflow Project.")
    parser.add_argument("--question", type=str, default="all", choices=["p1", "p2", "p3", "all"])
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--year", type=int, default=MODELING_SETTINGS.DEFAULT_YEAR)
    parser.add_argument("--n_estimators", type=int, default=MODELING_SETTINGS.DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max_depth", type=int, default=MODELING_SETTINGS.DEFAULT_MAX_DEPTH)
    parser.add_argument("--n_clusters", type=int, default=MODELING_SETTINGS.KMEANS_CLUSTERS)
    parser.add_argument("--search", type=parse_bool, default=False)
    args = parser.parse_args()

    if args.question == "all":
        run_training_pipeline(args.year)
        return

    df_gold = load_gold_demand(args.year)
    trainers = {"p1": (search_p1, train_p1), "p2": (search_p2, train_p2)}

    if args.question in trainers:
        search, train = trainers[args.question]
        n_estimators, max_depth, search_run_id = args.n_estimators, args.max_depth, None
        if args.search:
            if args.model == "baseline":
                raise ValueError("A search needs a model: use --model rf or --model hgb.")
            best, search_run_id = search(df_gold, args.year, args.model)
            n_estimators, max_depth = best["n_estimators"], best["max_depth"]
        train(df_gold, args.year, args.model, n_estimators, max_depth, search_run_id)
        return

    n_clusters, search_run_id = args.n_clusters, None
    model = args.model
    if args.search:
        best, search_run_id = search_p3(df_gold, args.year)
        n_clusters, model = best["n_clusters"], "kmeans"
    train_p3(df_gold, args.year, model, n_clusters, search_run_id)


if __name__ == "__main__":
    main()