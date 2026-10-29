"""Paths and settings for the MiBici data lake, features, and models.

`load_dotenv` runs on import so `HF_TOKEN` is available from a `.env` file.
"""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Project paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
TEMP_DATA_DIR = DATA_DIR / "temp"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# Data lake & scraper config
@dataclass(frozen=True)
class DataLakeSettings:
    """Settings for the MiBici data lake and scraper.

    - `MIBICI_BASE_URL`: base URL of the public MiBici file directory.
    - `HF_DATASET_REPO`: Hugging Face dataset repository, as `owner/name`.
    - `HF_DATASET_REVISION`: dataset commit the ingestion downloads, so every clone trains on
      the same files. Update it on purpose to train on a newer snapshot.
    - `TEMP_DIR`: local directory for temporary downloads.
    - `RAW_DIR`: local directory for raw files.
    - `HF_TOKEN`: Hugging Face token, read from the `HF_TOKEN` environment variable.
    - `PROCESSED_DIR`: data lake output for the yearly Parquet files. Points at `data/raw`.
    """

    MIBICI_BASE_URL: str = "https://www.mibici.net/site/assets/files/"
    HF_DATASET_REPO: str = "yako2792/mibici-dataset"
    HF_DATASET_REVISION: str = "11e35bd665983aad17d70b921ccf5c5218294c9d"
    TEMP_DIR: Path = TEMP_DATA_DIR
    RAW_DIR: Path = RAW_DATA_DIR
    HF_TOKEN: str | None = os.getenv("HF_TOKEN")
    PROCESSED_DIR: Path = RAW_DATA_DIR


DATA_LAKE_SETTINGS = DataLakeSettings()


# Feature store config
@dataclass(frozen=True)
class FeatureStoreSettings:
    """Settings for the feature store.

    - `MAX_TRIP_DURATION_MINUTES`: maximum trip duration in minutes. Defaults to 8 hours.
    - `MIN_TRIP_DURATION_MINUTES`: minimum trip duration in minutes. Defaults to 0 minutes.
    - `PROCESSED_DIR`: directory for gold feature Parquet files. Points at `data/processed`.
    """

    MAX_TRIP_DURATION_MINUTES: float = 480.0
    MIN_TRIP_DURATION_MINUTES: float = 0.0
    PROCESSED_DIR: Path = PROCESSED_DATA_DIR


FEATURE_STORE_SETTINGS = FeatureStoreSettings()


# Tracking config
@dataclass(frozen=True)
class TrackingSettings:
    """Settings for the MLflow tracking server.

    - `TRACKING_URI`: MLflow tracking server, read from `MLFLOW_TRACKING_URI`. An empty
      variable counts as unset, so it never falls back to a local file store.
    - `EXPERIMENT_NAME`: experiment that holds every ingestion, baseline, search, and
      candidate run. `mlflow run` reads it from `MLFLOW_EXPERIMENT_NAME`, so both must match.
    - `TEAM`: team number recorded in the `team` tag.
    """

    TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI") or "http://127.0.0.1:5000"
    EXPERIMENT_NAME: str = "mibici-equipo21"
    TEAM: str = "21"


TRACKING_SETTINGS = TrackingSettings()


# Modeling config
@dataclass(frozen=True)
class ModelingSettings:
    """Settings shared by every baseline, search, and candidate run.

    - `SEED`: random seed of every estimator and sample.
    - `TRAINING_YEAR`: year that `ingest` downloads and every model trains on.
    - `VAL_START_MONTH`: first month of the validation set. Earlier months are training.
    - `TEST_START_MONTH`: first month of the test set. Validation ends the month before.
    - `BALANCE_THRESHOLD`: absolute daily balance that marks a station as
      `possible_empty` or `possible_full`.
    - `PEAK_HOURS`: hours counted as peak departures.
    - `RISK_LAGS`: days back used for the P2 lag features.
    - `BASELINE_CLUSTERS`: number of station groups of the P3 baseline.
    - `KMEANS_N_INIT`: KMeans restarts per fit.
    - `SEARCH_N_ESTIMATORS`, `SEARCH_MAX_DEPTH`: grid of the P1 and P2 searches.
    - `SEARCH_N_CLUSTERS`: values of `k` tried by the P3 search.
    - `PLOT_SAMPLE_SIZE`: points drawn in scatter plots.
    - `PROCESSED_DIR`: directory of the gold demand Parquet files (`data/processed`).
    """

    SEED: int = 42
    TRAINING_YEAR: int = 2023
    VAL_START_MONTH: int = 8
    TEST_START_MONTH: int = 10
    BALANCE_THRESHOLD: int = 5
    PEAK_HOURS: tuple[int, ...] = (7, 8, 9, 17, 18, 19)
    RISK_LAGS: tuple[int, ...] = (1, 7)
    BASELINE_CLUSTERS: int = 3
    KMEANS_N_INIT: int = 10
    SEARCH_N_ESTIMATORS: tuple[int, ...] = (100, 200, 400)
    SEARCH_MAX_DEPTH: tuple[int, ...] = (8, 14, 20)
    SEARCH_N_CLUSTERS: tuple[int, ...] = (2, 3, 4, 5, 6)
    PLOT_SAMPLE_SIZE: int = 2000
    PROCESSED_DIR: Path = PROCESSED_DATA_DIR


MODELING_SETTINGS = ModelingSettings()


# Predict config
@dataclass(frozen=True)
class PredictSettings:
    """Settings for batch inference.

    - `OUTPUT_DIR`: directory for the prediction files. Points at `data/processed`.
    """

    OUTPUT_DIR: Path = PROCESSED_DATA_DIR


PREDICT_SETTINGS = PredictSettings()
