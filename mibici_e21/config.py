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
    - `TEMP_DIR`: local directory for temporary downloads.
    - `RAW_DIR`: local directory for raw files.
    - `HF_TOKEN`: Hugging Face token, read from the `HF_TOKEN` environment variable.
    - `PROCESSED_DIR`: data lake output for the yearly Parquet files. Points at `data/raw`.
    """

    MIBICI_BASE_URL: str = "https://www.mibici.net/site/assets/files/"
    HF_DATASET_REPO: str = "yako2792/mibici-dataset"
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


# Modeling config
@dataclass(frozen=True)
class ModelingSettings:
    """Settings for baseline training.

    - `SEED`: random seed for sampling and KMeans.
    - `DEFAULT_YEAR`: year used for baseline training and evaluation.
    - `TRACKING_URI`: MLflow tracking URI.
    - `EXPERIMENT_NAME`: MLflow experiment that holds the baseline runs.
    - `TEST_SPLIT_DATE`: first date of the test set, as `YYYY-MM-DD`.
    - `VAL_SPLIT_DATE`: first date of the validation set, as `YYYY-MM-DD`. Validation runs
      from this date up to `TEST_SPLIT_DATE`.
    - `BALANCE_THRESHOLD`: absolute daily balance that marks a station as
      `possible_empty` or `possible_full`.
    - `PICO_HOURS`: hours counted as peak departures.
    - `KMEANS_CLUSTERS`: number of station clusters.
    - `DEFAULT_N_ESTIMATORS`: number of trees in the Random Forest.
    - `DEFAULT_MAX_DEPTH`: maximum depth of the Random Forest trees.
    - `SEARCH_N_ESTIMATORS`, `SEARCH_MAX_DEPTH`: values tried by the P1 and P2 searches.
    - `SEARCH_N_CLUSTERS`: values of `k` tried by the P3 search.
    - `PROCESSED_DIR`: directory for gold demand Parquet files. Points at
      `data/processed`.
    """

    SEED: int = 42
    DEFAULT_YEAR: int = 2023
    TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    EXPERIMENT_NAME: str = "MiBici_Baselines_Equipo21"
    TEST_SPLIT_DATE: str = "2023-10-01"
    VAL_SPLIT_DATE: str = "2023-08-01"  # NUEVO
    BALANCE_THRESHOLD: int = 5
    PICO_HOURS: tuple[int, ...] = (7, 8, 9, 17, 18, 19)
    KMEANS_CLUSTERS: int = 3
    DEFAULT_N_ESTIMATORS: int = 100
    DEFAULT_MAX_DEPTH: int = 10
    SEARCH_N_ESTIMATORS: tuple[int, ...] = (50, 100)  # NUEVO
    SEARCH_MAX_DEPTH: tuple[int, ...] = (6, 12)  # NUEVO
    SEARCH_N_CLUSTERS: tuple[int, ...] = (2, 3, 4, 5, 6)  # NUEVO
    PROCESSED_DIR: Path = PROCESSED_DATA_DIR


MODELING_SETTINGS = ModelingSettings()


# Predict config
@dataclass(frozen=True)
class PredictSettings:
    """Settings for baseline inference output.

    - `P1_OUTPUT_DIR`: directory for P1 prediction files. Points at `data/processed`.
    - `P2_OUTPUT_DIR`: directory for P2 prediction files. Points at `data/processed`.
    - `P3_OUTPUT_DIR`: directory for P3 prediction files. Points at `data/processed`.
    """

    P1_OUTPUT_DIR: Path = PROCESSED_DATA_DIR
    P2_OUTPUT_DIR: Path = PROCESSED_DATA_DIR
    P3_OUTPUT_DIR: Path = PROCESSED_DATA_DIR


PREDICT_SETTINGS = PredictSettings()
