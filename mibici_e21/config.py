from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Project paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

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
