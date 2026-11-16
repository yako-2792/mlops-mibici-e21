"""Paths and settings for the MiBici data lake, features, tracking, and models.

`load_dotenv` runs on import, so `HF_TOKEN` and the MLflow variables can live in a `.env`
file. `huggingface_hub` reads `HF_TOKEN` from the environment when it uploads.
"""

from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
    - `TEMP_DIR`: bronze layer, the downloaded CSVs (`data/temp`).
    - `SILVER_DIR`: silver layer, the yearly trip and station Parquet files (`data/raw`).
    """

    MIBICI_BASE_URL: str = "https://www.mibici.net/site/assets/files/"
    HF_DATASET_REPO: str = "yako2792/mibici-dataset"
    HF_DATASET_REVISION: str = "11e35bd665983aad17d70b921ccf5c5218294c9d"
    TEMP_DIR: Path = TEMP_DATA_DIR
    SILVER_DIR: Path = RAW_DATA_DIR


DATA_LAKE_SETTINGS = DataLakeSettings()


# Feature store config
@dataclass(frozen=True)
class FeatureStoreSettings:
    """Settings for the feature store.

    - `MAX_TRIP_DURATION_MINUTES`: maximum trip duration in minutes. Defaults to 8 hours.
    - `MIN_TRIP_DURATION_MINUTES`: minimum trip duration in minutes. Defaults to 0 minutes.
    - `GOLD_DIR`: gold layer, the trip features, station demand, and data manifests
      (`data/processed`).
    """

    MAX_TRIP_DURATION_MINUTES: float = 480.0
    MIN_TRIP_DURATION_MINUTES: float = 0.0
    GOLD_DIR: Path = PROCESSED_DATA_DIR


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


MODELING_SETTINGS = ModelingSettings()


# Registry config
@dataclass(frozen=True)
class RegistrySettings:
    """Aliases of the Model Registry lifecycle.

    - `CHAMPION_ALIAS`: version in production, the one batch inference loads.
    - `CHALLENGER_ALIAS`: runner-up version, kept ready to compare or swap in.
    """

    CHAMPION_ALIAS: str = "champion"
    CHALLENGER_ALIAS: str = "challenger"


REGISTRY_SETTINGS = RegistrySettings()


# Monitoring config
@dataclass(frozen=True)
class MonitoringSettings:
    """Periods and thresholds of the P1 drift report, the retraining, and the A/B test.

    Every period is half open: it starts on its first date and stops before its second.

    - `REFERENCE`: data the P1 champion was fitted on, `train_full` of `TRAINING_YEAR`.
    - `CURRENT`: production period compared against the reference for drift.
    - `RETRAIN`: data the alternative version is fitted on, the reference plus the first
      nine months of the current period.
    - `RETRAIN_VAL_START`: first day of the retraining validation. The alternative version
      is scored there before its final fit on the whole `RETRAIN` period.
    - `EVALUATION`: A/B test period. Neither version trains on it.
    - `DRIFT_THRESHOLD`: distance from which a column counts as drifted. Numeric columns
      use the normed Wasserstein distance and categorical columns the Jensen-Shannon one.
    - `BOOTSTRAP_SAMPLES`: resamples of days in the paired bootstrap of the A/B test.
    - `CONFIDENCE`: level of the bootstrap interval of the MAE difference.
    - `KNOWN_STATION_TOLERANCE`: largest MAE increase, in departures per hour, that the
      challenger may show on the stations the champion already knew and still win.
    - `REPORTS_DIR`: directory of the HTML, JSON, CSV, and figures, `reports/monitoring`.
    """

    REFERENCE: tuple[date, date] = (date(2023, 1, 1), date(2023, 10, 1))
    CURRENT: tuple[date, date] = (date(2024, 1, 1), date(2025, 1, 1))
    RETRAIN: tuple[date, date] = (date(2023, 1, 1), date(2024, 10, 1))
    RETRAIN_VAL_START: date = date(2024, 7, 1)
    EVALUATION: tuple[date, date] = (date(2024, 10, 1), date(2025, 1, 1))
    DRIFT_THRESHOLD: float = 0.1
    BOOTSTRAP_SAMPLES: int = 1000
    CONFIDENCE: float = 0.95
    KNOWN_STATION_TOLERANCE: float = 0.02
    REPORTS_DIR: Path = REPORTS_DIR / "monitoring"


MONITORING_SETTINGS = MonitoringSettings()


# Predict config
@dataclass(frozen=True)
class PredictSettings:
    """Settings for batch inference.

    - `OUTPUT_DIR`: directory for the prediction files. Points at `data/processed`.
    """

    OUTPUT_DIR: Path = PROCESSED_DATA_DIR


PREDICT_SETTINGS = PredictSettings()
