"""Download processed MiBici Parquet files and clean local data directories."""

from pathlib import Path
import shutil

from huggingface_hub import HfApi, snapshot_download
from loguru import logger

from mibici_e21.config import (
    DATA_LAKE_SETTINGS,
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TEMP_DATA_DIR,
)


def download_dataset(
    repo_id: str = DATA_LAKE_SETTINGS.HF_DATASET_REPO,
    output_dir: Path = DATA_LAKE_SETTINGS.PROCESSED_DIR,
) -> Path:
    """Download the dataset Parquet files into `output_dir`.

    Only `*.parquet` files are copied. `repo_id` defaults to
    `DATA_LAKE_SETTINGS.HF_DATASET_REPO`, and `output_dir` defaults to
    `DATA_LAKE_SETTINGS.PROCESSED_DIR`.
    """
    if not repo_id:
        raise ValueError("HF_DATASET_REPO is not configured in DATA_LAKE_SETTINGS.")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Parquet files from {repo_id} into {output_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns="*.parquet",
    )

    api = HfApi()
    dataset_info = api.dataset_info(repo_id=repo_id)
    commit_sha = dataset_info.sha

    version_file = output_dir / "data-version.txt"
    version_file.write_text(f"{commit_sha}\n", encoding="utf-8")
    logger.info(f"Dataset version: {commit_sha}")

    logger.info(f"Dataset download complete: {output_dir}")
    return output_dir


def _clean_folder(directory: Path) -> None:
    """Delete every file and directory inside `directory`.

    The directory itself and any `.gitkeep` file are kept. When the directory
    does not exist, nothing is removed.
    """
    if not directory.exists():
        logger.info(f"Directory does not exist: {directory}")
        return

    removed = 0
    for item in directory.iterdir():
        if item.name == ".gitkeep":
            continue
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
        removed += 1
    logger.info(f"Removed {removed} items from {directory}")


def clean_temp_files(temp_dir: Path = TEMP_DATA_DIR) -> None:
    """Delete the contents of `data/temp` and keep the directory.

    - `temp_dir`: directory to clean. Defaults to `TEMP_DATA_DIR`.
    """
    logger.info(f"Cleaning temp files in {temp_dir}")
    _clean_folder(temp_dir)


def clean_raw_files(raw_dir: Path = RAW_DATA_DIR) -> None:
    """Delete the contents of `data/raw` and keep the directory.

    - `raw_dir`: directory to clean. Defaults to `RAW_DATA_DIR`.
    """
    logger.info(f"Cleaning raw files in {raw_dir}")
    _clean_folder(raw_dir)


def clean_interim_files(interim_dir: Path = INTERIM_DATA_DIR) -> None:
    """Delete the contents of `data/interim` and keep the directory.

    - `interim_dir`: directory to clean. Defaults to `INTERIM_DATA_DIR`.
    """
    logger.info(f"Cleaning interim files in {interim_dir}")
    _clean_folder(interim_dir)


def clean_processed_files(processed_dir: Path = PROCESSED_DATA_DIR) -> None:
    """Delete the contents of `data/processed` and keep the directory.

    - `processed_dir`: directory to clean. Defaults to `PROCESSED_DATA_DIR`.
    """
    logger.info(f"Cleaning processed files in {processed_dir}")
    _clean_folder(processed_dir)


def clean_external_files(external_dir: Path = EXTERNAL_DATA_DIR) -> None:
    """Delete the contents of `data/external` and keep the directory.

    - `external_dir`: directory to clean. Defaults to `EXTERNAL_DATA_DIR`.
    """
    logger.info(f"Cleaning external files in {external_dir}")
    _clean_folder(external_dir)


def clean_all_data_files() -> None:
    """Delete the contents of every directory under `data/`."""
    logger.info("Cleaning all data directories")
    clean_temp_files()
    clean_raw_files()
    clean_interim_files()
    clean_processed_files()
    clean_external_files()


if __name__ == "__main__":
    from mibici_e21.config import MODELING_SETTINGS
    from mibici_e21.features import materialize_yearly_trip_features

    download_dataset()
    materialize_yearly_trip_features(MODELING_SETTINGS.DEFAULT_YEAR)
