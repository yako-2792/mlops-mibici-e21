"""Download processed MiBici Parquet files and clean local data directories."""

from pathlib import Path
import shutil

from huggingface_hub import HfApi, snapshot_download
from loguru import logger

from mibici_e21.config import DATA_LAKE_SETTINGS


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


def clean_temp_files(temp_dir: Path = DATA_LAKE_SETTINGS.TEMP_DIR) -> None:
    """Delete every file and directory inside `temp_dir`.

    `temp_dir` defaults to `DATA_LAKE_SETTINGS.TEMP_DIR`. The directory itself
    is kept.
    """
    if not temp_dir.exists():
        logger.info(f"Temp directory does not exist: {temp_dir}")
        return

    removed = 0
    for item in temp_dir.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
        removed += 1
    logger.info(f"Removed {removed} items from {temp_dir}")


def clean_processed_files(processed_dir: Path = DATA_LAKE_SETTINGS.PROCESSED_DIR) -> None:
    """Delete Parquet files inside `processed_dir`.

    `processed_dir` defaults to `DATA_LAKE_SETTINGS.PROCESSED_DIR`. Other files
    in that directory are kept.
    """
    if not processed_dir.exists():
        logger.info(f"Processed directory does not exist: {processed_dir}")
        return

    parquet_files = list(processed_dir.glob("*.parquet"))
    for parquet_file in parquet_files:
        parquet_file.unlink()
    logger.info(f"Removed {len(parquet_files)} Parquet files from {processed_dir}")


def clean_all_data_files() -> None:
    """Delete temporary downloads and processed Parquet files."""
    logger.info("Cleaning temporary downloads and processed Parquet files")
    clean_temp_files()
    clean_processed_files()
