"""Reproducible ingestion: download a pinned snapshot, build gold features, write a manifest.

`python -m mibici_e21.dataset` is the `ingest` entry point of the MLflow Project. It ingests
`MODELING_SETTINGS.TRAINING_YEAR`, the year every model trains on.
"""

import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download
from loguru import logger
import mlflow
import pyarrow.parquet as pq

from mibici_e21.config import (
    DATA_LAKE_SETTINGS,
    EXTERNAL_DATA_DIR,
    FEATURE_STORE_SETTINGS,
    INTERIM_DATA_DIR,
    MODELING_SETTINGS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TEMP_DATA_DIR,
)
from mibici_e21.features import materialize_yearly_trip_features
from mibici_e21.tracking import base_tags, configure_tracking, log_data_manifest


def download_dataset(
    years: list[int] | None = None,
    repo_id: str = DATA_LAKE_SETTINGS.HF_DATASET_REPO,
    revision: str = DATA_LAKE_SETTINGS.HF_DATASET_REVISION,
    output_dir: Path = DATA_LAKE_SETTINGS.PROCESSED_DIR,
) -> Path:
    """Download the trip Parquet files of a pinned dataset revision into `output_dir`.

    - `years`: trip years to download. `None` downloads every Parquet file.
    - `repo_id`: Hugging Face dataset repository.
    - `revision`: dataset commit to download. Defaults to `HF_DATASET_REVISION`.
    - `output_dir`: destination directory. Defaults to `data/raw`.
    """
    patterns = [trips_file_name(year) for year in years] if years else ["*.parquet"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {patterns} from {repo_id}@{revision[:7]} into {output_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=output_dir,
        allow_patterns=patterns,
    )
    return output_dir


def trips_file_name(year: int) -> str:
    """Return the silver trips file name of a year."""
    return f"mibici-trips-{year}.parquet"


def data_manifest_path(year: int) -> Path:
    """Return the path of the data manifest of a year, next to its gold files."""
    return FEATURE_STORE_SETTINGS.PROCESSED_DIR / f"data_manifest_{year}.json"


def build_data_manifest(year: int, files: dict[str, Path]) -> dict:
    """Describe the files of one ingestion so a run can prove which data it used.

    - `year`: trip year of the ingestion.
    - `files`: layer name (`silver`, `gold_features`, `gold_demand`) mapped to its file.

    Each file entry records its name, SHA-256, size, and row count. The manifest has no
    timestamps, so the same inputs always produce the same manifest.
    """
    return {
        "dataset_repo": DATA_LAKE_SETTINGS.HF_DATASET_REPO,
        "dataset_revision": DATA_LAKE_SETTINGS.HF_DATASET_REVISION,
        "year": year,
        "files": {layer: _describe_file(path) for layer, path in files.items()},
    }


def read_data_manifest(year: int) -> dict:
    """Load the data manifest of a year and check that the gold demand file still matches it.

    Raises `FileNotFoundError` when the ingestion has not run for that year, and
    `ValueError` when the gold demand file changed after the ingestion.
    """
    path = data_manifest_path(year)
    if not path.exists():
        raise FileNotFoundError(
            f"No data manifest for {year} at {path}. Run `mlflow run . -e ingest` first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    demand = manifest["files"]["gold_demand"]
    demand_path = FEATURE_STORE_SETTINGS.PROCESSED_DIR / demand["name"]
    if _sha256(demand_path) != demand["sha256"]:
        raise ValueError(f"{demand_path.name} does not match its manifest. Run ingest again.")
    return manifest


def ingest_year(year: int) -> Path:
    """Download the silver trips of a year, build its gold files, and write its manifest.

    Returns the manifest path.
    """
    download_dataset(years=[year])
    features_path, demand_path = materialize_yearly_trip_features(year)
    manifest = build_data_manifest(
        year,
        {
            "silver": DATA_LAKE_SETTINGS.PROCESSED_DIR / trips_file_name(year),
            "gold_features": features_path,
            "gold_demand": demand_path,
        },
    )
    path = data_manifest_path(year)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Wrote data manifest to {path}")
    return path


def _describe_file(path: Path) -> dict:
    """Return the name, SHA-256, size in bytes, and row count of a Parquet file."""
    return {
        "name": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "rows": pq.ParquetFile(path).metadata.num_rows,
    }


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Delete the contents of `data/temp` and keep the directory."""
    logger.info(f"Cleaning temp files in {temp_dir}")
    _clean_folder(temp_dir)


def clean_raw_files(raw_dir: Path = RAW_DATA_DIR) -> None:
    """Delete the contents of `data/raw` and keep the directory."""
    logger.info(f"Cleaning raw files in {raw_dir}")
    _clean_folder(raw_dir)


def clean_interim_files(interim_dir: Path = INTERIM_DATA_DIR) -> None:
    """Delete the contents of `data/interim` and keep the directory."""
    logger.info(f"Cleaning interim files in {interim_dir}")
    _clean_folder(interim_dir)


def clean_processed_files(processed_dir: Path = PROCESSED_DATA_DIR) -> None:
    """Delete the contents of `data/processed` and keep the directory."""
    logger.info(f"Cleaning processed files in {processed_dir}")
    _clean_folder(processed_dir)


def clean_external_files(external_dir: Path = EXTERNAL_DATA_DIR) -> None:
    """Delete the contents of `data/external` and keep the directory."""
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


def main() -> None:
    """Ingest the training year and log its manifest to MLflow."""
    year = MODELING_SETTINGS.TRAINING_YEAR
    configure_tracking()
    manifest_path = ingest_year(year)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with mlflow.start_run(run_name=f"ingest_{year}"):
        mlflow.set_tags(base_tags("ingest"))
        mlflow.log_params(
            {
                "year": year,
                "dataset_repo": manifest["dataset_repo"],
                "dataset_revision": manifest["dataset_revision"],
            }
        )
        log_data_manifest(manifest_path)


if __name__ == "__main__":
    main()
