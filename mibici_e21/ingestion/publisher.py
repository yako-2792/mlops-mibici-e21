"""Write MiBici Parquet files and upload them to the Hugging Face dataset."""

from pathlib import Path

from huggingface_hub import HfApi
from loguru import logger
import pyarrow as pa
import pyarrow.parquet as pq

from mibici_e21.config import DATA_LAKE_SETTINGS


class MiBiciPublisher:
    """Publish transformed MiBici tables as local Parquet and sync them to Hugging Face."""

    def __init__(
        self,
        processed_dir: Path = DATA_LAKE_SETTINGS.PROCESSED_DIR,
        hf_repo_id: str | None = DATA_LAKE_SETTINGS.HF_DATASET_REPO,
    ) -> None:
        """Create a publisher.

        - `processed_dir`: directory where Parquet files are written.
        - `hf_repo_id`: Hugging Face dataset repository, as `owner/name`.
        """
        self.processed_dir = processed_dir
        self.hf_repo_id = hf_repo_id
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._api = HfApi() if hf_repo_id else None

    def publish_trips_yearly(self, tables: list[pa.Table], year: int) -> Path:
        """Write one yearly trip Parquet from the month tables, in list order.

        The tables are concatenated into a single write. PyArrow chooses the
        row groups and compresses with its default, Snappy. The file is
        `mibici-trips-{year}.parquet`.
        """
        if not tables:
            raise ValueError(f"No tables provided for year {year}")

        yearly_table = pa.concat_tables(tables)
        target_path = self.processed_dir / f"mibici-trips-{year}.parquet"
        pq.write_table(yearly_table, target_path)
        logger.info(f"Wrote {target_path.name} ({yearly_table.num_rows:,} rows)")
        return target_path

    def publish_stations(self, table: pa.Table, filename: str = "mibici-stations.parquet") -> Path:
        """Write the station table to `mibici-stations.parquet`."""
        target_path = self.processed_dir / filename
        pq.write_table(table, target_path)
        logger.info(f"Wrote {target_path.name} ({table.num_rows:,} rows)")
        return target_path

    def sync_to_huggingface(self, local_path: Path | str, path_in_repo: str | None = None) -> None:
        """Upload a local Parquet file to the configured Hugging Face dataset.

        A relative `local_path` is resolved inside `processed_dir`. The path
        inside the repository defaults to the file name.
        """
        path = Path(local_path)
        if not path.is_absolute():
            path = self.processed_dir / path

        if not path.exists():
            logger.error(f"Local file not found: {path.name}")
            raise FileNotFoundError(f"El archivo local '{path}' no existe.")

        if not self.hf_repo_id or not self._api:
            logger.error("Hugging Face repo is not configured")
            raise ValueError("HF_REPO_ID no está configurado en DATA_LAKE_SETTINGS.")

        target_repo_path = path_in_repo or path.name
        logger.info(f"Uploading {path.name} to {self.hf_repo_id}")
        self._api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=target_repo_path,
            repo_id=self.hf_repo_id,
            repo_type="dataset",
        )
        logger.info(f"Uploaded {target_repo_path}")
