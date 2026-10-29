"""Download, transform, and publish MiBici trip and station files."""

from pathlib import Path

from loguru import logger

from mibici_e21.ingestion.extractor import MiBiciExtractor
from mibici_e21.ingestion.publisher import MiBiciPublisher
from mibici_e21.ingestion.transformer import MiBiciTransformer


class IngestionPipeline:
    """Run extraction, transformation, and publishing for MiBici open data."""

    def __init__(
        self,
        extractor: MiBiciExtractor | None = None,
        transformer: MiBiciTransformer | None = None,
        publisher: MiBiciPublisher | None = None,
    ) -> None:
        """Create a pipeline.

        Missing collaborators are built with their default settings.

        - `extractor`: catalog download. Defaults to `MiBiciExtractor`.
        - `transformer`: CSV cast. Defaults to `MiBiciTransformer`.
        - `publisher`: Parquet write and Hugging Face upload. Defaults to `MiBiciPublisher`.
        """
        self.extractor = extractor or MiBiciExtractor()
        self.transformer = transformer or MiBiciTransformer()
        self.publisher = publisher or MiBiciPublisher()

    def ingest_trips_year(self, year: int, push_to_hf: bool = False) -> Path:
        """Download one year of trips and write `mibici-trips-{year}.parquet`.

        Months absent from the catalog are skipped. The Parquet contains only
        the months that exist. When `push_to_hf` is true, the file is uploaded
        to the Hugging Face dataset.
        """
        logger.info(f"Ingesting trips for {year}")
        tables = []
        for month in range(1, 13):
            try:
                csv_path = self.extractor.download_trips(year, month)
            except FileNotFoundError:
                logger.info(f"Trips {year}-{month:02d}: not in catalog")
                continue
            table = self.transformer.transform_trips(csv_path)
            tables.append(table)

        parquet_path = self.publisher.publish_trips_yearly(tables, year)
        logger.info(f"Trips {year}: {len(tables)} months ingested")

        if push_to_hf:
            self.publisher.sync_to_huggingface(parquet_path)

        return parquet_path

    def ingest_stations(self, push_to_hf: bool = False) -> Path:
        """Download the station catalog and write `mibici-stations.parquet`.

        When `push_to_hf` is true, the file is uploaded to the Hugging Face dataset.
        """
        logger.info("Ingesting stations")
        csv_path = self.extractor.download_stations()
        table = self.transformer.transform_stations(csv_path)
        parquet_path = self.publisher.publish_stations(table)

        if push_to_hf:
            self.publisher.sync_to_huggingface(parquet_path)

        logger.info("Stations ingestion complete")
        return parquet_path

    def ingest_trips_range(
        self, start_year: int, end_year: int, push_to_hf: bool = False
    ) -> list[Path]:
        """Ingest each year from `start_year` through `end_year`, inclusive.

        Each year is handled like `ingest_trips_year`, and `push_to_hf` is
        forwarded. A year with no trip files is skipped. The returned list
        holds the Parquet path of every year that was written.
        """
        logger.info(f"Ingesting trips {start_year}-{end_year}")
        parquet_paths = []
        for year in range(start_year, end_year + 1):
            try:
                path = self.ingest_trips_year(year, push_to_hf=push_to_hf)
                parquet_paths.append(path)
            except ValueError as error:
                logger.warning(f"Skipping {year}: {error}")
        logger.info(f"Trip range complete: {len(parquet_paths)} files")
        return parquet_paths
