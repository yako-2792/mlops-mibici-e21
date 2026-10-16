"""Cast downloaded MiBici CSVs to the Parquet schemas."""

from pathlib import Path

from loguru import logger
import pyarrow as pa
import pyarrow.csv as pa_csv

from mibici_e21.schemas import STATIONS, TRIPS, StationColumns, TripColumns


class MiBiciTransformer:
    """Read a MiBici CSV by column position and cast it to a schema."""

    def transform_trips(self, csv_path: Path) -> pa.Table:
        """Cast a trip CSV to `TRIPS`.

        The original header row is skipped. Columns are named from `TripColumns`
        by position, and the data rows are kept. Start and end times are accepted
        as ISO-8601 or as `%d/%m/%Y %H:%M`, then stored as `timestamp[ms]`.
        """
        column_names = [col.value for col in TripColumns]
        raw_table = self._read_csv_positional(csv_path, column_names)
        table = self._cast_schema(raw_table, TRIPS)
        logger.info(f"Trips {csv_path.name}: {table.num_rows:,} rows")
        return table

    def transform_stations(self, csv_path: Path) -> pa.Table:
        """Cast a station CSV to `STATIONS`.

        The original header row is skipped. Columns are named from `StationColumns`
        by position, and the data rows are kept.
        """
        column_names = [col.value for col in StationColumns]
        raw_table = self._read_csv_positional(csv_path, column_names)
        table = self._cast_schema(raw_table, STATIONS)
        logger.info(f"Stations {csv_path.name}: {table.num_rows:,} rows")
        return table

    def _read_csv_positional(self, csv_path: Path, column_names: list[str]) -> pa.Table:
        """Read a CSV, skip its header row, and name the columns by position.

        Nulls come from PyArrow's default tokens, including an empty cell and
        `NA`, and string columns can hold those nulls. Timestamps are parsed
        with ISO-8601 first and `%d/%m/%Y %H:%M` second.
        """
        read_options = pa_csv.ReadOptions(
            skip_rows=1,
            column_names=column_names,
            autogenerate_column_names=False,
        )
        parse_options = pa_csv.ParseOptions(delimiter=",")
        convert_options = pa_csv.ConvertOptions(
            strings_can_be_null=True,
            timestamp_parsers=[pa_csv.ISO8601, "%d/%m/%Y %H:%M"],
        )

        return pa_csv.read_csv(
            csv_path,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        )

    def _cast_schema(self, table: pa.Table, target_schema: pa.Schema) -> pa.Table:
        """Cast every column of `table` to `target_schema`."""
        return table.cast(target_schema)
