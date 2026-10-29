"""Build trip features and station demand from a yearly trips Parquet file."""

from pathlib import Path

from loguru import logger
import pandas as pd

from mibici_e21.config import DATA_LAKE_SETTINGS, FEATURE_STORE_SETTINGS
from mibici_e21.schemas.columns import GoldColumns, StationColumns, TripColumns


def load_silver_trips(
    year: int,
    silver_dir: Path | None = None,
) -> pd.DataFrame:
    """Load `mibici-trips-{year}.parquet` from the data lake.

    - `year`: trip year used in the file name.
    - `silver_dir`: directory that holds the silver Parquet files. Defaults to
      `DATA_LAKE_SETTINGS.SILVER_DIR` (`data/raw`).

    Raises `FileNotFoundError` when that file is not in the directory.
    """
    input_path = (silver_dir or DATA_LAKE_SETTINGS.SILVER_DIR) / f"mibici-trips-{year}.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Parquet file for trips not found for year {year}: {input_path}")

    logger.info(f"Loading trips for {year} from {input_path}")
    return pd.read_parquet(input_path, engine="pyarrow")


def _trip_duration_minutes(trips: pd.DataFrame) -> pd.Series:
    """Return each trip's end time minus its start time, in minutes."""
    start = pd.to_datetime(trips[TripColumns.START_TIME])
    end = pd.to_datetime(trips[TripColumns.END_TIME])
    return (end - start).dt.total_seconds() / 60.0


def _is_weekend(day_of_week: pd.Series) -> pd.Series:
    """Return 1 on Saturday and Sunday and 0 otherwise, with Monday as day 0."""
    return (day_of_week >= 5).astype(int)


def _extract_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the start and end times, both station ids, and the trip features.

    The features are `hour`, `day_of_week` (Monday is 0), `is_weekend`, and
    `duration_minutes`. The input frame is left unchanged.
    """
    start = pd.to_datetime(df[TripColumns.START_TIME])
    day_of_week = start.dt.dayofweek
    features = pd.DataFrame(
        {
            TripColumns.START_TIME: start,
            TripColumns.END_TIME: pd.to_datetime(df[TripColumns.END_TIME]),
            TripColumns.ORIGIN_ID: df[TripColumns.ORIGIN_ID],
            TripColumns.DESTINATION_ID: df[TripColumns.DESTINATION_ID],
            GoldColumns.HOUR: start.dt.hour,
            GoldColumns.DAY_OF_WEEK: day_of_week,
            GoldColumns.IS_WEEKEND: _is_weekend(day_of_week),
            GoldColumns.DURATION_MINUTES: _trip_duration_minutes(df),
        }
    )
    logger.info(f"Extracted trip features for {len(features)} rows")
    return features


def _aggregate_station_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Count departures and arrivals by station, date, and hour.

    `net_flow` is arrivals minus departures. A station that only appears on one
    side of the merge gets a zero for the missing count. `day_of_week` uses
    Monday as 0, and `is_weekend` is 1 on Saturday and Sunday.
    """
    df_copy = df.copy()

    start_dt = pd.to_datetime(df_copy[TripColumns.START_TIME])
    df_copy[GoldColumns.DATE] = start_dt.dt.date
    df_copy[GoldColumns.HOUR] = start_dt.dt.hour

    group_cols_origin = [GoldColumns.DATE, GoldColumns.HOUR, TripColumns.ORIGIN_ID]
    group_cols_dest = [GoldColumns.DATE, GoldColumns.HOUR, TripColumns.DESTINATION_ID]

    departures = (
        df_copy.groupby(group_cols_origin)
        .size()
        .reset_index(name=GoldColumns.DEPARTURES_COUNT)
        .rename(columns={TripColumns.ORIGIN_ID: GoldColumns.STATION_ID})
    )
    arrivals = (
        df_copy.groupby(group_cols_dest)
        .size()
        .reset_index(name=GoldColumns.ARRIVALS_COUNT)
        .rename(columns={TripColumns.DESTINATION_ID: GoldColumns.STATION_ID})
    )

    merge_keys = [GoldColumns.DATE, GoldColumns.HOUR, GoldColumns.STATION_ID]
    df_demand = pd.merge(departures, arrivals, on=merge_keys, how="outer").fillna(0)
    df_demand[GoldColumns.DEPARTURES_COUNT] = df_demand[GoldColumns.DEPARTURES_COUNT].astype(int)
    df_demand[GoldColumns.ARRIVALS_COUNT] = df_demand[GoldColumns.ARRIVALS_COUNT].astype(int)
    df_demand[GoldColumns.NET_FLOW] = (
        df_demand[GoldColumns.ARRIVALS_COUNT] - df_demand[GoldColumns.DEPARTURES_COUNT]
    )

    demand_date = pd.to_datetime(df_demand[GoldColumns.DATE])
    df_demand[GoldColumns.DAY_OF_WEEK] = demand_date.dt.dayofweek
    df_demand[GoldColumns.IS_WEEKEND] = _is_weekend(df_demand[GoldColumns.DAY_OF_WEEK])

    logger.info(f"Aggregated station demand into {len(df_demand)} rows")
    return df_demand


def _clean_and_filter_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Drop trips with null stations or times, and trips outside the duration bounds.

    A trip is kept when its duration is greater than
    `FEATURE_STORE_SETTINGS.MIN_TRIP_DURATION_MINUTES` and less than or equal to
    `FEATURE_STORE_SETTINGS.MAX_TRIP_DURATION_MINUTES`.
    """
    valid_stations = df[TripColumns.ORIGIN_ID].notna() & df[TripColumns.DESTINATION_ID].notna()
    valid_dates = df[TripColumns.START_TIME].notna() & df[TripColumns.END_TIME].notna()
    df_clean = df[valid_stations & valid_dates].copy()
    dropped_nulls = len(df) - len(df_clean)
    logger.info(f"Dropped {dropped_nulls} rows with null stations or times")

    duration_min = _trip_duration_minutes(df_clean)
    valid_duration = (duration_min > FEATURE_STORE_SETTINGS.MIN_TRIP_DURATION_MINUTES) & (
        duration_min <= FEATURE_STORE_SETTINGS.MAX_TRIP_DURATION_MINUTES
    )
    dropped_duration = len(df_clean) - int(valid_duration.sum())
    logger.info(f"Dropped {dropped_duration} rows outside the duration bounds")
    return df_clean[valid_duration].copy()


def materialize_yearly_trip_features(
    year: int,
    gold_dir: Path = FEATURE_STORE_SETTINGS.GOLD_DIR,
) -> tuple[Path, Path]:
    """Write `features_{year}.parquet` and `demand_aggregated_{year}.parquet`.

    - `year`: trip year to read and to use in the output names.
    - `gold_dir`: directory for the written files. Defaults to
      `FEATURE_STORE_SETTINGS.GOLD_DIR` (`data/processed`).

    Silver trips are read from `DATA_LAKE_SETTINGS.SILVER_DIR` (`data/raw`).
    Returns the path of the trip-feature file and the path of the demand file.
    """
    df_silver = load_silver_trips(year)
    df_clean = _clean_and_filter_trips(df_silver)
    df_features = _extract_trip_features(df_clean)
    df_demand = _aggregate_station_demand(df_clean)

    output_features = gold_dir / f"features_{year}.parquet"
    output_demand = gold_dir / f"demand_aggregated_{year}.parquet"
    df_features.to_parquet(output_features, index=False, engine="pyarrow")
    df_demand.to_parquet(output_demand, index=False, engine="pyarrow")

    logger.info(f"Wrote trip features to {output_features}")
    logger.info(f"Wrote station demand to {output_demand}")
    return output_features, output_demand


def load_gold_demand(year: int, gold_dir: Path | None = None) -> pd.DataFrame:
    """Load `demand_aggregated_{year}.parquet` with `date` parsed to datetime.

    - `year`: demand year used in the file name.
    - `gold_dir`: directory of the gold files. Defaults to
      `FEATURE_STORE_SETTINGS.GOLD_DIR` (`data/processed`).

    Raises `FileNotFoundError` when the file is missing.
    """
    directory = gold_dir or FEATURE_STORE_SETTINGS.GOLD_DIR
    demand_path = directory / f"demand_aggregated_{year}.parquet"
    if not demand_path.exists():
        raise FileNotFoundError(
            f"No gold demand for {year} at {demand_path}. Run `mlflow run . -e ingest` first."
        )

    logger.info(f"Loading gold demand for {year} from {demand_path}")
    demand = pd.read_parquet(demand_path, engine="pyarrow")
    demand[GoldColumns.DATE] = pd.to_datetime(demand[GoldColumns.DATE])
    return demand


def load_station_locations(silver_dir: Path | None = None) -> pd.DataFrame:
    """Load the id, latitude, and longitude of every station from the station catalog.

    - `silver_dir`: directory of the silver files. Defaults to `DATA_LAKE_SETTINGS.SILVER_DIR`.

    The id column is renamed to `station_id`, to join with the gold demand. Raises
    `FileNotFoundError` when the catalog is missing.
    """
    path = (silver_dir or DATA_LAKE_SETTINGS.SILVER_DIR) / "mibici-stations.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No station catalog at {path}. Run `mlflow run . -e ingest`.")
    columns = [StationColumns.ID, StationColumns.LATITUDE, StationColumns.LONGITUDE]
    stations = pd.read_parquet(path, columns=columns, engine="pyarrow")
    return stations.rename(columns={StationColumns.ID: GoldColumns.STATION_ID})


def materialize_features_range(
    start_year: int,
    end_year: int,
    gold_dir: Path = FEATURE_STORE_SETTINGS.GOLD_DIR,
) -> list[tuple[Path, Path]]:
    """Write trip features and station demand for each year in the range.

    - `start_year`: first year to read. It is included.
    - `end_year`: last year to read. It is included.
    - `gold_dir`: directory for the written files. Defaults to
      `FEATURE_STORE_SETTINGS.GOLD_DIR` (`data/processed`).

    Silver trips are read from `DATA_LAKE_SETTINGS.SILVER_DIR` (`data/raw`).

    Each year is handled like `materialize_yearly_trip_features`. A missing year raises
    `FileNotFoundError`. Returns one `(features path, demand path)` pair per year.
    """
    logger.info(f"Materializing features for {start_year}-{end_year}")
    generated_files = []
    for year in range(start_year, end_year + 1):
        paths = materialize_yearly_trip_features(year, gold_dir=gold_dir)
        generated_files.append(paths)
    logger.info(f"Feature range complete: {len(generated_files)} years")
    return generated_files
