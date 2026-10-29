"""Tests for trip features and station demand built from a small trips Parquet."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from mibici_e21 import features
from mibici_e21.config import DATA_LAKE_SETTINGS
from mibici_e21.features import (
    _aggregate_station_demand,
    _clean_and_filter_trips,
    _extract_trip_features,
    load_silver_trips,
    materialize_features_range,
    materialize_yearly_trip_features,
)
from mibici_e21.schemas.columns import GoldColumns, TripColumns


def _write_trips(tmp_path: Path, year: int = 2024) -> Path:
    """Write two trips: a Saturday departure and a Monday return."""
    frame = pd.DataFrame(
        {
            TripColumns.START_TIME: [
                datetime(2024, 1, 6, 10, 15),
                datetime(2024, 1, 8, 8, 0),
            ],
            TripColumns.END_TIME: [
                datetime(2024, 1, 6, 10, 45),
                datetime(2024, 1, 8, 8, 20),
            ],
            TripColumns.ORIGIN_ID: [1, 2],
            TripColumns.DESTINATION_ID: [2, 1],
        }
    )
    path = tmp_path / f"mibici-trips-{year}.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_load_silver_trips_raises_when_the_year_is_missing(tmp_path: Path):
    """A year without a Parquet file stops before feature extraction."""
    with pytest.raises(FileNotFoundError, match="1999"):
        load_silver_trips(1999, tmp_path)


def test_extract_trip_features_adds_time_and_duration():
    """Saturday is a weekend and the trip lasts 30 minutes."""
    frame = pd.DataFrame(
        {
            TripColumns.START_TIME: [datetime(2024, 1, 6, 10, 15)],
            TripColumns.END_TIME: [datetime(2024, 1, 6, 10, 45)],
            TripColumns.ORIGIN_ID: [1],
            TripColumns.DESTINATION_ID: [2],
        }
    )

    features = _extract_trip_features(frame)

    row = features.iloc[0]
    assert row[GoldColumns.HOUR] == 10
    assert row[GoldColumns.DAY_OF_WEEK] == 5
    assert row[GoldColumns.IS_WEEKEND] == 1
    assert row[GoldColumns.DURATION_MINUTES] == 30


def test_aggregate_station_demand_counts_both_sides_of_a_trip():
    """The origin loses one bike and the destination gains one."""
    frame = pd.DataFrame(
        {
            TripColumns.START_TIME: [datetime(2024, 1, 8, 8, 0)],
            TripColumns.END_TIME: [datetime(2024, 1, 8, 8, 20)],
            TripColumns.ORIGIN_ID: [1],
            TripColumns.DESTINATION_ID: [2],
        }
    )

    demand = _aggregate_station_demand(frame).set_index(GoldColumns.STATION_ID)

    assert demand.loc[1, GoldColumns.DEPARTURES_COUNT] == 1
    assert demand.loc[1, GoldColumns.ARRIVALS_COUNT] == 0
    assert demand.loc[1, GoldColumns.NET_FLOW] == -1
    assert demand.loc[2, GoldColumns.DEPARTURES_COUNT] == 0
    assert demand.loc[2, GoldColumns.ARRIVALS_COUNT] == 1
    assert demand.loc[2, GoldColumns.NET_FLOW] == 1
    assert demand.loc[1, GoldColumns.IS_WEEKEND] == 0


def test_clean_and_filter_trips_drops_nulls_and_out_of_range_durations():
    """Null stations and durations outside the configured bounds are removed."""
    frame = pd.DataFrame(
        {
            TripColumns.START_TIME: [
                datetime(2024, 1, 8, 8, 0),
                datetime(2024, 1, 8, 8, 0),
                datetime(2024, 1, 8, 8, 0),
                datetime(2024, 1, 8, 8, 0),
                datetime(2024, 1, 8, 8, 0),
            ],
            TripColumns.END_TIME: [
                datetime(2024, 1, 8, 8, 20),
                datetime(2024, 1, 8, 8, 0),
                datetime(2024, 1, 8, 16, 0),
                datetime(2024, 1, 8, 16, 1),
                datetime(2024, 1, 8, 8, 20),
            ],
            TripColumns.ORIGIN_ID: [1, 1, 1, 1, None],
            TripColumns.DESTINATION_ID: [2, 2, 2, 2, 2],
        }
    )

    cleaned = _clean_and_filter_trips(frame)
    durations = (
        pd.to_datetime(cleaned[TripColumns.END_TIME])
        - pd.to_datetime(cleaned[TripColumns.START_TIME])
    ).dt.total_seconds() / 60

    assert durations.tolist() == [20, 480]


def _point_lake_at(monkeypatch: pytest.MonkeyPatch, raw_dir: Path) -> None:
    """Make silver reads use a temporary data lake directory."""
    monkeypatch.setattr(
        features,
        "DATA_LAKE_SETTINGS",
        replace(DATA_LAKE_SETTINGS, SILVER_DIR=raw_dir),
    )


def test_materialize_trip_features_writes_both_parquet_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The yearly run reads silver trips from raw and writes both gold files."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    _write_trips(raw_dir)
    _point_lake_at(monkeypatch, raw_dir)

    features_path, demand_path = materialize_yearly_trip_features(2024, processed_dir)

    assert features_path == processed_dir / "features_2024.parquet"
    assert demand_path == processed_dir / "demand_aggregated_2024.parquet"
    assert len(pd.read_parquet(features_path)) == 2
    assert len(pd.read_parquet(demand_path)) == 4
    assert not (raw_dir / "features_2024.parquet").exists()


def test_materialize_features_range_writes_each_year(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An inclusive range reads each year from raw and writes gold files to processed."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    _write_trips(raw_dir, 2023)
    _write_trips(raw_dir, 2024)
    _point_lake_at(monkeypatch, raw_dir)

    paths = materialize_features_range(2023, 2024, processed_dir)

    assert paths == [
        (
            processed_dir / "features_2023.parquet",
            processed_dir / "demand_aggregated_2023.parquet",
        ),
        (
            processed_dir / "features_2024.parquet",
            processed_dir / "demand_aggregated_2024.parquet",
        ),
    ]


def test_materialize_features_range_stops_when_a_year_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A missing year in the data lake raises before later years are written."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    _write_trips(raw_dir, 2023)
    _point_lake_at(monkeypatch, raw_dir)

    with pytest.raises(FileNotFoundError, match="2024"):
        materialize_features_range(2023, 2024, processed_dir)

    assert not (processed_dir / "features_2024.parquet").exists()
