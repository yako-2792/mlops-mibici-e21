"""Canonical column names for the MiBici Parquet schemas.

`TripColumns` and `StationColumns` are the single source of truth for the
field names used in `TRIPS` and `STATIONS`.
"""

from enum import StrEnum


class TripColumns(StrEnum):
    """Column names for a MiBici trip record.

    Member names are English. Each value is the column name in the source file.

    - `TRIP_ID`: trip identifier (`viaje_id`).
    - `USER_ID`: user identifier (`usuario_id`).
    - `GENDER`: gender reported by the user (`genero`).
    - `BIRTH_YEAR`: user birth year (`año_de_nacimiento`).
    - `START_TIME`: trip start time (`inicio_del_viaje`).
    - `END_TIME`: trip end time (`fin_del_viaje`).
    - `ORIGIN_ID`: station where the trip started (`origen_id`).
    - `DESTINATION_ID`: station where the trip ended (`destino_id`).
    """

    TRIP_ID = "viaje_id"
    USER_ID = "usuario_id"
    GENDER = "genero"
    BIRTH_YEAR = "año_de_nacimiento"
    START_TIME = "inicio_del_viaje"
    END_TIME = "fin_del_viaje"
    ORIGIN_ID = "origen_id"
    DESTINATION_ID = "destino_id"


class StationColumns(StrEnum):
    """Column names for a MiBici station record.

    - `ID`: station identifier.
    - `NAME`: station name.
    - `LATITUDE`: latitude in decimal degrees.
    - `LONGITUDE`: longitude in decimal degrees.
    - `DPCAPACITY`: number of docks at the station.
    """

    ID = "id"
    NAME = "name"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    DPCAPACITY = "dpcapacity"


class GoldColumns(StrEnum):
    """Column names for a MiBici gold record.

    - `DATE`: date.
    - `HOUR`: hour.
    - `DAY_OF_WEEK`: day of week.
    - `IS_WEEKEND`: whether the day is a weekend.
    - `DURATION_MINUTES`: duration in minutes.
    - `STATION_ID`: station identifier.
    - `DEPARTURES_COUNT`: number of departures.
    - `ARRIVALS_COUNT`: number of arrivals.
    - `NET_FLOW`: net flow.
    """

    DATE = "date"
    HOUR = "hour"
    DAY_OF_WEEK = "day_of_week"
    IS_WEEKEND = "is_weekend"
    DURATION_MINUTES = "duration_minutes"
    STATION_ID = "station_id"
    DEPARTURES_COUNT = "departures_count"
    ARRIVALS_COUNT = "arrivals_count"
    NET_FLOW = "net_flow"
