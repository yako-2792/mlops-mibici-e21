"""PyArrow schema for MiBici trips.

`TRIPS` sets the type of each column when trip CSVs are converted to
Parquet:

- `viaje_id`: trip identifier.
- `usuario_id`: user identifier.
- `genero`: gender reported by the user.
- `año_de_nacimiento`: user birth year.
- `inicio_del_viaje`: trip start time, with millisecond resolution.
- `fin_del_viaje`: trip end time, with millisecond resolution.
- `origen_id`: station where the trip started.
- `destino_id`: station where the trip ended.
"""

import pyarrow as pa

from mibici_e21.schemas.columns import TripColumns

TRIPS = pa.schema(
    [
        (TripColumns.TRIP_ID, pa.int64()),
        (TripColumns.USER_ID, pa.int64()),
        (TripColumns.GENDER, pa.string()),
        (TripColumns.BIRTH_YEAR, pa.int16()),
        (TripColumns.START_TIME, pa.timestamp("ms")),
        (TripColumns.END_TIME, pa.timestamp("ms")),
        (TripColumns.ORIGIN_ID, pa.int32()),
        (TripColumns.DESTINATION_ID, pa.int32()),
    ]
)
