"""PyArrow schema for MiBici stations.

`STATIONS` sets the type of each column when the station catalog is
converted to Parquet:

- `id`: station identifier.
- `name`: station name.
- `latitude`: latitude in decimal degrees.
- `longitude`: longitude in decimal degrees.
- `dpcapacity`: number of docks at the station.
"""

import pyarrow as pa

from mibici_e21.schemas.columns import StationColumns

STATIONS = pa.schema(
    [
        (StationColumns.ID, pa.int32()),
        (StationColumns.NAME, pa.string()),
        (StationColumns.LATITUDE, pa.float64()),
        (StationColumns.LONGITUDE, pa.float64()),
        (StationColumns.DPCAPACITY, pa.int32()),
    ]
)
