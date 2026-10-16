"""Tests for `MiBiciTransformer` using small CSV fixtures."""

import datetime
from pathlib import Path

import pyarrow as pa

from mibici_e21.ingestion.transformer import MiBiciTransformer
from mibici_e21.schemas import STATIONS, TRIPS

TRIPS_CSV = """\
Viaje_Id,Usuario_Id,Genero,Año_de_nacimiento,Inicio_del_viaje,Fin_del_viaje,Origen_Id,Destino_Id
4601,10324,M,,2014-12-01 00:33:47,2014-12-01 00:36:54,47,47
4604,10324,,1989.0,2014-12-01 01:06:54,2014-12-01 01:08:45,5,5
"""

STATIONS_CSV = """\
"id","name","latitude","longitude","dpcapacity"
"2","(GDL-001) C. Epigmenio Glez./ Av. 16 de Sept.","20.666378","-103.34882","15"
"3","(GDL-002) C. Colonias  / Av.  Niños héroes","20.667228","-103.366","15"
"""


def _write_csv(tmp_path: Path, name: str, contents: str) -> Path:
    """Write a CSV fixture into the temporary test directory."""
    path = tmp_path / name
    path.write_text(contents, encoding="utf-8")
    return path


def test_transform_trips_casts_to_the_trip_schema(tmp_path: Path):
    """ISO trip times become `timestamp[ms]` and an empty birth year becomes null."""
    path = _write_csv(tmp_path, "datos_abiertos_2014_12.csv", TRIPS_CSV)

    table = MiBiciTransformer().transform_trips(path)

    assert table.schema.equals(TRIPS)
    assert table.column("inicio_del_viaje").type == pa.timestamp("ms")
    assert table.to_pylist() == [
        {
            "viaje_id": 4601,
            "usuario_id": 10324,
            "genero": "M",
            "año_de_nacimiento": None,
            "inicio_del_viaje": datetime.datetime(2014, 12, 1, 0, 33, 47),
            "fin_del_viaje": datetime.datetime(2014, 12, 1, 0, 36, 54),
            "origen_id": 47,
            "destino_id": 47,
        },
        {
            "viaje_id": 4604,
            "usuario_id": 10324,
            "genero": None,
            "año_de_nacimiento": 1989,
            "inicio_del_viaje": datetime.datetime(2014, 12, 1, 1, 6, 54),
            "fin_del_viaje": datetime.datetime(2014, 12, 1, 1, 8, 45),
            "origen_id": 5,
            "destino_id": 5,
        },
    ]


def test_transform_trips_names_columns_by_position(tmp_path: Path):
    """The original header row is replaced by `TripColumns` names, in order."""
    csv = TRIPS_CSV.replace("Año_de_nacimiento", "A}e_nacimiento", 1)
    path = _write_csv(tmp_path, "datos_abiertos_2024_02.csv", csv)

    table = MiBiciTransformer().transform_trips(path)

    assert table.column_names == list(TRIPS.names)
    assert table.column("año_de_nacimiento").to_pylist() == [None, 1989]


def test_transform_stations_casts_to_the_station_schema(tmp_path: Path):
    """Station rows keep names that contain `/` and cast coordinates to float64."""
    path = _write_csv(tmp_path, "nomenclatura_2026_09.csv", STATIONS_CSV)

    table = MiBiciTransformer().transform_stations(path)

    assert table.schema.equals(STATIONS)
    assert table.to_pylist() == [
        {
            "id": 2,
            "name": "(GDL-001) C. Epigmenio Glez./ Av. 16 de Sept.",
            "latitude": 20.666378,
            "longitude": -103.34882,
            "dpcapacity": 15,
        },
        {
            "id": 3,
            "name": "(GDL-002) C. Colonias  / Av.  Niños héroes",
            "latitude": 20.667228,
            "longitude": -103.366,
            "dpcapacity": 15,
        },
    ]
