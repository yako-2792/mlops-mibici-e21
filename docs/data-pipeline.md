# Data pipeline

This document follows MiBici data from the public portal to the gold tables that the models train on. It covers the data source, ingestion, storage and versioning, and the processing that runs before training.

For a runnable walkthrough from download to predictions, see [testing-e2e-flow.md](testing-e2e-flow.md).

---

## Overview

```text
mibici.net open-data catalog
        │  extractor.py      download monthly CSVs
        ▼
data/temp/        Bronze  raw CSVs
        │  transformer.py    positional names + PyArrow schema cast
        │  publisher.py      one Parquet per year
        ▼
data/raw/         Silver  mibici-trips-{year}.parquet, mibici-stations.parquet
        │  ▲
        │  └── mlflow run . -e ingest ◄── Hugging Face dataset (yako2792/mibici-dataset), pinned revision
        │                               ▲
        │                               └── upload when HF=true
        │  features.py       clean, enrich, aggregate
        ▼
data/processed/   Gold    features_{year}.parquet, demand_aggregated_{year}.parquet, data_manifest_{year}.json
        │
        ▼
training (mlflow run . -e train)
```

| Layer | Path | What happens |
| --- | --- | --- |
| Bronze | `data/temp/` | `extractor.py` downloads the trip and station CSVs. Months missing from the catalog are skipped. Trip files are saved as `datos_abiertos_YYYY_MM.csv`. |
| Silver | `data/raw/` | `transformer.py` drops the original header, names columns by position, and casts them with PyArrow. `publisher.py` writes `mibici-trips-{year}.parquet` and `mibici-stations.parquet`. |
| Gold | `data/processed/` | `features.py` reads the yearly trip Parquet from `data/raw/`, cleans it, and writes `features_{year}.parquet` and `demand_aggregated_{year}.parquet`. |
| Share | Hugging Face | `pipeline.py` uploads the silver Parquet files when `HF=true`. The dataset is `yako2792/mibici-dataset`. |

---

## Data source

The primary source is the [MiBici open-data portal](https://www.mibici.net/es/datos-abiertos/). It publishes:

| File | Pattern | Content |
| --- | --- | --- |
| Trips | `datos_abiertos_YYYY_MM*.csv` | One CSV per month with every trip: user, start and end time, origin and destination station. |
| Stations | `nomenclatura_*.csv` | The station catalog: id, name, coordinates, and dock capacity. |

Each download URL contains a file id that cannot be derived from the year and month. `MiBiciExtractor.discover_catalog()` reads the catalog page, collects every `.csv` link, and maps each file name to its absolute URL. Trip files are then matched by the `datos_abiertos_YYYY_MM` prefix, and the station catalog by the `nomenclatura_` prefix. The base URL is `DATA_LAKE_SETTINGS.MIBICI_BASE_URL` in `mibici_e21/config.py`.

> [!NOTE]
> The [Machine Learning Canvas](Machine%20Learning%20Canvas.pdf) lists holiday calendars and weather data as optional context sources. They are not ingested yet.

---

## Ingestion

Code: `mibici_e21/ingestion/`.

### Steps

| Step | Module | What it does |
| --- | --- | --- |
| Extract | `extractor.py` | `MiBiciExtractor` loads the catalog once per run and streams each CSV into `data/temp/`. |
| Transform | `transformer.py` | `MiBiciTransformer` reads the CSV with PyArrow, skips the original header, names the columns by position from `TripColumns` or `StationColumns`, and casts the table to the `TRIPS` or `STATIONS` schema. |
| Publish | `publisher.py` | `MiBiciPublisher` concatenates the month tables into one yearly Parquet file, writes it to `data/raw/`, and uploads it to Hugging Face when asked. |
| Orchestrate | `pipeline.py` | `IngestionPipeline` runs the three steps for one year, a range of years, or the station catalog. |

`IngestionPipeline.ingest_trips_year` loops over months 1 to 12. A month that is not in the catalog is skipped, so a yearly file holds only the months that exist. `ingest_trips_range` skips a year with no trip files and continues with the next one.

### Parsing rules

- The source header is ignored. Columns are named by position, so a renamed header in a new month does not break the load.
- Null values use PyArrow's default tokens, such as an empty cell or `NA`. String columns can hold nulls.
- Timestamps are parsed as ISO-8601 first and `%d/%m/%Y %H:%M` second, because the source files use both formats.
- The cast to the target schema is strict. A value that does not fit its type fails the load instead of being silently coerced.

### Schemas

Column names are defined once in `mibici_e21/schemas/columns.py`. The types are in `schemas/trips.py` and `schemas/stations.py`.

**`TRIPS`**

| Column | Type | Meaning |
| --- | --- | --- |
| `viaje_id` | `int64` | Trip id |
| `usuario_id` | `int64` | User id |
| `genero` | `string` | Gender reported by the user |
| `año_de_nacimiento` | `int16` | User birth year |
| `inicio_del_viaje` | `timestamp[ms]` | Trip start time |
| `fin_del_viaje` | `timestamp[ms]` | Trip end time |
| `origen_id` | `int32` | Origin station id |
| `destino_id` | `int32` | Destination station id |

**`STATIONS`**

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | `int32` | Station id |
| `name` | `string` | Station name |
| `latitude` | `float64` | Latitude in decimal degrees |
| `longitude` | `float64` | Longitude in decimal degrees |
| `dpcapacity` | `int32` | Number of docks |

### Commands

```bash
make ingest-stations HF=false
make ingest-trips YEAR=2024 HF=true
make ingest-trips-range START=2015 END=2024 HF=true
```

`HF` must be `true` or `false`. With `HF=false`, files are written locally only.

---

## Storage and versioning

### Local layout

| Directory | Layer | Contents |
| --- | --- | --- |
| `data/temp/` | Bronze | Downloaded CSVs. Safe to delete with `make clean-temp`. |
| `data/raw/` | Silver | `mibici-trips-{year}.parquet` and `mibici-stations.parquet`. |
| `data/processed/` | Gold | `features_{year}.parquet`, `demand_aggregated_{year}.parquet`, `data_manifest_{year}.json`, and prediction files. |

Silver files are Parquet with PyArrow's default Snappy compression, one file per year.

> [!NOTE]
> `DATA_LAKE_SETTINGS.PROCESSED_DIR` points at `data/raw/`, not `data/processed/`. In the data lake settings, "processed" means the silver Parquet output of ingestion. Gold files use `FEATURE_STORE_SETTINGS.PROCESSED_DIR`, which is `data/processed/`.

### Hugging Face dataset

The Hugging Face dataset [`yako2792/mibici-dataset`](https://huggingface.co/datasets/yako2792/mibici-dataset) is the shared copy of the silver layer. Teammates download it instead of running the scraper.

- Uploads happen only with `HF=true` and need `HF_TOKEN` in a `.env` file at the project root.
- Every upload or deletion is a commit on the dataset, so Hugging Face keeps the version history.

### Data version

The dataset revision every clone trains on is pinned in `DATA_LAKE_SETTINGS.HF_DATASET_REVISION`. Changing it is a deliberate commit, so a newer snapshot never reaches training by accident.

`mlflow run . -e ingest` downloads the trips of `MODELING_SETTINGS.TRAINING_YEAR` and the station catalog from that revision, builds the gold files, and writes `data/processed/data_manifest_{year}.json`:

| Field | Content |
| --- | --- |
| `dataset_repo`, `dataset_revision` | Hugging Face dataset and the commit downloaded. |
| `year` | Training year. |
| `files` | For `stations`, `silver`, `gold_features`, and `gold_demand`: file name, SHA-256, size in bytes, and row count. |

The manifest has no timestamps, so the same revision always yields the same manifest. The ingest run logs it under `data/`, and every training run logs it again next to its model. Before training, `read_data_manifest` checks the SHA-256 of the gold demand file and stops if the file changed after the ingestion.

`make download-data` still downloads every Parquet file of the pinned revision for exploration, without a manifest.

---

## Processing: silver to gold

Code: `mibici_e21/features.py`, entry point `materialize_yearly_trip_features`.

For each year, the step reads `data/raw/mibici-trips-{year}.parquet`, cleans it, and writes two gold tables to `data/processed/`. A missing year stops the command with `FileNotFoundError`.

### 1. Cleaning

`_clean_and_filter_trips` removes rows the models cannot use:

1. Trips with a null origin, destination, start time, or end time.
2. Trips whose duration is outside the bounds in `FEATURE_STORE_SETTINGS`. A trip is kept when `0 < duration ≤ 480` minutes (`MIN_TRIP_DURATION_MINUTES` and `MAX_TRIP_DURATION_MINUTES`). This drops zero or negative durations and trips longer than 8 hours.

The log reports how many rows each rule dropped.

### 2. Trip features: `features_{year}.parquet`

One row per clean trip.

| Column | Source |
| --- | --- |
| `inicio_del_viaje`, `fin_del_viaje` | Silver start and end times |
| `origen_id`, `destino_id` | Silver station ids |
| `hour` | Hour of the start time |
| `day_of_week` | Weekday of the start time, Monday = 0 |
| `is_weekend` | 1 on Saturday and Sunday, else 0 |
| `duration_minutes` | End time minus start time, in minutes |

### 3. Station demand: `demand_aggregated_{year}.parquet`

One row per station, date, and hour. This is the table every model trains on.

| Column | Source |
| --- | --- |
| `date` | Date of the trip start |
| `hour` | Hour of the trip start |
| `station_id` | Station id |
| `departures_count` | Trips that started at the station in that hour, grouped by `origen_id` |
| `arrivals_count` | Trips that ended at the station, grouped by `destino_id` and bucketed by start hour |
| `net_flow` | `arrivals_count − departures_count` |
| `day_of_week` | Weekday of `date`, Monday = 0 |
| `is_weekend` | 1 on Saturday and Sunday, else 0 |

Departures and arrivals are joined with an outer merge. A station that has only departures or only arrivals in an hour gets 0 for the missing count. Station-hours with no trips at all have no row.

> [!NOTE]
> Arrivals are bucketed by the trip's start time, not its end time. For trips that cross an hour boundary, the arrival is counted in the hour the trip started.

### Commands

```bash
make materialize-yearly-features YEAR=2023
make materialize-features-range START=2015 END=2024
```

---

## Handoff to modeling

The models in `mibici_e21/modeling/` read `demand_aggregated_{year}.parquet`, and P3 also reads the station coordinates. Their data settings live in `MODELING_SETTINGS` in `mibici_e21/config.py`:

| Setting | Default | Use |
| --- | --- | --- |
| `TRAINING_YEAR` | `2023` | Year that `ingest` downloads and every model trains on. |
| `VAL_START_MONTH` | `8` | Validation starts on August 1. Earlier dates are training data. |
| `TEST_START_MONTH` | `10` | Test starts on October 1 and runs to the end of the year. |
| `BALANCE_THRESHOLD` | `5` | Absolute daily net flow that marks a station-day as `possible_empty` or `possible_full`. |
| `PEAK_HOURS` | `7, 8, 9, 17, 18, 19` | Hours counted as peak departures for station clustering. |
| `SEED` | `42` | Random seed of every estimator and sample. |

To run the whole flow (ingest, train, register, predict), follow [testing-e2e-flow.md](testing-e2e-flow.md).
