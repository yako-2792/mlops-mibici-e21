# MLOps MiBici — Equipo 21

Demand forecasting, station rebalancing, and clustering for the MiBici bike-sharing system, built on a data-lake ingestion pipeline.

The pipeline downloads the public monthly catalogs, casts them to a strict PyArrow schema, and publishes Parquet files locally and, when requested, to the Hugging Face dataset.

| ID | Name |
| --- | --- |
| A0173318 | Berenice Chapuli Xochimitl |
| A01702737 | Walter Gonzalo Fernandez Deza |
| A01374964 | Jesús Zabdiel Sánchez Chávez |
| A01639802 | Alejandro Gil Ruiz |

<div align="left">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+" />&nbsp;
  <img src="https://img.shields.io/badge/uv-DE5FE9?style=flat-square&logo=astral&logoColor=white" alt="uv" />&nbsp;
  <img src="https://img.shields.io/badge/PyArrow-FF6B00?style=flat-square&logo=apachearrow&logoColor=white" alt="PyArrow" />&nbsp;
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face" />&nbsp;
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLflow" />&nbsp;
  <img src="https://img.shields.io/badge/Ruff-D7FF64?style=flat-square&logo=ruff&logoColor=black" alt="Ruff" />&nbsp;
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
</div>

---

## Contents

- [Data architecture and pipeline](#data-architecture-and-pipeline)
- [Quick start](#quick-start)
- [Makefile](#makefile)
- [Project organization](#project-organization)
- [Git and GitHub workflow](#git-and-github-workflow)

---

## Data architecture and pipeline

MiBici publishes one CSV per month. The file id in each URL comes from the catalog page, so the extractor reads that page instead of building the URL from the date.

| Layer | Path | What happens |
| --- | --- | --- |
| Bronze | `data/temp/` | `extractor.py` downloads the trip and station CSVs. Months missing from the catalog are skipped. Trip files are saved as `datos_abiertos_YYYY_MM.csv`. |
| Silver | `data/raw/` | `transformer.py` drops the original header, names columns by position, and casts them with PyArrow. Null tokens such as `NA` stay null. Timestamps are accepted as ISO-8601 or `%d/%m/%Y %H:%M`. `publisher.py` writes `mibici-trips-{year}.parquet` and `mibici-stations.parquet`. That folder is `DATA_LAKE_SETTINGS.PROCESSED_DIR`. |
| Gold | `data/processed/` | `features.py` reads the yearly trip Parquet from `data/raw/` and writes `features_{year}.parquet` and `demand_aggregated_{year}.parquet`. |
| Share | Hugging Face | `pipeline.py` uploads the silver Parquet files when `HF=true`. The dataset is `yako2792/mibici-dataset`. Hugging Face versions the data: each upload or deletion is a commit on that dataset. |

`make download-data` copies the published Parquet files back into `data/raw/` without running the scraper. The version history of those files lives in the Hugging Face dataset commits.

---

## Quick start

> [!NOTE]
> You need Python 3.11 and [`uv`](https://docs.astral.sh/uv/). On macOS, `brew install uv` is enough. `uv sync` creates the environment and installs the project.

```bash
git clone <repository-url>
cd mlops-mibici-e21
uv sync
```

> [!IMPORTANT]
> Uploading to Hugging Face reads `HF_TOKEN` from a `.env` file in the project root. `.env` is gitignored. The dataset id is `HF_DATASET_REPO` in `mibici_e21/config.py` (`yako2792/mibici-dataset`), not an environment variable.
>
> ```bash
> HF_TOKEN=hf_your_token_here
> ```

> [!TIP]
> `make help` lists every command. Ingestion writes Parquet locally. `HF` defaults to `false`, so a run does not upload until you pass `HF=true`.

---

## Makefile

### Ingestion

`HF` must be `true` or `false`. `YEAR` is required for one year. `START` and `END` are required for a range, and both ends are included.

| Command | What it does |
| --- | --- |
| `make ingest-stations HF=false` | Download the station catalog and write `mibici-stations.parquet`. |
| `make ingest-trips YEAR=2024 HF=true` | Ingest one year of trips and upload `mibici-trips-2024.parquet`. |
| `make ingest-trips-range START=2015 END=2024 HF=true` | Ingest every year in the range, including both ends, and upload each Parquet file. |

```bash
make ingest-stations HF=false
make ingest-trips YEAR=2024 HF=true
make ingest-trips-range START=2015 END=2024 HF=true
```

### Dataset consumption

| Command | What it does |
| --- | --- |
| `make download-data` | Download the dataset `*.parquet` files from Hugging Face into `data/raw/`. |

```bash
make download-data
```

### Gold features

These commands read `mibici-trips-{year}.parquet` from `data/raw/` and write `features_{year}.parquet` and `demand_aggregated_{year}.parquet` into `data/processed/`. `YEAR` is required for one year. `START` and `END` are required for a range, and both ends are included. A missing year stops the command.

| Command | What it does |
| --- | --- |
| `make materialize-yearly-features YEAR=2023` | Build trip features and station demand for one year. |
| `make materialize-features-range START=2015 END=2024` | Build trip features and station demand for every year in the range. |

```bash
make materialize-yearly-features YEAR=2023
make materialize-features-range START=2015 END=2024
```

### Maintenance

`make clean` deletes compiled Python files. The data commands below are Python, so they behave the same on macOS, Linux, and Windows. They change local files only.

| Command | What it does |
| --- | --- |
| `make clean-temp` | Delete the contents of `data/temp/` and keep the directory. |
| `make clean-raw` | Delete the contents of `data/raw/` and keep the directory. |
| `make clean-interim` | Delete the contents of `data/interim/` and keep the directory. |
| `make clean-processed` | Delete the contents of `data/processed/` and keep the directory. |
| `make clean-external` | Delete the contents of `data/external/` and keep the directory. |
| `make clean-all` | Run every data-directory clean above. |
| `make clean` | Delete `__pycache__` and `*.pyc` files. |

```bash
make clean-temp
make clean-raw
make clean-interim
make clean-processed
make clean-external
make clean-all
```

### Code quality

| Command | What it does |
| --- | --- |
| `make lint` | Check formatting and Ruff rules. |
| `make format` | Apply Ruff fixes and format the code. |
| `make test` | Run the unit tests with pytest. |

```bash
make lint
make format
make test
```

### Setup

| Command | What it does |
| --- | --- |
| `make requirements` | Install dependencies with `uv sync`. |
| `make create_environment` | Create a virtual environment for Python 3.11. |

---

## Project organization

```text
├── Makefile                 <- ingest, download, features, train, predict, clean, lint, and test
├── README.md
├── pyproject.toml           <- package metadata and Ruff configuration
├── data
│   ├── temp                 <- Bronze: CSVs downloaded by the extractor
│   ├── raw                  <- Silver: data lake Parquet files
│   ├── processed            <- Gold: trip features, station demand, and predictions
│   ├── interim
│   └── external
└── mibici_e21
    ├── config.py            <- paths, Hugging Face repo, and HF_TOKEN
    ├── dataset.py           <- download Parquet from Hugging Face and clean local data
    ├── schemas
    │   ├── columns.py       <- trip and station column names
    │   ├── trips.py         <- TRIPS PyArrow schema
    │   └── stations.py      <- STATIONS PyArrow schema
    └── ingestion
        ├── extractor.py     <- read the MiBici catalog and download CSVs
        ├── transformer.py   <- positional names and PyArrow casts
        ├── publisher.py     <- write Parquet and upload to Hugging Face
        └── pipeline.py      <- year, range, and station ingestion
```

---

## Git and GitHub workflow

### Collaboration rules

1. `main` is protected. Direct pushes are disabled.
2. Every change enters through a pull request targeting `main`.
3. Each member opens at least one pull request and reviews at least one teammate's pull request.
4. Merging requires at least one approval and a passing linter.

### Branch names

Create branches from `main` with these prefixes:

- `feat/<short-description>` adds a feature or model, for example `feat/ingestion-pipeline`.
- `fix/<short-description>` fixes a bug, for example `fix/config-paths`.
- `docs/<short-description>` updates documentation, for example `docs/readme-makefile`.
- `refactor/<short-description>` restructures code without changing behavior.

### Commit messages

```text
<type>: <description in lowercase>
```

| Type | Use |
| --- | --- |
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation |
| `style` | Formatting and whitespace |
| `refactor` | Restructuring without a behavior change |
| `chore` | Maintenance, dependencies, and `.gitignore` |

### Steps

```bash
git checkout main
git pull origin main
git checkout -b feat/my-feature
git add .
git commit -m "feat: add trip and station ingestion pipeline"
git push origin feat/my-feature
```

Open a pull request against `main`. After approval, squash-merge and delete the feature branch.
