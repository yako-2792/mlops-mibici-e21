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
- [MLflow Tracking & Reproducibility (MLproject)](#mlflow-tracking--reproducibility-mlproject)
- [Makefile](#makefile)
- [Project organization](#project-organization)
- [Git and GitHub workflow](#git-and-github-workflow)

---

## Data architecture and pipeline

The pipeline downloads the monthly CSVs from the MiBici open-data portal (bronze, `data/temp/`), casts them to strict PyArrow schemas and publishes yearly Parquet files (silver, `data/raw/`, shared on Hugging Face), then cleans and aggregates them into the station-demand tables the baselines train on (gold, `data/processed/`).

| Document | What it covers |
| --- | --- |
| [docs/data-pipeline.md](docs/data-pipeline.md) | Data source, ingestion and schemas, storage and versioning, cleaning and gold features, and the handoff to the baselines. |
| [docs/testing-e2e-flow.md](docs/testing-e2e-flow.md) | Commands to download the data, build features, train, and predict end to end. |

---

## Quick start

> [!NOTE]
> You need Python 3.11 and [`uv`](https://docs.astral.sh/uv/). On macOS, `brew install uv` is enough. `uv sync` creates the environment and installs the project.

```bash
git clone <repository-url>
cd mlops-mibici-e21
uv sync
source .venv/bin/activate
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
## MLflow Tracking & Reproducibility
To guarantee experiment traceability and enable the Model Registry, we use a local tracking server backed by a SQLite database (`mlflow.db`). Both `mlflow.db` and `mlartifacts/` are excluded in `.gitignore`.

In a dedicated terminal, start the local server supporting SQLite and custom artifact storage:

```bash
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 5000
```

In the terminal, point `MLFLOW_TRACKING_URI` to the local server:

```bash
export MLFLOW_TRACKING_URI=[http://127.0.0.1:5000](http://127.0.0.1:5000)
```

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Datos
mlflow run . -e ingest --env-manager=local

# Baselines
mlflow run . -e train -P question=p1 -P model=baseline --env-manager=local
mlflow run . -e train -P question=p2 -P model=baseline --env-manager=local
mlflow run . -e train -P question=p3 -P model=baseline --env-manager=local

# P1: tres candidatas (la tercera sale de una búsqueda con nested runs)
mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=100 -P max_depth=10 --env-manager=local
mlflow run . -e train -P question=p1 -P model=hgb -P n_estimators=200 -P max_depth=8 --env-manager=local
mlflow run . -e train -P question=p1 -P model=rf -P search=true --env-manager=local

# P2: tres candidatas
mlflow run . -e train -P question=p2 -P model=rf -P n_estimators=100 -P max_depth=10 --env-manager=local
mlflow run . -e train -P question=p2 -P model=hgb -P n_estimators=200 -P max_depth=8 --env-manager=local
mlflow run . -e train -P question=p2 -P model=rf -P search=true --env-manager=local

# P3: tres candidatas
mlflow run . -e train -P question=p3 -P model=kmeans -P n_clusters=4 --env-manager=local
mlflow run . -e train -P question=p3 -P model=kmeans -P n_clusters=5 --env-manager=local
mlflow run . -e train -P question=p3 -P search=true --env-manager=local

# Model Registry: registra la mejor candidata de cada pregunta con el alias champion
mlflow run . -e register -P question=p1 --env-manager=local
mlflow run . -e register -P question=p2 --env-manager=local
mlflow run . -e register -P question=p3 --env-manager=local
```

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
├── Makefile                <- ingest, download, features, train, predict, clean, lint, and test
├── MLproject               <- MLflow Project entry points (ingest, train) and parameters
├── python_env.yaml         <- pinned Python environment for MLflow reproducibility
├── README.md
├── pyproject.toml          <- package metadata and Ruff configuration
├── data
│   ├── temp                <- Bronze: CSVs downloaded by the extractor
│   ├── raw                 <- Silver: data lake Parquet files
│   ├── processed           <- Gold: trip features, station demand, and predictions
│   ├── interim
│   └── external
├── docs
│   ├── data-pipeline.md    <- data source, ingestion, storage, and gold processing
│   ├── testing-e2e-flow.md <- end-to-end download, features, train, and predict
│   └── Machine Learning Canvas.pdf
├── reports
│   └── figures             <- generated EDA plots (demanda_hora.png, top_estaciones.png, total_dias.png)
└── mibici_e21
    ├── config.py           <- paths, Hugging Face repo, HF_TOKEN, and MLflow/modeling settings
    ├── dataset.py          <- download Parquet from Hugging Face and clean local data
    ├── features.py         <- clean silver trips and build gold features
    ├── plots.py            <- EDA figures into reports/figures
    ├── modeling
    │   ├── train.py        <- train baselines and candidate models, log to MLflow & Registry
    │   └── predict.py      <- batch inference for P1, P2, and P3
    ├── schemas
    │   ├── columns.py      <- trip and station column names
    │   ├── trips.py        <- TRIPS PyArrow schema
    │   └── stations.py     <- STATIONS PyArrow schema
    └── ingestion
        ├── extractor.py    <- read the MiBici catalog and download CSVs
        ├── transformer.py  <- positional names and PyArrow casts
        ├── publisher.py    <- write Parquet and upload to Hugging Face
        └── pipeline.py     <- year, range, and station ingestion
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
