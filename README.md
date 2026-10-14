# MLOps MiBici Equipo 21

<div align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />&nbsp;&nbsp;
  <img src="https://img.shields.io/badge/uv-DE5FE9?style=flat-square&logo=astral&logoColor=white" alt="uv" />&nbsp;&nbsp;
  <img src="https://img.shields.io/badge/CCDS-328F97?style=flat-square&logo=cookiecutter&logoColor=white" alt="CCDS" />&nbsp;&nbsp;
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn" />&nbsp;&nbsp;
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLflow" />
</div>

Proyecto de MLOps para el sistema de bicicletas compartidas MiBici - Equipo 21

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici_e21 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici_e21   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici_e21 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---

## Prerequisites and Installation

This project uses `uv` for Python package and environment management.

### 1. Install `uv`

* **macOS / Linux:**

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

* **Windows (PowerShell):**

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

* **Homebrew (macOS):**

  ```bash
  brew install uv
  ```

* **Python | pip:**

  ```bash
  pip install uv
  ```

### 2. Project Setup

1. Clone the repository

   ```bash
   git clone {url}
   cd mlops-mibici-e21
   ```

2. Create environment and sync dependencies

   ```bash
   uv sync
   ```

### 3. Development Commands

* Run linter (Ruff):

  ```bash
  uv run ruff check .
  ```

* Format code:

  ```bash
  uv run ruff format .
  ```

* Run main script:

  ```bash
  uv run main.py
  ```

## Git and GitHub Workflow (GitHub Flow)

### Collaboration Rules

1. Protected main branch: Direct pushes to main are disabled.
2. Pull Requests (PR): All changes enter via PR targeting main.
3. Mandatory PR requirements:
   * Each member must open at least 1 Pull Request.
   * Each member must review and approve at least 1 Pull Request from a teammate.
   * Merging requires at least 1 approval and passing linter checks.

### Branching Naming Convention

Branches created from `main` using the following prefixes:

* `feat/<short-description>`: New features or models (e.g., feat/ingestion-pipeline-p1).

* `fix/<short-description>`: Bug fixes (e.g., fix/config-paths).

* `docs/<short-description>`: Documentation updates (e.g., docs/ml-canvas-update).

* `refactor/<short-description>`: Code refactoring without functional changes.

### Commit Message Convention

Format:

```text
<type>: <description in lowercase>
```

Allowed types:

* `feat`: New feature.

* `fix`: Bug fix.

* `docs`: Documentation changes.

* `style`: Formatting, missing semicolons, whitespace.

* `refactor`: Code refactoring.

* `chore`: Maintenance, dependency updates, `.gitignore`.

### Workflow Steps

1. Update main branch:

   ```bash
   git checkout main
   git pull origin main
   ```

2. Create a feature branch:

   ```bash
   git checkout -b feature/my-feature
   ```

3. Commit changes:

   ```bash
   git add .
   git commit -m "feat(p1): implement baseline model for question 1"
   ```

4. Push branch:

   ```bash
   git push origin feat/my-feature
   ```

5. Open Pull Request on GitHub and request review.

6. Merge after approval and delete remote/local feature branch (Squash and Merge).
