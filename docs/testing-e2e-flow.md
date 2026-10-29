# Testing the end-to-end flow

These steps go from a clean clone to batch predictions served by the production models. Run them from the project root, in the order shown.

## 1. Start the tracking server

In a dedicated terminal:

```bash
make mlflow-server
```

The server stores runs and the Model Registry in `mlflow.db` and artifacts in `mlartifacts/`. Both are gitignored.

## 2. Point the shell at it

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=mibici-equipo21
```

`mlflow run` creates its run in `MLFLOW_EXPERIMENT_NAME` before the code starts. Without the export, the run lands in `Default` and the code stops with a message that names the export.

## 3. Ingest the training year

```bash
mlflow run . -e ingest --env-manager uv
```

Expect `data/raw/mibici-trips-2023.parquet`, `data/raw/mibici-stations.parquet`, the two gold files in `data/processed/`, and `data/processed/data_manifest_2023.json`. The run `ingest_2023` logs the manifest under `data/`.

## 4. Train a baseline and a candidate

```bash
mlflow run . -e train -P question=p1 -P model=baseline --env-manager uv
mlflow run . -e train -P question=p1 -P model=hgb -P n_estimators=400 -P max_depth=14 --env-manager uv
```

Each command logs one run named after its model and hyperparameters, with `plots/`, `data/`, `config/`, and the model with its signature. A candidate without its hyperparameters stops with the `-P` flags it needs.

## 5. Promote the champion

```bash
mlflow run . -e register -P question=p1 --env-manager uv
```

The log ends with `models:/mibici-equipo21-p1-demanda@champion loads and scores 5 example rows`. Registration stops when no candidate beats the baseline on the validation metric.

Repeat steps 4 and 5 with `question=p2` and `question=p3`. The README lists the commands that reproduce the chosen model of each question.

## 6. Score a later year

Prediction reads the gold files of the year it scores, so build them first:

```bash
make download-data
make materialize-yearly-features YEAR=2024
make predict YEAR=2024 SAVE_CSV=true
```

`predict` loads the `@champion` version of each question and writes `predictions_p1_2024.parquet` (`predicted_departures`), `predictions_p2_2024.parquet` (`predicted_risk`), and `predictions_p3_2024.parquet` (`predicted_group`) to `data/processed/`. `SAVE_CSV=true` also writes CSV copies.
