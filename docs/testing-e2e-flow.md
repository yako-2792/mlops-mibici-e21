# Testing e2e flow

Run these four commands in order, from the project root. They download the published Parquet files, build gold features for 2023 and 2024, train the baselines on 2023, and score 2024.

```bash
make download-data
make materialize-features-range START=2023 END=2024
make train YEAR=2023
make predict YEAR=2024 SAVE_CSV=true
```

The model is fit on 2023 and applied to 2024. P1 and P2 reuse the 2023 station-hour means and station classes. A station-hour pair that was not in training falls back to the global mean or the global class.

## 1. Download the silver files

```bash
make download-data
```

Copies every `*.parquet` file from the Hugging Face dataset `yako2792/mibici-dataset` into `data/raw/`, and writes `data/raw/data-version.txt` with the dataset commit.

Expect `data/raw/mibici-trips-2023.parquet` and `data/raw/mibici-trips-2024.parquet`. The log ends with `Dataset download complete`.

## 2. Build gold features

```bash
make materialize-features-range START=2023 END=2024
```

Reads each yearly trip file from `data/raw/` and writes trip features and station demand into `data/processed/`. Both years are included. A missing year stops the command.

Expect these files:

- `data/processed/features_2023.parquet`
- `data/processed/demand_aggregated_2023.parquet`
- `data/processed/features_2024.parquet`
- `data/processed/demand_aggregated_2024.parquet`

The log ends with `Feature range complete: 2 years`.

## 3. Train the baselines

```bash
make train YEAR=2023
```

Reads `demand_aggregated_2023.parquet` and logs one MLflow run per baseline under `mlruns/`, in the experiment `MiBici_Baselines_Equipo21`.

Expect three finished runs:

- `P1_Historical_Mean`, with MAE, RMSE, and R²
- `P2_Historical_Mode`, with macro precision, recall, and F1
- `P3_KMeans_3_Clusters`, with the silhouette score

Each run also stores its figure and its model artifact (`p1_model.parquet`, `p2_model.parquet`, or `p3_pipeline.pkl`). The log ends with `Baseline training complete for 2023`.

## 4. Score a later year

```bash
make predict YEAR=2024 SAVE_CSV=true
```

Reads `demand_aggregated_2024.parquet`, loads the latest P1, P2, and P3 artifacts from MLflow, and writes predictions into `data/processed/`. `SAVE_CSV=true` also writes a CSV copy of each file.

Expect these files:

- `data/processed/predictions_p1_2024.parquet` and `.csv`, with `predicted_departures`
- `data/processed/predictions_p2_2024.parquet` and `.csv`, with `predicted_risk`
- `data/processed/predictions_p3_2024.parquet` and `.csv`, with `predicted_cluster`

P1 and P2 have one row per gold demand row. P3 has one row per station. The log ends with `Inference complete for 2024`.
