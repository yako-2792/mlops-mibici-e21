"""Data drift of the P1 inputs and target between the reference and the current period.

This is the `drift` entry point of the MLflow Project:

    mlflow run . -e drift

Evidently compares the gold demand the P1 champion was fitted on (`REFERENCE`) with a later
period (`CURRENT`). The methods are fixed, so a column drifts for the same reason at any
sample size: the normed Wasserstein distance for numeric columns and the Jensen-Shannon
distance for categorical ones, both against `DRIFT_THRESHOLD`. Evidently would otherwise
switch to p-values below 1,000 rows.

The run writes, under `reports/monitoring/`:

- `p1_drift_report.html` and `.json`: the Evidently report, drift per column and data quality.
- `p1_drift_scores.csv`: one row per column with its method, score, and verdict.
- `p1_drift_by_quarter.csv` and `.png`: the same scores for each quarter of the current
  period, plus the share of rows from stations the reference never saw.
- `p1_drift_summary.json`: the numbers above in one file.

Everything is also logged to one MLflow run tagged `stage_of_work=monitoring`.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from mibici_e21.config import MONITORING_SETTINGS
from mibici_e21.monitoring.periods import (
    CURRENT,
    REFERENCE,
    Period,
    load_period,
    unseen_station_share,
)
from mibici_e21.schemas.columns import GoldColumns
from mibici_e21.tracking import base_tags, configure_tracking

NUMERIC_COLUMNS = (GoldColumns.HOUR, GoldColumns.DEPARTURES_COUNT)
CATEGORICAL_COLUMNS = (GoldColumns.STATION_ID, GoldColumns.DAY_OF_WEEK, GoldColumns.IS_WEEKEND)
COLUMNS = (*CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS)
NUMERIC_METHOD = "wasserstein"
CATEGORICAL_METHOD = "jensenshannon"
VALUE_DRIFT_TYPE = "evidently:metric_v2:ValueDrift"
REPORT_PREFIX = "p1_drift"


@dataclass(frozen=True)
class DriftResult:
    """Outcome of one drift comparison.

    - `scores`: one row per column with `column`, `kind`, `method`, `score`, `threshold`,
      and `drifted`, sorted by score, highest first.
    - `unseen_station_share`: share of current rows from stations absent in the reference.
    - `reference_rows`, `current_rows`: rows compared on each side.
    """

    scores: pd.DataFrame
    unseen_station_share: float
    reference_rows: int
    current_rows: int

    @property
    def drifted_columns(self) -> list[str]:
        return self.scores.loc[self.scores["drifted"], "column"].tolist()


def data_definition() -> DataDefinition:
    """Declare which P1 columns Evidently treats as numeric and which as categorical.

    `station_id` is an identifier, so its drift is a change in the mix of stations, not in
    a numeric scale.
    """
    return DataDefinition(
        numerical_columns=[str(column) for column in NUMERIC_COLUMNS],
        categorical_columns=[str(column) for column in CATEGORICAL_COLUMNS],
    )


def drift_preset(threshold: float = MONITORING_SETTINGS.DRIFT_THRESHOLD) -> DataDriftPreset:
    """Return the drift preset with the methods and threshold fixed for every column."""
    return DataDriftPreset(
        columns=[str(column) for column in COLUMNS],
        num_method=NUMERIC_METHOD,
        cat_method=CATEGORICAL_METHOD,
        threshold=threshold,
    )


def run_report(reference: pd.DataFrame, current: pd.DataFrame, with_summary: bool = True):
    """Run the Evidently report of `current` against `reference` and return its snapshot.

    - `with_summary`: also add `DataSummaryPreset`, the data quality part of the report.
    """
    metrics = [drift_preset()]
    if with_summary:
        metrics.append(DataSummaryPreset(columns=[str(column) for column in COLUMNS]))
    definition = data_definition()
    columns = [str(column) for column in COLUMNS]
    return Report(metrics, include_tests=True).run(
        Dataset.from_pandas(current[columns], data_definition=definition),
        Dataset.from_pandas(reference[columns], data_definition=definition),
    )


def drift_scores(snapshot) -> pd.DataFrame:
    """Turn the `ValueDrift` metrics of a snapshot into one row per column.

    A column drifts when its distance reaches the threshold, the rule Evidently applies to
    both distance methods.
    """
    rows = []
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if config["type"] != VALUE_DRIFT_TYPE:
            continue
        column = config["column"]
        score = float(metric["value"])
        rows.append(
            {
                "column": column,
                "kind": "numeric" if column in NUMERIC_COLUMNS else "categorical",
                "method": config["method"],
                "score": score,
                "threshold": float(config["threshold"]),
                "drifted": score >= float(config["threshold"]),
            }
        )
    scores = pd.DataFrame(rows)
    return scores.sort_values("score", ascending=False, ignore_index=True)


def compare(reference: pd.DataFrame, current: pd.DataFrame, with_summary: bool = True):
    """Compare two frames and return `(snapshot, DriftResult)`."""
    snapshot = run_report(reference, current, with_summary=with_summary)
    result = DriftResult(
        scores=drift_scores(snapshot),
        unseen_station_share=unseen_station_share(reference, current),
        reference_rows=len(reference),
        current_rows=len(current),
    )
    return snapshot, result


def drift_by_period(
    reference: pd.DataFrame, current: pd.DataFrame, periods: list[Period]
) -> pd.DataFrame:
    """Compare each sub-period of the current data against the whole reference.

    Returns one row per period with its drift score per column, the share of rows from
    unseen stations, the mean departures per row, and the rows per day.
    """
    rows = []
    for period in periods:
        part = current[period.contains(current[GoldColumns.DATE])]
        if part.empty:
            continue
        _, result = compare(reference, part, with_summary=False)
        scores = dict(zip(result.scores["column"], result.scores["score"], strict=True))
        rows.append(
            {
                "period": period.name,
                "start": period.start.isoformat(),
                "end": period.last_day.isoformat(),
                "rows": len(part),
                "rows_per_day": len(part) / part[GoldColumns.DATE].nunique(),
                "mean_departures": float(part[GoldColumns.DEPARTURES_COUNT].mean()),
                "unseen_station_share": result.unseen_station_share,
                "drifted_columns": len(result.drifted_columns),
                **{f"drift_{column}": scores[str(column)] for column in COLUMNS},
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    snapshot, result: DriftResult, by_quarter: pd.DataFrame, output_dir: Path
) -> dict[str, Path]:
    """Write the report, tables, figure, and summary. Returns each file by its role."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "html": output_dir / f"{REPORT_PREFIX}_report.html",
        "json": output_dir / f"{REPORT_PREFIX}_report.json",
        "scores": output_dir / f"{REPORT_PREFIX}_scores.csv",
        "by_quarter": output_dir / f"{REPORT_PREFIX}_by_quarter.csv",
        "by_quarter_figure": output_dir / f"{REPORT_PREFIX}_by_quarter.png",
        "summary": output_dir / f"{REPORT_PREFIX}_summary.json",
    }
    snapshot.save_html(str(paths["html"]))
    snapshot.save_json(str(paths["json"]))
    result.scores.to_csv(paths["scores"], index=False)
    by_quarter.to_csv(paths["by_quarter"], index=False)
    figure = drift_by_quarter_figure(by_quarter)
    figure.savefig(paths["by_quarter_figure"], dpi=150, bbox_inches="tight")
    plt.close(figure)
    paths["summary"].write_text(json.dumps(summary(result), indent=2) + "\n", encoding="utf-8")
    return paths


def summary(result: DriftResult) -> dict:
    """Return the headline numbers of a drift comparison, ready to serialize."""
    return {
        "reference_period": REFERENCE.label,
        "current_period": CURRENT.label,
        "reference_rows": result.reference_rows,
        "current_rows": result.current_rows,
        "threshold": MONITORING_SETTINGS.DRIFT_THRESHOLD,
        "drifted_columns": result.drifted_columns,
        "unseen_station_share": round(result.unseen_station_share, 4),
        "scores": {
            row.column: round(row.score, 4) for row in result.scores.itertuples(index=False)
        },
    }


def drift_by_quarter_figure(by_quarter: pd.DataFrame) -> Figure:
    """Draw the drift score of each column per quarter, next to the unseen station share."""
    figure, (scores_ax, unseen_ax) = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]}
    )
    for column in COLUMNS:
        scores_ax.plot(
            by_quarter["period"], by_quarter[f"drift_{column}"], marker="o", label=str(column)
        )
    scores_ax.axhline(
        MONITORING_SETTINGS.DRIFT_THRESHOLD, color="black", linestyle="--", linewidth=1
    )
    scores_ax.text(
        0, MONITORING_SETTINGS.DRIFT_THRESHOLD, " drift threshold", va="bottom", fontsize=8
    )
    scores_ax.set_title("Drift score against the reference")
    scores_ax.set_ylabel("Distance (Wasserstein or Jensen-Shannon)")
    scores_ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))

    bars = unseen_ax.bar(by_quarter["period"], 100 * by_quarter["unseen_station_share"])
    unseen_ax.bar_label(bars, fmt="%.1f%%", fontsize=8)
    unseen_ax.set_title("Rows from stations unseen in training")
    unseen_ax.set_ylabel("% of rows")
    for ax in (scores_ax, unseen_ax):
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def main() -> None:
    configure_tracking()
    reference, reference_manifests = load_period(REFERENCE)
    current, current_manifests = load_period(CURRENT)
    logger.info(f"Reference {REFERENCE.label}: {len(reference)} rows")
    logger.info(f"Current {CURRENT.label}: {len(current)} rows")

    snapshot, result = compare(reference, current)
    by_quarter = drift_by_period(reference, current, CURRENT.quarters())
    paths = write_outputs(snapshot, result, by_quarter, MONITORING_SETTINGS.REPORTS_DIR)

    with mlflow.start_run(run_name=f"P1_drift_{CURRENT.start.year}"):
        mlflow.set_tags(
            {
                **base_tags("monitoring"),
                "question": "p1",
                "monitoring": "data_drift",
                "dataset_revision": current_manifests[0]["dataset_revision"],
                "drifted_columns": ",".join(result.drifted_columns) or "none",
            }
        )
        mlflow.log_params(
            {
                "reference_period": REFERENCE.label,
                "current_period": CURRENT.label,
                "reference_rows": result.reference_rows,
                "current_rows": result.current_rows,
                "drift_threshold": MONITORING_SETTINGS.DRIFT_THRESHOLD,
                "numeric_method": NUMERIC_METHOD,
                "categorical_method": CATEGORICAL_METHOD,
            }
        )
        mlflow.log_metrics(
            {
                **{f"drift_{row.column}": row.score for row in result.scores.itertuples()},
                "drifted_columns": len(result.drifted_columns),
                "unseen_station_share": result.unseen_station_share,
            }
        )
        for manifest in (*reference_manifests, *current_manifests):
            mlflow.log_dict(manifest, f"data/data_manifest_{manifest['year']}.json")
        for path in paths.values():
            mlflow.log_artifact(str(path), artifact_path="drift")

    drifted = ", ".join(result.drifted_columns) or "none"
    logger.info(f"Drifted columns: {drifted}. Report in {paths['html']}")


if __name__ == "__main__":
    main()
