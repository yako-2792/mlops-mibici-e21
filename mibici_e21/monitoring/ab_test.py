"""Compare the P1 champion and challenger on the same unseen period: an offline A/B test.

This is the `ab_test` entry point of the MLflow Project:

    mlflow run . -e ab_test
    mlflow run . -e ab_test -P promote=true

Version A is `@champion` and version B is `@challenger`. Both are loaded from the registry
and score every row of `EVALUATION`, a period neither of them trained on. The test reports:

1. MAE, RMSE, and R2 of each version, overall and by segment: stations the champion knew
   or not, weekdays or weekends, and peak or off-peak hours.
2. A paired bootstrap over days of the MAE difference, B minus A, with its interval.
3. A simulated traffic split: each station is assigned to A or B by a hash of its id, as a
   router in front of the API would do, and each group is served by its version.
4. Prediction time and model size, as secondary criteria.

The decision rule is fixed before the test runs. B wins only when the whole interval of the
MAE difference is below zero and B is not worse than A on the known stations by more than
`KNOWN_STATION_TOLERANCE`. With `promote=true`, a winning B takes `champion` and A becomes
`challenger`. Without it, the aliases stay and the decision is only recorded.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
import mlflow.artifacts
import mlflow.pyfunc
import numpy as np
import pandas as pd

from mibici_e21.config import MODELING_SETTINGS, MONITORING_SETTINGS, REGISTRY_SETTINGS
from mibici_e21.modeling.base import SupervisedQuestion
from mibici_e21.modeling.demand import DemandQuestion
from mibici_e21.modeling.train import parse_bool
from mibici_e21.monitoring.periods import EVALUATION, REFERENCE, load_period
from mibici_e21.schemas.columns import GoldColumns
from mibici_e21.tracking import base_tags, configure_tracking

STAGE = "ab_test"
CHAMPION = REGISTRY_SETTINGS.CHAMPION_ALIAS
CHALLENGER = REGISTRY_SETTINGS.CHALLENGER_ALIAS
REPORT_PREFIX = "p1_ab_test"
METRICS = ("MAE", "RMSE", "R2")


@dataclass(frozen=True)
class Arm:
    """One side of the test: a registry version reached through an alias.

    - `label`: `A` or `B`.
    - `alias`: alias the version was loaded through.
    - `version`: registry version number.
    - `run_id`: run that logged the model.
    - `uri`: URI the model was loaded from, pinned to the version.
    """

    label: str
    alias: str
    version: str
    run_id: str
    uri: str

    @property
    def name(self) -> str:
        return f"{self.label} (v{self.version} @{self.alias})"


@dataclass(frozen=True)
class Bootstrap:
    """Paired bootstrap of the MAE difference, B minus A. Negative means B errs less.

    - `delta`: difference on the whole period.
    - `low`, `high`: ends of the interval at `confidence`.
    - `share_below_zero`: share of resamples where B beats A.
    - `samples`, `confidence`, `days`: how the interval was built.
    """

    delta: float
    low: float
    high: float
    share_below_zero: float
    samples: int
    confidence: float
    days: int


@dataclass(frozen=True)
class Decision:
    """Outcome of the decision rule.

    - `winner`: `B` when the challenger should become champion, otherwise `A`.
    - `reasons`: each condition of the rule with its value, in plain words.
    """

    winner: str
    reasons: list[str]


def load_arm(question: SupervisedQuestion, alias: str, label: str) -> Arm:
    """Resolve an alias to its version, so both arms are pinned for the whole test."""
    name = question.registered_model_name
    version = MlflowClient().get_model_version_by_alias(name, alias)
    number = str(version.version)
    return Arm(label, alias, number, version.run_id, f"models:/{name}/{number}")


def segments(data: pd.DataFrame, known_stations: np.ndarray) -> dict[str, pd.Series]:
    """Return the row mask of every segment the metrics are broken down by."""
    known = data[GoldColumns.STATION_ID].isin(known_stations)
    weekend = data[GoldColumns.IS_WEEKEND].astype(bool)
    peak = data[GoldColumns.HOUR].isin(MODELING_SETTINGS.PEAK_HOURS)
    return {
        "all": pd.Series(True, index=data.index),
        "known_stations": known,
        "new_stations": ~known,
        "weekday": ~weekend,
        "weekend": weekend,
        "peak_hours": peak,
        "off_peak_hours": ~peak,
    }


def metrics_table(
    question: SupervisedQuestion,
    data: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    masks: dict[str, pd.Series],
) -> pd.DataFrame:
    """Score both arms on every segment.

    Returns one row per segment and metric with the value of A, of B, and `delta` = B - A.
    For MAE and RMSE a negative delta favors B; for R2 a positive one does.
    """
    target = data[question.target_column].to_numpy()
    rows = []
    for segment, mask in masks.items():
        selected = mask.to_numpy()
        if not selected.any():
            continue
        values = {
            label: question.metrics(target[selected], pred[selected], "m")
            for label, pred in predictions.items()
        }
        for metric in METRICS:
            a, b = values["A"][f"m_{metric}"], values["B"][f"m_{metric}"]
            rows.append(
                {
                    "segment": segment,
                    "rows": int(selected.sum()),
                    "metric": metric,
                    "A": a,
                    "B": b,
                    "delta": b - a,
                }
            )
    return pd.DataFrame(rows)


def paired_bootstrap(
    dates: pd.Series,
    abs_error_a: np.ndarray,
    abs_error_b: np.ndarray,
    samples: int = MONITORING_SETTINGS.BOOTSTRAP_SAMPLES,
    confidence: float = MONITORING_SETTINGS.CONFIDENCE,
    seed: int = MODELING_SETTINGS.SEED,
) -> tuple[Bootstrap, np.ndarray]:
    """Resample whole days with replacement and recompute the MAE difference each time.

    Days are the unit because the rows of one day share weather, events, and outages, so
    resampling rows would understate the uncertainty. Both arms are scored on the same
    resampled days, which is what makes the test paired. Returns the summary and every
    resampled difference.
    """
    daily = (
        pd.DataFrame({"date": dates.to_numpy(), "a": abs_error_a, "b": abs_error_b})
        .groupby("date")
        .agg(a=("a", "sum"), b=("b", "sum"), rows=("a", "size"))
    )
    a, b, rows = (daily[column].to_numpy() for column in ("a", "b", "rows"))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(daily), size=(samples, len(daily)))
    resampled_rows = rows[picks].sum(axis=1)
    deltas = (b[picks].sum(axis=1) - a[picks].sum(axis=1)) / resampled_rows
    tail = (1 - confidence) / 2
    low, high = np.quantile(deltas, [tail, 1 - tail])
    summary = Bootstrap(
        delta=float((b.sum() - a.sum()) / rows.sum()),
        low=float(low),
        high=float(high),
        share_below_zero=float((deltas < 0).mean()),
        samples=samples,
        confidence=confidence,
        days=len(daily),
    )
    return summary, deltas


def traffic_group(station_ids: pd.Series) -> pd.Series:
    """Assign each station to `A` or `B` by a stable hash of its id.

    The hash does not depend on the run or the Python process, so a station always lands in
    the same group, as a router in front of the API would keep it.
    """
    groups = {
        station: "A" if hashlib.sha256(str(station).encode()).digest()[0] % 2 == 0 else "B"
        for station in station_ids.unique()
    }
    return station_ids.map(groups)


def traffic_split_table(
    data: pd.DataFrame,
    abs_errors: dict[str, np.ndarray],
    groups: pd.Series,
    known_stations: np.ndarray,
) -> pd.DataFrame:
    """Score the simulated split: each group is served by its own arm.

    Both arms are also scored on both groups, so the table shows whether a gap comes from
    the version or from the stations that landed in each group.
    """
    known = data[GoldColumns.STATION_ID].isin(known_stations)
    rows = []
    for group in ("A", "B"):
        mask = (groups == group).to_numpy()
        stations = data.loc[mask, GoldColumns.STATION_ID]
        rows.append(
            {
                "group": group,
                "served_by": group,
                "stations": int(stations.nunique()),
                "new_stations": int(stations[~known[mask]].nunique()),
                "rows": int(mask.sum()),
                "served_MAE": float(abs_errors[group][mask].mean()),
                "A_MAE": float(abs_errors["A"][mask].mean()),
                "B_MAE": float(abs_errors["B"][mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def decide(
    table: pd.DataFrame,
    bootstrap: Bootstrap,
    tolerance: float = MONITORING_SETTINGS.KNOWN_STATION_TOLERANCE,
) -> Decision:
    """Apply the decision rule fixed before the test.

    B wins only when the whole bootstrap interval of the MAE difference is below zero and
    B does not raise the MAE of the known stations by more than `tolerance`.
    """
    mae = table[table["metric"] == "MAE"].set_index("segment")
    known_delta = float(mae.loc["known_stations", "delta"])
    level = f"{bootstrap.confidence:.0%}"
    interval = f"[{bootstrap.low:+.4f}, {bootstrap.high:+.4f}]"
    significant = bootstrap.high < 0
    safe = known_delta <= tolerance
    reasons = [
        f"{level} interval of MAE(B) - MAE(A): {interval}, "
        + ("entirely below zero." if significant else "not entirely below zero."),
        f"MAE change on known stations: {known_delta:+.4f}, "
        + (f"within the {tolerance} tolerance." if safe else f"above the {tolerance} tolerance."),
    ]
    return Decision("B" if significant and safe else "A", reasons)


def weekly_mae_figure(
    dates: pd.Series, abs_errors: dict[str, np.ndarray], arms: dict[str, Arm]
) -> Figure:
    """Draw the MAE of each arm per week of the evaluation period."""
    week = dates.dt.to_period("W-SUN").dt.start_time.to_numpy()
    figure, ax = plt.subplots(figsize=(9, 3.5))
    for label, errors in abs_errors.items():
        weekly = pd.Series(errors).groupby(week).mean()
        ax.plot(weekly.index, weekly.to_numpy(), marker="o", label=arms[label].name)
    ax.set_title("MAE per week on the evaluation period")
    ax.set_ylabel("MAE (departures per hour)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def bootstrap_figure(deltas: np.ndarray, bootstrap: Bootstrap) -> Figure:
    """Draw the resampled MAE differences with the interval and zero marked."""
    figure, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(deltas, bins=40, color="tab:blue", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    for end in (bootstrap.low, bootstrap.high):
        ax.axvline(end, color="tab:red", linestyle="--", linewidth=1)
    ax.set_title(
        f"MAE(B) - MAE(A) over {bootstrap.samples} day resamples, "
        f"{bootstrap.confidence:.0%} interval in red"
    )
    ax.set_xlabel("MAE difference (departures per hour), negative favors B")
    ax.set_ylabel("Resamples")
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def model_size_mb(uri: str) -> float:
    """Size on disk of a model's files, in megabytes."""
    path = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / 1e6


def promote_challenger(question: SupervisedQuestion, a: Arm, b: Arm) -> None:
    """Swap the aliases: B becomes `champion` and A becomes `challenger`."""
    client = MlflowClient()
    name = question.registered_model_name
    client.set_registered_model_alias(name, CHAMPION, b.version)
    client.set_registered_model_alias(name, CHALLENGER, a.version)
    logger.info(f"{name}: v{b.version} is now @{CHAMPION}, v{a.version} is @{CHALLENGER}")


def write_outputs(
    output_dir: Path,
    table: pd.DataFrame,
    split: pd.DataFrame,
    summary: dict,
    figures: dict[str, Figure],
) -> list[Path]:
    """Write the tables, summary, and figures. Returns the written files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / f"{REPORT_PREFIX}_metrics.csv",
        output_dir / f"{REPORT_PREFIX}_traffic_split.csv",
        output_dir / f"{REPORT_PREFIX}_summary.json",
    ]
    table.to_csv(paths[0], index=False)
    split.to_csv(paths[1], index=False)
    paths[2].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, figure in figures.items():
        path = output_dir / f"{REPORT_PREFIX}_{name}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A/B test the P1 champion and challenger.")
    parser.add_argument("--promote", type=parse_bool, default=False)
    args = parser.parse_args(argv)

    question = DemandQuestion()
    configure_tracking()
    arms = {"A": load_arm(question, CHAMPION, "A"), "B": load_arm(question, CHALLENGER, "B")}
    if arms["A"].version == arms["B"].version:
        raise ValueError("Champion and challenger are the same version. Run retrain first.")

    data, manifests = load_period(EVALUATION)
    reference, _ = load_period(REFERENCE)
    known_stations = reference[GoldColumns.STATION_ID].unique()
    features = data[list(question.feature_columns)]
    target = data[question.target_column].to_numpy()

    predictions, abs_errors, predict_seconds, sizes = {}, {}, {}, {}
    for label, arm in arms.items():
        model = mlflow.pyfunc.load_model(arm.uri)
        started = time.perf_counter()
        predictions[label] = np.asarray(model.predict(features), dtype=float)
        predict_seconds[label] = time.perf_counter() - started
        abs_errors[label] = np.abs(target - predictions[label])
        sizes[label] = model_size_mb(arm.uri)
        logger.info(f"{arm.name}: MAE {abs_errors[label].mean():.4f}")

    table = metrics_table(question, data, predictions, segments(data, known_stations))
    bootstrap, deltas = paired_bootstrap(data[GoldColumns.DATE], abs_errors["A"], abs_errors["B"])
    groups = traffic_group(data[GoldColumns.STATION_ID])
    split = traffic_split_table(data, abs_errors, groups, known_stations)
    decision = decide(table, bootstrap)
    promoted = args.promote and decision.winner == "B"

    summary = {
        "evaluation_period": EVALUATION.label,
        "rows": len(data),
        "arms": {label: asdict(arm) for label, arm in arms.items()},
        "bootstrap": asdict(bootstrap),
        "decision": asdict(decision),
        "promoted": promoted,
        "predict_seconds": predict_seconds,
        "model_size_mb": sizes,
    }
    figures = {
        "weekly_mae": weekly_mae_figure(data[GoldColumns.DATE], abs_errors, arms),
        "bootstrap": bootstrap_figure(deltas, bootstrap),
    }
    paths = write_outputs(MONITORING_SETTINGS.REPORTS_DIR, table, split, summary, figures)

    with mlflow.start_run(run_name=f"P1_ab_test_v{arms['A'].version}_v{arms['B'].version}"):
        mlflow.set_tags(
            {
                **base_tags(STAGE),
                "question": question.key,
                "dataset_revision": manifests[0]["dataset_revision"],
                "version_a": arms["A"].version,
                "version_b": arms["B"].version,
                "winner": arms[decision.winner].name,
                "promoted": str(promoted).lower(),
            }
        )
        mlflow.log_params(
            {
                "evaluation_period": EVALUATION.label,
                "rows": len(data),
                "bootstrap_samples": bootstrap.samples,
                "confidence": bootstrap.confidence,
                "known_station_tolerance": MONITORING_SETTINGS.KNOWN_STATION_TOLERANCE,
            }
        )
        overall = table[table["segment"] == "all"].set_index("metric")
        mae = table[table["metric"] == "MAE"].set_index("segment")
        mlflow.log_metrics(
            {
                **{
                    f"{label}_{metric}": overall.loc[metric, label]
                    for label in "AB"
                    for metric in METRICS
                },
                **{
                    f"{label}_MAE_{segment}": mae.loc[segment, label]
                    for label in "AB"
                    for segment in mae.index
                },
                "delta_MAE": bootstrap.delta,
                "delta_MAE_low": bootstrap.low,
                "delta_MAE_high": bootstrap.high,
                **{f"{label}_predict_seconds": value for label, value in predict_seconds.items()},
                **{f"{label}_model_size_mb": value for label, value in sizes.items()},
            }
        )
        for path in paths:
            mlflow.log_artifact(str(path), artifact_path=STAGE)

    if promoted:
        promote_challenger(question, arms["A"], arms["B"])
    for reason in decision.reasons:
        logger.info(reason)
    logger.info(f"Winner: {arms[decision.winner].name}. Promoted: {promoted}.")


if __name__ == "__main__":
    main()
