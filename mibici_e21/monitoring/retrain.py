"""Retrain the P1 champion recipe on a wider window and register it as the challenger.

This is the `retrain` entry point of the MLflow Project:

    mlflow run . -e retrain
    mlflow run . -e retrain -P n_estimators=400 -P max_depth=14

The alternative version repeats the champion's model and hyperparameters, read from the
registry, unless they are passed. Only the data changes: it is fitted on `RETRAIN`, the
reference plus the first months of the current period, so the A/B test measures the effect
of retraining and nothing else.

Protocol, the same as `train` on a longer window: a clone is fitted on the days before
`RETRAIN_VAL_START` and scored on the rest of `RETRAIN` (`val_*` metrics and plots), then
the final estimator is fitted on the whole window. The `EVALUATION` period stays unseen for
the A/B test. The new version gets the `challenger` alias. `champion` does not move.
"""

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
import time

from loguru import logger
import mlflow
from mlflow import MlflowClient
import mlflow.data
from mlflow.entities.model_registry import ModelVersion
import pandas as pd
from sklearn.base import BaseEstimator, clone

from mibici_e21.config import (
    FEATURE_STORE_SETTINGS,
    MODELING_SETTINGS,
    MONITORING_SETTINGS,
    REGISTRY_SETTINGS,
    TRACKING_SETTINGS,
)
from mibici_e21.modeling.base import Evaluation, SupervisedQuestion
from mibici_e21.modeling.demand import DemandQuestion
from mibici_e21.modeling.runs import (
    _known_mlflow_warnings_silenced,
    _log_config,
    _log_figures,
    _log_model,
)
from mibici_e21.modeling.train import optional_int
from mibici_e21.monitoring.periods import RETRAIN, Period, load_period
from mibici_e21.schemas.columns import GoldColumns
from mibici_e21.tracking import base_tags, configure_tracking

STAGE = "retrain"
CHAMPION = REGISTRY_SETTINGS.CHAMPION_ALIAS
CHALLENGER = REGISTRY_SETTINGS.CHALLENGER_ALIAS


@dataclass(frozen=True)
class Recipe:
    """Model and hyperparameters of a version, and the version they were read from.

    - `model`: model name of the question, such as `rf`.
    - `hyperparameters`: values the model needs.
    - `source_version`: registry version the recipe was copied from, or `None` when passed.
    """

    model: str
    hyperparameters: dict[str, int]
    source_version: str | None


def champion_recipe(question: SupervisedQuestion) -> Recipe:
    """Read the model and hyperparameters of the champion from the params of its run."""
    client = MlflowClient()
    version = client.get_model_version_by_alias(question.registered_model_name, CHAMPION)
    params = client.get_run(version.run_id).data.params
    model = params["model"]
    names = question.model_spec(model).hyperparameters
    return Recipe(model, {name: int(params[name]) for name in names}, str(version.version))


def resolve_recipe(
    question: SupervisedQuestion, model: str | None, overrides: Mapping[str, int | None]
) -> Recipe:
    """Return the champion recipe with any passed value on top.

    - `model`: model to train instead of the champion's. Then every hyperparameter it needs
      must be passed, since the champion's may not apply.
    - `overrides`: hyperparameter values to use instead of the champion's. `None` keeps it.
    """
    if model:
        spec = question.model_spec(model)
        estimator_params = {name: overrides.get(name) for name in spec.hyperparameters}
        question.build_estimator(model, estimator_params)  # Fails on a missing value.
        return Recipe(model, estimator_params, None)

    recipe = champion_recipe(question)
    passed = {name: value for name, value in overrides.items() if value is not None}
    unknown = set(passed) - set(recipe.hyperparameters)
    if unknown:
        raise ValueError(f"The champion {recipe.model} does not use {sorted(unknown)}.")
    if not passed:
        return recipe
    return Recipe(recipe.model, {**recipe.hyperparameters, **passed}, None)


def evaluate_window(
    question: SupervisedQuestion, data: pd.DataFrame, estimator: BaseEstimator, val_start: date
) -> Evaluation:
    """Score a clone on the days from `val_start`, then fit `estimator` on all of `data`.

    Returns the final estimator, the `val_*` metrics, the fit time, and the figures of the
    clone on the validation days.
    """
    features = list(question.feature_columns)
    target = question.target_column
    is_val = data[GoldColumns.DATE] >= pd.Timestamp(val_start)
    train, val = data[~is_val], data[is_val]
    if train.empty or val.empty:
        raise ValueError(f"Validation start {val_start} leaves an empty train or val split.")

    probe = clone(estimator).fit(train[features], train[target])
    val_pred = probe.predict(val[features])

    started = time.perf_counter()
    estimator.fit(data[features], data[target])
    train_seconds = time.perf_counter() - started

    return Evaluation(
        estimator=estimator,
        metrics={**question.metrics(val[target], val_pred, "val"), "train_seconds": train_seconds},
        figures=question.figures(val, val_pred, probe),
        training_frame=data,
        row_counts={"train_rows": len(train), "val_rows": len(val), "fit_rows": len(data)},
    )


def retrain_run(
    question: SupervisedQuestion,
    data: pd.DataFrame,
    recipe: Recipe,
    period: Period,
    val_start: date,
    manifests: list[dict],
) -> str:
    """Train a recipe on a period and log it like a candidate run. Returns the run id."""
    family = question.model_spec(recipe.model).family
    estimator = question.build_estimator(recipe.model, recipe.hyperparameters)
    evaluation = evaluate_window(question, data, estimator, val_start)
    name_parts = [f"{key}={value}" for key, value in recipe.hyperparameters.items()]
    run_name = "_".join([question.key.upper(), recipe.model, *name_parts, STAGE])

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                **base_tags(STAGE),
                "question": question.key,
                "task": question.task,
                "model_family": family,
                "dataset_revision": manifests[-1]["dataset_revision"],
                "train_start": period.start.isoformat(),
                "train_end": period.last_day.isoformat(),
                "retrained_from_version": recipe.source_version or "none",
            }
        )
        mlflow.log_params(
            {
                "question": question.key,
                "model": recipe.model,
                **recipe.hyperparameters,
                "train_period": period.label,
                "val_start_date": val_start.isoformat(),
                "years": ",".join(str(year) for year in period.years),
                "seed": MODELING_SETTINGS.SEED,
                "features": ",".join(question.feature_columns),
            }
        )
        mlflow.log_metrics(evaluation.metrics)
        _log_figures(evaluation)
        for manifest in manifests:
            mlflow.log_dict(manifest, f"data/data_manifest_{manifest['year']}.json")
        with _known_mlflow_warnings_silenced():
            dataset = mlflow.data.from_pandas(
                evaluation.training_frame,
                source=str(FEATURE_STORE_SETTINGS.GOLD_DIR),
                name=f"{question.key}_{STAGE}_{period.start}_{period.last_day}",
            )
            mlflow.log_input(dataset, context="training")
        _log_config(question, recipe.model, recipe.hyperparameters, evaluation)
        _log_model(question, evaluation)

    return run.info.run_id


def register_challenger(question: SupervisedQuestion, run_id: str) -> ModelVersion:
    """Register the model of a retrain run as a new version and point `challenger` at it.

    The version records its lineage and training window as tags. `champion` does not move:
    the A/B test decides whether it should.
    """
    client = MlflowClient()
    name = question.registered_model_name
    run = client.get_run(run_id)
    tags = run.data.tags
    version = mlflow.register_model(tags["model_uri"], name)

    metric = question.selection_metric
    value = run.data.metrics[metric]
    client.update_model_version(
        name,
        version.version,
        description=(
            f"{run.info.run_name}: retrained on {tags['train_start']} to {tags['train_end']}, "
            f"{metric}={value:.4f} on the last months of that window."
        ),
    )
    lineage = {
        "experiment": TRACKING_SETTINGS.EXPERIMENT_NAME,
        "source_run_id": run_id,
        "source_run_name": run.info.run_name,
        "git_commit": tags["git_commit"],
        "dataset_revision": tags["dataset_revision"],
        "model_family": tags["model_family"],
        "selection_metric": metric,
        "selection_value": f"{value:.6f}",
        "train_start": tags["train_start"],
        "train_end": tags["train_end"],
        "retrained_from_version": tags["retrained_from_version"],
    }
    for key, tag_value in lineage.items():
        client.set_model_version_tag(name, version.version, key, tag_value)
    client.set_registered_model_alias(name, CHALLENGER, version.version)
    return client.get_model_version(name, version.version)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain P1 and register it as challenger.")
    parser.add_argument("--model", default="", help="Empty repeats the champion's model.")
    parser.add_argument("--n_estimators", type=optional_int, default=None)
    parser.add_argument("--max_depth", type=optional_int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    question = DemandQuestion()
    val_start = MONITORING_SETTINGS.RETRAIN_VAL_START
    if not RETRAIN.start < val_start < RETRAIN.end:
        raise ValueError(f"RETRAIN_VAL_START {val_start} must fall inside {RETRAIN.label}.")

    configure_tracking()
    overrides = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}
    recipe = resolve_recipe(question, args.model.strip() or None, overrides)
    logger.info(f"Retraining {recipe.model} {recipe.hyperparameters} on {RETRAIN.label}")

    data, manifests = load_period(RETRAIN)
    run_id = retrain_run(question, data, recipe, RETRAIN, val_start, manifests)
    version = register_challenger(question, run_id)
    logger.info(f"{version.name} version {version.version} is now @{CHALLENGER}")


if __name__ == "__main__":
    main()
