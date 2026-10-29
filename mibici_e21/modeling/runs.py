"""Log training runs and nested searches to MLflow, identically for every question.

A run started here resumes the run that `mlflow run` created, when there is one, so the
project run is the training run (or the search parent) instead of an empty sibling.
"""

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd

from mibici_e21.config import MODELING_SETTINGS, PROJ_ROOT
from mibici_e21.dataset import data_manifest_path
from mibici_e21.modeling.base import (
    BASELINE,
    Evaluation,
    Hyperparameters,
    Question,
    split_boundaries,
)
from mibici_e21.tracking import base_tags, log_data_manifest

CANDIDATE = "candidate"
SEARCH = "search"
INPUT_EXAMPLE_ROWS = 5
RUN_CONFIG_FILES = ("MLproject", "python_env.yaml")


def train_run(
    question: Question,
    data: pd.DataFrame,
    model: str,
    hyperparameters: Hyperparameters,
    manifest: dict,
    search_run_id: str | None = None,
) -> str:
    """Train one model under the question protocol and log it as a run.

    - `question`: question being answered.
    - `data`: output of `question.prepare`.
    - `model`: model name, `baseline` or a candidate such as `rf`.
    - `hyperparameters`: values for the model hyperparameters. Extra keys are ignored.
    - `manifest`: data manifest of the ingestion the data comes from.
    - `search_run_id`: search that chose `hyperparameters`, recorded as a tag.

    The run holds the metrics, the model with its signature and input example, the figures
    under `plots/`, the manifest and training input under `data/`, and the run config under
    `config/`. Returns the run id.
    """
    spec = question.model_spec(model)
    used = {name: hyperparameters[name] for name in spec.hyperparameters}
    year = manifest["year"]
    evaluation = question.evaluate(data, question.build_estimator(model, used), year)
    stage = BASELINE if model == BASELINE else CANDIDATE

    with mlflow.start_run(run_name=_run_name(question, model, used)) as run:
        tags = {**base_tags(stage), **_question_tags(question, spec.family, manifest)}
        if search_run_id:
            tags["search_run_id"] = search_run_id
        mlflow.set_tags(tags)
        mlflow.log_params({**_data_params(question, manifest), "model": model, **used})
        mlflow.log_metrics(evaluation.metrics)
        _log_figures(evaluation)
        _log_data(question, evaluation, year)
        _log_config(question, model, used, evaluation)
        _log_model(question, evaluation)

    return run.info.run_id


def search_run(
    question: Question, data: pd.DataFrame, model: str, manifest: dict
) -> tuple[Hyperparameters, str]:
    """Search the hyperparameters of a model with one parent run and one child per combination.

    Children are scored on the validation data only, so the test set stays untouched until
    the chosen combination is trained by `train_run`. The parent records the best
    combination and its selection metric. Returns `(best_hyperparameters, parent_run_id)`.
    """
    if model == BASELINE:
        raise ValueError("The baseline has no hyperparameters to search. Pick a candidate model.")
    spec = question.model_spec(model)
    space = question.search_space(model)
    year = manifest["year"]
    tags = {**base_tags(SEARCH), **_question_tags(question, spec.family, manifest)}

    with mlflow.start_run(run_name=f"{question.key.upper()}_{model}_search") as parent:
        mlflow.set_tags(tags)
        mlflow.log_params({**_data_params(question, manifest), "model": model})
        mlflow.log_dict({"search_space": space}, "config/search_space.json")

        results = []
        for hyperparameters in space:
            with mlflow.start_run(
                run_name=_run_name(question, model, hyperparameters), nested=True
            ):
                mlflow.set_tags(tags)
                mlflow.log_params({"model": model, **hyperparameters})
                estimator = question.build_estimator(model, hyperparameters)
                metrics = question.score(data, estimator, year)
                mlflow.log_metrics(metrics)
            results.append((hyperparameters, metrics[question.selection_metric]))

        best, best_value = results[0]
        for hyperparameters, value in results[1:]:
            if question.is_better(value, best_value):
                best, best_value = hyperparameters, value
        mlflow.log_params({f"best_{name}": value for name, value in best.items()})
        mlflow.log_metric(f"best_{question.selection_metric}", best_value)

    return best, parent.info.run_id


def _run_name(question: Question, model: str, hyperparameters: Hyperparameters) -> str:
    """Return a readable run name, such as `P1_rf_n_estimators=100_max_depth=10`."""
    parts = [question.key.upper(), model]
    parts += [f"{name}={value}" for name, value in hyperparameters.items()]
    return "_".join(parts)


def _question_tags(question: Question, model_family: str, manifest: dict) -> dict[str, str]:
    return {
        "question": question.key,
        "task": question.task,
        "model_family": model_family,
        "dataset_revision": manifest["dataset_revision"],
    }


def _data_params(question: Question, manifest: dict) -> dict:
    val_start, test_start = split_boundaries(manifest["year"])
    return {
        "question": question.key,
        "year": manifest["year"],
        "dataset_revision": manifest["dataset_revision"],
        "val_start_date": val_start.isoformat(),
        "test_start_date": test_start.isoformat(),
        "seed": MODELING_SETTINGS.SEED,
        "features": ",".join(question.feature_columns),
    }


def _log_figures(evaluation: Evaluation) -> None:
    for name, figure in evaluation.figures.items():
        mlflow.log_figure(figure, f"plots/{name}")
        plt.close(figure)


def _log_data(question: Question, evaluation: Evaluation, year: int) -> None:
    """Log the data manifest and the training frame as the run input.

    The frames are aggregated by station, so no rider data reaches MLflow.
    """
    log_data_manifest(data_manifest_path(year))
    source = MODELING_SETTINGS.PROCESSED_DIR / f"demand_aggregated_{year}.parquet"
    dataset = mlflow.data.from_pandas(
        evaluation.training_frame, source=str(source), name=f"{question.key}_train_{year}"
    )
    mlflow.log_input(dataset, context="training")


def _log_config(
    question: Question, model: str, hyperparameters: Hyperparameters, evaluation: Evaluation
) -> None:
    """Log the resolved run configuration and the project files that launched the run."""
    settings = {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in asdict(MODELING_SETTINGS).items()
        if not isinstance(value, Path)
    }
    estimator_params = {
        name: value
        for name, value in evaluation.estimator.get_params(deep=True).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    mlflow.log_dict(
        {
            "question": question.key,
            "model": model,
            "hyperparameters": hyperparameters,
            "estimator_params": estimator_params,
            "row_counts": evaluation.row_counts,
            "modeling_settings": settings,
        },
        "config/run_config.json",
    )
    for file_name in RUN_CONFIG_FILES:
        mlflow.log_artifact(str(PROJ_ROOT / file_name), artifact_path="config")


def _log_model(question: Question, evaluation: Evaluation) -> None:
    """Log the estimator with its signature, an input example, and the project code.

    `code_paths` ships the `mibici_e21` package with the model, so the baselines load
    anywhere. The model URI goes into the `model_uri` tag, which the registry reads.
    """
    example = evaluation.training_frame[list(question.feature_columns)].head(INPUT_EXAMPLE_ROWS)
    info = mlflow.sklearn.log_model(
        evaluation.estimator,
        name="model",
        signature=infer_signature(example, evaluation.estimator.predict(example)),
        input_example=example,
        code_paths=[str(PROJ_ROOT / "mibici_e21")],
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    mlflow.set_tag("model_uri", info.model_uri)
