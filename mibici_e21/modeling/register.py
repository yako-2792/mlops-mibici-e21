"""Promote the best candidate of a question to production in the Model Registry.

This is the `register` entry point of the MLflow Project:

    mlflow run . -e register -P question=p1
    mlflow run . -e register -P question=p1 -P run_id=<candidate run id>

Lifecycle scheme: aliases, not stages. MLflow deprecated stages, and an alias is a movable
pointer, so promoting or rolling back is one call and consumers keep loading the same URI.
`champion` is the version in production and `challenger` the runner-up. A candidate is
only promoted when it beats the latest baseline of its question on the selection metric.
"""

import argparse
from dataclasses import dataclass

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
import mlflow.pyfunc

from mibici_e21.config import REGISTRY_SETTINGS, TRACKING_SETTINGS
from mibici_e21.modeling.base import BASELINE, Question
from mibici_e21.modeling.questions import QUESTIONS
from mibici_e21.modeling.runs import CANDIDATE
from mibici_e21.tracking import base_tags, configure_tracking

CHAMPION = REGISTRY_SETTINGS.CHAMPION_ALIAS
CHALLENGER = REGISTRY_SETTINGS.CHALLENGER_ALIAS


@dataclass(frozen=True)
class Promotion:
    """Outcome of a registration.

    - `model_name`: registered model name.
    - `champion`: version now behind the `champion` alias.
    - `challenger`: version behind the `challenger` alias, if there is a runner-up.
    """

    model_name: str
    champion: ModelVersion
    challenger: ModelVersion | None


def rank_candidates(
    question: Question, baseline_value: float, candidate_values: dict[str, float]
) -> list[str]:
    """Return the run ids of the candidates that beat the baseline, best first.

    - `question`: question whose selection metric is compared.
    - `baseline_value`: selection metric of the baseline.
    - `candidate_values`: candidate run id mapped to its selection metric.
    """
    beating = [
        run_id
        for run_id, value in candidate_values.items()
        if question.is_better(value, baseline_value)
    ]
    return sorted(
        beating,
        key=lambda run_id: candidate_values[run_id],
        reverse=question.greater_is_better,
    )


def promote(question: Question, experiment_id: str, run_id: str | None = None) -> Promotion:
    """Register the best candidate of a question and point `champion` at it.

    - `question`: question to promote.
    - `experiment_id`: experiment that holds its runs.
    - `run_id`: candidate to promote instead of the best one, for example a cheaper model
      with a similar score. It must still beat the baseline.

    The runner-up, when there is one, gets `challenger`. Raises `ValueError` when no
    candidate beats the baseline.
    """
    client = MlflowClient()
    metric = question.selection_metric
    baseline = _latest_baseline(client, experiment_id, question)
    baseline_value = baseline.data.metrics[metric]
    candidates = _candidate_runs(client, experiment_id, question)
    ranked = rank_candidates(
        question, baseline_value, {run.info.run_id: run.data.metrics[metric] for run in candidates}
    )
    if not ranked:
        raise ValueError(
            f"No {question.key} candidate beats the baseline {metric}={baseline_value:.4f}."
        )
    if run_id is not None:
        if run_id not in ranked:
            raise ValueError(
                f"Run {run_id} is not a {question.key} candidate beating the baseline."
            )
        ranked.remove(run_id)
        ranked.insert(0, run_id)

    runs = {run.info.run_id: run for run in candidates}
    name = question.registered_model_name
    _ensure_registered_model(client, question)
    champion = _register_version(client, question, runs[ranked[0]], baseline)
    client.set_registered_model_alias(name, CHAMPION, champion.version)

    challenger = None
    if len(ranked) > 1:
        challenger = _register_version(client, question, runs[ranked[1]], baseline)
        client.set_registered_model_alias(name, CHALLENGER, challenger.version)
    else:
        _delete_alias_if_set(client, name, CHALLENGER)

    _check_serving(name)
    return Promotion(model_name=name, champion=champion, challenger=challenger)


def _latest_baseline(client: MlflowClient, experiment_id: str, question: Question) -> Run:
    runs = _finished_runs(client, experiment_id, question, BASELINE)
    if not runs:
        raise ValueError(f"No finished {question.key} baseline. Train it before registering.")
    return runs[0]


def _candidate_runs(client: MlflowClient, experiment_id: str, question: Question) -> list[Run]:
    return [
        run
        for run in _finished_runs(client, experiment_id, question, CANDIDATE)
        if "model_uri" in run.data.tags and question.selection_metric in run.data.metrics
    ]


def _finished_runs(
    client: MlflowClient, experiment_id: str, question: Question, stage: str
) -> list[Run]:
    """Return the finished runs of a question and stage, newest first."""
    return client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.question = '{question.key}' and tags.stage_of_work = '{stage}' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
    )


def _ensure_registered_model(client: MlflowClient, question: Question) -> None:
    name = question.registered_model_name
    if not client.search_registered_models(filter_string=f"name = '{name}'"):
        client.create_registered_model(name)
    client.update_registered_model(name, description=question.description)


def _register_version(
    client: MlflowClient, question: Question, run: Run, baseline: Run
) -> ModelVersion:
    """Register the model of a run, or reuse its version, and record its lineage as tags."""
    name = question.registered_model_name
    run_id = run.info.run_id
    existing = client.search_model_versions(f"name = '{name}' and run_id = '{run_id}'")
    version = existing[0] if existing else mlflow.register_model(run.data.tags["model_uri"], name)

    metric = question.selection_metric
    value = run.data.metrics[metric]
    baseline_value = baseline.data.metrics[metric]
    improvement = abs(value - baseline_value) / abs(baseline_value)
    client.update_model_version(
        name,
        version.version,
        description=(
            f"{run.info.run_name}: {metric}={value:.4f} against baseline {baseline_value:.4f} "
            f"({improvement:.1%} better)."
        ),
    )
    lineage = {
        "experiment": TRACKING_SETTINGS.EXPERIMENT_NAME,
        "source_run_id": run_id,
        "source_run_name": run.info.run_name,
        "git_commit": run.data.tags["git_commit"],
        "dataset_revision": run.data.tags["dataset_revision"],
        "model_family": run.data.tags["model_family"],
        "selection_metric": metric,
        "selection_value": f"{value:.6f}",
        "baseline_run_id": baseline.info.run_id,
        "baseline_value": f"{baseline_value:.6f}",
    }
    for key, tag_value in lineage.items():
        client.set_model_version_tag(name, version.version, key, tag_value)
    return client.get_model_version(name, version.version)


def _delete_alias_if_set(client: MlflowClient, name: str, alias: str) -> None:
    if alias in client.get_registered_model(name).aliases:
        client.delete_registered_model_alias(name, alias)


def _check_serving(name: str) -> None:
    """Load the champion by name and alias and score its own input example."""
    uri = f"models:/{name}@{CHAMPION}"
    model = mlflow.pyfunc.load_model(uri)
    predictions = model.predict(model.input_example)
    logger.info(f"{uri} loads and scores {len(predictions)} example rows")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Promote the best MiBici candidate.")
    parser.add_argument("--question", required=True, choices=sorted(QUESTIONS))
    parser.add_argument("--run_id", default="", help="Candidate to promote. Empty picks the best.")
    args = parser.parse_args(argv)

    question = QUESTIONS[args.question]
    experiment_id = configure_tracking()
    promotion = promote(question, experiment_id, args.run_id.strip() or None)

    with mlflow.start_run(run_name=f"{question.key.upper()}_register"):
        tags = {**base_tags("register"), "question": question.key}
        tags["registered_model"] = promotion.model_name
        tags["champion_version"] = promotion.champion.version
        tags["champion_run_id"] = promotion.champion.run_id
        mlflow.set_tags(tags)

    challenger = promotion.challenger.version if promotion.challenger else "none"
    logger.info(
        f"{promotion.model_name}: champion v{promotion.champion.version}, challenger {challenger}"
    )


if __name__ == "__main__":
    main()
