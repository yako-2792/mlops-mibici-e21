"""Register the best candidate of each question in the MLflow Model Registry.

Run it through the MLflow Project: `mlflow run . -e register -P question=p1`.

For each question the script picks the candidate run with the best validation metric,
registers its model, describes it, records its lineage as version tags, sets the
`champion` alias on that version, and checks that the model loads by name and alias.
Aliases replace the stages of the registry, which MLflow deprecated.

Importing this module points the tracking URI at `MODELING_SETTINGS.TRACKING_URI`.
"""

import argparse

from loguru import logger
import mlflow
from mlflow import MlflowClient
import mlflow.pyfunc

from mibici_e21.config import MODELING_SETTINGS

mlflow.set_tracking_uri(MODELING_SETTINGS.TRACKING_URI)

REGISTRY = {
    "p1": {
        "name": "mibici-equipo21-p1-demanda",
        "metric": "val_MAE",
        "minimize": True,
        "description": "P1: hourly bike departures per station. Regression model.",
    },
    "p2": {
        "name": "mibici-equipo21-p2-riesgo",
        "metric": "val_f1_macro",
        "minimize": False,
        "description": "P2: daily shortage or saturation risk per station. Classification model.",
    },
    "p3": {
        "name": "mibici-equipo21-p3-grupos",
        "metric": "silhouette",
        "minimize": False,
        "description": "P3: groups of stations by usage. Clustering model (scaler and KMeans).",
    },
}


def select_best_run(client: MlflowClient, question: str):
    """Return the candidate run of a question with the best selection metric.

    - `client`: MLflow client.
    - `question`: `p1`, `p2`, or `p3`.

    Only runs tagged `stage_of_work = candidate` that logged a model (`model_uri` tag) and
    the selection metric are considered. Raises `ValueError` when there is none.
    """
    spec = REGISTRY[question]
    experiment = client.get_experiment_by_name(MODELING_SETTINGS.EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{MODELING_SETTINGS.EXPERIMENT_NAME}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.question = '{question}' and tags.stage_of_work = 'candidate'",
        max_results=1000,
    )
    runs = [
        run for run in runs if spec["metric"] in run.data.metrics and "model_uri" in run.data.tags
    ]
    if not runs:
        raise ValueError(f"No candidate run with a model and '{spec['metric']}' for {question}.")

    pick = min if spec["minimize"] else max
    return pick(runs, key=lambda run: run.data.metrics[spec["metric"]])


def register_question(question: str, run_id: str = "auto", alias: str = "champion") -> str:
    """Register the model of one question and set its alias.

    - `question`: `p1`, `p2`, or `p3`.
    - `run_id`: run to register, or `auto` to take the best candidate by the selection metric.
      Raises `ValueError` when that run is of another question or logged no model.
    - `alias`: alias to set on the new version. Defaults to `champion`.

    The registered model gets a description. The new version gets a description and the tags
    `source_run_id`, `experiment`, `git_commit`, `dataset_version`, `model_family`,
    `selection_metric`, and `selection_value`, which give its lineage. Returns the version.
    """
    spec = REGISTRY[question]
    client = MlflowClient()
    run = select_best_run(client, question) if run_id == "auto" else client.get_run(run_id)
    name = spec["name"]
    if run.data.tags.get("question") != question:
        raise ValueError(f"Run {run.info.run_id} is not a {question} run.")
    if "model_uri" not in run.data.tags or spec["metric"] not in run.data.metrics:
        raise ValueError(f"Run {run.info.run_id} has no model or no '{spec['metric']}'.")
    value = run.data.metrics[spec["metric"]]

    version = mlflow.register_model(run.data.tags["model_uri"], name)
    client.update_registered_model(name, description=spec["description"])
    client.update_model_version(
        name,
        version.version,
        description=f"Run {run.info.run_name} ({run.info.run_id}). {spec['metric']}={value:.4f}.",
    )
    lineage = {
        "source_run_id": run.info.run_id,
        "experiment": MODELING_SETTINGS.EXPERIMENT_NAME,
        "git_commit": run.data.tags.get("git_commit", "unknown"),
        "dataset_version": run.data.params.get("dataset_version", "unknown"),
        "model_family": run.data.tags.get("model_family", "unknown"),
        "selection_metric": spec["metric"],
        "selection_value": f"{value:.6f}",
    }
    for key, tag_value in lineage.items():
        client.set_model_version_tag(name, version.version, key, tag_value)
    client.set_registered_model_alias(name, alias, version.version)

    uri = f"models:/{name}@{alias}"
    mlflow.pyfunc.load_model(uri)
    logger.info(f"{uri} -> version {version.version} ({spec['metric']}={value:.4f}) loads fine")
    return str(version.version)


def main() -> None:
    """Register from the command line, which is how the MLflow Project calls it.

    - `--question`: `p1`, `p2`, `p3`, or `all`.
    - `--run_id`: run to register, or `auto` (default) for the best candidate.
    - `--alias`: alias to set. Defaults to `champion`.
    """
    parser = argparse.ArgumentParser(description="Register the best MiBici models.")
    parser.add_argument("--question", type=str, default="all", choices=["p1", "p2", "p3", "all"])
    parser.add_argument("--run_id", type=str, default="auto")
    parser.add_argument("--alias", type=str, default="champion")
    args = parser.parse_args()

    questions = list(REGISTRY) if args.question == "all" else [args.question]
    if len(questions) > 1 and args.run_id != "auto":
        raise ValueError("--run_id needs a single --question.")
    for question in questions:
        register_question(question, args.run_id, args.alias)


if __name__ == "__main__":
    main()