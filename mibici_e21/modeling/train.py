"""Train a baseline, a candidate, or a hyperparameter search for one question.

This is the `train` entry point of the MLflow Project:

    mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=300 -P max_depth=12
    mlflow run . -e train -P question=p2 -P model=hgb -P search=true

The year, split dates, seed, and search grids come from `config.py`. Hyperparameters a
model does not use are ignored. A search logs one child run per combination under the
project run, then trains the best combination as a candidate linked by `search_run_id`.
"""

import argparse

from loguru import logger

from mibici_e21.config import MODELING_SETTINGS
from mibici_e21.dataset import read_data_manifest
from mibici_e21.features import load_gold_demand
from mibici_e21.modeling.questions import QUESTIONS
from mibici_e21.modeling.runs import search_run, train_run
from mibici_e21.tracking import configure_tracking

HYPERPARAMETERS = ("n_estimators", "max_depth", "n_clusters")


def optional_int(value: str) -> int | None:
    """Read an int, or `None` for the empty value `mlflow run` passes for an unset parameter."""
    return None if value.strip() == "" else int(value)


def parse_bool(value: str) -> bool:
    """Read `true` or `false`, as `mlflow run` passes every parameter as text."""
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no", ""):
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got '{value}'.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MiBici model and log it to MLflow.")
    parser.add_argument("--question", required=True, choices=sorted(QUESTIONS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--search", type=parse_bool, default=False)
    for name in HYPERPARAMETERS:
        parser.add_argument(f"--{name}", type=optional_int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    question = QUESTIONS[args.question]
    question.model_spec(args.model)  # Fail on an unknown model before loading any data.

    configure_tracking()
    year = MODELING_SETTINGS.TRAINING_YEAR
    manifest = read_data_manifest(year)
    data = question.prepare(load_gold_demand(year))

    search_run_id = None
    hyperparameters = {name: getattr(args, name) for name in HYPERPARAMETERS}
    if args.search:
        hyperparameters, search_run_id = search_run(question, data, args.model, manifest)
        logger.info(f"{question.key} {args.model} search picked {hyperparameters}")

    run_id = train_run(question, data, args.model, hyperparameters, manifest, search_run_id)
    logger.info(f"{question.key} {args.model} logged as run {run_id}")


if __name__ == "__main__":
    main()
