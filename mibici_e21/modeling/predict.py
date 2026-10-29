"""Batch inference with the production model of each question.

Each question is scored with `models:/<registered name>@champion`, so promoting a new
version in the registry changes what this script serves without touching its code.

    python -m mibici_e21.modeling.predict --year 2024 --csv
"""

import argparse

from loguru import logger
import mlflow
import mlflow.pyfunc
import pandas as pd

from mibici_e21.config import PREDICT_SETTINGS, REGISTRY_SETTINGS, TRACKING_SETTINGS
from mibici_e21.features import load_gold_demand
from mibici_e21.modeling.base import Question
from mibici_e21.modeling.questions import QUESTIONS


def champion_uri(question: Question) -> str:
    """Return the URI of the production model of a question."""
    return f"models:/{question.registered_model_name}@{REGISTRY_SETTINGS.CHAMPION_ALIAS}"


def predict_question(question: Question, gold: pd.DataFrame) -> pd.DataFrame:
    """Score the gold demand with the champion of a question.

    Returns the key columns of the question and its prediction column.
    """
    data = question.prepare(gold)
    model = mlflow.pyfunc.load_model(champion_uri(question))
    predictions = model.predict(data[list(question.feature_columns)])
    return data[list(question.key_columns)].assign(**{question.prediction_column: predictions})


def run_batch_inference(year: int, save_csv: bool = False) -> None:
    """Write `predictions_{question}_{year}.parquet` for every question.

    - `year`: gold demand year to score. Its gold files must already exist.
    - `save_csv`: also write a CSV copy of each file.
    """
    mlflow.set_tracking_uri(TRACKING_SETTINGS.TRACKING_URI)
    gold = load_gold_demand(year)
    PREDICT_SETTINGS.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for question in QUESTIONS.values():
        predictions = predict_question(question, gold)
        path = PREDICT_SETTINGS.OUTPUT_DIR / f"predictions_{question.key}_{year}.parquet"
        predictions.to_parquet(path, index=False)
        if save_csv:
            predictions.to_csv(path.with_suffix(".csv"), index=False)
        logger.info(f"{champion_uri(question)}: {len(predictions)} predictions in {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a year with the champion models.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--csv", action="store_true", help="Also write CSV copies.")
    args = parser.parse_args(argv)
    run_batch_inference(args.year, save_csv=args.csv)


if __name__ == "__main__":
    main()
