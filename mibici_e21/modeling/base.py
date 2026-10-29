"""Contract shared by the three questions and the temporal split they evaluate on.

A `Question` knows how to turn the gold demand into its modeling frame, which models it
accepts, and how to fit and score them. `runs.py` logs whatever a question returns, so
the MLflow code is written once for P1, P2, and P3.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
import time
from typing import ClassVar

from matplotlib.figure import Figure
import pandas as pd
from sklearn.base import BaseEstimator, clone

from mibici_e21.config import MODELING_SETTINGS, TRACKING_SETTINGS
from mibici_e21.schemas.columns import GoldColumns

Hyperparameters = dict[str, int]

BASELINE = "baseline"


@dataclass(frozen=True)
class ModelSpec:
    """A model a question accepts.

    - `family`: value of the `model_family` tag.
    - `hyperparameters`: names the model needs, such as `n_estimators`.
    - `build`: function that maps those hyperparameters to an unfitted estimator.
    """

    family: str
    hyperparameters: tuple[str, ...]
    build: Callable[[Hyperparameters], BaseEstimator]


@dataclass
class Evaluation:
    """What one training run produces, ready to be logged.

    - `estimator`: final fitted estimator, the one logged as the run model.
    - `metrics`: metric name mapped to its value.
    - `figures`: artifact path under `plots/` mapped to its figure.
    - `training_frame`: frame the final estimator was fitted on, logged as the run input.
    - `row_counts`: rows of each part of the data, recorded in the run config.
    """

    estimator: BaseEstimator
    metrics: dict[str, float]
    figures: dict[str, Figure]
    training_frame: pd.DataFrame
    row_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TemporalSplit:
    """Rows of a frame split by date, identical for every run of a question.

    - `train`: dates before the validation month. Searches fit here.
    - `val`: dates from the validation month up to the test month. Searches score here.
    - `train_full`: `train` plus `val`. The final estimator fits here.
    - `test`: dates from the test month to the end of the year.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    train_full: pd.DataFrame
    test: pd.DataFrame

    def row_counts(self) -> dict[str, int]:
        return {
            "train_rows": len(self.train),
            "val_rows": len(self.val),
            "test_rows": len(self.test),
        }


def split_boundaries(year: int) -> tuple[date, date]:
    """Return the first day of the validation set and of the test set for a year."""
    return (
        date(year, MODELING_SETTINGS.VAL_START_MONTH, 1),
        date(year, MODELING_SETTINGS.TEST_START_MONTH, 1),
    )


def temporal_split(frame: pd.DataFrame, year: int) -> TemporalSplit:
    """Split a frame with a datetime `date` column into the validation and test protocols."""
    val_start, test_start = (pd.Timestamp(day) for day in split_boundaries(year))
    dates = frame[GoldColumns.DATE]
    return TemporalSplit(
        train=frame[dates < val_start],
        val=frame[(dates >= val_start) & (dates < test_start)],
        train_full=frame[dates < test_start],
        test=frame[dates >= test_start],
    )


class Question(ABC):
    """A business question answered by one registered model.

    Subclasses set the class attributes and implement the abstract methods.

    - `key`: `p1`, `p2`, or `p3`.
    - `task`: `regression`, `classification`, or `clustering`.
    - `registered_model_suffix`: end of the registered model name, such as `p1-demanda`.
    - `description`: description of the registered model.
    - `selection_metric`: metric that picks the best search combination and the champion.
    - `greater_is_better`: whether a higher `selection_metric` is better.
    - `models`: model name mapped to its `ModelSpec`. Must include `baseline`.
    - `feature_columns`: columns the estimator reads, in order.
    - `key_columns`: columns that identify a row of the modeling frame in predictions.
    - `prediction_column`: name of the prediction column in batch inference output.
    """

    key: str
    task: str
    registered_model_suffix: str
    description: str
    selection_metric: str
    greater_is_better: bool
    models: ClassVar[dict[str, ModelSpec]]
    feature_columns: tuple[str, ...]
    key_columns: tuple[str, ...]
    prediction_column: str

    @abstractmethod
    def prepare(self, gold: pd.DataFrame) -> pd.DataFrame:
        """Turn the gold demand into the modeling frame of the question."""

    @abstractmethod
    def evaluate(self, data: pd.DataFrame, estimator: BaseEstimator, year: int) -> Evaluation:
        """Fit `estimator` under the question protocol and return everything to log."""

    @abstractmethod
    def score(self, data: pd.DataFrame, estimator: BaseEstimator, year: int) -> dict[str, float]:
        """Fit and score one search combination. Only the selection data is used."""

    @abstractmethod
    def search_space(self, model: str) -> list[Hyperparameters]:
        """Return the hyperparameter combinations a search of `model` tries."""

    @property
    def registered_model_name(self) -> str:
        """Name in the Model Registry, such as `mibici-equipo21-p1-demanda`."""
        return f"{TRACKING_SETTINGS.EXPERIMENT_NAME}-{self.registered_model_suffix}"

    def build_estimator(self, model: str, hyperparameters: Hyperparameters) -> BaseEstimator:
        """Build the unfitted estimator of a model.

        Raises `ValueError` for an unknown model or a missing hyperparameter.
        """
        spec = self.model_spec(model)
        missing = [name for name in spec.hyperparameters if hyperparameters.get(name) is None]
        if missing:
            flags = " ".join(f"-P {name}=..." for name in missing)
            raise ValueError(f"{self.key} {model} needs {', '.join(missing)}. Pass {flags}.")
        return spec.build({name: hyperparameters[name] for name in spec.hyperparameters})

    def model_spec(self, model: str) -> ModelSpec:
        """Return the spec of a model. Raises `ValueError` for an unknown model."""
        if model not in self.models:
            raise ValueError(
                f"Unknown {self.key} model '{model}'. Use one of {list(self.models)}."
            )
        return self.models[model]

    def is_better(self, value: float, reference: float) -> bool:
        """Return whether `value` beats `reference` on the selection metric."""
        return value > reference if self.greater_is_better else value < reference


class SupervisedQuestion(Question):
    """A question with a target, evaluated on the temporal split.

    The final estimator is fitted on `train_full` and scored on `test`. A clone is fitted on
    `train` and scored on `val`, which is the score the search and the registry compare.
    """

    target_column: str

    @abstractmethod
    def metrics(self, y_true: pd.Series, y_pred, prefix: str) -> dict[str, float]:
        """Return the metrics of a prediction, each name starting with `prefix_`."""

    @abstractmethod
    def figures(
        self, test: pd.DataFrame, test_pred, estimator: BaseEstimator
    ) -> dict[str, Figure]:
        """Return the diagnostic figures of the final estimator on the test set."""

    def evaluate(self, data: pd.DataFrame, estimator: BaseEstimator, year: int) -> Evaluation:
        split = temporal_split(data, year)
        val_metrics = self._fit_and_score(estimator, split.train, split.val, "val")

        started = time.perf_counter()
        estimator.fit(self._features(split.train_full), split.train_full[self.target_column])
        train_seconds = time.perf_counter() - started

        started = time.perf_counter()
        test_pred = estimator.predict(self._features(split.test))
        predict_seconds = time.perf_counter() - started

        return Evaluation(
            estimator=estimator,
            metrics={
                **val_metrics,
                **self.metrics(split.test[self.target_column], test_pred, "test"),
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
            },
            figures=self.figures(split.test, test_pred, estimator),
            training_frame=split.train_full,
            row_counts=split.row_counts(),
        )

    def score(self, data: pd.DataFrame, estimator: BaseEstimator, year: int) -> dict[str, float]:
        split = temporal_split(data, year)
        started = time.perf_counter()
        metrics = self._fit_and_score(estimator, split.train, split.val, "val")
        return {**metrics, "train_seconds": time.perf_counter() - started}

    def _fit_and_score(
        self, estimator: BaseEstimator, fit_on: pd.DataFrame, score_on: pd.DataFrame, prefix: str
    ) -> dict[str, float]:
        fitted = clone(estimator).fit(self._features(fit_on), fit_on[self.target_column])
        return self.metrics(
            score_on[self.target_column], fitted.predict(self._features(score_on)), prefix
        )

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[list(self.feature_columns)]
