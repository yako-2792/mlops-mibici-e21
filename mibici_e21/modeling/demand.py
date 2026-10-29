"""P1: hourly departures per station, a regression on the gold demand."""

import itertools
from typing import ClassVar

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mibici_e21.config import MODELING_SETTINGS
from mibici_e21.modeling.base import (
    BASELINE,
    Hyperparameters,
    ModelSpec,
    SupervisedQuestion,
)
from mibici_e21.modeling.baselines import HistoricalMeanRegressor
from mibici_e21.schemas.columns import GoldColumns

SEED = MODELING_SETTINGS.SEED


class DemandQuestion(SupervisedQuestion):
    """How many bikes leave each station in each hour.

    The baseline is the training mean of the station and hour. MAE is the selection
    metric, because the rebalancing plan is sized in bikes per hour.
    """

    key = "p1"
    task = "regression"
    registered_model_suffix = "p1-demanda"
    description = (
        "P1: hourly departures per station, used to size rebalancing. "
        "Input: station_id, hour, day_of_week, is_weekend. Output: expected departures."
    )
    selection_metric = "val_MAE"
    greater_is_better = False
    target_column = GoldColumns.DEPARTURES_COUNT
    feature_columns = (
        GoldColumns.STATION_ID,
        GoldColumns.HOUR,
        GoldColumns.DAY_OF_WEEK,
        GoldColumns.IS_WEEKEND,
    )
    key_columns = (GoldColumns.STATION_ID, GoldColumns.DATE, GoldColumns.HOUR)
    prediction_column = "predicted_departures"
    models: ClassVar[dict[str, ModelSpec]] = {
        BASELINE: ModelSpec(
            family="historical_mean",
            hyperparameters=(),
            build=lambda _: HistoricalMeanRegressor(
                keys=(GoldColumns.STATION_ID, GoldColumns.HOUR)
            ),
        ),
        "rf": ModelSpec(
            family="random_forest",
            hyperparameters=("n_estimators", "max_depth"),
            build=lambda params: RandomForestRegressor(**params, random_state=SEED, n_jobs=-1),
        ),
        "hgb": ModelSpec(
            family="hist_gradient_boosting",
            hyperparameters=("n_estimators", "max_depth"),
            build=lambda params: HistGradientBoostingRegressor(
                max_iter=params["n_estimators"], max_depth=params["max_depth"], random_state=SEED
            ),
        ),
    }

    def prepare(self, gold: pd.DataFrame) -> pd.DataFrame:
        return gold

    def search_space(self, model: str) -> list[Hyperparameters]:
        return tree_search_space()

    def metrics(self, y_true: pd.Series, y_pred, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_MAE": float(mean_absolute_error(y_true, y_pred)),
            f"{prefix}_RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            f"{prefix}_R2": float(r2_score(y_true, y_pred)),
        }

    def figures(
        self, test: pd.DataFrame, test_pred, estimator: BaseEstimator
    ) -> dict[str, Figure]:
        frame = pd.DataFrame(
            {"actual": test[self.target_column].to_numpy(), "predicted": test_pred}
        )
        return {
            "actual_vs_predicted.png": _actual_vs_predicted(frame),
            "residuals_by_hour.png": _residuals_by_hour(
                frame.assign(hour=test[GoldColumns.HOUR].to_numpy())
            ),
        }


def tree_search_space() -> list[Hyperparameters]:
    """Return the `n_estimators` by `max_depth` grid shared by the P1 and P2 searches."""
    return [
        {"n_estimators": n_estimators, "max_depth": max_depth}
        for n_estimators, max_depth in itertools.product(
            MODELING_SETTINGS.SEARCH_N_ESTIMATORS, MODELING_SETTINGS.SEARCH_MAX_DEPTH
        )
    ]


def _actual_vs_predicted(frame: pd.DataFrame) -> Figure:
    sample_size = min(MODELING_SETTINGS.PLOT_SAMPLE_SIZE, len(frame))
    sample = frame.sample(n=sample_size, random_state=SEED)
    top = float(sample[["actual", "predicted"]].to_numpy().max())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sample["actual"], sample["predicted"], alpha=0.4, s=10)
    ax.plot([0, top], [0, top], color="gray", linestyle="--", label="perfect prediction")
    ax.set(xlabel="Actual departures", ylabel="Predicted departures", title="Actual vs predicted")
    ax.legend()
    fig.tight_layout()
    return fig


def _residuals_by_hour(frame: pd.DataFrame) -> Figure:
    residuals = frame["actual"] - frame["predicted"]
    by_hour = residuals.groupby(frame["hour"]).agg(["mean", lambda values: values.abs().mean()])
    by_hour.columns = ["mean residual", "mean absolute residual"]

    fig, ax = plt.subplots(figsize=(7, 4))
    by_hour.plot(ax=ax, marker="o")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(xlabel="Hour", ylabel="Departures", title="Residuals by hour (actual - predicted)")
    fig.tight_layout()
    return fig
