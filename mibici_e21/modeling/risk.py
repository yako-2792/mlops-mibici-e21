"""P2: daily risk that a station runs empty or full, a classification per station-day."""

from typing import ClassVar

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from mibici_e21.config import MODELING_SETTINGS
from mibici_e21.modeling.base import (
    BASELINE,
    Hyperparameters,
    ModelSpec,
    SupervisedQuestion,
)
from mibici_e21.modeling.baselines import HistoricalModeClassifier
from mibici_e21.modeling.demand import tree_search_space
from mibici_e21.schemas.columns import GoldColumns

SEED = MODELING_SETTINGS.SEED
RISK = "risk"
BALANCE = "balance"
POSSIBLE_EMPTY = "possible_empty"
BALANCED = "balanced"
POSSIBLE_FULL = "possible_full"
RISK_CLASSES = (POSSIBLE_EMPTY, BALANCED, POSSIBLE_FULL)
ROLLING_WINDOWS = (7, 28)

LAG_FEATURES = tuple(
    f"{column}_lag{lag}"
    for lag in MODELING_SETTINGS.RISK_LAGS
    for column in (BALANCE, GoldColumns.DEPARTURES_COUNT)
)
ROLLING_FEATURES = tuple(f"{BALANCE}_mean{window}d" for window in ROLLING_WINDOWS)


class RiskQuestion(SupervisedQuestion):
    """Which stations will end the day short of bikes or short of docks.

    A station-day is `possible_empty` when departures exceed arrivals by at least
    `BALANCE_THRESHOLD`, `possible_full` in the opposite case, and `balanced` otherwise.
    Features only look at earlier days, because the label is built from the same day.
    Macro F1 is the selection metric, because missing either risk class costs a truck trip.
    """

    key = "p2"
    task = "classification"
    registered_model_suffix = "p2-riesgo"
    description = (
        "P2: daily risk per station of running empty or full, used to plan rebalancing. "
        "Input: station, calendar, and balance of earlier days. "
        "Output: possible_empty, balanced, or possible_full."
    )
    selection_metric = "val_f1_macro"
    greater_is_better = True
    target_column = RISK
    feature_columns = (
        GoldColumns.STATION_ID,
        GoldColumns.DAY_OF_WEEK,
        GoldColumns.IS_WEEKEND,
        *LAG_FEATURES,
        *ROLLING_FEATURES,
    )
    key_columns = (GoldColumns.STATION_ID, GoldColumns.DATE)
    prediction_column = "predicted_risk"
    models: ClassVar[dict[str, ModelSpec]] = {
        BASELINE: ModelSpec(
            family="historical_mode",
            hyperparameters=(),
            build=lambda _: HistoricalModeClassifier(key=GoldColumns.STATION_ID),
        ),
        "rf": ModelSpec(
            family="random_forest",
            hyperparameters=("n_estimators", "max_depth"),
            build=lambda params: RandomForestClassifier(
                **params, class_weight="balanced", random_state=SEED, n_jobs=-1
            ),
        ),
        "hgb": ModelSpec(
            family="hist_gradient_boosting",
            hyperparameters=("n_estimators", "max_depth"),
            build=lambda params: HistGradientBoostingClassifier(
                max_iter=params["n_estimators"],
                max_depth=params["max_depth"],
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    }

    def prepare(self, gold: pd.DataFrame) -> pd.DataFrame:
        return build_station_days(gold)

    def search_space(self, model: str) -> list[Hyperparameters]:
        return tree_search_space()

    def metrics(self, y_true: pd.Series, y_pred, prefix: str) -> dict[str, float]:
        recalls = recall_score(
            y_true, y_pred, labels=list(RISK_CLASSES), average=None, zero_division=0
        )
        return {
            f"{prefix}_precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            f"{prefix}_recall_macro": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            f"{prefix}_f1_macro": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            f"{prefix}_recall_{POSSIBLE_EMPTY}": float(recalls[0]),
            f"{prefix}_recall_{POSSIBLE_FULL}": float(recalls[2]),
        }

    def figures(
        self, test: pd.DataFrame, test_pred, estimator: BaseEstimator
    ) -> dict[str, Figure]:
        figures = {"confusion_matrix.png": _confusion_matrix(test[RISK], test_pred)}
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(test[list(self.feature_columns)])
            figures["precision_recall.png"] = _precision_recall(
                test[RISK], probabilities, list(estimator.classes_)
            )
        return figures


def build_station_days(gold: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the gold demand to one row per station and day, with label and features.

    Lag features hold the value of the same station `k` calendar days before, and zero when
    the station had no trips that day. Rolling features hold the mean balance of the
    station over its previous 7 and 28 days with trips, excluding the current day.
    """
    keys = [GoldColumns.STATION_ID, GoldColumns.DATE]
    daily = (
        gold.groupby(keys)[[GoldColumns.DEPARTURES_COUNT, GoldColumns.ARRIVALS_COUNT]]
        .sum()
        .reset_index()
    )
    daily[GoldColumns.DAY_OF_WEEK] = daily[GoldColumns.DATE].dt.dayofweek
    daily[GoldColumns.IS_WEEKEND] = (daily[GoldColumns.DAY_OF_WEEK] >= 5).astype(int)
    daily[BALANCE] = daily[GoldColumns.ARRIVALS_COUNT] - daily[GoldColumns.DEPARTURES_COUNT]
    daily[RISK] = label_risk(daily[BALANCE])

    for lag in MODELING_SETTINGS.RISK_LAGS:
        lagged = daily[[*keys, BALANCE, GoldColumns.DEPARTURES_COUNT]].assign(
            **{GoldColumns.DATE: daily[GoldColumns.DATE] + pd.Timedelta(days=lag)}
        )
        daily = daily.merge(lagged, on=keys, how="left", suffixes=("", f"_lag{lag}"))

    previous_balance = daily.sort_values(keys).groupby(GoldColumns.STATION_ID)[BALANCE].shift(1)
    for window in ROLLING_WINDOWS:
        daily[f"{BALANCE}_mean{window}d"] = (
            previous_balance.groupby(daily[GoldColumns.STATION_ID])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    feature_columns = [*LAG_FEATURES, *ROLLING_FEATURES]
    daily[feature_columns] = daily[feature_columns].fillna(0)
    return daily


def label_risk(balance: pd.Series) -> np.ndarray:
    """Label each daily balance as `possible_empty`, `balanced`, or `possible_full`."""
    threshold = MODELING_SETTINGS.BALANCE_THRESHOLD
    return np.select(
        [balance <= -threshold, balance >= threshold],
        [POSSIBLE_EMPTY, POSSIBLE_FULL],
        default=BALANCED,
    )


def _confusion_matrix(y_true: pd.Series, y_pred) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=list(RISK_CLASSES), normalize="true", ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix (rows sum to 1)")
    fig.tight_layout()
    return fig


def _precision_recall(y_true: pd.Series, probabilities: np.ndarray, classes: list[str]) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    for index, label in enumerate(classes):
        PrecisionRecallDisplay.from_predictions(
            y_true == label, probabilities[:, index], name=label, ax=ax
        )
    ax.set_title("Precision-recall by class")
    fig.tight_layout()
    return fig
