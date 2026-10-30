"""Tests for the baselines, the temporal split, the question features, and model selection."""

from datetime import date

import pandas as pd
import pytest
from sklearn.base import clone

from mibici_e21.config import MODELING_SETTINGS
from mibici_e21.modeling.base import split_boundaries, temporal_split
from mibici_e21.modeling.baselines import HistoricalMeanRegressor, HistoricalModeClassifier
from mibici_e21.modeling.clusters import build_station_profiles
from mibici_e21.modeling.demand import DemandQuestion
from mibici_e21.modeling.register import rank_candidates
from mibici_e21.modeling.risk import (
    BALANCED,
    POSSIBLE_EMPTY,
    POSSIBLE_FULL,
    RiskQuestion,
    build_station_days,
    label_risk,
)
from mibici_e21.modeling.train import optional_int, parse_args
from mibici_e21.schemas.columns import GoldColumns

THRESHOLD = MODELING_SETTINGS.BALANCE_THRESHOLD


def _gold(rows: list[tuple[str, int, int, int, int]]) -> pd.DataFrame:
    """Build gold demand from `(date, hour, station, departures, arrivals)` rows."""
    frame = pd.DataFrame(
        rows,
        columns=[
            GoldColumns.DATE,
            GoldColumns.HOUR,
            GoldColumns.STATION_ID,
            GoldColumns.DEPARTURES_COUNT,
            GoldColumns.ARRIVALS_COUNT,
        ],
    )
    frame[GoldColumns.DATE] = pd.to_datetime(frame[GoldColumns.DATE])
    return frame


def test_historical_mean_regressor_falls_back_to_the_global_mean():
    """A known station-hour gets its own mean, and an unseen one gets the global mean."""
    X = pd.DataFrame({"station": [1, 1, 2], "hour": [8, 8, 9]})
    model = HistoricalMeanRegressor(keys=("station", "hour")).fit(X, [2.0, 4.0, 6.0])

    predictions = model.predict(pd.DataFrame({"station": [1, 3], "hour": [8, 8]}))

    assert predictions.tolist() == [3.0, 4.0]


def test_historical_mode_classifier_falls_back_to_the_global_mode():
    """A known station gets its own mode, and an unseen one gets the overall mode."""
    X = pd.DataFrame({"station": [1, 1, 2, 2, 2]})
    y = ["full", "full", "empty", "empty", "empty"]
    model = HistoricalModeClassifier(key="station").fit(X, y)

    predictions = model.predict(pd.DataFrame({"station": [1, 2, 9]}))

    assert predictions.tolist() == ["full", "empty", "empty"]
    assert sorted(model.classes_) == ["empty", "full"]


def test_baselines_are_cloneable_estimators():
    """`clone` keeps the constructor params, so the search and the split can refit them."""
    assert clone(HistoricalMeanRegressor(keys=("a", "b"))).keys == ("a", "b")
    assert clone(HistoricalModeClassifier(key="a")).key == "a"


def test_temporal_split_uses_the_configured_months_without_overlap():
    """Validation starts and test starts on the first day of their configured months."""
    year = 2023
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    frame = pd.DataFrame({GoldColumns.DATE: days})
    val_start, test_start = split_boundaries(year)

    split = temporal_split(frame, year)

    assert val_start == date(year, MODELING_SETTINGS.VAL_START_MONTH, 1)
    assert split.train[GoldColumns.DATE].max() < pd.Timestamp(val_start)
    assert split.val[GoldColumns.DATE].min() == pd.Timestamp(val_start)
    assert split.test[GoldColumns.DATE].min() == pd.Timestamp(test_start)
    assert len(split.train) + len(split.val) == len(split.train_full)
    assert len(split.train_full) + len(split.test) == len(frame)


def test_label_risk_uses_the_balance_threshold():
    """Balances at the threshold count as risk, and smaller ones as balanced."""
    labels = label_risk(pd.Series([-THRESHOLD, -THRESHOLD + 1, 0, THRESHOLD - 1, THRESHOLD]))

    assert labels.tolist() == [POSSIBLE_EMPTY, BALANCED, BALANCED, BALANCED, POSSIBLE_FULL]


def test_station_day_features_only_look_at_earlier_days():
    """Lags and rolling means of a day ignore that day, so the label does not leak."""
    gold = _gold(
        [
            ("2023-01-01", 8, 1, 10, 0),
            ("2023-01-02", 8, 1, 0, 6),
            ("2023-01-08", 8, 1, 3, 3),
        ]
    )

    daily = build_station_days(gold).set_index(GoldColumns.DATE)

    second = daily.loc["2023-01-02"]
    assert second["balance"] == 6
    assert second["balance_lag1"] == -10
    assert second["balance_mean7d"] == -10
    eighth = daily.loc["2023-01-08"]
    assert eighth["balance_lag1"] == 0
    assert eighth["balance_lag7"] == -10
    assert eighth["balance_mean7d"] == -2
    first = daily.loc["2023-01-01"]
    assert first["balance_lag1"] == 0
    assert first["balance_mean7d"] == 0


def test_risk_features_exist_in_the_station_day_frame():
    """Every feature the P2 estimators read is built by `prepare`."""
    gold = _gold([("2023-01-01", 8, 1, 1, 1)])

    daily = RiskQuestion().prepare(gold)

    assert set(RiskQuestion.feature_columns) <= set(daily.columns)


def test_station_profiles_count_only_peak_hour_departures_as_peak():
    """Peak departures add up the configured peak hours only."""
    peak_hour, off_hour = MODELING_SETTINGS.PEAK_HOURS[0], 3
    gold = _gold(
        [
            ("2023-01-01", peak_hour, 1, 4, 1),
            ("2023-01-01", off_hour, 1, 5, 2),
            ("2023-01-01", off_hour, 2, 7, 0),
        ]
    )

    profiles = build_station_profiles(gold).set_index(GoldColumns.STATION_ID)

    assert profiles.loc[1].tolist() == [9, 3, 4]
    assert profiles.loc[2].tolist() == [7, 0, 0]


def test_build_estimator_names_the_missing_hyperparameters():
    """A candidate without its hyperparameters fails with the flags to pass."""
    with pytest.raises(ValueError, match="-P n_estimators=... -P max_depth=..."):
        DemandQuestion().build_estimator("rf", {"n_estimators": None, "max_depth": None})


def test_build_estimator_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="Unknown p1 model 'svm'"):
        DemandQuestion().build_estimator("svm", {})


@pytest.mark.parametrize(
    ("question", "baseline", "candidates", "expected"),
    [
        (DemandQuestion(), 1.4, {"a": 1.3, "b": 1.5, "c": 1.2}, ["c", "a"]),
        (RiskQuestion(), 0.64, {"a": 0.66, "b": 0.60, "c": 0.65}, ["a", "c"]),
        (RiskQuestion(), 0.64, {"a": 0.64}, []),
    ],
)
def test_rank_candidates_keeps_only_those_beating_the_baseline(
    question, baseline, candidates, expected
):
    """Candidates are ranked by the selection metric, in its own direction, past the baseline."""
    assert rank_candidates(question, baseline, candidates) == expected


def test_train_arguments_treat_empty_hyperparameters_as_unset():
    """`mlflow run` passes an empty string for a hyperparameter left at its default."""
    args = parse_args(
        ["--question", "p1", "--model", "baseline", "--n_estimators", "", "--search", "false"]
    )

    assert args.n_estimators is None
    assert args.search is False
    assert optional_int("300") == 300
