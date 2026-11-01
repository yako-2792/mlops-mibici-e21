"""Training is reproducible: the configured seed fixes every prediction and metric.

Forests train their trees in parallel (`n_jobs=-1`), so the final average is summed in an
order that can change the last floating-point bit. Results are therefore compared with a
relative tolerance of 1e-12, far below any difference that could change a decision.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mibici_e21.config import MODELING_SETTINGS
from mibici_e21.modeling.base import Evaluation, Question
from mibici_e21.modeling.clusters import ClusterQuestion, build_station_profiles
from mibici_e21.modeling.demand import DemandQuestion
from mibici_e21.modeling.risk import RiskQuestion
from mibici_e21.schemas.columns import GoldColumns, StationColumns

YEAR = MODELING_SETTINGS.TRAINING_YEAR
STATIONS = 6
HOURS = (7, 8, 13, 18)
SMALL_FOREST = {"n_estimators": 10, "max_depth": 6}
TIMING_METRICS = ("train_seconds", "predict_seconds")
TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def gold() -> pd.DataFrame:
    """One synthetic year of gold demand, drawn from a fixed random generator."""
    rng = np.random.default_rng(0)
    days = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D")
    grid = pd.MultiIndex.from_product(
        [days, HOURS, range(1, STATIONS + 1)],
        names=[GoldColumns.DATE, GoldColumns.HOUR, GoldColumns.STATION_ID],
    ).to_frame(index=False)
    station_rate = grid[GoldColumns.STATION_ID].to_numpy()
    grid[GoldColumns.DEPARTURES_COUNT] = rng.poisson(station_rate)
    grid[GoldColumns.ARRIVALS_COUNT] = rng.poisson(STATIONS + 1 - station_rate)
    grid[GoldColumns.DAY_OF_WEEK] = grid[GoldColumns.DATE].dt.dayofweek
    grid[GoldColumns.IS_WEEKEND] = (grid[GoldColumns.DAY_OF_WEEK] >= 5).astype(int)
    return grid


@pytest.fixture(scope="module")
def station_profiles(gold: pd.DataFrame) -> pd.DataFrame:
    """P3 input built from the synthetic year, with fixed coordinates instead of the catalog."""
    return build_station_profiles(gold).assign(
        **{StationColumns.LATITUDE: 20.67, StationColumns.LONGITUDE: -103.35}
    )


def _evaluate(question: Question, data: pd.DataFrame, model: str, hyperparameters: dict):
    """Train a fresh estimator and return its evaluation, closing the figures it drew."""
    estimator = question.build_estimator(model, hyperparameters)
    evaluation = question.evaluate(data, estimator, YEAR)
    for figure in evaluation.figures.values():
        plt.close(figure)
    return evaluation


def _scores(evaluation: Evaluation) -> dict[str, float]:
    """Return the metrics of an evaluation without the wall-clock timings."""
    return {
        name: value for name, value in evaluation.metrics.items() if name not in TIMING_METRICS
    }


def _assert_same_predictions(first: np.ndarray, second: np.ndarray) -> None:
    """Class labels must match exactly, and numeric predictions up to `TOLERANCE`."""
    if first.dtype.kind in "fc":
        np.testing.assert_allclose(first, second, rtol=TOLERANCE, atol=0)
    else:
        np.testing.assert_array_equal(first, second)


def _raw_output(estimator, features: pd.DataFrame) -> np.ndarray:
    """Return class probabilities for a classifier and predictions for a regressor.

    Probabilities expose a seed change even when the predicted classes happen to agree.
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)
    return estimator.predict(features)


def _predict(evaluation: Evaluation, question: Question, data: pd.DataFrame) -> np.ndarray:
    return evaluation.estimator.predict(data[list(question.feature_columns)])


SEEDED_CASES = [
    pytest.param(DemandQuestion(), "rf", SMALL_FOREST, id="p1-random-forest"),
    pytest.param(DemandQuestion(), "hgb", SMALL_FOREST, id="p1-gradient-boosting"),
    pytest.param(RiskQuestion(), "rf", SMALL_FOREST, id="p2-random-forest"),
    pytest.param(RiskQuestion(), "hgb", SMALL_FOREST, id="p2-gradient-boosting"),
    pytest.param(ClusterQuestion(), "kmeans", {"n_clusters": 3}, id="p3-kmeans"),
]


def _data_for(question: Question, gold: pd.DataFrame, station_profiles: pd.DataFrame):
    return station_profiles if isinstance(question, ClusterQuestion) else question.prepare(gold)


@pytest.mark.parametrize(("question", "model", "hyperparameters"), SEEDED_CASES)
def test_same_seed_gives_identical_predictions_and_metrics(
    question, model, hyperparameters, gold, station_profiles
):
    """Two trainings with the configured seed predict and score exactly the same."""
    data = _data_for(question, gold, station_profiles)

    first = _evaluate(question, data, model, hyperparameters)
    second = _evaluate(question, data, model, hyperparameters)

    _assert_same_predictions(_predict(first, question, data), _predict(second, question, data))
    assert _scores(first) == pytest.approx(_scores(second), rel=TOLERANCE)


@pytest.mark.parametrize(
    ("question", "model", "seed_parameter"),
    [
        pytest.param(DemandQuestion(), "rf", "random_state", id="p1-random-forest"),
        pytest.param(RiskQuestion(), "rf", "random_state", id="p2-random-forest"),
    ],
)
def test_a_different_seed_changes_the_model(question, model, seed_parameter, gold):
    """Changing only the seed changes the forest, so the seed really reaches the estimator."""
    data = question.prepare(gold)
    features = data[list(question.feature_columns)]
    target = data[question.target_column]

    seeded = question.build_estimator(model, SMALL_FOREST).fit(features, target)
    reseeded = question.build_estimator(model, SMALL_FOREST)
    reseeded.set_params(**{seed_parameter: MODELING_SETTINGS.SEED + 1}).fit(features, target)

    assert seeded.random_state == MODELING_SETTINGS.SEED
    assert not np.array_equal(_raw_output(seeded, features), _raw_output(reseeded, features))


@pytest.mark.parametrize(
    "question", [DemandQuestion(), RiskQuestion(), ClusterQuestion()], ids=["p1", "p2", "p3"]
)
def test_baselines_are_deterministic(question, gold, station_profiles):
    """Baselines have no randomness of their own, so two fits are identical too."""
    data = _data_for(question, gold, station_profiles)

    first = _evaluate(question, data, "baseline", {})
    second = _evaluate(question, data, "baseline", {})

    _assert_same_predictions(_predict(first, question, data), _predict(second, question, data))
    assert _scores(first) == pytest.approx(_scores(second), rel=TOLERANCE)
