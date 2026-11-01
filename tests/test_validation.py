"""The DeepChecks gates pass clean data and catch each defect they are meant to catch.

DeepChecks lives in the isolated environment of `validation/`, so these tests run there:

    uv run --project validation pytest tests/test_validation.py

In the project environment they are skipped.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("deepchecks")

from mibici_e21.config import MODELING_SETTINGS
from mibici_e21.modeling.base import temporal_split
from mibici_e21.modeling.demand import DemandQuestion
from mibici_e21.modeling.risk import RISK, RiskQuestion
from mibici_e21.schemas.columns import GoldColumns
from mibici_e21.validation import (
    gating_failures,
    integrity_suite,
    to_dataset,
    train_test_suite,
)

YEAR = MODELING_SETTINGS.TRAINING_YEAR
DEMAND = DemandQuestion()
RISK_QUESTION = RiskQuestion()


@pytest.fixture(scope="module")
def gold() -> pd.DataFrame:
    """One synthetic year of gold demand for six stations, drawn from a fixed generator."""
    rng = np.random.default_rng(0)
    days = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D")
    frame = pd.MultiIndex.from_product(
        [days, (7, 8, 13, 18), range(1, 7)],
        names=[GoldColumns.DATE, GoldColumns.HOUR, GoldColumns.STATION_ID],
    ).to_frame(index=False)
    rate = frame[GoldColumns.STATION_ID].to_numpy()
    frame[GoldColumns.DEPARTURES_COUNT] = rng.poisson(rate)
    frame[GoldColumns.ARRIVALS_COUNT] = rng.poisson(7 - rate)
    frame[GoldColumns.DAY_OF_WEEK] = frame[GoldColumns.DATE].dt.dayofweek
    frame[GoldColumns.IS_WEEKEND] = (frame[GoldColumns.DAY_OF_WEEK] >= 5).astype(int)
    return frame


def _integrity_failures(question, frame: pd.DataFrame) -> tuple[str, ...]:
    return gating_failures(integrity_suite(question).run(to_dataset(question, frame)))


def _train_test_failures(question, train: pd.DataFrame, test: pd.DataFrame) -> tuple[str, ...]:
    suite = train_test_suite(question)
    return gating_failures(suite.run(to_dataset(question, train), to_dataset(question, test)))


def test_clean_data_passes_every_gate(gold):
    """Synthetic data without defects passes integrity and train-test validation."""
    split = temporal_split(gold, YEAR)

    assert _integrity_failures(DEMAND, gold) == ()
    assert _train_test_failures(DEMAND, split.train_full, split.test) == ()


def test_missing_values_fail_the_integrity_gate(gold):
    with_nulls = gold.astype({GoldColumns.HOUR: "float"})
    with_nulls.loc[with_nulls.index[:10], GoldColumns.HOUR] = np.nan

    assert "Percent Of Nulls" in _integrity_failures(DEMAND, with_nulls)


def test_duplicated_rows_fail_the_integrity_gate(gold):
    duplicated = pd.concat([gold, gold.head(50)], ignore_index=True)

    assert "Data Duplicates" in _integrity_failures(DEMAND, duplicated)


def test_a_shifted_label_fails_the_drift_gate(gold):
    """Test months with five times the usual departures are flagged as label drift."""
    split = temporal_split(gold, YEAR)
    shifted = split.test.assign(
        **{GoldColumns.DEPARTURES_COUNT: split.test[GoldColumns.DEPARTURES_COUNT] * 5 + 10}
    )

    assert "Label Drift" in _train_test_failures(DEMAND, split.train_full, shifted)


def test_a_class_only_seen_in_test_fails_the_new_label_gate(gold):
    station_days = RISK_QUESTION.prepare(gold)
    split = temporal_split(station_days, YEAR)
    test = split.test.copy()
    test.loc[test.index[:20], RISK] = "unknown"

    assert "New Label Train Test" in _train_test_failures(RISK_QUESTION, split.train_full, test)
