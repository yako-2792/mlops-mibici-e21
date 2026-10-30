"""Validate the modeling data with DeepChecks: integrity of each frame and drift between periods.

Every question is validated on the same frame its models train on, built by
`Question.prepare`. P1 and P2 are also compared across the temporal split the models use,
from January through September against October through December.

DeepChecks 0.19 needs older NumPy, pandas, and scikit-learn than the project, so this
module runs in the isolated environment of `validation/`:

    uv run --project validation python -m mibici_e21.validation

Gating checks carry conditions, and any failed condition or check error ends the run with
exit status 1. Monitoring checks carry no conditions: they appear in the HTML reports under
`reports/data-validation/` for review, but never fail the run.
"""

from dataclasses import dataclass
import sys

from deepchecks.core import SuiteResult
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    ConflictingLabels,
    DataDuplicates,
    DateTrainTestLeakageDuplicates,
    DateTrainTestLeakageOverlap,
    FeatureDrift,
    FeatureFeatureCorrelation,
    IsSingleValue,
    LabelDrift,
    MixedDataTypes,
    MultivariateDrift,
    NewCategoryTrainTest,
    NewLabelTrainTest,
    OutlierSampleDetection,
    PercentOfNulls,
    TrainTestSamplesMix,
)
from loguru import logger
import pandas as pd

from mibici_e21.config import MODELING_SETTINGS, REPORTS_DIR
from mibici_e21.features import load_gold_demand
from mibici_e21.modeling.base import Question, SupervisedQuestion, temporal_split
from mibici_e21.modeling.questions import QUESTIONS
from mibici_e21.schemas.columns import GoldColumns

REPORT_DIR = REPORTS_DIR / "data-validation"
LABEL_TYPES = {"regression": "regression", "classification": "multiclass"}


@dataclass(frozen=True)
class SuiteOutcome:
    """Result of one suite on one question.

    - `name`: report name, such as `p1_integrity`.
    - `failed_checks`: names of the gating checks whose conditions failed or that errored.
    """

    name: str
    failed_checks: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failed_checks


def integrity_suite(question: Question) -> Suite:
    """Checks on a single frame: missing values, types, duplicates, and degenerate columns.

    Gating: no missing values, no mixed data types, no duplicated rows, no single-value
    column, and, for a classification label, no identical rows with different labels.
    Monitoring: feature-feature correlation and outlier samples.
    """
    checks = [
        PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0),
        MixedDataTypes().add_condition_rare_type_ratio_not_in_range(),
        DataDuplicates().add_condition_ratio_less_or_equal(0),
        IsSingleValue().add_condition_not_single_value(),
    ]
    if question.task == "classification":
        checks.append(
            ConflictingLabels().add_condition_ratio_of_conflicting_labels_less_or_equal(0)
        )
    checks += [FeatureFeatureCorrelation(), OutlierSampleDetection()]
    return Suite(f"{question.key} data integrity", *checks)


def train_test_suite(question: SupervisedQuestion) -> Suite:
    """Checks between the training months and the test months of the temporal split.

    Gating: feature, label, and multivariate drift below the DeepChecks default thresholds,
    no dates shared between train and test, and, for classification, no label that only
    appears in test. Monitoring: stations first seen in test, which the models handle with
    their fallback, and feature rows repeated across periods, which a small feature space
    such as station, hour, and weekday produces by design.
    """
    checks = [
        FeatureDrift().add_condition_drift_score_less_than(),
        LabelDrift().add_condition_drift_score_less_than(),
        MultivariateDrift().add_condition_overall_drift_value_less_than(),
        DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_less_or_equal(0),
        DateTrainTestLeakageOverlap().add_condition_leakage_ratio_less_or_equal(0),
    ]
    if question.task == "classification":
        checks.append(NewLabelTrainTest().add_condition_new_labels_number_less_or_equal(0))
    checks += [NewCategoryTrainTest(), TrainTestSamplesMix()]
    return Suite(f"{question.key} train-test validation", *checks)


def to_dataset(question: Question, frame: pd.DataFrame) -> Dataset:
    """Wrap a modeling frame in a DeepChecks dataset with its label, date, and categories.

    Only the columns the models read are kept, plus the label and the date. Column names
    become plain strings, as scikit-learn rejects the `StrEnum` names of `GoldColumns`.
    """
    frame = frame.rename(columns=str)
    if not isinstance(question, SupervisedQuestion):
        return Dataset(frame.drop(columns=[GoldColumns.STATION_ID]), cat_features=[])

    columns = [*question.feature_columns, question.target_column, GoldColumns.DATE]
    categorical = [GoldColumns.STATION_ID]
    return Dataset(
        frame[[str(column) for column in columns]],
        label=str(question.target_column),
        label_type=LABEL_TYPES[question.task],
        cat_features=[str(column) for column in categorical],
        datetime_name=str(GoldColumns.DATE),
    )


def gating_failures(result: SuiteResult) -> tuple[str, ...]:
    """Return the names of the checks that errored or whose conditions did not pass.

    Names come from the check, because result headers are not consistent across checks.
    """
    results = [*result.get_not_passed_checks(), *result.get_not_ran_checks()]
    return tuple(check_result.check.name() for check_result in results)


def validate_question(question: Question, gold: pd.DataFrame, year: int) -> list[SuiteOutcome]:
    """Run the suites of one question, write their HTML reports, and return their outcomes."""
    frame = question.prepare(gold)
    runs = [
        (f"{question.key}_integrity", integrity_suite(question), (to_dataset(question, frame),))
    ]
    if isinstance(question, SupervisedQuestion):
        split = temporal_split(frame, year)
        datasets = (to_dataset(question, split.train_full), to_dataset(question, split.test))
        runs.append((f"{question.key}_train_test", train_test_suite(question), datasets))

    outcomes = []
    for name, suite, datasets in runs:
        result = suite.run(*datasets)
        result.save_as_html(str(REPORT_DIR / f"{name}.html"), as_widget=False)
        outcomes.append(SuiteOutcome(name=name, failed_checks=gating_failures(result)))
    return outcomes


def main() -> None:
    """Validate the data of every question and exit with status 1 if a gating check fails."""
    year = MODELING_SETTINGS.TRAINING_YEAR
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gold = load_gold_demand(year)

    outcomes = [
        outcome
        for question in QUESTIONS.values()
        for outcome in validate_question(question, gold, year)
    ]
    for outcome in outcomes:
        if outcome.passed:
            logger.info(f"PASS {outcome.name}")
        else:
            logger.error(f"FAIL {outcome.name}: {', '.join(outcome.failed_checks)}")
    logger.info(f"HTML reports in {REPORT_DIR}")

    if not all(outcome.passed for outcome in outcomes):
        sys.exit(1)


if __name__ == "__main__":
    main()
