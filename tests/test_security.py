"""The repository exposes no secrets, tracks no data or models, and keeps rider data out of gold."""

from datetime import datetime
import re
import shutil
import subprocess

import pandas as pd
import pytest

from mibici_e21.config import PROJ_ROOT
from mibici_e21.features import _aggregate_station_demand, _extract_trip_features
from mibici_e21.schemas.columns import TripColumns

SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
FORBIDDEN_TRACKED_FILES = re.compile(
    r"(^|/)\.env$|\.(parquet|csv|pkl|pickle|joblib)$|(^|/)mlflow\.db$|(^|/)(mlruns|mlartifacts)/"
)
MUST_BE_IGNORED = (
    ".env",
    "data/raw/mibici-trips-2023.parquet",
    "data/processed/demand_aggregated_2023.parquet",
    "mlflow.db",
    "mlruns/0/meta.yaml",
    "mlartifacts/1/model.pkl",
    "models/model.pkl",
)
RIDER_COLUMNS = {
    TripColumns.TRIP_ID,
    TripColumns.USER_ID,
    TripColumns.GENDER,
    TripColumns.BIRTH_YEAR,
}

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(*args: str) -> str:
    """Run a read-only git command at the project root and return its output."""
    git = shutil.which("git")
    return subprocess.run(
        [git, *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=True
    ).stdout


def test_git_history_contains_no_secrets():
    """No commit on any branch ever added a token or a private key."""
    added_lines = [
        line
        for line in _git("log", "--all", "-p", "--no-color", "--unified=0").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]

    leaks = {
        name: [line[:40] for line in added_lines if pattern.search(line)]
        for name, pattern in SECRET_PATTERNS.items()
    }

    assert {name: lines for name, lines in leaks.items() if lines} == {}


def test_no_data_secrets_or_models_are_tracked():
    """Data files, `.env`, the MLflow store, and serialized models stay out of git."""
    tracked = _git("ls-files").splitlines()

    assert [path for path in tracked if FORBIDDEN_TRACKED_FILES.search(path)] == []


@pytest.mark.parametrize("path", MUST_BE_IGNORED)
def test_gitignore_covers_secrets_data_and_models(path):
    """`git check-ignore` exits with 0 only for an ignored path."""
    result = subprocess.run(
        [shutil.which("git"), "check-ignore", "--quiet", "--no-index", path],
        cwd=PROJ_ROOT,
        check=False,
    )

    assert result.returncode == 0, f"{path} is not gitignored"


def test_rider_data_never_reaches_the_gold_layer():
    """Trip ids, user ids, gender, and birth year are dropped before any gold table."""
    silver = pd.DataFrame(
        {
            TripColumns.TRIP_ID: [1, 2],
            TripColumns.USER_ID: [501, 502],
            TripColumns.GENDER: ["F", "M"],
            TripColumns.BIRTH_YEAR: [1990, 1985],
            TripColumns.START_TIME: [datetime(2023, 3, 1, 8, 0), datetime(2023, 3, 1, 9, 0)],
            TripColumns.END_TIME: [datetime(2023, 3, 1, 8, 20), datetime(2023, 3, 1, 9, 15)],
            TripColumns.ORIGIN_ID: [10, 11],
            TripColumns.DESTINATION_ID: [11, 10],
        }
    )

    for gold in (_extract_trip_features(silver), _aggregate_station_demand(silver)):
        assert RIDER_COLUMNS.isdisjoint(gold.columns)
