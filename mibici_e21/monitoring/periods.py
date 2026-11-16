"""Date periods of the monitoring layer and the gold demand that falls inside them.

The drift report, the retraining, and the A/B test read the same periods from
`MONITORING_SETTINGS`, so the report, the model, and the comparison never disagree on dates.
"""

from dataclasses import dataclass
from datetime import date, timedelta

from loguru import logger
import pandas as pd

from mibici_e21.config import MONITORING_SETTINGS
from mibici_e21.dataset import data_manifest_path, ingest_year, read_data_manifest
from mibici_e21.features import load_gold_demand
from mibici_e21.schemas.columns import GoldColumns


@dataclass(frozen=True)
class Period:
    """A half-open range of dates: `start` is included and `end` is not.

    - `name`: short name used in run names, file names, and tables, such as `reference`.
    - `start`: first date of the period.
    - `end`: first date after the period.
    """

    name: str
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"Period {self.name} ends on {self.end}, before it starts.")

    @property
    def years(self) -> list[int]:
        """Calendar years with at least one day in the period."""
        return list(range(self.start.year, self.last_day.year + 1))

    @property
    def last_day(self) -> date:
        """Last date included in the period."""
        return self.end - timedelta(days=1)

    @property
    def label(self) -> str:
        """Readable range with both ends included, such as `2023-01-01 to 2023-09-30`."""
        return f"{self.start.isoformat()} to {self.last_day.isoformat()}"

    def contains(self, dates: pd.Series) -> pd.Series:
        """Return a mask of the dates that fall inside the period."""
        return (dates >= pd.Timestamp(self.start)) & (dates < pd.Timestamp(self.end))

    def quarters(self) -> list["Period"]:
        """Split the period into calendar quarters, clipped to its own start and end."""
        quarters = []
        start = self.start
        while start < self.end:
            first_month = 3 * ((start.month - 1) // 3) + 1
            next_month = first_month + 3
            quarter_end = (
                date(start.year + 1, 1, 1) if next_month > 12 else date(start.year, next_month, 1)
            )
            end = min(quarter_end, self.end)
            quarter = (start.month - 1) // 3 + 1
            quarters.append(Period(f"{start.year}Q{quarter}", start, end))
            start = end
        return quarters


REFERENCE = Period("reference", *MONITORING_SETTINGS.REFERENCE)
CURRENT = Period("current", *MONITORING_SETTINGS.CURRENT)
RETRAIN = Period("retrain", *MONITORING_SETTINGS.RETRAIN)
EVALUATION = Period("evaluation", *MONITORING_SETTINGS.EVALUATION)


def ensure_gold_year(year: int) -> dict:
    """Return the data manifest of a year, ingesting the year first when it is missing.

    The ingestion downloads the pinned dataset revision, so a clean clone builds the same
    gold files. Raises `ValueError` when the gold demand changed after its ingestion.
    """
    if not data_manifest_path(year).exists():
        logger.info(f"No gold data for {year} yet. Ingesting it from the pinned revision.")
        ingest_year(year)
    return read_data_manifest(year)


def load_period(period: Period) -> tuple[pd.DataFrame, list[dict]]:
    """Load the gold demand of a period, ingesting any year that is not on disk yet.

    Returns the rows inside the period, sorted by date, hour, and station, and the data
    manifest of every year read.
    """
    manifests = [ensure_gold_year(year) for year in period.years]
    frames = [load_gold_demand(year) for year in period.years]
    gold = pd.concat(frames, ignore_index=True)
    inside = gold[period.contains(gold[GoldColumns.DATE])]
    keys = [GoldColumns.DATE, GoldColumns.HOUR, GoldColumns.STATION_ID]
    return inside.sort_values(keys, ignore_index=True), manifests


def unseen_station_share(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    """Share of the rows of `current` whose station never appears in `reference`."""
    known = reference[GoldColumns.STATION_ID].unique()
    return float((~current[GoldColumns.STATION_ID].isin(known)).mean())
