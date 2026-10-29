"""Exploratory figures of one year of gold data, written to `reports/figures`.

python -m mibici_e21.plots --year 2023
"""

import argparse

import matplotlib

matplotlib.use("Agg")

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mibici_e21.config import FEATURE_STORE_SETTINGS, FIGURES_DIR, MODELING_SETTINGS
from mibici_e21.schemas.columns import GoldColumns, TripColumns

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
TOP_STATIONS = 10


def plot_mean_departures_by_hour(demand: pd.DataFrame) -> plt.Figure:
    by_hour = demand.groupby(GoldColumns.HOUR)[GoldColumns.DEPARTURES_COUNT].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=by_hour, x=GoldColumns.HOUR, y=GoldColumns.DEPARTURES_COUNT, ax=ax)
    ax.set(title="Mean departures per station by hour", xlabel="Hour", ylabel="Departures")
    return fig


def plot_trips_by_weekday(trips: pd.DataFrame) -> plt.Figure:
    counts = trips[GoldColumns.DAY_OF_WEEK].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=[WEEKDAYS[day] for day in counts.index], y=counts.to_numpy(), ax=ax)
    ax.set(title="Trips by weekday", ylabel="Trips")
    return fig


def plot_busiest_stations(trips: pd.DataFrame) -> plt.Figure:
    top = trips[TripColumns.ORIGIN_ID].value_counts().head(TOP_STATIONS)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top.index.astype(str), y=top.to_numpy(), order=top.index.astype(str), ax=ax)
    ax.set(title=f"Top {TOP_STATIONS} origin stations", xlabel="Station id", ylabel="Trips")
    return fig


def write_figures(year: int) -> None:
    """Write the exploratory figures of a year as PNG files in `reports/figures`."""
    directory = FEATURE_STORE_SETTINGS.GOLD_DIR
    demand = pd.read_parquet(directory / f"demand_aggregated_{year}.parquet")
    trips = pd.read_parquet(directory / f"features_{year}.parquet")
    figures = {
        f"departures_by_hour_{year}.png": plot_mean_departures_by_hour(demand),
        f"trips_by_weekday_{year}.png": plot_trips_by_weekday(trips),
        f"busiest_stations_{year}.png": plot_busiest_stations(trips),
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.tight_layout()
        figure.savefig(FIGURES_DIR / name)
        plt.close(figure)
        logger.info(f"Wrote {FIGURES_DIR / name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write exploratory figures of one year.")
    parser.add_argument("--year", type=int, default=MODELING_SETTINGS.TRAINING_YEAR)
    write_figures(parser.parse_args(argv).year)


if __name__ == "__main__":
    main()
