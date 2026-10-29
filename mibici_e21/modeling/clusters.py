"""P3: groups of stations with a similar usage profile, a clustering over the year."""

import time
from typing import ClassVar

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici_e21.config import MODELING_SETTINGS
from mibici_e21.features import load_station_locations
from mibici_e21.modeling.base import BASELINE, Evaluation, Hyperparameters, ModelSpec, Question
from mibici_e21.schemas.columns import GoldColumns, StationColumns

SEED = MODELING_SETTINGS.SEED
DEPARTURES = "departures"
ARRIVALS = "arrivals"
PEAK_DEPARTURES = "peak_departures"
STABILITY_SEEDS = 3


def build_kmeans(n_clusters: int, seed: int = SEED) -> Pipeline:
    """Return a scaler followed by KMeans, so raw station totals can be scored directly."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    n_init=MODELING_SETTINGS.KMEANS_N_INIT,
                    random_state=seed,
                ),
            ),
        ]
    )


class ClusterQuestion(Question):
    """Which stations behave alike, to plan rebalancing by group instead of one by one.

    Stations are described by their yearly departures, arrivals, and peak-hour departures.
    There is no target, so every run fits on the whole year, as the baseline did.
    Silhouette is the selection metric. Davies-Bouldin, Calinski-Harabasz, and the
    stability across seeds are logged to judge it.
    """

    key = "p3"
    task = "clustering"
    registered_model_suffix = "p3-grupos"
    description = (
        "P3: station groups by usage profile, used to plan rebalancing by group. "
        "Input: yearly departures, arrivals, and peak-hour departures of a station. "
        "Output: group id."
    )
    selection_metric = "silhouette"
    greater_is_better = True
    feature_columns = (DEPARTURES, ARRIVALS, PEAK_DEPARTURES)
    key_columns = (GoldColumns.STATION_ID,)
    prediction_column = "predicted_group"
    models: ClassVar[dict[str, ModelSpec]] = {
        BASELINE: ModelSpec(
            family="kmeans",
            hyperparameters=(),
            build=lambda _: build_kmeans(MODELING_SETTINGS.BASELINE_CLUSTERS),
        ),
        "kmeans": ModelSpec(
            family="kmeans",
            hyperparameters=("n_clusters",),
            build=lambda params: build_kmeans(params["n_clusters"]),
        ),
    }

    def prepare(self, gold: pd.DataFrame) -> pd.DataFrame:
        profiles = build_station_profiles(gold)
        return profiles.merge(load_station_locations(), on=GoldColumns.STATION_ID, how="left")

    def search_space(self, model: str) -> list[Hyperparameters]:
        return [{"n_clusters": k} for k in MODELING_SETTINGS.SEARCH_N_CLUSTERS]

    def evaluate(self, data: pd.DataFrame, estimator: BaseEstimator, year: int) -> Evaluation:
        features = data[list(self.feature_columns)]
        metrics = self.score(data, estimator, year)

        started = time.perf_counter()
        labels = estimator.predict(features)
        metrics["predict_seconds"] = time.perf_counter() - started

        return Evaluation(
            estimator=estimator,
            metrics=metrics,
            figures={
                "clusters.png": _clusters_scatter(features, labels),
                "cluster_profiles.png": _cluster_profiles(features, labels),
                "station_map.png": _station_map(data, labels),
            },
            training_frame=data,
            row_counts={"stations": len(data)},
        )

    def score(self, data: pd.DataFrame, estimator: BaseEstimator, year: int) -> dict[str, float]:
        features = data[list(self.feature_columns)]
        started = time.perf_counter()
        estimator.fit(features)
        train_seconds = time.perf_counter() - started

        scaled = estimator[:-1].transform(features)
        labels = estimator.predict(features)
        return {
            "silhouette": float(silhouette_score(scaled, labels)),
            "davies_bouldin": float(davies_bouldin_score(scaled, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(scaled, labels)),
            "stability_ari": _stability(estimator, features, labels),
            "train_seconds": train_seconds,
        }


def build_station_profiles(gold: pd.DataFrame) -> pd.DataFrame:
    """Return one row per station with its yearly departures, arrivals, and peak departures."""
    peak = gold[GoldColumns.HOUR].isin(MODELING_SETTINGS.PEAK_HOURS)
    return (
        gold.assign(**{PEAK_DEPARTURES: gold[GoldColumns.DEPARTURES_COUNT].where(peak, 0)})
        .groupby(GoldColumns.STATION_ID)
        .agg(
            **{
                DEPARTURES: (GoldColumns.DEPARTURES_COUNT, "sum"),
                ARRIVALS: (GoldColumns.ARRIVALS_COUNT, "sum"),
                PEAK_DEPARTURES: (PEAK_DEPARTURES, "sum"),
            }
        )
        .reset_index()
    )


def _stability(estimator: Pipeline, features: pd.DataFrame, labels: np.ndarray) -> float:
    """Mean adjusted Rand index between the clustering and refits with other seeds."""
    scores = []
    for offset in range(1, STABILITY_SEEDS + 1):
        refit = clone(estimator).set_params(kmeans__random_state=SEED + offset).fit(features)
        scores.append(adjusted_rand_score(labels, refit.predict(features)))
    return float(np.mean(scores))


def _clusters_scatter(features: pd.DataFrame, labels: np.ndarray) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_by_group(ax, features[DEPARTURES], features[ARRIVALS], labels)
    ax.set(xlabel="Yearly departures", ylabel="Yearly arrivals", title="Stations by group")
    fig.tight_layout()
    return fig


def _cluster_profiles(features: pd.DataFrame, labels: np.ndarray) -> Figure:
    profile = features.groupby(labels).mean()
    profile.index.name = "Group"

    fig, ax = plt.subplots(figsize=(7, 4))
    profile.plot.bar(ax=ax, rot=0)
    ax.set(ylabel="Mean per station", title="Group profiles")
    fig.tight_layout()
    return fig


def _station_map(data: pd.DataFrame, labels: np.ndarray) -> Figure:
    """Plot each station at its coordinates, colored by group. Unlocated stations are skipped."""
    located = data[StationColumns.LATITUDE].notna().to_numpy()
    stations = data[located]

    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_by_group(
        ax, stations[StationColumns.LONGITUDE], stations[StationColumns.LATITUDE], labels[located]
    )
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Station groups in Guadalajara")
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    fig.tight_layout()
    return fig


def _scatter_by_group(ax: plt.Axes, x: pd.Series, y: pd.Series, labels: np.ndarray) -> None:
    """Draw one scatter series per group, so each group gets its own legend entry."""
    for group in np.unique(labels):
        members = labels == group
        ax.scatter(x[members], y[members], s=14, alpha=0.8, label=f"Group {group}")
    ax.legend()
