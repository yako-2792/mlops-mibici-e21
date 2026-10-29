"""Historical baselines written as scikit-learn estimators.

Being estimators, the baselines are logged, registered, and served exactly like the
candidates, so every comparison against them uses the same code path.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class HistoricalMeanRegressor(RegressorMixin, BaseEstimator):
    """Predict the training mean of the target for each group of key columns.

    - `keys`: columns that define a group, such as station and hour.

    A group not seen in training gets the global training mean.
    """

    def __init__(self, keys: tuple[str, ...] = ()):
        self.keys = keys

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalMeanRegressor":
        frame = X[list(self.keys)].assign(_target=np.asarray(y))
        self.group_means_ = frame.groupby(list(self.keys))["_target"].mean().rename("_mean")
        self.global_mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        merged = X[list(self.keys)].merge(
            self.group_means_, left_on=list(self.keys), right_index=True, how="left"
        )
        return merged["_mean"].fillna(self.global_mean_).to_numpy(dtype=float)


class HistoricalModeClassifier(ClassifierMixin, BaseEstimator):
    """Predict the most frequent training class of each value of a key column.

    - `key`: column that defines a group, such as the station.

    A value not seen in training gets the most frequent class overall.
    """

    def __init__(self, key: str = ""):
        self.key = key

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalModeClassifier":
        labels = pd.Series(np.asarray(y), index=X.index)
        self.classes_ = np.unique(labels)
        self.group_modes_ = labels.groupby(X[self.key]).agg(lambda group: group.mode().iloc[0])
        self.global_mode_ = labels.mode().iloc[0]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.key].map(self.group_modes_).fillna(self.global_mode_).to_numpy(dtype=object)
