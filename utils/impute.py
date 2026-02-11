import numpy as np

from utils.utils import (
    _np_max,
    _np_mean,
    _np_median,
    _np_min,
    _np_percentile,
    _np_std,
    _np_unique,
)


class SimpleImputer:
    """Replace missing values (np.nan) using simple strategies.

    Params:
    ------
    strategy: str, default: "mean"
      imputation strategy.
      - if "mean", then then replace missing values using the mean along each column.
      - if "median", then replace missing values using the median along each column.
      - if "most_frequent", then replace missing values using the most frequent value along each column.
      - if "constant", then replace missing values using the `fill_value`.

    fill_value: float, default: 0
      When strategy == "constant", `fill_value` is used to replace all occurrences of missing_values.

    copy: bool, default=True
      If True, a copy of X will be created. If False, imputation will be done in-place.

    Attrs:
    ------
    statistics_: array of shape (n_features,)
      The imputation fill value for each feature.
    """

    def __init__(self, strategy="mean", fill_value=0.0, copy=True):
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy
        self.statistics_ = None

    def fit(self, X):
        X = np.array(X, dtype=float)
        n_features = X.shape[1]
        self.statistics_ = np.zeros(n_features)

        for col in range(n_features):
            values = X[:, col]
            non_missing = values[~np.isnan(values)]

            if len(non_missing) == 0:
                self.statistics_[col] = np.nan
                continue

            if self.strategy == "mean":
                self.statistics_[col] = _np_mean(non_missing)

            elif self.strategy == "median":
                self.statistics_[col] = _np_median(non_missing)

            elif self.strategy == "most_frequent":
                uniques, counts = _np_unique(non_missing, return_counts=True)
                self.statistics_[col] = uniques[np.argmax(counts)]

            elif self.strategy == "constant":
                if self.fill_value is None:
                    raise ValueError("fill_value must be set for strategy='constant'")
                self.statistics_[col] = self.fill_value

            else:
                raise ValueError("Invalid strategy")

        return self

    def transform(self, X):

        if self.statistics_ is None:
            raise ValueError("Call fit() before transform()")

        X = np.array(X, dtype=float, copy=self.copy)

        for col in range(X.shape[1]):
            missing_mask = np.isnan(X[:, col])
            X[missing_mask, col] = self.statistics_[col]

        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class KNNImputer:
    def __init__(self, n_neighbors=5, weights="uniform", copy=True):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.copy = copy
        self.X_train = None

    def _nan_euclidean(self, x, y):
        mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask) == 0:
            return np.inf
        diff = x[mask] - y[mask]
        return np.sqrt(np.sum(diff**2))

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.X_train = X
        return self

    def transform(self, X):

        if self.X_train is None:
            raise ValueError("You must call fit() before transform().")

        X = np.array(X, dtype=float, copy=self.copy)
        n_samples, n_features = X.shape

        for i in range(n_samples):
            missing_cols = np.where(np.isnan(X[i]))[0]
            if len(missing_cols) == 0:
                continue

            for col in missing_cols:
                neighbors = []
                for j in range(self.X_train.shape[0]):
                    if np.isnan(self.X_train[j, col]):
                        continue
                    dist = self._nan_euclidean(X[i], self.X_train[j])
                    if dist == np.inf:
                        continue
                    neighbors.append((dist, self.X_train[j, col]))
                if len(neighbors) == 0:
                    continue
                neighbors.sort(key=lambda x: x[0])
                k_nearest = neighbors[: self.n_neighbors]
                distances = np.array([d for d, _ in k_nearest])
                values = np.array([v for _, v in k_nearest])

                if self.weights == "uniform":
                    X[i, col] = _np_mean(values)

                elif self.weights == "distance":
                    w = 1 / (distances + 1e-8)
                    X[i, col] = np.sum(w * values) / np.sum(w)

                else:
                    raise ValueError("weights must be 'uniform' or 'distance'")

        return X

    def fit_transform(self, X):

        self.fit(X)
        return self.transform(X)
