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


class MinMaxScaler:
    """ """

    def __init__(self, *, feature_range=(0, 1), copy=True):
        self.feature_range = feature_range
        self.copy = copy

    def fit(self, X):
        X = np.asarray(X)
        self.data_min_ = _np_min(X, axis=0)
        self.data_max_ = _np_max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        return self

    def transform(self, X):
        X = np.array(X, copy=self.copy)
        scale = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
        X = (X - self.data_min_) * scale + self.feature_range[0]
        return X

    def inverse_transform(self, X):
        X = np.array(X, copy=self.copy)
        scale = self.data_range_ / (self.feature_range[1] - self.feature_range[0])
        X = (X - self.feature_range[0]) * scale + self.data_min_
        return X


class StandardScaler:
    """ """

    def __init__(self, *, with_mean=True, with_std=True, copy=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = _np_mean(X, axis=0) if self.with_mean else None
        self.scale_ = _np_std(X, axis=0) if self.with_std else None
        return self

    def transform(self, X):
        X = np.array(X, copy=self.copy)
        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            X = X / self.scale_
        return X

    def inverse_transform(self, X):
        X = np.array(X, copy=self.copy)
        if self.with_std:
            X = X * self.scale_
        if self.with_mean:
            X = X + self.mean_
        return X


class RobustScaler:
    """ """

    def __init__(
        self,
        *,
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
    ):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy

    def fit(self, X):
        X = np.asarray(X)
        q_min, q_max = self.quantile_range
        self.center_ = _np_median(X, axis=0) if self.with_centering else None
        q_low = _np_percentile(X, q_min, axis=0)
        q_high = _np_percentile(X, q_max, axis=0)
        self.scale_ = q_high - q_low if self.with_scaling else None
        return self

    def transform(self, X):
        X = np.array(X, copy=self.copy)
        if self.with_centering:
            X = X - self.center_
        if self.with_scaling:
            X = X / self.scale_
        return X

    def inverse_transform(self, X):
        X = np.array(X, copy=self.copy)
        if self.with_scaling:
            X = X * self.scale_
        if self.with_centering:
            X = X + self.center_
        return X
