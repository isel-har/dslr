import unittest

import numpy as np

from utils.utils import (
    _np_max,
    _np_mean,
    _np_median,
    _np_min,
    _np_percentile,
    _np_std,
    _np_unique,
    train_test_split,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.X, self.y = np.arange(10).reshape((5, 2)), np.arange(5)

    def test_train_test_split(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.4, random_state=42
        )

        np.testing.assert_allclose(X_train, np.array([[6, 7], [2, 3], [0, 1]]))
        np.testing.assert_allclose(y_train, np.array([3, 1, 0]))
        np.testing.assert_allclose(X_test, np.array([[8, 9], [4, 5]]))
        np.testing.assert_allclose(y_test, np.array([4, 2]))

    def test_np_mean(self):
        np.testing.assert_allclose(_np_mean(self.X), np.mean(self.X))
        np.testing.assert_allclose(_np_mean(self.X, axis=0), np.mean(self.X, axis=0))
        np.testing.assert_allclose(_np_mean(self.X, axis=1), np.mean(self.X, axis=1))

    def test_np_std(self):
        np.testing.assert_allclose(_np_std(self.X), np.std(self.X))
        np.testing.assert_allclose(_np_std(self.X, axis=0), np.std(self.X, axis=0))
        np.testing.assert_allclose(_np_std(self.X, axis=1), np.std(self.X, axis=1))

    def test_np_min(self):
        np.testing.assert_allclose(_np_min(self.X), np.min(self.X))
        np.testing.assert_allclose(_np_min(self.X, axis=0), np.min(self.X, axis=0))
        np.testing.assert_allclose(_np_min(self.X, axis=1), np.min(self.X, axis=1))

    def test_np_max(self):
        np.testing.assert_allclose(_np_max(self.X), np.max(self.X))
        np.testing.assert_allclose(_np_max(self.X, axis=0), np.max(self.X, axis=0))
        np.testing.assert_allclose(_np_max(self.X, axis=1), np.max(self.X, axis=1))

    def test_np_median(self):
        np.testing.assert_allclose(_np_median(self.X), np.median(self.X))
        np.testing.assert_allclose(
            _np_median(self.X, axis=0), np.median(self.X, axis=0)
        )
        np.testing.assert_allclose(
            _np_median(self.X, axis=1), np.median(self.X, axis=1)
        )

    def test_np_percentile(self):
        np.testing.assert_allclose(
            _np_percentile(self.X, 0.25), np.percentile(self.X, 0.25)
        )
        np.testing.assert_allclose(
            _np_percentile(self.X, 0.75, axis=0), np.percentile(self.X, 0.75, axis=0)
        )
        np.testing.assert_allclose(
            _np_percentile(self.X, 0.5, axis=1), np.percentile(self.X, 0.5, axis=1)
        )

    def test_np_unique(self):
        np.testing.assert_allclose(_np_unique(self.X), np.unique(self.X))
        # np.testing.assert_allclose(
        #     _np_unique(self.X, axis=0), np.unique(self.X)
        # )
        # np.testing.assert_allclose(
        #     _np_unique(self.X, axis=1), np.unique(self.X, axis=1)
        # )
