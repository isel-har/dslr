import unittest

import numpy as np

from utils.scale import MinMaxScaler, RobustScaler, StandardScaler


class TestMinMaxScaler(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])

    def test_fit(self):
        scaler = MinMaxScaler().fit(self.X)
        np.testing.assert_allclose(scaler.data_max_, np.array([1, 18], dtype=float))
        np.testing.assert_allclose(scaler.data_min_, np.array([-1, 2], dtype=float))
        np.testing.assert_allclose(scaler.data_range_, np.array([2, 16], dtype=float))

    def test_transform(self):
        scaler = MinMaxScaler().fit(self.X)
        np.testing.assert_allclose(
            scaler.transform(self.X),
            np.array([[0, 0], [0.25, 0.25], [0.5, 0.5], [1, 1]], dtype=float),
        )

    def test_inverse_transform(self):
        scaler = RobustScaler()
        scaler.fit(self.X)
        X_scaled = scaler.transform(self.X)
        X_unscaled = scaler.inverse_transform(X_scaled)
        np.testing.assert_allclose(X_unscaled, self.X)


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=float)

    def test_fit(self):
        scaler = StandardScaler().fit(self.X)
        np.testing.assert_allclose(scaler.mean_, np.array([0.5, 0.5], dtype=float))
        np.testing.assert_allclose(scaler.scale_, np.array([0.5, 0.5], dtype=float))

    def test_transform(self):
        scaler = StandardScaler().fit(self.X)
        np.testing.assert_allclose(
            scaler.transform(self.X),
            np.array([[-1, -1], [-1, -1], [1, 1], [1, 1]], dtype=float),
        )
        np.testing.assert_allclose(
            scaler.transform([[2.0, 2.0]]), np.array([[3, 3]], dtype=float)
        )

    def test_inverse_transform(self):
        scaler = StandardScaler().fit(self.X)
        X_scaled = scaler.transform(self.X)
        X_unscaled = scaler.inverse_transform(X_scaled)
        np.testing.assert_allclose(X_unscaled, self.X)


class TestRobustScaler(unittest.TestCase):
    def setUp(self):
        self.X = [[1.0, -2.0, 2.0], [-2.0, 1.0, 3.0], [4.0, 1.0, -2.0]]

    def test_fit(self):
        scaler = RobustScaler().fit(self.X)
        np.testing.assert_allclose(scaler.scale_, np.array([3, 1.5, 2.5], dtype=float))
        np.testing.assert_allclose(scaler.center_, np.array([1, 1, 2], dtype=float))

    def test_transform(self):
        scaler = RobustScaler().fit(self.X)
        np.testing.assert_allclose(
            scaler.transform(self.X),
            np.array(
                [[0.0, -2.0, 0.0], [-1.0, 0.0, 0.4], [1.0, 0.0, -1.6]], dtype=float
            ),
        )

    def test_inverse_transform(self):
        scaler = RobustScaler().fit(self.X)
        X_scaled = scaler.transform(self.X)
        X_unscaled = scaler.inverse_transform(X_scaled)
        np.testing.assert_allclose(X_unscaled, self.X)
