import unittest
import numpy as np
from utils.impute import SimpleImputer, KNNImputer


class TestSimpleImputer(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, np.nan, 3], [2, 5, np.nan], [np.nan, 7, 9]], dtype=float)

    def test_fit_mean(self):
        imputer = SimpleImputer(strategy="mean")
        imputer.fit(self.X)

        expected = np.array([np.mean([1, 2]), np.mean([5, 7]), np.mean([3, 9])])
        np.testing.assert_allclose(imputer.statistics_, expected)

    def test_fit_median(self):
        imputer = SimpleImputer(strategy="median")
        imputer.fit(self.X)

        expected = np.array([np.median([1, 2]), np.median([5, 7]), np.median([3, 9])])
        np.testing.assert_allclose(imputer.statistics_, expected)

    def test_fit_most_frequent(self):
        X = np.array([[1, np.nan], [1, 5], [2, 5], [1, np.nan]])

        imputer = SimpleImputer(strategy="most_frequent")
        imputer.fit(X)

        expected = np.array([1, 5])
        np.testing.assert_array_equal(imputer.statistics_, expected)

    def test_fit_constant(self):
        imputer = SimpleImputer(strategy="constant", fill_value=-1)
        imputer.fit(self.X)

        expected = np.array([-1, -1, -1])
        np.testing.assert_array_equal(imputer.statistics_, expected)

    def test_invalid_strategy(self):
        imputer = SimpleImputer(strategy="invalid")
        with self.assertRaises(ValueError):
            imputer.fit(self.X)

    def test_transform(self):
        imputer = SimpleImputer(strategy="mean")
        imputer.fit(self.X)
        X_t = imputer.transform(self.X)

        self.assertFalse(np.isnan(X_t).any())

    def test_transform_before_fit(self):
        imputer = SimpleImputer()
        with self.assertRaises(ValueError):
            imputer.transform(self.X)

    def test_copy_false(self):
        X = self.X.copy()
        imputer = SimpleImputer(copy=False)

        X_id = id(X)
        imputer.fit_transform(X)
        self.assertEqual(id(X), X_id)

    def test_column_all_nan(self):
        X = np.array([[np.nan, 1], [np.nan, 2]])

        imputer = SimpleImputer(strategy="mean")
        imputer.fit(X)

        self.assertTrue(np.isnan(imputer.statistics_[0]))


class TestKNNImputer(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]], dtype=float
        )

    def test_fit(self):
        imputer = KNNImputer()
        imputer.fit(self.X)

        self.assertIsNotNone(imputer.X_train)

    def test_nan_euclidean(self):
        imputer = KNNImputer()

        x = np.array([1, np.nan])
        y = np.array([4, 2])

        dist = imputer._nan_euclidean(x, y)

        self.assertAlmostEqual(dist, 3)

    def test_nan_euclidean_all_nan(self):
        imputer = KNNImputer()

        x = np.array([np.nan, np.nan])
        y = np.array([np.nan, np.nan])

        self.assertEqual(imputer._nan_euclidean(x, y), np.inf)

    def test_transform_uniform(self):
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        X_t = imputer.fit_transform(self.X)
        expected = np.array(
            [[1, 2, 4], [3, 4, 3], [5.5, 6, 5], [8.0, 8.0, 7.0]], dtype=float
        )
        np.testing.assert_allclose(X_t, expected)

    def test_transform_distance(self):
        imputer = KNNImputer(n_neighbors=2, weights="distance")
        X_t = imputer.fit_transform(self.X)

        expected = np.array(
            [[1, 2, 3.8284271], [3, 4, 3], [5.5, 6, 5], [8.0, 8.0, 7.0]], dtype=float
        )
        np.testing.assert_allclose(X_t, expected)
        # self.assertFalse(np.isnan(X_t).any())

    def test_transform_before_fit(self):
        imputer = KNNImputer()

        with self.assertRaises(ValueError):
            imputer.transform(self.X)

    def test_invalid_weights(self):
        imputer = KNNImputer(weights="invalid")
        imputer.fit(self.X)

        with self.assertRaises(ValueError):
            imputer.transform(self.X)

    def test_copy_false(self):
        X = self.X.copy()
        imputer = KNNImputer(copy=False)

        X_id = id(X)
        imputer.fit_transform(X)

        self.assertEqual(id(X), X_id)

    def test_no_valid_neighbors(self):
        X = np.array([[np.nan, 1], [np.nan, 2]])

        imputer = KNNImputer()
        X_t = imputer.fit_transform(X)

        self.assertTrue(np.isnan(X_t[:, 0]).all())
