import unittest
import numpy as np
from utils.utils import train_test_split


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
