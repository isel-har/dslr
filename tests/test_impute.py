
import unittest
import numpy as np
from utils.impute import SimpleImputer, KNNImputer 

class TestSimpleImputer(unittest.TestCase):

    def setUp(self):
        self.data = np.arange(100).reshape(10,10)
        self.imputer_mean = SimpleImputer(strategy='mean')
        self.imputer_median = SimpleImputer(strategy='median')
        self.imputer_mode = SimpleImputer(strategy='most_frequent')

    def test_fit_mean(self):
        ...

    def test_fit_median(self):
        ...

    def test_fit_mode(self):
        ...
 
    def test_transform(self):
        ...

    def test_fit_transform(self):
        ...

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            invalid_imputer = SimpleImputer(strategy='invalid')
            invalid_imputer.fit(self.data)

class TestKNNImputer(unittest.TestCase):

    def setUp(self):
        self.data = np.array([1, 2, np.nan, 4, 5])
        self.imputer = KNNImputer()

    def test_fit(self):
        ...

    def test_transform(self):
        ...

    def test_fit_transform(self):
        ...

