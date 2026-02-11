from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed


class OneVsRestClassifier:
    """
    One-vs-Rest multiclass strategy.
    """

    def __init__(self, estimator, *, n_jobs=None, verbose=0):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _check_estimator(self):
        if not hasattr(self.estimator, "fit"):
            raise TypeError("Base estimator must implement fit()")

        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError("Base estimator must implement predict_proba()")

    def _fit_binary(self, X, y, cls):

        y_bin = (y == cls).astype(int)

        model = deepcopy(self.estimator)
        model.fit(X, y_bin)

        return model

    def fit(self, X, y):
        self._check_estimator()

        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        if self.classes_.shape[0] < 2:
            raise ValueError("OneVsRestClassifier requires at least 2 classes")

        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._fit_binary)(X, y, cls) for cls in self.classes_
        )

        return self

    def predict_proba(self, X):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("You must call fit() before predict_proba()")

        X = np.asarray(X)
        probs = []
        for model in self.estimators_:
            probs.append(model.predict_proba(X))
        probs = np.column_stack(probs)

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
