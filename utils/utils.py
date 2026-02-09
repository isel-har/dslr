import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=42):

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    n_samples = X.shape[0]
    rng = np.random.default_rng(random_state)

    test_size = int(n_samples * test_size)

    shuffled_indices = rng.permutation(np.arange(n_samples))
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
