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


def _np_mean(a, axis=None):
    a = np.asarray(a)

    assert 1 <= a.ndim <= 2, "only 1D or 2D arrays are supported."

    if axis is None:
        total = np.sum(a)
        count = a.size
        return total / count

    if a.ndim == 1:
        if axis != 0:
            raise ValueError("axis out of bounds for 1D array")
        return np.sum(a) / a.shape[0]

    if a.ndim == 2:
        if axis == 0:
            total = np.sum(a, axis=0)
            return total / a.shape[0]

        elif axis == 1:
            total = np.sum(a, axis=1)
            return total / a.shape[1]

        else:
            raise ValueError("axis must be 0, 1, or None")


def _np_std(a, axis=None, ddof=0):
    a = np.asarray(a)
    assert 1 <= a.ndim <= 2, "Only 1D or 2D arrays supported."

    mean = _np_mean(a, axis=axis)

    if axis is None:
        diff = a - mean
        var = np.sum(diff * diff) / (a.size - ddof)
        return np.sqrt(var)

    if a.ndim == 1:
        diff = a - mean
        var = np_sum(diff * diff) / (a.shape[0] - ddof)
        return np.sqrt(var)

    if axis == 0:
        diff = a - mean
        var = np.sum(diff * diff, axis=0) / (a.shape[0] - ddof)
        return np.sqrt(var)

    if axis == 1:
        diff = a - mean.reshape(-1, 1)
        var = np.sum(diff * diff, axis=1) / (a.shape[1] - ddof)
        return np.sqrt(var)

    raise ValueError("axis must be 0, 1, or None")


def _np_min(a, axis=None):
    a = np.asarray(a)

    if axis is None:
        m = a.flat[0]
        for x in a.flat:
            if x < m:
                m = x
        return m

    if axis == 0:
        result = []
        for col in range(a.shape[1]):
            m = a[0, col]
            for row in range(a.shape[0]):
                if a[row, col] < m:
                    m = a[row, col]
            result.append(m)
        return np.array(result)

    if axis == 1:
        result = []
        for row in range(a.shape[0]):
            m = a[row, 0]
            for col in range(a.shape[1]):
                if a[row, col] < m:
                    m = a[row, col]
            result.append(m)
        return np.array(result)

    raise ValueError("axis must be 0, 1, or None")


def _np_max(a, axis=None):
    a = np.asarray(a)

    if axis is None:
        m = a.flat[0]
        for x in a.flat:
            if x > m:
                m = x
        return m

    if axis == 0:
        result = []
        for col in range(a.shape[1]):
            m = a[0, col]
            for row in range(a.shape[0]):
                if a[row, col] > m:
                    m = a[row, col]
            result.append(m)
        return np.array(result)

    if axis == 1:
        result = []
        for row in range(a.shape[0]):
            m = a[row, 0]
            for col in range(a.shape[1]):
                if a[row, col] > m:
                    m = a[row, col]
            result.append(m)
        return np.array(result)

    raise ValueError("axis must be 0, 1, or None")


def _np_median(a, axis=None):
    a = np.asarray(a)

    if axis is None:
        flat = np.sort(a.flatten())
        n = flat.size
        mid = n // 2

        if n % 2 == 0:
            return (flat[mid - 1] + flat[mid]) / 2
        else:
            return flat[mid]

    if axis == 0:
        result = []
        for col in range(a.shape[1]):
            result.append(_np_median(a[:, col]))
        return np.array(result)

    if axis == 1:
        result = []
        for row in range(a.shape[0]):
            result.append(_np_median(a[row, :]))
        return np.array(result)

    raise ValueError("axis must be 0, 1, or None")


def _np_percentile(a, q, axis=None):
    a = np.asarray(a)
    q = float(q)

    if axis is None:
        flat = np.sort(a.flatten())
        n = flat.size
        index = (n - 1) * (q / 100)

        lower = int(np.floor(index))
        upper = int(np.ceil(index))

        if lower == upper:
            return flat[lower]

        weight = index - lower
        return flat[lower] * (1 - weight) + flat[upper] * weight

    if axis == 0:
        return np.array([_np_percentile(a[:, col], q) for col in range(a.shape[1])])

    if axis == 1:
        return np.array([_np_percentile(a[row, :], q) for row in range(a.shape[0])])

    raise ValueError("axis must be 0, 1, or None")


def _np_unique(a, return_counts=False):
    a = np.asarray(a).flatten()

    if a.size == 0:
        if return_counts:
            return np.array([]), np.array([])
        return np.array([])

    sorted_a = np.sort(a)

    unique_vals = []
    counts = []

    current = sorted_a[0]
    count = 1

    for i in range(1, len(sorted_a)):
        if sorted_a[i] == current:
            count += 1
        else:
            unique_vals.append(current)
            counts.append(count)
            current = sorted_a[i]
            count = 1

    unique_vals.append(current)
    counts.append(count)

    unique_vals = np.array(unique_vals)
    counts = np.array(counts)

    if return_counts:
        return unique_vals, counts

    return unique_vals
