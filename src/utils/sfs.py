import pandas as pd
import math

def count_score(scores):
    return sum(not math.isnan(x) for x in scores if x is not None)

def count_(df: pd.DataFrame):

    data_counts = [count_score(df[c]) for c in df.columns]
    return pd.Series(data=data_counts, index=df.columns)


def max_(df: pd.DataFrame):

    data_counts = []
    for c in df.columns:
        valid_values = [x for x in df[c] if not math.isnan(x)]
        data_counts.append(max(valid_values) if valid_values else None)
    return pd.Series(data=data_counts, index=df.columns)


def min_(df: pd.DataFrame):

    data_counts = []
    for c in df.columns:
        valid_values = [x for x in df[c] if not math.isnan(x)]
        data_counts.append(min(valid_values) if valid_values else None)
    return pd.Series(data=data_counts, index=df.columns)


def mean_(df: pd.DataFrame):
    data_counts = []
    for c in df.columns:
        values = [x for x in df[c] if not math.isnan(x)]
        mean_val = sum(values) / len(values) if values else None
        data_counts.append(mean_val)
    return pd.Series(data=data_counts, index=df.columns)

def std_(df: pd.DataFrame):

    means = mean_(df)
    data_counts = []

    for c in df.columns:
        values = [x for x in df[c] if not math.isnan(x)]
        n = len(values)
        if n == 0:
            data_counts.append(None)
            continue
        variance = sum((x - means[c]) ** 2 for x in values) / n
        data_counts.append(math.sqrt(variance))

    return pd.Series(data=data_counts, index=df.columns)


def quantile_(df: pd.DataFrame, percent=0.25):

    if not (0 <= percent <= 1):
        raise ValueError("percent must be between 0 and 1")

    data_quantiles = []

    for c in df.columns:
        col = [x for x in df[c] if not math.isnan(x)]
        col.sort()
        n = len(col)

        if n == 0:
            data_quantiles.append(None)
            continue

        pos = percent * (n - 1)
        lower = math.floor(pos)
        upper = math.ceil(pos)

        if lower == upper:
            q = col[int(pos)]
        else:
            q = col[lower] + (col[upper] - col[lower]) * (pos - lower) ## to understand
        data_quantiles.append(q)

    return pd.Series(data=data_quantiles, index=df.columns)

def mean_series(serie):

    values = [x for x in serie if not math.isnan(x)]
    return sum(values) / len(values)


def variance_(df: pd.DataFrame, sample: bool = True):
    means = mean_(df)
    variances = []

    for c in df.columns:
        values = [x for x in df[c] if pd.notna(x)]
        n = len(values)

        if n == 0 or (sample and n == 1):
            variances.append(None)
            continue

        denom = n - 1 if sample else n
        variance = sum((x - means[c]) ** 2 for x in values) / denom
        variances.append(variance)

    return pd.Series(data=variances, index=df.columns)

def mode_(df: pd.DataFrame):
    
    modes = list()
    for c in df.columns:
        pass
        


def correlation_(x, y):
    # Compute means
    x_h = mean_series(x)
    y_h = mean_series(y)
    
    # Numerator: sum of products of deviations
    num = sum(
        (a - x_h) * (b - y_h)
        for a, b in zip(x, y)
        if not math.isnan(a) and not math.isnan(b)
    )

    # Denominator: product of standard deviations (not means)
    x_denom = sum((a - x_h)**2 for a in x if not math.isnan(a))
    y_denom = sum((b - y_h)**2 for b in y if not math.isnan(b))

    denom = math.sqrt(x_denom * y_denom)

    return num / denom if denom != 0 else 0
