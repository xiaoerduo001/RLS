"""Position hits, R², MAPE, and pooled MAE."""

import numpy as np


def mae(y, yhat):
    """Mean absolute error. Averaged pointwise, this is the pooled MAE."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.size == 0 or yhat.shape != y.shape or not np.all(np.isfinite(yhat)):
        return np.inf
    return float(np.mean(np.abs(y - yhat)))


def r2_score(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.size == 0 or yhat.shape != y.shape or not np.all(np.isfinite(yhat)):
        return None
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# Points with |y| below this value are left out of MAPE; the percentage has no meaningful denominator.
_MAPE_ABS_Y_MIN = 1e-8


def mape_score(y, yhat):
    """MAPE as a percentage: 100 / n * Σ |(y - ŷ) / y|.

    The formula is undefined at y = 0. Points with absolute value below 1e-8
    are also left out, so one near-zero point cannot push the percentage above 1e8.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.size == 0 or yhat.shape != y.shape or not np.all(np.isfinite(yhat)):
        return None
    usable = np.isfinite(y) & (np.abs(y) >= _MAPE_ABS_Y_MIN)
    if not np.any(usable):
        return None
    ratio = np.abs((y[usable] - yhat[usable]) / y[usable])
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return None
    return float(100.0 * np.mean(ratio))


def position_hit(predicted, truth, tolerance):
    """Same count, and every paired breakpoint is within tolerance. Empty on both sides is a hit."""
    predicted = np.round(np.sort(np.asarray(predicted, dtype=float)), 2)
    truth = np.round(np.sort(np.asarray(truth, dtype=float)), 2)
    if predicted.shape != truth.shape:
        return False
    if predicted.size == 0:
        return True
    return bool(np.all(np.round(np.abs(predicted - truth), 2) <= tolerance))


def missed_breakpoint_distances(predicted, truth, tolerance):
    """Distance from each true breakpoint outside tolerance to the nearest prediction. Undefined if nothing was predicted."""
    predicted = np.round(np.asarray(predicted, dtype=float), 2)
    truth = np.round(np.asarray(truth, dtype=float), 2)
    distances = []
    if predicted.size == 0:
        return distances
    for point in truth:
        distance = float(np.min(np.abs(predicted - point)))
        if distance > tolerance:
            distances.append(distance)
    return distances


def mean_defined(records, key):
    values = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return None, 0
    return float(np.mean(values)), len(values)


def mean_miss_distance(records, point_key, tolerance):
    distances = []
    for record in records:
        distances.extend(
            missed_breakpoint_distances(record[point_key], record["true_breakpoints"], tolerance)
        )
    if not distances:
        return None
    return float(np.mean(distances))


def rank_key(records, tolerance):
    """Rank by more hits, then a smaller miss distance, then a smaller pooled MAE."""
    hits = sum(bool(record["hit"]) for record in records)
    distance = mean_miss_distance(records, "predicted_breakpoints", tolerance)
    mae_value, _used = mean_defined(records, "pooled_mae")
    return (
        hits,
        -(distance if distance is not None else 0.0),
        -(mae_value if mae_value is not None else 0.0),
    )
