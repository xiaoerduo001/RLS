"""Enumerate candidate breakpoint sets and pick one by the configured metric. True breakpoints are not used."""

from itertools import combinations

import numpy as np

from .formulas import rms
from .metrics import mae


def enumerate_splits(points, count, min_gap, free_count=False):
    """If free_count is false, keep only sets of exactly count breakpoints. If true, compare sizes from 0 through count."""
    points = [float(p) for p in points]
    if not free_count:
        if count <= 0:
            return [[]]
        sizes = (count,)
    else:
        sizes = range(0, min(count, len(points)) + 1)
    splits = []
    for size in sizes:
        if size == 0:
            splits.append([])
            continue
        if len(points) < size:
            continue
        for combo in combinations(points, size):
            if _gaps_ok(combo, min_gap):
                splits.append(list(combo))
    return splits


def split_series(x, y, breakpoints):
    """Split as in data generation: the breakpoint itself belongs to the left piece."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pieces = []
    start = 0
    for bp in sorted(float(b) for b in breakpoints):
        end = int(np.searchsorted(x, bp, side="right"))
        pieces.append((x[start:end], y[start:end]))
        start = end
    pieces.append((x[start:], y[start:]))
    return pieces


def score_split(x, y, breakpoints, fitter, metric):
    """Score one breakpoint set. Smaller is better.

    pooled_mae: segment MAE weighted by the number of samples.
    mean_rms: unweighted mean of segment RMS; each segment has one vote.
    """
    pieces = split_series(x, y, breakpoints)
    if any(px.shape[0] < 2 for px, _py in pieces):
        return None
    total_absolute = 0.0
    total_count = 0
    segment_rms = []
    fits = []
    for px, py in pieces:
        found = fitter.fit(px, py)
        yhat = found.predict(px)
        segment_mae = mae(py, yhat)
        one_rms = rms(py, yhat)
        if not np.isfinite(segment_mae) or not np.isfinite(one_rms):
            return None
        total_absolute += segment_mae * py.shape[0]
        total_count += int(py.shape[0])
        segment_rms.append(one_rms)
        fits.append(found)
    if metric == "pooled_mae":
        score = float(total_absolute / total_count)
    elif metric == "mean_rms":
        score = float(np.mean(segment_rms))
    else:
        raise ValueError(f"Unknown breakpoint-set metric: {metric}")
    return score, [float(b) for b in breakpoints], fits


def choose_breakpoints(x, y, points, count, min_gap, fitter, metric, free_count=False):
    """Keep the lowest score. When free_count is true, the size is chosen from 0 through count."""
    ranked = []
    for breakpoints in enumerate_splits(points, count, min_gap, free_count):
        scored = score_split(x, y, breakpoints, fitter, metric)
        if scored is not None:
            ranked.append(scored)
    if not ranked:
        scored = score_split(x, y, [], fitter, metric)
        if scored is None:
            raise RuntimeError("The series cannot be split; check the sample length")
        return scored
    ranked.sort(key=lambda item: item[0])
    best = ranked[0]
    if free_count and best[1]:
        empty = next((item for item in ranked if not item[1]), None)
        if empty is not None and best[0] > 0.5 * empty[0]:
            return empty
    return best


def _gaps_ok(combo, min_gap):
    return all(combo[i + 1] - combo[i] >= min_gap - 1e-9 for i in range(len(combo) - 1))
