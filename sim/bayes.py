"""Offline and online Bayesian changepoint detection from sdt.

Offline is BayesOffline("const", "gauss") with threshold 0.4.
Online is the BayesOnline() default: 3 past points and threshold 0.4.
The return values are sample indices; divide by 100, as in the original notebook, to obtain x.
"""

import numpy as np
from sdt import changepoint


def make_offline():
    return changepoint.BayesOffline("const", "gauss")


def make_online():
    return changepoint.BayesOnline()


def offline_points(detector, y):
    indices = detector.find_changepoints(np.asarray(y, dtype=float), prob_threshold=0.4)
    return indices_to_x(indices)


def online_points(detector, y):
    indices = detector.find_changepoints(np.asarray(y, dtype=float), past=3, prob_threshold=0.4)
    return indices_to_x(indices)


def indices_to_x(indices):
    values = np.round(np.asarray(indices, dtype=float) / 100.0, 2)
    points = []
    for value in values:
        point = float(value)
        if not points or abs(point - points[-1]) > 1e-9:
            points.append(point)
    return points
