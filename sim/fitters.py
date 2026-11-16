"""Piecewise fitting: least squares over polynomials of degree 0 through 5."""

from dataclasses import dataclass

import numpy as np

from .formulas import predict_formula, rms


@dataclass
class SegmentFit:
    formula: str
    error: float
    size: int = None

    def predict(self, x):
        return predict_formula(self.formula, x)


class LibraryFitter:
    """Least squares over the same function family used to generate the data; keep the smallest RMS.

    The polynomial inside a segment is still chosen by RMS. Breakpoint sets are compared in select.py.
    """

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        best = None
        for formula, yhat in _library_candidates(x, y):
            error = rms(y, yhat)
            # A higher degree replaces the current fit only when it is clearly more accurate.
            if best is None or error < best.error - 1e-8:
                best = SegmentFit(formula=formula, error=error, size=None)
        if best is None:
            return SegmentFit(formula="", error=np.inf)
        return best


def _library_candidates(x, y):
    builders = (
        _constant,
        _linear_in_x,
        _quadratic,
        _cubic,
        _quartic,
        _quintic,
    )
    for build in builders:
        if x.shape[0] < 2 and build is not _constant:
            continue
        try:
            yield build(x, y)
        except (np.linalg.LinAlgError, ValueError):
            continue


def _lstsq(features, y):
    coef, _, _, _ = np.linalg.lstsq(features, y, rcond=None)
    return coef, features @ coef


def _fmt(value):
    return format(float(value), ".12g")


def _constant(x, y):
    a = float(np.mean(y))
    return _fmt(a), np.full(y.shape, a)


def _linear_in_x(x, y):
    coef, yhat = _lstsq(np.column_stack([x, np.ones_like(x)]), y)
    return f"({_fmt(coef[0])})*x+({_fmt(coef[1])})", yhat


def _quadratic(x, y):
    if x.shape[0] < 3:
        raise ValueError("too few points")
    coef, yhat = _lstsq(np.column_stack([x ** 2, x, np.ones_like(x)]), y)
    return (
        f"({_fmt(coef[0])})*x**2+({_fmt(coef[1])})*x+({_fmt(coef[2])})",
        yhat,
    )


def _cubic(x, y):
    if x.shape[0] < 4:
        raise ValueError("too few points")
    coef, yhat = _lstsq(np.column_stack([x ** 3, x ** 2, x, np.ones_like(x)]), y)
    return (
        f"({_fmt(coef[0])})*x**3+({_fmt(coef[1])})*x**2+({_fmt(coef[2])})*x+({_fmt(coef[3])})",
        yhat,
    )


def _quartic(x, y):
    if x.shape[0] < 5:
        raise ValueError("too few points")
    coef, yhat = _lstsq(np.column_stack([x ** 4, x ** 3, x ** 2, x, np.ones_like(x)]), y)
    return (
        f"({_fmt(coef[0])})*x**4+({_fmt(coef[1])})*x**3+({_fmt(coef[2])})*x**2"
        f"+({_fmt(coef[3])})*x+({_fmt(coef[4])})",
        yhat,
    )


def _quintic(x, y):
    if x.shape[0] < 6:
        raise ValueError("too few points")
    coef, yhat = _lstsq(
        np.column_stack([x ** 5, x ** 4, x ** 3, x ** 2, x, np.ones_like(x)]),
        y,
    )
    return (
        f"({_fmt(coef[0])})*x**5+({_fmt(coef[1])})*x**4+({_fmt(coef[2])})*x**3"
        f"+({_fmt(coef[3])})*x**2+({_fmt(coef[4])})*x+({_fmt(coef[5])})",
        yhat,
    )
