"""Generate piecewise curves. The ranges for x and the coefficients match the original simulation."""

from dataclasses import dataclass

import numpy as np

from .config import FUNCTION_NAMES, X_START, X_STEP, X_STOP


def make_x():
    return np.round(np.arange(X_START, X_STOP, X_STEP), 2)


def coefficient_grid(low, high, step):
    return np.round(np.arange(low, high, step), 1)


_N_COEF = {
    "zero_powers": 1,
    "one_powers": 2,
    "two_powers": 3,
    "three_powers": 4,
    "four_powers": 5,
    "five_powers": 6,
}


def apply_piece(name, x, coef):
    """Evaluate one clean piece by name. coef follows the old script order a, b, c, ..."""
    x = np.asarray(x, dtype=float)
    c = [float(v) for v in coef]
    if name == "zero_powers":
        return np.full(x.shape, c[0])
    if name == "one_powers":
        return c[0] * x + c[1]
    if name == "two_powers":
        return c[0] * x ** 2 + c[1] * x + c[2]
    if name == "three_powers":
        return c[0] * x ** 3 + c[1] * x ** 2 + c[2] * x + c[3]
    if name == "four_powers":
        return c[0] * x ** 4 + c[1] * x ** 3 + c[2] * x ** 2 + c[3] * x + c[4]
    if name == "five_powers":
        return (
            c[0] * x ** 5
            + c[1] * x ** 4
            + c[2] * x ** 3
            + c[3] * x ** 2
            + c[4] * x
            + c[5]
        )
    raise KeyError(name)


def piece_formula(name, coef):
    c = [float(v) for v in coef]
    g = lambda v: format(v, ".12g")
    if name == "zero_powers":
        return g(c[0])
    if name == "one_powers":
        return f"({g(c[0])})*x+({g(c[1])})"
    if name == "two_powers":
        return f"({g(c[0])})*x**2+({g(c[1])})*x+({g(c[2])})"
    if name == "three_powers":
        return f"({g(c[0])})*x**3+({g(c[1])})*x**2+({g(c[2])})*x+({g(c[3])})"
    if name == "four_powers":
        return (
            f"({g(c[0])})*x**4+({g(c[1])})*x**3+({g(c[2])})*x**2"
            f"+({g(c[3])})*x+({g(c[4])})"
        )
    if name == "five_powers":
        return (
            f"({g(c[0])})*x**5+({g(c[1])})*x**4+({g(c[2])})*x**3"
            f"+({g(c[3])})*x**2+({g(c[4])})*x+({g(c[5])})"
        )
    raise KeyError(name)


@dataclass
class Series:
    x: np.ndarray
    y: np.ndarray
    y_clean: np.ndarray
    breakpoints: np.ndarray
    piece_names: tuple
    piece_coefs: tuple
    true_formulas: tuple


def _sample_breakpoints(rng, x, n_breakpoints):
    if n_breakpoints == 0:
        return np.zeros(0, dtype=float)
    low = float(x[0] + 1.0)
    high = float(x[-1] - 1.0)
    if n_breakpoints == 1:
        return np.array([round(float(rng.uniform(low, high)), 2)])

    gaps = n_breakpoints - 1
    while True:
        draws = [round(float(rng.uniform(low, high)), 2) for _ in range(n_breakpoints)]
        ordered = True
        for i in range(gaps):
            if draws[i + 1] - draws[i] <= 1.0:
                ordered = False
                break
        if ordered:
            return np.array(draws)


def _sample_piece(rng, grid):
    name = str(rng.choice(FUNCTION_NAMES))
    coef = rng.choice(grid, size=_N_COEF[name])
    return name, np.asarray(coef, dtype=float)


def generate_series(rng, x, n_breakpoints, additive, multiplicative, grid):
    """Generate one piecewise curve, then add multiplicative noise and additive noise.

    Multiplicative noise is equation (5): y_clean * (1 + θ_m * U).
    Additive noise is θ_a * mean(|y_clean|) * U. The two U draws are independent.
    """
    breakpoints = _sample_breakpoints(rng, x, n_breakpoints)
    names = []
    coefs = []
    y_clean = np.empty(x.shape, dtype=float)
    start = 0
    edges = list(breakpoints) + [np.inf]
    for edge in edges:
        name, coef = _sample_piece(rng, grid)
        end = int(np.searchsorted(x, edge, side="right"))
        y_clean[start:end] = apply_piece(name, x[start:end], coef)
        names.append(name)
        coefs.append(coef)
        start = end

    multiplicative_draw = rng.random(y_clean.shape[0])
    additive_draw = rng.random(y_clean.shape[0])
    scale = float(np.mean(np.abs(y_clean)))
    y = y_clean * (1.0 + float(multiplicative) * multiplicative_draw)
    y = y + float(additive) * scale * additive_draw

    return Series(
        x=x,
        y=y,
        y_clean=y_clean,
        breakpoints=breakpoints,
        piece_names=tuple(names),
        piece_coefs=tuple(coefs),
        true_formulas=tuple(piece_formula(n, c) for n, c in zip(names, coefs)),
    )
