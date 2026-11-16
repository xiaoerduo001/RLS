"""Turn a fitted formula string into predicted values."""

import re

import numpy as np


_FUNCTIONS = (
    ("log10", "np.log10"),
    ("log2", "np.log2"),
    ("sinh", "np.sinh"),
    ("cosh", "np.cosh"),
    ("tanh", "np.tanh"),
    ("sin", "np.sin"),
    ("cos", "np.cos"),
    ("tan", "np.tan"),
    ("exp", "np.exp"),
    ("log", "np.log"),
    ("sqrt", "np.sqrt"),
    ("abs", "np.abs"),
)


def prepare_expression(formula):
    """Rewrite a formula as an expression that can be evaluated at x."""
    expr = str(formula).strip()
    if "=" in expr:
        expr = expr.split("=", 1)[1].strip()
    for name, target in _FUNCTIONS:
        expr = re.sub(rf"\b{name}\b", target, expr)
    return expr.replace("^", "**")


def predict_formula(formula, x):
    """Evaluate the formula at x. A constant formula is broadcast to every x."""
    x = np.asarray(x, dtype=float)
    expr = prepare_expression(formula)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw = eval(expr, {"np": np, "x": x, "__builtins__": {}})
    value = np.asarray(raw, dtype=float)
    if value.ndim == 0:
        value = np.full(x.shape, float(value))
    return value


def rms(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.size == 0 or not np.all(np.isfinite(yhat)):
        return np.inf
    return float(np.sqrt(np.mean((y - yhat) ** 2)))
