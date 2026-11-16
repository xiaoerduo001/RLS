"""LOF detection and breakpoint selection for one seed, then Bayes on the best seed."""

import numpy as np

from .config import COEFF_HIGH, COEFF_LOW, COEFF_STEP
from .detect import (
    detect_breakpoints,
    detect_single_breakpoint,
    detect_strong_breakpoints,
    detect_two_breakpoints,
)
from .fitters import LibraryFitter
from .generate import coefficient_grid, generate_series, make_x
from .metrics import mae, mape_score, position_hit, r2_score
from .select import choose_breakpoints, score_split, split_series
from .turingbot import TuringBotConfig, TuringBotFitter


def collect_seed(spec, seed):
    library = LibraryFitter()
    turingbot = _maybe_turingbot(spec)
    x = make_x()
    grid = coefficient_grid(COEFF_LOW, COEFF_HIGH, COEFF_STEP)
    records = []
    try:
        for n_bp, index, series in iter_series(spec, seed, x, grid):
            if int(n_bp) == 0:
                points = detect_strong_breakpoints(series.x, series.y, spec.detect)
                scored = score_split(
                    series.x, series.y, points, library, spec.selection_metric
                )
                if scored is None:
                    scored = score_split(series.x, series.y, [], library, spec.selection_metric)
                selection_score, predicted, fits = scored
            elif int(n_bp) == 1:
                points = detect_single_breakpoint(series.x, series.y, spec.detect)
                count = int(n_bp)
                free_count = False
            elif int(n_bp) == 2:
                points = detect_two_breakpoints(series.x, series.y, spec.detect)
                count = int(n_bp)
                free_count = False
            else:
                points = detect_breakpoints(series.x, series.y, spec.detect)
                count = int(n_bp)
                free_count = False
            if int(n_bp) != 0:
                # Two breakpoints need not be 1 apart; larger counts keep the original minimum gap.
                min_gap = 0.0 if int(n_bp) == 2 else spec.detect.min_gap
                selection_score, predicted, fits = choose_breakpoints(
                    series.x,
                    series.y,
                    points,
                    count,
                    min_gap,
                    library,
                    spec.selection_metric,
                    free_count,
                )
            fits = _formulas(series.x, series.y, predicted, fits, spec, turingbot)
            yhat = stitch(series.x, series.y, predicted, fits)
            truth = [float(v) for v in series.breakpoints]
            records.append({
                "additive": float(spec.additive),
                "multiplicative": float(spec.multiplicative),
                "selection_metric": spec.selection_metric,
                "formula_fitter": spec.formula_fitter,
                "selection_score": float(selection_score),
                "n_breakpoints": int(n_bp),
                "index": int(index),
                "true_breakpoints": truth,
                "detected_points": [float(v) for v in points],
                "predicted_breakpoints": [float(v) for v in predicted],
                "hit": position_hit(predicted, truth, spec.position_tolerance),
                "pooled_mae": float(mae(series.y, yhat)),
                "r2": r2_score(series.y, yhat),
                "mape": mape_score(series.y, yhat),
                "formulas": [item.formula for item in fits],
                "true_formulas": list(series.true_formulas),
            })
    finally:
        if turingbot is not None:
            turingbot.close()
    return records


def _maybe_turingbot(spec):
    if spec.formula_fitter != "turingbot":
        return None
    return TuringBotFitter(
        TuringBotConfig(
            binary=spec.turingbot_binary,
            seconds=spec.turingbot_seconds,
            threads=spec.turingbot_threads,
            allowed_functions=spec.turingbot_allowed_functions,
        )
    )


def _formulas(x, y, predicted, library_fits, spec, turingbot):
    """The breakpoint set is already chosen. TuringBot only rewrites each segment formula."""
    if turingbot is None:
        return library_fits
    recovered = score_split(x, y, predicted, turingbot, spec.selection_metric)
    if recovered is None:
        return library_fits
    return recovered[2]


def attach_bayes(spec, seed, records, offline, online):
    from .bayes import offline_points, online_points

    x = make_x()
    grid = coefficient_grid(COEFF_LOW, COEFF_HIGH, COEFF_STEP)
    replay = iter_series(spec, seed, x, grid)
    expected = len(spec.breakpoint_counts) * spec.n_samples
    if len(records) != expected:
        raise RuntimeError("Record count does not match the experiment settings")
    for record, (n_bp, index, series) in zip(records, replay):
        if int(n_bp) != record["n_breakpoints"] or int(index) != record["index"]:
            raise RuntimeError("Replay order does not match the first detection pass")
        if not np.allclose(series.breakpoints, record["true_breakpoints"], atol=1e-9):
            raise RuntimeError(f"Seed {seed} replayed different true breakpoints")
        truth = record["true_breakpoints"]
        offline_bp = offline_points(offline, series.y)
        online_bp = online_points(online, series.y)
        record["bayes_offline"] = offline_bp
        record["bayes_offline_hit"] = position_hit(offline_bp, truth, spec.position_tolerance)
        record["bayes_online"] = online_bp
        record["bayes_online_hit"] = position_hit(online_bp, truth, spec.position_tolerance)
    return records


def iter_series(spec, seed, x, grid):
    for n_bp in spec.breakpoint_counts:
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(spec.row_index), int(n_bp)])
        )
        for index in range(spec.n_samples):
            series = generate_series(
                rng, x, int(n_bp), spec.additive, spec.multiplicative, grid
            )
            yield int(n_bp), index, series


def stitch(x, y, breakpoints, fits):
    pieces = split_series(x, y, breakpoints)
    if len(pieces) != len(fits):
        raise RuntimeError("Number of segments does not match the number of formulas")
    parts = []
    for (px, _py), fit in zip(pieces, fits):
        yhat = np.asarray(fit.predict(px), dtype=float)
        if yhat.shape != px.shape:
            raise RuntimeError("Formula prediction length does not match the segment")
        parts.append(yhat)
    return np.concatenate(parts)
