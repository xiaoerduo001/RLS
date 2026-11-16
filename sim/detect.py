"""Breakpoint detection: scan jumps with local outlier factor, then add breaks it missed.

A point is an outlier when LOF > 1.5, the scikit-learn rule for contamination="auto".
On a constant piece the density is infinite, so LOF on the other side is +inf; that is an outlier, not an inlier.
"""

import numpy as np
from itertools import combinations


def detect_breakpoints(x, y, cfg):
    """Return corrected breakpoint locations, unique and sorted."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    candidates = x[cfg.edge_margin:-cfg.edge_margin]
    scores = np.empty(candidates.shape[0], dtype=float)

    for i, bp in enumerate(candidates):
        index = int(np.argmin(np.abs(x - bp)))
        left = int(index - cfg.bandwidth / 2 * n)
        right = int(index + cfg.bandwidth / 2 * n + 1)
        # A negative index would wrap to the end of the array; use an empty window instead.
        left = max(left, 0)
        right = min(right, n)
        window = y[left:right]
        scores[i] = _outlier_fraction(window, cfg.n_outlier, cfg.n_neighbors)

    points = _collapse(candidates, scores, cfg.score_threshold)
    if cfg.bias_correction:
        points = [round(float(p) + cfg.bias_correction, 2) for p in points]
    points = _unique_sorted(points)
    return _refine_breakpoints(x, y, points, cfg)


def detect_two_breakpoints(x, y, cfg):
    """For two breakpoints, pick the best-fitting LOF pair; fill any missing cut."""
    from .fitters import LibraryFitter
    from .select import score_split

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fitter = LibraryFitter()
    points = _unique_sorted(
        round(float(run[0][0]) + cfg.bias_correction, 2) for run in _lof_runs(x, y, cfg)
    )
    if len(points) >= 2:
        chosen = _best_pair(x, y, points, cfg, fitter, score_split)
        if chosen is not None:
            return chosen
    return _fill_cuts(x, y, points, 2, cfg, fitter, score_split)


def _best_pair(x, y, points, cfg, fitter, score_split):
    best = None
    for left, right in combinations(points, 2):
        scored = score_split(x, y, [left, right], fitter, "pooled_mae")
        if scored is None:
            continue
        if best is None or scored[0] < best[0]:
            best = (scored[0], [left, right])
    return None if best is None else best[1]


def _fill_cuts(x, y, points, count, cfg, fitter, score_split):
    chosen = list(points)
    lo = cfg.edge_margin
    hi = len(x) - cfg.edge_margin
    while len(chosen) < count:
        rough = None
        for index in range(lo, hi, 2):
            point = float(x[index])
            scored = score_split(x, y, sorted(chosen + [point]), fitter, "pooled_mae")
            if scored is None:
                continue
            if rough is None or scored[0] < rough[0]:
                rough = (scored[0], index)
        if rough is None:
            break
        center = rough[1]
        fine = None
        for index in range(max(lo, center - 3), min(hi, center + 4)):
            point = float(x[index])
            scored = score_split(x, y, sorted(chosen + [point]), fitter, "pooled_mae")
            if scored is None:
                continue
            if fine is None or scored[0] < fine[0]:
                fine = (scored[0], point)
        if fine is None:
            break
        chosen.append(fine[1])
    return _unique_sorted(chosen)


def detect_strong_breakpoints(x, y, cfg, peak_threshold=0.7):
    """When the count is not prescribed, take it from strong outlier runs and fill a missed break by the fit jump.

    Keep runs whose peak outlier fraction reaches 0.7. With exactly three, place each breakpoint at the run midpoint.
    With more than three, drop the closer and weaker jump until three remain.
    With one or two, add a polynomial candidate: fit a quadratic on each side, and keep the cut only when
    the jump in the fitted values is large relative to the local fluctuation. The bar is 2 after one run and 0.8 after two.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    runs = [
        run for run in _lof_runs(x, y, cfg)
        if max(score for _point, score in run) >= peak_threshold
    ]
    if len(runs) == 3:
        return _unique_sorted(round(float(run[len(run) // 2][0]), 2) for run in runs)
    points = [round(float(run[0][0]) + cfg.bias_correction, 2) for run in runs]
    if len(points) > 3:
        points = _trim_to_three(x, y, points)
    if len(points) in (1, 2):
        pool = detect_breakpoints(x, y, cfg)
        if len(points) == 1:
            points = _add_jumpy_cut(x, y, points, pool, 2.0)
        if len(points) == 2:
            points = _add_jumpy_cut(x, y, points, pool, 0.8)
    return _unique_sorted(points)


def _add_jumpy_cut(x, y, points, pool, min_jump):
    """Add the candidate, outside the breakpoints already kept, with the largest quadratic jump above the bar."""
    extras = []
    for point in pool:
        if any(abs(point - kept) <= 0.3 for kept in points):
            continue
        jump = _quadratic_jump(x, y, points, point)
        if jump is not None:
            extras.append((jump, point))
    if extras and max(extras)[0] >= min_jump:
        points = list(points) + [max(extras)[1]]
    return points


def _quadratic_jump(x, y, anchors, point):
    """Fit a quadratic on each side of the candidate and return the jump in fitted values divided by the local standard deviation."""
    edges = [0]
    for anchor in sorted(anchors):
        edges.append(int(np.searchsorted(x, anchor, side="right")))
    edges.append(len(x))
    cut = int(np.searchsorted(x, point, side="right"))
    for start, end in zip(edges, edges[1:]):
        if start < cut < end and cut - start >= 4 and end - cut >= 4:
            left = _quadratic_values(x[start:cut], y[start:cut])
            right = _quadratic_values(x[cut:end], y[cut:end])
            scale = float(np.std(y[max(start, cut - 20):min(end, cut + 20)])) + 1e-8
            return abs(float(left[-1]) - float(right[0])) / scale
    return None


def _quadratic_values(x_seg, y_seg):
    columns = np.column_stack([x_seg ** 2, x_seg, np.ones_like(x_seg)])
    coef, *_ = np.linalg.lstsq(columns, y_seg, rcond=None)
    return columns @ coef


def _trim_to_three(x, y, points):
    """When more than three strong points remain, drop down to three. In a pair closer than 0.5, keep the larger jump."""
    from .fitters import LibraryFitter

    fitter = LibraryFitter()
    points = list(points)
    while len(points) > 3:
        points = sorted(points)
        close = None
        for index in range(len(points) - 1):
            gap = points[index + 1] - points[index]
            if gap < 0.5 and (close is None or gap < close[0]):
                close = (gap, index)
        if close is None:
            scored = [
                (_library_jump(x, y, [q for j, q in enumerate(points) if j != i], p, fitter), i)
                for i, p in enumerate(points)
            ]
            points.pop(min(scored)[1])
            continue
        index = close[1]
        left_jump = _library_jump(
            x, y, [q for j, q in enumerate(points) if j != index], points[index], fitter
        )
        right_jump = _library_jump(
            x, y, [q for j, q in enumerate(points) if j != index + 1], points[index + 1], fitter
        )
        points.pop(index if left_jump <= right_jump else index + 1)
    return points


def _library_jump(x, y, anchors, point, fitter):
    edges = [0]
    for anchor in sorted(anchor for anchor in anchors if abs(anchor - point) > 1e-9):
        edges.append(int(np.searchsorted(x, anchor, side="right")))
    edges.append(len(x))
    cut = int(np.searchsorted(x, point, side="right"))
    for start, end in zip(edges, edges[1:]):
        if start < cut < end and cut - start >= 2 and end - cut >= 2:
            left = fitter.fit(x[start:cut], y[start:cut])
            right = fitter.fit(x[cut:end], y[cut:end])
            left_end = float(np.asarray(left.predict(x[cut - 1:cut])).reshape(-1)[0])
            right_end = float(np.asarray(right.predict(x[cut:cut + 1])).reshape(-1)[0])
            scale = float(np.std(y[max(start, cut - 20):min(end, cut + 20)])) + 1e-8
            return abs(left_end - right_end) / scale
    return 0.0


def detect_single_breakpoint(x, y, cfg):
    """For one breakpoint, keep the run with the highest outlier fraction; otherwise cut by fit error."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    runs = _lof_runs(x, y, cfg)
    if runs:
        best = max(runs, key=lambda run: (max(score for _point, score in run), len(run)))
        return [round(float(best[0][0]) + cfg.bias_correction, 2)]
    cut = _best_single_cut(x, y, cfg)
    return [] if cut is None else [cut]


def _lof_runs(x, y, cfg):
    n = y.shape[0]
    candidates = x[cfg.edge_margin:-cfg.edge_margin]
    scores = np.empty(candidates.shape[0], dtype=float)
    for i, bp in enumerate(candidates):
        index = int(np.argmin(np.abs(x - bp)))
        left = max(int(index - cfg.bandwidth / 2 * n), 0)
        right = min(int(index + cfg.bandwidth / 2 * n + 1), n)
        scores[i] = _outlier_fraction(y[left:right], cfg.n_outlier, cfg.n_neighbors)
    runs = []
    current = []
    for i, score in enumerate(scores):
        if score >= cfg.score_threshold:
            current.append((float(candidates[i]), float(score)))
            if i + 1 == len(scores) or scores[i + 1] < cfg.score_threshold:
                runs.append(current)
                current = []
    return runs


def _best_single_cut(x, y, cfg):
    """When LOF finds no jump, take the cut in the allowed range with the smallest pooled MAE."""
    from .fitters import LibraryFitter
    from .select import score_split

    fitter = LibraryFitter()
    lo = cfg.edge_margin
    hi = len(x) - cfg.edge_margin
    best = None
    for index in range(lo, hi, 2):
        scored = score_split(x, y, [float(x[index])], fitter, "pooled_mae")
        if scored is None:
            continue
        if best is None or scored[0] < best[0]:
            best = (scored[0], index)
    if best is None:
        return None
    center = best[1]
    best = None
    for index in range(max(lo, center - 3), min(hi, center + 4)):
        scored = score_split(x, y, [float(x[index])], fitter, "pooled_mae")
        if scored is None:
            continue
        if best is None or scored[0] < best[0]:
            best = (scored[0], float(x[index]))
    return None if best is None else best[1]


def _outlier_fraction(window, n_outlier, n_neighbors):
    window = np.asarray(window, dtype=float).reshape(-1, 1)
    if window.shape[0] <= n_neighbors or window.shape[0] <= n_outlier:
        return 0.0
    used = window[: int(window.shape[0] / 2) + n_outlier]
    if used.shape[0] <= n_neighbors:
        return 0.0
    labels = _lof_labels(used.reshape(-1), n_neighbors)
    return float(np.sum(labels[-n_outlier:] == -1) / n_outlier)


def _lof_labels(values, n_neighbors):
    """Return 1 (inlier) or -1 (outlier). Distance is absolute, and a point is not its own neighbor."""
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    distance = np.abs(values[:, None] - values[None, :])
    np.fill_diagonal(distance, np.inf)
    neighbor_index = np.argpartition(distance, n_neighbors - 1, axis=1)[:, :n_neighbors]
    rows = np.arange(n)[:, None]
    neighbor_distance = distance[rows, neighbor_index]
    order = np.argsort(neighbor_distance, axis=1)
    neighbor_index = neighbor_index[rows, order]
    neighbor_distance = neighbor_distance[rows, order]
    k_distance = neighbor_distance[:, -1]
    reachability = np.maximum(k_distance[neighbor_index], neighbor_distance)
    mean_reach = np.mean(reachability, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        local_density = 1.0 / mean_reach
        lof = np.mean(local_density[neighbor_index], axis=1) / local_density
    # LOF is NaN on a constant piece and still counts as an inlier. A jump off a constant piece gives +inf and counts as an outlier.
    outlier = (np.isfinite(lof) & (lof > 1.5)) | np.isposinf(lof)
    return np.where(outlier, -1, 1)


def _collapse(breakpoints, scores, threshold):
    groups = []
    current = []
    n = len(scores)
    for i in range(n):
        if scores[i] >= threshold:
            current.append(float(breakpoints[i]))
            if i + 1 == n or scores[i + 1] < threshold:
                groups.append(current)
                current = []
    return [group[0] for group in groups]


def _refine_breakpoints(x, y, points, cfg):
    """Inside each segment, find the extra cut that most reduces the polynomial fit error.

    A break missed by the outlier scan is usually a level change that is small relative to the curvature.
    One polynomial on the whole segment then fits much worse than separate polynomials on the two sides.
    """
    from .fitters import LibraryFitter

    fitter = LibraryFitter()
    points = list(points)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def fit_error(start, end):
        if end - start < 2:
            return np.inf
        return fitter.fit(x[start:end], y[start:end]).error

    def cut_quality(start, cut, end):
        whole = fit_error(start, end)
        scale = float(np.std(y[start:end])) + 1e-8
        if not np.isfinite(whole) or cut - start < 2 or end - cut < 2:
            return None
        left = fit_error(start, cut)
        right = fit_error(cut, end)
        if not np.isfinite(left) or not np.isfinite(right):
            return None
        halves = 0.5 * (left + right)
        gain = (whole - halves) / scale
        relative = halves / (whole + 1e-15)
        return gain, relative, whole

    def worth_keeping(gain, relative, whole):
        sharp = relative < 0.05 and whole > 1e-4
        return gain >= cfg.refine_min_gain or sharp

    # The scan may already have reported 3 points, one of them a false cut on a smooth piece.
    # Still allow two extra candidates; the fit error chooses later, and nothing is dropped here.
    while len(points) < cfg.max_breakpoints + 2:
        edges = [0]
        for point in points:
            edges.append(int(np.searchsorted(x, point, side="right")))
        edges.append(len(x))
        best = None
        for seg in range(len(edges) - 1):
            start, end = edges[seg], edges[seg + 1]
            if end - start < 20:
                continue
            # A true breakpoint is at least about 1 from either end. A cut against the end only removes a short stretch of noise.
            inner_lo = cfg.edge_margin
            inner_hi = len(x) - cfg.edge_margin
            for cut in range(max(start + 8, inner_lo), min(end - 8, inner_hi)):
                bp = float(x[cut - 1])
                if any(abs(bp - point) < cfg.min_gap - 1e-9 for point in points):
                    continue
                quality = cut_quality(start, cut, end)
                if quality is None:
                    continue
                gain = quality[0]
                if best is None or gain > best[0]:
                    best = (gain, bp, quality[1], quality[2])
        if best is None or not worth_keeping(best[0], best[2], best[3]):
            break
        points = _unique_sorted(points + [best[1]])
    lo = float(x[cfg.edge_margin])
    hi = float(x[-cfg.edge_margin - 1])
    return [point for point in points if lo <= point <= hi]


def _unique_sorted(points):
    out = []
    for point in sorted(points):
        if not out or abs(point - out[-1]) > 1e-9:
            out.append(point)
    return out
