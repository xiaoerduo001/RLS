#!/usr/bin/env python3
"""Run LOF from settings.py, keep the best random seed, then run offline and online Bayes on it."""

import json
import multiprocessing as mp
import os
from dataclasses import replace

import numpy as np

from sim.config import OUTPUT_DIR, load_experiments
from sim.metrics import mean_defined, mean_miss_distance, rank_key
from sim.pipeline import attach_bayes, collect_seed


def main():
    experiments = load_experiments()
    print(_plan(experiments))
    groups = []
    for spec in experiments:
        groups.append(run_group(spec))
    path = _save(groups)
    print(f"Best-seed results written to {path}")


def run_group(spec, seeds=None, workers=None):
    if seeds is None:
        seeds = list(spec.seeds)
    print(
        f"additive noise {spec.additive:g}, multiplicative noise {spec.multiplicative:g}, "
        f"seeds {list(seeds)}, breakpoint counts {list(spec.breakpoint_counts)}, "
        f"{spec.n_samples} curves per seed and count, selection metric {spec.selection_metric}",
        flush=True,
    )
    best_seed = None
    best_key = None
    best_records = None
    jobs = [(spec, seed) for seed in seeds]
    for seed, records in _imap(jobs, workers):
        key = rank_key(records, spec.position_tolerance)
        _print_seed(spec, seed, records)
        if best_key is None or key > best_key:
            best_seed = seed
            best_key = key
            best_records = records

    print(f"Best seed {best_seed}; starting offline and online Bayes", flush=True)
    from sim.bayes import make_offline, make_online

    offline = make_offline()
    online = make_online()
    attach_bayes(spec, best_seed, best_records, offline, online)
    for n_bp in spec.breakpoint_counts:
        subset = [item for item in best_records if item["n_breakpoints"] == n_bp]
        _print_block(spec, f"breakpoints {n_bp}  LOF", subset, "hit", "predicted_breakpoints", with_fit=True)
        _print_block(spec, f"breakpoints {n_bp}  offline Bayes", subset, "bayes_offline_hit", "bayes_offline", with_fit=False)
        _print_block(spec, f"breakpoints {n_bp}  online Bayes", subset, "bayes_online_hit", "bayes_online", with_fit=False)
    return _payload(spec, best_seed, seeds, best_records)


def _imap(jobs, workers):
    if len(jobs) == 1:
        spec, seed = jobs[0]
        yield seed, collect_seed(spec, seed)
        return
    processes = workers if workers is not None else max(1, (os.cpu_count() or 2) - 2)
    processes = min(processes, len(jobs))
    context = mp.get_context("fork")
    with context.Pool(processes) as pool:
        for seed, records in pool.imap_unordered(_worker, jobs):
            yield seed, records


def _worker(task):
    spec, seed = task
    return seed, collect_seed(spec, seed)


def _print_seed(spec, seed, records):
    hits = sum(bool(item["hit"]) for item in records)
    total = len(records)
    r2, _r2_used = mean_defined(records, "r2")
    mape, mape_used = mean_defined(records, "mape")
    pooled, _mae_used = mean_defined(records, "pooled_mae")
    distance = mean_miss_distance(records, "predicted_breakpoints", spec.position_tolerance)
    distance_text = "n/a" if distance is None else f"{distance:.3f}"
    print(
        f"  seed {seed}: hits {hits}/{total} = {100 * hits / total:.1f}%, "
        f"R² {_fmt(r2)}, MAPE {_fmt_percent(mape)} ({mape_used} curves), "
        f"Pooled MAE {_fmt(pooled)}, mean miss distance {distance_text}",
        flush=True,
    )


def _print_block(spec, title, records, hit_key, point_key, with_fit):
    hits = sum(bool(item[hit_key]) for item in records)
    total = len(records)
    distance = mean_miss_distance(records, point_key, spec.position_tolerance)
    distance_text = "n/a" if distance is None else f"{distance:.3f}"
    line = f"  {title}: {hits}/{total} = {100 * hits / total:.1f}%, mean miss distance {distance_text}"
    if with_fit:
        r2, _used = mean_defined(records, "r2")
        mape, mape_used = mean_defined(records, "mape")
        pooled, _mae_used = mean_defined(records, "pooled_mae")
        line += f", R² {_fmt(r2)}, MAPE {_fmt_percent(mape)} ({mape_used} curves), Pooled MAE {_fmt(pooled)}"
    print(line, flush=True)


def _payload(spec, best_seed, seeds, records):
    by_breakpoints = []
    for n_bp in spec.breakpoint_counts:
        subset = [item for item in records if item["n_breakpoints"] == n_bp]
        by_breakpoints.append({
            "n_breakpoints": int(n_bp),
            "n_samples": len(subset),
            "lof": _method_summary(spec, subset, "hit", "predicted_breakpoints", with_fit=True),
            "bayes_offline": _method_summary(spec, subset, "bayes_offline_hit", "bayes_offline", with_fit=False),
            "bayes_online": _method_summary(spec, subset, "bayes_online_hit", "bayes_online", with_fit=False),
        })
    return {
        "additive": spec.additive,
        "multiplicative": spec.multiplicative,
        "seeds": [int(seed) for seed in seeds],
        "best_seed": int(best_seed),
        "breakpoint_counts": [int(v) for v in spec.breakpoint_counts],
        "n_samples": spec.n_samples,
        "position_tolerance": spec.position_tolerance,
        "selection": spec.selection_metric,
        "noise": {
            "multiplicative": "y_clean * (1 + multiplicative * U(0,1))",
            "additive": "additive * mean(abs(y_clean)) * U(0,1)",
        },
        "by_breakpoints": by_breakpoints,
        "records": records,
    }


def _method_summary(spec, records, hit_key, point_key, with_fit):
    hits = sum(bool(item[hit_key]) for item in records)
    total = len(records)
    summary = {
        "hits": hits,
        "n": total,
        "hit_rate": hits / total if total else None,
        "mean_miss_distance": mean_miss_distance(records, point_key, spec.position_tolerance),
    }
    if with_fit:
        r2, r2_used = mean_defined(records, "r2")
        mape, mape_used = mean_defined(records, "mape")
        pooled, mae_used = mean_defined(records, "pooled_mae")
        summary.update({
            "r2": r2,
            "r2_used": r2_used,
            "mape": mape,
            "mape_used": mape_used,
            "pooled_mae": pooled,
            "pooled_mae_used": mae_used,
        })
    return summary


def _save(groups):
    path = _next_result_path()
    path.write_text(json.dumps({"groups": groups}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _next_result_path():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = OUTPUT_DIR / f"best_seed_exp{index}.json"
        if not path.exists():
            return path
        index += 1


def _plan(experiments):
    jobs = 0
    for spec in experiments:
        jobs += len(spec.seeds) * spec.n_samples * len(spec.breakpoint_counts)
    return (
        f"{len(experiments)} noise setting(s); LOF will score {jobs} curves. "
        "JSON keeps only the best seed of each setting."
    )


def _fmt_percent(value):
    if value is None:
        return "n/a"
    return f"{value:.4g}%"


def _fmt(value):
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def small_experiment(**changes):
    """For checks: copy the first loaded experiment and override a few fields."""
    base = load_experiments()[0]
    return replace(base, **changes)


if __name__ == "__main__":
    main()
