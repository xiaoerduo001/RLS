"""Load settings.py and hold the fixed detection parameters."""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import settings


OUTPUT_DIR = ROOT / "output"
OUTPUT_PATH = OUTPUT_DIR / "best_seed.json"

# x = 0.01, 0.02, ..., 5.00
X_START = 0.01
X_STOP = 5.01
X_STEP = 0.01

FUNCTION_NAMES = (
    "zero_powers",
    "one_powers",
    "two_powers",
    "three_powers",
    "four_powers",
    "five_powers",
)

COEFF_LOW = -5.0
COEFF_HIGH = 5.1
COEFF_STEP = 0.1

SELECTION_METRICS = ("pooled_mae", "mean_rms")
FORMULA_FITTERS = ("library", "turingbot")


@dataclass(frozen=True)
class DetectConfig:
    bandwidth: float = 0.1
    n_neighbors: int = 20
    n_outlier: int = 15
    score_threshold: float = 0.6
    bias_correction: float = 0.05
    edge_margin: int = 100
    max_breakpoints: int = 3
    min_gap: float = 1.0
    refine_min_gain: float = 0.02


@dataclass(frozen=True)
class Experiment:
    additive: float
    multiplicative: float
    seeds: tuple
    row_index: int
    breakpoint_counts: tuple
    n_samples: int
    position_tolerance: float
    selection_metric: str
    formula_fitter: str = "library"
    turingbot_binary: str = "/Applications/TuringBot.app/Contents/MacOS/TuringBot"
    turingbot_seconds: float = 8.0
    turingbot_threads: int = 2
    turingbot_allowed_functions: str = "+ *"
    detect: DetectConfig = DetectConfig()


def load_experiments():
    counts = _breakpoint_counts(settings.BREAKPOINT_COUNTS)
    if not counts or any(v < 0 for v in counts):
        raise ValueError("BREAKPOINT_COUNTS must be greater than or equal to 0")
    seeds = draw_seeds(settings.N_SEEDS)
    tolerance = float(settings.POSITION_TOLERANCE)
    if tolerance < 0:
        raise ValueError("POSITION_TOLERANCE cannot be negative")
    selection_metric = str(settings.SELECTION_METRIC)
    if selection_metric not in SELECTION_METRICS:
        names = ", ".join(SELECTION_METRICS)
        raise ValueError(f"SELECTION_METRIC must be one of: {names}")
    formula_fitter = str(settings.FORMULA_FITTER)
    if formula_fitter not in FORMULA_FITTERS:
        names = ", ".join(FORMULA_FITTERS)
        raise ValueError(f"FORMULA_FITTER must be one of: {names}")
    turingbot_binary = str(settings.TURINGBOT_BIN)
    turingbot_seconds = float(settings.TURINGBOT_SECONDS)
    turingbot_threads = int(settings.TURINGBOT_THREADS)
    turingbot_allowed_functions = str(settings.TURINGBOT_ALLOWED_FUNCTIONS)
    if formula_fitter == "turingbot" and not Path(turingbot_binary).is_file():
        raise FileNotFoundError(f"TuringBot not found: {turingbot_binary}")

    rows = list(settings.EXPERIMENTS)
    if not rows:
        raise ValueError("EXPERIMENTS needs at least one row")
    experiments = []
    for index, row in enumerate(rows):
        if len(row) != 3:
            raise ValueError(
                f"Row {index + 1} must be (additive noise, multiplicative noise, number of runs)"
            )
        additive, multiplicative, n_samples = row
        additive = float(additive)
        multiplicative = float(multiplicative)
        n_samples = int(n_samples)
        if additive < 0 or multiplicative < 0:
            raise ValueError(f"Noise in row {index + 1} cannot be negative")
        if n_samples < 1:
            raise ValueError(f"Number of runs in row {index + 1} must be at least 1")
        experiments.append(
            Experiment(
                additive=additive,
                multiplicative=multiplicative,
                seeds=seeds,
                row_index=index,
                breakpoint_counts=counts,
                n_samples=n_samples,
                position_tolerance=tolerance,
                selection_metric=selection_metric,
                formula_fitter=formula_fitter,
                turingbot_binary=turingbot_binary,
                turingbot_seconds=turingbot_seconds,
                turingbot_threads=turingbot_threads,
                turingbot_allowed_functions=turingbot_allowed_functions,
            )
        )
    return experiments


def _breakpoint_counts(value):
    """A single count may be written as 3 or (3,); several as (0, 3)."""
    if isinstance(value, int):
        counts = (value,)
    else:
        counts = tuple(int(v) for v in value)
    if not counts or any(v < 0 for v in counts):
        raise ValueError("BREAKPOINT_COUNTS must be greater than or equal to 0")
    return counts


def draw_seeds(count):
    """Draw distinct random seeds from 100000 to 2^31-2, not small labels such as 1, 2, 3."""
    count = int(count)
    if count < 1:
        raise ValueError("N_SEEDS must be at least 1")
    low = 100_000
    high = 2**31 - 1
    if count > high - low:
        raise ValueError("N_SEEDS exceeds the number of seeds that can be drawn")
    rng = np.random.default_rng()
    drawn = set()
    while len(drawn) < count:
        for value in rng.integers(low, high, size=count - len(drawn)):
            drawn.add(int(value))
    return tuple(drawn)
