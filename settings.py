"""Experiment settings. Edit only this file before a run.

Each row is one experiment. The three numbers are, in order:

    1st: additive noise
    2nd: multiplicative noise
    3rd: number of runs

The number of runs is how many curves to generate for each random seed
and each breakpoint count.
N_SEEDS is how many random seeds to draw. Each launch draws that many
distinct large integers, not the consecutive labels 1, 2, 3.
Each seed generates that many curves.
The drawn seeds are written to JSON; curve records keep only the seed
with the most hits.
"""

# Three numbers per row, left to right:
#   1st: additive noise θ_a. y = y_clean + θ_a * mean(|y_clean|) * U
#   2nd: multiplicative noise θ_m, equation (5). y = y_clean * (1 + θ_m * U)
#   3rd: number of runs. Curves per seed and per breakpoint count
# Both U are independent Uniform[0, 1]. An entry written as 0 is not applied.
# (0.3, 0.0, 500): additive noise 0.3, no multiplicative noise, 500 curves per seed.
# (0.0, 0.5, 500): no additive noise, multiplicative noise 0.5, 500 curves per seed.
EXPERIMENTS = [
    # (0.0, 0.0, 500),  # no noise
    # (0.0, 0.3, 750),  # 2nd entry: multiplicative noise 0.3
    (0.0, 0.05, 300),  # 2nd entry: multiplicative noise 0.05
    # (0.3, 0.0, 500),    # 1st entry: additive noise 0.3
    # (0.5, 0.0, 500),  # 1st entry: additive noise 0.5
]

# How many random seeds to draw. Each seed runs the configured number of curves.
N_SEEDS = 20

# Breakpoint counts for every noise row.
# 0: the count is not prescribed; the detector decides how many to report.
# 1, 2, or 3: report exactly that many. Write one count as 3, not (3).
# Several counts: (0, 3).
BREAKPOINT_COUNTS = 0

# A hit requires the same count and, after pairing in order, every absolute error at most this value.
POSITION_TOLERANCE = 0.05

# Metric used to choose a breakpoint set, one of:
# pooled_mae: segment MAE weighted by the number of samples; longer segments weigh more.
# mean_rms: unweighted mean of segment RMS. The first version; each segment has one vote.
SELECTION_METRIC = "mean_rms"

# The breakpoint set is still chosen by the metric above and polynomial least squares.
# library: formulas also come from least squares; the app is not launched.
# turingbot: after the breakpoints are chosen, each segment is sent to the local TuringBot.
FORMULA_FITTER = "library"
TURINGBOT_BIN = "/Applications/TuringBot.app/Contents/MacOS/TuringBot"
TURINGBOT_SECONDS = 16
TURINGBOT_THREADS = 4
TURINGBOT_ALLOWED_FUNCTIONS = "+ *"
