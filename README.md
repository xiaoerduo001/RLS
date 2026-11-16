# Breakpoint simulation

This folder reproduces the numerical experiments for piecewise symbolic regression. It draws piecewise polynomial curves on \(x \in [0.01, 5]\), finds breakpoints with a local-outlier-factor scan, fits each piece by least squares, and then runs offline and online Bayesian changepoint detection on the best random seed.

Edit only `settings.py`. Everything else is the program.

## Setup

```bash
pip install -r requirements.txt
```

`sdt-python` needs PyTables. If `import sdt` fails on Apple Silicon, install HDF5 first and then `tables`:

```bash
brew install hdf5 c-blosc c-blosc2
HDF5_DIR=/opt/homebrew/opt/hdf5 pip install tables
```

## Settings

Each row of `EXPERIMENTS` is `(additive noise, multiplicative noise, number of runs)`:

- Additive: `y = y_clean + θ_a * mean(|y_clean|) * U`
- Multiplicative, equation (5): `y = y_clean * (1 + θ_m * U)`

`U` is uniform on `[0, 1]`. A zero coefficient drops that term. The third number is how many curves to draw for each seed and each breakpoint count.

`N_SEEDS` is how many distinct seeds to draw on each launch, from 100000 upward, not 1, 2, 3.

`BREAKPOINT_COUNTS`:

- `0`: the count is not prescribed. The detector reports strong outlier runs, and may add a point where the piecewise fit jumps.
- `1`, `2`, or `3` and above: report exactly that many. Write one count as `3`, not `(3)`. Several counts are written as `(0, 3)`.

`POSITION_TOLERANCE` is the hit tolerance, currently `0.05`. A hit needs the same count and every paired absolute error within that tolerance.

`SELECTION_METRIC` is `mean_rms` (mean of segment RMS) or `pooled_mae` (sample-weighted MAE). `FORMULA_FITTER` is `library` for least squares, or `turingbot` to call the local TuringBot app after the breakpoints are chosen.

## Run

From this folder:

```bash
python3 run.py
```

The program prints the LOF result for every seed, keeps the seed with the most hits (then the smaller miss distance, then the smaller pooled MAE), and reruns offline and online Bayes on that seed.

Each launch writes a new file, `output/best_seed_exp1.json`, then `best_seed_exp2.json`, and so on. Earlier files are left in place. `mape` in the JSON is already a percentage. `by_breakpoints` summarizes LOF and the two Bayes methods; `records` is one row per curve for the winning seed.
