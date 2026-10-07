# Textbook autocorrelation via statsmodels (D-028, D-029): implementation handoff

| Date | Change |
|---|---|
| 2026-10-07 | Created for the implementation chat ("EEG optimization pipeline implementation"). Every line number, call site and command below was read from `main` at `589a0de` on 2026-10-07 **[src]**; `Ih Fit/` has not changed since `3dfa263`. The `noise_stats.py` text of §4.1 was run in the 2026-10-07 chat against statsmodels 0.15.0 **[run]**; nothing here has been committed as code. |

Operational handoff: what to change in `Ih Fit/`, in which order, how to check
it, and what to recompute afterwards. The reasoning (why textbook, the exact
relation to the old estimator, the expected numbers) is in the change plan,
`claude/TEEG_acf_textbook_change_plan_2026-10-07.md` (project knowledge).

## 1. What to load

| What | Where | Why |
|---|---|---|
| This handoff | repo, `Passive Features/HPC script/Ih Fit/docs/TEEG_acf_textbook_implementation_handoff_2026-10-07.md` | the work list |
| Decision log, D-028 and D-029 | project knowledge, **`TEEG_decisions_and_ideas_log.md` at the root** | the binding statements |
| Change plan | project knowledge, `claude/TEEG_acf_textbook_change_plan_2026-10-07.md` | derivation, library check, §3.5 table of expected values |
| `CHANGELOG_ih_fit.md`, `README.md` | repo, `Ih Fit/` | stage log and cluster commands to keep current |

**Decision IDs.** D-028 was first written as "D-025" at 09:47 on 2026-10-07 and
renumbered the same morning: D-025, D-026, D-027 and I-002 belong to the
diameter workstream and sit in
`Passive Features/Diameter Re-measurement/docs/TEEG_decision_log_additions_2026-10-06.md`,
not yet merged into the project log. The next free ID is **D-030**.

## 2. The change

Every lag-$L$ sample autocorrelation in `Ih Fit/` becomes, for each window
$x_1, \dots, x_n$ with $u_j = x_j - \bar x$ and each lag $L \in \{1, \dots, n-1\}$
(samples),

$$\hat\rho_{\rm tb}(L) = \frac{\sum_{j=1}^{n-L} u_j u_{j+L}}{\sum_{j=1}^{n} u_j^2} \qquad \text{(D-028.1)}$$

computed as `statsmodels.tsa.stattools.acf(x, adjusted=False, nlags=L_max)[L]`
through ONE wrapper, `noise_stats.sample_acf`. The estimator it replaces,
$\hat\rho_{\rm adj}(L) = \bigl(\sum u_j u_{j+L}/(n-L)\bigr)/\bigl(\sum u_j^2/n\bigr)$, relates
to it exactly, window by window: $\hat\rho_{\rm tb}(L) = \frac{n-L}{n}\,\hat\rho_{\rm adj}(L)$ (D-028.2).

| Unchanged | Why |
|---|---|
| `Biological Fit/` (all of it) | D-029 (i): the regression oracle stays byte-untouched |
| the SD estimator (`np.std(..., ddof=1)`) | already NumPy |
| `trend_share` (`scipy.signal.detrend`) | already SciPy |
| `locked_share` | no library equivalent; gets a `# Custom:` marker only |
| the generator (`synthetic_ground_truth.add_recording_noise`) | its `rho_lag1` is a model parameter; its input moves by $1/n$ |

## 3. Decisions in force

| ID | What it fixes for this work |
|---|---|
| D-028 | textbook estimator (D-028.1), computed by statsmodels, one wrapper; supersedes the estimator convention of D-012.1 ($r_k$), D-014 (Statement) and D-017 (Statement) |
| D-029 (i) | `Biological Fit/` not changed |
| D-029 (ii) | no opt-in for the old (adjusted) form anywhere |
| D-029 (iii) | NaN at every lag with $n - L < 10$; NaN for a flat or non-finite window, as before |
| D-029 (iv) | the wrapper is `Ih Fit/noise_stats.py`, `sample_acf(x, lags)` |
| D-029 (v) | statsmodels is believed absent from the cluster env `prova` **[user]**: install and check it before any run (§5) |
| D-006 (viii) | the regression gate must still PASS bit for bit |
| D-010 (iii) | no liveness check by `isfinite` alone (applies to any new smoke) |

## 4. Blocks, in order

Follow the `scientific-coding` block protocol (spec entry, code, verify twice,
record), `hpc-python-compat` (ASCII, LF) for every `.py`. `Ih Fit/` is not in
the Coverage of the root `specs/SPEC.md` (D-022 covers only the diameter
workstream): record each block's interface in `CHANGELOG_ih_fit.md` as the
earlier stages did, unless the user asks to add `Ih Fit/` to the Coverage.

### 4.1 Block 1 -- new `noise_stats.py`

Text checked in the 2026-10-07 chat **[run]**: against the hand formula
(D-028.1) at $n$ = 12, 500, 4750, 13 250 and lags up to $n$, max difference
$5.6 \times 10^{-16}$; against the old estimator times $(n-L)/n$ to $10^{-12}$;
$|\hat\rho| \le 1$; flat window and NaN sample give NaN; a scalar lag gives a
length-1 array; 0.38 ms per call at $n$ = 4750 with three lags; pure ASCII.

```python
"""noise_stats.py -- sample statistics of recording-noise windows (D-028, D-029).

One function, sample_acf, is the only autocorrelation estimator of Ih Fit/.
It is the textbook (biased) estimator, computed by statsmodels:

    rho_hat(L) = sum_{j=1}^{n-L} u_j u_{j+L} / sum_{j=1}^{n} u_j^2,
    u_j = x_j - mean(x),

i.e. statsmodels.tsa.stattools.acf(x, adjusted=False)[L]. It replaces the
hand-written estimator that divided the lag sum by n - L (D-028.2:
textbook = (n - L)/n * old, window by window). Pure ASCII.
"""
from typing import Sequence, Union

import numpy as np
from statsmodels.tsa.stattools import acf

# D-029 (iii): NaN at any lag with fewer than this many lag products (n - L).
MIN_PRODUCTS = 10


def sample_acf(x: np.ndarray, lags: Union[int, Sequence[int]]) -> np.ndarray:
    """Textbook sample autocorrelation of ONE window at each lag in `lags`.

    x     : 1-D array of samples (any unit; the result is dimensionless).
    lags  : int or sequence of ints, in samples.
    return: float array, one value per lag, in the order given. NaN where
            L < 1 or n - L < MIN_PRODUCTS, and everywhere when the window
            has a non-finite sample or zero variance (as the code it
            replaces did). One statsmodels call per window, nlags = max lag.
    """
    x = np.asarray(x, dtype=float).ravel()
    lags = np.atleast_1d(np.asarray(lags, dtype=int))
    out = np.full(lags.shape, np.nan)
    n = x.size
    ok = (lags >= 1) & (n - lags >= MIN_PRODUCTS)
    if not ok.any() or not np.all(np.isfinite(x)):
        return out
    u = x - x.mean()
    if not float(np.dot(u, u)) > 0.0:
        return out
    r = acf(x, adjusted=False, nlags=int(lags[ok].max()), fft=True)
    out[ok] = r[lags[ok]]
    return out
```

The flat-window guard is needed: statsmodels returns NaN with a
`RuntimeWarning` (0/0) on a constant window **[run]**. The import of
statsmodels is at module level on purpose, so a missing install fails at
import, not inside a per-cell loop that logs and continues.

### 4.2 Block 2 -- the fitter, `passive_fitting_hpc_fixed.py`

| Line on `589a0de` | Now | Becomes | Feeds |
|---|---|---|---|
| 3230 (`_estimate_noise`) | `rho_lag1 = float(np.mean(s_centered[1:] * s_centered[:-1]) / var)` | `rho_lag1 = float(sample_acf(samples, 1)[0])` (keep the `var <= 0` branch above it) | `noise_rho_lag1` of every fit result, `is_correlated` (threshold 0.5), the n_eff warning; the calibration's `rho_lag1` columns through `mono._estimate_noise` |
| 3035 (`_estimate_residual_noise_at_mle`) | `rho = float(np.sum(c[:-1] * c[1:]) / (np.sum(c**2) + 1e-12))` | `rho = float(sample_acf(pool, 1)[0])` | `residuals_rho_lag1`, the AR(1) of Phase 3's parametric bootstrap |

- Import once near the top: `from noise_stats import sample_acf`. Every entry
  point that loads the monolith puts the `Ih Fit/` folder on `sys.path`
  (scripts run from the folder; `--code-dir` entry points insert it, e.g.
  `noise_calibration.py` l. 547, `run_ih_fit.py` l. 406, `run_ih_recovery.py`
  l. 495, `ls_baseline_qc.py` l. 385, `check_dep_sweeps.py` l. 208), and
  `regression_passive_identity.py` loads the new monolith by path from inside
  the same folder **[src]**. Check each one imports cleanly after the change.
- Line 3035 is numerically the textbook form already; the only behaviour
  change is a flat residual pool, which now gives NaN instead of 0. Its
  readers fall back on NaN: l. 6097 to `noise_rho_lag1`, l. 6852 and l. 7558
  to 0.0 **[src]**.
- Line 3230 drops by a factor $(n-1)/n$: 0.2 % at $n$ = 500.
- Not in scope, noted in passing: the residual pool concatenates windows of
  different bundles, so a few lag products straddle two windows **[reasoning]**.

### 4.3 Block 3 -- the calibration, `noise_calibration.py`

| Line on `589a0de` | Change |
|---|---|
| 190-211 `lag_autocorr` | body becomes `return float(sample_acf(x, lag)[0])`; the docstring cites D-028 and drops the paragraph at l. 198 claiming statsmodels "would not reproduce the fitter's rho at lag 1" (false for `adjusted=True`, and moot now) |
| 214-232 `_acf_diagnostic`, 288-340 `_ls_structure_diagnostic` | optional: one `sample_acf(x, [1] + lags)` per window instead of one call per lag; values identical |
| 260 `locked_share` | add `# Custom: project-specific statistic (D-016.4); no library equivalent.` |
| module docstring, l. 20-50 | "the fitter's OWN estimator" stays true; add one line citing D-028 |

Columns that change by (D-028.2): `noise_rho_lag1`, `noise_ss_rho_lag1`
(through `mono._estimate_noise`), `acf_*`, `ar1_*` (= textbook $\hat\rho(1)^L$),
`rho_ls_late`, `acf_ls_late_*`, `ar1_ls_late_*`.

### 4.4 Block 4 -- tests

| File | Change | Why |
|---|---|---|
| new `smoke_noise_stats.py` | N1 library oracle: `sample_acf` equals `acf(x, adjusted=False)` and the hand sum (D-028.1) at several $n$ and lags; N2 relation (D-028.2) to the old formula, written out in the test as an oracle only; N3 the NaN rules ($n-L<10$, $L<1$, flat, non-finite); N4 $\lvert\hat\rho\rvert \le 1$; N5 the default is textbook (a window where the two forms differ by more than $10^{-3}$, asserted against the textbook value); N6 ASCII | the new block's smoke, and the assertion that keeps the default |
| `smoke_ih_recovery.py` l. 560, 585, 587 (`_estimator_band`, `_acf_gap_band`) | replace the hand-written `(c[:, L:] * c[:, :-L]).mean(axis=1) / v` with `sample_acf` per trace | **R10 compares the data's statistic with a Monte-Carlo band computed by these helpers; if they keep the old formula, data and band use different estimators.** Cost: about 0.4 ms per call **[run]** |
| `smoke_ih_recovery.py` R10 l. 662 | unchanged: still asserts the calibration's lag-1 equals the fitter's (both call `sample_acf`) | identity of the two call sites |
| R13 (l. 911-939), R14 (l. ~1150-1220) | re-run. R13 uses a 1 s window at 50 kHz: its 1 ms lag shrinks by 0.999. R14's fixture is a 1 s pre-window at 50 kHz, so its late segment is 500 ms ($n$ = 25 000) and the 10 ms lag shrinks by 0.98; its docstring's 40-seed ranges (e.g. scenario C at 10 ms, [0.174, 0.449], against the > 0.1 margin) leave room **[src; arithmetic]**. R14's last sub-check (`acf_ls_late_100ms` NaN on the 95 ms twin) is kept by the NaN rule | expected to hold; **not run** |
| `smoke_ih_fit.py` (12/12), `smoke_ls_baseline_qc.py` (10/10) | no change; run | no assertion on either estimator **[src]** |
| `regression_passive_identity.py` | no change; must PASS bit for bit | compares loss, `result.x`, `result.fun`; no lag-1 coefficient enters `_build_loss_function` **[src]** |

### 4.5 Block 5 -- docs in the repo

`CHANGELOG_ih_fit.md`: one row (D-028, D-029, files, gate counts).
`README.md`: statsmodels in the environment section; the new smoke and the
updated counts in "First run on the cluster"; the `noise_calibration.py` row
of the file table now names `sample_acf`.

## 5. Cluster: environment, then gates

Delivery as in that chat: a patch applied on the laptop clone (`~/Towards-EEG`),
committed and pushed by the user, pulled on davinci; or pushed by the chat if
the user asks. Then, on the login node (`conda activate` stays outside the
`&&` chain, it can return non-zero when it works):

```bash
conda activate prova; module load proxy && python -m pip install statsmodels && python -c "import statsmodels, numpy as np; from statsmodels.tsa.stattools import acf; x=np.random.default_rng(0).normal(size=1000).cumsum(); u=x-x.mean(); print('statsmodels', statsmodels.__version__, 'textbook OK:', abs(acf(x, nlags=5)[5]-(u[5:]*u[:-5]).sum()/(u*u).sum())<1e-12)"
```

Expected: `statsmodels <version> textbook OK: True`. If pip conflicts with
the env's NumPy/SciPy, `conda install -c conda-forge statsmodels` is the
alternative. Neither command has been run on davinci **[not verified]**; the
check was run here on statsmodels 0.15.0, NumPy 2.5.3, SciPy 1.18.1.

```bash
conda activate prova; cd "/davinci-1/home/ldellamea/TEEG/Towards-EEG" && git pull && cd "Passive Features/HPC script/Ih Fit" && python smoke_noise_stats.py && python smoke_ih_fit.py && python smoke_ih_recovery.py && python smoke_ls_baseline_qc.py && python regression_passive_identity.py --ref-dir "../Biological Fit" --archive-cell "/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc/specimen_508282493"
```

Each prints its own `N/N passed` line; the last must print
`regression_passive_identity: PASS (...)`.

## 6. Recompute once the code is in

| What | How | Check |
|---|---|---|
| L3_exc noise table | `python noise_calibration.py --group-dir "/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc" --out noise_L3_v3.csv` (seconds per cell); keep `noise_L3_v2.csv` as the pre-D-028 table | cohort medians against the table below |
| Stage-7 manifests built from v2 | regenerate from v3 | `noise_rho_lag1` columns move by $(n-1)/n$ |
| Topic doc `claude/TEEG_Stage7_noise_model_2026-09-28.md` | eq. (3) and the "not confined to [-1,1]" convention line; tables of §3.2.3, §3.4.2, §3.5.2-3.5.4, §3.6.5, §3.7.2, App. A; closed form (10) becomes $\mathbb E[S_L]/\mathbb E[S_0]$; cohort fits re-run | update when the code lands, not before |
| The 09-28 diagnostic scripts (zip) | import `sample_acf` instead of their own estimator | -- |
| D-017's per-cell law fit (not written yet) | `sample_acf` on the data and on the simulations from the start | -- |

Expected re-run cohort medians, from (D-028.2) applied to the 3-decimal medians
of the topic doc §3.2.3 **[arithmetic]**. Exact for the LS windows (every L3_exc
LS pre-window is 265 ms at 50 kHz); the SS rows assume every cell's SS pulses
are at 50 kHz, which is **not verified** (only the cohort median rate was
pasted). A value off by more than the 3-decimal rounding flags a window of a
different length.

| Window, $n$ | Lag | Old | Expected |
|---|---|---|---|
| SS 10 ms, 500 | 0.02 / 0.1 / 1 ms | 0.275 / 0.155 / 0.036 | 0.274 / 0.153 / 0.032 |
| LS 265 ms, 13 250 | 0.02 / 0.1 / 1 ms | 0.668 / 0.613 / 0.566 | 0.668 / 0.613 / 0.564 |
| LS last 132.5 ms, 6625 | 0.02 / 0.1 / 1 / 10 / 100 ms | 0.593 / 0.514 / 0.448 / 0.244 / -0.088 | 0.593 / 0.514 / 0.445 / 0.226 / -0.022 |

## 7. Out of scope, recorded

- The two AR(1) recursion loops written in Python
  (`synthetic_ground_truth.add_recording_noise` l. ~428; the fitter's
  residual-noise generator l. ~5640): `scipy.signal.lfilter([1], [1, -rho], e)`
  is the same recursion. Replacing them shifts every seeded synthetic cohort at
  rounding level; needs its own decision.
- `Ih Fit/` in the Coverage of the root `specs/SPEC.md`: the user's call.
- Merging D-025-D-027 and I-002 from the diameter additions file into the
  project log: belongs to the diameter workstream.
