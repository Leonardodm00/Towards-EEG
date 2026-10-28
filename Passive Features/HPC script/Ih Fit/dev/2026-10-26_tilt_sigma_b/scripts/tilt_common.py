"""tilt_common.py -- shared configuration and helpers of the checks and figures
behind TEEG_tilt_sigma_b_2026-10-26.md (the per-sweep tilt b and its SD
sigma_b(c) in the Stage-7 noise law). Standalone study scripts, not pipeline
code: they import the repository's own modules (Ih Fit Blocks 2-4:
noise_law, noise_windows, noise_cell_stats, noise_fit) so that every
simulated statistic is computed by the pipeline's estimator.

Repository path: environment variable TEEG_IH_FIT, else the Ih Fit folder
three levels above this file (this copy sits in Ih Fit/dev/2026-10-26_tilt_sigma_b/
scripts/ since 2026-10-28, D-097; the sandbox clone when the document was
written). Blocks 3-4 are on main since 2026-10-28.

Layers (separation of concerns): CFG (parameters) | simulation (data
generation) | estimators (analysis) | the scripts' own plotting.
Pure ASCII, LF.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import special
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Configuration (every parameter shared by the scripts)
# ---------------------------------------------------------------------------
REPO_IH_FIT = Path(os.environ.get("TEEG_IH_FIT", str(Path(__file__).resolve().parents[3])))   # repo: Ih Fit/dev/<doc>/scripts
CFG = dict(
    fs=50000.0,                 # Hz, SS sampling rate of every L3_exc cell (SPEC section 3)
    period=10025,               # samples, onset to onset of the SS train (200.5 ms; SPEC 3)
    train=37,                   # pulses per SS sweep (SPEC 3, data inspected)
    pre_n=500, post_n=10000,    # stored window (-10, +200) ms
    theta_cf=(0.0380, 0.175, 0.0508, 0.0366),   # sigma_f mV, rho_f, sigma_s mV, tau_s s (SPEC, checks of 2026-10-23)
    t_ss_ms=80.0,               # D-076 (ii) floor; the window [80, 200) ms is 120 ms long
    lag_periods=4,              # h* = 4 periods = 0.802 s (D-051, SPEC 2.4)
    # per-cell stored SS windows K and SS sweeps, L3_exc (K: topic doc Appendix A.1 [cluster];
    # sweeps: SPEC section 3, data inspected 2026-10-21 on the cluster)
    cohort=(
        (508282493, 2, 40), (526785799, 2, 47), (529878215, 2, 39), (531520401, 2, 40),
        (531523484, 2, 37), (531526539, 2, 46), (537297470, 2, 66), (537313600, 2, 47),
        (539653924, 2, 47), (541545716, 2, 56), (548408551, 2, 52), (548425467, 2, 69),
        (548473051, 2, 56), (569844159, 10, 348), (571430815, 6, 199), (571483807, 10, 303),
        (571624532, 10, 317), (571691416, 2, 58),
    ),
)
FS = CFG["fs"]
DT = 1.0 / FS
P = CFG["period"]
TRAIN = CFG["train"]
PRE_N = CFG["pre_n"]
N_W = CFG["pre_n"] + CFG["post_n"]
L_SWEEP = (TRAIN - 1) * P + N_W
J0 = PRE_N + int(round(CFG["t_ss_ms"] * 1e-3 * FS))
M_WIN = N_W - J0                         # samples of an enlarged window (6000 = 120 ms)


def import_repo():
    """The repository's modules (Blocks 2-4)."""
    sys.path.insert(0, str(REPO_IH_FIT))
    import noise_cell_stats as ncs   # noqa: E402
    import noise_fit as nf           # noqa: E402
    import noise_law as nl           # noqa: E402
    import noise_windows as nw       # noqa: E402
    return SimpleNamespace(ncs=ncs, nf=nf, nl=nl, nw=nw)


R = import_repo()
LAW_CF = R.nl.NoiseLaw(*CFG["theta_cf"])


# ---------------------------------------------------------------------------
# Layouts (which pulses of which sweep are stored)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    name: str
    sweep: np.ndarray        # (K,) sweep index 0..S-1 of each stored window
    pos: np.ndarray          # (K,) position 0..36 in its sweep's train
    n_sweeps: int

    @property
    def K(self) -> int:
        return int(self.pos.size)


def build_layout(n_sweeps: int, K: int, rng: np.random.Generator, name: str = "") -> Layout:
    """K stored windows out of n_sweeps x 37, the dropped ones drawn at random
    (the archive's drop positions are in trace_qc_results.csv on the cluster,
    not here: random drops are an assumption of the checks)."""
    n_tot = n_sweeps * TRAIN
    if not 1 <= K <= n_tot:
        raise ValueError("build_layout: K = %d outside 1..%d" % (K, n_tot))
    keep = np.sort(rng.choice(n_tot, size=K, replace=False))
    return Layout(name or "%d sweeps, K %d" % (n_sweeps, K), keep // TRAIN, keep % TRAIN, n_sweeps)


def pair_index(lay: Layout, lag: int):
    """Same-sweep, same-polarity pairs whose positions differ by `lag` (even):
    (first window index, second window index, sweep of the pair)."""
    key = {(int(s), int(p)): k for k, (s, p) in enumerate(zip(lay.sweep, lay.pos))}
    a, b, s_of = [], [], []
    for (s, p), k in key.items():
        j = key.get((s, p + lag))
        if j is not None and (p % 2) == ((p + lag) % 2):
            a.append(k)
            b.append(j)
            s_of.append(s)
    order = np.argsort(np.array(a), kind="stable")
    return np.array(a)[order], np.array(b)[order], np.array(s_of)[order]


def pairs_per_sweep(lay: Layout, lag: int) -> np.ndarray:
    _, _, s_of = pair_index(lay, lag)
    return np.bincount(s_of, minlength=lay.n_sweeps)


def s_eff(npp: np.ndarray) -> float:
    """Effective number of tilt draws behind V_hat: (sum N_s)^2 / sum N_s^2."""
    npp = np.asarray(npp, dtype=float)
    return float(npp.sum() ** 2 / np.sum(npp ** 2))


# ---------------------------------------------------------------------------
# Simulation (data generation only)
# ---------------------------------------------------------------------------
def simulate_windows(law, lay: Layout, rng: np.random.Generator, sigma_b: float = 0.0, units=None):
    """The stored (-10, +200) ms windows of one simulated cell: continuous
    sweeps of the law (D-056 (i)), cut at the stored positions. Returns
    (V (K, n_w) mV, units)."""
    if units is None:
        units = R.nl.draw_units(rng, lay.n_sweeps, L_SWEEP)
    tr = R.nl.simulate_traces(law, DT, units.fast, units.slow, xi_tilt=units.tilt, sigma_b=sigma_b)
    idx = lay.pos[:, None] * P + np.arange(N_W)[None, :]
    return tr[lay.sweep[:, None], idx], units


def make_ss(v: np.ndarray, lay: Layout):
    """A noise_windows.SSWindows around simulated windows (the pipeline's container)."""
    return R.nw.SSWindows(specimen_id=0, fs=FS, dt=DT, t=(np.arange(N_W) - PRE_N) / FS, v=-60.0 + v,
                          polarity_is_dep=(lay.pos % 2 == 0), sweep_number=lay.sweep + 100,
                          position=lay.pos, train_length=TRAIN, period_samples=P, pre_n=PRE_N,
                          n_raw=lay.n_sweeps * TRAIN, n_sweeps=lay.n_sweeps)


def window_means(v: np.ndarray, lay: Layout) -> np.ndarray:
    """Means of the enlarged windows [t_SS, 200) ms after (D-051.1): the
    pipeline's enlarged_windows and subtract_mean_response."""
    ss = make_ss(v, lay)
    X, _ = R.nw.enlarged_windows(ss, CFG["t_ss_ms"])
    Xt, _, _ = R.nw.subtract_mean_response(X, ss.polarity_is_dep)
    return Xt.mean(axis=1)


def noise_bank(law, lay: Layout, n_cells: int, rng: np.random.Generator, lags=(4,)):
    """Pair differences of window means under the stationary law alone
    (sigma_b = 0) for n_cells simulated cells of one layout. With the tilt,
    each difference gains exactly b_s h (check01), so any sigma_b is applied
    afterwards. Returns {lag: (d (n_cells, N_pairs) mV, sweep of each pair)}."""
    M = noise_means(law, lay, n_cells, rng)
    out = {}
    for lag in lags:
        a, b, s_of = pair_index(lay, lag)
        out[lag] = (M[:, b] - M[:, a], s_of)
    return out


def noise_means(law, lay: Layout, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    """Enlarged-window means (after D-051.1) of n_cells simulated cells under the
    stationary law alone: (n_cells, K) mV."""
    M = np.empty((n_cells, lay.K))
    for c in range(n_cells):
        v, _ = simulate_windows(law, lay, rng, 0.0)
        M[c] = window_means(v, lay)
    return M


def window_centres(lay: Layout) -> np.ndarray:
    """(K,) s: the centre of each enlarged window, measured from the start of its sweep."""
    return (lay.pos * P + J0 + 0.5 * (M_WIN - 1)) * DT


def pairs_from_means(M: np.ndarray, lay: Layout, lag: int):
    a, b, s_of = pair_index(lay, lag)
    return M[:, b] - M[:, a], s_of


# ---------------------------------------------------------------------------
# Estimators (analysis)
# ---------------------------------------------------------------------------
def h_of(lag: int) -> float:
    return lag * P * DT


def v_model(law, lag: int = None) -> float:
    """V(h | theta, 0) at a lag of `lag` periods, exact for the sampled law."""
    lag = CFG["lag_periods"] if lag is None else lag
    return R.nl.semivariogram(law, DT, M_WIN, M_WIN, lag * P)


def sigma_b2_from_pairs(d: np.ndarray, pair_sweep: np.ndarray, b: np.ndarray, lag: int, V0) -> np.ndarray:
    """Unclipped (T1) for cells whose noise-only pair differences are d
    (n, N) and whose tilts are b (n, S): D = d + b_s h, V_hat = mean(D^2)/2,
    sigma_b_hat^2 = 2 (V_hat - V0) / h^2. V0 scalar or (n,)."""
    h = h_of(lag)
    D = d + b[:, pair_sweep] * h
    V_hat = 0.5 * np.mean(D * D, axis=1)
    return 2.0 * (V_hat - V0) / h ** 2


def chi2_quantiles(nu: float, q) -> np.ndarray:
    """Quantiles of sqrt(chi2_nu / nu): the spread of sigma_b_hat / sigma_b from
    nu realised tilts alone."""
    from scipy.stats import chi2
    return np.sqrt(chi2.ppf(np.asarray(q), nu) / nu)


def log_chi2_moments(nu):
    """Mean and variance of log(chi2_nu / nu): psi(nu/2) - log(nu/2) and
    psi_1(nu/2) (digamma, trigamma)."""
    nu = np.asarray(nu, dtype=float)
    return special.digamma(nu / 2) - np.log(nu / 2), special.polygamma(1, nu / 2)


def trigamma_inverse(y: float) -> float:
    """x > 0 with psi_1(x) = y (scipy has no inverse; brentq on a bracket)."""
    # Custom: no inverse trigamma in numpy/scipy; psi_1 is decreasing on (0, inf).
    f = lambda x: special.polygamma(1, x) - y
    lo, hi = 1e-8, 1.0
    while f(hi) > 0:
        hi *= 2.0
        if hi > 1e12:
            return np.inf
    return brentq(f, lo, hi, xtol=1e-12)


def pool_cohort(s2, nu):
    """Option (b): one cohort sigma_b^2, the nu-weighted mean of the unclipped
    per-cell estimates (inverse-variance weights when the cells share one
    sigma_b and the tilt draws dominate), clipped at 0."""
    s2, nu = np.asarray(s2, float), np.asarray(nu, float)
    return max(0.0, float(np.sum(nu * s2) / np.sum(nu)))


def pool_dl_log(s2, nu, floor):
    """Option (c), DL form (the D-076 machinery on log sigma_b^2): y_c =
    log max(s2_c, floor_c) - E[log(chi2_nu/nu)], v_c = psi_1(nu_c/2);
    between-cell variance omega^2 by DerSimonian-Laird
    (statsmodels.combine_effects, method_re='dl'), CLIPPED AT 0 HERE
    (statsmodels 0.15 returns the unclipped moment estimate and uses it in its
    random-effects weights; check04 asserts this); random-effects mean and the
    shrunk y_c = mu + omega^2/(omega^2 + v_c) (y_c - mu). Returns
    (sigma_b^2 per cell, omega^2, mu)."""
    from statsmodels.stats.meta_analysis import combine_effects
    s2, nu, floor = (np.asarray(x, float) for x in (s2, nu, floor))
    bias, var = log_chi2_moments(nu)
    y = np.log(np.maximum(s2, floor)) - bias
    om2 = max(0.0, float(combine_effects(y, var, method_re="dl").tau2))
    w = 1.0 / (var + om2)
    mu = float(np.sum(w * y) / np.sum(w))
    y_shrunk = mu + om2 / (om2 + var) * (y - mu)
    return np.exp(y_shrunk), om2, mu


def pool_eb_chi2(s2, nu, floor):
    """Option (c), empirical-Bayes form (moderated variances, Smyth 2004 as
    described in its abstract and in limma: the per-cell variance estimates
    shrunk towards a pooled value): s2_c | sigma_c^2 ~ sigma_c^2 chi2_nu / nu,
    prior 1/sigma_c^2 ~ chi2_d0 / (d0 s0^2); (d0, s0^2) by the moments of
    log s2 (mean and variance against digamma/trigamma); moderated value
    (d0 s0^2 + nu_c s2_c)/(d0 + nu_c). Custom: no Python library implements
    it (limma is R). Returns (sigma_b^2 per cell, d0, s0^2)."""
    s2, nu, floor = (np.asarray(x, float) for x in (s2, nu, floor))
    k = s2.size
    e = np.log(np.maximum(s2, floor)) - (special.digamma(nu / 2) - np.log(nu / 2))
    e_bar = float(np.mean(e))
    excess = float(np.sum((e - e_bar) ** 2) / (k - 1) - np.mean(special.polygamma(1, nu / 2)))
    if excess <= 0:
        d0 = np.inf
        s02 = float(np.exp(e_bar))
        return np.full(k, s02), d0, s02
    d0 = 2.0 * trigamma_inverse(excess)
    s02 = float(np.exp(e_bar + special.digamma(d0 / 2) - np.log(d0 / 2)))
    s2c = np.maximum(s2, 0.0)
    return (d0 * s02 + nu * s2c) / (d0 + nu), d0, s02
