"""check_jitter_leak.py -- standalone check (not pipeline code; not in the
repository), 2026-10-26, behind the explanation of the SS window-start rule
(D-051 (iv), D-076 (ii)): what the mean-window subtraction (D-051.1) lets
through, and what the transient test sees of it.

Model (chosen here): a cell at the archive's SS layout (2 sweeps x 37
alternating pulses, 11 not stored, K = 63; layout "2 sweeps, K 63" of
check_findings_2026-10-23.py), noise theta_cf with sigma_b = 0, plus a pulse
response A * R_tau(t) (RC, 113 MOhm, time constant tau_m, +/-200 pA x 0.5 ms,
the whole train, so every window also holds the previous pulses' tails) whose
amplitude varies from pulse to pulse: window k carries a_k A R_tau, with
a_k = 1 + cv z_k, z_k ~ N(0, 1) iid (one draw per window, shared by all
configurations: common random numbers). For the bias at 80 ms every cell is
also run with -z_k (antithetic gains: the noise x leak cross term is odd in
z and cancels exactly in the pair; the expectation over z is unchanged).

Run 1 (this file's first version, output kept as
jitter_leak_2026-10-26_run1_L1_failed.txt) stated the leak law for the
per-cell MEDIAN sigma_hat, as mean_k delta_k / (2 sigma_med); it FAILED
(tau_m 60 ms, A 3, cv 0.3: measured 0.58 +/- 0.06 SD against 0.99), the
median moving by about 0.6 of that first-order value: the cross term, zero
on average in sigma_hat^2, lowers sigma_hat (a square root) at second order
by an amount of the same order as the leak. The claim is corrected to L1a
(exact, in sigma_hat^2) and L1b (the scaling, in the median), below.

Claims tested (tolerances fixed before running):
 L1a (the leak law, exact in expectation). After (D-051.1) the response
    survives only through (a_k - abar_p) A R_tau and adds, in expectation
    over the gains, to window k's sigma_hat^2 (n samples, polarity p with
    K_p windows)
        delta_k = cv^2 [ (1 - 1/K_p)^2 q_k + K_p^-2 sum_{j in p, j != k} q_j ],
        q_j = sum_t (f_j(t) - fbar_j)^2 / (n - 1),  f_j = A R_tau in window j on [80, 200) ms.
    PASS if the mean over windows and cells of sigma_hat_k^2 (response) -
    sigma_hat_k^2 (same noise, no response) equals mean delta_k within 3 SE,
    for every configuration.
 L1b (quadratic in cv). The bias of the per-cell median sigma_hat at
    cv = 0.3 is 9 times that at cv = 0.1, same response: PASS if
    |mean(d(0.3) - 9 d(0.1))| <= 3 SE + 0.1 |mean d(0.3)| for both response
    pairs (d the per-cell antithetic mean of the median's change; 10 % for
    terms of order cv^4).
 L2 (the test reads the mean response, not its variation). The transient
    test with option (b)'s threshold -- per 5 ms bin, max(kappa_tr sigma_SS,
    z x the SD of r_odd's bin mean under noise alone, simulated at this
    layout and law), z = 3.5 -- gives the same t_SS = max(80 ms, t_tr) at
    cv = 0 and cv = 0.3 for the same A R_tau: PASS if the medians differ by
    at most one bin (5 ms) and the 10-90 % ranges overlap.
Also reported, no pass/fail: the five SS statistics' biases at 80 ms
(option (c)); the ratio of the median sigma_hat's bias to the first-order
value mean delta / (2 sigma_med); under option (b), the start it picks, the
share sent to `ss_enlarged_none`, and the bias at its start.

Bias = mean over cells of (statistic with the response - the statistic of
the same noise draws without any response, both after (D-051.1)), in units
of the across-cell SD of the noise-only statistic at t_SS = 80 ms. The
statistics are the pipeline's (noise_cell_stats.ss_window_stats, the
per-cell median over windows); the subtraction is noise_windows.
subtract_mean_response; the test is noise_windows.transient_start.

Run (from this folder, next to check_findings_2026-10-23.py, which sets the
path to the repository's Ih Fit folder):
    python check_jitter_leak.py      (about 6 min, one core)
Writes jitter_leak_2026-10-26.txt next to this file; exit 0 iff L1a, L1b
and L2 pass. Pure ASCII, LF.
"""
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("chk", HERE / "check_findings_2026-10-23.py")
chk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chk)        # layout, noise simulator, SSWindows builder; sets sys.path
import noise_cell_stats as ncs       # noqa: E402
import noise_windows as nw           # noqa: E402
import smoke_noise_windows as snw    # noqa: E402

# ---------------------------------------------------------------------------
# Configuration (every parameter of the check)
# ---------------------------------------------------------------------------
TAU_FIX = 28.143                     # ms, the fixture's tau_m
CFG = dict(
    seed=20261027, n_rep=200, n_null=200, layout="2 sweeps, K 63", sigma_b=0.0,
    t_floor_ms=80.0, kappa=0.1, z=3.5, bin_ms=5.0, fallback_ms=190.0, r_in_mohm=113.0,
    # (tau_m ms, A, cv): bias at t_SS = 80 ms, antithetic gains
    configs_c=((TAU_FIX, 1.0, 0.2), (TAU_FIX, 5.0, 0.1), (TAU_FIX, 5.0, 0.3), (60.0, 3.0, 0.1), (60.0, 3.0, 0.3)),
    pairs_L1b=(((TAU_FIX, 5.0, 0.1), (TAU_FIX, 5.0, 0.3)), ((60.0, 3.0, 0.1), (60.0, 3.0, 0.3))),
    # (tau_m ms, A, cv): option (b), one gain draw
    configs_b=((TAU_FIX, 5.0, 0.0), (TAU_FIX, 5.0, 0.3), (60.0, 3.0, 0.3)),
    twin=((TAU_FIX, 5.0, 0.0), (TAU_FIX, 5.0, 0.3)),
    tol_se=3.0, tol_hi=0.10,
    out=HERE / "jitter_leak_2026-10-26.txt",
)
FS, PRE_N, N_W, P = chk.FS, chk.PRE_N, chk.N_W, chk.P
T_TRAIN, L_SWEEP = chk.T_TRAIN, chk.L_SWEEP
T_MS = (np.arange(N_W) - PRE_N) / FS * 1e3
EDGES = np.arange(0.0, 200.0 + 1e-9, CFG["bin_ms"])
LABELS = ("sigma", "rho0.02", "rho0.1", "rho1", "rho10")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def response_windows(pos, tau_ms):
    """RC response of one sweep's 37 alternating pulses (dep first), cut at the stored positions."""
    i = np.zeros(L_SWEEP)
    for k in range(T_TRAIN):
        s0 = k * P + PRE_N
        i[s0:s0 + chk.CFG["pulse_n"]] = chk.CFG["pulse_pA"] if k % 2 == 0 else -chk.CFG["pulse_pA"]
    v = snw.rc_response(i, r_mohm=CFG["r_in_mohm"], tau_ms=tau_ms)
    return v[pos[:, None] * P + np.arange(N_W)[None, :]]


# ---------------------------------------------------------------------------
# Analysis (pipeline functions)
# ---------------------------------------------------------------------------
def j_of(t_ms):
    return PRE_N + int(round(t_ms * 1e-3 * FS))


def window_stats(V, dep, j0):
    """Per-window 5 SS statistics on [t, 200) ms after (D-051.1): (K, 5)."""
    Xt, _, _ = nw.subtract_mean_response(V[:, j0:], dep)
    return ncs.ss_window_stats(Xt, FS)


def odd_bin_means(V, sweep, pos):
    a = nw.transient_start(chk.make_ss(V, sweep, pos), nw.WindowConfig(t_ss_floor_ms=0.0))
    return chk.all_bin_means(a.r_odd, T_MS), a.sigma_ss_mV


def t_ss_option_b(V, sweep, pos, sd_null):
    bm, s_ss = odd_bin_means(V, sweep, pos)
    thr = np.maximum(CFG["kappa"] * s_ss, CFG["z"] * sd_null)
    return max(CFG["t_floor_ms"], chk.tr_from_exceeds(np.abs(bm) >= thr, EDGES))


def delta_windows(F, dep, cv):
    """Expected leak added to each window's sigma_hat^2 (claim L1a); F (K, n) = A R_tau on the window."""
    n = F.shape[1]
    q = np.sum((F - F.mean(axis=1, keepdims=True)) ** 2, axis=1) / (n - 1)
    out = np.empty(F.shape[0])
    for mask in (dep, ~dep):
        kp, qp = int(mask.sum()), q[mask]
        out[mask] = cv ** 2 * ((1.0 - 1.0 / kp) ** 2 * qp + (qp.sum() - qp) / kp ** 2)
    return out


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------
def run():
    t0 = time.time()
    rng = np.random.default_rng(CFG["seed"])
    sweep, pos = chk.build_layout(chk.CFG["layouts"][CFG["layout"]], rng)
    dep = pos % 2 == 0
    j80 = j_of(CFG["t_floor_ms"])
    taus = sorted({c[0] for c in CFG["configs_c"] + CFG["configs_b"]})
    resp = {tau: response_windows(pos, tau) for tau in taus}

    null = np.array([odd_bin_means(chk.simulate_noise_windows(CFG["sigma_b"], sweep, pos, rng), sweep, pos)[0]
                     for _ in range(CFG["n_null"])])
    sd_null = null.std(axis=0, ddof=1)                                    # mV, per bin

    out = dict(truth_med=[], truth_s2=[], s_ss=[],
               c_med={c: [] for c in CFG["configs_c"]}, c_s2={c: [] for c in CFG["configs_c"]},
               c_delta={c: [] for c in CFG["configs_c"]}, c_first={c: [] for c in CFG["configs_c"]},
               b_t={c: [] for c in CFG["configs_b"]}, b_leak={c: [] for c in CFG["configs_b"]},
               b_truth={c: [] for c in CFG["configs_b"]})
    for _ in range(CFG["n_rep"]):
        noise = chk.simulate_noise_windows(CFG["sigma_b"], sweep, pos, rng)
        zk = rng.standard_normal(pos.size)
        w0 = window_stats(noise, dep, j80)
        med0 = np.nanmedian(w0, axis=0)
        out["truth_med"].append(med0)
        out["truth_s2"].append(np.mean(w0[:, 0] ** 2))
        out["s_ss"].append(odd_bin_means(noise, sweep, pos)[1])
        for c in CFG["configs_c"]:
            tau, A, cv = c
            F = A * resp[tau]
            meds, s2s = [], []
            for sgn in (1.0, -1.0):                                       # antithetic gains
                w = window_stats(noise + (1.0 + sgn * cv * zk)[:, None] * F, dep, j80)
                meds.append(np.nanmedian(w, axis=0))
                s2s.append(np.mean(w[:, 0] ** 2))
            out["c_med"][c].append(0.5 * (meds[0] + meds[1]))
            out["c_s2"][c].append(0.5 * (s2s[0] + s2s[1]))
            dk = delta_windows(F[:, j80:], dep, cv)
            out["c_delta"][c].append(np.mean(dk))
            out["c_first"][c].append(np.mean(dk) / (2.0 * med0[0]))
        for c in CFG["configs_b"]:
            tau, A, cv = c
            V = noise + (1.0 + cv * zk)[:, None] * (A * resp[tau])
            tss = t_ss_option_b(V, sweep, pos, sd_null)
            out["b_t"][c].append(tss)
            if tss >= CFG["fallback_ms"]:                                  # `ss_enlarged_none` (D-051 (i))
                out["b_leak"][c].append(np.full(5, np.nan))
                out["b_truth"][c].append(np.full(5, np.nan))
                continue
            jb = j_of(tss)
            out["b_truth"][c].append(np.nanmedian(window_stats(noise, dep, jb), axis=0))
            out["b_leak"][c].append(np.nanmedian(window_stats(V, dep, jb), axis=0))
    res = {k: (np.array(v) if isinstance(v, list) else {c: np.array(x) for c, x in v.items()})
           for k, v in out.items()}
    res.update(sweep=sweep, pos=pos, dep=dep, resp=resp, sd_null=sd_null, seconds=time.time() - t0)
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(r):
    lines = []
    out = lines.append
    sd80 = r["truth_med"].std(axis=0, ddof=1)
    s_ss = float(np.median(r["s_ss"]))
    dep = r["dep"]
    n = CFG["n_rep"]
    out("check_jitter_leak.py, 2026-10-26: layout %s (K = %d: %d dep, %d hyp), theta_cf, sigma_b = %.2f mV/s, "
        "%d cells (null %d), seed %d; sigma_SS median %.4f mV"
        % (CFG["layout"], dep.size, int(dep.sum()), int((~dep).sum()), CFG["sigma_b"], n, CFG["n_null"],
           CFG["seed"], s_ss))
    out("unit of every bias: the across-cell SD of the noise-only statistic at t_SS = 80 ms: %s"
        % "  ".join("%s %.4g" % (l, s) for l, s in zip(LABELS, sd80)))
    out("")
    out("Bias at t_SS = 80 ms (option (c)), antithetic gains; L1a: mean sigma_hat^2 shift against mean delta_k")
    out("%-26s %9s %9s %22s %9s %24s %9s   %s" % ("config", "R80/sSS", "leak_rms", "d(s^2) meas. (1e-6 mV2)",
                                               "pred", "median sigma: bias (SD)", "/1st ord",
                                               "rho0.02 rho0.1 rho1 rho10 (SD)"))
    l1a_ok, bias_med = True, {}
    for c in CFG["configs_c"]:
        tau, A, cv = c
        F = A * r["resp"][tau]
        ro = 0.5 * (F[dep].mean(axis=0) - F[~dep].mean(axis=0))
        ro = ro - ro[:PRE_N].mean()
        r80 = ro[(T_MS >= 80.0) & (T_MS < 85.0)].mean() / s_ss
        leak_rms = np.sqrt(r["c_delta"][c].mean()) / s_ss
        ds2 = r["c_s2"][c] - r["truth_s2"]
        m, se = ds2.mean(), ds2.std(ddof=1) / np.sqrt(n)
        pred = r["c_delta"][c].mean()
        ok = abs(m - pred) <= CFG["tol_se"] * se
        l1a_ok &= bool(ok)
        d = r["c_med"][c] - r["truth_med"]
        bias_med[c] = d[:, 0]
        b = d.mean(axis=0) / sd80
        seb = d.std(axis=0, ddof=1) / np.sqrt(n) / sd80
        first = r["c_first"][c].mean() / sd80[0]
        out("tau %5.1f, A %3.1f, cv %4.2f %9.2f %9.3f %12.3f +/- %6.3f %9.3f %4s %9.4f +/- %.4f %9.2f   %s"
            % (tau, A, cv, r80, leak_rms, 1e6 * m, 1e6 * se, 1e6 * pred, "ok" if ok else "FAIL", b[0], seb[0],
               b[0] / first, " ".join("%7.3f" % x for x in b[1:])))
    out("L1a %s" % ("PASS" if l1a_ok else "FAIL"))
    l1b_ok = True
    for lo, hi in CFG["pairs_L1b"]:
        diff = bias_med[hi] - (hi[2] / lo[2]) ** 2 * bias_med[lo]
        m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(n)
        ok = abs(m) <= CFG["tol_se"] * se + CFG["tol_hi"] * abs(bias_med[hi].mean())
        l1b_ok &= bool(ok)
        out("L1b tau %.1f, A %.1f: bias(cv %.1f) / bias(cv %.1f) = %.2f (quadratic: %.0f); "
            "mean(d_hi - %.0f d_lo) = %.4f +/- %.4f SD: %s"
            % (lo[0], lo[1], hi[2], lo[2], bias_med[hi].mean() / bias_med[lo].mean(), (hi[2] / lo[2]) ** 2,
               (hi[2] / lo[2]) ** 2, m / sd80[0], se / sd80[0], "PASS" if ok else "FAIL"))
    out("L1b %s" % ("PASS" if l1b_ok else "FAIL"))
    out("")
    out("Option (b) (z = %.1f per-bin simulated null, start max(80 ms, t_tr)): the start it picks and the bias there"
        % CFG["z"])
    out("   null SD of r_odd's 5 ms bin mean / sigma_SS at 0, 80, 100, 150, 195 ms: %s"
        % np.array2string(r["sd_null"][[0, 16, 20, 30, 39]] / s_ss, precision=3))
    for c in CFG["configs_b"]:
        tau, A, cv = c
        tb = r["b_t"][c]
        kept = tb < CFG["fallback_ms"]
        head = ("tau %5.1f ms, A %3.1f, cv %4.2f: t_SS median %.0f ms [10-90 %%: %.0f-%.0f], >= %.0f ms "
                "(`ss_enlarged_none`, no enlarged window) in %.1f %%"
                % (tau, A, cv, np.median(tb), np.percentile(tb, 10), np.percentile(tb, 90), CFG["fallback_ms"],
                   100 * np.mean(~kept)))
        if kept.sum() < 2:
            out(head + "; no bias reported (fewer than 2 cells keep a window)")
            continue
        d = (r["b_leak"][c] - r["b_truth"][c])[kept]
        b = np.nanmean(d, axis=0) / sd80
        se = np.nanstd(d, axis=0, ddof=1) / np.sqrt(kept.sum()) / sd80
        out(head + "; bias at that start over the %d cells that keep a window (unit as above): sigma %.4f +/- %.4f, "
            "rho %s" % (int(kept.sum()), b[0], se[0], " ".join("%.3f" % x for x in b[1:])))
    out("")
    ca, cb = CFG["twin"]
    xa, xb = r["b_t"][ca], r["b_t"][cb]
    qa, qb = np.percentile(xa, [10, 50, 90]), np.percentile(xb, [10, 50, 90])
    l2_ok = bool(abs(qa[1] - qb[1]) <= CFG["bin_ms"] and qa[0] <= qb[2] and qb[0] <= qa[2])
    out("L2 -- option (b)'s start, same response (tau %.1f ms, A %.1f), cv %.2f vs %.2f: median %.0f vs %.0f ms, "
        "10-90 %% %.0f-%.0f vs %.0f-%.0f ms; same start in %.1f %% of cells: %s"
        % (ca[0], ca[1], ca[2], cb[2], qa[1], qb[1], qa[0], qa[2], qb[0], qb[2], 100 * np.mean(xa == xb),
           "PASS" if l2_ok else "FAIL"))
    out("(%.0f s)" % r["seconds"])
    text = "\n".join(lines) + "\n"
    print(text, end="")
    CFG["out"].write_text(text)
    return l1a_ok and l1b_ok and l2_ok


if __name__ == "__main__":
    # rho_hat at 10 ms is NaN on windows shorter than 10.2 ms (noise_stats.sample_acf); not an error here
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice")
    raise SystemExit(0 if report(run()) else 1)
