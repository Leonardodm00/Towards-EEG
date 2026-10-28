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
configurations: common random numbers).

Claims tested (tolerances fixed before running):
 L1 (the leak law). After (D-051.1) the response survives only through
    (a_k - abar_p) A R_tau, and in expectation it adds to window k's
    sigma_hat^2 (window of n samples, polarity p with K_p windows)
        delta_k = cv^2 [ (1 - 1/K_p)^2 q_k + K_p^-2 sum_{j in p, j != k} q_j ],
        q_j = sum_t (f_j(t) - fbar_j)^2 / (n - 1),  f_j = A R_tau in window j on [t_SS, 200) ms,
    so the per-cell median sigma_hat moves by about mean_k delta_k / (2 sigma_med),
    sigma_med the cell's noise-only median: quadratic in cv x A.
    PASS if |measured - predicted| <= 3 SE + 0.15 |predicted| for every
    configuration with cv > 0 (first order in delta / sigma^2; the median's
    shift taken as the mean shift), and |measured| <= 0.01 SD at cv = 0.
 L2 (the test reads the mean response, not its variation). The transient
    test with option (b)'s threshold -- per 5 ms bin, max(kappa_tr sigma_SS,
    z x the SD of r_odd's bin mean under noise alone, simulated at this
    layout and law), z = 3.5 -- gives the same t_SS = max(80 ms, t_tr) at
    cv = 0 and cv = 0.3 for the same A R_tau: PASS if the medians differ by
    at most one bin (5 ms) and the 10-90 % ranges overlap.
Also reported, no pass/fail: the other four SS statistics' biases; under
option (b), the start it picks and the bias there; the precision cost of
starting at 120 instead of 80 ms (noise alone).

Bias = mean over cells of (statistic with the response - the statistic of
the same noise draws without any response, both after (D-051.1)), in units
of the across-cell SD of the noise-only statistic at t_SS = 80 ms. The
statistics are the pipeline's (noise_cell_stats.ss_window_stats, the
per-cell median over windows); the subtraction is noise_windows.
subtract_mean_response; the test is noise_windows.transient_start.

Run (from this folder, next to check_findings_2026-10-23.py, which sets the
path to the repository's Ih Fit folder):
    python check_jitter_leak.py      (about 5 min, one core)
Writes jitter_leak_2026-10-26.txt next to this file. Pure ASCII, LF.
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
CFG = dict(
    seed=20261027, n_rep=200, n_null=200, layout="2 sweeps, K 63", sigma_b=0.0,
    t_floor_ms=80.0, t_late_ms=120.0, kappa=0.1, z=3.5, bin_ms=5.0, fallback_ms=190.0,
    r_in_mohm=113.0,
    # (tau_m ms, A, cv, run option (b) on it)
    configs=((28.143, 1.0, 0.2, False),
             (28.143, 5.0, 0.0, True),
             (28.143, 5.0, 0.1, False),
             (28.143, 5.0, 0.3, True),
             (60.0, 1.0, 0.3, False),
             (60.0, 3.0, 0.3, True)),
    twin=((28.143, 5.0, 0.0), (28.143, 5.0, 0.3)),
    tol_se=3.0, tol_rel=0.15, tol_zero=0.01,
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


def cell_stats(V, dep, j0):
    """Per-cell medians of the 5 SS statistics on [t, 200) ms after (D-051.1)."""
    Xt, _, _ = nw.subtract_mean_response(V[:, j0:], dep)
    return np.nanmedian(ncs.ss_window_stats(Xt, FS), axis=0)


def odd_bin_means(V, sweep, pos):
    a = nw.transient_start(chk.make_ss(V, sweep, pos), nw.WindowConfig(t_ss_floor_ms=0.0))
    return chk.all_bin_means(a.r_odd, T_MS), a.sigma_ss_mV


def t_ss_option_b(V, sweep, pos, sd_null):
    bm, s_ss = odd_bin_means(V, sweep, pos)
    thr = np.maximum(CFG["kappa"] * s_ss, CFG["z"] * sd_null)
    return max(CFG["t_floor_ms"], chk.tr_from_exceeds(np.abs(bm) >= thr, EDGES))


def delta_windows(F, dep, cv):
    """Expected leak added to each window's sigma_hat^2 (claim L1); F (K, n) = A R_tau on the window."""
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
    j80, j120 = j_of(CFG["t_floor_ms"]), j_of(CFG["t_late_ms"])
    resp = {tau: response_windows(pos, tau) for tau in sorted({c[0] for c in CFG["configs"]})}

    null = np.array([odd_bin_means(chk.simulate_noise_windows(CFG["sigma_b"], sweep, pos, rng), sweep, pos)[0]
                     for _ in range(CFG["n_null"])])
    sd_null = null.std(axis=0, ddof=1)                                    # mV, per bin

    nc = len(CFG["configs"])
    truth80, truth120 = [], []
    leak80 = [[] for _ in range(nc)]
    pred80 = [[] for _ in range(nc)]
    tb = [[] for _ in range(nc)]
    leakb, truthb, predb = ([[] for _ in range(nc)] for _ in range(3))
    s_ss_all = []
    for _ in range(CFG["n_rep"]):
        noise = chk.simulate_noise_windows(CFG["sigma_b"], sweep, pos, rng)
        zk = rng.standard_normal(pos.size)
        t80 = cell_stats(noise, dep, j80)
        truth80.append(t80)
        truth120.append(cell_stats(noise, dep, j120))
        s_ss_all.append(odd_bin_means(noise, sweep, pos)[1])
        for ic, (tau, A, cv, do_b) in enumerate(CFG["configs"]):
            F = A * resp[tau]
            V = noise + (1.0 + cv * zk)[:, None] * F
            leak80[ic].append(cell_stats(V, dep, j80))
            pred80[ic].append(np.mean(delta_windows(F[:, j80:], dep, cv)) / (2.0 * t80[0]))
            if do_b:
                tss = t_ss_option_b(V, sweep, pos, sd_null)
                tb[ic].append(tss)
                if tss >= CFG["fallback_ms"]:          # `ss_enlarged_none`: no enlarged window (D-051 (i))
                    truthb[ic].append(np.full(5, np.nan))
                    leakb[ic].append(np.full(5, np.nan))
                    predb[ic].append(np.nan)
                    continue
                jb = j_of(tss)
                tr_b = cell_stats(noise, dep, jb)
                truthb[ic].append(tr_b)
                leakb[ic].append(cell_stats(V, dep, jb))
                predb[ic].append(np.mean(delta_windows(F[:, jb:], dep, cv)) / (2.0 * tr_b[0]))
    return dict(sweep=sweep, pos=pos, dep=dep, resp=resp, sd_null=sd_null, s_ss=np.median(s_ss_all),
                truth80=np.array(truth80), truth120=np.array(truth120),
                leak80=[np.array(x) for x in leak80], pred80=[np.array(x) for x in pred80],
                tb=[np.array(x) for x in tb], leakb=[np.array(x) for x in leakb],
                truthb=[np.array(x) for x in truthb], predb=[np.array(x) for x in predb],
                seconds=time.time() - t0)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(r):
    lines = []
    out = lines.append
    sd80 = r["truth80"].std(axis=0, ddof=1)
    s_ss = r["s_ss"]
    K = r["pos"].size
    out("check_jitter_leak.py, 2026-10-26: layout %s (K = %d: %d dep, %d hyp), theta_cf, sigma_b = %.2f mV/s, "
        "%d cells (null %d), seed %d; sigma_SS median %.4f mV"
        % (CFG["layout"], K, int(r["dep"].sum()), int((~r["dep"]).sum()), CFG["sigma_b"], CFG["n_rep"],
           CFG["n_null"], CFG["seed"], s_ss))
    out("unit of every bias: the across-cell SD of the noise-only statistic at t_SS = 80 ms: %s"
        % "  ".join("%s %.4g" % (l, s) for l, s in zip(LABELS, sd80)))
    out("precision cost of starting at 120 ms instead of 80 ms (noise alone), SD(120)/SD(80): %s"
        % "  ".join("%s %.2f" % (l, s) for l, s in
                    zip(LABELS, r["truth120"].std(axis=0, ddof=1) / sd80)))
    out("")
    out("L1 -- bias at t_SS = 80 ms (option (c)): measured (+/- SE) against the leak law")
    out("%-28s %10s %10s %16s %10s   %s" % ("config", "R(80)/sSS", "leak_rms", "sigma: measured", "predicted",
                                           "rho0.02 rho0.1 rho1 rho10 (measured)"))
    l1_ok = True
    for ic, (tau, A, cv, _) in enumerate(CFG["configs"]):
        F = A * r["resp"][tau]
        dep = r["dep"]
        resp_odd = 0.5 * (F[dep].mean(axis=0) - F[~dep].mean(axis=0))
        resp_odd = resp_odd - resp_odd[:PRE_N].mean()
        r80 = resp_odd[(T_MS >= 80.0) & (T_MS < 85.0)].mean() / s_ss
        leak_rms = np.sqrt(np.mean(delta_windows(F[:, j_of(80.0):], dep, cv))) / s_ss
        d = r["leak80"][ic] - r["truth80"]
        b = d.mean(axis=0) / sd80
        se = d.std(axis=0, ddof=1) / np.sqrt(d.shape[0]) / sd80
        pred = r["pred80"][ic].mean() / sd80[0]
        if cv > 0:
            ok = abs(b[0] - pred) <= CFG["tol_se"] * se[0] + CFG["tol_rel"] * abs(pred)
        else:
            ok = abs(b[0]) <= CFG["tol_zero"]
        l1_ok &= bool(ok)
        out("tau %5.1f ms, A %3.1f, cv %4.2f   %10.2f %10.3f %8.4f +/- %.4f %10.4f   %s   %s"
            % (tau, A, cv, r80, leak_rms, b[0], se[0], pred,
               " ".join("%7.3f" % x for x in b[1:]), "PASS" if ok else "FAIL"))
    out("L1 %s" % ("PASS" if l1_ok else "FAIL"))
    out("")
    out("Option (b) (z = %.1f per-bin simulated null, start max(80 ms, t_tr)): the start it picks and the bias there"
        % CFG["z"])
    out("   null SD of r_odd's 5 ms bin mean / sigma_SS at 0, 80, 100, 150, 195 ms: %s"
        % np.array2string(r["sd_null"][[0, 16, 20, 30, 39]] / s_ss, precision=3))
    for ic, (tau, A, cv, do_b) in enumerate(CFG["configs"]):
        if not do_b:
            continue
        tb = r["tb"][ic]
        kept = tb < CFG["fallback_ms"]
        head = ("tau %5.1f ms, A %3.1f, cv %4.2f: t_SS median %.0f ms [10-90 %%: %.0f-%.0f], >= %.0f ms "
                "(`ss_enlarged_none`, no enlarged window) in %.1f %%"
                % (tau, A, cv, np.median(tb), np.percentile(tb, 10), np.percentile(tb, 90), CFG["fallback_ms"],
                   100 * np.mean(~kept)))
        if kept.sum() < 2:
            out(head + "; no bias reported (fewer than 2 cells keep a window)")
            continue
        d = (r["leakb"][ic] - r["truthb"][ic])[kept]
        b = d.mean(axis=0) / sd80
        se = d.std(axis=0, ddof=1) / np.sqrt(d.shape[0]) / sd80
        pred = r["predb"][ic][kept].mean() / sd80[0]
        out(head + "; bias at that start over the %d cells that keep a window (unit as above): sigma %.4f +/- %.4f "
            "(predicted %.4f), rho %s" % (int(kept.sum()), b[0], se[0], pred, " ".join("%.3f" % x for x in b[1:])))
    out("")
    (ta, Aa, ca), (tb_, Ab, cb) = CFG["twin"]
    ia = [i for i, c in enumerate(CFG["configs"]) if c[:3] == (ta, Aa, ca)][0]
    ib = [i for i, c in enumerate(CFG["configs"]) if c[:3] == (tb_, Ab, cb)][0]
    xa, xb = r["tb"][ia], r["tb"][ib]
    qa, qb = np.percentile(xa, [10, 50, 90]), np.percentile(xb, [10, 50, 90])
    l2_ok = abs(qa[1] - qb[1]) <= CFG["bin_ms"] and qa[0] <= qb[2] and qb[0] <= qa[2]
    out("L2 -- option (b)'s start, same response (tau %.1f ms, A %.1f), cv %.2f vs %.2f: median %.0f vs %.0f ms, "
        "10-90 %% %.0f-%.0f vs %.0f-%.0f ms; same start in %.1f %% of cells: %s"
        % (ta, Aa, ca, cb, qa[1], qb[1], qa[0], qa[2], qb[0], qb[2], 100 * np.mean(xa == xb),
           "PASS" if l2_ok else "FAIL"))
    out("(%.0f s)" % r["seconds"])
    text = "\n".join(lines) + "\n"
    print(text, end="")
    CFG["out"].write_text(text)
    return l1_ok and l2_ok


if __name__ == "__main__":
    # rho_hat at 10 ms is NaN on windows shorter than 10.2 ms (noise_stats.sample_acf); not an error here
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice")
    raise SystemExit(0 if report(run()) else 1)
