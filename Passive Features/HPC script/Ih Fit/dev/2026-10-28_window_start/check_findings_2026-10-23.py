"""check_findings_2026-10-23.py -- standalone checks behind the explanation of
two findings of Blocks 3-4 (noise fit, Ih Fit/specs/SPEC.md). Not pipeline
code; not in the repository. Uses the repository's own functions
(noise_law, noise_windows, noise_cell_stats, noise_fit) so the checks run
the pipeline's estimators, on simulated cells whose truth is chosen here.

A. The D-051 (iv) transient test (noise_windows.transient_start) on
   simulated cells with the archive's SS layout (37-pulse trains, onset
   period 10025 samples, stored (-10, +200) ms windows, some pulses not
   stored) at four layouts: 2 sweeps with K = 40 and K = 63 stored windows,
   6 sweeps (K = 199), 10 sweeps (K = 348). Noise alone and noise + an RC
   pulse response (R_in 113 MOhm, tau_m 28.1 ms, +/-200 pA x 0.5 ms).
   Reported: the noise floor of the test statistic (SD of a 5 ms bin mean
   of r_odd under noise alone, late bins), the test as written
   (kappa_tr = 0.1) and with the option transient_floor_z = 3.
B. The spread of sigma_b_hat from (T1) (noise_fit.tilt_from_semivariogram
   on noise_cell_stats.window_mean_semivariogram, at the TRUE law) per
   layout, against sqrt(chi2_S_eff / S_eff); and a pooled cohort of 18
   cells (7 x 2 sweeps K 40, 7 x 2 sweeps K 63, 1 x 6, 3 x 10 sweeps).
C. How much sigma_b moves the binding function q_sim (common random
   numbers, M = 3000) relative to the statistics' simulated SD, and the
   linearised shift of u_hat it causes, on the fixture cell of
   smoke_noise_fit.py.
D. A candidate fix for A: a per-bin threshold max(kappa_tr sigma_SS,
   z x SD of r_odd's bin mean simulated at the cell's layout and law).

Run (from anywhere; about 17 min on one core -- A 4.5, B+C 9, D 3.5 min;
the checks can run separately, e.g. in parallel on two cores):
    python check_findings_2026-10-23.py          # all four
    python check_findings_2026-10-23.py A        # one check
Results of the run behind the explanation of 2026-10-23: results_2026-10-23.txt.
Pure ASCII, LF.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Configuration (every parameter of the checks)
# ---------------------------------------------------------------------------
# the Ih Fit folder of a checkout with Blocks 1-4 (env IH_FIT_DIR; default: the sandbox of 2026-10-23)
REPO_IH_FIT = Path(os.environ.get("IH_FIT_DIR", str(Path(__file__).resolve().parents[2])))   # repo: Ih Fit/dev/<this folder>
CFG = dict(
    seed=20261023,
    fs=50000.0, period=10025, train=37, pre_n=500, post_n=10000,
    theta=(0.0380, 0.175, 0.0508, 0.0366),          # theta_cf: sigma_f mV, rho_f, sigma_s mV, tau_s s
    sigma_b_A=(0.0, 0.1),                           # mV/s, check A
    sigma_b_B=0.1,                                  # mV/s, check B
    r_in_mohm=113.0, tau_m_ms=28.143, pulse_n=25, pulse_pA=200.0,
    layouts={
        "2 sweeps, K 40": dict(n_sweeps=2, n_drop=34),
        "2 sweeps, K 63": dict(n_sweeps=2, drops={0: (2, 3, 11, 13, 20, 29, 36), 1: (0, 5, 6, 27)}),
        "6 sweeps, K 199": dict(n_sweeps=6, n_drop=23),
        "10 sweeps, K 348": dict(n_sweeps=10, n_drop=22),
    },
    n_rep_A=200,
    n_rep_B={"2 sweeps, K 40": 2000, "2 sweeps, K 63": 2000, "6 sweeps, K 199": 600, "10 sweeps, K 348": 400},
    kappa_tr=0.1, z=3.0, bin_ms=5.0, t_floor_ms=80.0, fallback_ms=190.0,
    late_bins_ms=(100.0, 200.0),
    t_ss_B_ms=80.0, lag_periods=4,
    cohort=(("2 sweeps, K 40", 7), ("2 sweeps, K 63", 7), ("6 sweeps, K 199", 1), ("10 sweeps, K 348", 3)),
    n_cohorts=4000,
    layouts_D=("2 sweeps, K 63", "10 sweeps, K 348"), sigma_b_D=0.1, n_null_D=300, n_test_D=300, z_D=(3.0, 3.5, 4.0),
    M_C=3000, B_boot=2000, sigma_b_C=(0.0, 0.1, 0.17), R_C=400, fd_u_C=(0.02, 0.01, 0.05, 0.05),
)

sys.path.insert(0, str(REPO_IH_FIT))
import noise_cell_stats as ncs      # noqa: E402
import noise_fit as nf              # noqa: E402
import noise_law as nl              # noqa: E402
import noise_windows as nw          # noqa: E402
import smoke_noise_fit as snf       # noqa: E402  (fixture cell builder)
import smoke_noise_windows as snw   # noqa: E402  (rc_response, single_pulse_response, expected_t_tr)

FS = CFG["fs"]
DT = 1.0 / FS
P = CFG["period"]
T_TRAIN = CFG["train"]
PRE_N = CFG["pre_n"]
N_W = CFG["pre_n"] + CFG["post_n"]
L_SWEEP = (T_TRAIN - 1) * P + N_W
LAW = nl.NoiseLaw(*CFG["theta"])


# ---------------------------------------------------------------------------
# Simulation of a cell at a layout (data generation only)
# ---------------------------------------------------------------------------
def build_layout(spec, rng):
    S = spec["n_sweeps"]
    if "drops" in spec:
        drop = {s: set(spec["drops"].get(s, ())) for s in range(S)}
    else:
        flat = rng.choice(S * T_TRAIN, size=spec["n_drop"], replace=False)
        drop = {s: set() for s in range(S)}
        for f in flat:
            drop[int(f) // T_TRAIN].add(int(f) % T_TRAIN)
    sweep, pos = [], []
    for s in range(S):
        for k in range(T_TRAIN):
            if k not in drop[s]:
                sweep.append(s)
                pos.append(k)
    return np.array(sweep), np.array(pos)


def response_windows(pos):
    """RC response of one sweep's 37 alternating pulses, cut at the stored positions."""
    i = np.zeros(L_SWEEP)
    for k in range(T_TRAIN):
        s0 = k * P + PRE_N
        i[s0:s0 + CFG["pulse_n"]] = CFG["pulse_pA"] if k % 2 == 0 else -CFG["pulse_pA"]
    v = snw.rc_response(i, r_mohm=CFG["r_in_mohm"], tau_ms=CFG["tau_m_ms"])
    idx = pos[:, None] * P + np.arange(N_W)[None, :]
    return v[idx]


def simulate_noise_windows(sigma_b, sweep, pos, rng):
    S = int(sweep.max()) + 1
    units = nl.draw_units(rng, S, L_SWEEP)
    tr = nl.simulate_traces(LAW, DT, units.fast, units.slow, xi_tilt=units.tilt, sigma_b=sigma_b)
    idx = pos[:, None] * P + np.arange(N_W)[None, :]
    return tr[sweep[:, None], idx]


def make_ss(v, sweep, pos):
    S = int(sweep.max()) + 1
    return nw.SSWindows(specimen_id=0, fs=FS, dt=DT, t=(np.arange(N_W) - PRE_N) / FS, v=-60.0 + v,
                        polarity_is_dep=(pos % 2 == 0), sweep_number=sweep + 100, position=pos,
                        train_length=T_TRAIN, period_samples=P, pre_n=PRE_N, n_raw=S * T_TRAIN, n_sweeps=S)


def pairs_per_sweep(sweep, pos, lag):
    out = []
    for s in np.unique(sweep):
        p = set(pos[sweep == s].tolist())
        out.append(sum(1 for q in p if q + lag in p))
    return np.array(out)


# ---------------------------------------------------------------------------
# Check A: the transient test
# ---------------------------------------------------------------------------
def all_bin_means(r, t_ms):
    edges = np.arange(0.0, 200.0 + 1e-9, CFG["bin_ms"])
    return np.array([r[(t_ms >= a) & (t_ms < b)].mean() for a, b in zip(edges[:-1], edges[1:])])


def late_bin_means(r_odd, t_ms):
    lo, hi = CFG["late_bins_ms"]
    edges = np.arange(lo, hi + 1e-9, CFG["bin_ms"])
    return np.array([r_odd[(t_ms >= a) & (t_ms < b)].mean() for a, b in zip(edges[:-1], edges[1:])])


def check_a(rng):
    print("\n=== A. transient test (D-051 (iv)); theta_cf; kappa_tr %.2f, 5 ms bins, z = %.0f ===" % (CFG["kappa_tr"], CFG["z"]))
    cfg0 = nw.WindowConfig(t_ss_floor_ms=0.0)                                    # t_tr itself
    cfgz = nw.WindowConfig(t_ss_floor_ms=0.0, transient_floor_z=CFG["z"])
    t_ms = (np.arange(N_W) - PRE_N) / FS * 1e3
    resp1 = snw.single_pulse_response(True, tau_ms=CFG["tau_m_ms"])            # one pulse, no cross-talk
    rows = []
    for name, spec in CFG["layouts"].items():
        sweep, pos = build_layout(spec, rng)
        rw = response_windows(pos)
        K = pos.size
        K_dep = int((pos % 2 == 0).sum())
        for sb in CFG["sigma_b_A"]:
            sig, fl, bins, tn, tnz, trr, trz, thz = [], [], [], [], [], [], [], []
            odd_all, even_all, last_bad_z = [], [], []
            t0 = time.time()
            for _ in range(CFG["n_rep_A"]):
                vn = simulate_noise_windows(sb, sweep, pos, rng)
                ssn = make_ss(vn, sweep, pos)
                a = nw.transient_start(ssn, cfg0)
                az = nw.transient_start(ssn, cfgz)
                ssr = make_ss(vn + rw, sweep, pos)
                b = nw.transient_start(ssr, cfg0)
                bz = nw.transient_start(ssr, cfgz)
                pre = ssn.v[:, :PRE_N]
                r = ssn.v - pre.mean(axis=1, keepdims=True)
                dep = ssn.polarity_is_dep
                r_even = 0.5 * (r[dep].mean(axis=0) + r[~dep].mean(axis=0))
                odd_all.append(all_bin_means(a.r_odd, t_ms) / a.sigma_ss_mV)
                even_all.append(all_bin_means(r_even, t_ms) / a.sigma_ss_mV)
                bad = np.flatnonzero(az.bin_exceeds)
                last_bad_z.append(float(az.bin_edges_ms[bad[-1]]) if bad.size else np.nan)
                sig.append(a.sigma_ss_mV)
                fl.append(az.floor_mV / az.sigma_ss_mV)
                bins.append(late_bin_means(a.r_odd, t_ms) / a.sigma_ss_mV)
                tn.append(a.t_tr_ms); tnz.append(az.t_tr_ms)
                trr.append(b.t_tr_ms); trz.append(bz.t_tr_ms); thz.append(bz.threshold_mV / bz.sigma_ss_mV)
            sig, fl, bins = np.array(sig), np.array(fl), np.array(bins)
            tn, tnz, trr, trz, thz = map(np.array, (tn, tnz, trr, trz, thz))
            stat_floor = float(np.sqrt(np.mean(bins.var(axis=0, ddof=1))))     # SD of a late 5 ms bin mean / sigma_SS
            p_exceed = float(np.mean(np.abs(bins) >= CFG["kappa_tr"]))
            sig_med = float(np.median(sig))
            t_noiseless_kappa = snw.expected_t_tr(resp1, CFG["kappa_tr"] * sig_med)
            t_noiseless_z = snw.expected_t_tr(resp1, float(np.median(thz)) * sig_med)
            enl_none = lambda t: float(np.mean(np.maximum(CFG["t_floor_ms"], t) >= CFG["fallback_ms"]))
            q = lambda x: "%.0f [%.0f-%.0f]" % (np.median(x), np.percentile(x, 10), np.percentile(x, 90))
            print("\n-- %s (K = %d: %d dep, %d hyp), sigma_b = %.2f mV/s, %d cells (%.0f s)"
                  % (name, K, K_dep, K - K_dep, sb, CFG["n_rep_A"], time.time() - t0))
            print("   sigma_SS median %.4f mV" % sig_med)
            print("   noise floor of the statistic: SD of a 5 ms bin mean of r_odd in [100, 200) ms = %.3f sigma_SS"
                  " (kappa_tr = %.2f); P(|bin mean| >= kappa_tr sigma_SS) per bin = %.2f" % (stat_floor, CFG["kappa_tr"], p_exceed))
            print("   option's floor estimate (per-sample SD of r_even, t >= 0): median %.3f sigma_SS [10-90%%: %.3f-%.3f];"
                  " z x floor = %.3f sigma_SS" % (np.median(fl), np.percentile(fl, 10), np.percentile(fl, 90), CFG["z"] * np.median(fl)))
            print("   NOISE ONLY  as written: t_tr %s ms; ss_enlarged_none (t_SS >= 190) in %.0f %% of cells"
                  % (q(tn), 100 * enl_none(tn)))
            print("   NOISE ONLY  z = 3     : t_tr %s ms; t_tr > 80 ms (noise alone moves the start) in %.0f %%; ss_enlarged_none %.0f %%"
                  % (q(tnz), 100 * float(np.mean(tnz > CFG["t_floor_ms"])), 100 * enl_none(tnz)))
            print("   + RESPONSE  as written: t_tr %s ms; ss_enlarged_none %.0f %%; noiseless response at kappa sigma_SS: %.0f ms"
                  % (q(trr), 100 * enl_none(trr), t_noiseless_kappa))
            print("   + RESPONSE  z = 3     : t_tr %s ms; ss_enlarged_none %.0f %%; threshold %.3f sigma_SS; noiseless response at it: %.0f ms;"
                  " t_SS = max(80, t_tr) %s ms" % (q(trz), 100 * enl_none(trz), np.median(thz), t_noiseless_z,
                                                 q(np.maximum(CFG["t_floor_ms"], trz))))
            sd_odd = np.array(odd_all).std(axis=0, ddof=1)
            sd_even = np.array(even_all).std(axis=0, ddof=1)
            show = [0, 10, 20, 30, 37, 38, 39]
            print("   per-bin SD of the bin mean (noise only) / sigma_SS, bins starting at %s ms: r_odd %s; r_even %s"
                  % ([int(5 * i) for i in show], np.array2string(sd_odd[show], precision=3), np.array2string(sd_even[show], precision=3)))
            lb = np.array(last_bad_z)
            lb = lb[np.isfinite(lb)]
            if lb.size:
                print("   NOISE ONLY  z = 3: in the %d cells with an exceeding bin, the last exceeding bin starts at (ms): median %.0f, share in [190, 200): %.2f"
                      % (lb.size, np.median(lb), float(np.mean(lb >= 190.0))))
            rows.append((name, sb, stat_floor, enl_none(tn), np.median(trz)))
    return rows


# ---------------------------------------------------------------------------
# Check D: a per-bin null simulated at the cell's layout (a candidate fix)
# ---------------------------------------------------------------------------
def tr_from_exceeds(exceeds, edges):
    """The transient test's rule: first boundary after which no later bin exceeds."""
    bad = np.flatnonzero(exceeds)
    if bad.size == 0:
        return float(edges[0])
    return float(edges[bad[-1] + 1])


def check_d(rng):
    print("\n=== D. candidate fix: threshold per bin = max(kappa sigma_SS, z x SD of r_odd's bin mean simulated at the layout); theta_cf ===")
    t_ms = (np.arange(N_W) - PRE_N) / FS * 1e3
    edges = np.arange(0.0, 200.0 + 1e-9, CFG["bin_ms"])
    for name in CFG["layouts_D"]:
        spec = CFG["layouts"][name]
        sweep, pos = build_layout(spec, rng)
        rw = response_windows(pos)
        sb = CFG["sigma_b_D"]
        t0 = time.time()
        null = []
        for _ in range(CFG["n_null_D"]):
            a = nw.transient_start(make_ss(simulate_noise_windows(sb, sweep, pos, rng), sweep, pos), nw.WindowConfig(t_ss_floor_ms=0.0))
            null.append(all_bin_means(a.r_odd, t_ms))
        sd_bin = np.array(null).std(axis=0, ddof=1)                               # mV, per bin
        res = {z: ([], []) for z in CFG["z_D"]}
        for _ in range(CFG["n_test_D"]):
            vn = simulate_noise_windows(sb, sweep, pos, rng)
            for with_resp in (False, True):
                a = nw.transient_start(make_ss(vn + (rw if with_resp else 0.0), sweep, pos), nw.WindowConfig(t_ss_floor_ms=0.0))
                bm = np.abs(all_bin_means(a.r_odd, t_ms))
                for z in CFG["z_D"]:
                    thr = np.maximum(CFG["kappa_tr"] * a.sigma_ss_mV, z * sd_bin)
                    res[z][int(with_resp)].append(tr_from_exceeds(bm >= thr, edges))
        # display only: the null SD is divided by 0.0403 mV, the median sigma_SS of check A at theta_cf
        print("-- %s, sigma_b %.2f: null from %d cells, tested on %d (%.0f s); null SD per bin / sigma_SS at 0, 100, 150, 185, 195 ms: %s"
              % (name, sb, CFG["n_null_D"], CFG["n_test_D"], time.time() - t0,
                 np.array2string(sd_bin[[0, 20, 30, 37, 39]] / 0.0403, precision=3)))
        for z in CFG["z_D"]:
            tn, tr_ = np.array(res[z][0]), np.array(res[z][1])
            print("   z = %.1f: NOISE ONLY t_tr > 80 ms in %.1f %%, >= 190 ms in %.1f %%;  + RESPONSE t_SS = max(80, t_tr) median %.0f [10-90%%: %.0f-%.0f] ms, >= 190 ms in %.1f %%"
                  % (z, 100 * np.mean(tn > 80.0), 100 * np.mean(tn >= 190.0), np.median(np.maximum(80.0, tr_)),
                     np.percentile(np.maximum(80.0, tr_), 10), np.percentile(np.maximum(80.0, tr_), 90), 100 * np.mean(tr_ >= 190.0)))


# ---------------------------------------------------------------------------
# Check B: the spread of sigma_b_hat from (T1)
# ---------------------------------------------------------------------------
def check_b(rng):
    sb = CFG["sigma_b_B"]
    print("\n=== B. sigma_b_hat from (T1) at the true law; sigma_b = %.2f mV/s; t_SS = %.0f ms; h* = %d periods ==="
          % (sb, CFG["t_ss_B_ms"], CFG["lag_periods"]))
    excess = {}
    for name, spec in CFG["layouts"].items():
        sweep, pos = build_layout(spec, rng)
        npp = pairs_per_sweep(sweep, pos, CFG["lag_periods"])
        S_eff = float(npp.sum() ** 2 / np.sum(npp ** 2))
        j0 = PRE_N + int(round(CFG["t_ss_B_ms"] * 1e-3 * FS))
        lay = SimpleNamespace(period_samples=P, dt=DT, m=N_W - j0)
        e, N = [], []
        t0 = time.time()
        for _ in range(CFG["n_rep_B"][name]):
            vn = simulate_noise_windows(sb, sweep, pos, rng)
            ss = make_ss(vn, sweep, pos)
            X, _j0 = nw.enlarged_windows(ss, CFG["t_ss_B_ms"])
            Xt, _, _ = nw.subtract_mean_response(X, ss.polarity_is_dep)
            V_hat, n_pairs = ncs.window_mean_semivariogram(Xt, ss, CFG["lag_periods"])
            tilt = nf.tilt_from_semivariogram(V_hat, n_pairs, LAW, lay, CFG["lag_periods"])
            e.append(2.0 * (V_hat - tilt.V_model) / tilt.h_star ** 2)              # unclipped sigma_b^2 estimate
            N.append(n_pairs)
        e, N = np.array(e), np.array(N)
        excess[name] = (e, N)
        r = np.sqrt(np.clip(e, 0.0, None)) / sb
        qs = np.percentile(r, [2.5, 50, 97.5])
        cq = np.sqrt(chi2.ppf([0.025, 0.5, 0.975], S_eff) / S_eff)
        print("-- %s: pairs per sweep %s (S_eff %.2f), %d cells (%.0f s): sigma_b_hat / sigma_b  2.5/50/97.5 %% = %.2f / %.2f / %.2f;"
              " clipped at 0 in %.1f %%; mean of sigma_b_hat^2 / sigma_b^2 = %.3f (SE %.3f).  b draws alone, sqrt(chi2_S_eff / S_eff): %.2f / %.2f / %.2f"
              % (name, npp.tolist(), S_eff, e.size, time.time() - t0, qs[0], qs[1], qs[2], 100 * np.mean(e < 0),
                 np.mean(e) / sb ** 2, np.std(e, ddof=1) / np.sqrt(e.size) / sb ** 2, cq[0], cq[1], cq[2]))
    # pooled cohort: one replicate per cell, pair-weighted pooled excess (all pairs pooled)
    pooled = []
    for _ in range(CFG["n_cohorts"]):
        num = den = 0.0
        for name, n_cells in CFG["cohort"]:
            e, N = excess[name]
            idx = rng.choice(e.size, size=n_cells, replace=False)
            num += float(np.sum(N[idx] * e[idx]))
            den += float(np.sum(N[idx]))
        pooled.append(num / den)
    pooled = np.array(pooled)
    rp = np.sqrt(np.clip(pooled, 0.0, None)) / sb
    qp = np.percentile(rp, [2.5, 50, 97.5])
    n_sw = sum(CFG["layouts"][nm]["n_sweeps"] * c for nm, c in CFG["cohort"])
    print("-- pooled cohort (%s; %d SS sweeps; common sigma_b): sigma_b_hat / sigma_b 2.5/50/97.5 %% = %.2f / %.2f / %.2f"
          % (", ".join("%d x %s" % (c, nm) for nm, c in CFG["cohort"]), n_sw, qp[0], qp[1], qp[2]))


# ---------------------------------------------------------------------------
# Check C: sensitivity of q_sim to sigma_b
# ---------------------------------------------------------------------------
def check_c():
    print("\n=== C. q_sim(theta_cf, sigma_b), CRN, M = %d, on the smoke_noise_fit fixture cell; differences / bootstrap SE ===" % CFG["M_C"])
    tmp = tempfile.mkdtemp(prefix="check_c_")
    lay, tg, _ss, _truth = snf.build_cell(tmp, np.random.default_rng(2026))
    rng_boot = np.random.default_rng(CFG["seed"])
    SE = np.concatenate([ncs.bootstrap_median_se(tg.per_window_ss, CFG["B_boot"], rng_boot),
                         ncs.bootstrap_median_se(tg.per_window_ls, CFG["B_boot"], rng_boot)])
    cfg = nf.NoiseFitConfig(M_binding=CFG["M_C"])
    q = {}
    for sb in CFG["sigma_b_C"]:
        t0 = time.time()
        q[sb] = nf.binding_function(LAW, sb, lay, cfg, CFG["seed"])
        print("   q_sim at sigma_b = %.2f mV/s (%.0f s)" % (sb, time.time() - t0))
    base = CFG["sigma_b_C"][1]
    t0 = time.time()
    sig = nf.simulated_covariance(LAW, base, lay, CFG["R_C"], np.random.default_rng(CFG["seed"] + 1))
    sd_stat = np.sqrt(np.diag(sig.Sigma))
    print("   Sigma_hat at theta_cf, sigma_b %.2f, R = %d (%.0f s)" % (base, CFG["R_C"], time.time() - t0))
    # Jacobian of q_sim in u = (log sigma_f, rho_f, log sigma_s, log tau_s), forward differences, CRN
    u0 = nf.law_to_u(LAW)
    Jac = np.empty((15, 4))
    for i, h in enumerate(CFG["fd_u_C"]):
        u1 = u0.copy(); u1[i] += h
        Jac[:, i] = (nf.binding_function(nf.u_to_law(u1), base, lay, cfg, CFG["seed"]) - q[base]) / h
    Winv = np.linalg.inv(sig.Sigma)
    F = Jac.T @ Winv @ Jac
    cov_u = np.linalg.inv(F)
    sd_u = np.sqrt(np.diag(cov_u))
    print("   SD of u_hat (efficient bound (J^T Sigma^-1 J)^-1 at theta_cf, this layout): %s  [log sigma_f, rho_f, log sigma_s, log tau_s]"
          % np.array2string(sd_u, precision=3))
    for sb in (CFG["sigma_b_C"][0], CFG["sigma_b_C"][2]):
        dq = q[sb] - q[base]
        d_se = dq / SE
        d_sd = dq / sd_stat
        j = int(np.nanargmax(np.abs(d_sd)))
        du = -cov_u @ Jac.T @ Winv @ dq
        print("   simulator sigma_b %.2f instead of the data's %.2f: dq / SD(stat) max %.2f (%s), per statistic %s; dq / bootstrap SE max %.2f;"
              " linearised shift of u_hat / SD(u_hat) = %s"
              % (sb, base, abs(d_sd[j]), ncs.STAT_LABELS[j], np.array2string(d_sd, precision=2), float(np.nanmax(np.abs(d_se))),
                 np.array2string(du / sd_u, precision=2)))


def main(which="ABCD"):
    t_all = time.time()
    seeds = np.random.SeedSequence(CFG["seed"]).spawn(2)
    if "A" in which:
        check_a(np.random.default_rng(seeds[0]))
    if "B" in which:
        check_b(np.random.default_rng(seeds[1]))
    if "C" in which:
        check_c()
    if "D" in which:
        check_d(np.random.default_rng(np.random.SeedSequence(CFG["seed"] + 7)))
    print("\nDONE %s (%.0f s)" % (which, time.time() - t_all))


if __name__ == "__main__":
    # python check_findings_2026-10-23.py [A|B|C|D|AB|...]   (default: all four)
    main(sys.argv[1] if len(sys.argv) > 1 else "ABCD")
