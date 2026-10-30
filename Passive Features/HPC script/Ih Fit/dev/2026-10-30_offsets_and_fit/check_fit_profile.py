"""check_fit_profile.py -- standalone check (not pipeline code), 2026-10-30:
the window-start decay fit of D-110 (Block 3's noise_windows.fit_decay), as
explained to the user, checked on simulated groups, with the explanation's
figure.

The fit. A group's median window m(t) (mV) on the samples t_j in [5, 200) ms
is fitted by y = B + A exp(-t / tau). For each fixed tau the best (A, B) is a
linear least-squares problem with a closed form (the regression of y on
e_j = exp(-t_j / tau)), so the residual becomes a function of tau alone,
RSS(tau) = S_yy - S_ey^2 / S_ee (S the centred sums), the profile. tau_hat is
its minimum in [1, 1000] ms: a 301-point log grid, a bounded Brent step
between the grid minimum's neighbours, the root of dRSS/dtau. A grid minimum
at a bound is a failed fit.

Claims tested (predictions fixed before running), on the groups of 150
simulated 2-sweep cells per tilt (theta_cf; RC 113 MOhm / 28.143 ms; 27 %
drops; tilt per sweep ~ N(0, sigma_b^2), sigma_b 0 and 0.1 mV/s):
 PU1  exact within the bounds: for every group with an interior fit, no
      point of a 3001-point log grid over [1, 1000] ms has a profile RSS
      below RSS(tau_hat) by more than 1e-9 of it;
 PU2  at sigma_b = 0, at least 99 % of the groups have exactly one interior
      local minimum of the profile on the 3001-point grid;
 PT1  the three-start curve_fit of f293d4a (the fit D-110 replaced; copied
      below) never beats it inside the bounds: whenever its tau lies in
      [1, 1000] ms, RSS_exact <= RSS_three-start (1 + 1e-9).
Reported besides: the share of groups with more than one local minimum at
sigma_b = 0.1; where the three-start fit was worse (by > 1e-6 of the RSS) or
left the bounds, and what that did to the group's start t_sp.
Worked case and figure: the first sweep's depolarising group of the first
simulated untilted cell (seed below): fig_fit_profile.png.
Seed 20261052. Uses check_start_noise.py's simulator (../2026-10-29_start_noise).
Run (from this folder; about 8 min):  python check_fit_profile.py
The worked case and the figure alone (seconds):  python check_fit_profile.py --figure-only
Pure ASCII, LF.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeWarning, brentq, curve_fit, minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-29_start_noise"))
import check_start_noise as csn            # noqa: E402  (simulator, t0; puts the Ih Fit folder on the path)

nw = csn.nw

CFG = dict(seed=20261052, n_cells=150, sweeps=2, sigma_b=(0.0, 0.1), dense_n=3001, chunk=1000,
           pu1_rel=1e-9, pu2_share=0.99, pt1_rel=1e-9, worse_rel=1e-6, min_rel=1e-12,
           old_tau0_ms=(5.0, 20.0, 80.0), old_maxfev=2000, figure="fig_fit_profile.png")
WCFG = nw.WindowConfig()


# ---------------------------------------------------------------- analysis
def fit_window(ss):
    """t (ms) of the fitted samples and the boolean selector, [fit_start, 200) ms."""
    t_ms = ss.t * 1e3
    sel = np.arange(ss.n_w) >= ss.pre_n + int(round(WCFG.fit_start_ms * 1e-3 * ss.fs))
    return t_ms, sel


def profile_many(t, y, taus, chunk):
    """RSS(tau) for many tau at once: S_yy - S_ey^2 / S_ee (centred sums)."""
    yc = y - y.mean()
    syy = float(yc @ yc)
    out = np.empty(taus.size)
    for a in range(0, taus.size, chunk):
        E = np.exp(-t[None, :] / taus[a:a + chunk, None])
        E -= E.mean(axis=1, keepdims=True)
        sey = E @ yc
        see = np.einsum("ij,ij->i", E, E)
        out[a:a + chunk] = syy - sey ** 2 / see
    return out


def interior_minima(rss, rel):
    """Number of strict interior local minima of a sampled profile (tolerance rel of the value)."""
    tol = rel * np.abs(rss[1:-1])
    return int(np.sum((rss[1:-1] < rss[:-2] - tol) & (rss[1:-1] < rss[2:] - tol)))


def fit_steps(t, y):
    """fit_decay's steps, kept for the figure: grid, its RSS, the grid minimum, the Brent
    bracket and result, the polished root. Must agree with nw.fit_decay (asserted)."""
    lo, hi = WCFG.decay_tau_bounds_ms
    grid = np.geomspace(lo, hi, WCFG.decay_grid_n)
    rss = np.array([nw.profile_decay(t, y, g)[2] for g in grid])
    i = int(np.argmin(rss))
    res = minimize_scalar(lambda lt: nw.profile_decay(t, y, float(np.exp(lt)))[2],
                          bounds=(float(np.log(grid[i - 1])), float(np.log(grid[i + 1]))),
                          method="bounded", options={"xatol": 1e-12, "maxiter": 500})
    tau_brent = float(np.exp(res.x))
    f = nw.fit_decay(t, y, WCFG)
    return dict(grid=grid, rss=rss, i=i, bracket=(grid[i - 1], grid[i + 1]), tau_brent=tau_brent,
                nfev_brent=int(res.nfev), fit=f)


def old_three_start_fit(t, y):
    """The fit of f293d4a (copied): curve_fit 'lm', unbounded, from tau0 = 5, 20, 80 ms with
    (A0, B0) the linear least-squares solution at tau0; the lowest RSS among the converged."""
    best = None
    for tau0 in CFG["old_tau0_ms"]:
        basis = np.column_stack([np.exp(-t / tau0), np.ones_like(t)])
        a0, b0 = np.linalg.lstsq(basis, y, rcond=None)[0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                    B, A, tau = curve_fit(lambda x, B, A, tau: B + A * np.exp(-x / tau), t, y,
                                          p0=(b0, a0, tau0), method="lm", maxfev=CFG["old_maxfev"])[0]
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        if not (np.isfinite(A) and np.isfinite(B) and np.isfinite(tau) and tau > 0):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            rss = float(np.sum((y - B - A * np.exp(-t / tau)) ** 2))
        if np.isfinite(rss) and (best is None or rss < best[3]):
            best = (float(A), float(tau), float(B), rss)
    return best


def start_of(A, tau, thr):
    return tau * np.log(abs(A) / thr) if abs(A) > thr else 0.0


def analyse_groups(rng, sigma_b, resp):
    """Per valid group: exact fit, dense profile, minima count, the old fit."""
    rows = []
    for _ in range(CFG["n_cells"]):
        ss = csn.simulate_cell(CFG["sweeps"], sigma_b, resp, rng)
        st = nw.decay_start(ss)
        thr = WCFG.kappa_tr * st.sigma_ss_mV
        t_ms, sel = fit_window(ss)
        t = t_ms[sel]
        dense = np.geomspace(*WCFG.decay_tau_bounds_ms, CFG["dense_n"])
        for s in np.unique(ss.sweep_number):
            for dep in (True, False):
                idx = np.flatnonzero((ss.sweep_number == s) & (ss.polarity_is_dep == dep))
                if idx.size < WCFG.min_windows_per_group:
                    continue
                y = np.median(ss.v[idx], axis=0)[sel]
                f = nw.fit_decay(t, y, WCFG)
                rss_d = profile_many(t, y, dense, CFG["chunk"])
                old = old_three_start_fit(t, y)
                rows.append(dict(ok=f.ok, at_bound=f.at_bound, tau=f.tau_ms, A=f.A_mV, rss=f.rss,
                                 dense_min=float(rss_d.min()), n_min=interior_minima(rss_d, CFG["min_rel"]),
                                 old=old, thr=thr))
    return rows


def summarise(rows, label):
    ok = [r for r in rows if r["ok"]]
    pu1_bad = sum(r["dense_min"] < r["rss"] * (1 - CFG["pu1_rel"]) for r in ok)
    n_min = np.array([r["n_min"] for r in rows])
    old_in = [r for r in ok if r["old"] is not None and WCFG.decay_tau_bounds_ms[0] <= r["old"][1] <= WCFG.decay_tau_bounds_ms[1]]
    pt1_bad = sum(r["rss"] > r["old"][3] * (1 + CFG["pt1_rel"]) for r in old_in)
    worse = [r for r in old_in if r["old"][3] > r["rss"] * (1 + CFG["worse_rel"])]
    d_tsp = [abs(start_of(r["old"][0], r["old"][1], r["thr"]) - start_of(r["A"], r["tau"], r["thr"])) for r in worse]
    out_bounds = sum(r["old"] is not None and not (WCFG.decay_tau_bounds_ms[0] <= r["old"][1] <= WCFG.decay_tau_bounds_ms[1])
                     for r in rows)
    old_none = sum(r["old"] is None for r in rows)
    print("%s: %d groups, %d exact fits interior, %d at a bound; profile minima on the dense grid: "
          "one %d, two or more %d, none %d" % (label, len(rows), len(ok), sum(r["at_bound"] for r in rows),
                                               int(np.sum(n_min == 1)), int(np.sum(n_min >= 2)), int(np.sum(n_min == 0))))
    print("   dense grid below RSS(tau_hat): %d groups; three-start fit inside the bounds in %d groups, "
          "beats the exact fit in %d, worse by > 1e-6 in %d (|change of t_sp| max %.2f ms, median %.2f ms); "
          "three-start tau outside [1, 1000] ms in %d groups, no converged start in %d"
          % (pu1_bad, len(old_in), pt1_bad, len(worse), max(d_tsp) if d_tsp else 0.0,
             float(np.median(d_tsp)) if d_tsp else 0.0, out_bounds, old_none))
    for r in rows:
        old_out = r["old"] is not None and not (WCFG.decay_tau_bounds_ms[0] <= r["old"][1] <= WCFG.decay_tau_bounds_ms[1])
        if r["at_bound"] or old_out:
            old_txt = ("tau %.4g ms, A %.4g mV, t_sp %.1f ms" % (r["old"][1], r["old"][0],
                                                                start_of(r["old"][0], r["old"][1], r["thr"]))
                       if r["old"] is not None else "none")
            print("   group with no interior exact fit or a three-start tau outside the bounds: exact %s; "
                  "three-start %s" % ("at a bound (failed fit)" if r["at_bound"] else "tau %.4g ms" % r["tau"], old_txt))
    share_one = float(np.mean(n_min == 1)) if n_min.size else float("nan")
    return pu1_bad == 0, share_one, pt1_bad == 0


# ---------------------------------------------------------------- plotting
def plot_worked_case(ss, idx, steps, thr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid_c, c1, c2, surf = "#0b0b0b", "#52514e", "#d9d8d4", "#2a78d6", "#eb6834", "#fcfcfb"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False})
    f = steps["fit"]
    t_ms = ss.t * 1e3
    m = np.median(ss.v[idx], axis=0)
    t_sp = start_of(f.A_mV, f.tau_ms, thr)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.7), facecolor=surf)
    for ax in (ax1, ax2):
        ax.set_facecolor(surf)
    # (a) data space
    ax1.axvspan(WCFG.fit_start_ms, 200.0, color=grid_c, alpha=0.35, lw=0)
    ax1.plot(t_ms, m - f.B_mV, color="#8f8e89", lw=0.8, label=r"median window $m_{s,p}(t)-\hat B$ (simulated)")
    tf = t_ms[t_ms >= WCFG.fit_start_ms]
    ax1.plot(tf, f.A_mV * np.exp(-tf / f.tau_ms), color=c1, lw=2.0,
             label=r"fit $\hat A\,e^{-t/\hat\tau}$: $\hat\tau$ = %.2f ms, $\hat A$ = %.3f mV" % (f.tau_ms, f.A_mV))
    ax1.axvline(t_sp, color=c2, lw=1.5, ls="--",
                label=r"$t_{s,p}$ = %.1f ms, where $|\hat A|e^{-t/\hat\tau}=\kappa_{\rm tr}\hat\sigma_{\rm SS}(c)$" % t_sp)
    ax1.text(128, 0.40, "fitted range [5, 200) ms", color=ink2, fontsize=8, ha="left")
    ax1.axhline(0.0, color=ink2, lw=0.6)
    ax1.set_xlabel(r"$t$ from the pulse onset (ms)")
    ax1.set_ylabel(r"voltage relative to $\hat B$ (mV)")
    ax1.set_xlim(-10, 200)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    ax1.set_title("(a) data space: one group's median window", fontsize=9, loc="left", color=ink)
    # (b) parameter space: the profile
    g, r, i = steps["grid"], steps["rss"], steps["i"]
    fit_sel = t_ms >= WCFG.fit_start_ms
    tt, yy = t_ms[fit_sel], m[fit_sel]
    fine = np.geomspace(g[0], g[-1], 3001)
    ax2.plot(g, r, "o", ms=1.6, color="#a9a8a3", zorder=2, label="RSS at the 301 grid points")
    ax2.plot(fine, [nw.profile_decay(tt, yy, x)[2] for x in fine], color=c1, lw=1.6, zorder=3,
             label=r"profile RSS($\tau$) (A, B solved exactly at each $\tau$)")
    ax2.axvspan(*steps["bracket"], color=c2, alpha=0.25, lw=0, label="Brent bracket: the grid minimum's neighbours")
    ax2.axvline(f.tau_ms, color=c2, lw=1.5, ls="--", zorder=4, label=r"$\hat\tau$: the root of dRSS/d$\tau$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(g[0], g[-1])
    ax2.set_xlabel(r"$\tau$ (ms) over the bounds [1, 1000] ms")
    ax2.set_ylabel(r"RSS($\tau$) (mV$^2$)")
    ax2.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax2.set_title(r"(b) parameter space: the fit is the minimum of RSS($\tau$)", fontsize=9, loc="left", color=ink)
    ins = ax2.inset_axes([0.62, 0.68, 0.32, 0.24])
    zx = np.linspace(steps["bracket"][0], steps["bracket"][1], 201)
    zr = np.array([nw.profile_decay(tt, yy, x)[2] for x in zx])
    ins.plot(zx, (zr - f.rss) * 1e6, color=c1, lw=1.5)
    ins.plot(g[i - 1:i + 2], (r[i - 1:i + 2] - f.rss) * 1e6, "o", ms=3, color=ink2)
    ins.axvline(f.tau_ms, color=c2, lw=1.2, ls="--")
    ins.set_title(r"zoom on the bracket: RSS $-$ RSS($\hat\tau$) ($\mu$V$^2$)" "\n" r"against $\tau$ (ms)", fontsize=6.5, color=ink)
    ins.tick_params(labelsize=6.5)
    ins.set_facecolor(surf)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=surf)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    t_start = time.time()
    rng = np.random.default_rng(CFG["seed"])
    resp = csn.train_response()
    # worked case: the first untilted cell, first sweep, depolarising group
    ss = csn.simulate_cell(CFG["sweeps"], 0.0, resp, rng)
    st = nw.decay_start(ss)
    thr = WCFG.kappa_tr * st.sigma_ss_mV
    t_ms, sel = fit_window(ss)
    s0 = int(ss.sweep_number[0])
    idx = np.flatnonzero((ss.sweep_number == s0) & ss.polarity_is_dep)
    steps = fit_steps(t_ms[sel], np.median(ss.v[idx], axis=0)[sel])
    f = steps["fit"]
    assert abs(steps["tau_brent"] / f.tau_ms - 1.0) < 1e-6, "fit_steps disagrees with fit_decay"
    print("worked case: K = %d windows; grid minimum at tau = %.3f ms (index %d of 301); Brent in [%.3f, %.3f] ms "
          "-> %.6f ms (%d evaluations); polished root %.6f ms (%s); A_hat %.4f mV, B_hat %.4f mV, RSS %.6e mV^2; "
          "sigma_SS_hat %.4f mV, t_sp %.1f ms (noiseless t0 %.1f ms)"
          % (idx.size, steps["grid"][steps["i"]], steps["i"], *steps["bracket"], steps["tau_brent"],
             steps["nfev_brent"], f.tau_ms, f.message, f.A_mV, f.B_mV, f.rss, st.sigma_ss_mV,
             start_of(f.A_mV, f.tau_ms, thr), csn.t0_of(st.sigma_ss_mV)))
    plot_worked_case(ss, idx, steps, thr, str(HERE / CFG["figure"]))
    if "--figure-only" in sys.argv:
        return True
    verdict = {}
    for sb in CFG["sigma_b"]:
        rows = analyse_groups(rng, sb, resp)
        pu1, share_one, pt1 = summarise(rows, "sigma_b %.1f mV/s" % sb)
        verdict["PU1 (%.1f)" % sb] = pu1
        verdict["PT1 (%.1f)" % sb] = pt1
        if sb == 0.0:
            verdict["PU2"] = share_one >= CFG["pu2_share"]
        print("   share with exactly one interior minimum: %.4f   (%.0f s)" % (share_one, time.time() - t_start))
    print("  ".join("%s %s" % (k, "PASS" if v else "FAIL") for k, v in verdict.items())
          + "   (%.0f s)" % (time.time() - t_start))
    return all(verdict.values())


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
