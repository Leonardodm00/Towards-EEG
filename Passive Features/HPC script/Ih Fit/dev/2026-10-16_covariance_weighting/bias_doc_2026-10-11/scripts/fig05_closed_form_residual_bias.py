"""Figure 5 (plotted) and Check 3: why matching medians absorbs a biased statistic, and why the closed form leaves
a residual bias that more data do not remove.  Toy indirect inference with one parameter and one statistic.

Law:        pure OU with correlation time tau, unit SD (the statistic is scale-free), sampled every dt = 0.2 ms;
            truth tau0 = 40 ms.
Statistic:  T = rho_hat(10 ms) = S_L / S_0 on a demeaned window of n = 663 samples (132.6 ms), L = 50.
Binding:    q(tau) = med(T | tau, n), estimated by q_sim(tau) (document: q^sim) = median over M = 20 000 simulated windows with common
            random numbers (seed 2), on a grid of tau; closed form q_cf(tau) = E[S_L | tau, n] / E[S_0 | tau, n]
            (topic doc eqs. (9)-(10)).  Grid 5-300 ms; both curves must be increasing on it (asserted).
Data:       a pool of P = 100 000 windows at tau0 (seed 1).  For K = 25, 125, 625, 3125, 15625 (odd) and R = 2000
            replicates, K windows are resampled with replacement from the pool (seed 3) and their median q_hat is
            the data statistic; tau_hat = q^-1(q_hat) for each binding curve (linear interpolation on the grid).
            Because the pool is fixed, its own median error, like q_sim's, does not shrink with K; a reference median
            from 10^6 fresh windows (seed 4) splits the limit's offset between the two.
Predictions fixed before running: (i) tau_hat_sim is centred on tau0 for every K, with Pr(tau_hat_sim <= tau0) near
1/2 (exactly 1/2 for an exact binding curve, fresh data and odd K, document eq. (14)), up to the Monte Carlo error of q_sim;
(ii) tau_hat_cf converges, as K grows, to tau_inf with q_cf(tau_inf) = q(tau0), well below tau0, and
Pr(tau_hat_cf <= tau0) tends to 1.
Run:  python3 fig05_closed_form_residual_bias.py   (about 1 min; writes ../figures/fig05_closed_form_residual_bias.png
      and logs/check03.txt)
"""
import numpy as np
from scipy.signal import lfilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bias_common import expected_S

CONFIG = dict(dt=2e-4, n=663, lag=50, tau0=0.040,
              M=20_000, seed_model=2, P=100_000, seed_pool=1, chunk=10_000, P_ref=1_000_000, seed_ref=4,
              K_list=(25, 125, 625, 3125, 15625), R=2000, R_chunk=100, seed_resample=3,
              tau_grid=(0.005, 0.300, 150),
              out_png="../figures/fig05_closed_form_residual_bias.png", out_log="logs/check03.txt", dpi=160)
COL = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", purple="#7b3fbf", ink="#0b0b0b", ink2="#52514e",
           grey="#8a8781", surface="#fcfcfb")


# --- model and statistic --------------------------------------------------------------------------
def ou_windows(tau, Z, dt):
    """Rows of standard normals Z -> stationary OU windows (unit SD) with correlation time tau."""
    a = np.exp(-dt / tau)
    e = Z * np.sqrt(1 - a**2)
    e[:, 0] = Z[:, 0]
    return lfilter([1.0], [1.0, -a], e, axis=1)


def acf_stat(x, L):
    u = x - x.mean(axis=1, keepdims=True)
    return np.sum(u[:, :-L] * u[:, L:], axis=1) / np.sum(u * u, axis=1)


def q_closed_form(tau, cfg):
    gam = np.exp(-np.arange(cfg["n"]) * cfg["dt"] / tau)
    return expected_S(gam, cfg["n"], cfg["lag"]) / expected_S(gam, cfg["n"], 0)


# --- analysis ---------------------------------------------------------------------------------------
def binding_curves(cfg):
    taus = np.geomspace(*cfg["tau_grid"])
    Zm = np.random.default_rng(cfg["seed_model"]).standard_normal((cfg["M"], cfg["n"]))
    q_sim = np.array([np.median(acf_stat(ou_windows(t, Zm, cfg["dt"]), cfg["lag"])) for t in taus])
    q_cf = np.array([q_closed_form(t, cfg) for t in taus])
    return taus, q_sim, q_cf


def data_pool(cfg, P=None, seed=None):
    P = cfg["P"] if P is None else P
    rng = np.random.default_rng(cfg["seed_pool"] if seed is None else seed)
    out = []
    for start in range(0, P, cfg["chunk"]):
        m = min(cfg["chunk"], P - start)
        out.append(acf_stat(ou_windows(cfg["tau0"], rng.standard_normal((m, cfg["n"])), cfg["dt"]), cfg["lag"]))
    return np.concatenate(out)


def invert(q_curve, taus, q_hat):
    """tau with q_curve(tau) = q_hat, by linear interpolation; values beyond the grid are clipped to its ends.
    Returns the estimates and a code per value: -1 below the curve's range, +1 above it, 0 inside."""
    side = np.where(q_hat < q_curve[0], -1, np.where(q_hat > q_curve[-1], 1, 0))
    return np.interp(q_hat, q_curve, taus), side


def fits_by_K(pool, taus, q_sim, q_cf, cfg):
    rng = np.random.default_rng(cfg["seed_resample"])
    res = {}
    for K in cfg["K_list"]:
        q_hat = np.concatenate([np.median(rng.choice(pool, size=(cfg["R_chunk"], K), replace=True), axis=1)
                                for _ in range(cfg["R"] // cfg["R_chunk"])])
        t_sim, c_sim = invert(q_sim, taus, q_hat)
        t_cf, c_cf = invert(q_cf, taus, q_hat)
        res[K] = dict(sim=t_sim, cf=t_cf, clip_sim=(np.mean(c_sim < 0), np.mean(c_sim > 0)),
                      clip_cf=(np.mean(c_cf < 0), np.mean(c_cf > 0)))
    return res


# --- plotting ---------------------------------------------------------------------------------------
def plot(taus, q_sim, q_cf, q_pool, res, lim, cfg):
    tau0 = cfg["tau0"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.7), dpi=cfg["dpi"], facecolor=COL["surface"])
    for x in (a, b):
        x.set_facecolor(COL["surface"])
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            x.spines[s].set_color(COL["ink2"])
        x.tick_params(colors=COL["ink2"], labelsize=8.3)
    ms = taus * 1e3
    sel = (ms >= 10) & (ms <= 150)
    a.plot(ms[sel], np.exp(-cfg["lag"] * cfg["dt"] / taus[sel]), color=COL["orange"], lw=2.0,
           label=r"$\rho(10\,{\rm ms} \mid \tau) = e^{-10\,{\rm ms}/\tau}$: the law's autocorrelation")
    a.plot(ms[sel], q_cf[sel], color=COL["grey"], lw=2.0, ls=":",
           label=r"$q^{\rm cf}(\tau)$: closed form (ratio of expectations)")
    a.plot(ms[sel], q_sim[sel], color=COL["aqua"], lw=2.0,
           label=r"$q^{\rm sim}(\tau)$: median of $T$ over $M$ simulated windows" + "\n" + r"(estimates $q(\tau) = {\rm med}(T \mid \tau, n)$)")
    a.axhline(q_pool, color=COL["purple"], lw=1.2, ls="--")
    a.text(148, q_pool - 0.015, r"data median $\hat q$ as $K \to \infty$", color=COL["purple"], fontsize=8.0,
           ha="right", va="top")
    a.plot([lim["sim"] * 1e3] * 2, [-0.1, q_pool], color=COL["aqua"], lw=1.0, ls="--")
    a.plot([lim["cf"] * 1e3] * 2, [-0.1, q_pool], color=COL["grey"], lw=1.0, ls="--")
    a.text(lim["cf"] * 1e3 - 0.5, -0.07, r"$\tau_\infty$ (pool) = %.1f ms" % (lim["cf"] * 1e3), color=COL["ink"],
           fontsize=8.0, ha="right", va="bottom")
    a.text(lim["sim"] * 1e3 + 1.0, -0.07, r"%.1f ms ($\tau_0$ + Monte Carlo errors)" % (lim["sim"] * 1e3), color=COL["ink"],
           fontsize=8.0, ha="left", va="bottom")
    i0 = np.argmin(np.abs(taus - tau0))
    a.annotate("", xy=(ms[i0], q_cf[i0]), xytext=(ms[i0], q_sim[i0]),
               arrowprops=dict(arrowstyle="<->", color=COL["ink"], lw=1.1))
    a.annotate(r"$\delta(\tau_0, n) \approx q^{\rm cf} - q^{\rm sim}$ = %+.3f" % (q_cf[i0] - q_sim[i0]), xy=(ms[i0], (q_cf[i0] + q_sim[i0]) / 2),
               xytext=(55, 0.385), fontsize=8.0, color=COL["ink"], ha="left", va="center",
               arrowprops=dict(arrowstyle="-", color=COL["ink2"], lw=0.8))
    a.set_xscale("log")
    a.set_xticks([10, 20, 40, 80, 150])
    a.set_xticklabels(["10", "20", "40", "80", "150"])
    a.set_ylim(-0.1, 1.0)
    a.set_xlabel(r"$\tau$ (ms)", fontsize=9.0, color=COL["ink"])
    a.set_ylabel("statistic value (dimensionless)", fontsize=9.0, color=COL["ink"])
    a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), fontsize=7.4, frameon=True, facecolor=COL["surface"],
             edgecolor="none", framealpha=0.95)
    a.set_title("(a) the fit inverts a binding curve at the data median", fontsize=9.6, color=COL["ink"], loc="left")

    K = np.array(cfg["K_list"], dtype=float)
    for key, col, dx, lab in (("sim", COL["aqua"], 0.93, r"simulated medians: $\hat\tau_{\rm sim} = (q^{\rm sim})^{-1}(\hat q)$"),
                              ("cf", COL["grey"], 1.07, r"closed form: $\hat\tau_{\rm cf} = (q^{\rm cf})^{-1}(\hat q)$")):
        med = np.array([np.median(res[k][key]) for k in cfg["K_list"]]) * 1e3
        lo = np.array([np.quantile(res[k][key], 0.1) for k in cfg["K_list"]]) * 1e3
        hi = np.array([np.quantile(res[k][key], 0.9) for k in cfg["K_list"]]) * 1e3
        b.errorbar(K * dx, med, yerr=[med - lo, hi - med], fmt="o", color=col, ecolor=col, capsize=3, lw=1.6,
                   markersize=6, label=lab + "  (median, 10\u201390 %)")
    b.axhline(tau0 * 1e3, color=COL["orange"], lw=1.8)
    b.text(K[-1] * 1.9, tau0 * 1e3 + 1.0, r"$\tau_0$ = 40 ms", color=COL["orange"], fontsize=8.2, va="bottom")
    b.axhline(lim["cf"] * 1e3, color=COL["grey"], lw=1.0, ls="--")
    b.text(K[-1] * 1.9, lim["cf"] * 1e3 - 1.0, r"$\tau_\infty$ (pool) = %.1f ms" % (lim["cf"] * 1e3), color=COL["ink2"],
           fontsize=8.2, va="top")
    b.set_xscale("log")
    b.set_xticks(K)
    b.set_xticklabels([str(int(k)) for k in K])
    b.set_xlim(K[0] * 0.7, K[-1] * 6.0)
    b.set_xlabel(r"$K$: number of data windows whose median $\hat q$ is matched", fontsize=9.0, color=COL["ink"])
    b.set_ylabel(r"$\hat\tau_{\rm sim}$, $\hat\tau_{\rm cf}$ (ms)", fontsize=9.0, color=COL["ink"])
    b.legend(loc="upper right", fontsize=7.6, frameon=False)
    b.set_title(r"(b) more data: one estimate converges to $\tau_0$, the other to $\tau_\infty$", fontsize=9.6,
                color=COL["ink"], loc="left")
    fig.text(0.01, 0.012, r"Pure OU, n = 663 samples at 0.2 ms (132.6 ms), lag L = 50 (10 ms), $\tau_0$ = 40 ms; binding curves "
             "from M = 20 000 windows (common random numbers);\nR = 2000 data replicates per K, resampled with replacement from a pool of 100 000 windows.", fontsize=8.0,
             color=COL["ink2"], ha="left", va="bottom")
    fig.tight_layout(w_pad=4.0, rect=(0, 0.09, 1, 1))
    fig.savefig(cfg["out_png"], facecolor=COL["surface"])


def main(cfg=CONFIG):
    taus, q_sim, q_cf = binding_curves(cfg)
    mono_sim, mono_cf = bool(np.all(np.diff(q_sim) > 0)), bool(np.all(np.diff(q_cf) > 0))
    assert mono_sim and mono_cf, "a binding curve is not increasing on the grid: inversion would be ambiguous"
    pool = data_pool(cfg)
    q_pool = np.median(pool)
    lim = dict(sim=invert(q_sim, taus, np.array([q_pool]))[0][0], cf=invert(q_cf, taus, np.array([q_pool]))[0][0])
    i0 = np.argmin(np.abs(taus - cfg["tau0"]))
    h = 1e-3
    slope = (np.interp(cfg["tau0"] + h, taus, q_sim) - np.interp(cfg["tau0"] - h, taus, q_sim)) / (2 * h)
    delta0 = np.interp(cfg["tau0"], taus, q_cf) - np.interp(cfg["tau0"], taus, q_sim)
    ref = data_pool(cfg, P=cfg["P_ref"], seed=cfg["seed_ref"])
    q_ref = np.median(ref)
    se_ref = 1.2533 * ref.std(ddof=1) / np.sqrt(ref.size)       # normal approximation to the SE of a median
    t_ref = invert(q_sim, taus, np.array([q_ref]))[0][0]
    res = fits_by_K(pool, taus, q_sim, q_cf, cfg)
    plot(taus, q_sim, q_cf, q_pool, res, lim, cfg)
    lines = [f"grid: {taus.size} tau from {taus[0]*1e3:.1f} to {taus[-1]*1e3:.1f} ms; increasing: q_sim {mono_sim}, q_cf {mono_cf}",
             f"at tau0 = {cfg['tau0']*1e3:.0f} ms: rho(10 ms|tau0) = {np.exp(-cfg['lag']*cfg['dt']/cfg['tau0']):.4f};"
             f" q_sim = {np.interp(cfg['tau0'], taus, q_sim):.4f}; q_cf = {np.interp(cfg['tau0'], taus, q_cf):.4f};"
             f" delta = q_cf - q_sim = {delta0:+.4f}; pool median (P = {pool.size}) = {q_pool:.4f}; pool mean = {pool.mean():.4f}",
             f"slope dq_sim/dtau at tau0 = {slope:.3f} per s",
             f"limits as K -> inf: tau_inf (closed form) = {lim['cf']*1e3:.2f} ms; simulated medians -> {lim['sim']*1e3:.2f} ms"
             f" (tau0 + Monte Carlo errors)",
             f"reference median from {ref.size} fresh windows (seed {cfg['seed_ref']}): q(tau0) = {q_ref:.4f} (SE {se_ref:.4f});"
             f" mean {ref.mean():.4f} (SE {ref.std(ddof=1)/np.sqrt(ref.size):.4f})",
             f"offset of the simulated-medians limit, {1e3*(lim['sim']-cfg['tau0']):+.2f} ms = q_sim's error"
             f" {1e3*(t_ref-cfg['tau0']):+.2f} ms (q_sim(tau0) - q_ref = {np.interp(cfg['tau0'], taus, q_sim)-q_ref:+.4f})"
             f" + the pool's error {1e3*(lim['sim']-t_ref):+.2f} ms (pool median - q_ref = {q_pool-q_ref:+.4f});"
             f" 1 SE of q_ref is {1e3*se_ref/slope:.2f} ms",
             f"closed-form limit against the reference: q_cf^-1(q_ref) = {1e3*invert(q_cf, taus, np.array([q_ref]))[0][0]:.2f} ms",
             "median deviation of T from the law's autocorrelation, q_sim(tau) - rho(10 ms | tau), on the grid: " + "; ".join(
                 f"{t:g} ms {np.interp(t * 1e-3, taus, q_sim) - np.exp(-cfg['lag'] * cfg['dt'] / (t * 1e-3)):+.3f}"
                 for t in (10, 20, 40, 80, 150))]
    for K in cfg["K_list"]:
        r = res[K]
        f = lambda v: f"median {np.median(v)*1e3:6.2f} ms, 10-90 % [{np.quantile(v,0.1)*1e3:6.2f}, {np.quantile(v,0.9)*1e3:6.2f}]"
        cs, cc = r["clip_sim"], r["clip_cf"]
        lines.append(f"K = {K:5d}: sim: {f(r['sim'])}, Pr(<= tau0) {np.mean(r['sim'] <= cfg['tau0']):.3f}, "
                     f"clipped below/above {cs[0]:.3f}/{cs[1]:.3f}"
                     f" | cf: {f(r['cf'])}, Pr(<= tau0) {np.mean(r['cf'] <= cfg['tau0']):.3f}, "
                     f"clipped below/above {cc[0]:.3f}/{cc[1]:.3f}")
    text = "\n".join(lines)
    print(text)
    with open(cfg["out_log"], "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
