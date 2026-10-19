"""fig04_fig05_table.py -- Figures 4 and 5 of "The thin-plate spline in three dimensions" (2026-10-18; standalone;
first made as fig_tps_deep.py for the Step 4 explanation of the same day).

Figure 4 (shape): the table along one line of its input space, ln d at phi = 10 deg and alpha_hat = 1,
for six values of the smoothing lambda, against the fixture's truth and the rows near the line;
each curve labelled with its effective degrees of freedom df = trace(H).
Figure 5 (mechanism): the weights l_n(x) with which the spline averages the ratios at one point
d = 1 um, phi = 30 deg, alpha_hat = 1.2, at lambda 1 and 10, against the normalised distance |s(x_n) - s(x)|,
beside the Gaussian weights varpi_n / sum varpi of tau_hat and SE.
Colours: one blue ramp for the smoothing (light = small lambda_s), gray for the Gaussian window, so that
blue keeps meaning 'thin_plate_spline' across the document's figures. Both plotted from check_table_se_coverage.py's batch-0 table (seed 20261018, N_S = 1500, hetero tau(d),
alpha_hat := true alpha). Run (same folder):  python3 fig04_fig05_table.py
"""
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.interpolate import RBFInterpolator

import check_table_se_coverage as cov

PAL = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
           seq=("#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"), s1="#2a78d6", lam_lo="#86b6ef",
           lam_hi="#1c5cab", window="#898781")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "mathtext.fontset": "dejavusans",
                     "axes.edgecolor": PAL["axis"], "axes.labelcolor": PAL["ink2"],
                     "xtick.color": PAL["muted"], "ytick.color": PAL["muted"]})
MU, DEG = "\u00b5", "\u00b0"      # micro sign and degree sign, written as escapes (ASCII source)
FIG = dict(phi_deg=10.0, alpha=1.0, lams=(1e-4, 1e-2, 1.0, 10.0, 100.0, 1e4), near_phi_deg=5.0, near_lna=0.3,
           query=(1.0, 30.0, 1.2), kernel_lams=(1.0, 10.0))


def style(ax):
    ax.set_facecolor(PAL["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=PAL["grid"], lw=0.7)


def fig4(T):
    X, y, Xs, sc = T["X"], T["y"], T["Xs"], T["sc"]
    phi = math.radians(FIG["phi_deg"])
    lo, hi = cov.CFG["d_range"]
    dg = np.exp(np.linspace(math.log(lo), math.log(hi), 300))
    S = np.column_stack([np.log(dg), np.full(dg.size, phi), np.full(dg.size, math.log(FIG["alpha"]))])
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=170)
    fig.patch.set_facecolor(PAL["surface"])
    style(ax)
    ax.set_xscale("log")
    near = (np.abs(X[:, 1] - phi) <= math.radians(FIG["near_phi_deg"])) & (np.abs(X[:, 2] - math.log(FIG["alpha"]))
                                                                            <= FIG["near_lna"])
    ax.plot(np.exp(X[near, 0]), y[near], "o", ms=3, color=PAL["muted"], alpha=0.6, mew=0,
            label="ratios $y_n$ of rows near the line (%d)" % int(near.sum()))
    b_lo, b_hi = FIG["alpha"] * math.cos(phi) / cov.CFG["mu_range"][1], FIG["alpha"] * math.cos(phi) / cov.CFG["mu_range"][0]
    ax.axvspan(max(lo, b_lo), min(hi, b_hi), color=PAL["s1"], alpha=0.06, lw=0)
    for lam, col in zip(FIG["lams"], PAL["seq"]):
        f = RBFInterpolator(Xs, y, kernel="thin_plate_spline", smoothing=lam)
        L = cov.smoother_weights(Xs, lam, Xs)
        df = float(np.trace(L))
        lw = 2.4 if lam == T["lam"] else 1.4
        ax.plot(dg, f(sc(S)), color=col, lw=lw, label=r"$\hat b$, $\lambda_s$ = %g  (df %.1f)%s"
                % (lam, df, "  <- the CV choice" if lam == T["lam"] else ""))
    ax.plot(dg, cov.truth(S[:, 0], S[:, 1], S[:, 2]), color=PAL["ink"], lw=1.3, ls=(0, (4, 2)),
            label=r"truth $b = 1 + 0.06/d + 0.15\sin^2\varphi + 0.05\ln\alpha$")
    ax.set_ylim(0.75, 1.75)
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(ticker.FixedLocator((0.2, 0.3, 0.5, 1, 2, 3, 4)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel(r"true diameter $d$ (%sm, log scale)  at $\varphi$ = 10%s, $\hat\alpha$ = 1" % (MU, DEG),
                  color=PAL["ink2"])
    ax.set_ylabel(r"bias factor: ratio $y$, table $\hat b$, truth $b$ (dimensionless)", color=PAL["ink2"])
    ax.set_title(r"One line through the table: the smoothing $\lambda_s$ runs from chasing the noise to a plane",
                 color=PAL["ink"], fontsize=9, loc="left")
    ax.text(1.0, 1.70, "shaded: the phantoms' band on this line", color=PAL["s1"], fontsize=7.5, ha="center")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=7.2, frameon=False)
    for t in leg.get_texts():
        t.set_color(PAL["ink2"])
    fig.text(0.01, -0.21, r"plotted (computed level): batch-0 table of check_table_se_coverage.py, seed 20261018, $N_{\mathcal{S}}$ = 1500, "
             r"heteroscedastic $\tau(d)$, $\hat\alpha$ set to the true $\alpha$; df = trace of the hat matrix", color=PAL["muted"],
             fontsize=6.6)
    fig.savefig("../figures/fig04_table_lambda_slice.png", bbox_inches="tight", facecolor=PAL["surface"])
    plt.close(fig)


def fig5(T):
    X, Xs, sc = T["X"], T["Xs"], T["sc"]
    q = FIG["query"]
    qraw = np.array([[math.log(q[0]), math.radians(q[1]), math.log(q[2])]])
    qs = sc(qraw)
    rho = np.linalg.norm(Xs - qs, axis=1)
    n = X.shape[0]
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=170)
    fig.patch.set_facecolor(PAL["surface"])
    style(ax)
    w = cov.kernel_w(qraw, X)[0]
    ax.plot(rho, n * w / w.sum(), "o", ms=2.6, color=PAL["window"], alpha=0.75, mew=0,
            label=r"Gaussian window of $\hat\tau$, SE: $N\,\varpi_n/\sum\varpi$  ($N_{\rm eff}^{\rm loc}$ = %.0f)"
                  % (w.sum() ** 2 / (w ** 2).sum()))
    for lam, col in zip(FIG["kernel_lams"], (PAL["lam_lo"], PAL["lam_hi"])):
        l = cov.smoother_weights(Xs, lam, qs)[0]
        ax.plot(rho, n * l, "o", ms=2.6, color=col, alpha=0.7, mew=0,
                label=r"spline weights $N\,\ell_n(x)$, $\lambda_s$ = %g  ($1/\sum\ell_n^2$ = %.0f; negative sum %.2f)"
                      % (lam, 1.0 / np.sum(l ** 2), float(l[l < 0].sum())))
    ax.axhline(0.0, color=PAL["ink2"], lw=0.8)
    ax.axhline(1.0, color=PAL["muted"], lw=0.8, ls=(0, (3, 3)))
    ax.text(1.08, 1.12, "1 = plain average of all N", color=PAL["muted"], fontsize=7.2, ha="right")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(r"distance of row $n$ from the point, $\Vert s(x_n) - s(x)\Vert$ (normalised input units)",
                  color=PAL["ink2"])
    ax.set_ylabel("weight x N (symlog)", color=PAL["ink2"])
    ax.set_title(r"How the table averages at one point ($d$ = 1 %sm, $\varphi$ = 30%s, $\hat\alpha$ = 1.2): far-reaching weights, some negative" % (MU, DEG),
                 color=PAL["ink"], fontsize=9, loc="left")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=7.2, frameon=False)
    for t in leg.get_texts():
        t.set_color(PAL["ink2"])
    fig.text(0.01, -0.17, r"plotted (computed level): batch-0 design of check_table_se_coverage.py, $N_{\mathcal{S}}$ = 1500; weights from "
             r"the saddle-point system of the spline (they reproduce RBFInterpolator to $10^{-14}$)", color=PAL["muted"],
             fontsize=6.6)
    fig.savefig("../figures/fig05_table_weights.png", bbox_inches="tight", facecolor=PAL["surface"])
    plt.close(fig)


if __name__ == "__main__":
    cov.CFG["tau_mode"] = "hetero"
    T = cov.one_table(0)
    fig4(T)
    fig5(T)
    print("wrote ../figures/fig04_table_lambda_slice.png, ../figures/fig05_table_weights.png; CV lambda", T["lam"])
