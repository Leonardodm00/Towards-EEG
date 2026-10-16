"""Figure 1 and checks for the answer of 2026-10-14 16:02: the two readings of the condition number.

Worked case: two standardised statistics with correlation c (unit variances), Sigma = [[1, c], [c, 1]].
Eigen-decomposition: u_max = (1, 1)/sqrt2 ('sum') with lambda_max = 1 + c, u_min = (1, -1)/sqrt2
('difference') with lambda_min = 1 - c; kappa = (1 + c)/(1 - c).
Perturbation: E = -delta * u_min u_min^T (the worst case: aimed at the short axis), so that
eps = ||E||_2 / ||Sigma||_2 = delta / lambda_max and lambda_min -> lambda_min - delta.

Checks, predictions fixed BEFORE the run (exact algebra, verified numerically):
  Q1 kappa = (1 + c)/(1 - c) from the eigenvalues (to 1e-12).
  Q2 the relative change of Sigma^-1 in the 2-norm equals the bound kappa*eps/(1 - kappa*eps) exactly
     (to 1e-9), i.e. the bound is attained by a perturbation along u_min.
  Q3 for the correlation matrix C of the ten LS medians (../../weights_1013/results/weights.npz), a
     perturbation of eps = 0.5/kappa(C) along its thinnest axis changes C^-1 by exactly 1 (doubles it).

Run:  python fig_condition_number.py   -> ../figures/fig01_condition_number.png, prints the checks
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = dict(c=0.99, delta=0.005, n_points=600, seed=2026101460)
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures" / "fig01_condition_number.png"
WEIGHTS = HERE.parent.parent / "weights_1013" / "results" / "weights.npz"


# ---------------- computation ----------------
def two_by_two(cfg):
    c, delta = cfg["c"], cfg["delta"]
    S = np.array([[1.0, c], [c, 1.0]])
    lam, U = np.linalg.eigh(S)                      # ascending: lam[0] = 1 - c (difference), lam[1] = 1 + c (sum)
    u_min, u_max = U[:, 0], U[:, 1]
    E = -delta * np.outer(u_min, u_min)
    Sp = S + E
    kappa = lam[1] / lam[0]
    eps = np.linalg.norm(E, 2) / np.linalg.norm(S, 2)
    rel = np.linalg.norm(np.linalg.inv(Sp) - np.linalg.inv(S), 2) / np.linalg.norm(np.linalg.inv(S), 2)
    return dict(S=S, Sp=Sp, lam=lam, lam_p=np.linalg.eigvalsh(Sp), u_min=u_min, u_max=u_max,
                kappa=kappa, eps=eps, rel=rel, bound=kappa * eps / (1 - kappa * eps))


def corr_case():
    C = np.load(WEIGHTS)["corr"]
    mu, V = np.linalg.eigh(C)
    kappa = mu[-1] / mu[0]
    eps = 0.5 / kappa
    E = -eps * np.linalg.norm(C, 2) * np.outer(V[:, 0], V[:, 0])
    rel = np.linalg.norm(np.linalg.inv(C + E) - np.linalg.inv(C), 2) / np.linalg.norm(np.linalg.inv(C), 2)
    return dict(kappa=kappa, eps=eps, rel=rel)


# ---------------- plotting ----------------
def ellipse(M, n=400):
    """Image of the unit circle under M^(1/2): half-axes along M's eigenvectors, lengths sqrt(eigenvalues)."""
    lam, U = np.linalg.eigh(M)
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)])
    return (U @ np.diag(np.sqrt(lam)) @ U.T) @ circle


def plot(r, cfg):
    rng = np.random.default_rng(cfg["seed"])
    z = rng.multivariate_normal([0, 0], r["S"], cfg["n_points"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 6.9))
    lam, lam_p = r["lam"], r["lam_p"]

    # (a) Sigma: the scatter of the statistics, its 1-SD ellipse, the long half-axis; inset: the short one
    a1.scatter(z[:, 0], z[:, 1], s=4, color="0.75", label="%d draws of $(z_1, z_2)$" % cfg["n_points"])
    a1.plot(*ellipse(r["S"]), color="C0", lw=2, label=r"$\Sigma$: 1-SD ellipse")
    a1.plot(*ellipse(r["Sp"]), color="C3", lw=1.2, ls="--", label=r"$\Sigma+E$")
    v = np.sqrt(lam[1]) * r["u_max"] * np.sign(r["u_max"][0])
    a1.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    a1.text(v[0] + 0.12, v[1] - 0.45, "$u_{\\max}$ (sum): variance\n" r"$\lambda_{\max}$ = %.2f, half-axis %.2f"
            % (lam[1], np.sqrt(lam[1])), fontsize=9)
    a1.set_title("(a) $\\Sigma$: how the two statistics scatter\n"
                 r"$\kappa=\lambda_{\max}/\lambda_{\min}$ = %.0f; $E$ is %.2f %% of $\|\Sigma\|_2$"
                 % (r["kappa"], 100 * r["eps"]), fontsize=10)
    a1.set_xlabel(r"standardised statistic $z_1$ (dimensionless)")
    a1.set_ylabel(r"standardised statistic $z_2$ (dimensionless)")
    a1.set_xlim(-3.2, 3.2); a1.set_ylim(-3.2, 3.2); a1.set_aspect("equal"); a1.grid(alpha=0.3)
    a1.legend(loc="upper left", fontsize=8)
    ins = a1.inset_axes([0.52, 0.04, 0.45, 0.42])
    ins.plot(*ellipse(r["S"]), color="C0", lw=2)
    ins.plot(*ellipse(r["Sp"]), color="C3", lw=1.2, ls="--")
    um = r["u_min"] * np.sign(r["u_min"][1])
    for l, col, sgn in ((lam[0], "C0", 1), (lam_p[0], "C3", -1)):      # opposite sides, so both show
        ins.annotate("", xy=sgn * np.sqrt(l) * um, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
    ins.set_xlim(-0.25, 0.25); ins.set_ylim(-0.25, 0.25); ins.set_aspect("equal")
    ins.tick_params(labelsize=7); ins.grid(alpha=0.3)
    ins.set_title("zoom: $u_{\\min}$ (difference)\n" r"$\lambda_{\min}$ %.3f $\to$ %.3f, half-axis %.2f $\to$ %.2f"
                  % (lam[0], lam_p[0], np.sqrt(lam[0]), np.sqrt(lam_p[0])), fontsize=8,
                  bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    # (b) Sigma^-1 drawn the same way: same axes, lengths inverted
    inv, invp = np.linalg.inv(r["S"]), np.linalg.inv(r["Sp"])
    a2.plot(*ellipse(inv), color="C0", lw=2, label=r"$\Sigma^{-1}$")
    a2.plot(*ellipse(invp), color="C3", lw=1.2, ls="--", label=r"$(\Sigma+E)^{-1}$")
    for M, col, zo in ((invp, "C3", 2), (inv, "C0", 3)):
        lm, Um = np.linalg.eigh(M)
        w = np.sqrt(lm[1]) * Um[:, 1] * np.sign(Um[0, 1])
        a2.annotate("", xy=w, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4), zorder=zo)
    a2.text(-8.5, -14.2, "along $u_{\\min}$ (difference): eigenvalue $1/\\lambda_{\\min}$ %.0f $\\to$ %.0f\n"
            "(half-axis %.1f $\\to$ %.1f): the weight on the difference doubles"
            % (1 / lam[0], 1 / lam_p[0], np.sqrt(1 / lam[0]), np.sqrt(1 / lam_p[0])), fontsize=8.5, color="C3")
    a2.text(-15.3, 12.6, "along $u_{\\max}$ (sum): eigenvalue $1/\\lambda_{\\max}$ = %.2f\n(half-axis %.2f, the thin direction here)"
            % (1 / lam[1], np.sqrt(1 / lam[1])), fontsize=8.5)
    a2.set_title("(b) $\\Sigma^{-1}$, the weighting, drawn the same way: same axes, lengths inverted\n"
                 r"relative change of $\Sigma^{-1}$ = %.2f = $\kappa\varepsilon/(1-\kappa\varepsilon)$ with $\kappa\varepsilon$ = %.2f"
                 % (r["rel"], r["kappa"] * r["eps"]), fontsize=10)
    a2.set_xlabel(r"$z_1$ (dimensionless)"); a2.set_ylabel(r"$z_2$ (dimensionless)")
    a2.set_xlim(-16, 16); a2.set_ylim(-16, 16); a2.set_aspect("equal"); a2.grid(alpha=0.3)
    a2.legend(loc="upper right", fontsize=8)
    fig.text(0.01, 0.035,
             "Plotted from the stated 2x2 matrix (unit variances, correlation c = %.2f) and %d simulated draws (seed %d); "
             "E = -%.3f u_min u_min^T, the worst case (aimed at the short axis)." % (cfg["c"], cfg["n_points"], cfg["seed"],
                                                                                  cfg["delta"]), fontsize=7.5)
    fig.text(0.01, 0.012,
             "Each ellipse is the image of the unit circle under the matrix's square root, so half-axis = sqrt(eigenvalue). "
             "In (a) it is also the set where the cost r^T Sigma^-1 r = 1; (b) is not a cost contour.", fontsize=7.5)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.14, wspace=0.18)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("saved", OUT)


def main():
    cfg = CONFIG
    r = two_by_two(cfg)
    ok = {}
    ok["Q1"] = bool(abs(r["kappa"] - (1 + cfg["c"]) / (1 - cfg["c"])) < 1e-12 * r["kappa"])
    ok["Q2"] = bool(abs(r["rel"] - r["bound"]) < 1e-9)
    print("2x2, c = %.2f: lambda = %s, kappa = %.4f ((1+c)/(1-c) = %.4f) -> Q1 %s"
          % (cfg["c"], np.array2string(r["lam"], precision=4), r["kappa"], (1 + cfg["c"]) / (1 - cfg["c"]), ok["Q1"]))
    print("  E = -%.3f u_min u_min^T: eps = %.5f, kappa*eps = %.4f, lambda_min %.4f -> %.4f, "
          "weight 1/lambda_min %.1f -> %.1f; rel. change of inverse %.6f vs bound %.6f -> Q2 %s"
          % (cfg["delta"], r["eps"], r["kappa"] * r["eps"], r["lam"][0], r["lam_p"][0], 1 / r["lam"][0],
             1 / r["lam_p"][0], r["rel"], r["bound"], ok["Q2"]))
    cc = corr_case()
    ok["Q3"] = bool(abs(cc["rel"] - 1.0) < 1e-6)
    print("correlation matrix of the ten LS medians: kappa %.4g; eps = 0.5/kappa = %.3g (%.3f %% of ||C||): "
          "rel. change of C^-1 %.6f -> Q3 %s" % (cc["kappa"], cc["eps"], 100 * cc["eps"], cc["rel"], ok["Q3"]))
    plot(r, cfg)
    print("ALL CHECKS:", ok)


if __name__ == "__main__":
    main()
