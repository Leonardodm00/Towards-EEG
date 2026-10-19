"""check_b_target.py -- what the table's target b(x | C, S) is, on a toy of the D-059/D-060 design whose
conditional means are known in closed form (2026-10-18; standalone, not pipeline code).

Toy at fixed (d, phi); the target conditions on both exactly, so only a draw's theta, mu and its
realizations' nuisances vary:
  draw g:        theta_g ~ U[0, pi), ln mu_g ~ U[ln 0.3, ln 3] /um (the D-024 laws), so at fixed (d, phi)
                 ln alpha_g = ln(mu_g d / cos phi) ~ U[L0, L1];
  realization r: (eta1, eta2) ~ N(0, [[1, rho], [rho, 1]]), rho = -0.9 (corr of the d_hat and alpha_hat errors
                 on thin tubes, D-052 Monte Carlo, data inspected earlier); ln alpha_hat = ln alpha_g + s2 eta2,
                 s2 = 0.4 (alpha_hat's relative SD near 0.4 at 0.6 um);
                 y = b0(ln alpha_g) + delta(theta_g) + s1 eta1, with b0 = 1 + 0.06/d + 0.15 sin^2 phi
                 + 0.05 ln alpha (the fixture of check_table_se_coverage.py), delta = 0.03 cos(2 theta) (a heading
                 effect), s1 = 0.18; counted (in S) with probability 0.8, independently of everything else.
Closed form: ln alpha_g given ln alpha_hat = l is N(l, s2^2) truncated to [L0, L1], hence
  b(l)       = E[y | l]          = b0(E[ln alpha | l]) + s1 rho E[eta2 | l],   E[eta2 | l] = (l - E[ln alpha | l]) / s2,
  b_draws(l) = E[E[y | c_g] | l] = b0(E[ln alpha | l])          (the rows' own draw means, averaged at l).
T1  The mean ratio of the counted rows within h of l, (a) from all k realizations, (b) from first realizations
    only (one row per draw, independent rows), each minus the closed form b averaged over the same rows:
    |difference| <= 3 SE (Poisson bootstrap over draws, so the siblings' dependence is in the SE).
T2  The mean over the same rows of their draws' own expected ratio E[y | c_g] = b0(ln alpha_g) + delta(theta_g),
    minus b_draws over the same rows: |difference| <= 3 SE; the gap b - b_draws is printed (closed form).
Tolerances fixed before running. Run (same folder):  python3 check_b_target.py      (a few seconds)
"""
import math

import numpy as np
from scipy.stats import norm, truncnorm

CFG = dict(seed=20261021, G=200_000, k=5, d_um=0.6, phi_deg=30.0, mu_range_per_um=(0.3, 3.0), rho=-0.9, s2=0.4,
           s1=0.18, heading_amp=0.03, p_count=0.8, half_width=0.025, n_boot=400, n_se=3.0)


# ----------------------------------------------------------------- model --
def design_bounds(cfg):
    """[L0, L1]: the range of ln alpha_g at the fixed (d, phi) under the log-uniform mu law."""
    d, phi = cfg["d_um"], math.radians(cfg["phi_deg"])
    lo, hi = cfg["mu_range_per_um"]
    return math.log(lo * d / math.cos(phi)), math.log(hi * d / math.cos(phi))


def b0(cfg, ln_alpha):
    """The fixture's mean ratio at the TRUE optical depth (affine in ln alpha)."""
    phi = math.radians(cfg["phi_deg"])
    return 1.0 + 0.06 / cfg["d_um"] + 0.15 * math.sin(phi) ** 2 + 0.05 * np.asarray(ln_alpha)


def post_mean_ln_alpha(cfg, l):
    """E[ln alpha_g | ln alpha_hat = l]: mean of N(l, s2^2) truncated to [L0, L1], by the standard formula
    l + s (pdf(a) - pdf(b)) / (cdf(b) - cdf(a)), a = (L0 - l)/s, b = (L1 - l)/s (cross-checked against
    scipy.stats.truncnorm.mean in check_formula)."""
    L0, L1 = design_bounds(cfg)
    s = cfg["s2"]
    l = np.asarray(l, dtype=float)
    a, b = (L0 - l) / s, (L1 - l) / s
    return l + s * (norm.pdf(a) - norm.pdf(b)) / (norm.cdf(b) - norm.cdf(a))


def check_formula(cfg):
    L0, L1 = design_bounds(cfg)
    ls = np.array([L0 - 0.2, 0.5 * (L0 + L1), L1 - 0.1, L1, L1 + 0.3])
    s = cfg["s2"]
    ref = np.array([truncnorm.mean((L0 - v) / s, (L1 - v) / s, loc=v, scale=s) for v in ls])
    err = float(np.max(np.abs(post_mean_ln_alpha(cfg, ls) - ref)))
    print("formula check: truncated-normal mean vs scipy truncnorm.mean, max |diff| = %.1e (%s)"
          % (err, "PASS" if err <= 1e-10 else "FAIL"))
    return err <= 1e-10


def b_closed(cfg, l):
    m = post_mean_ln_alpha(cfg, l)
    return b0(cfg, m) + cfg["s1"] * cfg["rho"] * (np.asarray(l) - m) / cfg["s2"]


def b_draws_closed(cfg, l):
    return b0(cfg, post_mean_ln_alpha(cfg, l))


# -------------------------------------------------------------- simulate --
def simulate(cfg):
    rng = np.random.default_rng(cfg["seed"])
    G, k = cfg["G"], cfg["k"]
    L0, L1 = design_bounds(cfg)
    theta = rng.uniform(0.0, math.pi, G)
    ln_alpha = rng.uniform(L0, L1, G)
    e2 = rng.standard_normal((G, k))
    e1 = cfg["rho"] * e2 + math.sqrt(1.0 - cfg["rho"] ** 2) * rng.standard_normal((G, k))
    l_hat = ln_alpha[:, None] + cfg["s2"] * e2
    draw_mean = b0(cfg, ln_alpha) + cfg["heading_amp"] * np.cos(2.0 * theta)       # E[y | c_g]
    y = draw_mean[:, None] + cfg["s1"] * e1
    counted = rng.random((G, k)) < cfg["p_count"]
    return dict(l_hat=l_hat, y=y, draw_mean=np.broadcast_to(draw_mean[:, None], (G, k)), counted=counted)


# -------------------------------------------------------------- analysis --
def cluster_mean_diff(cfg, values, refs, mask, rng):
    """Mean of (values - refs) over the masked rows and its Poisson-bootstrap SE over draws."""
    num = np.where(mask, values - refs, 0.0).sum(axis=1)
    cnt = mask.sum(axis=1)
    keep = cnt > 0
    num, cnt = num[keep], cnt[keep]
    est = num.sum() / cnt.sum()
    w = rng.poisson(1.0, size=(cfg["n_boot"], num.size)).astype(np.float64)
    boot = (w @ num) / (w @ cnt)
    return float(est), float(boot.std(ddof=1)), int(cnt.sum())


def run(cfg):
    sim = simulate(cfg)
    L0, L1 = design_bounds(cfg)
    rng = np.random.default_rng([cfg["seed"], 1])
    queries = (("interior", 0.5 * (L0 + L1)), ("top edge", L1))
    h = cfg["half_width"]
    print("toy: d %.2f um, phi %.0f deg, ln alpha design range [%.3f, %.3f]; G %d draws x k %d; bin |ln a_hat - l| < %.3f"
          % (cfg["d_um"], cfg["phi_deg"], L0, L1, cfg["G"], cfg["k"], h))
    ok_all = check_formula(cfg)
    for name, l in queries:
        inbin = sim["counted"] & (np.abs(sim["l_hat"] - l) < h)
        ref_b = np.zeros(inbin.shape)
        ref_bd = np.zeros(inbin.shape)
        ref_b[inbin] = b_closed(cfg, sim["l_hat"][inbin])          # closed forms only where used
        ref_bd[inbin] = b_draws_closed(cfg, sim["l_hat"][inbin])
        first = np.zeros_like(inbin)
        first[:, 0] = inbin[:, 0]
        a = cluster_mean_diff(cfg, sim["y"], ref_b, inbin, rng)
        b = cluster_mean_diff(cfg, sim["y"], ref_b, first, rng)
        c = cluster_mean_diff(cfg, sim["draw_mean"], ref_bd, inbin, rng)
        gap = float(b_closed(cfg, l) - b_draws_closed(cfg, l))
        okq = [abs(t[0]) <= cfg["n_se"] * t[1] for t in (a, b, c)]
        ok_all &= all(okq)
        print("%-9s l = %.3f: closed form b = %.4f, b_draws = %.4f (gap b - b_draws %+.4f)" %
              (name, l, float(b_closed(cfg, l)), float(b_draws_closed(cfg, l)), gap))
        print("   T1 all realizations   : mean y - b over %6d rows = %+.4f (SE %.4f) %s" % (a[2], a[0], a[1], "PASS" if okq[0] else "FAIL"))
        print("   T1 first realizations : mean y - b over %6d rows = %+.4f (SE %.4f) %s" % (b[2], b[0], b[1], "PASS" if okq[1] else "FAIL"))
        print("   T2 rows' draw means   : mean E[y|c_g] - b_draws        = %+.4f (SE %.4f) %s" % (c[0], c[1], "PASS" if okq[2] else "FAIL"))
    print("overall:", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    run(CFG)
