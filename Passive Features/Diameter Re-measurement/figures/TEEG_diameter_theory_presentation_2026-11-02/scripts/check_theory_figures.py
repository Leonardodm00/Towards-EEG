"""check_theory_figures.py -- checks of the numbers drawn in Figures 7-11 of the diameter theory pack (2026-11-02).

Standalone check code, not pipeline code. Predictions are fixed in PREDICTIONS before the run. Groups:
  G   (Fig. 7, T1)  chord ratios of handoff Eq. 8 against the handoff's printed numbers; width invariance by
                    Monte Carlo on random 3-D tubes (the chord along y_hat measured by bisection on Eq. 6).
  P   (Fig. 8, T2)  (B3.1) by Gauss-Legendre against an FFT convolution on a fine grid; the dip area against
                    (pi d/2)(I1 - L1); the squared widths d^2/16 and d^2/12 in the faint and opaque limits.
                    P1 and P2 failed on the first run (2026-11-02) and are kept as they came out; P1b and P2b
                    test the diagnosed causes (the reference grid's h^1.5 error; cancellation in I1 - L1).
                    K3 printed PASS on the first run only because 0/0 = NaN in the gennorm tail was skipped by
                    Python's max; with the zeros masked (second run) it FAILs on denormal values, kept; K3b
                    compares where the reference exceeds 1e-12 of its peak.
  K   (Fig. 9, T5)  every SGT member: unit integral, FWHM = w, the named members against scipy.stats, the skew's
                    half-width ratio zeta^2, tails, and agreement with the repository's own sgt_kernel_pdf.
  L   (Fig. 10)     every constant drawn, read back from its repository source (config.py, the decision log's
                    row l.139, SPEC 584, checks/d094_2026-10-28/dark_node_growth.out).
  I   (Fig. 11)     the toy inversion: root, shortcut, Eq. 21, and the scatter propagation by Monte Carlo.

Run:  python3 check_theory_figures.py --repo /path/to/clone-of-sci/diameter-pipeline   (about 10 s)
Writes check_theory_figures.out beside this script (also printed).
"""
import argparse
import math
import os
import re
import sys

import numpy as np
from scipy import integrate, signal, stats
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import theory_figs_model as tm  # noqa: E402

PREDICTIONS = dict(
    G1=(1.2649, 5e-4),        # handoff Eq. 8 example: 1.265 d at theta 45 deg, phi 30 deg
    G2=((1.015, 1.064, 1.155), 1e-3),   # flat branch misaligned by 10/20/30 deg: +1.5 / +6 / +15 %
    G3=1e-9,                  # MC width invariance: max |chord/d - 1| along y_hat
    P1=3e-5,                  # (B3.1) vs FFT convolution on a 1e-4 um grid, max abs diff (units of B)
    P2=1e-8,                  # dip area vs (pi d/2)(I1 - L1) in scipy's double precision, relative
    P1b=(2e-5, (6.0, 10.0)),  # after P1 failed: grid at h = 2.5e-5 um within 2e-5 of (B3.1); error ratio h -> h/4 ~ 4^1.5 = 8
    P2b=1e-8,                 # after P2 failed: the closed form at 40 digits (mpmath)
    P3=(1e-3, 3e-3),          # V/d^2 vs 1/16 (alpha 1e-4) and vs 1/12 (alpha 300), relative
    K1=1e-8,                  # |integral - 1|
    K2=1e-9,                  # |FWHM - w| (um)
    K3=1e-10,                 # max relative difference to scipy.stats over |z| <= 3 w
    K3b=1e-10,                # after K3 failed: the same where the reference exceeds 1e-12 of its peak
    K4=1e-12,                 # |right/left half-width ratio - zeta^2|
    K5=0.02,                  # tail exponents within 0.02 of -(1 + beta/xi): t4 -5, Lorentz -2
    K6=1e-12,                 # max |g_here - g_repo| / g_repo(0)
    L=0.0,                    # every constant equals its source exactly
    I1=1e-12,                 # root of m(d) = d_hat(d*) equals d* = 1
    I2=(-0.0411, 5e-4),       # exact shortcut error of the power-law toy: 1.15^-0.3 - 1
    I3=((-0.0431, -0.045), 5e-4),   # Eq. 21 full and first order (mathematics s.3.6: -4.3 %, -4.5 %)
    I4=0.03,                  # MC SD of the inverted d against tau d*/m'(d*), relative
)

LINES = []


def out(s=""):
    print(s)
    LINES.append(s)


def verdict(name, ok, text):
    out("%-3s %s  %s" % (name, "PASS" if ok else "FAIL", text))
    return ok


def group_G():
    r = []
    v = tm.chord_ratio(math.radians(45.0), math.radians(30.0))
    r.append(verdict("G1", abs(v - PREDICTIONS["G1"][0]) < PREDICTIONS["G1"][1],
                     "chord/d at eps 45 deg (alpha = theta = 45 deg), phi 30 deg = %.5f (handoff: 1.265)" % v))
    vals = [float(tm.chord_ratio(math.radians(e), 0.0)) for e in (10, 20, 30)]
    ok = all(abs(a - b) < PREDICTIONS["G2"][1] for a, b in zip(vals, PREDICTIONS["G2"][0]))
    r.append(verdict("G2", ok, "flat branch, eps 10/20/30 deg: %s (handoff: +1.5 / +6 / +15 %%)" %
                     " / ".join("%.4f" % x for x in vals)))
    # Monte Carlo: random tubes in 3-D; along y_hat (in-plane normal to the projection) the chord through c is d
    rng = np.random.default_rng(20261102)
    worst = 0.0
    for _ in range(2000):
        theta, phi = rng.uniform(-np.pi, np.pi), rng.uniform(0, np.pi / 2 - 1e-3)
        r0 = rng.uniform(0.1, 2.0)
        t = np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)])
        y = np.array([-np.sin(theta), np.cos(theta), 0.0])
        inside = lambda s: np.dot(s * y, s * y) - np.dot(s * y, t) ** 2 - r0 ** 2   # handoff Eq. 6, p - c = s y
        s_edge = brentq(inside, 0.0, 10 * r0, xtol=1e-15)
        worst = max(worst, abs(2 * s_edge / (2 * r0) - 1))
    r.append(verdict("G3", worst < PREDICTIONS["G3"], "2000 random tubes: max |chord along y_hat / d - 1| = %.1e" % worst))
    return r


def group_P():
    r = []
    sig = tm.SIGMA_FIT_UM
    kern = lambda z: tm.gauss_kernel(z, sig)
    worst = 0.0
    for d in (0.3, 0.5, 1.0, 2.0):
        for alpha in (0.3, 3.0, 30.0):
            h = 1e-4
            u = np.arange(-6.0, 6.0 + h / 2, h)
            a = 1 - tm.transmission(u, d, alpha)
            g = tm.gauss_kernel(u, sig) * h
            conv = signal.fftconvolve(a, g, mode="same")
            v = np.linspace(-2, 2, 81)
            direct = 1 - np.interp(v, u, conv)
            gl = tm.tube_profile(v, d, alpha, kern)
            worst = max(worst, np.max(np.abs(gl - direct)))
    r.append(verdict("P1", worst < PREDICTIONS["P1"], "(B3.1) by Gauss-Legendre vs FFT convolution "
                     "(h = 1e-4 um), d 0.3-2 um, alpha 0.3/3/30: max |diff| = %.1e B" % worst))
    # P1b: P1's reference, not (B3.1), carried the error: the grid sum converges as h^1.5 at the tube's
    # square-root edges, while Gauss-Legendre in psi is converged to 1e-13 (n_psi 200 -> 4000).
    errs = []
    for h in (1e-4, 2.5e-5):
        worst_h = 0.0
        for d in (0.3, 2.0):
            u = np.arange(-4.0, 4.0 + h / 2, h)
            conv = signal.fftconvolve(1 - tm.transmission(u, d, 30.0), tm.gauss_kernel(u, sig) * h, mode="same")
            v = np.linspace(-2, 2, 81)
            worst_h = max(worst_h, np.max(np.abs((1 - np.interp(v, u, conv)) - tm.tube_profile(v, d, 30.0, kern))))
        errs.append(worst_h)
    ratio = errs[0] / errs[1]
    ok = errs[1] < PREDICTIONS["P1b"][0] and PREDICTIONS["P1b"][1][0] < ratio < PREDICTIONS["P1b"][1][1]
    r.append(verdict("P1b", ok, "alpha 30, d 0.3/2 um: grid error %.1e (h 1e-4) -> %.1e (h 2.5e-5), ratio %.1f "
                     "(h^1.5: 8); (B3.1) is the converged side" % (errs[0], errs[1], ratio)))
    worst = 0.0
    for d in (0.3, 1.0, 2.0):
        for alpha in (0.01, 0.3, 3.0, 30.0):
            num, _ = integrate.quad(lambda u: 1 - tm.transmission(u, d, alpha), -d / 2, d / 2, epsabs=1e-14,
                                    epsrel=1e-13, limit=400)
            worst = max(worst, abs(num / tm.dip_area_exact(d, alpha, precise=False) - 1))
    r.append(verdict("P2", worst < PREDICTIONS["P2"], "dip area vs (pi d/2)(I1 - L1), scipy double precision: "
                     "max rel diff %.1e" % worst))
    worst_b = 0.0
    for d in (0.3, 1.0, 2.0):
        for alpha in (0.01, 0.3, 3.0, 30.0):
            num, _ = integrate.quad(lambda u: 1 - tm.transmission(u, d, alpha), -d / 2, d / 2, epsabs=1e-14,
                                    epsrel=1e-13, limit=400)
            worst_b = max(worst_b, abs(num / tm.dip_area_exact(d, alpha, precise=True) - 1))
    r.append(verdict("P2b", worst_b < PREDICTIONS["P2b"], "P2 failed by cancellation in I1 - L1 (both ~ e^alpha; "
                     "7e-5 at alpha 30); at 40 digits (mpmath): max rel diff %.1e" % worst_b))

    def v_norm(d, alpha):
        u = np.linspace(-d / 2, d / 2, 400001)
        a = 1 - tm.transmission(u, d, alpha)
        m0 = np.trapezoid(a, u)
        return np.trapezoid(a * u ** 2, u) / m0
    f, o = v_norm(1.0, 1e-4), v_norm(1.0, 300.0)
    ok = abs(f * 16 - 1) < PREDICTIONS["P3"][0] and abs(o * 12 - 1) < PREDICTIONS["P3"][1]
    r.append(verdict("P3", ok, "V/d^2: faint %.6f (1/16 = 0.0625), opaque %.6f (1/12 = 0.083333)" % (f, o)))
    return r


def _fwhm_numeric(g, w):
    peak = g(np.array([0.0]))[0]
    right = brentq(lambda z: g(np.array([z]))[0] - peak / 2, 1e-9, 20 * w, xtol=1e-15)
    left = brentq(lambda z: g(np.array([z]))[0] - peak / 2, -20 * w, -1e-9, xtol=1e-15)
    return right - left, right, -left


def group_K(repo_src):
    r = []
    w = 0.5
    worst_int = worst_fwhm = 0.0
    for name, (b, x, z) in tm.SGT_MEMBERS:
        g = lambda t, b=b, x=x, z=z: tm.sgt_kernel(t, w, b, x, z)
        tot = integrate.quad(lambda t: g(np.array([t]))[0], -np.inf, 0, epsabs=1e-13, limit=500)[0] + \
            integrate.quad(lambda t: g(np.array([t]))[0], 0, np.inf, epsabs=1e-13, limit=500)[0]
        worst_int = max(worst_int, abs(tot - 1))
        fw, _, _ = _fwhm_numeric(g, w)
        worst_fwhm = max(worst_fwhm, abs(fw - w))
    r.append(verdict("K1", worst_int < PREDICTIONS["K1"], "six members at w = 0.5 um: max |integral - 1| = %.1e" % worst_int))
    r.append(verdict("K2", worst_fwhm < PREDICTIONS["K2"], "max |FWHM - w| = %.1e um" % worst_fwhm))
    z = np.linspace(-3 * w, 3 * w, 2001)
    ref = {
        "Gaussian": stats.norm(scale=w / (2 * math.sqrt(2 * math.log(2)))).pdf(z),
        "Laplace": stats.laplace(scale=w / (2 * math.log(2))).pdf(z),
        "Student $t_4$": stats.t(4, scale=w / (4 * math.sqrt(2 ** 0.4 - 1))).pdf(z),   # half max at 2 sqrt(2^0.4 - 1) s
        "Lorentzian ($t_1$)": stats.cauchy(scale=w / 2).pdf(z),
        "gen. normal, $\\beta$ = 4": stats.gennorm(4, scale=w / (2 * math.log(2) ** 0.25)).pdf(z),
    }
    worst = 0.0
    for name, (b, x, zz) in tm.SGT_MEMBERS:
        if name in ref:
            mine = tm.sgt_kernel(z, w, b, x, zz)
            ok_z = ref[name] > 0
            worst = max(worst, np.max(np.abs(mine[ok_z] / ref[name][ok_z] - 1)))
    r.append(verdict("K3", worst < PREDICTIONS["K3"], "Gaussian, Laplace, t4, Cauchy, gennorm(4) vs scipy.stats "
                     "(scales from FWHM = w): max rel diff %.1e" % worst))
    worst_b = 0.0
    for name, (b, x, zz) in tm.SGT_MEMBERS:
        if name in ref:
            mine = tm.sgt_kernel(z, w, b, x, zz)
            keep = ref[name] > 1e-12 * ref[name].max()
            worst_b = max(worst_b, np.max(np.abs(mine[keep] / ref[name][keep] - 1)))
    r.append(verdict("K3b", worst_b < PREDICTIONS["K3b"], "K3 failed on denormal tail values (gennorm(4) at |z| near "
                     "3w is ~1e-308); where the reference exceeds 1e-12 of its peak: max rel diff %.1e" % worst_b))
    g = lambda t: tm.sgt_kernel(t, w, 2.0, 0.0, 1.5)
    _, rh, lh = _fwhm_numeric(g, w)
    r.append(verdict("K4", abs(rh / lh - 1.5 ** 2) < PREDICTIONS["K4"],
                     "skewed Gaussian zeta 1.5: right/left half-width = %.12f (zeta^2 = 2.25)" % (rh / lh)))
    exps = []
    for (b, x) in ((2.0, 0.5), (2.0, 2.0)):
        t1, t2 = 200.0, 400.0
        e = math.log(tm.sgt_kernel(np.array([t2]), w, b, x, 1.0)[0] / tm.sgt_kernel(np.array([t1]), w, b, x, 1.0)[0]) / math.log(2)
        exps.append(e)
    ok = abs(exps[0] + 5) < PREDICTIONS["K5"] and abs(exps[1] + 2) < PREDICTIONS["K5"]
    r.append(verdict("K5", ok, "tail exponents at |z| 200-400 um: t4 %.4f (-5), Lorentz %.4f (-2)" % tuple(exps)))
    try:
        sys.path.insert(0, repo_src)
        from allen_diameter.model import blur_kernels as bk
        worst = 0.0
        for name, (b, x, zz) in tm.SGT_MEMBERS:
            grepo = bk.sgt_kernel_pdf(w, b, x, zz)
            zz_ = np.linspace(-5, 5, 4001)
            worst = max(worst, np.max(np.abs(tm.sgt_kernel(zz_, w, b, x, zz) - grepo(zz_))) / grepo(np.array([0.0]))[0])
        r.append(verdict("K6", worst < PREDICTIONS["K6"], "against the repository's model/blur_kernels.sgt_kernel_pdf, "
                         "six members, |z| <= 5 um: max |diff| / peak = %.1e" % worst))
    except Exception as exc:  # the repository is optional for the other checks
        r.append(verdict("K6", False, "repository kernel not importable from %s: %r" % (repo_src, exc)))
    return r


def group_L(repo):
    r = []
    dr = os.path.join(repo, "Passive Features", "Diameter Re-measurement")
    cfg = open(os.path.join(dr, "src", "allen_diameter", "config.py")).read()
    knots = tuple(float(x) for x in re.search(r"kernel_table_delta_um: Tuple\[float, \.\.\.\] = \(([^)]*)\)", cfg).group(1).split(","))
    sig = tuple(float(x) for x in re.search(r"kernel_table_sigma_um: Tuple\[float, \.\.\.\] = \(([^)]*)\)", cfg).group(1).split(","))
    gam = float(re.search(r"kernel_continuation_slope: float = ([0-9.]+)", cfg).group(1))
    ok = knots == tm.CONFIGURED_KNOTS_UM and sig == tm.CONFIGURED_SIGMA_UM and gam == tm.CONFIGURED_GAMMA
    r.append(verdict("L1", ok, "configured table and slope = config.py: %s / %s / %s" % (knots, sig, gam)))
    log = open(os.path.join(repo, "docs", "project", "TEEG_decisions_and_ideas_log.md"), encoding="utf-8").read()
    row = re.search(r"Near fit \(30 nodes\): \$\\sigma_\{\\rm r\}\$ = ([0-9., ]+) um at \$\\delta\$ = ([0-9., ]+) um", log)
    near = tuple(float(x) for x in row.group(1).split(","))
    dl = tuple(float(x) for x in row.group(2).split(","))
    gam_row = re.search(r"\$\\hat\\gamma\$ = ([0-9.]+), \$t_1\$ interval \[(-?[0-9.]+), ([0-9.]+)\]", log)
    ok = near == tm.PILOT_SIGMA_UM and dl == tm.PILOT_KNOTS_UM and float(gam_row.group(1)) == tm.PILOT_GAMMA_HAT and \
        (float(gam_row.group(2)), float(gam_row.group(3))) == tm.PILOT_GAMMA_T1
    r.append(verdict("L2", ok, "pilot near table %s at %s, gamma_hat %s [%s, %s] = the log's 2026-10-30 row" %
                     (near, dl, gam_row.group(1), gam_row.group(2), gam_row.group(3))))
    spec = open(os.path.join(repo, "specs", "SPEC.md"), encoding="utf-8").read()
    m = re.search(r"the SE is ([0-9.]+) um with 1000 of 1000 resamples kept", spec)
    r.append(verdict("L3", m is not None and float(m.group(1)) == tm.PILOT_ANCHOR_SE_UM,
                     "anchor SE after D-115.4 = SPEC 10.A: %s um" % (m.group(1) if m else None)))
    dout = open(os.path.join(dr, "checks", "d094_2026-10-28", "dark_node_growth.out")).read()
    ok = True
    for d, vals in tm.PHANTOM_FIRST_STAGE.items():
        blk = dout.split("d = %.2f um" % d)[1]
        rowa = re.search(r"^\s+0\.10\s+[0-9.]+\s+-?[0-9.]+\s+-?[0-9.]+\s+-?[0-9.]+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", blk, re.M)
        got = tuple(float(x) for x in rowa.groups())
        ok &= got == vals
    r.append(verdict("L4", ok, "phantom first stage (alpha 0.1) at d 0.3 / 0.5 um = dark_node_growth.out"))
    far = re.search(r"\$P_\{\\rm far\}\$ = `far_offsets_planes` = (\d+)", spec)
    kmin = re.search(r"`far_min_k_offset` = (\d+)", spec)
    ok = int(far.group(1)) == tm.P_FAR and int(kmin.group(1)) == tm.FAR_MIN_K_OFFSET
    r.append(verdict("L5", ok, "far scan +-%s planes, far fit |k - k*| >= %s = SPEC 15.1 / 15.3" % (far.group(1), kmin.group(1))))
    return r


def group_I():
    r = []
    d_hat = float(tm.m_toy(1.0))
    root = tm.invert(d_hat)
    r.append(verdict("I1", abs(root - 1.0) < PREDICTIONS["I1"], "d_hat = m(1 um) = %.4f um -> root %.15f um" % (d_hat, root)))
    sc = tm.shortcut(d_hat) / root - 1
    r.append(verdict("I2", abs(sc - PREDICTIONS["I2"][0]) < PREDICTIONS["I2"][1], "shortcut error (exact, toy) %.5f" % sc))
    full, first = tm.eq21()
    ok = abs(full - PREDICTIONS["I3"][0][0]) < PREDICTIONS["I3"][1] and abs(first - PREDICTIONS["I3"][0][1]) < PREDICTIONS["I3"][1]
    r.append(verdict("I3", ok, "Eq. 21: %.5f full, %.5f first order (mathematics s.3.6: -4.3 %%, -4.5 %%)" % (full, first)))
    rng = np.random.default_rng(7)
    tau = tm.TOY["tau"]
    draws = d_hat + tau * 1.0 * rng.standard_normal(200000)        # SD of D_hat = tau d* (Eq. 19: tau = SD of D_hat/d)
    inv = np.array([tm.invert(x) for x in draws[:20000]])
    pred = tau * 1.0 / (tm.TOY["b_star"] * (1 + tm.TOY["beta"]))   # tau d / m'(d), m' = b (1 + beta)
    rel = np.std(inv) / pred - 1
    r.append(verdict("I4", abs(rel) < PREDICTIONS["I4"], "MC SD of the inverted d %.5f vs tau d*/[b(1+beta)] = %.5f "
                     "(rel %+.3f); mean %.5f (Jensen gap %+.5f um)" % (np.std(inv), pred, rel, inv.mean(), inv.mean() - 1)))
    return r


def main():
    ap = argparse.ArgumentParser()
    # default: the repository this script sits in (figures/<pack>/scripts/ -> 5 levels up), else --repo is needed
    here_repo = os.path.abspath(os.path.join(HERE, *([os.pardir] * 5)))
    default = here_repo if os.path.isfile(os.path.join(here_repo, "specs", "SPEC.md")) else None
    ap.add_argument("--repo", default=default, required=default is None,
                    help="clone of Towards-EEG at sci/diameter-pipeline (default: the repository holding this script)")
    a = ap.parse_args()
    src = os.path.join(a.repo, "Passive Features", "Diameter Re-measurement", "src")
    out("check_theory_figures.py -- repo %s" % a.repo)
    res = group_G() + group_P() + group_K(src) + group_L(a.repo) + group_I()
    out("%d of %d PASS" % (sum(res), len(res)))
    open(os.path.join(HERE, "check_theory_figures.out"), "w").write("\n".join(LINES) + "\n")
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(main())
