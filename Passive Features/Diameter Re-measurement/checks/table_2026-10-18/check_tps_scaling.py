"""check_tps_scaling.py -- does the scaling of the inputs change a thin-plate
spline fit (scipy.interpolate.RBFInterpolator, kernel='thin_plate_spline')?

Standalone check for the explanation of Q4b (axis scaling of the bias table's
three inputs), 2026-10-16.  Not pipeline code.

Truth (chosen): b(d, phi, alpha) = 1 + 0.06/d + 0.15 sin^2(phi) + 0.05 ln(alpha),
noise SD tau = 0.03 on the ratio, N = 600 random design, seed 20261016.
Raw coordinates: x = (ln d / um, phi / rad, ln alpha).

C1  isotropic scaling by c with smoothing s equals no scaling with smoothing
    s / c^2 (predictions agree to 1e-9): the SciPy docstring's "epsilon has
    the same effect as scaling the smoothing parameter".
C2  scaling ONE axis (phi) by 10 at fixed smoothing changes the predictions
    (max |delta| reported): no smoothing value undoes an anisotropic rescale.
C3  raw coordinates vs range-normalised coordinates, each with its own
    5-fold CV smoothing: RMS error against the truth on an independent test
    set, reported for both.
C4  range-normalised, then per-axis scale factors for phi and ln alpha chosen
    by the same CV together with the smoothing (option B): the chosen scales,
    the CV error and the test RMS.

Run:  python3 check_tps_scaling.py
"""
import numpy as np
from scipy.interpolate import RBFInterpolator

CFG = dict(seed=20261016, n=600, n_test=2000, tau=0.03,
           ln_d=(np.log(0.2), np.log(4.0)), phi=(0.0, np.pi / 2),
           ln_alpha=(np.log(0.05), np.log(12.0)),
           smoothing_grid=10.0 ** np.arange(-3, 3.01, 0.5), folds=5,
           scale_grid=(0.25, 0.5, 1.0, 2.0, 4.0))


def truth(ln_d, phi, ln_alpha):
    return 1.0 + 0.06 / np.exp(ln_d) + 0.15 * np.sin(phi) ** 2 + 0.05 * ln_alpha


def design(rng, n):
    lo = np.array([CFG["ln_d"][0], CFG["phi"][0], CFG["ln_alpha"][0]])
    hi = np.array([CFG["ln_d"][1], CFG["phi"][1], CFG["ln_alpha"][1]])
    return lo + (hi - lo) * rng.random((n, 3))


def fit(X, y, smoothing):
    return RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=smoothing)


def cv_error(X, y, smoothing, folds, rng):
    idx = rng.permutation(len(y))
    err = 0.0
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        f = fit(X[train], y[train], smoothing)
        err += np.sum((f(X[test]) - y[test]) ** 2)
    return err / len(y)


def choose_smoothing(X, y, grid, folds, seed):
    scores = [cv_error(X, y, s, folds, np.random.default_rng(seed)) for s in grid]
    i = int(np.argmin(scores))
    return grid[i], scores[i]


def main():
    rng = np.random.default_rng(CFG["seed"])
    X = design(rng, CFG["n"])
    y = truth(*X.T) + CFG["tau"] * rng.standard_normal(CFG["n"])
    Xt = design(rng, CFG["n_test"])
    yt = truth(*Xt.T)
    lo = X.min(axis=0)
    span = X.max(axis=0) - lo
    results = []

    # C1: isotropic scaling c <-> smoothing / c^2
    c, s = 7.0, 10.0
    f0 = fit(X, y, s / c ** 2)
    f1 = fit(c * X, y, s)
    d1 = np.max(np.abs(f1(c * Xt) - f0(Xt)))
    results.append(("C1 isotropic c=7, s=10 vs s/c^2", d1 <= 1e-9, "max|delta| = %.2e" % d1))

    # C2: one axis scaled by 10 at fixed smoothing
    A = np.diag([1.0, 10.0, 1.0])
    f2 = fit(X @ A, y, s)
    d2 = np.max(np.abs(f2(Xt @ A) - fit(X, y, s)(Xt)))
    results.append(("C2 phi-axis x10, same smoothing", d2 > 1e-3, "max|delta| = %.3f" % d2))

    # C3: raw vs range-normalised, each with CV smoothing
    Xn = (X - lo) / span
    Xtn = (Xt - lo) / span
    s_raw, cv_raw = choose_smoothing(X, y, CFG["smoothing_grid"], CFG["folds"], CFG["seed"])
    s_nrm, cv_nrm = choose_smoothing(Xn, y, CFG["smoothing_grid"], CFG["folds"], CFG["seed"])
    rms_raw = np.sqrt(np.mean((fit(X, y, s_raw)(Xt) - yt) ** 2))
    rms_nrm = np.sqrt(np.mean((fit(Xn, y, s_nrm)(Xtn) - yt) ** 2))
    results.append(("C3 raw: CV smoothing, test RMS", True,
                    "s = %g, CV err = %.5f, RMS vs truth = %.4f" % (s_raw, cv_raw, rms_raw)))
    results.append(("C3 range-normalised: same", True,
                    "s = %g, CV err = %.5f, RMS vs truth = %.4f" % (s_nrm, cv_nrm, rms_nrm)))

    # C4: per-axis scales for phi and ln alpha chosen by CV with the smoothing
    best = None
    for a_phi in CFG["scale_grid"]:
        for a_al in CFG["scale_grid"]:
            S = np.diag([1.0, a_phi, a_al])
            s_b, cv_b = choose_smoothing(Xn @ S, y, CFG["smoothing_grid"], CFG["folds"], CFG["seed"])
            if best is None or cv_b < best[0]:
                best = (cv_b, a_phi, a_al, s_b)
    cv_b, a_phi, a_al, s_b = best
    S = np.diag([1.0, a_phi, a_al])
    rms_b = np.sqrt(np.mean((fit(Xn @ S, y, s_b)(Xtn @ S) - yt) ** 2))
    results.append(("C4 CV over per-axis scales (option B)", True,
                    "scales (phi, ln alpha) = (%g, %g), s = %g, CV err = %.5f, RMS vs truth = %.4f"
                    % (a_phi, a_al, s_b, cv_b, rms_b)))
    results.append(("   noise floor tau^2", True, "%.5f" % CFG["tau"] ** 2))

    print("%-42s %-6s %s" % ("check", "pass", "detail"))
    for name, ok, detail in results:
        print("%-42s %-6s %s" % (name, "PASS" if ok else "FAIL", detail))


if __name__ == "__main__":
    main()
