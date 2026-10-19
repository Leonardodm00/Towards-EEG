"""check05_smoothness.py -- how smooth a fitted 3-D spline is at a data point: r^2 ln r against -r
(2026-10-18; standalone). Document section 3.9, Figure 3.

Data: N points uniform in [0, 1]^3, y = sin(3 u1) + u2^2 - 0.5 u3 + 0.05 eps (seed below). Fits: SciPy
'thin_plate_spline' (degree 1) and 'linear' with degree 1 (the 3-D bending-energy spline), at smoothing 0
and at the smoothing each kernel's 5-fold CV picks. Along a line u_* + t e through the data point u_* nearest
the centre (e a fixed random unit vector): the slope jump  D(h) = [f(u_* + h e) - f(u_*)]/h - [f(u_*) - f(u_* - h e)]/h.
Prediction (check01 (d)): for r^2 ln r, D(h) -> 0 like h ln h (C^1); for -r, D(h) -> -2 c_* (a kink of the
cone -c_* |x - u_*|, c_* the point's coefficient), nonzero whenever c_* != 0 -- at smoothing > 0,
c_* = (residual at u_*) / lambda.  Pass rule: |D(1e-6)| < 1e-4 for r^2 ln r; |D(1e-6) + 2 c_*| < 1e-4 for -r.
Writes check05_slices.npz for Figure 3.
Run:  python3 check05_smoothness.py      (about 20 s)
"""
import numpy as np
from scipy.interpolate import RBFInterpolator

from tps_common import fit_coeffs

CFG = dict(seed=20261025, n=300, noise=0.05, folds=5, grid=tuple(10.0 ** k for k in range(-6, 3)),
           hs=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6), t_half=0.12, n_t=801)


def data(rng):
    X = rng.uniform(0, 1, (CFG["n"], 3))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 - 0.5 * X[:, 2] + CFG["noise"] * rng.standard_normal(CFG["n"])
    return X, y


def cv_lambda(X, y, kernel, rng):
    fold = rng.permutation(CFG["n"]) % CFG["folds"]
    best = None
    for lam in CFG["grid"]:
        err = 0.0
        for f in range(CFG["folds"]):
            tr = fold != f
            err += float(np.sum((RBFInterpolator(X[tr], y[tr], kernel=kernel, degree=1, smoothing=lam)(X[~tr])
                                 - y[~tr]) ** 2))
        if best is None or err < best[1]:
            best = (lam, err)
    return best[0]


def main():
    rng = np.random.default_rng(CFG["seed"])
    X, y = data(rng)
    i_star = int(np.argmin(np.linalg.norm(X - 0.5, axis=1)))
    u = X[i_star]
    e = rng.standard_normal(3)
    e /= np.linalg.norm(e)
    ok = True
    out = dict(t=np.linspace(-CFG["t_half"], CFG["t_half"], CFG["n_t"]), u=u, e=e)
    for kernel in ("thin_plate_spline", "linear"):
        lam_cv = cv_lambda(X, y, kernel, np.random.default_rng([CFG["seed"], 1]))
        for lam in (0.0, lam_cv):
            f = RBFInterpolator(X, y, kernel=kernel, degree=1, smoothing=lam)
            c, q = fit_coeffs(X, y, lam, kernel=kernel, degree=1)
            f0 = float(f(u[None, :])[0])
            jumps = []
            for h in CFG["hs"]:
                fp = float(f((u + h * e)[None, :])[0])
                fm = float(f((u - h * e)[None, :])[0])
                jumps.append((fp - f0) / h - (f0 - fm) / h)
            c_star = float(c[i_star])
            if kernel == "thin_plate_spline":
                good = abs(jumps[-1]) < 1e-4
            else:
                good = abs(jumps[-1] + 2 * c_star) < 1e-4
            ok &= good
            print("%-18s smoothing %-8g (CV choice %g): c_* = %+.4f; slope jump D(h) for h = %s: %s  -> %s"
                  % (kernel, lam, lam_cv, c_star, CFG["hs"], " ".join("%+.2e" % v for v in jumps),
                     "PASS" if good else "FAIL"))
            key = "%s_%s" % ("tps" if kernel == "thin_plate_spline" else "lin", "interp" if lam == 0 else "cv")
            out[key] = f(u[None, :] + out["t"][:, None] * e[None, :])
            out[key + "_lam"] = lam
            out[key + "_cstar"] = c_star
    np.savez("check05_slices.npz", **out)
    print("overall:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
