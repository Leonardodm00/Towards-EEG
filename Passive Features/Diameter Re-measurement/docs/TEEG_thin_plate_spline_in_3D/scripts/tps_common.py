"""tps_common.py -- shared pieces of the checks and figures of the document
"The thin-plate spline in three dimensions" (2026-10-18; standalone, not pipeline code).

Kernels exactly as SciPy 1.18.1 defines them (scipy/interpolate/_rbfinterp_xp.py):
  'thin_plate_spline'  phi(r) = r^2 ln r   (0 at r = 0)
  'linear'             phi(r) = -r
  'cubic'              phi(r) = r^3
The saddle-point system is SciPy's: [[K + diag(lam), P], [P^T, 0]] [c; q] = [y; 0],
P = monomials up to `degree` (here 0 or 1). The fixture (design, truth, tau(d), range
normalisation) is the one of check_table_se_coverage.py, copied so that this folder runs alone.
"""
import math

import numpy as np

# ----------------------------------------------------------------- kernels --
def phi_tps(r):
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r * r * np.log(r)
    return np.where(r > 0, out, 0.0)


def phi_linear(r):
    return -np.asarray(r, dtype=float)


def phi_cubic(r):
    return np.asarray(r, dtype=float) ** 3


KERNELS = {"thin_plate_spline": phi_tps, "linear": phi_linear, "cubic": phi_cubic}


def pairwise(A, B):
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)


def poly(X, degree):
    X = np.atleast_2d(X)
    if degree < 0:
        return np.zeros((X.shape[0], 0))
    if degree == 0:
        return np.ones((X.shape[0], 1))
    if degree == 1:
        return np.column_stack([np.ones(X.shape[0]), X])
    raise ValueError("degree 0 or 1 only")


def saddle_matrix(X, lam, kernel, degree):
    n = X.shape[0]
    K = KERNELS[kernel](pairwise(X, X))
    P = poly(X, degree)
    lam = np.broadcast_to(np.asarray(lam, dtype=float), (n,))
    M = np.block([[K + np.diag(lam), P], [P.T, np.zeros((P.shape[1], P.shape[1]))]])
    return K, P, M


def fit_coeffs(X, y, lam, kernel="thin_plate_spline", degree=1):
    """(c, q) of SciPy's system for data (X, y)."""
    K, P, M = saddle_matrix(X, lam, kernel, degree)
    sol = np.linalg.solve(M, np.concatenate([y, np.zeros(P.shape[1])]))
    return sol[:X.shape[0]], sol[X.shape[0]:]


def evaluate(Q, X, c, q, kernel="thin_plate_spline", degree=1):
    return KERNELS[kernel](pairwise(Q, X)) @ c + poly(Q, degree) @ q


def smoother_weights(X, lam, Q, kernel="thin_plate_spline", degree=1):
    """L (m, n): the fit at the query rows Q is L @ y, for every y (the spline is linear in y)."""
    K, P, M = saddle_matrix(X, lam, kernel, degree)
    kq = KERNELS[kernel](pairwise(Q, X))
    pq = poly(Q, degree)
    Z = np.linalg.solve(M, np.hstack([kq, pq]).T)       # M symmetric
    return Z[:X.shape[0], :].T


def hat_trace(X, lam, kernel="thin_plate_spline", degree=1):
    """df = trace of the hat matrix. Fitted values are y - lam c, c = A y with A the
    top-left n x n block of M^-1, so trace H = n - lam trace(A) (scalar lam)."""
    n = X.shape[0]
    K, P, M = saddle_matrix(X, lam, kernel, degree)
    E = np.zeros((M.shape[0], n))
    E[:n, :] = np.eye(n)
    A = np.linalg.solve(M, E)[:n, :]
    return float(n - lam * np.trace(A))


def affine_null_basis(X, degree):
    """Orthonormal basis Z of {c : P^T c = 0}."""
    P = poly(X, degree)
    Qf, _ = np.linalg.qr(P, mode="complete")
    return Qf[:, P.shape[1]:]


# ----------------------------------------------------------------- fixture --
FIX = dict(d_range=(0.2, 4.0), mu_range=(0.3, 3.0), phi_hi=0.5 * math.pi,
           queries=((0.25, 5.0, 0.10), (0.3, 20.0, 0.25), (0.6, 10.0, 0.6), (1.0, 30.0, 1.2),
                    (2.0, 10.0, 2.0), (3.0, 45.0, 4.0), (1.0, 70.0, 3.0)))


def truth(ln_d, phi, ln_a):
    """The fixture's mean ratio b (handoff s.8.3): 1 + 0.06/d + 0.15 sin^2 phi + 0.05 ln alpha."""
    return 1.0 + 0.06 / np.exp(ln_d) + 0.15 * np.sin(phi) ** 2 + 0.05 * ln_a


def tau_of_d(d):
    """Heteroscedastic SD of the ratio: 0.009 at 1.5 um and above, 0.18 at 0.6 um, capped at 0.30."""
    return np.clip(0.009 * (np.asarray(d) / 1.5) ** (-3.27), 0.009, 0.30)


def design(rng, n):
    lo, hi = FIX["d_range"]
    ln_d = rng.uniform(math.log(lo), math.log(hi), n)
    phi = rng.uniform(0.0, FIX["phi_hi"], n)
    mlo, mhi = FIX["mu_range"]
    ln_mu = rng.uniform(math.log(mlo), math.log(mhi), n)
    ln_a = ln_mu + ln_d - np.log(np.cos(phi))
    return np.column_stack([ln_d, phi, ln_a])


class Scaler:
    """Range normalisation (D-054 (i)): ln d and phi by the design ranges, ln alpha by the rows' range."""

    def __init__(self, X):
        lo, hi = FIX["d_range"]
        self.lo = np.array([math.log(lo), 0.0, X[:, 2].min()])
        self.span = np.array([math.log(hi) - math.log(lo), FIX["phi_hi"], X[:, 2].max() - X[:, 2].min()])

    def __call__(self, X):
        return (np.asarray(X) - self.lo) / self.span


def query_raw():
    q = np.array(FIX["queries"], dtype=float)
    return np.column_stack([np.log(q[:, 0]), np.radians(q[:, 1]), np.log(q[:, 2])])
