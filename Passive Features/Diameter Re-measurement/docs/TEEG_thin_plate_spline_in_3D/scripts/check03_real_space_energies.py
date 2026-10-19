"""check03_real_space_energies.py -- the roughness of a spline, integrated in real space, equals a
constant times c^T K c (2026-10-18; standalone). Document sections 3.3-3.4. Independent of Fourier theory.

f(x) = sum_n c_n phi(x - u_n) (+ a polynomial, which the second derivatives do not see), with the
side conditions P^T c = 0 of each case. Integrated by adaptive quadrature over the whole line or plane:
 1-D, phi = |x|   (SciPy 'linear'), sum c = 0:              int f'^2 dx            = 2  c^T K c
 1-D, phi = |x|^3 (SciPy 'cubic'),  sum c = sum c u = 0:    int f''^2 dx           = 12 c^T K c
 2-D, phi = r^2 ln r ('thin_plate_spline'), affine moments: int (f_xx^2 + 2 f_xy^2 + f_yy^2) = 8 pi c^T K c
(the constants are the point-source strengths of check01: |x|'' = 2 delta, (|x|^3)'''' = 12 delta,
 Delta^2 r^2 ln r = 8 pi delta). The 1-D integrands vanish outside the hull of the u_n, so those are exact up to
quadrature; the 2-D integral runs in polar coordinates about the centroid, with break points at the centres.
Tolerance: relative 1e-8 in 1-D, 1e-5 in 2-D (fixed before running).
Run:  python3 check03_real_space_energies.py      (about 1 min)
"""
import math

import numpy as np
from scipy import integrate

from tps_common import KERNELS, affine_null_basis, pairwise

CFG = dict(seed=20261023, n1=6, n2=6, trials=2, tol1=1e-8, tol2=1e-5)


def one_d(rng):
    ok = True
    for t in range(CFG["trials"]):
        u = np.sort(rng.uniform(0, 1, CFG["n1"]))
        U = u[:, None]
        R = pairwise(U, U)
        # linear, degree 0: f' = sum c sign(x - u)
        c = affine_null_basis(U, 0) @ rng.standard_normal(CFG["n1"] - 1)
        fp = lambda x: float(np.sum(c * np.sign(x - u)))
        J = sum(integrate.quad(lambda x: fp(x) ** 2, a, b)[0] for a, b in zip(u[:-1], u[1:]))
        q = float(c @ KERNELS["linear"](R) @ c)
        rel = abs(J - 2 * q) / abs(2 * q)
        ok &= rel <= CFG["tol1"]
        print("1-D |x|  : int f'^2 = %.10f   2 c^T K c = %.10f   rel %.1e %s" % (J, 2 * q, rel,
                                                                             "PASS" if rel <= CFG["tol1"] else "FAIL"))
        # cubic, degree 1: f'' = 6 sum c |x - u|
        c = affine_null_basis(U, 1) @ rng.standard_normal(CFG["n1"] - 2)
        fpp = lambda x: 6.0 * float(np.sum(c * np.abs(x - u)))
        J = sum(integrate.quad(lambda x: fpp(x) ** 2, a, b)[0] for a, b in zip(u[:-1], u[1:]))
        outside = abs(fpp(u[0] - 0.5)) + abs(fpp(u[-1] + 0.5))
        q = float(c @ KERNELS["cubic"](R) @ c)
        rel = abs(J - 12 * q) / abs(12 * q)
        ok &= rel <= CFG["tol1"]
        print("1-D |x|^3: int f''^2 = %.10f  12 c^T K c = %.10f  rel %.1e (f'' outside the hull: %.1e) %s"
              % (J, 12 * q, rel, outside, "PASS" if rel <= CFG["tol1"] else "FAIL"))
    return ok


def hessian_tps_2d(x, U, c):
    """Hessian of sum c_n r_n^2 ln r_n at the point x (2,): (2 ln r + 1) I + 2 e e^T per centre."""
    D = x[None, :] - U
    r2 = np.sum(D * D, axis=1)
    lnr = 0.5 * np.log(r2)
    a = c * (2 * lnr + 1)
    b = 2 * c / r2
    h11 = np.sum(a + b * D[:, 0] ** 2)
    h22 = np.sum(a + b * D[:, 1] ** 2)
    h12 = np.sum(b * D[:, 0] * D[:, 1])
    return h11, h12, h22


def two_d(rng):
    ok = True
    for t in range(CFG["trials"]):
        U = rng.uniform(0, 1, (CFG["n2"], 2))
        c = affine_null_basis(U, 1) @ rng.standard_normal(CFG["n2"] - 3)
        cen = U.mean(axis=0)
        rho_n = np.linalg.norm(U - cen, axis=1)
        th_n = np.mod(np.arctan2(U[:, 1] - cen[1], U[:, 0] - cen[0]), 2 * math.pi)

        def ring(rho):
            pts = [th for th, rn in zip(th_n, rho_n) if abs(rn - rho) < 0.05]

            def g(th):
                x = cen + rho * np.array([math.cos(th), math.sin(th)])
                h11, h12, h22 = hessian_tps_2d(x, U, c)
                return h11 * h11 + 2 * h12 * h12 + h22 * h22
            val, _ = integrate.quad(g, 0, 2 * math.pi, points=sorted(pts) or None, limit=400, epsabs=1e-12,
                                    epsrel=1e-10)
            return rho * val

        r1 = 1.5
        near, _ = integrate.quad(ring, 0, r1, points=sorted(rho_n.tolist()), limit=400, epsabs=1e-10, epsrel=1e-9)
        far, _ = integrate.quad(ring, r1, np.inf, limit=400, epsabs=1e-12, epsrel=1e-9)
        J = near + far
        q = float(c @ KERNELS["thin_plate_spline"](pairwise(U, U)) @ c)
        rel = abs(J - 8 * math.pi * q) / abs(8 * math.pi * q)
        good = rel <= CFG["tol2"]
        ok &= good
        print("2-D r^2 ln r: bending energy = %.8f (near %.6f + far %.2e)   8 pi c^T K c = %.8f   rel %.1e %s"
              % (J, near, far, 8 * math.pi * q, rel, "PASS" if good else "FAIL"))
    return ok


def main():
    rng = np.random.default_rng(CFG["seed"])
    ok = one_d(rng)
    ok &= two_d(rng)
    print("overall:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
