"""Check by computation for Q10 (the renderer's kernel becoming the SGT family): which 1-D profile
kernels g(v) of the family are the line projection of a NONNEGATIVE isotropic 2-D point-spread function
h(r)?  For an even g, h(r) = -(1/pi) int_r^inf g'(v) / sqrt(v^2 - r^2) dv (inverse Abel transform;
textbook, from memory).  We evaluate h on a grid for members of the symmetric SGT (zeta = 1) at unit
scale: generalized normal exp(-|v|^beta) for beta = 1, 1.5, 2, 3, 4, 6, 10, and Student-like
(1 + xi v^2)^(-(1/xi + 1/2)) for xi = 0.5, 2.  Report min h / max h (negative = not a valid PSF).
Tolerance: the quadrature's relative error is ~1e-8; a negative dip below -1e-4 of the peak counts."""
import numpy as np
from scipy.integrate import quad

def inv_abel(dg, r, upper=40.0):
    # h(r) = -(1/pi) int_r^upper g'(v)/sqrt(v^2 - r^2) dv, substitution v = r cosh t removes the singularity
    if r == 0.0:
        f = lambda v: -dg(v) / v if v > 0 else 0.0
        val, _ = quad(f, 1e-12, upper, limit=400)
        return val / np.pi
    tmax = np.arccosh(upper / r)
    f = lambda t: -dg(r * np.cosh(t))
    val, _ = quad(f, 0.0, tmax, limit=400)
    return val / np.pi

def gennorm_dg(beta):
    return lambda v: -beta * v ** (beta - 1) * np.exp(-v ** beta) if v > 0 else 0.0

def student_dg(xi):
    p = 1.0 / xi + 0.5
    return lambda v: -p * 2 * xi * v * (1 + xi * v * v) ** (-p - 1)

rs = np.linspace(0.0, 6.0, 121)
print("member                 min h / max h   first negative r (if any)")
for beta in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0):
    h = np.array([inv_abel(gennorm_dg(beta), r) for r in rs])
    neg = rs[h < -1e-4 * h.max()]
    print("gen. normal beta=%-5.1f  %+9.2e        %s" % (beta, h.min() / h.max(), ("%.2f" % neg[0]) if neg.size else "none"))
for xi in (0.5, 2.0):
    h = np.array([inv_abel(student_dg(xi), r, upper=400.0) for r in rs])
    neg = rs[h < -1e-4 * h.max()]
    print("Student xi=%-4.1f         %+9.2e        %s" % (xi, h.min() / h.max(), ("%.2f" % neg[0]) if neg.size else "none"))
