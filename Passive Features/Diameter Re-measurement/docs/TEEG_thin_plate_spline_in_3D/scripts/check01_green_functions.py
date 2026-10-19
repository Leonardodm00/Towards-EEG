"""check01_green_functions.py -- the Green's functions of the polyharmonic family, exactly (sympy).
Document: "The thin-plate spline in three dimensions", sections 3.3-3.6 (2026-10-18; standalone).

For a radial function f(r) on R^d (r = |x| > 0) the Laplacian is  Delta f = f'' + (d - 1) f' / r.
For each (d, phi, m) of CFG:
 (a) Delta^m phi for r > 0: zero means phi is m-harmonic away from its centre;
 (b) the strength C of the point source, Delta^m phi = C delta:  C = lim_{eps -> 0+} |S^{d-1}| eps^(d-1)
     d/dr (Delta^(m-1) phi)(eps), the outward flux of grad Delta^(m-1) phi through a small sphere
     (|S^0| = 2: in 1-D the 'sphere' is the two points +-eps, and C is the jump of phi^(2m-1) at 0);
 (c) r^2 ln r on R^3: Delta^k for k = 1..4, the source strength of Delta^2 (it has none: the residual
     6 / r^2 is locally integrable), so no integer power of the Laplacian reduces it to a point source;
 (d) regularity at the centre: the limits of phi', phi'', phi''' as r -> 0+.
Predictions (textbook, from memory, fixed before running): 1-D |x| (m=1): C = 2; 1-D |x|^3 (m=2): C = 12;
2-D r^2 ln r (m=2): C = 8 pi; 3-D r (m=2): C = -8 pi; 3-D r^3 (m=3): C = -96 pi; 3-D r^2 ln r: Delta^2 = 6/r^2,
no point source. Pass rule: exact symbolic equality.
Run:  python3 check01_green_functions.py      (a few seconds)
"""
import sympy as sp

r, eps = sp.symbols("r epsilon", positive=True)

CFG = [  # (d, name, phi(r), m, predicted source strength C)
    (1, "|x|", r, 1, 2),
    (1, "|x|^3", r ** 3, 2, 12),
    (2, "r^2 ln r", r ** 2 * sp.log(r), 2, 8 * sp.pi),
    (3, "r", r, 2, -8 * sp.pi),
    (3, "r^3", r ** 3, 3, -96 * sp.pi),
]


def lap(f, d):
    return sp.simplify(sp.diff(f, r, 2) + (d - 1) / r * sp.diff(f, r))


def lap_power(f, d, m):
    for _ in range(m):
        f = lap(f, d)
    return sp.simplify(f)


def sphere_area(d):
    return 2 * sp.pi ** sp.Rational(d, 2) / sp.gamma(sp.Rational(d, 2))


def source_strength(f, d, m):
    g = lap_power(f, d, m - 1)
    flux = sphere_area(d) * eps ** (d - 1) * sp.diff(g, r).subs(r, eps)
    return sp.simplify(sp.limit(flux, eps, 0, "+"))


def main():
    ok = True
    print("(a)-(b) Delta^m phi away from the centre, and the point-source strength C")
    for d, name, f, m, c_pred in CFG:
        resid = lap_power(f, d, m)
        c = source_strength(f, d, m)
        good = (sp.simplify(resid) == 0) and (sp.simplify(c - c_pred) == 0)
        ok &= good
        print("  d=%d  phi=%-9s m=%d: Delta^m phi (r>0) = %-6s  C = %-8s predicted %-8s %s"
              % (d, name, m, sp.simplify(resid), c, c_pred, "PASS" if good else "FAIL"))
    print("(c) r^2 ln r on R^3")
    f = r ** 2 * sp.log(r)
    for k in range(1, 5):
        print("  Delta^%d (r^2 ln r) = %s" % (k, lap_power(f, 3, k)))
    c2 = source_strength(f, 3, 2)
    good = sp.simplify(lap_power(f, 3, 2) - 6 / r ** 2) == 0 and c2 == 0
    ok &= good
    print("  Delta^2 = 6/r^2 and point-source strength of Delta^2 = %s (predicted 0) %s" % (c2, "PASS" if good else "FAIL"))
    print("  for contrast, 2-D: Delta^2 (r^2 ln r) = %s" % lap_power(f, 2, 2))
    print("(d) regularity at the centre: limits as r -> 0+ of phi', phi'', phi'''")
    for name, g in (("r^2 ln r", r ** 2 * sp.log(r)), ("r", r), ("r^3", r ** 3)):
        lims = [sp.limit(sp.diff(g, r, k), r, 0, "+") for k in (1, 2, 3)]
        print("  %-9s phi'(0+) = %-4s phi''(0+) = %-5s phi'''(0+) = %s" % (name, *lims))
    print("overall:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
