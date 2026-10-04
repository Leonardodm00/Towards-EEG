"""Debye LSF core sigma vs defocus; compare three growth forms for D(dz)=sigma^2(dz)-sigma^2(0)."""
import numpy as np, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    exec(open('psf_check.py').read().split('print("\\n=== Part B')[0])
from scipy.optimize import curve_fit
dzs = np.array([0, .07, .14, .21, .28, .42, .56, .70, .84])
x = np.linspace(0, 1.5, 751)
s = []
for dz in dzs:
    Ig = debye_I(rgrid, dz=dz) / I0
    l = lsf_from_radial(lambda rr, Ig=Ig: np.interp(rr, rgrid, Ig, right=0.0), x); l = l / l[0]
    fw = 2 * np.interp(0.5, l[::-1], x[::-1]); c = x <= fw
    s.append(curve_fit(gauss, x[c], l[c], p0=[0.1])[0][0])
s = np.array(s); D = s**2 - s[0]**2
print("dz   sigma   D"); [print("%.2f %.4f %.5f" % r) for r in zip(dzs, s, D)]
forms = {"k2 dz^2": (lambda z, k: k*z**2, [1]),
         "k2 dz^2 + l dz^4": (lambda z, k, l: k*z**2 + l*z**4, [1, 0]),
         "k2 dz^4/(dz^2+a^2)": (lambda z, k, a: k*z**4/(z**2 + a**2), [1, .2])}
for fit_to in (0.28, 0.84):
    m = dzs <= fit_to + 1e-9
    print("\nfit on dz <= %.2f; rms error on all points (um^2):" % fit_to)
    for n, (f, p0) in forms.items():
        p, _ = curve_fit(f, dzs[m], D[m], p0=p0, maxfev=20000)
        print("  %-22s params %s  rms_all %.5f  pred@0.84 %.4f (true %.4f)" % (n, np.round(p, 4), np.sqrt(np.mean((f(dzs, *p) - D)**2)), f(.84, *p), D[-1]))
