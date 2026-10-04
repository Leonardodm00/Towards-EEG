"""Axial FWHM of the scalar Debye PSF (NA 1.4, n 1.515, lambda 0.55) and pixel pitch check."""
import io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("psf_check.py").read().split('print("=== Part A')[0])
dzs = np.linspace(0, 1.0, 1001)
I = np.array([debye_I(np.array([0.0]), dz=z)[0] for z in dzs]); I /= I[0]
i = np.argmax(I < 0.5)
print("axial FWHM (on-axis intensity) = %.3f um" % (2 * np.interp(0.5, [I[i], I[i-1]], [dzs[i], dzs[i-1]])))
print("paraxial-style 2 n lambda / NA^2 = %.3f um" % (2 * N_OIL * LAM / NA**2))
print("pixel pitch 4.54 / (63*0.63) = %.5f um" % (4.54 / (63 * 0.63)))
print("critical angle glass->air = %.1f deg; theta_obj = %.2f deg" % (np.degrees(np.arcsin(1/1.515)), np.degrees(np.arcsin(1.4/1.515))))
for lam in (0.45, 0.55, 0.65):
    print("lambda %.2f: Airy first zero %.3f, FWHM %.3f um" % (lam, 0.61*lam/NA, 0.51*lam/NA))
