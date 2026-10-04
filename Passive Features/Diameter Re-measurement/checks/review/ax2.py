import io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("psf_check.py").read().split('print("=== Part A')[0])
dzs = np.linspace(0, 1.2, 1201)
I = np.array([debye_I(np.array([0.0]), dz=z)[0] for z in dzs]); I /= I[0]
i = np.argmax(np.diff(I) > 0)  # first local min
print("Debye axial first minimum at %.3f um, I=%.4f" % (dzs[i], I[i]))
n,lam,NA=N_OIL,LAM,NA
print("closed form lam/(n-sqrt(n^2-NA^2)) = %.3f" % (lam/(n-np.sqrt(n*n-NA*NA))))
print("paraxial FWHM 1.772 n lam/NA^2 = %.3f" % (1.772*n*lam/NA**2))
print("0.88 lam/(n-sqrt) FWHM approx = %.3f" % (0.88*lam/(n-np.sqrt(n*n-NA*NA))))
# focal-shift / geometric growth in stage units for mismatched medium
for nm in (1.33,1.42,1.45,1.49):
    th_o=np.arcsin(min(1.4,nm)/n)
    print(nm, "stage-unit marginal growth tan(theta_oil)/2 = %.3f" % (np.tan(th_o)/2), 
          "tissue-unit tan(theta_m)/2 =", "%.3f"%(np.tan(np.arcsin(1.4/nm))/2) if nm>1.4 else "inf")
