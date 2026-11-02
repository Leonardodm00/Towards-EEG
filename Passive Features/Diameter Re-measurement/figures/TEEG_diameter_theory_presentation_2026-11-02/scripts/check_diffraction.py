"""check_diffraction.py -- Check by computation behind a brief explanation of
diffraction (TEEG, renderers document section 3.6 and glossary).

Model (model level, analytic): scalar Debye theory for an aberration-free,
index-matched objective whose pupil is uniform in the direction sines
s = (s_x, s_y) on the disc |s| <= s_o, with s_o = NA_o / n. For each fixed
defocus delta (um) and lateral radius r (um) from the axis,

    U(r, delta) = int_0^{s_o} exp(i k n sqrt(1 - rho^2) delta) J0(k n rho r) 2 pi rho d rho,

with k = 2 pi / lambda (lambda = vacuum wavelength) and I = |U|^2; the
azimuthal integral has been done analytically (2 pi J0). The integral is
evaluated by Gauss-Legendre quadrature (computed level).

Claims tested; tolerances fixed before running:
  T1  at delta = 0, I(r, 0)/I(0, 0) equals the Airy pattern [2 J1(v)/v]^2,
      v = (2 pi NA_o / lambda) r: max |difference| < 1e-8 for r in (0, 1] um.
  T2  the first zero of U(r, 0) lies at r0 = j_{1,1} lambda / (2 pi NA_o)
      (= 0.6098 lambda / NA_o): |difference| < 1e-6 um.
  T3  at the quarter-wave defocus delta_qw = lambda / (4 n (1 - cos theta_o)),
      where the largest pupil phase error is pi/2, the spot is essentially
      the in-focus one: Strehl I(0, delta_qw)/I(0, 0) >= 0.75 and FWHM
      within 10 % of the in-focus FWHM.
  T4  (reported, no tolerance) Strehl and FWHM at 2 delta_qw; the geometric
      blur-disc radius |delta| tan(theta_o) at delta_qw and 2 delta_qw; the
      paraxial Strehl [sin(phi/2)/(phi/2)]^2 at peak-to-valley phase phi;
      and, for information, the first zero with the aplanatic point-imaging
      apodisation P(s) = (1 - |s|^2)^(-1/4) instead of a uniform pupil.
No random numbers are used. Run: python3 -I check_diffraction.py
"""
import numpy as np
from scipy import special, optimize

CFG = dict(
    lam=0.55,     # vacuum wavelength, um
    na=1.4,       # objective numerical aperture NA_o
    n=1.515,      # immersion / specimen index (index-matched)
    n_gl=400,     # Gauss-Legendre nodes over rho in [0, s_o]
    r_max=1.0,    # largest lateral radius tested in T1, um
    n_r=4001,     # grid points in r for T1
)


# ---------- model ----------
def pupil_nodes(cfg):
    """Gauss-Legendre nodes and weights for rho in [0, s_o]."""
    s_o = cfg["na"] / cfg["n"]
    x, w = np.polynomial.legendre.leggauss(cfg["n_gl"])
    return 0.5 * s_o * (x + 1.0), 0.5 * s_o * w


def field(r, delta, cfg, apod=None):
    """U(r, delta) for an array r (um), scalar defocus delta (um).
    apod: None for a uniform pupil, or a function of rho giving P(s)."""
    rho, w = pupil_nodes(cfg)
    kn = 2.0 * np.pi / cfg["lam"] * cfg["n"]
    p = np.ones_like(rho) if apod is None else apod(rho)
    weights = p * np.exp(1j * kn * np.sqrt(1.0 - rho**2) * delta) * 2.0 * np.pi * rho * w
    return special.j0(kn * np.outer(np.atleast_1d(r), rho)) @ weights


def intensity(r, delta, cfg, apod=None):
    return np.abs(field(r, delta, cfg, apod)) ** 2


# ---------- analysis ----------
def airy(r, cfg):
    v = 2.0 * np.pi * cfg["na"] / cfg["lam"] * np.asarray(r)
    return (2.0 * special.j1(v) / v) ** 2


def first_zero(cfg, apod=None):
    f = lambda r: field(np.array([r]), 0.0, cfg, apod).real[0]
    return optimize.brentq(f, 0.15, 0.30, xtol=1e-12)


def fwhm(delta, cfg):
    """Full width at half of the spot's own on-axis maximum."""
    i0 = intensity(np.array([0.0]), delta, cfg)[0]
    g = lambda r: intensity(np.array([r]), delta, cfg)[0] / i0 - 0.5
    return 2.0 * optimize.brentq(g, 1e-6, 0.30, xtol=1e-12)


def main(cfg):
    s_o = cfg["na"] / cfg["n"]
    cos_o, tan_o = np.sqrt(1.0 - s_o**2), s_o / np.sqrt(1.0 - s_o**2)
    d_qw = cfg["lam"] / (4.0 * cfg["n"] * (1.0 - cos_o))
    results = {}

    r = np.linspace(cfg["r_max"] / cfg["n_r"], cfg["r_max"], cfg["n_r"])
    i_norm = intensity(r, 0.0, cfg) / intensity(np.array([0.0]), 0.0, cfg)[0]
    err1 = np.max(np.abs(i_norm - airy(r, cfg)))
    results["T1"] = err1 < 1e-8
    print(f"T1 max|I/I0 - Airy| on (0, {cfg['r_max']}] um = {err1:.2e}  "
          f"{'PASS' if results['T1'] else 'FAIL'}")

    r0 = first_zero(cfg)
    r0_pred = special.jn_zeros(1, 1)[0] * cfg["lam"] / (2.0 * np.pi * cfg["na"])
    results["T2"] = abs(r0 - r0_pred) < 1e-6
    print(f"T2 first zero {r0:.6f} um vs j11*lam/(2 pi NA) = {r0_pred:.6f} um "
          f"(= {r0_pred * cfg['na'] / cfg['lam']:.4f} lam/NA)  "
          f"{'PASS' if results['T2'] else 'FAIL'}")

    i00 = intensity(np.array([0.0]), 0.0, cfg)[0]
    w0 = fwhm(0.0, cfg)
    print(f"   delta_qw = {d_qw:.4f} um; in-focus FWHM = {w0:.4f} um "
          f"(= {w0 * cfg['na'] / cfg['lam']:.4f} lam/NA)")
    for mult in (1, 2):
        d = mult * d_qw
        strehl = intensity(np.array([0.0]), d, cfg)[0] / i00
        wd = fwhm(d, cfg)
        on_axis_max = np.all(intensity(r, d, cfg) <= intensity(np.array([0.0]), d, cfg)[0] * (1 + 1e-12))
        par = (np.sin(mult * np.pi / 4) / (mult * np.pi / 4)) ** 2
        line = (f"   delta = {mult} x delta_qw = {d:.4f} um: Strehl {strehl:.4f} "
                f"(paraxial {par:.4f}), FWHM {wd:.4f} um ({wd / w0 - 1:+.1%}), "
                f"max on axis: {on_axis_max}, geometric disc radius {d * tan_o:.4f} um")
        if mult == 1:
            results["T3"] = strehl >= 0.75 and abs(wd / w0 - 1) <= 0.10
            line = "T3" + line[2:] + f"  {'PASS' if results['T3'] else 'FAIL'}"
        else:
            line = "T4" + line[2:]
        print(line)

    r0_apod = first_zero(cfg, apod=lambda rho: (1.0 - rho**2) ** -0.25)
    print(f"T4 info: first zero with (1-|s|^2)^(-1/4) apodisation = {r0_apod:.4f} um "
          f"({r0_apod / r0 - 1:+.1%} vs uniform pupil)")
    print("ALL PASS" if all(results.values()) else f"FAILED: {[k for k, v in results.items() if not v]}")


if __name__ == "__main__":
    main(CFG)
