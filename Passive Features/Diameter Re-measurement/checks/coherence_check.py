"""Coherent, Koehler (S = 1) and incoherent-model images of a straight absorbing branch, in focus.

Question (theory, 2026-10-08): how far is the incoherent form B (T * h_0) of handoff Eq. 11 from
the image the microscope actually forms when the condenser aperture is filled to the objective's
NA (S = 1), and what changes when the illumination is coherent (one plane wave, S -> 0)?

Model (thin object, scalar optics, in focus, monochromatic, no camera chain, no noise):
  object      amplitude transmission t(x) = exp(-mu * chord(x) / 2), chord = 2 sqrt(r^2 - x^2),
              so the intensity transmittance is T = |t|^2 = exp(-mu * chord)
  coherent    one plane wave along the axis:  I = |t * h_a|^2  (pupil |f| <= NA / lambda)
  koehler     every point (a, b) of the condenser aperture, a^2 + b^2 <= (NA/lambda)^2, is a
              plane wave; the points are mutually incoherent, so the intensities of their
              coherent images add (uniform weight over the aperture). A branch along y makes
              the image depend on x only; for the point (a, b) the pupil passes
              |f_x| <= sqrt((NA/lambda)^2 - b^2).
  model       I = T * LSF_0, the line-spread function of the incoherent PSF h_0 = |h_a|^2
              (OTF of a circular pupil, cut-off 2 NA / lambda, along f_x).
Reported per case: centre dip 1 - I(0); the brightest point outside the branch, I - 1; the FWHM
of the dip. These are illustrations of mechanisms, not calibrated values: no defocus, no depth
extent of the node (thin-object projection), no camera; the diaphragm setting of the Allen
acquisitions is not known (optics section 3.6).

Smoke checks (asserted before the table is printed):
  1. no object (t = 1): all three images equal 1;
  2. weak-object limit (mu d = 0.01): Koehler and model agree to second order in 1 - t;
  3. at S = 1 the true image is never brighter than the model, nor than the background
     (I_koehler <= I_model <= 1), the sign derived in the theory chat of 2026-10-08
     (optics section 3.6, note of v1.3).

Run from inside checks/ (about 4 minutes):
    python coherence_check.py
Reference output: coherence_check.out (2026-10-08).
"""
import numpy as np

LAM_UM, NA = 0.55, 1.4              # wavelength in vacuum (um); objective = condenser NA
FC = NA / LAM_UM                    # coherent cut-off (cycles / um)
N, L_UM = 2 ** 14, 40.0             # samples, field width (um)
X = (np.arange(N) - N // 2) * (L_UM / N)
F = np.fft.fftfreq(N, L_UM / N)
CASES = ((0.5, 0.1), (0.5, 1.0), (0.5, 3.0), (1.0, 0.1), (1.0, 1.0), (1.0, 3.0))  # (d um, mu d)


def transmission(d, mud):
    """Amplitude transmission of a flat round branch of diameter d with optical depth mu d."""
    r, mu = d / 2, mud / d
    chord = 2 * np.sqrt(np.clip(r ** 2 - X ** 2, 0, None))
    return np.exp(-mu * chord / 2)


def coherent(t):
    return np.abs(np.fft.ifft(np.fft.fft(t) * (np.abs(F) <= FC))) ** 2


def koehler(t, step=1):
    """Incoherent sum over aperture points; tilts on the FFT grid (a * L integer), every step-th.

    step=1 samples the aperture at the FFT grid spacing 1/L. A coarser step aliases the image
    with period L/step (replica dips 13 um away for step=3, which break check 3 by up to 2e-4).
    """
    k = np.arange(-int(FC * L_UM), int(FC * L_UM) + 1)[::step]
    g = k / L_UM
    acc, n = np.zeros(N), 0
    for a in g:
        ramp = np.exp(2j * np.pi * a * X)
        spec = np.fft.fft(t * ramp)
        for b in g:
            if a * a + b * b > FC * FC:
                continue
            acc += np.abs(np.fft.ifft(spec * (np.abs(F) <= np.sqrt(FC * FC - b * b)))) ** 2
            n += 1
    return acc / n


def model(t):
    nu = np.clip(np.abs(F) / (2 * FC), 0, 1)
    otf = (2 / np.pi) * (np.arccos(nu) - nu * np.sqrt(1 - nu ** 2))
    return np.real(np.fft.ifft(np.fft.fft(np.abs(t) ** 2) * otf))


def metrics(img, d):
    """Centre dip, brightest point outside the branch (I - 1) and dip FWHM within |x| < 5 um."""
    win = np.abs(X) < 5
    i_w, x_w = img[win], X[win]
    dip = 1 - i_w
    half = dip.max() / 2
    above = x_w[dip >= half]
    return dip[np.argmin(np.abs(x_w))], i_w[np.abs(x_w) > d / 2].max() - 1, above.max() - above.min()


def smoke():
    one = np.ones(N, dtype=complex)
    for name, fn in (("coherent", coherent), ("koehler", koehler), ("model", model)):
        assert np.allclose(fn(one), 1, atol=1e-9), name
    t = transmission(0.5, 0.01)
    diff = np.abs(koehler(t) - model(t)).max()
    dip = (1 - model(t)).max()
    assert diff < 0.02 * dip, (diff, dip)
    for d, mud in CASES:
        t = transmission(d, mud)
        ik, im = koehler(t), model(t)
        assert (ik <= im + 1e-6).all() and (im <= 1 + 1e-6).all(), (d, mud)
    print(f"smoke: 3 checks passed (weak limit: max |koehler - model| = {diff:.1e}, dip {dip:.4f})")


def main():
    smoke()
    print("per method: centre dip 1-I(0) ; brightest point outside the branch, I-1 ; dip FWHM (um)")
    for d, mud in CASES:
        t = transmission(d, mud)
        imgs = {"coherent": coherent(t), "koehler S=1": koehler(t), "model B(T*h0)": model(t)}
        cells = []
        for name, img in imgs.items():
            c, o, w = metrics(img, d)
            cells.append(f"{name} {c:.3f} ; {o:+.3f} ; {w:.3f}")
        excess = imgs["model B(T*h0)"] - imgs["koehler S=1"]
        print(f"d={d} um, mu*d={mud}: " + " | ".join(cells)
              + f" || model brighter than koehler by at most {excess.max():.4f} B")


if __name__ == "__main__":
    main()
