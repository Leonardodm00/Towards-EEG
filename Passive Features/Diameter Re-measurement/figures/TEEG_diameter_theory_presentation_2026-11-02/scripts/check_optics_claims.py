"""check_optics_claims.py -- Checks by computation behind the explanation of
diffraction and coherence (TEEG, 2026-11-02). Tolerances fixed before running.
Uses optics_model.py (same folder). Run: python3 check_optics_claims.py
Writes check_results.json (numbers quoted in the explanation).

Configuration (model level): scalar optics, lam = 0.55 um, n_oil = 1.515,
NA_o = 1.4, Koehler source uniform in direction sines (the sine document's W),
thin-object (projection) tube, in focus unless stated. No random numbers.

  D1  in-focus Debye spot = Airy pattern; first zero 0.61 lam/NA_o.        tol 1e-10 / 1e-6 um
  D2  encircled energy of the in-focus spot = 1 - J0^2 - J1^2 (Rayleigh).    tol 1e-4
  D3  at the quarter-wave defocus the 50 %-energy radius grows < 10 %.        (fixed before run)
  D4  far from focus the wave radius approaches the geometric one: at
      |delta| = 0.84 um they agree within 15 %.                                (fixed before run)
  C1  van Cittert-Zernike: source quadrature of <exp(i 2 pi nu_c dx)>_W = 2J1/beta. tol 1e-8
  C2  H_1 (S = 1) = the incoherent OTF; H_S cutoff at (1+S) NA_o/lam;
      H_S = 1 below (1-S) NA_o/lam.                                            tol 1e-12
  C3  S = 1 Abbe image of a tube matches the incoherent model to FIRST order:
      max|I_inc - I_S1| scales as mu^2 (ratio for mu d 0.05 -> 0.1 in [3.6, 4.4]);
      for S = 0.75 it scales as mu (ratio in [1.8, 2.2]).
  C4  I_inc - I_S1 >= -1e-6 everywhere (mu d = 0.5, 1, 3, 10): the incoherent
      model is never darker.
  C5  I_inc = I_S1 + dark-field image of the directions outside the objective's
      cone, within 5 % of max(I_inc - I_S1) (mu d = 1 and 3).
  C6  a weak phase tube (no absorption) has no first-order contrast in focus:
      in-focus contrast ratio for phase 0.05 -> 0.1 rad in [3.5, 4.5] (second
      order), and the contrast at delta = 0.28 um exceeds 5x the in-focus one.

Added after the first run, because D3, C4 and C6 FAILED there (their claims
were corrected, not their tolerances); tolerances fixed before these ran:
  D3b at the quarter-wave defocus the CORE is unchanged: FWHM grows < 5 % and
      the peak keeps >= 0.80 of its in-focus value (D3 failed: the 50 %-energy
      radius grows 16 %, light moves into the rings).
  C4b I_inc - I_S1 >= -1e-6 within |v| <= 2 um of the axis, and the global
      minimum (far from the tube) shrinks by >= 1.5x when the source quadrature
      is refined 32x64 -> 64x128 (C4 failed at -1e-4 .. -5e-4, 9-40 um away).
  C6b the expectation for S = 1 was wrong: at S = 1 the phase stays SECOND order
      off focus too (ratio phase 0.05 -> 0.1 rad at delta = 0.28 um in
      [3.5, 4.5]); at S = 0.75 off focus it is FIRST order (ratio in [1.8, 2.2]).
"""
import json
import os
import sys
import numpy as np
from scipy import special, optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optics_model as om  # noqa: E402

CFG = dict(lam=0.55, n=1.515, na_o=1.4, n_v=8192, L_v=80.0, d_tube=1.0)


def main(cfg):
    out, ok = {}, {}
    s_o = cfg["na_o"] / cfg["n"]
    cos_o = np.sqrt(1 - s_o**2)
    d_qw = cfg["lam"] / (4 * cfg["n"] * (1 - cos_o))
    out["delta_qw_um"] = d_qw

    # ---------- D1
    r = np.linspace(1e-4, 1.5, 3001)
    i0 = np.abs(om.psf_field(np.array([0.0]), 0.0, cfg))[0] ** 2
    err = np.max(np.abs(np.abs(om.psf_field(r, 0.0, cfg)) ** 2 / i0 - om.airy_intensity(r, cfg)))
    r0 = optimize.brentq(lambda x: om.psf_field(np.array([x]), 0.0, cfg).real[0], 0.15, 0.3, xtol=1e-13)
    r0_pred = special.jn_zeros(1, 1)[0] * cfg["lam"] / (2 * np.pi * cfg["na_o"])
    ok["D1"] = err < 1e-10 and abs(r0 - r0_pred) < 1e-6
    out.update(first_zero_um=r0, airy_max_err=err)
    half = optimize.brentq(lambda x: om.airy_intensity(np.array([x]), cfg)[0] - 0.5, 0.01, 0.2)
    out["fwhm_um"] = 2 * half

    # ---------- D2, D3, D4 (encircled energy, wave vs geometric)
    R = np.linspace(0, 3.0, 6001)
    ee0 = om.encircled_energy(R, 0.0, cfg)
    ok["D2"] = np.max(np.abs(ee0 - om.encircled_energy_airy(R, cfg))) < 1e-4
    out["ee_err_max"] = float(np.max(np.abs(ee0 - om.encircled_energy_airy(R, cfg))))
    deltas = np.round(np.arange(0, 0.8401, 0.02), 4)
    r50_wave = np.array([om.radius_at_fraction(R, om.encircled_energy(R, d, cfg)) for d in deltas])
    tan50 = (s_o / np.sqrt(2)) / np.sqrt(1 - s_o**2 / 2)
    r50_geo = deltas * tan50
    r50_qw = om.radius_at_fraction(R, om.encircled_energy(R, d_qw, cfg))
    ok["D3"] = r50_qw / r50_wave[0] - 1 < 0.10
    ok["D4"] = abs(r50_wave[-1] / r50_geo[-1] - 1) < 0.15
    strehl_qw = np.abs(om.psf_field(np.array([0.0]), d_qw, cfg))[0] ** 2 / i0

    rr = np.linspace(0, 1.5, 6001)

    def fwhm_at(dl):
        """Core FWHM: twice the innermost radius where I falls to half of its on-axis
        value; NaN unless the centre is the brightest point of the profile and the
        first minimum falls below half of it (otherwise there is no separate core)."""
        ip = np.abs(om.psf_field(rr, dl, cfg)) ** 2
        if ip[0] < ip.max() * (1 - 1e-9):
            return np.nan
        mins = np.nonzero((ip[1:-1] < ip[:-2]) & (ip[1:-1] <= ip[2:]))[0]
        if mins.size and ip[mins[0] + 1] >= 0.5 * ip[0]:
            return np.nan                      # no separate core: the first minimum stays above half max
        k = int(np.argmax(ip < 0.5 * ip[0])) - 1
        x0, x1, y0, y1 = rr[k], rr[k + 1], ip[k], ip[k + 1]
        return 2 * (x0 + (0.5 * ip[0] - y0) * (x1 - x0) / (y1 - y0))
    fwhm_d = np.array([fwhm_at(dl) for dl in deltas])
    strehl_d = np.array([abs(om.psf_field(np.array([0.0]), dl, cfg)[0]) ** 2 for dl in deltas]) / i0
    fwhm_qw = fwhm_at(d_qw)
    ok["D3b"] = fwhm_qw / fwhm_d[0] - 1 < 0.05 and strehl_qw >= 0.80
    out.update(fwhm_qw_um=fwhm_qw, fwhm_d=[None if np.isnan(x) else float(x) for x in fwhm_d],
               strehl_d=strehl_d.tolist(),
               core_defined_up_to_um=float(deltas[max(int(np.argmax(~np.isfinite(fwhm_d))) - 1, 0)]))
    out.update(r50_focus_um=r50_wave[0], r50_qw_um=r50_qw, r50_geo_qw_um=d_qw * tan50,
               tan_theta50=tan50, strehl_qw=strehl_qw,
               r50_wave_084=r50_wave[-1], r50_geo_084=r50_geo[-1],
               delta_equal_geo_focus=r50_wave[0] / tan50,
               deltas=deltas.tolist(), r50_wave=r50_wave.tolist(), r50_geo=r50_geo.tolist())

    # ---------- C1 van Cittert-Zernike
    dx = np.linspace(0, 2.0, 401)
    errs = []
    for na_c in (1.4, 1.05):
        cx, cy, w = om.source_quadrature(na_c / cfg["lam"], 32, 64)
        gam_q = (w[None, :] * np.exp(2j * np.pi * np.outer(dx, cx))).sum(axis=1)
        errs.append(np.max(np.abs(gam_q - om.coherence_function(dx, na_c, cfg["lam"]))))
    ok["C1"] = max(errs) < 1e-8
    out["vcz_err"] = max(errs)
    out["coh_zero_um"] = {str(na): float(special.jn_zeros(1, 1)[0] * cfg["lam"] / (2 * np.pi * na))
                          for na in (1.4, 1.12, 1.05)}

    # ---------- C2 transfer functions
    a_o = cfg["na_o"] / cfg["lam"]
    nu = np.linspace(0, 2.2 * a_o, 2201)
    e_otf = np.max(np.abs(om.transfer_absorption(nu, 1.0, cfg) - om.otf_closed_form(nu, cfg)))
    c2 = e_otf < 1e-12
    for S in (0.25, 0.5, 0.75):
        H = om.transfer_absorption(nu, S, cfg)
        c2 &= np.all(H[nu >= (1 + S) * a_o] < 1e-12) and np.all(np.abs(H[nu <= (1 - S) * a_o] - 1) < 1e-12)
    ok["C2"] = bool(c2)
    out.update(otf_err=e_otf, cutoff_coherent=a_o, cutoff_incoherent=2 * a_o)

    # ---------- C3, C4, C5: the flat tube, in focus
    v, nux = om.profile_grid(cfg)
    d = cfg["d_tube"]
    diffs = {}
    for md in (0.05, 0.1):
        t = om.tube_amplitude(v, d, md / d)
        Ii = om.incoherent_image(t, nux, cfg)
        diffs[md] = (np.max(np.abs(Ii - om.abbe_image(t, nux, 1.0, cfg))),
                     np.max(np.abs(Ii - om.abbe_image(t, nux, 0.75, cfg))))
    rat1 = diffs[0.1][0] / diffs[0.05][0]
    rat075 = diffs[0.1][1] / diffs[0.05][1]
    ok["C3"] = 3.6 <= rat1 <= 4.4 and 1.8 <= rat075 <= 2.2
    out.update(ratio_S1=rat1, ratio_S075=rat075)

    profiles, c4, c4b, c5 = {}, True, True, True
    centre = np.argmin(np.abs(v))
    near = np.abs(v) <= 2.0
    for md in (0.5, 1.0, 3.0, 10.0):
        t = om.tube_amplitude(v, d, md / d)
        Ii = om.incoherent_image(t, nux, cfg)
        I1 = om.abbe_image(t, nux, 1.0, cfg)
        I075 = om.abbe_image(t, nux, 0.75, cfg)
        I0 = om.abbe_image(t, nux, 0.0, cfg)
        c4 &= bool(np.min(Ii - I1) >= -1e-6)
        I1_fine = om.abbe_image(t, nux, 1.0, cfg, n_rho=64, n_psi=128)
        c4b &= bool(np.min((Ii - I1)[near]) >= -1e-6) and \
            (np.min(Ii - I1) >= -1e-6 or abs(np.min(Ii - I1)) / max(abs(np.min(Ii - I1_fine)), 1e-12) >= 1.5)
        rec = dict(inc_centre=Ii[centre], S1_centre=I1[centre], S075_centre=I075[centre],
                   S0_centre=I0[centre], max_inc_minus_S1=float(np.max(Ii - I1)),
                   argmax_inc_minus_S1_um=float(v[np.argmax(Ii - I1)]),
                   max_abs_inc_minus_S075=float(np.max(np.abs(Ii - I075))),
                   max_abs_inc_minus_S0=float(np.max(np.abs(Ii - I0))))
        if md in (1.0, 3.0):
            Idf = om.darkfield_image(t, nux, cfg)
            resid = np.max(np.abs(Ii - (I1 + Idf)))
            rec["darkfield_resid_rel"] = float(resid / np.max(Ii - I1))
            c5 &= rec["darkfield_resid_rel"] < 0.05
            sel = np.abs(v) <= 2.0
            profiles[str(md)] = dict(v=v[sel].tolist(), inc=Ii[sel].tolist(), S1=I1[sel].tolist(),
                                     S075=I075[sel].tolist(), S0=I0[sel].tolist(),
                                     darkfield=Idf[sel].tolist(), T=np.abs(t[sel]).__pow__(2).tolist())
        out[f"tube_mud_{md}"] = rec
    ok["C4"], ok["C4b"], ok["C5"] = c4, c4b, c5
    out["profiles"] = profiles

    # ---------- C6 weak phase tube
    def contrast(phi, delta, S):
        t = om.tube_amplitude(v, d, 0.0, phase_peak=phi)
        return float(np.max(np.abs(om.abbe_image(t, nux, S, cfg, delta=delta) - 1)))
    c6, ph = True, {}
    for S in (1.0, 0.75):
        a, b, c = contrast(0.05, 0.0, S), contrast(0.1, 0.0, S), contrast(0.1, 0.28, S)
        c005 = contrast(0.05, 0.28, S)
        ph[str(S)] = dict(focus_005=a, focus_01=b, defocus028_005=c005, defocus028_01=c,
                          ratio_focus=b / a, ratio_defocus=c / c005)
        c6 &= 3.5 <= b / a <= 4.5 and c > 5 * b
    ok["C6"] = c6
    ok["C6b"] = (3.5 <= ph["1.0"]["ratio_defocus"] <= 4.5) and (1.8 <= ph["0.75"]["ratio_defocus"] <= 2.2)
    out["phase"] = ph

    for k in ("D1", "D2", "D3", "D3b", "D4", "C1", "C2", "C3", "C4", "C4b", "C5", "C6", "C6b"):
        print(f"{k}: {'PASS' if ok[k] else 'FAIL'}")
    out["pass"] = {k: bool(x) for k, x in ok.items()}
    with open("check_results.json", "w") as fh:
        json.dump(out, fh, default=float, indent=1)
    keys = ["first_zero_um", "fwhm_um", "r50_focus_um", "delta_qw_um", "fwhm_qw_um", "core_defined_up_to_um",
            "r50_qw_um",
            "r50_geo_qw_um", "strehl_qw", "delta_equal_geo_focus", "r50_wave_084", "r50_geo_084",
            "ratio_S1", "ratio_S075"]
    for k in keys:
        print(f"  {k} = {out[k]:.4f}")
    for md in (0.5, 1.0, 3.0, 10.0):
        print(f"  mu d = {md}:", {k: round(float(x), 4) for k, x in out[f'tube_mud_{md}'].items()})
    print("  phase:", {S: {k: round(x, 6) for k, x in rec.items()} for S, rec in ph.items()})
    failed = [k for k, x in ok.items() if not x]
    print("ALL PASS" if not failed else f"FAILED: {failed} (D3, C4, C6 are the original claims; "
          "see D3b, C4b, C6b for the corrected ones)")


if __name__ == "__main__":
    main(CFG)
