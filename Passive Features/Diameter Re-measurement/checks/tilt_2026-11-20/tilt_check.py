"""Check: on the thick nodes of Cell 23A (cell 529878215, uploaded 2026-11-20 18:09), the clearness features of the
measuring line on k* - 1 .. k* + 1, measured against the flat background B_k (as Block 40 does) and against a
straight background L_k(v) = a_k + b_k v fitted by least squares to the line's two ends (|v| >= ends_um).
Usage: python3 -I tilt_check.py <repo src dir> <upload dir> [ends_um]"""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from allen_diameter import config as C
from allen_diameter.analysis import clearness as CL

up = sys.argv[2]
ends = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
q = C.DiameterConfig().clearness
rows = {r["node_id"]: r for r in csv.DictReader(open(os.path.join(up, "gallery", "gallery_529878215.csv")))}
print("clearness: smooth %s samples, window %s um, outside beyond %s W, skew levels %s; E max %s, S max %s"
      % (q.smooth_samples, q.window_um, q.outside_fwhm, tuple(q.skew_levels), q.outside_max, q.skew_max))
print("ends for the straight background: |v| >= %.1f um" % ends)
hdr = "node  Allen  k   | flat B: W(um)  W/A   E     S    | line: W(um)  W/A   E     S    | slope gl/um  wing rms gl  noise gl"
print(hdr)
for nid in sorted(rows, key=lambda s: int(s)):
    r = rows[nid]
    with np.load(os.path.join(up, "cleaning_529878215_node%s.npz" % nid)) as z:
        ks, P, B, v = z["ks"], np.array(z["profiles_before"], float), np.array(z["B_gl"], float), np.array(z["v"], float)
    kst = int(r["k_star"]); A = float(r["d_allen_um"]); sig = float(r["sigma_gl"])
    for dk in (-1, 0, 1):
        i = int(np.flatnonzero(ks == kst + dk)[0]) if np.any(ks == kst + dk) else None
        if i is None:
            continue
        I = P[i]
        fl = CL.dip_features(CL.smooth_dip(B[i] - I, q.smooth_samples), v, 0.0, q.window_um, q.skew_levels, q.outside_fwhm)
        m = (np.abs(v) >= ends) & np.isfinite(I)
        X = np.column_stack([np.ones(m.sum()), v[m]])
        coef, *_ = np.linalg.lstsq(X, I[m], rcond=None)
        L = coef[0] + coef[1] * v
        rms = float(np.sqrt(np.mean((I[m] - X @ coef) ** 2)))
        ln = CL.dip_features(CL.smooth_dip(L - I, q.smooth_samples), v, 0.0, q.window_um, q.skew_levels, q.outside_fwhm)
        f = lambda x: ("%5.2f" % x) if np.isfinite(x) else "  nan"
        print("%-5s %.2f %+d  | %s  %s %s %s | %s  %s %s %s | %+6.2f       %5.1f        %4.1f"
              % (nid, A, dk, f(fl["fwhm_um"]), f(fl["fwhm_um"] / A), f(fl["outside"]), f(fl["skew"]),
                 f(ln["fwhm_um"]), f(ln["fwhm_um"] / A), f(ln["outside"]), f(ln["skew"]), coef[1], rms, sig))
