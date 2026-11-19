"""Standalone check (NOT pipeline code), 2026-11-19: why the near-bump cut (SPEC Block 39, D-171.3/.4, D-176.1) kept
every sample of node 2669 in the user's rerun (kappa 0.5). The rule on the recorded profiles: on the mean dip over
the planes used, smoothed by one sample (as run), and plane by plane on the raw samples (bump_cut).
Run:  python3 -I bump_rule_2669_check.py <code folder> <cleaning dir> <inference dir>
"""
import json, os, sys, types
import numpy as np
from scipy.signal import find_peaks
WS, CD, ID = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(WS, "src")); sys.path.insert(0, os.path.join(WS, "scripts"))
import infer_nodes as INF
from allen_diameter.analysis import line_weights as LW, node_inputs
rec = json.load(open(CD + "/cleaning_529878215.json")); inf = json.load(open(ID + "/inference_529878215.json"))
info, run = rec["nodes"]["2669"], inf["nodes"]["2669"]
pil = info["pilot"]
pilot = types.SimpleNamespace(**{k: float(pil[k]) for k in ("cx_um", "cy_um", "theta_rad", "phi_rad", "z_sub_um", "d_hat_um", "mu_hat_per_um")})
z = np.load(CD + "/cleaning_529878215_node2669.npz", allow_pickle=False)
node = INF.node_from_npz(2669, z, None)
P, v, B, ks = np.array(z["profiles_after"]), node.v, node.B_gl, node.ks
sigma = float(info["sigma_gl"])
v_t = node_inputs.trace_offset(pilot, node.node_xyz)
c0 = float(run["theta0"][2]) - v_t
fwhm = LW.allen_fwhm_um(float(info["d_allen_um"]), 0.08)
print("sigma %.2f gl, centre0 %.3f um, window (FWHM_A) %.3f um, planes used %s" % (sigma, c0, fwhm, run.get("planes")))
keep, lo, hi = LW.bump_mask(P, B, sigma, v, c0, fwhm, 2.0, 0.5, 1.0, planes=np.array(run["planes"], dtype=bool))
print("as run (mean over planes, smoothed 1): cut lo %s hi %s" % (lo, hi))
Dm = LW.mean_dip(P, B, 1.0, np.array(run["planes"], dtype=bool))
n0 = int(np.argmin(np.abs(v - c0)))
print("mean dip at v = -2.0..0.5 um: " + "  ".join("%.2f:%.0f" % (v[n], Dm[n]) for n in range(v.size) if -2.1 <= v[n] <= 0.6))
for j, k in enumerate(ks):
    D = B[j] - P[j]
    lo_k, hi_k = LW.bump_cut(D, sigma, v, c0, fwhm, 2.0, 0.5)
    left = (v >= -2.2) & (v <= -0.2)
    nmax = np.flatnonzero(left)[np.argmax(D[left] * (v[left] < -1.0))]
    print("k %d: central depth %.0f gl; D at -1.4 um %.0f gl, at -0.9 um %.0f gl; per-plane cut lo %s hi %s"
          % (k, D[np.argmin(np.abs(v - c0))], D[np.argmin(np.abs(v + 1.4))], D[np.argmin(np.abs(v + 0.9))],
             ("%.2f" % lo_k) if np.isfinite(lo_k) else "none", ("%.2f" % hi_k) if np.isfinite(hi_k) else "none"))
# the mean's left side, outwards from the central extreme: where it falls to half depth, and its peaks
near = np.flatnonzero(np.abs(v - c0) <= fwhm); nc = int(near[np.argmax(Dm[near])]); Dc = Dm[nc]
side = Dm[nc::-1]; vs = v[nc::-1]
below = np.flatnonzero(side <= 0.5 * Dc)
print("mean: central depth %.1f gl at v %.2f; falls to half depth (%.1f) first at v %s" % (Dc, v[nc], 0.5 * Dc, ("%.2f" % vs[below[0]]) if below.size else "never"))
if below.size:
    seg = side[below[0]:]
    pk, pr = find_peaks(seg, prominence=0.0)
    print("   left peaks beyond: " + ", ".join("v %.2f D %.1f prom %.2f gl (%.1f sd)" % (vs[below[0] + p], seg[p], q, q / sigma) for p, q in zip(pk, pr["prominences"])))
