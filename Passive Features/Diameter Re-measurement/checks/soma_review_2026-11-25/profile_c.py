"""Profile of the soma fit's misfit over the depth semi-axis c held, on a simulated soma, with the far slope held at
0.79 in the fit while the simulated world has the far slope GAMMA_TRUE. Standalone check (not pipeline code),
2026-11-25. Run: OMP_NUM_THREADS=1 python3 profile_c.py GAMMA_TRUE OUT.json

World: the pilot's acquisition (35 planes, 0.56 um apart, about the traced depth 32.76 um), the pilot's own blur table
(0.080, 0.1536, 0.3924, 0.5620 um at 0, 0.28, 0.56, 0.84 um; D-170) continued linearly with slope GAMMA_TRUE; an
ellipsoid a 7.0, b 5.6, c 4.5 um, centre (0.6, -0.4, 30.96) um, phi 1.13 rad, 2 mu c = 0.87; background 110 gl;
NO noise (the profile's shape is the systematic part; noise of SD 3 gl adds about 9 gl^2 per pixel to the mean
squared residual at every c). Fit: the region R (disc of 5.97 um about the traced centre (0, 0), no mask), each
plane's brightness B_k by variable projection (soma_fit._vp_columns), the far slope held at 0.79, c held at each
value of C_GRID, the other seven parameters free in Block 46's boxes; then one fit with c free as well.
Custom: a 7-parameter least-squares problem (Block 46's fit_soma has no hold-c option); forward differences with
Block 46's absolute steps, as in soma_fit._Problem.jac. Checked: soma_fit.fit_soma (no option to hold c).
"""
import json, sys, time, dataclasses
import numpy as np
from scipy.optimize import least_squares
import common as C
from allen_diameter.analysis.soma_fit import _vp_columns, ellipsoid_area

GAMMA_TRUE = float(sys.argv[1]); OUT = sys.argv[2]
GAMMA_FIT, B0 = 0.79, 110.0
C_GRID = (4.5, 4.0, 3.5, 3.0, 2.5, 5.0, 5.5, 6.0, 6.5)      # walked from the truth outward (warm starts)
TRUTH = np.array([0.6, -0.4, 30.96, 7.0, 5.6, 4.5, 1.13, 0.87 / 9.0])
PILOT = dataclasses.replace(C.RCFG, kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),
                            kernel_table_sigma_um=(0.0800, 0.1536, 0.3924, 0.5620))
s = C.setup()
left, top, width, height = s["crop"]
zpl = s["ks_all"][s["idx"]] * C.DZ
render = C.SomaRenderer((left, top, width, height), C.RES0, zpl, PILOT, C.Q, C.LIGHT).render
reg = s["region"]
DR = (B0 * render(TRUTH, GAMMA_TRUE))[:, reg]
N = DR.size
xs, ys, zs = C.CONFIG["traced_centre_um"]
LO = np.array([xs - 4, ys - 4, zs - 6, 1.0, 1.0, 1.0, -np.pi, 0.001])
HI = np.array([xs + 4, ys + 4, zs + 6, 15.0, 15.0, 15.0, np.pi, 3.0])
STEP = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.004, 0.0001])
n_render = [0]


def resid_full(th8):
    n_render[0] += 1
    B, r = _vp_columns(DR, render(th8, GAMMA_FIT)[:, reg])
    return r.reshape(-1), B


def problem(free):
    """residual and jacobian over the parameters `free` (indices into theta8), the others at `base`."""
    def unpack(x, base):
        th = base.copy(); th[free] = x; return th
    def make(base):
        memo = {}
        def res(x):
            k = x.tobytes()
            if k not in memo:
                memo.clear(); memo[k] = resid_full(unpack(x, base))[0]
            return memo[k].copy()
        def jac(x):
            r0 = res(x); J = np.empty((r0.size, len(free)), order="F")
            for j, i in enumerate(free):
                h = STEP[i]; xp = x.copy()
                xp[j] = x[j] + h if x[j] + h <= HI[i] else x[j] - h
                J[:, j] = (resid_full(unpack(xp, base))[0] - r0) / (xp[j] - x[j])
            return J
        return res, jac
    return make


def fit(free, base, start):
    res, jac = problem(free)(base)
    t0, n0 = time.time(), n_render[0]
    out = least_squares(res, start[free], jac=jac, bounds=(LO[free], HI[free]), method="trf", x_scale="jac",
                        ftol=1e-6, xtol=1e-6, max_nfev=80)
    th = base.copy(); th[free] = out.x
    r, B = resid_full(th)
    return th, dict(ssr=float(r @ r), rms_gl=float(np.sqrt(r @ r / N)), B=B.tolist(), status=int(out.status),
                    message=out.message, nfev=int(out.nfev), renders=n_render[0] - n0, seconds=time.time() - t0)


rows, prev = [], {}
FREE7 = [0, 1, 2, 3, 4, 6, 7]
for c in C_GRID:
    start = TRUTH.copy() if c == 4.5 else prev[4.0 if c < 4.5 else 4.5].copy() if False else None
    # warm start from the nearest c already fitted on the same side (the truth for the first)
    done = [r["c"] for r in rows]
    near = min(done, key=lambda v: abs(v - c)) if done else None
    start = TRUTH.copy() if near is None else np.array(prev[near])
    start[5] = c
    th, info = fit(FREE7, start, start)
    info.update(c=c, theta=th.tolist(), S_um2=ellipsoid_area(th[3], th[4], th[5]))
    rows.append(info); prev[c] = th.tolist()
    print("c %.2f  rms %.4f gl  a %.4f b %.4f z0 %.4f mu %.5f mu*c %.4f  S %.1f  status %d  %d renders %.0f s"
          % (c, info["rms_gl"], th[3], th[4], th[2], th[7], th[7] * c, info["S_um2"], info["status"], info["renders"],
             info["seconds"]), flush=True)
best = min(rows, key=lambda r: r["ssr"])
th, info = fit(list(range(8)), np.array(best["theta"]), np.array(best["theta"]))
info.update(theta=th.tolist(), S_um2=ellipsoid_area(th[3], th[4], th[5]))
print("c free: c %.4f  rms %.4f gl  a %.4f b %.4f  S %.1f  status %d" % (th[5], info["rms_gl"], th[3], th[4],
      info["S_um2"], info["status"]), flush=True)
json.dump(dict(gamma_true=GAMMA_TRUE, gamma_fit=GAMMA_FIT, B0=B0, truth=TRUTH.tolist(),
               S_true=ellipsoid_area(*TRUTH[3:6]), n_pixels=N, profile=sorted(rows, key=lambda r: r["c"]),
               c_free=info), open(OUT, "w"), indent=1)
