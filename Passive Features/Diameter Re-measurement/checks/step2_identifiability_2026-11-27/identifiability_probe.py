"""Standalone check (not pipeline code), 2026-11-27.
Question: in step 2's near stage, how strongly are a twin's absorption mu_i and axis depth z_ax,i tied
to each other and to the shared table, and what does adding the in-focus width sigma_r(0) to the fit cost?
Proxy forward model: the fast line renderer (Block 35, hybrid, 8 x 16 directions) + dip_statistics, noise-free,
the table read LINEARLY between knots (D-181's spline is not built). NOT the full stack the user chose.
Method: finite-difference Jacobian of the near core widths omega_{i,k} (k = k*-3..k*+3) with respect to
(ln sigma0, ln theta_1..3 shared; z_i and ln mu_i per twin); for iid unit noise on omega, Cov = (J^T J)^-1.
"""
import math, dataclasses, sys
import numpy as np
from allen_diameter.config import default_config
from allen_diameter.model import line_fast as LF, geometry as G
from allen_diameter.analysis import profiles as PR
from allen_diameter.analysis.calibration import dip_statistics

CFG = dict(seed=20261127, n_twins=12, d_range_um=(0.20, 0.60), mud_range=(0.3, 1.5), z_half_um=0.14,
           table=(0.080, 0.1536, 0.3924, 0.5620), gamma=0.79, B_gl=150.0, rel_step=0.02, z_step_um=0.02)
c0 = default_config()
v = PR.profile_offsets(c0.measure); A = PR.n_along(c0.measure); step = c0.measure.profile_step_um
th0 = 0.37; o = np.array([50.03, 40.71]); yh = np.array([-math.sin(th0), math.cos(th0)]); eu = np.array([math.cos(th0), math.sin(th0)])
ks = np.arange(-3, 4)

def cfg_for(table, gamma):
    r = dataclasses.replace(c0.renderer, kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84), kernel_table_sigma_um=tuple(table),
                            sigma_r0_um=table[0], kernel_continuation="linear", kernel_continuation_slope=gamma)
    return dataclasses.replace(c0, renderer=r)

def omega(d, mu, z, table):
    tube = G.Tube((o[0], o[1], z), d / 2, 0.0, th0, 1.0, 10.0, "axial")
    tau = LF.line_profiles(tube, mu, ks, o, yh, eu, v, A, step, cfg_for(table, CFG["gamma"]), absorption="hybrid")
    st = [dip_statistics(v, CFG["B_gl"] * t, CFG["B_gl"], c0.measure, c0.calibration) for t in tau]
    assert all(s["status"] == "ok" for s in st), [s["status"] for s in st]
    return np.array([s["omega_um2"] for s in st] + [s["area_gl_um"] for s in st])

rng = np.random.default_rng(CFG["seed"])
N = CFG["n_twins"]
d = rng.uniform(*CFG["d_range_um"], N); mud = rng.uniform(*CFG["mud_range"], N); mu = mud / d
z = rng.uniform(-CFG["z_half_um"], CFG["z_half_um"], N)
T = np.array(CFG["table"]); h = CFG["rel_step"]; hz = CFG["z_step_um"]
K = ks.size
K2 = 2 * K
Jsh = np.zeros((N * K2, 4)); Jz = np.zeros((N * K2, N)); Jmu = np.zeros((N * K2, N)); Jd = np.zeros(N * K2); base0 = np.zeros(N * K2)
for i in range(N):
    rows = slice(i * K2, (i + 1) * K2)
    base0[rows] = omega(d[i], mu[i], z[i], T)
    Jd[rows] = (omega(d[i] * (1 + h), mu[i], z[i], T) - omega(d[i] * (1 - h), mu[i], z[i], T)) / (math.log(1 + h) - math.log(1 - h))
    for m in range(4):
        tp, tm = T.copy(), T.copy(); tp[m] *= (1 + h); tm[m] *= (1 - h)
        Jsh[rows, m] = (omega(d[i], mu[i], z[i], tp) - omega(d[i], mu[i], z[i], tm)) / (math.log(1 + h) - math.log(1 - h))
    Jz[rows, i] = (omega(d[i], mu[i], z[i] + hz, T) - omega(d[i], mu[i], z[i] - hz, T)) / (2 * hz)
    Jmu[rows, i] = (omega(d[i], mu[i] * (1 + h), z[i], T) - omega(d[i], mu[i] * (1 - h), z[i], T)) / (math.log(1 + h) - math.log(1 - h))
    print("twin %2d  d %.2f um  mu*d %.2f  z %+.3f um" % (i, d[i], mud[i], z[i]), flush=True)

isw = np.tile(np.r_[np.ones(K, bool), np.zeros(K, bool)], N)     # omega rows; the others are area rows
FULL = dict(Jsh=Jsh, Jz=Jz, Jmu=Jmu, Jd=Jd)
Jsh, Jz, Jmu = Jsh[isw], Jz[isw], Jmu[isw]
def cov(J):
    return np.linalg.inv(J.T @ J)
def cond_scaled(J):
    return np.linalg.cond(J / np.linalg.norm(J, axis=0))

print("\n(1) per twin: size and likeness of the two per-twin patterns in omega (um^2)")
print("    twin   d    mu*d | max|d omega| for +10%% mu | for +0.05 um depth | cosine(mu pattern, depth pattern)")
cosv = []
for i in range(N):
    a = Jmu[i * K:(i + 1) * K, i]; b = Jz[i * K:(i + 1) * K, i]
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))); cosv.append(c)
    print("    %2d   %.2f  %.2f |        %.4f           |      %.4f        |  %+.2f" % (i, d[i], mud[i], np.max(np.abs(a)) * math.log(1.1), np.max(np.abs(b)) * 0.05, c))
print("    cosine: median %+.2f, range %+.2f to %+.2f" % (np.median(cosv), min(cosv), max(cosv)))

sets = {"knots only": np.c_[Jsh[:, 1:]],
        "knots + depth_i": np.c_[Jsh[:, 1:], Jz],
        "knots + depth_i + mu_i": np.c_[Jsh[:, 1:], Jz, Jmu],
        "sigma0 + knots + depth_i": np.c_[Jsh, Jz],
        "sigma0 + knots + depth_i + mu_i": np.c_[Jsh, Jz, Jmu]}
base = np.sqrt(np.diag(cov(sets["knots only"])))
print("\n(2) SE of ln(theta_1), ln(theta_2), ln(theta_3) relative to 'knots only' (iid noise on omega), and the scaled condition number")
for name, J in sets.items():
    C = cov(J); off = 1 if name.startswith("sigma0") else 0
    se = np.sqrt(np.diag(C))
    line = "    %-34s SE ratio %5.2f %5.2f %5.2f | cond %8.1f" % (name, *(se[off:off + 3] / base), cond_scaled(J))
    R = C / np.outer(se, se)
    if off:
        line += " | SE(ln sigma0)/SE(ln theta_1) %.1f, corr(sigma0, theta_1) %+.2f" % (se[0] / se[1], R[0, 1])
    if name.endswith("mu_i"):
        iz = off + 3 + np.arange(N); im = iz + N
        cz = np.array([R[iz[i], im[i]] for i in range(N)])
        line += " | corr(depth_i, mu_i): median %+.2f, range %+.2f to %+.2f; SE(ln mu_i)/SE(ln theta_1): median %.0f" % (
            np.median(cz), cz.min(), cz.max(), np.median(se[im] / se[off]))
        if off:
            line += "; corr(sigma0, mu_i) median %+.2f" % np.median(R[0, im])
    print(line)

print("\n(3) the same with the dip's area added (omega AND area on k*-3..k*+3), residuals RELATIVE (each divided by its value), equal relative noise assumed")
W = 1.0 / base0
F = {k: (v.T * W).T for k, v in FULL.items()}
sets3 = {"knots + depth_i + mu_i, omega only": (np.c_[F["Jsh"][:, 1:], F["Jz"], F["Jmu"]][isw], 0),
         "knots + depth_i + mu_i, omega + area": (np.c_[F["Jsh"][:, 1:], F["Jz"], F["Jmu"]], 0),
         "sigma0 + knots + depth_i + mu_i, omega + area": (np.c_[F["Jsh"], F["Jz"], F["Jmu"]], 1)}
ref = np.sqrt(np.diag(cov(F["Jsh"][:, 1:][isw])))
for name, (J, off) in sets3.items():
    C = cov(J); se = np.sqrt(np.diag(C)); R = C / np.outer(se, se)
    iz = off + 3 + np.arange(N); im = iz + N
    cz = np.array([R[iz[i], im[i]] for i in range(N)])
    extra = (" | SE(ln sigma0)/SE(ln theta_1) %.1f, corr(sigma0, theta_1) %+.2f" % (se[0] / se[1], R[0, 1])) if off else ""
    print("    %-46s knots' SE ratio %5.2f %5.2f %5.2f | cond %7.1f | corr(depth_i, mu_i) median %+.2f | SE(ln mu_i)/SE(ln theta_1) median %.1f%s" % (
        name, *(se[off:off + 3] / ref), cond_scaled(J), np.median(cz), np.median(se[im] / se[off]), extra))

print("\n(4) first-order bias if every twin's diameter is 10 % too large (truth narrower), omega only, iid noise; percent")
dv = -FULL["Jd"][isw] * math.log(1.10)        # data minus model at the true table when the twins are 10 % too wide
for name in ("knots + depth_i", "sigma0 + knots + depth_i"):
    J = sets[name]; b = np.linalg.lstsq(J, dv, rcond=None)[0]; off = 1 if name.startswith("sigma0") else 0
    txt = ("sigma0 %+.1f, " % (100 * b[0])) if off else ""
    print("    %-28s %stheta_1 %+.1f, theta_2 %+.1f, theta_3 %+.1f" % (name, txt, *(100 * b[off:off + 3])))
