"""What the soma fit's data can and cannot fix, by region and by how the planes' brightness is handled.
Standalone check (not pipeline code) for the review discussion of 2026-11-25. Run: python3 sensitivity.py OUT.json

For a simulated soma (truths below; pilot acquisition, the pilot's blur table, far slope held), and for
each fit configuration (region radius; brightness B_k re-solved per plane = today's variable projection, or held
at its true value), it computes at the truth:
  1. the noise SE of c and of S from the Jacobian of the residuals (noise SD 3 gl), Cov = sigma^2 (J^T J)^-1;
  2. the bias of c and S after ONE Gauss-Newton step when the data come from a world the model does not hold:
     a far slope of 0.95 or 0.60 against the held 0.79; an in-focus blur SD of 0.258 um against 0.080 um;
     a camera black level of 10 gl the model does not have.
The bias is linearised (one step from the truth), so it is a sensitivity, not the result of a fit.
"""
import json, sys, dataclasses
import numpy as np
import common as C
from allen_diameter.model.kernel import with_kernel

SIGMA, B0 = 3.0, 110.0                                    # noise SD and background (gl), as in the dry run
TRUTHS = {                                                # (a, b, c) um; 2 mu c = 0.87 kept (the pilot's free-slope fit)
    "silhouette-sized, flat (a 7.0, b 5.6, c 2.5)": (7.0, 5.6, 2.5),
    "silhouette-sized (a 7.0, b 5.6, c 4.5)": (7.0, 5.6, 4.5),
    "the free-slope fit's axes, compact (a 8.08, b 6.52, c 7.26)": (8.08, 6.52, 7.26),
}
CONFIGS = [(5.97, "free"), (5.97, "held"), (10.0, "free"), (10.0, "held"), (13.0, "free")]
STEPS = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01, 0.0])   # um, um, um, um, um, um, rad, (mu: relative, below)
NAMES = ["x0", "y0", "z0", "a", "b", "c", "phi", "mu"]

s = C.setup()
left, top, width, height = s["crop"]
zpl = s["ks_all"][s["idx"]] * C.DZ
xs, ys, zs = C.CONFIG["traced_centre_um"]
R = lambda rcfg: C.SomaRenderer((left, top, width, height), C.RES0, zpl, rcfg, C.Q, C.LIGHT).render   # noqa: E731
# the pilot's own table (decision log D-170: 0.080, 0.1536, 0.3924, 0.5620 um at 0, 0.28, 0.56, 0.84 um; Cell 7),
# not the configuration's default table
PILOT = dataclasses.replace(C.RCFG, kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),
                            kernel_table_sigma_um=(0.0800, 0.1536, 0.3924, 0.5620))
render = R(PILOT)
render_s0 = R(with_kernel(PILOT, 0.258))
P2 = C.RES0 ** 2
k_focus = int(np.argmin(np.abs(zpl - 30.96)))
regions = {r: C.soma_view.disk_mask((height, width), left, top, C.RES0, (xs, ys), r) for r in {c[0] for c in CONFIGS}}


def model(tau, D, reg, mode, B=B0):
    """the model's planes on the region, (K, n_px): B_k tau_k with B_k re-solved against the data D ('free')
    or held at B0 ('held')."""
    t = tau[:, reg]
    if mode == "held":
        return B * t
    Bk = (D[:, reg] * t).sum(1) / (t * t).sum(1)
    return Bk[:, None] * t


def area(th):
    return C.soma_fit.ellipsoid_area(th[3], th[4], th[5])


out = {}
for label, (a, b, c) in TRUTHS.items():
    th0 = np.array([0.6, -0.4, 30.96, a, b, c, 1.13, 0.87 / (2 * c)])
    tau0 = render(th0, C.SLOPE0)
    h = STEPS.copy(); h[7] = 0.02 * th0[7]
    dtau = []
    for i in range(8):
        e = np.zeros(8); e[i] = h[i]
        dtau.append((render(th0 + e, C.SLOPE0), render(th0 - e, C.SLOPE0)))
    worlds = {
        "far slope 0.95 (model 0.79)": B0 * render(th0, 0.95),
        "far slope 0.60 (model 0.79)": B0 * render(th0, 0.60),
        "in-focus SD 0.258 um (model 0.080)": B0 * render_s0(th0, C.SLOPE0),
        "black level 10 gl (model 0)": 10.0 + B0 * tau0,
    }
    B_out = {w: B0 for w in worlds}; B_out["black level 10 gl (model 0)"] = B0 + 10.0   # what is measured outside the soma
    dip = 1.0 - tau0[k_focus]                                   # the in-focus plane's absorption
    outline = dict(half_depth_area_um2=float((dip > 0.5 * dip.max()).sum() * P2), ellipse_area_um2=float(np.pi * a * b))
    gS = np.zeros(8)
    for i in (3, 4, 5):
        e = np.zeros(8); e[i] = 1e-4
        gS[i] = (area(th0 + e) - area(th0 - e)) / 2e-4
    S0 = area(th0)
    rows = {}
    for rr, mode in CONFIGS:
        reg = regions[rr]
        D0 = B0 * tau0
        # Jacobian of the model on the region; with 'free' the brightness is re-solved inside each perturbed model
        J = np.stack([((model(tp, D0, reg, mode) - model(tm, D0, reg, mode)) / (2 * h[i])).ravel()
                      for i, (tp, tm) in enumerate(dtau)], axis=1)
        JTJ = J.T @ J
        cov = SIGMA ** 2 * np.linalg.inv(JTJ)
        se = np.sqrt(np.diag(cov))
        corr = cov / np.outer(se, se)
        row = dict(n_px=int(reg.sum()), se_c_um=float(se[5]), se_c_rel=float(se[5] / c),
                   se_S_rel=float(np.sqrt(gS @ cov @ gS) / S0), se_a_rel=float(se[3] / a), se_b_rel=float(se[4] / b), se_z0_um=float(se[2]),
                   se_muc_rel=float(np.sqrt(np.array([th0[7], th0[5]]) @ cov[np.ix_([5, 7], [5, 7])] @ np.array([th0[7], th0[5]])) / (th0[5] * th0[7])),
                   corr_c_mu=float(corr[5, 7]), corr_c_z0=float(corr[5, 2]), cond=float(np.linalg.cond(J / np.linalg.norm(J, axis=0))),
                   bias={})
        for wname, Dw in worlds.items():
            r0 = (Dw[:, reg] - model(tau0, Dw, reg, mode, B_out[wname])).ravel()
            dth = np.linalg.solve(JTJ, J.T @ r0)
            row["bias"][wname] = dict(dc_rel=float(dth[5] / c), dS_rel=float(gS @ dth / S0), da_rel=float(dth[3] / a), db_rel=float(dth[4] / b), dmuc_rel=float(dth[5] / c + dth[7] / th0[7]),
                                      dz0_um=float(dth[2]), dmu_rel=float(dth[7] / th0[7]),
                                      rms_r0_gl=float(np.sqrt(np.mean(r0 ** 2))))
        rows["R %.2f um, B_k %s" % (rr, mode)] = row
    out[label] = dict(theta=th0.tolist(), S_um2=S0, outline=outline, configs=rows)
    print("\n== %s  (S %.0f um^2)" % (label, S0))
    print("%-22s %7s %9s %9s %8s %8s | bias of c (S), %%: slope .95 | slope .60 | sigma0 .258 | black 10 gl" %
          ("configuration", "n_px", "SE c (%)", "SE S (%)", "r(c,mu)", "r(c,z0)"))
    for k, v in rows.items():
        bb = " | ".join("%+6.1f (%+5.1f)" % (100 * x["dc_rel"], 100 * x["dS_rel"]) for x in v["bias"].values())
        print("%-22s %7d %9.3f %9.3f %8.3f %8.3f | %s" % (k, v["n_px"], 100 * v["se_c_rel"], 100 * v["se_S_rel"],
                                                           v["corr_c_mu"], v["corr_c_z0"], bb))
    print("   in-plane axes and absorbance: SE a, b, mu*c (%) and their largest |bias| over the four worlds (%)")
    for k, v in rows.items():
        mx = lambda key: 100 * max(abs(x[key]) for x in v["bias"].values())   # noqa: E731
        print("   %-22s SE %.3f %.3f %.3f | max bias a %.1f  b %.1f  mu*c %.1f" % (k, 100 * v["se_a_rel"], 100 * v["se_b_rel"],
              100 * v["se_muc_rel"], mx("da_rel"), mx("db_rel"), mx("dmuc_rel")))
    print("   in-focus plane: area inside the half-depth contour %.1f um^2 against pi a b = %.1f um^2 (ratio %.2f)"
          % (outline["half_depth_area_um2"], outline["ellipse_area_um2"], outline["half_depth_area_um2"] / outline["ellipse_area_um2"]))
    sys.stdout.flush()
json.dump(dict(sigma_gl=SIGMA, B0_gl=B0, steps=STEPS.tolist(), results=out), open(sys.argv[1], "w"), indent=1)
