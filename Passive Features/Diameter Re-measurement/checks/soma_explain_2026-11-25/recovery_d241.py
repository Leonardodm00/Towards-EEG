"""Recovery check: the soma fit as it stands (far slope held, D-240; region = the disc of Allen's radius,
D-241; crop 14 um, D-243) on a pilot-like simulated soma. Standalone; run: python3 recovery_d241.py OUT.json
Tolerance fixed before running: each of a, b, c within 5 % of the truth and S within 5 %."""
import json, sys, time
import numpy as np
import common as C

s = C.setup()
truth = C.theta_of(C.CONFIG["truth"])
left, top, width, height = s["crop"]
ks = s["ks_all"]
tau = s["make"](ks * C.DZ)(truth, C.SLOPE0)
rng = np.random.default_rng(C.CONFIG["seed"])
block = np.clip(np.rint(C.CONFIG["B_gl"] * tau + rng.normal(0.0, C.CONFIG["noise_gl"], tau.shape)), 0, 255)
mask = np.zeros((height, width), dtype=bool)
t0 = time.time()
fit = C.soma_fit.fit_soma(block, ks, np.ones(ks.size, bool), left, top, C.RES0, C.DZ, C.CONFIG["traced_centre_um"],
                          C.CONFIG["traced_radius_um"], mask, s["make"], C.Q, C.SLOPE0)
S_true = C.soma_fit.ellipsoid_area(*truth[3:6])
out = dict(config=C.CONFIG, settings=dict(far_slope=C.Q.far_slope, region_radius_um=C.CONFIG["traced_radius_um"],
           crop_half_um=s["half"], n_planes=int(fit.ks.size), n_region_px=int(s["region"].sum())),
           truth=truth.tolist(), theta=fit.theta.tolist(), se=fit.se.tolist(), status=fit.status,
           at_bound=list(fit.at_bound), n_eval=fit.n_eval, seconds=time.time() - t0, chi2_dof=fit.chi2 / fit.dof,
           S_true=S_true, S_fit=fit.area_um2, se_S=fit.se_area_um2, r_eq=fit.r_eq_um, message=fit.message)
json.dump(out, open(sys.argv[1], "w"), indent=1)
names = C.soma_fit.PARAMS[:8]
for n, t, f, e in zip(names, truth, fit.theta[:8], fit.se[:8]):
    print("%-10s truth %8.4f  fit %8.4f  +- %.4f  (%+.2f %%)" % (n, t, f, e, 100 * (f - t) / abs(t) if t else 0))
print("S truth %.1f fit %.1f +- %.1f um^2 (%+.2f %%); status %s; at_bound %s; chi2/dof %.2f; %d evals, %.0f s"
      % (S_true, fit.area_um2, fit.se_area_um2, 100 * (fit.area_um2 / S_true - 1), fit.status, fit.at_bound,
         fit.chi2 / fit.dof, fit.n_eval, time.time() - t0))
