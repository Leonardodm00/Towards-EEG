"""Timing probe (not pipeline code): one 'twin' through the fast line renderer
(Block 35, hybrid, the configured 8 x n_psi directions of the inference
config) and the near core-width statistic, per evaluation of a trial table.
Flat tube, half-length 10 um, mu*d = 1, B = 150 gl, no noise, no camera."""
import sys, time, math, dataclasses
import numpy as np
from allen_diameter.config import default_config
from allen_diameter.model import line_fast as LF, geometry as G
from allen_diameter.analysis import profiles as PR
from allen_diameter.analysis.calibration import dip_statistics

cfg0 = default_config()
print("inference.absorption:", cfg0.inference.absorption, "| directions:", cfg0.inference.ray_n_rho, "x", cfg0.inference.ray_n_psi)

def with_table(cfg, th, gamma):
    r = dataclasses.replace(cfg.renderer, kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),
                            kernel_table_sigma_um=(0.080,) + tuple(th), kernel_continuation="linear",
                            kernel_continuation_slope=gamma)
    return dataclasses.replace(cfg, renderer=r)

v = PR.profile_offsets(cfg0.measure); A = PR.n_along(cfg0.measure); step = cfg0.measure.profile_step_um
theta = 0.37
o = np.array([50.03, 40.71]); yh = np.array([-math.sin(theta), math.cos(theta)]); eu = np.array([math.cos(theta), math.sin(theta)])
B = 150.0
truth = (0.1536, 0.3924, 0.5620)
for d in (0.3, 0.5, 0.85):
    tube = G.Tube((o[0] + 0.02, o[1] - 0.01, 0.05), d / 2, 0.0, theta, 1.0, 10.0, "axial")
    mu = 1.0 / d
    for label, ks in (("near k*+-3 (7 planes)", np.arange(-3, 4)), ("far 4<=|k|<=16 (26 planes)", np.r_[np.arange(-16, -3), np.arange(4, 17)])):
        cache = LF.SectionCache()
        times = []
        for j, th in enumerate([truth, (0.16, 0.39, 0.56), (0.15, 0.40, 0.57), (0.15, 0.39, 0.58)]):
            cfg = with_table(cfg0, th, 0.79 + 0.01 * j)
            t0 = time.perf_counter()
            tau = LF.line_profiles(tube, mu, ks, o, yh, eu, v, A, step, cfg, absorption="hybrid", cache=cache)
            om = [dip_statistics(v, B * tau[i], B, cfg0.measure, cfg0.calibration) for i in range(len(ks))]
            times.append(time.perf_counter() - t0)
        st = [s["status"] for s in om]
        print("d %.2f um | %-27s | first call %.2f s, later calls %.3f s (mean of 3) | statuses ok: %d/%d | omega at last table (um^2): %s" % (
            d, label, times[0], float(np.mean(times[1:])), sum(x == "ok" for x in st), len(st),
            np.array2string(np.array([s["omega_um2"] for s in om])[:4], precision=4)))
