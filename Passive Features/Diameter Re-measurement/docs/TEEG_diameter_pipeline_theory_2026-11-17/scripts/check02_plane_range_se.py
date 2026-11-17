"""Check 2 of TEEG_diameter_pipeline_theory_2026-11-17.md (python3 check02_plane_range_se.py, about 2 min): standard errors of the joint full-model least squares (all planes together) as the plane
range widens. Flat tube, hybrid, pilot kernel, gamma 0.79, white noise 1 gl, B 92 gl, unweighted; J by central
differences of the whole shadow S_k(v_n | theta). Planes k* - H .. k* + H around plane 0."""
import numpy as np
from _common import load_check
chk = load_check()
cfg, cache = chk.make_cfg(0.79), chk.LF.SectionCache()
p = chk.default_config().acquisition.res0_um
HMAX = 8
ks = np.arange(-HMAX, HMAX + 1)
geom = (np.array([437 * p, 211 * p]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.arange(-26, 27) * 0.1144, ks)
for name, d, mu, z in chk.NODES:
    th = np.array([d, mu, 0.0, z]); steps = np.array([0.02, 0.02 * mu, 0.02, 0.02])
    J = []
    for a in range(4):
        e = np.zeros(4); e[a] = steps[a]
        J.append((chk.shadow(th + e, cfg, None, True, geom, cache) - chk.shadow(th - e, cfg, None, True, geom, cache)) / (2 * steps[a]))
    J = np.array(J)                                   # (4, n_planes, n_v)
    print("node %s d %.3f um mu %.4f /um, tube spans %+.2f..%+.2f um about plane 0" % (name, d, mu, z - d / 2, z + d / 2))
    for H in (4, 6, 8):
        sel = np.abs(ks) <= H
        Jm = J[:, sel, :].reshape(4, -1).T
        cov = np.linalg.inv(Jm.T @ Jm) * (chk.NOISE_GL / chk.B_GL) ** 2
        se = np.sqrt(np.diag(cov)); corr = cov[0, 1] / (se[0] * se[1])
        print("  planes +-%d (+-%.2f um): SE d %.4f um (%.2f %%), mu %.5f /um (%.2f %%), z_ax %.4f um; corr(d, mu) %+.3f"
              % (H, H * 0.28, se[0], 100 * se[0] / d, se[1], 100 * se[1] / mu, se[3], corr))
