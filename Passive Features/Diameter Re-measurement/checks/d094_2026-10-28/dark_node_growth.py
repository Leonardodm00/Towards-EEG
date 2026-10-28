"""D-094 check (2026-10-28): how dark calibration nodes move the near-field growth fit (Block 10, Eq. 4).

Question. D-094 removed the optical-depth criterion alpha_hat <= 0.5 from the choice of calibration nodes. In
the faint limit the dip is the kernel applied to the projected absorbance (D-026 (b)). In a darker node the
absorbed fraction 1 - exp(-mu l) saturates in the tube's centre, so the dip's core width changes plane by
plane. Does that change the fitted growth G(delta) and the first-stage kernel
sigma_r(delta) = sqrt(sigma_r(0)^2 + G(delta))?

Model level (the phantoms' truth): flat tubes of diameter d (um) and absorption mu (1/um), optical depth
alpha = mu d; the default renderer (partition along vertical rays, D-024 (i)) with the default kernel table
(ideal Debye, NOT calibrated; sigma_r = 0.122 / 0.438 / 0.603 um at delta = 0.28 / 0.56 / 0.84 um); the camera
chain without noise and without JPEG; the true line; planes k* - 3 .. k* + 3, dz = 0.28 um; axis depth z_ax
uniform in +-0.14 um. Every alpha is rendered on the same geometries (heading, centre, z_ax) and the same
sub-plane depths given to the fit (z_ax + N(0, 0.03 um), as Phase I), so the alphas are compared pairwise.

Computed level: omega_{i,k} (um^2) by calibration.plane_scan; G_hat at the knots 0.28, 0.56, 0.84 um by
calibration.fit_growth (the default convention and interpolation); sigma_hat_r(delta_m) = sqrt(sigma_r(0)^2 +
G_hat_m) with sigma_r(0) = 0.080 um configured.

Hypotheses. The partition renderer stands for the microscope: D-026 (e) found it unreliable for dark THICK
tubes, and for thin dark ones it is not verified. The far-field set (Block 15, offsets to 4.48 um) is not
checked. No noise: the flag faint (a dip under 6 gl) is about noise, so this check says nothing about it.

Run (about 5-10 min on one core):
    cd "Passive Features/Diameter Re-measurement"
    python checks/d094_2026-10-28/dark_node_growth.py > checks/d094_2026-10-28/dark_node_growth.out
The same question under Block 21's hybrid absorption (another model of the dark tube; slower):
    python checks/d094_2026-10-28/dark_node_growth.py --absorption hybrid --diameters 0.3 --alphas 0.1,1,1.5 \
        --nodes 4 > checks/d094_2026-10-28/dark_node_growth_hybrid.out
"""
import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "tests" / "smoke"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_smoke_calibration as T  # noqa: E402  (scan_phantom: Phase I's rendered flat phantom and its scan)
from allen_diameter.analysis import calibration as CAL  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402

SEED = 20261028
DIAMETERS_UM = (0.3, 0.5)                     # the near set reads Allen's 2 r_A <= 0.6 um
ALPHAS = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0)      # 0.5 was the removed cut (node_alpha_max)
N_NODES = 5                                   # nodes per (d, alpha); noise-free, paired geometries


def config(absorption="partition_vertical"):
    base = default_config()
    return dataclasses.replace(base, renderer=dataclasses.replace(base.renderer, noise_sd_gl=0.0, jpeg=False,
                                                                  absorption=absorption))


def geometries(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        theta = rng.uniform(0.0, math.pi)
        cx, cy = rng.uniform(-0.057, 0.057, 2)
        z_ax = rng.uniform(-0.14, 0.14)
        out.append((theta, cx, cy, z_ax, z_ax + rng.normal(0.0, 0.03)))
    return out


def fit_set(cfg, d, alpha, geoms):
    """The growth fit on the rendered phantoms of diameter d and optical depth alpha, one per geometry."""
    mu = alpha / d
    scans, depth0, bad = [], [], 0
    for i, (theta, cx, cy, z_ax, z_sub) in enumerate(geoms):
        rng = np.random.default_rng([SEED, i])          # unused without noise; passed for the renderer's contract
        sc = T.scan_phantom(cfg, d, mu, theta, cx, cy, z_ax, rng)
        bad += sum(1 for s in sc.status if s != "ok")
        scans.append(dataclasses.replace(sc, node_id=i, z_sub_um=z_sub))
        depth0.append(float(np.nanmax(sc.depth_gl)))
    g = CAL.fit_growth(scans, cfg)
    return g, float(np.median(depth0)), bad, scans


def projected_growth(cfg, d, scans, geoms):
    """Phase I's oracle on the same design: the growth fit of ideal Eq. (4) data whose curve is the faint tube's
    expected growth E_eta[sigma_r(delta + eta)^2] - E_eta[sigma_r(eta)^2] (eta: the semicircle law of radius d / 2;
    test_smoke_calibration.semicircle_growth), at the true axis depths."""
    ideal = [dataclasses.replace(sc, omega_um2=0.02 + T.semicircle_growth(z_ax - sc.z_um, 0.5 * d, cfg.renderer),
                                 z_sub_um=z_ax) for sc, (_t, _x, _y, z_ax, _zs) in zip(scans, geoms)]
    return CAL.fit_growth(ideal, cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--absorption", default="partition_vertical", help="RendererConfig.absorption")
    ap.add_argument("--diameters", default=",".join(str(x) for x in DIAMETERS_UM), help="um, comma-separated")
    ap.add_argument("--alphas", default=",".join(str(x) for x in ALPHAS), help="mu d, comma-separated; the first "
                    "is the reference of the last column")
    ap.add_argument("--nodes", type=int, default=N_NODES, help="phantoms per (d, alpha)")
    a = ap.parse_args(argv)
    diameters = [float(x) for x in a.diameters.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    cfg = config(a.absorption)
    print("absorption: %s; noise-free, no JPEG, the true line" % cfg.renderer.absorption)
    rc = cfg.renderer
    knots = np.array([0.28, 0.56, 0.84])
    truth = np.asarray(K.sigma_r(knots, rc), dtype=float)
    s0 = float(rc.sigma_r0_um)
    print("configured kernel (model level): sigma_r(0) = %.3f um; sigma_r(0.28, 0.56, 0.84) = %s um"
          % (s0, np.array2string(truth, precision=3)))
    print("true growth sigma_r^2 - sigma_r(0)^2 at the knots: %s um^2"
          % np.array2string(truth ** 2 - s0 ** 2, precision=4))
    for d in diameters:
        geoms = geometries(a.nodes, SEED + int(round(1000 * d)))
        print("\nd = %.2f um, %d nodes per alpha (same geometries)" % (d, a.nodes))
        print("%6s %8s %10s %10s %10s %9s %9s %9s %9s %10s %5s" % ("alpha", "mu/um", "G(0.28)", "G(0.56)", "G(0.84)",
                                                                    "sr(0.28)", "sr(0.56)", "sr(0.84)", "vs ref",
                                                                    "dip0 gl", "bad"))
        ref = None
        for alpha in alphas:
            t0 = time.perf_counter()
            g, dip0, bad, scans = fit_set(cfg, d, alpha, geoms)
            gk = np.array([float(g.growth(np.array([x]))[0]) for x in knots])
            sr = np.sqrt(np.clip(s0 ** 2 + gk, 0.0, None))
            if ref is None:
                ref, ref_scans = sr, scans
            print("%6.2f %8.3f %10.4f %10.4f %10.4f %9.3f %9.3f %9.3f %8.1f%% %10.1f %5d   (%.0f s)%s"
                  % (alpha, alpha / d, gk[0], gk[1], gk[2], sr[0], sr[1], sr[2], 100.0 * (sr[2] / ref[2] - 1.0), dip0,
                     bad, time.perf_counter() - t0, "" if g.success else "  fit not converged"))
        gp = projected_growth(cfg, d, ref_scans, geoms)
        gpk = np.array([float(gp.growth(np.array([x]))[0]) for x in knots])
        print("%6s %8s %10.4f %10.4f %10.4f %9.3f %9.3f %9.3f   (Phase I's oracle: the projected faint-limit growth "
              "of a tube of this d)" % ("proj", "", gpk[0], gpk[1], gpk[2],
                                        *np.sqrt(np.clip(s0 ** 2 + gpk, 0.0, None))))
        print("  (sr = sigma_hat_r(delta) in um; vs %g = sigma_hat_r(0.84) against alpha = %g's; dip0 = the median "
              "deepest dip over the planes, gl; bad = planes whose core fit is not ok)" % (alphas[0], alphas[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
