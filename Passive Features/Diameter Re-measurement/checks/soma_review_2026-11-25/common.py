"""Shared setup of the soma-fit explanation checks (standalone; not pipeline code).

A pilot-like simulated soma rendered with the pipeline's own renderer (Block 45) under the default
configuration (kernel table "fixed", far slope 0.79 held, D-240), on a 14 um half-width crop (D-243).
All numbers of the simulated truth are in CONFIG; nothing here reads the user's data.
"""
import os, sys
import numpy as np

W = os.environ.get("TEEG_W", "/home/claude/towards-eeg/Passive Features/Diameter Re-measurement")
sys.path.insert(0, os.path.join(W, "src"))
from allen_diameter.config import default_config            # noqa: E402
from allen_diameter.model.soma_render import SomaRenderer   # noqa: E402
from allen_diameter.analysis import soma_fit, soma_view     # noqa: E402

CONFIG = dict(
    traced_centre_um=(0.0, 0.0, 32.76),   # Allen's traced soma (pilot: depth 32.76 um; x, y set to 0 here)
    traced_radius_um=5.97,                # Allen's soma radius on the pilot
    truth=dict(x0=0.6, y0=-0.4, z0=30.96, a=7.0, b=5.6, c=5.0, phi=1.13, mu=0.087),  # simulated truth (chosen)
    B_gl=110.0, noise_gl=3.0, seed=529,   # background grey level and noise SD (the dry run's values)
)
cfg = default_config()
RCFG, Q = cfg.renderer, cfg.soma_fit
RES0, DZ, LIGHT = float(cfg.acquisition.res0_um), float(cfg.acquisition.dz_um), int(cfg.acquisition.light_direction)
SLOPE0 = float(RCFG.kernel_continuation_slope)


def theta_of(d):
    return np.array([d["x0"], d["y0"], d["z0"], d["a"], d["b"], d["c"], d["phi"], d["mu"]], dtype=float)


def setup(plane_step=None):
    """-> dict(crop, ks, z, region, make): the crop (D-243), the fitted planes (Block 46's rule), the region
    (D-241: the disc of the traced radius; no branches in this simulation, so no mask)."""
    xs, ys, zs = CONFIG["traced_centre_um"]
    r = CONFIG["traced_radius_um"]
    half = max(float(Q.render_half_um), r + float(Q.crop_margin_um))
    left, top, width, height = soma_view.crop_box((xs, ys), half, RES0)
    k_lo, k_hi = soma_view.plane_range(zs, float(Q.plane_half_um), DZ)
    ks_all = np.arange(k_lo, k_hi + 1)
    step = Q.plane_step if plane_step is None else plane_step
    idx = soma_fit.select_planes(ks_all, np.ones(ks_all.size, bool), zs, DZ, Q.plane_half_um, step)
    region = soma_view.disk_mask((height, width), left, top, RES0, (xs, ys), r)
    make = lambda z: SomaRenderer((left, top, width, height), RES0, z, RCFG, Q, LIGHT).render   # noqa: E731
    return dict(crop=(left, top, width, height), ks_all=ks_all, idx=idx, region=region, make=make, half=half)
