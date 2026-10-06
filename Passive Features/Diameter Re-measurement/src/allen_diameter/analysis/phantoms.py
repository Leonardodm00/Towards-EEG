"""Phantoms and one replicate of the bias table -- Block 6 in specs/SPEC.md
(procedure s.3.5, s.3.7; D-024).

Replicate n draws, in a fixed order, from numpy.random.default_rng([seed, n]):
d, phi, theta, mu, the node's lateral offset from the pixel centre, the axis
depth about plane 0, and the per-node jitter of the phantom "SWC"; the same
generator then feeds the camera noise. The draw of replicate n is therefore
the same whatever the chunking of the run.

The phantom is a Block 2 tube; its branch carries nodes every
phantom_node_step_um along the axis (radius d/2, the mask of D-018.1 around
the known axis), and the middle node is measured by the per-node chain of
Block 5 -- the same code as on real nodes (procedure s.3.7).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace

import numpy as np

from ..model import geometry, render
from .node_pipeline import Branch, measure_node

SELECTION_FLAGS = ("faint", "crossing", "stack_edge", "dark", "bbar_few", "profile_nan")


@dataclass(frozen=True)
class Draw:
    index: int
    seed: int
    d_um: float
    phi_rad: float
    theta_rad: float
    mu_per_um: float
    cx_um: float
    cy_um: float
    cz_um: float


def _draw(rng, cfg, index):
    ph, acq = cfg.phantom, cfg.acquisition
    if ph.design == "random":
        lo, hi = ph.d_range_um
        d = math.exp(rng.uniform(math.log(lo), math.log(hi))) if ph.d_log_uniform else rng.uniform(lo, hi)
        phi = math.radians(rng.uniform(*ph.phi_range_deg))
    elif ph.design == "grid":
        cell = index % (len(ph.grid_d_um) * len(ph.grid_phi_deg))
        d = float(ph.grid_d_um[cell // len(ph.grid_phi_deg)])
        phi = math.radians(ph.grid_phi_deg[cell % len(ph.grid_phi_deg)])
    else:
        raise ValueError("unknown phantom design %r" % (ph.design,))
    theta = rng.uniform(*ph.theta_range_rad)
    lo, hi = ph.mu_range_per_um
    mu = math.exp(rng.uniform(math.log(lo), math.log(hi))) if (ph.mu_log_uniform and lo > 0) else rng.uniform(lo, hi)
    p = acq.res0_um
    off = rng.uniform(-0.5 * p, 0.5 * p, 2) if ph.subpixel_offset else np.zeros(2)
    cz = rng.uniform(-0.5 * acq.dz_um, 0.5 * acq.dz_um) if ph.axis_depth_jitter else 0.0
    return d, phi, theta, mu, float(off[0]), float(off[1]), float(cz)


def draw_replicate(cfg, seed, index):
    """(Draw, generator positioned after the draws) of replicate index."""
    rng = np.random.default_rng([int(seed), int(index)])
    d, phi, theta, mu, cx, cy, cz = _draw(rng, cfg, int(index))
    return Draw(int(index), int(seed), d, phi, theta, mu, cx, cy, cz), rng


def phantom_tube(draw, cfg):
    r = cfg.renderer
    return geometry.Tube((draw.cx_um, draw.cy_um, draw.cz_um), 0.5 * draw.d_um, draw.phi_rad, draw.theta_rad,
                         r.cross_section_aspect, r.U_um, r.end_cut)


def phantom_branch(draw, cfg, rng):
    """Nodes every phantom_node_step_um along the axis through c, nodes_each_way on each side,
    plus the configured jitter; radius d/2. The middle node is index nodes_each_way."""
    ph = cfg.phantom
    t = geometry.axis_direction(draw.phi_rad, draw.theta_rad)
    s = np.arange(-ph.nodes_each_way, ph.nodes_each_way + 1) * ph.phantom_node_step_um
    xyz = np.array([draw.cx_um, draw.cy_um, draw.cz_um])[None, :] + s[:, None] * t[None, :]
    if ph.jitter_xy_um > 0 or ph.jitter_z_um > 0:
        xyz = xyz + rng.normal(0.0, 1.0, xyz.shape) * np.array([ph.jitter_xy_um, ph.jitter_xy_um, ph.jitter_z_um])
    return Branch.from_points(xyz, 0.5 * draw.d_um)


def reject_reasons(result):
    """The criteria of the selection S a NodeResult fails (empty: retained). A
    registration verdict other than ON fails S (real nodes; phantoms have none)."""
    out = [] if result.fit_status == "converged" else ["status:" + result.fit_status]
    if result.reg_verdict and result.reg_verdict != "ON":
        out.append("registration:" + result.reg_verdict)
    return out + [f for f in SELECTION_FLAGS if f in result.flags]


def run_replicate(cfg, seed, index, backend=None, d_override_um=None):
    """Render and measure replicate index; returns a flat dict (one CSV row): the
    draw (true d, phi, theta, mu, node centre), the chain's outputs (meas_* for
    the measured centre and angles), flags, in_S, reject, ratio = d_hat / d and
    the wall time. The table is indexed by the TRUE (d, phi) (procedure Eq. 2)."""
    t0 = time.perf_counter()
    draw, rng = draw_replicate(cfg, seed, index)
    if d_override_um is not None:      # end-to-end checks at fixed diameters (scripts/end_to_end.py)
        draw = replace(draw, d_um=float(d_override_um))
    tube = phantom_tube(draw, cfg)
    branch = phantom_branch(draw, cfg, rng)
    pad_px = int(math.ceil(cfg.renderer.pad_um / cfg.acquisition.res0_um))

    def provider(left, top, width, height, k_lo, k_hi):
        return render.synthetic_block(tube, draw.mu_per_um, np.arange(k_lo, k_hi + 1), left, top, width, height,
                                      cfg, rng, pad_px, backend=backend)

    res = measure_node(branch, cfg.phantom.nodes_each_way, provider, cfg)
    reasons = reject_reasons(res)
    row = asdict(draw)
    for name in ("k_star", "z_sub_um", "cx_um", "cy_um", "cz_um", "theta_rad", "phi_rad", "steep", "vertical",
                 "B_bar", "d_hat_um", "mu_hat_per_um", "v0_hat_um", "alpha_hat", "fit_status"):
        row[("meas_" if name in ("cx_um", "cy_um", "cz_um", "theta_rad", "phi_rad") else "") + name] = getattr(res, name)
    row.update(flags=";".join(res.flags), in_S=not reasons, reject=";".join(reasons),
               ratio=res.d_hat_um / draw.d_um, seconds=time.perf_counter() - t0)
    return row
