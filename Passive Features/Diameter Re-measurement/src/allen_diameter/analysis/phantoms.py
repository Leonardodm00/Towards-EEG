"""Phantoms and one realization of the bias table's design -- Blocks 6 and
14.2 in specs/SPEC.md (procedure s.3.5, s.3.7; D-024, D-058, D-059, D-062).

The design (D-059, D-062): G independent draws g = 1..G of a cylinder's
features (d, phi, theta, mu), each rendered and measured n_real times,
realization r = 1..n_real with its own nuisances. Generators (D-062's keys):

  features of draw g          numpy.random.default_rng([seed, g]): d, phi,
                              theta, mu, in this order
  nuisances of realization r  default_rng([seed, g, r]): the lateral offset
                              (cx, cy) within one pixel, the axis depth cz
                              within +-dz/2, then the branch jitter, then the
                              camera noise (the renderer)
  its branch radius           default_rng([seed, g, r, 1]): eps ~ N(0, sd^2)
                              drawn until d/2 + eps > 0 (D-062.1)

so a draw and each of its realizations are the same whatever the chunking
and the order of the run. Indices start at 1: NumPy's SeedSequence pads a
key with zeros, so default_rng([s, g, 0]) is default_rng([s, g]) and
default_rng([s, 0]) is default_rng(s) (checked, NumPy 2.5.3) -- a
realization 0 would replay its draw's features.

The phantom is a Block 2 tube of radius d/2; its branch carries nodes every
phantom_node_step_um along the axis with the realization's radius r_br =
d/2 + eps at every node, so that one radius reaches the four places a real
node's Allen radius reaches (D-062.1); the middle node is measured by the
per-node chain -- the same code as on real nodes (procedure s.3.7, D-057
(iii)).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from ..config import SELECTION_FLAGS  # noqa: F401  (re-exported here, its old home)
from ..model import geometry, render
from .node_pipeline import Branch, measure_node
from .registration import registration_category

# SELECTION_FLAGS: the flags that fail the selection S (Block 6), defined in config.py since D-096 so that they enter
# the estimator signature. "dark" is a label only since Block 14.1 (D-030), "faint" since D-096 (2026-10-28): such a
# node is retained, inverted and keeps the label.
MAX_RADIUS_REDRAWS = 100000      # a guard: at sd 0.2 um and d 0.2 um a draw is rejected with probability 0.31


@dataclass(frozen=True)
class Draw:
    """One realization (g, r) of the design: the draw's features and the
    realization's nuisances (the camera noise is drawn while rendering)."""

    draw_index: int          # g >= 1
    realization_index: int   # r >= 1
    seed: int
    d_um: float
    phi_rad: float
    theta_rad: float
    mu_per_um: float
    cx_um: float
    cy_um: float
    cz_um: float
    radius_um: float         # the branch radius r_br = d/2 + eps (D-062.1)
    radius_redraws: int      # draws of eps rejected because d/2 + eps <= 0


def check_index(name, value):
    """value as an int >= 1; ValueError otherwise (see the module docstring)."""
    if isinstance(value, bool) or not float(value).is_integer() or int(value) < 1:
        raise ValueError("%s must be an integer >= 1 (NumPy pads a seed key with zeros, so index 0 would replay "
                         "another stream), got %r" % (name, value))
    return int(value)


def draw_features(cfg, seed, g):
    """(d um, phi rad, theta rad, mu 1/um) of draw g from default_rng([seed, g]): the laws of D-024 (iii) (random
    design), or cell (g - 1) mod (n_d n_phi) of the grid design with theta and mu drawn."""
    ph = cfg.phantom
    g = check_index("draw index g", g)
    rng = np.random.default_rng([int(seed), g])
    if ph.design == "random":
        lo, hi = ph.d_range_um
        d = math.exp(rng.uniform(math.log(lo), math.log(hi))) if ph.d_log_uniform else rng.uniform(lo, hi)
        phi = math.radians(rng.uniform(*ph.phi_range_deg))
    elif ph.design == "grid":
        cell = (g - 1) % (len(ph.grid_d_um) * len(ph.grid_phi_deg))
        d = float(ph.grid_d_um[cell // len(ph.grid_phi_deg)])
        phi = math.radians(ph.grid_phi_deg[cell % len(ph.grid_phi_deg)])
    else:
        raise ValueError("unknown phantom design %r" % (ph.design,))
    theta = rng.uniform(*ph.theta_range_rad)
    lo, hi = ph.mu_range_per_um
    mu = math.exp(rng.uniform(math.log(lo), math.log(hi))) if (ph.mu_log_uniform and lo > 0) else rng.uniform(lo, hi)
    return float(d), float(phi), float(theta), float(mu)


def branch_radius(cfg, seed, g, r, d_um):
    """(r_br um, rejected draws): d/2 + eps, eps ~ N(0, radius_noise_sd_um^2) from default_rng([seed, g, r, 1]),
    drawn until positive (D-062.1); d/2 and 0 when the SD is 0."""
    sd = float(cfg.phantom.radius_noise_sd_um)
    g, r = check_index("draw index g", g), check_index("realization index r", r)
    if not sd > 0:
        return 0.5 * float(d_um), 0
    rng = np.random.default_rng([int(seed), g, r, 1])
    for redraws in range(MAX_RADIUS_REDRAWS):
        rb = 0.5 * float(d_um) + sd * float(rng.standard_normal())
        if rb > 0:
            return rb, redraws
    raise ValueError("branch_radius: no positive radius in %d draws (d %g um, sd %g um)" % (MAX_RADIUS_REDRAWS, d_um, sd))


def draw_replicate(cfg, seed, g, r, d_override_um=None):
    """(Draw, generator) of realization r of draw g: the generator is the realization's nuisance stream
    default_rng([seed, g, r]) positioned after cx, cy, cz, so the branch jitter and then the camera noise follow.
    d_override_um replaces the drawn d (end-to-end checks at fixed diameters); the radius is drawn about it."""
    acq, ph = cfg.acquisition, cfg.phantom
    g, r = check_index("draw index g", g), check_index("realization index r", r)
    d, phi, theta, mu = draw_features(cfg, seed, g)
    if d_override_um is not None:
        d = float(d_override_um)
    rng = np.random.default_rng([int(seed), g, r])
    p = acq.res0_um
    off = rng.uniform(-0.5 * p, 0.5 * p, 2) if ph.subpixel_offset else np.zeros(2)
    cz = rng.uniform(-0.5 * acq.dz_um, 0.5 * acq.dz_um) if ph.axis_depth_jitter else 0.0
    rb, redraws = branch_radius(cfg, seed, g, r, d)
    return Draw(g, r, int(seed), d, phi, theta, mu, float(off[0]), float(off[1]), float(cz), rb, redraws), rng


def phantom_tube(draw, cfg):
    r = cfg.renderer
    return geometry.Tube((draw.cx_um, draw.cy_um, draw.cz_um), 0.5 * draw.d_um, draw.phi_rad, draw.theta_rad,
                         r.cross_section_aspect, r.U_um, r.end_cut)


def phantom_branch(draw, cfg, rng):
    """Nodes every phantom_node_step_um along the axis through c, nodes_each_way on each side,
    plus the configured jitter (from rng); radius draw.radius_um at every node (D-062.1: the one
    radius of the four places). The middle node is index nodes_each_way."""
    ph = cfg.phantom
    t = geometry.axis_direction(draw.phi_rad, draw.theta_rad)
    s = np.arange(-ph.nodes_each_way, ph.nodes_each_way + 1) * ph.phantom_node_step_um
    xyz = np.array([draw.cx_um, draw.cy_um, draw.cz_um])[None, :] + s[:, None] * t[None, :]
    if ph.jitter_xy_um > 0 or ph.jitter_z_um > 0:
        xyz = xyz + rng.normal(0.0, 1.0, xyz.shape) * np.array([ph.jitter_xy_um, ph.jitter_xy_um, ph.jitter_z_um])
    return Branch.from_points(xyz, draw.radius_um)


def reject_reasons(result):
    """The criteria of the selection S a NodeResult fails (empty: retained). A
    registration category other than ON fails S (real nodes; phantoms have no
    verdict); the category is read from the registration_check sentence by
    registration.registration_category."""
    out = [] if result.fit_status == "converged" else ["status:" + result.fit_status]
    cat = registration_category(result.reg_verdict)
    if cat and cat != "ON":
        out.append("registration:" + cat)
    return out + [f for f in SELECTION_FLAGS if f in result.flags]


def _cell_text(value):
    """A stored CSV value as text: '' for None and NaN (an empty cell read back as a float)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def row_reject_reasons(row, flags=None):
    """reject_reasons on a stored row -- Block 8's per-node CSV or Block 6's
    replicate CSV, as table_io.read_rows gives it -- from its own columns
    fit_status, reg_verdict and flags (';'-joined), under the flags of S of
    the code that reads it (SELECTION_FLAGS, or `flags`), not the stored
    in_S and reject, which carry the rule of the code that wrote them
    (D-098.3: a CSV written before D-096 lists `faint` among its reasons).
    KeyError when the row has no fit_status or flags column."""
    if "fit_status" not in row or "flags" not in row:
        raise KeyError("row_reject_reasons: the row has no fit_status or flags column")
    status = _cell_text(row["fit_status"])
    out = [] if status == "converged" else ["status:" + status]
    cat = registration_category(_cell_text(row.get("reg_verdict")))
    if cat and cat != "ON":
        out.append("registration:" + cat)
    have = {f for f in _cell_text(row["flags"]).split(";") if f}
    use = SELECTION_FLAGS if flags is None else tuple(flags)
    return out + [f for f in use if f in have]


def run_replicate(cfg, seed, g, r, backend=None, d_override_um=None):
    """Render and measure realization r of draw g (Block 14.2); returns a flat dict (one CSV row): the
    Draw (draw_index, realization_index, seed, true d, phi, theta, mu, node centre, radius_um,
    radius_redraws), the chain's outputs (meas_* for
    the measured centre and angles; since Block 13 the deliverable's fit at
    k_iter, the iteration's k_iter, iter_status, n_rounds, n_restacks, the SGT
    fit's w, beta, xi, zeta, irls_ok and at_bound, and the pilot's d_pilot_um,
    alpha_pilot, pilot_status), flags, in_S, reject, ratio = d_hat / d and the
    wall time. The table is indexed by the TRUE (d, phi) (procedure Eq. 2). A
    phantom passes no scene to measure_node: its background masks its own
    tube (D-064)."""
    t0 = time.perf_counter()
    draw, rng = draw_replicate(cfg, seed, g, r, d_override_um)     # d_override_um: scripts/end_to_end.py
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
    for name in ("k_iter", "iter_status", "n_rounds", "n_restacks", "w_hat_um", "beta_hat", "xi_hat", "zeta_hat",
                 "irls_ok", "d_pilot_um", "alpha_pilot", "pilot_status"):     # Block 13: the iteration, the pilot
        row[name] = getattr(res, name)
    row.update(at_bound=";".join(res.at_bound), flags=";".join(res.flags), in_S=not reasons,
               reject=";".join(reasons), ratio=res.d_hat_um / draw.d_um, seconds=time.perf_counter() - t0)
    return row
