"""Reproducer and check (testers, 2026-11-14; rewritten at 18:30 against Block 36 amendments 2 and 3,
D-142, D-143). The earlier version showed finding F1 of the 11:00 report: a node's straight tube cut at
tube_half_length_um = 5 um misses the shadow of its own branch beyond, 1 to 7 % of the dip on tilted nodes.
Amendments 2-3 keep the 5 um tube but treat the stretch beyond the own nodes (own_ids_within) as neighbours
whose predicted shadow is removed by the cleaning. On a straight 20 um tube at tilt phi the decomposition
is checked here without rendering an image:

    tau_full(v) + S_beyond(v)  ~=  tau_tube(v)      in every plane k* +- 4,

tau_full = line_fast of the whole tube (half-length 20 um), tau_tube = line_fast of the node's tube over its
own nodes (local_tube with axis_extent), S_beyond = Block 25's stretch_profiles of the segments beyond them
(the same absorption, partition_vertical; true features). The amended oracle bounds the cleaned profile's
departure from the tube model at 1.5 % of the dip; printed: that departure, and without the cleaning the
shortfall (should exceed 1 % on tilted nodes, the amendment's "reproduced").

Run (about 1-3 min): python3 tests/spec/repro_block36_tube_cut.py  (from the code folder)
Pure ASCII.
"""
import dataclasses
import math
import types

import numpy as np

import nc_helpers  # noqa: F401  (sys.path)
from allen_diameter.config import default_config
from allen_diameter.analysis import node_inputs as NI, profiles
from allen_diameter.analysis.node_features import FeaturedScene
from allen_diameter.analysis.node_pipeline import Branch
from allen_diameter.model import geometry, line_fast, shadow


def cfg_pv():
    cfg = default_config()
    return dataclasses.replace(
        cfg, renderer=dataclasses.replace(cfg.renderer, absorption="partition_vertical"),
        cleaning=dataclasses.replace(cfg.cleaning, shadow_absorption="partition_vertical", soma_mu_per_um=0.0),
        inference=dataclasses.replace(cfg.inference, absorption="partition_vertical"))


def case(phi_deg, spacing, d=1.0, mu=1.0, L=20.0, cfg=None):
    """-> (relative departure after cleaning (n_planes,), shortfall before (n_planes,), own ids count)."""
    cfg = cfg or cfg_pv()
    m = cfg.measure
    v, n_avg, step = profiles.profile_offsets(m), profiles.n_along(m), float(m.profile_step_um)
    ks = np.arange(-cfg.cleaning.plane_half_range, cfg.cleaning.plane_half_range + 1)
    phi = math.radians(phi_deg)
    a = np.array([math.cos(phi), 0.0, math.sin(phi)])
    s_nodes = np.linspace(-L, L, int(round(2 * L / spacing)) + 1)
    xyz = s_nodes[:, None] * a[None, :]
    ids = np.arange(len(s_nodes)) + 2
    br = Branch(ids, np.full(len(ids), 3), xyz, np.full(len(ids), d / 2), s_nodes - s_nodes[0])
    i = int(np.argmin(np.abs(s_nodes)))
    own, Um, Up = NI.own_ids_within(br, i, cfg.inference.tube_half_length_um)
    pl = types.SimpleNamespace(cx_um=0.0, cy_um=0.0, theta_rad=0.0, phi_rad=phi, z_sub_um=0.0)
    Ub, Uf = NI.axis_extent(pl, br, i, Um, Up)
    o, yh, eu = np.zeros(2), np.array([0.0, 1.0]), np.array([1.0, 0.0])
    tube = NI.local_tube(pl, xyz[i], d, 0.0, 0.0, Ub, Uf)
    tau_tube = line_fast.line_profiles(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg)
    full = geometry.Tube((0.0, 0.0, 0.0), d / 2, phi, 0.0, 1.0, L, "axial")
    tau_full = line_fast.line_profiles(full, mu, ks, o, yh, eu, v, n_avg, step, cfg)
    seg_xyz = np.hstack([xyz[:-1], xyz[1:]])                       # parent end, child end
    fs = FeaturedScene(seg_xyz=seg_xyz, seg_d_um=np.full((len(ids) - 1, 2), d),
                       seg_mu_per_um=np.full((len(ids) - 1, 2), mu),
                       seg_ids=np.column_stack([ids[1:], ids[:-1]]).astype(np.int64),
                       seg_type=np.full(len(ids) - 1, 3), soma=np.zeros((0, 3)))
    beyond = [int(x) for x in ids if int(x) not in own] + [own[0], own[-1]]
    S_beyond = shadow.stretch_profiles(fs, beyond, ks, o, yh, eu, v, n_avg, step, cfg)
    dip = np.max(1.0 - tau_tube, axis=1)
    after = np.max(np.abs(tau_full + S_beyond - tau_tube), axis=1) / dip
    before = np.max(np.abs(tau_full - tau_tube), axis=1) / dip
    return after, before, len(own), (Ub, Uf)


def main():
    for phi in (0.0, 30.0, 60.0):
        for spacing in (1.0, 2.0):
            after, before, n_own, U = case(phi, spacing)
            print("phi %2.0f deg, nodes %.0f um apart: %d own nodes, tube -%.2f..+%.2f um; departure / dip after "
                  "cleaning max %.4f; shortfall before %.4f..%.4f"
                  % (phi, spacing, n_own, U[0], U[1], after.max(), before.min(), before.max()))


if __name__ == "__main__":
    main()
