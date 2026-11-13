"""Several tubes rendered in one stack (Eqs. 5-6 applied jointly) -- Block 24 in specs/SPEC.md (contract stub, wave 1; D-129, D-130).

The entry of Block 24 in specs/SPEC.md is the specification: signatures, shapes,
axis order, dtypes, units, oracles and tolerances. Pure ASCII, for the cluster.
"""
from __future__ import annotations

def render_scene(tubes, mus, z_planes, grid, rcfg, light_direction=+1, reduce=None):
    """-> render.RenderResult; tau (n_planes, ny, nx) or the reduced planes. tubes: Block 2 Tubes; mus: 1/um >= 0."""
    raise NotImplementedError("contract stub")


def synthetic_scene_block(tubes, mus, ks, left, top, width, height, cfg, rng, pad_px, qtables=None):
    """-> (block uint8 (n_planes, height, width), ks, valid, frame) in the fetch_zblock contract."""
    raise NotImplementedError("contract stub")
