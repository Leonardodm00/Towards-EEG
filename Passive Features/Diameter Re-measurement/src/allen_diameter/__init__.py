"""allen_diameter -- dendrite diameters from the Allen 63x brightfield stacks.

Layout (scientific-coding skill, section 4; decision D-022):
    config.py   every parameter, with its source (decisions D-021 to D-024)
    loading/    SWC and image-block access (no computation)
    model/      geometry, the blurred-tube forward model, the renderer
    analysis/   focus score, path fit, background, fit, table, inversion
    plotting/   figures only

The per-node chain in analysis/node_pipeline.py is ONE function fed either by
a real image block (allen_image_io.fetch_zblock, Colab only) or by the
renderer (model/render.py), so the bias table measures the bias of the code
actually used (procedure section 3.7).

Documents (D-020): Passive Features/Diameter Re-measurement/docs/ on main.
Spec: specs/SPEC.md at the repository root (Coverage = this folder).
"""
from __future__ import annotations

__version__ = "0.1.0"
