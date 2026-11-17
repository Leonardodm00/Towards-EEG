"""The same-node clearance figures -- SPEC Block 38 (D-156.2). Computes nothing scientific.

Keep this file pure ASCII so it survives transfer to the cluster.
"""
from __future__ import annotations


def figure_own(v, ks, profiles_gl, B_gl, clearance, meta, planes=None, v0_um=0.0):
    """-> matplotlib Figure: one panel per plane shown (planes: values of ks; None: all) with the profile, the
    cleared profile, the in-band model, the whole model, the background level and the chord as a bar from
    v0_um - c_k / 2 to v0_um + c_k / 2; one y range; clearance an own_clearance.OwnClearance (SPEC Block 38)."""
    raise NotImplementedError("SPEC Block 38")


def figure_layers(v, layer_z, layers_gl, ks, plane, planes_z, band_half_um, meta):
    """-> matplotlib Figure: for the plane `plane` (a value of ks) the image of the layers' shadows (gl) over
    the offset v and the layer depth, with the plane's depth and its band drawn (SPEC Block 38)."""
    raise NotImplementedError("SPEC Block 38")


def save(fig, path, dpi):
    """Write fig as a PNG at dpi; -> path (SPEC Block 38)."""
    raise NotImplementedError("SPEC Block 38")
