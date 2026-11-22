"""The soma fit's figures -- Block 48 in specs/SPEC.md (D-215, D-222.5). CONTRACT STUB (wave 6).

Figures only: every number drawn is passed in (the entry point builds them from Block 46's fit, Block 45's model planes
and Block 47's outlines and SomaEdit); nothing scientific is computed here. Image panels in the image frame, um, with
matplotlib's extent (left, right, bottom, top), as Block 44's figures. See SPEC Block 48.

Pure ASCII, LF.
"""
from __future__ import annotations


def figure_soma_fit_planes(data, model, region, mask, ks, dz_from_centre_um, extent_um, silhouette_xy, sections_xy,
                           allen_circle, eq_circle, title=""):
    """-> matplotlib Figure: one row per plane shown (data, model, residual = data - model on one symmetric diverging
    scale), the region's boundary, the masked pixels dimmed, the silhouette (dashed), the plane's section (solid),
    Allen's circle (dotted), the sphere of equal area (dash-dot). data, model (K, H, W) gl; region, mask (H, W) bool;
    ks (K,); dz_from_centre_um (K,) um; extent_um (4,); silhouette_xy (2, n) um; sections_xy a list of K (2, n_k) arrays
    or None; allen_circle, eq_circle (x, y, r) um."""
    raise NotImplementedError("Block 48 (wave 6)")


def figure_soma_fit_summary(names, theta, se, fitted, derived, allen, z_um, B_gl, rms_gl, noise_gl=None, title=""):
    """-> Figure: the parameters' table (names, theta, se; the ones not in `fitted` marked held), the derived area,
    volume and r_eq with their SEs beside Allen's radius and 4 pi r^2 (dicts `derived`, `allen`), and two panels
    against depth z_um (K,): the planes' scales B_gl (K,) and the residual RMS per plane rms_gl (K,) (a line at noise_gl
    when given)."""
    raise NotImplementedError("Block 48 (wave 6)")


def figure_soma_morphology(top, side, title=""):
    """-> Figure: the top view (x, y) and the side view (u, w): each a dict of (2, n) arrays `allen`, `silhouette`,
    `sphere`, and `stems`, a list of dicts (old (2,), new (2,), next (2, m), flag): the old first node (open circle), the
    new (filled), an arrow old -> new, the stem's next nodes as a line, the flag in the legend when not "moved"."""
    raise NotImplementedError("Block 48 (wave 6)")
