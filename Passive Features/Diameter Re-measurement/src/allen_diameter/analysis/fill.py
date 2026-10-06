"""Fill flagged nodes and smooth along a stretch -- Block 7 in specs/SPEC.md
(D5; design handoff step 6).

Along one unbranched stretch, nodes ordered proximal -> distal:
  fill_policy "same_branch_then_allen": a node without d_tilde takes the
      median of the nearest node with d_tilde on each side (one or two
      values); with none on the stretch, Allen's diameter;
  "allen_only": Allen's diameter; "none": NaN stays.
Then a running median of median_window_nodes nodes (scipy.ndimage.median_filter,
mode "nearest") removes spine bumps.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def fill_stretch(d_tilde, allen_diameter, cfg):
    """(d_final (n,), filled_from (n,) of 'self' | 'neighbours' | 'allen' | 'none')."""
    c = cfg.correction
    d = np.asarray(d_tilde, dtype=float).copy()
    allen = np.broadcast_to(np.asarray(allen_diameter, dtype=float), d.shape)
    have = np.isfinite(d)
    src = np.where(have, "self", "none").astype(object)
    if c.fill_policy == "same_branch_then_allen":
        idx = np.flatnonzero(have)
        for i in np.flatnonzero(~have):
            left, right = idx[idx < i], idx[idx > i]
            vals = ([d[left[-1]]] if left.size else []) + ([d[right[0]]] if right.size else [])
            if vals:
                d[i], src[i] = float(np.median(vals)), "neighbours"
            else:
                d[i], src[i] = allen[i], "allen"
    elif c.fill_policy == "allen_only":
        d[~have], src[~have] = allen[~have], "allen"
    elif c.fill_policy != "none":
        raise ValueError("unknown fill_policy %r" % (c.fill_policy,))
    w = int(c.median_window_nodes)
    if w > 1:
        ok = np.isfinite(d)
        if ok.all():
            d = ndimage.median_filter(d, size=w, mode="nearest")
        elif ok.any():
            d[ok] = ndimage.median_filter(d[ok], size=w, mode="nearest")
    return d, np.array(list(src), dtype=object)
