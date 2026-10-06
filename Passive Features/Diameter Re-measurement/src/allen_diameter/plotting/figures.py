"""Figures -- plotting only, nothing scientific is computed here (Block 11 in
specs/SPEC.md).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np


def node_figure(result, v, I, model, title=""):
    """Two panels for one measured node: the focus score over the node's planes
    (handoff Eq. 1, the final pass) and the fitted profile in plane k* with the
    model and the background B_bar. Returns the matplotlib Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    F = np.asarray(result.focus_F, dtype=float)
    if F.size:
        a1.plot(np.arange(F.size), F, "o-", color="0.3")
        if np.any(np.isfinite(F)):
            a1.axvline(int(np.nanargmax(F)), color="tab:red", lw=0.8)
    a1.set_xlabel("plane (index in the node's range)")
    a1.set_ylabel("focus score F")
    a2.plot(v, I, ".", color="0.2", ms=4, label="profile")
    if model is not None:
        a2.plot(v, model, "-", color="tab:blue", lw=1.2, label="model")
    if math.isfinite(result.B_bar):
        a2.axhline(result.B_bar, color="0.6", lw=0.8, ls="--", label="B_bar")
    a2.set_xlabel("v (um)")
    a2.set_ylabel("grey level")
    a2.legend(fontsize=7, frameon=False)
    fig.suptitle(title or "node %d: d_hat %.3f um, mu_hat %.2f /um, phi %.1f deg, %s"
                 % (result.node_id, result.d_hat_um, result.mu_hat_per_um, math.degrees(result.phi_rad),
                    result.fit_status), fontsize=9)
    fig.tight_layout()
    return fig
