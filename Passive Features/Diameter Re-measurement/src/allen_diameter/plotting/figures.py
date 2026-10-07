"""Figures -- plotting only, nothing scientific is computed here (Block 11 in
specs/SPEC.md).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math

import numpy as np


_SCORE_LABELS = {"gradient_energy": "gradient energy G (1/um)", "dip_depth": "dip depth F"}


def node_figure(result, v, I, model, title="", k_swc=None):
    """Two panels for one measured node. Left: the focus curves of the final
    pass over the node's planes -- the configured rule (D-030; k* is its
    maximum, red) and, on a second axis, the dip depth of handoff Eq. 1 with
    the plane it would have chosen (grey) -- and the SWC's own plane k_swc
    (dotted) when given. Right: the fitted profile in plane k* with the model
    and the background B_bar. Returns the matplotlib Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ks = np.asarray(result.focus_planes, dtype=float)
    S = np.asarray(result.focus_score, dtype=float)
    Dp = np.asarray(result.focus_depth, dtype=float)
    if S.size:
        a1.plot(ks, S, "o-", color="tab:red", ms=4, label=_SCORE_LABELS.get(result.focus_rule, result.focus_rule))
        if math.isfinite(result.z_sub_um):
            a1.axvline(result.k_star, color="tab:red", lw=0.9)
        if result.focus_rule != "dip_depth":
            a1b = a1.twinx()
            a1b.plot(ks, Dp, "s--", color="0.55", ms=3, lw=0.9, label=_SCORE_LABELS["dip_depth"])
            a1b.set_ylabel("dip depth F", color="0.45")
            if result.k_star_depth != result.k_star and np.any(np.isfinite(Dp)):
                a1b.axvline(result.k_star_depth, color="0.55", lw=0.9, ls="--")
            h1, l1 = a1.get_legend_handles_labels()
            h2, l2 = a1b.get_legend_handles_labels()
            a1.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")
    if k_swc is not None:
        a1.axvline(k_swc, color="k", lw=0.8, ls=":")
    a1.set_xlabel("plane k (red: k*; grey: dip-depth choice; dotted: SWC)")
    a1.set_ylabel(_SCORE_LABELS.get(result.focus_rule, result.focus_rule))
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
