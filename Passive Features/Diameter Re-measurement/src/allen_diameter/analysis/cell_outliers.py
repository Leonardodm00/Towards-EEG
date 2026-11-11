"""Block 19.8 (D-120.3): cells whose mean residual is too rare under the between-cell Gaussian.

For one set (near or far) of Block 19.5's points (calibration_report.kernel_law_by_cell), a node's value is the
mean of its used, finite residuals from the pooled law (um) and a cell's mean m_c the mean of its N_c nodes'
values. For each cell c, the one-way random-effects ANOVA of the OTHER cells' node values gives sigma_b^2 (the
method of moments with the effective size n0 for unequal cells, truncated at 0) and sigma_w^2 (the within mean
square), and m_ref, the mean of their cell means; then

    z_c = (m_c - m_ref) / sqrt(sigma_b^2 + sigma_w^2 / N_c),    p_c = 2 Phi(-|z_c|),

and the cell leaves the set when p_c < alpha / C (Bonferroni over the set's C cells), only when C >=
cfg.calibration.bootstrap_cell_min. The cell's own scatter sigma_w^2 / N_c is in the denominator because a cell
mean carries it; leaving the cell out of its own yardstick keeps an outlier from widening it.

Pure computation: no file is read or written here.
"""
import math

import numpy as np
from scipy import stats

_NONE = "none"                                  # the label of points whose cell is unknown (cell None)


def _key(cell):
    return _NONE if cell is None else cell


def node_values(kc, which):
    """(labels, values) for one set of kernel_law_by_cell's points: one entry per node with at least one used,
    finite residual -- the node's cell (None kept as None) and the mean of those residuals (um), in the order of
    the nodes' first point."""
    acc = {}
    for p in kc["points"]:
        if p["set"] != which or not p["used"] or not math.isfinite(p["residual_um"]):
            continue
        acc.setdefault(int(p["node_id"]), [p["cell"], []])[1].append(float(p["residual_um"]))
    labels = [v[0] for v in acc.values()]
    values = np.array([float(np.mean(v[1])) for v in acc.values()], dtype=float)
    return labels, values


def one_way_anova(values, labels):
    """The one-way random-effects ANOVA of node values grouped by their cells (sizes may differ). Returns dict:
    n_cells C, n_nodes N, cells (keys in first-appearance order), sizes, means (per cell), grand_mean (over the
    nodes), ms_w = sum (y - ybar_c)^2 / (N - C), ms_b = sum N_c (ybar_c - ybar)^2 / (C - 1),
    n0 = (N - sum N_c^2 / N) / (C - 1), sigma_b2 = max(0, (ms_b - ms_w) / n0), sigma_w2 = ms_w,
    icc = sigma_b2 / (sigma_b2 + sigma_w2). A quantity whose degrees of freedom are zero is NaN."""
    values = np.asarray(values, dtype=float)
    keys = [_key(c) for c in labels]
    cells = list(dict.fromkeys(keys))
    groups = [values[[k == c for k in keys]] for c in cells]
    C, N = len(cells), int(values.size)
    sizes = np.array([g.size for g in groups], dtype=int)
    means = np.array([float(np.mean(g)) for g in groups], dtype=float)
    grand = float(np.mean(values)) if N else float("nan")
    ms_w = float(sum(float(np.sum((g - m) ** 2)) for g, m in zip(groups, means)) / (N - C)) if N > C else float("nan")
    if C >= 2:
        ms_b = float(np.sum(sizes * (means - grand) ** 2) / (C - 1))
        n0 = float((N - np.sum(sizes ** 2) / N) / (C - 1))
    else:
        ms_b = n0 = float("nan")
    sb2 = max(0.0, (ms_b - ms_w) / n0) if all(math.isfinite(x) for x in (ms_b, ms_w, n0)) and n0 > 0 else float("nan")
    tot = sb2 + ms_w
    icc = sb2 / tot if math.isfinite(tot) and tot > 0 else float("nan")
    return dict(n_cells=C, n_nodes=N, cells=cells, sizes=sizes, means=means, grand_mean=grand, ms_w=ms_w, ms_b=ms_b,
                n0=n0, sigma_b2=sb2, sigma_w2=ms_w, icc=icc)


def assess_set(labels, values, cfg):
    """The test of every cell of one set (module docstring). Returns rec: n_cells, applied, alpha, p_threshold,
    min_cells, anova (one_way_anova over all cells), rows (one per cell, in first-appearance order:
    specimen_id, n_nodes, mean_um, ref_mean_um, sigma_b2_um2, sigma_w2_um2, sd_um, z, p, removed) and removed
    (the removed cells' specimen ids)."""
    cal = cfg.calibration
    keys = [_key(c) for c in labels]
    full = one_way_anova(values, labels)
    C = full["n_cells"]
    alpha, min_cells = float(cal.cell_outlier_alpha), int(cal.bootstrap_cell_min)
    applied = bool(cal.cell_outlier_removal) and C >= min_cells
    thr = alpha / C if C else float("nan")
    rows, removed = [], []
    values = np.asarray(values, dtype=float)
    for i, c in enumerate(full["cells"]):
        mine = np.array([k == c for k in keys])
        others = one_way_anova(values[~mine], [lab for lab, m in zip(labels, mine) if not m])
        n_c = int(full["sizes"][i])
        m_c = float(full["means"][i])
        m_ref = float(np.mean(others["means"])) if others["n_cells"] else float("nan")
        var = others["sigma_b2"] + others["sigma_w2"] / n_c
        sd = math.sqrt(var) if math.isfinite(var) and var > 0 else float("nan")
        z = (m_c - m_ref) / sd if math.isfinite(sd) and others["n_cells"] >= 2 else float("nan")
        p = float(2.0 * stats.norm.sf(abs(z))) if math.isfinite(z) else float("nan")
        out = bool(applied and math.isfinite(p) and p < thr)
        cell = None if c == _NONE else c
        rows.append(dict(specimen_id=cell, n_nodes=n_c, mean_um=m_c, ref_mean_um=m_ref,
                         sigma_b2_um2=others["sigma_b2"], sigma_w2_um2=others["sigma_w2"], sd_um=sd, z=z, p=p,
                         removed=out))
        if out:
            removed.append(cell)
    anova = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in full.items()}
    anova["cells"] = [None if c == _NONE else c for c in anova["cells"]]
    return dict(n_cells=C, applied=applied, alpha=alpha, p_threshold=thr, min_cells=min_cells, anova=anova,
                rows=rows, removed=removed)


def cell_outliers(kc, cfg):
    """{"near": rec, "far": rec}: assess_set on each set's node values of kernel_law_by_cell's points kc."""
    out = {}
    for which in ("near", "far"):
        labels, values = node_values(kc, which)
        out[which] = assess_set(labels, values, cfg)
    return out
