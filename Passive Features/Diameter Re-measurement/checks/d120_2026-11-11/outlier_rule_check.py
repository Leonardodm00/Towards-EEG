"""Check by computation (Block 19.8, D-120.3; 2026-11-11): how Block 19.8's rule behaves on simulated cells, and a
variant of it.

Model (simulated, truth chosen): a node's value y_{c,j} = b_c + e_{c,j} + s 1[c shifted], c = 1..C cells, j = 1..n
nodes per cell, b_c ~ N(0, s_b^2), e_{c,j} ~ N(0, s_w^2), all independent; s the shift of the cells made to stand
out.

Rule A, as implemented (cell_outliers.assess_set at the default configuration): z_c = (m_c - m_ref) /
sqrt(sb2_(-c) + sw2_(-c) / N_c), p = 2 Phi(-|z|), removed when p < 0.05 / C (applied from 10 cells).
Rule B, the variant: the same z with the variance of m_ref added to the denominator,
    Var(m_ref) = (1 / (C - 1)^2) sum_{c' != c} (sb2_(-c) + sw2_(-c) / N_c'),
and p from Student's t with C - 2 degrees of freedom instead of the Normal (for equal N_c and sb2 not truncated,
sb2 + sw2 / n is the sample variance of the other cells' means, so z_B is the deleted (externally studentized)
residual of a cell mean, t with C - 2 df when no cell stands out).

Reported over R datasets, both rules on the same datasets:
  null:    no cell shifted -- the fraction of datasets where any cell is removed (the family-wise error rate the
           Bonferroni threshold aims to hold at 0.05);
  masking: m cells shifted by s -- the fraction of datasets where exactly the m shifted cells are removed.
Standalone; numpy, scipy and the package. Run from the workstream folder:
    python3 checks/d120_2026-11-11/outlier_rule_check.py          (about 4 min)
"""
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "src")
from allen_diameter.analysis import cell_outliers as CO  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402

CFG = dict(seed=20261111, R=400, s_b=0.01, s_w=0.02, shift=0.2)


def simulate(rng, C, n, s_b, s_w, n_shift, shift):
    labels, values = [], []
    for c in range(C):
        eff = s_b * rng.standard_normal() + (shift if c >= C - n_shift else 0.0)
        labels += [c] * n
        values += list(eff + s_w * rng.standard_normal(n))
    return labels, np.array(values)


def removed_b(rec, alpha=0.05):
    """Rule B from rule A's rows (each row holds the other cells' sb2, sw2 and the cell's own N_c)."""
    rows = rec["rows"]
    C = len(rows)
    if C < 10:
        return []
    sizes = np.array([r["n_nodes"] for r in rows], dtype=float)
    out = []
    for i, r in enumerate(rows):
        others = np.delete(sizes, i)
        var_ref = float(np.sum(r["sigma_b2_um2"] + r["sigma_w2_um2"] / others)) / (C - 1) ** 2
        z = (r["mean_um"] - r["ref_mean_um"]) / np.sqrt(r["sigma_b2_um2"] + r["sigma_w2_um2"] / r["n_nodes"] + var_ref)
        if 2.0 * stats.t.sf(abs(z), C - 2) < alpha / C:
            out.append(r["specimen_id"])
    return out


def main():
    cfg = default_config()
    rng = np.random.default_rng(CFG["seed"])
    R = CFG["R"]
    print("s_b %.3f um, s_w %.3f um, shift %.2f um, %d datasets each, seed %d" % (
        CFG["s_b"], CFG["s_w"], CFG["shift"], R, CFG["seed"]))
    print("\nnull (no cell shifted): P(any cell removed), target <= 0.05")
    print("  %-12s %8s %8s" % ("", "rule A", "rule B"))
    for C in (10, 13, 51):
        for n in (2, 5, 30):
            a = b = 0
            for _ in range(R):
                lab, val = simulate(rng, C, n, CFG["s_b"], CFG["s_w"], 0, 0.0)
                rec = CO.assess_set(lab, val, cfg)
                a += bool(rec["removed"])
                b += bool(removed_b(rec))
            print("  C %2d, n %2d %8.3f %8.3f" % (C, n, a / R, b / R))
    print("\nm cells shifted by %.2f um (n 5): P(exactly the m shifted removed)" % CFG["shift"])
    print("  %-12s %8s %8s" % ("", "rule A", "rule B"))
    for C in (13, 51):
        for m in (1, 2, 3, 5):
            a = b = 0
            for _ in range(R):
                lab, val = simulate(rng, C, 5, CFG["s_b"], CFG["s_w"], m, CFG["shift"])
                rec = CO.assess_set(lab, val, cfg)
                want = list(range(C - m, C))
                a += sorted(rec["removed"]) == want
                b += sorted(removed_b(rec)) == want
            print("  C %2d, m %d   %8.3f %8.3f" % (C, m, a / R, b / R))


if __name__ == "__main__":
    main()
