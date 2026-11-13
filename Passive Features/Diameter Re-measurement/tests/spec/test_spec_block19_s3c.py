"""Spec-derived tests of Block 19 (S3c: the pooled calibration, 19.1-19.8) --
integration tester run D4, 2026-11-13, commit 1555036. Written from
specs/SPEC.md, not from the code. Expected values come from spec_helpers_D4
(the round-robin pool, the ANOVA and the leave-one-out z, an independent
weighted Eq. 4 minimum), scipy.stats, or by hand.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block19_s3c.py -q
"""
from __future__ import annotations

import dataclasses
import json
import math
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

import spec_helpers_D4 as D4

from allen_diameter.analysis import calibration as CAL
from allen_diameter.analysis import calibration_pool as POOL
from allen_diameter.analysis import calibration_report as CR
from allen_diameter.analysis import cell_outliers as CO
from allen_diameter.analysis import far_field as FF
from allen_diameter.loading import calibration_io

NAN = float("nan")
Q = 10 ** 6
G_SYM = np.array([0.0, 0.010, 0.060, 0.150, 0.280])


def cell_scans(sid, n_near, n_far, seed):
    """One cell's scans in its file order: n_near near scans then n_far far scans, node ids 1.. (unqualified)."""
    rng = np.random.default_rng(seed)
    near = D4.eq4_scans(G_SYM, rng.uniform(0, 0.03, n_near), np.zeros(n_near), np.zeros(n_near),
                        np.zeros(n_near, dtype=int), node_ids=list(range(1, n_near + 1)))
    far = D4.f1_scans(np.zeros(n_far), np.zeros(n_far), np.zeros(n_far), np.zeros(n_far, dtype=int), 0.603, 0.79,
                      node_ids=list(range(n_near + 1, n_near + n_far + 1)))
    order = list(near) + list(far)
    rng.shuffle(order)
    return int(sid), order


# ------------------------------------------------------------ 19.4 ----------

def test_qualify_and_cell_of():
    """O19.4, O19.6: q = s 10^6 + n; cell_of: the map's specimen, else q // 10^6 for q >= 10^6, else none."""
    sid, scans = cell_scans(529878215, 2, 1, 1)
    qs = POOL.qualify(scans, sid)
    assert [q.node_id for q in qs] == [sid * Q + s.node_id for s in scans]
    assert POOL.cell_of(529878215 * Q + 17) == (529878215, 17)
    assert POOL.cell_of(17) == (None, 17)
    assert POOL.cell_of(Q) == (1, 0)
    assert POOL.cell_of(5 * Q + 3, {str(5 * Q + 3): [99, 42]}) == (99, 42)


def test_sufficiency_threshold():
    """O19.3: sufficient iff n >= calib_min_nodes_per_set (10)."""
    cfg = D4.cfg_with()
    assert POOL.sufficiency(9, cfg)["sufficient"] is False
    assert POOL.sufficiency(10, cfg)["sufficient"] is True


@pytest.mark.parametrize("per_cell_cap", [1, 2, 30])
@pytest.mark.parametrize("pool_cap", [None, 1, 3, 4, 30])
def test_pool_scans_equals_round_robin_reference(per_cell_cap, pool_cap):
    """O19.5: the pooled scans (qualified, in order) equal the round-robin of the spec over the per-cell and
    pool caps; the cells' input order changes nothing; meta records the caps and the counts."""
    cfg = D4.cfg_with(survey=dict(calib_scans_per_cell=per_cell_cap, calib_pool_max_per_set=pool_cap))
    per_cell = [cell_scans(303, 2, 4, 3), cell_scans(101, 3, 0, 1), cell_scans(202, 1, 4, 2)]
    pooled, meta = POOL.pool_scans(per_cell, cfg)
    ref = D4.ref_pool(per_cell, per_cell_cap, pool_cap)
    got = {w: [POOL.cell_of(s.node_id, meta["id_map"]) for s in pooled if s.set == w] for w in ("near", "far")}
    assert got == ref
    assert [s.set for s in pooled] == ["near"] * len(ref["near"]) + ["far"] * len(ref["far"])
    for w in ("near", "far"):
        m = meta["sets"][w]
        assert m["taken"] == len(ref[w]) and m["max_per_cell"] == per_cell_cap and m["max_nodes"] == pool_cap
        assert m["available"] == sum(1 for _, sc in per_cell for x in sc if x.set == w)
    pooled2, meta2 = POOL.pool_scans(per_cell[::-1], cfg)
    assert [s.node_id for s in pooled2] == [s.node_id for s in pooled] and meta2 == meta


def test_pool_two_cells_sharing_a_node_id_stay_two_nodes():
    cfg = D4.cfg_with()
    a = (1, D4.eq4_scans(G_SYM, [0.01], [0.0], [0.0], [0], node_ids=[7]))
    b = (2, D4.eq4_scans(G_SYM, [0.02], [0.0], [0.0], [0], node_ids=[7]))
    pooled, meta = POOL.pool_scans([a, b], cfg)
    assert sorted(s.node_id for s in pooled) == [1 * Q + 7, 2 * Q + 7]


# ------------------------------------------------------------ 19.6 ----------

@pytest.mark.parametrize("ids, unit, rule, n_cells", [
    ([1, 2, 3], "cell", "node", 1),
    ([7 * Q + 1, 7 * Q + 2], "cell", "node", 1),
    ([c * Q + 1 for c in range(1, 6)], "cell", "larger", 5),
    ([c * Q + 1 for c in range(1, 10)], "cell", "larger", 9),
    ([c * Q + 1 for c in range(1, 11)], "cell", "cell", 10),
    ([c * Q + 1 for c in range(1, 13)], "cell", "cell", 12),
    ([c * Q + 1 for c in range(1, 13)], "node", "node", 12)])
def test_bootstrap_plan(ids, unit, rule, n_cells):
    """O19.9: node (unit node, or one cell), cell (>= bootstrap_cell_min = 10 cells), larger (2-9 cells)."""
    cfg = D4.cfg_with(calibration=dict(bootstrap_unit=unit))
    assert CAL.bootstrap_plan(ids, cfg) == (rule, n_cells)


def test_resample_node_and_cell_draws():
    """O19.9: the node draw is rng.integers(0, n, n); the cell draw is C draws over the cells in increasing label,
    each drawn cell contributing all its indices in order; the labels (19.7) do not change the draws: a node
    keeps its cell, a cell copy is labelled by its draw number."""
    cells = np.array([30, 10, 30, 20, 10, 30, 20])
    idx = CAL.resample(cells, np.random.default_rng(5), "node")
    assert_array_equal(idx, np.random.default_rng(5).integers(0, 7, 7))
    i2, lab2 = CAL.resample(cells, np.random.default_rng(5), "node", with_labels=True)
    assert_array_equal(i2, idx)
    assert_array_equal(lab2, cells[idx])
    rng = np.random.default_rng(8)
    pick = np.random.default_rng(8).integers(0, 3, 3)
    members = [np.flatnonzero(cells == c) for c in (10, 20, 30)]
    want = np.concatenate([members[k] for k in pick])
    got = CAL.resample(cells, rng, "cell")
    assert_array_equal(got, want)
    g2, lab = CAL.resample(cells, np.random.default_rng(8), "cell", with_labels=True)
    assert_array_equal(g2, want)
    assert_array_equal(lab, np.concatenate([np.full(members[k].size, t) for t, k in enumerate(pick)]))
    with pytest.raises(ValueError):
        CAL.resample(cells, np.random.default_rng(0), "branch")


def test_pick_se_larger_and_validation():
    """O19.9: under larger the larger finite SE is reported (the other when one is NaN), both recorded,
    few_cells True; bootstrap_cell_min < 2 and an unknown unit refused."""
    r = CAL.pick_se("larger", 5, 0.02, 90, 0.05, 80)
    assert r["se"] == 0.05 and r["unit"] == "cell" and r["few_cells"] and r["se_node"] == 0.02
    r = CAL.pick_se("larger", 5, 0.06, 90, 0.05, 80)
    assert r["se"] == 0.06 and r["unit"] == "node"
    r = CAL.pick_se("larger", 5, 0.06, 90, NAN, 0)
    assert r["se"] == 0.06 and r["se_cell"] is None
    assert CAL.pick_se("cell", 12, NAN, None, 0.04, 99)["few_cells"] is False
    for bad in (dict(bootstrap_cell_min=1), dict(bootstrap_unit="branch")):
        with pytest.raises(ValueError):
            D4.cfg_with(calibration=bad)


def test_anchor_se_larger_rule_on_five_cells():
    """O19.9 / 19.6 (c): at 5 cells the rule is larger -- both bootstraps run, the larger SE reported."""
    cfg = D4.cfg_with(calibration=dict(bootstrap_resamples=15))
    scans = []
    for c in range(1, 6):
        rng = np.random.default_rng(c)
        part = D4.eq4_scans(G_SYM * (1 + 0.3 * rng.normal()), rng.uniform(0, 0.03, 3), np.zeros(3), np.zeros(3),
                            np.zeros(3, dtype=int), node_ids=[c * Q + n for n in (1, 2, 3)], noise_sd=0.002, rng=rng)
        scans += part
    rec = CAL.anchor_se(scans, cfg, np.random.default_rng(2))
    assert rec["rule"] == "larger" and rec["n_cells"] == 5 and rec["few_cells"]
    assert rec["se_node"] is not None and rec["se_cell"] is not None
    assert rec["se"] == max(rec["se_node"], rec["se_cell"])


# ------------------------------------------------------------ 19.7 ----------

def test_cell_weights_rule():
    """O19.10: w_i = N / (C N_c(i)), mean 1; ones for one cell, equal cells and pool_weighting node."""
    cfg = D4.cfg_with()
    lab = np.array([5] * 6 + [9] * 2)
    assert_allclose(CAL.cell_weights(lab, cfg), [8 / 12] * 6 + [8 / 4] * 2, rtol=1e-15)
    assert_allclose(np.mean(CAL.cell_weights(np.array([1, 1, 2, 3, 3, 3]), cfg)), 1.0, rtol=1e-15)
    assert_array_equal(CAL.cell_weights(np.array([4, 4, 4]), cfg), np.ones(3))
    assert_array_equal(CAL.cell_weights(np.array([1, 1, 2, 2]), cfg), np.ones(4))
    assert_array_equal(CAL.cell_weights(lab, D4.cfg_with(calibration=dict(pool_weighting="node"))), np.ones(8))


@pytest.mark.parametrize("weighting, W", [("cell", (1.0, 1.0)), ("node", (6.0, 2.0))])
def test_weighted_growth_is_the_weighted_mean_law(weighting, W):
    """O19.10, 19.7 (a): two cells of 6 and 2 nodes, exact scans with the axis on plane 0 and growth
    a_c 0.3 delta^2 (a = 1, 2): the knots at 0.28-0.84 um equal (sum W_c a_c / sum W_c) 0.3 delta^2 to 1e-8 um^2
    (the spec's figure); residual_um2 is the unweighted model minus data (1e-12) and cost the weighted one."""
    cfg = D4.cfg_with(calibration=dict(pool_weighting=weighting))
    scans = []
    for cell, a, n in ((101, 1.0, 6), (202, 2.0, 2)):
        delta = np.arange(5) * D4.DZ
        scans += D4.eq4_scans(a * 0.3 * delta ** 2, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, dtype=int),
                              node_ids=[cell * Q + i for i in range(1, n + 1)])
    g = CAL.fit_growth(scans, cfg)
    abar = (W[0] * 1.0 + W[1] * 2.0) / (W[0] + W[1])
    assert_allclose(g.g_um2[1:4], abar * 0.3 * (np.arange(1, 4) * D4.DZ) ** 2, atol=1e-8)
    w = CAL.cell_weights(np.array([101] * 6 + [202] * 2), cfg)
    assert_allclose(g.node_weights, w, rtol=1e-15)
    model = np.array([g.c_um2[i] + D4.ref_growth(g.z_ax_um[i] - z, g.g_um2) for i, s in enumerate(scans)
                      for z in s.z_um])
    data = np.concatenate([s.omega_um2 for s in scans])
    assert_allclose(g.residual_um2, model - data, atol=1e-12)
    assert_allclose(g.cost, 0.5 * np.sum(w[g.obs_node] * g.residual_um2 ** 2), rtol=1e-9, atol=1e-20)


def test_weighted_growth_matches_independent_weighted_minimum():
    """O19.10: on noisy scans of cells of 8, 3 and 5 nodes the weighted fit is the minimum of
    sum_i w_i sum_k r_ik^2 found independently (cost no larger by 1e-9 relative; parameters within 2e-6, as
    O10.7)."""
    cfg = D4.cfg_with()
    rng = np.random.default_rng(77)
    scans, labels = [], []
    for cell, n in ((11, 8), (22, 3), (33, 5)):
        k = rng.integers(0, 30, n)
        z_ax = k * D4.DZ + rng.uniform(-0.1, 0.1, n)
        scans += D4.eq4_scans(G_SYM * (1 + 0.2 * rng.normal()), rng.uniform(0, 0.03, n), z_ax,
                              z_ax + rng.uniform(-0.03, 0.03, n), k, noise_sd=0.003, rng=rng,
                              node_ids=[cell * Q + i for i in range(1, n + 1)])
        labels += [cell] * n
    w = np.array([16 / (3 * {11: 8, 22: 3, 33: 5}[c]) for c in labels])
    g = CAL.fit_growth(scans, cfg)
    ref = D4.ref_fit_eq4(scans, weights=w, z_bound=cfg.calibration.z_ax_bound_um)
    assert g.cost <= ref["cost"] * (1 + 1e-9) + 1e-18
    ok = np.isfinite(g.g_um2)
    assert_allclose(g.g_um2[ok], ref["G"][ok], atol=2e-6)
    assert_allclose(g.z_ax_um, ref["z"], atol=2e-6)


def far_two_cells(gam_a=0.6, gam_b=1.0, qualified=True):
    J = 8
    rng = np.random.default_rng(3)
    k_star = np.arange(J) * 40
    z_p = k_star * D4.DZ + rng.uniform(-0.1, 0.1, J)
    gam = np.array([gam_a] * 6 + [gam_b] * 2)
    obs = [[], [], [], []]
    for j in range(J):
        ks = k_star[j] + np.r_[np.arange(-16, -4), np.arange(5, 17)]
        obs[0].append(np.full(ks.size, j)); obs[1].append(ks); obs[2].append(ks * D4.DZ)
        obs[3].append(D4.f1_omega(z_p[j], 0.0, ks * D4.DZ, 0.01, 0.603, gam[j]))
    cat = [np.concatenate(x) for x in obs]
    ids = np.array([(101 if j < 6 else 202) * Q + j + 1 for j in range(J)]) if qualified else np.arange(1, J + 1)
    return FF.FarSelection(ids, np.full(J, 40.0), np.full(J, 1.0), z_p, cat[0].astype(int), cat[1].astype(int),
                           cat[2], cat[3], np.full(cat[3].size, 1.0), np.full(cat[3].size, 1.0), {}, 0, 0,
                           np.full(J, 24))


def test_far_fit_by_cell_nearer_the_cells_mean_slope():
    """O19.10 / 19.7 (d): cell A 6 nodes at gamma 0.6, cell B 2 nodes at 1.0 -- the slope by cell lies nearer
    the cells' mean slope 0.8 than the slope by node, both within [0.6, 1.0]; one cell: the same slope under
    both weightings (1e-12)."""
    by_cell = FF.fit_far_slope(far_two_cells(), 0.603, D4.cfg_with())
    by_node = FF.fit_far_slope(far_two_cells(), 0.603, D4.cfg_with(calibration=dict(pool_weighting="node")))
    assert 0.6 <= by_node.gamma <= 1.0 and 0.6 <= by_cell.gamma <= 1.0
    assert abs(by_cell.gamma - 0.8) < abs(by_node.gamma - 0.8)
    one_a = FF.fit_far_slope(far_two_cells(qualified=False), 0.603, D4.cfg_with())
    one_b = FF.fit_far_slope(far_two_cells(qualified=False), 0.603, D4.cfg_with(calibration=dict(pool_weighting="node")))
    assert_allclose(one_a.gamma, one_b.gamma, atol=1e-12)


# ------------------------------------------------------------ 19.8 ----------

def test_one_way_anova_against_scipy_and_textbook():
    """O19.11: F = MS_B / MS_W equals scipy.stats.f_oneway's F (1e-12 relative); n0 and sigma_b^2 equal the
    textbook formulas (1e-12)."""
    rng = np.random.default_rng(4)
    groups = [rng.normal(0.01 * i, 0.02, n) for i, n in enumerate((2, 3, 4))]
    values = np.concatenate(groups)
    labels = [10] * 2 + [20] * 3 + [30] * 4
    a = CO.one_way_anova(values, labels)
    assert_allclose(a["ms_b"] / a["ms_w"], stats.f_oneway(*groups).statistic, rtol=1e-12)
    ref = D4.ref_anova(values, labels)
    for k in ("ms_w", "ms_b", "n0", "sigma_b2"):
        assert_allclose(a[k], ref[k], rtol=1e-12, atol=1e-18)


def outlier_set(n_cells, shift_cell=None, shift=0.2, seed=6, n_nodes=5):
    rng = np.random.default_rng(seed)
    labels, values = [], []
    for c in range(1, n_cells + 1):
        m = rng.normal(0.0, 0.01)
        if c == shift_cell:
            m += shift
        labels += [c * 100] * n_nodes
        values += list(m + rng.normal(0.0, 0.02, n_nodes))
    return labels, np.array(values)


@pytest.mark.parametrize("n_cells, shift_cell, shift", [(12, 7, 0.2), (12, 3, -0.2), (6, 2, 0.2), (13, None, 0.0)])
def test_assess_set_equals_leave_one_out_rule(n_cells, shift_cell, shift):
    """O19.11: every cell's z and p equal the leave-one-out rule written from the spec (1e-10 relative); a cell
    leaves only when p < alpha / C and C >= 10; at 6 cells nothing leaves (applied False) though the shifted
    cell's |z| is the largest."""
    cfg = D4.cfg_with()
    labels, values = outlier_set(n_cells, shift_cell, shift)
    rec = CO.assess_set(labels, values, cfg)
    ref = D4.ref_cell_z(values, labels)
    for row in rec["rows"]:
        z, p = ref[row["specimen_id"]]
        assert_allclose(row["z"], z, rtol=1e-10)
        assert_allclose(row["p"], p, rtol=1e-10, atol=1e-300)
    thr = 0.05 / n_cells
    want = [c for c, (z, p) in ref.items() if n_cells >= 10 and p < thr]
    assert rec["removed"] == want and rec["applied"] == (n_cells >= 10)
    if shift_cell is not None:
        zs = {r["specimen_id"]: abs(r["z"]) for r in rec["rows"]}
        assert max(zs, key=zs.get) == shift_cell * 100
        if n_cells >= 10:
            assert rec["removed"] == [shift_cell * 100]


def test_assess_set_nan_z_below_three_cells_and_switch():
    cfg = D4.cfg_with()
    labels, values = outlier_set(2)
    rec = CO.assess_set(labels, values, cfg)
    assert all(math.isnan(r["z"]) for r in rec["rows"]) and rec["removed"] == []
    off = D4.cfg_with(calibration=dict(cell_outlier_removal=False))
    labels, values = outlier_set(12, 7, 0.2)
    rec = CO.assess_set(labels, values, off)
    assert rec["removed"] == [] and not rec["applied"]
    assert max(abs(r["z"]) for r in rec["rows"]) > 5


def test_node_values_means_of_used_finite_residuals():
    kc = dict(points=[dict(node_id=1, cell=5, set="near", used=True, residual_um=0.1),
                      dict(node_id=1, cell=5, set="near", used=True, residual_um=0.3),
                      dict(node_id=1, cell=5, set="near", used=False, residual_um=9.0),
                      dict(node_id=2, cell=6, set="near", used=True, residual_um=NAN),
                      dict(node_id=3, cell=6, set="far", used=True, residual_um=0.5),
                      dict(node_id=4, cell=None, set="near", used=True, residual_um=-0.2)])
    labels, values = CO.node_values(kc, "near")
    assert labels == [5, None]
    assert_allclose(values, [0.2, -0.2], rtol=1e-15)


# ------------------------------------------------------------ 19.1 ----------

def three_stretch_swc(breadth_first=False):
    """A soma and three dendrite stretches of 6 nodes from it: A along x (radius 0.25: near), B along y (0.35:
    far), C along -x (0.25: near). Rows depth-first (A, B, C) or breadth-first (interleaved)."""
    from allen_diameter.loading.swc_io import SWC
    rows = [(1, 1, 0.0, 0.0, 0.0, 5.0, -1)]
    for sid, (dx, dy, r) in enumerate(((1, 0, 0.25), (0, 1, 0.35), (-1, 0, 0.25))):
        for t in range(6):
            nid = 2 + 6 * sid + t
            parent = 1 if t == 0 else nid - 1
            rows.append((nid, 3, 8.0 * dx + 0.5 * t * dx, 8.0 * dy + 0.5 * t * dy, 1.0, r, parent))
    if breadth_first:
        head, rest = rows[:1], rows[1:]
        rest = [rest[s * 6 + t] for t in range(6) for s in range(3)]
        rows = head + rest
    a = np.array(rows, dtype=float)
    lines = ["%d %d %.3f %.3f %.3f %.3f %d\n" % (int(r[0]), int(r[1]), r[2], r[3], r[4], r[5], int(r[6])) for r in rows]
    return SWC(a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2:5], a[:, 5], a[:, 6].astype(int), lines,
               list(range(len(rows))), "")


def stub_scan_node(calls):
    def f(branch, i, provider, cfg, reg=None, scene=None, keep_images=False):
        nid = int(branch.ids[i])
        calls.append(nid)
        which = CAL.calibration_set(2.0 * float(branch.radius_um[i]), cfg) or "near"
        sc = D4.make_scan(nid, 0, 0.0, np.full(7, 0.1), which=which)
        return D4.node_result(node_id=nid), sc, []
    return f


def test_scan_cell_cap_and_order(monkeypatch):
    """O19.1: with max_per_set a candidate whose set holds max_per_set scans is not measured (set_full, the stub
    never called); with order_rng the candidates come in the order of order_rng.permutation over the SWC's order;
    the returned list holds every candidate."""
    cfg = D4.cfg_with()
    swc = three_stretch_swc()
    cand = set(int(i) for i in swc.ids[1:])
    calls = []
    monkeypatch.setattr(CAL, "scan_node", stub_scan_node(calls))
    out = CAL.scan_cell(swc, None, cfg, candidates=cand)
    base = [int(i) for i in swc.ids[1:]]                       # SWC order (depth-first rows)
    assert [o[0] for o in out] == base and calls == base
    calls.clear()
    out = CAL.scan_cell(swc, None, cfg, candidates=cand, order_rng=np.random.default_rng(5), max_per_set=3)
    order = [base[k] for k in np.random.default_rng(5).permutation(len(base))]
    assert [o[0] for o in out] == order
    near_ids, far_ids = set(range(2, 8)) | set(range(14, 20)), set(range(8, 14))
    want_calls = []
    n_near = n_far = 0
    for nid in order:
        if nid in far_ids:
            if n_far < 3:
                want_calls.append(nid); n_far += 1
        elif n_near < 3:
            want_calls.append(nid); n_near += 1
    assert calls == want_calls
    full = [o for o in out if o[3] == ["set_full"]]
    assert len(full) == len(base) - 6 and all(o[1] is None and o[2] is None for o in full)


def test_scan_cell_order_is_the_swc_order_for_breadth_first_rows(monkeypatch):
    """O19.1: 'in the SWC's order before it' -- an SWC whose rows interleave three stretches."""
    cfg = D4.cfg_with()
    swc = three_stretch_swc(breadth_first=True)
    calls = []
    monkeypatch.setattr(CAL, "scan_node", stub_scan_node(calls))
    out = CAL.scan_cell(swc, None, cfg, candidates=set(int(i) for i in swc.ids[1:]))
    assert [o[0] for o in out] == [int(i) for i in swc.ids[1:]]


def test_pilot_record_and_node_records():
    """O19.2: pilot_record(None) is None; NaN -> None; k_star None when z_sub is not finite and kept (-1 too)
    when it is; the record reads the pilot's view; node_records carries Allen's diameter, the set and the
    reasons."""
    assert CAL.pilot_record(None) is None
    r = CAL.pilot_record(D4.node_result(k_star=-1, z_sub_um=-0.1, B_bar=NAN, flags=("dark",)))
    assert r["k_star"] == -1 and r["B_bar"] is None and r["flags"] == ["dark"]
    assert set(r) == {"k_star", "z_sub_um", "cx_um", "cy_um", "theta_rad", "phi_rad", "B_bar", "mu_hat_per_um",
                      "d_hat_um", "alpha_hat", "fit_status", "flags", "reg_verdict"}
    assert CAL.pilot_record(D4.node_result(k_star=-1, z_sub_um=NAN))["k_star"] is None
    pv = CAL.pilot_record(D4.node_result(d_hat_um=2.0, alpha_hat=0.9, fit_status="failed", d_pilot_um=0.4,
                                         alpha_pilot=0.2, mu_pilot_per_um=0.5, v0_pilot_um=0.0, B_bar_pilot=170.0,
                                         pilot_status="converged", pilot_flags=("faint",)))
    assert pv["d_hat_um"] == 0.4 and pv["fit_status"] == "converged" and pv["flags"] == ["faint"]
    json.dumps(pv, allow_nan=False)
    swc = three_stretch_swc()
    out = [(2, D4.node_result(node_id=2), None, ["not_flat"]), (9, None, None, ["set_full"])]
    recs = CAL.node_records(out, swc, D4.cfg_with())
    assert recs[0]["set"] == "near" and recs[0]["allen_diameter_um"] == 0.5 and recs[0]["reasons"] == "not_flat"
    assert recs[1]["set"] == "far" and recs[1]["pilot"] is None and recs[1]["scanned"] is False


# ------------------------------------------------------------ 19.5 ----------

def test_pooled_sigma_law_is_the_table_then_the_line():
    """O19.7: numpy.interp of the table up to its last delta, sigma_0 + gamma (delta - delta_0) beyond with a
    far fit, NaN beyond without one and below 0."""
    calib = dict(kernel=dict(kernel_table_delta_um=[0.0, 0.28, 0.56, 0.84],
                             kernel_table_sigma_um=[0.08, 0.122, 0.438, 0.603]),
                 anchor=dict(delta0_um=0.84, sigma0_um=0.603))
    d = np.array([-0.1, 0.0, 0.1, 0.5, 0.84, 1.0, 3.0])
    law = CR.pooled_sigma_law(calib, dict(far_fit=dict(gamma=0.79)))
    want = np.where(d <= 0.84, np.interp(d, [0, 0.28, 0.56, 0.84], [0.08, 0.122, 0.438, 0.603]),
                    0.603 + 0.79 * (d - 0.84))
    want[0] = np.nan
    assert_allclose(law(d), want, rtol=1e-15)
    law0 = CR.pooled_sigma_law(calib, None)
    assert np.all(np.isnan(law0(d[5:]))) and np.isnan(law0(d[:1])).all()
    assert CR.pooled_sigma_law(dict(growth={}), None) is None


# ------------------------------------------------------------ silent-failure hunt ---------------------------

def pooled_set(n_cells=3, n_near=4, n_far=3, far_only_one=False):
    cfg = D4.cfg_with()
    r = cfg.renderer
    sig = D4.ref_sigma_r(np.arange(5) * D4.DZ, r.kernel_table_delta_um, r.kernel_table_sigma_um,
                         r.kernel_continuation_slope)
    G = sig ** 2 - r.sigma_r0_um ** 2
    per_cell = []
    for c in range(n_cells):
        rng = np.random.default_rng(100 + c)
        k = rng.integers(0, 30, n_near)
        z_ax = k * D4.DZ + rng.uniform(-0.1, 0.1, n_near)
        near = D4.eq4_scans(G, rng.uniform(0, 0.03, n_near), z_ax, z_ax, k, node_ids=list(range(1, n_near + 1)))
        nf = 1 if (far_only_one and c == 0) else (0 if far_only_one else n_far)
        kf = 200 + np.arange(nf) * 40
        far = D4.f1_scans(np.zeros(nf), np.zeros(nf), kf * D4.DZ, kf, sig[3], 0.79, amp=60.0, rms=0.5,
                          node_ids=list(range(100, 100 + nf)))
        per_cell.append((1000 + c, near + far))
    return per_cell


def test_pooled_fits_leaves_no_stale_fit_of_an_earlier_run(tmp_path):
    """Silent-failure hunt (19.4, 19.8): after a second pooled run whose far fit cannot run (one far node), the
    folder must not hold the first run's far_fit_pooled.json beside the new calibration_pooled.json -- a reader
    of s3c/ (survey_pool.pbs archives it) would take the old gamma_hat with the new anchor."""
    import survey_cells as SC
    cfg = D4.cfg_with(calibration=dict(bootstrap_resamples=5))
    s3c = str(tmp_path / "s3c")
    os.makedirs(s3c)
    logs = []
    for run, per_cell in ((1, pooled_set()), (2, pooled_set(far_only_one=True))):
        pooled, meta = POOL.pool_scans(per_cell, cfg)
        calibration_io.save_scans(pooled, os.path.join(s3c, "scans_pooled.json"), meta)
        done, _out = SC.pooled_fits(cfg, pooled, meta, s3c, log=logs.append, figures=False)
        if run == 1:
            assert done == dict(fit=True, far_fit=True)
            assert os.path.exists(os.path.join(s3c, "far_fit_pooled.json"))
        else:
            assert done == dict(fit=True, far_fit=False)
    assert not os.path.exists(os.path.join(s3c, "far_fit_pooled.json")), \
        "far_fit_pooled.json of run 1 survives run 2, whose far fit did not run"


def test_fits_and_pool_do_not_mutate_their_inputs():
    """Silent-failure hunt: fit_growth, pool_scans, select_far_planes and fit_far_slope leave the scans they
    read unchanged (node ids, widths, statuses)."""
    cfg = D4.cfg_with()
    per_cell = pooled_set()
    before = [(s, [(x.node_id, x.omega_um2.copy(), tuple(x.status)) for x in sc]) for s, sc in per_cell]
    pooled, _meta = POOL.pool_scans(per_cell, cfg)
    near = [x for x in pooled if x.set == "near"]
    far = [x for x in pooled if x.set == "far"]
    snap = [(x.node_id, x.omega_um2.copy(), tuple(x.status)) for x in pooled]
    CAL.fit_growth(near, cfg)
    sel = FF.select_far_planes(far, cfg)
    om = sel.omega_um2.copy()
    FF.fit_far_slope(sel, 0.603, cfg)
    assert_array_equal(sel.omega_um2, om)
    for (nid, w, st), x in zip(snap, pooled):
        assert nid == x.node_id and np.array_equal(w, x.omega_um2, equal_nan=True) and st == tuple(x.status)
    for (s, recs), (s2, sc) in zip(before, per_cell):
        for (nid, w, st), x in zip(recs, sc):
            assert nid == x.node_id and np.array_equal(w, x.omega_um2, equal_nan=True) and st == tuple(x.status)


def test_far_fit_keeps_an_anchor_se_of_zero(tmp_path, capsys):
    """Silent-failure hunt (15.5): an anchor SE of exactly 0.0 in the calibration file is a number, not a missing
    value -- the far fit's SE must be the bootstrap's alone, not None."""
    import calibrate_kernel as CK
    per_cell = pooled_set(n_cells=1, n_near=10, n_far=6)
    scans = per_cell[0][1]
    d = str(tmp_path)
    p = os.path.join(d, "scans.json")
    calibration_io.save_scans(scans, p, {})
    cj = os.path.join(d, "cfg.json")
    with open(cj, "w") as f:
        json.dump(dict(calibration=dict(bootstrap_resamples=8)), f)
    cal = os.path.join(d, "C.json")
    CK.main(["fit", "--scans", p, "--out", cal, "--config-json", cj])
    C = calibration_io.load_calibration(cal)
    C["anchor"]["se_um"] = 0.0
    with open(cal, "w") as f:
        json.dump(C, f)
    out = os.path.join(d, "F.json")
    CK.main(["far-fit", "--scans", p, "--calibration", cal, "--out", out, "--config-json", cj])
    F = calibration_io.load_calibration(out)["far_fit"]
    assert F["anchor"]["se_um"] == 0.0, F["anchor"]
    assert F["se"] is not None
    assert_allclose(F["se"], F["se_boot"], rtol=1e-12)


def test_pooled_fits_removes_an_earlier_runs_figures_all(tmp_path):
    """Silent-failure hunt (19.8): 'every call first deletes an earlier run's conditional outputs ... so none
    outlives the run that wrote it'; the first pass's by-cell figures of a run that removes a cell go to
    s3c/figures_all/. A second call that removes nothing (cell_outlier_removal false) must not leave the first
    call's figures_all/ (they mark a cell '[removed: near]' that this run kept)."""
    import survey_cells as SC
    rng_cells = []
    cfg_r = D4.cfg_with()
    r = cfg_r.renderer
    sig = D4.ref_sigma_r(np.arange(5) * D4.DZ, r.kernel_table_delta_um, r.kernel_table_sigma_um,
                         r.kernel_continuation_slope)
    G = sig ** 2 - r.sigma_r0_um ** 2
    for c in range(12):
        sid = 1000 + c
        rng = np.random.default_rng(500 + c)
        k = rng.integers(0, 60, 3)
        z_ax = k * D4.DZ + rng.uniform(-0.1, 0.1, 3)
        near = D4.eq4_scans(G * (2.0 if sid == 1005 else 1.0), rng.uniform(0, 0.03, 3), z_ax, z_ax, k,
                            noise_sd=0.003, rng=rng, node_ids=[10, 11, 12])
        kf = 300 + np.arange(2) * 40
        far = D4.f1_scans(np.zeros(2), np.zeros(2), kf * D4.DZ, kf, sig[3], 0.79, amp=60.0, rms=0.5,
                          node_ids=[20, 21], noise_sd=0.003, rng=rng)
        rng_cells.append((sid, near + far))
    s3c = str(tmp_path / "s3c")
    os.makedirs(s3c)
    for removal in (True, False):
        cfg = D4.cfg_with(calibration=dict(bootstrap_resamples=5, cell_outlier_removal=removal))
        pooled, meta = POOL.pool_scans(rng_cells, cfg)
        calibration_io.save_scans(pooled, os.path.join(s3c, "scans_pooled.json"), meta)
        _done, out = SC.pooled_fits(cfg, pooled, meta, s3c, log=lambda m: None, figures=True)
        if removal:
            assert out["near"]["removed"] == [1005] and os.path.isdir(os.path.join(s3c, "figures_all"))
        else:
            assert out["near"]["removed"] == []
            assert not os.path.exists(os.path.join(s3c, "scans_pooled_kept.json"))
    assert not os.path.exists(os.path.join(s3c, "figures_all")), "figures_all/ of the first call survives"
