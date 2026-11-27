"""Smoke test for the twins of step 2 and the training/validation split -- Block 53 in specs/SPEC.md
(section 5, "Kernel calibration, step 2"; D-261.5, D-274.1, D-274.4): analysis.step2_twins.build_twins,
split_twins and fit_planes.

Inputs are built here from the contract: synthetic PlaneScan objects and node-record dicts in the format of
calibration.node_records / pilot_record. No other step-2 block is imported.

Checks
    test_known_answer   split_twins' counts by SPEC's rule on hand-computed cases (one cell of 8 near; cells of
                        1, 2, 5 near and 3 far; f = 0, 0.25, 0.5; a case where the leftover seats need a second
                        pass; one-twin cells), every count derived in the comments from the rule; fit_planes'
                        masks written out by hand on a scan with a non-ok and a NaN plane
    test_reference      build_twins: every field equals its source (scan or record); split_twins: the validation
                        nodes equal the first V_c of default_rng([seed, cell_id, 0|1]).permutation of the sorted
                        node ids, restated here independently
    test_convergence    not a discretisation (SkipTest)
    test_invariants     no cell emptied of training twins (random cases), sum_c V_c <= V_s, the same roles under a
                        shuffled input, another seed changes the choice but not the counts
    test_contract       types, shapes, dtypes, read-only arrays, input arrays left writeable and unchanged
    test_determinism    two splits with one seed agree
    test_edge_cases     each refusal of build_twins names the node (and every node when several fail); the
                        refusals of split_twins and fit_planes

Run (from Passive Features/Diameter Re-measurement/)
    python3 tests/smoke/test_smoke_step2_twins.py
    python3 -m pytest tests/smoke/test_smoke_step2_twins.py -q

Pure ASCII, for the cluster.
"""
from __future__ import annotations

import copy
import dataclasses
import importlib.metadata
import platform
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from allen_diameter.config import Step2Config  # noqa: E402
from allen_diameter.analysis.calibration import PlaneScan, PILOT_RECORD_FIELDS  # noqa: E402
from allen_diameter.analysis import step2_twins as T  # noqa: E402

SEED = 20261127  # Step2Config.split_seed
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure.

    On failure both arrays are saved to ARTIFACT_DIR/<name>.npz before the
    AssertionError propagates. Justify rtol and atol at the call site
    (float64 roundoff, discretisation-error estimate, statistical spread).
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / f"{safe_name}.npz"
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def observed_order(h, err):
    """Least-squares slope p of log e(h) = log C + p * log h.

    h: step sizes (any order); err: positive errors e(h) at those steps.
    """
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    if np.any(err <= 0.0):
        raise ValueError("observed_order needs strictly positive errors")
    slope, _intercept = np.polyfit(np.log(h), np.log(err), 1)
    return float(slope)


# ------------------------------------------------------- synthetic inputs ---

def make_scan(node_id, set_="near", k_star=10, P=3, bg_sd=True, se=True):
    """A PlaneScan of planes k* - P .. k* + P with values distinct per node and plane."""
    ks = np.arange(k_star - P, k_star + P + 1, dtype=np.int64)
    n = ks.size
    q = np.arange(n, dtype=np.float64)
    base = 0.001 * node_id
    return PlaneScan(
        node_id=node_id, d_hat_um=0.9, phi_rad=0.1, alpha_hat=0.3, k_star=k_star, z_sub_um=-99.0,
        ks=ks, z_um=0.28 * ks, omega_um2=0.05 + base + 0.01 * q, vw_um2=0.06 + 0.01 * q,
        area_gl_um=20.0 + base + q, depth_gl=10.0 + q, B_bar=180.0 + base + 0.5 * q,
        status=tuple("ok" for _ in range(n)), set=set_,
        se_omega_um2=(0.002 + 0.0001 * q) if se else None,
        bg_sd_gl=(3.0 + base + 0.1 * q) if bg_sd else None)


def make_pilot(node_id, k_star=10):
    """A pilot record (calibration.pilot_record's format: PILOT_RECORD_FIELDS)."""
    p = dict(k_star=k_star, z_sub_um=2.8 + 0.01 * node_id, cx_um=50.0 + node_id, cy_um=60.0 + node_id,
             theta_rad=0.3, phi_rad=0.2, B_bar=181.0, mu_hat_per_um=0.4 + 0.001 * node_id, d_hat_um=0.95,
             alpha_hat=0.31, fit_status="ok", flags=[], reg_verdict="ok")
    assert set(p) == set(PILOT_RECORD_FIELDS)
    return p


def make_record(node_id, set_="near", d=0.8, k_star=10):
    """A node record (calibration.node_records' format)."""
    return dict(node_id=node_id, allen_diameter_um=d, set=set_, reasons="", scanned=True,
                pilot=make_pilot(node_id, k_star))


def make_twin(cell_id, node_id, set_="near"):
    """A Twin for the split (its arrays do not matter there)."""
    z = np.zeros(1)
    return T.Twin(cell_id=cell_id, node_id=node_id, set=set_, role="train", d_um=1.0, mu_per_um=0.4,
                  k_star=0, z_sub_um=0.0, cx_um=0.0, cy_um=0.0, theta_rad=0.0, phi_rad=0.0,
                  ks=np.zeros(1, dtype=np.int64), B_gl=z, noise_sd_gl=z, omega_um2=z, se_omega_um2=z,
                  area_gl_um=z, status=("ok",))


def twins_of(layout):
    """layout {cell_id: (n_near, n_far)} -> twins, node ids 1.. per cell (near first, then far)."""
    out = []
    for c, (nn, nf) in layout.items():
        for i in range(nn):
            out.append(make_twin(c, 1 + i, "near"))
        for i in range(nf):
            out.append(make_twin(c, 1 + nn + i, "far"))
    return out


def counts(twins):
    """{(cell_id, set): number of validation twins}, every (cell, set) present."""
    out = {}
    for t in twins:
        key = (t.cell_id, t.set)
        out[key] = out.get(key, 0) + (t.role == "validation")
    return out


def three_cells():
    """The SPEC's case: cell 101 has 1 near, cell 102 has 2 near, cell 103 has 5 near and 3 far."""
    return twins_of({101: (1, 0), 102: (2, 0), 103: (5, 3)})


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    # Rule (SPEC Block 53): per set, V_s = floor(f N_s + 1/2); quota of cell c = V_s n_c / N_s; floors, then the
    # leftover seats by largest remainder (ties to the lower cell_id); cap V_c <= n_c - 1.
    # -- one cell (7) of 8 near
    one = twins_of({7: (8, 0)})
    # f = 0:    V = floor(0 + 0.5) = 0.
    # f = 0.25: V = floor(2 + 0.5) = 2; quota 2, cap 7: V_7 = 2.
    # f = 0.5:  V = floor(4 + 0.5) = 4; quota 4, cap 7: V_7 = 4.
    for f, v in ((0.0, 0), (0.25, 2), (0.5, 4)):
        got = counts(T.split_twins(one, f, SEED))
        assert got == {(7, "near"): v}, (f, got)
    # -- cells 101 (1 near), 102 (2 near), 103 (5 near, 3 far); near N = 8, far N = 3
    # f = 0: every V = 0.
    # f = 0.25, near: V = floor(2 + 0.5) = 2; V n_c = 2, 4, 10; floors //8 = 0, 0, 1; remainders %8 = 2, 4, 2;
    #   caps 0, 1, 4. One seat left: largest remainder 102 (4/8) -> 102 gets it. V_c = 0, 1, 1.
    # f = 0.25, far: V = floor(0.75 + 0.5) = 1; only 103 (n 3, cap 2): V_103 = 1.
    # f = 0.5, near: V = floor(4 + 0.5) = 4; V n_c = 4, 8, 20; floors 0, 1, 2; remainders 4, 0, 4; one seat
    #   left: 101 and 103 tie at 4/8 -> 101 first (lower id) but is at its cap 0, so the seat goes to the next,
    #   103 (2 -> 3 <= cap 4). V_c = 0, 1, 3 (sum 4 = V).
    #   [This line rests on the reading "a seat a capped cell cannot take goes to the next cell in the ranking";
    #   under "it is dropped" V_103 would be 2. Reported to the orchestrator as the open reading.]
    # f = 0.5, far: V = floor(1.5 + 0.5) = 2; cap of 103 is 2: V_103 = 2.
    cases = {
        0.0: {(101, "near"): 0, (102, "near"): 0, (103, "near"): 0, (103, "far"): 0},
        0.25: {(101, "near"): 0, (102, "near"): 1, (103, "near"): 1, (103, "far"): 1},
        0.5: {(101, "near"): 0, (102, "near"): 1, (103, "near"): 3, (103, "far"): 2},
    }
    for f, expect in cases.items():
        got = counts(T.split_twins(three_cells(), f, SEED))
        assert got == expect, (f, got, expect)
    # -- leftover seats needing a second pass: cells 201, 202, 203 with 1 near each, 204 with 10 near; N = 13.
    # f = 0.5: V = floor(6.5 + 0.5) = 7; V n_c = 7, 7, 7, 70; floors //13 = 0, 0, 0, 5; remainders %13 =
    #   7, 7, 7, 5; caps 0, 0, 0, 9. Two seats left; the one-twin cells cannot take any, so both go to 204:
    #   V_204 = 7 (sum 7 = V). [Same open reading as above.]
    got = counts(T.split_twins(twins_of({201: (1, 0), 202: (1, 0), 203: (1, 0), 204: (10, 0)}), 0.5, SEED))
    assert got == {(201, "near"): 0, (202, "near"): 0, (203, "near"): 0, (204, "near"): 7}, got
    # -- what cannot be placed is dropped: one cell with 1 near, f = 0.5: V = floor(0.5 + 0.5) = 1, cap 0 -> 0.
    #    Two one-twin cells, f = 0.75: V = floor(1.5 + 0.5) = 2, caps 0, 0 -> 0, 0.
    assert counts(T.split_twins(twins_of({5: (1, 0)}), 0.5, SEED)) == {(5, "near"): 0}
    got = counts(T.split_twins(twins_of({5: (1, 0), 6: (1, 0)}), 0.75, SEED))
    assert got == {(5, "near"): 0, (6, "near"): 0}, got

    # -- fit_planes: k* = 10, planes 10 - 17 .. 10 + 17 = -7 .. 27 (35 planes); offsets j = |k - 10|.
    scan = make_scan(1, k_star=10, P=17)
    status = list(scan.status)
    omega = scan.omega_um2.copy()
    status[scan.ks.tolist().index(12)] = "no_core"     # j = 2: inside the near band, not ok
    omega[scan.ks.tolist().index(8)] = np.nan          # j = 2: inside the near band, ok but NaN
    status[scan.ks.tolist().index(20)] = "no_core"     # j = 10: inside the far band, not ok
    omega[scan.ks.tolist().index(-5)] = np.nan         # j = 15: inside the far band, NaN
    scan = dataclasses.replace(scan, status=tuple(status), omega_um2=omega)
    tw = T.build_twins(1, [scan], {1: make_record(1)}, [1])[0]
    cfg = Step2Config()                                # near_planes 3, far_planes (4, 16)
    # near: j <= 3 -> k = 7..13, minus k = 12 (not ok) and k = 8 (NaN): {7, 9, 10, 11, 13}
    near_expect = np.isin(tw.ks, [7, 9, 10, 11, 13])
    # far: 4 <= j <= 16 -> k = -6..6 and 14..26, minus k = 20 (not ok) and k = -5 (NaN)
    far_k = [k for k in list(range(-6, 7)) + list(range(14, 27)) if k not in (20, -5)]
    far_expect = np.isin(tw.ks, far_k)
    assert_array_equal(T.fit_planes(tw, cfg, "near"), near_expect)
    assert_array_equal(T.fit_planes(tw, cfg, "far"), far_expect)
    assert int(T.fit_planes(tw, cfg, "near").sum()) == 5
    assert int(T.fit_planes(tw, cfg, "far").sum()) == 26 - 2   # 13 + 13 planes in the bands, 2 removed
    # another configuration's bands: near_planes 1 -> k = 9..11 (all usable); far (15, 17) -> k = -7, -6, -5,
    # 25, 26, 27 minus k = -5 (NaN)
    cfg2 = dataclasses.replace(cfg, near_planes=1, far_planes=(15, 17))
    assert_array_equal(T.fit_planes(tw, cfg2, "near"), np.isin(tw.ks, [9, 10, 11]))
    assert_array_equal(T.fit_planes(tw, cfg2, "far"), np.isin(tw.ks, [-7, -6, 25, 26, 27]))


def test_reference():
    # build_twins: every field equals its source
    scans = [make_scan(4, "near", k_star=10, P=3), make_scan(9, "far", k_star=12, P=16)]
    nodes = {4: make_record(4, "near", d=0.7, k_star=10), 9: make_record(9, "far", d=2.5, k_star=12),
             11: make_record(11)}                       # a record not chosen: ignored
    twins = T.build_twins(55, scans, nodes, [9, 4])
    assert [t.node_id for t in twins] == [9, 4]        # the order of chosen_ids
    by_scan = {s.node_id: s for s in scans}
    for t in twins:
        s, r = by_scan[t.node_id], nodes[t.node_id]
        p = r["pilot"]
        assert t.cell_id == 55 and t.role == "train" and t.set == s.set == r["set"]
        assert t.d_um == r["allen_diameter_um"] and t.mu_per_um == p["mu_hat_per_um"]
        assert t.k_star == p["k_star"]
        # the pilot's z_sub_um, not the scan's (-99 in these synthetic scans)
        for name in ("z_sub_um", "cx_um", "cy_um", "theta_rad", "phi_rad"):
            assert getattr(t, name) == p[name], name
        assert_array_equal(t.ks, s.ks)
        assert_array_equal(t.B_gl, s.B_bar)
        assert_array_equal(t.noise_sd_gl, s.bg_sd_gl)
        assert_array_equal(t.omega_um2, s.omega_um2)
        assert_array_equal(t.se_omega_um2, s.se_omega_um2)
        assert_array_equal(t.area_gl_um, s.area_gl_um)
        assert t.status == tuple(s.status)

    # split_twins: the validation nodes are the first V_c of the seeded permutation of the sorted node ids.
    # f = 0.25 on three_cells(): V = (0, 1, 1) near for 101, 102, 103, and 1 far for 103 (test_known_answer).
    twins = T.split_twins(three_cells(), 0.25, SEED)
    got = {}
    for t in twins:
        if t.role == "validation":
            got.setdefault((t.cell_id, t.set), []).append(t.node_id)
    expect = {}
    for (c, s, ids, v) in ((102, "near", [1, 2], 1), (103, "near", [1, 2, 3, 4, 5], 1),
                           (103, "far", [6, 7, 8], 1)):
        perm = np.random.default_rng([SEED, c, 0 if s == "near" else 1]).permutation(np.array(ids))
        expect[(c, s)] = sorted(int(x) for x in perm[:v])
    assert {k: sorted(v) for k, v in got.items()} == expect, (got, expect)


def test_convergence():
    raise unittest.SkipTest("not a discretisation: bookkeeping and an integer allocation")


def test_invariants():
    rng = make_rng(7)
    for trial in range(200):
        n_cells = int(rng.integers(1, 6))
        layout = {int(c): (int(rng.integers(0, 7)), int(rng.integers(0, 5)))
                  for c in rng.choice(1000, size=n_cells, replace=False)}
        twins = twins_of(layout)
        f = float(rng.choice([0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 0.9]))
        out = T.split_twins(twins, f, SEED)
        assert [(t.cell_id, t.node_id) for t in out] == [(t.cell_id, t.node_id) for t in twins]
        cnt = counts(out)
        for s_i, s in enumerate(T.SETS):
            n_s = {c: layout[c][s_i] for c in layout if layout[c][s_i] > 0}
            N = sum(n_s.values())
            V = int(np.floor(f * N + 0.5))
            total = sum(cnt[(c, s)] for c in n_s)
            assert total <= V, (layout, f, s, total, V)
            # every seat that fits under the caps is placed (the reading of "what cannot be placed")
            assert total == min(V, sum(n - 1 for n in n_s.values())), (layout, f, s, total, V)
            for c, n in n_s.items():
                assert cnt[(c, s)] <= n - 1, (layout, f, c, s)     # a cell keeps a training twin
        # the same roles under a shuffled input
        perm = rng.permutation(len(twins))
        shuffled = T.split_twins([twins[i] for i in perm], f, SEED)
        roles = {(t.cell_id, t.node_id): t.role for t in out}
        assert all(roles[(t.cell_id, t.node_id)] == t.role for t in shuffled)
    # another seed: the counts unchanged, the choice changed for at least one of a few seeds
    # (cell 7 of 8 near, f = 0.25, 2 of 8 chosen: one other seed gives the same pair with probability 1/28)
    one = twins_of({7: (8, 0)})
    base = T.split_twins(one, 0.25, SEED)
    chosen0 = {t.node_id for t in base if t.role == "validation"}
    differ = 0
    for s in range(1, 6):
        other = T.split_twins(one, 0.25, SEED + s)
        assert counts(other) == counts(base)
        differ += {t.node_id for t in other if t.role == "validation"} != chosen0
    assert differ >= 1, "five other seeds all chose the same validation nodes"


def test_contract():
    scans = [make_scan(4, "near")]
    snap = {k: np.array(getattr(scans[0], k), copy=True) for k in ("ks", "B_bar", "bg_sd_gl", "omega_um2",
                                                                       "se_omega_um2", "area_gl_um")}
    t = T.build_twins(3, scans, {4: make_record(4)}, [4])[0]
    n = scans[0].ks.size
    assert type(t.cell_id) is int and type(t.node_id) is int and type(t.k_star) is int
    for name in ("d_um", "mu_per_um", "z_sub_um", "cx_um", "cy_um", "theta_rad", "phi_rad"):
        assert type(getattr(t, name)) is float, name
    assert t.ks.dtype == np.int64 and t.ks.shape == (n,)
    for name in ("B_gl", "noise_sd_gl", "omega_um2", "se_omega_um2", "area_gl_um"):
        a = getattr(t, name)
        assert a.dtype == np.float64 and a.shape == (n,), name
        assert not a.flags.writeable, name
    assert not t.ks.flags.writeable
    assert isinstance(t.status, tuple) and len(t.status) == n and all(isinstance(x, str) for x in t.status)
    try:
        t.role = "validation"
        raise AssertionError("Twin is not frozen")
    except dataclasses.FrozenInstanceError:
        pass
    # the scan's arrays are left writeable and unchanged
    for k, v in snap.items():
        a = getattr(scans[0], k)
        assert a.flags.writeable, k
        assert_array_equal(a, v)
    # a scan without se_omega_um2 (an older file): the twin's SE is NaN (an assumption reported)
    t2 = T.build_twins(3, [make_scan(4, se=False)], {4: make_record(4)}, [4])[0]
    assert t2.se_omega_um2.shape == (n,) and np.all(np.isnan(t2.se_omega_um2))
    # fit_planes: (n,) bool
    m = T.fit_planes(t, Step2Config(), "near")
    assert m.dtype == bool and m.shape == (n,)
    # split_twins returns Twins with roles in ROLES, the input left unchanged
    tw = three_cells()
    out = T.split_twins(tw, 0.5, SEED)
    assert all(isinstance(x, T.Twin) and x.role in T.ROLES for x in out)
    assert all(x.role == "train" for x in tw)


def test_determinism():
    a = T.split_twins(three_cells(), 0.5, SEED)
    b = T.split_twins(three_cells(), 0.5, SEED)
    assert [x.role for x in a] == [x.role for x in b]


def _refusal(cell_id, scans, nodes, chosen, must_name):
    try:
        T.build_twins(cell_id, scans, nodes, chosen)
    except ValueError as exc:
        msg = str(exc)
        for nid in must_name:
            assert ("node %d:" % nid) in msg, (must_name, msg)
        return msg
    raise AssertionError("no ValueError for %r" % (must_name,))


def test_edge_cases():
    good_scan, good_rec = make_scan(1), make_record(1)
    # each refusal of build_twins names the node
    _refusal(1, [], {1: good_rec}, [1], [1])                                       # no scan
    _refusal(1, [good_scan], {}, [1], [1])                                         # no record
    _refusal(1, [good_scan], {1: make_record(1, set_="far")}, [1], [1])            # set differs
    _refusal(1, [good_scan], {1: make_record(1, set_="")}, [1], [1])               # set "" (calibration_set)
    for d in (None, float("nan"), float("inf"), 0.0, -0.5):
        _refusal(1, [good_scan], {1: make_record(1, d=d)}, [1], [1])               # diameter
    rec = copy.deepcopy(good_rec)
    rec["pilot"] = None
    _refusal(1, [good_scan], {1: rec}, [1], [1])                                   # no pilot
    for name in ("mu_hat_per_um", "cx_um", "cy_um", "theta_rad", "phi_rad", "z_sub_um"):
        for bad in (None, float("nan")):
            rec = copy.deepcopy(good_rec)
            rec["pilot"][name] = bad
            msg = _refusal(1, [good_scan], {1: rec}, [1], [1])
            assert name in msg, msg
        rec = copy.deepcopy(good_rec)
        del rec["pilot"][name]
        _refusal(1, [good_scan], {1: rec}, [1], [1])
    for bad in (0.0, -0.1):
        rec = copy.deepcopy(good_rec)
        rec["pilot"]["mu_hat_per_um"] = bad
        _refusal(1, [good_scan], {1: rec}, [1], [1])                               # mu not > 0
    rec = copy.deepcopy(good_rec)
    rec["pilot"]["k_star"] = None
    msg = _refusal(1, [good_scan], {1: rec}, [1], [1])                             # no k_star
    assert "k_star" in msg
    _refusal(1, [make_scan(1, bg_sd=False)], {1: good_rec}, [1], [1])              # no bg_sd_gl
    # several offending nodes: every one is named; a good node is not
    scans = [make_scan(1), make_scan(2, bg_sd=False), make_scan(3)]
    nodes = {1: make_record(1), 2: make_record(2), 3: make_record(3, d=-1.0), 5: make_record(5)}
    msg = _refusal(1, scans, nodes, [1, 2, 3, 5], [2, 3, 5])
    assert "node 1:" not in msg, msg
    # split_twins: f outside [0, 1), a duplicate (cell_id, node_id)
    for f in (-0.01, 1.0, 1.5, float("nan")):
        try:
            T.split_twins(three_cells(), f, SEED)
            raise AssertionError("no ValueError for f = %r" % (f,))
        except ValueError:
            pass
    try:
        T.split_twins(three_cells() + [make_twin(102, 2, "near")], 0.25, SEED)
        raise AssertionError("no ValueError for a duplicate twin")
    except ValueError as exc:
        assert "(102, 2)" in str(exc), str(exc)
    # the same node id in two cells is not a duplicate
    out = T.split_twins([make_twin(1, 1), make_twin(2, 1)], 0.25, SEED)
    assert len(out) == 2
    # no twins at all: an empty list
    assert T.split_twins([], 0.25, SEED) == []
    # fit_planes: another which
    t = T.build_twins(1, [good_scan], {1: good_rec}, [1])[0]
    try:
        T.fit_planes(t, Step2Config(), "middle")
        raise AssertionError("no ValueError for which 'middle'")
    except ValueError:
        pass


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"seed {SEED}")
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print(f"== {Path(__file__).name}")
    print(_environment())

    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))

    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline}".rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
