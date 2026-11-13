"""Spec-derived tests of Block 8 (SWC I/O, stretches, frame, per-node CSV
row, membrane area, the radius-distribution script) -- integration tester,
run D1, 2026-11-13. Expected values: hand-written SWC files, the frustum and
cylinder formulas, the stretch rule walked by hand, allen_image_align's
equations (1)-(2) (spec_helpers_D1) and numpy.percentile; the code under test
only produces the outputs that are checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block8_cell.py -q
"""
from __future__ import annotations

import dataclasses
import math
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import spec_helpers_D1 as H

import allen_radius_distribution as ARD
from allen_diameter import config as C
from allen_diameter.analysis import cell as CELL
from allen_diameter.analysis import invert as INV
from allen_diameter.loading import swc_io

NAN = float("nan")

# soma 1; apical-like chain 2-3-4 branching at 4 into 5-6 and 7; axon 8 off the soma;
# a dendrite 9 whose parent is the axon 8; a basal one-node stretch 10 off the soma
ROWS = [
    (1, 1, 0.0, 0.0, 0.0, "5.0000", -1),
    (2, 4, 0.0, 6.0, 0.0, "1.2500", 1),
    (3, 4, 0.0, 9.0, 0.5, "1.0", 2),
    (4, 4, 0.0, 12.0, 0.5, "0.875", 3),
    (5, 4, 2.0, 14.0, 0.5, "0.5000", 4),
    (6, 4, 4.0, 16.0, 1.0, "0.4000", 5),
    (7, 4, -2.0, 15.0, 0.5, "0.6000", 4),
    (8, 2, 0.0, -6.0, 0.0, "0.3000", 1),
    (9, 3, 1.0, -8.0, 0.0, "0.30000", 8),
    (10, 3, -6.0, 0.0, 0.0, "0.7000", 1),
]


@pytest.fixture
def swcfile(tmp_path):
    p = str(tmp_path / "cell.swc")
    H.write_swc_text(p, ROWS, comments=("# ORIGINAL_SOURCE test", "# SCALE 1.0 1.0 1.0", "#"))
    return p


# ---------------------------------------------------------------- O8.1 ----

def test_read_and_byte_identical_write(swcfile, tmp_path):
    swc = swc_io.read_swc(swcfile)
    assert_array_equal(swc.ids, [r[0] for r in ROWS])
    assert_array_equal(swc.types, [r[1] for r in ROWS])
    assert_array_equal(swc.parent, [r[6] for r in ROWS])
    assert_allclose(swc.xyz, [[r[2], r[3], r[4]] for r in ROWS], rtol=0, atol=0)
    assert_allclose(swc.radius, [float(r[5]) for r in ROWS], rtol=0, atol=0)
    out = str(tmp_path / "same.swc")
    swc_io.write_swc(swc, out)
    assert open(out, "rb").read() == open(swcfile, "rb").read()


def test_new_radii_change_only_the_sixth_token(swcfile, tmp_path):
    swc = swc_io.read_swc(swcfile)
    new = np.array([5.0, 0.123456789, 0.5, 2.0, 0.25, 0.4, 0.61, 0.3, 0.33333333, 0.7])
    out = str(tmp_path / "new.swc")
    swc_io.write_swc(swc, out, radius=new)
    a = open(swcfile).read().split("\n")
    b = open(out).read().split("\n")
    assert len(a) == len(b)
    for la, lb in zip(a, b):
        if not la or la.startswith("#"):
            assert la == lb
            continue
        ta, tb = la.split(" "), lb.split(" ")
        assert len(ta) == len(tb) == 7
        assert ta[:5] == tb[:5] and ta[6] == tb[6]           # only the radius token changes
        dec_a = len(ta[5].split(".")[1]) if "." in ta[5] else 0
        dec_b = len(tb[5].split(".")[1]) if "." in tb[5] else 0
        assert dec_b == max(dec_a, swc_io.MIN_DECIMALS) or dec_b >= max(dec_a, 4), (ta[5], tb[5])
    back = swc_io.read_swc(out)
    assert_allclose(back.radius, new, rtol=0, atol=0.5e-4 + 1e-12)   # rounded at >= 4 decimals


@pytest.mark.parametrize("text", ["1 3 0 0 0 1.0\n", "1 3 0 0 zero 1.0 -1\n", "1 3 0 0 0 1.0 -1\n2 3 a 0 0 1 1\n"])
def test_malformed_lines_refused(tmp_path, text):
    p = tmp_path / "bad.swc"
    p.write_text(text)
    with pytest.raises(ValueError):
        swc_io.read_swc(str(p))


@pytest.mark.parametrize("bad", [np.full(10, -0.1), np.array([NAN] + [0.5] * 9), np.full(9, 0.5)])
def test_bad_radii_refused(swcfile, tmp_path, bad):
    swc = swc_io.read_swc(swcfile)
    with pytest.raises(ValueError):
        swc_io.write_swc(swc, str(tmp_path / "x.swc"), radius=bad)


# ---------------------------------------------------------------- O8.2 ----

def test_frustum_and_dendrite_area(swcfile):
    swc = swc_io.read_swc(swcfile)
    idx = {int(i): k for k, i in enumerate(swc.ids)}
    fr = swc_io.frustum_areas_um2(swc)
    for (i, t, x, y, z, rad, p) in ROWS:
        if p < 0:
            assert fr[idx[i]] == 0.0
            continue
        r1, r2 = float(rad), float(ROWS[idx[p]][5])
        h = math.dist((x, y, z), ROWS[idx[p]][2:5])
        assert abs(fr[idx[i]] - H.ref_frustum(r1, r2, h)) < 1e-12 * max(1.0, fr[idx[i]])
    # dendrite area: frusta to dendrite parents, a cylinder of the node's own radius to a
    # non-dendrite parent (the soma, or the axon for node 9)
    exp = 0.0
    for (i, t, x, y, z, rad, p) in ROWS:
        if t not in (3, 4) or p < 0:
            continue
        r1 = float(rad)
        pr = ROWS[idx[p]]
        h = math.dist((x, y, z), pr[2:5])
        exp += H.ref_frustum(r1, float(pr[5]), h) if pr[1] in (3, 4) else 2 * math.pi * r1 * h
    got = CELL.dendrite_area_um2(swc)
    assert abs(got - exp) < 1e-12 * exp
    two = swc.radius * 2.0
    ratio = CELL.area_ratio(swc, two)
    # doubling every radius doubles cylinders exactly; frusta grow by a factor in (1, 2]
    exp2 = 0.0
    for (i, t, x, y, z, rad, p) in ROWS:
        if t not in (3, 4) or p < 0:
            continue
        pr = ROWS[idx[p]]
        h = math.dist((x, y, z), pr[2:5])
        exp2 += H.ref_frustum(2 * float(rad), 2 * float(pr[5]), h) if pr[1] in (3, 4) else 4 * math.pi * float(rad) * h
    assert abs(ratio - exp2 / exp) < 1e-12


# ---------------------------------------------------------------- O8.3 ----

def test_stretches_by_hand(swcfile):
    swc = swc_io.read_swc(swcfile)
    got = [[int(swc.ids[k]) for k in rows] for rows in CELL.stretches(swc, (3, 4))]
    exp = H.ref_stretches([r[0] for r in ROWS], [r[1] for r in ROWS], [r[6] for r in ROWS])
    assert sorted(got) == sorted(exp)
    assert sorted(exp) == sorted([[2, 3, 4], [5, 6], [7], [9], [10]])
    flat = sorted(i for s in got for i in s)
    assert flat == sorted(r[0] for r in ROWS if r[1] in (3, 4))      # a partition of the dendrite nodes


def test_stretch_branch_prepends_the_parent(swcfile):
    swc = swc_io.read_swc(swcfile)
    xyz = np.asarray(swc.xyz, float)
    by_first = {int(swc.ids[r[0]]): r for r in CELL.stretches(swc, (3, 4))}
    br, off = CELL.stretch_branch(swc, by_first[5], xyz)
    assert list(map(int, br.ids)) == [4, 5, 6] and off == 1                     # parent dendrite node 4
    br, off = CELL.stretch_branch(swc, by_first[2], xyz)
    assert list(map(int, br.ids)) == [2, 3, 4] and off == 0                     # parent is the soma
    br, off = CELL.stretch_branch(swc, by_first[10], xyz)
    assert list(map(int, br.ids)) == [1, 10] and off == 1                       # one node: its parent, any type
    assert_allclose(br.s_um, [0.0, 6.0], atol=1e-12)
    root_only = [k for k in range(len(swc.ids)) if int(swc.ids[k]) == 1]
    with pytest.raises(ValueError):
        CELL.stretch_branch(swc, np.array(root_only), xyz)                     # a lone root


# ---------------------------------------------------------------- O8.4 ----

@pytest.mark.parametrize("shift,flip,z0", [((0.0, 0.0), None, 0.0), ((3.5, -2.0), None, 1.4), ((1.0, 2.0), 4000.0, -0.3)])
def test_to_image_um(shift, flip, z0):
    rng = np.random.default_rng(1)
    xyz = rng.uniform(0, 300, size=(20, 3))
    got = CELL.to_image_um(xyz, 0.1144, shift, flip, z0)
    assert_allclose(got, H.ref_swc_to_full_um(xyz, 0.1144, shift, flip, z0), rtol=1e-14, atol=1e-12)


# ---------------------------------------------------------------- O8.5 ----

def test_node_row_contract():
    import test_spec_block6_phantoms as B6
    res = B6.make_result(node_id=7, d_hat_um=1.1, mu_hat_per_um=0.8, flags=("faint", "steep"))
    inv = INV.Inversion(1.02, 1.02, 1.078, ("large_correction",))
    row = CELL.node_row(res, inv, 1.05, "self", 0.3, 0.099)
    for col in CELL.CSV_COLUMNS:
        assert col in row, col
    assert abs(row["mu_d_hat"] - 0.8 * 1.1) < 1e-15
    assert row["d_tilde_um"] == 1.02 and row["b_hat"] == 1.078
    assert row["d_final_um"] == 1.05 and row["filled_from"] == "self"
    assert row["allen_radius_um"] == 0.3 and row["sigma_fit_um"] == 0.099
    fl = row["flags"].split(";")
    assert {"faint", "steep", "large_correction"} <= set(fl)        # node flags and inversion flags, ';'-joined


# ---------------------------------------------------------------- O8.6 ----

def test_radius_distribution_script(swcfile, tmp_path):
    """Block 8 / Block 18.1: collect([swc], types=(3, 4)) gives per dendrite
    node its id, type and D_A = 2 r_A, and the file's summary; the pooled
    percentiles equal numpy.percentile (default linear interpolation)."""
    nodes, summary = ARD.collect([swcfile], (3, 4))
    dend = [r for r in ROWS if r[1] in (3, 4)]
    assert sorted(int(n["node_id"]) for n in nodes) == sorted(r[0] for r in dend)
    for n in nodes:
        src = [r for r in dend if r[0] == int(n["node_id"])][0]
        assert int(n["type"]) == src[1]
        assert abs(float(n["diameter_um"]) - 2 * float(src[5])) < 1e-12
    (s,) = summary
    for label, types in (("dend", (3, 4)), ("basal", (3,)), ("apical", (4,))):
        diam = np.array([2 * float(r[5]) for r in dend if r[1] in types])
        assert s["%s_n" % label] == diam.size
        for q in ARD.PERCENTILES:
            assert abs(s["%s_p%g" % (label, q)] - np.percentile(diam, q)) < 1e-12, (label, q)
        assert abs(s["%s_mean" % label] - diam.mean()) < 1e-12
        for t in ARD.THRESHOLDS_UM:
            assert abs(s["%s_share_above_%gum" % (label, t)] - np.mean(diam > t)) < 1e-15
