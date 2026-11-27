"""Spec-derived tests of Block 53 (analysis/step2_twins.py; tester, 2026-11-27). Oracles from SPEC.md, Block 53 and
its amendment 1: build_twins copies every field from its source (scan, record, pilot) and refuses, naming every
node; split_twins' counts by the largest-remainder rule with the cap n_c - 1 and repeated passes, worked by hand
below; the choice inside a cell is the first V_c of default_rng([seed, cell, 0|1]).permutation(sorted ids);
independent of the input order; fit_planes' bands.

Pure ASCII.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import step2_helpers as H
from allen_diameter import config as C
from allen_diameter.analysis import step2_twins as T


def _cell(specs, cell=7):
    """specs: [(node_id, set)] -> scans, records."""
    scans = [H.plane_scan(n, s, seed=i) for i, (n, s) in enumerate(specs)]
    recs = {n: H.record(n, s, d=0.4 + 0.01 * i) for i, (n, s) in enumerate(specs)}
    return scans, recs


def test_build_twins_fields_equal_their_sources():
    scans, recs = _cell([(5, "near"), (9, "far"), (2, "near")])
    tw = T.build_twins(7, scans, recs, [9, 2, 5])
    assert [t.node_id for t in tw] == [9, 2, 5]                       # the order of chosen_ids
    by = {s.node_id: s for s in scans}
    for t in tw:
        s, r, p = by[t.node_id], recs[t.node_id], recs[t.node_id]["pilot"]
        assert t.cell_id == 7 and t.role == "train" and t.set == s.set
        assert t.d_um == r["allen_diameter_um"] and t.mu_per_um == p["mu_hat_per_um"]
        assert t.k_star == p["k_star"] and t.z_sub_um == p["z_sub_um"]
        assert (t.cx_um, t.cy_um, t.theta_rad, t.phi_rad) == (p["cx_um"], p["cy_um"], p["theta_rad"], p["phi_rad"])
        assert (t.B_pilot_gl, t.d_pilot_um, t.alpha_pilot) == (p["B_bar"], p["d_hat_um"], p["alpha_hat"])
        assert t.ks.dtype == np.int64 and t.B_gl.dtype == np.float64
        for a, b in ((t.ks, s.ks), (t.B_gl, s.B_bar), (t.noise_sd_gl, s.bg_sd_gl), (t.omega_um2, s.omega_um2),
                     (t.se_omega_um2, s.se_omega_um2), (t.area_gl_um, s.area_gl_um)):
            assert_array_equal(a, b)
            assert not a.flags.writeable
        assert t.status == tuple(s.status)
    # the source arrays are not frozen by the copy
    assert scans[0].omega_um2.flags.writeable


def _bad(case):
    scans, recs = _cell([(5, "near"), (9, "far")])
    s5 = scans[0]
    if case == "no_scan":
        scans = scans[1:]
    elif case == "no_record":
        recs.pop(5)
    elif case == "set_differs":
        recs[5]["set"] = "far"
    elif case == "allen_nan":
        recs[5]["allen_diameter_um"] = None
    elif case == "allen_zero":
        recs[5]["allen_diameter_um"] = 0.0
    elif case in ("mu_hat_per_um", "B_bar", "d_hat_um"):
        recs[5]["pilot"][case] = 0.0                                    # finite but not > 0
    elif case in ("cx_um", "cy_um", "theta_rad", "phi_rad", "z_sub_um", "alpha_hat"):
        recs[5]["pilot"][case] = None
    elif case == "no_k_star":
        recs[5]["pilot"]["k_star"] = None
    elif case == "no_bg_sd":
        scans[0] = dataclasses.replace(s5, bg_sd_gl=None)
    elif case == "k_star_differs":
        scans[0] = dataclasses.replace(s5, k_star=s5.k_star + 1)
    elif case == "z_sub_differs":
        scans[0] = dataclasses.replace(s5, z_sub_um=s5.z_sub_um + 1e-6)
    return scans, recs


@pytest.mark.parametrize("case", ["no_scan", "no_record", "set_differs", "allen_nan", "allen_zero", "mu_hat_per_um",
                                  "B_bar", "d_hat_um", "cx_um", "cy_um", "theta_rad", "phi_rad", "z_sub_um",
                                  "alpha_hat", "no_k_star", "no_bg_sd", "k_star_differs", "z_sub_differs"])
def test_build_twins_refusals_name_the_node(case):
    scans, recs = _bad(case)
    with pytest.raises(ValueError) as e:
        T.build_twins(7, scans, recs, [9, 5])
    assert "5" in str(e.value)
    assert "node 9" not in str(e.value)                                # only the offending node


def test_build_twins_names_every_offending_node():
    scans, recs = _cell([(5, "near"), (9, "far"), (3, "near")])
    recs[5]["allen_diameter_um"] = float("nan")
    recs[3]["pilot"]["mu_hat_per_um"] = float("inf")
    with pytest.raises(ValueError) as e:
        T.build_twins(7, scans, recs, [5, 9, 3])
    assert "5" in str(e.value) and "3" in str(e.value)


def test_build_twins_z_sub_within_tolerance_accepted():
    scans, recs = _cell([(5, "near")])
    scans[0] = dataclasses.replace(scans[0], z_sub_um=scans[0].z_sub_um + 5e-10)   # below 1e-9 um
    assert len(T.build_twins(7, scans, recs, [5])) == 1


# --------------------------------------------------------------------------------------------- split_twins ---

def _twins(layout):
    """layout: {cell: {"near": [ids], "far": [ids]}} -> Twins (built directly as the dataclass)."""
    out = []
    for c, sets in layout.items():
        for s, ids in sets.items():
            for nid in ids:
                scans, recs = _cell([(nid, s)])
                out += T.build_twins(c, scans, recs, [nid])
    return out


def _counts(tw):
    out = {}
    for t in tw:
        if t.role == "validation":
            out[(t.cell_id, t.set)] = out.get((t.cell_id, t.set), 0) + 1
    return out


def _expected_choice(tw, seed, counts):
    want = set()
    for (c, s), v in counts.items():
        ids = sorted(t.node_id for t in tw if t.cell_id == c and t.set == s)
        perm = np.random.default_rng([seed, c, 0 if s == "near" else 1]).permutation(np.array(ids))
        want |= {(c, int(x)) for x in perm[:v]}
    return want


ONE = {100: {"near": list(range(1, 9))}}
# hand-worked (see the module docstring): cells 1, 2, 3 with 1, 2 and 5 near twins, far twins 1 (cell 1), 2 (cell 3)
THREE = {1: {"near": [11], "far": [12]}, 2: {"near": [21, 22]}, 3: {"near": [31, 32, 33, 34, 35], "far": [36, 37]}}
EXPECTED = {
    ("ONE", 0.0): {}, ("ONE", 0.25): {(100, "near"): 2}, ("ONE", 0.5): {(100, "near"): 4},
    # f = 0.25, near: N 8, V 2; quotas 0.25, 0.5, 1.25 -> floors 0, 0, 1; the seat left to cell 2 (remainder 4/8)
    #           far: N 3, V 1; quotas 1/3, 2/3 -> cell 3 (remainder 2/3); cell 1 capped at 0 anyway
    ("THREE", 0.25): {(2, "near"): 1, (3, "near"): 1, (3, "far"): 1},
    # f = 0.5, near: V 4; quotas 0.5, 1, 2.5 -> floors 0, 1, 2; the seat: cell 1 (rem 4/8, capped at 0) -> cell 3
    #          far: V 2; quotas 2/3, 4/3 -> floors 0 (cap 0), 1 (cap 1); the seat dropped: sum = min(2, 0 + 1) = 1
    ("THREE", 0.5): {(2, "near"): 1, (3, "near"): 3, (3, "far"): 1},
    ("THREE", 0.0): {},
}


@pytest.mark.parametrize("name,f", sorted(EXPECTED))
def test_split_counts_by_hand(name, f):
    layout = {"ONE": ONE, "THREE": THREE}[name]
    tw = _twins(layout)
    out = T.split_twins(tw, f, 20261127)
    assert [(t.cell_id, t.node_id) for t in out] == [(t.cell_id, t.node_id) for t in tw]   # same order
    assert _counts(out) == EXPECTED[(name, f)]
    got = {(t.cell_id, t.node_id) for t in out if t.role == "validation"}
    assert got == _expected_choice(tw, 20261127, EXPECTED[(name, f)])
    for c, sets in layout.items():                                     # no cell emptied of training twins
        for s, ids in sets.items():
            assert any(t.role == "train" for t in out if t.cell_id == c and t.set == s)


def test_split_order_independent_and_seed():
    tw = _twins(THREE)
    a = T.split_twins(tw, 0.5, 3)
    rng = np.random.default_rng(1)
    shuffled = [tw[i] for i in rng.permutation(len(tw))]
    b = T.split_twins(shuffled, 0.5, 3)
    ra = {(t.cell_id, t.node_id): t.role for t in a}
    rb = {(t.cell_id, t.node_id): t.role for t in b}
    assert ra == rb
    big = _twins({100: {"near": list(range(1, 21))}})
    choices = {frozenset((t.node_id for t in T.split_twins(big, 0.25, s) if t.role == "validation")) for s in range(6)}
    assert len(choices) > 1                                            # another seed changes the choice
    assert all(len(c) == 5 for c in choices)                           # not the counts


@pytest.mark.parametrize("f", [-0.1, 1.0, 1.5, float("nan")])
def test_split_refuses_fraction(f):
    with pytest.raises(ValueError):
        T.split_twins(_twins(ONE), f, 1)


def test_split_refuses_duplicates():
    tw = _twins(ONE)
    with pytest.raises(ValueError):
        T.split_twins(tw + tw[:1], 0.25, 1)


def test_fit_planes_bands_status_and_nan():
    s2 = C.default_config().step2
    scans = [H.plane_scan(5, "far", k_star=20, nan_plane=20, bad_plane=12)]      # ks 4..36; offsets of q: q - 16
    recs = {5: H.record(5, "far", k_star=20)}
    t = T.build_twins(7, scans, recs, [5])[0]
    j = np.abs(t.ks - 20)
    ok = np.array([s == "ok" for s in t.status]) & np.isfinite(t.omega_um2)
    assert_array_equal(T.fit_planes(t, s2, "near"), (j <= 3) & ok)
    assert_array_equal(T.fit_planes(t, s2, "far"), (j >= 4) & (j <= 16) & ok)
    assert not T.fit_planes(t, s2, "far")[20] and not T.fit_planes(t, s2, "far")[12]
    assert T.fit_planes(t, s2, "near").dtype == bool


# ---------------------------------------------------------------------------------------------------------------
# Amendment 3 (tester, 2026-11-27 22:11; SPEC Block 53 split_twins): V_s = floor(f N_s + 1/2), "computed as
# floor(f * N_s + 0.5 + 1e-9) so that an exact half is not lost to floating point". Oracles, expected counts by
# exact rational arithmetic (fractions.Fraction of the decimal f): (T6) the five pairs the last report found
# (0.29 x 50, 0.57 x 50, 0.58 x 25, 0.7 x 45, 0.82 x 75) give the rounded-up count; (T7) over f = 0.01 .. 0.99 and
# N = 2 .. 60 in one cell of N near twins (cap N - 1), the validation count equals min(floor(f N + 1/2) exact, N - 1).
# ---------------------------------------------------------------------------------------------------------------

import math  # noqa: E402
from fractions import Fraction  # noqa: E402


def _exact_V(f, N):
    q = Fraction(str(f)) * N + Fraction(1, 2)
    return q.numerator // q.denominator


def _one_cell(N, cell=500):
    """N near twins in one cell, built directly as the dataclass (split_twins reads cell_id, node_id, set)."""
    base = _twins({cell: {"near": [1]}})[0]
    return [dataclasses.replace(base, node_id=i) for i in range(1, N + 1)]


@pytest.mark.parametrize("f,N", [(0.29, 50), (0.57, 50), (0.58, 25), (0.7, 45), (0.82, 75)])
def test_T6_exact_halves_round_up(f, N):
    out = T.split_twins(_one_cell(N), f, 20261127)
    v = sum(t.role == "validation" for t in out)
    assert v == _exact_V(f, N), (f, N, v)
    assert math.floor(f * N + 0.5) == v - 1          # non-vacuous: the plain float expression loses the half


def test_T7_count_grid_against_exact_rounding():
    bad = []
    cache = {N: _one_cell(N) for N in range(2, 61)}
    for i in range(1, 100):
        f = i / 100
        for N in range(2, 61):
            v = sum(t.role == "validation" for t in T.split_twins(cache[N], f, 1))
            if v != min(_exact_V(f, N), N - 1):
                bad.append((f, N, v, _exact_V(f, N)))
    assert not bad, bad[:10]
