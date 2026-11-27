"""Spec-derived tests of Block 52 (loading/step2_io.py; tester, 2026-11-27). Oracles from SPEC.md, Block 52: the
chosen-nodes CSV format and its refusals, load_cell's CellScans (the configuration rebuilt from meta["config"] with
meta["calibration"], validated, its full hash), load_start with and without the far-fit file, save_step2 / load_step2
with NaN written as null. Files are written by hand or by calibration_io (Block 10's I/O, a neighbour).

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import step2_helpers as H
from allen_diameter import config as C
from allen_diameter.loading import calibration_io
from allen_diameter.loading import step2_io as IO


def _write(tmp_path, text, name="chosen.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_read_chosen_nodes_format(tmp_path):
    p = _write(tmp_path, "# chosen by the user\n\nspecimen_id,node_id\n 529878215 , 44 \n529878215,7\n\n"
                         "# a comment between rows\n600,3\n529878215,12\n")
    got = IO.read_chosen_nodes(p)
    assert got == {529878215: (7, 12, 44), 600: (3,)}
    assert all(isinstance(k, int) for k in got) and all(isinstance(x, int) for v in got.values() for x in v)


@pytest.mark.parametrize("text,line", [
    ("node_id,specimen_id\n1,2\n", 1),                 # a different header
    ("1,2\n3,4\n", 1),                                 # a missing header (the first row is not it)
    ("specimen_id,node_id\n1,2,3\n", 2),               # three values
    ("specimen_id,node_id\n1\n", 2),                   # one value
    ("specimen_id,node_id\n1,x\n", 2),                 # not an integer
    ("specimen_id,node_id\n1,2.5\n", 2),               # not an integer
    ("specimen_id,node_id\n1,2\n\n1,2\n", 4),          # a node twice for one cell
    ("specimen_id,node_id\n# nothing\n", None),        # no row
])
def test_read_chosen_nodes_refusals(tmp_path, text, line):
    p = _write(tmp_path, text)
    with pytest.raises(ValueError) as e:
        IO.read_chosen_nodes(p)
    if line is not None:
        assert ("line %d" % line) in str(e.value), str(e.value)


def test_same_node_in_two_cells_is_allowed(tmp_path):
    assert IO.read_chosen_nodes(_write(tmp_path, "specimen_id,node_id\n1,5\n2,5\n")) == {1: (5,), 2: (5,)}


def _scans_file(tmp_path, cfg=None, nodes=None, scans=None, drop=None):
    cfg = C.default_config() if cfg is None else cfg
    scans = [H.plane_scan(5, "near"), H.plane_scan(9, "far", seed=1)] if scans is None else scans
    nodes = [H.record(5, "near"), H.record(9, "far", d=0.7)] if nodes is None else nodes
    meta = H.scans_meta(nodes, cfg)
    for k in drop or ():
        meta.pop(k)
    p = tmp_path / "scans.json"
    calibration_io.save_scans(scans, str(p), meta)
    return p, scans, cfg


def test_load_cell_round_trip(tmp_path):
    cfg = C.with_overrides(C.default_config(), {"calibration.patch_half_um": 12.0,
                                                "calibration.stripe_half_widths_um": [1.0, 2.0]})
    p, scans, cfg = _scans_file(tmp_path, cfg)
    cs = IO.load_cell(529878215, p)
    assert cs.cell_id == 529878215 and isinstance(cs.cell_id, int)
    assert dataclasses.is_dataclass(cs) and type(cs).__dataclass_params__.frozen
    assert cs.config.signature_hash("full") == cfg.signature_hash("full")
    assert cs.config.calibration == cfg.calibration                     # the calibration section survives (tuples)
    assert isinstance(cs.config.calibration.stripe_half_widths_um, tuple)
    assert set(cs.nodes) == {5, 9} and cs.nodes[9]["allen_diameter_um"] == 0.7
    assert [s.node_id for s in cs.scans] == [5, 9]
    for a, b in zip(cs.scans, scans):
        for name in ("ks", "omega_um2", "B_bar", "bg_sd_gl", "se_omega_um2", "area_gl_um"):
            assert_array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)))
        assert tuple(a.status) == tuple(b.status) and a.set == b.set and a.k_star == b.k_star
    assert "nodes" in cs.meta and "config" in cs.meta


@pytest.mark.parametrize("missing", ["config", "calibration", "nodes"])
def test_load_cell_refuses_incomplete_meta(tmp_path, missing):
    p, _, _ = _scans_file(tmp_path, drop=[missing])
    with pytest.raises(ValueError):
        IO.load_cell(1, p)


def test_load_cell_refuses_duplicate_ids(tmp_path):
    p, _, _ = _scans_file(tmp_path, scans=[H.plane_scan(5), H.plane_scan(5, seed=3)])
    with pytest.raises(ValueError):
        IO.load_cell(1, p)
    p, _, _ = _scans_file(tmp_path, nodes=[H.record(5), H.record(5), H.record(9, "far")])
    with pytest.raises(ValueError):
        IO.load_cell(1, p)


def _kernel_json(path, delta, sigma, s0, slope, key="kernel"):
    obj = {key: dict(kernel_table_delta_um=list(delta), kernel_table_sigma_um=list(sigma), sigma_r0_um=s0,
                     kernel_continuation="linear", kernel_continuation_slope=slope)} if key else {"kernel_error": "no fit"}
    path.write_text(json.dumps(obj))
    return path


def test_load_start_with_and_without_far_fit(tmp_path):
    cal = _kernel_json(tmp_path / "cal.json", (0, 0.28, 0.56, 0.84), (0.08, 0.15, 0.39, 0.56), 0.08, 0.79)
    far = _kernel_json(tmp_path / "far.json", (0, 0.28, 0.56, 0.84), (0.08, 0.15, 0.39, 0.56), 0.08, 0.911)
    a = IO.load_start(cal)
    assert a == {"kernel_table_delta_um": (0.0, 0.28, 0.56, 0.84), "kernel_table_sigma_um": (0.08, 0.15, 0.39, 0.56),
                 "sigma_r0_um": 0.08, "gamma": 0.79}
    assert isinstance(a["kernel_table_delta_um"], tuple) and isinstance(a["gamma"], float)
    assert IO.load_start(cal, far)["gamma"] == 0.911
    bad = _kernel_json(tmp_path / "bad.json", (), (), 0, 0, key=None)
    with pytest.raises(ValueError) as e:
        IO.load_start(bad)
    assert "no fit" in str(e.value)                                       # the kernel_error in the message


def test_save_and_load_step2(tmp_path):
    r = dataclasses.replace(C.default_config().renderer, kernel_family="gaussian_table_spline",
                            kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84), kernel_table_sigma_um=(0.08, 0.16, 0.4, 0.57),
                            sigma_r0_um=0.08, kernel_continuation="linear", kernel_continuation_slope=0.85)
    report = {"gamma_hat": 0.85, "se": float("nan"), "per_twin": [{"id": 3, "ratio": float("nan")}, {"id": 4, "ratio": 1.1}]}
    p = tmp_path / "step2.json"
    IO.save_step2(p, r, report, meta={"note": "t"})
    text = p.read_text()
    assert "NaN" not in text                                               # strict JSON
    got = IO.load_step2(p)
    k = got["kernel"]
    # the keys notebook Cell 16 reads
    assert k["kernel_table_delta_um"] == [0.0, 0.28, 0.56, 0.84]
    assert k["kernel_table_sigma_um"] == [0.08, 0.16, 0.4, 0.57]
    assert k["sigma_r0_um"] == 0.08 and k["kernel_continuation_slope"] == 0.85
    assert k["kernel_family"] == "gaussian_table_spline"
    assert got["step2"] == {"gamma_hat": 0.85, "se": None, "per_twin": [{"id": 3, "ratio": None}, {"id": 4, "ratio": 1.1}]}
    assert got["meta"] == {"note": "t"}


def test_save_step2_numpy_nan_scalar(tmp_path):
    # a report built from numpy scalars (as Block 59 will build it) -- NaN as null for np.float64
    r = C.default_config().renderer
    p = tmp_path / "s.json"
    IO.save_step2(p, r, {"x": np.float64("nan"), "y": np.float64(1.5)})
    assert IO.load_step2(p)["step2"] == {"x": None, "y": 1.5}


# ---------------------------------------------------------------------------------------------------------------
# Amendment 3 (tester, 2026-11-27 22:11; SPEC Block 52 save_step2): "written as given, except that a float that is
# not finite -- NaN, +-inf -- becomes null and numpy scalars and arrays become Python numbers and lists". Oracles,
# expected values written by hand: (I7) NaN, +inf, -inf, as Python floats and as numpy float16/32/64 scalars and
# inside arrays, at any depth of dicts, lists and tuples, become null; the file is strict JSON (json.loads with a
# parse_constant that raises); (I8) numpy integers, bools and arrays become Python numbers, bools and (nested)
# lists, finite floats exact (float64 repr round trip; float32 values as their float64 value); (I9) the inputs are
# not mutated; (I10) the same rule holds in meta.
# ---------------------------------------------------------------------------------------------------------------

def _strict_load(p):
    def bad(c):
        raise AssertionError("non-strict JSON constant %r" % c)
    return json.loads(p.read_text(), parse_constant=bad)


def test_I7_I8_non_finite_and_numpy_values(tmp_path):
    r = C.default_config().renderer
    inf = float("inf")
    arr = np.array([[1.0, np.nan], [np.inf, -2.5]])
    report = {"a": inf, "b": -inf, "c": float("nan"), "d": np.float32("nan"), "e": np.float32(np.inf),
              "f": np.float16(-np.inf), "g": np.float64(-np.inf), "h": np.int64(7), "i": np.bool_(True),
              "j": arr, "k": (1.5, np.float32(0.25), [np.nan, {"x": np.int32(-3)}]), "l": np.arange(3),
              "m": np.float32(0.1), "n": 0.1, "o": np.array(np.nan), "p": [], "q": "text", "r": None}
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in report.items()}
    p = tmp_path / "s.json"
    IO.save_step2(p, r, report, meta={"inf": inf, "arr": np.array([np.nan, 2.0])})
    got = _strict_load(p)["step2"]
    want = {"a": None, "b": None, "c": None, "d": None, "e": None, "f": None, "g": None, "h": 7, "i": True,
            "j": [[1.0, None], [None, -2.5]], "k": [1.5, 0.25, [None, {"x": -3}]], "l": [0, 1, 2],
            "m": float(np.float32(0.1)), "n": 0.1, "o": None, "p": [], "q": "text", "r": None}
    assert got == want
    assert type(got["h"]) is int and type(got["i"]) is bool and type(got["l"][0]) is int
    assert IO.load_step2(p)["step2"] == want
    assert _strict_load(p)["meta"] == {"inf": None, "arr": [None, 2.0]}
    for k, v in before.items():                                            # (I9) nothing mutated
        if isinstance(v, np.ndarray):
            assert_array_equal(report[k], v)
        else:
            assert report[k] is v or report[k] == v or (isinstance(v, float) and math.isnan(v))


def test_I7_non_finite_kernel_value_is_null(tmp_path):
    # the kernel dict goes through the same rule ("in kernel, report and meta alike" in the code's docstring; the
    # SPEC states the rule for the report): an infinite slope must not crash the write
    r = dataclasses.replace(C.default_config().renderer, kernel_continuation_slope=float("inf"))
    p = tmp_path / "k.json"
    IO.save_step2(p, r, {})
    assert _strict_load(p)["kernel"]["kernel_continuation_slope"] is None


def test_I11_numpy_integer_dict_keys(tmp_path):
    # spec gap probe (not an oracle of the SPEC): a report keyed by node id as numpy integers, as a per-twin table
    # built from Twin.node_id arrays would be; the SPEC's "numpy scalars ... become Python numbers" does not say
    # whether keys are included. Recorded, not asserted either way.
    r = C.default_config().renderer
    p = tmp_path / "keys.json"
    try:
        IO.save_step2(p, r, {"per_node": {np.int64(5): 1.0}})
        outcome = "written: %s" % (_strict_load(p)["step2"],)
    except Exception as e:  # noqa: BLE001 -- the outcome is what is recorded
        outcome = "%s: %s" % (type(e).__name__, e)
    print("\nnumpy int64 dict key ->", outcome)
