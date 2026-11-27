"""Smoke test for step 2's files -- Block 52 in specs/SPEC.md (section 5, "Kernel calibration, step 2"):
loading.step2_io.read_chosen_nodes, load_cell (CellScans), load_start, save_step2 and load_step2.

Checks
    test_known_answer   read_chosen_nodes on a hand-written file (comments, blank lines, spaces, unordered rows,
                        two cells) gives the hand-written answer; load_start gives the tables written, with gamma
                        from the far fit when given and from the calibration kernel otherwise
    test_reference      round trip of a scans file written by calibration_io.save_scans from synthetic PlaneScans
                        and default_config's dictionary, as calibrate_kernel.py real writes its meta: every array
                        bit for bit (NaN where NaN), the node records, signature_hash("full") and the calibration
                        section (a non-default value and a tuple field included)
    test_convergence    not a discretisation (SkipTest)
    test_invariants     save_step2 then load_step2: kernel = kernel_to_dict + kernel_family, NaN -> null in the
                        report, meta kept; the kernel readable by the lines of the notebook's Cell 16
                        (notebooks/cleaning_pilot_2026-11-14.py, cleaning_config) that read
                        kernel_table_delta_um, kernel_table_sigma_um, sigma_r0_um, kernel_continuation_slope
    test_contract       types: dict[int, tuple[int, ...]]; CellScans fields; load_start's key set and types
    test_determinism    no randomness in this block (SkipTest); the synthetic inputs use a fixed seed
    test_edge_cases     each refusal: read_chosen_nodes (no header, a different header, a row of one or three
                        values, a non-integer, a duplicate node, no row), naming the line; load_cell (meta without
                        config / calibration / nodes, two scans or two records with one node id); load_start
                        (no kernel, its kernel_error in the message); save_step2 under amendment 3 (NaN, +-inf,
                        numpy float32 / int64 / bool / arrays in kernel, report and meta)

Run (from Passive Features/Diameter Re-measurement/)
    python tests/smoke/test_smoke_step2_io.py
    python -m pytest tests/smoke/test_smoke_step2_io.py -q

Pure ASCII, for the cluster.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.analysis.calibration import PlaneScan  # noqa: E402
from allen_diameter.loading import calibration_io  # noqa: E402
from allen_diameter.loading import step2_io as S2  # noqa: E402

SEED = 12345  # the synthetic scans' values only; the block has no randomness
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



# ----------------------------------------------------------- fixtures ---

def _scan(rng, node_id, which, n=7, k_star=10):
    ks = np.arange(k_star - n // 2, k_star - n // 2 + n, dtype=int)
    omega = rng.uniform(0.01, 0.2, n)
    omega[1] = np.nan
    status = tuple("ok" if i != 1 else "poor_fit" for i in range(n))
    return PlaneScan(int(node_id), 0.4, 0.05, 0.2, k_star, 2.83, ks, ks * 0.28, omega, rng.uniform(0.01, 0.3, n),
                     rng.uniform(-5, -1, n), rng.uniform(1, 9, n), rng.uniform(180, 220, n), status, which,
                     se_omega_um2=rng.uniform(1e-4, 1e-3, n), bg_sd_gl=rng.uniform(2, 4, n))


def _record(node_id, which):
    return dict(node_id=int(node_id), allen_diameter_um=0.4 if which == "near" else 0.7, set=which, reasons="",
                scanned=True, pilot=dict(mu_hat_per_um=1.3, cx_um=10.0, cy_um=-3.5, theta_rad=0.4, phi_rad=0.05,
                                         z_sub_um=2.83, k_star=10))


def _cfg_nondefault():
    cfg = default_config()
    cal = dataclasses.replace(cfg.calibration, far_min_k_offset=5, stripe_half_widths_um=(1.0, 3.0))
    return dataclasses.replace(cfg, calibration=cal)


def _write_scans(tmp, scans, records, cfg, drop=(), name="scans.json"):
    meta = dict(source="real", specimen="529878215", config=json.loads(cfg.to_json()),
                calibration=dataclasses.asdict(cfg.calibration), nodes=records)
    for k in drop:
        meta.pop(k)
    path = os.path.join(tmp, name)
    calibration_io.save_scans(scans, path, meta)
    return path


def _write_text(tmp, name, text):
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _calibration_json(tmp, name, kernel=None, error=""):
    obj = {"growth": {}, "meta": {}}
    if kernel is not None:
        obj["kernel"] = kernel
    else:
        obj["kernel_error"] = error
    path = os.path.join(tmp, name)
    with open(path, "w") as f:
        json.dump(obj, f)
    return path


_CAL_KERNEL = dict(kernel_table_delta_um=[0.0, 0.28, 0.56, 0.84], kernel_table_sigma_um=[0.08, 0.1536, 0.3924, 0.562],
                   sigma_r0_um=0.08, kernel_continuation="linear", kernel_continuation_slope=0.79)
_FAR_KERNEL = dict(_CAL_KERNEL, kernel_continuation_slope=0.4321)

CHOSEN_TEXT = """# chosen calibration nodes, D-261
specimen_id,node_id

529878215, 41
# a comment between rows
529878215,7
   555000111 ,3
529878215,19

555000111,1
"""


def _expect_value_error(func, *args, must=()):
    try:
        func(*args)
    except ValueError as exc:
        msg = str(exc)
        for m in must:
            assert m in msg, "message %r lacks %r" % (msg, m)
        return msg
    raise AssertionError("%s%r did not raise ValueError" % (getattr(func, "__name__", func), args))


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    with tempfile.TemporaryDirectory() as tmp:
        got = S2.read_chosen_nodes(_write_text(tmp, "chosen.csv", CHOSEN_TEXT))
        assert got == {529878215: (7, 19, 41), 555000111: (1, 3)}, got
        cal = _calibration_json(tmp, "calibration_1.json", _CAL_KERNEL)
        far = _calibration_json(tmp, "far_fit_1.json", _FAR_KERNEL)
        a = S2.load_start(cal)
        b = S2.load_start(cal, far)
    expect = dict(kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84), kernel_table_sigma_um=(0.08, 0.1536, 0.3924, 0.562),
                  sigma_r0_um=0.08)
    assert a == dict(expect, gamma=0.79), a
    assert b == dict(expect, gamma=0.4321), b


def test_reference():
    rng = make_rng()
    cfg = _cfg_nondefault()
    scans = [_scan(rng, 5, "near"), _scan(rng, 12, "far"), _scan(rng, 3, "near")]
    records = [_record(5, "near"), _record(12, "far"), _record(3, "near"), dict(_record(99, "near"), scanned=False)]
    with tempfile.TemporaryDirectory() as tmp:
        cs = S2.load_cell("529878215", _write_scans(tmp, scans, records, cfg))
    assert cs.cell_id == 529878215
    assert [s.node_id for s in cs.scans] == [5, 12, 3]
    for a, b in zip(cs.scans, scans):
        assert a.set == b.set and a.k_star == b.k_star and a.status == b.status
        for f in ("ks", "z_um", "omega_um2", "vw_um2", "area_gl_um", "depth_gl", "B_bar", "se_omega_um2", "bg_sd_gl"):
            assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f)   # bit for bit: JSON floats repr round trip
        for f in ("d_hat_um", "phi_rad", "alpha_hat", "z_sub_um"):
            assert getattr(a, f) == getattr(b, f), f
    assert cs.nodes == {r["node_id"]: r for r in records}, cs.nodes
    assert cs.config.signature_hash("full") == cfg.signature_hash("full")
    assert cs.config.signature_hash("full") == default_config().signature_hash("full")
    assert cs.config.calibration == cfg.calibration, "calibration section"
    assert cs.config.calibration.far_min_k_offset == 5
    assert cs.config.calibration.stripe_half_widths_um == (1.0, 3.0)
    assert cs.config.calibration != default_config().calibration
    assert cs.config == cfg


def test_convergence():
    raise unittest.SkipTest("file reading and writing: not a discretisation")


def test_invariants():
    rcfg = dataclasses.replace(default_config().renderer, kernel_family="gaussian_table_spline",
                               kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),
                               kernel_table_sigma_um=(0.08, 0.15, 0.39, 0.56), kernel_continuation="linear",
                               kernel_continuation_slope=0.55)
    report = {"stage1": {"theta_um": [0.15, 0.39, 0.56], "cost": float("nan"), "z_ax_um": {"5": 2.8}},
              "rows": [[1, float("nan")], (2, 0.5)], "ok": True, "message": "x"}
    meta = {"source": "test", "nan": float("nan")}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "step2_1.json")
        S2.save_step2(path, rcfg, report, meta)
        got = S2.load_step2(path)
        with open(path) as f:
            text = f.read()
        S2.save_step2(os.path.join(tmp, "nometa.json"), rcfg, {"a": 1})
        nometa = S2.load_step2(os.path.join(tmp, "nometa.json"))
        # the lines of Cell 16 (notebooks/cleaning_pilot_2026-11-14.py, cleaning_config) that read the kernel
        with open(path) as f:
            cal = json.load(f)
        k = cal.get("kernel")
        assert k, "Cell 16: no kernel"
        ren = {"kernel_table_delta_um": k["kernel_table_delta_um"], "kernel_table_sigma_um": k["kernel_table_sigma_um"],
               "sigma_r0_um": k["sigma_r0_um"]}
        with open(path) as f:
            far = json.load(f).get("kernel")
        assert far and "kernel_continuation_slope" in far
        ren["kernel_continuation_slope"] = float(far["kernel_continuation_slope"])
    assert "NaN" not in text and "null" in text
    assert set(got) == {"kernel", "step2", "meta"}, sorted(got)
    assert got["kernel"] == dict(calibration_io.kernel_to_dict(rcfg), kernel_family="gaussian_table_spline")
    assert got["step2"] == {"stage1": {"theta_um": [0.15, 0.39, 0.56], "cost": None, "z_ax_um": {"5": 2.8}},
                            "rows": [[1, None], [2, 0.5]], "ok": True, "message": "x"}, got["step2"]
    assert got["meta"] == {"source": "test", "nan": None}
    assert nometa["meta"] == {} and nometa["step2"] == {"a": 1}
    assert tuple(ren["kernel_table_delta_um"]) == rcfg.kernel_table_delta_um
    assert tuple(ren["kernel_table_sigma_um"]) == rcfg.kernel_table_sigma_um
    assert ren["kernel_continuation_slope"] == 0.55 and ren["sigma_r0_um"] == 0.08


def test_contract():
    rng = make_rng()
    with tempfile.TemporaryDirectory() as tmp:
        chosen = S2.read_chosen_nodes(_write_text(tmp, "chosen.csv", CHOSEN_TEXT))
        cs = S2.load_cell(7, _write_scans(tmp, [_scan(rng, 5, "near")], [_record(5, "near")], default_config()))
        st = S2.load_start(_calibration_json(tmp, "c.json", _CAL_KERNEL))
    assert all(type(k) is int and type(v) is tuple and all(type(x) is int for x in v) for k, v in chosen.items())
    assert all(list(v) == sorted(v) for v in chosen.values())
    assert isinstance(cs, S2.CellScans) and dataclasses.is_dataclass(cs)
    try:
        cs.cell_id = 3
        raise AssertionError("CellScans is not frozen")
    except dataclasses.FrozenInstanceError:
        pass
    assert type(cs.cell_id) is int and type(cs.scans) is tuple and isinstance(cs.scans[0], PlaneScan)
    assert type(cs.nodes) is dict and all(type(k) is int for k in cs.nodes) and type(cs.meta) is dict
    assert type(cs.config).__name__ == "DiameterConfig"
    assert set(st) == {"kernel_table_delta_um", "kernel_table_sigma_um", "sigma_r0_um", "gamma"}, sorted(st)
    assert type(st["kernel_table_delta_um"]) is tuple and type(st["kernel_table_sigma_um"]) is tuple
    assert type(st["sigma_r0_um"]) is float and type(st["gamma"]) is float
    assert all(math.isfinite(x) for x in st["kernel_table_sigma_um"])


def test_determinism():
    raise unittest.SkipTest("no randomness in this block (file reading and writing)")


def test_edge_cases():
    rng = make_rng()
    cfg = default_config()
    with tempfile.TemporaryDirectory() as tmp:
        bad = {
            "noheader.csv": ("529878215,7\n", ["line 1"]),
            "otherheader.csv": ("# c\nspecimen,node_id\n1,2\n", ["line 2", "header"]),
            "empty.csv": ("# only a comment\n\n", ["header"]),
            "onevalue.csv": ("specimen_id,node_id\n1,2\n529878215\n", ["line 3"]),
            "three.csv": ("specimen_id,node_id\n1,2,3\n", ["line 2"]),
            "float.csv": ("specimen_id,node_id\n1,2.0\n", ["line 2"]),
            "text.csv": ("specimen_id,node_id\nabc,2\n", ["line 2"]),
            "emptyfield.csv": ("specimen_id,node_id\n1,\n", ["line 2"]),
            "twice.csv": ("specimen_id,node_id\n1,2\n\n1, 2\n", ["line 4", "line 2"]),
            "norow.csv": ("specimen_id,node_id\n# nothing\n", ["no row"]),
        }
        for name, (text, must) in bad.items():
            _expect_value_error(S2.read_chosen_nodes, _write_text(tmp, name, text), must=must)
        # the same node id in two cells is not a duplicate
        ok = S2.read_chosen_nodes(_write_text(tmp, "two.csv", "specimen_id,node_id\n1,2\n3,2\n"))
        assert ok == {1: (2,), 3: (2,)}, ok
        scans = [_scan(rng, 5, "near"), _scan(rng, 6, "far")]
        records = [_record(5, "near"), _record(6, "far")]
        for k in ("config", "calibration", "nodes"):
            p = _write_scans(tmp, scans, records, cfg, drop=(k,), name="no_%s.json" % k)
            _expect_value_error(S2.load_cell, 1, p, must=[k])
        p = _write_scans(tmp, scans + [_scan(rng, 5, "near")], records, cfg, name="twoscans.json")
        _expect_value_error(S2.load_cell, 1, p, must=["scans", "5"])
        p = _write_scans(tmp, scans, records + [_record(6, "far")], cfg, name="tworecs.json")
        _expect_value_error(S2.load_cell, 1, p, must=["records", "6"])
        # amendment 3: non-finite floats -> null, numpy scalars and arrays -> Python numbers and lists
        rcfg = dataclasses.replace(cfg.renderer, kernel_continuation_slope=float("inf"))
        rep = {"inf": float("inf"), "minf": -np.inf, "nan": np.nan, "f32": np.float32(0.5), "i64": np.int64(7),
               "b": np.bool_(True), "f32nan": np.float32("nan"), "arr": np.array([[1.0, np.nan], [np.inf, 2.0]]),
               "iarr": np.arange(3, dtype=np.int32), "nested": [(np.float64(-np.inf), {"x": np.array(1.5)})]}
        p = os.path.join(tmp, "amend3.json")
        S2.save_step2(p, rcfg, rep, {"m": np.float32(np.inf), "a": np.zeros(2, dtype=np.float32)})
        got = S2.load_step2(p)
        assert got["kernel"]["kernel_continuation_slope"] is None, got["kernel"]
        assert got["step2"] == {"inf": None, "minf": None, "nan": None, "f32": 0.5, "i64": 7, "b": True,
                                "f32nan": None, "arr": [[1.0, None], [None, 2.0]], "iarr": [0, 1, 2],
                                "nested": [[None, {"x": 1.5}]]}, got["step2"]
        assert got["meta"] == {"m": None, "a": [0.0, 0.0]}, got["meta"]
        with open(p) as f:
            text = f.read()
        assert "NaN" not in text and "Infinity" not in text
        cal_bad = _calibration_json(tmp, "nokernel.json", None, error="insufficient: 4 near-field nodes")
        _expect_value_error(S2.load_start, cal_bad, must=["insufficient: 4 near-field nodes"])
        _expect_value_error(S2.load_start, cal_bad, _calibration_json(tmp, "far.json", _FAR_KERNEL),
                            must=["insufficient"])


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
