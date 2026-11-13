"""Smoke test for the features table -- Block 23 in specs/SPEC.md (wave 1; D-129, D-130).

Checks (Block 23's oracles; tolerances in brackets as SPEC.md lists them)
    test_known_answer   round 0 of a hand-made SWC equals Allen's radii doubled
                        and the trace, node by node as written by hand [exact]
    test_reference      featured_scene with zero offsets and d = 2 x Allen's
                        radius reproduces cell_scene's seg_xyz and
                        2 * seg_radius [exact]; max_relative_change against a
                        loop written by hand [1e-15]
    test_convergence    skipped: no discretisation
    test_invariants     an offset on one node moves exactly the segment ends
                        that carry its id, by the offset; with_rows leaves its
                        input unchanged and the untouched nodes identical;
                        write_features then read_features returns every array
                        bit for bit [exact]
    test_contract       shapes, dtypes, read-only arrays, FeaturedScene layout
    test_determinism    skipped: no randomness in the block
    test_edge_cases     ValueError for a non-finite entry, d_um <= 0,
                        mu_per_um < 0, an unknown node_id, an unknown source,
                        arrays of unequal length (and the I/O refusals)

Run
    cd "Passive Features/Diameter Re-measurement"
    python3 tests/smoke/test_smoke_node_features.py
    python3 -m pytest tests/smoke/test_smoke_node_features.py -q

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import importlib.metadata
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

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import node_features as nf  # noqa: E402
from allen_diameter.analysis.cell import cell_scene  # noqa: E402
from allen_diameter.analysis.plane_iteration import Scene  # noqa: E402
from allen_diameter.loading import features_io, swc_io  # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy",)
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run

TYPES = (2, 3, 4)   # D-130.O4: axon, basal, apical
RES0 = 0.1144       # um per full-resolution pixel (section 2.1)
TRANSFORM = {"shift_full_px": (12.0, -7.0), "flip_y_full_h": 900.0, "z0_um": 3.5}

# A hand-made cell: a soma (1) with a basal chain 2-3-4 that branches at 3
# (3-5), an apical chain 6-7, an axon 8-9, a second soma node 10 (child of 1,
# not traced as a neurite), and a dendrite root 20 (no parent) with child 21.
# Fields: id type x y z radius parent.
SAMPLE = (
    "# test_smoke_node_features\n"
    "1 1 10.0 10.0 5.0 4.0 -1\n"
    "2 3 14.0 10.0 5.0 0.55 1\n"
    "3 3 16.0 11.0 5.5 0.45 2\n"
    "4 3 18.0 11.0 6.0 0.35 3\n"
    "5 3 17.0 13.0 4.0 0.3 3\n"
    "6 4 10.0 5.0 5.0 0.8 1\n"
    "7 4 10.5 2.0 6.0 0.7 6\n"
    "8 2 7.0 10.0 5.0 0.25 1\n"
    "9 2 5.0 9.0 5.2 0.2 8\n"
    "10 1 9.0 10.0 5.0 3.0 1\n"
    "20 3 30.0 30.0 2.0 0.4 -1\n"
    "21 3 31.0 30.0 2.0 0.375 20\n"
)
# By hand: the nodes of TYPES with a parent in the file, in file order.
EXPECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 21]
EXPECT_TYPES = [3, 3, 3, 3, 4, 4, 2, 2, 3]
EXPECT_D = [1.1, 0.9, 0.7, 0.6, 1.6, 1.4, 0.5, 0.4, 0.75]


def _swc(tmp):
    p = os.path.join(tmp, "cell.swc")
    with open(p, "w", encoding="ascii", newline="") as f:
        f.write(SAMPLE)
    return swc_io.read_swc(p)


def _cell():
    with tempfile.TemporaryDirectory() as tmp:
        s = _swc(tmp)
    return s, cell_scene(s, TYPES, RES0, TRANSFORM)


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure."""
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


def _same(name, a, b):
    """Bit-for-bit equality: dtype, shape and every element's bytes."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, "%s: dtype %s vs %s" % (name, a.dtype, b.dtype)
    assert a.shape == b.shape, "%s: shape %s vs %s" % (name, a.shape, b.shape)
    assert a.tobytes() == b.tobytes(), "%s: arrays differ" % name


def _assert_features_same(a, b, label):
    for f in ("node_id", "swc_type", "d_um", "mu_per_um", "offset_um", "source", "flags"):
        _same("%s.%s" % (label, f), getattr(a, f), getattr(b, f))
    assert a.round_index == b.round_index, label


def _raises(func, *args, **kw):
    try:
        func(*args, **kw)
    except ValueError:
        return
    raise AssertionError("no ValueError from %s" % getattr(func, "__name__", func))


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    s, _scene = _cell()
    f = nf.initial_features(s, TYPES, 0.7)
    assert_array_equal(f.node_id, EXPECT_IDS)
    assert_array_equal(f.swc_type, EXPECT_TYPES)
    # exact: 2 * r is exact in binary floating point (a power-of-two scaling)
    rows = [s.index_of()[i] for i in EXPECT_IDS]
    _same("d = 2 r", f.d_um, 2.0 * s.radius[rows])
    check_close("d by hand", f.d_um, EXPECT_D, rtol=0, atol=0)
    assert np.all(f.mu_per_um == 0.7)
    assert np.all(f.offset_um == 0.0)
    assert f.source.tolist() == ["allen"] * len(EXPECT_IDS)
    assert f.flags.tolist() == [""] * len(EXPECT_IDS)
    assert f.round_index == 0
    # the trace: the scene's children are exactly the table's nodes
    _s, scene = _cell()
    assert sorted(scene.seg_ids[:, 0].tolist()) == sorted(EXPECT_IDS)


def test_reference():
    s, scene = _cell()
    f = nf.initial_features(s, TYPES, 0.9)
    fs = nf.featured_scene(scene, f)
    _same("seg_xyz", fs.seg_xyz, scene.seg_xyz)          # zero offsets: exact
    _same("seg_d_um", fs.seg_d_um, 2.0 * scene.seg_radius)  # d = 2 r: exact
    _same("seg_ids", fs.seg_ids, scene.seg_ids.astype(np.int64))
    _same("soma", fs.soma, scene.soma)
    # every end takes mu0: both the traced ends and the soma ends (child's mu)
    assert np.all(fs.seg_mu_per_um == 0.9)
    pos = {i: t for i, t in zip(EXPECT_IDS, EXPECT_TYPES)}
    assert fs.seg_type.tolist() == [pos[int(c)] for c in scene.seg_ids[:, 0]]

    # max_relative_change against a loop written by hand [1e-15]
    rng = make_rng()
    n = len(f)
    mu_a = rng.uniform(0.0, 2.0, n)
    mu_a[[1, 4]] = 0.0                    # nodes left out of the mu term
    a = nf.with_rows(f, [{"node_id": int(i), "mu_per_um": float(m)} for i, m in zip(f.node_id, mu_a)], 1)
    rows = []
    for k, i in enumerate(f.node_id):
        rows.append({"node_id": int(i), "d_um": float(a.d_um[k] * rng.uniform(0.8, 1.2)),
                     "mu_per_um": float(rng.uniform(0.0, 2.0)),
                     "offset_um": rng.normal(0.0, 0.3, 3), "source": "refined"})
    b = nf.with_rows(a, rows[::-1], 2)    # rows in another order: matched by id
    got = nf.max_relative_change(a, b)
    dd, dm, do = 0.0, 0.0, 0.0
    for k in range(n):
        dd = max(dd, abs(b.d_um[k] - a.d_um[k]) / a.d_um[k])
        if a.mu_per_um[k] > 0.0:
            dm = max(dm, abs(b.mu_per_um[k] - a.mu_per_um[k]) / a.mu_per_um[k])
        do = max(do, math.sqrt(sum((b.offset_um[k, j] - a.offset_um[k, j]) ** 2 for j in range(3))))
    # 1e-15 relative: the same IEEE operations, at most one rounding apart
    check_close("mrc d", got["d"], dd, rtol=1e-15)
    check_close("mrc mu", got["mu"], dm, rtol=1e-15)
    check_close("mrc offset", got["offset_um"], do, rtol=1e-15)
    assert set(got) == {"d", "mu", "offset_um"}
    # the identity gives zero; the mu term is NaN when no node has mu > 0
    z = nf.max_relative_change(b, b)
    assert z == {"d": 0.0, "mu": 0.0, "offset_um": 0.0}, z
    zero = nf.initial_features(s, TYPES, 0.0)
    assert math.isnan(nf.max_relative_change(zero, zero)["mu"])


def test_convergence():
    raise unittest.SkipTest("no discretisation: tables and copies")


def test_invariants():
    s, scene = _cell()
    f0 = nf.initial_features(s, TYPES, 0.5)
    # an offset on one node moves exactly the segment ends that carry its id
    for nid in (3, 6, 21):
        o = np.array([0.25, -0.125, 0.375])
        f1 = nf.with_rows(f0, [{"node_id": nid, "offset_um": o}], 1)
        fs = nf.featured_scene(scene, f1)
        moved = np.zeros((scene.seg_ids.shape[0], 2), dtype=bool)
        moved[:, 0] = scene.seg_ids[:, 1] == nid      # end 0 = parent end
        moved[:, 1] = scene.seg_ids[:, 0] == nid      # end 1 = child end
        assert moved.any(), nid
        for q in range(moved.shape[0]):
            for e in range(2):
                old, new = scene.seg_xyz[q, 3 * e:3 * e + 3], fs.seg_xyz[q, 3 * e:3 * e + 3]
                want = old + o if moved[q, e] else old
                _same("offset %d seg %d end %d" % (nid, q, e), new, want)
    # node 2's parent is the soma: its parent end keeps the scene's position and
    # diameter and takes the child's mu
    f2 = nf.with_rows(f0, [{"node_id": 2, "d_um": 2.5, "mu_per_um": 1.75, "offset_um": [1.0, 1.0, 1.0]}], 1)
    fs = nf.featured_scene(scene, f2)
    q = int(np.flatnonzero(scene.seg_ids[:, 0] == 2)[0])
    _same("soma end xyz", fs.seg_xyz[q, :3], scene.seg_xyz[q, :3])
    assert fs.seg_d_um[q, 0] == 2.0 * scene.seg_radius[q, 0] and fs.seg_d_um[q, 1] == 2.5
    assert fs.seg_mu_per_um[q, 0] == 1.75 and fs.seg_mu_per_um[q, 1] == 1.75
    # the root 20 (type 3, no parent) is not in the table: the same rule
    q = int(np.flatnonzero(scene.seg_ids[:, 0] == 21)[0])
    f3 = nf.with_rows(f0, [{"node_id": 21, "mu_per_um": 0.25}], 1)
    fs = nf.featured_scene(scene, f3)
    assert fs.seg_mu_per_um[q].tolist() == [0.25, 0.25]
    assert fs.seg_d_um[q, 0] == 2.0 * scene.seg_radius[q, 0]

    # with_rows leaves its input unchanged and the untouched nodes identical
    snap = {k: np.array(getattr(f0, k), copy=True) for k in
            ("node_id", "swc_type", "d_um", "mu_per_um", "offset_um", "source", "flags")}
    rows = [{"node_id": 4, "d_um": 0.65, "source": "pilot", "flags": "faint;crossing"},
            {"node_id": 8, "mu_per_um": 2.0, "offset_um": (0.1, 0.2, 0.3), "source": "fine"}]
    g = nf.with_rows(f0, rows, 3)
    for k, v in snap.items():
        _same("input %s" % k, getattr(f0, k), v)
    assert f0.round_index == 0 and g.round_index == 3
    touched = np.isin(g.node_id, [4, 8])
    for k in ("node_id", "swc_type", "d_um", "mu_per_um", "offset_um"):
        _same("carried %s" % k, getattr(g, k)[~touched], getattr(f0, k)[~touched])
    for k in ("source", "flags"):  # string arrays: the width follows the longest string held
        assert getattr(g, k)[~touched].tolist() == getattr(f0, k)[~touched].tolist(), k
    k4, k8 = g.index_of()[4], g.index_of()[8]
    assert g.d_um[k4] == 0.65 and g.source[k4] == "pilot" and g.flags[k4] == "faint;crossing"
    assert g.mu_per_um[k4] == f0.mu_per_um[k4] and np.all(g.offset_um[k4] == 0.0)
    assert g.mu_per_um[k8] == 2.0 and g.offset_um[k8].tolist() == [0.1, 0.2, 0.3] and g.source[k8] == "fine"
    assert g.d_um[k8] == f0.d_um[k8]

    # write_features then read_features: every array bit for bit [exact]
    rng = make_rng()
    n = len(f0)
    awkward = [{"node_id": int(i), "d_um": float(d), "mu_per_um": float(m), "offset_um": o,
                "source": src, "flags": fl}
               for i, d, m, o, src, fl in zip(
                   f0.node_id, rng.lognormal(0.0, 1.0, n), rng.uniform(0.0, 3.0, n),
                   rng.normal(0.0, 1.0, (n, 3)), [nf.SOURCES[k % 4] for k in range(n)],
                   ['', 'a,b', 'say "x"', 'faint;crossing', 'x\ny', ' pad ', '', 'overlap_in_projection', ';'])]
    awkward[0].update(d_um=0.1, mu_per_um=1.0 / 3.0, offset_um=(-0.0, 5e-324, 1e300))
    awkward[1].update(d_um=5e-324, mu_per_um=0.0, offset_um=(1e-300, -1e-300, 2.0 ** 0.5))
    h = nf.with_rows(f0, awkward, 7)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "features.csv")
        features_io.write_features(p, h)
        back = features_io.read_features(p)
        with open(p, "rb") as fh:
            raw = fh.read()
    _assert_features_same(back, h, "round trip")
    assert b"\r\n" not in raw.replace(b'"x\ny"', b""), "LF line endings"
    assert np.signbit(back.offset_um[0, 0]), "-0.0 kept"


def test_contract():
    s, scene = _cell()
    f = nf.initial_features(s, TYPES, 0.5)
    n = len(EXPECT_IDS)
    assert len(f) == n
    for k, shape, kind in (("node_id", (n,), "i"), ("swc_type", (n,), "i"), ("d_um", (n,), "f"),
                           ("mu_per_um", (n,), "f"), ("offset_um", (n, 3), "f"),
                           ("source", (n,), "U"), ("flags", (n,), "U")):
        a = getattr(f, k)
        assert a.shape == shape, (k, a.shape)
        assert a.dtype.kind == kind, (k, a.dtype)
        assert not a.flags.writeable, k
    assert f.node_id.dtype == np.int64 and f.swc_type.dtype == np.int64
    assert f.d_um.dtype == np.float64 and f.offset_um.dtype == np.float64
    assert isinstance(f.round_index, int)
    try:
        f.d_um = f.d_um
    except Exception:
        pass
    else:
        raise AssertionError("NodeFeatures is not frozen")
    fs = nf.featured_scene(scene, f)
    m = scene.seg_xyz.shape[0]
    for k, shape, dt in (("seg_xyz", (m, 6), np.float64), ("seg_d_um", (m, 2), np.float64),
                         ("seg_mu_per_um", (m, 2), np.float64), ("seg_ids", (m, 2), np.int64),
                         ("seg_type", (m,), np.int64), ("soma", (2, 3), np.float64)):
        a = getattr(fs, k)
        assert a.shape == shape and a.dtype == dt, (k, a.shape, a.dtype)
    assert np.all(np.isfinite(fs.seg_xyz)) and np.all(fs.seg_d_um > 0) and np.all(fs.seg_mu_per_um >= 0)
    # featured_scene does not write into the scene
    assert fs.seg_xyz is not scene.seg_xyz and fs.soma is not scene.soma


def test_determinism():
    raise unittest.SkipTest("no randomness in the block")


def test_edge_cases():
    s, scene = _cell()
    f = nf.initial_features(s, TYPES, 0.5)
    good = dict(node_id=[1, 2], swc_type=[3, 3], d_um=[0.5, 0.6], mu_per_um=[0.0, 1.0],
                offset_um=np.zeros((2, 3)), source=["allen", "fine"], flags=["", "x"], round_index=0)
    nf.NodeFeatures(**good)

    def bad(**kw):
        args = dict(good)
        args.update(kw)
        _raises(nf.NodeFeatures, **args)

    # non-finite entries
    bad(d_um=[np.nan, 0.6]); bad(d_um=[np.inf, 0.6]); bad(mu_per_um=[np.inf, 0.0])
    bad(offset_um=[[0, 0, np.nan], [0, 0, 0]])
    # d <= 0, mu < 0
    bad(d_um=[0.0, 0.6]); bad(d_um=[-0.1, 0.6]); bad(mu_per_um=[-1e-12, 0.0])
    # unknown source
    bad(source=["allen", "guess"])
    # arrays of unequal length / shape
    bad(d_um=[0.5]); bad(source=["allen"]); bad(flags=["", "", ""]); bad(offset_um=np.zeros((2, 2)))
    bad(swc_type=[3]); bad(mu_per_um=[0.0, 1.0, 2.0])
    # a repeated id, a negative or non-integer round
    bad(node_id=[1, 1]); bad(round_index=-1); bad(round_index=1.5)
    # through the public functions
    _raises(nf.initial_features, s, TYPES, -0.1)
    _raises(nf.initial_features, s, TYPES, float("nan"))
    _raises(nf.with_rows, f, [{"node_id": 999, "d_um": 1.0}], 1)       # unknown node_id
    _raises(nf.with_rows, f, [{"node_id": 2, "source": "guess"}], 1)   # unknown source
    _raises(nf.with_rows, f, [{"node_id": 2, "d_um": 0.0}], 1)
    _raises(nf.with_rows, f, [{"node_id": 2, "d_um": float("nan")}], 1)
    _raises(nf.with_rows, f, [{"node_id": 2, "mu_per_um": -1.0}], 1)
    _raises(nf.with_rows, f, [{"node_id": 2, "offset_um": [0.0, float("inf"), 0.0]}], 1)
    _raises(nf.with_rows, f, [{"node_id": 2, "offset_um": [0.0, 0.0]}], 1)
    _raises(nf.with_rows, f, [{"node_id": 2, "diameter": 1.0}], 1)     # unknown key
    _raises(nf.with_rows, f, [{"d_um": 1.0}], 1)                       # no node_id
    _raises(nf.with_rows, f, [{"node_id": 2}, {"node_id": 2}], 1)      # named twice
    # a segment whose child is not in the table: unknown node_id
    short = nf.NodeFeatures(f.node_id[1:], f.swc_type[1:], f.d_um[1:], f.mu_per_um[1:],
                            f.offset_um[1:], f.source[1:], f.flags[1:], 0)
    _raises(nf.featured_scene, scene, short)
    _raises(nf.max_relative_change, f, short)
    # the empty scene and an empty-node table
    empty_scene = Scene(np.empty((0, 6)), np.empty((0, 2)), np.empty((0, 2), dtype=np.int64), np.empty((0, 3)))
    es = nf.featured_scene(empty_scene, f)
    assert es.seg_xyz.shape == (0, 6) and es.seg_type.shape == (0,)
    e = nf.NodeFeatures([], [], [], [], np.empty((0, 3)), [], [], 0)
    assert len(e) == 0
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "f.csv")
        _raises(features_io.write_features, p, e)
        with open(p, "w") as fh:
            fh.write("node_id,d_um\n1,0.5\n")
        _raises(features_io.read_features, p)                           # wrong header
        features_io.write_features(p, f)
        with open(p) as fh:
            lines = fh.read().splitlines()
        with open(p, "w") as fh:
            fh.write("\n".join(lines[:2] + [lines[2].rsplit(",", 1)[0] + ",1"] + lines[3:]) + "\n")
        _raises(features_io.read_features, p)                           # round differs
        with open(p, "w") as fh:
            fh.write(lines[0] + "\n")
        _raises(features_io.read_features, p)                           # no row
        with open(p, "w") as fh:
            fh.write("\n".join([lines[0], lines[1].replace("allen", "guess")]) + "\n")
        _raises(features_io.read_features, p)                           # unknown source


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
