"""Smoke test for the screening views and files -- Block 61 in specs/SPEC.md (section 5, "The screening export"):
analysis.screening (apt_nodes, crop_bounds, node_view, diameter_marks, image_extent, index_row) and
loading.screening_io (save_views, load_views, write_index, write_candidates).

Checks (the entry's oracles V1 to V7)
    test_known_answer   (V1) node_view on a 33-plane far scan and a 7-plane near scan: profile_gl = bhat_gl - w_gl
                        and background_gl = bhat_gl bit for bit (NaN where NaN), stack and ks unchanged, the pilot's
                        values carried; (V2) diameter_marks for theta = 0 and pi/2 [1e-12]
    test_reference      (V3) crop_bounds equals calibration_scene.patch_bounds with the set's half-width and
                        res0_um; image_extent puts the pilot's centre within half a pixel of the crop's middle and
                        equals the outer pixel edges of calibration_scene.pixel_centres
    test_convergence    not a discretisation (SkipTest)
    test_invariants     (V2) the marks' distance is D for any theta [1e-12], their midpoint the pilot's centre, and
                        they lie on the measuring line at v = -+D/2; index_row's values on a hand-computed case
    test_contract       (V4) save_views then load_views: every field bit for bit, types and dtypes kept; (V5)
                        write_candidates read by step2_io.read_chosen_nodes gives every node id of the cell, the
                        file's rows near first then by node id; write_index's rows read by csv.DictReader equal
                        index_row's values, columns in INDEX_COLUMNS order
    test_determinism    no randomness in this block (SkipTest); the synthetic inputs use a fixed seed
    test_edge_cases     (V6) each refusal: apt_nodes (a scan without a record, a record without a pilot), node_view
                        (patch ks not the scan's, no v_um / w_gl / bhat_gl, non-finite or None allen_diameter_um);
                        a plane with valid False has its image zeros; (V7) apt_nodes' order, records as a list and
                        as a dict

Run (from Passive Features/Diameter Re-measurement/)
    python tests/smoke/test_smoke_screening.py
    python -m pytest tests/smoke/test_smoke_screening.py -q

Pure ASCII, for the cluster.
"""
from __future__ import annotations

import csv
import dataclasses
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from allen_image_io import CropFrame  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.analysis.calibration import PlaneScan  # noqa: E402
from allen_diameter.analysis import calibration_scene  # noqa: E402
from allen_diameter.analysis import screening as SC  # noqa: E402
from allen_diameter.loading import screening_io as SIO  # noqa: E402
from allen_diameter.loading import step2_io as S2  # noqa: E402

SEED = 12345  # the synthetic inputs' values only; the block has no randomness
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


# ------------------------------------------------------------- fixtures ---

CELL_ID = 501
N_V = 41


def make_scan(node_id, which, k_star, rng, nan_plane=True):
    """A synthetic PlaneScan with the per-plane profiles: 7 planes (near) or 33 (far) about k_star."""
    P = 3 if which == "near" else 16
    ks = np.arange(k_star - P, k_star + P + 1, dtype=np.int64)
    n = ks.size
    reach = 3.0 if which == "near" else 10.0
    v = np.linspace(-reach, reach, N_V)
    bhat = 180.0 + rng.normal(0.0, 3.0, (n, N_V))
    w = 60.0 * np.exp(-v[None, :] ** 2 / 0.5) + rng.normal(0.0, 1.0, (n, N_V))
    if nan_plane:
        w[0, :5] = np.nan          # a plane where the scan has no samples at the line's start
        bhat[1, -3:] = np.nan
    status = tuple("ok" if q % 4 else "no_dip" for q in range(n))
    omega = np.where(np.array([s == "ok" for s in status]), rng.uniform(0.1, 2.0, n), np.nan)
    return PlaneScan(node_id=int(node_id), d_hat_um=1.0, phi_rad=0.2, alpha_hat=0.5, k_star=int(k_star),
                     z_sub_um=float(k_star) * 0.28, ks=ks, z_um=ks * 0.28, omega_um2=omega,
                     vw_um2=rng.uniform(0.1, 2.0, n), area_gl_um=rng.uniform(1, 9, n), depth_gl=rng.uniform(5, 60, n),
                     B_bar=rng.uniform(150, 200, n), status=status, set=which, v_um=v, w_gl=w, bhat_gl=bhat)


def make_record(node_id, which, d_a, cx=812.37, cy=455.91, theta=0.7, phi=0.15):
    pilot = {"k_star": 40, "z_sub_um": 11.2, "cx_um": cx, "cy_um": cy, "theta_rad": theta, "phi_rad": phi,
             "B_bar": 170.0, "mu_hat_per_um": 0.1, "d_hat_um": 1.0, "alpha_hat": 0.5, "fit_status": "ok",
             "flags": [], "reg_verdict": "ok"}
    return {"node_id": int(node_id), "allen_diameter_um": d_a, "set": which, "reasons": "", "scanned": True,
            "pilot": pilot}


def make_patch(scan, record, cfg, rng, invalid=()):
    """(block, ks, valid, frame) in the fetch_zblock contract over the scan's planes."""
    left, top, width, height = SC.crop_bounds(record, scan.set, cfg)
    ks = np.asarray(scan.ks).copy()
    block = rng.integers(0, 256, (ks.size, height, width), dtype=np.uint8)
    valid = np.ones(ks.size, dtype=bool)
    for q in invalid:
        valid[q] = False
        block[q] = 255             # fetch_zblock leaves a missing section white
    return block, ks, valid, CropFrame(int(left), int(top), 0, float(cfg.acquisition.res0_um))


def build_views(cfg, rng):
    """Three views of one cell: a far node and two near nodes."""
    spec = [(31, "far", 2.4, 40), (12, "near", 0.8, 38), (7, "near", 1.1, 41)]
    out = []
    for nid, which, d_a, k_star in spec:
        sc = make_scan(nid, which, k_star, rng)
        rec = make_record(nid, which, d_a)
        out.append(SC.node_view(CELL_ID, sc, rec, make_patch(sc, rec, cfg, rng)))
    return out


def _view_with(view, **kw):
    return dataclasses.replace(view, **kw)


def assert_views_equal(a, b, label):
    for f in dataclasses.fields(SC.NodeView):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            assert isinstance(y, np.ndarray), "%s.%s: not an array" % (label, f.name)
            assert x.dtype == y.dtype and x.shape == y.shape, "%s.%s: %s %s vs %s %s" % (
                label, f.name, x.dtype, x.shape, y.dtype, y.shape)
            assert x.tobytes() == y.tobytes(), "%s.%s: bytes differ" % (label, f.name)
        elif isinstance(x, float):
            assert type(y) is float and (x == y or (math.isnan(x) and math.isnan(y))), (
                "%s.%s: %r vs %r" % (label, f.name, x, y))
        else:
            assert type(x) is type(y) and x == y, "%s.%s: %r vs %r" % (label, f.name, x, y)


def expect_value_error(fn, needle, label):
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), "%s: message %r lacks %r" % (label, str(exc), needle)
        return
    raise AssertionError("%s: no ValueError" % label)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = default_config()
    rng = make_rng()
    for nid, which, n_expected in ((31, "far", 33), (12, "near", 7)):
        sc = make_scan(nid, which, 40, rng)
        rec = make_record(nid, which, 1.3)
        patch = make_patch(sc, rec, cfg, rng)
        block_before, ks_before = patch[0].copy(), patch[1].copy()
        v = SC.node_view(CELL_ID, sc, rec, patch)
        label = "V1 %s" % which
        assert v.ks.size == n_expected, (label, v.ks.size)
        assert v.ks.dtype == np.int64 and v.stack.dtype == np.uint8 and v.valid.dtype == bool, label
        assert v.profile_gl.dtype == np.float64 and v.background_gl.dtype == np.float64, label
        expected_profile = np.asarray(sc.bhat_gl) - np.asarray(sc.w_gl)
        assert (expected_profile.tobytes() == v.profile_gl.tobytes()), label + ": profile_gl not bit for bit"
        assert np.asarray(sc.bhat_gl).tobytes() == v.background_gl.tobytes(), label + ": background_gl"
        assert np.isnan(v.profile_gl[0, :5]).all() and np.isnan(v.profile_gl[1, -3:]).all(), label + ": NaN kept"
        assert_array_equal(v.stack, block_before, err_msg=label + ": stack")
        assert_array_equal(v.ks, ks_before, err_msg=label + ": ks")
        assert_array_equal(patch[0], block_before, err_msg=label + ": the patch was modified")
        assert v.v_um.tobytes() == np.asarray(sc.v_um, dtype=np.float64).tobytes(), label + ": v_um"
        assert v.status == sc.status and v.set == which and v.k_star == 40 and v.node_id == nid, label
        assert v.omega_um2.tobytes() == sc.omega_um2.tobytes() and v.B_bar.tobytes() == sc.B_bar.tobytes(), label
        p = rec["pilot"]
        assert (v.cx_um, v.cy_um, v.theta_rad, v.phi_rad) == (p["cx_um"], p["cy_um"], p["theta_rad"], p["phi_rad"])
        assert (v.left, v.top) == (patch[3].left, patch[3].top) and v.res0_um == cfg.acquisition.res0_um, label
        assert v.allen_diameter_um == 1.3 and v.cell_id == CELL_ID, label
    # (V2) the marks for theta = 0 and pi/2; float64 roundoff of cos(pi/2) ~ 6e-17 times D/2: tolerance 1e-12
    cx, cy, d = 812.37, 455.91, 1.7
    sc5, rec5 = make_scan(5, "near", 40, rng), make_record(5, "near", d)
    base = SC.node_view(CELL_ID, sc5, rec5, make_patch(sc5, rec5, cfg, rng))
    m0 = SC.diameter_marks(_view_with(base, theta_rad=0.0))
    assert m0["v_um"] == (-d / 2, d / 2), m0["v_um"]
    check_close("V2 theta 0", m0["xy_um"], [[cx, cy - d / 2], [cx, cy + d / 2]], rtol=0, atol=1e-12)
    m1 = SC.diameter_marks(_view_with(base, theta_rad=math.pi / 2))
    check_close("V2 theta pi/2", m1["xy_um"], [[cx + d / 2, cy], [cx - d / 2, cy]], rtol=0, atol=1e-12)
    assert m1["xy_um"].shape == (2, 2) and m1["xy_um"].dtype == np.float64


def test_reference():
    cfg = default_config()
    res = cfg.acquisition.res0_um
    for which, half in (("near", cfg.survey.screening_near_half_um), ("far", cfg.survey.screening_far_half_um)):
        for cx, cy in ((812.37, 455.91), (100.0, 57.2), (3.05, 1234.5678)):
            rec = make_record(1, which, 1.0, cx=cx, cy=cy)
            got = SC.crop_bounds(rec, which, cfg)
            ref = calibration_scene.patch_bounds(cx, cy, half, res)
            assert got == ref, "V3 crop_bounds %s (%r, %r): %s vs %s" % (which, cx, cy, got, ref)
            left, top, width, height = got
            view = SC.NodeView(cell_id=1, node_id=1, set=which, allen_diameter_um=1.0, k_star=0,
                               ks=np.zeros(1, np.int64), stack=np.zeros((1, height, width), np.uint8),
                               valid=np.ones(1, bool), left=left, top=top, res0_um=res, cx_um=cx, cy_um=cy,
                               theta_rad=0.0, phi_rad=0.0, v_um=np.zeros(3), profile_gl=np.zeros((1, 3)),
                               background_gl=np.zeros((1, 3)), status=("ok",), omega_um2=np.zeros(1),
                               B_bar=np.zeros(1))
            x0, x1, y0, y1 = SC.image_extent(view)
            assert abs(0.5 * (x0 + x1) - cx) <= 0.5 * res + 1e-9, ("V3 x middle", which, x0, x1, cx)
            assert abs(0.5 * (y0 + y1) - cy) <= 0.5 * res + 1e-9, ("V3 y middle", which, y0, y1, cy)
            px, py = calibration_scene.pixel_centres((height, width), CropFrame(left, top, 0, res))
            # the outer edges are half a pixel beyond the outermost pixel centres; float64 roundoff of ~1e3 um
            check_close("V3 extent", [x0, x1, y0, y1], [px[0, 0] - res / 2, px[0, -1] + res / 2,
                                                         py[0, 0] - res / 2, py[-1, 0] + res / 2], rtol=0, atol=1e-9)
            assert x0 < x1 and y0 < y1, "V3: y0 is the top edge (smaller y), y increasing downwards"


def test_convergence():
    raise unittest.SkipTest("not a discretisation: the block copies, subtracts and writes arrays")


def test_invariants():
    cfg = default_config()
    rng = make_rng()
    sc = make_scan(9, "far", 40, rng)
    rec = make_record(9, "far", 3.1)
    v = SC.node_view(CELL_ID, sc, rec, make_patch(sc, rec, cfg, rng))
    for theta in np.linspace(-2 * math.pi, 2 * math.pi, 37):
        m = SC.diameter_marks(_view_with(v, theta_rad=float(theta)))
        xy = m["xy_um"]
        # (V2) float64 roundoff on coordinates ~1e3 um: |error| ~ 1e-13, tolerance 1e-12
        check_close("V2 distance", np.hypot(*(xy[1] - xy[0])), 3.1, rtol=0, atol=1e-12)
        check_close("V2 midpoint", xy.mean(axis=0), [v.cx_um, v.cy_um], rtol=0, atol=1e-12)
        u = np.array([-math.sin(theta), math.cos(theta)])
        check_close("V2 on the line", (xy - [v.cx_um, v.cy_um]) @ u, m["v_um"], rtol=0, atol=1e-12)
    row = SC.index_row(v)
    q = int(np.flatnonzero(v.ks == 40)[0])
    expected = {"specimen_id": CELL_ID, "node_id": 9, "set": "far", "allen_diameter_um": 3.1, "k_star": 40,
                "phi_deg": math.degrees(0.15), "n_planes": 33, "n_ok": sum(1 for s in sc.status if s == "ok"),
                "B_focus_gl": float(sc.B_bar[q]), "omega_focus_um2": float(sc.omega_um2[q])}
    assert tuple(row) == SIO.INDEX_COLUMNS, tuple(row)
    for k, val in expected.items():
        assert type(row[k]) is type(val), (k, type(row[k]), type(val))
        assert row[k] == val or (isinstance(val, float) and math.isnan(val) and math.isnan(row[k])), (k, row[k], val)
    assert row["n_ok"] == 24, row["n_ok"]      # every 4th of 33 planes is "no_dip": 9 of them
    # k* absent from ks: the focus values are NaN
    r2 = SC.index_row(_view_with(v, k_star=99))
    assert math.isnan(r2["B_focus_gl"]) and math.isnan(r2["omega_focus_um2"]), r2


def test_contract():
    cfg = default_config()
    rng = make_rng()
    views = build_views(cfg, rng)
    # a view with an unfetched plane, a NaN pilot value and an empty-ish status variety
    sc = make_scan(44, "near", 39, rng)
    rec = make_record(44, "near", 0.6)
    rec["pilot"]["phi_rad"] = None
    views.append(SC.node_view(CELL_ID, sc, rec, make_patch(sc, rec, cfg, rng, invalid=(2,))))
    with tempfile.TemporaryDirectory() as tmp:
        # (V4) round trip
        path = os.path.join(tmp, "screening_%d.npz" % CELL_ID)
        SIO.save_views(path, views)
        back = SIO.load_views(path)
        assert len(back) == len(views), (len(back), len(views))
        for q, (a, b) in enumerate(zip(views, back)):
            assert_views_equal(a, b, "V4 view %d" % q)
        assert os.listdir(tmp) == [os.path.basename(path)], os.listdir(tmp)
        # an empty cell
        SIO.save_views(os.path.join(tmp, "empty.npz"), [])
        assert SIO.load_views(os.path.join(tmp, "empty.npz")) == []
        # (V5) candidates
        cpath = os.path.join(tmp, "candidates_%d.csv" % CELL_ID)
        SIO.write_candidates(cpath, views)
        chosen = S2.read_chosen_nodes(cpath)
        assert chosen == {CELL_ID: tuple(sorted(v.node_id for v in views))}, chosen
        with open(cpath, "rb") as f:
            raw = f.read()
        assert b"\r" not in raw, "candidates: LF line endings"
        lines = raw.decode("ascii").splitlines()
        assert lines[0] == "specimen_id,node_id", lines[0]
        assert lines[1:] == ["%d,%d" % (CELL_ID, n) for n in (7, 12, 44, 31)], "V5: near first, then by node id"
        # (V5) index
        rows = [SC.index_row(v) for v in views]
        ipath = os.path.join(tmp, "screening_index_%d.csv" % CELL_ID)
        SIO.write_index(ipath, rows)
        with open(ipath, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            assert tuple(rd.fieldnames) == SIO.INDEX_COLUMNS, rd.fieldnames
            got = list(rd)
        assert len(got) == len(rows)
        for q, (g, r) in enumerate(zip(got, rows)):
            for k in SIO.INDEX_COLUMNS:
                x = r[k]
                if isinstance(x, str):
                    assert g[k] == x, (q, k, g[k], x)
                elif isinstance(x, int):
                    assert int(g[k]) == x, (q, k, g[k], x)
                else:
                    y = float(g[k])
                    assert y == x or (math.isnan(x) and math.isnan(y)), (q, k, g[k], x)
        assert math.isnan(float(got[3]["phi_deg"])), "the None pilot phi_rad is NaN in the index"
        expect_value_error(lambda: SIO.write_index(ipath, [dict(rows[0], extra=1)]), "extra", "write_index extra")
        r_missing = dict(rows[0])
        del r_missing["n_ok"]
        expect_value_error(lambda: SIO.write_index(ipath, [r_missing]), "n_ok", "write_index missing")


def test_determinism():
    raise unittest.SkipTest("no randomness in this block; the synthetic inputs use a fixed seed")


def test_edge_cases():
    cfg = default_config()
    rng = make_rng()
    # (V6) node_view's refusals
    sc = make_scan(21, "near", 40, rng)
    rec = make_record(21, "near", 0.9)
    block, ks, valid, frame = make_patch(sc, rec, cfg, rng)
    expect_value_error(lambda: SC.node_view(CELL_ID, sc, rec, (block, ks + 1, valid, frame)), "node 21",
                       "patch ks shifted")
    expect_value_error(lambda: SC.node_view(CELL_ID, sc, rec, (block[:-1], ks[:-1], valid[:-1], frame)),
                       "planes", "patch ks shorter")
    for name in ("v_um", "w_gl", "bhat_gl"):
        expect_value_error(lambda: SC.node_view(CELL_ID, dataclasses.replace(sc, **{name: None}), rec,
                                                (block, ks, valid, frame)), name, "no " + name)
    for d_a in (None, float("nan"), float("inf")):
        expect_value_error(lambda: SC.node_view(CELL_ID, sc, dict(rec, allen_diameter_um=d_a),
                                                (block, ks, valid, frame)), "allen_diameter_um", "D_A %r" % d_a)
    # a plane that could not be fetched: its image zeros, the others unchanged
    patch = make_patch(sc, rec, cfg, rng, invalid=(0, 4))
    v = SC.node_view(CELL_ID, sc, rec, patch)
    assert not v.valid[0] and not v.valid[4] and v.valid[[1, 2, 3, 5, 6]].all(), v.valid
    assert not v.stack[[0, 4]].any(), "an unfetched plane's image is zeros"
    assert_array_equal(v.stack[[1, 2, 3, 5, 6]], patch[0][[1, 2, 3, 5, 6]])
    assert (patch[0][0] == 255).all(), "the patch itself is not modified"
    # (V6) apt_nodes' refusals
    s1, s2 = make_scan(3, "near", 40, rng, False), make_scan(8, "far", 40, rng, False)
    r1, r2 = make_record(3, "near", 0.7), make_record(8, "far", 2.0)
    expect_value_error(lambda: SC.apt_nodes([s1, s2], [r1]), "node 8", "scan without a record")
    expect_value_error(lambda: SC.apt_nodes([s1, s2], [r1, dict(r2, pilot=None)]), "node 8", "record without pilot")
    # (V7) apt_nodes' order: near first, then node id; records not scanned are ignored; list or dict of records
    scans = [make_scan(n, w, 40, rng, False) for n, w in ((50, "far"), (9, "near"), (2, "far"), (30, "near"))]
    recs = [make_record(n, w, 1.0) for n, w in ((50, "far"), (9, "near"), (2, "far"), (30, "near"), (77, "near"))]
    recs.append(dict(make_record(78, "far", 2.0), pilot=None))     # not scanned: not an apt node, no refusal
    for nodes in (recs, {r["node_id"]: r for r in recs}):
        pairs = SC.apt_nodes(scans, nodes)
        assert [(p[0].set, p[0].node_id) for p in pairs] == [("near", 9), ("near", 30), ("far", 2), ("far", 50)], pairs
        assert all(p[1]["node_id"] == p[0].node_id for p in pairs)
    assert SC.apt_nodes([], recs) == []
    expect_value_error(lambda: SC.crop_bounds(rec, "mid", cfg), "which", "crop_bounds which")


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
