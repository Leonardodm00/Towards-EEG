"""Smoke test for the pooled kernel law by cell -- Block 19.5 in
specs/SPEC.md (D-090.2; 2026-10-27).

Four cells (specimens 101, 202, 303, 404), three near-field and three
far-field nodes each, made by test_smoke_calibration_figures.make_scans:
exact near scans of one growth (omega = c_j + G_true(delta)) and exact far
scans (omega = c_j + (sigma_0 + gamma x)^2 beyond the anchor) with gamma 0.79
for three cells and 0.99 for the fourth; node ids qualified as in the pool
(specimen x 10^6 + node). The near fit is Block 10's own on the pooled near
scans (exact data: it recovers c_j and the growth); the far fit is given by
hand at gamma_law = 0.79 with the true c_j and s_j = 0, so every far residual
is known in closed form.

Checks
    test_known_answer   near residuals within the table's linear
                        interpolation error where |delta| <= delta_T; far
                        residuals = (sigma_0 - sigma_0,cal) + (gamma_cell - 0.79)
                        x with x = |delta| - delta_0, to 1e-9; the fourth cell's
                        far median = 0.2 median(x) over its used planes; the
                        node and plane counts
    test_reference      cell_of against the map and the division; the pooled
                        law against numpy.interp and the line; the summary's
                        quartiles against numpy.percentile
    test_convergence    not a discretisation: skipped
    test_invariants     kernel_law_inputs' points are the by-cell points
                        without cell and residual, in order; the scans' order
                        changes no summary row
    test_contract       the summary's columns; write_all with four cells
                        writes the two by-cell figures (one panel per cell),
                        with three cells one panel, with one cell neither;
                        pool --what calibration on four cells' files writes
                        kernel_law_by_cell.csv, the figures, and the fourth
                        cell stands out
    test_determinism    the by-cell figures twice are byte-identical
    test_edge_cases     an unqualified node has no cell; without a far fit no
                        far point and the near points beyond the table have a
                        NaN residual, left out of the summary; a cell whose far
                        planes are all unused gives NaN medians

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_calibration_by_cell.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import io
import math
import os
import platform
import sys
import tempfile
import time
import traceback
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_smoke_calibration_figures as TF  # noqa: E402  (make_scans, fake_calib: exact near and far scans)
from allen_diameter.analysis import calibration_pool as CP  # noqa: E402
from allen_diameter.analysis import calibration_report as CR  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import calibration_io as CIO  # noqa: E402
from allen_diameter.loading import cell_pool, survey_io  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402
from allen_diameter.plotting import calibration_figures as CF  # noqa: E402

SEED = 20261027
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
CELLS = (101, 202, 303, 404)
GAMMA, GAMMA_DEV, DEV = 0.79, 0.99, 404
FAR_ID = 100                            # far nodes' SWC ids start here, apart from the near nodes'


# --------------------------------------------------------------- fixtures ---

def cell_scans(cfg, rng, cells=CELLS, n_near=3, n_far=3):
    """{specimen: (near scans, far scans)} with SWC ids (unqualified), and {qualified far id: (c, gamma)}."""
    out, truth = {}, {}
    for s in cells:
        near, _ = TF.make_scans(cfg, "near", n_near, rng)
        g = GAMMA_DEV if s == DEV else GAMMA
        far, t_far = TF.make_scans(cfg, "far", n_far, rng, gamma=g)
        far = [dataclasses.replace(sc, node_id=FAR_ID + int(sc.node_id)) for sc in far]
        for sc, (c, _z) in zip(far, t_far):
            truth[s * CP.QUALIFIER + int(sc.node_id)] = (c, g)
        out[s] = (near, far)
    return out, truth


def pooled(per_cell):
    near = [q for s, (n, _f) in sorted(per_cell.items()) for q in CP.qualify(n, s)]
    far = [q for s, (_n, f) in sorted(per_cell.items()) for q in CP.qualify(f, s)]
    return near, far


def hand_far_fit(cfg, far, truth, gamma=GAMMA):
    """Block 15's far-fit fields stage_inputs reads, by hand: the true c_j, s_j = 0, the given gamma."""
    s0 = float(K.sigma_r(cfg.calibration.table_max_delta_um, cfg.renderer))
    ids = [int(sc.node_id) for sc in far]
    return {"far_fit": dict(node_ids=ids, c_um2=[truth[i][0] for i in ids], s_um=[0.0] * len(ids), gamma=gamma,
                            anchor=dict(sigma0_um=s0))}


_STD = {}


def standard():
    if not _STD:
        cfg = default_config()
        rng = np.random.default_rng(SEED)
        per_cell, truth = cell_scans(cfg, rng)
        near, far = pooled(per_cell)
        calib = TF.fake_calib(cfg, near, None)
        ff = hand_far_fit(cfg, far, truth)
        _STD.update(cfg=cfg, per_cell=per_cell, truth=truth, near=near, far=far, calib=calib, ff=ff,
                    kc=CR.kernel_law_by_cell(near + far, cfg, calib, ff))
    return _STD


def row(kc, cell, which):
    return next(r for r in kc["summary"] if r["specimen_id"] == cell and r["set"] == which)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    st = standard()
    kc, cfg, calib = st["kc"], st["cfg"], st["calib"]
    assert kc["cells"] == list(CELLS), kc["cells"]
    d_T = float(calib["kernel"]["kernel_table_delta_um"][-1])
    d0, s0_cal = float(calib["anchor"]["delta0_um"]), float(calib["anchor"]["sigma0_um"])
    s0 = float(K.sigma_r(cfg.calibration.table_max_delta_um, cfg.renderer))
    assert abs(d_T - 0.84) <= 1e-12 and abs(d0 - 0.84) <= 1e-12
    # 0.84 um is a knot of the growth (step 0.28 um), so the fitted table ends at the true sigma_r(0.84)
    assert abs(s0 - s0_cal) <= 1e-6, (s0, s0_cal)
    near_in = [p for p in kc["points"] if p["set"] == "near" and p["abs_delta_um"] <= d_T]
    # the near points follow the planted growth, a cubic in G through the knots (0.28 um apart): sigma_hat =
    # (G_true(|delta|) + sigma_r(0)^2)^(1/2) once the fit has recovered c_j (to 1e-9, Block 10); the pooled law
    # there is the table, linear in sigma between the same knots -- the two differ between knots (up to 0.075 um
    # on this fixture, where the cubic dips below zero near 0.1 um), and that difference is the residual
    _, _, g_true = TF.truth_growth(cfg)
    td = np.asarray(calib["kernel"]["kernel_table_delta_um"], dtype=float)
    ts = np.asarray(calib["kernel"]["kernel_table_sigma_um"], dtype=float)
    s_r0 = float(cfg.renderer.sigma_r0_um)
    err = 0.0
    for p in near_in:
        expect = math.sqrt(float(g_true(np.array([p["abs_delta_um"]]))[0]) + s_r0 ** 2) - float(np.interp(
            p["abs_delta_um"], td, ts))
        err = max(err, abs(p["residual_um"] - expect))
    assert len(near_in) >= 60 and err <= 1e-6, (len(near_in), err)
    far_used = [p for p in kc["points"] if p["set"] == "far" and p["used"]]
    assert len(far_used) == 4 * 3 * 24, len(far_used)                       # 24 used planes per far node (Block 15)
    for p in far_used:
        g = st["truth"][p["node_id"]][1]
        x = p["abs_delta_um"] - d0
        expect = (s0 - s0_cal) + (g - GAMMA) * x
        assert abs(p["residual_um"] - expect) <= 1e-9, (p, expect)
    for c in CELLS:
        r = row(kc, c, "far")
        xs = [p["abs_delta_um"] - d0 for p in far_used if p["cell"] == c]
        expect = (s0 - s0_cal) + ((GAMMA_DEV - GAMMA) * float(np.median(xs)) if c == DEV else 0.0)
        assert abs(r["median_residual_um"] - expect) <= 1e-9, (c, r, expect)
        assert (r["n_nodes"], r["n_points_used"]) == (3, 72), r
        assert row(kc, c, "near")["n_nodes"] == 3
    assert row(kc, DEV, "far")["median_residual_um"] > 0.1                    # 0.2 x the median x of 1-4 um


def test_reference():
    assert CP.cell_of(5000007, {"5000007": [5, 7]}) == (5, 7)
    assert CP.cell_of(5000007) == (5, 7) and CP.cell_of(5000007, {"1": [9, 9]}) == (5, 7)
    assert CP.cell_of(123) == (None, 123) and CP.cell_of(123, {"123": [8, 3]}) == (8, 3)
    st = standard()
    calib, ff = st["calib"], st["ff"]
    law = CR.pooled_sigma_law(calib, ff)
    td = np.asarray(calib["kernel"]["kernel_table_delta_um"], dtype=float)
    ts = np.asarray(calib["kernel"]["kernel_table_sigma_um"], dtype=float)
    d = np.random.default_rng(SEED + 3).uniform(0.0, td[-1], 50)
    assert np.max(np.abs(law(d) - np.interp(d, td, ts))) <= 1e-12
    db = np.linspace(td[-1] + 0.01, 4.5, 20)
    line = calib["anchor"]["sigma0_um"] + GAMMA * (db - calib["anchor"]["delta0_um"])
    assert np.max(np.abs(law(db) - line)) <= 1e-12
    assert np.isnan(law(np.array([-0.1]))[0])
    # the far line starts at the calibration's anchor, not at the table's last point: an anchor set apart by hand
    hand = dict(kernel=dict(kernel_table_delta_um=[0.0, 0.42, 0.84], kernel_table_sigma_um=[0.1, 0.3, 0.6]),
                anchor=dict(delta0_um=0.9, sigma0_um=0.7))
    lh = CR.pooled_sigma_law(hand, {"far_fit": dict(gamma=0.5)})
    assert abs(float(lh(np.array([2.9]))[0]) - (0.7 + 0.5 * (2.9 - 0.9))) <= 1e-12
    assert abs(float(lh(np.array([0.21]))[0]) - 0.2) <= 1e-12
    assert np.isnan(CR.pooled_sigma_law(hand, None)(np.array([1.0]))[0])
    kc = st["kc"]
    for r in kc["summary"]:
        e = np.array([p["residual_um"] for p in kc["points"] if p["cell"] == r["specimen_id"] and p["set"] == r["set"]
                      and p["used"] and math.isfinite(p["residual_um"])])
        q25, q50, q75 = np.percentile(e, [25, 50, 75])
        assert abs(r["q25_residual_um"] - q25) <= 1e-12 and abs(r["median_residual_um"] - q50) <= 1e-12
        assert abs(r["q75_residual_um"] - q75) <= 1e-12


def test_convergence():
    raise unittest.SkipTest("no discretisation: residuals and their quantiles")


def test_invariants():
    st = standard()
    kl = CR.kernel_law_inputs(st["near"] + st["far"], st["cfg"], st["calib"], st["ff"])
    assert kl["points"] == [(p["abs_delta_um"], p["sigma_um"], p["side"], p["used"], p["set"]) for p in st["kc"]["points"]]
    rev = CR.kernel_law_by_cell(list(reversed(st["near"] + st["far"])), st["cfg"], st["calib"], st["ff"])
    for a, b in zip(sorted(st["kc"]["summary"], key=lambda r: (r["specimen_id"], r["set"])),
                    sorted(rev["summary"], key=lambda r: (r["specimen_id"], r["set"]))):
        assert a.keys() == b.keys() and all(
            (a[k] == b[k]) or (isinstance(a[k], float) and abs(a[k] - b[k]) <= 1e-12) for k in a), (a, b)


def _write_all(st, out, cells_scans, id_map=None, calib=None, ff=None):
    os.makedirs(out, exist_ok=True)
    return CF.write_all(cells_scans, calib or st["calib"], ff or st["ff"], {}, st["cfg"], out, every_node=False, dpi=60,
                        id_map=id_map)


def test_contract():
    st = standard()
    kc = st["kc"]
    assert all(tuple(r) == CR.BY_CELL_COLUMNS for r in kc["summary"])
    assert {k for k in kc} == {"points", "cells", "summary", "table", "line", "anchor"}
    with tempfile.TemporaryDirectory() as tmp:
        w = _write_all(st, os.path.join(tmp, "four"), st["near"] + st["far"])
        names = sorted(os.path.basename(x) for x in w)
        assert "kernel_law_by_cell.png" in names and "kernel_law_residuals_by_cell.png" in names, names
        fig = CF.kernel_law_by_cell_figure(kc)
        assert sum(1 for a in fig.axes if a.get_visible() and a.axison) == 4     # one panel per cell
        three = [sc for sc in st["near"] + st["far"] if sc.node_id // CP.QUALIFIER != DEV]
        kc3 = CR.kernel_law_by_cell(three, st["cfg"], TF.fake_calib(st["cfg"], [s for s in three if s.set == "near"], None),
                                    st["ff"])
        assert kc3["cells"] == [101, 202, 303] and len(CF.kernel_law_by_cell_figure(kc3).axes) == 1
        one = [dataclasses.replace(sc, node_id=sc.node_id % CP.QUALIFIER) for sc in st["per_cell"][101][0]]
        w1 = _write_all(st, os.path.join(tmp, "one"), one, calib=TF.fake_calib(st["cfg"], one, None))
        assert not any("by_cell" in x for x in w1), w1
        # pool --what calibration on four cells' files: the CSV and the figures (the real near and far fits)
        import survey_cells as SC
        out = os.path.join(tmp, "survey")
        for s, (near, far) in st["per_cell"].items():
            cell = survey_io.CellDir(out, s)
            CIO.save_scans(near + far, cell.path_of("calibration", "scans_%d.json" % s), dict(source="test"))
            survey_io.write_marker(cell, "calibration", {}, 0.0, "test")
        lines = []
        with redirect_stdout(io.StringIO()):
            SC.pool_calibration(st["cfg"], out, log=lines.append)
        tab = cell_pool.read_table(os.path.join(out, "s3c", "kernel_law_by_cell.csv"),
                                   dict(specimen_id="int", set="str", n_nodes="int", n_points_used="int",
                                        n_points="int", median_residual_um="float", q25_residual_um="float",
                                        q75_residual_um="float", median_abs_delta_um="float"))
        assert len(tab) == 8 and sorted({r["specimen_id"] for r in tab}) == list(CELLS), tab
        far_med = {r["specimen_id"]: r["median_residual_um"] for r in tab if r["set"] == "far"}
        others = [far_med[c] for c in CELLS if c != DEV]
        # the pooled fit's gamma lies between 0.79 and 0.99, so the fourth cell sits above the others
        assert far_med[DEV] > max(others) + 0.05, far_med
        for name in ("kernel_law_by_cell", "kernel_law_residuals_by_cell", "kernel_law"):
            assert os.path.getsize(os.path.join(out, "s3c", "figures", "%s.png" % name)) > 5000, name
        assert sum(1 for ln in lines if ln.startswith("by cell: specimen")) == 8, lines


def test_determinism():
    st = standard()
    with tempfile.TemporaryDirectory() as tmp:
        a = _write_all(st, os.path.join(tmp, "a"), st["near"] + st["far"])
        b = _write_all(st, os.path.join(tmp, "b"), st["near"] + st["far"])
        for x, y in zip(sorted(a), sorted(b)):
            if "by_cell" in x:
                with open(x, "rb") as fx, open(y, "rb") as fy:
                    assert fx.read() == fy.read(), x


def test_edge_cases():
    st = standard()
    cfg = st["cfg"]
    one = [dataclasses.replace(sc, node_id=sc.node_id % CP.QUALIFIER) for sc in st["per_cell"][101][0]]
    kc = CR.kernel_law_by_cell(one, cfg, TF.fake_calib(cfg, one, None), None)
    assert kc["cells"] == [None] and [r["specimen_id"] for r in kc["summary"]] == [None]
    # without a far fit: no far point; near points beyond the table have no law (NaN) and leave the summary
    kc = CR.kernel_law_by_cell(st["near"] + st["far"], cfg, st["calib"], None)
    assert not any(p["set"] == "far" for p in kc["points"])
    beyond = [p for p in kc["points"] if p["abs_delta_um"] > 0.84 + 1e-9]
    assert beyond and all(math.isnan(p["residual_um"]) for p in beyond)
    r = row(kc, 101, "near")
    assert r["n_points_used"] == sum(1 for p in kc["points"] if p["cell"] == 101 and p["used"]
                                     and math.isfinite(p["residual_um"])) < r["n_points"]
    # a cell whose far planes are all unused (the far plane selection drops them): NaN medians
    bad = [dataclasses.replace(sc, status=tuple("low_snr" for _ in sc.ks)) if sc.node_id // CP.QUALIFIER == 202 else sc
           for sc in st["far"]]
    kc = CR.kernel_law_by_cell(st["near"] + bad, cfg, st["calib"], st["ff"])
    r = row(kc, 202, "far")
    assert r["n_points_used"] == 0 and r["n_nodes"] == 0 and math.isnan(r["median_residual_um"]), r
    assert CR.kernel_law_by_cell(st["near"], cfg, None, None) is None


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
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
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, lines[0] if lines else ""))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
