"""Smoke test for S3c, one defocus kernel calibrated on nodes pooled across
cells -- Block 19 in specs/SPEC.md (D-088; 2026-10-26).

Checks
    test_known_answer   the round-robin pool of three cells' hand-made scans
                        (near 3, 1, 2; far 0, 4, 4) at a cap of 4: each
                        cell's first, then each one's second; the qualified
                        ids s x 10^6 + n; sufficiency at 9 and 10 nodes
    test_reference      the pool equals an independent re-implementation
    test_convergence    not a discretisation: skipped
    test_invariants     the cells' order in the input changes nothing; two
                        cells sharing a node id give two nodes
    test_contract       scan_cell with max_per_set stops the set at the cap
                        (the rest set_full, never measured) and with
                        order_rng scans in the permutation's order; far-fit
                        on analytic far scans writes the production kernel
                        when both sets hold 10 nodes and withholds it below
                        (9 far, or 3 near); fit on 2 near phantoms records
                        sufficient false; the steps align (manual), nodes,
                        calibration and pool --what calibration end to end
                        on the synthetic cell
    test_determinism    the pool twice gives identical files
    test_edge_cases     no scans in a set; a cell without candidates; a
                        node id beyond the qualifier refused; the
                        configuration's refusals

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_s3c.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import io
import json
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

import calibrate_kernel as CK  # noqa: E402
import survey_cells as SC  # noqa: E402
from allen_diameter.analysis import calibration as CAL  # noqa: E402
from allen_diameter.analysis import calibration_pool as CP  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import calibration_io as CIO  # noqa: E402
from allen_diameter.loading import cell_pool, survey_io  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402

SEED = 20261026
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")


def fake_scan(nid, which="near", k_star=0):
    ks = np.arange(k_star - 3, k_star + 4)
    n = ks.size
    return CAL.PlaneScan(node_id=nid, d_hat_um=0.4, phi_rad=0.0, alpha_hat=0.2, k_star=k_star, z_sub_um=0.0, ks=ks,
                         z_um=ks * 0.28, omega_um2=np.full(n, 0.01), vw_um2=np.full(n, 0.01),
                         area_gl_um=np.full(n, 5.0), depth_gl=np.full(n, 20.0), B_bar=np.full(n, 200.0),
                         status=tuple("ok" for _ in range(n)), set=which)


def cells3():
    return [(3, [fake_scan(1), fake_scan(2), fake_scan(3)]),
            (1, [fake_scan(7), fake_scan(10, "far"), fake_scan(11, "far"), fake_scan(12, "far"), fake_scan(13, "far")]),
            (2, [fake_scan(1), fake_scan(20, "far"), fake_scan(5), fake_scan(21, "far"), fake_scan(22, "far"),
                 fake_scan(23, "far")])]


def capped(cap, min_nodes=2):
    cfg = default_config()
    return dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, calib_max_nodes_per_set=cap,
                                                                    calib_min_nodes_per_set=min_nodes))


def independent_pool(per_cell, cap):
    out = []
    for which in ("near", "far"):
        lists = [(s, [x for x in sc if x.set == which]) for s, sc in sorted(per_cell, key=lambda c: c[0])]
        k = 0
        took = []
        while len(took) < min(cap, sum(len(x) for _, x in lists)):
            for s, xs in lists:
                if k < len(xs) and len(took) < cap:
                    took.append(s * 10 ** 6 + xs[k].node_id)
            k += 1
        out += took
    return out


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = capped(4)
    pooled, meta = CP.pool_scans(cells3(), cfg)
    near = [s.node_id for s in pooled if s.set == "near"]
    far = [s.node_id for s in pooled if s.set == "far"]
    assert near == [1000007, 2000001, 3000001, 2000005], near            # cells 1, 2, 3; then 2's and 3's second
    assert far == [1000010, 2000020, 1000011, 2000021], far
    assert meta["sets"]["near"]["available"] == 6 and meta["sets"]["near"]["taken"] == 4
    assert meta["sets"]["near"]["by_cell"] == {"1": 1, "2": 2, "3": 1}, meta["sets"]["near"]["by_cell"]
    assert meta["id_map"]["3000001"] == [3, 1] and meta["id_map"]["2000001"] == [2, 1]
    d = default_config()
    assert CP.sufficiency(9, d)["sufficient"] is False and CP.sufficiency(10, d)["sufficient"] is True
    assert CP.sufficiency(10, d)["min_nodes"] == 10 and CP.sufficiency(10, d)["max_nodes"] == 30


def test_reference():
    for cap in (1, 3, 4, 30):
        pooled, _ = CP.pool_scans(cells3(), capped(cap))
        assert [s.node_id for s in pooled] == independent_pool(cells3(), cap), cap


def test_convergence():
    raise unittest.SkipTest("no discretisation: a selection of scans")


def test_invariants():
    cfg = capped(4)
    a, ma = CP.pool_scans(cells3(), cfg)
    b, mb = CP.pool_scans(list(reversed(cells3())), cfg)
    assert [s.node_id for s in a] == [s.node_id for s in b] and ma == mb
    ids = [s.node_id for s in a if s.set == "near"]
    assert len(ids) == len(set(ids)) and {2000001, 3000001} <= set(ids)   # node 1 of cells 2 and 3: two nodes


def far_scan(nid, cfg, gamma, sigma0, rng):
    """A far scan whose ok planes follow omega = c + (sigma0 + gamma x)^2 exactly (F1), k* = 0, the axis at the
    pilot's depth, amplitude 40 gl, RMS 1 gl: select_far_planes keeps it."""
    dz, d0 = cfg.acquisition.dz_um, cfg.calibration.table_max_delta_um
    P = cfg.calibration.far_offsets_planes
    ks = np.arange(-P, P + 1)
    zk = ks * dz
    x = np.maximum(np.abs(zk) - d0, 0.0)
    c = rng.uniform(-0.02, 0.03)
    om = c + (sigma0 + gamma * x) ** 2
    n = ks.size
    return CAL.PlaneScan(node_id=nid, d_hat_um=0.7, phi_rad=0.0, alpha_hat=0.3, k_star=0, z_sub_um=0.0, ks=ks, z_um=zk,
                         omega_um2=om, vw_um2=om.copy(), area_gl_um=np.full(n, 30.0), depth_gl=np.full(n, 20.0),
                         B_bar=np.full(n, 200.0), status=tuple("ok" for _ in range(n)), set="far",
                         amp_gl=np.full(n, 40.0), rms_gl=np.ones(n))


def write_calibration(path, cfg, n_near):
    """The fields far-fit reads from Block 10's calibration file, by hand: the near fit's nodes, the anchor and the
    default table cut at 0.84 um."""
    rc = cfg.renderer
    delta = (0.0, 0.28, 0.56, 0.84)
    sig = tuple(float(K.sigma_r(np.array([d]), rc)[0]) for d in delta)
    with open(path, "w") as f:
        json.dump(dict(growth=dict(node_ids=list(range(n_near))), anchor=dict(sigma0_um=sig[-1], se_um=0.005),
                       kernel=dict(sigma_r0_um=sig[0], kernel_table_delta_um=list(delta), kernel_table_sigma_um=list(sig))),
                  f)
    return sig[-1]


def test_contract():
    import fixtures_cell
    import run_cell
    cfg0 = default_config()
    # scan_cell's cap and order (the calibration test's miniature cell: a 0.2 um tube, Allen radius 0.1 um)
    with tempfile.TemporaryDirectory() as tmp:
        cfg_cell = dataclasses.replace(cfg0, calibration=dataclasses.replace(cfg0.calibration, patch_half_um=3.5,
                                                                             isolation_depth_gl=50.0,
                                                                             near_candidates="table",
                                                                             far_candidates="table"),
                                       measure=dataclasses.replace(cfg0.measure, block_half_um=3.5))
        swc, fetcher, planes, swc_path = fixtures_cell.synthetic_cell(tmp, cfg_cell, d_true=0.2, mu=1.5, allen_radius=0.1)
        provider = run_cell.real_provider(fetcher, planes, cfg_cell.acquisition.res0_um)
        full = CAL.scan_cell(swc, provider, cfg_cell, candidates={4, 5, 6})
        n_ok = sum(1 for o in full if o[2] is not None)
        assert n_ok >= 2, [o[3] for o in full]
        calls = []

        def counting(*args):
            calls.append(args)
            return provider(*args)
        perm = np.random.default_rng([7, 1]).permutation(3)
        lines = []
        out = CAL.scan_cell(swc, counting, cfg_cell, candidates={4, 5, 6}, order_rng=np.random.default_rng([7, 1]),
                            max_per_set=1, log=lines.append)
        assert [o[0] for o in out] == [[4, 5, 6][k] for k in perm], ([o[0] for o in out], perm)
        # 2026-10-29: the set_full nodes are counted in one line at the end, not one line each
        n_full = sum(1 for o in out if o[3] == ["set_full"])
        assert n_full and not any("set_full" in x for x in lines[:-1]), lines
        assert lines[-1] == "not measured, their set already holding 1 scans (set_full): %d near" % n_full, lines
        assert sum(1 for o in out if o[2] is not None) == 1
        first = next(i for i, o in enumerate(out) if o[2] is not None)
        assert all(o[3] == ["set_full"] and o[1] is None for o in out[first + 1:]), out
        n_calls_capped = len(calls)
        calls.clear()
        CAL.scan_cell(swc, counting, cfg_cell, candidates={4, 5, 6}, order_rng=np.random.default_rng([7, 1]))
        assert len(calls) > n_calls_capped, (len(calls), n_calls_capped)        # set_full nodes were not measured
        # the steps align (manual), nodes, calibration, then the pooled calibration (near only: insufficient)
        out_dir = os.path.join(tmp, "survey")
        rows = [dict(specimen_id=11, species="Homo Sapiens", layer="3", dendrite_type="spiny",
                     reconstruction_type="full", area="MTG", donor_id="9", hemisphere="left", normalized_depth=0.3,
                     neuron_class="exc", group="exc|3", rank=0)]
        sel = os.path.join(tmp, "selection.csv")
        cell_pool.write_table(sel, rows, cell_pool.SELECTION_COLUMNS)
        cell = survey_io.CellDir(out_dir, 11)
        survey_io.write_json(cell.path_of("align", "alignment_manual.json"),
                             dict(shift_x_px=0.0, shift_y_px=0.0, flip_h_px=None, z0_um=0.0))

        class Source:
            def __init__(self):
                self.fetcher, self.cache_dir = fetcher, tmp

            def swc_path(self, sid):
                return swc_path

            def planes(self, sid):
                return planes
        logs = []
        cfg_s = dataclasses.replace(cfg_cell, calibration=dataclasses.replace(cfg_cell.calibration, bootstrap_resamples=8),
                                    survey=dataclasses.replace(cfg_cell.survey, calib_scans_per_cell=2))
        st = SC.run(cfg_s, sel, out_dir, Source(), ["align", "nodes", "calibration"], log=logs.append)
        assert st[0]["calibration"] == "done", (st, logs)
        summ = survey_io.read_marker(cell, "calibration")["summary"]
        assert summ["n_near"] == 2 and summ["n_far"] == 0, summ                 # capped at calib_scans_per_cell
        scans, meta = CIO.load_scans(cell.path_of("calibration", "scans_11.json"))
        assert len(scans) == 2 and meta["max_per_set"] == 2 and "set_full" in summ["reasons"], (meta, summ)
        # D-115.5: meta.nodes carries the pilot's values -- None for a node never measured (set_full), every field for
        # a measured one, the scan's own k_star and phi_rad for a scanned one (the file is strict JSON: loaded above)
        recs = meta["nodes"]
        assert recs and all("pilot" in n for n in recs), recs
        by_scan = {int(s.node_id): s for s in scans}
        n_meas = 0
        for n in recs:
            p = n["pilot"]
            if "set_full" in n["reasons"].split(";"):
                assert p is None, n
                continue
            n_meas += 1
            assert p is not None and set(p) == set(CAL.PILOT_RECORD_FIELDS), n
            if n["scanned"]:
                s = by_scan[int(n["node_id"])]
                assert p["k_star"] == int(s.k_star) and p["phi_rad"] == float(s.phi_rad), (p, s.k_star, s.phi_rad)
        assert n_meas >= len(scans) and any(n["pilot"] is None for n in recs), recs
        assert sum(1 for n in recs if n["scanned"]) == len(scans), recs
        assert CAL.pilot_record(None) is None
        # NaN and inf become None; k_star None only when the pilot has no sharpest plane (z_sub_um not finite), -1
        # being a plane of a phantom stack; duck-typed pilot (pilot_view returns it as it is: pilot_status "")
        from types import SimpleNamespace
        base = dict(pilot_status="", k_star=-1, z_sub_um=float("nan"), cx_um=1.0, cy_um=2.0, theta_rad=0.5,
                    phi_rad=float("nan"), B_bar=150.0, mu_hat_per_um=float("inf"), d_hat_um=0.3, alpha_hat=0.1,
                    fit_status="converged", flags=("faint",), reg_verdict="ok")
        r0 = CAL.pilot_record(SimpleNamespace(**base))
        assert (r0["k_star"] is None and r0["z_sub_um"] is None and r0["phi_rad"] is None
                and r0["mu_hat_per_um"] is None and r0["flags"] == ["faint"] and r0["B_bar"] == 150.0), r0
        r1 = CAL.pilot_record(SimpleNamespace(**dict(base, z_sub_um=-0.3)))
        assert r1["k_star"] == -1 and r1["z_sub_um"] == -0.3, r1
        json.dumps([r0, r1], allow_nan=False)
        # the SPEC's field list written out (a field dropped from PILOT_RECORD_FIELDS is caught)
        assert tuple(CAL.PILOT_RECORD_FIELDS) == ("k_star", "z_sub_um", "cx_um", "cy_um", "theta_rad", "phi_rad", "B_bar",
                                                  "mu_hat_per_um", "d_hat_um", "alpha_hat", "fit_status", "flags",
                                                  "reg_verdict"), CAL.PILOT_RECORD_FIELDS
        # node_records on a hand-made scan_cell result: a measured node the verdict rejected keeps its pilot (the point
        # of D-115.5: its verdict can be recomputed), a set_full node has none, a scanned one is marked
        swc_f = SimpleNamespace(ids=np.array([1, 2, 3, 4]), radius=np.array([0.1, 0.1, np.nan, 0.1]))
        res_f = SimpleNamespace(**dict(base, z_sub_um=2.0, k_star=7, phi_rad=0.1))
        out_f = [(1, res_f, None, ["not_flat"]), (2, None, None, ["set_full"]), (3, res_f, None, ["allen_diameter_out"]),
                 (4, res_f, object(), [])]
        nr = CAL.node_records(out_f, swc_f, cfg_s)
        assert [n["node_id"] for n in nr] == [1, 2, 3, 4] and [n["scanned"] for n in nr] == [False, False, False, True]
        assert nr[0]["pilot"] == CAL.pilot_record(res_f) and nr[0]["pilot"]["k_star"] == 7, nr[0]
        assert nr[0]["reasons"] == "not_flat" and nr[1]["pilot"] is None and nr[3]["reasons"] == "", nr
        assert nr[2]["allen_diameter_um"] is None and nr[2]["set"] == "" and nr[2]["pilot"] is not None, nr[2]
        assert nr[0]["allen_diameter_um"] == 0.2 and nr[0]["set"] == CAL.calibration_set(0.2, cfg_s), nr[0]
        json.dumps(nr, allow_nan=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            meta_p, done = SC.pool_calibration(cfg_s, out_dir, sel, log=logs.append)
        assert done["fit"] and not done["far_fit"], (done, logs[-5:])
        assert meta_p["sets"]["near"]["taken"] == 2 and not meta_p["sets"]["near"]["sufficient"]
        cal = CIO.load_calibration(os.path.join(out_dir, "s3c", "calibration_pooled.json"))
        assert cal["meta"]["sufficiency"]["sufficient"] is False and cal["meta"]["sufficiency"]["n_nodes"] == 2
        assert "INSUFFICIENT: 2 near-field nodes < 10" in buf.getvalue(), buf.getvalue()
        assert all(n // CP.QUALIFIER == 11 for n in cal["growth"]["node_ids"]), cal["growth"]["node_ids"]
        assert os.path.exists(os.path.join(out_dir, "s3c", "figures", "width_by_plane_near.png"))
        # the step with Block 10.C's search (D-112, the default): the candidates are the search's chosen nodes, its
        # records in the cell's calibration folder, its lines printed; no table candidate is used
        cfg_x = dataclasses.replace(cfg_s, calibration=dataclasses.replace(cfg_s.calibration, near_candidates="cell",
                                                                           far_candidates="cell"))
        ctx = SC.Context(rows[0], cell, Source())
        buf = io.StringIO()
        with redirect_stdout(buf):
            sx = SC.step_calibration(ctx, cfg_x)
        near_csv = cell.path_of("calibration", "near_search_11.csv")
        assert os.path.exists(near_csv) and os.path.exists(cell.path_of("calibration", "far_search_11.csv"))
        from allen_diameter.loading import table_io
        chosen = {int(r["node_id"]) for w in ("near", "far")
                  for r in table_io.read_rows([cell.path_of("calibration", "%s_search_11.csv" % w)])
                  if str(r["chosen"]) == "True"}
        _scans, meta_x = CIO.load_scans(cell.path_of("calibration", "scans_11.json"))
        assert chosen and meta_x["n_candidates"] == sx["n_candidates"] == len(chosen), (meta_x["n_candidates"], sx,
                                                                                      chosen, buf.getvalue())
        assert {int(n["node_id"]) for n in meta_x["nodes"]} <= chosen, (meta_x["nodes"], chosen)
        assert "11 near search over the cell:" in buf.getvalue() and "11 far search over the cell:" in buf.getvalue(), \
            buf.getvalue()
        # D-115.6 (10.C.R): run(..., reuse_search=DIR) -- a Cell-7-like folder holding this cell's searches and their
        # records -- reaches the step through the Context: the step searches nothing and takes the same candidates
        import shutil
        from allen_diameter.analysis import candidate_search as FS
        c7 = os.path.join(tmp, "cell7_calibration")
        os.makedirs(c7)
        for w in ("near", "far"):
            for ext in ("csv", "json"):
                shutil.copy(cell.path_of("calibration", "%s_search_11.%s" % (w, ext)), c7)

        def boom(*a, **k):
            raise AssertionError("search called on a reuse")
        saved_search = FS.search
        FS.search = boom
        buf, logs_r = io.StringIO(), []
        try:
            with redirect_stdout(buf):
                st_r = SC.run(cfg_x, sel, out_dir, Source(), ["calibration"], force=["calibration"], log=logs_r.append,
                              reuse_search=c7)
        finally:
            FS.search = saved_search
        assert st_r[0]["calibration"] == "done", (st_r, logs_r, buf.getvalue())
        _scans_r, meta_r = CIO.load_scans(cell.path_of("calibration", "scans_11.json"))
        assert meta_r["n_candidates"] == len(chosen), (meta_r["n_candidates"], chosen)
        assert "11 near search reused from %s" % os.path.join(c7, "near_search_11.csv") in buf.getvalue(), buf.getvalue()
        assert "11 far search reused from %s" % os.path.join(c7, "far_search_11.csv") in buf.getvalue(), buf.getvalue()
        with open(cell.path_of("calibration", "far_search_11.json")) as f:
            assert json.load(f)["reused_from"] == os.path.abspath(os.path.join(c7, "far_search_11.json"))
        # main hands --reuse-search to run (and nothing when it is not given)
        seen = []
        saved_run = SC.run
        SC.run = lambda *a, **k: seen.append(k.get("reuse_search")) or []
        try:
            for extra in (["--reuse-search", c7], []):
                SC.main(["run", "--selection", sel, "--out-dir", out_dir, "--cache-dir", os.path.join(tmp, "cache"),
                         "--steps", "calibration"] + extra)
        finally:
            SC.run = saved_run
        assert seen == [c7, None], seen
    # far-fit's production kernel only with both sets sufficient (analytic far scans, a hand-made calibration file)
    cfg = dataclasses.replace(cfg0, calibration=dataclasses.replace(cfg0.calibration, bootstrap_resamples=8))
    rng = np.random.default_rng(SEED)
    with tempfile.TemporaryDirectory() as tmp:
        cp = os.path.join(tmp, "cal.json")
        sigma0 = write_calibration(cp, cfg, n_near=12)
        for n_far, n_near, want in ((10, 12, True), (9, 12, False), (10, 3, False)):
            write_calibration(cp, cfg, n_near=n_near)
            sp = os.path.join(tmp, "far_%d_%d.json" % (n_far, n_near))
            CIO.save_scans([far_scan(100 + j, cfg, 0.79, sigma0, rng) for j in range(n_far)], sp, {})
            fp = os.path.join(tmp, "ff_%d_%d.json" % (n_far, n_near))
            buf = io.StringIO()
            with redirect_stdout(buf):
                CK.run_far_fit(cfg, sp, cp, fp)
            obj = CIO.load_calibration(fp)
            assert ("kernel" in obj) is want, (n_far, n_near, obj.get("kernel_error"), buf.getvalue())
            assert obj["meta"]["sufficiency"]["far"]["n_nodes"] == n_far
            assert obj["meta"]["sufficiency"]["near"]["n_nodes"] == n_near
            if want:
                assert abs(obj["kernel"]["kernel_continuation_slope"] - 0.79) < 1e-6, obj["kernel"]
            else:
                assert "insufficient" in obj["kernel_error"], obj["kernel_error"]


def test_determinism():
    cfg = capped(4)
    with tempfile.TemporaryDirectory() as tmp:
        a, b = os.path.join(tmp, "a.json"), os.path.join(tmp, "b.json")
        for path in (a, b):
            pooled, meta = CP.pool_scans(cells3(), cfg)
            CIO.save_scans(pooled, path, meta)
        assert open(a).read() == open(b).read()


def test_edge_cases():
    cfg = capped(4)
    pooled, meta = CP.pool_scans([(1, [fake_scan(7)])], cfg)
    assert meta["sets"]["far"]["taken"] == 0 and meta["sets"]["far"]["available"] == 0 and len(pooled) == 1
    try:
        CP.qualify([fake_scan(10 ** 6)], 5)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("a node id beyond the qualifier must be refused")
    # a cell without candidates: the step writes an empty scans file and says so
    with tempfile.TemporaryDirectory() as out:
        cell = survey_io.CellDir(out, 4)
        import csv
        with open(cell.path_of("nodes", "pilot_4.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node_id", "fit_status", "flags", "phi_rad", "alpha_hat", "allen_radius_um"])
            w.writerow([2, "failed", "", 0.0, 0.2, 0.1])
        survey_io.write_marker(cell, "nodes", {}, 0.0, "x")
        ctx = SC.Context(dict(specimen_id=4), cell, None)
        d0 = default_config()      # the table's candidates (no SWC here); the search's own edges: candidate_search
        s = SC.step_calibration(ctx, dataclasses.replace(d0, calibration=dataclasses.replace(
            d0.calibration, near_candidates="table", far_candidates="table")))
        assert s == dict(n_candidates=0, n_near=0, n_far=0, reasons={}), s
        scans, meta = CIO.load_scans(cell.path_of("calibration", "scans_4.json"))
        assert scans == [] and meta["n_candidates"] == 0
    d = default_config()
    for bad in (dict(calib_min_nodes_per_set=0), dict(calib_min_nodes_per_set=40)):
        try:
            dataclasses.replace(d, calibration=dataclasses.replace(d.calibration, **bad)).validate()
        except ValueError:
            continue
        raise AssertionError("calibration %s must be refused" % bad)
    try:
        dataclasses.replace(d, survey=dataclasses.replace(d.survey, calib_scans_per_cell=0)).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("calib_scans_per_cell 0 must be refused")


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
