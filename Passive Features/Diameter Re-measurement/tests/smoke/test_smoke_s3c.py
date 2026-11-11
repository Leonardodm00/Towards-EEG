"""Smoke test for S3c, one defocus kernel calibrated on nodes pooled across
cells -- Block 19 in specs/SPEC.md (D-088; 2026-10-26).

Checks
    test_known_answer   the round-robin pool of three cells' hand-made scans
                        (near 3, 1, 2; far 0, 4, 4) at a pool cap of 4: each
                        cell's first, then each one's second; the qualified
                        ids s x 10^6 + n; sufficiency at 9 and 10 nodes;
                        D-118: at the defaults (30 per cell, no pool cap)
                        every scan, in the round-robin order; at 2 per cell
                        each cell's first two
    test_reference      the pool equals an independent re-implementation
                        (a sort by depth and cell) over per-cell caps 1, 2,
                        30 and pool caps None, 1, 3, 4, 30
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
    test_cell_bootstrap 2026-11-04 (19.6, D-118.3): the plan (node / larger /
                        cell by the number of cells); a cell resample is whole
                        cells, a node resample the old draw; the cell SE of
                        the anchor above the node SE when cells differ, near
                        it when they do not; "larger" at 5 cells; the far
                        side's own plan and t_(C-1); determinism; refusals
    test_cell_weights   2026-11-10 (19.7, D-120.2): exact near scans of two
                        cells (6 nodes, growth 0.3 d^2; 2 nodes, 0.6 d^2):
                        the pooled G equals the cells' own G averaged with
                        weights 6:2 by node and 1:1 by cell; one cell, equal
                        cells and a node dropped from the larger cell give the
                        same G under both weightings; the labelled resample
                        draws as the unlabelled one; the far slope by cell
                        nearer the cells' mean slope than by node; the CLI
                        records and prints the weighting
    test_cell_outliers  2026-11-11 (19.8, D-120.3): the one-way ANOVA against
                        scipy.stats.f_oneway and the textbook n0; the
                        reference the other cells' means averaged equally;
                        thirteen cells, one shifted by 0.2 um (up or down):
                        only it removed; six: nothing removed, its z the
                        largest; leaving the cell out of its yardstick raises
                        its z; a one-node cell 0.05 um off kept thanks to
                        sigma_w^2 / N_c; end to end pooled_fits on twelve
                        cells, one with twice the growth: removed from the
                        near set only, refitted without it, the first pass
                        kept; switched off on the same folder: nothing
                        removed and the first run's conditional files gone
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


def capped(cap, min_nodes=2, per_cell=30):
    """The pool's cap (None: none, D-118.2), the per-cell cap and the minimum."""
    cfg = default_config()
    return dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, calib_min_nodes_per_set=min_nodes),
                               survey=dataclasses.replace(cfg.survey, calib_pool_max_per_set=cap,
                                                          calib_scans_per_cell=per_cell))


def independent_pool(per_cell, cap, per_cell_cap=30):
    """Written apart from the module: every allowed scan keyed by (depth in its cell, specimen), sorted, cut at
    the cap -- the round-robin's order without its loop."""
    out = []
    for which in ("near", "far"):
        keyed = []
        for s, sc in per_cell:
            xs = [x for x in sc if x.set == which][:per_cell_cap]
            keyed += [(depth, s, s * 10 ** 6 + x.node_id) for depth, x in enumerate(xs)]
        keyed.sort()
        out += [q for _, _, q in (keyed if cap is None else keyed[:cap])]
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
    assert meta["sets"]["near"]["max_nodes"] == 4 and meta["sets"]["near"]["max_per_cell"] == 30, meta["sets"]["near"]
    # D-118.2: at the defaults -- 30 per cell, no pool cap -- every scan, in the round-robin order
    assert d.survey.calib_scans_per_cell == 30 and d.survey.calib_pool_max_per_set is None
    pooled, meta = CP.pool_scans(cells3(), d)
    near = [s.node_id for s in pooled if s.set == "near"]
    far = [s.node_id for s in pooled if s.set == "far"]
    assert near == [1000007, 2000001, 3000001, 2000005, 3000002, 3000003], near
    assert far == [1000010, 2000020, 1000011, 2000021, 1000012, 2000022, 1000013, 2000023], far
    m = meta["sets"]["near"]
    assert (m["taken"], m["available"], m["max_nodes"], m["max_per_cell"]) == (6, 6, None, 30), m
    assert meta["sets"]["far"]["by_cell"] == {"1": 4, "2": 4}, meta["sets"]["far"]["by_cell"]
    # D-118.1: at most 2 per cell, no pool cap -- each cell's first two
    pooled, meta = CP.pool_scans(cells3(), capped(None, per_cell=2))
    near = [s.node_id for s in pooled if s.set == "near"]
    far = [s.node_id for s in pooled if s.set == "far"]
    assert near == [1000007, 2000001, 3000001, 2000005, 3000002], near
    assert far == [1000010, 2000020, 1000011, 2000021], far
    assert meta["sets"]["near"]["available"] == 6 and meta["sets"]["near"]["taken"] == 5
    assert meta["sets"]["near"]["by_cell"] == {"1": 1, "2": 2, "3": 2}, meta["sets"]["near"]["by_cell"]


def test_reference():
    for per_cell in (1, 2, 30):
        for cap in (None, 1, 3, 4, 30):
            pooled, _ = CP.pool_scans(cells3(), capped(cap, per_cell=per_cell))
            assert [s.node_id for s in pooled] == independent_pool(cells3(), cap, per_cell), (per_cell, cap)


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
        # D-117.5: meta.nodes carries the pilot's values -- None for a node never measured (set_full), every field for
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
        # of D-117.5: its verdict can be recomputed), a set_full node has none, a scanned one is marked
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
        # D-118: the pooled line names the per-cell cap and no pool cap
        assert any(ln.startswith("pooled near field: 2 of 2 scans taken (at most 2 per cell) from 1 cells")
                   for ln in logs), [ln for ln in logs if ln.startswith("pooled")]
        cal = CIO.load_calibration(os.path.join(out_dir, "s3c", "calibration_pooled.json"))
        assert cal["meta"]["sufficiency"]["sufficient"] is False and cal["meta"]["sufficiency"]["n_nodes"] == 2
        assert "INSUFFICIENT: 2 near-field nodes < 10" in buf.getvalue(), buf.getvalue()
        # 19.8 (D-120.3) on one cell: no z, nothing removed, the table written
        assert "cells that stand out (D-120.3): near field, 1 cell: no z (fewer than 3 cells)" in logs, logs
        assert os.path.exists(os.path.join(out_dir, "s3c", "cell_outliers.csv"))
        assert not os.path.exists(os.path.join(out_dir, "s3c", "scans_pooled_kept.json"))
        assert all(n // CP.QUALIFIER == 11 for n in cal["growth"]["node_ids"]), cal["growth"]["node_ids"]
        boot = cal["anchor"]["bootstrap"]                                    # 19.6 (D-118.3): one cell, nodes
        assert boot["rule"] == "node" and boot["unit"] == "node" and boot["n_cells"] == 1, boot
        assert boot["few_cells"] is False and boot["se_cell"] is None, boot
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
        # D-117.6 (10.C.R): run(..., reuse_search=DIR) -- a Cell-7-like folder holding this cell's searches and their
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
    for bad in (dict(calib_scans_per_cell=0), dict(calib_pool_max_per_set=0), dict(calib_pool_max_per_set=30.0),
                dict(calib_pool_max_per_set=True)):
        try:
            dataclasses.replace(d, survey=dataclasses.replace(d.survey, **bad)).validate()
        except ValueError:
            continue
        raise AssertionError("survey %s must be refused" % bad)
    dataclasses.replace(d, survey=dataclasses.replace(d.survey, calib_pool_max_per_set=30)).validate()   # D-088.3's


def near_cells(n_cells, n_per, scale_sd, rng, noise=0.003):
    """Near scans of n_cells cells x n_per nodes, qualified ids s x 10^6 + n: omega = c + a_s 0.3 (z_ax - z_k)^2 +
    noise, the cell's own growth scale a_s ~ N(1, scale_sd^2) (one blur per cell); 7 planes, z_ax in +- dz/2."""
    dz = default_config().acquisition.dz_um
    ks = np.arange(-3, 4)
    out = []
    for s in range(1, n_cells + 1):
        a = 1.0 + scale_sd * rng.normal()
        for n in range(n_per):
            z_ax = rng.uniform(-0.5, 0.5) * dz
            om = rng.uniform(0.02, 0.06) + a * 0.3 * (z_ax - ks * dz) ** 2 + rng.normal(0.0, noise, ks.size)
            m = ks.size
            out.append(CAL.PlaneScan(node_id=s * CP.QUALIFIER + n, d_hat_um=0.4, phi_rad=0.0, alpha_hat=0.2, k_star=0,
                                     z_sub_um=z_ax, ks=ks, z_um=ks * dz, omega_um2=om, vw_um2=om.copy(),
                                     area_gl_um=np.full(m, 5.0), depth_gl=np.full(m, 20.0), B_bar=np.full(m, 200.0),
                                     status=tuple("ok" for _ in range(m)), set="near"))
    return out


def test_cell_bootstrap():
    d = default_config()
    assert d.calibration.bootstrap_unit == "cell" and d.calibration.bootstrap_cell_min == 10
    q = lambda cells, per=3: [s * CP.QUALIFIER + n for s in range(1, cells + 1) for n in range(per)]
    # (a) the plan
    assert CAL.bootstrap_plan(list(range(20)), d) == ("node", 1)                  # one cell's own (unqualified) ids
    assert CAL.bootstrap_plan(q(1), d) == ("node", 1)
    assert CAL.bootstrap_plan(q(5), d) == ("larger", 5) and CAL.bootstrap_plan(q(9), d) == ("larger", 9)
    assert CAL.bootstrap_plan(q(10), d) == ("cell", 10) and CAL.bootstrap_plan(q(12), d) == ("cell", 12)
    d_node = dataclasses.replace(d, calibration=dataclasses.replace(d.calibration, bootstrap_unit="node"))
    assert CAL.bootstrap_plan(q(12), d_node) == ("node", 12)
    d_min4 = dataclasses.replace(d, calibration=dataclasses.replace(d.calibration, bootstrap_cell_min=4))
    assert CAL.bootstrap_plan(q(3), d_min4) == ("larger", 3) and CAL.bootstrap_plan(q(4), d_min4) == ("cell", 4)
    assert list(CAL.node_cells([7, 2000005, 2000001, 3000004])) == [-1, 2, 2, 3]
    # (a') a cell resample is whole cells, each in its order; a node resample is the node bootstrap's old draw
    cells = np.array([3, 3, 1, 1, 1, 2, 3])
    members = {lab: list(np.flatnonzero(cells == lab)) for lab in (1, 2, 3)}
    rng = np.random.default_rng(5)
    for _ in range(50):
        r = list(CAL.resample(cells, rng, "cell"))
        pos, runs = 0, 0
        while pos < len(r):
            lab = int(cells[r[pos]])
            assert r[pos:pos + len(members[lab])] == members[lab], (r, pos, lab)
            pos += len(members[lab])
            runs += 1
        assert runs == 3, r
    r1 = CAL.resample(cells, np.random.default_rng(9), "node")
    assert np.array_equal(r1, np.random.default_rng(9).integers(0, cells.size, cells.size)), r1
    # (b) reference: cells that differ in blur -> the cell SE well above the node SE; one blur -> close
    rng = np.random.default_rng(SEED)
    scans_diff = near_cells(12, 8, 0.3, rng)
    scans_same = near_cells(12, 8, 0.0, rng)
    for scans, low, high in ((scans_diff, 1.3, np.inf), (scans_same, 1.0 / 1.6, 1.6)):
        cell = CAL.anchor_se(scans, d, np.random.default_rng(1), n_resamples=150)
        node = CAL.anchor_se(scans, d_node, np.random.default_rng(1), n_resamples=150)
        assert cell["rule"] == "cell" and cell["unit"] == "cell" and cell["n_cells"] == 12 and node["unit"] == "node"
        ratio = cell["se"] / node["se"]
        assert low <= ratio <= high, (low, ratio, high, cell["se"], node["se"])
    # a one-cell set: the plan's node SE equals the old node bootstrap's for the same seed
    one = [dataclasses.replace(sc, node_id=int(sc.node_id) % CP.QUALIFIER + 100 * i) for i, sc in
           enumerate(scans_same[:20])]
    se_old, ok_old, _ = CAL.bootstrap_sigma0(one, d, np.random.default_rng(3), n_resamples=40)
    plan_one = CAL.anchor_se(one, d, np.random.default_rng(3), n_resamples=40)
    assert plan_one["rule"] == "node" and plan_one["se"] == se_old and plan_one["n_ok"] == ok_old, (plan_one, se_old)
    # (c) "larger" at 5 cells: both run, the larger reported and flagged
    five = scans_diff[:40]
    lg = CAL.anchor_se(five, d, np.random.default_rng(2), n_resamples=80)
    assert lg["rule"] == "larger" and lg["few_cells"] is True and lg["n_cells"] == 5, lg
    assert lg["se_node"] is not None and lg["se_cell"] is not None and lg["n_ok_node"] and lg["n_ok_cell"], lg
    assert lg["se"] == max(lg["se_node"], lg["se_cell"]) and lg["unit"] == ("cell" if lg["se_cell"] >= lg["se_node"]
                                                                           else "node"), lg
    # the larger's node SE is the node bootstrap's from the same generator, before the cell one
    se_n_alone, _, _ = CAL.bootstrap_sigma0(five, d, np.random.default_rng(2), n_resamples=80)
    assert lg["se_node"] == se_n_alone, (lg["se_node"], se_n_alone)
    # (c') the far side: its own cells; under the cell unit the interval uses t_(C-1)
    from allen_diameter.analysis import far_field as FF
    cfg_f = dataclasses.replace(d, calibration=dataclasses.replace(d.calibration, calib_min_nodes_per_set=2))
    frng = np.random.default_rng(11)
    gam = {s: 0.79 + 0.1 * frng.normal() for s in range(1, 11)}         # one slope per cell: the SEs are not 0
    far = [dataclasses.replace(far_scan(1, cfg_f, gam[s], 0.6, frng), node_id=s * CP.QUALIFIER + n)
           for s in range(1, 11) for n in range(2)]
    sel = FF.select_far_planes(far, cfg_f)
    assert sel.node_ids.size == 20, sel.node_ids
    assert CAL.bootstrap_plan(sel.node_ids, cfg_f) == ("cell", 10)
    fb = FF.gamma_se(sel, 0.6, cfg_f, np.random.default_rng(4), n_resamples=12)
    assert fb["rule"] == "cell" and fb["unit"] == "cell" and fb["n_cells"] == 10, fb
    se_fc, _, _ = FF.bootstrap_gamma(sel, 0.6, cfg_f, np.random.default_rng(4), n_resamples=12, unit="cell")
    se_fn, _, _ = FF.bootstrap_gamma(sel, 0.6, cfg_f, np.random.default_rng(4), n_resamples=12, unit="node")
    assert fb["se"] == se_fc and se_fc != se_fn, (fb["se"], se_fc, se_fn)      # the plan ran the cell bootstrap
    fit = FF.fit_far_slope(sel, 0.6, cfg_f)
    summ = FF.far_fit_summary(sel, fit, None, None, fb["se"], fb["n_ok"], 0.0, 0.6, 0.01, cfg_f, seed=4, boot=fb)
    from scipy import stats
    assert summ["t_df"] == 9 and abs(summ["t_quantile"] - stats.t.ppf(0.975, 9)) < 1e-12, summ["t_df"]
    assert summ["bootstrap"]["unit"] == "cell" and summ["bootstrap"]["n_cells"] == 10
    summ_n = FF.far_fit_summary(sel, fit, None, None, fb["se"], fb["n_ok"], 0.0, 0.6, 0.01, cfg_f, seed=4)
    assert summ_n["t_df"] == 19 and summ_n["bootstrap"] is None, summ_n["t_df"]
    # (e) determinism
    a1 = CAL.anchor_se(five, d, np.random.default_rng(8), n_resamples=30)
    a2 = CAL.anchor_se(five, d, np.random.default_rng(8), n_resamples=30)
    assert a1 == a2, (a1, a2)
    # (f) refusals
    for bad in (dict(bootstrap_cell_min=1), dict(bootstrap_cell_min=10.0), dict(bootstrap_unit="both")):
        try:
            dataclasses.replace(d, calibration=dataclasses.replace(d.calibration, **bad)).validate()
        except ValueError:
            continue
        raise AssertionError("calibration %s must be refused" % bad)
    try:
        CAL.resample(cells, np.random.default_rng(0), "sweep")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown resampling unit must be refused")


def exact_near(cell, n_nodes, a, dz, start=0):
    """Exact near scans of one cell (qualified ids): the axis on plane 0, 7 planes, omega = c_n + a 0.3 delta^2."""
    ks = np.arange(-3, 4)
    out = []
    for n in range(start, start + n_nodes):
        om = 0.03 + 0.01 * n + a * 0.3 * (ks * dz) ** 2
        m = ks.size
        out.append(CAL.PlaneScan(node_id=cell * CP.QUALIFIER + n, d_hat_um=0.4, phi_rad=0.0, alpha_hat=0.2, k_star=0,
                                 z_sub_um=0.0, ks=ks, z_um=ks * dz, omega_um2=om, vw_um2=om.copy(),
                                 area_gl_um=np.full(m, 5.0), depth_gl=np.full(m, 20.0), B_bar=np.full(m, 200.0),
                                 status=tuple("ok" for _ in range(m)), set="near"))
    return out


def test_cell_weights():
    d = default_config()
    assert d.calibration.pool_weighting == "cell"
    d_node = dataclasses.replace(d, calibration=dataclasses.replace(d.calibration, pool_weighting="node"))
    dz = d.acquisition.dz_um
    A, B = exact_near(1, 6, 1.0, dz), exact_near(2, 2, 2.0, dz)
    gA, gB = CAL.fit_growth(A, d).g_um2, CAL.fit_growth(B, d).g_um2
    g_node, g_cell = CAL.fit_growth(A + B, d_node), CAL.fit_growth(A + B, d)
    ok = np.isfinite(gA) & np.isfinite(gB)
    assert ok.sum() >= 3, gA
    # (a) known answer: Eq. 4 is linear in (c, G) at fixed axes, the axes stay on plane 0 by symmetry, each node's
    # c_i is free, so the pooled G is the cells' own G averaged with their total weights; 1e-8 um^2 is far above
    # the solver's 1e-12 tolerances on a cost of order 1e-2 and far below the 0.05-0.3 um^2 the two answers differ by
    want_node, want_cell = (6 * gA + 2 * gB) / 8, (gA + gB) / 2
    assert np.max(np.abs(g_node.g_um2[ok] - want_node[ok])) < 1e-8, (g_node.g_um2, want_node)
    assert np.max(np.abs(g_cell.g_um2[ok] - want_cell[ok])) < 1e-8, (g_cell.g_um2, want_cell)
    assert np.allclose(g_cell.node_weights, np.r_[np.full(6, 8 / 12.0), np.full(2, 8 / 4.0)], rtol=0, atol=1e-15)
    assert np.all(g_node.node_weights == 1.0)
    # (b) invariants: one cell; equal cells; a node dropped from cell A (2 finite planes < 3) leaves A two used
    # nodes, as many as B's: the weights are all 1 and the fits identical
    for scans in (A, A[:4] + exact_near(2, 4, 2.0, dz)):
        assert np.array_equal(CAL.fit_growth(scans, d).g_um2, CAL.fit_growth(scans, d_node).g_um2)
    A3 = exact_near(1, 3, 1.0, dz)
    bad = A3[0]
    om = bad.omega_um2.copy()
    om[2:] = np.nan
    A3[0] = dataclasses.replace(bad, omega_um2=om, vw_um2=om.copy())
    gd_cell, gd_node = CAL.fit_growth(A3 + B, d), CAL.fit_growth(A3 + B, d_node)
    assert len(gd_cell.dropped_nodes) == 1 and np.all(gd_cell.node_weights == 1.0)
    assert np.array_equal(gd_cell.g_um2, gd_node.g_um2)
    # (c) the labelled resample draws as the unlabelled one; node labels are the nodes' cells; cell copies 0..C-1
    cells = np.array([3, 3, 1, 1, 1, 2, 3])
    for seed in range(5):
        i1 = CAL.resample(cells, np.random.default_rng(seed), "cell")
        i2, lab = CAL.resample(cells, np.random.default_rng(seed), "cell", with_labels=True)
        assert np.array_equal(i1, i2) and sorted(set(lab.tolist())) == [0, 1, 2]
        members = {c: list(np.flatnonzero(cells == c)) for c in (1, 2, 3)}
        for t in range(3):                       # copy t: one contiguous run holding one whole cell, in order
            run = np.flatnonzero(lab == t)
            assert np.array_equal(run, np.arange(run[0], run[-1] + 1)), (lab, i2)
            owner = set(cells[i2[run]].tolist())
            assert len(owner) == 1 and list(i2[run]) == members[owner.pop()], (lab, i2)
        j1 = CAL.resample(cells, np.random.default_rng(seed), "node")
        j2, labn = CAL.resample(cells, np.random.default_rng(seed), "node", with_labels=True)
        assert np.array_equal(j1, j2) and np.array_equal(labn, cells[j2])
    w = CAL.cell_weights(np.array([0, 0, 1, 1, 1, 1]), d)
    assert np.allclose(w, [1.5, 1.5, 0.75, 0.75, 0.75, 0.75]) and abs(w.mean() - 1.0) < 1e-15
    # (d) the far fit: cell A 6 nodes at gamma 0.6, cell B 2 nodes at 1.0, exact F1 scans, unweighted rounds
    from allen_diameter.analysis import far_field as FF
    rng = np.random.default_rng(11)
    far = [far_scan(1 * CP.QUALIFIER + j, d, 0.6, 0.5, rng) for j in range(6)] + \
          [far_scan(2 * CP.QUALIFIER + j, d, 1.0, 0.5, rng) for j in range(2)]
    sel = FF.select_far_planes(far, d)
    assert sel.node_ids.size == 8, sel.dropped
    f_node = FF.fit_far_slope(sel, 0.5, d_node, weights="none")
    f_cell = FF.fit_far_slope(sel, 0.5, d, weights="none")
    assert 0.6 < f_node.gamma < f_cell.gamma < 1.0, (f_node.gamma, f_cell.gamma)
    assert abs(f_cell.gamma - 0.8) < abs(f_node.gamma - 0.8), (f_node.gamma, f_cell.gamma)
    sel1 = FF.select_far_planes(far[:6], d)
    assert FF.fit_far_slope(sel1, 0.5, d).gamma == FF.fit_far_slope(sel1, 0.5, d_node).gamma
    # (e) the CLI records the weighting and prints it for two cells or more
    with tempfile.TemporaryDirectory() as tmp:
        sp, cp = os.path.join(tmp, "scans.json"), os.path.join(tmp, "calib.json")
        CIO.save_scans(A + B, sp, {})
        buf = io.StringIO()
        with redirect_stdout(buf):
            CK.run_fit(dataclasses.replace(d, calibration=dataclasses.replace(d.calibration, bootstrap_resamples=5)),
                       sp, cp)
        rec = CIO.load_calibration(cp)
        assert rec["meta"]["weighting"] == dict(unit="cell", n_cells=2), rec["meta"].get("weighting")
        assert "each of the 2 cells weighs the same (D-120.2)" in buf.getvalue(), buf.getvalue()[:400]
        assert len(rec["growth"]["node_weights"]) == 8
    # (g) on noisy cells of unequal size (8, 3, 5 nodes) the weighted fit is the same with the sparse and the dense
    # Jacobian at the table's knots (<= delta_0 = 0.84 um; the knot beyond, extrapolated, is flat enough for the two
    # stopping points to differ by 1e-6 um^2 at equal cost, weighted or not); 1e-7 um^2 is far below the 5e-2 an
    # unweighted Jacobian gives; and residual_um2 is the unweighted model minus data, recomputed here from the
    # fit's own c, axes and G
    d_dense = dataclasses.replace(d, calibration=dataclasses.replace(d.calibration, growth_jacobian="dense"))
    noisy = [sc for sc in near_cells(3, 8, 0.4, np.random.default_rng(3), noise=0.01)
             if not (sc.node_id // CP.QUALIFIER == 2 and sc.node_id % CP.QUALIFIER >= 3)
             and not (sc.node_id // CP.QUALIFIER == 3 and sc.node_id % CP.QUALIFIER >= 5)]
    gs, gdn = CAL.fit_growth(noisy, d), CAL.fit_growth(noisy, d_dense)
    okn = np.isfinite(gs.g_um2) & np.isfinite(gdn.g_um2) & (gs.knots_um <= d.calibration.table_max_delta_um + 1e-9)
    assert len(set(np.round(gs.node_weights, 12).tolist())) == 3, gs.node_weights
    assert np.max(np.abs(gs.g_um2[okn] - gdn.g_um2[okn])) < 1e-7, np.max(np.abs(gs.g_um2[okn] - gdn.g_um2[okn]))
    om_obs = np.concatenate([sc.values(d.calibration.statistic)[np.isfinite(sc.values(d.calibration.statistic))]
                             for sc in noisy])
    model = gs.c_um2[gs.obs_node] + gs.growth(gs.delta_um)
    assert np.max(np.abs(gs.residual_um2 - (model - om_obs))) < 1e-12
    # (f) the bootstraps refit each resample with its copies as cells: near, cells of 6 and 2 nodes -- every draw
    # is one cell's own anchor or the two cells' equal-weight anchor, never the 6:2 one (exact data: 1e-9 um)
    a_A, a_B = CAL.anchor_value(CAL.fit_growth(A, d), d), CAL.anchor_value(CAL.fit_growth(B, d), d)
    a_cell, a_node = CAL.anchor_value(g_cell, d), CAL.anchor_value(g_node, d)
    assert abs(a_cell - a_node) > 1e-3, (a_cell, a_node)
    _se, _n, draws = CAL.bootstrap_sigma0(A + B, d, np.random.default_rng(2), n_resamples=40, unit="cell")
    near_ok = [min(abs(x - a_A), abs(x - a_B), abs(x - a_cell)) < 1e-9 for x in draws]
    assert all(near_ok) and any(abs(x - a_cell) < 1e-9 for x in draws), (draws, a_A, a_B, a_cell, a_node)
    # far, three cells of 2 nodes (0.6, 0.6, 1.0): a copy of a cell is a cell, so with equal cells every draw
    # equals the resample's fit with every node weighing the same (a cell drawn twice weighing twice)
    rng = np.random.default_rng(12)
    far3 = [far_scan(c * CP.QUALIFIER + j, d, g, 0.5, rng) for c, g in ((1, 0.6), (2, 0.6), (3, 1.0)) for j in range(2)]
    sel3 = FF.select_far_planes(far3, d)
    cells3 = CAL.node_cells(sel3.node_ids)
    rng_a, rng_b = np.random.default_rng(13), np.random.default_rng(13)
    _se, _n, fdraws = FF.bootstrap_gamma(sel3, 0.5, d, rng_a, n_resamples=30, unit="cell")
    want = []
    for _ in range(30):
        pick = CAL.resample(cells3, rng_b, "cell")
        want.append(FF.fit_far_slope(sel3.subset(pick), 0.5, d_node).gamma)
    assert fdraws.size == 30 and np.max(np.abs(fdraws - np.array(want))) < 1e-9, (fdraws, want)
    # refusal
    try:
        dataclasses.replace(d, calibration=dataclasses.replace(d.calibration, pool_weighting="cells")).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown pool_weighting must be refused")


def noisy_near(n_nodes, a, dz, rng, noise=0.004):
    """One cell's near scans (unqualified ids 0..n-1): the axis on plane 0, 7 planes, omega = c_n + a 0.3 delta^2
    plus Gaussian noise of SD noise (um^2)."""
    ks = np.arange(-3, 4)
    out = []
    for n in range(n_nodes):
        om = 0.03 + 0.01 * n + a * 0.3 * (ks * dz) ** 2 + noise * rng.standard_normal(ks.size)
        m = ks.size
        out.append(CAL.PlaneScan(node_id=n, d_hat_um=0.4, phi_rad=0.0, alpha_hat=0.2, k_star=0, z_sub_um=0.0, ks=ks,
                                 z_um=ks * dz, omega_um2=om, vw_um2=om.copy(), area_gl_um=np.full(m, 5.0),
                                 depth_gl=np.full(m, 20.0), B_bar=np.full(m, 200.0),
                                 status=tuple("ok" for _ in range(m)), set="near"))
    return out


def test_cell_outliers():
    from scipy import stats
    from allen_diameter.analysis import cell_outliers as CO
    import survey_cells as SC
    d = default_config()
    # (a) the ANOVA: F = MS_B / MS_W equals scipy's one-way F on unequal groups; n0 and sigma_b^2 by hand
    A, B, C3 = [1.0, 1.2], [0.5, 0.7, 0.6], [2.0, 2.2, 1.9, 2.1]
    an = CO.one_way_anova(np.array(A + B + C3), [1] * 2 + [2] * 3 + [3] * 4)
    assert abs(an["ms_b"] / an["ms_w"] - stats.f_oneway(A, B, C3).statistic) < 1e-12 * an["ms_b"] / an["ms_w"]
    n0 = (9 - (4 + 9 + 16) / 9.0) / 2.0
    gm = np.mean(A + B + C3)
    msb = (2 * (np.mean(A) - gm) ** 2 + 3 * (np.mean(B) - gm) ** 2 + 4 * (np.mean(C3) - gm) ** 2) / 2.0
    msw = (np.var(A) * 2 + np.var(B) * 3 + np.var(C3) * 4) / 6.0
    assert abs(an["n0"] - n0) < 1e-12 and abs(an["ms_b"] - msb) < 1e-12 and abs(an["ms_w"] - msw) < 1e-12
    assert abs(an["sigma_b2"] - max(0.0, (msb - msw) / n0)) < 1e-12 and an["n_cells"] == 3 and an["n_nodes"] == 9
    r3 = next(r for r in CO.assess_set([1] * 2 + [2] * 3 + [3] * 4, np.array(A + B + C3), d)["rows"]
              if r["specimen_id"] == 3)
    assert abs(r3["ref_mean_um"] - (np.mean(A) + np.mean(B)) / 2.0) < 1e-15, r3     # each other cell equal, not 2:3
    # (b) thirteen cells of five node values (between SD 0.01 um, within 0.02 um), cell 99 shifted by 0.2 um
    rng = np.random.default_rng(5)
    labels, values = [], []
    for c in list(range(1, 13)) + [99]:
        eff = 0.01 * rng.standard_normal() + (0.2 if c == 99 else 0.0)
        for _ in range(5):
            labels.append(c)
            values.append(eff + 0.02 * rng.standard_normal())
    values = np.array(values)
    rec = CO.assess_set(labels, values, d)
    assert rec["applied"] and rec["removed"] == [99], rec["removed"]
    assert abs(rec["p_threshold"] - 0.05 / 13) < 1e-15
    row99 = next(r for r in rec["rows"] if r["specimen_id"] == 99)
    assert row99["removed"] and row99["p"] < rec["p_threshold"] < min(r["p"] for r in rec["rows"] if r["specimen_id"] != 99)
    six = [i for i, c in enumerate(labels) if c in (1, 2, 3, 4, 5, 99)]
    neg = values.copy()
    neg[np.array(labels) == 99] -= 0.4                       # the same cell 0.2 um below instead of above
    assert CO.assess_set(labels, neg, d)["removed"] == [99]
    rec6 = CO.assess_set([labels[i] for i in six], values[six], d)
    z6 = {r["specimen_id"]: abs(r["z"]) for r in rec6["rows"]}
    assert not rec6["applied"] and rec6["removed"] == [] and max(z6, key=z6.get) == 99, z6
    # (c) leave-one-out: cell 99 judged against a yardstick that includes it has a smaller z
    full = CO.one_way_anova(values, labels)
    i99 = full["cells"].index(99)
    z_in = (full["means"][i99] - np.mean(full["means"])) / math.sqrt(full["sigma_b2"] + full["sigma_w2"] / 5)
    assert row99["z"] > z_in > 0, (row99["z"], z_in)
    # (d) a one-node cell 0.05 um off where the within SD is 0.05 um: kept, because its mean carries sigma_w^2 / 1;
    # a yardstick of sigma_b^2 alone would remove it
    rng = np.random.default_rng(6)
    lab2, val2 = [], []
    for c in range(1, 13):
        eff = 0.01 * rng.standard_normal()
        for _ in range(6):
            lab2.append(c)
            val2.append(eff + 0.05 * rng.standard_normal())
    others = CO.one_way_anova(np.array(val2), lab2)
    lab2.append(77)
    val2.append(float(np.mean(others["means"])) + 0.05)
    rec2 = CO.assess_set(lab2, np.array(val2), d)
    row77 = next(r for r in rec2["rows"] if r["specimen_id"] == 77)
    assert not row77["removed"] and abs(row77["z"]) < 2.0, row77
    z_alone = 0.05 / math.sqrt(max(row77["sigma_b2_um2"], 1e-30))
    assert 2.0 * stats.norm.sf(z_alone) < rec2["p_threshold"], (z_alone, row77)
    # node_values: a node's value is the mean of its used, finite residuals; other points ignored
    kc = dict(points=[dict(set="near", used=True, residual_um=0.1, node_id=1, cell=4),
                      dict(set="near", used=True, residual_um=0.3, node_id=1, cell=4),
                      dict(set="near", used=False, residual_um=9.0, node_id=1, cell=4),
                      dict(set="near", used=True, residual_um=float("nan"), node_id=2, cell=5),
                      dict(set="far", used=True, residual_um=0.5, node_id=3, cell=5)])
    labs, vals = CO.node_values(kc, "near")
    assert labs == [4] and np.allclose(vals, [0.2]), (labs, vals)
    # (e) end to end: twelve cells of six near nodes (cell 12 with twice the growth) and two far nodes each; few
    # resamples for speed; then the same directory without the removal: the first run's conditional files go
    cfg = dataclasses.replace(d, calibration=dataclasses.replace(d.calibration, bootstrap_resamples=4))
    dz = cfg.acquisition.dz_um
    rng = np.random.default_rng(7)
    per_cell = [(s_, noisy_near(6, 2.0 if s_ == 12 else 1.0, dz, rng) + [far_scan(100 + j, cfg, 0.6, 0.5, rng)
                                                                        for j in range(2)]) for s_ in range(1, 13)]
    pooled, meta = CP.pool_scans(per_cell, cfg)
    with tempfile.TemporaryDirectory() as s3c:
        CIO.save_scans(pooled, os.path.join(s3c, "scans_pooled.json"), dict(meta, source="pooled"))
        lines = []
        with redirect_stdout(io.StringIO()):
            done, out = SC.pooled_fits(cfg, pooled, meta, s3c, log=lines.append, figures=False)
        assert done["fit"] and done["far_fit"], (done, lines)
        assert out["near"]["removed"] == [12] and out["near"]["applied"], out["near"]["rows"]
        assert out["far"]["n_cells"] == 12 and out["far"]["removed"] == [], out["far"]["rows"]
        for name in ("calibration_pooled_all.json", "far_fit_pooled_all.json", "scans_pooled_kept.json",
                     "cell_outliers.json", "cell_outliers.csv", "calibration_pooled.json", "far_fit_pooled.json",
                     "kernel_law_by_cell.csv"):
            assert os.path.exists(os.path.join(s3c, name)), name
        final = CIO.load_calibration(os.path.join(s3c, "calibration_pooled.json"))
        first = CIO.load_calibration(os.path.join(s3c, "calibration_pooled_all.json"))
        ids_final = [CP.cell_of(q, meta["id_map"])[0] for q in final["growth"]["node_ids"]]
        ids_first = [CP.cell_of(q, meta["id_map"])[0] for q in first["growth"]["node_ids"]]
        assert 12 not in ids_final and 12 in ids_first and len(set(ids_final)) == 11, set(ids_final)
        kept, kmeta = CIO.load_scans(os.path.join(s3c, "scans_pooled_kept.json"))
        kept12 = [sc.set for sc in kept if CP.cell_of(sc.node_id, meta["id_map"])[0] == 12]
        assert kmeta["removed"] == {"near": [12]} and kept12 == ["far", "far"], (kmeta.get("removed"), kept12)
        assert any("removed [12]" in ln for ln in lines), lines
        off = dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, cell_outlier_removal=False))
        with redirect_stdout(io.StringIO()):
            done, out = SC.pooled_fits(off, pooled, meta, s3c, log=lines.append, figures=False)
        assert done["fit"] and out is None
        for name in ("calibration_pooled_all.json", "far_fit_pooled_all.json", "scans_pooled_kept.json",
                     "cell_outliers.json", "cell_outliers.csv"):
            assert not os.path.exists(os.path.join(s3c, name)), name
        again = CIO.load_calibration(os.path.join(s3c, "calibration_pooled.json"))
        assert 12 in [CP.cell_of(q, meta["id_map"])[0] for q in again["growth"]["node_ids"]]


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
