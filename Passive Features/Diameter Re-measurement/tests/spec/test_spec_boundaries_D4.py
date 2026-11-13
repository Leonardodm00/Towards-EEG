"""Boundary tests of run D4 (integration tester, 2026-11-13, commit 1555036):
the producer -> consumer pairs between the changed Blocks 10, 15, 19 and their
neighbours 3, 4, 5, 6, 8, 13, 14, 16, 17. Written from specs/SPEC.md; expected
values from spec_helpers_D4 or by hand.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_boundaries_D4.py -q
"""
from __future__ import annotations

import dataclasses
import json
import math
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import spec_helpers_D4 as D4

from allen_diameter.analysis import calibration as CAL
from allen_diameter.analysis import calibration_pool as POOL
from allen_diameter.analysis import calibration_report as CR
from allen_diameter.analysis import candidate_search as CSR
from allen_diameter.analysis import far_field as FF
from allen_diameter.analysis import phantoms
from allen_diameter.loading import calibration_io
from allen_diameter.model import kernel as KER

NAN = float("nan")
Q = 10 ** 6


def table_growth(cfg, n=10, seed=60, sigma_r0=None):
    """Exact near scans generated from the default table's sigma at the knots (so the anchor is
    sigma_r(0.84) = 0.603 um) and their growth fit."""
    r = cfg.renderer
    sig = D4.ref_sigma_r(np.arange(5) * D4.DZ, r.kernel_table_delta_um, r.kernel_table_sigma_um,
                         r.kernel_continuation_slope)
    G = sig ** 2 - r.sigma_r0_um ** 2
    rng = np.random.default_rng(seed)
    k = rng.integers(0, 40, n)
    z_ax = k * D4.DZ + rng.uniform(-0.1, 0.1, n)
    scans = D4.eq4_scans(G, rng.uniform(0, 0.03, n), z_ax, z_ax + rng.uniform(-0.03, 0.03, n), k)
    return scans, CAL.fit_growth(scans, cfg), sig, G


def test_block10_kernel_reaches_block4_sigma_r():
    """OB3: the table kernel_from_growth writes (cut at 0.84 um) is what model.kernel.sigma_r renders -- the knots
    exactly, numpy.interp between, and beyond the last knot the linear continuation from the anchor with the
    configured slope and with a far-fit slope (1e-12: the same table read twice)."""
    cfg = D4.cfg_with()
    _scans, g, sig, _G = table_growth(cfg)
    k = CAL.kernel_from_growth(g, cfg.renderer, max_delta_um=cfg.calibration.table_max_delta_um)
    d = np.array([0.0, 0.1, 0.28, 0.5, 0.84, 1.0, 2.5, 4.48, -0.6, -3.0])
    want = D4.ref_sigma_r(d, k.kernel_table_delta_um, k.kernel_table_sigma_um, k.kernel_continuation_slope)
    assert_allclose(KER.sigma_r(d, k), want, rtol=1e-12)
    assert_allclose(KER.sigma_r(np.array([0.84]), k), sig[3], atol=1e-9)
    k2 = dataclasses.replace(k, kernel_continuation="linear", kernel_continuation_slope=1.21)
    assert_allclose(KER.sigma_r(np.array([2.0]), k2), k.kernel_table_sigma_um[-1] + 1.21 * (2.0 - 0.84), rtol=1e-12)


def test_block10_anchor_feeds_block15_far_fit():
    """OB4: the anchor read from the near growth (anchor_value) is the sigma_0 the far fit holds fixed: far data
    generated with that anchor and gamma 0.79 are recovered to 1e-8."""
    cfg = D4.cfg_with()
    _scans, g, sig, _G = table_growth(cfg)
    s0 = CAL.anchor_value(g, cfg)
    assert_allclose(s0, sig[3], atol=1e-9)
    J = 6
    rng = np.random.default_rng(61)
    kf = np.arange(J) * 40
    zp = kf * D4.DZ + rng.uniform(-0.1, 0.1, J)
    far = D4.f1_scans(rng.uniform(0, 0.03, J), np.zeros(J), zp, kf, s0, 0.79, amp=60.0, rms=0.5)
    sel = FF.select_far_planes(far, cfg)
    assert sel.node_ids.size == J and sel.omega_um2.size == J * 24
    fit = FF.fit_far_slope(sel, s0, cfg)
    assert_allclose(fit.gamma, 0.79, atol=1e-8)


def test_block10_scans_file_reaches_block19_cells(tmp_path):
    """OB5: a pooled scans file round trip keeps the set, the qualified ids (-> the specimens of node_cells),
    bg_sd_gl (10.G) and the widths bit for bit, and the growth fit on the loaded scans equals the fit in memory."""
    cfg = D4.cfg_with()
    per_cell = []
    for sid, seed in ((404, 1), (101, 2), (202, 3)):
        sc, _g, _s, _G = table_growth(cfg, n=4, seed=seed)
        sc = [dataclasses.replace(s, bg_sd_gl=np.full(s.ks.size, 2.5)) for s in sc]
        per_cell.append((sid, sc))
    pooled, meta = POOL.pool_scans(per_cell, cfg)
    path = str(tmp_path / "scans_pooled.json")
    calibration_io.save_scans(pooled, path, meta)
    back, meta2 = calibration_io.load_scans(path)
    assert [s.node_id for s in back] == [s.node_id for s in pooled]
    assert_array_equal(CAL.node_cells([s.node_id for s in back]), [s.node_id // Q for s in pooled])
    for a, b in zip(pooled, back):
        assert a.set == b.set and np.array_equal(a.omega_um2, b.omega_um2) and np.array_equal(a.bg_sd_gl, b.bg_sd_gl)
    g1, g2 = CAL.fit_growth(pooled, cfg), CAL.fit_growth(back, cfg)
    assert_array_equal(g1.g_um2, g2.g_um2)
    assert_array_equal(g1.node_weights, np.ones(12))          # three cells of four nodes: equal weights
    assert meta2["id_map"] == meta["id_map"]


def test_block13_pilot_view_reaches_calibration_reasons():
    """OB1: calibration_reasons reads the pilot (pilot_view), not the deliverable: a node whose deliverable fails
    S and is dark but whose pilot passes is a calibration node, and the reverse fails by the pilot's status."""
    cfg = D4.cfg_with()
    good_pilot = D4.node_result(fit_status="failed", flags=("crossing",), alpha_hat=3.0, d_hat_um=2.0,
                                pilot_status="converged", pilot_flags=(), alpha_pilot=0.2, d_pilot_um=0.3,
                                mu_pilot_per_um=0.6, v0_pilot_um=0.0, B_bar_pilot=180.0)
    assert CAL.calibration_reasons(good_pilot, cfg, 0.4) == []
    bad_pilot = D4.node_result(fit_status="converged", flags=(), pilot_status="at_bound",
                               pilot_flags=("crossing",), alpha_pilot=0.2, d_pilot_um=0.3, mu_pilot_per_um=0.6,
                               v0_pilot_um=0.0, B_bar_pilot=180.0)
    assert CAL.calibration_reasons(bad_pilot, cfg, 0.4) == ["status:at_bound", "crossing"]


def test_block6_reject_reasons_reach_selection_reasons():
    """OB2: selection_reasons = Block 6's reject_reasons by default (no faint since D-096); with
    node_faint_flag_check the flag faint first among the flags."""
    cfg = D4.cfg_with()
    r = D4.node_result(flags=("faint", "stack_edge", "dark", "crossing"))
    base = phantoms.reject_reasons(r)
    assert "faint" not in base and "dark" not in base
    assert CAL.selection_reasons(r, cfg) == base
    on = D4.cfg_with(calibration=dict(node_faint_flag_check=True))
    got = CAL.selection_reasons(r, on)
    flags_part = [x for x in got if x in ("faint",) + phantoms.SELECTION_FLAGS]
    assert flags_part[0] == "faint" and set(flags_part[1:]) == {"stack_edge", "crossing"}


def test_block17_csv_rows_are_read_as_the_pilot():
    """OB6: the per-node CSV rows give the candidates and the typical background through the pilot's columns
    (pilot_row_view): a row whose deliverable fails but whose pilot passes is a candidate and enters the medians;
    the reverse does not."""
    import calibrate_kernel as CK
    cfg = D4.cfg_with()
    rows = [dict(node_id="5", fit_status="failed", flags="crossing", phi_rad="0.0", alpha_hat="3", allen_radius_um="0.2",
                 B_bar="999", mu_hat_per_um="9", pilot_status="converged", pilot_flags="", alpha_pilot="0.2",
                 B_bar_pilot="150", mu_pilot_per_um="0.3"),
            dict(node_id="6", fit_status="converged", flags="", phi_rad="0.0", alpha_hat="0.2", allen_radius_um="0.2",
                 B_bar="160", mu_hat_per_um="0.4", pilot_status="at_bound", pilot_flags="", alpha_pilot="0.2",
                 B_bar_pilot="777", mu_pilot_per_um="7")]
    assert CK.candidates_from_rows(rows, cfg) == {5}
    B, mu = CSR.typical_background(rows, cfg)
    assert (B, mu) == (150.0, 0.3)


def straight_swc(n=8, radius=0.25, step=0.6):
    """One straight flat dendrite along x at y = 0, z = 1.4 um, without a soma (root parent -1), so that nothing
    shades it: D_A = 2 radius."""
    from allen_diameter.loading.swc_io import SWC
    ids = np.arange(1, n + 1)
    xyz = np.column_stack([60.0 + step * np.arange(n), np.zeros(n), np.full(n, 1.4)])
    return SWC(ids, np.full(n, 3), xyz, np.full(n, radius), np.r_[-1, ids[:-1]], ["%d\n" % i for i in ids],
               list(range(n)), "")


def test_block16_transform_moves_the_search_rigidly():
    """OB7: the search reads the trace in the image frame of the cell's transform; an integer full-resolution
    pixel shift moves every record's x by that many pitches and changes no verdict, plane or pixel count (the
    patch frames move by whole pixels)."""
    cfg = D4.cfg_with(calibration=dict(patch_half_um=6.0))
    swc = straight_swc()
    a = CSR.search(swc, cfg, "near", transform=None, B_bar=150.0, mu=0.3)
    b = CSR.search(swc, cfg, "near", transform=dict(shift_full_px=(7.0, -3.0)), B_bar=150.0, mu=0.3)
    assert len(a) == len(b) == 8
    assert all(r["reason"] == "" and r["n_pixels_k"] > 0 for r in a)       # isolated: every node passes
    p = cfg.acquisition.res0_um
    for ra, rb in zip(a, b):
        assert ra["node_id"] == rb["node_id"] and ra["reason"] == rb["reason"]
        assert (ra["n_pixels_k"], ra["n_fit"], ra["n_fit_usable"], ra["k_trace"]) == \
            (rb["n_pixels_k"], rb["n_fit"], rb["n_fit_usable"], rb["k_trace"])
        assert_allclose([rb["x_um"] - ra["x_um"], rb["y_um"] - ra["y_um"]], [7 * p, -3 * p], atol=1e-9)


def test_block14_hashes_ignore_calibration_and_survey_settings():
    """OB8: no CalibrationConfig or SurveyConfig field is in a signature; the kernel table is in the simulation
    hash, not in the estimator's."""
    cfg = D4.cfg_with()
    h = {w: cfg.signature_hash(w) for w in ("estimator", "simulation", "full")}
    other = D4.cfg_with(calibration=dict(growth_jacobian="dense", bootstrap_unit="node", bootstrap_cell_min=5,
                                         pool_weighting="node", cell_outlier_removal=False, cell_outlier_alpha=0.01,
                                         gof_kappa=2.0, isolation_depth_gl=1.0, far_offsets_planes=12,
                                         calib_min_nodes_per_set=5),
                        survey=dict(calib_scans_per_cell=5, calib_pool_max_per_set=30))
    assert {w: other.signature_hash(w) for w in h} == h
    _s, g, _sig, _G = table_growth(cfg)
    k = CAL.kernel_from_growth(g, cfg.renderer, max_delta_um=0.84)
    with_k = dataclasses.replace(cfg, renderer=dataclasses.replace(k, kernel_continuation_slope=1.0))
    assert with_k.signature_hash("estimator") == h["estimator"]
    assert with_k.signature_hash("simulation") != h["simulation"]


@pytest.mark.parametrize("s0", [None, 0.12])
def test_block10_fit_sigma_r0_reaches_block19_law(tmp_path, s0):
    """Boundary 10 -> 19.5: `fit --sigma-r0 S` writes a table whose sigma(0) is S (Block 10: sigma_r(0)
    'configured ... or given'). On exact near scans with the axes on the planes every |delta_hat| is a knot,
    so the near points sigma = (omega - c + sigma_r(0)^2)^(1/2) of 19.5 lie on the pooled law there: residual 0
    to 1e-6 um (the fit's 1e-9 recovery through a square root) -- whichever sigma_r(0) the fit was given."""
    import calibrate_kernel as CK
    cfg = D4.cfg_with()
    r = cfg.renderer
    sig = D4.ref_sigma_r(np.arange(5) * D4.DZ, r.kernel_table_delta_um, r.kernel_table_sigma_um,
                         r.kernel_continuation_slope)
    G = sig ** 2 - r.sigma_r0_um ** 2
    n = 10
    k = np.arange(n) * 9
    scans = D4.eq4_scans(G, np.linspace(0.0, 0.03, n), k * D4.DZ, k * D4.DZ, k)
    p = str(tmp_path / "scans.json")
    calibration_io.save_scans(scans, p, {})
    cj = str(tmp_path / "cfg.json")
    with open(cj, "w") as f:
        json.dump(dict(calibration=dict(bootstrap_resamples=5)), f)
    out = str(tmp_path / "C.json")
    argv = ["fit", "--scans", p, "--out", out, "--config-json", cj] + ([] if s0 is None else ["--sigma-r0", str(s0)])
    CK.main(argv)
    calib = calibration_io.load_calibration(out)
    assert_allclose(calib["kernel"]["kernel_table_sigma_um"][0], r.sigma_r0_um if s0 is None else s0)
    kc = CR.kernel_law_by_cell(scans, cfg, calib, None)
    near = [q for q in kc["points"] if q["set"] == "near"]
    d_T = calib["kernel"]["kernel_table_delta_um"][-1]              # 3 x 0.28 = 0.8400000000000001 in float64
    beyond = [q for q in near if q["abs_delta_um"] > d_T]          # past the table, no far fit: NaN (19.5 (f))
    assert all(math.isnan(q["residual_um"]) for q in beyond)
    pts = [q for q in near if q["abs_delta_um"] <= d_T]
    assert len(pts) >= 50
    res = np.array([q["residual_um"] for q in pts])
    assert np.max(np.abs(res)) <= 1e-6, "largest |residual| %.4g um at the knots (sigma_r0 %s)" % (
        np.max(np.abs(res)), s0)
