"""Boundary tests of Block 11 with its neighbours (Blocks 0, 4, 5, 6, 8, 10, 12,
13), integration tester run D2, 2026-11-13, commit 1555036. Each test checks
that what a producer returns is what its consumer reads (shape, units, index
base, columns, the pilot/deliverable distinction of Block 13.3b). Oracle IDs
B1-B8 are those of specs/test_reports/2026-11-13_0830_incremental_D2.md.

Run from `Passive Features/Diameter Re-measurement/`:
    python -m pytest -q tests/spec/test_spec_boundaries_D2.py
Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math
import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spec_helpers_D2 as H

from allen_diameter.analysis import calibration, camera_fit, cell, phantoms, profiles, survey  # noqa: E402
from allen_diameter.analysis import fit_variants as FV  # noqa: E402
from allen_diameter.analysis import plane_iteration as PI  # noqa: E402
from allen_diameter.analysis.node_pipeline import pilot_view  # noqa: E402
from allen_diameter.loading import table_io  # noqa: E402
from allen_diameter.model import camera  # noqa: E402

CFG = H.default_cfg()
M = CFG.measure


def _eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    return a == b


@pytest.fixture(scope="module")
def node():
    """(measure_node's result, Allen radius, Branch, index) of node 4 of the synthetic cell."""
    return H.measured_node(4)


# ------------------------------------------------------------ B1: Block 11 -> Block 8 ---

def test_B1_pilot_row_view_of_pilot_row_is_the_pilots_row(node):
    """B1: Block 8's reader of a pilot row (cell.pilot_row_view) gives the row Block 11 writes for the pilot
    itself (pilot_row(pilot_view(result))) on every column the pilot/deliverable split touches."""
    res, r_allen, _b, _i = node
    assert res.pilot_status, "the fixture's node has no pilot (measure_node did not run its pilot?)"
    row = survey.pilot_row(res, r_allen, CFG)
    seen = cell.pilot_row_view(row)
    want = survey.pilot_row(pilot_view(res), r_allen, CFG)
    for col, _pcol in cell.PILOT_COLUMNS:
        assert _eq(seen[col], want[col]), (col, seen[col], want[col])
    for col in ("k_star", "z_sub_um", "cx_um", "cy_um", "theta_rad", "phi_rad", "calibration_node", "calibration_set"):
        assert _eq(seen[col], want[col]), col


def test_B1_csv_round_trip_of_pilot_rows(node):
    """B1 through the file: rows written by table_io.write_rows and read by read_rows, then pilot_row_view,
    keep the pilot's values (floats by repr, booleans, the ';'-joined flags) -- what node_planes and
    plane_differences select on."""
    res, r_allen, _b, _i = node
    row = survey.pilot_row(res, r_allen, CFG)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pilot_1.csv")
        table_io.write_rows([row], path)
        back = table_io.read_rows([path])[0]
    a, b = cell.pilot_row_view(row), cell.pilot_row_view(back)
    for col in ("d_hat_um", "mu_hat_per_um", "v0_hat_um", "alpha_hat", "B_bar", "fit_status", "flags", "in_S",
                "reject", "k_star", "k_star_depth", "z_sub_um", "phi_rad", "allen_radius_um", "calibration_node"):
        x, y = a[col], b[col]
        if isinstance(x, (float, np.floating)) and math.isnan(float(x)):
            assert isinstance(y, float) and math.isnan(y), col
        else:
            assert x == y, (col, x, y)


# ----------------------------------------------------------- B2: Block 11 -> Block 10 ---

def test_B2_calibration_reader_agrees_with_calibration_node(node):
    """B2: Block 10's CSV pre-filter (calibrate_kernel.candidates_from_rows) holds a node exactly when its pilot
    row says calibration_node, on hand variants of the node's row (flags, status, tilt, Allen radius) and after
    a CSV round trip (no registration verdict here: the pre-filter leaves it to the re-measurement)."""
    import calibrate_kernel
    res, r_allen, _b, _i = node
    p = pilot_view(res)
    variants = []
    for k, (flags, status, phi, ra) in enumerate([((), "converged", 0.0, 0.3), (("faint",), "converged", 0.0, 0.3),
                                                  (("crossing",), "converged", 0.0, 0.3), ((), "at_bound", 0.0, 0.3),
                                                  ((), "converged", 0.25, 0.3), ((), "converged", 0.0, 0.35),
                                                  ((), "converged", 0.2, 0.39), ((), "converged", 0.0, 0.5),
                                                  ((), "converged", 0.0, 0.0), (("dark",), "converged", 0.1, 0.25)]):
        r = dataclasses.replace(res, node_id=1000 + k, pilot_flags=tuple(flags), pilot_status=status, phi_rad=phi)
        variants.append(survey.pilot_row(r, ra, CFG))
    expect = {v["node_id"] for v in variants if v["calibration_node"]}
    assert calibrate_kernel.candidates_from_rows(variants, CFG) == expect
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pilot_2.csv")
        table_io.write_rows(variants, path)
        back = table_io.read_rows([path])
    assert calibrate_kernel.candidates_from_rows(back, CFG) == expect
    assert p.pilot_status == ""          # pilot_view clears the Block 13 fields


# ----------------------------------------------------------- B3: Block 11 -> Block 12 ---

def test_B3_sgt_start_fits_are_block11s_and_the_w_box_is_swappable():
    """B3: Block 12's two Gaussian start fits are Block 11's (gauss_fixed = the baseline, gauss_fitted the
    Gaussian of fitted width in FITTED_FWHM_BOUNDS_UM, equal to the default sgt_w_bounds_um); with another
    sgt_w_bounds_um in the configuration the SGT fit still runs and its w stays in the new box (swappability
    across the boundary)."""
    from allen_diameter.analysis import fit_sgt
    assert tuple(FV.FITTED_FWHM_BOUNDS_UM) == tuple(M.sgt_w_bounds_um)
    v, I, phi, B = H.noisy_profile(CFG, d=1.2, mu=0.7, v0=0.03, phi=0.15, B=175.0, noise_sd=1.0, seed=21)
    pf = fit_sgt.fit_plane(v, I, phi, B, M)
    plane = FV.PlaneFits(v, I, phi, B, M)
    for got, want in ((pf.gauss_fixed, plane.base), (pf.gauss_fitted, plane.unweighted(("gauss", None, False), True))):
        assert (got.d_um, got.mu_per_um, got.v0_um, got.fwhm_um) == (want.d_um, want.mu_per_um, want.v0_um,
                                                                     want.fwhm_um)
    m2 = dataclasses.replace(M, sgt_w_bounds_um=(0.3, 1.5))
    pf2 = fit_sgt.fit_plane(v, I, phi, B, m2)
    assert 0.3 <= pf2.final.w_um <= 1.5 and math.isfinite(pf2.final.d_um)


# ----------------------------------------------------------- B4: Block 13 -> Block 11 ---

def test_B4_pilot_view_of_measure_node_is_measure_pilot(node):
    """B4: what Block 11's diagnostics read as the pilot (pilot_view of measure_node) equals Block 5's chain
    (measure_pilot) field by field, the Block 3 fit included (NaN equal to NaN)."""
    res, _r, _b, _i = node
    ref, _b2, _i2 = H.measured_pilot(4)
    pv = pilot_view(res)
    for f in dataclasses.fields(ref):
        if f.name in ("plane_fit", "iteration"):
            continue
        a, b = getattr(pv, f.name), getattr(ref, f.name)
        if f.name == "fit":
            for g in dataclasses.fields(b):
                assert _eq(getattr(a, g.name), getattr(b, g.name)), ("fit", g.name)
            continue
        assert _eq(a, b), f.name


# ------------------------------------------------------- B5: Block 13 -> Block 11 scripts ---

def test_B5_refit_inputs_at_k_star_are_block5s(node):
    """B5: refit_at_plane(with_profile=True) on node_stack's square at k* hands the scripts (Cells 4k, 4l, 4m)
    Block 5's inputs in Block 5's units: v = profile_offsets (um), the profile through the pilot's centre
    (grey levels), the pilot's tilt (rad) and B_bar (grey levels); its d is the pilot's own (Block 13.3a)."""
    import plane_differences as PD
    cfg, swc, provider, *_ = H.synthetic_cell()
    st = PD.node_stack(swc, provider, cfg, 4, None, 6, None, 0.0, None, None, 3.0)
    p = st["result"]
    out = PD.refit_at_plane(st, cfg, int(p.k_star), with_profile=True)
    assert_allclose(out["v_um"], profiles.profile_offsets(M), rtol=0, atol=0)
    assert out["phi_rad"] == p.phi_rad and out["B_bar"] == p.B_bar
    assert out["d_hat_um"] == p.d_hat_um and out["v0_hat_um"] == p.v0_hat_um
    assert np.asarray(out["profile"]).dtype == np.float64 and np.asarray(out["profile"]).shape == out["v_um"].shape
    # the variants' baseline on these inputs is that same fit (Cell 4l's d_base = Cell 4j's d at k_iter)
    base = FV.fit_unweighted(np.asarray(out["v_um"]), np.asarray(out["profile"]), out["phi_rad"], out["B_bar"], M,
                             FV.VariantSpec("gauss", None, False, False))
    assert abs(base.d_um - out["d_hat_um"]) <= 1e-6


# ------------------------------------------------------------ B6: Block 4 -> Block 11 ---

def test_B6_post_chain_sd_is_the_camera_chain_on_a_flat_field():
    """B6: camera_fit.post_chain_sd is the clipped SD of Block 4's camera_chain applied to tau = 1 on 256 x 256
    pixels with the injected SD (the same generator state: equal to the bit); the noise match inverts the
    running maximum of that curve (np.interp in the test)."""
    rc = CFG.renderer
    for s in (0.5, 3.0):
        a = camera_fit.post_chain_sd(s, rc, np.random.default_rng(7))
        img = camera.camera_chain(np.ones((256, 256)), dataclasses.replace(rc, noise_sd_gl=s), np.random.default_rng(7))
        assert a == camera_fit.clipped_sd(img)
    grid = np.linspace(0.0, 6.0, 13)
    sigma, g, curve = camera_fit.noise_for_post_chain_sd(1.2, rc, 5, grid=grid, size=96)
    env = np.maximum.accumulate(np.nan_to_num(curve, nan=0.0))
    keep = np.r_[True, np.diff(env) > 0]
    assert sigma == pytest.approx(float(np.interp(1.2, env[keep], g[keep])), rel=1e-12)
    assert math.isnan(camera_fit.noise_for_post_chain_sd(1e6, rc, 5, grid=grid, size=96)[0])


# ------------------------------------------------------------ B7: Block 0 -> Block 11 ---

def test_B7_provider_contract_missing_planes():
    """B7: the provider the survey consumes keeps SPEC section 3's contract: block (n_planes, height, width)
    uint8, ks the requested planes, valid False and white (255) for planes the stack lacks, a frame whose
    left/top are the request's (full-resolution pixels) and res_um_px the pixel pitch."""
    cfg, swc, provider, fetcher, planes, _ = H.synthetic_cell()
    block, ks, valid, frame = provider(100, 20, 30, 12, -9, -5)
    assert block.dtype == np.uint8 and block.shape == (5, 12, 30)
    assert list(map(int, ks)) == [-9, -8, -7, -6, -5]
    assert list(map(bool, valid)) == [False, False, True, True, True]
    assert np.all(block[:2] == 255)
    assert frame.left == 100 and frame.top == 20 and frame.res_um_px == pytest.approx(cfg.acquisition.res0_um)


# ------------------------------------------------------------ B8: Block 6 -> Block 11 ---

def test_B8_pilot_row_membership_follows_block6(node):
    """B8: in_S follows Block 6's rule with SELECTION_FLAGS = crossing, stack_edge, bbar_few, profile_nan:
    faint and dark are labels; a registration category other than ON fails (Block 11's category reader)."""
    res, r_allen, _b, _i = node
    assert tuple(phantoms.SELECTION_FLAGS) == ("crossing", "stack_edge", "bbar_few", "profile_nan")
    ok = dataclasses.replace(res, fit_status="converged", flags=("faint", "dark", "iter_cycle"), reg_verdict="")
    assert survey.pilot_row(ok, r_allen, CFG)["in_S"] is True
    for fl in ("crossing", "stack_edge", "bbar_few", "profile_nan"):
        bad = dataclasses.replace(ok, flags=("faint", fl))
        row = survey.pilot_row(bad, r_allen, CFG)
        assert row["in_S"] is False and fl in row["reject"].split(";")
    for verdict, inS in (("ON THE PROCESS (lateral offset +0.00 um)", True), ("ALONGSIDE: 0.8 um", False),
                         ("NOT ON A VISIBLE PROCESS (x)", False)):
        row = survey.pilot_row(dataclasses.replace(ok, reg_verdict=verdict), r_allen, CFG)
        assert row["in_S"] is inS, (verdict, row["reject"])
