"""Spec-derived boundary tests between run D1's blocks (0, 1, 4-9, 21) and
their neighbours (2, 13, 14, 17, 18) -- integration tester, 2026-11-13. Each
test checks that what a producer returns is what its consumer expects
(shape, axis order, dtype, units, index base, pixel-centre convention, plane
depth, missing-value encoding). Expected values come from the spec, numpy /
scipy or by hand; the code under test only produces the outputs checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_boundaries_D1.py -q
The synthetic-cell test renders a six-node cell (about a minute on a free core).
"""
from __future__ import annotations

import dataclasses
import math
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import spec_helpers_D1 as H

import allen_image_io as aio
from allen_diameter import config as C
from allen_diameter.analysis import cell as CELL
from allen_diameter.analysis import invert as INV
from allen_diameter.analysis import phantoms as PH
from allen_diameter.analysis import profiles as PR
from allen_diameter.analysis import s3_inputs
from allen_diameter.analysis import table as T
from allen_diameter.loading import swc_io
from allen_diameter.loading import table_io as TIO
from allen_diameter.model import geometry as G
from allen_diameter.model import hybrid as HY
from allen_diameter.model import ray_world as RW
from allen_diameter.model import render as R
from allen_diameter.model.geometry import Tube

CFG = C.default_config()
PX = CFG.acquisition.res0_um
DZ = CFG.acquisition.dz_um
NAN = float("nan")


# ------------------------------------------- Block 4 -> Block 5 (pixels) --

def test_synthetic_block_pixels_meet_block5_sampling():
    """A tube rendered by Block 4 at a sub-pixel position is centred at v = 0
    in the profile Block 5 samples through its true node point: integer
    full-resolution coordinates are pixel centres on both sides (a half-pixel
    slip would show as a 0.057 um offset)."""
    lin = dataclasses.replace(CFG, renderer=dataclasses.replace(
        CFG.renderer, absorption="linear", noise_sd_gl=0.0, jpeg=False))
    left, top, w, h = 400, 300, 60, 60
    theta = 0.3
    c = ((left + 30.37) * PX, (top + 29.81) * PX, 0.0)
    tube = Tube(c, 0.3, 0.0, theta, 1.0, 10.0, "axial")
    block, ks, valid, frame = R.synthetic_block(tube, 1.0, np.array([0]), left, top, w, h, lin,
                                                np.random.default_rng(0), 4)
    y_hat = np.array([-math.sin(theta), math.cos(theta)])
    e_u = np.array([math.cos(theta), math.sin(theta)])
    v = np.arange(-30, 31) * 0.02
    I = PR.sample_profile(block[0].astype(float), frame, np.array(c[:2]), y_hat, e_u, v, 0, 0.0)
    dip = np.nanmax(I) - I
    core = np.abs(v) <= 1.0
    centroid = float(np.sum(v[core] * dip[core]) / np.sum(dip[core]))
    assert abs(centroid) < 0.01, centroid       # 8-bit rounding moves it by ~2e-3 um


def test_synthetic_block_and_fetch_zblock_share_one_contract():
    tube = Tube((10.0, 10.0, 0.0), 0.3, 0.0, 0.0, 1.0, 10.0, "axial")
    sb = R.synthetic_block(tube, 1.0, np.array([-1, 0, 1]), 80, 80, 10, 8, CFG, np.random.default_rng(1), 2)
    img = np.full((200, 200), 120, np.uint8)
    planes = __import__("pandas").DataFrame({"plane_index": [-1, 0, 1], "id": [0, 1, 2]})

    class F:
        def get(self, image_id, left, top, width, height, downsample=0):
            return img[top:top + height, left:left + width].copy()

    fz = aio.fetch_zblock(F(), planes, -1, 1, 80, 80, 10, 8, PX)
    for a, b in zip(sb, fz):
        assert type(a) is type(b) or (isinstance(a, np.ndarray) and isinstance(b, np.ndarray))
    assert sb[0].dtype == fz[0].dtype == np.uint8 and sb[0].shape == fz[0].shape == (3, 8, 10)
    assert_array_equal(np.asarray(sb[1]), np.asarray(fz[1]))
    assert np.asarray(sb[2]).dtype == np.asarray(fz[2]).dtype == bool
    assert sb[3] == fz[3]                                            # CropFrame(left, top, 0, p_x)


# ---------------------------------------- Block 2 -> Block 21 (chords) -----

@pytest.mark.parametrize("phi_deg,aspect,cut", [(30.0, 0.7, "axial"), (60.0, 1.3, "vertical"), (0.0, 0.7, "vertical")])
def test_back_lengths_follow_block2_line_interval(phi_deg, aspect, cut):
    """Block 21 (b): back_lengths equals Block 2's line_interval,
    max(0, min(t2, 0) - t1) (light +1) / max(0, t2 - max(t1, 0)) (-1),
    also on squashed sections and the vertical cut (not covered by the
    round-tube reference)."""
    tube = Tube((0.1, 0.0, 0.2), 0.45, math.radians(phi_deg), 0.4, aspect, 1.2, cut)
    S, W = RW.directions(8, 16, 1.4, 1.515)
    P = np.array([0.1, 0.0, 0.2]) + np.random.default_rng(3).uniform(-0.6, 0.6, size=(300, 3)) * [1, 0.6, 0.4]
    for light in (+1, -1):
        got = HY.back_lengths(P, tube, S, light)
        t1, t2, hit = G.line_interval(P[:, None, :], S[None, :, :], tube)
        exp = np.maximum(0.0, np.minimum(t2, 0.0) - t1) if light > 0 else np.maximum(0.0, t2 - np.maximum(t1, 0.0))
        exp = np.where(hit, exp, 0.0)
        assert np.max(np.abs(got - exp)) < 1e-9


# ------------------------------------------- Block 9 -> Block 21 ----------

@pytest.mark.parametrize("n_rho,n_psi", [(4, 8), (16, 32), (5, 7)])
def test_hybrid_accepts_block9_directions(n_rho, n_psi):
    S, W = RW.directions(n_rho, n_psi, 1.4, 1.515)
    tube = Tube((0.0, 0.0, 0.0), 0.4, 0.2, 0.0, 1.0, 2.0, "axial")
    h = HY.ray_history(np.zeros((2, 3)), tube, 0.0, S, W, +1)
    assert_allclose(h, RW.mean_inverse_cos(S, W), rtol=1e-14)


# --------------------------------- Block 1 -> Block 4 -> Block 9 dispatch ---

def test_ray_world_dispatch_through_block4():
    rc = dataclasses.replace(CFG.renderer, absorption="ray_world", ray_n_rho=4, ray_n_psi=8)
    tube = Tube((0.0, 0.0, 0.0), 0.3, 0.1, 0.2, 1.0, 1.0, "axial")
    g = R.FineGrid(-0.6, -0.6, 0.04, 30, 30)
    res = R.render_transmittance(tube, 1.0, [0.0, 0.28], g, rc)
    assert res.backend == "ray_world" and res.tau.shape == (2, 30, 30)
    ref = np.asarray(RW.render_planes(tube, 1.0, [0.0, 0.28], g, rc))
    assert_array_equal(res.tau, ref)


# ---------------------- Block 5/13 -> Block 8 CSV -> Block 6 rule -> Block 17

def test_row_reject_reasons_through_the_csv_round_trip(tmp_path):
    """D-098.3 / Block 17: through write_rows / read_rows the stored row gives
    the NodeResult's own reasons, NaN and empty cells meaning nothing."""
    import test_spec_block6_phantoms as B6
    verdicts = ["", "ON THE PROCESS (lateral offset +0.10 um)",
                "ALONGSIDE: a ridge +1.00 um to the side -- snap needed before measuring",
                "NOT ON A VISIBLE PROCESS (p = 0.3): wrong structure"]
    statuses = ["converged", "at_bound", "failed"]
    flagsets = [(), ("faint",), ("crossing", "dark"), ("stack_edge", "bbar_few", "profile_nan", "iter_cycle")]
    rows, results = [], []
    inv = INV.Inversion(NAN, NAN, NAN, ())
    for v in verdicts:
        for s in statuses:
            for f in flagsets:
                res = B6.make_result(reg_verdict=v, fit_status=s, flags=f)
                results.append(res)
                rows.append(CELL.node_row(res, inv, NAN, "none", 0.3, 0.099))
    p = str(tmp_path / "nodes.csv")
    TIO.write_rows(rows, p)
    back = TIO.read_rows([p])
    assert len(back) == len(results)
    for res, row in zip(results, back):
        assert PH.row_reject_reasons(row) == PH.reject_reasons(res), (res.reg_verdict, res.fit_status, res.flags)


# ------------------------------------- Block 6 rows -> Block 14 / Block 17 ---

def test_replicate_rows_feed_the_table_and_the_coverage(tmp_path):
    rows = []
    for g in range(1, 4):
        for r in (1, 2):
            d, dh, muh = 0.5 * g, 0.55 * g, 1.1 / g
            rows.append(dict(draw_index=g, realization_index=r, seed=1, d_um=d, phi_rad=0.1 * g,
                             theta_rad=0.2, mu_per_um=1.0, meas_phi_rad=0.1 * g + 0.01, d_hat_um=dh,
                             mu_hat_per_um=muh, ratio=dh / d, fit_status="converged", flags="",
                             in_S=(g != 2), reject="" if g != 2 else "crossing", sim_hash="sim-abc"))
    p = str(tmp_path / "rows_x_1_4.csv")
    TIO.write_rows(rows, p)
    cols = TIO.read_columns([p])
    assert cols["in_S"].dtype == bool and cols["draw_index"].dtype.kind == "i"
    assert_allclose(cols["d_hat_um"], [r["d_hat_um"] for r in rows], rtol=0, atol=0)
    tr = T.table_rows(cols, CFG)
    keep = np.array([r["in_S"] for r in rows])
    exp_X = np.column_stack([np.log([r["d_um"] for r in rows]), [r["phi_rad"] for r in rows],
                             np.log([r["mu_hat_per_um"] * r["d_hat_um"] for r in rows])])
    assert_allclose(tr.X, exp_X[keep], rtol=1e-15, atol=0)
    assert_allclose(tr.y, np.array([r["ratio"] for r in rows])[keep], rtol=1e-15)
    assert_array_equal(tr.draw, np.array([r["draw_index"] for r in rows])[keep])
    # the failure count at the TRUE third coordinate ln(mu_g d_g) (D-062.5 with D-063.1)
    exp_F = np.column_stack([np.log([r["d_um"] for r in rows]), [r["phi_rad"] for r in rows],
                             np.log([r["mu_per_um"] * r["d_um"] for r in rows])])
    assert_allclose(tr.X_fail, exp_F, rtol=1e-15, atol=0)
    assert_array_equal(tr.failed, ~keep)


# --------------------------------------------- Block 8 -> Block 18 --------

def test_radius_collect_feeds_d_max(tmp_path):
    import allen_radius_distribution as ARD
    import test_spec_block8_cell as B8
    p = str(tmp_path / "c.swc")
    H.write_swc_text(p, B8.ROWS)
    nodes, _summary = ARD.collect([p], (3, 4))
    cells = [{"specimen_id": 42, "node_ids": [n["node_id"] for n in nodes],
              "D_allen": [n["diameter_um"] for n in nodes]}]
    out = s3_inputs.d_max(cells)
    dend = [r for r in B8.ROWS if r[1] in (3, 4)]
    big = max(dend, key=lambda r: float(r[5]))
    assert out["value_um"] == 2 * float(big[5]) and out["node_id"] == big[0] and out["specimen_id"] == 42


# -------------------- Blocks 5/13 -> 7 -> 8 on a synthetic cell (end to end) --

def test_synthetic_cell_measure_correct_fill_write(tmp_path):
    """Block 8's chain on the fixture's six-node cell served through
    fetch_zblock: the rows carry the CSV columns, mu_d_hat = mu_hat d_hat,
    the flags follow Block 5's thresholds, the new radius is d_final/2 on
    measured dendrite nodes and Allen's elsewhere, the area ratio is
    area_ratio(swc, radius_new), and the written SWC changes radius tokens only."""
    import fixtures_cell
    import test_spec_block7_invert as B7
    cfg = CFG
    tcfg = B7.small_cfg()
    table = T.fit_table(B7.make_rows(tcfg, B7.b_lin), tcfg)
    swc, fetcher, planes, path = fixtures_cell.synthetic_cell(str(tmp_path), cfg)

    def provider(left, top, width, height, k_lo, k_hi):
        return aio.fetch_zblock(fetcher, planes, k_lo, k_hi, left, top, width, height, PX)

    rows, radius_new, ratio = CELL.measure_cell(swc, provider, table, cfg)
    assert len(rows) == int(np.sum(np.isin(swc.types, cfg.acquisition.dendrite_swc_types)))
    m = cfg.measure
    for row in rows:
        assert set(CELL.CSV_COLUMNS) <= set(row)
        if math.isfinite(row["mu_hat_per_um"]) and math.isfinite(row["d_hat_um"]):
            assert abs(row["mu_d_hat"] - row["mu_hat_per_um"] * row["d_hat_um"]) < 1e-12
        fl = set(filter(None, str(row["flags"]).split(";")))
        phi = row["phi_rad"]
        if math.isfinite(phi):
            assert ("steep" in fl) == (math.tan(phi) >= m.steep_tan_diagnostic)
            assert ("vertical" in fl) == (phi > math.radians(m.phi_vertical_deg))
        if math.isfinite(row["alpha_hat"]):
            assert ("dark" in fl) == (row["alpha_hat"] > m.alpha_dark_flag)
        assert row["filled_from"] in ("self", "neighbours", "allen", "none")
        assert math.isfinite(row["d_final_um"])
    idx = {int(i): k for k, i in enumerate(swc.ids)}
    measured = {int(r["node_id"]): r for r in rows}
    for nid, k in idx.items():
        if nid in measured:
            assert abs(radius_new[k] - measured[nid]["d_final_um"] / 2) < 1e-12
        else:
            assert radius_new[k] == swc.radius[k]                     # Allen's where nothing was measured
    assert abs(ratio - CELL.area_ratio(swc, radius_new)) < 1e-12
    out = str(tmp_path / "corrected.swc")
    swc_io.write_swc(swc, out, radius=radius_new)
    a, b = open(path).read().split("\n"), open(out).read().split("\n")
    assert len(a) == len(b)
    for la, lb in zip(a, b):
        if la.startswith("#") or not la.strip():
            assert la == lb
        else:
            ta, tb = la.split(), lb.split()
            assert ta[:5] == tb[:5] and ta[6] == tb[6]
