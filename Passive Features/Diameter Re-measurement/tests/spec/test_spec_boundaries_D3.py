"""Spec-derived boundary tests between Blocks 12, 13, 14 and their neighbours
3, 5, 6, 7, 8, 11, 15, 17, 21 -- integration tester, run D3, 2026-11-13.
Expected values come from specs/SPEC.md or by hand; the code under test only
produces the outputs that are checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_boundaries_D3.py -q
test_replicate_row_contract renders two phantoms (a minute or more on a shared
machine); the rest takes seconds.
"""
from __future__ import annotations

import dataclasses
import io
import math
import os
import tempfile
from contextlib import redirect_stdout

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import spec_helpers_D3 as H

from allen_image_io import CropFrame
from allen_diameter.analysis import calibration_scene as CS
from allen_diameter.analysis import cell, fit_variants, invert, node_pipeline as NP, phantoms, survey, table
from allen_diameter.analysis import plane_iteration as PI
from allen_diameter.config import default_config
from allen_diameter.loading import table_io

CFG = default_config()


def test_block11_centre_weights_used_by_the_irls():
    # Boundary 11 -> 12: the IRLS's weights are Block 11's centre_weights, exp(-|v - v0| / lambda) (section 2.2).
    v = np.linspace(-3, 3, 53)
    assert_array_equal(fit_variants.centre_weights(v, 0.13, 0.5), np.exp(-np.abs(v - 0.13) / 0.5))


def test_block15_segment_rows_equal_scene_segments():
    # Boundary 13 -> 15: calibration_scene.segment_rows is "plane_iteration.scene_segments' rule, with the reach":
    # at reach 0 the two give the same rows.
    rng = np.random.default_rng(4)
    a = rng.uniform(-2, 8, (40, 3))
    b = a + rng.normal(0, 1.5, (40, 3))
    sc = PI.Scene(np.column_stack([a, b]), rng.uniform(0.1, 1, (40, 2)),
                  np.column_stack([np.arange(2, 42), np.arange(1, 41)]).astype(np.int64), np.empty((0, 3)))
    br = NP.Branch.from_points(np.array([[0, 0, 0], [1, 0, 0.0], [2, 0, 0]]), 0.3, ids=np.array([5, 6, 7]))
    fr = CropFrame(10, 5, 0, 0.1144)
    assert_array_equal(CS.segment_rows(sc, fr, (40, 50), br, 0.0), PI.scene_segments(sc, fr, (40, 50), br))


def test_block8_per_node_columns():
    # Boundary 13/14 -> 8: CSV_COLUMNS ends with ITER_COLUMNS; node_row writes exactly CSV_COLUMNS, mu_d_hat right
    # after fit_status, the tuples joined by ';'.
    assert cell.CSV_COLUMNS[-len(cell.ITER_COLUMNS):] == cell.ITER_COLUMNS
    i = cell.CSV_COLUMNS.index("fit_status")
    assert cell.CSV_COLUMNS[i + 1] == "mu_d_hat"
    res = NP.NodeResult(node_id=3, type=3, x_um=0.0, y_um=0.0, z_um=0.0, path_um=0.0, reg_verdict="", s_star_um=0.0,
                        dz_star_um=0.0, k_star=0, z_sub_um=0.0, cx_um=0.0, cy_um=0.0, cz_um=0.0, theta_rad=0.0,
                        phi_rad=0.1, steep=False, vertical=False, B_bar=200.0, B_bar_region="block_masked",
                        d_hat_um=1.5, mu_hat_per_um=0.4, v0_hat_um=0.0, alpha_hat=0.6, fit_status="converged",
                        flags=("faint",), fit=None, focus_rule="gradient_energy", focus_planes=np.arange(2),
                        focus_score=np.ones(2), focus_depth=np.ones(2), k_star_depth=0, at_bound=("xi_lo", "beta_hi"),
                        pilot_flags=("dark", "faint"), pilot_status="converged")
    inv = invert.Inversion(1.4, 1.4, 1.07, ("mu_d_out_of_domain",))
    row = cell.node_row(res, inv, 1.4, "self", 0.3, 0.099)
    assert list(row) == list(cell.CSV_COLUMNS)
    assert row["at_bound"] == "xi_lo;beta_hi" and row["pilot_flags"] == "dark;faint"
    assert row["flags"] == "faint;mu_d_out_of_domain" and abs(row["mu_d_hat"] - 0.6) < 1e-15


def test_block21_hybrid_changes_the_simulation_hash_only():
    # Boundary 21 -> 14: the hybrid renderer is not in the estimator signature, so a table of either renderer
    # corrects the same fits, while the merge refuses rows rendered under the other one.
    hyb = dataclasses.replace(CFG, renderer=dataclasses.replace(CFG.renderer, absorption="hybrid"))
    assert hyb.signature_hash("estimator") == CFG.signature_hash("estimator")
    assert hyb.signature_hash("simulation") != CFG.signature_hash("simulation")
    rows = [dict(sim_hash=table.row_provenance(hyb))]
    with pytest.raises(ValueError):
        table.check_rows_config(rows, CFG)
    assert table.check_rows_config(rows, hyb) == 0


def test_block7_evaluate_uses_the_nodes_own_tilt_and_mu():
    # Boundary 14 -> 7 (gate 2): end_to_end.evaluate inverts each test phantom at its own (d_hat, phi_hat, mu_hat),
    # Block 14.5. With b = 1 + 0.2 phi + 0.05 ln(mu d) (independent of d) the root is d_hat / b(phi_hat, mu_hat d_hat).
    import end_to_end
    h = CFG.signature_hash("estimator")
    T = H.grid_table((np.linspace(math.log(0.2), math.log(4.0), 41), np.linspace(0, math.pi / 2, 37),
                      np.linspace(math.log(0.06), math.log(12.0), 37)),
                     lambda l1, l2, l3: 1.0 + 0.2 * l2 + 0.05 * l3 + 0 * l1, estimator_hash=h)
    rows = []
    for k, (phi, mphi, mu) in enumerate(((0.1, 0.4, 0.5), (0.2, 0.6, 1.5), (0.3, 0.1, 2.5))):
        d_hat = 1.1 + 0.05 * k
        rows.append(dict(d_target_um=1.0, in_S=True, d_hat_um=d_hat, phi_rad=phi, meas_phi_rad=mphi,
                         mu_hat_per_um=mu))
    want = np.mean([r["d_hat_um"] / (1 + 0.2 * r["meas_phi_rad"] + 0.05 * math.log(r["mu_hat_per_um"] * r["d_hat_um"]))
                    for r in rows])
    with tempfile.TemporaryDirectory() as tmp:
        stem = os.path.join(tmp, "bias_table_%s" % h)
        table_io.save_table(T, stem)
        path = os.path.join(tmp, "test.csv")
        table_io.write_rows(rows, path)
        buf = io.StringIO()
        with redirect_stdout(buf):
            end_to_end.evaluate(CFG, stem, path, 0.10)
    line = [x for x in buf.getvalue().splitlines() if x.startswith("1.00")][0].split()
    assert int(line[3]) == 3 and abs(float(line[4]) - want) < 1e-4, (line, want)


def test_block17_pilot_row_carries_block13_columns():
    # Boundary 13 -> 11/17: survey.pilot_row writes Block 13's columns and the pilot's membership of S.
    res = NP.NodeResult(node_id=3, type=3, x_um=0.0, y_um=0.0, z_um=0.0, path_um=0.0, reg_verdict="", s_star_um=0.0,
                        dz_star_um=0.0, k_star=0, z_sub_um=0.0, cx_um=0.0, cy_um=0.0, cz_um=0.0, theta_rad=0.0,
                        phi_rad=0.1, steep=False, vertical=False, B_bar=200.0, B_bar_region="block_masked",
                        d_hat_um=1.5, mu_hat_per_um=0.4, v0_hat_um=0.0, alpha_hat=0.6, fit_status="at_bound",
                        flags=("at_bound",), fit=None, focus_rule="gradient_energy", focus_planes=np.arange(2),
                        focus_score=np.ones(2), focus_depth=np.ones(2), k_star_depth=0, d_pilot_um=1.3,
                        mu_pilot_per_um=0.5, v0_pilot_um=0.0, alpha_pilot=0.65, B_bar_pilot=201.0,
                        pilot_status="converged", pilot_flags=(), k_iter=2, iter_status="converged")
    row = survey.pilot_row(res, 0.3, CFG)
    for c in cell.ITER_COLUMNS:
        assert c in row, c
    assert row["in_S"] is False and row["in_S_pilot"] is True and row["reject_pilot"] == ""
    assert row["reject"] == "status:at_bound"
    view = cell.pilot_row_view(row)
    assert view["d_hat_um"] == 1.3 and view["in_S"] is True and view["fit_status"] == "converged"


@pytest.fixture(scope="module")
def replicate_rows():
    cfg = dataclasses.replace(CFG, phantom=dataclasses.replace(CFG.phantom, phi_range_deg=(0.0, 10.0),
                                                               mu_range_per_um=(0.6, 1.2)))
    a = phantoms.run_replicate(cfg, 777, 3, 2, d_override_um=0.9)
    b = phantoms.run_replicate(cfg, 777, 3, 2, d_override_um=0.9)
    return cfg, a, b


def test_replicate_row_contract(replicate_rows):
    # Boundary 13 -> 6/14: the replicate row carries the Draw, the deliverable's fit at k_iter and Block 13's
    # columns; ratio = d_hat / d; in_S is S recomputed from the row's own columns (D-098.3 rule); the row passes
    # the table's reader at (ln d, phi, ln(mu_hat d_hat)); the CSV round trip keeps the merge's columns; a repeat
    # under the same seed is identical but for the wall time.
    cfg, row, again = replicate_rows
    draw_keys = [f.name for f in dataclasses.fields(phantoms.Draw)]
    block13 = ["k_iter", "iter_status", "n_rounds", "n_restacks", "w_hat_um", "beta_hat", "xi_hat", "zeta_hat",
               "irls_ok", "at_bound", "d_pilot_um", "alpha_pilot", "pilot_status"]
    for k in draw_keys + block13 + ["d_hat_um", "mu_hat_per_um", "alpha_hat", "fit_status", "meas_phi_rad", "flags",
                                    "in_S", "reject", "ratio", "seconds"]:
        assert k in row, k
    assert row["draw_index"] == 3 and row["realization_index"] == 2 and row["d_um"] == 0.9
    assert row["ratio"] == row["d_hat_um"] / row["d_um"]
    assert row["in_S"] == (phantoms.row_reject_reasons(row) == [])
    assert row["reject"] == ";".join(phantoms.row_reject_reasons(row))
    if row["fit_status"] != "none":
        assert abs(row["alpha_hat"] - row["mu_hat_per_um"] * row["d_hat_um"] / math.cos(row["meas_phi_rad"])) \
            <= 1e-12 * row["alpha_hat"]
    if row["in_S"]:
        R = table.table_rows([row], cfg)
        assert_allclose(R.X, [[math.log(0.9), row["phi_rad"], math.log(row["mu_hat_per_um"] * row["d_hat_um"])]],
                        rtol=1e-15)
    for k in row:
        if k != "seconds":
            x, y = row[k], again[k]
            assert (x == y) or (isinstance(x, float) and math.isnan(x) and math.isnan(y)), k
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "rows.csv")
        table_io.write_rows([dict(row, sim_hash=table.row_provenance(cfg))], p)
        cols = table_io.read_columns(p)
        for k in table_io.FLOAT_COLUMNS:
            x = float(row[k])
            assert cols[k][0] == x or (math.isnan(x) and math.isnan(cols[k][0])), k
        assert cols["in_S"][0] == row["in_S"] and cols["draw_index"][0] == 3
