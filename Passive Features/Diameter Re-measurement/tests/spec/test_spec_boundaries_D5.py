"""Boundary tests between Blocks 16-18 and their neighbours (0, 4, 8, 13, 19)
-- tester run D5, 2026-11-13, from specs/SPEC.md. Producer output is fed to
the consumer as the pipeline does (files written and read back), and the
result is compared with what the spec says the consumer must see.

Run (from the workstream directory):
    MPLBACKEND=Agg python -m pytest tests/spec/test_spec_boundaries_D5.py -q
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import os
import subprocess
import sys

import numpy as np
import pytest

import spec_helpers_D5 as H

import survey_cells as SC
from allen_diameter.analysis import cell as cellmod
from allen_diameter.analysis import cell_alignment as CA
from allen_diameter.analysis import coverage as COV
from allen_diameter.analysis import survey
from allen_diameter.analysis.node_pipeline import NodeResult
from allen_diameter.config import SELECTION_FLAGS, default_config
from allen_diameter.loading import cell_pool as CP
from allen_diameter.loading import survey_io, table_io

NAN = float("nan")
POOL_CANON = ("specimen_id", "species", "layer", "dendrite_type", "reconstruction_type", "area", "donor_id",
              "hemisphere", "normalized_depth")


# ---------------------------------------------------- B1: 16.1 -> 16.2 ----

def _canned_pool_csv(path):
    rows = []
    for k in range(14):
        rows.append(dict(specimen_id=600000 + 37 * k, species="Homo Sapiens" if k != 3 else "Mus musculus",
                         layer=("3", "5", "6a")[k % 3], dendrite_type=("spiny", "aspiny")[k % 2],
                         reconstruction_type="full" if k != 5 else "", area="MTG", donor_id=str(k), hemisphere="left",
                         normalized_depth=0.1 * k))
    CP.write_table(path, rows, POOL_CANON)
    return rows


def _cli(args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, str(H.WS / "scripts" / "cell_pool.py")] + args, cwd=str(H.WS),
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_B1_selection_from_the_cli_reaches_the_runner_in_file_order(tmp_path, monkeypatch):
    canned = str(tmp_path / "canned.csv")
    rows = _canned_pool_csv(canned)
    pool, sel, pick = (str(tmp_path / n) for n in ("pool.csv", "sel.csv", "pick.csv"))
    _cli(["pool", "--from-csv", canned, "--out", pool])
    _cli(["select", "--pool", pool, "--out", sel, "--per-group", "2"])
    human = [r for r in rows if r["species"] == "Homo Sapiens" and r["reconstruction_type"]]
    recs = survey_io.read_selection(sel)
    assert all(isinstance(r["specimen_id"], int) and isinstance(r["rank"], int) for r in recs)
    assert {r["specimen_id"] for r in recs} <= {r["specimen_id"] for r in human}
    for r in recs:
        assert r["group"] == "%s|%s" % (r["neuron_class"], r["layer"])
    ids = [human[4]["specimen_id"], human[0]["specimen_id"]]
    _cli(["pick", "--pool", pool, "--cells", ",".join(str(i) for i in ids), "--out", pick])
    assert [r["specimen_id"] for r in survey_io.read_selection(pick)] == ids
    seen = []
    monkeypatch.setitem(SC.STEPS, "align", lambda ctx, cfg: seen.append(ctx.sid) or {})
    monkeypatch.setitem(SC.SAY, "align", str)
    SC.run(default_config(), sel, str(tmp_path / "survey"), None, ["align"], log=lambda m: None)
    assert seen == [r["specimen_id"] for r in recs]


# ------------------------------------------------ B2: 16.2 -> Block 8 -------

def test_B2_alignment_record_through_json_feeds_to_image_um(tmp_path):
    xyz = np.array([[10.0, 20.0, 3.0], [55.5, 1.25, 9.0]])
    for flip in (None, 3200.0):
        rec = dict(method="auto", shift_x_px=1328.0 + 3, shift_y_px=-816.0 - 2, flip_h_px=flip, z0_um=0.0,
                   score_ratio=7.0, set_aside=False, reasons=[])
        p = str(tmp_path / "a.json")
        survey_io.write_json(p, rec)
        t = CA.full_transform(survey_io.read_json(p))
        got = cellmod.to_image_um(xyz, 0.1144, **t)
        X = xyz[:, 0] / 0.1144 + rec["shift_x_px"]
        Y = (flip - xyz[:, 1] / 0.1144 if flip else xyz[:, 1] / 0.1144) + rec["shift_y_px"]
        np.testing.assert_allclose(got[:, :2], np.c_[X, Y] * 0.1144, rtol=0, atol=1e-9)   # um, float64 round-off


# ------------------------------------- Block 13 -> 17: the per-node CSV ----

CATS = {"": "", "ON THE PROCESS (lateral offset +0.10 um)": "ON",
        "ALONGSIDE: a ridge +0.40 um to the side -- snap needed": "ALONGSIDE",
        "FAR: nearest matching ridge +2.00 um away -- treat as unregistered": "FAR",
        "NOT ON A VISIBLE PROCESS (p 0.4): wrong structure": "NOT_ON"}


def _spec_reasons(status, verdict, flags):
    """Section 2.2's S after D-030, D-053 (ii), D-096: not converged, a registration other than ON, and the flags
    crossing, stack_edge, bbar_few, profile_nan (in that order) fail S; faint, dark, steep, iter_* are labels."""
    out = [] if status == "converged" else ["status:" + status]
    if CATS[verdict] not in ("", "ON"):
        out.append("registration:" + CATS[verdict])
    return out + [f for f in ("crossing", "stack_edge", "bbar_few", "profile_nan") if f in flags]


def _result(i, status, verdict, flags):
    return NodeResult(node_id=i, type=3, x_um=1.0, y_um=2.0, z_um=0.5, path_um=3.0, reg_verdict=verdict,
                      s_star_um=0.0, dz_star_um=0.0, k_star=2, z_sub_um=0.56, cx_um=1.0, cy_um=2.0, cz_um=0.56,
                      theta_rad=0.3, phi_rad=0.05, steep=False, vertical=False, B_bar=205.0,
                      B_bar_region="block_masked", d_hat_um=0.7, mu_hat_per_um=1.1, v0_hat_um=0.01, alpha_hat=0.77,
                      fit_status=status, flags=tuple(flags), fit=None, focus_rule="gradient_energy",
                      focus_planes=np.arange(5), focus_score=np.ones(5), focus_depth=np.ones(5), k_star_depth=2,
                      k_iter=2, iter_status="converged", n_rounds=2, pilot_status="converged",
                      pilot_flags=("faint",), d_pilot_um=0.66, mu_pilot_per_um=1.0, alpha_pilot=0.66, B_bar_pilot=204.0)


def test_B5_pilot_rows_through_the_csv_give_the_spec_selection(tmp_path):
    assert SELECTION_FLAGS == ("crossing", "stack_edge", "bbar_few", "profile_nan")
    cfg = default_config()
    statuses = ("converged", "at_bound", "failed", "none")
    flag_sets = ((), ("faint",), ("crossing",), ("dark", "steep", "iter_max_rounds"), ("bbar_few", "faint"),
                 ("stack_edge", "profile_nan"))
    results, want = [], {}
    for i, (st, v, fl) in enumerate(itertools.product(statuses, CATS, flag_sets), start=1):
        results.append(_result(i, st, v, fl))
        want[i] = _spec_reasons(st, v, fl)
    rows = [survey.pilot_row(r, 0.3, cfg) for r in results]
    p = str(tmp_path / "pilot_1.csv")
    table_io.write_rows(rows, p)
    back = table_io.read_rows(p)
    nodes = COV.node_table([(dict(specimen_id=1, neuron_class="exc", layer="3"), back)])
    assert len(nodes) == len(want) == 120
    for n in nodes:
        w = want[n["node_id"]]
        assert n["s_source"] == "recomputed"
        assert (n["in_S"], n["reject"]) == (not w, ";".join(w)), (n["node_id"], n["reject"], w)
        assert n["in_S_stored"] == (not w)                     # the writer's in_S under the same rule
    # what the S3a CSV holds for the consumers named in Blocks 17-19
    assert {"d_hat_um", "mu_hat_per_um", "phi_rad", "alpha_hat", "allen_radius_um", "fit_status", "flags",
            "reg_verdict", "in_S", "reject", "pilot_status", "pilot_flags", "in_S_pilot", "B_bar"} <= set(back[0])


def test_B5_mu_d_hat_column_named_by_block17():
    # Block 17's notation: "mu_hat_i d_hat_i = mu_d_hat" beside the per-node CSV's d_hat_um, mu_hat_per_um, ...
    row = survey.pilot_row(_result(1, "converged", "", ()), 0.3, default_config())
    assert "mu_d_hat" in row, sorted(row)


# --------------------------------------- 17.3: the phantoms' rows (Block 6) --

def test_phantom_rows_S_read_as_the_real_rows_S(tmp_path):
    # D-098.3: the pooled coverage judges each real row by the current S (faint a label since D-096). A replicate
    # CSV written before D-096 carries in_S False for its faint rows; the phantoms' band of 17.3 is "rows in S"
    rows = []
    for k in range(40):
        faint = k < 20
        rows.append(dict(draw_index=k, realization_index=0, d_um=0.5, phi_rad=0.1, theta_rad=0.0, mu_per_um=0.3,
                         d_hat_um=0.5, mu_hat_per_um=0.35 if faint else 2.0, ratio=1.0, sim_hash="x",
                         fit_status="converged", reg_verdict="", flags="faint" if faint else "",
                         in_S=not faint, reject="faint" if faint else ""))
    p = str(tmp_path / "replicates.csv")
    table_io.write_rows(rows, p)
    ph = SC.read_phantom_rows([p])
    real = COV.node_table([(dict(specimen_id=1), table_io.read_rows(p) and [
        dict(r, node_id=r["draw_index"], allen_radius_um=0.3, alpha_hat=0.1) for r in table_io.read_rows(p)])])
    n_real_in_S = sum(1 for n in real if n["in_S"])
    assert n_real_in_S == 40
    assert ph["d_hat"].size == n_real_in_S, "phantom rows in S: %d, the same rows as real nodes: %d" % (
        ph["d_hat"].size, n_real_in_S)


# ---------------------------------- B6: Block 0's verdicts -> 18.4's jitter ----

def test_B6_pool_inputs_takes_the_ON_nodes_of_the_registration_sentences(tmp_path):
    # 18.4: the jitter over the ON nodes (registration_category of Block 0's registration_check sentence) of every
    # cell with a registration marker; a cell with |median dz*| > 1 flagged z0; d_max from the radius files
    out = str(tmp_path / "survey")
    regs = {11: {"5": ("ON THE PROCESS (lateral offset +0.10 um)", 0.10, 0.2),
                 "6": ("ON THE PROCESS (lateral offset -0.30 um)", -0.30, 0.4),
                 "7": ("ALONGSIDE: a ridge +2.00 um to the side -- snap needed", 2.0, 3.0),
                 "8": ("NOT ON A VISIBLE PROCESS (p 0.4): wrong structure", None, None),
                 "9": ("ON THE PROCESS (lateral offset +0.05 um)", 0.05, None)},
            12: {"3": ("ON THE PROCESS (lateral offset +0.20 um)", 0.20, 1.6),
                 "4": ("FAR: nearest matching ridge +4.00 um away -- treat as unregistered", 4.0, -2.0),
                 "6": ("ON THE PROCESS (lateral offset +0.00 um)", 0.0, 1.4)}}
    for sid, res in regs.items():
        cell = survey_io.CellDir(out, sid)
        survey_io.write_json(cell.path_of("registration", "registration_%d.json" % sid),
                             {k: dict(verdict=v, lateral_offset_um=s, z_offset_um=z) for k, (v, s, z) in res.items()})
        survey_io.write_marker(cell, "registration", {}, 0.0, "x")
        table_io.write_rows([dict(node_id=n, type=3, diameter_um=d) for n, d in ((1, 0.5), (2, 2.0 + sid / 100.0))],
                            cell.path_of("radius", "radius_nodes_%d.csv" % sid))
        survey_io.write_marker(cell, "radius", {}, 0.0, "x")
    cfg = default_config()
    sug = SC.pool_inputs(cfg, out, log=lambda m: None)
    s_on = [0.10, -0.30, 0.05, 0.20, 0.0]
    z_on = [0.2, 0.4, 1.6, 1.4]
    assert abs(sug["jitter"]["jitter_xy_um"] - H.ref_robust_sd(s_on)) <= 1e-12
    assert abs(sug["jitter"]["jitter_z_um"] - H.ref_robust_sd(z_on)) <= 1e-12
    per = {r["specimen_id"]: r for r in sug["jitter"]["per_cell"]}
    assert per[12]["z0_flag"] and not per[11]["z0_flag"]                      # medians 1.5 and 0.3 um
    assert sug["d_max"]["value_um"] == 2.12 and sug["d_max"]["specimen_id"] == 12 and sug["d_max"]["node_id"] == 2
    js = survey_io.read_json(os.path.join(out, "s3b", "inputs.json"))
    assert js["suggested"]["jitter_xy_um"] == sug["jitter"]["jitter_xy_um"]
    assert js["suggested"]["background_B_gl"] is None and "camera" in js["camera"].get("note", "")


# --------------------------------- B7: 18.3's camera step -> Block 4's chain ----

def test_B7_camera_noise_suggestion_reproduces_the_target_through_block4(tmp_path):
    # 18.3: noise_sd_gl "matched through the camera chain to the median clipped background SD"; B the median of the
    # nodes' background medians. Check: Block 4's camera_chain on flat fields at the suggestion gives that SD back.
    import json as _json
    import camera_calibration
    from allen_diameter.analysis import camera_fit
    from allen_diameter.model import camera
    cfg = default_config()
    cache = tmp_path / "cache"
    cache.mkdir()
    ps = tmp_path / "pilot_summary_5.json"
    target = 3.0
    ps.write_text(_json.dumps({"background": {"B_bar_gl": {"p50": 210.0, "n": 5}, "clipped_sd_gl": {"p50": target, "n": 5}}}))
    out = camera_calibration.run(str(cache), str(tmp_path / "cam"), 5, pilot_summary=str(ps), cfg=cfg)
    r = out["renderer"]
    assert r["background_B_gl"] == 210.0
    rc = dataclasses.replace(cfg.renderer, background_B_gl=float(r["background_B_gl"]), noise_sd_gl=float(r["noise_sd_gl"]))
    sds = np.array([camera_fit.clipped_sd(camera.camera_chain(np.ones((128, 128)), rc, np.random.default_rng(1000 + s))
                                          .astype(float).ravel()) for s in range(20)])
    se = sds.std(ddof=1) / math.sqrt(sds.size)
    # tolerance: 4 standard errors of the 20-field mean (about 0.04 gl here) plus 1 % for the match's own grid
    assert abs(sds.mean() - target) <= 4 * se + 0.01 * target, (sds.mean(), se, r["noise_sd_gl"])
