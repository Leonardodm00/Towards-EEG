"""Smoke test for Cell 4k, the planes the Allen-guided iteration chose with the
line that chose them and the diameter fitted there -- Block 11 in
specs/SPEC.md (D-045, the user's request of 2026-10-12, 17:58).

For node j of a Cell 4j record with k_iter: the selection round i* is the last
round that picked a plane (its pick is k_iter), h_sel its half-length; the
line L_j = {o_j + v y_hat : |v| <= h_sel} through the SWC node o_j across the
pilot's heading theta_j, y_hat = (-sin theta_j, cos theta_j),
e_u = (cos theta_j, sin theta_j). Block 5's fit redone at k_iter gives d and
v0 (along y_hat from the pilot's fitted centre c_j); with s = (c_j - o_j) . y_hat
the caliper's ends are E-+ = o_j + (s + v0 -+ d / 2) y_hat; u = (c_j - o_j) . e_u.

Checks
    test_known_answer   selection_round on hand-built trajectories (converged,
                        cycle, round limit, a later round without a plane, no
                        k_iter, a k_iter the rounds do not hold); s and u for a
                        fitted centre moved by a y_hat + b e_u (to 1e-12); on
                        the synthetic tubes of 0.8 and 3.0 um, whose axis
                        passes through the SWC nodes, the caliper's middle
                        within 0.05 um of the node and its length the fitted d
    test_reference      the fit redone at k_iter equals Cell 4j's d_iter to the
                        last bit, and refit_at_plane's own result; its profile
                        equals profiles.sample_profile called by hand on the
                        block's plane through c_j; the model equals
                        tube_model.model_profile called by hand; the line's
                        profile equals line_profiles at h_sel in plane k_iter;
                        at k_iter = k* the fit is the pilot's
    test_convergence    SKIP: nothing iterates or is discretised here
    test_invariants     the caliper has length d (1e-12) and lies along y_hat,
                        symmetric about o + (s + v0) y_hat; a fitted centre
                        moved across the branch by 0.2 um or along it by 0.3 um
                        leaves the caliper on the tube (its middle to 0.01 um,
                        d to 0.01 um) and u reports the move; a gain of 0.75 on
                        every plane leaves d and v0 (1e-6 um)
    test_contract       read_records refuses a non-list and a record of another
                        evaluation; node_square refuses planes that are not
                        k_SWC +- n and notes a square of another half-width;
                        fit_overlay refuses a record without k_iter and a
                        k_iter outside the square, and notes a d that differs
                        from the record; a fit without a background (a dark
                        plane) gives no caliper and no model; run() lists the
                        nodes skipped by Cell 4j or without k_iter with their
                        reasons and draws them nowhere
    test_determinism    two runs: the same records and arrays
    test_edge_cases     run() end to end on the 0.8 um cell from Cell 4j's
                        JSON: one figure per node, the overview and the JSON
                        (keys, values, the soma skipped), every text inside
                        both figures, a subset of nodes, an overview of one
                        node; on the 3.0 um tube the square rebuilt equals
                        Cell 4j's after its restacks, and off k* the model
                        uses plane k_iter's own B_bar; the CLI passes its
                        options to run (server stubbed, run replaced) and
                        refuses --ncol 0

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_iteration_fits.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib.metadata
import inspect
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
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for p in (SRC, HERE, WS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from allen_diameter.config import default_config  # noqa: E402

SEED = 20261012
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
DELTA = 0.1144
NAN = float("nan")


def _cell(tmp, d_true=0.8):
    """(cfg, swc, provider, swc path) of fixtures_cell's synthetic cell: a tube of d_true um whose axis passes
    through the SWC nodes 2..7, Allen radius 0.3 um, planes -7..7; block_half_um 3.5 as in the iteration test."""
    import run_cell
    from fixtures_cell import synthetic_cell
    base = default_config()
    cfg = dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5))
    swc, fetcher, planes, swc_path = synthetic_cell(tmp, cfg, d_true=d_true, mu=1.0)
    return cfg, swc, run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um), swc_path


def _cell4j(tmp, d_true=0.8, nodes=(3, 4, 1)):
    """The cell and Cell 4j's records (the iterate evaluation at its default m = 3), read back from its JSON."""
    import iteration_fits as IF
    import plane_differences as PD
    cfg, swc, prov, swc_path = _cell(tmp, d_true)
    out = os.path.join(tmp, "iter")
    PD.run(swc, prov, cfg, list(nodes), out, "999", planes_half=6, evaluation="iterate", log=lambda m: None)
    path = os.path.join(out, "planeiter_999.json")
    return cfg, swc, prov, swc_path, IF.read_records(path), path


def _overlay(cfg, swc, prov, rec):
    import iteration_fits as IF
    st, notes = IF.node_square(swc, prov, cfg, rec)
    return st, notes, IF.fit_overlay(st, cfg, rec)


def _moved_centre(st, a, b):
    """st with the pilot's fitted centre moved by a y_hat + b e_u (um)."""
    th = st["theta"]
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    r = st["result"]
    c = np.array([r.cx_um, r.cy_um]) + a * y_hat + b * e_u
    return dict(st, result=dataclasses.replace(r, cx_um=float(c[0]), cy_um=float(c[1])))


def _texts_inside(fig):
    """Every figure-level text, legend entry and axes title lies inside the figure, to the pixel."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    texts = list(fig.texts) + [ax.title for ax in fig.axes]
    texts += [t for leg in fig.legends for t in leg.get_texts()]
    for ax in fig.axes:
        texts += list(ax.texts)
    for t in texts:
        if t.get_text():
            bb = t.get_window_extent(renderer=ren)
            assert bb.x0 >= -1.0 and bb.x1 <= fig.bbox.width + 1.0 and bb.y0 >= -1.0 and \
                bb.y1 <= fig.bbox.height + 1.0, (t.get_text()[:60], bb.x0, bb.x1, bb.y0, bb.y1)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    import iteration_fits as IF
    R = lambda i, k, h: dict(round=i, k=k, half_um=h)  # noqa: E731
    cases = [  # (rounds, k_iter, the selection round's index)
        ([R(1, 3, 0.9), R(2, 4, 1.2), R(3, 4, 1.5)], 4, 2),             # converged: the repeat round
        ([R(1, 3, 0.9), R(2, 4, 1.2), R(3, 3, 1.5)], 3, 2),             # cycle: the round that picked 3 again
        ([R(i, i, 0.1 * i) for i in range(1, 6)], 5, 4),                # the round limit
        ([R(1, 3, 0.9), R(2, 4, 1.2), R(3, None, 6.0)], 4, 1),          # a later round without a plane
    ]
    for rounds, k, want in cases:
        got = IF.selection_round(dict(node_id=9, k_iter=k, rounds=rounds))
        assert got is rounds[want], (rounds, got)
    assert IF.selection_round(dict(node_id=9, k_iter=None, rounds=[R(1, None, 0.03)])) is None
    for rounds in ([R(1, 3, 0.9)], [], [R(1, 4, 0.9), R(2, None, 2.0)]):
        try:
            IF.selection_round(dict(node_id=9, k_iter=5, rounds=rounds))
        except ValueError:
            continue
        raise AssertionError("selection_round accepted k_iter 5 for %r" % rounds)
    for d_true, tol_d in ((0.8, 0.1), (3.0, 0.6)):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, swc, prov, _p, recs, _j = _cell4j(tmp, d_true, nodes=(4,))
            st, notes, ov = _overlay(cfg, swc, prov, recs[0])
            # s and u: the fitted centre moved by a y_hat + b e_u gives s = s0 + a, u = u0 + b (dot products)
            ov2 = IF.fit_overlay(_moved_centre(st, 0.25, -0.4), cfg, recs[0])
            assert abs(ov2["s_um"] - (ov["s_um"] + 0.25)) <= 1e-12 and abs(ov2["u_um"] - (ov["u_um"] - 0.4)) <= 1e-12
            # the tube's axis passes through the SWC node: the caliper's middle within 0.05 um of it (under half a
            # pixel; the fitted centres of the pilot are 0.01 um off), its length the fitted d, d near the truth
            # (the fit's own error: 0.8 um tubes fit 0.84-0.86 um, 3.0 um within 20 %, the iteration test's bounds)
            mid = 0.5 * (ov["edges_v"][0] + ov["edges_v"][1])
            assert abs(mid) <= 0.05, mid
            assert abs((ov["edges_v"][1] - ov["edges_v"][0]) - ov["d"]) <= 1e-12
            assert abs(ov["d"] - d_true) <= tol_d, (d_true, ov["d"])
            assert ov["k"] == recs[0]["k_iter"] and not notes and not ov["notes"], (notes, ov["notes"])


def test_reference():
    import iteration_fits as IF
    import plane_differences as PD
    from allen_diameter.analysis import profiles as PR
    from allen_diameter.model import tube_model
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, _p, recs, _j = _cell4j(tmp, 0.8, nodes=(3, 4))
        for rec in recs:
            st, notes, ov = _overlay(cfg, swc, prov, rec)
            k, m = rec["k_iter"], cfg.measure
            # the fit redone equals Cell 4j's d_iter to the last bit, and refit_at_plane without the profile
            assert ov["d"] == rec["d_iter_um"], (ov["d"], rec["d_iter_um"])
            plain = PD.refit_at_plane(st, cfg, k)
            assert all(plain[key] == ov["fit"][key] for key in plain), (plain, ov["fit"])
            # its profile: sample_profile by hand on the block's plane k through the pilot's centre
            r = st["result"]
            th = float(r.theta_rad)
            y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
            v = PR.profile_offsets(m)
            I = PR.sample_profile(st["block"][k - int(st["ks"][0])], st["block_frame"],
                                  np.array([r.cx_um, r.cy_um]), y_hat, e_u, v, PR.n_along(m), m.profile_step_um)
            assert np.array_equal(ov["fit"]["v_um"], v) and np.array_equal(ov["fit"]["profile"], I)
            # the model: tube_model by hand with the fit's parameters
            f = ov["fit"]
            ref = tube_model.model_profile(v, f["d_hat_um"], f["alpha_hat"], f["v0_hat_um"], m.sigma_fit_um,
                                           f["B_bar"], f["n_nodes"])
            assert np.array_equal(ov["model"], ref)
            # the line that chose the plane: line_profiles at the selection round's half-length, plane k_iter
            h = IF.selection_round(rec)["half_um"]
            _h, vv, _y, _e, prof = PD.line_profiles(st, cfg, h)
            assert ov["h_sel"] == h and np.array_equal(ov["v_sel"], vv) and \
                np.array_equal(ov["prof_sel"], prof[k - int(st["ks"][0])], equal_nan=True)
            assert np.allclose(ov["line_xy"], [st["o"] - h * y_hat, st["o"] + h * y_hat], rtol=0, atol=1e-12)
            # the selection round's half-length, whatever the longest line (here shortened by hand to 0.7 um)
            rounds = [dict(x) for x in rec["rounds"]]
            rounds[-1]["half_um"] = 0.7
            ov7 = IF.fit_overlay(st, cfg, dict(rec, rounds=rounds))
            assert ov7["h_sel"] == 0.7 and abs(np.linalg.norm(ov7["line_xy"][1] - ov7["line_xy"][0]) - 1.4) <= 1e-12
            # k_iter = k* on this tube: the fit is the pilot's
            assert k == int(r.k_star) and ov["d"] == r.d_hat_um and ov["v0"] == r.v0_hat_um


def test_convergence():
    raise unittest.SkipTest("nothing iterates or is discretised here: a figure of fits made by Block 5")


def test_invariants():
    import iteration_fits as IF
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, _p, recs, _j = _cell4j(tmp, 0.8, nodes=(4,))
        rec = recs[0]
        st, _n, ov = _overlay(cfg, swc, prov, rec)
        E, y_hat = np.asarray(ov["edges_xy"]), ov["y_hat"]
        axis = ov["o"] + (ov["s_um"] + ov["v0"]) * y_hat
        assert abs(np.linalg.norm(E[1] - E[0]) - ov["d"]) <= 1e-12
        assert abs(abs(np.dot((E[1] - E[0]) / ov["d"], y_hat)) - 1.0) <= 1e-12            # along y_hat
        assert np.allclose(0.5 * (E[0] + E[1]), axis, rtol=0, atol=1e-12)
        # the caliper is the tube's, not the sampling origin's: moved across by 0.2 um or along by 0.3 um, the fit
        # follows the tube (its middle to 0.01 um, d to 0.01 um: the fit on a profile sampled elsewhere)
        for a, b in ((0.2, 0.0), (-0.2, 0.0), (0.0, 0.3)):
            ovm = IF.fit_overlay(_moved_centre(st, a, b), cfg, rec)
            mid0 = 0.5 * (ov["edges_v"][0] + ov["edges_v"][1])
            mid1 = 0.5 * (ovm["edges_v"][0] + ovm["edges_v"][1])
            assert abs(mid1 - mid0) <= 0.01 and abs(ovm["d"] - ov["d"]) <= 0.01, (a, b, mid0, mid1, ovm["d"])
            assert abs(ovm["u_um"] - (ov["u_um"] + b)) <= 1e-12
        # a gain on every plane: G and the fit scale out (d, v0 to 1e-6 um, the fit's termination tolerance)
        stc = dict(st, stack=0.75 * st["stack"], block=0.75 * st["block"])
        ovc = IF.fit_overlay(stc, cfg, rec)
        assert abs(ovc["d"] - ov["d"]) <= 1e-6 and abs(ovc["v0"] - ov["v0"]) <= 1e-6, (ovc["d"], ov["d"])


def test_contract():
    import matplotlib.pyplot as plt
    import iteration_fits as IF
    from allen_diameter.plotting import figures as FG
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, _p, recs, path = _cell4j(tmp, 0.8, nodes=(4, 1))
        rec = recs[0]
        bad = os.path.join(tmp, "bad.json")
        for content in ({"node_id": 4}, [dict(rec, evaluation="gradient")], [{"k_iter": 0}]):
            with open(bad, "w") as f:
                json.dump(content, f)
            try:
                IF.read_records(bad)
            except ValueError:
                continue
            raise AssertionError("read_records accepted %r" % (content,))
        try:
            IF.node_square(swc, prov, cfg, dict(rec, ks=[k for k in rec["ks"] if k != 2]))
        except ValueError:
            pass
        else:
            raise AssertionError("node_square accepted planes with a gap")
        st, notes = IF.node_square(swc, prov, cfg, rec, half_um=3.9)       # Cell 4j's square was +-3.5 um
        assert len(notes) == 1 and notes[0].startswith("square +-3.900 um"), notes
        # the longest line sizes the square (+ 2 pixels), not the selection round's line
        st5, notes5 = IF.node_square(swc, prov, cfg, dict(rec, profile_half_um=4.5, half_um=4.5 + 2 * DELTA))
        assert st5["square_half_um"] == 4.5 + 2 * DELTA and not notes5, (st5["square_half_um"], notes5)
        # with no line, the measurement's square raised for the pilot's d-line and the Allen line at the record's m:
        # m = 8 makes the d-line (8 d_hat / 2 + 2 pixels, about 3.6 um) the largest term
        d_hat = float(st5["result"].d_hat_um)
        want = 0.5 * 8.0 * d_hat + 2 * DELTA
        st8, notes8 = IF.node_square(swc, prov, cfg, dict(rec, profile_half_um=None, line_mult=8.0, half_um=want))
        assert want > max(3.5, 8.0 * 0.3 + 2 * DELTA) and st8["square_half_um"] == want and not notes8, \
            (want, st8["square_half_um"], notes8)
        st, notes = IF.node_square(swc, prov, cfg, rec)
        for r2 in (dict(rec, k_iter=None), dict(rec, k_iter=9, rounds=rec["rounds"] + [dict(round=9, k=9,
                                                                                            half_um=1.0)])):
            try:
                IF.fit_overlay(st, cfg, r2)
            except ValueError:
                continue
            raise AssertionError("fit_overlay accepted k_iter %r" % r2["k_iter"])
        ov = IF.fit_overlay(st, cfg, dict(rec, d_iter_um=rec["d_iter_um"] + 0.01))
        assert len(ov["notes"]) == 1 and ov["notes"][0].startswith("the fit redone here gives d"), ov["notes"]
        # a dark plane: no background, no fit; the line is drawn, without caliper and model
        dark = np.array(st["block"])
        dark[rec["k_iter"] - int(st["ks"][0])] = 0.0
        ovd = IF.fit_overlay(dict(st, block=dark), cfg, rec)
        assert ovd["fit"]["fit_status"] == "bbar_nonpositive" and ovd["edges_xy"] is None and ovd["model"] is None
        assert math.isnan(ovd["d"]) and len(ovd["notes"]) == 1
        fig = FG.iteration_fit_figure(dict(label="x", sublabel="y", plane=ovd["plane"], extent=ovd["extent"],
                                           o=ovd["o"], e_u=ovd["e_u"], line_xy=ovd["line_xy"], edges_xy=None,
                                           edges_v=None, d=NAN, v_sel=ovd["v_sel"], prof_sel=ovd["prof_sel"],
                                           s_um=ovd["s_um"], v_fit=ovd["fit"]["v_um"] + ovd["s_um"],
                                           prof_fit=ovd["fit"]["profile"], model=None, B_bar=0.0,
                                           fit_half_um=3.0), "t")
        plt.close(fig)
        # run(): the soma (skipped by Cell 4j) and a node without k_iter are listed with their reasons, drawn nowhere
        logs = []
        out = IF.run(swc, prov, cfg, recs + [dict(rec, node_id=99, k_iter=None, iter_status="line_short")],
                     os.path.join(tmp, "fits"), "999", log=logs.append)
        by = {x["node_id"]: x for x in out}
        assert by[1]["skipped"].startswith("Cell 4j: ") and \
            by[99]["skipped"] == "no plane chosen (iteration line_short)"
        assert not os.path.exists(os.path.join(tmp, "fits", "planefit_1.png"))
        assert not os.path.exists(os.path.join(tmp, "fits", "planefit_99.png"))
        assert "png" in by[4] and os.path.exists(by[4]["png"])


def test_determinism():
    import iteration_fits as IF
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, _p, recs, _j = _cell4j(tmp, 0.8, nodes=(4,))
        a = IF.run(swc, prov, cfg, recs, os.path.join(tmp, "a"), "999", log=lambda m: None)
        b = IF.run(swc, prov, cfg, recs, os.path.join(tmp, "b"), "999", log=lambda m: None)
        for x, y in zip(a, b):
            x, y = dict(x), dict(y)
            x.pop("png"), y.pop("png")
            assert json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True), (x, y)


def test_edge_cases():
    import matplotlib.pyplot as plt
    import iteration_fits as IF
    import plane_differences as PD
    from allen_diameter.plotting import figures as FG
    keys = ("node_id", "k_iter", "k_star", "k_swc", "d_iter_um", "d_refit_um", "fit_status", "v0_hat_um", "s_um",
            "u_um", "edges_v_um", "edges_xy_um", "line_xy_um", "sel_round", "sel_half_um", "line_mult",
            "iter_status", "d_hat_um", "allen_2r_um", "square_half_um", "notes", "png")
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, swc_path, recs, path = _cell4j(tmp, 0.8, nodes=(3, 4, 1))
        out_dir = os.path.join(tmp, "fits")
        logs = []
        out = IF.run(swc, prov, cfg, recs, out_dir, "999", ncol=2, log=logs.append)
        assert [x["node_id"] for x in out] == [r["node_id"] for r in recs] and "skipped" in out[2]
        for x, r in zip(out[:2], recs[:2]):
            assert set(x) == set(keys), set(keys) ^ set(x)
            assert x["d_refit_um"] == r["d_iter_um"] and x["notes"] == [] and x["line_mult"] == 3.0
            assert x["sel_round"] == r["n_rounds"] and x["iter_status"] == "converged"     # the repeat round
            assert abs(x["sel_half_um"] - 1.5 * x["d_refit_um"]) <= 1e-12               # sized from d_iter itself
            assert os.path.basename(x["png"]) == "planefit_%d.png" % x["node_id"] and os.path.exists(x["png"])
        with open(os.path.join(out_dir, "planefit_999.json")) as f:
            assert len(json.load(f)) == 3
        assert os.path.exists(os.path.join(out_dir, "planefit_overview_999.png"))
        assert any("the fit redone at k_iter equals Cell 4j's d_iter on 2 of 2" in m for m in logs), logs[-2:]
        # every text inside both figures, rebuilt from the run's own inputs
        rec = recs[1]
        st, _n, ov = _overlay(cfg, swc, prov, rec)
        blk = dict(label="node 4: " + "a long label " * 14, sublabel="the line " * 30, plane=ov["plane"],
                   extent=ov["extent"], o=ov["o"], e_u=ov["e_u"], line_xy=ov["line_xy"], edges_xy=ov["edges_xy"],
                   edges_v=ov["edges_v"], d=ov["d"], v_sel=ov["v_sel"], prof_sel=ov["prof_sel"], s_um=ov["s_um"],
                   v_fit=ov["fit"]["v_um"] + ov["s_um"], prof_fit=ov["fit"]["profile"], model=ov["model"],
                   B_bar=ov["fit"]["B_bar"], fit_half_um=3.0)
        fig = FG.iteration_fit_figure(blk, "a test title")
        _texts_inside(fig)
        assert len(fig.legends[0].get_texts()) == 6
        drawn = [np.column_stack([ln.get_xdata(), ln.get_ydata()]) for ln in fig.axes[0].get_lines()]
        assert any(x.shape == (2, 2) and np.allclose(x, ov["line_xy"], rtol=0, atol=1e-12) for x in drawn)
        assert any(x.shape == (2, 2) and np.allclose(x, ov["edges_xy"], rtol=0, atol=1e-12) for x in drawn)
        verticals = sorted(float(ln.get_xdata()[0]) for ln in fig.axes[1].get_lines()
                           if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1])
        assert np.allclose(verticals, ov["edges_v"], rtol=0, atol=1e-12), (verticals, ov["edges_v"])
        plt.close(fig)
        pan = dict(label="node 4: k_iter 0 (k* 0)\nd 0.85 um", plane=ov["plane"], extent=ov["extent"], o=ov["o"],
                   e_u=ov["e_u"], line_xy=ov["line_xy"], edges_xy=ov["edges_xy"])
        for blocks, ncol, n_ax in (([pan] * 7, 3, 7), ([pan], 5, 1)):            # an overview of one node: 1 column
            fig = FG.iteration_fits_overview(blocks, "an overview title", ncol=ncol)
            assert len(fig.axes) == n_ax
            _texts_inside(fig)
            plt.close(fig)
        # a subset of nodes
        sub = IF.run(swc, prov, cfg, recs, os.path.join(tmp, "sub"), "999", nodes=[4], log=lambda m: None)
        assert [x["node_id"] for x in sub] == [4]
        _check_cli(IF, swc_path, path, tmp)
    with tempfile.TemporaryDirectory() as tmp:                 # 3.0 um: Cell 4j made the square again, larger
        cfg, swc, prov, _p, recs, _j = _cell4j(tmp, 3.0, nodes=(4,))
        rec = recs[0]
        assert rec["n_restacks"] >= 1, rec["n_restacks"]
        st, notes, ov = _overlay(cfg, swc, prov, rec)
        assert st["square_half_um"] == rec["half_um"] and not notes and not ov["notes"], (st["square_half_um"],
                                                                                         rec["half_um"], notes)
        assert ov["h_sel"] + 2 * DELTA <= st["square_half_um"] + 1e-9                 # the line lies in the square
        assert np.all(np.isfinite(ov["prof_sel"]))
        # off k*, the model is drawn with plane k_iter's own background, not the pilot's (fitted at k*)
        from allen_diameter.model import tube_model
        f, r = ov["fit"], st["result"]
        assert ov["k"] != int(r.k_star) and f["B_bar"] != float(r.B_bar), (ov["k"], r.k_star, f["B_bar"], r.B_bar)
        ref = tube_model.model_profile(f["v_um"], f["d_hat_um"], f["alpha_hat"], f["v0_hat_um"],
                                       cfg.measure.sigma_fit_um, f["B_bar"], f["n_nodes"])
        assert np.array_equal(ov["model"], ref)
        assert PD.ITER_LINE_MULT == rec["line_mult"] == 3.0


def _check_cli(IF, swc_path, iter_json, tmp):
    import allen_image_io as aio
    import run_cell

    class _Fetcher:
        def __init__(self, cache_dir="", verbose=True):
            self.n_cache_hits, self.n_requests, self.bytes_downloaded = 0, 0, 0
    saved = (aio.HttpFetcher, aio.list_images, aio.plane_table, run_cell.real_provider, IF.run)
    calls = []
    try:
        aio.HttpFetcher, aio.list_images, aio.plane_table = _Fetcher, (lambda specimen: None), (lambda images: None)
        run_cell.real_provider = lambda *x, **k: None
        IF.run = lambda *x, **k: calls.append((x, k)) or []
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            assert IF.main(["--specimen", "999", "--iter-json", iter_json, "--out-dir", os.path.join(tmp, "cli"),
                            "--swc", swc_path, "--nodes", "4,3", "--ncol", "4", "--half-um", "3.5", "--shift-x",
                            "2.5", "--z0", "1.0"]) == 0
            try:
                IF.main(["--specimen", "999", "--iter-json", iter_json, "--out-dir", tmp, "--swc", swc_path,
                         "--ncol", "0"])
            except SystemExit as e:
                assert e.code == 2
            else:
                raise AssertionError("--ncol 0 accepted")
    finally:
        aio.HttpFetcher, aio.list_images, aio.plane_table, run_cell.real_provider, IF.run = saved
    (x, k), = calls
    got = inspect.signature(IF.run).bind(*x, **k).arguments          # by name: the call's positions may change
    assert got["nodes"] == [4, 3] and got["ncol"] == 4 and got["half_um"] == 3.5 and got["specimen"] == "999", got
    assert got["transform"]["shift_full_px"] == (2.5, 0.0) and got["transform"]["z0_um"] == 1.0, got["transform"]
    assert [r["node_id"] for r in got["records"]] == [3, 4, 1], got["records"]


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
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
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
        headline = lines[0] if lines else ""
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, headline))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
