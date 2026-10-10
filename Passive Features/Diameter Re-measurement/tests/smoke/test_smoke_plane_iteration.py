"""Smoke test for the Allen-guided iteration of the gradient energy -- Block
11 in specs/SPEC.md (D-042, the user's request of 2026-10-10, 15:30, and the
option chosen: "Allen first, then refine"; a diagnostic, not a focus rule).

For node j with Allen's radius r_A, the multiplier m (2) and at most R (5)
rounds: d_0 = 2 r_A; round i = 1..R takes the line of half-length
h_i = m d_{i-1} / 2 through the SWC node across the fitted heading, the
gradient energy over the whole line
    G^(h_i)_k = B_k^-2 * trapezoid over the line of (dI~_k/dv)^2
(focus.plane_gradient_energies, B_k the block background of the gradient
evaluation, the same in every round), and its plane k_i = argmax_k G (the
first of equal values). If k_i was picked before the iteration stops --
"converged" when k_i = k_{i-1}, "cycle" otherwise -- and the fit made there
is reused; else d_i = d_hat(k_i), Block 5's final fit (fit.fit_profile) at
plane k_i with every other input the pilot's (offsets, the pilot's centre,
heading and tilt, B_bar of plane k_i), so that d_hat(k*) is the pilot's own
d_hat. Other endings: "fit_failed", "no_plane", "line_short",
"square_small", "max_rounds".

Checks
    test_known_answer   iterate_planes on hand-built tables: a fixed point in
                        3 rounds (the planes, the diameters, the half-lengths
                        m d / 2 with m = 2 and 3, which fits are made and how
                        often), a cycle, the round limit (R = 5 and R = 1), a
                        fit failing (NaN, 0, negative, absent), a round with
                        no plane, a line too short; the trajectory text by
                        hand; check_iteration's accepted forms
    test_reference      on the synthetic cell (fixtures_cell, d 0.8 um):
                        refit_at_plane at k* equals the pilot's own fit to the
                        last bit (d, mu, v0, B_bar, status); at two other
                        planes it equals Block 5's functions called by hand on
                        the pipeline's own block (survey.node_planes, its own
                        plane range); B_bar is the plane's own (a gain on one
                        plane scales its B_bar, not d); started from the
                        pilot's d_hat, round
                        1's G and pick equal gradient_evaluation's d-line
                        (Cell 4h); every round's G equals its definition
    test_convergence    on synthetic tubes: 0.8 um (Allen 2r 0.6 um) converges
                        in 2 rounds at the in-focus plane with the pilot's
                        d_hat; 3.0 um (Allen's 0.6 um line inside the tube)
                        ends within one plane of the axis plane, converged in
                        at most 4 rounds, d within 20 % of 3.0 um, although
                        k* is 4 planes off; the restack path gives the same
                        planes and fits as a square large from the start (G to
                        1e-12), and run() restacks once with m = 4
    test_invariants     a gain on every plane leaves the trajectory unchanged
                        (planes equal, G to 1e-12, d to 1e-6 um); h_i = m
                        d_{i-1} / 2 and d_in of round i + 1 = the fit of round
                        i on the real chain
    test_contract       refusals (d_0, m, R) in iterate_planes and in run()
                        before any node and before the output folder; the
                        fit's endings ("no_plane", "profile_nan",
                        "bbar_nonpositive"); "square_small" without restack;
                        a restack returning other planes refused; "line_short"
                        in round 1
    test_determinism    two runs of gradient_iteration: equal records
    test_edge_cases     scripts/plane_differences.run --evaluation iterate on
                        the 0.8 um cell: the records (keys, values, the soma
                        skipped), the JSON, the summary CSV and figure, the
                        per-node figure (no line on the planes, every text
                        inside the figure, the note line in the summary);
                        picks_summary_figure skips a curve held as None; the
                        CLI passes --evaluation iterate, --max-rounds and
                        --line-mult on (server stubbed, run replaced)

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_plane_iteration.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import contextlib
import csv
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

from allen_diameter.analysis import focus as FO  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402

SEED = 20261010
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
DELTA = 0.1144
NAN = float("nan")


def _cell(tmp, d_true=0.8):
    """(cfg, swc, provider, swc path) of fixtures_cell's synthetic cell: a tube of d_true um at plane 0 (z 0.05
    um), Allen radius 0.3 um, planes -7..7; block_half_um 3.5 as in the gradient smoke test."""
    import run_cell
    from fixtures_cell import synthetic_cell
    base = default_config()
    cfg = dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5))
    swc, fetcher, planes, swc_path = synthetic_cell(tmp, cfg, d_true=d_true, mu=1.0)
    return cfg, swc, run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um), swc_path


class _Table:
    """plane_of and fit_at from hand-built tables, with the calls recorded."""

    def __init__(self, planes, fits):
        self.planes, self.fits, self.asked, self.fitted = planes, fits, [], []

    def plane_of(self, h):
        self.asked.append(h)
        k = self.planes[round(h, 9)]
        return (k, {}) if not isinstance(k, dict) else (None, k)

    def fit_at(self, k):
        self.fitted.append(k)
        return self.fits[k]


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    import plane_differences as PD
    # a fixed point, m = 2 (h = d): d0 0.5 -> k 3, fit 1.0 -> k 4, fit 1.2 -> k 4 again: converged in 3 rounds
    t = _Table({0.5: 3, 1.0: 4, 1.2: 4}, {3: dict(d_hat_um=1.0, fit_status="converged"),
                                          4: dict(d_hat_um=1.2, fit_status="at_bound")})
    res = PD.iterate_planes(0.5, 2.0, t.plane_of, t.fit_at, 5)
    assert res["status"] == "converged" and res["n_rounds"] == 3 and res["k_first"] == 3 and res["k_final"] == 4
    assert res["d_final_um"] == 1.2 and t.fitted == [3, 4] and t.asked == [0.5, 1.0, 1.2], (res, t.fitted, t.asked)
    want = [(1, 0.5, None, 0.5, 3, 1.0, True), (2, 1.0, 3, 1.0, 4, 1.2, True), (3, 1.2, 4, 1.2, 4, 1.2, False)]
    got = [(x["round"], x["d_in_um"], x["d_in_k"], x["half_um"], x["k"], x["d_hat_um"], x["refit"])
           for x in res["rounds"]]
    assert got == want, got
    assert res["rounds"][1]["fit_status"] == "at_bound"     # a fit at a bound sizes the next line like any other
    assert PD.trajectory_text(dict(res, d_start_um=0.5)) == (
        "start d 0.50 um | r1 +-0.50 um k 3 d 1.00 um | r2 +-1.00 um k 4 d 1.20 um | r3 +-1.20 um k 4 d 1.20 um "
        "repeat | converged")
    # m = 3: h = 1.5 d (0.75, 1.5, 1.8 exactly representable as computed)
    t = _Table({0.75: 3, 1.5: 4, 1.8: 4}, {3: dict(d_hat_um=1.0), 4: dict(d_hat_um=1.2)})
    res = PD.iterate_planes(0.5, 3.0, t.plane_of, t.fit_at, 5)
    assert [x["half_um"] for x in res["rounds"]] == [0.75, 1.5, 0.5 * 3.0 * 1.2] and res["status"] == "converged"
    # a cycle: 3 -> 5 -> 3; the repeated plane's fit is reused, not made again
    t = _Table({0.5: 3, 1.0: 5, 0.4: 3}, {3: dict(d_hat_um=1.0), 5: dict(d_hat_um=0.4)})
    res = PD.iterate_planes(0.5, 2.0, t.plane_of, t.fit_at, 5)
    assert res["status"] == "cycle" and res["n_rounds"] == 3 and res["k_final"] == 3 and res["d_final_um"] == 1.0
    assert t.fitted == [3, 5] and res["rounds"][-1]["refit"] is False
    # the round limit: five new planes in five rounds; R = 1 stops after round 1 with its fit
    t = _Table({0.5: 1, 0.6: 2, 0.7: 3, 0.8: 4, 0.9: 5}, {1: dict(d_hat_um=0.6), 2: dict(d_hat_um=0.7),
                                                         3: dict(d_hat_um=0.8), 4: dict(d_hat_um=0.9),
                                                         5: dict(d_hat_um=1.0)})
    res = PD.iterate_planes(0.5, 2.0, t.plane_of, t.fit_at, 5)
    assert res["status"] == "max_rounds" and res["n_rounds"] == 5 and res["k_final"] == 5 and res["d_final_um"] == 1.0
    assert t.fitted == [1, 2, 3, 4, 5]
    t.fitted = []
    res = PD.iterate_planes(0.5, 2.0, t.plane_of, t.fit_at, 1)
    assert res["status"] == "max_rounds" and res["n_rounds"] == 1 and res["k_final"] == 1 and t.fitted == [1]
    # a fit failing in round 2: the plane stands, its d is NaN
    for bad in (dict(d_hat_um=NAN, fit_status="profile_nan"), dict(d_hat_um=0.0), dict(d_hat_um=-1.0), dict()):
        t = _Table({0.5: 3, 1.0: 4}, {3: dict(d_hat_um=1.0), 4: bad})
        res = PD.iterate_planes(0.5, 2.0, t.plane_of, t.fit_at, 5)
        assert res["status"] == "fit_failed" and res["n_rounds"] == 2 and res["k_final"] == 4, (bad, res)
        assert (math.isnan(res["d_final_um"]) or res["d_final_um"] <= 0) and t.fitted == [3, 4], bad
    # no plane in round 2: the iteration stands at round 1's plane and fit; the reason given is kept
    for stop, want in (({}, "no_plane"), (dict(stop="line_short"), "line_short"),
                       (dict(stop="square_small"), "square_small")):
        t = _Table({0.5: 3, 1.0: stop or dict(stop=None)}, {3: dict(d_hat_um=1.0)})
        res = PD.iterate_planes(0.5, 2.0, t.plane_of, t.fit_at, 5)
        assert res["status"] == want and res["k_final"] == 3 and res["d_final_um"] == 1.0 and res["n_rounds"] == 2, \
            (want, res)
        assert res["rounds"][-1]["k"] is None and "stop" not in res["rounds"][-1]
    # no plane in round 1: nothing picked
    t = _Table({0.5: dict(stop="line_short")}, {})
    res = PD.iterate_planes(0.5, 2.0, t.plane_of, t.fit_at, 5)
    assert res["status"] == "line_short" and res["k_first"] is None and res["k_final"] is None and \
        math.isnan(res["d_final_um"]) and t.fitted == []
    assert PD.trajectory_text(dict(res, d_start_um=0.5)) == "start d 0.50 um | r1 +-0.50 um k n/a | line_short"
    # check_iteration's accepted forms
    assert PD.check_iteration(2, 5) == (2.0, 5) and PD.check_iteration(2.5, 5.0) == (2.5, 5)
    assert PD.check_iteration(1.0, np.int64(3)) == (1.0, 3)


def test_reference():
    import plane_differences as PD
    from allen_diameter.analysis import node_pipeline, survey
    from allen_diameter.analysis import profiles as PR
    from allen_diameter.analysis.fit import fit_profile
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, _ = _cell(tmp)
        st = PD.node_stack(swc, prov, cfg, 4, None, 6, None, 0.0, None, 2.0, 2.0)
        r = st["result"]
        # at k* the inputs are Block 5's own: the pilot's fit to the last bit
        out = PD.refit_at_plane(st, cfg, r.k_star)
        assert (out["d_hat_um"], out["mu_hat_per_um"], out["v0_hat_um"], out["B_bar"], out["fit_status"]) == \
            (r.d_hat_um, r.mu_hat_per_um, r.v0_hat_um, r.B_bar, r.fit_status), (out, r.d_hat_um)
        # at other planes: Block 5's functions by hand on the pipeline's own block (its own plane range, -4..4); at
        # k* - 3 the background differs from k*'s (209 against 210 grey levels)
        pl = survey.node_planes(swc, prov, cfg, 4)
        m = cfg.measure
        v = PR.profile_offsets(m)
        th = float(r.theta_rad)
        y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
        for k in (int(r.k_star) - 3, int(r.k_star) + 1):
            j = k - int(pl["ks"][0])
            assert 0 <= j < len(pl["ks"]) and int(pl["ks"][j]) == k
            plane = pl["block"][j]
            B, _ok = node_pipeline.node_background(plane, pl["frame"], pl["branch"].xyz_um[pl["index"], :2],
                                                   pl["branch"], m)
            I = PR.sample_profile(plane, pl["frame"], np.array([r.cx_um, r.cy_um]), y_hat, e_u, v, PR.n_along(m),
                                  m.profile_step_um)
            ref = fit_profile(v, I, float(r.phi_rad), float(B), m)
            got = PD.refit_at_plane(st, cfg, k)
            assert got["d_hat_um"] == ref.d_hat_um and got["B_bar"] == B and got["fit_status"] == ref.status, \
                (k, got, ref.d_hat_um)
            assert got["d_hat_um"] > r.d_hat_um          # out of focus the profile is wider (blur)
        # B_bar is the plane's own: a gain of 0.8 on plane k alone scales its B_bar by 0.8 (a median: one sample, or
        # the mean of two, so to 1e-12) and leaves its d_hat to 1e-6 um (the fit's termination tolerance)
        k = int(r.k_star) + 1
        j = k - int(st["ks"][0])
        dim = np.array(st["block"])
        dim[j] *= 0.8
        a, b = PD.refit_at_plane(st, cfg, k), PD.refit_at_plane(dict(st, block=dim), cfg, k)
        assert abs(b["B_bar"] - 0.8 * a["B_bar"]) <= 1e-12 * a["B_bar"] and abs(b["d_hat_um"] - a["d_hat_um"]) <= 1e-6, \
            (a, b)
        # started from the pilot's d_hat, round 1 is Cell 4h's d-line: the same G and pick
        it = PD.gradient_iteration(st, cfg, d_start_um=float(r.d_hat_um), max_rounds=1)
        gd = PD.gradient_evaluation(st, cfg, lines_um=())["lines"][0]
        assert gd["key"] == "Gd" and it["rounds"][0]["half_um"] == gd["half_um"]
        assert np.array_equal(it["rounds"][0]["G"], gd["G"], equal_nan=True)
        assert it["k_first"] == PD._argmax_plane(st["ks"], gd["G"])
        # every round's G is its definition: plane_gradient_energies on that line's profiles with the block background
        it = PD.gradient_iteration(st, cfg)
        B = PD.block_background(st)[0]
        assert np.array_equal(it["background"], B, equal_nan=True)
        for x in it["rounds"]:
            _h, vv, _y, _e, prof = PD.line_profiles(st, cfg, x["half_um"])
            assert x["n_samples"] == vv.size == 2 * int(round(x["half_um"] / DELTA)) + 1
            assert np.array_equal(x["G"], FO.plane_gradient_energies(prof, vv, B, cfg.measure, st["valid"]),
                                  equal_nan=True), x["round"]
            assert x["k"] == PD._argmax_plane(st["ks"], x["G"])


def test_convergence():
    import plane_differences as PD
    with tempfile.TemporaryDirectory() as tmp:          # 0.8 um: Allen's 0.6 um line already sees the edges
        cfg, swc, prov, _ = _cell(tmp, 0.8)
        st = PD.node_stack(swc, prov, cfg, 4, None, 6, None, 0.0, None, 2.0, 2.0)
        it = PD.gradient_iteration(st, cfg)
        assert it["status"] == "converged" and it["n_rounds"] == 2 and it["k_first"] == 0 and it["k_final"] == 0
        assert it["d_final_um"] == st["result"].d_hat_um and it["rounds"][0]["half_um"] == 0.6
    with tempfile.TemporaryDirectory() as tmp:          # 3.0 um: Allen's 0.6 um line lies inside the tube
        cfg, swc, prov, _ = _cell(tmp, 3.0)
        st = PD.node_stack(swc, prov, cfg, 4, None, 6, None, 0.0, None, 2.0, 2.0)
        it = PD.gradient_iteration(st, cfg)
        # the axis plane is 0 (z 0.05 um, dz 0.28 um); k* of the pipeline misses it by 4 planes on this tube
        assert abs(int(st["result"].k_star)) >= 3, st["result"].k_star
        assert it["status"] == "converged" and it["n_rounds"] <= 4 and abs(it["k_final"]) <= 1, it["status"]
        # 20 %: the fit's own error one plane from the axis on a 3 um tube (the pilot's d_hat is 9 % low at k*)
        assert abs(it["d_final_um"] - 3.0) <= 0.6, it["d_final_um"]
        # the restack path: a square too small for the later lines is made again, with the same planes and fits as a
        # square large from the start (G to 1e-12: the two squares' frames differ, so the bilinear weights differ
        # in the last bits)
        calls = []

        def restack(h):
            calls.append(h)
            return PD.node_stack(swc, prov, cfg, 4, None, 4, None, 0.0, h, 4.0, 4.0)
        st_s = PD.node_stack(swc, prov, cfg, 4, None, 4, None, 0.0, None, 4.0, 4.0)
        it_s = PD.gradient_iteration(st_s, cfg, line_mult=4.0, restack=restack)
        it_b = PD.gradient_iteration(PD.node_stack(swc, prov, cfg, 4, None, 4, None, 0.0, 7.0, 4.0, 4.0), cfg,
                                     line_mult=4.0)
        assert it_s["n_restacks"] == len(calls) >= 1 and it_b["n_restacks"] == 0, calls
        assert it_s["st"]["square_half_um"] >= max(x["half_um"] for x in it_s["rounds"]) + 2 * DELTA - 1e-9
        assert [x["k"] for x in it_s["rounds"]] == [x["k"] for x in it_b["rounds"]]
        for xs, xb in zip(it_s["rounds"], it_b["rounds"]):
            assert xs["d_hat_um"] == xb["d_hat_um"] and np.allclose(xs["G"], xb["G"], rtol=1e-12, atol=0,
                                                                     equal_nan=True), xs["round"]
        recs = PD.run(swc, prov, cfg, [4], os.path.join(tmp, "m4"), "999", planes_half=4, evaluation="iterate",
                      line_mult=4.0, log=lambda m: None)
        assert recs[0]["n_restacks"] == 1 and recs[0]["square_fetched"] is True, recs[0]["n_restacks"]
        assert [x["k"] for x in recs[0]["rounds"]] == [x["k"] for x in it_b["rounds"]]
        assert recs[0]["half_um"] >= max(x["half_um"] for x in recs[0]["rounds"]) + 2 * DELTA - 1e-9
        prof = np.array(recs[0]["profiles"], dtype=float)
        assert np.all(np.isfinite(prof[np.array(recs[0]["valid"])])), "profiles beyond the square"


def test_invariants():
    import plane_differences as PD
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, _ = _cell(tmp, 0.8)
        st = PD.node_stack(swc, prov, cfg, 4, None, 6, None, 0.0, None, 2.0, 2.0)
        it = PD.gradient_iteration(st, cfg, max_rounds=5)
        # a gain on every plane: G is divided by B^2 and the fit's model scales with B_bar, so nothing moves (G to
        # 1e-12, rounding in the background's mean; d to 1e-6 um, the fit's termination tolerance)
        st_c = dict(st, stack=0.75 * st["stack"], block=0.75 * st["block"])
        it_c = PD.gradient_iteration(st_c, cfg, max_rounds=5)
        assert [x["k"] for x in it_c["rounds"]] == [x["k"] for x in it["rounds"]] and it_c["status"] == it["status"]
        for x, y in zip(it["rounds"], it_c["rounds"]):
            assert np.allclose(x["G"], y["G"], rtol=1e-12, atol=0, equal_nan=True), x["round"]
            assert abs(x["d_hat_um"] - y["d_hat_um"]) <= 1e-6, (x["d_hat_um"], y["d_hat_um"])
        # the chain, on a 4-round run from a small start (m = 3): h_i = m d_{i-1} / 2, d_in of round i + 1 is the fit
        # of round i, d_in_k its plane
        it = PD.gradient_iteration(st, cfg, d_start_um=0.3, line_mult=3.0, max_rounds=4)
        rs = it["rounds"]
        assert rs[0]["d_in_um"] == 0.3 and rs[0]["d_in_k"] is None
        for x in rs:
            assert x["half_um"] == 0.5 * 3.0 * x["d_in_um"]
        for a, b in zip(rs[:-1], rs[1:]):
            assert b["d_in_um"] == a["d_hat_um"] and b["d_in_k"] == a["k"]


def test_contract():
    import plane_differences as PD
    for d, m, R in ((0.0, 2.0, 5), (-1.0, 2.0, 5), (NAN, 2.0, 5), (float("inf"), 2.0, 5), (0.5, 0.0, 5),
                    (0.5, -2.0, 5), (0.5, NAN, 5), (0.5, 2.0, 0), (0.5, 2.0, -1), (0.5, 2.0, 2.5), (0.5, 2.0, True),
                    (0.5, 2.0, "5"), (0.5, 2.0, None), (0.5, 2.0, NAN)):
        try:
            PD.iterate_planes(d, m, lambda h: (1, {}), lambda k: dict(d_hat_um=1.0), R)
        except ValueError:
            continue
        raise AssertionError("iterate_planes accepted d=%r, m=%r, R=%r" % (d, m, R))
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, _ = _cell(tmp, 0.8)
        for m, R in ((0.0, 5), (2.0, 0), (2.0, 1.5)):          # refused before any node and before the folder
            out = os.path.join(tmp, "refused_%g_%g" % (m, R))
            try:
                PD.run(swc, prov, cfg, [4], out, "999", evaluation="iterate", line_mult=m, max_rounds=R,
                       log=lambda x: None)
            except ValueError:
                assert not os.path.exists(out)
                continue
            raise AssertionError("run accepted line_mult=%r, max_rounds=%r" % (m, R))
        st = PD.node_stack(swc, prov, cfg, 4, None, 3, None, 0.0, None, 2.0, 2.0)
        # the fit's endings
        assert PD.refit_at_plane(st, cfg, 9)["fit_status"] == "no_plane"             # not among -3..3
        st_m = dict(st, valid=np.array([k != 1 for k in st["ks"]]))
        f = PD.refit_at_plane(st_m, cfg, 1)
        assert f["fit_status"] == "no_plane" and math.isnan(f["d_hat_um"])
        far = dataclasses.replace(st["result"], cx_um=st["result"].cx_um + 50.0)
        f = PD.refit_at_plane(dict(st, result=far), cfg, 0)
        assert f["fit_status"] == "profile_nan" and math.isnan(f["d_hat_um"]) and f["B_bar"] > 0
        dark = np.array(st["block"])
        dark[list(st["ks"]).index(0)] = 0.0
        f = PD.refit_at_plane(dict(st, block=dark), cfg, 0)
        assert f["fit_status"] == "bbar_nonpositive" and f["B_bar"] == 0.0 and math.isnan(f["d_hat_um"])
        # a square too small without restack: the round stops, the iteration stands at the last plane picked
        st_small = PD.node_stack(swc, prov, cfg, 4, None, 3, 0.3, 0.0, None, None, 2.0)
        assert abs(st_small["square_half_um"] - (0.6 + 2 * DELTA)) <= 1e-12
        it = PD.gradient_iteration(st_small, cfg)
        assert it["status"] == "square_small" and it["n_rounds"] == 2 and it["k_final"] == it["k_first"] == 0, it
        assert "G" not in it["rounds"][1] and it["n_restacks"] == 0
        # round 2's line (about +-0.85 um) lies inside a +-0.9 um square but closer than 2 pixels to its edge: stopped
        it = PD.gradient_iteration(PD.node_stack(swc, prov, cfg, 4, None, 3, 0.9, 0.0, None, None, 2.0), cfg)
        assert it["status"] == "square_small" and it["n_rounds"] == 2 and 0.9 - 2 * DELTA < it["rounds"][1]["half_um"]
        # a restack returning other planes is refused, by the check on the planes
        try:
            PD.gradient_iteration(st_small, cfg, restack=lambda h: PD.node_stack(swc, prov, cfg, 4, None, 2, None, 0.0,
                                                                                  h, 2.0, 2.0))
        except ValueError as e:
            assert "restack returned planes" in str(e), e
        else:
            raise AssertionError("a restack with other planes accepted")
        # a starting line of fewer than 3 samples
        it = PD.gradient_iteration(st, cfg, d_start_um=0.05)
        assert it["status"] == "line_short" and it["k_final"] is None and it["n_rounds"] == 1


def test_determinism():
    import plane_differences as PD
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, _ = _cell(tmp, 0.8)
        st = PD.node_stack(swc, prov, cfg, 4, None, 4, None, 0.0, None, 2.0, 2.0)
        a, b = PD.gradient_iteration(st, cfg), PD.gradient_iteration(st, cfg)
        assert a["status"] == b["status"] and len(a["rounds"]) == len(b["rounds"])
        for x, y in zip(a["rounds"], b["rounds"]):
            assert set(x) == set(y)
            for key in x:
                if key == "G":
                    assert np.array_equal(x[key], y[key], equal_nan=True)
                else:
                    assert x[key] == y[key] or (isinstance(x[key], float) and math.isnan(x[key]) and
                                                math.isnan(y[key])), key


def _plane_axes(fig):
    return [ax for ax in fig.axes if ax.get_title().startswith("k ")]


def _texts_inside(fig):
    """Every figure-level text (the block labels, the suptitle, the legends' entries) and every axes title lies
    inside the figure's width, to the pixel."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    texts = list(fig.texts) + [ax.title for ax in fig.axes]
    texts += [t for leg in fig.legends for t in leg.get_texts()]
    if getattr(fig, "_suptitle", None) is not None:
        texts.append(fig._suptitle)
    for t in texts:
        if t.get_text():
            bb = t.get_window_extent(renderer=ren)
            assert bb.x0 >= -1.0 and bb.x1 <= fig.bbox.width + 1.0, (t.get_text()[:60], bb.x0, bb.x1, fig.bbox.width)


RECORD_KEYS = ("line_mult", "max_rounds", "d_start_um", "rounds", "iter_status", "n_rounds", "k_allen", "k_iter",
               "d_iter_um", "G_allen", "G_iter", "G_Gd", "k_Gd", "d_line_half_um", "trajectory", "iter_note",
               "n_restacks", "background", "bg_frac", "bg_margin_used", "v_um", "profiles", "profile_half_um",
               "pick_keys", "png")
ROUND_KEYS = ("round", "d_in_um", "d_in_k", "half_um", "n_samples", "k", "G", "d_hat_um", "fit_status", "v0_hat_um",
              "mu_hat_per_um", "B_bar", "bbar_ok", "refit")


def test_edge_cases():
    import matplotlib.pyplot as plt
    import plane_differences as PD
    from allen_diameter.plotting import figures as FG
    with tempfile.TemporaryDirectory() as tmp:
        cfg, swc, prov, swc_path = _cell(tmp, 0.8)
        out = os.path.join(tmp, "iter")
        logs = []
        recs = PD.run(swc, prov, cfg, [3, 4, 1], out, "999", planes_half=6, evaluation="iterate", log=logs.append)
        assert "skipped" in recs[2] and recs[2]["node_id"] == 1               # the soma
        for rec in recs[:2]:
            assert all(k in rec for k in RECORD_KEYS), [k for k in RECORD_KEYS if k not in rec]
            assert all(set(ROUND_KEYS) == set(x) for x in rec["rounds"]), rec["rounds"][0].keys()
            assert rec["d_start_um"] == 0.6 and rec["line_mult"] == 2.0 and rec["max_rounds"] == 5
            assert rec["iter_status"] == "converged" and rec["n_rounds"] == 2 and rec["n_restacks"] == 0
            assert rec["k_allen"] == rec["k_iter"] == rec["k_Gd"] == rec["k_star"] == 0
            assert rec["d_iter_um"] == rec["d_hat_um"] and rec["d_line_half_um"] == rec["d_hat_um"]
            assert rec["pick_keys"] == ["k_allen", "k_iter", "k_Gd"] and rec["G_allen"] == rec["rounds"][0]["G"]
            assert rec["G_iter"] == rec["rounds"][1]["G"] and rec["rounds"][1]["refit"] is False
            assert rec["half_um"] >= max(0.6, rec["d_hat_um"]) + 2 * DELTA - 1e-9    # the square holds both lines
            assert rec["profile_half_um"] == max(x["half_um"] for x in rec["rounds"]) and \
                len(rec["v_um"]) == 2 * int(round(rec["profile_half_um"] / DELTA)) + 1
            assert "," not in rec["trajectory"] and rec["trajectory"].endswith("| converged")
            assert os.path.basename(rec["png"]) == "planeiter_%d.png" % rec["node_id"] and os.path.exists(rec["png"])
        assert any("iteration endings: converged 2" in x for x in logs), logs[-3:]
        with open(os.path.join(out, "planeiter_999.json")) as f:
            assert len(json.load(f)) == 3
        with open(os.path.join(out, "planeiter_summary_999.csv")) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2 and all(c in rows[0] for c in (
            "k_allen", "k_allen_minus_k_star", "k_iter", "k_iter_minus_k_star", "k_Gd", "k_Gd_minus_k_star",
            "d_iter_um", "iter_status", "n_rounds", "n_restacks", "trajectory")), list(rows[0])
        assert rows[0]["k_iter_minus_k_star"] == "0" and rows[0]["iter_status"] == "converged"
        assert os.path.exists(os.path.join(out, "planeiter_summary_999.png"))
        # the per-node figure, rebuilt from the record: no line on the planes, every text inside the figure
        r4 = recs[1]
        ks = np.asarray(r4["ks"])
        idx = lambda k: None if k is None else int(k) - int(ks[0])  # noqa: E731
        st = PD.node_stack(swc, prov, cfg, 4, None, 6, None, 0.0, None, 2.0, 2.0)
        frames, lines = PD._marks(dict(k_iter=0, k_allen=0, k_Gd=0, k_star=0, k_star_depth=0, k_swc=0),
                                  PD._MARKS_ITER + PD._MARKS_PIPELINE)
        blk = dict(label="node 4: " + "a long label " * 25, stack=st["stack"], ks=ks, valid=st["valid"],
                   extent=st["extent"], frames=frames, lines=lines, v=np.array(r4["v_um"]),
                   prof=np.array(r4["profiles"], dtype=float),
                   rounds=[dict(round=x["round"], half_um=x["half_um"], G=x["G"], pick=idx(x["k"]))
                           for x in r4["rounds"]] + [dict(round=i, half_um=0.9, G=r4["rounds"][0]["G"], pick=6)
                                                     for i in (3, 4, 5)],
                   gd=dict(half_um=r4["d_line_half_um"], G=r4["G_Gd"], pick=idx(r4["k_Gd"])), d_start_um=0.6,
                   fits=[(1, 0, r4["d_iter_um"], True), (2, 0, r4["d_iter_um"], False), (3, 1, NAN, True)],
                   status="converged", n_rounds=5, d_pilot_um=r4["d_hat_um"], k_star=0, k_ref=0)
        fig = FG.iteration_figure([blk], "Allen-guided iteration, a test title")
        pa = _plane_axes(fig)
        assert len(pa) == ks.size and all(len(ax.get_lines()) == 0 for ax in pa)
        assert pa[list(ks).index(0)].get_title() == "k 0\nit A Gd k* dip SWC"
        _texts_inside(fig)
        assert len(fig.legends[0].get_texts()) == 5 + 3 + 3 + 1, [t.get_text() for t in fig.legends[0].get_texts()]
        plt.close(fig)
        assert FG.iteration_style(1) == ("#eda100", "o--") and FG.iteration_style(2) == ("#6da7ec", "s-") and \
            FG.iteration_style(9)[0] == "#0d366b"
        # the summary figure: the note is the panel title's third line; a curve held as None is not drawn
        series = [dict(curve="G_allen", pick="k_allen", tag="A", label="a", colour="#eda100", style="o--"),
                  dict(curve="G_Gd", pick="k_Gd", tag="Gd", label="b", colour="#1baf7a", style="^-.")]
        fig = FG.picks_summary_figure([dict(recs[0], G_Gd=None), recs[1]], series, "t", note_key="iter_note")
        t0 = fig.axes[0].get_title().split("\n")
        assert len(t0) == 3 and t0[2] == recs[0]["iter_note"] and t0[1] == "pick minus k*: A +0", t0
        assert len([ln for ln in fig.axes[0].get_lines() if len(ln.get_ydata()) == ks.size]) == 1
        plt.close(fig)
        # Allen's radius 1.0 um: round 1's line (+-2 um) is the longest, and the first square holds it (no restack)
        import copy
        swc_r = copy.deepcopy(swc)
        swc_r.radius[:] = 1.0
        rec = PD.run(swc_r, prov, cfg, [4], os.path.join(tmp, "r1"), "999", planes_half=3, half_um=0.3,
                     evaluation="iterate", log=lambda m: None)[0]
        assert rec["d_start_um"] == 2.0 and rec["rounds"][0]["half_um"] == 2.0 and rec["n_restacks"] == 0
        assert rec["half_um"] >= 2.0 + 2 * DELTA - 1e-9, rec["half_um"]
        # Allen's radius 0.02 um: round 1's line holds fewer than 3 samples; nothing is picked, the figure is drawn
        swc_r.radius[:] = 0.02
        rec = PD.run(swc_r, prov, cfg, [4], os.path.join(tmp, "r002"), "999", planes_half=3, evaluation="iterate",
                     log=lambda m: None)[0]
        assert rec["iter_status"] == "line_short" and rec["k_allen"] is None and rec["k_iter"] is None
        assert rec["G_allen"] is None and rec["G_iter"] is None and math.isnan(rec["d_iter_um"]) and \
            rec["n_rounds"] == 1 and rec["k_Gd"] is not None and os.path.exists(rec["png"])
        _check_cli(PD, swc_path, tmp)


def _check_cli(PD, swc_path, tmp):
    import allen_image_io as aio
    import run_cell

    class _Fetcher:
        def __init__(self, cache_dir="", verbose=True):
            self.n_cache_hits, self.n_requests, self.bytes_downloaded = 0, 0, 0
    saved = (aio.HttpFetcher, aio.list_images, aio.plane_table, run_cell.real_provider, PD.run)
    calls = []
    try:
        aio.HttpFetcher, aio.list_images, aio.plane_table = _Fetcher, (lambda specimen: None), (lambda images: None)
        run_cell.real_provider = lambda *x, **k: None
        PD.run = lambda *x, **k: calls.append((x, k)) or []
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            assert PD.main(["--specimen", "999", "--nodes", "4,5", "--out-dir", os.path.join(tmp, "cli"), "--swc",
                            swc_path, "--evaluation", "iterate", "--max-rounds", "3", "--line-mult", "3"]) == 0
            assert PD.main(["--specimen", "999", "--nodes", "4", "--out-dir", os.path.join(tmp, "cli"), "--swc",
                            swc_path, "--evaluation", "iterate"]) == 0
    finally:
        aio.HttpFetcher, aio.list_images, aio.plane_table, run_cell.real_provider, PD.run = saved
    (x1, k1), (x2, k2) = calls
    got = inspect.signature(PD.run).bind(*x1, **k1).arguments          # by name: the call's positions may change
    assert got["node_ids"] == [4, 5] and got["evaluation"] == "iterate" and got["max_rounds"] == 3 and \
        got["line_mult"] == 3.0, got
    got = inspect.signature(PD.run).bind(*x2, **k2).arguments
    assert got["max_rounds"] == 5 and got["line_mult"] == 2.0, got


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
