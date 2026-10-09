"""Smoke test for the gradient energy along lines of three lengths and its
sigmoid-weighted blend with the strip entropy -- Block 11 in specs/SPEC.md
(D-040, the user's request of 2026-10-09, 16:19; diagnostics, not focus rules).

For node j and plane k, with the profile I_{j,k}(v_n), v_n = n * Delta,
|n| <= N = round(h / Delta), along a line of half-length h, smoothed to
I~ by focus_smooth_px samples, and B_{j,k} the plane's background:
    G^(h)_{j,k} = B_{j,k}^-2 * trapezoid over the whole line of (dI~/dv)^2
(focus.plane_gradient_energies: focus.gradient_energy with the window
"whole_profile" and B as the override). The lines: h = 3, 5 um and the
d-line h_d = m * d_hat / 2 (m = 2: [-d_hat, d_hat]). On the d-line, over V,
the planes where G and the strip entropy H are both finite:
    g   = (G - min_V G) / (max_V G - min_V G)
    eta = (max_V H - H) / (max_V H - min_V H)
    w   = 1 / (1 + exp((d_hat - d0) / s))          (d0 = 1.5, s = 0.3 um)
    J   = w g + (1 - w) eta
Picks: argmax G per line, argmax J, argmin H. The blend's choice between
G's pick p and H's pick q flips at w* = (1 - eta_p) / ((1 - eta_p) + (1 - g_q)).

Checks
    test_known_answer   the sigmoid at d0 (1/2) and at d0 +- s ln 3 (1/4,
                        3/4); min-max by hand (flat curve: zeros); the blend
                        by hand, its pick on both sides of w* = 1/2 and at the
                        tie (first plane); G of a linear ramp, smoothing off:
                        alpha^2 * 2 N Delta / B^2 exactly; the d-line's
                        half-length; the node choice by diameter on the
                        hand-built seven-stretch SWC worked out by hand
                        (2 per bin, 1 per bin, the added nodes, the order by
                        d_hat, the refusals)
    test_reference      plane_gradient_energies against focus.gradient_energy
                        called plane by plane with the window and background
                        rules replaced; blend_scores against a loop by hand
    test_convergence    G of a Gaussian dip, smoothing off, against the
                        integral A^2 sqrt(pi) / (2 sigma B^2): halving the
                        step divides the error by 3.5 to 4.5 (second order)
    test_invariants     a gain on one plane (profile and background) leaves
                        its G unchanged; positive affine maps of G and of H
                        leave g, eta and J unchanged; w(d0 + x) + w(d0 - x) = 1
                        and w decreases; w = 1 picks G's plane, w = 0 the
                        entropy's
    test_noise_floor    flat ground with white noise of SD s, smoothing off:
                        the mean of G over 4000 planes within 5 standard
                        errors of (2 s^2 / Delta + (M - 2) s^2 / (2 Delta)) / B^2
    test_contract       NaN for an invalid plane, a non-finite sample, a
                        background not positive; refusals of bad shapes,
                        s <= 0, w outside [0, 1], a d_hat or multiplier not
                        positive, too many fixed lines
    test_determinism    pure functions: equal inputs, equal outputs
    test_edge_cases     scripts/plane_differences.run on the synthetic cell
                        (fixtures_cell, d 0.8 um): the gradient evaluation
                        frames plane 0 on every line, the d-line is m d_hat / 2
                        with 2 round(h / Delta) + 1 samples, the square is
                        fetched for the +-5 um line, the missing planes are
                        NaN, the background equals the profile evaluation's;
                        a node without d_hat keeps the fixed lines (gradient)
                        and is refused (blend); the blend evaluation: w from
                        d_hat, J from g and eta, the summary CSV and figure;
                        the gradient summary; the per-node figure draws no
                        line on the planes; the CLI passes the node choice and
                        the parameters on (server stubbed, run replaced)

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_plane_gradient.py

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

from allen_diameter.analysis import focus as FO  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402

SEED = 20261009
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
DELTA = 0.1144


def _measure(smooth_px=0.0):
    m = default_config().measure
    return dataclasses.replace(m, focus_smooth_px=float(smooth_px))


def _selection_rows():
    """Pilot rows on the hand-built SWC of test_smoke_stretches: stretches A 2-13, B 14-18, T1 19-24, O 25-34,
    T2 35-44 (no rows), L 45-56, R 57-60, with d_hat 0.5, 0.9, 1.2, 1.8, -, 2.5, 4.0 um, and rows made to fail."""
    good = dict(fit_status="converged", z_sub_um=1.0, steep=False, flags=None)
    dh = {}
    for n in range(2, 14):
        dh[n] = 0.5
    for n in range(14, 19):
        dh[n] = 0.9
    for n in range(19, 25):
        dh[n] = 1.2
    for n in range(25, 35):
        dh[n] = 1.8
    for n in range(45, 57):
        dh[n] = 2.5
    for n in range(57, 61):
        dh[n] = 4.0
    rows = {n: dict(good, node_id=n, d_hat_um=d) for n, d in dh.items()}
    rows[3].update(fit_status="bounds")
    rows[4].update(steep=True)
    rows[5].update(flags="crossing;stack_edge")
    rows[6].update(z_sub_um=float("nan"))
    rows[7].update(d_hat_um=float("nan"))
    rows[8].update(d_hat_um=0.0)
    del rows[9]["steep"]                            # a missing field does not qualify
    rows[10].update(steep="True")                  # a string, as a CSV read without parsing gives it
    rows[11].update(flags="faint")                 # another flag: kept
    rows[12].update(flags=float("nan"))            # an empty flags cell read as NaN: kept
    rows[13].update(d_hat_um=0.8)                  # on a bin edge: the bins are closed on the right
    rows[14].update(d_hat_um=1.0)
    return rows


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    import plane_differences as PD
    # the sigmoid: 1/2 at d0, 1/4 and 3/4 at d0 +- s ln 3 (expit(-+ln 3) = 1/(1 + 3^{+-1}))
    assert FO.sigmoid_weight(1.5, 1.5, 0.3) == 0.5
    assert abs(FO.sigmoid_weight(1.5 + 0.3 * math.log(3.0), 1.5, 0.3) - 0.25) <= 1e-15
    assert abs(FO.sigmoid_weight(1.5 - 0.3 * math.log(3.0), 1.5, 0.3) - 0.75) <= 1e-15
    assert abs(FO.sigmoid_weight(1.0, 1.5, 0.3) - 1.0 / (1.0 + math.exp(-0.5 / 0.3))) <= 1e-15
    assert abs(FO.sigmoid_weight(1.0 + 0.5 * math.log(3.0), 1.0, 0.5) - 0.25) <= 1e-15      # other d0 and s
    assert abs(FO.sigmoid_weight(2.0 - 0.8 * math.log(3.0), 2.0, 0.8) - 0.75) <= 1e-15
    assert math.isnan(FO.sigmoid_weight(float("nan"), 1.5, 0.3))
    # min-max by hand
    out = FO.minmax_scores([2.0, 4.0, 6.0, float("nan")])
    assert np.allclose(out[:3], [0.0, 0.5, 1.0], rtol=0, atol=0) and math.isnan(out[3])
    out = FO.minmax_scores([2.0, 4.0, 6.0, float("nan")], higher_is_better=False)
    assert np.allclose(out[:3], [1.0, 0.5, 0.0], rtol=0, atol=0) and math.isnan(out[3])
    out = FO.minmax_scores([3.0, 3.0, float("nan")])
    assert list(out[:2]) == [0.0, 0.0] and math.isnan(out[2])
    out = FO.minmax_scores([1.0, 5.0, 9.0], valid=[True, False, True])
    assert out[0] == 0.0 and math.isnan(out[1]) and out[2] == 1.0
    # the blend by hand: G = 1, 3, 2 and H = 5, 4, 3 give g = 0, 1, 1/2 and eta = 0, 1/2, 1
    b = FO.blend_scores([1.0, 3.0, 2.0], [5.0, 4.0, 3.0], 0.5)
    assert list(b["g"]) == [0.0, 1.0, 0.5] and list(b["eta"]) == [0.0, 0.5, 1.0] and list(b["J"]) == [0.0, 0.75, 0.75]
    # G's pick p = 1, H's pick q = 2: w* = (1 - 1/2) / ((1 - 1/2) + (1 - 1/2)) = 1/2; a tie goes to the first plane
    ks = np.array([10, 11, 12])
    assert PD._argmax_plane(ks, b["J"]) == 11
    assert PD._argmax_plane(ks, FO.blend_scores([1.0, 3.0, 2.0], [5.0, 4.0, 3.0], 0.6)["J"]) == 11
    assert PD._argmax_plane(ks, FO.blend_scores([1.0, 3.0, 2.0], [5.0, 4.0, 3.0], 0.4)["J"]) == 12
    nb = FO.blend_scores([1.0, 3.0, 2.0], [float("nan"), 4.0, 3.0], 0.5)       # plane 0 leaves V
    assert list(nb["V"]) == [False, True, True] and math.isnan(nb["J"][0]) and list(nb["g"][1:]) == [1.0, 0.0] and \
        list(nb["eta"][1:]) == [0.0, 1.0], nb
    # G of a linear ramp, smoothing off: the slope alpha everywhere, so G = alpha^2 * 2 N Delta / B^2
    N = 26
    v = np.arange(-N, N + 1) * DELTA
    P = np.vstack([2.0 * v + 50.0, -3.0 * v + 80.0])
    G = FO.plane_gradient_energies(P, v, np.array([100.0, 50.0]), _measure(0.0))
    L = 2 * N * DELTA
    assert np.allclose(G, [4.0 * L / 1e4, 9.0 * L / 2500.0], rtol=1e-12, atol=0), G
    # the d-line
    assert PD.d_line_half_um(0.8, 2.0) == 0.8 and abs(PD.d_line_half_um(0.8, 3.0) - 1.2) <= 1e-15
    # the node choice by diameter on the hand-built SWC
    import test_smoke_stretches as TS
    rows = _selection_rows()
    assert [n for n in range(2, 15) if PD.diameter_candidate(rows[n])] == [2, 11, 12, 13, 14]
    with tempfile.TemporaryDirectory() as tmp:
        swc = TS.fixture_swc(tmp)
    # bins in stretch order: (0, .8] [2, 11, 12, 13] -> spread_ranks(4, 2) = 0, 3 -> 2, 13; (.8, 1] B 14-18 -> 14, 18;
    # (1, 1.5] T1 19-24 -> 19, 24; (1.5, 2] O 25-34 -> 25, 34; (2, 3] L 45-56 -> 45, 56; (3, inf) R 57-60 -> 57, 60;
    # then node 3 (in the pilot, whatever its row says); 2 is already in; ordered by d_hat, ties by id
    got = PD.select_nodes_by_diameter(swc, list(rows.values()), add_nodes=(2, 3, 40))
    assert got == [2, 3, 13, 18, 14, 19, 24, 25, 34, 45, 56, 57, 60], got
    # one per bin: the middle candidate, (m - 1) // 2
    assert PD.select_nodes_by_diameter(swc, list(rows.values()), per_bin=1, add_nodes=()) == [11, 16, 21, 29, 50, 58]
    assert PD.select_nodes_by_diameter(swc, list(rows.values()), per_bin=0, add_nodes=(3,)) == [3]
    assert PD.select_nodes_by_diameter(swc, [], add_nodes=(2, 3)) == []
    for bad in ([], [1.0, 0.8], [0.0, 1.0], [1.0, float("nan")]):
        try:
            PD.select_nodes_by_diameter(swc, list(rows.values()), bins_um=bad)
        except ValueError:
            continue
        raise AssertionError("bins %r accepted" % (bad,))


def test_reference():
    rng = np.random.default_rng(SEED)
    m = _measure(1.0)
    v = np.arange(-30, 31) * DELTA
    P = 100.0 + 20.0 * rng.standard_normal((7, v.size))
    B = rng.uniform(80.0, 120.0, 7)
    P[2, 5] = np.nan                                 # a non-finite sample
    B[4] = -1.0                                      # a background not positive
    valid = np.array([True, True, True, True, True, False, True])
    G = FO.plane_gradient_energies(P, v, B, m, valid)
    whole = dataclasses.replace(m, focus_grad_window="whole_profile", focus_bg_rule="same_as_bbar")
    for k in range(7):
        if not valid[k] or not B[k] > 0:
            assert math.isnan(G[k]), k
            continue
        ref = FO.gradient_energy(P[k], v, whole, B_override=B[k])[0]
        assert (math.isnan(ref) and math.isnan(G[k])) or G[k] == ref, (k, G[k], ref)
    assert math.isnan(G[2])
    # the blend against a loop written out by hand
    Gc = rng.uniform(0.0, 5.0, 9)
    Hc = rng.uniform(3.0, 5.0, 9)
    Gc[3], Hc[6] = np.nan, np.nan
    w = 0.37
    b = FO.blend_scores(Gc, Hc, w)
    V = [k for k in range(9) if np.isfinite(Gc[k]) and np.isfinite(Hc[k])]
    g_lo, g_hi = min(Gc[k] for k in V), max(Gc[k] for k in V)
    h_lo, h_hi = min(Hc[k] for k in V), max(Hc[k] for k in V)
    for k in range(9):
        if k in V:
            g = (Gc[k] - g_lo) / (g_hi - g_lo)
            e = (h_hi - Hc[k]) / (h_hi - h_lo)
            assert abs(b["J"][k] - (w * g + (1.0 - w) * e)) <= 1e-15, k
        else:
            assert math.isnan(b["J"][k]) and math.isnan(b["g"][k]) and math.isnan(b["eta"][k]), k


def test_convergence():
    # I(v) = B - A exp(-v^2 / (2 sigma^2)): integral of I'^2 over the line = A^2 sqrt(pi) / (2 sigma)
    A, sig, B = 60.0, 0.4, 200.0
    exact = A * A * math.sqrt(math.pi) / (2.0 * sig) / B ** 2
    errs = []
    for step in (0.08, 0.04, 0.02):
        n = int(round(6.0 * sig / step))
        v = np.arange(-n, n + 1) * step
        I = B - A * np.exp(-v ** 2 / (2.0 * sig ** 2))
        G = FO.plane_gradient_energies(I[None, :], v, np.array([B]), _measure(0.0))[0]
        errs.append(abs(G - exact))
    r1, r2 = errs[0] / errs[1], errs[1] / errs[2]
    assert 3.5 <= r1 <= 4.5 and 3.5 <= r2 <= 4.5, (errs, r1, r2)


def test_invariants():
    rng = np.random.default_rng(SEED + 1)
    m = _measure(1.0)
    v = np.arange(-20, 21) * DELTA
    P = 120.0 + 15.0 * rng.standard_normal((5, v.size))
    B = rng.uniform(100.0, 140.0, 5)
    G = FO.plane_gradient_energies(P, v, B, m)
    P2, B2 = P.copy(), B.copy()
    P2[3] *= 0.98                                    # one plane 2 % darker after the camera: profile and background
    B2[3] *= 0.98
    G2 = FO.plane_gradient_energies(P2, v, B2, m)
    assert np.allclose(G2, G, rtol=1e-12, atol=0), (G, G2)
    Gc = rng.uniform(0.0, 5.0, 11)
    Hc = rng.uniform(3.0, 5.0, 11)
    b1 = FO.blend_scores(Gc, Hc, 0.3)
    b2 = FO.blend_scores(7.0 * Gc + 2.0, 0.5 * Hc - 1.0, 0.3)
    for key in ("g", "eta", "J"):
        assert np.allclose(b1[key], b2[key], rtol=0, atol=1e-12), key
    for x in (0.0, 0.1, 0.7, 3.0):
        assert abs(FO.sigmoid_weight(1.5 + x, 1.5, 0.3) + FO.sigmoid_weight(1.5 - x, 1.5, 0.3) - 1.0) <= 1e-15
    ws = [FO.sigmoid_weight(d, 1.5, 0.3) for d in np.linspace(0.2, 6.0, 30)]
    assert all(a > b for a, b in zip(ws, ws[1:])), ws
    ks = np.arange(11)
    import plane_differences as PD
    assert PD._argmax_plane(ks, FO.blend_scores(Gc, Hc, 1.0)["J"]) == int(np.argmax(Gc))
    assert PD._argmax_plane(ks, FO.blend_scores(Gc, Hc, 0.0)["J"]) == int(np.argmin(Hc))


def test_noise_floor():
    # flat ground, white noise of SD s, smoothing off: the central difference of an interior sample has variance
    # s^2 / (2 Delta^2), the one-sided one at each end 2 s^2 / Delta^2; with the trapezoid's weights,
    # E[integral] = 2 s^2 / Delta + (M - 2) s^2 / (2 Delta)
    rng = np.random.default_rng(SEED + 2)
    s, B, n_planes = 3.0, 100.0, 4000
    v = np.arange(-26, 27) * DELTA
    M = v.size
    P = B + s * rng.standard_normal((n_planes, M))
    G = FO.plane_gradient_energies(P, v, np.full(n_planes, B), _measure(0.0))
    expect = (2.0 * s * s / DELTA + (M - 2) * s * s / (2.0 * DELTA)) / B ** 2
    se = G.std(ddof=1) / math.sqrt(n_planes)
    assert abs(G.mean() - expect) <= 5.0 * se, (G.mean(), expect, se)


def test_contract():
    import plane_differences as PD
    m = _measure(1.0)
    v = np.arange(-5, 6) * DELTA
    P = np.full((3, v.size), 90.0)
    G = FO.plane_gradient_energies(P, v, np.array([90.0, float("nan"), 0.0]), m)
    assert G.shape == (3,) and G[0] == 0.0 and math.isnan(G[1]) and math.isnan(G[2])
    for bad in (dict(profiles=P[:, :5], v=v, B=np.ones(3)), dict(profiles=P, v=v, B=np.ones(2)),
                dict(profiles=P, v=v[::-1], B=np.ones(3)), dict(profiles=P[:, :2], v=v[:2], B=np.ones(3)),
                dict(profiles=P, v=v, B=np.ones(3), valid=[True])):
        try:
            FO.plane_gradient_energies(cfg=m, **bad)
        except ValueError:
            continue
        raise AssertionError("plane_gradient_energies accepted %r" % (list(bad),))
    for s_um, d0 in ((0.0, 1.5), (-1.0, 1.5), (float("nan"), 1.5), (0.3, float("nan"))):
        try:
            FO.sigmoid_weight(1.0, d0, s_um)
        except ValueError:
            continue
        raise AssertionError("sigmoid_weight accepted s=%r, d0=%r" % (s_um, d0))
    for w in (-0.1, 1.1, float("nan")):
        try:
            FO.blend_scores([1.0, 2.0], [1.0, 2.0], w)
        except ValueError:
            continue
        raise AssertionError("blend_scores accepted w=%r" % w)
    for G_, H_ in (([1.0, 2.0], [1.0]), (np.ones((2, 2)), np.ones((2, 2)))):
        try:
            FO.blend_scores(G_, H_, 0.5)
        except ValueError:
            continue
        raise AssertionError("blend_scores accepted shapes")
    allnan = FO.minmax_scores([float("nan")] * 3)
    assert allnan.shape == (3,) and np.all(np.isnan(allnan))
    for d, mult in ((float("nan"), 2.0), (0.0, 2.0), (-1.0, 2.0), (1.0, 0.0), (1.0, float("nan"))):
        try:
            PD.d_line_half_um(d, mult)
        except ValueError:
            continue
        raise AssertionError("d_line_half_um accepted %r, %r" % (d, mult))
    assert PD.fixed_lines_um(()) == [] and PD.fixed_lines_um((3, "5")) == [3.0, 5.0]
    for bad in ((3.0, 5.0, 7.0, 9.0), (3.0, 3.0), (0.0,), (-1.0,), (float("inf"),), (float("nan"),)):
        try:
            PD.fixed_lines_um(bad)
        except ValueError:
            continue
        raise AssertionError("fixed_lines_um accepted %r" % (bad,))


def test_determinism():
    rng = np.random.default_rng(SEED + 3)
    m = _measure(1.0)
    v = np.arange(-15, 16) * DELTA
    P = 100.0 + 10.0 * rng.standard_normal((6, v.size))
    B = rng.uniform(90.0, 110.0, 6)
    a, b = FO.plane_gradient_energies(P, v, B, m), FO.plane_gradient_energies(P.copy(), v.copy(), B.copy(), m)
    assert np.array_equal(a, b, equal_nan=True)
    x, y = FO.blend_scores(a, B, 0.4), FO.blend_scores(a.copy(), B.copy(), 0.4)
    for key in x:
        assert np.array_equal(x[key], y[key], equal_nan=True), key


def _plane_axes(fig):
    return [ax for ax in fig.axes if ax.get_title().startswith("k ")]


def test_edge_cases():
    import matplotlib.pyplot as plt
    import run_cell
    import plane_differences as PD
    from fixtures_cell import synthetic_cell
    from allen_diameter.loading import table_io
    from allen_diameter.plotting import figures as FG
    base = default_config()
    ccfg = dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5))
    with tempfile.TemporaryDirectory() as tmp:
        swc, fetcher, planes, swc_path = synthetic_cell(tmp, ccfg, d_true=0.8, mu=1.0)
        prov = run_cell.real_provider(fetcher, planes, ccfg.acquisition.res0_um)
        # the gradient evaluation: planes +-8 about the SWC plane (the fixture has -7..7), the soma refused
        out = os.path.join(tmp, "grad")
        rec, skip = PD.run(swc, prov, ccfg, [4, 1], out, "999", planes_half=8, evaluation="gradient",
                           log=lambda m: None)
        assert "skipped" in skip and rec["ks"] == list(range(-8, 9)) and rec["n_missing"] == 2
        assert rec["k_G3"] == 0 and rec["k_G5"] == 0 and rec["k_Gd"] == 0 and rec["k_star"] == 0, \
            (rec["k_G3"], rec["k_G5"], rec["k_Gd"])
        assert rec["pick_keys"] == ["k_G3", "k_G5", "k_Gd"] and rec["lines_um"] == [3.0, 5.0]
        assert rec["d_line_half_um"] == 0.5 * 2.0 * rec["d_hat_um"] and rec["d_line_note"] is None
        assert rec["square_fetched"] is True and rec["half_um"] >= 5.0 + 2 * DELTA - 1e-9
        for key in ("G_G3", "G_G5", "G_Gd"):
            g = np.array(rec[key], dtype=float)
            assert np.isnan(g[0]) and np.isnan(g[-1]) and np.all(np.isfinite(g[1:-1])) and np.all(g[1:-1] > 0), key
        assert len(rec["v_um"]) == 2 * int(round(5.0 / DELTA)) + 1           # the longest line's profiles are kept
        st = PD.node_stack(swc, prov, ccfg, 4, None, 8, None, 0.0, 5.0, 2.0)
        ev = PD.gradient_evaluation(st, ccfg)
        dl = [x for x in ev["lines"] if x["key"] == "Gd"][0]
        assert dl["v"].size == 2 * int(round(dl["half_um"] / DELTA)) + 1
        assert np.array_equal(ev["background"], PD.profile_evaluation(st, ccfg)["background"], equal_nan=True), \
            "the background of the gradient evaluation is the profile evaluation's"
        for ln in ev["lines"]:                   # each line: its own profiles, the one background, the valid planes
            _h, v_ref, _y, _e, prof_ref = PD.line_profiles(st, ccfg, ln["half_um"])
            assert np.array_equal(ln["v"], v_ref) and np.array_equal(ln["prof"], prof_ref, equal_nan=True), ln["key"]
            assert np.array_equal(ln["G"], FO.plane_gradient_energies(prof_ref, v_ref, ev["background"], ccfg.measure,
                                                                      st["valid"]), equal_nan=True), ln["key"]
        assert [x["half_um"] for x in ev["lines"]] == [3.0, 5.0, float(st["result"].d_hat_um)]
        # a d-line shorter than 3 samples: the gradient evaluation keeps the fixed lines; the blend refuses the node
        try:
            PD._d_line_or_error(st, ccfg, 0.01)
        except ValueError:
            pass
        else:
            raise AssertionError("a d-line of fewer than 3 samples accepted")
        ev_s = PD.gradient_evaluation(st, ccfg, line_mult=0.01)
        assert [x["key"] for x in ev_s["lines"]] == ["G3", "G5"] and "fewer than 3 samples" in ev_s["d_line_note"]
        logs_s = []
        recs_s = PD.run(swc, prov, ccfg, [3, 4], os.path.join(tmp, "blend_short"), "999", planes_half=2,
                        evaluation="blend", line_mult=0.01, log=logs_s.append)
        assert all("skipped" in r for r in recs_s) and len(recs_s) == 2, recs_s
        # a long d-line widens the square (+-(7 d_hat + 2 px), about 6.2 um: past the pilot's block, 81 px high,
        # so it is fetched from the provider)
        st_long = PD.node_stack(swc, prov, ccfg, 4, None, 2, None, 0.0, None, 14.0)
        assert abs(st_long["square_half_um"] - (7.0 * float(st_long["result"].d_hat_um) + 2 * DELTA)) <= 1e-12 and \
            st_long["square_fetched"] is True, (st_long["square_half_um"], st_long["square_fetched"])
        assert os.path.exists(rec["png"]) and os.path.basename(rec["png"]) == "planegrad_4.png"
        with open(os.path.join(out, "planegrad_999.json")) as f:
            assert len(json.load(f)) == 2
        # a node without d_hat: the gradient evaluation keeps the fixed lines, the blend refuses it
        st_nan = dict(st, result=dataclasses.replace(st["result"], d_hat_um=float("nan")))
        ev_nan = PD.gradient_evaluation(st_nan, ccfg)
        assert [x["key"] for x in ev_nan["lines"]] == ["G3", "G5"] and ev_nan["d_line_note"] and \
            ev_nan["d_line_half_um"] is None
        try:
            PD.blend_evaluation(st_nan, ccfg)
        except ValueError:
            pass
        else:
            raise AssertionError("blend_evaluation accepted a node without d_hat")
        for bad in ((3.0, 5.0, 7.0, 9.0), (3.0, 3.0), (0.0,)):
            try:
                PD.gradient_evaluation(st, ccfg, lines_um=bad)
            except ValueError:
                continue
            raise AssertionError("gradient_evaluation accepted lines %r" % (bad,))
        ev4 = PD.gradient_evaluation(st, ccfg, line_mult=4.0)
        assert ev4["d_line_half_um"] == 2.0 * float(st["result"].d_hat_um)
        # no fixed line: the d-line alone; and nothing at all is refused (run: that node skipped)
        ev0 = PD.gradient_evaluation(st, ccfg, lines_um=())
        assert [x["key"] for x in ev0["lines"]] == ["Gd"] and np.array_equal(ev0["lines"][0]["G"], dl["G"],
                                                                              equal_nan=True)
        try:
            PD.gradient_evaluation(st_nan, ccfg, lines_um=())
        except ValueError:
            pass
        else:
            raise AssertionError("gradient_evaluation accepted no line at all")
        rec0, = PD.run(swc, prov, ccfg, [4], os.path.join(tmp, "grad0"), "999", planes_half=2, evaluation="gradient",
                       grad_lines_um=(), log=lambda m: None)
        assert rec0["pick_keys"] == ["k_Gd"] and rec0["k_Gd"] == 0 and rec0["lines_um"] == []
        recs_n = PD.run(swc, prov, ccfg, [4], os.path.join(tmp, "grad_none"), "999", planes_half=2,
                        evaluation="gradient", grad_lines_um=(), line_mult=0.01, log=lambda m: None)
        assert len(recs_n) == 1 and "no line to evaluate" in recs_n[0].get("skipped", ""), recs_n
        # bad fixed lines are refused before any node, before the output folder is made
        bad_dir = os.path.join(tmp, "grad_bad")
        try:
            PD.run(swc, prov, ccfg, [4], bad_dir, "999", evaluation="gradient", grad_lines_um=(3, 5, 7, 9),
                   log=lambda m: None)
        except ValueError:
            assert not os.path.exists(bad_dir)
        else:
            raise AssertionError("run accepted four fixed lines")
        # the blend evaluation on three nodes: w from d_hat, J from g and eta, the summary CSV and figure
        logs = []
        out3 = os.path.join(tmp, "blend")
        recs = PD.run(swc, prov, ccfg, [3, 4, 5], out3, "999", planes_half=4, evaluation="blend", log=logs.append)
        for r in recs:
            assert abs(r["w"] - FO.sigmoid_weight(r["d_hat_um"], 1.5, 0.3)) <= 1e-15
            J = np.array(r["J"], dtype=float)
            ref = r["w"] * np.array(r["g"], dtype=float) + (1.0 - r["w"]) * np.array(r["eta"], dtype=float)
            assert np.allclose(J, ref, rtol=0, atol=1e-15, equal_nan=True)
            G, Hs = np.array(r["G_Gd"], dtype=float), np.array(r["h_strip"], dtype=float)
            V = np.isfinite(G) & np.isfinite(Hs)       # g from G on the d-line, eta from the strip's entropy, over V
            assert np.array_equal(np.array(r["g"], dtype=float), FO.minmax_scores(G, V), equal_nan=True)
            assert np.array_equal(np.array(r["eta"], dtype=float), FO.minmax_scores(Hs, V, higher_is_better=False),
                                  equal_nan=True)
            assert r["k_blend"] == r["ks"][int(np.nanargmax(J))] and r["k_Gd"] == r["ks"][int(np.nanargmax(G))] == 0 \
                and r["k_Hd"] == r["ks"][int(np.nanargmin(Hs))], (r["k_blend"], r["k_Gd"], r["k_Hd"])
            assert r["d_line_half_um"] == r["profile_half_um"] == r["d_hat_um"]
        summ = table_io.read_rows([os.path.join(out3, "planeblend_summary_999.csv")])
        assert [r["node_id"] for r in summ] == [3, 4, 5]
        assert all(r["k_blend_minus_k_star"] == r["k_blend"] - r["k_star"] for r in summ)
        assert all(abs(r["w"] - x["w"]) <= 1e-12 for r, x in zip(summ, recs))
        assert os.path.exists(os.path.join(out3, "planeblend_summary_999.png"))
        assert any(m.startswith("[planediff] summary over 3 nodes: k_Gd within 1 plane of k*") for m in logs), logs
        # the gradient summary
        out_g3 = os.path.join(tmp, "grad3")
        recs_g = PD.run(swc, prov, ccfg, [3, 4, 5], out_g3, "999", planes_half=4, evaluation="gradient",
                        log=lambda m: None)
        summ_g = table_io.read_rows([os.path.join(out_g3, "planegrad_summary_999.csv")])
        assert len(summ_g) == 3 and all(r["k_%s_minus_k_star" % key] == r["k_" + key] - r["k_star"]
                                        for r in summ_g for key in ("G3", "G5", "Gd"))
        assert all(abs(r["d_line_half_um"] - x["d_line_half_um"]) <= 1e-12 for r, x in zip(summ_g, recs_g))
        assert os.path.exists(os.path.join(out_g3, "planegrad_summary_999.png"))
        # the per-node figure: no line on the planes, one panel per plane, the summary figure one panel per node
        r0 = recs_g[1]
        stk = np.zeros((len(r0["ks"]), 9, 9))
        variants = [dict(key=k[2:], label=k, colour=c, style=s, half_um=1.0, G=r0["G_" + k[2:]], pick=None)
                    for k, c, s in (("k_G3", "#2a78d6", "o-"), ("k_Gd", "#1baf7a", "^-"))]
        fig = FG.gradient_lines_figure([dict(label="x", stack=stk, ks=r0["ks"], valid=r0["valid"], extent=None,
                                             frames={0: ("#2a78d6", "-", "G3")}, lines=[(0, "#ff1744", "-", "k*")],
                                             variants=variants, v=r0["v_um"], prof=r0["profiles"],
                                             background=r0["background"], k_ref=0)], "t")
        pa = _plane_axes(fig)
        assert len(pa) == len(r0["ks"]) and all(len(ax.get_lines()) == 0 for ax in pa)
        plt.close(fig)
        moved = dict(recs_g[1], k_star=1)              # the title's offsets are against k*, not the SWC plane
        fig = FG.picks_summary_figure([moved] + recs_g + recs_g[:1], [dict(curve="G_G3", pick="k_G3", label="a",
                                                                         colour="#2a78d6", style="o-")], "t")
        assert sum(1 for ax in fig.axes if ax.axison) == 5 and len(fig.axes) == 8, len(fig.axes)
        assert fig.axes[0].get_title().endswith("pick minus k*: G3 -1"), fig.axes[0].get_title()
        plt.close(fig)
        # the CLI: --nodes bydiameter reaches select_nodes_by_diameter, the parameters reach run (server stubbed)
        _check_cli(PD, swc_path, tmp)


def _check_cli(PD, swc_path, tmp):
    import allen_image_io as aio
    import run_cell

    class _Fetcher:
        def __init__(self, cache_dir="", verbose=True):
            self.n_cache_hits, self.n_requests, self.bytes_downloaded = 0, 0, 0
    pilot = os.path.join(tmp, "pilot.csv")
    from allen_diameter.loading import table_io
    table_io.write_rows([dict(node_id=4, d_hat_um=0.9, fit_status="converged", z_sub_um=0.0, steep=False, flags="")],
                        pilot)
    saved = (aio.HttpFetcher, aio.list_images, aio.plane_table, run_cell.real_provider, PD.run,
             PD.select_nodes_by_diameter)
    calls, chosen = [], []
    try:
        aio.HttpFetcher, aio.list_images, aio.plane_table = _Fetcher, (lambda specimen: None), (lambda images: None)
        run_cell.real_provider = lambda *x, **k: None
        PD.run = lambda *x, **k: calls.append((x, k)) or []
        PD.select_nodes_by_diameter = lambda swc, rows, bins, per_bin, add, types: chosen.append(
            (list(bins), per_bin, list(add), [r["node_id"] for r in rows])) or [4]
        args = ["--specimen", "999", "--nodes", "bydiameter", "--pilot-csv", pilot, "--out-dir",
                os.path.join(tmp, "cli"), "--swc", swc_path, "--evaluation", "blend", "--dhat-bins", "0.7,2",
                "--per-bin", "3", "--add-nodes", "2", "--line-mult", "3", "--sigmoid-d0-um", "1.2",
                "--sigmoid-s-um", "0.4", "--grad-lines-um", "2.5,4"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            assert PD.main(args) == 0
            try:
                PD.main(["--specimen", "999", "--nodes", "bydiameter", "--out-dir", tmp, "--swc", swc_path])
            except SystemExit as e:
                assert e.code == 2
            else:
                raise AssertionError("--nodes bydiameter without --pilot-csv accepted")
    finally:
        (aio.HttpFetcher, aio.list_images, aio.plane_table, run_cell.real_provider, PD.run,
         PD.select_nodes_by_diameter) = saved
    assert chosen == [([0.7, 2.0], 3, [2], [4])], chosen
    (x, k), = calls
    got = inspect.signature(PD.run).bind(*x, **k).arguments          # by name: the call's positions may change
    assert got["node_ids"] == [4] and got["evaluation"] == "blend" and got["planes_half"] == 6 and \
        got["line_mult"] == 3.0 and got["sigmoid_d0_um"] == 1.2 and got["sigmoid_s_um"] == 0.4 and \
        got["grad_lines_um"] == [2.5, 4.0] and got["stripe_half_um"] == 1.0, got


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
