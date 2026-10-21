"""Smoke test for the kernel calibration, first stage -- Block 10 in
specs/SPEC.md (procedure Eq. 4, s.3.4; mathematics Eq. 18, s.3.5).

Checks
    test_known_answer   dip statistics of a sampled Gaussian dip: omega = s^2
                        to 1e-12 relative; with a window of 10 s, A_W =
                        a s sqrt(2 pi) and V_W = s^2 to 1e-9 relative;
                        S2.1 (D-061.4, D-067.4): calibration_set at the edges
                        of the two sets of Allen's diameter; Allen's diameter
                        selects, not d_hat; the reasons and their order;
                        every other criterion read from the pilot
    test_reference      (a) the growth fit recovers c_i, z_ax,i, the knot
                        values and the curve from exact Eq. (4) data of its
                        family, both conventions, cubic and linear, to 1e-9;
                        (b) Phase I: on rendered thin flat phantoms
                        (d = 0.25 um, alpha = 0.25, camera chain without noise,
                        the true line) the fitted growth agrees with the
                        least-squares projection, on the same design, of the
                        expected growth E_eta[sigma_r(delta + eta)^2] -
                        E_eta[sigma_r(eta)^2] (eta: the faint tube's
                        semicircle law) within 0.025 um^2 up to 3 planes, the
                        axis depths within 0.02 um, and A_W(k) / A_W(k*) stays
                        in [0.97, 1.0]
    test_convergence    the knot error on noisy Eq. (4) data falls as 1/sqrt(N)
                        (N = 10 -> 40 nodes: ratio 2 within 35 %, 20 seeds)
    test_invariants     a constant added to every omega moves only the c_i; a
                        common depth shift t of planes and starts moves z_ax
                        by t; symmetric data fitted under min_origin give a
                        symmetric two-sided curve; at half-plane knots the
                        model of linear hats has a null direction (the
                        zig-zag of period dz absorbed by the c_i), which is
                        why validate() refuses knot_step_um < dz_um
    test_contract       PlaneScan fields; kernel_from_growth round trip (a
                        table on plane-offset knots -> its growth -> exact
                        Eq. (4) data -> fit -> the same table to 1e-10, and
                        validate() passes); scan_node on a rendered phantom
                        returns a NodeResult and a PlaneScan; the scans file
                        round trip; scan_cell on a synthetic cell served
                        through fetch_zblock (three candidates, >= 2 scanned,
                        fitted); the command line synthetic -> fit; S2.1: a
                        near-field node makes two provider calls, a far-field
                        phantom one and is not scanned (FAR_FIELD_NOT_BUILT);
                        candidates_from_rows on pilot rows equals
                        calibration_reasons == []
    test_determinism    the growth fit and a rendered scan repeat bit for bit
    test_edge_cases     flat profile -> no_dip; a dip wider than the profile
                        -> no_fwhm; one usable node -> ValueError; min_origin
                        or sigma_r(0)^2 + G <= 0 -> kernel_from_growth refuses;
                        a node without a sharpest plane -> plane_scan refuses;
                        S2.1: validate() refuses overlapping, reversed, empty,
                        non-finite or mis-sized sets of Allen's diameter

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_calibration.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import json
import os
import platform
import subprocess
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
for _p in (SRC, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter.analysis import calibration as CAL  # noqa: E402
from allen_diameter.analysis import profiles  # noqa: E402
from allen_diameter.loading import calibration_io as CIO  # noqa: E402
from allen_diameter.analysis.node_pipeline import Branch, NodeResult  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402
from fixtures_cell import synthetic_cell  # noqa: E402

SEED = 20261006
REPORT_PACKAGES = ("numpy", "scipy", "Pillow")
NAN = float("nan")


def cfg_with(**cal):
    cfg = default_config()
    return dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, **cal))


def fake_node(cx, cy, z_ax, theta, d, mu, z_sub=None):
    """A NodeResult carrying the true line of a phantom (the scan needs centre, heading, k*, z_sub)."""
    z_sub = z_ax if z_sub is None else z_sub
    return NodeResult(0, 3, cx, cy, z_ax, 0.0, "", NAN, NAN, 0, z_sub, cx, cy, z_ax, theta, 0.0, False, False,
                      NAN, "block_masked", d, mu, 0.0, mu * d, "converged", (),
                      None, "gradient_energy", np.empty(0, dtype=int), np.empty(0), np.empty(0), -1)


def scan_phantom(cfg, d, mu, theta, cx, cy, z_ax, rng):
    """Render a flat phantom over planes -P..P about plane 0 and scan it along its true line."""
    tube = G.Tube((cx, cy, z_ax), 0.5 * d, 0.0, theta, 1.0, cfg.renderer.U_um, cfg.renderer.end_cut)
    t = G.axis_direction(0.0, theta)
    br = Branch.from_points(np.array([cx, cy, z_ax])[None, :] + np.arange(-6, 7)[:, None] * 1.18 * t[None, :], 0.5 * d)
    p, m, P = cfg.acquisition.res0_um, cfg.measure, cfg.calibration.offsets_planes
    left = int(math.floor((cx - m.block_half_um) / p)) - 1
    top = int(math.floor((cy - m.block_half_um) / p)) - 1
    width = int(math.ceil((cx + m.block_half_um) / p)) + 2 - left
    height = int(math.ceil((cy + m.block_half_um) / p)) + 2 - top
    blk = R.synthetic_block(tube, mu, np.arange(-P, P + 1), left, top, width, height, cfg, rng,
                            int(math.ceil(cfg.renderer.pad_um / p)))
    return CAL.plane_scan(blk, fake_node(cx, cy, z_ax, theta, d, mu), br, cfg)


def semicircle_growth(delta, r, rc, n=64):
    """E_eta[sigma_r(delta + eta)^2] - E_eta[sigma_r(eta)^2], eta with density ~ sqrt(r^2 - eta^2)
    (the absorbed mass of a faint round section by height)."""
    x, w = np.polynomial.legendre.leggauss(n)
    eta = r * x
    ww = w * np.sqrt(np.clip(1.0 - x ** 2, 0.0, None))
    ww = ww / ww.sum()

    def f(dd):
        return float(np.sum(ww * K.sigma_r(dd + eta, rc) ** 2))
    return np.array([f(dd) for dd in np.atleast_1d(delta)]) - f(0.0)


def eq4_scans(cfg, curve, n_nodes, rng, noise=0.0, z_err=0.05, c_range=(0.01, 0.03), shift=0.0):
    """Exact Eq. (4) data: omega = c_i + curve(z_ax,i - z_k) (+ noise), planes -P..P (+ shift)."""
    dz, P = cfg.acquisition.dz_um, cfg.calibration.offsets_planes
    ks = np.arange(-P, P + 1)
    scans, truth = [], []
    for i in range(n_nodes):
        z_ax = rng.uniform(-0.5 * dz, 0.5 * dz) + shift
        c = rng.uniform(*c_range)
        z = ks * dz + shift
        om = c + curve(z_ax - z) + (rng.normal(0.0, noise, ks.size) if noise > 0 else 0.0)
        nan = np.full(ks.size, np.nan)
        scans.append(CAL.PlaneScan(i, 0.25, 0.0, 0.3, 0, z_ax + rng.normal(0.0, z_err), ks, z, om, nan, nan, nan,
                                   nan, ("ok",) * ks.size))
        truth.append((z_ax, c))
    return scans, np.array([t[0] for t in truth]), np.array([t[1] for t in truth])


def pl_curve(knots, values):
    """Piecewise-linear even curve through (knots, values), linear extrapolation beyond the last knot."""
    knots, values = np.asarray(knots, float), np.asarray(values, float)

    def f(delta):
        x = np.abs(np.asarray(delta, float))
        h = knots[1] - knots[0]
        j = np.clip(np.floor(x / h).astype(int), 0, knots.size - 2)
        t = x / h - j
        return (1 - t) * values[j] + t * values[j + 1]
    return f


def cubic_curve(knots, values):
    """Even not-a-knot cubic spline through (+-knots, values) (scipy CubicSpline, built here
    independently of the module's basis), continued linearly beyond the last knot."""
    from scipy.interpolate import CubicSpline
    knots, values = np.asarray(knots, float), np.asarray(values, float)
    sp = CubicSpline(np.r_[-knots[:0:-1], knots], np.r_[values[:0:-1], values], bc_type="not-a-knot")
    edge = knots[-1]

    def f(delta):
        dd = np.asarray(delta, float)
        x = np.clip(dd, -edge, edge)
        return sp(x) + sp(x, 1) * (dd - x)
    return f


def family_curve(cfg, knots, values):
    """The curve family the configuration fits (growth_interp)."""
    return (cubic_curve if cfg.calibration.growth_interp == "cubic" else pl_curve)(knots, values)


def fit_knots(cfg):
    """The knots fit_growth uses for scans over planes -P..P: m h, m = 0..ceil((P + 1/2) dz / h)
    (a cubic spline depends on its knot range through the not-a-knot ends, so exact data must
    be generated on the same knots)."""
    cal, dz = cfg.calibration, cfg.acquisition.dz_um
    M = int(math.ceil((cal.offsets_planes + 0.5) * dz / cal.knot_step_um - 1e-9))
    return np.arange(0, M + 1) * cal.knot_step_um


def table_curve(cfg):
    """The growth of the default kernel table at the fit's knots, joined in the configured family."""
    knots = fit_knots(cfg)
    return knots, family_curve(cfg, knots, K.sigma_r(knots, cfg.renderer) ** 2 - cfg.renderer.sigma_r0_um ** 2)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = default_config()
    m = cfg.measure
    cal = dataclasses.replace(cfg.calibration, window_half_um=2.9)
    v = profiles.profile_offsets(m)
    for s, vc in ((0.15, 0.037), (0.25, -0.051), (0.28, 0.11)):
        a, B = 30.0, 200.0
        st = CAL.dip_statistics(v, B - a * np.exp(-0.5 * ((v - vc) / s) ** 2), B, m, cal)
        assert st["status"] == "ok", st
        assert abs(st["omega_um2"] / s ** 2 - 1) <= 1e-12, (s, st["omega_um2"])
        # a Riemann sum of a Gaussian at step h is exact to ~exp(-2 pi^2 s^2 / h^2) (< 1e-9 for s >= 0.15,
        # h = 0.1144) and the window |v - v_c| <= 2.9 holds >= 10 s
        assert abs(st["area_gl_um"] / (a * s * math.sqrt(2 * math.pi)) - 1) <= 1e-9, (s, st["area_gl_um"])
        assert abs(st["vw_um2"] / s ** 2 - 1) <= 1e-9, (s, st["vw_um2"])
    # S2.1 (D-061.4, D-067.4): the two sets of Allen's diameter at their edges, the defaults 0.6 and (0.6, 0.8] um;
    # 2 x 0.3 and 2 x 0.4 are exactly 0.6 and 0.8 in binary (doubling is exact), so radii written with one
    # decimal fall on the side the decision names
    eps = 1e-9
    for d_a, want in ((0.05, "near"), (0.6, "near"), (2 * 0.3, "near"), (0.6 + eps, "far"), (0.7, "far"),
                      (0.8, "far"), (2 * 0.4, "far"), (0.8 + eps, ""), (5.0, ""), (0.0, ""), (-0.1, ""), (NAN, ""),
                      (float("inf"), "")):
        assert CAL.calibration_set(d_a, cfg) == want, (d_a, CAL.calibration_set(d_a, cfg), want)
        assert want == "" or want in CAL.CALIBRATION_SETS
    assert CAL.CALIBRATION_SETS == ("near", "far")
    # Allen's diameter selects, not the fit's d_hat; the order of the reasons
    base = fake_node(0.0, 0.0, 0.0, 0.0, 2.0, 0.1)                       # d_hat 2 um, alpha 0.2, flat, converged
    thin_fit = fake_node(0.0, 0.0, 0.0, 0.0, 0.1, 2.0)                   # d_hat 0.1 um, alpha 0.2
    assert CAL.calibration_reasons(base, cfg, 0.4) == [] and CAL.calibration_reasons(thin_fit, cfg, 0.7) == []
    assert CAL.calibration_reasons(thin_fit, cfg, 1.0) == ["allen_diameter_out"]
    for bad in (NAN, 0.0, -1.0, float("inf")):
        assert CAL.calibration_reasons(thin_fit, cfg, bad) == ["allen_diameter_missing"], bad
    worst = dataclasses.replace(base, fit_status="at_bound", reg_verdict="FAR: nearest matching ridge 2 um",
                                flags=("dark", "crossing", "steep"), phi_rad=0.5, alpha_hat=0.9)
    want = ["status:at_bound", "registration:FAR", "crossing", "dark", "allen_diameter_out", "not_flat", "not_faint"]
    assert CAL.calibration_reasons(worst, cfg, 1.0) == want, CAL.calibration_reasons(worst, cfg, 1.0)
    assert CAL.criteria_reasons(0.5, 0.9, 1.0, cfg) == want[4:]
    assert CAL.criteria_reasons(NAN, NAN, 0.3, cfg) == ["not_flat", "not_faint"]
    # every criterion but Allen's diameter reads the pilot (pilot_view): a deliverable failing everything with a
    # passing pilot passes; a pilot failing one criterion gives that reason whatever the deliverable
    passing_pilot = dict(pilot_status="converged", pilot_flags=(), d_pilot_um=0.3, mu_pilot_per_um=1.0,
                         v0_pilot_um=0.0, alpha_pilot=0.3, B_bar_pilot=200.0)
    bad_fit = dataclasses.replace(base, fit_status="failed", flags=("crossing", "dark"), alpha_hat=0.9,
                                  **passing_pilot)
    assert CAL.calibration_reasons(bad_fit, cfg, 0.4) == [], CAL.calibration_reasons(bad_fit, cfg, 0.4)
    for change, reason in ((dict(alpha_pilot=0.9), "not_faint"), (dict(pilot_status="at_bound"), "status:at_bound"),
                           (dict(pilot_flags=("stack_edge",)), "stack_edge"), (dict(alpha_pilot=NAN), "not_faint")):
        node = dataclasses.replace(base, **dict(passing_pilot, **change))
        assert CAL.calibration_reasons(node, cfg, 0.4) == [reason], (change, CAL.calibration_reasons(node, cfg, 0.4))


def test_reference():
    # (a) exact Eq. (4) data from the fitted family on plane-offset knots: recovered to 1e-9
    for conv in ("symmetric", "min_origin"):
        for interp in ("cubic", "linear"):
            cfg = cfg_with(origin_convention=conv, growth_interp=interp)
            knots, curve = table_curve(cfg)
            scans, z_true, c_true = eq4_scans(cfg, curve, 10, np.random.default_rng(SEED))
            g = CAL.fit_growth(scans, cfg)
            err = np.max(np.abs(g.g_um2 - curve(g.knots_um)))
            assert g.success and err <= 1e-9, (conv, interp, g.g_um2 - curve(g.knots_um))
            assert np.max(np.abs(g.z_ax_um - z_true)) <= 1e-9 and np.max(np.abs(g.c_um2 - c_true)) <= 1e-9, (conv, interp)
            dd = np.linspace(-1.0, 1.0, 41)
            assert np.max(np.abs(g.growth(dd) - curve(dd))) <= 1e-9, (conv, interp)
    # (b) Phase I: rendered thin flat phantoms, noise-free camera chain, the true line.
    # Measured 2026-10-06 (8 phantoms, cubic): G - projection = -0.010 / -0.012 / -0.003 um^2 at
    # 1 / 2 / 3 planes (the core width of a mixture of defocus values is narrower than its mean
    # variance, most where sigma_r grows fastest), axis depths within 0.016 um, area ratios
    # 0.976-0.998 (the +-1.5 um window loses the tails of the 3-plane spot)
    base = default_config()
    cfg = dataclasses.replace(base, renderer=dataclasses.replace(base.renderer, noise_sd_gl=0.0, jpeg=False))
    rng = np.random.default_rng(SEED)
    d, mu = 0.25, 1.0
    scans, z_true = [], []
    for i in range(8):
        theta = rng.uniform(0.0, math.pi)
        cx, cy = rng.uniform(-0.057, 0.057, 2)
        z_ax = rng.uniform(-0.14, 0.14)
        sc = scan_phantom(cfg, d, mu, theta, cx, cy, z_ax, rng)
        assert all(s == "ok" for s in sc.status), sc.status
        scans.append(dataclasses.replace(sc, node_id=i, z_sub_um=z_ax + rng.normal(0.0, 0.03)))
        z_true.append(z_ax)
    g = CAL.fit_growth(scans, cfg)
    ideal = [dataclasses.replace(sc, omega_um2=0.02 + semicircle_growth(z - sc.z_um, 0.5 * d, cfg.renderer),
                                 z_sub_um=z) for sc, z in zip(scans, z_true)]
    gp = CAL.fit_growth(ideal, cfg)
    near = g.knots_um <= 3 * cfg.acquisition.dz_um + 1e-9
    err = np.abs(g.g_um2 - gp.g_um2)[near]
    assert np.max(err) <= 0.025, (g.g_um2, gp.g_um2)
    assert np.max(np.abs(g.z_ax_um - np.array(z_true))) <= 0.02, g.z_ax_um - np.array(z_true)
    ratios = CAL.area_ratio_by_offset(scans)
    assert all(0.97 <= r <= 1.0 + 1e-12 for r in ratios.values()), ratios


def test_convergence():
    cfg = default_config()
    knots, curve = table_curve(cfg)
    rms = {}
    for n in (10, 40):
        errs = []
        for seed in range(20):
            scans, _, _ = eq4_scans(cfg, curve, n, np.random.default_rng([SEED, n, seed]), noise=2e-3)
            g = CAL.fit_growth(scans, cfg)
            errs.append(g.g_um2[1:4] - curve(g.knots_um[1:4]))
        rms[n] = float(np.sqrt(np.mean(np.square(errs))))
    # four times the nodes: the error halves; 20 seeds x 3 knots give the RMS to about 15 %
    assert abs(rms[10] / rms[40] / 2.0 - 1) <= 0.35, rms


def test_invariants():
    cfg = default_config()
    knots, curve = table_curve(cfg)
    scans, _, _ = eq4_scans(cfg, curve, 8, np.random.default_rng(SEED), noise=1e-3)
    g0 = CAL.fit_growth(scans, cfg)
    up = [dataclasses.replace(sc, omega_um2=sc.omega_um2 + 0.05) for sc in scans]
    g1 = CAL.fit_growth(up, cfg)
    assert np.max(np.abs(g1.g_um2 - g0.g_um2)) <= 1e-9 and np.max(np.abs(g1.z_ax_um - g0.z_ax_um)) <= 1e-9
    assert np.max(np.abs(g1.c_um2 - g0.c_um2 - 0.05)) <= 1e-9
    t = 3.7
    moved = [dataclasses.replace(sc, z_um=sc.z_um + t, z_sub_um=sc.z_sub_um + t) for sc in scans]
    g2 = CAL.fit_growth(moved, cfg)
    assert np.max(np.abs(g2.z_ax_um - g0.z_ax_um - t)) <= 1e-8 and np.max(np.abs(g2.g_um2 - g0.g_um2)) <= 1e-8
    # min_origin on symmetric data: a symmetric two-sided curve (the convention pins the shift at 0)
    cmo = cfg_with(origin_convention="min_origin")
    exact, _, _ = eq4_scans(cmo, curve, 10, np.random.default_rng(SEED + 1))
    gm = CAL.fit_growth(exact, cmo)
    assert np.max(np.abs(gm.g_um2 - gm.g_um2[::-1])) <= 1e-10, gm.g_um2
    # half-plane knots (linear hats): G + Z with Z the zig-zag (0 at even, A at odd multiples of
    # dz / 2) and c_i - A |z_ax,i| / (dz / 2) fit exactly the same data when every |z_ax,i| <= dz / 2
    dz = cfg.acquisition.dz_um
    h = 0.5 * dz
    zig = pl_curve(np.arange(0, 12) * h, 0.01 * (np.arange(0, 12) % 2))
    for sc in exact:
        z_ax = sc.z_sub_um
        delta = z_ax - sc.z_um
        if abs(z_ax) <= h:
            assert np.max(np.abs(zig(delta) - 0.01 * abs(z_ax) / h)) <= 1e-12
    try:
        cfg_with(knot_step_um=h).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("knot_step_um < dz_um must be refused")


def test_contract():
    cfg = default_config()
    # kernel_from_growth round trip on plane-offset knots
    knots = fit_knots(cfg)
    sig = np.array([0.08, 0.13, 0.40, 0.61, 0.83])
    assert knots.size == sig.size, knots
    curve = family_curve(cfg, knots, sig ** 2 - sig[0] ** 2)
    scans, _, _ = eq4_scans(cfg, curve, 10, np.random.default_rng(SEED))
    g = CAL.fit_growth(scans, cfg)
    rc = CAL.kernel_from_growth(g, cfg.renderer, sigma_r0_um=0.08)
    n = min(len(rc.kernel_table_sigma_um), sig.size)
    assert np.max(np.abs(np.array(rc.kernel_table_sigma_um[:n]) - sig[:n])) <= 1e-10, rc.kernel_table_sigma_um
    assert rc.kernel_table_delta_um[0] == 0.0 and rc.sigma_r0_um == 0.08
    dataclasses.replace(cfg, renderer=rc).validate()
    # PlaneScan fields from a rendered phantom, and scan_node end to end
    rng = np.random.default_rng(SEED)
    sc = scan_phantom(cfg, 0.25, 1.0, 0.4, 0.01, -0.02, 0.05, rng)
    P = cfg.calibration.offsets_planes
    assert sc.ks.tolist() == list(range(-P, P + 1)) and np.allclose(sc.z_um, sc.ks * cfg.acquisition.dz_um)
    for name in ("omega_um2", "vw_um2", "area_gl_um", "depth_gl", "B_bar"):
        assert getattr(sc, name).shape == (2 * P + 1,), name
    assert len(sc.status) == 2 * P + 1
    d, mu, theta = 0.2, 1.5, 0.7
    tube = G.Tube((0.02, -0.03, 0.04), 0.5 * d, 0.0, theta, 1.0, cfg.renderer.U_um, cfg.renderer.end_cut)
    t = G.axis_direction(0.0, theta)
    br = Branch.from_points(np.array([0.02, -0.03, 0.04])[None, :] + np.arange(-6, 7)[:, None] * 1.18 * t[None, :],
                            0.5 * d)
    pad = int(math.ceil(cfg.renderer.pad_um / cfg.acquisition.res0_um))

    calls = []

    def provider(left, top, width, height, k_lo, k_hi):
        calls.append((k_lo, k_hi))
        return R.synthetic_block(tube, mu, np.arange(k_lo, k_hi + 1), left, top, width, height, cfg, rng, pad)
    res, scan, reasons = CAL.scan_node(br, 6, provider, cfg)
    assert isinstance(res, NodeResult)
    assert (scan is None) == bool(reasons), reasons
    assert scan is not None, ("a thin faint flat phantom must be a calibration node", reasons, res.d_hat_um,
                              res.alpha_hat)
    assert scan.k_star == res.k_star and np.sum(np.isfinite(scan.omega_um2)) >= 5, scan.status
    assert len(calls) == 2 and calls[1] == (res.k_star - P, res.k_star + P), calls      # the pilot's, then the scan's
    # S2.1: a far-field phantom (Allen's diameter = its branch's 2 r = d = 0.7 um, alpha 0.35, flat) is selected and
    # not scanned: the pilot's one provider call only, no scan, the one reason FAR_FIELD_NOT_BUILT
    d_far, mu_far = 0.7, 0.5
    tube_far = G.Tube((0.02, -0.03, 0.04), 0.5 * d_far, 0.0, theta, 1.0, cfg.renderer.U_um, cfg.renderer.end_cut)
    br_far = Branch.from_points(np.array([0.02, -0.03, 0.04])[None, :] + np.arange(-6, 7)[:, None] * 1.18 * t[None, :],
                                0.5 * d_far)
    calls_far = []

    def provider_far(left, top, width, height, k_lo, k_hi):
        calls_far.append((k_lo, k_hi))
        return R.synthetic_block(tube_far, mu_far, np.arange(k_lo, k_hi + 1), left, top, width, height, cfg, rng, pad)
    res_far, scan_far, reasons_far = CAL.scan_node(br_far, 6, provider_far, cfg)
    assert CAL.calibration_set(2.0 * br_far.radius_um[6], cfg) == "far"
    assert CAL.calibration_reasons(res_far, cfg, 2.0 * br_far.radius_um[6]) == [], \
        ("the far-field phantom must pass S, flat and faint", res_far.fit_status, res_far.flags, res_far.alpha_hat)
    assert scan_far is None and reasons_far == [CAL.FAR_FIELD_NOT_BUILT] and len(calls_far) == 1, (reasons_far,
                                                                                                    calls_far)
    assert CAL.scan_verdict(scan_far, reasons_far).startswith("far-field calibration node, not scanned")
    assert CAL.scan_verdict(scan, []) == "scanned (near field)"
    assert CAL.scan_verdict(None, ["not_flat"]) == "not a calibration node (not_flat)"
    # the CSV readers' rule (calibrate_kernel.candidates_from_rows on survey.pilot_row rows) holds a node exactly
    # when calibration_reasons is empty -- pilots and deliverables, both sets, every criterion (no registration
    # verdict: the CSV rule leaves the registration to the re-measurement)
    sys.path.insert(0, str(WS / "scripts"))
    import calibrate_kernel
    from allen_diameter.analysis import survey
    base = fake_node(0.0, 0.0, 0.0, 0.0, 2.0, 0.1)
    pil = dict(pilot_status="converged", pilot_flags=(), d_pilot_um=0.3, mu_pilot_per_um=1.0, v0_pilot_um=0.0,
               alpha_pilot=0.3, B_bar_pilot=200.0)
    cases = [(dict(), 0.2), (dict(), 0.35), (dict(), 0.5), (dict(), NAN), (dict(phi_rad=0.5), 0.2),
             (dict(alpha_hat=0.9), 0.2), (dict(fit_status="at_bound"), 0.2), (dict(flags=("crossing",)), 0.2),
             (dict(pil), 0.2), (dict(pil, alpha_pilot=0.9), 0.2), (dict(pil, pilot_status="failed"), 0.35),
             (dict(pil, pilot_flags=("dark",)), 0.2), (dict(pil, alpha_hat=0.9, fit_status="failed"), 0.35),
             (dict(pil, flags=("bbar_few",)), 0.2)]
    rows, want = [], set()
    for n, (change, r_a) in enumerate(cases):
        node = dataclasses.replace(base, node_id=100 + n, **change)
        rows.append(survey.pilot_row(node, r_a, cfg))
        if not CAL.calibration_reasons(node, cfg, 2.0 * r_a):
            want.add(100 + n)
    got = calibrate_kernel.candidates_from_rows(rows, cfg)
    assert got == want and want == {100, 101, 108, 112, 113}, (sorted(got), sorted(want))
    assert [r["calibration_set"] for r in rows[:4]] == ["near", "far", "", ""] and rows[2]["calibration_node"] is False
    assert rows[4]["calibration_set"] == "" and rows[4]["calibration_node"] is False      # Allen near, not flat
    # the scans file round trip (NaN <-> null)
    back = CIO.scan_from_dict(json.loads(json.dumps(CIO.scan_to_dict(scan))))
    for name in ("ks", "z_um", "omega_um2", "vw_um2", "area_gl_um", "depth_gl", "B_bar"):
        assert np.array_equal(getattr(back, name), getattr(scan, name), equal_nan=True), name
    assert back.status == scan.status and back.node_id == scan.node_id and back.k_star == scan.k_star
    # the Phase II path in miniature: a cell served through fetch_zblock, three candidate nodes
    sys.path.insert(0, str(WS / "scripts"))
    import run_cell
    with tempfile.TemporaryDirectory() as tmp:
        cfg_cell = dataclasses.replace(cfg, measure=dataclasses.replace(cfg.measure, block_half_um=3.5))
        swc, fetcher, planes, _ = synthetic_cell(tmp, cfg_cell, d_true=0.2, mu=1.5, allen_radius=0.1)
        provider = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
        out = CAL.scan_cell(swc, provider, cfg_cell, candidates={4, 5, 6})
        assert [o[0] for o in out] == [4, 5, 6], [o[0] for o in out]
        scanned = [o[2] for o in out if o[2] is not None]
        assert len(scanned) >= 2, [o[3] for o in out]
        g = CAL.fit_growth(scanned, cfg_cell)
        assert g.success and g.node_ids.tolist() == [s_.node_id for s_ in scanned]
        # the command line: two synthetic calibration phantoms, then the fit
        sp = os.path.join(tmp, "scans.json")
        cp = os.path.join(tmp, "calibration.json")
        run = [sys.executable, str(WS / "scripts" / "calibrate_kernel.py")]
        p1 = subprocess.run(run + ["synthetic", "--n", "2", "--seed", "5", "--out", sp], capture_output=True, text=True,
                            timeout=900)
        assert p1.returncode == 0 and "2 of 2 phantoms scanned" in p1.stdout, (p1.stdout, p1.stderr)
        p2 = subprocess.run(run + ["fit", "--scans", sp, "--out", cp], capture_output=True, text=True, timeout=300)
        assert p2.returncode == 0 and "growth fit (gaussian_core_width2, symmetric, cubic): 2 nodes" in p2.stdout, \
            (p2.stdout, p2.stderr)
        with open(cp) as f:
            obj = json.load(f)
        assert obj["growth"]["knots_um"][0] == 0.0 and ("kernel" in obj) != ("kernel_error" in obj), sorted(obj)


def test_determinism():
    cfg = default_config()
    knots, curve = table_curve(cfg)
    scans, _, _ = eq4_scans(cfg, curve, 8, np.random.default_rng(SEED), noise=1e-3)
    a, b = CAL.fit_growth(scans, cfg), CAL.fit_growth(scans, cfg)
    assert np.array_equal(a.g_um2, b.g_um2) and np.array_equal(a.z_ax_um, b.z_ax_um)
    s1 = scan_phantom(cfg, 0.25, 1.0, 0.4, 0.01, -0.02, 0.05, np.random.default_rng(SEED))
    s2 = scan_phantom(cfg, 0.25, 1.0, 0.4, 0.01, -0.02, 0.05, np.random.default_rng(SEED))
    assert np.array_equal(s1.omega_um2, s2.omega_um2, equal_nan=True) and s1.status == s2.status


def test_edge_cases():
    cfg = default_config()
    m, cal = cfg.measure, cfg.calibration
    v = profiles.profile_offsets(m)
    assert CAL.dip_statistics(v, np.full(v.size, 200.0), 200.0, m, cal)["status"] == "no_dip"
    wide = CAL.dip_statistics(v, 200.0 - 20.0 * np.exp(-0.5 * (v / 3.0) ** 2), 200.0, m, cal)
    assert wide["status"] == "no_fwhm" and math.isnan(wide["omega_um2"]), wide
    knots, curve = table_curve(cfg)
    scans, _, _ = eq4_scans(cfg, curve, 3, np.random.default_rng(SEED))
    sparse = [dataclasses.replace(sc, omega_um2=np.where(np.arange(sc.ks.size) < 2, sc.omega_um2, np.nan))
              for sc in scans[1:]]
    for bad in ([scans[0]], [scans[0]] + sparse):
        try:
            CAL.fit_growth(bad, cfg)
        except ValueError:
            continue
        raise AssertionError("one usable node must be refused")
    gm = CAL.fit_growth(eq4_scans(cfg_with(origin_convention="min_origin"), curve, 6,
                                  np.random.default_rng(SEED))[0], cfg_with(origin_convention="min_origin"))
    g = CAL.fit_growth(scans, cfg)
    for call in (lambda: CAL.kernel_from_growth(gm, cfg.renderer),
                 lambda: CAL.kernel_from_growth(dataclasses.replace(g, g_um2=g.g_um2 - 1.0), cfg.renderer)):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("kernel_from_growth accepted an invalid growth")
    # S2.1: validate() keeps the two sets of Allen's diameter disjoint, finite and of two band values
    for bad in (dict(node_allen_diameter_max_um=0.7), dict(far_field_allen_diameter_um=(0.8, 0.6)),
                dict(far_field_allen_diameter_um=(0.6, 0.6)), dict(far_field_allen_diameter_um=(0.6,)),
                dict(far_field_allen_diameter_um=(0.6, 0.7, 0.8)), dict(node_allen_diameter_max_um=0.0),
                dict(node_allen_diameter_max_um=-0.1), dict(far_field_allen_diameter_um=(0.6, NAN)),
                dict(node_allen_diameter_max_um=NAN), dict(far_field_allen_diameter_um=(0.6, float("inf")))):
        try:
            cfg_with(**bad).validate()
        except ValueError:
            continue
        raise AssertionError("validate() accepted %r" % (bad,))
    cfg_with(node_allen_diameter_max_um=0.5).validate()                  # a gap between the sets is allowed
    gap = cfg_with(node_allen_diameter_max_um=0.5)
    assert [CAL.calibration_set(x, gap) for x in (0.5, 0.55, 0.6, 0.6 + 1e-9)] == ["near", "", "", "far"]
    node = fake_node(0.0, 0.0, NAN, 0.0, 0.25, 1.0)
    try:
        CAL.plane_scan((np.zeros((1, 4, 4)), np.array([0]), np.array([True]), None), node, None, cfg)
    except ValueError:
        return
    raise AssertionError("a node without a sharpest plane must be refused")


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
