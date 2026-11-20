"""Smoke test for the in-focus width fitted on thick nodes (analysis/sigma_calibration.py) -- Block 42 in specs/SPEC.md (D-201).

Checks (T1 to T8 are the entry's oracles)
    test_known_answer         T1: vp_background on I = (b0 + b1 v) tau exactly gives b [1e-10 relative] and resid 0
                              [1e-9 gl]; T7: summarize against values computed by hand (four nodes, one not ok;
                              one ok; none ok) [1e-12]; start_theta on a Gaussian dip: d0 and mu0 from the
                              smoothed Gaussian's closed-form half-depth width and depth [2e-3 relative], v0 and
                              z_ax = pilot_start's [exact], the k* plane chosen by k_index
    test_reference            T1: on random I, b and resid equal numpy.linalg.lstsq on [tau, v tau] [1e-12]; NaN
                              samples dropped (lstsq on the finite subset), resid NaN exactly where I or tau is;
                              T2: forward_with_kernel at (sigma_r(0), 1) equals node_inputs.forward_factory with the
                              unchanged configuration [exact], under FAST and under the hybrid (4 x 8 directions)
    test_convergence          not a discretisation: skipped
    test_noise_free_recovery  T3: profiles (150 - 2 v) tau(theta_true) on the synthetic node (5 planes), theta_true =
                              (1.5 um, 0.25 /um, v0_start + 0.04 um, z_start + 0.1 um, 0.15 um, 1.3), fitted from
                              start_theta: |d sigma0| <= 0.005 um, |d d| / d <= 1 %, |d c| / c <= 3 %; converged
    test_noisy_recovery       T4: Gaussian noise 1.1 gl per sample, 2 seeds: |theta_hat - theta_true| <= 3 SE_s for
                              d, sigma0, c; converged
    test_invariants           the SigmaFit's relations on T4's first fit: model_gl = (b0 + b1 v) forward6(theta_hat)
                              [1e-12], b = vp_background's at theta_hat [exact], chi2 and chi2_per_plane from the
                              residuals [1e-10], se_scaled = se sqrt(chi2 / dof) [1e-12], dof = n_res - 16, theta
                              inside its bounds; vp_background linear in I [1e-10]
    test_contract             shapes, dtypes and types of SigmaFit, select_planes, vp_background, forward_with_kernel
                              and summarize; PARAMS and STATUSES
    test_determinism          T5: two fits of one input give the same SigmaFit [exact] (seconds aside)
    test_start_background     T8 (amendment 4, D-203): start_theta is blind to a straight background -- on
                              (150 + b1 v) tau(theta_true) at d 1.5 um, noise-free, d0 and mu0 with b1 = +6 and
                              -6 gl/um within 5 % of those with b1 = 0; a profile with no measurable width (flat, a
                              bump, a NaN sample) gives the pilot's d (the bump and the NaN its mu too: D <= 0 or not
                              finite), without an exception; the line through the ends equals numpy.polyfit's
                              [1e-10]; with fewer than 3 samples at an end L = B_gl of k* (the start then equals the
                              rule computed by hand [exact])
    test_edge_cases           T6: a forward returning NaN gives status "failed" without an exception; a forward NaN
                              only beyond c's upper bound (the truth within h_c of it) gives a backward Jacobian
                              column (the forward never called beyond the bound) and a converged fit; a forward
                              NaN beyond c = 1.31 inside the bounds gives backward columns there too and a converged
                              fit; fewer than 3 planes is a ValueError; a start outside the bounds is clipped
                              (start_theta and fit_sigma_node); plus forward_with_kernel's NaN on a kernel
                              with_kernel refuses and on ExtrusionInvalid, a start that is not finite, and the
                              ValueErrors of every function

T3 to T6 and T8 test the solver and the start, not the forward's fidelity: they run under the cheap configuration
FAST of test_smoke_pilot_run.py (absorption "partition_vertical", 4 x 8 directions), as the entry allows, on the node of
its test_noise_free_fit (a rendered single flat tube, measure_pilot, node_inputs.build_node), prepared as notebook
Cell 23A prepares the thick nodes (D-201.6): the planes k* +- 2 (cleaning.plane_half_range 2) and the line
+-4.5 um (measure.profile_half_um). The tube is 1.5 um wide, mu 0.25 /um, traced at Allen's diameter 1.5 um. The
profiles are then replaced by the forward's own output at theta_true. The production hybrid's numbers are the
identifiability check's (checks/sigma_calibration_2026-11-20/identifiability.py).

Run
    cd "Passive Features/Diameter Re-measurement"
    python3 tests/smoke/test_smoke_sigma_calibration.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO. About 1 to 2 minutes on one core.
    python -m pytest tests/smoke/test_smoke_sigma_calibration.py -q
        The same checks through pytest.

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import os
import platform
import re
import sys
import tempfile
import time
import traceback
import types
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_smoke_pilot_run as PR  # noqa: E402  (FAST, fast_cfg, write_swc, block_provider, node_of, fscene_of)
from allen_diameter.analysis import cell, clearness  # noqa: E402
from allen_diameter.analysis import node_inputs as NI  # noqa: E402
from allen_diameter.analysis import sigma_calibration as SC  # noqa: E402
from allen_diameter.analysis.node_pipeline import measure_pilot  # noqa: E402
from allen_diameter.model import geometry, line_fast, render  # noqa: E402

SEED = 20261120
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run
_CACHE = {}

D_TRUE, MU_TRUE = 1.5, 0.25          # the rendered tube (um, 1/um) and theta_true's d and mu (T3)
B0, B1 = 150.0, -2.0                 # T3's straight background (gl, gl/um)
NOISE_GL = 1.1                       # T4's noise SD per sample (gl)


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure.

    On failure both arrays are saved to ARTIFACT_DIR/<name>.npz before the
    AssertionError propagates. Justify rtol and atol at the call site
    (float64 roundoff, discretisation-error estimate, statistical spread).
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / f"{safe_name}.npz"
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def thick_cfg():
    """FAST of test_smoke_pilot_run.py over the defaults, with Cell 23A's thick-node preparation (D-201.6): the
    planes k* +- 2 and the measuring line +-4.5 um."""
    cfg = PR.fast_cfg()
    cfg = dataclasses.replace(cfg, cleaning=dataclasses.replace(cfg.cleaning, plane_half_range=2),
                              measure=dataclasses.replace(cfg.measure, profile_half_um=4.5))
    cfg.validate()
    return cfg


def node_case():
    """(cfg, pilot, node, info, forward6, k_index) of the synthetic thick node, cached: a noise-free rendering of
    one flat tube along x (d 1.5 um, mu 0.25 /um, axis at z 0.05 um; Block 4, no camera chain, 210 tau) traced
    at Allen's diameter 1.5 um, its pilot (measure_pilot), its NodeData on the 5 planes k* +- 2 (build_node) and
    forward_with_kernel under FAST, as test_smoke_pilot_run.test_noise_free_fit builds its node."""
    if "node" in _CACHE:
        return _CACHE["node"]
    cfg = thick_cfg()
    res0, dz = cfg.acquisition.res0_um, cfg.acquisition.dz_um
    tmp = tempfile.mkdtemp(prefix="b42_node_")
    swc = PR.write_swc(os.path.join(tmp, "one.swc"), [((0.0, 0.0, 0.05), (20.0, 0.0, 0.05), 11, 0.5 * D_TRUE)])
    tube = geometry.Tube((10.0, 0.0, 0.05), 0.5 * D_TRUE, 0.0, 0.0, 1.0, 14.0, "axial")
    left, top = int(math.floor(-0.5 / res0)), int(math.floor(-10.5 / res0))
    width = height = int(math.ceil(21.0 / res0))
    ks = np.arange(-6, 7)
    f = render.fine_factor(D_TRUE, cfg.renderer, res0)
    grid, inner = render.block_fine_grid(left, top, width, height, res0, f, 8)
    out = render.render_transmittance(tube, MU_TRUE, ks * dz, grid, cfg.renderer, +1,
                                      reduce=lambda pl: render.pixel_integrate(pl[inner[0], inner[1]], f))
    block = cfg.renderer.background_B_gl * np.asarray(out.tau, dtype=float)
    provider = PR.block_provider(block, ks, left, top, res0)
    full, i_full, ob, t = PR.node_of(swc, cfg, 7)
    pilot = measure_pilot(full, i_full, provider, cfg)
    scene = cell.cell_scene(swc, (2, 3, 4), res0)
    node, info = NI.build_node(ob, t, pilot, provider, scene, cfg, NI.own_ids_within(ob, t, 5.0)[0],
                               PR.fscene_of(swc, scene, d=D_TRUE))
    assert node is not None, info
    assert len(node.ks) == 5, node.ks
    f6 = SC.forward_with_kernel(pilot, info["U_axis_um"], node, cfg, line_fast.SectionCache())
    k_index = int(np.flatnonzero(np.asarray(node.ks) == int(pilot.k_star))[0])
    _CACHE["node"] = (cfg, pilot, node, info, f6, k_index)
    return _CACHE["node"]


def theta_true():
    """T3's theta_true and its start's (v0, z_ax): (1.5, 0.25, v0_start + 0.04, z_start + 0.1, 0.15, 1.3)."""
    cfg, pilot, node, info, f6, k_index = node_case()
    st = NI.pilot_start(pilot, node.node_xyz)
    v0s, zs = float(st[2]), float(node.node_xyz[2]) + float(st[3])
    return np.array([D_TRUE, MU_TRUE, v0s + 0.04, zs + 0.1, 0.15, 1.3]), v0s, zs


def clean_profiles():
    """(I, v): (b0 + b1 v) tau(theta_true) (gl) on the node's planes and line, the forward's own output."""
    if "clean" in _CACHE:
        return _CACHE["clean"]
    cfg, pilot, node, info, f6, k_index = node_case()
    th, _, _ = theta_true()
    v = np.asarray(node.v, dtype=float)
    I = (B0 + B1 * v)[None, :] * f6(*th)
    _CACHE["clean"] = (I, v)
    return _CACHE["clean"]


def start_of(I, v):
    cfg, pilot, node, info, f6, k_index = node_case()
    return SC.start_theta(I, np.full(len(node.ks), B0), v, k_index, pilot, node.node_xyz, cfg.sigma_calibration)


def t3_fit():
    if "t3" not in _CACHE:
        cfg, pilot, node, info, f6, k_index = node_case()
        I, v = clean_profiles()
        th0 = start_of(I, v)
        _CACHE["t3"] = (th0, SC.fit_sigma_node(I, NOISE_GL, v, f6, th0, cfg.sigma_calibration))
    return _CACHE["t3"]


def noisy_profiles(seed):
    I, v = clean_profiles()
    return I + make_rng(seed).normal(0.0, NOISE_GL, I.shape), v


def t4_fits():
    if "t4" not in _CACHE:
        cfg, pilot, node, info, f6, k_index = node_case()
        out = []
        for seed in (SEED, SEED + 1):
            In, v = noisy_profiles(seed)
            th0 = start_of(In, v)
            out.append((seed, In, th0, SC.fit_sigma_node(In, NOISE_GL, v, f6, th0, cfg.sigma_calibration)))
        _CACHE["t4"] = out
    return _CACHE["t4"]


def fake_pilot(**kw):
    base = dict(cx_um=1.3, cy_um=-0.7, theta_rad=0.4, phi_rad=0.0, z_sub_um=0.31, k_star=0, d_hat_um=0.7,
                mu_hat_per_um=0.9)
    base.update(kw)
    return types.SimpleNamespace(**base)


def recorder(f6, nan_if=None):
    """(g, calls): g(*theta) = f6(*theta), NaN where nan_if(theta) is True; calls the list of (theta tuple,
    returned NaN) in call order."""
    calls = []

    def g(*th):
        bad = bool(nan_if(th)) if nan_if is not None else False
        calls.append((tuple(float(x) for x in th), bad))
        out = np.asarray(f6(*th), dtype=float)
        return np.full(out.shape, np.nan) if bad else out
    return g, calls


def toy_forward(shape, v):
    """A cheap finite forward of six parameters (no physics): 1 - a Gaussian dip whose depth and width move with
    every parameter; for the checks that need a finite model only."""
    def g(d, mu, v0, z_ax, s0, c):
        w = 0.3 + 0.2 * d + s0 + 0.05 * c + 0.01 * z_ax ** 2
        dip = (1.0 - math.exp(-mu * d)) * np.exp(-0.5 * ((v - v0) / w) ** 2)
        return np.broadcast_to(1.0 - dip, shape).copy()
    return g


def raises(exc, func, *args, **kw):
    try:
        func(*args, **kw)
    except exc:
        return True
    return False


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    rng = make_rng()
    # T1: I = (b0 + b1 v) tau exactly -> b [1e-10 relative: lstsq on a well-conditioned 2-column system, float64
    # roundoff ~1e-14 times the condition number] and resid 0 [1e-9 gl]
    v = np.linspace(-4.5, 4.5, 79)
    tau = 1.0 - 0.4 * np.exp(-0.5 * (v[None, :] / rng.uniform(0.4, 1.2, (5, 1))) ** 2)
    b_true = np.column_stack([rng.uniform(100.0, 220.0, 5), rng.uniform(-3.0, 3.0, 5)])
    I = (b_true[:, 0:1] + b_true[:, 1:2] * v[None, :]) * tau
    b, resid = SC.vp_background(I, tau, v)
    check_close("t1_exact_b0", b[:, 0], b_true[:, 0], rtol=1e-10)
    check_close("t1_exact_b1", b[:, 1], b_true[:, 1], rtol=1e-10)
    check_close("t1_exact_resid", resid, 0.0, rtol=0, atol=1e-9)
    print("    T1 exact: max |b - b_true| / |b_true| %.1e, max |resid| %.1e gl"
          % (float(np.max(np.abs(b - b_true) / np.abs(b_true))), float(np.max(np.abs(resid)))))

    # T7: summarize against values by hand. ok nodes sigma0 = 0.10, 0.12, 0.15 um, se_s = 0.01, 0.02, 0.01 um:
    # median 0.12; MAD = median(0.02, 0, 0.03) = 0.02; weights 10000, 2500, 10000 (sum 22500); mean = 2800 / 22500
    # = 28/225 um; SE = 1/150 um; chi2 = (302500 + 2500 + 330625) / 50625 = 113/9; dof 2.
    s = SC.summarize([0.10, 0.12, 0.15, 0.30], [0.01, 0.02, 0.01, 0.05], np.array([True, True, True, False]))
    assert s["n"] == 3 and s["dof"] == 2, s
    check_close("t7_median", s["median_um"], 0.12, rtol=1e-12)
    check_close("t7_mad", s["mad_um"], 0.02, rtol=1e-12)
    check_close("t7_mean", s["ivw_mean_um"], 28.0 / 225.0, rtol=1e-12)
    check_close("t7_se", s["ivw_se_um"], 1.0 / 150.0, rtol=1e-12)
    check_close("t7_chi2", s["chi2_cons"], 113.0 / 9.0, rtol=1e-12)
    one = SC.summarize([0.2, 0.5], [0.03, 0.01], np.array([True, False]))
    assert one["n"] == 1 and one["dof"] == 0 and math.isnan(one["chi2_cons"]), one
    check_close("t7_one", [one["median_um"], one["mad_um"], one["ivw_mean_um"], one["ivw_se_um"]],
                [0.2, 0.0, 0.2, 0.03], rtol=1e-12)
    none = SC.summarize([0.2, 0.5], [0.03, 0.01], np.array([False, False]))
    assert none["n"] == 0 and all(math.isnan(none[k]) for k in none if k != "n"), none
    print("    T7: mean %.6f (28/225), SE %.6f (1/150), chi2 %.6f (113/9), dof %d" % (
        s["ivw_mean_um"], s["ivw_se_um"], s["chi2_cons"], s["dof"]))

    # start_theta on a Gaussian dip D0 exp(-v^2 / (2 s^2)) on a 0.01 um line: smoothed by a Gaussian of one sample
    # it stays a Gaussian of SD sqrt(s^2 + h^2) and peak D0 s / sqrt(s^2 + h^2); W = 2 sqrt(2 ln 2) sqrt(s^2 + h^2).
    # [2e-3 relative: the smoothing's discrete kernel and the linear interpolation of the half crossing on a
    # 0.01 um grid, both of order (h / s)^2 ~ 3e-4]
    scfg = node_case()[0].sigma_calibration
    h, sd, D0, B = 0.01, 0.6, 40.0, 200.0
    vl = np.arange(-4.5, 4.5 + 0.5 * h, h)
    prof = np.vstack([B - 5.0 * np.exp(-0.5 * (vl / 0.3) ** 2), B - D0 * np.exp(-0.5 * (vl / sd) ** 2),
                      B - 10.0 * np.exp(-0.5 * (vl / 1.0) ** 2)])
    p = fake_pilot()
    trace = np.array([1.2, -0.65, 0.05])
    th0 = SC.start_theta(prof, np.full(3, B), vl, 1, p, trace, scfg)
    s_eff = math.sqrt(sd ** 2 + h ** 2)
    W = 2.0 * math.sqrt(2.0 * math.log(2.0)) * s_eff
    D = D0 * sd / s_eff
    d0 = W / scfg.width_to_d
    mu0 = -math.log(1.0 - D / B) / (scfg.hybrid_faint_factor * d0)
    check_close("start_d0", th0[0], d0, rtol=2e-3)
    check_close("start_mu0", th0[1], mu0, rtol=2e-3)
    st = NI.pilot_start(p, trace)
    assert th0[2] == st[2] and th0[3] == trace[2] + st[3], (th0, st)
    assert th0[4] == scfg.sigma0_start_um and th0[5] == scfg.growth_scale_start, th0
    other = SC.start_theta(prof, np.full(3, B), vl, 2, p, trace, scfg)          # the k* plane is k_index's
    assert other[0] > 1.5 * th0[0], (other, th0)
    print("    start on a Gaussian dip: d0 %.5f (closed form %.5f), mu0 %.5f (%.5f)" % (th0[0], d0, th0[1], mu0))


def test_reference():
    rng = make_rng(SEED + 7)
    v = np.linspace(-3.0, 3.0, 41)
    tau = rng.uniform(0.3, 1.0, (4, 41))
    I = rng.normal(150.0, 20.0, (4, 41))
    b, resid = SC.vp_background(I, tau, v)
    for k in range(4):                    # [1e-12: the same lstsq on the same matrix; roundoff only]
        A = np.column_stack([tau[k], v * tau[k]])
        ref = np.linalg.lstsq(A, I[k], rcond=None)[0]
        check_close("t1_lstsq_b_%d" % k, b[k], ref, rtol=1e-12, atol=1e-12)
        check_close("t1_lstsq_resid_%d" % k, resid[k], I[k] - A @ ref, rtol=1e-12, atol=1e-10)
    # NaN samples dropped: NaN in I on plane 0, in tau on plane 1; plane 2 left with 2 finite samples
    In, Tn = I.copy(), tau.copy()
    In[0, [3, 17, 30]] = np.nan
    Tn[1, [0, 5]] = np.nan
    In[2, 2:] = np.nan
    b, resid = SC.vp_background(In, Tn, v)
    for k in (0, 1, 3):
        m = np.isfinite(In[k]) & np.isfinite(Tn[k])
        A = np.column_stack([Tn[k, m], v[m] * Tn[k, m]])
        ref = np.linalg.lstsq(A, In[k, m], rcond=None)[0]
        check_close("t1_nan_b_%d" % k, b[k], ref, rtol=1e-12, atol=1e-12)
        assert_array_equal(np.isnan(resid[k]), ~m)
    assert np.all(np.isnan(b[2])) and np.all(np.isnan(resid[2])), "a plane of 2 finite samples gives NaN"

    # T2: forward_with_kernel at (sigma_r(0), 1) equals forward_factory with the unchanged configuration [exact]
    cfg, pilot, node, info, f6, k_index = node_case()
    th, _, _ = theta_true()
    U = info["U_axis_um"]
    s0 = float(cfg.renderer.sigma_r0_um)
    for label, c in (("FAST", cfg),
                     ("hybrid 4 x 8", dataclasses.replace(cfg, inference=dataclasses.replace(
                         cfg.inference, absorption="hybrid", ray_n_rho=4, ray_n_psi=8)))):
        a = SC.forward_with_kernel(pilot, U, node, c, line_fast.SectionCache())(th[0], th[1], th[2], th[3], s0, 1.0)
        ref = NI.forward_factory(pilot, U, node, c, line_fast.SectionCache())(th[0], th[1], th[2], th[3])
        assert_array_equal(a, ref)
        assert np.all(np.isfinite(a)) and np.min(a) < 0.9, (label, float(np.min(a)))
        print("    T2 %s: forward_with_kernel at (sigma_r(0) %.3f um, 1) equals forward_factory exactly; min tau %.4f"
              % (label, s0, float(np.min(a))))
    # and another kernel changes it (the kernel enters through the configuration)
    moved = float(np.max(np.abs(f6(th[0], th[1], th[2], th[3], 0.25, 1.0) - f6(th[0], th[1], th[2], th[3], s0, 1.0))))
    assert moved > 1e-3, moved


def test_convergence():
    raise unittest.SkipTest("not a discretisation: the block fits a forward model (the forward's discretisation is "
                            "Block 35's)")


def test_noise_free_recovery():
    th, _, _ = theta_true()
    th0, fit = t3_fit()
    err = fit.theta - th
    print("    T3 start %s" % np.array2string(th0, precision=4))
    print("    T3 fit   %s, %s, %d forward evaluations, %.1f s; %s" % (
        np.array2string(fit.theta, precision=6), fit.status, fit.n_eval, fit.seconds, fit.message))
    print("    T3 errors: d %.2e um (%.2e relative), mu %.2e /um, v0 %.2e um, z_ax %.2e um, sigma0 %.2e um, "
          "c %.2e (%.2e relative); chi2 %.2e" % (err[0], err[0] / th[0], err[1], err[2], err[3], err[4], err[5],
                                                 err[5] / th[5], fit.chi2))
    assert fit.status == "converged", (fit.status, fit.message)
    assert abs(err[4]) <= 0.005, ("sigma0", err[4])            # the entry's tolerances (T3)
    assert abs(err[0]) / th[0] <= 0.01, ("d", err[0])
    assert abs(err[5]) / th[5] <= 0.03, ("c", err[5])
    check_close("t3_background", fit.b, np.tile([B0, B1], (5, 1)), rtol=1e-4)   # the straight background found


def test_noisy_recovery():
    th, _, _ = theta_true()
    for seed, In, th0, fit in t4_fits():
        err = fit.theta - th
        z = err / fit.se_scaled
        print("    T4 seed %d: %s, %d evaluations, %.1f s, chi2/dof %.3f; errors d %+.4f, sigma0 %+.4f, c %+.4f; "
              "SE_s %.4f, %.4f, %.4f; z %+.2f, %+.2f, %+.2f" % (
                  seed, fit.status, fit.n_eval, fit.seconds, fit.chi2 / fit.dof, err[0], err[4], err[5],
                  fit.se_scaled[0], fit.se_scaled[4], fit.se_scaled[5], z[0], z[4], z[5]))
        assert fit.status == "converged", (seed, fit.status, fit.message)
        for j in (0, 4, 5):
            assert abs(err[j]) <= 3.0 * fit.se_scaled[j], (seed, SC.PARAMS[j], err[j], fit.se_scaled[j])


def test_invariants():
    cfg, pilot, node, info, f6, k_index = node_case()
    seed, In, th0, fit = t4_fits()[0]
    v = np.asarray(node.v, dtype=float)
    tau = f6(*fit.theta)
    b, resid = SC.vp_background(In, tau, v)
    assert_array_equal(fit.b, b)
    check_close("model_gl", fit.model_gl, (b[:, 0:1] + b[:, 1:2] * v[None, :]) * tau, rtol=1e-12)
    r = (In - fit.model_gl) / NOISE_GL
    check_close("chi2", fit.chi2, float(np.sum(r ** 2)), rtol=1e-10)
    check_close("chi2_per_plane", fit.chi2_per_plane, np.mean(r ** 2, axis=1), rtol=1e-10)
    assert fit.n_res == In.size and fit.dof == In.size - (6 + 2 * In.shape[0]), (fit.n_res, fit.dof)
    check_close("se_scaled", fit.se_scaled, fit.se * math.sqrt(fit.chi2 / fit.dof), rtol=1e-12)
    scfg = cfg.sigma_calibration
    lo, hi = SC._bounds(scfg, th0[2], th0[3])
    assert np.all(fit.theta >= lo) and np.all(fit.theta <= hi) and fit.at_bound == (), (fit.theta, fit.at_bound)
    # vp_background is linear in I [1e-10: roundoff of two lstsq solves]
    rng = make_rng(SEED + 3)
    I1, I2 = rng.normal(150, 10, In.shape), rng.normal(150, 10, In.shape)
    b1, r1 = SC.vp_background(I1, tau, v)
    b2, r2 = SC.vp_background(I2, tau, v)
    b12, r12 = SC.vp_background(2.5 * I1 - 0.7 * I2, tau, v)
    check_close("vp_linear_b", b12, 2.5 * b1 - 0.7 * b2, rtol=1e-10, atol=1e-10)
    check_close("vp_linear_resid", r12, 2.5 * r1 - 0.7 * r2, rtol=1e-10, atol=1e-9)


def test_contract():
    cfg, pilot, node, info, f6, k_index = node_case()
    assert SC.PARAMS == ("d_um", "mu_per_um", "v0_um", "z_ax_um", "sigma0_um", "growth_scale")
    assert SC.STATUSES == ("converged", "max_nfev", "failed")
    th0, fit = t3_fit()
    n_planes, n_v = len(node.ks), len(node.v)
    for name in ("theta", "se", "se_scaled"):
        a = getattr(fit, name)
        assert isinstance(a, np.ndarray) and a.shape == (6,) and a.dtype == np.float64, (name, a)
    assert fit.b.shape == (n_planes, 2) and fit.b.dtype == np.float64, fit.b.shape
    assert fit.model_gl.shape == (n_planes, n_v) and np.all(np.isfinite(fit.model_gl)), fit.model_gl.shape
    assert fit.chi2_per_plane.shape == (n_planes,), fit.chi2_per_plane.shape
    assert isinstance(fit.chi2, float) and isinstance(fit.seconds, float) and fit.seconds > 0
    for name in ("n_res", "dof", "n_eval"):
        assert isinstance(getattr(fit, name), int), name
    assert fit.n_eval >= 7, fit.n_eval            # at least the start and its Jacobian
    assert fit.status in SC.STATUSES and isinstance(fit.at_bound, tuple) and isinstance(fit.message, str)
    assert all(p in SC.PARAMS for p in fit.at_bound)
    assert np.all(np.isfinite(fit.theta)) and np.all(np.isfinite(fit.se)) and np.all(fit.se > 0)
    try:
        dataclasses.replace(fit, status="x").__setattr__("status", "y")
        raise AssertionError("SigmaFit must be frozen")
    except dataclasses.FrozenInstanceError:
        pass
    sel = SC.select_planes(np.array([7, 3, 4, 5, 6, 9]), 5, 2)
    assert sel.dtype == np.int64 and list(sel) == [0, 1, 2, 3, 4], sel
    tau = f6(*fit.theta)
    assert tau.shape == (n_planes, n_v) and tau.dtype == np.float64
    b, resid = SC.vp_background(fit.model_gl, tau, node.v)
    assert b.shape == (n_planes, 2) and resid.shape == (n_planes, n_v)
    s = SC.summarize(np.array([0.1, 0.2]), np.array([0.01, 0.02]), np.array([True, True]))
    assert set(s) == {"n", "median_um", "mad_um", "ivw_mean_um", "ivw_se_um", "chi2_cons", "dof"}, s
    assert isinstance(s["n"], int) and isinstance(s["dof"], int)


def test_determinism():
    cfg, pilot, node, info, f6, k_index = node_case()
    In, v = noisy_profiles(SEED + 5)
    scfg = dataclasses.replace(cfg.sigma_calibration, max_nfev=6)
    fits = []
    for _ in range(2):
        th0 = start_of(In, v)
        fits.append(SC.fit_sigma_node(In, NOISE_GL, v, f6, th0, scfg))
    a, b = fits
    for f in dataclasses.fields(SC.SigmaFit):
        if f.name == "seconds":
            continue
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            assert_array_equal(x, y, err_msg=f.name)
        else:
            assert x == y, (f.name, x, y)
    print("    T5: two fits (max_nfev 6) identical; %s, %d evaluations" % (a.status, a.n_eval))


def test_start_background():
    cfg, pilot, node, info, f6, k_index = node_case()
    scfg = cfg.sigma_calibration
    th, _, _ = theta_true()
    v = np.asarray(node.v, dtype=float)
    tau = f6(*th)
    n_planes = len(node.ks)
    # T8: d0 and mu0 on (150 + b1 v) tau(theta_true), b1 = -6, 0, +6 gl/um, noise-free [5 %, the amendment's bound]
    starts, flat_rule = {}, {}
    for b1 in (-6.0, 0.0, 6.0):
        I = (B0 + b1 * v)[None, :] * tau
        starts[b1] = SC.start_theta(I, np.full(n_planes, B0), v, k_index, pilot, node.node_xyz, scfg)
        # beside it, for the record only: the rule before amendment 4 (the flat B_k* = 150 gl)
        f = clearness.dip_features(clearness.smooth_dip(B0 - I[k_index], 1.0), v)
        flat_rule[b1] = f["fwhm_um"] / scfg.width_to_d
    ref = starts[0.0]
    for b1 in (-6.0, 6.0):
        rel = np.abs(starts[b1][[0, 1]] / ref[[0, 1]] - 1.0)
        print("    T8 b1 %+.0f gl/um: d0 %.4f um, mu0 %.4f /um (b1 0: %.4f, %.4f; relative %.2e, %.2e); against the flat "
              "B_k* (the rule before amendment 4) d0 would be %.4f um" % (b1, starts[b1][0], starts[b1][1], ref[0],
                                                                        ref[1], rel[0], rel[1], flat_rule[b1]))
        assert np.all(rel <= 0.05), (b1, starts[b1], ref)
    # the line through the ends against numpy.polyfit [1e-10: two least-squares routines, roundoff only]
    rng = make_rng(SEED + 11)
    Ik = 140.0 + 5.0 * v + rng.normal(0.0, 1.0, v.size)
    ends = np.abs(v) >= scfg.start_ends_um
    b_ref, a_ref = np.polyfit(v[ends], Ik[ends], 1)
    check_close("end_line", SC._end_line(Ik, v, scfg.start_ends_um), [a_ref, b_ref], rtol=1e-10, atol=1e-10)
    # a profile with no measurable width gives the pilot's d: flat (its depth D is roundoff of either sign, so mu0 is
    # the pilot's or the formula's ~0 clipped to the lower bound: only kept inside the bounds); a bump (D < 0) and
    # a NaN sample (D not finite) give the pilot's mu too; none raises
    lo_d, hi_d = scfg.d_bounds_um
    lo_m, hi_m = scfg.mu_bounds_per_um
    p_d = min(max(float(pilot.d_hat_um), lo_d), hi_d)
    p_m = min(max(float(pilot.mu_hat_per_um), lo_m), hi_m)
    flat = np.full((n_planes, v.size), B0)
    st = SC.start_theta(flat, np.full(n_planes, B0), v, k_index, pilot, node.node_xyz, scfg)
    assert st[0] == p_d and lo_m <= st[1] <= hi_m, (st, p_d)
    bump = flat.copy()
    bump[k_index] += 20.0 * np.exp(-0.5 * (v / 0.5) ** 2)
    holed = (B0 * tau).copy()
    holed[k_index, v.size // 2] = np.nan
    for label, prof in (("a bump", bump), ("a NaN sample", holed)):
        st = SC.start_theta(prof, np.full(n_planes, B0), v, k_index, pilot, node.node_xyz, scfg)
        assert st[0] == p_d and st[1] == p_m, (label, st, p_d, p_m)
    print("    T8 no measurable width: flat d0 %.4f um (the pilot's); a bump, a NaN sample: d0 %.4f um, mu0 %.4f /um = "
          "the pilot's (clipped)" % (p_d, st[0], st[1]))
    # fewer than 3 samples at an end (start_ends_um 4.4 on the +-4.46 um line: one sample each): L = B_gl[k*]
    s44 = dataclasses.replace(scfg, start_ends_um=4.4)
    assert SC._end_line(B0 * tau[k_index], v, 4.4) is None
    I = B0 * tau
    for B in (B0, 140.0):
        st = SC.start_theta(I, np.full(n_planes, B), v, k_index, pilot, node.node_xyz, s44)
        f = clearness.dip_features(clearness.smooth_dip(B - I[k_index], 1.0), v)
        d0 = f["fwhm_um"] / scfg.width_to_d
        c = f["centre_um"] if math.isfinite(f["centre_um"]) else 0.0
        mu0 = -math.log(max(1.0 - f["depth_gl"] / B, 1e-3)) / (scfg.hybrid_faint_factor * d0)
        assert st[0] == min(max(d0, lo_d), hi_d) and st[1] == min(max(mu0, lo_m), hi_m), (B, st, d0, mu0, c)
    assert SC.start_theta(I, np.full(n_planes, 140.0), v, k_index, pilot, node.node_xyz, s44)[0] != \
        SC.start_theta(I, np.full(n_planes, B0), v, k_index, pilot, node.node_xyz, s44)[0]


def test_edge_cases():
    cfg, pilot, node, info, f6, k_index = node_case()
    scfg = cfg.sigma_calibration
    I, v = clean_profiles()
    th, _, _ = theta_true()
    shape = I.shape

    # T6 (a): a forward returning NaN -> "failed", no exception, theta NaN
    fit = SC.fit_sigma_node(I, NOISE_GL, v, lambda *a: np.full(shape, np.nan), th, scfg)
    assert fit.status == "failed" and np.all(np.isnan(fit.theta)) and "not finite" in fit.message, fit
    assert fit.n_eval == 1, fit.n_eval
    print("    T6 NaN forward: %s after %d evaluation(s): %s" % (fit.status, fit.n_eval, fit.message))

    # T6 (b): the forward NaN only beyond c's upper bound 1.31, the truth c = 1.3 within h_c = 0.02 of it: the
    # Jacobian's c column is taken backwards (the bound rule), the forward is never called beyond the bound
    h_c = float(scfg.diff_steps[5])
    th_near = th + np.array([0.02, -0.01, 0.02, -0.03, 0.01, -0.03])       # a start near the truth (for time)
    sb = dataclasses.replace(scfg, growth_scale_bounds=(0.05, 1.31))
    g, calls = recorder(f6, nan_if=lambda t: t[5] > 1.31)
    fit = SC.fit_sigma_node(I, NOISE_GL, v, g, th_near, sb)
    xh = fit.theta
    back = tuple(float(x) for x in np.r_[xh[:5], xh[5] - h_c])
    assert fit.status == "converged", (fit.status, fit.message)
    assert not any(bad for _, bad in calls) and max(c[0][5] for c in calls) <= 1.31, "called beyond the bound"
    assert any(c[0] == back for c in calls), "no backward c column at the solution"
    assert abs(xh[5] - 1.3) <= 1e-3 and abs(xh[4] - 0.15) <= 1e-3, xh
    print("    T6 NaN beyond the bound: %s, c %.6f (bound 1.31), %d evaluations, backward column at c - %.2f found"
          % (fit.status, xh[5], fit.n_eval, h_c))

    # T6 (b'): the forward NaN beyond c = 1.31 inside the bounds (0.05, 10): the non-finite rule
    g, calls = recorder(f6, nan_if=lambda t: t[5] > 1.31)
    fit = SC.fit_sigma_node(I, NOISE_GL, v, g, th_near, scfg)
    xh = fit.theta
    back = tuple(float(x) for x in np.r_[xh[:5], xh[5] - h_c])
    n_nan = sum(1 for _, bad in calls if bad)
    assert fit.status == "converged", (fit.status, fit.message)
    assert n_nan > 0 and any(c[0] == back for c in calls), (n_nan, "no backward c column at the solution")
    assert abs(xh[5] - 1.3) <= 1e-3 and abs(xh[4] - 0.15) <= 1e-3, xh
    print("    T6 NaN beyond c 1.31 inside the bounds: %s, c %.6f, %d of %d evaluations NaN, backward column found"
          % (fit.status, xh[5], n_nan, len(calls)))

    # T6 (c): fewer than 3 planes is a ValueError
    assert raises(ValueError, SC.select_planes, np.arange(5), 0, 1)                  # planes 0, 1
    assert raises(ValueError, SC.select_planes, np.arange(5), 2, 0.5)                # half < 1
    assert list(SC.select_planes(np.arange(5), 0, 2)) == [0, 1, 2]
    assert raises(ValueError, SC.fit_sigma_node, I[:2], NOISE_GL, v, f6, th, scfg)

    # T6 (d): a start outside the bounds is clipped -- start_theta (a dip 8 um wide on a +-20 um line, its straight
    # background through |v| >= 12 um, where the dip has fallen below 0.2 % of its depth; sigma0's start 0.9 um)...
    vl = np.linspace(-20.0, 20.0, 1601)
    wide = np.vstack([200.0 - 30.0 * np.exp(-0.5 * (vl / 3.4) ** 2)] * 3)
    st = SC.start_theta(wide, np.full(3, 200.0), vl, 1, fake_pilot(), [1.2, -0.65, 0.05],
                        dataclasses.replace(scfg, sigma0_start_um=0.9, growth_scale_start=20.0, start_ends_um=12.0))
    assert st[0] == scfg.d_bounds_um[1] and st[4] == scfg.sigma0_bounds_um[1], st
    assert st[5] == scfg.growth_scale_bounds[1], st
    # ... and fit_sigma_node (the first forward call at the clipped start; a finite toy forward, one evaluation)
    g, calls = recorder(toy_forward(shape, v))
    far = np.array([10.0, 20.0, th[2], th[3], 1.0, 50.0])
    fit = SC.fit_sigma_node(I, NOISE_GL, v, g, far, dataclasses.replace(scfg, max_nfev=1))
    first = np.array(calls[0][0])
    check_close("clipped_start", first[[0, 1, 4, 5]], [6.0, 10.0, 0.6, 10.0], rtol=1e-8)
    assert fit.status == "max_nfev" and "clipped" in fit.message and "d_um" in fit.message, fit.message
    print("    T6 start outside the bounds: first call at %s; %s" % (np.array2string(first, precision=4),
                                                                      fit.message))

    # start_theta on a flat profile: no measurable width, the pilot's d (amendment 4), no exception (mu0: see
    # test_start_background); with a pilot that has neither d nor mu (NaN), a start that is not finite -> the fit
    # "failed", no evaluation
    flat = np.full((3, vl.size), 200.0)
    st = SC.start_theta(flat, np.full(3, 200.0), vl, 1, fake_pilot(), [1.2, -0.65, 0.05], scfg)
    assert st[0] == 0.7 and scfg.mu_bounds_per_um[0] <= st[1] <= scfg.mu_bounds_per_um[1], st
    st = SC.start_theta(flat, np.full(3, 200.0), vl, 1, fake_pilot(d_hat_um=float("nan"), mu_hat_per_um=float("nan")),
                        [1.2, -0.65, 0.05], scfg)
    assert math.isnan(st[0]) and math.isnan(st[1]) and np.all(np.isfinite(st[2:])), st
    fit = SC.fit_sigma_node(I, NOISE_GL, v, f6, np.r_[np.nan, th[1:]], scfg)
    assert fit.status == "failed" and fit.n_eval == 0 and "start" in fit.message, fit
    # a forward NaN at the start only -> "failed" (the entry: "at the start it is failed")
    g, calls = recorder(f6, nan_if=lambda t: t == tuple(float(x) for x in th))
    fit = SC.fit_sigma_node(I, NOISE_GL, v, g, th, scfg)
    assert fit.status == "failed" and len(calls) == 1, (fit.status, len(calls))

    # forward_with_kernel: NaN on a kernel with_kernel refuses and on ExtrusionInvalid; ValueError for another family
    assert np.all(np.isnan(f6(th[0], th[1], th[2], th[3], -0.1, 1.0)))             # sigma0 <= 0
    assert np.all(np.isnan(f6(th[0], th[1], th[2], th[3], 0.15, 0.0)))             # c <= 0
    dec = dataclasses.replace(cfg, renderer=dataclasses.replace(
        cfg.renderer, sigma_r0_um=0.3, kernel_table_sigma_um=(0.3, 0.1, 0.122, 0.262, 0.438, 0.603)))
    fd = SC.forward_with_kernel(pilot, info["U_axis_um"], node, dec, None)
    assert np.all(np.isnan(fd(th[0], th[1], th[2], th[3], 0.05, 1.0)))              # a knot variance <= 0
    fe = SC.forward_with_kernel(pilot, (1.0, 1.0), node, cfg, None)                   # caps within the margin
    assert np.all(np.isnan(fe(*th))), "ExtrusionInvalid must give NaN"
    emp = dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer, kernel_family="empirical"))
    assert raises(ValueError, SC.forward_with_kernel, pilot, info["U_axis_um"], node, emp, None)

    # ValueErrors
    assert raises(ValueError, SC.vp_background, I, I[:, :-1], v)
    assert raises(ValueError, SC.vp_background, I, I, v[:-1])
    assert raises(ValueError, SC.fit_sigma_node, I, 0.0, v, f6, th, scfg)
    assert raises(ValueError, SC.fit_sigma_node, I, NOISE_GL, v[:-1], f6, th, scfg)
    assert raises(ValueError, SC.fit_sigma_node, I, NOISE_GL, v, f6, th[:5], scfg)
    few = I.copy()
    few[1, 2:] = np.nan
    assert raises(ValueError, SC.fit_sigma_node, few, NOISE_GL, v, f6, th, scfg)
    assert raises(ValueError, SC.fit_sigma_node, I, NOISE_GL, v, lambda *a: np.ones((2, 2)), th, scfg)
    assert raises(ValueError, SC.start_theta, I, np.full(5, 150.0), v, 7, pilot, node.node_xyz, scfg)
    assert raises(ValueError, SC.start_theta, I, np.full(4, 150.0), v, 2, pilot, node.node_xyz, scfg)
    assert raises(ValueError, SC.start_theta, I, np.full(5, 150.0), v[:-1], 2, pilot, node.node_xyz, scfg)
    assert raises(ValueError, SC.start_theta, I, np.full(5, 150.0), np.r_[v[:-1], np.nan], 2, pilot, node.node_xyz,
                  scfg)
    assert raises(ValueError, SC.start_theta, I, np.full(5, 150.0), v, 2, pilot, [1.0, np.nan, 0.0], scfg)
    # B_gl is only the fallback background now (amendment 4): a negative one is ignored when the ends are measured
    assert_array_equal(SC.start_theta(I, np.full(5, -1.0), v, 2, pilot, node.node_xyz, scfg),
                       SC.start_theta(I, np.full(5, 150.0), v, 2, pilot, node.node_xyz, scfg))
    assert raises(ValueError, SC.summarize, [0.1], [0.01, 0.02], np.array([True, True]))
    assert raises(ValueError, SC.summarize, [0.1, 0.2], [0.01, 0.02], np.array([1, 1]))
    assert raises(ValueError, SC.summarize, [0.1, 0.2], [0.01, np.nan], np.array([True, True]))
    assert raises(ValueError, SC.summarize, [0.1, np.nan], [0.01, 0.02], np.array([True, True]))
    # NaN on a node that is not ok is allowed
    assert SC.summarize([0.1, np.nan], [0.01, np.nan], np.array([True, False]))["n"] == 1
    # the inputs are unchanged by the fit
    I0 = I.copy()
    SC.fit_sigma_node(I, NOISE_GL, v, lambda *a: np.full(shape, np.nan), th, scfg)
    assert_array_equal(I, I0)


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"seed {SEED}")
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print(f"== {Path(__file__).name}")
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
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))

    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline}".rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
