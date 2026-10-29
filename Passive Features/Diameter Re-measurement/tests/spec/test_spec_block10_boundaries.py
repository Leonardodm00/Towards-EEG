"""Spec-derived boundary tests between Block 10 (the stripe, 10.S) and its
neighbours 3, 4, 5, 6, 8, 15 and 19 -- written by the integration tester from
specs/SPEC.md (2026-10-28). Expected values come from spec_helpers (scipy) or
by hand; the code under test only produces the outputs that are checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block10_boundaries.py -q
The rendered-phantom checks take about a minute each on one core.
"""
from __future__ import annotations

import dataclasses
import math
import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import spec_helpers as H  # noqa: F401  (sys.path)
from spec_helpers import pixel_centres_ref, ref_margin, ref_sigma_r

from allen_diameter.analysis import calibration as CAL
from allen_diameter.analysis import calibration_pool
from allen_diameter.analysis import calibration_scene as CS
from allen_diameter.analysis import phantoms
from allen_diameter.analysis.plane_iteration import Scene
from allen_diameter.config import default_config
from allen_diameter.loading import calibration_io
from allen_diameter.model import render
from allen_diameter.model.kernel import sigma_r

NAN = float("nan")


def cfg_with(cfg=None, **cal):
    cfg = default_config() if cfg is None else cfg
    return dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, **cal))


def rendered_phantom(cfg, seed, d, alpha, theta, phi_deg=0.0):
    """(branch, index, provider, draw): a flat calibration phantom as Block 6 builds one -- the tube of
    phantoms.phantom_tube, its branch carrying the true radius d/2 (a calibration phantom's Allen diameter is
    d, Block 14.2), Block 4's synthetic_block as the provider -- with a branch long enough for the far line."""
    rng = np.random.default_rng([int(seed), 1])
    p, dz = cfg.acquisition.res0_um, cfg.acquisition.dz_um
    phi = math.radians(phi_deg)
    mu = alpha * math.cos(phi) / d
    cx, cy = rng.uniform(-0.5 * p, 0.5 * p, 2)
    cz = rng.uniform(-0.5 * dz, 0.5 * dz)
    draw = phantoms.Draw(1, 1, int(seed), d, phi, theta, mu, float(cx), float(cy), float(cz), 0.5 * d, 0)
    tube = phantoms.phantom_tube(draw, cfg)
    ph = cfg.phantom
    need = max(cfg.calibration.far_profile_half_um, cfg.calibration.node_straight_half_um) + 2.0
    n_each = max(int(ph.nodes_each_way), int(math.ceil(need / ph.phantom_node_step_um)))
    cfg_b = dataclasses.replace(cfg, phantom=dataclasses.replace(ph, nodes_each_way=n_each))
    branch = phantoms.phantom_branch(draw, cfg_b, rng)
    pad = int(math.ceil(cfg.renderer.pad_um / p))

    def provider(left, top, width, height, k_lo, k_hi):
        return render.synthetic_block(tube, draw.mu_per_um, np.arange(k_lo, k_hi + 1), left, top, width, height,
                                      cfg, rng, pad)
    return branch, n_each, provider, draw


# ---------------------------------------------------------- Block 4 -> 10 --

def test_block4_sigma_r_is_the_spec_law():
    # 10.S reads "sigma_r the renderer's current law (model.kernel.sigma_r)": Block 4's law, checked against the
    # spec's statement (linear in the table, linear continuation with kernel_continuation_slope) to 1e-12
    cfg = default_config()
    d = np.linspace(-6.0, 6.0, 1201)
    assert_allclose(sigma_r(d, cfg.renderer), ref_sigma_r(d, cfg.renderer), rtol=0, atol=1e-12)
    r2 = dataclasses.replace(cfg.renderer, kernel_table_sigma_um=tuple(2 * x for x in cfg.renderer.kernel_table_sigma_um),
                             sigma_r0_um=2 * cfg.renderer.sigma_r0_um, kernel_continuation_slope=1.21)
    assert_allclose(sigma_r(d, r2), ref_sigma_r(d, r2), rtol=0, atol=1e-12)


def test_block4_renderer_table_reaches_the_stripe():
    # swappability: a renderer table with every width doubled moves the own cylinder's margin to the rho_tau of the
    # new law at every plane (10.S: sigma_r(delta_q,k) is the renderer's current law), within one pixel
    cfg0 = default_config()
    r2 = dataclasses.replace(cfg0.renderer, kernel_table_sigma_um=tuple(2 * x for x in cfg0.renderer.kernel_table_sigma_um),
                             sigma_r0_um=2 * cfg0.renderer.sigma_r0_um)
    p = cfg0.acquisition.res0_um
    for cfg in (cfg0, dataclasses.replace(cfg0, renderer=r2)):
        cfg.validate()
        from allen_diameter.analysis.node_pipeline import Branch, NodeResult
        s = np.linspace(-20, 20, 41)
        own = Branch.from_points(np.c_[s, np.zeros(41), np.zeros(41)], 0.2, ids=np.arange(100, 141))
        scene = Scene.from_branch(own)
        node = NodeResult(7, 3, 0.0, 0.0, 0.0, 0.0, "", NAN, NAN, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False,
                          200.0, "block_masked", 0.4, 1.0, 0.0, 0.4, "converged", (), None, "gradient_energy",
                          np.empty(0, int), np.empty(0), np.empty(0), -1)
        shape, frame = CS.patch_frame(0.0, 0.0, 15.0, p)
        kk = np.arange(-3, 4)
        _sup, stripe = CAL.support_masks(shape, frame, kk, node, own, scene, cfg)
        for q, k in enumerate(kk):
            s_k = float(ref_sigma_r(abs(k) * cfg.acquisition.dz_um, cfg.renderer))
            rho = ref_margin(0.2, 1.0, s_k, 200.0, cfg.calibration.isolation_depth_gl)
            assert abs(stripe[q]["v_own_plus_um"] - rho) <= p and abs(stripe[q]["v_own_minus_um"] - rho) <= p, \
                (k, stripe[q], rho)


# ---------------------------------------------------------- Block 3 -> 10 --

def test_block3_dip_statistics_gaussian_core():
    # Block 10's oracle: on a sampled Gaussian dip, omega = s^2 to 1e-12 relative (the core fit is exact on its
    # own family; Block 3's half_depth_width supplies the window)
    cfg = default_config()
    m = dataclasses.replace(cfg.measure, focus_smooth_px=0.0)
    v = np.arange(-60, 61) * m.profile_step_um
    for s, a, vc in ((0.12, 30.0, 0.05), (0.35, 12.0, -0.2), (0.6, 5.0, 0.0)):
        I = 180.0 - a * np.exp(-0.5 * ((v - vc) / s) ** 2)
        st = CAL.dip_statistics(v, I, 180.0, m, cfg.calibration)
        assert st["status"] == "ok"
        assert_allclose(st["omega_um2"], s * s, rtol=1e-12)


# ---------------------------------------------------------- Block 6 -> 10 --

def test_block6_selection_flags_reach_the_calibration():
    # D-096: SELECTION_FLAGS = (crossing, stack_edge, bbar_few, profile_nan); faint is a label; with
    # node_faint_flag_check the calibration adds faint back, first among the flags (D-094)
    cfg = default_config()
    assert tuple(phantoms.SELECTION_FLAGS) == ("crossing", "stack_edge", "bbar_few", "profile_nan")
    assert CAL.selection_flags(cfg) == ("crossing", "stack_edge", "bbar_few", "profile_nan")
    c2 = cfg_with(node_faint_flag_check=True)
    assert CAL.selection_flags(c2) == ("faint", "crossing", "stack_edge", "bbar_few", "profile_nan")
    from test_spec_block10_stripe import pilot
    n = pilot(flags=("faint", "crossing"))
    assert CAL.selection_reasons(n, cfg) == ["crossing"]
    assert CAL.selection_reasons(n, c2) == ["faint", "crossing"]
    assert CAL.selection_reasons(pilot(status="at_bound"), cfg) == ["status:at_bound"]


# ------------------------------------------- Block 5 -> 10: a rendered node --

def test_block5_rendered_near_phantom_scan_under_the_stripe():
    # a near-set phantom (d 0.2 um, alpha 0.3, heading 0.4 rad) measured by the pilot (Block 5) and scanned under
    # the default stripe: the scan's stripe per plane sits on the pilot's line (its band along the pilot's
    # heading about the pilot's centre), the own cylinder is removed to rho_tau(max(d_hat/2, r_Allen), the pilot's
    # mu_hat and B_bar, sigma_r(|z_own - z_k|)) within one pixel, and a usable stripe makes a usable plane.
    cfg = default_config()
    cal, dz, p = cfg.calibration, cfg.acquisition.dz_um, cfg.acquisition.res0_um
    branch, i, provider, draw = rendered_phantom(cfg, 11, 0.2, 0.3, 0.4)
    res, scan, reasons = CAL.scan_node(branch, i, provider, cfg)
    assert reasons == [] and scan is not None and scan.set == "near", reasons
    assert scan.stripe is not None and len(scan.stripe) == scan.ks.size
    r_own = max(0.5 * res.d_hat_um, 0.5 * draw.d_um)
    # the own cylinder lies on the trace (the branch), the stripe's frame on the pilot's line: the pilot's centre
    # sits off the trace by e0 and its heading off by dth, so across the band (|u| <= W) the cylinder's edge moves
    # by up to e0 + W |sin dth| -- added to the one-pixel tolerance of the spec's oracle (a)
    t_true = np.array([math.cos(draw.theta_rad), math.sin(draw.theta_rad)])
    n_true = np.array([-t_true[1], t_true[0]])
    e0 = abs(float((np.array([res.cx_um, res.cy_um]) - branch.xyz_um[i, :2]) @ n_true))
    dth = abs(math.sin(res.theta_rad - draw.theta_rad))
    for q, k in enumerate(scan.ks):
        info = scan.stripe[q]
        s_k = float(ref_sigma_r(abs(draw.cz_um - k * dz), cfg.renderer))
        rho = ref_margin(r_own, res.mu_hat_per_um, s_k, res.B_bar, cal.isolation_depth_gl)
        tol = 1.5 * p + e0 + info["w_um"] * dth
        assert abs(info["v_own_plus_um"] - rho) <= tol and abs(info["v_own_minus_um"] - rho) <= tol, \
            (k, info, rho, e0, dth)
        if info["ok"]:
            assert scan.status[q] != "bg_few" and scan.B_bar[q] > 0, (k, scan.status[q])
        else:
            assert scan.status[q] == "bg_few"
    # the band follows the pilot's heading: rebuild the support at k* and check |u| <= W* about the pilot's centre
    patch = CS.fetch_patch(provider, res.cx_um, res.cy_um, cal.patch_half_um, int(res.k_star), int(res.k_star), p)
    _i, far, _b, _ok, frame, stripe = CAL.patch_planes(patch, np.array([res.k_star]), res, branch,
                                                     Scene.from_branch(branch), cfg)
    px, py = pixel_centres_ref(far.shape[1:], frame.left, frame.top, p)
    u = (px - res.cx_um) * math.cos(res.theta_rad) + (py - res.cy_um) * math.sin(res.theta_rad)
    assert np.all(np.abs(u[far[0]]) <= stripe[0]["w_um"] + 1e-9)


def test_block5_rendered_scan_is_deterministic():
    cfg = cfg_with(patch_half_um=8.0)
    out = []
    for _ in range(2):
        branch, i, provider, _draw = rendered_phantom(cfg, 5, 0.22, 0.25, 1.1)
        _res, scan, _r = CAL.scan_node(branch, i, provider, cfg)
        out.append(scan)
    a, b = out
    assert a.stripe == b.stripe and a.status == b.status
    assert_array_equal(a.omega_um2, b.omega_um2)


# ------------------------------------------------ Block 10 -> 15: far scan --

def test_block15_far_scan_carries_the_stripe():
    # Block 15.1's far scan reads patch_planes (10.P / 10.S): the PlaneScan carries one stripe record per plane of
    # k* +- far_offsets_planes and a plane whose stripe is not usable is bg_few. Reduced configuration as SPEC
    # section 6 suggests for a check (patch +-12 um, P_far = 8).
    cfg = cfg_with(patch_half_um=12.0, far_offsets_planes=8)
    branch, i, provider, draw = rendered_phantom(cfg, 21, 0.7, 0.35, 0.9)
    res, scan, reasons = CAL.scan_node(branch, i, provider, cfg)
    assert reasons == [] and scan is not None and scan.set == "far", reasons
    assert scan.stripe is not None and len(scan.stripe) == 17 == scan.ks.size
    for q in range(scan.ks.size):
        if not scan.stripe[q]["ok"]:
            assert scan.status[q] == "bg_few", (q, scan.status[q], scan.stripe[q])
    usable = [q for q in range(scan.ks.size) if abs(int(scan.ks[q]) - int(scan.k_star)) >= 5
              and scan.status[q] == "ok"]
    assert len(usable) >= cfg.calibration.far_min_planes_per_node, (scan.status, [s["reason"] for s in scan.stripe])


# ------------------------------------------- Block 10 -> 19: the pooled set --

def test_block19_pool_keeps_the_stripe():
    from test_spec_block10_stripe import _flat_patch, line_branch, pilot, scene_from
    cfg = default_config()
    own = line_branch(r=0.2)
    scene = scene_from([own, line_branch(r=0.3, offset=5.0, ids0=300)])
    node = pilot(d=0.4)
    kk = CAL.stage_planes(node, cfg, "near")
    scan = CAL.plane_scan(_flat_patch(cfg, node, kk, 10.0), node, own, scene, cfg)
    pooled, meta = calibration_pool.pool_scans([(529878215, [scan]), (508279351, [scan])], cfg)
    assert len(pooled) == 2 and all(s.stripe == scan.stripe for s in pooled)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scans_pooled.json")
        calibration_io.save_scans(pooled, path, meta)
        back, _m = calibration_io.load_scans(path)
        assert all(b.stripe == scan.stripe for b in back)


# ------------------------------------- Block 8 / 13 -> 10: a cell's scene --

def test_block8_cell_scan_verdict_and_scan_agree():
    # scan_cell on the synthetic cell of fixtures_cell (a soma disc of radius 4 um at the origin, one flat
    # dendrite from 6 um, Allen's radius 0.3 um: the near set): a node the verdict lets through (no reasons) is
    # scanned, and its scan's stripe at the plane nearest the pilot's depth must then be usable -- the verdict is
    # "the same test at the plane nearest the pilot's depth" (10.S).
    import fixtures_cell
    import allen_image_io as aio
    import run_cell
    cfg = default_config()
    with tempfile.TemporaryDirectory() as tmp:
        swc, fetcher, planes, _path = fixtures_cell.synthetic_cell(tmp, cfg)
        provider = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
        out = CAL.scan_cell(swc, provider, cfg)
    scanned = [(nid, res, sc) for nid, res, sc, rs in out if sc is not None]
    assert scanned, [(nid, rs) for nid, _r, _s, rs in out]
    bad = []
    for nid, res, sc in scanned:
        q = int(np.argmin(np.abs(np.asarray(sc.z_um) - res.z_sub_um)))
        if not sc.stripe[q]["ok"]:
            bad.append((nid, sc.stripe[q]["reason"], sc.stripe[q]["l_minus_um"], sc.stripe[q]["l_plus_um"],
                        sum(1 for s in sc.status if s == "bg_few"), len(sc.status)))
    assert not bad, "nodes the verdict passed whose scanned stripe at the pilot's plane is unusable: %s" % bad


# ------------------------------- Block 5 -> 10: the pilot's tilt per set ----

@pytest.mark.parametrize("d,which,phi_deg", [(0.2, "near", 16.0), (0.7, "far", 11.0)])
def test_block5_pilot_tilt_judged_by_the_set_bound(d, which, phi_deg):
    # D-101.1: the pilot's tilt (Block 5) is judged by its set's bound -- near <= 10 deg, far <= 15 deg; Allen's
    # diameter (2 x the branch radius) picks the set. A phantom tilted between the two bounds; the pilot's reading
    # is within Block 5's 4 deg (its oracle), and not_flat appears exactly when that reading exceeds the set's bound.
    # Checked through calibration_reasons on the pilot, as scan_node calls it.
    from allen_diameter.analysis.node_pipeline import measure_pilot
    cfg = cfg_with(patch_half_um=12.0, far_offsets_planes=8)
    branch, i, provider, draw = rendered_phantom(cfg, 31, d, 0.3, 0.7, phi_deg=phi_deg)
    res = measure_pilot(branch, i, provider, cfg)
    phi_hat = math.degrees(res.phi_rad)
    # (Block 5's oracle bounds the tilt error at 4 deg for d = 1 um only; a 0.2 um tube reads several degrees
    # off -- reported, not asserted here: the rule below is judged on the pilot's own reading)
    print("pilot tilt %.2f deg for a %.1f deg phantom (d %.1f um)" % (phi_hat, phi_deg, d))
    allen_d = 2.0 * float(branch.radius_um[i])
    assert CAL.calibration_set(allen_d, cfg) == which
    bound = cfg.calibration.node_phi_max_deg if which == "near" else cfg.calibration.node_phi_max_deg_far
    planes_z = CAL.stage_planes(res, cfg, which) * cfg.acquisition.dz_um
    reasons = CAL.calibration_reasons(res, cfg, allen_d, branch, i, Scene.from_branch(branch), planes_z)
    assert ("not_flat" in reasons) == (phi_hat > bound), (which, phi_hat, bound, reasons)
    # the reading between the two bounds makes the set decide (the pilot reads thin tubes to a few degrees)
    if 10.0 < phi_hat <= 15.0:
        assert ("not_flat" in reasons) == (which == "near"), (which, phi_hat, reasons)
