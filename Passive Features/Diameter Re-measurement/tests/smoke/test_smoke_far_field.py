"""Smoke test for the far-field calibration -- Block 15 in specs/SPEC.md
(S2b; D-026.1, D-074 to D-081; 2026-10-23).

Checks
    test_known_answer   on a sampled Gaussian dip the iterative core fit
                        returns omega = s^2 to 1e-9 and stops at pass 2; with
                        the stop disabled and N = 2 it equals a two-pass fit
                        written here independently (the S2b checks' S2) on
                        noisy draws; se_model on a Gaussian of SD s equals the
                        Gauss-Newton SE of a core fit to that Gaussian plus
                        white noise, averaged over draws, within 15 %
    test_reference      exact (F1) data (omega = c_j + (sigma_0 + gamma x)^2 on
                        planes |k| >= 5, both sides, axes off-plane) return
                        gamma, c and s to 1e-8 by the unweighted and the IRLS
                        fits; the per-sign fit returns gamma_+ = gamma_-; the
                        quadratic fit returns gamma_2 = 0 to 1e-6; Phase I:
                        far-set phantoms (d = 0.7 um, flat, mu d 0.35, the
                        camera chain without noise, +-8 planes on a 12 um
                        patch) rendered with gamma = 0.79 and 1.21 return
                        gamma_hat within 10 % through scan_node
    test_convergence    no discretisation step: skipped
    test_invariants     a constant added to every omega moves only the c_j;
                        shifting every z_p,j by u moves the s_j by -u; the SNR
                        rule drops the farthest planes as RMS grows; the anchor
                        sensitivity is negative and below 1 per um in size
    test_contract       FarSelection and FarFit fields and shapes; the far
                        scans' JSON round trip; save_far_fit's table ends at
                        0.84 um with the fitted slope and validate() passes;
                        the command line synthetic --set far -> far-fit (with
                        a near calibration from synthetic --set near -> fit)
    test_determinism    the same selection and seed give bit-identical fits
                        and bootstrap draws
    test_edge_cases     a flat profile (no_dip); a width at the bound
                        (at_bound); one far node (ValueError); gamma at
                        gamma_max (success False); a node without an area
                        dropped and named; a near scan refused by the
                        selection

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_far_field.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
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
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for _p in (SRC, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter.analysis import calibration as CAL  # noqa: E402
from allen_diameter.analysis import far_field as FF  # noqa: E402
from allen_diameter.analysis.node_pipeline import Branch  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import calibration_io as CIO  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402

SEED = 20261023
REPORT_PACKAGES = ("numpy", "scipy")
FW = math.sqrt(8.0 * math.log(2.0))


# --------------------------------------------------------------- helpers ---

def cfg_with(base=None, **cal):
    cfg = default_config() if base is None else base
    return dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, **cal))


def far_cfg(gamma=0.79, noise=False):
    """A reduced far-field configuration for rendering: 12 um patch, +-8 planes, few bootstrap resamples. Without
    noise the goodness of fit (Block 10.G, D-115.3) is off: the background's noise SD is then the quantization's,
    below every core fit's deterministic misfit."""
    base = default_config()
    rc = dataclasses.replace(base.renderer, kernel_continuation_slope=gamma,
                             **({} if noise else dict(noise_sd_gl=0.0, jpeg=False)))
    return cfg_with(dataclasses.replace(base, renderer=rc), patch_half_um=12.0, far_offsets_planes=8,
                    bootstrap_resamples=12, gof_check=bool(noise))


def f1_selection(cfg, gamma, sigma0, rng, J=6, noise=0.0, k_max=16, c_range=(-0.02, 0.03), u_sd=0.0):
    """Exact (F1) data as a FarSelection: J nodes with axes z_ax ~ U(-dz/2, dz/2), pilots z_p = z_ax + N(0, u_sd),
    planes 5 <= |k| <= k_max on both sides, omega = c_j + (sigma0 + gamma x)^2 (+ noise), RMS 1 gl, area 35 gl um."""
    dz, d0 = cfg.acquisition.dz_um, cfg.calibration.table_max_delta_um
    ids, areas, rmss, zps, counts = [], [], [], [], []
    obs = {k: [] for k in ("node", "k", "z", "omega", "rms", "sp")}
    truth = []
    for j in range(J):
        z_ax = rng.uniform(-0.5 * dz, 0.5 * dz)
        c = rng.uniform(*c_range)
        z_p = z_ax + (rng.normal(0.0, u_sd) if u_sd > 0 else 0.0)
        ks = np.array([k for k in range(-k_max, k_max + 1) if abs(k) >= 5])
        zk = ks * dz
        x = np.abs(z_ax - zk) - d0
        om = c + (sigma0 + gamma * x) ** 2 + (rng.normal(0.0, noise, ks.size) if noise > 0 else 0.0)
        ids.append(j); areas.append(35.0); rmss.append(1.0); zps.append(z_p); counts.append(ks.size)
        obs["node"].append(np.full(ks.size, j)); obs["k"].append(ks); obs["z"].append(zk); obs["omega"].append(om)
        obs["rms"].append(np.ones(ks.size)); obs["sp"].append(K.sigma_r(np.abs(z_p - zk), cfg.renderer))
        truth.append((c, z_ax - z_p))
    cat = {k: np.concatenate(v) for k, v in obs.items()}
    sel = FF.FarSelection(np.array(ids), np.array(areas), np.array(rmss), np.array(zps), cat["node"].astype(int),
                          cat["k"].astype(int), cat["z"], cat["omega"], cat["rms"], cat["sp"], {}, 0, 0,
                          np.array(counts))
    return sel, np.array([t[0] for t in truth]), np.array([t[1] for t in truth])


def two_pass_reference(v, w, depth, v_peak, sigma_p, cal):
    """The S2b checks' two-pass statistic, written independently of the module (the same window rule and bounds)."""
    slo, shi = cal.far_core_width_bounds[0] * sigma_p, cal.far_core_width_bounds[1] * sigma_p
    c_cen, s_cur = float(v_peak), float(sigma_p)
    for _ in range(2):
        W = cal.core_fit_fwhm_factor * FW * s_cur
        win = np.abs(v - c_cen) <= W

        def resid(q, vv=v[win], ww=w[win]):
            return q[0] * np.exp(-0.5 * ((vv - q[1]) / q[2]) ** 2) - ww
        r = least_squares(resid, [max(depth, 1e-3), c_cen, min(max(s_cur, slo * 1.01), shi * 0.99)],
                          bounds=([0.0, c_cen - 0.5 * W, slo], [np.inf, c_cen + 0.5 * W, shi]), method="trf",
                          x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=200)
        c_cen, s_cur = float(r.x[1]), float(r.x[2])
    return s_cur ** 2


def render_far_phantoms(cfg, specs, seed):
    """scan_node on rendered far phantoms (d 0.7 um, flat); specs: list of (mu, theta, cz). Returns the far scans."""
    scans = []
    pad = int(math.ceil(cfg.renderer.pad_um / cfg.acquisition.res0_um))
    for n, (mu, theta, cz) in enumerate(specs):
        d = 0.7
        tube = G.Tube((0.02, -0.03, cz), 0.5 * d, 0.0, theta, 1.0, cfg.renderer.U_um, cfg.renderer.end_cut)
        t = G.axis_direction(0.0, theta)
        br = Branch.from_points(np.array([0.02, -0.03, cz])[None, :] + np.arange(-10, 11)[:, None] * 1.18 * t[None, :],
                                0.5 * d, ids=np.arange(21) + 100 * n)
        rng = np.random.default_rng([seed, n])

        def provider(left, top, width, height, k_lo, k_hi, tube=tube, mu=mu, rng=rng):
            return R.synthetic_block(tube, mu, np.arange(k_lo, k_hi + 1), left, top, width, height, cfg, rng, pad)
        res, scan, reasons = CAL.scan_node(br, 10, provider, cfg)
        assert scan is not None and scan.set == "far", (n, reasons)
        scans.append(dataclasses.replace(scan, node_id=n))
    return scans


FAR_SPECS = ((0.5, 0.4, 0.04), (0.35, 1.3, -0.09), (0.65, 2.2, 0.11))
_PHANTOM_CACHE = {}


def far_phantom_scans(gamma):
    """Rendered once per gamma (about 90 s each) and shared by the checks."""
    if gamma not in _PHANTOM_CACHE:
        _PHANTOM_CACHE[gamma] = render_far_phantoms(far_cfg(gamma), FAR_SPECS, SEED)
    return _PHANTOM_CACHE[gamma]


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = default_config()
    cal, m = cfg.calibration, cfg.measure
    v = FF.far_offsets(cfg)
    a, c, s = 6.0, 0.3, 1.7
    w = a * np.exp(-0.5 * ((v - c) / s) ** 2)
    # a sampled Gaussian: omega = s^2 to 1e-9 (float64 least squares on exact data), stopped at pass 2
    out = FF.core_fit_iterative(v, w, a, c, 1.4, cal, m)
    assert out["status"] == "ok" and abs(out["omega_um2"] / s ** 2 - 1.0) <= 1e-9 and out["passes"] == 2, out
    assert abs(out["v_c_um"] - c) <= 1e-9 and abs(out["amp_gl"] - a) <= 1e-9 and out["rms_gl"] <= 1e-9
    st = FF.far_plane_statistics(v, 210.0 - w, 210.0, 1.4, m, cal)
    # the depth is read off the profile smoothed by sigma_p / (2 step) = 6.1 samples (0.70 um): a Gaussian of SD s
    # loses the factor s / sqrt(s^2 + 0.70^2) = 0.925 at its peak, so the start depth is 5.55; omega is exact
    assert st["status"] == "ok" and abs(st["omega_um2"] / s ** 2 - 1.0) <= 1e-9, st
    assert abs(st["depth_gl"] - a * s / math.sqrt(s * s + (6.1 * m.profile_step_um) ** 2)) <= 0.02 * a, st["depth_gl"]
    # the line-end zone starts 2 FWHM = 8 um out, where this Gaussian's tail is 6 exp(-(8 / 1.7)^2 / 2) = 1e-4
    assert math.isfinite(st["line_end_gl"]) and abs(st["line_end_gl"]) <= 1e-3, st["line_end_gl"]
    # with the stop disabled and N = 2: the two-pass statistic of the S2b checks, on noisy draws (same window rule)
    cal2 = dataclasses.replace(cal, far_core_max_passes=2, far_core_stop_tol=0.0)
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for _ in range(20):
        wn = w + rng.normal(0.0, 1.0, v.size)
        out2 = FF.core_fit_iterative(v, wn, a, c, 1.4, cal2, m)
        ref = two_pass_reference(v, wn, a, c, 1.4, cal)
        if out2["status"] in ("ok", "not_converged"):
            worst = max(worst, abs(out2["omega_um2"] / ref - 1.0))
    assert worst <= 1e-6, worst     # the same solver, start, bounds and windows: equal to the solver's tolerance
    # se_model against the Gauss-Newton SE of core fits to the Gaussian plus white noise (SD 1 gl), averaged over draws
    se_pred = float(FF.se_model(a, s, 1.0, m.profile_step_um, cal.far_profile_half_um)[0])
    ses, oms = [], []
    for _ in range(200):
        wn = w + rng.normal(0.0, 1.0, v.size)
        o = FF.core_fit_iterative(v, wn, a, c, s, cal, m)
        if o["status"] == "ok":
            ses.append(o["se_omega_um2"]); oms.append(o["omega_um2"])
    assert len(ses) >= 180, len(ses)
    # within 15 %: the mean fitted SE and the empirical SD of omega both estimate the same Fisher bound (200 draws:
    # the SD's own SE is 5 %)
    assert abs(np.mean(ses) / se_pred - 1.0) <= 0.15 and abs(np.std(oms, ddof=1) / se_pred - 1.0) <= 0.15, \
        (se_pred, np.mean(ses), np.std(oms, ddof=1))


def test_reference():
    cfg = default_config()
    s0 = float(K.sigma_r(0.84, cfg.renderer))
    for gamma in (0.79, 1.21):
        sel, c_true, s_true = f1_selection(cfg, gamma, s0, np.random.default_rng(SEED), J=6)
        for weights in ("none", "model"):
            fit = FF.fit_far_slope(sel, s0, cfg, weights=weights)
            assert fit.success and abs(fit.gamma - gamma) <= 1e-8, (gamma, weights, fit.gamma, fit.message)
            assert np.max(np.abs(fit.c_um2 - c_true)) <= 1e-8 and np.max(np.abs(fit.s_um - s_true)) <= 1e-8, weights
        fs = FF.fit_far_slope(sel, s0, cfg, per_sign=True)
        assert abs(fs.gamma - gamma) <= 1e-7 and abs(fs.gamma_minus - gamma) <= 1e-7, (fs.gamma, fs.gamma_minus)
        fq = FF.fit_far_slope(sel, s0, cfg, quadratic=True)
        assert abs(fq.gamma - gamma) <= 1e-6 and abs(fq.gamma2) <= 1e-6, (fq.gamma, fq.gamma2)
    # Phase I: rendered far phantoms through scan_node (reduced configuration), gamma within 10 % without noise
    for gamma in (0.79, 1.21):
        cfg_r = far_cfg(gamma)
        scans = far_phantom_scans(gamma)
        sel = FF.select_far_planes(scans, cfg_r)
        assert sel.node_ids.size == 3 and not sel.dropped, sel.dropped
        s0r = float(K.sigma_r(0.84, cfg_r.renderer))
        fit = FF.fit_far_slope(sel, s0r, cfg_r)
        assert fit.success and abs(fit.gamma / gamma - 1.0) <= 0.10, (gamma, fit.gamma, fit.message)


def test_convergence():
    raise unittest.SkipTest("no discretisation step in Block 15: the quadrature-free fits are checked exactly in test_reference")


def test_invariants():
    cfg = default_config()
    s0 = float(K.sigma_r(0.84, cfg.renderer))
    gamma = 0.9
    sel, c_true, s_true = f1_selection(cfg, gamma, s0, np.random.default_rng(SEED + 1), J=6, noise=0.02, u_sd=0.05)
    fit = FF.fit_far_slope(sel, s0, cfg)
    # a constant on every omega moves only the c_j: exactly for the unweighted fit; the model weights read the
    # predicted core width sqrt(c_j + sigma_r^2), which the constant changes, so the IRLS fit moves by O(1e-3)
    shifted = dataclasses.replace(sel, omega_um2=sel.omega_um2 + 0.3)
    fu, fu2 = FF.fit_far_slope(sel, s0, cfg, weights="none"), FF.fit_far_slope(shifted, s0, cfg, weights="none")
    assert abs(fu2.gamma - fu.gamma) <= 1e-6 and np.max(np.abs(fu2.c_um2 - fu.c_um2 - 0.3)) <= 1e-6 \
        and np.max(np.abs(fu2.s_um - fu.s_um)) <= 1e-6
    fit2 = FF.fit_far_slope(shifted, s0, cfg)
    assert abs(fit2.gamma / fit.gamma - 1.0) <= 1e-3 and np.max(np.abs(fit2.c_um2 - fit.c_um2 - 0.3)) <= 5e-3
    # shifting every pilot depth by u moves every s_j by -u (within the bound)
    u = 0.05
    moved = dataclasses.replace(sel, z_p_um=sel.z_p_um + u)
    fit3 = FF.fit_far_slope(moved, s0, cfg)
    assert abs(fit3.gamma - fit.gamma) <= 1e-6 and np.max(np.abs(fit3.s_um - (fit.s_um - u))) <= 1e-6, fit3.s_um - fit.s_um
    # the SNR rule drops the farthest planes as the RMS grows: a far node with a rising RMS keeps fewer planes
    scans = far_phantom_scans(0.79)
    cfg_r = far_cfg(0.79)
    sel_a = FF.select_far_planes(scans, cfg_r)
    # scale every node's RMS so that the rule's threshold lands on the predicted dip at |k - k*| = 7: planes 8 drop
    noisier = []
    for sc, A, rms in zip(scans, sel_a.area_gl_um, sel_a.rms_gl):
        sp7 = float(K.sigma_r(7 * cfg_r.acquisition.dz_um, cfg_r.renderer))
        factor = (A / (math.sqrt(2.0 * math.pi) * sp7)) / (cfg_r.calibration.far_min_snr * rms)
        noisier.append(dataclasses.replace(sc, rms_gl=sc.rms_gl * factor * 1.001))
    sel_b = FF.select_far_planes(noisier, cfg_r)
    kept_a = int(sel_a.n_planes_per_node.sum())
    kept_b = int(sel_b.n_planes_per_node.sum()) if sel_b.node_ids.size else 0
    assert kept_b < kept_a and sel_b.n_snr_dropped > sel_a.n_snr_dropped, (kept_a, kept_b, sel_b.dropped)
    if sel_b.node_ids.size:
        far_a = np.max(np.abs(sel_a.obs_k)) if sel_a.obs_k.size else 0
        far_b = np.max(np.abs(sel_b.obs_k))
        assert far_b <= far_a
    # the anchor sensitivity: a higher anchor needs a smaller slope; its size below 1 per um (S2b: -0.27 to -0.46)
    dg = FF.anchor_sensitivity(sel, s0, cfg)
    assert -1.0 < dg < 0.0, dg


def test_contract():
    cfg = default_config()
    s0 = float(K.sigma_r(0.84, cfg.renderer))
    sel, _, _ = f1_selection(cfg, 0.79, s0, np.random.default_rng(SEED + 2), J=5, noise=0.01)
    n = sel.omega_um2.size
    for name in ("obs_node", "obs_k", "z_um", "omega_um2", "obs_rms_gl", "sigma_p_um"):
        assert getattr(sel, name).shape == (n,), name
    assert sel.node_ids.shape == (5,) == sel.area_gl_um.shape == sel.rms_gl.shape == sel.z_p_um.shape
    sub = sel.subset([0, 0, 3])
    assert sub.node_ids.tolist() == [0, 0, 3] and sub.omega_um2.size == 3 * (n // 5) and set(sub.obs_node.tolist()) == {0, 1, 2}
    fit = FF.fit_far_slope(sel, s0, cfg)
    for name in ("obs_node", "obs_k", "obs_offset", "delta_um", "residual_um2", "weights"):
        assert getattr(fit, name).shape == (n,), name
    assert fit.c_um2.shape == (5,) == fit.s_um.shape and math.isnan(fit.gamma_minus) and math.isnan(fit.gamma2)
    assert fit.rounds == cfg.calibration.far_irls_rounds and isinstance(fit.rms_by_offset(), dict)
    se, n_ok, draws = FF.bootstrap_gamma(sel, s0, cfg, np.random.default_rng(SEED), n_resamples=8)
    assert draws.shape == (n_ok,) and 0 < n_ok <= 8 and (math.isnan(se) or se >= 0.0)
    summary = FF.far_fit_summary(sel, fit, FF.fit_far_slope(sel, s0, cfg, per_sign=True),
                                 FF.fit_far_slope(sel, s0, cfg, quadratic=True), se, n_ok, -0.3, s0, 0.01, cfg, 1)
    json.dumps(summary)           # strict JSON: no NaN
    assert summary["n_nodes"] == 5 and summary["interval"][0] < summary["gamma"] < summary["interval"][1]
    # the far scans' JSON round trip and save_far_fit's table
    scans = far_phantom_scans(0.79)
    back = CIO.scan_from_dict(json.loads(json.dumps(CIO.scan_to_dict(scans[0]))))
    for name in ("omega_um2", "se_omega_um2", "amp_gl", "rms_gl", "line_end_gl", "passes", "n_win", "w_gl", "bhat_gl",
                 "v_um", "b_coef"):
        assert np.array_equal(getattr(back, name), getattr(scans[0], name), equal_nan=True), name
    assert back.set == "far" and back.status == scans[0].status
    with tempfile.TemporaryDirectory() as tmp:
        growth_scans = None
        # a production kernel: the default table cut at 0.84 um with the fitted slope
        kernel = dataclasses.replace(cfg.renderer, kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),
                                     kernel_table_sigma_um=(0.08, 0.122, 0.438, 0.603), kernel_continuation="linear",
                                     kernel_continuation_slope=fit.gamma)
        path = os.path.join(tmp, "far_fit.json")
        CIO.save_far_fit(path, summary, kernel)
        obj = CIO.load_calibration(path)
        assert obj["kernel"]["kernel_table_delta_um"][-1] == 0.84 and obj["kernel"]["kernel_continuation_slope"] == fit.gamma
        dataclasses.replace(cfg, renderer=kernel).validate()
        # the command line: near phantoms -> fit (the anchor), far phantoms -> far-fit, on the reduced configuration
        cfg_cli = far_cfg(0.79)
        # 2 near and 3 far phantoms are below D-088.3's minimum of 10 per set (Block 19.2): lowered here so that the
        # production path runs; test_smoke_s3c.py checks the minimum itself
        cfg_cli = dataclasses.replace(cfg_cli, calibration=dataclasses.replace(cfg_cli.calibration, calib_min_nodes_per_set=2))
        cj = os.path.join(tmp, "config.json")
        with open(cj, "w") as f:
            json.dump(dict(json.loads(cfg_cli.to_json()), calibration=dataclasses.asdict(cfg_cli.calibration)), f)
        run = [sys.executable, str(WS / "scripts" / "calibrate_kernel.py")]
        near, far, calp, farp = (os.path.join(tmp, x) for x in ("scans_near.json", "scans_far.json", "cal.json",
                                                                 "far_fit.json"))
        p1 = subprocess.run(run + ["synthetic", "--set", "near", "--n", "2", "--seed", "5", "--out", near,
                                   "--config-json", cj], capture_output=True, text=True, timeout=900)
        assert p1.returncode == 0 and "2 of 2 phantoms scanned" in p1.stdout, (p1.stdout, p1.stderr)
        p2 = subprocess.run(run + ["fit", "--scans", near, "--out", calp, "--config-json", cj], capture_output=True,
                            text=True, timeout=600)
        assert p2.returncode == 0 and "anchor sigma_0_hat" in p2.stdout, (p2.stdout, p2.stderr)
        p3 = subprocess.run(run + ["synthetic", "--set", "far", "--n", "3", "--seed", "7", "--out", far,
                                   "--config-json", cj], capture_output=True, text=True, timeout=1200)
        assert p3.returncode == 0 and "3 of 3 phantoms scanned" in p3.stdout, (p3.stdout, p3.stderr)
        assert os.path.exists(os.path.join(tmp, "patches_far.npz")), os.listdir(tmp)
        p4 = subprocess.run(run + ["far-fit", "--scans", far, "--calibration", calp, "--out", farp, "--config-json", cj],
                            capture_output=True, text=True, timeout=600)
        assert p4.returncode == 0 and "far fit: gamma_hat" in p4.stdout, (p4.stdout, p4.stderr)
        obj = CIO.load_calibration(farp)
        assert obj["far_fit"]["n_nodes"] == 3 and obj["far_fit"]["success"], obj["far_fit"]
        assert ("kernel" in obj) and abs(obj["kernel"]["kernel_table_delta_um"][-1] - 0.84) <= 1e-9, obj.get("kernel_error")
        # the phantoms were rendered with the table's own law; the near table from two noise-free phantoms is only
        # a starting point (SPEC Block 10 status), so the slope is checked loosely here (test_reference is strict)
        assert abs(obj["far_fit"]["gamma"] / 0.79 - 1.0) <= 0.25, obj["far_fit"]["gamma"]
        del growth_scans


def test_determinism():
    cfg = default_config()
    s0 = float(K.sigma_r(0.84, cfg.renderer))
    sel, _, _ = f1_selection(cfg, 0.79, s0, np.random.default_rng(SEED + 3), J=5, noise=0.02)
    a, b = FF.fit_far_slope(sel, s0, cfg), FF.fit_far_slope(sel, s0, cfg)
    assert a.gamma == b.gamma and np.array_equal(a.c_um2, b.c_um2) and np.array_equal(a.weights, b.weights)
    d1 = FF.bootstrap_gamma(sel, s0, cfg, np.random.default_rng(7), n_resamples=6)[2]
    d2 = FF.bootstrap_gamma(sel, s0, cfg, np.random.default_rng(7), n_resamples=6)[2]
    assert np.array_equal(d1, d2)


def test_edge_cases():
    cfg = default_config()
    cal, m = cfg.calibration, cfg.measure
    v = FF.far_offsets(cfg)
    assert FF.far_plane_statistics(v, np.full(v.size, 210.0), 210.0, 1.0, m, cal)["status"] == "no_dip"
    # a dip far wider than the prior: the width stops at the upper bound 4 sigma_p
    wide = 8.0 * np.exp(-0.5 * (v / 3.0) ** 2)
    out = FF.core_fit_iterative(v, wide, 8.0, 0.0, 0.3, cal, m)
    assert out["status"] == "at_bound", out
    s0 = float(K.sigma_r(0.84, cfg.renderer))
    sel, _, _ = f1_selection(cfg, 0.79, s0, np.random.default_rng(SEED + 4), J=2)
    one = sel.subset([0])
    try:
        FF.fit_far_slope(one, s0, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("one far node must be refused")
    # gamma at gamma_max: data generated with a slope above the bound -> success False, at_bound True
    steep, _, _ = f1_selection(cfg, 8.0, s0, np.random.default_rng(SEED + 5), J=4)
    fit = FF.fit_far_slope(steep, s0, cfg)
    assert fit.at_bound and not fit.success and abs(fit.gamma - cal.far_gamma_max) <= 1e-6, fit.gamma
    # a node without an area (no ok plane in the area offsets) is dropped and named; a near scan is refused
    scans = far_phantom_scans(0.79)
    sc = scans[0]
    st = tuple("fit_failed" if 4 <= abs(k - sc.k_star) <= 6 else s for k, s in zip(sc.ks, sc.status))
    broken = dataclasses.replace(sc, status=st)
    sel2 = FF.select_far_planes([broken] + scans[1:], far_cfg(0.79))
    assert sel2.dropped == {int(sc.node_id): "no_area"} and sel2.node_ids.size == 2, sel2.dropped
    try:
        FF.select_far_planes([dataclasses.replace(sc, set="near")], far_cfg(0.79))
    except ValueError:
        return
    raise AssertionError("a near scan must be refused by the far selection")


# ---------------------------------------------------------------- runner ---

def test_goodness_of_fit():
    """Block 10.G (D-115.3) on the far scan: one rendered far phantom with the camera's noise, scanned twice from
    the same seed -- as it is, and with the plane at k* + 6 (a plane of the far fit) stubbed to a core-fit RMS above
    the bound: that plane alone is poor_fit, its width NaN, and select_far_planes does not take it."""
    cfg = far_cfg(0.79, noise=True)
    assert cfg.calibration.gof_check
    spec = [FAR_SPECS[0]]
    sc0 = render_far_phantoms(cfg, spec, SEED + 5)[0]
    assert sc0.bg_sd_gl is not None and sc0.bg_sd_gl.shape == sc0.ks.shape, sc0.bg_sd_gl
    ok0 = np.array(sc0.status) == "ok"
    assert np.all(np.isfinite(sc0.bg_sd_gl[ok0])) and np.all(sc0.bg_sd_gl[ok0] > 0)
    target = int(np.flatnonzero(sc0.offsets() == 6)[0])
    assert ok0[target], sc0.status
    saved, calls = FF.far_plane_statistics, []

    def stub(*a, **k):
        st = saved(*a, **k)
        calls.append(1)
        return dict(st, rms_gl=1e6) if len(calls) == target + 1 else st
    FF.far_plane_statistics = stub
    try:
        sc1 = render_far_phantoms(cfg, spec, SEED + 5)[0]
    finally:
        FF.far_plane_statistics = saved
    assert sc1.status[target] == "poor_fit", sc1.status
    assert math.isnan(float(sc1.omega_um2[target])) and math.isnan(float(sc1.se_omega_um2[target]))
    other = np.arange(sc1.ks.size) != target
    assert tuple(np.array(sc1.status)[other]) == tuple(np.array(sc0.status)[other])
    k_t = int(sc1.ks[target])
    sel0, sel1 = FF.select_far_planes([sc0], cfg), FF.select_far_planes([sc1], cfg)
    assert k_t in set(int(k) for k in sel0.obs_k) and k_t not in set(int(k) for k in sel1.obs_k), (k_t, sel1.obs_k)


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
