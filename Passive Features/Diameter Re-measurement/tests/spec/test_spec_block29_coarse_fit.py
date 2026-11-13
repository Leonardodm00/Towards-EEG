"""Spec-derived tests, Block 29 (the coarse refinement) -- tester 2026-11-13.

Oracles (SPEC.md Block 29): with a closed-form predict from
calibration_scene.shadow_depths (a flat tube shifted by v0, kernel at
|z_k - dz|): noise-free recovery of the four parameters from a start 30 %
off [relative 1e-6; 1e-6 um on the shifts]; 100 seeded noisy draws (SD 3 gl,
B = 210): mean d within 3 SE; a truth outside the bounds ends at_bound with
the name; a NaN predict ends failed without raising; inputs unchanged;
determinism; cost = sum w r^2 and rms unweighted at the returned point (by
hand); offset_xyz by hand; the ValueErrors.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import nc_helpers as H
from allen_diameter.analysis import coarse_fit as CF
from allen_diameter.analysis.calibration_scene import shadow_depths

B = 210.0
KS = np.array([-1, 0, 1, 2])


def make(cfg):
    dz = cfg.acquisition.dz_um
    v = np.arange(-26, 27) * cfg.measure.profile_step_um

    def predict(d, mu, v0, dzz):
        return np.stack([shadow_depths(np.abs(v - v0), 0.5 * d, mu, float(H.ref_sigma_r(k * dz - dzz, cfg.renderer)), 1.0)
                         for k in KS])
    return v, predict


TRUTH = (0.8, 1.2, 0.15, 0.10)


def test_noise_free_recovery_and_cost():
    cfg = H.cfg_with()
    v, pred = make(cfg)
    I = B * (1 - pred(*TRUTH))
    th0 = (0.8 * 1.3, 1.2 * 0.7, 0.15 * 1.3, 0.10 * 0.7)
    out = CF.refine_node(I, np.full(KS.size, B), v, pred, th0, cfg)
    print(out)
    assert out["status"] == "converged"
    assert_allclose([out["d_um"], out["mu_per_um"]], TRUTH[:2], rtol=1e-6)
    assert_allclose([out["v0_um"], out["dz_um"]], TRUTH[2:], atol=1e-6)
    # cost (weighted) and rms (unweighted) at the returned point, by hand
    w = np.exp(-np.abs(v) / cfg.measure.sgt_lambda_um)
    r = I - B * (1 - pred(out["d_um"], out["mu_per_um"], out["v0_um"], out["dz_um"]))
    assert_allclose(out["cost"], float(np.sum(w[None, :] * r ** 2)), rtol=1e-9, atol=1e-18)
    assert_allclose(out["rms_gl"], float(np.sqrt(np.mean(r ** 2))), rtol=1e-9, atol=1e-15)


def test_noisy_unbiased_and_deterministic():
    cfg = H.cfg_with()
    v, pred = make(cfg)
    I0 = B * (1 - pred(*TRUTH))
    rng = np.random.default_rng(2911)
    ds = []
    for _ in range(100):
        I = I0 + rng.normal(0, 3.0, size=I0.shape)
        Ic = I.copy()
        out = CF.refine_node(I, np.full(KS.size, B), v, pred, TRUTH, cfg)
        assert_array_equal(I, Ic)
        ds.append(out["d_um"])
    ds = np.asarray(ds)
    se = ds.std(ddof=1) / np.sqrt(ds.size)
    z = (ds.mean() - TRUTH[0]) / se
    print("mean d - truth = %.3g SE" % z)
    assert abs(z) <= 3.0
    a = CF.refine_node(I0 + 1.0, np.full(KS.size, B), v, pred, TRUTH, cfg)
    b = CF.refine_node(I0 + 1.0, np.full(KS.size, B), v, pred, TRUTH, cfg)
    assert a == b


def test_at_bound_and_failed():
    cfg = H.cfg_with()
    v, pred = make(cfg)
    I = B * (1 - pred(0.8, 1.2, 1.6, 0.0))              # v0 = 1.6 beyond the +-1 um box
    out = CF.refine_node(I, np.full(KS.size, B), v, pred, (0.8, 1.2, 0.5, 0.0), cfg)
    assert out["status"] == "at_bound" and "v0_um" in out["at_bound"]
    out = CF.refine_node(I, np.full(KS.size, B), v, lambda *a: np.full((KS.size, v.size), np.nan), TRUTH, cfg)
    assert out["status"] == "failed" and np.isnan(out["d_um"])
    calls = {"n": 0}

    def flaky(d, mu, v0, dzz):
        calls["n"] += 1
        s = pred(d, mu, v0, dzz)
        return s if calls["n"] < 3 else s * np.nan
    out = CF.refine_node(I, np.full(KS.size, B), v, flaky, TRUTH, cfg)
    assert out["status"] == "failed"


def test_offset_xyz_and_errors():
    yh = np.array([-0.6, 0.8])
    assert_array_equal(CF.offset_xyz(0.25, -0.1, yh), [0.25 * -0.6, 0.25 * 0.8, -0.1])
    cfg = H.cfg_with()
    v, pred = make(cfg)
    I = B * (1 - pred(*TRUTH))
    with pytest.raises(ValueError):
        CF.refine_node(I, np.full(KS.size + 1, B), v, pred, TRUTH, cfg)
    with pytest.raises(ValueError):
        CF.refine_node(I, np.full(KS.size, 0.0), v, pred, TRUTH, cfg)
    with pytest.raises(ValueError):
        CF.refine_node(I, np.full(KS.size, B), v, pred, (7.0, 1.0, 0.0, 0.0), cfg)
    with pytest.raises(ValueError):
        CF.refine_node(I, np.full(KS.size, B), v[:-1], pred, TRUTH, cfg)
