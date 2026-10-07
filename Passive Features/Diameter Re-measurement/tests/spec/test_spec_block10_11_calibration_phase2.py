"""Blocks 10 (kernel calibration) and 11 (Phase II tools): oracles from SPEC.md Blocks 10 and 11."""
import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from allen_diameter.config import default_config
from allen_diameter.analysis import calibration as CAL, camera_fit, registration, survey

CFG = default_config()


def test_dip_statistics_on_sampled_gaussian():
    s, a, vc, B = 0.2, 30.0, 0.05, 200.0
    v = np.arange(-400, 401) * 0.01
    I = B - a * np.exp(-0.5 * ((v - vc) / s) ** 2)
    cal = dataclasses.replace(CFG.calibration, window_half_um=10 * s)     # window of +-10 s (tails < 1e-21)
    m = dataclasses.replace(CFG.measure, focus_smooth_px=0.0)
    out = CAL.dip_statistics(v, I, B, m, cal)
    assert out["status"] == "ok"
    assert_allclose(out["omega_um2"], s * s, rtol=1e-9)          # least squares to 1e-12 tolerances
    assert_allclose(out["area_gl_um"], a * s * math.sqrt(2 * math.pi), rtol=1e-9)
    assert_allclose(out["vw_um2"], s * s, rtol=1e-9)
    flat = CAL.dip_statistics(v, np.full(v.size, B), B, m, cal)
    assert flat["status"] == "no_dip"


def _scans(G_of, nodes=6, seed=0, P=3, dz=0.28):
    rng = np.random.default_rng(seed)
    scans, truth = [], []
    for i in range(nodes):
        kstar = 10 + i
        zsub = kstar * dz + rng.uniform(-0.1, 0.1)
        zax = zsub + rng.uniform(-0.05, 0.05)
        c = rng.uniform(0.01, 0.03)
        ks = np.arange(kstar - P, kstar + P + 1)
        z = ks * dz
        om = c + G_of(zax - z)
        n = ks.size
        scans.append(CAL.PlaneScan(node_id=i, d_hat_um=0.25, phi_rad=0.0, alpha_hat=0.2, k_star=kstar, z_sub_um=zsub,
                                   ks=ks, z_um=z, omega_um2=om, vw_um2=om, area_gl_um=np.ones(n),
                                   depth_gl=np.ones(n), B_bar=np.full(n, 200.0), status=("ok",) * n))
        truth.append((c, zax))
    return scans, truth


def test_fit_growth_exact_linear_family():
    knots = np.arange(5) * 0.28
    Gk = np.array([0.0, 0.01, 0.05, 0.12, 0.2])
    G_of = lambda d: np.interp(np.abs(d), knots, Gk)          # the linear family on the fit's knots
    scans, truth = _scans(G_of)
    cfg = dataclasses.replace(CFG, calibration=dataclasses.replace(CFG.calibration, growth_interp="linear"))
    g = CAL.fit_growth(scans, cfg)
    assert_allclose(g.knots_um, knots, atol=1e-12)
    assert_allclose(g.g_um2, Gk, atol=1e-9)
    assert_allclose(g.c_um2, [t[0] for t in truth], atol=1e-9)
    assert_allclose(g.z_ax_um, [t[1] for t in truth], atol=1e-9)
    # a constant added to every omega moves only the c_i
    sc2 = [dataclasses.replace(s, omega_um2=s.omega_um2 + 0.5) for s in scans]
    g2 = CAL.fit_growth(sc2, cfg)
    assert_allclose(g2.g_um2, Gk, atol=1e-9) and None is None
    assert_allclose(g2.c_um2, g.c_um2 + 0.5, atol=1e-9)
    k = CAL.kernel_from_growth(g, CFG.renderer)
    assert_allclose(k.kernel_table_sigma_um, np.sqrt(CFG.renderer.sigma_r0_um ** 2 + Gk), rtol=1e-9)
    assert k.kernel_table_delta_um[0] == 0.0
    with pytest.raises(ValueError):
        CAL.fit_growth(scans[:1], cfg)
    neg = dataclasses.replace(g, g_um2=np.array([0.0, -0.01, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        CAL.kernel_from_growth(neg, CFG.renderer)


def test_registration_category():
    rc = registration.registration_category
    assert rc("ON THE PROCESS (lateral offset +0.00 um)") == "ON"
    assert rc("ON THE PROCESS (lateral offset +0.10 um) [and focus is +1.4 um off in z: probably a DIFFERENT process]") == "ON"
    assert rc("ALONGSIDE: a ridge +0.80 um to the side -- snap needed") == "ALONGSIDE"
    assert rc("FAR: nearest matching ridge +3.00 um away") == "FAR"
    assert rc("NOT ON A VISIBLE PROCESS (no peak): wrong structure") == "NOT_ON"
    assert rc("") == "" and rc(None) == ""
    assert rc("ON") == "ON" and rc("NOT_ON") == "NOT_ON" and rc("something else") == "UNKNOWN"
    assert registration.different_process("x [probably a DIFFERENT process]")


def test_clipped_sd():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(100, 2, 5000), [10.0, 250.0]]
    med = np.median(x)
    rsd = max(1.0, 1.4826 * np.median(np.abs(x - med)))
    keep = x[np.abs(x - med) <= 5 * rsd]
    assert camera_fit.clipped_sd(x) == pytest.approx(np.std(keep, ddof=1), rel=1e-12)
    assert camera_fit.clipped_sd(np.full(50, 7.0)) == 0.0


def test_summarize_percentiles():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(60):
        ins = bool(i % 3)
        rows.append(dict(in_S=ins, reject="" if ins else "faint", fit_status="converged",
                         d_hat_um=rng.uniform(0.3, 2), mu_hat_per_um=rng.uniform(0.2, 3), alpha_hat=rng.uniform(0, 1.5),
                         allen_radius_um=0.4, phi_rad=rng.uniform(0, 1), calibration_node=False))
    out = survey.summarize(rows, CFG)
    mu = np.array([r["mu_hat_per_um"] for r in rows if r["in_S"]])
    assert out["n_in_S"] == 40 and out["reject_counts"] == {"faint": 20}
    assert out["mu_hat_per_um"]["p10"] == pytest.approx(np.percentile(mu, 10))
    assert out["suggested_phantom_mu_range_per_um"] == [pytest.approx(np.percentile(mu, 10)),
                                                         pytest.approx(np.percentile(mu, 90))]
    dark = sum(1 for r in rows if r["alpha_hat"] > 1.0) / 60
    assert out["dark_share_of_converged"] == pytest.approx(dark)
