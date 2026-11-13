"""Spec-derived tests of Block 6 (phantom draws, the phantom branch, the
selection rule on results and on stored rows, the table coordinates, rows and
files) -- integration tester, run D1, 2026-11-13. Expected values: the draw
laws and generator keys of SPEC Blocks 6 / 14.2 recomputed from NumPy alone,
scipy.stats for the laws, and the selection rule written from the spec; the
code under test only produces the outputs that are checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block6_phantoms.py -q
The last test renders one phantom twice (about a minute on a free core).
"""
from __future__ import annotations

import dataclasses
import math
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

import spec_helpers_D1 as H

from allen_diameter import config as C
from allen_diameter.analysis import node_pipeline as NP
from allen_diameter.analysis import phantoms as PH
from allen_diameter.analysis import table as T
from allen_diameter.loading import table_io as TIO

CFG = C.default_config()
P = CFG.phantom
PX = CFG.acquisition.res0_um
DZ = CFG.acquisition.dz_um
NAN = float("nan")


# ------------------------------------------------------------- O6.1 -------

def ref_features(cfg, seed, g):
    p = cfg.phantom
    rng = np.random.default_rng([seed, g])
    lo, hi = p.d_range_um
    d = math.exp(rng.uniform(math.log(lo), math.log(hi))) if p.d_log_uniform else rng.uniform(lo, hi)
    phi = math.radians(rng.uniform(*p.phi_range_deg))
    theta = rng.uniform(*p.theta_range_rad)
    mlo, mhi = p.mu_range_per_um
    mu = math.exp(rng.uniform(math.log(mlo), math.log(mhi))) if p.mu_log_uniform else rng.uniform(mlo, mhi)
    return d, phi, theta, mu


def ref_nuisance(cfg, seed, g, r):
    rng = np.random.default_rng([seed, g, r])
    cx = rng.uniform(-PX / 2, PX / 2)
    cy = rng.uniform(-PX / 2, PX / 2)
    cz = rng.uniform(-DZ / 2, DZ / 2)
    return (cx, cy, cz), rng


def ref_radius(cfg, seed, g, r, d):
    rng = np.random.default_rng([seed, g, r, 1])
    n = 0
    while True:
        eps = rng.normal(0.0, cfg.phantom.radius_noise_sd_um)
        if d / 2 + eps > 0:
            return d / 2 + eps, n
        n += 1


@pytest.mark.parametrize("g,r", [(1, 1), (1, 2), (2, 1), (7, 10), (12345, 3)])
def test_draw_recomputed_from_numpy(g, r):
    assert P.subpixel_offset and P.axis_depth_jitter        # the default draws both
    seed = P.seed
    draw, gen = PH.draw_replicate(CFG, seed, g, r)
    d, phi, theta, mu = ref_features(CFG, seed, g)
    assert_allclose([draw.d_um, draw.phi_rad, draw.theta_rad, draw.mu_per_um], [d, phi, theta, mu],
                    rtol=1e-12, atol=0)                       # equivalent float formulas allowed
    (cx, cy, cz), rng = ref_nuisance(CFG, seed, g, r)
    assert_allclose([draw.cx_um, draw.cy_um, draw.cz_um], [cx, cy, cz], rtol=1e-12, atol=1e-18)
    # the returned generator is positioned after cx, cy, cz
    assert gen.random() == rng.random()
    rad, n = ref_radius(CFG, seed, g, r, draw.d_um)
    assert abs(draw.radius_um - rad) <= 1e-15 and draw.radius_redraws == n
    assert (draw.draw_index, draw.realization_index, draw.seed) == (g, r, seed)


def test_realizations_share_features_and_order_free():
    a = PH.draw_replicate(CFG, 5, 3, 2)[0]
    b = PH.draw_replicate(CFG, 5, 3, 1)[0]
    c = PH.draw_replicate(CFG, 5, 3, 2)[0]
    assert (a.d_um, a.phi_rad, a.theta_rad, a.mu_per_um) == (b.d_um, b.phi_rad, b.theta_rad, b.mu_per_um)
    assert (a.cx_um, a.cy_um, a.cz_um) != (b.cx_um, b.cy_um, b.cz_um)
    assert a == c                                            # whatever was drawn in between


@pytest.mark.parametrize("g,r", [(0, 1), (1, 0), (-1, 1)])
def test_index_zero_refused(g, r):
    with pytest.raises(ValueError):
        PH.draw_replicate(CFG, 5, g, r)


def test_laws_by_kolmogorov_smirnov():
    n = 500
    draws = [PH.draw_replicate(CFG, 99, g, 1)[0] for g in range(1, n + 1)]
    d = np.array([x.d_um for x in draws])
    phi = np.array([x.phi_rad for x in draws])
    th = np.array([x.theta_rad for x in draws])
    mu = np.array([x.mu_per_um for x in draws])
    cx = np.array([x.cx_um for x in draws])
    cz = np.array([x.cz_um for x in draws])
    lo, hi = np.log(P.d_range_um)
    mlo, mhi = np.log(P.mu_range_per_um)
    plo, phi_hi = np.radians(P.phi_range_deg)
    tests = [(np.log(d), lo, hi), (phi, plo, phi_hi), (th, *P.theta_range_rad), (np.log(mu), mlo, mhi),
             (cx, -PX / 2, PX / 2), (cz, -DZ / 2, DZ / 2)]
    for x, a, b in tests:
        assert np.all(x >= a) and np.all(x < b)              # upper end excluded
        assert stats.kstest(x, stats.uniform(loc=a, scale=b - a).cdf).pvalue > 1e-3


def test_radius_truncated_normal_and_redraw_count():
    d, s = 0.2, P.radius_noise_sd_um
    draws = [PH.draw_replicate(CFG, 4, g, 1, d_override_um=d)[0] for g in range(1, 1001)]
    eps = np.array([x.radius_um for x in draws]) - d / 2
    assert np.all(eps > -d / 2)
    tn = stats.truncnorm(a=(-d / 2) / s, b=np.inf, loc=0.0, scale=s)
    assert stats.kstest(eps, tn.cdf).pvalue > 0.01
    red = np.array([x.radius_redraws for x in draws], float)
    q = stats.norm.cdf(-d / (2 * s))
    mean_exp = q / (1 - q)                                   # geometric number of rejections
    se = math.sqrt(q / (1 - q) ** 2 / len(red))
    assert abs(red.mean() - mean_exp) < 4 * se
    zero = dataclasses.replace(CFG, phantom=dataclasses.replace(P, radius_noise_sd_um=0.0))
    z = PH.draw_replicate(zero, 4, 1, 1, d_override_um=0.7)[0]
    assert z.radius_um == 0.35 and z.radius_redraws == 0


# ------------------------------------------------------------- O6.2 -------

def test_phantom_branch_on_the_axis():
    draw, gen = PH.draw_replicate(CFG, 11, 3, 1)
    tube = PH.phantom_tube(draw, CFG)
    assert_allclose(tube.c, [draw.cx_um, draw.cy_um, draw.cz_um], atol=0)
    assert abs(tube.r - draw.d_um / 2) < 1e-15
    br = PH.phantom_branch(draw, CFG, gen)
    n = 2 * P.nodes_each_way + 1
    assert br.xyz_um.shape == (n, 3)
    a = H.axis_dir(draw.phi_rad, draw.theta_rad)
    c = np.array([draw.cx_um, draw.cy_um, draw.cz_um])
    q = br.xyz_um - c
    perp = q - np.outer(q @ a, a)
    assert np.max(np.linalg.norm(perp, axis=1)) < 1e-12            # on the axis (jitter 0)
    assert_allclose(np.linalg.norm(np.diff(br.xyz_um, axis=0), axis=1), P.phantom_node_step_um, atol=1e-12)
    assert_allclose(br.xyz_um[n // 2], c, atol=1e-12)               # the measured node is the middle one
    assert_array_equal(br.radius_um, draw.radius_um)
    assert_allclose(br.s_um, np.arange(n) * P.phantom_node_step_um, atol=1e-12)


# --------------------------------------------------------- O6.3, O6.4 -----

def make_result(**kw):
    base = {}
    for f in dataclasses.fields(NP.NodeResult):
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
            continue
        base[f.name] = None
    base.update(node_id=1, type=3, x_um=0.0, y_um=0.0, z_um=0.0, path_um=0.0, reg_verdict="",
                s_star_um=NAN, dz_star_um=NAN, k_star=0, z_sub_um=0.0, cx_um=0.0, cy_um=0.0, cz_um=0.0,
                theta_rad=0.0, phi_rad=0.1, steep=False, vertical=False, B_bar=200.0,
                B_bar_region="block_masked", d_hat_um=1.0, mu_hat_per_um=1.0, v0_hat_um=0.0,
                alpha_hat=1.0, fit_status="converged", flags=(), fit=None, focus_rule="gradient_energy",
                focus_planes=np.arange(3), focus_score=np.ones(3), focus_depth=np.ones(3), k_star_depth=0)
    base.update(kw)
    return NP.NodeResult(**{k: v for k, v in base.items() if k in {f.name for f in dataclasses.fields(NP.NodeResult)}})


ON = "ON THE PROCESS (lateral offset +0.00 um)"
FAR = "FAR: nearest matching ridge +2.00 um away -- treat as unregistered"


def ref_reasons(status, verdict, flags, sflags=C.SELECTION_FLAGS):
    """The rule of S on (status, verdict sentence, flags), written from SPEC
    Block 6 and D-096: the reasons' content (status first, then the
    registration, then the flags of S in the node's order)."""
    out = []
    if status != "converged":
        out.append("status")
    if verdict and not verdict.startswith("ON THE PROCESS") and verdict != "ON":
        out.append("registration")
    out += [f for f in flags if f in sflags]
    return out


CASES = [
    ("converged", "", ()),
    ("converged", "", ("faint", "dark", "steep", "vertical", "iter_max_rounds")),
    ("converged", ON, ("crossing",)),
    ("at_bound", "", ("bbar_few",)),
    ("failed", FAR, ("stack_edge", "faint", "profile_nan")),
    ("converged", FAR, ()),
    ("none", "", ("profile_nan",)),
]


def _kinds(reasons, status):
    """Map reason strings to the categories of ref_reasons (the strings
    themselves are not specified)."""
    out = []
    for r in reasons:
        if r.startswith("status"):
            assert r.endswith(status)
            out.append("status")
        elif r.startswith("registration"):
            out.append("registration")
        else:
            out.append(r)
    return out


@pytest.mark.parametrize("status,verdict,flags", CASES)
def test_reject_reasons_rule(status, verdict, flags):
    res = make_result(fit_status=status, reg_verdict=verdict, flags=tuple(flags))
    got = PH.reject_reasons(res)
    assert _kinds(got, status) == ref_reasons(status, verdict, flags)
    # the same rule on the stored row (D-098.3), NaN and empty cells meaning nothing
    row = {"fit_status": status, "reg_verdict": verdict if verdict else NAN,
           "flags": ";".join(flags) if flags else NAN, "in_S": "garbage", "reject": "faint"}
    assert PH.row_reject_reasons(row) == got
    row2 = dict(row, reg_verdict=verdict, flags=";".join(flags))
    assert PH.row_reject_reasons(row2) == got


def test_row_reject_reasons_flags_argument_and_keyerror():
    row = {"fit_status": "converged", "reg_verdict": "", "flags": "faint;crossing"}
    assert PH.row_reject_reasons(row) == ["crossing"]
    assert set(PH.row_reject_reasons(row, flags=("faint", "crossing"))) == {"faint", "crossing"}
    with pytest.raises(KeyError):
        PH.row_reject_reasons({"reg_verdict": "", "flags": ""})
    with pytest.raises(KeyError):
        PH.row_reject_reasons({"fit_status": "converged", "reg_verdict": ""})


# ------------------------------------------------------------- O6.6 -------

def test_table_inputs():
    d = np.array([0.5, 1.0, 2.0])
    phi = np.array([0.0, 0.3, 1.2])
    mu, dh = np.array([0.8, 1.1, 2.0]), np.array([0.55, 1.05, 2.2])
    X = T.table_inputs(d, phi, mu, dh)
    assert X.shape == (3, 3)
    assert_allclose(X, np.column_stack([np.log(d), phi, np.log(mu * dh)]), rtol=1e-15, atol=0)
    for bad in [(0.0, 0.1, 1.0, 1.0), (-1.0, 0.1, 1.0, 1.0), (np.inf, 0.1, 1.0, 1.0),
                (1.0, np.nan, 1.0, 1.0), (1.0, 0.1, 0.0, 1.0), (1.0, 0.1, 1.0, np.nan), (1.0, 0.1, -1.0, 1.0)]:
        with pytest.raises(ValueError):
            T.table_inputs(*[np.array([b]) for b in bad])


# ------------------------------------------------------------- O6.5 -------

def test_row_provenance_and_check():
    prov = T.row_provenance(CFG)
    assert prov == "sim-" + CFG.signature_hash("simulation")
    rows = [{"sim_hash": prov}, {"sim_hash": prov}, {}]
    assert T.check_rows_config(rows, CFG) == 1
    other = dataclasses.replace(CFG, renderer=dataclasses.replace(CFG.renderer, noise_sd_gl=4.0))
    with pytest.raises(ValueError):
        T.check_rows_config(rows, other)


def test_rows_file_round_trip(tmp_path):
    rows = [{"draw_index": 1, "realization_index": 2, "d_um": 0.5, "phi_rad": NAN, "in_S": True,
             "flags": "faint;crossing", "reject": "", "sim_hash": "sim-0123456789012345", "note": "x y"},
            {"draw_index": 3, "realization_index": 1, "d_um": 1.25e-3, "phi_rad": 0.1, "in_S": False,
             "flags": "", "reject": "crossing", "sim_hash": "sim-1e10", "note": ""}]
    p = str(tmp_path / "rows.csv")
    TIO.write_rows(rows, p)
    back = TIO.read_rows([p])
    assert len(back) == 2
    for a, b in zip(rows, back):
        assert b["draw_index"] == a["draw_index"] and isinstance(b["draw_index"], (int, np.integer))
        assert b["in_S"] is a["in_S"] or b["in_S"] == a["in_S"]
        assert b["sim_hash"] == a["sim_hash"] and isinstance(b["sim_hash"], str)
        assert b["d_um"] == a["d_um"]                          # floats survive exactly (repr)
        assert (math.isnan(b["phi_rad"]) and math.isnan(a["phi_rad"])) or b["phi_rad"] == a["phi_rad"]
    assert back[0]["flags"] == "faint;crossing"


# ------------------------------------------------- run_replicate (O6.5) ---

def test_run_replicate_row_contract_and_determinism():
    cfg = dataclasses.replace(CFG, phantom=dataclasses.replace(P, phi_range_deg=(0.0, 1.0)))
    r1 = PH.run_replicate(cfg, 20261113, 1, 1, d_override_um=0.5)
    r2 = PH.run_replicate(cfg, 20261113, 1, 1, d_override_um=0.5)
    timing = {k for k in r1 if "time" in k or k.endswith("_s") or k == "seconds"}   # the wall time
    for k in r1:
        if k in timing:
            continue
        a, b = r1[k], r2[k]
        if isinstance(a, float) and math.isnan(a):
            assert isinstance(b, float) and math.isnan(b), k
        else:
            assert a == b, k                                   # bit-identical repeat
    # the row: the draws, the NodeResult columns, in_S and reject, the seed and the index
    # (sim_hash is a column of the rows file, checked on the CLI's output)
    for f in dataclasses.fields(PH.Draw):
        assert f.name in r1, f.name
    # the measured centre / heading / tilt sit under meas_* beside the draw's true values
    # (the names are not in the spec: see the report's "Problems in the spec")
    for col in ("k_star", "z_sub_um", "meas_cx_um", "meas_theta_rad", "meas_phi_rad", "B_bar",
                "v0_hat_um", "alpha_hat", "steep", "vertical"):
        assert col in r1, col
    assert r1["d_um"] == 0.5 and r1["draw_index"] == 1 and r1["realization_index"] == 1
    for col in ("k_iter", "iter_status", "n_rounds", "n_restacks", "w_hat_um", "beta_hat", "xi_hat",
                "zeta_hat", "irls_ok", "at_bound", "d_pilot_um", "alpha_pilot", "pilot_status"):
        assert col in r1, col
    for col in ("d_hat_um", "mu_hat_per_um", "fit_status", "flags", "in_S", "reject", "phi_rad"):
        assert col in r1, col
    # the stored S equals the rule applied to the row's own columns
    reasons = PH.row_reject_reasons(r1)
    assert bool(r1["in_S"]) == (len(reasons) == 0)
    assert str(r1["reject"] or "") == ";".join(reasons)
