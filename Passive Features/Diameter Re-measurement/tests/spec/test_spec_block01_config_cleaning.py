"""Spec-derived tests, Block 1 additions of 2026-11-13 (CleaningConfig, InferenceConfig) -- tester 2026-11-13.

Block 1's diff is outside the listed scope (23-31) and was added to it
(incremental mode: source changed outside the scope). Oracles: the defaults
the SPEC entries of Blocks 23-31 state; the sections outside every signature
("sub-config; outside every signature"): the estimator and simulation
hashes stay the pinned defaults 59678d4e2aedf2b6 / 52aa73e479eb77cf and do
not move with a cleaning or inference change; with_overrides reaches the
new sections; validate() refuses illegal values (Block 1: "validate()
refuses the illegal values"); the JSON round trip of Block 1 ("JSON round
trip is identity") with a non-default cleaning/inference section.
"""
from __future__ import annotations

import json

import pytest

import nc_helpers as H  # noqa: F401
from allen_diameter import config as C


def test_defaults_as_specified():
    c = C.default_config()
    cl, inf = c.cleaning, c.inference
    assert cl.shadow_floor_gl == 0.3 and cl.shadow_fine_factor == 8 and cl.shadow_reach_sigmas == 6.0
    assert cl.close_lateral_um == 2.0 and cl.close_defocus_um == 0.84 and cl.clean_index_max == 0.02
    assert tuple(cl.example_counts) == (4, 6, 6, 2, 2) and sum(cl.example_counts) == 20
    assert cl.tier_reach_um == 3.0 and cl.over_background_gl == 3.0
    assert cl.max_rounds == 5 and cl.stop_rel_d == 0.01 and cl.shadow_absorption == "hybrid"
    assert cl.neighbour_swc_types == (2, 3, 4)
    assert inf.absorption == "hybrid" and inf.line_fine_factor == 8 and inf.line_reach_sigmas == 6.0
    assert c.signature_hash("estimator") == "16be09f944e45972"   # since D-178 (2026-11-18, measure.pilot_diameter); 59678d4e2aedf2b6 before
    assert c.signature_hash("simulation") == "dc84ed2e89bcfea6"   # since D-178; 52aa73e479eb77cf before


def test_overrides_reach_sections_and_hashes_unmoved():
    c = C.default_config()
    c2 = C.with_overrides(c, {"cleaning.shadow_absorption": "partition_vertical", "inference.max_nfev": 50})
    assert c2.cleaning.shadow_absorption == "partition_vertical" and c2.inference.max_nfev == 50
    for w in ("estimator", "simulation"):
        assert c2.signature_hash(w) == c.signature_hash(w)


@pytest.mark.parametrize("override", [
    {"cleaning.shadow_absorption": "linear"},
    {"inference.absorption": "ray_world"},
    {"cleaning.shadow_fine_factor": 0},
    {"cleaning.refine_d_bounds_um": [6.0, 0.1]},
    {"inference.multistart_factors": []},
    {"cleaning.example_counts": [-1, 6, 6, 2, 2]},
    {"cleaning.shadow_floor_gl": -1.0},
])
def test_validate_refuses_illegal_cleaning_values(override):
    with pytest.raises(ValueError):
        C.with_overrides(C.default_config(), override)


def test_json_is_the_full_signature_without_cleaning_and_inference():
    """Adapted 2026-11-13 after D-136.5 (final check, F4 kept as designed; at the user's request): to_json is the
    full signature, and the cleaning and inference sections are outside every signature, so an override of
    either changes neither the JSON nor a hash. (Was: the JSON round trip keeps the two sections.)"""
    c0 = C.default_config()
    c2 = C.with_overrides(c0, {"cleaning.shadow_absorption": "partition_vertical",
                               "inference.plane_snr_min": 2.0})
    assert c2.cleaning.shadow_absorption == "partition_vertical" and c2.inference.plane_snr_min == 2.0
    assert c2.to_json() == c0.to_json()
    for which in ("estimator", "simulation", "full"):
        assert c2.signature_hash(which) == c0.signature_hash(which)
    doc = json.loads(c2.to_json())
    assert "cleaning" not in doc and "inference" not in doc


def test_background_support_amendment4():
    """Block 36 amendment 4 (D-144.1), 2026-11-14 18:30: background_support defaults to "model", the four names
    accepted (the other three the labelled alternatives), background_shadow_max 0.10 and refused outside
    (0, 1); neither moves a signature hash (the cleaning section is outside every signature)."""
    c = C.default_config()
    assert c.cleaning.background_support == "model" and c.cleaning.background_shadow_max == 0.10
    for name in ("model", "unshaded", "square", "stripe"):
        c2 = C.with_overrides(c, {"cleaning.background_support": name})
        assert c2.cleaning.background_support == name
        assert c2.signature_hash("full") == c.signature_hash("full")
    for bad in ({"cleaning.background_support": "far"}, {"cleaning.background_shadow_max": 0.0},
                {"cleaning.background_shadow_max": 1.0}, {"cleaning.background_shadow_max": float("nan")}):
        with pytest.raises(ValueError):
            C.with_overrides(c, bad)


# ---------------------------------------------------------------------------------------------------------------
# Tester 2026-11-22 (incremental at a749cc9): Block 1 additions of 2026-11-18 to 2026-11-20 -- measure.pilot_diameter
# (D-178), the inference weights (Block 39), the sections clearness (Block 40) and sigma_calibration (Block 42).
# Oracles: SPEC Block 1 (the pinned hashes since D-178), Blocks 39, 40, 42 (defaults, "outside every signature",
# "a configuration JSON without the section takes the defaults"), Block 1 ("validate() raises ValueError").
# ---------------------------------------------------------------------------------------------------------------

import ast as _ast
import dataclasses as _dc
import math as _math
from pathlib import Path as _Path

_PINNED = ("16be09f944e45972", "dc84ed2e89bcfea6")     # SPEC Block 1 test oracles, since D-178 (2026-11-18)


def _hashes(cfg):
    return tuple(cfg.signature_hash(w) for w in ("estimator", "simulation", "full"))


def test_new_defaults_as_specified_2026_11_22():
    c = C.default_config()
    assert (c.signature_hash("estimator"), c.signature_hash("simulation")) == _PINNED
    assert c.measure.pilot_diameter == "allen" and C.PILOT_DIAMETERS == ("allen", "free")
    i = c.inference
    assert (i.weights, i.weight_kappa, i.weight_irls_max_iter, i.weight_irls_tol_um, i.bump_sd, i.bump_half_depth,
            i.bump_smooth_samples) == ("allen", 1.25, 3, 0.01, 2.0, 0.5, 1.0)
    q = c.clearness
    assert (q.focus_half_planes, q.along_half_um, q.along_step_um, q.smooth_samples, q.window_um,
            tuple(q.skew_levels), q.outside_fwhm, q.outside_max, q.skew_max, q.width_range_max, q.depth_range_max,
            q.centre_range_max, q.min_depth_sd, q.judge_along) == (1, 2.0, 0.5, 1.0, 1.0, (0.75, 0.25), 1.0, 0.25,
                                                                   0.15, 0.25, 0.30, 0.50, 10.0, False)
    s = c.sigma_calibration
    assert (s.half_planes, s.start_ends_um, s.sigma0_start_um, s.growth_scale_start, s.width_to_d,
            s.hybrid_faint_factor, tuple(s.d_bounds_um), tuple(s.mu_bounds_per_um), tuple(s.v0_bounds_um),
            tuple(s.z_bounds_um), tuple(s.sigma0_bounds_um), tuple(s.growth_scale_bounds), tuple(s.diff_steps),
            s.max_nfev, s.ftol, s.xtol, s.workers) == (
        2, 3.0, 0.15, 1.0, 0.9, 1.447, (0.2, 6.0), (0.005, 10.0), (-1.5, 1.5), (-1.5, 1.5), (0.02, 0.6),
        (0.05, 10.0), (0.01, 0.005, 0.005, 0.01, 0.005, 0.02), 40, 1e-6, 1e-6, 0)


_OUTSIDE = {
    "clearness.focus_half_planes": 2, "clearness.along_half_um": 1.5, "clearness.along_step_um": 0.25,
    "clearness.smooth_samples": 0.0, "clearness.window_um": 0.8, "clearness.skew_levels": [0.8, 0.2],
    "clearness.outside_fwhm": 1.5, "clearness.outside_max": 0.3, "clearness.skew_max": 0.2,
    "clearness.width_range_max": 0.3, "clearness.depth_range_max": 0.4, "clearness.centre_range_max": 0.6,
    "clearness.min_depth_sd": 5.0, "clearness.judge_along": True,
    "sigma_calibration.half_planes": 3, "sigma_calibration.start_ends_um": 2.0,
    "sigma_calibration.sigma0_start_um": 0.1, "sigma_calibration.growth_scale_start": 1.5,
    "sigma_calibration.width_to_d": 1.0, "sigma_calibration.hybrid_faint_factor": 1.0,
    "sigma_calibration.d_bounds_um": [0.3, 5.0], "sigma_calibration.max_nfev": 60, "sigma_calibration.ftol": 1e-5,
    "sigma_calibration.diff_steps": [0.02, 0.01, 0.01, 0.02, 0.01, 0.04], "sigma_calibration.workers": 2,
    "inference.weights": "none", "inference.weight_kappa": 2.0, "inference.weight_irls_max_iter": 5,
    "inference.weight_irls_tol_um": 0.02, "inference.bump_sd": 0.0, "inference.bump_half_depth": 0.4,
    "inference.bump_smooth_samples": 2.0,
}


@pytest.mark.parametrize("key", sorted(_OUTSIDE))
def test_new_sections_outside_every_signature(key):
    c = C.default_config()
    c2 = C.with_overrides(c, {key: _OUTSIDE[key]})
    sec, name = key.split(".")
    assert getattr(getattr(c2, sec), name) != getattr(getattr(c, sec), name)
    assert _hashes(c2) == _hashes(c)


def test_pilot_diameter_in_the_estimator_signature():
    c = C.default_config()
    f = C.with_overrides(c, {"measure.pilot_diameter": "free"})
    assert f.signature_hash("estimator") != c.signature_hash("estimator")
    assert f.signature_hash("simulation") != c.signature_hash("simulation")
    with pytest.raises(ValueError):
        C.with_overrides(c, {"measure.pilot_diameter": "fixed"})


def test_json_without_the_new_keys_takes_the_defaults():
    c = C.default_config()
    d = json.loads(c.to_json())
    d["measure"].pop("pilot_diameter")
    for sec in ("clearness", "sigma_calibration"):
        d.pop(sec, None)
    c2 = C.config_from_dict(d)
    assert c2.measure.pilot_diameter == "allen" and c2.clearness == c.clearness
    assert c2.sigma_calibration == c.sigma_calibration
    # a partial section: the rest at the defaults, tuples restored
    d["clearness"] = {"skew_levels": [0.7, 0.3], "judge_along": True}
    d["sigma_calibration"] = {"diff_steps": [0.02, 0.01, 0.01, 0.02, 0.01, 0.04]}
    c3 = C.config_from_dict(d)
    assert c3.clearness.skew_levels == (0.7, 0.3) and c3.clearness.judge_along is True
    assert c3.clearness.outside_max == 0.25 and isinstance(c3.sigma_calibration.diff_steps, tuple)


@pytest.mark.parametrize("override", [
    {"clearness.judge_along": 1}, {"clearness.judge_along": "yes"}, {"clearness.skew_levels": [0.25, 0.75]},
    {"clearness.outside_max": 0.0}, {"clearness.along_step_um": 0.0}, {"clearness.window_um": -1.0},
    {"clearness.focus_half_planes": -1}, {"clearness.min_depth_sd": float("nan")},
    {"sigma_calibration.half_planes": 0}, {"sigma_calibration.max_nfev": 0}, {"sigma_calibration.ftol": 1.0},
    {"sigma_calibration.diff_steps": [0.01, 0.01, 0.01, 0.01, 0.01]}, {"sigma_calibration.sigma0_start_um": 0.7},
    {"sigma_calibration.d_bounds_um": [6.0, 0.2]}, {"sigma_calibration.workers": -1},
    {"inference.weights": "gaussian"}, {"inference.weight_kappa": 0.0}, {"inference.bump_half_depth": 1.0},
    {"inference.bump_sd": -1.0}, {"inference.weight_irls_max_iter": 0}, {"measure.pilot_diameter": "Allen"},
])
def test_validate_refuses_new_illegal_values(override):
    with pytest.raises(ValueError):
        C.with_overrides(C.default_config(), override)


@pytest.mark.parametrize("override", [
    {"sigma_calibration.half_planes": float("inf")}, {"sigma_calibration.max_nfev": float("inf")},
    {"sigma_calibration.workers": float("inf")}, {"clearness.focus_half_planes": float("inf")},
    {"inference.weight_irls_max_iter": float("inf")},
])
@pytest.mark.xfail(strict=True, raises=OverflowError, reason="2026-11-22 report F1: int(inf) in the new "
                   "checks of config.py raises OverflowError, not ValueError")
def test_validate_raises_valueerror_for_an_infinite_count(override):
    """Block 1: validate() raises ValueError on an illegal value. An infinite count is illegal; the check must
    say so with ValueError (not another exception type)."""
    with pytest.raises(ValueError):
        C.with_overrides(C.default_config(), override)


def test_every_new_field_carries_a_source_tag():
    src = (_Path(C.__file__)).read_text()
    tree = _ast.parse(src)
    lines = src.splitlines()
    want = {"ClearnessConfig": None, "SigmaCalibrationConfig": None}
    for node in tree.body:
        if isinstance(node, _ast.ClassDef) and node.name in want:
            fields = [s for s in node.body if isinstance(s, _ast.AnnAssign)]
            assert fields, node.name
            for s in fields:
                assert "# source:" in lines[s.lineno - 1], (node.name, lines[s.lineno - 1])
            want[node.name] = len(fields)
    assert want["ClearnessConfig"] == len(_dc.fields(C.ClearnessConfig))
    assert want["SigmaCalibrationConfig"] == len(_dc.fields(C.SigmaCalibrationConfig))
    for cls, names in ((C.MeasureConfig, ("pilot_diameter",)),
                       (C.InferenceConfig, ("weights", "weight_kappa", "weight_irls_max_iter", "weight_irls_tol_um",
                                            "bump_sd", "bump_half_depth", "bump_smooth_samples"))):
        for name in names:
            hit = [ln for ln in lines if ln.strip().startswith(name + ":")]
            assert hit and all("# source:" in ln for ln in hit), (cls.__name__, name)
