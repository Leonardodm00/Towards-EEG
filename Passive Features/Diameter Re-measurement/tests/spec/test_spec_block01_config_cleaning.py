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
# 2026-11-22 (14:40) report F1, fixed by D-212.1: a strict xfail until then, a plain test since D-224.3.
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

# Tester, 2026-11-22 (incremental run at 97f5104): the final check of wave 4's F1 settled by D-212.1 -- every count
# setting is checked by `config._is_count`: a finite whole number (an int, or a float with an integral value), not a
# bool, at least its floor; anything else is refused with ValueError (SPEC, "The final check of wave 4", Block 1
# amendment). Also the two sections of wave 5 (soma_view, soma_fit). Expected values from the spec's text.

COUNT_FIELDS = [("inference.weight_irls_max_iter", 1), ("inference.line_fine_factor", 1),
                ("cleaning.shadow_fine_factor", 1), ("cleaning.refine_line_fine_factor", 1),
                ("sigma_calibration.half_planes", 1), ("sigma_calibration.max_nfev", 1), ("sigma_calibration.workers", 0),
                ("clearness.focus_half_planes", 0), ("soma_view.montage_step_planes", 1), ("soma_view.montage_cols", 1),
                ("soma_fit.fine_factor", 1), ("soma_fit.plane_step", 1), ("soma_fit.max_nfev", 1)]


@pytest.mark.parametrize("field,floor", COUNT_FIELDS)
def test_count_fields_refuse_non_counts_with_valueerror(field, floor):
    c = C.default_config()
    for bad in (float("inf"), float("-inf"), float("nan"), 2.5, True, floor - 1):
        try:
            C.with_overrides(c, {field: bad})
        except ValueError:
            continue
        except Exception as exc:        # OverflowError (int(inf)) was the wave-4 finding: any other type is a fail
            pytest.fail("%s = %r raised %s, not ValueError" % (field, bad, type(exc).__name__))
        pytest.fail("%s = %r was accepted" % (field, bad))
    for good in (floor, float(floor + 2)):          # an int, or a float with an integral value
        c2 = C.with_overrides(c, {field: good})
        assert c2.signature_hash("estimator") == c.signature_hash("estimator")


def test_example_counts_each_a_count():
    c = C.default_config()
    for bad in ([4, 6, 6, 2, float("inf")], [4, 6, 6, 2, 2.5], [4, 6, 6, 2, -1], [4, 6, 6, 2, True]):
        with pytest.raises(ValueError):
            C.with_overrides(c, {"cleaning.example_counts": bad})
    assert C.with_overrides(c, {"cleaning.example_counts": [4, 6, 6, 2, 0.0]}).cleaning.example_counts[4] == 0.0

# ---------------------------------------------------------------------------------------------------------------------
# Appended by the tester, 2026-11-23 (incremental run at f1bc7cb; Block 1's diff: SomaFitConfig's start settings
# (Block 46 amendment 1, D-220.4) and the entry point's crop_margin_um, figure_planes_um (section 6, D-222.5)).
# Oracles: the defaults SPEC Blocks 44-46 state; the soma sections outside every signature (the pinned hashes do not
# move); with_overrides reaches the new fields in their declared form; a configuration file written before wave 6
# (without the new fields) loads with the defaults.

def test_soma_sections_defaults_as_specified():
    c = C.default_config()
    q, v = c.soma_fit, c.soma_view
    assert (q.fine_factor, q.dzeta_um) == (1, 0.14)                                         # Block 45
    # [2026-11-24] D-241 (user): the region is Allen's soma radius (None), the mask unwidened; D-240: the slope held
    assert (q.plane_step, q.plane_half_um, q.region_radius_um, q.mask_widen_um) == (2, 10.0, None, 0.0)
    assert (q.far_slope, q.shift_xy_um, q.shift_z_um) == ("fixed", 4.0, 6.0)
    assert tuple(q.axis_bounds_um) == (1.0, 15.0) and tuple(q.mu_bounds_per_um) == (0.001, 3.0)
    assert tuple(q.slope_bounds) == (0.2, 3.0) and (q.max_nfev, q.ftol, q.xtol) == (80, 1e-6, 1e-6)
    assert tuple(q.diff_steps) == (0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.004, 1e-4, 0.005)
    assert (q.start_far_radii, q.start_core_radii, q.start_core_percentile) == (1.5, 0.5, 5.0)   # amendment 1 (i)
    assert tuple(q.start_ratio_clip) == (0.05, 0.95) and q.start_ratio_none == 0.5
    assert (v.half_um, v.z_margin_um, v.focus_margin_um, v.focus_sigma_px) == (40, 6, 1, 1)        # Block 44
    assert (v.montage_step_planes, v.montage_cols) == (2, 6)


def test_soma_fit_overrides_and_hashes_unmoved():
    c = C.default_config()
    c2 = C.with_overrides(c, {"soma_fit.start_ratio_clip": [0.1, 0.9], "soma_fit.figure_planes_um": [-2, 0, 2],
                              "soma_fit.start_core_percentile": 10, "soma_fit.crop_margin_um": 2.5})
    assert c2.soma_fit.start_ratio_clip == (0.1, 0.9) and isinstance(c2.soma_fit.start_ratio_clip, tuple)
    assert tuple(c2.soma_fit.figure_planes_um) == (-2.0, 0.0, 2.0)
    assert c2.soma_fit.start_core_percentile == 10.0 and c2.soma_fit.crop_margin_um == 2.5
    for which in ("estimator", "simulation"):
        assert c2.signature_hash(which) == c.signature_hash(which)
    assert c.signature_hash("estimator") == "16be09f944e45972" and c.signature_hash("simulation") == "dc84ed2e89bcfea6"


def test_pre_wave6_file_loads_with_the_defaults():
    old = {"soma_fit": {"plane_step": 2, "plane_half_um": 8.0, "region_radius_um": 12.0, "far_slope": "fixed"}}
    c = C.config_from_dict(old)
    q = c.soma_fit
    assert (q.plane_half_um, q.far_slope) == (8.0, "fixed")
    assert (q.start_far_radii, q.start_core_radii, q.start_core_percentile, q.start_ratio_none) == (1.5, 0.5, 5.0, 0.5)
    assert tuple(q.start_ratio_clip) == (0.05, 0.95) and q.crop_margin_um == 1.0


@pytest.mark.parametrize("override", [
    {"soma_fit.start_ratio_none": 0.0},          # varrho must stay in (0, 1): mu_0 = -ln(varrho) / (2 r_s) >= 0
    {"soma_fit.start_ratio_none": 1.5},
    {"soma_fit.start_ratio_clip": [0.0, 0.9]},
    {"soma_fit.start_core_radii": -0.5},
    {"soma_fit.crop_margin_um": float("nan")},
    {"soma_fit.figure_planes_um": [float("inf")]},
    {"soma_fit.render_half_um": 0.0},            # [2026-11-24, D-243] the crop's floor: finite and > 0
    {"soma_fit.render_half_um": float("inf")},
])
def test_validate_refuses_illegal_soma_start_values(override):
    with pytest.raises(ValueError):
        C.with_overrides(C.default_config(), override)


# ------------------------------- wave 7's inference fields (tester, 2026-11-23 13:55) ---
# Oracles: Block 49's parameters (sgt_reach_sigmas 12, sgt_samples_per_fwhm 16), Block 30 amendment 3 (the shape
# bounds, sgt_skew "off"/"fit", cov_steps in the order d, mu, v0, z_ax, log10 beta, xi, ln zeta, the first 4, 6 or 7),
# Block 36 amendments 22-23 (kernel_family "gaussian"/"sgt", sgt_start_shape the Gaussian member, cov_fine_factor 0 =
# the fit's own Jacobian), Block 1 (the inference section outside every signature, # source: tags, validate refusing
# illegal values, every field read outside config.py, a file without the new keys taking the defaults).

import dataclasses  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

W7_FIELDS = ("kernel_family", "sgt_skew", "sgt_log10_beta_bounds", "sgt_xi_bounds", "sgt_ln_zeta_bounds",
             "sgt_start_shape", "sgt_reach_sigmas", "sgt_samples_per_fwhm", "cov_fine_factor", "cov_steps")


def test_w7_defaults_as_specified():
    """B1.5: the defaults the spec names, and the shape box equal to Block 12's (log10 of sgt_beta_bounds (1, 10),
    sgt_xi_bounds (0, 2), sgt_ln_zeta_bounds (-0.7, 0.7); D-052.2) so that the per-node SGT and Block 12's fit share
    one family box."""
    c = C.default_config()
    i, m = c.inference, c.measure
    assert i.kernel_family == "gaussian" and C.INFERENCE_KERNELS[0] == "gaussian" and "sgt" in C.INFERENCE_KERNELS
    assert i.sgt_skew == "off" and tuple(C.SGT_SKEWS) == ("off", "fit")
    assert i.sgt_reach_sigmas == 12.0 and i.sgt_samples_per_fwhm == 16
    assert tuple(i.sgt_start_shape) == (2.0, 0.0, 1.0)
    assert tuple(i.sgt_log10_beta_bounds) == tuple(math.log10(b) for b in m.sgt_beta_bounds)
    assert tuple(i.sgt_xi_bounds) == tuple(m.sgt_xi_bounds) and tuple(i.sgt_ln_zeta_bounds) == tuple(m.sgt_ln_zeta_bounds)
    assert len(i.cov_steps) == 7 and all(s > 0 for s in i.cov_steps)
    assert i.cov_fine_factor == 32 and i.cov_fine_factor % i.line_fine_factor == 0   # Block 49: "8 and 32 (the error bars' lattice)"


@pytest.mark.parametrize("key,value", [("kernel_family", "sgt"), ("sgt_skew", "fit"), ("sgt_reach_sigmas", 20.0),
                                       ("sgt_samples_per_fwhm", 24), ("cov_fine_factor", 0),
                                       ("cov_steps", (1e-3,) * 7), ("sgt_start_shape", (2.0, 0.5, 1.2)),
                                       ("sgt_xi_bounds", (0.0, 1.5))])
def test_w7_fields_outside_every_signature(key, value):
    """B1.1: the inference section is outside every signature: the estimator, simulation and full hashes do not
    move with a wave-7 field (pinned 16be09f944e45972 / dc84ed2e89bcfea6)."""
    c = C.default_config()
    c2 = C.with_overrides(c, {"inference." + key: list(value) if isinstance(value, tuple) else value})
    assert getattr(c2.inference, key) == value
    for which in ("estimator", "simulation", "full"):
        assert c2.signature_hash(which) == c.signature_hash(which)
    assert c2.signature_hash("estimator") == "16be09f944e45972" and c2.signature_hash("simulation") == "dc84ed2e89bcfea6"


def test_w7_overrides_in_declared_form():
    """B1.3: with_overrides puts the new fields in their declared form (lists to tuples, an integer given for a
    float field to a float), validated, without touching its input."""
    c = C.default_config()
    c2 = C.with_overrides(c, {"inference.cov_steps": [1, 1, 1, 1, 1, 1, 1], "inference.sgt_start_shape": [2, 0, 1],
                              "inference.sgt_reach_sigmas": 15, "inference.sgt_log10_beta_bounds": [0, 0.8]})
    assert isinstance(c2.inference.cov_steps, tuple) and all(isinstance(x, float) for x in c2.inference.cov_steps)
    assert isinstance(c2.inference.sgt_start_shape, tuple) and isinstance(c2.inference.sgt_reach_sigmas, float)
    assert c.inference.cov_steps != c2.inference.cov_steps


@pytest.mark.parametrize("override", [
    {"inference.kernel_family": "laplace"}, {"inference.kernel_family": ""}, {"inference.sgt_skew": "on"},
    {"inference.sgt_log10_beta_bounds": [1.0, 0.0]}, {"inference.sgt_log10_beta_bounds": [0.0, float("nan")]},
    {"inference.sgt_xi_bounds": [-0.1, 2.0]}, {"inference.sgt_ln_zeta_bounds": [0.7, -0.7]},
    {"inference.sgt_start_shape": [20.0, 0.0, 1.0]}, {"inference.sgt_start_shape": [2.0, 3.0, 1.0]},
    {"inference.sgt_start_shape": [2.0, 0.0, 3.0]}, {"inference.sgt_start_shape": [0.0, 0.0, 1.0]},
    {"inference.sgt_start_shape": [2.0, 0.0, 0.0]}, {"inference.sgt_start_shape": [2.0, float("nan"), 1.0]},
    {"inference.sgt_start_shape": [2.0, 0.0]},
    {"inference.sgt_reach_sigmas": 0.0}, {"inference.sgt_reach_sigmas": float("nan")},
    {"inference.sgt_reach_sigmas": float("inf")},
    {"inference.sgt_samples_per_fwhm": 0}, {"inference.sgt_samples_per_fwhm": 2.5},
    {"inference.sgt_samples_per_fwhm": True},
    {"inference.cov_fine_factor": -1}, {"inference.cov_fine_factor": 2.5}, {"inference.cov_fine_factor": True},
    {"inference.cov_steps": [1e-3] * 6}, {"inference.cov_steps": [1e-3] * 6 + [0.0]},
    {"inference.cov_steps": [1e-3] * 6 + [float("nan")]}, {"inference.cov_steps": [1e-3] * 6 + [-1e-3]}])
def test_w7_validate_refuses(override):
    """B1.2: validate() refuses each illegal value with ValueError (through with_overrides, which validates)."""
    with pytest.raises(ValueError):
        C.with_overrides(C.default_config(), override)


def test_w7_record_without_the_new_keys_loads_with_defaults():
    """B1.3: a preparation's record written before wave 7 (its inference section without the ten new keys) loads
    through infer_nodes.run_config with the new fields at their defaults, the recorded ones kept."""
    import infer_nodes
    c = C.default_config()
    inf = dataclasses.asdict(c.inference)
    for k in W7_FIELDS:
        inf.pop(k)
    inf["max_nfev"] = 77
    rec = dict(config=json.loads(c.to_json()), cleaning=dataclasses.asdict(c.cleaning), inference=inf)
    cfg = infer_nodes.run_config(rec, "", None)
    assert cfg.inference.max_nfev == 77
    for k in W7_FIELDS:
        assert getattr(cfg.inference, k) == getattr(c.inference, k), k


def test_w7_source_tags_and_readers():
    """B1.1 / B1.4: every new field's line carries '# source:', and every new field is read by code outside
    config.py (an attribute access in src/ or scripts/)."""
    root = Path(H.WS)
    text = (root / "src" / "allen_diameter" / "config.py").read_text()
    body = text.split("class InferenceConfig", 1)[1].split("\n@dataclass", 1)[0]
    for k in W7_FIELDS:
        lines = [ln for ln in body.splitlines() if re.match(r"\s+%s\s*:" % k, ln)]
        assert len(lines) == 1 and "# source:" in lines[0], k
    others = [p for p in list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py"))
              if p.name != "config.py"]
    blob = "\n".join(p.read_text(errors="replace") for p in others)
    for k in W7_FIELDS:
        # kernel_family is also a RendererConfig field: require the inference section's own reads
        pat = r"(inference|icfg|\bi)\.kernel_family\b" if k == "kernel_family" else r"\.%s\b" % k
        assert re.search(pat, blob), "%s is read by nothing outside config.py" % k


# ---------------------------------------------------------------------------------------------------------------
# Tester, 2026-11-27 (incremental run at a67b4df): Step2Config (SPEC "Kernel calibration, step 2", Configuration)
# and Block 15's amendment (far_min_k_offset = 4, D-273.2). Expected values from the spec's text.
# ---------------------------------------------------------------------------------------------------------------

_STEP2_DEFAULTS = dict(absorption="hybrid", kernel_family="gaussian_table_spline", knots_um=(0.28, 0.56, 0.84),
                       knot_min_margin_um=0.001, knot_max_um=2.0, near_planes=3, far_planes=(4, 16),
                       z_ax_bound_um=0.28, weights="inverse_variance", noise_replicates=4, noise_scale=1.0,
                       seed=20261127, validation_fraction=0.25, split_seed=20261127, diff_step_knot_um=0.005,
                       diff_step_z_um=0.02, diff_step_gamma=0.02, max_nfev=40, fit_sigma0=False,
                       tube_half_length_um=40.0, max_workers=1)


def test_step2_defaults_as_specified():
    s = C.default_config().step2
    assert isinstance(s, C.Step2Config)
    got = {f.name: getattr(s, f.name) for f in _dc.fields(s)}
    assert set(got) == set(_STEP2_DEFAULTS)                       # no field the spec does not list
    for k, v in _STEP2_DEFAULTS.items():
        assert (tuple(got[k]) if isinstance(v, tuple) else got[k]) == v, k


def test_block15_amendment_far_min_k_offset_is_4():
    c = C.default_config()
    assert c.calibration.far_min_k_offset == 4
    # step 2's far stage reads 4 <= |k - k*|: the same lower edge (SPEC Block 15, amendment)
    assert c.step2.far_planes[0] == c.calibration.far_min_k_offset


@pytest.mark.parametrize("key,value", [("step2.noise_replicates", 8), ("step2.knots_um", [0.3, 0.6, 0.9]),
                                       ("step2.weights", "none"), ("step2.seed", 5), ("step2.fit_sigma0", True),
                                       ("step2.tube_half_length_um", 30.0), ("step2.validation_fraction", 0.5)])
def test_step2_outside_every_signature(key, value):
    c = C.default_config()
    c2 = C.with_overrides(c, {key: value})
    sec, name = key.split(".")
    assert getattr(c2.step2, name) != getattr(c.step2, name)
    assert _hashes(c2) == _hashes(c)
    assert (c.signature_hash("estimator"), c.signature_hash("simulation")) == _PINNED


@pytest.mark.parametrize("override", [{"step2.weights": "inverse"}, {"step2.validation_fraction": 1.0},
                                      {"step2.validation_fraction": -0.1}, {"step2.noise_replicates": 0},
                                      {"step2.knots_um": [0.56, 0.28, 0.84]}, {"step2.absorption": "nope"}])
def test_step2_validate_refuses(override):
    with pytest.raises(ValueError):
        C.with_overrides(C.default_config(), override)


def test_step2_source_tags():
    src = (_Path(C.__file__)).read_text()
    tree = _ast.parse(src)
    lines = src.splitlines()
    cls = [n for n in tree.body if isinstance(n, _ast.ClassDef) and n.name == "Step2Config"]
    assert cls
    fields = [s for s in cls[0].body if isinstance(s, _ast.AnnAssign)]
    assert len(fields) == len(_dc.fields(C.Step2Config))
    for s in fields:
        assert "# source:" in lines[s.lineno - 1], lines[s.lineno - 1]


# ---------------------------------------------------------------------------------------------------------------
# Step2Config.kernel_family after amendment 3 (tester, 2026-11-27 22:11): the common definitions list only
# "gaussian_table_spline" (the near stage's "tangent" continuation needs it, D-265); F4 of the 1904 report. Oracles:
# "gaussian_table" and an unknown family are refused through with_overrides; the default still validates; the
# configuration files shipped in configs/ still load and validate (regression of the new renderer checks).
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fam", ["gaussian_table", "empirical", "nope"])
def test_step2_kernel_family_only_the_spline(fam):
    with pytest.raises(ValueError):
        C.with_overrides(C.default_config(), {"step2.kernel_family": fam})
    assert C.with_overrides(C.default_config(), {"step2.kernel_family": "gaussian_table_spline"}).step2.kernel_family \
        == "gaussian_table_spline"


def test_shipped_config_files_still_validate():
    import json as _json
    cdir = _Path(C.__file__).resolve().parents[2] / "configs"
    files = sorted(cdir.glob("*.json"))
    assert files
    for p in files:
        d = _json.loads(p.read_text())
        try:
            cfg = C.config_from_dict(d)
        except (TypeError, KeyError):
            # an overrides file rather than a full dictionary: applied to the default
            cfg = C.with_overrides(C.default_config(), d)
        cfg.validate()
