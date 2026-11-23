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
    assert (q.plane_step, q.plane_half_um, q.region_radius_um, q.mask_widen_um) == (2, 10.0, 13.0, 2.0)
    assert (q.far_slope, q.shift_xy_um, q.shift_z_um) == ("fit", 4.0, 6.0)
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
])
def test_validate_refuses_illegal_soma_start_values(override):
    with pytest.raises(ValueError):
        C.with_overrides(C.default_config(), override)
