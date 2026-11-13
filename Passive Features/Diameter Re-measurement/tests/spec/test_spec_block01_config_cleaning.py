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
    assert c.signature_hash("estimator") == "59678d4e2aedf2b6"
    assert c.signature_hash("simulation") == "52aa73e479eb77cf"


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


def test_json_round_trip_keeps_cleaning_and_inference():
    c2 = C.with_overrides(C.default_config(), {"cleaning.shadow_absorption": "partition_vertical",
                                                "inference.plane_snr_min": 2.0})
    back = C.config_from_dict(json.loads(c2.to_json()))
    assert back == c2, "to_json drops the cleaning/inference sections: %r -> %r" % (
        (c2.cleaning.shadow_absorption, c2.inference.plane_snr_min),
        (back.cleaning.shadow_absorption, back.inference.plane_snr_min))
