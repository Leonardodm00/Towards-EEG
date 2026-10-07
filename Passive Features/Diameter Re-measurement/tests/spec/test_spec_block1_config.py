"""Block 1 (configuration): oracles from specs/SPEC.md sections 2, 4 and Block 1."""
import dataclasses
import json
import math
import re

import pytest

from allen_diameter import config as C


def test_decided_defaults():
    cfg = C.default_config()
    m, r, c, p = cfg.measure, cfg.renderer, cfg.correction, cfg.phantom
    # D-023
    assert m.mu_mode == "per_node" and m.bbar_region == "block_masked"
    assert m.focus_bg_rule == "profile_ends_median" and m.sigma_fit_um == 0.099
    # D-024
    assert r.absorption == "partition_vertical" and r.kernel_continuation == "linear"
    assert r.kernel_continuation_slope == 0.79 and m.alpha_dark_flag == 1.0 and r.U_um == 10.0
    assert p.design == "random" and p.phi_range_deg == (0.0, 90.0) and c.response_estimator == "tps_spline"
    assert p.jitter_xy_um == 0.0 and p.jitter_z_um == 0.0
    # D5 and D7
    assert c.bias_flag_threshold == 0.2 and c.fill_policy == "same_branch_then_allen"
    assert cfg.acquisition.specimen_id == 529878215
    # section 2.1 / 3 facts
    assert cfg.acquisition.res0_um == 0.1144 and cfg.acquisition.dz_um == 0.28
    assert cfg.acquisition.dendrite_swc_types == (3, 4)


def test_choice_tuples_first_entry_is_default():
    cfg = C.default_config()
    pairs = [(C.MU_MODES, cfg.measure.mu_mode), (C.BBAR_REGIONS, cfg.measure.bbar_region),
             (C.FOCUS_BG_RULES, cfg.measure.focus_bg_rule), (C.ABSORPTIONS, cfg.renderer.absorption),
             (C.KERNEL_FAMILIES, cfg.renderer.kernel_family),
             (C.KERNEL_CONTINUATIONS, cfg.renderer.kernel_continuation),
             (C.RESPONSE_ESTIMATORS, cfg.correction.response_estimator), (C.PHANTOM_DESIGNS, cfg.phantom.design),
             (C.TABLE_STATISTICS, cfg.correction.table_statistic), (C.FILL_POLICIES, cfg.correction.fill_policy),
             (C.END_CUTS, cfg.renderer.end_cut), (C.TABLE_PHI_AXES, cfg.correction.table_phi_axis),
             (C.FIT_START_RULES, cfg.measure.fit_start_rule), (C.RENDER_BACKENDS, cfg.renderer.backend)]
    for tup, val in pairs:
        assert tup[0] == val, (tup, val)


def test_signatures_and_hash():
    cfg = C.default_config()
    h = cfg.signature_hash("estimator")
    assert re.fullmatch(r"[0-9a-f]{16}", h)
    assert "sigma_fit_study_um" not in cfg.estimator_signature()["measure"]
    assert "sigma_fit_um" in cfg.estimator_signature()["measure"]
    r2 = dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer, noise_sd_gl=1.0))
    assert r2.signature_hash("estimator") == h
    assert r2.signature_hash("full") != cfg.signature_hash("full")
    s2 = C.with_sigma_fit(cfg, 0.125)
    assert s2.signature_hash("estimator") != h
    with pytest.raises(ValueError):
        C.with_sigma_fit(cfg, 0.11)


def test_json_round_trip_default():
    cfg = C.default_config()
    js = cfg.to_json()
    js.encode("ascii")
    assert C.config_from_dict(json.loads(js)) == cfg


def test_json_round_trip_every_section():
    """Spec Block 1: 'JSON round trip is identity'. Section 3: the table's JSON is
    config.full_signature(). A non-default value in each sub-config must survive."""
    cfg = C.default_config()
    cfg2 = dataclasses.replace(
        cfg,
        calibration=dataclasses.replace(cfg.calibration, node_alpha_max=0.4, offsets_planes=2, trust_planes=1),
        phantom=dataclasses.replace(cfg.phantom, n_replicates=17),
    )
    cfg2.validate()
    back = C.config_from_dict(json.loads(cfg2.to_json()))
    assert back.phantom == cfg2.phantom
    assert back.calibration == cfg2.calibration, "calibration settings are lost by to_json / config_from_dict"


def test_frozen():
    cfg = C.default_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.measure.sigma_fit_um = 0.1


@pytest.mark.parametrize("section,field,value", [
    ("renderer", "noise_sd_gl", float("nan")),
    ("measure", "alpha_dark_flag", float("nan")),
    ("measure", "faint_min_dip_gl", float("nan")),
    ("renderer", "kernel_continuation_slope", float("nan")),
    ("acquisition", "dz_um", float("nan")),
    ("renderer", "background_B_gl", float("nan")),
])
def test_validate_refuses_nan(section, field, value):
    """Silent-failure probe: NaN must not slip past comparison-based validation
    (a NaN alpha_dark_flag would switch the dark flag off silently)."""
    cfg = C.default_config()
    bad = dataclasses.replace(cfg, **{section: dataclasses.replace(getattr(cfg, section), **{field: value})})
    with pytest.raises(ValueError):
        bad.validate()


def test_validate_refuses_listed_illegal_values():
    cfg = C.default_config()
    for section, kw in [("calibration", dict(knot_step_um=0.14)), ("renderer", dict(sigma_r0_um=0.09)),
                        ("measure", dict(mu_mode="bogus")), ("renderer", dict(end_cut="x")),
                        ("phantom", dict(phi_range_deg=(0.0, 95.0))), ("measure", dict(fit_params=("d", "alpha", "v0")))]:
        bad = dataclasses.replace(cfg, **{section: dataclasses.replace(getattr(cfg, section), **kw)})
        with pytest.raises(ValueError):
            bad.validate()


def test_every_field_has_source_tag():
    import inspect
    src = inspect.getsource(C)
    n_fields = sum(len(dataclasses.fields(cls)) for cls in (C.AcquisitionConfig, C.MeasureConfig, C.CorrectionConfig,
                                                             C.RendererConfig, C.PhantomConfig, C.CalibrationConfig))
    tagged = len(re.findall(r"^\s{4}\w+: [^=]+= .*# source:", src, flags=re.M))
    assert tagged == n_fields
