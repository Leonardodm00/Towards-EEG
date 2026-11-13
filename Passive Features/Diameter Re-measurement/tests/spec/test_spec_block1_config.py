"""Spec-derived tests of Block 1 (configuration) -- integration tester, run D1,
2026-11-13. Expected values come from specs/SPEC.md (Block 1, section 4's
decisions, Blocks 9, 13, 14, 21) or from the standard library (ast, json,
hashlib); the code under test only produces the values that are checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block1_config.py -q
"""
from __future__ import annotations

import ast
import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

import spec_helpers_D1 as H  # noqa: F401  (sys.path)

from allen_diameter import config as C
from allen_diameter.analysis import phantoms

WS = H.WS
NAN = float("nan")


def cfg0():
    return C.default_config()


def replace(cfg, section, **kw):
    return dataclasses.replace(cfg, **{section: dataclasses.replace(getattr(cfg, section), **kw)})


# --------------------------------------------------------------- O1.1 -----

def test_signature_hash_names_and_form():
    cfg = cfg0()
    for which in ("estimator", "simulation", "full"):
        h = cfg.signature_hash(which)
        assert re.fullmatch(r"[0-9a-f]{16}", h), (which, h)
    for bad in ("Estimator", "sim", "", "correction", None):
        with pytest.raises(ValueError):
            cfg.signature_hash(bad)


# --------------------------------------------------------------- O1.2 -----

def test_estimator_signature_keys_and_selection():
    cfg = cfg0()
    sig = cfg.estimator_signature()
    assert set(sig) == {"acquisition", "measure", "selection"}
    # D-096: selection = {"flags": SELECTION_FLAGS}; neither faint (D-096) nor dark (D-030)
    assert C.SELECTION_FLAGS == ("crossing", "stack_edge", "bbar_few", "profile_nan")
    assert sig["selection"] == {"flags": list(C.SELECTION_FLAGS)} or \
        sig["selection"] == {"flags": C.SELECTION_FLAGS}
    assert phantoms.SELECTION_FLAGS == C.SELECTION_FLAGS


# --------------------------------------------------------------- O1.3 -----

def test_simulation_signature_is_full_without_correction():
    cfg = cfg0()
    full = cfg.full_signature()
    sim = cfg.simulation_signature()
    assert "correction" in full and "correction" not in sim
    expect = {k: v for k, v in full.items() if k != "correction"}
    assert json.dumps(sim, sort_keys=True) == json.dumps(expect, sort_keys=True)
    for k in ("acquisition", "measure", "renderer", "phantom"):
        assert k in sim


def test_hash_dependencies():
    cfg = cfg0()
    h = {w: cfg.signature_hash(w) for w in ("estimator", "simulation", "full")}
    # a correction change moves the full hash only
    c2 = replace(cfg, "correction", bias_flag_threshold=0.25)
    assert c2.signature_hash("full") != h["full"]
    assert c2.signature_hash("simulation") == h["simulation"]
    assert c2.signature_hash("estimator") == h["estimator"]
    # a renderer change moves simulation and full, not the estimator
    c3 = replace(cfg, "renderer", noise_sd_gl=cfg.renderer.noise_sd_gl + 1.0)
    assert c3.signature_hash("estimator") == h["estimator"]
    assert c3.signature_hash("simulation") != h["simulation"]
    # a sigma_fit change moves the estimator
    c4 = replace(cfg, "measure", sigma_fit_um=0.11)
    assert c4.signature_hash("estimator") != h["estimator"]


def test_selection_rule_moves_estimator_and_simulation_hash():
    """D-096: the flags of S enter the estimator signature, so another S
    changes the estimator and the simulation hashes."""
    cfg = cfg0()
    e0, s0 = cfg.signature_hash("estimator"), cfg.signature_hash("simulation")
    sig = cfg.estimator_signature()
    assert sig["selection"]["flags"]  # the rule is part of the signature
    old = C.SELECTION_FLAGS
    try:
        C.SELECTION_FLAGS = old + ("faint",)
        e1, s1 = cfg.signature_hash("estimator"), cfg.signature_hash("simulation")
    finally:
        C.SELECTION_FLAGS = old
    assert e1 != e0 and s1 != s0


# --------------------------------------------------------------- O1.4 -----

def test_pinned_default_hashes_block1():
    """SPEC Block 1 (2026-10-28, D-096): default estimator 59678d4e2aedf2b6,
    simulation 52aa73e479eb77cf."""
    cfg = cfg0()
    assert cfg.signature_hash("estimator") == "59678d4e2aedf2b6"
    assert cfg.signature_hash("simulation") == "52aa73e479eb77cf"


# --------------------------------------------------------------- O1.5 -----

def test_json_round_trip_ascii_frozen():
    cfg = cfg0()
    text = cfg.to_json()
    text.encode("ascii")  # canonical ASCII JSON
    back = C.config_from_dict(json.loads(text))
    assert back == cfg
    for w in ("estimator", "simulation", "full"):
        assert back.signature_hash(w) == cfg.signature_hash(w)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.measure.sigma_fit_um = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.renderer = None


# --------------------------------------------------------------- O1.6 -----

CHOICES = {
    # (section, field): (tuple name, expected default from the spec)
    ("measure", "mu_mode"): ("MU_MODES", "per_node"),                      # D-023
    ("measure", "bbar_region"): ("BBAR_REGIONS", "block_masked"),          # D-023
    ("measure", "focus_bg_rule"): ("FOCUS_BG_RULES", "profile_ends_median"),  # D-023
    ("measure", "focus_rule"): ("FOCUS_RULES", "gradient_energy"),         # D-030
    ("measure", "focus_grad_window"): ("FOCUS_GRAD_WINDOWS", "radius_margin"),  # D-030
    ("measure", "fit_start_rule"): ("FIT_START_RULES", "profile"),         # Block 3
    ("renderer", "absorption"): ("ABSORPTIONS", "partition_vertical"),     # D-024, D-084.2
    ("renderer", "kernel_family"): ("KERNEL_FAMILIES", "gaussian_table"),  # Block 4
    ("renderer", "kernel_continuation"): ("KERNEL_CONTINUATIONS", "linear"),  # D-024
    ("renderer", "end_cut"): ("END_CUTS", "axial"),                        # Block 2
    ("renderer", "backend"): ("RENDER_BACKENDS", "fft"),                   # Block 4
    ("correction", "response_estimator"): ("RESPONSE_ESTIMATORS", "local_linear"),  # D-063
    ("correction", "table_phi_axis"): ("TABLE_PHI_AXES", "true"),          # Block 7
    ("correction", "fill_policy"): ("FILL_POLICIES", "same_branch_then_allen"),  # Block 7
    ("correction", "table_statistic"): ("TABLE_STATISTICS", "mean"),       # Block 6
    ("phantom", "design"): ("PHANTOM_DESIGNS", "random"),                  # D-024
}


@pytest.mark.parametrize("key", sorted(CHOICES))
def test_choice_tuples_first_entry_is_the_default(key):
    section, field = key
    tup_name, expected = CHOICES[key]
    tup = getattr(C, tup_name)
    value = getattr(getattr(cfg0(), section), field)
    assert value == expected
    assert tup[0] == expected, (tup_name, tup)


def test_hybrid_is_a_choice_and_validates():
    assert "hybrid" in C.ABSORPTIONS
    cfg = replace(cfg0(), "renderer", absorption="hybrid")
    cfg.validate()
    assert cfg.signature_hash("estimator") == cfg0().signature_hash("estimator")
    assert cfg.signature_hash("simulation") != cfg0().signature_hash("simulation")


DECIDED = [
    ("measure", "sigma_fit_um", 0.099),            # D-023 / D-057 (ii)
    ("measure", "focus_grad_margin_um", 0.5),      # D-030
    ("measure", "focus_bg_ends_um", 1.5),          # section 2.2
    ("measure", "alpha_dark_flag", 1.0),           # D-024
    ("measure", "faint_min_dip_gl", 6.0),          # D-096 (PROVISIONAL 6 gl)
    ("measure", "profile_half_um", 3.0),           # section 2.2, v in [-3, 3]
    ("measure", "iter_min_first_samples", 3),      # D-069
    ("measure", "iter_line_mult", 3.0),            # D-044
    ("measure", "iter_max_rounds", 5),             # D-042
    ("measure", "iter_planes_half", 6),            # D-052.3
    ("measure", "iter_bg_margin_um", 4.0),         # D-036
    ("measure", "sgt_max_nfev", 3000),             # D-054 (iv)
    ("measure", "sgt_lambda_um", 0.5),             # D-052
    ("acquisition", "res0_um", 0.1144),            # section 2.1
    ("acquisition", "dz_um", 0.28),                # section 2.1
    ("renderer", "U_um", 10.0),                    # D-024
    ("renderer", "kernel_continuation_slope", 0.79),  # D-024 (i), D-070
    ("renderer", "ray_na", 1.4),                   # Block 9
    ("renderer", "ray_n_oil", 1.515),              # Block 9
    ("renderer", "ray_n_rho", 16),                 # Block 9
    ("renderer", "ray_n_psi", 32),                 # Block 9
    ("renderer", "ray_post_sigma_um", 0.08),       # Block 9
    ("correction", "bias_flag_threshold", 0.2),    # D5
    ("correction", "max_failure_rate", 0.2),       # D-054 (iii)
    ("correction", "failure_rate_ignore", ()),     # D-057.1
    ("correction", "tau_kernel_bandwidth", (0.15, 5.0, 0.3)),  # D-054 (ii), D-063.1
    ("correction", "loo_mass_percentile", 5.0),    # D-063.4
    ("correction", "min_neff", 10.0),              # Block 14.4
    ("correction", "bandwidth_cv_grid", (0.5, 0.7071, 1.0, 1.4142, 2.0)),  # Block 14.4
    ("phantom", "n_draws", 100000),                # D-062.2
    ("phantom", "n_realizations", 10),             # D-062.2
    ("phantom", "radius_noise_sd_um", 0.2),        # D-062.1
    ("phantom", "mu_range_per_um", (0.3, 3.0)),    # D-065.1 (Block 17)
]


@pytest.mark.parametrize("section,field,expected", DECIDED)
def test_decided_default_values(section, field, expected):
    value = getattr(getattr(cfg0(), section), field)
    if isinstance(expected, tuple):
        assert tuple(value) == expected
    else:
        assert value == expected


def test_kernel_table_default_knots_and_sigma_r0():
    """section 8: sigma_r(0.84) = 0.603 um, 0.086 um at 0.14 um, 0.438 um at
    0.56 um in the default table; sigma_r0_um equals the table at 0."""
    r = cfg0().renderer
    tab = dict(zip(r.kernel_table_delta_um, r.kernel_table_sigma_um))
    assert tab[0.84] == 0.603 and tab[0.14] == 0.086 and tab[0.56] == 0.438
    assert r.sigma_r0_um == tab[0.0]


# --------------------------------------------------------------- O1.7 -----

def test_with_overrides_identity_coercion_and_refusals():
    cfg = cfg0()
    assert C.with_overrides(cfg, {}) == cfg
    snap = cfg.to_json()
    # an int for a float field, a list for a tuple field (Block 1)
    assert cfg.renderer.black_level_gl == 0.0
    o = C.with_overrides(cfg, {"renderer.black_level_gl": 0,
                               "phantom.mu_range_per_um": list(cfg.phantom.mu_range_per_um),
                               "measure.fit_multistart_factors": list(cfg.measure.fit_multistart_factors),
                               "correction.failure_rate_ignore": []})
    assert o == cfg and o.signature_hash("full") == cfg.signature_hash("full")
    assert isinstance(o.renderer.black_level_gl, float)
    assert cfg.to_json() == snap  # input untouched
    o2 = C.with_overrides(cfg, {"measure.sigma_fit_um": 0.12, "phantom.d_range_um": [0.3, 3]})
    assert o2.measure.sigma_fit_um == 0.12 and tuple(o2.phantom.d_range_um) == (0.3, 3.0)
    assert cfg.measure.sigma_fit_um == 0.099
    for bad in ({"nosuch.field": 1}, {"measure.nosuch": 1}, {"measure": 1},
                {"measure.sigma_fit_um": -1.0}, {"renderer.absorption": "nosuch"}):
        with pytest.raises(ValueError):
            C.with_overrides(cfg, bad)


# --------------------------------------------------------------- O1.8 -----

@pytest.mark.parametrize("section,kw", [
    ("measure", dict(sigma_fit_um=0.0)),
    ("measure", dict(sigma_fit_um=-0.1)),
    ("measure", dict(sigma_fit_um=NAN)),
    ("measure", dict(iter_min_first_samples=4)),
    ("measure", dict(iter_min_first_samples=1)),
    ("renderer", dict(absorption="nosuch")),
    ("renderer", dict(jpeg_qtables=(tuple(range(1, 64)),))),           # 63 entries
    ("renderer", dict(jpeg_qtables=(tuple([0] * 64),))),               # 0 out of range
    ("renderer", dict(jpeg_qtables=(tuple([65536] * 64),))),           # too large
    ("renderer", dict(jpeg_qtables=tuple(tuple([1] * 64) for _ in range(5)))),  # 5 tables
    ("renderer", dict(jpeg_qtables=([1] * 64,))),                      # a list refused
    ("renderer", dict(sigma_r0_um=0.09)),                              # differs from the table at 0
    ("correction", dict(tau_kernel_bandwidth=(0.15, 5.0))),
    ("correction", dict(tau_kernel_bandwidth=(0.15, 5.0, -0.3))),
    ("correction", dict(tau_kernel_bandwidth=(0.15, NAN, 0.3))),
])
def test_validate_refuses(section, kw):
    with pytest.raises(ValueError):
        replace(cfg0(), section, **kw).validate()


def test_validate_accepts_tables_of_1_to_4():
    for n in (1, 2, 4):
        cfg = replace(cfg0(), "renderer", jpeg_qtables=tuple(tuple(range(1, 65)) for _ in range(n)))
        cfg.validate()
        back = C.config_from_dict(json.loads(cfg.to_json()))
        assert back.renderer.jpeg_qtables == cfg.renderer.jpeg_qtables
        assert back.signature_hash("full") == cfg.signature_hash("full")


# --------------------------------------------------------------- O1.9 -----

def test_config_from_dict_legacy_keys():
    d = json.loads(cfg0().to_json())
    d["renderer"]["jpeg_qtables_file"] = ""
    assert C.config_from_dict(d) == cfg0()
    d["renderer"]["jpeg_qtables_file"] = "camera.json"
    with pytest.raises(ValueError, match="jpeg_qtables"):
        C.config_from_dict(d)
    d = json.loads(cfg0().to_json())
    d["phantom"]["n_replicates"] = 2000
    with pytest.raises(ValueError, match="n_replicates"):
        C.config_from_dict(d)


# -------------------------------------------------------------- O1.10 -----

def _config_classes(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            fields = [s for s in node.body if isinstance(s, ast.AnnAssign)]
            if fields:
                out[node.name] = fields
    return out


def test_every_field_line_carries_a_source_tag():
    path = WS / "src" / "allen_diameter" / "config.py"
    text = path.read_text(encoding="ascii")
    lines = text.split("\n")
    tree = ast.parse(text)
    missing = []
    n = 0
    for cls, fields in _config_classes(tree).items():
        if cls == "DiameterConfig":   # the container of the sections, not parameters
            continue
        for f in fields:
            n += 1
            span = "\n".join(lines[f.lineno - 1:f.end_lineno])
            if "# source:" not in span:
                missing.append("%s.%s (line %d)" % (cls, ast.unparse(f.target), f.lineno))
    assert n > 50
    assert not missing, missing


def _attrs_read_outside_config():
    names = set()
    for root in (WS / "src", WS / "scripts"):
        for p in root.rglob("*.py"):
            if p.name == "config.py" and p.parent.name == "allen_diameter":
                continue
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    names.add(node.attr)
    return names


def test_dead_fields_are_only_the_declared_records():
    """Block 1: every field is read by code outside config.py (attribute
    access in src/ and scripts/, parsed with ast) except the declared
    records acquisition.specimen_id, measure.node_step, measure.fit_params,
    correction.inversion_method (phantom.n_replicates was removed by 14.2)."""
    declared = {"acquisition.specimen_id", "measure.node_step", "measure.fit_params",
                "correction.inversion_method", "phantom.n_replicates"}
    read = _attrs_read_outside_config()
    cfg = cfg0()
    dead = []
    for f in dataclasses.fields(cfg):
        sec = getattr(cfg, f.name)
        if not dataclasses.is_dataclass(sec):
            continue
        for g in dataclasses.fields(sec):
            if g.name not in read:
                dead.append("%s.%s" % (f.name, g.name))
    assert set(dead) <= declared, sorted(set(dead) - declared)


# -------------------------------------------------------------- O1.11 -----

def test_code_is_ascii_with_lf_endings():
    """Section 4: code destined for the cluster is pure ASCII with LF line
    endings (hpc-python-compat): every .py, .sh and .pbs under src/, scripts/
    and tests/."""
    bad = []
    for root in (WS / "src", WS / "scripts", WS / "tests"):
        for p in sorted(root.rglob("*")):
            if p.suffix not in (".py", ".sh", ".pbs") or not p.is_file():
                continue
            data = p.read_bytes()
            if b"\r" in data:
                bad.append("%s: CR" % p.relative_to(WS))
            try:
                data.decode("ascii")
            except UnicodeDecodeError as e:
                bad.append("%s: non-ASCII at byte %d" % (p.relative_to(WS), e.start))
    assert not bad, bad


@pytest.mark.xfail(strict=True, reason="D1-F4: validate() accepts NaN in 15 float fields "
                   "(bbar_mask_margin_um NaN silently disables the D-018.1 mask; NaN label thresholds never fire)")
def test_validate_refuses_nan_in_every_float_field():
    cfg = cfg0()
    accepted = []
    for sec in dataclasses.fields(cfg):
        s = getattr(cfg, sec.name)
        if not dataclasses.is_dataclass(s):
            continue
        for f in dataclasses.fields(s):
            v = getattr(s, f.name)
            if isinstance(v, float):
                bad = replace(cfg, sec.name, **{f.name: NAN})
                try:
                    bad.validate()
                    accepted.append("%s.%s" % (sec.name, f.name))
                except (ValueError, TypeError):
                    pass
    assert not accepted, accepted
