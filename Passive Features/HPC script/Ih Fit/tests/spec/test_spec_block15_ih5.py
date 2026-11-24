"""Spec-derived tests of Block 15 (run_ih_fit.py, param_spec.py; SPEC 2.8 (a), (f) and
the Block 15 entry). Integration tester, 2026-11-22.

Pure parts only (no NEURON); the fit itself is exercised end to end through
Block 17's `twin` (tests/spec/test_spec_block8_twin_recovery.py) and in the
report's step 6. Expected values:
- tau_h^{base,pk}: the closed-form maximum of the Rich et al. (2021) tau(v) whose
  constants are read from mod/Ih_human.mod (the compiled mechanism itself):
  tau(v) = f + 1 / (e^{-a - b v} + e^{-c + d v}) is largest where
  b e^{-a - b v} = d e^{-c + d v}, i.e. v* = (ln(b / d) + c - a) / (b + d).
- m_inf(v) = 1 / (1 + exp((v - v_h) / k)), v_h = -90.87 mV, k = 8.05 mV (2.8 (a)).
"""
import math
import re
from pathlib import Path

import numpy as np
import pytest

import param_spec as PS
import run_ih_fit as R

CODE_DIR = Path(__file__).resolve().parents[2]
BASE_ARGS = ["--archive-dir", "/nonexistent", "--output-dir", "/nonexistent", "--code-dir", str(CODE_DIR)]


def _mod_constants():
    txt = (CODE_DIR / "mod" / "Ih_human.mod").read_text()
    out = {}
    for k in ("vh", "kk", "aa", "bb", "cc", "dd", "ff"):
        m = re.search(r"^\s*%s\s*=\s*([-+0-9.eE]+)" % k, txt, re.M)
        out[k] = float(m.group(1))
    return out


def _tau_peak_closed_form(vshift=0.0):
    c = _mod_constants()
    v = (math.log(c["bb"] / c["dd"]) + c["cc"] - c["aa"]) / (c["bb"] + c["dd"])
    tau = c["ff"] + 1.0 / (math.exp(-c["aa"] - c["bb"] * v) + math.exp(-c["cc"] + c["dd"] * v))
    return v + vshift, tau


def test_arm_table():
    assert R.IH5_NAMES == ("Cm", "Rm", "Ra", "gbar", "kappa_tau")
    ih5, ih6 = R.ARMS["ih5"], R.ARMS["ih6"]
    assert ih5.fit_params == R.IH5_NAMES
    assert (ih5.role_rule, ih5.weighting, ih5.ls_sample_weight) == ("d207", "ls_amplitude", "peak_decay")
    pf = R.ARMS["passive_fullstep"]
    assert (pf.role_rule, pf.weighting, pf.ls_sample_weight) == ("d207", "ls_amplitude", "peak_decay")
    for a in ("ih6", "baseline_runB"):
        c = R.ARMS[a]
        assert (c.role_rule, c.weighting, c.ls_sample_weight) == ("d006", "relative", "none"), a
    # "as ih6" apart from the names and the protocol
    # renewed 2026-11-24 (Block 20): ArmConfig gained `refine`, nelder_mead on ih5 and none on ih6 by design
    for f in ih6.__dataclass_fields__:
        if f not in ("name", "fit_params", "role_rule", "weighting", "ls_sample_weight", "refine"):
            assert getattr(ih5, f) == getattr(ih6, f), f


def test_overrides_and_defaults():
    a = R._parse_args(BASE_ARGS + ["--arm", "ih5"])
    c = R.resolve_arm_config(a)
    assert (c.role_rule, c.weighting, c.ls_sample_weight) == ("d207", "ls_amplitude", "peak_decay")
    # renewed 2026-11-24: the default gbar box is [1e-5, 1e-3] since SPEC 2.9 (a) (Block 20)
    assert a.ls_tau_d_ms == 500.0 and a.gbar_bounds == "1e-5,1e-3"
    assert PS.DEFAULT_GBAR_BOUNDS == (1e-5, 1e-3)
    a = R._parse_args(BASE_ARGS + ["--arm", "ih5", "--ls-role-rule", "d006", "--weighting", "relative",
                                   "--ls-sample-weight", "none"])
    c = R.resolve_arm_config(a)
    assert (c.role_rule, c.weighting, c.ls_sample_weight) == ("d006", "relative", "none")
    a = R._parse_args(BASE_ARGS + ["--arm", "ih6"])
    assert R.resolve_arm_config(a).weighting == "relative"         # today's default is the arm's value
    with pytest.raises(SystemExit) as e:
        R._parse_args(BASE_ARGS + ["--arm", "ih5", "--ls-role-rule", "d999"])
    assert e.value.code == 2


@pytest.mark.parametrize("vshift", [0.0, -6.0, 4.5])
def test_tau_h_base_peak(vshift):
    import ih_mechanism as IM
    a = R._parse_args(BASE_ARGS + ["--arm", "ih5", "--vshift-base=%r" % vshift])
    spec = R.build_ih_spec(a)
    base = R.tau_h_base_peak_ms(spec)
    v_star, tau_star = _tau_peak_closed_form(vshift)
    assert -140.0 <= v_star <= -40.0
    # 1e-3 ms: the code scans a 0.01 mV grid; near the maximum the error is (1/2)|tau''| (0.005 mV)^2,
    # far below 1e-3 ms for this curve
    assert abs(base - tau_star) <= 1e-3, (base, tau_star)
    assert 340.0 <= base <= 346.0
    assert isinstance(spec, IM.IhSpec)


def test_kappa_box_from_tau_ms():
    a = R._parse_args(BASE_ARGS + ["--arm", "ih5", "--tau-h-pk-bounds-ms", "50,1000", "--kappa-bounds", "0.7,0.8"])
    base = R.tau_h_base_peak_ms(R.build_ih_spec(a))
    spec = R.build_spec_from_args(a, R.resolve_arm_config(a))
    assert spec.names == R.IH5_NAMES
    ax = spec.axis("kappa_tau")
    assert ax.log
    assert ax.lo == pytest.approx(50.0 / base, rel=1e-15) and ax.hi == pytest.approx(1000.0 / base, rel=1e-15)
    g = spec.axis("gbar")
    assert (g.lo, g.hi, g.log) == (1e-5, 1e-3, True)     # renewed 2026-11-24, SPEC 2.9 (a)
    assert "dv_h" not in spec.names
    # without the option --kappa-bounds holds
    a = R._parse_args(BASE_ARGS + ["--arm", "ih5", "--kappa-bounds", "0.7,0.8"])
    ax = R.build_spec_from_args(a, R.resolve_arm_config(a)).axis("kappa_tau")
    assert (ax.lo, ax.hi) == (0.7, 0.8)


def test_dv_h_zero_on_ih5_cell():
    """The ih5 spec applies dv_h = 0 (2.8 (a)): the apply function reads a missing dv_h as 0."""
    calls = []

    class Cell:
        def set_passive(self, **k):
            calls.append(("p", k))

    import ih_mechanism as IM
    orig_set, orig_bal = IM.set_ih, IM.balance_e_pas
    try:
        IM.set_ih = lambda cell, g, dv, kap: calls.append(("ih", g, dv, kap))
        IM.balance_e_pas = lambda cell, v: calls.append(("bal", v))
        spec = PS.make_ih_spec(dvh_bounds=None, kappa_bounds=(0.2, 3.0))
        spec.apply(Cell(), dict(Cm=1.0, Rm=2e4, Ra=150.0, gbar=1e-4, kappa_tau=1.3), -65.0)
    finally:
        IM.set_ih, IM.balance_e_pas = orig_set, orig_bal
    assert ("ih", 1e-4, 0.0, 1.3) in calls


@pytest.mark.parametrize("text,ok", [
    ("Cm=0.9,Rm=30000,Ra=150,gbar=1e-4,kappa_tau=1.1", True),
    ("kappa_tau=1.1, gbar=1e-4 ,Ra=150,Rm=3e4,Cm=0.9", True),
    ("Cm=0.9,Rm=30000,Ra=150,gbar=1e-4", False),                         # missing
    ("Cm=0.9,Rm=30000,Ra=150,gbar=1e-4,kappa_tau=1.1,dv_h=0", False),    # extra
    ("Cm=0.9,Cm=0.8,Rm=30000,Ra=150,gbar=1e-4,kappa_tau=1.1", False),    # twice
    ("Cm=abc,Rm=30000,Ra=150,gbar=1e-4,kappa_tau=1.1", False),
    ("Cm=0.9,Rm=30000,Ra=150,gbar=nan,kappa_tau=1.1", False),
    ("Cm=0.9,Rm=30000,Ra=150,gbar=0,kappa_tau=1.1", False),
    ("Cm0.9,Rm=30000,Ra=150,gbar=1e-4,kappa_tau=1.1", False),
])
def test_eval_theta(text, ok):
    if ok:
        th = R.parse_eval_theta(text, R.IH5_NAMES)
        assert list(th) == list(R.IH5_NAMES)
        assert th["Cm"] == 0.9 and th["Rm"] == 30000.0 and th["gbar"] == 1e-4 and th["kappa_tau"] == 1.1
        R._parse_args(BASE_ARGS + ["--arm", "ih5", "--eval-theta", text])
    else:
        with pytest.raises(ValueError):
            R.parse_eval_theta(text, R.IH5_NAMES)
        with pytest.raises(SystemExit) as e:
            R._parse_args(BASE_ARGS + ["--arm", "ih5", "--eval-theta=" + text])
        assert e.value.code == 2


def test_eval_theta_six_names_for_ih6():
    R.parse_eval_theta("Cm=1,Rm=2e4,Ra=100,gbar=1e-4,dv_h=-2.5,kappa_tau=1", R.IH6_NAMES)
    with pytest.raises(SystemExit):
        R._parse_args(BASE_ARGS + ["--arm", "ih6", "--eval-theta", "Cm=1,Rm=2e4,Ra=100,gbar=1e-4,kappa_tau=1"])


def test_minf_and_gh():
    c = _mod_constants()
    import ih_mechanism as IM
    spec = IM.IhSpec()
    assert spec.mechanism == "Ih_human"
    for v in (-140.0, -110.3, -90.87, -75.0, -60.0):
        m = 1.0 / (1.0 + math.exp((v - (-90.87)) / 8.05))
        assert c["vh"] == -90.87 and c["kk"] == 8.05
        g = R.gh_at_voltage_S_cm2(3.2e-4, v, spec)
        # 1e-12 relative: the same two operations in another order
        assert g == pytest.approx(3.2e-4 * m, rel=1e-12)
    assert math.isnan(R.gh_at_voltage_S_cm2(1e-4, float("nan"), spec))


def test_v_deepest_train():
    recs = {"records": [dict(role="train", v_trough_mV=-80.0, polarity="hyp"),
                        dict(role="validate", v_trough_mV=-95.0, polarity="hyp"),
                        dict(role="train", v_trough_mV=-88.5, polarity="hyp"),
                        dict(role="report", v_trough_mV=-99.0, polarity="hyp"),
                        dict(role="validate", v_trough_mV=float("nan"), polarity="dep")]}
    assert R.v_deepest_train_mV(recs) == -88.5
    assert math.isnan(R.v_deepest_train_mV({"records": [dict(role="validate", v_trough_mV=-90.0)]}))
    assert math.isnan(R.v_deepest_train_mV(None))


def test_fit_record_columns_listed():
    assert R.FIT_RECORD_COLUMNS == ("tau_h_pk_ms", "tau_h_base_pk_ms", "ls_role_rule", "weighting",
                                    "ls_sample_weight", "ls_tau_d_ms", "loss_at_estimate", "loss_at_eval_theta",
                                    "v_deepest_train_mV", "gh_at_v_deepest_S_cm2")


def test_ascii_lf():
    for f in ("run_ih_fit.py", "param_spec.py"):
        raw = (CODE_DIR / f).read_bytes()
        assert all(c < 128 for c in raw) and b"\r" not in raw, f
