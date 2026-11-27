"""Spec-derived tests of Block 4, amendments 1 and 2 (kernel.with_in_focus_width, kernel.with_kernel; tester,
2026-11-22). Written from specs/SPEC.md (Block 4 amendments 1 and 2; D-165, D-201) before the code was read.

Expected values are the amendments' formulas evaluated with numpy, and Block 4's kernel law as an earlier
tester wrote it (tests/spec/spec_helpers.ref_sigma_r); the code under test only produces what is checked.

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from spec_helpers import ref_sigma_r  # noqa: E402

from allen_diameter import config as C  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402


def rcfg(**kw):
    r = C.default_config().renderer
    return dataclasses.replace(r, **kw) if kw else r


def test_with_in_focus_width_amendment1():
    r = rcfg()
    old = np.asarray(r.kernel_table_sigma_um)
    for s0 in (0.05, 0.20, 0.30):
        n = K.with_in_focus_width(r, s0)
        new = np.asarray(n.kernel_table_sigma_um)
        # new minus old variance = s0^2 - sigma_r(0)^2 at every knot (1e-15 um^2: a few float64 operations on 0.1)
        assert_allclose(new ** 2 - old ** 2, s0 ** 2 - old[0] ** 2, rtol=0, atol=1e-15)
        assert new[0] == s0 and n.sigma_r0_um == s0                      # exact
        assert (n.kernel_continuation, n.kernel_continuation_slope) == (r.kernel_continuation,
                                                                         r.kernel_continuation_slope)
        assert n.kernel_table_delta_um == r.kernel_table_delta_um
        dataclasses.replace(C.default_config(), renderer=n).validate()
    same = K.with_in_focus_width(r, float(old[0]))
    assert_allclose(same.kernel_table_sigma_um, old, rtol=0, atol=1e-15)
    falling = rcfg(kernel_table_sigma_um=(0.2, 0.1, 0.12, 0.26, 0.44, 0.6), sigma_r0_um=0.2)
    for bad_r, s0 in ((rcfg(kernel_family="empirical"), 0.1), (r, 0.0), (r, -0.1), (r, np.nan), (r, np.inf),
                      (falling, 0.05)):
        with pytest.raises(ValueError):
            K.with_in_focus_width(bad_r, s0)


@pytest.mark.parametrize("cont", ["linear", "frozen", "proportional"])
def test_with_kernel_amendment2(cont):
    r = rcfg(kernel_continuation=cont)
    old = np.asarray(r.kernel_table_sigma_um)
    for s0, c in ((0.15, 1.3), (0.08, 0.5), (0.3, 4.0), (0.12, 1.0)):
        n = K.with_kernel(r, s0, c)
        new = np.asarray(n.kernel_table_sigma_um)
        assert_allclose(new ** 2, c * (old ** 2 - old[0] ** 2) + s0 ** 2, rtol=1e-14)
        assert new[0] == s0 and n.sigma_r0_um == s0
        dataclasses.replace(C.default_config(), renderer=n).validate()
        # the law read back through sigma_r equals Block 4's law (independent) on the new table
        delta = np.concatenate([np.linspace(-3.0, 3.0, 241), [1e4]])
        assert_allclose(K.sigma_r(delta, n), ref_sigma_r(delta, n), rtol=1e-14, atol=0)
        if cont == "linear":
            assert_allclose(n.kernel_continuation_slope, r.kernel_continuation_slope * math.sqrt(c), rtol=1e-15)
            ratio = float(K.sigma_r(1e4, n)) / float(K.sigma_r(1e4, r))
            assert abs(ratio - math.sqrt(c)) <= 1e-3                     # where the growth dominates
        else:
            assert n.kernel_continuation_slope == r.kernel_continuation_slope
        if cont == "proportional":                                       # follows the new table's last knot
            assert_allclose(K.sigma_r(2.0, n), new[-1] * 2.0 / r.kernel_table_delta_um[-1], rtol=1e-14)
        if cont == "frozen":
            assert_allclose(K.sigma_r(5.0, n), new[-1], rtol=0, atol=0)
    one = K.with_kernel(r, 0.2, 1.0)
    ref = K.with_in_focus_width(r, 0.2)
    assert dataclasses.asdict(one) == dataclasses.asdict(ref)            # c = 1: every field equal, bit for bit


def test_with_kernel_refusals():
    r = rcfg()
    falling = rcfg(kernel_table_sigma_um=(0.2, 0.1, 0.12, 0.26, 0.44, 0.6), sigma_r0_um=0.2)
    for args in ((rcfg(kernel_family="empirical"), 0.1, 1.0), (r, 0.0, 1.0), (r, 0.1, 0.0), (r, 0.1, -1.0),
                 (r, np.nan, 1.0), (r, 0.1, np.inf), (falling, 0.05, 1.0)):
        with pytest.raises(ValueError):
            K.with_kernel(*args)
    r0 = rcfg()
    K.with_kernel(r0, 0.2, 2.0)
    assert r0 == rcfg()                                                  # the input is a frozen value, unchanged


# ---------------------------------------------------------------------------------------------------------------
# Block 4, amendment 4 (2026-11-27; D-181, D-265, D-272, D-275): the spline reading "gaussian_table_spline" and the
# continuation "tangent" (tester, 2026-11-27). Written from SPEC.md's amendment text before kernel.py was read.
# Reference: the amendment's formula evaluated here with scipy.interpolate.CubicSpline (the library the amendment
# names) on signed knots built here; kernel.growth_spline and kernel.sigma_r are only called to be checked.
# ---------------------------------------------------------------------------------------------------------------

from scipy.interpolate import CubicSpline  # noqa: E402

PILOT_TABLE = dict(kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84), kernel_table_sigma_um=(0.080, 0.1536, 0.3924, 0.5620),
                   sigma_r0_um=0.080)


def _ref_spline(r):
    k = np.asarray(r.kernel_table_delta_um, float)
    s = np.asarray(r.kernel_table_sigma_um, float)
    g = s ** 2 - s[0] ** 2
    return CubicSpline(np.r_[-k[:0:-1], k], np.r_[g[:0:-1], g], bc_type="not-a-knot")


def _ref_sigma_spline(delta, r):
    d = np.abs(np.asarray(delta, float))
    k = np.asarray(r.kernel_table_delta_um, float)
    s = np.asarray(r.kernel_table_sigma_um, float)
    S = _ref_spline(r)
    dm = k[-1]
    inside = np.sqrt(s[0] ** 2 + np.maximum(S(np.minimum(d, dm)), 0.0))
    c = r.kernel_continuation
    if c == "tangent":
        out = np.sqrt(s[0] ** 2 + np.maximum(S(dm) + S(dm, 1) * (d - dm), 0.0))
    else:
        smax = math.sqrt(s[0] ** 2 + max(float(S(dm)), 0.0))     # sigma_r(delta_max) of the spline = the knot value
        gam = {"linear": r.kernel_continuation_slope, "frozen": 0.0, "proportional": smax / dm}[c]
        out = smax + gam * (d - dm)
    return np.where(d > dm, out, inside)


def _sp(**kw):
    return rcfg(kernel_family="gaussian_table_spline", **kw)


@pytest.mark.parametrize("table", ["default", "pilot"])
@pytest.mark.parametrize("cont", ["linear", "frozen", "proportional", "tangent"])
def test_am4_spline_reading_against_reference(table, cont):
    r = _sp(kernel_continuation=cont, **({} if table == "default" else PILOT_TABLE))
    delta = np.concatenate([np.linspace(-3.0, 3.0, 601), [0.0, 0.28, 0.56, 0.84, -0.84, 1e3]])
    got = np.asarray(K.sigma_r(delta, r))
    # 1e-13 um: the same closed form evaluated twice; only operation order may differ
    assert_allclose(got, _ref_sigma_spline(delta, r), rtol=0, atol=1e-13)
    assert_allclose(K.sigma_r(-delta, r), got, rtol=0, atol=0)                 # even in delta, exactly
    # knot values returned (1e-13 um, the amendment's own tolerance)
    k = np.asarray(r.kernel_table_delta_um)
    assert_allclose(K.sigma_r(k, r), r.kernel_table_sigma_um, rtol=0, atol=1e-13)
    assert np.all(got >= r.kernel_table_sigma_um[0] - 1e-15)                    # the floor (D-275)
    # continuity at the last knot (1e-9 um over a 1e-10 um step)
    dm = k[-1]
    assert abs(float(K.sigma_r(dm + 1e-10, r)) - float(K.sigma_r(dm, r))) < 1e-9


def test_am4_growth_spline_is_the_not_a_knot_spline():
    for kw in ({}, PILOT_TABLE):
        r = _sp(**kw)
        S, ref = K.growth_spline(r), _ref_spline(r)
        x = np.linspace(-1.4, 1.4, 281)
        assert_allclose(S(x), ref(x), rtol=0, atol=1e-12)                      # um^2
        assert_allclose(S(x, 1), ref(x, 1), rtol=0, atol=1e-11)


def test_am4_pilot_numbers_of_d265_and_d275():
    r = _sp(kernel_continuation="tangent", **PILOT_TABLE)
    S = K.growth_spline(r)
    assert abs(float(S(0.84, 1)) - 0.469) <= 5e-4                              # the log's digits
    assert_allclose(K.sigma_r(np.array([0.98, 1.12, 1.28]), r), [0.618, 0.669, 0.723], rtol=0, atol=5e-4)
    # D-275: the unfloored spline is negative near 0 on this table; the code's sigma_r floors it
    t = np.linspace(0.0, 0.84, 84001)
    v = _ref_spline(r)(t)
    assert v.min() < 0
    assert_allclose(K.sigma_r(t[v < 0], r), 0.080, rtol=0, atol=1e-15)
    # the SPEC's figures: min -6.06e-4 um^2 at 0.090 um, negative for |delta| < 0.130 um (checked loosely: see report)
    assert abs(v.min() - (-6.06e-4)) < 2e-6
    # the default six-knot table is never negative (SPEC [run])
    vd = _ref_spline(_sp())(np.linspace(0, rcfg().kernel_table_delta_um[-1], 20001))
    assert vd.min() >= -1e-15


def test_am4_first_reading_unchanged_bit_for_bit():
    r = rcfg()
    assert r.kernel_family == "gaussian_table"
    d = np.linspace(0, r.kernel_table_delta_um[-1], 1001)
    assert np.array_equal(K.sigma_r(d, r), np.interp(d, r.kernel_table_delta_um, r.kernel_table_sigma_um))
    assert_allclose(K.sigma_r(np.linspace(-4, 4, 161), r), ref_sigma_r(np.linspace(-4, 4, 161), r), rtol=0, atol=0)


def test_am4_refusals():
    with pytest.raises(ValueError):
        K.sigma_r(0.3, rcfg(kernel_continuation="tangent"))                   # tangent needs the spline family
    with pytest.raises(ValueError):
        K.sigma_r(0.3, _sp(kernel_table_delta_um=(0.0, 0.5), kernel_table_sigma_um=(0.1, 0.3)))  # two knots
    for bad in (np.nan, np.inf, np.array([0.1, np.nan])):
        with pytest.raises(ValueError):
            K.sigma_r(bad, _sp())
    with pytest.raises(NotImplementedError):
        K.sigma_r(0.3, rcfg(kernel_family="empirical"))


def test_am4_with_kernel_keeps_the_family():
    assert set(K.TABLE_FAMILIES) == {"gaussian_table", "gaussian_table_spline"}
    for cont in ("linear", "tangent"):
        r = _sp(kernel_continuation=cont, **PILOT_TABLE)
        a = K.with_in_focus_width(r, 0.1)
        b = K.with_kernel(r, 0.1, 1.5)
        for n in (a, b):
            assert n.kernel_family == "gaussian_table_spline" and n.kernel_continuation == cont
        old = np.asarray(r.kernel_table_sigma_um)
        assert_allclose(np.asarray(b.kernel_table_sigma_um) ** 2, 1.5 * (old ** 2 - old[0] ** 2) + 0.01, rtol=1e-14)


def test_am4_config_choice_tuples_and_hashes():
    assert "gaussian_table_spline" in C.KERNEL_FAMILIES and "tangent" in C.KERNEL_CONTINUATIONS
    assert C.KERNEL_FAMILIES[0] == "gaussian_table"                              # default still the first reading
    c = C.default_config()
    assert (c.signature_hash("estimator"), c.signature_hash("simulation")) == ("16be09f944e45972", "dc84ed2e89bcfea6")
    dataclasses.replace(c, renderer=_sp(kernel_continuation="tangent")).validate()


# ---------------------------------------------------------------------------------------------------------------
# validate() and Block 4 amendment 4's refusals (tester, 2026-11-27 22:11; commit 98df4b0, "validate() refuses
# tangent without the spline and a two-knot spline", answering F3 of the 1904 report). Oracles: (V1) over every
# table family x continuation x number of knots (2, 3, 4, 6) on a valid ascending table, DiameterConfig.validate()
# refuses exactly the renderer configurations sigma_r refuses at render time (ValueError both); (V2) the refusal
# reaches with_overrides and config_from_dict; (V3) the default configuration validates and its hashes are
# unchanged.
# ---------------------------------------------------------------------------------------------------------------

_TABLES = {2: ((0.0, 0.84), (0.08, 0.56)), 3: ((0.0, 0.42, 0.84), (0.08, 0.2, 0.56)),
           4: ((0.0, 0.28, 0.56, 0.84), (0.080, 0.1536, 0.3924, 0.5620)),
           6: ((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), (0.08, 0.1, 0.2, 0.3, 0.45, 0.6))}


@pytest.mark.parametrize("fam", ["gaussian_table", "gaussian_table_spline"])
@pytest.mark.parametrize("cont", ["linear", "frozen", "proportional", "tangent"])
@pytest.mark.parametrize("nk", sorted(_TABLES))
def test_V1_validate_refuses_what_sigma_r_refuses(fam, cont, nk):
    d, s = _TABLES[nk]
    c = C.default_config()
    r = dataclasses.replace(c.renderer, kernel_family=fam, kernel_continuation=cont, kernel_table_delta_um=d,
                            kernel_table_sigma_um=s, sigma_r0_um=s[0], kernel_continuation_slope=0.5)
    try:
        K.sigma_r(np.array([0.0, 0.3, 1.5]), r)
        render_ok = True
    except ValueError:
        render_ok = False
    try:
        dataclasses.replace(c, renderer=r).validate()
        valid = True
    except ValueError:
        valid = False
    assert valid == render_ok, (fam, cont, nk, valid, render_ok)


def test_V2_refusal_through_overrides_and_dict():
    c = C.default_config()
    with pytest.raises(ValueError):
        C.with_overrides(c, {"renderer.kernel_continuation": "tangent"})
    with pytest.raises(ValueError):
        C.with_overrides(c, {"renderer.kernel_family": "gaussian_table_spline",
                             "renderer.kernel_table_delta_um": [0.0, 0.84],
                             "renderer.kernel_table_sigma_um": [0.08, 0.56], "renderer.sigma_r0_um": 0.08})
    import json as _json
    d = _json.loads(c.to_json())
    d["renderer"]["kernel_continuation"] = "tangent"
    with pytest.raises(ValueError):
        C.config_from_dict(d).validate()
    ok = C.with_overrides(c, {"renderer.kernel_family": "gaussian_table_spline",
                              "renderer.kernel_continuation": "tangent"})
    assert ok.renderer.kernel_continuation == "tangent"


def test_V3_default_validates_and_hashes_unchanged():
    c = C.default_config()
    c.validate()
    assert (c.signature_hash("estimator"), c.signature_hash("simulation")) == ("16be09f944e45972", "dc84ed2e89bcfea6")
