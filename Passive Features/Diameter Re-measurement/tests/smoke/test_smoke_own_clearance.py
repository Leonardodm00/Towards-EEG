"""Smoke test for the same-node clearance -- the cleared profiles and the chord
per plane -- Block 37 in specs/SPEC.md (D-151.1, D-156.2).

Checks
    test_known_answer   profiles synthesised as B (1 - S_in - S_out) from two
                        non-negative Gaussian arrays of different widths: the
                        residual is 0 and the cleared profile equals the
                        in-band model [1e-12 gl at B = 92]; s_out = 0 returns
                        the profiles as float64 [exact]; the chord is d at
                        z_k = z_ax and 0 at and beyond |z_k - z_ax| cos phi =
                        d / 2
    test_reference      the chord against an independent reference: the
                        extent of the offsets v on a grid of 1e-4 um at which
                        (0, v, z_k - z_ax) lies within d/2 of the line through
                        the origin with direction (cos phi, 0, sin phi), the
                        distance by the cross product, for d in {0.5, 2.44}
                        um, tilt 0, 30, 60 deg, nine depths [2e-4 um, the
                        grid]; the peaks, the in-band share and the residual's
                        root mean square against loops written by hand
    test_convergence    SKIP (no discretisation in Block 37)
    test_invariants     for any profiles, cleared_gl - model_in_gl equals
                        residual_gl [1e-12 gl]; residual_gl = I - model_gl;
                        layers_gl equals B_k times layers [exact]; a plane
                        whose whole shadow is 0 has share NaN and peaks 0
    test_contract       shapes, float64, finiteness (in_share NaN only where
                        the whole shadow's maximum is not positive);
                        layers None passes through; chord_um keeps dz's shape
    test_determinism    two calls give bit-identical output; the inputs are
                        not changed
    test_edge_cases     ValueError for shapes that disagree, a non-finite
                        input, B <= 0, d <= 0, a split entry below -1e-9
                        (and -1e-10 accepted)

line_fast.line_shadow_split (Block 35, amendment 2) is not imported: the split
arrays are built here from the contract.

Run
    cd "Passive Features/Diameter Re-measurement"
    python3 tests/smoke/test_smoke_own_clearance.py
    python3 -m pytest tests/smoke/test_smoke_own_clearance.py -q

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import importlib.metadata
import math
import platform
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import own_clearance as OC  # noqa: E402

SEED = 20261117
REPORT_PACKAGES = ("numpy",)
ARTIFACT_DIR = HERE / "artifacts"
_SAVED_ARTIFACTS = []
MEASURED = {}          # largest deviation per oracle, printed by main() after the table
B_REF = 92.0           # gl, the SPEC's background for the known answer
PX = 0.1144            # um, a line step of the order of the pixel


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name, equal_nan=True)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / f"{safe_name}.npz"
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def record(name, value):
    MEASURED[name] = max(float(value), MEASURED.get(name, 0.0))


def maxdev(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 0.0 if a.size == 0 else float(np.max(np.abs(a - b)))


def gaussian_split(n_planes=7, n_v=61, rng=None):
    """Two non-negative arrays (n_planes, n_v), Gaussians in v of different widths and plane-dependent
    amplitudes: S_in narrow (sd 0.15 um), S_out wide (sd 0.45 um), offset by 0.05 um, so the in-band share
    at the maximum differs from plane to plane. Plane 0's amplitudes are 0 (a plane with no shadow)."""
    v = (np.arange(n_v) - (n_v - 1) / 2.0) * PX
    k = np.arange(n_planes) - (n_planes - 1) // 2
    a_in = 0.6 * np.exp(-0.5 * (k / 1.5) ** 2)
    a_out = 0.25 + 0.05 * np.abs(k)
    if rng is not None:
        a_in = a_in * rng.uniform(0.8, 1.2, n_planes)
        a_out = a_out * rng.uniform(0.8, 1.2, n_planes)
    a_in[0] = 0.0
    a_out[0] = 0.0
    s_in = a_in[:, None] * np.exp(-0.5 * (v / 0.15) ** 2)[None, :]
    s_out = a_out[:, None] * np.exp(-0.5 * ((v - 0.05) / 0.45) ** 2)[None, :]
    return v, k, s_in, s_out


def default_inputs(rng=None, B=None):
    v, k, s_in, s_out = gaussian_split(rng=rng)
    n_planes, n_v = s_in.shape
    if B is None:
        B = np.full(n_planes, B_REF)
    planes_z = 0.28 * k.astype(float)
    if rng is None:
        I = B[:, None] * (1.0 - s_in - s_out)
    else:
        I = B[:, None] * (1.0 - s_in - s_out) + rng.normal(0.0, 2.0, (n_planes, n_v))
    return dict(profiles_gl=I, B_gl=B, s_in=s_in, s_out=s_out, planes_z=planes_z, d_um=1.0,
                z_ax_um=0.07, phi_rad=math.radians(30.0))


def chord_reference(d, dz, phi, h=1e-4):
    """Independent chord: the extent of the offsets v on a grid of step h at which the point (0, v, dz)
    lies within d/2 of the line through the origin with direction u = (cos phi, 0, sin phi), the distance
    |p x u| (u is a unit vector). The extent is measured as (number of grid points inside) x h, whose error
    is within +-h; max - min of the points inside is biased low by up to 2 h and would sit on the SPEC's
    2e-4 um bound (measured 1.99999999999976e-4 um with it)."""
    n = int(math.ceil((0.5 * d + 0.01) / h))
    v = np.arange(-n, n + 1) * h
    p = np.stack([np.zeros_like(v), v, np.full_like(v, dz)], axis=1)
    u = np.array([math.cos(phi), 0.0, math.sin(phi)])
    dist = np.linalg.norm(np.cross(p, u), axis=1)
    return float(np.count_nonzero(dist <= 0.5 * d)) * h


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    # (1) Profiles synthesised exactly as the whole model: R = 0 and I_clr = M_in.
    # Tolerance 1e-12 gl (SPEC): the values are O(B) = 92 gl, so float64 rounding of the few products and
    # sums is a few ulps of 92, about 1e-14 gl; 1e-12 leaves two orders of margin and nothing more.
    for B in (np.full(7, B_REF), np.linspace(60.0, 210.0, 7)):
        inp = default_inputs(B=B)
        out = OC.clear_profiles(**inp)
        check_close("known_residual_zero", out.residual_gl, np.zeros_like(out.residual_gl), rtol=0, atol=1e-12)
        check_close("known_cleared_eq_model_in", out.cleared_gl, out.model_in_gl, rtol=0, atol=1e-12)
        check_close("known_rms_zero", out.residual_rms_gl, 0.0, rtol=0, atol=1e-12)
        record("known: max |residual| (gl)", np.max(np.abs(out.residual_gl)))
        record("known: max |cleared - model_in| (gl)", maxdev(out.cleared_gl, out.model_in_gl))

    # (2) s_out = 0 returns the profiles as float64 [exact]; integer profiles too.
    inp = default_inputs(rng=make_rng())
    inp["s_out"] = np.zeros_like(inp["s_in"])
    out = OC.clear_profiles(**inp)
    assert out.cleared_gl.dtype == np.float64
    assert_array_equal(out.cleared_gl, inp["profiles_gl"])
    inp8 = dict(inp)
    inp8["profiles_gl"] = np.clip(np.round(inp["profiles_gl"]), 0, 255).astype(np.uint8)
    out8 = OC.clear_profiles(**inp8)
    assert out8.cleared_gl.dtype == np.float64
    assert_array_equal(out8.cleared_gl, inp8["profiles_gl"].astype(np.float64))

    # (3) The chord: d at z_k = z_ax [exact: sqrt((d/2)^2) * 2 = d in float64 for these d], 0 at and
    # beyond the edge.
    for d in (0.5, 1.0, 2.44):
        for phi_deg in (0.0, 30.0, 60.0, 89.0):
            c0 = OC.chord_um(d, 0.0, math.radians(phi_deg))
            assert float(c0) == d, (d, phi_deg, float(c0))
    # At tilt 0 the edge |dz| = d/2 is exactly representable for these d: exactly 0 at and beyond it.
    for d in (0.5, 1.0, 2.44):
        dz = np.array([0.5 * d, -0.5 * d, 0.5 * d + 1e-6, -d, 3.0 * d, 1e3])
        assert_array_equal(OC.chord_um(d, dz, 0.0), np.zeros_like(dz))
    # Tilted: dz = (d/2) / cos(phi) puts the point on the edge only up to the rounding of the division and
    # of cos; |c| <= 2 sqrt(d * |a - d/2|) with |a - d/2| a few ulps of d/2, so c <= 2 sqrt(d^2 * 4 eps)
    # ~ 6e-8 d. Tolerance 1e-7 um. Beyond the edge (by 1e-6 um or more) it is exactly 0.
    worst = 0.0
    for d in (0.5, 2.44):
        for phi_deg in (30.0, 60.0):
            phi = math.radians(phi_deg)
            edge = 0.5 * d / math.cos(phi)
            c_edge = OC.chord_um(d, np.array([edge, -edge]), phi)
            worst = max(worst, float(np.max(c_edge)))
            beyond = np.array([edge + 1e-6, -(edge + 1e-6), 2.0 * edge, 50.0])
            assert_array_equal(OC.chord_um(d, beyond, phi), np.zeros(4))
    assert worst <= 1e-7, worst
    record("known: chord at the tilted edge (um)", worst)


def test_reference():
    # (1) The chord against the independent grid reference, d in {0.5, 2.44} um, tilt 0, 30, 60 deg, nine
    # depths each spanning inside and beyond the edge. Tolerance 2e-4 um (SPEC, the grid): the count of the
    # grid points inside times the step 1e-4 um is within one step of the chord, so the bound has a factor
    # 2 of margin.
    worst = 0.0
    n_cases = 0
    for d in (0.5, 2.44):
        for phi_deg in (0.0, 30.0, 60.0):
            phi = math.radians(phi_deg)
            dzs = np.linspace(-1.2, 1.2, 9) * (0.5 * d / math.cos(phi))   # nine depths; the outer two beyond
            got = OC.chord_um(d, dzs, phi)
            ref = np.array([chord_reference(d, dz, phi) for dz in dzs])
            check_close("chord_reference_d%g_phi%g" % (d, phi_deg), got, ref, rtol=0, atol=2e-4)
            worst = max(worst, maxdev(got, ref))
            n_cases += dzs.size
    record("reference: chord vs grid (um)", worst)
    MEASURED["reference: chord cases"] = n_cases

    # (2) The peaks, the share and the rms against loops written by hand, on noisy profiles, varying B,
    # random positive split arrays (and the Gaussian ones). The quantities are a product of two float64
    # numbers (peaks), a quotient (share) or the root of a mean of squares (rms): rtol 1e-12 covers the
    # ordering differences of numpy's pairwise sums against the loop.
    rng = make_rng()
    cases = [default_inputs(rng=rng, B=np.linspace(60.0, 210.0, 7))]
    for _ in range(5):
        n_planes, n_v = int(rng.integers(1, 9)), int(rng.integers(1, 40))
        s_in = rng.uniform(0.0, 0.5, (n_planes, n_v)) * (rng.uniform(size=(n_planes, n_v)) > 0.3)
        s_out = rng.uniform(0.0, 0.5, (n_planes, n_v)) * (rng.uniform(size=(n_planes, n_v)) > 0.3)
        B = rng.uniform(50.0, 220.0, n_planes)
        I = rng.uniform(0.0, 255.0, (n_planes, n_v))
        cases.append(dict(profiles_gl=I, B_gl=B, s_in=s_in, s_out=s_out,
                          planes_z=rng.uniform(-1, 1, n_planes), d_um=float(rng.uniform(0.2, 3.0)),
                          z_ax_um=float(rng.uniform(-0.5, 0.5)), phi_rad=float(rng.uniform(0, 1.5))))
    dev = {"peaks": 0.0, "share": 0.0, "rms": 0.0, "fields": 0.0}
    for inp in cases:
        out = OC.clear_profiles(**inp)
        I, B, si, so = inp["profiles_gl"], inp["B_gl"], inp["s_in"], inp["s_out"]
        n_planes, n_v = I.shape
        for k in range(n_planes):
            ip = B[k] * max(si[k, n] for n in range(n_v))
            op = B[k] * max(so[k, n] for n in range(n_v))
            best, n_star = -1.0, -1
            for n in range(n_v):
                w = si[k, n] + so[k, n]
                if w > best:                      # strict: the first index on a tie (numpy's argmax)
                    best, n_star = w, n
            share = si[k, n_star] / best if best > 0 else float("nan")
            acc = 0.0
            for n in range(n_v):
                m = B[k] * (1.0 - si[k, n] - so[k, n])
                r = I[k, n] - m
                acc += r * r
                check_close("ref_cleared", out.cleared_gl[k, n], I[k, n] + B[k] * so[k, n], rtol=1e-15, atol=0)
                check_close("ref_model_in", out.model_in_gl[k, n], B[k] * (1.0 - si[k, n]), rtol=1e-15, atol=0)
                check_close("ref_model", out.model_gl[k, n], m, rtol=1e-15, atol=0)
                dev["fields"] = max(dev["fields"], abs(out.residual_gl[k, n] - r))
            rms = math.sqrt(acc / n_v)
            check_close("ref_in_peak", out.in_peak_gl[k], ip, rtol=1e-12)
            check_close("ref_out_peak", out.out_peak_gl[k], op, rtol=1e-12)
            check_close("ref_share", out.in_share[k], share, rtol=1e-12)
            check_close("ref_rms", out.residual_rms_gl[k], rms, rtol=1e-12)
            dev["peaks"] = max(dev["peaks"], abs(out.in_peak_gl[k] - ip), abs(out.out_peak_gl[k] - op))
            if not math.isnan(share):
                dev["share"] = max(dev["share"], abs(out.in_share[k] - share))
            dev["rms"] = max(dev["rms"], abs(out.residual_rms_gl[k] - rms))
    # residual by hand: I - B (1 - si - so) against I - model; ordering of the operations is identical,
    # so 1e-12 gl is ample.
    assert dev["fields"] <= 1e-12, dev["fields"]
    for key, val in dev.items():
        record("reference: %s vs hand loop" % key, val)


def test_convergence():
    raise unittest.SkipTest("no discretisation in Block 37 (closed-form, pointwise formulas)")


def test_invariants():
    # (1) For any profiles, cleared - model_in = residual [1e-12 gl]: both sides are I - B + B S_in + B S_out
    # in a different order of operations; with values O(255) gl the rounding is a few ulps of 255 (~6e-14).
    rng = make_rng(SEED + 1)
    worst = 0.0
    for _ in range(20):
        n_planes, n_v = int(rng.integers(1, 10)), int(rng.integers(1, 80))
        inp = dict(profiles_gl=rng.uniform(0.0, 255.0, (n_planes, n_v)),
                   B_gl=rng.uniform(20.0, 250.0, n_planes),
                   s_in=rng.uniform(0.0, 1.0, (n_planes, n_v)), s_out=rng.uniform(0.0, 1.0, (n_planes, n_v)),
                   planes_z=rng.normal(0, 1, n_planes), d_um=1.3, z_ax_um=0.0, phi_rad=0.2)
        out = OC.clear_profiles(**inp)
        check_close("inv_cleared_minus_model_in", out.cleared_gl - out.model_in_gl, out.residual_gl,
                    rtol=0, atol=1e-12)
        worst = max(worst, maxdev(out.cleared_gl - out.model_in_gl, out.residual_gl))
        assert_array_equal(out.residual_gl, inp["profiles_gl"] - out.model_gl)
        # The split's two parts and the whole model: M = M_in - B S_out [1e-12 gl, same argument].
        check_close("inv_model_split", out.model_gl, out.model_in_gl - inp["B_gl"][:, None] * inp["s_out"],
                    rtol=0, atol=1e-12)
    record("invariant: |cleared - model_in - residual| (gl)", worst)

    # (2) layers_gl = B_k layers [exact: the same single product].
    inp = default_inputs(rng=make_rng(SEED + 2), B=np.linspace(60.0, 210.0, 7))
    layers = make_rng(SEED + 3).uniform(0.0, 0.3, (5,) + inp["s_in"].shape)
    out = OC.clear_profiles(layers=layers, **inp)
    assert out.layers_gl.shape == layers.shape
    assert_array_equal(out.layers_gl, inp["B_gl"][None, :, None] * layers)
    # An integer-free check that the layers are scaled per plane, not per layer or per sample.
    for m in range(layers.shape[0]):
        for k in range(layers.shape[1]):
            assert_array_equal(out.layers_gl[m, k], inp["B_gl"][k] * layers[m, k])

    # (3) A plane whose whole shadow is 0 has share NaN and peaks 0 (plane 0 of gaussian_split; also
    # a split that is 0 everywhere).
    out = OC.clear_profiles(**default_inputs())
    assert math.isnan(out.in_share[0]), out.in_share[0]
    assert out.in_peak_gl[0] == 0.0 and out.out_peak_gl[0] == 0.0
    assert np.all(np.isfinite(out.in_share[1:]))
    assert np.all((out.in_share[1:] >= 0.0) & (out.in_share[1:] <= 1.0))
    z = np.zeros((3, 11))
    out0 = OC.clear_profiles(np.full((3, 11), 92.0), np.full(3, 92.0), z, z, np.zeros(3), 1.0, 0.0)
    assert np.all(np.isnan(out0.in_share)) and np.all(out0.in_peak_gl == 0) and np.all(out0.out_peak_gl == 0)
    # Only s_out present on a plane: share 0; only s_in: share 1 [exact].
    si = np.zeros((2, 5))
    so = np.zeros((2, 5))
    so[0, 2] = 0.3
    si[1, 1] = 0.4
    out1 = OC.clear_profiles(np.full((2, 5), 92.0), np.full(2, 92.0), si, so, np.zeros(2), 1.0, 0.0)
    assert out1.in_share[0] == 0.0 and out1.in_share[1] == 1.0, out1.in_share


def test_contract():
    inp = default_inputs(rng=make_rng())
    n_planes, n_v = inp["profiles_gl"].shape
    for layers in (None, np.zeros((4, n_planes, n_v))):
        out = OC.clear_profiles(layers=layers, **inp)
        for name in ("cleared_gl", "model_in_gl", "model_gl", "residual_gl"):
            a = getattr(out, name)
            assert isinstance(a, np.ndarray) and a.shape == (n_planes, n_v) and a.dtype == np.float64, name
            assert np.all(np.isfinite(a)), name
        for name in ("chord_um", "in_peak_gl", "out_peak_gl", "in_share", "residual_rms_gl"):
            a = getattr(out, name)
            assert isinstance(a, np.ndarray) and a.shape == (n_planes,) and a.dtype == np.float64, name
        for name in ("chord_um", "in_peak_gl", "out_peak_gl", "residual_rms_gl"):
            assert np.all(np.isfinite(getattr(out, name))), name
        whole_max = (inp["s_in"] + inp["s_out"]).max(axis=1)
        assert_array_equal(np.isnan(out.in_share), ~(whole_max > 0))
        assert np.all(out.chord_um >= 0) and np.all(out.chord_um <= inp["d_um"])
        if layers is None:
            assert out.layers_gl is None
        else:
            assert out.layers_gl.shape == layers.shape and out.layers_gl.dtype == np.float64
    # The chord in clear_profiles is chord_um at z_k - z_ax [exact: the same call].
    assert_array_equal(out.chord_um, OC.chord_um(inp["d_um"], inp["planes_z"] - inp["z_ax_um"], inp["phi_rad"]))
    # frozen dataclass
    try:
        out.chord_um = None
    except Exception:
        pass
    else:
        raise AssertionError("OwnClearance is not frozen")
    # chord_um keeps dz's shape (scalar, 1-D, 2-D) and is float64
    for dz in (0.1, np.zeros(4), np.zeros((3, 2)), np.zeros((0,))):
        c = OC.chord_um(1.0, dz)
        assert isinstance(c, np.ndarray) and c.shape == np.shape(dz) and c.dtype == np.float64
    # Zero planes is allowed: empty outputs.
    e = OC.clear_profiles(np.zeros((0, 5)), np.zeros(0), np.zeros((0, 5)), np.zeros((0, 5)), np.zeros(0), 1.0, 0.0)
    assert e.cleared_gl.shape == (0, 5) and e.in_share.shape == (0,)


def test_determinism():
    inp = default_inputs(rng=make_rng(), B=np.linspace(60.0, 210.0, 7))
    layers = make_rng(SEED + 5).uniform(0.0, 0.3, (3,) + inp["s_in"].shape)
    copies = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in inp.items()}
    layers_copy = layers.copy()
    a = OC.clear_profiles(layers=layers, **inp)
    b = OC.clear_profiles(layers=layers, **inp)
    for f in ("cleared_gl", "model_in_gl", "model_gl", "residual_gl", "chord_um", "in_peak_gl",
              "out_peak_gl", "in_share", "residual_rms_gl", "layers_gl"):
        assert_array_equal(getattr(a, f), getattr(b, f))
    # inputs unchanged, and the outputs do not alias them
    for k, v in inp.items():
        if isinstance(v, np.ndarray):
            assert_array_equal(v, copies[k])
    assert_array_equal(layers, layers_copy)
    a.cleared_gl[...] = -1.0
    a.layers_gl[...] = -1.0
    assert_array_equal(inp["profiles_gl"], copies["profiles_gl"])
    assert_array_equal(layers, layers_copy)
    dz = np.linspace(-1, 1, 7)
    dz_copy = dz.copy()
    assert_array_equal(OC.chord_um(1.0, dz, 0.3), OC.chord_um(1.0, dz, 0.3))
    assert_array_equal(dz, dz_copy)


def _raises(func, **kw):
    try:
        func(**kw)
    except ValueError:
        return
    raise AssertionError("no ValueError for %r" % sorted(kw))


def test_edge_cases():
    base = default_inputs(rng=make_rng())
    n_planes, n_v = base["profiles_gl"].shape
    n_errors = 0

    def bad(**changes):
        nonlocal n_errors
        kw = dict(base)
        kw.update(changes)
        _raises(OC.clear_profiles, **kw)
        n_errors += 1

    # shapes that disagree
    bad(s_in=base["s_in"][:, :-1])
    bad(s_out=base["s_out"][:-1])
    bad(B_gl=base["B_gl"][:-1])
    bad(B_gl=float(B_REF))                               # the contract's B is (n_planes,)
    bad(planes_z=base["planes_z"][:-1])
    bad(profiles_gl=base["profiles_gl"][0])              # 1-D
    bad(profiles_gl=np.zeros((n_planes, 0)), s_in=np.zeros((n_planes, 0)), s_out=np.zeros((n_planes, 0)))
    bad(layers=np.zeros((2, n_planes, n_v + 1)))
    bad(layers=np.zeros((n_planes, n_v)))
    bad(d_um=np.array([1.0, 2.0]))
    # non-finite inputs
    for name in ("profiles_gl", "s_in", "s_out", "B_gl", "planes_z"):
        for val in (np.nan, np.inf):
            arr = np.array(base[name], dtype=float, copy=True)
            arr.flat[1] = val
            bad(**{name: arr})
    for name in ("d_um", "z_ax_um", "phi_rad"):
        bad(**{name: float("nan")})
        bad(**{name: float("inf")})
    lay = np.zeros((2, n_planes, n_v))
    lay[1, 2, 3] = np.nan
    bad(layers=lay)
    # B <= 0, d <= 0
    for b0 in (0.0, -5.0):
        B = base["B_gl"].copy()
        B[3] = b0
        bad(B_gl=B)
    bad(d_um=0.0)
    bad(d_um=-1.0)
    # a split entry below -1e-9; -1e-10 is a rounding and is accepted
    for name in ("s_in", "s_out"):
        arr = base[name].copy()
        arr[2, 4] = -2e-9
        bad(**{name: arr})
        arr[2, 4] = -1e-10
        kw = dict(base)
        kw[name] = arr
        OC.clear_profiles(**kw)
    lay = np.zeros((2, n_planes, n_v))
    lay[0, 0, 0] = -2e-9
    bad(layers=lay)
    # non-real dtype
    bad(profiles_gl=base["profiles_gl"].astype(complex))
    # chord_um
    for kw in (dict(d_um=0.0, dz_um=0.1), dict(d_um=-1.0, dz_um=0.1), dict(d_um=np.nan, dz_um=0.1),
               dict(d_um=1.0, dz_um=np.array([0.0, np.nan])), dict(d_um=1.0, dz_um=0.1, phi_rad=np.inf)):
        _raises(OC.chord_um, **kw)
        n_errors += 1
    MEASURED["edge: ValueError cases raised"] = n_errors


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"seed {SEED}")
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print(f"== {Path(__file__).name}")
    print(_environment())

    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))

    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline}".rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")

    if MEASURED:
        print("\n-- measured")
        for key, val in MEASURED.items():
            print(f"   {key:<48} {val:.3g}")

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
