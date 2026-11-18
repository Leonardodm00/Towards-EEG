"""Spec-derived tests of Block 37 (the same-node clearance: the cleared profiles and the chord per plane;
D-151.1, D-156.2) -- integration tester, 2026-11-17. Oracles from specs/SPEC.md Block 37; expected values
written by hand (numpy) or by an independent geometric construction, never from own_clearance itself.

Run: python3 -m pytest -q tests/spec/test_spec_block37_own_clearance.py
Pure ASCII.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import nc_helpers  # noqa: F401  (sys.path)
from allen_diameter.analysis import own_clearance as OC

RNG_SEED = 20261117


def _arrays(n_planes=5, n_v=31, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    v = np.linspace(-3, 3, n_v)
    s_in = 0.3 * np.exp(-0.5 * (v / 0.4) ** 2)[None, :] * rng.uniform(0.2, 1.0, (n_planes, 1))
    s_out = 0.2 * np.exp(-0.5 * (v / 1.3) ** 2)[None, :] * rng.uniform(0.2, 1.0, (n_planes, 1))
    B = rng.uniform(80, 110, n_planes)
    prof = B[:, None] * (1 - s_in - s_out) + rng.normal(0, 1.0, (n_planes, n_v))
    z = 0.28 * np.arange(-(n_planes // 2), n_planes - n_planes // 2)
    return v, prof, B, s_in, s_out, z


def test_formulas_by_hand():
    """cleared = I + B S_out; model_in = B (1 - S_in); model = B (1 - S_in - S_out); residual = I - model;
    peaks B max_n S; share S_in/(S_in + S_out) at the first argmax of the whole; rms over n. Tolerance 1e-12
    gl (float64 rounding of sums of order 100 gl)."""
    v, I, B, si, so, z = _arrays()
    c = OC.clear_profiles(I, B, si, so, z, 0.8, 0.05, 0.3)
    Bc = B[:, None]
    assert_allclose(c.cleared_gl, I + Bc * so, rtol=0, atol=1e-12)
    assert_allclose(c.model_in_gl, Bc * (1 - si), rtol=0, atol=1e-12)
    assert_allclose(c.model_gl, Bc * (1 - si - so), rtol=0, atol=1e-12)
    assert_allclose(c.residual_gl, I - Bc * (1 - si - so), rtol=0, atol=1e-12)
    assert_allclose(c.cleared_gl - c.model_in_gl, c.residual_gl, rtol=0, atol=1e-12)
    assert_allclose(c.in_peak_gl, B * si.max(1), rtol=0, atol=1e-12)
    assert_allclose(c.out_peak_gl, B * so.max(1), rtol=0, atol=1e-12)
    w = si + so
    share = []
    for k in range(w.shape[0]):
        n = int(np.argmax(w[k]))
        share.append(si[k, n] / w[k, n])
    assert_allclose(c.in_share, share, rtol=1e-14)
    assert_allclose(c.residual_rms_gl, np.sqrt(np.mean((I - Bc * (1 - si - so)) ** 2, axis=1)), rtol=1e-13)
    for name in ("cleared_gl", "model_in_gl", "model_gl", "residual_gl"):
        a = getattr(c, name)
        assert a.shape == I.shape and a.dtype == np.float64
    for name in ("chord_um", "in_peak_gl", "out_peak_gl", "in_share", "residual_rms_gl"):
        assert getattr(c, name).shape == (I.shape[0],)
    assert c.layers_gl is None


def test_noise_free_synthesis_gives_zero_residual():
    """Profiles synthesised as B (1 - S_in - S_out): residual 0 and cleared = in-band model [1e-12 gl at B = 92]."""
    v, _I, _B, si, so, z = _arrays()
    B = np.full(si.shape[0], 92.0)
    I = B[:, None] * (1 - si - so)
    c = OC.clear_profiles(I, B, si, so, z, 1.0, 0.0)
    assert_allclose(c.residual_gl, 0.0, atol=1e-12)
    assert_allclose(c.cleared_gl, c.model_in_gl, rtol=0, atol=1e-12)
    assert_allclose(c.residual_rms_gl, 0.0, atol=1e-12)


def test_zero_out_and_zero_shadow_planes():
    """S_out = 0 returns the profiles as float64 [exact]; a plane whose whole shadow is 0 has share NaN and
    peaks 0; integer profiles accepted and returned as float64."""
    v, I, B, si, so, z = _arrays()
    c = OC.clear_profiles(I, B, si, np.zeros_like(so), z, 1.0, 0.0)
    assert_array_equal(c.cleared_gl, I)
    si2, so2 = si.copy(), so.copy()
    si2[1] = 0.0
    so2[1] = 0.0
    c2 = OC.clear_profiles(I, B, si2, so2, z, 1.0, 0.0)
    assert math.isnan(c2.in_share[1]) and c2.in_peak_gl[1] == 0.0 and c2.out_peak_gl[1] == 0.0
    assert np.all(np.isfinite(np.delete(c2.in_share, 1)))
    Ii = np.round(I).astype(np.int64)
    c3 = OC.clear_profiles(Ii, B, si, so, z, 1.0, 0.0)
    assert c3.cleared_gl.dtype == np.float64


def chord_ref(d, dz, phi, step=1e-4):
    """Independent chord: the extent of offsets v on a grid of `step` um for which the point (0, v, dz) lies
    within d/2 of the line through the origin with direction (cos phi, 0, sin phi) (distance by the cross
    product), the spec's own construction. Accuracy +- 2 steps."""
    a = np.array([math.cos(phi), 0.0, math.sin(phi)])
    vs = np.arange(-d, d + step / 2, step)
    P = np.stack([np.zeros_like(vs), vs, np.full_like(vs, dz)], axis=1)
    dist = np.linalg.norm(np.cross(P, a), axis=1)
    inside = vs[dist <= d / 2]
    return 0.0 if inside.size == 0 else float(inside.max() - inside.min() + step)


@pytest.mark.parametrize("d", [0.3, 1.1, 3.0])
@pytest.mark.parametrize("phi_deg", [0.0, 45.0, 75.0])
def test_chord_against_geometry(d, phi_deg):
    """c_k = the part of the line at depth z_k inside the tube: against the cross-product grid [3e-4 um,
    the grid's +-1.5 steps]; d at z_k = z_ax [1e-12]; 0 at and beyond |dz| cos phi = d/2 [exact]."""
    phi = math.radians(phi_deg)
    dzs = np.linspace(-1.2, 1.2, 13) * d / 2 / max(math.cos(phi), 0.2)
    got = OC.chord_um(d, dzs, phi)
    assert got.shape == dzs.shape
    ref = np.array([chord_ref(d, float(x), phi) for x in dzs])
    assert_allclose(got, ref, rtol=0, atol=3e-4)
    assert abs(float(OC.chord_um(d, 0.0, phi)) - d) < 1e-12
    edge = d / 2 / math.cos(phi)
    assert float(OC.chord_um(d, edge * 1.0000001, phi)) == 0.0
    assert float(OC.chord_um(d, -3 * edge, phi)) == 0.0


def test_chord_from_clear_profiles_uses_planes_and_axis():
    """clear_profiles' chord is chord(d, planes_z - z_ax, phi) [1e-12]."""
    v, I, B, si, so, z = _arrays(n_planes=9)
    c = OC.clear_profiles(I, B, si, so, z, 0.9, 0.17, 0.4)
    d, zax, phi = 0.9, 0.17, 0.4
    exp = 2 * np.sqrt(np.maximum(d * d / 4 - ((z - zax) * math.cos(phi)) ** 2, 0))
    assert_allclose(c.chord_um, exp, rtol=0, atol=1e-12)


def test_layers_scaled_and_inputs_unchanged_and_deterministic():
    """layers_gl = B_k layers [exact up to one multiplication]; inputs not changed; determinism [exact]."""
    v, I, B, si, so, z = _arrays()
    L = np.abs(np.random.default_rng(1).normal(size=(6,) + I.shape)) * 0.01
    copies = [x.copy() for x in (I, B, si, so, z, L)]
    c = OC.clear_profiles(I, B, si, so, z, 1.0, 0.0, 0.2, layers=L)
    assert_array_equal(c.layers_gl, B[None, :, None] * L)
    for a, b in zip((I, B, si, so, z, L), copies):
        assert_array_equal(a, b)
    c2 = OC.clear_profiles(I, B, si, so, z, 1.0, 0.0, 0.2, layers=L)
    for name in ("cleared_gl", "residual_gl", "chord_um", "in_share", "layers_gl"):
        assert_array_equal(getattr(c, name), getattr(c2, name))


def test_errors():
    """ValueError: shapes that disagree, non-finite input, B <= 0, d <= 0, a split entry below -1e-9 (and
    one at -5e-10 accepted)."""
    v, I, B, si, so, z = _arrays()
    bad_cases = [
        dict(profiles_gl=I[:, :-1]), dict(B_gl=B[:-1]), dict(s_in=si[:-1]), dict(s_out=so[:, 1:]),
        dict(planes_z=z[:-1]), dict(profiles_gl=np.where(np.arange(I.size).reshape(I.shape) == 3, np.nan, I)),
        dict(B_gl=np.where(np.arange(B.size) == 0, 0.0, B)), dict(B_gl=-B), dict(d_um=0.0), dict(d_um=-1.0),
        dict(s_in=si - 2e-9), dict(s_out=np.where(np.arange(so.size).reshape(so.shape) == 4, -2e-9, so)), dict(z_ax_um=float("inf")),
        dict(layers=np.zeros((2, 3, 3))), dict(layers=np.full((2,) + I.shape, -1e-8)),
    ]
    base = dict(profiles_gl=I, B_gl=B, s_in=si, s_out=so, planes_z=z, d_um=1.0, z_ax_um=0.0, phi_rad=0.1)
    for case in bad_cases:
        kw = dict(base, **case)
        with pytest.raises(ValueError):
            OC.clear_profiles(**kw)
    ok = OC.clear_profiles(**dict(base, s_in=si - 5e-10))
    assert np.all(np.isfinite(ok.cleared_gl))
    with pytest.raises(ValueError):
        OC.chord_um(0.0, 0.1)
    with pytest.raises(ValueError):
        OC.chord_um(1.0, np.array([0.0, np.nan]))
