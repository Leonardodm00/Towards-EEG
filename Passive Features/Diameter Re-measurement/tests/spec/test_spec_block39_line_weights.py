"""Spec-derived tests of Block 39 (residual weights about the centre and the near-bump cut; tester, 2026-11-22).

Written from specs/SPEC.md (Block 39 in the amendment of 2026-11-18, D-169, D-171, D-176) before the code was
read. Expected values: arithmetic by hand, numpy/scipy, and ref_bump_cut below -- the rule's text written
independently with scipy.signal.find_peaks; nothing here calls the code under test to make an expected value.

Pure ASCII.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter.analysis import line_weights as LW  # noqa: E402

V = np.round(np.arange(-26, 27) * 0.1144, 10)


def ref_bump_cut(D, sigma, v, centre, window, m, half):
    """Block 39's bump_cut from its text: the central extreme = the largest D within window of the centre; no
    cut when D_c < m sigma; per side, outwards, the central dip ends at the first sample <= half D_c (no cut if
    never); from there outwards a bump is a local maximum of prominence >= m sigma (find_peaks on that side's
    samples from the dip's end outwards); the cut is at the lightest sample between the dip's end and the
    nearest bump's extreme; -inf / +inf where a side has no cut."""
    near = np.flatnonzero(np.abs(v - centre) <= window)
    n_c = int(near[np.argmax(D[near])])
    Dc = D[n_c]
    if Dc < m * sigma:
        return -math.inf, math.inf
    out = []
    for side in (-1, +1):
        idx = np.arange(n_c, -1, -1) if side < 0 else np.arange(n_c, D.size)
        d = D[idx]
        ends = np.flatnonzero(d <= half * Dc)
        if ends.size == 0:
            out.append(None)
            continue
        e = int(ends[0])
        seg = d[e:]
        pk, _ = find_peaks(seg, prominence=m * sigma)
        if pk.size == 0:
            out.append(None)
            continue
        b = int(pk[0])
        out.append(float(v[idx[e + int(np.argmin(seg[:b + 1]))]]))
    lo, hi = out
    return (-math.inf if lo is None else lo), (math.inf if hi is None else hi)


def test_allen_fwhm_by_hand():
    # sqrt((sqrt(3)/2 * 0.45)^2 + (2 sqrt(2 ln 2) * 0.080)^2) = 0.43286, and 1.05617 at 1.2 um (by hand)
    assert_allclose(LW.allen_fwhm_um(0.45, 0.080), 0.43286, atol=1e-4)
    assert_allclose(LW.allen_fwhm_um(1.2, 0.080), 1.05617, atol=1e-4)
    for d, s in ((0.3, 0.0), (2.0, 0.25)):
        want = math.sqrt((math.sqrt(3) / 2 * d) ** 2 + (2 * math.sqrt(2 * math.log(2)) * s) ** 2)
        assert_allclose(LW.allen_fwhm_um(d, s), want, rtol=1e-15)
    # the chord factor: the half-depth width of 2 sqrt(r^2 - v^2) is sqrt(3) r
    r = 0.7
    vv = np.linspace(-r, r, 2000001)
    chord = 2 * np.sqrt(np.maximum(r * r - vv * vv, 0))
    w = vv[chord >= r].max() - vv[chord >= r].min()
    assert_allclose(LW.allen_fwhm_um(2 * r, 0.0), w, atol=2e-6)
    for bad in ((0.0, 0.08), (-1.0, 0.08), (np.nan, 0.08), (0.5, -0.1), (0.5, np.inf)):
        with pytest.raises(ValueError):
            LW.allen_fwhm_um(*bad)


def test_centre_weights():
    w = LW.centre_weights(V, 0.2, 0.7)
    assert_allclose(w, np.exp(-np.abs(V - 0.2) / 0.7), rtol=1e-15)
    w = LW.centre_weights(np.array([-1.0, 0.0, 0.7]), 0.0, 0.7)
    assert w[1] == 1.0 and abs(w[2] - math.exp(-1.0)) <= 1e-15
    for bad in ((V, 0.0, 0.0), (V, np.nan, 0.5), (V[::-1], 0.0, 0.5)):
        with pytest.raises(ValueError):
            LW.centre_weights(*bad)


def test_bump_cut_known_answers():
    D = 60 * np.exp(-0.5 * (V / 0.35) ** 2) + 30 * np.exp(-0.5 * ((V - 1.3) / 0.25) ** 2)
    lo, hi = LW.bump_cut(D, 2.5, V, 0.0, 0.5, 2.0, 0.5)
    right = (V >= 0) & (V <= 1.3)
    assert hi == V[right][np.argmin(D[right])] and lo == -math.inf       # by hand: the lightest sample between
    lo, hi = LW.bump_cut(60 * np.exp(-0.5 * (V / 0.35) ** 2), 2.5, V, 0.0, 0.5, 2.0, 0.5)
    assert (lo, hi) == (-math.inf, math.inf)                              # a single dip: no cut
    # a second dip of prominence below m sigma: no cut; lower m: a cut
    D = 60 * np.exp(-0.5 * (V / 0.35) ** 2) + 4 * np.exp(-0.5 * ((V + 1.5) / 0.25) ** 2)
    assert LW.bump_cut(D, 2.5, V, 0.0, 0.5, 2.0, 0.5) == (-math.inf, math.inf)
    lo, _hi = LW.bump_cut(D, 2.5, V, 0.0, 0.5, 1.0, 0.5)
    assert math.isfinite(lo)
    # flat-bottomed dip with texture of 3 SDs inside its half depth: no cut (the half-depth condition)
    flat = np.where(np.abs(V) < 1.2, 60.0, 0.0) + 3 * 2.5 * np.sin(V * 9.0) * (np.abs(V) < 1.0)
    flat = gaussian_filter1d(flat, 1.0, mode="nearest")
    assert LW.bump_cut(flat, 2.5, V, 0.0, 0.5, 2.0, 0.5) == (-math.inf, math.inf)
    # no readable central dip (D_c < m sigma): no cut whatever beside it
    D = 4 * np.exp(-0.5 * (V / 0.35) ** 2) + 30 * np.exp(-0.5 * ((V - 1.3) / 0.25) ** 2)
    assert LW.bump_cut(D, 2.5, V, 0.0, 0.5, 2.0, 0.5) == (-math.inf, math.inf)


def test_bump_cut_against_reference_on_random_profiles():
    rng = np.random.default_rng(11)
    n_cut = 0
    for _ in range(500):
        D = rng.uniform(5, 80) * np.exp(-0.5 * ((V - rng.uniform(-0.4, 0.4)) / rng.uniform(0.15, 0.9)) ** 2)
        for _k in range(rng.integers(0, 3)):
            D += rng.uniform(0, 40) * np.exp(-0.5 * ((V - rng.uniform(-3, 3)) / rng.uniform(0.1, 0.6)) ** 2)
        D = gaussian_filter1d(D + rng.normal(0, 2.0, V.size), 1.0, mode="nearest")
        args = (D, 2.0, V, rng.uniform(-0.2, 0.2), rng.uniform(0.3, 1.2), rng.uniform(1.0, 3.0),
                rng.uniform(0.3, 0.7))
        got, want = LW.bump_cut(*args), ref_bump_cut(*args)
        assert got == want, (got, want)
        n_cut += math.isfinite(got[0]) + math.isfinite(got[1])
    print("cuts found on 500 random profiles:", n_cut)
    assert n_cut > 50                      # the comparison covered the cut branch, not only "no cut"


def test_mean_dip_and_mask():
    rng = np.random.default_rng(2)
    P = 200.0 - 40 * np.exp(-0.5 * (V / 0.4) ** 2)[None, :] + rng.normal(0, 1, (5, V.size))
    B = np.array([200.0, 201, 199, 200, 202])
    use = np.array([True, False, True, True, False])
    want = gaussian_filter1d((B[use][:, None] - P[use]).mean(axis=0), 1.0, mode="nearest")
    assert_allclose(LW.mean_dip(P, B, 1.0, planes=use), want, rtol=1e-14)
    assert_allclose(LW.mean_dip(P, B, 0.0), (B[:, None] - P).mean(axis=0), rtol=1e-14)
    P2 = P.copy()
    P2[:, V > 1.0] -= 25 * np.exp(-0.5 * ((V[V > 1.0] - 1.6) / 0.25) ** 2)[None, :]
    keep, lo, hi = LW.bump_mask(P2, B, 1.0, V, 0.0, 0.5, 2.0, 0.5, 1.0)
    assert_allclose(keep, (V >= lo) & (V <= hi))
    assert math.isfinite(hi) and lo == -math.inf and keep.sum() < V.size
    keep0, lo0, hi0 = LW.bump_mask(P2, B, 1.0, V, 0.0, 0.5, 0.0, 0.5, 1.0)
    assert keep0.all() and (lo0, hi0) == (-math.inf, math.inf)            # m_sd = 0 disables
    P0 = P.copy()
    LW.bump_mask(P, B, 1.0, V, 0.0, 0.5, 2.0, 0.5, 1.0)
    assert np.array_equal(P, P0)
    for bad in (dict(planes=np.zeros(5, bool)), dict(planes=np.ones(4, bool)), dict(m_sd=-1.0)):
        kw = dict(planes=None, m_sd=2.0)
        kw.update(bad)
        with pytest.raises(ValueError):
            LW.bump_mask(P, B, 1.0, V, 0.0, 0.5, kw["m_sd"], 0.5, 1.0, planes=kw["planes"])
