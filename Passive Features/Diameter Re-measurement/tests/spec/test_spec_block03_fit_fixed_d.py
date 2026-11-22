"""Spec-derived tests of Block 3, amendment 2 (fit.fit_profile(..., d_fixed_um); D-173; tester, 2026-11-22).

Written from specs/SPEC.md (Block 3 amendment 2, section 2.2's Eq. (3)) before the code was read. The profiles
are made by an independent numerical convolution of Eq. (3)'s transmittance with the Gaussian (a fine grid
and numpy; not Block 3's quadrature); the base commit's fit_profile (ccf69db, extracted by `git archive`
into a temporary folder by the test) is the reference for "without it the function is unchanged [exact]".

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import importlib
import io
import math
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
REPO = WS.parent.parent
BASE = "ccf69db327de49b764fcc60f31e9ced31d671a0e"
for _p in (WS / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter import config as C  # noqa: E402
from allen_diameter.analysis import fit as F  # noqa: E402

V = np.round(np.arange(-26, 27) * 0.1144, 10)


def ref_profile(d, mu, v0, phi, sigma, B, v=V):
    """Eq. (3): B (T * g_sigma)(v), T(u) = exp(-(mu d / cos phi) sqrt(1 - (2 (u - v0) / d)^2)) inside |u - v0| <=
    d/2, by a fine-grid sum (step 2e-4 um; 1 - T vanishes outside the tube, so the sum runs over it only)."""
    h = 2e-4
    u = v0 + np.arange(-0.5 * d + 0.5 * h, 0.5 * d, h)
    alpha = mu * d / math.cos(phi)
    one_minus_T = 1.0 - np.exp(-alpha * np.sqrt(np.maximum(1.0 - (2.0 * (u - v0) / d) ** 2, 0.0)))
    g = np.exp(-0.5 * ((v[:, None] - u[None, :]) / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)
    return B * (1.0 - h * (g @ one_minus_T))


def mcfg():
    return C.default_config().measure


@pytest.mark.parametrize("d,mu,v0,phi", [(1.0, 0.6, 0.05, 0.0), (0.5, 3.0, -0.08, math.radians(30)),
                                         (2.0, 0.3, 0.0, math.radians(15))])
def test_fixed_d_recovers_mu_and_v0(d, mu, v0, phi):
    m = mcfg()
    I = ref_profile(d, mu, v0, phi, m.sigma_fit_um, 210.0)
    r = F.fit_profile(V, I, phi, 210.0, m, d_fixed_um=d)
    # the reference sum and Block 3's Gauss-Legendre model differ by ~1e-7 of B (the fine grid's midpoint
    # rule on a square-root edge); that moves (mu, v0) by ~1e-6 relative: 2e-5 relative and 2e-6 um allow it
    assert r.d_hat_um == d
    assert_allclose(r.mu_hat_per_um, mu, rtol=2e-5)
    assert abs(r.v0_hat_um - v0) <= 2e-6
    assert r.status == "converged" and r.at_bound == ()
    assert_allclose(r.alpha_hat, r.mu_hat_per_um * d / math.cos(phi), rtol=1e-15)


def test_fixed_d_at_six_tenths_gives_a_darker_tube_and_names_mu_v0_only():
    m = mcfg()
    I = ref_profile(1.0, 0.6, 0.0, 0.0, m.sigma_fit_um, 210.0)
    r = F.fit_profile(V, I, 0.0, 210.0, m, d_fixed_um=0.6)
    assert r.d_hat_um == 0.6 and r.mu_hat_per_um > 0.6
    # a flat profile: mu at its lower bound -- only mu and v0 names may appear
    flat = F.fit_profile(V, np.full(V.size, 210.0), 0.0, 210.0, m, d_fixed_um=0.6)
    assert flat.at_bound and set(flat.at_bound) <= {"mu_lo", "mu_hi", "v0_lo", "v0_hi"}
    assert flat.d_hat_um == 0.6


@pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf")])
def test_fixed_d_refusals(bad):
    m = mcfg()
    I = ref_profile(1.0, 0.6, 0.0, 0.0, m.sigma_fit_um, 210.0)
    with pytest.raises(ValueError):
        F.fit_profile(V, I, 0.0, 210.0, m, d_fixed_um=bad)


def _base_fit_module(tmp):
    """Block 3 of the base commit, from `git archive` (no worktree, nothing written in the repository)."""
    blob = subprocess.run(["git", "-C", str(REPO), "archive", BASE, "Passive Features/Diameter Re-measurement/src"],
                          check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(tmp, filter="data")
    src = Path(tmp) / "Passive Features" / "Diameter Re-measurement" / "src"
    code = ("import sys, json, numpy as np; sys.path.insert(0, %r)\n"
            "from allen_diameter import config as C; from allen_diameter.analysis import fit as F\n"
            "m = C.default_config().measure\n"
            "data = np.load(sys.argv[1])\n"
            "out = []\n"
            "for k in range(data['I'].shape[0]):\n"
            "    r = F.fit_profile(data['v'], data['I'][k], float(data['phi'][k]), 210.0, m)\n"
            "    out.append([r.d_hat_um, r.mu_hat_per_um, r.v0_hat_um, r.rss_gl2, r.nfev])\n"
            "np.save(sys.argv[2], np.array(out))\n") % str(src)
    return code


def test_without_d_fixed_bit_identical_to_the_base_commit():
    """Block 3 amendment 2: "Without it the function is unchanged [exact]". The base commit's fit is run in a
    subprocess on the same profiles; the base's MeasureConfig had no pilot_diameter, the fit reads none."""
    m = mcfg()
    rng = np.random.default_rng(4)
    cases = [(0.5, 0.6, 0.02, 0.0), (1.0, 3.0, -0.05, 0.3), (2.0, 0.3, 0.1, 0.6), (0.3, 1.0, 0.0, 0.1)]
    I = np.array([ref_profile(d, mu, v0, phi, 0.1, 210.0) + rng.normal(0, 1.0, V.size) for d, mu, v0, phi in cases])
    phi = np.array([c[3] for c in cases])
    with tempfile.TemporaryDirectory() as tmp:
        code = _base_fit_module(tmp)
        np.savez(Path(tmp) / "in.npz", I=I, v=V, phi=phi)
        subprocess.run([sys.executable, "-c", code, str(Path(tmp) / "in.npz"), str(Path(tmp) / "out.npy")],
                       check=True, capture_output=True)
        base = np.load(Path(tmp) / "out.npy")
    now = np.array([[r.d_hat_um, r.mu_hat_per_um, r.v0_hat_um, r.rss_gl2, r.nfev]
                    for r in (F.fit_profile(V, I[k], float(phi[k]), 210.0, m) for k in range(len(cases)))])
    assert np.array_equal(now, base), (now, base)
