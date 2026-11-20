"""Spec-derived tests of Block 10 (twin_noise.py; SPEC 2.7 (c), (d), (e) and the
Block 10 entry). Integration tester, 2026-11-20.

Every expected value is written out here from the spec: the seed rule of
2.7 (e) with numpy's SeedSequence, the law (L1')-(L2') of 2.6 as an explicit
Python recursion (not noise_law.simulate_traces), the tilt line of (TW1) /
(TW2) from its definition u(k, j) = (pos_k P + j) dt and u_bar = the mean over
the stored windows' samples.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import noise_law as nl
import twin_noise as tn

DT = 2e-5          # 50 kHz, the archive's rate
P = 10025          # the archive's onset period (samples)


def _units(entropy, n):
    """The unit draws of 2.6 (b) from default_rng(SeedSequence(entropy)), in
    the order fast, mid, slow, tilt, offset (each (1, n) or (1,))."""
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    fast = rng.standard_normal((1, n))
    mid = rng.standard_normal((1, n))
    slow = rng.standard_normal((1, n))
    return fast[0], mid[0], slow[0]


def _law_trace(law, dt, fast, mid, slow):
    """(L1') without tilt and offset, by the explicit stationary recursions of
    (L2') (a Python loop: independent of scipy.signal.lfilter)."""
    n = fast.size
    rm = np.exp(-dt / law.tau_m)
    rs = np.exp(-dt / law.tau_s)
    em = np.empty(n)
    es = np.empty(n)
    em[0] = law.sigma_m * mid[0]
    es[0] = law.sigma_s * slow[0]
    am = law.sigma_m * np.sqrt(1.0 - rm * rm)
    as_ = law.sigma_s * np.sqrt(1.0 - rs * rs)
    for j in range(1, n):
        em[j] = rm * em[j - 1] + am * mid[j]
        es[j] = rs * es[j - 1] + as_ * slow[j]
    return law.sigma_f * fast + em + es


LAW = nl.NoiseLaw(0.035, 0.017, 3e-4, 0.055, 0.040)


def test_tw1_reference_written_out():
    """(TW1) without r: row k = e(g, k) from SeedSequence([base, g, 0, s, pos_k])
    + b (u(k, j) - u_bar). Tolerance 1e-12 mV: roundoff of the recursion's
    float64 sums on mV-sized terms (lfilter and a loop differ in order)."""
    pos = np.array([1, 4, 5, 9, 30], dtype=np.int64)
    n_w, b, base, g, s = 400, 0.137, 11, 900000123, 92
    got = tn.ss_sweep_noise(LAW, b, DT, n_w, pos, P, base_seed=base, twin_id=g, sweep_number=s)
    assert got.shape == (pos.size, n_w) and got.dtype == np.float64
    j = np.arange(n_w)
    u = (pos[:, None] * P + j[None, :]) * DT
    u_bar = u.mean()
    # the spec's closed form of u_bar
    assert_allclose(u_bar, pos.mean() * P * DT + (n_w - 1) * DT / 2, rtol=1e-14)
    for k, pk in enumerate(pos):
        f, m, sl = _units([base, g, 0, s, int(pk)], n_w)
        exp = _law_trace(LAW, DT, f, m, sl) + b * (u[k] - u_bar)
        assert_allclose(got[k], exp, rtol=0, atol=1e-12)


def test_tw2_reference_written_out():
    """(TW2) without r: one continuous draw from SeedSequence([base, g, 1, s])
    + b (j - (n - 1) / 2) dt. 1e-12 mV as above."""
    n, b, base, g, s = 3000, -0.21, 5, 900000201, 61
    got = tn.ls_sweep_noise(LAW, b, DT, n, base_seed=base, twin_id=g, sweep_number=s)
    assert got.shape == (n,) and got.dtype == np.float64
    f, m, sl = _units([base, g, 1, s], n)
    exp = _law_trace(LAW, DT, f, m, sl) + b * (np.arange(n) - (n - 1) / 2) * DT
    assert_allclose(got, exp, rtol=0, atol=1e-12)


def test_tilt_line_sums_to_zero_and_window_independence():
    """The tilt term sums to zero over the K n_w stored samples (TW1); a
    window's random part (tilt 0) does not depend on the other positions
    passed (2.7 (e)). Sum bound 1e-9 mV: K n_w = 2000 terms of up to 1e-1 mV."""
    pos = np.array([0, 2, 7, 36], dtype=np.int64)
    kw = dict(base_seed=1, twin_id=2, sweep_number=3)
    a = tn.ss_sweep_noise(LAW, 0.5, DT, 500, pos, P, **kw)
    z = tn.ss_sweep_noise(LAW, 0.0, DT, 500, pos, P, **kw)
    assert abs((a - z).sum()) < 1e-9
    alone = tn.ss_sweep_noise(LAW, 0.0, DT, 500, pos[2:3], P, **kw)
    assert_array_equal(alone[0], z[2])


def test_ls_draw_continuous_and_stationary_mc():
    """TW2's draw is ONE stationary continuous draw: over 600 twin ids with a
    slow-only law (sigma_s 1 mV, tau_s 50 ms, dt 1 ms) the variance at the
    first and last sample equals sigma_s^2 and the covariance 50 samples apart
    equals sigma_s^2 exp(-1); bounds 4 SE of the sample moment (Gaussian:
    SE(var) = sqrt(2) s^2 / sqrt(N); SE(cov) <= sqrt((1 + r^2)/N) s^2)."""
    law = nl.NoiseLaw(0.0, 0.0, 3e-4, 1.0, 0.05)
    N, n = 600, 200
    X = np.array([tn.ls_sweep_noise(law, 0.0, 1e-3, n, base_seed=7, twin_id=g, sweep_number=1)
                  for g in range(N)])
    for col in (0, n - 1):
        assert abs(X[:, col].var() - 1.0) < 4 * np.sqrt(2.0 / N)
    c = np.mean(X[:, 10] * X[:, 60])
    r = np.exp(-1.0)
    assert abs(c - r) < 4 * np.sqrt((1 + r * r) / N)


def test_ss_windows_independent_mc():
    """D-180.2: the window means of two adjacent positions of one sweep are
    uncorrelated (slow-dominated law: a continuous train would give ~0.8).
    300 twin ids; bound 4 / sqrt(300) = 0.23 against 0.8."""
    law = nl.NoiseLaw(0.0, 0.0, 3e-4, 1.0, 1.0)
    N = 300
    m = np.array([tn.ss_sweep_noise(law, 0.0, 1e-3, 50, np.array([3, 4]), 200, base_seed=0, twin_id=g,
                                    sweep_number=5).mean(axis=1) for g in range(N)])
    r = np.corrcoef(m[:, 0], m[:, 1])[0, 1]
    assert abs(r) < 4 / np.sqrt(N)


@pytest.mark.parametrize("kwargs", [
    dict(position=np.array([], dtype=np.int64)),
    dict(position=np.array([1, 1])),
    dict(position=np.array([-1, 2])),
    dict(position=np.array([1.0, 2.0])),
    dict(n_w=1),
    dict(dt=0.0),
    dict(dt=float("nan")),
    dict(tilt=float("inf")),
    dict(period=0),
])
def test_ss_refusals(kwargs):
    a = dict(position=np.array([0, 1]), n_w=10, dt=DT, tilt=0.1, period=P)
    a.update(kwargs)
    with pytest.raises(ValueError):
        tn.ss_sweep_noise(LAW, a["tilt"], a["dt"], a["n_w"], a["position"], a["period"],
                          base_seed=0, twin_id=1, sweep_number=2)


def test_rng_refusals_and_seed_dependence():
    with pytest.raises(ValueError):
        tn.twin_rng(0, 1, "SS", 2)
    with pytest.raises(ValueError):
        tn.twin_rng(0, 1, "LS", 2, 3)
    with pytest.raises(ValueError):
        tn.twin_rng(0, 1, "ss", 2, 3)
    with pytest.raises(ValueError):
        tn.twin_rng(-1, 1, "LS", 2)
    a = tn.ls_sweep_noise(LAW, 0.0, DT, 50, base_seed=0, twin_id=1, sweep_number=2)
    for kw in (dict(base_seed=1, twin_id=1, sweep_number=2), dict(base_seed=0, twin_id=2, sweep_number=2),
               dict(base_seed=0, twin_id=1, sweep_number=3)):
        assert not np.array_equal(a, tn.ls_sweep_noise(LAW, 0.0, DT, 50, **kw))


def test_zero_law_gives_exact_tilt_only():
    law = nl.NoiseLaw(0.0, 0.0, 3e-4, 0.0, 0.04)
    z = tn.ss_sweep_noise(law, 0.0, DT, 20, np.array([0, 3]), P, base_seed=0, twin_id=1, sweep_number=2)
    assert np.all(z == 0.0)
    y = tn.ls_sweep_noise(law, 0.0, DT, 20, base_seed=0, twin_id=1, sweep_number=2)
    assert np.all(y == 0.0)
