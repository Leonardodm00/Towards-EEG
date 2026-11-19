# -*- coding: utf-8 -*-
"""
twin_noise.py -- the twin's noise: independent SS windows, one continuous LS draw (Block 10)
==========================================================================================

specs/SPEC.md, section 2.7 (c), (d), (e) and the entry of Block 10 (D-180,
D-182, D-183.3, D-186). Model layer: no I/O, no NEURON, no module-level RNG.

What it computes (the response r is added by Block 12, not here):

  SS window k of sweep s of twin g, (TW1) without r:

      e(g, k)[j] + b_hat(c, s) ( u(k, j) - u_bar(c, s) ),   j = 0 .. n_w - 1,

  e(g, k) one stationary draw of n_w samples of the sweep's law (2.6 (a)),
  from its OWN generator (2.7 (e)), so windows are independent (D-180.2);
  u(k, j) = (pos_k P + j) dt; u_bar the mean of u over the positions passed
  and all j. The tilt term is the composition of Block 2's centred in-window
  ramp b (j - (n_w - 1) / 2) dt (`noise_law.simulate_traces(...,
  tilt_mV_per_s=...)`) and (OF1)'s offset b (pos_k P dt - mean) (`noise_windows.
  tilt_offsets`); their sum is b (u(k, j) - u_bar) exactly in algebra.

  LS sweep s of twin g, (TW2) without r:

      e(g, s)[j] + b_hat(c, s) ( j - (n - 1) / 2 ) dt,   j = 0 .. n - 1,

  ONE continuous stationary draw over the whole sweep (D-182), through
  `noise_law.simulate_sweep(..., tilt_mV_per_s=...)`.

  Seeds (2.7 (e), D-186.6): SS window: default_rng(SeedSequence([base_seed,
  twin_id, 0, sweep_number, position])); LS sweep: SeedSequence([base_seed,
  twin_id, 1, sweep_number]); the unit draws by noise_law.draw_units(rng, 1, n)
  (fast, mid, slow, tilt, offset; the last two unused). No offset term and no
  random tilt (D-183.3).

Units: mV, s, mV/s; positions and periods in samples. Pure ASCII, LF.
"""
from __future__ import annotations

import operator
from typing import Optional

import numpy as np

from noise_law import NoiseLaw, draw_units, simulate_sweep, simulate_traces
from noise_windows import tilt_offsets
from twin_config import PROTOCOL_CODE


def _nonneg_int(name: str, value) -> int:
    """`value` as a Python int; ValueError when it is not an integer (bool
    included) or is negative."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("%s must be a non-negative integer, got %r" % (name, value))
    try:
        v = operator.index(value)
    except TypeError:
        raise ValueError("%s must be a non-negative integer, got %r" % (name, value)) from None
    if v < 0:
        raise ValueError("%s must be a non-negative integer, got %r" % (name, value))
    return int(v)


def _check_common(tilt_mV_per_s: float, dt: float, n: int, n_name: str) -> int:
    """The refusals shared by the two sweep functions; returns n as an int."""
    try:
        dt_f = float(dt)
        b_f = float(tilt_mV_per_s)
    except (TypeError, ValueError):
        raise ValueError("dt and tilt_mV_per_s must be real numbers, got %r, %r" % (dt, tilt_mV_per_s)) from None
    if not (np.isfinite(dt_f) and dt_f > 0.0):
        raise ValueError("dt must be finite and > 0 (s), got %r" % (dt,))
    if not np.isfinite(b_f):
        raise ValueError("tilt_mV_per_s must be finite (mV/s), got %r" % (tilt_mV_per_s,))
    n_i = _nonneg_int(n_name, n)
    if n_i < 2:
        raise ValueError("%s must be >= 2, got %r" % (n_name, n))
    return n_i


def twin_rng(base_seed: int, twin_id: int, protocol: str, sweep_number: int,
             position: Optional[int] = None) -> np.random.Generator:
    """The generator of one twin SS window or LS sweep, SPEC 2.7 (e):
    numpy.random.default_rng(SeedSequence([base_seed, twin_id, 0, sweep_number,
    position])) for protocol 'SS' (position required) and
    SeedSequence([base_seed, twin_id, 1, sweep_number]) for 'LS' (position
    forbidden). ValueError: an unknown protocol, 'SS' without a position,
    'LS' with one, a negative integer."""
    if not isinstance(protocol, str) or protocol not in PROTOCOL_CODE:
        raise ValueError("protocol must be 'SS' or 'LS', got %r" % (protocol,))
    entropy = [_nonneg_int("base_seed", base_seed), _nonneg_int("twin_id", twin_id),
               PROTOCOL_CODE[protocol], _nonneg_int("sweep_number", sweep_number)]
    if protocol == "SS":
        if position is None:
            raise ValueError("twin_rng: protocol 'SS' needs the window's position")
        entropy.append(_nonneg_int("position", position))
    elif position is not None:
        raise ValueError("twin_rng: protocol 'LS' takes no position, got %r" % (position,))
    return np.random.default_rng(np.random.SeedSequence(entropy))


def ss_sweep_noise(law: NoiseLaw, tilt_mV_per_s: float, dt: float, n_w: int, position, period_samples: int, *,
                   base_seed: int, twin_id: int, sweep_number: int) -> np.ndarray:
    """The noise and tilt of ONE SS sweep's stored windows, (TW1) without the
    response: float64 (K, n_w) mV, row k for the window at position[k].

    law            : the sweep's law theta_tilde(c, s) (mV, mV, s, mV, s)
    tilt_mV_per_s  : b_hat(c, s), mV/s, finite
    dt             : s, the sampling interval (> 0)
    n_w            : samples per stored window (>= 2)
    position       : int (K,), the 0-based places of ALL the stored windows of
                     the sweep in its train (unique, >= 0); u_bar(c, s) is
                     taken over the positions passed
    period_samples : onset to onset, samples (>= 1)
    Row k = one stationary draw of `law` from twin_rng(base_seed, twin_id,
    'SS', sweep_number, position[k]) through noise_law.draw_units(rng, 1, n_w)
    + tilt_mV_per_s * (u(k, j) - u_bar), u(k, j) = (position[k] *
    period_samples + j) * dt. No offset. ValueError: SPEC, Block 10."""
    n_w = _check_common(tilt_mV_per_s, dt, n_w, "n_w")
    P = _nonneg_int("period_samples", period_samples)
    if P < 1:
        raise ValueError("period_samples must be >= 1, got %r" % (period_samples,))
    pos = np.asarray(position)
    if pos.ndim != 1 or pos.size == 0:
        raise ValueError("position must be a non-empty 1-D array of integers, got shape %r" % (pos.shape,))
    if not np.issubdtype(pos.dtype, np.integer):
        raise ValueError("position must hold integers, got dtype %s" % (pos.dtype,))
    if np.any(pos < 0):
        raise ValueError("position must be >= 0, got %r" % (pos.tolist(),))
    if np.unique(pos).size != pos.size:
        raise ValueError("position must be unique, got %r" % (pos.tolist(),))
    base_seed = _nonneg_int("base_seed", base_seed)
    twin_id = _nonneg_int("twin_id", twin_id)
    sweep_number = _nonneg_int("sweep_number", sweep_number)
    b = float(tilt_mV_per_s)
    dt = float(dt)
    K = pos.size

    # One generator per window (2.7 (e)): independent windows (D-180.2).
    fast = np.empty((K, n_w))
    mid = np.empty((K, n_w))
    slow = np.empty((K, n_w))
    for k in range(K):
        units = draw_units(twin_rng(base_seed, twin_id, "SS", sweep_number, int(pos[k])), 1, n_w)
        fast[k], mid[k], slow[k] = units.fast[0], units.mid[0], units.slow[0]

    # e(g, k) + the in-window ramp b (j - (n_w - 1) / 2) dt (Block 2) ...
    out = simulate_traces(law, dt, fast, mid, slow, tilt_mV_per_s=np.full(K, b))
    # ... + (OF1)'s offset b (pos_k P dt - mean over the positions passed).
    off = tilt_offsets(pos.astype(np.int64), np.full(K, sweep_number, dtype=np.int64), P, dt, {sweep_number: b})
    return np.ascontiguousarray(out + off[:, None], dtype=np.float64)


def ls_sweep_noise(law: NoiseLaw, tilt_mV_per_s: float, dt: float, n_samples: int, *,
                   base_seed: int, twin_id: int, sweep_number: int) -> np.ndarray:
    """The noise and tilt of one Long Square twin sweep, (TW2) without the
    response: float64 (n_samples,) mV. One continuous stationary draw of
    `law` over the whole sweep from twin_rng(base_seed, twin_id, 'LS',
    sweep_number) through noise_law.draw_units(rng, 1, n_samples), plus
    tilt_mV_per_s * (j - (n_samples - 1) / 2) * dt. No offset. ValueError:
    SPEC, Block 10."""
    n = _check_common(tilt_mV_per_s, dt, n_samples, "n_samples")
    units = draw_units(twin_rng(base_seed, twin_id, "LS", sweep_number), 1, n)
    out = simulate_sweep(law, float(dt), n, units, 0, tilt_mV_per_s=float(tilt_mV_per_s))
    return np.ascontiguousarray(out, dtype=np.float64)
