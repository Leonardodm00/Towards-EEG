# -*- coding: utf-8 -*-
"""
twin_noise.py -- the twin's noise: independent SS windows, one continuous LS draw (Block 10)
==========================================================================================

specs/SPEC.md, section 2.7 (c), (d), (e) and the entry of Block 10 (D-180,
D-182, D-183.3, D-186). Model layer: no I/O, no NEURON, no module-level RNG.

CONTRACT STUB (contract: wave 1). The signatures, shapes, dtypes and units
below are the contract; the bodies are written by Block 10's implementer.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from noise_law import NoiseLaw


def twin_rng(base_seed: int, twin_id: int, protocol: str, sweep_number: int,
             position: Optional[int] = None) -> np.random.Generator:
    """The generator of one twin SS window or LS sweep, SPEC 2.7 (e):
    numpy.random.default_rng(SeedSequence([base_seed, twin_id, 0, sweep_number,
    position])) for protocol 'SS' (position required) and
    SeedSequence([base_seed, twin_id, 1, sweep_number]) for 'LS' (position
    forbidden). ValueError: an unknown protocol, 'SS' without a position,
    'LS' with one, a negative integer."""
    raise NotImplementedError("Block 10")


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
    raise NotImplementedError("Block 10")


def ls_sweep_noise(law: NoiseLaw, tilt_mV_per_s: float, dt: float, n_samples: int, *,
                   base_seed: int, twin_id: int, sweep_number: int) -> np.ndarray:
    """The noise and tilt of one Long Square twin sweep, (TW2) without the
    response: float64 (n_samples,) mV. One continuous stationary draw of
    `law` over the whole sweep from twin_rng(base_seed, twin_id, 'LS',
    sweep_number) through noise_law.draw_units(rng, 1, n_samples), plus
    tilt_mV_per_s * (j - (n_samples - 1) / 2) * dt. No offset. ValueError:
    SPEC, Block 10."""
    raise NotImplementedError("Block 10")
