# -*- coding: utf-8 -*-
"""
twin_config.py -- the parameters of the twins (SPEC section 2.7; Blocks 5, 10, 11, 12)
=====================================================================================

Parameters only: no logic, no I/O. Every number the twins' modules need that is
not read from the real archive or from the noise-fit tables lives here and is
passed explicitly (specs/SPEC.md, section 2.7). The orchestrator's file: the
blocks read it and do not change it.

Units: seconds, millivolts, picoamperes, hertz; NEURON's step in ms.
"""
from __future__ import annotations

from dataclasses import dataclass

# Protocol codes of the seed rule, SPEC 2.7 (e): the codes of
# noise_sweep_fit.sweep_seeds (0 for SS, 1 for LS).
PROTOCOL_CODE = {"SS": 0, "LS": 1}


@dataclass(frozen=True)
class TwinConfig:
    """The twins' parameters.

    base_seed        : root of every twin noise stream, SPEC 2.7 (e)
                       (D-186.6); set once per run.
    slow_flag_tau_s  : s; a cell whose slow time scale tau_s_hat(c) exceeds it
                       is flagged in the twins' table (D-182.3): 0.265 s, the
                       longest window the noise fit used.
    failed_law_rtol  : relative tolerance of the check that a failed sweep's
                       row already holds the cell's law theta_hat(c)
                       (SPEC 2.7 (b), D-122.3); 1e-9: the tables carry 17
                       significant digits.
    neuron_dt_ms     : ms; NEURON's fixed step for the clean responses
                       (Block 11): 0.025, the generator's value.
    settle_ms        : ms; a clean response is simulated from the state
                       reached after this long without stimulus (SPEC 2.7 (f),
                       D-191). 0 for the twins: their cell is the fitter's
                       model, whose rest is a uniform equilibrium (SPEC 2.7
                       (h), D-192); the build checks it (rest_drift_max_mV).
    rest_drift_max_mV: mV; the build refuses a twin whose unstimulated drift
                       from V_rest exceeds it (D-192): 1e-3, the bound of the
                       fit's smoke S3.
    rest_drift_ms    : ms; the unstimulated run of that check: 2000, the
                       default of ih_mechanism.rest_drift.
    ls_dep_min_amplitude_pA : pA; a Long Square sweep above it is a
                       depolarising sweep, on whose real trace the loader's
                       spike test decides whether the twin has it (D-190): 5.0,
                       the loader's threshold.
    """
    base_seed: int = 20261119
    slow_flag_tau_s: float = 0.265
    failed_law_rtol: float = 1e-9
    neuron_dt_ms: float = 0.025
    settle_ms: float = 0.0
    rest_drift_max_mV: float = 1e-3
    rest_drift_ms: float = 2000.0
    ls_dep_min_amplitude_pA: float = 5.0
