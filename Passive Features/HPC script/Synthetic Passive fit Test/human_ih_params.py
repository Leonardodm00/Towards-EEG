"""human_ih_params.py -- literature-sourced h-current and passive priors.

SINGLE SOURCE OF TRUTH for the "is this parameter human or rodent?" question in
the synthetic passive-fit benchmark.  Import this module instead of hard-coding
numbers in synth_gt_grid.py / submit_all_cohorts.sh.

Every value below carries the citation it came from.  Nothing here is invented.

NOTATION AND UNITS (carried explicitly, never abbreviated)
----------------------------------------------------------
    v          membrane potential                              [mV]
    m          h-current activation gating variable, m in [0,1] [dimensionless]
    mInf(v)    steady-state activation, for each fixed v        [dimensionless]
    mTau(v)    activation time constant, for each fixed v       [ms]
    gIhbar     maximal h-conductance density                    [S/cm^2]
    ehcn       h-current reversal potential                     [mV]
    Cm         specific membrane capacitance                    [uF/cm^2]
    Rm         specific membrane resistance                     [Ohm*cm^2]
    Ra         specific axial resistivity                       [Ohm*cm]
    tau_m      passive membrane time constant = Rm*Cm*1e-3      [ms]
    R_in       somatic input resistance                         [MOhm]

CONVENTIONS
-----------
  * tau_m is written with its 1e-3 factor explicit (Ohm*uF -> ms); it is never
    absorbed into a constant.
  * "rodent" below always means rat L5 (Kole et al. 2006 / Hay et al. 2011).
  * R_in values are reported WITH the holding potential at which they were
    measured, because R_in in a cell with I_h is holding-potential dependent.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

__all__ = [
    "IH_KINETICS_RODENT_KOLE",
    "IH_KINETICS_HUMAN_RICH",
    "IH_DENSITY",
    "HUMAN_L23_L3_PASSIVE",
    "HUMAN_R_IN_MOHM",
    "HUMAN_TAU_M_MS",
    "minf_rich_human",
    "mtau_rich_human_ms",
    "minf_kole_rodent",
    "mtau_kole_rodent_ms",
    "peak_mtau_ms",
]


# ===========================================================================
#  1. h-current KINETICS
# ===========================================================================

IH_KINETICS_RODENT_KOLE: Dict[str, object] = {
    "mechanism": "Ih",
    "source": (
        "Kole, Hallermann & Stuart (2006), rat L5; used UNALTERED in "
        "Hay, Hill, Schuermann, Markram & Segev (2011), ModelDB 139653."
    ),
    "species": "rat",
    "layer": "L5",
    "ehcn_mV": -45.0,
    # mAlpha = 0.001*6.43*(v+154.9)/(exp((v+154.9)/11.9)-1)
    # mBeta  = 0.001*193*exp(v/33.1)
    # mTau   = 1/(mAlpha+mBeta);  mInf = mAlpha/(mAlpha+mBeta)
    "note": "This is what Ih.mod in the repository currently implements.",
}

IH_KINETICS_HUMAN_RICH: Dict[str, object] = {
    "mechanism": "Ih_human",
    "source": (
        "Rich, Moradi Chameh, Sekulic, Valiante & Skinner (2021), "
        "Cereb Cortex 31(2):845-872, doi:10.1093/cercor/bhaa261, "
        "Eq. (1) and Table 1 ('L5 Human model')."
    ),
    "species": "human",
    "layer": "L5",
    "ehcn_mV": -49.85,
    "vh_mV": -90.87,
    "k": 8.05,
    "a": 23.45,
    "b": 0.22,
    "c": 1.31e-09,
    "d": 0.083,
    "f": 1.50e-09,
    "note": (
        "Fitted from scratch against human L5 TTX current-clamp data. "
        "Cross-layer transfer to L3 morphologies is supported by "
        "Moradi Chameh et al. (2021), Nat Commun 12:2497, "
        "doi:10.1038/s41467-021-22741-9, Supplementary Fig. 5b: human L2&3 "
        "and L5 I_h ACTIVATION TIME CONSTANTS are statistically "
        "indistinguishable (p >= 0.9999; L2&3 n=6, L5 n=10); only the "
        "amplitude differs. See also the notes on IH_DENSITY below."
    ),
}


# ===========================================================================
#  2. h-current DENSITY  (gIhbar, S/cm^2)
# ===========================================================================
# The density is the parameter that DOES differ by layer, so it must not be
# taken from the L5 fit when the morphologies are L3.

IH_DENSITY: Dict[str, Dict[str, object]] = {
    "hay_rat_L5": {
        "gIhbar_S_cm2": 2.00e-04,
        "distribution": "hay_exponential",
        "source": "Hay et al. (2011), rat L5. Reported in Rich et al. 2021 Table 1.",
        "species": "rat",
        "WARNING": (
            "This is the value the benchmark currently uses. It is the RAT L5 "
            "somatic density. Eyal et al. (2016), eLife 5:e16553, "
            "Fig. 1-figure supplement 3, show that inserting I_h at exactly "
            "this somatic density (0.2 mS/cm^2, Kole rat L5) into human L2/3 "
            "cells raises the BEST-FIT Cm from ~0.45 to ~0.76 uF/cm^2. In a "
            "benchmark whose whole purpose is Cm recovery, that is a large "
            "species-imported bias."
        ),
    },
    "rich_human_L5": {
        "gIhbar_S_cm2": 5.14e-05,
        "distribution": "hay_exponential",
        "source": "Rich et al. (2021) Table 1, soma + basilar dendrites.",
        "species": "human",
        "layer": "L5",
    },
    "kalmbach_human_L3": {
        "gIhbar_S_cm2": 1.00e-04,
        "distribution": "uniform",
        "source": (
            "Kalmbach, Buchin, Long, ... Ting (2018), Neuron 100(5):1194-1208, "
            "doi:10.1016/j.neuron.2018.10.012 (human deep L3 model, uniform "
            "h-channel density over soma + axon + dendrites). Exact value as "
            "re-implemented verbatim and tabulated by Rich et al. (2021), "
            "Table 3."
        ),
        "species": "human",
        "layer": "deep L3",
        "PREFERRED_FOR_THIS_BENCHMARK": (
            "The morphology pool is L3_exc, so this is the layer-matched "
            "density. NOTE that Kalmbach et al. themselves used the RODENT "
            "Kole kinetics shifted by -20 mV; only the density is human. "
            "Pairing this density with Ih_human kinetics is a deliberate "
            "recombination, not something any single paper did."
        ),
    },
}


# ===========================================================================
#  3. HUMAN passive properties and R_in for L2/3 - L3 pyramidal neurons
# ===========================================================================

HUMAN_L23_L3_PASSIVE: Dict[str, object] = {
    "Cm_uF_cm2_range": (0.43, 0.52),
    "Cm_uF_cm2_mean_sd": (0.49, 0.08),
    "Rm_Ohm_cm2_range": (21400.0, 48300.0),
    "Ra_Ohm_cm_range": (203.0, 384.0),
    "Ra_Ohm_cm_mean_sd": (268.5, 30.0),
    "spine_factor_F": 1.9,
    "source": (
        "Eyal, Verhoog, Testa-Silva, ... Segev (2016), eLife 5:e16553, "
        "doi:10.7554/eLife.16553, Fig. 1 (n=6 model-fitted cells) plus "
        "nucleated-patch validation (n=5). Spine factor F=1.9 beyond 60 um is "
        "the same value Kalmbach et al. (2018) used."
    ),
}

# R_in depends on holding potential when I_h is present -- carried explicitly.
HUMAN_R_IN_MOHM: Dict[str, Dict[str, object]] = {
    "chameh2021_L2_3_at_RMP": {
        "mean": 83.0, "sd": 38.1, "n": 56, "holding": "RMP",
        "source": ("Moradi Chameh et al. (2021), Nat Commun 12:2497, Fig. 1c; "
                   "measured from hyperpolarising sweeps -50 to -200 pA."),
    },
    "chameh2021_L3c_at_RMP": {
        "mean": 79.4, "sd": 21.4, "n": 15, "holding": "RMP",
        "source": "Moradi Chameh et al. (2021), Nat Commun 12:2497, Fig. 1c.",
    },
    "chameh2021_L5_at_RMP": {
        "mean": 94.2, "sd": 41.3, "n": 105, "holding": "RMP",
        "source": "Moradi Chameh et al. (2021), Nat Commun 12:2497, Fig. 1c.",
    },
    "kalmbach2018_deepL3_at_minus65_Ih_intact": {
        "mean": 48.92, "sem": 4.54, "holding": "-65 mV",
        "source": ("Kalmbach et al. (2018), Neuron, ZD7288 experiment, "
                   "pre-drug value."),
    },
    "kalmbach2018_deepL3_at_minus65_Ih_blocked": {
        "mean": 76.39, "sem": 6.47, "holding": "-65 mV, +10 uM ZD7288",
        "source": ("Kalmbach et al. (2018), Neuron. Blocking I_h raised R_in "
                   "by 60.43 +/- 9.94 % in human vs 13.27 +/- 4.73 % in mouse "
                   "(mouse: 136.61 +/- 4.27 -> 155.91 +/- 7.86 MOhm)."),
    },
}

HUMAN_TAU_M_MS: Dict[str, Dict[str, object]] = {
    "chameh2021_L2_3": {"mean": 13.7, "sd": 7.1, "n": 56,
                        "source": "Moradi Chameh et al. (2021), Fig. 1d."},
    "chameh2021_L3c": {"mean": 17.1, "sd": 5.7, "n": 15,
                       "source": "Moradi Chameh et al. (2021), Fig. 1d."},
    "chameh2021_L5": {"mean": 19.3, "sd": 9.1, "n": 105,
                      "source": "Moradi Chameh et al. (2021), Fig. 1d."},
    "eyal2016_L2_3_model_fits": {"mean": 16.5, "sd": 3.7, "n": 6,
                                 "range": (10.0, 22.0),
                                 "source": "Eyal et al. (2016), eLife."},
}


# ===========================================================================
#  4. Reference implementations of mInf(v) and mTau(v)  (pure Python)
# ===========================================================================
# These exist so the compiled NMODL mechanisms can be checked against an
# independent implementation of the SAME published equations (smoke test).

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def minf_rich_human(v_mV: float) -> float:
    """Steady-state h-current activation, human, for each fixed v_mV [mV].

    Rich et al. (2021) Eq. (1): mInf(v) = 1/(1 + exp((v - vh)/k)).
    Returns a dimensionless value in (0, 1).
    """
    p = IH_KINETICS_HUMAN_RICH
    x = _clamp((v_mV - float(p["vh_mV"])) / float(p["k"]), -50.0, 50.0)
    return 1.0 / (1.0 + math.exp(x))


def mtau_rich_human_ms(v_mV: float, mtau_min_ms: float = 0.0) -> float:
    """h-current activation time constant [ms], human, for each fixed v_mV [mV].

    Rich et al. (2021) Eq. (1):
        mTau(v) = f + 1/(exp(-a - b*v) + exp(-c + d*v)).
    mtau_min_ms > 0 imposes a floor and is a DEVIATION from the published model.
    """
    p = IH_KINETICS_HUMAN_RICH
    e1 = math.exp(_clamp(-float(p["a"]) - float(p["b"]) * v_mV, -200.0, 200.0))
    e2 = math.exp(_clamp(-float(p["c"]) + float(p["d"]) * v_mV, -200.0, 200.0))
    tau = float(p["f"]) + 1.0 / (e1 + e2)
    return max(tau, float(mtau_min_ms))


def _kole_rates(v_mV: float) -> Tuple[float, float]:
    """(mAlpha, mBeta) in 1/ms for the Kole et al. (2006) rat model."""
    vv = v_mV + 154.9
    if abs(vv) < 1e-9:
        vv = 1e-9
    m_alpha = 0.001 * 6.43 * vv / (math.exp(vv / 11.9) - 1.0)
    m_beta = 0.001 * 193.0 * math.exp(v_mV / 33.1)
    return m_alpha, m_beta


def minf_kole_rodent(v_mV: float) -> float:
    """Steady-state activation, rat (Kole et al. 2006), for each fixed v_mV."""
    a, b = _kole_rates(v_mV)
    return a / (a + b)


def mtau_kole_rodent_ms(v_mV: float) -> float:
    """Activation time constant [ms], rat (Kole et al. 2006), for fixed v_mV."""
    a, b = _kole_rates(v_mV)
    return 1.0 / (a + b)


def peak_mtau_ms(fn, v_lo: float = -140.0, v_hi: float = -40.0,
                 dv: float = 0.1) -> Tuple[float, float]:
    """Return (v_at_peak_mV, peak_mTau_ms) of a mTau(v) callable on a grid.

    The scan is over v in [v_lo, v_hi] with step dv; all in mV.
    """
    n = int(round((v_hi - v_lo) / dv)) + 1
    best_v, best_t = v_lo, -1.0
    for i in range(n):
        v = v_lo + i * dv
        t = fn(v)
        if t > best_t:
            best_v, best_t = v, t
    return best_v, best_t
