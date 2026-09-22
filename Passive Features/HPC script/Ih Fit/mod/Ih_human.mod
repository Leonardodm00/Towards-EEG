TITLE Ih_human
COMMENT
Hyperpolarisation-activated cation current (h-current, HCN) -- HUMAN kinetics,
extended with three RANGE knobs for the I_h fitting campaign (decisions D-005,
D-006, D-007 in TEEG_decisions_and_ideas_log.md).

Kinetics: Rich, Moradi Chameh, Sekulic, Valiante & Skinner (2021),
"Modeling Reveals Human-Rodent Differences in H-Current Kinetics Influencing
Resonance in Cortical Layer 5 Neurons", Cerebral Cortex 31(2):845-872,
doi:10.1093/cercor/bhaa261, Equation (1) and Table 1 ("L5 Human model").

    ihcn  = gIh * (v - ehcn)                          (mA/cm2)
    gIh   = gIhbar * m                                (S/cm2)
    dm/dt = (mInf - m) / mTau
    mInf  = 1 / (1 + exp((vm - vh)/k)),   vm = v - vshift - vshift_minf
    mTau  = tau_scale * ( f + 1 / (exp(-a - b*vt) + exp(-c + d*vt)) ),
                                          vt = v - vshift

Table 1 values: a = 23.45, b = 0.22, c = 1.31e-09, d = 0.083, f = 1.50e-09,
k = 8.05, vh = -90.87 mV, ehcn = -49.85 mV, gIhbar (soma + basilar) =
5.14e-05 S/cm2.

The three knobs (all default to the published model):

    vshift      (mV)  configuration shift of BOTH curves along v; positive =
                      more activation at a given v. No published human variant
                      uses it; it exists so Ih and Ih_human expose one
                      interface. NOT fitted. Default 0.
    vshift_minf (mV)  FITTED shift of the activation curve ONLY (Delta v_h):
                      vh -> vh + vshift_minf. Positive = more activation at a
                      given v.
    tau_scale   (1)   FITTED multiplicative scale on mTau at every v (kappa_tau).

NUMERICAL NOTE (unchanged from the earlier file): Rich et al. caution that mTau
in their model decays toward 0 at strongly hyperpolarised voltages faster than
the voltage-clamp data suggest; mTauMin (ms) is an optional floor. 0 = the
published model. A floor is applied AFTER tau_scale.

UNITS CAVEAT (unchanged): a, b, c, d, f are dimensionless / mV^-1 / ms as the
published formula uses them; the rates block is UNITSOFF.
ENDCOMMENT

NEURON {
    SUFFIX Ih_human
    NONSPECIFIC_CURRENT ihcn
    RANGE gIhbar, gIh, ihcn, ehcn, vshift, vshift_minf, tau_scale, mTauMin, mInf, mTau
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gIhbar      = 5.14e-05 (S/cm2)   : Rich et al. 2021 Table 1, soma + basilar
    ehcn        = -49.85   (mV)      : Rich et al. 2021 Table 1
    vshift      = 0.0      (mV)      : configuration shift of both curves (not fitted)
    vshift_minf = 0.0      (mV)      : fitted shift of the activation curve only
    tau_scale   = 1.0                : fitted multiplicative scale on mTau
    mTauMin     = 0.0      (ms)      : 0 = published model; >0 = documented deviation

    : --- Rich et al. 2021, Eq. (1) + Table 1 ---
    vh = -90.87 (mV)
    kk = 8.05                        : "k" in the paper; renamed to avoid a clash
    aa = 23.45
    bb = 0.22
    cc = 1.31e-09
    dd = 0.083
    ff = 1.50e-09
}

ASSIGNED {
    v      (mV)
    ihcn   (mA/cm2)
    gIh    (S/cm2)
    mInf
    mTau   (ms)
}

STATE { m }

BREAKPOINT {
    SOLVE states METHOD cnexp
    gIh  = gIhbar * m
    ihcn = gIh * (v - ehcn)
}

DERIVATIVE states {
    rates()
    m' = (mInf - m) / mTau
}

INITIAL {
    rates()
    m = mInf
}

PROCEDURE rates() {
    LOCAL x, e1, e2, vt, vm
    UNITSOFF
    vt = v - vshift
    vm = v - vshift - vshift_minf

    : ---- steady-state activation (shifted curve) ----
    x = (vm - vh) / kk
    if (x > 50) { x = 50 }
    if (x < -50) { x = -50 }
    mInf = 1 / (1 + exp(x))

    : ---- activation time constant (scaled) ----
    x = -aa - bb * vt
    if (x > 200) { x = 200 }
    if (x < -200) { x = -200 }
    e1 = exp(x)

    x = -cc + dd * vt
    if (x > 200) { x = 200 }
    if (x < -200) { x = -200 }
    e2 = exp(x)

    mTau = tau_scale * (ff + 1 / (e1 + e2))
    if (mTau < mTauMin) { mTau = mTauMin }
    UNITSON
}
