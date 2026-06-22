TITLE Ih
COMMENT
Hyperpolarisation-activated cation current (h-current, HCN).

Kinetics: Kole, Hallermann & Stuart (2006), used UNALTERED in Hay, Hill,
Schuermann, Markram & Segev (2011) (ModelDB 139653), and adopted by Eyal,
Verhoog, Testa-Silva, Deitcher, ... Segev (2018) "Human cortical pyramidal
neurons: from spines to spikes via models" (Front Cell Neurosci 12:181),
whose supplementary states the channel models are "as described in
(Hay et al., 2011)".

    ihcn = gIh * (v - ehcn)            (mA/cm2)
    gIh  = gIhbar * m                  (S/cm2)
    dm/dt = (mInf - m)/mTau
    mAlpha = 0.001 * 6.43 * (v + 154.9) / (exp((v + 154.9)/11.9) - 1)
    mBeta  = 0.001 * 193  * exp(v/33.1)
    mInf = mAlpha/(mAlpha + mBeta)
    mTau = 1/(mAlpha + mBeta)

ehcn = -45 mV is the Hay/Kole reversal. gIhbar (S/cm2) is the optimised
density; the Hay apical exponential-with-distance distribution is applied by
the caller (SyntheticPassiveCell._insert_ih, mode "hay_exponential"); here
gIhbar is the per-segment value. ehcn and gIhbar are RANGE so they can be
set per segment from Python.
ENDCOMMENT

NEURON {
    SUFFIX Ih
    NONSPECIFIC_CURRENT ihcn
    RANGE gIhbar, gIh, ihcn, ehcn
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gIhbar = 0.00001 (S/cm2)
    ehcn   = -45.0   (mV)
}

ASSIGNED {
    v      (mV)
    ihcn   (mA/cm2)
    gIh    (S/cm2)
    mInf
    mTau
    mAlpha
    mBeta
}

STATE { m }

BREAKPOINT {
    SOLVE states METHOD cnexp
    gIh = gIhbar * m
    ihcn = gIh * (v - ehcn)
}

DERIVATIVE states {
    rates()
    m' = (mInf - m)/mTau
}

INITIAL {
    rates()
    m = mInf
}

PROCEDURE rates() {
    UNITSOFF
    if (v == -154.9) { v = v + 0.0001 }
    mAlpha = 0.001 * 6.43 * (v + 154.9) / (exp((v + 154.9)/11.9) - 1)
    mBeta  = 0.001 * 193 * exp(v/33.1)
    mInf = mAlpha/(mAlpha + mBeta)
    mTau = 1/(mAlpha + mBeta)
    UNITSON
}
