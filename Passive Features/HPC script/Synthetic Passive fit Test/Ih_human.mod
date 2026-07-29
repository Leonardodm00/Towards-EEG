TITLE Ih_human
COMMENT
Hyperpolarisation-activated cation current (h-current, HCN) -- HUMAN kinetics.

Kinetics: Rich, Moradi Chameh, Sekulic, Valiante & Skinner (2021),
"Modeling Reveals Human-Rodent Differences in H-Current Kinetics Influencing
Resonance in Cortical Layer 5 Neurons", Cerebral Cortex 31(2):845-872.
doi:10.1093/cercor/bhaa261.  Equation (1) and Table 1 of that paper.

This is the drop-in HUMAN counterpart of Ih.mod, which carries the RODENT
kinetics of Kole, Hallermann & Stuart (2006) as used unaltered in
Hay et al. (2011).  Same NEURON interface (SUFFIX name differs), so the
caller only has to switch the mechanism name.

    ihcn  = gIh * (v - ehcn)                    (mA/cm2)
    gIh   = gIhbar * m                          (S/cm2)
    dm/dt = (mInf - m) / mTau
    mInf  = 1 / (1 + exp((v - vh)/k))
    mTau  = f + 1 / (exp(-a - b*v) + exp(-c + d*v))

Fitted parameter values (Rich et al. 2021, Table 1, "L5 Human model"):
    a    =  23.45          b    =   0.22
    c    =   1.31e-09      d    =   0.083
    f    =   1.50e-09      k    =   8.05
    vh   = -90.87  mV      ehcn = -49.85  mV
    gIhbar (soma + basilar) = 5.14e-05 S/cm2
        (compare Hay et al. 2011 rodent L5: 2.00e-04 S/cm2)

Sign/shape check (independent of the paper's figures): with k = +8.05 and
vh = -90.87 mV, mInf increases as v becomes MORE negative, i.e. the current
activates on hyperpolarisation, as required.  At v = -72.43 mV (the resting
potential Rich et al. report for their L5 Human model) this gives
mInf = 0.092, consistent with the ~0.075 they quote for that model.

NUMERICAL NOTE / DEVIATION FROM THE PUBLISHED MODEL
---------------------------------------------------
Rich et al. explicitly caution (their Discussion of Fig. 6/Fig. 8) that in
their model mTau decays toward 0 at strongly hyperpolarised voltages faster
than the voltage-clamp data suggest.  At v = -140 mV the formula above gives
mTau ~ 6.4e-04 ms.  That is numerically harmless for METHOD cnexp (which is
exact for a linear ODE) but is not biophysically meaningful.  The RANGE
parameter mTauMin (ms) imposes an optional lower floor on mTau.

    mTauMin = 0    -> faithful to the published model  (DEFAULT)
    mTauMin > 0    -> DEVIATION; use only for an explicit sensitivity check,
                      and report it.

f = 1.50e-09 ms already guarantees mTau > 0, so there is no division hazard
when mTauMin = 0.

UNITS CAVEAT
------------
Rich et al. label a, b, c, d, f as "optimized parameters (ms)".  Dimensionally
that cannot hold for b and d, which multiply a voltage inside an exponential
(so b and d carry mV^-1) and for a and c, which are pure numbers.  Only f is a
time.  The rates() block is therefore wrapped in UNITSOFF and these five
parameters are declared without units, exactly as the published formula uses
them.  This is a transcription of their equation, not a reinterpretation.

ehcn and gIhbar are RANGE so they can be set per segment from Python, matching
the interface of Ih.mod (see SyntheticPassiveCell._insert_ih).
ENDCOMMENT

NEURON {
    SUFFIX Ih_human
    NONSPECIFIC_CURRENT ihcn
    RANGE gIhbar, gIh, ihcn, ehcn, mTauMin
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gIhbar  = 5.14e-05 (S/cm2)   : Rich et al. 2021 Table 1, soma + basilar
    ehcn    = -49.85   (mV)      : Rich et al. 2021 Table 1
    mTauMin = 0.0      (ms)      : 0 = published model; >0 = documented deviation

    : --- Rich et al. 2021, Eq. (1) + Table 1 (see UNITS CAVEAT above) ---
    vh = -90.87 (mV)
    kk = 8.05                    : "k" in the paper; renamed to avoid a clash
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
    LOCAL x, e1, e2
    UNITSOFF
    : ---- steady-state activation ----
    x = (v - vh) / kk
    if (x > 50) { x = 50 }
    if (x < -50) { x = -50 }
    mInf = 1 / (1 + exp(x))

    : ---- activation time constant ----
    x = -aa - bb * v
    if (x > 200) { x = 200 }
    if (x < -200) { x = -200 }
    e1 = exp(x)

    x = -cc + dd * v
    if (x > 200) { x = 200 }
    if (x < -200) { x = -200 }
    e2 = exp(x)

    mTau = ff + 1 / (e1 + e2)
    if (mTau < mTauMin) { mTau = mTauMin }
    UNITSON
}
