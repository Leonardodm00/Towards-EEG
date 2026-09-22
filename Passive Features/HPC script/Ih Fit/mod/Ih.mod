TITLE Ih
COMMENT
Hyperpolarisation-activated cation current (h-current, HCN) -- RODENT kinetics,
extended with three RANGE knobs for the I_h fitting campaign (decisions D-005,
D-006, D-007 in TEEG_decisions_and_ideas_log.md).

Kinetics: Kole, Hallermann & Stuart (2006), used UNALTERED in Hay, Hill,
Schuermann, Markram & Segev (2011) (ModelDB 139653).

    ihcn   = gIh * (v - ehcn)                     (mA/cm2)
    gIh    = gIhbar * m                           (S/cm2)
    dm/dt  = (mInf - m) / mTau
    alpha(u) = 0.001 * 6.43 * (u + 154.9) / (exp((u + 154.9)/11.9) - 1)
    beta(u)  = 0.001 * 193 * exp(u/33.1)
    mInf   = alpha(vm) / (alpha(vm) + beta(vm)),   vm = v - vshift - vshift_minf
    mTau   = tau_scale / (alpha(vt) + beta(vt)),    vt = v - vshift

The three knobs (all default to the published model, so an unmodified caller
reproduces the original Ih.mod bit-for-bit):

    vshift      (mV)  configuration shift of BOTH curves along v; positive
                      moves both curves to more depolarised voltages (more
                      activation at a given v). Kalmbach et al. 2018 (human
                      deep-L3 model) evaluate the Kole rates at (v - 20) --
                      their text calls this a "-20 mV shift" because the sign
                      refers to the term inside the rate argument -- which in
                      THIS convention is vshift = +20. NOT fitted.
    vshift_minf (mV)  FITTED shift of the activation curve ONLY (Delta v_h in the
                      plan). Positive = more activation at a given v. It moves
                      mInf at rest AND at the trough, which is what makes it
                      identifiable from the sag fraction; mTau is untouched.
    tau_scale   (1)   FITTED multiplicative scale on mTau at every v (kappa_tau).

ehcn = -45 mV is the Hay/Kole reversal. gIhbar is the per-segment density; any
spatial law (uniform, Eyal 2016 Eq. 4, Hay 2011) is applied by the caller
(ih_mechanism.attach_ih / set_ih). mInf and mTau are RANGE so Python can read
them back after finitialize() and check the compiled mechanism against the
pure-Python reference in human_ih_params.py (smoke_ih_fit.py, check S1).
ENDCOMMENT

NEURON {
    SUFFIX Ih
    NONSPECIFIC_CURRENT ihcn
    RANGE gIhbar, gIh, ihcn, ehcn, vshift, vshift_minf, tau_scale, mInf, mTau
}

UNITS {
    (S)  = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gIhbar      = 0.00001 (S/cm2)
    ehcn        = -45.0   (mV)
    vshift      = 0.0     (mV)   : configuration shift of both curves (not fitted)
    vshift_minf = 0.0     (mV)   : fitted shift of the activation curve only
    tau_scale   = 1.0            : fitted multiplicative scale on mTau
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
    LOCAL vt, vm, at, bt, am, bm
    UNITSOFF
    vt = v - vshift
    vm = v - vshift - vshift_minf
    : removable singularity of alpha at u = -154.9 mV (same guard as the original)
    if (fabs(vt + 154.9) < 1e-6) { vt = vt + 1e-4 }
    if (fabs(vm + 154.9) < 1e-6) { vm = vm + 1e-4 }
    at = 0.001 * 6.43 * (vt + 154.9) / (exp((vt + 154.9)/11.9) - 1)
    bt = 0.001 * 193 * exp(vt/33.1)
    am = 0.001 * 6.43 * (vm + 154.9) / (exp((vm + 154.9)/11.9) - 1)
    bm = 0.001 * 193 * exp(vm/33.1)
    mInf = am/(am + bm)
    mTau = tau_scale/(at + bt)
    UNITSON
}
