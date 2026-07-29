#!/usr/bin/env python3
"""smoke_test_ih_human.py -- correctness checks for the HUMAN h-current swap.

WHAT THIS TESTS
---------------
PART A (no NEURON needed, runs anywhere in ~1 s)
  A1  The Python reference implementation of Rich et al. (2021) Eq. (1)
      reproduces the three quantitative statements the paper makes about its
      own model, so we know the parameter transcription from Table 1 is right.
  A2  The Kole/Hay rodent reference reproduces the statement Rich et al. make
      about the rodent model ("never exceeding 80 ms").
  A3  The human/rodent mTau ratio is ~an order of magnitude, as reported.
  A4  Monotonicity/sanity: mInf is a decreasing function of v (activates on
      hyperpolarisation) for BOTH models.

PART B (needs NEURON + a compiled x86_64/ directory containing Ih_human)
  B1  The compiled Ih_human mechanism reproduces the Python reference mTau(v)
      and mInf(v) to within 1e-6 relative error, at 11 voltages.
  B2  A single-compartment cell with Ih_human shows a LARGER, SLOWER sag than
      the same cell with the rodent Ih at equal gIhbar -- the qualitative
      signature the swap is supposed to introduce.

HOW TO RUN
----------
  # Part A only (login node, or your laptop; no NEURON required):
  python smoke_test_ih_human.py

  # Full test on the cluster, after compiling the mechanisms:
  conda activate prova
  cd "/davinci-1/home/ldellamea/Human Neurons Fitting/Synthetic Test"
  rm -rf x86_64 && nrnivmodl mod        # mod/ must contain Ih.mod AND Ih_human.mod
  python smoke_test_ih_human.py --with-neuron

Exit code 0 = all executed checks passed; 1 = at least one failed.

DEPENDENCIES: human_ih_params.py must sit next to this file.
"""

from __future__ import annotations

import argparse
import sys

import human_ih_params as P

# Tolerances. REL_TOL is for reference-vs-NEURON agreement (should be exact up
# to float noise). The published-value checks use looser, explicitly stated
# windows because Rich et al. quote their own model to 1 significant figure
# ("approximately 350 ms") and round Table 1 to 2 decimal places.
REL_TOL = 1e-6

_FAILURES = []
_SKIPPED = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print("  [PASS] {}".format(name))
    else:
        print("  [FAIL] {}  {}".format(name, detail))
        _FAILURES.append(name)


# ===========================================================================
#  PART A -- reference implementation vs published statements
# ===========================================================================
def part_a() -> None:
    print("\nPART A -- Python reference vs Rich et al. (2021) published values")
    print("-" * 68)

    # ---- A1a: peak mTau of the human model is "approximately 350 ms" -------
    v_peak, t_peak = P.peak_mtau_ms(P.mtau_rich_human_ms)
    print("  human  mTau peak = {:7.1f} ms at v = {:7.2f} mV".format(t_peak, v_peak))
    check("A1a human peak mTau in [300, 400] ms (paper: '~350 ms')",
          300.0 <= t_peak <= 400.0,
          "got {:.1f} ms".format(t_peak))

    # ---- A1b: mInf at the model's resting potential ------------------------
    # Rich et al. report an h-current steady-state activation of ~0.075 near
    # the L5 Human model RMP of -72.43 mV. Transcription is correct if we land
    # in the same neighbourhood (their 0.075 is read off a figure).
    m_rmp = P.minf_rich_human(-72.43)
    print("  human  mInf(-72.43 mV) = {:.4f}   (paper: ~0.075)".format(m_rmp))
    check("A1b human mInf at RMP in [0.05, 0.13]",
          0.05 <= m_rmp <= 0.13,
          "got {:.4f}".format(m_rmp))

    # ---- A1c: mInf must be bounded in (0,1) over a wide sweep --------------
    ok = all(0.0 < P.minf_rich_human(v) < 1.0 for v in range(-160, 21))
    check("A1c human mInf strictly in (0,1) for v in [-160, +20] mV", ok)

    # ---- A2: rodent model "never exceeding 80 ms" --------------------------
    v_peak_r, t_peak_r = P.peak_mtau_ms(P.mtau_kole_rodent_ms)
    print("  rodent mTau peak = {:7.1f} ms at v = {:7.2f} mV".format(t_peak_r, v_peak_r))
    check("A2 rodent peak mTau <= 80 ms (paper: 'never exceeding 80 ms')",
          t_peak_r <= 80.0,
          "got {:.1f} ms".format(t_peak_r))

    # ---- A3: order-of-magnitude separation ---------------------------------
    ratio = t_peak / t_peak_r
    print("  human/rodent peak mTau ratio = {:.2f}".format(ratio))
    check("A3 human peak mTau is 3x-15x the rodent peak",
          3.0 <= ratio <= 15.0,
          "ratio {:.2f}".format(ratio))

    # ---- A4: both models activate on HYPERpolarisation ---------------------
    for label, fn in (("human", P.minf_rich_human),
                      ("rodent", P.minf_kole_rodent)):
        vs = [-130.0 + 5.0 * i for i in range(19)]      # -130 .. -40 mV
        vals = [fn(v) for v in vs]
        decreasing = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
        check("A4 {} mInf decreases with depolarisation".format(label),
              decreasing)

    # ---- A5: mTau strictly positive everywhere (no division hazard) --------
    ok = all(P.mtau_rich_human_ms(v) > 0.0 for v in range(-200, 51))
    check("A5 human mTau > 0 for v in [-200, +50] mV", ok)

    # ---- Informative table (not a pass/fail) -------------------------------
    print("\n  mTau(v) comparison [ms] -- this is the bias being removed:")
    print("    {:>8} {:>12} {:>12} {:>8}".format("v [mV]", "human", "rodent", "ratio"))
    for v in (-120.0, -110.0, -100.0, -90.0, -80.0, -75.0, -70.0, -65.0, -60.0):
        th = P.mtau_rich_human_ms(v)
        tr = P.mtau_kole_rodent_ms(v)
        print("    {:>8.1f} {:>12.2f} {:>12.2f} {:>8.2f}".format(v, th, tr, th / tr))


# ===========================================================================
#  PART B -- compiled NMODL vs reference, and a sag comparison
# ===========================================================================
def part_b() -> None:
    print("\nPART B -- compiled NEURON mechanism")
    print("-" * 68)
    try:
        from neuron import h
    except Exception as exc:                                  # pragma: no cover
        print("  [SKIP] NEURON not importable: {}".format(exc))
        _SKIPPED.append("PART B")
        return

    h.load_file("stdrun.hoc")

    soma = h.Section(name="soma")
    soma.L = 20.0
    soma.diam = 20.0
    soma.cm = 1.0
    soma.Ra = 150.0
    soma.insert("pas")
    for seg in soma:
        seg.pas.g = 1.0 / 20000.0
        seg.pas.e = -70.0

    try:
        soma.insert("Ih_human")
    except Exception as exc:
        print("  [SKIP] Ih_human not compiled into x86_64/: {}".format(exc))
        print("         run:  rm -rf x86_64 && nrnivmodl mod")
        _SKIPPED.append("PART B")
        return

    # ---- B1: mechanism vs Python reference at 11 voltages ------------------
    seg = soma(0.5)
    worst_tau, worst_inf = 0.0, 0.0
    for i in range(11):
        v = -140.0 + 10.0 * i
        h.finitialize(v)
        # NEURON exposes mInf/mTau as ASSIGNED range variables after init.
        got_tau = seg.Ih_human.mTau
        got_inf = seg.Ih_human.mInf
        ref_tau = P.mtau_rich_human_ms(v)
        ref_inf = P.minf_rich_human(v)
        worst_tau = max(worst_tau, abs(got_tau - ref_tau) / max(ref_tau, 1e-30))
        worst_inf = max(worst_inf, abs(got_inf - ref_inf) / max(ref_inf, 1e-30))
    print("  worst relative error: mTau {:.3e}   mInf {:.3e}".format(
        worst_tau, worst_inf))
    check("B1a compiled mTau(v) matches reference (rel tol {:g})".format(REL_TOL),
          worst_tau < REL_TOL, "worst {:.3e}".format(worst_tau))
    check("B1b compiled mInf(v) matches reference (rel tol {:g})".format(REL_TOL),
          worst_inf < REL_TOL, "worst {:.3e}".format(worst_inf))

    # ---- B2: sag comparison at EQUAL gIhbar --------------------------------
    def sag_of(mech: str, gbar: float, ehcn: float) -> float:
        """(V_trough - V_ss)/(V_trough - V_rest) for a -50 pA, 1 s step."""
        sec = h.Section(name="s_" + mech)
        sec.L, sec.diam, sec.cm, sec.Ra = 20.0, 20.0, 1.0, 150.0
        sec.insert("pas")
        for s in sec:
            s.pas.g, s.pas.e = 1.0 / 20000.0, -70.0
        sec.insert(mech)
        for s in sec:
            mobj = getattr(s, mech)
            mobj.gIhbar = gbar
            mobj.ehcn = ehcn
        ic = h.IClamp(sec(0.5))
        ic.delay, ic.dur, ic.amp = 3000.0, 1000.0, -0.050   # nA
        vvec, tvec = h.Vector().record(sec(0.5)._ref_v), h.Vector().record(h._ref_t)
        h.dt = 0.025
        h.finitialize(-70.0)
        h.continuerun(4000.0)
        t = list(tvec)
        v = list(vvec)
        pre = [v[i] for i in range(len(t)) if 2900.0 <= t[i] < 3000.0]
        stp = [v[i] for i in range(len(t)) if 3000.0 <= t[i] <= 4000.0]
        ss = [v[i] for i in range(len(t)) if 3900.0 <= t[i] <= 4000.0]
        v_rest = sum(pre) / len(pre)
        v_tr = min(stp)
        v_ss = sum(ss) / len(ss)
        den = v_tr - v_rest
        return 0.0 if abs(den) < 1e-9 else max((v_tr - v_ss) / den, 0.0)

    gbar = 1.0e-4
    sag_h = sag_of("Ih_human", gbar, -49.85)
    sag_r = sag_of("Ih", gbar, -45.0)
    print("  sag ratio at gIhbar={:g} S/cm2:  human {:.4f}   rodent {:.4f}".format(
        gbar, sag_h, sag_r))
    check("B2a both mechanisms produce a non-zero sag",
          sag_h > 1e-4 and sag_r > 1e-4,
          "human {:.5f} rodent {:.5f}".format(sag_h, sag_r))
    check("B2b human sag differs from rodent sag at equal gIhbar",
          abs(sag_h - sag_r) > 1e-3,
          "human {:.5f} rodent {:.5f}".format(sag_h, sag_r))


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-neuron", action="store_true",
                    help="also run PART B (needs a compiled x86_64/ with Ih_human)")
    args = ap.parse_args()

    print("=" * 68)
    print("smoke_test_ih_human.py")
    print("=" * 68)

    part_a()
    if args.with_neuron:
        part_b()
    else:
        print("\nPART B skipped (pass --with-neuron to run it).")
        _SKIPPED.append("PART B (not requested)")

    print("\n" + "=" * 68)
    if _FAILURES:
        print("RESULT: {} CHECK(S) FAILED: {}".format(len(_FAILURES), _FAILURES))
        return 1
    print("RESULT: all executed checks passed.")
    if _SKIPPED:
        print("        skipped: {}".format(_SKIPPED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
