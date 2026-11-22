#!/usr/bin/env python3
"""Read-out of a Stage-7 twins run (run_twin_recovery.py's outputs): what came back, per configuration.

Usage (from anywhere; the results directory is the run's output folder or an extract of it):
    python3 tw1_readout.py <results dir> [<real cell ls_sweeps.npz> <its v_rest_mV>]

<results dir> must hold manifest.csv, twins_status.csv, report/ and fits/twin_<id>/{phase2_results,ls_roles}.csv.
The optional real cell adds its measured sag on the hyperpolarising Long Squares.

What it prints (2026-11-22, run tw1, the twins chat):
  1. per full twin: truth and estimate of gbar, dv_h, kappa_tau; errors of all six parameters; bounds hit;
  2. the voltages the training steps reach and the truth's I_h relative to the leak there;
  3. g_h at rest, gbar * m_inf(V_rest; dv_h): the combination the exponential foot leaves identifiable;
  4. the passive_fit and active_fit twins; the run times;
  5. the foot check: over a voltage range, how far a shift dv of the activation curve is undone by
     scaling gbar by exp(-dv/k) (Rich et al. 2021, Eq. 1 and Table 1: vh = -90.87 mV, k = 8.05 mV,
     the constants of mod/Ih_human.mod).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

VH_MV, K_MV = -90.87, 8.05          # Ih_human: half-activation and slope (Rich et al. 2021, Table 1)


def m_inf(v_mV, dvh_mV, vshift_mV=0.0):
    """Steady-state activation of Ih_human at v, with the fitted shift dv_h (vshift_minf) and the configuration shift."""
    return 1.0 / (1.0 + np.exp((v_mV - vshift_mV - dvh_mV - VH_MV) / K_MV))


def load(d):
    man = pd.read_csv(os.path.join(d, "manifest.csv")).set_index("specimen_id")
    fits = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(d, "fits", "twin_*", "phase2_results.csv")))],
                     ignore_index=True)
    keep = ["configuration", "twin_of_specimen_id", "cell_tau_s_above_window", "cm_true", "rm_true", "ra_true",
            "ih_gihbar_S_cm2", "ih_dvh_mV", "ih_kappa_tau", "ih_vshift_base_mV", "use_ih"]
    f = fits.join(man[keep], on="specimen_id")
    lo, hi = {}, {}
    for path in glob.glob(os.path.join(d, "fits", "twin_*", "ls_roles.csv")):
        r = pd.read_csv(path)
        t = r[r.role == "train"]
        lo[int(r.specimen_id.iloc[0])] = t.v_trough_mV.min()
        hi[int(r.specimen_id.iloc[0])] = t.v_trough_mV.max()
    f["v_train_lo"] = f.specimen_id.map(lo)
    f["v_train_hi"] = f.specimen_id.map(hi)
    vs = f.ih_vshift_base_mV.fillna(0.0)
    f["ih_over_leak_rest_true"] = f.ih_gihbar_S_cm2 * m_inf(f.v_rest_mV, f.ih_dvh_mV, vs) * f.rm_true
    f["ih_over_leak_lo_true"] = f.ih_gihbar_S_cm2 * m_inf(f.v_train_lo, f.ih_dvh_mV, vs) * f.rm_true
    f["ln_gh_rest"] = np.log((f.gbar * m_inf(f.v_rest_mV, f.dv_h, vs)) / (f.ih_gihbar_S_cm2 * m_inf(f.v_rest_mV, f.ih_dvh_mV, vs)))
    for p, t in (("Cm", "cm_true"), ("Rm", "rm_true"), ("Ra", "ra_true")):
        f["ln_" + p] = np.log(f[p] / f[t])
    f["ln_gbar"] = np.log(f.gbar / f.ih_gihbar_S_cm2)
    f["d_dvh"] = f.dv_h - f.ih_dvh_mV
    f["ln_kappa"] = np.log(f.kappa_tau / f.ih_kappa_tau)
    return f


def main(argv):
    d = argv[1]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    f = load(d)
    full = f[f.configuration == "full"]
    print("== 1. full twins (six-parameter fit on traces with I_h): truth, estimate, errors (ln ratio; dv_h in mV)")
    print(full[["specimen_id", "twin_of_specimen_id", "cell_tau_s_above_window", "ih_gihbar_S_cm2", "gbar", "ln_gbar",
                "ih_dvh_mV", "dv_h", "d_dvh", "ih_kappa_tau", "kappa_tau", "ln_kappa", "ln_Cm", "ln_Rm", "ln_Ra",
                "rail_gbar", "rail_dv_h", "rail_kappa_tau", "rail_Cm"]].round(4).to_string(index=False))
    print("bounds hit (of %d): kappa_tau %d, dv_h %d, gbar %d, Cm %d" % (len(full), full.rail_kappa_tau.sum(),
          full.rail_dv_h.sum(), full.rail_gbar.sum(), full.rail_Cm.sum()))
    print("absolute training residual / noise sigma: median %.2f" % (full.train_rmsd_abs_mV / full.noise_sigma_mV).median())

    ih = f[f.use_ih == True]                                            # noqa: E712
    print("\n== 2. voltages reached by the training steps; the truth's I_h conductance over the leak (twins with I_h)")
    print(ih[["specimen_id", "configuration", "v_rest_mV", "v_train_hi", "v_train_lo", "ih_over_leak_rest_true",
              "ih_over_leak_lo_true"]].round(3).to_string(index=False))
    print("deepest training trough below rest: median %.1f mV; I_h/leak at rest: median %.3f; at the deepest trough: median %.3f"
          % ((ih.v_rest_mV - ih.v_train_lo).median(), ih.ih_over_leak_rest_true.median(), ih.ih_over_leak_lo_true.median()))

    print("\n== 3. full twins: g_h at rest (gbar * m_inf(V_rest; dv_h)), split by the truth's I_h at the deepest trough")
    strong = full[full.ih_over_leak_lo_true >= 0.3]
    weak = full[full.ih_over_leak_lo_true < 0.3]
    print("I_h >= 0.3 x leak (%d): %s; |ln g_h at rest| %s; ln gbar %s; dv_h error %s" % (
        len(strong), strong.specimen_id.tolist(), strong.ln_gh_rest.abs().round(2).tolist(),
        strong.ln_gbar.round(2).tolist(), strong.d_dvh.round(1).tolist()))
    print("I_h <  0.3 x leak (%d): |ln g_h at rest| %s" % (len(weak), weak.ln_gh_rest.abs().round(2).tolist()))

    pf = f[f.configuration == "passive_fit"]
    print("\n== 4a. passive_fit twins (passive arm on traces with I_h): ln ratio of C_m, R_m, R_a")
    print(pf[["specimen_id", "cell_tau_s_above_window", "ln_Cm", "ln_Rm", "ln_Ra", "ih_over_leak_rest_true"]].round(3).to_string(index=False))
    print("|ln Cm| > ln 1.25 in %d of %d; median |ln Cm| %.3f" % ((pf.ln_Cm.abs() > np.log(1.25)).sum(), len(pf), pf.ln_Cm.abs().median()))
    af = f[f.configuration == "active_fit"]
    print("\n== 4b. active_fit twins (no I_h; six-parameter fit): gbar, its bound, fitted I_h at rest over the leak, C_m")
    print(af[["specimen_id", "gbar", "rail_gbar", "dv_h", "kappa_tau", "gh_rest_over_gpas", "ln_Cm", "ln_Rm", "rail_Cm"]].round(5).to_string(index=False))
    print("fitted I_h at rest / leak: max %.4f; |ln Cm| > ln 1.25 in %d of %d; Cm on a bound in %d" % (
        af.gh_rest_over_gpas.max(), (af.ln_Cm.abs() > np.log(1.25)).sum(), len(af), af.rail_Cm.sum()))
    st = pd.read_csv(os.path.join(d, "twins_status.csv"))
    hours = (st.build_seconds + st.fit_seconds) / 3600.0
    print("\n== 4c. status %s; hours per twin (build + fit): median %.1f, max %.1f; max by configuration %s" % (
        st.status.value_counts().to_dict(), hours.median(), hours.max(),
        st.assign(h=hours).groupby("configuration").h.max().round(1).to_dict()))

    print("\n== 5. the foot: max |gbar e^{-dv/k} m_inf(V; dv) / (gbar m_inf(V; 0)) - 1| over V")
    v_rest = float(full.v_rest_mV.median())
    for lo, hi_, name in ((v_rest - 11.0, v_rest, "median rest (%.1f mV) to 11 mV below" % v_rest), (-95.0, -60.0, "-95 to -60 mV")):
        v = np.linspace(lo, hi_, 500)
        print("  %-40s %s" % (name, "  ".join("dv=%+d: %.3f" % (dv, np.max(np.abs(np.exp(-dv / K_MV) * m_inf(v, dv) / m_inf(v, 0.0) - 1.0)))
                                                for dv in (-10, -5, 5, 10))))

    if len(argv) > 3:
        z = np.load(argv[2], allow_pickle=False)
        on, off, fs = 13500, 63500, 50000.0                     # the archive's Long Square step (50 kHz)
        print("\n== 6. the real cell: sag = (V_end - V_trough)/(V_base - V_trough) on its hyperpolarising steps (rest %s mV)" % argv[3])
        for k in range(int(z["n_sweeps"][0])):
            i_, v_ = z["i_%d" % k], z["v_%d" % k]
            amp = np.median(i_[on + 100: off - 100]) - np.median(i_[: on - 100])
            if amp > -5.0:
                continue
            w = int(0.001 * fs)
            vs_ = np.convolve(v_, np.ones(w) / w, mode="same")        # 1 ms boxcar before the minimum
            base, vmin, vend = v_[on - 5000: on].mean(), vs_[on: on + int(0.3 * fs)].min(), v_[off - 5000: off].mean()
            print("  %7.1f pA: trough %.2f mV (%.2f below base), sag %.3f" % (amp, vmin, base - vmin, (vend - vmin) / (base - vmin)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
