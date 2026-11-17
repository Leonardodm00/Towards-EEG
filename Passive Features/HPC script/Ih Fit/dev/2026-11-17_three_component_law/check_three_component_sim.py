#!/usr/bin/env python3
"""Stage 2 of the three-component check: the closed-form result tested by simulation. Standalone dev check, 2026-11-17.

For each nf1 cell, with the pipeline's own binding function (noise_fit.binding_function, M = M_BINDING windows, the
cell's own binding seed) on a layout REBUILT from the cell's JSON (see rebuilt_layout: group sizes, t_SS and tilts are
the cell's; the positions of the kept windows in their trains are not stored, so they are spread evenly):

  (a) q_sim at theta_hat(c), the run's two-component law      -> J_a, to compare with the run's stored J
      (tests the rebuilt layout: with the true positions (a) would reproduce the stored q_sim bit for bit);
  (b) q_sim at the three-component law of stage 1 (M3)         -> J_b
  (c) [--refine] (b) refined by scipy least_squares on the simulated residual, started at stage 1's law -> J_c

J = r^T Sigma^-1 r with r = q_sim - q_hat(c) and Sigma the run's stored covariance (simulated at theta_hat(c) under
the two-component law: an approximation for (b) and (c)).

The middle component is added by wrapping noise_law.simulate_traces (the one function every simulated trace of
Block 4 goes through): out + AR(1)(sigma_m, exp(-dt / tau_m)) driven by its own generator, re-seeded at every call of
the binding function (common random numbers). Nothing in the pipeline is changed.

Usage: python3 check_three_component_sim.py <dir with specimen_<id>.json> <stage-1 csv> <out.csv> [--refine] [--cells a,b]
"""
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize

CODE = Path(os.environ.get("TEEG_IH_FIT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(CODE))
import noise_fit as nf             # noqa: E402
import noise_law as nl             # noqa: E402
import noise_windows as nw         # noqa: E402

# ---- configuration --------------------------------------------------------------------------------------------
M_BINDING = 3000                  # as the run (D-056)
M_REFINE = 1000                   # binding size inside the refinement (c); J_c is re-evaluated at M_BINDING
N_W, PRE_N, N_LS, PERIOD, TRAIN = 10500, 500, 13250, 10025, 37     # samples / pulses, the archive's SS and LS geometry
SEED_MID = 20261117
WORKERS = 2
REFINE_MAX_NFEV = 6
REFINE_STEP = (1e-3, 1e-2, 2e-2, 2e-2, 1e-3, 1e-3)   # relative steps in (log sf, rho_f, log sm, log tau_m, log ss, log tau_s)
TAU_M_BOUNDS_S, TAU_S_BOUNDS_S = (4e-5, 1e-3), (1e-3, 10.0)
# ---------------------------------------------------------------------------------------------------------------

MID = {"sigma": 0.0, "tau": 3e-4, "rng": None}
_ORIG = nl.simulate_traces


def _traces_with_middle(law, dt, xi_fast, xi_slow, **kw):
    out = _ORIG(law, dt, xi_fast, xi_slow, **kw)
    if MID["sigma"] > 0.0:
        xi = MID["rng"].standard_normal(np.shape(xi_fast))
        out = out + nl._stationary_ar1(MID["sigma"], math.exp(-dt / MID["tau"]), -math.expm1(-2.0 * dt / MID["tau"]), xi)
    return out


nl.simulate_traces = _traces_with_middle


def q_sim3(th, layout, cfg, seed):
    """Binding function of the three-component law th (dict); sigma_m = 0 gives the pipeline's own."""
    MID.update(sigma=float(th.get("sigma_m", 0.0)), tau=float(th.get("tau_m", 3e-4)),
               rng=np.random.default_rng([SEED_MID, int(seed) % (2 ** 32)]))
    law = nl.NoiseLaw(th["sigma_f"], th["rho_f"], th["sigma_s"], th["tau_s"])
    return nf.binding_function(law, layout, cfg, seed)


def rebuilt_layout(rec):
    """A CellLayout from the JSON: the cell's group sizes, t_SS, tilts, LS sweeps; positions spread evenly."""
    w, cf = rec["windows"], rec["cell_fit"]
    fs = float(w["fs_Hz"])
    wc = nw.WindowConfig()
    sweeps = tuple(int(s) for s in cf["ss_tilt_mV_per_s"].keys())
    pos, dep, swn, sidx = [], [], [], []
    for g in w["start_groups"]:
        s, is_dep, K = int(g["sweep"]), bool(g["dep"]), int(g["K"])
        slots = np.arange(0, TRAIN, 2) if is_dep else np.arange(1, TRAIN, 2)      # 19 dep, 18 hyp, alternating from dep
        pick = slots[np.unique(np.round(np.linspace(0, slots.size - 1, K)).astype(int))]
        assert pick.size == K, (rec["specimen_id"], s, is_dep, K)
        pos += pick.tolist(); dep += [is_dep] * K; swn += [s] * K; sidx += [sweeps.index(s)] * K
    order = np.lexsort((np.array(pos), np.array(sidx)))
    ls_sw = np.array([int(s) for s in cf["ls_tilt_mV_per_s"].keys()], int)
    lay = nf.CellLayout(specimen_id=int(rec["specimen_id"]), fs=fs, dt=1.0 / fs, n_w=N_W, pre_n=PRE_N,
                        j0=PRE_N + int(round(w["t_ss_ms"] * 1e-3 * fs)), j_floor=nw.floor_index(PRE_N, fs, wc),
                        period_samples=PERIOD, train_length=TRAIN, sweep_numbers=sweeps,
                        sweep_index=np.array(sidx, int)[order], position=np.array(pos, int)[order],
                        polarity_is_dep=np.array(dep, bool)[order], sweep_number=np.array(swn, int)[order],
                        n_ls=N_LS, ls_sweep_number=ls_sw,
                        ss_tilt=np.array([cf["ss_tilt_mV_per_s"][str(s)] for s in sweeps], float),
                        ls_tilt=np.array([cf["ls_tilt_mV_per_s"][str(s)] for s in ls_sw], float))
    assert lay.K == w["K"] and lay.K_LS == w["K_LS"]
    return lay


def one_cell(args):
    path, th3, refine = args
    rec = json.load(open(path))
    cf = rec["cell_fit"]
    sid = int(rec["specimen_id"])
    lay = rebuilt_layout(rec)
    cfg = nf.NoiseFitConfig(M_binding=M_BINDING)
    seed = nf._child_seeds(cfg.seed, sid, 3)[0]
    q_hat, q_run = np.array(cf["q_hat"], float), np.array(cf["q_sim"], float)
    C = linalg.cholesky(np.array(cf["Sigma"], float), lower=True)
    sd = np.sqrt(np.diag(np.array(cf["Sigma"], float)))

    def J_of(q):
        z = linalg.solve_triangular(C, q - q_hat, lower=True)
        return float(z @ z)

    t0 = time.time()
    th2 = dict(sigma_f=cf["law"]["sigma_f_mV"], rho_f=cf["law"]["rho_f"], sigma_s=cf["law"]["sigma_s_mV"],
               tau_s=cf["law"]["tau_s_s"], sigma_m=0.0)
    qa = q_sim3(th2, lay, cfg, seed)
    qb = q_sim3(th3, lay, cfg, seed)
    row = dict(specimen_id=sid, J_run=float(cf["J"]), J_a=J_of(qa), J_b=J_of(qb),
               max_abs_z_a_vs_run=float(np.max(np.abs(qa - q_run) / sd)),
               z_ss_rho_1ms_run=float((q_run - q_hat)[3] / sd[3]), z_ss_rho_1ms_b=float((qb - q_hat)[3] / sd[3]),
               rel_ss_rho_1ms_b=float((qb - q_hat)[3] / q_hat[3]))
    zb = (qb - q_hat) / sd
    relb = (qb - q_hat) / q_hat
    if refine:
        cfg_r = nf.NoiseFitConfig(M_binding=M_REFINE)
        names = ("sigma_f", "rho_f", "sigma_m", "tau_m", "sigma_s", "tau_s")
        logs = (True, False, True, True, True, True)

        def to_th(u):
            return {n: (math.exp(x) if lg else float(x)) for n, x, lg in zip(names, u, logs)}

        u0 = np.array([math.log(th3[n]) if lg else th3[n] for n, lg in zip(names, logs)])
        lo = np.array([math.log(1e-4), 1e-3, math.log(1e-4), math.log(TAU_M_BOUNDS_S[0]), math.log(1e-4), math.log(TAU_S_BOUNDS_S[0])])
        hi = np.array([math.log(10.0), 0.999, math.log(10.0), math.log(TAU_M_BOUNDS_S[1]), math.log(10.0), math.log(TAU_S_BOUNDS_S[1])])
        u0 = np.clip(u0, lo + 1e-4, hi - 1e-4)
        res = optimize.least_squares(
            lambda u: linalg.solve_triangular(C, q_sim3(to_th(u), lay, cfg_r, seed) - q_hat, lower=True),
            u0, bounds=(lo, hi), method="trf", diff_step=list(REFINE_STEP), ftol=1e-4, xtol=1e-3, gtol=1e-8,
            max_nfev=REFINE_MAX_NFEV)
        thc = to_th(res.x)
        qc = q_sim3(thc, lay, cfg, seed)
        zc, relc = (qc - q_hat) / sd, (qc - q_hat) / q_hat
        row.update(J_c=J_of(qc), J_c_at_M_refine=2.0 * float(res.cost), nfev=int(res.nfev),
                   c_sigma_f=thc["sigma_f"], c_rho_f=thc["rho_f"], c_sigma_m=thc["sigma_m"], c_tau_m_ms=1e3 * thc["tau_m"],
                   c_sigma_s=thc["sigma_s"], c_tau_s_ms=1e3 * thc["tau_s"], z_ss_rho_1ms_c=float(zc[3]),
                   rel_ss_rho_1ms_c=float(relc[3]), max_abs_z_c=float(np.max(np.abs(zc))),
                   med_abs_rel_c=float(np.median(np.abs(relc))))
    row["seconds"] = time.time() - t0
    print("%d  J run %.1f | (a) rebuilt layout %.1f | (b) M3 closed-form law %.1f%s  (%.0f s)"
          % (sid, row["J_run"], row["J_a"], row["J_b"], (" | (c) refined %.1f" % row["J_c"]) if refine else "",
             row["seconds"]), flush=True)
    return row, zb, relb


def main():
    d, cf_csv, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    refine = "--refine" in sys.argv
    only = None
    if "--cells" in sys.argv:
        only = {int(x) for x in sys.argv[sys.argv.index("--cells") + 1].split(",")}
    S1 = pd.read_csv(cf_csv).set_index("specimen_id")
    jobs = []
    for f in sorted(glob.glob(str(d / "specimen_*.json"))):
        sid = int(Path(f).stem.split("_")[1])
        if only and sid not in only:
            continue
        r = S1.loc[sid]
        th3 = dict(sigma_f=float(r.sigma_f), rho_f=float(r.rho_f), sigma_m=float(r.sigma_m), tau_m=float(r.tau_m_ms) / 1e3,
                   sigma_s=float(r.sigma_s), tau_s=float(r.tau_s_ms) / 1e3)
        if refine and th3["tau_s"] >= 5.0:
            # stage 1 ran to the tau_s bound (a ridge of the closed form): start the refinement at the run's slow part
            th3.update(sigma_s=float(r.run_sigma_s), tau_s=max(float(r.run_tau_s_ms) / 1e3, 2e-3))
        jobs.append((f, th3, refine))
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(one_cell, jobs))
    T = pd.DataFrame([r[0] for r in res])
    T.to_csv(out_csv, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print()
    print(T.to_string(index=False, float_format=lambda x: "%.4g" % x))
    print()
    print("medians: J run %.1f | (a) %.1f | (b) %.1f%s" % (T.J_run.median(), T.J_a.median(), T.J_b.median(),
                                                           (" | (c) %.1f" % T.J_c.median()) if refine else ""))
    print("(a)/run: median %.2f, range %.2f-%.2f" % ((T.J_a / T.J_run).median(), (T.J_a / T.J_run).min(), (T.J_a / T.J_run).max()))
    print("(b)/(a): median %.2f, range %.2f-%.2f; (b) < (a) in %d of %d" % (
        (T.J_b / T.J_a).median(), (T.J_b / T.J_a).min(), (T.J_b / T.J_a).max(), (T.J_b < T.J_a).sum(), len(T)))
    if refine:
        print("(c)/(a): median %.2f, range %.2f-%.2f; (c) < 21.7 (chi2_9 99 %%) in %d of %d" % (
            (T.J_c / T.J_a).median(), (T.J_c / T.J_a).min(), (T.J_c / T.J_a).max(), (T.J_c < 21.67).sum(), len(T)))


if __name__ == "__main__":
    main()
