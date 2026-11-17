#!/usr/bin/env python3
"""Does a third noise component (a "middle" one, time scale <= 1 ms) remove the misfit of the nf1 run?
Stage 1: closed form, no simulation. Standalone dev check (not pipeline code), 2026-11-17.

Laws compared, each fitted per cell to the cell's stored statistic vector q_hat(c) (15 statistics, Block 3):
  M2   the pipeline's law (SPEC 2.1): fast AR(1) (sigma_f, rho_f) + slow OU (sigma_s, tau_s)            4 parameters
  M3   M2 + a middle OU (sigma_m, tau_m), tau_m in [TAU_M_MIN, 1 ms], tau_s in [1 ms, 10 s]             6 parameters
  M3w  M3 with the fast part white (rho_f fixed at RHO_WHITE)                                          5 parameters
The three components are independent, stationary and Gaussian, so the autocovariance of the sum is the sum:
  gamma(k) = sigma_f^2 rho_f^k + sigma_m^2 exp(-k dt / tau_m) + sigma_s^2 exp(-k dt / tau_s)   (mV^2).

Binding function used here (an approximation; the pipeline's is a simulation):
  q_cf(theta)      closed-form statistics (noise_law.expected_s0 / expected_sl, summed over the components, which is
                   exact because E[S_0] and E[S_L] are linear in gamma): sqrt(E[S_0]/(n-1)), E[S_L]/E[S_0]
  q_approx(theta)  q_cf(theta) corrected by the cell's own stored simulation at theta_hat(c) (the M2 fit of the run):
                   an SD statistic  -> q_cf(theta) * q_sim(theta_hat) / q_cf(theta_hat)
                   an autocorrelation -> q_cf(theta) + q_sim(theta_hat) - q_cf(theta_hat)
                   so that q_approx(theta_hat) = q_sim(theta_hat) exactly (M3 with sigma_m -> 0 contains M2).
Cost: J(theta) = r^T Sigma^-1 r, r = q_approx(theta) - q_hat, Sigma the run's final simulated covariance (at theta_hat).
Solver: scipy.optimize.least_squares (trf) on the whitened residual, several starts, the lowest J kept.

Usage: python3 check_three_component_cf.py <dir with specimen_<id>.json> <out.csv>
       (TEEG_IH_FIT or the folder two levels up must hold the Ih Fit code)
"""
import glob
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize
from scipy.stats import chi2

CODE = Path(os.environ.get("TEEG_IH_FIT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(CODE))
import noise_cell_stats as ncs     # noqa: E402
import noise_law as nl             # noqa: E402

# ---- configuration --------------------------------------------------------------------------------------------
N_W, PRE_N, N_LS = 10500, 500, 13250          # samples: stored SS window, its pre-pulse part, LS pre-window (50 kHz)
LS_FIRST_MS, LS_LAST_MS = 50.0, 132.5         # WindowConfig defaults (SPEC 2.2)
TAU_M_MIN_S, TAU_SPLIT_S, TAU_S_MAX_S = 4e-5, 1e-3, 10.0   # middle in [2 samples, 1 ms]; slow in [1 ms, 10 s]
LOG_SIGMA = (math.log(1e-4), math.log(10.0))  # mV, as Block 4
RHO = (1e-3, 0.999)
RHO_WHITE = 1e-3
SD_IDX = np.array([0, 5, 9, 10])              # the SD statistics among the 15
STARTS_TAU_M_S = (1e-4, 6e-4)
STARTS_SIGMA_M_REL = (0.5,)                   # sigma_m start, relative to sigma_s of theta_hat
MAX_NFEV = 80
WORKERS = 2
# ---------------------------------------------------------------------------------------------------------------


def components(theta):
    """theta = dict(sigma_f, rho_f, sigma_m, tau_m, sigma_s, tau_s) -> list of one-component NoiseLaw objects."""
    out = [nl.NoiseLaw(theta["sigma_f"], theta["rho_f"], theta["sigma_s"], theta["tau_s"])]
    if theta.get("sigma_m", 0.0) > 0.0:
        out.append(nl.NoiseLaw(0.0, 0.5, theta["sigma_m"], theta["tau_m"]))
    return out


def q_cf(theta, dt, fs, m):
    """Closed-form statistic vector (15,) of the sum of the components, in Block 3's order."""
    comps = components(theta)

    cache = {}

    def s0(n):
        if n not in cache:
            cache[n] = sum(nl.expected_s0(c, dt, n) for c in comps)
        return cache[n]

    def acf(n, lags_ms):
        lags = ncs.lags_in_samples(lags_ms, fs)
        return sum(nl.expected_sl(c, dt, n, lags) for c in comps) / s0(n)

    n1, n2 = int(round(LS_FIRST_MS * 1e-3 * fs)), int(round(LS_LAST_MS * 1e-3 * fs))
    q = [math.sqrt(s0(m) / (m - 1))] + list(acf(m, ncs.SS_LAGS_MS))
    q += [math.sqrt(s0(N_LS) / (N_LS - 1))] + list(acf(N_LS, ncs.LS_FULL_LAGS_MS))
    q += [math.sqrt(s0(n1) / (n1 - 1)), math.sqrt(s0(n2) / (n2 - 1))] + list(acf(n2, ncs.LS_LAST_LAGS_MS))
    return np.array(q, float)


class Cell:
    def __init__(self, rec):
        cf, w = rec["cell_fit"], rec["windows"]
        self.sid = rec["specimen_id"]
        self.fs = float(w["fs_Hz"])
        self.dt = 1.0 / self.fs
        self.m = N_W - (PRE_N + int(round(w["t_ss_ms"] * 1e-3 * self.fs)))
        self.q_hat = np.array(cf["q_hat"], float)
        self.q_sim = np.array(cf["q_sim"], float)
        self.Sigma = np.array(cf["Sigma"], float)
        self.C = linalg.cholesky(self.Sigma, lower=True)
        self.J_run = float(cf["J"])
        law = cf["law"]
        self.theta_hat = dict(sigma_f=law["sigma_f_mV"], rho_f=law["rho_f"], sigma_m=0.0, tau_m=3e-4,
                              sigma_s=law["sigma_s_mV"], tau_s=law["tau_s_s"])
        self.q_cf_hat = q_cf(self.theta_hat, self.dt, self.fs, self.m)

    def q_approx(self, theta):
        q = q_cf(theta, self.dt, self.fs, self.m)
        out = q + (self.q_sim - self.q_cf_hat)
        out[SD_IDX] = q[SD_IDX] * self.q_sim[SD_IDX] / self.q_cf_hat[SD_IDX]
        return out

    def J_of(self, theta):
        z = linalg.solve_triangular(self.C, self.q_approx(theta) - self.q_hat, lower=True)
        return float(z @ z)


MODELS = {
    "M2": dict(names=("log_sigma_f", "rho_f", "log_sigma_s", "log_tau_s"),
               lo=(LOG_SIGMA[0], RHO[0], LOG_SIGMA[0], math.log(1e-4)), hi=(LOG_SIGMA[1], RHO[1], LOG_SIGMA[1], math.log(TAU_S_MAX_S))),
    "M3": dict(names=("log_sigma_f", "rho_f", "log_sigma_m", "log_tau_m", "log_sigma_s", "log_tau_s"),
               lo=(LOG_SIGMA[0], RHO[0], LOG_SIGMA[0], math.log(TAU_M_MIN_S), LOG_SIGMA[0], math.log(TAU_SPLIT_S)),
               hi=(LOG_SIGMA[1], RHO[1], LOG_SIGMA[1], math.log(TAU_SPLIT_S), LOG_SIGMA[1], math.log(TAU_S_MAX_S))),
    "M3w": dict(names=("log_sigma_f", "log_sigma_m", "log_tau_m", "log_sigma_s", "log_tau_s"),
                lo=(LOG_SIGMA[0], LOG_SIGMA[0], math.log(TAU_M_MIN_S), LOG_SIGMA[0], math.log(TAU_SPLIT_S)),
                hi=(LOG_SIGMA[1], LOG_SIGMA[1], math.log(TAU_SPLIT_S), LOG_SIGMA[1], math.log(TAU_S_MAX_S))),
}


def to_theta(model, u):
    d = dict(zip(MODELS[model]["names"], u))
    return dict(sigma_f=math.exp(d["log_sigma_f"]), rho_f=float(d.get("rho_f", RHO_WHITE)),
                sigma_m=math.exp(d["log_sigma_m"]) if "log_sigma_m" in d else 0.0,
                tau_m=math.exp(d["log_tau_m"]) if "log_tau_m" in d else 3e-4,
                sigma_s=math.exp(d["log_sigma_s"]), tau_s=math.exp(d["log_tau_s"]))


def from_theta(model, th):
    full = dict(log_sigma_f=math.log(th["sigma_f"]), rho_f=th["rho_f"], log_sigma_m=math.log(max(th["sigma_m"], 1e-4)),
                log_tau_m=math.log(th["tau_m"]), log_sigma_s=math.log(th["sigma_s"]), log_tau_s=math.log(th["tau_s"]))
    return np.array([full[n] for n in MODELS[model]["names"]], float)


def fit(cell, model):
    lo, hi = np.array(MODELS[model]["lo"]), np.array(MODELS[model]["hi"])
    starts = []
    th0 = dict(cell.theta_hat)
    if model == "M2":
        starts = [th0, dict(th0, tau_s=th0["tau_s"] * 2)]
    else:
        for tm in STARTS_TAU_M_S:
            for rel in STARTS_SIGMA_M_REL:
                for ts in (max(th0["tau_s"], 2e-3), max(3.0 * th0["tau_s"], 2e-2)):
                    starts.append(dict(th0, sigma_m=rel * th0["sigma_s"], tau_m=tm, tau_s=ts))
    best = None
    for th in starts:
        u0 = np.clip(from_theta(model, th), lo + 1e-6, hi - 1e-6)
        res = optimize.least_squares(
            lambda u: linalg.solve_triangular(cell.C, cell.q_approx(to_theta(model, u)) - cell.q_hat, lower=True),
            u0, bounds=(lo, hi), method="trf", x_scale="jac", ftol=1e-8, xtol=1e-8, gtol=1e-10, max_nfev=MAX_NFEV)
        J = 2.0 * float(res.cost)
        if best is None or J < best[0]:
            best = (J, res.x)
    J, u = best
    span = hi - lo
    at = [n for n, x, a, b, s in zip(MODELS[model]["names"], u, lo, hi, span) if x <= a + 1e-3 * s or x >= b - 1e-3 * s]
    return J, to_theta(model, u), at


def one_cell(path):
    c = Cell(json.load(open(path)))
    assert abs(c.J_of(c.theta_hat) - c.J_run) <= 1e-6 * c.J_run      # q_approx(theta_hat) = q_sim(theta_hat)
    J2, th2, at2 = fit(c, "M2")
    J3, th3, at3 = fit(c, "M3")
    J3w, th3w, at3w = fit(c, "M3w")
    sd = np.sqrt(np.diag(c.Sigma))
    r2 = (c.q_approx(th2) - c.q_hat) / sd
    r3 = (c.q_approx(th3) - c.q_hat) / sd
    row = (dict(specimen_id=c.sid, J_run=c.J_run, J2=J2, J3=J3, J3w=J3w,
                     p3=chi2.sf(J3, 9), p3w=chi2.sf(J3w, 10),
                     sigma_f=th3["sigma_f"], rho_f=th3["rho_f"], sigma_m=th3["sigma_m"], tau_m_ms=1e3 * th3["tau_m"],
                     sigma_s=th3["sigma_s"], tau_s_ms=1e3 * th3["tau_s"], at_bound3=",".join(at3),
                     run_sigma_s=c.theta_hat["sigma_s"], run_tau_s_ms=1e3 * c.theta_hat["tau_s"],
                     w_sigma_f=th3w["sigma_f"], w_sigma_m=th3w["sigma_m"], w_tau_m_ms=1e3 * th3w["tau_m"],
                     w_sigma_s=th3w["sigma_s"], w_tau_s_ms=1e3 * th3w["tau_s"], at_bound3w=",".join(at3w)))
    print("%d  J run %.1f | M2 %.1f | M3 %.1f | M3w %.1f | tau_m %.3g ms sigma_m %.3g mV %s"
          % (c.sid, c.J_run, J2, J3, J3w, 1e3 * th3["tau_m"], th3["sigma_m"], at3), flush=True)
    return row, r2, r3


def main():
    d, out_csv = Path(sys.argv[1]), Path(sys.argv[2])
    labels = None
    rows, R2, R3 = [], [], []
    files = sorted(glob.glob(str(d / "specimen_*.json")))
    labels = json.load(open(files[0]))["cell_fit"]["labels_kept"]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(one_cell, files))
    for row, r2, r3 in results:
        rows.append(row)
        R2.append(r2)
        R3.append(r3)
    T = pd.DataFrame(rows)
    T.to_csv(out_csv, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print()
    print(T[["specimen_id", "J_run", "J2", "J3", "J3w", "p3", "sigma_f", "rho_f", "sigma_m", "tau_m_ms", "sigma_s",
             "tau_s_ms", "run_sigma_s", "run_tau_s_ms", "at_bound3"]].to_string(index=False, float_format=lambda x: "%.4g" % x))
    print()
    print("medians: J_run %.1f | J2 (closed form, refitted) %.1f | J3 %.1f | J3w %.1f" % (T.J_run.median(), T.J2.median(), T.J3.median(), T.J3w.median()))
    print("J3 / J2: median %.2f, range %.2f-%.2f ; J2 - J3 > 9.21 (chi2_2 99 %%) in %d of %d cells" % (
        (T.J3 / T.J2).median(), (T.J3 / T.J2).min(), (T.J3 / T.J2).max(), ((T.J2 - T.J3) > 9.21).sum(), len(T)))
    print("J3 below chi2_9's 99 %% point (21.7) in %d cells; J3w below chi2_10's (23.2) in %d; J2 below chi2_11's (24.7) in %d" % (
        (T.J3 < 21.67).sum(), (T.J3w < 23.21).sum(), (T.J2 < 24.72).sum()))
    print("tau_m at its upper bound (1 ms) in %d cells, at its lower in %d" % (
        (T.tau_m_ms > 0.99).sum(), (T.tau_m_ms < 1e3 * TAU_M_MIN_S * 1.01).sum()))
    R2, R3 = np.array(R2), np.array(R3)
    print()
    print("marginal standardised residuals z (law minus data, over the law's simulated SD), medians of |z| and cells with |z| > 3:")
    print(pd.DataFrame(dict(stat=labels, M2_abs_med=np.median(np.abs(R2), 0), M3_abs_med=np.median(np.abs(R3), 0),
                            M2_med=np.median(R2, 0), M3_med=np.median(R3, 0),
                            M2_gt3=(np.abs(R2) > 3).sum(0), M3_gt3=(np.abs(R3) > 3).sum(0))).to_string(index=False, float_format=lambda x: "%.2f" % x))


if __name__ == "__main__":
    main()
