"""Covariance of the statistics: Monte Carlo error vs condition number; what the correlations do (check, 2026-10-14).

Reads ../results/weights.npz (check_weights.py: ten LS statistics, medians over K = 6 windows, only tau_s
free at theta_sim; Sigma estimated there from 3000 simulated cells and treated here as the truth).

A. Monte Carlo error of a simulated Sigma_hat vs kappa. Draw R vectors from N(0, Sigma) (a Gaussian stand-in
   for R simulated cells' medians), Sigma_hat = their sample covariance. Compare with an error of the same
   size on each entry but independent across entries: entry jk drawn N(0, sd_jk^2) with
   sd_jk^2 = (S_jj S_kk + S_jk^2) / (R - 1), the Gaussian sampling variance of entry jk (textbook, from memory).
   Reported: relative 2-norm error of the inverse; the range over all r != 0 of the cost ratio
   r^T Sigma_hat^-1 r / r^T Sigma^-1 r (= 1 / eigenvalues of Sigma^-1/2 Sigma_hat Sigma^-1/2);
   kappa * eps with eps = ||Sigma_hat - Sigma||_2 / ||Sigma||_2.
B. Linearised fit (p = 1 parameter): G = dq/dtau_s at tau_s0 from the cubic spline (in log tau) of the
   binding function; asymptotic SD of tau_hat for W = Sigma^-1, diag(1/SE^2), diag(1/omega^2) (sandwich
   formula), and the 'naive' SD of the diagonal weighting, (G^T W G)^-1/2, i.e. as if the ten statistics
   were independent; share of the information G^T Sigma^-1 G per eigen-direction of the correlation matrix.
C. Model error: noise-free input q(tau_s0) + delta_j e_j, delta_j = 1 SE_j in ONE statistic j at a time
   (plus one pair: the 0.02 ms autocorrelation in both windows); tau_hat refitted (nonlinear, on the fine
   grid) under each weighting; shift = tau_hat(biased input) - tau_hat(unbiased input).

Predictions, fixed BEFORE the first run:
  P10 (A) the relative 2-norm error of Sigma_hat^-1 has 95th percentile <= 3 sqrt(m / R) (0.17 at R = 3000,
      0.39 at R = 600), although kappa * eps has median > 10 (the worst-case bound says nothing).
  P11 (A) with independent entry errors of the same per-entry SD, the median relative error of the inverse
      exceeds 1 at both R (a perturbed matrix that is not positive definite counts as > 1).
  P12 (B) each linearised sandwich SD is within +-25 % of the robust SD of the simulated fits reported in
      check_weights.log (full 7.57, diag 15.53, hand 16.98 ms).
  P13 (B) the naive SD of the diagonal weighting is below 1/1.5 of its sandwich SD.
  P14 (C) the root-sum-square over the ten single-statistic shifts is larger under Sigma^-1 than under
      diag(1/SE^2) (linearised, this follows from the Cauchy-Schwarz inequality).
[Result of the first run, 2026-10-14: P10, P11, P13, P14 and P12 for diag and hand held; P12 FAILED for
Sigma^-1: linearised SD 5.07 ms against the simulated robust SD 7.57 ms (ratio 0.67). Added after that run,
to locate the cause, with its prediction fixed before the second run:
  D. Gaussian inputs q_hat ~ N(q(tau_s0), Sigma), R_gauss draws, fitted with the same nonlinear fit.
  P15 the robust SD of tau_hat under Sigma^-1 with Gaussian inputs is within +-10 % of the linearised SD.
      If it holds, the binding function's curvature is not the cause, and the gap to 7.57 ms comes from the
      simulated medians themselves (their distribution, or their centre differing from q(tau_s0)).
Result of the second run: P15 FAILED too (robust SD 6.38 ms with Gaussian inputs, ratio 1.26). So the
linearised model is optimistic for Sigma^-1 on both counts: about half of the gap 5.07 -> 7.57 ms appears
with Gaussian inputs (curvature of the binding function, or error in its spline derivative, at the scale of
the thin directions), the rest only with the simulated medians. Consequence: no linearised number for
Sigma^-1 (SD, information shares) is quoted as a result; the diagonal and hand-set weightings, for which
P12 held and the Gaussian-input fit agrees (15.95 vs 15.97 ms), are unaffected.
From the third run the script also prints, without prediction, kappa(Sigma) with the SD statistics in
microvolts instead of millivolts, and the largest correlations to three decimals (part E).
Added before the fourth run (the comparison in A rests on the per-entry SD formula, written from memory):
  P16 (A) the SD of each entry of Sigma_hat over the n_rep replicates matches sd_jk within +-20 % for every
      entry, at both R. Part D is computed in chunks from this run on (same numbers, less memory).]
Added before the fifth run (two invariances the answer states; exact algebra, checked numerically):
  P17 (A) the eigenvalues of Sigma^-1/2 Sigma_hat Sigma^-1/2 equal those of the sample covariance of the
      same standard-normal draws taken with Sigma = identity (kappa = 1), to 1e-6, in every replicate:
      the Monte Carlo distortion of the cost does not depend on kappa.
  P18 (E) tau_hat under W = Sigma^-1 is the same grid point with the SD statistics in microvolts as in
      millivolts, for all eleven inputs of part C.
Notation: m here is the number of statistics (written d in the answer of 2026-10-14, where m is the index
of a statistic, as in the topic document).
Measured without prediction: information shares; loadings of the thinnest and of the widest direction;
the largest single-statistic shift per weighting; the minimised cost under Sigma^-1 for each biased input.

Run (after check_weights.py):  python check_cov_structure.py     (about 15 s)
"""
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

CONFIG = dict(
    R_list=(600, 3000), n_rep=300, seed=2026101420,       # part A
    tau0=0.03974, n_fine=2000,                             # s; fine grid as in check_weights.py
    bias_in_se=1.0,                                        # part C
    R_gauss=3000, seed_gauss=2026101421,                   # part D
    emp_rsd_ms=dict(full=7.57, diag=15.53, hand=16.98),    # check_weights.log
)
RES = Path(__file__).resolve().parent.parent / "results" / "weights.npz"


def load(path):
    d = np.load(path)
    se, C = d["se"], d["corr"]
    return dict(names=[str(s) for s in d["names"]], se=se, C=C, Sigma=C * np.outer(se, se),
                omega=d["omega"], tau_grid=d["tau_grid"], Qg=d["Qg"])


# ---------------- A: Monte Carlo error vs condition number ----------------
def part_a(D, cfg, rng):
    S, m = D["Sigma"], len(D["se"])
    lam, U = np.linalg.eigh(S)
    S_mhalf = (U / np.sqrt(lam)) @ U.T                  # Sigma^-1/2
    Sinv, nS = np.linalg.inv(S), np.linalg.norm(S, 2)
    nSinv, kappa, L = np.linalg.norm(Sinv, 2), lam.max() / lam.min(), np.linalg.cholesky(S)
    dg = np.diag(S)
    ok = {}
    for R in cfg["R_list"]:
        sd_entry = np.sqrt((np.outer(dg, dg) + S**2) / (R - 1))
        rel_s, keps, lo, hi, rel_i, npd, Shs, dev_I = [], [], [], [], [], 0, [], 0.0
        for _ in range(cfg["n_rep"]):
            Zd = rng.standard_normal((R, m))
            Sh = np.cov(Zd @ L.T, rowvar=False)
            Shs.append(Sh)
            keps.append(kappa * np.linalg.norm(Sh - S, 2) / nS)
            rel_s.append(np.linalg.norm(np.linalg.inv(Sh) - Sinv, 2) / nSinv)
            ev = np.linalg.eigvalsh(S_mhalf @ Sh @ S_mhalf)
            lo.append(1 / ev.max()); hi.append(1 / ev.min())
            dev_I = max(dev_I, np.max(np.abs(ev - np.linalg.eigvalsh(np.cov(Zd, rowvar=False)))))
            Z = rng.standard_normal((m, m)); Z = np.triu(Z) + np.triu(Z, 1).T
            A = S + Z * sd_entry
            if np.linalg.eigvalsh(A).min() <= 0:
                npd += 1; rel_i.append(np.inf)
            else:
                rel_i.append(np.linalg.norm(np.linalg.inv(A) - Sinv, 2) / nSinv)
        rel_s, keps, rel_i = map(np.array, (rel_s, keps, rel_i))
        tol = 3 * np.sqrt(m / R)
        ok[f"P10_R{R}"] = bool(np.quantile(rel_s, 0.95) <= tol and np.median(keps) > 10)
        ok[f"P11_R{R}"] = bool(np.median(rel_i) > 1)
        iu = np.triu_indices(m)
        ratio = (np.std(np.array(Shs), axis=0, ddof=1) / sd_entry)[iu]
        ok[f"P16_R{R}"] = bool(np.all(np.abs(ratio - 1) <= 0.20))
        ok[f"P17_R{R}"] = bool(dev_I < 1e-6)
        print("  R = %4d: eigenvalues of Sigma^-1/2 Sigma_hat Sigma^-1/2 vs same draws with Sigma = I: max |diff| %.1e"
              % (R, dev_I))
        print("  R = %4d: per-entry SD of Sigma_hat over replicates / formula sd_jk: median %.3f, range [%.3f, %.3f]"
              % (R, np.median(ratio), ratio.min(), ratio.max()))
        print("  R = %4d: sampling error -> rel. error of inverse median %.3f, 95%% %.3f (tol %.2f); "
              "kappa*eps median %.0f" % (R, np.median(rel_s), np.quantile(rel_s, 0.95), tol, np.median(keps)))
        print("            cost ratio r'Sh^-1 r / r'S^-1 r over all r: 2.5%% of minima %.3f, 97.5%% of maxima %.3f"
              % (np.quantile(lo, 0.025), np.quantile(hi, 0.975)))
        fin = rel_i[np.isfinite(rel_i)]
        print("            independent entry errors, same per-entry SD -> not positive definite %d/%d; "
              "rel. error of inverse median %s (finite cases: median %.2f)"
              % (npd, cfg["n_rep"], "inf" if np.isinf(np.median(rel_i)) else "%.2f" % np.median(rel_i),
                 np.median(fin) if fin.size else np.nan))
    return ok


# ---------------- B: linearised fit ----------------
def binding_spline(D):
    return CubicSpline(np.log(D["tau_grid"]), D["Qg"], axis=0)


def part_b(D, cfg):
    spl, tau0, S = binding_spline(D), cfg["tau0"], D["Sigma"]
    G = spl(np.log(tau0), 1) / tau0                       # dq/dtau, per s
    Sinv = np.linalg.inv(S)

    def sandwich(W):
        a = G @ W @ G
        return np.sqrt(G @ W @ S @ W @ G) / a

    W = {"full": Sinv, "diag": np.diag(1 / D["se"]**2), "hand": np.diag(1 / D["omega"]**2)}
    sd = {k: 1e3 * sandwich(w) for k, w in W.items()}
    print("  G = dq/dtau_s at tau_s0 (per ms): " + ", ".join("%s %+.2e" % (n, 1e-3 * g) for n, g in zip(D["names"], G)))
    naive = 1e3 / np.sqrt(G @ W["diag"] @ G)
    ok = {}
    for k in ("full", "diag", "hand"):
        e = cfg["emp_rsd_ms"][k]
        ok[f"P12_{k}"] = bool(abs(sd[k] / e - 1) <= 0.25)
        print("  %-4s linearised SD %.2f ms vs simulated robust SD %.2f ms (ratio %.2f)" % (k, sd[k], e, sd[k] / e))
    ok["P13"] = bool(naive < sd["diag"] / 1.5)
    print("  diag naive SD (statistics treated as independent) %.2f ms vs its sandwich SD %.2f ms (ratio %.2f)"
          % (naive, sd["diag"], naive / sd["diag"]))
    mu, V = np.linalg.eigh(D["C"])                         # ascending
    Gt = G / D["se"]                                       # standardised sensitivities
    proj2 = (V.T @ Gt)**2
    info = proj2 / mu
    print("  eigen-directions of the correlation matrix (widest first): mu, share of optimal information, "
          "share of the diagonal weighting's 'information'")
    for i in range(len(mu) - 1, -1, -1):
        print("    mu = %8.4f   %6.1f %%   %6.1f %%" % (mu[i], 100 * info[i] / info.sum(), 100 * proj2[i] / proj2.sum()))
    for lab, i in (("widest", len(mu) - 1), ("thinnest", 0)):
        v = V[:, i] * np.sign(V[np.argmax(abs(V[:, i])), i])
        print("  %-8s direction loadings: %s" % (lab, ", ".join("%s %+.2f" % (n, x) for n, x in zip(D["names"], v)
                                                              if abs(x) >= 0.15)))
    return ok, G, W, sd["full"]


# ---------------- C: model error ----------------
def part_c(D, cfg, W):
    spl, tau0, se = binding_spline(D), cfg["tau0"], D["se"]
    tau_fine = np.geomspace(D["tau_grid"][0], D["tau_grid"][-1], cfg["n_fine"])
    Qf, q0 = spl(np.log(tau_fine)), spl(np.log(tau0))

    def fit(q, w):
        r = Qf - q
        J = np.einsum("gi,ij,gj->g", r, w, r)
        k = J.argmin()
        return tau_fine[k], J[k], k in (0, len(tau_fine) - 1)

    base = {k: fit(q0, w)[0] for k, w in W.items()}
    print("  unbiased input: tau_hat " + ", ".join("%s %.2f ms" % (k, 1e3 * v) for k, v in base.items()))
    inputs = [(D["names"][j], np.eye(len(se))[j] * cfg["bias_in_se"] * se[j]) for j in range(len(se))]
    pair = np.zeros(len(se)); pair[[1, 6]] = cfg["bias_in_se"] * se[[1, 6]]
    inputs.append(("pair acf .02", pair))
    shifts = {k: [] for k in W}
    for lab, delta in inputs:
        row = []
        for k, w in W.items():
            t, J, edge = fit(q0 + delta, w)
            shifts[k].append(1e3 * (t - base[k]))
            row.append("%s %+7.2f ms%s" % (k, shifts[k][-1], " (EDGE)" if edge else ""))
        Jf = fit(q0 + delta, W["full"])[1]
        print("    +1 SE in %-13s: %s   | min cost under Sigma^-1 %.1f" % (lab, "  ".join(row), Jf))
    rss = {k: np.sqrt(np.sum(np.array(v[:-1])**2)) for k, v in shifts.items()}   # the ten singles only
    mx = {k: np.max(np.abs(v[:-1])) for k, v in shifts.items()}
    print("  ten single-statistic biases: root-sum-square shift " + ", ".join("%s %.2f" % kv for kv in rss.items())
          + " ms; largest |shift| " + ", ".join("%s %.2f" % kv for kv in mx.items()) + " ms")
    return {"P14": bool(rss["full"] > rss["diag"])}


# ---------------- D: is the P12 gap for Sigma^-1 the binding function's curvature? ----------------
def part_d(D, cfg, W, sd_lin_full_ms):
    spl, tau0 = binding_spline(D), cfg["tau0"]
    tau_fine = np.geomspace(D["tau_grid"][0], D["tau_grid"][-1], cfg["n_fine"])
    Qf, q0 = spl(np.log(tau_fine)), spl(np.log(tau0))
    rng = np.random.default_rng(cfg["seed_gauss"])
    qh = q0 + rng.standard_normal((cfg["R_gauss"], len(q0))) @ np.linalg.cholesky(D["Sigma"]).T
    ok = {}
    for k, w in W.items():
        th = []
        for start in range(0, len(qh), 250):                   # chunks: about 40 MB per step
            r = Qf[None, :, :] - qh[start:start + 250, None, :]
            th.append(tau_fine[np.einsum("rgi,ij,rgj->rg", r, w, r).argmin(axis=1)])
        th = 1e3 * np.concatenate(th)
        q10, q50, q90 = np.quantile(th, [0.1, 0.5, 0.9])
        rsd = (q90 - q10) / 2.563
        print("  %-4s Gaussian inputs: median %.2f ms, robust SD %.2f ms, SD %.2f ms" % (k, q50, rsd, th.std(ddof=1)))
        if k == "full":
            ok["P15"] = bool(abs(rsd / sd_lin_full_ms - 1) <= 0.10)
            print("  P15 robust SD %.2f vs linearised %.2f ms (ratio %.2f)" % (rsd, sd_lin_full_ms, rsd / sd_lin_full_ms))
    return ok


# ---------------- E: units and correlations (measured, no prediction) ----------------
def part_e(D, cfg):
    is_sd = np.array([n.startswith("sd") for n in D["names"]])
    scale = np.where(is_sd, 1e3, 1.0)                     # mV -> microvolt for the SD statistics
    S_uv = D["Sigma"] * np.outer(scale, scale)
    print("  kappa(Sigma): SDs in mV %.3g, SDs in microvolt %.3g; kappa(correlation matrix) %.3g (unit-free)"
          % (np.linalg.cond(D["Sigma"]), np.linalg.cond(S_uv), np.linalg.cond(D["C"])))
    C, n = D["C"], D["names"]
    pairs = sorted(((abs(C[i, j]), i, j) for i in range(len(n)) for j in range(i + 1, len(n))), reverse=True)
    print("  largest correlations: " + ", ".join("%s~%s %.3f" % (n[i], n[j], C[i, j]) for _, i, j in pairs[:6]))
    mu = np.linalg.eigvalsh(C)
    print("  participation ratio (sum mu)^2 / sum mu^2 = %.2f of %d" % (mu.sum()**2 / (mu**2).sum(), len(mu)))
    # P18: the Sigma^-1 fit does not depend on the units of the statistics
    spl, se = binding_spline(D), D["se"]
    tau_fine = np.geomspace(D["tau_grid"][0], D["tau_grid"][-1], cfg["n_fine"])
    Qf, q0 = spl(np.log(tau_fine)), spl(np.log(cfg["tau0"]))
    W_mv, W_uv = np.linalg.inv(D["Sigma"]), np.linalg.inv(S_uv)
    deltas = [np.eye(len(se))[j] * cfg["bias_in_se"] * se[j] for j in range(len(se))]
    pair = np.zeros(len(se)); pair[[1, 6]] = cfg["bias_in_se"] * se[[1, 6]]
    same = []
    for delta in deltas + [pair]:
        r_mv = Qf - (q0 + delta)
        r_uv = r_mv * scale
        k_mv = np.einsum("gi,ij,gj->g", r_mv, W_mv, r_mv).argmin()
        k_uv = np.einsum("gi,ij,gj->g", r_uv, W_uv, r_uv).argmin()
        same.append(k_mv == k_uv)
    print("  Sigma^-1 fit with SDs in microvolt: same tau_hat as in mV for %d of %d inputs" % (sum(same), len(same)))
    return {"P18": bool(all(same))}


def main():
    cfg = CONFIG
    D = load(RES)
    rng = np.random.default_rng(cfg["seed"])
    ok = {}
    print("== A. Monte Carlo error of Sigma_hat vs condition number (kappa(Sigma) = %.2e, m = %d)"
          % (np.linalg.cond(D["Sigma"]), len(D["se"])))
    ok.update(part_a(D, cfg, rng))
    print("\n== B. linearised fit at tau_s0 = %.2f ms" % (1e3 * cfg["tau0"]))
    okb, G, W, sd_full = part_b(D, cfg)
    ok.update(okb)
    print("\n== C. model error of one SE in one statistic (noise-free input)")
    ok.update(part_c(D, cfg, W))
    print("\n== D. Gaussian inputs with the same nonlinear fit (diagnosis of P12 for Sigma^-1)")
    ok.update(part_d(D, cfg, W, sd_full))
    print("\n== E. units and correlations (measured)")
    ok.update(part_e(D, cfg))
    print("\nALL PREDICTIONS:", ok)


if __name__ == "__main__":
    main()
