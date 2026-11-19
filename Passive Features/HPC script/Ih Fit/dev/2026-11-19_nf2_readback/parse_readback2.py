#!/usr/bin/env python3
"""Parse the user's paste of the nf2 second read-back (histories + per-sweep fits) and summarise it against nf1.
Usage: python3 -I parse_readback2.py <readback2.txt> <nf1 noise_cells csv> <nf1 noise_sweeps csv>"""
import re, sys
import numpy as np, pandas as pd
from scipy.stats import chi2

txt = open(sys.argv[1]).read().splitlines()
H = re.compile(r"^(\d+) (\S+)\s+R=(\d+)\s+J=\s*([\d.]+) max\|du\|=(\S+)\s+sigma_f ([\d.]+) sigma_m ([\d.]+) tau_m ([\d.]+) ms sigma_s ([\d.]+) tau_s ([\d.]+) ms")
S = re.compile(r"^(\d+) (ss|ls) sweeps fitted (\d+), J_s/dof_s median ([\d.]+), q90 ([\d.]+), max ([\d.]+), p<0.01 in (\d+)")
hist, sw = [], []
for line in txt:
    m = H.match(line)
    if m:
        g = m.groups()
        hist.append(dict(sid=int(g[0]), weight=g[1], R=int(g[2]), J=float(g[3]), du=(np.nan if g[4] == "-" else float(g[4])),
                         sf=float(g[5]), sm=float(g[6]), tm=float(g[7]), ss=float(g[8]), ts=float(g[9])))
        continue
    m = S.match(line)
    if m:
        g = m.groups()
        sw.append(dict(sid=int(g[0]), pr=g[1], n=int(g[2]), med=float(g[3]), q90=float(g[4]), mx=float(g[5]), p01=int(g[6])))
H_ = pd.DataFrame(hist); S_ = pd.DataFrame(sw)
print("cells:", H_.sid.nunique(), "history lines", len(H_), "| sweep lines", len(S_))
fin = H_[H_.weight == "final"].set_index("sid")
nsig = H_[H_.weight.str.startswith("Sigma")].groupby("sid").size()
last_sig_du = H_[H_.weight.str.startswith("Sigma")].groupby("sid").du.last()
c1 = pd.read_csv(sys.argv[2]).set_index("specimen_id")
T = pd.DataFrame(dict(n_sigma=nsig, last_sigma_du=last_sig_du, J=fin.J, sf=fin.sf, sm=fin.sm, tm=fin.tm, ss=fin.ss, ts=fin.ts))
T["no_plateau"] = T.n_sigma == 6
T["p"] = chi2.sf(T.J, 16)
T["J1"] = c1.noise_J; T["dof1"] = c1.noise_dof
T["Jdof1"] = T.J1 / T.dof1; T["Jdof2"] = T.J / 16
T["ts1"] = c1.noise_tau_s_ms; T["ss1"] = c1.noise_sigma_s_mV
T["ratio2"] = T.ss ** 2 / (T.ts / 1e3); T["ratio1"] = T.ss1 ** 2 / (T.ts1 / 1e3)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(T.to_string(float_format=lambda x: "%.4g" % x))
print()
print("converged (Sigma solves < 6): %s ; no_plateau: %d of %d" % (T.index[~T.no_plateau].tolist(), T.no_plateau.sum(), len(T)))
print("final J: median %.1f, range %.1f-%.1f; J/dof median %.2f (nf1 %.2f); J < 32.0 (chi2_16 99%%) in %d, < 39.25 (99.9%%) in %d"
      % (T.J.median(), T.J.min(), T.J.max(), T.Jdof2.median(), T.Jdof1.median(), (T.J < chi2.ppf(.99, 16)).sum(), (T.J < chi2.ppf(.999, 16)).sum()))
print("J/dof nf2 < nf1 in %d of %d; ratio (nf2/nf1) median %.2f, range %.2f-%.2f" % ((T.Jdof2 < T.Jdof1).sum(), len(T), (T.Jdof2 / T.Jdof1).median(), (T.Jdof2 / T.Jdof1).min(), (T.Jdof2 / T.Jdof1).max()))
for col, nm in (("sf", "sigma_f mV"), ("sm", "sigma_m mV"), ("tm", "tau_m ms"), ("ss", "sigma_s mV"), ("ts", "tau_s ms")):
    print("%-11s median %.4g range %.4g-%.4g" % (nm, T[col].median(), T[col].min(), T[col].max()))
print("tau_m at bound (<= 0.0401 or >= 0.999 ms): %d ; tau_s at bound (<= 1.001 or >= 9990 ms): %d" % (((T.tm <= 0.0401) | (T.tm >= 0.999)).sum(), ((T.ts <= 1.001) | (T.ts >= 9990)).sum()))
print("tau_s > 265 ms (the longest window): %s" % T.index[T.ts > 265].tolist())
print()
print("the last two Sigma solves and the final step: max|du| per cell")
for sid, g in H_.groupby("sid"):
    print(sid, np.round(g.du.values[1:], 3).tolist())
print()
# along the slow ridge: sigma_s^2 / tau_s over each history (mV^2/s)
print("sigma_s^2 / tau_s (mV^2/s) along the history, Sigma solves and final; and tau_s (ms):")
for sid in (571430815, 526785799, 508282493, 539653924, 541545716, 571624532, 529878215):
    g = H_[(H_.sid == sid) & (H_.weight != "W0")]
    print(sid, "ratio", np.round(g.ss.values ** 2 / (g.ts.values / 1e3), 4).tolist(), "| tau_s", g.ts.round(1).tolist(), "| J", g.J.round(1).tolist())
print()
sw1 = pd.read_csv(sys.argv[3])
fl = sw1.noise_sweep_flags.fillna("")
S_["n_cells"] = 1
agg = S_.groupby("pr").agg(n=("n", "sum"), p01=("p01", "sum"), cells=("n_cells", "sum"))
agg["frac"] = agg.p01 / agg.n
print("per-sweep fits nf2:", agg.to_dict("index"))
print("per-sweep medians of J_s/dof_s across cells: SS %s ; LS %s" % (np.round(np.sort(S_[S_.pr == "ss"].med.values), 2).tolist(), np.round(np.sort(S_[S_.pr == "ls"].med.values), 2).tolist()))
print("LS sweeps in the cohort (K_LS sum): %d -> failed in nf2: %d (nf1: %d)" % (c1.noise_K_LS.sum(), c1.noise_K_LS.sum() - S_[S_.pr == "ls"].n.sum(), fl.str.contains("sweep_fit_failed").sum()))
T.to_csv("/home/claude/scratch/nf2_run/nf2_cells_from_paste.csv")
