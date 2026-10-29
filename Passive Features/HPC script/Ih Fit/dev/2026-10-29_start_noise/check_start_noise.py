"""check_start_noise.py -- standalone check (not pipeline code), 2026-10-29:
how the SS window start of D-103 / D-104, as Block 3 implements it
(noise_windows.decay_start), behaves on noisy cells.

Rule (SPEC 2.2 (a)-(e)): per sweep and polarity, the median window's decay
B + A exp(-t / tau) fitted on [5, 200) ms; the group's start t_sp = tau_hat
ln(|A_hat| / (kappa_tr sigma_SS_hat)); t_star(c) = max(80 ms, max over
groups of t_sp); t_SS(c) = t_star(c) when 200 - t_star >= 50 ms, else the
80 ms fallback.

Simulated cells (chosen): S SS sweeps of the archive's train (37 pulses of
+/-200 pA x 0.5 ms, dep first, 10025 samples onset to onset at 50 kHz),
each a continuous sweep: the RC response of the whole train (113 MOhm,
tau_m 28.143 ms, the Block 3 fixture's) plus Block-2 noise at theta_cf =
(0.0380 mV, 0.175, 0.0508 mV, 36.6 ms); the stored windows (-10, +200) ms,
each pulse dropped independently with probability 0.27 (the cohort's median
trace-QC drop, D-072); S in {2, 6, 10}; tilt sigma_b 0 or 0.1 mV/s.
Reference start t0(c): the noiseless train window is exactly B + A exp(-t /
tau_m) with A = V0 / (1 + exp(-P / tau_m)), V0 = Q R_in / tau_m, Q = 0.1 pC
(check_d095_rc_start.py), so t0 = tau_m ln(A / (kappa_tr sigma_SS_hat(c))),
at the cell's own sigma_SS_hat.

Predictions, fixed before running:
 P1  the per-group start is centred on t0: the median over all groups of
     t_sp - t0 lies within 5 ms of 0 (sigma_b = 0);
 P2  it scatters: the SD of t_sp - t0 over groups is at least 8 ms
     (sigma_b = 0), from the slow component in a median of 10-19 windows;
 P3  the maximum over groups moves t_star later as the groups grow in
     number: the median of t_star - t0 rises from S = 2 to S = 10, and so
     does the fallback fraction (sigma_b = 0).
Seed 20261041; numpy, scipy; noise_windows, noise_law, smoke_noise_windows
from the Ih Fit folder (two levels up, or TEEG_IH_FIT).
Run (from this folder; about 10 min):  python check_start_noise.py
Pure ASCII, LF.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

IH_FIT = Path(os.environ.get("TEEG_IH_FIT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(IH_FIT))
import noise_law as nl                  # noqa: E402
import noise_windows as nw              # noqa: E402
import smoke_noise_windows as snw       # noqa: E402  (rc_response: the fixture's RC membrane)

CFG = dict(seed=20261041, n_cells=150, sweeps=(2, 6, 10), sigma_b=(0.0, 0.1), p_drop=0.27,
           r_mohm=113.0, tau_m_ms=28.143, q_pC=0.1, p1_ms=5.0, p2_ms=8.0)
FS, DT, P, TRAIN, PRE_N, POST_N, PULSE_N = 50000.0, 2e-5, 10025, 37, 500, 10000, 25
N_W = PRE_N + POST_N
L_SWEEP = (TRAIN - 1) * P + N_W
LAW = nl.NoiseLaw(0.0380, 0.175, 0.0508, 0.0366)
KAPPA = nw.WindowConfig().kappa_tr


def train_response():
    """Noiseless RC response of the whole train on one sweep of L_SWEEP samples
    (window of position p: samples [p P, p P + N_W), onset at p P + PRE_N)."""
    i = np.zeros(L_SWEEP)
    for k in range(TRAIN):
        s0 = k * P + PRE_N
        i[s0:s0 + PULSE_N] = 200.0 if k % 2 == 0 else -200.0
    return snw.rc_response(i, r_mohm=CFG["r_mohm"], tau_ms=CFG["tau_m_ms"])


def simulate_cell(n_sweeps, sigma_b, resp, rng):
    v, pol, sw, pos = [], [], [], []
    for s in range(n_sweeps):
        units = nl.draw_units(rng, 1, L_SWEEP)
        trace = -60.0 + resp + nl.simulate_sweep(LAW, DT, L_SWEEP, units, 0, sigma_b=sigma_b)
        kept = np.flatnonzero(rng.random(TRAIN) >= CFG["p_drop"])
        for p in kept:
            v.append(trace[p * P:p * P + N_W]); pol.append(p % 2 == 0); sw.append(100 + s); pos.append(p)
    return nw.SSWindows(specimen_id=0, fs=FS, dt=DT, t=(np.arange(N_W) - PRE_N) / FS, v=np.array(v),
                        polarity_is_dep=np.array(pol), sweep_number=np.array(sw, dtype=np.int64),
                        position=np.array(pos, dtype=np.int64), train_length=TRAIN, period_samples=P,
                        pre_n=PRE_N, n_raw=TRAIN * n_sweeps, n_sweeps=n_sweeps)


def t0_of(sigma_hat):
    v0 = CFG["q_pC"] * 1e-12 * CFG["r_mohm"] * 1e6 / (CFG["tau_m_ms"] * 1e-3) * 1e3        # mV
    a = v0 / (1.0 + np.exp(-P / FS * 1e3 / CFG["tau_m_ms"]))
    return CFG["tau_m_ms"] * np.log(a / (KAPPA * sigma_hat))


def main():
    t_start = time.time()
    rng = np.random.default_rng(CFG["seed"])
    resp = train_response()
    rows = {}
    for sb in CFG["sigma_b"]:
        for S in CFG["sweeps"]:
            dev_g, dev_star, fall, floor, bad, n_groups, t_ss = [], [], 0, 0, 0, [], []
            for c in range(CFG["n_cells"]):
                ss = simulate_cell(S, sb, resp, rng)
                st = nw.decay_start(ss)
                t0 = t0_of(st.sigma_ss_mV)
                ok = np.array([x == "ok" for x in st.group_status])
                bad += int((~ok).sum())
                dev_g.extend((st.t_group_ms[ok] - t0).tolist())
                n_groups.append(int(ok.sum()))
                if np.isfinite(st.t_star_ms):
                    dev_star.append(st.t_star_ms - t0)
                fall += int("ss_start_fallback" in st.flags)
                floor += int("ss_start_floor" in st.flags)
                t_ss.append(st.t_ss_ms)
            dev_g, dev_star = np.array(dev_g), np.array(dev_star)
            rows[(sb, S)] = dict(med_g=float(np.median(dev_g)), sd_g=float(np.std(dev_g, ddof=1)),
                                 q_g=np.percentile(dev_g, [5, 95]), med_star=float(np.median(dev_star)),
                                 q_star=np.percentile(dev_star, [5, 95]), fall=fall / CFG["n_cells"],
                                 floor=floor / CFG["n_cells"], bad=bad, groups=float(np.mean(n_groups)),
                                 t_ss_med=float(np.median(t_ss)))
            r = rows[(sb, S)]
            print("sigma_b %.1f mV/s, %2d sweeps (%4.1f valid groups per cell, %d fits failed): t_sp - t0 median %+6.1f ms, "
                  "SD %5.1f, 5-95 %% [%+6.1f, %+6.1f]; t_star - t0 median %+6.1f [%+6.1f, %+6.1f]; fallback %3.0f %%, "
                  "floor %3.0f %%; t_SS median %5.1f ms   (%.0f s)"
                  % (sb, S, r["groups"], r["bad"], r["med_g"], r["sd_g"], r["q_g"][0], r["q_g"][1], r["med_star"],
                     r["q_star"][0], r["q_star"][1], 100 * r["fall"], 100 * r["floor"], r["t_ss_med"], time.time() - t_start))
    base = [rows[(0.0, S)] for S in CFG["sweeps"]]
    p1 = all(abs(r["med_g"]) <= CFG["p1_ms"] for r in base)
    p2 = all(r["sd_g"] >= CFG["p2_ms"] for r in base)
    p3 = (base[-1]["med_star"] > base[0]["med_star"]) and (base[-1]["fall"] >= base[0]["fall"])
    print("noiseless reference t0 at sigma_SS_hat 0.0402 mV: %.1f ms" % t0_of(0.0402))
    print("P1 %s  P2 %s  P3 %s   (%.0f s)" % ("PASS" if p1 else "FAIL", "PASS" if p2 else "FAIL",
                                             "PASS" if p3 else "FAIL", time.time() - t_start))
    return p1 and p2 and p3


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
