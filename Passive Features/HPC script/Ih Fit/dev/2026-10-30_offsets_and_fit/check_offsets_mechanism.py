"""check_offsets_mechanism.py -- standalone check (not pipeline code), 2026-10-30:
the claims behind the proposal to remove each SS window's tilt offset before
the per-sample median window of D-103.1, as explained to the user.

Setting. A sweep's tilt b (mV/s; the term b (t - t_bar) of law (L1)) puts
window k of the sweep at the baseline offset b (t_k - t_bar) (t_k its start
time, t_bar the sweep's mid-time) and adds inside every window the ramp
b (t - t_k), which is the same in every window of a group (sweep, polarity)
and so passes through the per-sample median exactly. The proposal: subtract
b_hat_s (t_k - t_bar_s) from every window of sweep s before the median, with
b_hat_s the sweep's (TL1) slope read on the windows cut at the 80 ms floor
and t_bar_s the mean of the sweep's kept windows' t_k.

Claims tested (predictions fixed before running):
 M  which windows supply the per-sample median of a group, on [5, 200) ms
    (2-sweep cells, theta_cf noise, no response, the same tilt b in both
    sweeps; 100 cells per b):
    PM1  b = 0: the central third of a group's windows (onset order)
         supplies the median at a share within 0.03 of its share of the
         windows;
    PM2  b = 0.1 mV/s: the central third supplies at least 0.6 of the median.
 E  the subtraction with the offsets ESTIMATED as proposed (white law,
    sigma_f 0.05 mV, rho_f 1e-3, sigma_s 0; 2 sweeps; windows cut at 80 ms;
    60 cells per tilt):
    PE1  mean sigma_hat^2 / sigma_f^2 within 0.01 of its b = 0 value at
         every b in {0, 0.02, 0.1} mV/s.
 F  the start on tilted 2-sweep cells (theta_cf; RC 113 MOhm / 28.143 ms;
    27 % drops; tilt per sweep ~ N(0, 0.1^2) mV/s; 300 cells), paired: as
    implemented (Block 3's decay_start) and with the estimated offsets
    removed first; beside them 300 untilted cells as the reference:
    PF1  fewer fallbacks with the offsets removed than as implemented;
    PF2  SD of t_sp - t0 over the valid groups: above 30 ms as implemented,
         at most 24 ms with the offsets removed.
Seed 20261051. Uses check_start_noise.py's simulator (../2026-10-29_start_noise),
which finds the Ih Fit folder itself (or TEEG_IH_FIT).
Run (from this folder; about 5 min):  python check_offsets_mechanism.py
Pure ASCII, LF.
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-29_start_noise"))
import check_start_noise as csn            # noqa: E402  (simulator, t0; puts the Ih Fit folder on the path)

nw = csn.nw
nl = csn.nl
import noise_cell_stats as ncs             # noqa: E402

CFG = dict(seed=20261051, nM=100, b_M=(0.0, 0.1), nE=60, b_E=(0.0, 0.02, 0.1), nF=300, sigma_b_F=0.1,
           tilt_read_ms=80.0, pm1_tol=0.03, pm2_min=0.6, pe1_tol=0.01, pf2_plain_min=30.0, pf2_removed_max=24.0)
WCFG = nw.WindowConfig()
WHITE = nl.NoiseLaw(0.05, 1e-3, 0.0, 0.04)


# ---------------------------------------------------------------- helpers (analysis)
def add_tilt(ss, b):
    """The same tilt b (mV/s) on every sweep: b (t - t_bar), t the absolute time in the
    sweep, t_bar the sweep's mid-time ((L - 1) / 2 dt, the law's centring)."""
    t_bar = 0.5 * (csn.L_SWEEP - 1) * ss.dt
    t_abs = (ss.position[:, None] * ss.period_samples + np.arange(ss.n_w)[None, :]) * ss.dt
    return replace(ss, v=ss.v + b * (t_abs - t_bar))


def onset_times(ss, idx):
    return ss.position[idx] * ss.period_samples * ss.dt


def remove_offsets(ss, b_of):
    """Every window of sweep s minus b_of[s] (t_k - t_bar_s), t_bar_s the mean t_k of the sweep's windows."""
    v = ss.v.copy()
    for s, bs in b_of.items():
        g = np.flatnonzero(ss.sweep_number == s)
        t_k = onset_times(ss, g)
        v[g] -= (bs * (t_k - t_k.mean()))[:, None]
    return replace(ss, v=v)


def estimated_tilts(ss):
    """b_hat_s by (TL1) on the windows cut at the 80 ms floor (unsubtracted: the slope
    does not see the group subtraction, the polarity intercepts absorb it)."""
    X, j0 = nw.enlarged_windows(ss, CFG["tilt_read_ms"])
    return ncs.ss_sweep_tilts(X, ss, j0).as_dict()


def median_weights(V):
    """V (K, n): per window, the share of the n samples at which it is the median
    (numpy's median: the middle value for odd K, the mean of the two middle values for
    even K, each then counting 1/2)."""
    K, n = V.shape
    order = np.argsort(V, axis=0, kind="stable")
    w = np.zeros(K)
    if K % 2:
        np.add.at(w, order[(K - 1) // 2], 1.0)
    else:
        np.add.at(w, order[K // 2 - 1], 0.5)
        np.add.at(w, order[K // 2], 0.5)
    return w / n


def central(K):
    """Onset-order indices whose relative position (i + 0.5) / K lies in [1/3, 2/3]."""
    u = (np.arange(K) + 0.5) / K
    return (u >= 1.0 / 3.0) & (u <= 2.0 / 3.0)


# ---------------------------------------------------------------- parts
def part_m(rng):
    zero = np.zeros(csn.L_SWEEP)
    sel = slice(csn.PRE_N + int(round(WCFG.fit_start_ms * 1e-3 * csn.FS)), csn.N_W)
    out = {}
    for b in CFG["b_M"]:
        med_share, win_share, prof = [], [], np.zeros(5)
        for _ in range(CFG["nM"]):
            ss = add_tilt(csn.simulate_cell(2, 0.0, zero, rng), b)
            for s in np.unique(ss.sweep_number):
                for dep in (True, False):
                    idx = np.flatnonzero((ss.sweep_number == s) & (ss.polarity_is_dep == dep))
                    idx = idx[np.argsort(ss.position[idx])]
                    if idx.size < WCFG.min_windows_per_group:
                        continue
                    w = median_weights(ss.v[idx][:, sel])
                    c = central(idx.size)
                    med_share.append(w[c].sum())
                    win_share.append(c.mean())
                    bins = np.minimum((5 * (np.arange(idx.size) + 0.5) / idx.size).astype(int), 4)
                    prof += np.bincount(bins, weights=w, minlength=5)
        out[b] = (float(np.mean(med_share)), float(np.mean(win_share)), prof / prof.sum(), len(med_share))
        print("M  b %.2f mV/s: central third supplies %.3f of the median (it holds %.3f of the windows); "
              "median weight by fifth of the train %s  (%d groups)"
              % (b, out[b][0], out[b][1], np.array2string(out[b][2], precision=3), out[b][3]))
    pm1 = abs(out[0.0][0] - out[0.0][1]) <= CFG["pm1_tol"]
    pm2 = out[0.1][0] >= CFG["pm2_min"]
    return pm1, pm2


def part_e(rng):
    law_saved = csn.LAW
    csn.LAW = WHITE
    zero = np.zeros(csn.L_SWEEP)
    res = {}
    try:
        for b in CFG["b_E"]:
            acc = {"plain": [], "true": [], "estimated": []}
            for _ in range(CFG["nE"]):
                ss = add_tilt(csn.simulate_cell(2, 0.0, zero, rng), b)
                cells = {"plain": ss, "true": remove_offsets(ss, {int(s): b for s in np.unique(ss.sweep_number)}),
                         "estimated": remove_offsets(ss, estimated_tilts(ss))}
                for name, c in cells.items():
                    X, _ = nw.enlarged_windows(c, CFG["tilt_read_ms"])
                    Xt, _ = nw.subtract_median_response(X, c.polarity_is_dep, c.sweep_number)
                    acc[name].extend((Xt.var(axis=1, ddof=1) / WHITE.sigma_f ** 2).tolist())
            res[b] = {k: float(np.mean(v)) for k, v in acc.items()}
            print("E  b %.2f mV/s: mean sigma_hat^2 / sigma_f^2 after the median subtraction: plain %.4f, "
                  "true offsets removed %.4f, estimated offsets removed %.4f"
                  % (b, res[b]["plain"], res[b]["true"], res[b]["estimated"]))
    finally:
        csn.LAW = law_saved
    return all(abs(res[b]["estimated"] - res[0.0]["estimated"]) <= CFG["pe1_tol"] for b in res)


def part_f(rng):
    resp = csn.train_response()
    rows = {}
    for label, sb in (("untilted", 0.0), ("tilted", CFG["sigma_b_F"])):
        dev = {"plain": [], "removed": []}
        fall = {"plain": np.zeros(CFG["nF"], bool), "removed": np.zeros(CFG["nF"], bool)}
        for c in range(CFG["nF"]):
            ss = csn.simulate_cell(2, sb, resp, rng)
            for name, cell in (("plain", ss), ("removed", remove_offsets(ss, estimated_tilts(ss)))):
                st = nw.decay_start(cell)
                t0 = csn.t0_of(st.sigma_ss_mV)
                ok = np.array([x == "ok" for x in st.group_status])
                dev[name].extend((st.t_group_ms[ok] - t0).tolist())
                fall[name][c] = "ss_start_fallback" in st.flags
        rows[label] = (dev, fall)
        rescued = int(np.sum(fall["plain"] & ~fall["removed"]))
        lost = int(np.sum(~fall["plain"] & fall["removed"]))
        print("F  %s 2-sweep cells (sigma_b %.1f mV/s, %d cells): SD of t_sp - t0 as implemented %.1f ms, offsets "
              "removed %.1f ms; fallback as implemented %.1f %%, offsets removed %.1f %% (rescued %d, lost %d)"
              % (label, sb, CFG["nF"], np.std(dev["plain"], ddof=1), np.std(dev["removed"], ddof=1),
                 100 * fall["plain"].mean(), 100 * fall["removed"].mean(), rescued, lost))
    dev, fall = rows["tilted"]
    pf1 = fall["removed"].sum() < fall["plain"].sum()
    pf2 = np.std(dev["plain"], ddof=1) > CFG["pf2_plain_min"] and np.std(dev["removed"], ddof=1) <= CFG["pf2_removed_max"]
    return pf1, pf2


def main():
    t_start = time.time()
    rng = np.random.default_rng(CFG["seed"])
    pm1, pm2 = part_m(rng)
    pe1 = part_e(rng)
    pf1, pf2 = part_f(rng)
    verdict = dict(PM1=pm1, PM2=pm2, PE1=pe1, PF1=pf1, PF2=pf2)
    print("  ".join("%s %s" % (k, "PASS" if v else "FAIL") for k, v in verdict.items())
          + "   (%.0f s)" % (time.time() - t_start))
    return all(verdict.values())


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
