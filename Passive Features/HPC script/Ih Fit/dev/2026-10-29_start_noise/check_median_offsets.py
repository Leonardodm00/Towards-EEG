"""check_median_offsets.py -- standalone check (not pipeline code), 2026-10-29
(later, 5): is it the tilt's spread of the windows' BASELINES that degrades
the per-sample median window of D-103.1?

Hypothesis (the assistant's, to test): a sweep's tilt b shifts window k by
b (t_k - t_bar), up to +/-0.36 mV across a 37-pulse train at 0.1 mV/s, far
more than the noise; the per-sample median over a group's windows then picks,
at every sample, the window whose baseline is in the middle, so the median
window carries about one window's noise instead of a median's. Two
consequences, each tested here:
  A  the start (D-103.2, D-110, D-111): the scatter of a group's t_sp and the
     fallback share grow with the tilt; removing each window's tilt offset
     b_hat_s (t_k - t_bar_s) (b_hat_s read by (TL1) at the 80 ms floor)
     before the median should restore the untilted behaviour;
  B  the subtraction (D-103.1): at a white law, the pooled mean of
     sigma_hat^2 after the median subtraction rises from c(K) sigma_f^2
     (about 0.96-0.97) towards 2 sigma_f^2 (a window minus another one) as
     the tilt grows; removing the offsets first keeps it at c(K).
Predictions, fixed before running:
 PA1 plain median, sigma_b = 0.1, 2 sweeps: SD of t_sp - t0 above 30 ms;
     offsets removed first: below 22 ms, and the fallback share below 5 %;
 PB1 plain median: mean sigma_hat^2 / sigma_f^2 above 1.5 at b = 0.1 mV/s;
 PB2 offsets removed first: within 0.02 of the b = 0 value at every b.
Cells: check_start_noise.py's simulator (RC 113 MOhm, 28.143 ms, theta_cf,
27 % drops) for A, 100 cells per row; for B a white law (sigma_f 0.05 mV,
rho_f 1e-3, sigma_s 0), no response, 2 sweeps, 60 cells per tilt, the tilt
b the same in both sweeps, windows cut at 80 ms. Seed 20261045.
Run (from this folder; about 6 min):  python check_median_offsets.py
Pure ASCII, LF.
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_start_noise as csn            # noqa: E402  (simulator, t0, the Ih Fit imports)

nw = csn.nw
nl = csn.nl
import noise_cell_stats as ncs             # noqa: E402

CFG = dict(seed=20261045, nA=100, nB=60, tilts_B=(0.0, 0.02, 0.1))
WCFG = nw.WindowConfig()
WHITE = nl.NoiseLaw(0.05, 1e-3, 0.0, 0.04)


def _replace(ss, v):
    from dataclasses import replace
    return replace(ss, v=v)


def read_tilts(ss):
    X, j0 = nw.enlarged_windows(ss, WCFG.t_ss_floor_ms)
    return ncs.ss_sweep_tilts(X, ss, j0).as_dict()


def part_a():
    rng = np.random.default_rng(CFG["seed"])
    resp = csn.train_response()
    ok = True
    for sb in (0.0, 0.1):
        for S in (2, 6):
            dev = {"plain": [], "aligned": []}
            fall = {"plain": 0, "aligned": 0}
            for _ in range(CFG["nA"]):
                ss = csn.simulate_cell(S, sb, resp, rng)
                for name, cell in (("plain", ss), ("aligned", _replace(ss, offsets_removed_v(ss)))):
                    st = nw.decay_start(cell)
                    t0 = csn.t0_of(st.sigma_ss_mV)
                    good = np.array([x == "ok" for x in st.group_status])
                    dev[name].extend((st.t_group_ms[good] - t0).tolist())
                    fall[name] += int("ss_start_fallback" in st.flags)
            sd = {k: float(np.std(v, ddof=1)) for k, v in dev.items()}
            share = {k: fall[k] / CFG["nA"] for k in fall}
            print("A  sigma_b %.1f mV/s, %d sweeps: SD of t_sp - t0 plain %5.1f ms, offsets removed %5.1f ms; fallback "
                  "plain %3.0f %%, offsets removed %3.0f %%" % (sb, S, sd["plain"], sd["aligned"],
                                                               100 * share["plain"], 100 * share["aligned"]))
            if sb == 0.1 and S == 2:
                ok = sd["plain"] > 30.0 and sd["aligned"] < 22.0 and share["aligned"] < 0.05
    return ok


def offsets_removed_v(ss):
    """The stored windows minus each window's tilt offset b_hat_s (t_k - t_bar_s), b_hat_s read by (TL1) at 80 ms."""
    b = read_tilts(ss)
    v = ss.v.copy()
    for s, bs in b.items():
        g = np.flatnonzero(ss.sweep_number == s)
        t_on = ss.position[g] * ss.period_samples * ss.dt
        v[g] -= (bs * (t_on - t_on.mean()))[:, None]
    return v


def part_b():
    rng = np.random.default_rng(CFG["seed"] + 1)
    zero = csn.train_response() * 0.0
    res = {}
    for b in CFG["tilts_B"]:
        plain, aligned = [], []
        for _ in range(CFG["nB"]):
            ss = csn.simulate_cell(2, 0.0, zero, rng)                    # white noise only, no response
            # add the tilt b to every sweep (the same b), as a line in each sweep's own time
            v = ss.v.copy()
            for s in np.unique(ss.sweep_number):
                g = np.flatnonzero(ss.sweep_number == s)
                t_abs = (ss.position[g] * ss.period_samples)[:, None] * ss.dt + (np.arange(ss.n_w) * ss.dt)[None, :]
                v[g] += b * (t_abs - t_abs.mean())
            cell = _replace(ss, v)
            for name, c in (("plain", cell), ("aligned", _replace(cell, offsets_true(cell, b)))):
                X, _ = nw.enlarged_windows(c, WCFG.t_ss_floor_ms)
                Xt, _ = nw.subtract_median_response(X, c.polarity_is_dep, c.sweep_number)
                (plain if name == "plain" else aligned).extend((Xt.std(axis=1, ddof=1) ** 2 / WHITE.sigma_f ** 2).tolist())
        res[b] = (float(np.mean(plain)), float(np.mean(aligned)))
        print("B  b %.2f mV/s: mean sigma_hat^2 / sigma_f^2 after the median subtraction: plain %.4f, offsets removed %.4f"
              % (b, res[b][0], res[b][1]))
    pb1 = res[0.1][0] > 1.5
    pb2 = all(abs(res[b][1] - res[0.0][1]) <= 0.02 for b in res)
    return pb1, pb2


def offsets_true(cell, b):
    """The stored windows minus each window's offset from the known tilt b."""
    v = cell.v.copy()
    for s in np.unique(cell.sweep_number):
        g = np.flatnonzero(cell.sweep_number == s)
        t_on = cell.position[g] * cell.period_samples * cell.dt
        v[g] -= (b * (t_on - t_on.mean()))[:, None]
    return v


def main():
    t0 = time.time()
    pa1 = part_a()
    law_saved = csn.LAW
    csn.LAW = WHITE                                                       # part B: the white law
    pb1, pb2 = part_b()
    csn.LAW = law_saved
    print("PA1 %s  PB1 %s  PB2 %s   (%.0f s)" % ("PASS" if pa1 else "FAIL", "PASS" if pb1 else "FAIL",
                                               "PASS" if pb2 else "FAIL", time.time() - t0))
    return pa1 and pb1 and pb2


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
