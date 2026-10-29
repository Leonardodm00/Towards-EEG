"""check_start_detrended.py -- standalone check (not pipeline code), 2026-10-29
(later, 5): does removing each sweep's own tilt from its groups' median
windows before the decay fit keep the window start of D-110 / D-111 off
the fallback? A possible remedy for the residual fallback of tilted 2-sweep
cells (check_start_decided.py: 16 % at sigma_b = 0.1 mV/s); not a decision.

Variant (proposal): per SS sweep s, b_hat_s = (TL1) read on the windows cut at
the 80 ms floor (Block 3's ss_sweep_tilts on the unsubtracted enlarged
windows: the slope does not depend on the group's common response, which the
polarity intercepts absorb); each group's median window minus b_hat_s t (the
tilt's ramp inside the window, common to the group's windows, passes through
the per-sample median exactly); then Block 3's exact fit (fit_decay), (ST1),
the median over groups and (ST2) as decided. Compared, on the same cells
(check_start_noise.py's simulator, seed 20261044, 150 cells per row), with
Block 3's decay_start as implemented.

Predictions, fixed before running:
 P1  with sigma_b = 0.1 mV/s the variant falls back in at most 5 % of cells
     in every row;
 P2  the variant's median of t_star - t0 lies within 5 ms of 0 in every row
     (both tilts: removing a tilt that is not there does no harm).
Run (from this folder; about 9 min):  python check_start_detrended.py
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
import noise_cell_stats as ncs             # noqa: E402  (on the path through check_start_noise)

CFG = dict(seed=20261044, n_cells=150, p1_share=0.05, p2_ms=5.0, tilt_read_ms=80.0)
WCFG = nw.WindowConfig()


def detrended_start(ss):
    """t_star and the fallback under the variant (the decided rule otherwise)."""
    X, j0 = nw.enlarged_windows(ss, CFG["tilt_read_ms"])
    b = ncs.ss_sweep_tilts(X, ss, j0).as_dict()                       # mV/s per sweep
    pre = ss.v[:, :ss.pre_n]
    sigma = float(np.median(pre.std(axis=1, ddof=1)))
    thr = WCFG.kappa_tr * sigma
    t_ms = ss.t * 1e3
    sel = np.arange(ss.n_w) >= ss.pre_n + int(round(WCFG.fit_start_ms * 1e-3 * ss.fs))
    t_sp = []
    for s in dict.fromkeys(int(x) for x in ss.sweep_number):
        for dep in (True, False):
            idx = np.flatnonzero((ss.sweep_number == s) & (ss.polarity_is_dep == dep))
            if idx.size < WCFG.min_windows_per_group:
                continue
            med = np.median(ss.v[idx], axis=0) - b[s] * t_ms * 1e-3      # the sweep's ramp removed
            f = nw.fit_decay(t_ms[sel], med[sel], WCFG)
            if f.ok:
                t_sp.append(f.tau_ms * np.log(abs(f.A_mV) / thr) if abs(f.A_mV) > thr else 0.0)
    if not t_sp:
        return float("nan"), True, sigma
    t_star = max(WCFG.t_ss_floor_ms, float(np.median(t_sp)))
    return t_star, (200.0 - t_star < WCFG.min_window_ms), sigma


def main():
    t_start = time.time()
    rng = np.random.default_rng(CFG["seed"])
    resp = csn.train_response()
    ok = dict(P1=True, P2=True)
    for sb in csn.CFG["sigma_b"]:
        for S in csn.CFG["sweeps"]:
            dec_dev, dec_fall, var_dev, var_fall = [], 0, [], 0
            for _ in range(CFG["n_cells"]):
                ss = csn.simulate_cell(S, sb, resp, rng)
                st = nw.decay_start(ss)
                t0 = csn.t0_of(st.sigma_ss_mV)
                if np.isfinite(st.t_star_ms):
                    dec_dev.append(st.t_star_ms - t0)
                dec_fall += int("ss_start_fallback" in st.flags)
                t_star, fell, sigma = detrended_start(ss)
                if np.isfinite(t_star):
                    var_dev.append(t_star - csn.t0_of(sigma))
                var_fall += int(fell)
            share = var_fall / CFG["n_cells"]
            p1 = share <= CFG["p1_share"] if sb > 0 else True
            p2 = abs(float(np.median(var_dev))) <= CFG["p2_ms"]
            ok["P1"] &= p1
            ok["P2"] &= p2
            print("sigma_b %.1f mV/s, %2d sweeps: as decided t_star - t0 median %+5.1f ms, fallback %3.0f %% | "
                  "tilt removed first: %+5.1f ms [%+6.1f, %+6.1f], fallback %3.0f %%   (%.0f s)"
                  % (sb, S, float(np.median(dec_dev)), 100.0 * dec_fall / CFG["n_cells"], float(np.median(var_dev)),
                     *np.percentile(var_dev, [5, 95]), 100.0 * share, time.time() - t_start))
    print("  ".join("%s %s" % (k, "PASS" if v else "FAIL") for k, v in ok.items()) + "   (%.0f s)" % (time.time() - t_start))
    return all(ok.values())


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
