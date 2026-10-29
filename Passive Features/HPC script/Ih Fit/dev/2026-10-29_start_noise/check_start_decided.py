"""check_start_decided.py -- standalone check (not pipeline code), 2026-10-29
(later, 5): the SS window start as decided and implemented after D-110 and
D-111 -- each group's decay fitted exactly (variable projection, no
starting values) and the cell's start the MEDIAN of its valid groups'
starts -- on the simulated cells of check_start_noise.py (its simulator; a
new seed, 20261043).

Rule (SPEC 2.2 (d), (e)): t_sp = tau_hat ln(|A_hat| / (kappa_tr
sigma_SS_hat)); t_star(c) = max(80 ms, median over valid groups of t_sp);
t_SS(c) = t_star(c) when 200 - t_star >= 50 ms, else 80 ms; a fit whose
profiled minimum lies at a bound of [1, 1000] ms is a failed fit, left out.

Predictions, fixed before running:
 P1  each group's start is centred on t0: the median over all valid groups
     of t_sp - t0 lies within 5 ms of 0, in every row;
 P2  so is the cell's: the median over cells of t_star - t0 lies within 5 ms
     of 0, in every row (both tilts);
 P3  the fallback applies in at most 15 % of cells, in every row (the
     median over groups fell back in 0-13 % with the three-start fit,
     check_start_options.py, V1).
Reported besides: the failed fits (minimum at a bound) per row.
Run (from this folder; about 8 min):  python check_start_decided.py
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
CFG = dict(seed=20261043, n_cells=150, p_ms=5.0, p3_share=0.15)


def main():
    t_start = time.time()
    rng = np.random.default_rng(CFG["seed"])
    resp = csn.train_response()
    ok_all = dict(P1=True, P2=True, P3=True)
    for sb in csn.CFG["sigma_b"]:
        for S in csn.CFG["sweeps"]:
            dev_g, dev_star, fall, failed, groups = [], [], 0, 0, 0
            for _ in range(CFG["n_cells"]):
                ss = csn.simulate_cell(S, sb, resp, rng)
                st = nw.decay_start(ss)
                t0 = csn.t0_of(st.sigma_ss_mV)
                ok = np.array([x == "ok" for x in st.group_status])
                groups += ok.size
                failed += int((~ok).sum())
                dev_g.extend((st.t_group_ms[ok] - t0).tolist())
                if np.isfinite(st.t_star_ms):
                    dev_star.append(st.t_star_ms - t0)
                fall += int("ss_start_fallback" in st.flags)
            med_g, med_star = float(np.median(dev_g)), float(np.median(dev_star))
            share = fall / CFG["n_cells"]
            p1, p2, p3 = abs(med_g) <= CFG["p_ms"], abs(med_star) <= CFG["p_ms"], share <= CFG["p3_share"]
            ok_all["P1"] &= p1
            ok_all["P2"] &= p2
            ok_all["P3"] &= p3
            print("sigma_b %.1f mV/s, %2d sweeps: t_sp - t0 median %+5.1f ms (SD %5.1f, 5-95 %% [%+6.1f, %+6.1f]); "
                  "t_star - t0 median %+5.1f ms [%+6.1f, %+6.1f]; fallback %3.0f %%; failed fits %d of %d groups   (%.0f s)"
                  % (sb, S, med_g, float(np.std(dev_g, ddof=1)), *np.percentile(dev_g, [5, 95]), med_star,
                     *np.percentile(dev_star, [5, 95]), 100 * share, failed, groups, time.time() - t_start))
    print("noiseless t0 at sigma_SS_hat 0.0402 mV: %.1f ms" % csn.t0_of(0.0402))
    print("  ".join("%s %s" % (k, "PASS" if v else "FAIL") for k, v in ok_all.items())
          + "   (%.0f s)" % (time.time() - t_start))
    return all(ok_all.values())


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
