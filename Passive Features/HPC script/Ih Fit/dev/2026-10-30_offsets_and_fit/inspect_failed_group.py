"""inspect_failed_group.py -- standalone check (not pipeline code), 2026-10-30:
the one group of check_fit_profile.py (1200 simulated groups) whose exact decay
fit failed (RSS(tau) with no interior minimum in [1, 1000] ms), inspected, and
refitted with each window's estimated tilt offset removed first (the proposal
of check_offsets_mechanism.py).

Replays check_fit_profile.py's random stream (seed 20261052: the worked case,
150 untilted cells, then the tilted cells) without its expensive parts, finds
every group of the tilted cells whose fit is at a bound, prints its profile at a
few tau and its windows' positions, then fits the same group after removing
b_hat_s (t_k - t_bar_s) from every window of its sweep (b_hat_s by (TL1) on the
windows cut at 80 ms).
No prediction: an inspection of a case found by check_fit_profile.py.
Run (from this folder; about 1 min):  python inspect_failed_group.py
Pure ASCII, LF.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_fit_profile as cfp            # noqa: E402  (its CFG, simulator access, fit window)
import check_offsets_mechanism as com      # noqa: E402  (remove_offsets, estimated_tilts)

csn, nw = cfp.csn, cfp.nw
TAUS = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])


def main():
    rng = np.random.default_rng(cfp.CFG["seed"])
    resp = csn.train_response()
    csn.simulate_cell(cfp.CFG["sweeps"], 0.0, resp, rng)                   # the worked case
    for _ in range(cfp.CFG["n_cells"]):
        csn.simulate_cell(cfp.CFG["sweeps"], 0.0, resp, rng)               # the untilted cells
    found = 0
    for c in range(cfp.CFG["n_cells"]):
        ss = csn.simulate_cell(cfp.CFG["sweeps"], cfp.CFG["sigma_b"][1], resp, rng)
        t_ms, sel = cfp.fit_window(ss)
        tilts = com.estimated_tilts(ss)
        for s in np.unique(ss.sweep_number):
            for dep in (True, False):
                idx = np.flatnonzero((ss.sweep_number == s) & (ss.polarity_is_dep == dep))
                if idx.size < nw.WindowConfig().min_windows_per_group:
                    continue
                f = nw.fit_decay(t_ms[sel], np.median(ss.v[idx], axis=0)[sel])
                if not f.at_bound:
                    continue
                found += 1
                y = np.median(ss.v[idx], axis=0)[sel]
                print("tilted cell %d, sweep %d, %s group, K = %d (positions %s); b_hat of the sweep %.3f mV/s"
                      % (c, s, "dep" if dep else "hyp", idx.size, ss.position[idx].tolist(), tilts[int(s)]))
                print("  plain median: %s; RSS(tau) at tau = %s ms: %s mV^2"
                      % (f.message, TAUS.tolist(), np.round([nw.profile_decay(t_ms[sel], y, x)[2] for x in TAUS], 3).tolist()))
                cell = com.remove_offsets(ss, tilts)
                g = nw.fit_decay(t_ms[sel], np.median(cell.v[idx], axis=0)[sel])
                print("  offsets removed first: ok %s, tau_hat %.2f ms, A_hat %.4f mV, RSS %.3f mV^2"
                      % (g.ok, g.tau_ms, g.A_mV, g.rss))
    print("groups at a bound among the tilted cells: %d" % found)


if __name__ == "__main__":
    main()
