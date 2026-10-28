"""make_banks.py -- data generation for check03 and check04: enlarged-window
means of simulated cells under the stationary law theta_cf alone
(sigma_b = 0), through the pipeline's simulator and window functions
(tilt_common). The tilt is added afterwards, exactly (check01 (i)).

    python make_banks.py ref       # 4 reference layouts (2/2/6/10 sweeps), ~7 min
    python make_banks.py cohort    # the 18 L3_exc layouts (K, sweeps per cell), ~8 min

Writes ../results/banks/<mode>_<name>.npz with M (n_cells, K) mV, the
layout (sweep, pos, n_sweeps) and the seed. Pure ASCII, LF.
"""
import sys
import time
from pathlib import Path

import numpy as np

import tilt_common as tc

CFG = dict(
    seed=20261026,
    ref=(("2sw_K40", 2, 40, 2000), ("2sw_K63", 2, 63, 2000), ("6sw_K199", 6, 199, 600), ("10sw_K348", 10, 348, 400)),
    n_cohort=300,
    out=Path(__file__).resolve().parent.parent / "results" / "banks",
)


def build(mode: str):
    CFG["out"].mkdir(parents=True, exist_ok=True)
    if mode == "ref":
        items = [(name, S, K, n) for name, S, K, n in CFG["ref"]]
        ss = np.random.SeedSequence([CFG["seed"], 1]).spawn(len(items))
    elif mode == "cohort":
        items = [("c%d" % spec, S, K, CFG["n_cohort"]) for spec, S, K in tc.CFG["cohort"]]
        ss = np.random.SeedSequence([CFG["seed"], 2]).spawn(len(items))
    else:
        raise SystemExit("mode must be ref or cohort")
    for (name, S, K, n), s in zip(items, ss):
        t0 = time.time()
        rng = np.random.default_rng(s)
        lay = tc.build_layout(S, K, rng, name)
        M = tc.noise_means(tc.LAW_CF, lay, n, rng)
        np.savez(CFG["out"] / ("%s_%s.npz" % (mode, name)), M=M, sweep=lay.sweep, pos=lay.pos,
                 n_sweeps=lay.n_sweeps, seed_entropy=str(s.entropy), seed_spawn=str(s.spawn_key))
        print("%s %-10s S %2d K %3d: %d cells, pairs per sweep at lag 4 %s (%.0f s)"
              % (mode, name, S, K, n, tc.pairs_per_sweep(lay, 4).tolist(), time.time() - t0), flush=True)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "ref")
