"""Block 13.5: the wall time of one realization of the table's chain (D-062's
sizing), on rendered phantoms over d and phi at the default configuration.

For each (d, phi) of the grid, replicate 0's draw of phantoms.draw_replicate
with d and phi set (theta, mu, the sub-pixel offset and the depth jitter
drawn as the table draws them, seed SEED) [2026-10-21, Block 14.2: draw 1,
realization 1 of the new design, its branch radius set to d/2 as in 13.5's
run -- the timings of the .log were taken before 14.2], the phantom tube and branch of
Block 6, and measure_node through a provider that renders with
render.synthetic_block -- run_replicate's chain, timed in parts: the
rendering (the provider calls: the one request and the restacks), the SGT
fits (plane_iteration.sgt_at_plane, wrapped), and the rest (the pilot, the
gradient energies, the bookkeeping). Prints one line per phantom and the
totals; nothing is asserted (a measurement, not a check).

Run (from the workstream folder; on an idle machine, the numbers are wall
times):
    python checks/block13_2026-10-20/time_13_5.py

Pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
if str(WS / "src") not in sys.path:
    sys.path.insert(0, str(WS / "src"))

SEED = 20261020
D_UM = (0.3, 0.6, 1.2, 2.4, 3.5)
PHI_DEG = (0.0, 45.0)


def main():
    from allen_diameter.analysis import node_pipeline as NP
    from allen_diameter.analysis import phantoms
    from allen_diameter.config import default_config
    from allen_diameter.model import render
    cfg = default_config()
    clock = dict(render=0.0, sgt=0.0, n_sgt=0, n_calls=0)
    sgt = NP.sgt_at_plane

    def timed_sgt(cur, c, k):
        t0 = time.perf_counter()
        out = sgt(cur, c, k)
        clock["sgt"] += time.perf_counter() - t0
        clock["n_sgt"] += out.get("plane_fit") is not None
        return out
    NP.sgt_at_plane = timed_sgt
    rows = []
    try:
        for d in D_UM:
            for phi_deg in PHI_DEG:
                draw, rng = phantoms.draw_replicate(cfg, SEED, 1, 1)
                draw = dataclasses.replace(draw, d_um=float(d), phi_rad=math.radians(phi_deg), radius_um=0.5 * float(d))
                tube = phantoms.phantom_tube(draw, cfg)
                branch = phantoms.phantom_branch(draw, cfg, rng)
                pad_px = int(math.ceil(cfg.renderer.pad_um / cfg.acquisition.res0_um))

                def provider(left, top, width, height, k_lo, k_hi):
                    t0 = time.perf_counter()
                    out = render.synthetic_block(tube, draw.mu_per_um, np.arange(k_lo, k_hi + 1), left, top, width,
                                                 height, cfg, rng, pad_px)
                    clock["render"] += time.perf_counter() - t0
                    clock["n_calls"] += 1
                    return out
                for key in clock:
                    clock[key] = 0 if key.startswith("n_") else 0.0
                t0 = time.perf_counter()
                res = NP.measure_node(branch, cfg.phantom.nodes_each_way, provider, cfg)
                total = time.perf_counter() - t0
                rest = total - clock["render"] - clock["sgt"]
                rows.append((total, clock["render"], clock["sgt"], rest))
                print("d %.1f um phi %2.0f deg (mu %.2f /um): %6.1f s = render %6.1f s (%d calls) + SGT fits %5.1f s "
                      "(%d fits) + rest %5.1f s | k* %d, k_iter %d, %s, %d rounds, %d restacks, d_hat/d %.3f, %s"
                      % (d, phi_deg, draw.mu_per_um, total, clock["render"], clock["n_calls"], clock["sgt"],
                         clock["n_sgt"], rest, res.k_star, res.k_iter, res.iter_status, res.n_rounds, res.n_restacks,
                         res.d_hat_um / d, res.fit_status), flush=True)
    finally:
        NP.sgt_at_plane = sgt
    a = np.array(rows)
    print("over %d phantoms: mean %.1f s per realization (render %.1f, SGT %.1f, rest %.1f), median %.1f s, max %.1f s"
          % (a.shape[0], a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean(), np.median(a[:, 0]),
             a[:, 0].max()))


if __name__ == "__main__":
    main()
