"""check_noise_z.py -- the local linear table's noise against its sandwich SE, five seeds (Block 14.4;
2026-10-21; standalone).

For each seed: the smoke suite's fixture (1000 draws x 5 on gate 2's reduced ranges, c = 1) fitted twice, with
its noise and without (tau = 0, the same draws), so that b_hat - b_hat_0 is the noise alone and b_hat_0 - b the
O(h^2) smoothing bias; at the grid points inside the band (one bandwidth from every end): the SD of
z = (b_hat - b_hat_0) / SE, its share beyond 2 and its maximum; the bias over the SE; and z against b itself.

Run (from this folder):  python3 check_noise_z.py   (about a minute)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WS, "src"))
sys.path.insert(0, os.path.join(WS, "tests", "smoke"))

import test_smoke_table as T  # noqa: E402
from allen_diameter.analysis import table as TB  # noqa: E402


def main():
    cfg = T.reduced_cfg()
    for seed in range(5):
        tab = TB.fit_table(T.fixture(1000, 5, 1000 + seed, cfg), cfg)
        tab0 = TB.fit_table(T.fixture(1000, 5, 1000 + seed, cfg, tau=0.0), cfg)
        P = T.grid_points(tab)
        ins = T.inside_band(tab, cfg) & tab.supported & tab0.supported
        se = tab.arrays["se_hat"][ins]
        zn = (tab.arrays["b_hat"] - tab0.arrays["b_hat"])[ins] / se
        b = T.b_true(P[..., 0], P[..., 1], P[..., 2])[ins]
        z = (tab.arrays["b_hat"][ins] - b) / se
        bias = np.abs(tab0.arrays["b_hat"][ins] - b)
        print("seed %d, %d points: noise z SD %.3f, share > 2 %.3f, max %.2f | bias max %.4f (%.2f SE) | z vs b: "
              "share > 2 %.3f, max %.2f" % (seed, ins.sum(), zn.std(), np.mean(np.abs(zn) > 2), np.abs(zn).max(),
                                            bias.max(), (bias / se).max(), np.mean(np.abs(z) > 2), np.abs(z).max()))


if __name__ == "__main__":
    main()
