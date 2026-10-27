"""Rendered planes of the hybrid at 8 x 16, 16 x 32 and 32 x 64 directions (flat and tilted 1 um tubes,
mu 1 and 3 /um), max |tau - tau(32 x 64)| over the grid. Not pipeline code."""
import sys, math, time, dataclasses
sys.path.insert(0, sys.argv[1])
import numpy as np
from allen_diameter.config import default_config
from allen_diameter.model import geometry as G, render as R
rc = default_config().renderer
PX = 0.1144
m = int(round(2.0 / (PX / 4))); n = 2 * m + 1
g = R.FineGrid(-m * PX / 4, -m * PX / 4, PX / 4, n, n)
zk = np.array([-0.56, -0.28, 0.0, 0.28, 0.56])
for phi in (0.0, 40.0):
    for mu in (1.0, 3.0):
        tube = G.Tube((0.0, 0.0, 0.0), 0.5, math.radians(phi), 0.3, 1.0, 10.0, "axial")
        out = {}
        for nr, npsi in ((32, 64), (16, 32), (8, 16)):
            t0 = time.time()
            out[(nr, npsi)] = R.render_transmittance(tube, mu, zk, g, dataclasses.replace(rc, absorption="hybrid", ray_n_rho=nr, ray_n_psi=npsi)).tau
            out[(nr, npsi, "t")] = time.time() - t0
        ref = out[(32, 64)]
        print("phi %2.0f mu %.0f: max |dtau| 16x32 %.1e (%.0f s), 8x16 %.1e (%.0f s); 32x64 %.0f s" % (
            phi, mu, np.abs(out[(16, 32)] - ref).max(), out[(16, 32, "t")], np.abs(out[(8, 16)] - ref).max(), out[(8, 16, "t")], out[(32, 64, "t")]), flush=True)
