"""Wall time of one phantom block (87 x 87 px, 17 px padding, h = p_x / 8, 7 planes)
rendered with the partition and with the hybrid (Block 20), one core. Not pipeline code."""
import sys, math, time, dataclasses
sys.path.insert(0, sys.argv[1])
import numpy as np
from allen_diameter.config import default_config
from allen_diameter.model import geometry as G, render as R

cfg = default_config()
rc = cfg.renderer
PX = cfg.acquisition.res0_um
f = 8
ks = np.arange(-3, 4)
grid, inner = R.block_fine_grid(-43, -43, 87, 87, PX, f, 17)
print("grid %d x %d samples, h = %.4f um, %d planes" % (grid.ny, grid.nx, grid.h, ks.size), flush=True)
for d, phi in ((0.3, 0.0), (1.0, 0.0), (1.0, 45.0), (2.0, 0.0)):
    tube = G.Tube((0.013, -0.021, 0.05), 0.5 * d, math.radians(phi), 0.37, 1.0, rc.U_um, rc.end_cut)
    X, Y = grid.mesh()
    lo, hi, ins = G.column_interval(X, Y, tube)
    npts = float(np.sum((hi - lo)[ins])) / rc.dzeta_um
    row = []
    for name in ("partition_vertical", "hybrid"):
        rcx = dataclasses.replace(rc, absorption=name)
        t0 = time.time()
        res = R.render_transmittance(tube, 1.0, ks * cfg.acquisition.dz_um, grid, rcx, +1,
                                     reduce=lambda p: R.pixel_integrate(p[inner[0], inner[1]], f))
        row.append(time.time() - t0)
    print("d %.1f um, phi %2.0f deg: %.2e slab points; partition %.1f s, hybrid %.1f s (%.0f ns per point-direction)"
          % (d, phi, npts, row[0], row[1], (row[1] - row[0]) / (npts * rc.ray_n_rho * rc.ray_n_psi) * 1e9), flush=True)
