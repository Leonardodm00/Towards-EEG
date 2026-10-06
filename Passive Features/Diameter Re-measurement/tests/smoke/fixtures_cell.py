"""Shared fixture of the smoke tests: a synthetic cell served like Allen's planes.

synthetic_cell writes an SWC (a soma and one straight flat dendrite), renders
the dendrite as a tube over the region (Block 4, full camera chain) and returns
an allen_image_io.ImageFetcher that serves the rendered planes by plane index,
so that allen_image_io.fetch_zblock, run_cell.real_provider and the
2026-09-23 registration code read it exactly as they read the archive.

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.loading import swc_io  # noqa: E402
from allen_diameter.model import camera  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402


def synthetic_cell(tmp, cfg, d_true=0.8, mu=1.0, n_nodes=6, step=1.18, theta=0.3, allen_radius=0.3, seed=20261006,
                   ks=None):
    """(SWC, fetcher, planes DataFrame, swc path): soma id 1 at the origin, dendrite nodes ids 2..n_nodes+1 from
    (6.0, 0.03, 0.05) um every `step` um at heading theta, flat; planes ks (default -7..7) at k dz."""
    import allen_image_io as aio
    import pandas as pd
    px = cfg.acquisition.res0_um
    c0 = np.array([6.0, 0.03, 0.05])
    t = np.array([math.cos(theta), math.sin(theta), 0.0])
    pts = c0[None, :] + (np.arange(n_nodes) * step)[:, None] * t[None, :]
    path = os.path.join(tmp, "cell.swc")
    with open(path, "w") as f:
        f.write("# synthetic\n1 1 0.0 0.0 0.0 4.0 -1\n")
        for k, q in enumerate(pts):
            f.write("%d 3 %.4f %.4f %.4f %.4f %d\n" % (k + 2, q[0], q[1], q[2], allen_radius, k + 1))
    mid = pts.mean(axis=0)
    tube = G.Tube(tuple(mid), 0.5 * d_true, 0.0, theta, 1.0, 0.5 * (n_nodes - 1) * step + 6.0, "axial")
    lo = np.floor((pts[:, :2].min(axis=0) - 6.0) / px).astype(int)
    hi = np.ceil((pts[:, :2].max(axis=0) + 6.0) / px).astype(int)
    left, top, width, height = int(lo[0]), int(lo[1]), int(hi[0] - lo[0]), int(hi[1] - lo[1])
    ks = np.arange(-7, 8) if ks is None else np.asarray(ks)
    f8 = R.fine_factor(d_true, cfg.renderer, px)
    grid, inner = R.block_fine_grid(left, top, width, height, px, f8, 8)
    res = R.render_transmittance(tube, mu, ks * cfg.acquisition.dz_um, grid, cfg.renderer, +1,
                                 reduce=lambda pl: R.pixel_integrate(pl[inner[0], inner[1]], f8))
    planes8 = camera.camera_chain(res.tau, cfg.renderer, np.random.default_rng(seed))

    class PlaneFetcher(aio.ImageFetcher):
        def get(self, image_id, left_, top_, width_, height_, downsample=0):
            out = np.full((height_, width_), 255, dtype=np.uint8)
            img = planes8[int(image_id) - int(ks[0])]
            x0, y0 = max(left_, left), max(top_, top)
            x1, y1 = min(left_ + width_, left + width), min(top_ + height_, top + height)
            if x1 > x0 and y1 > y0:
                out[y0 - top_:y1 - top_, x0 - left_:x1 - left_] = img[y0 - top:y1 - top, x0 - left:x1 - left]
            return out

    return swc_io.read_swc(path), PlaneFetcher(), pd.DataFrame({"plane_index": ks, "id": ks}), path
