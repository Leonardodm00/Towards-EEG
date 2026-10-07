"""Reproducer (tester finding F5): pixels outside the image come back white (255) in a VALID plane, so a
node near the image border reads them as bright tissue: no `profile_nan`, B_bar inflated.
Run from 'Passive Features/Diameter Re-measurement':  python tests/spec/repro_image_edge.py"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import allen_image_io as aio  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.analysis import node_pipeline as NP  # noqa: E402

cfg = default_config()
P = cfg.acquisition.res0_um
img = np.full((400, 400), 200, np.uint8)
img[:, 393:398] = 120                       # a dark process 2 px from the right edge of a 400 x 400 image
fetcher = aio.SyntheticFetcher(img)
planes = pd.DataFrame(dict(plane_index=np.arange(20), id=np.arange(100, 120)))


def provider(left, top, width, height, k_lo, k_hi):
    return aio.fetch_zblock(fetcher, planes, k_lo, k_hi, left, top, width, height, P)


xyz = np.array([[395 * P, (150 + 10 * i) * P, 10 * 0.28] for i in range(10)])
r = NP.measure_node(NP.Branch.from_points(xyz, 0.3), 5, provider, cfg)
blk = provider(380, 190, 40, 20, 10, 10)
print("plane valid:", blk[2], "| last columns (outside the image):", blk[0][0, 0, -5:])
print("flags:", r.flags, "| B_bar:", r.B_bar, "(tissue background is 200)")
