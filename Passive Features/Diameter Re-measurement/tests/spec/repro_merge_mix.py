"""Reproducer (tester finding F6): `build_table.py merge` takes every rows_<estimator hash>_*.csv in the
directory, whatever simulator configuration or seed produced it, and records the merge-time
full_signature. Run from 'Passive Features/Diameter Re-measurement':  python tests/spec/repro_merge_mix.py"""
import dataclasses
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import table_io  # noqa: E402
import build_table  # noqa: E402

A = default_config()
B = dataclasses.replace(A, renderer=dataclasses.replace(A.renderer, absorption="ray_world", noise_sd_gl=0.5),
                        phantom=dataclasses.replace(A.phantom, d_range_um=(1.0, 4.0), seed=1))
assert A.signature_hash("estimator") == B.signature_hash("estimator")
d = tempfile.mkdtemp()
rng = np.random.default_rng(0)
for cfg, start, ratio in ((A, 0, 1.10), (B, 100, 1.30)):
    rows = [dict(index=start + i, seed=cfg.phantom.seed, d_um=float(np.exp(rng.uniform(np.log(0.2), np.log(4.0)))),
                 phi_rad=float(rng.uniform(0, 1.5)), meas_phi_rad=0.1, ratio=ratio, in_S=True, reject="")
            for i in range(30)]
    table_io.write_rows(rows, os.path.join(d, "rows_%s_%06d_%06d.csv" % (cfg.signature_hash("estimator"), start,
                                                                          start + 30)))
stem = build_table.merge(A, d)
meta = json.load(open(stem + ".json"))
print("rows merged:", meta["n_all"], "| recorded renderer.absorption:", meta["signature"]["renderer"]["absorption"],
      "| the second chunk was rendered with ray_world, seed 1, d in [1, 4]")
