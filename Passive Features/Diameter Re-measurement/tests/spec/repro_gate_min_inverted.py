"""Reproducer (tester finding F7): `end_to_end.py evaluate` passes a diameter when a single test node of
eight is inverted. Run from 'Passive Features/Diameter Re-measurement':  python tests/spec/repro_gate_min_inverted.py"""
import dataclasses
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.analysis import table as TB  # noqa: E402
from allen_diameter.loading import table_io  # noqa: E402
import end_to_end  # noqa: E402

cfg = default_config()
rng = np.random.default_rng(0)
d = np.exp(rng.uniform(math.log(0.2), math.log(4.0), 300))
rows = [dict(d_um=x, phi_rad=float(rng.uniform(0, 1.5)), ratio=1.1, in_S=True, reject="") for x in d]
tdir = tempfile.mkdtemp()
stem = os.path.join(tdir, "bias_table_" + cfg.signature_hash("estimator"))
table_io.save_table(TB.fit_table(rows, cfg), stem)
test = [dict(d_target_um=1.0, in_S=True, d_hat_um=1.1 if k == 0 else 50.0, meas_phi_rad=0.2) for k in range(8)]
table_io.write_rows(test, os.path.join(tdir, "test.csv"))
print("evaluate returns", end_to_end.evaluate(cfg, stem, os.path.join(tdir, "test.csv"), 0.10),
      "with 1 of 8 nodes inverted (7 out_of_domain)")
