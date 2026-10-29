"""Reproducer (tester, 2026-10-28): the pilot's tilt reading on thin near-set
phantoms (d = 0.2 um), against which D-101.1's near bound (10 deg) is applied.
Block 5's oracle bounds the tilt error at 4 deg for d = 1 um only.

Run (from the workstream directory):
    python tests/spec/repro_thin_tilt.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_helpers  # noqa: E402,F401
from test_spec_block10_boundaries import rendered_phantom  # noqa: E402

from allen_diameter.analysis.node_pipeline import measure_pilot  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402

if __name__ == "__main__":
    cfg = default_config()
    print("d_um  phi_true  seed  phi_pilot  error  (near bound %.0f deg)" % cfg.calibration.node_phi_max_deg)
    for phi in (6.0, 12.0):
        for seed in (41, 42, 43, 44):
            branch, i, provider, draw = rendered_phantom(cfg, seed, 0.2, 0.3, 0.7, phi_deg=phi)
            res = measure_pilot(branch, i, provider, cfg)
            ph = math.degrees(res.phi_rad)
            print("%.1f   %5.1f    %d   %6.2f    %+5.2f" % (0.2, phi, seed, ph, ph - phi), flush=True)
