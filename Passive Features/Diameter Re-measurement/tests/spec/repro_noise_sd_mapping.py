"""Reproducer (tester, 2026-11-27 22:11; report N2): the twin's measured background SD against the SD put into the
camera chain (D-270.2's mapping: input = the real scan's measured bg_sd_gl x noise_scale), default camera (8 bit,
JPEG on), a 0.4 um near phantom on a +-8 um patch, B 200 gl. Run from the package root:

    python3 tests/spec/repro_noise_sd_mapping.py

Pure ASCII.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import test_spec_block55_step2_measure as T55  # noqa: E402
from allen_diameter.analysis import step2_measure as M  # noqa: E402

cfg, st = T55._near()
print("jpeg", cfg.renderer.jpeg, "bit_depth", cfg.renderer.bit_depth)
for sd in (0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 3.0, 5.0):
    t = T55._twin(st.ks, np.full(7, 200.0), np.full(7, sd))
    sc = M.measure_twin(t, st, cfg, cfg.step2)
    print("SD in %.1f gl -> measured bg_sd_gl median %.3f gl" % (sd, float(np.nanmedian(sc.bg_sd_gl))))
