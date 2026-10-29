"""Reproducer (tester, 2026-10-28): the stripe removes the node's own cylinder
around the TRACE (the branch's segments, 10.S "removed the same way" as a
neighbour) while the walk starts at the pilot's FITTED centre. When the trace
lies off the fitted centre (Allen's trace beside the imaged dendrite), the
origin column is not own-shaded: at 0.4 um the run starts at the dendrite's
core, which enters the background's support (status ok); from about 0.5 um
the in-focus stripe is side_short and the node fails as not_isolated.
Noise-free phantom (d 0.25 um, alpha 0.3), patch +-12 um; square shown for
comparison (its 3-sigma far mask is also centred on the trace).

Run (from the workstream directory):
    python tests/spec/repro_trace_offset.py
"""
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_helpers  # noqa: E402,F401
from test_spec_block10_boundaries import rendered_phantom  # noqa: E402

from allen_diameter.analysis import calibration as CAL  # noqa: E402
from allen_diameter.analysis.node_pipeline import Branch  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402

if __name__ == "__main__":
    cfg0 = default_config()
    cfg0 = dataclasses.replace(cfg0, renderer=dataclasses.replace(cfg0.renderer, noise_sd_gl=0.0))
    for support in ("stripe", "square"):
        cfg = dataclasses.replace(cfg0, calibration=dataclasses.replace(cfg0.calibration, background_support=support,
                                                                        patch_half_um=12.0))
        for e in (0.0, 0.4, 0.6):
            branch, i, provider, draw = rendered_phantom(cfg, 11, 0.25, 0.3, 0.4)
            nv = np.array([-math.sin(draw.theta_rad), math.cos(draw.theta_rad), 0.0])
            trace = Branch(branch.ids, branch.types, branch.xyz_um + e * nv[None, :], branch.radius_um, branch.s_um)
            res, scan, reasons = CAL.scan_node(trace, i, provider, cfg)
            if scan is None:
                print("%s, trace %.1f um off: not scanned %s" % (support, e, reasons))
                continue
            q = int(np.flatnonzero(scan.ks == scan.k_star)[0])
            st = scan.stripe[q] if scan.stripe else None
            print("%s, trace %.1f um off: B_hat(k*) %.3f gl (true 210), omega(k*) %.5f um^2, stripe at k* %s" % (
                support, e, scan.B_bar[q], scan.omega_um2[q],
                None if st is None else "%s, own cylinder -%.3f/+%.3f um" % (st["reason"] or "ok", st["v_own_minus_um"],
                                                                            st["v_own_plus_um"])))
