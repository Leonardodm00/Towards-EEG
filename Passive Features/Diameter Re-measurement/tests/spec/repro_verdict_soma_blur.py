"""Reproducer (tester, 2026-10-28): the stripe's node verdict before the fetch
(calibration_reasons -> trace_reasons -> stripe_reasons) shades a soma disc at
the in-focus blur sigma_r(0), while the scan (patch_planes over the stage)
shades it at the stage's largest blur, as SPEC 10.S requires ("a soma disc as
a tube of radius r_s at the stage's largest blur (10.I's convention)"; "the
node's isolation verdict before any fetch is the same test at the plane
nearest the pilot's depth"). A node beside a soma passes the verdict, is
fetched and scanned, and every one of its planes is bg_few.

Run (from the workstream directory):
    python tests/spec/repro_verdict_soma_blur.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_helpers  # noqa: E402,F401
from test_spec_block10_boundaries import rendered_phantom  # noqa: E402

from allen_diameter.analysis import calibration as CAL  # noqa: E402
from allen_diameter.analysis.plane_iteration import Scene  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402


def run(which, cfg, d, soma_v):
    branch, i, provider, draw = rendered_phantom(cfg, 11, d, 0.3, 0.0)
    own = Scene.from_branch(branch)
    # a traced soma disc (radius 2 um) beside the node, soma_v um across the measuring line (theta = 0: along -y)
    soma = np.array([[branch.xyz_um[i, 0], branch.xyz_um[i, 1] + soma_v, 2.0]])
    scene = Scene(own.seg_xyz, own.seg_radius, own.seg_ids, soma)
    res, scan, reasons = CAL.scan_node(branch, i, provider, cfg, scene=scene)
    print("%s set: Allen 2r %.2f um, soma %.1f um across the line" % (which, d, abs(soma_v)))
    print("  verdict before the fetch: %s" % (CAL.scan_verdict(scan, reasons)))
    if scan is not None:
        q = int(np.argmin(np.abs(np.asarray(scan.z_um) - res.z_sub_um)))
        st = scan.stripe[q]
        print("  scan's stripe at the pilot's plane k=%d: %s (side -%.2f / +%.2f um, %d px)"
              % (int(scan.ks[q]), st["reason"] or "usable", st["side_minus_um"], st["side_plus_um"], st["n_pixels"]))
        print("  plane statuses: %s" % (dict((s, list(scan.status).count(s)) for s in set(scan.status))))
        print("  planes with omega: %d of %d" % (int(np.isfinite(scan.omega_um2).sum()), len(scan.status)))
    return res, scan, reasons


if __name__ == "__main__":
    cfg = default_config()
    run("near", cfg, 0.2, -6.4)
    cfg_far = dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, patch_half_um=12.0,
                                                                       far_offsets_planes=8))
    run("far", cfg_far, 0.7, -8.0)
