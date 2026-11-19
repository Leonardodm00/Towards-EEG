#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-19: the line part of the clearness screen (SPEC Block 40) on the
five pilot nodes of the clear-node fit, whose state the handoff of 2026-11-19 gives by eye (section 5: 5353 clean;
2669, 1013, 639 polluted on the line; 2620 polluted in the image, by a junction).

Reads the fit's folder only (own_<S>_node<id>.npz: the recorded line profiles, backgrounds, planes; the noise SD from
inference_<S>.json), so the image part of the screen (the lines along the dendrite) is NOT run here: the patches are
in the preparation's folder. Per node and plane: clearness.dip_features on the smoothed dip; per node the largest
E and S over the in-focus planes k* - 1 .. k* + 1 against the thresholds of ClearnessConfig.
Run:  python3 -I line_screen_pilot_check.py <code folder> <inference dir>
"""
import json
import sys

import numpy as np

SPEC = "529878215"
NODES = ("5353", "2669", "1013", "2620", "639")


def main(ws, d):
    sys.path.insert(0, ws + "/src")
    from allen_diameter.analysis import clearness as CL
    from allen_diameter.config import ClearnessConfig
    q = ClearnessConfig()
    rec = json.load(open(d + "/inference_%s.json" % SPEC))
    print("thresholds: outside_max %.2f, skew_max %.2f, min_depth_sd %.0f; smoothing %.1f sample, window %.1f um"
          % (q.outside_max, q.skew_max, q.min_depth_sd, q.smooth_samples, q.window_um))
    for nid in NODES:
        with np.load(d + "/own_%s_node%s.npz" % (SPEC, nid), allow_pickle=False) as z:
            P, v, B, ks = z["profiles"], z["v"], z["B_gl"], z["ks"]
        sig = float(rec["nodes"][nid]["sigma_gl"])
        j = ks.size // 2                                   # the planes are k* - 4 .. k* + 4
        D = CL.smooth_dip(B[:, None] - P, q.smooth_samples)
        f = [CL.dip_features(D[i], v, 0.0, q.window_um, q.skew_levels, q.outside_fwhm) for i in range(ks.size)]
        focus = [i for i in range(ks.size) if abs(i - j) <= q.focus_half_planes]
        E = max(f[i]["outside"] for i in focus)
        S = max(f[i]["skew"] for i in focus)
        ok = E <= q.outside_max and S <= q.skew_max and f[j]["depth_gl"] / sig >= q.min_depth_sd
        print("node %s: k* %d, depth %.1f gl = %.0f noise SDs, W(k*) %.2f um; in focus E %.2f (x%.2f of the threshold), "
              "S %.2f (x%.2f) -> the line %s" % (nid, ks[j], f[j]["depth_gl"], f[j]["depth_gl"] / sig, f[j]["fwhm_um"], E,
                                                E / q.outside_max, S, S / q.skew_max, "passes" if ok else "FAILS"))
        print("   k: W um / E / S   " + "  ".join("%d: %.2f/%.2f/%.2f" % (ks[i], f[i]["fwhm_um"], f[i]["outside"],
                                                                         f[i]["skew"]) for i in range(ks.size)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
