#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-19: how wide can the in-focus dip of a tube of ALLEN's diameter be,
under the widest in-focus kernel that the clear node 5353 allows, compared with the recorded dips?

For each node: the pipeline's forward (node_inputs.forward_factory, the run's rays) renders one tube of Allen's
diameter with its axis at the depth of the node's narrowest recorded dip (that plane in focus; the dip searched
within max(0.5 um, 0.6 x Allen's diameter) of the line's centre, so that a neighbour's dip is not taken), lateral centre as
fitted in the run, absorption mu from faint to nearly opaque; the full width at half depth (linear interpolation)
of the rendered dip in that plane is compared with the recorded one. Kernels: the run's Gaussian table (in-focus SD
0.080 um) and the same table re-anchored at 0.25 um (kernel.with_in_focus_width), the in-focus SD at which a
Gaussian of FWHM 0.59 um -- node 5353's narrowest recorded dip -- is reached; and two heavy-tailed kernels at the
run's in-focus SD, (1 - f) T + f T_h with T_h the table with every SD multiplied by h (the image is linear in the
kernel, so the mixture of two renderings is the rendering of the mixture).
Run:  python3 -I allen_width_bound_check.py <code folder> <cleaning dir> <inference dir>
"""
import dataclasses
import json
import os
import sys
import types

import numpy as np

WS = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(WS, "src"))
sys.path.insert(0, os.path.join(WS, "scripts"))
import infer_nodes as INF                                  # noqa: E402
from allen_diameter.analysis import node_inputs            # noqa: E402
from allen_diameter.model import kernel, line_fast         # noqa: E402

CLEAN_DIR, INFER_DIR, SPEC = sys.argv[2], sys.argv[3], "529878215"
MUS = (0.3, 1.0, 3.0, 10.0)                               # 1/um, faint to nearly opaque
S0S = (0.080, 0.25)
HALOS = ((0.3, 3.0), (0.5, 6.0))                          # (1 - f) T + f T_h at the run's in-focus SD 0.080 um


def scaled(ren, h):
    """The table with every SD and the continuation slope multiplied by h (the halo's width)."""
    return dataclasses.replace(ren, sigma_r0_um=h * ren.sigma_r0_um,
                               kernel_table_sigma_um=tuple(h * s for s in ren.kernel_table_sigma_um),
                               kernel_continuation_slope=h * ren.kernel_continuation_slope)


def fwhm(D, v, win):
    """Full width at half depth of the dip D (gl or dimensionless) about its deepest sample within |v| <= win (um)."""
    near = np.flatnonzero(np.abs(v) <= win)
    n0 = int(near[np.argmax(D[near])])
    half = 0.5 * D[n0]

    def cross(step):
        n = n0
        while 0 <= n + step < v.size and D[n + step] > half:
            n += step
        if not 0 <= n + step < v.size:
            return np.nan
        a, b = D[n], D[n + step]
        return v[n] + (v[n + step] - v[n]) * (a - half) / (a - b)
    return cross(+1) - cross(-1), D[n0]


def main():
    rec = json.load(open(os.path.join(CLEAN_DIR, "cleaning_%s.json" % SPEC)))
    inf = json.load(open(os.path.join(INFER_DIR, "inference_%s.json" % SPEC)))
    cfg = INF.run_config(rec, "", "")
    for nid in ("5353", "2669", "1013", "2620", "639"):
        info, run = rec["nodes"][nid], inf["nodes"][nid]
        pil = info["pilot"]
        pilot = types.SimpleNamespace(**{k: float(pil[k]) for k in ("cx_um", "cy_um", "theta_rad", "phi_rad",
                                                                       "z_sub_um", "d_hat_um", "mu_hat_per_um")})
        with np.load(os.path.join(CLEAN_DIR, "cleaning_%s_node%s.npz" % (SPEC, nid)), allow_pickle=False) as z:
            node = INF.node_from_npz(int(nid), z, None)
            P = np.array(z["profiles_after"], dtype=float)
        d_a = float(info["d_allen_um"])
        win = max(0.5, 0.6 * d_a)                         # um: the node's own dip, not a neighbour's
        rec_w = [fwhm(node.B_gl[j] - P[j], node.v, win) for j in range(len(node.ks))]
        j = int(np.nanargmin([w for w, _ in rec_w]))
        line = "node %s: Allen %.3f um; recorded narrowest dip k %d: FWHM %.3f um, depth %.0f gl |" % (
            nid, d_a, int(node.ks[j]), rec_w[j][0], rec_w[j][1])
        U = tuple(float(x) for x in info["U_axis_um"])

        def fwd_for(ren):
            return node_inputs.forward_factory(pilot, U, node, dataclasses.replace(cfg, renderer=ren),
                                               line_fast.SectionCache())
        kernels = [("s0 %.3f" % s0, fwd_for(kernel.with_in_focus_width(cfg.renderer, s0)), None, 0.0) for s0 in S0S]
        for f, h in HALOS:
            kernels.append(("halo f %.1f h %.0f" % (f, h), fwd_for(cfg.renderer), fwd_for(scaled(cfg.renderer, h)), f))
        for name, fa, fb, f in kernels:
            ws = []
            for mu in MUS:
                args = (d_a, mu, float(run["theta"][2]), float(node.planes_z[j]))
                tau = fa(*args) if fb is None else (1.0 - f) * fa(*args) + f * fb(*args)
                w, dep = fwhm(1.0 - tau[j], node.v, win)
                ws.append("%.2f" % w)
            line += "\n    %-16s Allen-tube FWHM %s um (mu %s /um)" % (name, "/".join(ws), "/".join("%g" % m for m in MUS))
        print(line, flush=True)


if __name__ == "__main__":
    main()
