#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-23 (D-222): the line renderers' point-sampled fine lattice and
the Jacobian. Blocks 26 and 35 sample the tube's cross-section at fine points h = p / f apart (f =
inference.line_fine_factor = 8). The model's values converge as f grows, but its derivative in d (and v0) at a
point is that of the sampled tube: the samples between the outermost one inside and the tube's edge, where the
chord's derivative r / sqrt(r^2 - y^2) diverges, are missing, so the local d tau / d d is low by about
(2 / pi) sqrt(h / r) on average [reasoning], and its two edges are sampled unequally (an odd part, correlated with
v0 and zeta). This script measures it: the Fisher SE(d), SE(v0), corr(d, v0), corr(d, zeta) and SE(d)'s inflation
(zf over the shape known) at f = 8, 16, 32, 64, heavy truth, mu 1.5 /um, the axis 0.5 um off k*, d = 0.4 and
2.5 um; beside them the recovery's empirical SD of d_hat (f = 8 data and fits) for the same cases.

Run (a few minutes):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 lattice_check.py > lattice_check.out
Writes lattice_check.json. Pure ASCII, LF.
"""
from __future__ import annotations

import json
import sys

import numpy as np

import cases as C
import sgt_line_model as M
from allen_diameter.config import default_config, with_overrides


def empirical_sd(d, truth, model):
    path = M.HERE / ("recovery_d%.1f.json" % d)
    if not path.exists():
        return float("nan")
    for c in json.loads(path.read_text())["cases"]:
        if c["d"] == d and c["mu"] == 1.5 and c["z_ax"] == 0.5 and c["truth"] == truth:
            return float(c["summary"][model]["d_err_sd"])
    return float("nan")


def main():
    rows = []
    truth = "heavy"
    for d in (0.4, 2.5):
        g = dict(d=d, mu=1.5, z_ax=0.5, v0=C.V0_UM)
        phi_t = C.phi_true(g, truth)
        for f in (8, 16, 32, 64):
            st = M.make_setup(with_overrides(default_config(), {"inference.line_fine_factor": f}))
            fk = M.fisher(phi_t[:4], M.ModelSpec(free=(), fixed_shape=C.TRUTHS[truth]), st)
            ff = M.fisher(phi_t, M.ModelSpec(free=("beta", "xi", "zeta")), st)
            rows.append(dict(d=d, f=f, se_d_known=float(fk["se"][0]), se_d_zf=float(ff["se"][0]),
                             se_v0_zf=float(ff["se"][2]), corr_d_v0_known=float(fk["corr"][0, 2]),
                             corr_d_v0_zf=float(ff["corr"][0, 2]), corr_d_zeta=float(ff["corr"][0, 6]),
                             corr_d_beta=float(ff["corr"][0, 4]), corr_d_xi=float(ff["corr"][0, 5]),
                             infl_zf=float(ff["se"][0] / fk["se"][0])))
            c = rows[-1]
            print("d %.1f f %2d: SE(d) known %.5f, zf %.5f (infl %.3f); SE(v0) zf %.5f; corr(d, v0) known %+.3f, "
                  "zf %+.3f; corr(d, beta, xi, zeta) %+.3f %+.3f %+.3f"
                  % (d, f, c["se_d_known"], c["se_d_zf"], c["infl_zf"], c["se_v0_zf"], c["corr_d_v0_known"],
                     c["corr_d_v0_zf"], c["corr_d_beta"], c["corr_d_xi"], c["corr_d_zeta"]))
            sys.stdout.flush()
        print("d %.1f: recovery (f = 8 data and fits), empirical SD of d_hat: gauss-truth model n/a; heavy truth z1 "
              "%.5f, zf %.5f" % (d, empirical_sd(d, truth, "z1"), empirical_sd(d, truth, "zf")))
        rows.append(dict(d=d, f="recovery_f8", sd_z1=empirical_sd(d, truth, "z1"),
                         sd_zf=empirical_sd(d, truth, "zf")))
    (M.HERE / "lattice_check.json").write_text(json.dumps(rows, indent=1))
    print("wrote lattice_check.json")


if __name__ == "__main__":
    main()
