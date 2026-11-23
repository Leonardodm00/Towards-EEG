#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-23 (D-222): are the results of (a) and (b) properties of the
kernels or of the line renderers' discretisation? The Gaussian fit's noise-free bias in d (part (b)) and the zf
model's SE(d) inflation and corr(d, shape) (part (a)) recomputed with the slabs halved (renderer.dzeta_um 0.01 um)
and with the fine lattice halved (inference.line_fine_factor 16), against the defaults (0.02 um, 8), for the three
SGT truths at d = 0.4 and 2.5 um, mu 1.5 /um, the axis 0.5 um off k*.

Run (a few minutes):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 discretization_check.py > discretization_check.out
Writes discretization_check.json. Pure ASCII, LF.
"""
from __future__ import annotations

import json
import sys

import numpy as np

import cases as C
import fitting as FT
import sgt_line_model as M
from allen_diameter.analysis import node_inference
from allen_diameter.config import default_config, with_overrides

VARIANTS = (("default (dzeta 0.02, f 8)", {}), ("dzeta 0.01", {"renderer.dzeta_um": 0.01}),
            ("f 16", {"inference.line_fine_factor": 16}))


def main():
    rows = []
    for label, ov in VARIANTS:
        st = M.make_setup(with_overrides(default_config(), ov) if ov else default_config())
        for d in (0.4, 2.5):
            g = dict(d=d, mu=1.5, z_ax=0.5, v0=C.V0_UM)
            for truth in ("heavy", "b13", "skew"):
                phi_t = C.phi_true(g, truth)
                tspec = M.ModelSpec(free=("beta", "xi", "zeta"), method="exact")
                t_true = M.tau_phi(phi_t, tspec, st)
                planes = node_inference.select_planes(t_true, st.B_gl, st.sigma_gl, st.cfg)
                r = FT.fit(st.B_gl * t_true, M.ModelSpec(), st, phi_t[:4], planes)
                fk = M.fisher(phi_t[:4], M.ModelSpec(free=(), fixed_shape=C.TRUTHS[truth]), st, planes=planes)
                ff = M.fisher(phi_t, tspec, st, planes=planes)
                rows.append(dict(variant=label, d=d, truth=truth, gauss_bias_d_rel=float((r["phi"][0] - d) / d),
                                 gauss_status=r["status"], se_d_known=float(fk["se"][0]), se_d_zf=float(ff["se"][0]),
                                 infl_zf=float(ff["se"][0] / fk["se"][0]),
                                 corr_d_beta=float(ff["corr"][0, 4]), corr_d_xi=float(ff["corr"][0, 5]),
                                 corr_d_zeta=float(ff["corr"][0, 6])))
                c = rows[-1]
                print("%-26s d %.1f %-5s Gaussian-fit bias %+.3f %%; SE(d) known %.5f, zf %.5f (infl %.3f); "
                      "corr(d, beta, xi, zeta) %+.3f %+.3f %+.3f" % (label, d, truth, 100 * c["gauss_bias_d_rel"],
                                                                    c["se_d_known"], c["se_d_zf"], c["infl_zf"],
                                                                    c["corr_d_beta"], c["corr_d_xi"],
                                                                    c["corr_d_zeta"]))
                sys.stdout.flush()
    (M.HERE / "discretization_check.json").write_text(json.dumps(rows, indent=1))
    print("wrote discretization_check.json")


if __name__ == "__main__":
    main()
