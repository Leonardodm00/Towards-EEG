#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-23 (D-222): what makes the correlation matrices of part (a)
ill-conditioned. For every geometry and truth, the zf model's Fisher information at the truth (as fisher_bias.py,
the planes Block 30 keeps): corr(beta, xi), corr(v0, zeta), corr(d, mu), and the smallest eigenvalue of the
correlation matrix with its eigenvector's two largest components and its d component.

Run (under a minute):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 corr_structure.py > corr_structure.out
Writes corr_structure.json. Pure ASCII, LF.
"""
from __future__ import annotations

import json

import numpy as np

import cases as C
import sgt_line_model as M
from allen_diameter.analysis import node_inference

NAMES = ("d", "mu", "v0", "z_ax", "beta", "xi", "zeta")


def main():
    st = M.make_setup()
    spec = M.ModelSpec(free=("beta", "xi", "zeta"), method="exact")
    rows = []
    for g in C.geometries():
        for truth in C.TRUTH_ORDER:
            phi = C.phi_true(g, truth)
            planes = node_inference.select_planes(M.tau_phi(phi, spec, st), st.B_gl, st.sigma_gl, st.cfg)
            fi = M.fisher(phi, spec, st, planes=planes)
            R = fi["corr"]
            ev, vec = np.linalg.eigh(R)
            v = vec[:, 0]
            top = np.argsort(-np.abs(v))[:2]
            rows.append(dict(geom=C.geom_id(g), truth=truth, corr_beta_xi=float(R[4, 5]), corr_v0_zeta=float(R[2, 6]),
                             corr_d_mu=float(R[0, 1]), min_eig=float(ev[0]),
                             min_eigvec_top=[NAMES[i] for i in top],
                             min_eigvec_top_abs=[float(abs(v[i])) for i in top], min_eigvec_d=float(abs(v[0]))))
            r = rows[-1]
            print("%s %-5s corr(beta, xi) %+.3f  corr(v0, zeta) %+.3f  corr(d, mu) %+.3f  min eig %.4f along %s "
                  "(%.2f, %.2f), d component %.2f" % (r["geom"], truth, r["corr_beta_xi"], r["corr_v0_zeta"],
                                                      r["corr_d_mu"], r["min_eig"], "/".join(r["min_eigvec_top"]),
                                                      r["min_eigvec_top_abs"][0], r["min_eigvec_top_abs"][1],
                                                      r["min_eigvec_d"]))
    def span(key):
        vals = [r[key] for r in rows]
        return min(vals), max(vals)
    print("ranges over the 64 cases: corr(beta, xi) %+.3f..%+.3f; corr(v0, zeta) %+.3f..%+.3f; corr(d, mu) %+.3f..%+.3f;"
          " the smallest eigenvector's d component <= %.2f; its two largest components beta/xi in %d of 64"
          % (span("corr_beta_xi") + span("corr_v0_zeta") + span("corr_d_mu")
             + (max(r["min_eigvec_d"] for r in rows), sum(set(r["min_eigvec_top"]) == {"beta", "xi"} for r in rows))))
    (M.HERE / "corr_structure.json").write_text(json.dumps(rows, indent=1))
    print("wrote corr_structure.json")


if __name__ == "__main__":
    main()
