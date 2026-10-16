#!/usr/bin/env python3
"""Independent check of the most striking impostors (2026-10-15): recompute the
truth and impostor profiles with a direct adaptive quadrature of

    I(v) = B_bar [1 - Int_{-d/2}^{d/2} (1 - exp(-alpha sqrt(1 - (2u/d)^2))) k(v - v0 - u) du],
    alpha = mu d / cos(phi),

(scipy.integrate.quad per offset; no Gauss-Legendre rule, no pipeline code
besides the offsets) and compare with tube_model.model_profile_kernel at 848
nodes, then recompute Delta chi^2 = sum_n (I_imp - I_true)^2 / sigma^2 at 1 gl.

Tolerance, fixed before the run: the two implementations agree to 1e-4 gl per
sample, and the recomputed Delta chi^2 agrees with identifiability_sgt.json to
1e-3 absolute (both far below the decision threshold of 1).

    python independent_model_check.py --repo "<repo>/..." --res <dir>/identifiability_sgt.json
Pure ASCII.
"""
import argparse
import json
import math

import numpy as np
from scipy import integrate

import check_objective as CO
import identifiability_sgt as IS
import sgt_kernel as SK

CASES = (("0.6", "G", 0.5), ("0.6", "S4", 0.8), ("1.5", "G", 0.8), ("1.5", "L", 0.95))


def profile_direct(v, d, mu, v0, w, p, r, gamma, phi, B):
    k = SK.sgt_pdf(w, p, r, gamma)
    alpha = mu * d / math.cos(phi)

    def integrand(u, vn):
        s = math.sqrt(max(0.0, 1.0 - (2.0 * u / d) ** 2))
        return -math.expm1(-alpha * s) * float(k(vn - v0 - u))
    out = []
    for vn in v:
        val = integrate.quad(integrand, -d / 2.0, d / 2.0, args=(vn,), epsabs=1e-13, epsrel=1e-11, limit=500)[0]
        out.append(B * (1.0 - val))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--res", required=True)
    a = ap.parse_args()
    P = CO.load(a.repo)
    c = IS.CONFIG
    R = json.load(open(a.res))
    v = P.profiles.profile_offsets(P.cfg)
    ok = True
    for dt, kname, rel in CASES:
        t = [x for x in R["impostors"] if abs(x["d_true"] - float(dt)) < 1e-9 and x["kernel"] == kname][0]
        r = [x for x in t["rows"] if abs(x["rel"] - rel) < 1e-9][0]
        sh = c["kernels"][kname]
        tr_pipe = IS.model(P, v, float(dt), c["mu"], c["v0"], c["w"], *sh)
        im_pipe = IS.model(P, v, r["d"], r["mu"], r["v0"], r["w"], r["p"], r["r"], r["gamma"])
        tr_dir = profile_direct(v, float(dt), c["mu"], c["v0"], c["w"], *sh, c["phi"], c["B"])
        im_dir = profile_direct(v, r["d"], r["mu"], r["v0"], r["w"], r["p"], r["r"], r["gamma"], c["phi"], c["B"])
        e1, e2 = np.max(np.abs(tr_pipe - tr_dir)), np.max(np.abs(im_pipe - im_dir))
        dchi_dir = float(np.sum((im_dir - tr_dir) ** 2))
        good = e1 < 1e-4 and e2 < 1e-4 and abs(dchi_dir - r["dchi2"]["1"]) < 1e-3
        ok &= good
        print("d_true %s %-2s rel %.2f: impostor d %.3f | pipeline vs direct: truth %.1e, impostor %.1e gl | "
              "Delta chi^2(1 gl) direct %.4f, run %.4f | max|dI| direct %.3f gl  %s"
              % (dt, kname, rel, r["d"], e1, e2, dchi_dir, r["dchi2"]["1"], np.max(np.abs(im_dir - tr_dir)),
                 "PASS" if good else "FAIL"))
    print("ALL PASS" if ok else "SOME FAIL")


if __name__ == "__main__":
    main()
