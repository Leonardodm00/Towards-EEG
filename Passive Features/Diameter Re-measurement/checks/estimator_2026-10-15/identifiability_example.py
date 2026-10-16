#!/usr/bin/env python3
"""Example curves for the identifiability figure (2026-10-15): the noise-free
truth and best-impostor profiles for a thin and a thick tube at the same
relative change of d. Reads identifiability.json, writes example.npz.

Thin: the Gaussian-end truth (d 0.6 um, nu 100) and its impostor at the
largest |d/d_true - 1| with Delta chi^2 < 1 at 1 gl. Thick: the Gaussian-end
truth (d 1.5 um, nu 100) and its impostor at the same d/d_true.

    python identifiability_example.py --repo "<repo>/..." --res <dir>/identifiability.json
Pure ASCII.
"""
import argparse
import json
import os

import numpy as np

import check_objective as CO
import identifiability_check as IC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--res", required=True)
    a = ap.parse_args()
    P = CO.load(a.repo)
    c = IC.CONFIG
    R = json.load(open(a.res))
    by = {(t["d_true"], t["nu_true"]): t for t in R["impostors"]}
    thin = by[(0.6, 100.0)]
    ok = [r for r in thin["rows"] if r["dchi2"]["1"] < 1.0]
    pick = max(ok, key=lambda r: abs(r["rel"] - 1.0))
    thick = by[(1.5, 100.0)]
    pick2 = min(thick["rows"], key=lambda r: abs(r["rel"] - pick["rel"]))
    vv = np.linspace(-3.0, 3.0, 601)
    out = {}
    for tag, t, r in (("thin", thin, pick), ("thick", thick, pick2)):
        out[tag + "_true"] = IC.model(P, vv, t["d_true"], c["mu"], c["v0"], c["w"], t["nu_true"])
        out[tag + "_imp"] = IC.model(P, vv, r["d"], r["mu"], r["v0"], r["w"], r["nu"])
        out[tag + "_meta"] = np.array([t["d_true"], t["nu_true"], r["d"], r["mu"], r["v0"], r["w"], r["nu"],
                                       r["dchi2"]["1"], r["max_abs_gl"], r["rel"]])
        v = np.array(t["v"])
        out[tag + "_diff_samples"] = IC.model(P, v, r["d"], r["mu"], r["v0"], r["w"], r["nu"]) - np.array(t["I_true"])
        out["v_samples"] = v
    out["vv"] = vv
    np.savez(os.path.join(os.path.dirname(os.path.abspath(a.res)), "example.npz"), **out)
    print("thin:", out["thin_meta"].round(4).tolist())
    print("thick:", out["thick_meta"].round(4).tolist())


if __name__ == "__main__":
    main()
