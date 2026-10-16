#!/usr/bin/env python3
"""Example curves for the three-shape-parameter figure (2026-10-15): the
noise-free thin Gaussian truth (d 0.6 um) and its best impostor at half the
diameter (rel 0.5), and the thick Gaussian truth (d 1.5 um) with its best
impostor at rel 0.8. Reads identifiability_sgt.json, writes example_sgt.npz.

    python identifiability_sgt_example.py --repo "<repo>/..." --res <dir>/identifiability_sgt.json
Pure ASCII.
"""
import argparse
import json
import os

import numpy as np

import check_objective as CO
import identifiability_sgt as IS

PICKS = (("thin", 0.6, "G", 0.5), ("thick", 1.5, "G", 0.8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--res", required=True)
    a = ap.parse_args()
    P = CO.load(a.repo)
    c = IS.CONFIG
    R = json.load(open(a.res))
    vv = np.linspace(-3.0, 3.0, 601)
    out = dict(vv=vv)
    for tag, dt, kname, rel in PICKS:
        t = [x for x in R["impostors"] if abs(x["d_true"] - dt) < 1e-9 and x["kernel"] == kname][0]
        r = [x for x in t["rows"] if abs(x["rel"] - rel) < 1e-9][0]
        sh = c["kernels"][kname]
        out[tag + "_true"] = IS.model(P, vv, dt, c["mu"], c["v0"], c["w"], *sh)
        out[tag + "_imp"] = IS.model(P, vv, r["d"], r["mu"], r["v0"], r["w"], r["p"], r["r"], r["gamma"])
        v = np.array(t["v"])
        out[tag + "_diff"] = IS.model(P, v, r["d"], r["mu"], r["v0"], r["w"], r["p"], r["r"], r["gamma"]) - np.array(t["I_true"])
        out["v_samples"] = v
        out[tag + "_meta"] = np.array([dt, r["d"], r["mu"], r["v0"], r["w"], r["p"], r["r"], r["gamma"],
                                       r["dchi2"]["1"], r["max_abs_gl"], rel])
        print(tag, out[tag + "_meta"].round(4).tolist())
    np.savez(os.path.join(os.path.dirname(os.path.abspath(a.res)), "example_sgt.npz"), **out)


if __name__ == "__main__":
    main()
