"""The tables of SPEC Block 10.S's status (2026-10-28, later) and of D-105 from the JSONL files of cap_check.py and
tau_check.py. Not pipeline code.
Usage: python summarize.py --tau-near tau_near.jsonl --tau-far tau_far.jsonl --cap-near cap_near.jsonl
       [--cap-far cap_far.jsonl]"""
import argparse
import json

import numpy as np


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def counts(cells):
    out = {}
    for c in cells:
        key = c["reason"] or "ok"
        out[key] = out.get(key, 0) + 1
    return out


def residual_line(cells, label):
    res = np.array([c["res_max"] for c in cells if c["ok"] and np.isfinite(c["res_max"])])
    if not res.size:
        return "%s: no usable node" % label
    return ("%s: residual on the line among the %d usable (gl): median %.3f, p90 %.3f, max %.3f; <= 0.03: %d, <= 0.1: "
            "%d, <= 0.3: %d, <= 1: %d" % (label, res.size, np.median(res), np.percentile(res, 90), res.max(),
                                         (res <= 0.03).sum(), (res <= 0.1).sum(), (res <= 0.3).sum(), (res <= 1).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-near", required=True)
    ap.add_argument("--tau-far", required=True)
    ap.add_argument("--cap-near", required=True)
    ap.add_argument("--cap-far", default="")
    a = ap.parse_args()
    near = rows(a.tau_near)
    print("== near set at k*, every predicted shadow bounds the stripe (tau_check.py): %d candidates" % len(near))
    for tau in sorted(near[0]["planes"][0]["taus"], key=float):
        cells = [r["planes"][0]["taus"][tau] for r in near]
        print("tau %s gl: %s" % (tau, counts(cells)))
        print("   " + residual_line(cells, "tau %s" % tau))
    far = rows(a.tau_far)
    print("== far set: the fit's planes (|k - k*| >= 5) with a usable stripe, and the median residual on the line over "
          "them (tau_check.py): %d candidates" % len(far))
    for tau in sorted(far[0]["planes"][0]["taus"], key=float):
        parts = []
        for r in far:
            ok = [p["taus"][tau] for p in r["planes"][1:] if p["taus"][tau]["ok"]]
            res = [c["res_max"] for c in ok if np.isfinite(c["res_max"])]
            parts.append("%d: %d (%s)" % (r["node_id"], len(ok), "%.1f gl" % np.median(res) if res else "-"))
        print("tau %s gl: %s" % (tau, ", ".join(parts)))
    cap = rows(a.cap_near)
    c0 = [r["planes"][0]["centre"] for r in cap]
    print("== near set at k*, what lies over the node (cap_check.py): %d candidates; a neighbour above tau at the "
          "centre: soma %d, junction %d, broad (sigma_r >= 2 um) %d, sharp %d; summed predicted shadow at the centre "
          "median %.2f gl (p10 %.2f, p90 %.2f)" % (
              len(cap), sum(x["soma"] for x in c0), sum(x["junction"] > 0 for x in c0), sum(x["broad"] > 0 for x in c0),
              sum(x["sharp"] > 0 for x in c0), np.median([x["sum_gl"] for x in c0]),
              np.percentile([x["sum_gl"] for x in c0], 10), np.percentile([x["sum_gl"] for x in c0], 90)))
    for capv in sorted(cap[0]["planes"][0]["caps"], key=float):
        cells = [r["planes"][0]["caps"][capv] for r in cap]
        print("shadows with sigma_r >= %s um left to the plane: %s" % (capv, counts(cells)))
        print("   " + residual_line(cells, "cap %s (the broad shadows only)" % capv))
    if a.cap_far:
        capf = rows(a.cap_far)
        print("== far set, caps (cap_check.py): %d candidates" % len(capf))
        for capv in sorted(capf[0]["planes"][0]["caps"], key=float):
            parts = []
            for r in capf:
                ok = [p["caps"][capv] for p in r["planes"][1:] if p["caps"][capv]["ok"]]
                parts.append("%d: %d" % (r["node_id"], len(ok)))
            print("cap %s um: usable fit planes %s" % (capv, ", ".join(parts)))


if __name__ == "__main__":
    main()
