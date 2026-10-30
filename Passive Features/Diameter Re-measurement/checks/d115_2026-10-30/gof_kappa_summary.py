"""Summarise gof_kappa_check.py's output: g = RMS / SD_bg over the planes whose core fit ran (status ok, or
poor_fit when the scan applied a rule), clean against composite; kappa_g by the rule of SPEC 10.G (the 99th
percentile of g over the clean planes, rounded up to 0.1) and what it drops.
Usage: python3 -I gof_kappa_summary.py CHECK_JSON [KAPPA ...]   (extra KAPPA values are evaluated too)"""
import json
import math
import sys

import numpy as np


def planes(recs, kind):
    out = []
    for r in recs:
        if r["kind"] != kind:
            continue
        for p in r["planes"]:
            if p["status"] in ("ok", "poor_fit") and p["sd_bg_gl"] and math.isfinite(p["sd_bg_gl"]) \
                    and math.isfinite(p["rms_gl"]):
                out.append(dict(p, g=p["rms_gl"] / p["sd_bg_gl"], node=r["index"], s_um=r["s_um"], dz_um=r["dz_um"],
                                alpha=r["alpha"]))
    return out


def main(path, *extra):
    d = json.load(open(path))
    recs = d["records"]
    print("%s: set %s, %d records (%.1f min)" % (path, d["set"], len(recs), d["minutes"]))
    for kind in ("clean", "composite", "wide"):
        rs = [r for r in recs if r["kind"] == kind]
        print("  %s: %d phantoms, %d scanned; reasons of the others: %s" % (
            kind, len(rs), sum(1 for r in rs if r["planes"]),
            sorted(set(";".join(r["reasons"]) for r in rs if not r["planes"]))))
    cl, co = planes(recs, "clean"), planes(recs, "composite")
    gc, gk = np.array([p["g"] for p in cl]), np.array([p["g"] for p in co])
    q = [50, 90, 95, 99, 100]
    if gc.size:
        print("  clean planes %d: g percentiles %s = %s" % (gc.size, q, np.round(np.percentile(gc, q), 3)))
        amp = np.array([p["amp_gl"] for p in cl])
        print("  clean: g against the core amplitude, rank correlation %.2f" % (
            np.corrcoef(np.argsort(np.argsort(gc)), np.argsort(np.argsort(amp)))[0, 1]))
    if gk.size:
        print("  composite planes %d: g percentiles %s = %s" % (gk.size, q, np.round(np.percentile(gk, q), 3)))
    wd = planes(recs, "wide")
    if wd:
        gw = np.array([p["g"] for p in wd])
        print("  wide planes %d: g percentiles %s = %s" % (gw.size, q, np.round(np.percentile(gw, q), 3)))
        for lo, hi in ((0.6, 1.0), (1.0, 1.6), (1.6, 2.6)):
            sel = [p["g"] for p in wd if lo <= next(r["d_um"] for r in recs if r["kind"] == "wide" and r["index"] == p["node"]) < hi]
            if sel:
                print("     d in [%.1f, %.1f) um: %d planes, g median %.2f, 99th %.2f, max %.2f" % (
                    lo, hi, len(sel), np.median(sel), np.percentile(sel, 99), np.max(sel)))
    if not gc.size:
        return
    kap = math.ceil(np.percentile(gc, 99) * 10.0 - 1e-9) / 10.0
    print("  kappa_g by the rule (99th percentile of the clean planes, rounded up to 0.1): %.1f" % kap)
    for k in [kap] + [float(x) for x in extra]:
        rc, rk = np.mean(gc > k), np.mean(gk > k)
        if wd:
            print("  kappa %.2f: wide clean planes dropped %.1f %%" % (k, 100 * np.mean(gw > k)))
        nodes_c = len({p["node"] for p in cl if p["g"] > k})
        nodes_k = len({p["node"] for p in co if p["g"] > k})
        print("  kappa %.2f: clean planes dropped %.1f %% (%d nodes touched), composite planes dropped %.1f %% "
              "(%d nodes touched)" % (k, 100 * rc, nodes_c, 100 * rk, nodes_k))
        by_off = {}
        for p in co:
            by_off.setdefault(abs(p["offset"]), []).append(p["g"] > k)
        print("     composite dropped by |k - k*|: %s" % {o: "%d/%d" % (sum(v), len(v)) for o, v in sorted(by_off.items())})
        by_s = {}
        for p in co:
            b = "s<0.8" if p["s_um"] < 0.8 else ("s<1.2" if p["s_um"] < 1.2 else "s>=1.2")
            by_s.setdefault(b, []).append(p["g"] > k)
        print("     composite dropped by the neighbour's offset: %s" % {b: "%d/%d" % (sum(v), len(v)) for b, v in sorted(by_s.items())})


if __name__ == "__main__":
    main(*sys.argv[1:])
