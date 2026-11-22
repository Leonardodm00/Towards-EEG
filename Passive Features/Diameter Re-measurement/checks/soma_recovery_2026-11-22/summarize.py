"""The recovery check's results table (D-216.2), from results/<CASE>.json: per case the errors of c, a, b, S and r_eq
against the truth (um or um^2, %, SE; "*" when outside 2 SE), the fit's status, evaluations, wall time and chi2/dof,
then the fitted parameters with their SEs. Markdown on stdout.

    python3 summarize.py
Pure ASCII.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recovery_lib import CASE_ORDER                                      # noqa: E402

KEYS = (("c", "um"), ("a", "um"), ("b", "um"), ("S", "um2"), ("r_eq", "um"))
PARAMS = ("x0_um", "y0_um", "z0_um", "a_um", "b_um", "c_um", "phi_rad", "mu_per_um", "slope_um_per_um")


def cell(e, unit):
    if e is None:
        return "n/a"
    se = "%+.1f SE" % e["err_se"] if e["err_se"] is not None else "SE n/a"
    star = "" if e["within_2se"] else " *"
    fmt = "%+.2f" if unit == "um2" else "%+.4f"
    return (fmt + " %s (%+.2f %%, %s)%s") % (e["err"], unit, e["err_pct"], se, star)


def main():
    rows = []
    for name in CASE_ORDER:
        path = os.path.join(HERE, "results", name + ".json")
        if not os.path.exists(path):
            rows.append((name, None))
            continue
        with open(path) as fh:
            rows.append((name, json.load(fh)))
    print("| case | " + " | ".join("%s err" % k for k, _u in KEYS) + " | status | n_eval | time | chi2/dof |")
    print("|---|" + "---|" * (len(KEYS) + 4))
    for name, r in rows:
        if r is None:
            print("| %s | " % name + " | ".join("not run" for _ in KEYS) + " | | | | |")
            continue
        f = r["fit"]
        e = r["errors"]
        print("| %s | %s | %s | %d | %.0f min | %.3f |"
              % (name, " | ".join(cell(e[k] if e else None, u) for k, u in KEYS), f["status"], f["n_eval"],
                 f["seconds"] / 60.0, f["chi2_dof"] if f["chi2_dof"] is not None else float("nan")))
    print()
    print("| case | truth c_ref, S_ref | " + " | ".join(PARAMS) + " | S (um2) | r_eq (um) | at bound |")
    print("|---|---|" + "---|" * (len(PARAMS) + 3))
    for name, r in rows:
        if r is None:
            continue
        f = r["fit"]
        t = r["truth"]
        vals = []
        for v, s in zip(f["theta"], f["se"]):
            vals.append("%.4f +- %s" % (v, ("%.4f" % s) if s is not None else "held"))
        print("| %s | %.3f um, %.2f um2 | %s | %.2f +- %.2f | %.4f +- %.4f | %s |"
              % (name, t["c_ref_um"], t["S_ref_um2"], " | ".join(vals), f["area_um2"], f["se_area_um2"],
                 f["r_eq_um"], f["se_r_eq_um"], ", ".join(f["at_bound"]) or "none"))


if __name__ == "__main__":
    main()
