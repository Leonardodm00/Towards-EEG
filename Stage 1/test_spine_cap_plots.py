"""test_spine_cap_plots -- smoke test for spine_cap_plots.

RUN
---
    python3 test_spine_cap_plots.py
    python3 test_spine_cap_plots.py --save /tmp/figs      # also write PNGs

A plotting smoke test cannot check that a figure is CORRECT -- only a human
looking at it can. What it can check is that every function returns a real
Figure on realistic input, survives the degenerate inputs that occur in
practice (no spines, no caps, a single cell, all-zero radii), draws the number
of axes it claims to, does not mutate its input, and -- where the figure makes
a quantitative claim in a label -- that the arithmetic behind that label is
right.

The one piece of real arithmetic in this module is the meridian of the cap
(spine_cap.cap_arc). It is checked against the closed form here as well as in
smoke_spine_cap.py, because a plot that draws a different surface from the one
being integrated is exactly the silent divergence the s1_plots contract exists
to prevent.

Test list:
  1  every function returns a matplotlib Figure on realistic input
  2  each returns the documented number of axes
  3  degenerate input is handled without raising, for all six
  4  a single audit dict is accepted as well as a {nid: dict} mapping
  5  a bare DataFrame is accepted as well as a {nid: frame} mapping
  6  input frames and profile dicts are not mutated
  7  the drawn cap meridian starts at r_t and ends at 0, and its rim
     radius reproduces r_t to machine precision
  8  the drawn cap meridian, revolved and integrated numerically, reproduces
     pi (r_t^2 + h^2) -- the plot and the number are the same surface
  9  gallery panel count respects n_max and n_cols
 10  profiles come back sorted by area, largest first
 11  an uncapped profile (cap off) draws no cap and reports A_cap = 0
 12  spine_profile follows the LONGEST branch of a branched spine
"""

import math
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

import spine_cap as sc                     # noqa: E402
import spine_cap_plots as scp              # noqa: E402
import spine_density as sd                 # noqa: E402
import spine_geometry as sg                # noqa: E402

FAILURES = []
_SAVE = None


def check(name, ok, detail=""):
    if ok:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s   %s" % (name, detail))
        FAILURES.append(name)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
def make_frame(rows):
    return pd.DataFrame(
        [{"id": i, "p": p, "x": x * 1000.0, "y": y * 1000.0, "z": z * 1000.0,
          "r": r * 1000.0, "annotated_type": lab}
         for (i, p, x, y, z, r, lab) in rows])


def demo_rows(n_shaft=6, specs=None):
    specs = specs if specs is not None else [
        [(0.45, 0.09, "neck"), (0.30, 0.32, "head")],
        [(0.55, 0.12, "neck"), (0.28, 0.26, "head")],
        [(0.35, 0.20, "spine")],
        [(0.40, 0.10, "neck"), (0.25, 0.30, "head"), (0.20, 0.22, "head")],
        [(0.60, 0.11, "neck"), (0.30, 0.28, "head")],
    ]
    rows = [(0, -1, 0.0, 0.0, 0.0, 0.5, "soma")]
    nid, shaft = 1, []
    for k in range(n_shaft):
        rows.append((nid, nid - 1, (k + 1) * 1.0, 0.0, 0.0, 0.5, "dendrite"))
        shaft.append(nid)
        nid += 1
    for i, spec in enumerate(specs):
        parent = shaft[min(i, len(shaft) - 1)]
        y = 0.0
        for (dy, r, lab) in spec:
            y += dy
            rows.append((nid, parent, rows[shaft[min(i, len(shaft) - 1)]][2],
                         y, 0.0, r, lab))
            parent = nid
            nid += 1
    return rows


def fixtures():
    df = make_frame(demo_rows())
    node, children, root = sd._prepare_nodes(
        df, sd.SHAFT_REGEX, sd.SPINE_LABELS, sd.DEFAULT_RADIUS_NM, "nm")
    prof = sc.spine_profiles(node, children, h_um=0.1)
    prof_nocap = sc.spine_profiles(node, children, h_um=None)
    audit = sc.audit_tips(node, children, root=root)
    geom = sg.build_spine_geometry(df, nid="C1", cap_tips=True, cap_h_um=0.1)
    f_by_mode = {}
    for lab, kw in (("no cap", dict(cap_tips=False)),
                    ("flat disc", dict(cap_tips=True, cap_h_um=0.0)),
                    ("cap h=100nm", dict(cap_tips=True, cap_h_um=0.1))):
        f_by_mode[lab] = {
            "C1": sd.cell_f_implied_from_phi(sd.build_phi(df, nid="C1", **kw))}
    return df, node, children, root, prof, prof_nocap, audit, geom, f_by_mode


def keep(fig, name):
    if _SAVE:
        import os
        os.makedirs(_SAVE, exist_ok=True)
        fig.savefig(os.path.join(_SAVE, name + ".png"), dpi=130,
                    bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    global _SAVE
    if "--save" in sys.argv:
        _SAVE = sys.argv[sys.argv.index("--save") + 1]

    print("test_spine_cap_plots")
    print("  %s / %s" % (scp.MODULE_VERSION, sc.MODULE_VERSION))
    print("-" * 74)

    df, node, children, root, prof, prof_nocap, audit, geom, f_by_mode = \
        fixtures()
    audits = {"C1": audit}

    # 1 + 2: returns a Figure, with the documented axes count
    specs = [
        ("spine_gallery", lambda: scp.spine_gallery(prof, n_cols=3, n_max=5),
         None),
        ("tip_radius_distribution",
         lambda: scp.tip_radius_distribution(audits), 2),
        ("cap_area_curve", lambda: scp.cap_area_curve(audits), 3),
        ("cap_contribution", lambda: scp.cap_contribution({"C1": geom}), 2),
        ("f_bracket", lambda: scp.f_bracket(f_by_mode), 1),
        ("tip_audit", lambda: scp.tip_audit(audits), 1),
    ]
    for name, fn, n_ax in specs:
        fig = fn()
        check("1 %s returns a Figure" % name,
              isinstance(fig, plt.Figure), type(fig).__name__)
        if n_ax is not None:
            check("2 %s has %d axes" % (name, n_ax),
                  len(fig.axes) == n_ax, "got %d" % len(fig.axes))
        keep(fig, name)

    # 3: degenerate input
    empty_geom = pd.DataFrame(columns=["A_spine_um2", "A_cap_um2"])
    empty_audit = {"spine_r_tip_um": [], "other_r_tip_um": [],
                   "spine_n_true_leaf": 0, "spine_n_label_boundary_end": 0,
                   "spine_n_true_leaf_bad_radius": 0}
    degenerate = [
        ("gallery/empty", lambda: scp.spine_gallery([])),
        ("tip_radius/empty",
         lambda: scp.tip_radius_distribution({"C1": empty_audit})),
        ("cap_area_curve/no audit", lambda: scp.cap_area_curve(None)),
        ("cap_contribution/empty", lambda: scp.cap_contribution(empty_geom)),
        ("f_bracket/empty", lambda: scp.f_bracket({})),
        ("tip_audit/empty", lambda: scp.tip_audit({})),
    ]
    for name, fn in degenerate:
        try:
            fig = fn()
            ok = isinstance(fig, plt.Figure)
            plt.close(fig)
        except Exception as exc:                       # noqa: BLE001
            ok, name = False, "%s (%s)" % (name, exc)
        check("3 %s degrades without raising" % name, ok)

    # 4: bare audit dict accepted
    for name, fn in (("tip_radius", scp.tip_radius_distribution),
                     ("tip_audit", scp.tip_audit)):
        fig = fn(audit)
        check("4 %s accepts a bare audit dict" % name,
              isinstance(fig, plt.Figure))
        plt.close(fig)

    # 5: bare DataFrame accepted
    fig = scp.cap_contribution(geom)
    check("5 cap_contribution accepts a bare DataFrame",
          isinstance(fig, plt.Figure))
    plt.close(fig)

    # 6: inputs not mutated
    geom_before = geom.copy(deep=True)
    prof_before = [dict(p) for p in prof]
    plt.close(scp.cap_contribution({"C1": geom}))
    plt.close(scp.spine_gallery(prof))
    check("6 geometry frame not mutated",
          geom.equals(geom_before))
    check("6 profile dicts not mutated",
          all(a["A_spine_um2"] == b["A_spine_um2"]
              and a["u_um"] == b["u_um"] and a["r_um"] == b["r_um"]
              for a, b in zip(prof, prof_before)))

    # 7 + 8: the drawn cap IS the integrated cap
    for (r_t, h) in ((0.30, 0.10), (0.08, 0.10), (0.50, 0.05)):
        du, rho = sc.cap_arc(r_t, h, n_points=4000)
        check("7 cap_arc rim radius == r_t (r=%.2f h=%.2f)" % (r_t, h),
              close(rho[0], r_t) and close(du[0], 0.0),
              "rim %.12f" % rho[0])
        check("7 cap_arc closes at the pole (r=%.2f h=%.2f)" % (r_t, h),
              rho[-1] == 0.0 and close(du[-1], h))
        z = np.asarray(du)
        y = np.asarray(rho)
        # surface of revolution of the DRAWN polyline, exactly
        seg = np.pi * (y[:-1] + y[1:]) * np.sqrt(
            (y[1:] - y[:-1]) ** 2 + (z[1:] - z[:-1]) ** 2)
        drawn = float(seg.sum())
        exact = sc.cap_area_um2(r_t, h)
        check("8 drawn meridian integrates to pi(r^2+h^2) (r=%.2f h=%.2f)"
              % (r_t, h),
              abs(drawn - exact) / exact < 5e-5,
              "drawn %.8f exact %.8f rel %.2e"
              % (drawn, exact, abs(drawn - exact) / exact))

    # 9: gallery respects n_max and n_cols
    fig = scp.spine_gallery(prof, n_cols=2, n_max=3)
    check("9 gallery n_max caps the panel count",
          len(fig.axes) == 4, "got %d axes for 3 spines in 2 cols"
                              % len(fig.axes))
    plt.close(fig)
    fig = scp.spine_gallery(prof, n_cols=5, n_max=2)
    check("9 gallery n_cols shrinks to n when n < n_cols",
          len(fig.axes) == 2, "got %d" % len(fig.axes))
    plt.close(fig)

    # 10: sorted by area, largest first
    areas = [p["A_spine_um2"] for p in prof]
    check("10 profiles sorted by area descending",
          all(areas[i] >= areas[i + 1] for i in range(len(areas) - 1)),
          str([round(a, 3) for a in areas]))

    # 11: cap off draws nothing
    check("11 h=None gives capped=False everywhere",
          all(not p["capped"] for p in prof_nocap))
    check("11 h=None gives A_cap == 0 everywhere",
          all(p["A_cap_um2"] == 0.0 for p in prof_nocap))
    check("11 capped profiles exceed uncapped by exactly the cap",
          all(close(a["A_spine_um2"],
                    b["A_spine_um2"] + a["A_cap_um2"])
              for a, b in zip(sorted(prof, key=lambda d: d["root_id"]),
                              sorted(prof_nocap, key=lambda d: d["root_id"]))))

    # 12: longest branch is followed
    y_rows = [
        (0, -1, 0.0, 0.0, 0.0, 0.5, "soma"),
        (1, 0, 1.0, 0.0, 0.0, 0.5, "dendrite"),
        (2, 1, 1.0, 0.4, 0.0, 0.10, "neck"),
        (3, 2, 1.0, 0.7, 0.0, 0.20, "head"),
        (4, 3, 1.05, 0.8, 0.0, 0.30, "head"),          # short branch
        (5, 3, 1.0, 1.9, 0.0, 0.25, "head"),           # long branch
    ]
    y_df = make_frame(y_rows)
    yn, yc, yr = sd._prepare_nodes(
        y_df, sd.SHAFT_REGEX, sd.SPINE_LABELS, sd.DEFAULT_RADIUS_NM, "nm")
    yp = sc.spine_profiles(yn, yc, h_um=0.1)[0]
    check("12 spine_profile follows the longest branch",
          yp["node_ids"][-1] == 5, "ended at %s" % yp["node_ids"][-1])
    check("12 branched spine still reports both tips",
          yp["n_tips"] == 2, "got %d" % yp["n_tips"])
    check("12 branched A_cap covers both tips",
          close(yp["A_cap_um2"],
                sc.cap_area_um2(0.30, 0.1) + sc.cap_area_um2(0.25, 0.1)))
    fig = scp.spine_gallery([yp], n_cols=1)
    check("12 branched spine renders", isinstance(fig, plt.Figure))
    keep(fig, "gallery_branched")

    print("-" * 74)
    if FAILURES:
        print("FAILED %d check(s): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
