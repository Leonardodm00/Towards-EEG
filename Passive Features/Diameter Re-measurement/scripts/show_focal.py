#!/usr/bin/env python3
"""Show nodes at one recorded plane with the estimated tube drawn on the image (SPEC Block 36, amendment 12).

For each node (default: the example nodes of a cleaning run) one panel: the node's patch at the first stage's
sharpest plane k* (or at the recorded plane nearest the estimated axis depth), with the two edges of the tube
the joint fit estimated -- the per-node fit of Block 30 when its results are given and finite, else the last
round's refinement of Block 29 -- and Allen's diameter about the traced position (dashed). Reads only the files
a cleaning run (scripts/clean_cell.py clean) and, optionally, the per-node inference (scripts/infer_nodes.py
real) wrote; changes none of them.

    python3 scripts/show_focal.py --cleaning-dir OUT/cleaning_tier1 --specimen 529878215 \
        [--inference-dir OUT/inference_tier1] [--nodes 8089,7448] [--plane kstar|axis] [--image before|after] \
        [--estimate auto|fit|refined] [--zoom-um 6] [--out OUT/cleaning_tier1/focal_529878215.png]

Writes the PNG and focal_<specimen>.csv beside it, prints the table, returns 0.
Keep this file pure ASCII so it survives transfer to the cluster.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import node_inputs  # noqa: E402

DPI = 170
COLUMNS = ("node_id", "plane_k", "plane_minus_axis_um", "d_allen_um", "d_pilot_um", "d_refined_um", "d_fit_um",
           "se_d_fit_um", "drawn", "d_drawn_um", "v0_drawn_um", "image")


def _finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def focal_panel(rec, arrays, fit=None, plane="kstar", image="before", estimate="auto"):
    """-> (panel, row): one figure_focal panel of a node and its table row (SPEC Block 36, amendment 12).

    rec: the node's record of cleaning_<S>.json (pilot, refine, d_allen_um, U_axis_um, ...); arrays: its
    cleaning_<S>_node<id>.npz (block, cleaned, ks, planes_z, origin_xy, y_hat, e_u, left, top, res0_um, node_xyz,
    line_xy); fit: its record of inference_<S>.json (theta = (d, mu, v0, z_ax), se) or None.
    plane: "kstar" (the first stage's sharpest plane) or "axis" (the recorded plane nearest the estimated axis
    depth); image: "before" (recorded) or "after" (after the neighbours' cleaning); estimate: "fit", "refined",
    or "auto" (the per-node fit when it is there and finite, else the refinement).
    The axis crosses the measuring line at origin_xy + (v0 - v_t) y_hat, v_t = node_inputs.trace_offset: the foot
    of the traced position on the line plus v0 y_hat. ValueError for an unknown option or no usable estimate."""
    if plane not in ("kstar", "axis") or image not in ("before", "after") or estimate not in ("auto", "fit", "refined"):
        raise ValueError("plane %r, image %r or estimate %r not understood" % (plane, image, estimate))
    refine = rec.get("refine") or {}
    has_fit = fit is not None and "theta" in fit and all(_finite(x) for x in fit["theta"])
    has_ref = _finite(refine.get("d_um")) and _finite(refine.get("v0_um")) and _finite(refine.get("dz_um"))
    trace = np.asarray(arrays["node_xyz"], dtype=float)
    if estimate == "fit" or (estimate == "auto" and has_fit):
        if not has_fit:
            raise ValueError("no finite per-node fit for this node")
        d, v0, z_ax = float(fit["theta"][0]), float(fit["theta"][2]), float(fit["theta"][3])
        drawn = "fit"
    else:
        if not has_ref:
            raise ValueError("no finite refinement for this node")
        d, v0, z_ax = float(refine["d_um"]), float(refine["v0_um"]), float(trace[2]) + float(refine["dz_um"])
        drawn = "refined"
    ks = [int(k) for k in np.asarray(arrays["ks"])]
    planes_z = np.asarray(arrays["planes_z"], dtype=float)
    k_star = int(rec["pilot"]["k_star"])
    if plane == "kstar" and k_star in ks:
        j = ks.index(k_star)
    else:
        j = int(np.argmin(np.abs(planes_z - z_ax)))
    stack = np.asarray(arrays["block"] if image == "before" else arrays["cleaned"])
    img = np.asarray(stack[j], dtype=float)
    p = float(arrays["res0_um"])
    left, top = float(arrays["left"]), float(arrays["top"])
    H, W = img.shape
    extent = ((left - 0.5) * p, (left + W - 0.5) * p, (top + H - 0.5) * p, (top - 0.5) * p)   # figures.image_extent_um
    origin = np.asarray(arrays["origin_xy"], dtype=float)
    y_hat = np.asarray(arrays["y_hat"], dtype=float)
    pil = rec["pilot"]
    pilot = types.SimpleNamespace(cx_um=float(pil["cx_um"]), cy_um=float(pil["cy_um"]), theta_rad=float(pil["theta_rad"]))
    axis_xy = origin + (v0 - node_inputs.trace_offset(pilot, trace)) * y_hat
    d_allen = float(rec["d_allen_um"]) if _finite(rec.get("d_allen_um")) else None
    d_fit = float(fit["theta"][0]) if has_fit else float("nan")
    se_fit = float(fit["se"][0]) if has_fit and "se" in fit and _finite(fit["se"][0]) else float("nan")
    d_ref = float(refine["d_um"]) if has_ref else float("nan")
    d_pilot = float(pil["d_hat_um"]) if _finite(pil.get("d_hat_um")) else float("nan")
    label = "per-node joint fit" if drawn == "fit" else "joint refinement, last round"
    title = "node %d, plane k = %d%s\nd = %.3f%s um (%s)\nAllen %s um, first stage %.3f um" % (
        int(rec["node_id"]), ks[j], " (k*)" if ks[j] == k_star else "", d,
        (" +- %.3f" % se_fit) if drawn == "fit" and math.isfinite(se_fit) else "", "fit" if drawn == "fit" else "refined",
        ("%.3f" % d_allen) if d_allen is not None else "n/a", d_pilot)
    panel = dict(image=img, extent=extent, axis_xy=axis_xy, e_u=np.asarray(arrays["e_u"], dtype=float), d_um=d,
                 half_length_um=np.asarray(rec["U_axis_um"], dtype=float) if rec.get("U_axis_um") is not None else None,
                 ref_xy=trace[:2] if d_allen is not None else None, d_ref_um=d_allen,
                 line_xy=np.asarray(arrays["line_xy"], dtype=float) if "line_xy" in arrays else None,
                 title=title, label="estimated diameter (%s)" % label)
    row = dict(node_id=int(rec["node_id"]), plane_k=ks[j], plane_minus_axis_um=float(planes_z[j] - z_ax),
               d_allen_um=d_allen if d_allen is not None else float("nan"), d_pilot_um=d_pilot, d_refined_um=d_ref,
               d_fit_um=d_fit, se_d_fit_um=se_fit, drawn=drawn, d_drawn_um=d, v0_drawn_um=v0, image=image)
    return panel, row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cleaning-dir", required=True, help="the folder a cleaning run wrote")
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--inference-dir", default=None, help="the folder of infer_nodes.py real (optional)")
    ap.add_argument("--nodes", default=None, help="node ids, comma-separated (default: the run's example nodes)")
    ap.add_argument("--plane", choices=("kstar", "axis"), default="kstar")
    ap.add_argument("--image", choices=("before", "after"), default="before")
    ap.add_argument("--estimate", choices=("auto", "fit", "refined"), default="auto")
    ap.add_argument("--zoom-um", type=float, default=6.0, help="half-width of the view about the axis (um)")
    ap.add_argument("--ncols", type=int, default=None)
    ap.add_argument("--out", default=None, help="the PNG (default: <cleaning-dir>/focal_<specimen>.png)")
    a = ap.parse_args(argv)

    with open(os.path.join(a.cleaning_dir, "cleaning_%s.json" % a.specimen)) as fh:
        rec = json.load(fh)
    fits = {}
    if a.inference_dir:
        path = os.path.join(a.inference_dir, "inference_%s.json" % a.specimen)
        if os.path.exists(path):
            with open(path) as fh:
                fits = json.load(fh).get("nodes", {})
        else:
            print("no %s: the refinement's estimates are drawn" % path)
    ids = [int(x) for x in a.nodes.split(",") if x.strip()] if a.nodes else [int(x) for x in rec["example_ids"]]
    panels, rows = [], []
    for nid in ids:
        node = rec["nodes"].get(str(nid))
        npz = os.path.join(a.cleaning_dir, "cleaning_%s_node%d.npz" % (a.specimen, nid))
        if node is None or not os.path.exists(npz):
            print("node %d: skipped (%s)" % (nid, "not in the cleaning's work list" if node is None
                                             else "no %s: images are kept for the example nodes only" % os.path.basename(npz)))
            continue
        try:
            with np.load(npz) as z:
                panel, row = focal_panel(node, z, fits.get(str(nid)), a.plane, a.image, a.estimate)
        except ValueError as exc:
            print("node %d: skipped (%s)" % (nid, exc))
            continue
        panels.append(panel)
        rows.append(row)
    if not panels:
        print("no node to show")
        return 1
    from allen_diameter.plotting import cleaning_figures as CF
    import matplotlib.pyplot as plt
    out = a.out or os.path.join(a.cleaning_dir, "focal_%s.png" % a.specimen)
    fig = CF.figure_focal(panels, ncols=a.ncols, zoom_um=a.zoom_um)
    CF.save(fig, out, DPI)
    plt.close(fig)
    csv_path = os.path.join(os.path.dirname(out) or ".", "focal_%s.csv" % a.specimen)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("node   plane  z_k - z_ax   Allen   first stage   refined   per-node fit     drawn")
    for r in rows:
        fit = "%.3f +- %.3f" % (r["d_fit_um"], r["se_d_fit_um"]) if math.isfinite(r["d_fit_um"]) else "      -      "
        print("%-6d %-6d %+.3f um    %.3f   %.3f         %.3f     %s   %s (%.3f um)"
              % (r["node_id"], r["plane_k"], r["plane_minus_axis_um"], r["d_allen_um"], r["d_pilot_um"], r["d_refined_um"],
                 fit, r["drawn"], r["d_drawn_um"]))
    print("wrote %s and %s" % (out, csv_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
