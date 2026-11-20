#!/usr/bin/env python3
"""A gallery of the nodes the image judges clean (SPEC Block 36, amendment 18; Blocks 40 and 41; D-194).

Reads the files of a preparation (scripts/clean_cell.py clean --no-clean: cleaning_<S>.json and, per example
node, cleaning_<S>_node<id>.npz), runs the clearness screen of Block 40 on every example node -- the recorded
image only, no forward model, no fit -- and shows the nodes it judges clean: per node one page with the patch at
the sharpest plane, the patch at every recorded plane, and the measuring line's profile at every plane.

    python3 scripts/show_clear_gallery.py --cleaning-dir OUT/cleaning_gallery --specimen 529878215 \
        [--out OUT/cleaning_gallery/gallery] [--d-bins 0.40,0.45,0.60,0.80,1.20,5.0] [--max-per-bin 4] \
        [--fill-to 0] [--nodes 5353,2669] [--zoom-um 5] [--config-json FILE] \
        [--outside-max X] [--skew-max X] [--width-range-max X] [--depth-range-max X] [--centre-range-max X] \
        [--min-depth-sd X] [--along-half-um X] [--along-step-um X] [--focus-half-planes N]

Shown: the nodes judged clean, per bin of Allen's diameter the lowest score first, at most --max-per-bin per bin
(0: all). With --fill-to N, when fewer than N are shown, the best-scoring nodes that are NOT judged clean are
added up to N, each labelled as rejected in its title and in its file name (gallery_rejected_node<id>.png).
--nodes shows exactly those nodes, whatever their verdict.

Writes in --out: gallery_<S>.csv (every screened node, one row), gallery_<S>.json (the thresholds and every
node's numbers per plane and per line), gallery_node<id>.png or gallery_rejected_node<id>.png per shown node,
gallery_sheet_<S>.png (the shown nodes at their sharpest plane). Changes none of the files it reads. Returns 0
when at least one node was screened, else 1.
Keep this file pure ASCII so it survives transfer to the cluster.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter import config as C  # noqa: E402
from allen_diameter.analysis import clearness  # noqa: E402

DPI = 130
COLUMNS = ("node_id", "d_allen_um", "bin_lo_um", "bin_hi_um", "clean", "score", "reasons", "k_star", "depth_gl",
           "depth_sd", "sigma_gl", "fwhm_kstar_um", "fwhm_over_allen", "fwhm_min_um", "k_fwhm_min", "outside_focus",
           "skew_focus", "outside_along", "skew_along", "width_range", "depth_range", "centre_range", "shown", "figure")
OVERRIDES = (("outside_max", "outside_max"), ("skew_max", "skew_max"), ("width_range_max", "width_range_max"),
             ("depth_range_max", "depth_range_max"), ("centre_range_max", "centre_range_max"),
             ("min_depth_sd", "min_depth_sd"), ("along_half_um", "along_half_um"), ("along_step_um", "along_step_um"),
             ("focus_half_planes", "focus_half_planes"))


def _finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def parse_bins(text):
    """"e0,e1,...,eJ" -> [e0, ..., eJ] (um) or None for an empty text; ValueError unless at least two finite
    edges > 0, strictly increasing (clean_cell.py's --d-bins)."""
    if not str(text).strip():
        return None
    edges = [float(x) for x in str(text).split(",")]
    if len(edges) < 2 or not all(math.isfinite(e) and e > 0 for e in edges) \
            or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError("--d-bins needs at least two finite edges > 0, strictly increasing (um), got %r" % (text,))
    return edges


def bin_of(d_um, edges):
    """(lo, hi) of the bin [lo, hi) that holds d_um; (nan, nan) without edges or outside them."""
    if edges is not None and _finite(d_um):
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo <= float(d_um) < hi:
                return float(lo), float(hi)
    return float("nan"), float("nan")


def _max(x):
    x = np.asarray(x, dtype=float)
    return float(np.max(x)) if x.size and np.all(np.isfinite(x)) else float("nan")


def screen_file(rec, arrays, ccfg):
    """-> (ClearnessResult, row): the screen of one node from its record of cleaning_<S>.json (sigma_gl,
    pilot.k_star, d_allen_um) and its cleaning_<S>_node<id>.npz (block, ks, B_gl, v, profiles_before, origin_xy,
    y_hat, e_u, n_avg, step, left, top, res0_um). row: the node's line of gallery_<S>.csv, without the bin and
    the columns of what is shown."""
    ks = np.asarray(arrays["ks"]).astype(int)
    res = clearness.screen_node(arrays["block"], ks, int(rec["pilot"]["k_star"]), arrays["B_gl"], float(rec["sigma_gl"]),
                                int(arrays["left"]), int(arrays["top"]), float(arrays["res0_um"]), arrays["origin_xy"],
                                arrays["y_hat"], arrays["e_u"], arrays["v"], int(arrays["n_avg"]), float(arrays["step"]),
                                ccfg, profiles_gl=arrays["profiles_before"])
    j = res.k_index
    W = res.line["fwhm_um"]
    d_allen = float(rec["d_allen_um"]) if _finite(rec.get("d_allen_um")) else float("nan")
    i_min = int(np.nanargmin(W)) if np.any(np.isfinite(W)) else -1
    row = dict(node_id=int(rec["node_id"]), d_allen_um=d_allen, clean=bool(res.clean), score=res.score,
               reasons=" ".join(res.reasons), k_star=int(ks[j]), depth_gl=float(res.line["depth_gl"][j]),
               depth_sd=res.depth_sd, sigma_gl=float(rec["sigma_gl"]), fwhm_kstar_um=float(W[j]),
               fwhm_over_allen=float(W[j]) / d_allen if d_allen > 0 else float("nan"),
               fwhm_min_um=float(W[i_min]) if i_min >= 0 else float("nan"),
               k_fwhm_min=int(ks[i_min]) if i_min >= 0 else -1,
               outside_focus=_max(res.line["outside"][res.focus]), skew_focus=_max(res.line["skew"][res.focus]),
               outside_along=_max(res.along["outside"]), skew_along=_max(res.along["skew"]),
               width_range=res.width_range, depth_range=res.depth_range, centre_range=res.centre_range)
    return res, row


def choose(rows, max_per_bin, fill_to, nodes=None):
    """The node ids to show, in order (the docstring's rule). rows: screen_file's rows with bin_lo_um.
    nodes: an explicit list replaces the rule."""
    if nodes is not None:
        have = {r["node_id"] for r in rows}
        return [i for i in nodes if i in have]

    def key(r):
        lo = r["bin_lo_um"] if math.isfinite(r["bin_lo_um"]) else float("inf")
        return (lo, r["score"], r["node_id"])
    shown, count = [], {}
    for r in sorted((r for r in rows if r["clean"]), key=key):
        b = r["bin_lo_um"] if math.isfinite(r["bin_lo_um"]) else None
        if max_per_bin > 0 and count.get(b, 0) >= max_per_bin:
            continue
        count[b] = count.get(b, 0) + 1
        shown.append(r["node_id"])
    rest = sorted((r for r in rows if not r["clean"]), key=lambda r: (r["score"], r["node_id"]))
    for r in rest:
        if len(shown) >= fill_to:
            break
        shown.append(r["node_id"])
    return shown


def panel_text(row, res, ccfg, rec):
    """The numbers of one node as the figure's text panel."""
    q = ccfg
    pil = rec.get("pilot", {})

    def line(name, value, limit):
        return "  %-11s %5.2f / %.2f%s" % (name, value, limit, "" if not value > limit else "  <-- above")
    out = ["JUDGED CLEAN" if row["clean"] else "NOT judged clean",
           "score Q     %.2f  (<= 1 passes)" % row["score"],
           "" if row["clean"] else "reasons: " + row["reasons"],
           "Allen d     %.3f um" % row["d_allen_um"],
           "k*          %d" % row["k_star"],
           "depth       %.1f gl = %.0f noise SDs" % (row["depth_gl"], row["depth_sd"]),
           "            (floor %.0f SDs)" % q.min_depth_sd,
           "W at k*     %.2f um = %.2f x Allen" % (row["fwhm_kstar_um"], row["fwhm_over_allen"]),
           "narrowest W %.2f um at k %d" % (row["fwhm_min_um"], row["k_fwhm_min"]),
           "",
           "measuring line, planes k* +- %d" % q.focus_half_planes,
           line("outside E", row["outside_focus"], q.outside_max),
           line("skew S", row["skew_focus"], q.skew_max),
           "lines along the dendrite, +- %.1f um" % q.along_half_um,
           line("outside E", row["outside_along"], q.outside_max),
           line("skew S", row["skew_along"], q.skew_max),
           line("width U_W", row["width_range"], q.width_range_max),
           line("depth U_D", row["depth_range"], q.depth_range_max),
           line("centre U_c", row["centre_range"], q.centre_range_max),
           ""]
    if _finite(pil.get("mu_hat_per_um")):
        out.append("first stage: mu %.3f /um at Allen's d" % float(pil["mu_hat_per_um"]))
    if _finite(pil.get("phi_rad")):
        out.append("             tilt %.0f deg" % math.degrees(float(pil["phi_rad"])))
    return "\n".join(x for x in out if x is not None)


def node_panel(row, res, arrays, ccfg, rec):
    """The mapping clearness_figures.figure_node draws, from one node's files and its screen."""
    from allen_diameter.plotting.figures import image_extent_um
    block = np.asarray(arrays["block"], dtype=float)
    frame = SimpleNamespace(left=int(arrays["left"]), top=int(arrays["top"]), res_um_px=float(arrays["res0_um"]),
                            downsample=0)
    j = res.k_index
    sm = clearness.smooth_dip(np.asarray(arrays["B_gl"], dtype=float)[:, None]
                              - np.asarray(arrays["profiles_before"], dtype=float), ccfg.smooth_samples)
    feats = [clearness.dip_features(sm[i], arrays["v"], 0.0, ccfg.window_um, ccfg.skew_levels, ccfg.outside_fwhm)
             for i in range(sm.shape[0])]
    verdict = "judged clean" if row["clean"] else "NOT judged clean (%s)" % row["reasons"]
    return dict(images=block, extent=image_extent_um(frame, block.shape), ks=np.asarray(arrays["ks"]).astype(int),
                k_index=j, origin_xy=np.asarray(arrays["origin_xy"], dtype=float),
                y_hat=np.asarray(arrays["y_hat"], dtype=float), e_u=np.asarray(arrays["e_u"], dtype=float),
                v_um=np.asarray(arrays["v"], dtype=float), profiles_gl=np.asarray(arrays["profiles_before"], dtype=float),
                B_gl=np.asarray(arrays["B_gl"], dtype=float), u_um=res.u_um, strips_gl=res.strips_gl, focus=res.focus,
                d_allen_um=row["d_allen_um"] if _finite(row["d_allen_um"]) and row["d_allen_um"] > 0 else None,
                x_lo_um=np.array([f["x_lo_um"] for f in feats]), x_hi_um=np.array([f["x_hi_um"] for f in feats]),
                title="node %d  |  Allen's diameter %.3f um  |  %s, score %.2f"
                      % (row["node_id"], row["d_allen_um"], verdict, row["score"]),
                text=panel_text(row, res, ccfg, rec))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cleaning-dir", required=True, help="the folder of clean_cell.py clean --no-clean")
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--out", default=None, help="the output folder (default: <cleaning-dir>/gallery)")
    ap.add_argument("--d-bins", default="", help="edges (um) of the bins of Allen's diameter the shown nodes are "
                                                  "spread over, e.g. 0.40,0.45,0.60,0.80,1.20,5.0")
    ap.add_argument("--max-per-bin", type=int, default=4, help="judged-clean nodes shown per bin at most (0: all)")
    ap.add_argument("--fill-to", type=int, default=0, help="when fewer nodes are shown, add the best-scoring "
                                                             "rejected nodes up to this many, labelled as rejected")
    ap.add_argument("--nodes", default=None, help="node ids, comma-separated: show exactly these")
    ap.add_argument("--zoom-um", type=float, default=5.0, help="half-width of the image panels about the line (um)")
    ap.add_argument("--config-json", default="", help="a configuration JSON (its clearness section); default: the defaults")
    for opt, _field in OVERRIDES:
        ap.add_argument("--" + opt.replace("_", "-"), type=float, default=None, help="clearness.%s" % _field)
    a = ap.parse_args(argv)
    if a.max_per_bin < 0 or a.fill_to < 0 or not a.zoom_um > 0:
        ap.error("--max-per-bin and --fill-to must be >= 0 and --zoom-um > 0")
    cfg = C.default_config()
    if a.config_json:
        with open(a.config_json, encoding="utf-8") as fh:
            cfg = C.config_from_dict(json.load(fh))
    over = {}
    for opt, field in OVERRIDES:
        val = getattr(a, opt)
        if val is not None:
            over["clearness." + field] = int(val) if field == "focus_half_planes" else float(val)
    if over:
        cfg = C.with_overrides(cfg, over)
    ccfg = cfg.clearness
    edges = parse_bins(a.d_bins)
    out_dir = a.out or os.path.join(a.cleaning_dir, "gallery")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(a.cleaning_dir, "cleaning_%s.json" % a.specimen)) as fh:
        rec = json.load(fh)
    ids = [int(i) for i in rec.get("example_ids", [])]
    rows, results, skipped = [], {}, {}
    for nid in ids:
        node = rec["nodes"].get(str(nid))
        npz = os.path.join(a.cleaning_dir, "cleaning_%s_node%d.npz" % (a.specimen, nid))
        if node is None or not os.path.exists(npz):
            skipped[nid] = "not in the record" if node is None else "no %s" % os.path.basename(npz)
            continue
        try:
            with np.load(npz, allow_pickle=False) as z:
                res, row = screen_file(node, z, ccfg)
        except (KeyError, ValueError) as exc:
            skipped[nid] = "%s: %s" % (type(exc).__name__, exc)
            continue
        row["bin_lo_um"], row["bin_hi_um"] = bin_of(row["d_allen_um"], edges)
        rows.append(row)
        results[nid] = res
    for nid, why in skipped.items():
        print("node %d: skipped (%s)" % (nid, why))
    for nid, why in (rec.get("dropped") or {}).items():
        print("node %s: dropped by the preparation (%s)" % (nid, why))
    if not rows:
        print("no node to screen in %s" % a.cleaning_dir)
        return 1

    wanted = [int(x) for x in a.nodes.split(",") if x.strip()] if a.nodes else None
    shown = choose(rows, a.max_per_bin, a.fill_to, wanted)
    by_id = {r["node_id"]: r for r in rows}
    panels = []
    if shown:
        from allen_diameter.plotting import clearness_figures as CF
        import matplotlib.pyplot as plt
        for nid in shown:
            row = by_id[nid]
            with np.load(os.path.join(a.cleaning_dir, "cleaning_%s_node%d.npz" % (a.specimen, nid)),
                         allow_pickle=False) as z:
                panel = node_panel(row, results[nid], z, ccfg, rec["nodes"][str(nid)])
            name = ("gallery_node%d.png" if row["clean"] else "gallery_rejected_node%d.png") % nid
            fig = CF.figure_node(panel, zoom_um=a.zoom_um)
            CF.save(fig, os.path.join(out_dir, name), DPI)
            plt.close(fig)
            row["figure"] = name
            panels.append(dict(panel, title="node %d, Allen %.2f um\n%s, Q %.2f"
                               % (nid, row["d_allen_um"], "clean" if row["clean"] else "REJECTED", row["score"])))
        fig = CF.figure_sheet(panels, zoom_um=a.zoom_um)
        CF.save(fig, os.path.join(out_dir, "gallery_sheet_%s.png" % a.specimen), DPI)
        plt.close(fig)
    order = {nid: n for n, nid in enumerate(shown)}
    for r in rows:
        r["shown"] = r["node_id"] in order
        r.setdefault("figure", "")
    rows.sort(key=lambda r: (order.get(r["node_id"], len(order)), not r["clean"], r["score"], r["node_id"]))
    csv_path = os.path.join(out_dir, "gallery_%s.csv" % a.specimen)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    full = dict(specimen=str(a.specimen), cleaning_dir=os.path.abspath(a.cleaning_dir),
                clearness=dataclasses.asdict(ccfg), d_bins_um=edges, max_per_bin=a.max_per_bin, fill_to=a.fill_to,
                shown=shown, judged_clean=[r["node_id"] for r in rows if r["clean"]], skipped=skipped,
                nodes={str(r["node_id"]): dict(r, ratios=results[r["node_id"]].ratios,
                                               line=results[r["node_id"]].line, along=results[r["node_id"]].along,
                                               u_um=results[r["node_id"]].u_um, focus=results[r["node_id"]].focus)
                       for r in rows})
    json_path = os.path.join(out_dir, "gallery_%s.json" % a.specimen)
    with open(json_path, "w") as fh:
        json.dump(_jsonable(full), fh, indent=1)

    n_clean = sum(r["clean"] for r in rows)
    print("screened %d nodes: %d judged clean, %d not; shown %d (%d judged clean, %d rejected)"
          % (len(rows), n_clean, len(rows) - n_clean, len(shown), sum(by_id[i]["clean"] for i in shown),
             sum(not by_id[i]["clean"] for i in shown)))
    print("node    Allen   clean  score   depth(SD)  W(k*)  W/Allen  E_line S_line E_along S_along  U_W   U_D   U_c   reasons")
    for r in rows:
        print("%-7d %.3f   %-5s  %-6.2f  %-9.0f  %-5.2f  %-7.2f  %-6.2f %-6.2f %-7.2f %-7.2f  %-5.2f %-5.2f %-5.2f %s"
              % (r["node_id"], r["d_allen_um"], "yes" if r["clean"] else "no", r["score"], r["depth_sd"],
                 r["fwhm_kstar_um"], r["fwhm_over_allen"], r["outside_focus"], r["skew_focus"], r["outside_along"],
                 r["skew_along"], r["width_range"], r["depth_range"], r["centre_range"], r["reasons"]))
    print("wrote %s, %s and %d figures in %s" % (csv_path, json_path, len(shown) + (1 if shown else 0), out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
