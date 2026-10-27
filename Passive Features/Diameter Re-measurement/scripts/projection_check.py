#!/usr/bin/env python3
"""What Allen's projections are -- Block 16.3 in specs/SPEC.md (D-090.1).
Colab: api.brain-map.org is blocked from the sandbox.

For each specimen listed: every SubImage that is not a Primary plane, with
every field the API returns; the planes' count and size; the xy candidates
of Block 16.2 (a) and the one its darkest rule takes (each fetched whole at
survey.align_downsample). For each specimen of --check: on one
full-resolution window centred on the soma, each xy candidate against the
minimum, mean, median and maximum over the usable planes -- the best integer
offset, the correlation r*, the line, the verdict.

Example (Colab, Cell 11b)
    python scripts/projection_check.py --selection .../survey/selection.csv --check 529878215 \\
        --cache-dir /content/drive/MyDrive/allen_cache --out-dir .../survey/projections

Writes projections_<sid>.json and projection_check_<sid>.png per specimen
and projection_types.csv over the specimens listed. Cost of a check: one
crop of 2 h x 2 h px (h = survey.projcheck_half_px) per usable plane, about
520 for 529878215, and one per candidate, through the crop cache.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import allen_image_io as aio  # noqa: E402
import allen_image_plot as aip  # noqa: E402
from allen_diameter.analysis import cell_alignment  # noqa: E402
from allen_diameter.analysis import projection_check as PC  # noqa: E402
from allen_diameter.loading import cell_pool, survey_io  # noqa: E402
from build_table import load_config  # noqa: E402

PILOT = 529878215
TYPES_COLUMNS = ("specimen_id", "id", "image_type", "width", "height", "xy_candidate", "darkest", "best", "r",
                 "shift_x_px", "shift_y_px", "verdict")
TYPES_TYPES = {"specimen_id": "int", "id": "int", "image_type": "str", "width": "float", "height": "float",
               "xy_candidate": "str", "darkest": "str", "best": "str", "r": "float", "shift_x_px": "float",
               "shift_y_px": "float", "verdict": "str"}


def _int_list(text):
    text = (text or "").strip()
    if not text or text.lower() == "none":
        return []
    return [int(x) for x in text.split(",") if x.strip()]


def list_specimen(source, sid, cfg):
    """Block 16.3 (a): the record's listing part, and the tables the check needs."""
    sv = cfg.survey
    rows = source.subimage_rows(sid)
    images = aio.images_frame(rows)
    planes = aio.plane_table(images)
    prim = images[images["image_type"] == "Primary"]
    n_failed = int(prim["failed"].fillna(False).astype(bool).sum()) if "failed" in prim.columns else 0
    cands = cell_alignment.xy_projections(images, sv.projection_size_rel_tol)
    rec = dict(specimen_id=int(sid), subimages=[dict(r) for r in rows if r.get("image_type") != "Primary"],
               planes=dict(n_usable=int(len(planes)), n_failed=n_failed, width_px=float(planes["width"].median()),
                           height_px=float(planes["height"].median())),
               candidates=[int(i) for i in cands["id"]], darkest=None)
    if len(cands):
        arrays = [aio.fetch_whole(source.fetcher, row, sv.align_downsample)[0] for _, row in cands.iterrows()]
        k, meds = cell_alignment.darkest(arrays)
        rec["darkest"] = dict(id=int(cands["id"].iloc[k]), downsample=int(sv.align_downsample),
                              medians={str(int(i)): float(m) for i, m in zip(cands["id"], meds)})
    return rec, planes, cands


def window_centre(source, sid, cfg, center, survey_dir):
    """(cx, cy, center_source): the given pixel, or the soma under the cell's alignment record (Block 16.2 (e))
    when survey_dir holds one, otherwise under no shift and no flip."""
    if center is not None:
        return float(center[0]), float(center[1]), "given"
    swc = aip.read_swc(source.swc_path(sid))
    path = os.path.join(survey_dir, str(int(sid)), "align", "alignment_%d.json" % int(sid)) if survey_dir else ""
    if path and os.path.exists(path):
        t = cell_alignment.full_transform(survey_io.read_json(path))
        how = "soma (alignment record %s)" % os.path.basename(path)
    else:
        t, how = dict(shift_full_px=(0.0, 0.0), flip_y_full_h=None), "soma (no shift, no flip)"
    cx, cy = PC.soma_center(swc, cfg.acquisition.res0_um, t["shift_full_px"], t["flip_y_full_h"])
    return cx, cy, how


def check_specimen(source, sid, rec, planes, cands, cfg, center=None, survey_dir=""):
    """Block 16.3 (b): adds window, n_planes_used, checks to rec; returns (ops, crops) for the figure."""
    sv, rho0 = cfg.survey, cfg.acquisition.res0_um
    h, s = int(sv.projcheck_half_px), int(sv.projcheck_max_shift_px)
    if not len(cands):
        rec.update(checks=[], note="no xy candidate: nothing to check")
        return None, {}
    X = int(min(float(planes["width"].median()), float(cands["width"].astype(float).min())))
    Y = int(min(float(planes["height"].median()), float(cands["height"].astype(float).min())))
    cx, cy, how = window_centre(source, sid, cfg, center, survey_dir)
    left, top = PC.place_window(cx, cy, h, s, X, Y)
    rec["window"] = dict(left=int(left), top=int(top), half_px=h, max_shift_px=s, center_x=float(cx),
                         center_y=float(cy), center_source=how)
    k_lo, k_hi = int(planes["plane_index"].min()), int(planes["plane_index"].max())
    block, ks, valid, _ = aio.fetch_zblock(source.fetcher, planes, k_lo, k_hi, left, top, 2 * h, 2 * h, rho0)
    rec["n_planes_used"] = int(valid.sum())
    ops = PC.plane_operators(block, valid)
    del block
    checks, crops = [], {}
    for _, row in cands.iterrows():
        pid = int(row["id"])
        P_big, _ = aio.fetch_crop(source.fetcher, pid, left - s, top - s, 2 * (h + s), 2 * (h + s), 0, rho0)
        res = PC.check_projection(P_big, ops, s, sv.projcheck_min_r, sv.projcheck_min_ratio)
        bo = res["by_operator"].get(res["best"]) if res["best"] else None
        crops[pid] = PC.aligned_crop(P_big, bo["shift_x_px"] if bo else None, bo["shift_y_px"] if bo else None, s)
        checks.append(dict(id=pid, image_type=str(row["image_type"]), **res))
    rec["checks"] = checks
    return ops, crops


def type_rows(rec):
    """projection_types.csv's rows of one specimen record (16.3 (c))."""
    by_id = {int(c["id"]): c for c in rec.get("checks", [])}
    dk = (rec.get("darkest") or {}).get("id")
    out = []
    for r in rec["subimages"]:
        i = int(r["id"])
        c = by_id.get(i)
        bo = c["by_operator"].get(c["best"], {}) if c and c.get("best") else {}
        out.append(dict(specimen_id=int(rec["specimen_id"]), id=i, image_type=str(r.get("image_type")),
                        width=float(r["width"]) if r.get("width") is not None else math.nan,
                        height=float(r["height"]) if r.get("height") is not None else math.nan,
                        xy_candidate="yes" if i in rec["candidates"] else "no",
                        darkest="" if dk is None else ("yes" if i == dk else "no"),
                        best=(c.get("best") or "") if c else "", r=float(bo.get("r", math.nan)) if bo else math.nan,
                        shift_x_px=float(bo["shift_x_px"]) if bo and bo.get("shift_x_px") is not None else math.nan,
                        shift_y_px=float(bo["shift_y_px"]) if bo and bo.get("shift_y_px") is not None else math.nan,
                        verdict=c["verdict"] if c else ""))
    return out


def _say_listing(rec, log):
    subs = "; ".join("%s '%s' %sx%s" % (r.get("id"), r.get("image_type"), r.get("width"), r.get("height"))
                     for r in rec["subimages"]) or "none"
    pl, dk = rec["planes"], rec.get("darkest")
    log("[proj] %d: %d non-Primary SubImages: %s" % (rec["specimen_id"], len(rec["subimages"]), subs))
    log("[proj] %d: planes %d usable (%d failed), %.0fx%.0f px; xy candidates %s; darkest at level %s: %s"
        % (rec["specimen_id"], pl["n_usable"], pl["n_failed"], pl["width_px"], pl["height_px"], rec["candidates"],
           dk["downsample"] if dk else "-", dk["id"] if dk else "none"))


def _say_checks(rec, log):
    for c in rec.get("checks", []):
        parts = []
        for op in PC.OPERATORS:
            m = c["by_operator"].get(op, {})
            r = m.get("r")
            parts.append("%s %s" % (op, "%.4f at (%+d, %+d)" % (r, m["shift_x_px"], m["shift_y_px"])
                                    if r is not None and math.isfinite(r) else "n/a"))
        log("[proj] %d: %d '%s': %s -- r* %s" % (rec["specimen_id"], c["id"], c["image_type"], c["verdict"],
                                                 "; ".join(parts)))


def run(cfg, specimens, check, source, out_dir, center=None, survey_dir="", log=print):
    """Lists every specimen of specimens (and of check), checks those of check; writes the files of 16.3 (c).
    Returns {specimen_id: record}."""
    os.makedirs(out_dir, exist_ok=True)
    order = list(dict.fromkeys([int(s) for s in specimens] + [int(s) for s in check]))
    recs, all_rows = {}, []
    for sid in order:
        t0 = time.time()
        rec, planes, cands = list_specimen(source, sid, cfg)
        _say_listing(rec, log)
        if sid in check:
            ops, crops = check_specimen(source, sid, rec, planes, cands, cfg, center, survey_dir)
            _say_checks(rec, log)
            if ops is not None:
                from allen_diameter.plotting import survey_figures
                import matplotlib.pyplot as plt
                fig = survey_figures.projection_check_figure(rec, ops, crops)
                fig.savefig(os.path.join(out_dir, "projection_check_%d.png" % sid), dpi=110)
                plt.close(fig)
        rec["seconds"] = round(time.time() - t0, 1)
        survey_io.write_json(os.path.join(out_dir, "projections_%d.json" % sid), rec)
        recs[sid] = rec
        all_rows += type_rows(rec)
    cell_pool.write_table(os.path.join(out_dir, "projection_types.csv"), all_rows, TYPES_COLUMNS)
    log("[proj] %d specimens listed, %d checked -> %s" % (len(order), len([s for s in order if s in check]), out_dir))
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specimens", default="", help="comma-separated specimen ids to list")
    ap.add_argument("--selection", default="", help="selection.csv of Block 16.1: list all its cells")
    ap.add_argument("--check", default=str(PILOT), help="comma-separated ids to check against the planes, or none")
    ap.add_argument("--cache-dir", default="allen_cache")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--center", default="soma", help="soma (the SWC's) or X,Y in full-res px")
    ap.add_argument("--survey-dir", default="", help="the survey folder, for a cell's alignment record (Block 16.2)")
    ap.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config_json)
    specimens = _int_list(a.specimens)
    if a.selection:
        specimens += [int(r["specimen_id"]) for r in survey_io.read_selection(a.selection)]
    check = _int_list(a.check)
    if not specimens and not check:
        specimens = [PILOT]
    center = None
    if a.center.strip().lower() != "soma":
        xy = [float(x) for x in a.center.split(",")]
        if len(xy) != 2:
            ap.error("--center takes soma or X,Y")
        center = (xy[0], xy[1])
    import survey_cells
    run(cfg, specimens, check, survey_cells.AllenSource(a.cache_dir), a.out_dir, center, a.survey_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
