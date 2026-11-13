"""End-to-end runs of the entry points of SPEC.md section 6 that exercise
Blocks 16-18 (tester run D5, 2026-11-13), on their minimal configurations,
each twice to check determinism. Allen's server is substituted as section 7
says: canned rows for cell_pool.py, a synthetic source for projection_check
and the runner's steps.

  A  cell_pool.py pool --from-csv, select, pick (Cell 11; CLI)
  B  projection_check.run on a synthetic stack and its exact min / max (Cell 11b)
  C  survey_cells.run --steps align on painted cells, then the CLI's status (Cell 12)
  D  survey_cells.run --steps align,nodes,radius,registration,camera on the
     synthetic cell of fixtures_cell.py (manual alignment), then the CLI's
     pool --what coverage and pool --what inputs (Cells 13-14)

Run (from the workstream directory):
    MPLBACKEND=Agg python tests/spec/e2e_survey_D5.py [A B C D]
Prints one line per check; exit 1 when one fails.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec_helpers_D5 as H  # noqa: E402

import allen_image_align as aia  # noqa: E402
import allen_image_io as aio  # noqa: E402
import survey_cells as SC  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import cell_pool as CP  # noqa: E402
from allen_diameter.loading import survey_io, swc_io, table_io  # noqa: E402

RHO0 = 0.1144
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, ("  -- " + str(detail)) if detail else ""), flush=True)


def cli(script, args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, str(H.WS / "scripts" / script)] + args, cwd=str(H.WS), capture_output=True,
                       text=True, env=env)
    return r


def tree_files(root, rel_root):
    """{relative path: bytes} of every file under root, with the run's own path replaced."""
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            b = open(p, "rb").read()
            if not f.endswith(".png"):
                b = b.replace(rel_root.encode(), b"<ROOT>")
            out[os.path.relpath(p, root)] = b
    return out


def strip_timing(rel, data):
    """Markers and records hold their own wall time: drop those keys before comparing."""
    if rel.endswith(".json"):
        try:
            obj = json.loads(data)
        except ValueError:
            return data

        def drop(o):
            if isinstance(o, dict):
                return {k: drop(v) for k, v in o.items() if k not in ("seconds", "finished_utc")}
            if isinstance(o, list):
                return [drop(v) for v in o]
            return o
        return json.dumps(drop(obj), sort_keys=True).encode()
    return data


def compare_runs(name, a, ra, b, rb, skip=()):
    fa, fb = tree_files(a, ra), tree_files(b, rb)
    ka = {k for k in fa if not any(s in k for s in skip)}
    kb = {k for k in fb if not any(s in k for s in skip)}
    diff = sorted(k for k in ka & kb if strip_timing(k, fa[k]) != strip_timing(k, fb[k]))
    check("%s: two runs give identical files (%d files; timing keys dropped)" % (name, len(ka)),
          ka == kb and not diff, "differ: %s; only in one: %s" % (diff[:6], sorted(ka ^ kb)[:6]))
    return diff


# ------------------------------------------------------------------ A -------

def part_a():
    def one(root):
        canned = os.path.join(root, "canned.csv")
        rows = []
        for k in range(18):
            rows.append(dict(specimen_id=500000 + 53 * k, species=("Homo Sapiens", "homo sapiens", "Mus musculus")[k % 3],
                             layer=("3", "5")[k % 2], dendrite_type=("spiny", "aspiny", "sparsely spiny")[k % 3],
                             reconstruction_type="full" if k % 7 else "", area="MTG", donor_id=str(k),
                             hemisphere="left", normalized_depth=0.05 * k))
        CP.write_table(canned, rows, ("specimen_id", "species", "layer", "dendrite_type", "reconstruction_type",
                                      "area", "donor_id", "hemisphere", "normalized_depth"))
        outs = []
        for args in (["pool", "--from-csv", canned, "--out", os.path.join(root, "pool.csv")],
                     ["select", "--pool", os.path.join(root, "pool.csv"), "--out", os.path.join(root, "sel.csv")],
                     ["pick", "--pool", os.path.join(root, "pool.csv"), "--cells", "500159,500053",
                      "--out", os.path.join(root, "pick.csv")]):
            r = cli("cell_pool.py", args)
            outs.append(r)
            check("A: cell_pool.py %s exits 0" % args[0], r.returncode == 0, r.stderr[-300:])
        return rows, outs
    ra, rb = tempfile.mkdtemp(), tempfile.mkdtemp()
    rows, _ = one(ra)
    one(rb)
    pool = CP.read_table(os.path.join(ra, "pool.csv"), CP.POOL_TYPES)
    want = sorted(r["specimen_id"] for r in rows if r["species"].lower() == "homo sapiens" and r["reconstruction_type"])
    check("A: the pool holds exactly the human cells with a reconstruction", [r["specimen_id"] for r in pool] == want)
    sel = survey_io.read_selection(os.path.join(ra, "sel.csv"))
    check("A: the selection's columns and types", all(isinstance(r["specimen_id"], int) and isinstance(r["rank"], int)
                                                      and r["group"] == "%s|%s" % (r["neuron_class"], r["layer"])
                                                      for r in sel) and len(sel) > 0)
    js = json.load(open(os.path.join(ra, "sel.json")))
    check("A: sel.json holds the seed, rule, groups and exclusions", {"seed", "rule", "groups", "excluded"} <= set(js))
    pick = survey_io.read_selection(os.path.join(ra, "pick.csv"))
    check("A: pick keeps the order given", [r["specimen_id"] for r in pick] == [500159, 500053])
    compare_runs("A", ra, ra, rb, rb, skip=("canned",))


# ------------------------------------------------------------------ B -------

class MultiFetcher(aio.ImageFetcher):
    def __init__(self, by_id):
        self.by_id = by_id

    def get(self, image_id, left, top, width, height, downsample=0):
        return self.by_id[int(image_id)].get(image_id, left, top, width, height, downsample)


def stack_source(root, seed=11):
    """5 planes of 600 x 640 px (dark tubes each sharpest at its own plane), the exact pixelwise min and max
    placed 2 px larger with the content at (+1, +1) -- as Allen's projections are (16.3 (a))."""
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    Hh, Ww, n = 600, 640, 5
    base = np.zeros((Hh, Ww))
    yy, xx = np.mgrid[:Hh, :Ww]
    planes = []
    for k in range(n):
        img = np.full((Hh, Ww), 215.0)
        for t in range(4):
            ang = rng.uniform(0, math.pi)
            off = rng.uniform(-200, 200)
            d = np.abs((xx - Ww / 2) * math.sin(ang) - (yy - Hh / 2) * math.cos(ang) - off)
            img -= 90.0 * np.exp(-0.5 * (d / (2.0 + 2.5 * abs(t - k))) ** 2)
        img = ndimage.gaussian_filter(img, 1.0) + rng.normal(0, 3.0, img.shape)
        planes.append(np.clip(np.round(img), 0, 255).astype(np.uint8))
    st = np.stack(planes)
    mn, mx = st.min(0), st.max(0)
    P_min = np.full((Hh + 2, Ww + 2), 255, np.uint8)
    P_max = np.full((Hh + 2, Ww + 2), 255, np.uint8)
    P_min[1:1 + Hh, 1:1 + Ww] = mn
    P_max[1:1 + Hh, 1:1 + Ww] = mx
    by = {100 + k: aio.SyntheticFetcher(planes[k]) for k in range(n)}
    by[900], by[901] = aio.SyntheticFetcher(P_min), aio.SyntheticFetcher(P_max)
    rows = [dict(id=100 + k, image_type="Primary", section_number=20 + k, width=Ww, height=Hh, resolution=RHO0,
                 data_set_id=1, failed=False, path="/x/%d" % k) for k in range(n)]
    rows += [dict(id=900, image_type="MinimumIntensityProjection - xy", section_number=None, width=Ww + 2,
                  height=Hh + 2, resolution=RHO0, data_set_id=None, failed=False, novel_field="kept verbatim"),
             dict(id=901, image_type="MaximumIntensityProjection - xy", section_number=None, width=Ww + 2,
                  height=Hh + 2, resolution=RHO0, data_set_id=None, failed=False, novel_field="kept verbatim")]
    swc = H.write_swc_text(os.path.join(root, "c.swc"), [(1, 1, 320 * RHO0, 300 * RHO0, 0.0, 5.0, -1),
                                                          (2, 3, 330 * RHO0, 300 * RHO0, 0.0, 0.5, 1)])

    class Src:
        fetcher = MultiFetcher(by)

        def subimage_rows(self, sid):
            return [dict(r) for r in rows]

        def swc_path(self, sid):
            return swc
    return Src()


def part_b():
    import projection_check as PCS
    cfg = default_config()
    outs = []
    for _ in range(2):
        root = tempfile.mkdtemp()
        src = stack_source(root)
        out = os.path.join(root, "proj")
        recs = PCS.run(cfg, [], [529878215], src, out, log=lambda m: None)
        outs.append((root, out, recs))
    root, out, recs = outs[0]
    rec = recs[529878215]
    v = {c["id"]: c for c in rec["checks"]}
    bm = v[900]["by_operator"]["min"]
    check("B: the min projection 'matches min' at offset (+1, +1), r* >= 0.999, zero residual",
          v[900]["verdict"] == "matches min" and (bm["shift_x_px"], bm["shift_y_px"]) == (1, 1) and bm["r"] >= 0.999
          and bm["rms_gl"] < 1e-6, "%s %s" % (v[900]["verdict"], bm))
    check("B: the max projection 'matches max'", v[901]["verdict"] == "matches max", v[901]["verdict"])
    check("B: chosen by type is the min projection", rec["chosen"] == dict(id=900, rule="type",
                                                                         image_type="MinimumIntensityProjection - xy"))
    check("B: the record's keys", {"specimen_id", "subimages", "planes", "candidates", "darkest", "chosen", "window",
                                   "n_planes_used", "checks", "seconds"} <= set(rec), sorted(rec))
    check("B: an unknown SubImage field listed verbatim",
          all(s.get("novel_field") == "kept verbatim" for s in rec["subimages"]))
    w = rec["window"]
    from_spec = H.ref_place_window(320.0, 300.0, 256, 3, 640, 600)
    check("B: the window at the soma by 16.3 (b)'s rule", (w["left"], w["top"]) == from_spec, "%s vs %s" % (w, from_spec))
    js = survey_io.read_json(os.path.join(out, "projections_529878215.json"))
    check("B: projections JSON round trip: strict JSON", js["specimen_id"] == 529878215)
    types = CP.read_table(os.path.join(out, "projection_types.csv"), {"id": "int"})
    check("B: projection_types.csv one row per non-Primary SubImage", sorted(r["id"] for r in types) == [900, 901])
    check("B: the figure written", os.path.exists(os.path.join(out, "projection_check_529878215.png")))
    compare_runs("B", outs[0][1], outs[0][0], outs[1][1], outs[1][0], skip=(".png",))


# ------------------------------------------------------------------ C -------

W_FULL, H_FULL = 1600, 2000


def tree_df(seed=21, n_branch=5, steps=28, step_um=2.6, x0=95.0, y0=115.0):
    rng = np.random.default_rng(seed)
    rows = [(1, 1, x0, y0, 15.0, 5.0, -1)]
    nid = 2
    for b in range(n_branch):
        a = 2 * math.pi * b / n_branch + 0.7
        par, x, y = 1, x0 + 5 * math.cos(a), y0 + 5 * math.sin(a)
        for _ in range(steps):
            a += rng.normal(0, 0.12)
            x += step_um * math.cos(a)
            y += step_um * math.sin(a)
            rows.append((nid, 3 + (b % 2), x, y, 15.0, 0.6, par))
            par = nid
            nid += 1
    return pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"])


def painted_min(swc, dx, dy, seed, empty=False):
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    img = np.full((H_FULL, W_FULL), 205.0)
    if not empty:
        m = aia.render_skeleton_mask(swc, (H_FULL, W_FULL), RHO0, (dx, dy), None, (3, 4))
        img[ndimage.binary_dilation(m, iterations=3)] = 75.0
    yy, xx = np.ogrid[:H_FULL, :W_FULL]
    for _ in range(25):
        r, c, rad = rng.integers(0, H_FULL), rng.integers(0, W_FULL), rng.uniform(8, 16)
        img[(yy - r) ** 2 + (xx - c) ** 2 <= rad ** 2] = 115.0
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


class PaintedSource:
    def __init__(self, root):
        self.root, self.cells, by = root, {}, {}
        self.fetcher = MultiFetcher(by)
        self._by = by

    def add(self, sid, dx, dy, empty=False):
        swc = tree_df()
        mn = painted_min(swc, dx, dy, seed=sid, empty=empty)
        mx = np.clip(240 + np.random.default_rng(sid + 1).normal(0, 2, mn.shape), 0, 255).astype(np.uint8)
        base = sid * 10
        self._by[base + 1], self._by[base + 2] = aio.SyntheticFetcher(mx), aio.SyntheticFetcher(mn)
        self._by[base + 3] = aio.SyntheticFetcher(np.full((H_FULL, 300), 200, np.uint8))
        rows = [dict(id=base + 1, image_type="MaximumIntensityProjection - xy", width=W_FULL, height=H_FULL),
                dict(id=base + 2, image_type="MinimumIntensityProjection - xy", width=W_FULL, height=H_FULL),
                dict(id=base + 3, image_type="MinimumIntensityProjection - yz", width=300, height=H_FULL)]
        rows += [dict(id=base + 5 + k, image_type="Primary", width=W_FULL - 2, height=H_FULL - 2, section_number=k)
                 for k in range(3)]
        for r in rows:
            r.setdefault("section_number", None)
            r.update(resolution=RHO0, data_set_id=None, failed=False)
        df = pd.DataFrame(rows)
        df["is_plane"] = df["image_type"] == "Primary"
        path = os.path.join(self.root, "%d.swc" % sid)
        H.write_swc_text(path, list(swc.itertuples(index=False, name=None)))
        self.cells[sid] = dict(images=df, path=path, truth=(dx, dy))
        return self

    def swc_path(self, sid):
        return self.cells[int(sid)]["path"]

    def images(self, sid):
        return self.cells[int(sid)]["images"]


def part_c():
    cfg = default_config()
    runs = []
    for _ in range(2):
        root = tempfile.mkdtemp()
        src = PaintedSource(root).add(31, 37.0, -21.0).add(32, 0.0, 0.0, empty=True)
        sel = os.path.join(root, "selection.csv")
        CP.write_table(sel, [dict(specimen_id=s, species="Homo Sapiens", layer="3", dendrite_type="spiny",
                                  reconstruction_type="full", area="MTG", donor_id="1", hemisphere="left",
                                  normalized_depth=0.5, neuron_class="exc", group="exc|3", rank=i)
                             for i, s in enumerate((31, 32))], CP.SELECTION_COLUMNS)
        out = os.path.join(root, "survey")
        t0 = time.time()
        st = SC.run(cfg, sel, out, src, ["align"], log=lambda m: None)
        runs.append((root, out, sel, st, time.time() - t0))
    root, out, sel, st, dt = runs[0]
    check("C: align done on both cells (%.0f s)" % dt, all(r["align"] == "done" for r in st), str(st))
    rec = survey_io.read_json(os.path.join(out, "31", "align", "alignment_31.json"))
    f_f = 2 ** rec["fine"]["downsample"]
    err = (rec["shift_x_px"] - 37.0, rec["shift_y_px"] + 21.0)
    check("C: the painted cell's shift within f_f + 0.5 px of the truth after the fine stage (16.2 oracle (a))",
          max(abs(err[0]), abs(err[1])) <= f_f + 0.5 and rec["flip_h_px"] is None, "error %s px, f_f %d" % (err, f_f))
    check("C: the painted cell kept, R >= 2", rec["set_aside"] is False and rec["score_ratio"] >= 2.0,
          "R %s, reasons %s" % (rec["score_ratio"], rec["reasons"]))
    check("C: projection chosen by type", rec["projection"]["rule"] == "type" and
          rec["projection"]["image_type"] == "MinimumIntensityProjection - xy")
    keys = {"specimen_id", "method", "shift_x_px", "shift_y_px", "flip_h_px", "z0_um", "coarse", "null", "score_ratio",
            "fine", "projection", "set_aside", "reasons"}
    check("C: the record's keys (16.2 (e))", keys <= set(rec), sorted(keys - set(rec)))
    e = survey_io.read_json(os.path.join(out, "32", "align", "alignment_32.json"))
    check("C: the empty field set aside on R", e["set_aside"] and e["score_ratio"] is not None and e["score_ratio"] < 2,
          "R %s, reasons %s" % (e["score_ratio"], e["reasons"]))
    check("C: overlays written", all(os.path.exists(os.path.join(out, s, "align", "overlay_%s.png" % s))
                                     for s in ("31", "32")))
    r = cli("survey_cells.py", ["status", "--out-dir", out, "--selection", sel])
    tab = CP.read_table(os.path.join(out, "status.csv"), {"specimen_id": "int"})
    by = {x["specimen_id"]: x for x in tab}
    check("C: CLI status rebuilds the table from the markers", r.returncode == 0 and by[31]["align"] == "done"
          and by[32]["nodes"] == "skipped:set_aside" and by[31]["nodes"] == "pending", str(tab))
    compare_runs("C", runs[0][1], runs[0][0], runs[1][1], runs[1][0], skip=(".png", "status.csv"))


# ------------------------------------------------------------------ D -------

class CellSource:
    def __init__(self, swc_path, fetcher, planes, cache_dir):
        self._swc, self.fetcher, self._planes, self.cache_dir = swc_path, fetcher, planes, cache_dir

    def swc_path(self, sid):
        return self._swc

    def images(self, sid):
        raise AssertionError("not used: the alignment is manual")

    def planes(self, sid):
        return self._planes


def part_d():
    import fixtures_cell
    base = default_config()
    cfg = dataclasses.replace(base, survey=dataclasses.replace(base.survey, nodes_per_bin=3, registration_min_nodes=5))
    runs = []
    for _ in range(2):
        root = tempfile.mkdtemp()
        swc, fetcher, planes, swc_path = fixtures_cell.synthetic_cell(root, cfg)
        sel = os.path.join(root, "selection.csv")
        CP.write_table(sel, [dict(specimen_id=11, species="Homo Sapiens", layer="3", dendrite_type="spiny",
                                  reconstruction_type="full", area="MTG", donor_id="9", hemisphere="left",
                                  normalized_depth=0.3, neuron_class="exc", group="exc|3", rank=0)],
                       CP.SELECTION_COLUMNS)
        out = os.path.join(root, "survey")
        survey_io.write_json(survey_io.CellDir(out, 11).path_of("align", "alignment_manual.json"),
                             dict(shift_x_px=0.0, shift_y_px=0.0, flip_h_px=None, z0_um=0.0))
        cache = os.path.join(root, "cache")
        os.makedirs(cache)
        t0 = time.time()
        st = SC.run(cfg, sel, out, CellSource(swc_path, fetcher, planes, cache),
                    ["align", "nodes", "radius", "registration", "camera"], log=lambda m: None)
        dt = time.time() - t0
        rc = cli("survey_cells.py", ["pool", "--out-dir", out, "--what", "coverage", "--selection", sel])
        ri = cli("survey_cells.py", ["pool", "--out-dir", out, "--what", "inputs", "--selection", sel])
        runs.append((root, out, sel, st, dt, rc, ri, swc))
    root, out, sel, st, dt, rc, ri, swc = runs[0]
    check("D: steps align, nodes, radius, registration, camera done (%.0f s)" % dt,
          all(st[0][s] == "done" for s in ("align", "nodes", "radius", "registration", "camera")), str(st))
    check("D: CLI pool --what coverage exits 0", rc.returncode == 0, rc.stderr[-400:])
    check("D: CLI pool --what inputs exits 0", ri.returncode == 0, ri.stderr[-400:])
    cell = survey_io.CellDir(out, 11)
    rows = table_io.read_rows(cell.path_of("nodes", "pilot_11.csv"))
    # the sample (17.1) by the independent reading: the fixture's one stretch (ids 2-7), Allen 2r 0.6 um: bin 2
    stretches = [[int(i) for i in swc.ids if i != 1]]
    r_of = {int(i): float(r) for i, r in zip(swc.ids, swc.radius)}
    ref_ids, ref_counts = H.ref_stratified(stretches, r_of, cfg.survey.nodes_allen_bins_um, cfg.survey.selection_seed,
                                           11, 3)
    check("D: the CSV holds exactly the sampled ids (whole_stretches=False), as the spec's sample",
          [int(r["node_id"]) for r in rows] == ref_ids or sorted(int(r["node_id"]) for r in rows) == sorted(ref_ids),
          "csv %s, reference %s" % ([r["node_id"] for r in rows], ref_ids))
    summ = survey_io.read_json(cell.path_of("nodes", "pilot_summary_11.json"))
    check("D: pilot_summary sample_counts per bin", summ["sample_counts"] == {"2": [6, 3]}, str(summ["sample_counts"]))
    fin = [r for r in rows if r["fit_status"] == "converged"]
    check("D: converged rows have finite d_hat, mu_hat, alpha and background statistics",
          len(fin) > 0 and all(all(math.isfinite(float(r[k])) for k in ("d_hat_um", "mu_hat_per_um", "alpha_hat",
                                                                        "bg_median_gl", "bg_robust_sd_gl"))
                               for r in fin), "%d converged of %d" % (len(fin), len(rows)))
    check("D: d_hat in um near the fixture's 0.8 um tube", all(0.5 < float(r["d_hat_um"]) < 1.2 for r in fin),
          str([round(float(r["d_hat_um"]), 3) for r in rows]))
    cam = survey_io.read_json(cell.path_of("camera", "camera_11.json"))
    bmed = float(np.median([float(r["bg_median_gl"]) for r in rows if math.isfinite(float(r["bg_median_gl"]))]))
    check("D: B3 -- the camera's B is the median of the nodes' background medians",
          abs(cam["renderer"]["background_B_gl"] - bmed) <= 1e-9 * bmed, "%s vs %s" % (cam["renderer"]["background_B_gl"],
                                                                              bmed))
    rad = table_io.read_rows(cell.path_of("radius", "radius_nodes_11.csv"))
    mk = survey_io.read_marker(cell, "radius")["summary"]
    check("D: radius -- one row per dendrite node, 2r and its maximum",
          len(rad) == 6 and all(abs(float(r["diameter_um"]) - 0.6) < 1e-12 for r in rad) and abs(mk["max_um"] - 0.6) < 1e-12)
    cov = survey_io.read_json(os.path.join(out, "s3a", "coverage.json"))
    in_s = sum(1 for r in rows if r["fit_status"] == "converged" and not any(
        f in str(r["flags"]).split(";") for f in ("crossing", "stack_edge", "bbar_few", "profile_nan")))
    check("D: coverage per_cell n_in_S by the spec's S", cov["per_cell"]["11"]["n_in_S"] == in_s,
          "%s vs %d" % (cov["per_cell"]["11"], in_s))
    for f in ("coverage.csv", "diameters.csv", "coverage_mud.png", "coverage_share.png", "coverage_alpha.png",
              "diameters.png"):
        check("D: s3a/%s written" % f, os.path.exists(os.path.join(out, "s3a", f)))
    inp = survey_io.read_json(os.path.join(out, "s3b", "inputs.json"))
    check("D: inputs d_max = the largest Allen 2r (0.6 um), cell 11",
          abs(inp["suggested"]["phantom_d_max_um"] - 0.6) < 1e-12 and inp["d_max"]["specimen_id"] == 11)
    reg = survey_io.read_json(cell.path_of("registration", "registration_11.json"))
    on = [v for v in reg.values() if str(v.get("verdict", "")).upper().startswith("ON THE PROCESS")]
    s = [float(v["lateral_offset_um"]) for v in on if v.get("lateral_offset_um") is not None]
    want = H.ref_robust_sd(s) if s else float("nan")
    got = inp["suggested"]["jitter_xy_um"]
    check("D: B6 -- jitter_xy from the ON nodes only (%d of %d nodes ON)" % (len(on), len(reg)),
          (got is None and not math.isfinite(want)) or (got is not None and abs(got - want) <= 1e-12),
          "%s vs %s" % (got, want))
    check("D: inputs camera B equals the cell's", inp["suggested"]["background_B_gl"] == cam["renderer"]["background_B_gl"])
    for f in ("inputs_per_cell.csv", "inputs.png"):
        check("D: s3b/%s written" % f, os.path.exists(os.path.join(out, "s3b", f)))
    # B4: what Block 19's step reads from this CSV (the pilot's view: Block 13 Readers, 19.3)
    need = {"node_id", "fit_status", "flags", "phi_rad", "alpha_hat", "allen_radius_um", "B_bar", "mu_hat_per_um",
            "pilot_status", "pilot_flags", "alpha_pilot", "B_bar_pilot", "mu_pilot_per_um", "in_S_pilot"}
    check("D: B4 -- the S3a CSV holds the columns Block 19's readers use", need <= set(rows[0]), sorted(need - set(rows[0])))
    check("D: Block 17 notation's mu_d_hat column in the S3a CSV", "mu_d_hat" in rows[0])
    compare_runs("D", runs[0][1], runs[0][0], runs[1][1], runs[1][0], skip=(".png", "errors.log"))


def part_e():
    """Swappability (directive audit): one component changed through the configuration at a time."""
    import fixtures_cell
    base = default_config()
    # E1: survey.nodes_mode "all" (17.1's other mode) through the runner, then the pooled coverage through the CLI
    # with a hand-written --config-json changing the coverage bins and the minimum count
    cfg = dataclasses.replace(base, survey=dataclasses.replace(base.survey, nodes_mode="all"))
    root = tempfile.mkdtemp()
    swc, fetcher, planes, swc_path = fixtures_cell.synthetic_cell(root, cfg)
    sel = os.path.join(root, "selection.csv")
    CP.write_table(sel, [dict(specimen_id=11, species="Homo Sapiens", layer="3", dendrite_type="spiny",
                              reconstruction_type="full", area="MTG", donor_id="9", hemisphere="left",
                              normalized_depth=0.3, neuron_class="exc", group="exc|3", rank=0)], CP.SELECTION_COLUMNS)
    out = os.path.join(root, "survey")
    survey_io.write_json(survey_io.CellDir(out, 11).path_of("align", "alignment_manual.json"),
                         dict(shift_x_px=0.0, shift_y_px=0.0, flip_h_px=None, z0_um=0.0))
    st = SC.run(cfg, sel, out, CellSource(swc_path, fetcher, planes, root), ["align", "nodes"], log=lambda m: None)
    rows = table_io.read_rows(survey_io.CellDir(out, 11).path_of("nodes", "pilot_11.csv"))
    check("E1: nodes_mode all measures every dendrite node", st[0]["nodes"] == "done" and
          sorted(int(r["node_id"]) for r in rows) == [2, 3, 4, 5, 6, 7], str([r["node_id"] for r in rows]))
    cj = os.path.join(root, "cfg.json")
    json.dump({"survey": {"coverage_dhat_bins_um": [0.5, 1.0], "coverage_min_nodes": 2}}, open(cj, "w"))
    r = cli("survey_cells.py", ["pool", "--out-dir", out, "--what", "coverage", "--selection", sel, "--config-json", cj])
    cov = survey_io.read_json(os.path.join(out, "s3a", "coverage.json")) if r.returncode == 0 else {}
    edges = [(b["lo"], b["hi"]) for b in cov.get("bins", [])]
    check("E1: pool --what coverage under a --config-json with other bins", r.returncode == 0 and
          edges == [(0.0, 0.5), (0.5, 1.0), (1.0, None)], "%s %s" % (edges, r.stderr[-300:]))
    # E2: survey.projection_xy_type naming no candidate: the darkest is taken (rule "darkest"), the rest unchanged
    cfg2 = dataclasses.replace(base, survey=dataclasses.replace(base.survey, projection_xy_type="NoSuchType"))
    root2 = tempfile.mkdtemp()
    src = PaintedSource(root2).add(31, 37.0, -21.0)
    sel2 = os.path.join(root2, "selection.csv")
    CP.write_table(sel2, [dict(specimen_id=31, species="Homo Sapiens", layer="3", dendrite_type="spiny",
                               reconstruction_type="full", area="MTG", donor_id="1", hemisphere="left",
                               normalized_depth=0.5, neuron_class="exc", group="exc|3", rank=0)], CP.SELECTION_COLUMNS)
    SC.run(cfg2, sel2, os.path.join(root2, "survey"), src, ["align"], log=lambda m: None)
    rec = survey_io.read_json(os.path.join(root2, "survey", "31", "align", "alignment_31.json"))
    check("E2: projection_xy_type swapped: rule darkest, the same projection and shift",
          rec["projection"]["rule"] == "darkest" and rec["projection"]["image_type"].startswith("Minimum")
          and max(abs(rec["shift_x_px"] - 37.0), abs(rec["shift_y_px"] + 21.0)) <= 2.5, str(rec["projection"]["rule"]))


def part_f():
    """17.3 through the CLI with --phantom-rows: the phantoms' band of a bin = numpy.quantile (0.005, 0.995) of
    their ln(mu_hat d_hat), and the real nodes' share inside it (a hand-made survey folder: one cell, 22 nodes)."""
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "survey")
    rng = np.random.default_rng(2)
    sel = os.path.join(tmp, "sel.csv")
    CP.write_table(sel, [dict(specimen_id=11, species="Homo Sapiens", layer="3", dendrite_type="spiny",
                              reconstruction_type="full", area="MTG", donor_id="9", hemisphere="left",
                              normalized_depth=0.3, neuron_class="exc", group="exc|3", rank=0)], CP.SELECTION_COLUMNS)
    cell = survey_io.CellDir(out, 11)
    rows = [dict(node_id=k, d_hat_um=float(rng.uniform(0.41, 0.59)), mu_hat_per_um=float(rng.uniform(0.4, 2.8)),
                 alpha_hat=0.5, fit_status="converged", flags="", reg_verdict="", allen_radius_um=0.3, in_S=True,
                 reject="") for k in range(22)]
    table_io.write_rows(rows, cell.path_of("nodes", "pilot_11.csv"))
    survey_io.write_marker(cell, "nodes", {}, 0.0, "x")
    rep = os.path.join(tmp, "rep.csv")
    pd_, pm = rng.uniform(0.41, 0.59, 30), rng.uniform(0.3, 3.0, 30)
    table_io.write_rows([dict(draw_index=i, realization_index=0, d_um=0.5, phi_rad=0.1, mu_per_um=1.0,
                              d_hat_um=float(pd_[i]), mu_hat_per_um=float(pm[i]), ratio=1.0, in_S=True, sim_hash="x",
                              reject="") for i in range(30)], rep)
    r = cli("survey_cells.py", ["pool", "--out-dir", out, "--what", "coverage", "--selection", sel,
                                "--phantom-rows", rep])
    check("F: pool --what coverage --phantom-rows exits 0", r.returncode == 0, r.stderr[-300:])
    cov = survey_io.read_json(os.path.join(out, "s3a", "coverage.json"))
    b = [x for x in cov["bins"] if x["lo"] == 0.4][0]
    lo, hi = np.quantile(np.log(pm * pd_), (0.005, 0.995))
    lnm = np.log([x["mu_hat_per_um"] * x["d_hat_um"] for x in rows])
    check("F: the phantoms' band = numpy.quantile of their ln(mu d) in the bin",
          abs(b["phantom_lo"] - lo) <= 1e-12 and abs(b["phantom_hi"] - hi) <= 1e-12, str((b["phantom_lo"], lo)))
    check("F: the share inside the phantoms' band", b["share_inside_phantom"] == float(np.mean((lnm >= lo) & (lnm <= hi)))
          and cov["n_phantom_rows"] == 30)


def main(argv=None):
    parts = (argv or sys.argv[1:]) or ["A", "B", "C", "D", "E", "F"]
    for p in parts:
        t0 = time.time()
        {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e, "F": part_f}[p.upper()]()
        print("-- part %s: %.0f s" % (p, time.time() - t0), flush=True)
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("== %d checks, %d failed" % (len(RESULTS), n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
