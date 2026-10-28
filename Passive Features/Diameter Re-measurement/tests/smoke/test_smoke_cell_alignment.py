"""Smoke test for each S3 cell's global alignment and the resumable runner --
Block 16.2 in specs/SPEC.md (D-085.2, D-085.12; 2026-10-26).

A synthetic cell: a tree of dendrites (SWC types 3 and 4, in um) painted as
dark tubes about 1 um wide on a bright noisy field at a known full-res shift
(and y-flip), the minimum-intensity projection; its maximum-intensity twin
(bright, no structure); a yz-shaped image; three Primary planes 2 px smaller
than the projections, as Allen's are. The SubImages carry Allen's image_type
strings ('MinimumIntensityProjection - xy', 'MaximumIntensityProjection - xy',
'MinimumIntensityProjection - yz'). A synthetic source serves the SWC, the
SubImage table and the crops (allen_image_io.SyntheticFetcher per image).

Checks
    test_known_answer   the minimum projection chosen by its type (D-098.2),
                        also when the maximum-intensity twin is Allen's
                        inverted image, the darkest of the two (Cell 11b,
                        2026-10-28); no flip / flip; the
                        coarse shift within 1.5 f_c full-res px (the level-4
                        pixel plus the offset of up to (f_c - 1)/2 between
                        box-averaged content and Block 0's corner convention),
                        the final shift within f_f + 0.5 = 2.5 px (the level-1
                        pixel and its half-pixel offset); score ratio >= 2;
                        the flip height f_c x rows; not set aside
    test_reference      the final transform puts the skeleton on the painted
                        tubes: swc_to_full_px with the result against the
                        truth, within 2.5 sqrt(2) px (max over the dendrite
                        nodes)
    test_convergence    not a discretisation: skipped
    test_invariants     painting the cell 37 px further along x moves the
                        shift by 37 within 2 px
    test_contract       the record's keys and JSON round trip; the marker and
                        status files; full_transform feeds cell.to_image_um
    test_determinism    the null with its seed: the same ratio twice
    test_edge_cases     choose_projection: the type matched in any case and
                        with outer spaces, the darkest of several of that
                        type, the darkest of all when none has it (D-089.2's
                        rule, the fallback), a length mismatch refused;
                        an empty field is set aside on the ratio; a manual
                        alignment is used as given; no xy projection fails the
                        step naming the types; the fine level under Block 0's
                        size guard; the runner: resume, --force,
                        a failing step recorded and the next cell run, a
                        set-aside cell skipping its later steps unless
                        --include-set-aside, status rebuilt from the markers

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_cell_alignment.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import allen_image_align as aia  # noqa: E402
import allen_image_io as aio  # noqa: E402
import allen_image_plot as aip  # noqa: E402
import survey_cells as SC  # noqa: E402
from allen_diameter.analysis import cell as cellmod  # noqa: E402
from allen_diameter.analysis import cell_alignment as CA  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import cell_pool, survey_io  # noqa: E402

SEED = 20261026
REPORT_PACKAGES = ("numpy", "scipy", "pandas", "matplotlib")
RHO0 = 0.1144                      # um per full-res pixel (acquisition.res0_um)
W_FULL, H_FULL = 2400, 3200        # full-res projection (H a multiple of 16: the flip height is exact)


# --------------------------------------------------------------- fixtures ---

def tree_swc(seed=3, n_branch=6, length_um=110.0, step_um=3.0, x0_um=137.0, y0_um=183.0):
    """A radial tree in um: a soma (type 1) and dendrites of types 3 and 4."""
    rng = np.random.default_rng(seed)
    rows = [(1, 1, x0_um, y0_um, 20.0, 6.0, -1)]
    nid = 2
    for b in range(n_branch):
        a = 2 * math.pi * b / n_branch + 0.3
        par, x, y = 1, x0_um + 6 * math.cos(a), y0_um + 6 * math.sin(a)
        for _ in range(int(length_um / step_um)):
            a += rng.normal(0, 0.10)
            x += step_um * math.cos(a)
            y += step_um * math.sin(a)
            rows.append((nid, 3 if b % 2 else 4, x, y, 20.0, 0.5, par))
            par = nid
            nid += 1
    return pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"])


def write_swc(df, path):
    with open(path, "w") as f:
        for r in df.itertuples(index=False):
            f.write("%d %d %.4f %.4f %.4f %.4f %d\n" % (r.id, r.type, r.x, r.y, r.z, r.r, r.parent))
    return path


def painted(swc, dx, dy, flip, seed, empty=False):
    """The minimum projection: the dendrites under the true transform as dark tubes (~1 um), noise, specks."""
    rng = np.random.default_rng(seed)
    img = np.full((H_FULL, W_FULL), 210.0)
    if not empty:
        m = aia.render_skeleton_mask(swc, (H_FULL, W_FULL), RHO0, (dx, dy), H_FULL if flip else None, (3, 4))
        img[ndimage.binary_dilation(m, iterations=4)] = 70.0
    for _ in range(40):                                   # clutter: dark specks of 2-4 um
        r, c = rng.integers(0, H_FULL), rng.integers(0, W_FULL)
        rad = rng.uniform(9, 18)
        yy, xx = np.ogrid[:H_FULL, :W_FULL]
        img[(yy - r) ** 2 + (xx - c) ** 2 <= rad ** 2] = 110.0
    return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)


MAX_XY, MIN_XY, MIN_YZ = "MaximumIntensityProjection - xy", "MinimumIntensityProjection - xy", \
    "MinimumIntensityProjection - yz"         # Allen's image_type strings (Cell 11b, 2026-10-28)


def image_table(base_id, with_xy=True, typed=True):
    """The SubImage table; typed=False gives every projection the generic type 'Projection'."""
    t_max, t_min, t_yz = (MAX_XY, MIN_XY, MIN_YZ) if typed else ("Projection",) * 3
    rows = []
    if with_xy:
        rows += [dict(id=base_id + 1, image_type=t_max, section_number=None, width=W_FULL, height=H_FULL,
                      resolution=RHO0, data_set_id=None, failed=False),
                 dict(id=base_id + 2, image_type=t_min, section_number=None, width=W_FULL, height=H_FULL,
                      resolution=RHO0, data_set_id=None, failed=False)]
    rows += [dict(id=base_id + 3, image_type=t_yz, section_number=None, width=500, height=H_FULL,
                  resolution=RHO0, data_set_id=None, failed=False)]
    rows += [dict(id=base_id + 10 + k, image_type="Primary", section_number=10 + k, width=W_FULL - 2,
                  height=H_FULL - 2, resolution=RHO0, data_set_id=base_id, failed=False) for k in range(3)]
    df = pd.DataFrame(rows)
    df["is_plane"] = df["image_type"] == "Primary"
    return df


class MultiFetcher(aio.ImageFetcher):
    def __init__(self):
        self.by_id, self.n_requests = {}, 0

    def get(self, image_id, left, top, width, height, downsample=0):
        self.n_requests += 1
        return self.by_id[int(image_id)].get(image_id, left, top, width, height, downsample)


class SynthSource:
    """The Allen source of Block 16.2 for synthetic cells: sid -> (swc path, image table, full arrays)."""

    def __init__(self, tmp):
        self.tmp, self.cells, self.fetcher = tmp, {}, MultiFetcher()

    def add(self, sid, dx, dy, flip=False, empty=False, with_xy=True, seed=None, swc_seed=3, inverted_max=False,
            typed=True):
        """inverted_max: the maximum-intensity twin is the minimum's negative (bright tubes on a dark field), as
        Allen's 'MaximumIntensityProjection - xy' (Cell 11b, 2026-10-28), so it is the darker of the two."""
        swc = tree_swc(seed=swc_seed)
        base = sid * 100
        mn = painted(swc, dx, dy, flip, seed if seed is not None else sid % 1000, empty)
        if inverted_max:
            mx = (255 - mn.astype(np.int16)).astype(np.uint8)
        else:
            mx = np.clip(235 + np.random.default_rng(sid % 997).normal(0, 3, mn.shape), 0, 255).astype(np.uint8)
        yz = np.full((H_FULL, 500), 200, dtype=np.uint8)
        for iid, arr in ((base + 1, mx), (base + 2, mn), (base + 3, yz)):   # the minimum is NOT the first row
            self.fetcher.by_id[iid] = aio.SyntheticFetcher(arr)
        self.cells[sid] = dict(swc=swc, swc_path=write_swc(swc, os.path.join(self.tmp, "%d.swc" % sid)),
                               images=image_table(base, with_xy, typed), truth=(dx, dy, flip), min_id=base + 2,
                               max_id=base + 1, arrays=dict(mn=mn, mx=mx))
        return self

    def swc_path(self, sid):
        return self.cells[int(sid)]["swc_path"]

    def images(self, sid):
        return self.cells[int(sid)]["images"]


def selection_csv(tmp, sids):
    rows = [dict(specimen_id=s, species="Homo Sapiens", layer="3", dendrite_type="spiny", reconstruction_type="full",
                 area="MTG", donor_id="9", hemisphere="left", normalized_depth=0.3, neuron_class="exc",
                 group="exc|3", rank=i) for i, s in enumerate(sids)]
    path = os.path.join(tmp, "selection.csv")
    cell_pool.write_table(path, rows, cell_pool.SELECTION_COLUMNS)
    return path


def align_one(src, sid, cfg=None):
    cfg = cfg or default_config()
    with tempfile.TemporaryDirectory() as out:
        cell = survey_io.CellDir(out, sid)
        ctx = SC.Context(dict(specimen_id=sid, group="exc|3"), cell, src)
        summary = SC.step_align(ctx, cfg)
        rec = survey_io.read_json(cell.path_of("align", "alignment_%d.json" % sid))
        assert os.path.exists(cell.path_of("align", "overlay_%d.png" % sid))
        return rec, summary


_CACHE = {}


def standard():
    """The two reference cells, aligned once: 101 (no flip), 102 (y-flipped)."""
    if not _CACHE:
        tmp = tempfile.mkdtemp()
        src = SynthSource(tmp).add(101, 83.0, -51.0).add(102, 45.0, 29.0, flip=True)
        _CACHE["src"] = src
        _CACHE[101] = align_one(src, 101)
        _CACHE[102] = align_one(src, 102)
    return _CACHE


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    c = standard()
    f_c, f_f = 16, 2
    for sid, (dx, dy, flip) in ((101, (83.0, -51.0, False)), (102, (45.0, 29.0, True))):
        rec, summ = c[sid]
        assert rec["projection"]["id"] == c["src"].cells[sid]["min_id"], rec["projection"]
        assert rec["projection"]["rule"] == "type" and rec["projection"]["image_type"] == MIN_XY, rec["projection"]
        assert len(rec["projection"]["candidates"]) == 2, "the yz-shaped image is not a candidate"
        assert rec["coarse"]["flip"] is flip and (rec["flip_h_px"] is not None) == flip, rec
        co = rec["coarse"]
        assert abs(co["shift_x_px"] - dx) <= 1.5 * f_c and abs(co["shift_y_px"] - dy) <= 1.5 * f_c, (sid, co, dx, dy)
        tol = f_f + 0.5                 # the level-1 pixel plus its half-pixel corner offset (SPEC 16.2 (d))
        assert abs(rec["shift_x_px"] - dx) <= tol and abs(rec["shift_y_px"] - dy) <= tol, \
            (sid, rec["shift_x_px"], rec["shift_y_px"], dx, dy)
        assert rec["fine"]["downsample"] == 1 and rec["coarse"]["downsample"] == 4
        assert rec["score_ratio"] >= 2.0 and not rec["set_aside"] and rec["reasons"] == [], rec
        if flip:
            assert rec["flip_h_px"] == f_c * (H_FULL // f_c), rec["flip_h_px"]
        assert summ["set_aside"] is False and summ["method"] == "auto"
    # D-098.2: Allen's 'MaximumIntensityProjection - xy' is an inverted image, the darker of the two; the darkest rule
    # (D-089.2) takes it, the rule by type keeps the minimum and the alignment holds
    tmp = tempfile.mkdtemp()
    src = SynthSource(tmp).add(104, 83.0, -51.0, inverted_max=True, seed=101 % 1000)
    cell = src.cells[104]
    k_dark, _m = CA.darkest([aio.SyntheticFetcher(cell["arrays"][a]).get(0, 0, 0, W_FULL, H_FULL, 4)
                              for a in ("mx", "mn")])
    assert k_dark == 0, "the inverted twin must be the darker (the test's premise)"
    rec, summ = align_one(src, 104)
    assert rec["projection"]["id"] == cell["min_id"] and rec["projection"]["rule"] == "type", rec["projection"]
    tol = f_f + 0.5
    assert abs(rec["shift_x_px"] - 83.0) <= tol and abs(rec["shift_y_px"] + 51.0) <= tol, rec
    assert not rec["set_aside"] and rec["score_ratio"] >= 2.0, (rec["score_ratio"], rec["reasons"])


def test_reference():
    c = standard()
    for sid in (101, 102):
        rec, _ = c[sid]
        swc = c["src"].cells[sid]["swc"]
        dx, dy, flip = c["src"].cells[sid]["truth"]
        d = swc[swc["type"].isin((3, 4))]
        xt, yt = aia.swc_to_full_px(d, RHO0, (dx, dy), H_FULL if flip else None)
        t = CA.full_transform(rec)
        xr, yr = aia.swc_to_full_px(d, RHO0, t["shift_full_px"], t["flip_y_full_h"])
        err = float(np.max(np.hypot(np.asarray(xr) - xt, np.asarray(yr) - yt)))
        assert err <= 2.5 * math.sqrt(2), (sid, err)          # within f_f + 0.5 = 2.5 px on each axis


def test_convergence():
    raise unittest.SkipTest("no discretisation: a search over integer shifts")


def test_invariants():
    c = standard()
    tmp = tempfile.mkdtemp()
    src = SynthSource(tmp).add(103, 83.0 + 37.0, -51.0, seed=101 % 1000)     # the same field, 37 px further
    rec, _ = align_one(src, 103)
    base = c[101][0]
    assert abs((rec["shift_x_px"] - base["shift_x_px"]) - 37.0) <= 2.0, (rec["shift_x_px"], base["shift_x_px"])
    assert abs(rec["shift_y_px"] - base["shift_y_px"]) <= 2.0


def test_contract():
    c = standard()
    rec, summ = c[101]
    keys = {"method", "shift_x_px", "shift_y_px", "flip_h_px", "z0_um", "coarse", "null", "score_ratio", "fine",
            "projection", "set_aside", "reasons", "specimen_id"}
    assert keys <= set(rec), sorted(keys - set(rec))
    assert {"downsample", "dx_px", "dy_px", "flip", "score", "score_noflip", "score_flip", "half_window_px",
            "at_border"} <= set(rec["coarse"])
    assert {"downsample", "crop", "dx_px", "dy_px", "score", "score_coarse", "half_window_px", "at_border"} <= set(rec["fine"])
    json.dumps(rec, allow_nan=False)
    t = CA.full_transform(rec)
    xyz = cellmod.to_image_um(np.array([[10.0, 20.0, 5.0]]), RHO0, **t)
    assert xyz.shape == (1, 3) and abs(xyz[0, 2] - 5.0) < 1e-12
    assert abs(xyz[0, 0] - (10.0 / RHO0 + rec["shift_x_px"]) * RHO0) < 1e-9
    with tempfile.TemporaryDirectory() as out:
        cell = survey_io.CellDir(out, 7)
        survey_io.write_marker(cell, "align", summ, 1.5, "abc")
        m = survey_io.read_marker(cell, "align")
        assert m["summary"]["shift_x_px"] == summ["shift_x_px"] and m["config_hash"] == "abc" and m["seconds"] == 1.5
        p = survey_io.write_status(out, [dict(specimen_id=7, group="g", align="done")], ["align", "probe"])
        rows = open(p).read().splitlines()
        assert rows[0] == "specimen_id,group,align,probe,error" and rows[1] == "7,g,done,pending,", rows
        survey_io.write_json(os.path.join(out, "x.json"), dict(a=float("nan"), b=np.float64(2.0), c=(1, 2)))
        assert survey_io.read_json(os.path.join(out, "x.json")) == dict(a=None, b=2.0, c=[1, 2])


def test_determinism():
    c = standard()
    rec2, _ = align_one(c["src"], 101)
    assert rec2["score_ratio"] == c[101][0]["score_ratio"] and rec2["null"] == c[101][0]["null"]
    assert rec2["shift_x_px"] == c[101][0]["shift_x_px"] and rec2["shift_y_px"] == c[101][0]["shift_y_px"]


def _probe_steps(calls, fail_on=()):
    def probe(ctx, cfg):
        calls.append(ctx.sid)
        if ctx.sid in fail_on:
            raise RuntimeError("probe failure on %d" % ctx.sid)
        return dict(sid=ctx.sid)
    return probe


def test_edge_cases():
    cfg = default_config()
    assert cfg.survey.projection_xy_type == MIN_XY
    dark, mid, light = (np.full((4, 4), v, dtype=np.uint8) for v in (40, 120, 220))
    # the type in any case and with outer spaces; the darkest of several of the type; none of it: the darkest of all
    assert CA.choose_projection([MAX_XY, "  minimumintensityprojection - XY "], [dark, light], MIN_XY)[:2] == (1, "type")
    assert CA.choose_projection([MIN_XY, MAX_XY, MIN_XY], [light, dark, mid], MIN_XY)[:2] == (2, "type")
    k, rule, meds = CA.choose_projection(["Projection", "Projection"], [mid, dark], MIN_XY)
    assert (k, rule) == (1, "darkest") and meds == [120.0, 40.0], (k, rule, meds)
    assert CA.choose_projection([MIN_XY + "x", MAX_XY], [light, dark], MIN_XY)[:2] == (1, "darkest")
    try:
        CA.choose_projection([MIN_XY], [light, dark], MIN_XY)
    except ValueError:
        pass
    else:
        raise AssertionError("a types list of another length must be refused")
    tmp = tempfile.mkdtemp()
    src = SynthSource(tmp).add(201, 30.0, 12.0).add(202, 0.0, 0.0, empty=True).add(203, 0.0, 0.0, with_xy=False)
    src.add(204, 30.0, 12.0, typed=False, seed=201 % 1000)            # generic types: the darkest rule, the fallback
    rec_f, _ = align_one(src, 204)
    assert rec_f["projection"]["rule"] == "darkest" and rec_f["projection"]["id"] == src.cells[204]["min_id"], \
        rec_f["projection"]
    # an empty field: set aside on the ratio
    rec, _ = align_one(src, 202)
    assert rec["set_aside"] and "low_score_ratio" in rec["reasons"], (rec["score_ratio"], rec["reasons"])
    # no xy projection: the step fails, naming the types
    try:
        align_one(src, 203)
    except RuntimeError as exc:
        assert "no xy projection" in str(exc) and "Primary" in str(exc), str(exc)
    else:
        raise AssertionError("a cell without an xy projection must fail the step")
    # the fine level: the finest that passes Block 0's size guard (12 Mpx returned), below the coarse level
    sv = cfg.survey
    assert CA.fine_level(3000, 4000, 4, sv) == 1                      # 1500 x 2000 = 3 Mpx at level 1
    assert CA.fine_level(8000, 9000, 4, sv) == 2                      # 18 Mpx at level 1, 4.5 Mpx at level 2
    try:
        CA.fine_level(40000, 40000, 3, sv)                            # 25 Mpx even at level 2
    except ValueError:
        pass
    else:
        raise AssertionError("a crop too large below the coarse level must be refused")
    # the runner, with a probe step after align
    calls = []
    saved = (dict(SC.STEPS), SC.ORDER, dict(SC.SAY))
    SC.STEPS["probe"], SC.ORDER = _probe_steps(calls, fail_on=(201,)), ("align", "probe")
    try:
        out = os.path.join(tmp, "survey")
        sel = selection_csv(tmp, [201, 202, 203])
        logs = []
        rows = SC.run(cfg, sel, out, src, ["align", "probe"], log=logs.append)
        st = {r["specimen_id"]: r for r in rows}
        assert st[201]["align"] == "done" and st[201]["probe"] == "failed" and "probe failure" in st[201]["error"]
        assert st[202]["align"] == "done" and st[202]["probe"] == "skipped:set_aside", st[202]
        assert st[203]["align"] == "failed" and st[203]["probe"] == "skipped:after_failure", st[203]
        assert calls == [201], calls
        assert os.path.exists(os.path.join(out, "201", "errors.log"))
        # resume: align is not run again, the failed probe is retried
        calls.clear()
        SC.STEPS["probe"] = _probe_steps(calls)
        n_before = src.fetcher.n_requests
        rows = SC.run(cfg, sel, out, src, ["align", "probe"], log=logs.append)
        st = {r["specimen_id"]: r for r in rows}
        assert st[201]["probe"] == "done" and calls == [201], (st, calls)
        assert src.fetcher.n_requests == n_before, "a done align must not fetch again"
        # include the set-aside cell
        rows = SC.run(cfg, sel, out, src, ["align", "probe"], include_set_aside=True, log=logs.append)
        assert {r["specimen_id"]: r for r in rows}[202]["probe"] == "done" and 202 in calls
        # --force align clears the later markers and runs both again on the named cell
        calls.clear()
        rows = SC.run(cfg, sel, out, src, ["align", "probe"], cells=[201], force=["align"], log=logs.append)
        assert len(rows) == 1 and rows[0]["align"] == "done" and calls == [201], (rows, calls)
        # a manual alignment, used as given
        cell = survey_io.CellDir(out, 202)
        survey_io.write_json(cell.path_of("align", "alignment_manual.json"),
                             dict(shift_x_px=3.0, shift_y_px=-4.0, flip_h_px=None, z0_um=0.0))
        rows = SC.run(cfg, sel, out, src, ["align", "probe"], cells=[202], force=["align"], log=logs.append)
        al = cell.context()["alignment"]
        assert al["method"] == "manual" and al["shift_x_px"] == 3.0 and not al["set_aside"] and rows[0]["probe"] == "done"
        # status from the markers alone
        st = {r["specimen_id"]: r for r in SC.status(out, sel, log=logs.append)}
        assert st[201]["align"] == "done" and st[203]["align"] == "pending", st
        try:
            SC.run(cfg, sel, out, src, ["align", "nonsense"], log=logs.append)
        except ValueError as exc:
            assert "nonsense" in str(exc)
        else:
            raise AssertionError("an unknown step must be refused")
    finally:
        SC.STEPS.clear()
        SC.STEPS.update(saved[0])
        SC.ORDER = saved[1]
    # the configuration's refusals
    for bad in (dict(align_fine_downsample=4), dict(align_null_n=3), dict(align_null_block_um=0.0),
                dict(projection_size_rel_tol=0.0), dict(align_min_score_ratio=0.0), dict(projection_xy_type=" "),
                dict(projection_xy_type=None)):
        try:
            dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, **bad)).validate()
        except ValueError:
            continue
        raise AssertionError("survey %s must be refused" % bad)


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
    print(_environment())
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, lines[0] if lines else ""))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
