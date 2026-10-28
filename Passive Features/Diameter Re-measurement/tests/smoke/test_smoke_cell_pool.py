"""Smoke test for the cell pool, the acquisition check and the stratified
selection of S3 -- Block 16.1 in specs/SPEC.md (D-065, D-085; 2026-10-26).

Checks
    test_known_answer   canned API rows (pages of 10 rows, raw names, with a
                        warning for the unordered pages; one page and no
                        warning at the default page size) give the human
                        cells with a reconstruction and nothing else; the
                        SDK's simplified names give the same records; groups
                        of 8, 5, 3, 2, 6 and 4 cells give 5, 5, 3, 2, 5 and 4
                        (sparsely spiny a third class, D-087); the
                        acquisition record of canned NeuronReconstruction and
                        SubImage rows -- the real list_images and plane_table
                        reached through a substituted api_query -- reads the
                        plane spacing, the pixel size and the planes
    test_reference      the selection equals an independent re-implementation
                        (sort by id, default_rng([seed, crc32(key)]).permutation,
                        the first n); the summary's counts by hand
    test_convergence    not a discretisation: skipped
    test_invariants     the pool's row order, and a cell removed from another
                        group, do not change a group's selection
    test_contract       CSV round trips of the pool, acquisition and selection
                        tables; the summary's keys; the configuration's survey
                        section and its refusals
    test_determinism    two runs of the script give identical files; another
                        seed changes a group of more than 5
    test_edge_cases     an empty pool, a required field missing, an unmapped
                        dendrite type, no layer, a NaN plane spacing, a failed
                        query, a cell with two pixel sizes, require_match
                        without the acquisition; the check resumed after an
                        interruption; the CLI's pool and select subcommands

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_cell_pool.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import warnings
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for _p in (SRC, WS / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import allen_image_io as aio  # noqa: E402
from allen_diameter.analysis import cell_selection as CS  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import cell_pool as CP  # noqa: E402

SEED = 20261026
REPORT_PACKAGES = ("numpy", "pandas")

# the canned pool: (layer, dendrite type, number of cells)
GROUPS = (("2", "spiny", 8), ("3", "spiny", 5), ("5", "spiny", 3), ("2", "aspiny", 2), ("3", "aspiny", 6),
          ("3", "sparsely spiny", 4))


def raw_rows():
    """API rows (raw names): the canned human cells with a reconstruction, plus mouse cells, human cells without a
    reconstruction and a repeated id, none of which belongs to the pool."""
    rows, sid = [], 500000000
    for layer, dtype, n in GROUPS:
        for _ in range(n):
            sid += 7
            rows.append({"specimen__id": sid, "donor__species": "Homo Sapiens", "structure__layer": layer,
                         "tag__dendrite_type": dtype, "nr__reconstruction_type": "full",
                         "structure_parent__acronym": "MTG", "donor__id": 9000 + sid % 5, "specimen__hemisphere": "left",
                         "csl__normalized_depth": 0.1 * (sid % 10), "line_name": None})
    for k in range(4):
        rows.append({"specimen__id": 400000000 + k, "donor__species": "Mus musculus", "structure__layer": "5",
                     "tag__dendrite_type": "spiny", "nr__reconstruction_type": "full", "csl__normalized_depth": None})
        rows.append({"specimen__id": 600000000 + k, "donor__species": "Homo Sapiens", "structure__layer": "3",
                     "tag__dendrite_type": "spiny", "nr__reconstruction_type": None, "csl__normalized_depth": None})
    rows.append(dict(rows[0], structure__layer="6"))          # a repeated id: the first row is kept
    return rows


def sdk_rows():
    """The same human cells under the Allen SDK's simplified names (CellTypesApi.simplify_cells_api)."""
    out = []
    for r in raw_rows():
        out.append({"id": r["specimen__id"], "species": r["donor__species"], "structure_layer_name": r["structure__layer"],
                    "dendrite_type": r["tag__dendrite_type"], "reconstruction_type": r["nr__reconstruction_type"],
                    "structure_area_abbrev": r.get("structure_parent__acronym"), "donor_id": r.get("donor__id"),
                    "structure_hemisphere": r.get("specimen__hemisphere"),
                    "normalized_depth": r.get("csl__normalized_depth")})
    return out


def paged_api(rows):
    calls = []

    def api(criteria, num_rows=200, start_row=0):
        assert criteria == CP.SPECIMEN_DETAIL_MODEL, criteria
        calls.append((num_rows, start_row))
        return rows[start_row:start_row + num_rows]
    return api, calls


class FakeAllen:
    """Answers allen_image_io.api_query for the queries of reconstruction_info and list_images, cell by cell:
    cells = {sid: dict(dz, scale, resolutions (one per plane), failed (plane indices), error (which query fails))}."""

    def __init__(self, cells):
        self.cells = cells

    def __call__(self, criteria, num_rows=200, start_row=0, timeout=60):
        m = re.match(r"model::NeuronReconstruction,rma::criteria,\[specimen_id\$eq(\d+)\]", criteria)
        if m:
            c = self.cells[int(m.group(1))]
            if c.get("error") == "reconstruction":
                raise RuntimeError("api_query failed after 4 attempts: timeout")
            return [] if c.get("dz") is None and c.get("error") == "no_row" else [
                {"scale_factor_x": c["scale"], "scale_factor_y": c["scale"], "scale_factor_z": c["dz"]}]
        m = re.match(r"model::Specimen,rma::criteria,\[id\$eq(\d+)\],rma::include,data_sets", criteria)
        if m:
            return [{"id": int(m.group(1)), "data_sets": [{"id": int(m.group(1)) + 1}]}]
        m = re.match(r"model::SubImage,rma::criteria,data_set\[id\$eq(\d+)\]", criteria)
        if m:
            sid = int(m.group(1)) - 1
            c = self.cells[sid]
            if c.get("error") == "images":
                return []
            rows = [{"id": sid * 100 + k, "image_type": "Primary", "section_number": 10 + k, "width": 7000 + k,
                     "height": 9000, "resolution": res, "data_set_id": sid + 1, "failed": k in c.get("failed", ())}
                    for k, res in enumerate(c["resolutions"])]
            return rows[start_row:start_row + num_rows]
        m = re.match(r"model::SubImage,rma::criteria,\[specimen_id\$eq(\d+)\]", criteria)
        if m:
            sid = int(m.group(1))
            if self.cells[sid].get("error") == "images":
                return []
            return [{"id": sid * 100 + 99, "image_type": "Projection", "section_number": None, "width": 7000,
                     "height": 9000, "resolution": 0.1144, "data_set_id": None, "failed": False}]
        raise AssertionError("unexpected query %r" % criteria)


def with_fake_api(cells, fn):
    saved = aio.api_query
    aio.api_query = FakeAllen(cells)
    try:
        return fn()
    finally:
        aio.api_query = saved


def acquisition_cells():
    return {11: dict(dz=0.28, scale=0.1144, resolutions=[0.1144] * 5, failed=(4,)),
            12: dict(dz=0.30, scale=0.1144, resolutions=[0.1144] * 3),
            13: dict(dz=0.28, scale=0.1144, resolutions=[0.1144, 0.1144, 0.2288]),
            14: dict(dz=0.28, scale=0.1144, resolutions=[0.1144] * 2, error="reconstruction"),
            15: dict(dz=0.28, scale=0.1144, resolutions=[], error="images"),
            16: dict(dz=float("nan"), scale=0.1144, resolutions=[0.1144] * 4)}


def independent_selection(pool, cfg):
    """Block 16.1 (d) written again without cell_selection."""
    cmap = {t.lower(): c for t, c in cfg.survey.class_map}
    groups = {}
    for r in pool:
        cls = cmap.get(str(r["dendrite_type"]).lower())
        if cls is None or r["layer"] is None:
            continue
        groups.setdefault("%s|%s" % (cls, r["layer"]), []).append(int(r["specimen_id"]))
    out = {}
    for key, ids in groups.items():
        ids = sorted(ids)
        perm = np.random.default_rng([cfg.survey.selection_seed, zlib.crc32(key.encode("utf-8"))]).permutation(len(ids))
        out[key] = [ids[i] for i in perm[:min(cfg.survey.cells_per_group, len(ids))]]
    return out


def chosen(sel):
    out = {}
    for r in sorted(sel.selected, key=lambda r: (r["group"], r["rank"])):
        out.setdefault(r["group"], []).append(int(r["specimen_id"]))
    return out


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = default_config()
    rows = raw_rows()
    api, calls = paged_api(rows)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = CP.fetch_specimen_details(api=api, page_rows=10)
    assert got == rows and calls == [(10, 0), (10, 10), (10, 20), (10, 30), (10, 40)][:len(calls)], calls
    assert len(calls) == len(rows) // 10 + 1, (len(calls), len(rows))          # the last page is short
    assert any(issubclass(w.category, RuntimeWarning) and "without an order" in str(w.message) for w in caught), \
        "a query that needs several unordered pages must warn"
    api1, calls1 = paged_api(rows)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert CP.fetch_specimen_details(api=api1) == rows                     # the default: one page
    assert calls1 == [(cfg.survey.pool_page_rows, 0)] and not caught, (calls1, [str(w.message) for w in caught])
    pool = CP.pool_records(got)
    n_human = sum(n for _, _, n in GROUPS)
    assert len(pool) == n_human, len(pool)
    assert all(r["species"] == "Homo Sapiens" and r["reconstruction_type"] == "full" for r in pool)
    assert [r["specimen_id"] for r in pool] == sorted(r["specimen_id"] for r in pool)
    first = next(r for r in pool if r["specimen_id"] == rows[0]["specimen__id"])
    assert first["layer"] == "2", first                                          # the first row of a repeated id
    assert CP.pool_records(sdk_rows()) == pool, "the SDK's simplified names must give the same records"
    sel = CS.select_cells(pool, cfg)
    assert {k: v for k, v in sel.groups.items()} == {"exc|2": (8, 5), "exc|3": (5, 5), "exc|5": (3, 3),
                                                     "inh|2": (2, 2), "inh|3": (6, 5),
                                                     "sparsely_spiny|3": (4, 4)}, sel.groups   # D-087: a third class
    assert len(sel.selected) == 24 and not sel.excluded, sel.excluded
    # without the third class in the map (the draft's default) its cells have no class and stay out
    two = dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, class_map=cfg.survey.class_map[:2]))
    sel2 = CS.select_cells(pool, two)
    assert len(sel2.selected) == 20 and sum(1 for v in sel2.excluded.values() if v == "no_class") == 4
    # the acquisition record through the real list_images / plane_table / reconstruction_info
    recs = with_fake_api(acquisition_cells(), lambda: [CP.acquisition_record(s) for s in sorted(acquisition_cells())])
    r11, r12, r13, r14, r15, r16 = recs
    assert r11["dz_um"] == 0.28 and r11["pixel_um"] == 0.1144 and r11["n_planes"] == 4 and r11["n_planes_failed"] == 1 \
        and r11["error"] == "" and r11["width_px"] == 7001.5 and r11["scale_x"] == 0.1144, r11
    assert r12["dz_um"] == 0.30 and r12["n_planes"] == 3, r12
    assert math.isnan(r13["pixel_um"]) and r13["pixel_values"] == "0.1144;0.2288", r13
    assert r14["error"].startswith("reconstruction:") and r14["n_planes"] == 2 and math.isnan(r14["dz_um"]), r14
    assert r15["error"].startswith("images:") and r15["n_planes"] == 0 and r15["dz_um"] == 0.28, r15
    assert math.isnan(r16["dz_um"]) and r16["error"] == "", r16
    verdicts = [CS.matches_acquisition(r, cfg) for r in recs]
    assert verdicts == [(True, ""), (False, "dz"), (False, "pixel"), (False, "error"), (False, "error"),
                        (False, "dz")], verdicts


def test_reference():
    cfg = default_config()
    pool = CP.pool_records(raw_rows())
    assert chosen(CS.select_cells(pool, cfg)) == independent_selection(pool, cfg)
    cfg3 = dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, cells_per_group=3, selection_seed=7))
    assert chosen(CS.select_cells(pool, cfg3)) == independent_selection(pool, cfg3)
    recs = with_fake_api(acquisition_cells(), lambda: [CP.acquisition_record(s) for s in sorted(acquisition_cells())])
    summ = CS.acquisition_summary(recs, cfg)
    assert summ["n_cells"] == 6 and summ["n_failed"] == 2 and summ["n_match"] == 1, summ
    assert summ["dz_um_counts"] == {"0.2800": 3, "0.3000": 1, "nan": 2}, summ["dz_um_counts"]
    assert summ["pixel_um_counts"] == {"0.1144": 4, "multiple: 0.1144;0.2288": 1, "nan": 1}, summ["pixel_um_counts"]
    assert summ["mismatch"] == {"12": "dz", "13": "pixel", "14": "error", "15": "error", "16": "dz"}, summ["mismatch"]


def test_convergence():
    raise unittest.SkipTest("no discretisation: a selection and a table read")


def test_invariants():
    cfg = default_config()
    pool = CP.pool_records(raw_rows())
    ref = chosen(CS.select_cells(pool, cfg))
    rng = np.random.default_rng(SEED)
    shuffled = [pool[i] for i in rng.permutation(len(pool))]
    assert chosen(CS.select_cells(shuffled, cfg)) == ref, "the row order must not matter"
    drop = next(r for r in pool if r["dendrite_type"] == "aspiny" and r["layer"] == "3")
    fewer = [r for r in pool if r is not drop]
    got = chosen(CS.select_cells(fewer, cfg))
    assert all(got[k] == ref[k] for k in ref if k != "inh|3"), "another group's membership must not move a group"
    # require_match keeps only the cells whose acquisition matches; the groups are drawn over those
    acq = [dict(specimen_id=r["specimen_id"], dz_um=0.28 if i % 4 else 0.30, pixel_um=0.1144, pixel_values="0.1144",
                n_planes=10, n_planes_failed=0, width_px=7000.0, height_px=9000.0, error="")
           for i, r in enumerate(pool)]
    sel = CS.select_cells(pool, cfg, acq, require_match=True)
    bad = {r["specimen_id"] for i, r in enumerate(pool) if i % 4 == 0}
    assert all(r["specimen_id"] not in bad for r in sel.selected)
    assert all(sel.excluded[s] in ("acquisition_mismatch:dz", "no_class") for s in bad), sel.excluded
    assert chosen(sel) == independent_selection([r for r in pool if r["specimen_id"] not in bad], cfg)


def test_contract():
    cfg = default_config()
    pool = CP.pool_records(raw_rows())
    recs = with_fake_api(acquisition_cells(), lambda: [CP.acquisition_record(s) for s in sorted(acquisition_cells())])
    sel = CS.select_cells(pool, cfg)
    with tempfile.TemporaryDirectory() as tmp:
        for name, rows, cols, types in (("pool.csv", pool, CP.POOL_COLUMNS, CP.POOL_TYPES),
                                        ("acq.csv", recs, CP.ACQUISITION_COLUMNS, CP.ACQUISITION_TYPES),
                                        ("sel.csv", list(sel.selected), CP.SELECTION_COLUMNS, CP.SELECTION_TYPES)):
            path = os.path.join(tmp, name)
            CP.write_table(path, rows, cols)
            back = CP.read_table(path, types)
            assert len(back) == len(rows), name
            for a, b in zip(rows, back):
                for k in cols:
                    x, y = a.get(k), b[k]
                    # the tables' convention: an empty cell is absent, so "" and None read back as None
                    same = (x == y) or (x in ("", None) and y is None) or (
                        isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))
                    assert same, (name, k, x, y)
            assert not os.path.exists(path + ".tmp")
        try:
            CP.write_table(os.path.join(tmp, "x.csv"), [dict(pool[0], extra=1)], CP.POOL_COLUMNS)
        except ValueError:
            pass
        else:
            raise AssertionError("a key outside the columns must be refused")
    summ = CS.acquisition_summary(recs, cfg)
    assert set(summ) == {"n_cells", "n_failed", "dz_um_counts", "pixel_um_counts", "n_match", "mismatch",
                         "configuration"}, sorted(summ)
    json.dumps(summ)
    assert cfg.survey.cells_per_group == 5 and cfg.survey.species == "Homo Sapiens"
    for bad in (dict(cells_per_group=0), dict(class_map=(("spiny", "exc"), ("Spiny", "inh"))),
                dict(class_map=(("spiny", ""),)), dict(acquisition_rel_tol=-1.0), dict(pool_page_rows=0)):
        try:
            dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, **bad)).validate()
        except ValueError:
            continue
        raise AssertionError("survey %s must be refused" % bad)
    # the survey section is in no signature
    other = dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, cells_per_group=9))
    assert other.signature_hash("estimator") == cfg.signature_hash("estimator")
    assert other.signature_hash("simulation") == cfg.signature_hash("simulation")


def _run_script(args, tmp):
    return subprocess.run([sys.executable, str(WS / "scripts" / "cell_pool.py")] + args, capture_output=True,
                          text=True, timeout=300, cwd=tmp)


def test_determinism():
    """Two runs give the same selection; another seed changes a large group. Also (2026-10-28, D-099.3) `pick`."""
    cfg = default_config()
    pool = CP.pool_records(raw_rows())
    with tempfile.TemporaryDirectory() as tmp:
        pool_csv = os.path.join(tmp, "pool.csv")
        CP.write_table(pool_csv, pool, CP.POOL_COLUMNS)
        outs = []
        for k in range(2):
            out = os.path.join(tmp, "sel%d.csv" % k)
            p = _run_script(["select", "--pool", pool_csv, "--out", out], tmp)
            assert p.returncode == 0, p.stderr
            outs.append((open(out).read(), json.load(open(os.path.join(tmp, "sel%d.json" % k)))))
        assert outs[0][0] == outs[1][0], "two runs must give the same selection"
        assert {k: v for k, v in outs[0][1].items() if k != "pool"} == {k: v for k, v in outs[1][1].items() if k != "pool"}
        # pick (D-099.3): the named cells in the order given, each with its class and group; the file reads back as a
        # selection; a cell outside the pool, without a class, or (require-match) without a record is refused
        ids = [int(r["specimen_id"]) for r in pool]
        want = [ids[5], ids[0]]
        out = os.path.join(tmp, "pick.csv")
        p = _run_script(["pick", "--pool", pool_csv, "--cells", "%d,%d" % tuple(want), "--out", out], tmp)
        assert p.returncode == 0 and "pick: 2 cells" in p.stdout, (p.stdout, p.stderr)
        back = CP.read_table(out, CP.SELECTION_TYPES)
        assert [int(r["specimen_id"]) for r in back] == want and all(r["rank"] == 0 for r in back)
        by_id = {int(r["specimen_id"]): r for r in pool}
        for r in back:
            src = by_id[int(r["specimen_id"])]
            assert r["neuron_class"] == CS.neuron_class(src["dendrite_type"], cfg.survey.class_map)
            assert r["group"] == "%s|%s" % (r["neuron_class"], src["layer"])
        js = json.load(open(os.path.join(tmp, "pick.json")))
        assert js["picked"] == want and js["excluded"] == {} and sum(v["selected"] for v in js["groups"].values()) == 2
        sel_pick = CS.pick_cells(pool, cfg, want)
        assert [int(r["specimen_id"]) for r in sel_pick.selected] == want and sel_pick.excluded == {}
        for bad, text in (([123], "not in the pool"),):
            try:
                CS.pick_cells(pool, cfg, bad)
            except ValueError as exc:
                assert text in str(exc), str(exc)
            else:
                raise AssertionError("pick must refuse %r" % bad)
        try:
            CS.pick_cells(pool, cfg, want, None, require_match=True)
        except ValueError:
            pass
        else:
            raise AssertionError("require_match without records must be refused")
        try:
            CS.pick_cells(pool, cfg, want, [], require_match=True)
        except ValueError as exc:
            assert "no acquisition record" in str(exc)
        else:
            raise AssertionError("a cell without a record must be refused under require_match")
        p = _run_script(["pick", "--pool", pool_csv, "--cells", "123", "--out", out], tmp)
        assert p.returncode != 0
    a = chosen(CS.select_cells(pool, cfg))
    other = dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, selection_seed=cfg.survey.selection_seed + 1))
    b = chosen(CS.select_cells(pool, other))
    assert a["exc|2"] != b["exc|2"] or a["inh|3"] != b["inh|3"], "another seed must change a group of more than 5"
    assert len(a["exc|5"]) == 3 and set(a["exc|5"]) == set(b["exc|5"]), "a group of 3 is taken whole, whatever the seed"


def test_edge_cases():
    cfg = default_config()
    sel = CS.select_cells([], cfg)
    assert sel.selected == () and sel.groups == {} and sel.excluded == {}
    row = dict(raw_rows()[0])
    del row["tag__dendrite_type"]
    try:
        CP.pool_records([row])
    except ValueError as exc:
        assert "dendrite_type" in str(exc) and "specimen__id" in str(exc), str(exc)
    else:
        raise AssertionError("a missing required field must raise")
    pool = CP.pool_records(raw_rows())
    nolayer = [dict(pool[0], layer=None)] + pool[1:]
    sel = CS.select_cells(nolayer, cfg)
    assert sel.excluded[pool[0]["specimen_id"]] == "no_layer"
    assert CS.neuron_class("Sparsely Spiny", cfg.survey.class_map) == "sparsely_spiny"           # D-087
    assert CS.neuron_class("pyramidal", cfg.survey.class_map) is None
    assert CS.neuron_class(" Spiny ", cfg.survey.class_map) == "exc" and CS.neuron_class(None, cfg.survey.class_map) is None
    try:
        CS.select_cells(pool, cfg, None, require_match=True)
    except ValueError:
        pass
    else:
        raise AssertionError("require_match without acquisition records must raise")
    sel = CS.select_cells(pool, cfg, [], require_match=True)
    assert sel.selected == () and set(sel.excluded.values()) <= {"no_class", "acquisition_missing"}
    # the check resumed after an interruption, in-process with the substituted API
    import cell_pool as script
    cells = acquisition_cells()
    with tempfile.TemporaryDirectory() as tmp:
        pool_csv = os.path.join(tmp, "pool.csv")
        CP.write_table(pool_csv, [dict(pool[0], specimen_id=s) for s in sorted(cells)], CP.POOL_COLUMNS)
        out = os.path.join(tmp, "acq.csv")
        recs1, _ = with_fake_api(cells, lambda: script.run_check(cfg, pool_csv, out, max_cells=2))
        assert [r["specimen_id"] for r in recs1] == [11, 12]
        recs2, summ = with_fake_api(cells, lambda: script.run_check(cfg, pool_csv, out))
        assert [r["specimen_id"] for r in recs2] == sorted(cells) and summ["n_cells"] == 6 and summ["n_match"] == 1
        assert os.path.exists(os.path.join(tmp, "acq.json"))
        recs3, _ = with_fake_api(cells, lambda: script.run_check(cfg, pool_csv, out, fresh=True, max_cells=1))
        assert [r["specimen_id"] for r in recs3] == [11]
        # the CLI's pool (from a CSV with the SDK's names) and select (with the acquisition, require-match)
        sdk_csv = os.path.join(tmp, "sdk.csv")
        rows = sdk_rows()
        CP.write_table(sdk_csv, rows, tuple(rows[0]))
        p = _run_script(["pool", "--from-csv", sdk_csv, "--out", os.path.join(tmp, "pool2.csv")], tmp)
        assert p.returncode == 0 and "pool: 28 Homo Sapiens cells" in p.stdout and "sparsely_spiny|3 4" in p.stdout \
            and "(left out of the selection): none" in p.stdout, (p.stdout, p.stderr)
        # a map without the third class, in a hand-written --config-json (to_json writes the signatures only, so
        # the survey section is given by hand, as the runbook says; the other sections take their defaults)
        two_json = os.path.join(tmp, "two_classes.json")
        with open(two_json, "w") as f:
            json.dump({"survey": {"class_map": [list(x) for x in cfg.survey.class_map[:2]]}}, f)
        p = _run_script(["pool", "--from-csv", sdk_csv, "--out", os.path.join(tmp, "pool3.csv"), "--config-json",
                         two_json], tmp)
        assert p.returncode == 0 and "{'sparsely spiny': 4}" in p.stdout, (p.stdout, p.stderr)
        back = CP.read_table(os.path.join(tmp, "pool2.csv"), CP.POOL_TYPES)
        assert [r["specimen_id"] for r in back] == [r["specimen_id"] for r in pool]
        acq_csv = os.path.join(tmp, "acq2.csv")
        acq = [dict(specimen_id=r["specimen_id"], dz_um=0.28, pixel_um=0.1144 if i % 3 else 0.2288,
                    pixel_values="0.1144", n_planes=5, n_planes_failed=0, width_px=7000.0, height_px=9000.0, error="")
               for i, r in enumerate(back)]
        CP.write_table(acq_csv, acq, CP.ACQUISITION_COLUMNS)
        p = _run_script(["select", "--pool", os.path.join(tmp, "pool2.csv"), "--acquisition", acq_csv,
                         "--require-match", "--per-group", "2", "--out", os.path.join(tmp, "sel.csv")], tmp)
        assert p.returncode == 0 and "select: " in p.stdout and "acquisition_mismatch:pixel" in p.stdout, \
            (p.stdout, p.stderr)
        sel = CP.read_table(os.path.join(tmp, "sel.csv"), CP.SELECTION_TYPES)
        n_groups = len({r["group"] for r in sel})
        assert all(r["rank"] < 2 for r in sel) and n_groups <= 6 and len(sel) <= 2 * n_groups, (n_groups, len(sel))
        bad = {r["specimen_id"] for i, r in enumerate(back) if i % 3 == 0}
        assert not bad & {r["specimen_id"] for r in sel}
        for args in (["pool", "--help"], ["check", "--help"], ["select", "--help"]):
            assert _run_script(args, tmp).returncode == 0


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
