"""Smoke test for S3b, the simulator's inputs across cells: d_max, the node
jitter, the camera -- Block 18 in specs/SPEC.md (D-065.2-3, D-085.8-9,
D-086; 2026-10-26).

Checks
    test_known_answer   d_max of three cells' hand-made diameters is their
                        largest, with its cell and node; the pooled robust SD
                        of hand-made offsets is 1.4826 x the MAD by hand; the
                        most common table set, the medians of B and noise; a
                        cell with median dz* 1.5 um flagged z0
    test_reference      robust_sd against scipy.stats.median_abs_deviation
                        (scale normal); the pooled p99, p99.5 against
                        numpy.percentile
    test_convergence    not a discretisation: skipped
    test_invariants     the cells' order changes nothing
    test_contract       the steps radius, registration and camera end to end
                        on the synthetic cell of fixtures_cell.py (after
                        align and nodes), then pool --what inputs: the
                        per-cell files, inputs.json's keys, the CSV, the
                        figure
    test_determinism    two pools give identical JSON
    test_edge_cases     a cell without ON nodes; no cell at all; the camera
                        step without nodes fails; the pool over hand-written
                        folders (d_max the largest 2r, the jitter from the ON
                        nodes only, the camera medians); the configuration's
                        refusals

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_s3b.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import survey_cells as SC  # noqa: E402
from allen_diameter.analysis import s3_inputs as S3I  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import cell_pool, survey_io  # noqa: E402

SEED = 20261026
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
T1 = [[1] * 64]
T2 = [[2] * 64]


def rad_cells():
    return [dict(specimen_id=30, node_ids=[1, 2, 3], D_allen=[0.4, 2.5, float("nan")]),
            dict(specimen_id=10, node_ids=[7, 8], D_allen=[3.1, 0.2]),
            dict(specimen_id=20, node_ids=[5], D_allen=[1.0])]


def jit_cells():
    return [dict(specimen_id=1, s_star=[0.1, -0.2, 0.0, 0.3], dz_star=[0.5, -0.5, 0.2, float("nan")]),
            dict(specimen_id=2, s_star=[0.05, -0.1], dz_star=[1.4, 1.6, 1.5])]


def cam_cells():
    return [dict(specimen_id=1, background_B_gl=200.0, noise_sd_gl=3.0, jpeg_qtables=T1),
            dict(specimen_id=2, background_B_gl=210.0, noise_sd_gl=None, jpeg_qtables=T2),
            dict(specimen_id=3, background_B_gl=190.0, noise_sd_gl=4.0, jpeg_qtables=T1)]


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    d = S3I.d_max(rad_cells())
    assert d["value_um"] == 3.1 and d["specimen_id"] == 10 and d["node_id"] == 7, d
    assert [r["specimen_id"] for r in d["per_cell"]] == [10, 20, 30] and d["per_cell"][2]["max_um"] == 2.5
    assert d["n_nodes"] == 5                                       # the NaN diameter is left out
    j = S3I.jitter(jit_cells(), 1.0)
    # pooled s*: median 0.025; |s - 0.025| = 0.075, 0.225, 0.025, 0.275, 0.025, 0.125, whose median is 0.1
    s = np.array([0.1, -0.2, 0.0, 0.3, 0.05, -0.1])
    assert abs(j["jitter_xy_um"] - 1.4826 * np.median(np.abs(s - np.median(s)))) < 1e-12
    assert abs(j["jitter_xy_um"] - 1.4826 * 0.1) < 1e-12
    z = np.array([0.5, -0.5, 0.2, 1.4, 1.6, 1.5])
    assert abs(j["jitter_z_um"] - 1.4826 * np.median(np.abs(z - np.median(z)))) < 1e-12
    assert j["z0_flagged"] == [2] and j["per_cell"][0]["median_dz_um"] == 0.2 and j["n_on_z"] == 6
    c = S3I.camera(cam_cells())
    assert c["jpeg_qtables"] == T1 and c["jpeg_cells"] == [1, 3] and c["n_table_sets"] == 2
    assert c["background_B_gl"] == 200.0 and c["noise_sd_gl"] == 3.5 and c["n_noise"] == 2


def test_reference():
    from scipy import stats
    rng = np.random.default_rng(SEED)
    x = rng.standard_t(3, 500)
    ref = float(stats.median_abs_deviation(x, scale="normal"))
    assert abs(S3I.robust_sd(x) - ref) <= 2e-6 * ref, (S3I.robust_sd(x), ref)     # 1.4826 against 1/Phi^-1(3/4)
    cells = [dict(specimen_id=i, node_ids=list(range(200)), D_allen=list(np.exp(rng.normal(-0.7, 0.6, 200))))
             for i in range(4)]
    d = S3I.d_max(cells)
    allD = np.concatenate([np.asarray(c["D_allen"]) for c in cells])
    assert d["p99_um"] == float(np.percentile(allD, 99.0)) and d["p99_5_um"] == float(np.percentile(allD, 99.5))
    assert d["value_um"] == float(allD.max())


def test_convergence():
    raise unittest.SkipTest("no discretisation: maxima, medians and MADs")


def test_invariants():
    for f, cells, extra in ((S3I.d_max, rad_cells(), ()), (S3I.jitter, jit_cells(), (1.0,)), (S3I.camera, cam_cells(), ())):
        a = f(cells, *extra)
        b = f(list(reversed(cells)), *extra)
        assert json.dumps(survey_io.clean(a), sort_keys=True) == json.dumps(survey_io.clean(b), sort_keys=True), f


class CellSource:
    def __init__(self, swc_path, fetcher, planes, cache_dir):
        self._swc, self.fetcher, self._planes, self.cache_dir = swc_path, fetcher, planes, cache_dir

    def swc_path(self, sid):
        return self._swc

    def images(self, sid):
        raise AssertionError("not used")

    def planes(self, sid):
        return self._planes


_RUN = {}


def run_cell():
    """The synthetic cell through align (manual identity), nodes, radius, registration and camera."""
    if _RUN:
        return _RUN
    import fixtures_cell
    cfg = default_config()
    cfg = dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, registration_min_nodes=5))
    tmp = tempfile.mkdtemp()
    swc, fetcher, planes, swc_path = fixtures_cell.synthetic_cell(tmp, cfg)
    out = os.path.join(tmp, "survey")
    rows = [dict(specimen_id=11, species="Homo Sapiens", layer="3", dendrite_type="spiny", reconstruction_type="full",
                 area="MTG", donor_id="9", hemisphere="left", normalized_depth=0.3, neuron_class="exc", group="exc|3",
                 rank=0)]
    sel = os.path.join(tmp, "selection.csv")
    cell_pool.write_table(sel, rows, cell_pool.SELECTION_COLUMNS)
    cell = survey_io.CellDir(out, 11)
    survey_io.write_json(cell.path_of("align", "alignment_manual.json"),
                         dict(shift_x_px=0.0, shift_y_px=0.0, flip_h_px=None, z0_um=0.0))
    cache = os.path.join(tmp, "cache")
    os.makedirs(cache)
    src = CellSource(swc_path, fetcher, planes, cache)
    logs = []
    st = SC.run(cfg, sel, out, src, list(SC.ORDER), log=logs.append)
    _RUN.update(cfg=cfg, out=out, sel=sel, cell=cell, st=st, logs=logs, src=src)
    return _RUN


def test_contract():
    r = run_cell()
    st, cell, cfg = r["st"][0], r["cell"], r["cfg"]
    for s in ("align", "nodes", "radius", "registration", "camera"):
        assert st[s] == "done", (s, st, r["logs"])
    sid = 11
    for f in (("radius", "radius_nodes_%d.csv" % sid), ("radius", "radius_summary_%d.csv" % sid),
              ("registration", "registration_%d.json" % sid), ("registration", "registration_summary_%d.json" % sid),
              ("registration", "stretches_%d.csv" % sid), ("camera", "camera_%d.json" % sid)):
        assert os.path.exists(cell.path_of(*f)), f
    m = survey_io.read_marker(cell, "radius")["summary"]
    assert m["n"] == 6 and abs(m["max_um"] - 0.6) < 1e-9, m                          # the fixture: 2r = 0.6 um
    cam = survey_io.read_json(cell.path_of("camera", "camera_%d.json" % sid))
    assert cam["renderer"].get("background_B_gl") is not None, cam
    assert any("no JPEG tables" in n for n in cam["notes"]), cam["notes"]           # the synthetic cache is empty
    sug = SC.pool_inputs(cfg, r["out"], r["sel"], log=r["logs"].append)
    s3b = os.path.join(r["out"], "s3b")
    js = survey_io.read_json(os.path.join(s3b, "inputs.json"))
    assert set(js["suggested"]) == {"phantom_d_max_um", "jitter_xy_um", "jitter_z_um", "background_B_gl",
                                    "noise_sd_gl", "jpeg_qtables"}, sorted(js["suggested"])
    assert abs(js["suggested"]["phantom_d_max_um"] - 0.6) < 1e-9 and js["d_max"]["specimen_id"] == sid
    assert js["suggested"]["background_B_gl"] == cam["renderer"]["background_B_gl"]
    assert {"d_max", "jitter", "camera", "note"} <= set(js)
    assert os.path.exists(os.path.join(s3b, "inputs_per_cell.csv")) and os.path.exists(os.path.join(s3b, "inputs.png"))
    head = open(os.path.join(s3b, "inputs_per_cell.csv")).readline().strip().split(",")
    assert {"specimen_id", "max_2r_um", "background_B_gl"} <= set(head), head
    reg = survey_io.read_json(cell.path_of("registration", "registration_%d.json" % sid))
    assert len(reg) >= 1 and all("verdict" in v and "lateral_offset_um" in v for v in reg.values()), reg


def fake_survey(out):
    """Two cells' folders written by hand with the files and markers the S3b steps leave."""
    import csv
    for sid, D, reg, B, N in ((1, [0.4, 2.0, 0.9], {"5": ("ON THE PROCESS", 0.2, 0.4), "6": ("NOT ON A VISIBLE PROCESS", 3.0, 3.0)}, 200.0, 3.0),
                              (2, [0.3, 1.0] + [0.5] * 200, {"7": ("ON THE PROCESS", -0.2, -0.4), "8": ("ALONGSIDE: a ridge", 1.0, 0.0)}, 220.0, 5.0)):
        cell = survey_io.CellDir(out, sid)
        with open(cell.path_of("radius", "radius_nodes_%d.csv" % sid), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["specimen", "node_id", "type", "radius_um", "diameter_um"])
            for i, d in enumerate(D):
                w.writerow([sid, 100 + i, 3, d / 2, d])
        survey_io.write_json(cell.path_of("registration", "registration_%d.json" % sid),
                             {k: dict(verdict=v, lateral_offset_um=s, z_offset_um=z) for k, (v, s, z) in reg.items()})
        survey_io.write_json(cell.path_of("camera", "camera_%d.json" % sid),
                             dict(renderer=dict(background_B_gl=B, noise_sd_gl=N, jpeg_qtables=T1)))
        for step in ("radius", "registration", "camera"):
            survey_io.write_marker(cell, step, {}, 0.0, "x")


def test_determinism():
    r = run_cell()
    s3b = os.path.join(r["out"], "s3b", "inputs.json")
    SC.pool_inputs(r["cfg"], r["out"], r["sel"], log=r["logs"].append)
    a = open(s3b).read()
    SC.pool_inputs(r["cfg"], r["out"], r["sel"], log=r["logs"].append)
    assert open(s3b).read() == a


def test_edge_cases():
    cfg = default_config()
    j = S3I.jitter([dict(specimen_id=1, s_star=[], dz_star=[]), dict(specimen_id=2, s_star=[0.1, 0.3], dz_star=[0.0])],
                   1.0)
    assert j["per_cell"][0]["sd_xy_um"] is None and j["per_cell"][0]["n_on"] == 0
    assert abs(j["jitter_xy_um"] - 1.4826 * 0.1) < 1e-12
    assert S3I.d_max([dict(specimen_id=1, node_ids=[1], D_allen=[float("nan")])])["value_um"] is None
    c = S3I.camera([dict(specimen_id=1, background_B_gl=None, noise_sd_gl=None, jpeg_qtables=None)])
    assert c["background_B_gl"] is None and c["jpeg_qtables"] is None and c["n_table_sets"] == 0
    with tempfile.TemporaryDirectory() as out:
        try:
            SC.pool_inputs(cfg, out, log=lambda m: None)
        except RuntimeError as exc:
            assert "no cell" in str(exc)
        else:
            raise AssertionError("pool inputs with no cell must say so")
        cell = survey_io.CellDir(out, 5)
        ctx = SC.Context(dict(specimen_id=5), cell, CellSource("x", None, None, out))
        try:
            SC.step_camera(ctx, cfg)
        except RuntimeError as exc:
            assert "needs the step nodes" in str(exc)
        else:
            raise AssertionError("camera without nodes must fail")
    # the step radius on an SWC with varied radii: the largest 2r and its node
    with tempfile.TemporaryDirectory() as out:
        swc = os.path.join(out, "c.swc")
        with open(swc, "w") as f:
            f.write("1 1 0 0 0 5 -1\n2 3 1 0 0 0.2 1\n3 3 2 0 0 0.9 2\n4 4 0 1 0 0.4 1\n5 2 0 2 0 3.0 1\n")
        ctx = SC.Context(dict(specimen_id=8), survey_io.CellDir(out, 8), CellSource(swc, None, None, out))
        ctx.data["swc_path"] = swc
        s = SC.step_radius(ctx, cfg)
        assert s["n"] == 3 and abs(s["max_um"] - 1.8) < 1e-12 and s["max_node_id"] == 3, s    # the axon (type 2) is out
    # the pool over hand-written folders: d_max the largest 2r (not a percentile), the jitter from the ON nodes only
    with tempfile.TemporaryDirectory() as out:
        fake_survey(out)
        sug = SC.pool_inputs(cfg, out, log=lambda m: None)
        js = survey_io.read_json(os.path.join(out, "s3b", "inputs.json"))["suggested"]
        assert js["phantom_d_max_um"] == 2.0 and sug["d_max"]["specimen_id"] == 1 and sug["d_max"]["node_id"] == 101
        assert js["phantom_d_max_um"] > sug["d_max"]["p99_5_um"]
        assert sug["jitter"]["n_on_xy"] == 2 and abs(js["jitter_xy_um"] - 1.4826 * 0.2) < 1e-12, sug["jitter"]
        assert js["background_B_gl"] == 210.0 and js["noise_sd_gl"] == 4.0 and js["jpeg_qtables"] == T1
    for bad in (dict(registration_per_type=0), dict(registration_min_nodes=1), dict(inputs_z0_flag_um=0.0)):
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
