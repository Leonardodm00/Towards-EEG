"""Reproducers of the tester's NOTE and MINOR findings on Blocks 16-18 that
are not pytest tests (run D5, 2026-11-13). Each prints one line of evidence.

Run (from the workstream directory):
    MPLBACKEND=Agg python tests/spec/repro_edges_D5.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec_helpers_D5 as H  # noqa: E402  (src/ and scripts/ on sys.path)

import numpy as np  # noqa: E402

import cell_pool as CPS  # noqa: E402  (scripts/cell_pool.py)
import survey_cells as SC  # noqa: E402
from allen_diameter.analysis import cell_alignment as CA  # noqa: E402
from allen_diameter.analysis import cell_selection as CS  # noqa: E402
from allen_diameter.analysis import coverage as COV  # noqa: E402
from allen_diameter.analysis import survey  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import cell_pool as CP  # noqa: E402
from allen_diameter.loading import survey_io, swc_io  # noqa: E402

CFG = default_config()
POOL_ROW = dict(specimen_id=5, species="Homo Sapiens", layer="3", dendrite_type="spiny", reconstruction_type="full",
                area="MTG", donor_id="1", hemisphere="left", normalized_depth=0.5)


def selection(tmp, sids):
    p = os.path.join(tmp, "sel.csv")
    CP.write_table(p, [dict(POOL_ROW, specimen_id=s, neuron_class="exc", group="exc|3", rank=0) for s in sids],
                   CP.SELECTION_COLUMNS)
    return p


def f4_signatures():
    try:
        a, b, S, S00 = CA.fine_search(np.ones((5, 5)), np.eye(5, dtype=bool), 1)
        print("F4 fine_search: 4 values as the spec lists")
    except ValueError as e:
        print("F4 fine_search(D, mask, half) -> 4 values (spec): %s" % e)
    k = CA.darkest([np.zeros((2, 2)), np.ones((2, 2))])
    print("F4 darkest(images) -> int (spec): got %r" % (k,))
    swc = __import__("pandas").DataFrame([(1, 1, 5.0, 5.0, 0, 1, -1), (2, 3, 6.0, 6.0, 0, 1, 1)],
                                         columns=["id", "type", "x", "y", "z", "r", "parent"])
    try:
        CA.fine_crop(swc, dict(shift_full_px=(0, 0), flip_y_full_h=None), 0.1144, 20.0, (1000, 1000))
        print("F4 fine_crop(swc, transform, rho0, ...) accepted")
    except Exception as e:                                               # noqa: BLE001
        print("F4 fine_crop(swc, transform, rho0, margin_um, extent) (spec order) -> %s: %s" % (type(e).__name__, e))


def f12_float32():
    rng = np.random.default_rng(5)
    D = rng.random((37, 52))
    m = rng.random((37, 52)) > 0.9
    ref = H.ref_search_score(D, [m], 3)
    print("F12 search_score relative error against the brute force: %.2e" % (abs(CA.search_score(D, [m], 3) - ref) / ref))


def f13_exit_code():
    tmp = tempfile.mkdtemp()
    sel = selection(tmp, [5])

    def boom(ctx, cfg):
        raise RuntimeError("Allen's server answered nothing")
    old = SC.STEPS["align"]
    SC.STEPS["align"] = boom
    try:
        rc = SC.main(["run", "--selection", sel, "--out-dir", os.path.join(tmp, "s"), "--cache-dir", tmp,
                      "--steps", "align", "--task-index", "0"])
    finally:
        SC.STEPS["align"] = old
    row = open(os.path.join(tmp, "s", "status.csv")).read().splitlines()[1]
    print("F13 survey_cells.py run --task-index 0, its one cell failing: exit status %d; status row: %s" % (rc, row))


def f14_named_cells():
    sel = CS.pick_cells([POOL_ROW], CFG, [5, 5])
    print("F14 pick_cells([5, 5]) -> %s, groups %s" % ([r["specimen_id"] for r in sel.selected], sel.groups))
    tmp = tempfile.mkdtemp()
    rows = SC.run(CFG, selection(tmp, [5]), os.path.join(tmp, "s"), None, ["align"], cells=[999], log=lambda m: None)
    print("F14 run(cells=[999]) on a selection without 999 -> %d rows, no error" % len(rows))


def f15_check_resume():
    tmp = tempfile.mkdtemp()
    pool = os.path.join(tmp, "pool.csv")
    CP.write_table(pool, [dict(POOL_ROW, specimen_id=s) for s in (5, 6)], CP.POOL_COLUMNS)
    calls = []

    def rec(sid):
        calls.append(int(sid))
        r = dict(specimen_id=int(sid), dz_um=0.28, scale_x=1.0, scale_y=1.0, pixel_um=0.1144, pixel_values="0.1144",
                 n_planes=4, n_planes_failed=0, width_px=1.0, height_px=1.0, error="")
        if len(calls) == 1:                                  # a transient network failure on the first cell
            r.update(dz_um=float("nan"), pixel_um=float("nan"), pixel_values="", n_planes=0,
                     error="images: api_query failed after 3 attempts")
        return r
    old = CPS.cell_pool.acquisition_record
    CPS.cell_pool.acquisition_record = rec
    try:
        acq = os.path.join(tmp, "acq.csv")
        CPS.run_check(CFG, pool, acq)
        _, summ = CPS.run_check(CFG, pool, acq)            # resumed with the network back
    finally:
        CPS.cell_pool.acquisition_record = old
    print("F15 check resumed after a transient failure: queries made %s; n_failed %d; mismatch %s"
          % (calls, summ["n_failed"], summ["mismatch"]))


def f16_json_nulls():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "c.json")
    survey_io.write_json(p, dict(bins=COV.coverage_table([], CFG)))
    b = json.load(open(p))["bins"]
    print("F16 coverage.json: the open bin's hi (inf) -> %r; an empty bin's share_inside (NaN) -> %r"
          % (b[-1]["hi"], b[0]["share_inside"]))


def f17_measure_nodes():
    tmp = tempfile.mkdtemp()
    path, _ = H.star_swc(os.path.join(tmp, "x.swc"), [[0.3, 0.3]])
    try:
        survey.measure_nodes(swc_io.read_swc(path), None, CFG, only=None, whole_stretches=False)
        print("F17 measure_nodes(only=None, whole_stretches=False): no error")
    except Exception as e:                                               # noqa: BLE001
        print("F17 measure_nodes(only=None, whole_stretches=False) -> %s: %s" % (type(e).__name__, e))


if __name__ == "__main__":
    for f in (f4_signatures, f12_float32, f13_exit_code, f14_named_cells, f15_check_resume, f16_json_nulls,
              f17_measure_nodes):
        f()
