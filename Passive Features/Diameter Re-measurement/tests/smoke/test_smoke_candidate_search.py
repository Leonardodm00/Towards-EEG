"""Smoke test for Block 10.C in specs/SPEC.md -- the calibration candidates
found over the whole cell from its trace, for both sets (D-109, the far set;
D-112, both sets; 2026-10-29), before any image is read: the trace's tilt
against the set's bound, the stripe of 10.S at the trace's plane, the stripes
of the set's fit planes (far: all 24; near: k_tr +- 3 by pairs until the near
fit's minimum), then the greedy choice at the set's spacing along each
stretch; calibrate_kernel.cell_candidates, which `real` and S3's calibration
step use.

Checks
    test_known_answer   painted cells (a soma and straight dendrites, SWC
                        written and read back): a flat far-range dendrite
                        with nothing beside it passes with every far-fit
                        plane usable; a flat near-range one passes after the
                        trace's plane and the pair +-1 (3 planes evaluated,
                        3 usable); each with a parallel in-focus dendrite
                        beside it fails (not_isolated:side_short or
                        few_planes); a stretch tilted at 30 deg is not_flat
                        with phi 30 deg; at 12 deg a near-range stretch is
                        not_flat (10 deg) and a far-range one is not (15
                        deg); each search lists its own range's nodes only
    test_reference      a record's n_pixels_k, reason, n_fit and
                        n_fit_usable equal calibration.support_masks called
                        directly on the same view (the trace's centre,
                        heading and plane, the given B_bar and mu, Allen's
                        radius; all of the stage's planes in one call, the
                        soma at the stage's blur), with the near walk's
                        early stop recomputed from those stripes; the flat
                        trace's heading and tilt equal the painted line's
                        (1e-9)
    test_convergence    not a discretisation: skipped
    test_invariants     choose() on hand-made records, per set: only passing
                        records of that set, each pair kept on one stretch
                        at least the set's spacing apart (near and far
                        spacings differ), the best first (far: most usable
                        planes, then pixels; near: most pixels), and every
                        passing record left out lies within the spacing of
                        a kept one ranked before it (the greedy rule)
    test_contract       the record keys and types (RECORD_KEYS); the CSV
                        through table_io, a header even with no record;
                        typical_background reads the medians of the rows in
                        S only; calibrate_kernel.py real hands scan_cell, set
                        by set, the search's chosen ids or the table's
                        candidates (both "cell", both "table", mixed), writes
                        <set>_search_<specimen>.csv only for the searched
                        sets and prints their lines and the knobs line
    test_determinism    two searches give identical records, both sets
    test_edge_cases     a cell without nodes of a set's range: no record;
                        B_bar and mu NaN: the geometric margins, the search
                        runs; spacing 0: every passing node kept; with
                        support_masks stubbed by the plane offset, the far
                        threshold (5 of 24 usable: few_planes; 6: passes)
                        and the near walk (which pairs are evaluated, when
                        it stops, n_fit and n_fit_usable, few_planes); a
                        bad set name or a validation error for a bad
                        <set>_candidates or spacing

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_candidate_search.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib.metadata
import io
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
import types
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for _p in (SRC, HERE, WS / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter.analysis import calibration as CAL  # noqa: E402
from allen_diameter.analysis import calibration_scene as CS  # noqa: E402
from allen_diameter.analysis import candidate_search as FS  # noqa: E402
from allen_diameter.analysis import cell  # noqa: E402
from allen_diameter.config import default_config, with_overrides  # noqa: E402
from allen_diameter.loading import swc_io, table_io  # noqa: E402

SEED = 20261029
REPORT_PACKAGES = ("numpy", "scipy")
NAN = float("nan")
B_BAR, MU = 200.0, 1.0           # gl, /um: the painted cells' background and absorption
Z = 10.0                         # um: the depth of the flat dendrites (plane 36 at 0.28 um)
R_FAR, R_NEAR, R_THIN = 0.35, 0.3, 0.25   # um: Allen 2r 0.7 (far set), 0.6 and 0.5 (near set)


# --------------------------------------------------------------- helpers ---

def cfg_with(**cal):
    """A smaller patch than the default 25 um (the runs reach its edge; only faster)."""
    cfg = default_config()
    cal = dict({"patch_half_um": 16.0}, **cal)
    return dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, **cal))


def write_cell(tmp, branches, name="cell.swc"):
    """SWC with a soma (id 1, type 1, radius 5 um) at (0, 0, Z) and one dendrite (type 3) per (points (n, 3),
    radius) leaving it; returns the SWC read back (read_swc)."""
    lines = ["# painted cell", "1 1 0.0000 0.0000 %.4f 5.0000 -1" % Z]
    nid = 1
    for pts, r in branches:
        parent = 1
        for p in np.atleast_2d(pts):
            nid += 1
            lines.append("%d 3 %.4f %.4f %.4f %.4f %d" % (nid, p[0], p[1], p[2], r, parent))
            parent = nid
    path = os.path.join(tmp, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return swc_io.read_swc(path)


def line(x0, x1, y=0.0, step=2.0, z=Z, tilt_deg=0.0, sign=1.0):
    """Points from (x0, y, z) every `step` um along x (sign) tilted out of the plane by tilt_deg."""
    n = int(round(abs(x1 - x0) / step)) + 1
    t = np.arange(n) * step
    a = math.radians(tilt_deg)
    return np.column_stack([x0 + sign * t * math.cos(a), np.full(n, y), z + t * math.sin(a)])


def far_dendrite():
    return (line(6.0, 50.0), R_FAR)


def near_dendrite():
    return (line(-6.0, -50.0, sign=-1.0), R_THIN)


def by_id(records):
    return {int(r["node_id"]): r for r in records}


def id_at(swc, x, y):
    """The node id nearest (x, y) among the dendrite nodes."""
    d = np.hypot(swc.xyz[:, 0] - x, swc.xyz[:, 1] - y)
    d[np.asarray(swc.types) == 1] = np.inf
    return int(swc.ids[int(np.argmin(d))])


def ids_of(swc, cfg, which):
    return {int(i) for i, r in zip(swc.ids, swc.radius) if CAL.calibration_set(2 * r, cfg) == which}


def near_walk(ok_at, P, need):
    """The near walk of 10.C recomputed from a map offset -> usable: (n_fit, n_fit_usable)."""
    n_fit, n_ok = 1, 1
    for d in range(1, P + 1):
        if n_ok >= need or n_ok + 2 * (P - d + 1) < need:
            break
        n_fit += 2
        n_ok += int(ok_at(-d)) + int(ok_at(d))
    return n_fit, n_ok


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = cfg_with()
    cal = cfg.calibration
    k_z = int(round(Z / cfg.acquisition.dz_um))
    with tempfile.TemporaryDirectory() as tmp:
        clear = write_cell(tmp, [far_dendrite(), near_dendrite()], "clear.swc")
        far = by_id(FS.search(clear, cfg, "far", None, B_BAR, MU))
        near = by_id(FS.search(clear, cfg, "near", None, B_BAR, MU))
        # each search lists its own range's nodes only
        assert set(far) == ids_of(clear, cfg, "far") and set(near) == ids_of(clear, cfg, "near"), (sorted(far), sorted(near))
        assert all(r["set"] == "far" for r in far.values()) and all(r["set"] == "near" for r in near.values())
        good = far[id_at(clear, 44.0, 0.0)]
        assert good["reason"] == "" and good["n_fit_usable"] == good["n_fit"] == 24, good
        assert abs(good["phi_deg"]) < 1e-6 and good["k_trace"] == k_z, good
        # a clear near node: the trace's plane and the pair +-1 are usable, so the walk stops at 3 of 3
        good = near[id_at(clear, -44.0, 0.0)]
        assert good["reason"] == "" and good["n_fit"] == 3 and good["n_fit_usable"] == 3, good
        assert cal.min_planes_per_node == 3 and abs(good["phi_deg"]) < 1e-6 and good["k_trace"] == k_z, good
        # each with a parallel in-focus dendrite beside it: the side run is short (far at 4 um, near at 2.5 um)
        crowded = write_cell(tmp, [far_dendrite(), (line(6.0, 50.0, y=4.0), R_NEAR),
                                   near_dendrite(), (line(-6.0, -50.0, y=2.5, sign=-1.0), R_THIN)], "crowded.swc")
        far2 = by_id(FS.search(crowded, cfg, "far", None, B_BAR, MU))
        near2 = by_id(FS.search(crowded, cfg, "near", None, B_BAR, MU))
        for x in (30.0, 36.0, 44.0):
            r = far2[id_at(crowded, x, 0.0)]
            assert r["reason"] in ("not_isolated:side_short", "few_planes"), (x, r)
            r = near2[id_at(crowded, -x, 0.0)]
            assert r["reason"] in ("not_isolated:side_short", "few_planes"), (-x, r)
        # tilted 30 deg: not_flat in both sets, phi read as 30 deg
        tilted = write_cell(tmp, [(line(-6.0, -40.0, sign=-1.0, tilt_deg=30.0), R_FAR),
                                  (line(6.0, 40.0, tilt_deg=30.0), R_THIN)], "tilted.swc")
        for which, inner in (("far", lambda r: r["x_um"] < -10.0), ("near", lambda r: r["x_um"] > 10.0)):
            got = [r for r in FS.search(tilted, cfg, which, None, B_BAR, MU) if inner(r)]
            # 0.01 deg: the SWC holds 4 decimals (1e-4 um over 2 um steps moves the angle by up to ~3e-3 deg)
            assert got and all(r["reason"] == "not_flat" and abs(r["phi_deg"] - 30.0) < 0.01 for r in got), (which, got)
        # 12 deg: beyond the near set's bound, within the far set's
        assert cal.node_phi_max_deg < 12.0 < cal.node_phi_max_deg_far
        mild = write_cell(tmp, [(line(-6.0, -40.0, sign=-1.0, tilt_deg=12.0), R_FAR),
                                (line(6.0, 40.0, tilt_deg=12.0), R_THIN)], "mild.swc")
        got = [r for r in FS.search(mild, cfg, "far", None, B_BAR, MU) if r["x_um"] < -10.0]
        assert got and all(r["reason"] != "not_flat" and abs(r["phi_deg"] - 12.0) < 0.01 for r in got), got
        got = [r for r in FS.search(mild, cfg, "near", None, B_BAR, MU) if r["x_um"] > 10.0]
        assert got and all(r["reason"] == "not_flat" and abs(r["phi_deg"] - 12.0) < 0.01 for r in got), got


def test_reference():
    cfg = cfg_with()
    cal, dz, res0 = cfg.calibration, cfg.acquisition.dz_um, cfg.acquisition.res0_um
    with tempfile.TemporaryDirectory() as tmp:
        swc = write_cell(tmp, [far_dendrite(), (line(6.0, 50.0, y=9.0), R_NEAR), near_dendrite(),
                               (line(-6.0, -50.0, y=-5.0, sign=-1.0), R_THIN)])
        xyz_img = cell.to_image_um(swc.xyz, res0)
        scene = cell.cell_scene(swc, cfg.acquisition.dendrite_swc_types, res0, None)
        for which in ("far", "near"):
            recs = by_id(FS.search(swc, cfg, which, None, B_BAR, MU))
            checked = 0
            for run in cell.stretches(swc, cfg.acquisition.dendrite_swc_types):
                branch, off = cell.stretch_branch(swc, run, xyz_img)
                for t, row in enumerate(run):
                    nid = int(swc.ids[row])
                    if nid not in recs or recs[nid]["reason"] in ("not_flat", "no_direction"):
                        continue
                    i = off + t
                    theta, phi = FS.trace_direction(branch, i, cfg.measure.line_fit_window_um)
                    rec = recs[nid]
                    # the painted lines run along x in the plane: heading 0 or pi, tilt 0
                    assert min(abs(theta), abs(abs(theta) - math.pi)) < 1e-9 and abs(phi) < 1e-9, (nid, theta, phi)
                    assert rec["theta_rad"] == theta, (nid, rec)
                    z = float(branch.xyz_um[i, 2])
                    k = int(round(z / dz))
                    view = types.SimpleNamespace(cx_um=float(branch.xyz_um[i, 0]), cy_um=float(branch.xyz_um[i, 1]),
                                                 theta_rad=theta, z_sub_um=z, k_star=k, B_bar=B_BAR, mu_hat_per_um=MU,
                                                 d_hat_um=NAN)
                    shape, frame = CS.patch_frame(view.cx_um, view.cy_um, cal.patch_half_um, res0)
                    kk = CAL.stage_planes(view, cfg, which)        # the whole stage at once: the soma at its blur
                    _sup, st = CAL.support_masks(shape, frame, kk, view, branch, scene, cfg)
                    ok = {int(kq) - k: bool(s["ok"]) for kq, s in zip(kk, st)}
                    q0 = int(np.flatnonzero(kk == k)[0])
                    assert rec["n_pixels_k"] == st[q0]["n_pixels"], (which, nid, rec, st[q0])
                    if not st[q0]["ok"]:
                        assert rec["reason"] == "not_isolated:" + st[q0]["reason"] and rec["n_fit"] == 0, (nid, rec)
                        continue
                    if which == "far":
                        fit = [o for o in ok if abs(o) >= cal.far_min_k_offset]
                        want = (len(fit), sum(ok[o] for o in fit))
                        need = cal.far_min_planes_per_node
                    else:
                        want = near_walk(lambda o: ok[o], cal.offsets_planes, cal.min_planes_per_node)
                        need = cal.min_planes_per_node
                    assert (rec["n_fit"], rec["n_fit_usable"]) == want, (which, nid, rec, want)
                    assert rec["reason"] == ("" if want[1] >= need else "few_planes"), (which, nid, rec)
                    checked += 1
            assert checked >= 10, (which, checked)


def test_convergence():
    raise unittest.SkipTest("the search is a rule on the trace, not a discretisation")


def test_invariants():
    cfg = cfg_with(far_search_spacing_um=5.0, near_search_spacing_um=3.0)
    rng = np.random.default_rng(SEED)
    for which, sp in (("far", 5.0), ("near", 3.0)):
        recs = []
        for stretch in range(3):
            for s in np.arange(0.0, 30.0, 1.0):
                reason = "" if rng.random() < 0.7 else rng.choice(["few_planes", "not_flat", "not_isolated:too_few"])
                recs.append(dict(set=which, node_id=1000 * stretch + int(s), stretch=stretch, s_um=float(s),
                                 reason=str(reason), n_fit_usable=int(rng.integers(3, 25)),
                                 n_pixels_k=int(rng.integers(500, 9000))))
        # a record of the other set, the best of all and far from every other: kept if the set were not checked
        other = dict(recs[0], set="near" if which == "far" else "far", node_id=99999, reason="", s_um=1000.0,
                     n_fit_usable=99, n_pixels_k=10 ** 6)
        kept = FS.choose(recs + [other], cfg, which)
        by = {r["node_id"]: r for r in recs}
        assert kept and 99999 not in kept and all(by[i]["reason"] == "" for i in kept), kept
        for a in kept:
            for b in kept:
                if a != b and by[a]["stretch"] == by[b]["stretch"]:
                    assert abs(by[a]["s_um"] - by[b]["s_um"]) >= sp, (which, a, b)

        def rank(r):
            if which == "far":
                return (-r["n_fit_usable"], -r["n_pixels_k"], r["node_id"])
            return (-r["n_pixels_k"], r["node_id"])
        passing = [r for r in recs if not r["reason"]]
        assert rank(by[kept[0]]) == min(rank(r) for r in passing), (which, by[kept[0]])
        assert [rank(by[i]) for i in kept] == sorted(rank(by[i]) for i in kept), which
        for r in passing:
            if r["node_id"] in kept:
                continue
            blockers = [i for i in kept if by[i]["stretch"] == r["stretch"] and abs(by[i]["s_um"] - r["s_um"]) < sp]
            assert blockers and any(rank(by[i]) < rank(r) for i in blockers), (which, r)


def test_contract():
    cfg = cfg_with()
    with tempfile.TemporaryDirectory() as tmp:
        swc = write_cell(tmp, [far_dendrite(), near_dendrite()])
        for which in ("far", "near"):
            recs = FS.search(swc, cfg, which, None, B_BAR, MU)
            assert recs and all(tuple(r) == FS.RECORD_KEYS for r in recs), [tuple(r) for r in recs[:1]]
            for r in recs:
                assert r["set"] == which and isinstance(r["node_id"], int) and isinstance(r["reason"], str)
                assert isinstance(r["n_fit"], int) and isinstance(r["n_fit_usable"], int) and isinstance(r["k_trace"], int)
                assert all(isinstance(r[k], float) for k in ("allen_diameter_um", "s_um", "x_um", "y_um", "z_um",
                                                             "theta_rad", "phi_deg"))
            path = os.path.join(tmp, "%s.csv" % which)
            table_io.write_rows([dict(r, chosen=False) for r in recs], path)
            back = table_io.read_rows([path])
            assert len(back) == len(recs) and int(back[0]["node_id"]) == recs[0]["node_id"]
        # a file with no record still has its header
        empty = os.path.join(tmp, "empty.csv")
        table_io.write_rows([], empty, fieldnames=list(FS.RECORD_KEYS) + ["chosen"])
        with open(empty) as f:
            assert f.readline().strip() == ",".join(list(FS.RECORD_KEYS) + ["chosen"])
        assert table_io.read_rows([empty]) == []
        # typical_background: the medians over the rows in S only
        rows = [dict(node_id=i, fit_status="converged", flags="", B_bar=b, mu_hat_per_um=m)
                for i, b, m in ((1, 100.0, 0.2), (2, 200.0, 0.4), (3, 300.0, 0.6))]
        rows += [dict(node_id=4, fit_status="failed", flags="", B_bar=9999.0, mu_hat_per_um=99.0),
                 dict(node_id=5, fit_status="converged", flags="crossing", B_bar=9999.0, mu_hat_per_um=99.0),
                 dict(node_id=6, fit_status="converged", flags="", B_bar="", mu_hat_per_um="nan")]
        assert "crossing" in CAL.selection_flags(cfg)
        assert FS.typical_background(rows, cfg) == (200.0, 0.4), FS.typical_background(rows, cfg)
        b, m = FS.typical_background([], cfg)
        assert math.isnan(b) and math.isnan(m)
        _real_command_line(tmp, swc)
        _record_and_reuse(tmp, swc)


def _record_and_reuse(tmp, swc):
    """10.C.R (D-115.6): the record beside each search CSV, the reuse of a search made earlier from the same trace
    under the same settings (whatever B_bar, mu_hat and the transform), and every reason a search is not reused."""
    import hashlib
    import shutil
    import calibrate_kernel as CK
    cfg = cfg_with()
    rows = [dict(node_id=99, fit_status="converged", flags="", phi_rad=0.0, alpha_hat=0.3, allen_radius_um=R_NEAR,
                 B_bar=B_BAR, mu_hat_per_um=MU)]           # B_bar, mu_hat its medians; node 99 is in no SWC
    tr1 = dict(shift_full_px=(0.0, 0.0), flip_y_full_h=None, z0_um=0.0)
    first, second = os.path.join(tmp, "rec_first"), os.path.join(tmp, "rec_second")
    os.makedirs(first)
    logs1 = []
    cand1 = CK.cell_candidates(swc, cfg, tr1, rows, first, 7, log=logs1.append)
    key = CK.search_key(swc, cfg)
    assert key == dict(swc_sha256=FS.swc_digest(swc), config=FS.search_config(cfg)), key
    # (a) a record beside each CSV
    for which in ("near", "far"):
        with open(CK.search_record_name(first, which, 7)) as f:
            rec = json.load(f)
        assert set(rec) == {"set", "specimen", "key", "csv_sha256", "B_bar_gl", "mu_hat_per_um", "transform",
                            "n_records", "n_chosen", "reused_from"}, sorted(rec)
        assert rec["set"] == which and rec["specimen"] == "7" and rec["key"] == key and rec["reused_from"] is None
        with open(CK.search_name(first, which, 7), "rb") as f:
            assert rec["csv_sha256"] == hashlib.sha256(f.read()).hexdigest()
        assert rec["B_bar_gl"] == B_BAR and rec["mu_hat_per_um"] == MU, rec
        assert rec["transform"] == {"shift_full_px": [0.0, 0.0], "flip_y_full_h": None, "z0_um": 0.0}, rec
        back = table_io.read_rows([CK.search_name(first, which, 7)])
        assert rec["n_records"] == len(back) and rec["n_chosen"] == sum(str(r["chosen"]) == "True" for r in back)
        assert rec["n_chosen"] > 0, rec
    # (b) reused with other B_bar, mu_hat and transform: no search, the same candidates, both files copied
    rows2 = [dict(r, B_bar=B_BAR / 2.0, mu_hat_per_um=MU * 2.0) for r in rows]
    tr2 = dict(shift_full_px=(2.3, -1.7), flip_y_full_h=None, z0_um=0.1)

    def boom(*a, **k):
        raise AssertionError("search called on a reuse")
    saved = FS.search
    FS.search = boom
    logs2 = []
    try:
        cand2 = CK.cell_candidates(swc, cfg, tr2, rows2, second, 7, log=logs2.append, reuse_dir=first)
    finally:
        FS.search = saved
    assert cand2 == cand1, (sorted(cand1), sorted(cand2))
    for which in ("near", "far"):
        with open(CK.search_name(first, which, 7), "rb") as f1, open(CK.search_name(second, which, 7), "rb") as f2:
            assert f1.read() == f2.read(), which
        with open(CK.search_record_name(first, which, 7)) as f1, open(CK.search_record_name(second, which, 7)) as f2:
            r1, r2 = json.load(f1), json.load(f2)
        assert r2 == dict(r1, reused_from=os.path.abspath(CK.search_record_name(first, which, 7))), (r1, r2)
        line = [m for m in logs2 if m.startswith("%s search reused from " % which)]
        assert len(line) == 1, logs2
        assert "made with B_bar 200.0 gl, mu_hat 1.000 /um" in line[0], line
        assert "this step's own: B_bar 100.0 gl, mu_hat 2.000 /um" in line[0] and "2.3" in line[0], line
    # reusing a folder from itself copies nothing and keeps the record as it is
    with open(CK.search_record_name(first, "far", 7)) as f:
        before = f.read()
    FS.search = boom
    try:
        assert CK.cell_candidates(swc, cfg, tr1, rows, first, 7, reuse_dir=first) == cand1
    finally:
        FS.search = saved
    with open(CK.search_record_name(first, "far", 7)) as f:
        assert f.read() == before
    # (c) every reason not to reuse; the search then runs (counted) and its record is written afresh
    third = os.path.join(tmp, "rec_third")
    os.makedirs(third)
    ids, rec, why = CK.reuse_search(third, "far", 7, key)
    assert ids is None and rec is None and why.startswith("no search file "), why
    shutil.copy(CK.search_name(first, "far", 7), third)
    ids, rec, why = CK.reuse_search(third, "far", 7, key)
    assert ids is None and why.startswith("no record "), why
    shutil.copy(CK.search_record_name(first, "far", 7), third)
    ids, rec, why = CK.reuse_search(third, "far", 7, key)
    assert why == "" and sorted(ids) == sorted(n for n in cand1 if n in ids_of(swc, cfg, "far")), (why, ids)
    with open(CK.search_name(third, "far", 7), "a") as f:
        f.write("\n")
    ids, rec, why = CK.reuse_search(third, "far", 7, key)
    assert ids is None and why == "the CSV changed since its record", why
    radius = np.array(swc.radius, dtype=float)
    radius[-1] *= 1.01
    other = dataclasses.replace(swc, radius=radius)
    ids, rec, why = CK.reuse_search(first, "far", 7, CK.search_key(other, cfg))
    assert ids is None and why == "another trace", why
    cfg_tau = cfg_with(isolation_depth_gl=cfg.calibration.isolation_depth_gl * 2.0)
    ids, rec, why = CK.reuse_search(first, "far", 7, CK.search_key(swc, cfg_tau))
    assert ids is None and why == "other settings: calibration", why
    calls = []

    def counted(*a, **k):
        calls.append(a[2])
        return saved(*a, **k)
    FS.search = counted
    logs3 = []
    try:
        cand3 = CK.cell_candidates(swc, cfg_tau, tr1, rows, third, 7, log=logs3.append, reuse_dir=first)
    finally:
        FS.search = saved
    assert sorted(calls) == ["far", "near"], calls
    assert sum(m == "%s search not reused (other settings: calibration): searching" % w for m in logs3
               for w in ("near", "far")) == 2, logs3
    with open(CK.search_record_name(third, "far", 7)) as f:
        r3 = json.load(f)
    assert r3["reused_from"] is None and r3["key"] == CK.search_key(swc, cfg_tau), r3
    assert isinstance(cand3, set)
    # (d) the key's parts: the digest follows each of the five arrays and nothing else; the settings' four sections
    d0 = FS.swc_digest(swc)
    for name in ("ids", "types", "parent", "xyz", "radius"):
        a = np.array(getattr(swc, name), copy=True)
        a.flat[-1] = a.flat[-1] + 1
        assert FS.swc_digest(dataclasses.replace(swc, **{name: a})) != d0, name
    assert FS.swc_digest(dataclasses.replace(swc, lines=["# another text"], path="/elsewhere.swc")) == d0
    conf = FS.search_config(cfg)
    assert tuple(sorted(conf)) == tuple(sorted(FS.SEARCH_CONFIG_SECTIONS)) == ("acquisition", "calibration", "measure",
                                                                              "renderer"), sorted(conf)
    json.dumps(conf, allow_nan=False)
    conf_nan = FS.search_config(cfg_with(isolation_depth_gl=float("nan")))
    assert conf_nan["calibration"]["isolation_depth_gl"] == "nan", conf_nan["calibration"]["isolation_depth_gl"]
    json.dumps(conf_nan, allow_nan=False)
    assert FS.search_config(cfg) == json.loads(json.dumps(conf)), "the settings survive a JSON round trip"


def _real_command_line(tmp, swc):
    """calibrate_kernel.py real in process with scan_cell replaced by a recorder: the candidates it is handed."""
    import allen_image_io as aio
    import calibrate_kernel as CK
    base = with_overrides(default_config(), {"calibration.patch_half_um": 16.0})
    far_ids = sorted(ids_of(swc, base, "far"))
    near_ids = sorted(ids_of(swc, base, "near"))
    near_extra = 99                                    # a near-set row of the table that is not in this SWC
    csv_path = os.path.join(tmp, "pilot.csv")
    table_io.write_rows([dict(node_id=n, fit_status="converged", flags="", phi_rad=0.0, alpha_hat=0.3,
                              allen_radius_um=r, B_bar=B_BAR, mu_hat_per_um=MU)
                         for n, r in [(far_ids[0], R_FAR), (far_ids[1], R_FAR), (near_ids[0], R_THIN),
                                      (near_extra, R_NEAR)]], csv_path)
    table = {far_ids[0], far_ids[1], near_ids[0], near_extra}
    transform = dict(shift_full_px=(0.0, 0.0), flip_y_full_h=None, z0_um=0.0)
    chosen = {w: set(FS.choose(FS.search(swc, base, w, transform, B_BAR, MU), base, w)) for w in ("near", "far")}
    assert chosen["near"] and chosen["far"], chosen
    seen = {}

    def recorder(swc_, provider, cfg_, transform=None, regs=None, candidates=None, **kw):
        seen["cand"] = set(candidates)
        return []
    saved = (aio.list_images, aio.plane_table, aio.HttpFetcher, CAL.scan_cell)
    aio.list_images, aio.plane_table = (lambda sid: None), (lambda images: None)
    aio.HttpFetcher = lambda cache_dir=None: None
    CAL.scan_cell = recorder
    try:
        for near_mode, far_mode in (("cell", "cell"), ("table", "table"), ("table", "cell"), ("cell", "table")):
            tag = near_mode + "_" + far_mode
            cj = os.path.join(tmp, "cfg_%s.json" % tag)
            with open(cj, "w") as f:
                json.dump({"calibration": dict(near_candidates=near_mode, far_candidates=far_mode,
                                               patch_half_um=16.0)}, f)
            out_dir = os.path.join(tmp, "real_%s" % tag)
            os.makedirs(out_dir)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                CK.main(["real", "--specimen", "7", "--nodes-csv", csv_path, "--swc", swc.path, "--cache-dir", tmp,
                         "--out", os.path.join(out_dir, "scans.json"), "--config-json", cj, "--no-isolation-figures"])
            text = buf.getvalue()
            want = set(table)
            for which, mode in (("near", near_mode), ("far", far_mode)):
                csv_out = os.path.join(out_dir, "%s_search_7.csv" % which)
                if mode == "cell":
                    want = {n for n in want if n not in ids_of(swc, base, which)} | chosen[which]
                    assert os.path.exists(csv_out), (tag, which)
                    n_range = len(ids_of(swc, base, which))
                    assert "%s search over the cell: %d %s-range nodes" % (which, n_range, which) in text, text
                    assert "; %s set from the whole cell, at least 5 um apart" % which in text, text
                    back = table_io.read_rows([csv_out])
                    assert sorted(int(r["node_id"]) for r in back if str(r["chosen"]) == "True") == sorted(chosen[which])
                    assert all(r["set"] == which for r in back)
                else:
                    assert not os.path.exists(csv_out) and "%s search over" % which not in text, (tag, which, text)
                    assert "; %s set from the pilot table" % which in text, text
            assert seen["cand"] == want, (tag, sorted(seen["cand"]), sorted(want))
            searched = "cell" in (near_mode, far_mode)
            assert ("typical background 200.0 gl and absorption 1.000 /um" in text) == searched, (tag, text)
    finally:
        aio.list_images, aio.plane_table, aio.HttpFetcher, CAL.scan_cell = saved


def test_determinism():
    cfg = cfg_with()
    with tempfile.TemporaryDirectory() as tmp:
        swc = write_cell(tmp, [far_dendrite(), (line(6.0, 50.0, y=6.0), R_NEAR), near_dendrite()])
        for which in ("far", "near"):
            a = FS.search(swc, cfg, which, None, B_BAR, MU)
            b = FS.search(swc, cfg, which, None, B_BAR, MU)
            assert a == b and FS.choose(a, cfg, which) == FS.choose(b, cfg, which), which


def _stub_masks(usable):
    """A support_masks stand-in: the stripe of plane k is usable when k - k_star (the trace's plane) is in usable;
    it records the offsets it is asked for."""
    asked = []

    def stub(shape, frame, kk, node, branch, scene, cfg_, sigma_soma_um=None):
        offs = [int(k) - int(node.k_star) for k in kk]
        asked.append(offs)
        return None, tuple(dict(ok=o in usable, reason="" if o in usable else "side_short", n_pixels=1000)
                           for o in offs)
    return stub, asked


def _search_stubbed(cfg, which, usable, swc_branches):
    real_masks = FS.calibration.support_masks
    stub, asked = _stub_masks(usable)
    FS.calibration.support_masks = stub
    try:
        with tempfile.TemporaryDirectory() as tmp:
            swc = write_cell(tmp, swc_branches, "stub.swc")
            recs = FS.search(swc, cfg, which, None, B_BAR, MU)
    finally:
        FS.calibration.support_masks = real_masks
    return recs, asked


def test_edge_cases():
    cfg = cfg_with()
    with tempfile.TemporaryDirectory() as tmp:
        near_only = write_cell(tmp, [(line(6.0, 30.0), R_NEAR)], "near.swc")
        assert FS.search(near_only, cfg, "far", None, B_BAR, MU) == []
        far_only = write_cell(tmp, [far_dendrite()], "far.swc")
        assert FS.search(far_only, cfg, "near", None, B_BAR, MU) == []
        assert FS.choose([], cfg, "far") == [] and FS.choose([], cfg, "near") == []
        swc = write_cell(tmp, [far_dendrite(), near_dendrite()], "both.swc")
        for which, x in (("far", 44.0), ("near", -44.0)):
            geo = FS.search(swc, cfg, which, None, NAN, NAN)    # no prediction: 10.S's geometric margins r + 3 sigma
            assert geo and all(isinstance(r["reason"], str) for r in geo)
            good = by_id(geo)[id_at(swc, x, 0.0)]
            assert good["reason"] in ("", "few_planes"), (which, good)
            recs = FS.search(swc, cfg, which, None, B_BAR, MU)
            every = FS.choose(recs, cfg_with(**{which + "_search_spacing_um": 0.0}), which)
            assert sorted(every) == sorted(r["node_id"] for r in recs if not r["reason"]), (which, every)
            spaced = FS.choose(recs, cfg, which)
            assert 0 < len(spaced) < len(every), (which, len(spaced), len(every))
    for bad_call in (lambda: FS.search(swc, cfg, "both"), lambda: FS.choose([], cfg, "mid"),
                     lambda: FS.summary_line([], [], cfg, "x")):
        try:
            bad_call()
        except ValueError:
            continue
        raise AssertionError("a bad set name was not refused")
    # the far threshold far_min_planes_per_node: usable at the trace's plane and at m of the far fit's planes
    need = cfg.calibration.far_min_planes_per_node
    fit = [o for o in range(-16, 17) if abs(o) >= cfg.calibration.far_min_k_offset]
    for m, want in ((need - 1, "few_planes"), (need, "")):
        recs, asked = _search_stubbed(cfg, "far", {0} | set(fit[:m]), [far_dendrite()])
        assert recs and all(r["reason"] == want and r["n_fit_usable"] == m and r["n_fit"] == 24 for r in recs), \
            (m, want, recs[:2])
        assert asked[:2] == [[0], fit], asked[:2]
    # the near walk: the trace's plane counts; the pairs +-1, +-2, +-3 in turn until 3 are usable or cannot be
    for usable, n_fit, n_ok, reason, calls in (({0, -1, 1}, 3, 3, "", [[0], [-1, 1]]),
                                               ({0, 1, -2}, 5, 3, "", [[0], [-1, 1], [-2, 2]]),
                                               ({0}, 7, 1, "few_planes", [[0], [-1, 1], [-2, 2], [-3, 3]]),
                                               ({0, 3}, 7, 2, "few_planes", [[0], [-1, 1], [-2, 2], [-3, 3]]),
                                               ({1, -1}, 0, 0, "not_isolated:side_short", [[0]])):
        recs, asked = _search_stubbed(cfg, "near", usable, [near_dendrite()])
        assert recs and all((r["n_fit"], r["n_fit_usable"], r["reason"]) == (n_fit, n_ok, reason) for r in recs), \
            (usable, recs[:1])
        assert asked[:len(calls)] == calls and len(asked) == len(calls) * len(recs), (usable, asked[:6])
    # a walk that can no longer reach the minimum stops early (min_planes_per_node 5, only the trace's plane usable:
    # after +-1 and +-2 at most 1 + 2 < 5)
    recs, asked = _search_stubbed(cfg_with(min_planes_per_node=5), "near", {0}, [near_dendrite()])
    assert recs and all((r["n_fit"], r["n_fit_usable"], r["reason"]) == (5, 1, "few_planes") for r in recs), recs[:1]
    assert asked[:3] == [[0], [-1, 1], [-2, 2]] and len(asked) == 3 * len(recs), asked[:6]
    for bad in ({"calibration.far_candidates": "both"}, {"calibration.near_candidates": "all"},
                {"calibration.far_search_spacing_um": -1.0}, {"calibration.near_search_spacing_um": NAN}):
        try:
            with_overrides(default_config(), bad)
        except ValueError:
            continue
        raise AssertionError("not refused: %r" % (bad,))


# ----------------------------------------------------------------- runner --

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
