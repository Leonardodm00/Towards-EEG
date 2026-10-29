"""Smoke test for Block 10.C in specs/SPEC.md -- the far set's candidates
found over the cell from its trace (D-109, 2026-10-29), before any image is
read: the trace's tilt, the stripe of 10.S at the trace's plane, the stripes
of the far fit's planes, then the greedy choice at a spacing along each
stretch.

Checks
    test_known_answer   painted cells (a soma and straight dendrites, SWC
                        written and read back): a flat far-range dendrite
                        with nothing beside it passes with every far-fit
                        plane usable; the same dendrite with a parallel
                        in-focus near-range dendrite 4 um away fails
                        (not_isolated:side_short or few_planes); a far-range
                        stretch tilted at 30 deg is not_flat with phi 30 deg,
                        one at 12 deg is not (the far bound, 15 deg);
                        near-range nodes are not listed
    test_reference      a record's n_pixels_k, reason and n_fit_usable
                        equal calibration.support_masks called directly on
                        the same view (the trace's centre, heading and plane,
                        the given B_bar and mu, Allen's radius; all 33
                        planes in one call, the soma at the stage's blur); the flat
                        trace's heading and tilt equal the painted line's
                        (1e-9)
    test_convergence    not a discretisation: skipped
    test_invariants     choose() on hand-made records: only passing records,
                        each pair kept on one stretch at least the spacing
                        apart, the most usable planes first, and every
                        passing record left out lies within the spacing of
                        a kept one at least as good (the greedy rule)
    test_contract       the record keys and types (RECORD_KEYS); the CSV
                        through table_io; typical_background reads the
                        medians of the rows in S only; calibrate_kernel.py
                        real hands scan_cell the table's near candidates and
                        the search's chosen ids -- not the table's far ones
                        -- writes far_search_<specimen>.csv and prints the
                        search's lines; with far_candidates "table" it hands
                        it the table's candidates and writes no CSV
    test_determinism    two searches give identical records
    test_edge_cases     a cell without far-range nodes: no record; B_bar and
                        mu NaN: the geometric margins, the search runs;
                        spacing 0: every passing node kept; with the stripe
                        stubbed usable on 5 of the 24 far-fit planes the
                        node is few_planes, on 6 it passes; a validation
                        error for a bad far_candidates or spacing

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_far_search.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import contextlib
import dataclasses
import importlib.metadata
import io
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
from allen_diameter.analysis import cell  # noqa: E402
from allen_diameter.analysis import far_search as FS  # noqa: E402
from allen_diameter.config import default_config, with_overrides  # noqa: E402
from allen_diameter.loading import swc_io, table_io  # noqa: E402
from allen_diameter.model.kernel import sigma_r  # noqa: E402

SEED = 20261029
REPORT_PACKAGES = ("numpy", "scipy")
NAN = float("nan")
B_BAR, MU = 200.0, 1.0           # gl, /um: the painted cells' background and absorption
Z = 10.0                         # um: the depth of the flat dendrites (plane 36 at 0.28 um)
R_FAR, R_NEAR = 0.35, 0.3        # um: Allen 2r 0.7 (far set) and 0.6 (near set)


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


def by_id(records):
    return {int(r["node_id"]): r for r in records}


def id_at(swc, x, y):
    """The node id nearest (x, y) among the dendrite nodes."""
    d = np.hypot(swc.xyz[:, 0] - x, swc.xyz[:, 1] - y)
    d[np.asarray(swc.types) == 1] = np.inf
    return int(swc.ids[int(np.argmin(d))])


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    cfg = cfg_with()
    cal = cfg.calibration
    with tempfile.TemporaryDirectory() as tmp:
        clear = write_cell(tmp, [far_dendrite(), (line(-6.0, -40.0, step=2.0, sign=-1.0), R_NEAR)], "clear.swc")
        recs = by_id(FS.search(clear, cfg, None, B_BAR, MU))
        # only the far-range nodes are listed (the near-range dendrite's are not)
        far_ids = {int(i) for i, r in zip(clear.ids, clear.radius) if CAL.calibration_set(2 * r, cfg) == "far"}
        assert set(recs) == far_ids, (sorted(set(recs) ^ far_ids))
        good = recs[id_at(clear, 44.0, 0.0)]
        assert good["reason"] == "" and good["n_fit_usable"] == good["n_fit"] == 24, good
        assert abs(good["phi_deg"]) < 1e-6 and good["k_trace"] == int(round(Z / cfg.acquisition.dz_um)), good
        # the same dendrite with a parallel in-focus near-range dendrite 4 um away on one side
        crowded = write_cell(tmp, [far_dendrite(), (line(6.0, 50.0, y=4.0), R_NEAR)], "crowded.swc")
        recs2 = by_id(FS.search(crowded, cfg, None, B_BAR, MU))
        for x in (30.0, 36.0, 44.0):
            r = recs2[id_at(crowded, x, 0.0)]
            assert r["reason"] in ("not_isolated:side_short", "few_planes"), (x, r)
        assert not FS.choose(list(recs2.values()), cfg) or all(
            recs2[i]["x_um"] < 20.0 for i in FS.choose(list(recs2.values()), cfg)), FS.choose(list(recs2.values()), cfg)
        # a far-range stretch tilted at 30 deg out of the plane
        tilted = write_cell(tmp, [(line(-6.0, -40.0, step=2.0, sign=-1.0, tilt_deg=30.0), R_FAR)], "tilted.swc")
        recs3 = FS.search(tilted, cfg, None, B_BAR, MU)
        inner = [r for r in recs3 if r["x_um"] < -10.0]
        # 0.01 deg: the SWC holds 4 decimals (1e-4 um over 2 um steps moves the angle by up to ~3e-3 deg)
        assert inner and all(r["reason"] == "not_flat" and abs(r["phi_deg"] - 30.0) < 0.01 for r in inner), inner
        assert cal.node_phi_max_deg_far < 30.0
        # 12 deg: beyond the near set's bound, within the far set's -- not not_flat
        mild = write_cell(tmp, [(line(-6.0, -40.0, step=2.0, sign=-1.0, tilt_deg=12.0), R_FAR)], "mild.swc")
        inner = [r for r in FS.search(mild, cfg, None, B_BAR, MU) if r["x_um"] < -10.0]
        assert cal.node_phi_max_deg < 12.0 < cal.node_phi_max_deg_far
        assert inner and all(r["reason"] != "not_flat" and abs(r["phi_deg"] - 12.0) < 0.01 for r in inner), inner


def test_reference():
    cfg = cfg_with()
    cal, dz, res0 = cfg.calibration, cfg.acquisition.dz_um, cfg.acquisition.res0_um
    with tempfile.TemporaryDirectory() as tmp:
        swc = write_cell(tmp, [far_dendrite(), (line(6.0, 50.0, y=9.0), R_NEAR)])
        recs = by_id(FS.search(swc, cfg, None, B_BAR, MU))
        xyz_img = cell.to_image_um(swc.xyz, res0)
        scene = cell.cell_scene(swc, cfg.acquisition.dendrite_swc_types, res0, None)
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
                # the painted line runs along +x in the plane: heading 0, tilt 0
                assert abs(theta) < 1e-9 and abs(phi) < 1e-9 and rec["theta_rad"] == theta, (nid, theta, phi)
                z = float(branch.xyz_um[i, 2])
                k = int(round(z / dz))
                view = types.SimpleNamespace(cx_um=float(branch.xyz_um[i, 0]), cy_um=float(branch.xyz_um[i, 1]),
                                             theta_rad=theta, z_sub_um=z, k_star=k, B_bar=B_BAR, mu_hat_per_um=MU,
                                             d_hat_um=NAN)
                shape, frame = CS.patch_frame(view.cx_um, view.cy_um, cal.patch_half_um, res0)
                kk = CAL.stage_planes(view, cfg, "far")            # all 33 planes at once: the soma at the stage's blur
                _sup, st = CAL.support_masks(shape, frame, kk, view, branch, scene, cfg)
                q0 = int(np.flatnonzero(kk == k)[0])
                assert rec["n_pixels_k"] == st[q0]["n_pixels"], (nid, rec, st[q0])
                if not st[q0]["ok"]:
                    assert rec["reason"] == "not_isolated:" + st[q0]["reason"] and rec["n_fit"] == 0, (nid, rec)
                if st[q0]["ok"]:
                    fit = np.abs(kk - k) >= cal.far_min_k_offset
                    want = int(sum(1 for q in np.flatnonzero(fit) if st[q]["ok"]))
                    assert rec["n_fit_usable"] == want and rec["n_fit"] == int(fit.sum()), (nid, rec, want)
                    checked += 1
        assert checked >= 10, checked


def test_convergence():
    raise unittest.SkipTest("the search is a rule on the trace, not a discretisation")


def test_invariants():
    cfg = cfg_with(far_search_spacing_um=5.0)
    rng = np.random.default_rng(SEED)
    recs = []
    for stretch in range(3):
        for s in np.arange(0.0, 30.0, 1.0):
            reason = "" if rng.random() < 0.7 else rng.choice(["few_planes", "not_flat", "not_isolated:too_few"])
            recs.append(dict(node_id=1000 * stretch + int(s), stretch=stretch, s_um=float(s), reason=str(reason),
                             n_fit_usable=int(rng.integers(6, 25)), n_pixels_k=int(rng.integers(500, 9000))))
    kept = FS.choose(recs, cfg)
    by = {r["node_id"]: r for r in recs}
    assert kept and all(by[i]["reason"] == "" for i in kept)
    for a in kept:
        for b in kept:
            if a != b and by[a]["stretch"] == by[b]["stretch"]:
                assert abs(by[a]["s_um"] - by[b]["s_um"]) >= 5.0, (a, b)
    order = [(-by[i]["n_fit_usable"], -by[i]["n_pixels_k"], i) for i in kept]
    assert by[kept[0]]["n_fit_usable"] == max(r["n_fit_usable"] for r in recs if not r["reason"])
    passing = [r for r in recs if not r["reason"]]
    for r in passing:
        if r["node_id"] in kept:
            continue
        rk = (-r["n_fit_usable"], -r["n_pixels_k"], r["node_id"])
        blockers = [i for i in kept if by[i]["stretch"] == r["stretch"] and abs(by[i]["s_um"] - r["s_um"]) < 5.0]
        assert blockers and any((-by[i]["n_fit_usable"], -by[i]["n_pixels_k"], i) < rk for i in blockers), r
    assert sorted(order) == sorted(order)          # (the order itself is checked by the greedy rule above)


def test_contract():
    cfg = cfg_with()
    with tempfile.TemporaryDirectory() as tmp:
        swc = write_cell(tmp, [far_dendrite()])
        recs = FS.search(swc, cfg, None, B_BAR, MU)
        assert recs and all(tuple(r) == FS.RECORD_KEYS for r in recs), [tuple(r) for r in recs[:1]]
        for r in recs:
            assert isinstance(r["node_id"], int) and isinstance(r["reason"], str) and isinstance(r["n_fit"], int)
            assert all(isinstance(r[k], float) for k in ("allen_diameter_um", "s_um", "x_um", "y_um", "z_um",
                                                         "theta_rad", "phi_deg"))
        path = os.path.join(tmp, "far.csv")
        table_io.write_rows([dict(r, chosen=False) for r in recs], path)
        back = table_io.read_rows([path])
        assert len(back) == len(recs) and int(back[0]["node_id"]) == recs[0]["node_id"]
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
        _real_command_line(tmp, cfg, swc)


def _real_command_line(tmp, cfg, swc):
    """calibrate_kernel.py real in process with scan_cell replaced by a recorder: the candidates it is handed."""
    import json
    import allen_image_io as aio
    import calibrate_kernel as CK
    far_ids = [int(i) for i, r in zip(swc.ids, swc.radius) if CAL.calibration_set(2 * r, cfg) == "far"]
    near_extra = 99                                    # a near-set row of the table that is not in this SWC
    csv_path = os.path.join(tmp, "pilot.csv")
    table_io.write_rows([dict(node_id=n, fit_status="converged", flags="", phi_rad=0.0, alpha_hat=0.3,
                              allen_radius_um=r, B_bar=B_BAR, mu_hat_per_um=MU)
                         for n, r in [(far_ids[0], R_FAR), (far_ids[1], R_FAR), (near_extra, R_NEAR)]], csv_path)
    swc_path = swc.path
    seen = {}

    def recorder(swc_, provider, cfg_, transform=None, regs=None, candidates=None, **kw):
        seen["cand"] = set(candidates)
        return []
    saved = (aio.list_images, aio.plane_table, aio.HttpFetcher, CAL.scan_cell)
    aio.list_images, aio.plane_table = (lambda sid: None), (lambda images: None)
    aio.HttpFetcher = lambda cache_dir=None: None
    CAL.scan_cell = recorder
    try:
        for mode in ("cell", "table"):
            cj = os.path.join(tmp, "cfg_%s.json" % mode)
            with open(cj, "w") as f:
                json.dump({"calibration": dict(far_candidates=mode, patch_half_um=16.0)}, f)
            out_dir = os.path.join(tmp, "real_%s" % mode)
            os.makedirs(out_dir)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                CK.main(["real", "--specimen", "7", "--nodes-csv", csv_path, "--swc", swc_path, "--cache-dir", tmp,
                         "--out", os.path.join(out_dir, "scans.json"), "--config-json", cj, "--no-isolation-figures"])
            text = buf.getvalue()
            csv_out = os.path.join(out_dir, "far_search_7.csv")
            if mode == "cell":
                cfg_c = with_overrides(default_config(), {"calibration.patch_half_um": 16.0})
                recs = FS.search(swc, cfg_c, dict(shift_full_px=(0.0, 0.0), flip_y_full_h=None, z0_um=0.0),
                                 B_BAR, MU)
                chosen = set(FS.choose(recs, cfg_c))
                assert chosen and seen["cand"] == chosen | {near_extra}, (sorted(seen["cand"]), sorted(chosen))
                assert os.path.exists(csv_out) and "far search over the cell: %d far-range nodes" % len(far_ids) in text, \
                    text
                assert "typical background 200.0 gl and absorption 1.000 /um" in text, text
                assert "; far set from the whole cell, at least 5 um apart" in text, text
                back = table_io.read_rows([csv_out])
                assert sorted(int(r["node_id"]) for r in back if str(r["chosen"]) == "True") == sorted(chosen)
            else:
                assert seen["cand"] == {far_ids[0], far_ids[1], near_extra}, sorted(seen["cand"])
                assert not os.path.exists(csv_out) and "far search" not in text, text
                assert "; far set from the pilot table" in text, text
    finally:
        aio.list_images, aio.plane_table, aio.HttpFetcher, CAL.scan_cell = saved


def test_determinism():
    cfg = cfg_with()
    with tempfile.TemporaryDirectory() as tmp:
        swc = write_cell(tmp, [far_dendrite(), (line(6.0, 50.0, y=6.0), R_NEAR)])
        a = FS.search(swc, cfg, None, B_BAR, MU)
        b = FS.search(swc, cfg, None, B_BAR, MU)
        assert a == b and FS.choose(a, cfg) == FS.choose(b, cfg)


def test_edge_cases():
    cfg = cfg_with()
    with tempfile.TemporaryDirectory() as tmp:
        near_only = write_cell(tmp, [(line(6.0, 30.0), R_NEAR)], "near.swc")
        assert FS.search(near_only, cfg, None, B_BAR, MU) == []
        assert FS.choose([], cfg) == []
        swc = write_cell(tmp, [far_dendrite()], "far.swc")
        geo = FS.search(swc, cfg, None, NAN, NAN)         # no prediction: 10.S's geometric margins r + 3 sigma
        assert geo and all(isinstance(r["reason"], str) for r in geo)
        good = by_id(geo)[id_at(swc, 44.0, 0.0)]
        assert good["reason"] in ("", "few_planes"), good
        recs = FS.search(swc, cfg, None, B_BAR, MU)
        every = FS.choose(recs, cfg_with(far_search_spacing_um=0.0))
        assert sorted(every) == sorted(r["node_id"] for r in recs if not r["reason"]), every
        spaced = FS.choose(recs, cfg)
        assert 0 < len(spaced) < len(every), (len(spaced), len(every))
    # the threshold far_min_planes_per_node: support_masks replaced by a stub whose stripe is usable at the trace's
    # plane and at the first m of the far fit's planes
    need = cfg.calibration.far_min_planes_per_node
    real_masks = FS.calibration.support_masks
    for m, want in ((need - 1, "few_planes"), (need, "")):
        def stub(shape, frame, kk, node, branch, scene, cfg_, sigma_soma_um=None, m=m):
            infos = tuple(dict(ok=(len(kk) == 1 or q < m), reason="" if (len(kk) == 1 or q < m) else "side_short",
                               n_pixels=1000) for q in range(len(kk)))
            return None, infos
        FS.calibration.support_masks = stub
        try:
            with tempfile.TemporaryDirectory() as tmp:
                swc = write_cell(tmp, [far_dendrite()], "stub.swc")
                recs = FS.search(swc, cfg, None, B_BAR, MU)
        finally:
            FS.calibration.support_masks = real_masks
        assert recs and all(r["reason"] == want and r["n_fit_usable"] == m and r["n_fit"] == 24 for r in recs), \
            (m, want, recs[:2])
    for bad in ({"calibration.far_candidates": "both"}, {"calibration.far_search_spacing_um": -1.0},
                {"calibration.far_search_spacing_um": NAN}):
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
