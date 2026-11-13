"""Smoke test for the contamination survey -- Block 27 in specs/SPEC.md (plan step C0; D-129, D-130).

Synthetic Branch and Scene objects only (the contract's objects); no other
wave-1 block is imported.

Checks, from the strongest oracle to the weakest
    test_known_answer   isolated branch: P = 0, C = 0, clean [exact]; one flat
                        parallel neighbour at lateral distance rho: P(v_n) =
                        shadow_depths(|rho - v_n|) [1e-12]; D_i and C_i by hand
    test_reference      a loop written out by hand (projection distance,
                        scalar shadow_depth per sample) on a random scene of
                        tilted segments [1e-12]
    test_convergence    skipped: no discretisation
    test_invariants     per_neighbour sums to P [1e-12]; C decreases over 5
                        lateral steps and over 5 defocus steps; the scene
                        rotated and translated with the branch leaves C unchanged
    test_contract       shapes, dtypes, keys; the four classes on four scenes
    test_determinism    pick_examples under one seed; no node twice; a short
                        class reported
    test_edge_cases     ValueError for B <= 0, mu <= 0, an index outside the branch

Run
    python3 tests/smoke/test_smoke_contamination.py
    python3 -m pytest tests/smoke/test_smoke_contamination.py -q

Pure ASCII, for the cluster.
"""
from __future__ import annotations

import importlib.metadata
import platform
import re
import sys
import time
import traceback
import unittest
import warnings
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import contamination as CT  # noqa: E402
from allen_diameter.analysis.calibration_scene import shadow_depth, shadow_depths  # noqa: E402
from allen_diameter.analysis.node_pipeline import Branch  # noqa: E402
from allen_diameter.analysis.plane_iteration import Scene  # noqa: E402
from allen_diameter.analysis.profiles import profile_offsets  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model.kernel import sigma_r  # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy")  # EDIT: packages whose versions matter
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure.

    On failure both arrays are saved to ARTIFACT_DIR/<name>.npz before the
    AssertionError propagates. Justify rtol and atol at the call site
    (float64 roundoff, discretisation-error estimate, statistical spread).
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / f"{safe_name}.npz"
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def observed_order(h, err):
    """Least-squares slope p of log e(h) = log C + p * log h.

    h: step sizes (any order); err: positive errors e(h) at those steps.
    """
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    if np.any(err <= 0.0):
        raise ValueError("observed_order needs strictly positive errors")
    slope, _intercept = np.polyfit(np.log(h), np.log(err), 1)
    return float(slope)


# ---------------------------------------------------------------- checks ---

CFG = default_config()
B, MU, R = 200.0, 1.0, 0.2          # gl, 1/um, um: a node of the pilot's order
DZ = CFG.acquisition.dz_um
Z0 = 10 * DZ                         # the branch lies exactly in plane 10


def straight_branch(theta=0.0, shift=(0.0, 0.0), z=Z0, r=R, half=10.0, step=0.5):
    """A straight flat branch along heading theta through shift, nodes every step um, ids 1..n."""
    t = np.arange(-half, half + 1e-9, step)
    c, s_ = np.cos(theta), np.sin(theta)
    xyz = np.column_stack([shift[0] + t * c, shift[1] + t * s_, np.full(t.size, z)])
    return Branch.from_points(xyz, r, ids=np.arange(1, t.size + 1))


def with_segments(branch, segs, radii, soma=None):
    """The branch's own scene plus extra segments (rows x0 y0 z0 x1 y1 z1), ids 900 + 2m (child), 901 + 2m."""
    own = Scene.from_branch(branch)
    segs = np.asarray(segs, dtype=float).reshape(-1, 6)
    radii = np.asarray(radii, dtype=float).reshape(-1, 2)
    ids = np.array([[900 + 2 * m, 901 + 2 * m] for m in range(segs.shape[0])], dtype=np.int64).reshape(-1, 2)
    return Scene(np.vstack([own.seg_xyz, segs]), np.vstack([own.seg_radius, radii]),
                 np.vstack([own.seg_ids, ids]), np.empty((0, 3)) if soma is None else np.asarray(soma, float))


def parallel(rho, dz=0.0, r=R, half=12.0):
    """A flat neighbour parallel to the x-axis branch at y = rho, depth Z0 + dz."""
    return [[-half, rho, Z0 + dz, half, rho, Z0 + dz]], [[r, r]]


def mid(branch):
    return len(branch.ids) // 2


def test_known_answer():
    br = straight_branch()
    i = mid(br)
    v = profile_offsets(CFG.measure)
    # (a) isolated branch: exactly zero, clean [exact]
    P, per = CT.pollution_profile(br, i, Scene.from_branch(br), v, B, MU, CFG)
    assert P.shape == v.shape and per.shape == (0, v.size), (P.shape, per.shape)
    assert np.all(P == 0.0), "isolated branch: P must be exactly 0"
    row = CT.node_contamination(br, i, Scene.from_branch(br), B, MU, CFG)
    assert row["index"] == 0.0 and row["class"] == "clean" and row["n_neighbours"] == 0, row
    # (b) one flat parallel neighbour at rho (in plane): P(v_n) = shadow_depths(|rho - v_n|).
    # Tolerance 1e-12 of the peak: the only difference is the roundoff of the point-segment distance.
    for rho, dz in ((1.3, 0.0), (-0.7, 0.56), (2.2, 1.4)):
        segs, radii = parallel(rho, dz)
        P, per = CT.pollution_profile(br, i, with_segments(br, segs, radii), v, B, MU, CFG)
        expected = shadow_depths(np.abs(rho - v), R, MU, float(sigma_r(dz, CFG.renderer)), B)
        check_close("known_answer_parallel_%g" % rho, P, expected, rtol=1e-12, atol=1e-12 * expected.max())
    # (c) D_i and C_i by hand, on the last scene
    row = CT.node_contamination(br, i, with_segments(br, segs, radii), B, MU, CFG)
    D = shadow_depth(0.0, R, MU, float(sigma_r(0.0, CFG.renderer)), B)
    w = np.exp(-np.abs(v) / CFG.measure.sgt_lambda_um)
    check_close("own_dip", row["own_dip_gl"], D, rtol=0.0)
    check_close("index_by_hand", row["index"], (np.sum(w * expected) / np.sum(w)) / D, rtol=1e-12)


def _hand_profile(br, i, scene, v):
    """Brute force: the line through node i (heading from the trace's window), every off-branch segment, the
    projection distance written out, scalar shadow_depth per sample at the nearest point's interpolated depth."""
    xyz = br.xyz_um
    win = [j for j in range(len(br.ids)) if abs(br.s_um[j] - br.s_um[i]) <= CFG.measure.line_fit_window_um / 2]
    pts = xyz[win] - xyz[win].mean(axis=0)
    t = np.linalg.svd(pts)[2][0]
    if np.dot(t, xyz[win[-1]] - xyz[win[0]]) < 0:
        t = -t
    th = np.arctan2(t[1], t[0])
    yh = np.array([-np.sin(th), np.cos(th)])
    zk = round(xyz[i, 2] / DZ) * DZ
    ids = set(int(x) for x in br.ids)
    out = []
    for q in range(scene.seg_xyz.shape[0]):
        if int(scene.seg_ids[q, 0]) in ids and int(scene.seg_ids[q, 1]) in ids:
            continue
        a, b = scene.seg_xyz[q, :3], scene.seg_xyz[q, 3:]
        rq = max(scene.seg_radius[q])
        prof = []
        for vn in v:
            p = xyz[i, :2] + vn * yh
            d = b[:2] - a[:2]
            tt = min(1.0, max(0.0, np.dot(p - a[:2], d) / np.dot(d, d)))
            sig = float(sigma_r(a[2] + tt * (b[2] - a[2]) - zk, CFG.renderer))
            prof.append(shadow_depth(np.linalg.norm(p - a[:2] - tt * d), rq, MU, sig, B))
        out.append(prof)
    return np.array(out).reshape(-1, v.size)


def test_reference():
    rng = make_rng()
    # a slightly wiggly, tilted branch and 12 random segments around it, some crossing the line, some deep
    t = np.arange(0, 21)
    xyz = np.column_stack([t * 0.5, 0.05 * np.sin(t), Z0 + 0.03 * t])
    br = Branch.from_points(xyz, 0.25, ids=np.arange(1, 22))
    i = 10
    a = np.column_stack([rng.uniform(1, 9, 12), rng.uniform(-3, 3, 12), Z0 + rng.uniform(-3, 3, 12)])
    b = a + np.column_stack([rng.uniform(-4, 4, 12), rng.uniform(-4, 4, 12), rng.uniform(-2, 2, 12)])
    scene = with_segments(br, np.column_stack([a, b]), rng.uniform(0.1, 0.6, (12, 2)))
    v = profile_offsets(CFG.measure)
    P, per = CT.pollution_profile(br, i, scene, v, B, MU, CFG)
    ref = _hand_profile(br, i, scene, v)
    # 1e-12 of the largest value: the two differ only by the roundoff of the distance and the line's heading
    check_close("reference_per_neighbour", per, ref, rtol=0.0, atol=1e-12 * max(1.0, ref.max()))
    check_close("reference_P", P, ref.sum(axis=0), rtol=0.0, atol=1e-12 * max(1.0, ref.max()))


def test_convergence():
    raise unittest.SkipTest("no discretisation: sums of an existing predictor at the profile's samples")


def test_invariants():
    br = straight_branch()
    i = mid(br)
    v = profile_offsets(CFG.measure)
    # (a) per_neighbour sums to P [1e-12 relative to the peak: summation order only]
    segs = [[-5, 1.0, Z0, 5, 1.5, Z0 + 0.5], [2, -4, Z0 - 1.0, 3, 4, Z0 + 2.0], [-8, -0.6, Z0 + 0.3, 8, -0.6, Z0 + 0.3]]
    scene = with_segments(br, segs, [[0.3, 0.2], [0.5, 0.5], [0.15, 0.15]])
    P, per = CT.pollution_profile(br, i, scene, v, B, MU, CFG)
    check_close("sum_of_neighbours", per.sum(axis=0), P, rtol=0.0, atol=1e-12 * P.max())
    assert np.all(per >= 0.0)
    # (b) C decreases as the neighbour moves away laterally (in focus, 5 steps of 0.5 um from 0.5 um)
    lat = [CT.node_contamination(br, i, with_segments(br, *parallel(rho)), B, MU, CFG)["index"]
           for rho in 0.5 + 0.5 * np.arange(5)]
    assert np.all(np.diff(lat) < 0), "lateral: %r" % (lat,)
    # (c) C decreases as a neighbour directly over the line (rho = 0) moves out of focus (5 planes; amendment 1,
    # D-131: for rho > 0 the blur first carries the shadow toward the line's centre, so no monotony there).
    foc = [CT.node_contamination(br, i, with_segments(br, *parallel(0.0, dz)), B, MU, CFG)["index"]
           for dz in DZ * np.arange(5)]
    assert np.all(np.diff(foc) < 0), "defocus: %r" % (foc,)
    # (d) rotating and translating the branch with its scene leaves C unchanged [1e-10: trig roundoff]
    th, sh = 0.7, np.array([13.3, -4.1])
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    br2 = straight_branch(theta=th, shift=sh)
    segs2 = []
    for sg in segs:
        a2, b2 = rot @ np.array(sg[:2]) + sh, rot @ np.array(sg[3:5]) + sh
        segs2.append([a2[0], a2[1], sg[2], b2[0], b2[1], sg[5]])
    r1 = CT.node_contamination(br, i, scene, B, MU, CFG)
    r2 = CT.node_contamination(br2, i, with_segments(br2, segs2, [[0.3, 0.2], [0.5, 0.5], [0.15, 0.15]]), B, MU, CFG)
    check_close("rotation_index", r2["index"], r1["index"], rtol=1e-10)
    assert r1["class"] == r2["class"], (r1["class"], r2["class"])


def four_scenes():
    """(name, branch, scene): one scene per class, the node at the middle of an x-axis branch in plane 10."""
    br = straight_branch()
    out = [("clean", br, with_segments(br, *parallel(4.0))),                      # far: C ~ 1e-29
           ("diffuse", br, with_segments(br, *parallel(1.0, 1.4))),               # 5 planes deep: not in focus
           ("close_in_focus", br, with_segments(br, *parallel(1.0)))]             # 1 um aside, in focus
    # a crossing segment 1.4 um above the node: footprints meet in projection
    out.append(("overlap", br, with_segments(br, [[0.0, -5.0, Z0 + 1.4, 0.0, 5.0, Z0 + 1.4]], [[0.2, 0.2]])))
    return out


def test_contract():
    v = profile_offsets(CFG.measure)
    keys = {"node_id", "own_dip_gl", "index", "class", "n_neighbours", "worst_child_id", "worst_lateral_um",
            "worst_defocus_um", "worst_share", "soma_lateral_um"}
    for name, br, scene in four_scenes():
        i = mid(br)
        P, per = CT.pollution_profile(br, i, scene, v, B, MU, CFG)
        n_q = scene.seg_xyz.shape[0] - (len(br.ids) - 1)
        assert P.shape == (v.size,) and per.shape == (n_q, v.size), (P.shape, per.shape)
        assert P.dtype == np.float64 and per.dtype == np.float64 and np.all(np.isfinite(per))
        row = CT.node_contamination(br, i, scene, B, MU, CFG)
        assert set(row) == keys, sorted(row)
        assert row["class"] in CT.CLASSES
        assert row["class"] == name, "%s scene classed %s (index %.4g, share %.3g)" % (
            name, row["class"], row["index"], row["worst_share"])
        assert row["node_id"] == int(br.ids[i]) and row["worst_child_id"] == 900, row
    # a close in-focus neighbour (1.9 um, plane 0) that carries under half of sum w P, beside a deep wide one
    # 0.8 um aside: diffuse, not close_in_focus (the "at least half" rule)
    br = straight_branch()
    (s1, r1), (s2, r2) = parallel(1.9), parallel(-0.8, 1.4, r=0.4)
    row = CT.node_contamination(br, mid(br), with_segments(br, s1 + s2, r1 + r2), B, MU, CFG)
    assert row["class"] == "diffuse" and row["worst_child_id"] == 902 and row["worst_share"] > 0.5, row
    # clean first (amendment 1): a lone in-focus neighbour 2 um aside carries all of sum w P but C ~ 0.0067
    row = CT.node_contamination(br, mid(br), with_segments(br, *parallel(2.0)), B, MU, CFG)
    assert row["class"] == "clean" and row["worst_share"] > 0.99 and row["index"] < CFG.cleaning.clean_index_max, row
    # a segment attached to the stretch (one end on the node's own id) cannot make an overlap, even when it
    # crosses the line in focus; the same segment detached is an overlap
    seg = [[0.0, 0.0, Z0, 0.0, 4.0, Z0]]
    sc = with_segments(br, seg, [[0.2, 0.2]])
    detached = CT.node_contamination(br, mid(br), sc, B, MU, CFG)
    sc = Scene(sc.seg_xyz, sc.seg_radius, np.vstack([sc.seg_ids[:-1], [[900, int(br.ids[mid(br)])]]]), sc.soma)
    attached = CT.node_contamination(br, mid(br), sc, B, MU, CFG)
    assert detached["class"] == "overlap", detached
    assert attached["class"] == "close_in_focus" and attached["index"] == detached["index"], attached
    # the worst neighbour's lateral distance and defocus at its point nearest the line's centre: a steep segment
    # from (1.0, -1, Z0 - 2) to (1.0, 1, Z0 + 2) is nearest the centre at its middle, 1.0 um aside, in plane
    sc = with_segments(br, [[1.0, -1.0, Z0 - 2.0, 1.0, 1.0, Z0 + 2.0]], [[0.3, 0.3]])
    row = CT.node_contamination(br, mid(br), sc, B, MU, CFG)
    check_close("worst_lateral", row["worst_lateral_um"], 1.0, rtol=0.0, atol=1e-12)
    check_close("worst_defocus", row["worst_defocus_um"], 0.0, rtol=0.0, atol=1e-12)
    # soma_lateral_um: NaN without a soma; the distance to the nearest disc's edge; 0 inside a disc
    assert np.isnan(CT.node_contamination(br, mid(br), Scene.from_branch(br), B, MU, CFG)["soma_lateral_um"])
    o = br.xyz_um[mid(br)]
    for soma, expect in (([[o[0] + 3.0, o[1] + 4.0, 2.0], [o[0] - 20.0, o[1], 5.0]], 3.0), ([[o[0], o[1] + 1.0, 2.0]], 0.0)):
        own = Scene.from_branch(br)
        sc = Scene(own.seg_xyz, own.seg_radius, own.seg_ids, np.array(soma, dtype=float))
        check_close("soma_lateral", CT.node_contamination(br, mid(br), sc, B, MU, CFG)["soma_lateral_um"], expect,
                    rtol=0.0, atol=1e-12)
    print("  four scenes: " + ", ".join("%s C=%.3g" % (n, CT.node_contamination(b, mid(b), s, B, MU, CFG)["index"])
                                       for n, b, s in four_scenes()))


def _rows(n_per_class, rng):
    rows, nid = [], 1
    for cls, n in n_per_class.items():
        for _ in range(n):
            rows.append({"node_id": nid, "class": cls, "soma_lateral_um": float(rng.uniform(0, 300))})
            nid += 1
    return rows


def test_determinism():
    rows = _rows({"clean": 5, "diffuse": 10, "close_in_focus": 10, "overlap": 1}, make_rng(1))
    counts = CFG.cleaning.example_counts
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = CT.pick_examples(rows, counts, make_rng())
        b = CT.pick_examples(rows, counts, make_rng())
    assert a == b, "not deterministic under one seed"
    assert len(a) == len(set(a)), "a node picked twice"
    # overlap has 1 row for 2 wanted: 19 picked, the shortfall reported
    assert len(a) == sum(counts) - 1, len(a)
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(msgs) == 2 and all("overlap: wanted 2, got 1" in m for m in msgs), msgs
    by_id = {r["node_id"]: r for r in rows}
    got = [by_id[n]["class"] for n in a[:sum(counts[:4]) - 1]]
    assert got.count("clean") == 4 and got.count("diffuse") == 6 and got.count("close_in_focus") == 6, got
    # nearest the soma: the two smallest distances among the rows not yet picked
    free = sorted((r for r in rows if r["node_id"] not in a[:-2]), key=lambda r: r["soma_lateral_um"])
    assert a[-2:] == [free[0]["node_id"], free[1]["node_id"]], (a[-2:], free[:2])
    # node_contamination is deterministic
    name, br, scene = four_scenes()[1]
    r1 = CT.node_contamination(br, mid(br), scene, B, MU, CFG)
    r2 = CT.node_contamination(br, mid(br), scene, B, MU, CFG)
    assert r1 == r2 or all((r1[k] == r2[k]) or (r1[k] != r1[k] and r2[k] != r2[k]) for k in r1)


def test_edge_cases():
    br = straight_branch()
    scene = Scene.from_branch(br)
    v = profile_offsets(CFG.measure)
    bad = [dict(B_gl=0.0), dict(B_gl=-1.0), dict(B_gl=float("nan")), dict(mu_per_um=0.0), dict(mu_per_um=-0.5),
           dict(index=-1), dict(index=len(br.ids)), dict(index=1.5)]
    for kw in bad:
        args = dict(index=mid(br), B_gl=B, mu_per_um=MU)
        args.update(kw)
        for call in (lambda: CT.pollution_profile(br, args["index"], scene, v, args["B_gl"], args["mu_per_um"], CFG),
                     lambda: CT.node_contamination(br, args["index"], scene, args["B_gl"], args["mu_per_um"], CFG)):
            try:
                call()
            except ValueError:
                continue
            raise AssertionError("no ValueError for %r" % (kw,))
    # an empty request returns nothing; an unknown example key is refused
    assert CT.pick_examples([], {"clean": 0}, make_rng()) == []
    try:
        CT.pick_examples([], {"bogus": 1}, make_rng())
    except ValueError:
        pass
    else:
        raise AssertionError("no ValueError for an unknown example key")
    # a neighbour joined to the branch (one end on it) is a neighbour, not own
    sc = with_segments(br, [[0.0, 0.0, Z0, 0.0, 4.0, Z0]], [[0.2, 0.2]])
    sc = Scene(sc.seg_xyz, sc.seg_radius, np.vstack([sc.seg_ids[:-1], [[900, int(br.ids[mid(br)])]]]), sc.soma)
    P, per = CT.pollution_profile(br, mid(br), sc, v, B, MU, CFG)
    assert per.shape == (1, v.size) and P.max() > 0


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"seed {SEED}")
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print(f"== {Path(__file__).name}")
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
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))

    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline}".rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
