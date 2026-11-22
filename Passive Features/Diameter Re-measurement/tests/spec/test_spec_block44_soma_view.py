"""Spec-derived tests of Block 44 (the soma's view in the image stack; D-210, D-211, D-213) and of its entry point
scripts/show_soma.py (SPEC section 6). Tester, 2026-11-22 (incremental run at 97f5104).

Every expected value is computed here from SPEC.md's text by hand, with numpy/scipy, or by brute force over the pixel
grid; nothing calls the code under test to make an expected value. Run from the workstream folder:

    python3 -m pytest -q -p no:cacheprovider tests/spec/test_spec_block44_soma_view.py

Pure ASCII.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.ndimage import gaussian_filter, gaussian_laplace

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter.analysis import soma_view as SV          # noqa: E402
from allen_diameter import config as C                         # noqa: E402

P = 0.1144     # the pixel pitch of section 2.1 (um)
DZ = 0.28      # the plane spacing of section 2.1 (um)


# ------------------------------------------------------------- references ---

def ref_pixel_centres(shape, left, top, p):
    """Block 44 / Block 45: pixel (row, col) of a crop with corner (left, top) centred on ((left+col) p, (top+row) p)."""
    H, W = shape
    return (left + np.arange(W)) * p, (top + np.arange(H)) * p


def ref_segment_mask(shape, left, top, p, p1, p2, r1, r2):
    """Amendment 2 (iii) by brute force over every pixel: centre within r1 + t (r2 - r1) of the segment, t in [0, 1]
    the nearest point's position."""
    x, y = ref_pixel_centres(shape, left, top, p)
    X, Y = np.meshgrid(x, y)
    out = np.zeros(shape, dtype=bool)
    for a, b, ra, rb in zip(np.reshape(p1, (-1, 2)), np.reshape(p2, (-1, 2)), np.ravel(r1), np.ravel(r2)):
        d = b - a
        L2 = float(d @ d)
        t = np.zeros_like(X) if L2 == 0 else np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / L2, 0, 1)
        dist = np.hypot(X - (a[0] + t * d[0]), Y - (a[1] + t * d[1]))
        out |= dist <= ra + t * (rb - ra)
    return out


# ------------------------------------------------------------- geometry ---

def test_soma_geometry_by_hand():
    xyz = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [10.0, 0.0, 0.0], [2.0, 0.0, 4.0]])
    rad = np.array([2.0, 5.0, 9.0, 3.0])
    types = np.array([1, 1, 3, 1])
    c, r = SV.soma_geometry(xyz, rad, types)
    assert_allclose(c, xyz[[0, 1, 3]].mean(axis=0), rtol=0, atol=1e-15)      # the mean of the type-1 nodes
    assert r == 5.0                                                           # the largest of their radii
    with pytest.raises(ValueError):
        SV.soma_geometry(xyz, rad, np.array([3, 3, 3, 4]))                    # no soma node


@pytest.mark.parametrize("cx,cy,half,res0", [(0.0, 0.0, 40.0, P), (12.345, -7.1, 20.0, P), (-3.0, 5.5, 0.3, 0.25),
                                             (100.0 * P, 50.0 * P, 10.0 * P, P)])
def test_crop_box_by_hand(cx, cy, half, res0):
    left, top, w, h = SV.crop_box((cx, cy), half, res0)
    assert (left, top) == (math.floor((cx - half) / res0), math.floor((cy - half) / res0))   # corner rounded down
    assert w == h == math.ceil(2 * half / res0) + 1                                          # side ceil(2 h / p) + 1
    # the crop's pixel centres cover the square [c - half, c + half] on both axes
    x, y = ref_pixel_centres((h, w), left, top, res0)
    assert x[0] <= cx - half + 1e-9 and x[-1] >= cx + half - res0 - 1e-9
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            SV.crop_box((cx, cy), bad, res0)


@pytest.mark.parametrize("z,half,dz", [(0.0, 10.0, DZ), (31.9, 4.5, DZ), (-2.5, 0.5, 0.25), (2.5, 0.75, 0.25),
                                       (7.3, 0.0, 0.1)])
def test_plane_range_brute_force(z, half, dz):
    k_lo, k_hi = SV.plane_range(z, half, dz)
    ks = np.arange(-2000, 2000)
    inside = np.abs(ks * dz - z) <= half + 1e-9 * dz      # tolerance: k dz is not exact in binary; 1e-9 of a plane
    assert (k_lo, k_hi) == (ks[inside].min(), ks[inside].max())
    # dyadic cases, exact in binary: |k dz - z| <= half is closed, so planes exactly at z -+ half are included
    if dz == 0.25 and ((z - half) / dz).is_integer():
        assert k_lo * dz == z - half and k_hi * dz == z + half


def test_plane_range_refusals():
    for args in ((0.0, 1.0, 0.0), (0.0, 1.0, -0.28), (float("nan"), 1.0, DZ), (0.0, -1.0, DZ)):
        with pytest.raises(ValueError):
            SV.plane_range(*args)


def test_disk_mask_by_hand():
    shape, left, top = (37, 41), -20, -15
    c, r = (0.31, -0.42), 1.234
    m = SV.disk_mask(shape, left, top, P, c, r)
    x, y = ref_pixel_centres(shape, left, top, P)
    ref = (x[None, :] - c[0]) ** 2 + (y[:, None] - c[1]) ** 2 <= r * r
    assert m.dtype == bool and m.shape == shape
    assert_array_equal(m, ref)


# ------------------------------------------------------- detail energy ---

def _blob_stack(n=7, size=61, seed=3):
    """A disk of radius 2 um blurred by an SD growing with |k - 3| (the oracle of Block 44's entry)."""
    x = (np.arange(size) - size // 2) * P
    X, Y = np.meshgrid(x, x)
    disk = 200.0 - 120.0 * ((X ** 2 + Y ** 2) <= 2.0 ** 2)
    rng = np.random.default_rng(seed)
    tex = gaussian_filter(rng.normal(0, 5.0, (size, size)), 0.8)   # some structure inside, fixed over planes
    planes = [gaussian_filter(disk + tex, 0.4 + 1.5 * abs(k - 3)) for k in range(n)]
    return np.array(planes), -(size // 2), -(size // 2)


def test_detail_energy_formula_and_invariances():
    blk, left, top = _blob_stack()
    valid = np.ones(blk.shape[0], dtype=bool)
    valid[5] = False
    centre, radius, sig = (0.0, 0.0), 2.5, 1.3
    E = SV.detail_energy(blk, valid, left, top, P, centre, radius, sig)
    disk = (lambda x, y: (x[None, :]) ** 2 + (y[:, None]) ** 2 <= radius ** 2)(*ref_pixel_centres(blk.shape[1:], left, top, P))
    ref = np.array([np.var(gaussian_laplace(b, sig)[disk]) if v else np.nan for b, v in zip(blk, valid)])
    assert np.isnan(E[5])                                        # NaN for a plane not recorded
    assert_allclose(E[valid], ref[valid], rtol=1e-12, atol=0)    # Var over the disk of gaussian_laplace [1e-12 rel]
    # E(2 I + 7) = 4 E(I): the Laplacian is linear and kills constants, the variance is quadratic [1e-12 rel]
    E2 = SV.detail_energy(2.0 * blk + 7.0, valid, left, top, P, centre, radius, sig)
    assert_allclose(E2[valid], 4.0 * E[valid], rtol=1e-12, atol=0)
    # the least blurred plane (k = 3) has the largest energy and it falls off on both sides
    Ev = SV.detail_energy(blk, np.ones(blk.shape[0], bool), left, top, P, centre, radius, sig)
    assert int(np.argmax(Ev)) == 3
    assert np.all(np.diff(Ev[:4]) > 0) and np.all(np.diff(Ev[3:]) < 0)


def test_detail_energy_quadratic_field_and_inputs():
    size = 81
    x = (np.arange(size) - size // 2) * P
    X, Y = np.meshgrid(x, x)
    quad = 50.0 + 3.0 * X ** 2 - 2.0 * X * Y + 5.0 * Y ** 2 + 4.0 * X
    blk = np.array([quad, quad])
    before = blk.copy()
    E = SV.detail_energy(blk, [True, True], -(size // 2), -(size // 2), P, (0.0, 0.0), 2.0, 1.0)
    # the LoG of a quadratic is the constant 2 (3 + 5) p^2 gl/px^2 away from the border, so its variance over a
    # central disk is ~0. Tolerance: scipy truncates its order-2 kernel at 4 sigma, whose sum is then not exactly 0,
    # which leaves an x^2 term of relative size ~1e-3: the variance must be under 1e-4 of the constant squared.
    lap = 2 * (3 + 5) * P ** 2
    assert np.all(E < 1e-4 * lap ** 2)
    assert_array_equal(blk, before)                              # the input is not modified
    with pytest.raises(ValueError):
        SV.detail_energy(blk, [True], 0, 0, P, (0, 0), 1.0, 1.0)   # valid of the wrong shape
    u8 = np.clip(_blob_stack()[0], 0, 255).astype(np.uint8)
    E8 = SV.detail_energy(u8, np.ones(7, bool), -30, -30, P, (0, 0), 2.0, 1.0)
    assert np.all(np.isfinite(E8))                               # uint8 blocks (real data) accepted


# ----------------------------------------------- reference plane, montage ---

def test_reference_plane_choice_and_refusals():
    ks = np.arange(85, 150)
    valid = np.ones(ks.size, bool)
    valid[(ks >= 100) & (ks <= 103)] = False
    z = 117 * DZ
    j, lab = SV.reference_plane(ks, valid, z, DZ)
    assert (int(ks[j]), lab) == (117, "traced depth")
    j, lab = SV.reference_plane(ks, valid, z, DZ, k_chosen=110)
    assert (int(ks[j]), lab) == (110, "chosen by eye")
    with pytest.raises(ValueError):
        SV.reference_plane(ks, valid, z, DZ, k_chosen=84)       # outside the window
    with pytest.raises(ValueError):
        SV.reference_plane(ks, valid, z, DZ, k_chosen=101)      # not recorded
    # the nearest recorded plane skips the unrecorded ones; on a tie the first (dyadic spacing: exact ties)
    j, _ = SV.reference_plane(ks, valid, 101.4 * DZ, DZ)
    assert int(ks[j]) == 99                                      # 99 is 2.4 planes away, 104 is 2.6
    ks2, v2 = np.arange(95, 110), np.ones(15, bool)
    v2[(ks2 >= 100) & (ks2 <= 102)] = False
    j, _ = SV.reference_plane(ks2, v2, 101.0 * 0.25, 0.25)
    assert int(ks2[j]) == 99                                     # 99 and 103 tie at 2 planes: the first
    j, _ = SV.reference_plane(np.array([10, 11, 12]), [True, True, True], 11.5 * 0.25, 0.25)
    assert int([10, 11, 12][j]) == 11                            # 11 and 12 tie: the first


def test_montage_planes_spec_examples():
    ks = np.arange(75, 160)
    valid = np.ones(ks.size, bool)
    idx = SV.montage_planes(ks, valid, 117, 2)
    got = ks[idx]
    assert len(idx) == 43 and 117 in got                         # the entry's example: 43 planes, 117 among them
    assert_array_equal(got, np.arange(75, 160, 2))
    idx = SV.montage_planes(ks, valid, 117, 1, 105, 145)
    assert len(idx) == 41                                        # a range 105..145 step 1: 41
    valid[ks == 119] = False
    got = ks[SV.montage_planes(ks, valid, 117, 2)]
    assert 119 not in got and len(got) == 42                     # only recorded planes
    assert list(SV.montage_planes(ks, valid, 117, 2)) == sorted(SV.montage_planes(ks, valid, 117, 2))
    for bad in ((ks, valid, 117, 0), (ks, valid, 117, 2, 130, 120)):
        with pytest.raises(ValueError):
            SV.montage_planes(*bad)


# ------------------------------------------------------------- the trace ---

def _tree():
    # ids 0..7: 0, 1 soma (type 1); 2 child of 0 (soma-rooted, left out); 3 child of 2; 4 child of 3;
    # 5 child of 1 (soma-rooted); 6 child of 5 far outside the box; 7 child of 6, crossing back into the box
    xyz = np.array([[0, 0, 0], [0.5, 0, 0], [3, 0, 1], [6, 1, 2], [9, 1, 2], [-3, 0, 0], [-40, 0, 0], [-30, 20, 0]],
                   dtype=float)
    rad = np.array([4, 4, 1.0, 0.8, 0.6, 1.2, 0.5, 0.5])
    parent = np.array([-1, 0, 0, 2, 3, 1, 5, 6])
    types = np.array([1, 1, 3, 3, 3, 4, 4, 2])
    return xyz, rad, parent, types


def test_trace_segments_rules():
    xyz, rad, parent, types = _tree()
    box = (-10.0, 10.0, -10.0, 10.0)
    seg = SV.trace_segments(xyz, rad, parent, types, box)
    # one segment per node with a parent, soma-rooted ones (children 2 and 5) left out; 7's segment (6 -> 7) lies at
    # x <= -30: outside the box; 6's segment (5 -> 6) crosses x = -10: kept
    pairs = {(tuple(a), tuple(b)) for a, b in zip(seg["p1"], seg["p2"])}
    expected = {((3.0, 0.0), (6.0, 1.0)), ((6.0, 1.0), (9.0, 1.0)), ((-3.0, 0.0), (-40.0, 0.0))}
    assert pairs == expected
    k = [i for i, (a, b) in enumerate(zip(seg["p1"], seg["p2"])) if tuple(b) == (6.0, 1.0)][0]
    assert (seg["r1"][k], seg["r2"][k], seg["z1"][k], seg["z2"][k], seg["type"][k]) == (1.0, 0.8, 1.0, 2.0, 3)
    for key in ("p1", "p2"):
        assert seg[key].shape == (3, 2)
    for key in ("r1", "r2", "z1", "z2", "type"):
        assert seg[key].shape == (3,)
    # a segment just outside the box by less than its larger radius is kept (bounding box widened by max(r1, r2))
    seg2 = SV.trace_segments(xyz, rad, parent, types, (9.5, 20.0, -10, 10))
    assert ((6.0, 1.0), (9.0, 1.0)) in {(tuple(a), tuple(b)) for a, b in zip(seg2["p1"], seg2["p2"])}
    seg3 = SV.trace_segments(xyz, rad, parent, types, (9.0 + 0.8 + 1e-6, 20.0, -10, 10))
    assert len(seg3["r1"]) == 0
    with pytest.raises(ValueError):
        SV.trace_segments(xyz, rad, np.array([-1, 0, 0, 2, 3, 1, 5, 8]), types, box)     # parent out of range
    with pytest.raises(ValueError):
        SV.trace_segments(xyz, rad[:-1], parent, types, box)                               # arrays disagree


def test_segment_mask_areas_and_brute_force():
    shape, left, top, p = (400, 400), -200, -200, 0.05
    r, L = 1.0, 6.0
    a, b = np.array([[-2.71, 0.13]]), np.array([[-2.71 + L * math.cos(0.3), 0.13 + L * math.sin(0.3)]])
    m = SV.segment_mask(shape, left, top, p, a, b, [r], [r])
    area = m.sum() * p * p
    assert abs(area - (2 * r * L + math.pi * r * r)) <= 0.03 * (2 * r * L + math.pi * r * r)     # [3 %]
    assert_array_equal(m, ref_segment_mask(shape, left, top, p, a, b, [r], [r]))
    # a cone between the areas of its end radii
    mc = SV.segment_mask(shape, left, top, p, a, b, [0.5], [1.5])
    lo = SV.segment_mask(shape, left, top, p, a, b, [0.5], [0.5]).sum()
    hi = SV.segment_mask(shape, left, top, p, a, b, [1.5], [1.5]).sum()
    assert lo < mc.sum() < hi
    assert_array_equal(mc, ref_segment_mask(shape, left, top, p, a, b, [0.5], [1.5]))
    # several random segments, partly outside the crop: equal to the brute force on every pixel
    rng = np.random.default_rng(11)
    p1 = rng.uniform(-14, 14, (6, 2))
    p2 = p1 + rng.uniform(-6, 6, (6, 2))
    r1, r2 = rng.uniform(0.2, 1.5, 6), rng.uniform(0.2, 1.5, 6)
    assert_array_equal(SV.segment_mask(shape, left, top, p, p1, p2, r1, r2),
                       ref_segment_mask(shape, left, top, p, p1, p2, r1, r2))
    empty = SV.segment_mask(shape, left, top, p, np.zeros((0, 2)), np.zeros((0, 2)), [], [])
    assert empty.shape == shape and not empty.any()


def test_segments_at_depth_edges():
    z1, z2 = np.array([1.0, 5.0, 2.0]), np.array([3.0, 4.0, 2.0])
    r1, r2 = np.array([0.5, 0.25, 0.0]), np.array([0.25, 0.5, 0.0])
    # edges: segment 0 spans [1 - 0.5, 3 + 0.5] = [0.5, 3.5]; segment 1 [3.5, 5.5]; segment 2 the point 2.0
    assert_array_equal(SV.segments_at_depth(z1, z2, r1, r2, 0.5), [True, False, False])
    assert_array_equal(SV.segments_at_depth(z1, z2, r1, r2, 3.5), [True, True, False])
    assert_array_equal(SV.segments_at_depth(z1, z2, r1, r2, 2.0), [True, False, True])
    assert_array_equal(SV.segments_at_depth(z1, z2, r1, r2, 5.5), [False, True, False])
    assert_array_equal(SV.segments_at_depth(z1, z2, r1, r2, 5.5 + 1e-12), [False, False, False])


# ------------------------------------------------------- the configuration ---

def test_soma_view_config_defaults_and_signatures():
    cfg = C.default_config()
    q = cfg.soma_view
    assert (q.half_um, q.z_margin_um, q.focus_margin_um, q.focus_sigma_px, q.montage_step_planes, q.montage_cols) \
        == (40.0, 6.0, 1.0, 1.0, 2, 6)                          # amendments 1 and 2
    h = {w: cfg.signature_hash(w) for w in ("estimator", "simulation", "full")}
    cfg2 = C.with_overrides(cfg, {"soma_view.half_um": 12.0, "soma_view.z_margin_um": 1.0,
                                  "soma_view.montage_step_planes": 3})
    assert {w: cfg2.signature_hash(w) for w in h} == h        # outside every signature
    for bad in ({"soma_view.half_um": 0.0}, {"soma_view.montage_step_planes": 0},
                {"soma_view.montage_cols": 2.5}, {"soma_view.focus_sigma_px": float("nan")}):
        with pytest.raises(ValueError):
            C.with_overrides(cfg, bad)


# ------------------------------------------------- the entry point, end to end ---

def _run(out, *extra, specimen="7"):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    cmd = [sys.executable, str(WS / "scripts" / "show_soma.py"), "--specimen", specimen, "--synthetic",
           "--out", str(out)] + [str(e) for e in extra]
    return subprocess.run(cmd, cwd=str(WS), env=env, capture_output=True, text=True, timeout=900)


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("soma_default")
    res = _run(out)
    assert res.returncode == 0, res.stdout + res.stderr
    return out


def test_script_default_outputs(default_run):
    out = default_run
    for name in ("soma_7_focus.png", "soma_7_planes.png", "soma_7.json", "soma_7_block.npz"):
        assert (out / name).exists(), name
    rec = json.load(open(out / "soma_7.json"))
    cfg = C.default_config()
    q, res0, dz = cfg.soma_view, cfg.acquisition.res0_um, cfg.acquisition.dz_um
    for key in ("centre_um", "radius_um", "crop", "dz_um", "ks", "valid", "detail_energy", "k_traced", "k_by_eye",
                "k_range_by_eye", "montage_ks", "half_um", "n_trace_segments", "primary_dendrite_roots_um"):
        assert key in rec, key
    for key in ("k_sharpest", "z_sharpest_minus_soma_um", "focus"):
        assert key not in rec, key                               # removed by amendment 1
    c, r = np.array(rec["centre_um"]), rec["radius_um"]
    crop = rec["crop"]
    assert crop["width"] == crop["height"] == math.ceil(2 * q.half_um / res0) + 1
    assert (crop["left"], crop["top"]) == (math.floor((c[0] - q.half_um) / res0), math.floor((c[1] - q.half_um) / res0))
    ks = np.array(rec["ks"])
    # the window: every plane k with |k dz - c_z| <= r + z_margin, and no other
    allk = np.arange(ks.min() - 5, ks.max() + 6)
    want = allk[np.abs(allk * dz - c[2]) <= r + q.z_margin_um + 1e-9]
    assert_array_equal(ks, want)
    z = np.load(out / "soma_7_block.npz")
    assert set(z.files) == {"block", "ks", "valid", "left", "top", "res0_um"}
    assert z["block"].shape == (ks.size, crop["height"], crop["width"])
    assert_array_equal(z["ks"], ks)
    assert_array_equal(z["valid"], rec["valid"])
    # the detail energy recomputed from the stored crop with numpy/scipy (disk radius r + focus_margin)
    blk, valid = z["block"].astype(float), z["valid"]
    x, y = ref_pixel_centres(blk.shape[1:], int(z["left"]), int(z["top"]), float(z["res0_um"]))
    disk = (x[None, :] - c[0]) ** 2 + (y[:, None] - c[1]) ** 2 <= (r + q.focus_margin_um) ** 2
    E = np.array([np.nan if e is None else e for e in rec["detail_energy"]])
    ref = np.array([np.var(gaussian_laplace(b, q.focus_sigma_px)[disk]) if v else np.nan for b, v in zip(blk, valid)])
    assert_array_equal(np.isnan(E), ~valid)
    assert_allclose(E[valid], ref[valid], rtol=1e-12)
    # default: no plane chosen; the montage spans the window and holds the traced depth's plane
    assert rec["k_by_eye"] is None and rec["k_range_by_eye"] is None
    rec_valid_ks = ks[np.array(rec["valid"])]
    k_tr = rec_valid_ks[np.argmin(np.abs(rec_valid_ks * dz - c[2]))]
    assert rec["k_traced"] == k_tr
    m = np.array(rec["montage_ks"])
    assert k_tr in m
    assert_array_equal(m, rec_valid_ks[(rec_valid_ks - k_tr) % q.montage_step_planes == 0])


def test_script_choices_recorded(tmp_path):
    res = _run(tmp_path, "--half-um", "6", "--soma-k", "2", "--soma-k-range", "-3", "4", "--montage-k", "-4", "5", "1")
    assert res.returncode == 0, res.stdout + res.stderr
    rec = json.load(open(tmp_path / "soma_7.json"))
    assert rec["k_by_eye"] == 2 and rec["k_range_by_eye"] == [-3, 4] and rec["k_shown"] == 2
    assert rec["montage_ks"] == list(range(-4, 6))
    assert rec["half_um"] == 6.0
    assert rec["crop"]["width"] == math.ceil(2 * 6.0 / P) + 1      # --half-um H crops ceil(2 H / p) + 1 px
    assert rec["n_trace_segments"] >= 1


@pytest.mark.parametrize("extra", [("--soma-k", "60"), ("--soma-k", "20")])
def test_script_refuses_plane_outside_or_unrecorded(tmp_path, extra):
    # 60 lies outside the window (-35..35 for the synthetic soma); 20 is in the window but not recorded
    res = _run(tmp_path, "--half-um", "6", *extra)
    assert res.returncode == 2, res.stdout + res.stderr          # an argparse error
    assert not (tmp_path / "soma_7.json").exists()


def test_script_deterministic(default_run, tmp_path):
    res = _run(tmp_path)
    assert res.returncode == 0
    a, b = json.load(open(default_run / "soma_7.json")), json.load(open(tmp_path / "soma_7.json"))
    assert a == b
    za, zb = np.load(default_run / "soma_7_block.npz"), np.load(tmp_path / "soma_7_block.npz")
    for k in za.files:
        assert_array_equal(za[k], zb[k])
