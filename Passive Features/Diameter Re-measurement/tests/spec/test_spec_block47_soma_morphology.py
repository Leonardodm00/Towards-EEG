"""Spec-derived tests, Block 47 (the soma in the morphology; D-215.2, D-216.3, D-222.5) -- tester 2026-11-23.

Oracles (SPEC Block 47 "Mathematics", "Public API", T1-T8, and the readings the orchestrator settled):
  T1 image_to_swc_um inverts the frame map written from the spec (x_img = (x/p + s_x) p, y_img = ((F - y/p) + s_y) p
     when flipped, z_img = z - z0) [1e-9 um]; lengths kept; cell.to_image_um equals that map [1e-12 um].
  T2 line_crossings: a sphere and a line through its centre give -+R/|d|; axis-aligned ellipsoid -+a, -+b, -+c; rotated
     ellipsoid: P + t d on the surface (|Q| <= 1e-12 by an independent Q) and equal to numpy.roots [1e-9]; a miss NaN.
  Outlines: on the in-plane ellipse (a, b, phi) [1e-12]; closed; the section scaled by sqrt(1 - ((z - z0)/c)^2); empty at
     |z - z0| >= c; the side outline (a cos t, c sin t).
  T3 correct_soma: moved first nodes on the surface (|Q| <= 1e-9), on their stem's line (collinear with the old position
     and P2, 1e-9), t* the root of smaller |t| outside, the positive root inside; the soma at the fitted centre (SWC frame)
     with radius r_eq; every other row unchanged bit for bit; radii kept.
  T4 the walk: P2, P3 inside -> P1 on the crossing of P3-P4, P2, P3 removed, P4 re-parented to P1; "branch inside",
     "ends inside", "missed", "on surface".
  T6 several type-1 nodes: the first kept, the others removed, their children re-parented to it.
  T7 the reflection: with flip_y_full_h the corrected SWC equals the unflipped case's with the ellipsoid mirrored [1e-9].
  T8 shapes, dtypes, no mutation, determinism, the ValueErrors.
  Boundary Block 47 -> Block 8 (write_swc): the corrected SWC written and read back holds the edit.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import soma_spec_helpers as H
from allen_diameter.analysis import cell
from allen_diameter.analysis import soma_morphology as SM
from allen_diameter.loading import swc_io

P_UM = 0.1144


def _swc(tmp_path, rows, name="c.swc"):
    p = tmp_path / name
    p.write_text(H.make_swc_text(rows, fmt="%.6f"))
    return swc_io.read_swc(str(p))


# ------------------------------------------------------------------------ T1 --

@pytest.mark.parametrize("flip", [None, 9000.0])
def test_t1_frames(flip):
    rng = np.random.default_rng(1)
    X = rng.uniform(-500, 500, (50, 3))
    shift, z0 = (123.4, -56.7), 12.5
    img = H.ref_to_image(X, P_UM, shift, flip, z0)
    # the forward map of Block 8 (cell.to_image_um) is the spec's formula
    assert_allclose(cell.to_image_um(X, P_UM, shift_full_px=shift, flip_y_full_h=flip, z0_um=z0), img, rtol=0,
                    atol=1e-9)
    back = SM.image_to_swc_um(img, P_UM, shift_full_px=shift, flip_y_full_h=flip, z0_um=z0)
    assert back.shape == (50, 3) and back.dtype == np.float64
    assert_allclose(back, X, rtol=0, atol=1e-9)                    # [1e-9 um] SPEC T1
    d_in = np.linalg.norm(X[1:] - X[:-1], axis=1)
    d_img = np.linalg.norm(img[1:] - img[:-1], axis=1)
    assert_allclose(d_img, d_in, rtol=1e-12)                       # lengths kept (an isometry)


# ------------------------------------------------------------------------ T2 --

def test_t2_sphere_and_axes():
    R = 3.7
    th = np.array([1.0, -2.0, 0.5, R, R, R, 0.3, 0.1])
    d = np.array([0.0, 2.0, 0.0])                                   # |d| = 2: t in units of |d|
    lo, hi = SM.line_crossings(np.array([[1.0, -2.0, 0.5]]), d, th)
    assert_allclose([lo[0], hi[0]], [-R / 2, R / 2], rtol=1e-14)
    a, b, c = 4.0, 3.0, 2.0
    th = np.array([0.0, 0.0, 0.0, a, b, c, 0.0, 0.2])
    P = np.zeros((3, 3))
    D = np.eye(3)
    lo, hi = SM.line_crossings(P, D, th)
    assert_allclose(lo, [-a, -b, -c], rtol=1e-14)
    assert_allclose(hi, [a, b, c], rtol=1e-14)


def test_t2_rotated_ellipsoid_on_surface_and_numpy_roots():
    rng = np.random.default_rng(7)
    th = np.array([10.0, -5.0, 3.0, 5.5, 3.2, 2.4, 0.83, 0.2])
    P = th[:3] + rng.uniform(-8, 8, (400, 3))
    D = rng.normal(size=(400, 3))
    lo, hi = SM.line_crossings(P, D, th)
    rlo, rhi = H.ref_roots(P, D, th)
    hit = np.isfinite(rlo)
    assert hit.sum() > 50 and (~hit).sum() > 20                    # both cases exercised
    assert_array_equal(np.isfinite(lo), hit)                       # a miss is NaN exactly where numpy.roots is complex
    assert np.all(lo[hit] <= hi[hit])
    assert_allclose(lo[hit], rlo[hit], rtol=1e-9, atol=1e-9)
    assert_allclose(hi[hit], rhi[hit], rtol=1e-9, atol=1e-9)
    for t in (lo[hit], hi[hit]):
        q = H.ref_q(P[hit] + t[:, None] * D[hit], th)
        assert np.max(np.abs(q)) <= 1e-12, np.max(np.abs(q))       # SPEC T2: |Q| <= 1e-12


def test_t2_refusals():
    th = np.array([0, 0, 0, 2.0, 2.0, 2.0, 0, 0.1])
    with pytest.raises(ValueError):
        SM.line_crossings(np.zeros((1, 3)), np.zeros(3), th)
    with pytest.raises(ValueError):
        SM.line_crossings(np.zeros((2, 3)), np.ones((3, 3)), th)
    for bad in ([0, 0, 0, 0.0, 2, 2, 0, 0.1], [0, 0, 0, 2, 2, 2, 0, -0.1], [np.nan, 0, 0, 2, 2, 2, 0, 0.1],
                [0, 0, 0, 2, 2, 2, 0]):
        with pytest.raises(ValueError):
            SM.line_crossings(np.zeros((1, 3)), np.ones(3), np.array(bad, dtype=float))


# ------------------------------------------------------------------ outlines --

def test_outlines():
    th = np.array([3.0, 4.0, 1.0, 5.0, 2.5, 2.0, 0.7, 0.1])
    x, y = SM.ellipse_outline(th, n=200)
    assert x.shape == (200,) and y.shape == (200,)
    P = np.column_stack([x, y, np.full(200, th[2])])
    U = H.ref_frame(P, th)
    assert_allclose((U[:, 0] / 5.0) ** 2 + (U[:, 1] / 2.5) ** 2, 1.0, rtol=0, atol=1e-12)
    # closed (the settled reading): the last point is the first to rounding (angle 2 pi: sin(2 pi) = -2.4e-16)
    assert_allclose([x[-1], y[-1]], [x[0], y[0]], rtol=0, atol=1e-12)
    z = th[2] + 1.2
    xs, ys = SM.ellipse_outline(th, z_um=z, n=64)
    s = math.sqrt(1 - (1.2 / 2.0) ** 2)
    U = H.ref_frame(np.column_stack([xs, ys, np.full(64, z)]), th)
    assert_allclose((U[:, 0] / (5.0 * s)) ** 2 + (U[:, 1] / (2.5 * s)) ** 2, 1.0, rtol=0, atol=1e-12)
    for zz in (th[2] + 2.0, th[2] - 2.0, th[2] + 7.0):              # |z - z0| >= c: empty
        e = SM.ellipse_outline(th, z_um=zz)
        assert e[0].shape == (0,) and e[1].shape == (0,)
    u, w = SM.side_outline(th, n=33)
    t = np.linspace(0, 2 * math.pi, 33)
    assert_allclose(u, 5.0 * np.cos(t), atol=1e-14)
    assert_allclose(w, 2.0 * np.sin(t), atol=1e-14)


# --------------------------------------------------------------------- stems --

def _four_stem_cell():
    """One soma node at (50, 60, 10) r 4 and four stems of 3-5 nodes leaving it in four directions, one tilted in z."""
    rows = [(1, 1, 50.0, 60.0, 10.0, 4.0, -1)]
    dirs = [(1, 0, 0), (0, 1, 0), (-0.6, -0.8, 0.0), (0.0, -0.6, 0.8)]
    lens = [3, 4, 5, 3]
    types = [3, 3, 4, 2]
    nid = 2
    for (dx, dy, dz), n, t in zip(dirs, lens, types):
        par = 1
        for j in range(n):
            s = 4.5 + 3.0 * j          # P2 beyond either test ellipsoid: no walk in T3
            rows.append((nid, t, 50 + s * dx, 60 + s * dy, 10 + s * dz, 0.4 + 0.01 * j, par))
            par = nid
            nid += 1
    return rows


def _image_theta(swc_centre_img, a, b, c, phi, mu=0.1, off=(0.3, -0.2, 0.4)):
    return np.array([swc_centre_img[0] + off[0], swc_centre_img[1] + off[1], swc_centre_img[2] + off[2],
                     a, b, c, phi, mu])


@pytest.mark.parametrize("axes", [(6.5, 5.5, 5.0), (3.4, 2.8, 2.5)])
def test_t3_correct_soma(tmp_path, axes):
    rows = _four_stem_cell()
    swc = _swc(tmp_path, rows)
    snap = copy.deepcopy(swc)
    shift, flip, z0 = (100.0, 40.0), None, 3.0
    img = H.ref_to_image(swc.xyz, P_UM, shift, flip, z0)
    th = _image_theta(img[0], *axes, phi=0.4)
    r_eq = 4.321
    e = SM.correct_soma(swc, th, r_eq, P_UM, shift_full_px=shift, flip_y_full_h=flip, z0_um=z0)
    # T8: no mutation, shapes, dtypes
    for name in ("ids", "types", "xyz", "radius", "parent"):
        assert_array_equal(getattr(swc, name), getattr(snap, name))
    assert e.xyz.shape == (len(swc), 3) and e.radius.shape == (len(swc),) and e.parent.shape == (len(swc),)
    assert e.keep.dtype == bool and e.keep.all() and e.soma_row == 0 and e.removed_soma_ids == ()
    # the soma: at the fitted centre in the SWC frame, radius r_eq
    centre_swc = np.array([(th[0] / P_UM - shift[0]) * P_UM, (th[1] / P_UM - shift[1]) * P_UM, th[2] + z0])
    assert_allclose(e.xyz[0], centre_swc, rtol=0, atol=1e-9)
    assert e.radius[0] == r_eq
    assert_array_equal(e.radius[1:], swc.radius[1:])                 # radii kept
    stems = {s["row"]: s for s in e.stems}
    first_rows = [1, 4, 8, 13]
    assert sorted(stems) == first_rows
    pidx = swc.parent_index()
    for r1 in first_rows:
        s = stems[r1]
        assert s["id"] == int(swc.ids[r1]) and s["type"] == int(swc.types[r1])
        P1 = img[r1]
        r2 = int(np.flatnonzero(pidx == r1)[0])
        P2 = img[r2]
        q1 = H.ref_q(P1[None, :], th)[0]
        new_img = H.ref_to_image(e.xyz[r1][None, :], P_UM, shift, flip, z0)[0]
        assert s["flag"] == "moved", s
        assert abs(H.ref_q(new_img[None, :], th)[0]) <= 1e-9          # on the fitted surface
        d = (P2 - P1) / np.linalg.norm(P2 - P1)
        lo, hi = H.ref_roots(P1[None, :], d, th)
        t_exp = (lo[0] if abs(lo[0]) <= abs(hi[0]) else hi[0]) if q1 > 0 else hi[0]
        assert_allclose(new_img, P1 + t_exp * d, rtol=0, atol=1e-9)   # on the stem's line, the right root
        assert_allclose(s["old_um"], swc.xyz[r1], rtol=0, atol=0)
        assert_allclose(s["new_um"], e.xyz[r1], rtol=0, atol=0)
        assert_allclose(s["moved_um"], np.linalg.norm(e.xyz[r1] - swc.xyz[r1]), rtol=1e-12)
        assert s["removed_ids"] == ()
    others = [k for k in range(len(swc)) if k not in [0] + first_rows]
    assert_array_equal(e.xyz[others], swc.xyz[others])                 # bit for bit
    assert_array_equal(e.parent, swc.parent)
    # determinism
    e2 = SM.correct_soma(swc, th, r_eq, P_UM, shift_full_px=shift, flip_y_full_h=flip, z0_um=z0)
    assert_array_equal(e2.xyz, e.xyz) and e2.stems[0]["flag"] == e.stems[0]["flag"]


def _line_cell(tmp_path, pts, extra=()):
    """soma id 1 at the origin (r 3), one stem 2..n along pts (image frame = SWC frame: identity transform)."""
    rows = [(1, 1, 0.0, 0.0, 0.0, 3.0, -1)]
    par = 1
    for j, p in enumerate(pts):
        rows.append((j + 2, 3, p[0], p[1], p[2], 0.5, par))
        par = j + 2
    rows.extend(extra)
    return _swc(tmp_path, rows)


def test_t4_walk(tmp_path):
    # sphere R = 5 at the origin; stem nodes at x = 1, 2.5, 4, 6, 8: P2 (2.5), P3 (4) inside, P4 (6) outside
    swc = _line_cell(tmp_path, [(1, 0, 0), (2.5, 0, 0), (4, 0, 0), (6, 0, 0), (8, 0, 0)])
    th = np.array([0, 0, 0, 5.0, 5.0, 5.0, 0.0, 0.1])
    e = SM.correct_soma(swc, th, 4.0, 1.0)
    s = e.stems[0]
    assert s["flag"] == "moved" and s["removed_ids"] == (3, 4)
    assert_allclose(e.xyz[1], [5.0, 0, 0], rtol=0, atol=1e-12)        # the crossing on the segment P3-P4
    assert_array_equal(e.keep, [True, True, False, False, True, True])
    assert e.parent[4] == 2                                           # P4 re-parented to P1
    # written and read back (Block 47 -> Block 8)
    out = tmp_path / "w.swc"
    swc_io.write_swc(swc, str(out), xyz=e.xyz, radius=e.radius, parent=e.parent, keep=e.keep)
    back = swc_io.read_swc(str(out))
    assert_array_equal(back.ids, [1, 2, 5, 6])
    assert_array_equal(back.parent, [-1, 1, 2, 5])
    assert_allclose(back.xyz[1], [5.0, 0, 0], atol=0.5e-6)


def test_t4_inside_exit_before_p2(tmp_path):
    swc = _line_cell(tmp_path, [(1, 0, 0), (7, 0, 0), (9, 0, 0)])
    th = np.array([0, 0, 0, 5.0, 5.0, 5.0, 0.0, 0.1])
    e = SM.correct_soma(swc, th, 4.0, 1.0)
    assert e.stems[0]["flag"] == "moved" and e.stems[0]["removed_ids"] == ()
    assert_allclose(e.xyz[1], [5.0, 0, 0], atol=1e-12)
    assert e.keep.all()


def test_t4_outside_both_roots_behind(tmp_path):
    # P1 outside, the stem pointing away: both roots negative, the nearer one (smaller |t|) is taken
    swc = _line_cell(tmp_path, [(7, 0, 0), (9, 0, 0)])
    th = np.array([0, 0, 0, 5.0, 5.0, 5.0, 0.0, 0.1])
    e = SM.correct_soma(swc, th, 4.0, 1.0)
    assert e.stems[0]["flag"] == "moved"
    assert_allclose(e.xyz[1], [5.0, 0, 0], atol=1e-12)


@pytest.mark.parametrize("case, flag", [("branch", "branch inside"), ("leaf", "ends inside"),
                                        ("p1_branch", "branch inside"), ("p1_leaf", "ends inside"),
                                        ("missed", "missed"), ("surface", "on surface"),
                                        ("zero_outside", "missed")])
def test_t4_flags(tmp_path, case, flag):
    th = np.array([0, 0, 0, 5.0, 5.0, 5.0, 0.0, 0.1])
    if case == "branch":       # P2 inside with two children, before the exit
        swc = _line_cell(tmp_path, [(1, 0, 0), (2, 0, 0), (8, 0, 0)], extra=[(5, 3, 2.0, 1.0, 0.0, 0.5, 3)])
    elif case == "leaf":       # P2 inside and a leaf
        swc = _line_cell(tmp_path, [(1, 0, 0), (2, 0, 0)])
    elif case == "p1_branch":  # P1 inside with two children
        swc = _line_cell(tmp_path, [(1, 0, 0), (8, 0, 0)], extra=[(4, 3, 1.0, 8.0, 0.0, 0.5, 2)])
    elif case == "p1_leaf":
        swc = _line_cell(tmp_path, [(1, 0, 0)])
    elif case == "missed":     # P1 outside, its line passes the soma by
        swc = _line_cell(tmp_path, [(0, 8, 0), (2, 8, 0), (4, 8, 0)])
    elif case == "surface":    # P1 exactly on the sphere (|Q| <= 1e-12)
        swc = _line_cell(tmp_path, [(0, 5.0, 0), (0, 7.0, 0)])
    else:                      # P1 outside and P2 on P1: no direction
        swc = _line_cell(tmp_path, [(7, 0, 0), (7, 0, 0), (9, 0, 0)])
    e = SM.correct_soma(swc, th, 4.0, 1.0)
    s = e.stems[0]
    assert s["flag"] == flag, (case, s)
    assert_array_equal(e.xyz[1:], swc.xyz[1:])                        # the stem unmoved
    assert e.keep.all() and s["removed_ids"] == () and s["moved_um"] == 0.0


def test_t6_several_soma_nodes(tmp_path):
    rows = [(1, 1, 0.0, 0.0, 0.0, 3.0, -1), (2, 1, 1.0, 0.0, 0.0, 2.0, 1), (3, 1, -1.0, 0.0, 0.0, 2.0, 1),
            (4, 3, 8.0, 0.0, 0.0, 0.5, 2), (5, 3, 10.0, 0.0, 0.0, 0.5, 4), (6, 4, -8.0, 0.0, 0.0, 0.5, 3),
            (7, 4, -10.0, 0.0, 0.0, 0.5, 6), (8, 2, 0.0, 8.0, 0.0, 0.3, 1), (9, 2, 0.0, 10.0, 0.0, 0.3, 8)]
    swc = _swc(tmp_path, rows)
    th = np.array([0.5, 0, 0, 5.0, 5.0, 5.0, 0.0, 0.1])
    e = SM.correct_soma(swc, th, 4.5, 1.0)
    assert e.soma_row == 0 and e.removed_soma_ids == (2, 3)
    assert_array_equal(e.keep, [True, False, False, True, True, True, True, True, True])
    assert e.parent[3] == 1 and e.parent[5] == 1 and e.parent[7] == 1 and e.parent[0] == -1
    assert sorted(s["id"] for s in e.stems) == [4, 6, 8]
    assert all(s["flag"] == "moved" for s in e.stems)
    out = tmp_path / "w.swc"
    swc_io.write_swc(swc, str(out), xyz=e.xyz, radius=e.radius, parent=e.parent, keep=e.keep)
    back = swc_io.read_swc(str(out))
    assert_array_equal(back.ids, [1, 4, 5, 6, 7, 8, 9])
    assert_array_equal(back.parent, [-1, 1, 4, 1, 6, 1, 8])


def test_t7_reflection(tmp_path):
    rows = _four_stem_cell()
    swc = _swc(tmp_path, rows)
    shift, z0, F = (100.0, 40.0), 3.0, 5000.0
    imgA = H.ref_to_image(swc.xyz, P_UM, shift, None, z0)
    imgB = H.ref_to_image(swc.xyz, P_UM, shift, F, z0)
    cst = F * P_UM + 2 * shift[1] * P_UM                              # y_B = cst - y_A for every point
    assert_allclose(imgB[:, 1], cst - imgA[:, 1], atol=1e-9)
    thA = _image_theta(imgA[0], 6.0, 5.0, 4.6, phi=0.4)
    thB = thA.copy()
    thB[1] = cst - thA[1]
    thB[6] = -thA[6]                                                  # the axis (cos phi, sin phi) mirrored
    eA = SM.correct_soma(swc, thA, 4.0, P_UM, shift_full_px=shift, flip_y_full_h=None, z0_um=z0)
    eB = SM.correct_soma(swc, thB, 4.0, P_UM, shift_full_px=shift, flip_y_full_h=F, z0_um=z0)
    assert_allclose(eB.xyz, eA.xyz, rtol=0, atol=1e-9)
    assert [s["flag"] for s in eA.stems] == [s["flag"] for s in eB.stems]


def test_t8_refusals(tmp_path):
    rows = [(1, 3, 0.0, 0.0, 0.0, 1.0, -1), (2, 3, 1.0, 0.0, 0.0, 1.0, 1)]
    swc_nosoma = _swc(tmp_path, rows, "n.swc")
    th = np.array([0, 0, 0, 5.0, 5.0, 5.0, 0.0, 0.1])
    with pytest.raises(ValueError):
        SM.correct_soma(swc_nosoma, th, 4.0, 1.0)
    swc = _line_cell(tmp_path, [(7, 0, 0), (9, 0, 0)])
    for bad in (np.array([0, 0, 0, 5.0, -1.0, 5.0, 0.0, 0.1]), np.array([0, 0, np.inf, 5, 5, 5, 0, 0.1]),
                np.zeros(7)):
        with pytest.raises(ValueError):
            SM.correct_soma(swc, bad, 4.0, 1.0)
    for r in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            SM.correct_soma(swc, th, r, 1.0)
    with pytest.raises(ValueError):
        SM.correct_soma(swc, th, 4.0, 0.0)


def test_stem_rows(tmp_path):
    rows = [(1, 1, 0.0, 0.0, 0.0, 3.0, -1), (2, 1, 1.0, 0.0, 0.0, 2.0, 1), (3, 3, 8.0, 0.0, 0.0, 0.5, 2),
            (4, 2, 0.0, 8.0, 0.0, 0.5, 1), (5, 3, 9.0, 0.0, 0.0, 0.5, 3), (6, 4, 0.0, -8, 0.0, 0.5, 1)]
    swc = _swc(tmp_path, rows)
    assert_array_equal(SM.stem_rows(swc), [2, 3, 5])                 # parent type 1, own type not 1 (axon included)


def test_t3_outside_leaf_moves_radially(tmp_path):
    """P1 outside and a leaf: d from the fitted centre to P1 (SPEC Block 47), so P1 lands on the surface on the ray
    from the centre through it (the root of smaller |t|)."""
    swc = _line_cell(tmp_path, [(6.0, 8.0, 0.0)])
    th = np.array([0.5, -0.5, 0.0, 5.0, 5.0, 5.0, 0.0, 0.1])
    e = SM.correct_soma(swc, th, 4.0, 1.0)
    c = th[:3]
    P1 = np.array([6.0, 8.0, 0.0])
    exp = c + 5.0 * (P1 - c) / np.linalg.norm(P1 - c)
    assert e.stems[0]["flag"] == "moved"
    assert_allclose(e.xyz[1], exp, rtol=0, atol=1e-12)
