"""Smoke test for the soma in the morphology (analysis/soma_morphology.py, loading/swc_io.edited_lines) -- Block 47
in specs/SPEC.md (D-215.2, D-216.3, D-222.5).

Checks (T1 to T8 are the entry's oracles)
    test_known_answer     T1: image_to_swc_um inverts cell.to_image_um (with and without the reflection, shifts, z0)
                          [1e-9 um] both ways, and keeps distances [1e-9 um]; T2: a sphere and a line through its
                          centre give -+R/|d| [1e-12 relative]; an axis-aligned ellipsoid along each axis gives -+a,
                          -+b, -+c [1e-12 relative]; a line that misses gives NaN; fit_frame of points on the rotated
                          axes [1e-12 um]
    test_reference        T2: on a rotated ellipsoid the points P + t d lie on the surface, |Q| <= 1e-12 (Q written
                          out in matrix form here), and the roots equal scipy.optimize.brentq's roots of that Q along
                          each line [1e-9]; ellipsoid_q and fit_frame against the matrix form [1e-12]
    test_convergence      not a discretisation: skipped
    test_correct_soma     T3: a synthetic SWC (one soma node, four stems of three to five nodes, the axon among them),
                          the fitted ellipsoid larger and smaller than the traced sphere, with and without the
                          reflection: every stem "moved", each moved first node on the surface (|Q| <= 1e-9) and on its
                          stem's line (within 1e-9 um of the line through its old position and P2) at the root the
                          entry names (inside the positive one, outside the nearer; brentq, 1e-9 um); the soma node at
                          the fitted centre in the SWC frame [1e-9 um] with radius r_eq [exact]; every other node
                          unchanged bit for bit
    test_walk_and_flags   T4: a stem whose second and third nodes lie inside -- P1 on the crossing of the third-fourth
                          segment (brentq on the written-out Q, 1e-9 um), the second and third removed, the fourth
                          re-parented to P1; a branch inside: unmoved, "branch inside"; a leaf inside: "ends inside";
                          a line that misses: "missed"; besides: an inside stem whose exit lies before P2 (moved along
                          its line), P1 itself a branch point inside ("branch inside") and a leaf inside ("ends
                          inside"), an outside leaf (moved along the line from the fitted centre), an outside P1 with
                          two children (moved along its first child's line), a node on the surface ("on surface",
                          unmoved bit for bit), and P2 on P1 (outside: "missed"; inside: the walk)
    test_swc_editor       T5: radius-only calls give the bytes of rewrite_radius_lines; an edit (a moved node, new
                          radii, a removed row, a re-parented row) round-trips through read_swc -- the new xyz, radius
                          and parents, the removed id absent, comments, blank lines, separators, CRLF and trailing
                          blanks verbatim; the edited tokens keep max(their decimals, MIN_DECIMALS); unchanged values
                          on 4-decimal tokens give the line byte for byte; a SomaEdit written by write_swc and read
                          back
    test_several_somas    T6: several type-1 nodes -- the first kept at the fitted centre, the others removed, their
                          children (stems among them) re-parented; a first type-1 node whose parent is a later one
                          becomes a root (no self-loop) and the edit is a valid file
    test_reflection       T7: with flip_y_full_h the corrected SWC equals the unflipped case's with the ellipsoid
                          mirrored (centre mapped, phi -> -phi) [1e-9 um], flags, removals and parents equal
    test_invariants       the outlines lie on the surface (the silhouette at z0, a section at z, the side outline)
                          [1e-12], are closed and vanish at |z - z0| >= c; line_crossings scales as 1/|d|, reverses
                          with d, and has roots of opposite signs inside and of one sign outside
    test_contract         T8: shapes, dtypes and types of every output and of SomaEdit; no mutation of the input SWC
                          or arrays
    test_determinism      T8: two runs give identical results
    test_edge_cases       T8: the ValueErrors (no type-1 node, a bad theta_g, r_eq <= 0, a zero direction, bad shapes,
                          the editor's refusals) and empty inputs

Run
    cd "Passive Features/Diameter Re-measurement"
    python3 tests/smoke/test_smoke_soma_morphology.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.
    python -m pytest tests/smoke/test_smoke_soma_morphology.py -q
        The same checks through pytest.

Debug
    A failed check_close() saves both arrays to
        tests/smoke/artifacts/<check name>.npz

Keep this file pure ASCII so it survives transfer to the cluster.
"""
from __future__ import annotations

import copy
import importlib.metadata
import math
import os
import platform
import re
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import brentq

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WS, "src"))
from allen_diameter.analysis import soma_morphology as SM   # noqa: E402
from allen_diameter.analysis.cell import to_image_um       # noqa: E402
from allen_diameter.loading import swc_io                   # noqa: E402

SEED = 20261122
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run

RES0 = 0.1144                       # um per px (the pilot's pitch)
SHIFT = (153.25, -87.5)             # px
FLIP_H = 9876.5                     # px, the full image's height when the y axis is reflected
Z0 = 12.3                           # um
BRENTQ_RTOL = 4.0 * np.finfo(float).eps   # brentq's smallest allowed rtol


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure."""
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


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def perp(v):
    """A unit vector perpendicular to v."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    w = np.cross(v, ref)
    return w / np.linalg.norm(w)


def q_ref(P, th):
    """Q written out in matrix form, independently of the module: U = R (P - centre), R the in-plane rotation."""
    x0, y0, z0, a, b, c, phi = th[:7]
    R = np.array([[math.cos(phi), math.sin(phi), 0.0], [-math.sin(phi), math.cos(phi), 0.0], [0.0, 0.0, 1.0]])
    U = (np.atleast_2d(np.asarray(P, dtype=float)) - np.array([x0, y0, z0])) @ R.T
    return (U[:, 0] / a) ** 2 + (U[:, 1] / b) ** 2 + (U[:, 2] / c) ** 2 - 1.0


def swc_from_rows(rows, header="# synthetic cell for test_smoke_soma_morphology\n"):
    """Write rows (id, type, x, y, z, r, parent) with repr floats (exact round trip) and read them back."""
    text = header + "".join("%d %d %r %r %r %r %d\n" % (i, t, float(x), float(y), float(z), float(r), p)
                            for (i, t, x, y, z, r, p) in rows)
    return swc_from_text(text)


def swc_from_text(text):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cell.swc")
        with open(path, "w", encoding="ascii", newline="") as f:
            f.write(text)
        return swc_io.read_swc(path)


def frame(flip):
    return dict(shift_full_px=SHIFT, flip_y_full_h=flip, z0_um=Z0)


def img(xyz, flip):
    return to_image_um(np.atleast_2d(xyz), RES0, **frame(flip))


def theta_at(centre_swc, axes, phi, flip, mu=0.25):
    """theta_g in the image frame for an ellipsoid centred at centre_swc (SWC frame)."""
    c = img(centre_swc, flip)[0]
    return np.array([c[0], c[1], c[2], axes[0], axes[1], axes[2], phi, mu])


def dist_to_line(x, p, q):
    """Distance (um) from x to the line through p and q."""
    return float(np.linalg.norm(np.cross(x - p, q - p)) / np.linalg.norm(q - p))


def roots_ref(P, d, th):
    """The two roots of the written-out Q along the line P + t d (a hit assumed): Q is a quadratic in t, its minimum
    at t_m from a three-point fit; brentq on each side of it."""
    f = lambda t: float(q_ref(P + t * d, th)[0])   # noqa: E731
    f0, f1, f2 = f(-1.0), f(0.0), f(1.0)
    t_m = -(f2 - f0) / (2.0 * (f2 - 2.0 * f1 + f0))
    assert f(t_m) < 0.0, "the line misses"
    span = 100.0 / np.linalg.norm(d)
    return (brentq(f, t_m - span, t_m, xtol=1e-14, rtol=BRENTQ_RTOL),
            brentq(f, t_m, t_m + span, xtol=1e-14, rtol=BRENTQ_RTOL))


def expected_move(P1, d, th):
    """P1 + t* d with t* the entry's root: the positive one inside, the one of smaller |t| outside."""
    r_lo, r_hi = roots_ref(P1, d, th)
    if q_ref(P1, th)[0] < 0.0:
        return P1 + r_hi * d
    return P1 + (r_lo if abs(r_lo) <= abs(r_hi) else r_hi) * d


def crossing_ref(Pa, Pb, th):
    """The crossing of the segment Pa Pb (Q(Pa) < 0 <= Q(Pb)) with the surface: brentq on the written-out Q."""
    f = lambda s: float(q_ref(Pa + s * (Pb - Pa), th)[0])   # noqa: E731
    s = brentq(f, 0.0, 1.0, xtol=1e-15, rtol=BRENTQ_RTOL, maxiter=200)
    return Pa + s * (Pb - Pa)


# --------------------------------------------------------------- cells ---

T3_SOMA = np.array([512.25, 347.5, 41.75])
T3_DELTA = np.array([0.6, -0.45, 0.5])            # the fitted centre's offset from the traced one (SWC frame, um)


def t3_cell():
    """One soma node (r 5 um) and four stems of 3, 5, 4 and 4 nodes (types 3, 4, 3, 2) at 5.2, 10, 13, 16, 19 um from
    the soma's centre along four directions, each bent off its line by 0.04 (r - 5.2)^2 um. -> (swc, first rows)."""
    stems = [(3, unit([1.0, 0.2, 0.1]), 3), (4, unit([-0.3, 1.0, -0.2]), 5),
             (3, unit([-0.8, -0.5, 0.3]), 4), (2, unit([0.1, -0.9, -0.4]), 4)]
    rr = [5.2, 10.0, 13.0, 16.0, 19.0]
    rows, nid, firsts = [(1, 1, *T3_SOMA, 5.0, -1)], 2, []
    for typ, d, count in stems:
        pv, par = perp(d), 1
        for k in range(count):
            pos = T3_SOMA + rr[k] * d + 0.04 * (rr[k] - 5.2) ** 2 * pv
            if k == 0:
                firsts.append(len(rows))
            rows.append((nid, typ, *pos, 1.0 - 0.1 * k, par))
            par, nid = nid, nid + 1
    return swc_from_rows(rows), firsts


T4_SOMA = np.array([300.0, 200.0, 30.0])
T4_DELTA = np.array([0.3, -0.2, 0.25])
T4_AXES, T4_PHI = (8.0, 7.0, 7.5), 0.4


def t4_cell():
    """The walk's cell (SWC frame): a soma node and one stem per case; -> (swc, {case: rows of its nodes}). With the
    fitted ellipsoid (8, 7, 7.5) um at T4_DELTA from the traced centre, a node within 6.5 um of the traced centre is
    inside and one beyond 8.5 um outside."""
    rows, cases, nid = [(1, 1, *T4_SOMA, 5.0, -1)], {}, [2]

    def add(pos, par_row, typ=3, r=0.8):
        rows.append((nid[0], typ, *pos, r, rows[par_row][0]))
        nid[0] += 1
        return len(rows) - 1

    def chain(name, d, radii, bend=0.05, typ=3):
        pv, par, out = perp(d), 0, []
        for r in radii:
            par = add(T4_SOMA + r * d + bend * (r - radii[0]) ** 2 * pv, par, typ)
            out.append(par)
        cases[name] = out
        return out

    chain("walk", unit([1.0, 0.1, 0.05]), [3.0, 4.5, 6.0, 11.0, 14.0])
    d = unit([-1.0, 0.3, 0.1])
    p1 = add(T4_SOMA + 3.0 * d, 0)
    p2 = add(T4_SOMA + 4.5 * d, p1)
    p3a = add(T4_SOMA + 6.0 * d + 0.5 * perp(d), p2)
    p3b = add(T4_SOMA + 6.0 * d - 0.5 * perp(d), p2)
    p4a = add(T4_SOMA + 12.0 * d + 0.5 * perp(d), p3a)
    cases["branch_p2"] = [p1, p2, p3a, p3b, p4a]
    chain("ends_p3", unit([0.2, 1.0, -0.1]), [3.0, 4.5, 5.5], typ=4)
    d = unit([0.3, -1.0, 0.2])
    p1 = add(T4_SOMA + 9.5 * d, 0)
    p2 = add(T4_SOMA + 9.5 * d + 2.0 * perp(d), p1)
    cases["missed"] = [p1, p2]
    chain("inside_short", unit([-0.2, -1.0, 0.3]), [5.0, 11.0, 14.0])
    d = unit([0.1, 0.2, 1.0])
    p1 = add(T4_SOMA + 4.0 * d, 0)
    p2a = add(T4_SOMA + 11.0 * d + perp(d), p1)
    p2b = add(T4_SOMA + 11.0 * d - perp(d), p1)
    cases["branch_p1"] = [p1, p2a, p2b]
    cases["leaf_p1_inside"] = [add(T4_SOMA + 4.0 * unit([0.5, 0.5, -1.0]), 0)]
    cases["leaf_p1_outside"] = [add(T4_SOMA + 9.0 * unit([-0.6, 0.6, -0.5]), 0, typ=2)]
    d = unit([-0.7, -0.7, 0.1])
    p1 = add(T4_SOMA + 9.0 * d, 0)
    p2a = add(T4_SOMA + 12.0 * d, p1)
    p2b = add(T4_SOMA + 12.0 * d + 2.0 * perp(d), p1)
    cases["outside_two_children"] = [p1, p2a, p2b]
    d = unit([0.9, -0.3, -0.3])
    p1 = add(T4_SOMA + 9.0 * d, 0)
    p2 = add(T4_SOMA + 9.0 * d, p1)                      # P2 on P1
    p3 = add(T4_SOMA + 12.0 * d, p2)
    cases["outside_p2_on_p1"] = [p1, p2, p3]
    d = unit([-0.4, 0.2, -0.9])
    p1 = add(T4_SOMA + 3.0 * d, 0)
    p2 = add(T4_SOMA + 3.0 * d, p1)                      # P2 on P1
    p3 = add(T4_SOMA + 12.0 * d, p2)
    cases["inside_p2_on_p1"] = [p1, p2, p3]
    return swc_from_rows(rows), cases


def _stem_by_row(edit):
    return {s["row"]: s for s in edit.stems}


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    rng = make_rng()
    # T1: the inverse, both ways, with and without the reflection, shifts and z0 [1e-9 um: about 400 um / 0.1144 um
    # per px = 3.5e3 px, so a few ulp of 3.5e3 px times 0.1144 um, about 1e-12 um]
    swc_pts = rng.uniform([200.0, 150.0, 10.0], [800.0, 700.0, 80.0], size=(50, 3))
    for flip in (None, FLIP_H):
        for shift, z0 in (((0.0, 0.0), 0.0), (SHIFT, Z0), ((-12.5, 40.25), -3.5)):
            fr = dict(shift_full_px=shift, flip_y_full_h=flip, z0_um=z0)
            im = to_image_um(swc_pts, RES0, **fr)
            back = SM.image_to_swc_um(im, RES0, **fr)
            check_close("T1 swc->img->swc flip=%s" % flip, back, swc_pts, rtol=0, atol=1e-9)
            again = to_image_um(SM.image_to_swc_um(im, RES0, **fr), RES0, **fr)
            check_close("T1 img->swc->img flip=%s" % flip, again, im, rtol=0, atol=1e-9)
            dsw = np.linalg.norm(swc_pts[:, None, :] - swc_pts[None, :, :], axis=2)
            back_img = SM.image_to_swc_um(im + np.array([0.3, -0.2, 0.1]), RES0, **fr)
            dim = np.linalg.norm(back_img[:, None, :] - back_img[None, :, :], axis=2)
            check_close("T1 distances kept flip=%s" % flip, dim, dsw, rtol=0, atol=1e-9)
    # T2: a sphere and a line through its centre: t = -+R/|d| [1e-12 relative: a few roundings]
    R, centre = 6.5, np.array([12.5, -7.25, 3.0])
    th = np.array([*centre, R, R, R, 0.7, 0.3])
    for d in ([1.0, 0.0, 0.0], [0.3, -2.0, 0.7], [0.0, 0.0, -0.01], [5.0, 5.0, 5.0]):
        t_lo, t_hi = SM.line_crossings(centre[None, :], d, th)
        k = R / np.linalg.norm(d)
        check_close("T2 sphere %s" % (d,), np.r_[t_lo, t_hi], [-k, k], rtol=1e-12)
    # an axis-aligned ellipsoid along each axis: -+a, -+b, -+c
    a, b, c = 6.0, 4.0, 3.0
    th = np.array([*centre, a, b, c, 0.0, 0.3])
    t_lo, t_hi = SM.line_crossings(np.tile(centre, (3, 1)), np.eye(3), th)
    check_close("T2 axes t_lo", t_lo, [-a, -b, -c], rtol=1e-12)
    check_close("T2 axes t_hi", t_hi, [a, b, c], rtol=1e-12)
    # a line that misses: NaN (x = centre + 7 > a = 6, along y)
    t_lo, t_hi = SM.line_crossings((centre + [7.0, 0.0, 0.0])[None, :], [0.0, 1.0, 0.0], th)
    assert np.isnan(t_lo[0]) and np.isnan(t_hi[0]), (t_lo, t_hi)
    # fit_frame: points on the rotated axes have (u, v, w) = (r, 0, 0), (0, r, 0), (0, 0, r)
    phi = 0.83
    th = np.array([*centre, a, b, c, phi, 0.3])
    pts = centre + np.array([[2.0 * math.cos(phi), 2.0 * math.sin(phi), 0.0],
                             [-3.0 * math.sin(phi), 3.0 * math.cos(phi), 0.0], [0.0, 0.0, -1.5]])
    check_close("fit_frame axes", SM.fit_frame(pts, th), np.diag([2.0, 3.0, -1.5]), rtol=0, atol=1e-12)
    # Q: the centre -1, the axes' ends 0
    ends = centre + np.array([[a * math.cos(phi), a * math.sin(phi), 0.0], [0.0, 0.0, c]])
    check_close("Q centre", SM.ellipsoid_q(centre[None, :], th), [-1.0], rtol=1e-15)
    check_close("Q axis ends", SM.ellipsoid_q(ends, th), [0.0, 0.0], rtol=0, atol=1e-15)


def test_reference():
    rng = make_rng(SEED + 1)
    phi = 0.83
    worst_q, worst_root, n_hit = 0.0, 0.0, 0
    for centre, label in ((np.array([12.5, -7.25, 3.0]), "moderate"), (np.array([4123.4, 2987.6, 55.5]), "pilot")):
        th = np.array([*centre, 6.0, 4.0, 3.0, phi, 0.3])
        P = centre + rng.uniform(-7.0, 7.0, size=(400, 3))
        D = rng.normal(size=(400, 3)) * rng.uniform(0.2, 3.0, size=(400, 1))
        t_lo, t_hi = SM.line_crossings(P, D, th)
        hit = np.isfinite(t_lo)
        assert np.all(np.isfinite(t_hi) == hit)
        assert hit.sum() > 60 and (~hit).sum() > 60, (hit.sum(), (~hit).sum())   # both kinds of line drawn
        assert np.all(t_lo[hit] <= t_hi[hit])
        # T2: the points P + t d on the surface, |Q| <= 1e-12 (Q in matrix form)
        for t in (t_lo, t_hi):
            q = q_ref(P[hit] + t[hit, None] * D[hit], th)
            worst_q = max(worst_q, float(np.max(np.abs(q))))
        # the roots equal brentq's roots of the written-out Q along the line: Q along a line is a quadratic in t, its
        # minimum at t_m found by a three-point fit; brentq on each side brackets one root [1e-9 um of |t d|]
        for i in np.flatnonzero(hit)[:60]:
            f = lambda t: float(q_ref(P[i] + t * D[i], th)[0])   # noqa: E731
            f0, f1, f2 = f(-1.0), f(0.0), f(1.0)
            t_m = -(f2 - f0) / (2.0 * (f2 - 2.0 * f1 + f0))
            span = 100.0 / np.linalg.norm(D[i])
            r_lo = brentq(f, t_m - span, t_m, xtol=1e-14, rtol=BRENTQ_RTOL)
            r_hi = brentq(f, t_m, t_m + span, xtol=1e-14, rtol=BRENTQ_RTOL)
            nd = np.linalg.norm(D[i])
            worst_root = max(worst_root, abs(r_lo - t_lo[i]) * nd, abs(r_hi - t_hi[i]) * nd)
            n_hit += 1
        # a miss has no real root: the written-out Q is positive along the whole line (its minimum > 0)
        for i in np.flatnonzero(~hit)[:30]:
            f = lambda t: float(q_ref(P[i] + t * D[i], th)[0])   # noqa: E731
            f0, f1, f2 = f(-1.0), f(0.0), f(1.0)
            t_m = -(f2 - f0) / (2.0 * (f2 - 2.0 * f1 + f0))
            assert f(t_m) > 0.0, (label, i, f(t_m))
        # ellipsoid_q and fit_frame against the matrix form [1e-12: roundings of coordinates up to 4e3 um, Q ~ 10]
        check_close("Q matrix form %s" % label, SM.ellipsoid_q(P, th), q_ref(P, th), rtol=1e-12, atol=1e-12)
        Rm = np.array([[math.cos(phi), math.sin(phi), 0.0], [-math.sin(phi), math.cos(phi), 0.0], [0.0, 0.0, 1.0]])
        check_close("fit_frame matrix form %s" % label, SM.fit_frame(P, th), (P - centre) @ Rm.T,
                    rtol=0, atol=1e-11)
    print("    T2: max |Q| at the roots %.2e (bound 1e-12); max |t - brentq| |d| %.2e um over %d lines"
          % (worst_q, worst_root, n_hit))
    assert worst_q <= 1e-12, worst_q
    assert worst_root <= 1e-9, worst_root


def test_convergence():
    raise unittest.SkipTest("not a discretisation: closed-form crossings and an exact tree edit")


def _check_t3(flip, axes, label):
    swc, firsts = t3_cell()
    th = theta_at(T3_SOMA + T3_DELTA, axes, 0.6, flip)
    r_eq = 5.83
    edit = SM.correct_soma(swc, th, r_eq, RES0, **frame(flip))
    by_row = _stem_by_row(edit)
    assert sorted(by_row) == firsts, (sorted(by_row), firsts)
    worst_q, worst_line = 0.0, 0.0
    for r1 in firsts:
        s = by_row[r1]
        assert s["flag"] == "moved" and s["removed_ids"] == (), (label, s)
        new_img = img(edit.xyz[r1], flip)
        q = abs(float(q_ref(new_img, th)[0]))
        line = dist_to_line(edit.xyz[r1], swc.xyz[r1], swc.xyz[r1 + 1])     # the stem's P2 is the next row
        worst_q, worst_line = max(worst_q, q), max(worst_line, line)
        assert q <= 1e-9, (label, r1, q)
        assert line <= 1e-9, (label, r1, line)
        # the root the entry names (inside: the positive one; outside: the nearer), by brentq on the written-out Q
        P1i, P2i = img(swc.xyz[r1], flip)[0], img(swc.xyz[r1 + 1], flip)[0]
        exp = expected_move(P1i, (P2i - P1i) / np.linalg.norm(P2i - P1i), th)
        assert np.max(np.abs(new_img[0] - exp)) <= 1e-9, (label, r1, new_img[0] - exp)
        assert_array_equal(s["old_um"], swc.xyz[r1])
        assert_array_equal(s["new_um"], edit.xyz[r1])
        assert abs(s["moved_um"] - np.linalg.norm(edit.xyz[r1] - swc.xyz[r1])) <= 1e-12
    # the soma node at the fitted centre (SWC frame) with radius r_eq
    assert edit.soma_row == 0
    check_close("T3 soma centre %s" % label, img(edit.xyz[0], flip)[0], th[:3], rtol=0, atol=1e-9)
    assert edit.radius[0] == r_eq
    # every other node unchanged bit for bit
    others = np.setdiff1d(np.arange(len(swc)), [0] + firsts)
    assert_array_equal(edit.xyz[others], swc.xyz[others])
    assert_array_equal(edit.radius[1:], swc.radius[1:])
    assert_array_equal(edit.parent, swc.parent)
    assert edit.keep.all() and edit.removed_soma_ids == ()
    return worst_q, worst_line, [round(by_row[r]["moved_um"], 3) for r in firsts]


def test_correct_soma():
    out = []
    for flip in (None, FLIP_H):
        for axes, label in (((7.6, 6.9, 7.1), "larger"), ((4.0, 3.5, 3.1), "smaller")):
            wq, wl, moved = _check_t3(flip, axes, "%s flip=%s" % (label, flip))
            out.append("%s/%s |Q| %.1e line %.1e moved %s" % (label, "flip" if flip else "noflip", wq, wl, moved))
    print("    T3: " + "; ".join(out))


def test_walk_and_flags():
    swc, cases = t4_cell()
    flip = None
    th = theta_at(T4_SOMA + T4_DELTA, T4_AXES, T4_PHI, flip)
    edit = SM.correct_soma(swc, th, 6.1, RES0, **frame(flip))
    by_row = _stem_by_row(edit)
    Pimg = img(swc.xyz, flip)
    ids = swc.ids
    expect = {"walk": "moved", "branch_p2": "branch inside", "ends_p3": "ends inside", "missed": "missed",
              "inside_short": "moved", "branch_p1": "branch inside", "leaf_p1_inside": "ends inside",
              "leaf_p1_outside": "moved", "outside_two_children": "moved", "outside_p2_on_p1": "missed",
              "inside_p2_on_p1": "moved"}
    assert sorted(by_row) == sorted(v[0] for v in cases.values()), (sorted(by_row), cases)
    for name, rows in cases.items():
        s = by_row[rows[0]]
        assert s["flag"] == expect[name], (name, s["flag"])
    # the cases' geometry is as designed: inside and outside by the written-out Q
    q = q_ref(Pimg, th)
    assert np.all(q[cases["walk"][:3]] < 0) and q[cases["walk"][3]] > 0, q[cases["walk"]]
    assert q[cases["missed"][0]] > 0 and q[cases["leaf_p1_outside"][0]] > 0
    # T4 walk: P1 on the crossing of the third-fourth segment, P2 and P3 removed, P4 re-parented to P1
    w = cases["walk"]
    exp = crossing_ref(Pimg[w[2]], Pimg[w[3]], th)
    err_walk = float(np.max(np.abs(img(edit.xyz[w[0]], flip)[0] - exp)))
    assert err_walk <= 1e-9, err_walk
    s = by_row[w[0]]
    assert s["removed_ids"] == (int(ids[w[1]]), int(ids[w[2]])), s["removed_ids"]
    assert not edit.keep[w[1]] and not edit.keep[w[2]] and edit.keep[w[0]] and edit.keep[w[3]] and edit.keep[w[4]]
    assert edit.parent[w[3]] == ids[w[0]] and edit.parent[w[4]] == ids[w[3]]
    assert edit.radius[w[0]] == swc.radius[w[0]]                       # radii kept
    assert abs(s["moved_um"] - np.linalg.norm(edit.xyz[w[0]] - swc.xyz[w[0]])) <= 1e-12
    # unmoved stems: bit for bit, nothing removed
    for name in ("branch_p2", "ends_p3", "missed", "branch_p1", "leaf_p1_inside", "outside_p2_on_p1"):
        r = cases[name]
        assert_array_equal(edit.xyz[r], swc.xyz[r])
        assert edit.keep[r].all() and by_row[r[0]]["moved_um"] == 0.0 and by_row[r[0]]["removed_ids"] == ()
    # inside, exit before P2: moved along its own line onto the surface
    r = cases["inside_short"]
    new = edit.xyz[r[0]]
    assert dist_to_line(new, swc.xyz[r[0]], swc.xyz[r[1]]) <= 1e-9
    assert abs(float(q_ref(img(new, flip), th)[0])) <= 1e-9
    exp = expected_move(Pimg[r[0]], unit(Pimg[r[1]] - Pimg[r[0]]), th)
    assert np.max(np.abs(img(new, flip)[0] - exp)) <= 1e-9
    # the outside leaf: along the line from the fitted centre through P1, onto the surface, between them
    r = cases["leaf_p1_outside"]
    new_img = img(edit.xyz[r[0]], flip)[0]
    assert dist_to_line(new_img, th[:3], Pimg[r[0]]) <= 1e-9
    assert abs(float(q_ref(new_img, th)[0])) <= 1e-9
    assert np.linalg.norm(new_img - th[:3]) < np.linalg.norm(Pimg[r[0]] - th[:3])
    assert np.max(np.abs(new_img - expected_move(Pimg[r[0]], unit(Pimg[r[0]] - th[:3]), th))) <= 1e-9
    # outside with two children: along the first child's line
    r = cases["outside_two_children"]
    new = edit.xyz[r[0]]
    assert dist_to_line(new, swc.xyz[r[0]], swc.xyz[r[1]]) <= 1e-9
    assert dist_to_line(new, swc.xyz[r[0]], swc.xyz[r[2]]) > 1e-3
    exp = expected_move(Pimg[r[0]], unit(Pimg[r[1]] - Pimg[r[0]]), th)
    assert np.max(np.abs(img(new, flip)[0] - exp)) <= 1e-9
    assert abs(float(q_ref(img(new, flip), th)[0])) <= 1e-9
    # inside with P2 on P1: the walk -- P1 on the crossing of P2 P3, P2 removed, P3 re-parented to P1
    r = cases["inside_p2_on_p1"]
    exp = crossing_ref(Pimg[r[1]], Pimg[r[2]], th)
    assert np.max(np.abs(img(edit.xyz[r[0]], flip)[0] - exp)) <= 1e-9
    assert by_row[r[0]]["removed_ids"] == (int(ids[r[1]]),) and not edit.keep[r[1]]
    assert edit.parent[r[2]] == ids[r[0]]
    # the whole edit is a valid file (every kept parent kept)
    swc_io.edited_lines(swc, xyz=edit.xyz, radius=edit.radius, parent=edit.parent, keep=edit.keep)
    # a node on the surface: unmoved bit for bit, "on surface" (the ellipsoid's a chosen through P1)
    S = np.array([100.0, 80.0, 20.0])
    p1 = S + 5.3 * unit([0.6, 0.7, 0.4])
    s_swc = swc_from_rows([(1, 1, *S, 5.0, -1), (2, 3, *p1, 1.0, 1), (3, 3, *(S + 9.0 * unit([0.6, 0.7, 0.4])), 0.9, 2)])
    c_img, p1_img = img(S + [0.2, -0.1, 0.15], None)[0], img(p1, None)[0]
    phi, b, c = 0.3, 6.0, 5.5
    U = (p1_img - c_img) @ np.array([[math.cos(phi), math.sin(phi), 0.0], [-math.sin(phi), math.cos(phi), 0.0],
                                     [0.0, 0.0, 1.0]]).T
    a = abs(U[0]) / math.sqrt(1.0 - (U[1] / b) ** 2 - (U[2] / c) ** 2)
    th_s = np.array([*c_img, a, b, c, phi, 0.2])
    q1 = float(SM.ellipsoid_q(p1_img[None, :], th_s)[0])
    assert abs(q1) <= SM.Q_SURFACE_TOL, q1
    e2 = SM.correct_soma(s_swc, th_s, 5.5, RES0, **frame(None))
    assert e2.stems[0]["flag"] == "on surface", e2.stems[0]
    assert_array_equal(e2.xyz[1:], s_swc.xyz[1:])
    print("    T4: walk crossing err %.1e um; flags %s; on-surface Q %.1e"
          % (err_walk, {n: by_row[r[0]]["flag"] for n, r in cases.items()}, q1))


SAMPLE = (
    "# generated by test_smoke_soma_morphology\n"
    "# id type x y z radius parent\n"
    "1 1 0.0 0.0 0.0 5.0 -1\n"
    "\n"
    "2 3 1.0000 0.0000 0.0000 0.2500 1\n"
    "3\t3\t2.0\t0.0\t0.0\t0.25\t2\r\n"
    "4  4   2.0  1.5  0.0  0.5  3 \n"
    "5 4 2.0 3.5 0.0 0.1 4\n"
    "6 2 -1.0 0.0 0.0 0.3 1\n"
    "7 3 -1.12345 0.5 0.25 0.123456 6\n"
)


def _decimals(tok):
    return len(tok.split(".")[1]) if ("." in tok and "e" not in tok.lower()) else 0


def test_swc_editor():
    s = swc_from_text(SAMPLE)
    n = len(s)
    new_r = s.radius * 1.5
    # T5: radius-only calls give the bytes of rewrite_radius_lines (edited_lines, and write_swc's unchanged path)
    assert swc_io.edited_lines(s, radius=new_r) == swc_io.rewrite_radius_lines(s, new_r)
    assert swc_io.edited_lines(s) == s.lines
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "r.swc")
        swc_io.write_swc(s, f, radius=new_r)
        assert open(f, "rb").read() == "".join(swc_io.rewrite_radius_lines(s, new_r)).encode("ascii")
        swc_io.write_swc(s, f)
        assert open(f, "rb").read() == SAMPLE.encode("ascii")
        # an edit: node 2 moved, radii x 1.5, node 5 removed, node 7 re-parented 6 -> 2
        xyz = s.xyz.copy()
        xyz[1] = (1.23456789, -0.5, 0.75)
        xyz[6] = (-1.123456789, 0.5, 0.25)              # a 5-decimal token: written with 5 decimals
        par = s.parent.copy()
        par[6] = 2
        keep = np.ones(n, dtype=bool)
        keep[4] = False
        swc_io.write_swc(s, f, radius=new_r, xyz=xyz, parent=par, keep=keep)
        out_lines = open(f, "rb").read().decode("ascii").splitlines(keepends=True)
        t = swc_io.read_swc(f)
    assert_array_equal(t.ids, [1, 2, 3, 4, 6, 7])
    assert_array_equal(t.types, s.types[keep])
    assert_array_equal(t.parent, par[keep])
    assert 5 not in set(t.ids.tolist())
    check_close("T5 xyz", t.xyz, xyz[keep], rtol=0, atol=0.5e-4)      # tokens of >= 4 decimals round to 0.5e-4
    check_close("T5 radius", t.radius, new_r[keep], rtol=0, atol=0.5e-4)
    assert_array_equal(t.xyz[[0, 2, 3, 4]], s.xyz[[0, 2, 3, 5]])  # nodes 1, 3, 4, 6: reformatted, values exact
    # every non-data line verbatim and in order; the removed line gone
    src = SAMPLE.splitlines(keepends=True)
    removed_line = s.data_rows[4]
    expect_lines = [ln for k, ln in enumerate(src) if k != removed_line]
    assert len(out_lines) == len(expect_lines), (len(out_lines), len(expect_lines))
    worst_dec = 99
    for k, (la, lb) in enumerate(zip(expect_lines, out_lines)):
        if not la.strip() or la.startswith("#"):
            assert la == lb, "non-data line %d changed" % k
            continue
        ta, tb = swc_io._split_keep(la), swc_io._split_keep(lb)
        assert len(ta) == len(tb), (k, ta, tb)
        for j in range(len(ta)):
            assert ta[j][0] == tb[j][0], "separator %d of line %d changed" % (j, k)   # incl. the line ending
        assert ta[0][1] == tb[0][1] and ta[1][1] == tb[1][1]
        for j in (2, 3, 4, 5):
            want = max(_decimals(ta[j][1]), swc_io.MIN_DECIMALS)
            assert _decimals(tb[j][1]) == want, (k, j, ta[j][1], tb[j][1])
            worst_dec = min(worst_dec, _decimals(tb[j][1]))
        assert re.fullmatch(r"-?\d+", tb[6][1]), tb[6][1]
    assert worst_dec >= swc_io.MIN_DECIMALS
    assert out_lines[5].endswith("\r\n") and out_lines[6].endswith(" \n"), (out_lines[5], out_lines[6])
    assert "-1.12346" in out_lines[-1], out_lines[-1]          # 5 decimals kept, rounded
    # unchanged values on 4-decimal tokens give the line byte for byte
    same = swc_io.edited_lines(s, xyz=s.xyz, radius=s.radius, parent=s.parent, keep=np.ones(n, dtype=bool))
    assert same[4] == src[4], (same[4], src[4])
    assert [ln for ln in same if not ln.strip() or ln.startswith("#")] == \
        [ln for ln in src if not ln.strip() or ln.startswith("#")]
    # a SomaEdit written by write_swc and read back
    swc, _cases = t4_cell()
    th = theta_at(T4_SOMA + T4_DELTA, T4_AXES, T4_PHI, None)
    e = SM.correct_soma(swc, th, 6.1, RES0, **frame(None))
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "soma.swc")
        swc_io.write_swc(swc, f, radius=e.radius, xyz=e.xyz, parent=e.parent, keep=e.keep)
        back = swc_io.read_swc(f)
    assert_array_equal(back.ids, swc.ids[e.keep])
    assert_array_equal(back.types, swc.types[e.keep])
    assert_array_equal(back.parent, e.parent[e.keep])
    check_close("T5 SomaEdit xyz", back.xyz, e.xyz[e.keep], rtol=0, atol=0.5e-4)
    check_close("T5 SomaEdit radius", back.radius, e.radius[e.keep], rtol=0, atol=0.5e-4)
    print("    T5: %d of %d rows kept after the SomaEdit; edited tokens >= %d decimals"
          % (int(e.keep.sum()), len(swc), worst_dec))


def test_several_somas():
    S = np.array([250.0, 180.0, 25.0])
    rows = [(1, 1, *S, 5.0, -1), (2, 1, *(S + [4.0, 0.0, 0.0]), 1.0, 1), (3, 1, *(S + [-4.0, 0.0, 0.0]), 1.0, 1),
            (4, 3, *(S + [5.5, 0.5, 0.0]), 0.9, 2), (5, 3, *(S + [9.0, 1.0, 0.0]), 0.8, 4),
            (6, 4, *(S + [-5.5, 0.3, 0.2]), 1.1, 3), (7, 4, *(S + [-9.5, 0.6, 0.4]), 1.0, 6),
            (8, 2, *(S + [0.2, -5.5, 0.0]), 0.5, 1), (9, 2, *(S + [0.4, -9.0, 0.0]), 0.4, 8)]
    swc = swc_from_rows(rows)
    th = theta_at(S + [0.2, 0.1, -0.1], (6.0, 5.5, 5.0), 0.2, None)
    e = SM.correct_soma(swc, th, 5.6, RES0, **frame(None))
    assert e.soma_row == 0 and e.removed_soma_ids == (2, 3), (e.soma_row, e.removed_soma_ids)
    assert_array_equal(e.keep, [True, False, False, True, True, True, True, True, True])
    assert_array_equal(e.parent, [-1, 1, 1, 1, 4, 1, 6, 1, 8])
    assert [s["id"] for s in e.stems] == [4, 6, 8], [s["id"] for s in e.stems]
    assert_array_equal(SM.stem_rows(swc), [3, 5, 7])
    check_close("T6 soma centre", img(e.xyz[0], None)[0], th[:3], rtol=0, atol=1e-9)
    assert e.radius[0] == 5.6
    lines = swc_io.edited_lines(swc, xyz=e.xyz, radius=e.radius, parent=e.parent, keep=e.keep)
    assert sum(1 for ln in lines if not ln.startswith("#")) == 7
    # the first type-1 node (row order) is a child of a later one: it becomes the root, no self-loop
    rows2 = [(5, 1, *S, 5.0, 1), (6, 3, *(S + [5.5, 0.0, 0.0]), 0.9, 5), (1, 1, *(S + [0.5, 0.0, 0.0]), 4.0, -1),
             (7, 4, *(S + [-5.5, 0.0, 0.0]), 0.9, 1)]
    swc2 = swc_from_rows(rows2)
    e2 = SM.correct_soma(swc2, th, 5.6, RES0, **frame(None))
    assert e2.soma_row == 0 and e2.removed_soma_ids == (1,)
    assert_array_equal(e2.parent, [-1, 5, -1, 5])
    assert_array_equal(e2.keep, [True, True, False, True])
    lines2 = swc_io.edited_lines(swc2, xyz=e2.xyz, radius=e2.radius, parent=e2.parent, keep=e2.keep)
    t = swc_from_text("".join(lines2))
    assert_array_equal(t.ids, [5, 6, 7])
    assert_array_equal(t.parent, [-1, 5, 5])


def test_reflection():
    worst = 0.0
    for build, delta, axes, phi in ((t3_cell, T3_DELTA, (7.6, 6.9, 7.1), 0.6), (t3_cell, T3_DELTA, (4.0, 3.5, 3.1), 0.6),
                                     (t4_cell, T4_DELTA, T4_AXES, T4_PHI)):
        swc, _ = build()
        soma = swc.xyz[0]
        th_u = theta_at(soma + delta, axes, phi, None)
        th_f = theta_at(soma + delta, axes, -phi, FLIP_H)       # the same ellipsoid seen in the reflected frame
        e_u = SM.correct_soma(swc, th_u, 5.9, RES0, **frame(None))
        e_f = SM.correct_soma(swc, th_f, 5.9, RES0, **frame(FLIP_H))
        check_close("T7 xyz", e_f.xyz, e_u.xyz, rtol=0, atol=1e-9)
        worst = max(worst, float(np.max(np.abs(e_f.xyz - e_u.xyz))))
        assert_array_equal(e_f.keep, e_u.keep)
        assert_array_equal(e_f.parent, e_u.parent)
        assert_array_equal(e_f.radius, e_u.radius)
        assert [s["flag"] for s in e_f.stems] == [s["flag"] for s in e_u.stems]
        assert [s["removed_ids"] for s in e_f.stems] == [s["removed_ids"] for s in e_u.stems]
    print("    T7: max |xyz(flip) - xyz(no flip, mirrored ellipsoid)| %.1e um" % worst)


def test_invariants():
    centre = np.array([12.5, -7.25, 3.0])
    a, b, c, phi = 6.0, 4.0, 3.0, 0.83
    th = np.array([*centre, a, b, c, phi, 0.3])
    # the silhouette: the points (x, y, z0) on the surface; closed (last = first)
    x, y = SM.ellipse_outline(th)
    assert x.shape == (256,) and y.shape == (256,)
    pts = np.column_stack([x, y, np.full(x.size, centre[2])])
    check_close("silhouette on the surface", q_ref(pts, th), np.zeros(x.size), rtol=0, atol=1e-12)
    assert abs(x[0] - x[-1]) <= 1e-12 and abs(y[0] - y[-1]) <= 1e-12
    # a section at z: the points (x, y, z) on the surface
    for dz in (-2.5, -1.0, 0.0, 1.7, 2.99):
        x, y = SM.ellipse_outline(th, z_um=centre[2] + dz, n=101)
        pts = np.column_stack([x, y, np.full(x.size, centre[2] + dz)])
        check_close("section dz=%g" % dz, q_ref(pts, th), np.zeros(101), rtol=0, atol=1e-12)
    for dz in (c, -c, 3.5, -10.0):
        x, y = SM.ellipse_outline(th, z_um=centre[2] + dz)
        assert x.shape == (0,) and y.shape == (0,), (dz, x.shape)
    # the side outline: (u/a)^2 + (w/c)^2 = 1
    u, w = SM.side_outline(th, n=64)
    check_close("side outline", (u / a) ** 2 + (w / c) ** 2, np.ones(64), rtol=0, atol=1e-12)
    # line_crossings: t scales as 1/|d| and reverses with d
    rng = make_rng(SEED + 2)
    P = centre + rng.uniform(-8.0, 8.0, size=(200, 3))
    D = rng.normal(size=(200, 3))
    t_lo, t_hi = SM.line_crossings(P, D, th)
    t_lo3, t_hi3 = SM.line_crossings(P, 3.0 * D, th)
    hit = np.isfinite(t_lo)
    check_close("scale lo", t_lo3[hit], t_lo[hit] / 3.0, rtol=1e-12, atol=1e-13)
    check_close("scale hi", t_hi3[hit], t_hi[hit] / 3.0, rtol=1e-12, atol=1e-13)
    r_lo, r_hi = SM.line_crossings(P, -D, th)
    check_close("reverse", np.c_[r_lo[hit], r_hi[hit]], np.c_[-t_hi[hit], -t_lo[hit]], rtol=1e-12, atol=1e-13)
    assert np.array_equal(np.isfinite(r_lo), hit)
    # inside: roots of opposite signs (never a miss); outside: one sign
    q = q_ref(P, th)
    inside = q < 0
    assert inside.any() and (~inside).any()
    assert np.all(hit[inside]) and np.all(t_lo[inside] < 0) and np.all(t_hi[inside] > 0)
    out = (~inside) & hit
    assert np.all(np.sign(t_lo[out]) == np.sign(t_hi[out]))


def test_contract():
    swc, cases = t4_cell()
    snap = copy.deepcopy(swc)
    th = list(theta_at(T4_SOMA + T4_DELTA, T4_AXES, T4_PHI, FLIP_H))
    th_copy = list(th)
    e = SM.correct_soma(swc, th, 6.1, RES0, **frame(FLIP_H))
    # no mutation of the input
    for name in ("ids", "types", "xyz", "radius", "parent"):
        assert_array_equal(getattr(swc, name), getattr(snap, name))
    assert swc.lines == snap.lines and swc.data_rows == snap.data_rows and th == th_copy
    n = len(swc)
    assert isinstance(e, SM.SomaEdit)
    assert e.xyz.shape == (n, 3) and e.xyz.dtype == np.float64
    assert e.radius.shape == (n,) and e.radius.dtype == np.float64
    assert e.parent.shape == (n,) and e.parent.dtype.kind == "i"
    assert e.keep.shape == (n,) and e.keep.dtype == np.bool_
    assert isinstance(e.soma_row, int) and isinstance(e.stems, tuple) and isinstance(e.removed_soma_ids, tuple)
    assert not np.shares_memory(e.xyz, swc.xyz) and not np.shares_memory(e.radius, swc.radius)
    keys = {"row", "id", "type", "flag", "old_um", "new_um", "moved_um", "removed_ids"}
    for s in e.stems:
        assert set(s) == keys, set(s)
        assert isinstance(s["row"], int) and isinstance(s["id"], int) and isinstance(s["type"], int)
        assert s["flag"] in SM.FLAGS
        assert s["old_um"].shape == (3,) and s["new_um"].shape == (3,) and s["old_um"].dtype == np.float64
        assert isinstance(s["moved_um"], float) and isinstance(s["removed_ids"], tuple)
    rows = SM.stem_rows(swc)
    assert rows.dtype.kind == "i" and np.all(np.diff(rows) > 0)
    assert_array_equal(rows, sorted(v[0] for v in cases.values()))
    pts = np.random.default_rng(SEED).normal(size=(7, 3))
    th_a = np.asarray(th)
    assert SM.image_to_swc_um(pts, RES0).shape == (7, 3) and SM.image_to_swc_um(pts, RES0).dtype == np.float64
    assert SM.fit_frame(pts, th_a).shape == (7, 3)
    assert SM.ellipsoid_q(pts, th_a).shape == (7,)
    t_lo, t_hi = SM.line_crossings(pts, [1.0, 0.0, 0.0], th_a)
    assert t_lo.shape == (7,) and t_hi.shape == (7,) and t_lo.dtype == np.float64
    x, y = SM.ellipse_outline(th_a, n=50)
    assert x.shape == (50,) and y.shape == (50,)
    u, w = SM.side_outline(th_a, n=50)
    assert u.shape == (50,) and w.shape == (50,)
    # a (3,) direction equals the same direction broadcast to (n, 3)
    t2 = SM.line_crossings(pts, np.tile([1.0, 0.0, 0.0], (7, 1)), th_a)
    assert_array_equal(np.r_[t_lo, t_hi], np.r_[t2[0], t2[1]])
    # the editor's inputs are not modified
    xyz, par, keep = e.xyz.copy(), e.parent.copy(), e.keep.copy()
    swc_io.edited_lines(swc, xyz=e.xyz, radius=e.radius, parent=e.parent, keep=e.keep)
    assert_array_equal(e.xyz, xyz)
    assert_array_equal(e.parent, par)
    assert_array_equal(e.keep, keep)


def test_determinism():
    swc, _ = t4_cell()
    th = theta_at(T4_SOMA + T4_DELTA, T4_AXES, T4_PHI, None)
    e1 = SM.correct_soma(swc, th, 6.1, RES0, **frame(None))
    e2 = SM.correct_soma(swc, th, 6.1, RES0, **frame(None))
    for name in ("xyz", "radius", "parent", "keep"):
        assert_array_equal(getattr(e1, name), getattr(e2, name))
    assert e1.soma_row == e2.soma_row and e1.removed_soma_ids == e2.removed_soma_ids
    for s1, s2 in zip(e1.stems, e2.stems):
        assert set(s1) == set(s2)
        for k in s1:
            if isinstance(s1[k], np.ndarray):
                assert_array_equal(s1[k], s2[k])
            else:
                assert s1[k] == s2[k], (k, s1[k], s2[k])


def _raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("no ValueError from %s(%r, %r)" % (getattr(fn, "__name__", fn), args, kwargs))


def test_edge_cases():
    swc, _ = t3_cell()
    th = theta_at(T3_SOMA + T3_DELTA, (7.6, 6.9, 7.1), 0.6, None)
    # correct_soma: no type-1 node, a bad theta_g, r_eq not finite and > 0, res0 <= 0, a bad alignment
    no_soma = swc_from_rows([(1, 3, 0.0, 0.0, 0.0, 1.0, -1), (2, 3, 1.0, 0.0, 0.0, 1.0, 1)])
    _raises(SM.correct_soma, no_soma, th, 5.0, RES0)
    for bad in (th[:7], np.r_[th, 0.0], np.r_[np.nan, th[1:]], np.r_[th[:3], 0.0, th[4:]],
                np.r_[th[:4], -1.0, th[5:]], np.r_[th[:5], np.inf, th[6:]], np.r_[th[:6], np.nan, th[7]],
                np.r_[th[:7], -0.1], np.r_[th[:7], np.nan], "abc"):
        _raises(SM.correct_soma, swc, bad, 5.0, RES0)
        _raises(SM.ellipsoid_q, np.zeros((1, 3)), bad)
    for r_eq in (0.0, -1.0, np.nan, np.inf, "x", [5.0]):
        _raises(SM.correct_soma, swc, th, r_eq, RES0)
    for res0 in (0.0, -0.1, np.nan):
        _raises(SM.correct_soma, swc, th, 5.0, res0)
        _raises(SM.image_to_swc_um, np.zeros((2, 3)), res0)
    _raises(SM.correct_soma, swc, th, 5.0, RES0, shift_full_px=(1.0,))
    _raises(SM.image_to_swc_um, np.zeros((2, 3)), RES0, flip_y_full_h=np.nan)
    _raises(SM.image_to_swc_um, np.zeros((2, 3)), RES0, z0_um=np.inf)
    # points and directions
    for P in (np.zeros(3), np.zeros((2, 2)), np.array([[0.0, np.nan, 0.0]])):
        _raises(SM.image_to_swc_um, P, RES0)
        _raises(SM.fit_frame, P, th)
        _raises(SM.line_crossings, P, [1.0, 0.0, 0.0], th)
    P = np.zeros((3, 3))
    for d in ([0.0, 0.0, 0.0], np.array([[1.0, 0, 0], [0, 0, 0], [0, 1.0, 0]]), [np.nan, 1.0, 0.0],
              np.ones((2, 3)), np.ones(2)):
        _raises(SM.line_crossings, P, d, th)
    # outlines
    for n in (2, 2.5, True, "x", 0):
        _raises(SM.ellipse_outline, th, None, n)
        _raises(SM.side_outline, th, n)
    _raises(SM.ellipse_outline, th, np.nan)
    # empty inputs give empty outputs
    E = np.zeros((0, 3))
    assert SM.image_to_swc_um(E, RES0).shape == (0, 3)
    assert SM.fit_frame(E, th).shape == (0, 3) and SM.ellipsoid_q(E, th).shape == (0,)
    t_lo, t_hi = SM.line_crossings(E, [1.0, 0.0, 0.0], th)
    assert t_lo.shape == (0,) and t_hi.shape == (0,)
    one_soma = swc_from_rows([(1, 1, 0.0, 0.0, 0.0, 5.0, -1)])
    e = SM.correct_soma(one_soma, th, 5.0, RES0)
    assert e.stems == () and e.keep.tolist() == [True]
    # the editor's refusals
    s = swc_from_text(SAMPLE)
    n = len(s)
    for kw in (dict(xyz=np.zeros((n, 2))), dict(xyz=np.zeros((n - 1, 3))), dict(radius=np.ones(n - 1)),
               dict(parent=np.ones(n + 1, dtype=int)), dict(keep=np.ones(n, dtype=int)),
               dict(keep=np.ones(n - 1, dtype=bool)), dict(parent=s.parent + 0.5)):
        _raises(swc_io.edited_lines, s, **kw)
    bad_xyz = s.xyz.copy()
    bad_xyz[2, 1] = np.nan
    _raises(swc_io.edited_lines, s, xyz=bad_xyz)
    for r in (np.zeros(n), -np.ones(n), np.full(n, np.inf)):
        _raises(swc_io.edited_lines, s, radius=r)
    keep = np.ones(n, dtype=bool)
    keep[2] = False                                       # node 3 removed, node 4 (parent 3) kept
    _raises(swc_io.edited_lines, s, keep=keep)
    par = s.parent.copy()
    par[3] = 99                                           # a parent id absent from the file
    _raises(swc_io.edited_lines, s, parent=par)
    par[3] = -1                                           # -1 allowed
    swc_io.edited_lines(s, parent=par)
    par = s.parent.astype(float)                          # integral floats accepted as ids
    assert swc_io.edited_lines(s, parent=par) == swc_io.edited_lines(s, parent=s.parent)
    keep[3] = False
    keep[4] = False                                       # node 3, 4, 5 removed together: valid
    lines = swc_io.edited_lines(s, keep=keep)
    assert sum(1 for ln in lines if ln.strip() and not ln.startswith("#")) == n - 3


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
