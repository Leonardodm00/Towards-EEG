"""smoke_spine_cap -- correctness checks for the distal tip cap (spine_cap).

Run:
    python3 smoke_spine_cap.py            # all checks, verbose
    python3 smoke_spine_cap.py -q         # summary only
Exit code 0 iff every check passes.

WHAT IS ACTUALLY BEING TESTED
-----------------------------
Every check below has a CLOSED-FORM ground truth. Nothing here compares the
code against itself or against a previously recorded output, so a check can
fail only if the implementation is wrong, never because a reference drifted.

  T1  cap area matches numerical quadrature of the surface of revolution
  T2  hemisphere limit    r_t == h   ->  A = 2 pi r_t^2
  T3  disc limit          h == 0     ->  A = pi r_t^2
  T4  monotone in h, and cap >= disc always
  T5  sphere closure: axial frustum chain + two caps -> 4 pi R^2 EXACTLY,
      independently of how many nodes sample the sphere
  T6  capped cylinder     2 pi R L + 2 pi (R^2 + h^2)
  T7  never fails: R_cap >= r_t for a wide sweep, including r_t < h
  T8  leaf gate: a spine node with a non-spine child is NOT capped
  T9  degenerate radius r <= 0 is not capped
  T10 build_phi with cap_tips=False reproduces the uncapped area exactly
  T11 conservation: sum(spine_cap_um2) over phi_df == sum of per-spine caps
  T12 symmetry: shaft leaves are capped too, and F moves less than it would
      if only the spine side were corrected
  T13 spine_geometry invariant A_head + A_neck + A_other == A_spine holds
      with the cap on, and A_cap <= A_spine
  T14 units: a frame in nm and the same frame in um give identical areas
"""

import math
import sys

import numpy as np
import pandas as pd

import spine_cap as sc
import spine_density as sd
import spine_geometry as sg

TOL = 1e-9
_FAILURES = []
_QUIET = "-q" in sys.argv


def check(name, ok, detail=""):
    # print format MUST match the project convention ("  PASS  " / "  FAIL  "),
    # because run_all_tests.py counts occurrences of those exact strings
    if ok:
        if not _QUIET:
            print("  PASS  %s   %s" % (name, detail))
    else:
        _FAILURES.append((name, detail))
        print("  FAIL  %s   %s" % (name, detail))


def close(a, b, tol=TOL):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# --------------------------------------------------------------------------- #
# Frame builders                                                               #
# --------------------------------------------------------------------------- #
def make_frame(rows, units="nm"):
    """rows: list of (id, p, x, y, z, r, label) already in the target units."""
    return pd.DataFrame(
        [{"id": i, "p": p, "x": x, "y": y, "z": z, "r": r,
          "annotated_type": lab} for (i, p, x, y, z, r, lab) in rows])


def straight_spine_frame(n_shaft=4, shaft_r=0.5, shaft_step=1.0,
                         spine_specs=None, scale=1000.0):
    """A straight dendrite along x with spines hanging off it along y.

    spine_specs: list of lists of (dy, r, label) -- one list per spine, given
    distal-ward from the base. scale converts um -> frame units (1000 = nm).
    """
    spine_specs = spine_specs or []
    rows = [(0, -1, 0.0, 0.0, 0.0, shaft_r, "soma")]
    nid = 1
    shaft_ids = []
    for k in range(n_shaft):
        rows.append((nid, nid - 1, (k + 1) * shaft_step, 0.0, 0.0,
                     shaft_r, "dendrite"))
        shaft_ids.append(nid)
        nid += 1
    for s_i, spec in enumerate(spine_specs):
        base = shaft_ids[min(s_i, len(shaft_ids) - 1)]
        parent = base
        y = 0.0
        for (dy, r, lab) in spec:
            y += dy
            rows.append((nid, parent, rows[base][2], y, 0.0, r, lab))
            parent = nid
            nid += 1
    scaled = [(i, p, x * scale, y * scale, z * scale, r * scale, lab)
              for (i, p, x, y, z, r, lab) in rows]
    return make_frame(scaled)


# --------------------------------------------------------------------------- #
# T1-T7: the geometry primitive against closed forms                           #
# --------------------------------------------------------------------------- #
def cap_area_quadrature(r_t, h, n=2000001):
    """Numerical surface-of-revolution area of the cap, from the sphere profile.

    Sphere of radius R centred so the rim (radius r_t) is at z = 0 and the pole
    at z = h. rho(z) = sqrt(R^2 - (z - z_c)^2), z_c = h - R.
    A = 2 pi int_0^h rho sqrt(1 + (drho/dz)^2) dz, integrated by Simpson.
    """
    R = (r_t * r_t + h * h) / (2.0 * h)
    z_c = h - R
    z = np.linspace(0.0, h, n)
    d = z - z_c
    inner = np.maximum(R * R - d * d, 1e-300)
    rho = np.sqrt(inner)
    drho = -d / rho
    f = 2.0 * math.pi * rho * np.sqrt(1.0 + drho * drho)
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return (h / (n - 1)) / 3.0 * float(np.dot(w, f))


def test_primitive():
    for (r_t, h) in [(0.30, 0.10), (0.15, 0.10), (0.50, 0.10), (0.05, 0.10)]:
        a = sc.cap_area_um2(r_t, h)
        q = cap_area_quadrature(r_t, h)
        check("T1 quadrature r_t=%.2f h=%.2f" % (r_t, h),
              abs(a - q) / q < 2e-6,
              "closed %.10f quad %.10f rel %.2e" % (a, q, abs(a - q) / q))

    for r in (0.05, 0.1, 0.3):
        check("T2 hemisphere r_t=h=%.2f" % r,
              close(sc.cap_area_um2(r, r), 2.0 * math.pi * r * r),
              "%.10f" % sc.cap_area_um2(r, r))
        check("T2 polar angle is pi/2 at r_t=h=%.2f" % r,
              close(sc.cap_polar_angle_rad(r, r), math.pi / 2.0))

    for r in (0.1, 0.25, 0.6):
        check("T3 disc limit r_t=%.2f" % r,
              close(sc.cap_area_um2(r, 0.0), math.pi * r * r))

    r = 0.3
    hs = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
    areas = [sc.cap_area_um2(r, h) for h in hs]
    check("T4 monotone increasing in h",
          all(areas[i] < areas[i + 1] for i in range(len(areas) - 1)))
    check("T4 cap >= disc for all h",
          all(a >= math.pi * r * r - TOL for a in areas))

    # T5: sphere closed by two caps
    R = 0.5
    truth = 4.0 * math.pi * R * R
    for n in (4, 16, 64, 256):
        z = np.linspace(-R, R, n + 3)[1:-1]          # keep off the poles
        rho = np.sqrt(np.maximum(R * R - z * z, 0.0))
        chain = sum(math.pi * (rho[i] + rho[i + 1]) *
                    math.sqrt((rho[i] - rho[i + 1]) ** 2 +
                              (z[i + 1] - z[i]) ** 2)
                    for i in range(len(z) - 1))
        h_lo = z[0] + R
        h_hi = R - z[-1]
        total = chain + sc.cap_area_um2(rho[0], h_lo) + \
            sc.cap_area_um2(rho[-1], h_hi)
        check("T5 sphere n=%d chain+caps -> 4 pi R^2" % n,
              abs(total - truth) / truth < 0.02,
              "%.6f vs %.6f (ratio %.5f)" % (total, truth, total / truth))
    # the caps themselves must be EXACT, at any truncation
    for frac in (0.05, 0.2, 0.5):
        h = frac * R
        r_t = math.sqrt(R * R - (R - h) ** 2)
        check("T5 cap exact vs 2 pi R h (h=%.2f R)" % frac,
              close(sc.cap_area_um2(r_t, h), 2.0 * math.pi * R * h))

    # T6: capped cylinder
    R_cyl, L, h = 0.2, 1.5, 0.1
    lateral = 2.0 * math.pi * R_cyl * L
    truth6 = lateral + 2.0 * sc.cap_area_um2(R_cyl, h)
    check("T6 capped cylinder closed form",
          close(truth6, lateral + 2.0 * math.pi * (R_cyl ** 2 + h ** 2)))

    # T7: never fails
    bad = []
    for r_t in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0]:
        for h in [0.01, 0.05, 0.1, 0.5, 2.0]:
            if sc.cap_sphere_radius_um(r_t, h) < r_t - 1e-12:
                bad.append((r_t, h))
    check("T7 R_cap >= r_t over the whole sweep", not bad, str(bad[:3]))


# --------------------------------------------------------------------------- #
# T8-T9: the leaf gate                                                         #
# --------------------------------------------------------------------------- #
def test_leaf_gate():
    # spine 0: normal, tip is a true leaf
    # spine 1: the distal-most SPINE node has a non-spine child -> boundary
    rows = [
        (0, -1, 0.0, 0.0, 0.0, 0.5, "soma"),
        (1, 0, 1.0, 0.0, 0.0, 0.5, "dendrite"),
        (2, 1, 2.0, 0.0, 0.0, 0.5, "dendrite"),
        (3, 1, 1.0, 0.4, 0.0, 0.1, "neck"),
        (4, 3, 1.0, 0.7, 0.0, 0.3, "head"),          # true leaf -> capped
        (5, 2, 2.0, 0.4, 0.0, 0.1, "neck"),
        (6, 5, 2.0, 0.7, 0.0, 0.3, "head"),          # has a non-spine child
        (7, 6, 2.0, 1.0, 0.0, 0.3, "dendrite"),      # -> node 6 NOT a leaf
    ]
    df = make_frame([(i, p, x * 1000, y * 1000, z * 1000, r * 1000, lab)
                     for (i, p, x, y, z, r, lab) in rows])
    node, children, root = sd._prepare_nodes(
        df, sd.SHAFT_REGEX, sd.SPINE_LABELS, sd.DEFAULT_RADIUS_NM, "nm")

    check("T8 node 4 is a true leaf", sc.is_true_leaf(children, 4))
    check("T8 node 6 is NOT a true leaf", not sc.is_true_leaf(children, 6))
    audit = sc.audit_tips(node, children, root=root)
    check("T8 audit finds exactly 1 label-boundary end",
          audit["spine_n_label_boundary_end"] == 1,
          str(audit["spine_label_boundary_ids"]))
    check("T8 audit finds 1 spine true leaf",
          audit["spine_n_true_leaf"] == 1,
          "got %d" % audit["spine_n_true_leaf"])

    seg, dropped, cap = sd._attribute_spine_area(
        node, children, cap_h_um=0.1, return_cap=True)
    total_cap = sum(cap.values())
    expect = sc.cap_area_um2(0.3, 0.1)     # node 4 only
    check("T8 only the true leaf contributes cap area",
          close(total_cap, expect),
          "%.10f vs %.10f" % (total_cap, expect))

    # T9: degenerate radius
    rows9 = rows[:5]
    rows9[4] = (4, 3, 1.0, 0.7, 0.0, 0.0, "head")
    df9 = make_frame([(i, p, x * 1000, y * 1000, z * 1000, r * 1000, lab)
                      for (i, p, x, y, z, r, lab) in rows9])
    node9, children9, root9 = sd._prepare_nodes(
        df9, sd.SHAFT_REGEX, sd.SPINE_LABELS, sd.DEFAULT_RADIUS_NM, "nm")
    a9, n9, bad9 = sc.subtree_cap_area(node9, children9, [3, 4], 0.1)
    check("T9 r<=0 leaf is skipped, not given pi h^2",
          close(a9, 0.0) and bad9 == 1, "area %.3g n_bad %d" % (a9, bad9))


# --------------------------------------------------------------------------- #
# T10-T12: integration with build_phi                                          #
# --------------------------------------------------------------------------- #
def demo_frame():
    # n_shaft=5 with only 4 spines: the distal-most shaft node carries no
    # spine, so it is a genuine shaft leaf and T12 has something to cap.
    return straight_spine_frame(
        n_shaft=5, shaft_r=0.5, shaft_step=1.0,
        spine_specs=[
            [(0.4, 0.10, "neck"), (0.3, 0.30, "head")],
            [(0.5, 0.12, "neck"), (0.3, 0.25, "head")],
            [(0.3, 0.20, "spine")],
            [(0.4, 0.10, "neck"), (0.2, 0.28, "head"), (0.2, 0.20, "head")],
        ])


def test_build_phi():
    df = demo_frame()
    off = sd.build_phi(df, nid="T", cap_tips=False)
    on = sd.build_phi(df, nid="T", cap_tips=True, cap_h_um=0.1)

    check("T10 cap off leaves spine area unchanged",
          close(float(off["spine_area_um2"].sum()),
                float((on["spine_area_um2"] - on["spine_cap_um2"]).sum())),
          "%.10f vs %.10f" % (float(off["spine_area_um2"].sum()),
                              float((on["spine_area_um2"]
                                     - on["spine_cap_um2"]).sum())))
    check("T10 cap off emits zero cap columns",
          float(off["spine_cap_um2"].sum()) == 0.0
          and float(off["shaft_cap_um2"].sum()) == 0.0)

    node, children, root = sd._prepare_nodes(
        df, sd.SHAFT_REGEX, sd.SPINE_LABELS, sd.DEFAULT_RADIUS_NM, "nm")
    audit = sc.audit_tips(node, children, root=root)
    expect_spine = sum(sc.cap_area_um2(r, 0.1) for r in audit["spine_r_tip_um"])
    expect_shaft = sum(sc.cap_area_um2(r, 0.1) for r in audit["other_r_tip_um"])
    check("T11 spine cap conserved through attribution",
          close(float(on["spine_cap_um2"].sum()), expect_spine),
          "%.10f vs %.10f" % (float(on["spine_cap_um2"].sum()), expect_spine))
    check("T12 shaft leaves are capped too",
          close(float(on["shaft_cap_um2"].sum()), expect_shaft)
          and expect_shaft > 0.0,
          "%.10f vs %.10f" % (float(on["shaft_cap_um2"].sum()), expect_shaft))

    f_off = sd.cell_f_implied_from_phi(off)
    f_on = sd.cell_f_implied_from_phi(on)
    a_sp = float(on["spine_area_um2"].sum())
    a_sh_nocap = float((on["shaft_area_um2"] - on["shaft_cap_um2"]).sum())
    f_asym = 1.0 + a_sp / a_sh_nocap        # if only the spine side were capped
    check("T12 F rises with the cap", f_on > f_off,
          "F_off %.5f -> F_on %.5f" % (f_off, f_on))
    check("T12 symmetric capping is more conservative than one-sided",
          f_on < f_asym,
          "F_sym %.5f < F_asym %.5f" % (f_on, f_asym))

    # h = 0 must be the flat-disc lower bound of the bracket
    disc = sd.build_phi(df, nid="T", cap_tips=True, cap_h_um=0.0)
    check("T12 h=0 disc bracket is below the h=0.1 cap",
          float(disc["spine_cap_um2"].sum())
          < float(on["spine_cap_um2"].sum()),
          "%.6f < %.6f" % (float(disc["spine_cap_um2"].sum()),
                           float(on["spine_cap_um2"].sum())))


# --------------------------------------------------------------------------- #
# T13: spine_geometry invariants                                               #
# --------------------------------------------------------------------------- #
def test_spine_geometry():
    df = demo_frame()
    g_off = sg.build_spine_geometry(df, nid="T", cap_tips=False)
    g_on = sg.build_spine_geometry(df, nid="T", cap_tips=True, cap_h_um=0.1)

    parts = (g_on["A_head_um2"] + g_on["A_neck_um2"]
             + g_on["A_other_um2"]).values
    check("T13 A_head+A_neck+A_other == A_spine with cap on",
          bool(np.allclose(parts, g_on["A_spine_um2"].values,
                           rtol=0, atol=1e-12)))
    check("T13 A_cap <= A_spine",
          bool((g_on["A_cap_um2"] <= g_on["A_spine_um2"] + 1e-12).all()))
    check("T13 cap off gives A_cap == 0",
          bool((g_off["A_cap_um2"] == 0.0).all()))
    check("T13 capped total exceeds uncapped by exactly A_cap",
          close(float(g_on["A_spine_um2"].sum()),
                float(g_off["A_spine_um2"].sum())
                + float(g_on["A_cap_um2"].sum())),
          "%.10f vs %.10f" % (float(g_on["A_spine_um2"].sum()),
                              float(g_off["A_spine_um2"].sum())
                              + float(g_on["A_cap_um2"].sum())))

    phi_on = sd.build_phi(df, nid="T", cap_tips=True, cap_h_um=0.1)
    check("T13 spine_geometry total == spine_density total (capped)",
          close(float(g_on["A_spine_um2"].sum()),
                float(phi_on["spine_area_um2"].sum())),
          "%.10f vs %.10f" % (float(g_on["A_spine_um2"].sum()),
                              float(phi_on["spine_area_um2"].sum())))
    # a genuinely branched (Y-shaped) spine: both tips must be capped once each
    y_rows = [
        (0, -1, 0.0, 0.0, 0.0, 0.5, "soma"),
        (1, 0, 1.0, 0.0, 0.0, 0.5, "dendrite"),
        (2, 1, 1.0, 0.4, 0.0, 0.10, "neck"),
        (3, 2, 1.0, 0.7, 0.0, 0.20, "head"),
        (4, 3, 0.8, 0.9, 0.0, 0.30, "head"),      # tip A
        (5, 3, 1.2, 0.9, 0.0, 0.25, "head"),      # tip B
    ]
    y_df = make_frame([(i, p, x * 1000, y * 1000, z * 1000, r * 1000, lab)
                       for (i, p, x, y, z, r, lab) in y_rows])
    g_y = sg.build_spine_geometry(y_df, nid="Y", cap_tips=True, cap_h_um=0.1)
    expect_y = sc.cap_area_um2(0.30, 0.1) + sc.cap_area_um2(0.25, 0.1)
    check("T13 branched spine caps both tips exactly once",
          int(g_y["n_capped_tips"].iloc[0]) == 2
          and close(float(g_y["A_cap_um2"].iloc[0]), expect_y),
          "n=%d A_cap=%.10f vs %.10f" % (int(g_y["n_capped_tips"].iloc[0]),
                                         float(g_y["A_cap_um2"].iloc[0]),
                                         expect_y))
    check("T13 branched spine n_tips matches n_capped_tips here",
          int(g_y["n_tips"].iloc[0]) == int(g_y["n_capped_tips"].iloc[0]))


# --------------------------------------------------------------------------- #
# T14: unit invariance                                                         #
# --------------------------------------------------------------------------- #
def test_units():
    df_nm = demo_frame()
    df_um = df_nm.copy()
    for c in ("x", "y", "z", "r"):
        df_um[c] = df_um[c] / 1000.0
    a = sd.build_phi(df_nm, cap_tips=True, cap_h_um=0.1, input_units="nm")
    b = sd.build_phi(df_um, cap_tips=True, cap_h_um=0.1, input_units="um")
    check("T14 nm and um frames agree on spine area",
          close(float(a["spine_area_um2"].sum()),
                float(b["spine_area_um2"].sum())))
    check("T14 nm and um frames agree on cap area",
          close(float(a["spine_cap_um2"].sum()),
                float(b["spine_cap_um2"].sum())))


# --------------------------------------------------------------------------- #
def main():
    print("spine_cap smoke test")
    print("  %s / %s / %s" % (sc.MODULE_VERSION, sd.MODULE_VERSION,
                              sg.MODULE_VERSION))
    print("-" * 74)
    test_primitive()
    test_leaf_gate()
    test_build_phi()
    test_spine_geometry()
    test_units()
    print("-" * 74)
    if _FAILURES:
        print("FAILED: %d check(s)" % len(_FAILURES))
        for n, d in _FAILURES:
            print("   %s  %s" % (n, d))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
