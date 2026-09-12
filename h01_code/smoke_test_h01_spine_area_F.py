#!/usr/bin/env python3
"""Smoke test for h01_spine_area_F. Offline, no Drive, no Stage 1 folder.

RUN
    python3 smoke_test_h01_spine_area_F.py        (quiet: failures + tally)
    python3 smoke_test_h01_spine_area_F.py -v     (every check)

Exit code 0 = all passed, else the number of failures.

WHAT IS CHECKED, AND AGAINST WHAT
  T1  frame convention: mesh centroids nudged inward land inside the mask
  T2  isolated sphere: everything is membrane; nothing cut, boxed, unresolved
  T3  cylinder abutting shaft context: cut area ~ pi a^2, membrane ~ lateral+disc
  T4  cylinder + interpolated bridge: bridge area ~ 2 pi a gap, counted
  T5  sphere with an internal hole: cavity area ~ 4 pi r_hole^2, excluded
  T6  object touching the cutout face: BOX class appears, clipped flag set
  T7  calibration: constant g gives A_raw / g exactly; histogram re-application
      of a smooth table within 0.5 percent of the per-triangle sum
  T8  table gates: the v6 artefact (g max 2.76), NaN, wrong grid all refused
  T9  skeleton side against a FAKE spine_density written from the documented
      v1.1 rule: gate passes, tampering fails it, F algebra exact
  T10 kappa function recovers a planted two-level kappa(A_skel)
  T11 end to end: stub seg fetch -> Voronoi ROI -> mesh -> ledger; resume
      reproduces the first run exactly
  T12 figure: four classes coloured, a node on the spine axis lands on a
      blue pixel (extent convention), cut crosses sit on the seam and are
      closer with the +res/2 shift than without it
  T13 figure callback inside the batch: PNG written and recorded; a failing
      callback is recorded and does NOT fail the measurement
  T15 junction terms: rind area on a cylinder against the exact disc/total
      area, the axial window, NaN with no axis, the base frustum against the
      closed-form frustum on the toy cell, and the kappa variants
  T14 minimum-length filter: exact metrics on the toy cell, threshold report,
      demotion keeps sigma ids stable, moves exactly A_skel(sigma) from the
      spine column to the shaft, keeps the attribution gate passing, and a
      demoted spine inherits its base's shaft label (apical stays apical)

T9 proves internal consistency only. Agreement with the REAL spine_density is
established at run time by attribution_gate on every cell.

Pure ASCII, LF only.
"""

import os
import re
import shutil
import sys
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")                     # headless; set before any pyplot import
import numpy as np
import pandas as pd

import h01_area_calibration as CAL
import h01_spine_area_F as SAF
import h01_spine_batch as B
import h01_spine_roi as SR

VERBOSE = "-v" in sys.argv
RES = (8.0, 8.0, 33.0)
RESULTS = []
SKIPPED = []


def skip(name, reason):
    """Recorded as neither pass nor fail. Used for the figure tests, which
    need matplotlib and h01_spine_area_F_figures -- absent by design in the
    lean HPC bundle, where no figures are drawn."""
    SKIPPED.append((name, reason))
    print("  [SKIP] %-56s %s" % (name, reason))
    return False


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    if VERBOSE or not ok:
        print("  [%s] %-58s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


def _run(fn):
    print("== %s" % fn.__name__)
    try:
        fn()
    except Exception:                                   # noqa: BLE001
        RESULTS.append((fn.__name__ + " raised", False))
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def grid(shape, lo=(0, 0, 0)):
    """Node-frame voxel centres (g + 0.5) * res for a cutout at lo."""
    r = np.asarray(RES)
    ax = [(np.arange(n) + lo[i] + 0.5) * r[i] for i, n in enumerate(shape)]
    return np.meshgrid(*ax, indexing="ij")


def roi_dict(mask, ctx=None, bridge=None, lo=(0, 0, 0), seg=None):
    return {"mask": mask, "bridge_mask": bridge,
            "shaft_context_mask": np.zeros_like(mask) if ctx is None else ctx,
            "seg": seg, "meta": {"layers": {"seg": {
                "resolution_nm": list(RES), "lo_vox": list(lo)}}}}


def const_lookup(g0):
    return lambda n: np.full(len(np.atleast_2d(n)), float(g0))


def measure(roi, g0=1.0, cell_id=None):
    return SAF.measure_spine_area(roi, const_lookup(g0), dict(B.DEFAULTS),
                                  cell_id=cell_id)


def y_cylinder(a, y_lo, y_hi, shape, cx, cz, lo=(0, 0, 0)):
    X, Y, Z = grid(shape, lo)
    return ((X - cx) ** 2 + (Z - cz) ** 2 <= a * a) & (Y >= y_lo) & (Y < y_hi)


# --------------------------------------------------------------------------- #
def _inside_fraction(verts, faces, U, eps, mesh_convention=True):
    c, a, n, _ = SAF.triangle_geometry(verts, faces)
    p = c - eps * n
    q = (SAF.mesh_point_to_local_voxel(p, (0, 0, 0), RES) if mesh_convention
         else np.floor(p / np.asarray(RES)).astype(int))
    ok = np.all((q >= 0) & (q < np.asarray(U.shape)), axis=1)
    inside = np.zeros(len(q), bool)
    inside[ok] = U[q[ok, 0], q[ok, 1], q[ok, 2]]
    return float((a * inside).sum() / a.sum())


def test_T1_frame_convention():
    """On an axis-aligned BOX the raw marching-cubes faces lie exactly on
    voxel faces, so a 2 nm inward nudge must land inside under the correct
    convention -- no smoothing or corner-cutting to blur the answer. (On a
    sphere the smoothed mesh legitimately sits up to ~17 nm off the voxel
    boundary, so a nudge test there cannot isolate the frame.)"""
    X, Y, Z = grid((60, 60, 20))
    m = (X > 100) & (X < 380) & (Y > 90) & (Y < 400) & (Z > 120) & (Z < 500)
    raw = SR.surface_from_mask(m, RES, (0, 0, 0))
    f_mesh = _inside_fraction(raw["verts_nm"], raw["faces"], m, 2.0, True)
    f_node = _inside_fraction(raw["verts_nm"], raw["faces"], m, 2.0, False)
    check("T1a box, mesh convention: inward nudge inside >= 0.999",
          f_mesh >= 0.999, "%.4f" % f_mesh)
    check("T1b box, node convention fails (half-voxel offset) <= 0.6",
          f_node <= 0.6, "%.4f" % f_node)
    verts, faces, _, _, _, _ = B.analysis_mesh(roi_dict(m), dict(B.DEFAULTS))
    check("T1c normals point outward (volume > 0) on the smoothed mesh",
          SAF.triangle_geometry(verts, faces)[3] > 0)


def test_T2_isolated_sphere():
    X, Y, Z = grid((80, 80, 24))
    R = 250.0
    m = (X - 321.0) ** 2 + (Y - 318.0) ** 2 + (Z - 395.0) ** 2 <= R * R
    rec, H = measure(roi_dict(m))
    check("T2a membrane fraction >= 0.995", rec["frac_membrane"] >= 0.995,
          "%.4f" % rec["frac_membrane"])
    check("T2b no cut / box / cavity", rec["frac_cut"] + rec["frac_box"]
          + rec["frac_cavity"] < 1e-9)
    ratio = rec["A_mesh_raw_um2"] * 1e6 / (4 * np.pi * R * R)
    check("T2c raw area / 4 pi R^2 in [0.98, 1.12]", 0.98 <= ratio <= 1.12,
          "%.4f" % ratio)
    check("T2d histogram carries all counted area",
          abs(H.sum() / 1e6 - rec["A_mesh_raw_um2"]) < 1e-9)


def test_T3_cut_face():
    a, L = 150.0, 900.0
    shape = (70, 180, 24)
    cx, cz, y0 = 281.0, 393.0, 400.0
    S = y_cylinder(a, y0, y0 + L, shape, cx, cz)
    X, Y, Z = grid(shape)
    ctx = (Y < y0) & (Y > y0 - 300.0) & (np.abs(X - cx) < 260.0) \
        & (np.abs(Z - cz) < 300.0)
    rec, _ = measure(roi_dict(S, ctx=ctx))
    cut = rec["A_cut_raw_um2"] * 1e6 / (np.pi * a * a)
    mem = rec["A_membrane_raw_um2"] * 1e6 / (2 * np.pi * a * L + np.pi * a * a)
    check("T3a cut area / pi a^2 in [0.75, 1.30]", 0.75 <= cut <= 1.30,
          "%.3f" % cut)
    check("T3b membrane / (lateral + top disc) in [0.95, 1.15]",
          0.95 <= mem <= 1.15, "%.3f" % mem)
    check("T3c cut area is NOT counted",
          abs(rec["A_mesh_raw_um2"] - rec["A_membrane_raw_um2"]) < 1e-12)


def test_T4_bridge():
    a, gap, L = 120.0, 200.0, 700.0
    shape = (70, 180, 24)
    cx, cz, y0 = 281.0, 393.0, 300.0
    S = y_cylinder(a, y0 + gap, y0 + gap + L, shape, cx, cz)
    Bm = y_cylinder(a, y0, y0 + gap, shape, cx, cz)
    X, Y, Z = grid(shape)
    ctx = (Y < y0) & (Y > y0 - 250.0) & (np.abs(X - cx) < 250.0) \
        & (np.abs(Z - cz) < 300.0)
    rec, _ = measure(roi_dict(S, ctx=ctx, bridge=Bm))
    br = rec["A_bridge_raw_um2"] * 1e6 / (2 * np.pi * a * gap)
    check("T4a has_bridge", rec["has_bridge"])
    check("T4b bridge area / 2 pi a gap in [0.70, 1.35]", 0.70 <= br <= 1.35,
          "%.3f" % br)
    check("T4c bridge area is counted",
          abs(rec["A_mesh_raw_um2"] - rec["A_membrane_raw_um2"]
              - rec["A_bridge_raw_um2"]) < 1e-12)
    check("T4d bridge area reported separately (A_bridge_um2 > 0)",
          rec["A_bridge_um2"] > 0)


def test_T5_cavity():
    X, Y, Z = grid((90, 90, 26))
    R, r = 330.0, 120.0
    d2 = (X - 361.0) ** 2 + (Y - 357.0) ** 2 + (Z - 430.0) ** 2
    m = (d2 <= R * R) & ~(d2 <= r * r)
    rec, _ = measure(roi_dict(m))
    cav = rec["A_cavity_raw_um2"] * 1e6 / (4 * np.pi * r * r)
    check("T5a cavity area / 4 pi r^2 in [0.70, 1.30]", 0.70 <= cav <= 1.30,
          "%.3f" % cav)
    check("T5b cavity not counted", rec["A_mesh_raw_um2"] < 1.1 * 4 * np.pi
          * R * R / 1e6)


def test_T6_box():
    X, Y, Z = grid((60, 60, 20))
    m = (X - 240.0) ** 2 + (Z - 330.0) ** 2 <= 150.0 ** 2      # runs through y
    rec, _ = measure(roi_dict(m))
    check("T6a clipped flag", rec["clipped"])
    check("T6b BOX class present", rec["frac_box"] > 0.01,
          "%.4f" % rec["frac_box"])


def test_T7_calibration_arithmetic():
    X, Y, Z = grid((80, 80, 24))
    m = (X - 321.0) ** 2 + (Y - 318.0) ** 2 + (Z - 395.0) ** 2 <= 260.0 ** 2
    rec1, H = measure(roi_dict(m), g0=1.0)
    rec2, _ = measure(roi_dict(m), g0=1.05)
    check("T7a constant g: A_corr = A_raw / g",
          abs(rec2["A_mesh_um2"] - rec1["A_mesh_raw_um2"] / 1.05) < 1e-12)
    tmp = tempfile.mkdtemp()
    try:
        for lab, th, ph in (("build_g_table axes", np.arange(46) * 2.0,
                             np.arange(23) * 2.0),
                            ("fundamental_grid axes",) + tuple(CAL.fundamental_grid(2.0))):
            T, P = np.meshgrid(th, ph, indexing="ij")
            g = (1.0 + 0.07 * np.sin(np.radians(T)) ** 2
                 + 0.02 * np.sin(np.radians(2 * P)) * np.sin(np.radians(T)))
            tab, look, _ = SAF.load_calibration(_write_table(
                os.path.join(tmp, "t.npz"), th, ph, g))
            rec3, H3 = SAF.measure_spine_area(roi_dict(m), look, dict(B.DEFAULTS))
            rel = abs(SAF.area_from_histogram(H3, tab) / 1e6 - rec3["A_mesh_um2"]) \
                / rec3["A_mesh_um2"]
            check("T7b histogram re-application within 0.5%% (%s)" % lab,
                  rel < 0.005, "%.5f" % rel)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # nearest-axis binning: phi = 43.1 deg -> 44 on build axes, 42.95 on fundamental
    n = np.array([[np.cos(np.radians(43.1)), np.sin(np.radians(43.1)), 0.0]])
    Hb = SAF.normal_histogram(n, [1.0], np.arange(46) * 2.0, np.arange(23) * 2.0)
    thf, phf = CAL.fundamental_grid(2.0)
    Hf = SAF.normal_histogram(n, [1.0], thf, phf)
    check("T7c bins follow the table's own phi axis (43.1 -> 44 vs 42.95)",
          Hb[45, 22] == 1.0 and Hf[45, 21] == 1.0,
          "%s %s" % (np.argwhere(Hb).tolist(), np.argwhere(Hf).tolist()))


def _write_table(path, th, ph, g, meta=True):
    CAL.save_table(path, {"theta_deg": th, "phi_deg": ph, "g": g,
                          "meta": {"resolution_nm": [8.0, 8.0, 33.0]} if meta else {}})
    return path


def test_T8_table_gates():
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(0)
        th_b, ph_b = np.arange(46) * 2.0, np.arange(23) * 2.0      # build_g_table
        th_f, ph_f = CAL.fundamental_grid(2.0)                    # calibrate_g_table
        good = 1.0 + 0.05 * rng.random((46, 23))
        t, look, rep = SAF.load_calibration(_write_table(
            os.path.join(tmp, "b.npz"), th_b, ph_b, good))
        check("T8a REGRESSION: build_g_table axes (phi 0..44) load",
              rep["axes"]["phi"][1] == 44.0, str(rep["axes"]))
        check("T8b lookup carries the table's axes for histogram binning",
              np.array_equal(look.phi_deg, ph_b))
        _, _, rep_f = SAF.load_calibration(_write_table(
            os.path.join(tmp, "f.npz"), th_f, ph_f, good))
        check("T8c fundamental_grid axes (phi 0..45) also load",
              abs(rep_f["axes"]["phi"][1] - 45.0) < 1e-9)
        check("T8d lookup is NOT clipped at one",
              float(np.min(look(np.array([[0.0, 0.0, 1.0]])))) >= 0.0)
        th_nu = th_b.copy(); th_nu[10] += 0.7
        bad = (("v6 artefact", th_b, ph_b, np.where(np.arange(good.size).reshape(
                    good.shape) == 300, 2.76, good)),
               ("NaN", th_b, ph_b, np.where(good > 1.04, np.nan, good)),
               ("shape/axes mismatch", th_b, ph_b, good[:-1]),
               ("non-uniform theta", th_nu, ph_b, good),
               ("theta stops at 88", th_b[:-1], ph_b, good[:-1]),
               ("phi stops at 40", th_b, ph_b[:-2], good[:, :-2]))
        for name, th, ph, g in bad:
            q = os.path.join(tmp, "bad.npz")
            np.savez_compressed(q, theta_deg=th, phi_deg=ph, g=g)
            try:
                SAF.load_calibration(q)
                check("T8 refuses %s" % name, False, "accepted")
            except SAF.SpineAreaError:
                check("T8 refuses %s" % name, True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Fake Stage 1 spine_density, from the documented v1.1 rule                    #
# --------------------------------------------------------------------------- #
class FakeSD(object):
    SHAFT_REGEX = r"dendrite|apical|^1$"
    SPINE_LABELS = ("spine", "head", "neck")
    DEFAULT_RADIUS_NM = 50.0

    @staticmethod
    def _frustum_lateral_area(r1, r2, L):
        return np.pi * (r1 + r2) * np.sqrt((r1 - r2) ** 2 + L ** 2)

    @staticmethod
    def _segment_length_um(node, a, b):
        pa, pb = node[a], node[b]
        return float(np.sqrt((pa["x"] - pb["x"]) ** 2 + (pa["y"] - pb["y"]) ** 2
                             + (pa["z"] - pb["z"]) ** 2))

    def _prepare_nodes(self, df, shaft_regex, spine_labels, rdef, units):
        s = 1000.0 if units == "nm" else 1.0
        node, children = {}, {}
        for r in df.itertuples(index=False):
            t = str(r.annotated_type).lower()
            rad = float(r.r) if np.isfinite(r.r) and r.r > 0 else rdef
            node[int(r.id)] = {"p": int(r.p), "x": r.x / s, "y": r.y / s,
                               "z": r.z / s, "r": rad / s,
                               "is_spine": t in spine_labels,
                               "is_shaft": bool(re.search(shaft_regex, t))}
        for i, a in node.items():
            if a["p"] in node:
                children.setdefault(a["p"], []).append(i)
        root = [i for i, a in node.items() if a["p"] not in node][0]
        return node, children, root

    def build_phi(self, df, nid=None, input_units="nm"):
        node, ch, root = self._prepare_nodes(df, self.SHAFT_REGEX,
                                             self.SPINE_LABELS,
                                             self.DEFAULT_RADIUS_NM, input_units)
        dist, stack = {root: 0.0}, [root]
        while stack:
            u = stack.pop()
            for c in ch.get(u, ()):
                dist[c] = dist[u] + self._segment_length_um(node, u, c)
                stack.append(c)
        # attribution, coded independently: walk spine roots, sum subtree
        acc = {}
        for i, a in node.items():
            if not a["is_spine"] or node.get(a["p"], {}).get("is_spine"):
                continue
            area, todo = 0.0, [i]
            while todo:
                s_ = todo.pop()
                sp = node[s_]["p"]
                area += self._frustum_lateral_area(
                    node[sp]["r"], node[s_]["r"],
                    self._segment_length_um(node, sp, s_))
                todo += [c for c in ch.get(s_, ()) if node[c]["is_spine"]]
            b = a["p"]
            if node[b]["p"] in node:
                k = (node[b]["p"], b)
            else:
                k = (b, [c for c in ch[b] if node[c]["is_shaft"]][0])
            acc[k] = acc.get(k, 0.0) + area
        rows = []
        for b, nb in node.items():
            if not nb["is_shaft"] or nb["p"] not in node:
                continue
            a_ = nb["p"]
            L = self._segment_length_um(node, a_, b)
            sp = acc.get((a_, b), 0.0)
            dlt = node[a_]["r"] + nb["r"]
            rows.append({"nid": nid, "branch_id": 0, "node_from": a_,
                         "node_to": b, "seg_len_um": L,
                         "d_from_um": dist[a_], "d_to_um": dist[b],
                         "shaft_diam_um": dlt,
                         "shaft_area_um2": self._frustum_lateral_area(
                             node[a_]["r"], nb["r"], L),
                         "spine_area_um2": sp, "phi_um": sp / L,
                         "psi": sp / L / (np.pi * dlt)})
        return pd.DataFrame(rows)

    @staticmethod
    def cell_f_beyond_cutoff(phi, cutoff_um=60.0, by="d_from_um"):
        s = phi[phi[by] >= cutoff_um]
        return {"F": 1.0 + s["spine_area_um2"].sum() / s["shaft_area_um2"].sum()}


def toy_cell():
    """Soma root, 12 shaft nodes 10 um apart, four spines (one on the root)."""
    rows = [dict(id=0, p=-1, x=0.0, y=0.0, z=0.0, r=5000.0,
                 annotated_type="soma")]
    for i in range(1, 13):
        rows.append(dict(id=i, p=i - 1, x=10000.0 * i, y=0.0, z=0.0,
                         r=500.0 - 20.0 * i, annotated_type="dendrite"))
    nid = 100
    for base, n in ((0, 2), (3, 3), (7, 4), (9, 2)):
        par = base
        for j in range(n):
            rows.append(dict(id=nid, p=par, x=10000.0 * base,
                             y=400.0 * (j + 1), z=0.0, r=120.0 + 30.0 * j,
                             annotated_type="neck" if j < n - 1 else "head"))
            par = nid
            nid += 1
    df = pd.DataFrame(rows)
    comp = np.full(len(df), -1, dtype=np.int64)
    k = 0
    for base in (0, 3, 7, 9):
        roots = df.index[(df["p"] == base) & (df["id"] >= 100)]
        stack = list(roots)
        while stack:
            i = stack.pop()
            comp[i] = k
            stack += list(df.index[df["p"] == df.loc[i, "id"]])
        k += 1
    return df, comp


def test_T9_skeleton_and_F():
    sd = FakeSD()
    df, comp = toy_cell()
    sk = SAF.skeleton_spine_table(sd, df, df, comp)
    phi = sd.build_phi(df, nid=1)
    gate = SAF.attribution_gate(phi, sk)
    check("T9a attribution gate passes", gate["pass"], str(gate))
    check("T9b spine on the root is mapped (fallback rule)",
          gate["n_spines_unmapped"] == 0)
    bad = sk.copy()
    bad.loc[1, "A_skel_um2"] *= 1.01
    check("T9c gate FAILS when one spine is off by 1%",
          not SAF.attribution_gate(phi, bad)["pass"])
    sk["A2"] = 2.0 * sk["A_skel_um2"]
    pm = SAF.phi_with_spine_areas(phi, sk, "A2")
    f0, f2 = SAF.cell_F(phi, 60.0)["F"], SAF.cell_F(pm, 60.0)["F"]
    check("T9d doubling every spine doubles F_lit - 1",
          abs((f2 - 1) - 2 * (f0 - 1)) < 1e-12, "%.6f %.6f" % (f0, f2))
    check("T9e cutoff excludes the proximal spines",
          SAF.cell_F(phi, 60.0)["A_spine_um2"]
          < SAF.cell_F(phi, 0.0)["A_spine_um2"])
    check("T9f F_lit equals Stage 1's own function",
          abs(f0 - sd.cell_f_beyond_cutoff(phi)["F"]) < 1e-12)
    check("T9g phi^mesh conserves the substituted area",
          abs(pm["spine_area_um2"].sum() - sk["A2"].sum()) < 1e-12)
    recs = {0: {"sigma_id": 0, "ok": True, "A_mesh_um2": 3.0 * sk.loc[0, "A_skel_um2"],
                "A_mesh_raw_um2": 3.3 * sk.loc[0, "A_skel_um2"], "clipped": False,
                "frac_cut": .1, "frac_bridge": 0., "frac_box": 0.,
                "frac_unresolved": 0.},
            2: {"sigma_id": 2, "ok": False, "error": "x"}}
    out = SAF.assemble_cell(sd, df, df, comp, recs, 1)
    s = out["spines"].set_index("sigma_id")
    check("T9h measured spine uses A_mesh", abs(s.loc[0, "A_used_skelfill_um2"]
                                                 - 3.0 * sk.loc[0, "A_skel_um2"]) < 1e-12)
    check("T9i unmeasured spine: skelfill = A_skel, kappafill = 3 A_skel",
          abs(s.loc[3, "A_used_skelfill_um2"] - sk.loc[3, "A_skel_um2"]) < 1e-12
          and abs(s.loc[3, "A_used_kappafill_um2"] - 3.0 * sk.loc[3, "A_skel_um2"]) < 1e-9)
    check("T9j summary counts", out["summary"]["n_measured"] == 1
          and out["summary"]["n_failed"] == 1)
    pick = SAF.choose_sigmas(sk.iloc[:3], subset=2, seed=1)
    check("T9k stratified pick on a cell with fewer spines than strata",
          len(pick) == 2 and set(pick) <= set(sk["sigma_id"].iloc[:3]), str(pick))
    check("T9l subset=None returns every spine",
          sorted(SAF.choose_sigmas(sk)) == sorted(sk["sigma_id"].tolist()))


def test_T10_kappa_function():
    rng = np.random.default_rng(3)
    A = np.exp(rng.normal(0.0, 0.8, 400))
    k = np.where(A < np.median(A), 1.5, 2.0) * np.exp(rng.normal(0, 0.05, 400))
    sk = pd.DataFrame({"A_skel_um2": A, "A_mesh_um2": k * A, "measured": True})
    tab = SAF.kappa_function(sk, n_bins=4)
    lo, hi = tab["kappa"].iloc[0], tab["kappa"].iloc[-1]
    check("T10a small-spine kappa ~1.5", abs(lo - 1.5) < 0.05, "%.3f" % lo)
    check("T10b large-spine kappa ~2.0", abs(hi - 2.0) < 0.05, "%.3f" % hi)
    kh = SAF.kappa_apply(tab, np.array([A.min(), A.max() * 10]))
    check("T10c kappa_apply extrapolates with the edge bins",
          abs(kh[0] - lo) < 1e-12 and abs(kh[1] - hi) < 1e-12)
    tie = pd.DataFrame({"A_skel_um2": [1.0] * 12, "A_mesh_um2": [1.7] * 12,
                        "measured": True})
    tt = SAF.kappa_function(tie)
    check("T10d REGRESSION: tied A_skel still gives a non-empty table",
          len(tt) == 1 and abs(tt["kappa"].iloc[0] - 1.7) < 1e-12, str(len(tt)))


# --------------------------------------------------------------------------- #
# End to end, offline                                                          #
# --------------------------------------------------------------------------- #
CELL = 424242
ORIGIN = np.array([100000.0, 200000.0, 60000.0])


def phantom_nodes():
    rows = []
    for i in range(13):
        rows.append(dict(id=i, p=i - 1 if i else -1, x=ORIGIN[0] + 250.0 * i,
                         y=ORIGIN[1], z=ORIGIN[2], r=300.0,
                         annotated_type="dendrite"))
    ys = [380.0, 480.0, 580.0, 680.0, 780.0, 900.0, 1050.0, 1200.0]
    par = 6
    for j, yy in enumerate(ys):
        rows.append(dict(id=100 + j, p=par, x=ORIGIN[0] + 1500.0,
                         y=ORIGIN[1] + yy, z=ORIGIN[2],
                         r=70.0 if yy < 850 else 250.0, annotated_type="spine"))
        par = 100 + j
    df = pd.DataFrame(rows)
    comp = np.where(df["id"] >= 100, 0, -1).astype(np.int64)
    return df, comp


def stub_factory():
    def factory(cloudpath, mip=0, **kw):
        info = {"cloudpath": cloudpath, "mip": 0, "resolution_nm": list(RES),
                "dtype": "uint64", "bounds_vox": [[0, 0, 0], [10 ** 7] * 3],
                "available_mips": [0]}

        def reader(lo, hi):
            lo, hi = np.asarray(lo), np.asarray(hi)
            X, Y, Z = grid(tuple(int(v) for v in hi - lo), lo)
            x, y, z = X - ORIGIN[0], Y - ORIGIN[1], Z - ORIGIN[2]
            shaft = (y ** 2 + z ** 2 <= 300.0 ** 2) & (x >= 0) & (x <= 3000.0)
            neck = ((x - 1500.0) ** 2 + z ** 2 <= 70.0 ** 2) & (y >= 0) \
                & (y <= 850.0)
            head = (x - 1500.0) ** 2 + (y - 1050.0) ** 2 + z ** 2 <= 250.0 ** 2
            a = np.zeros(X.shape, dtype=np.uint64)
            a[shaft | neck | head] = CELL
            return a
        return reader, info
    return factory


def test_T11_end_to_end_and_resume():
    nodes, comp = phantom_nodes()
    tmp = tempfile.mkdtemp()
    try:
        fac = SAF.memoized_reader_factory(stub_factory())
        roi_fn = lambda sid: SAF.get_spine_roi(nodes, comp, sid, CELL,
                                               roi_dir=tmp, pad_nm=500.0,
                                               reader_factory=fac)
        led = os.path.join(tmp, "ledger.npz")
        recs, H = SAF.measure_all_spines([0], roi_fn, const_lookup(1.0), led,
                                         cell_id=CELL, verbose=False)
        r = recs[0]
        check("T11a spine measured", r["ok"], r.get("error", ""))
        if not r["ok"]:
            return
        check("T11b on the cell's own segid", r["on_cell_segid"])
        check("T11c no box / unresolved", r["frac_box"] + r["frac_unresolved"]
              < 0.005, "%.4f" % (r["frac_box"] + r["frac_unresolved"]))
        check("T11d a real cut face exists and is excluded (3-40%)",
              0.03 <= r["frac_cut"] <= 0.40, "%.3f" % r["frac_cut"])
        check("T11e counted area plausible for neck+head+rind (0.9-1.6 um2)",
              0.9 <= r["A_mesh_raw_um2"] <= 1.6, "%.3f" % r["A_mesh_raw_um2"])
        check("T11f one CloudVolume per layer for the run",
              len(fac.cache) == 1)
        before = dict(r)
        recs2, H2 = SAF.measure_all_spines([0], roi_fn, const_lookup(1.0), led,
                                           cell_id=CELL, verbose=False)
        check("T11g resume skips done spines and returns them unchanged",
              recs2[0]["A_mesh_um2"] == before["A_mesh_um2"]
              and np.allclose(H2[0], H[0], rtol=1e-6))
        recs3, _ = SAF.measure_all_spines([0, 7], roi_fn, const_lookup(1.0),
                                          led, cell_id=CELL, verbose=False)
        check("T11h an invalid sigma is recorded as a failure, not a crash",
              (not recs3[7]["ok"]) and recs3[7]["stage"] == "roi")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def figure_roi():
    """Spine cylinder + bridge + touching shaft slab + a detached blob."""
    a, gap, L = 110.0, 160.0, 650.0
    shape = (80, 170, 26)
    cx, cz, y0 = 321.0, 430.0, 330.0
    S = y_cylinder(a, y0 + gap, y0 + gap + L, shape, cx, cz)
    Bm = y_cylinder(a, y0, y0 + gap, shape, cx, cz)
    X, Y, Z = grid(shape)
    slab = (Y < y0) & (Y > y0 - 250.0) & (np.abs(Z - cz) < 330.0)
    blob = (X - 560.0) ** 2 + (Y - 1050.0) ** 2 + (Z - 430.0) ** 2 <= 60.0 ** 2
    return roi_dict(S, ctx=slab | blob, bridge=Bm), (cx, cz, y0, gap, L)


def test_T12_figure():
    try:
        import h01_spine_area_F_figures as FIG
    except ImportError as exc:
        return skip("test_T12_figure", "%s -- figures are not part of the HPC bundle" % exc)
    roi, (cx, cz, y0, gap, L) = figure_roi()
    rec, H, det = SAF.measure_spine_area(roi, const_lookup(1.0),
                                         dict(B.DEFAULTS), return_detail=True)
    rec["sigma_id"] = 7
    Lv = FIG.label_volume(roi)
    present = set(np.unique(Lv).tolist())
    check("T12a all four classes present in the label volume",
          {FIG.CODE_DETACHED, FIG.CODE_SHAFT, FIG.CODE_SPINE,
           FIG.CODE_BRIDGE} <= present, str(sorted(present)))
    nodes = pd.DataFrame({"x": [cx] * 4, "z": [cz] * 4,
                          "y": y0 + gap + np.array([100., 250., 400., 550.])})
    base = pd.DataFrame({"x": [cx], "y": [y0 - 120.0], "z": [cz]})
    fig, info = FIG.spine_planes_figure(roi, det, rec, nodes, base,
                                        margin_nm=300.0)
    check("T12b three panels", len(fig.axes) == 3)
    # extent convention: the node (cx, y) must fall on a SPINE pixel in XY
    ax = fig.axes[0]
    im = ax.get_images()[0]
    arr = np.asarray(im.get_array())
    x0, x1, ya, yb = im.get_extent()
    px = int((cx / 1000.0 - x0) / (x1 - x0) * arr.shape[1])
    py = int((nodes["y"].iloc[1] / 1000.0 - ya) / (yb - ya) * arr.shape[0])
    check("T12c node on the spine axis lands on a blue (spine) pixel",
          int(arr[py, px]) == FIG.CODE_SPINE, "code %s" % arr[py, px])
    seam = info.get("cut_to_seam_median_nm", np.nan)
    res = np.asarray(RES)
    cut = np.asarray(det["centroid_nm"])[det["cls"] == SAF.CLASS_CUT]
    unshifted = FIG._seam_distance(Lv, np.zeros(3, int), res, cut)
    check("T12d cut crosses ON the seam: perpendicular offset < 2 nm "
          "(quarter in-plane voxel)", seam < 2.0, "%.2f nm" % seam)
    check("T12e without the +res/2 shift they sit ~half a voxel off (> 3 nm)",
          unshifted > 3.0 and unshifted - seam > 2.0,
          "%.2f vs %.2f nm" % (seam, unshifted))
    tmp = tempfile.mkdtemp()
    try:
        path = FIG.save_figure(fig, os.path.join(tmp, "s.png"))
        check("T12f PNG written", os.path.getsize(path) > 10000,
              "%d B" % os.path.getsize(path))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_T13_figure_callback():
    try:
        import h01_spine_area_F_figures as FIG
    except ImportError as exc:
        return skip("test_T13_figure_callback", "%s -- figures are not part of the HPC bundle" % exc)
    nodes, comp = phantom_nodes()
    tmp = tempfile.mkdtemp()
    try:
        fac = SAF.memoized_reader_factory(stub_factory())
        roi_fn = lambda sid: SAF.get_spine_roi(nodes, comp, sid, CELL,
                                               roi_dir=tmp, pad_nm=500.0,
                                               reader_factory=fac)

        draw = FIG.make_figure_callback(nodes, comp, os.path.join(tmp, "figs"))
        recs, _ = SAF.measure_all_spines([0], roi_fn, const_lookup(1.0),
                                         os.path.join(tmp, "l1.npz"),
                                         cell_id=CELL, verbose=False,
                                         on_success=draw)
        r = recs[0]
        check("T13a figure written and recorded", r["ok"] and
              os.path.isfile(r.get("figure", "")), r.get("figure_error", ""))
        check("T13b oblique Voronoi seam: crosses within 2 nm of it",
              r.get("cut_to_seam_median_nm", 99.0) < 2.0,
              "%.2f nm" % r.get("cut_to_seam_median_nm", np.nan))

        def boom(*_):
            raise ValueError("plotting broke")
        recs2, _ = SAF.measure_all_spines([0], roi_fn, const_lookup(1.0),
                                          os.path.join(tmp, "l2.npz"),
                                          cell_id=CELL, verbose=False,
                                          on_success=boom)
        check("T13c failing callback: measurement still ok, error recorded",
              recs2[0]["ok"] and "plotting broke" in recs2[0].get("figure_error", ""))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_T15_junction_terms():
    """Rind on an ISOLATED z-cylinder, radius a, its own axis, sweeping r_nm.
    r_nm = 0.6a  -> the two end discs inside that radius: 2 pi (0.6a)^2.
    r_nm > a     -> every counted triangle: the whole mesh area.
    Both are closed form, so this tests the rho computation, not the code."""
    a, L = 200.0, 800.0
    shape = (80, 80, 50)
    cx, cy, z0 = 321.0, 318.0, 400.0
    X, Y, Z = grid(shape)
    m = ((X - cx) ** 2 + (Y - cy) ** 2 <= a * a) & (Z >= z0) & (Z < z0 + L)
    mid = np.array([cx, cy, z0 + 0.5 * L])
    ax = lambda r, p=mid: {"point_nm": p, "dir": np.array([0., 0., 1.]), "r_nm": r}

    rec, _ = SAF.measure_spine_area(roi_dict(m), const_lookup(1.0),
                                    dict(B.DEFAULTS), shaft_axis=ax(0.6 * a))
    ratio = rec["A_rind_um2"] * 1e6 / (2 * np.pi * (0.6 * a) ** 2)
    check("T15a rind at r=0.6a equals the two end discs (ratio 0.85-1.15)",
          0.85 <= ratio <= 1.15, "%.3f" % ratio)
    rec2, _ = SAF.measure_spine_area(roi_dict(m), const_lookup(1.0),
                                     dict(B.DEFAULTS), shaft_axis=ax(2.0 * a))
    check("T15b rind at r=2a swallows the whole mesh",
          abs(rec2["A_rind_um2"] - rec2["A_mesh_um2"]) < 1e-12
          and abs(rec2["A_mesh_norind_um2"]) < 1e-12)
    check("T15c frac_rind and the bookkeeping identity hold",
          abs(rec["A_mesh_norind_um2"] + rec["A_rind_um2"] - rec["A_mesh_um2"]) < 1e-12
          and abs(rec["frac_rind"] - rec["A_rind_um2"] / rec["A_mesh_um2"]) < 1e-9)
    # axial window: axis point at the LOW cap, window 0.5 L keeps that cap only
    low = np.array([cx, cy, z0])
    rec3, _ = SAF.measure_spine_area(roi_dict(m), const_lookup(1.0),
                                     dict(B.DEFAULTS), shaft_axis=ax(0.6 * a, low),
                                     axial_window_nm=0.5 * L)
    half = rec["A_rind_um2"] / 2.0
    check("T15d axial window keeps one disc of the two",
          abs(rec3["A_rind_um2"] / half - 1.0) < 0.15,
          "%.4f vs %.4f um2" % (rec3["A_rind_um2"], half))
    rec4, _ = SAF.measure_spine_area(roi_dict(m), const_lookup(1.0), dict(B.DEFAULTS))
    check("T15e no axis -> rind NaN, not zero",
          np.isnan(rec4["A_rind_um2"]) and np.isnan(rec4["A_mesh_norind_um2"]))

    # ---- skeleton side: base frustum, closed form on the toy cell
    sd = FakeSD()
    df, comp = toy_cell()
    sk = SAF.skeleton_spine_table(sd, df, df, comp).set_index("sigma_id")
    r1, r2, Lb = 0.440, 0.120, 0.400            # base r, root r, segment (um)
    want = np.pi * (r1 + r2) * np.sqrt((r1 - r2) ** 2 + Lb ** 2)
    check("T15f base frustum = pi (r_base + r_root) sqrt(dr^2 + L^2)",
          abs(sk.loc[1, "A_skel_base_um2"] - want) < 1e-9,
          "%.6f vs %.6f" % (sk.loc[1, "A_skel_base_um2"], want))
    check("T15g A_skel_nobase = A_skel - base, and is a large share here",
          abs(sk.loc[1, "A_skel_nobase_um2"]
              - (sk.loc[1, "A_skel_um2"] - want)) < 1e-12
          and want / sk.loc[1, "A_skel_um2"] > 0.3,
          "base is %.1f%% of A_skel" % (100 * want / sk.loc[1, "A_skel_um2"]))
    sk = sk.reset_index()
    axes = SAF.shaft_axes_from_table(sk)
    check("T15h REGRESSION: no axis off a soma base (r would be the soma "
          "radius, and the whole spine would read as rind)",
          set(axes) == {1, 2, 3} and np.isnan(
              sk.set_index("sigma_id").loc[0, "r_shaft_nm"]), str(sorted(axes)))
    u = axes[1]["dir"]
    check("T15i axis is the unit vector along the attributed segment (+x)",
          abs(np.linalg.norm(u) - 1) < 1e-12 and abs(u[0] - 1.0) < 1e-12, str(u))

    # ---- kappa variants: plant A_mesh = 2 A_skel with a quarter as rind
    recs = {}
    for _, r in sk.iterrows():
        sid = int(r["sigma_id"])
        am = 2.0 * r["A_skel_um2"]
        recs[sid] = {"sigma_id": sid, "ok": True, "A_mesh_um2": am,
                     "A_mesh_raw_um2": am, "A_rind_um2": 0.25 * am,
                     "A_mesh_norind_um2": 0.75 * am, "frac_rind": 0.25,
                     "clipped": False, "frac_cut": .1, "frac_bridge": 0.,
                     "frac_box": 0., "frac_unresolved": 0., "stage": "done"}
    recs[0]["A_rind_um2"] = np.nan          # the axis-less spine
    recs[0]["A_mesh_norind_um2"] = np.nan
    out = SAF.assemble_cell(sd, df, df, comp, recs, 1, sk=sk, min_per_bin=1)
    j = out["summary"]
    check("T15j kappa_norind = 0.75 * 2 (axis-less spine excluded)",
          abs(j["kappa_pooled_norind"] - 1.5) < 1e-9, "%.6f" % j["kappa_pooled_norind"])
    check("T15k kappa_nobase > kappa_pooled (removing the base lifts it)",
          j["kappa_pooled_nobase"] > j["kappa_pooled"] > 0,
          "%.4f vs %.4f" % (j["kappa_pooled_nobase"], j["kappa_pooled"]))
    check("T15l kappa_both = 0.75 * kappa_nobase on this planted set",
          abs(j["kappa_pooled_both"] - 0.75 * (
              mnr := float(sk.set_index("sigma_id").loc[[1, 2, 3], "A_skel_um2"].sum()
                           * 2.0 / sk.set_index("sigma_id").loc[
                               [1, 2, 3], "A_skel_nobase_um2"].sum()))) < 1e-9,
          "%.4f vs %.4f" % (j["kappa_pooled_both"], 0.75 * mnr))
    check("T15m one spine had no shaft axis and is counted as such",
          j["n_no_shaft_axis"] == 1, str(j["n_no_shaft_axis"]))
    check("T15n F_lit mesh_norind sits below F_lit mesh",
          j["F_lit_mesh_norind"] < j["F_lit_mesh"],
          "%.4f vs %.4f" % (j["F_lit_mesh_norind"], j["F_lit_mesh"]))


def test_T14_min_length_filter():
    sd = FakeSD()
    df, comp = toy_cell()
    df.loc[df["id"] == 9, "annotated_type"] = "apical"      # base of sigma 3
    sk = SAF.skeleton_spine_table(sd, df, df, comp).set_index("sigma_id")
    exp = {1: (800.0, 1200.0, 1200.0 - 440.0), 2: (1200.0, 1600.0, 1600.0 - 360.0),
           3: (400.0, 800.0, 800.0 - 320.0)}
    ok = all(abs(sk.loc[k, "L_skel_nm"] - v[0]) < 1e-6
             and abs(sk.loc[k, "tip_dist_nm"] - v[1]) < 1e-6
             and abs(sk.loc[k, "protrusion_nm"] - v[2]) < 1e-6 for k, v in exp.items())
    check("T14a L_skel / tip distance / protrusion exact", ok,
          sk[["L_skel_nm", "tip_dist_nm", "protrusion_nm"]].to_string())
    check("T14a' protrusion undefined (NaN) for the spine on the soma root",
          np.isnan(sk.loc[0, "protrusion_nm"]))
    sk = sk.reset_index()
    rep = SAF.threshold_report(sk, "protrusion_nm", (500.0, 1000.0))
    check("T14b threshold report (500 -> 1, 1000 -> 2; soma spine kept, 1 NaN)",
          rep["n_removed"].tolist() == [1, 2] and rep.attrs["n_metric_nan"] == 1,
          "%s nan=%s" % (rep["n_removed"].tolist(), rep.attrs["n_metric_nan"]))
    check("T14b' the NaN spine is never selected for removal",
          0 not in SAF.short_spine_ids(sk, "protrusion_nm", 1e9))
    check("T14c no threshold -> nothing selected",
          SAF.short_spine_ids(sk, "protrusion_nm", None) == [])
    short = [3]
    lab2, nod2, comp2, prov = SAF.demote_spines(df, df, comp, sk, short,
                                                 sd.SHAFT_REGEX)
    check("T14d sigma ids stable: demoted -> -1, others untouched",
          np.array_equal(comp2[comp != 3], comp[comp != 3])
          and np.all(comp2[comp == 3] == -1))
    check("T14e demoted nodes inherit the base's label ('apical')",
          set(lab2.loc[comp == 3, "annotated_type"]) == {"apical"})
    phi0, phi2 = sd.build_phi(df, nid=1), sd.build_phi(lab2, nid=1)
    sk2 = SAF.skeleton_spine_table(sd, lab2, nod2, comp2)
    g2 = SAF.attribution_gate(phi2, sk2)
    check("T14f gate passes on the demoted partition", g2["pass"], str(g2))
    a3 = float(sk.loc[sk["sigma_id"] == 3, "A_skel_um2"].iloc[0])
    check("T14g exactly A_skel(sigma 3) leaves the spine column",
          abs(phi0["spine_area_um2"].sum() - phi2["spine_area_um2"].sum() - a3) < 1e-9)
    check("T14h its frustum area joins the shaft column",
          abs(phi2["shaft_area_um2"].sum() - phi0["shaft_area_um2"].sum() - a3) < 1e-9
          and len(phi2) == len(phi0) + prov["n_nodes_demoted"])
    check("T14i F_lit drops when a distal spine is demoted",
          SAF.cell_F(phi2)["F"] < SAF.cell_F(phi0)["F"])
    try:
        SAF.demote_spines(df, df, comp, sk, short, sd.SHAFT_REGEX,
                          fallback_label="spine")
        check("T14j non-shaft fallback label refused", False, "accepted")
    except SAF.SpineAreaError:
        check("T14j non-shaft fallback label refused", True)


def main():
    for fn in (test_T1_frame_convention, test_T2_isolated_sphere,
               test_T3_cut_face, test_T4_bridge, test_T5_cavity, test_T6_box,
               test_T7_calibration_arithmetic, test_T8_table_gates,
               test_T9_skeleton_and_F, test_T10_kappa_function,
               test_T11_end_to_end_and_resume, test_T12_figure,
               test_T13_figure_callback, test_T14_min_length_filter,
               test_T15_junction_terms):
        _run(fn)
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("\n%d checks passed, %d failed, %d test(s) skipped"
          % (len(RESULTS) - n_fail, n_fail, len(SKIPPED)))
    print("ALL GREEN" if n_fail == 0 else "FAILURES")
    return n_fail


if __name__ == "__main__":
    sys.exit(main())
