#!/usr/bin/env python3
"""Smoke test for p1_plots.py (auto-discovered by run_smoke_tests.sh).

Builds a real P1 export in a temp campaign root with the fixtures of
smoke_test_p1_export (gate stubbed, no NEURON needed), then checks every
loader and transform against numbers known from the fixture, renders every
figure to PNG on the Agg backend, and exercises the LFPy path through a fake
cell so the compartment plotting is tested where LFPy is absent.

Run:  cd h01_code && python3 smoke_test_p1_plots.py
"""
from __future__ import print_function

import importlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE1 = os.path.join(HERE, "stage1")
sys.path.insert(0, HERE)
sys.path.insert(0, STAGE1)
os.environ.pop("DISPLAY", None)
import matplotlib                                   # noqa: E402
matplotlib.use("Agg")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %s%s" % ("ok" if ok else "FAIL", name, ("  -- " + str(detail)[:200]) if (detail and not ok) else ""))


class FakeCell(object):
    """Minimal stand-in for an LFPy cell: a straight 100 um dendrite cut into
    `n` compartments along +x, an axon compartment, diameters as given."""

    def __init__(self, n=10, two_d=True):
        edges = np.linspace(0.0, 100.0, n + 1)
        x0 = np.concatenate([[0.0], edges[:-1]])
        x1 = np.concatenate([[-5.0], edges[1:]])
        self.totnsegs = n + 1
        self._names = ["axon[0]"] + ["dend[0]"] * n
        d = np.concatenate([[0.3], np.full(n, 0.6)])
        if two_d:
            self.x = np.stack([x0, x1], axis=1)
            self.y = np.zeros((n + 1, 2))
            self.z = np.zeros((n + 1, 2))
        else:
            self.xstart, self.xend = x0, x1
            self.ystart = self.yend = np.zeros(n + 1)
            self.zstart = self.zend = np.zeros(n + 1)
        self.d = d

    def get_idx_name(self, i):
        return (i, self._names[i], 0.5)


def main():
    T = importlib.import_module("smoke_test_p1_export")
    R = importlib.import_module("run_p1_export")
    B = importlib.import_module("build_p1_manifest")
    P = importlib.import_module("p1_plots")
    import hoc_qc as hq
    tmp = tempfile.mkdtemp(prefix="p1plots_")
    try:
        # ---- a real export of the fixture cell ------------------------------ #
        root = T.campaign_tree(tmp, cell_ids=(T.CELL,))
        ptab = T.passive_csv(tmp)
        mp = os.path.join(root, "p1", "manifests", "L3_exc.csv")
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        B.build_manifest(root, "L3", "exc", [T.CELL]).to_csv(mp, index=False)
        hq.gate_hoc = T.gate_stub_factory("pass")
        rc, out = T.run_driver(R, ["--root", root, "--manifest", mp, "--stage1-dir", STAGE1,
                                   "--passive-table", ptab, "--no-neuron-validate", "--task", "0"])
        check("F0 fixture export ran", rc == 0, out[-300:])
        rc, out = T.run_driver(R, ["--root", root, "--summarise", "--manifest", mp])
        check("F0' fixture summarised", rc == 0, out[-300:])
        out_dir = os.path.join(root, "p1")
        cid = T.CELL

        # ---- loaders ---------------------------------------------------------- #
        raw = P.load_raw_skeleton(os.path.join(root, "neurons"), cid)
        rec = P.load_record(out_dir, cid)
        check("L0 the record carries the segmentation the downsampling figure needs",
              set(["cm", "Ra", "lambda_f", "d_lambda"]) <= set(rec["result"]["segmentation"]))
        alj = P.load_alignment(out_dir, cid)
        sn = P.load_spine_nodes(out_dir, cid)
        st = P.load_spine_stats(out_dir, cid)
        syn = P.load_mapped_synapses(out_dir, cid)
        phi = P.load_phi(out_dir, cid)
        check("L1 loaders: 109 raw nodes, 7 spine nodes, 2 spines, 3 synapses",
              len(raw) == 109 and len(sn) == 7 and len(st) == 2 and len(syn) == 3,
              (len(raw), len(sn), len(st), len(syn)))
        check("L2 a missing artefact is a SystemExit naming the path",
              T._refused(lambda: P.load_record(out_dir, 999)) is not None)
        check("L3 no synapse file -> empty frame with the plotting columns",
              set(["x", "y", "z", "synapse_type", "lfpy_idx"]) <= set(
                  P.load_mapped_synapses(tmp, 1).columns))

        # ---- transforms ------------------------------------------------------- #
        ra = P.align_raw(raw, alj)
        soma = ra[ra["id"] == 0].iloc[0]
        check("T1 align_raw keeps every node, puts the soma at the origin, nm -> um",
              len(ra) == len(raw) and abs(soma["x_al_um"]) < 1e-9
              and abs(ra.loc[ra["id"] == 100, "x_al_um"].iloc[0] - 100.0) < 1e-9)
        bad = json.loads(json.dumps(alj))
        bad["alignment"]["mean_matrix"] = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        try:
            P.align_raw(raw, bad); t2 = False
        except ValueError:
            t2 = True
        check("T2 a mean_matrix with det != 1 is refused", t2)
        rot = json.loads(json.dumps(alj))
        rot["alignment"]["mean_matrix"] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]   # +x -> +y
        rr = P.align_raw(raw, rot)
        check("T3 the rotation is applied as centred @ R^T (x axis lands on +y)",
              abs(rr.loc[rr["id"] == 100, "y_al_um"].iloc[0] - 100.0) < 1e-9
              and abs(rr.loc[rr["id"] == 100, "x_al_um"].iloc[0]) < 1e-9)
        kept, pruned = P.split_pruned(ra, sn)
        check("T4 split_pruned: kept + pruned == raw, pruned ids == spine node ids",
              len(kept) + len(pruned) == len(ra)
              and set(pruned["id"]) == set(sn["node_id"]) and len(pruned) == 7)
        segs, child = P.frame_segments(ra)
        check("T5 frame_segments: one segment per non-root node, (N,2,3)",
              segs.shape == (108, 2, 3) and len(child) == 108)
        segs_p, _ = P.frame_segments(pruned)
        check("T5' segments of the pruned set alone exclude the two base links",
              segs_p.shape[0] == 5)
        pt = P.parse_hoc_pt3d(P._artefact(out_dir, cid, "aligned.hoc"))
        check("H1 parse_hoc_pt3d: dend[0] has 101 pt3d, soma 2, axon 2",
              (pt["array"] == "dend").sum() == 101 and (pt["array"] == "soma").sum() == 2
              and (pt["array"] == "axon").sum() == 2, pt["array"].value_counts().to_dict())
        c2 = P.compartments_from_cell(FakeCell(10, True))
        c1 = P.compartments_from_cell(FakeCell(10, False))
        check("C1 compartments_from_cell: LFPy>=2.2 and legacy attribute layouts agree",
              len(c2) == 11 and c2[["x0", "x1", "d"]].equals(c1[["x0", "x1", "d"]])
              and c2["sec"].iloc[0] == "axon[0]")
        mids = P.compartment_midpoints(c2)
        check("C2 midpoints indexed by compartment idx", abs(mids.loc[1, "x"] - 5.0) < 1e-9
              and abs(mids.loc[10, "x"] - 95.0) < 1e-9)
        syn2 = syn.copy()
        syn2["lfpy_idx"] = [8, 9, 6]                  # 70 um -> comp 8 (70-80), etc.
        conn, used = P.synapse_connectors(syn2, mids)
        on = syn2["on_pruned_spine"].astype(bool).to_numpy()
        check("C3 connectors start at the ANCHOR for on-pruned-spine synapses, at the synapse otherwise",
              conn.shape == (3, 2, 3)
              and np.allclose(conn[on, 0, :], syn2.loc[on, ["anchor_x", "anchor_y", "anchor_z"]].to_numpy(float))
              and np.allclose(conn[~on, 0, :], syn2.loc[~on, ["x", "y", "z"]].to_numpy(float)))
        check("C3' connectors end at the midpoint of lfpy_idx",
              np.allclose(conn[:, 1, 0], [75.0, 85.0, 55.0]))
        syn3 = syn2.copy(); syn3.loc[0, "lfpy_idx"] = 999
        conn3, used3 = P.synapse_connectors(syn3, mids)
        check("C3'' a synapse whose lfpy_idx is not in the cell is dropped, not crashed",
              conn3.shape[0] == 2 and len(used3) == 2)
        dens = P.spine_density_profile(st, phi, 20.0)
        check("D1 density profile: spines sum to the table, cable sums to phi, density = n/cable",
              int(dens["n_spines"].sum()) == len(st)
              and abs(dens["cable_um"].sum() - phi["seg_len_um"].sum()) < 1e-9
              and np.isfinite(dens["spines_per_um"]).any()
              and abs(dens.loc[3, "spines_per_um"] - 1.0 / 20.0) < 1e-9, dens.to_dict("records"))
        prof = P.arbour_profile(phi, 20.0)
        check("D2 arbour profile conserves shaft and spine area and cable",
              abs(prof["A_shaft_um2"].sum() - phi["shaft_area_um2"].sum()) < 1e-9
              and abs(prof["A_spine_um2"].sum() - phi["spine_area_um2"].sum()) < 1e-9
              and abs(prof["cable_um"].sum() - phi["seg_len_um"].sum()) < 1e-9)
        check("D2' length-weighted diameter of a uniform 0.6 um dendrite is 0.6 beyond the soma bin",
              abs(prof.loc[1, "shaft_diam_um"] - 0.6) < 1e-9)
        # ---- which phi the arbour figure draws (mesh vs skeleton fallback) --- #
        check("Y0 resolve_phi defaults to mesh: the deliverable, or a refusal",
              T._refused(lambda: P.resolve_phi(root, out_dir, cid)) is not None)
        ph, lab, is_mesh = P.resolve_phi(root, out_dir, cid, "auto")
        check("M1 auto falls back to P1's SKELETON phi when no P3 phi_mesh exists",
              is_mesh is False and "SKELETON FALLBACK" in lab and len(ph) == len(phi))
        check("M2 --phi mesh is REFUSED rather than silently falling back",
              T._refused(lambda: P.resolve_phi(root, out_dir, cid, "mesh")) is not None)
        check("M2' --phi takes only auto|mesh|skel",
              T._refused(lambda: P.resolve_phi(root, out_dir, cid, "nonsense")) is not None)
        # a P3-style deliverable phi: same rows, same SHAFT areas, spine column replaced
        mesh_dir = os.path.join(root, "out")
        os.makedirs(mesh_dir, exist_ok=True)
        pm = phi.copy()
        pm["spine_area_skel_um2"] = phi["spine_area_um2"].to_numpy(float)
        pm["spine_area_um2"] = phi["spine_area_um2"].to_numpy(float) * 3.0
        pm.to_csv(os.path.join(mesh_dir, "neuron_%d_phi_mesh.csv" % cid), index=False)
        ph2, lab2, is_mesh2 = P.resolve_phi(root, out_dir, cid, "auto")
        check("M3 auto prefers the P3 mesh table once it exists, and says so",
              is_mesh2 is True and "mesh" in lab2
              and abs(ph2["spine_area_um2"].sum() - 3.0 * phi["spine_area_um2"].sum()) < 1e-9)
        check("M3' the shaft column is the SAME in both -- shaft area never comes from the mesh",
              abs(ph2["shaft_area_um2"].sum() - phi["shaft_area_um2"].sum()) < 1e-12)
        check("M4 --phi skel still forces P1's table with the mesh one present",
              P.resolve_phi(root, out_dir, cid, "skel")[2] is False)
        bad = phi.copy()
        bad.to_csv(os.path.join(mesh_dir, "neuron_%d_phi_mesh.csv" % cid), index=False)
        check("M5 a phi_mesh without spine_area_skel_um2 is refused, not drawn as a mesh",
              T._refused(lambda: P.resolve_phi(root, out_dir, cid, "auto")) is not None)
        pm.to_csv(os.path.join(mesh_dir, "neuron_%d_phi_mesh.csv" % cid), index=False)
        f_sk = P.F_beyond(phi, STAGE1)
        f_me = P.F_beyond(ph2, STAGE1)
        check("M6 F is computed from the phi actually drawn (mesh F > skeleton F here)",
              f_sk is not None and f_me is not None and f_me > f_sk
              and abs(f_sk - float(rec["result"]["F_lit"])) < 1e-9,
              (f_sk, f_me, rec["result"]["F_lit"]))
        fdir0 = os.path.join(tmp, "figs_phi")
        o_mesh = P.cell_figures(root, out_dir, cid, fdir0, skip_lfpy=True, phi_mode="mesh")
        o_skel = P.cell_figures(root, out_dir, cid, fdir0, skip_lfpy=True, phi_mode="skel")
        check("M7 the two arbour figures are written under DIFFERENT names",
              o_mesh["arbour"].endswith("_arbour_mesh.png")
              and o_skel["arbour"].endswith("_arbour_skel.png")
              and os.path.getsize(o_mesh["arbour"]) > 5000
              and os.path.getsize(o_skel["arbour"]) > 5000)
        shutil.rmtree(mesh_dir, ignore_errors=True)
        import io as _io, contextlib as _ctx
        _b = _io.StringIO()
        with _ctx.redirect_stdout(_b):
            o_def = P.cell_figures(root, out_dir, cid, os.path.join(tmp, "figs_def"),
                                   skip_lfpy=True)
        check("M8 with the mesh gone, the DEFAULT skips the arbour figure and says why "
              "-- it never falls back to the skeleton silently",
              "arbour" not in o_def and "skipped" in _b.getvalue()
              and "mesh_beyond" in _b.getvalue()
              and set(o_def) == {"pruning", "spines"}, _b.getvalue()[-200:])

        # ---- style and 3-D ---------------------------------------------------- #
        st = P.resolve_style()
        st2 = P.resolve_style(lw_scale=2.0, syn_size=1.5, elev=45.0, azim=10.0)
        check("Y1 lw_scale multiplies every line width and leaves marker sizes alone",
              abs(st2["lw_shaft"] - 2.0 * st["lw_shaft"]) < 1e-12
              and abs(st2["lw_spine"] - 2.0 * st["lw_spine"]) < 1e-12
              and abs(st2["comp_lw_max"] - 2.0 * st["comp_lw_max"]) < 1e-12
              and st2["syn_size"] == 1.5 and st2["syn_spine_ratio"] == st["syn_spine_ratio"])
        check("Y1' an unset override does not overwrite the default",
              P.resolve_style(syn_size=None)["syn_size"] == st["syn_size"]
              and st2["elev"] == 45.0 and st2["azim"] == 10.0)
        check("Y2 the default marks are thicker skeleton, smaller synapses than v1.0",
              st["lw_shaft"] > 0.6 and st["syn_size"] ** 2 < 9.0)
        fdir3 = os.path.join(tmp, "figs3d")
        o3 = P.cell_figures(root, out_dir, cid, fdir3, proj="3d", skip_lfpy=True,
                            phi_mode="skel", style=st2)
        check("Y3 a 3-D pruning figure is written under its own name",
              o3["pruning"].endswith("_pruning_3d.png")
              and os.path.getsize(o3["pruning"]) > 5000)
        import matplotlib.pyplot as _plt
        f3 = _plt.figure()
        ax3 = P._panels(f3, 2, "3d")
        check("Y4 _panels gives 3-D axes for 3d and 2-D axes otherwise",
              hasattr(ax3[0], "get_zlim") and len(ax3) == 2
              and not hasattr(P._panels(_plt.figure(), 2, "xz")[0], "get_zlim"))
        segs = np.array([[[0., 0., 0.], [1., 2., 3.]]])
        P._lc(ax3[0], segs, "3d", "#000000", 1.0)
        P._points(ax3[0], [[1., 2., 3.]], "3d", s=4.0)
        P._equalise(ax3, np.array([[0., 0., 0.], [10., 2., 4.]]), "3d", st)
        spans = [ax3[0].get_xlim(), ax3[0].get_ylim(), ax3[0].get_zlim()]
        widths = [hi - lo for lo, hi in spans]
        check("Y5 3-D limits are a cube of the largest range (no axis stretched)",
              max(widths) - min(widths) < 1e-9 and widths[0] > 10.0)
        P._lc(ax3[0], np.zeros((0, 2, 3)), "3d", "#000000", 1.0)
        check("Y6 an empty segment array is a no-op in both projections",
              P._points(ax3[0], np.zeros((0, 3)), "3d", s=1.0) is None)
        _plt.close("all")
        rc = P.main(["--root", root, "--out-dir", out_dir, "--fig-dir", fdir3,
                     "--cell", str(cid), "--proj", "3d", "--phi", "skel",
                     "--skip-lfpy", "--lw-scale", "1.5", "--syn-size", "2",
                     "--elev", "25", "--azim", "-60", "--dpi", "120"])
        check("Y7 the CLI accepts the 3-D and style knobs", rc == 0)

        s1 = P.load_summary(out_dir)
        s2 = s1.copy(); s2["totnsegs"] = s2["totnsegs"] * 1.3
        check("D3 pair_summaries joins on cell_id with _a/_b suffixes",
              set(["totnsegs_a", "totnsegs_b", "qc_status_a"]) <= set(P.pair_summaries(s1, s2).columns))

        # ---- figures ---------------------------------------------------------- #
        fdir = os.path.join(tmp, "figs")
        outp = P.cell_figures(root, out_dir, cid, fdir, proj="xz", skip_lfpy=True,
                              phi_mode="skel")
        check("G1 three per-cell figures written without LFPy",
              set(outp) == {"pruning", "spines", "arbour"}
              and all(os.path.getsize(v) > 5000 for v in outp.values()), outp)
        saved = P.build_lfpy_cell
        P.build_lfpy_cell = lambda hoc, seg, s1d=None: FakeCell(49, True)   # 49 + axon = 50 = stub totnsegs
        try:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                outp2 = P.cell_figures(root, out_dir, cid, fdir, proj="xz", phi_mode="skel")
            check("G2 downsampling figure written through the cell builder; totnsegs matches (no warning)",
                  "downsampling" in outp2 and os.path.getsize(outp2["downsampling"]) > 5000
                  and "WARNING" not in buf.getvalue(), buf.getvalue()[-300:])
            P.build_lfpy_cell = lambda hoc, seg, s1d=None: FakeCell(20, True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                P.cell_figures(root, out_dir, cid, fdir, proj="yz", phi_mode="skel")
            check("G3 a compartment count that disagrees with the record's totnsegs is WARNED",
                  "WARNING" in buf.getvalue() and "totnsegs" in buf.getvalue(), buf.getvalue()[-300:])

            def _raise(hoc, seg, s1d=None):
                raise ImportError("No module named 'LFPy'")
            P.build_lfpy_cell = _raise
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                outp3 = P.cell_figures(root, out_dir, cid, fdir, proj="xy", phi_mode="skel")
            check("G4 without LFPy the other figures are still written and the skip is announced",
                  "downsampling" not in outp3 and "skipped" in buf.getvalue()
                  and set(outp3) == {"pruning", "spines", "arbour"})
        finally:
            P.build_lfpy_cell = saved
        pth = P.population_figure(out_dir, fdir)
        check("G5 population figure from one summary", os.path.getsize(pth) > 5000)
        cmp_dir = os.path.join(tmp, "p1_other")
        os.makedirs(cmp_dir)
        s2.to_csv(os.path.join(cmp_dir, "p1_summary.csv"), index=False)
        pth2 = P.population_figure(out_dir, fdir, compare=cmp_dir)
        check("G6 paired population figure against a second tree", os.path.getsize(pth2) > 5000
              and pth2.endswith("p1_population_vs_p1_other.png"))
        rc = P.main(["--root", root, "--out-dir", out_dir, "--fig-dir", fdir, "--cell", str(cid),
                     "--population", "--skip-lfpy", "--phi", "skel"])
        check("G7 CLI exits 0", rc == 0)
        try:
            P.main(["--root", root, "--out-dir", out_dir, "--fig-dir", fdir])
            g8 = False
        except SystemExit:
            g8 = True
        check("G8 CLI refuses a call with neither --cell nor --population", g8)
        src = open(os.path.join(HERE, "p1_plots.py"), "rb").read()
        check("S1 p1_plots.py is pure ASCII, LF only",
              all(b < 128 for b in src) and b"\r" not in src)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n%d checks passed, %d failed" % (len(RESULTS) - n_fail, n_fail))
    print("ALL GREEN" if n_fail == 0 else "FAILURES")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
