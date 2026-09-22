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
        s1 = P.load_summary(out_dir)
        s2 = s1.copy(); s2["totnsegs"] = s2["totnsegs"] * 1.3
        check("D3 pair_summaries joins on cell_id with _a/_b suffixes",
              set(["totnsegs_a", "totnsegs_b", "qc_status_a"]) <= set(P.pair_summaries(s1, s2).columns))

        # ---- figures ---------------------------------------------------------- #
        fdir = os.path.join(tmp, "figs")
        outp = P.cell_figures(root, out_dir, cid, fdir, proj="xz", skip_lfpy=True)
        check("G1 three per-cell figures written without LFPy",
              set(outp) == {"pruning", "spines", "arbour"}
              and all(os.path.getsize(v) > 5000 for v in outp.values()), outp)
        saved = P.build_lfpy_cell
        P.build_lfpy_cell = lambda hoc, seg, s1d=None: FakeCell(49, True)   # 49 + axon = 50 = stub totnsegs
        try:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                outp2 = P.cell_figures(root, out_dir, cid, fdir, proj="xz")
            check("G2 downsampling figure written through the cell builder; totnsegs matches (no warning)",
                  "downsampling" in outp2 and os.path.getsize(outp2["downsampling"]) > 5000
                  and "WARNING" not in buf.getvalue(), buf.getvalue()[-300:])
            P.build_lfpy_cell = lambda hoc, seg, s1d=None: FakeCell(20, True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                P.cell_figures(root, out_dir, cid, fdir, proj="yz")
            check("G3 a compartment count that disagrees with the record's totnsegs is WARNED",
                  "WARNING" in buf.getvalue() and "totnsegs" in buf.getvalue(), buf.getvalue()[-300:])

            def _raise(hoc, seg, s1d=None):
                raise ImportError("No module named 'LFPy'")
            P.build_lfpy_cell = _raise
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                outp3 = P.cell_figures(root, out_dir, cid, fdir, proj="xy")
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
                     "--population", "--skip-lfpy"])
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
