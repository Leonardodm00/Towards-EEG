"""Smoke test for the staging + propagation gate in alignment.align_and_export.

Runs standalone (python3 test_align_gate.py) or under pytest. No NEURON: both
morphology_exporter.export_neuron and hoc_qc.gate_hoc are replaced by stubs, so
what is under test here is the ORCHESTRATION -- staging, the commit decision,
the cell lifetime and the removal of the df_labelled duplication -- and not the
exporter or the simulator, each of which has its own suite.

The claim that matters most, and the reason staging exists at all:

    a cell that does not conduct leaves output_dir byte-for-byte as it was.

Coverage
  T1  gate PASS: every staged artefact lands in output_dir, no staging dir
      survives, and the provenance JSON is repointed at the committed paths
  T2  gate FAIL: output_dir is untouched -- not merely empty, but identical to
      its pre-call listing -- staging is removed, and res['files'] is empty
  T3  gate LOW: files ARE committed and the verdict is downgraded to
      pass_low_confidence with a machine-readable reason
  T4  ONE cell: gate_hoc's cell is handed to snap_synapses, default_cell_factory
      is never called, and the cell is released exactly once
  T5  the duplication is gone: df_labelled comes from export_neuron's
      return_frames, the caller passes nothing, and res carries no DataFrame
  T6  an explicit df_labelled still overrides, and return_frames is not asked
  T7  a soma-level export failure short-circuits before the gate and commits
      nothing
  T8  qc_propagation=False restores the direct-write path and the legacy
      cell_factory
  T9  exporter_qc_status: the PRE-gate verdict is recorded, so a rigidity
      control (which never runs the gate) can be compared against it without
      a spurious "alignment moved a quantity" on every gate-downgraded cell
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

import alignment as al
import hoc_qc as hq
import morphology_exporter as mx


NID = 777


# --------------------------------------------------------------------------- #
#  Stubs                                                                       #
# --------------------------------------------------------------------------- #
class _StubCell(object):
    def __init__(self):
        self.totnsegs = 4
        self.released = 0

    def get_closest_idx(self, x, y, z):
        return 0

    def __del__(self):
        self.released += 1


def _labelled_frame():
    return pd.DataFrame({"id": [0, 1], "p": [-1, 0],
                         "x": [0.0, 1000.0], "y": [0.0, 0.0], "z": [0.0, 0.0],
                         "r": [5000.0, 500.0],
                         "compartment_class": ["soma", "dend"],
                         "annotated_type": ["Soma", "Dendrite"]})


def _make_export_stub(record, qc_status="pass"):
    """Writes four plausible artefacts into whatever directory it is handed."""

    def _stub(df_raw, nid, output_dir, **kwargs):
        record["export_dir"] = output_dir
        record["return_frames"] = bool(kwargs.get("return_frames", False))
        os.makedirs(output_dir, exist_ok=True)
        files = {}
        hoc = os.path.join(output_dir, "neuron_%s_aligned.hoc" % nid)
        with open(hoc, "w", newline="\n") as fh:
            fh.write("create soma[1]\n")
        files["hoc"] = hoc
        for name in ("phi", "section_table", "spine_bases"):
            p = os.path.join(output_dir, "neuron_%s_%s.csv" % (nid, name))
            pd.DataFrame({"section_id": [0], "array": ["soma"],
                          "type_idx": [0], "parent_sec_id": [-1]}).to_csv(
                p, index=False)
            files[name] = p
        prov = os.path.join(output_dir, "neuron_%s_provenance.json" % nid)
        with open(prov, "w", newline="\n") as fh:
            fh.write(json.dumps({"files": {k: v for k, v in files.items()},
                                 "record": {}}, indent=2))
        files["provenance"] = prov
        # an artefact the exporter writes but does not register
        with open(os.path.join(output_dir, "neuron_%s_soma_qc.html" % nid),
                  "w", newline="\n") as fh:
            fh.write("<html></html>\n")
        res = {"nid": nid, "qc_status": qc_status, "reasons": [], "files": files,
               "n_sections": 1, "exporter_id": "stub"}
        if kwargs.get("return_frames"):
            res["frames"] = {"labelled": _labelled_frame()}
        return res

    return _stub


def _make_gate_stub(record, status="pass", reasons=None, cell=None):
    def _stub(hoc_path, section_table, **kwargs):
        record["gate_hoc_path"] = hoc_path
        record["gate_cm"] = kwargs.get("cm")
        record["gate_Ra"] = kwargs.get("Ra")
        record["gate_Rm"] = kwargs.get("Rm")
        rep = {"qc_status": status, "reasons": list(reasons or []),
               "totnsegs": 4}
        return rep, (cell if status != hq.QC_FAIL else None)

    return _stub


def _syn_df():
    return pd.DataFrame({"x": [900.0], "y": [0.0], "z": [0.0],
                         "synapse_label": ["exc"]})


def _metadata():
    return pd.DataFrame({"nid": [1], "x": [0.0], "y": [0.0], "z": [0.0]})


class _Harness(object):
    """Patch alignment's collaborators, restore them whatever happens."""

    def __init__(self, export_stub=None, gate_stub=None, anchors=True,
                 snap=True):
        self.export_stub = export_stub
        self.gate_stub = gate_stub
        self.anchors = anchors
        self.snap = snap
        self._saved = {}
        self.tmp = None
        self.out = None

    def __enter__(self):
        self.tmp = tempfile.mkdtemp(prefix="align_gate_")
        self.out = os.path.join(self.tmp, "bank")
        os.makedirs(self.out, exist_ok=True)

        self._saved["export_neuron"] = mx.export_neuron
        self._saved["gate_hoc"] = hq.gate_hoc
        self._saved["soma_position_nm"] = al.soma_position_nm
        self._saved["neighbourhood_rotation"] = al.neighbourhood_rotation
        self._saved["make_align_fn"] = al.make_align_fn
        self._saved["resolve_synapse_anchors"] = al.resolve_synapse_anchors
        self._saved["snap_synapses"] = al.snap_synapses
        self._saved["write_mapped_synapses"] = al.write_mapped_synapses
        self._saved["spine_base_section_map"] = al.spine_base_section_map
        self._saved["default_cell_factory"] = al.default_cell_factory

        if self.export_stub is not None:
            mx.export_neuron = self.export_stub
        if self.gate_stub is not None:
            hq.gate_hoc = self.gate_stub
        al.soma_position_nm = lambda df: np.zeros(3)
        al.neighbourhood_rotation = lambda s, m, k: (np.eye(3), {"k": k})
        al.make_align_fn = lambda s, m: (lambda df, nid: df)

        if self.anchors:
            def _anchors(syn_df, df_labelled, soma_pos, mean_matrix, **kw):
                out = syn_df.copy()
                out["on_pruned_spine"] = False
                out["anchor_x"] = out["x"]
                out["anchor_y"] = out["y"]
                out["anchor_z"] = out["z"]
                out.attrs["n_unresolved_spine_bases"] = 0
                return out
            al.resolve_synapse_anchors = _anchors
        if self.snap:
            def _snap(anchored, cell):
                out = anchored.copy()
                out["lfpy_idx"] = 0
                out["lfpy_idx_naive"] = 0
                out["redirected"] = False
                return out
            al.snap_synapses = _snap
            # Must mirror the REAL return contract -- a dict, not a path. A
            # stub that lies about its return type hides exactly the kind of
            # integration bug these tests exist to catch.
            def _write(df, path, **kw):
                df.to_csv(path, index=False)
                return {"path": str(path), "n_rows": int(len(df)),
                        "n_label_defaulted": 0, "n_unknown_type": 0,
                        "n_redirected": int(df["redirected"].sum())
                        if "redirected" in df.columns else 0}
            al.write_mapped_synapses = _write
            al.spine_base_section_map = lambda df: {}
        return self

    def __exit__(self, *exc):
        for name, fn in self._saved.items():
            if name in ("export_neuron",):
                mx.export_neuron = fn
            elif name in ("gate_hoc",):
                hq.gate_hoc = fn
            else:
                setattr(al, name, fn)
        shutil.rmtree(self.tmp, ignore_errors=True)
        return False

    def listing(self):
        return sorted(os.listdir(self.out))

    def staging_left(self):
        parent = os.path.dirname(os.path.abspath(self.out))
        return [n for n in os.listdir(parent) if n.startswith(".s1_stage_")]


def _call(h, **kw):
    return al.align_and_export(pd.DataFrame({"id": [0]}), NID, h.out,
                               _metadata(), cm=0.5, Ra=200.0, verbose=False,
                               **kw)


# --------------------------------------------------------------------------- #
def test_t1_gate_pass_commits():
    rec = {}
    cell = _StubCell()
    with _Harness(_make_export_stub(rec), _make_gate_stub(rec, "pass", cell=cell)) as h:
        res = _call(h)                 # NOTE: no syn_df -- the "no redirect" path
        assert res["qc_status"] == "pass", res
        listing = h.listing()
        for want in ("neuron_777_aligned.hoc", "neuron_777_phi.csv",
                     "neuron_777_section_table.csv", "neuron_777_spine_bases.csv",
                     "neuron_777_provenance.json", "neuron_777_soma_qc.html",
                     "neuron_777_alignment.json"):
            assert want in listing, (want, listing)
        assert h.staging_left() == [], h.staging_left()
        # the export ran into staging, not into the bank
        assert rec["export_dir"] != h.out, rec["export_dir"]
        # every recorded path points at the committed copy and exists
        for key, path in res["files"].items():
            assert os.path.isfile(path), (key, path)
            assert os.path.dirname(os.path.abspath(path)) == \
                os.path.abspath(h.out), (key, path)
        # THE regression this guards: syn_df=None must still get a provenance
        # file. It is the resume signal a bank-scale run checkpoints against,
        # and a cell whose synapse CSV is simply missing (a normal, expected
        # case, not a failure) must not be invisible to that mechanism.
        assert "alignment_provenance" in res["files"], res["files"]
        with open(res["files"]["alignment_provenance"]) as fh:
            prov = json.load(fh)
        assert prov["qc_status"] == "pass", prov
        assert "n_synapses" not in prov, \
            "no redirect ran -- n_synapses must not appear"
        with open(res["files"]["provenance"]) as fh:
            exporter_prov = json.load(fh)
        assert exporter_prov["files"]["hoc"] == res["files"]["hoc"], \
            exporter_prov["files"]
        assert res["module_versions"]["hoc_qc"] == hq.MODULE_VERSION, res


def test_t2_gate_fail_leaves_output_untouched():
    rec = {}
    with _Harness(_make_export_stub(rec),
                  _make_gate_stub(rec, hq.QC_FAIL,
                                  reasons=["zero_deflection_compartments:3"])) as h:
        with open(os.path.join(h.out, "pre_existing.txt"), "w") as fh:
            fh.write("do not touch\n")
        before = h.listing()
        res = _call(h)
        assert res["qc_status"] == "fail", res
        assert res["files"] == {}, res["files"]
        assert h.listing() == before, (h.listing(), before)
        assert h.staging_left() == [], h.staging_left()
        assert any("propagation_qc:zero_deflection" in r
                   for r in res["reasons"]), res["reasons"]


def test_t3_gate_low_commits_and_downgrades():
    rec = {}
    cell = _StubCell()
    with _Harness(_make_export_stub(rec),
                  _make_gate_stub(rec, hq.QC_LOW,
                                  reasons=["monotonicity_violations:2"],
                                  cell=cell)) as h:
        res = _call(h)
        assert res["qc_status"] == "pass_low_confidence", res
        assert "neuron_777_aligned.hoc" in h.listing(), h.listing()
        assert any("monotonicity_violations" in r for r in res["reasons"]), res


def test_t4_one_cell_is_built_and_reused():
    rec = {}
    cell = _StubCell()
    built = {"legacy": 0}

    with _Harness(_make_export_stub(rec), _make_gate_stub(rec, "pass", cell=cell)) as h:
        def _legacy(*a, **k):
            built["legacy"] += 1
            return _StubCell()
        al.default_cell_factory = _legacy

        snapped_with = {}
        def _snap(anchored, c):
            snapped_with["cell"] = c
            out = anchored.copy()
            out["lfpy_idx"] = 0
            out["lfpy_idx_naive"] = 0
            out["redirected"] = False
            return out
        al.snap_synapses = _snap

        res = _call(h, syn_df=_syn_df())
        assert snapped_with["cell"] is cell, "the gate's cell must be reused"
        assert built["legacy"] == 0, "default_cell_factory must not be called"
        assert res["totnsegs"] == 4, res
        assert "mapped_synapses" in res["files"], res["files"]
        assert cell.released == 1, cell.released


def test_t5_df_labelled_comes_from_the_exporter():
    rec = {}
    cell = _StubCell()
    seen = {}
    with _Harness(_make_export_stub(rec), _make_gate_stub(rec, "pass", cell=cell)) as h:
        def _anchors(syn_df, df_labelled, soma_pos, mean_matrix, **kw):
            seen["frame"] = df_labelled
            out = syn_df.copy()
            out["on_pruned_spine"] = False
            out["anchor_x"] = out["x"]
            out["anchor_y"] = out["y"]
            out["anchor_z"] = out["z"]
            out.attrs["n_unresolved_spine_bases"] = 0
            return out
        al.resolve_synapse_anchors = _anchors

        res = _call(h, syn_df=_syn_df())          # NOTE: no df_labelled passed
        assert rec["return_frames"] is True, rec
        assert seen["frame"] is not None
        assert list(seen["frame"]["compartment_class"]) == ["soma", "dend"]
        assert "frames" not in res, "the DataFrame must not leak into res"
        # and the provenance JSON must still serialise
        with open(res["files"]["alignment_provenance"]) as fh:
            json.load(fh)

        # opt-in: the caller CAN have the frame, for plots, and asking for it
        # must not change anything else about the record
        res2 = _call(h, syn_df=_syn_df(), return_frames=True)
        assert "frames" in res2, res2.keys()
        assert list(res2["frames"]["labelled"]["compartment_class"]) == \
            ["soma", "dend"], res2["frames"]["labelled"]
        assert res2["qc_status"] == "pass", res2
        with open(res2["files"]["alignment_provenance"]) as fh:
            json.load(fh)             # still serialisable with frames present

        # and with no syn_df at all, which is how the plotting driver calls it
        res3 = _call(h, return_frames=True)
        assert "frames" in res3, res3.keys()
        assert "mapped_synapses" not in res3["files"], res3["files"]


def test_t6_explicit_frame_overrides():
    rec = {}
    cell = _StubCell()
    seen = {}
    mine = _labelled_frame()
    mine["marker"] = 1
    with _Harness(_make_export_stub(rec), _make_gate_stub(rec, "pass", cell=cell)) as h:
        def _anchors(syn_df, df_labelled, soma_pos, mean_matrix, **kw):
            seen["frame"] = df_labelled
            out = syn_df.copy()
            out["on_pruned_spine"] = False
            out["anchor_x"] = out["x"]
            out["anchor_y"] = out["y"]
            out["anchor_z"] = out["z"]
            out.attrs["n_unresolved_spine_bases"] = 0
            return out
        al.resolve_synapse_anchors = _anchors

        _call(h, syn_df=_syn_df(), df_labelled=mine)
        assert "marker" in seen["frame"].columns, seen["frame"].columns
        assert rec["return_frames"] is False, "no need to build it twice"


def test_t7_export_failure_short_circuits():
    rec = {}
    gate_calls = {"n": 0}

    def _gate(*a, **k):
        gate_calls["n"] += 1
        return {"qc_status": "pass", "reasons": []}, _StubCell()

    with _Harness(_make_export_stub(rec, qc_status="fail"), _gate) as h:
        before = h.listing()
        res = _call(h)
        assert res["qc_status"] == "fail", res
        assert gate_calls["n"] == 0, "the gate must not run on a failed export"
        assert h.listing() == before, h.listing()
        assert h.staging_left() == [], h.staging_left()


def test_t8_gate_off_restores_direct_write():
    rec = {}
    gate_calls = {"n": 0}
    built = {"legacy": 0}

    def _gate(*a, **k):
        gate_calls["n"] += 1
        return {"qc_status": "pass", "reasons": []}, _StubCell()

    with _Harness(_make_export_stub(rec), _gate) as h:
        def _legacy(*a, **k):
            built["legacy"] += 1
            return _StubCell()
        al.default_cell_factory = _legacy

        res = _call(h, syn_df=_syn_df(), qc_propagation=False)
        assert gate_calls["n"] == 0, "gate must be off"
        assert built["legacy"] == 1, "legacy factory must be used"
        assert rec["export_dir"] == h.out, "no staging when the gate is off"
        assert "propagation_qc" not in res, res.keys()
        assert "neuron_777_aligned.hoc" in h.listing(), h.listing()


def test_t9_exporter_qc_status_survives_the_gate():
    """The false rigidity failure, reproduced and fixed.

    regression_check compares qc_status (it is in al.REGRESSION_KEYS). The
    rigidity control is a bare export_neuron call that never runs the
    propagation gate, so comparing it against the POST-gate res["qc_status"]
    reports "alignment moved a section 7 quantity" for every cell the gate
    downgrades -- with byte-identical geometry. res["exporter_qc_status"] is
    the pre-gate verdict a caller must use instead.
    """
    rec = {}
    cell = _StubCell()

    # gate returns LOW: res["qc_status"] is downgraded, the exporter's is not
    with _Harness(_make_export_stub(rec),
                  _make_gate_stub(rec, hq.QC_LOW,
                                  reasons=["monotonicity_violations:2"],
                                  cell=cell)) as h:
        res = _call(h)
        assert res["qc_status"] == "pass_low_confidence", res
        assert res["exporter_qc_status"] == "pass", res

        # the naive comparison FAILS on identical geometry ...
        ctl = {k: res.get(k) for k in al.REGRESSION_KEYS}
        ctl["qc_status"] = "pass"                 # control never sees the gate
        naive = al.regression_check(ctl, res)
        assert not naive["identical"], "should reproduce the false positive"
        assert set(naive["diffs"]) == {"qc_status"}, naive["diffs"]

        # ... and the like-for-like comparison passes
        fixed = dict(res)
        fixed["qc_status"] = res["exporter_qc_status"]
        assert al.regression_check(ctl, fixed)["identical"], \
            al.regression_check(ctl, fixed)["diffs"]

    # a PASSING gate must leave the two agreeing, so the fix changes nothing
    with _Harness(_make_export_stub({}),
                  _make_gate_stub({}, "pass", cell=_StubCell())) as h2:
        res2 = _call(h2)
        assert res2["exporter_qc_status"] == res2["qc_status"] == "pass", res2


# --------------------------------------------------------------------------- #
def main():
    def _order(item):
        digits = item[0].split("_")[1][1:]
        return int(digits) if digits.isdigit() else 999

    tests = sorted([(k, v) for k, v in globals().items()
                    if k.startswith("test_")], key=_order)
    for name, fn in tests:
        fn()
        print("  ok  %s" % name)
    print("%d/%d align gate tests passed" % (len(tests), len(tests)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
