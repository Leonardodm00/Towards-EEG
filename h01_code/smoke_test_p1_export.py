#!/usr/bin/env python3
"""Smoke test of the P1 package: run_p1_export.py, build_p1_manifest.py,
p1_spine_stats.py, p1_hoc_audit.py, passive_params.csv, p1_export.pbs.

  python3 smoke_test_p1_export.py        (quiet)
  python3 smoke_test_p1_export.py -v     (every check)

Sections
  A  passive table: resolution, refusal of a blank / missing / non-shaft row;
     the two shipped INH variant tables (SST cm 1.0, PV/VIP cm 2.0), inh-only
     so an exc population is refused against them
  B  manifest builder: bank DISCOVERY (neurons/ vs alignment/, --bank,
     --bank-dir, the both-present warning, refusal naming every path
     tried), bank membership, missing skeleton, missing synapse file,
     --allow-unbanked, duplicate ids; manifest loader refusals
  C  section 4.1 tables on the HAND-LABELLED fixture the handoff specifies
     (spine A: 3-node neck [60, 40, 80] nm + 2-node head; spine B: stubby,
     1-node neck + 1 head; three synapses): neck_r_min == 40,
     neck_r_median == 60, neck_len == sum of the fixture's segment lengths,
     n_syn per spine, base is a shaft node, wide == groupby(long), spine_id
     format, R_neck arithmetic, a demoted continuation yields no row,
     idempotent re-write
  D  END TO END through the REAL export_neuron + align_and_export (Stage 1
     modules from the symlink farm) on a synthetic cell, with ONLY
     hoc_qc.gate_hoc stubbed (no NEURON in the sandbox): every artefact
     written, spine tables consistent with spine_bases and the C-09 file,
     record + fingerprint, resume, --force, --dry-run, --summarise; the same
     cell under BOTH passive variants into two trees, records naming their
     table, fingerprint on values not names (D16)
  E  structural audit: the emitted .hoc passes; a tampered .hoc (orphan
     section) fails and is quarantined, record says so, exit 0
  F  refusals: bad --task, cell not in manifest, blank passive row for a
     manifest population
  G  p1_export.pbs: text guards, bash -n, LF/ASCII, run with stub python3 /
     conda -> argv, MANIFEST required, wrong H01_CODE refused, the
     PASSIVE_TABLE / OUT_DIR variant knobs (relative, absolute, missing)

Pure ASCII, LF only. No network.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE1 = os.path.join(HERE, "stage1")
sys.path.insert(0, HERE)
sys.path.append(STAGE1)

VERBOSE = "-v" in sys.argv
RESULTS = []
CELL = 4242


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    if VERBOSE or not ok:
        print("  [%s] %-62s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _refused(fn):
    try:
        fn()
        return False
    except SystemExit:
        return True


def passive_csv(tmp, blank_l3=False):
    rows = [
        "layer,cell_type,cm_uF_cm2,Ra_ohm_cm,Rm_qc_ohm_cm2,cm_reference,source,provenance",
        "L2,exc,0.50,268.5,24000.0,shaft,literature,Eyal 2016",
        ("L3,exc,,,,,," if blank_l3 else "L3,exc,0.50,268.5,24000.0,shaft,literature,Eyal 2016"),
        "L3,inh,,,,,,TO BE FILLED",
        "L5,exc,1.0,495.7,24000.0,effective,literature,wrong reference on purpose",
    ]
    p = os.path.join(tmp, "passive_params.csv")
    with open(p, "w", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    return p


def synthetic_cell():
    """Soma at the origin, a 100 um dendrite along +x in 1 um steps, spine A
    (5 nodes) on the shaft node at 70 um, spine B (2 nodes) at 85 um, an axon.
    Radii chosen so the labeller and the three-vote rule keep both as spines.
    Returns the RAW frame (H01 vocabulary, no head/neck labels)."""
    rows = [(0, -1, 0.0, 0.0, 0.0, 5000.0, "Soma")]
    nid, prev, shaft = 1, 0, {}
    for k in range(1, 101):
        rows.append((nid, prev, 1000.0 * k, 0.0, 0.0, 300.0, "Dendrite"))
        shaft[k] = nid
        prev = nid
        nid += 1
    par = shaft[70]
    for dy, r in ((300.0, 60.0), (600.0, 40.0), (900.0, 80.0), (1200.0, 250.0), (1500.0, 300.0)):
        rows.append((nid, par, 70000.0, dy, 0.0, r, "Dendrite"))
        par = nid
        nid += 1
    par = shaft[85]
    for dy, r in ((400.0, 90.0), (800.0, 260.0)):
        rows.append((nid, par, 85000.0, dy, 0.0, r, "Dendrite"))
        par = nid
        nid += 1
    rows.append((nid, 0, -5000.0, 0.0, 0.0, 150.0, "Axon"))
    return pd.DataFrame(rows, columns=["id", "p", "x", "y", "z", "r", "annotated_type"])


def hand_labelled_fixture():
    """The handoff section 4.1 fixture, labelled BY HAND (annotated_type
    head/neck, compartment_class spine) so the pure table functions are
    tested independently of the labeller's own head/neck split.

    shaft: 0 (soma) - 1 - 2 - 3 - 4 - 5   along +x, 1000 nm steps, r 300
    spine A on node 3: neck 10 (r60) - 11 (r40) - 12 (r80), head 13 (r250) - 14 (r300)
                       segments along +y of 300, 300, 300, 300, 300 nm
    spine B on node 5: neck 20 (r90), head 21 (r260); segments 400, 400 nm
    axon 30 off the soma
    """
    rows = [
        (0, -1, 0.0, 0.0, 0.0, 5000.0, "Soma", "soma"),
        (1, 0, 1000.0, 0.0, 0.0, 300.0, "Dendrite", "dend"),
        (2, 1, 2000.0, 0.0, 0.0, 300.0, "Dendrite", "dend"),
        (3, 2, 3000.0, 0.0, 0.0, 300.0, "Dendrite", "dend"),
        (4, 3, 4000.0, 0.0, 0.0, 300.0, "Dendrite", "dend"),
        (5, 4, 5000.0, 0.0, 0.0, 300.0, "Dendrite", "dend"),
        (10, 3, 3000.0, 300.0, 0.0, 60.0, "neck", "spine"),
        (11, 10, 3000.0, 600.0, 0.0, 40.0, "neck", "spine"),
        (12, 11, 3000.0, 900.0, 0.0, 80.0, "neck", "spine"),
        (13, 12, 3000.0, 1200.0, 0.0, 250.0, "head", "spine"),
        (14, 13, 3000.0, 1500.0, 0.0, 300.0, "head", "spine"),
        (20, 5, 5000.0, 400.0, 0.0, 90.0, "neck", "spine"),
        (21, 20, 5000.0, 800.0, 0.0, 260.0, "head", "spine"),
        (30, 0, -1000.0, 0.0, 0.0, 150.0, "Axon", "axon"),
    ]
    return pd.DataFrame(rows, columns=["id", "p", "x", "y", "z", "r",
                                       "annotated_type", "compartment_class"])


def fixture_spine_bases():
    return pd.DataFrame({"spine_root_id": [10, 20], "base_node_id": [3, 5],
                         "n_nodes": [5, 2], "spine_base_section": ["dend[0]", "dend[0]"],
                         "section_id": [1, 1]})


def fixture_mapped_synapses():
    return pd.DataFrame({
        "syn_id": [101, 102, 103], "node_id": [14, 13, 21],
        "on_pruned_spine": [True, True, True], "spine_root_id": [10, 10, 20],
        "base_node_id": [3, 3, 5], "lfpy_idx": [7, 7, 9],
        "spine_base_section": ["dend[0]"] * 3,
        "synapse_label": ["exc_syn", "inh_syn", "exc_syn"]})


class StubCell(object):
    def __init__(self):
        self.totnsegs = 50

    def get_closest_idx(self, x=0.0, y=0.0, z=0.0):
        return int(min(49, max(0, round(float(x) / 2.0))))


def gate_stub_factory(status="pass"):
    def gate_stub(hoc_path, section_table, **kw):
        rep = {"qc_status": status, "reasons": [] if status == "pass" else ["stub fail"],
               "totnsegs": 50, "C4_soma": {"dv_soma_mV": 1.0}, "dv_min_mV": 0.1,
               "C3_monotone": {"n_violations": 0}}
        return rep, (StubCell() if status != "fail" else None)
    return gate_stub


def aspiny_cell():
    """Soma, a 30 um dendrite, an axon: no spine at all (every interneuron)."""
    rows = [(0, -1, 0.0, 0.0, 0.0, 5000.0, "Soma")]
    prev = 0
    for k in range(1, 31):
        rows.append((k, prev, 1000.0 * k, 0.0, 0.0, 300.0, "Dendrite"))
        prev = k
    rows.append((31, 0, -5000.0, 0.0, 0.0, 150.0, "Axon"))
    return pd.DataFrame(rows, columns=["id", "p", "x", "y", "z", "r", "annotated_type"])


def bank_path_of(root, subdir="neurons", layer="L3"):
    return os.path.join(root, subdir, "alignment_metadata_%s.csv" % layer)


def campaign_tree(tmp, cell_ids=(CELL,), with_synapses=True, bank_ids=None,
                  rotation=None, cell_frames=None, bank_subdir="neurons"):
    """A campaign root. The bank goes in neurons/ by default, because that is
    where extract_alignment_metadata writes it -- its output_directory is its
    own input_dir, the folder of neuron_<id>.csv (Alignment Metadata/Usage.py;
    confirmed on the cluster by the user, 2026-09-20). `bank_subdir` moves it,
    for the discovery cases in section B."""
    root = os.path.join(tmp, "h01")
    for d in ("neurons", "synapses", "alignment"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    df = synthetic_cell()
    for cid in cell_ids:
        frame = (cell_frames or {}).get(cid, df)
        frame.to_csv(os.path.join(root, "neurons", "neuron_%d.csv" % cid), index=False)
        if with_synapses:
            syn = pd.DataFrame({
                "synapse_id": [910011, 910012, 910013, 910014],
                "partner_neuron_id": [1, 2, 3, 4],
                "location_x": [70000 / 8.0, 85000 / 8.0, 50000 / 8.0, 60000 / 8.0],
                "location_y": [1500 / 8.0, 800 / 8.0, 0.0, 0.0],
                "location_z": [0.0, 0.0, 0.0, 0.0],
                "direction": ["incoming", "incoming", "incoming", "outgoing"],
                "synapse_type": [2, 1, 2, 2]})
            syn.to_csv(os.path.join(root, "synapses", "neuron_%d_synapses.csv" % cid),
                       index=False)
    ids = list(bank_ids if bank_ids is not None else cell_ids) + [9001, 9002]
    n = len(ids)
    R = np.eye(3) if rotation is None else np.asarray(rotation, float)
    bank = pd.DataFrame({"neuron_id": ids,
                         "soma_x": np.linspace(0.0, 2.0e5, n), "soma_y": np.zeros(n),
                         "soma_z": np.zeros(n),
                         "rotation_matrix": [R.tolist()] * n,
                         "FA_2D": [0.9] * n, "angle_from_mean": [1.0] * n})
    os.makedirs(os.path.join(root, bank_subdir), exist_ok=True)
    bank.to_csv(bank_path_of(root, bank_subdir), index=False)
    return root


# --------------------------------------------------------------------------- #
# A. passive table                                                            #
# --------------------------------------------------------------------------- #
def section_a(R, tmp):
    p = passive_csv(tmp)
    t = R.load_passive_table(p)
    v = R.resolve_passive(t, "L3", "exc")
    check("A1 passive row resolves (cm, Ra, Rm_qc, shaft)",
          v["cm"] == 0.5 and v["Ra"] == 268.5 and v["Rm_qc"] == 24000.0
          and v["cm_reference"] == "shaft" and v["source"] == "literature", str(v))
    check("A1' cell_type matching is case-insensitive",
          R.resolve_passive(t, "L3", "EXC")["cm"] == 0.5)
    for lay, ct, why in (("L3", "inh", "blank row"), ("L4", "exc", "missing row"),
                         ("L5", "exc", "cm_reference effective")):
        try:
            R.resolve_passive(t, lay, ct)
            check("A2 refuses %s (%s, %s)" % (why, lay, ct), False, "accepted")
        except SystemExit as e:
            check("A2 refuses %s (%s, %s)" % (why, lay, ct), True, str(e)[:60])
    # the shipped table: L2/L3 exc filled, every other population blank
    shipped = R.load_passive_table(os.path.join(HERE, "passive_params.csv"))
    ok = True
    for lay in ("L2", "L3"):
        ok &= R.resolve_passive(shipped, lay, "exc")["cm"] == 0.5
    n_blank = 0
    for lay in R.LAYERS:
        for ct in R.CELL_TYPES:
            if (lay, ct) in (("L2", "exc"), ("L3", "exc")):
                continue
            try:
                R.resolve_passive(shipped, lay, ct)
            except SystemExit:
                n_blank += 1
    check("A3 shipped passive_params.csv: L2/L3 exc resolve, the other 8 are refused",
          ok and n_blank == 8, "blank refused: %d" % n_blank)

    # ---- A4: the two INH variant tables (decision D-003) -------------------
    # inh-only on purpose: an exc manifest pointed at a variant is refused,
    # so exc cells can never be exported into an inh variant tree by mistake.
    want = {"passive_params_inh_SST.csv": (1.0, 100.0, 43103.0),
            "passive_params_inh_PVVIP.csv": (2.0, 100.0, 38760.0)}
    for fname, (cm, ra, rm) in want.items():
        vt = R.load_passive_table(os.path.join(HERE, fname))
        got = [R.resolve_passive(vt, lay, "inh") for lay in ("L2", "L3")]
        check("A4 %s: L2 and L3 inh resolve to cm %.1f Ra %.0f Rm_qc %.0f" % (fname, cm, ra, rm),
              all(g["cm"] == cm and g["Ra"] == ra and g["Rm_qc"] == rm
                  and g["cm_reference"] == "shaft" for g in got),
              str([(g["cm"], g["Ra"], g["Rm_qc"]) for g in got]))
        try:
            R.resolve_passive(vt, "L2", "exc")
            check("A4' %s refuses an EXC population (inh-only by design)" % fname, False, "accepted")
        except SystemExit as e:
            check("A4' %s refuses an EXC population (inh-only by design)" % fname,
                  "no row" in str(e), str(e)[:60])
        n_blank = sum(1 for lay in ("L4", "L5", "L6")
                      if _refused(lambda: R.resolve_passive(vt, lay, "inh")))
        check("A4'' %s: L4-L6 inh still blank and refused" % fname, n_blank == 3)
    s_cm = R.resolve_passive(R.load_passive_table(os.path.join(HERE, "passive_params_inh_SST.csv")), "L2", "inh")["cm"]
    p_cm = R.resolve_passive(R.load_passive_table(os.path.join(HERE, "passive_params_inh_PVVIP.csv")), "L2", "inh")["cm"]
    check("A5 the two variants differ in cm (1.0 vs 2.0), so their fingerprints differ",
          s_cm != p_cm, "%s vs %s" % (s_cm, p_cm))


# --------------------------------------------------------------------------- #
# B. manifest                                                                 #
# --------------------------------------------------------------------------- #
def section_b(R, B, tmp):
    root = campaign_tree(tmp, cell_ids=(CELL, 4243), with_synapses=True)
    os.remove(os.path.join(root, "synapses", "neuron_4243_synapses.csv"))
    m = B.build_manifest(root, "L3", "exc", [CELL, 4243])
    check("B1 manifest: two rows, relative paths, layer_source bank",
          len(m) == 2 and list(m["cell_id"]) == [CELL, 4243]
          and m["neuron_csv"].iloc[0] == os.path.join("neurons", "neuron_%d.csv" % CELL)
          and m["alignment_metadata"].iloc[0] == "neurons/alignment_metadata_L3.csv"
          and set(m["layer_source"]) == {"bank"}, m.to_string())
    check("B1' missing synapse file -> empty synapse_csv, not a refusal",
          m["synapse_csv"].iloc[0] != "" and m["synapse_csv"].iloc[1] == "")
    try:
        B.build_manifest(root, "L3", "exc", [CELL, 5555])
        check("B2 id without a skeleton is refused", False, "accepted")
    except SystemExit as e:
        check("B2 id without a skeleton is refused", "no skeleton" in str(e), str(e)[:60])
    # an id with a skeleton but outside the bank
    synthetic_cell().to_csv(os.path.join(root, "neurons", "neuron_7777.csv"), index=False)
    try:
        B.build_manifest(root, "L3", "exc", [7777])
        check("B3 id outside the layer bank is refused", False, "accepted")
    except SystemExit as e:
        check("B3 id outside the layer bank is refused", "not in" in str(e), str(e)[:60])
    m2 = B.build_manifest(root, "L3", "exc", [7777], allow_unbanked=True)
    check("B3' --allow-unbanked accepts it with layer_source manifest",
          len(m2) == 1 and m2["layer_source"].iloc[0] == "manifest")
    try:
        B.read_ids(ids="1,2,2")
        check("B4 duplicate ids refused", False, "accepted")
    except SystemExit:
        check("B4 duplicate ids refused", True)
    idf = os.path.join(tmp, "ids.txt")
    np.savetxt(idf, np.array([CELL, 4243]), fmt="%d")
    check("B4' --ids-file in Save-nids format", B.read_ids(ids_file=idf) == [CELL, 4243])
    mp = os.path.join(tmp, "L3_exc.csv")
    m.to_csv(mp, index=False)
    lm = R.load_manifest(mp)
    check("B5 loader round-trips the builder's file", len(lm) == 2 and list(lm["cell_id"]) == [CELL, 4243])
    for bad, why in (({"layer": "L9"}, "bad layer"), ({"cell_type": "pv"}, "bad cell_type")):
        mm = m.copy()
        for k, v in bad.items():
            mm.loc[0, k] = v
        mm.to_csv(mp, index=False)
        try:
            R.load_manifest(mp)
            check("B6 loader refuses %s" % why, False, "accepted")
        except SystemExit as e:
            check("B6 loader refuses %s" % why, True, str(e)[:60])

    # ---- B7: where the bank lives is SEARCHED, never assumed ---------------
    # It moved once already (docs said alignment/, the script writes it beside
    # the skeletons in neurons/), so each branch is pinned here.
    p, note = B.resolve_bank(root, "L3")
    check("B7 bank found in neurons/ (the extraction script's own output dir)",
          p == os.path.abspath(bank_path_of(root, "neurons")) and "neurons/" in note, note)

    alt = os.path.join(tmp, "b7alt")
    root_alt = campaign_tree(alt, cell_ids=(CELL,), bank_subdir="alignment")
    p2, note2 = B.resolve_bank(root_alt, "L3")
    check("B7' bank found in alignment/ when that is where it is",
          p2 == os.path.abspath(bank_path_of(root_alt, "alignment"))
          and "alignment/" in note2, note2)
    m_alt = B.build_manifest(root_alt, "L3", "exc", [CELL])
    check("B7'' the manifest carries the path actually found, not a fixed one",
          m_alt["alignment_metadata"].iloc[0] == "alignment/alignment_metadata_L3.csv",
          m_alt["alignment_metadata"].iloc[0])

    # the same name in two searched places: neurons/ wins, loudly
    both = pd.read_csv(bank_path_of(root, "neurons")).copy()
    both.loc[both.index[0], "soma_x"] = -1.0
    both.to_csv(bank_path_of(root, "alignment"), index=False)
    p3, note3 = B.resolve_bank(root, "L3")
    check("B7''' same bank name in two places -> first wins, WARNING names the other",
          p3 == os.path.abspath(bank_path_of(root, "neurons"))
          and "WARNING" in note3 and "alignment" in note3, note3)
    p4, note4 = B.resolve_bank(root, "L3", bank=os.path.join("alignment",
                                                             "alignment_metadata_L3.csv"))
    check("B7'''' --bank disambiguates (relative resolves against --root)",
          p4 == os.path.abspath(bank_path_of(root, "alignment")) and note4 == "--bank", note4)
    p5, note5 = B.resolve_bank(root, "L3", bank_dir="alignment")
    check("B7''''' --bank-dir picks the directory",
          p5 == os.path.abspath(bank_path_of(root, "alignment")), note5)
    os.remove(bank_path_of(root, "alignment"))

    empty = os.path.join(tmp, "b7none", "h01")
    os.makedirs(os.path.join(empty, "neurons"), exist_ok=True)
    try:
        B.resolve_bank(empty, "L5")
        check("B8 no bank anywhere -> refusal naming every path tried", False, "accepted")
    except SystemExit as e:
        s = str(e)
        check("B8 no bank anywhere -> refusal naming every path tried",
              "alignment_metadata_L5.csv" in s and s.count("alignment_metadata_L5.csv") >= 3
              and "--bank" in s, s[:120])
    for kw, why in (({"bank": "nope.csv"}, "--bank missing"),
                    ({"bank_dir": "nowhere"}, "--bank-dir missing"),
                    ({"bank": "a.csv", "bank_dir": "b"}, "both given")):
        try:
            B.resolve_bank(root, "L3", **kw)
            check("B8' refuses: %s" % why, False, "accepted")
        except SystemExit as e:
            check("B8' refuses: %s" % why, True, str(e)[:70])
    return root


# --------------------------------------------------------------------------- #
# C. section 4.1 tables on the hand-labelled fixture                          #
# --------------------------------------------------------------------------- #
def section_c(pss, sd, tmp):
    lab = hand_labelled_fixture()
    long_df = pss.long_table(lab, CELL, align_fn=None)
    comps = pss.spine_components(lab)
    check("C1 two spine components with roots 10 and 20, k from the root",
          sorted(comps.keys()) == [10, 20] and comps[10][0] == (10, 0)
          and comps[10][-1] == (14, 4) and comps[20] == [(20, 0), (21, 1)], str(comps))
    check("C2 long table: 7 rows, spine_id format, leaf flags, spine_part head/neck",
          len(long_df) == 7
          and set(long_df["spine_id"]) == {"%d:10" % CELL, "%d:20" % CELL}
          and long_df.loc[long_df["node_id"] == 14, "is_leaf"].iloc[0]
          and not long_df.loc[long_df["node_id"] == 13, "is_leaf"].iloc[0]
          and list(long_df.columns) == list(pss.LONG_COLUMNS)
          and set(long_df["spine_part"]) == {"head", "neck"})
    wide = pss.wide_table(long_df, lab, CELL, sd, spine_bases=fixture_spine_bases(),
                          mapped_synapses=fixture_mapped_synapses(), votes=None)
    w = wide.set_index("root_node_id")
    check("C3 neck_r_min == 40, neck_r_median == 60, neck_r_mean == 60 (spine A)",
          w.loc[10, "neck_r_min_nm"] == 40.0 and w.loc[10, "neck_r_median_nm"] == 60.0
          and abs(w.loc[10, "neck_r_mean_nm"] - 60.0) < 1e-9)
    # spine A's root sits 300 nm from a 300 nm-radius shaft: its base segment
    # lies entirely inside the shaft, so lengths count 0 for it (module doc)
    check("C3' lengths beyond the shaft: neck 600 (2 x 300), head 600, path 1200; "
          "centreline path 1500; base segment 300 of which 0 beyond",
          abs(w.loc[10, "neck_len_nm"] - 600.0) < 1e-9
          and abs(w.loc[10, "head_len_nm"] - 600.0) < 1e-9
          and abs(w.loc[10, "path_len_nm"] - 1200.0) < 1e-9
          and abs(w.loc[10, "path_len_centreline_nm"] - 1500.0) < 1e-9
          and abs(w.loc[10, "base_seg_len_nm"] - 300.0) < 1e-9
          and w.loc[10, "base_seg_len_beyond_shaft_nm"] == 0.0)
    check("C3'' stubby spine B: 1 neck node r 90, head max 260, root 400 nm off a 300 nm "
          "shaft -> neck_len 100, path 500 (centreline 800)",
          w.loc[20, "n_neck_nodes"] == 1 and w.loc[20, "neck_r_min_nm"] == 90.0
          and w.loc[20, "neck_r_base_nm"] == 90.0 and w.loc[20, "head_r_max_nm"] == 260.0
          and abs(w.loc[20, "neck_len_nm"] - 100.0) < 1e-9
          and abs(w.loc[20, "path_len_nm"] - 500.0) < 1e-9
          and abs(w.loc[20, "path_len_centreline_nm"] - 800.0) < 1e-9)
    check("C4 n_syn per spine (2, 1), ids joined, exc/inh split, lfpy idx",
          w.loc[10, "n_syn"] == 2 and w.loc[10, "syn_ids"] == "101;102"
          and w.loc[10, "n_syn_exc"] == 1 and w.loc[10, "n_syn_inh"] == 1
          and w.loc[20, "n_syn"] == 1 and w.loc[20, "syn_ids"] == "103"
          and w.loc[10, "base_lfpy_idx"] == "7" and w.loc[20, "base_lfpy_idx"] == "9")
    check("C5 base is the shaft node the spine hangs off, with its radius and class",
          w.loc[10, "base_node_id"] == 3 and w.loc[20, "base_node_id"] == 5
          and w.loc[10, "base_shaft_r_nm"] == 300.0 and w.loc[10, "base_class"] == "dend"
          and w.loc[10, "spine_base_section"] == "dend[0]" and w.loc[10, "section_id"] == 1)
    check("C5' d_from_um is the path distance soma -> base (3 um, 5 um)",
          abs(w.loc[10, "d_from_um"] - 3.0) < 1e-9 and abs(w.loc[20, "d_from_um"] - 5.0) < 1e-9)
    check("C5'' branch_order 0 on an unbranched dendrite (soma fan-out and axon do not count)",
          w.loc[10, "branch_order"] == 0 and w.loc[20, "branch_order"] == 0)
    # a dendritic side branch at node 4: spine B (base 5) becomes order 1, A stays 0
    lab_b = pd.concat([lab, pd.DataFrame(
        [(40, 4, 4000.0, -1000.0, 0.0, 200.0, "Dendrite", "dend")], columns=lab.columns)],
        ignore_index=True)
    bo = pss.branch_orders(lab_b)
    check("C5b a dendritic branch point raises the order distal to it only",
          bo[5] == 1 and bo[3] == 0 and bo[40] == 1 and bo[0] == 0,
          str({k: bo[k] for k in (0, 3, 4, 5, 40)}))
    # R_neck by hand: sum Ra*l/(pi r^2), Ra=100 ohm cm, l=300 nm for nodes 11
    # (r40) and 12 (r80); node 10 (r60) contributes 0: its segment is inside the shaft
    r_hand = sum(100.0 * 300e-7 / (np.pi * (r * 1e-7) ** 2) for r in (40.0, 80.0)) / 1e6
    check("C6 R_neck_skel_Ra100_MOhm matches the hand sum (spine A)",
          abs(w.loc[10, "R_neck_skel_Ra100_MOhm"] - r_hand) < 1e-9 * max(1.0, r_hand),
          "%.4f vs %.4f MOhm" % (w.loc[10, "R_neck_skel_Ra100_MOhm"], r_hand))
    # A_skel: frustum areas from spine_density's helper, including base->root
    a_hand = 0.0
    node, children, root = sd._prepare_nodes(lab, sd.SHAFT_REGEX, sd.SPINE_LABELS,
                                             sd.DEFAULT_RADIUS_NM, "nm")
    for s, p in ((10, 3), (11, 10), (12, 11), (13, 12), (14, 13)):
        a_hand += sd._frustum_lateral_area(node[p]["r"], node[s]["r"],
                                           sd._segment_length_um(node, p, s))
    check("C6' A_skel_frustum_um2 is the sum of spine_density frustums incl. the base segment",
          abs(w.loc[10, "A_skel_frustum_um2"] - a_hand) < 1e-12
          and w.loc[10, "A_skel_nobase_um2"] < w.loc[10, "A_skel_frustum_um2"])
    ok, why = pss.check_wide_is_groupby_of_long(long_df, wide)
    check("C7 wide == groupby(long)", ok, why)
    check("C7' wide columns are exactly WIDE_COLUMNS", list(wide.columns) == list(pss.WIDE_COLUMNS))
    # a demoted continuation produces no row: demote spine B (relabel as shaft,
    # as morphology_exporter._demote_roots does) and rebuild
    lab2 = lab.copy()
    lab2.loc[lab2["id"].isin([20, 21]), "annotated_type"] = "Dendrite"
    lab2.loc[lab2["id"].isin([20, 21]), "compartment_class"] = "dend"
    long2 = pss.long_table(lab2, CELL)
    wide2 = pss.wide_table(long2, lab2, CELL, sd)
    check("C8 a demoted continuation yields no row (spine B gone, A intact)",
          len(wide2) == 1 and int(wide2["root_node_id"].iloc[0]) == 10
          and len(long2) == 5)
    # aligned columns through a real make_align_fn map
    import alignment as al
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # +90 deg about z
    fn = al.make_align_fn(np.zeros(3), Rz)
    long3 = pss.long_table(lab, CELL, align_fn=fn)
    wide3 = pss.add_aligned_base(pss.wide_table(long3, lab, CELL, sd), lab, fn, CELL)
    w3 = wide3.set_index("root_node_id")
    # node 14 raw (3000, 1500, 0) nm -> rotated (-1500, 3000, 0) nm -> (-1.5, 3.0, 0) um
    check("C9 aligned columns through a real rotation (+90 deg about z), nm -> um once",
          abs(long3.loc[long3["node_id"] == 14, "x_al_um"].iloc[0] + 1.5) < 1e-12
          and abs(long3.loc[long3["node_id"] == 14, "y_al_um"].iloc[0] - 3.0) < 1e-12
          and abs(w3.loc[10, "base_x_al_um"] - 0.0) < 1e-12
          and abs(w3.loc[10, "base_y_al_um"] - 3.0) < 1e-12
          and abs(w3.loc[10, "tip_x_al_um"] + 1.5) < 1e-12
          and abs(w3.loc[10, "tip_y_al_um"] - 3.0) < 1e-12)
    # idempotent write
    out = os.path.join(tmp, "tables")
    p1 = pss.write_tables(out, CELL, long_df, wide)
    b1 = open(p1["spine_stats"], "rb").read()
    p2 = pss.write_tables(out, CELL, long_df, wide)
    b2 = open(p2["spine_stats"], "rb").read()
    check("C10 write_tables is idempotent and LF-only",
          b1 == b2 and b"\r" not in b1 and os.path.basename(p1["spine_nodes"])
          == "neuron_%d_spine_nodes.csv" % CELL)
    # partition votes through the REAL scorer/taper on a two-candidate branch
    # point: A (rho 0.97, cos 1, tapering) wins the tie and is demoted at step
    # 4b; B (rho 0.67, cos 0.91, tapering) loses the tie and survives. Voted
    # alone afterwards B is shaft-like AND branch-like -> kept_at_tie, never
    # 'undecidable' (the mislabel the review caught).
    import morphology_exporter as mx, node_classify as nc, shaft_continuation as shc
    import continuation_inspect as cinsp, sma_run as sr, spine_labeller as sl
    rows = [(0, -1, 0.0, 0.0, 0.0, 5000.0, "Soma")]
    nid_, prev = 1, 0
    for k in range(1, 101):
        rows.append((nid_, prev, 1000.0 * k, 0.0, 0.0, 300.0, "Dendrite"))
        prev = nid_
        nid_ += 1
    end = prev
    par = end
    for i, r in enumerate([290.0, 280.0, 270.0, 260.0, 250.0, 240.0]):
        rows.append((nid_, par, 100000.0 + 500.0 * (i + 1), 0.0, 0.0, r, "Dendrite"))
        par = nid_
        nid_ += 1
    par, th = end, np.deg2rad(25.0)
    for i, r in enumerate([200.0, 190.0, 180.0, 170.0, 160.0, 150.0]):
        rows.append((nid_, par, round(100000.0 + 500.0 * (i + 1) * np.cos(th), 3),
                     round(500.0 * (i + 1) * np.sin(th), 3), 0.0, r, "Dendrite"))
        par = nid_
        nid_ += 1
    rows.append((nid_, 0, -5000.0, 0.0, 0.0, 150.0, "Axon"))
    tie = pd.DataFrame(rows, columns=["id", "p", "x", "y", "z", "r", "annotated_type"])
    tlab, _ = sr.label_spines_project(tie, 77, sl, 4000.0)
    tlab = nc.classify_frame(tlab)
    before = sorted(pss.spine_components(tlab).keys())
    tdem, trep = mx.demote_shaft_continuations_three_vote(tlab)
    tdem = nc.classify_frame(tdem)
    after = sorted(pss.spine_components(tdem).keys())
    tv = pss.vote_table(tdem, shc, cinsp, sd)
    check("C12 tie-break fixture: two roots, one demoted at step 4b, the loser survives",
          before == [101, 107] and trep["demoted_roots"] == [101] and after == [107],
          "before %s demoted %s after %s" % (before, trep.get("demoted_roots"), after))
    check("C12' the survivor is labelled kept_at_tie with taper branch_like (not undecidable)",
          len(tv) == 1 and tv["partition_source"].iloc[0] == "kept_at_tie"
          and tv["taper_vote"].iloc[0] == "branch_like",
          tv.to_string())
    tw = pss.wide_table(pss.long_table(tdem, 77), tdem, 77, sd, votes=tv)
    check("C12'' and the wide table carries it; a root the scorer skips gets no_vote",
          tw["partition_source"].iloc[0] == "kept_at_tie"
          and pss.wide_table(pss.long_table(lab, CELL), lab, CELL, sd, votes=tv.iloc[0:0])
          ["partition_source"].eq("no_vote").all())
    # empty cell keeps the schema
    lab_e = lab[lab["compartment_class"] != "spine"].copy()
    le = pss.long_table(lab_e, CELL)
    we = pss.wide_table(le, lab_e, CELL, sd)
    check("C11 a cell with no spines yields empty tables with the schema",
          len(le) == 0 and len(we) == 0 and list(we.columns) == list(pss.WIDE_COLUMNS))


# --------------------------------------------------------------------------- #
# D. end to end                                                               #
# --------------------------------------------------------------------------- #
def run_driver(R, argv):
    """Call main() in-process, capturing stdout."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = R.main(argv)
        except SystemExit as e:
            rc = ("SystemExit", str(e))
    return rc, buf.getvalue()


def section_d(R, B, tmp):
    import hoc_qc as hq
    root = campaign_tree(tmp, cell_ids=(CELL,))
    ptab = passive_csv(tmp)
    mp = os.path.join(root, "p1", "manifests", "L3_exc.csv")
    B.build_manifest(root, "L3", "exc", [CELL]).pipe(
        lambda m: (os.makedirs(os.path.dirname(mp), exist_ok=True), m.to_csv(mp, index=False)))
    base = ["--root", root, "--manifest", mp, "--stage1-dir", STAGE1,
            "--passive-table", ptab, "--no-neuron-validate", "-v"]
    saved = hq.gate_hoc
    hq.gate_hoc = gate_stub_factory("pass")
    try:
        rc, out = run_driver(R, base + ["--task", "0", "--dry-run"])
        check("D1 --dry-run resolves inputs and exits 0 without writing",
              rc == 0 and "DRY RUN" in out
              and not os.path.isdir(os.path.join(root, "p1", str(CELL))), out[-300:])
        rc, out = run_driver(R, base + ["--task", "0"])
        odir = os.path.join(root, "p1", str(CELL))
        check("D2 the task exits 0", rc == 0, out[-600:] if rc else "")
        want = ["neuron_%d_%s" % (CELL, s) for s in (
            "aligned.hoc", "phi.csv", "segment_map.csv", "section_table.csv",
            "spine_bases.csv", "synapses.csv", "provenance.json", "alignment.json",
            "mapped_synapses.csv", "spine_nodes.csv", "spine_stats.csv",
            "hoc_validation.json", "p1.json")]
        have = sorted(os.listdir(odir)) if os.path.isdir(odir) else []
        check("D3 every P1 artefact is in p1/<id>/", all(w in have for w in want),
              str([w for w in want if w not in have]))
        rec = json.load(open(os.path.join(odir, "neuron_%d_p1.json" % CELL)))
        r = rec.get("result") or {}
        check("D4 record: status ok, qc pass, gate pass, rigidity identical, hoc pass",
              rec["status"] == "ok" and r["qc_status"] == "pass"
              and r.get("gate_status") == "pass"
              and rec["rigidity"]["identical"] is True
              and rec["hoc_validation_summary"]["verdict"] == "pass", json.dumps(
                  {k: rec.get(k) for k in ("status", "rigidity", "hoc_validation_summary")})[:300])
        check("D4' record carries the passive set, the population and the fingerprint",
              rec["passive"]["cm"] == 0.5 and rec["passive"]["Ra"] == 268.5
              and rec["layer"] == "L3" and rec["cell_type"] == "exc"
              and rec["layer_source"] == "bank"
              and len(rec["fingerprint"]) == 12
              and rec["fingerprint_detail"]["partition"]["rule"] == "three_vote"
              and rec["fingerprint_detail"]["cap"] == {"cap_tips": True, "cap_h_um": 0.1})
        check("D4'' standing decisions reached the exporter (cap on, three votes applied)",
              r.get("cap_tips") is True and abs(float(r.get("cap_h_um")) - 0.1) < 1e-12
              and (r.get("continuation_report") or {}).get("applied") is True)
        sb = pd.read_csv(os.path.join(odir, "neuron_%d_spine_bases.csv" % CELL))
        ws = pd.read_csv(os.path.join(odir, "neuron_%d_spine_stats.csv" % CELL))
        ln = pd.read_csv(os.path.join(odir, "neuron_%d_spine_nodes.csv" % CELL))
        ms = pd.read_csv(os.path.join(odir, "neuron_%d_mapped_synapses.csv" % CELL))
        check("D5 spine_stats rows == exporter's pruned spines, same roots and bases",
              len(ws) == len(sb) == 2
              and sorted(ws["root_node_id"]) == sorted(sb["spine_root_id"])
              and sorted(ws["base_node_id"]) == sorted(sb["base_node_id"]))
        check("D5' spine_nodes rows == n_spine_nodes the exporter removed",
              len(ln) == int(r["spine_report"]["n_nodes_removed"]) == 7)
        on = ms[ms["on_pruned_spine"].astype(bool)]
        by_root = on.groupby("spine_root_id")["syn_id"].apply(lambda s: ";".join(str(v) for v in s))
        check("D6a syn_id is the H01 synapse_id, not a row position",
              set(ms["syn_id"]) == {910011, 910012, 910013}
              and rec["inputs"]["syn_id_source"] == "synapse_id"
              and all(int(v) >= 910011 for v in ws["syn_ids"].astype(str).str.split(";").explode()),
              str(sorted(ms["syn_id"])))
        wsr = ws.set_index("root_node_id")
        check("D6 n_syn / syn_ids per spine agree with the C-09 file (3 incoming kept, 1 outgoing "
              "dropped; 2 on spines, 1 on the shaft)",
              int(ws["n_syn"].sum()) == len(on) == 2 and r["n_synapses"] == 3
              and all(str(wsr.loc[int(k), "syn_ids"]) == v for k, v in by_root.items())
              and int(wsr.loc[int(by_root.index[0]), "n_syn"]) == len(on[on["spine_root_id"] == by_root.index[0]]))
        check("D6' base_lfpy_idx comes from the snapped synapses; base section from spine_bases",
              all(str(v) != "" for v in ws["base_lfpy_idx"])
              and set(ws["spine_base_section"]) == set(sb["spine_base_section"]))
        check("D7 both fixture spines are plain labeller spines with finite votes",
              list(ws["partition_source"]) == ["labeller", "labeller"]
              and np.isfinite(ws["rho_vote"]).all() and (ws["taper_vote"] == "spine_like").all(),
              str(list(zip(ws["partition_source"], ws["taper_vote"]))))
        check("D7' aligned coordinates filled (identity bank -> raw/1000)",
              np.isfinite(ln["x_al_um"]).all() and np.isfinite(ws["base_x_al_um"]).all()
              and abs(float(wsr.loc[wsr.index[0], "base_x_al_um"]) * 1000.0
                      - float(wsr.loc[wsr.index[0], "base_x_nm"])) < 1e-6)
        check("D8 angle_from_z_deg recorded and finite",
              np.isfinite(float(rec.get("angle_from_z_deg", float("nan")))))
        # resume
        mt = os.path.getmtime(os.path.join(odir, "neuron_%d_aligned.hoc" % CELL))
        rc, out = run_driver(R, base + ["--task", "0"])
        check("D9 second run resumes on a matching fingerprint (nothing rewritten)",
              rc == 0 and "resumed" in out
              and os.path.getmtime(os.path.join(odir, "neuron_%d_aligned.hoc" % CELL)) == mt)
        rc, out = run_driver(R, base + ["--task", "0", "--force"])
        check("D9' --force reprocesses", rc == 0 and "resumed" not in out and "record:" in out)
        # a changed passive set changes the fingerprint and reprocesses
        with open(ptab) as fh:
            txt = fh.read()
        p2 = os.path.join(tmp, "passive2.csv")
        with open(p2, "w", newline="\n") as fh:
            fh.write(txt.replace("L3,exc,0.50,268.5", "L3,exc,0.55,268.5"))
        rc, out = run_driver(R, ["--root", root, "--manifest", mp, "--stage1-dir", STAGE1,
                                 "--passive-table", p2, "--no-neuron-validate", "--task", "0"])
        rec2 = json.load(open(os.path.join(odir, "neuron_%d_p1.json" % CELL)))
        check("D10 a changed cm changes the fingerprint and reprocesses",
              rc == 0 and "reprocessing" in out and rec2["fingerprint"] != rec["fingerprint"]
              and rec2["passive"]["cm"] == 0.55)
        # --cell selection and --summarise
        rc, out = run_driver(R, base + ["--cell", str(CELL), "--force"])
        check("D11 --cell selects the row", rc == 0 and "record:" in out)
        rc, out = run_driver(R, ["--root", root, "--summarise"])
        sp = os.path.join(root, "p1", "p1_summary.csv")
        summ = pd.read_csv(sp)
        check("D12 --summarise writes one row per cell with the key columns",
              rc == 0 and len(summ) == 1 and int(summ["cell_id"].iloc[0]) == CELL
              and summ["status"].iloc[0] == "ok" and summ["qc_status"].iloc[0] == "pass"
              and int(summ["n_spines"].iloc[0]) == 2 and summ["hoc_verdict"].iloc[0] == "pass"
              and list(summ.columns) == list(R.SUMMARY_KEYS), str(list(summ.columns))[:200])
        m3 = pd.read_csv(mp)
        m3 = pd.concat([m3, m3.assign(cell_id=4299)], ignore_index=True)
        mp3 = os.path.join(root, "p1", "manifests", "L3_exc_plus.csv")
        m3.to_csv(mp3, index=False)
        rc, out = run_driver(R, ["--root", root, "--summarise", "--manifest", mp3])
        check("D12' --summarise --manifest names the manifest cells without a record and exits 1",
              rc == 1 and "1 cell(s) without a record: [4299]" in out, out[-200:])
        # gate failure: nothing but the record
        hq.gate_hoc = gate_stub_factory("fail")
        shutil.rmtree(odir, ignore_errors=True)
        rc, out = run_driver(R, base + ["--task", "0"])
        have = sorted(os.listdir(odir)) if os.path.isdir(odir) else []
        recf = json.load(open(os.path.join(odir, "neuron_%d_p1.json" % CELL)))
        check("D13 gate FAIL: exit 0, GATED OUT, only the record written, qc fail in it",
              rc == 0 and "GATED OUT" in out and have == ["neuron_%d_p1.json" % CELL]
              and recf["result"]["qc_status"] == "fail" and recf.get("gated_out") is True)
        rc, out = run_driver(R, base + ["--task", "0"])
        check("D13'' a gated-out verdict is final on rerun (resumed), reprocessed only by --force",
              rc == 0 and "resumed" in out)
        check("D13' staging left nothing behind", not any(n.startswith(".s1_stage_")
                                                          for n in os.listdir(os.path.join(root, "p1"))))
        # ---- an ASPINY cell with synapses (every interneuron): no crash, empty
        # tables with the schema, the C-09 file written (the crash the review found)
        hq.gate_hoc = gate_stub_factory("pass")
        ASPINY = 4300
        root2 = campaign_tree(os.path.join(tmp, "aspiny"), cell_ids=(ASPINY,),
                              cell_frames={ASPINY: aspiny_cell()})
        mp2 = os.path.join(root2, "p1", "manifests", "L3_exc.csv")
        os.makedirs(os.path.dirname(mp2), exist_ok=True)
        B.build_manifest(root2, "L3", "exc", [ASPINY]).to_csv(mp2, index=False)
        rc, out = run_driver(R, ["--root", root2, "--manifest", mp2, "--stage1-dir", STAGE1,
                                 "--passive-table", ptab, "--no-neuron-validate", "--task", "0"])
        od2 = os.path.join(root2, "p1", str(ASPINY))
        rec2 = json.load(open(os.path.join(od2, "neuron_%d_p1.json" % ASPINY)))
        ws2 = pd.read_csv(os.path.join(od2, "neuron_%d_spine_stats.csv" % ASPINY))
        check("D14 an aspiny cell with synapses: exit 0, status ok, C-09 written, empty spine tables "
              "with the schema, no_spines recorded",
              rc == 0 and rec2["status"] == "ok" and rec2["result"]["qc_status"] == "pass"
              and os.path.isfile(os.path.join(od2, "neuron_%d_mapped_synapses.csv" % ASPINY))
              and len(ws2) == 0 and list(ws2.columns) == list(R.import_modules(
                  type("A", (), {"stage1_dir": STAGE1})())["pss"].WIDE_COLUMNS)
              and rec2["spine_tables"]["n_spines"] == 0 and rec2["result"]["n_synapses"] == 3,
              "rc=%s status=%s err=%s\n%s" % (rc, rec2.get("status"), rec2.get("error"), out[-500:]))
        sb2 = os.path.join(od2, "neuron_%d_spine_bases.csv" % ASPINY)
        check("D14' the exporter (1.2.1) writes a header-only spine_bases.csv for it",
              os.path.getsize(sb2) > 1 and list(pd.read_csv(sb2).columns) == list(R.SPINE_BASES_COLUMNS))
        # ---- a regenerated input changes the fingerprint (bank rewritten)
        bank_p = bank_path_of(root, "neurons")
        bank = pd.read_csv(bank_p)
        bank.loc[bank.index[-1], "soma_x"] += 1.0
        bank.to_csv(bank_p, index=False)
        rc, out = run_driver(R, base + ["--task", "0", "--dry-run"])
        check("D15 a regenerated bank changes the fingerprint (input identity is fingerprinted)",
              rc == 0 and "reprocessing" in out, out[-300:])

        # ---- D16: PASSIVE VARIANTS (decision D-003) -------------------------
        # The same cell, exported under two passive tables into two trees.
        # Both must exist afterwards, name their table, differ in fingerprint,
        # and the fingerprint must NOT depend on the table's name -- only on
        # its values -- or two identically-filled tables would never resume.
        vroot = campaign_tree(os.path.join(tmp, "variants"), cell_ids=(CELL,))
        vman = os.path.join(vroot, "p1", "manifests", "L3_inh.csv")
        os.makedirs(os.path.dirname(vman), exist_ok=True)
        B.build_manifest(vroot, "L3", "exc", [CELL]).assign(cell_type="inh").to_csv(vman, index=False)
        sst = os.path.join(HERE, "passive_params_inh_SST.csv")
        pvv = os.path.join(HERE, "passive_params_inh_PVVIP.csv")
        trees = {}
        for name, table in (("p1_inh_SST", sst), ("p1_inh_PVVIP", pvv)):
            od = os.path.join(vroot, name)
            rc, out = run_driver(R, ["--root", vroot, "--manifest", vman, "--stage1-dir", STAGE1,
                                     "--passive-table", table, "--out-dir", od,
                                     "--no-neuron-validate", "--task", "0"])
            recp = os.path.join(od, str(CELL), "neuron_%d_p1.json" % CELL)
            trees[name] = (rc, out, json.load(open(recp)) if os.path.isfile(recp) else None)
        a, b = trees["p1_inh_SST"][2], trees["p1_inh_PVVIP"][2]
        check("D16 the same cell exports under BOTH variants, each into its own tree",
              a is not None and b is not None and a["status"] == "ok" and b["status"] == "ok",
              "rc=%s/%s" % (trees["p1_inh_SST"][0], trees["p1_inh_PVVIP"][0]))
        if a and b:
            check("D16a each record names its passive table and carries its sha",
                  a["passive"]["passive_table"] == "passive_params_inh_SST.csv"
                  and b["passive"]["passive_table"] == "passive_params_inh_PVVIP.csv"
                  and len(a["passive"]["passive_table_sha"]) == 12
                  and a["passive"]["passive_table_sha"] != b["passive"]["passive_table_sha"])
            check("D16b cm 1.0 vs 2.0 -> different fingerprints, so neither resumes into the other",
                  a["passive"]["cm"] == 1.0 and b["passive"]["cm"] == 2.0
                  and a["fingerprint"] != b["fingerprint"],
                  "%s vs %s" % (a["fingerprint"], b["fingerprint"]))
            check("D16c the fingerprint hashes the VALUES only: no table name inside it",
                  "passive_table" not in a["fingerprint_detail"]["passive"]
                  and "passive_table_sha" not in a["fingerprint_detail"]["passive"]
                  and set(a["fingerprint_detail"]["passive"]) == {"cm", "Ra", "Rm_qc", "cm_reference"},
                  str(sorted(a["fingerprint_detail"]["passive"])))
            # a COPY of the SST table under another name must resume, not reprocess
            sst_copy = os.path.join(tmp, "renamed_sst.csv")
            shutil.copy(sst, sst_copy)
            rc, out = run_driver(R, ["--root", vroot, "--manifest", vman, "--stage1-dir", STAGE1,
                                     "--passive-table", sst_copy,
                                     "--out-dir", os.path.join(vroot, "p1_inh_SST"),
                                     "--no-neuron-validate", "--task", "0"])
            check("D16d an identically-filled table under another NAME resumes (values, not names)",
                  rc == 0 and "resumed" in out, out[-200:])
            # the summary of each tree names the table
            for name in ("p1_inh_SST", "p1_inh_PVVIP"):
                rc, out = run_driver(R, ["--root", vroot, "--out-dir", os.path.join(vroot, name),
                                         "--summarise"])
                sm = pd.read_csv(os.path.join(vroot, name, "p1_summary.csv"))
                check("D16e %s/p1_summary.csv carries passive_table" % name,
                      rc == 0 and "passive_table" in sm.columns
                      and str(sm["passive_table"].iloc[0]).endswith(name.replace("p1_inh_", "") + ".csv"),
                      str(sm.get("passive_table", pd.Series()).tolist()))
        # an EXC manifest against an inh-only variant table is refused
        rc, out = run_driver(R, base + ["--task", "0", "--dry-run", "--passive-table", sst])
        check("D16f an exc manifest against an inh-only variant table is refused",
              isinstance(rc, tuple) and "no row" in rc[1], str(rc)[:120])
    finally:
        hq.gate_hoc = saved
    return root, mp, ptab


# --------------------------------------------------------------------------- #
# E. structural audit and quarantine                                          #
# --------------------------------------------------------------------------- #
def section_e(R, pha, root, mp, ptab, tmp):
    import hoc_qc as hq
    odir = os.path.join(root, "p1", str(CELL))
    saved = hq.gate_hoc
    hq.gate_hoc = gate_stub_factory("pass")
    try:
        base = ["--root", root, "--manifest", mp, "--stage1-dir", STAGE1,
                "--passive-table", ptab, "--no-neuron-validate", "--force", "--task", "0"]
        rc, out = run_driver(R, base)
        hoc = os.path.join(odir, "neuron_%d_aligned.hoc" % CELL)
        geo = pha.audit_hoc_geometry(hoc)
        v = pha.classify_violations(geo)
        check("E1 the emitted .hoc passes the structural audit",
              pha.verdict(v) == "pass" and len(geo["roots"]) == 1 and not geo["orphans"],
              str(v))
        # tamper: add an unconnected section
        txt = open(hoc).read()
        bad = txt + "\ncreate junk[1]\njunk[0] {\n  pt3dadd(0, 0, 0, 1)\n  pt3dadd(1, 0, 0, 1)\n}\n"
        with open(os.path.join(tmp, "bad.hoc"), "w", newline="\n") as fh:
            fh.write(bad)
        geo2 = pha.audit_hoc_geometry(os.path.join(tmp, "bad.hoc"))
        v2 = pha.classify_violations(geo2)
        check("E2 an orphan section is a STRUCTURAL violation -> fail",
              pha.verdict(v2) == "fail" and any("debris" in m for m in v2[pha.STRUCTURAL]))
        check("E2' merge_verdict only lowers", pha.merge_verdict("pass", "fail") == "fail"
              and pha.merge_verdict("pass_low_confidence", "pass") == "pass_low_confidence")
        # run the driver with a patched audit that always fails -> quarantine
        real_audit = pha.audit_hoc_geometry
        pha.audit_hoc_geometry = lambda p: geo2
        try:
            rc, out = run_driver(R, base)
        finally:
            pha.audit_hoc_geometry = real_audit
        qdir = os.path.join(odir, "_quarantine")
        rec = json.load(open(os.path.join(odir, "neuron_%d_p1.json" % CELL)))
        check("E3 structural fail -> QUARANTINED, artefacts moved, record says so, exit 0",
              rc == 0 and "QUARANTINED" in out and os.path.isdir(qdir)
              and os.path.isfile(os.path.join(qdir, "neuron_%d_aligned.hoc" % CELL))
              and not os.path.isfile(hoc)
              and rec.get("quarantined") is True and rec["result"]["qc_status"] == "fail"
              and rec["hoc_validation_summary"]["verdict"] == "fail"
              and os.path.isfile(rec["hoc_validation_summary"]["path"]))
        check("E3' no spine tables for a quarantined cell",
              not os.path.isfile(os.path.join(odir, "neuron_%d_spine_stats.csv" % CELL)))
        # neuron_validate degrades gracefully without NEURON
        nrep = pha.neuron_validate(os.path.join(tmp, "bad.hoc"), sys_path=[HERE, STAGE1])
        check("E4 neuron_validate reports neuron_available False here (no NEURON) or a report",
              isinstance(nrep, dict) and "neuron_available" in nrep)
    finally:
        hq.gate_hoc = saved


# --------------------------------------------------------------------------- #
# F. refusals                                                                 #
# --------------------------------------------------------------------------- #
def section_f(R, root, mp, tmp):
    ptab = passive_csv(tmp)
    base = ["--root", root, "--manifest", mp, "--stage1-dir", STAGE1, "--passive-table", ptab]
    for argv, why in ((base + ["--task", "7"], "task out of range"),
                      (base + ["--cell", "999"], "cell not in manifest"),
                      (base, "neither --task nor --cell"),
                      (base + ["--task", "0", "--cell", str(CELL)], "both --task and --cell"),
                      (["--root", os.path.join(tmp, "nope"), "--manifest", mp, "--task", "0"], "missing root")):
        rc, out = run_driver(R, argv)
        check("F1 refuses: %s" % why, isinstance(rc, tuple) and rc[0] == "SystemExit", str(rc)[:60])
    pb = passive_csv(tmp, blank_l3=True)
    rc, out = run_driver(R, ["--root", root, "--manifest", mp, "--stage1-dir", STAGE1,
                             "--passive-table", pb, "--task", "0"])
    check("F2 a blank passive row for the manifest's population is refused before any export",
          isinstance(rc, tuple) and "blank" in rc[1])


# --------------------------------------------------------------------------- #
# G. the job script                                                           #
# --------------------------------------------------------------------------- #
STUB_PY = """#!/bin/bash
printf '%s\\n' "$@" >> "$STUB_ARGV_OUT"
exit 0
"""
STUB_CONDA = r"""#!/bin/bash
case "$1" in
    shell.bash)
        cat <<EOF
conda() {
    if [ "\$1" = activate ]; then
        echo "INFO: stub activate.d hook (\$2)"
        if [ -d "$STUB_ENVS/\$2/bin" ]; then
            export PATH="$STUB_ENVS/\$2/bin:\$PATH"
            export CONDA_DEFAULT_ENV="\$2"
        fi
        return 1
    fi
    return 0
}
EOF
        ;;
esac
exit 0
"""


def _write_exec(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, 0o755)


def section_g(tmp):
    pbs = os.path.join(HERE, "p1_export.pbs")
    text = open(pbs).read()
    raw = open(pbs, "rb").read()
    rc = subprocess.run(["bash", "-n", pbs], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        universal_newlines=True)
    check("G1 p1_export.pbs: bash -n, LF only, pure ASCII",
          rc.returncode == 0 and b"\r" not in raw and all(b < 128 for b in raw))
    check("G2 text guards: H01_ENV knob, ENV_NAME/CODE/ROOT reported, passive table + farm checked",
          'H01_ENV="${H01_ENV:-spine_env}"' in text and "NOTE: ENV_NAME is set" in text
          and "for _stale in ROOT CODE" in text and '[ ! -f "$PASSIVE_TABLE" ]' in text
          and 'PASSIVE_TABLE="${PASSIVE_TABLE:-$H01_CODE/passive_params.csv}"' in text
          and 'OUT_DIR="${OUT_DIR:-$H01_ROOT/p1}"' in text
          and '"$H01_CODE/stage1" ]' in text and "${MANIFEST:?" in text)
    # fixture
    fx = os.path.join(tmp, "jobfix")
    code = os.path.join(fx, "h01_code")
    root = os.path.join(fx, "h01")
    envs = os.path.join(fx, "envs", "spine_env", "bin")
    for d in (os.path.join(code, "stage1"), os.path.join(root, "neurons"), os.path.join(fx, "bin"), envs):
        os.makedirs(d, exist_ok=True)
    open(os.path.join(code, "run_p1_export.py"), "w").close()
    open(os.path.join(code, "passive_params.csv"), "w").close()
    man = os.path.join(root, "p1", "manifests", "L3_exc.csv")
    os.makedirs(os.path.dirname(man), exist_ok=True)
    open(man, "w").close()
    _write_exec(os.path.join(fx, "bin", "python3"), STUB_PY)
    _write_exec(os.path.join(envs, "python3"), STUB_PY)
    _write_exec(os.path.join(fx, "bin", "conda"), STUB_CONDA)

    def run(extra, code_dir=code):
        argv_out = os.path.join(fx, "argv_%d.txt" % len(RESULTS))
        env = {k: v for k, v in os.environ.items()
               if not (k in ("CODE", "ROOT", "ENV_NAME", "SKIP_CONDA", "MANIFEST", "DRY_RUN",
                             "FORCE", "NO_RIGIDITY", "NO_NEURON_VALIDATE",
                             "PASSIVE_TABLE", "OUT_DIR")
                       or k.startswith("H01_") or k.startswith("PBS_") or k.startswith("CONDA"))}
        env.update({"PATH": os.path.join(fx, "bin") + os.pathsep + os.environ.get("PATH", ""),
                    "HOME": fx, "H01_CODE": code_dir, "H01_ROOT": root,
                    "STUB_ARGV_OUT": argv_out, "STUB_ENVS": os.path.join(fx, "envs")})
        env.update(extra)
        p = subprocess.run(["bash", pbs], cwd=fx, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)
        argv = open(argv_out).read().splitlines() if os.path.isfile(argv_out) else []
        return p.returncode, p.stdout, argv

    def argval(argv, flag):
        return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None

    rc, out, argv = run({"MANIFEST": man, "PBS_ARRAY_INDEX": "2", "ENV_NAME": "sbi_export",
                         "DRY_RUN": "1", "NO_RIGIDITY": "1"})
    check("G3 runs through the stub hook: spine_env activated, ENV_NAME ignored, runner invoked",
          rc == 0 and "NOTE: ENV_NAME is set (sbi_export)" in out
          and "hook (spine_env)" in out and argv[:1] == ["run_p1_export.py"],
          "rc=%d\n%s" % (rc, out[-400:]) if rc else "")
    check("G3' argv: --manifest, --task 2, --passive-table, --stage1-dir, --dry-run, --no-rigidity-control",
          argval(argv, "--manifest") == man and argval(argv, "--task") == "2"
          and argval(argv, "--passive-table") == os.path.join(code, "passive_params.csv")
          and argval(argv, "--stage1-dir") == os.path.join(code, "stage1")
          and argval(argv, "--out-dir") == os.path.join(root, "p1")
          and "--dry-run" in argv and "--no-rigidity-control" in argv
          and "--force" not in argv, str(argv))
    rc, out, argv = run({"MANIFEST": os.path.join("..", "h01", "p1", "manifests", "L3_exc.csv"),
                         "SKIP_CONDA": "1", "PBS_ARRAYID": "1"})
    check("G4 a relative MANIFEST resolves against H01_CODE; Torque PBS_ARRAYID fallback",
          rc == 0 and argval(argv, "--manifest") == os.path.join(code, "..", "h01", "p1", "manifests", "L3_exc.csv")
          and argval(argv, "--task") == "1", "rc=%d %s" % (rc, out[-300:]))
    rc, out, argv = run({"SKIP_CONDA": "1"})
    check("G5 MANIFEST unset: refused, python never invoked", rc != 0 and "MANIFEST" in out and argv == [])
    rc, out, argv = run({"SKIP_CONDA": "1", "MANIFEST": os.path.join(fx, "nope.csv")})
    check("G5' MANIFEST missing: refused", rc != 0 and "does not exist" in out and argv == [])
    rc, out, argv = run({"SKIP_CONDA": "1", "MANIFEST": man}, code_dir=os.path.join(fx, "nope"))
    check("G5'' wrong H01_CODE: refused", rc != 0 and "not h01_code" in out and argv == [])
    os.remove(os.path.join(code, "passive_params.csv"))
    rc, out, argv = run({"SKIP_CONDA": "1", "MANIFEST": man})
    check("G5''' missing passive table: refused before python", rc != 0 and "passive table missing" in out and argv == [])
    # ---- G6: the PASSIVE_TABLE / OUT_DIR knobs (decision D-003) -------------
    open(os.path.join(code, "passive_params.csv"), "w").close()   # restore for G6
    open(os.path.join(code, "passive_params_inh_SST.csv"), "w").close()
    rc, out, argv = run({"SKIP_CONDA": "1", "MANIFEST": man})
    check("G6 defaults: --passive-table <code>/passive_params.csv, --out-dir <root>/p1",
          rc == 0 and argval(argv, "--passive-table") == os.path.join(code, "passive_params.csv")
          and argval(argv, "--out-dir") == os.path.join(root, "p1"), str(argv))
    rc, out, argv = run({"SKIP_CONDA": "1", "MANIFEST": man,
                         "PASSIVE_TABLE": "passive_params_inh_SST.csv", "OUT_DIR": "p1_inh_SST"})
    check("G6a relative PASSIVE_TABLE resolves against H01_CODE, relative OUT_DIR against H01_ROOT",
          rc == 0 and argval(argv, "--passive-table") == os.path.join(code, "passive_params_inh_SST.csv")
          and argval(argv, "--out-dir") == os.path.join(root, "p1_inh_SST")
          and "passive passive_params_inh_SST.csv | out " + os.path.join(root, "p1_inh_SST") in out,
          "rc=%d %s" % (rc, out[-300:]))
    abs_tab = os.path.join(fx, "elsewhere.csv")
    open(abs_tab, "w").close()
    rc, out, argv = run({"SKIP_CONDA": "1", "MANIFEST": man,
                         "PASSIVE_TABLE": abs_tab, "OUT_DIR": os.path.join(fx, "abs_out")})
    check("G6b absolute PASSIVE_TABLE and OUT_DIR pass through unchanged",
          rc == 0 and argval(argv, "--passive-table") == abs_tab
          and argval(argv, "--out-dir") == os.path.join(fx, "abs_out"), str(argv))
    rc, out, argv = run({"SKIP_CONDA": "1", "MANIFEST": man, "PASSIVE_TABLE": "nope.csv"})
    check("G6c a missing PASSIVE_TABLE is refused before python, naming the resolved path",
          rc != 0 and "passive table missing" in out and "nope.csv" in out and argv == [])


# --------------------------------------------------------------------------- #
def main():
    import run_p1_export as R
    import build_p1_manifest as B
    import p1_spine_stats as pss
    import p1_hoc_audit as pha
    import spine_density as sd

    tmp = tempfile.mkdtemp(prefix="p1smoke_")
    for d in ("b", "c", "d", "e", "f", "g"):
        os.makedirs(os.path.join(tmp, d), exist_ok=True)
    try:
        print("== A passive table")
        section_a(R, tmp)
        print("== B manifest")
        section_b(R, B, os.path.join(tmp, "b"))
        print("== C spine tables (fixture)")
        section_c(pss, sd, os.path.join(tmp, "c"))
        print("== D end to end")
        root, mp, ptab = section_d(R, B, os.path.join(tmp, "d"))
        print("== E audit and quarantine")
        section_e(R, pha, root, mp, ptab, os.path.join(tmp, "e"))
        print("== F refusals")
        section_f(R, root, mp, os.path.join(tmp, "f"))
        print("== G job script")
        section_g(os.path.join(tmp, "g"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("\n%d checks passed, %d failed" % (len(RESULTS) - n_fail, n_fail))
    print("ALL GREEN" if n_fail == 0 else "FAILURES")
    return n_fail


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
