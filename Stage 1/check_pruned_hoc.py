"""check_pruned_hoc -- is the SHAFT intact after spine pruning?

QUESTION
--------
Spines are removed from the skeleton by label (morphology_exporter step 8,
prune_spines). The claim to verify: removing them leaves every shaft node in
place with its radius unchanged, so the spine-deprived arbour used in the
simulation is the same cable, minus spines, and nothing has been shrunk or
truncated.

WHAT IS COMPARED, AND AGAINST WHAT
----------------------------------
The emitted neuron_{nid}_aligned.hoc is parsed back into points and compared
with the frame it was written from, rebuilt here by the SAME exporter call
(export_neuron with write_files=False, return_frames=True, align_fn=None):

  A. raw CSV  -> labelled frame : which nodes changed position or radius
                                  before pruning (expected: the soma only,
                                  via soma_enforce)
  B. labelled -> pruned frame   : which nodes were removed, split into
                                  spine-classified and non-spine descendants
  C. pruned   -> .hoc           : every pruned-frame node appears in the hoc
                                  at its aligned position with diam = 2r,
                                  no extra points, and the total cable
                                  length matches the pruned tree's edges

C is exact: the check applies alignment.aligned_um with the soma_pos and
mean_matrix recorded in neuron_{nid}_alignment.json, which is bit-for-bit the
transform write_hoc used, and %r-formatted floats round-trip exactly.

SHAFT CONTINUATIONS
-------------------
CORRECTION to an earlier version of this module: "pruned component longer than
the spine threshold" is NOT a usable detector and is structurally always zero.
spine_labeller's criterion is literally

    if 0 < total_subtree_length <= spine_length_threshold_nm:

so no labelled spine can exceed the threshold, by construction. On neuron
15543554616 the largest pruned components sit at 3694, 3670, 3649 nm against a
4000 nm threshold -- pressed against the ceiling, which is the signature, not
the count. The column is retained for the record and reported as vacuous.

The decision is shaft_continuation's, on two observables that must BOTH agree
and neither of which is length: calibre rho = r(root)/r(branch point) >=
RHO_SHAFT_MIN, and collinearity cos >= COS_SHAFT_MIN. score_continuations()
below runs that scoring on the SAME labelled frame the export used and reports
what would be demoted, including the resulting shift in F -- without changing
the bank. The exporter (v1.1.0) does not run the correction.

Pure ASCII, LF only.
"""

import json
import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

MODULE_VERSION = "check_pruned_hoc v1.2"
NM_PER_UM = 1000.0
TOL_UM = 1e-6            # 1e-3 nm: only repr round-trip noise is allowed


# --------------------------------------------------------------------------- #
# .hoc parsing                                                                 #
# --------------------------------------------------------------------------- #
_RE_SEC = re.compile(r"^\s*([A-Za-z_]\w*)\[(\d+)\]\s*\{\s*$")
_RE_PT = re.compile(r"^\s*pt3dadd\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)\s*$")
_RE_CONNECT = re.compile(
    r"^\s*connect\s+([A-Za-z_]\w*)\[(\d+)\]\(0\)\s*,\s*([A-Za-z_]\w*)\[(\d+)\]\(1\)\s*$")


def parse_hoc(path):
    """Sections in file order: name, points (N,4) [x,y,z,diam] um, parent."""
    sections, parents, cur = [], {}, None
    with open(path) as fh:
        for line in fh:
            m = _RE_CONNECT.match(line)
            if m:
                parents["%s[%s]" % (m.group(1), m.group(2))] = \
                    "%s[%s]" % (m.group(3), m.group(4))
                continue
            m = _RE_SEC.match(line)
            if m:
                cur = {"name": "%s[%s]" % (m.group(1), m.group(2)),
                       "array": m.group(1), "points": []}
                sections.append(cur)
                continue
            if cur is not None:
                m = _RE_PT.match(line)
                if m:
                    cur["points"].append([float(m.group(i)) for i in (1, 2, 3, 4)])
                elif line.strip() == "}":
                    cur = None
    for s in sections:
        s["points"] = np.asarray(s["points"], dtype=float).reshape(-1, 4)
        s["parent"] = parents.get(s["name"])
    return sections


def is_one_node_section(pts):
    """write_hoc emits a single node as (x,y,z-r,2r),(x,y,z+r,2r)."""
    if len(pts) != 2:
        return False
    a, b = pts
    return (abs(a[0] - b[0]) < TOL_UM and abs(a[1] - b[1]) < TOL_UM
            and abs(a[3] - b[3]) < TOL_UM
            and abs((b[2] - a[2]) - a[3]) < 10 * TOL_UM)


def hoc_nodes_and_cable(sections):
    """Unique node points (x,y,z,diam) and the cable length (um) that
    corresponds to skeleton EDGES: polyline lengths, minus the artificial
    2r of every one-node section, with the parent repeat contributing the
    real parent->child edge exactly once."""
    pts, cable, n_one = [], 0.0, 0
    for s in sections:
        P = s["points"]
        if len(P) == 0:
            continue
        if is_one_node_section(P):
            a, b = P
            pts.append([a[0], a[1], 0.5 * (a[2] + b[2]), a[3]])
            n_one += 1
            continue
        pts.extend(P.tolist())
        cable += float(np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1).sum())
    pts = np.asarray(pts, dtype=float).reshape(-1, 4)
    key = np.round(pts / TOL_UM).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(first)], cable, n_one


# --------------------------------------------------------------------------- #
# Frames                                                                       #
# --------------------------------------------------------------------------- #
def rebuild_frames(df_raw, nid, mx, label_fn, threshold_nm, export_kw=None):
    """The exporter's own labelled and pruned frames, raw nm, no alignment.

    export_kw MUST carry every keyword the committed .hoc was exported with --
    `cap_tips`, `demote_continuations`, `continuation_kw`. They change the
    PARTITION, so rebuilding without them compares a corrected .hoc against an
    uncorrected frame: on neuron 15543554616 that reported 1566 spurious
    "extra" points and a 470 um cable excess, which were the restored
    continuation branches, not a defect in the bank."""
    tmp = tempfile.mkdtemp()
    try:
        res = mx.export_neuron(df_raw, nid, tmp, label_fn=label_fn,
                               spine_length_threshold_nm=threshold_nm,
                               align_fn=None, write_files=False,
                               return_frames=True, verbose=False,
                               **(export_kw or {}))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if "frames" not in res:
        raise RuntimeError("export_neuron returned no frames (qc=%s: %s)"
                           % (res.get("qc_status"), res.get("reasons")))
    labelled = res["frames"]["labelled"]
    pruned, info = mx.prune_spines(labelled)
    return labelled, pruned, info, res


def load_transform(hoc_dir, nid):
    """(soma_pos_nm, mean_matrix) from neuron_{nid}_alignment.json, or the
    identity when the file is absent (an unaligned export)."""
    p = os.path.join(hoc_dir, "neuron_%s_alignment.json" % nid)
    if not os.path.isfile(p):
        return np.zeros(3), np.eye(3), False
    with open(p) as fh:
        a = json.load(fh).get("alignment", {})
    return (np.asarray(a["soma_pos_nm"], dtype=float),
            np.asarray(a["mean_matrix"], dtype=float), True)


def frame_diff(a, b, cols=("x", "y", "z", "r")):
    """Nodes present in both frames whose (cols) differ; and ids only in a."""
    ia, ib = a.set_index("id"), b.set_index("id")
    common = ia.index.intersection(ib.index)
    da = ia.loc[common, list(cols)].to_numpy(float)
    db = ib.loc[common, list(cols)].to_numpy(float)
    changed = np.any(np.abs(da - db) > 1e-9, axis=1)
    return (pd.DataFrame({"id": common[changed],
                          **{c + "_before": da[changed, i] for i, c in enumerate(cols)},
                          **{c + "_after": db[changed, i] for i, c in enumerate(cols)}}),
            sorted(set(ia.index) - set(ib.index)))


def pruned_components(labelled, pruned, cls_spine, class_column="compartment_class"):
    """One row per removed component: size, path length, raw annotations."""
    removed = set(labelled["id"]) - set(pruned["id"])
    if not removed:
        return pd.DataFrame(columns=["root_id", "base_id", "n_nodes",
                                     "n_non_spine_class", "path_len_nm",
                                     "raw_types"])
    L = labelled.set_index("id")
    par = L["p"].to_dict()
    xyz = L[["x", "y", "z"]].to_dict("index")
    kids = {}
    for i, p in par.items():
        kids.setdefault(p, []).append(i)
    roots = [i for i in removed if par[i] not in removed]
    rows = []
    for r in roots:
        comp, stack, dist = [], [r], {r: 0.0}
        while stack:
            c = stack.pop()
            comp.append(c)
            for k in kids.get(c, ()):
                if k in removed:
                    a, b = xyz[c], xyz[k]
                    dist[k] = dist[c] + float(np.sqrt(
                        (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2
                        + (a["z"] - b["z"]) ** 2))
                    stack.append(k)
        cls = L.loc[comp, class_column].astype(str)
        raw = L.loc[comp, "annotated_type_raw"].astype(str) \
            if "annotated_type_raw" in L.columns else pd.Series([], dtype=str)
        rows.append({"root_id": int(r), "base_id": int(par[r]),
                     "n_nodes": len(comp),
                     "n_non_spine_class": int((cls != cls_spine).sum()),
                     "path_len_nm": float(max(dist.values())),
                     "raw_types": ",".join(sorted(set(raw))) if len(raw) else ""})
    return pd.DataFrame(rows).sort_values("path_len_nm", ascending=False)


# --------------------------------------------------------------------------- #
# The check                                                                    #
# --------------------------------------------------------------------------- #
def check_pruned_hoc(nid, df_raw, hoc_dir, mx, al, nc, label_fn,
                     threshold_nm=None, export_kw=None):
    """Returns a report dict; report['ok'] is the verdict.

    export_kw: the SAME keywords CELL 6 passed to align_and_export. Omitting
    them silently compares two different partitions -- see rebuild_frames."""
    from scipy.spatial import cKDTree

    threshold_nm = (float(mx.SPINE_LENGTH_THRESHOLD_NM) if threshold_nm is None
                    else float(threshold_nm))
    hoc_path = os.path.join(hoc_dir, "neuron_%s_aligned.hoc" % nid)
    if not os.path.isfile(hoc_path):
        raise FileNotFoundError(hoc_path)

    raw = df_raw.copy()
    raw["annotated_type_raw"] = raw["annotated_type"].astype(str)
    labelled, pruned, info, res = rebuild_frames(df_raw, nid, mx, label_fn,
                                                 threshold_nm, export_kw)
    cont = res.get("continuation_report") or {}
    labelled = labelled.merge(raw[["id", "annotated_type_raw"]], on="id", how="left")
    pruned = pruned.merge(raw[["id", "annotated_type_raw"]], on="id", how="left")

    # A. raw -> labelled
    changed_A, lost_A = frame_diff(raw, labelled)
    soma_ids = set(labelled.loc[labelled["compartment_class"].astype(str)
                                == nc.CLS_SOMA, "id"].astype(int))
    changed_A["is_soma"] = changed_A["id"].astype(int).isin(soma_ids)

    # B. labelled -> pruned
    comps = pruned_components(labelled, pruned, nc.CLS_SPINE)
    n_removed = len(labelled) - len(pruned)
    non_spine_removed = int(comps["n_non_spine_class"].sum()) if len(comps) else 0

    # C. pruned -> hoc
    soma_pos, M, aligned = load_transform(hoc_dir, nid)
    exp_xyz = al.aligned_um(pruned[["x", "y", "z"]].to_numpy(float), soma_pos, M)
    exp_d = 2.0 * pruned["r"].to_numpy(float) / NM_PER_UM
    sections = parse_hoc(hoc_path)
    hoc_pts, hoc_cable, n_one = hoc_nodes_and_cable(sections)

    tree = cKDTree(hoc_pts[:, :3])
    d_near, j = tree.query(exp_xyz, k=1)
    found = d_near <= TOL_UM
    diam_bad = found & (np.abs(hoc_pts[j, 3] - exp_d) > TOL_UM)
    tree_e = cKDTree(exp_xyz)
    d_back, _ = tree_e.query(hoc_pts[:, :3], k=1)
    extra = d_back > TOL_UM

    P = pruned.set_index("id")
    par = pruned["p"].to_numpy(np.int64)
    has_par = np.isin(par, pruned["id"].to_numpy(np.int64))
    a = pruned[["x", "y", "z"]].to_numpy(float)[has_par]
    b = P.loc[par[has_par], ["x", "y", "z"]].to_numpy(float)
    exp_cable = float(np.linalg.norm(a - b, axis=1).sum()) / NM_PER_UM

    missing = pruned.loc[~found, ["id", "p", "compartment_class",
                                  "annotated_type_raw", "r"]]
    shrunk = pruned.loc[diam_bad, ["id", "r"]].copy()
    if diam_bad.any():
        shrunk["diam_hoc_um"] = hoc_pts[j[diam_bad], 3]
        shrunk["diam_expected_um"] = exp_d[diam_bad]

    ok = (not missing.size and not extra.any() and not diam_bad.any()
          and abs(hoc_cable - exp_cable) < 1e-4 and non_spine_removed == 0)
    return {
        "ok": bool(ok), "nid": nid, "hoc": hoc_path, "aligned": aligned,
        "export_kw": dict(export_kw or {}),
        "continuation_applied": bool(cont.get("applied")),
        "n_continuations_demoted": cont.get("n_demoted"),
        "n_rescued_by_taper": cont.get("n_rescued_by_taper"),
        "n_continuation_undecidable": cont.get("n_undecidable"),
        "module_version": MODULE_VERSION,
        "exporter_version": getattr(mx, "MODULE_VERSION", None),
        "n_raw": int(len(raw)), "n_labelled": int(len(labelled)),
        "n_pruned_frame": int(len(pruned)), "n_removed": int(n_removed),
        "n_spines_removed": int(info["n_spines"]),
        "n_non_spine_nodes_removed": non_spine_removed,
        "n_hoc_sections": len(sections), "n_hoc_unique_points": int(len(hoc_pts)),
        "n_one_node_sections": n_one,
        "n_missing_in_hoc": int((~found).sum()), "n_extra_in_hoc": int(extra.sum()),
        "n_diam_mismatch": int(diam_bad.sum()),
        "cable_hoc_um": hoc_cable, "cable_expected_um": exp_cable,
        "cable_diff_um": hoc_cable - exp_cable,
        "n_changed_raw_to_labelled": int(len(changed_A)),
        "n_changed_non_soma": int((~changed_A["is_soma"]).sum()),
        "ids_lost_raw_to_labelled": lost_A,
        "changed_raw_to_labelled": changed_A,
        "missing": missing, "shrunk": shrunk,
        "extra_points_um": hoc_pts[extra],
        "pruned_components": comps,
        "n_pruned_components_over_threshold": int(
            (comps["path_len_nm"] > threshold_nm).sum()) if len(comps) else 0,
        "pruned_component_max_nm": (float(comps["path_len_nm"].max())
                                    if len(comps) else 0.0),
        "frac_components_near_ceiling": (
            float((comps["path_len_nm"] > 0.9 * threshold_nm).mean())
            if len(comps) else 0.0),
        "threshold_nm": threshold_nm,
        "labelled_frame": labelled,
    }


def score_continuations(labelled, sk_shc, sd, cutoff_um=60.0, nid=None):
    """What shaft_continuation would demote, and what that does to F.

    Runs score_spine_roots on the exporter's OWN labelled frame, then rebuilds
    phi on the corrected frame with spine_density and reports both F values.
    Nothing is written and the bank is untouched: this quantifies the open
    item rather than acting on it.

    The demotion restores each shaft-like component's nodes to the label its
    BASE carries, matching shaft_continuation.demote_shaft_continuations'
    own rule, so a component off an apical stays apical.
    """
    table, report = sk_shc.score_spine_roots(labelled, spine_density=sd)
    out = {"n_spine_roots": report["n_spine_roots"],
           "n_shaft_like": report["n_shaft_like"],
           "method": report["method"], "use_radius": report["use_radius"],
           "rho_shaft_min": report["rho_shaft_min"],
           "cos_shaft_min": report["cos_shaft_min"],
           "n_ambiguous_bp": len(report.get("ambiguous_branch_points", []) or []),
           "table": table}
    phi0 = sd.build_phi(labelled, nid=nid, input_units="nm")
    out["F_lit_as_exported"] = float(
        sd.cell_f_beyond_cutoff(phi0, cutoff_um=cutoff_um, by="d_from_um")["F"])
    out["A_spine_um2_as_exported"] = float(phi0["spine_area_um2"].sum())
    out["A_shaft_um2_as_exported"] = float(phi0["shaft_area_um2"].sum())
    if not len(table) or not int(report["n_shaft_like"]):
        out.update({"F_lit_corrected": out["F_lit_as_exported"],
                    "n_nodes_demoted": 0, "A_moved_um2": 0.0,
                    "dF_lit": 0.0, "corrected": labelled})
        return out

    roots = [int(r) for r in table.loc[table["is_shaft"], "root"]]
    corr, n_nodes = demote_roots(labelled, roots, sd)
    phi1 = sd.build_phi(corr, nid=nid, input_units="nm")
    out.update({
        "n_nodes_demoted": n_nodes,
        "F_lit_corrected": float(sd.cell_f_beyond_cutoff(
            phi1, cutoff_um=cutoff_um, by="d_from_um")["F"]),
        "A_spine_um2_corrected": float(phi1["spine_area_um2"].sum()),
        "A_shaft_um2_corrected": float(phi1["shaft_area_um2"].sum()),
        "corrected": corr})
    out["A_moved_um2"] = out["A_spine_um2_as_exported"] - out["A_spine_um2_corrected"]
    out["dF_lit"] = out["F_lit_corrected"] - out["F_lit_as_exported"]
    return out


def demote_roots(labelled, roots, sd):
    """Relabel each root's whole subtree to its BASE's label (a shaft label,
    else 'dendrite') -- shaft_continuation.demote_shaft_continuations' own
    rule. Returns (frame, n_nodes_relabelled). Coordinates untouched."""
    import re

    corr = labelled.copy()
    col = corr.columns.get_loc("annotated_type")
    ids = corr["id"].to_numpy(dtype=np.int64)
    par = corr["p"].to_numpy(dtype=np.int64)
    pos = {int(v): i for i, v in enumerate(ids)}
    kids = {}
    for i, p in enumerate(par):
        kids.setdefault(int(p), []).append(i)
    shaft_re = re.compile(sd.SHAFT_REGEX, re.IGNORECASE)
    n_nodes = 0
    for r in roots:
        i = pos[int(r)]
        base = int(par[i])
        btype = str(corr.iloc[pos[base], col]) if base in pos else ""
        new = btype if shaft_re.search(btype) else "dendrite"
        stack = [i]
        while stack:
            j = stack.pop()
            corr.iloc[j, col] = new
            n_nodes += 1
            stack.extend(kids.get(int(ids[j]), []))
    return corr, n_nodes


def print_continuations(sc, n_show=8):
    print("D. shaft continuations (shaft_continuation, NOT run by the exporter):")
    print("   %d spine-component root(s) scored by %s (rho >= %.2f, cos >= %.2f)"
          " -> %d shaft-like, %d ambiguous branch point(s)"
          % (sc["n_spine_roots"], sc["method"], sc["rho_shaft_min"],
             sc["cos_shaft_min"], sc["n_shaft_like"], sc["n_ambiguous_bp"]))
    if sc["n_shaft_like"]:
        t = sc["table"]
        print(t.loc[t["is_shaft"], ["root", "bp", "rho", "cos", "own_len_nm",
                                    "category", "reason"]].head(n_show)
              .to_string(index=False))
    print("   demoting them would move %.1f um2 (%.1f%% of spine area) from the "
          "spine bucket to the shaft, and F_lit from %.4f to %.4f (%+.4f)"
          % (sc["A_moved_um2"],
             100 * sc["A_moved_um2"] / max(sc["A_spine_um2_as_exported"], 1e-30),
             sc["F_lit_as_exported"], sc["F_lit_corrected"], sc["dF_lit"]))


def print_report(rep, n_show=8):
    print("%s | exporter %s | neuron %s | %s"
          % (rep["module_version"], rep["exporter_version"], rep["nid"],
             "ALIGNED" if rep["aligned"] else "unaligned"))
    if rep["continuation_applied"]:
        print("   rebuilt WITH the three-vote correction: %d demoted, %d held "
              "back by the taper, %d too short"
              % (rep["n_continuations_demoted"], rep["n_rescued_by_taper"],
                 rep["n_continuation_undecidable"]))
    elif rep["export_kw"]:
        print("   rebuilt with export_kw=%s" % rep["export_kw"])
    else:
        print("   rebuilt with NO export keywords. If the .hoc was exported "
              "with demote_continuations=True or cap_tips=True, section C "
              "below is comparing two different partitions and its counts are "
              "meaningless. Pass export_kw.")
    print("A. raw -> labelled: %d node(s) changed, %d of them non-soma, %d lost"
          % (rep["n_changed_raw_to_labelled"], rep["n_changed_non_soma"],
             len(rep["ids_lost_raw_to_labelled"])))
    if rep["n_changed_raw_to_labelled"]:
        print(rep["changed_raw_to_labelled"].head(n_show).to_string(index=False))
    print("B. labelled -> pruned: %d node(s) removed in %d spine(s); "
          "%d removed node(s) NOT spine-classified. Longest pruned component "
          "%.0f nm against the %.0f nm labeller threshold (that count is "
          "structurally 0 -- see the module docstring; use section D)"
          % (rep["n_removed"], rep["n_spines_removed"],
             rep["n_non_spine_nodes_removed"],
             rep["pruned_component_max_nm"], rep["threshold_nm"]))
    c = rep["pruned_components"]
    if len(c):
        print("   largest pruned components:")
        print(c.head(n_show).to_string(index=False))
    print("C. pruned -> hoc: %d sections, %d unique points vs %d frame nodes | "
          "missing %d, extra %d, diameter mismatches %d | cable hoc %.3f um, "
          "expected %.3f um (diff %.2e)"
          % (rep["n_hoc_sections"], rep["n_hoc_unique_points"],
             rep["n_pruned_frame"], rep["n_missing_in_hoc"],
             rep["n_extra_in_hoc"], rep["n_diam_mismatch"], rep["cable_hoc_um"],
             rep["cable_expected_um"], rep["cable_diff_um"]))
    if rep["n_missing_in_hoc"]:
        print(rep["missing"].head(n_show).to_string(index=False))
    if rep["n_diam_mismatch"]:
        print(rep["shrunk"].head(n_show).to_string(index=False))
    print("VERDICT: %s" % ("shaft intact -- every non-spine node present, "
                           "radii unchanged, cable length conserved"
                           if rep["ok"] else "PROBLEM -- see above"))
