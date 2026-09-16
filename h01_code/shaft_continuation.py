"""shaft_continuation -- demote terminal shaft stubs wrongly labelled as spines.

MODULE_VERSION = "shaft_continuation-1.1.0"

THE DEFECT
----------
spine_labeller.label_dendritic_spines_robust (the verbatim extraction, lines
1653-1825 of morpholgy_pathways__6_.py) classifies a side branch as a spine by
subtree length alone:

    for bp in branch_points:                      # dendrite/apical only
        for child in G.successors(bp):
            if 0 < subtree_length(child) <= threshold:
                spine_nodes.update(subtree(child))

`G.successors(bp)` includes the child that CONTINUES the parent dendrite. At the
last branch point of an arbour the continuation is a short terminal stub, so it
falls under the threshold and is relabelled head/neck. It is shaft, not spine.

Consequences, all signed:
  * the spine count per cell is inflated by roughly one sigma per terminal stub
    below the threshold;
  * shaft membrane area is attributed to the spine bucket, so
    F = 1 + A_spine/A_shaft is biased UPWARD -- numerator up, denominator down;
  * in the mesh pipeline the affected sigma enter the G0 denominator and would
    be handed to the Pappus construction as if they were spines.

WHY THIS IS A SEPARATE MODULE
-----------------------------
spine_labeller.py is a byte-faithful copy whose SOURCE_SHA256 is checked against
the notebook export. Editing it would destroy that correspondence and hide which
definition produced a given bank. The correction is therefore applied AFTER the
labeller, from the labeller's own output, and is switchable and auditable.

THE CRITERION
-------------
The question is NOT "which of these siblings continues the shaft" -- calibre
separates a thin spine neck from shaft, but it does not separate a stub from a
sister daughter dendrite of similar calibre, and a sibling contest goes
ambiguous exactly where it matters. The question is per component:

    is this spine-labelled subtree a PROTRUSION, or is it shaft?

Two observables answer it, and BOTH must agree before anything is demoted:

  1. CALIBRE. A spine is attached by a neck that is thin relative to its parent
     shaft (roughly 0.05-0.2 um against 0.4-1.0 um). A shaft continuation is of
     comparable calibre.   rho(c) = r(c) / r(bp) >= RHO_SHAFT_MIN
  2. COLLINEARITY. A continuation carries on in the parent's direction; a
     protrusion leaves at an angle.
     cos(c) = <u_in, u_c> >= COS_SHAFT_MIN, where u_in = unit(x_bp - x_p(bp))
     and u_c = unit(x_c - x_bp)

  3. TOPOLOGY -- RECORDED ONLY, NEVER PART OF THE DECISION.
     Every labelled component root sits at a branch point BY CONSTRUCTION:
     the labeller iterates

         branch_points = [n for n in G.nodes() if G.out_degree(n) > 1]

     so a one-child parent is never examined and its child is never labelled.
     out_degree(bp) >= 2 therefore holds for EVERY labelled root, spine and
     stub alike, and a "1 child means tip, 2 means spine" test can never fire.
     A corollary worth stating: a plain unbranched dendrite tail is never
     mislabelled at all -- the defect requires the tail to have a sibling, and
     that sibling is usually the very spine that created the branch point.

     What does discriminate is the number of UNLABELLED siblings, which is the
     same question asked in the labeller's own terms: an unlabelled sibling IS
     the continuation that survived labelling.

       n_unlabelled_siblings == 0 -> TERMINAL ZONE. The parent process has no
         surviving continuation, so its ending is among these children.
         Demotion here is category "shaft_ending" -- the original defect.
       n_unlabelled_siblings >= 1 -> the process carries on through one of
         them, so no child here is its ending. A demotion is then category
         "short_terminal_branch": a short daughter dendrite that is also not a
         spine, but a different claim.

     `sibling_continues` (max sibling cable length > CONTINUATION_THRESHOLD_NM)
     asks the same thing from geometry and is kept as an independent
     cross-check; `n_category_disagreements` counts where the two differ.

     These three fields LABEL what was demoted; they do not decide it. The
     decision is rho AND cos, and nothing else. They were briefly wired into
     the decision as a veto and that was reverted: the defect touches a small
     fraction of segments, so a third gate buys little and adds a way for the
     correction to behave differently on tissue than it did in test.

     Note this identifies WHETHER a shaft ending is present among the children,
     not WHICH child it is -- in the terminal zone both the spine and the stub
     score False, and calibre and collinearity still make the assignment.

Requiring both of the first two is deliberately conservative and fails safe: a stub wrongly kept
is a known, quantified bias, while a spine wrongly deleted is unrecoverable.

Calibre is skipped when the cell's radii are not trustworthy -- H01 radii are
Kimimaro DBF values and a cell sitting at the 50 nm fallback carries no calibre
information at all (phi_pipeline_colab.radius_report is the authoritative gate).
In that case collinearity alone decides, at a STRICTER threshold
(COS_SHAFT_MIN_NORADIUS), because it is then the only evidence.

At most ONE component is demoted per branch point: only one child can continue
the parent. If two candidates both pass and their scores are within
TIE_MARGIN, the branch point is reported ambiguous and NEITHER is demoted.

WHAT IS DEMOTED
---------------
Only a spine component whose root is the continuation child of its branch point.
Its nodes are restored to their PRE-LABELLING annotated_type, taken from the
frame the labeller was given -- not guessed.

Pure computation. No network, no plotting, no I/O beyond what the caller passes.
ASCII only, LF only.

CHANGELOG (bump this whenever the public signature or output columns change --
a stale copy on Drive should be catchable by comparing this string, not by
waiting for a TypeError three calls deep):
  1.0.0  first delivery. demote_shaft_continuations(df_before, df_after,
         **kwargs) -> (frame, report). Decision: rho AND cos only.
  1.1.0  added continuation_threshold_nm and three descriptive-only fields
         (n_unlabelled_siblings, sibling_continues, category) to
         score_spine_roots's output table and report. A require_terminal_bp
         veto was added and then REMOVED in the same version bump -- the
         decision is still rho AND cos only; these fields label a demotion
         after the fact, they never gate one (T14 in the smoke suite is the
         regression test for that). Also: demote_shaft_continuations now
         raises a diagnostic TypeError, naming this constant, if the loaded
         score_spine_roots does not accept a keyword the caller passed --
         the signature this file expects moved out from under an older copy
         is exactly the failure this changelog exists to make diagnosable.
"""
from __future__ import annotations

import inspect
import re
import sys
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

MODULE_VERSION = "shaft_continuation-1.1.0"
# and the resolved values are echoed in every report.
DEFAULT_SHAFT_REGEX = r"dendrite|apical|^1$"
DEFAULT_SPINE_LABELS = ("spine", "head", "neck")
DEFAULT_RADIUS_NM = 50.0

# A child is shaft-like when its calibre is at least this fraction of the
# parent's. Spine necks sit far below; continuations sit near 1.
RHO_SHAFT_MIN = 0.50
# ...and when it leaves within this cosine of the parent's direction (0.70 is
# about 45 degrees).
COS_SHAFT_MIN = 0.70
# When radii are unusable, collinearity is the only evidence, so demand more.
COS_SHAFT_MIN_NORADIUS = 0.85
# Two candidates at one branch point closer than this are a tie: demote neither.
TIE_MARGIN = 0.10
# A sibling subtree longer than this means the parent process CONTINUES through
# that sibling, so no child at this branch point is the dendrite's own ending.
# Should match the labeller's spine_length_threshold_nm; the wrapper passes it.
CONTINUATION_THRESHOLD_NM = 4000.0
# Below this fraction of trustworthy radii, calibre carries no information.
MIN_TRUSTWORTHY_RADIUS_FRACTION = 0.80


def _resolve_vocab(spine_density=None, shaft_regex=None, spine_labels=None,
                   default_radius_nm=None) -> Dict[str, Any]:
    return {
        "shaft_regex": shaft_regex or str(
            getattr(spine_density, "SHAFT_REGEX", DEFAULT_SHAFT_REGEX)),
        "spine_labels": tuple(spine_labels or getattr(
            spine_density, "SPINE_LABELS", DEFAULT_SPINE_LABELS)),
        "default_radius_nm": float(default_radius_nm if default_radius_nm is not None
                                   else getattr(spine_density, "DEFAULT_RADIUS_NM",
                                                DEFAULT_RADIUS_NM)),
        "spine_density_version": getattr(spine_density, "MODULE_VERSION", None),
    }


def _topology(df: pd.DataFrame):
    ids = df["id"].to_numpy()
    par = df["p"].to_numpy()
    index = {int(v): i for i, v in enumerate(ids)}
    children: Dict[int, list] = {}
    for i, q in enumerate(par):
        q = int(q)
        if q == -1 or q not in index:
            continue
        children.setdefault(q, []).append(int(ids[i]))
    return ids, par, index, children


def _subtree(root: int, children: Dict[int, list]) -> list:
    out, stack = [], [int(root)]
    while stack:
        u = stack.pop()
        out.append(u)
        stack.extend(children.get(u, ()))
    return out


def _cable_length(root: int, children: Dict[int, list], par, index, xyz) -> float:
    """Total cable length (nm) of the subtree at `root`, edges to parents included."""
    tot = 0.0
    for n in _subtree(root, children):
        q = int(par[index[n]])
        if q in index:
            tot += float(np.linalg.norm(xyz[index[n]] - xyz[index[q]]))
    return tot


def radius_trustworthy(df: pd.DataFrame, default_radius_nm: float = DEFAULT_RADIUS_NM,
                       shaft_regex: str = DEFAULT_SHAFT_REGEX,
                       min_fraction: float = MIN_TRUSTWORTHY_RADIUS_FRACTION
                       ) -> Dict[str, Any]:
    """Can calibre be used on this cell?

    Deliberately weaker and more local than phi_pipeline_colab.radius_report,
    which remains the authoritative gate for quoting any resistance. This asks
    only whether enough radii are distinguishable from the fallback for a
    ratio at a branch point to mean anything. Pass the authoritative verdict in
    via `radius_ok=` if you have it.
    """
    if "r" not in df.columns:
        return {"usable": False, "reason": "no r column", "fraction_real": 0.0}
    r = pd.to_numeric(df["r"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(r) & (r > 0)
    at_default = finite & np.isclose(r, float(default_radius_nm), rtol=0, atol=1e-9)
    n = int(finite.sum())
    frac_real = float((finite & ~at_default).sum() / n) if n else 0.0
    distinct = int(len(np.unique(np.round(r[finite], 6)))) if n else 0
    usable = bool(n and frac_real >= float(min_fraction) and distinct >= 3)
    return {"usable": usable, "fraction_real": frac_real, "n_finite": n,
            "n_distinct": distinct, "min_fraction": float(min_fraction),
            "reason": None if usable else
            ("flat or fallback-dominated radii (%.2f real, %d distinct)"
             % (frac_real, distinct))}


def score_spine_roots(df: pd.DataFrame, *, spine_density=None,
                      shaft_regex: Optional[str] = None,
                      spine_labels: Optional[Iterable[str]] = None,
                      default_radius_nm: Optional[float] = None,
                      radius_ok: Optional[bool] = None,
                      rho_shaft_min: float = RHO_SHAFT_MIN,
                      cos_shaft_min: float = COS_SHAFT_MIN,
                      cos_shaft_min_noradius: float = COS_SHAFT_MIN_NORADIUS,
                      tie_margin: float = TIE_MARGIN,
                      continuation_threshold_nm: float = CONTINUATION_THRESHOLD_NM,
                      annotation_column: str = "annotated_type"
                      ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Score every spine-component ROOT for shaft-likeness.

    A spine-component root is a spine-labelled node whose parent is NOT
    spine-labelled: the point at which the labeller decided a subtree was a
    spine. Returns (table, report), one row per such root:

      root, bp, rho, cos, rho_ok, cos_ok, is_shaft, method, reason

    `is_shaft` True means the component looks like a continuation of the parent
    process and is a candidate for demotion. Ties at a branch point are resolved
    to False for BOTH candidates and the branch point is reported ambiguous.
    """
    voc = _resolve_vocab(spine_density, shaft_regex, spine_labels, default_radius_nm)
    spine_set = {s.lower() for s in voc["spine_labels"]}
    ids, par, index, children = _topology(df)
    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    rad = (pd.to_numeric(df["r"], errors="coerce").to_numpy(dtype=float)
           if "r" in df.columns else np.full(len(df), np.nan))
    lab = df[annotation_column].astype(str).str.lower().to_numpy()
    is_spine = np.isin(lab, list(spine_set))
    shaft_re = re.compile(voc["shaft_regex"], re.IGNORECASE)

    rt = radius_trustworthy(df, voc["default_radius_nm"], voc["shaft_regex"])
    use_radius = bool(rt["usable"] if radius_ok is None else radius_ok)
    cos_min = float(cos_shaft_min if use_radius else cos_shaft_min_noradius)
    method = "calibre+collinearity" if use_radius else "collinearity only"

    rows = []
    for i, nid_ in enumerate(ids):
        if not is_spine[i]:
            continue
        q = int(par[i])
        if q == -1 or q not in index:
            continue
        j = index[q]
        if is_spine[j]:
            continue                       # not a component root
        if not shaft_re.search(str(df[annotation_column].iloc[j])):
            continue                       # parent is not shaft: leave it alone

        rho = float(rad[i] / rad[j]) if (np.isfinite(rad[i]) and np.isfinite(rad[j])
                                         and rad[j] > 0) else float("nan")
        qq = int(par[j])
        cos = float("nan")
        if qq != -1 and qq in index:
            u_in = xyz[j] - xyz[index[qq]]
            u_c = xyz[i] - xyz[j]
            n1, n2 = float(np.linalg.norm(u_in)), float(np.linalg.norm(u_c))
            if n1 > 0 and n2 > 0:
                cos = float(np.dot(u_in / n1, u_c / n2))

        rho_ok = bool(use_radius and np.isfinite(rho) and rho >= float(rho_shaft_min))
        cos_ok = bool(np.isfinite(cos) and cos >= cos_min)
        if use_radius:
            is_shaft = rho_ok and cos_ok
            reason = None if is_shaft else (
                "rho %.3f < %.2f" % (rho, rho_shaft_min) if not rho_ok
                else "cos %.3f < %.2f" % (cos, cos_min))
        else:
            is_shaft = cos_ok
            reason = None if is_shaft else "cos %.3f < %.2f (no calibre)" % (cos, cos_min)

        # TOPOLOGY. Every labelled component root sits at a branch point by
        # construction -- the labeller only ever examines nodes with
        # out_degree > 1 -- so "is there a bifurcation here" is true for spines
        # and stubs alike and discriminates nothing. What DOES discriminate is
        # whether a SIBLING subtree continues past the threshold. If one does,
        # the parent process carries on through that sibling and no child here
        # is the dendrite's own ending; if none does, this branch point is in
        # the terminal zone and a shaft ending is present among these children.
        own_len = _cable_length(int(nid_), children, par, index, xyz)
        sibs = [c for c in children.get(q, []) if c != int(nid_)]
        sib_lens = [_cable_length(c, children, par, index, xyz) for c in sibs]
        sib_max = float(max(sib_lens)) if sib_lens else 0.0
        sibling_continues = bool(sib_max > float(continuation_threshold_nm))
        # The same question in the LABELLER's own terms, which is cheaper and
        # cannot drift from its threshold: an unlabelled sibling IS the
        # unlabelled continuation. Zero of them means the parent process has no
        # continuation that survived labelling, i.e. its ending is among these
        # children. This is the primary signal; sibling_continues is kept as an
        # independent cross-check and a disagreement is counted in the report.
        n_unlabelled_siblings = int(sum(1 for c in sibs if not is_spine[index[c]]))
        all_children_labelled = bool(n_unlabelled_siblings == 0)
        category = "shaft_ending" if all_children_labelled else "short_terminal_branch"
        category_disagreement = bool(all_children_labelled == sibling_continues)

        rows.append({"root": int(nid_), "bp": q, "rho": rho, "cos": cos,
                     "rho_ok": rho_ok, "cos_ok": cos_ok,
                     "own_len_nm": own_len, "sibling_max_len_nm": sib_max,
                     "sibling_continues": sibling_continues,
                     "n_children_of_bp": int(len(children.get(q, []))),
                     "n_unlabelled_siblings": n_unlabelled_siblings,
                     "all_children_labelled": all_children_labelled,
                     "category_disagreement": category_disagreement,
                     "category": category,
                     "is_shaft": bool(is_shaft),
                     "method": method, "reason": reason})

    table = pd.DataFrame(rows, columns=["root", "bp", "rho", "cos", "rho_ok",
                                        "cos_ok", "own_len_nm", "sibling_max_len_nm",
                                        "sibling_continues", "n_children_of_bp",
                                        "n_unlabelled_siblings",
                                        "all_children_labelled",
                                        "category_disagreement", "category",
                                        "is_shaft", "method", "reason"])

    # At most one continuation per branch point; a near-tie demotes neither.
    ambiguous = []
    if len(table):
        for bp, grp in table[table["is_shaft"]].groupby("bp"):
            if len(grp) < 2:
                continue
            key = "rho" if use_radius else "cos"
            vals = grp[key].to_numpy(dtype=float)
            order = np.argsort(vals)[::-1]
            best, second = vals[order[0]], vals[order[1]]
            if not np.isfinite(best - second) or (best - second) < float(tie_margin):
                ambiguous.append(int(bp))
                table.loc[grp.index, "is_shaft"] = False
                table.loc[grp.index, "reason"] = "tie at branch point (margin %.3f)" % (
                    best - second)
            else:
                losers = grp.index[order[1:]]
                table.loc[losers, "is_shaft"] = False
                table.loc[losers, "reason"] = "not the best candidate at this bp"

    report = {
        "module_version": MODULE_VERSION,
        "method": method,
        "use_radius": use_radius,
        "radius_trustworthy": rt,
        "radius_ok_override": radius_ok,
        "rho_shaft_min": float(rho_shaft_min),
        "cos_shaft_min": cos_min,
        "tie_margin": float(tie_margin),
        "continuation_threshold_nm": float(continuation_threshold_nm),
        "vocabulary": voc,
        "n_spine_roots": int(len(table)),
        "n_shaft_like": int(table["is_shaft"].sum()) if len(table) else 0,
        "n_terminal_zone_roots": int(table["all_children_labelled"].sum()) if len(table) else 0,
        "n_category_disagreements": int(table["category_disagreement"].sum()) if len(table) else 0,
        "min_children_of_bp": int(table["n_children_of_bp"].min()) if len(table) else None,
        "ambiguous_branch_points": ambiguous,
    }
    return table, report


def _call_score_spine_roots(df_after: pd.DataFrame, kwargs: Dict[str, Any]):
    """Call score_spine_roots(df_after, **kwargs), turning a signature
    mismatch into a message that names the likely cause instead of a bare
    TypeError three calls deep.

    The one way this fails in practice is a STALE shaft_continuation.py: an
    older copy of THIS FILE sitting on Drive with a score_spine_roots that
    does not yet accept a keyword the caller (also from this file, but
    reloaded fresh) is passing. importlib.reload() picks up whatever bytes are
    actually on disk, so this is a real version mismatch, not a caching
    artifact -- the fix is to overwrite the file, not to reload harder.
    """
    try:
        return score_spine_roots(df_after, **kwargs)
    except TypeError as exc:
        bad = [k for k in kwargs if k not in
              inspect.signature(score_spine_roots).parameters]
        if not bad:
            raise  # a real TypeError unrelated to a stale signature
        loaded_from = getattr(sys.modules.get(__name__), "__file__", "?")
        raise TypeError(
            "score_spine_roots() in the loaded shaft_continuation.py does not "
            "accept %s.\n"
            "Loaded from: %s\n"
            "Loaded MODULE_VERSION: %s\n"
            "This is very likely a STALE COPY of shaft_continuation.py on "
            "Drive -- CELL 5 reloads this module fresh every run, so whatever "
            "is on disk at that path is what ran. Overwrite the file with the "
            "current version (see the CHANGELOG at the top of this module) "
            "and re-run CELL 5. Original error: %s"
            % (bad, loaded_from, MODULE_VERSION, exc)
        ) from exc


def demote_shaft_continuations(df_before: pd.DataFrame, df_after: pd.DataFrame,
                               **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Restore spine components that are really shaft continuations.

    df_before : the frame GIVEN to the labeller (pre-labelling annotated_type)
    df_after  : the frame RETURNED by the labeller (head/neck/spine written in)

    Both must be the same rows in the same order -- the labeller relabels, it
    does not move nodes, and this is checked. Nodes of a demoted component are
    restored to their df_before label, never to a guessed one.

    Returns (corrected_frame, report).
    """
    annotation_column = kwargs.get("annotation_column", "annotated_type")
    if len(df_before) != len(df_after):
        raise ValueError("df_before has %d rows, df_after %d"
                         % (len(df_before), len(df_after)))
    for col in ("id", "p"):
        if not np.array_equal(np.asarray(df_before[col]), np.asarray(df_after[col])):
            raise ValueError("column %r differs between df_before and df_after; "
                             "they are not the same nodes in the same order" % col)

    voc = _resolve_vocab(kwargs.get("spine_density"), kwargs.get("shaft_regex"),
                         kwargs.get("spine_labels"), kwargs.get("default_radius_nm"))
    spine_set = {s.lower() for s in voc["spine_labels"]}

    table, report = _call_score_spine_roots(df_after, kwargs)
    ids, par, index, children = _topology(df_after)
    lab_after = df_after[annotation_column].astype(str).str.lower().to_numpy()
    lab_before = df_before[annotation_column].astype(str).to_numpy()
    is_spine = np.isin(lab_after, list(spine_set))
    xyz = df_after[["x", "y", "z"]].to_numpy(dtype=float)

    out = df_after.copy()
    demoted_rows, demoted_nodes = [], set()
    for row in (table[table["is_shaft"]].itertuples(index=False)
                if len(table) else []):
        c = int(row.root)
        comp = [n for n in _subtree(c, children) if is_spine[index[n]]]
        if not comp:
            continue
        length_nm = 0.0
        for n in comp:
            q = int(par[index[n]])
            if q in index:
                length_nm += float(np.linalg.norm(xyz[index[n]] - xyz[index[q]]))
        demoted_rows.append({
            "root": c, "bp": int(row.bp), "n_nodes": len(comp),
            "length_nm": length_nm, "rho": float(row.rho), "cos": float(row.cos),
            "method": row.method, "category": str(row.category),
            "sibling_max_len_nm": float(row.sibling_max_len_nm),
            "restored_to": str(lab_before[index[c]]),
            "labels_removed": sorted(set(lab_after[index[n]] for n in comp)),
        })
        demoted_nodes.update(comp)

    if demoted_nodes:
        sel = out["id"].isin(list(demoted_nodes)).to_numpy()
        out.loc[sel, annotation_column] = lab_before[sel]

    n_spine_before = int(is_spine.sum())
    lab_final = out[annotation_column].astype(str).str.lower().to_numpy()
    n_spine_after = int(np.isin(lab_final, list(spine_set)).sum())
    report.update({
        "n_components_demoted": len(demoted_rows),
        "n_nodes_demoted": len(demoted_nodes),
        "demoted": demoted_rows,
        "n_spine_nodes_before": n_spine_before,
        "n_spine_nodes_after": n_spine_after,
        "spine_node_reduction_fraction": (
            (n_spine_before - n_spine_after) / n_spine_before) if n_spine_before else 0.0,
        "total_length_demoted_nm": float(sum(d["length_nm"] for d in demoted_rows)),
        "demoted_by_category": {
            "shaft_ending": sum(1 for d in demoted_rows
                                if d["category"] == "shaft_ending"),
            "short_terminal_branch": sum(1 for d in demoted_rows
                                         if d["category"] == "short_terminal_branch"),
        },
    })
    return out, report


# --------------------------------------------------------------------------- #
# Drop-in replacement for Stage 1                                              #
# --------------------------------------------------------------------------- #
def label_dendritic_spines_corrected(neuron_ids, input_dir=None, output_dir=None,
                                     spine_length_threshold_nm=None,
                                     labeller=None, spine_density=None,
                                     correction=True, reports=None,
                                     correction_kwargs=None, **kwargs):
    """Signature-compatible wrapper around label_dendritic_spines_robust.

    Stage 1 can swap this in wherever it currently calls the labeller; the
    return value is the same {nid: DataFrame} mapping. Because
    morphology_exporter.exporter_id records `labeller:<qualname>`, swapping it
    CHANGES the exporter id, so a corrected bank can never be silently mixed
    with an uncorrected one.

    `spine_length_threshold_nm` is required: the extraction's own default is
    5000 nm and the project value is
    morphology_exporter.SPINE_LENGTH_THRESHOLD_NM = 4000 nm.

    `reports`, if a dict, receives {nid: correction report}.
    """
    import os

    if spine_length_threshold_nm is None:
        raise TypeError(
            "spine_length_threshold_nm must be given explicitly; the extracted "
            "labeller defaults to 5000 nm and the project value is 4000 nm")
    if labeller is None:
        import spine_labeller as labeller  # noqa: PLC0415 - optional dependency

    before = {}
    for nid in neuron_ids:
        p = os.path.join(input_dir, "neuron_%s.csv" % nid)
        if os.path.isfile(p):
            before[nid] = pd.read_csv(p)

    # The inner call must NOT write: its output is the uncorrected partition.
    out = labeller.label_dendritic_spines_robust(
        list(neuron_ids), input_dir=input_dir, output_dir=None,
        spine_length_threshold_nm=float(spine_length_threshold_nm), **kwargs)

    corrected = {}
    for nid, df_after in (out or {}).items():
        if not correction or nid not in before:
            corrected[nid] = df_after
            continue
        fixed, rep = demote_shaft_continuations(
            before[nid], df_after, spine_density=spine_density,
            continuation_threshold_nm=float(spine_length_threshold_nm),
            **{k: v for k, v in (correction_kwargs or {}).items()})
        rep["nid"] = nid
        rep["threshold_nm"] = float(spine_length_threshold_nm)
        corrected[nid] = fixed
        if isinstance(reports, dict):
            reports[nid] = rep
        print("[shaft_continuation] neuron %s: demoted %d component(s), %d node(s)"
              " by %s -- %d shaft_ending, %d short_terminal_branch; "
              "%d ambiguous branch point(s)"
              % (nid, rep["n_components_demoted"], rep["n_nodes_demoted"],
                 rep["method"], rep["demoted_by_category"]["shaft_ending"],
                 rep["demoted_by_category"]["short_terminal_branch"],
                 len(rep["ambiguous_branch_points"])))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for nid, df in corrected.items():
            df.to_csv(os.path.join(output_dir, "neuron_%s_spines.csv" % nid),
                      index=False)
    return corrected
