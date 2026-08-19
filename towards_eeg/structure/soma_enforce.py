"""soma_enforce -- stage S1.2: identify the soma, and make it the root.

WHAT THIS MODULE ESTABLISHES
----------------------------
Exactly one somatic section exists and is the topology root (I-15), identified
by DIAMETER AND POSITION rather than by name, with the name-based and
geometry-based identifications compared rather than trusted (I-18). I-18 is
deliberately redundant with I-15 because the failure mode is that one of the
two identifications is wrong, and with a single test you cannot tell which.

WHY THE GEOMETRIC TEST EARNS ITS KEEP -- MEASURED, NOT ASSUMED
--------------------------------------------------------------
Two things, both observed on real skeletons on 23 July 2026:

1. In every file inspected there is exactly ONE node annotated 'Soma', it
   carries file_source='promoted_root', and it IS the topological root
   (p == -1). So the multi-node collapse path below should never fire on this
   bank. It is implemented anyway, with a warning and a diagnostic hook,
   because "should never" is not "cannot".

2. The geometric test catches something else entirely, and it is the more
   valuable of the two. neuron_606394351 has a node annotated 'Soma' whose
   radius is 331.9 nm -- 0.66 um across. That is not a soma; it is a truncated
   arbour fragment with a promoted root. The intact neuron_794820508 has a soma
   radius of 5325.5 nm. So the diameter test is a CELL-COMPLETENESS gate, not
   merely an O8 cross-check. This matters downstream: the fragment yields
   F_lit = 1.64 while the intact cell yields 1.895 against a literature value
   near 1.9.

QC POLICY (decision Q6, closed 23 July 2026): SOFT.
Every anomaly short of structural impossibility yields
qc_status = 'pass_low_confidence' with a machine-readable reason, so the cell
stays in the bank and can be filtered downstream. Only a frame with no root at
all is 'fail', because no tree can be built from it.

SOMA AREA CONVENTION
--------------------
A collapsed single-node soma is emitted by the exporter as a two-point section
spanning z-r to z+r with diam = 2r. Its lateral area is
    pi * diam * L = pi * (2r) * (2r) = 4 * pi * r^2,
i.e. exactly the area of a sphere of radius r. The inherited exporter's
"NEURON needs two points" workaround is therefore already area-preserving for
the soma; this module keeps that convention and makes it explicit rather than
incidental. Multi-node collapse preserves total sphere area by taking
    r_eq = sqrt(sum_i r_i^2).

PLOTTING
--------
This module does no plotting. The pathological-collapse figure required by the
S1.2 spec is produced by passing a callable via plot_fn (dependency injection),
so that the visualisation module can be swapped or omitted without touching
this logic. plot_fn is called as plot_fn(df, report, path).

ASCII only, LF only, no top-level side effects, no I/O of its own.
"""

import math

import numpy as np
import pandas as pd

import node_classify as nc


MODULE_VERSION = "soma_enforce-1.0.0"

# Defaults are PROJECT DECISIONS, not literature values, and are recorded in
# every report so a run can be reproduced or challenged.
#   min_soma_radius_nm: 2000 nm radius = 4 um diameter. Well below any intact
#     human pyramidal soma, well above the 332 nm fragment case above.
#   min_radius_ratio: the soma must be an outlier against the rest of the cell.
#     Measured ratios: 38.0 (intact cell), 2.2 (fragment).
DEFAULT_MIN_SOMA_RADIUS_NM = 2000.0
DEFAULT_MIN_RADIUS_RATIO = 5.0

QC_PASS = "pass"
QC_LOW = "pass_low_confidence"
QC_FAIL = "fail"


# --------------------------------------------------------------------------- #
# Primitive identifications                                                   #
# --------------------------------------------------------------------------- #
def topological_root(df):
    """The node with p == -1. Returns (root_id, n_roots); root_id is None if 0."""
    roots = df.loc[df["p"] == -1, "id"].tolist()
    if not roots:
        return None, 0
    return roots[0], len(roots)


def identify_soma_by_name(df, class_column="compartment_class"):
    """Node ids classified as soma. This is the NAME-based identification."""
    if class_column not in df.columns:
        raise ValueError("frame has no %r column; run classify_frame first"
                         % class_column)
    return df.loc[df[class_column].astype(str) == nc.CLS_SOMA, "id"].tolist()


def identify_soma_by_geometry(df,
                              min_soma_radius_nm=DEFAULT_MIN_SOMA_RADIUS_NM,
                              min_radius_ratio=DEFAULT_MIN_RADIUS_RATIO,
                              class_column="compartment_class",
                              exclude_classes=(nc.CLS_SPINE, nc.CLS_GLIA),
                              origin_tolerance_nm=None):
    """The largest-calibre node, plus the tests that decide whether it is a soma.

    Parameters
    ----------
    min_soma_radius_nm : float
        Absolute floor on the candidate radius.
    min_radius_ratio : float
        The candidate must exceed this multiple of the median radius of the
        remaining nodes, i.e. it must be an outlier and not merely the largest.
    origin_tolerance_nm : float or None
        If set, additionally require the candidate to lie within this distance
        of the origin. Meaningful ONLY on an aligned frame, where
        align_neurons_to_neighborhood has translated the soma to (0, 0, 0).
        None (default) skips the test, so the function is valid pre-alignment.

    Returns
    -------
    dict with candidate_id, candidate_r_nm, radius_ratio, distance_to_origin_nm,
    the three booleans, and 'passed' = their conjunction over the tests that
    were actually applied.
    """
    if "r" not in df.columns:
        return {"candidate_id": None, "candidate_r_nm": float("nan"),
                "radius_ratio": float("nan"), "distance_to_origin_nm": float("nan"),
                "radius_above_floor": False, "radius_is_outlier": False,
                "near_origin": None, "passed": False,
                "reason": "no_radius_column"}

    work = df
    if class_column in df.columns and exclude_classes:
        work = df.loc[~df[class_column].astype(str).isin(list(exclude_classes))]
    if len(work) == 0:
        work = df

    r = work["r"].astype(float).values
    j = int(np.argmax(r))
    cand_id = work["id"].iloc[j]
    cand_r = float(r[j])

    others = np.delete(r, j)
    med_other = float(np.median(others)) if len(others) else float("nan")
    ratio = (cand_r / med_other) if (med_other and med_other > 0) else float("inf")

    row = work.iloc[j]
    d_origin = float(math.sqrt(float(row["x"]) ** 2 + float(row["y"]) ** 2
                               + float(row["z"]) ** 2))

    above_floor = bool(cand_r >= float(min_soma_radius_nm))
    is_outlier = bool(ratio >= float(min_radius_ratio))
    near_origin = None
    if origin_tolerance_nm is not None:
        near_origin = bool(d_origin <= float(origin_tolerance_nm))

    passed = above_floor and is_outlier and (near_origin is not False)
    return {
        "candidate_id": (int(cand_id) if not isinstance(cand_id, str) else cand_id),
        "candidate_r_nm": cand_r,
        "median_other_r_nm": med_other,
        "radius_ratio": float(ratio),
        "distance_to_origin_nm": d_origin,
        "radius_above_floor": above_floor,
        "radius_is_outlier": is_outlier,
        "near_origin": near_origin,
        "passed": bool(passed),
        "min_soma_radius_nm": float(min_soma_radius_nm),
        "min_radius_ratio": float(min_radius_ratio),
    }


# --------------------------------------------------------------------------- #
# Collapse                                                                    #
# --------------------------------------------------------------------------- #
def collapse_soma_nodes(df, soma_ids, class_column="compartment_class"):
    """Collapse several soma-classified nodes into one, preserving sphere area.

    The kept node is the largest-radius member. Its position becomes the
    r^2-weighted centroid of the group and its radius becomes
    r_eq = sqrt(sum_i r_i^2), so that 4*pi*r_eq^2 equals the summed sphere area.
    Children of removed nodes are reparented onto the kept node.

    Returns (df_out, info). If len(soma_ids) <= 1 this is a no-op.
    """
    info = {"n_soma_nodes_in": len(soma_ids), "collapsed": False}
    if len(soma_ids) <= 1:
        info["kept_id"] = (soma_ids[0] if soma_ids else None)
        return df.copy(), info

    out = df.copy()
    sub = out[out["id"].isin(soma_ids)]
    r = sub["r"].astype(float).values
    w = r ** 2
    keep_id = sub["id"].iloc[int(np.argmax(r))]
    r_eq = float(math.sqrt(float(np.sum(w))))
    cx = float(np.average(sub["x"].astype(float).values, weights=w))
    cy = float(np.average(sub["y"].astype(float).values, weights=w))
    cz = float(np.average(sub["z"].astype(float).values, weights=w))

    drop = [i for i in soma_ids if i != keep_id]
    out.loc[out["p"].isin(drop), "p"] = keep_id
    k = out.index[out["id"] == keep_id][0]
    out.at[k, "x"], out.at[k, "y"], out.at[k, "z"] = cx, cy, cz
    out.at[k, "r"] = r_eq
    out = out[~out["id"].isin(drop)].copy()

    info.update({
        "collapsed": True,
        "kept_id": (int(keep_id) if not isinstance(keep_id, str) else keep_id),
        "n_removed": len(drop),
        "r_equivalent_nm": r_eq,
        "centroid_nm": [cx, cy, cz],
        "area_preserved_um2": 4.0 * math.pi * (r_eq / 1000.0) ** 2,
    })
    return out, info


def soma_area_um2(r_nm):
    """Membrane area of the emitted single-node soma section, in um^2.

    Equals the sphere area 4*pi*r^2 under the two-point z-extent convention
    documented in the module docstring.
    """
    r_um = float(r_nm) / 1000.0
    return 4.0 * math.pi * r_um * r_um


# --------------------------------------------------------------------------- #
# The S1.2 entry point                                                        #
# --------------------------------------------------------------------------- #
def enforce_soma(df,
                 nid=None,
                 class_column="compartment_class",
                 min_soma_radius_nm=DEFAULT_MIN_SOMA_RADIUS_NM,
                 min_radius_ratio=DEFAULT_MIN_RADIUS_RATIO,
                 origin_tolerance_nm=None,
                 reroot=True,
                 plot_fn=None,
                 plot_path=None,
                 verbose=True):
    """Identify the soma two ways, compare them, collapse it, make it the root.

    Returns
    -------
    (df_out, report)
        df_out : copy of df with at most one soma-classified node, which is the
                 topological root (p == -1) when reroot is True
        report : JSON-safe dict carrying qc_status, every reason code, both
                 identifications and the collapse info

    Reason codes (all soft except no_root / empty_frame)
        no_root                 no p == -1 node                        -> fail
        multiple_roots          more than one p == -1 node             -> low
        no_soma_labelled_node   nothing classified as soma             -> low
        multiple_soma_nodes     collapse fired                         -> low
        name_geometry_disagree  I-18: the two identifications differ   -> low
        soma_below_radius_floor candidate radius under the floor       -> low
        soma_not_outlier        candidate not an outlier in calibre    -> low
        soma_off_origin         aligned frame, soma far from (0,0,0)   -> low
        soma_not_root           soma node was not the topological root -> low
    """
    report = {
        "module_version": MODULE_VERSION,
        "nid": nid,
        "qc_status": QC_PASS,
        "reasons": [],
        "soma_id": None,
        "soma_r_nm": None,
        "soma_area_um2": None,
        "n_nodes_in": int(len(df)),
        "n_nodes_out": int(len(df)),
    }

    def flag(code):
        if code not in report["reasons"]:
            report["reasons"].append(code)
        if report["qc_status"] == QC_PASS:
            report["qc_status"] = QC_LOW

    if len(df) == 0:
        report["qc_status"] = QC_FAIL
        report["reasons"].append("empty_frame")
        return df.copy(), report

    root_id, n_roots = topological_root(df)
    if root_id is None:
        report["qc_status"] = QC_FAIL
        report["reasons"].append("no_root")
        return df.copy(), report
    if n_roots > 1:
        flag("multiple_roots")
    report["root_id"] = (int(root_id) if not isinstance(root_id, str) else root_id)

    by_name = identify_soma_by_name(df, class_column=class_column)
    by_geom = identify_soma_by_geometry(
        df, min_soma_radius_nm=min_soma_radius_nm,
        min_radius_ratio=min_radius_ratio, class_column=class_column,
        origin_tolerance_nm=origin_tolerance_nm)
    report["by_name_ids"] = [int(i) if not isinstance(i, str) else i
                             for i in by_name]
    report["by_geometry"] = by_geom

    if not by_name:
        flag("no_soma_labelled_node")
    if len(by_name) > 1:
        flag("multiple_soma_nodes")

    # --- I-18: do the two identifications agree? ---------------------------
    cand = by_geom.get("candidate_id")
    if by_name and cand is not None and cand not in by_name:
        flag("name_geometry_disagree")
    if not by_geom.get("radius_above_floor", False):
        flag("soma_below_radius_floor")
    if not by_geom.get("radius_is_outlier", False):
        flag("soma_not_outlier")
    if by_geom.get("near_origin") is False:
        flag("soma_off_origin")

    # --- collapse ----------------------------------------------------------
    out, coll = collapse_soma_nodes(df, by_name, class_column=class_column)
    report["collapse"] = coll
    if coll.get("collapsed"):
        if verbose:
            print("[WARN] neuron %s: %d soma-classified nodes collapsed to one "
                  "(r_eq = %.1f nm). This should not occur on the H01 bank."
                  % (nid, coll["n_soma_nodes_in"], coll["r_equivalent_nm"]))
        if plot_fn is not None and plot_path is not None:
            try:
                plot_fn(df, report, plot_path)
                report["collapse_plot_path"] = str(plot_path)
            except Exception as e:            # noqa: BLE001
                report["collapse_plot_error"] = repr(e)

    # --- choose the soma node ---------------------------------------------
    soma_id = coll.get("kept_id")
    if soma_id is None:
        soma_id = root_id
    if soma_id != root_id:
        flag("soma_not_root")

    # --- re-root -----------------------------------------------------------
    if reroot and soma_id != root_id:
        out = _reroot(out, soma_id)
        report["rerooted"] = True
    else:
        report["rerooted"] = False

    r_soma = float(out.loc[out["id"] == soma_id, "r"].iloc[0]) \
        if "r" in out.columns and (out["id"] == soma_id).any() else float("nan")
    report["soma_id"] = (int(soma_id) if not isinstance(soma_id, str) else soma_id)
    report["soma_r_nm"] = r_soma
    report["soma_area_um2"] = soma_area_um2(r_soma) if r_soma == r_soma else None
    report["n_nodes_out"] = int(len(out))
    return out, report


def _reroot(df, new_root_id):
    """Re-orient parent pointers so new_root_id becomes the p == -1 node.

    Reverses the parent chain from new_root_id up to the old root; every other
    edge keeps its direction. Pure topology, no geometry is touched.
    """
    out = df.copy()
    par = dict(zip(out["id"].tolist(), out["p"].tolist()))
    chain = []
    cur = new_root_id
    seen = set()
    while cur in par and par[cur] != -1 and cur not in seen:
        seen.add(cur)
        chain.append((cur, par[cur]))
        cur = par[cur]
    pos = {v: i for i, v in enumerate(out["id"].tolist())}
    pcol = out.columns.get_loc("p")
    for child, parent in chain:
        out.iat[pos[parent], pcol] = child
    out.iat[pos[new_root_id], pcol] = -1
    return out
