"""spine_geometry -- per-spine geometry and neck axial resistance from H01.

Towards-EEG stage S1 (data-generation side; runs in Colab, not on the HPC).

WHAT THIS MODULE DOES
---------------------
spine_density.build_phi collapses every spine into a scalar area contribution
attributed to its base shaft segment. That is the right representation for the
F-factor path (assumption A1), and it deliberately throws away the spine's own
geometry. This module recovers that geometry, one row per spine, without
touching spine_density's output:

    sigma  ->  L_neck(sigma), A_neck(sigma), A_head(sigma),
               G(sigma) = axial conductance factor of the neck [1/cm],
               R_neck(sigma) = rho_a * G(sigma)               [Ohm]

It is the input to two downstream studies:

  * the analytic sensitivity sweep over R_neck (does the spine neck matter at
    all, for this reconstruction, over the plausible rho_a range?), and
  * the spine-resolved export, which needs real neck and head dimensions to
    emit two sections per spine.

It is a PURE READ of the labelled frame. It writes nothing, prunes nothing, and
changes no existing output. Nothing in the current export path depends on it.

THE AXIAL RESISTANCE, AND WHY NOT THE CYLINDER FORMULA
------------------------------------------------------
The literature (Eyal et al. 2018; Harnett et al. 2012; Acker et al. 2016) uses
the uniform-cylinder form

    R_neck = 4 rho_a L_neck / (pi d_neck^2)                            (Eq. 4)

which needs a single diameter d_neck for a structure that, in H01, is a chain of
frustums with varying radii. Rather than picking a summary diameter, this module
integrates along the neck. For ONE frustum of length L with end radii r1, r2 and
linearly interpolated radius r(x) = r1 + (r2 - r1) x / L, the axial resistance is

    R = rho_a * integral_0^L dx / (pi r(x)^2)
      = rho_a * L / (pi r1 r2)                                         (Eq. G1)

which is EXACT, not an approximation: the integral of 1/(a + bx)^2 evaluates in
closed form and the (r2 - r1) factors cancel. Summing the frustums in series,

    G(sigma) = sum_i L_i / (pi r1_i r2_i)          [1/cm]              (Eq. G2)
    R_neck(sigma) = rho_a * G(sigma)               [Ohm]               (Eq. G3)

Eq. (4) is the special case r1 = r2 = d/2. This module also reports

    d_neck_equiv(sigma) = sqrt( 4 L_neck(sigma) / (pi G_um(sigma)) )   (Eq. G4)

the diameter of the uniform cylinder of the same length and the same axial
resistance. That is the number to compare against published neck diameters:
it is what Eq. (4) would have to be fed to reproduce this neck's resistance.

WHY G IS STORED, NOT R
----------------------
G is geometry alone. rho_a is a contested parameter (published spine-neck
resistances span roughly 20 MOhm to 3 GOhm across preparations and methods, and
part of that spread is rho_a assumption rather than measurement). Storing G and
multiplying at analysis time means the entire rho_a sweep is free and no stored
number silently embeds a resistivity choice. Under a UNIFORM scaling of all
radii by s, G -> G / s^2 exactly, so radius-scale sensitivity is also free.
An ADDITIVE radius offset does not reduce that way, so offsets must be applied
at build time: see the radius_offsets_um argument.

SCOPE AND LIMITS -- read before using the numbers
-------------------------------------------------
1. Neck radii in H01 sit near the imaging resolution (about 4 nm in plane, about
   33 nm in z). Since R_neck goes as r^-2, a fractional radius error doubles
   into the resistance. This module does not correct for that; it reports
   G at the measured radii AND at the requested offsets so the sensitivity is
   visible, and it flags spines whose radii sit at the fallback value.
2. If the input frame carries no 'r' column, spine_density's fallback radius
   (50 nm) is used for EVERY node, and every reported resistance is then a
   function of length alone. The `radius_default_frac` column and the
   `n_nodes_at_default_radius` summary field exist to make that impossible to
   miss. Check them before quoting any resistance.

   THE AUTHORITATIVE CHECK IS ELSEWHERE. phi_pipeline_colab.radius_report()
   is the cell-level radius gate for this project, and it is stricter than
   anything here: it assesses flatness on NON-SOMA nodes only (a whole-cell
   uniqueness test sees 2 distinct radii and silently passes a cell whose
   entire dendrite is at the fallback), and it emits `radius_suspect` as a
   single boolean combining the flat-radius and default-dominated cases.
   Pass its output to cell_spine_summary(radius_report=...) so the two
   travel together. The per-spine `radius_default_frac` column here is
   COMPLEMENTARY, not a replacement: it localises the problem to individual
   spines, which a cell-level gate cannot do. If `radius_suspect` is True,
   no resistance in this frame means anything, regardless of what the
   per-spine fractions say.
3. Head area is the summed membrane area of the head-labelled part of the
   spine, i.e. the same frustum-lateral-area convention spine_density uses. It
   is NOT a fitted sphere. `head_equiv_sphere_diam_um` is provided for
   comparison with head diameters reported in the literature, but the area, not
   the diameter, is the quantity that is actually measured here.
4. The head/neck split is whatever label_dendritic_spines_robust produced. This
   module does not re-derive it. Spines whose label sequence has no neck (the
   labeller assigns everything to 'head' when a path has fewer than 3 distinct
   nodes) get L_neck = 0, G = 0, has_neck = False, and MUST be excluded from
   any resistance statistic rather than counted as zero-resistance spines.

DEPENDENCIES: numpy, pandas, and spine_density (for the shared geometry
primitives and label conventions). No scipy, no networkx.

Pure ASCII source (HPC-safe).

NOTE ON REUSED PRIVATES: this module calls spine_density._prepare_nodes,
._frustum_lateral_area, ._segment_length_um and ._path_distances_um. They are
underscore-private but re-typing them here would recreate exactly the duplicated
-definition problem the project already has one instance of. The correct fix is
to promote them to public names in a later spine_density revision and update the
imports here; until then this file depends on them deliberately and says so.
"""

import hashlib
import math
from collections import defaultdict

import numpy as np
import pandas as pd

import spine_density as sd

MODULE_VERSION = "spine_geometry-1.0.0"

# Label conventions. SPINE_LABELS is the full set (spine / head / neck); the
# head and neck subsets are what label_dendritic_spines_robust actually writes.
HEAD_LABELS = ("head",)
NECK_LABELS = ("neck",)

# A node label that is in SPINE_LABELS but in neither subset (i.e. the bare
# 'spine' label, produced by the non-head/neck labeller variant) is counted
# here. Such nodes have no head/neck assignment and are reported separately.
UM_PER_CM = 1.0e4

_REQUIRED_COLUMNS = ("id", "p", "x", "y", "z", "annotated_type")


# --------------------------------------------------------------------------- #
# Geometry primitives                                                          #
# --------------------------------------------------------------------------- #
def frustum_axial_factor_um(r1_um, r2_um, length_um):
    """Axial resistance factor of one frustum, in 1/um.

    Returns L / (pi r1 r2), so that R [Ohm] = rho_a [Ohm um] * factor. See
    Eq. (G1) in the module docstring: this is exact for a linearly tapering
    radius, not a mean-radius approximation.

    Returns 0.0 for a degenerate segment (zero length, or a non-positive
    radius at either end); the caller is responsible for counting those.
    """
    if length_um <= 0.0 or r1_um <= 0.0 or r2_um <= 0.0:
        return 0.0
    return length_um / (math.pi * r1_um * r2_um)


def axial_factor_to_per_cm(factor_um):
    """Convert an axial factor from 1/um to 1/cm.

    G [1/cm] = 1e4 * G [1/um], so that R [Ohm] = rho_a [Ohm cm] * G [1/cm].
    """
    return UM_PER_CM * factor_um


def neck_resistance_ohm(g_per_cm, rho_a_ohm_cm):
    """R_neck [Ohm] = rho_a [Ohm cm] * G [1/cm].   Eq. (G3)."""
    return np.asarray(g_per_cm, dtype=float) * float(rho_a_ohm_cm)


def neck_resistance_mohm(g_per_cm, rho_a_ohm_cm):
    """R_neck in MOhm, the unit the literature reports."""
    return neck_resistance_ohm(g_per_cm, rho_a_ohm_cm) / 1.0e6


def equivalent_uniform_diameter_um(length_um, g_per_cm):
    """Diameter of the uniform cylinder with the same length and R.  Eq. (G4).

    d_equiv = sqrt(4 L / (pi G_um)), with G_um = G_per_cm / 1e4.
    Returns nan when either argument is non-positive.
    """
    length_um = np.asarray(length_um, dtype=float)
    g_um = np.asarray(g_per_cm, dtype=float) / UM_PER_CM
    out = np.full(np.shape(length_um), np.nan, dtype=float)
    ok = (length_um > 0) & (g_um > 0)
    out[ok] = np.sqrt(4.0 * length_um[ok] / (math.pi * g_um[ok]))
    return out


def equivalent_sphere_diameter_um(area_um2):
    """Diameter of the sphere with the given surface area: d = sqrt(A / pi)."""
    area = np.asarray(area_um2, dtype=float)
    out = np.full(np.shape(area), np.nan, dtype=float)
    ok = area > 0
    out[ok] = np.sqrt(area[ok] / math.pi)
    return out


# --------------------------------------------------------------------------- #
# Spine decomposition                                                          #
# --------------------------------------------------------------------------- #
def _label_map(df, input_column="annotated_type"):
    """id -> lowercased annotated_type string."""
    return {i: str(t).lower()
            for i, t in zip(df["id"].tolist(), df[input_column].tolist())}


def _spine_roots(node):
    """Spine-labelled nodes whose parent is not spine-labelled."""
    out = []
    for i, a in node.items():
        if not a["is_spine"]:
            continue
        p = a["p"]
        if p not in node or not node[p]["is_spine"]:
            out.append(i)
    return sorted(out)


def _spine_subtree(node, children, root_id):
    """All spine-labelled nodes reachable from root_id through spine children."""
    out = [root_id]
    stack = [root_id]
    while stack:
        cur = stack.pop()
        for c in children.get(cur, ()):
            if c in node and node[c]["is_spine"]:
                out.append(c)
                stack.append(c)
    return out


def _spine_tips(node, children, members):
    """Members of the spine with no spine-labelled child."""
    member_set = set(members)
    tips = []
    for m in members:
        if not any(c in member_set for c in children.get(m, ())):
            tips.append(m)
    return sorted(tips)


def _path_from_base(node, base_id, tip_id, members):
    """Node path [base, root, ..., tip], walking parents up from the tip.

    Returns None if the walk leaves the spine before reaching base (which would
    mean the frame is not a tree, and is treated as a hard error upstream).
    """
    member_set = set(members)
    path = [tip_id]
    cur = tip_id
    while cur != base_id:
        p = node[cur]["p"]
        if p not in node:
            return None
        path.append(p)
        if p == base_id:
            break
        if p not in member_set:
            return None
        cur = p
    path.reverse()
    return path


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def build_spine_geometry(df,
                         nid=None,
                         shaft_regex=None,
                         spine_labels=None,
                         head_labels=HEAD_LABELS,
                         neck_labels=NECK_LABELS,
                         default_radius_nm=None,
                         input_units="nm",
                         radius_offsets_um=(-0.025, 0.025),
                         annotation_column="annotated_type",
                         self_check=True):
    """One row per spine, with geometry and neck axial conductance factor.

    Parameters
    ----------
    df : pandas.DataFrame
        The SPINE-LABELLED frame, BEFORE pruning: columns id, p, x, y, z,
        annotated_type, and (strongly preferred) r. Same frame
        spine_density.build_phi consumes.
    nid : hashable or None
        Neuron id, copied into every row. Not used in any computation.
    shaft_regex, spine_labels, default_radius_nm : optional
        Defaults are taken from spine_density so the two modules cannot drift
        apart in what counts as shaft or spine. Pass explicitly only to test.
    head_labels, neck_labels : tuple of str
        Which annotated_type values mean head and neck. Matched
        case-insensitively against the raw annotation, NOT against the
        compartment class.
    input_units : {'nm', 'um'}
        Units of x, y, z and r in `df`. H01 is 'nm'.
    radius_offsets_um : tuple of float
        Additive offsets applied to EVERY radius, producing extra columns
        g_per_cm_off{k}. Intended for the H01 z-resolution sensitivity: an
        offset of +/-0.025 um is +/-50 nm on diameter. Pass () to skip.
        A uniform multiplicative scale needs no offset column: G scales as
        1/s^2 exactly.
    self_check : bool
        Run internal consistency assertions (area conservation against
        spine_density's own attribution, path/parent agreement).

    Returns
    -------
    pandas.DataFrame, one row per spine, columns:

      identity      nid, spine_root_id, base_node_id, base_is_shaft
      counts        n_nodes, n_head_nodes, n_neck_nodes, n_other_spine_nodes,
                    n_tips, n_segments, n_degenerate_segments
      lengths       L_total_um, L_neck_um, L_head_um
      areas         A_spine_um2, A_neck_um2, A_head_um2
      resistance    g_per_cm, g_per_cm_min, g_per_cm_max, g_per_cm_off{k}
      derived       d_neck_equiv_um, head_equiv_sphere_diam_um,
                    neck_r_min_um, neck_r_mean_um
      position      d_base_um, base_x_um, base_y_um, base_z_um
      flags         has_neck, has_head, radius_default_frac, is_single_node

    The primary g_per_cm is measured along the path from the base to the
    PRIMARY tip, defined as the tip whose head-labelled portion carries the
    largest membrane area (ties broken by smallest node id, so the choice is
    deterministic). g_per_cm_min and g_per_cm_max span all tips; for the
    single-tip spines that dominate any real reconstruction all three are equal.
    """
    shaft_regex = sd.SHAFT_REGEX if shaft_regex is None else shaft_regex
    spine_labels = sd.SPINE_LABELS if spine_labels is None else spine_labels
    default_radius_nm = (sd.DEFAULT_RADIUS_NM if default_radius_nm is None
                         else default_radius_nm)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("input DataFrame is missing required columns: %r"
                         % missing)
    if input_units not in ("nm", "um"):
        raise ValueError("input_units must be 'nm' or 'um', got %r"
                         % (input_units,))

    node, children, root = sd._prepare_nodes(
        df, shaft_regex, spine_labels, default_radius_nm, input_units)
    labels = _label_map(df, annotation_column)
    path_dist = sd._path_distances_um(node, children, root)

    head_set = {s.lower() for s in head_labels}
    neck_set = {s.lower() for s in neck_labels}
    default_radius_um = default_radius_nm / (
        sd.NM_PER_UM if input_units == "nm" else 1.0)

    offsets = tuple(float(o) for o in (radius_offsets_um or ()))

    rows = []
    for root_id in _spine_roots(node):
        base_id = node[root_id]["p"]
        members = _spine_subtree(node, children, root_id)
        member_set = set(members)
        tips = _spine_tips(node, children, members)

        # -- per-segment pass over the whole spine (segments = parent -> member)
        a_head = a_neck = a_other = 0.0
        l_head = l_neck_all = l_other = 0.0
        n_head = n_neck = n_other = 0
        n_degen = 0
        n_default_radius = 0
        neck_radii = []
        head_radii = []
        for m in members:
            lab = labels.get(m, "")
            if lab in head_set:
                n_head += 1
                head_radii.append(node[m]["r"])
            elif lab in neck_set:
                n_neck += 1
            else:
                n_other += 1
            if abs(node[m]["r"] - default_radius_um) < 1e-12:
                n_default_radius += 1

            p = node[m]["p"]
            if p not in node:
                continue
            seg_len = sd._segment_length_um(node, p, m)
            area = sd._frustum_lateral_area(node[p]["r"], node[m]["r"], seg_len)
            if seg_len <= 0.0 or node[p]["r"] <= 0.0 or node[m]["r"] <= 0.0:
                n_degen += 1
            # a segment belongs to the compartment its DISTAL node is labelled
            if lab in head_set:
                a_head += area
                l_head += seg_len
            elif lab in neck_set:
                a_neck += area
                l_neck_all += seg_len
                neck_radii.append(node[m]["r"])
            else:
                a_other += area
                l_other += seg_len

        # -- per-tip pass: the neck is the base-side prefix of the base->tip path
        per_tip = []
        for tip in tips:
            path = _path_from_base(node, base_id, tip, members)
            if path is None:
                raise ValueError(
                    "spine %r: parent walk from tip %r did not reach base %r; "
                    "the frame is not a tree" % (root_id, tip, base_id))
            g_um = 0.0
            g_um_off = {o: 0.0 for o in offsets}
            l_neck = 0.0
            head_area_on_path = 0.0
            in_neck = True
            for a_id, b_id in zip(path[:-1], path[1:]):
                lab = labels.get(b_id, "")
                seg_len = sd._segment_length_um(node, a_id, b_id)
                r1, r2 = node[a_id]["r"], node[b_id]["r"]
                if lab in head_set:
                    in_neck = False
                if in_neck and lab in neck_set:
                    g_um += frustum_axial_factor_um(r1, r2, seg_len)
                    for o in offsets:
                        g_um_off[o] += frustum_axial_factor_um(
                            r1 + o, r2 + o, seg_len)
                    l_neck += seg_len
                elif lab in head_set:
                    head_area_on_path += sd._frustum_lateral_area(r1, r2,
                                                                  seg_len)
            per_tip.append({
                "tip": tip,
                "g_per_cm": axial_factor_to_per_cm(g_um),
                "g_off": {o: axial_factor_to_per_cm(v)
                          for o, v in g_um_off.items()},
                "L_neck_um": l_neck,
                "head_area": head_area_on_path,
            })

        # primary tip: largest head area, ties by smallest node id
        primary = sorted(per_tip, key=lambda t: (-t["head_area"], t["tip"]))[0]
        g_values = [t["g_per_cm"] for t in per_tip]

        row = {
            "nid": nid,
            "spine_root_id": root_id,
            "base_node_id": base_id if base_id in node else -1,
            "base_is_shaft": bool(node[base_id]["is_shaft"])
            if base_id in node else False,
            "primary_tip_id": primary["tip"],
            "n_nodes": len(members),
            "n_head_nodes": n_head,
            "n_neck_nodes": n_neck,
            "n_other_spine_nodes": n_other,
            "n_tips": len(tips),
            "n_segments": sum(1 for m in members if node[m]["p"] in node),
            "n_degenerate_segments": n_degen,
            "L_total_um": l_head + l_neck_all + l_other,
            "L_neck_um": primary["L_neck_um"],
            "L_neck_all_um": l_neck_all,
            "L_head_um": l_head,
            "A_spine_um2": a_head + a_neck + a_other,
            "A_neck_um2": a_neck,
            "A_head_um2": a_head,
            "A_other_um2": a_other,
            # -- head SHAPE diagnostics -------------------------------------
            # A_head_um2 above is a sum of frustum LATERAL areas along the
            # skeleton path. For a tube that is correct. For a blob it is not:
            # lateral area is 2 pi r L against a sphere's 4 pi r^2, so the two
            # agree only when the path length L through the head equals the
            # head diameter 2r. A head represented by one or two skeleton
            # nodes has L << 2r and the lateral sum under-counts in direct
            # proportion. These columns make that visible instead of leaving
            # it as an unstated modelling choice:
            #
            #   A_head_sphere_um2      4 pi r_max^2, the head treated as a
            #                          sphere of its own largest radius
            #   head_lateral_over_sphere   ratio of the two. Near 1.0 means
            #                          the two models agree and A_head_um2 is
            #                          safe. Well below 1.0 means the head is
            #                          blob-like and A_head_um2 is a lower
            #                          bound, NOT a measurement of head area.
            #   head_path_over_radius  L_head / r_max. Near 2.0 is a head the
            #                          skeleton traverses fully; near 0 is a
            #                          head collapsed to a point.
            #
            # head_lateral_over_sphere carries a constant positive offset from
            # the neck->head transition segment: the radius jumps from neck to
            # head over almost no length, so that frustum's slant is large and
            # it contributes a "shoulder" annulus. That is real membrane where
            # the neck meets the head, not an artefact, so the ratio can
            # slightly EXCEED 1 for a fully traversed head. Read changes in
            # the ratio across a population, not its absolute value against 1.
            #
            # Neither is "the" answer: the truth for a real spine head lies
            # between a lateral sum and a circumscribed sphere. They bracket
            # it, and the width of the bracket is the honest uncertainty.
            "head_r_max_um": (max(head_radii) if head_radii else float("nan")),
            "head_r_mean_um": (float(np.mean(head_radii)) if head_radii
                               else float("nan")),
            "A_head_sphere_um2": (4.0 * math.pi * max(head_radii) ** 2
                                  if head_radii else float("nan")),
            "head_lateral_over_sphere": (
                a_head / (4.0 * math.pi * max(head_radii) ** 2)
                if head_radii and max(head_radii) > 0 else float("nan")),
            "head_path_over_radius": (
                l_head / max(head_radii)
                if head_radii and max(head_radii) > 0 else float("nan")),
            "g_per_cm": primary["g_per_cm"],
            "g_per_cm_min": min(g_values),
            "g_per_cm_max": max(g_values),
            "neck_r_min_um": min(neck_radii) if neck_radii else float("nan"),
            "neck_r_mean_um": (float(np.mean(neck_radii)) if neck_radii
                               else float("nan")),
            "d_base_um": path_dist.get(base_id, float("nan")),
            "base_x_um": node[base_id]["x"] if base_id in node else float("nan"),
            "base_y_um": node[base_id]["y"] if base_id in node else float("nan"),
            "base_z_um": node[base_id]["z"] if base_id in node else float("nan"),
            "has_neck": n_neck > 0 and primary["L_neck_um"] > 0.0,
            "has_head": n_head > 0,
            "is_single_node": len(members) == 1,
            "radius_default_frac": (n_default_radius / float(len(members))
                                    if members else float("nan")),
        }
        for k, o in enumerate(offsets):
            row["g_per_cm_off%d" % k] = primary["g_off"][o]
            row["radius_offset_um_%d" % k] = o
        rows.append(row)

    columns_order = [
        "nid", "spine_root_id", "base_node_id", "base_is_shaft",
        "primary_tip_id", "n_nodes", "n_head_nodes", "n_neck_nodes",
        "n_other_spine_nodes", "n_tips", "n_segments",
        "n_degenerate_segments", "L_total_um", "L_neck_um", "L_neck_all_um",
        "L_head_um", "A_spine_um2", "A_neck_um2", "A_head_um2", "A_other_um2",
        "head_r_max_um", "head_r_mean_um", "A_head_sphere_um2",
        "head_lateral_over_sphere", "head_path_over_radius",
        "g_per_cm", "g_per_cm_min", "g_per_cm_max", "neck_r_min_um",
        "neck_r_mean_um", "d_base_um", "base_x_um", "base_y_um", "base_z_um",
        "has_neck", "has_head", "is_single_node", "radius_default_frac",
    ]
    for k in range(len(offsets)):
        columns_order += ["g_per_cm_off%d" % k, "radius_offset_um_%d" % k]

    if not rows:
        return pd.DataFrame(columns=columns_order)

    out = pd.DataFrame(rows)[columns_order]
    out = out.sort_values("spine_root_id",
                          kind="mergesort").reset_index(drop=True)

    # derived columns
    out["d_neck_equiv_um"] = equivalent_uniform_diameter_um(
        out["L_neck_um"].values, out["g_per_cm"].values)
    out["head_equiv_sphere_diam_um"] = equivalent_sphere_diameter_um(
        out["A_head_um2"].values)

    if self_check:
        _self_check(out, df, node, children, shaft_regex, spine_labels,
                    default_radius_nm, input_units)
    return out


def _self_check(out, df, node, children, shaft_regex, spine_labels,
                default_radius_nm, input_units):
    """Cross-module consistency assertions. Cheap; run by default."""
    # 1. total spine area must equal what spine_density attributes to the shaft
    seg_spine_area, dropped = sd._attribute_spine_area(node, children)
    total_sd = float(sum(seg_spine_area.values())) + float(dropped)
    total_here = float(out["A_spine_um2"].sum())
    if total_sd > 0:
        rel = abs(total_here - total_sd) / total_sd
        assert rel < 1e-9, (
            "spine area disagrees with spine_density._attribute_spine_area: "
            "%.12g here vs %.12g there (rel %.3g)" % (total_here, total_sd, rel))
    # 2. area decomposition must be exhaustive
    parts = (out["A_neck_um2"] + out["A_head_um2"] + out["A_other_um2"]).values
    assert np.allclose(parts, out["A_spine_um2"].values, rtol=0, atol=1e-12), (
        "A_neck + A_head + A_other != A_spine")
    # 3. min <= primary <= max over tips
    assert (out["g_per_cm_min"] <= out["g_per_cm"] + 1e-12).all(), "g < g_min"
    assert (out["g_per_cm"] <= out["g_per_cm_max"] + 1e-12).all(), "g > g_max"
    # 4. spine ids unique
    assert out["spine_root_id"].is_unique, "duplicate spine_root_id"


# --------------------------------------------------------------------------- #
# Aggregation and joins                                                        #
# --------------------------------------------------------------------------- #
def cell_spine_summary(spine_df, rho_a_ohm_cm=(100.0, 200.0, 300.0, 400.0),
                       radius_report=None):
    """Cell-level summary of a spine geometry frame.

    Resistance statistics are computed over spines with has_neck == True only.
    Spines without a labelled neck have G = 0 by construction and including
    them would drag every quantile toward zero.

    Parameters
    ----------
    radius_report : dict or None
        Output of phi_pipeline_colab.radius_report() for the same cell. When
        supplied, its fields are copied in under a 'radius_' prefix and
        `resistance_trustworthy` is set to (not radius_suspect). Supply it:
        a resistance computed on fallback radii is a function of length alone,
        and nothing else in this summary will tell you that.
    """
    n = len(spine_df)
    if n == 0:
        out = {"n_spines": 0}
        if radius_report is not None:
            out.update({"radius_" + k: v for k, v in radius_report.items()})
            out["resistance_trustworthy"] = not bool(
                radius_report.get("radius_suspect", True))
        return out
    withneck = spine_df[spine_df["has_neck"]]
    out = {
        "n_spines": int(n),
        "n_with_neck": int(len(withneck)),
        "frac_with_neck": float(len(withneck)) / n,
        "n_with_head": int(spine_df["has_head"].sum()),
        "n_single_node": int(spine_df["is_single_node"].sum()),
        "n_multi_tip": int((spine_df["n_tips"] > 1).sum()),
        "n_degenerate_segments": int(spine_df["n_degenerate_segments"].sum()),
        "n_nodes_at_default_radius": float(
            (spine_df["radius_default_frac"] * spine_df["n_nodes"]).sum()),
        "frac_nodes_at_default_radius": float(
            (spine_df["radius_default_frac"] * spine_df["n_nodes"]).sum()
            / spine_df["n_nodes"].sum()),
        "A_spine_total_um2": float(spine_df["A_spine_um2"].sum()),
        "A_head_total_um2": float(spine_df["A_head_um2"].sum()),
        "A_neck_total_um2": float(spine_df["A_neck_um2"].sum()),
        "L_neck_median_um": float(withneck["L_neck_um"].median())
        if len(withneck) else float("nan"),
        "d_neck_equiv_median_um": float(withneck["d_neck_equiv_um"].median())
        if len(withneck) else float("nan"),
        "A_head_median_um2": float(spine_df["A_head_um2"].median()),
        # Head-shape bracket. If lateral_over_sphere is well below 1, the
        # lateral sum is a LOWER BOUND on head area, not a measurement, and
        # any F or spine-area total built from it is biased low.
        "A_head_sphere_median_um2": float(
            spine_df["A_head_sphere_um2"].median(skipna=True)),
        "head_lateral_over_sphere_median": float(
            spine_df["head_lateral_over_sphere"].median(skipna=True)),
        "head_path_over_radius_median": float(
            spine_df["head_path_over_radius"].median(skipna=True)),
        "head_r_max_median_um": float(
            spine_df["head_r_max_um"].median(skipna=True)),
        "d_base_median_um": float(spine_df["d_base_um"].median()),
    }
    for rho in rho_a_ohm_cm:
        if len(withneck) == 0:
            continue
        r = neck_resistance_mohm(withneck["g_per_cm"].values, rho)
        key = "R_neck_MOhm_rho%d" % int(round(rho))
        out[key + "_p05"] = float(np.percentile(r, 5))
        out[key + "_median"] = float(np.median(r))
        out[key + "_p95"] = float(np.percentile(r, 95))

    if radius_report is not None:
        out.update({"radius_" + k: v for k, v in radius_report.items()})
        out["resistance_trustworthy"] = not bool(
            radius_report.get("radius_suspect", True))
    return out


def attach_to_phi(spine_df, phi_df):
    """Join per-spine rows to the phi segment they were attributed to.

    spine_density attributes a spine's area to the shaft segment whose DISTAL
    node is the spine's base node. phi_df stores that segment as
    (node_from, node_to), so the join key is base_node_id == node_to. Returns a
    copy of spine_df with branch_id, seg_index, d_from_um, d_to_um added; rows
    that do not match (spines on the root, whose area spine_density re-attributes
    to the root's first shaft child) get NaN and are counted in the report.
    """
    keep = ["node_to", "branch_id", "seg_index", "d_from_um", "d_to_um",
            "seg_len_um", "shaft_diam_um"]
    right = phi_df[keep].drop_duplicates(subset=["node_to"])
    merged = spine_df.merge(right, how="left",
                            left_on="base_node_id", right_on="node_to")
    merged = merged.drop(columns=["node_to"])
    n_unmatched = int(merged["branch_id"].isna().sum())
    return merged, {"n_rows": len(merged), "n_unmatched": n_unmatched}


def save_spine_geometry(spine_df, path):
    """Write the frame to CSV and return its sha256 (hex)."""
    spine_df.to_csv(path, index=False)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_and_save_spine_geometry(df, nid, output_dir, **kwargs):
    """build_spine_geometry + write neuron_{nid}_spine_geometry.csv.

    Returns (spine_df, provenance_dict).
    """
    import os
    spine_df = build_spine_geometry(df, nid=nid, **kwargs)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "neuron_%s_spine_geometry.csv" % nid)
    sha = save_spine_geometry(spine_df, path)
    prov = {
        "module_version": MODULE_VERSION,
        "spine_density_version": sd.MODULE_VERSION,
        "nid": nid,
        "path": path,
        "sha256": sha,
        "n_spines": int(len(spine_df)),
        "kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in kwargs.items()},
    }
    prov.update(cell_spine_summary(spine_df))
    return spine_df, prov
