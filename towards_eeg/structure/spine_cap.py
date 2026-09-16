"""spine_cap -- close the open distal tip of every skeleton leaf with a
spherical cap, to recover the membrane lost to the H01 endpoint erosion.

Towards-EEG stage S1.3 (data-generation side; runs in Colab, not on the HPC).

WHY THIS MODULE EXISTS
----------------------
The H01 skeletons were produced by block-wise TEASAR (Kimimaro) and then, per
Shapson-Coe et al. 2024 supplementary methods, "eroded back from endpoints by
100 nm, and sparsified to approximately 300 nm node spacing while retaining all
branch points and endpoints".

Consequences for membrane area:

  (a) Every skeleton LEAF sits about 100 nm proximal to the true distal extent
      of the membrane. The frustum chain of spine_density therefore represents
      each spine head (and each dendritic/axonal tip) as an OPEN TUBE: the
      lateral area of the last frustum is counted, but nothing closes it.
  (b) Because erosion applied to every endpoint, the same deficit exists on the
      shaft. Capping spine tips alone would inflate F by construction, so this
      module caps BOTH and the caller decides.
  (c) Because sparsification retained endpoints, the surviving last node IS the
      eroded endpoint. The 100 nm is therefore a documented pipeline constant,
      not a free parameter to be fitted.

GEOMETRY (the "A" construction; chosen by the user over the tangent variant)
---------------------------------------------------------------------------
At a leaf with terminal radius r_t, place a sphere that passes THROUGH the rim
circle of radius r_t and whose pole sits at axial height h beyond the plane of
that rim. Nothing of the frustum chain is removed; the cap is added on top.

    R_cap  = (r_t^2 + h^2) / (2 h)                                       (1)
    A_cap  = 2 pi R_cap h = pi (r_t^2 + h^2)                             (2)

Equation (2) is Archimedes' hat-box theorem: R_cap cancels, so the cap area
depends only on (r_t, h). Properties that make this construction safe:

  * It never fails. R_cap >= r_t reduces to (r_t - h)^2 >= 0, true for all
    r_t, h >= 0. There is no discriminant, no runaway root, no dependence on
    the local taper. Contrast the tangent-sphere variant, which has no valid
    solution whenever the tip is flaring.
  * It is a hemisphere exactly when r_t == h; shallower for r_t > h; deeper
    for r_t < h. All are valid spherical caps.
  * h -> 0 gives A_cap -> pi r_t^2, the flat disc. The disc is therefore the
    h = 0 special case of the same formula, not a separate code path, and it
    serves as a strict lower bound for bracketing.
  * A_cap is exactly the area of the spherical cap of a sphere of radius R
    truncated at distance h from its pole: substituting r_t^2 = R^2 - (R-h)^2
    into (2) returns 2 pi R h. So a sphere sampled by axial nodes and closed
    with (2) at both poles recovers 4 pi R^2 exactly (verified in the smoke
    test).

WHICH NODES GET CAPPED (the gate)
---------------------------------
ONLY nodes that are true leaves of the FULL skeleton, i.e. nodes with zero
children in the unfiltered parent map. This is deliberately NOT the same test
as spine_geometry._spine_tips, which asks whether a node has any SPINE-LABELLED
child. A spine-labelled node whose only children are non-spine is a LABEL
BOUNDARY, not a skeleton endpoint: no erosion happened there, and capping it
would fabricate membrane. audit_tips() reports both counts so the discrepancy
is visible per cell.

Nodes with a non-positive radius are never capped (counted separately), because
pi (0 + h^2) is a spurious non-zero area at a degenerate node.

UNITS: everything in this module is in um. Callers convert at their boundary.

DEPENDENCIES: standard library only (math). No numpy, no pandas, no scipy.
Pure ASCII source (HPC-safe).
"""

import math

MODULE_VERSION = "spine_cap-1.0.0"

# H01 endpoint erosion, Shapson-Coe et al. 2024 supplementary methods.
CAP_H_UM_DEFAULT = 0.100


# --------------------------------------------------------------------------- #
# Geometry primitives                                                          #
# --------------------------------------------------------------------------- #
def cap_area_um2(r_tip_um, h_um):
    """Lateral area of the spherical cap closing a tube of rim radius r_tip_um
    with its pole h_um beyond the rim plane. Equation (2) above.

    Returns 0.0 for a non-positive rim radius (degenerate node); returns the
    flat disc pi r^2 for h_um == 0.
    """
    if r_tip_um is None or r_tip_um <= 0.0:
        return 0.0
    if h_um < 0.0:
        raise ValueError("cap height h_um must be >= 0, got %r" % (h_um,))
    return math.pi * (r_tip_um * r_tip_um + h_um * h_um)


def cap_sphere_radius_um(r_tip_um, h_um):
    """Radius of the sphere the cap is a section of. Equation (1) above.

    Diagnostic only: the area does not depend on it. Returns inf for h == 0
    (the flat disc is the limit of an infinitely large sphere).
    """
    if h_um == 0.0:
        return float("inf")
    if h_um < 0.0:
        raise ValueError("cap height h_um must be >= 0, got %r" % (h_um,))
    return (r_tip_um * r_tip_um + h_um * h_um) / (2.0 * h_um)


def cap_polar_angle_rad(r_tip_um, h_um):
    """Half-angle subtended by the cap at the sphere centre. Diagnostic only.

    pi/2 exactly when r_tip_um == h_um (hemisphere), < pi/2 when the cap is
    shallow (r_tip_um > h_um), > pi/2 when it is deep.
    """
    R = cap_sphere_radius_um(r_tip_um, h_um)
    if math.isinf(R):
        return 0.0
    return math.acos(max(-1.0, min(1.0, (R - h_um) / R)))


# --------------------------------------------------------------------------- #
# Leaf identification on the FULL skeleton                                     #
# --------------------------------------------------------------------------- #
def is_true_leaf(children, node_id):
    """True iff node_id has zero children in the full (unfiltered) parent map.

    `children` is the map built by spine_density._prepare_nodes, which is built
    over every node in the frame regardless of label. Passing a label-filtered
    map here would silently reintroduce the label-boundary bug this gate exists
    to prevent.
    """
    return len(children.get(node_id, ())) == 0


def true_leaves(node, children, root=None):
    """Sorted ids of every true leaf, excluding the root of a degenerate
    single-node tree (a root with no children is not an eroded endpoint).
    """
    out = [i for i in node if is_true_leaf(children, i)]
    if root is not None and root in node and is_true_leaf(children, root):
        out = [i for i in out if i != root]
    return sorted(out)


def cap_area_for_node(node, children, node_id, h_um):
    """Cap area for one node: 0.0 unless it is a true leaf with r > 0."""
    if not is_true_leaf(children, node_id):
        return 0.0
    return cap_area_um2(node[node_id]["r"], h_um)


def subtree_cap_area(node, children, member_ids, h_um):
    """Total cap area over the true leaves among `member_ids`.

    Returns (total_area_um2, n_capped, n_skipped_bad_radius).
    """
    total = 0.0
    n_capped = 0
    n_bad_r = 0
    for m in member_ids:
        if not is_true_leaf(children, m):
            continue
        r = node[m]["r"]
        if r is None or r <= 0.0:
            n_bad_r += 1
            continue
        total += cap_area_um2(r, h_um)
        n_capped += 1
    return total, n_capped, n_bad_r


# --------------------------------------------------------------------------- #
# Gate 1 audit: true leaves vs label-boundary ends                             #
# --------------------------------------------------------------------------- #
def audit_tips(node, children, root=None):
    """Per-cell audit of what would and would not be capped.

    Returns a dict with, separately for spine-labelled and non-spine nodes:
      n_true_leaf                 -- zero children in the full skeleton -> CAPPED
      n_label_boundary_end        -- spine node with only non-spine children,
                                     i.e. what spine_geometry._spine_tips would
                                     call a tip but which is NOT an endpoint
                                     -> NOT CAPPED
      n_true_leaf_bad_radius      -- true leaf with r <= 0 -> NOT CAPPED
      r_tip_um                    -- list of terminal radii of the capped nodes,
                                     the input needed to turn any projected
                                     change in F into a measured one.

    The `is_spine` flag comes from spine_density._prepare_nodes, so this
    function makes no independent assumption about the label vocabulary.
    """
    leaves = set(true_leaves(node, children, root=root))

    spine_leaf_r = []
    other_leaf_r = []
    n_spine_leaf_bad = 0
    n_other_leaf_bad = 0
    n_boundary = 0
    boundary_ids = []

    for i, a in node.items():
        if i in leaves:
            r = a["r"]
            if r is None or r <= 0.0:
                if a["is_spine"]:
                    n_spine_leaf_bad += 1
                else:
                    n_other_leaf_bad += 1
            elif a["is_spine"]:
                spine_leaf_r.append(float(r))
            else:
                other_leaf_r.append(float(r))
        elif a["is_spine"]:
            kids = children.get(i, ())
            if kids and not any(node[c]["is_spine"] for c in kids if c in node):
                n_boundary += 1
                boundary_ids.append(i)

    return {
        "module_version": MODULE_VERSION,
        "n_nodes": len(node),
        "n_true_leaf_total": len(leaves),
        "spine_n_true_leaf": len(spine_leaf_r),
        "spine_n_true_leaf_bad_radius": n_spine_leaf_bad,
        "spine_n_label_boundary_end": n_boundary,
        "spine_label_boundary_ids": sorted(boundary_ids),
        "other_n_true_leaf": len(other_leaf_r),
        "other_n_true_leaf_bad_radius": n_other_leaf_bad,
        "spine_r_tip_um": spine_leaf_r,
        "other_r_tip_um": other_leaf_r,
    }


def summarise_audit(audit, h_um=CAP_H_UM_DEFAULT):
    """One-line-per-field summary of audit_tips(), with the implied cap totals.

    Reports the total cap area that WOULD be added on the spine side and on the
    shaft side, so the effect on F can be read off before committing to it.
    """
    sp = audit["spine_r_tip_um"]
    ot = audit["other_r_tip_um"]
    a_sp = sum(cap_area_um2(r, h_um) for r in sp)
    a_ot = sum(cap_area_um2(r, h_um) for r in ot)
    mean_sp = (sum(sp) / len(sp)) if sp else float("nan")
    mean_ot = (sum(ot) / len(ot)) if ot else float("nan")
    return {
        "h_um": h_um,
        "spine_n_capped": len(sp),
        "spine_mean_r_tip_um": mean_sp,
        "spine_total_cap_um2": a_sp,
        "spine_mean_cap_um2": (a_sp / len(sp)) if sp else float("nan"),
        "other_n_capped": len(ot),
        "other_mean_r_tip_um": mean_ot,
        "other_total_cap_um2": a_ot,
        "spine_n_label_boundary_end": audit["spine_n_label_boundary_end"],
    }


# --------------------------------------------------------------------------- #
# Meridian profiles, for plotting                                              #
# --------------------------------------------------------------------------- #
def _euclid_um(node, a, b):
    dx = node[a]["x"] - node[b]["x"]
    dy = node[a]["y"] - node[b]["y"]
    dz = node[a]["z"] - node[b]["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def cap_arc(r_tip_um, h_um, n_points=48):
    """Sample the meridian of the cap: (du, rho) from the rim to the pole.

    du runs 0 -> h_um, measured from the rim plane. rho runs r_tip_um -> 0.
    Returned so a plot can draw the actual surface being added rather than a
    schematic of it. Purely a rendering aid: the area comes from Eq. (2), not
    from these samples.
    """
    if r_tip_um <= 0.0 or h_um <= 0.0:
        return [0.0], [max(r_tip_um, 0.0)]
    R = cap_sphere_radius_um(r_tip_um, h_um)
    z_c = h_um - R
    du, rho = [], []
    for i in range(n_points + 1):
        z = h_um * i / float(n_points)
        d = z - z_c
        inner = R * R - d * d
        rho.append(math.sqrt(inner) if inner > 0.0 else 0.0)
        du.append(z)
    rho[-1] = 0.0
    return du, rho


def spine_profile(node, children, spine_root_id, member_ids, h_um=None,
                  n_cap_points=48):
    """Meridian profile of one spine along its longest root-to-tip branch.

    Returns a dict with, for the primary branch only:
        u_um      cumulative path length from the BASE node (the shaft node the
                  spine hangs off), so u=0 is the shaft surface
        r_um      radius at each node
        labels    annotated_type of each node
        node_ids  the nodes traversed
        cap_u_um, cap_r_um   meridian of the cap, offset into the same u frame
                             (empty if h_um is None or the tip is not a
                             true leaf)
        A_base_um2  area of the base segment alone (shaft node -> spine root),
                    which spine_density attributes wholly to the spine
        capped    whether a cap was appended
        n_tips    number of true leaves in the whole spine (>1 means the
                  gallery is showing one branch of a branched spine)
        A_spine_um2, A_cap_um2   totals over the WHOLE spine, not just the
                                 branch drawn
    Distances are Euclidean per segment, matching spine_density exactly; no
    resampling and no interpolation is performed, because with a piecewise
    linear radius the frustum area is invariant under subdivision anyway.
    """
    member_set = set(member_ids)

    def depth(nid):
        kids = [c for c in children.get(nid, ()) if c in member_set]
        if not kids:
            return 0.0
        return max(_euclid_um(node, nid, c) + depth(c) for c in kids)

    chain = [spine_root_id]
    cur = spine_root_id
    while True:
        kids = [c for c in children.get(cur, ()) if c in member_set]
        if not kids:
            break
        cur = max(kids, key=lambda c: _euclid_um(node, cur, c) + depth(c))
        chain.append(cur)

    base = node[spine_root_id]["p"]
    u = []
    acc = 0.0
    if base in node:
        acc = 0.0
        prev = base
        for nid in chain:
            acc += _euclid_um(node, prev, nid)
            u.append(acc)
            prev = nid
        u = [0.0] + u
        r = [node[base]["r"]] + [node[i]["r"] for i in chain]
        labs = ["<base>"] + [node[i].get("label", "") for i in chain]
        ids = [base] + list(chain)
    else:
        for k, nid in enumerate(chain):
            if k:
                acc += _euclid_um(node, chain[k - 1], nid)
            u.append(acc)
        r = [node[i]["r"] for i in chain]
        labs = [node[i].get("label", "") for i in chain]
        ids = list(chain)

    tip = chain[-1]
    cap_u, cap_r = [], []
    capped = False
    if h_um is not None and is_true_leaf(children, tip) and node[tip]["r"] > 0:
        du, rho = cap_arc(node[tip]["r"], h_um, n_cap_points)
        cap_u = [u[-1] + d for d in du]
        cap_r = list(rho)
        capped = True

    a_spine = 0.0
    for m in member_ids:
        p = node[m]["p"]
        if p not in node:
            continue
        r1, r2 = node[p]["r"], node[m]["r"]
        L = _euclid_um(node, p, m)
        a_spine += math.pi * (r1 + r2) * math.sqrt((r1 - r2) ** 2 + L * L)
    # the base segment (shaft node -> spine root) is a frustum from the FULL
    # shaft radius down to the neck radius. spine_density attributes all of it
    # to the spine. On a thick shaft this single segment can dominate
    # A_spine, so it is reported separately here and drawn in a different
    # colour by the gallery: it is a property of the attribution convention,
    # not of the spine.
    a_base = 0.0
    base_id = node[spine_root_id]["p"]
    if base_id in node:
        r1, r2 = node[base_id]["r"], node[spine_root_id]["r"]
        L = _euclid_um(node, base_id, spine_root_id)
        a_base = math.pi * (r1 + r2) * math.sqrt((r1 - r2) ** 2 + L * L)

    a_cap = 0.0
    n_tips = 0
    for m in member_ids:
        if is_true_leaf(children, m):
            n_tips += 1
            if h_um is not None and node[m]["r"] > 0:
                a_cap += cap_area_um2(node[m]["r"], h_um)

    return {
        "root_id": spine_root_id, "node_ids": ids,
        "u_um": u, "r_um": r, "labels": labs,
        "cap_u_um": cap_u, "cap_r_um": cap_r, "capped": capped,
        "n_tips": n_tips, "n_nodes": len(member_set),
        "A_spine_um2": a_spine + a_cap, "A_cap_um2": a_cap,
        "A_base_um2": a_base,
    }


def spine_profiles(node, children, h_um=None, max_n=None, sort_by="area"):
    """Profiles for every spine in the cell.

    sort_by : "area" (largest first), "length", "n_nodes", or None (node order).
    max_n   : keep only the first max_n after sorting.
    """
    kids_of = children
    roots = []
    for nid, a in node.items():
        if not a["is_spine"]:
            continue
        p = a["p"]
        if p in node and node[p]["is_spine"]:
            continue
        roots.append(nid)

    out = []
    for rid in roots:
        stack, members = [rid], []
        while stack:
            n = stack.pop()
            members.append(n)
            stack.extend(c for c in kids_of.get(n, ())
                         if c in node and node[c]["is_spine"])
        out.append(spine_profile(node, kids_of, rid, members, h_um=h_um))

    if sort_by == "area":
        out.sort(key=lambda d: -d["A_spine_um2"])
    elif sort_by == "length":
        out.sort(key=lambda d: -(d["u_um"][-1] if d["u_um"] else 0.0))
    elif sort_by == "n_nodes":
        out.sort(key=lambda d: -d["n_nodes"])
    return out[:max_n] if max_n else out
