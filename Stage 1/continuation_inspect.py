"""continuation_inspect -- look at the components shaft_continuation calls shaft.

On neuron 15543554616, 312 of 2023 spine roots scored shaft-like and demoting
them moves F_lit by -0.143. Before that becomes a pipeline step, the question
is whether the 312 are what the scorer says they are. Several sit at
rho = 0.50-0.52, right on the calibre threshold; one is a 46 nm stub. This
module gives three views, all offline, all on the labelled frame CELL 6c
already holds:

  1. where the roots sit in the (rho, cos) plane relative to the thresholds,
     and how many are within a margin of either -- the boundary population;
  2. what each band of rho costs in F when demoted ALONE, so the -0.143 can
     be attributed: if most of it comes from rho >= 0.8 the decision is
     robust to the threshold; if it comes from the 0.50-0.55 band it is not;
  3. the local skeleton around any root -- the component, the branch point,
     the parent path and the siblings, with radii -- so a human can judge a
     handful of them by eye;
  4. the TAPER test, a third observable neither rho nor cos uses.

THE TAPER TEST
--------------
A dendritic branch thins monotonically from its base to its tip. A spine does
not: it has a thin neck and then a head, so its radius along the path has a
minimum and then RISES. The signature is therefore a distal radius maximum,
and it is computable from the skeleton alone.

For a component with root rho and nodes ordered by path distance s from the
root, let r(s) be the radius and define

    bulge = max_{s >= s_min} r(s) / min_{s <= argmax} r(s)                (1)
    s_peak_frac = argmax_s r(s) / s_max                                  (2)

A component is SPINE-LIKE BY TAPER when bulge >= BULGE_MIN and the peak sits
in the distal part (s_peak_frac >= PEAK_FRAC_MIN); it is BRANCH-LIKE when the
radius never recovers by that factor. This is orthogonal to calibre (which
compares the root to the BRANCH POINT, one step) and to collinearity (which is
purely directional), so a component that tapers monotonically AND passes rho
AND passes cos is a continuation with three independent votes.

The labeller's own head/neck tags give a free cross-check: a component
carrying a 'head' node should be spine-like by taper, and disagreement between
the two is reported rather than silently resolved.

MINIMUM LENGTH
--------------
Below MIN_LEN_NM the observables are not measurable: at ~300 nm node spacing a
60 nm component is one node essentially on top of its parent, so rho and cos
are both estimated from a baseline shorter than the skeleton jitter. Such
components are marked `undecidable` and no continuation decision is made for
them. On neuron 15543554616, 21 of the 312 shaft-like roots were under 100 nm.

Nothing here changes a label or writes to the bank.

Pure ASCII, LF only.
"""

import numpy as np
import pandas as pd

MODULE_VERSION = "continuation_inspect v1.1"

# Names morphology_exporter's step 4b needs. Checked by
# demote_shaft_continuations_three_vote at entry, so a stale copy of this file
# fails at the door with a readable message instead of raising AttributeError
# deep inside an export that has already run the labeller.
#   v1.0  boundary_population, dF_by_band, local_skeleton, pick_gallery
#   v1.1  + taper_table, three_vote, vote_summary, dF_by_cos_band, taper_figure
CAPABILITIES = ("taper_table", "three_vote", "vote_summary", "dF_by_band",
                "dF_by_cos_band", "boundary_population", "local_skeleton",
                "pick_gallery")

RHO_BANDS = ((0.50, 0.55), (0.55, 0.65), (0.65, 0.80), (0.80, 1.00),
             (1.00, np.inf))
COS_BANDS = ((0.70, 0.75), (0.75, 0.85), (0.85, 0.95), (0.95, 1.01))

MIN_LEN_NM = 150.0        # below this, rho and cos are not measurable
BULGE_MIN = 1.25          # distal max / preceding min, to call it a head
PEAK_FRAC_MIN = 0.30      # the peak must sit in the distal 70 percent


# --------------------------------------------------------------------------- #
def boundary_population(table, rho_min, cos_min, rho_margin=0.05,
                        cos_margin=0.05):
    """Counts of shaft-like roots that would flip if either threshold moved
    by its margin, and of near-miss spines that would flip the other way."""
    t = table.copy()
    sl = t["is_shaft"]
    near_rho = sl & (t["rho"] < rho_min + rho_margin)
    near_cos = sl & (t["cos"] < cos_min + cos_margin)
    miss_rho = (~sl) & t["rho_ok"].eq(False) & t["cos_ok"].eq(True) \
        & (t["rho"] >= rho_min - rho_margin)
    miss_cos = (~sl) & t["cos_ok"].eq(False) & t["rho_ok"].eq(True) \
        & (t["cos"] >= cos_min - cos_margin)
    return {"n_shaft_like": int(sl.sum()),
            "n_within_rho_margin": int(near_rho.sum()),
            "n_within_cos_margin": int(near_cos.sum()),
            "n_within_either": int((near_rho | near_cos).sum()),
            "n_near_miss_rho": int(miss_rho.sum()),
            "n_near_miss_cos": int(miss_cos.sum()),
            "rho_margin": rho_margin, "cos_margin": cos_margin}


def dF_by_band(labelled, table, sd, nid=None, cutoff_um=60.0, bands=RHO_BANDS):
    """Demote each rho band ALONE and report its F shift. Also by category.

    Bands are demoted independently (each from the exported frame), so the
    per-band shifts are not exactly additive in F, which is a ratio; their
    sum is reported next to the all-at-once shift for that reason."""
    import check_pruned_hoc as CK

    phi0 = sd.build_phi(labelled, nid=nid, input_units="nm")
    F0 = float(sd.cell_f_beyond_cutoff(phi0, cutoff_um=cutoff_um,
                                       by="d_from_um")["F"])
    A0 = float(phi0["spine_area_um2"].sum())
    sl = table[table["is_shaft"]]
    rows = []

    def _one(label, roots):
        if not len(roots):
            return {"group": label, "n_roots": 0, "n_nodes": 0,
                    "A_moved_um2": 0.0, "F_lit": F0, "dF_lit": 0.0}
        corr, n = CK.demote_roots(labelled, list(roots), sd)
        phi = sd.build_phi(corr, nid=nid, input_units="nm")
        F = float(sd.cell_f_beyond_cutoff(phi, cutoff_um=cutoff_um,
                                          by="d_from_um")["F"])
        return {"group": label, "n_roots": int(len(roots)), "n_nodes": int(n),
                "A_moved_um2": A0 - float(phi["spine_area_um2"].sum()),
                "F_lit": F, "dF_lit": F - F0}

    for lo, hi in bands:
        sel = sl[(sl["rho"] >= lo) & (sl["rho"] < hi)]
        rows.append(_one("rho [%.2f, %s)" % (lo, "inf" if np.isinf(hi) else "%.2f" % hi),
                         sel["root"].astype(int)))
    nan_sel = sl[~np.isfinite(sl["rho"])]
    if len(nan_sel):
        rows.append(_one("rho undefined", nan_sel["root"].astype(int)))
    for cat, sel in sl.groupby("category"):
        rows.append(_one("category %s" % cat, sel["root"].astype(int)))
    rows.append(_one("ALL shaft-like at once", sl["root"].astype(int)))
    out = pd.DataFrame(rows)
    out.attrs["F_lit_as_exported"] = F0
    out.attrs["sum_of_band_dF"] = float(
        out[out["group"].str.startswith("rho")]["dF_lit"].sum())
    return out


# --------------------------------------------------------------------------- #
def local_skeleton(labelled, root, n_up=8, sibling_depth=6):
    """Nodes around one root, with a `role` column:
    component (the scored subtree), branch_point, parent_path, sibling."""
    L = labelled.set_index("id")
    par = L["p"].astype(int).to_dict()
    kids = {}
    for i, p in par.items():
        kids.setdefault(p, []).append(i)
    root = int(root)
    bp = par[root]
    roles = {}

    def _down(start, role, depth=None):
        stack = [(start, 0)]
        while stack:
            n, d = stack.pop()
            if n in roles:
                continue
            roles[n] = role
            if depth is None or d < depth:
                stack.extend((k, d + 1) for k in kids.get(n, []))

    _down(root, "component")
    roles[bp] = "branch_point"
    p, k = bp, 0
    while p in par and par[p] in L.index and k < n_up:
        p = par[p]
        roles.setdefault(p, "parent_path")
        k += 1
    for sib in kids.get(bp, []):
        if sib != root:
            _down(sib, "sibling", sibling_depth)
    sub = L.loc[list(roles)].reset_index()
    sub["role"] = sub["id"].map(roles)
    return sub, bp


def pick_gallery(table, k=12, seed=0):
    """Stratified over rho bands, so the gallery spans the boundary cases and
    the unambiguous ones. Returns root ids."""
    sl = table[table["is_shaft"]].copy()
    if not len(sl):
        return []
    rng = np.random.default_rng(seed)
    sl["band"] = pd.cut(sl["rho"].fillna(-1), [-2, 0] + [b[1] for b in RHO_BANDS[:-1]] + [np.inf],
                        labels=False)
    per = int(np.ceil(k / max(sl["band"].nunique(), 1)))
    out = []
    for _, g in sl.groupby("band"):
        out.extend(rng.choice(g["root"].to_numpy(), size=min(per, len(g)),
                              replace=False).tolist())
    return [int(v) for v in out[:k]]


# --------------------------------------------------------------------------- #
def scatter_figure(table, rho_min, cos_min, title=None):
    """Plotly: every spine root in the (rho, cos) plane; shaft-like in red."""
    import plotly.graph_objects as go
    t = table[np.isfinite(table["rho"]) & np.isfinite(table["cos"])]
    fig = go.Figure()
    for lab, sel, col in (("spine", ~t["is_shaft"], "#4C72B0"),
                          ("shaft-like", t["is_shaft"], "#C44E52")):
        d = t[sel]
        fig.add_trace(go.Scattergl(
            x=d["rho"], y=d["cos"], mode="markers", name=lab,
            marker=dict(size=5, color=col, opacity=0.6),
            text=["root %d  len %.0f nm  %s" % (r, l, c) for r, l, c in
                  zip(d["root"], d["own_len_nm"], d["category"])],
            hoverinfo="text"))
    fig.add_vline(x=rho_min, line_dash="dash", line_color="#333")
    fig.add_hline(y=cos_min, line_dash="dash", line_color="#333")
    fig.update_xaxes(title_text="rho = r(root) / r(branch point)", range=[0, 2])
    fig.update_yaxes(title_text="cos(angle to parent segment)", range=[-1, 1])
    fig.update_layout(title=title or "spine roots: calibre vs collinearity",
                      height=480)
    return fig


def local_figure(sub, root, bp, title=None):
    """Plotly 3D of the local skeleton, marker size ~ radius, coloured by role."""
    import plotly.graph_objects as go
    cols = {"component": "#C44E52", "branch_point": "#111111",
            "parent_path": "#DD8452", "sibling": "#4C72B0"}
    pos = sub.set_index("id")
    fig = go.Figure()
    xs, ys, zs = [], [], []
    for r in sub.itertuples(index=False):
        if int(r.p) in pos.index:
            q = pos.loc[int(r.p)]
            xs += [r.x, q["x"], None]
            ys += [r.y, q["y"], None]
            zs += [r.z, q["z"], None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                               line=dict(color="#999", width=2), name="edges",
                               showlegend=False))
    rmax = max(float(sub["r"].max()), 1.0)
    for role, col in cols.items():
        d = sub[sub["role"] == role]
        if not len(d):
            continue
        fig.add_trace(go.Scatter3d(
            x=d["x"], y=d["y"], z=d["z"], mode="markers", name=role,
            marker=dict(size=3 + 12 * d["r"] / rmax, color=col, opacity=0.85),
            text=["id %d  r %.0f nm  %s" % (i, rr, a) for i, rr, a in
                  zip(d["id"], d["r"], d["annotated_type"])],
            hoverinfo="text"))
    fig.update_layout(title=title or "root %d at branch point %d" % (root, bp),
                      scene=dict(aspectmode="data"), height=520,
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig


# --------------------------------------------------------------------------- #
# 4. the taper test                                                            #
# --------------------------------------------------------------------------- #
def _component_paths(labelled, roots):
    """{root: DataFrame(id, s_nm, r, annotated_type)} along the LONGEST path of
    each component, ordered from the root outward. The longest path is the one
    a spine's neck-then-head profile lives on; side twigs of a component would
    otherwise mix two profiles into one sequence."""
    L = labelled.set_index("id")
    par = L["p"].astype(np.int64).to_dict()
    xyz = L[["x", "y", "z"]].to_numpy(dtype=float)
    rad = L["r"].to_numpy(dtype=float)
    ann = L["annotated_type"].astype(str).to_numpy()
    pos = {int(v): i for i, v in enumerate(L.index.to_numpy())}
    kids = {}
    for i, q in par.items():
        kids.setdefault(int(q), []).append(int(i))
    spine = set(labelled.loc[
        labelled["compartment_class"].astype(str).str.lower().str.contains("spine")
        | labelled["annotated_type"].astype(str).str.lower()
        .isin(("spine", "head", "neck")), "id"].astype(int))
    out = {}
    for r0 in roots:
        r0 = int(r0)
        if r0 not in pos:
            continue
        best, stack = [], [(r0, [r0], 0.0)]
        while stack:
            n, path, s = stack.pop()
            ch = [c for c in kids.get(n, ()) if c in spine and c in pos]
            if not ch:
                if s >= (best[1] if best else -1.0):
                    best = (path, s)
                continue
            for c in ch:
                d = float(np.linalg.norm(xyz[pos[c]] - xyz[pos[n]]))
                stack.append((c, path + [c], s + d))
        if not best:
            continue
        path = best[0]
        ss, acc = [0.0], 0.0
        for a, b in zip(path[:-1], path[1:]):
            acc += float(np.linalg.norm(xyz[pos[b]] - xyz[pos[a]]))
            ss.append(acc)
        out[r0] = pd.DataFrame({"id": path, "s_nm": ss,
                                "r": [rad[pos[i]] for i in path],
                                "annotated_type": [ann[pos[i]] for i in path]})
    return out


def taper_table(labelled, roots, min_len_nm=MIN_LEN_NM, bulge_min=BULGE_MIN,
                peak_frac_min=PEAK_FRAC_MIN):
    """Eq. (1) and (2) per component. One row per root.

    verdict: 'spine_like'    a distal radius maximum -- a head
             'branch_like'   radius never recovers -- a taper
             'undecidable'   shorter than min_len_nm, or fewer than 3 nodes
    """
    paths = _component_paths(labelled, roots)
    rows = []
    for r0 in [int(v) for v in roots]:
        d = paths.get(r0)
        if d is None or len(d) < 2:
            rows.append({"root": r0, "n_path": 0 if d is None else len(d),
                         "path_len_nm": 0.0, "bulge": np.nan,
                         "s_peak_frac": np.nan, "r_root_nm": np.nan,
                         "r_min_nm": np.nan, "r_peak_nm": np.nan,
                         "has_head_label": False, "verdict": "undecidable",
                         "why": "fewer than 2 path nodes"})
            continue
        r = d["r"].to_numpy(dtype=float)
        s = d["s_nm"].to_numpy(dtype=float)
        head = bool(d["annotated_type"].str.lower().eq("head").any())
        L = float(s[-1])
        # Two nodes suffice: root and tip already distinguish a rise from a
        # taper. Requiring three would abstain on the many 2-node components
        # real H01 spines produce.
        if L < float(min_len_nm):
            rows.append({"root": r0, "n_path": len(d), "path_len_nm": L,
                         "bulge": np.nan, "s_peak_frac": np.nan,
                         "r_root_nm": float(r[0]), "r_min_nm": float(r.min()),
                         "r_peak_nm": float(r.max()), "has_head_label": head,
                         "verdict": "undecidable",
                         "why": "shorter than %.0f nm" % min_len_nm})
            continue
        k = int(np.argmax(r))
        r_min_before = float(r[:k + 1].min())
        bulge = float(r[k] / max(r_min_before, 1e-30))
        frac = float(s[k] / max(L, 1e-30))
        spine_like = (bulge >= float(bulge_min)) and (frac >= float(peak_frac_min))
        rows.append({"root": r0, "n_path": len(d), "path_len_nm": L,
                     "bulge": bulge, "s_peak_frac": frac,
                     "r_root_nm": float(r[0]), "r_min_nm": r_min_before,
                     "r_peak_nm": float(r[k]), "has_head_label": head,
                     "verdict": "spine_like" if spine_like else "branch_like",
                     "why": ("bulge %.2f at %.0f%% of the path"
                             % (bulge, 100 * frac))})
    return pd.DataFrame(rows)


def three_vote(table, taper):
    """Join the scorer's table with the taper verdict and resolve.

    A root is demoted only when the two current observables agree AND the
    taper does not object: is_shaft and verdict == 'branch_like'. Anything
    'undecidable' is KEPT as a spine, which is the conservative direction --
    it leaves membrane in the spine bucket rather than moving a protrusion
    into the cable.
    """
    t = table.merge(taper, on="root", how="left", suffixes=("", "_tap"))
    t["verdict"] = t["verdict"].fillna("undecidable")
    t["demote"] = t["is_shaft"] & t["verdict"].eq("branch_like")
    t["decision"] = np.where(
        ~t["is_shaft"], "spine (rho/cos)",
        np.where(t["verdict"].eq("branch_like"), "DEMOTE (three votes)",
                 np.where(t["verdict"].eq("spine_like"),
                          "spine (taper overrides rho/cos)",
                          "spine (undecidable: too short)")))
    return t


def vote_summary(t):
    sl = t[t["is_shaft"]]
    return {"n_roots": int(len(t)), "n_shaft_like_rho_cos": int(len(sl)),
            "n_demote_three_vote": int(t["demote"].sum()),
            "n_rescued_by_taper": int((sl["verdict"] == "spine_like").sum()),
            "n_undecidable": int((sl["verdict"] == "undecidable").sum()),
            "n_head_label_among_shaft_like": int(sl["has_head_label"].sum()),
            "n_taper_vs_headlabel_disagree": int(
                (sl["has_head_label"] & sl["verdict"].eq("branch_like")).sum()),
            "decision_counts": t["decision"].value_counts().to_dict()}


def dF_by_cos_band(labelled, table, sd, nid=None, cutoff_um=60.0,
                   bands=COS_BANDS):
    """dF_by_band, on the collinearity axis -- the soft one."""
    import check_pruned_hoc as CK

    phi0 = sd.build_phi(labelled, nid=nid, input_units="nm")
    F0 = float(sd.cell_f_beyond_cutoff(phi0, cutoff_um=cutoff_um,
                                       by="d_from_um")["F"])
    A0 = float(phi0["spine_area_um2"].sum())
    sl = table[table["is_shaft"]]
    rows = []
    for lo, hi in bands:
        sel = sl[(sl["cos"] >= lo) & (sl["cos"] < hi)]
        if not len(sel):
            rows.append({"group": "cos [%.2f, %.2f)" % (lo, hi), "n_roots": 0,
                         "n_nodes": 0, "A_moved_um2": 0.0, "F_lit": F0,
                         "dF_lit": 0.0})
            continue
        corr, n = CK.demote_roots(labelled, sel["root"].astype(int).tolist(), sd)
        phi = sd.build_phi(corr, nid=nid, input_units="nm")
        F = float(sd.cell_f_beyond_cutoff(phi, cutoff_um=cutoff_um,
                                          by="d_from_um")["F"])
        rows.append({"group": "cos [%.2f, %.2f)" % (lo, hi),
                     "n_roots": int(len(sel)), "n_nodes": int(n),
                     "A_moved_um2": A0 - float(phi["spine_area_um2"].sum()),
                     "F_lit": F, "dF_lit": F - F0})
    out = pd.DataFrame(rows)
    out.attrs["F_lit_as_exported"] = F0
    return out


def taper_figure(labelled, roots, taper, max_n=40, title=None):
    """r(s) along each component's longest path, normalised by the root radius;
    branch-like in one colour, spine-like in another. The two shapes should
    separate by eye: monotone decay vs a dip and a rise."""
    import plotly.graph_objects as go
    paths = _component_paths(labelled, list(roots)[:max_n])
    tv = taper.set_index("root")["verdict"].to_dict()
    cols = {"branch_like": "#C44E52", "spine_like": "#4C72B0",
            "undecidable": "#BBBBBB"}
    fig = go.Figure()
    seen = set()
    for r0, d in paths.items():
        v = tv.get(r0, "undecidable")
        fig.add_trace(go.Scatter(
            x=d["s_nm"], y=d["r"] / max(float(d["r"].iloc[0]), 1e-30),
            mode="lines+markers", name=v, legendgroup=v,
            showlegend=v not in seen, line=dict(color=cols[v], width=1.5),
            marker=dict(size=4), opacity=0.75,
            text=["root %d  id %d  r %.0f nm  %s" % (r0, i, rr, a)
                  for i, rr, a in zip(d["id"], d["r"], d["annotated_type"])],
            hoverinfo="text"))
        seen.add(v)
    fig.add_hline(y=1.0, line_dash="dot", line_color="#888")
    fig.update_xaxes(title_text="path distance from the root (nm)")
    fig.update_yaxes(title_text="r(s) / r(root)")
    fig.update_layout(title=title or "radius along the component: a branch "
                      "tapers, a spine dips then rises", height=440)
    return fig
