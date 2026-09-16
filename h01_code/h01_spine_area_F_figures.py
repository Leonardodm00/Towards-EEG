"""Static per-spine QC figure: three orthogonal projections. Plotting only.

Static (matplotlib PNG) companion to the interactive h01_spine_roi_figures,
for batches: one file per spine that can be flipped through, or opened on a
machine with no notebook. Colours are imported from h01_spine_roi_figures so
the two cannot drift apart:

    BLUE   spine          (mask_local_to_spine's kept half)
    AMBER  shaft          same-cell material TOUCHING the spine
    GREY   detached       same segment id, touching nothing (faint)
    RED    bridge         INTERPOLATED voxels, not in the segmentation

WHY PROJECTIONS AND NOT SLICES
A slice through the centroid can miss the neck or the bridge entirely. A
maximum projection along each axis cannot: every voxel of every class
appears in all three panels. Along each ray the class drawn is the highest
priority present, bridge > spine > shaft > detached, so the interpolated part
is never hidden behind real tissue.

WHAT ELSE IS DRAWN
  * skeleton nodes of sigma (orange line) and the shaft base node (black)
  * optionally the centroids of the triangles h01_spine_area_F classified as
    CUT (purple x) -- the artificial face excluded from the area. They should
    sit on the blue/amber interface. If they sit anywhere else, the
    classifier is wrong for this spine.

FRAME. Voxel i of the cutout spans [(lo+i) res, (lo+i+1) res) in the node
frame, so the image extent is [lo res, (lo+n) res]. Mesh coordinates carry
the -res/2 offset of surface_from_mask and are shifted back by +res/2 here.

Pixel aspect is physical (8 x 8 x 33 nm voxels), axes in micrometres.

DEPENDENCIES: numpy, matplotlib. Pure ASCII, LF only.
"""

import numpy as np

import h01_spine_roi_figures as RF

MODULE_VERSION = "h01_spine_area_F_figures v1.0"

NM_PER_UM = 1000.0
CUT_COLOR = "#6A3D9A"
NODE_COLOR = "#E4572E"

CODE_BG, CODE_DETACHED, CODE_SHAFT, CODE_SPINE, CODE_BRIDGE = 0, 1, 2, 3, 4
PLANES = (("XY", 2, (0, 1)), ("XZ", 1, (0, 2)), ("YZ", 0, (1, 2)))
AXIS_NAMES = "xyz"


def label_volume(roi):
    """int8 volume of display codes, priority encoded by magnitude.

    "Shaft" = same-cell material touching the spine OR the bridge. The ROI's
    own shaft_connected_mask is deliberately NOT used: extract_spine_roi
    computes it from contact with the spine alone, before bridging, so for a
    bridged spine the very shaft it is bridged to would be drawn as detached.
    """
    import h01_spine_roi as SR

    S = np.asarray(roi["mask"], dtype=bool)
    Bm = (np.zeros_like(S) if roi.get("bridge_mask") is None
          else np.asarray(roi["bridge_mask"], dtype=bool))
    L = np.zeros(S.shape, dtype=np.int8)
    ctx = roi.get("shaft_context_mask")
    if ctx is not None and np.asarray(ctx).any():
        ctx = np.asarray(ctx, dtype=bool)
        con, _, _ = SR.split_excluded_by_contact(S | Bm, ctx)
        con = np.asarray(con, dtype=bool)
        L[ctx & ~con] = CODE_DETACHED
        L[con] = CODE_SHAFT
    L[S] = CODE_SPINE
    L[Bm] = CODE_BRIDGE
    return L


def crop_box(roi, margin_nm, resolution_nm):
    """Index box around spine | bridge, grown by margin_nm, clipped to array."""
    S = np.asarray(roi["mask"], dtype=bool)
    if roi.get("bridge_mask") is not None:
        S = S | np.asarray(roi["bridge_mask"], dtype=bool)
    idx = np.argwhere(S)
    m = np.ceil(float(margin_nm) / np.asarray(resolution_nm, float)).astype(int)
    lo = np.maximum(idx.min(axis=0) - m, 0)
    hi = np.minimum(idx.max(axis=0) + 1 + m, np.asarray(S.shape))
    return lo, hi


def spine_planes_figure(roi, detail=None, rec=None, spine_nodes=None,
                        base_node=None, margin_nm=400.0, show_cut=True,
                        title=None, dpi=110, max_cut_markers=400):
    """Three-panel matplotlib Figure. Returns (fig, info).

    roi    : the dict h01_spine_area_F.get_spine_roi returns
    detail : the dict measure_spine_area(..., return_detail=True) returns;
             needed only for the cut-face overlay
    rec    : the per-spine record, for the title line
    info   : per-class voxel counts inside the crop, and the median
             perpendicular offset of the cut centroids from the spine/shaft
             voxel faces (a numeric "the purple crosses are on the seam")
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    seg = roi["meta"]["layers"]["seg"]
    res = np.asarray(seg["resolution_nm"], dtype=float)
    lo_vox = np.asarray(seg["lo_vox"], dtype=np.int64)
    L = label_volume(roi)
    c_lo, c_hi = crop_box(roi, margin_nm, res)
    Lc = L[c_lo[0]:c_hi[0], c_lo[1]:c_hi[1], c_lo[2]:c_hi[2]]
    origin_nm = (lo_vox + c_lo) * res
    top_nm = (lo_vox + c_hi) * res

    cmap = ListedColormap(["#FFFFFF", RF.DETACHED_COLOR, RF.SHAFT_COLOR,
                           RF.SPINE_COLOR, RF.BRIDGE_COLOR])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=dpi)

    cut_pts = None
    if show_cut and detail is not None:
        cls = np.asarray(detail["cls"])
        import h01_spine_area_F as SAF
        sel = cls == SAF.CLASS_CUT
        if sel.any():
            cut_pts = np.asarray(detail["centroid_nm"])[sel] + 0.5 * res
    # Draw at most max_cut_markers crosses (deterministic stride): a cut face
    # seen face-on otherwise carpets whatever lies behind it. The QC number
    # in `info` still uses every cut triangle.
    cut_draw = (None if cut_pts is None else
                cut_pts[::max(1, int(np.ceil(len(cut_pts) / float(max_cut_markers))))])

    sn = (None if spine_nodes is None or len(spine_nodes) == 0 else
          spine_nodes[["x", "y", "z"]].to_numpy(dtype=float))
    bn = (None if base_node is None or len(base_node) == 0 else
          base_node[["x", "y", "z"]].to_numpy(dtype=float))

    for ax, (name, proj_axis, (ia, ib)) in zip(axes, PLANES):
        img = Lc.max(axis=proj_axis)              # priority = max code
        ext = [origin_nm[ia] / NM_PER_UM, top_nm[ia] / NM_PER_UM,
               origin_nm[ib] / NM_PER_UM, top_nm[ib] / NM_PER_UM]
        ax.imshow(img.T, origin="lower", extent=ext, cmap=cmap, vmin=-0.5,
                  vmax=4.5, interpolation="nearest", aspect="equal")
        if cut_draw is not None:
            ax.plot(cut_draw[:, ia] / NM_PER_UM, cut_draw[:, ib] / NM_PER_UM,
                    "x", ms=2.5, mew=0.6, color=CUT_COLOR, alpha=0.45)
        if sn is not None:
            ax.plot(sn[:, ia] / NM_PER_UM, sn[:, ib] / NM_PER_UM, "-o",
                    ms=3, lw=1.2, color=NODE_COLOR)
        if bn is not None:
            ax.plot(bn[:, ia] / NM_PER_UM, bn[:, ib] / NM_PER_UM, "D", ms=6,
                    color="#111111")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("%s (um)" % AXIS_NAMES[ia])
        ax.set_ylabel("%s (um)" % AXIS_NAMES[ib])
        ax.set_title("%s  (max projection along %s)" % (name, AXIS_NAMES[proj_axis]),
                     fontsize=10)

    handles = [Patch(color=RF.SPINE_COLOR, label="spine"),
               Patch(color=RF.SHAFT_COLOR, label="shaft (touching spine/bridge)"),
               Patch(color=RF.DETACHED_COLOR, label="same id, detached"),
               Patch(color=RF.BRIDGE_COLOR, label="bridge (interpolated)"),
               Line2D([], [], color=NODE_COLOR, marker="o", label="skeleton"),
               Line2D([], [], color="#111111", marker="D", ls="", label="base node")]
    if cut_pts is not None:
        handles.append(Line2D([], [], color=CUT_COLOR, marker="x", ls="",
                              label="cut face (excluded)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=False)
    if title is None and rec is not None:
        title = ("sigma %s   A_mesh %.3f um2 (raw %.3f, g %.3f)   cut %.1f%%   "
                 "bridge %.1f%%   box %.2f%%   %s"
                 % (rec.get("sigma_id", "?"), rec.get("A_mesh_um2", np.nan),
                    rec.get("A_mesh_raw_um2", np.nan), rec.get("g_aw_mean", np.nan),
                    100 * rec.get("frac_cut", np.nan),
                    100 * rec.get("frac_bridge", np.nan),
                    100 * rec.get("frac_box", np.nan),
                    "cache" if rec.get("from_cache") else "fetched"))
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    info = {"voxels_in_crop": {n: int((Lc == c).sum()) for n, c in
                               (("detached", CODE_DETACHED), ("shaft", CODE_SHAFT),
                                ("spine", CODE_SPINE), ("bridge", CODE_BRIDGE))},
            "n_cut_points": 0 if cut_pts is None else int(len(cut_pts))}
    if cut_pts is not None:
        info["cut_to_seam_median_nm"] = _seam_distance(L, lo_vox, res, cut_pts)
    return fig, info            # the CALLER closes it (save_figure does)


def _seam_distance(L, lo_vox, res, pts_nm):
    """Median PERPENDICULAR offset (nm) of cut centroids from the seam.

    The seam is the set of voxel faces shared by an object voxel (spine or
    bridge) and a shaft/detached voxel. For each point the nearest face
    centre is found in 3D; the offset reported is along that face's normal
    axis only. Distance to voxel CENTRES is the wrong measure: centres sit
    half a voxel off the face, and lateral grid phase then dominates.
    Node frame: voxel i spans [(lo+i) res, (lo+i+1) res).
    """
    from scipy.spatial import cKDTree

    obj = L >= CODE_SPINE
    ctx = (L == CODE_SHAFT) | (L == CODE_DETACHED)
    lo = np.asarray(lo_vox, dtype=float)
    cen, axis = [], []
    for ax in range(3):
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[ax] = slice(0, -1)
        sl_b[ax] = slice(1, None)
        pair = (obj[tuple(sl_a)] & ctx[tuple(sl_b)]) | \
               (ctx[tuple(sl_a)] & obj[tuple(sl_b)])
        idx = np.argwhere(pair).astype(float)
        if len(idx) == 0:
            continue
        c = (idx + lo + 0.5) * res
        c[:, ax] = (idx[:, ax] + lo[ax] + 1.0) * res[ax]      # the shared face
        cen.append(c)
        axis.append(np.full(len(c), ax))
    if not cen:
        return float("nan")
    cen, axis = np.vstack(cen), np.concatenate(axis)
    pts = np.asarray(pts_nm, dtype=float)
    _, k = cKDTree(cen).query(pts, k=1)
    off = np.abs(pts[np.arange(len(pts)), axis[k]] - cen[k, axis[k]])
    return float(np.median(off))


def save_figure(fig, path):
    """Write the PNG and release the figure. Returns the path."""
    import matplotlib.pyplot as plt
    fig.savefig(path)
    plt.close(fig)
    return path


def make_figure_callback(nodes, comp, fig_dir, show_inline=False,
                         **figure_kw):
    """The on_success callable for h01_spine_area_F.measure_all_spines.

    Draws the three-plane figure while the ROI is still in memory, writes
    fig_dir/sigmaNNNNN.png, stores the seam offset in the spine's record
    (cut_to_seam_median_nm), and optionally displays it inline (Colab).
    """
    import os
    import h01_spine_roi as SR

    os.makedirs(fig_dir, exist_ok=True)
    disp = None
    if show_inline:
        try:
            from IPython.display import display as disp
        except ImportError:
            disp = None

    def on_success(sid, roi, rec, detail):
        sn, bn, _ = SR.spine_subframe(nodes, comp, sid)
        fig, info = spine_planes_figure(roi, detail, rec, sn, bn, **figure_kw)
        rec["cut_to_seam_median_nm"] = info.get("cut_to_seam_median_nm")
        if disp is not None:
            disp(fig)
        return save_figure(fig, os.path.join(fig_dir, "sigma%05d.png" % int(sid)))
    return on_success


def union_profile_figure(profile, base_rec, spine_profile=None, title=None,
                         rind_tol_nm=None):
    """Interactive (plotly) trace of the UNION-mesh cross sections from the base
    node outward, with the shaft stations shaded, the measured base marked, and
    -- if given -- the ordinary spine-mesh profile overlaid for comparison.
    Same axes as h01_spine_geometry_figures.area_profile."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    s = np.array([r["s_nm"] for r in profile], float) / NM_PER_UM
    a = np.array([r["area_nm2"] for r in profile], float)
    touch = np.array([bool(r.get("touches_box", False)) for r in profile])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if touch.any():
        fig.add_vrect(x0=s[touch].min(), x1=s[touch].max() + 0.5 * (s[1] - s[0]),
                      fillcolor=RF.SHAFT_COLOR, opacity=0.18, line_width=0,
                      annotation_text="shaft (loop touches cutout box)",
                      annotation_position="top left")
    fig.add_trace(go.Scatter(x=s, y=a, mode="lines+markers", name="union mesh",
                             line=dict(color=RF.SHAFT_COLOR)), secondary_y=False)
    if spine_profile is not None:
        s2 = np.array([r["s_nm"] for r in spine_profile], float) / NM_PER_UM
        a2 = np.array([r["area_nm2"] for r in spine_profile], float)
        fig.add_trace(go.Scatter(x=s2, y=a2, mode="lines+markers",
                                 name="spine mesh (Voronoi cut)",
                                 line=dict(color=RF.SPINE_COLOR, dash="dot")),
                      secondary_y=False)
    fig.add_trace(go.Scatter(x=s, y=np.sqrt(np.maximum(a, 0) / np.pi),
                             mode="lines", name="equivalent radius",
                             line=dict(color="#999999", dash="dash")),
                  secondary_y=True)
    sb = base_rec.get("s_base_nm", np.nan)
    if np.isfinite(sb):
        fig.add_vline(x=sb / NM_PER_UM, line_dash="dash", line_color=CUT_COLOR,
                      annotation_text="measured base %.0f nm (%s)"
                      % (sb, base_rec.get("base_method", "?")))
    else:
        fig.add_annotation(x=0.5, y=0.95, xref="paper", yref="paper",
                           showarrow=False, font=dict(color=RF.BRIDGE_COLOR, size=13),
                           text="NO BASE: %s -- cross-sections never separate "
                                "from the dendrite" % base_rec.get("verdict", "?"))
    r_sh = base_rec.get("r_shaft_nm", np.nan)
    if np.isfinite(r_sh):
        # Where the CYLINDER model puts the junction: one shaft radius along
        # the centreline. It coincides with the measured base only for a spine
        # leaving radially; in general s_base = r_shaft / sin(alpha) >= r_shaft.
        fig.add_vline(x=r_sh / NM_PER_UM, line_dash="dot", line_color="#111111",
                      annotation_text="r_shaft %.0f nm (cylinder model)" % r_sh,
                      annotation_position="bottom right")
        if rind_tol_nm is not None:
            fig.add_vrect(x0=0.0, x1=(r_sh + rind_tol_nm) / NM_PER_UM,
                          fillcolor=CUT_COLOR, opacity=0.10, line_width=0,
                          annotation_text="cylinder rind", layer="below",
                          annotation_position="top right")
        fig.add_hline(y=r_sh, line_dash="dot", line_color="#111111",
                      opacity=0.5, secondary_y=True)
    for key, lab, col in (("r_eq_base_nm", "neck radius at the base", "#2E8B57"),):
        v = base_rec.get(key, np.nan)
        if np.isfinite(v):
            fig.add_hline(y=v, line_dash="dash", line_color=col, opacity=0.6,
                          secondary_y=True, annotation_text="%s %.0f nm" % (lab, v),
                          annotation_position="right")
    fig.update_xaxes(title_text="arc length from the base node (um)")
    fig.update_yaxes(title_text="area (nm^2)", secondary_y=False)
    fig.update_yaxes(title_text="equivalent radius (nm)", secondary_y=True)
    fig.update_layout(title=dict(text=title or "union-mesh cross sections: "
                                 "shaft slab -> neck"), height=420)
    return fig


def union_mesh_3d(verts_nm, faces, regions, base_rec=None, spine_nodes=None,
                  base_node=None, title=None, opacity=None, show=("spine",
                  "rind", "bridge", "shaft", "detached")):
    """Interactive 3D of the UNION mesh, one plotly trace per region.

    Regions come from h01_spine_base.classify_union_triangles. One trace each
    so the legend toggles them: switching `shaft` off is how you look inside
    the junction and see the rind from the dendrite side.

    Shaft is drawn semi-transparent by default because it is the largest
    surface and would otherwise hide everything; spine, rind and bridge are
    opaque. The base plane is drawn as a translucent quad when base_rec is
    given, so the cut the areas were split on is visible rather than implied.
    """
    import numpy as np
    import plotly.graph_objects as go
    import h01_spine_base as SB

    col = {"spine": RF.SPINE_COLOR, "rind": CUT_COLOR,
           "bridge": RF.BRIDGE_COLOR, "shaft": RF.SHAFT_COLOR,
           "detached": RF.DETACHED_COLOR, "unresolved": "#000000"}
    opa = dict({"spine": 1.0, "rind": 1.0, "bridge": 1.0, "shaft": 0.25,
                "detached": 0.15, "unresolved": 1.0}, **(opacity or {}))
    v = np.asarray(verts_nm, dtype=float) / NM_PER_UM
    f = np.asarray(faces, dtype=np.int64)
    reg = np.asarray(regions["region"])
    a = regions["area_um2_by_region"]
    fig = go.Figure()
    for i, name in enumerate(SB.REGION_NAMES):
        if name not in show:
            continue
        sel = reg == i
        if not sel.any():
            continue
        ff = f[sel]
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=ff[:, 0], j=ff[:, 1], k=ff[:, 2],
            color=col[name], opacity=opa[name], flatshading=True,
            name="%s  %.3f um2" % (name, a.get(name, 0.0)), showlegend=True,
            hoverinfo="name"))
    if spine_nodes is not None and len(spine_nodes):
        p = spine_nodes[["x", "y", "z"]].to_numpy(dtype=float) / NM_PER_UM
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2],
                                   mode="lines+markers", name="skeleton",
                                   line=dict(color=NODE_COLOR, width=4),
                                   marker=dict(size=3, color=NODE_COLOR)))
    if base_node is not None and len(base_node):
        p = base_node[["x", "y", "z"]].to_numpy(dtype=float) / NM_PER_UM
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2],
                                   mode="markers", name="base node",
                                   marker=dict(size=6, color="#111111",
                                               symbol="diamond")))
    if base_rec is not None and base_rec.get("base_point_nm") is not None:
        c = np.asarray(base_rec["base_point_nm"], dtype=float) / NM_PER_UM
        n = np.asarray(base_rec["base_tangent"], dtype=float)
        n = n / max(float(np.linalg.norm(n)), 1e-30)
        e1 = np.cross(n, [0.0, 0.0, 1.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(n, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        h = 0.35
        q = np.array([c + h * (su * e1 + sv * e2)
                      for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        fig.add_trace(go.Mesh3d(x=q[:, 0], y=q[:, 1], z=q[:, 2],
                                i=[0, 0], j=[1, 2], k=[2, 3],
                                color="#6A3D9A", opacity=0.25,
                                name="base plane", showlegend=True,
                                hoverinfo="name"))
    fig.update_layout(title=title or "union mesh by region",
                      scene=dict(aspectmode="data",
                                 xaxis_title="x (um)", yaxis_title="y (um)",
                                 zaxis_title="z (um)"),
                      height=620, margin=dict(l=0, r=0, t=40, b=0))
    return fig
