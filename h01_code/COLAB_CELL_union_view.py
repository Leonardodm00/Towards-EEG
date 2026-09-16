# %% CELL UNION -- inspect a few spines: union mesh by region + profile  [NETWORK]
# Paste after the SAF cell. Independent of it: it fetches its own ROIs, so it
# runs on any spine whether or not the SAF batch has measured it.
#
# WHAT YOU GET, per selected spine
#   1. an interactive 3D of the UNION mesh (spine + bridge + shaft), one
#      plotly trace per region so the legend toggles them:
#         blue    spine     membrane beyond the base plane -- the spine's own
#         purple  rind      spine-masked surface that is really dendrite,
#                           handed over by the Voronoi cut. Already counted in
#                           the shaft frustum, so counting it again inflates F
#         red     bridge    INTERPOLATED voxels, not in the segmentation
#         amber   shaft     dendrite touching the spine (semi-transparent)
#         grey    detached  same segid, touching nothing
#      plus the skeleton, the base node, and the BASE PLANE itself as a
#      translucent quad -- the cut the areas were split on, drawn not implied.
#      Turn `shaft` off in the legend to look at the rind from inside.
#   2. the cross-section profile of the same union mesh, with every threshold
#      that enters the decision drawn on it:
#         shaded band  stations whose section touches the cutout box = shaft
#         dashed       the MEASURED base, s_base
#         dotted       r_shaft, where the CYLINDER model puts the junction
#         tinted       the cylinder rind region, r_shaft + tolerance
#         horizontal   r_shaft and the neck radius at the base
#
# The two rind rules disagreed by ~14% on real spines (cylinder removes less),
# so RIND_RULE says which one produced the picture. Switch it and re-run to
# see the difference directly.
#
# Needs in CODE_DIR: h01_spine_area_F.py (v1.5+), h01_spine_base.py,
# h01_spine_area_F_figures.py, and the calibration table.
import os, sys, json, importlib
import numpy as np, pandas as pd

SIGMA_IDS   = [3, 3588]      # [] = pick automatically (see AUTO_PICK below)
AUTO_PICK   = 4              # how many to choose if SIGMA_IDS is empty
AUTO_BY     = "rind"         # "rind" = largest frac_rind in the SAF ledger,
                             # "area" = largest A_skel, "random"
RIND_RULE   = "plane"        # "plane" (measured) or "cylinder" (modelled)
RIND_TOL_NM = None           # None -> one in-plane voxel (8 nm); 0 = v1.4
PAD_NM      = 500.0
SHOW_SHAFT  = True           # False = omit the shaft trace entirely
OUT_UNION   = os.path.join(P["OUT"], "union_view")      # noqa: F821
os.makedirs(OUT_UNION, exist_ok=True)

import h01_spine_area_F as SAF, h01_spine_base as SB
import h01_spine_area_F_figures as SAFF
import h01_spine_batch as B, h01_spine_roi as SR
for _m in (SAF, SB, SAFF, B, SR):
    importlib.reload(_m)

_table, _g, _cal = SAF.load_calibration(G_TABLE)        # noqa: F821
_sk = SAF.skeleton_spine_table(sd, labelled, nodes, comp)   # noqa: F821
_axes = SAF.shaft_axes_from_table(_sk)
# Reuse the SAF cell's reader factory when it is in the namespace: it carries
# a warm CloudVolume per layer, so re-inspecting a spine the batch already
# fetched costs no new connections. Falls back to a fresh one.
_fac = globals().get("reader_factory") or SAF.memoized_reader_factory()
_opts = dict(B.DEFAULTS)

# ---- choose the spines -----------------------------------------------------
_ids = [int(v) for v in SIGMA_IDS]
if not _ids:
    _csv = os.path.join(P["OUT"], "spine_area_F",                 # noqa: F821
                        "cell%d_spines.csv" % CELL_ID)            # noqa: F821
    if AUTO_BY == "rind" and os.path.isfile(_csv):
        _d = pd.read_csv(_csv)
        _d = _d[_d.get("measured", False) == True]                # noqa: E712
        _ids = _d.nlargest(AUTO_PICK, "frac_rind")["sigma_id"].astype(int).tolist()
        print("picked the %d largest frac_rind from the SAF ledger" % len(_ids))
    elif AUTO_BY == "area":
        _ids = _sk.nlargest(AUTO_PICK, "A_skel_um2")["sigma_id"].astype(int).tolist()
    else:
        _ids = SAF.choose_sigmas(_sk, AUTO_PICK, seed=CELL_ID)    # noqa: F821
print("inspecting sigma %s with the %s rind rule" % (_ids, RIND_RULE))

_rows = []
for _sid in _ids:
    print("\n" + "=" * 72 + "\nsigma %d" % _sid)
    _roi = SAF.get_spine_roi(nodes, comp, _sid, CELL_ID,          # noqa: F821
                             roi_dir=None, pad_nm=PAD_NM,
                             reader_factory=_fac)
    _ax = _axes.get(_sid)
    _rec, _H, _det = SAF.measure_spine_area(
        _roi, _g, _opts, cell_id=CELL_ID, return_detail=True,     # noqa: F821
        shaft_axis=_ax, rind_tol_nm=RIND_TOL_NM)
    _sn, _bn, _ = SR.spine_subframe(nodes, comp, _sid)            # noqa: F821
    _r_sh = float(_sk.loc[_sk["sigma_id"] == _sid, "r_shaft_nm"].iloc[0])

    _verdict = "ok"
    try:
        _b, _prof, _meta = SB.measure_base(_roi, _sn, _bn, _opts, detail=_det,
                                           g_lookup=_g, r_shaft_nm=_r_sh)
        print("  base by %s" % _b["base_method"])
    except SB.ShaftTerminates as _e:
        _verdict = "shaft_terminates"
        print("  ** SHAFT ENDING, not a spine ** %s" % _e)
        print("     shaft reaches the cutout box: %s" % SB.shaft_reaches_box(_roi))
        _b, _prof, _meta = None, None, None
        _prof, _meta = SB.union_profile(_roi, _sn, _bn, _opts)   # still worth seeing
    except SB.BaseError as _e:
        _verdict = "error"
        print("  no measurable base (%s) -- falling back to the cylinder rule" % _e)
        _b, _prof, _meta = None, None, None

    _seg = _roi["meta"]["layers"]["seg"]
    if _meta is None:                      # build the union mesh anyway
        _U = SB.union_mask(_roi)
        _uv, _uf, _ = SB.mesh_from_mask(_U, _seg, _opts)
        _meta = {"union": {"verts_nm": _uv, "faces": _uf}}
    _un = _meta["union"]
    _reg = SB.classify_union_triangles(
        _un["verts_nm"], _un["faces"], _roi, _seg,
        base_point_nm=(_b or {}).get("base_point_nm"),
        base_tangent=(_b or {}).get("base_tangent"),
        shaft_axis=_ax, rind_tol_nm=RIND_TOL_NM, g_lookup=_g,
        rind_rule=RIND_RULE if (_b is not None or RIND_RULE == "cylinder")
        else "cylinder")
    _a = _reg["area_um2_by_region"]
    print("  union mesh %d faces | rind by %s | areas %s"
          % (len(_un["faces"]), _reg["rind_rule"],
             "calibrated" if _reg["calibrated"] else "RAW (uncalibrated)"))
    print("  areas um2: " + "  ".join("%s %.3f" % (k, v)
                                      for k, v in _a.items() if v > 1e-9))
    print("  analysis mesh: A_mesh %.3f (rind %.3f by the cylinder, %.1f%%), "
          "A_beyond %.3f" % (_rec["A_mesh_um2"], _rec["A_rind_um2"],
                             100 * _rec["frac_rind"],
                             (_b or {}).get("A_beyond_um2", float("nan"))))
    if _b is not None:
        print("  base at s = %.0f nm, r_shaft %.0f nm, difference %+.0f nm "
              "(expected >= 0: s_base = r_shaft / sin(alpha) for a spine "
              "leaving at angle alpha)"
              % (_b["s_base_nm"], _r_sh, _b["s_base_minus_r_shaft_nm"]))

    _show = ("spine", "rind", "bridge", "shaft", "detached") if SHOW_SHAFT \
        else ("spine", "rind", "bridge")
    _f3 = SAFF.union_mesh_3d(
        _un["verts_nm"], _un["faces"], _reg, base_rec=_b, spine_nodes=_sn,
        base_node=_bn, show=_show,
        title="cell %d sigma %d -- union mesh by region (%s rule)"
              % (CELL_ID, _sid, RIND_RULE))                        # noqa: F821
    _f3.write_html(os.path.join(OUT_UNION, "sigma%05d_union3d.html" % _sid),
                   include_plotlyjs="cdn")
    _f3.show()

    if _prof is not None:
        _f2 = SAFF.union_profile_figure(
            _prof, _b or {"s_base_nm": float("nan"), "r_shaft_nm": _r_sh,
                          "verdict": _verdict},
            rind_tol_nm=(8.0 if RIND_TOL_NM is None else RIND_TOL_NM),
            title="cell %d sigma %d -- union cross sections and thresholds"
                  % (CELL_ID, _sid))                               # noqa: F821
        _f2.write_html(os.path.join(OUT_UNION,
                                    "sigma%05d_profile.html" % _sid),
                       include_plotlyjs="cdn")
        _f2.show()

    _row = {"sigma_id": _sid, "r_shaft_nm": _r_sh, "base_verdict": _verdict,
            "base_method": (_b or {}).get("base_method"),
            "A_mesh_um2": _rec["A_mesh_um2"],
            "A_rind_cylinder_um2": _rec["A_rind_um2"],
            "frac_rind_cylinder": _rec["frac_rind"], "rind_rule": RIND_RULE,
            "union_faces": len(_un["faces"])}
    _row.update({"union_%s_um2" % k: v for k, v in _a.items()})
    if _b is not None:
        _row.update({k: _b[k] for k in ("s_base_nm", "s_base_minus_r_shaft_nm",
                                        "A_beyond_um2", "A_before_base_um2",
                                        "r_eq_base_nm", "area_drop_ratio")})
    _rows.append(_row)

if _rows:
    _df = pd.DataFrame(_rows)
    _p = os.path.join(OUT_UNION, "cell%d_union_regions.csv" % CELL_ID)  # noqa: F821
    _df.to_csv(_p, index=False, lineterminator="\n")
    print("\n" + "=" * 72)
    print(_df[[c for c in ("sigma_id", "union_spine_um2", "union_rind_um2",
                           "union_bridge_um2", "union_shaft_um2",
                           "A_beyond_um2", "A_rind_cylinder_um2",
                           "s_base_minus_r_shaft_nm") if c in _df.columns]]
          .to_string(index=False))
    print("\nwrote %s and %d html view(s) to %s" % (_p, 2 * len(_rows), OUT_UNION))
    del _df
for _k in ("_table", "_g", "_cal", "_sk", "_axes", "_fac", "_opts", "_ids",
           "_rows", "_sid", "_roi", "_ax", "_rec", "_H", "_det", "_sn", "_bn",
           "_r_sh", "_b", "_prof", "_meta", "_seg", "_un", "_reg", "_a",
           "_show", "_f3", "_f2", "_row", "_p", "_d", "_csv", "_U", "_uv",
           "_uf", "_e", "_m", "_", "_verdict"):
    globals().pop(_k, None)
del _k
