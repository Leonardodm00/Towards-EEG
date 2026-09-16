# %% CELL SAF -- calibrated spine area -> phi^mesh -> F, per cell  [NETWORK]
# Needs CELL 1 only (P, CODE_DIR on sys.path). Re-runs CELL 4 + CELL 5 logic
# per cell, so nothing else has to be in memory.
#
# Put in CODE_DIR: h01_spine_area_F.py, h01_spine_area_F_figures.py,
# h01_spine_base.py, smoke_test_h01_spine_area_F.py, g_table_cyl_2deg.npz AND
# g_table_cyl_2deg.json (from build_g_table.py).
# Run the smoke test first:  !cd "$CODE_DIR" && python3 smoke_test_h01_spine_area_F.py
#
# Resumable: every CHECKPOINT_EVERY spines the ledger is rewritten atomically;
# re-running the cell skips spines already measured -- and does NOT redraw
# their figures (delete the pilot ledger to redraw). Pure ASCII, LF only.
import os, sys, json, time, importlib
import numpy as np, pandas as pd

CELL_IDS     = [1302789404]     # sequential; ALL_CELLS once the pilot is clean
SPINE_SUBSET = 30               # None = every spine; int = size-stratified pilot
G_TABLE      = os.path.join(CODE_DIR, "g_table_cyl_2deg.npz")
OUT_SAF      = os.path.join(P["OUT"], "spine_area_F")
ROI_CACHE    = os.path.join(P["OUT"], "spine_roi")    # CELL 12 ROIs, read-only
PAD_NM, MAX_ROI_BYTES, CHECKPOINT_EVERY = 500.0, 1 << 30, 25
CORRECT_SHAFT_STUBS = True      # must equal CELL 5
MAKE_FIGURES = SPINE_SUBSET is not None   # pilot QC; thousands of PNGs is not QC

# ---- minimum spine size: demotes WHOLE spines to shaft BEFORE phi is built,
# so F_skel and F_mesh keep one partition. Sigma ids are not renumbered.
# CAUTION: human L2/3 stubby spines are ~584 nm long (KB, J Neurophysiol
# 10.1152/jn.00622.2024) and at ~300 nm node spacing get 1-2 skeleton nodes
# too -- read the threshold table and the demoted gallery before trusting a
# value. protrusion_nm = farthest node's distance from the base node minus the
# base radius (how far the skeleton gets beyond the shaft surface); it is NaN,
# and the spine is kept, when the base is not a shaft node (e.g. soma root).
MIN_SPINE_METRIC = "protrusion_nm"   # protrusion_nm | L_skel_nm | tip_dist_nm | n_nodes
MIN_SPINE_VALUE  = 100.0             # nm (a node count for n_nodes); None = off
INSPECT_SIGMAS   = [3588]            # print these spines' metrics
N_DEMOTED_FIGS   = 12                # pilot: figures of what the filter removes

# ---- junction terms (v1.4). The two estimators disagree about where the
# spine ends: A_skel includes a base frustum at SHAFT radius (deflates kappa),
# A_mesh includes the Voronoi rind of dendrite surface (inflates kappa, and
# that surface is double-counted because the shaft frustum has it too).
# AXIAL_WINDOW_NM caps how far from the base a triangle may be and still count
# as rind; None = no cap (an infinite cylinder, which can clip a spine that
# runs back alongside its dendrite). Records written by an earlier version
# lack these columns and are re-measured automatically.
AXIAL_WINDOW_NM = None
KAPPA_MIN_PER_BIN = 25               # >=25/bin; 5 fits noise at pilot sizes
# ---- measured base (v1.5, h01_spine_base). The UNION mesh (spine + shaft
# context) is sliced from the base node outward; shaft stations are those
# whose section loop touches the cutout box, and the base is the first that
# does not. Gives s_base (vs the skeleton r_shaft: a direct test of the rind
# cylinder) and A_beyond (the rind removed by measurement, not by a model).
# One union mesh per spine, ~70k faces: a few seconds each. Pilot only.
MEASURE_BASE = SPINE_SUBSET is not None
BASE_FIGS    = MEASURE_BASE          # interactive union-profile html per spine
SHOW_INLINE  = True             # also display each figure in the notebook
RECON_DIR  = "/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons"
STAGE1_DIR = "/content/drive/MyDrive/Colab Notebooks/New algorithms/Stage 1"
os.makedirs(OUT_SAF, exist_ok=True)

if STAGE1_DIR not in sys.path:
    sys.path.append(STAGE1_DIR)                # APPEND: CODE_DIR stays first
import sma_run as sr, s0_ingest, shaft_continuation as shc
import spine_labeller as sl, spine_density as sd, spine_geometry as sg
import morphology_exporter as mx
import h01_spine_batch as B, h01_spine_area_F as SAF
import h01_spine_area_F_figures as SAFF, h01_spine_base as SB
for _m in (sr, s0_ingest, shc, sl, sd, sg, mx, B, SAF, SAFF, SB):
    importlib.reload(_m)

VOC = sr.label_vocabularies(sd, sg)
assert VOC["sourced_from_project"], "vocabulary fell back to sma_run literals"
THRESHOLD_NM = float(mx.SPINE_LENGTH_THRESHOLD_NM)

# ---- Gate 0: the table, and the operator it was measured through ---------
table, g_lookup, cal_rep = SAF.load_calibration(G_TABLE)
print(json.dumps(cal_rep, indent=2))
assert int(B.DEFAULTS["taubin_iterations"]) == 14, \
    "the cylinder table was built at 14 Taubin iterations; rebuild it if this moved"
reader_factory = SAF.memoized_reader_factory()


def prepare_cell(cid):
    """CELL 4 + CELL 5: same calls, same order, same partition."""
    csv, _ = s0_ingest.write_s0_table(
        os.path.join(RECON_DIR, "neuron_%d.csv" % cid), P["DATA"], cid)
    nodes = pd.read_csv(csv)
    labelled, _ = sr.label_spines_project(nodes, cid, sl, THRESHOLD_NM,
                                          output_dir=None)
    if CORRECT_SHAFT_STUBS:
        labelled, _ = shc.demote_shaft_continuations(
            nodes, labelled, spine_density=sd,
            continuation_threshold_nm=THRESHOLD_NM)
    nodes["spine_label"] = labelled["annotated_type"].astype(str).str.lower().to_numpy()
    comp = sr.spine_components(
        nodes, sr.spine_mask(nodes, spine_values=VOC["spine_labels"]))
    return nodes, labelled, comp


RESULTS_SAF = {}
for cid in CELL_IDS:
    t0 = time.time()
    print("\n" + "=" * 72 + "\ncell %d" % cid)
    nodes, labelled, comp = prepare_cell(cid)
    sk_all = SAF.skeleton_spine_table(sd, labelled, nodes, comp)  # offline

    # ---- minimum-size filter: report, inspect, demote --------------------
    cand = (2, 3, 4) if MIN_SPINE_METRIC == "n_nodes" else \
        (100.0, 200.0, 300.0, 400.0, 500.0, 600.0)
    rep = SAF.threshold_report(sk_all, MIN_SPINE_METRIC, cand)
    print("  what a minimum on %s would remove (%d spines undefined, kept):"
          % (MIN_SPINE_METRIC, rep.attrs["n_metric_nan"]))
    print(rep.to_string(index=False))
    ins = sk_all[sk_all["sigma_id"].isin(INSPECT_SIGMAS)]
    if len(ins):
        print(ins[["sigma_id", "n_nodes", "L_skel_nm", "tip_dist_nm", "r_base_nm",
                   "protrusion_nm", "A_skel_um2"]].to_string(index=False))
    short = SAF.short_spine_ids(sk_all, MIN_SPINE_METRIC, MIN_SPINE_VALUE)
    labelled_d, nodes_d, comp_d, dprov = SAF.demote_spines(
        labelled, nodes, comp, sk_all, short, sd.SHAFT_REGEX)
    sk = SAF.skeleton_spine_table(sd, labelled_d, nodes_d, comp_d)
    F_lit_skel_nomin = SAF.cell_F(sd.build_phi(labelled, nid=cid,
                                               input_units="nm"))["F"]
    print("  filter %s < %s: %d spines demoted to shaft (%.1f%% of skeleton "
          "spine area), %d remain" % (MIN_SPINE_METRIC, MIN_SPINE_VALUE,
                                      dprov["n_spines_demoted"],
                                      100 * dprov["A_skel_demoted_um2"]
                                      / max(sk_all["A_skel_um2"].sum(), 1e-30),
                                      len(sk)))

    if MAKE_FIGURES and short and N_DEMOTED_FIGS:
        # Evenly spread over the metric, so the gallery shows the whole range
        # of what is removed -- from obvious stubs up to the threshold.
        v = sk_all.set_index("sigma_id").loc[short, MIN_SPINE_METRIC].sort_values()
        pick = [int(v.index[i]) for i in np.unique(np.linspace(
            0, len(v) - 1, min(int(N_DEMOTED_FIGS), len(v))).round().astype(int))]
        print("  drawing %d demoted spines for review -> figures/cell%d/demoted"
              % (len(pick), cid))
        roi_orig = lambda sid, _n=nodes, _c=comp, _id=cid: SAF.get_spine_roi(
            _n, _c, sid, _id, roi_dir=ROI_CACHE, pad_nm=PAD_NM,
            reader_factory=reader_factory, max_bytes=MAX_ROI_BYTES)
        SAF.measure_all_spines(
            pick, roi_orig, g_lookup,
            os.path.join(OUT_SAF, "cell%d_demoted_review_ledger.npz" % cid),
            cell_id=cid, verbose=False,
            on_success=SAFF.make_figure_callback(
                nodes, comp, os.path.join(OUT_SAF, "figures", "cell%d" % cid,
                                          "demoted"), show_inline=SHOW_INLINE))

    sigmas = SAF.choose_sigmas(sk, SPINE_SUBSET, seed=cid)
    print("  %d spines on the cell, measuring %d" % (len(sk), len(sigmas)))

    roi_fn = lambda sid, _n=nodes_d, _c=comp_d, _id=cid: SAF.get_spine_roi(
        _n, _c, sid, _id, roi_dir=ROI_CACHE, pad_nm=PAD_NM,
        reader_factory=reader_factory, max_bytes=MAX_ROI_BYTES)
    # Per-spine figure: XY / XZ / YZ max projections, spine blue, shaft
    # amber, bridge red, same-id detached grey, excluded cut face purple.
    fig_cb = (SAFF.make_figure_callback(
        nodes_d, comp_d, os.path.join(OUT_SAF, "figures", "cell%d" % cid),
        show_inline=SHOW_INLINE) if MAKE_FIGURES else None)
    axes = SAF.shaft_axes_from_table(sk)
    base_cb = (SB.make_base_callback(
        nodes_d, comp_d, sk, dict(B.DEFAULTS), g_lookup,
        fig_dir=os.path.join(OUT_SAF, "figures", "cell%d" % cid, "union_profile")
        if BASE_FIGS else None, show_inline=SHOW_INLINE and BASE_FIGS)
        if MEASURE_BASE else None)
    # base_method and base_verdict are v1.6 fields: without them in this list,
    # records cached before the shaft-ending fix keep their old s_base_nm and
    # the flag never appears for them.
    need = ("A_rind_um2", "rind_tol_nm") + \
        (("s_base_nm", "base_method", "base_verdict") if MEASURE_BASE else ())
    recs, H = SAF.measure_all_spines(
        sigmas, roi_fn, g_lookup,
        os.path.join(OUT_SAF, "cell%d_ledger.npz" % cid),
        cell_id=cid, checkpoint_every=CHECKPOINT_EVERY,
        on_success=SB.chain_callbacks(fig_cb, base_cb),
        shaft_axes=axes, axial_window_nm=AXIAL_WINDOW_NM, require_keys=need)

    # Raises SpineAreaError if the per-spine bookkeeping does not reproduce
    # Stage 1's phi row by row. A failed gate stops the cell.
    out = SAF.assemble_cell(sd, labelled_d, nodes_d, comp_d, recs, cid, sk=sk,
                            min_per_bin=KAPPA_MIN_PER_BIN)
    out["summary"].update({"min_spine_metric": MIN_SPINE_METRIC,
                           "min_spine_value": MIN_SPINE_VALUE,
                           "n_spines_demoted_min": dprov["n_spines_demoted"],
                           "A_skel_demoted_um2": dprov["A_skel_demoted_um2"],
                           "F_lit_skel_no_min_filter": F_lit_skel_nomin})
    paths = SAF.write_cell_outputs(OUT_SAF, cid, out)
    RESULTS_SAF[cid] = out
    s = out["summary"]
    print("  gate max|diff| %.2e um2 -> PASS"
          % s["attribution_gate"]["max_abs_diff_um2"])
    print("  measured %d/%d (%.1f%% of skeleton spine area), %d failed, %d clipped"
          % (s["n_measured"], s["n_spines"], 100 * s["coverage_area"],
             s["n_failed"], s["n_clipped"]))
    if s["coverage_area"] < 0.95:
        print("  NOTE: F_mesh below fills %.0f%% of spine area from kappa_hat --"
              " an extrapolation, not a measurement" % (100 * (1 - s["coverage_area"])))
    print("  kappa pooled %.3f | median cut %.1f%% | median bridge %.1f%% | "
          "max box %.2f%%" % (s["kappa_pooled"], 100 * s["median_frac_cut"],
                              100 * s["median_frac_bridge"], 100 * s["max_frac_box"]))
    print("  junction terms (the two estimators' disagreement at the neck):")
    print("    Voronoi rind   %.1f%% of A_mesh (median, max %.1f%%), %d spine(s) "
          "had no shaft axis" % (100 * s["frac_rind_median"],
                                 100 * s["frac_rind_max"], s["n_no_shaft_axis"]))
    print("    base frustum   %.1f%% of A_skel (pooled) -- shaft-radius surface "
          "credited to the spine" % (100 * s["base_frac_of_skel_pooled"]))
    print("    kappa   raw %.3f | rind removed %.3f | base removed %.3f | "
          "both %.3f" % (s["kappa_pooled"], s["kappa_pooled_norind"],
                         s["kappa_pooled_nobase"], s["kappa_pooled_both"]))
    if s["n_measured_base"]:
        print("    measured base on %d spine(s): s_base - r_shaft median %+.0f nm, "
              "|.| p90 %.0f nm (phantom: +5 nm) | rind-cylinder / plane-cut area "
              "median %.3f (phantom: 1.001) | kappa beyond %.3f"
              % (s["n_measured_base"], s["s_base_minus_r_shaft_median_nm"],
                 s["s_base_minus_r_shaft_p90_abs_nm"],
                 s["norind_vs_beyond_median_ratio"], s["kappa_pooled_beyond"]))
        if s["n_shaft_terminates"]:
            print("    %d component(s) are SHAFT ENDINGS by the mesh: their cross-"
                  "sections never separate from the dendrite (no box contact, "
                  "no >= 3x area drop). Not counted as A_beyond; kappa-filled and "
                  "flagged base_verdict='shaft_terminates'. %d base(s) came from "
                  "the area-drop fallback (shallow-angle spines)."
                  % (s["n_shaft_terminates"], s["n_base_by_area_drop"]))
        nb = out["spines"]["base_error"].notna().sum() \
            if "base_error" in out["spines"] else 0
        if nb:
            print("    %d spine(s) had no measurable base:" % nb)
            print(out["spines"].loc[out["spines"]["base_error"].notna(),
                                    ["sigma_id", "base_error"]].head(8).to_string(index=False))
    for lab in ("skel", "mesh_uncalibrated", "mesh_skelfill", "mesh",
                "mesh_norind", "mesh_beyond"):
        print("  F_lit %-18s %.4f   F_whole %.4f"
              % (lab, s["F_lit_" + lab], s["F_whole_" + lab]))
    print("  F_lit skel WITHOUT the min-size filter      %.4f   (the filter's own "
          "effect on F)" % F_lit_skel_nomin)
    if "F_lit_skel_stage1_fn" in s:
        print("  F_lit skel via sd.cell_f_beyond_cutoff: %.4f (must equal above)"
              % s["F_lit_skel_stage1_fn"])
    print(out["kappa_table"].to_string(index=False))
    if "cut_to_seam_median_nm" in out["spines"]:
        seam = out["spines"]["cut_to_seam_median_nm"].dropna()
        print("  cut crosses vs seam (perpendicular): median %.2f nm, max %.2f nm "
              "over %d spines -- phantoms give < 1 nm; >> 4 nm means the cut "
              "classifier is off for those spines" % (seam.median(), seam.max(),
                                                      len(seam)))
    if "figure_error" in out["spines"]:
        n_fe = int(out["spines"]["figure_error"].notna().sum())
        if n_fe:
            print("  %d figure(s) failed to draw (measurements unaffected)" % n_fe)
    fails = out["spines"][out["spines"]["ok"].astype("boolean").fillna(True) == False]
    if len(fails):
        print("  failures by stage:", fails["stage"].value_counts().to_dict())
    print("  wrote %s  (%.1f min)" % (json.dumps(paths, indent=1),
                                      (time.time() - t0) / 60))
