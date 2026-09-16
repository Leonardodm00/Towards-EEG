# =========================================================================== #
# CELL 6e -- NEW. Paste after CELL 6d. Answers O-3: are the 312 real?        #
# =========================================================================== #
# Three views of the roots shaft_continuation calls shaft-like, on the
# labelled frame CELL 6c holds in `frames`. Nothing is written to the bank.
#
#   1. (rho, cos) scatter with the thresholds -- how many sit on the line
#   2. dF_lit when each rho band is demoted ALONE -- where the -0.143 lives
#   3. interactive 3D gallery of a stratified sample -- judge them by eye
#
# Needs continuation_inspect.py and check_pruned_hoc.py in CODE_DIR, and
# shaft_continuation.py reachable (CELL 6d's SHC_DIR logic is reused).

import importlib
import os
import sys

import numpy as np
import pandas as pd

import check_pruned_hoc as CK
import continuation_inspect as CI
import spine_density as sd
for _m in (CK, CI):
    importlib.reload(_m)

SHC_DIR = ("/content/drive/MyDrive/Colab Notebooks/New algorithms/"
           "Spine Mesh Analysis")
try:
    import shaft_continuation as shc
except ImportError:
    if SHC_DIR not in sys.path:
        sys.path.append(SHC_DIR)
    import shaft_continuation as shc

INSPECT_NID = None            # None = first cell in `frames`
GALLERY_K = 12                # interactive 3D views, stratified over rho
GALLERY_SEED = 0
ROOTS_TO_SHOW = []            # add specific root ids here after a first look
F_CUTOFF_UM = 60.0
INSPECT_DIR = os.path.join(FIGURE_DIR, "continuations")          # noqa: F821
os.makedirs(INSPECT_DIR, exist_ok=True)

_nid = INSPECT_NID or next(iter(frames))                          # noqa: F821
_lab = frames[_nid]["raw"]                                        # noqa: F821
_table, _rep = shc.score_spine_roots(_lab, spine_density=sd)
_rho_min, _cos_min = _rep["rho_shaft_min"], _rep["cos_shaft_min"]
print("neuron %s: %d spine roots, %d shaft-like by %s (rho >= %.2f, cos >= %.2f)"
      % (_nid, _rep["n_spine_roots"], _rep["n_shaft_like"], _rep["method"],
         _rho_min, _cos_min))

# ---- 1. boundary population ----------------------------------------------
_bp = CI.boundary_population(_table, _rho_min, _cos_min)
print("\n1. boundary population (margin %.2f on rho, %.2f on cos):"
      % (_bp["rho_margin"], _bp["cos_margin"]))
print("   %d of %d shaft-like roots would flip to SPINE if rho_min rose by "
      "the margin; %d if cos_min did; %d for either"
      % (_bp["n_within_rho_margin"], _bp["n_shaft_like"],
         _bp["n_within_cos_margin"], _bp["n_within_either"]))
print("   %d spines are near-misses on rho (cos passes), %d on cos (rho passes) "
      "-- would flip to SHAFT if the threshold dropped"
      % (_bp["n_near_miss_rho"], _bp["n_near_miss_cos"]))
_sl = _table[_table["is_shaft"]]
print("   shaft-like own_len_nm: p05 %.0f  median %.0f  p95 %.0f | %d shorter "
      "than 100 nm (sub-voxel stubs)"
      % (_sl["own_len_nm"].quantile(.05), _sl["own_len_nm"].median(),
         _sl["own_len_nm"].quantile(.95), int((_sl["own_len_nm"] < 100).sum())))
print("   by category:", _sl["category"].value_counts().to_dict())
CI.scatter_figure(_table, _rho_min, _cos_min,
                  title="neuron %s: spine roots, calibre vs collinearity" % _nid
                  ).show()

# ---- 2. dF by band ---------------------------------------------------------
print("\n2. F_lit shift when each group is demoted ALONE (from the exported "
      "frame, cutoff %.0f um):" % F_CUTOFF_UM)
_band = CI.dF_by_band(_lab, _table, sd, nid=_nid, cutoff_um=F_CUTOFF_UM)
print(_band[["group", "n_roots", "n_nodes", "A_moved_um2", "F_lit", "dF_lit"]]
      .to_string(index=False))
print("   F_lit as exported %.4f | sum of rho-band shifts %+.4f | all at once "
      "%+.4f (bands are not exactly additive: F is a ratio)"
      % (_band.attrs["F_lit_as_exported"], _band.attrs["sum_of_band_dF"],
         float(_band[_band["group"].str.startswith("ALL")]["dF_lit"].iloc[0])))
_band.to_csv(os.path.join(INSPECT_DIR, "neuron_%s_dF_by_band.csv" % _nid),
             index=False, lineterminator="\n")
_table.to_csv(os.path.join(INSPECT_DIR, "neuron_%s_root_scores.csv" % _nid),
              index=False, lineterminator="\n")

# ---- 3. the taper test, the third vote -------------------------------------
_tap = CI.taper_table(_lab, _table["root"])
_vote = CI.three_vote(_table, _tap)
_vs = CI.vote_summary(_vote)
print("\n3. taper test (a branch thins; a spine dips then rises into its head):")
print("   of %d shaft-like roots -> %d demoted on three votes, %d held back by "
      "the taper (a distal radius maximum), %d too short to judge (< %.0f nm)"
      % (_vs["n_shaft_like_rho_cos"], _vs["n_demote_three_vote"],
         _vs["n_rescued_by_taper"], _vs["n_undecidable"], CI.MIN_LEN_NM))
print("   %d of the shaft-like roots carry a 'head' label from the labeller; "
      "the taper disagrees with that label on %d of them"
      % (_vs["n_head_label_among_shaft_like"],
         _vs["n_taper_vs_headlabel_disagree"]))
print("  ", _vs["decision_counts"])
_dem = _vote.loc[_vote["demote"], "root"].astype(int).tolist()
if _dem:
    _corr, _n = CK.demote_roots(_lab, _dem, sd)
    _p0 = sd.build_phi(_lab, nid=_nid, input_units="nm")
    _p1 = sd.build_phi(_corr, nid=_nid, input_units="nm")
    _F0 = float(sd.cell_f_beyond_cutoff(_p0, cutoff_um=F_CUTOFF_UM,
                                        by="d_from_um")["F"])
    _F1 = float(sd.cell_f_beyond_cutoff(_p1, cutoff_um=F_CUTOFF_UM,
                                        by="d_from_um")["F"])
    print("   THREE-VOTE dF_lit %+.4f (%.4f -> %.4f), %.1f um2 moved, vs the "
          "rho/cos-only shift reported in CELL 6d"
          % (_F1 - _F0, _F0, _F1,
             float(_p0["spine_area_um2"].sum() - _p1["spine_area_um2"].sum())))
_vote.to_csv(os.path.join(INSPECT_DIR, "neuron_%s_three_vote.csv" % _nid),
             index=False, lineterminator="\n")
CI.taper_figure(_lab, _table.loc[_table["is_shaft"], "root"], _tap,
                title="neuron %s: radius along shaft-like components" % _nid).show()

# ---- 4. dF by cos band, the soft axis --------------------------------------
print("\n4. F_lit shift by COLLINEARITY band (the axis with ~106 roots within "
      "0.05 of its threshold):")
_cb = CI.dF_by_cos_band(_lab, _table, sd, nid=_nid, cutoff_um=F_CUTOFF_UM)
print(_cb[["group", "n_roots", "n_nodes", "A_moved_um2", "F_lit", "dF_lit"]]
      .to_string(index=False))
_cb.to_csv(os.path.join(INSPECT_DIR, "neuron_%s_dF_by_cos_band.csv" % _nid),
           index=False, lineterminator="\n")

# ---- 5. gallery ------------------------------------------------------------
_roots = list(ROOTS_TO_SHOW) or CI.pick_gallery(_table, k=GALLERY_K,
                                                seed=GALLERY_SEED)
print("\n5. gallery of %d shaft-like root(s): red = the component, black = "
      "branch point, orange = parent path, blue = siblings; marker size ~ "
      "radius. Ask of each: is the red part a continuation of the orange "
      "path (calibre and direction), or a side protrusion?" % len(_roots))
_ts = _table.set_index("root")
for _r in _roots:
    _sub, _bpid = CI.local_skeleton(_lab, _r)
    _row = _ts.loc[_r]
    _fig = CI.local_figure(
        _sub, _r, _bpid,
        title="root %d @ bp %d | rho %.2f cos %.2f | %.0f nm, %d nodes | %s"
        % (_r, _bpid, _row["rho"], _row["cos"], _row["own_len_nm"],
           int((_sub["role"] == "component").sum()), _row["category"]))
    _fig.write_html(os.path.join(INSPECT_DIR, "root_%d.html" % _r),
                    include_plotlyjs="cdn")
    _fig.show()
print("\nwrote scores, dF table and %d html view(s) to %s" % (len(_roots), INSPECT_DIR))
for _k in ("_lab", "_table", "_rep", "_sl", "_bp", "_band", "_roots", "_ts",
           "_tap", "_vote", "_vs", "_dem", "_corr", "_n", "_p0", "_p1",
           "_F0", "_F1", "_cb",
           "_m", "_nid", "_rho_min", "_cos_min", "_r", "_sub", "_bpid", "_row",
           "_fig"):
    globals().pop(_k, None)
del _k
