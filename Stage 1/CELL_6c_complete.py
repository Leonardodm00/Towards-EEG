# =========================================================================== #
# CELL 6c -- REPLACE the existing CELL 6c in full. Paste after CELL 6.        #
# =========================================================================== #
# Reads `frames` from CELL 6, or rebuilds it from the skeletons if absent.
# Runs in two passes because the truncation boundary estimate is POOLED: it
# needs every cell's extent before it can judge any single cell's tips.
#
# CHANGED in this version -- the rebuild block only. Everything else is
# unchanged from the previous CELL 6c.
#
#   The rebuild used to call spine_labeller directly on the RAW skeleton CSV.
#   The exporter does not: export_neuron runs three steps before the labeller
#   (extract_synapse_frame, classify_frame, resolve_mislabelled_nodes), and the
#   third REWRITES annotated_type for every node whose annotation was a known
#   mislabel. The labeller therefore saw a different frame in the two paths.
#
#   Measured on neuron 15543554616:
#       export path (CELL 6, CELL 6d)   14116 spine nodes, 2023 spines
#       old rebuild path (CELL 6c)      13802 spine nodes, 1974 spines
#
#   so 314 nodes and 49 spines -- about 2 percent -- differed, and every
#   statistic below (L_neck, d_neck, R_neck, the cap bracket, the tip audit)
#   was computed on a partition that was NOT the one exported and NOT the one
#   F was built from.
#
#   The fix runs the exporter's own steps 1-3, then the labeller, then step 5's
#   mandatory re-classify. Nothing is reimplemented: the calls are
#   node_classify's, with the exporter's own arguments.

import json

import numpy as np
import pandas as pd

import spine_cap as spc
import spine_cap_plots as scp
import spine_density as sd
import spine_geometry as sg
import truncation_flag as tf

try:
    import phi_pipeline_colab as ppc
    _HAVE_RADIUS_REPORT = True
except ImportError:
    _HAVE_RADIUS_REPORT = False
    print("WARNING: phi_pipeline_colab not uploaded -- the authoritative "
          "radius gate is unavailable, and every resistance printed below "
          "must be treated as unverified. Upload it and re-run this cell.")

# CELL 6c gets its labelled frames from CELL 6, which parks them in `frames`.
# If that is not available -- kernel restarted, CELL 6 modified, or CELL 6 run
# without return_frames=True -- this cell REBUILDS them from the skeletons
# rather than failing.
#
# Note `records` cannot substitute for `frames`: CELL 6 does
# res.pop("frames"), which REMOVES the labelled frame from the record, so the
# labelled geometry survives only in `frames`.

import os
import sys

_ns = globals()          # not dir(): explicit, and correct inside a
                         # comprehension on every Python version


def _s1_have(name):
    return name in _ns and _ns[name] is not None


if not _s1_have("OUTPUT_DIR"):
    raise RuntimeError(
        "CELL 6c needs OUTPUT_DIR, which CELL 4 defines. Run CELL 4 first.")

_frames = _ns.get("frames") if _s1_have("frames") else None

if not _frames:
    # -- diagnose before deciding, so a surprising namespace is visible ------
    _relevant = sorted(k for k in _ns
                       if not k.startswith("_")
                       and ("frame" in k.lower() or k in
                            ("records", "NEURON_IDS", "SKELETONS_DIR",
                             "gated_out", "failures")))
    print("`frames` is not available. Related names in the namespace: %s"
          % (_relevant if _relevant else "none"))

    _can_rebuild = _s1_have("NEURON_IDS") and _s1_have("SKELETONS_DIR")
    if not _can_rebuild:
        raise RuntimeError(
            "`frames` is missing and it cannot be rebuilt either, because "
            "NEURON_IDS and/or SKELETONS_DIR are not defined. Run CELL 4, "
            "then CELL 6 -- or set those two and re-run this cell to rebuild "
            "from the skeletons directly.")

    import shutil as _shutil
    import tempfile as _tempfile

    import pandas as _pd
    import morphology_exporter as _mx
    import node_classify as _nc
    import spine_labeller as _sl

    _thr = _mx.SPINE_LENGTH_THRESHOLD_NM
    print("rebuilding labelled frames from %s (threshold %.0f nm, through the "
          "exporter's steps 1-3 + 5, so the partition matches CELL 6) for "
          "%d cell(s)..." % (SKELETONS_DIR, _thr, len(NEURON_IDS)))  # noqa: F821

    def _label_like_exporter(_df_raw, _nid):
        """export_neuron steps 1-5, minus the soma enforcement (step 6, which
        does not touch spine nodes) and the export itself."""
        _d = _df_raw.copy()
        _nc.extract_synapse_frame(_d, nid=_nid, input_units="nm")    # step 1
        _d = _nc.classify_frame(_d)                                  # step 2
        _d, _ = _nc.resolve_mislabelled_nodes(                       # step 3
            _d, policy="knn_relabel", k=5, max_distance_nm=None,
            rewrite_annotation=True)
        _tmp = _tempfile.mkdtemp()                                   # step 4
        try:
            _d.to_csv(os.path.join(_tmp, "neuron_%s.csv" % _nid), index=False)
            _o = _sl.label_dendritic_spines_robust(
                [_nid], input_dir=_tmp, output_dir=None,
                spine_length_threshold_nm=_thr)
        finally:
            _shutil.rmtree(_tmp, ignore_errors=True)
        _d = _o[_nid] if isinstance(_o, dict) else _o
        return _nc.classify_frame(_d)                                # step 5

    _frames = {}
    for _nid in NEURON_IDS:                                        # noqa: F821
        _path = "%s/neuron_%s.csv" % (SKELETONS_DIR, _nid)         # noqa: F821
        if not os.path.isfile(_path):
            print("   skipping %s: no skeleton at %s" % (_nid, _path))
            continue
        _frames[_nid] = {"raw": _label_like_exporter(_pd.read_csv(_path), _nid)}
    if not _frames:
        raise RuntimeError(
            "rebuild produced nothing -- no skeleton CSV was found in %s for "
            "any id in NEURON_IDS." % SKELETONS_DIR)               # noqa: F821
    print("rebuilt %d labelled frame(s)" % len(_frames))

    # Gate: the rebuilt partition must equal the exported one. `records` keeps
    # spine_report per cell (_compact strips frames/files/qc, not that), so the
    # comparison is free when CELL 6 ran in this session. When it did not,
    # there is nothing to compare against and the check is skipped.
    _recs = {int(_r["nid"]): _r for _r in _ns.get("records", []) or []
             if isinstance(_r, dict) and _r.get("nid") is not None}
    for _nid, _f in _frames.items():
        _rep = (_recs.get(int(_nid)) or {}).get("spine_report")
        if not _rep:
            continue
        _n = int((_f["raw"]["compartment_class"].astype(str)
                  == _nc.CLS_SPINE).sum())
        if _n != int(_rep["n_spine_nodes"]):
            print("   WARNING %s: rebuilt %d spine node(s), CELL 6 exported %d."
                  " The partitions disagree -- the statistics below do not "
                  "describe the bank." % (_nid, _n, _rep["n_spine_nodes"]))
        else:
            print("   %s: %d spine node(s), matches CELL 6" % (_nid, _n))

    print("NOTE these are labelled but NOT gated: CELL 6's propagation gate "
          "has not been applied, so cells that would have been gated out are "
          "included here. For a bank run, prefer re-running CELL 6.")

frames = _frames          # from CELL 6, or rebuilt above

S1_DIR = OUTPUT_DIR + "/s1"                             # noqa: F821
os.makedirs(S1_DIR, exist_ok=True)

RHO_A_SWEEP = (100.0, 200.0, 300.0, 400.0)

# ---- pass 1: per-cell spine geometry, and the pooled boundary -------------
spine_frames, s1_records = {}, []
tip_audits, spine_profiles, f_bracket_by_mode = {}, {}, {}

for nid, fr in frames.items():                                  # noqa: F821
    df_lab = fr["raw"]                       # pre-prune, raw nm, head/neck
    spine_df = sg.build_spine_geometry(df_lab, nid=nid, input_units="nm",
                                       cap_tips=CAP_TIPS,   # noqa: F821
                                       cap_h_um=CAP_H_UM)   # noqa: F821
    spine_frames[nid] = spine_df

    # Gate 1 + the meridian profiles for the gallery. Both are recorded
    # whether or not the cap is enabled: the audit is what tells you whether
    # enabling it would be legitimate, so it has to come first.
    _node, _children, _root = sd._prepare_nodes(
        df_lab, sd.SHAFT_REGEX, sd.SPINE_LABELS, sd.DEFAULT_RADIUS_NM, "nm")
    tip_audits[nid] = spc.audit_tips(_node, _children, root=_root)
    spine_profiles[nid] = spc.spine_profiles(
        _node, _children, h_um=(CAP_H_UM if CAP_TIPS else None),  # noqa: F821
        max_n=20, sort_by="area")

    # the F bracket, per cell. Three build_phi calls; they differ only in the
    # tip closure, so any other difference between them is a bug.
    f_bracket_by_mode.setdefault("no cap", {})
    f_bracket_by_mode.setdefault("flat disc", {})
    f_bracket_by_mode.setdefault("cap h=%.0fnm" % (CAP_H_UM * 1000), {})  # noqa: F821
    for _lab, _kw in (("no cap", dict(cap_tips=False)),
                      ("flat disc", dict(cap_tips=True, cap_h_um=0.0)),
                      ("cap h=%.0fnm" % (CAP_H_UM * 1000),               # noqa: F821
                       dict(cap_tips=True, cap_h_um=CAP_H_UM))):         # noqa: F821
        f_bracket_by_mode[_lab][nid] = float(sd.cell_f_implied_from_phi(
            sd.build_phi(df_lab, nid=nid, input_units="nm", **_kw)))

    _ta = spc.summarise_audit(tip_audits[nid], h_um=CAP_H_UM)  # noqa: F821
    print("  tips: %d true leaf (capped), %d label-boundary (NOT capped), "
          "mean r_t %.3f um" % (_ta["spine_n_capped"],
                                _ta["spine_n_label_boundary_end"],
                                _ta["spine_mean_r_tip_um"]))
    print("       cap would add %.2f um2 spine / %.2f um2 shaft"
          % (_ta["spine_total_cap_um2"], _ta["other_total_cap_um2"]))

    rq = ppc.radius_report(df_lab) if _HAVE_RADIUS_REPORT else None
    summ = sg.cell_spine_summary(spine_df, rho_a_ohm_cm=RHO_A_SWEEP,
                                 radius_report=rq)
    summ["nid"] = nid
    s1_records.append(summ)

    path = "%s/neuron_%s_spine_geometry.csv" % (S1_DIR, nid)
    spine_df.to_csv(path, index=False)

    trust = summ.get("resistance_trustworthy")
    flag = "" if trust is not False else "   [!] RADII SUSPECT"
    print("neuron %s: %d spines, %.0f%% with a labelled neck%s"
          % (nid, summ["n_spines"], 100.0 * summ.get("frac_with_neck", 0.0),
             flag))
    if summ.get("n_with_neck"):
        print("    L_neck median %.3f um, d_neck_equiv median %.3f um"
              % (summ["L_neck_median_um"], summ["d_neck_equiv_median_um"]))
        for rho in RHO_A_SWEEP:
            k = "R_neck_MOhm_rho%d" % int(rho)
            print("    R_neck at rho=%3.0f ohm cm:  p05 %7.1f  median %7.1f  "
                  "p95 %7.1f MOhm"
                  % (rho, summ[k + "_p05"], summ[k + "_median"],
                     summ[k + "_p95"]))

# ---- pass 2: pooled bounds, then per-cell truncation ---------------------
_frames_for_bounds = [(nid, fr["raw"]) for nid, fr in frames.items()]  # noqa: F821
bounds = tf.pool_axis_bounds(_frames_for_bounds, input_units="nm")

print("\npooled boundary over %d cell(s), %d nodes"
      % (bounds["n_cells"], bounds["n_nodes"]))
print("  x %s   y %s   z %s   (um)"
      % tuple(tuple(round(v, 1) for v in bounds[a]) for a in "xyz"))
print("  z_range_fraction_of_slab = %.3f" % bounds["z_range_fraction_of_slab"])
if bounds["z_range_fraction_of_slab"] < 0.5:
    print("  [!] the pooled cells span less than half the ~170 um H01 depth, "
          "so the z bound is an underestimate of the true slab face and the "
          "z-boundary signal will UNDER-report truncation. Pool more cells "
          "before trusting it; the taper signal is unaffected.")

trunc_frames = {}
for nid, fr in frames.items():                                  # noqa: F821
    trunc = tf.build_truncation_table(fr["raw"], bounds, nid=nid,
                                      input_units="nm")
    trunc_frames[nid] = trunc
    tsumm = tf.cell_truncation_summary(trunc)
    trunc.to_csv("%s/neuron_%s_truncation.csv" % (S1_DIR, nid),
                 index=False)
    for rec in s1_records:
        if rec["nid"] == nid:
            rec["n_tips"] = tsumm["n_tips"]
            rec["n_truncated"] = tsumm["n_truncated"]
            rec["frac_truncated"] = tsumm["frac_truncated"]
            rec["truncation_basis_counts"] = json.dumps(
                tsumm["basis_counts"], sort_keys=True)
    print("neuron %s: %d tips, %d truncated (%.0f%%), basis %s"
          % (nid, tsumm["n_tips"], tsumm["n_truncated"],
             100.0 * tsumm["frac_truncated"], tsumm["basis_counts"]))

s1_df = pd.DataFrame(s1_records)
s1_df.to_csv("%s/s1_summary.csv" % S1_DIR, index=False)
print("\nwrote %s/s1_summary.csv  (%d cells)"
      % (S1_DIR, len(s1_df)))
