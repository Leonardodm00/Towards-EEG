"""Colab cells for the S1 S1 modules, against the existing pipeline.

HOW TO USE THIS FILE
--------------------
This is not a script to run. It is the set of cells to paste into the notebook
that currently runs colab_run_s1_full.py, in the order given. Two of them are
small EDITS to existing cells (shown as before/after); two are NEW cells to
paste in whole.

Nothing here changes the export, the alignment, the gate, or any existing
output. Every new module is a read of frames the pipeline already builds. If
you delete all four cells the pipeline behaves exactly as it does today.

WHY SO LITTLE HAS TO CHANGE
---------------------------
CELL 6 already keeps what the new modules need:

    df_lab = res.pop("frames")["labelled"]        # colab_run_s1_full.py:345
    frames[nid] = {"raw": df_lab, ...}            # colab_run_s1_full.py:352

morphology_exporter.py sets frames["labelled"] at line 627 and calls
prune_spines at line 630, so that frame is PRE-PRUNE -- spines still attached,
head/neck labels intact. It is the same frame spine_density.build_phi consumes.
And because CELL 6 parks it in `frames`, the whole spine-geometry and
truncation pass can happen AFTER the loop, in one new cell, with zero edits to
the loop body.

Only syn_uid needs an in-loop edit, because identifiers must be assigned to the
RAW synapse export before the nearest-node mapping consumes it.

ORDER
-----
    CELL 1   EDIT   add four filenames to the upload list
    CELL 3b  NEW    paste immediately after CELL 3
    CELL 6   EDIT   three lines inserted in the loop body
    CELL 6c  NEW    paste immediately after CELL 6 (before or after CELL 6b,
                    either works -- it touches nothing CELL 6b touches)
    CELL 8b  NEW    paste after CELL 8, before CELL 9 -- the figures
    CELL 9   REPLACE  full rewrite of the summary/manifest cell

ASCII only, LF only.
"""

# =========================================================================== #
# CELL 1 -- EDIT                                                              #
# =========================================================================== #
# The four new files must be uploaded alongside the existing modules. Add them
# to REQUIRED_MODULES and REQUIRED_TESTS so the upload prompt lists them and a
# missing one is obvious before anything runs.
#
# BEFORE:
#     REQUIRED_MODULES = [
#         "spine_density.py", "node_classify.py", "soma_enforce.py",
#         "morphology_exporter.py", "spine_labeller.py",
#         "alignment.py", "alignment_plots.py", "hoc_qc.py",
#         "synapse_redirect_audit.py",
#     ]
#
# AFTER:
#     REQUIRED_MODULES = [
#         "spine_density.py", "node_classify.py", "soma_enforce.py",
#         "morphology_exporter.py", "spine_labeller.py",
#         "alignment.py", "alignment_plots.py", "hoc_qc.py",
#         "synapse_redirect_audit.py",
#         # --- S1 ---
#         "spine_geometry.py", "truncation_flag.py", "syn_uid.py",
#         "s1_plots.py", "phi_pipeline_colab.py",
#     ]
#
# and append to REQUIRED_TESTS:
#         "test_spine_geometry.py", "test_truncation_flag.py",
#         "test_syn_uid.py", "test_s1_plots.py", "run_all_tests.py",
#
# phi_pipeline_colab.py is on that list because CELL 6c calls its
# radius_report() -- the authoritative radius gate. It is an existing project
# module, not a new one; it simply was not needed by this driver before.


# =========================================================================== #
# CELL 3b -- NEW. Paste immediately after CELL 3.                             #
# =========================================================================== #
# CELL 3's pass/fail matcher looks for the string "N / N" in a suite's stdout.
# The S1 suites do not print that; they print "  PASS  <name>" per check and
# "ALL CHECKS PASSED" at the end. Adding them to CELL 3's SUITES list would
# therefore mark them FAILED even when green. They get their own cell, which
# calls their own runner -- which additionally does the HPC transfer-safety
# checks (non-ASCII AND carriage-return bytes: different bugs, different byte
# ranges, an ASCII scan will never find CRLF) before running anything.
#
# The runner is located EXPLICITLY rather than by relative name. A bare
# "run_all_tests.py" resolves against the notebook's cwd (/content), not
# against CODE_DIR on Drive, and the interpreter then exits 2 with its message
# on stderr -- which, if only stdout is printed, looks like an empty run.

import os
import subprocess
import sys

# CODE_DIR is defined in CELL 1/2. Fall back to searching sys.path so this
# cell also works in the upload-to-/content layout.
_runner = None
for _d in ([CODE_DIR] if "CODE_DIR" in dir() else []) + sys.path + [os.getcwd()]:
    if _d and os.path.exists(os.path.join(_d, "run_all_tests.py")):
        _runner = os.path.join(_d, "run_all_tests.py")
        break
if _runner is None:
    raise FileNotFoundError(
        "run_all_tests.py not found in CODE_DIR, on sys.path, or in cwd. "
        "Put the S1 files next to the other S1 modules.")
print("runner: %s" % _runner)

# A fresh interpreter does not inherit sys.path entries added at runtime, so
# forward the live path or the runner cannot import modules kept on Drive.
_env = dict(os.environ)
_env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p) + (
    os.pathsep + _env["PYTHONPATH"] if _env.get("PYTHONPATH") else "")

_r = subprocess.run([sys.executable, _runner], capture_output=True,
                    text=True, env=_env)
print(_r.stdout)
if _r.stderr.strip():
    print("--- stderr ---")
    print(_r.stderr[-3000:])

if _r.returncode == 3:
    raise RuntimeError(
        "S1 dependencies not importable -- see the list above. They are "
        "existing project modules; check CODE_DIR is on sys.path.")
if _r.returncode == 4:
    raise RuntimeError(
        "one or more suite FILES are missing from CODE_DIR -- see the "
        "directory listing and the closest-name suggestions above. This is a "
        "setup problem, not a test failure; nothing is wrong with the code.")
if _r.returncode != 0:
    raise RuntimeError(
        "S1 suites not green (exit %d) -- stop" % _r.returncode)
print("S1 modules green")

# Exit codes: 0 green, 1 a real test failure, 3 a dependency is not
# importable, 4 a suite FILE is absent. Deliberately NOT 2 -- the interpreter
# itself returns 2 for "can't open file", so reusing it would make a wrong
# runner path indistinguishable from a missing module. 4 is separated from 1
# because "the file is not there" and "the code is broken" have different
# fixes, and conflating them sends you debugging code that never ran.
#
# Expected: '157 checks passed, 0 failed, across 4 suite(s) run' and
# 'ALL GREEN'. If ANY check fails, do not run CELL 6 -- an S1 module that
# fails its own fixtures will produce plausible-looking numbers on real data,
# which is worse than crashing.
# =========================================================================== #
# CELL 6 -- EDIT. Three lines, inside the per-neuron loop.                    #
# =========================================================================== #
# Identifiers must be assigned to the RAW export, before the direction filter
# and before the nearest-node mapping. Assigning them afterwards would key them
# to rows that have already been filtered, which defeats the point: the whole
# reason syn_uid exists is that it survives filtering, re-mapping, re-export
# and a change of segmentation.
#
# BEFORE (colab_run_s1_full.py, inside `for nid in NEURON_IDS:`):
#
#             else:
#                 syn_raw = pd.read_csv(syn_path)
#                 syn_df = sra.map_synapses_to_nodes_raw(
#                     syn_raw, df_raw, voxel_res=VOXEL_RES_NM,
#                     direction=SYNAPSE_DIRECTION)
#
# AFTER:
#
#             else:
#                 syn_raw = pd.read_csv(syn_path)
#                 syn_raw, _uid_rep = su.assign_syn_uid(
#                     syn_raw, nid=nid, voxel_scale=VOXEL_RES_NM)
#                 syn_uid_reports[nid] = _uid_rep
#                 syn_df = sra.map_synapses_to_nodes_raw(
#                     syn_raw, df_raw, voxel_res=VOXEL_RES_NM,
#                     direction=SYNAPSE_DIRECTION)
#
# Two supporting edits:
#   * CELL 5, with the other imports:      import syn_uid as su
#   * CELL 6, next to the other accumulators (the `records, regression_records,
#     frames, failures, gated_out = [], [], {}, [], []` line):
#         syn_uid_reports = {}
#
# The added column rides along into syn_df untouched -- map_synapses_to_nodes_raw
# selects the columns it needs and ignores the rest, so nothing downstream sees
# a schema change.
#
# NOTE ON WHAT THIS IS AND IS NOT FIXING. The current pipeline does NOT lose
# synapses: sra.map_synapses_to_nodes_raw returns one row per surviving synapse,
# each carrying its own node_id and snap_distance_nm. (The last-write-wins
# collapse exists only in the legacy standalone map_synapses_to_segments.py,
# which this driver does not call.) syn_uid is not repairing a leak. It is
# supplying the stable key the raw export lacks -- needed the moment two exports
# of the same cell have to be joined synapse-by-synapse, which is exactly what a
# spine-resolved versus spine-pruned comparison requires, and which lfpy_idx
# cannot do because it moves whenever the segmentation moves.


# =========================================================================== #
# CELL 6c -- NEW. Paste immediately after CELL 6.                             #
# =========================================================================== #
# Reads `frames` from CELL 6, or rebuilds it from the skeletons if absent.
# Runs in two passes because the truncation boundary estimate is POOLED: it
# needs every cell's extent before it can judge any single cell's tips.

import json

import numpy as np
import pandas as pd

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
# rather than failing. Rebuilding is the same operation CELL 6 performs
# (read the CSV, run the labeller at the exporter's threshold); it costs one
# labeller pass per cell and needs nothing from CELL 6's namespace.
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

    import pandas as _pd
    import spine_labeller as _sl
    import morphology_exporter as _mx

    _thr = _mx.SPINE_LENGTH_THRESHOLD_NM
    print("rebuilding labelled frames from %s (threshold %.0f nm, the "
          "exporter's own value) for %d cell(s)..."
          % (SKELETONS_DIR, _thr, len(NEURON_IDS)))                # noqa: F821

    _frames = {}
    for _nid in NEURON_IDS:                                        # noqa: F821
        _path = "%s/neuron_%s.csv" % (SKELETONS_DIR, _nid)         # noqa: F821
        if not os.path.isfile(_path):
            print("   skipping %s: no skeleton at %s" % (_nid, _path))
            continue
        _out = _sl.label_dendritic_spines_robust(
            [_nid], input_dir=SKELETONS_DIR, output_dir=None,      # noqa: F821
            spine_length_threshold_nm=_thr)
        _df = _out[_nid] if isinstance(_out, dict) else _out
        _frames[_nid] = {"raw": _df}
    if not _frames:
        raise RuntimeError(
            "rebuild produced nothing -- no skeleton CSV was found in %s for "
            "any id in NEURON_IDS." % SKELETONS_DIR)               # noqa: F821
    print("rebuilt %d labelled frame(s)" % len(_frames))
    print("NOTE these are labelled but NOT gated: CELL 6's propagation gate "
          "has not been applied, so cells that would have been gated out are "
          "included here. For a bank run, prefer re-running CELL 6.")

frames = _frames          # from CELL 6, or rebuilt above

S1_DIR = OUTPUT_DIR + "/s1"                             # noqa: F821
os.makedirs(S1_DIR, exist_ok=True)

RHO_A_SWEEP = (100.0, 200.0, 300.0, 400.0)

# ---- pass 1: per-cell spine geometry, and the pooled boundary -------------
spine_frames, s1_records = {}, []

for nid, fr in frames.items():                                  # noqa: F821
    df_lab = fr["raw"]                       # pre-prune, raw nm, head/neck
    spine_df = sg.build_spine_geometry(df_lab, nid=nid, input_units="nm")
    spine_frames[nid] = spine_df

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

# ---- READ THIS BEFORE QUOTING ANY NUMBER ABOVE --------------------------
# 1. resistance_trustworthy False  ->  every R_neck above is a function of
#    length alone, because the radii are flat or overwhelmingly at the 50 nm
#    fallback. The numbers are arithmetic, not measurement. Nothing else in
#    the output will tell you this.
# 2. frac_with_neck low  ->  the labeller assigns everything to 'head' when a
#    spine path has fewer than 3 distinct nodes. Those spines get G = 0 and
#    are EXCLUDED from the statistics (not counted as zero-resistance), so a
#    low fraction means the quantiles rest on a minority of spines.
# 3. The R_neck spread across the rho_a sweep is the point, not any single
#    row. Published spine-neck resistances span a wide range partly because
#    rho_a is assumed rather than measured, so a result that survives the
#    whole sweep is worth more than one that holds at a single rho_a.
# 4. truncation_basis 'unresolved' means the tip was too short to judge taper
#    AND not near the z bound. It is absence of evidence, not evidence of a
#    real ending -- those tips are left unflagged deliberately.


# =========================================================================== #
# CELL 8b -- NEW. Paste after CELL 8 (batch plots), before CELL 9.            #
# =========================================================================== #
# Follows the CELL 7/8 pattern exactly: build a figure, hand it to
# ap.save_figure, append the path to `saved` so it lands in the manifest.
# s1_plots computes nothing -- it only draws what CELL 6c already
# produced, so these figures cannot disagree with s1_summary.csv.
#
# READ THEM IN THIS ORDER. The first is a gate on the other four: if the
# radii are the fallback, everything downstream is describing an artefact.

import s1_plots as s0p

if "spine_frames" not in dir() or not spine_frames:              # noqa: F821
    print("CELL 6c has not run (or produced nothing) -- no S1 figures to "
          "draw. Run CELL 6c first; it is what builds `spine_frames`.")
else:
    _figs = [
        # 1. GATE. Are the measured dimensions plausible, and are the radii
        #    real? Panel 4 is the one that matters: a single spike at the
        #    fallback means every resistance below is length alone.
        ("s1_neck_geometry",
         s0p.neck_geometry(spine_frames)),                       # noqa: F821

        # 2. Does spine density peak near the ~90 um the literature reports
        #    for human temporal basal dendrites? A peak in the wrong place
        #    points at the labeller, not the biology.
        ("s1_spine_profile",
         s0p.spine_distance_profile(spine_frames)),              # noqa: F821

        # 3. RESULT. R_neck across the rho_a sweep, against the published
        #    range. The width matters more than agreement with any one
        #    source: three of the five references are rodent and they
        #    disagree with each other by ~an order of magnitude.
        ("s1_neck_resistance",
         s0p.neck_resistance_sweep(spine_frames,                 # noqa: F821
                                   rho_a_ohm_cm=RHO_A_SWEEP)),   # noqa: F821

        # 4. RESULT, and the decision figure for assumption A2. If median
        #    kappa is ~1 across the whole grid, relocating spine synapses to
        #    the shaft is close to exact and the spine-resolved export is not
        #    worth building. If it is not, this says by how much.
        ("s1_attenuation",
         s0p.attenuation_factor(spine_frames,                    # noqa: F821
                                rho_a_ohm_cm=RHO_A_SWEEP)),      # noqa: F821
    ]

    if "trunc_frames" in dir() and trunc_frames:                 # noqa: F821
        # 5. Is the taper threshold defensible? Panel 1 is the evidence:
        #    bimodal means the trough calibrates it, unimodal means no
        #    threshold separates the populations and frac_truncated should be
        #    read as a ranking rather than a count.
        _figs.append(("s1_truncation",
                      s0p.truncation_diagnostics(
                          trunc_frames,                          # noqa: F821
                          ratio_threshold=tf.DEFAULT_TAPER_RATIO_THRESHOLD,
                          margin_um=tf.DEFAULT_MARGIN_UM)))      # noqa: F821

    if "s1_df" in dir() and len(s1_df) > 1:              # noqa: F821
        # 6. Per-cell rollup. Only worth drawing for more than one cell.
        _figs.append(("s1_batch", s0p.s1_batch(s1_df)))  # noqa: F821

    for _name, _fig in _figs:
        _p = ap.save_figure(_fig, "%s/%s.png" % (FIGURE_DIR, _name))  # noqa: F821
        saved.append(_p)                                         # noqa: F821
        print("  %s" % _p)
        if SHOW_INLINE:                                          # noqa: F821
            from IPython.display import Image, display
            display(Image(filename=_p))

    print("\n%d S1 figure(s) written" % len(_figs))

# WHAT TO LOOK FOR, in one line each:
#   neck_geometry     panel 4 a spike at 0.05 um  -> stop, radii are synthetic
#   spine_profile     peak far from ~90 um        -> suspect the labeller
#   neck_resistance   the SPREAD across rho_a     -> how much is assumption
#   attenuation       median kappa near 1.000     -> A2 is close to exact
#   truncation        panel 1 bimodal             -> threshold is calibratable
#   batch             red bars                    -> those cells are unusable


# =========================================================================== #
# CELL 9 -- REPLACE. Full rewrite of the existing summary/manifest cell.       #
# =========================================================================== #
# Changes relative to the current CELL 9:
#   * the S1 columns are merged in, guarded so the cell still runs if
#     CELL 6c was skipped;
#   * the merge key is dtype-checked before it is used;
#   * S1 module versions, the pooled boundary, and the syn_uid reports
#     go into the manifest, so a bank can be reproduced from it;
#   * cells whose radii are suspect are called out explicitly at the end,
#     because that is the one condition under which the R_neck columns are
#     arithmetic rather than measurement.
#
# The DataFrame is called `summary` (not `summary_df`). Everything below is a
# drop-in replacement for lines 701-768 of colab_run_s1_full.py.

rows = []
for r in records:                                               # noqa: F821
    d = r.get("alignment", {})
    nd = d.get("neighbour_distance_um", [np.nan])
    qc = r.get("propagation_qc", {})
    rows.append({
        "nid": r["nid"], "qc_status": r["qc_status"],
        "reasons": ";".join(r.get("reasons", [])),
        "n_sections": r.get("n_sections"), "n_branches": r.get("n_branches"),
        "f_implied": r.get("f_implied"), "F_lit": r.get("F_lit"),
        "nearest_ref_um": round(float(np.min(nd)), 1),
        "pairwise_spread_deg": round(
            float(d.get("pairwise_angle_deg_max", np.nan)), 1),
        "angle_from_z_deg": round(float(r.get("angle_from_z_deg", np.nan)), 1),
        "totnsegs": r.get("totnsegs", qc.get("totnsegs")),
        "gate_status": qc.get("qc_status"),
        "gate_dv_soma_mV": qc.get("C4_soma", {}).get("dv_soma_mV"),
        "gate_dv_min_mV": qc.get("dv_min_mV"),
        "gate_monotone_violations": qc.get("C3_monotone", {}).get("n_violations"),
        "n_synapses": r.get("n_synapses"),
        "n_on_pruned_spine": r.get("n_on_pruned_spine"),
        "n_redirected": r.get("n_redirected"),
        "n_unresolved_spine_bases": r.get("n_unresolved_spine_bases"),
        "n_unknown_type": r.get("n_unknown_type"),
        "cm": CM_BASE, "Ra": RA, "Rm_qc": RM_QC,                # noqa: F821
        "k_neighbors": K_NEIGHBORS,                             # noqa: F821
    })

summary = pd.DataFrame(rows)

# --- S1 merge ------------------------------------------------------- #
# Guarded on the NAME, not on a try/except around the merge: a NameError from
# a skipped CELL 6c and a genuine merge bug should not produce the same
# silence.
_S1_COLS = [
    "n_spines", "n_with_neck", "frac_with_neck",
    "A_spine_total_um2", "A_head_total_um2", "A_neck_total_um2",
    "L_neck_median_um", "d_neck_equiv_median_um", "A_head_median_um2",
    "resistance_trustworthy", "radius_frac_at_default_r",
    "radius_radius_suspect",
    "n_tips", "n_truncated", "frac_truncated", "truncation_basis_counts",
]

if "s1_df" in dir() and len(s1_df):                     # noqa: F821
    _s0 = s1_df                                             # noqa: F821

    # R_neck columns depend on RHO_A_SWEEP, so pick them up by pattern rather
    # than hard-coding a rho. Median only here; p05/p95 stay in
    # s1_summary.csv, which carries the full per-cell detail.
    _rcols = sorted(c for c in _s0.columns
                    if c.startswith("R_neck_MOhm_rho") and c.endswith("_median"))
    _want = [c for c in (_S1_COLS + _rcols) if c in _s0.columns]

    # A silent all-NaN merge from an int64/object key mismatch is the single
    # most likely way this goes wrong, and it looks exactly like "CELL 6c
    # produced nothing". Check it rather than discover it later.
    _lk, _rk = summary["nid"].dtype, _s0["nid"].dtype
    if _lk != _rk:
        print("  merge key dtype mismatch: summary.nid is %s, s1_df.nid "
              "is %s -- coercing both to str" % (_lk, _rk))
        summary["nid"] = summary["nid"].astype(str)
        _s0 = _s0.copy()
        _s0["nid"] = _s0["nid"].astype(str)

    # Re-running this cell must not produce _x/_y suffixed duplicates.
    _dupes = [c for c in _want if c in summary.columns]
    if _dupes:
        summary = summary.drop(columns=_dupes)

    summary = summary.merge(_s0[["nid"] + _want], how="left", on="nid")

    _missing = summary.loc[summary["n_spines"].isna(), "nid"].tolist()
    if _missing:
        print("  %d cell(s) passed the gate but have no S1 row: %s"
              % (len(_missing), _missing))
    print("  S1 merge: %d column(s) added" % len(_want))
else:
    print("  S1 columns NOT merged (CELL 6c not run, or it produced no "
          "rows). The summary is still valid; it simply carries no spine "
          "geometry or truncation evidence.")

summary_path = "%s/alignment_summary.csv" % FIGURE_DIR           # noqa: F821
summary.to_csv(summary_path, index=False)
print(summary.to_string(index=False))

# --- manifest ------------------------------------------------------------ #
_module_versions = {"alignment": al.MODULE_VERSION,              # noqa: F821
                    "alignment_plots": ap.MODULE_VERSION,        # noqa: F821
                    "morphology_exporter": mx.MODULE_VERSION,    # noqa: F821
                    "hoc_qc": hq.MODULE_VERSION,                 # noqa: F821
                    "node_classify": nc.MODULE_VERSION,          # noqa: F821
                    "synapse_redirect_audit": sra.MODULE_VERSION}  # noqa: F821
for _name, _mod in (("spine_geometry", "sg"), ("truncation_flag", "tf"),
                    ("syn_uid", "su"), ("spine_density", "sd")):
    if _mod in dir():
        _module_versions[_name] = eval(_mod).MODULE_VERSION      # noqa: S307

_s1_manifest = None
if "s1_df" in dir() and len(s1_df):                      # noqa: F821
    _s1_manifest = {
        "n_cells": int(len(s1_df)),                          # noqa: F821
        "summary_csv": "%s/s1_summary.csv" % S1_DIR,      # noqa: F821
        "rho_a_sweep_ohm_cm": list(RHO_A_SWEEP),                  # noqa: F821
        "pooled_bounds_um": {k: list(bounds[k]) for k in "xyz"},  # noqa: F821
        "pooled_n_cells": bounds["n_cells"],                      # noqa: F821
        "pooled_n_nodes": bounds["n_nodes"],                      # noqa: F821
        "z_range_fraction_of_slab": bounds["z_range_fraction_of_slab"],  # noqa: F821, E501
        "taper_ratio_threshold": tf.DEFAULT_TAPER_RATIO_THRESHOLD,  # noqa: F821
        "reference_path_um": tf.DEFAULT_REFERENCE_PATH_UM,        # noqa: F821
        "boundary_margin_um": tf.DEFAULT_MARGIN_UM,               # noqa: F821
        "n_cells_radius_suspect": int(
            (summary["resistance_trustworthy"] == False).sum())   # noqa: E712
        if "resistance_trustworthy" in summary.columns else None,
    }

manifest = {
    "module_versions": _module_versions,
    "metadata_csv": METADATA_CSV,                                # noqa: F821
    "n_references": int(len(metadata_df)),                       # noqa: F821
    "k_neighbors": K_NEIGHBORS,                                  # noqa: F821
    "segmentation": {"cm": CM_BASE, "Ra": RA, "lambda_f": 100.0,  # noqa: F821
                     "d_lambda": 0.1, "nsegs_method": "lambda_f"},
    "propagation_gate": {"Rm_ohm_cm2": RM_QC,                    # noqa: F821
                         "g_pas_S_cm2": 1.0 / RM_QC,             # noqa: F821
                         "amp_nA": hq.DEFAULT_AMP_NA,            # noqa: F821
                         "dur_ms": hq.DEFAULT_DUR_MS,            # noqa: F821
                         "dt_ms": hq.DEFAULT_DT_MS,              # noqa: F821
                         "monotone_rtol": hq.MONOTONE_RTOL,      # noqa: F821
                         "soma_window_mV": [hq.SOMA_DV_MIN_MV,   # noqa: F821
                                            hq.SOMA_DV_MAX_MV]},  # noqa: F821
    "domain_split": "retired -- every dendrite is 'dend'",
    "synapse_redirect": {"enabled": RUN_SYNAPSE_REDIRECT,        # noqa: F821
                         "synapses_dir": SYNAPSES_DIR,           # noqa: F821
                         "direction": SYNAPSE_DIRECTION,         # noqa: F821
                         "voxel_res_nm": list(VOXEL_RES_NM)},    # noqa: F821
    "syn_uid": ({str(k): v for k, v in syn_uid_reports.items()}  # noqa: F821
                if "syn_uid_reports" in dir() else None),
    "s1": _s1_manifest,
    "neuron_ids": list(NEURON_IDS),                              # noqa: F821
    "n_aligned": len(records),                                   # noqa: F821
    "gated_out": gated_out, "quarantined": quarantined,          # noqa: F821
    "failures": failures, "figures": saved,                      # noqa: F821
    "summary_csv": summary_path,
}
with open("%s/alignment_manifest.json" % FIGURE_DIR, "w",        # noqa: F821
          newline="\n") as fh:
    fh.write(json.dumps(manifest, indent=2, sort_keys=True, default=str))

if failures:                                                     # noqa: F821
    print("\nFAILURES:")
    for f in failures:                                           # noqa: F821
        print("  %s: %s" % (f["nid"], f["error"]))

# The one warning that must not be buried: a cell with flat or
# fallback-dominated radii yields R_neck values that look entirely plausible
# against the published range and mean nothing, because they are a function
# of neck LENGTH alone. Nothing else in this table distinguishes them.
if "resistance_trustworthy" in summary.columns:
    _bad = summary.loc[summary["resistance_trustworthy"] == False,  # noqa: E712
                       "nid"].tolist()
    if _bad:
        print("\n  [!] %d cell(s) with SUSPECT RADII -- every R_neck and "
              "d_neck_equiv column above is arithmetic, not measurement: %s"
              % (len(_bad), _bad))
    else:
        print("\n  radii OK on all %d cell(s); R_neck columns are usable"
              % len(summary))

print("\nmanifest: %s/alignment_manifest.json" % FIGURE_DIR)     # noqa: F821
