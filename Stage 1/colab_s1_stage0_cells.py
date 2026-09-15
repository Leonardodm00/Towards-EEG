"""Colab cells for the S1 stage-0 modules, against the existing pipeline.

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
    CELL 9   EDIT   optional: carry the new columns into the summary table

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
#         # --- S1 stage 0 ---
#         "spine_geometry.py", "truncation_flag.py", "syn_uid.py",
#         "phi_pipeline_colab.py",
#     ]
#
# and append to REQUIRED_TESTS:
#         "test_spine_geometry.py", "test_truncation_flag.py",
#         "test_syn_uid.py", "run_all_tests.py",
#
# phi_pipeline_colab.py is on that list because CELL 6c calls its
# radius_report() -- the authoritative radius gate. It is an existing project
# module, not a new one; it simply was not needed by this driver before.


# =========================================================================== #
# CELL 3b -- NEW. Paste immediately after CELL 3.                             #
# =========================================================================== #
# CELL 3's pass/fail matcher looks for the string "N / N" in a suite's stdout.
# The stage-0 suites do not print that; they print "  PASS  <name>" per check
# and "ALL CHECKS PASSED" at the end. Adding them to CELL 3's SUITES list would
# therefore mark them FAILED even when green. They get their own cell, which
# calls their own runner -- which additionally does the HPC transfer-safety
# checks (non-ASCII AND carriage-return bytes: different bugs, different byte
# ranges, an ASCII scan will never find CRLF) before running anything.

import os
import subprocess
import sys

_env = dict(os.environ)
_env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p) + (
    os.pathsep + _env["PYTHONPATH"] if _env.get("PYTHONPATH") else "")

_r = subprocess.run([sys.executable, "run_all_tests.py"],
                    capture_output=True, text=True, env=_env)
print(_r.stdout)
if _r.returncode == 2:
    raise RuntimeError(
        "stage-0 dependencies missing -- see the list above. These are "
        "existing project modules; re-run CELL 1 and include them.")
if _r.returncode != 0:
    print(_r.stderr[-2000:])
    raise RuntimeError("stage-0 suites not green -- stop")
print("stage-0 modules green")

# Expected: '122 checks passed, 0 failed, across 3 suite(s)' and 'ALL GREEN'.
# Exit 2 means a dependency file is absent (not a test failure); exit 1 means a
# real failure. If ANY check fails here, do not run CELL 6 -- a stage-0 module
# that fails its own fixtures will produce plausible-looking numbers on real
# data, which is worse than crashing.


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
# Reads `frames`, which CELL 6 has already populated. Touches nothing else.
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

STAGE0_DIR = OUTPUT_DIR + "/stage0"                             # noqa: F821
os.makedirs(STAGE0_DIR, exist_ok=True)

RHO_A_SWEEP = (100.0, 200.0, 300.0, 400.0)

# ---- pass 1: per-cell spine geometry, and the pooled boundary -------------
spine_frames, stage0_records = {}, []

for nid, fr in frames.items():                                  # noqa: F821
    df_lab = fr["raw"]                       # pre-prune, raw nm, head/neck
    spine_df = sg.build_spine_geometry(df_lab, nid=nid, input_units="nm")
    spine_frames[nid] = spine_df

    rq = ppc.radius_report(df_lab) if _HAVE_RADIUS_REPORT else None
    summ = sg.cell_spine_summary(spine_df, rho_a_ohm_cm=RHO_A_SWEEP,
                                 radius_report=rq)
    summ["nid"] = nid
    stage0_records.append(summ)

    path = "%s/neuron_%s_spine_geometry.csv" % (STAGE0_DIR, nid)
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
    trunc.to_csv("%s/neuron_%s_truncation.csv" % (STAGE0_DIR, nid),
                 index=False)
    for rec in stage0_records:
        if rec["nid"] == nid:
            rec["n_tips"] = tsumm["n_tips"]
            rec["n_truncated"] = tsumm["n_truncated"]
            rec["frac_truncated"] = tsumm["frac_truncated"]
            rec["truncation_basis_counts"] = json.dumps(
                tsumm["basis_counts"], sort_keys=True)
    print("neuron %s: %d tips, %d truncated (%.0f%%), basis %s"
          % (nid, tsumm["n_tips"], tsumm["n_truncated"],
             100.0 * tsumm["frac_truncated"], tsumm["basis_counts"]))

stage0_df = pd.DataFrame(stage0_records)
stage0_df.to_csv("%s/stage0_summary.csv" % STAGE0_DIR, index=False)
print("\nwrote %s/stage0_summary.csv  (%d cells)"
      % (STAGE0_DIR, len(stage0_df)))

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
# CELL 9 -- EDIT (optional). Carry the new columns into the summary table.    #
# =========================================================================== #
# CELL 9 builds a per-cell row at colab_run_s1_full.py:711. To have the stage-0
# quantities travel with the rest of the bank rather than sitting in a separate
# CSV, join on nid after that table is built:
#
#     summary_df = summary_df.merge(
#         stage0_df[["nid", "n_spines", "frac_with_neck",
#                    "d_neck_equiv_median_um", "resistance_trustworthy",
#                    "n_tips", "frac_truncated"]],
#         how="left", left_on="nid", right_on="nid")
#
# Adjust the left key if CELL 9's column is called something other than "nid".
# Keep the merge LEFT: a cell that was gated out has no stage-0 row, and an
# inner join would silently drop it from the summary.
