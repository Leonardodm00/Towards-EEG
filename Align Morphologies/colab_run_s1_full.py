"""Colab driver: run the FULL S1 pipeline on real cells, gated on propagation.

Replaces untitled13.py. Two reasons it is a rewrite and not a patch:

  1. untitled13.py as exported does not parse. Colab's "commented out IPython
     magic" transform fired on 19 CONTINUATION lines that begin with the
     string-formatting operator '%', turning e.g.

         v[STRUCTURAL].append("I-15: %d root sections %r"
                              % (len(geo["roots"]), geo["roots"][:6]))
     into
         v[STRUCTURAL].append("I-15: %d root sections %r"
     #                          % (len(geo["roots"]), geo["roots"][:6]))

     which is a SyntaxError at line 351. Every such line here is written so the
     '%' can never start a line: the operator is placed at the END of the
     preceding line. This is a permanent fix, not a one-off repair -- re-export
     this file from Colab and it will still parse.

  2. align_and_export now stages and gates internally, so CELL 6b's manual
     staging/promotion is redundant for the electrical check. It is KEPT for
     the structural check, which is complementary -- see CELL 6b's header.

WHAT IS NEW RELATIVE TO untitled13.py
  CELL 0   clones the repo read-only and sets STRUCTURE_DIR / TESTS_DIR, so
           the code that runs is the code that is committed.
  CELL 1   DELETED. Nothing is uploaded. Uploading was the hazard: Colab
           renames on collision, so a stale /content copy shadowed a fresh one
           and the whole bank passed against code that was not being shipped.
  CELL 2   purges /content, clears sys.modules so an edit is picked up on
           re-run, and RAISES if any module resolves outside the clone.
           neuron + LFPy install moved UP: the gate needs them at CELL 6.
  CELL 3   runs the smoke bank with ABSOLUTE script paths (a bare filename
           resolves against /content and is not found), cwd=TESTS_DIR, and the
           current expected counts.
  CELL 4   RM_QC added.
  CELL 6   passes Rm; no longer rebuilds the labelled frame by hand
           (return_frames=True); reports the gate verdict per cell.
  CELL 6b  no longer promotes -- align_and_export already did that. It now
           audits what is in OUTPUT_DIR and quarantines structural failures.

ORDER OF THE TWO CHECKS, and why it is not ideal
  The propagation gate (CELL 6) runs BEFORE the structural audit (CELL 6b),
  because the gate is inside align_and_export and the audit is a notebook cell.
  Logically the audit should come first: a zero-length section makes lambda_f
  divide by L, so nseg is NaN and the gate fails with a confusing NEURON error
  rather than the precise reason 'zero-length section'. The gate still CATCHES
  it -- nothing broken gets promoted either way -- but the message is worse. If
  this bites on real data, the fix is a pre_gate_fn hook in align_and_export
  calling audit_hoc_geometry on the staged file; flagged, not done.

ASCII only, LF only.
"""

# %% CELL 0 -- clone the repository (run FIRST) -----------------------------
# Source of truth is the repo, not Drive. Cloning here means the code that runs
# is the code that is committed: `git status` shows exactly what differs, and a
# stale hand-copied file cannot silently persist.
#
# REPOSITORY LAYOUT (fixed at S0.4, do not flatten):
#   towards_eeg/structure/   the importable modules
#   tests/                   the smoke suites
#   Align Morphologies/      Colab drivers, INCLUDING this file
# Drivers live outside the package by necessity: they carry /content/drive path
# literals, which S0.9 assertion 4 forbids inside towards_eeg/ (see the commit
# message on "S1: Colab drivers (colab scope)").
#
# READ-ONLY by design. No token is entered here and none should be: a token
# pasted into a Colab cell is written into the notebook's saved output and its
# execution history. Commit and push from your own machine -- see README.
import os
import subprocess

REPO_URL = "https://github.com/Leonardodm00/Towards-EEG.git"
REPO_BRANCH = "s1-morphology"
REPO_DIR = "/content/Towards-EEG"        # /content, NOT Drive: git on a mounted
                                         # Drive is slow and corrupts .git locks


def _git(*args, cwd=REPO_DIR, check=True):
    r = subprocess.run(("git",) + args, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError("git %s failed:\n%s" % (" ".join(args), r.stderr))
    return r.stdout.strip()


if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    # Already cloned this session: fetch and hard-reset, so a half-finished
    # local edit cannot be mistaken for the branch state.
    _git("fetch", "--depth", "1", "origin", REPO_BRANCH)
    _git("checkout", "-B", REPO_BRANCH, "FETCH_HEAD")
    print("updated existing clone")
else:
    r = subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH,
                        REPO_URL, REPO_DIR], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "clone failed:\n%s\nIf the branch does not exist, create it on "
            "GitHub first or change REPO_BRANCH." % r.stderr)
    print("cloned %s @ %s" % (REPO_URL, REPO_BRANCH))

print("  HEAD %s  %s" % (_git("rev-parse", "--short", "HEAD"),
                         _git("log", "-1", "--pretty=%s")))

STRUCTURE_DIR = os.path.join(REPO_DIR, "towards_eeg", "structure")
TESTS_DIR = os.path.join(REPO_DIR, "tests")
for _label, _d in (("STRUCTURE_DIR", STRUCTURE_DIR), ("TESTS_DIR", TESTS_DIR)):
    if not os.path.isdir(_d):
        raise FileNotFoundError(
            "%s missing: %r\nTop-level entries in the branch:\n  %s"
            % (_label, _d, "\n  ".join(sorted(
                d for d in os.listdir(REPO_DIR) if not d.startswith(".")))))
print("STRUCTURE_DIR = %s" % STRUCTURE_DIR)
print("TESTS_DIR     = %s" % TESTS_DIR)


# %% CELL 1 -- (deleted) ----------------------------------------------------
# Nothing to upload: every module and suite comes from the clone. The
# upload step was itself the hazard -- Colab RENAMES on collision, so a fresh
# alignment.py landed as 'alignment (1).py' while `import alignment` kept
# loading the stale /content copy, and the whole bank passed against code that
# was not being shipped. Reading from ONE pinned directory removes the failure
# mode instead of guarding against it.

# %% CELL 2 -- environment, one pinned source directory, NEURON -------------
import importlib
import os
import shutil
import sys

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# neuron and LFPy are needed at CELL 6, not CELL 7: the propagation gate builds
# an LFPy.Cell before anything is committed. Install them later and every cell
# fails the gate with 'qc_raised:ModuleNotFoundError'.
get_ipython().system("pip install -q plotly neuron LFPy")          # noqa: F821

# ---- Where the code lives --------------------------------------------------
# CELL 0 sets STRUCTURE_DIR and TESTS_DIR from the clone; that is the intended
# path. The fallback below is for running straight off a FLAT Drive folder with
# no clone, in which case both point at the same directory. It is skipped
# whenever CELL 0 has run, so the two cells cannot disagree about which copy is
# authoritative.
if "STRUCTURE_DIR" not in globals():
    _flat = "/content/drive/MyDrive/New algorithms/Stage 1"
    STRUCTURE_DIR = TESTS_DIR = _flat
    print("CELL 0 not run -- falling back to flat Drive folder: %s" % _flat)
else:
    print("using the clone from CELL 0")

REQUIRED_MODULES = [
    "spine_density.py", "node_classify.py", "soma_enforce.py",
    "morphology_exporter.py", "spine_labeller.py",
    "alignment.py", "alignment_plots.py", "hoc_qc.py",
    "synapse_redirect_audit.py",
    # not imported by the driver, but test_phi_pipeline.py imports it -- listing
    # it here means a missing copy fails LOUDLY in CELL 2 with the other
    # modules, instead of surfacing as a confusing CELL 3 suite error.
    "phi_pipeline_colab.py",
]
REQUIRED_TESTS = [
    "test_node_classify.py", "test_soma_enforce.py", "test_spine_density.py",
    "test_phi_pipeline.py", "test_morphology_exporter.py",
    "smoke_synapse_redirect_audit.py", "smoke_alignment.py",
    "smoke_alignment_plots.py", "test_hoc_qc.py", "test_align_gate.py",
]

for _label, _d in (("STRUCTURE_DIR", STRUCTURE_DIR), ("TESTS_DIR", TESTS_DIR)):
    if not os.path.isdir(_d):
        raise FileNotFoundError("%s does not exist: %r" % (_label, _d))

missing = [f for f in REQUIRED_MODULES
           if not os.path.isfile(os.path.join(STRUCTURE_DIR, f))]
missing += [f for f in REQUIRED_TESTS
            if not os.path.isfile(os.path.join(TESTS_DIR, f))]
if missing:
    raise FileNotFoundError(
        "missing from the clone:\n  %s\n\nModules belong in %r and suites in "
        "%r. A file in neither place is not merely absent -- an older copy "
        "elsewhere on sys.path would shadow it silently." % (
            "\n  ".join(missing), STRUCTURE_DIR, TESTS_DIR))

# Remove any leftover .py in /content from an earlier upload-based run. With
# '' (the CWD) ahead of the clone on sys.path such a file WINS, and importing a
# module you did not edit is the failure that looks like a passing test suite.
stale = [os.path.join("/content", f) for f in os.listdir("/content")
         if f.endswith(".py") and os.path.isfile(os.path.join("/content", f))]
for p in stale:
    os.remove(p)
shutil.rmtree("/content/__pycache__", ignore_errors=True)
if stale:
    print("removed %d stale .py from /content (would have shadowed "
          "the clone)" % len(stale))

_own = {os.path.abspath(STRUCTURE_DIR), os.path.abspath(TESTS_DIR)}
sys.path = [STRUCTURE_DIR, TESTS_DIR] + [
    p for p in sys.path if os.path.abspath(p or ".") not in _own]

# Drop anything already imported. Without this, editing a module on Drive and
# re-running the notebook silently keeps testing the version from the FIRST
# run of the session.
for name in [f[:-3] for f in REQUIRED_MODULES]:
    sys.modules.pop(name, None)
importlib.invalidate_caches()

print()
for f in REQUIRED_MODULES:
    m = importlib.import_module(f[:-3])
    src = os.path.abspath(getattr(m, "__file__", ""))
    if os.path.dirname(src) not in _own:
        raise ImportError(
            "%s was imported from %r, which is neither STRUCTURE_DIR nor "
            "TESTS_DIR -- something is shadowing it." % (f, src))
    print("OK  %-30s %s" % (f, getattr(m, "MODULE_VERSION", "")))

try:
    import LFPy
    import neuron
    print("\nNEURON %s, LFPy %s" % (
        neuron.__version__, getattr(LFPy, "__version__", "?")))
except Exception as exc:                                     # noqa: BLE001
    raise RuntimeError(
        "NEURON/LFPy did not import (%r). The propagation gate cannot run "
        "without them, and every cell would be reported as failing to "
        "conduct." % exc)

print("\nall %d modules imported from the clone" % len(REQUIRED_MODULES))

# %% CELL 3 -- the full smoke bank ------------------------------------------
import subprocess

# A subprocess inherits NEITHER runtime sys.path entries NOR any notion of
# where a script lives. Both must be supplied, and they are DIFFERENT fixes:
#   PYTHONPATH    makes imports work INSIDE the subprocess
#   absolute path makes python find the SCRIPT ITSELF -- a bare filename
#                 resolves against the CWD (/content), which is empty now
#   cwd=TESTS_DIR runs each suite beside its own dependencies, as it does
#                 locally, so a suite reading a sibling file still works
_env = dict(os.environ)
_env["PYTHONPATH"] = os.pathsep.join([STRUCTURE_DIR, TESTS_DIR]) + (
    os.pathsep + _env["PYTHONPATH"] if _env.get("PYTHONPATH") else "")

SUITES = [
    ("test_node_classify.py", 11),
    ("test_soma_enforce.py", 8),
    ("test_spine_density.py", 7),
    ("test_phi_pipeline.py", 5),
    ("test_morphology_exporter.py", 12),
    ("smoke_synapse_redirect_audit.py", 10),     # R9 loader, R10 integration
    ("smoke_alignment.py", 11),
    ("smoke_alignment_plots.py", 12),
    ("test_hoc_qc.py", 12),
    ("test_align_gate.py", 9),
]

_failed = []
for script, n_expect in SUITES:
    path = os.path.join(TESTS_DIR, script)
    r = subprocess.run([sys.executable, path], capture_output=True,
                       text=True, env=_env, cwd=TESTS_DIR)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)"
    ok = r.returncode == 0 and (
        ("%d / %d" % (n_expect, n_expect)) in r.stdout
        or ("%d/%d" % (n_expect, n_expect)) in r.stdout)
    print("%-34s %-28s %s" % (script, tail, "OK" if ok else "*** FAILED ***"))
    if not ok:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        _failed.append(script)

if _failed:
    raise RuntimeError("smoke bank not green: %s -- stop" % _failed)
print("\nall %d suites green" % len(SUITES))

# NOTE test_hoc_qc Q12 now runs for real: with NEURON installed it builds a
# two-section .hoc and gates it. If it printed 'skipped', LFPy is not visible
# to the SUBPROCESS even though CELL 2 imported it -- a PYTHONPATH problem.

# %% CELL 4 -- EDIT THIS CELL -----------------------------------------------
# Bank source. Two ways to supply the id list -- use ONE:
#   (a) NEURON_IDS_CSV points at a CSV with an id column (e.g. an L3_exc list
#       you maintain elsewhere). This is the normal path once a bank is more
#       than a handful of cells -- hand-typing 20-100 ids invites transcription
#       errors that show up as silent 'file not found' skips three cells in.
#   (b) leave NEURON_IDS_CSV = None and edit the literal NEURON_IDS list below,
#       as before. Fine for a handful of cells.
NEURON_IDS_CSV = None
# e.g. "/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons/L3_exc_ids.csv"
NEURON_IDS_COLUMN = "nid"
NEURON_IDS = [4683431368, 4683651279]

METADATA_CSV = ("/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons/"
                "alignment_metadata_L3.csv")
SKELETONS_DIR = "/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons"
# Synapse export per neuron: neuron_{id}_synapses.csv, H01 format (location_x/
# y/z in VOXEL units, optional direction/synapse_type columns). Same directory
# as the skeletons unless you keep them separately -- edit if so.
SYNAPSES_DIR = "/content/drive/MyDrive/Colab Notebooks/Synapse database"
# 'incoming' keeps only postsynaptic sites ON this cell -- the ones that matter
# for a single-cell passive model. None disables the filter.
SYNAPSE_DIRECTION = "incoming"
VOXEL_RES_NM = (8.0, 8.0, 33.0)          # H01 default; anisotropic z-sampling
# False restores the pre-redirect behaviour: export + align + gate only, no
# synapse loading, no C-09. A cell whose synapse file is simply MISSING (not
# unusual across a real bank) is not an error either way -- CELL 6 proceeds
# without the redirect for that cell alone and says so.
RUN_SYNAPSE_REDIRECT = True

OUTPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/Aligned Neurons HOC/temp"
FIGURE_DIR = OUTPUT_DIR + "/figures"

# Skip a neuron whose neuron_{id}_alignment.json already exists in OUTPUT_DIR
# rather than re-exporting it. This is what makes a bank run resumable across
# a Colab disconnect: re-running the notebook from CELL 6 picks up where it
# left off instead of redoing every already-committed cell. A gate FAILURE
# commits nothing (see align_and_export's staging design), so a previously
# failed cell is correctly NOT skipped and gets retried.
CHECKPOINT = True

# EVERY neuron gets the full treatment: export, align, gate, redirect, hoc
# validation, its own four figures. Nothing is sampled. The loop is STREAMING
# instead: one neuron is processed to completion, every artefact written to
# OUTPUT_DIR, its summary row appended to BANK_SUMMARY_CSV, and then every
# large object it created is deleted before the next neuron is loaded. Peak
# RAM is therefore one neuron's worth regardless of bank size, and only a
# ~1 KB scalar record per neuron is retained in memory for the batch plots.
#
# The rigidity control re-exports each cell a SECOND time without alignment, so
# it roughly doubles per-cell cost. It is the check that alignment never leaks
# into a phi-affecting quantity, so it stays ON for every cell; set False only
# if wall-clock forces it.
RUN_RIGIDITY_CONTROL = True

# Save the four per-neuron figures. Each is written to disk and the figure
# object freed immediately. False skips figure GENERATION only -- every neuron
# is still fully processed, and angle_from_z_deg is still computed, so the
# batch orientation plot works either way. Set False for a large bank run.
SAVE_PER_NEURON_FIGURES = True

K_NEIGHBORS = 3          # local alignment, settled 25 July 2026

# Segmentation. These fix the lfpy_idx index space and are stamped into
# provenance. Eyal 2016 for human L2/3 pyramidal; for INTERNEURONS use
# alignment.RA_YAO_INTERNEURON (100.0) with the subtype's cm instead.
CM_BASE = 0.50           # uF/cm2, intrinsic and shaft-referenced (contract C-14)
RA = 268.5               # ohm cm

# QC-only. Rm enters NOTHING downstream: it is the passive leak used to make a
# transient decay during the gate, and is discarded afterwards. cm and Ra are
# NOT QC-only -- they fix nseg, so the gate must run at the same discretisation
# the bank will be used with, which is why they are shared above.
# 24000 ohm cm2 = tau_m 12.03 ms (Deitcher 2017) / cm 0.5 uF/cm2 (Eyal 2016).
RM_QC = 24000.0

MAX_SEGMENTS_INTERACTIVE = 30000
MAX_SEGMENTS_STATIC = 20000
# Inline Plotly rendering keeps every figure ALIVE in the notebook's output,
# which defeats the streaming design: the browser accumulates what the kernel
# just freed. Leave False for a bank run. Figures are always saved to disk.
SHOW_INLINE = False

# %% CELL 5 -- imports + the metadata bank + neuron id list -----------------
import json
import time

import numpy as np
import pandas as pd

import morphology_exporter as mx
import spine_labeller as sl
import node_classify as nc
import alignment as al
import alignment_plots as ap
import hoc_qc as hq
import synapse_redirect_audit as sra

os.makedirs(FIGURE_DIR, exist_ok=True)
metadata_df = al.load_alignment_metadata(METADATA_CSV)
print("bank %s: %d references" % (os.path.basename(METADATA_CSV),
                                  len(metadata_df)))
print("gate: Rm %.0f ohm cm2 -> g_pas %.3e S/cm2, step %.2f nA for %.0f ms" % (
    RM_QC, 1.0 / RM_QC, hq.DEFAULT_AMP_NA, hq.DEFAULT_DUR_MS))

if NEURON_IDS_CSV:
    if not os.path.isfile(NEURON_IDS_CSV):
        raise FileNotFoundError(
            "NEURON_IDS_CSV does not exist: %r" % NEURON_IDS_CSV)
    _ids_df = pd.read_csv(NEURON_IDS_CSV)
    if NEURON_IDS_COLUMN not in _ids_df.columns:
        raise ValueError(
            "column %r not in %r -- has: %s" % (
                NEURON_IDS_COLUMN, NEURON_IDS_CSV, list(_ids_df.columns)))
    NEURON_IDS = _ids_df[NEURON_IDS_COLUMN].dropna().astype(np.int64).tolist()
    _dupes = len(NEURON_IDS) - len(set(NEURON_IDS))
    if _dupes:
        print("WARNING: %d duplicate id(s) in %s -- each will be (re)processed "
              "once, per CHECKPOINT" % (_dupes, NEURON_IDS_CSV))

if not NEURON_IDS:
    raise ValueError("NEURON_IDS is empty -- nothing to run")

print("bank to process: %d neuron ids" % len(NEURON_IDS))


BANK_SUMMARY_CSV = os.path.join(FIGURE_DIR, "bank_summary.csv")

# Every scalar the batch plots need. alignment_plots._records_frame reads 11
# fields and rigidity() compares 6 keys, ALL of them scalars -- so a compact
# record keeps every batch plot working while holding nothing large.
COMPACT_DROP = ("frames", "files", "propagation_qc", "hoc_validation")


def _compact(res):
    """Strip a per-neuron record to scalars, for the in-memory batch list.

    Removes: `frames` (a full labelled DataFrame), `files` (paths, re-derivable
    from disk), `propagation_qc` (nested per-criterion reports with violation
    lists) and `hoc_validation` (a geometry dict carrying section-name lists).
    All four are ALREADY persisted -- propagation_qc and the rest inside
    neuron_{nid}_alignment.json, the geometry inside
    neuron_{nid}_hoc_validation.json -- so dropping them from memory loses
    nothing recoverable.

    `alignment` is KEPT: it is a 3-vector, a 3x3 matrix and a few scalars, and
    _records_frame reads det/orthonormality_error/pairwise_angle_deg_max and
    neighbour_distance_um straight out of it.
    """
    out = {k: v for k, v in res.items() if k not in COMPACT_DROP}
    qc = res.get("propagation_qc") or {}
    out["gate_status"] = qc.get("qc_status")
    out["gate_dv_soma_mV"] = qc.get("C4_soma", {}).get("dv_soma_mV")
    out["gate_dv_min_mV"] = qc.get("dv_min_mV")
    out["gate_monotone_violations"] = (
        qc.get("C3_monotone", {}).get("n_violations"))
    out["totnsegs"] = res.get("totnsegs", qc.get("totnsegs"))
    hv = res.get("hoc_validation") or {}
    geo = hv.get("geometry") or {}
    out["hoc_verdict"] = hv.get("verdict")
    out["soma_diameter_um"] = geo.get("soma_diameter_um")
    out["largest_diameter_um"] = geo.get("largest_diameter_um")
    out["largest_diameter_section"] = geo.get("largest_diameter_section")
    out["total_length_um"] = geo.get("total_length_um")
    return out


def _summary_row(rec):
    """One flat row for BANK_SUMMARY_CSV. Scalars only, no nesting."""
    d = rec.get("alignment", {}) or {}
    nd = d.get("neighbour_distance_um") or [float("nan")]
    return {
        "nid": rec.get("nid"), "qc_status": rec.get("qc_status"),
        "reasons": ";".join(rec.get("reasons", [])),
        "n_sections": rec.get("n_sections"),
        "n_branches": rec.get("n_branches"),
        "f_implied": rec.get("f_implied"), "F_lit": rec.get("F_lit"),
        "A_shaft_um2": rec.get("A_shaft_um2"),
        "A_spine_um2": rec.get("A_spine_um2"),
        "nearest_ref_um": round(float(np.min(nd)), 1),
        "pairwise_spread_deg": d.get("pairwise_angle_deg_max"),
        "angle_from_z_deg": rec.get("angle_from_z_deg"),
        "totnsegs": rec.get("totnsegs"),
        "gate_status": rec.get("gate_status"),
        "gate_dv_soma_mV": rec.get("gate_dv_soma_mV"),
        "gate_dv_min_mV": rec.get("gate_dv_min_mV"),
        "gate_monotone_violations": rec.get("gate_monotone_violations"),
        "hoc_verdict": rec.get("hoc_verdict"),
        "soma_diameter_um": rec.get("soma_diameter_um"),
        "largest_diameter_um": rec.get("largest_diameter_um"),
        "largest_diameter_section": rec.get("largest_diameter_section"),
        "total_length_um": rec.get("total_length_um"),
        "n_synapses": rec.get("n_synapses"),
        "n_on_pruned_spine": rec.get("n_on_pruned_spine"),
        "n_redirected": rec.get("n_redirected"),
        "n_unresolved_spine_bases": rec.get("n_unresolved_spine_bases"),
        "n_unknown_type": rec.get("n_unknown_type"),
        "rigidity_identical": rec.get("rigidity_identical"),
        "exporter_id": rec.get("exporter_id"),
        "cm": CM_BASE, "Ra": RA, "Rm_qc": RM_QC, "k_neighbors": K_NEIGHBORS,
    }


def _append_summary_row(rec):
    """Append ONE row to BANK_SUMMARY_CSV immediately, header on first write.

    Written per neuron rather than once at the end so that a kernel death at
    neuron 60 of 80 still leaves 59 complete rows on disk. This file, not the
    in-memory list, is the bank's record of itself.
    """
    row = pd.DataFrame([_summary_row(rec)])
    header = not os.path.isfile(BANK_SUMMARY_CSV)
    row.to_csv(BANK_SUMMARY_CSV, mode="a", header=header, index=False)


RUN_LOG_PATH = os.path.join(OUTPUT_DIR, "run_log.jsonl")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _log_run(entry):
    """Append one JSON line per cell. Not the source of truth -- each cell's
    own alignment.json/hoc_validation.json is -- but gives a flat, greppable
    per-cell history of a bank run across possibly several sessions, without
    having to re-derive it from directory listings."""
    entry = dict(entry)
    entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(RUN_LOG_PATH, "a", newline="\n") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _reconstruct_committed_files(output_dir, nid):
    """Rebuild a res['files']-shaped dict from what is ACTUALLY on disk, for
    RESUME only. Never trusts a path string inside an old JSON -- a Drive
    folder can be renamed after the fact -- every path is re-derived from
    output_dir + nid and checked to exist before being included."""
    files = {}
    hoc_path = os.path.join(output_dir, "neuron_%s_aligned.hoc" % nid)
    if os.path.isfile(hoc_path):
        files["hoc"] = hoc_path
    for name in ("phi", "segment_map", "section_table", "spine_bases",
                "synapses", "mapped_synapses"):
        p = os.path.join(output_dir, "neuron_%s_%s.csv" % (nid, name))
        if os.path.isfile(p):
            files[name] = p
    for name in ("provenance", "hoc_validation"):
        p = os.path.join(output_dir, "neuron_%s_%s.json" % (nid, name))
        if os.path.isfile(p):
            files[name] = p
    prov_path = os.path.join(output_dir, "neuron_%s_alignment.json" % nid)
    if os.path.isfile(prov_path):
        files["alignment_provenance"] = prov_path
    return files


# %% CELL 6a -- helpers: structural audit, quarantine, per-neuron figures ---
# Definitions only; nothing runs here. They are used INSIDE the CELL 6 loop so
# each neuron is fully judged and fully drawn before its artefacts are freed.
#
# The structural audit is COMPLEMENTARY to the propagation gate, not a
# duplicate. The gate asks "does a transient reach every compartment"; this
# asks "is the FILE a well-formed NEURON model". They overlap (a detached
# section fails both) but neither contains the other: the gate cannot see two
# somatic sections or a wrong root, and the audit cannot see an inverted
# connection. Neither promotes -- align_and_export already committed -- so the
# audit's job is to quarantine what the electrical test did not catch.
import gc
import re
import shutil

from IPython.display import display

# Every figure path written across the whole bank run.
saved = []

STRUCTURAL, SOFT = "structural", "soft"
MIN_SOMA_DIAM_UM = 4.0


def audit_hoc_geometry(path):
    """Structural audit of an emitted .hoc. No NEURON needed."""
    txt = open(path, "r", encoding="ascii").read()

    arrays = {m[0]: int(m[1]) for m in re.findall(r"create (\w+)\[(\d+)\]", txt)}
    secs = {"%s[%d]" % (a, n) for a, k in arrays.items() for n in range(k)}

    conn = re.findall(
        r"connect\s+(\w+\[\d+\])\(([\d.]+)\),\s*(\w+\[\d+\])\(([\d.]+)\)", txt)
    child_of = {c: p for c, _, p, _ in conn}
    kids = {}
    for c, _, p, _ in conn:
        kids.setdefault(p, []).append(c)

    roots = sorted(secs - set(child_of))
    somatic = sorted(s for s in secs if s.startswith("soma"))

    start = "soma[0]" if "soma[0]" in secs else (roots[0] if roots else None)
    seen, stack = set(), ([start] if start else [])
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        stack.extend(kids.get(s, []))
    orphans = sorted(secs - seen)

    cur, pts, n_declared, unparsable = None, {}, {}, []
    for line in txt.splitlines():
        m = re.match(r"\s*(\w+\[\d+\])\s*\{", line)
        if m:
            cur = m.group(1)
            pts.setdefault(cur, [])
        # Permissive token capture, then explicit float conversion. A strict
        # numeric regex would FAIL TO MATCH 'pt3dadd(nan, ...)' and drop the
        # point silently, which is worse than a loud failure.
        m2 = re.search(r"pt3dadd\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)",
                       line)
        if m2 and cur is not None:
            n_declared[cur] = n_declared.get(cur, 0) + 1
            try:
                pts[cur].append([float(g.strip()) for g in m2.groups()])
            except ValueError:
                unparsable.append(cur)

    zero_len, bad_diam, few_pts, nonfinite, lengths = [], [], [], [], {}
    for s in sorted(secs):
        P = pts.get(s, [])
        if len(P) < 2:
            few_pts.append(s)
            continue
        A = np.asarray(P, dtype=float)
        if not np.isfinite(A).all():
            nonfinite.append(s)
            continue
        L = float(np.sum(np.linalg.norm(np.diff(A[:, :3], axis=0), axis=1)))
        lengths[s] = L
        if L <= 1e-9:
            zero_len.append(s)
        if A[:, 3].min() <= 0.0:
            bad_diam.append(s)

    dropped = sorted(s for s, n in n_declared.items() if len(pts.get(s, [])) != n)
    diam_max = {s: float(np.asarray(P)[:, 3].max()) for s, P in pts.items() if P}
    biggest = max(diam_max, key=diam_max.get) if diam_max else None

    return {"path": str(path), "n_sections": len(secs), "arrays": arrays,
            "n_connect": len(conn), "roots": roots, "somatic_sections": somatic,
            "n_reachable": len(seen), "orphans": orphans,
            "zero_length": zero_len, "nonpositive_diam": bad_diam,
            "few_pt3d": few_pts, "nonfinite": nonfinite,
            "unparsable_pt3d": sorted(set(unparsable)), "dropped_pt3d": dropped,
            "total_length_um": float(sum(lengths.values())),
            "largest_diameter_section": biggest,
            "largest_diameter_um": (float(diam_max[biggest]) if biggest
                                    else float("nan")),
            "soma_diameter_um": float(diam_max.get("soma[0]", float("nan")))}


def classify_violations(geo, neuron_rep=None, min_soma_diam_um=MIN_SOMA_DIAM_UM):
    """Q6 ladder: STRUCTURAL -> fail, SOFT -> pass_low_confidence."""
    v = {STRUCTURAL: [], SOFT: []}

    if len(geo["roots"]) != 1:
        v[STRUCTURAL].append("I-15: %d root sections %r" % (
            len(geo["roots"]), geo["roots"][:6]))
    if len(geo["somatic_sections"]) != 1:
        v[STRUCTURAL].append("I-15: %d somatic sections" % (
            len(geo["somatic_sections"]),))
    if geo["roots"] and geo["somatic_sections"] and (
            geo["roots"][0] != geo["somatic_sections"][0]):
        v[STRUCTURAL].append("I-15: root %s is not the soma" % geo["roots"][0])
    if geo["orphans"]:
        v[STRUCTURAL].append("debris: %d of %d sections unreachable, e.g. %r" % (
            len(geo["orphans"]), geo["n_sections"], geo["orphans"][:5]))
    if geo["n_connect"] != geo["n_sections"] - 1:
        v[STRUCTURAL].append("not a tree: %d connect for %d sections" % (
            geo["n_connect"], geo["n_sections"]))

    for key, msg in (("zero_length", "zero-length sections (NaN nseg)"),
                     ("nonpositive_diam", "non-positive diameters"),
                     ("few_pt3d", "sections with fewer than 2 pt3d points"),
                     ("nonfinite", "non-finite coordinates"),
                     ("unparsable_pt3d", "unparsable pt3dadd coordinates"),
                     ("dropped_pt3d", "pt3d points lost during parsing")):
        if geo[key]:
            v[STRUCTURAL].append("%s: %d %r" % (msg, len(geo[key]), geo[key][:5]))

    # apic_dend / basal_dend must never appear: the split is retired and
    # node_classify.assert_domain_collapsed guards it at emission. Checking the
    # FILE too means a hand-edited .hoc cannot smuggle one past.
    retired = sorted(set(geo["arrays"]) & set(nc.RETIRED_DOMAIN_ARRAYS))
    if retired:
        v[STRUCTURAL].append("retired domain arrays present: %r" % retired)

    if geo["largest_diameter_section"] != "soma[0]":
        v[SOFT].append("I-18: largest-calibre section is %s (%.3f um), not the "
                       "soma (%.3f um)" % (geo["largest_diameter_section"],
                                           geo["largest_diameter_um"],
                                           geo["soma_diameter_um"]))
    if not (geo["soma_diameter_um"] >= float(min_soma_diam_um)):
        v[SOFT].append("I-18: soma diameter %.3f um is below the %.2f um floor" % (
            geo["soma_diameter_um"], float(min_soma_diam_um)))

    if neuron_rep is not None and neuron_rep.get("neuron_available"):
        if neuron_rep.get("I16_no_synapse_sections") is False:
            v[STRUCTURAL].append("I-16: synapse label in the section vocabulary")
        if neuron_rep.get("I13a_vocabulary_admissible") is False:
            v[STRUCTURAL].append("I-13a: inadmissible arrays %r" % (
                neuron_rep.get("arrays"),))
        if neuron_rep.get("I15_single_soma_is_root") is False:
            v[STRUCTURAL].append("I-15 (NEURON): roots=%r" % (
                neuron_rep.get("root_sections"),))
        n_sec = neuron_rep.get("n_sections")
        if n_sec is not None and n_sec != geo["n_sections"]:
            v[STRUCTURAL].append("NEURON loaded %d sections, file declares %d" % (
                n_sec, geo["n_sections"]))
    return v


def _verdict(v):
    if v[STRUCTURAL]:
        return "fail"
    return "pass_low_confidence" if v[SOFT] else "pass"


def _merge_qc(existing, hoc_verdict):
    """The emitted file can only lower confidence, never raise it."""
    order = {"pass": 0, "pass_low_confidence": 1, "fail": 2}
    return max([existing, hoc_verdict], key=lambda s: order.get(s, 2))


# NEURON is loaded in a SUBPROCESS. h.load_file() is global and cumulative:
# validating several cells in one kernel leaves earlier sections in h.allsec(),
# so the second cell would report the sum of both and fail on n_sections. An
# isolated interpreter per file is the only clean way, and a NEURON segfault
# then cannot take the notebook down.
_NEURON_SNIPPET = r"""
import json, sys
sys.path[:0] = %r
import morphology_exporter as mx
print("@@@" + json.dumps(mx.validate_hoc(%r, min_soma_diam_um=%r), default=str))
"""


def neuron_validate(hoc_path, timeout=300):
    code = _NEURON_SNIPPET % ([p for p in sys.path if p], str(hoc_path),
                              float(MIN_SOMA_DIAM_UM))
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=timeout, env=_env)
        for line in r.stdout.splitlines():
            if line.startswith("@@@"):
                return json.loads(line[3:])
        return {"neuron_available": False, "import_error": (r.stderr or "")[-300:]}
    except Exception as exc:                                 # noqa: BLE001
        return {"neuron_available": False, "import_error": repr(exc)}



def _merge_hoc_verdict(existing, hoc_verdict):
    """The emitted file can only LOWER confidence, never raise it."""
    order = {"pass": 0, "pass_low_confidence": 1, "fail": 2}
    return max([existing, hoc_verdict], key=lambda x: order.get(x, 2))


def _write_validation_json(dest_dir, nid, validation):
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, "neuron_%s_hoc_validation.json" % nid)
    with open(path, "w", newline="\n") as fh:
        fh.write(json.dumps(validation, indent=2, sort_keys=True, default=str))
    return path


def _quarantine(output_dir, nid, res):
    """Move every artefact of a structurally invalid cell out of the bank.

    Matches on the neuron_{nid} prefix rather than res['files'], so an artefact
    the exporter wrote but did not register travels too and nothing is left
    behind to be mistaken for a valid bank member. Also rewrites res['files']
    to the new location, so the record stays truthful about where things are.
    """
    qdir = os.path.join(output_dir, "_quarantine")
    os.makedirs(qdir, exist_ok=True)
    prefix = "neuron_%s" % nid
    for f in sorted(os.listdir(output_dir)):
        src_p = os.path.join(output_dir, f)
        if f.startswith(prefix) and os.path.isfile(src_p):
            shutil.move(src_p, os.path.join(qdir, f))
    for k, val in list(res.get("files", {}).items()):
        pth = val if isinstance(val, str) else (
            val.get("path") if isinstance(val, dict) else None)
        if not pth:
            continue
        newp = os.path.join(qdir, os.path.basename(pth))
        if isinstance(val, str):
            res["files"][k] = newp
        else:
            val["path"] = newp
    return qdir


def _save_neuron_figures(nid, df_lab, df_ali, soma_pos, diag):
    """Write this neuron's four figures and release every figure object.

    Plotly figures are NOT closed by ap.save_figure (only matplotlib is), so
    they are deleted explicitly here -- an interactive arbour figure carries
    the full coordinate arrays and would otherwise survive the iteration.
    """
    written = []
    fig = ap.arbour_comparison(df_lab, df_ali, nid, raw_units="nm",
                               aligned_units="nm",
                               max_segments=MAX_SEGMENTS_INTERACTIVE)
    written.append(ap.save_figure(fig, "%s/neuron_%s_arbour.html" % (
        FIGURE_DIR, nid)))
    if SHOW_INLINE:
        display(fig)
    del fig

    fig = ap.arbour_static(df_lab, df_ali, nid,
                           max_segments=MAX_SEGMENTS_STATIC)
    written.append(ap.save_figure(fig, "%s/neuron_%s_arbour.png" % (
        FIGURE_DIR, nid)))
    del fig

    fig = ap.depth_profile(df_ali, nid, units="nm")
    written.append(ap.save_figure(fig, "%s/neuron_%s_depth.png" % (
        FIGURE_DIR, nid)))
    del fig

    fig = ap.neighbourhood(soma_pos, metadata_df, diag, nid)
    written.append(ap.save_figure(fig, "%s/neuron_%s_neighbourhood.png" % (
        FIGURE_DIR, nid)))
    del fig

    saved.extend(written)
    return written


# %% CELL 6 -- export + align + GATE, one cell at a time --------------------
# STREAMING: one neuron processed to completion, persisted, then every large
# object it created is released before the next is read. Peak RAM is one
# neuron's worth regardless of how many are in the bank.
import traceback


def make_label_fn(nid):
    """In-memory adapter around the verbatim L1653 labeller, which is
    disk-based. Bound per neuron because the labeller keys on the filename."""
    def label_fn(df, threshold_nm):
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp()
        try:
            df.to_csv(os.path.join(tmp, "neuron_%s.csv" % nid), index=False)
            out = sl.label_dendritic_spines_robust(
                [nid], input_dir=tmp, output_dir=None,
                spine_length_threshold_nm=threshold_nm)
            return out[nid] if isinstance(out, dict) else out
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return label_fn


# Only SCALAR records are retained across iterations. Everything large is
# written to disk and deleted before the next neuron is loaded.
records, regression_records, failures, gated_out, quarantined = [], [], [], [], []
n_resumed = 0
_t_loop_start = time.time()
_durations = []          # fresh cells only -- resumed cells would skew the ETA

for _i, nid in enumerate(NEURON_IDS):
    print("\n" + "=" * 70)
    print("neuron %s  [%d/%d]" % (nid, _i + 1, len(NEURON_IDS)))
    if _durations:
        eta_s = np.mean(_durations) * (len(NEURON_IDS) - _i)
        print("  elapsed %.0f min, avg %.0f s/fresh cell, ETA ~%.0f min" % (
            (time.time() - _t_loop_start) / 60.0, np.mean(_durations),
            eta_s / 60.0))
    t0 = time.time()

    prov_path = os.path.join(OUTPUT_DIR, "neuron_%s_alignment.json" % nid)
    if CHECKPOINT and os.path.isfile(prov_path):
        with open(prov_path) as fh:
            res = json.load(fh)
        committed = _reconstruct_committed_files(OUTPUT_DIR, nid)
        if "hoc" in committed:
            n_resumed += 1
            print("  resumed from %s (qc=%s) -- skipping re-export" % (
                os.path.basename(prov_path), res.get("qc_status")))
            hv_path = os.path.join(OUTPUT_DIR,
                                   "neuron_%s_hoc_validation.json" % nid)
            if os.path.isfile(hv_path):
                with open(hv_path) as fh:
                    res["hoc_validation"] = json.load(fh)
            rec = _compact(res)
            records.append(rec)
            _log_run({"nid": nid, "event": "resumed",
                      "qc_status": rec.get("qc_status")})
            del res, committed, rec
            continue
        print("  checkpoint found but no committed .hoc -- reprocessing")
        del res, committed

    # Names bound inside the try, so `finally` must tolerate their absence.
    df_raw = syn_raw = syn_df = res = ctl = df_lab = df_ali = None
    try:
        path = "%s/neuron_%s.csv" % (SKELETONS_DIR, nid)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        df_raw = pd.read_csv(path)
        label_fn = make_label_fn(nid)

        if RUN_SYNAPSE_REDIRECT:
            syn_path = "%s/neuron_%s_synapses.csv" % (SYNAPSES_DIR, nid)
            if not os.path.isfile(syn_path):
                print("  no synapse file at %s -- proceeding WITHOUT the "
                      "redirect for this cell" % syn_path)
            else:
                syn_raw = pd.read_csv(syn_path)
                syn_df = sra.map_synapses_to_nodes_raw(
                    syn_raw, df_raw, voxel_res=VOXEL_RES_NM,
                    direction=SYNAPSE_DIRECTION)
                del syn_raw
                syn_raw = None
                labels = syn_df["synapse_label"].value_counts().to_dict()
                print("  %d synapses mapped to nodes (%s), median snap "
                      "%.1f nm, by label %s" % (
                          len(syn_df), SYNAPSE_DIRECTION or "all",
                          syn_df["snap_distance_nm"].median(), labels))
                # An all-unknown result is an upstream schema mismatch, not
                # data: H01 codes synapse_type as int 2=exc / 1=inh, so a fully
                # unclassified batch means the column is absent or renamed.
                if labels.get("unknown_syn", 0) == len(syn_df):
                    print("  WARNING: every synapse is unknown_syn -- check "
                          "the synapse CSV's type column name against "
                          "map_synapses_to_nodes_raw(type_column=...)")

        # Staging is INTERNAL: export + alignment go to a scratch dir, the gate
        # runs there, and OUTPUT_DIR is written only if the cell conducts.
        res = al.align_and_export(
            df_raw, nid, OUTPUT_DIR, metadata_df,
            cm=CM_BASE, Ra=RA, Rm=RM_QC, label_fn=label_fn,
            k_neighbors=K_NEIGHBORS, syn_df=syn_df,
            return_frames=True, verbose=True)

        qc = res.get("propagation_qc", {})
        if qc:
            print("  gate: %s  dv soma %.3f mV, min %.3g mV, %d segs" % (
                qc.get("qc_status"),
                qc.get("C4_soma", {}).get("dv_soma_mV", float("nan")),
                qc.get("dv_min_mV", float("nan")),
                qc.get("totnsegs", -1)))
            for r in qc.get("reasons", []):
                print("        %s" % r)

        if syn_df is not None and "n_synapses" in res:
            print("  redirect: %d synapses, %d on pruned spines, %d "
                  "redirected, %d unresolved bases, %d unclassified" % (
                      res["n_synapses"], res["n_on_pruned_spine"],
                      res["n_redirected"], res["n_unresolved_spine_bases"],
                      res.get("n_unknown_type", 0)))

        if res["qc_status"] == "fail":
            gated_out.append({"nid": nid, "reasons": res.get("reasons", [])})
            print("  NOT WRITTEN to %s" % OUTPUT_DIR)
            _log_run({"nid": nid, "event": "gated_out",
                      "reasons": res.get("reasons", [])})
            _durations.append(time.time() - t0)
            continue

        # ---- rigidity control, every cell ---------------------------------- #
        if RUN_RIGIDITY_CONTROL:
            ctl = mx.export_neuron(df_raw, "%s_unaligned" % nid, OUTPUT_DIR,
                                   label_fn=label_fn, align_fn=None,
                                   write_files=False, verbose=False)
            # COMPARE LIKE WITH LIKE. regression_check compares qc_status (it
            # is in al.REGRESSION_KEYS), but `ctl` is a bare export_neuron call
            # that never runs the propagation gate, while `res` has been
            # through it. Using res["qc_status"] here reports a spurious
            # "alignment moved a quantity" on every cell the GATE downgraded --
            # a false rigidity failure with identical geometry. The geometric
            # keys are all computed at export step 7, before align_fn is
            # applied at step 9, so they cannot differ; qc_status was the only
            # key that ever could.
            res_rigid = dict(res)
            res_rigid["qc_status"] = res.get("exporter_qc_status",
                                             res["qc_status"])
            chk = al.regression_check(ctl, res_rigid)
            res["rigidity_identical"] = bool(chk["identical"])
            print("  rigidity: %s" % ("IDENTICAL" if chk["identical"] else
                                      "*** MOVED *** %s" % chk["diffs"]))
            # Keep the six scalars rigidity() plots AND the diffs -- the diffs
            # are what make a failure diagnosable at the point it is raised,
            # instead of sending you back through 80 cells of scrollback.
            regression_records.append({
                "nid": nid,
                "unaligned": {k: ctl.get(k) for k in ap.REGRESSION_LABELS},
                "aligned": {k: res.get(k) for k in ap.REGRESSION_LABELS},
                "check": {"identical": bool(chk["identical"]),
                          "diffs": dict(chk["diffs"])}})
            del ctl, chk, res_rigid
            ctl = None

        # ---- structural validation of the committed .hoc ------------------- #
        # Moved INTO the loop (was CELL 6b): a cell must be fully judged before
        # its artefacts are released, or a quarantine decision would need the
        # record resurrected later.
        entry = res["files"]["hoc"]
        hoc_path = entry if isinstance(entry, str) else entry["path"]
        geo = audit_hoc_geometry(hoc_path)
        nrep = neuron_validate(hoc_path)
        viol = classify_violations(geo, nrep)
        hoc_verdict = _verdict(viol)
        res["hoc_validation"] = {"geometry": geo, "neuron": nrep,
                                 "violations": viol, "verdict": hoc_verdict}
        res["qc_status"] = _merge_hoc_verdict(res["qc_status"], hoc_verdict)
        print("  hoc: %s  %d sections, %d orphans, %.1f um cable, "
              "soma %.3f um" % (
                  hoc_verdict.upper(), geo["n_sections"], len(geo["orphans"]),
                  geo["total_length_um"], geo["soma_diameter_um"]))
        for level in (STRUCTURAL, SOFT):
            for msg in viol[level]:
                print("    [%s] %s" % (level, msg))

        if hoc_verdict == "fail":
            dest = _quarantine(OUTPUT_DIR, nid, res)
            quarantined.append(nid)
            failures.append({
                "nid": nid,
                "error": "hoc validation FAILED: %s" % "; ".join(
                    viol[STRUCTURAL])})
            _write_validation_json(dest, nid, res["hoc_validation"])
            print("  QUARANTINED -> %s (excluded from bank)" % dest)
            _log_run({"nid": nid, "event": "quarantined",
                      "reasons": viol[STRUCTURAL]})
            _durations.append(time.time() - t0)
            continue
        _write_validation_json(os.path.dirname(hoc_path), nid,
                               res["hoc_validation"])

        # ---- angle from +z (ALWAYS), then optional figures, then free ------ #
        # angle_from_z_deg is NOT a plotting by-product. It is the input to
        # ap.orientation_consistency, which its own docstring calls the acid
        # test of alignment across cells: a bank whose arbours do not cluster
        # near +z has an alignment problem no per-cell check can see. Computing
        # it only when figures were requested silently emptied that plot for
        # every figure-less run. The frame is transient either way -- it is
        # built, used, and freed inside this block -- so asking for it
        # unconditionally costs nothing in the streaming design.
        if "frames" in res:
            soma_pos = np.asarray(res["alignment"]["soma_pos_nm"], float)
            mean_matrix = np.asarray(res["alignment"]["mean_matrix"], float)
            df_lab = res.pop("frames")["labelled"]
            df_ali = al.make_align_fn(soma_pos, mean_matrix)(df_lab, nid)

            res["angle_from_z_deg"] = ap.angle_from_z_deg(
                ap.arbour_direction(df_ali))
            print("  aligned arbour is %.1f deg from +z" %
                  res["angle_from_z_deg"])

            if SAVE_PER_NEURON_FIGURES:
                _save_neuron_figures(nid, df_lab, df_ali, soma_pos,
                                     res["alignment"])

            del df_lab, df_ali, soma_pos, mean_matrix
            df_lab = df_ali = None
        else:
            # align_and_export withheld the frame -- angle stays absent rather
            # than being silently filled with a wrong value.
            res.pop("frames", None)
            print("  WARNING: no labelled frame returned; angle_from_z_deg "
                  "unavailable for this cell")

        # ---- persist the summary row, keep only scalars -------------------- #
        rec = _compact(res)
        _append_summary_row(rec)
        records.append(rec)

        dt = time.time() - t0
        _durations.append(dt)
        print("  done in %.1f s" % dt)
        _log_run({"nid": nid, "event": "processed",
                  "qc_status": rec.get("qc_status"), "duration_s": dt})
        del rec
    except Exception as exc:                                 # noqa: BLE001
        failures.append({"nid": nid, "error": repr(exc)})
        print("  FAILED: %r" % exc)
        traceback.print_exc()
        _log_run({"nid": nid, "event": "error", "error": repr(exc)})
    finally:
        # THE point of the streaming design: whatever happened above -- success,
        # gate failure, quarantine or exception -- every large object this
        # iteration created is released before the next neuron is read. Peak
        # RAM is one neuron's worth, not the bank's.
        for _name in ("df_raw", "syn_raw", "syn_df", "res", "ctl",
                      "df_lab", "df_ali", "geo", "nrep", "viol", "labels"):
            globals().pop(_name, None)
        gc.collect()

print("\n%d in hand (%d resumed, %d freshly processed), %d gated out, "
      "%d quarantined, %d errored" % (
          len(records), n_resumed, len(records) - n_resumed, len(gated_out),
          len(quarantined), len(failures)))
for g in gated_out:
    print("  gated out %s: %s" % (g["nid"], "; ".join(g["reasons"])))
if quarantined:
    print("  quarantined: %s" % quarantined)
if not records:
    raise RuntimeError("no cell survived -- nothing to aggregate")
print("\nbank summary: %s" % BANK_SUMMARY_CSV)

# %% CELL 8 -- batch plots --------------------------------------------------
# Built from the COMPACT scalar records (and, for rigidity, the six scalars per
# cell kept in CELL 6). No morphology is reloaded and nothing large was held to
# get here -- alignment_plots' three batch functions read only scalars.
if records:
    fig = ap.batch_quality(records)
    saved.append(ap.save_figure(fig, "%s/batch_quality.png" % FIGURE_DIR))
    del fig

    # Guard the exact failure this plot had: every angle NaN produces a valid
    # but EMPTY figure and no error anywhere. Say so loudly instead.
    n_ang = sum(1 for r in records if r.get("angle_from_z_deg") is not None
                and not pd.isna(r.get("angle_from_z_deg")))
    if n_ang == 0:
        print("WARNING: angle_from_z_deg missing on ALL %d records -- the "
              "orientation plot would be empty. Every cell here was resumed "
              "from a checkpoint written before this was computed; delete "
              "those neuron_*_alignment.json files to recompute." % len(
                  records))
    else:
        if n_ang < len(records):
            print("note: angle_from_z_deg present on %d/%d records (the rest "
                  "were resumed from older checkpoints)" % (n_ang, len(records)))
        fig = ap.orientation_consistency(records)
        saved.append(ap.save_figure(fig,
                                    "%s/batch_orientation.png" % FIGURE_DIR))
        del fig

if regression_records:
    fig = ap.rigidity(regression_records)
    saved.append(ap.save_figure(fig, "%s/batch_rigidity.png" % FIGURE_DIR))
    del fig
    bad = [r for r in regression_records if not r["check"]["identical"]]
    if bad:
        detail = "\n".join(
            "  %s: %s" % (r["nid"], r["check"].get("diffs") or "(not recorded)")
            for r in bad)
        raise AssertionError(
            "alignment moved a section 7 quantity on %d cell(s) -- the "
            "transform is not rigid or was applied at the wrong step. Do not "
            "use this bank.\n%s" % (len(bad), detail))
    print("rigidity: every cell bit-identical (%d cells)" % len(
        regression_records))


gc.collect()
print("\n%d figures written to %s" % (len(saved), FIGURE_DIR))

# %% CELL 9 -- bank summary and manifest ------------------------------------
# The summary is READ from BANK_SUMMARY_CSV, not rebuilt from memory: CELL 6
# appended a row per neuron as it went, so this file is complete even for a
# bank assembled across several sessions after a disconnect. The in-memory
# `records` list only covers the CURRENT session.
summary = pd.read_csv(BANK_SUMMARY_CSV)
# A resumed cell re-appends its row; keep the most recent per neuron.
n_dupes = int(summary.duplicated(subset=["nid"], keep="last").sum())
if n_dupes:
    summary = summary.drop_duplicates(subset=["nid"], keep="last")
    summary.to_csv(BANK_SUMMARY_CSV, index=False)
    print("collapsed %d duplicate row(s) from re-processed neurons" % n_dupes)

print("bank: %d neurons in %s" % (len(summary), BANK_SUMMARY_CSV))
print(summary["qc_status"].value_counts().to_string())
if "hoc_verdict" in summary.columns:
    print(summary["hoc_verdict"].value_counts().to_string())

with pd.option_context("display.max_rows", 200, "display.width", 250):
    print(summary[["nid", "qc_status", "n_sections", "f_implied", "F_lit",
                   "angle_from_z_deg", "gate_dv_soma_mV", "n_synapses",
                   "n_redirected", "soma_diameter_um"]].to_string(index=False))

# Bank-level statistics -- the numbers that only exist at bank scale.
_num = lambda c: pd.to_numeric(summary[c], errors="coerce") if c in summary else pd.Series(dtype=float)
_f, _F = _num("f_implied"), _num("F_lit")
_deficit = ((_f - _F) / _F * 100.0).dropna()
bank_stats = {
    "n_neurons": int(len(summary)),
    "qc_status_counts": summary["qc_status"].value_counts().to_dict(),
    "f_implied": {"median": float(_f.median()), "mean": float(_f.mean()),
                  "min": float(_f.min()), "max": float(_f.max())},
    "f_implied_vs_F_lit_pct": {
        "median": float(_deficit.median()) if len(_deficit) else None,
        "min": float(_deficit.min()) if len(_deficit) else None,
        "max": float(_deficit.max()) if len(_deficit) else None,
        "n_below_lit": int((_deficit < 0).sum()),
        "n_above_lit": int((_deficit > 0).sum())},
    "angle_from_z_deg": {
        "n_measured": int(_num("angle_from_z_deg").notna().sum()),
        "median": float(_num("angle_from_z_deg").median()),
        "p90": float(_num("angle_from_z_deg").quantile(0.9))},
    "n_synapses_total": int(_num("n_synapses").sum()),
    "n_redirected_total": int(_num("n_redirected").sum()),
    "n_unknown_type_total": int(_num("n_unknown_type").sum()),
    "soma_diameter_um": {"median": float(_num("soma_diameter_um").median()),
                         "min": float(_num("soma_diameter_um").min()),
                         "max": float(_num("soma_diameter_um").max())},
}
print("\nBANK STATISTICS")
print(json.dumps(bank_stats, indent=2, default=str))
if _deficit.median() is not None and len(_deficit):
    print("\nf_implied vs F_lit: %d/%d cells BELOW literature, "
          "median %+.1f%%" % (
              bank_stats["f_implied_vs_F_lit_pct"]["n_below_lit"],
              len(_deficit), _deficit.median()))
    print("  A one-directional deficit across most of the bank points at spine"
          " detection or the pruning threshold, not shaft geometry -- S1.7.")

manifest = {
    "module_versions": {"alignment": al.MODULE_VERSION,
                        "alignment_plots": ap.MODULE_VERSION,
                        "morphology_exporter": mx.MODULE_VERSION,
                        "hoc_qc": hq.MODULE_VERSION,
                        "node_classify": nc.MODULE_VERSION,
                        "synapse_redirect_audit": sra.MODULE_VERSION},
    "metadata_csv": METADATA_CSV, "n_references": int(len(metadata_df)),
    "k_neighbors": K_NEIGHBORS,
    "segmentation": {"cm": CM_BASE, "Ra": RA, "lambda_f": 100.0,
                     "d_lambda": 0.1, "nsegs_method": "lambda_f"},
    "propagation_gate": {"Rm_ohm_cm2": RM_QC, "g_pas_S_cm2": 1.0 / RM_QC,
                         "amp_nA": hq.DEFAULT_AMP_NA,
                         "dur_ms": hq.DEFAULT_DUR_MS,
                         "dt_ms": hq.DEFAULT_DT_MS,
                         "monotone_rtol": hq.MONOTONE_RTOL,
                         "soma_window_mV": [hq.SOMA_DV_MIN_MV,
                                            hq.SOMA_DV_MAX_MV]},
    "domain_split": "retired -- every dendrite is 'dend'",
    "synapse_redirect": {"enabled": RUN_SYNAPSE_REDIRECT,
                         "synapses_dir": SYNAPSES_DIR,
                         "direction": SYNAPSE_DIRECTION,
                         "voxel_res_nm": list(VOXEL_RES_NM)},
    "execution": {"streaming": True, "checkpoint": CHECKPOINT,
                  "rigidity_control": RUN_RIGIDITY_CONTROL,
                  "per_neuron_figures": SAVE_PER_NEURON_FIGURES,
                  "n_requested": len(NEURON_IDS),
                  "n_resumed_this_session": n_resumed},
    "bank_statistics": bank_stats,
    "neuron_ids": list(NEURON_IDS), "n_aligned": int(len(summary)),
    "gated_out": gated_out, "quarantined": quarantined,
    "failures": failures, "figures": saved,
    "summary_csv": BANK_SUMMARY_CSV, "run_log": RUN_LOG_PATH,
}
with open("%s/alignment_manifest.json" % FIGURE_DIR, "w", newline="\n") as fh:
    fh.write(json.dumps(manifest, indent=2, sort_keys=True, default=str))

if failures:
    print("\nFAILURES:")
    for f in failures:
        print("  %s: %s" % (f["nid"], f["error"]))
print("\nmanifest: %s/alignment_manifest.json" % FIGURE_DIR)
