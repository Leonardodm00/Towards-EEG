# Colab batch driver for stage S1 alignment.
#
# Aligns a list of neuron ids YOU choose against a metadata bank YOU point at,
# verifies rigidity against an unaligned control run, and produces the full
# plot repertoire -- interactive plotly displayed inline AND every figure saved.
#
# Paste CELL BY CELL. Boundaries are marked '# %% CELL N'.
# Only CELL 4 needs editing.
#
#   1  upload alignment.py, alignment_plots.py and their smoke tests
#   2  mount Drive, install plotly, locate or upload the S1 dependency modules
#   3  run BOTH smoke tests -- hard stop unless 11/11 and 12/12
#   4  EDIT: neuron ids, metadata path, directories, cm/Ra
#   5  imports + load the metadata bank
#   6  per-neuron alignment, with an unaligned control for the rigidity check
#   6b validate every emitted .hoc (I-13a/I-15/I-16/I-18) and promote;
#      a failing cell is quarantined and never reaches the plots
#   7  per-neuron plots: interactive arbour, static arbour, depth, neighbourhood
#   8  batch plots: quality, orientation consistency, rigidity
#   9  summary table + manifest, written next to the figures
#
# CELL 3 must run AFTER cell 2, not before: smoke_alignment.py imports
# node_classify, soma_enforce and morphology_exporter directly, so the smoke
# test cannot even start until those modules are on the path.
#
# One deliberate cost: CELL 6 exports every neuron TWICE, once unaligned and
# once aligned. That doubles the runtime (about 40 s per pass on a 63k-node
# cell) and it is the only way to prove the transform moved nothing it should
# not have. Set RUN_RIGIDITY_CONTROL = False to skip it once you trust a bank.


# %% CELL 1 -- upload the modules delivered with this driver
from google.colab import files

uploaded = files.upload()
for needed in ("alignment.py", "alignment_plots.py",
               "smoke_alignment.py", "smoke_alignment_plots.py"):
    if needed not in uploaded:
        raise FileNotFoundError(
            "expected %r -- select all four files together" % needed)
print("uploaded:", sorted(uploaded))


# %% CELL 2 -- environment and the S1 dependency modules
# These are NOT the 4 files above. alignment.py needs the full S1 chain
# (spine_density, node_classify, soma_enforce, morphology_exporter,
# spine_labeller), which per TEEG_16 section 8 is "still loose files" and is
# usually NOT on Drive on a fresh session. This cell finds them if they are on
# Drive, or prompts you to upload them.
import importlib
import os
import sys

from google.colab import drive
drive.mount('/content/drive')
get_ipython().system('pip install -q plotly neuron LFPy')   # noqa: F821
# neuron and LFPy are REQUIRED by CELL 6b: without them mx.validate_hoc
# cannot run and I-13a, I-15 and I-16 go unchecked.

if "" not in sys.path:
    sys.path.insert(0, "")

NEEDED = ["spine_density.py", "node_classify.py", "soma_enforce.py",
          "morphology_exporter.py", "spine_labeller.py"]


def _have(fname):
    return any(os.path.isfile(os.path.join(d, fname)) for d in sys.path)


missing = [f for f in NEEDED if not _have(f)]
if missing:
    print("searching Drive for:", missing)
    hits = {}
    for root, dirs, fnames in os.walk("/content/drive/MyDrive"):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in list(missing):
            if f in fnames:
                hits.setdefault(f, os.path.join(root, f))
        if len(hits) == len(missing):
            break
    for f, p in hits.items():
        d = os.path.dirname(p)
        if d not in sys.path:
            sys.path.insert(0, d)
        print("  found %-26s %s" % (f, d))
    missing = [f for f in NEEDED if not _have(f)]

if missing:
    print("\nNOT on Drive -- select these in the dialog:", missing)
    up = files.upload()
    still = [f for f in NEEDED if f not in up and not _have(f)]
    if still:
        raise FileNotFoundError("still missing: %s" % still)

for f in NEEDED:
    m = importlib.import_module(f[:-3])
    print("OK  %-26s %s" % (f, getattr(m, "MODULE_VERSION", "")))
print("\nall S1 dependency modules importable")


# %% CELL 3 -- both smoke tests, now that the dependencies are on the path
import subprocess

# subprocess.run spawns a FRESH interpreter, which does not inherit sys.path
# entries added at runtime in this kernel (only PYTHONPATH and the script's own
# directory). Forward the live sys.path explicitly, or a smoke test subprocess
# cannot see modules that were located on Drive rather than uploaded into
# /content -- this is exactly the failure that looks like a missing package
# but is actually a search-path mismatch between kernel and subprocess.
_env = dict(os.environ)
_env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p) + (
    os.pathsep + _env["PYTHONPATH"] if _env.get("PYTHONPATH") else "")

for script, n_expect in (("smoke_alignment.py", 11),
                         ("smoke_alignment_plots.py", 12)):
    r = subprocess.run([sys.executable, script], capture_output=True,
                       text=True, env=_env)
    print(r.stdout)
    expect = "%d / %d" % (n_expect, n_expect)
    if r.returncode != 0 or expect not in r.stdout:
        print(r.stderr)
        raise RuntimeError("%s did not pass %s -- stop" % (script, expect))
print("both smoke suites green")


# %% CELL 4 -- EDIT THIS CELL ------------------------------------------------
# The neurons to align. Every id must have neuron_{id}.csv in SKELETONS_DIR.
NEURON_IDS = [794820508, 606394351]

# The reference bank for the (layer, subpopulation) these cells belong to.
# Membership decides this, NOT spatial proximity: an L2 exc cell uses the L2
# exc bank even if an L5 bank happens to sit nearer in the volume.
METADATA_CSV = "/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons/alignment_metadata_L2.csv"

SKELETONS_DIR = "/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons"
OUTPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/Aligned Neurons HOC"
FIGURE_DIR = OUTPUT_DIR + "/figures"

K_NEIGHBORS = 3          # local alignment, settled 25 July 2026

# Segmentation parameters. These define the lfpy_idx index space and are
# stamped into provenance. Eyal 2016 for human L2/3 pyramidal; for INTERNEURONS
# use alignment.RA_YAO_INTERNEURON (100.0) with the subtype's cm instead.
CM_BASE = 0.50           # uF/cm2, intrinsic and shaft-referenced (contract C-14)
RA = 268.5               # ohm cm

RUN_RIGIDITY_CONTROL = True    # export unaligned too, to prove rigidity
MAX_SEGMENTS_INTERACTIVE = 30000
MAX_SEGMENTS_STATIC = 20000
SHOW_INLINE = True


# %% CELL 5 -- imports + load the metadata bank
import numpy as np
import pandas as pd
import morphology_exporter as mx
import spine_labeller as sl
import node_classify as nc
import alignment as al
import alignment_plots as ap

os.makedirs(FIGURE_DIR, exist_ok=True)
metadata_df = al.load_alignment_metadata(METADATA_CSV)
print("bank %s: %d references" % (os.path.basename(METADATA_CSV),
                                  len(metadata_df)))


# %% CELL 6 -- align every neuron, with the unaligned control
import time
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


records, regression_records, frames, failures = [], [], {}, []

for nid in NEURON_IDS:
    print("\n" + "=" * 70)
    print("neuron %s" % nid)
    t0 = time.time()
    try:
        path = "%s/neuron_%s.csv" % (SKELETONS_DIR, nid)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        df_raw = pd.read_csv(path)
        label_fn = make_label_fn(nid)

        res = al.align_and_export(df_raw, nid, OUTPUT_DIR, metadata_df,
                                  cm=CM_BASE, Ra=RA, label_fn=label_fn,
                                  k_neighbors=K_NEIGHBORS, verbose=True)

        if RUN_RIGIDITY_CONTROL:
            ctl = mx.export_neuron(df_raw, "%s_unaligned" % nid, OUTPUT_DIR,
                                   label_fn=label_fn, align_fn=None,
                                   write_files=False, verbose=False)
            chk = al.regression_check(ctl, res)
            print("  rigidity: %s" % ("IDENTICAL" if chk["identical"]
                                      else "*** MOVED *** %s" % chk["diffs"]))
            regression_records.append({"nid": nid, "unaligned": ctl,
                                       "aligned": res, "check": chk})

        # the aligned frame, reread from the emitted hoc's source of truth:
        # export_neuron does not return it, so rebuild via the same align_fn
        soma_pos = np.asarray(res["alignment"]["soma_pos_nm"], float)
        mean_matrix = np.asarray(res["alignment"]["mean_matrix"], float)
        df_cls = nc.classify_frame(df_raw)
        df_ali = al.make_align_fn(soma_pos, mean_matrix)(df_cls, nid)

        v = ap.arbour_direction(df_ali)
        res["angle_from_z_deg"] = ap.angle_from_z_deg(v)
        print("  aligned arbour is %.1f deg from +z" % res["angle_from_z_deg"])

        frames[nid] = {"raw": df_cls, "aligned": df_ali,
                       "soma_pos": soma_pos, "diag": res["alignment"]}
        records.append(res)
        print("  done in %.1f s" % (time.time() - t0))
    except Exception as exc:                            # noqa: BLE001
        failures.append({"nid": nid, "error": repr(exc)})
        print("  FAILED: %r" % exc)
        traceback.print_exc()

print("\n%d aligned, %d failed" % (len(records), len(failures)))


# %% CELL 6b -- VALIDATE THE EMITTED HOC, THEN PROMOTE ----------------------
# COMPLETE CELL. Paste this as the whole of CELL 6b, replacing anything that
# is there. Runs between CELL 6 and CELL 7. Nothing to splice.
#
# TEEG_16 exit invariants I-13a, I-15, I-16 and I-18 live in
# morphology_exporter.validate_hoc, which until now was called from ONE place:
# test E8, which skips when NEURON is absent. No aligned .hoc had ever been
# validated. This cell closes that.
#
# TWO LAYERS, because they fail differently:
#   audit_hoc_geometry   pure text and geometry, ALWAYS runs, no NEURON needed.
#                        Covers the connectivity half of I-15 (a detached
#                        fragment always contributes its own parentless root)
#                        plus the degenerate geometry validate_hoc does not
#                        check -- a zero-length section makes lambda_f divide
#                        by L, giving a NaN nseg and a silently corrupt
#                        lfpy_idx space.
#   mx.validate_hoc      the NEURON-level assertions, when neuron imports.
#                        If this reports NOT AVAILABLE, add neuron and LFPy to
#                        CELL 2's pip line -- I-13a, I-15 and I-16 go unchecked
#                        without it.
#
# VERDICTS follow the Q6 ladder, not validate_hoc's blunt ok=True/False:
#   STRUCTURAL -> fail                 wrong as a NEURON model
#   SOFT       -> pass_low_confidence  I-18 disagreement, soma below floor
# soma_enforce already emits those same two as SOFT reasons, so hard-failing
# them here would make the two layers contradict each other on the same cell.
#
# WHERE THE FILES ARE is read from res["files"], never from STAGING_DIR. This
# cell works whether or not CELL 6 staged. Staging only changes WHEN the check
# bites: staged means a failing cell never reaches Drive; unstaged means it is
# MOVED OUT of OUTPUT_DIR into quarantine. Either way it does not survive in
# your results directory, and it is dropped from records, frames,
# regression_records and therefore from the plots, summary and manifest.

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

STRUCTURAL, SOFT = "structural", "soft"
MIN_SOMA_DIAM_UM = 4.0


def audit_hoc_geometry(path):
    """Structural audit of an emitted .hoc. No NEURON."""
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
        # numeric regex would simply FAIL TO MATCH 'pt3dadd(nan, ...)' and drop
        # the point silently, which is worse than a loud failure.
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
            "largest_diameter_um": float(diam_max[biggest]) if biggest else float("nan"),
            "soma_diameter_um": float(diam_max.get("soma[0]", float("nan")))}


def classify_violations(geo, neuron_rep=None, min_soma_diam_um=MIN_SOMA_DIAM_UM):
    v = {STRUCTURAL: [], SOFT: []}

    if len(geo["roots"]) != 1:
        v[STRUCTURAL].append("I-15: %d root sections %r"
                             % (len(geo["roots"]), geo["roots"][:6]))
    if len(geo["somatic_sections"]) != 1:
        v[STRUCTURAL].append("I-15: %d somatic sections"
                             % len(geo["somatic_sections"]))
    if geo["roots"] and geo["somatic_sections"] and \
            geo["roots"][0] != geo["somatic_sections"][0]:
        v[STRUCTURAL].append("I-15: root %s is not the soma" % geo["roots"][0])
    if geo["orphans"]:
        v[STRUCTURAL].append("debris: %d of %d sections unreachable from the "
                             "soma, e.g. %r"
                             % (len(geo["orphans"]), geo["n_sections"],
                                geo["orphans"][:5]))
    if geo["n_connect"] != geo["n_sections"] - 1:
        v[STRUCTURAL].append("not a tree: %d connect statements for %d sections"
                             % (geo["n_connect"], geo["n_sections"]))
    for key, msg in (("zero_length", "zero-length sections (NaN nseg)"),
                     ("nonpositive_diam", "non-positive diameters"),
                     ("few_pt3d", "sections with fewer than 2 pt3d points"),
                     ("nonfinite", "non-finite coordinates"),
                     ("unparsable_pt3d", "unparsable pt3dadd coordinates"),
                     ("dropped_pt3d", "pt3d points lost during parsing")):
        if geo[key]:
            v[STRUCTURAL].append("%s: %d %r" % (msg, len(geo[key]), geo[key][:5]))

    if geo["largest_diameter_section"] != "soma[0]":
        v[SOFT].append("I-18: largest-calibre section is %s (%.3f um), not the "
                       "soma (%.3f um)" % (geo["largest_diameter_section"],
                                           geo["largest_diameter_um"],
                                           geo["soma_diameter_um"]))
    if not (geo["soma_diameter_um"] >= float(min_soma_diam_um)):
        v[SOFT].append("I-18: soma diameter %.3f um is below the %.2f um floor"
                       % (geo["soma_diameter_um"], float(min_soma_diam_um)))

    if neuron_rep is not None and neuron_rep.get("neuron_available"):
        if neuron_rep.get("I16_no_synapse_sections") is False:
            v[STRUCTURAL].append("I-16: synapse label in the section vocabulary")
        if neuron_rep.get("I13a_vocabulary_admissible") is False:
            v[STRUCTURAL].append("I-13a: inadmissible arrays %r"
                                 % neuron_rep.get("arrays"))
        if neuron_rep.get("I15_single_soma_is_root") is False:
            v[STRUCTURAL].append("I-15 (NEURON): roots=%r"
                                 % neuron_rep.get("root_sections"))
        n_sec = neuron_rep.get("n_sections")
        if n_sec is not None and n_sec != geo["n_sections"]:
            v[STRUCTURAL].append("NEURON loaded %d sections, the file declares %d"
                                 % (n_sec, geo["n_sections"]))
    return v


def _verdict(v):
    if v[STRUCTURAL]:
        return "fail"
    return "pass_low_confidence" if v[SOFT] else "pass"


def _merge_qc(existing, hoc_verdict):
    """The emitted file can only lower confidence, never raise it."""
    order = {"pass": 0, "pass_low_confidence": 1, "fail": 2}
    return max([existing, hoc_verdict], key=lambda s: order.get(s, 2))


# --------------------------------------------------------------------------- #
# NEURON is loaded in a SUBPROCESS. h.load_file() is global and cumulative:
# validating several cells in one kernel leaves earlier sections in h.allsec(),
# so the second cell would report the sum of both and fail on n_sections. An
# isolated interpreter per file is the only clean way, and it also means a
# NEURON segfault cannot take the notebook down.
_NEURON_SNIPPET = r'''
import json, sys
sys.path[:0] = %r
import morphology_exporter as mx
print("@@@" + json.dumps(mx.validate_hoc(%r, min_soma_diam_um=%r), default=str))
'''


def neuron_validate(hoc_path, timeout=180):
    import subprocess
    code = _NEURON_SNIPPET % ([p for p in sys.path if p], str(hoc_path),
                              float(MIN_SOMA_DIAM_UM))
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=timeout)
        for line in r.stdout.splitlines():
            if line.startswith("@@@"):
                return json.loads(line[3:])
        return {"neuron_available": False, "import_error": (r.stderr or "")[-300:]}
    except Exception as exc:                                # noqa: BLE001
        return {"neuron_available": False, "import_error": repr(exc)}


# --- driver --------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
QUARANTINE_DIR = os.path.join(OUTPUT_DIR, "_quarantine")

promoted, quarantined = [], []

for rec in list(records):
    nid = rec["nid"]

    # res["files"] is NOT homogeneous: write_hoc returns a dict, the CSV
    # entries are plain strings.
    entry = rec["files"]["hoc"]
    hoc_path = entry if isinstance(entry, str) else entry["path"]
    if not os.path.isfile(hoc_path):
        raise FileNotFoundError(
            "the recorded .hoc for %s does not exist: %r -- CELL 6 did not "
            "write where it says it did" % (nid, hoc_path))
    src_dir = os.path.dirname(hoc_path)
    staged = os.path.abspath(src_dir) != os.path.abspath(OUTPUT_DIR)

    geo = audit_hoc_geometry(hoc_path)
    nrep = neuron_validate(hoc_path)
    viol = classify_violations(geo, nrep)
    hoc_verdict = _verdict(viol)

    print("\n" + "-" * 70)
    print("neuron %s  ->  %s   [%s]"
          % (nid, hoc_verdict.upper(),
             "staged, nothing on Drive yet" if staged
             else "NOT staged, file already in OUTPUT_DIR"))
    print("  source          %s" % src_dir)
    print("  sections %d | connect %d | roots %s"
          % (geo["n_sections"], geo["n_connect"], geo["roots"]))
    print("  reachable from soma %d/%d | orphans %d | cable %.1f um"
          % (geo["n_reachable"], geo["n_sections"], len(geo["orphans"]),
             geo["total_length_um"]))
    print("  largest calibre %s %.3f um | soma %.3f um"
          % (geo["largest_diameter_section"], geo["largest_diameter_um"],
             geo["soma_diameter_um"]))
    print("  NEURON: %s"
          % ("ok=%s, %d sections" % (nrep.get("ok"), nrep.get("n_sections"))
             if nrep.get("neuron_available")
             else "NOT AVAILABLE -- geometry layer only"))
    for kind in (STRUCTURAL, SOFT):
        for m in viol[kind]:
            print("  [%-10s] %s" % (kind, m))

    rec["hoc_validation"] = {"verdict": hoc_verdict, "geometry": geo,
                             "neuron": nrep, "violations": viol,
                             "staged": staged, "source_dir": src_dir}
    before = rec["qc_status"]
    rec["qc_status"] = _merge_qc(before, hoc_verdict)
    if rec["qc_status"] != before:
        print("  qc_status %s -> %s" % (before, rec["qc_status"]))
        rec.setdefault("reasons", []).extend(
            ["hoc:" + m for m in viol[STRUCTURAL] + viol[SOFT]])

    # Collect this neuron's artefacts from wherever they actually are.
    own = sorted(f for f in os.listdir(src_dir)
                 if f.startswith("neuron_%s" % nid)
                 and os.path.isfile(os.path.join(src_dir, f)))

    if hoc_verdict == "fail":
        os.makedirs(QUARANTINE_DIR, exist_ok=True)
        for f in own:
            # MOVE, not copy: a failing cell must not remain in OUTPUT_DIR.
            shutil.move(os.path.join(src_dir, f),
                        os.path.join(QUARANTINE_DIR, f))
        dest = QUARANTINE_DIR
        quarantined.append(nid)
        records.remove(rec)
        frames.pop(nid, None)
        regression_records[:] = [r for r in regression_records
                                 if r["nid"] != nid]
        failures.append({"nid": nid,
                         "error": "hoc validation FAILED: %s"
                                  % "; ".join(viol[STRUCTURAL])})
        print("  QUARANTINED -> %s" % QUARANTINE_DIR)
        print("  excluded from plots, summary and manifest")
    else:
        dest = OUTPUT_DIR
        if staged:
            for f in own:
                shutil.copy2(os.path.join(src_dir, f),
                             os.path.join(OUTPUT_DIR, f))
            print("  promoted %d files -> %s" % (len(own), OUTPUT_DIR))
        else:
            print("  %d files already in %s, left in place"
                  % (len(own), OUTPUT_DIR))
        promoted.append(nid)

    # Repoint the recorded paths at wherever the files ended up, so cells 7-9
    # keep working.
    for k, val in list(rec["files"].items()):
        p = val if isinstance(val, str) else (val.get("path")
                                              if isinstance(val, dict) else None)
        if p and os.path.dirname(p) == src_dir:
            newp = os.path.join(dest, os.path.basename(p))
            if isinstance(val, str):
                rec["files"][k] = newp
            else:
                val["path"] = newp

    with open(os.path.join(dest, "neuron_%s_hoc_validation.json" % nid),
              "w", newline="\n") as fh:
        fh.write(json.dumps(rec["hoc_validation"], indent=2, sort_keys=True,
                            default=str))

print("\n" + "=" * 70)
print("promoted   %d: %s" % (len(promoted), promoted))
print("quarantined %d: %s" % (len(quarantined), quarantined))
if quarantined:
    print("\nquarantined artefacts are in %s" % QUARANTINE_DIR)
if not promoted:
    raise RuntimeError("no cell survived hoc validation -- nothing to plot")


# %% CELL 7 -- per-neuron plots
from IPython.display import display

saved = []
for nid, fr in frames.items():
    print("\nplots for %s" % nid)

    fig = ap.arbour_comparison(fr["raw"], fr["aligned"], nid,
                               raw_units="nm", aligned_units="nm",
                               max_segments=MAX_SEGMENTS_INTERACTIVE)
    saved.append(ap.save_figure(fig, "%s/neuron_%s_arbour.html"
                                % (FIGURE_DIR, nid)))
    if SHOW_INLINE:
        display(fig)

    fig = ap.arbour_static(fr["raw"], fr["aligned"], nid,
                           max_segments=MAX_SEGMENTS_STATIC)
    saved.append(ap.save_figure(fig, "%s/neuron_%s_arbour.png"
                                % (FIGURE_DIR, nid)))

    fig = ap.depth_profile(fr["aligned"], nid, units="nm")
    saved.append(ap.save_figure(fig, "%s/neuron_%s_depth.png"
                                % (FIGURE_DIR, nid)))

    fig = ap.neighbourhood(fr["soma_pos"], metadata_df, fr["diag"], nid)
    saved.append(ap.save_figure(fig, "%s/neuron_%s_neighbourhood.png"
                                % (FIGURE_DIR, nid)))
    print("  4 figures saved")


# %% CELL 8 -- batch plots
if records:
    fig = ap.batch_quality(records)
    saved.append(ap.save_figure(fig, "%s/batch_quality.png" % FIGURE_DIR))

    fig = ap.orientation_consistency(records)
    saved.append(ap.save_figure(fig, "%s/batch_orientation.png" % FIGURE_DIR))

if regression_records:
    fig = ap.rigidity(regression_records)
    saved.append(ap.save_figure(fig, "%s/batch_rigidity.png" % FIGURE_DIR))
    bad = [r["nid"] for r in regression_records if not r["check"]["identical"]]
    if bad:
        raise AssertionError(
            "alignment moved a section 7 quantity on %s -- the transform is not "
            "rigid or was applied at the wrong step. Do not use this bank." % bad)
    print("rigidity: every cell bit-identical")

print("\n%d figures written to %s" % (len(saved), FIGURE_DIR))


# %% CELL 9 -- summary table and manifest
import json

rows = []
for r in records:
    d = r.get("alignment", {})
    nd = d.get("neighbour_distance_um", [np.nan])
    rows.append({
        "nid": r["nid"], "qc_status": r["qc_status"],
        "reasons": ";".join(r.get("reasons", [])),
        "n_sections": r.get("n_sections"), "n_branches": r.get("n_branches"),
        "f_implied": r.get("f_implied"), "F_lit": r.get("F_lit"),
        "nearest_ref_um": round(float(np.min(nd)), 1),
        "pairwise_spread_deg": round(float(d.get("pairwise_angle_deg_max", np.nan)), 1),
        "angle_from_z_deg": round(float(r.get("angle_from_z_deg", np.nan)), 1),
        "totnsegs": r.get("totnsegs"),
        "cm": CM_BASE, "Ra": RA, "k_neighbors": K_NEIGHBORS,
    })

summary = pd.DataFrame(rows)
summary_path = "%s/alignment_summary.csv" % FIGURE_DIR
summary.to_csv(summary_path, index=False)
print(summary.to_string(index=False))

manifest = {
    "module_versions": {"alignment": al.MODULE_VERSION,
                        "alignment_plots": ap.MODULE_VERSION,
                        "morphology_exporter": mx.MODULE_VERSION},
    "metadata_csv": METADATA_CSV, "n_references": int(len(metadata_df)),
    "k_neighbors": K_NEIGHBORS,
    "segmentation": {"cm": CM_BASE, "Ra": RA, "lambda_f": 100.0,
                     "d_lambda": 0.1, "nsegs_method": "lambda_f"},
    "neuron_ids": list(NEURON_IDS), "n_aligned": len(records),
    "failures": failures, "figures": saved, "summary_csv": summary_path,
}
with open("%s/alignment_manifest.json" % FIGURE_DIR, "w", newline="\n") as fh:
    fh.write(json.dumps(manifest, indent=2, sort_keys=True, default=str))

if failures:
    print("\nFAILURES:")
    for f in failures:
        print("  %s: %s" % (f["nid"], f["error"]))
print("\nmanifest: %s/alignment_manifest.json" % FIGURE_DIR)
