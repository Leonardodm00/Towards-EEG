#!/usr/bin/env python3
"""P1: partition + export + gate one cell, headless, one PBS array task per cell.

    python3 run_p1_export.py --root ROOT --manifest M.csv --task $PBS_ARRAY_INDEX
    python3 run_p1_export.py --root ROOT --manifest M.csv --cell 1302789404 --dry-run
    python3 run_p1_export.py --root ROOT --summarise            # p1_summary.csv

A thin wrapper over `alignment.align_and_export`, which already IS the whole
single-cell pipeline: export steps 1-8 (including step 4b, the three-vote
demotion of shaft continuations), alignment at step 9, the propagation gate at
step 12 with staging committed only on a pass, the synapse snapper and the
C-09 emitter at step 14. Nothing here re-implements any of it. This module
supplies what the Colab notebook supplies by hand and a cluster cannot:

  WHICH CELL       a manifest row selected by array index, so N cells are N
                   tasks and the cell list is data, not code (build it with
                   build_p1_manifest.py);
  WHICH CONSTANTS  cm, Ra and the gate's Rm resolved from the cell's (layer,
                   cell_type) in passive_params.csv -- align_and_export
                   refuses to default them because they fix the lfpy_idx
                   index space, and they differ by population;
  WHAT ELSE        the rigidity control, the structural .hoc audit with
                   quarantine, the arbour angle, the per-spine tables of
                   handoff section 4.1, a parameter fingerprint and one JSON
                   record per cell; `--summarise` folds the records into
                   p1_summary.csv (array tasks never share a file).

STANDING DECISIONS BAKED IN (handoff section 6, decisions log):
  demote_continuations=True always, thresholds at the defaults and in the
  fingerprint; cap_tips=True, cap_h_um=0.1 (shaft-side cap kept; the
  notebook's CAP_TIPS=False is superseded); propagation gate ON; synapse
  redirect ON when the synapse CSV exists; cm shaft-referenced, F never
  folded in (C-14, I-17).

OUTPUT, per cell, in <out>/<cell_id>/:
  neuron_<id>_aligned.hoc, _phi.csv, _segment_map.csv, _section_table.csv,
  _spine_bases.csv, _synapses.csv, _provenance.json, _alignment.json,
  _mapped_synapses.csv                         (align_and_export)
  neuron_<id>_spine_nodes.csv, _spine_stats.csv (section 4.1, p1_spine_stats)
  neuron_<id>_hoc_validation.json               (p1_hoc_audit)
  neuron_<id>_p1.json                           (this driver's record)
A cell the gate rejects gets only the _p1.json (align_and_export writes
nothing else on a fail). A cell the structural audit rejects is moved whole
into <out>/<cell_id>/_quarantine/.

SEPARATION OF CONCERNS
    manifest / passive table I+O   load_manifest, select_row,
                                   load_passive_table, resolve_passive
    input loading                  load_inputs
    the pipeline call              export_one (delegates entirely)
    controls and audits            rigidity_control, hoc_audit, arbour_angle
    section 4.1 tables             spine_tables (delegates to p1_spine_stats)
    reporting                      p1_fingerprint, cell_record, summarise

Pure ASCII, LF only. No network. NEURON is touched only inside
align_and_export's gate and inside p1_hoc_audit.neuron_validate (subprocess).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

DRIVER_VERSION = "run_p1_export v1.0"

MANIFEST_COLUMNS = ("cell_id", "layer", "cell_type", "neuron_csv",
                    "alignment_metadata", "synapse_csv", "layer_source")
LAYERS = ("L2", "L3", "L4", "L5", "L6")
CELL_TYPES = ("exc", "inh")
PASSIVE_COLUMNS = ("layer", "cell_type", "cm_uF_cm2", "Ra_ohm_cm",
                   "Rm_qc_ohm_cm2", "cm_reference", "source", "provenance")

# Standing decisions (handoff section 6). Not flags.
CAP_TIPS = True
CAP_H_UM = 0.1
CONTINUATION_DEFAULTS = {"rho_shaft_min": 0.50, "cos_shaft_min": 0.70,
                         "require_taper": True, "min_len_nm": 150.0,
                         "bulge_min": 1.25, "peak_frac_min": 0.30}
REGRESSION_LABELS = ("n_sections", "n_branches", "f_implied", "F_lit",
                     "A_shaft_um2", "A_spine_um2")
SPINE_BASES_COLUMNS = ("spine_root_id", "base_node_id", "n_nodes",
                       "spine_base_section", "section_id")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True,
                   help="campaign root (H01_ROOT): neurons/, synapses/, alignment/")
    p.add_argument("--manifest", default=None,
                   help="manifest CSV from build_p1_manifest.py")
    p.add_argument("--task", type=int, default=None,
                   help="0-based row of the manifest (PBS_ARRAY_INDEX)")
    p.add_argument("--cell", type=int, default=None,
                   help="select the manifest row by cell_id instead of --task")
    p.add_argument("--stage1-dir", default=None,
                   help="default <code dir>/stage1 (the symlink farm)")
    p.add_argument("--passive-table", default=None,
                   help="default <code dir>/passive_params.csv")
    p.add_argument("--out-dir", default=None, help="default <root>/p1")
    p.add_argument("--k-neighbors", type=int, default=3)
    p.add_argument("--lambda-f", type=float, default=100.0)
    p.add_argument("--d-lambda", type=float, default=0.1)
    p.add_argument("--voxel-res-nm", default="8,8,33")
    p.add_argument("--synapse-direction", default="incoming",
                   help="'incoming' keeps postsynaptic sites on this cell; "
                        "'none' disables the filter")
    p.add_argument("--no-synapse-redirect", dest="synapse_redirect",
                   action="store_false", default=True)
    p.add_argument("--no-qc-propagation", dest="qc_propagation",
                   action="store_false", default=True,
                   help="skip the NEURON propagation gate (testing only)")
    p.add_argument("--no-rigidity-control", dest="rigidity_control",
                   action="store_false", default=True)
    p.add_argument("--no-neuron-validate", dest="neuron_validate",
                   action="store_false", default=True,
                   help="skip morphology_exporter.validate_hoc (subprocess)")
    p.add_argument("--force", action="store_true",
                   help="reprocess a cell whose record already matches")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve inputs and constants, export nothing")
    p.add_argument("--summarise", action="store_true",
                   help="fold every <out>/*/neuron_*_p1.json into p1_summary.csv")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def resolve(args):
    here = os.path.dirname(os.path.abspath(__file__))
    args.root = os.path.abspath(args.root)
    args.stage1_dir = os.path.abspath(args.stage1_dir or os.path.join(here, "stage1"))
    args.passive_table = os.path.abspath(args.passive_table
                                         or os.path.join(here, "passive_params.csv"))
    args.out_dir = os.path.abspath(args.out_dir or os.path.join(args.root, "p1"))
    if not os.path.isdir(args.root):
        raise SystemExit("--root does not exist: %s" % args.root)
    if args.manifest:
        args.manifest = os.path.abspath(args.manifest)
        if not os.path.isfile(args.manifest):
            raise SystemExit("--manifest does not exist: %s" % args.manifest)
    if not args.summarise:
        if not args.manifest:
            raise SystemExit("--manifest is required (or --summarise)")
        if (args.task is None) == (args.cell is None):
            raise SystemExit("give exactly one of --task or --cell")
        if not os.path.isdir(args.stage1_dir):
            raise SystemExit("--stage1-dir does not exist: %s (run stage1_link.sh)"
                             % args.stage1_dir)
        if not os.path.isfile(args.passive_table):
            raise SystemExit("--passive-table does not exist: %s" % args.passive_table)
    try:
        vr = tuple(float(v) for v in str(args.voxel_res_nm).split(","))
        assert len(vr) == 3
    except Exception:
        raise SystemExit("--voxel-res-nm must be three comma-separated numbers")
    args.voxel_res = vr
    if str(args.synapse_direction).lower() in ("none", ""):
        args.synapse_direction = None
    return args


# --------------------------------------------------------------------------- #
# Modules                                                                     #
# --------------------------------------------------------------------------- #
def import_modules(args):
    """Code dir first, Stage 1 farm after -- the same order the P2 runner
    uses, so shaft_continuation resolves from h01_code and the labeller,
    exporter, alignment, gate and audit resolve from the farm."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    if args.stage1_dir not in sys.path:
        sys.path.append(args.stage1_dir)
    names = {"al": "alignment", "mx": "morphology_exporter", "hq": "hoc_qc",
             "sl": "spine_labeller", "sra": "synapse_redirect_audit",
             "sd": "spine_density", "nc": "node_classify",
             "shc": "shaft_continuation", "cinsp": "continuation_inspect",
             "sr": "sma_run", "pss": "p1_spine_stats", "pha": "p1_hoc_audit"}
    mods = {}
    for k, n in names.items():
        try:
            mods[k] = __import__(n)
        except Exception as exc:                        # noqa: BLE001
            raise SystemExit("cannot import %r (%s).\n  code dir : %s\n  stage1   : %s"
                             % (n, exc, here, args.stage1_dir))
    return mods


# --------------------------------------------------------------------------- #
# Manifest and passive table                                                  #
# --------------------------------------------------------------------------- #
def load_manifest(path):
    m = pd.read_csv(path, dtype={"synapse_csv": str, "layer_source": str})
    missing = [c for c in MANIFEST_COLUMNS[:5] if c not in m.columns]
    if missing:
        raise SystemExit("manifest %s lacks columns %s" % (path, missing))
    if "synapse_csv" not in m.columns:
        m["synapse_csv"] = ""
    if "layer_source" not in m.columns:
        m["layer_source"] = ""
    m["synapse_csv"] = m["synapse_csv"].fillna("").astype(str)
    m["layer_source"] = m["layer_source"].fillna("").astype(str)
    if m["cell_id"].isna().any():
        raise SystemExit("manifest has an empty cell_id")
    m["cell_id"] = m["cell_id"].astype(np.int64)
    if m["cell_id"].duplicated().any():
        d = m.loc[m["cell_id"].duplicated(), "cell_id"].tolist()
        raise SystemExit("manifest has duplicate cell_id(s): %s" % d[:5])
    bad_layer = sorted(set(m["layer"].astype(str)) - set(LAYERS))
    if bad_layer:
        raise SystemExit("manifest layer(s) not in %s: %s" % (LAYERS, bad_layer))
    m["cell_type"] = m["cell_type"].astype(str).str.lower()
    bad_type = sorted(set(m["cell_type"]) - set(CELL_TYPES))
    if bad_type:
        raise SystemExit("manifest cell_type(s) not in %s: %s" % (CELL_TYPES, bad_type))
    if not len(m):
        raise SystemExit("manifest is empty")
    return m.reset_index(drop=True)


def select_row(manifest, task=None, cell=None):
    if task is not None:
        if not (0 <= task < len(manifest)):
            raise SystemExit("--task %d out of range for a %d-row manifest"
                             % (task, len(manifest)))
        return manifest.iloc[task]
    hit = manifest.index[manifest["cell_id"] == int(cell)]
    if len(hit) != 1:
        raise SystemExit("cell %d is not in the manifest" % int(cell))
    return manifest.iloc[int(hit[0])]


def load_passive_table(path):
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in PASSIVE_COLUMNS if c not in t.columns]
    if missing:
        raise SystemExit("passive table %s lacks columns %s" % (path, missing))
    t["layer"] = t["layer"].str.strip()
    t["cell_type"] = t["cell_type"].str.strip().str.lower()
    if t.duplicated(subset=["layer", "cell_type"]).any():
        raise SystemExit("passive table has a duplicate (layer, cell_type) row")
    return t


def resolve_passive(table, layer, cell_type):
    """The (cm, Ra, Rm_qc) row for one population. A missing row, or a row
    with any blank numeric field, is REFUSED: a defaulted cm or Ra would
    silently fix a wrong lfpy_idx index space for the whole population."""
    sel = table[(table["layer"] == str(layer)) & (table["cell_type"] == str(cell_type).lower())]
    if len(sel) != 1:
        raise SystemExit("passive table has no row for (%s, %s)" % (layer, cell_type))
    r = sel.iloc[0]
    out = {}
    for key, col in (("cm", "cm_uF_cm2"), ("Ra", "Ra_ohm_cm"), ("Rm_qc", "Rm_qc_ohm_cm2")):
        raw = str(r[col]).strip()
        try:
            val = float(raw)
        except ValueError:
            raise SystemExit("passive table row (%s, %s): %s is blank or not a "
                             "number (%r) -- fill it in before running this "
                             "population" % (layer, cell_type, col, raw))
        if not np.isfinite(val) or val <= 0:
            raise SystemExit("passive table row (%s, %s): %s must be positive, got %r"
                             % (layer, cell_type, col, raw))
        out[key] = val
    out["cm_reference"] = str(r["cm_reference"]).strip()
    if out["cm_reference"] != "shaft":
        raise SystemExit("passive table row (%s, %s): cm_reference must be 'shaft' "
                         "(F is applied at model-build time, I-17); got %r"
                         % (layer, cell_type, out["cm_reference"]))
    out["source"] = str(r["source"]).strip()
    out["provenance"] = str(r["provenance"]).strip()
    return out


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def _resolve_path(p, root):
    """Manifest paths are POSIX-relative to --root; a backslash from a manifest
    built on Windows is normalised rather than taken literally."""
    p = str(p).replace("\\", "/")
    return p if os.path.isabs(p) else os.path.join(root, p)


def _sha12(path):
    import hashlib as _h
    if not path or not os.path.isfile(path):
        return None
    h = _h.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _read_csv_or_empty(path, columns=()):
    """An exporter table with zero rows is a header-only CSV since
    morphology_exporter 1.2.1; older files were a bare newline, which pandas
    refuses. Both read as an empty frame with the expected columns."""
    if not path or not os.path.isfile(path) or os.path.getsize(path) <= 1:
        return pd.DataFrame(columns=list(columns))
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(columns))


def load_inputs(row, root, mods, voxel_res, direction, use_synapses, verbose=False):
    al, sra = mods["al"], mods["sra"]
    cid = int(row["cell_id"])
    ncsv = _resolve_path(row["neuron_csv"], root)
    if not os.path.isfile(ncsv):
        raise SystemExit("cell %d: neuron CSV not found: %s" % (cid, ncsv))
    mcsv = _resolve_path(row["alignment_metadata"], root)
    if not os.path.isfile(mcsv):
        raise SystemExit("cell %d: alignment metadata not found: %s" % (cid, mcsv))
    df_raw = pd.read_csv(ncsv)
    metadata_df = al.load_alignment_metadata(mcsv)
    syn_df, n_syn_raw, syn_note = None, 0, "synapse redirect off"
    syn_id_source, syn_empty_after_filter = None, False
    if use_synapses:
        scsv = str(row.get("synapse_csv", "") or "")
        if not scsv:
            syn_note = "no synapse_csv in the manifest -- proceeding without the redirect"
        else:
            scsv = _resolve_path(scsv, root)
            if not os.path.isfile(scsv):
                syn_note = "synapse CSV missing at %s -- proceeding without the redirect" % scsv
            else:
                syn_raw = pd.read_csv(scsv)
                n_syn_raw = int(len(syn_raw))
                # Pre-filter HERE (direction, usable coordinate) so the rows
                # of the mapped frame align 1:1 with `kept` and the H01
                # synapse id can be carried through into C-09 and the spine
                # tables. map_synapses_to_nodes_raw returns fixed columns only.
                kept = syn_raw
                if direction is not None and "direction" in kept.columns:
                    kept = kept[kept["direction"] == direction]
                    dir_note = "direction == %s" % direction
                elif direction is not None:
                    dir_note = "no direction column: filter NOT applied"
                else:
                    dir_note = "all directions"
                kept = kept.dropna(subset=["location_x", "location_y", "location_z"])
                kept = kept.reset_index(drop=True)
                if len(kept):
                    syn_df = sra.map_synapses_to_nodes_raw(kept, df_raw, voxel_res=voxel_res,
                                                           direction=None)
                    # The H01 extractor (Synapse Retrival/Extract_synapses.py)
                    # writes `synapse_id`; `id` is accepted for older files;
                    # a positional fallback is a WARNING and is recorded, so a
                    # join key that is not an H01 id can never pass unnoticed.
                    if "synapse_id" in kept.columns:
                        syn_df["syn_id"] = kept["synapse_id"].to_numpy()
                        syn_id_source = "synapse_id"
                    elif "id" in kept.columns:
                        syn_df["syn_id"] = kept["id"].to_numpy()
                        syn_id_source = "id"
                    else:
                        syn_df["syn_id"] = np.arange(len(kept))
                        syn_id_source = "positional"
                        print("  WARNING: synapse CSV has neither synapse_id nor id; "
                              "syn_id is POSITIONAL (row index after the direction filter)")
                    syn_note = "%d of %d synapses kept (%s; ids from %s)" % (
                        len(syn_df), n_syn_raw, dir_note, syn_id_source)
                else:
                    syn_note = ("0 of %d synapses left after the direction filter (%s) "
                                "-- exported WITHOUT the redirect" % (n_syn_raw, dir_note))
                    syn_empty_after_filter = True
    if verbose:
        print("  %d nodes | bank %s: %d references | %s"
              % (len(df_raw), os.path.basename(mcsv), len(metadata_df), syn_note))
    return df_raw, metadata_df, syn_df, {
        "neuron_csv": ncsv, "alignment_metadata": mcsv,
        "synapse_csv": (_resolve_path(row.get("synapse_csv", ""), root)
                        if str(row.get("synapse_csv", "") or "") else ""),
        "n_synapses_raw": n_syn_raw, "synapse_note": syn_note,
        "syn_id_source": syn_id_source, "syn_empty_after_filter": syn_empty_after_filter}


def label_fn_from(mods, cell_id):
    """export_neuron's label_fn(df, threshold_nm) -> df, through the project's
    own tempdir adapter (sma_run.label_spines_project), the same one the P2
    runner uses, so P1 and P2 label with one code path."""
    sr, sl = mods["sr"], mods["sl"]

    def label_fn(df, threshold_nm):
        labelled, _prov = sr.label_spines_project(df, int(cell_id), sl, float(threshold_nm))
        return labelled
    label_fn.__qualname__ = "sma_run.label_spines_project"
    return label_fn


def continuation_kw():
    return {"require_taper": CONTINUATION_DEFAULTS["require_taper"],
            "min_len_nm": CONTINUATION_DEFAULTS["min_len_nm"],
            "bulge_min": CONTINUATION_DEFAULTS["bulge_min"],
            "peak_frac_min": CONTINUATION_DEFAULTS["peak_frac_min"],
            "rho_shaft_min": CONTINUATION_DEFAULTS["rho_shaft_min"],
            "cos_shaft_min": CONTINUATION_DEFAULTS["cos_shaft_min"]}


def export_kw():
    return dict(cap_tips=CAP_TIPS, cap_h_um=CAP_H_UM, demote_continuations=True,
                continuation_kw=continuation_kw())


# --------------------------------------------------------------------------- #
# Fingerprint                                                                 #
# --------------------------------------------------------------------------- #
def p1_fingerprint(args, params, mods, inputs=None):
    """sha256[:12] of everything that changes the exported artefact OR the
    completeness of its record: module versions, constants, decisions, the
    checks that were on, and the identity (sha256[:12]) of the three input
    files. A record whose fingerprint matches is resumed; any change here
    reprocesses -- a regenerated bank or synapse export included."""
    sl = mods["sl"]
    inputs = inputs or {}
    d = {"driver": DRIVER_VERSION,
         "checks": {"rigidity_control": bool(args.rigidity_control),
                    "neuron_validate": bool(args.neuron_validate)},
         "inputs": {"neuron_csv_sha": inputs.get("neuron_csv_sha"),
                    "synapse_csv_sha": inputs.get("synapse_csv_sha"),
                    "bank_sha": inputs.get("bank_sha")},
         "modules": {k: getattr(mods[k], "MODULE_VERSION", None)
                     for k in ("al", "mx", "hq", "sd", "nc", "shc", "cinsp", "pss", "pha")},
         "labeller_sha256": getattr(sl, "SOURCE_SHA256", None),
         "spine_length_threshold_nm": float(mods["mx"].SPINE_LENGTH_THRESHOLD_NM),
         "passive": {"cm": params["cm"], "Ra": params["Ra"], "Rm_qc": params["Rm_qc"],
                     "cm_reference": params["cm_reference"]},
         "segmentation": {"lambda_f": float(args.lambda_f), "d_lambda": float(args.d_lambda),
                          "nsegs_method": "lambda_f", "k_neighbors": int(args.k_neighbors)},
         "cap": {"cap_tips": CAP_TIPS, "cap_h_um": CAP_H_UM},
         "partition": dict(CONTINUATION_DEFAULTS, rule="three_vote"),
         "synapses": {"redirect": bool(args.synapse_redirect),
                      "direction": args.synapse_direction,
                      "voxel_res_nm": list(args.voxel_res)},
         "qc_propagation": bool(args.qc_propagation)}
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("ascii")).hexdigest()[:12], d


# --------------------------------------------------------------------------- #
# The pipeline call and the checks around it                                  #
# --------------------------------------------------------------------------- #
def export_one(mods, df_raw, cell_id, out_dir, metadata_df, params, label_fn,
               syn_df, args):
    al = mods["al"]
    return al.align_and_export(
        df_raw, int(cell_id), out_dir, metadata_df,
        cm=params["cm"], Ra=params["Ra"], Rm=params["Rm_qc"],
        label_fn=label_fn, k_neighbors=int(args.k_neighbors),
        lambda_f=float(args.lambda_f), d_lambda=float(args.d_lambda),
        nsegs_method="lambda_f", syn_df=syn_df,
        qc_propagation=bool(args.qc_propagation),
        return_frames=True, verbose=bool(args.verbose), **export_kw())


def rigidity_control(mods, df_raw, cell_id, out_dir, label_fn, res):
    """Re-export WITHOUT alignment and compare the geometric keys: alignment
    must never move a phi-affecting quantity. Same export kwargs as the
    aligned run (cap + continuation), or every cell with a continuation
    reports a spurious MOVED. Compared against the exporter's own qc_status,
    not the post-gate one (CELL 6 comment, 2026-09-16)."""
    mx, al = mods["mx"], mods["al"]
    ctl = mx.export_neuron(df_raw, "%s_unaligned" % cell_id, out_dir,
                           label_fn=label_fn, align_fn=None, write_files=False,
                           verbose=False, **export_kw())
    res_rigid = dict(res)
    res_rigid["qc_status"] = res.get("exporter_qc_status", res["qc_status"])
    chk = al.regression_check(ctl, res_rigid)
    return {"identical": bool(chk["identical"]), "diffs": dict(chk["diffs"]),
            "unaligned": {k: ctl.get(k) for k in REGRESSION_LABELS},
            "aligned": {k: res.get(k) for k in REGRESSION_LABELS}}


def hoc_audit(mods, res, cell_id, out_dir, validate_with_neuron, stage1_dir):
    pha = mods["pha"]
    entry = res["files"]["hoc"]
    hoc_path = entry if isinstance(entry, str) else entry["path"]
    geo = pha.audit_hoc_geometry(hoc_path)
    nrep = None
    if validate_with_neuron:
        here = os.path.dirname(os.path.abspath(__file__))
        nrep = pha.neuron_validate(hoc_path, sys_path=[here, stage1_dir] + list(sys.path))
    viol = pha.classify_violations(geo, nrep)
    if validate_with_neuron and not (nrep or {}).get("neuron_available"):
        # asked for and not delivered: a NEURON crash, refusal or timeout is
        # indistinguishable from "not installed" inside neuron_validate, so
        # it is a SOFT violation here rather than silence
        viol[pha.SOFT].append("neuron_validate unavailable: %s"
                              % str((nrep or {}).get("import_error", "?"))[:120])
    v = pha.verdict(viol)
    validation = {"geometry": geo, "neuron": nrep, "violations": viol, "verdict": v}
    path = os.path.join(out_dir, "neuron_%d_hoc_validation.json" % int(cell_id))
    with open(path, "w", newline="\n") as fh:
        fh.write(json.dumps(validation, indent=2, sort_keys=True, default=str))
    return validation, v, path


def arbour_angle(mods, res, labelled, cell_id):
    """angle_from_z_deg of the aligned arbour, the input to the bank-level
    orientation-consistency check. Transient frame, freed by the caller."""
    al, pha = mods["al"], mods["pha"]
    soma_pos = np.asarray(res["alignment"]["soma_pos_nm"], float)
    mean_matrix = np.asarray(res["alignment"]["mean_matrix"], float)
    df_ali = al.make_align_fn(soma_pos, mean_matrix)(labelled, int(cell_id))
    return float(pha.angle_from_z_deg(pha.arbour_direction(df_ali)))


def spine_tables(mods, res, labelled, cell_id, out_dir):
    """Handoff section 4.1: long + wide per-spine tables from the pre-prune
    labelled frame (post three-vote, post re-classify, post soma enforce),
    joined to the exporter's spine_bases and the C-09 mapped synapses."""
    al, pss, sd = mods["al"], mods["pss"], mods["sd"]
    soma_pos = np.asarray(res["alignment"]["soma_pos_nm"], float)
    mean_matrix = np.asarray(res["alignment"]["mean_matrix"], float)
    align_fn = al.make_align_fn(soma_pos, mean_matrix)
    files = res.get("files", {})
    sb = (_read_csv_or_empty(files["spine_bases"], SPINE_BASES_COLUMNS)
          if files.get("spine_bases") else None)
    ms = None
    if files.get("mapped_synapses"):
        mp = files["mapped_synapses"]
        mp = mp if isinstance(mp, str) else mp.get("path")
        if mp and os.path.isfile(mp):
            ms = _read_csv_or_empty(mp)
    ck = continuation_kw()
    votes = pss.vote_table(labelled, mods["shc"], mods["cinsp"], sd,
                           rho_shaft_min=ck["rho_shaft_min"], cos_shaft_min=ck["cos_shaft_min"],
                           min_len_nm=ck["min_len_nm"], bulge_min=ck["bulge_min"],
                           peak_frac_min=ck["peak_frac_min"])
    long_df = pss.long_table(labelled, int(cell_id), align_fn=align_fn)
    wide_df = pss.wide_table(long_df, labelled, int(cell_id), sd, spine_bases=sb,
                             mapped_synapses=ms, votes=votes)
    wide_df = pss.add_aligned_base(wide_df, labelled, align_fn, int(cell_id))
    ok, why = pss.check_wide_is_groupby_of_long(long_df, wide_df)
    if not ok:
        raise RuntimeError("spine tables inconsistent: %s" % why)
    if sb is not None:
        if len(sb) != len(wide_df):
            raise RuntimeError("spine_stats has %d rows but the exporter pruned %d spines"
                               % (len(wide_df), len(sb)))
        if len(sb):
            # prune_spines removes EVERY descendant of a root; the tables walk
            # spine-class children only. A shaft-class node under a spine root
            # would make the two disagree, and that must not pass silently.
            a = sb.set_index("spine_root_id")["n_nodes"].astype(int)
            b = wide_df.set_index("root_node_id")["n_nodes"].astype(int)
            bad = [int(r) for r in a.index if int(a[r]) != int(b.get(r, -1))]
            if bad:
                raise RuntimeError("n_nodes differs between spine_bases and spine_stats "
                                   "for root(s) %s" % bad[:5])
    paths = pss.write_tables(out_dir, int(cell_id), long_df, wide_df)
    return {"n_spines": int(len(wide_df)), "n_spine_nodes": int(len(long_df)),
            "n_spines_with_synapses": int((wide_df["n_syn"] > 0).sum()) if len(wide_df) else 0,
            "partition_source_counts": (wide_df["partition_source"].value_counts().to_dict()
                                        if len(wide_df) else {}),
            "files": paths}


# --------------------------------------------------------------------------- #
# Records                                                                     #
# --------------------------------------------------------------------------- #
def record_path(out_dir, cell_id):
    return os.path.join(out_dir, "neuron_%d_p1.json" % int(cell_id))


def cell_record(row, params, fp, fpd, res, extra, error=None, seconds=None):
    rec = {"driver": DRIVER_VERSION, "cell_id": int(row["cell_id"]),
           "layer": str(row["layer"]), "cell_type": str(row["cell_type"]),
           "layer_source": str(row.get("layer_source", "") or ""),
           "fingerprint": fp, "fingerprint_detail": fpd,
           "passive": params, "status": "error" if error else "ok",
           "error": error, "seconds": seconds,
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if res is not None:
        keep = {k: v for k, v in res.items()
                if k not in ("frames", "files", "propagation_qc", "hoc_validation")
                and not isinstance(v, pd.DataFrame)}
        qc = res.get("propagation_qc") or {}
        keep["gate_status"] = qc.get("qc_status")
        keep["gate_dv_soma_mV"] = (qc.get("C4_soma") or {}).get("dv_soma_mV")
        keep["gate_dv_min_mV"] = qc.get("dv_min_mV")
        keep["gate_monotone_violations"] = (qc.get("C3_monotone") or {}).get("n_violations")
        keep["files"] = {k: (v if isinstance(v, str) else (v or {}).get("path"))
                         for k, v in (res.get("files") or {}).items()}
        rec["result"] = keep
    rec.update(extra or {})
    return rec


def write_record(out_dir, cell_id, rec):
    os.makedirs(out_dir, exist_ok=True)
    p = record_path(out_dir, cell_id)
    tmp = p + ".tmp"
    with open(tmp, "w", newline="\n") as fh:
        fh.write(json.dumps(rec, indent=2, sort_keys=True, default=str))
    os.replace(tmp, p)
    return p


SUMMARY_KEYS = (
    "cell_id", "layer", "cell_type", "layer_source", "status", "error", "qc_status",
    "reasons", "gate_status", "hoc_verdict", "rigidity_identical", "quarantined",
    "resumed", "n_sections", "n_branches", "f_implied", "F_lit",
    "f_implied_nocap", "F_lit_nocap", "f_implied_cap", "F_lit_cap",
    "A_shaft_um2", "A_spine_um2", "cap_tips", "cap_h_um",
    "n_spine_roots", "n_demoted", "n_nodes_demoted", "n_rescued_by_taper",
    "n_undecidable", "n_spines", "n_spine_nodes", "n_spines_with_synapses",
    "n_synapses", "n_on_pruned_spine", "n_redirected", "n_unresolved_spine_bases",
    "n_unknown_type", "totnsegs", "angle_from_z_deg", "soma_diameter_um",
    "total_length_um", "cm", "Ra", "Rm_qc", "fingerprint", "seconds", "written_utc")


def summary_row(rec):
    r = rec.get("result") or {}
    cr = r.get("continuation_report") or {}
    hv = rec.get("hoc_validation_summary") or {}
    row = {"cell_id": rec.get("cell_id"), "layer": rec.get("layer"),
           "cell_type": rec.get("cell_type"), "layer_source": rec.get("layer_source"),
           "status": rec.get("status"), "error": rec.get("error"),
           "qc_status": r.get("qc_status"),
           "reasons": ";".join(r.get("reasons") or []),
           "gate_status": r.get("gate_status"),
           "hoc_verdict": hv.get("verdict"),
           "rigidity_identical": (rec.get("rigidity") or {}).get("identical"),
           "quarantined": rec.get("quarantined", False),
           "resumed": rec.get("resumed", False),
           "n_spine_roots": cr.get("n_spine_roots"), "n_demoted": cr.get("n_demoted"),
           "n_nodes_demoted": cr.get("n_nodes_demoted"),
           "n_rescued_by_taper": cr.get("n_rescued_by_taper"),
           "n_undecidable": cr.get("n_undecidable"),
           "n_spines": (rec.get("spine_tables") or {}).get("n_spines"),
           "n_spine_nodes": (rec.get("spine_tables") or {}).get("n_spine_nodes"),
           "n_spines_with_synapses": (rec.get("spine_tables") or {}).get("n_spines_with_synapses"),
           "angle_from_z_deg": rec.get("angle_from_z_deg"),
           "soma_diameter_um": hv.get("soma_diameter_um"),
           "total_length_um": hv.get("total_length_um"),
           "cm": (rec.get("passive") or {}).get("cm"),
           "Ra": (rec.get("passive") or {}).get("Ra"),
           "Rm_qc": (rec.get("passive") or {}).get("Rm_qc"),
           "fingerprint": rec.get("fingerprint"), "seconds": rec.get("seconds"),
           "written_utc": rec.get("written_utc")}
    for k in ("n_sections", "n_branches", "f_implied", "F_lit", "f_implied_nocap",
              "F_lit_nocap", "f_implied_cap", "F_lit_cap", "A_shaft_um2", "A_spine_um2",
              "cap_tips", "cap_h_um", "n_synapses", "n_on_pruned_spine", "n_redirected",
              "n_unresolved_spine_bases", "n_unknown_type", "totnsegs"):
        row[k] = r.get(k)
    return {k: row.get(k) for k in SUMMARY_KEYS}


def summarise(out_root, manifest_path=None):
    """Every <out>/*/neuron_*_p1.json -> <out>/p1_summary.csv, one row per
    cell, sorted by cell_id. Written by ONE process after the array, never
    by the tasks, so there is no shared file to race on. With a manifest,
    also returns the manifest cells that have NO record (an array narrower
    than the manifest leaves them unprocessed without any error)."""
    rows = []
    if os.path.isdir(out_root):
        for d in sorted(os.listdir(out_root)):
            sub = os.path.join(out_root, d)
            if not os.path.isdir(sub) or d.startswith("_") or d.startswith("."):
                continue
            for f in sorted(os.listdir(sub)):
                if f.startswith("neuron_") and f.endswith("_p1.json"):
                    with open(os.path.join(sub, f)) as fh:
                        rows.append(summary_row(json.load(fh)))
    df = pd.DataFrame(rows, columns=SUMMARY_KEYS)
    if len(df):
        df = df.sort_values("cell_id").reset_index(drop=True)
    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, "p1_summary.csv")
    df.to_csv(path, index=False, lineterminator="\n")
    missing = []
    if manifest_path:
        m = load_manifest(manifest_path)
        have = set(int(v) for v in df["cell_id"].dropna()) if len(df) else set()
        missing = [int(v) for v in m["cell_id"] if int(v) not in have]
    return path, df, missing


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = resolve(build_parser().parse_args(argv))
    if args.summarise:
        path, df, missing = summarise(args.out_dir, args.manifest)
        n_ok = int((df["status"] == "ok").sum()) if len(df) else 0
        print("p1_summary: %d cells (%d ok, %d error) -> %s"
              % (len(df), n_ok, len(df) - n_ok, path))
        if len(df):
            print(df[["cell_id", "layer", "cell_type", "status", "qc_status", "hoc_verdict",
                      "n_spines", "F_lit"]].to_string(index=False))
        if args.manifest:
            print("manifest %s: %d cell(s) without a record%s"
                  % (os.path.basename(args.manifest), len(missing),
                     (": %s" % missing[:10]) if missing else ""))
        return 1 if missing else 0

    t0 = time.time()
    mods = import_modules(args)
    manifest = load_manifest(args.manifest)
    row = select_row(manifest, task=args.task, cell=args.cell)
    cid = int(row["cell_id"])
    ptab = load_passive_table(args.passive_table)
    params = resolve_passive(ptab, row["layer"], row["cell_type"])
    out_dir = os.path.join(args.out_dir, str(cid))

    df_raw, metadata_df, syn_df, inputs = load_inputs(
        row, args.root, mods, args.voxel_res, args.synapse_direction,
        args.synapse_redirect, verbose=False)
    inputs["neuron_csv_sha"] = _sha12(inputs["neuron_csv"])
    inputs["synapse_csv_sha"] = _sha12(inputs.get("synapse_csv")) if args.synapse_redirect else None
    inputs["bank_sha"] = _sha12(inputs["alignment_metadata"])
    fp, fpd = p1_fingerprint(args, params, mods, inputs)

    print("%s | cell %d (%s %s) | row %d/%d | cm %.3g Ra %.4g Rm_qc %.4g | fp %s"
          % (DRIVER_VERSION, cid, row["layer"], row["cell_type"],
             int(manifest.index[manifest["cell_id"] == cid][0]), len(manifest),
             params["cm"], params["Ra"], params["Rm_qc"], fp))
    print("  %d nodes | bank %s: %d references | %s"
          % (len(df_raw), os.path.basename(inputs["alignment_metadata"]),
             len(metadata_df), inputs["synapse_note"]))

    # resume: a record with the same fingerprint and no error is final. That
    # includes a cell the gate rejected or the audit quarantined -- both are
    # deterministic verdicts on these inputs with these constants; --force
    # or any fingerprint change (inputs included) reprocesses.
    rp = record_path(out_dir, cid)
    if os.path.isfile(rp) and not args.force:
        try:
            with open(rp) as fh:
                old = json.load(fh)
        except Exception:                                    # noqa: BLE001
            old = {}
        if old.get("fingerprint") == fp and old.get("status") == "ok":
            print("  resumed: record %s matches fingerprint %s -- nothing to do "
                  "(--force to reprocess)" % (os.path.basename(rp), fp))
            return 0
        if old.get("fingerprint") not in (None, fp):
            print("  record fingerprint %s != %s: reprocessing"
                  % (old.get("fingerprint"), fp))
        elif old.get("status") == "error":
            print("  record status error: reprocessing")

    if args.dry_run:
        print("DRY RUN: inputs and constants resolved, nothing exported")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    label_fn = label_fn_from(mods, cid)
    res, err, extra = None, None, {"inputs": inputs}
    try:
        res = export_one(mods, df_raw, cid, out_dir, metadata_df, params, label_fn,
                         syn_df, args)
        labelled = (res.pop("frames", {}) or {}).get("labelled")
        cr = res.get("continuation_report") or {}
        if cr.get("applied"):
            print("  continuations: %d of %d spine root(s) demoted (%d nodes); "
                  "%d rescued by taper, %d undecidable"
                  % (cr.get("n_demoted", 0), cr.get("n_spine_roots", 0),
                     cr.get("n_nodes_demoted", 0), cr.get("n_rescued_by_taper", 0),
                     cr.get("n_undecidable", 0)))
        qc = res.get("propagation_qc") or {}
        if qc:
            print("  gate: %s | dv soma %s mV, min %s mV, %s segs"
                  % (qc.get("qc_status"), (qc.get("C4_soma") or {}).get("dv_soma_mV"),
                     qc.get("dv_min_mV"), qc.get("totnsegs")))
        if res["qc_status"] == "fail":
            print("  GATED OUT: %s -- nothing written to %s"
                  % ("; ".join(res.get("reasons", [])), out_dir))
            extra["gated_out"] = True
        else:
            if inputs.get("syn_empty_after_filter"):
                res["reasons"] = list(res.get("reasons", [])) + [
                    "synapse_file_nonempty_but_0_kept_after_direction_filter"]
                res["qc_status"] = mods["pha"].merge_verdict(res["qc_status"],
                                                             "pass_low_confidence")
            if args.rigidity_control:
                extra["rigidity"] = rigidity_control(mods, df_raw, cid, out_dir,
                                                     label_fn, res)
                print("  rigidity: %s" % ("IDENTICAL" if extra["rigidity"]["identical"]
                                          else "*** MOVED *** %s" % extra["rigidity"]["diffs"]))
            validation, v, vpath = hoc_audit(mods, res, cid, out_dir,
                                             args.neuron_validate, args.stage1_dir)
            geo = validation["geometry"]
            extra["hoc_validation_summary"] = {
                "verdict": v, "path": vpath, "n_sections": geo["n_sections"],
                "n_orphans": len(geo["orphans"]), "total_length_um": geo["total_length_um"],
                "soma_diameter_um": geo["soma_diameter_um"],
                "neuron_available": bool((validation.get("neuron") or {}).get("neuron_available")),
                "structural": validation["violations"]["structural"],
                "soft": validation["violations"]["soft"]}
            res["qc_status"] = mods["pha"].merge_verdict(res["qc_status"], v)
            print("  hoc: %s | %d sections, %d orphans, %.1f um cable, soma %.3f um"
                  % (v.upper(), geo["n_sections"], len(geo["orphans"]),
                     geo["total_length_um"], geo["soma_diameter_um"]))
            for lvl in ("structural", "soft"):
                for msg in validation["violations"][lvl]:
                    print("    [%s] %s" % (lvl, msg))
            if v == "fail":
                qdir, moved = mods["pha"].quarantine(out_dir, cid, res.get("files"))
                extra["quarantined"] = True
                extra["quarantine_dir"] = qdir
                extra["hoc_validation_summary"]["path"] = os.path.join(
                    qdir, os.path.basename(vpath))
                print("  QUARANTINED -> %s (%d files)" % (qdir, len(moved)))
            else:
                if labelled is not None:
                    extra["angle_from_z_deg"] = arbour_angle(mods, res, labelled, cid)
                    print("  aligned arbour is %.1f deg from +z" % extra["angle_from_z_deg"])
                    extra["spine_tables"] = spine_tables(mods, res, labelled, cid, out_dir)
                    st = extra["spine_tables"]
                    print("  spine tables: %d spines, %d nodes, %d with synapses, sources %s"
                          % (st["n_spines"], st["n_spine_nodes"],
                             st["n_spines_with_synapses"], st["partition_source_counts"]))
                else:
                    extra["spine_tables_error"] = "align_and_export returned no labelled frame"
                    print("  WARNING: no labelled frame -- spine tables and angle not written")
    except SystemExit:
        raise
    except Exception as exc:                                 # noqa: BLE001
        err = "%s: %s" % (type(exc).__name__, exc)
        traceback.print_exc()

    rec = cell_record(row, params, fp, fpd, res, extra, error=err,
                      seconds=round(time.time() - t0, 1))
    write_record(out_dir, cid, rec)
    print("  record: %s | status %s | qc %s | %.1f s"
          % (record_path(out_dir, cid), rec["status"],
             (rec.get("result") or {}).get("qc_status"), time.time() - t0))
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
