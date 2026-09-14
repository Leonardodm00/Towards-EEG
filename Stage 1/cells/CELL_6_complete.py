# %% CELL 6 -- export + align + GATE, one cell at a time --------------------
# REPLACE the existing CELL 6 in full. CELL 6a is unchanged.
#
# CHANGED in this version -- O-1, the shaft-continuation correction, in THREE
# places. Everything else is identical to the previous CELL 6.
#
#   1. DEMOTE_CONTINUATIONS / CONTINUATION_KW, the switches.
#   2. align_and_export(..., demote_continuations=..., continuation_kw=...).
#      align_and_export forwards **export_kwargs straight to export_neuron, so
#      no change to alignment.py is needed.
#   3. THE SAME two keywords on the rigidity control's own export_neuron call.
#      The control must be built the same way as the aligned run or the check
#      reports a spurious "MOVED" on every cell that has a continuation --
#      exactly the trap the cap_tips comment below already warns about.
#
# WHY. spine_labeller's only criterion is subtree length <= threshold, so a
# real dendritic branch ending within the threshold is labelled a spine, pruned
# from the cable, and its membrane counted as spine area -- inflating F. On
# neuron 15543554616 the two-observable scorer flagged 312 of 2023 components,
# worth -0.143 in F_lit, 22 percent of the whole spine contribution.
#
# The exporter applies the correction at step 4b, after the labeller (so the
# head/neck labels are available as the third vote) and before both the
# re-classify and phi (so the .hoc and F describe the same cell). Three votes
# must agree; a component too short to measure is left ALONE.
#
# NEEDS morphology_exporter >= 1.2.0, plus shaft_continuation.py and
# continuation_inspect.py in CODE_DIR.
#
# RESUME. A cell whose alignment.json already exists is skipped, so a bank
# exported before this change is NOT corrected by re-running. Delete the
# checkpoints first:
#     !rm "$OUTPUT_DIR"/neuron_*_alignment.json
#
# STREAMING: one neuron processed to completion, persisted, then every large
# object it created is released before the next is read. Peak RAM is one
# neuron's worth regardless of how many are in the bank.
import traceback

# ---- O-1 switches ---------------------------------------------------------
DEMOTE_CONTINUATIONS = True          # apply the correction in the export path
CONTINUATION_KW = {                  # None for the module defaults
    "require_taper": True,           # all three votes must agree
    "min_len_nm": 150.0,             # below this rho and cos are unmeasurable
    "bulge_min": 1.25,               # distal max / preceding min, to call a head
    "peak_frac_min": 0.30,           # the peak must sit in the distal 70 percent
}
_EXPORT_KW = dict(cap_tips=CAP_TIPS, cap_h_um=CAP_H_UM,                # noqa: F821
                  demote_continuations=DEMOTE_CONTINUATIONS,
                  continuation_kw=CONTINUATION_KW)
if DEMOTE_CONTINUATIONS and getattr(mx, "MODULE_VERSION", "") < \
        "morphology_exporter-1.2.0":                                   # noqa: F821
    raise RuntimeError(
        "DEMOTE_CONTINUATIONS needs morphology_exporter >= 1.2.0 (found %r). "
        "Update the module in CODE_DIR, or set the switch to False."
        % getattr(mx, "MODULE_VERSION", None))                         # noqa: F821


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
            # A checkpoint written before O-1 describes an UNCORRECTED cell.
            # Resuming it into a corrected bank would mix two partitions in one
            # summary, so it is reprocessed instead of silently accepted.
            _cr = res.get("continuation_report") or {}
            if DEMOTE_CONTINUATIONS and not _cr.get("applied"):
                print("  checkpoint predates the continuation correction -- "
                      "reprocessing rather than resuming")
                del res, committed, _cr
            else:
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
                del res, committed, rec, _cr
                continue
        else:
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
        # align_and_export forwards **export_kwargs to export_neuron, so the
        # continuation switches ride through unchanged.
        res = al.align_and_export(
            df_raw, nid, OUTPUT_DIR, metadata_df,
            cm=CM_BASE, Ra=RA, Rm=RM_QC, label_fn=label_fn,
            k_neighbors=K_NEIGHBORS, syn_df=syn_df,
            return_frames=True, verbose=True, **_EXPORT_KW)

        _cr = res.get("continuation_report") or {}
        if _cr.get("applied"):
            print("  continuations: %d of %d spine root(s) demoted to shaft "
                  "(%d nodes); %d held back by the taper, %d too short to "
                  "judge" % (_cr["n_demoted"], _cr["n_spine_roots"],
                             _cr["n_nodes_demoted"], _cr["n_rescued_by_taper"],
                             _cr["n_undecidable"]))

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
            # The cap setting MUST match the aligned run, and so must the
            # continuation setting -- both change f_implied / A_spine_um2,
            # which are regression keys. Running an uncorrected control against
            # a corrected `res` would report "alignment moved a quantity" on
            # every cell that has a continuation, with identical geometry.
            # _EXPORT_KW is passed to BOTH calls for exactly that reason.
            ctl = mx.export_neuron(df_raw, "%s_unaligned" % nid, OUTPUT_DIR,
                                   label_fn=label_fn, align_fn=None,
                                   write_files=False, verbose=False,
                                   **_EXPORT_KW)
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
        # The continuation counts travel into the bank summary, so a row can be
        # told apart from an uncorrected one without reopening the json.
        if _cr.get("applied"):
            res["n_continuations_demoted"] = _cr["n_demoted"]
            res["n_continuation_nodes_demoted"] = _cr["n_nodes_demoted"]
            res["n_rescued_by_taper"] = _cr["n_rescued_by_taper"]
            res["n_continuation_undecidable"] = _cr["n_undecidable"]
            res["continuation_require_taper"] = _cr["require_taper"]
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
                      "df_lab", "df_ali", "geo", "nrep", "viol", "labels",
                      "_cr"):
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
if DEMOTE_CONTINUATIONS:
    _nd = [r.get("n_continuations_demoted") for r in records
           if r.get("n_continuations_demoted") is not None]
    if _nd:
        print("continuations demoted: %d total over %d cell(s), median %.0f "
              "per cell" % (int(np.sum(_nd)), len(_nd), float(np.median(_nd))))
    del _nd
print("\nbank summary: %s" % BANK_SUMMARY_CSV)
