# =========================================================================== #
# CELL 6d -- NEW. Paste immediately after CELL 6c.                            #
# =========================================================================== #
# Is the SHAFT intact in the pruned .hoc?
#
# Spines are removed by label (morphology_exporter step 8, prune_spines). This
# cell verifies the claim that removing them leaves every shaft node in place
# with its radius unchanged -- the assumption the spine-deprived arbour used in
# the microcolumn simulation rests on.
#
# Offline: no CloudVolume, no NEURON. Reads the committed .hoc back and
# compares it with the exporter's OWN labelled and pruned frames, rebuilt here
# from the raw skeleton through the SAME make_label_fn (CELL 6a) and the same
# threshold. Uses the transform recorded in neuron_{nid}_alignment.json, which
# is bit-for-bit what write_hoc used.
#
# Placement: after CELL 6, because it needs the .hoc files committed; after
# CELL 6c only so it does not sit between CELL 6 and 6c's use of `frames`.
# It needs nothing from either namespace beyond the CELL 4 paths.
#
# Put check_pruned_hoc.py and smoke_check_pruned_hoc.py in CODE_DIR and run
# the smoke test once:
#   !cd "$CODE_DIR" && python3 smoke_check_pruned_hoc.py     -> 12/12

import importlib
import os
import sys

import pandas as pd

import alignment as al
import check_pruned_hoc as CK
import morphology_exporter as mx
import node_classify as nc
importlib.reload(CK)

# Section D: what shaft_continuation WOULD demote, and what that does to F.
# Nothing is written and the bank is untouched -- this quantifies the open
# item (the exporter prunes on the LABEL alone and does not run the
# correction) rather than acting on it. Set False to skip it outright.
WANT_CONTINUATIONS = True

# shaft_continuation.py lives in the Spine Mesh Analysis folder, not in Stage 1.
# Either copy it into CODE_DIR (it needs only spine_labeller, which is here),
# or leave SHC_DIR pointing at the folder that has it. Sections A-C do not
# depend on it and run either way.
SHC_DIR = ("/content/drive/MyDrive/Colab Notebooks/New algorithms/"
           "Spine Mesh Analysis")


def _import_shc():
    for _try in (0, 1):
        try:
            import shaft_continuation as _shc
            import spine_density as _sd
            return _shc, _sd
        except ImportError as _exc:
            if _try or not (SHC_DIR and os.path.isdir(SHC_DIR)):
                print("section D SKIPPED: %s. Copy shaft_continuation.py into "
                      "CODE_DIR, or set SHC_DIR to the folder holding it. "
                      "Sections A-C below are unaffected." % _exc)
                return None, None
            if SHC_DIR not in sys.path:
                sys.path.append(SHC_DIR)
    return None, None


shc, sd = _import_shc() if WANT_CONTINUATIONS else (None, None)
SCORE_CONTINUATIONS = shc is not None
if SCORE_CONTINUATIONS:
    print("section D: %s, found at %s"
          % (shc.MODULE_VERSION, os.path.dirname(shc.__file__)))

CHECK_HOC_DIR = OUTPUT_DIR                                            # noqa: F821
CHECK_SUMMARY_CSV = os.path.join(FIGURE_DIR, "pruned_hoc_check.csv")  # noqa: F821
CHECK_N_SHOW = 8

# The keywords the committed .hoc was exported with. They change the PARTITION,
# so the reference frames must be rebuilt with the same ones or section C
# compares a corrected .hoc against an uncorrected frame -- on neuron
# 15543554616 that reported 1566 spurious "extra" points and a 470 um cable
# excess, which were the restored continuation branches. _EXPORT_KW is what
# CELL 6 defines; fall back to an explicit dict if CELL 6 has not run.
CHECK_EXPORT_KW = globals().get("_EXPORT_KW") or {
    "cap_tips": globals().get("CAP_TIPS", False),
    "cap_h_um": globals().get("CAP_H_UM", 0.1),
    "demote_continuations": globals().get("DEMOTE_CONTINUATIONS", False),
    "continuation_kw": globals().get("CONTINUATION_KW"),
}
print("rebuilding reference frames with export_kw = %s" % CHECK_EXPORT_KW)

# The cells actually on disk, so a gated-out or quarantined neuron is skipped
# rather than raising. NEURON_IDS is the REQUEST; this is the RESULT.
_have = sorted(int(f.split("_")[1]) for f in os.listdir(CHECK_HOC_DIR)
               if f.startswith("neuron_") and f.endswith("_aligned.hoc")
               and "unaligned" not in f)
_want = [int(n) for n in NEURON_IDS]                                  # noqa: F821
CHECK_IDS = [n for n in _want if n in _have]
if len(CHECK_IDS) < len(_want):
    print("no .hoc for %s -- gated out, quarantined, or not yet exported"
          % sorted(set(_want) - set(_have)))

_rows = []
for _nid in CHECK_IDS:
    print("\n" + "=" * 76)
    _raw = pd.read_csv("%s/neuron_%s.csv" % (SKELETONS_DIR, _nid))    # noqa: F821
    _rep = CK.check_pruned_hoc(_nid, _raw, CHECK_HOC_DIR, mx, al, nc,
                               make_label_fn(_nid),                   # noqa: F821
                               threshold_nm=mx.SPINE_LENGTH_THRESHOLD_NM,
                               export_kw=CHECK_EXPORT_KW)
    CK.print_report(_rep, n_show=CHECK_N_SHOW)
    _row = {k: v for k, v in _rep.items()
            if not isinstance(v, (pd.DataFrame, list))
            and k not in ("extra_points_um", "labelled_frame")}
    if SCORE_CONTINUATIONS:
        _sc = CK.score_continuations(_rep["labelled_frame"], shc, sd,
                                     cutoff_um=60.0, nid=_nid)
        CK.print_continuations(_sc, n_show=CHECK_N_SHOW)
        _row.update({k: _sc[k] for k in (
            "n_spine_roots", "n_shaft_like", "n_ambiguous_bp", "method",
            "use_radius", "n_nodes_demoted", "A_moved_um2", "dF_lit",
            "F_lit_as_exported", "F_lit_corrected")})
        _row["frac_roots_shaft_like"] = (
            _sc["n_shaft_like"] / max(_sc["n_spine_roots"], 1))
    _rows.append(_row)

if _rows:
    _df = pd.DataFrame(_rows)
    _df.to_csv(CHECK_SUMMARY_CSV, index=False, lineterminator="\n")
    print("\n" + "=" * 76)
    _cols = ["nid", "ok", "n_removed", "n_spines_removed",
             "n_non_spine_nodes_removed", "n_missing_in_hoc", "n_extra_in_hoc",
             "n_diam_mismatch", "cable_diff_um"]
    if SCORE_CONTINUATIONS:
        _cols += ["n_shaft_like", "frac_roots_shaft_like", "A_moved_um2",
                  "F_lit_as_exported", "F_lit_corrected", "dF_lit"]
    print(_df[[c for c in _cols if c in _df.columns]].to_string(index=False))
    print("\n%d/%d cell(s) verified shaft-intact -> %s"
          % (int(_df["ok"].sum()), len(_df), CHECK_SUMMARY_CSV))
    if SCORE_CONTINUATIONS and "dF_lit" in _df.columns:
        print("shaft continuations would move F_lit by %+.4f on average "
              "(%+.4f worst). The bank is UNCHANGED; deciding to apply the "
              "correction is a Stage 1 decision, and the HPC single-pass "
              "pipeline should run it before prune_spines."
              % (_df["dF_lit"].mean(), _df["dF_lit"].min()))
    del _df
del _rows, _have, _want
