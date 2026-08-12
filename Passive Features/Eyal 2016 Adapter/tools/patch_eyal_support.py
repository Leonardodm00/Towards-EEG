"""
patch_eyal_support.py -- apply the five Eyal-support patches to the
Towards-EEG passive-fitting pipeline.

Every patch is GUARDED: each adds an opt-in parameter whose default
reproduces the current Allen behaviour exactly, so an Allen run that does
not pass the new flags is unaffected. The one deliberate exception is
documented below (patch 1, sweep_numbers) and is a bug fix, not a
behaviour change.

    1. load_cell_from_archive      Square-Subthreshold averaging may be
                                   grouped by AMPLITUDE, not polarity alone
                                   (group_ss_by_amplitude=False by default)
    2. _rmsd_for_validation_bundle the brief-pulse validation RMSD window is
                                   no longer hard-coded to (1, 100) ms
                                   (resolved from a module global at call
                                   time, exactly as the dt globals are)
    3. load_cell_from_archive      morphology filename read from
                                   metadata["morphology_file"], defaulting
                                   to "reconstruction.swc"
    4. PassiveCell._import_swc     Import3d reader dispatched on file suffix
                                   (.asc -> Import3d_Neurolucida3)
    5. load_cells_from_archive     require_long_square=True by default; set
                                   False for datasets with no Long Square
    6. run_biological_fit.py       CLI flags so 1-5 are actually reachable:
                                   --no-long-square, --group-ss-by-amplitude,
                                   --ss-amplitude-tol-pA, --axon-replacement
    7. _classify_fit               an EMPTY validation set now yields
                                   "not_evaluated", not "failed" 

Usage
-----
    python3 patch_eyal_support.py --code-dir <dir with passive_fitting_hpc_fixed.py> --check
    python3 patch_eyal_support.py --code-dir <same dir> --apply
    python3 patch_eyal_support.py --code-dir <same dir> --revert

--check   verifies every anchor is present and reports which patches are
          already applied. Changes nothing. Run this first.
--apply   writes <file>.orig.bak once (never overwritten), then patches.
          Idempotent: re-running is a no-op.
--revert  restores from the .orig.bak files.

ASCII-only, LF-only by construction.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

MONOLITH = "passive_fitting_hpc_fixed.py"
LONGSTEP = "passive_long_step_training.py"
RUNNER = "run_biological_fit.py"


# ===========================================================================
#  PATCH 1 -- Square-Subthreshold averaging grouped by amplitude
# ===========================================================================
#
# WHY. load_cell_from_archive groups SS pulses by POLARITY only and averages
# everything in a polarity into one bundle, labelling it with the MEAN of the
# constituent peak amplitudes. That is safe for Allen, where every SS pulse
# that survives QC is within SQ_SUB_AMPLITUDE_TOL_PA (30 pA) of the nominal
# 200 pA, so a polarity group is already amplitude-homogeneous. It is wrong
# for any dataset carrying several nominal amplitudes: Eyal et al. (2016)
# cell 0603_cell08 has +50, +100 and +200 pA traces, which the current code
# averages into a single bundle labelled +116.7 pA -- an amplitude at which
# no experiment was ever performed. Nothing raises; the corrupted bundle
# flows into the optimiser looking entirely normal.
#
# GUARD. group_ss_by_amplitude defaults to False, which reproduces the
# current partitioning exactly. Pass True for multi-amplitude datasets.
# When it is False and the pool IS heterogeneous, a loud warning fires, so
# the dangerous case cannot pass unnoticed even if the flag is forgotten.
#
# DELIBERATE BEHAVIOUR CHANGE (bug fix, applies to Allen too).
# The original built sweep_numbers as
#     int(ss_npz["sweep_number"][i]) for i in part
# where `part` indexes the POLARITY-FILTERED list `group`, while
# ss_npz["sweep_number"] is the FULL pulse array. For the "hyp" pass those
# indices point at the wrong pulses (typically at "dep" ones). The rewritten
# block carries absolute indices into all_pulses throughout, so
# sweep_numbers is now correct. This is the only field whose Allen output
# changes, it is provenance metadata not used by the fit, and the previous
# values were provably wrong -- see regression_allen_unchanged.py, which
# asserts every other field is bit-identical and demonstrates the old
# sweep_numbers were incorrect.

P1_SIG_OLD = '''def load_cell_from_archive(
    specimen_dir: "str | Path",
    *,
    n_avg_groups: int = 1,
    ls_max_amplitude_pA: float = 100.0,
    ls_fallback_amplitude_pA: float = 300.0,
    verbose: bool = True,
) -> "CellData":'''

P1_SIG_NEW = '''def _cluster_pulse_indices_by_amplitude(
    all_pulses: "List[Dict[str, Any]]",
    indices: "List[int]",
    *,
    tol_pA: float = 30.0,
) -> "List[List[int]]":
    """Single-linkage clustering of pulse indices on signed peak amplitude.

    Two pulses land in the same cluster iff their peak amplitudes differ by
    at most `tol_pA` (transitively). The default matches
    SQ_SUB_AMPLITUDE_TOL_PA, so an Allen pool -- every pulse within 30 pA of
    the nominal 200 pA -- yields exactly ONE cluster, while a pool holding
    genuinely distinct nominal amplitudes (50 / 100 / 200 pA) yields one
    cluster per amplitude.

    Returns absolute indices into `all_pulses`, clusters ordered by
    ascending amplitude, original order preserved within a cluster.
    """
    if not indices:
        return []
    order = sorted(indices, key=lambda k: float(all_pulses[k]["peak_pA"]))
    clusters: "List[List[int]]" = [[order[0]]]
    for k in order[1:]:
        prev = float(all_pulses[clusters[-1][-1]]["peak_pA"])
        if abs(float(all_pulses[k]["peak_pA"]) - prev) <= float(tol_pA):
            clusters[-1].append(k)
        else:
            clusters.append([k])
    return [sorted(c) for c in clusters]


def load_cell_from_archive(
    specimen_dir: "str | Path",
    *,
    n_avg_groups: int = 1,
    ls_max_amplitude_pA: float = 100.0,
    ls_fallback_amplitude_pA: float = 300.0,
    group_ss_by_amplitude: bool = False,
    ss_amplitude_tol_pA: float = 30.0,
    verbose: bool = True,
) -> "CellData":'''

P1_BODY_OLD = '''    ss_bundles: List[SweepBundle] = []
    for pol in ("dep", "hyp"):
        group = [w for w in all_pulses if w["polarity"] == pol]
        if not group:
            continue

        n_groups = max(1, min(n_avg_groups, len(group)))
        partitions = np.array_split(np.arange(len(group)), n_groups)

        for part in partitions:
            if len(part) == 0:
                continue

            v_stack = np.stack([group[i]["v"] for i in part])
            i_stack = np.stack([group[i]["i"] for i in part])
            t_ref = group[int(part[0])]["t"]
            stim_dur = float(
                np.mean([group[i]["stim_duration_s"] for i in part])
            )
            amp = float(np.mean([group[i]["peak_pA"] for i in part]))
            sr = group[int(part[0])]["sampling_rate_Hz"]

            ss_bundles.append(SweepBundle(
                polarity=pol,
                amplitude_pA=amp,
                t=t_ref,
                v_mV=v_stack.mean(axis=0),
                i_pA=i_stack.mean(axis=0),
                stim_onset_s=0.0,
                stim_duration_s=stim_dur,
                n_repeats_averaged=int(len(part)),
                sweep_numbers=sorted(set(
                    int(ss_npz["sweep_number"][i])
                    for i in part
                )) if ss_file.exists() and "sweep_number" in ss_npz else [],
                sampling_rate_Hz=sr,
                stimulus_name="Square Subthreshold",
            ))'''

P1_BODY_NEW = '''    ss_bundles: List[SweepBundle] = []
    for pol in ("dep", "hyp"):
        # Absolute indices into all_pulses throughout, so that any index used
        # to address ss_npz["sweep_number"] addresses the SAME pulse.
        idx_pol = [k for k, w in enumerate(all_pulses)
                   if w["polarity"] == pol]
        if not idx_pol:
            continue

        if group_ss_by_amplitude:
            clusters = _cluster_pulse_indices_by_amplitude(
                all_pulses, idx_pol, tol_pA=ss_amplitude_tol_pA)
        else:
            clusters = [idx_pol]
            amps = [float(all_pulses[k]["peak_pA"]) for k in idx_pol]
            if amps and (max(amps) - min(amps)) > ss_amplitude_tol_pA:
                msg = (
                    f"[load_archive] specimen {sid}: the {pol} pulse pool "
                    f"spans {min(amps):+.1f} to {max(amps):+.1f} pA, a range "
                    f"wider than ss_amplitude_tol_pA={ss_amplitude_tol_pA} "
                    f"pA. With group_ss_by_amplitude=False these will be "
                    f"AVERAGED INTO ONE BUNDLE labelled with their mean "
                    f"({float(np.mean(amps)):+.1f} pA), which corresponds to "
                    f"no experiment. Pass group_ss_by_amplitude=True.")
                warnings.warn(msg)
                if verbose:
                    print(msg)

        for cluster in clusters:
            n_groups = max(1, min(n_avg_groups, len(cluster)))
            partitions = np.array_split(np.arange(len(cluster)), n_groups)

            for part in partitions:
                if len(part) == 0:
                    continue
                sel = [int(cluster[j]) for j in part]

                v_stack = np.stack([all_pulses[k]["v"] for k in sel])
                i_stack = np.stack([all_pulses[k]["i"] for k in sel])
                t_ref = all_pulses[sel[0]]["t"]
                stim_dur = float(
                    np.mean([all_pulses[k]["stim_duration_s"] for k in sel])
                )
                amp = float(np.mean([all_pulses[k]["peak_pA"] for k in sel]))
                sr = all_pulses[sel[0]]["sampling_rate_Hz"]

                ss_bundles.append(SweepBundle(
                    polarity=pol,
                    amplitude_pA=amp,
                    t=t_ref,
                    v_mV=v_stack.mean(axis=0),
                    i_pA=i_stack.mean(axis=0),
                    stim_onset_s=0.0,
                    stim_duration_s=stim_dur,
                    n_repeats_averaged=int(len(sel)),
                    sweep_numbers=sorted(set(
                        int(ss_npz["sweep_number"][k])
                        for k in sel
                    )) if ss_file.exists() and "sweep_number" in ss_npz else [],
                    sampling_rate_Hz=sr,
                    stimulus_name="Square Subthreshold",
                ))'''


# ===========================================================================
#  PATCH 2 -- brief-pulse validation RMSD window is no longer hard-coded
# ===========================================================================
#
# WHY. Two RMSD paths exist and were not kept in sync. bundle_rmsd (the
# TRAINING loss, in passive_long_step_training.py) takes its window from
# --ss-window-ms. _rmsd_for_validation_bundle (the POST-FIT score that
# drives validation_status) hard-coded rmsd_window_s = (1e-3, 100e-3).
#
# For Allen the two agree closely enough not to matter. For Eyal the pulse
# is 2 ms and is followed by a bridge-balance artefact that only clears by
# about 3 ms post-onset and reaches roughly four times the physiological
# signal, which is exactly why the training window must start at 3 ms.
# Validation, starting at 1 ms, would integrate 2 ms of that artefact.
# Measured on the only held-out Eyal SS bundle, and assuming a PERFECT fit
# everywhere outside the artefact, this alone forces a validation RMSD floor
# of about 0.10 mV -- which sits just BELOW DEFAULT_VALID_RMSD_GOOD_MV = 0.2,
# the escape hatch that returns "good" regardless of the train/valid ratio.
# The failure mode is therefore an unconditional "good", not a visible
# failure.
#
# GUARD. The window is resolved at CALL time from a module global whose
# default (1.0, 100.0) ms reproduces the previous hard-coded values exactly.
# This mirrors how DEFAULT_DT_BRIEF_MS / DEFAULT_DT_LONG_MS are already
# handled, so integrate_long_step can set it once and every call path
# follows. An explicit ss_window_ms argument overrides it per call.

P2_GLOBAL_OLD = '''DEFAULT_VALID_RMSD_GOOD_MV = 0.2  # below this, accept regardless of ratio'''

P2_GLOBAL_NEW = '''DEFAULT_VALID_RMSD_GOOD_MV = 0.2  # below this, accept regardless of ratio

# Brief-pulse (Square Subthreshold) window used when SCORING a validation
# bundle, in ms with t = 0 at pulse onset. The default reproduces the value
# that was previously hard-coded inside _rmsd_for_validation_bundle. It is a
# module global, resolved at call time, so that integrate_long_step can align
# it with the TRAINING window (--ss-window-ms) in one place -- the two must
# agree, or validation scores a window the fit never optimised. Datasets
# whose pulse is followed by a bridge artefact (e.g. Eyal 2016: 2 ms pulse,
# artefact clearing at ~3 ms) require a start beyond that artefact.
DEFAULT_SS_VALIDATION_WINDOW_MS = (1.0, 100.0)'''

P2_FN_OLD = '''def _rmsd_for_validation_bundle(
    cell: PassiveCell,
    bundle: SweepBundle,
    v_rest_mV: float,
    valid_window_ms_after_onset: float,
) -> float:'''

P2_FN_NEW = '''def _rmsd_for_validation_bundle(
    cell: PassiveCell,
    bundle: SweepBundle,
    v_rest_mV: float,
    valid_window_ms_after_onset: float,
    ss_window_ms: "Optional[Tuple[float, float]]" = None,
) -> float:'''

P2_BODY_OLD = '''    if _is_brief_pulse(bundle):
        t_sim_s, v_sim = _simulate_square_subthreshold(cell, bundle, v_rest_mV)
        pre_window_s = (-10e-3, 0.0)
        rmsd_window_s = (1e-3, 100e-3)'''

P2_BODY_NEW = '''    if _is_brief_pulse(bundle):
        # Resolved at CALL time so integrate_long_step can align the
        # validation window with the training window globally.
        win_ms = (DEFAULT_SS_VALIDATION_WINDOW_MS if ss_window_ms is None
                  else tuple(ss_window_ms))
        t_sim_s, v_sim = _simulate_square_subthreshold(cell, bundle, v_rest_mV)
        pre_window_s = (-10e-3, 0.0)
        rmsd_window_s = (float(win_ms[0]) * 1e-3, float(win_ms[1]) * 1e-3)
        # _interp_to_grid uses np.interp, which CLAMPS silently outside the
        # simulated span. _simulate_square_subthreshold pads 100 ms after the
        # pulse, so a 2 ms pulse is simulated out to +102 ms; a window ending
        # beyond that would be scored against a flat clamped tail.
        if len(t_sim_s) and rmsd_window_s[1] > float(t_sim_s[-1]) + 1e-9:
            warnings.warn(
                f"validation SS window ends at {win_ms[1]:.1f} ms but the "
                f"simulation only reaches {float(t_sim_s[-1]) * 1e3:.1f} ms; "
                f"the tail would be silently clamped. Increase post_pad_ms "
                f"or shorten the window.")'''


# ===========================================================================
#  PATCH 3 -- morphology filename read from metadata
# ===========================================================================
#
# WHY. The loader hard-codes "reconstruction.swc". The Eyal morphologies are
# Neurolucida .asc and are archived as "morphology.asc". Unpatched, the
# loader either raises FileNotFoundError later at model-build time or -- if
# a stale reconstruction.swc from an earlier Allen run happens to be in the
# directory -- silently builds the WRONG cell.
#
# GUARD. Allen metadata.json has no "morphology_file" key, so the .get
# default reproduces the hard-coded path byte for byte.

P3_OLD = '''    sid = int(meta["specimen_id"])
    swc_path = specimen_dir / "reconstruction.swc"'''

P3_NEW = '''    sid = int(meta["specimen_id"])
    # Morphology filename comes from metadata when present. Allen archives
    # carry no "morphology_file" key, so this default is byte-identical to
    # the previously hard-coded path. Eyal archives set it to
    # "morphology.asc". NOTE the field on CellData is still called swc_path;
    # it now holds a path to EITHER format (see PassiveCell._import_swc).
    swc_path = specimen_dir / str(
        meta.get("morphology_file", "reconstruction.swc"))'''


# ===========================================================================
#  PATCH 4 -- Import3d reader dispatched on file suffix
# ===========================================================================
#
# WHY. _import_swc hard-codes h.Import3d_SWC_read(). SWC and Neurolucida ASC
# are different grammars parsed by different NEURON classes; feeding an .asc
# to the SWC reader does not degrade gracefully.
#
# GUARD. Any suffix that is not .asc takes the SWC branch, so every existing
# Allen call is unchanged. The dispatch below is the same three-line branch
# already exercised against all six Eyal morphologies.

P4_OLD = '''    def _import_swc(self) -> None:
        reader = h.Import3d_SWC_read()
        reader.input(str(self.swc_path))
        gui = h.Import3d_GUI(reader, 0)
        gui.instantiate(None)'''

P4_NEW = '''    def _import_swc(self) -> None:
        """Import the morphology, dispatching on file suffix.

        Despite the name (kept so that nothing downstream has to change),
        this handles both SWC and Neurolucida ASC. Anything that is not
        .asc takes the SWC branch, so Allen behaviour is unchanged.
        """
        suffix = Path(self.swc_path).suffix.lower()
        if suffix == ".asc":
            reader = h.Import3d_Neurolucida3()
        else:
            reader = h.Import3d_SWC_read()
        try:
            reader.quiet = 1
        except Exception:
            pass
        reader.input(str(self.swc_path))
        gui = h.Import3d_GUI(reader, 0)
        gui.instantiate(None)'''


# ===========================================================================
#  PATCH 5 -- require_long_square
# ===========================================================================
#
# WHY. load_cells_from_archive drops any cell whose long_square_subthreshold
# list is empty, unconditionally. The Eyal release contains no Long Square
# sweeps at all, so all six cells are dropped and the batch loader returns an
# empty list, leaving only a one-line "SKIPPED: no LS bundles" per cell in
# the log. The visible symptom is "the fit produced zero results", which
# points the investigation at the optimiser rather than at this gate.
#
# GUARD. Default True reproduces the current behaviour exactly. The Eyal path
# passes require_long_square=False.
#
# NOTE this deliberately does NOT add the parameter to the singular
# load_cell_from_archive, which performs no such check -- adding an unused
# argument there would be noise. The handoff suggested both; only the plural
# has a gate to guard.

P5_SIG_OLD = '''def load_cells_from_archive(
    archive_dir: "str | Path",
    *,
    n_avg_groups: int = 1,
    ls_max_amplitude_pA: float = 100.0,
    ls_fallback_amplitude_pA: float = 300.0,
    specimen_ids: "Optional[List[int]]" = None,
    max_cells: "Optional[int]" = None,
    verbose: bool = True,
) -> "List[CellData]":'''

P5_SIG_NEW = '''def load_cells_from_archive(
    archive_dir: "str | Path",
    *,
    n_avg_groups: int = 1,
    ls_max_amplitude_pA: float = 100.0,
    ls_fallback_amplitude_pA: float = 300.0,
    specimen_ids: "Optional[List[int]]" = None,
    max_cells: "Optional[int]" = None,
    require_long_square: bool = True,
    group_ss_by_amplitude: bool = False,
    ss_amplitude_tol_pA: float = 30.0,
    verbose: bool = True,
) -> "List[CellData]":'''

P5_CALL_OLD = '''            cd = load_cell_from_archive(
                d,
                n_avg_groups=n_avg_groups,
                ls_max_amplitude_pA=ls_max_amplitude_pA,
                ls_fallback_amplitude_pA=ls_fallback_amplitude_pA,
                verbose=verbose,
            )
            # Completeness check (mirrors load_complete_cells behaviour)
            if not cd.square_subthreshold:
                if verbose:
                    print(f"[load_archive]   SKIPPED: no SS bundles")
                n_failed += 1
                continue
            if not cd.long_square_subthreshold:
                if verbose:
                    print(f"[load_archive]   SKIPPED: no LS bundles")
                n_failed += 1
                continue'''

P5_CALL_NEW = '''            cd = load_cell_from_archive(
                d,
                n_avg_groups=n_avg_groups,
                ls_max_amplitude_pA=ls_max_amplitude_pA,
                ls_fallback_amplitude_pA=ls_fallback_amplitude_pA,
                group_ss_by_amplitude=group_ss_by_amplitude,
                ss_amplitude_tol_pA=ss_amplitude_tol_pA,
                verbose=verbose,
            )
            # Completeness check (mirrors load_complete_cells behaviour)
            if not cd.square_subthreshold:
                if verbose:
                    print(f"[load_archive]   SKIPPED: no SS bundles")
                n_failed += 1
                continue
            if require_long_square and not cd.long_square_subthreshold:
                if verbose:
                    print(f"[load_archive]   SKIPPED: no LS bundles")
                n_failed += 1
                continue
            if not require_long_square and not cd.long_square_subthreshold:
                if verbose:
                    print(f"[load_archive]   NOTE: no LS bundles; kept "
                          f"because require_long_square=False. Held-out "
                          f"validation will rest on SS bundles alone.")'''


# ===========================================================================
#  PATCH 6 -- wire the new parameters into the run_biological_fit.py CLI
# ===========================================================================
#
# WHY. Patches 1-5 add PARAMETERS to the library. The CLI never passes them,
# so an unmodified `run_biological_fit.py --archive-dir <eyal_archive>` still
# behaves exactly as before:
#
#   * load_cells_from_archive is called without require_long_square, so all
#     six Eyal cells are dropped and the run dies at
#         [ABORT] No cells loaded from archive: ...
#   * it is called without group_ss_by_amplitude, so cell 0603_cell08 would
#     be corrupted into a +116.7 pA bundle even if the cells did load
#   * build_neuron_model is called without axon_replacement, so it defaults
#     to the Hay two-section stub, while Eyal's .hoc deletes the axon with
#     NO replacement -- not a like-for-like comparison
#
# GUARD. Three new flags, all defaulting to the current behaviour:
#     --no-long-square           absent  -> require_long_square=True
#     --group-ss-by-amplitude    absent  -> False
#     --axon-replacement         default -> "hay_stub"
# An Allen invocation that does not name them is byte-for-byte unchanged.

P6_ARGS_OLD = '''    ap.add_argument("--n-avg-groups", type=int, default=1)
    ap.add_argument("--max-cells", type=int, default=None)'''

P6_ARGS_NEW = '''    ap.add_argument("--n-avg-groups", type=int, default=1)
    ap.add_argument("--max-cells", type=int, default=None)
    # -- dataset-shape flags (all default to the Allen behaviour) ----------
    ap.add_argument("--no-long-square", action="store_true",
                    help="Keep cells that have no Long Square sweeps. "
                         "REQUIRED for the Eyal 2016 archive, which has "
                         "none; without it every cell is silently dropped "
                         "and the run aborts with 'No cells loaded'.")
    ap.add_argument("--group-ss-by-amplitude", action="store_true",
                    help="Group Square-Subthreshold pulses by AMPLITUDE as "
                         "well as polarity before averaging. REQUIRED for "
                         "any archive holding several nominal amplitudes "
                         "(e.g. Eyal cell 0603_cell08 at +-50/100/200 pA), "
                         "otherwise they are averaged into one bundle "
                         "labelled with their mean.")
    ap.add_argument("--ss-amplitude-tol-pA", type=float, default=30.0,
                    help="Amplitude clustering tolerance for the above; "
                         "the default matches SQ_SUB_AMPLITUDE_TOL_PA so an "
                         "Allen pool forms exactly one cluster.")
    ap.add_argument("--axon-replacement", default="hay_stub",
                    choices=["hay_stub", "none"],
                    help="'hay_stub' (default) replaces the reconstructed "
                         "axon with two 30x1 um sections; 'none' deletes it "
                         "outright, matching Eyal's delete_axon().")'''

P6_LOAD_OLD = '''    cells_data = mono.load_cells_from_archive(
        args.archive_dir, n_avg_groups=args.n_avg_groups,
        specimen_ids=None, max_cells=args.max_cells, verbose=True)'''

P6_LOAD_NEW = '''    cells_data = mono.load_cells_from_archive(
        args.archive_dir, n_avg_groups=args.n_avg_groups,
        specimen_ids=None, max_cells=args.max_cells,
        require_long_square=not args.no_long_square,
        group_ss_by_amplitude=args.group_ss_by_amplitude,
        ss_amplitude_tol_pA=args.ss_amplitude_tol_pA,
        verbose=True)'''

P6_BUILD1_OLD = '''            cell = mono.build_neuron_model(cd.swc_path, F=args.F)'''

P6_BUILD1_NEW = '''            cell = mono.build_neuron_model(
                cd.swc_path, F=args.F,
                axon_replacement=args.axon_replacement)'''

P6_BUILD2_OLD = '''            pc = mono.build_neuron_model(cd.swc_path, F=float(fr.F))'''

P6_BUILD2_NEW = '''            pc = mono.build_neuron_model(
                cd.swc_path, F=float(fr.F),
                axon_replacement=args.axon_replacement)'''


# ===========================================================================
#  PATCH 7 -- "not_evaluated" is not the same verdict as "failed"
# ===========================================================================
#
# WHY. _classify_fit returns "failed" whenever valid_rmsd is not finite. That
# conflates two different situations:
#
#   (a) validation ran and the model could not reproduce the held-out data
#       -> "failed" is the correct verdict
#   (b) there was NOTHING to validate against, so no verdict is possible
#       -> "failed" is a false negative
#
# Case (b) never arises on Allen data, where every cell carries Long Square
# sweeps. It is the NORMAL case for Eyal 2016, which has no Long Square data
# at all and only one cell (0603_cell08) with an opposite-polarity brief
# pulse to hold out. Measured on a real run: the fit converged cleanly
# (Cm=0.450 vs published 0.43, train RMSD 0.0443 mV against a 0.0404 mV
# noise floor) and was still labelled "failed", purely because
# opt_in.validation_bundles was empty.
#
# That is not cosmetic. Phase 3 and rebuild_neuron_cells_for_phase3 both gate
# on status in ("good", "to_refine"), and aggregate_population filters on it,
# so every downstream stage silently discards the entire dataset.
#
# GUARD. _classify_fit gains n_validation_bundles, defaulting to -1 meaning
# "not reported", which reproduces the current behaviour exactly. Only an
# explicit 0 produces the new label. Allen never passes 0, because Allen
# cells always have validation bundles, so no Allen result can change.

P7_FN_OLD = '''def _classify_fit(
    train_rmsd: float,
    valid_rmsd: float,
    k_good: float,
    k_fail: float,
    train_rmsd_fail_mV: float,
    valid_rmsd_good_mV: float = DEFAULT_VALID_RMSD_GOOD_MV,
) -> str:'''

P7_FN_NEW = '''def _classify_fit(
    train_rmsd: float,
    valid_rmsd: float,
    k_good: float,
    k_fail: float,
    train_rmsd_fail_mV: float,
    valid_rmsd_good_mV: float = DEFAULT_VALID_RMSD_GOOD_MV,
    n_validation_bundles: int = -1,
) -> str:'''

P7_BODY_OLD = '''    if not np.isfinite(train_rmsd) or train_rmsd > train_rmsd_fail_mV:
        return "failed"
    if not np.isfinite(valid_rmsd):
        return "failed"'''

P7_BODY_NEW = '''    if not np.isfinite(train_rmsd) or train_rmsd > train_rmsd_fail_mV:
        return "failed"
    if not np.isfinite(valid_rmsd):
        # Distinguish "validation ran and the model lost" from "there was
        # nothing to validate against". n_validation_bundles == 0 is the
        # second case and cannot be a verdict on fit quality; -1 means the
        # caller did not report a count, which preserves the old behaviour.
        if n_validation_bundles == 0:
            return "not_evaluated"
        return "failed"'''

P7_CALL_OLD = '''    status = _classify_fit(
        train_rmsd, valid_rmsd, k_good, k_fail, train_rmsd_fail_mV,
        valid_rmsd_good_mV=valid_rmsd_good_mV,
    )'''

P7_CALL_NEW = '''    status = _classify_fit(
        train_rmsd, valid_rmsd, k_good, k_fail, train_rmsd_fail_mV,
        valid_rmsd_good_mV=valid_rmsd_good_mV,
        n_validation_bundles=len(opt_in.validation_bundles),
    )
    if status == "not_evaluated":
        print(f"[fit_one_cell]   NO held-out data for specimen {sid}: the "
              f"fit is UNVALIDATED, not failed. Train RMSD "
              f"{train_rmsd:.4f} mV. Treat the parameters as provisional.")'''

P7_DOC_OLD = '''    validation_status: str            # "good" | "to_refine" | "failed"'''

P7_DOC_NEW = '''    validation_status: str            # "good" | "to_refine" | "failed"
                                      # | "not_evaluated" (no held-out data
                                      #   existed; NOT a quality verdict)'''

P7_P3_OLD = '''    statuses: Sequence[str] = ("good", "to_refine"),'''

P7_P3_NEW = '''    statuses: Sequence[str] = ("good", "to_refine", "not_evaluated"),'''

P7_GATE_OLD = '''    status = classify_fn(train_abs, valid_abs, k_good, k_fail, train_fail_mV,
                         valid_rmsd_good_mV=valid_good_mV)
    return train_abs, valid_abs, ratio_abs, status'''

P7_GATE_NEW = '''    # Report how many validation bundles actually existed, so the classifier
    # can tell "validation ran and lost" from "nothing to validate against".
    # An empty valid_bundle_rmsds_mV is the second case; without this the
    # absolute gate re-labels a perfectly good unvalidated fit as "failed",
    # overwriting the "not_evaluated" that fit_one_cell just assigned.
    try:
        status = classify_fn(train_abs, valid_abs, k_good, k_fail,
                             train_fail_mV,
                             valid_rmsd_good_mV=valid_good_mV,
                             n_validation_bundles=len(va))
    except TypeError:
        # classify_fn predates patch 7 (or is a test double): fall back.
        status = classify_fn(train_abs, valid_abs, k_good, k_fail,
                             train_fail_mV, valid_rmsd_good_mV=valid_good_mV)
    return train_abs, valid_abs, ratio_abs, status'''


P7_RUNNER_OLD = '''        if (fr.validation_status not in ("good", "to_refine")
                or getattr(fr, "gp_result", None) is None):'''

P7_RUNNER_NEW = '''        if (fr.validation_status not in ("good", "to_refine",
                                        "not_evaluated")
                or getattr(fr, "gp_result", None) is None):'''


# ===========================================================================
#  COMPANION EDIT (passive_long_step_training.py) -- carry the window across
# ===========================================================================

P2C_OLD = '''    mono.DEFAULT_DT_BRIEF_MS = float(DEFAULT_DT_BRIEF_MS)
    mono.DEFAULT_DT_LONG_MS = float(DEFAULT_DT_LONG_MS)'''

P2C_NEW = '''    mono.DEFAULT_DT_BRIEF_MS = float(DEFAULT_DT_BRIEF_MS)
    mono.DEFAULT_DT_LONG_MS = float(DEFAULT_DT_LONG_MS)
    # Align the VALIDATION brief-pulse window with the TRAINING one. Without
    # this, _rmsd_for_validation_bundle keeps its (1, 100) ms default while
    # training uses --ss-window-ms, so validation scores a window the fit
    # never optimised -- and for a dataset with a post-pulse bridge artefact
    # it scores the artefact.
    mono.DEFAULT_SS_VALIDATION_WINDOW_MS = tuple(float(x) for x in ss_window_ms)'''


PATCHES: List[Dict[str, Any]] = [
    {"n": 1, "file": MONOLITH, "name": "SS averaging: cluster helper + signature",
     "old": P1_SIG_OLD, "new": P1_SIG_NEW},
    {"n": 1, "file": MONOLITH, "name": "SS averaging: grouping body",
     "old": P1_BODY_OLD, "new": P1_BODY_NEW},
    {"n": 2, "file": MONOLITH, "name": "validation window: module global",
     "old": P2_GLOBAL_OLD, "new": P2_GLOBAL_NEW},
    {"n": 2, "file": MONOLITH, "name": "validation window: signature",
     "old": P2_FN_OLD, "new": P2_FN_NEW},
    {"n": 2, "file": MONOLITH, "name": "validation window: body",
     "old": P2_BODY_OLD, "new": P2_BODY_NEW},
    {"n": 2, "file": LONGSTEP, "name": "validation window: companion global",
     "old": P2C_OLD, "new": P2C_NEW},
    {"n": 3, "file": MONOLITH, "name": "morphology filename from metadata",
     "old": P3_OLD, "new": P3_NEW},
    {"n": 4, "file": MONOLITH, "name": "Import3d suffix dispatch",
     "old": P4_OLD, "new": P4_NEW},
    {"n": 5, "file": MONOLITH, "name": "require_long_square: signature",
     "old": P5_SIG_OLD, "new": P5_SIG_NEW},
    {"n": 5, "file": MONOLITH, "name": "require_long_square: gate",
     "old": P5_CALL_OLD, "new": P5_CALL_NEW},
    {"n": 6, "file": RUNNER, "name": "CLI: new dataset-shape flags",
     "old": P6_ARGS_OLD, "new": P6_ARGS_NEW},
    {"n": 6, "file": RUNNER, "name": "CLI: pass flags to the archive loader",
     "old": P6_LOAD_OLD, "new": P6_LOAD_NEW},
    {"n": 6, "file": RUNNER, "name": "CLI: axon_replacement (fit path)",
     "old": P6_BUILD1_OLD, "new": P6_BUILD1_NEW},
    {"n": 6, "file": RUNNER, "name": "CLI: axon_replacement (Phase 3 path)",
     "old": P6_BUILD2_OLD, "new": P6_BUILD2_NEW},
    {"n": 7, "file": MONOLITH, "name": "not_evaluated: classifier signature",
     "old": P7_FN_OLD, "new": P7_FN_NEW},
    {"n": 7, "file": MONOLITH, "name": "not_evaluated: classifier branch",
     "old": P7_BODY_OLD, "new": P7_BODY_NEW},
    {"n": 7, "file": MONOLITH, "name": "not_evaluated: report bundle count",
     "old": P7_CALL_OLD, "new": P7_CALL_NEW},
    {"n": 7, "file": MONOLITH, "name": "not_evaluated: dataclass docstring",
     "old": P7_DOC_OLD, "new": P7_DOC_NEW},
    {"n": 7, "file": MONOLITH, "name": "not_evaluated: Phase 3 rebuild gate",
     "old": P7_P3_OLD, "new": P7_P3_NEW},
    {"n": 7, "file": RUNNER, "name": "not_evaluated: Phase 3 subset gate",
     "old": P7_RUNNER_OLD, "new": P7_RUNNER_NEW},
    {"n": 7, "file": RUNNER, "name": "not_evaluated: absolute-mV gate",
     "old": P7_GATE_OLD, "new": P7_GATE_NEW},
]


def _status(text: str, p: Dict[str, Any]) -> str:
    """Classify a patch against the current file text.

    IMPORTANT: test for `new` BEFORE `old`. Two of these patches append to
    their anchor rather than rewriting it, so `old` is a SUBSTRING of `new`
    and remains present after a successful apply. Checking `old` first would
    report an applied patch as "ready" and a second --apply would insert the
    block twice.
    """
    n_new = text.count(p["new"])
    if n_new == 1:
        return "applied"
    if n_new > 1:
        return "APPLIED %d TIMES (file is damaged; --revert)" % n_new
    n_old = text.count(p["old"])
    if n_old == 1:
        return "ready"
    if n_old == 0:
        return "ANCHOR MISSING"
    return "ANCHOR AMBIGUOUS (%d matches)" % n_old


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code-dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    args = ap.parse_args(argv)

    code_dir = Path(args.code_dir)
    for fn in {p["file"] for p in PATCHES}:
        if not (code_dir / fn).is_file():
            print("missing: %s" % (code_dir / fn))
            return 1

    if args.revert:
        n = 0
        for fn in {p["file"] for p in PATCHES}:
            bak = code_dir / (fn + ".orig.bak")
            if bak.is_file():
                shutil.copy2(bak, code_dir / fn)
                print("reverted %s" % fn)
                n += 1
        print("reverted %d file(s)" % n)
        return 0

    texts = {fn: (code_dir / fn).read_text(encoding="utf-8")
             for fn in {p["file"] for p in PATCHES}}

    print("=" * 74)
    n_bad = n_ready = n_done = 0
    for p in PATCHES:
        st = _status(texts[p["file"]], p)
        print("patch %d  %-45s %s" % (p["n"], p["name"], st))
        if st == "ready":
            n_ready += 1
        elif st == "applied":
            n_done += 1
        else:
            n_bad += 1
    print("=" * 74)
    print("ready=%d  already-applied=%d  problems=%d" % (n_ready, n_done, n_bad))

    if n_bad:
        print("\nRefusing to touch anything: an anchor did not match exactly "
              "once.\nThe target file is not the revision these patches were "
              "written against.")
        return 1
    if args.check:
        print("\n--check only; nothing written.")
        return 0
    if n_ready == 0:
        print("\nNothing to do; all patches already applied.")
        return 0

    for fn in texts:
        bak = code_dir / (fn + ".orig.bak")
        if not bak.exists():
            shutil.copy2(code_dir / fn, bak)
            print("backup -> %s" % bak.name)

    for p in PATCHES:
        # Drive the edit from the STATUS, not from a bare `old in text` test:
        # for the append-style patches `old` survives the replacement, so a
        # substring test would re-apply them on every run.
        if _status(texts[p["file"]], p) == "ready":
            texts[p["file"]] = texts[p["file"]].replace(p["old"], p["new"], 1)

    for fn, t in texts.items():
        (code_dir / fn).write_text(t, encoding="utf-8", newline="\n")
        print("wrote %s" % fn)

    print("\nAll patches applied. Now run:")
    print("  python3 -m py_compile %s/*.py" % code_dir)
    print("  python3 regression_allen_unchanged.py --help")
    return 0


if __name__ == "__main__":
    sys.exit(main())
