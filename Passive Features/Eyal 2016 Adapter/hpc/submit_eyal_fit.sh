#!/bin/bash

#PBS -S /bin/bash
#PBS -N "eyal_passive_fit"
#PBS -q cpu
#PBS -l select=1:ncpus=1,walltime=48:00:00
#PBS -k eo

##########################################################################
# Eyal et al. (2016) human L2/3 passive fit -- ONE PBS job, all six cells.
#
# This is submit_biological_fit.sh specialised for the Eyal archive. It is a
# SEPARATE file rather than a flag on the Allen script because five of the
# defaults below are wrong for Allen and would silently corrupt an Allen run
# if they leaked into it.
#
# -- Dispatch ------------------------------------------------------------
#     qsub submit_eyal_fit.sh
#     qsub -v RUN_TAG=nseg_test submit_eyal_fit.sh
#     qsub -v FIT_TARGET=both,RUN_TAG=both submit_eyal_fit.sh
#
# There is no GROUP variable: the Eyal dataset is a single cohort of six
# cells, so the archive directory is named directly.
#
# -- Why each non-Allen default is what it is ----------------------------
#
#   --no-long-square         The release contains NO Long Square sweeps.
#                            Without this every cell is dropped and the run
#                            aborts with "No cells loaded from archive".
#
#   --group-ss-by-amplitude  Cell 0603_cell08 carries +-50/100/200 pA. The
#                            loader groups by polarity alone by default and
#                            would average them into one bundle labelled
#                            +116.7 pA -- an amplitude at which no
#                            experiment was performed.
#
#   --ss-window-ms 3.0,102.0 The 2 ms pulse is followed by a bridge-balance
#                            artefact reaching ~4x the physiological signal
#                            and clearing only by 3 ms. Every record ends
#                            exactly 102 ms after onset (= pulse + 100 ms),
#                            and the simulator pads to the same point, so
#                            this window matches Eyal's own [3,102] ms with
#                            zero extrapolation.
#
#   --n-long-train 0         Nothing to fold in; there are no long steps.
#
#   --axon-replacement none  Eyal's delete_axon() removes the axon with NO
#                            replacement. The pipeline default (Hay two-
#                            section stub) would not be like-for-like.
#
#   --fit-target dep         FIVE of the six cells have ONLY a +200 pA
#                            depolarising trace. The pipeline default 'hyp'
#                            would leave them with an empty training set.
#                            'dep' is more I_h-contaminated than 'hyp' would
#                            be -- record that as a caveat, it is forced by
#                            the data, not chosen.
#
#   --skip-phase2p5          Phase 2.5 profiles Ra and applies a cohort
#                            floor/gate policy; with six cells and no LS
#                            data the arithmetic is not meaningful. Revisit
#                            only for a specific question.
#
#   --phase3-subset none     The bootstrap resamples individual pre-average
#                            pulses. Eyal distributes only the 50-sweep
#                            AVERAGES, so there is exactly one pulse per
#                            amplitude per cell and the nonparametric
#                            bootstrap is degenerate.
#
# -- Expected outcome ----------------------------------------------------
# Every cell will report validation_status = "not_evaluated", because there
# is no held-out data for five of the six. That is NOT a failure verdict --
# it means no verdict was possible. Judge the run on the per-cell deviation
# of Cm_hat from the published Cm*, in comparison_targets.csv.
#
# REQUIRES patches 1-7 (patch_eyal_support.py). Without patch 6 the four
# dataset-shape flags below do not exist and qsub fails immediately with an
# argparse error; without patch 5 the run aborts with "No cells loaded".
##########################################################################

# --- USER CONFIG ------------------------------------------------------------
CODE_DIR="/davinci-1/home/ldellamea/Human Neurons Fitting"
ARCHIVE_DIR="/davinci-1/home/ldellamea/Human Neurons Fitting/eyal_archive"
OUTPUT_BASE="/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs"
RUN_TAG="${RUN_TAG:-eyal2016}"
CONDA_ENV="prova"
ENTRYPOINT="$CODE_DIR/run_biological_fit.py"

# --- Data / fit -------------------------------------------------------------
N_AVG_GROUPS=1
FIT_TARGET="${FIT_TARGET:-dep}"       # dep | hyp | both  -- see note above
F_FACTOR="${F_FACTOR:-1.9}"           # matches Eyal's F_Spines exactly
N_CALLS="${N_CALLS:-100}"             # validated benchmark budget
N_INITIAL="${N_INITIAL:-50}"
MAX_CELLS="${MAX_CELLS:-}"            # empty = all six

# --- Eyal-specific dataset shape (patch 6) ----------------------------------
NO_LONG_SQUARE=1                      # 1 = pass --no-long-square (REQUIRED)
GROUP_SS_BY_AMPLITUDE=1               # 1 = pass --group-ss-by-amplitude
SS_AMPLITUDE_TOL_PA=30.0
AXON_REPLACEMENT="${AXON_REPLACEMENT:-none}"   # none | hay_stub

# --- Loss / window ----------------------------------------------------------
N_LONG_TRAIN=0                        # no long steps exist
SS_WINDOW_MS="${SS_WINDOW_MS:-3.0,102.0}"
SS_T0_MS=""                           # empty => window start (3.0 ms)
SS_TIME_WEIGHT="exp"
TAU_W_GRID_MS="${TAU_W_GRID_MS:-5.0}"
SWEEP_RHO=0.5
SWEEP_N_GRID=15
R_IN_TARGET="peak"
WEIGHTING="relative"
LS_WINDOW_MS=150.0                    # unused (no LS bundles) but required
LS_DEFLECTION_CAP_MV=12.0             # unused
MAX_SAG_AMPLITUDE_MV=-1               # disabled

# --- Integration time step (ENFORCED) ---------------------------------------
DT_BRIEF_MS="${DT_BRIEF_MS:-0.025}"
DT_LONG_MS="${DT_LONG_MS:-0.025}"

# --- Stages disabled, with reasons in the header ----------------------------
SKIP_PHASE2P5="${SKIP_PHASE2P5:-1}"
PHASE3_SUBSET="${PHASE3_SUBSET:-none}"
# ----------------------------------------------------------------------------

case "$RUN_TAG" in
    ""|*[!A-Za-z0-9_.-]*)
        echo "[FATAL] RUN_TAG must be non-empty and contain only letters," >&2
        echo "        digits, '_', '.', '-'  (got: '$RUN_TAG')" >&2
        exit 4
        ;;
esac

OUTPUT_DIR="$OUTPUT_BASE/$RUN_TAG"

if [ ! -d "$ARCHIVE_DIR" ]; then
    echo "[FATAL] Archive directory does not exist: $ARCHIVE_DIR" >&2
    exit 3
fi
N_SPEC=$(find "$ARCHIVE_DIR" -maxdepth 1 -type d -name 'specimen_*' | wc -l)
if [ "$N_SPEC" -ne 6 ]; then
    echo "[WARN] expected 6 specimen_* dirs in $ARCHIVE_DIR, found $N_SPEC" >&2
fi

cd "$PBS_O_WORKDIR" || exit 6

source "$(conda info --base)/etc/profile.d/conda.sh" \
    || { echo "[FATAL] cannot source conda.sh" >&2; exit 5; }
conda activate "$CONDA_ENV" \
    || { echo "[FATAL] cannot activate conda env '$CONDA_ENV'" >&2; exit 5; }

mkdir -p "$OUTPUT_DIR"

echo "Running on node:        $(hostname)"
echo "Job ID:                 $PBS_JOBID"
echo "Run tag:                $RUN_TAG"
echo "Dataset:                Eyal et al. 2016, human L2/3 (n=$N_SPEC)"
echo "Python:                 $(which python)   | conda: $CONDA_DEFAULT_ENV"
echo "Code dir:               $CODE_DIR"
echo "Archive dir:            $ARCHIVE_DIR"
echo "Output dir:             $OUTPUT_DIR"
echo "Fit:                    target=$FIT_TARGET  F=$F_FACTOR  n_calls=$N_CALLS  n_initial=$N_INITIAL"
echo "Eyal shape:             no_long_square=$NO_LONG_SQUARE  group_ss_by_amp=$GROUP_SS_BY_AMPLITUDE  axon=$AXON_REPLACEMENT"
echo "Window:                 ss=[$SS_WINDOW_MS] ms  tau_w grid=[$TAU_W_GRID_MS] ms  weight=$SS_TIME_WEIGHT"
echo "dt (ENFORCED):          brief=${DT_BRIEF_MS}ms  long=${DT_LONG_MS}ms"
echo "Phase 2.5:              skip=$SKIP_PHASE2P5     Phase 3: subset='$PHASE3_SUBSET'"
echo "-----------------------------------------"

ARGS=(
    --archive-dir         "$ARCHIVE_DIR"
    --output-dir          "$OUTPUT_DIR"
    --code-dir            "$CODE_DIR"
    --n-avg-groups        "$N_AVG_GROUPS"
    --fit-target          "$FIT_TARGET"
    --F                   "$F_FACTOR"
    --n-calls             "$N_CALLS"
    --n-initial           "$N_INITIAL"
    --axon-replacement    "$AXON_REPLACEMENT"
    --ss-amplitude-tol-pA "$SS_AMPLITUDE_TOL_PA"
    --n-long-train        "$N_LONG_TRAIN"
    --ls-deflection-cap   "$LS_DEFLECTION_CAP_MV"
    "--max-sag-amplitude-mV=$MAX_SAG_AMPLITUDE_MV"
    --r-in-target         "$R_IN_TARGET"
    --weighting           "$WEIGHTING"
    --ss-window-ms        "$SS_WINDOW_MS"
    --ss-time-weight      "$SS_TIME_WEIGHT"
    --ls-window-ms        "$LS_WINDOW_MS"
    --tau-w-grid-ms       "$TAU_W_GRID_MS"
    --sweep-rho           "$SWEEP_RHO"
    --sweep-n-grid        "$SWEEP_N_GRID"
    --dt-brief-ms         "$DT_BRIEF_MS"
    --dt-long-ms          "$DT_LONG_MS"
    --phase3-subset       "$PHASE3_SUBSET"
)

[ "$NO_LONG_SQUARE" = "1" ]        && ARGS+=(--no-long-square)
[ "$GROUP_SS_BY_AMPLITUDE" = "1" ] && ARGS+=(--group-ss-by-amplitude)
[ -n "$SS_T0_MS" ]                 && ARGS+=(--ss-t0-ms "$SS_T0_MS")
[ -n "$MAX_CELLS" ]                && ARGS+=(--max-cells "$MAX_CELLS")
[ "$SKIP_PHASE2P5" = "1" ]         && ARGS+=(--skip-phase2p5)

python "$ENTRYPOINT" "${ARGS[@]}"
status=$?

conda deactivate
echo "[done] Eyal run '$RUN_TAG' exited with status $status"
echo
echo "Compare phase2_results.csv against comparison_targets.csv in the"
echo "archive root. The primary quantity is the per-cell relative deviation"
echo "in Cm: (Cm_hat - Cm_star) / Cm_star. Eyal's own resampling analysis"
echo "gives a mean relative STATISTICAL error of 6.5% in Cm for these cells,"
echo "so agreement at that level is the target, and a systematic offset of"
echo "the same sign across all six would indicate a convention mismatch"
echo "rather than optimiser noise."
sleep 5s
exit $status
