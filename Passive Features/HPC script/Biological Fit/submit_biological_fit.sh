#!/bin/bash

#PBS -S /bin/bash
#PBS -N "bio_passive_fit"
#PBS -q cpu
#PBS -l select=1:ncpus=1,walltime=100:00:00
#PBS -k eo

##########################################################################
# Biological passive-fit pipeline -- ONE (layer x type) GROUP per PBS job.
#
# This is the REAL-DATA counterpart of submit_synth_benchmark.sh. It runs the
# SAME fitting algorithm validated on synthetic data (the monolith patched with
# the multi-protocol / relative / time-weighted loss + long-step training, and
# the two-pass auto-tau_w loop), via the entrypoint run_biological_fit.py.
#
# Differences vs the old submit_passive_fit.sh (which called the monolith's own
# __main__ and therefore used the OLD single-protocol mV loss with none of the
# report's features):
#   * calls run_biological_fit.py, which PATCHES the loss + drives the sweep;
#   * forwards the new-feature flags (n-long-train, ls-deflection-cap,
#     r-in-target, weighting, ss-window, ss-time-weight, tau-w-grid, ...);
#   * NO nrnivmodl step -- the biological FIT model is purely passive (only
#     `pas`; PassiveCell inserts nothing else), so no .mod compilation is
#     needed. (Ih.mod exists in the synthetic folder ONLY to GENERATE the
#     I_h-contaminated synthetic data; it is irrelevant to fitting real cells.)
#
# Sequential, one cell at a time (NEURON's global section list is wiped between
# cells; N_WORKERS is hard-coded to 1 inside the monolith).
#
# -- Dispatch ------------------------------------------------------------------
#     qsub -v GROUP=L3_exc submit_biological_fit.sh
#     qsub -v GROUP=L3_exc,RUN_TAG=freeRa submit_biological_fit.sh
# Use submit_all_groups_biological.sh to fan out one job per group.
#
# -- Multiple concurrent runs --------------------------------------------------
# RUN_TAG isolates the outputs of independent runs so two configurations can be
# in flight at the same time WITHOUT editing this file between submissions:
#     <OUTPUT_BASE>/<RUN_TAG>/<GROUP>/...
# Every artefact the pipeline writes goes under --output-dir, and CODE_DIR /
# ARCHIVE_ROOT are read-only during a run, so runs with different RUN_TAGs
# cannot collide. RUN_TAG also goes into the PBS job name, so the logs in $HOME
# are distinguishable by run rather than only by job id.
#
# EDIT THE "USER CONFIG" BLOCK BELOW BEFORE SUBMITTING.
##########################################################################

# --- USER CONFIG ------------------------------------------------------------
# Absolute paths recommended (jobs do not inherit $PWD reliably).
# The per-job archive/output dirs are <ROOT>/<GROUP>.
CODE_DIR="/davinci-1/home/ldellamea/Human Neurons Fitting"
ARCHIVE_ROOT="/davinci-1/home/ldellamea/Human Neurons Fitting"
# Outputs land in <OUTPUT_BASE>/<RUN_TAG>/<GROUP>. Override RUN_TAG at
# submission time (qsub -v RUN_TAG=...) to keep two runs apart; the default
# tag "default" reproduces a single-run setup.
OUTPUT_BASE="/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs"
RUN_TAG="${RUN_TAG:-default}"
CONDA_ENV="prova"
ENTRYPOINT="$CODE_DIR/run_biological_fit.py"

# The entrypoint imports these from CODE_DIR (must all be present there):
#   run_biological_fit.py  passive_fitting_hpc_fixed.py
#   cm_profile_sweep.py    passive_long_step_training.py

# --- Data / fit (run-level) -------------------------------------------------
N_AVG_GROUPS=1            # sweep-average groups per polarity
FIT_TARGET="hyp"         # dep | hyp | both  (hyp minimises I_h)
# Spine-area correction (Eyal L2/3 default). Overridable per-group by the
# wrapper via `qsub -v F_FACTOR=...`; the value below is the manual fallback.
F_FACTOR="${F_FACTOR:-1.9}"
# GP optimiser budget per cell. The validated synthetic benchmark used 100/50;
# for a production biological run you may raise these (e.g. 200/120) for a more
# thorough search at higher wall-time cost.
N_CALLS=100
N_INITIAL=50
MAX_CELLS=""             # cap cells for a test run (empty = all in the group)

# --- Two-pass auto-tau_w + multi-protocol loss (the report's features) ------
N_LONG_TRAIN=2           # smallest-|amp| hyp long steps folded into TRAINING
LS_DEFLECTION_CAP_MV=12.0  # I_h deflection guard for long-step admission (mV)
MAX_SAG_AMPLITUDE_MV=-1   # optional sag-gated I_h guard (mV); <0 disables.
                          #   Needs sag_ratio on CellData (NOT carried by the
                          #   current loader) -- falls back to the mV cap if NaN.
R_IN_TARGET="peak"       # peak (pre-sag passive R_in) | steady (sagged R_in)
WEIGHTING="relative"     # cross-bundle loss weighting (relative|equal|custom)
SS_WINDOW_MS="0.5,100.0"   # SS window (start=pulse offset -> the C_m choice)
SS_T0_MS=""              # empty => SS window start
SS_TIME_WEIGHT="exp"     # exp | gauss | none
LS_WINDOW_MS=150.0       # LS RMSD window length from onset (ms)
TAU_W_GRID_MS="5.0"      # per-cell sweep grid; winner = sharpest HW_rho.
                          #   Single point (5.0) reproduces the validated
                          #   benchmark. A multi-point grid (e.g. "3,5,7")
                          #   actually selects tau_w per cell (~3x cost/point).
SWEEP_RHO=0.5            # relative-rise threshold for HW_rho
SWEEP_N_GRID=15          # log C_m grid points for the profile

# --- Phase 2.5 (MANDATORY here: fix Ra per group + refit Cm,Rm) -------------
SKIP_PHASE2P5="${SKIP_PHASE2P5:-0}"   # 1 = legacy free-Ra diagnostic; 0 = standard
N_FLOOR="${N_FLOOR:-4}"               # min qualifying cells for cohort-median Ra
N_RA_PROFILE="${N_RA_PROFILE:-50}"    # Ra grid points for the RMSD-vs-Ra profile

# --- Phase 3 (bootstrap CIs) -- SUBSET selectable ---------------------------
# Which cells in the group get bootstrapped:
#   "none"      -> none
#   "all"       -> every fittable cell (expensive)
#   "first:N"   -> first N fittable cells
#   "frac:F"    -> ~F fraction of fittable cells (deterministic by specimen_id)
# COST NOTE: Phase 3 dominates wall time. Each bootstrapped cell costs
# BOOTSTRAP_B * BOOTSTRAP_N_CALLS loss evaluations (here 200*70 = 14,000),
# vs ~1,000 for the whole of Phase 2. Shrink B or the subset first if a group
# risks the walltime.
PHASE3_SUBSET="${PHASE3_SUBSET:-frac:0.5}"
BOOTSTRAP_B=200                  # replicates (>=200 for stable BCa)
BOOTSTRAP_MODE="nonparametric"   # nonparametric (resample real pulses) | parametric
NOISE_MODE="block"               # iid | ar1 | block  (parametric only)
BOOTSTRAP_N_CALLS=70
BOOTSTRAP_N_INITIAL=20
# ----------------------------------------------------------------------------

# --- Resolve per-job paths from $GROUP and $RUN_TAG --------------------------
if [ -z "${GROUP:-}" ]; then
    echo "[FATAL] \$GROUP is not set. Submit with:" >&2
    echo "    qsub -v GROUP=<layer_type> submit_biological_fit.sh" >&2
    echo "    e.g. qsub -v GROUP=L3_exc submit_biological_fit.sh" >&2
    exit 2
fi

# RUN_TAG becomes a path component AND part of the PBS job name, and it travels
# through the comma-separated `qsub -v` list -- so reject anything that would
# break any of those (spaces, commas, slashes, quotes).
case "$RUN_TAG" in
    ""|*[!A-Za-z0-9_.-]*)
        echo "[FATAL] RUN_TAG must be non-empty and contain only letters," >&2
        echo "        digits, '_', '.', '-'  (got: '$RUN_TAG')" >&2
        exit 4
        ;;
esac

OUTPUT_ROOT="$OUTPUT_BASE/$RUN_TAG"
ARCHIVE_DIR="$ARCHIVE_ROOT/$GROUP"
OUTPUT_DIR="$OUTPUT_ROOT/$GROUP"

if [ ! -d "$ARCHIVE_DIR" ]; then
    echo "[FATAL] Archive directory does not exist: $ARCHIVE_DIR" >&2
    exit 3
fi
# ----------------------------------------------------------------------------

cd "$PBS_O_WORKDIR"

# Activate the conda env (it supplies python + NEURON). We do NOT
# `module load python` -- on this cluster that produced Lmod errors; conda's
# python is the one we want. No nrnivmodl is needed (passive-only fit).
source "$(conda info --base)/etc/profile.d/conda.sh" \
    || { echo "[FATAL] cannot source conda.sh" >&2; exit 5; }
conda activate "$CONDA_ENV" \
    || { echo "[FATAL] cannot activate conda env '$CONDA_ENV'" >&2; exit 5; }

mkdir -p "$OUTPUT_DIR"

# Sanity prints
echo "Running on node:        $(hostname)"
echo "Job ID:                 $PBS_JOBID"
echo "Run tag:                $RUN_TAG"
echo "Group:                  $GROUP"
echo "Working dir:            $PBS_O_WORKDIR"
echo "Python:                 $(which python)   | conda: $CONDA_DEFAULT_ENV"
echo "Entrypoint:             $ENTRYPOINT"
echo "Code dir:               $CODE_DIR"
echo "Archive dir (this job): $ARCHIVE_DIR"
echo "Output dir (this job):  $OUTPUT_DIR"
echo "Fit:                    target=$FIT_TARGET  F=$F_FACTOR  n_calls=$N_CALLS  n_initial=$N_INITIAL  [sequential]"
echo "Auto-tau_w:             grid=[$TAU_W_GRID_MS] ms  shape=$SS_TIME_WEIGHT  rho=$SWEEP_RHO  n_grid=$SWEEP_N_GRID"
echo "Loss:                   n_long_train=$N_LONG_TRAIN  defl_cap=${LS_DEFLECTION_CAP_MV}mV  r_in=$R_IN_TARGET  weighting=$WEIGHTING  ss_window=[$SS_WINDOW_MS]ms"
if [ "$SKIP_PHASE2P5" = "1" ]; then
    echo "Phase 2.5:              SKIPPED (legacy free-Ra; Phase 3 = full 3-D)"
else
    echo "Phase 2.5:              ON  n_floor=$N_FLOOR  n_ra_profile=$N_RA_PROFILE  (Ra fixed at cohort median)"
fi
echo "Phase 3:                subset='$PHASE3_SUBSET'  B=$BOOTSTRAP_B  mode=$BOOTSTRAP_MODE  noise=$NOISE_MODE"
echo "-----------------------------------------"

# --- Build the argument list (this IS the entrypoint CLI contract) ----------
ARGS=(
    --archive-dir         "$ARCHIVE_DIR"
    --output-dir          "$OUTPUT_DIR"
    --code-dir            "$CODE_DIR"
    --n-avg-groups        "$N_AVG_GROUPS"
    --fit-target          "$FIT_TARGET"
    --F                   "$F_FACTOR"
    --n-calls             "$N_CALLS"
    --n-initial           "$N_INITIAL"
    # two-pass auto-tau_w + loss
    --n-long-train        "$N_LONG_TRAIN"
    --ls-deflection-cap   "$LS_DEFLECTION_CAP_MV"
    # value may be negative (-1 = disabled); pass with '=' so argparse does not
    # mistake the leading minus for an option flag.
    "--max-sag-amplitude-mV=$MAX_SAG_AMPLITUDE_MV"
    --r-in-target         "$R_IN_TARGET"
    --weighting           "$WEIGHTING"
    --ss-window-ms        "$SS_WINDOW_MS"
    --ss-time-weight      "$SS_TIME_WEIGHT"
    --ls-window-ms        "$LS_WINDOW_MS"
    --tau-w-grid-ms       "$TAU_W_GRID_MS"
    --sweep-rho           "$SWEEP_RHO"
    --sweep-n-grid        "$SWEEP_N_GRID"
    # Phase 2.5
    --n-floor             "$N_FLOOR"
    --n-ra-profile        "$N_RA_PROFILE"
    # Phase 3
    --phase3-subset       "$PHASE3_SUBSET"
    --bootstrap-B         "$BOOTSTRAP_B"
    --bootstrap-mode      "$BOOTSTRAP_MODE"
    --noise-mode          "$NOISE_MODE"
    --bootstrap-n-calls   "$BOOTSTRAP_N_CALLS"
    --bootstrap-n-initial "$BOOTSTRAP_N_INITIAL"
)

# ls-hyp-amps not passed: the real-data loader reads the actual recorded
# amplitudes from the archive (synthetic-only knobs like SS_N_REPEATS /
# LS_HYP_AMPS do not apply to real recordings).

[ -n "$SS_T0_MS" ]         && ARGS+=(--ss-t0-ms "$SS_T0_MS")
[ -n "$MAX_CELLS" ]        && ARGS+=(--max-cells "$MAX_CELLS")
[ "$SKIP_PHASE2P5" = "1" ] && ARGS+=(--skip-phase2p5)

# --- Run --------------------------------------------------------------------
python "$ENTRYPOINT" "${ARGS[@]}"
status=$?

conda deactivate
echo "[done] run '$RUN_TAG' group $GROUP exited with status $status"
sleep 5s
exit $status
