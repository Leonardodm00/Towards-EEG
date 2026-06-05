#!/bin/bash

#PBS -S /bin/bash
#PBS -N "passive_fit"
#PBS -q cpu
#PBS -l select=1:ncpus=10,walltime=06:00:00
#PBS -k eo

##########################################################################
# Passive-property fitting pipeline (HPC sequential mode)
#
# Phase 2 and Phase 3 both run sequentially — one cell at a time —
# matching the Colab pipeline's one-session-per-cell semantics.
# NEURON's process-global section list is wiped between cells via
# PassiveCell.destroy(), so no section-contamination bugs can occur.
#
# N_WORKERS and BOOTSTRAP_WORKERS have been removed: they are hard-coded
# to 1 inside the Python script and cannot be overridden from here.
#
# ── Multi-node dispatch ─────────────────────────────────────────────────
# Cells are grouped into per-(layer × type) folders under ARCHIVE_ROOT
# (e.g. ARCHIVE_ROOT/L2_exc, ARCHIVE_ROOT/L3_inh, ...). Each PBS job
# processes exactly one group, selected via the $GROUP env variable:
#
#     qsub -v GROUP=L2_exc submit_passive_fit.sh
#
# Use submit_all_groups.sh to fan out one job per group automatically.
#
# EDIT THE VARIABLES IN THE "USER CONFIG" BLOCK BELOW BEFORE SUBMITTING.
##########################################################################

# ─── USER CONFIG ────────────────────────────────────────────────────────
# Absolute paths are recommended (jobs don't always inherit $PWD).
# The actual archive/output dirs used by the job are <ROOT>/<GROUP>.

ARCHIVE_ROOT="/davinci-1/home/ldellamea/Human Neurons Fitting"
OUTPUT_ROOT="/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs"
SCRIPT_PATH="/davinci-1/home/ldellamea/Human Neurons Fitting/passive_fitting_hpc_fixed.py"

# ─── Phase 1 / Phase 2 parameters ───────────────────────────────────────
N_AVG_GROUPS=3          # sweep-average groups per polarity
FIT_TARGET="hyp"        # dep | hyp | both
# Spine-area correction (Eyal 2016 L2/3 default). Can be overridden per-group
# by the wrapper via `qsub -v F_FACTOR=...`; the value below is the fallback
# used when submitting this script manually without an override.
F_FACTOR="${F_FACTOR:-1.9}"
N_CALLS=100             # GP optimiser evaluations per cell
N_INITIAL=50            # random initial points before GP takes over
MAX_CELLS=""            # max cells to process (empty = all in archive)

# ─── Phase 3 / Bootstrap parameters ─────────────────────────────────────
SKIP_PHASE3=0                    # 1 = skip Phase 3 entirely, 0 = run it
BOOTSTRAP_B=200                  # bootstrap replicates (≥200 for stable BCa)
BOOTSTRAP_MODE="nonparametric"   # parametric | nonparametric
NOISE_MODE="block"               # iid | ar1 | block  (parametric only)
BOOTSTRAP_N_CALLS=40             # GP budget per bootstrap replicate
BOOTSTRAP_N_INITIAL=20           # random initial points per bootstrap replicate
# ────────────────────────────────────────────────────────────────────────

# ─── Resolve per-job paths from $GROUP ──────────────────────────────────
# $GROUP must be provided at submission time:
#     qsub -v GROUP=L2_exc submit_passive_fit.sh
if [ -z "$GROUP" ]; then
    echo "[FATAL] \$GROUP is not set. Submit with:" >&2
    echo "    qsub -v GROUP=<layer_type> submit_passive_fit.sh" >&2
    echo "    e.g. qsub -v GROUP=L2_exc submit_passive_fit.sh" >&2
    exit 2
fi

ARCHIVE_DIR="$ARCHIVE_ROOT/$GROUP"
OUTPUT_DIR="$OUTPUT_ROOT/$GROUP"

if [ ! -d "$ARCHIVE_DIR" ]; then
    echo "[FATAL] Archive directory does not exist: $ARCHIVE_DIR" >&2
    exit 3
fi
# ────────────────────────────────────────────────────────────────────────

# Move to the directory from which the job was submitted
cd "$PBS_O_WORKDIR"

# Load modules
module load python3

# Activate the conda env (must source conda.sh in non-interactive shells)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prova

# Make output dir if it does not exist yet
mkdir -p "$OUTPUT_DIR"

# Sanity prints
echo "Running on node:        $(hostname)"
echo "Job ID:                 $PBS_JOBID"
echo "Group:                  $GROUP"
echo "Working dir:            $PBS_O_WORKDIR"
echo "Python:                 $(which python3)"
echo "Conda env:              $CONDA_DEFAULT_ENV"
echo "Script:                 $SCRIPT_PATH"
echo "Archive root:           $ARCHIVE_ROOT"
echo "Archive dir (this job): $ARCHIVE_DIR"
echo "Output root:            $OUTPUT_ROOT"
echo "Output dir (this job):  $OUTPUT_DIR"
echo "Fit target:             $FIT_TARGET  (F=$F_FACTOR)"
echo "Phase 2:                n_calls=$N_CALLS  n_initial=$N_INITIAL  [sequential]"
echo "Bootstrap:              B=$BOOTSTRAP_B  mode=$BOOTSTRAP_MODE  n_calls=$BOOTSTRAP_N_CALLS  n_initial=$BOOTSTRAP_N_INITIAL  [sequential]"
echo "-----------------------------------------"

# ─── Build the argument list ────────────────────────────────────────────
ARGS=(
    --archive-dir         "$ARCHIVE_DIR"
    --output-dir          "$OUTPUT_DIR"
    --n-avg-groups        "$N_AVG_GROUPS"
    --fit-target          "$FIT_TARGET"
    --F                   "$F_FACTOR"
    --n-calls             "$N_CALLS"
    --n-initial           "$N_INITIAL"
    --bootstrap-B         "$BOOTSTRAP_B"
    --bootstrap-mode      "$BOOTSTRAP_MODE"
    --noise-mode          "$NOISE_MODE"
    --bootstrap-n-calls   "$BOOTSTRAP_N_CALLS"
    --bootstrap-n-initial "$BOOTSTRAP_N_INITIAL"
)

# Optional: cap the number of cells processed (useful for test runs)
if [ -n "$MAX_CELLS" ]; then
    ARGS+=(--max-cells "$MAX_CELLS")
fi

# Optional: skip Phase 3 entirely
if [ "$SKIP_PHASE3" = "1" ]; then
    ARGS+=(--skip-phase3)
fi

# ─── Run ────────────────────────────────────────────────────────────────
python3 "$SCRIPT_PATH" "${ARGS[@]}"

# Clean up
conda deactivate

sleep 5s
