#!/bin/bash

#PBS -S /bin/bash
#PBS -N "passive_fit"
#PBS -q cpu
#PBS -l select=1:ncpus=10,walltime=02:00:00
#PBS -k eo

##########################################################################
# Passive-property fitting pipeline (HPC offline mode)
#
# EDIT THE VARIABLES IN THE "USER CONFIG" BLOCK BELOW BEFORE SUBMITTING.
##########################################################################

# ─── USER CONFIG ────────────────────────────────────────────────────────
# Absolute paths recommended (jobs do not always inherit your $PWD nicely).

ARCHIVE_DIR="$HOME/path/to/phase0_archive"     # <-- CHANGE ME
OUTPUT_DIR="$HOME/path/to/pipeline_outputs"    # <-- CHANGE ME
SCRIPT_PATH="./passive_fitting_hpc_fixed.py"   # path to the python script

# ─── Phase 1 / Phase 2 parameters ───────────────────────────────────────
N_AVG_GROUPS=3
FIT_TARGET="hyp"           # dep | hyp | both
F_FACTOR=1.9
N_CALLS=10                 # GP optimiser budget per cell (Phase 2)
N_INITIAL=10               # Random initial points before GP (Phase 2)
MAX_CELLS=1                # Limit number of cells to process
N_WORKERS=""               # Phase 2 parallel workers (empty = auto-detect)

# ─── Phase 3 / Bootstrap parameters ─────────────────────────────────────
SKIP_PHASE3=0              # 1 = skip Phase 3, 0 = run it
BOOTSTRAP_B=10             # Number of bootstrap replicates
BOOTSTRAP_MODE="nonparametric"   # parametric | nonparametric
NOISE_MODE="block"               # iid | ar1 | block (parametric only)
BOOTSTRAP_WORKERS=""       # Parallel workers for the bootstrap loop (empty = auto-detect)
BOOTSTRAP_N_CALLS=100      # GP budget PER bootstrap replicate
BOOTSTRAP_N_INITIAL=50     # Random initial points PER bootstrap replicate
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
echo "Running on node:       $(hostname)"
echo "Job ID:                $PBS_JOBID"
echo "Working dir:           $PBS_O_WORKDIR"
echo "Python:                $(which python3)"
echo "Conda env:             $CONDA_DEFAULT_ENV"
echo "Archive dir:           $ARCHIVE_DIR"
echo "Output dir:            $OUTPUT_DIR"
echo "Bootstrap workers:     $BOOTSTRAP_WORKERS"
echo "Bootstrap B:           $BOOTSTRAP_B  (n_calls=$BOOTSTRAP_N_CALLS, n_initial=$BOOTSTRAP_N_INITIAL)"
echo "-----------------------------------------"

# ─── Build the argument list ────────────────────────────────────────────
ARGS=(
    --archive-dir          "$ARCHIVE_DIR"
    --output-dir           "$OUTPUT_DIR"
    --n-avg-groups         "$N_AVG_GROUPS"
    --fit-target           "$FIT_TARGET"
    --F                    "$F_FACTOR"
    --n-calls              "$N_CALLS"
    --n-initial            "$N_INITIAL"
    --max-cells            "$MAX_CELLS"
    --bootstrap-B          "$BOOTSTRAP_B"
    --bootstrap-mode       "$BOOTSTRAP_MODE"
    --noise-mode           "$NOISE_MODE"
    --bootstrap-n-calls    "$BOOTSTRAP_N_CALLS"
    --bootstrap-n-initial  "$BOOTSTRAP_N_INITIAL"
)

# Optional Phase 2 workers override
if [ -n "$N_WORKERS" ]; then
    ARGS+=(--n-workers "$N_WORKERS")
fi

# Optional bootstrap workers override
if [ -n "$BOOTSTRAP_WORKERS" ]; then
    ARGS+=(--bootstrap-workers "$BOOTSTRAP_WORKERS")
fi

# Optional skip Phase 3
if [ "$SKIP_PHASE3" = "1" ]; then
    ARGS+=(--skip-phase3)
fi

# ─── Run ────────────────────────────────────────────────────────────────
python3 "$SCRIPT_PATH" "${ARGS[@]}"

# Clean up
conda deactivate

sleep 5s
