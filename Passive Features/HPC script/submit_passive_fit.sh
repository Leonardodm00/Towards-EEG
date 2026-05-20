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

ARCHIVE_DIR="."     # <-- CHANGE ME
OUTPUT_DIR="./pipeline_outputs"    # <-- CHANGE ME
SCRIPT_PATH="./passive_fitting_hpc.py"   # path to the python script

# ─── Phase 1 / Phase 2 parameters ───────────────────────────────────────
N_AVG_GROUPS=3          # sweep-average groups per polarity
FIT_TARGET="hyp"        # dep | hyp | both
F_FACTOR=1.9            # spine-area correction (Eyal 2016 L2/3 default)
N_CALLS=10             # GP optimiser evaluations per cell
N_INITIAL=5            # random initial points before GP takes over
MAX_CELLS=""            # max cells to process (empty = all in archive)
 
# ─── Phase 3 / Bootstrap parameters ─────────────────────────────────────
SKIP_PHASE3=0                    # 1 = skip Phase 3 entirely, 0 = run it
BOOTSTRAP_B=10                  # bootstrap replicates (≥200 for stable BCa)
BOOTSTRAP_MODE="nonparametric"   # parametric | nonparametric
NOISE_MODE="block"               # iid | ar1 | block  (parametric only)
BOOTSTRAP_N_CALLS=10             # GP budget per bootstrap replicate
BOOTSTRAP_N_INITIAL=10           # random initial points per bootstrap replicate
# ────────────────────────────────────────────────────────────────────────
 
# Move to the directory from which the job was submitted
cd "$PBS_O_WORKDIR"
 
# Load modules
module load python
 
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
echo "Script:                $SCRIPT_PATH"
echo "Archive dir:           $ARCHIVE_DIR"
echo "Output dir:            $OUTPUT_DIR"
echo "Fit target:            $FIT_TARGET  (F=$F_FACTOR)"
echo "Phase 2:               n_calls=$N_CALLS  n_initial=$N_INITIAL  [sequential]"
echo "Bootstrap:             B=$BOOTSTRAP_B  mode=$BOOTSTRAP_MODE  n_calls=$BOOTSTRAP_N_CALLS  n_initial=$BOOTSTRAP_N_INITIAL  [sequential]"
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
python "$SCRIPT_PATH" "${ARGS[@]}"
 
# Clean up
conda deactivate
 
sleep 5s
