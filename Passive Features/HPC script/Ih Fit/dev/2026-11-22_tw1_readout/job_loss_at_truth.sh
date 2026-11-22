#!/bin/bash
#PBS -N loss_at_truth
#PBS -q cpu
#PBS -k eo
#PBS -l select=1:ncpus=1
#PBS -l walltime=06:00:00
##########################################################################
# The fit's loss at the truth and at the estimate, for the `full` twins of a
# Stage-7 run (loss_at_truth.py, same folder). One sequential process, 1 core.
#
# Submit from the "Ih Fit" folder (the job finds the code through PBS_O_WORKDIR):
#     qsub dev/2026-11-22_tw1_readout/job_loss_at_truth.sh
# Knobs, with `qsub -v NAME=value,...` (no spaces in the values):
#     TWINS     comma-separated twin ids (default: every `full` twin with status ok)
#     RUN_TAG   the run's tag (default tw1);  GROUP (default L3_exc)
#     CONDA_ENV (default prova; empty = use the python already on PATH)
# Or run it directly, without PBS:  bash dev/2026-11-22_tw1_readout/job_loss_at_truth.sh
# Output: <run dir>/loss_at_truth/loss_at_truth.csv and logs/loss_at_truth_<job>.log;
# the last line of a complete run starts with "LOSS_AT_TRUTH DONE".
# Pure ASCII, LF.
##########################################################################
set -eo pipefail
unset DISPLAY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CODE_DIR="${PBS_O_WORKDIR:-$(pwd)}"
[ -f "$CODE_DIR/run_twin_recovery.py" ] || { echo "ERROR: $CODE_DIR is not the Ih Fit folder (submit or run from it)" >&2; exit 2; }
ARCHIVE_ROOT="${IH_ARCHIVE_ROOT:-/davinci-1/home/ldellamea/Human Neurons Fitting}"
OUTPUT_BASE="${IH_OUTPUT_BASE:-$ARCHIVE_ROOT/pipeline_outputs_ih}"
RUN_DIR="${RUN_DIR:-$OUTPUT_BASE/twin_recovery/${RUN_TAG:-tw1}/${GROUP:-L3_exc}}"
[ -f "$RUN_DIR/manifest.csv" ] || { echo "ERROR: no manifest.csv in $RUN_DIR" >&2; exit 3; }

CONDA_ENV="${CONDA_ENV-prova}"
if [ -n "$CONDA_ENV" ]; then
    # set -u is off here on purpose: conda's activate.d / deactivate.d scripts are not set -u clean
    base="$(conda info --base 2>/dev/null)" || { echo "ERROR: conda not found on PATH" >&2; exit 5; }
    source "$base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || { echo "ERROR: cannot activate conda env $CONDA_ENV" >&2; exit 5; }
fi
set -u

mkdir -p "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/loss_at_truth_${PBS_JOBID:-manual_$$}.log"
ARGS=(--run-dir "$RUN_DIR")
[ -n "${TWINS:-}" ] && ARGS+=(--twins "$TWINS")
cd "$CODE_DIR"
echo "[loss_at_truth] code $CODE_DIR | run $RUN_DIR | python $(command -v python) | twins ${TWINS:-all full}" | tee "$LOG"
python dev/2026-11-22_tw1_readout/loss_at_truth.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
