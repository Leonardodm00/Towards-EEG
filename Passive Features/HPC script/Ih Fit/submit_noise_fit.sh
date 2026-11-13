#!/bin/bash
#PBS -S /bin/bash
#PBS -q cpu
#PBS -k eo
##########################################################################
# submit_noise_fit.sh -- Block 6 of specs/SPEC.md: the noise-fit run on
# davinci (D-123 to D-126). One script, three roles:
#
#   run by hand        THE SUBMITTER. Lists the cohort (run_noise_fit.py list),
#                      refuses a submission of more than 120 cores at once
#                      (cells x NCPUS, D-125.1), writes the run file and
#                      submits the PBS array -- one subjob per cell, NCPUS
#                      cores each -- and the merge job after it
#                      (-W depend=afterany:<array>).
#   NOISE_ROLE=cell    AN ARRAY SUBJOB: `run_noise_fit.py cell` on the
#                      PBS_ARRAY_INDEX-th cell of the cohort, --workers NCPUS
#                      (Blocks 3, 4 and 9; Block 9's sweeps in a pool, D-126).
#   NOISE_ROLE=merge   THE MERGE JOB: `run_noise_fit.py merge` over the cohort
#                      -> noise_cells_<GROUP>.csv, noise_sweeps_<GROUP>.csv.
#
# Run it from the "Ih Fit" folder, after `git pull`, with the conda env
# active (the script activates CONDA_ENV itself as well):
#
#   RUN_TAG=nf1 CELLS=508282493,569844159 DRY_RUN=1 bash submit_noise_fit.sh
#   RUN_TAG=nf1 CELLS=508282493,569844159 WALLTIME=48:00:00 bash submit_noise_fit.sh
#   RUN_TAG=nf1 WALLTIME=<from the pilot> bash submit_noise_fit.sh
#
# The third line is the whole group with the pilot's run tag: the pilot's
# cells are skipped by the resume rule while the noise-fit code is the same
# commit (run_noise_fit.py, D-125.5). Submit it after the pilot has ended.
#
# Knobs (environment; every one has a default except RUN_TAG):
#   RUN_TAG          required: letters, digits, '_', '.', '-'
#   GROUP            L3_exc
#   CELLS            comma-separated specimen IDs (the pilot, D-124); empty = every cell of the group
#   NCPUS            6 cores per cell job = Block 9's pool (D-125.1, D-126)
#   WALLTIME         12:00:00 per cell job, until the pilot's timings set it (D-125.5)
#   SIZES            full (Blocks 4 and 9's defaults, D-125.5) | smoke (the smoke suites' reduced sizes)
#   IH_CODE_DIR      the folder of run_noise_fit.py; default: this script's folder
#   IH_ARCHIVE_ROOT  /davinci-1/home/ldellamea/Human Neurons Fitting (the cells: $IH_ARCHIVE_ROOT/$GROUP/specimen_<id>/)
#   IH_OUTPUT_BASE   $IH_ARCHIVE_ROOT/pipeline_outputs_ih
#                    (outputs: $IH_OUTPUT_BASE/noise_fit/$RUN_TAG/$GROUP/ -- cells/, the two tables, logs/)
#   CONDA_ENV        prova
#   NOISE_RUN_DIR    $HOME/noise_fit_runs: the run files (no space, no comma: its path goes through qsub -v)
#   DRY_RUN=1        list the cohort, check the commands against run_noise_fit.py's own parser, print the
#                    run file, every qsub and every command the jobs would run; write nothing, submit nothing
#   ROADMAP_DISABLE=1  no roadmap marker for this submission's jobs
#
# Why a run file: the paths hold spaces ("Human Neurons Fitting", "Passive
# Features/HPC script"), and `qsub -v` is comma-separated and does not carry
# them; the submitter freezes every resolved value and the cohort in
# $NOISE_RUN_DIR/<RUN_TAG>_<GROUP>_<UTC>.env (a copy goes to logs/), and the
# jobs receive only that file's path. A job therefore runs exactly the cohort
# that was submitted, whatever happens to the archive or the knobs meanwhile.
#
# Logs: PBS keeps each job's own .o/.e in $HOME (-k eo: noise_fit.o<id>,
# noise_merge.o<id>); each job also tees its output to
# <outputs>/logs/cell_<id>_<job>.log or merge_<job>.log.
# Exit codes: 2 usage, 3 refused (archive, cohort or core cap), 4 qsub, 5 conda.
#
# Pure ASCII, LF (hpc-python-compat). Spec: specs/SPEC.md, Block 6.
##########################################################################

set -euo pipefail

# The BLAS threads at 1 (D-126): Block 9's sweeps run in a pool of NCPUS
# processes, one core each. PBS exports OMP_NUM_THREADS = ncpus to a job;
# overridden here, before any python starts.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

MAX_CORES=120                 # D-125.1: never more than 120 cores at once
MERGE_WALLTIME=01:00:00       # the merge reads JSONs and writes two tables: minutes
JOB_NAME_CELL=noise_fit       # roadmap allen-mtg-fit N6, jobs: [..., noise_fit]
JOB_NAME_MERGE=noise_merge

say() { echo "[noise_fit] $*"; }
die() { echo "[noise_fit] FATAL: $1" >&2; exit "${2:-2}"; }

activate_env() {
    # The activation block of submit_ih_fit.sh (verified on davinci), with
    # set -u relaxed across it: conda's activate.d / deactivate.d scripts are
    # third-party and not set -u clean (hpc-git-delivery).
    local base
    set +u
    base="$(conda info --base 2>/dev/null)" || die "conda not found on PATH (needed for env '$1')" 5
    # shellcheck disable=SC1091
    source "$base/etc/profile.d/conda.sh" || die "cannot source $base/etc/profile.d/conda.sh" 5
    conda activate "$1" || die "cannot activate conda env '$1'" 5
    set -u
}

cell_cmd() {   # cell_cmd CODE_DIR ARCHIVE_DIR SPECIMEN OUT_DIR NCPUS SIZES -> the array CMD
    CMD=(python "$1/run_noise_fit.py" cell --group-dir "$2" --specimen "$3" --out "$4" --workers "$5" --sizes "$6")
}

merge_cmd() {  # merge_cmd CODE_DIR OUT_DIR GROUP CELLS_CSV -> the array CMD
    CMD=(python "$1/run_noise_fit.py" merge --out "$2" --group "$3" --cells "$4")
}

print_cmd() {  # print_cmd PREFIX WORD...: one line, every word shell-quoted (copy-pastable)
    local prefix="$1"
    shift
    printf '%s' "$prefix"
    printf ' %q' "$@"
    printf '\n'
}

ROLE="${NOISE_ROLE:-}"
if [ -z "$ROLE" ] && [ -n "${PBS_JOBID:-}" ]; then
    die "PBS_JOBID is set but NOISE_ROLE is not: this script submits itself -- run it by hand on the login node (bash submit_noise_fit.sh), not with qsub and not inside a PBS job" 2
fi

##########################################################################
#  A JOB: an array subjob (NOISE_ROLE=cell) or the merge (NOISE_ROLE=merge)
##########################################################################
if [ -n "$ROLE" ]; then
    case "$ROLE" in cell|merge) ;; *) die "NOISE_ROLE must be cell or merge (got '$ROLE')" 2 ;; esac
    [ -n "${NOISE_RUN_FILE:-}" ] && [ -f "$NOISE_RUN_FILE" ] \
        || die "run file '${NOISE_RUN_FILE:-}' not found (written by the submitter)" 2
    # shellcheck disable=SC1090
    . "$NOISE_RUN_FILE"
    for _v in NOISE_CODE_DIR NOISE_ARCHIVE_DIR NOISE_OUT_DIR NOISE_GROUP NOISE_NCPUS NOISE_SIZES NOISE_CONDA_ENV; do
        [ -n "${!_v:-}" ] || die "the run file $NOISE_RUN_FILE sets no $_v" 2
    done
    declare -p NOISE_COHORT >/dev/null 2>&1 && [ "${#NOISE_COHORT[@]}" -ge 1 ] \
        || die "the run file $NOISE_RUN_FILE holds no cohort" 2
    activate_env "$NOISE_CONDA_ENV"
    cd "$NOISE_CODE_DIR" || die "cannot cd to the code dir $NOISE_CODE_DIR" 2
    N_CELLS="${#NOISE_COHORT[@]}"
    JOB_TAG="$(printf '%s' "${PBS_JOBID:-nojob}" | tr -c 'A-Za-z0-9._-' '_')"
    mkdir -p "$NOISE_OUT_DIR/logs"
    if [ "$ROLE" = cell ]; then
        IDX="${PBS_ARRAY_INDEX:-${NOISE_INDEX:-}}"
        case "$IDX" in ''|*[!0-9]*) die "no array index (PBS_ARRAY_INDEX or NOISE_INDEX)" 2 ;; esac
        [ "$IDX" -lt "$N_CELLS" ] || die "array index $IDX outside the cohort of $N_CELLS cell(s)" 2
        SPECIMEN="${NOISE_COHORT[$IDX]}"
        cell_cmd "$NOISE_CODE_DIR" "$NOISE_ARCHIVE_DIR" "$SPECIMEN" "$NOISE_OUT_DIR" "$NOISE_NCPUS" "$NOISE_SIZES"
        LOG="$NOISE_OUT_DIR/logs/cell_${SPECIMEN}_${JOB_TAG}.log"
        TASKS="$N_CELLS"
    else
        merge_cmd "$NOISE_CODE_DIR" "$NOISE_OUT_DIR" "$NOISE_GROUP" "$(IFS=,; echo "${NOISE_COHORT[*]}")"
        LOG="$NOISE_OUT_DIR/logs/merge_${JOB_TAG}.log"
        TASKS=1
    fi

    # Roadmap marker (subproject-roadmap, references/hpc_marker.md): the first
    # subjob of the array pushes "running" for stage N6 to the hpc-status
    # branch, the last one "done" or "failed"; the merge job marks itself the
    # same way. Never fails the job; ROADMAP_DISABLE=1 turns it off.
    export ROADMAP_STAGE=N6
    export ROADMAP_N_TASKS="$TASKS"
    export ROADMAP_SCRIPT="Passive Features/HPC script/Ih Fit/submit_noise_fit.sh"
    REPO_ROOT="$(git -C "$NOISE_CODE_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "$REPO_ROOT" ] && [ -f "$REPO_ROOT/specs/roadmap/hpc_status.sh" ]; then
        # shellcheck disable=SC1091
        . "$REPO_ROOT/specs/roadmap/hpc_status.sh"
        roadmap_job_start
        trap 'roadmap_job_end $?' EXIT
    else
        echo "[noise_fit] no specs/roadmap/hpc_status.sh above $NOISE_CODE_DIR: no roadmap marker" >&2
    fi

    say "role $ROLE, job ${PBS_JOBID:-none}, index ${PBS_ARRAY_INDEX:-${NOISE_INDEX:--}} of $N_CELLS, host $(hostname 2>/dev/null || uname -n)"
    say "python $(command -v python), conda env ${CONDA_DEFAULT_ENV:-none}, threads OMP/OPENBLAS/MKL $OMP_NUM_THREADS/$OPENBLAS_NUM_THREADS/$MKL_NUM_THREADS"
    say "checkout $(git -C "$NOISE_CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown); log $LOG"
    print_cmd "[noise_fit] run:" "${CMD[@]}"
    "${CMD[@]}" 2>&1 | tee -a "$LOG"
    exit 0
fi

##########################################################################
#  THE SUBMITTER (run by hand)
##########################################################################
RUN_TAG="${RUN_TAG:-}"
case "$RUN_TAG" in
    ""|*[!A-Za-z0-9_.-]*) die "RUN_TAG must be set: letters, digits, '_', '.', '-' only (got '$RUN_TAG')" 2 ;;
esac
GROUP="${GROUP:-L3_exc}"
case "$GROUP" in ""|*[!A-Za-z0-9_.-]*) die "GROUP: letters, digits, '_', '.', '-' only (got '$GROUP')" 2 ;; esac
CELLS="${CELLS:-}"
case "$CELLS" in *[!0-9,]*) die "CELLS: comma-separated specimen IDs (got '$CELLS')" 2 ;; esac
NCPUS="${NCPUS:-6}"
case "$NCPUS" in ""|*[!0-9]*|0*) die "NCPUS: a positive integer (got '$NCPUS')" 2 ;; esac
WALLTIME="${WALLTIME:-12:00:00}"
[[ "$WALLTIME" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || die "WALLTIME: HH:MM:SS (got '$WALLTIME')" 2
SIZES="${SIZES:-full}"
case "$SIZES" in full|smoke) ;; *) die "SIZES: full or smoke (got '$SIZES')" 2 ;; esac
CONDA_ENV="${CONDA_ENV:-prova}"
DRY_RUN="${DRY_RUN:-0}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_NAME="$(basename "${BASH_SOURCE[0]}")"
CODE_DIR="${IH_CODE_DIR:-$SELF_DIR}"
[ -f "$CODE_DIR/run_noise_fit.py" ] || die "run_noise_fit.py not found in the code dir '$CODE_DIR' (IH_CODE_DIR)" 2
CODE_DIR="$(cd "$CODE_DIR" && pwd)"
ARCHIVE_ROOT="${IH_ARCHIVE_ROOT:-/davinci-1/home/ldellamea/Human Neurons Fitting}"
OUTPUT_BASE="${IH_OUTPUT_BASE:-$ARCHIVE_ROOT/pipeline_outputs_ih}"
ARCHIVE_DIR="$ARCHIVE_ROOT/$GROUP"
OUT_DIR="$OUTPUT_BASE/noise_fit/$RUN_TAG/$GROUP"
RUN_DIR="${NOISE_RUN_DIR:-${HOME:-}/noise_fit_runs}"
case "$RUN_DIR" in
    /*) ;;
    *) die "NOISE_RUN_DIR must be an absolute path (got '$RUN_DIR')" 2 ;;
esac
case "$RUN_DIR" in
    *[[:space:],]*) die "NOISE_RUN_DIR '$RUN_DIR' holds a space or a comma; its path goes through qsub -v -- set NOISE_RUN_DIR to a plain path" 2 ;;
esac
[ -d "$ARCHIVE_DIR" ] || die "the archive group does not exist: $ARCHIVE_DIR (IH_ARCHIVE_ROOT, GROUP)" 3

activate_env "$CONDA_ENV"

# The cohort: run_noise_fit.py's own listing (read-only).
LIST_ARGS=(list --group-dir "$ARCHIVE_DIR")
if [ -n "$CELLS" ]; then
    LIST_ARGS+=(--cells "$CELLS")
fi
COHORT_TXT="$(python "$CODE_DIR/run_noise_fit.py" "${LIST_ARGS[@]}")" \
    || die "run_noise_fit.py list refused the cohort (its message above)" 3
mapfile -t COHORT <<< "$COHORT_TXT"
N="${#COHORT[@]}"
TOTAL=$((N * NCPUS))
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_FILE="$RUN_DIR/${RUN_TAG}_${GROUP}_${STAMP}.env"

say "submitter: run tag $RUN_TAG, group $GROUP, sizes $SIZES, conda env $CONDA_ENV"
say "code dir:  $CODE_DIR"
say "archive:   $ARCHIVE_DIR"
say "outputs:   $OUT_DIR"
say "run file:  $RUN_FILE"
say "cohort:    $N cell(s): ${COHORT[*]}"
say "cores:     $N cell(s) x NCPUS $NCPUS = $TOTAL (cap $MAX_CORES, D-125.1); walltime $WALLTIME per cell job"
if [ "$TOTAL" -gt "$MAX_CORES" ]; then
    die "$N cell(s) x $NCPUS cores = $TOTAL cores > $MAX_CORES (D-125.1): submit fewer cells (CELLS) or fewer cores per cell (NCPUS)" 3
fi

write_run_file() {
    printf '# noise-fit run file -- written by submit_noise_fit.sh at %s; sourced by its PBS jobs\n' "$STAMP"
    printf 'NOISE_CODE_DIR=%q\n' "$CODE_DIR"
    printf 'NOISE_ARCHIVE_DIR=%q\n' "$ARCHIVE_DIR"
    printf 'NOISE_OUT_DIR=%q\n' "$OUT_DIR"
    printf 'NOISE_GROUP=%q\n' "$GROUP"
    printf 'NOISE_RUN_TAG=%q\n' "$RUN_TAG"
    printf 'NOISE_NCPUS=%q\n' "$NCPUS"
    printf 'NOISE_SIZES=%q\n' "$SIZES"
    printf 'NOISE_CONDA_ENV=%q\n' "$CONDA_ENV"
    printf 'NOISE_COHORT=('
    printf ' %q' "${COHORT[@]}"
    printf ' )\n'
}

V_EXTRA=""
if [ "${ROADMAP_DISABLE:-0}" = "1" ]; then
    V_EXTRA=",ROADMAP_DISABLE=1"
fi
if [ "$N" -ge 2 ]; then
    CELL_JOB_ARGS=(-J "0-$((N - 1))" -v "NOISE_ROLE=cell,NOISE_RUN_FILE=$RUN_FILE$V_EXTRA")
else
    # PBS refuses an array of one subjob: a plain job, its index passed as NOISE_INDEX
    CELL_JOB_ARGS=(-v "NOISE_ROLE=cell,NOISE_INDEX=0,NOISE_RUN_FILE=$RUN_FILE$V_EXTRA")
fi
QSUB_CELL=(qsub -N "$JOB_NAME_CELL" -q cpu -k eo -l "select=1:ncpus=$NCPUS" -l "walltime=$WALLTIME"
           "${CELL_JOB_ARGS[@]}" "$SELF_NAME")
merge_qsub() {   # merge_qsub ARRAY_ID -> the array QSUB_MERGE
    QSUB_MERGE=(qsub -N "$JOB_NAME_MERGE" -q cpu -k eo -l "select=1:ncpus=1" -l "walltime=$MERGE_WALLTIME"
                -W "depend=afterany:$1" -v "NOISE_ROLE=merge,NOISE_RUN_FILE=$RUN_FILE$V_EXTRA" "$SELF_NAME")
}
COHORT_CSV="$(IFS=,; echo "${COHORT[*]}")"

# The CLI contract, checked rather than printed: run_noise_fit.py's own
# parser reads exactly the arguments the jobs will pass (no fit, no merge).
check_args() {   # check_args LABEL ARG...
    local label="$1"
    shift
    (cd "$CODE_DIR" && python - "$@" <<'PYEOF'
import sys
import run_noise_fit as R
R.build_parser().parse_args(sys.argv[1:])
PYEOF
    ) || die "run_noise_fit.py's parser refused the arguments of the $label job (above)" 2
}
cell_cmd "$CODE_DIR" "$ARCHIVE_DIR" "${COHORT[0]}" "$OUT_DIR" "$NCPUS" "$SIZES"
check_args cell "${CMD[@]:2}"
merge_cmd "$CODE_DIR" "$OUT_DIR" "$GROUP" "$COHORT_CSV"
check_args merge "${CMD[@]:2}"
say "CLI contract: run_noise_fit.py's parser accepts the cell and the merge arguments"

if [ "$DRY_RUN" = "1" ]; then
    say "DRY RUN -- the run file would read:"
    write_run_file | sed 's/^/    /'
    print_cmd "[noise_fit] DRY RUN: cd $(printf '%q' "$SELF_DIR") &&" "${QSUB_CELL[@]}"
    for i in "${!COHORT[@]}"; do
        cell_cmd "$CODE_DIR" "$ARCHIVE_DIR" "${COHORT[$i]}" "$OUT_DIR" "$NCPUS" "$SIZES"
        print_cmd "[noise_fit] DRY RUN:   subjob $i:" "${CMD[@]}"
    done
    merge_qsub "ARRAY_JOB_ID"
    print_cmd "[noise_fit] DRY RUN:" "${QSUB_MERGE[@]}"
    merge_cmd "$CODE_DIR" "$OUT_DIR" "$GROUP" "$COHORT_CSV"
    print_cmd "[noise_fit] DRY RUN:   merge:" "${CMD[@]}"
    say "DRY RUN: nothing written, nothing submitted"
    exit 0
fi

command -v qsub >/dev/null 2>&1 || die "qsub not found: run the submitter on the login node" 4
mkdir -p "$RUN_DIR" "$OUT_DIR/cells" "$OUT_DIR/logs"
write_run_file > "$RUN_FILE"
cp "$RUN_FILE" "$OUT_DIR/logs/"
cd "$SELF_DIR"
ARRAY_ID="$("${QSUB_CELL[@]}")" || die "qsub of the cell job(s) failed (nothing submitted)" 4
ARRAY_ID="${ARRAY_ID//[[:space:]]/}"
say "cell job(s) submitted: $ARRAY_ID ($N subjob(s), $NCPUS cores each, walltime $WALLTIME)"
merge_qsub "$ARRAY_ID"
MERGE_ID="$("${QSUB_MERGE[@]}")" \
    || die "qsub of the merge failed; the cell jobs $ARRAY_ID are queued -- merge by hand afterwards: python run_noise_fit.py merge --out <outputs> --group $GROUP --cells $COHORT_CSV" 4
MERGE_ID="${MERGE_ID//[[:space:]]/}"
say "merge job submitted:   $MERGE_ID (after $ARRAY_ID)"
printf '# cell job(s): %s\n# merge job: %s\n' "$ARRAY_ID" "$MERGE_ID" | tee -a "$RUN_FILE" >> "$OUT_DIR/logs/$(basename "$RUN_FILE")"
say "expect: one 'NOISE_CELL <id>: status=...' line per cell in $OUT_DIR/logs/cell_*.log, then"
say "        'NOISE_MERGE $GROUP: <a> ok, <b> refused, <c> failed, <d> missing' in logs/merge_*.log"
exit 0
