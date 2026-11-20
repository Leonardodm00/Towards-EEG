#!/bin/bash
#PBS -S /bin/bash
#PBS -q cpu
#PBS -k eo
##########################################################################
# submit_twin_recovery.sh -- Block 8 of specs/SPEC.md: the Stage-7 run on the
# twins on davinci (D-184, D-185, D-188, D-182.3). One script, three roles:
#
#   run by hand        THE SUBMITTER. Plans the run (run_twin_recovery.py plan;
#                      a no-op when the run directory is already planned with
#                      the same options), lists the twins (list), refuses a
#                      submission of more than 120 cores at once (twins x
#                      NCPUS, D-185), writes the run file and submits the PBS
#                      array -- one subjob per twin, NCPUS cores each -- and
#                      the merge job after it (-W depend=afterany:<array>).
#   TWIN_ROLE=twin     AN ARRAY SUBJOB: `run_twin_recovery.py twin` on the
#                      PBS_ARRAY_INDEX-th twin: the build (Block 12), then the
#                      fit of its configuration's arm (run_ih_fit).
#   TWIN_ROLE=merge    THE MERGE JOB: `run_twin_recovery.py merge` -> report/,
#                      report_without_flagged/, recovery_by_configuration.csv,
#                      twins_status.csv.
#
# Run it from the "Ih Fit" folder, after `git pull`, with the conda env
# active (the script activates CONDA_ENV itself as well). The order on the
# cluster: the gates, DRY_RUN=1, a pilot (MAX_TWINS), then the 30 twins:
#
#   RUN_TAG=tw1 NOISE_TAG=nf2 DRY_RUN=1 bash submit_twin_recovery.sh
#   RUN_TAG=tw1 NOISE_TAG=nf2 MAX_TWINS=2 bash submit_twin_recovery.sh
#   RUN_TAG=tw1 NOISE_TAG=nf2 WALLTIME=<from the pilot> bash submit_twin_recovery.sh
#
# The third line resubmits every twin with the pilot's run tag: a twin whose
# archive and fit exist is neither rebuilt nor refitted (run_twin_recovery.py
# twin, the resume rule). Submit it after the pilot has ended.
#
# Knobs (environment; every one has a default except RUN_TAG and NOISE_TAG):
#   RUN_TAG          required: letters, digits, '_', '.', '-'
#   NOISE_TAG        required: the noise-fit run tag; the tables are read from
#                    $IH_OUTPUT_BASE/noise_fit/$NOISE_TAG/$GROUP
#   GROUP            L3_exc
#   TWINS            comma-separated twin ids (a subset of the manifest); empty = every twin
#   MAX_TWINS        submit only the first k twins (of the manifest, or of TWINS): the pilot
#   NCPUS            4 cores per twin job (30 x 4 = 120, D-185)
#   WALLTIME         24:00:00 per twin job, until the pilot's timings set it
#   CONDA_ENV        prova
#   PLAN_ARGS        extra options for `plan`, split on white space (use the '=' form for a
#                    negative value, e.g. --dvh-range=-5,5; no value may hold a space)
#   IH_CODE_DIR      the folder of run_twin_recovery.py; default: this script's folder
#   IH_ARCHIVE_ROOT  /davinci-1/home/ldellamea/Human Neurons Fitting (the cells: $IH_ARCHIVE_ROOT/$GROUP/specimen_<id>/)
#   IH_OUTPUT_BASE   $IH_ARCHIVE_ROOT/pipeline_outputs_ih
#                    (outputs: $IH_OUTPUT_BASE/twin_recovery/$RUN_TAG/$GROUP/, logs/ in it)
#   TWIN_RUN_DIR     $HOME/twin_recovery_runs: the run files (no space, no comma: its path goes through qsub -v)
#   DRY_RUN=1        plan (into a temporary directory, removed, unless the run is already planned),
#                    list, check the commands against run_twin_recovery.py's own parser, print the
#                    run file, every qsub and every command the jobs would run; write nothing, submit nothing
#   ROADMAP_DISABLE=1  no roadmap marker for this submission's jobs
#
# Why a run file: the paths hold spaces ("Human Neurons Fitting", "Passive
# Features/HPC script"), and `qsub -v` is comma-separated and does not carry
# them; the submitter freezes every resolved value and the twins in
# $TWIN_RUN_DIR/<RUN_TAG>_<GROUP>_<UTC>.env (a copy goes to logs/), and the
# jobs receive only that file's path.
#
# Threads: the submitter and the merge run with the BLAS threads at 1; a twin
# job runs one process (NEURON and run_ih_fit are sequential), with the BLAS
# threads at NCPUS for the optimiser's linear algebra.
#
# Logs: PBS keeps each job's own .o/.e in $HOME (-k eo: twin_recovery.o<id>,
# twin_merge.o<id>); each job also tees its output to
# <outputs>/logs/twin_<id>_<job>.log or merge_<job>.log.
# Exit codes: 2 usage, 3 refused (archive, noise tables, plan, twins or core cap), 4 qsub, 5 conda.
#
# Pure ASCII, LF (hpc-python-compat). Spec: specs/SPEC.md, Block 8.
##########################################################################

set -euo pipefail

# The BLAS threads at 1 for the submitter and the merge; a twin job sets them
# to NCPUS below (PBS exports OMP_NUM_THREADS = ncpus to a job; overridden
# here, before any python starts).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

MAX_CORES=120                 # D-185: never more than 120 cores at once
MERGE_WALLTIME=01:00:00       # the merge reads CSVs and writes the report: minutes
JOB_NAME_TWIN=twin_recovery   # roadmap allen-mtg-fit S7
JOB_NAME_MERGE=twin_merge

say() { echo "[twin_recovery] $*"; }
die() { echo "[twin_recovery] FATAL: $1" >&2; exit "${2:-2}"; }

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

twin_cmd() {   # twin_cmd CODE_DIR OUT_DIR TWIN_ID -> the array CMD
    CMD=(python "$1/run_twin_recovery.py" twin --out "$2" --twin "$3")
}

merge_cmd() {  # merge_cmd CODE_DIR OUT_DIR -> the array CMD
    CMD=(python "$1/run_twin_recovery.py" merge --out "$2")
}

print_cmd() {  # print_cmd PREFIX WORD...: one line, every word shell-quoted (copy-pastable)
    local prefix="$1"
    shift
    printf '%s' "$prefix"
    printf ' %q' "$@"
    printf '\n'
}

ROLE="${TWIN_ROLE:-}"
if [ -z "$ROLE" ] && [ -n "${PBS_JOBID:-}" ]; then
    die "PBS_JOBID is set but TWIN_ROLE is not: this script submits itself -- run it by hand on the login node (bash submit_twin_recovery.sh), not with qsub and not inside a PBS job" 2
fi

##########################################################################
#  A JOB: an array subjob (TWIN_ROLE=twin) or the merge (TWIN_ROLE=merge)
##########################################################################
if [ -n "$ROLE" ]; then
    case "$ROLE" in twin|merge) ;; *) die "TWIN_ROLE must be twin or merge (got '$ROLE')" 2 ;; esac
    [ -n "${TWIN_RUN_FILE:-}" ] && [ -f "$TWIN_RUN_FILE" ] \
        || die "run file '${TWIN_RUN_FILE:-}' not found (written by the submitter)" 2
    # shellcheck disable=SC1090
    . "$TWIN_RUN_FILE"
    for _v in TWIN_CODE_DIR TWIN_OUT_DIR TWIN_GROUP TWIN_NCPUS TWIN_CONDA_ENV; do
        [ -n "${!_v:-}" ] || die "the run file $TWIN_RUN_FILE sets no $_v" 2
    done
    declare -p TWIN_IDS >/dev/null 2>&1 && [ "${#TWIN_IDS[@]}" -ge 1 ] \
        || die "the run file $TWIN_RUN_FILE holds no twins" 2
    activate_env "$TWIN_CONDA_ENV"
    cd "$TWIN_CODE_DIR" || die "cannot cd to the code dir $TWIN_CODE_DIR" 2
    N_TWINS="${#TWIN_IDS[@]}"
    JOB_TAG="$(printf '%s' "${PBS_JOBID:-nojob}" | tr -c 'A-Za-z0-9._-' '_')"
    mkdir -p "$TWIN_OUT_DIR/logs"
    if [ "$ROLE" = twin ]; then
        IDX="${PBS_ARRAY_INDEX:-${TWIN_INDEX:-}}"
        case "$IDX" in ''|*[!0-9]*) die "no array index (PBS_ARRAY_INDEX or TWIN_INDEX)" 2 ;; esac
        [ "$IDX" -lt "$N_TWINS" ] || die "array index $IDX outside the $N_TWINS twin(s) of the run file" 2
        TWIN_ID="${TWIN_IDS[$IDX]}"
        twin_cmd "$TWIN_CODE_DIR" "$TWIN_OUT_DIR" "$TWIN_ID"
        LOG="$TWIN_OUT_DIR/logs/twin_${TWIN_ID}_${JOB_TAG}.log"
        TASKS="$N_TWINS"
        # one process per twin: the BLAS threads may use the job's cores
        export OMP_NUM_THREADS="$TWIN_NCPUS" OPENBLAS_NUM_THREADS="$TWIN_NCPUS" MKL_NUM_THREADS="$TWIN_NCPUS"
    else
        merge_cmd "$TWIN_CODE_DIR" "$TWIN_OUT_DIR"
        LOG="$TWIN_OUT_DIR/logs/merge_${JOB_TAG}.log"
        TASKS=1
    fi

    # Roadmap marker (subproject-roadmap, references/hpc_marker.md): the first
    # subjob of the array pushes "running" for stage S7 to the hpc-status
    # branch, the last one "done" or "failed"; the merge job marks itself the
    # same way. Never fails the job; ROADMAP_DISABLE=1 turns it off.
    export ROADMAP_STAGE=S7
    export ROADMAP_N_TASKS="$TASKS"
    export ROADMAP_SCRIPT="Passive Features/HPC script/Ih Fit/submit_twin_recovery.sh"
    REPO_ROOT="$(git -C "$TWIN_CODE_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "$REPO_ROOT" ] && [ -f "$REPO_ROOT/specs/roadmap/hpc_status.sh" ]; then
        # shellcheck disable=SC1091
        . "$REPO_ROOT/specs/roadmap/hpc_status.sh"
        roadmap_job_start
        trap 'roadmap_job_end $?' EXIT
    else
        echo "[twin_recovery] no specs/roadmap/hpc_status.sh above $TWIN_CODE_DIR: no roadmap marker" >&2
    fi

    say "role $ROLE, job ${PBS_JOBID:-none}, index ${PBS_ARRAY_INDEX:-${TWIN_INDEX:--}} of $N_TWINS, host $(hostname 2>/dev/null || uname -n)"
    say "python $(command -v python), conda env ${CONDA_DEFAULT_ENV:-none}, threads OMP/OPENBLAS/MKL $OMP_NUM_THREADS/$OPENBLAS_NUM_THREADS/$MKL_NUM_THREADS"
    say "checkout $(git -C "$TWIN_CODE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown); log $LOG"
    print_cmd "[twin_recovery] run:" "${CMD[@]}"
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
NOISE_TAG="${NOISE_TAG:-}"
case "$NOISE_TAG" in
    ""|*[!A-Za-z0-9_.-]*) die "NOISE_TAG must be set to the noise-fit run tag: letters, digits, '_', '.', '-' only (got '$NOISE_TAG')" 2 ;;
esac
GROUP="${GROUP:-L3_exc}"
case "$GROUP" in ""|*[!A-Za-z0-9_.-]*) die "GROUP: letters, digits, '_', '.', '-' only (got '$GROUP')" 2 ;; esac
TWINS="${TWINS:-}"
case "$TWINS" in *[!0-9,]*) die "TWINS: comma-separated twin ids (got '$TWINS')" 2 ;; esac
MAX_TWINS="${MAX_TWINS:-}"
case "$MAX_TWINS" in ''|[1-9]|[1-9]*[0-9]) ;; *) die "MAX_TWINS: a positive integer (got '$MAX_TWINS')" 2 ;; esac
case "$MAX_TWINS" in *[!0-9]*) die "MAX_TWINS: a positive integer (got '$MAX_TWINS')" 2 ;; esac
NCPUS="${NCPUS:-4}"
case "$NCPUS" in ""|*[!0-9]*|0*) die "NCPUS: a positive integer (got '$NCPUS')" 2 ;; esac
WALLTIME="${WALLTIME:-24:00:00}"
[[ "$WALLTIME" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || die "WALLTIME: HH:MM:SS (got '$WALLTIME')" 2
CONDA_ENV="${CONDA_ENV:-prova}"
DRY_RUN="${DRY_RUN:-0}"
PLAN_EXTRA=()
if [ -n "${PLAN_ARGS:-}" ]; then
    read -r -a PLAN_EXTRA <<< "$PLAN_ARGS"
fi

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_NAME="$(basename "${BASH_SOURCE[0]}")"
CODE_DIR="${IH_CODE_DIR:-$SELF_DIR}"
[ -f "$CODE_DIR/run_twin_recovery.py" ] || die "run_twin_recovery.py not found in the code dir '$CODE_DIR' (IH_CODE_DIR)" 2
CODE_DIR="$(cd "$CODE_DIR" && pwd)"
ARCHIVE_ROOT="${IH_ARCHIVE_ROOT:-/davinci-1/home/ldellamea/Human Neurons Fitting}"
OUTPUT_BASE="${IH_OUTPUT_BASE:-$ARCHIVE_ROOT/pipeline_outputs_ih}"
GROUP_DIR="$ARCHIVE_ROOT/$GROUP"
NOISE_DIR="$OUTPUT_BASE/noise_fit/$NOISE_TAG/$GROUP"
OUT_DIR="$OUTPUT_BASE/twin_recovery/$RUN_TAG/$GROUP"
RUN_DIR="${TWIN_RUN_DIR:-${HOME:-}/twin_recovery_runs}"
case "$RUN_DIR" in
    /*) ;;
    *) die "TWIN_RUN_DIR must be an absolute path (got '$RUN_DIR')" 2 ;;
esac
case "$RUN_DIR" in
    *[[:space:],]*) die "TWIN_RUN_DIR '$RUN_DIR' holds a space or a comma; its path goes through qsub -v -- set TWIN_RUN_DIR to a plain path" 2 ;;
esac
[ -d "$GROUP_DIR" ] || die "the archive group does not exist: $GROUP_DIR (IH_ARCHIVE_ROOT, GROUP)" 3
for _t in "noise_cells_$GROUP.csv" "noise_sweeps_$GROUP.csv"; do
    [ -f "$NOISE_DIR/$_t" ] || die "the noise-fit table $NOISE_DIR/$_t does not exist (IH_OUTPUT_BASE, NOISE_TAG, GROUP)" 3
done

activate_env "$CONDA_ENV"

# The plan: run_twin_recovery.py plan (a no-op when the run is already planned
# with the same options, refused (3) when the options differ). A dry run
# writes nothing: it plans into a temporary directory, removed at exit,
# unless the run is already planned (then `plan` on it writes nothing either).
PLAN_FORCE=0
for _a in "${PLAN_EXTRA[@]+"${PLAN_EXTRA[@]}"}"; do
    [ "$_a" = "--force" ] && PLAN_FORCE=1
done
PLAN_OUT="$OUT_DIR"
PLAN_NOTE=""
if [ "$DRY_RUN" = "1" ] && { [ ! -f "$OUT_DIR/manifest.csv" ] || [ "$PLAN_FORCE" = "1" ]; }; then
    PLAN_OUT="$(mktemp -d "${TMPDIR:-/tmp}/twin_recovery_dryrun.XXXXXX")" || die "mktemp failed" 2
    trap 'rm -rf "$PLAN_OUT"' EXIT
    PLAN_NOTE=" (DRY RUN: planned into a temporary directory, removed at exit)"
fi
PLAN_CMD=(python "$CODE_DIR/run_twin_recovery.py" plan --group-dir "$GROUP_DIR" --noise-dir "$NOISE_DIR"
          --out "$PLAN_OUT" --group "$GROUP" "${PLAN_EXTRA[@]+"${PLAN_EXTRA[@]}"}")
print_cmd "[twin_recovery] plan$PLAN_NOTE:" "${PLAN_CMD[@]}"
"${PLAN_CMD[@]}" || die "run_twin_recovery.py plan refused the run (its message above)" 3

# The twins: run_twin_recovery.py's own listing of the manifest (read-only).
LIST_TXT="$(python "$CODE_DIR/run_twin_recovery.py" list --out "$PLAN_OUT")" \
    || die "run_twin_recovery.py list failed (its message above)" 3
mapfile -t ALL_IDS <<< "$LIST_TXT"
[ "${#ALL_IDS[@]}" -ge 1 ] && [ -n "${ALL_IDS[0]}" ] || die "the manifest lists no twin" 3
IDS=()
if [ -n "$TWINS" ]; then
    IFS=, read -r -a WANT <<< "$TWINS"
    for w in "${WANT[@]}"; do
        [ -n "$w" ] || continue
        found=0
        for a in "${ALL_IDS[@]}"; do [ "$a" = "$w" ] && found=1; done
        [ "$found" = 1 ] || die "TWINS: twin $w is not in the manifest (${ALL_IDS[*]})" 3
    done
    for a in "${ALL_IDS[@]}"; do          # the manifest's order
        for w in "${WANT[@]}"; do [ "$a" = "$w" ] && IDS+=("$a") && break; done
    done
else
    IDS=("${ALL_IDS[@]}")
fi
if [ -n "$MAX_TWINS" ] && [ "${#IDS[@]}" -gt "$MAX_TWINS" ]; then
    IDS=("${IDS[@]:0:$MAX_TWINS}")
fi
N="${#IDS[@]}"
TOTAL=$((N * NCPUS))
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_FILE="$RUN_DIR/${RUN_TAG}_${GROUP}_${STAMP}.env"

say "submitter: run tag $RUN_TAG, group $GROUP, noise tag $NOISE_TAG, conda env $CONDA_ENV"
say "code dir:  $CODE_DIR"
say "archive:   $GROUP_DIR"
say "noise:     $NOISE_DIR"
say "outputs:   $OUT_DIR"
say "run file:  $RUN_FILE"
say "twins:     $N of ${#ALL_IDS[@]}: ${IDS[*]}"
say "cores:     $N twin(s) x NCPUS $NCPUS = $TOTAL (cap $MAX_CORES, D-185); walltime $WALLTIME per twin job"
if [ "$TOTAL" -gt "$MAX_CORES" ]; then
    die "$N twin(s) x $NCPUS cores = $TOTAL cores > $MAX_CORES (D-185): submit fewer twins (TWINS, MAX_TWINS) or fewer cores per twin (NCPUS)" 3
fi

write_run_file() {
    printf '# twin-recovery run file -- written by submit_twin_recovery.sh at %s; sourced by its PBS jobs\n' "$STAMP"
    printf 'TWIN_CODE_DIR=%q\n' "$CODE_DIR"
    printf 'TWIN_ARCHIVE_DIR=%q\n' "$GROUP_DIR"
    printf 'TWIN_NOISE_DIR=%q\n' "$NOISE_DIR"
    printf 'TWIN_OUT_DIR=%q\n' "$OUT_DIR"
    printf 'TWIN_GROUP=%q\n' "$GROUP"
    printf 'TWIN_RUN_TAG=%q\n' "$RUN_TAG"
    printf 'TWIN_NOISE_TAG=%q\n' "$NOISE_TAG"
    printf 'TWIN_NCPUS=%q\n' "$NCPUS"
    printf 'TWIN_CONDA_ENV=%q\n' "$CONDA_ENV"
    printf 'TWIN_IDS=('
    printf ' %q' "${IDS[@]}"
    printf ' )\n'
}

V_EXTRA=""
if [ "${ROADMAP_DISABLE:-0}" = "1" ]; then
    V_EXTRA=",ROADMAP_DISABLE=1"
fi
if [ "$N" -ge 2 ]; then
    TWIN_JOB_ARGS=(-J "0-$((N - 1))" -v "TWIN_ROLE=twin,TWIN_RUN_FILE=$RUN_FILE$V_EXTRA")
else
    # PBS refuses an array of one subjob: a plain job, its index passed as TWIN_INDEX
    TWIN_JOB_ARGS=(-v "TWIN_ROLE=twin,TWIN_INDEX=0,TWIN_RUN_FILE=$RUN_FILE$V_EXTRA")
fi
QSUB_TWIN=(qsub -N "$JOB_NAME_TWIN" -q cpu -k eo -l "select=1:ncpus=$NCPUS" -l "walltime=$WALLTIME"
           "${TWIN_JOB_ARGS[@]}" "$SELF_NAME")
merge_qsub() {   # merge_qsub ARRAY_ID -> the array QSUB_MERGE
    QSUB_MERGE=(qsub -N "$JOB_NAME_MERGE" -q cpu -k eo -l "select=1:ncpus=1" -l "walltime=$MERGE_WALLTIME"
                -W "depend=afterany:$1" -v "TWIN_ROLE=merge,TWIN_RUN_FILE=$RUN_FILE$V_EXTRA" "$SELF_NAME")
}

# The CLI contract, checked rather than printed: run_twin_recovery.py's own
# parser reads exactly the arguments the jobs will pass (no build, no fit).
check_args() {   # check_args LABEL ARG...
    local label="$1"
    shift
    (cd "$CODE_DIR" && python - "$@" <<'PYEOF'
import sys
import run_twin_recovery as R
R.build_parser().parse_args(sys.argv[1:])
PYEOF
    ) || die "run_twin_recovery.py's parser refused the arguments of the $label job (above)" 2
}
for i in "${!IDS[@]}"; do
    twin_cmd "$CODE_DIR" "$OUT_DIR" "${IDS[$i]}"
    check_args "twin ${IDS[$i]}" "${CMD[@]:2}"
done
merge_cmd "$CODE_DIR" "$OUT_DIR"
check_args merge "${CMD[@]:2}"
say "CLI contract: run_twin_recovery.py's parser accepts the twin and the merge arguments"

if [ "$DRY_RUN" = "1" ]; then
    say "DRY RUN -- the run file would read:"
    write_run_file | sed 's/^/    /'
    print_cmd "[twin_recovery] DRY RUN: cd $(printf '%q' "$SELF_DIR") &&" "${QSUB_TWIN[@]}"
    for i in "${!IDS[@]}"; do
        twin_cmd "$CODE_DIR" "$OUT_DIR" "${IDS[$i]}"
        print_cmd "[twin_recovery] DRY RUN:   subjob $i:" "${CMD[@]}"
    done
    merge_qsub "ARRAY_JOB_ID"
    print_cmd "[twin_recovery] DRY RUN:" "${QSUB_MERGE[@]}"
    merge_cmd "$CODE_DIR" "$OUT_DIR"
    print_cmd "[twin_recovery] DRY RUN:   merge:" "${CMD[@]}"
    say "DRY RUN: nothing written, nothing submitted"
    exit 0
fi

command -v qsub >/dev/null 2>&1 || die "qsub not found: run the submitter on the login node" 4
mkdir -p "$RUN_DIR" "$OUT_DIR/logs"
write_run_file > "$RUN_FILE"
cp "$RUN_FILE" "$OUT_DIR/logs/"
cd "$SELF_DIR"
ARRAY_ID="$("${QSUB_TWIN[@]}")" || die "qsub of the twin job(s) failed (nothing submitted)" 4
ARRAY_ID="${ARRAY_ID//[[:space:]]/}"
say "twin job(s) submitted: $ARRAY_ID ($N subjob(s), $NCPUS cores each, walltime $WALLTIME)"
merge_qsub "$ARRAY_ID"
MERGE_ID="$("${QSUB_MERGE[@]}")" \
    || die "qsub of the merge failed; the twin jobs $ARRAY_ID are queued -- merge by hand afterwards: python run_twin_recovery.py merge --out <outputs>" 4
MERGE_ID="${MERGE_ID//[[:space:]]/}"
say "merge job submitted:   $MERGE_ID (after $ARRAY_ID)"
printf '# twin job(s): %s\n# merge job: %s\n' "$ARRAY_ID" "$MERGE_ID" | tee -a "$RUN_FILE" >> "$OUT_DIR/logs/$(basename "$RUN_FILE")"
say "expect: one 'TWIN <id>: status=...' line per twin in $OUT_DIR/logs/twin_*.log, then"
say "        the report and the gate's verdict in logs/merge_*.log ($OUT_DIR/report/gate_verdict.txt)"
exit 0
