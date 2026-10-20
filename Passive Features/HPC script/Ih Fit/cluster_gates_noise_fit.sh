#!/bin/bash
##########################################################################
# cluster_gates_noise_fit.sh -- the cluster-side checks of the noise-fit
# code (specs/SPEC.md Blocks 1-2) and of the archive input Block 3 reads.
#
# Run on the LOGIN NODE, from this folder, after `git pull`, with the conda
# env already active (conda activate prova). Nothing to edit:
#
#   bash cluster_gates_noise_fit.sh            # every check, about 15 min
#   QUICK=1 bash cluster_gates_noise_fit.sh    # bytes, imports, fast suites, < 1 min
#
# It never edits a tracked file and never submits a job. It may build
# x86_64/ (nrnivmodl, only when missing or older than mod/*.mod, behind the
# submit scripts' lock) and writes its logs to $GATE_LOG_DIR (default
# ~/ih_fit_gate_logs/<date_time>/), never into the repository.
#
# Knobs (environment variables; the defaults are the davinci archive):
#   IH_ARCHIVE_GROUP  Phase-0 group directory
#                     (/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc)
#   REG_SPECIMEN      the cell of the regression gate             (508282493)
#   NPZ_SPECIMENS     cells whose ss_pulses.npz is described ("508282493 569844159")
#   PYTHON            the interpreter                                (python)
#
# Output: one line per check, "PASS|FAIL|SKIP  <check>  -- <what was seen>",
# then "GATES noise-fit: <p> passed, <f> failed, <s> skipped" and a VERDICT
# line. A check passes only on a line it must SEE -- each suite's own summary
# line -- never on the absence of an error. Exit 0 when nothing failed.
##########################################################################
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 2
cd "$HERE" || exit 2
if [ ! -f noise_law.py ] || [ ! -f noise_stats.py ] || [ ! -f noise_fit_gates.py ]; then
    echo "[FATAL] noise_stats.py, noise_law.py or noise_fit_gates.py missing in $HERE:" >&2
    echo "        run this from 'Ih Fit' after pulling the noise-fit commits." >&2
    exit 2
fi

PY="${PYTHON:-python}"
QUICK="${QUICK:-0}"
ARCHIVE_GROUP="${IH_ARCHIVE_GROUP:-/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc}"
REG_SPECIMEN="${REG_SPECIMEN:-508282493}"
NPZ_SPECIMENS="${NPZ_SPECIMENS:-508282493 569844159}"
LOG_DIR="${GATE_LOG_DIR:-$HOME/ih_fit_gate_logs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR" || { echo "[FATAL] cannot create the log folder $LOG_DIR" >&2; exit 2; }

N_PASS=0
N_FAIL=0
N_SKIP=0
report() {    # report STATUS CHECK NOTE
    printf '%-4s  %s  -- %s\n' "$1" "$2" "$3" | tee -a "$LOG_DIR/summary.txt"
    case "$1" in
        PASS) N_PASS=$((N_PASS + 1)) ;;
        FAIL) N_FAIL=$((N_FAIL + 1)) ;;
        SKIP) N_SKIP=$((N_SKIP + 1)) ;;
    esac
}

# has_line LOG PREFIX: true when some line of LOG starts with PREFIX (fixed string)
has_line() {
    awk -v p="$2" 'index($0, p) == 1 { found = 1 } END { exit found ? 0 : 1 }' "$1"
}

# run_check CHECK LOGNAME PREFIX COMMAND...: PASS when the log shows PREFIX
run_check() {
    local check="$1" log="$LOG_DIR/$2.log" prefix="$3"
    shift 3
    local t0=$SECONDS rc
    "$@" > "$log" 2>&1
    rc=$?
    if has_line "$log" "$prefix"; then
        report PASS "$check" "$(grep -m 1 -F "$prefix" "$log") (exit $rc, $((SECONDS - t0)) s)"
    else
        report FAIL "$check" "no line starting \"$prefix\"; last line: \"$(tail -n 1 "$log")\" (exit $rc; log $log)"
    fi
}

echo "cluster_gates_noise_fit.sh  $(date '+%Y-%m-%d %H:%M:%S')  on $(hostname)"
echo "folder: $HERE"
echo "logs:   $LOG_DIR"
echo "mode:   $([ "$QUICK" = "1" ] && echo QUICK || echo FULL)"

# --- G0 interpreter ----------------------------------------------------------
if out="$("$PY" -c 'import sys; print("python %s at %s" % (sys.version.split()[0], sys.executable))' 2>&1)"; then
    report PASS "G0 interpreter" "$(printf '%s' "$out" | tail -n 1); conda env ${CONDA_DEFAULT_ENV:-none}"
else
    report FAIL "G0 interpreter" "'$PY' does not run: $(printf '%s' "$out" | tail -n 1)"
fi

# --- G1 libraries of the noise code (D-029 (v)) --------------------------------
if out="$("$PY" -c 'import numpy, scipy, statsmodels; print("numpy %s, scipy %s, statsmodels %s" % (numpy.__version__, scipy.__version__, statsmodels.__version__))' 2>&1)"; then
    report PASS "G1 numpy, scipy, statsmodels" "$(printf '%s' "$out" | tail -n 1)"
else
    report FAIL "G1 numpy, scipy, statsmodels" "$(printf '%s' "$out" | tail -n 1) -- install once at the prompt: module load proxy && python -m pip install statsmodels"
fi

# --- G2 the code under test is the committed code ------------------------------
subjects="$(git log --format=%s -n 40 -- . 2>/dev/null)"
b1=no; b2=no
case "$subjects" in *"block 1: textbook sample autocorrelation"*) b1=yes ;; esac
case "$subjects" in *"block 2: the noise law"*) b2=yes ;; esac
modified="$(git status --porcelain --untracked-files=no -- . 2>/dev/null | cut -c4- | head -n 5 | tr '\n' ';')"
if [ "$b1" = yes ] && [ "$b2" = yes ] && [ -z "$modified" ]; then
    report PASS "G2 git" "HEAD $(git rev-parse --short HEAD 2>/dev/null); Blocks 1 and 2 committed; no tracked file of this folder modified"
else
    report FAIL "G2 git" "block 1 committed: $b1, block 2 committed: $b2, tracked files modified here: ${modified:-none} -- git pull on main; restore or commit local edits"
fi

# --- G3 bytes, G4 syntax (hpc-python-compat) -----------------------------------
run_check "G3 bytes" g3_bytes "BYTES: ok" "$PY" noise_fit_gates.py bytes
run_check "G4 syntax" g4_compile "COMPILE: ok" "$PY" noise_fit_gates.py compile

# --- G5, G6 the noise-fit suites (no NEURON) -----------------------------------
run_check "G5 smoke_noise_stats (Block 1)" g5_noise_stats "SMOKE noise_stats: 6/6 passed" "$PY" smoke_noise_stats.py
run_check "G6 smoke_noise_law (Block 2)" g6_noise_law "SMOKE noise_law: 9/9 passed" "$PY" smoke_noise_law.py

if [ "$QUICK" != "1" ]; then
    # --- G7 NEURON, skopt and the compiled mechanisms --------------------------
    if out="$("$PY" -c 'import neuron, skopt; print("neuron %s, skopt %s" % (neuron.__version__, skopt.__version__))' 2>&1)"; then
        need_build=0
        if [ ! -x x86_64/special ]; then
            need_build=1
        elif [ -n "$(find mod -maxdepth 1 -name '*.mod' -newer x86_64/special -print 2>/dev/null | head -n 1)" ]; then
            need_build=1
        fi
        if [ "$need_build" = "1" ]; then
            if mkdir .nrnivmodl.lock 2>/dev/null; then
                rm -rf x86_64
                nrnivmodl mod > "$LOG_DIR/g7_nrnivmodl.log" 2>&1
                rmdir .nrnivmodl.lock 2>/dev/null
                built="built now (log $LOG_DIR/g7_nrnivmodl.log)"
            else
                built="NOT built: .nrnivmodl.lock exists (a job is building; retry later)"
            fi
        else
            built="up to date"
        fi
        if [ -x x86_64/special ]; then
            report PASS "G7 NEURON and mechanisms" "$(printf '%s' "$out" | tail -n 1); x86_64/special $built"
        else
            report FAIL "G7 NEURON and mechanisms" "$(printf '%s' "$out" | tail -n 1); x86_64/special missing, $built"
        fi
    else
        report FAIL "G7 NEURON and mechanisms" "$(printf '%s' "$out" | tail -n 1)"
    fi

    # --- G8-G10 the existing suites, re-run after Blocks 1-2 -------------------
    run_check "G8 smoke_ih_fit" g8_ih_fit "smoke_ih_fit: 12/12 passed" "$PY" smoke_ih_fit.py
    run_check "G9 smoke_ls_baseline_qc" g9_ls_baseline_qc "SMOKE ls_baseline_qc: 10/10 passed" "$PY" smoke_ls_baseline_qc.py
    run_check "G10 smoke_ih_recovery" g10_ih_recovery "smoke_ih_recovery: 14/14 passed" "$PY" smoke_ih_recovery.py

    # --- G11 the bit-for-bit regression gate against Biological Fit/ -----------
    reg_cell="$ARCHIVE_GROUP/specimen_$REG_SPECIMEN"
    if [ ! -d "../Biological Fit" ]; then
        report SKIP "G11 regression_passive_identity" "the oracle folder ../Biological Fit is not here"
    elif [ ! -d "$reg_cell" ]; then
        report SKIP "G11 regression_passive_identity" "no archive cell at $reg_cell (set IH_ARCHIVE_GROUP / REG_SPECIMEN)"
    else
        run_check "G11 regression_passive_identity" g11_regression "regression_passive_identity: PASS" \
            "$PY" regression_passive_identity.py --ref-dir "../Biological Fit" --archive-cell "$reg_cell"
    fi

    # --- G12 the SS sweep structure Block 3 reads (D-056 open point) -----------
    dirs=""
    missing=""
    for sid in $NPZ_SPECIMENS; do
        if [ -d "$ARCHIVE_GROUP/specimen_$sid" ]; then
            dirs="$dirs|$ARCHIVE_GROUP/specimen_$sid"
        else
            missing="$missing $sid"
        fi
    done
    if [ -z "$dirs" ]; then
        report SKIP "G12 ss_pulses.npz sweep structure" "no specimen directory found under $ARCHIVE_GROUP for:$missing"
    else
        log="$LOG_DIR/g12_ss_pulses.log"
        IFS='|' read -r -a dir_list <<< "${dirs#|}"
        "$PY" noise_fit_gates.py ss-pulses "${dir_list[@]}" > "$log" 2>&1
        rc=$?
        lines="$(grep '^SS_PULSES ' "$log")"
        if [ -n "$lines" ] && [ "$rc" = "0" ]; then
            report PASS "G12 ss_pulses.npz sweep structure" "sweep_number stored in every cell read$([ -n "$missing" ] && echo "; not found:$missing")"
        else
            report FAIL "G12 ss_pulses.npz sweep structure" "exit $rc; see the SS_PULSES lines below and $log"
        fi
        printf '%s\n' "$lines" | tee -a "$LOG_DIR/summary.txt"
    fi
fi

echo "GATES noise-fit: $N_PASS passed, $N_FAIL failed, $N_SKIP skipped  (logs: $LOG_DIR)" | tee -a "$LOG_DIR/summary.txt"
if [ "$N_FAIL" -gt 0 ]; then
    verdict="NOT SETTLED -- $N_FAIL check(s) failed; read the FAIL lines and their logs"
elif [ "$QUICK" = "1" ]; then
    verdict="NOT SETTLED -- QUICK mode ran no NEURON suite, no regression gate and no archive read"
elif [ "$N_SKIP" -gt 0 ]; then
    verdict="NOT SETTLED -- $N_SKIP check(s) skipped; read the SKIP lines"
else
    verdict="SETTLED -- Blocks 1-2 verified on this machine and the SS sweep structure read"
fi
echo "VERDICT: $verdict" | tee -a "$LOG_DIR/summary.txt"
[ "$N_FAIL" -eq 0 ]
