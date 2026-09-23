#!/bin/bash

#PBS -S /bin/bash
#PBS -N "ih_fit"
#PBS -q cpu
#PBS -l select=1:ncpus=1,walltime=100:00:00
#PBS -k eo

##########################################################################
# I_h campaign -- ONE ARM x ONE (layer x type) GROUP per PBS job.
#
# Entry point: run_ih_fit.py. See the plan, stage 6. Differences from
# submit_biological_fit.sh, which this file does NOT replace:
#
#   * ARM. `qsub -v GROUP=L3_exc,ARM=ih6` selects a complete configuration:
#       baseline_runB    | run B reproduced: 3-D, legacy loader, 60 ms window
#       passive_fullstep | 3-D on the D-006 protocol (the control arm)
#       ih6              | 6-D: (C_m, R_m, R_a, gbar_h, dv_h, kappa_tau)
#     Outputs go to <OUTPUT_BASE>/<RUN_TAG>/<ARM>/<GROUP>, so the arms of one
#     run tag cannot overwrite each other.
#
#   * nrnivmodl IS NEEDED HERE. The biological passive fit inserted only
#     `pas` and compiled nothing; the I_h arms insert a real mechanism, so
#     mod/*.mod must be compiled into $CODE_DIR/x86_64/ BEFORE python starts.
#     The guard below is idempotent and mtime-aware: it rebuilds only when
#     x86_64/special is missing or some .mod is newer than it, and it holds a
#     directory lock so that a fan-out of group jobs cannot race on one build.
#
#   * cd "$CODE_DIR" before python. NEURON auto-loads ./x86_64 relative to the
#     PROCESS working directory, so the job must run there or the mechanisms
#     are silently absent and every I_h fit would be a passive fit wearing an
#     'ih6' label. Every path passed on the command line is absolute, so
#     moving the working directory changes nothing else.
#
#   * NO tau_w sweep. run_ih_fit.py takes a scalar --ss-tau-w-ms; run B used a
#     single-point grid, so 5.0 reproduces it. See the entrypoint's docstring.
#
#   * Phase 2.5 is OFF (D-006 Q8). Set RUN_PHASE2P5=1 for a 3-D arm only.
#
# -- Dispatch -----------------------------------------------------------------
#     qsub -v GROUP=L3_exc,ARM=ih6 submit_ih_fit.sh
#     qsub -v GROUP=L3_exc,ARM=passive_fullstep,RUN_TAG=dryrun submit_ih_fit.sh
#     qsub -v GROUP=L3_exc,ARM=ih6,MAX_CELLS=1,PHASE3_SUBSET=none \
#          submit_ih_fit.sh            # the stage-8 dry run
#
# NOTHING NEEDS EDITING BEFORE THE FIRST SUBMISSION. The code directory is
# wherever you run qsub from, and every other path is a knob with a default.
# The one value worth checking is IH_ARCHIVE_ROOT (where the specimen_*
# archives live); the run header prints every resolved path before any work.
##########################################################################

# --- Configuration ----------------------------------------------------------
# --- Paths: derived, not hard-coded -----------------------------------------
# SUBMIT FROM THE "Ih Fit" DIRECTORY. PBS sets PBS_O_WORKDIR to the directory
# qsub was invoked from, and that is where this script looks for the code:
#
#   cd "/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/HPC script/Ih Fit"
#   qsub -v GROUP=L3_exc,ARM=ih6 submit_ih_fit.sh
#
# Deriving the code directory this way rather than hard-coding it means a
# `git pull` into a different checkout, or a renamed clone, cannot leave the
# job running last week's code. It also avoids passing a path containing
# SPACES through `qsub -v`, which is comma-separated and does not survive one.
#
# RESERVED NAMES. The login shell on this cluster exports CODE (and ENV_NAME),
# and PBS jobs source .bashrc -- which has already sent one campaign into the
# wrong directory (paths reference, sections 8.2 and 8.4). Every knob here is
# therefore prefixed IH_, and a bare name is reported and ignored.
for _n in CODE ROOT ARCHIVE_ROOT OUTPUT_BASE CODE_DIR; do
    eval "_v=\${$_n:-}"
    if [ -n "$_v" ]; then
        echo "NOTE: $_n is set ($_v) and is IGNORED; the knobs are"
        echo "      IH_CODE_DIR, IH_ARCHIVE_ROOT, IH_OUTPUT_BASE."
    fi
done
unset _n _v

CODE_DIR="${IH_CODE_DIR:-${PBS_O_WORKDIR:-$(pwd)}}"
# The Phase-0 archives are NOT in git (there is no specimen_* anywhere in the
# repo), so their root is independent of the checkout. The value below is the
# passive-fit working directory the earlier runs used; override with
# IH_ARCHIVE_ROOT if the archives live elsewhere.
ARCHIVE_ROOT="${IH_ARCHIVE_ROOT:-/davinci-1/home/ldellamea/Human Neurons Fitting}"
OUTPUT_BASE="${IH_OUTPUT_BASE:-$ARCHIVE_ROOT/pipeline_outputs_ih}"
RUN_TAG="${RUN_TAG:-default}"
CONDA_ENV="${IH_ENV:-prova}"
ENTRYPOINT="$CODE_DIR/run_ih_fit.py"
MOD_DIR="$CODE_DIR/mod"

if [ ! -f "$ENTRYPOINT" ]; then
    echo "[FATAL] run_ih_fit.py not found under CODE_DIR." >&2
    echo "        CODE_DIR resolved to: $CODE_DIR" >&2
    echo "        Submit from the 'Ih Fit' directory:" >&2
    echo "          cd \"/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/HPC script/Ih Fit\"" >&2
    echo "          qsub -v GROUP=L3_exc,ARM=ih6 submit_ih_fit.sh" >&2
    echo "        (or export IH_CODE_DIR before qsub)." >&2
    exit 2
fi

# Morphology override (C4). Empty = use each archive's own swc_path. When set,
# the loader REFUSES a cell whose SWC is not found here rather than silently
# falling back -- a half-corrected cohort is worse than a failed job.
SWC_DIR="${SWC_DIR:-}"

# The entrypoint imports these from CODE_DIR (all must be present there):
#   run_ih_fit.py   passive_fitting_hpc_fixed.py   passive_long_step_training.py
#   param_spec.py   ih_mechanism.py                human_ih_params.py
#   mod/Ih.mod      mod/Ih_human.mod

# --- Arm (the one knob that matters) ----------------------------------------
ARM="${ARM:-ih6}"
# Override the arm's axis list, e.g. "Cm,Rm,Ra,gbar" for the 4-D arm. A knob
# left out is FROZEN at its base value (D-005). Empty = the arm's own list.
FIT_PARAMS="${FIT_PARAMS:-}"

# --- Data / fit (run-level) -------------------------------------------------
N_AVG_GROUPS=1
FIT_TARGET="hyp"
F_FACTOR="${F_FACTOR:-1.9}"      # D-007 C4: 1.9 for the L2/L3 optimisation
N_CALLS="${N_CALLS:-200}"        # D-005 Q11
N_INITIAL="${N_INITIAL:-100}"
MAX_CELLS="${MAX_CELLS:-}"       # empty = every cell in the group
FAIL_FAST="${FAIL_FAST:-0}"

# --- I_h configuration (D-005 / D-007; never fitted) ------------------------
IH_MECHANISM="${IH_MECHANISM:-Ih_human}"     # Rich 2021 base (D-007 C1)
IH_DISTRIBUTION="${IH_DISTRIBUTION:-uniform}"
IH_REGIONS="${IH_REGIONS:-soma,dend,apic}"   # axon stub excluded
VSHIFT_BASE="${VSHIFT_BASE:-0.0}"            # +20 with IH_MECHANISM=Ih -> Kalmbach
EHCN="${EHCN:-}"                             # empty = the mechanism's own E_h

# --- Search box -------------------------------------------------------------
CM_BOUNDS="0.3,3.0"
RM_BOUNDS="1000.0,100000.0"
RA_BOUNDS="50.0,1000.0"
GBAR_BOUNDS="${GBAR_BOUNDS:-1e-6,1e-3}"
DVH_BOUNDS="${DVH_BOUNDS:--10.0,10.0}"
KAPPA_BOUNDS="${KAPPA_BOUNDS:-0.5,2.0}"

# --- Protocol (plan section 5 / D-006); empty = the arm's own value ---------
LS_WINDOW="${LS_WINDOW:-}"               # after_onset | step | sweep
LS_WINDOW_MS="${LS_WINDOW_MS:-}"
LS_MAX_AMPLITUDE_PA="${LS_MAX_AMPLITUDE_PA:-}"   # 'none' admits every sweep
LS_DEP_MAX_AMPLITUDE_PA="${LS_DEP_MAX_AMPLITUDE_PA:-100.0}"
N_DROP_WEAKEST="${N_DROP_WEAKEST:-1}"    # h_1 validates
N_DROP_STRONGEST="${N_DROP_STRONGEST:-1}"  # h_n is report-only (PIR unmodelled)
DEP_N_VALIDATION="${DEP_N_VALIDATION:-3}"
V_TROUGH_MIN="${V_TROUGH_MIN:-}"         # empty = off
TRAIN_ALL_HYP="${TRAIN_ALL_HYP:-0}"      # D-007's labelled opt-in
GATE_VALID_VIA="${GATE_VALID_VIA:-}"     # empty = the arm's own choice

# --- Loss -------------------------------------------------------------------
R_IN_TARGET="peak"
WEIGHTING="relative"
SS_WINDOW_MS="0.5,100.0"
SS_T0_MS=""
SS_TIME_WEIGHT="exp"
SS_TAU_W_MS="${SS_TAU_W_MS:-5.0}"        # ONE value; a grid is refused

# --- Integration step -------------------------------------------------------
# D-006 Q9/C3: the baseline (run B) integrated at 0.1 ms, so every arm does,
# or the arms are not comparable. This is 4x cheaper than the shipped 0.025 ms
# AND a first-order accuracy change; it is inherited by decision, not default.
DT_BRIEF_MS="${DT_BRIEF_MS:-0.1}"
DT_LONG_MS="${DT_LONG_MS:-0.1}"

# --- Phase 2.5 (OFF; D-006 Q8) ----------------------------------------------
RUN_PHASE2P5="${RUN_PHASE2P5:-0}"        # 1 = on, 3-D arms only
N_FLOOR="${N_FLOOR:-4}"
N_RA_PROFILE="${N_RA_PROFILE:-50}"

# --- Phase 3 (bootstrap CIs) ------------------------------------------------
# COST: each bootstrapped cell costs BOOTSTRAP_B * BOOTSTRAP_N_CALLS loss
# evaluations (here 200 * 100 = 20,000) against ~200 for its Phase-2 fit.
# MEASURE ONE CELL in the stage-8 dry run before fanning a group out.
PHASE3_SUBSET="${PHASE3_SUBSET:-frac:0.5}"
BOOTSTRAP_B="${BOOTSTRAP_B:-200}"
BOOTSTRAP_MODE="${BOOTSTRAP_MODE:-nonparametric}"
NOISE_MODE="${NOISE_MODE:-block}"
BOOTSTRAP_N_CALLS="${BOOTSTRAP_N_CALLS:-100}"   # D-007 C5
BOOTSTRAP_N_INITIAL="${BOOTSTRAP_N_INITIAL:-30}"
# ----------------------------------------------------------------------------

# --- Resolve per-job paths from $GROUP, $ARM and $RUN_TAG -------------------
if [ -z "${GROUP:-}" ]; then
    echo "[FATAL] \$GROUP is not set. Submit with:" >&2
    echo "    qsub -v GROUP=<layer_type>,ARM=<arm> submit_ih_fit.sh" >&2
    exit 2
fi

case "$ARM" in
    baseline_runB|passive_fullstep|ih6) ;;
    *) echo "[FATAL] ARM must be baseline_runB | passive_fullstep | ih6" >&2
       echo "        (got: '$ARM')" >&2
       exit 2 ;;
esac

# RUN_TAG becomes a path component AND part of the PBS job name, and travels
# through the comma-separated `qsub -v` list -- reject anything that breaks any
# of those (spaces, commas, slashes, quotes).
case "$RUN_TAG" in
    ""|*[!A-Za-z0-9_.-]*)
        echo "[FATAL] RUN_TAG must be non-empty and contain only letters," >&2
        echo "        digits, '_', '.', '-'  (got: '$RUN_TAG')" >&2
        exit 4 ;;
esac

ARCHIVE_DIR="$ARCHIVE_ROOT/$GROUP"
OUTPUT_DIR="$OUTPUT_BASE/$RUN_TAG/$ARM/$GROUP"

if [ ! -d "$ARCHIVE_DIR" ]; then
    echo "[FATAL] Archive directory does not exist: $ARCHIVE_DIR" >&2
    exit 3
fi
if [ -n "$SWC_DIR" ] && [ ! -d "$SWC_DIR" ]; then
    echo "[FATAL] SWC_DIR set but missing: $SWC_DIR" >&2
    exit 3
fi
# ----------------------------------------------------------------------------

# Activate the conda env (it supplies python + NEURON + nrnivmodl). We do NOT
# `module load python` -- on this cluster that produced Lmod errors. This is
# the activation block already verified on davinci by submit_biological_fit.sh;
# it is reused verbatim rather than "improved", because the one that works is
# the one that works.
source "$(conda info --base)/etc/profile.d/conda.sh" \
    || { echo "[FATAL] cannot source conda.sh" >&2; exit 5; }
conda activate "$CONDA_ENV" \
    || { echo "[FATAL] cannot activate conda env '$CONDA_ENV'" >&2; exit 5; }

# NEURON auto-loads ./x86_64 from the PROCESS working directory. Compile there
# and run there. (PBS_O_WORKDIR is only logged; every path below is absolute.)
cd "$CODE_DIR" || { echo "[FATAL] cannot cd to CODE_DIR: $CODE_DIR" >&2; exit 5; }

# ============================================================================
#  nrnivmodl guard -- idempotent, mtime-aware, lock-protected
# ============================================================================
SPECIAL="$CODE_DIR/x86_64/special"
LOCK="$CODE_DIR/.nrnivmodl.lock"

if [ ! -d "$MOD_DIR" ]; then
    echo "[FATAL] mod dir not found: $MOD_DIR" >&2
    exit 6
fi

need_build=0
build_reason=""
if [ ! -x "$SPECIAL" ]; then
    need_build=1
    build_reason="x86_64/special missing or not executable"
else
    newer_mod=$(find "$MOD_DIR" -maxdepth 1 -name '*.mod' -newer "$SPECIAL" \
                     -print 2>/dev/null | head -n 1)
    if [ -n "$newer_mod" ]; then
        need_build=1
        build_reason="$(basename "$newer_mod") is newer than x86_64/special"
    fi
fi

if [ "$need_build" = "1" ]; then
    echo "[nrnivmodl] rebuild needed: $build_reason"
    if mkdir "$LOCK" 2>/dev/null; then
        trap 'rmdir "$LOCK" 2>/dev/null' EXIT
        # Remove the stale tree first: nrnivmodl otherwise keeps object files
        # whose .mod no longer exists, and a deleted mechanism stays loadable.
        rm -rf "$CODE_DIR/x86_64"
        nrnivmodl "$MOD_DIR"
        rc=$?
        rmdir "$LOCK" 2>/dev/null
        trap - EXIT
        if [ $rc -ne 0 ]; then
            echo "[FATAL] nrnivmodl failed (exit $rc)" >&2
            exit 6
        fi
    else
        echo "[nrnivmodl] another job holds the lock; waiting (<= 10 min) ..."
        waited=0
        while [ -d "$LOCK" ] && [ $waited -lt 600 ]; do
            sleep 5
            waited=$((waited + 5))
        done
        echo "[nrnivmodl] waited ${waited}s"
    fi
else
    echo "[nrnivmodl] up to date: $SPECIAL"
fi

if [ ! -x "$SPECIAL" ]; then
    echo "[FATAL] $SPECIAL still absent after the build step." >&2
    exit 6
fi

# VERIFY the mechanisms actually load. Without this a missing x86_64 makes
# every insert() fail at run time -- inside a loss builder that turns every
# exception into a finite 1e6 penalty, i.e. a silently wrong fit.
python - <<'PYCHK'
import sys
from neuron import h
missing = [m for m in ("Ih", "Ih_human") if not hasattr(h, m)]
if missing:
    sys.stderr.write("[FATAL] mechanism(s) not loaded: %s\n" % ", ".join(missing))
    sys.exit(7)
print("[nrnivmodl] mechanisms loaded: Ih, Ih_human")
PYCHK
if [ $? -ne 0 ]; then
    echo "[FATAL] mechanism verification failed." >&2
    exit 7
fi
# ============================================================================

mkdir -p "$OUTPUT_DIR"

# Sanity prints
echo "Running on node:        $(hostname)"
echo "Job ID:                 $PBS_JOBID"
echo "Run tag:                $RUN_TAG"
echo "Arm:                    $ARM   (fit_params override: '${FIT_PARAMS:-<arm default>}')"
echo "Group:                  $GROUP"
echo "Submit dir:             ${PBS_O_WORKDIR:-<unset>}"
echo "Working dir (process):  $(pwd)"
echo "Python:                 $(which python)   | conda: $CONDA_DEFAULT_ENV"
echo "Code dir (resolved):    $CODE_DIR"
echo "Entrypoint:             $ENTRYPOINT"
echo "Archive dir (this job): $ARCHIVE_DIR"
echo "SWC dir:                ${SWC_DIR:-<the archive own path>}"
echo "Output dir (this job):  $OUTPUT_DIR"
echo "Fit:                    target=$FIT_TARGET  F=$F_FACTOR  n_calls=$N_CALLS  n_initial=$N_INITIAL  [sequential]"
echo "I_h:                    $IH_MECHANISM / $IH_DISTRIBUTION / regions=$IH_REGIONS / vshift_base=${VSHIFT_BASE}mV"
echo "Box:                    gbar=[$GBAR_BOUNDS]  dv_h=[$DVH_BOUNDS]  kappa=[$KAPPA_BOUNDS]"
echo "Protocol:               drop $N_DROP_WEAKEST weakest / $N_DROP_STRONGEST strongest  dep_valid=$DEP_N_VALIDATION  ls_window='${LS_WINDOW:-<arm default>}'"
echo "Loss:                   r_in=$R_IN_TARGET  weighting=$WEIGHTING  ss_window=[$SS_WINDOW_MS]ms  tau_w=${SS_TAU_W_MS}ms"
echo "dt (ENFORCED):          brief=${DT_BRIEF_MS}ms  long=${DT_LONG_MS}ms"
if [ "$RUN_PHASE2P5" = "1" ]; then
    echo "Phase 2.5:              ON  n_floor=$N_FLOOR  n_ra_profile=$N_RA_PROFILE  (3-D arms only)"
else
    echo "Phase 2.5:              OFF (D-006 Q8); R_a free; Phase 3 over the full theta"
fi
echo "Phase 3:                subset='$PHASE3_SUBSET'  B=$BOOTSTRAP_B  mode=$BOOTSTRAP_MODE  ${BOOTSTRAP_N_CALLS}/${BOOTSTRAP_N_INITIAL}"
echo "-----------------------------------------"

# --- Build the argument list (this IS the entrypoint CLI contract) ----------
ARGS=(
    --archive-dir            "$ARCHIVE_DIR"
    --output-dir             "$OUTPUT_DIR"
    --code-dir               "$CODE_DIR"
    --arm                    "$ARM"
    --n-avg-groups           "$N_AVG_GROUPS"
    --fit-target             "$FIT_TARGET"
    --F                      "$F_FACTOR"
    --n-calls                "$N_CALLS"
    --n-initial              "$N_INITIAL"
    # I_h configuration
    --ih-mechanism           "$IH_MECHANISM"
    --ih-distribution        "$IH_DISTRIBUTION"
    --ih-regions             "$IH_REGIONS"
    --vshift-base            "$VSHIFT_BASE"
    # search box (values may start with '-', so pass every one with '=' to
    # keep argparse from reading the minus as the start of an option)
    "--cm-bounds=$CM_BOUNDS"
    "--rm-bounds=$RM_BOUNDS"
    "--ra-bounds=$RA_BOUNDS"
    "--gbar-bounds=$GBAR_BOUNDS"
    "--dvh-bounds=$DVH_BOUNDS"
    "--kappa-bounds=$KAPPA_BOUNDS"
    # protocol
    --n-drop-weakest         "$N_DROP_WEAKEST"
    --n-drop-strongest       "$N_DROP_STRONGEST"
    --dep-n-validation       "$DEP_N_VALIDATION"
    --ls-dep-max-amplitude-pA "$LS_DEP_MAX_AMPLITUDE_PA"
    # loss
    --r-in-target            "$R_IN_TARGET"
    --weighting              "$WEIGHTING"
    --ss-window-ms           "$SS_WINDOW_MS"
    --ss-time-weight         "$SS_TIME_WEIGHT"
    --ss-tau-w-ms            "$SS_TAU_W_MS"
    --dt-brief-ms            "$DT_BRIEF_MS"
    --dt-long-ms             "$DT_LONG_MS"
    # Phase 2.5 / Phase 3
    --n-floor                "$N_FLOOR"
    --n-ra-profile           "$N_RA_PROFILE"
    --phase3-subset          "$PHASE3_SUBSET"
    --bootstrap-B            "$BOOTSTRAP_B"
    --bootstrap-mode         "$BOOTSTRAP_MODE"
    --noise-mode             "$NOISE_MODE"
    --bootstrap-n-calls      "$BOOTSTRAP_N_CALLS"
    --bootstrap-n-initial    "$BOOTSTRAP_N_INITIAL"
)

# Optional flags: passed only when set, so the arm's own defaults survive.
[ -n "$FIT_PARAMS" ]             && ARGS+=(--fit-params "$FIT_PARAMS")
[ -n "$SWC_DIR" ]                && ARGS+=(--swc-dir "$SWC_DIR")
[ -n "$EHCN" ]                   && ARGS+=("--ehcn=$EHCN")
[ -n "$LS_WINDOW" ]              && ARGS+=(--ls-window "$LS_WINDOW")
[ -n "$LS_WINDOW_MS" ]           && ARGS+=(--ls-window-ms "$LS_WINDOW_MS")
[ -n "$LS_MAX_AMPLITUDE_PA" ]    && ARGS+=(--ls-max-amplitude-pA "$LS_MAX_AMPLITUDE_PA")
[ -n "$V_TROUGH_MIN" ]           && ARGS+=("--v-trough-min=$V_TROUGH_MIN")
[ -n "$GATE_VALID_VIA" ]         && ARGS+=(--gate-valid-via "$GATE_VALID_VIA")
[ -n "$SS_T0_MS" ]               && ARGS+=(--ss-t0-ms "$SS_T0_MS")
[ -n "$MAX_CELLS" ]              && ARGS+=(--max-cells "$MAX_CELLS")
[ "$TRAIN_ALL_HYP" = "1" ]       && ARGS+=(--train-all-hyp)
[ "$RUN_PHASE2P5" = "1" ]        && ARGS+=(--run-phase2p5)
[ "$FAIL_FAST" = "1" ]           && ARGS+=(--fail-fast)

# --- Dry run ----------------------------------------------------------------
# `qsub -v GROUP=...,ARM=ih6,DRY_RUN=1` exercises everything that can fail
# fast -- conda activation, the nrnivmodl guard, the mechanism load, the path
# resolution and the CLI contract -- and stops before the fit. Run this once
# per configuration before committing a 100-hour walltime to it.
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY RUN] would run:"
    echo "  python $ENTRYPOINT \\"
    printf '    %s\n' "${ARGS[@]}"
    # The CLI contract, checked rather than printed: the entrypoint's OWN
    # parser reads exactly these arguments. A renamed or mistyped flag then
    # fails here, in seconds, instead of after the job has queued.
    python - "${ARGS[@]}" <<'PYEOF' || { echo "[FATAL] run_ih_fit's parser REFUSED the arguments above" >&2; conda deactivate; exit 2; }
import sys
import run_ih_fit as _E
_E._parse_args(sys.argv[1:])
print("[DRY RUN] CLI contract: run_ih_fit's own parser accepts every argument")
PYEOF
    echo "[DRY RUN] env + mechanisms + CLI verified; no fit was run."
    conda deactivate
    exit 0
fi

# --- Run --------------------------------------------------------------------
python "$ENTRYPOINT" "${ARGS[@]}"
status=$?

conda deactivate
echo "[done] run '$RUN_TAG' arm $ARM group $GROUP exited with status $status"
sleep 5s
exit $status
