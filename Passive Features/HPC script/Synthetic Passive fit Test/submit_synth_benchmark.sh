#!/bin/bash

#PBS -S /bin/bash
#PBS -N "synth_passive_fit"
#PBS -q cpu
#PBS -l select=1:ncpus=10,walltime=10:00:00
#PBS -k eo

##########################################################################
# Synthetic passive-fit BENCHMARK — one PBS job per COHORT ($GROUP).
#
# A cohort == a PBS group == the unit Phase 2.5 fixes Ra over (so Ra is
# shared within a cohort by construction; see synth_gt_grid.py). The job:
#   1. reads its cohort's rows from the seeded MANIFEST,
#   2. GENERATES those synthetic archives on the compute node
#      (synthetic_ground_truth.py; per-cell Cm,Rm,gIhbar,noise from the row),
#   3. fits each cell with the TWO-PASS auto-tau_w loop
#      (interim cm_profile_sweep -> pick sharpest HW_rho -> refit at that tau_w),
#   4. runs Phase 2.5 (fix Ra at the cohort median, refit Cm,Rm),
#   5. runs Phase 3 (bootstrap CIs) ONLY for the held-out calibration subset.
#
# Sequential, one cell at a time (NEURON's global section list is wiped
# between cells), matching the real-data pipeline. N_WORKERS is hard-coded
# to 1 inside the Python entrypoint.
#
# ── Dispatch ────────────────────────────────────────────────────────────
#     qsub -v GROUP=cohort_0003 submit_synth_benchmark.sh
# Use submit_all_cohorts.sh to fan out one job per cohort automatically.
#
# EDIT THE "USER CONFIG" BLOCK BELOW BEFORE SUBMITTING.
##########################################################################

# ─── USER CONFIG ────────────────────────────────────────────────────────
# Absolute paths recommended (jobs don't inherit $PWD reliably).
CODE_DIR="/davinci-1/home/ldellamea/Human Neurons Fitting/synthetic_benchmark"
MANIFEST="$CODE_DIR/manifest.csv"                 # built once by submit_all_cohorts.sh
ARCHIVE_ROOT="$CODE_DIR/synthetic_archive"        # generated archives: <ROOT>/<GROUP>/
OUTPUT_ROOT="$CODE_DIR/synthetic_out"             # fit outputs:        <ROOT>/<GROUP>/
ENTRYPOINT="$CODE_DIR/run_synth_benchmark.py"     # the Phase-2 Python driver (next deliverable)

# The fitting/generation modules must be importable from $CODE_DIR:
#   synth_gt_grid.py  cm_profile_sweep.py  synthetic_ground_truth.py
#   passive_long_step_training.py  phase1_data_loader.py  phase2_optimiser.py
#   passive_fitting_hpc_fixed.py
# NEURON mechanisms for I_h generation (compiled here if not already):
MOD_DIR="$CODE_DIR/mod"                            # contains Ih.mod (+ na.mod/kv.mod if active)

# ─── Generation protocol (run-level; per-cell GT/noise come from MANIFEST) ─
N_AVG_GROUPS=3                 # sweep-average groups per polarity
SS_N_REPEATS=30               # square-subthreshold repeats
LS_HYP_AMPS="-10,-30,-50,-70,-90"   # long-square hyperpolarising amplitudes [pA]

# ─── Phase 2 (fit) ──────────────────────────────────────────────────────
FIT_TARGET="hyp"              # dep | hyp | both
N_CALLS=100                   # GP optimiser evaluations per cell
N_INITIAL=50                  # random initial points before GP takes over

# ─── Interim two-pass auto-tau_w (cm_profile_sweep) ─────────────────────
N_LONG_TRAIN=2                # smallest-|amp| hyp LS steps folded into TRAINING
LS_DEFLECTION_CAP_MV=12.0     # I_h deflection guard for long-step admission
R_IN_TARGET="peak"           # peak | steady
WEIGHTING="relative"         # cross-bundle loss weighting
SS_WINDOW_MS="0.5,100.0"      # SS (start=offset,end); start is the C_m choice
SS_T0_MS=""                   # empty => SS window start
SS_TIME_WEIGHT="exp"         # exp | gauss | none
LS_WINDOW_MS=150.0            # LS RMSD window length from onset
TAU_W_GRID_MS="2.0,5.0,10.0"  # per-cell sweep grid; winner = sharpest HW_rho
SWEEP_RHO=0.5                 # relative-rise threshold for HW_rho
SWEEP_N_GRID=41               # log C_m grid points
# (CM/RM/RA sweep boxes mirror the fit box inside the entrypoint.)

# ─── Phase 2.5 (MANDATORY here: fix Ra per cohort, refit Cm,Rm) ──────────
SKIP_PHASE2P5=0               # 1 = legacy free-Ra diagnostic; 0 = standard
N_FLOOR=4                     # min qualifying cells for cohort-median Ra
N_RA_PROFILE=50               # Ra grid points for the RMSD-vs-Ra profile

# ─── Phase 3 (bootstrap CIs) — SUBSET ONLY (calibration check) ──────────
# Which cells in this cohort get bootstrapped. Examples:
#   ""          -> none in this cohort
#   "all"       -> every cell (expensive)
#   "first:2"   -> first 2 cells of the cohort
#   "frac:0.2"  -> ~20% of the cohort (deterministic by specimen_id hash)
PHASE3_SUBSET="first:2"
BOOTSTRAP_B=200
BOOTSTRAP_MODE="nonparametric"   # parametric | nonparametric
NOISE_MODE="block"               # iid | ar1 | block  (parametric only)
BOOTSTRAP_N_CALLS=40
BOOTSTRAP_N_INITIAL=20
# ────────────────────────────────────────────────────────────────────────

# ─── Resolve per-job paths from $GROUP ──────────────────────────────────
if [ -z "$GROUP" ]; then
    echo "[FATAL] \$GROUP is not set. Submit with:" >&2
    echo "    qsub -v GROUP=<cohort_label> submit_synth_benchmark.sh" >&2
    echo "    e.g. qsub -v GROUP=cohort_0003 submit_synth_benchmark.sh" >&2
    exit 2
fi
if [ ! -f "$MANIFEST" ]; then
    echo "[FATAL] manifest not found: $MANIFEST" >&2
    echo "        build it first (submit_all_cohorts.sh does this)." >&2
    exit 3
fi

ARCHIVE_DIR="$ARCHIVE_ROOT/$GROUP"
OUTPUT_DIR="$OUTPUT_ROOT/$GROUP"
# ────────────────────────────────────────────────────────────────────────

cd "$PBS_O_WORKDIR"

# Load modules + conda env (source conda.sh in non-interactive shells)
module load python3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prova

mkdir -p "$ARCHIVE_DIR" "$OUTPUT_DIR"

# ─── Compile NEURON mechanisms for I_h generation (once per node/workdir) ─
# The entrypoint runs from $CODE_DIR; NEURON auto-loads x86_64/ from there.
if [ -d "$MOD_DIR" ] && [ ! -d "$CODE_DIR/x86_64" ]; then
    echo "[mech] compiling .mod from $MOD_DIR"
    ( cd "$CODE_DIR" && nrnivmodl "$MOD_DIR" ) || {
        echo "[FATAL] nrnivmodl failed — check $MOD_DIR/*.mod" >&2; exit 4; }
fi

# Sanity prints
echo "Running on node:        $(hostname)"
echo "Job ID:                 $PBS_JOBID"
echo "Cohort ($GROUP):        $GROUP"
echo "Python:                 $(which python3)   | conda: $CONDA_DEFAULT_ENV"
echo "Entrypoint:             $ENTRYPOINT"
echo "Manifest:               $MANIFEST"
echo "Archive dir (this job): $ARCHIVE_DIR"
echo "Output dir (this job):  $OUTPUT_DIR"
echo "Fit:                    target=$FIT_TARGET  n_calls=$N_CALLS  n_initial=$N_INITIAL  [sequential]"
echo "Auto-tau_w:             grid=[$TAU_W_GRID_MS] ms  shape=$SS_TIME_WEIGHT  rho=$SWEEP_RHO  n_grid=$SWEEP_N_GRID  (two-pass: sweep -> refit)"
if [ "$SKIP_PHASE2P5" = "1" ]; then
    echo "Phase 2.5:              SKIPPED (legacy free-Ra)"
else
    echo "Phase 2.5:              ON  n_floor=$N_FLOOR  n_ra_profile=$N_RA_PROFILE  (Ra fixed at cohort median)"
fi
echo "Phase 3:                subset='$PHASE3_SUBSET'  B=$BOOTSTRAP_B  mode=$BOOTSTRAP_MODE  noise=$NOISE_MODE"
echo "-----------------------------------------"

# ─── Build the argument list (this IS the entrypoint CLI contract) ───────
ARGS=(
    --manifest            "$MANIFEST"
    --group               "$GROUP"
    --archive-dir         "$ARCHIVE_DIR"
    --output-dir          "$OUTPUT_DIR"
    --code-dir            "$CODE_DIR"
    # generation
    --n-avg-groups        "$N_AVG_GROUPS"
    --ss-n-repeats        "$SS_N_REPEATS"
    --ls-hyp-amps         "$LS_HYP_AMPS"
    # fit
    --fit-target          "$FIT_TARGET"
    --n-calls             "$N_CALLS"
    --n-initial           "$N_INITIAL"
    # interim two-pass auto-tau_w
    --n-long-train        "$N_LONG_TRAIN"
    --ls-deflection-cap   "$LS_DEFLECTION_CAP_MV"
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
    # Phase 3 (subset only)
    --phase3-subset       "$PHASE3_SUBSET"
    --bootstrap-B         "$BOOTSTRAP_B"
    --bootstrap-mode      "$BOOTSTRAP_MODE"
    --noise-mode          "$NOISE_MODE"
    --bootstrap-n-calls   "$BOOTSTRAP_N_CALLS"
    --bootstrap-n-initial "$BOOTSTRAP_N_INITIAL"
)
[ -n "$SS_T0_MS" ]        && ARGS+=(--ss-t0-ms "$SS_T0_MS")
[ "$SKIP_PHASE2P5" = "1" ] && ARGS+=(--skip-phase2p5)

# ─── Run ────────────────────────────────────────────────────────────────
python3 "$ENTRYPOINT" "${ARGS[@]}"
status=$?

conda deactivate
echo "[done] cohort $GROUP exited with status $status"
sleep 5s
exit $status
