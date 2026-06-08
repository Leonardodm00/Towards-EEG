#!/bin/bash
#
# Wrapper: submits one PBS job per group folder, each pinned to its own node.
#
# Each job receives $GROUP as an environment variable; submit_passive_fit.sh
# then loads cells from <ARCHIVE_ROOT>/<GROUP> and writes results into
# <OUTPUT_ROOT>/<GROUP>.
#
# Usage:
#     ./submit_all_groups.sh                  # submit every group in DEFAULT_GROUPS
#     ./submit_all_groups.sh L2_exc L3_inh    # submit only the named groups
#
# Logs (kept in $HOME via `#PBS -k eo`) will be named
#     passive_fit_<group>.o<jobid>   passive_fit_<group>.e<jobid>
# because we override the job name on the qsub command line.
##########################################################################

# ─── USER CONFIG ────────────────────────────────────────────────────────
# Must match ARCHIVE_ROOT inside submit_passive_fit.sh — used here only to
# sanity-check that each group directory exists before we waste a qsub.
ARCHIVE_ROOT="/davinci-1/home/ldellamea/Human Neurons Fitting"
SUBMIT_SCRIPT="/davinci-1/home/ldellamea/Human Neurons Fitting/submit_passive_fit.sh"

# Default list of groups to submit when no CLI args are given.
# Comment out any you don't want to run.
DEFAULT_GROUPS=(
    L2_exc
    L2_inh
    L3_exc
    L3_inh
    L4_exc
    L4_inh
    L5_exc
    L5_inh
    L6_exc
    L6_inh
)

# Per-group spine-area correction factor F (Eyal-style scaling).
# Any group not listed here falls back to F_FACTOR_DEFAULT (with a warning).
F_FACTOR_DEFAULT=1.9
declare -A F_PER_GROUP=(
    [L2_exc]=1.9
    [L2_inh]=1
    [L3_exc]=1.9
    [L3_inh]=1
    [L4_exc]=1.5
    [L4_inh]=1
    [L5_exc]=1.3
    [L5_inh]=1
    [L6_exc]=1.4
    [L6_inh]=1
)

# ─── Phase 2.5 controls ──────────────────────────────────────────────────
# Phase 2.5 (fix Ra per group + refit Cm,Rm) runs by default for every group.
# Its tunables (SKIP_PHASE2P5, N_FLOOR, N_RA_PROFILE) are methodological
# constants, NOT per-layer quantities like F, so they are NOT mapped per group
# here. Set them once for the whole sweep below; they are forwarded to every
# job via `qsub -v`. To change them, edit these three lines (or override a
# single manual job directly, e.g. `qsub -v GROUP=L4_exc,N_FLOOR=3 ...`).
SKIP_PHASE2P5=0     # 1 = skip Phase 2.5 everywhere (legacy free-Ra), 0 = run it
N_FLOOR=3           # min qualifying cells for a cohort-median Ra (else literature)
N_RA_PROFILE=100     # Ra grid points for the RMSD-vs-Ra profile
# ────────────────────────────────────────────────────────────────────────

# If groups were passed on the command line, use those; else use defaults.
if [ "$#" -gt 0 ]; then
    SEL_GROUPS=( "$@" )
else
    SEL_GROUPS=( "${DEFAULT_GROUPS[@]}" )
fi

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "[FATAL] Submit script not found: $SUBMIT_SCRIPT" >&2
    exit 1
fi

echo "Submitting ${#SEL_GROUPS[@]} group(s) — one PBS job each."
echo "Archive root:  $ARCHIVE_ROOT"
echo "Submit script: $SUBMIT_SCRIPT"
if [ "$SKIP_PHASE2P5" = "1" ]; then
    echo "Phase 2.5:     SKIPPED for all groups (legacy free-Ra)"
else
    echo "Phase 2.5:     ON for all groups  (N_FLOOR=$N_FLOOR  N_RA_PROFILE=$N_RA_PROFILE)"
fi
echo "----------------------------------------"

n_ok=0
n_skip=0
for g in "${SEL_GROUPS[@]}"; do
    if [ ! -d "$ARCHIVE_ROOT/$g" ]; then
        echo "[skip] $g — directory does not exist: $ARCHIVE_ROOT/$g"
        n_skip=$((n_skip + 1))
        continue
    fi
    f="${F_PER_GROUP[$g]}"
    if [ -z "$f" ]; then
        f="$F_FACTOR_DEFAULT"
        echo "[warn] $g not in F_PER_GROUP — falling back to F=$f"
    fi
    echo "[submit] $g  (F=$f)"
    qsub -v "GROUP=$g,F_FACTOR=$f,SKIP_PHASE2P5=$SKIP_PHASE2P5,N_FLOOR=$N_FLOOR,N_RA_PROFILE=$N_RA_PROFILE" \
         -N "passive_fit_${g}" "$SUBMIT_SCRIPT"
    n_ok=$((n_ok + 1))
done

echo "----------------------------------------"
echo "Submitted: $n_ok   Skipped: $n_skip"
