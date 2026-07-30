#!/bin/bash
##########################################################################
# submit_all_cohorts.sh -- build the manifest ONCE, then fan out one
# PBS job per cohort (qsub -v GROUP=<cohort>).
#
# Run on the LOGIN node, AFTER activating the env:
#     conda activate prova
#     bash submit_all_cohorts.sh
#
# v3 changes vs pilot:
#   - Physiological (Cm, tau_m) rejection filter active by default.
#   - CELLS_PER_COHORT raised to 8 so Phase 2.5 median is meaningful.
#   - MAX_CELLS removed (empty = all morphologies).
#   - Ra stream isolated: changing Cm/tau_m constraints does not shift Ra.
#
# v4 changes vs v3 (HUMAN h-current):
#   - h-current kinetics switched from the RODENT Kole et al. (2006) model
#     (Ih.mod, used unaltered in Hay et al. 2011) to the HUMAN model of
#     Rich et al. (2021) (Ih_human.mod). Peak activation time constant
#     ~78 ms -> ~343 ms.
#   - gIhbar 2e-4 (Hay rat L5) -> 1e-4 S/cm2 (Kalmbach et al. 2018 human
#     deep L3), distribution hay_exponential -> uniform to match.
#   - ehcn -45.0 (rodent) -> -49.85 mV (Rich et al. 2021 Table 1).
#   - Ra ground-truth draw restricted to a human physiological window; the
#     FITTER box is unchanged, so Ra recovery is still a real test.
#   - Manifest / archive / output names bumped to v70human so the rodent
#     v60ls run is preserved for comparison.
#   NOTE: all text in this file is plain ASCII on purpose (transfer safety).
##########################################################################

set -uo pipefail

# --- USER CONFIG -----------------------------------------------------------
CODE_DIR="/davinci-1/home/ldellamea/Human Neurons Fitting/Synthetic Test"
MANIFEST="$CODE_DIR/manifest_v70human.csv"
JOB_SCRIPT="$CODE_DIR/submit_synth_benchmark.sh"
CONDA_ENV="prova"

# Morphology pool:
MORPH_ROOT="/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc"
MORPH_GLOB="specimen_*/reconstruction.swc"

# Manifest design:
SEED=0
DRAWS_PER_MORPH=1       # increase if you want more draws per morphology
CELLS_PER_COHORT=3      # NOTE: the v3 comment says >=8 for a meaningful
                        # cohort-median Ra in Phase 2.5. Left at 3 because
                        # SKIP_PHASE2P5=1 in submit_synth_benchmark.sh.
                        # Raise this if you turn Phase 2.5 back on.
RA_MODE="per_cohort"
E_PAS=-70.0
F_FACTOR=1.9            # spine-area correction; Eyal 2016 / Kalmbach 2018
# MAX_CELLS intentionally omitted -> use all morphologies

# --- Physiological prior constraints (v3) ----------------------------------
# Rejection-sample (Cm, Rm) so that:
#   CM_PHYS_LO <= Cm [uF/cm^2] <= CM_PHYS_HI
#   TAU_LO_MS  <= tau_m = Rm*Cm*1e-3 [ms] <= TAU_HI_MS
#
# Rationale:
#   Human L2/3 Cm ~0.45 uF/cm^2 (Eyal 2016); standard ~1.0. Window [0.4, 1.5]
#   covers the anomalously-low human value and the standard value with margin.
#   tau_m [3, 40] ms spans the physiological range for L3 pyramidal neurons
#   fitted by this pipeline (Allen data) without reaching the 100+ ms regime
#   where the 100 ms fit window becomes the binding constraint.
#
#   NOTE (v4): this window is on the PASSIVE tau_m = Rm*Cm*1e-3, whereas the
#   experimental human values (Moradi Chameh et al. 2021: L2&3 13.7 +/- 7.1 ms,
#   L3c 17.1 +/- 5.7 ms) are MEASURED with I_h intact and are therefore
#   systematically smaller than the passive tau_m. Tightening this window onto
#   the measured numbers would bias the ground truth low. Left unchanged.
#
# Individual (Cm, Rm) draws remain within the fitter search box
# ([0.3, 3.0] and [1000, 100000]), so assert_bounds_match_phase1 still passes.
# Set --no-phy-filter on the synth_gt_grid.py call to disable (v2 behaviour).
CM_PHYS_LO=0.4      # uF/cm^2
CM_PHYS_HI=1.5      # uF/cm^2
TAU_LO_MS=3.0       # ms
TAU_HI_MS=40.0      # ms

# --- Ra ground-truth draw window (v4, NEW) ---------------------------------
# Eyal et al. 2016 (eLife 5:e16553) human L2/3 model fits, n=6:
#   Ra 203-384 Ohm*cm, mean 268.5 +/- 30.0.
# Kalmbach et al. 2018 human deep-L3, uniform-passive variant: Ra = 350 Ohm*cm.
# Previously Ra was drawn log-uniform over the WHOLE fitter box [50, 1000],
# so many synthetic cells received a frankly non-human Ra (60 or 900 Ohm*cm).
# The FITTER box is untouched, so Ra recovery remains a genuine test.
# Set both to empty strings to restore the v3 full-box behaviour.
RA_PHYS_LO=100.0    # Ohm*cm
RA_PHYS_HI=500.0    # Ohm*cm

# --- h-current: HUMAN parameters (v4) --------------------------------------
# KINETICS  Rich, Moradi Chameh, Sekulic, Valiante & Skinner (2021),
#           Cereb Cortex 31(2):845-872, doi:10.1093/cercor/bhaa261,
#           Eq. (1) + Table 1 "L5 Human model". Implemented in Ih_human.mod.
#           Peak activation time constant ~343 ms (at -74 mV) versus ~78 ms
#           for the Kole et al. (2006) rat model that Ih.mod implements.
#           The two curves CROSS near -85 mV: above it the human current is
#           4-7x slower, below it faster. The hyperpolarising long steps used
#           here sit in the -70 to -85 mV band, i.e. squarely in the region
#           where the rodent model was running 4-7x too fast.
#           Cross-layer transfer (L5 fit -> L3 morphologies) is licensed by
#           Moradi Chameh et al. (2021), Nat Commun 12:2497, Suppl. Fig. 5b:
#           human L2&3 and L5 I_h time constants are indistinguishable
#           (p >= 0.9999; n=6 vs 10); only the amplitude differs.
#
# DENSITY   Kalmbach et al. (2018), Neuron 100(5):1194-1208, human deep L3:
#           gIh = 1e-4 S/cm2, UNIFORM over soma + axon + dendrites (exact
#           value as re-implemented and tabulated in Rich et al. 2021 Table 3).
#           The previous 2e-4 + hay_exponential was the RAT L5 setting of Hay
#           et al. (2011). Eyal et al. (2016) Fig. 1-fig. suppl. 3 show that
#           inserting I_h at exactly that rat density into human L2/3 cells
#           drags the best-fit Cm from ~0.45 to ~0.76 uF/cm^2 -- a large
#           species-imported bias in a benchmark that measures Cm recovery.
#
# EHCN      -49.85 mV, Rich et al. (2021) Table 1 (rodent value was -45.0).
#
# To reproduce the v60ls RODENT baseline, set:
#   IH_KINETICS="Ih"; IH_GIHBAR=2e-4; IH_EHCN=-45.0; IH_DIST="hay_exponential"
IH_KINETICS="Ih_human"      # "Ih" (rodent Kole/Hay) | "Ih_human" (Rich 2021)
IH_GIHBAR=1e-4              # S/cm^2
IH_GIHBAR_CV=0.5
IH_EHCN=-49.85              # mV
IH_DIST="uniform"           # "uniform" | "hay_exponential"

NOISE_SIGMA=0.05
NOISE_BASELINE=0.05
NOISE_DRIFT=0.10
NOISE_CV=0.3
USE_IH=1
# --------------------------------------------------------------------------

cd "$CODE_DIR" \
    || { echo "[FATAL] cannot cd to CODE_DIR: $CODE_DIR" >&2; exit 1; }

# --- 0) Ensure the conda env is active (provides python, pandas) -----------
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
    echo "[fanout] activating conda env '$CONDA_ENV'..."
    source "$(conda info --base)/etc/profile.d/conda.sh" \
        || { echo "[FATAL] cannot source conda.sh" >&2; exit 1; }
    conda activate "$CONDA_ENV" \
        || { echo "[FATAL] cannot activate env '$CONDA_ENV'" >&2; exit 1; }
fi
echo "[fanout] python: $(which python)   env: ${CONDA_DEFAULT_ENV:-none}"

# --- 0b) Guard: human kinetics require the compiled mechanism (v4) ---------
# Fail here, on the login node, rather than in every one of N PBS jobs with
# "argument not a density mechanism name".
if [ "$USE_IH" = "1" ] && [ "$IH_KINETICS" = "Ih_human" ]; then
    if [ ! -f "$CODE_DIR/mod/Ih_human.mod" ]; then
        echo "[FATAL] IH_KINETICS=Ih_human but $CODE_DIR/mod/Ih_human.mod is missing." >&2
        exit 1
    fi
    if ! python -c "
import sys
sys.path.insert(0, '$CODE_DIR')
try:
    from neuron import h
except Exception:
    sys.exit(0)          # no NEURON on the login node: defer to the job
import os
os.chdir('$CODE_DIR')
try:
    s = h.Section(name='probe'); s.insert('Ih_human')
except Exception:
    sys.exit(3)
" ; then
        st=$?
        if [ $st -eq 3 ]; then
            echo "[FATAL] Ih_human is not compiled into $CODE_DIR/x86_64/." >&2
            echo "        The compile guard in submit_synth_benchmark.sh SKIPS" >&2
            echo "        nrnivmodl when x86_64/special already exists, so this" >&2
            echo "        will not fix itself. Run once, by hand:" >&2
            echo "            cd '$CODE_DIR' && rm -rf x86_64 && nrnivmodl mod" >&2
            exit 1
        fi
    fi
    echo "[fanout] Ih_human mechanism: OK"
fi

# --- 1) Build the manifest once (idempotent) --------------------------------
if [ ! -f "$MANIFEST" ]; then
    echo "[fanout] building manifest -> $MANIFEST"
    ARGS=(
        --morph-root       "$MORPH_ROOT"
        --morph-glob       "$MORPH_GLOB"
        --out              "$MANIFEST"
        --seed             "$SEED"
        --draws-per-morph  "$DRAWS_PER_MORPH"
        --cells-per-cohort "$CELLS_PER_COHORT"
        --ra-mode          "$RA_MODE"
        --e-pas            "$E_PAS"
        --F                "$F_FACTOR"
        --ih-gihbar        "$IH_GIHBAR"
        --ih-gihbar-cv     "$IH_GIHBAR_CV"
        --ih-ehcn          "$IH_EHCN"
        --ih-dist          "$IH_DIST"
        --ih-kinetics      "$IH_KINETICS"
        --noise-sigma      "$NOISE_SIGMA"
        --noise-baseline   "$NOISE_BASELINE"
        --noise-drift      "$NOISE_DRIFT"
        --noise-cv         "$NOISE_CV"
        # physiological prior constraints (v3)
        --cm-phys-lo       "$CM_PHYS_LO"
        --cm-phys-hi       "$CM_PHYS_HI"
        --tau-lo-ms        "$TAU_LO_MS"
        --tau-hi-ms        "$TAU_HI_MS"
    )
    # Ra window is opt-in: empty strings restore the v3 full-box draw.
    [ -n "${RA_PHYS_LO:-}" ] && ARGS+=(--ra-phys-lo "$RA_PHYS_LO")
    [ -n "${RA_PHYS_HI:-}" ] && ARGS+=(--ra-phys-hi "$RA_PHYS_HI")
    [ "$USE_IH" = "0" ] && ARGS+=(--no-ih)
    python synth_gt_grid.py "${ARGS[@]}" \
        || { echo "[FATAL] synth_gt_grid.py failed" >&2; exit 1; }
else
    echo "[fanout] manifest exists, reusing -> $MANIFEST"
    echo "         (delete it to redraw with new settings)"
fi

# --- 1b) Echo what the manifest actually says about I_h (v4) ---------------
python - "$MANIFEST" <<'PYEOF'
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
kin = sorted(set(df.get("ih_kinetics", pd.Series([""])).fillna("").astype(str)))
print("[fanout] manifest I_h kinetics : {}".format(kin if kin != [""] else ["Ih (legacy/blank)"]))
print("[fanout] manifest I_h dist     : {}".format(sorted(set(df["ih_dist"].fillna("").astype(str)))))
print("[fanout] manifest ehcn [mV]    : {}".format(sorted(set(df["ih_ehcn_mV"].dropna()))))
print("[fanout] manifest Ra  [Ohm cm] : {:.1f} - {:.1f}".format(df["ra_true"].min(), df["ra_true"].max()))
PYEOF

# --- 2) Read unique cohort labels via the manifest's own loader -------------
echo "[fanout] reading cohort labels..."
COHORT_LIST="$(python -c '
import sys
from synth_gt_grid import load_manifest, list_groups
print("\n".join(list_groups(load_manifest(sys.argv[1]))))
' "$MANIFEST")"
STATUS=$?

if [ $STATUS -ne 0 ] || [ -z "$COHORT_LIST" ]; then
    echo "[FATAL] could not read cohort labels from $MANIFEST" >&2
    echo "        Run the reader directly to see the traceback:" >&2
    echo "        cd '$CODE_DIR' && python -c \"from synth_gt_grid import " \
         "load_manifest, list_groups; print(list_groups(load_manifest('$MANIFEST')))\"" >&2
    exit 1
fi

mapfile -t COHORTS <<< "$COHORT_LIST"
if [ "${#COHORTS[@]}" -eq 0 ]; then
    echo "[FATAL] no cohort labels found in $MANIFEST" >&2
    exit 1
fi
echo "[fanout] cohorts found: ${COHORTS[*]}"

# --- 3) One PBS job per cohort ----------------------------------------------
N=0
for g in "${COHORTS[@]}"; do
    echo "[fanout] submitting GROUP=$g ..."
    qsub -v GROUP="$g" "$JOB_SCRIPT" \
        || { echo "[WARN] qsub failed for GROUP=$g" >&2; }
    N=$((N + 1))
done
echo "[fanout] done -- $N cohort job(s) submitted."
