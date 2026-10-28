#!/bin/bash
##########################################################################
# Shared shell functions of the survey jobs on davinci (Block 16.4,
# specs/SPEC.md; D-099): sourced by survey_prepare.pbs, survey_cells.pbs
# and survey_pool.pbs as
#
#   . "$DIAM_CODE/scripts/pbs/survey_common.sh"
#
# after DIAM_CODE is known. Every function prints with the caller's TAG
# ([diam-survey], ...). Pure ASCII, LF line endings (hpc-python-compat).
##########################################################################

# diam_note_reserved: the login shell exports CODE and ENV_NAME, and PBS jobs
# source .bashrc (paths reference, sections 8.2 and 8.4): a set CODE, ROOT or
# ENV_NAME is reported and IGNORED (the knobs are DIAM_*).
diam_note_reserved() {
    local _n _v
    for _n in CODE ROOT ENV_NAME; do
        eval "_v=\${$_n:-}"
        if [ -n "$_v" ]; then
            echo "NOTE: $_n is set ($_v) and is IGNORED; the knobs are DIAM_CODE, DIAM_ENV, DIAM_SURVEY, DIAM_CACHE."
        fi
    done
}

# diam_check_code TAG SCRIPT: DIAM_CODE must hold scripts/SCRIPT (exit 2).
diam_check_code() {
    if [ ! -f "$DIAM_CODE/scripts/$2" ]; then
        echo "[FATAL] scripts/$2 not found under DIAM_CODE=$DIAM_CODE" >&2
        echo "        Submit from the workstream directory:" >&2
        echo "          cd \"/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/Diameter Re-measurement\"" >&2
        echo "          qsub scripts/pbs/<job>.pbs   (or qsub -v DIAM_CODE=...)" >&2
        exit 2
    fi
}

# diam_check_config: DIAM_CONFIG_JSON, when set, must exist (exit 3).
diam_check_config() {
    if [ -n "${DIAM_CONFIG_JSON:-}" ] && [ ! -f "$DIAM_CONFIG_JSON" ]; then
        echo "[FATAL] DIAM_CONFIG_JSON set but missing: $DIAM_CONFIG_JSON" >&2
        exit 3
    fi
}

# diam_activate TAG: the conda environment. On davinci `conda activate
# spine_env` returns non-zero (a binutils activate.d hook exits 1) while
# activating correctly, and foreign deactivate.d hooks are not `set -u`
# clean (paths reference, section 9): bracket the activation, then check the
# OUTCOME, not the exit status (exit 5). Sets PY.
diam_activate() {
    local tag="$1"
    set +eu
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "$DIAM_ENV" >/dev/null 2>&1
    set -eu
    PY="$(command -v python3 || true)"
    case "$PY" in
        */envs/"$DIAM_ENV"/bin/python3) ;;
        *) echo "[FATAL] conda env '$DIAM_ENV' not active: python3 resolves to '${PY:-nothing}'" >&2; exit 5 ;;
    esac
    echo "$tag python $PY"
    "$PY" -c "import numpy, scipy, PIL, pandas; print('$tag numpy', numpy.__version__, 'scipy', scipy.__version__, 'Pillow', PIL.__version__, 'pandas', pandas.__version__)"
    echo "$tag commit $(git -C "$DIAM_CODE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
}

# diam_network TAG: the compute nodes reach the outside only through the
# proxy module (H01 reference, section 9; `module load proxy`); then one RMA
# query to api.brain-map.org proves the way is open before any work (exit 6
# otherwise: a queue wait is not spent on a blocked network). PY and
# DIAM_CODE must be set. DIAM_PROXY_MODULE empty skips the module.
diam_network() {
    local tag="$1"
    if [ -n "${DIAM_PROXY_MODULE:-}" ]; then
        if module load "$DIAM_PROXY_MODULE" 2>/dev/null; then
            echo "$tag module $DIAM_PROXY_MODULE loaded (http_proxy ${http_proxy:-<unset>})"
        else
            echo "$tag WARNING: module load $DIAM_PROXY_MODULE failed (continuing; the probe decides)"
        fi
    fi
    if ! (cd "$DIAM_CODE" && "$PY" -c "import sys; sys.path.insert(0, 'src'); import allen_image_io as aio; r = aio.api_query('model::Specimen,rma::criteria,[id\$eq529878215]', num_rows=1); print('$tag api.brain-map.org reachable: specimen', r[0]['id'])"); then
        echo "[FATAL] api.brain-map.org is not reachable from this node: the survey fetches Allen's images and" >&2
        echo "        reconstructions over the network. Is the proxy module loaded (DIAM_PROXY_MODULE=${DIAM_PROXY_MODULE:-<empty>})?" >&2
        echo "        Try on a login node: module load proxy && curl -sI http://api.brain-map.org/api/v2/data/query.json | head -1" >&2
        exit 6
    fi
}

# diam_selection_rows FILE: the number of cells in a selection CSV (its rows
# but the header; 0 when the file is missing).
diam_selection_rows() {
    if [ -f "$1" ]; then
        awk 'NR > 1 && NF > 0 { n++ } END { print n + 0 }' "$1"
    else
        echo 0
    fi
}
