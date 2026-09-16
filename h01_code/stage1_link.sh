#!/bin/bash
# Assemble h01_code/stage1/ as SYMLINKS to the canonical Stage 1 sources in
# this repo, so the runner imports the REAL modules with no copies to drift.
#
#   bash stage1_link.sh          # create/refresh the links
#   bash stage1_link.sh --check  # verify only, exit 1 if anything is missing
#
# Run ONCE per clone, on the machine that runs the jobs (links are created
# cluster-side rather than committed, because a Windows working copy of the
# repo cannot be trusted to preserve symlinks).
#
# Why links and not copies: the attribution gate exists to check the mesh
# pipeline against the REAL spine_density; a copy that could drift would
# defeat it (see stage1/PUT_STAGE1_MODULES_HERE.txt).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
DEST="$HERE/stage1"

# module -> path relative to the repo root
declare -A SRC=(
  [morphology_exporter.py]="towards_eeg/structure/morphology_exporter.py"
  [spine_density.py]="towards_eeg/structure/spine_density.py"
  [spine_labeller.py]="towards_eeg/structure/spine_labeller.py"
  [node_classify.py]="towards_eeg/structure/node_classify.py"
  [soma_enforce.py]="towards_eeg/structure/soma_enforce.py"
  [spine_cap.py]="towards_eeg/structure/spine_cap.py"
  [alignment.py]="towards_eeg/structure/alignment.py"
  [hoc_qc.py]="towards_eeg/structure/hoc_qc.py"
  [synapse_redirect_audit.py]="towards_eeg/structure/synapse_redirect_audit.py"
  [spine_geometry.py]="Stage 1/spine_geometry.py"
  [continuation_inspect.py]="Stage 1/continuation_inspect.py"
)

mkdir -p "$DEST"
fail=0
for name in "${!SRC[@]}"; do
  src="$REPO/${SRC[$name]}"
  dst="$DEST/$name"
  if [ ! -f "$src" ]; then
    echo "MISSING SOURCE: $src"
    fail=1
    continue
  fi
  if [ "${1:-}" = "--check" ]; then
    if [ -e "$dst" ]; then echo "OK      $name -> ${SRC[$name]}"
    else echo "MISSING LINK: $dst"; fail=1; fi
  else
    ln -sfn "$src" "$dst"
    echo "linked  $name -> ${SRC[$name]}"
  fi
done

# shaft_continuation.py is imported from h01_code itself (it sits beside the
# runner); verify it is there rather than linking a second copy.
if [ ! -f "$HERE/shaft_continuation.py" ]; then
  echo "MISSING: $HERE/shaft_continuation.py"
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "stage1_link: FAILED"
  exit 1
fi
echo "stage1_link: OK ($DEST)"
