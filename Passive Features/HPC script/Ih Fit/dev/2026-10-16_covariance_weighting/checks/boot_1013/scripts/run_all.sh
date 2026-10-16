#!/usr/bin/env bash
# Bootstrap-types checks of 2026-10-13: check first (writes ../results and the log), then the figures.
# Needs: numpy, scipy, matplotlib, statsmodels (>= 0.12), arch (pip install arch).
set -euo pipefail
cd "$(dirname "$0")"
python3 check_boot_types.py | tee ../logs/check_boot_types.log
python3 fig01_two_worlds.py
python3 fig02_block_length.py
