#!/usr/bin/env bash
# Regenerates every figure and check of TEEG_bias_of_a_statistic_2026-10-11.md (about 2-3 min on a laptop).
# Run from this folder:  bash run_all.sh
# Needs numpy, scipy, matplotlib. Outputs: ../figures/*.png and logs/*.txt (compare with the numbers in the document).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs ../figures
python3 check01_bias_mc_vs_closed_form.py            # writes logs/check01.txt
python3 check02_zero_sum.py                          # writes logs/check02.txt
python3 check04_exact_mean_and_known_mean.py         # writes logs/check04.txt
python3 fig01_spaces_and_maps.py
python3 fig02_error_vs_bias.py        | tee logs/fig02.txt
python3 fig03_three_targets.py        | tee logs/fig03.txt
python3 fig04_bias_vs_precision.py    | tee logs/fig04.txt
python3 fig05_closed_form_residual_bias.py           # writes logs/check03.txt
echo "all done"
