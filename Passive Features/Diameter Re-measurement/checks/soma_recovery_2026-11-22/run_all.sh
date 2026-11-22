#!/bin/sh
# The recovery check's cases (D-216.2) in the brief's order -- A (c 4.5), B (fit, fixed), C, D, then A (c 3.0, 6.0) --
# each a separate single-threaded process, at most two at a time (xargs -P 2 starts the next case when one ends).
# Each case writes results/<CASE>.json and figures/<CASE>.png when it finishes, its log to logs/<CASE>.log.
#     sh run_all.sh            (about 1-2 h on two CPUs)
# One case alone:  OMP_NUM_THREADS=1 python3 run_case.py C > logs/C.log 2>&1
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf "%s\n" A_c4.5 B_fit B_fixed C D A_c3.0 A_c6.0 \
    | xargs -P 2 -I{} sh -c 'python3 run_case.py {} > logs/{}.log 2>&1; echo "{} exited $?"'
