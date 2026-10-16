# dev/2026-10-16_covariance_weighting -- D-048 handoff and the sandbox checks behind it

Not pipeline code. Nothing here is imported by the Ih Fit scripts, and nothing here runs in a job.
This folder holds the handoff for implementing D-048 (the per-cell Stage-7 noise-law fit weighted by
the simulated inverse covariance of its statistics, refined iteratively) and the standalone checks
whose numbers the handoff quotes. It is in the repo because project knowledge was full on 2026-10-16.

| Path | What |
|---|---|
| `TEEG_Stage7_covariance_weighting_handoff_2026-10-16.md` | the handoff: read first (its sec. 0 lists the project docs to read with it) |
| `checks/weights_1013/` | hand vs diagonal vs full weighting; where Sigma-hat can come from; Sigma vs theta; conditioning; Monte Carlo error vs kappa; missing-noise sensitivity (P1-P18, PM1-PM3) |
| `checks/boot_1013/` | bootstrap types on autocorrelated windows (iid, block, window level) |
| `checks/boot_err_1014/` | sampling vs internal error of a bootstrap SE (Efron & Narasimhan sec. 3) |
| `checks/cond_fig_1014/` | the two readings of the condition number (figure + exact checks Q1-Q3) |
| `bias_doc_2026-10-11/` | the study document "bias of a statistic" (2026-10-11) with its figures and scripts |

Each check folder has its own `README.txt`; every script states its predictions in its docstring,
and `logs/` holds the run the numbers were quoted from.

## Run (cluster or laptop)

In `checks/`, paths are resolved from each script's location, so run from anywhere;
`cond_fig_1014` reads `../weights_1013/results/weights.npz`: keep `checks/` intact.
The bias-document scripts use paths relative to their own folder: run them from
`bias_doc_2026-10-11/scripts/`, or all at once with `bash bias_doc_2026-10-11/scripts/run_all.sh`.

    conda activate prova
    cd "Passive Features/HPC script/Ih Fit/dev/2026-10-16_covariance_weighting/checks/weights_1013/scripts"
    python3 check_conditioning.py      # seconds
    python3 check_missing_noise.py     # about 30 s, needs results/weights.npz (shipped)
    python3 check_weights.py           # about 2 min, rewrites results/weights.npz

Needs numpy, scipy, matplotlib; `boot_1013` also needs statsmodels and arch (`pip install arch`).
Reported runs used Python 3, NumPy 2.5.3, SciPy 1.18.1. Re-running overwrites the shipped
`logs/`, `results/` and `figures/`; `git checkout -- .` restores them.

## Byte safety

All `.py` files are pure ASCII. The bias-document scripts were converted on 2026-10-16:
comments transliterated, non-ASCII plot labels written as `\u` escapes (labels render the same;
the parsed code is identical to the original). Both `run_all.sh` are LF-only.
