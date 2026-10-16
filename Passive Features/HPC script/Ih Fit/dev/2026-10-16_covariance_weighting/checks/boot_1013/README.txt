Bootstrap types: checks and figures behind the answer of 2026-10-13 (TEEG, Stage 7).

scripts/boot_model.py         Stage-7 noise law (TD (5b)/(6)) and the statistic T = S_L/S_0 (D-028.1). No resampling.
scripts/boot_worlds.py        bootstrap worlds from library resamplers (arch: iid, MBB, CBB, SB; SciPy: window level).
scripts/check_boot_types.py   predictions P1-P6 fixed in the docstring; E0 true law, E1 one-window worlds, E2 K = 6 cells.
scripts/fig01_two_worlds.py   Figure 1, schematic (no data).
scripts/fig02_block_length.py Figure 2, plotted from results/boot_types.npz.
scripts/run_all.sh            runs everything (about 3-4 min on 2 cores).
logs/check_boot_types.log     the run reported in the answer (all predictions passed).
logs/check_boot_types_run1_rounded_P6_refs.log  first run, P6 against the bias doc's rounded values (also passed).
results/boot_types.npz        per-window bootstrap summaries used by Figure 2.

Quick start:  cd scripts && ./run_all.sh
Smoke run (seconds):  cd scripts && python3 -c "import check_boot_types as c; cfg=dict(c.CONFIG); cfg.update(M_true=600,n_win=2,B=60,B_cell=200); e0=c.true_law(cfg); w=c.window_worlds(cfg); c.report(cfg,e0,w,c.join_predictions(cfg,w),c.cell_worlds(cfg,e0['sim']['T']))"
Software used for the reported run: Python 3, NumPy 2.5.3, SciPy 1.18.1, matplotlib 3.11.2, statsmodels 0.15.0, arch 8.0.0.
