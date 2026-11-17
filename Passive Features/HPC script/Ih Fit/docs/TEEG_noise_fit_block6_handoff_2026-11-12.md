# TEEG -- noise fit (G4): Block 6 handoff (2026-11-12)

For the chat that writes Block 6's code (the noise-fit run: driver, PBS array, merge) and then runs it on davinci with
the user. Supersedes, as the place to start, `TEEG_noise_fit_implementation_handoff_2026-11-11.md` (this folder),
which stays the history of Block 9 and of D-122 to D-125.

| Date | Change |
|---|---|
| 2026-11-12 | Created at the user's request ("The handoff then"), after D-125 (Block 6's five questions answered) and D-126 (its two readings confirmed). Same day, before it: Block 9 written and pushed (`2171519`); the noise chat's D-117/D-118 renumbered D-122/D-123 (`a05ccdb`); D-124 (the cohort); the project instructions became v2 (the repository is the source of truth); the diameter chat's D-117 to D-121 and I-005 merged into `main`'s log (`3dc3703`); Block 6's SPEC entry drafted (`3dc3703`). |
| 2026-11-13 | Step 1 done: Block 6 written, smoke-tested (R1-R9 9/9; 26 planted defects, each caught) and pushed (`1d848f6`). §1, §3 (what was built beyond the SPEC entry), §4 (the final commands, what to read back), §5 and §9 brought up to date. No decision taken (next free IDs unchanged: D-127, I-006). |
| 2026-11-14 | Step 2 under way **[cluster, pasted 16:16]**: the F2 scan of the 18 cells finds nothing to drop; the FULL gates SETTLED at `62f3324` (16 checks); `smoke_noise_run.py` 9/9 on davinci (536 s); the first dry run refused inside an interactive job (the guard). §1, §4, §5, §9 updated. |
| 2026-11-14 (later) | The pilot submitted, 16:19 **[cluster]**: array `2096989[].pbsserver01`, merge `2096990.pbsserver01`, walltime 48 h; §4 holds the job IDs, the run file and how to follow it. |
| 2026-11-16 | The pilot done: 2 ok (4.0 h and 7.6 h); the 18-cell run's WALLTIME proposed at 24:00:00; §4 holds the timings and the launch commands. |
| 2026-11-16 (later) | WALLTIME for the 18 cells: 48:00:00, the user's decision (D-147); §4 corrected. |
| 2026-11-17 | Step 2 done: the 18-cell run, 18 ok (§4 holds the per-cell table); step 3 (Block 5) next. |
| 2026-11-17 (later) | The run's two tables read back **[run, here, on the CSVs the user attached]** (§4, "The two tables read back"; §6 item 8): every integrity check passes; the law is rejected in all 18 cells (J/dof 2.7-160); D-055's S3.1 first half holds, S3.2 first half fails; 10 LS sweeps failed; the LS slow SD exceeds the SS one in 16 of 18 cells; a few LS sweeps carry degenerate fast parts into their twins. Next: the per-cell JSONs, for the residuals. No decision taken. |
| 2026-11-17 (later, 2) | The 18 cell JSONs read back **[run, here, on the files the user attached]** (§4, "The cell JSONs read back"; §6 item 8 rewritten as two decisions for the user): the misfit sits at the 1 ms autocorrelation (the law above the data in 16 of 18 cells, SS), not at 0.1 ms; D-055's S3.1 second half and S3.3 fail; the per-sweep laws misfit too; the real statistics scatter 1.3 x what the law simulates; the failed sweeps sit at rho_f's lower bound; the `no_plateau` cells wander in tau_s. No decision taken. |
| 2026-11-17 (later, 3) | A three-component law tested at the user's request (12:28: a faster component, the two slow-side parts split at 1 ms; 12:30: "Go on with the test") **[run, here]**: `dev/2026-11-17_three_component_law/` (README). Fitted by simulation on the 18 cells, the middle component lowers J in every cell (median 121 to 38) and removes the 1 ms excess; 13 of 18 cells stay rejected. §4 "A third component, tested"; §6 item 8 updated. No decision taken. |

**Basis tags.** **[run]** executed in the assistant's sandbox (date given); **[src]** read from the code, SPEC or log
on `main`; **[log]** the decision log's text; **[user]** the user's decision or words; **[reasoning]** the assistant's
argument, not checked; **[open]** not decided or not verified.

## Start here: the order of work

| # | Step | Blocked by | Where it runs |
|---|---|---|---|
| 1 | **Block 6's code**: `run_noise_fit.py`, `submit_noise_fit.sh`, `smoke_noise_run.py` (R1-R8), exactly as SPEC Block 6 states -- **done 2026-11-13, `1d848f6`** (R1-R9) | nothing: D-123 to D-126 settle every point | sandbox |
| 2 | **The run on davinci** (the user runs it, §4): `git pull`, the FULL gates, a dry run, the 2-cell pilot at full size, then the 18 cells with the pilot's walltime -- **next** (§4's commands are final) [corrected 2026-11-17: **done**, 18 ok, 0 refused, 0 failed, 0 missing] | step 1 | davinci |
| 3 | **Block 5's SPEC entry, then its code**: the twins built from the run's tables (§6's questions first) -- **next** (2026-11-17): the two tables inspected first [2026-11-17 (later): tables read back (§4): sound, but the law misfits every cell; next the per-cell JSONs, to see where the misfit sits (§6 item 8)] [2026-11-17 (later, 2): JSONs read back (§4); two decisions wait for the user before the SPEC entry: the law kept or extended, and a rule for degenerate per-sweep fits (§6 item 8)] | the run's tables | sandbox |
| 4 | **Block 7** (the record), then **Stage 7** (Block 8, the recovery gate, D-083) | Blocks 5-6 | sandbox, davinci |

Open with the user, unanswered: whether the S0-S3 checks get their own roadmap stage (proposal: N6b after N6, under G4).
Roadmap rule 4: propose, add only on a yes.

## 1. Where it stands (roadmap at `9526401`, 2026-11-12) [run]

Implementation planned 6, pushed 5, tested 7, done 8; theory open 8, studying 4, closed 5. [corrected 2026-11-13: planned 5, pushed 6, tested 7, done 8 at `1d848f6`, `roadmap.py status`]

| Stage | Status | Note |
|---|---|---|
| N1-N4 Blocks 1-4 | pushed | no tester report covers them (last 2026-10-07, `7a49a28`) |
| NT tools (gates) | done | FULL run on davinci only 2026-10-20 at `ca028ea` (13 checks); now 14 (G2 and G6d cover Block 9): re-run before the pilot [corrected 2026-11-14: 16 checks (G0-G12 with G6b, G6c, G6d); FULL run at `62f3324` SETTLED, 16 passed **[cluster]**] |
| N4b Block 9 | pushed | `noise_sweep_fit.py` at `2171519`, smoke P1-P11 11/11 |
| N6 Block 6 | planned [corrected 2026-11-13: pushed (roadmap.py status: no tester report covers 1d848f6)] | SPEC entry drafted 2026-11-12; code not written [corrected 2026-11-13: written, smoke-tested, pushed at `1d848f6`] |
| N5, N7, S7 | planned | after N6 (D-123) |

## 2. Decisions that bind Block 6

| ID | The decision |
|---|---|
| D-123 | the noise fit of real cells (Block 6 on davinci) before the twins (Block 5), built from its fits; then Stage 7 **[log]** |
| D-124 | the cohort: all 18 L3_exc cells, after a pilot of the same driver on 508282493 and 569844159 **[user]** |
| D-125 | (a) one PBS array job per cell (Blocks 3, 4, 9 in sequence) then a merge job; at most 120 cores at once; per cell a JSON, per group `noise_cells_<GROUP>.csv` and `noise_sweeps_<GROUP>.csv` under `$IH_OUTPUT_BASE/noise_fit/<RUN_TAG>/<GROUP>/`; S0-S3 dropped for now (D-055's superseded elements listed in D-125.3); a refused or failed cell recorded with its reason, the run goes on; the pilot at full size, its two cells kept **[user]** |
| D-126 | D-125's readings confirmed: 6 cores per cell, Block 9's sweeps in a pool of 6, BLAS threads 1; D-055 up to date as D-125.3 lists **[user]** |
| D-114 | a cell whose sweep tilt cannot be read is refused (`CellLayout.from_cell` raises `ValueError`) **[log]** |
| D-122 | Block 9's defaults: B_v = 50, M_refit = 1000; a failed sweep's twin at theta_hat(c) **[log]** |
| D-083 | the six-parameter recovery on synthetic twins before any six-parameter fit of a real cell **[log]** |

The log: `docs/project/TEEG_decisions_and_ideas_log.md` on `main`, about 1.06 MB **[run]**: search it, never read it whole;
read its last changelog rows before taking an ID. **Next free IDs: D-127, I-006.** Every chat now logs in this copy
(the project instructions' v2).

## 3. Step 1 -- Block 6's code [done 2026-11-13, `1d848f6`]

Built as SPEC Block 6 states; the as-built notes of its entry list every addition, the assistant's choices, flagged, none decided by the user **[src]**:

| Addition | Why |
|---|---|
| the resume rule also needs the same `sizes` record and clean code (`code_dirty`): a JSON written from code that differed from its commit is re-run, and nothing is skipped while `git status` shows a noise-fit file modified | a `--sizes smoke` or hand-edited result must never stand in for a full-size committed one |
| the submitter writes a run file `~/noise_fit_runs/<RUN_TAG>_<GROUP>_<UTC>.env` (every resolved path, the cohort) and passes the jobs only its path | the archive and code paths hold spaces, which `qsub -v` does not carry; a job runs exactly the cohort submitted |
| `DRY_RUN=1` runs `list` and checks both argument lists against the driver's own parser (read-only Python) | the SPEC's "runs no Python" would need a second, bash copy of the cohort rule; read as "no fit, no merge, no qsub" |
| knobs `SIZES` (full / smoke), `NOISE_RUN_DIR`, `ROADMAP_DISABLE` (passed on to the jobs) | a 10-minute check of the PBS path at smoke sizes is possible; the run file's place; a submission can stay off the roadmap |
| a one-cell cohort is a plain job with `NOISE_INDEX=0` | PBS refuses an array of one subjob **[reasoning: not checked on davinci]** |
| the JSON also holds `code_dirty`, `job_id`, `started_utc`, `finished_utc`, `versions`, `seconds.total`, `tables` (Block 9's `cell_columns()` and `table_rows()`, read by the merge) and more of Block 3 (`start_groups`, the tilts) | provenance; the merge needs no result object rebuilt from its dict |
| progress lines `[noise_fit <id>] ...` (Block 4's solves, each sweep's fit and bootstrap) | a job killed by the walltime leaves no JSON; its log still shows how far it got and how long each step took |
| the LS sweeps submitted to the pool first, rows kept in `fit_sweeps`' order | the LS sweeps are the slow ones (SPEC 2.5 (h)); the numbers do not change (R1) |

The original plan of this step follows, kept as written.


Write it from SPEC Block 6 (`Ih Fit/specs/SPEC.md`, section 5, before "Tools"), not from this summary. What it calls,
all on `main` **[src]**:

| From | Names |
|---|---|
| Block 3 (`noise_windows`) | `load_ss_windows(spec_dir)`, `ss_noise_windows(ss)`, `load_ls_prewindows(spec_dir)` |
| Block 4 (`noise_fit`) | `CellLayout.from_cell(nwin, ls)` (a `ValueError` is a refusal, D-114), `CellTarget.from_cell(nwin, ls, layout)`, `fit_cell(target, layout, cfg)`, `NoiseFitConfig`, `CellFitResult.to_dict()` |
| Block 9 (`noise_sweep_fit`) | `run_sweep(cell_fit, target, layout, cfg, stage_cfg, protocol, sweep)` per sweep (SS sweeps `layout.sweep_numbers`, then LS `layout.ls_sweep_number`), then `pool_cell(rows, cell_fit.law, specimen_id, stage_cfg=...)`; equal to `fit_sweeps` (P9); `SweepStageResult.to_dict()`, `table_rows()`, `cell_columns()` |

| Item | Value | Basis |
|---|---|---|
| `--sizes smoke` preset | `NoiseFitConfig(M_binding=60, R_iter=40, R_final=60, B_boot=400, k_max=2, max_nfev=6)` (Block 4's F7) and `SweepStageConfig(B_v=4, M_refit=60)` (Block 9's P9) | [src] |
| a writer-like archive cell | `smoke_noise_fit.build_cell(dir, rng)` writes one with `smoke_noise_windows.write_fixture` and reads it back (2 SS sweeps, K = 63; 5 LS sweeps) | [src] |
| R2's refused cell | the same with `ls_tail_n=smoke_noise_windows.LS_TAIL_N` (0.5 s tail): every LS tilt unreadable, `from_cell` raises with 'tail_short' | [src: Block 4's F11] |
| the job script's conventions | `submit_ih_fit.sh` (env knobs `IH_CODE_DIR`, `IH_ARCHIVE_ROOT`, `IH_OUTPUT_BASE`, `CONDA_ENV`, `DRY_RUN=1`, `#PBS -q cpu`) and an existing PBS array script of the repository for `-J` and `PBS_ARRAY_INDEX` | [open: not read in the 2026-11-12 chat] |
| roadmap marker lines | `subproject-roadmap` skill, `references/hpc_marker.md`; job `noise_fit` (already in N6's `jobs`) | [src: roadmap yaml] |
| cost, for the walltime | Block 9 about 6.6 h on one core for a 2-SS + 5-LS cell at B_v = 50, M_refit = 1000 (sandbox, fixture); with 6 workers about the slowest sweep's share (an LS sweep: about 1.1 h); Block 4 at full M and R never timed | [run 2026-10-30, SPEC 2.5 (h); reasoning; open] |

Protocol: the `scientific-coding` block protocol (check 1 against the SPEC, check 2 by running R1-R8, planted defects),
`hpc-python-compat` (ASCII, LF), SPEC status `smoke-tested <date>`, a `CHANGELOG_ih_fit.md` row, commit `block 6: ...`,
push; then `roadmap.py status` and `render` (N6 should turn pushed). Proposal, not decided [reasoning]: a gates line
for `smoke_noise_run.py` (G2 for its commit, a G6e in FULL mode).

## 4. Step 2 -- the run on davinci (the user runs it)

The shape of the commands; finalise them with the code **[open]** [corrected 2026-11-13: final, below this block]:

    cd /davinci-1/home/ldellamea/TEEG/Towards-EEG && git pull
    cd "Passive Features/HPC script/Ih Fit" && conda activate prova && bash cluster_gates_noise_fit.sh   # FULL, about 25 min; VERDICT SETTLED needed
    RUN_TAG=<tag> CELLS=508282493,569844159 DRY_RUN=1 bash submit_noise_fit.sh   # prints every command, submits nothing
    RUN_TAG=<tag> CELLS=508282493,569844159 bash submit_noise_fit.sh             # the pilot, full size
    RUN_TAG=<tag> WALLTIME=<from the pilot> bash submit_noise_fit.sh            # the 18 cells; the pilot's two skipped (same commit)

The final commands (2026-11-13, `1d848f6`) **[src; run in the sandbox against stubs of conda, qsub and python only]**:

    cd /davinci-1/home/ldellamea/TEEG/Towards-EEG && git pull
    cd "Passive Features/HPC script/Ih Fit" && conda activate prova
    bash cluster_gates_noise_fit.sh                  # FULL, about 25 min; needs "VERDICT: SETTLED"
    python smoke_noise_run.py                        # about 9 min; needs "SMOKE noise_run: 9/9 passed" (Block 6 is not in the gates)
    RUN_TAG=nf1 CELLS=508282493,569844159 DRY_RUN=1 bash submit_noise_fit.sh
    RUN_TAG=nf1 CELLS=508282493,569844159 WALLTIME=48:00:00 bash submit_noise_fit.sh   # the pilot
    # after the pilot has ended:
    RUN_TAG=nf1 WALLTIME=<from the pilot> DRY_RUN=1 bash submit_noise_fit.sh
    RUN_TAG=nf1 WALLTIME=<from the pilot> bash submit_noise_fit.sh

The dry run must print `cohort: 2 cell(s): 508282493 569844159`, `cores: 2 cell(s) x NCPUS 6 = 12 (cap 120, D-125.1)`, `CLI contract: ... accepts the cell and the merge arguments`, one `subjob` line per cell and the merge's line, and end with `DRY RUN: nothing written, nothing submitted`. The pilot's walltime is 48 h, not the default 12 h **[reasoning]**: Block 4 alone is estimated at 1-3.5 h (2-sweep) and 2-5.5 h (10-sweep) per cell on one core (SPEC 2.3 (h), from evaluations timed in the sandbox; never run whole at full size), Block 9 adds about the slowest LS sweep's 1.1 h or more with 6 workers, and 569844159 is a 10-sweep cell; a longer request costs nothing if the job ends early. Then WALLTIME for the 18 cells: the pilot's largest `seconds.total` times about 1.5, rounded up **[reasoning]** (571483807 and 571624532 are 10-sweep cells too).

Reading the pilot back (OUT = `/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs_ih/noise_fit/nf1/L3_exc`):

    OUT="/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs_ih/noise_fit/nf1/L3_exc"
    grep -h '^NOISE_CELL' "$OUT"/logs/cell_*.log; grep -h -A20 '^NOISE_MERGE' "$OUT"/logs/merge_*.log
    python -c "import json,glob,sys; [print(f.split('/')[-1], json.load(open(f))['status'], json.load(open(f))['seconds']) for f in sorted(glob.glob(sys.argv[1] + '/cells/*.json'))]" "$OUT"

Done 2026-11-14 **[cluster, pasted 16:16]**: the F2 scan (0 sweeps to drop in the 18 cells), `git pull` to `62f3324`, the FULL gates (16 passed, VERDICT SETTLED, G11 1220 s), `smoke_noise_run.py` 9/9 (536 s) -- all inside an interactive job (`qsub -I`, dvnode018). The dry run there was refused by the submitter's guard (`PBS_JOBID` set); it and the pilot run from the **login node** (dvlogin02): `exit` the interactive job first.

**The pilot is running** (submitted 2026-11-14 16:19 from dvlogin04 at `52aa4c1` **[cluster]**): cell array `2096989[].pbsserver01` (subjob 0 = 508282493, subjob 1 = 569844159; 6 cores, 48 h each), merge `2096990.pbsserver01` after it; run file `~/noise_fit_runs/nf1_L3_exc_20261114T151931Z.env`. Following it:

    qstat -t '2096989[]' 2096990           # the two subjobs and the merge (R running, Q queued, F finished)
    OUT="/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs_ih/noise_fit/nf1/L3_exc"
    tail -n 4 "$OUT"/logs/cell_*.log     # progress: the windows line first, then each solve and sweep

Expected durations **[reasoning, from SPEC 2.3 (h) and 2.5 (h); never run at full size]**: 508282493 (2 SS sweeps) about 2-5 h, 569844159 (10 SS sweeps) about 4-10 h, the merge minutes after the later one.

**The pilot is done** (read back 2026-11-16 **[cluster]**): `NOISE_MERGE L3_exc: 2 ok, 0 refused, 0 failed, 0 missing`.

| Cell | SS sweeps | windows (s) | cell fit (s) | sweep stage (s) | total |
|---|---|---|---|---|---|
| 508282493 | 2 | 0.7 | 3371 | 11165 | 14536 s = 4.0 h |
| 569844159 | 10 | 1.7 | 6202 | 21104 | 27308 s = 7.6 h |

The sweep stage is 77 % of both. WALLTIME for the 18 cells **[reasoning, the assistant's proposal]** [corrected 2026-11-16: **48:00:00**, the user's decision, D-147; the commands below carry it]: 24:00:00, 3.2 x the slower pilot cell, rather than 1.5 x (11.4 h): the stage grows with a cell's sweeps, mostly the LS ones, not known for the other 16 cells, and a killed cell loses all its work. The code is unchanged since the pilot (`1d848f6`), so its two cells are skipped:

    cd "/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/HPC script/Ih Fit" && git pull && conda activate prova
    RUN_TAG=nf1 WALLTIME=48:00:00 DRY_RUN=1 bash submit_noise_fit.sh
    RUN_TAG=nf1 WALLTIME=48:00:00 bash submit_noise_fit.sh

**The 18-cell run is done** (read back 2026-11-17 **[cluster]**): `NOISE_MERGE L3_exc: 18 ok, 0 refused, 0 failed, 0 missing` (newest merge log); the pilot's two cells kept. Outputs: `$OUT/cells/specimen_<id>.json`, `$OUT/noise_cells_L3_exc.csv`, `$OUT/noise_sweeps_L3_exc.csv`.

| Cell | status | total (s) | K | K_LS |
|---|---|---|---|---|
| 508282493 | ok | 14536 | 40 | 22 |
| 526785799 | ok | 16034 | 47 | 28 |
| 529878215 | ok | 16620 | 39 | 38 |
| 531520401 | ok | 17370 | 40 | 25 |
| 531523484 | ok | 20981 | 37 | 33 |
| 531526539 | ok | 16981 | 46 | 25 |
| 537297470 | ok | 20936 | 66 | 35 |
| 537313600 | ok | 22675 | 47 | 38 |
| 539653924 | ok | 21699 | 47 | 27 |
| 541545716 | ok | 21139 | 56 | 27 |
| 548408551 | ok | 19139 | 52 | 29 |
| 548425467 | ok | 18907 | 69 | 29 |
| 548473051 | ok | 20991 | 56 | 33 |
| 569844159 | ok | 27308 | 348 | 38 |
| 571430815 | ok | 12971 | 199 | 16 |
| 571483807 | ok | 18051 | 303 | 28 |
| 571624532 | ok | 22121 | 317 | 28 |
| 571691416 | ok | 15682 | 58 | 26 |

Per cell 3.6-7.6 h (median 5.3 h), about 574 core-hours in all. For later walltimes **[run, here, on these numbers]**: t ~ 6061 + 10.7 K + 410 K_LS s (residual SD 2368 s; r = 0.69 with K_LS, 0.34 with K).

**The two tables read back** (2026-11-17; `noise_cells_L3_exc.csv` and `noise_sweeps_L3_exc.csv` as the user attached
them from `$OUT`; everything below **[run, here]** unless tagged; the scripts are in the assistant's sandbox, not
committed: `read_nf1_tables.py`, `check_sweep_precision.py`, `compare_spread.py`).

Integrity:

| Check | Result |
|---|---|
| status, commit, dof | 18 ok; `1d848f6` on every row; dof 11 everywhere (d_eff = 15: no statistic dropped) |
| sweep rows | 589 = 64 SS + 525 LS; LS rows = K_LS in every cell; SS sweeps 2 (14 cells), 6 (571430815), 10 (569844159, 571483807, 571624532), as the data contract says |
| pooling recomputed from the sweep rows ((P3), (P4), (D-076.1), written out independently) | tau2 to 5.8e-16, mu to 2.2e-15, the shrunk sigma_s and lambda to 2.0e-14 |
| failed sweeps | the twin's law equals theta_hat(c) exactly (D-122.3); raw sigma_s, v and lambda empty |

The cell laws theta_hat(c):

| Quantity | min | median | max | Where |
|---|---|---|---|---|
| sigma_f (mV) | 0.0346 | 0.0379 | 0.0404 | nearly the same in every cell |
| rho_f (per 20 us sample) | 0.105 | 0.149 | 0.249 | max 569844159 |
| sigma_s (mV) | 0.0417 | 0.0578 | 0.328 | 571430815 0.328, 571624532 0.298 |
| tau_s (ms) | 6.7 | 54.9 | 1863 | 571624532 1863; 571483807 6.7, 569844159 7.7 |
| slow share of the variance, sigma_s^2 / (sigma_f^2 + sigma_s^2) | 0.56 | 0.71 | 0.99 | |
| J (11 dof) | 29.9 | 119 | 1756 | J/dof 2.7-160; max 569844159 |
| p (chi^2_11 tail) | 0 (underflow, 569844159) | | 0.0016 (571430815) | p < 0.01 in 18 of 18 |
| kappa | 174 | 682 | 4800 | max 571430815 |
| t_SS (ms) | 80 | 98.9 | 140.8 | 80 ms in 529878215, 539653924, 571430815 |

- No coordinate within Block 4's at-bound distance (1e-3 of the range); the closest are rho_f of 571691416 (0.105 of
  its range from the lower bound) and log tau_s of 571624532 (0.146 from the upper).
- `no_plateau` (k_max = 6 reached) in 5 cells: 531520401, 531523484, 531526539, 539653924, 571483807. n_solves: 4
  (5 cells), 5 (2), 6 (3), 7 (3), 8 (5).
- J does not follow the amount of data (Spearman with K 0.10, with K_LS 0.33) but rises with rho_f (0.67, p 0.002).
- Under the law, J would be about chi^2_11, inflated by the Monte Carlo of q_sim by about 1 / ceil(M / K): 1.2 % at
  K = 37, 11 % at K = 348 **[reasoning; D-055's S0 would have checked it, not run; PubMed, two queries on the
  simulated-moments J test, nothing; bioRxiv has no keyword search]**. Even 12 % leaves the smallest J at p = 0.005:
  the law misfits every cell.

D-055's S3 predictions read on this run (fixed 2026-10-18; the checks dropped for now, D-125.3; a reading, not a
scheduled check):

| Prediction | Reading |
|---|---|
| S3.1 (a) alarm (p < 0.01) on >= 12 of 18 cells | **holds**: 18 of 18 |
| S3.1 (b) the largest residuals at the SS 0.1 ms autocorrelation | **open**: needs q_hat, q_sim and Sigma from the cells' JSONs |
| S3.2 (a) tau_s in [15, 100] ms for >= 15 cells | **fails**: 10 of 18; above 100 ms 526785799 (198.5), 539653924 (103.7), 541545716 (234.9), 548408551 (116.4), 571430815 (198.1), 571624532 (1863); below 15 ms 569844159 (7.7), 571483807 (6.7) |
| S3.2 (b) 526785799 and 571430815 among the three largest sigma_b | void (no sigma_b, D-102, D-107, D-125.3); the two have the third- and fourth-longest tau_s (198.5 and 198.1 ms) and sigma_s 0.131 and 0.328 mV |
| S3.3 J_SS > J_LS in >= 12 cells | **open**: needs the JSONs |

The per-sweep stage (Block 9):

| Finding | Numbers |
|---|---|
| failed sweeps | 10 of 525 LS, 0 of 64 SS: 8 `at_bound:rho_f` (508282493 LS 64; 529878215 LS 59, 64; 531523484 LS 42, 65; 541545716 LS 44; 571624532 LS 39, 52), 1 `at_bound:log_sigma_f,rho_f` (537313600 LS 85), 1 `no_convergence` (548473051 LS 53). Which bound: in the JSONs |
| `v_refits_failed` (> 10 % of the 50 refits) | 39 LS sweeps, 6-29 of 50: 571624532 11, 541545716 11, 526785799 5, 548425467 3, 548473051 3, 539653924 2, four cells 1 (529878215, 531523484, 537297470, 571430815) |
| the slow SD by protocol | exp(mu_LS) / exp(mu_SS) median 1.154 (IQR 1.08-1.37), > 1 in 16 of 18 cells (Wilcoxon signed-rank p 3.3e-4); not related to t_SS (Spearman 0.08), weakly to tau_s (0.40, p 0.1) |
| sweeps differ in log sigma_s beyond their v | Cochran's Q p < 0.05 in 12 of 18 cells (LS), 5 of 18 (SS); tau2 = 0 in 4 cells (LS), 9 (SS); sqrt(tau2_LS) median 0.14, max 0.40 (526785799) |
| the fast part per sweep (not pooled, D-076 (i)) | sigma_f(c, s) / sigma_f(c), 1-99 %: SS 0.88-1.10, LS 0.66-2.07; rho_f(c, s) - rho_f(c): SS -0.05 to 0.06, LS -0.13 to 0.14 |
| LS fast parts far from the cell's | of 515 fitted LS sweeps: 13 with sigma_f > 1.5 x (each with its own sigma_s 1.7-5.1 x the cell's: noisy sweeps); 3 collapsed below 0.5 x (537297470 LS 67, 0.0018 mV; 541545716 LS 36, 0.0061; 548425467 LS 63, 0.0030); 3 with rho_f > 0.4 (531520401 LS 50, 0.76, and LS 52, 0.63; 571624532 LS 50, 0.97) |
| a twin from a collapsed fit | 537297470 LS 67: its own sigma_s 0.299 mV with v = 0.65, so lambda = 0.005 and the twin's sigma_s 0.051 mV: the twin's total SD 0.051 mV against the sweep's own fit 0.299 mV, with no fast part |

Precision of one per-sweep fit (delta method; the same construction as SPEC 2.5 (h)): V = (G^T W G)^-1, W the inverse
covariance of one replicate sweep's statistics at theta_hat(c) (R = 1000 LS, 600 SS), G the central-difference
Jacobian (step 0.02) of q_sim(. | s) at M = 1000, on Block 4's smoke fixture (a 265 ms LS pre-window at 50 kHz; an SS
sweep of 30 kept windows from t_SS = 80 ms), tilt 0, 18 cells, 338 s on 2 cores.

| Quantity, LS | Result |
|---|---|
| sd(log sigma_s) against Block 9's bootstrap SE (median per cell) | ratio median 1.00, range 0.84-1.14: the linearisation reproduces the bootstrap |
| sd(log sigma_f), sd(rho_f) | median 0.043 (0.011-0.142), 0.012 (0.011-0.015) |
| observed robust (MAD) between-sweep SD / that sd | log sigma_f median 1.64 (0.75-11.7; 571483807 11.7, 569844159 3.8); rho_f 1.50 (0.93-8.1; 531520401 8.1) |
| LS sweeps beyond 4 such SDs from their cell's LS median | 120 of 515 (sigma_f 81, rho_f 59) |

Not checked on the tables (needs the 18 JSONs, `$OUT/cells/`): where J sits by statistic and by protocol; the failed sweeps'
bound side; the per-sweep J_s of the outlying sweeps; the 5 `no_plateau` histories. [done 2026-11-17 (later, 2): below]

**The cell JSONs read back** (2026-11-17; `$OUT/cells/specimen_<id>.json`, the 18 files as the user attached them;
everything **[run, here]** unless tagged; scripts in the assistant's sandbox, not committed: `read_nf1_cells.py`,
`read_nf1_sweeps_json.py`, `se_ratio.py`). With r = q_sim(theta_hat(c)) - q_hat(c) (15 statistics) and Sigma the
final simulated covariance: J recomputed as r^T Sigma^-1 r equals the JSON's to 8e-14 (relative) in every cell.
z_m = r_m / sqrt(Sigma_mm) (the marginal standardised residual); c_m = (Sigma^-1 r)_m^2 / (Sigma^-1)_mm (the drop of
J when statistic m is removed at fixed theta_hat(c)); "rel" = r_m / q_hat_m.

Where the misfit sits, over the 18 cells (law minus data):

| Statistic | law above data (cells) | median rel | median abs rel | largest c_m in (cells) | c_m > 9 in (cells) |
|---|---|---|---|---|---|
| SS SD | 13 | +2.9 % | 5.4 % | 2 | 6 |
| SS rho(0.02 ms) | 12 | +3.5 % | 4.6 % | 0 | 3 |
| SS rho(0.1 ms) | 11 | +4.9 % | 7.3 % | 0 | 9 |
| **SS rho(1 ms)** | **16** | **+12.2 %** | 14.4 % | **9** | **15** |
| SS rho(10 ms) | 6 | -8.7 % | 19.1 % | 0 | 2 |
| LS pre-window SD | 12 | +2.0 % | 4.4 % | 3 | 5 |
| LS rho(0.02 ms) | 9 | +0.2 % | 2.4 % | 0 | 3 |
| LS rho(0.1 ms) | 7 | -0.5 % | 3.0 % | 0 | 8 |
| LS rho(1 ms) | 13 | +1.5 % | 4.7 % | 2 | 7 |
| LS first 50 ms SD | 16 | +4.6 % | 4.6 % | 0 | 4 |
| LS last 132.5 ms SD | 13 | +4.0 % | 4.6 % | 0 | 5 |
| LS last rho(0.02 ms) | 13 | +4.6 % | 5.3 % | 0 | 1 |
| LS last rho(0.1 ms) | 11 | +5.4 % | 6.9 % | 0 | 5 |
| LS last rho(1 ms) | 14 | +10.7 % | 10.7 % | 2 | 8 |
| LS last rho(10 ms) | 9 | +0.9 % | 23.2 % | 0 | 8 |

- Medians over cells of the SS autocorrelation, data against law: 0.559 / 0.563 (0.02 ms), 0.456 / 0.482 (0.1 ms),
  0.366 / 0.427 (1 ms), 0.165 / 0.144 (10 ms). The real noise loses correlation gradually from 0.02 to 10 ms; the
  law (one fast AR(1) part gone within a few samples, one slow exponential) is flatter between 0.1 and 1 ms.
  **[reasoning, Likely]** the noise holds a component with a correlation time of the order of 1 ms that the law
  lacks, or a range of time scales.
- The SDs are matched to a median 4-5 % (largest 22 %, LS last, 526785799).
- By protocol, at the joint theta_hat(c): J_SS (5 statistics alone) median 36.8 (13.3-841), above chi^2_5's 99 %
  point in 17 cells; J_LS (10 statistics) median 56.9 (17.3-869), above chi^2_10's in 16. In several cells one
  protocol is fitted and the other is not: 541545716 and 571624532 (SS z +3 to +8, LS within 1.4), 571483807 (the
  four LS pre-window statistics z -6 to -8), 529878215 (SS z -2 to -6, LS last +1 to +5).
- D-055 read on it: **S3.1 (b) fails as written** (the largest residual at SS rho(0.1 ms) in no cell; at SS rho(1 ms)
  in 9 by c_m, 7 by |z_m|); **S3.3 fails** (J_SS > J_LS in 6 of 18; per statistic, J_SS / 5 > J_LS / 10 in 14).
- The real statistics scatter more than the law simulates: the data's window-bootstrap SE of each median (W_0's,
  D-049) over sqrt(Sigma_mm) has median 1.32 (SS 1.20; LS 1.42; the four LS pre-window statistics 1.5-1.6). The sum
  of z^2 over the 15 statistics has median 41.8 with the law's SDs and 32.4 with the data's SEs, against J's 119:
  the marginal residuals alone give about a third of J (medians); the rest comes through the correlations in Sigma
  (kappa 174-4800). With the data's own SEs the sum still exceeds chi^2_11's 99 % point (24.7) in 10 of 18 cells
  (a rough reading: it ignores the correlations).

The per-sweep laws' own fits (J_s of (P1), dof_s = 2 for SS, 7 for LS):

| | SS (64 fitted) | LS (515 fitted) |
|---|---|---|
| J_s / dof_s: median (q25-q75), max | 9.1 (3.5-18.8), 126 | 2.9 (1.7-5.8), 1107 |
| p < 0.01 | 44 (69 %) | 289 (56 %) |
| failed refits of the bootstrap | 0 of 3200 | 725 of 25750 (2.8 %) |

- So a law per sweep (tau_s held at the cell's) removes the protocol mismatch and still misfits most sweeps: the
  shape misfit is inside each protocol. Only 571430815 is near the law (LS median J_s / dof_s 0.96, SS 1.1).
- The 10 failed sweeps: 9 end at rho_f's LOWER bound (0.001; 537313600 LS 85 also at sigma_f's lower bound) with a
  slow SD 1.4-4.5 x the cell's and J_s 27-935: sweeps with far more slow noise than their cell, for which the fit
  turns the fast part white. 548473051 LS 53 ran out of evaluations at rho_f = 0.991.
- The outlying fitted sweeps of the tables' read-back carry it in J_s: 537297470 LS 67 J_s 7751, 548425467 LS 63
  1444, 541545716 LS 36 60, 531520401 LS 50 and 52 71 and 52, 571624532 LS 50 77; the 13 with sigma_f > 1.5 x the
  cell's have J_s / dof_s 4.7-80.

The 5 `no_plateau` cells, tau_s per Sigma-weighted solve (ms): 531526539 alternates 8.1, 21.3, 9.6, 21.4, 8.1,
21.6, final 16.6 (the final step itself moved log tau_s by 0.26: not settled); 531520401 55, 34, 37, 27, 35, 34,
final 34; 531523484 33, 20, 17, 23, 19, 20, final 20; 539653924 left a 2 s basin at the third solve and then
settled at 104 (last steps 0.014 and 0.006 in u); 571483807 6.1-7.2, final 6.7 (last steps 0.02-0.03 against
tol_u = 0.01). The moving coordinate is log tau_s in every case; J moves between solves with the R = 600 draw of
Sigma (152 to 201 in 571483807). **[reasoning]** a misspecified law makes the optimum depend on the weight, so the
iteration's wander is a second sign of the misfit, and tau_s(c) is the least determined of the four parameters
(Block 9 holds it fixed for every sweep).

**A third component, tested** (2026-11-17, at the user's request; `dev/2026-11-17_three_component_law/`, README and
outputs there; **[run, here]**). The user's proposal (12:28-12:30): the fast part as it is, a middle part with a time
scale of at most 1 ms, the slow part above 1 ms. [why not two: the fitted fast part has tau_f = 0.009-0.014 ms
(rho_f 0.105-0.249 per 20 us sample; the bound 0.999 allows 20 ms), because the data's autocorrelation falls to 0.56
(SS) and 0.70 (LS) within one sample; a component of 0.1-1 ms keeps 0.82-0.98 after one sample **[arithmetic]**.]
M3 = SPEC 2.1's law + an OU (sigma_m, tau_m), tau_m in [0.04 ms, 1 ms], tau_s in [1 ms, 10 s]; J with the run's
stored Sigma.

| Step | Result over the 18 cells |
|---|---|
| closed form (corrected by each cell's stored simulation) | median J 116.9 (M2 refitted) to 36.7 (M3); the drop exceeds chi^2_2's 99 % point in 18 of 18; a white fast part (5 parameters) does as well except in 529878215 and 569844159 |
| simulation, the layout rebuilt from the JSON | M2 at theta_hat(c) reproduces the run's J within 0.97-1.07 |
| simulation, M3 refined (6 trf iterations at M = 1000, J at M = 3000) | median J 120.7 to 37.9 (ratio median 0.36, 0.08-0.87), lower in 18 of 18; below chi^2_9's 99 % point (21.7) in 5 |
| SS rho(1 ms) | standardised residual median +2.27 to -0.50; the law above the data in 16 to 6 cells |
| fitted middle part | sigma_m 0.012-0.024 mV; tau_m 0.07-0.11 ms in 8 cells, 0.15-0.39 ms in 4, 0.77-1.0 ms in 6 (3 at the 1 ms bound) |
| fitted slow part | tau_s 10-650 ms in 17 cells (median 118 ms); 571624532 189 ms (1863 ms under M2); 569844159 at the 10 s bound |
| still far | 569844159 398, 529878215 220, 571483807 159, 541545716 144, 526785799 103: the cells whose SS and LS statistics disagree under one law per cell |

Limits: Sigma is M2's; the refinement is short (its J an upper bound of M3's minimum); the per-sweep stage is not
tested (the JSONs hold no per-sweep targets); the statistic set was chosen for four parameters.

Read back from the pilot: the `NOISE_CELL` lines, each JSON's per-stage seconds (they set `WALLTIME`), the
`NOISE_MERGE` line. A code change between the pilot and the run re-runs every cell, the pilot's included (the resume
rule keys on the commit). Gate logs: `~/ih_fit_gate_logs/<date_time>/`.

## 5. Testing state

| What | State | Basis |
|---|---|---|
| Block 9 smoke | 11/11 in 404 s at `2171519`; 17 planted defects, each caught | [run 2026-11-12] |
| Block 6 smoke | R1-R9 9/9 at `1d848f6` (2 cores, Python 3.13; 508 s before the day's last edit, 675 s after it with two other runs sharing the cores); 26 planted defects, each caught | [run 2026-11-13] |
| Blocks 1-4 smoke | 6/6, 10/10, 15/15, 11/11 at `a916c2c`; their code unchanged since | [run 2026-11-11; src] |
| gates smoke | 8/8 after the G6d change | [run 2026-11-12] |
| sandbox packages | numpy 2.5.3, scipy 1.18.1; statsmodels is NOT preinstalled: `pip install --break-system-packages statsmodels` (minutes) before Block 9's P3 | [run 2026-11-12] |
| real cells | Blocks 3, 4 and 9 have never run on a real cell nor at full size | [log] |
| on davinci, 2026-11-14 | FULL gates SETTLED at `62f3324` (16 passed); `smoke_noise_run.py` 9/9 in 536 s (in an interactive job on dvnode018) | [cluster, pasted 16:16] |
| the job script on davinci | never run there: its roles, the run file, `qsub -J`, `-W depend=afterany` on an array id and the marker were checked only against stubs | [run 2026-11-13, sandbox] |

## 6. After the run: Block 5's questions (carried from the 2026-11-11 handoff, §6)

1. The twins' SS sweeps as continuous 37-pulse trains at the real stored positions (slow part and tilt continuous across
   pulses), not `simulate_ss_protocol`'s independent pulses -- open since 2026-10-20.
2. Which positions: every pulse of the real train, or only the kept windows'.
3. The per-sweep offset in a twin (invisible to the noise statistics, visible to the I_h fit's absolute-mV gate).
4. The LS twins: one per real LS sweep at its own amplitude (D-076 (iii)), with theta_tilde(c, s) and b_hat(c, s).
5. The sampling rate: rho_f is per 20 us sample (50 kHz); the twin's trace must be at that rate when the noise is added.
6. The manifest's per-sweep columns, the back-fill of old manifests, the false-positive controls carrying the same noise.
7. Stage 7's cohort size, raw against corrected SWCs (D-013), the tolerances (the assistant's proposal still).
8. [added 2026-11-17, from the tables' read-back, §4] The law misfits every cell (J/dof 2.7-160, p < 0.01 in 18 of
   18): whether the twins may carry it as it is depends on where the misfit sits -- the fast part's shape, as D-055's
   S3.1 expects (sub-ms; probably of little weight for a passive fit on ms-to-s responses **[reasoning]**), or the
   slow part. The per-cell JSONs answer it (S3.1 (b), S3.3). Also a rule for degenerate per-sweep fits: the twin of
   537297470 LS 67 would carry no fast part and a sixth of its sweep's SD (§4); D-076 (i) leaves sigma_f and rho_f
   unpooled, and D-122.3's fallback covers failed sweeps only.
   [updated 2026-11-17 (later, 2), from the JSONs' read-back, §4: the misfit is not sub-ms. Two decisions for the
   user before Block 5's SPEC entry: (a) the law kept as it is for the twins -- SDs right to about 5 %, the 1 ms
   autocorrelation about 12 % too high, tau_s loosely determined -- with the misfit recorded as a limit of Stage 7
   and D-055's S2-type sensitivity check as the way to size it, or the law extended (a component at about 1 ms:
   SPEC 2.1, Blocks 2, 4 and 9 and the run again); (b) a rule for a per-sweep fit that is degenerate but not
   "failed" -- the assistant's proposal: sigma_f(c, s) < 0.5 sigma_f(c) or rho_f(c, s) > 0.4 (6 of 515 LS sweeps,
   none of 64 SS) counts as failed, so the twin carries theta_hat(c) (D-122.3).]
   [updated 2026-11-17 (later, 3): the extension tested (§4, "A third component, tested") -- supported in every
   cell, J to about a third, the 1 ms excess gone, 13 of 18 cells still rejected. If the user extends the law, to
   settle before SPEC 2.1 is rewritten: tau_m's bounds (the 1 ms cap binds in 3 cells; the lower one); the fast
   part's rho_f free or white; which of sigma_m, tau_m are per sweep in Block 9; the statistic set (lags for six
   parameters); then Blocks 2, 4, 9, their smokes, the gates and the run again. Decision (b) waits for it.]

## 7. Repository, environment, cluster

| Item | Value |
|---|---|
| repository | `Leonardodm00/Towards-EEG` (public), branch `main` |
| push from a new session | call `add_repo` (owner Leonardodm00, repo Towards-EEG, access push) first: without it the git proxy refuses the push (403) [run 2026-11-12] |
| a light clone | `git clone --filter=blob:none --sparse https://github.com/Leonardodm00/Towards-EEG.git && cd Towards-EEG && git sparse-checkout set "Passive Features/HPC script/Ih Fit" docs/project specs/roadmap`; avoid `git grep` over the whole tree (it fetches every blob) |
| commit identity | the user's: `git config user.name "$(git log -1 --format=%an)"`, the same for the address (`%ae`); never write the address into a file |
| cluster clone | `/davinci-1/home/ldellamea/TEEG/Towards-EEG`; env `prova` |
| archive | `/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc/specimen_*` (18 cells, 14 with two SS sweeps) |
| roadmap page | redrawn by the scheduled "Roadmap sync" every 2 h (08:12-22:12 Rome); a chat need not write it |

## 8. Rules of the road

- The project instructions' v2: the documents live in the repository (`main`); read and write them there; the copies
  in project knowledge are frozen snapshots; end a reply with the docs written and the commit.
- SPEC first, then code; decisions logged the same turn (`project-decision-log`).
- The assistant never runs anything on davinci; fixes travel through git (`hpc-git-delivery`).
- `.py` and `.sh` ASCII and LF; no `__pycache__` in a commit; theory statuses only at the user's word.

## 9. Traps

| Trap | Why |
|---|---|
| the decision log is 1 MB | a whole read can exhaust a chat's context; search it with grep or a script and edit it with a script |
| D-117 to D-121 | the diameter chat's (merged into `main`'s log 2026-11-12); this chat's decisions of 2026-11-11 are D-122 and D-123, though the commits of 2026-11-11 call them D-117 and D-118 |
| fork and BLAS | the process pool forks after numpy is imported: set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1 in the job script, before Python starts |
| the resume rule | a JSON with status ok and the current code commit is skipped; the code commit is the last commit touching the noise-fit modules (SPEC Block 6), not HEAD, so docs and roadmap commits between the pilot and the run do not re-run the pilot's cells |
| "Block 4b" | is Block 9 |
| P8 (Block 9's smoke) | its at-bound target lies 100 times above the bounds' largest SD: below them the whitened cost is flat and the solve stops short |
| D-055 | S0-S3 dropped for now; its superseded elements are listed in D-125.3 |
| the run file | the jobs read `~/noise_fit_runs/<RUN_TAG>_<GROUP>_<UTC>.env`, not the knobs: a knob changed after `qsub` does not reach a queued job; submit again |
| modified code on davinci | while `git status` shows a modified `noise_*.py` or `run_noise_fit.py`, the resume rule skips nothing (every cell re-runs) and the JSONs are marked `code_dirty`; fixes travel through git |
| the pilot's marker | the pilot's array marks N6 done on the roadmap before the 18-cell run; `ROADMAP_DISABLE=1` at submission keeps a submission off it |
| the submitter inside a PBS job | refused (`PBS_JOBID` set): in an interactive job (`qsub -I`) PBS also sets `NCPUS` (8 there), which the submitter would have taken as the cores per cell; submit from the login node |
| the marker from a compute node | the jobs' roadmap marker pushes to GitHub from the compute node; on dvnode018 `module load proxy` preceded `git pull` **[cluster]**, so the push may fail there -- harmless (the helper never fails the job), the stage just shows no marker **[reasoning, not checked]**; `submit_noise_fit.sh` is outside the resume rule's pathspec, so adding the module to it re-runs no cell |
| a killed job | (walltime, memory) writes no JSON: the merge lists the cell as missing; its log's progress lines show where it stopped; the same `RUN_TAG` again re-runs only the cells without an ok JSON |

## 10. Documents

| Path (repository, `main`) | What for |
|---|---|
| `Passive Features/HPC script/Ih Fit/specs/SPEC.md` | Block 6's entry (the source of the code), Block 9's, section 2.5 |
| `docs/project/TEEG_decisions_and_ideas_log.md` | D-123 to D-126; D-055, D-114, D-122 |
| `Passive Features/HPC script/Ih Fit/docs/TEEG_noise_fit_implementation_handoff_2026-11-11.md` | the history of Block 9 and of D-122 to D-125 |
| `Passive Features/HPC script/Ih Fit/README.md`, `CHANGELOG_ih_fit.md` | the folder's map and stage log |
| `specs/roadmap/allen-mtg-fit.md` | the rendered roadmap |

Skills to load: `scientific-coding`, `hpc-python-compat`, `hpc-git-delivery`, `subproject-roadmap`,
`project-decision-log`, `technical-document`, `explanation-modes`.
