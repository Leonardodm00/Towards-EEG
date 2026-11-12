# TEEG -- noise fit (G4): Block 6 handoff (2026-11-12)

For the chat that writes Block 6's code (the noise-fit run: driver, PBS array, merge) and then runs it on davinci with
the user. Supersedes, as the place to start, `TEEG_noise_fit_implementation_handoff_2026-11-11.md` (this folder),
which stays the history of Block 9 and of D-122 to D-125.

| Date | Change |
|---|---|
| 2026-11-12 | Created at the user's request ("The handoff then"), after D-125 (Block 6's five questions answered) and D-126 (its two readings confirmed). Same day, before it: Block 9 written and pushed (`2171519`); the noise chat's D-117/D-118 renumbered D-122/D-123 (`a05ccdb`); D-124 (the cohort); the project instructions became v2 (the repository is the source of truth); the diameter chat's D-117 to D-121 and I-005 merged into `main`'s log (`3dc3703`); Block 6's SPEC entry drafted (`3dc3703`). |

**Basis tags.** **[run]** executed in the assistant's sandbox (date given); **[src]** read from the code, SPEC or log
on `main`; **[log]** the decision log's text; **[user]** the user's decision or words; **[reasoning]** the assistant's
argument, not checked; **[open]** not decided or not verified.

## Start here: the order of work

| # | Step | Blocked by | Where it runs |
|---|---|---|---|
| 1 | **Block 6's code**: `run_noise_fit.py`, `submit_noise_fit.sh`, `smoke_noise_run.py` (R1-R8), exactly as SPEC Block 6 states | nothing: D-123 to D-126 settle every point | sandbox |
| 2 | **The run on davinci** (the user runs it, §4): `git pull`, the FULL gates, a dry run, the 2-cell pilot at full size, then the 18 cells with the pilot's walltime | step 1 | davinci |
| 3 | **Block 5's SPEC entry, then its code**: the twins built from the run's tables (§6's questions first) | the run's tables | sandbox |
| 4 | **Block 7** (the record), then **Stage 7** (Block 8, the recovery gate, D-083) | Blocks 5-6 | sandbox, davinci |

Open with the user, unanswered: whether the S0-S3 checks get their own roadmap stage (proposal: N6b after N6, under G4).
Roadmap rule 4: propose, add only on a yes.

## 1. Where it stands (roadmap at `9526401`, 2026-11-12) [run]

Implementation planned 6, pushed 5, tested 7, done 8; theory open 8, studying 4, closed 5.

| Stage | Status | Note |
|---|---|---|
| N1-N4 Blocks 1-4 | pushed | no tester report covers them (last 2026-10-07, `7a49a28`) |
| NT tools (gates) | done | FULL run on davinci only 2026-10-20 at `ca028ea` (13 checks); now 14 (G2 and G6d cover Block 9): re-run before the pilot |
| N4b Block 9 | pushed | `noise_sweep_fit.py` at `2171519`, smoke P1-P11 11/11 |
| N6 Block 6 | planned | SPEC entry drafted 2026-11-12; code not written |
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

## 3. Step 1 -- Block 6's code

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

The shape of the commands; finalise them with the code **[open]**:

    cd /davinci-1/home/ldellamea/TEEG/Towards-EEG && git pull
    cd "Passive Features/HPC script/Ih Fit" && conda activate prova && bash cluster_gates_noise_fit.sh   # FULL, about 25 min; VERDICT SETTLED needed
    RUN_TAG=<tag> CELLS=508282493,569844159 DRY_RUN=1 bash submit_noise_fit.sh   # prints every command, submits nothing
    RUN_TAG=<tag> CELLS=508282493,569844159 bash submit_noise_fit.sh             # the pilot, full size
    RUN_TAG=<tag> WALLTIME=<from the pilot> bash submit_noise_fit.sh            # the 18 cells; the pilot's two skipped (same commit)

Read back from the pilot: the `NOISE_CELL` lines, each JSON's per-stage seconds (they set `WALLTIME`), the
`NOISE_MERGE` line. A code change between the pilot and the run re-runs every cell, the pilot's included (the resume
rule keys on the commit). Gate logs: `~/ih_fit_gate_logs/<date_time>/`.

## 5. Testing state

| What | State | Basis |
|---|---|---|
| Block 9 smoke | 11/11 in 404 s at `2171519`; 17 planted defects, each caught | [run 2026-11-12] |
| Blocks 1-4 smoke | 6/6, 10/10, 15/15, 11/11 at `a916c2c`; their code unchanged since | [run 2026-11-11; src] |
| gates smoke | 8/8 after the G6d change | [run 2026-11-12] |
| sandbox packages | numpy 2.5.3, scipy 1.18.1; statsmodels is NOT preinstalled: `pip install --break-system-packages statsmodels` (minutes) before Block 9's P3 | [run 2026-11-12] |
| real cells | Blocks 3, 4 and 9 have never run on a real cell nor at full size | [log] |

## 6. After the run: Block 5's questions (carried from the 2026-11-11 handoff, §6)

1. The twins' SS sweeps as continuous 37-pulse trains at the real stored positions (slow part and tilt continuous across
   pulses), not `simulate_ss_protocol`'s independent pulses -- open since 2026-10-20.
2. Which positions: every pulse of the real train, or only the kept windows'.
3. The per-sweep offset in a twin (invisible to the noise statistics, visible to the I_h fit's absolute-mV gate).
4. The LS twins: one per real LS sweep at its own amplitude (D-076 (iii)), with theta_tilde(c, s) and b_hat(c, s).
5. The sampling rate: rho_f is per 20 us sample (50 kHz); the twin's trace must be at that rate when the noise is added.
6. The manifest's per-sweep columns, the back-fill of old manifests, the false-positive controls carrying the same noise.
7. Stage 7's cohort size, raw against corrected SWCs (D-013), the tolerances (the assistant's proposal still).

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
