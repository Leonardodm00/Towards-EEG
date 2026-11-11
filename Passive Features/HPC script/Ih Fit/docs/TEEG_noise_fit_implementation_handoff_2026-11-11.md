# TEEG -- noise fit (G4): implementation handoff (2026-11-11)

For the chat that continues the implementation of the recording-noise fit of the I_h fit (roadmap allen-mtg-fit, group
G4; `Passive Features/HPC script/Ih Fit/specs/SPEC.md`, Blocks 1-4 written, Block 9 specified, Blocks 5-7 to specify).
It supersedes, as the place to start, the noise-fit chat's handoff of 2026-10-28 (`docs/TEEG_noise_fit_handoff_2026-10-28.md`),
which stays the history of Blocks 1-4 and of D-095 to D-116.

| Date | Change |
|---|---|
| 2026-11-11 | Created at the user's request ("We are quite behind in the implementation of the G4 Noise fit ... Write an handoff for another chat to go on with the implementation"). The same turn: Block 9's four open points answered (**D-117**), the order after Block 9 decided (**D-118**), both logged; SPEC §2.5, Block 9's entry and §8 updated (`b820498`); roadmap N5 now after N6 (`99760f0`, rendered `ed1514c`; page doc v165). Blocks 1-4's smoke suites re-run on `main` (§5). Project knowledge is full: this document lives only in the repository. |

**Basis tags.** **[run]** executed in the assistant's sandbox (date given); **[src]** read from the code or SPEC on
`main` at `ed1514c`; **[log]** the decision log's text; **[user]** the user's decision or words; **[reasoning]** the
assistant's argument, not checked; **[open]** not decided or not verified.

## Start here: the order of work (D-118)

| # | Step | Blocked by | Where it runs |
|---|---|---|---|
| 1 | **Block 9's code**: `noise_sweep_fit.py` + `smoke_noise_sweep_fit.py` (P1-P11), exactly as SPEC §2.5 and the Block 9 entry state | nothing: D-116 and D-117 settle every point | sandbox |
| 2 | **Block 6's SPEC entry, then its code**: the driver `run_noise_fit.py` and `submit_noise_fit.sh`, which fit Blocks 3, 4 and 9 on $n$ real cells | the user's answers to §3's questions; confirm D-118's reading first | sandbox (smoke), then davinci |
| 3 | **The run on davinci**: `git pull`, the FULL cluster gates, a dry run, a pilot on 1-2 cells, then the $n$ cells | the user runs it (the assistant never reaches the cluster) | davinci |
| 4 | **Block 5's SPEC entry, then its code**: the twins built from the run's per-sweep laws and tilts | the run's tables; the user's answers to §4's questions | sandbox |
| 5 | **Block 7** (the record), then **Stage 7** (Block 8, the recovery gate, D-083) | Blocks 5-6 | sandbox, davinci |

D-118's words **[user]**: "I would fit the noise features on n cells in order then to use these fits to construct the
twins on which to run the synthetic recovery run". The reading above (the run before the twins, Block 6 before Block 5)
is the assistant's, logged as D-118 "to confirm": **confirm it with the user in the first exchange**, together with $n$.

## 1. Where it stands (roadmap at `ed1514c`, 2026-11-11)

`python3 specs/roadmap/roadmap.py status --yaml specs/roadmap/allen-mtg-fit.yaml --json <file>` **[run]**: implementation
planned 7, pushed 4, tested 7, done 8; theory open 8, studying 4, closed 5. The counts have not moved since `854004c`
(2026-11-02).

| Stage | Status | What it is | Note |
|---|---|---|---|
| N1 Block 1 | pushed | `noise_stats.sample_acf` (textbook autocorrelation) | no tester report since 2026-10-07 (`7a49a28`) covers it |
| N2 Block 2 | pushed | `noise_law.py`: the law's simulator, a fixed tilt per trace (D-102.1, D-107.2), closed-form moments | as N1 |
| NT tools | done | `noise_fit_gates.py`, `cluster_gates_noise_fit.sh` | FULL run on davinci 2026-10-20 at `ca028ea` only |
| N3 Block 3 | pushed | `noise_windows.py`, `noise_cell_stats.py`: windows, start (D-110, D-111), median subtraction, tilt offsets (D-115), tilts (D-102, D-107), statistics | as N1 |
| N4 Block 4 | pushed | `noise_fit.py`: the per-cell indirect-inference fit, `CellFitResult.per_sweep_cov` for Block 9 | as N1; never run on a real cell nor at full $M$, $R$ |
| N4b Block 9 | planned | `noise_sweep_fit.py`: per-sweep fit, bootstrap $v(c,s)$, DerSimonian-Laird per protocol | SPEC complete; the page warns that `noise_sweep_fit.py` is not in the tree until it exists |
| N6 Block 6 | planned | driver, S0-S3 checks, PBS job | now depends on N3, N4, N4b only (D-118) |
| N5 Block 5 | planned | the generator's law and the per-sweep twins | now depends on N6 as well (D-118) |
| N7 Block 7 | planned | the record (manifest, noise table, S0-S3 report) | after N6 |
| S7 Stage 7 | planned | the recovery gate on davinci (Block 8, to specify) | after I12, N5, N7; D-083: before any real six-parameter fit |

Theory (the user's, never changed by the assistant): T5, T6, T7, T9 studying; T8 open (the tilt study document of
2026-10-26 not read yet); T10 closed.

## 2. Decided: what binds the next code

| ID | Binds | The decision |
|---|---|---|
| D-073, D-076 (i) | the form | each twin sweep carries noise calibrated on its own real sweep; $\sigma_f$, $\rho_f$ per sweep, $\sigma_s$ per sweep with $\tau_s$ held at the cell's and shrunk towards the protocol's mean, $\tau_s$ per cell **[log]** |
| D-102, D-107 | the tilt | each twin sweep carries its real sweep's realised tilt $\hat b(c,s)$ (SS: (TL1); LS: the line through the 265 ms pre-window and the last 500 ms); every simulated sweep of the fit carries it too; no $\sigma_b$ anywhere **[log]** |
| D-114 | Block 4 | a cell whose sweep tilt cannot be read is refused, quoting the flag **[log]** |
| D-116 | Block 9 | one solve per sweep at the cell's weights from $\hat\theta(c)$; $v(c,s)$ by a parametric bootstrap; DerSimonian-Laird per protocol (SS and LS apart), clipped, the random-effects mean recomputed; a failed sweep out of the pooling **[log]** |
| **D-117** (2026-11-11) | Block 9 | $B_v$ = 50 refits per sweep, each at the binding size $M_{\rm refit}$ = 1000 (the main per-sweep solve keeps $M$ = 3000); a failed refit left out of (P2) and counted, the sweep flagged `v_refits_failed:<n>/<B_v>` above 10 % (a report only); $B_{\rm ok} < 2$ gives `v_none`, a failed sweep; **a failed sweep's twin carries the cell's whole law** $\hat\theta(c)$ (the assistant had recommended the pooled $\sigma_s$; not taken) **[user]** |
| **D-118** (2026-11-11) | order | the noise fit of $n$ real cells (Block 6 on davinci) before the twins (Block 5), which are built from those fits; then Stage 7 -- the assistant's reading, to confirm; $n$ open **[user]** |
| D-083 | Stage 7 | the six-parameter recovery on synthetic twins before any six-parameter fit of a real cell **[log]** |
| D-097 | delivery | the assistant pushes its own work; every document is also in the repository **[log]** |
| D-091 | roadmap | a new SPEC block enters the roadmap in the same turn **[log]** |

The full entries: `docs/project/TEEG_decisions_and_ideas_log.md` (the byte copy of the project's log). **Next free IDs:
D-119, I-005** -- other chats (the diameter chats) log too: re-read the log's last changelog row immediately before
every write.

## 3. Step 1 -- Block 9's code (`noise_sweep_fit.py`)

Everything is in SPEC §2.5 (P0)-(P4), (a)-(h) and the Block 9 entry **[src]**; write the code from them, not from this
summary. The interface, in one place:

| Name | Role |
|---|---|
| `SweepStageConfig` | frozen: `B_v=50`, `M_refit=1000`, `max_failed_refit_frac=0.1` (D-117), `min_stats=4`, `seed=20261030` |
| `sweep_layout(layout, protocol, sweep)` | the cell's `CellLayout` restricted to one sweep (SS: its kept windows and tilt; LS: its pre-window and tilt) |
| `sweep_target(target, layout, protocol, sweep)` | `(q_hat_s (15,), mask (15,))`, NaN/False outside the protocol or where not finite |
| `sweep_seeds(seed, specimen_id, protocol, sweep)` | `SeedSequence([seed, specimen_id, 0 SS / 1 LS, sweep]).spawn(2)` -> `(seed_binding_s, seed_v_s)` |
| `fit_sweep(...)` -> `SweepFit` | (P1): `least_squares(method='trf')` in $u_s = (\log\sigma_f, \rho_f, \log\sigma_s)$, $\tau_s$ held at $\hat\tau_s(c)$, $W_s = \hat\Sigma_s^{-1}$ from `CellFitResult.per_sweep_cov[s]`; fails on status $\le 0$ or a coordinate within $10^{-3}$ of a bound |
| `bootstrap_v(..., refit=fit_sweep)` -> `SweepBootstrap` | (P2): $B_v$ replicates at $\hat\theta(c,s)$ on `seed_v_s`, refitted with the binding function at `M_refit` on `seed_binding_s`; `refit` swappable for a test double |
| `pool_protocol(y, v)` -> `PoolResult` | (P3), (P4), (D-076.1) with the clip and the recomputed mean; the $k$ = 1 and $k$ = 0 rules |
| `run_sweep(...)`, `pool_cell(...)`, `fit_sweeps(...)` -> `SweepStageResult` | one sweep's job; the pooling per protocol and the twins' laws (P0), a failed sweep at $\hat\theta(c)$ (D-117.3); the whole stage; `to_dict()`, `table_rows()` with §2.5 (g)'s columns |

What Block 9 consumes from Block 4 **[src, `noise_fit.py` at `ed1514c`]**: `NoiseFitConfig` (`M_binding=3000`, the
bounds, `fd_rel_step`, `ftol`, `xtol`, `gtol`, `max_nfev=15`, `seed`), `CellLayout`, `CellTarget`,
`simulate_cell_windows`, `binding_function`, `simulated_covariance`, `law_to_u` / `u_to_law`, `_cholesky_lower`,
`CellFitResult` (`per_sweep_cov: Dict[int, ndarray]`, keyed by sweep number, SS and LS alike); Block 3's
`per_sweep_vectors` (`noise_cell_stats.py`).

Smoke `smoke_noise_sweep_fit.py`: P1-P11 as the Block 9 entry lists them (P3's oracle is statsmodels' `combine_effects(...,
method_re="dl").tau2`, which does **not** clip; P11, added 2026-11-11, asserts D-117's defaults). Reduced sizes, about 5
min (the SPEC's estimate) **[open]**. Then the scientific-coding block protocol: check 1 against the SPEC, check 2 by
running, planted defects (mutation test), the SPEC status line `smoke-tested <date>`, `CHANGELOG_ih_fit.md`, commit
`block 9: ...`, push. Also add Block 9's suite to `cluster_gates_noise_fit.sh` (a G6d line, FULL mode) and to the
SPEC's tools section, so the cluster gates cover it [reasoning: proposal; the gates are a tool, not a block].

Cost on Block 4's fixture (2 SS sweeps of 30 and 33 kept windows, 5 LS pre-windows) **[run 2026-10-30, SPEC §2.5 (h)]**:
one solve 0.9 min (SS) or 4.0 min (LS) at $M$ = 3000; at $B_v$ = 50 and $M_{\rm refit}$ = 1000 the whole stage about 6.6 h
on one core for that cell.

## 4. Step 2 -- Block 6, the driver for $n$ cells: ask before writing logic

The per-cell chain the driver wires **[src]**: `noise_windows.load_ss_windows(spec_dir)` -> `ss_noise_windows(ss)` ->
`load_ls_prewindows(spec_dir)` -> `noise_fit.CellLayout.from_cell(nwin, ls)` -> `CellTarget.from_cell(nwin, ls, layout)`
-> `fit_cell(target, layout, cfg)` -> `noise_sweep_fit.fit_sweeps(cell_fit, target, layout, cfg, stage_cfg)`. Inputs per
cell: `ss_pulses.npz` (Format A), `trace_qc_results.csv`, `metadata.json`, `ls_sweeps.npz` in
`<archive root>/<GROUP>/specimen_<id>/`.

Questions for the user (SPEC entry after the answers; proposals are the assistant's):

| # | Question | Proposal | Why it matters |
|---|---|---|---|
| 1 | D-118's reading: the run before the twins, Block 6 before Block 5 | as logged | it reorders the roadmap (done 2026-11-11 on that reading) |
| 2 | $n$ and which cells | all 18 L3_exc cells (the group of the noise tables and of the Stage-7 twins), after a pilot on 508282493 and 569844159 (the two cells G12 read) | sizes the run; Stage 7's cohort size is also open (S7) |
| 3 | job layout | a PBS array over cells, each job running Blocks 3, 4 and 9 for its cell in sequence, then one merge step writing the group tables; the alternative is an array over (cell, sweep) for Block 9 (SPEC §2.5 (h)), faster but with a join step | wall time against moving parts; one cell is about 6.6 h of Block 9 plus the per-cell fit [run, fixture] |
| 4 | outputs | per cell a JSON (`CellFitResult.to_dict()` + `SweepStageResult.to_dict()`); per group `noise_cells_<GROUP>.csv` (Block 4's `noise_*` columns) and `noise_sweeps_<GROUP>.csv` (§2.5 (g)'s `noise_sweep_*` columns); under `$IH_OUTPUT_BASE/noise_fit/<RUN_TAG>/<GROUP>/` | Block 5 reads these |
| 5 | the S0-S3 checks of D-055 (in N6's summary) | not in the first run; specified after it | D-055 predates D-102/D-107/D-115/D-116 (it still names $\sigma_b$ and the tilt alternation, both retired): its scenarios must be revised with the user before any `sens_*.py` is written [log] |
| 6 | a refused or failed cell | reported with its reason (D-114's flag, a failed fit), the run goes on; the merge lists them | a refused cell has no twins |
| 7 | the pilot's sizes | a dry run (`DRY_RUN=1`, no python), then the 2-cell pilot at full size | the full-size per-cell fit has never run [open] |

The PBS script follows `submit_ih_fit.sh` **[src]**: `#PBS -q cpu`, `select=1:ncpus=1`, the walltime set from the pilot,
`IH_CODE_DIR`, `IH_ARCHIVE_ROOT` (default `/davinci-1/home/ldellamea/Human Neurons Fitting`), `IH_OUTPUT_BASE`,
`CONDA_ENV`, `DRY_RUN=1`; it carries the roadmap marker lines (`subproject-roadmap` skill, `references/hpc_marker.md`;
the job name `noise_fit` is already in N6's `jobs`). The noise fit needs no NEURON (numpy, scipy, statsmodels only)
[src: imports of `noise_*.py`]. Apply `hpc-python-compat` (ASCII, LF) and `hpc-git-delivery` (verify, ship through git,
dry run first).

## 5. Testing and delivery state

| What | State | Basis |
|---|---|---|
| Blocks 1-4 smoke on `main` at `a916c2c` | `smoke_noise_stats` 6/6, `smoke_noise_law` 10/10, `smoke_noise_fit_gates` 8/8, `smoke_noise_windows` 15/15 (16 s), `smoke_noise_fit` 11/11 (84 s); numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0 | [run 2026-11-11] |
| night tester (integration tester) | no report covers N1-N4 (last report 2026-10-07 at `7a49a28`); that is why the roadmap says "pushed" | [run: roadmap status] |
| cluster gates | last FULL run 2026-10-20 at `ca028ea` (13/13, SETTLED); Blocks 2-4 changed since; the script expects the current counts (M10, W15, F11) | [log; src] |
| real cells | Blocks 3-4 have never run on a real cell nor at full $M$, $R$ | [log] |

Propose to the user (their call, not the assistant's): (a) the night test on Blocks 1-4 (and 9 once pushed) -- the
`scientific-coding` skill's integration tester, on the user's "run the night test"; (b) the FULL cluster gates on
davinci before the driver's first real run:

    cd /davinci-1/home/ldellamea/TEEG/Towards-EEG && git pull
    cd "Passive Features/HPC script/Ih Fit" && conda activate <env> && bash cluster_gates_noise_fit.sh

(about 17 min; the env of 2026-10-20 was `prova`; logs in `~/ih_fit_gate_logs/<date_time>/`; VERDICT SETTLED needed).

## 6. Step 4 -- Block 5, the twins: questions to settle when its SPEC entry is written

| # | Question | State |
|---|---|---|
| 1 | the twins' SS sweeps: continuous 37-pulse trains at the real stored positions, the slow part and the tilt continuous across pulses (SPEC §8), instead of `simulate_ss_protocol`'s independent pulses | open since 2026-10-20 [log] |
| 2 | which positions: the real train's every pulse, or only the stored (kept) windows' | open [reasoning] |
| 3 | the per-sweep offset $a$ in a twin (invisible to the fit's statistics, visible to the I_h fit's absolute-mV gate) | open [reasoning] |
| 4 | the LS twins: one per real LS sweep at its own amplitude (D-076 (iii)), with $\tilde\theta(c,s)$ and $\hat b(c,s)$ | the amplitudes decided (D-076 (iii)) [log] |
| 5 | the noise's sampling rate: the law's $\rho_f$ is per 20 $\mu$s sample (50 kHz); the twin's trace must be at that rate when the noise is added | to state in the SPEC [reasoning] |
| 6 | the manifest's per-sweep columns and the back-fill of existing manifests; the false-positive controls carry the same noise | N5's summary [log] |
| 7 | Stage 7's cohort size, raw against corrected SWCs (D-013), the tolerances (still the assistant's proposal) | open (S7's note) [log] |

## 7. Repository, environment, cluster

| Item | Value |
|---|---|
| repository | `Leonardodm00/Towards-EEG` (public: no credentials, no personal data, no PDFs), branch `main` |
| sparse checkout that runs every noise smoke | `git clone --filter=blob:none --sparse https://github.com/Leonardodm00/Towards-EEG.git && cd Towards-EEG && git sparse-checkout set "Passive Features/HPC script/Ih Fit" docs/project specs/roadmap` |
| commit identity | the user's, set in the clone's git config (check `git config user.name`); never write the address into a file |
| commit trailers | the two lines the session's reminder gives (Co-Authored-By, Claude-Session) |
| cluster clone | `/davinci-1/home/ldellamea/TEEG/Towards-EEG` [src: `Ih Fit/README.md`] |
| archives | `/davinci-1/home/ldellamea/Human Neurons Fitting/<GROUP>/specimen_*` (L3_exc: 18 cells; 14 with two SS sweeps) [src; log] |
| roadmap page | "Subproject Roadmaps", artifact `Fe6Zqjr5iq3XAQZfrTspqB`, collection `roadmaps`, doc `towards-eeg:allen-mtg-fit` (v165 written 2026-11-11 from `ed1514c`); the scheduled "Roadmap sync" redraws it every 2 h |

## 8. Rules of the road

- Push your own work (D-097); every document also in the repository; workstream docs beside the code
  (`Ih Fit/docs/`, `Ih Fit/dev/<date>_<topic>/`; `docs/project/README.md`).
- Project knowledge is full since 2026-10-28: a new document is refused; an existing one can be rewritten (the log
  was, 2026-11-11). Write the project copy first, then the repository copy, byte for byte.
- A decision the user makes is logged the same turn (`project-decision-log` skill), with the index and changelog rows.
- SPEC first, then code (`scientific-coding`); a new SPEC block enters the roadmap the same turn (D-091; Block 6 and
  Block 5 already have nodes N6, N5 with `block:` set, so their SPEC entries link by themselves).
- Never change a theory status, never add a topic or step unasked (roadmap rules 4-5); fetch and fast-forward before
  editing the roadmap; commit the roadmap files separately.
- `.py` and `.sh` ASCII and LF; delete `__pycache__` before committing; dev checks are standalone scripts in
  `Ih Fit/dev/`, not pipeline code.
- The assistant never runs anything on davinci: the user runs the commands; fixes travel through git, never by hand
  edits on the cluster (`hpc-git-delivery`).

## 9. Traps

| Trap | Why |
|---|---|
| "Block 4b" = Block 9 | the roadmap tool and the tester read integer block numbers; 5-8 are held by the later stages (Block 8 = Stage 7's gate, proposed); the log's D-082 to D-116 say "Block 4b" |
| statsmodels `combine_effects(method_re="dl")` | returns $\hat\tau^2$ unclipped and weights with it; Block 9 clips and recomputes the mean; statsmodels is only the smoke's oracle (D-076's clarification) |
| $\hat\tau^2_{\rm pr}(c)$ vs $\tau_s$ | the first is a between-sweep variance of $\log\sigma_s$, the second the slow time constant |
| $M_{\rm refit}$ = 1000 vs $M$ = 3000 | the refits use a noisier binding function than the main solve; $v(c,s)$ is the variance of the $M$ = 1000 estimator ($-$0.5 % to +2.3 % from $M$ = 3000's, `dev/2026-10-30_sweep_stage_cost/`) |
| a failed sweep | D-117.3: its twin takes $\hat\theta(c)$, all four values; a `v_none` sweep is a failed sweep too |
| the fixture cell's start | Block 4's smoke fixture falls back to the 80 ms start (its sweep 92 carries a $-$0.25 mV/s tilt); expected, not a bug (`dev/2026-10-29_start_noise/`) |
| D-055 (S0-S3) | written before D-102/D-107: it names $\sigma_b$ and the tilt alternation, both retired |
| `per_sweep_cov` keys | sweep numbers, SS and LS in one dict |
| the roadmap page's warning | "module tokens of SPEC.md not found in the tree: N4b noise_sweep_fit.py" until Block 9's code is pushed |

## 10. Documents and files

| Path (repository) | What for |
|---|---|
| `Passive Features/HPC script/Ih Fit/specs/SPEC.md` | the source of truth: §2.1-2.5, §5 Blocks 1-4 and 9, §6 commands, §8 open questions |
| `docs/project/TEEG_decisions_and_ideas_log.md` | D-017, D-033, D-034, D-048 to D-051, D-055, D-056, D-072, D-073, D-076, D-083, D-095, D-102 to D-107, D-110, D-111, D-114 to D-118 |
| `Passive Features/HPC script/Ih Fit/docs/TEEG_noise_fit_handoff_2026-10-28.md` | the previous noise-fit handoff: history, numbers at $\theta_{\rm cf}$ (§4), traps (§6) |
| `docs/project/claude/TEEG_Stage7_noise_handoff_2026-10-01.md` | the Stage-7 noise thread's living handoff: paths, traps |
| `Passive Features/HPC script/Ih Fit/dev/2026-10-30_sweep_stage_cost/` | Block 9's cost and the refits' binding size |
| `Passive Features/HPC script/Ih Fit/dev/2026-11-02_bootstrap_explained/` | why $B_v$ = 50 suffices |
| `Passive Features/HPC script/Ih Fit/README.md`, `CHANGELOG_ih_fit.md` | the folder's map and stage log |
| `specs/roadmap/allen-mtg-fit.md` | the rendered roadmap |
| `Passive Features/HPC script/Ih Fit/docs/TEEG_deck_handoff_passive_ih_2026-11-03.md` | the whole passive + I_h story, for context |

Skills to load: `scientific-coding` (block protocol, smoke tests), `subproject-roadmap`, `project-decision-log`,
`hpc-python-compat`, `hpc-git-delivery`, `technical-document` (handoffs), `explanation-modes` (explanations to the user).
