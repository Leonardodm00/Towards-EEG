# TEEG -- the twins (Block 5): handoff (2026-11-19)

For the chat that writes Block 5 -- the twins' generator: synthetic copies of the 18 L3_exc cells, each on its own
real morphology, stimulated with its own real protocol and carrying each real sweep's fitted recording noise -- and
then builds the twins on davinci for Stage 7's recovery of the six parameters (D-083). Supersedes, as the place to
start, `TEEG_noise_fit_block6_handoff_2026-11-12.md` (this folder), which stays the history of the noise-fit run
(nf1 under the two-component law, the three-component law, nf2).

| Date | Change |
|---|---|
| 2026-11-19 | Created at the user's request ("Now we will build the synthetic runs in another chat. Write the extensive handoff and update the roadmap"), after D-180 (the twins no longer than the noise fit's windows) and the nf2 run (18 ok). Facts checked against `main` at `b17f768` and the user's pastes of 2026-11-18/19. |
| 2026-11-19 (later) | The twins chat (Fable 5.2): section 5a added -- the user's answers of 21:23 to section 5's questions (D-182 to D-186), what they leave open, and the plan as it stands; "Start here" step 1, section 9's next free IDs and a trap (binary files through the Drive connector) updated. Two standalone checks in `Ih Fit/dev/2026-11-19_twins/`. |
| 2026-11-19 (later, 2) | Section 5a: the user's answers of 21:46 (D-187 to D-189) -- single SS pulses from rest, the configurations' cells and arms, the bundles' check dropped; the plan approved; the nf2 tables and the archive cell 508282493 received in the chat. |
| 2026-11-20 | The twins chat: section 5b added -- the generator and the Stage-7 run as built on `sci/twins` (Blocks 5, 10, 11, 12, 13, 8 merged, head `577445d`), the decisions taken on the way (D-190 to D-193, D-195), the commands for davinci, and what is still open. The independent final check of the assembled pipeline was running when this was written. Facts from the code and the runs of this chat. |
| 2026-11-20 (later) | Section 5b brought to `sci/twins` at `216ecf1`: the final check (an independent Opus tester on `577445d`: PASS WITH ISSUES, report `Ih Fit/specs/test_reports/2026-11-20_1432_incremental.md`) and its three findings, fixed in Block 8 (D-196; TS1-TS10 10/10); a second check of the changed block was running when this was written. |
| 2026-11-20 (later, 2) | Section 5b at the end of the implementation: `sci/twins` at `01ebd72`; the second independent check (`216ecf1`: PASS WITH ISSUES, nothing major; report `Ih Fit/specs/test_reports/2026-11-20_1532_incremental.md`) and its two points fixed (D-197); the six twins' suites run in sequence on the final branch all exit 0; the cluster checkout line added. Not yet run on davinci. |
| 2026-11-20 (later, 3) | Section 5b, the cluster steps: the gate line now carries `conda activate prova` and the `cd` (run from the repository root under `(base)` it found no file); the pilot is one twin per configuration with a 48 h ceiling; a one-liner for the pilot's times; the expected cost. A trap added (section 10). |
| 2026-11-20 (later, 4) | The gates on davinci (16:35): 58 checks passed; TS7 of `smoke_twin_recovery.py` ERROR (`manifest.csv` not found). Cause: NEURON exits (`can't open DISPLAY`) when `DISPLAY` is set and no X server answers; the test's fake `HOME` hid the X11 cookie of the MobaXterm login. Fixed by SPEC Block 8 amendment 10 (`84d099d`; the submitter unsets `DISPLAY`; NEURON's version recorded without pip metadata); `sci/twins` at `a28d1a5`. A trap added. |
| 2026-11-20 (later, 5) | Section 7 brought up to date: Block 8 built (the draft around `submit_ih_recovery.sh` superseded), how to read the run's outputs, the gate's default tolerances and what a failing axis means for Stage 8 (D-005, D-083). |
| 2026-11-20 (later, 6) | Section 5b: the run as decided (D-200: no pilot; the 30 twins at once with a 48 h wall time) and the early check. |
| 2026-11-22 | Section 5b: the run has ended (the user's early-check line: 30 status files, 30 logs in the fit, no traceback); the read-out is next. |
| 2026-11-22 (later) | Section 7: the result of run tw1 (gate FAIL; 30 ok; job IDs; run times) and its reading (the foot of `Ih_human` at the voltages the steps reach; the optimiser question left open), from the user's extract and `dev/2026-11-22_tw1_readout/`. |
| 2026-11-22 (later, 2) | Section 7: the user's decision after the FAIL (D-204: a new training set, step and sample weights, 5 noise-free twins; forms pending) and the split of the 10 `full` twins by sag against noise. |
| 2026-11-22 (later, 3) | Section 7: the settled forms (D-206 to D-209), the fan-out in progress (contract `aac46d3`), and the loss-at-truth diagnostic delivered for davinci. |
| 2026-11-22 (later, 4) | Section 7: Blocks 14-18 built and checked (`c974a62`), the commands of run `tw2`, the real-cell check of the new loss. |
| 2026-11-24 | Section 7: the result of run `tw2` (gate FAIL; the truth fits better than the estimate in 14 of 15 twins fitted with I_h: the optimiser is the limit). |
| 2026-11-25 (the review chat) | Section 7: the user's decision after the independent review (D-245: TRF on the squared loss; the sensitivity module dropped), the code facts the review asked for **[src at `393b681`]** and the re-read of tw3a/tw3b's `refine_starts.csv` **[data inspected]**. |
| 2026-11-25 (the review chat, evening) | Section 7: Blocks 23 to 28 built and checked on `sci/twins` (`de4cc53`), what changed in the fit and the twins' run, the final check's verdict, and the two commands for the next run on the noisy twins (D-245 to D-250 [corrected 2026-11-27: the last is now D-262, renumbered; D-250 is the soma fit's]). |
| 2026-11-27 (the defocus-table chat) | With the user's OK, the review chat's D-250 (the final check of Blocks 23 to 28) is renumbered **D-262** in the log, ending the clash with the soma fit's D-250; this handoff's two citations corrected. The citations of D-250 in `Ih Fit/` on `sci/twins` (SPEC Block 28's notes, `run_twin_recovery.py`, `smoke_twin_sensitivity.py`, `tests/spec/test_spec_block8_twin_recovery.py`, `tests/spec/test_spec_block22_twin_recovery_run3.py`) were left untouched so as not to mark Block 28 as changed: they mean D-262; update them the next time those files change. |
| 2026-11-27 (the review chat) | Section 7: the results of the sensitivity run `tw5` on the ten noisy twins **[data inspected]** and what they imply for the gate, the exclusion rule and $\tau_d$ (not decided). |

**Basis tags.** **[src]** read from the code, SPEC or roadmap on `main` (`b17f768`); **[log]** the decision log's
text; **[user]** the user's words or decision; **[cluster]** output the user pasted from davinci; **[run]** executed
in the assistant's sandbox (date given); **[reasoning]** the assistant's argument, not checked; **[open]** not decided.

**Model.** The coding directives (v2, 2026-11-13) put planning and orchestration on the most powerful model
available -- Claude Fable 5.2 on 2026-11-19 -- and the implementation and the final check on Opus agents. This
handoff was written on Opus 5.5. Start the twins chat on Fable 5.2.

## Start here: the order of work

| # | Step | Blocked by | Where it runs |
|---|---|---|---|
| 1 | **Settle Block 5's open questions with the user** (§5), first Q1-Q3: what D-180 means for the Long Square twins, for the SS response (trains or single pulses) and for the SS bundles [2026-11-19 (later): mostly settled, D-182 to D-186 -- §5a lists what is still open] | nothing | chat |
| 2 | **Block 5's SPEC entry** in `Ih Fit/specs/SPEC.md`: interfaces, shapes, units, oracles; the roadmap's N5 reads it by itself (§8) | 1 | sandbox |
| 3 | **Plan, then implement** (directives 6-9; the `scientific-coding` skill, "Fan-out"): the orchestrator writes the plan and the interfaces, the user approves, Opus implementers write the blocks, an independent Opus agent makes the final check | 2 | sandbox |
| 4 | **The twins on davinci** (the user runs it): `git pull`, gates, a dry run, the twins of 1-2 cells, then the 18 | 3 | davinci |
| 5 | **Stage 7 itself**: Block 8's SPEC entry (the gate run), the recovery fits of the four arms, the gate on `ih6`; then Block 7, the record (N7) | 4 | sandbox, davinci |

## 1. Where it stands (2026-11-19)

- **The noise law is three-component** (D-152 to D-155; SPEC 2.6) **[log; src]**: a white Gaussian fast part
  $\sigma_f$, a middle OU part ($\sigma_m$, $\tau_m \in [0.04, 1]$ ms) and a slow OU part ($\sigma_s$,
  $\tau_s \in [1\,{\rm ms}, 10\,{\rm s}]$); per sweep the three sizes are fitted with both time scales held at the
  cell's; $\sigma_m$ and $\sigma_s$ are shrunk towards their protocol's mean, $\sigma_f$ is not; a sweep whose fit
  fails (no convergence, a bound, $\hat\sigma_f(c,s) < 0.5\,\hat\sigma_f(c)$, a bootstrap variance NaN or below
  $10^{-12}$) gives its twin the cell's whole law.
- **The nf2 run is done** **[cluster]**: pilot 2 ok (2026-11-18; array `2098689[]`, merge `2098690`), then
  `NOISE_MERGE L3_exc: 18 ok, 0 refused, 0 failed, 0 missing` (merge job `2099736`, 2026-11-19). Code commit of the
  JSONs: `e9a6469` (the pilot's; the 18-cell run kept the pilot's two cells, which the resume rule allows only at the
  same commit **[reasoning]**).
- **Read back** (the Block 6 handoff, §4 "The nf2 run read back"; `Ih Fit/dev/2026-11-19_nf2_readback/`)
  **[cluster; run]**: misfit per degree of freedom lower than the two-component law's in 18 of 18 cells (median 10.8 to
  2.65); no coordinate at a bound; the per-sweep laws rejected at 1 % in 19 % of SS and 23 % of LS sweeps (nf1: 69 %
  and 56 %); 15 of 18 cells used every refinement round (`no_plateau`, D-050) without drifting at the end.
- **D-180** (user, 2026-11-19) **[log]**: the twins are no longer than the windows the noise fit used (at most
  265 ms), and nothing relies on how the noise behaves across pulses. It answers the slow-part finding below; how it
  applies to the Long Square twins and to D-083.2's 37-pulse trains is open (§5, Q1-Q3).
- **The slow-part finding** **[cluster; run]**: where $\hat\tau_s(c)$ is long against the windows, the fit
  determines $\sigma_s^2/\tau_s$, not $\sigma_s$ and $\tau_s$ apart. Over the late refinement rounds $\tau_s$ moved by
  more than 20 % in 7 cells while $\sigma_s^2/\tau_s$ moved by 5-29 % (5 % in 526785799, $\tau_s$ 0.89-1.36 s, and in
  571430815, 5.66-8.06 s). A closed-form check on 571430815 (`check_slow_ridge.py`: each window's mean removed;
  the median-window subtraction and the tilt left out) moved its slow part from 5.66 s to 1 s along that ridge: no statistic
  changed by more than 0.35 SE, while over a 7.42 s SS train the slow part's SD went from 0.92 to 0.59 mV.
- **Roadmap** (allen-mtg-fit, status at `b17f768`) **[src]**: N5 (Block 5) planned -- no SPEC entry; N7 (the record)
  planned; S7 (Stage 7) planned -- no SPEC entry for Block 8; I12 (Stage 7 as built: manifest, generator, report,
  driver) tested; N1-N4b and N6 show `pushed` (§10: the tool does not see this folder's test reports, and N6's
  marker never reached GitHub).

## 2. The input: the nf2 outputs

### 2.1 Where they are (davinci) [cluster; src]

| What | Path |
|---|---|
| group folder `OUT` | `/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs_ih/noise_fit/nf2/L3_exc` |
| one JSON per cell | `$OUT/cells/specimen_<id>.json` (18) |
| per-cell table | `$OUT/noise_cells_L3_exc.csv` (18 rows) |
| per-sweep table | `$OUT/noise_sweeps_L3_exc.csv` (one row per SS sweep with kept windows and per LS sweep: 64 SS + 525 LS **[run, on nf1's table; the layouts are unchanged]**) |
| logs | `$OUT/logs/cell_<id>_<job>.log`, `$OUT/logs/merge_<job>.log` |
| run files | `~/noise_fit_runs/nf2_L3_exc_<UTC>.env` |
| the superseded run | `.../noise_fit/nf1/L3_exc` (two-component law): feeds no twin |

To bring them into a sandbox (one file to attach):

    cd "/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs_ih/noise_fit/nf2/L3_exc" && tar -czf ~/nf2_L3_exc.tar.gz noise_cells_L3_exc.csv noise_sweeps_L3_exc.csv cells && ls -la ~/nf2_L3_exc.tar.gz

### 2.2 What a twin sweep takes from them [src: SPEC 2.6 (e), (g); `noise_sweep_fit.py`]

For real cell $c$ and its sweep $s$ (protocol SS or LS), the twin sweep's noise law is (P0')

$$\tilde\theta(c,s) = \bigl(\hat\sigma_f(c,s),\ \tilde\sigma_m(c,s),\ \hat\tau_m(c),\ \tilde\sigma_s(c,s),\ \hat\tau_s(c)\bigr),$$

the sweep's own fast SD, its shrunk middle and slow SDs, and the cell's two time scales; for a failed sweep,
$\hat\theta(c)$ whole (D-122.3).

| Quantity | Per-sweep table (`noise_sweeps_L3_exc.csv`) | Per-cell table / JSON |
|---|---|---|
| keys | `specimen_id`, `noise_sweep_protocol` (`SS`/`LS`, upper case), `noise_sweep_number` | the JSON rows use `ss`/`ls` |
| sigma_f (mV) | `noise_sweep_sigma_f_mV` (unpooled) | `noise_sigma_f_mV` = the cell's |
| sigma_m (mV) | `noise_sweep_sigma_m_mV` = the twin's (shrunk); `noise_sweep_sigma_m_raw_mV` the sweep's own | `noise_sigma_m_mV` |
| tau_m | -- (the cell's) | `noise_tau_m_ms` (**ms**; `noise_law` takes s) |
| sigma_s (mV) | `noise_sweep_sigma_s_mV` = the twin's (shrunk); `noise_sweep_sigma_s_raw_mV` the sweep's own | `noise_sigma_s_mV` |
| tau_s | -- (the cell's) | `noise_tau_s_ms` (**ms**) |
| tilt $\hat b(c,s)$ (mV/s) | `noise_tilt_mV_per_s` (copied into the twin, D-102, D-107) | `cell_fit.ss_tilt_mV_per_s`, `cell_fit.ls_tilt_mV_per_s` |
| failure | `noise_sweep_flags` holds `sweep_fit_failed:<pr>:<sweep>:<reason>`; its `_mV` columns already carry $\hat\theta(c)$; the `raw`, `v` and `w` columns are empty | `sweep_stage.rows[].failed`, `.reason` |
| the whole twin law | -- | `sweep_stage.twin_laws["<pr>:<sweep>"]` = {`sigma_f_mV`, `sigma_m_mV`, `tau_m_s`, `sigma_s_mV`, `tau_s_s`} |

The simulator that matches what was fitted is `noise_law` (Block 2): `NoiseLaw(sigma_f, sigma_m, tau_m, sigma_s,
tau_s)` (mV, mV, s, mV, s), `draw_units(rng, n_traces, n_samples)` (draws fast, mid, slow, tilt, offset in that
order), `simulate_sweep(law, dt, n_samples, units, index=0, *, sigma_b=0.0, sigma_a=0.0, tilt_mV_per_s=None)` with
the tilt fixed at $\hat b(c,s)$ **[src: SPEC 2.6 (b), (h)]**.

### 2.3 The laws in numbers (nf2) [cluster; run]

| Parameter | Range over the 18 cells | Median |
|---|---|---|
| sigma_f | 0.0310-0.0349 mV | 0.0327 mV |
| sigma_m | 0.0161-0.0267 mV | 0.0197 mV |
| tau_m | 0.053-0.111 ms (2.7-5.5 samples at 50 kHz) | 0.077 ms |
| sigma_s | 0.0475-1.615 mV | 0.073 mV |
| tau_s | 11.2 ms - 5.66 s; above 265 ms in 526785799 (1.28 s), 548408551 (0.29 s), 571430815 (5.66 s), 571624532 (0.35 s) | 96 ms |
| J / dof (16 dof) | 1.41-58.6; not rejected at 0.1 % in 9 cells, at 1 % in 2 | 2.65 |

The per-cell table, the histories and the per-sweep fits: the Block 6 handoff §4 and
`Ih Fit/dev/2026-11-19_nf2_readback/` (the user's paste, `nf2_cells_from_paste_2026-11-19.csv`). The 9 failed LS
sweeps' reasons were not pasted (in the JSONs' `sweep_stage.rows[].reason`).

## 3. Decisions that bind Block 5 [log]

| ID | Binds | In short |
|---|---|---|
| D-005, D-011 | the six parameters; the cohort | $\theta = (C_m, R_m, R_a, \bar g_h, \Delta v_h, \kappa_\tau)$; real archive morphologies, $F$ = 1.9; a quarter of the cohort, every 4th cell by index, as false-positive control ($\bar g_h = 10^{-6}$ S/cm$^2$, $\Delta v_h = 0$, $\kappa_\tau = 1$); the error measure (D-011.1) and the gate on `ih6` |
| D-013 | the SWCs | every test on the raw SWCs until the diameter-corrected ones arrive |
| D-014 | acquisition twins | each twin sampled at its real cell's rates, with its real cell's SS pulse count per polarity (manifest `acq_*`) |
| D-017 | the twin's noise and LS timing | each twin carries its own real cell's fitted noise law (its form since superseded by D-152) and its real Long Square timing (`ProtocolConfig.ls_onset_ms`) |
| D-073, D-076 (i) | per-sweep twins | each twin sweep is the twin of one real sweep and carries noise calibrated on that sweep |
| D-076 (iii) | the LS set | the twin's LS sweeps are the real cell's, each at its own amplitude, one twin sweep per real sweep; `run_ih_recovery.py`'s cohort amplitudes only as a labelled fallback |
| D-076 (iv), D-072 | QC | no QC on the twins |
| D-083 | Stage 7 | the recovery runs on davinci before any six-parameter fit of a real cell; the twins carry their cell's morphology, fitted noise and protocol -- "its SS sweeps as 37-pulse trains at the stored positions" (qualified by D-180) -- and are fitted by `run_ih_recovery.py` through `run_ih_fit.main()`, four arms per twin (`baseline_runB`, `passive_fullstep`, `ih4`, `ih6`), the gate on `ih6` |
| D-102, D-107 | the tilt | each twin sweep carries the realised tilt $\hat b(c,s)$ of its own real sweep, fixed, not drawn (SS: D-102.2's reading; LS: the line through the pre-window and the end-window means) |
| D-122.3 | failed sweeps | the twin carries $\hat\theta(c)$ |
| D-152 to D-155 | the law | §1 |
| D-180 | the twins' span | the noise drawn over spans no longer than the fit's windows (an SS window, $[-10, 200)$ ms; the LS pre-window's 265 ms); nothing relies on the noise across pulses; open: the LS twins, the trains, the bundles, the offset |

## 4. What exists in the code, and what Block 5 changes

### 4.1 Inventory [src at `b17f768`]

| File | What it does today | What the decisions above ask of it [reasoning] |
|---|---|---|
| `synthetic_ground_truth.py` | builds the cell in NEURON as the fitter does (I_h through `extra_mechs`); `simulate_ss_protocol`: one clean response per polarity from rest, `ss_n_repeats` independent noisy copies averaged into a bundle, every pulse stored with its own sweep number; `simulate_ls_protocol`: one sweep per amplitude, onset 100 ms, 1 s step, 200 ms after, 20 kHz by default; `add_recording_noise`: white or AR(1) fast noise, a per-sweep DC offset and a linear drift -- **no slow OU, no middle part**; `write_archive_cell`: `ss_pulses.npz`, `ls_sweeps.npz`, `metadata.json` | the noise of each twin sweep from `noise_law` with $\tilde\theta(c,s)$ and $\hat b(c,s)$; the real LS timing and amplitudes; the SS windows as Q1-Q6 settle |
| `gen_from_manifest.py` | manifest row to kwargs; noise columns `noise_sigma_mV`, `noise_rho_lag1`, `noise_baseline_sigma_mV`, `noise_drift_sigma_mV`, `noise_seed`, `noise_ss_sigma_mV`, `noise_ss_rho_lag1`; acquisition `acq_ss_n_repeats`, `acq_fs_ss_Hz`, `acq_fs_ls_Hz` (D-014) | per-sweep laws do not fit per-cell columns (Q9) |
| `synth_gt_grid.py` | draws the manifest: truths, the false-positive control by index, noise from a noise table | its noise source becomes the nf2 tables |
| `run_ih_recovery.py` | phases noise (`noise_calibration`, the old per-sweep AR(1) table), manifest, generate (LS amplitudes `--ls-hyp-amps`/`--ls-dep-amps`, the cohort set), fit (`run_ih_fit.main()`, the arms), report (`ih_recovery_report`, the gate) | phase noise reads the nf2 tables; generate takes each cell's own LS amplitudes and timing |
| `noise_calibration.py` | the AR(1) noise measurement of D-012 to D-016 | superseded as the twins' noise source; only `lag_autocorr` is in the SPEC's coverage |
| `noise_law.py` | the three-component simulator and closed-form moments (Block 2) | the generator's noise |
| `run_ih_fit.py` | the fitter; arms `baseline_runB` (LS window 60 ms after onset), `passive_fullstep` and `ih6` (LS window mode `step`) | unchanged; it fixes what span of an LS twin is read (Q1) |
| `submit_ih_recovery.sh`, `smoke_ih_recovery.py` | Stage 7's PBS script (S7's marker lines); the smoke R1-R14 (14/14 on davinci, gate G10) | follow the changes |

### 4.2 Two facts the twins' noise has to respect [src]

- **The fit's simulator simulated each SS sweep as a continuous train** -- the windows cut at their real positions,
  the slow part and the tilt continuous across them (D-056 (i); SPEC section 8; Block 2's data flow) -- and every
  statistic was read after subtracting, per sweep and polarity, the group's median window, each window's tilt offset
  removed first (D-103.1, D-115). Twins with independent windows (D-180) carry the law within each window, but their
  median windows, and the mean windows the fitter averages into bundles, behave differently across windows (Q3, Q12).
- **The noise is made at 50 kHz.** All 18 cells are at 50 kHz **[run: the nf1 JSONs' `windows.fs_Hz`]**;
  $\hat\tau_m$ is 2.7-5.5 samples and the fast part is white per 20 us sample, so the noise must be drawn in the
  twin's own samples at that rate, never resampled; the generator's LS default of 20 kHz must not apply (D-014's
  `acq_fs_*` set the rates per cell).

## 5. Block 5's open questions (to settle with the user before the SPEC entry)

| # | Question | State, sources | The assistant's proposal [reasoning] |
|---|---|---|---|
| Q1 | **The Long Square twins under D-180.** D-076 (iii) gives one twin sweep per real LS sweep; on L3_exc a sweep has its pre-window $[0, 265)$ ms, the step from 270 ms for 1 s and 6 s stored after it (D-107); `ih6` and `passive_fullstep` read the step (`ls_window_mode='step'`), `baseline_runB` 60 ms after onset. The noise over the step lies beyond the fit's 265 ms | open (D-180 open point 1) **[log; src]** | keep the real geometry and draw the law over the whole sweep: an extrapolation in time, mild where $\hat\tau_s$ is at most 352 ms (16 cells) [corrected 2026-11-19: first written "$\le$ 350 ms (16 cells)"; 571624532 has 352 ms], strong for 526785799 and 571430815, which the report flags; read D-180 as binding the SS windows |
| Q2 | **The SS response.** D-083.2: 37-pulse trains at the stored positions (a train simulated in NEURON, the noise added per window), or each pulse simulated alone from rest as `simulate_ss_protocol` does | open (D-180 open point 2) | measure first: one cell, train against single pulse, the difference at each window against the noise SD; trains only if it matters |
| Q3 | **The SS bundles.** The fitter averages a polarity's windows (SPEC section 8); independent twin windows average the slow part down by $1/K$, the real ones perhaps not, so a twin's bundle may be cleaner than its real cell's | open (D-180 open point 3) | a model-free check: the correlation of window means across the real trains, read from the archive, and the twins' bundle residual noise against the real cells' (run B's residual noise at the MLE) |
| Q4 | **Which SS windows.** Every pulse of the 37-pulse train, or the stored (kept) windows per polarity and sweep | open since 2026-11-11 | the stored windows, the ones the fitter averages (D-014's pulse count) |
| Q5 | **The per-sweep offset $a$** (invisible to the noise statistics, visible to the I_h fit's absolute-mV gate) | open since 2026-11-11 | to discuss: none, or the real sweep's offset copied like the tilt |
| Q6 | **The tilt in SS windows.** D-102 copies $\hat b(c,s)$ for the whole sweep | settled in substance (D-102) | each window carries the tilt line at its real position, $\hat b(c,s)\,(t - \bar t_s)$: measured, not noise, so D-180.2 is kept |
| Q7 | **The sampling rate** | settled by D-014 | 50 kHz, the noise drawn at the twin's own rate (§4.2) |
| Q8 | **The LS timing.** D-017 asks for the real timing; the generator's defaults (onset 100 ms, 200 ms after) are not the archive's (270 ms, 6 s after), and no code reads them from the archive | D-017 **[log; src: grep]** | read onset, duration and length per cell from its `ls_sweeps.npz` |
| Q9 | **The manifest.** Per-sweep laws do not fit per-cell columns; the false-positive controls carry the same noise (D-011 (ii)); old manifests | N5's summary **[src]** | a per-sweep table beside the manifest (specimen, protocol, sweep, amplitude, law, tilt, failed flag), keyed to the manifest's rows; the old noise columns a labelled fallback |
| Q10 | **Seeds.** Per twin sweep, derived from (specimen, protocol, sweep), so adding a cell or a sweep moves no other draw (D-011 (ii)'s rule) | open | as stated |
| Q11 | **Stage 7's own points.** The cohort size (default 20; the L3_exc group has 18; L2_exc?), raw against corrected SWCs (D-013), the gate's tolerances (the assistant's proposal, unowned, D-011), Block 8's SPEC entry | open (D-083) | after Block 5 |
| Q12 | **Block 5's validation** | open | (a) closure: Block 3's statistics on a twin's archive against its real cell's $\hat q(c)$, within the twins' Monte Carlo spread (the independent windows change the median-window subtraction: measure by how much); (b) zero noise reproduces the clean response; (c) the laws in the manifest equal the nf2 tables; (d) the fitter loads a twin archive like a real one |

The option D-180 left aside -- a statistic across windows and the noise fit again -- remains the route if Q3's check
shows the bundles matter **[reasoning]**.

## 5a. The user's answers (2026-11-19, 21:23) [user; log]

| # | Answer | Decision |
|---|---|---|
| Q1 | option (a): the law as fitted, one continuous draw over the whole LS sweep; D-180's limit binds the SS windows; the four cells with $\hat\tau_s(c) > 265$ ms flagged, the gate reported with and without them | D-182 |
| Q2 | "the responses MUST be simulated given the real cell morphology. What is added from the data is the per sweep or window noise. This also holds for the long squared responses" -- the principle; **train or single pulses is not answered** and was asked again | D-183.1, D-183.2 |
| Q3 | not decided: the user asked what the model-free check is (explained in the chat: the correlation of consecutive real window means and the real bundles' residual noise against what independent windows give) | -- |
| Q4 | all stored SS windows and all stored LS sweeps | D-186.1 |
| Q5 | no offset | D-183.3 |
| Q6 | ok: the tilt line at the real positions | D-186.2 |
| Q7 | yes: 50 kHz, the twin's own rate | D-186.3 |
| Q8 | from the real cell | D-186.4 |
| Q9 | the per-sweep twin table beside the manifest | D-186.5 |
| Q10 | seeds per twin sweep | D-186.6 |
| Q11 | 10 cells per configuration, three configurations: `full` (traces with I_h, six parameters fitted), `passive fit` (traces with I_h, passive fit), `active fit` (traces without I_h, the full model fitted); one PBS job per cell, 120-core cap | D-184, D-185 |
| Q12 | no closure test: "this is simply a synthetic run to test whether our pipeline recovers the parameter values from a known truth ... LET's KEEP THINGS SIMPLE"; Block 5 keeps its smoke tests (directive 4) only [corrected 2026-11-19 (later): proposals (a) and "measure by how much" of Q12 are dropped] | D-184.4 |

**Settled on 2026-11-19, 21:46 [user; log].** Q2: each SS pulse simulated alone from rest, no train (D-187). Q3: the
check dropped (D-189). Stage 7: 10 twins per configuration on real cells drawn at random from the 18; `passive fit`
optimises only the passive properties on traces with I_h; `active fit` is D-011 (ii)'s control fitted with `ih6`; one
job per twin builds it and runs its fit, a merge job reports (D-188). Two readings of the assistant flagged in D-188:
the three configurations drawn independently (30 twins), and `passive_fullstep` as the passive arm. The plan below is
approved; the code is on the branch `sci/twins`; roadmap stages N5b (Block 10), N5c (Block 11), N5d (Block 12) added.
The inputs are in the chat of 2026-11-19 (the user's upload `OneDrive_1_19-11-2026.zip`: `nf2_L3_exc.tar.gz`,
`cell_508282493.tar.gz`), not in the repository.

**[corrected 2026-11-19 (later, 2): the list that follows was open at 21:23 and is settled as stated above.]** Still open, put to the user on 2026-11-19 (later) [open]. (1) Q2: the SS response as the 37-pulse train or each
pulse alone from rest (the assistant now leans to single pulses: today's code, the passive benchmark's way, no model
mismatch with the fitter's own simulation -- but (D-083.2) says trains, so it is the user's call). (2) D-184's readings:
the same 10 real cells in the three configurations, `full` and `passive fit` as two fits of one I_h twin (20 twins,
30 fits); which 10 of the 18; which passive arm (`passive_fullstep` or `baseline_runB`); `active fit` as D-011 (ii)'s
control; the note's "5 or 6" against 10. (3) D-185.2: each cell's job builds its twin and runs its fits, a merge job
writes the report. (4) Q3's check: with the pilot and recorded, or not at all. (5) The roadmap stages for the new blocks.

**The plan [approved by the user 2026-11-19, 21:46].** Blocks: 5 `twin_table.py` (the nf2 tables joined
with each real cell's protocol, one row per twin sweep); 10 `twin_noise.py` (independent SS windows plus the tilt line,
one continuous LS draw plus the tilt, the seed rule); 11 the clean responses in `synthetic_ground_truth.py` (real LS
amplitudes and timing; the legacy path bit for bit); 12 `twin_build.py` and `gen_from_manifest.py` (clean plus noise
into an archive in the real layout); 8 the Stage-7 run (the manifest's three configurations, `run_ih_recovery.py` per
cell and merge, `submit_ih_recovery.sh` as a PBS array). Wave 1: Blocks 5, 10, 11 in parallel; wave 2: Blocks 12 and 8;
the final check by an independent Opus tester.

## 5b. As built (2026-11-20) [src at `01ebd72` on `sci/twins`; run; log]

The plan of section 5a is implemented on the branch `sci/twins` (not yet merged to `main`: the user's call). Six
blocks, each written by its own implementer agent, gated and merged by the orchestrator:

| Block | Files | What it does | Smoke (sandbox) |
|---|---|---|---|
| 5 | `twin_table.py`, `twin_config.py` | the twins' table: per real cell and sweep, the law, the tilt, the protocol, and whether the sweep is in the twin | `smoke_twin_table.py` TT1-TT11 11/11 |
| 10 | `twin_noise.py` | a sweep's noise: independent draws per SS window, one continuous draw per LS sweep, the centred tilt line, the seeds | `smoke_twin_noise.py` TN1-TN9 9/9 |
| 11 | `synthetic_ground_truth.py` (two new functions) | the clean responses: one SS pulse per polarity from rest, each LS sweep at its real amplitude and timing | `smoke_twin_response.py` TR1-TR10 10/10 |
| 13 | `twin_manifest.py` | the manifest: 10 twins per configuration (`full`, `passive_fit`, `active_fit`), real cells drawn at random from the 18, the truths | `smoke_twin_manifest.py` TM1-TM10 10/10 |
| 12 | `twin_build.py` | one twin: the fitter's own cell at the truth, the responses, the noise, an archive the fitter's loader reads | `smoke_twin_build.py` TB1-TB9 9/9 |
| 8 | `run_twin_recovery.py`, `submit_twin_recovery.sh` | the Stage-7 run: `plan`, one job per twin (build, then fit), `merge` with the recovery report | `smoke_twin_recovery.py` TS1-TS10 10/10 |

**Decisions taken after section 5a [log].** D-190 (user): a twin leaves out the depolarising LS sweeps on which the
real cell spiked. D-191 [orchestrator]: the settle before a response, the SS current's samples, a tolerance, non-step
sweeps left out. D-192 (user): the twin's cell is the fitter's own model (the leak reversal balanced per segment at the
real cell's resting potential), so truth and fit share one model. D-193 [orchestrator]: the twin-fitter identity is
tested around the loader's Long Square step, which is one sample short (found, not changed). D-195 [orchestrator]: one
core per twin job (the fit is sequential). D-196 [orchestrator]: the final check's findings -- `merge` writes its
report when only the no-$I_h$ controls have results; the twins' gates on davinci are the six suites of step 1 below;
the noise's base seed is an option of the plan (`PLAN_ARGS=--base-seed=<int>`, default unchanged) and a twin is built
with the configuration recorded at plan time.

**The final check [run].** An Opus tester that wrote none of the code, on `577445d`, from the SPEC alone: PASS WITH
ISSUES (one MAJOR, one MINOR, one NOTE, all in Block 8, all fixed by D-196); the 16 suites and the 54 earlier spec
tests pass; 93 new spec tests in `tests/spec/`; one real `active_fit` twin of 508282493 built (427 s) and fitted.
A second tester on `216ecf1`, on the changed block: PASS WITH ISSUES, nothing major (`tests/spec` 145 passed); its two
points (the merge under `--drop-railed`, a negative base seed) are fixed by D-197 and verified by its own reproducers
(Block 8's spec tests 12/12). No third independent run was made.

**Measured here [run].** One real `full` twin of cell 508282493 built in 1547 s on one core (about 105 s per 7.27 s
sweep on its 5247 sections); 15 of its 22 LS sweeps are in the twin, 7 left out (`action_potential`). The fit's time on
a real twin was not measured here: the pilot gives it.

**On davinci.** One line each, pasted at the prompt. The branch is checked out in the cluster clone (the user, 2026-11-20
16:18: `01ebd72`; the clone was on `main` at `7c890ff` before). Every command after step 1 runs in the "Ih Fit" folder
with `prova` active.

0. the branch (done): `cd /davinci-1/home/ldellamea/TEEG/Towards-EEG && git fetch origin sci/twins && git checkout sci/twins && git pull && git log --oneline -1`
   (expect `a28d1a5` or later).
1. the gates, from the repository root: `conda activate prova && cd "Passive Features/HPC script/Ih Fit" && python smoke_twin_table.py && python smoke_twin_noise.py && python smoke_twin_manifest.py && python smoke_twin_response.py --build && python smoke_twin_build.py && python smoke_twin_recovery.py`
   -- each suite ends `-- N pass, 0 fail, 0 error ...` (10, 9, 10, 10, 9 and 10 pass, with skips); about 5 min.
2. the dry run: `RUN_TAG=tw1 NOISE_TAG=nf2 DRY_RUN=1 bash submit_twin_recovery.sh` -- must print `plan: cohort of 18 real
   cell(s)`, `plan: 30 twin(s) (full 10, passive_fit 10, active_fit 10)`, `cores: 30 twin(s) x NCPUS 1 = 30 (cap 120, D-185)`,
   the CLI-contract line, an array `qsub` with `select=1:ncpus=1` and `-J 0-29`, a merge `qsub` with
   `depend=afterany:ARRAY_JOB_ID`, and `DRY RUN: nothing written, nothing submitted`.
3. the pilot, one twin per configuration, with a 48 h ceiling so that it is not killed before it gives its times
   (the noise fit's pilot had the same, D-147): `RUN_TAG=tw1 NOISE_TAG=nf2 TWINS=900000000,900000100,900000200 WALLTIME=48:00:00 bash submit_twin_recovery.sh`
   -- plans the run for real, then `twins: 3 of 30`, `cores: 3 twin(s) x NCPUS 1 = 3`, the two `submitted:` lines. Each
   twin's log `logs/twin_<id>_<job>.log` ends `[twin_recovery] TWIN <id>: status=ok ...`; the merge writes `report/`.
   [2026-11-20, replaces "pilot of two twins, `MAX_TWINS=2`": the first two twins of the manifest are both `full`, so the
   passive and the no-I_h paths would not have run on real data before the 30.]
4. the pilot's times and a wall time for the rest (2x the slowest pilot twin, at least 24 h: a proposal, the user's
   call): `cd "/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs_ih/twin_recovery/tw1/L3_exc" && python -c "import math, pandas as pd; d=pd.read_csv('twins_status.csv'); d=d[d.status!='missing'].copy(); d['hours']=(d.build_seconds+d.fit_seconds)/3600; print(d[['twin_id','configuration','status','build_seconds','fit_seconds','hours']].round(2).to_string(index=False)); print('suggested WALLTIME=%d:00:00' % max(24, math.ceil(2*d.hours.max())))"`
5. the 30 twins, from the "Ih Fit" folder after the pilot has ended: `RUN_TAG=tw1 NOISE_TAG=nf2 WALLTIME=<from step 4> bash submit_twin_recovery.sh`
   (the three pilot twins are skipped).

**As run [D-200, user, 2026-11-20 17:08: no pilot].** Steps 3 and 4 are replaced by one submission of the 30 twins, from
the repository root: `cd /davinci-1/home/ldellamea/TEEG/Towards-EEG && git pull && git log --oneline -1 && cd "Passive Features/HPC script/Ih Fit" && conda activate prova && python smoke_twin_recovery.py && RUN_TAG=tw1 NOISE_TAG=nf2 WALLTIME=48:00:00 bash submit_twin_recovery.sh`.
The early check, about an hour after the jobs start (no twin ended, every log in the fit, no traceback):
`cd "/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs_ih/twin_recovery/tw1/L3_exc" && echo "twins ended: $(ls status 2>/dev/null | wc -l)   in the fit: $(grep -l 'n_calls=200' logs/twin_*.log 2>/dev/null | wc -l)   with a traceback: $(grep -l Traceback logs/twin_*.log 2>/dev/null | wc -l)"`.
**Ended [the user's paste, 2026-11-22 13:52]:** the early check read `twins ended: 30   in the fit: 30   with a
traceback: 0` -- every twin wrote its status file after reaching the fit; whether each is `ok`, and the merge's
report, are read next (section 7). The job IDs were not recorded.

`RUN_TAG` and `PLAN_ARGS` stay the same in steps 2 to 5: the plan is frozen at the pilot, and a different option is
refused. One R_a truth per twin instead of one per configuration is `PLAN_ARGS=--ra-mode=per_cell` added to all of them.
Expected cost [estimate, not measured at full size]: in the sandbox the `active_fit` twin of 508282493 (5245 segments)
spent about 16 min on the fit's fixed steps and about 4 min on one optimiser call with its validation; the fit makes
200 calls (`--n-calls 200 --n-initial 100`, `run_ih_recovery.py`'s defaults) -- hours per twin, not minutes.

Outputs: `$IH_OUTPUT_BASE/twin_recovery/<RUN_TAG>/L3_exc/` -- `manifest.csv`, `twins/`, `fits/`, `status/`, `logs/`,
`report/` (all twins), `report_without_flagged/` (without the twins of the four cells of D-182.3),
`recovery_by_configuration.csv`, `twins_status.csv`.

**Open, the user's [reported 2026-11-20].** (1) With the existing drawer's default the 10 twins of a configuration share
one $R_a$ truth; one per twin is `PLAN_ARGS=--ra-mode=per_cell` at the first submission. (2) Whether the fitter's loader is corrected for the
one-sample-short Long Square step (D-193). (3) The roadmap has no stage for Block 13 (the manifest). (4) Two readings in
D-188: the cells are drawn independently per configuration (30 twins), and the passive arm is `passive_fullstep`.
(5) The gate's tolerances are still the unowned defaults of D-011. (6) Raw or corrected SWCs for the twins (D-013).

## 6. Limits of the law the twins inherit [cluster; run]

- **Misfit.** J/dof median 2.65 (worst 569844159, 58.6; 571483807, 11.0); the per-sweep laws rejected at 1 % in 19 %
  of SS and 23 % of LS sweeps.
- **The protocols disagree in some cells** (one law per cell cannot meet both): in 508282493 the law is below every
  SS statistic and above every LS one; in 569844159 the SDs miss by about 4 SD in opposite directions (the pilot's
  read-back). The per-sweep laws, pooled per protocol, absorb the sizes.
- **The slow ridge** (§1): $\sigma_s$ and $\tau_s$ apart are not determined where $\tau_s$ is long; within the fit's
  windows the laws are equivalent.
- **No plateau** in 15 of 18 cells (D-050: the last iterate kept; the late steps go back and forth).
- **9 of 525 LS sweeps failed**; their twins carry $\hat\theta(c)$.
- **$\hat\tau_m$ is 0.05-0.11 ms in every cell**: whether it belongs to the cells or to the recording chain is not
  studied **[open]**.

## 7. After Block 5: Stage 7 and the record

- **Block 8** (the gate run; roadmap S7) is built [corrected 2026-11-20: the draft around `submit_ih_recovery.sh` and
  `MAX_CELLS=2` is superseded]: the steps are section 5b's -- pilot of one twin per configuration, then the 30 twins.
- **Reading the run** [src at `a28d1a5`]: `twins_status.csv` (one row per twin; 30 `ok` expected), and in `report/`
  and `report_without_flagged/` (without the twins of the four cells of D-182.3): `gate_verdict.txt`
  (`STAGE 7 GATE: PASS|FAIL` and the tolerances), `gate_verdict.csv` (per axis: n, median $|\varepsilon_i|$, tolerance,
  verdict), `recovery_summary.csv`, `recovery_per_cell.csv`, `cm_inflation.csv` (each arm's $C_m$ bias on the twins with
  I_h: the `passive_fit` question) and `fp_control.csv` (the `active_fit` twins: $\bar g_h$ on its lower bound). The
  gate is on the `ih6` arm; $R_m$ and $R_a$ are reported, not gated.
- **The gate** (D-011, D-083, D-005): median $|\varepsilon_i|$ per axis against tolerances still to be owned by the user
  (`run_ih_recovery.py`'s defaults: $C_m$ and $\bar g_h$ within x1.25, $\Delta v_h$ within 3 mV, $\kappa_\tau$ within x1.5;
  control: at least 80 % of the no-I_h twins with $\bar g_h$ on its lower bound and $C_m$ within x1.25); they are plan
  options, frozen when the pilot is submitted. A failing kinetic knob ($\Delta v_h$, $\kappa_\tau$) is frozen at its base
  value for the campaign; a failing $C_m$ or $\bar g_h$ means a different protocol, not a smaller fit
  (`ih_recovery_report.frozen_axes_on_failure`).
- **Result of run tw1 (2026-11-22) [data inspected: the run's tables and logs, the user's extract;
  `dev/2026-11-22_tw1_readout/` reproduces every number].** PBS array `2101455[]` (30 twins) and merge `2101456`;
  30 `ok`; build + fit median 11.0 h, max 29.0 h (`full`; `passive_fit` at most 7.2 h). **STAGE 7 GATE: FAIL**, in
  `report/` and in `report_without_flagged/` (7 twins on the four cells of D-182.3 left out). All 30, `ih6` arm,
  median $|\varepsilon|$: $C_m$ 0.126 (pass), $\bar g_h$ 1.84, $\Delta v_h$ 7.47 mV, $\kappa_\tau$ 0.61 (all FAIL);
  control: $\bar g_h$ on its lower bound in 3 of 10 (FAIL), its $C_m$ 0.49 (FAIL); $R_m$ 0.086, $R_a$ 0.207 not
  gated. Without the flagged twins $C_m$ fails too (0.239).
- **Reading [checked by computation; reasoning; Rich et al. 2021 full text in project knowledge: Eq. 1, Table 1].**
  (1) The training steps reach a median 10.6 mV below rest (to about -70 mV); there `Ih_human` ($v_h$ = -90.87 mV,
  $k$ = 8.05 mV) is in its exponential foot: a shift of $\Delta v_h$ by +-5 mV is undone by scaling $\bar g_h$ by
  $e^{\mp 5/k}$ to within 3.5-6 % in $g_h(V)$ over rest to 11 mV below (35-40 % over -95 to -60 mV), so $\bar g_h$ and
  $\Delta v_h$ are not separately identifiable from these steps. (2) The truths' I_h is small there: across the 20
  twins with I_h, median 1.3 % of the leak at rest and 4.6 % at the deepest training trough. (3) In the 3 `full` twins
  whose I_h is at least 0.3x the leak at that trough, $g_h$ at rest comes back within x1.6 while $\Delta v_h$ runs to its
  bound and $\bar g_h$ compensates; in the other 7 nothing about I_h comes back. (4) $\kappa_\tau$ ends on a bound in
  8 of 10. (5) `active_fit`: the fitted I_h at rest is at most 1.8 % of the leak in all 10, but $C_m$ is off by more
  than x1.25 in 7 (on a bound in 6). (6) **Open:** the absolute training residual is a median 1.9x the noise sigma;
  whether the optimiser (200 calls in 6-D) reached the best fit is not established -- it needs the objective at the
  truth. The real cell 508282493 shows a sag of 0.10-0.12 on its -90 and -110 pA steps (troughs -70 and -72 mV).
  By D-005 and D-083 a failing $\bar g_h$ makes the six-parameter fit unusable as it is; the next step is the user's.
- **Next [D-204, user, 2026-11-22 14:24].** The user does not accept "not recoverable" and changes the fit's
  protocol before a second Stage-7 run: train on the stronger hyperpolarising long squares (a medium one validates);
  weight a step by its strength; weight each long square's samples to peak at the response's peak and decay slowly;
  add 5 twins fitted without noise. The exact forms are proposed and await the user's confirmation; nothing is
  implemented. Refinement of the reading above [data inspected; one-compartment estimate]: of the 10 `full` twins,
  4 (-000, -005, -008, -009) have a sag far above the noise yet a wrong I_h conductance (the fit's miss), 3 (-003,
  -004, -006) a sag at or below the noise, 3 (-001, -002, -007) a conductance at rest within x1.6. Including the
  strongest step deepens the reach from a median 10.6 to 13.0 mV below rest; $\bar g_h$ and $\Delta v_h$ stay tied.
- **In progress [2026-11-22 15:10, the user's go].** The forms are settled (D-206 to D-209): $\Delta v_h$ fixed at 0
  (Rich et al. 2021), five fitted parameters, $\bar g_h$ up to $10^{-2}$ S/cm$^2$, the I_h time constant stated as
  its peak in ms (box 50-1000 ms; truths 150-700 ms), roles `d207`, step weight by current, the peak-decay sample
  weight (500 ms), 5 noise-free twins paired with the first five `full` twins, the loss at the truth recorded.
  SPEC section 2.8 and Blocks 14, 15, 16, 18 (wave 3a, implementers running) and 17 (wave 3b) on `sci/twins`
  (contract `aac46d3`). Also delivered: `dev/2026-11-22_tw1_readout/loss_at_truth.py` with
  `job_loss_at_truth.sh` (`83111d8`) -- on davinci `qsub dev/2026-11-22_tw1_readout/job_loss_at_truth.sh` from the
  "Ih Fit" folder; it tells, per `full` twin of `tw1`, whether the truth fits better than the estimate (the
  optimiser's miss) or not. Its result was not yet returned when this was written.
- **Built [2026-11-22 17:40; src at `c974a62` on `sci/twins`; run].** SPEC 2.8 and Blocks 14-18 are implemented,
  gated, merged and checked by an independent tester (PASS WITH ISSUES, two NOTEs, fixed; `tests/spec` 255 passed).
  The second run is `tw2`: 35 twins (10 `full`, 10 `passive_fit`, 10 `active_fit`, 5 `full_no_noise` with ids
  900000300-304 paired with 900000000-004), arms `ih5` / `passive_fullstep`. On davinci, from the repository root:
  gates `conda activate prova && git pull && cd "Passive Features/HPC script/Ih Fit" && python smoke_ls_protocol.py && python smoke_ih5.py && python smoke_twin_manifest.py && python smoke_twin_build.py && python smoke_twin_recovery.py`;
  dry run `RUN_TAG=tw2 NOISE_TAG=nf2 DRY_RUN=1 bash submit_twin_recovery.sh` (expect `-J 0-34`); pilot
  `RUN_TAG=tw2 NOISE_TAG=nf2 TWINS=900000000,900000100,900000200,900000300 WALLTIME=48:00:00 bash submit_twin_recovery.sh`;
  then the same line without `TWINS=`. Read-out: `/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs_ih/twin_recovery/tw2/L3_exc/recovery_extras.csv` (loss at truth vs
  estimate, `loss_verdict`, $\tau_h^{\rm pk}$ in ms, $g_h$ at the deepest trained voltage), `report/`,
  `report_without_flagged/`, `report_no_noise/`. Real-cell check here: the new loss on a real twin of 508282493
  is 0.0109 at the truth, 0.042 with $\bar g_h$ x3, 0.018 with $\bar g_h$ /3.
- **Result of run tw2 (2026-11-24) [data inspected: the run's tables, the user's extract].** 35 twins `ok` (build +
  fit median 10.7 h, max 25.5 h). **STAGE 7 GATE: FAIL** in `report/`, `report_without_flagged/` and
  `report_no_noise/`. `ih5`, the 10 `full` twins, median $|\varepsilon|$: $C_m$ 0.364 (FAIL; 0.126 in `tw1`),
  $\bar g_h$ 0.883 (1.84 in `tw1`), $\kappa_\tau$ 1.124, $R_m$ 0.246, $R_a$ 0.727; control: $\bar g_h$ on its lower
  bound in 1 of 10. The 5 noise-free twins: $C_m$ 0.331, $\bar g_h$ 0.293, $\kappa_\tau$ 0.980. **The cause is the
  search [recovery_extras.csv]:** the loss at the truth is below the loss at the estimate in 9 of 10 `full` twins
  (estimate/truth median 1.30) and in 5 of 5 noise-free twins (median 7.1, range 2.0-10.9: truth 0.001-0.002,
  estimate 0.004-0.019). $C_m$ ends on its lower bound in 6 of 10 `full` twins and $\kappa_\tau$ on a bound in 5.
  So the 200-call Bayesian search (100 random) does not reach the minimum of the five-parameter loss; noise is not
  the limit. Next: the optimiser (the user's decision; proposals in the chat of 2026-11-24).
- **Block 7** (roadmap N7): the fitted laws and flags into the Stage-7 manifest and the noise table, and the
  decision-log and plan entries that close the noise prerequisite (N7's summary; the S0-S3 report dropped, D-125.3).
- **Stage 8** (the campaign) waits for the gate's verdict and the corrected SWCs (S8's note).
- **After tw2: the optimiser changed (2026-11-24) [user: D-234; orchestrator: D-235, D-237; src at `257b351` on
  `sci/twins`; run].** SPEC 2.9 and Blocks 19-22 are implemented, gated, merged and checked by an independent tester
  (PASS WITH ISSUES: one MINOR, six spec-text points, all settled; report
  `specs/test_reports/2026-11-24_1510_incremental.md`; `tests/spec` 361 passed, 2 skipped in the tester's run).
  What changed: (1) after the 200-evaluation search, bounded Nelder-Mead from the 3 lowest-loss points at least 0.1
  apart in unit coordinates, at most 100 evaluations each (`local_refine.py`; `--refine`, `--refine-starts`,
  `--refine-maxfev`); (2) $\bar g_h \in [10^{-5}, 10^{-3}]$ S/cm$^2$, control twins at $10^{-5}$; (3) the arm
  `ih6_d207`: $\Delta v_h$ fitted again under the loss of D-207, truths in $\pm 5$ mV; the run's arm is the plan
  option `--ih-arm` (default `ih6_d207`; `ih5` keeps $\Delta v_h = 0$); (4) the sweeps of one loss evaluation in
  `--n-workers` forked processes, the loss equal to the serial one exactly; `NCPUS` default 3, linear-algebra threads
  at 1. New columns in `phase2_results.csv` and `recovery_extras.csv`: `loss_after_search`, `loss_after_refine`,
  `refine_improved`, `refine_n_evals`, `refine_best_start`, `n_workers`, `dvh_true_mV`, `dvh_est_mV`; per start,
  `refine_starts.csv`. **Measured here:** on a 5-parameter quadratic, 100 evaluations per start stop on the cap at a
  loss ratio of 6.2e-5; uncapped, 140-190 evaluations converge to 1.1e-6 -- if the test below stops short, raise
  `--refine-maxfev`. **Not verified here:** the speed-up beyond 2 cores; PBS.
  **Next on davinci: the test on the 5 noise-free twins, five and six parameters (10 jobs, 80 cores).** Gates:
  `conda activate prova && git pull && cd "Passive Features/HPC script/Ih Fit" && python smoke_ls_pool.py && python smoke_local_refine.py && python smoke_ih_refine.py && python smoke_twin_manifest.py && python smoke_twin_recovery.py`;
  then `RUN_TAG=tw3a NOISE_TAG=nf2 NCPUS=8 PLAN_ARGS="--ih-arm ih5" TWINS=900000300,900000301,900000302,900000303,900000304 WALLTIME=48:00:00 bash submit_twin_recovery.sh`
  and `RUN_TAG=tw3b NOISE_TAG=nf2 NCPUS=8 TWINS=900000300,900000301,900000302,900000303,900000304 WALLTIME=48:00:00 bash submit_twin_recovery.sh`
  (add `DRY_RUN=1` first; expect `-J 0-4` and `--n-workers 8`). Pass condition per twin, in `recovery_extras.csv`:
  `loss_at_estimate` down to `loss_at_truth` (`loss_verdict` not `optimiser`). A run planned before Block 22 (tw1,
  tw2) is merged only with the code of its own commit. The study document of roadmap topic T18 was started by an
  agent and stopped by the user before it was written.
- **Result of the test runs tw3a and tw3b (2026-11-25) [data inspected: the runs' tables, the user's extract
  `tw3_results.tar.gz`; code `257b351`].** The 5 noise-free twins (900000300-304), 8 workers, 200 search + 300 local
  evaluations each; all 10 `ok`. **Pass condition not met in any twin** (`loss_verdict = optimiser` in all 10), but
  the gap closed a great deal. Ratio loss at estimate / loss at truth, twins 300 to 304: `tw2` (search only, five
  parameters) 10.9, 10.9, 7.1, 5.7, 2.0; `tw3a` (five parameters, refined) 2.84, 1.10, 1.13, 2.41, 1.48 (median
  1.48; after the search alone, median 5.1); `tw3b` (six parameters, refined) 3.64, 1.95, 4.02, 2.38, 1.63 (median
  2.38; search alone 11.0). **Every one of the 30 Nelder-Mead starts stopped on the 100-evaluation cap** (`status 1`
  in `refine_starts.csv`), each having lowered its loss: the budget is the limit, as the 5-parameter quadratic
  measured here predicted. The estimate came from the start of lowest initial loss in only 4 of 10 fits. Parameters:
  $\ln(\hat{\bar g}_h/\bar g_h)$ in `tw3a` $-0.40, -0.03, -0.03, -0.67, -2.37$ (2 of 5 inside the gate's 0.223; twin
  304 on the floor $10^{-5}$ with a loss ratio of only 1.48 -- its deepest trained voltage is $-59.7$ mV, where I_h
  is nearly closed); in `tw3b` $-0.70, +0.02, -1.16, -0.21, +0.13$, with $\Delta v_h$ errors $+0.9, +0.1, +11.0,
  -6.6, -7.0$ mV (2 of 5 inside 3 mV; twin 302 shows the trade: $\bar g_h$ a third of the truth, $\Delta v_h$ 11 mV
  too high). Time: 29 to 102 s per evaluation against 98 to 401 s in `tw2` on one core, a factor of 3.0 to 4.3 from
  8 workers (approximate: total fit time over evaluations); fits of 3.9 to 14.1 h. Open, the user's: the local
  budget (`--refine-maxfev`), and five against six parameters.
- **After tw3: the user's decisions and the pilot tw4 (2026-11-25) [user: D-244; src at `393b681` on `sci/twins`].**
  Five parameters kept (`--ih-arm ih5`; the code's default arm is still `ih6_d207`); a larger local budget; cells
  whose training steps do not open I_h are to be removed from the I_h fit (rule NOT set: candidates a threshold on
  the deepest trained voltage $v^\ast$ or on $m_\infty(v^\ast)$; in tw3a the three badly recovered twins are the
  three shallowest, $v^\ast$ of $-66.0$, $-66.8$, $-59.7$ mV); the submitter's cap is 240 cores. Pilot, from the
  code folder on davinci:
  `RUN_TAG=tw4 NOISE_TAG=nf2 NCPUS=24 PLAN_ARGS="--ih-arm ih5 --refine-maxfev 300" TWINS=900000000,900000001,900000002,900000003,900000004,900000005,900000006,900000007,900000008,900000009 WALLTIME=48:00:00 bash submit_twin_recovery.sh`
  (the 10 noisy `full` twins [corrected 2026-11-25, 12:08: the user, "All noisy"; first written as 5 noise-free and 5 noisy]; about 1,100 evaluations per fit; told to the user: 24 cores per
  cell are unlikely to beat 8, a cell trains on 3 to 4 long steps). Read first, per twin: `status` in
  `refine_starts.csv` (0 = converged), then `loss_at_estimate / loss_at_truth`. Not investigated: why the loss at
  the truth of a noise-free twin is not 0 (8e-4 to 2.2e-3). A study document for an independent evaluation of the
  whole approach was sent to the user as files in this chat:
  `TEEG_topic_Ih_fit_and_optimisation_2026-11-25.md` (with `figures/`, `scripts/`); it is not in the repository.
- **After the independent review (2026-11-25, 13:17) [user: D-245; src at `393b681`; data inspected:
  `tw3_results.tar.gz`; run in the review chat].** The local stage becomes scipy `least_squares` (TRF, bounded) on
  the squared loss $L_2(q) = \sum_b c_b (\mathrm{RMSD}_b(q)/D_b)^2$ (D-245; not implemented); the review's
  sensitivity module (its CR-2) is not built [corrected 2026-11-25, 14:08: built after all, D-246, to settle the exclusion rule and the sample weight's $\tau_d$]. The review's files (`TEEG_tech_Ih_fit_change_requests_2026-11-25.md`,
  `smoke_sensitivity_reference.py`, `check02_nm_vs_lsq_anisotropic.py`, `check03_nm_restart.py`) were sent to the
  user as files and are not in the repository; its three scripts re-run in the review chat reproduce its numbers.
  *Code facts the review asked for [src]:* (a) `ih_mechanism.balance_e_pas` resets each segment's `e_pas` at every
  $q$, so $V_{\rm rest}$ is stationary; (b) `vshift_minf` in `mod/Ih_human.mod` shifts $m_\infty$ only, not the time
  constant; (c) fixed step, `cnexp`, float64, `settle_ms` 0; **[corrected 2026-11-25, about 15:30: first written "0.025 ms in the builder and in the fit ... the non-zero loss at a noise-free twin's truth is not a time-step mismatch", read from the module's defaults; wrong]** the twins are built at 0.025 ms (`TwinConfig.neuron_dt_ms`) and every twins' fit so far ran at 0.1 ms (`--dt-brief-ms 0.1 --dt-long-ms 0.1` in `fit_argv.json` of tw3a; `run_ih_fit.py`'s default, D-007), as D-192 (iv) states; with the loader's one-sample-short long step (D-193) this IS the non-zero loss at a noise-free twin's truth. Measured on the fixture of `smoke_ih_trf.py` (Block 27's implementer): at 0.1 ms a converged least-squares fit ends 2.1e-2 (unit box, max-norm) from the truth at an $L_2$ below the truth's (3.1e-6 against 6.7e-6); at 0.025 ms it ends 2.2e-6 from the truth; (d) `gp_minimize` receives the raw loss, the $10^6$ penalty included; (f) `fit_one_cell` gets
  `seed=i`, the cell's index in its run, so every twin's search has seed 0 and the same initial design; (g) $F$
  scales `cm` and `g_pas` beyond the proximal cutoff, not `gIhbar`; (h) a long bundle holds the whole sweep and
  `--ls-window sweep` exists, the rebound kept out of training by D-006 (ii).
  *Re-read of `refine_starts.csv` of tw3a and tw3b [data inspected]:* the three endpoints of one fit are far apart
  in 9 of 10 fits (largest pairwise max-norm distance in the unit box 0.53 to 0.99; 0.12 in twin 900000302 of
  tw3a), mostly along $\tau_h^{\rm pk}$, with losses within a factor of about 2 in the shallow twins; estimates on a
  bound that the report's rail flag does not mark: $\tau_h^{\rm pk}$ of twin 900000303 at the lower bound in both
  runs (50.2 ms against a truth of 664 ms in tw3a; 51.3 against 243 in tw3b); endpoints that are not the estimate
  also end on bounds ($\tau_h^{\rm pk}$ at its upper bound in twin 900000300; $C_m$ and $R_a$ at their upper bounds
  in two starts of twin 900000303, tw3a). In tw3b, $\ln(\hat{\bar g}_h/\bar g_h) + (\widehat{\Delta v_h} - \Delta v_h)/k$
  is $-0.59, +0.03, +0.21, -1.04, -0.74$: zero under the $\bar g_h$-$\Delta v_h$ trade-off, so twins 900000301 and
  900000302 follow it and the three shallow twins do not. Not in the archive: the search's 200 evaluations (the
  count of penalty evaluations), any noisy twin.
- **Blocks 23 to 28 built (2026-11-25) [src: `sci/twins` at `de4cc53`; run: the smoke suites and the final check
  on the merged branch; log: D-245 to D-249 and D-262 [corrected 2026-11-27: D-262 was logged as D-250]].** The fit's refinement is `trf` by default (scipy `least_squares` on
  the residual vector, `--refine-max-nfev 30`, 3 starts; `--refine nelder_mead` a comparison); `phase2_results.csv`
  and `recovery_extras.csv` carry $L_2$, the converged starts, the failed points and `on_bound_<name>`; the twins'
  verdict is on $\sqrt{L_2}$. New: `run_twin_recovery.py sensitivity` and `MODE=sensitivity` of the submitter (the
  sensitivities at a twin's truth, the singular values, the predicted scatter of each parameter for the current
  weighting `W0` and for no decay `W1`, and the truth trace check). Final check: PASS WITH ISSUES
  (`specs/test_reports/2026-11-25_1700_incremental.md`); its 7 new spec test files are NOT committed and 8 earlier
  spec tests are stale (the old default, the old column lists): both put to the user. Next on davinci, the noisy
  twins only, at the fit's usual 0.1 ms (D-248), from the code folder after `git pull` on `sci/twins`, the gates
  first (the header of `submit_twin_recovery.sh`, plus `python smoke_residuals.py && python smoke_fd_jacobian.py &&
  python smoke_refine_trf.py && python smoke_identifiability.py && python smoke_ih_trf.py && python
  smoke_twin_sensitivity.py`), each line first with `DRY_RUN=1`:
  `MODE=sensitivity RUN_TAG=tw5 NOISE_TAG=nf2 NCPUS=24 PLAN_ARGS="--ih-arm ih5" TWINS=900000000,900000001,900000002,900000003,900000004,900000005,900000006,900000007,900000008,900000009 WALLTIME=04:00:00 bash submit_twin_recovery.sh`
  then
  `RUN_TAG=tw5 NOISE_TAG=nf2 NCPUS=24 PLAN_ARGS="--ih-arm ih5" TWINS=900000000,900000001,900000002,900000003,900000004,900000005,900000006,900000007,900000008,900000009 WALLTIME=48:00:00 bash submit_twin_recovery.sh`
  (the walltimes are the assistant's guesses, not measured). Read: `identifiability.csv` (`sd_lin_gbar` per twin
  and weighting: the exclusion rule of D-244 item 3 and the $\tau_d$ question of D-246 are decided on it),
  `truth_trace_check.csv`, then per fit `refine_n_converged`, the endpoints' agreement in `refine_starts.csv`
  (`x_solver_*`, `sv_*`), `on_bound_*`, and the errors against `sd_lin_*`. Noise-free twins later, if time allows,
  with `--dt-long-ms 0.025` in `PLAN_ARGS` (D-248). Not built: the review's CR-5 (the global stage).
- **The sensitivity run `tw5` on the ten noisy twins (2026-11-25/27) [data inspected: `tw5_sensitivity_1.tar.gz`,
  `identifiability.csv`, 20 rows, K = 200; code `718c778` after amendment 4, D-260].** Predicted standard deviation
  of $\ln \bar g_h$ at the truth under the twin's own noise, current weighting (`W0`, $\tau_d = 500$ ms), by twin in
  order of $m_\infty(v^\ast)$: 900000001 (0.58) 0.55; 900000009 (0.36) 0.14; 900000007 (0.23) 0.012; 900000002
  (0.17) 0.019; 900000008 (0.13) 1.22; 900000005 (0.10) 0.43; 900000003 (0.05) 2.5; 900000000 (0.05) 0.24;
  900000006 (0.03) 21.5; 900000004 (0.02) 9.4. So: (i) depth and predicted scatter are related (Spearman -0.65 on
  ten twins) but depth alone misorders them (900000001 the deepest and the noisiest, `loss_at_theta` 0.020;
  900000000 shallow yet 0.24; 900000008 at 0.13 yet 1.2); (ii) $\ln \kappa_\tau$ is the least-determined direction
  in 9 of 10 twins (predicted sd 0.025 to 32); (iii) the passive parameters are predicted within 0.15 in all ten;
  (iv) without decay (`W1`) the predicted sd of $\ln \bar g_h$ is smaller in 4 of 10 twins, geometric-mean ratio
  W1/W0 1.06 ($\ln \kappa_\tau$: 1.00): no gain from removing the decay; (v) `fd_step_check` at most 8e-4. A
  predicted sd above about 1 is outside the linear regime and reads as "not determined", not as a number.
  **[reasoning]** a perfect optimiser would not pass the gate of D-011 on these ten twins as a group: with
  median absolute error about 0.67 sd per twin, the group's median is about 0.33 for $\ln \bar g_h$ (tolerance
  0.223) and about 0.95 for $\ln \kappa_\tau$ (0.405). Decided on these numbers [user, D-266, 2026-11-27]: $\tau_d = 500$ ms is kept; no exclusion rule is
  implemented. The fit run on the same twins (array 2104213) was queued on
  2026-11-26.

## 8. Repository, environment, cluster

| Item | Value |
|---|---|
| repository | `Leonardodm00/Towards-EEG` (public), branch `main` |
| push from a new session | `add_repo` (owner Leonardodm00, repo Towards-EEG, access push) first, or the proxy refuses the push |
| a light clone | `git clone --filter=blob:none --sparse https://github.com/Leonardodm00/Towards-EEG.git && cd Towards-EEG && git sparse-checkout set "Passive Features/HPC script/Ih Fit" docs/project specs/roadmap` |
| commit identity | the user's: `git config user.name "$(git log -1 --format=%an)"`, the same for the address (`%ae`); never write the address into a file |
| cluster clone, env | `/davinci-1/home/ldellamea/TEEG/Towards-EEG`; conda env `prova` (python 3.12.13, numpy 2.4.4, scipy 1.17.1, statsmodels 0.14.6, neuron 9.0.1) |
| archive | `/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc/specimen_*` (18 cells; 14 with two SS sweeps; SS trains of 37 pulses 200.5 ms apart, windows stored $[-10, 200)$ ms at 50 kHz) |
| the noise-fit gates | `bash cluster_gates_noise_fit.sh` (FULL: 16 checks incl. `smoke_ih_recovery` as G10 and the regression G11; last SETTLED 2026-11-17 at `7c890ff`) |
| the read-back one-liners | the Block 6 handoff, §4 (per cell: stage times, law, J, flags, sweeps by reason, residuals; histories and per-sweep fits) |
| spec tests | `python -m pytest -q tests/spec` (sandbox; 54 passed at `b17f768`) |
| roadmap | `specs/roadmap/allen-mtg-fit.yaml`; `python3 specs/roadmap/roadmap.py status --yaml specs/roadmap/allen-mtg-fit.yaml`; the page "Subproject Roadmaps" is redrawn by the scheduled "Roadmap sync" (every 2 h, 08:12-22:12 Rome) |

## 9. Rules of the road

- Coding directives v2: SPEC first; the orchestrator (Fable 5.2) plans, settles the science with the user, writes the
  interfaces; one Opus implementer per block, in parallel where blocks are ready together; the orchestrator settles
  their issues and asks the user only about the approach; the change report at the end; the final check by an
  independent Opus agent.
- A decision the user makes is logged in the same turn (`project-decision-log`): read the log's last changelog rows
  before taking an ID -- other chats log too (next free on 2026-11-19: D-181, I-006) [corrected 2026-11-19 (later):
  the defocus-table chat took D-181 and I-006; next free after the twins chat's D-182 to D-186: D-187, I-007].
- Documents live in the repository on `main` (DOCUMENT UPKEEP v2); a new SPEC block enters the roadmap in the same
  turn; theory statuses and new topics or steps only at the user's word.
- The assistant never runs anything on davinci; fixes travel through git (`hpc-git-delivery`); `.py` and `.sh` ASCII
  and LF; no `__pycache__` in a commit; dev checks are standalone scripts in `Ih Fit/dev/<date>_<topic>/`.

## 10. Traps

| Trap | Why |
|---|---|
| units | the tables give $\tau$ in ms, `noise_law` takes s; the per-sweep table's protocol is `SS`/`LS`, the JSON's `ss`/`ls` |
| a failed sweep | its `_mV` columns already hold $\hat\theta(c)$: do not fall back a second time |
| old dev scripts | the checks under `dev/` dated before 2026-11-17 use the four-parameter law and fail against today's `noise_law` (final check, F3); each runs at its own commit |
| run tags | `nf1` (two-component, superseded) and `nf2` (current) sit side by side; a new law needs a new tag |
| the roadmap's `pushed` | the tool reads tester reports only from the repository root's `specs/test_reports/`; this folder's reports (`Ih Fit/specs/test_reports/`, e.g. the final check of 2026-11-17) are not seen, so N1-N6 do not turn `tested` |
| N6's marker | the jobs' marker pushes from the compute node, which needs `module load proxy`; none arrived, so N6 shows `pushed` although the run is done; the same holds for Stage 7's jobs unless the job script loads the proxy |
| X11 forwarding and NEURON | `import neuron` exits with status 1 (`can't open DISPLAY`) when `DISPLAY` is set and no X server accepts the connection (MobaXterm's X server off, or a `HOME` without the X11 cookie); the submitter unsets `DISPLAY` (amendment 10); by hand, prefix `env -u DISPLAY`. The SSH server's `X11 connection rejected because of wrong authentication` lines are the sign |
| the gates from the repository root | the `smoke_twin_*.py` suites live in the "Ih Fit" folder and need `prova`; run from the root under `(base)` they are not found (2026-11-20) |
| the submitter inside a PBS job | refused (`PBS_JOBID` set); submit from the login node |
| multi-line pastes | the user pastes commands into a terminal: give one chained line or a tested one-liner, never a heredoc |
| the decision log is about 1 MB | search it with grep or a script and edit it with a script |
| binary files through the Drive connector | the Drive connector returns a file's content into the conversation as text; the archive's `.npz` files (0.2-8 MB per file on Drive, folders `specimen_508282493` and `specimen_569844159`) cannot reach the sandbox that way: the user attaches them, or the check runs on davinci |
| "Block 4b" | is Block 9; Blocks 5-8 are the twins, the noise-fit run (6), the record (7) and the gate run (8) |

## 11. Documents

| Path (repository, `main`) | What for |
|---|---|
| `Passive Features/HPC script/Ih Fit/specs/SPEC.md` | section 2.6 (the law), 2.4 (the tilt), 2.5 (the per-sweep stage), section 8 (Block 5's open question), the block entries |
| `docs/project/TEEG_decisions_and_ideas_log.md` | the decisions of §3 |
| `Passive Features/HPC script/Ih Fit/docs/TEEG_noise_fit_block6_handoff_2026-11-12.md` | the noise-fit run, its read-backs (§4), the earlier list of Block 5's questions (§6) |
| `Passive Features/HPC script/Ih Fit/docs/TEEG_noise_fit_implementation_handoff_2026-11-11.md` | the history of Block 9 and D-122 to D-125 |
| `Passive Features/HPC script/Ih Fit/dev/2026-11-17_three_component_law/` | the test that led to D-152 |
| `Passive Features/HPC script/Ih Fit/dev/2026-11-19_nf2_readback/` | the nf2 read-back and the slow-ridge check |
| `Passive Features/HPC script/Ih Fit/specs/test_reports/2026-11-17_1700_incremental.md`, `tests/spec/` | the final check of the three-component law and its 54 tests |
| `Passive Features/HPC script/Ih Fit/README.md`, `CHANGELOG_ih_fit.md` | the folder's map, commands and stage log |
| `specs/roadmap/allen-mtg-fit.yaml`, `.md` | the roadmap (N5, N7, S7) |

Skills to load: `scientific-coding`, `hpc-python-compat`, `hpc-git-delivery`, `subproject-roadmap`,
`project-decision-log`, `technical-document`, `explanation-modes`.
