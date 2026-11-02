# Handoff — deck pack: where the passive and I_h fits of the Allen human cells stand (2026-11-02)

| Date | Change |
|---|---|
| 2026-11-02 | v1. Written by the diameter calibration chat (Claude Code session `session_015FuK9w6BoUtXZmcfHQi8ra`) at the user's request (17:26): a handoff for a new chat that builds the **deck pack** — slide-by-slide md, the figures, and where each figure sits in Drive — which the user then gives to the design step that makes the slides. Read or run on 2026-11-02: project knowledge `claude/TEEG_annual_report_2026_00_overview.md` and `_01_passive_properties.md` (both 2026-09-28, read in full); repo `main` @`854004c` (`Passive Features/HPC script/Ih Fit/`: README, CHANGELOG, SPEC, `docs/TEEG_noise_fit_handoff_2026-10-28.md`; `docs/project/claude/TEEG_Ih_fit_staged_plan.md` v0.12; `docs/project/claude/TEEG_Stage7_noise_model_2026-09-28.md`); `roadmap.py status` on `specs/roadmap/allen-mtg-fit.yaml` (run here); branch `sci/diameter-pipeline` @`399180e`; Drive listings; past-chat search. Two read-only subagents summarised the `Ih Fit/` folder and the rest of `Passive Features/` on `main`. Their facts used below were spot-checked against the files, and some of the line numbers they gave were wrong, so this file cites files, not line numbers. |

Paths are relative to the repo `Leonardodm00/Towards-EEG` unless they start with `claude/` (project knowledge) or `Drive:` (`My Drive` of leonardodellamea.00@gmail.com).

---

## 1. The task

Build the deck pack for a talk on **where the fit of the passive and I_h parameters of human cortical neurons from the Allen Cell Types data stands on 2026-11-02**. Do not build the slides themselves; the pack is the input of the design step.

The pack has four parts:
- `deck_pack.md`, the slide script;
- the figures, plus the script that made each one;
- the placement of every figure in Drive;
- a short "for the designer" section.

No result is recomputed or reinterpreted for the deck. Numbers come only from the sources in §8 with their tags. A figure drawn from data cites the data file.

---

## 2. Ask the user first (answers change what gets built)

| # | Question | Default if unanswered |
|---|---|---|
| Q1 | Audience, occasion, date, talk length (sets the slide count) | lab meeting / supervisor update, ~20 min: 14–17 main slides + appendix |
| Q2 | Scope: passive (3 parameters) + I_h (6 parameters) as asked. Include the **inputs being corrected** (dendrite diameters re-measured from the Allen stacks; spine factor $F$ from H01) as 1–2 slides? They are the Stage-8 campaign's inputs (D-013). | yes, 1–2 slides |
| Q3 | Equations on the slides, or only in the speaker notes? | key equations in the notes; at most one per slide |
| Q4 | **Run B's per-cell data** for the result figures: does the user still have the 2026-09-19/30 slide figures (chat in §8.3), or will they upload the run B CSVs from davinci (`/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs/dt01ls60/<GROUP>/phase2_results.csv`, and the Phase 3 summary)? | without them, use group medians from the KB (§6) and say on the slide "medians only" |
| Q5 | Design target (Claude Design? a lab template? PowerPoint?), palette and font constraints | 16:9, neutral palette, figures as PNG + SVG |
| Q6 | Drive location for the pack (§4) | `Drive: TEEG_slides/2026-11_passive_Ih_status/` |
| Q7 | Reuse the 2026-09-30 "Presentation Section" (10 + 3 slides on run B and Phase 2.5, §8.3) as the passive-results block? | yes, updated to this storyline |

---

## 3. Deliverables and formats (the contract with the design step)

**`deck_pack.md`, one block per slide, in this order of fields:**

| Field | Content |
|---|---|
| `id` | S01, S02, …; appendix A01, … |
| `title` | ≤ 8 words |
| `message` | one sentence: what the audience must take away |
| `on_slide` | the text that appears, ≤ 40 words or ≤ 4 bullets; plain words, jargon defined in the notes |
| `figure` | file name, Drive path, Drive file ID once uploaded, size, and whether it already exists or was made for the deck |
| `caption` / `alt` | caption ≤ 25 words; alt text for accessibility |
| `notes` | speaker notes. Formal statement plus a plain one, following the project instruction "jargon + plainer explanation" |
| `sources` | every number with its tag (§7 classes) |
| `caveats` | what the slide must not imply |

**Figures**
- 16:9, 1920×1080 px PNG (also SVG when vector).
- Fonts large enough to read when projected; no title baked into the figure (the slide carries it).
- One script per generated figure, with data loading separate from plotting (project coding directives). These are standalone scripts, not pipeline code.
- Load the `dataviz` skill before any chart.

**Manifest**: `figures_manifest.csv` with columns `slide_id, file, drive_path, drive_id, origin (existing/new), script, data_source`.

**For the designer** (end of the md):
- the palette;
- which figures must not be cropped or recoloured, because their colours encode data;
- that no number may be edited.

---

## 4. Drive placement

**Proposed tree** (confirm Q6):

```
TEEG_slides/2026-11_passive_Ih_status/
  00_deck_pack.md
  figures_manifest.csv
  figures/01_context/  02_passive_method/  03_passive_results/  04_Ih_fitter/  05_noise/  06_inputs/  07_status/
  scripts/
```

**Mechanics**
- The Drive connector (`create_file`) creates folders cheaply and uploads text (`textContent`, with `disableConversionToGoogleType: true` so the `.md` stays Markdown).
- Uploading a PNG through it means emitting the image's base64 in the tool call. That is impractical beyond a few small files. So: create the folders through the connector, deliver the PNGs as files plus one zip that mirrors the tree for the user to drop in, then record the Drive IDs in the manifest once they are in place (`search_files` by title).
- Figures that **already exist in Drive** (§6.1) are referenced by path and ID, not copied.

**Where the md lives**
- The project's designated place for a deck pack is `claude/deck_pack/` in project knowledge (DOCUMENT UPKEEP).
- But project knowledge is **over its size limit**: 2,303,170 of 2,000,000 bytes (`project_info`, 2026-11-02), and writes are refused.
- Keep the md in the Drive folder and tell the user. Write it to project knowledge only if space is freed.

---

## 5. Proposed storyline (defaults; confirm with Q1–Q3)

| # | Slide | Message | Content from | Figure |
|---|---|---|---|---|
| S01 | Title | — | — | — |
| S02 | Where this sits in Towards-EEG | A cell's membrane parameters decide how its dendrites turn synaptic input into the current dipoles that sum to the EEG. | KB part 0 §3 | the 2026-09-16 pipeline figures (§6.3) |
| S03 | The data | Human Allen cells with both recordings and reconstructions: 68 cells in 10 layer × type groups, Square Subthreshold pulses and Long Square steps. | KB part 1 §3.1, §6 | `membrane_traces.png` of one archive cell (§6.1) |
| S04 | Which part of which trace pins which parameter | The early SS band carries $C_m$, the LS plateau the input resistance, the slow decay $\tau_m$. | the 2026-07-21 figure (§6.3) | `SS_LS_parameter_identifiability.png`, regenerated |
| S05 | Model and fit | NEURON on the reconstructed cell (Hay axon stub, spine factor $F$ beyond 60 µm). Gaussian-process Bayesian optimisation in log parameters, bootstrap confidence intervals. | KB part 1 §3.2–3.6 | new method diagram |
| S06 | The identifiability trap, and the fix | Plain RMSD pins $\tau_m$ but not $C_m$ and $R_m$ separately. Time-weighting, multi-protocol training and a 60 ms window separate them. | KB part 1 §4.1–4.3 | new; two panels kept apart (the before/after comparison is confounded, §6.1) |
| S07 | Validation | Synthetic recovery: $C_m$ ~12 %, $R_a$ ~15 %. On Eyal 2016's own six cells, $C_m$ is recovered at −1.8 ± 10.9 %. | KB part 1 §4.2, §5 | new: per-parameter error dots; Eyal-vs-ours dot plot |
| S08 | Results on the Allen cells (run B) | L2 and L3 $C_m$ come out 2–3× Eyal's 0.47 µF cm⁻² because the passive model absorbs $I_h$. | KB part 1 §6 | 09-30 slide figures, or new from the CSVs (Q4) |
| S09 | What the real data identify | $R_m$ is the least determined parameter (relative CI width 0.592), not $R_a$ (0.186) or $C_m$ (0.131). | KB part 1 §6 | new bar chart |
| S10 | Why the h-current has to be fitted | The three published human $I_h$ kinetics disagree several-fold at rest: $m_\infty$ 0.026 / 0.166 / 0.104, $\tau^h$ 46 / 73 / 341 ms. | KB part 1 §7.1 (D-005) | new: $m_\infty(v)$, $\tau^h(v)$ from the mod files (§6.2 F7) |
| S11 | The six-parameter fitter | $(C_m, R_m, R_a, \bar g_h, \Delta v_h, \kappa_\tau)$; the leak battery balanced so the model rests where the cell rests; full-step training, depolarising validation. | KB part 1 §7.2–7.3; staged plan v0.12 | new protocol-roles schematic |
| S12 | Before trusting it: the recovery gate | Synthetic twins on real morphologies, a quarter without $I_h$ as a false-positive control. It runs on davinci before any real cell (D-083). | KB part 1 §7.4; decision log D-011, D-083 | new gate schematic |
| S13 | The noise the twins need | Long Square noise is not AR(1). Each cell gets a fitted law (fast AR(1) + slow OU, plus per-sweep offset and tilt), estimated by indirect inference. | KB part 1 §7.5; noise model doc; Ih Fit SPEC | repo `dev/` method figures (§6.1), or new |
| S14 | Correcting the inputs | Dendrite diameters are being re-measured from the Allen image stacks, and $F$ from H01 meshes. Both feed the campaign (D-013). | branch `sci/diameter-pipeline`; KB part 2 | Drive diameter figures (§6.1); the three-domain kernel idea (§8.2) |
| S15 | Status board | What is done, tested, implemented and planned, from the roadmaps. | §6.5 | new, from `roadmap.py status --json` |
| S16 | Next steps | Block 9 → Blocks 5–7 → Stage 7 gate on davinci → Stage 8 campaign on the corrected morphologies. | §6.3, §6.4 | new timeline |
| A01+ | Appendix | Phase 2.5 and why it was rejected; CI caveats; noise numbers; open decisions | KB part 1 §3.6, §6; §6.3 | 09-30 Phase 2.5 figures |

---

## 6. Status on 2026-11-02 (facts with tags)

Tags:
- **[KB pN §x]**: project-knowledge annual report part N, section x (2026-09-28). Its numbers were computed by the assistant from CSVs the user uploaded from davinci, on the dates the report cites.
- **[main]**: a file on repo `main` @`854004c`.
- **[RM]**: `roadmap.py status` on `allen-mtg-fit`, run 2026-11-02.
- **[DI]**: data inspected in the diameter chat on 2026-11-02.
- **[user]**, **[reasoning]**, **[open]**: as in the project's TRANSPARENCY classes.

### 6.1 Passive fit, 3 parameters $(C_m, R_m, R_a)$: estimate of record exists

| Item | State | Tag |
|---|---|---|
| Archive | 68 Allen human cells in 10 groups (L2–L6 × exc/inh) on davinci, `/davinci-1/home/ldellamea/Human Neurons Fitting`; the L3_exc archive (18 cells) is also on Drive | [KB p1 §6]; [RM I0] |
| Production runs | run A `dt01ls60ph25` (Phase 2.5 on), run B `dt01ls60` (free $R_a$); **run B is the estimate of record**. Phase 2.5 was rejected: its second pass degrades 56 of 68 training fits and shifts $\tau_m$ by a median 20 % | [KB p1 §6] |
| Configuration | SS hyperpolarising target, exponential time-weight $\tau_w$ = 5 ms, LS window 60 ms, `dt` 0.1 ms, $F$ per group (see §7 on $F$) | [KB p1 §6] |
| Where run B's per-cell results live | only on davinci (`pipeline_outputs/dt01ls60/<GROUP>/`); no $C_m$/$R_m$/$R_a$ table of runs A or B is in the repo | [main], subagent check of all of `Passive Features/` |
| Validation | synthetic 30-cell recovery with human $I_h$ (2026-08-11); Eyal 2016 replication on six cells (2026-08-13) | [KB p1 §4.2, §5] |
| Roadmap | G1 archive and G2 passive fit **done** (set by hand: run A/B on davinci, the user's passive-only synthetic check of 2026-10-26); theory TG1 closed | [RM] |
| Caveat to carry | Phase 3 intervals: resampling covers SS pulses only (the LS steps are absent from every replicate), the warm-start ball bounds the replicate spread, BCa collapses on 34/99 (run A) and 41/93 (run B) intervals | [KB p1 §3.6] |

### 6.2 I_h extension, 6 parameters $(C_m, R_m, R_a, \bar g_h, \Delta v_h, \kappa_\tau)$: built and smoke-tested, never fitted to a real cell

| Item | State | Tag |
|---|---|---|
| Design | `Ih_human` base kinetics (Rich 2021); fitted shift $\Delta v_h \in [-10, 10]$ mV and time-constant scale $\kappa_\tau \in [0.5, 2]$; $\bar g_h \in [10^{-6}, 10^{-3}]$ S cm⁻², uniform; $e_{\rm pas}$ set per segment so that $V_{\rm rest}$ is a stationary state (KB p1 Eq. 7.2) | [KB p1 §7.2] |
| Protocol | train on the SS hyperpolarising pulses and LS $h_2 \dots h_{n-1}$ over the whole step; validate on $h_1$, the spike-free depolarising steps and the SS depolarising pulses; report-only $h_n$ and the rebound windows. Arms: `ih6`, `passive_fullstep`, `baseline_runB` | [KB p1 §7.3] |
| Stages 0–6 | Stage 0 regression PASS bit-for-bit on the real cell 508282493 (davinci, 2026-10-20); Stages 1–6 smoke-tested on davinci 2026-10-20 (12/12); **"no real cell fitted yet"** | [RM I5–I11] |
| Stage 7 (recovery gate) | built and smoke-tested (14/14, 2026-10-20), **the gate itself never run**; D-083 (user, 2026-10-26): the six-parameter synthetic recovery runs on davinci, on twins with each cell's own morphology and fitted noise, before any six-parameter fit of a real cell | [RM I12]; decision log D-083 |
| Gate tolerances | median $\lvert\varepsilon\rvert \le \log 1.25$ ($C_m$, $\bar g_h$), ≤ 3 mV ($\Delta v_h$), $\le \log 1.5$ ($\kappa_\tau$), ≥ 80 % low-rail on the no-$I_h$ control. These are **the assistant's proposals, not decided** | [KB p1 §7.4] |
| Stages 8 (campaign), 9 (record) | not started; Stage 8 waits for the gate's verdict and the diameter-corrected SWCs (D-013) | [RM S8, S9] |

### 6.3 The noise model of the synthetic twins: the current work (noise-fit chat, on `main`)

| Item | State | Tag |
|---|---|---|
| Why | the Long Square pre-window autocorrelation on L3_exc is 0.566 at 1 ms against 0.000 for AR(1): AR(1) does not reproduce the LS noise | [KB p1 §7.5] |
| The law | per cell: fast AR(1) $(\sigma_f, \rho_f)$ + slow OU $(\sigma_s, \tau_s)$, plus a per-sweep offset and tilt (each real sweep's tilt copied into its twin); fitted by indirect inference with simulated weights | `Ih Fit/specs/SPEC.md`, `README.md` [main] |
| A cohort value | closed-form fit on L3_exc: $\hat\sigma_f$ 0.0380 mV, $\hat\rho_f$ 0.175, $\hat\sigma_s$ 0.0508 mV, $\hat\tau_s$ 36.6 ms. An older handoff (2026-10-06) shows slightly different values (0.0381, 0.176, 0.0506, 36.3); use the noise model doc's | `docs/project/claude/TEEG_Stage7_noise_model_2026-09-28.md` table "θ_cf" [main] |
| Blocks | 1 (textbook ACF), 2 (the law), 3 (windows and statistics), 4 (per-cell fit): implemented and sandbox-tested, **pushed, no independent tester report**. Block 9 ("4b", per-sweep stage, D-116): SPEC drafted, no code. Blocks 5–7: planned, no SPEC entry. Cluster gates 13/13 SETTLED on davinci 2026-10-20 (not re-run since Blocks 3–4) | [RM N1–N7, NT] |
| Waiting on the user | Block 9's four open points: bootstrap size $B_v$ (proposal **50**, corrected from 100 on 2026-11-02, commit `854004c`); the refits' binding size $M$ (proposal 1000); a failed refit (proposal: left out and counted); D-116.4's "the cell's values". The handoff says: no Block 9 logic before the answers | `Ih Fit/docs/TEEG_noise_fit_handoff_2026-10-28.md` item 6 [main] |
| Theory side | T5 noise law, T6 window statistics, T7 indirect inference, T9 per-sweep calibration: **studying**. T8 tilt and semivariogram: open | [RM theory] |

### 6.4 Inputs being corrected (S14)

| Item | State | Tag |
|---|---|---|
| Spine factor $F$ | fixed model input in runs A/B (KB p1 §9 item 7: 1.9 L2/3, 1.5 deep exc, 1 inh); the mesh-based $F$ of the H01 pipeline has not been fed back into any passive fit. Part 2 of the report (2026-09-28) gives the structural status; no later decision-log entry on the H01 campaign was found (checked 2026-11-02) | [KB p1 §9]; [KB p0 §5] |
| Dendrite diameters | the re-measurement pipeline (branch `sci/diameter-pipeline`) runs on Allen's image stacks in Colab for the pilot 529878215 and is set up for the S3 survey of 51 selected cells on davinci. The bias table and the corrected SWCs (the D-013 input of Stage 8) **do not exist yet** | branch files; [user] |
| Diameter: the defocus-kernel calibration on the pilot | Cell 7 (2026-11-02): near-field anchor $\hat\sigma_0$ = 0.562 ± 0.049 µm on 29 nodes; far slope $\hat\gamma$ = 0.404 from 2 far nodes, a diagnostic only (below the minimum of 10 nodes per set). D-118 (today): S3c scans up to 30 nodes per cell and set, and the pool keeps every cell's | [DI] Cell 7 fit files; decision log (branch copy) D-117, D-118 |
| Is the davinci S3 run submitted? | not known to this chat | [open], ask the user |

### 6.5 Status board source

Run on a clone of `main`:

```
python3 specs/roadmap/roadmap.py status --yaml specs/roadmap/allen-mtg-fit.yaml
```

This gives the colours on 2026-11-02:
- **done**: G1, G2, I0–I5, NT, I13;
- **tested**: I6–I12;
- **pushed** (implemented, untested): N1–N4;
- **planned**: N4b, N5–N7, S7–S9.

The diameter subproject: `specs/roadmap/diameter-pipeline.yaml` on the branch `sci/diameter-pipeline`. The live page "Subproject Roadmaps": https://claude.ai/artifact/Fe6Zqjr5iq3XAQZfrTspqB.

---

## 7. Numbers cleared for slides

Copy them as written, with the tag, into `sources`.

| Quantity | Value | Tag |
|---|---|---|
| Human Allen cells with 3-D reconstructions, at the time of the guide | 158 | [KB p1 §3.1] |
| Cells in runs A/B | 68 in 10 groups | [KB p1 §6] |
| Run B exc medians $C_m$ (µF cm⁻²) / $R_m$ (Ω cm²), n | L2 1.478 / 17773, 10; L3 1.062 / 26220, 18; L4 0.785 / 44647, 6; L5 0.797 / 51064, 8; L6 1.100 / 42057, 5 | [KB p1 §6] |
| Eyal 2016 human L2/3 $C_m$ | 0.47 ± 0.03 µF cm⁻² (L2 ~3×, L3 ~2.3× above it) | [KB p1 §6], citing the KB full text of Eyal 2016 |
| Median relative bootstrap width (run B, 31 cells) | $C_m$ 0.131, $R_a$ 0.186, $R_m$ 0.592 | [KB p1 §6] |
| Cells at the $R_a$ floor (50 Ω cm) | 16 of 68 | [KB p1 §6] |
| Phase 2.5 | degrades 56/68 training fits; $\tau_m$ shifted by a median 20 %; L2 exc $C_m$ 1.478 → 2.730 | [KB p1 §6] |
| Plain-RMSD synthetic failure (8 cells, one truth) | $\tau_m$ ratio 1.09 (CV 8.9 %); $C_m$ 1.41; $R_m$ 0.78; $R_a$ 1.82; corr(log $C_m$, log $R_m$) = −0.972 | [KB p1 §4.1] |
| Recovery after the redesign (30 L3 morphologies, human $I_h$) | typical error $C_m$ ~12 %, $R_m$ ~24 %, $R_a$ ~15 %, $\tau_m$ ~25 %; $R_m^{\rm fit} < R_m^{\rm true}$ in 27/30 | [KB p1 §4.2] |
| 60 ms vs 150 ms LS window (12 paired cells) | $R_m$ ratio 0.769 → 0.848, paired p = 0.021, improved in 10/12 | [KB p1 §4.3] |
| Eyal 2016 replication | $C_m$ deviation −1.8 ± 10.9 % (exp-weighted), −4.8 ± 12.1 % (unweighted) | [KB p1 §5] |
| Residual lag-1 autocorrelation | > 0.5 in 61 of 68 cells (run B) | [KB p1 §6] |
| $I_h$ kinetics at $V_{\rm rest}$ = −73.5 mV | $m_\infty$: Kole 2006 0.026, Kole + Kalmbach shift 0.166, Rich 2021 0.104; $\tau^h$: 46 / 73 / 341 ms | [KB p1 §7.1] (D-005, run from the repository's reference implementations) |
| LS noise autocorrelation, L3_exc | 0.613 at 0.1 ms and 0.566 at 1 ms, vs AR(1)'s 0.134 and 0.000 | [KB p1 §7.5] |
| Noise law, closed form, L3_exc | 0.0380 mV, 0.175, 0.0508 mV, 36.6 ms | §6.3 |

**Not cleared**:
- any per-cell or inhibitory-group value (it needs the CSVs, Q4);
- any $I_h$ parameter of a real cell (none exists);
- the "Rm 14 % too low" phrasing of the 2026-08-11 report (§8.3): which synthetic $I_h$ it rests on is unclear (§9 item 8). Use the KB p1 §4.2–4.3 values, with their caveats.

---

## 8. Figure inventory

### 8.1 Existing figures in Drive (reference by ID; do not copy)

| What | Drive location | File ID(s) |
|---|---|---|
| Per-cell recorded traces and viability check (Phase 0 archive, June 2026) | `Colab Notebooks/Allen_Intitute_Data/` (folder `L3_exc` = `1IAcrKTXXS3IaFsd4XbnhUJgbjEutX3Yd`); per-cell folders hold `membrane_traces.png`, `electrical_viability.png` | e.g. `L3_exc/specimen_571483807/`: `membrane_traces.png` `18v37o6EE3N-lGf9Wleu5NBgBvlFyHGD4`, `electrical_viability.png` `1GhJcHkdkaoTVFwBf-ewI6h1tV3Y4qFLX` |
| Diameter calibration on the pilot (Cell 7, updated 2026-11-02) | `diameters/calibration/figures/` (`1qdvMy7CqSIf8rsqtH5IocAjkw3HixK2N`) | `kernel_law.png` `1iy0c7ApTbNdfnMUAbDHsQqTnRra-9KiI`; `width_by_plane_near.png` `1np4C1JI8Ba1Ctnm9-r6J58yHFXWm1xaa`; `width_by_plane_far.png` `1cizBWhKWbIAumECo8zKScmg8zgYLfZbe` |
| Allen image stack with the SWC drawn over it (pilot; S3 alignment) | `diameters/survey_pilot/…/align/` | `overlay_529878215.png` `1lI5rJmSJrN-_hV3UlO2rC53D3wsIdInR` |
| May 2026 Phase 3 diagnostics (test runs; old; appendix at most) | several folders | `gp_mean_heatmap_Cm_Rm.png`, `profile_Cm.png` (search by title) |

Folder names under `diameters/` follow the code's output paths (`OUT/calibration/figures`, `OUT/survey_pilot/...`); the parent folder names were not all opened.

### 8.2 Existing figures in the repo

All are method or synthetic figures. **None shows a real recording or a fit to a real cell.**

**`Passive Features/HPC script/Ih Fit/dev/`** on `main`, captions in each folder's doc:
- `2026-10-26_tilt_sigma_b/figures/fig01_window_means.png`, `fig02_semivariogram.png`, `fig03_lag_ratio.png`, `fig04_sampling.png` (`fig05_pooling.png` is no longer used for the twins, D-102);
- `2026-10-28_tilt_explanation/fig01_overview.png` (shows the design before D-102: check before use), `fig_nugget.png`;
- `2026-10-30_offsets_and_fit/fig_fit_profile.png`;
- `2026-10-16_covariance_weighting/bias_doc_2026-10-11/figures/fig01`–`fig05` (bias of a statistic, explanatory).

**Diameter workstream:**
- `Passive Features/Diameter Re-measurement/docs/TEEG_diameter_presentation_ideas_2026-10-10.md` (main): Idea 1, the defocus kernel in three domains, with a figure specification;
- `TEEG_ray_world_vs_partition_2026-10-07.svg` in the same folder.

### 8.3 Figures and slide text in past chats

Retrieve with `conversation_search` / `read_conversation`, or ask the user for the files.

| Date | Chat | What it holds |
|---|---|---|
| 2026-07-21 | "Passive parameter fitting visualization" https://claude.ai/chat/68e62322-c7cf-4c9f-a299-754fa80079c9 | `SS_LS_parameter_identifiability.png` and its scripts (`traces.py`, `plot_figure.py`): schematic traces, not measured |
| 2026-08-11 | "Synthetic passive fit test results" https://claude.ai/chat/f9955170-f31f-4e8a-9065-9694f435bbf5 | the synthetic benchmark report with "Presentation notes for whoever builds the deck" (ratio vs dex, systematic vs scatter) |
| 2026-09-16 | "Improving handmade drawings" https://claude.ai/chat/82c4d2db-882c-4b13-91bf-502d939b99fd | the two pipeline overview figures (point network, co-simulator, EEG), published there as a Design artifact "TEEG Pipeline Figures", plus faithful redraws of the user's sheets |
| 2026-09-19 to 09-30 | "Passive parameter optimization analysis and documentation" https://claude.ai/chat/126e0cd6-8be6-46f4-87d8-93bb45bd1d04 | "Presentation Section — Passive-Parameter Fitting … Results on Real Data" (10 + 3 slides + appendix) and `make_presentation_figures.py` (16:9, from run B CSVs; e.g. `slideB3_fit_quality.png`) |
| 2026-09-28 | "PhD year progress report" https://claude.ai/chat/9c8d155b-36db-4bf9-a83d-10c70c0a26f4 | the annual report (KB parts 0–2) |

### 8.4 Figures to make

| Id | Figure | Data | How |
|---|---|---|---|
| F5 | Method diagram (S05) | none | schematic |
| F6 | Identifiability trap (S06): panel 1 the 8-cell plain-RMSD ratios, panel 2 the 30-cell errors after the redesign, kept as separate panels with the caveat that they are not a like-for-like comparison | KB p1 §4.1–4.2 tables | summary values only |
| F7a | Eyal replication (S07): Eyal vs ours, per cell | KB p1 §5 table (6 cells) | dot plot |
| F8 | Run B medians by layer (S08), with the Eyal band | §7 table, or the CSVs (Q4) for per-cell points | — |
| F9 | Relative CI widths (S09) | §7 | bar chart |
| F10 | $I_h$ kinetics (S10): $m_\infty(v)$ and $\tau^h(v)$ for Kole 2006, Kole + 20 mV (Kalmbach), Rich 2021 | equations in `Ih Fit/mod/Ih.mod` (Kole-type rates, `vshift`) and `Ih Fit/mod/Ih_human.mod` (Rich 2021) [main] | reproduce the §7 values at −73.5 mV before drawing (check by computation) |
| F11 | Protocol roles (S11): LS family coloured by role (train / validate / report), SS pulses | schematic | — |
| F12 | Recovery gate (S12) | KB p1 §7.4 | schematic: cohort, the quarter without $I_h$, error coordinates, tolerances labelled "proposed" |
| F15 | Status board (S15) | `roadmap.py status --json` on both roadmaps | — |
| F16 | Timeline (S16) | §6.2–6.4 | no dates the user has not given |

---

## 9. Known inconsistencies

Do not put these on a slide without resolving them with the user.

1. **$F$ for L4–L6 excitatory cells** differs across scripts and docs on `main`: 1.5 (`Biological Fit/Biological_Passive_Fit_HPC_Guide.md`), 1.6 (`submit_all_groups_biological.sh`), 1.5/1.3/1.4 (`Multiple nodes/submit_all_groups.sh`), 2.0 (`Multiple Sweeps with phase 2.5/...description.md`, `benchmark_usage_guide.md`). KB p1 §3.2 already flags that whether runs A/B used 1.5 is **not verified** (read the wrapper on the cluster).
2. **Spine-factor cutoff**: > 60 µm (Phase 1 README, KB) vs "within 50 µm" (the phase 2.5 description).
3. **Bootstrap type on real data**: parametric residual (`Phase 3/README.md`) vs nonparametric pulse resampling (`Biological Fit` guide and submit script). KB p1 §3.6 describes both.
4. **Run B's LS window**: 60 ms (decision log, KB) while every committed version of the submit script has `LS_WINDOW_MS=150.0`. [inferred by the subagent: edited on the cluster]
5. **Noise-law values**: 2026-10-06 handoff vs noise model doc (§6.3); use the latter.
6. **The $I_h$ staged plan**: the project-knowledge copy (created 2026-09-23) is older than the repo copy `docs/project/claude/TEEG_Ih_fit_staged_plan.md` v0.12 (2026-10-21). Read the repo copy.
7. **The decision log**: project knowledge's copy is behind `main`'s, and `main`'s is behind the branch copy on `sci/diameter-pipeline` (D-117, D-118 exist only there; next free ID D-119).
8. **Which synthetic $I_h$ the benchmark used.**
   - KB p1 §4.2 says the 30-cell recovery injected human kinetics (`Ih_human`).
   - KB p1 §4.3 says the $I_h$ "then in use was rodent-fast" for the 60 ms vs 150 ms comparison.
   - The repo records that the rodent-$I_h$ `v60ls` benchmark ran on davinci, but has no record that the human-$I_h$ `v70human` configuration ran (subagent read of `Synthetic Passive fit Test/`).
   - Check the 2026-08-11 chat before a slide names the kinetics.

---

## 10. Source map

| Source | Where | Use it for |
|---|---|---|
| Annual report part 0 (overview), 2026-09-28 | `claude/TEEG_annual_report_2026_00_overview.md` | context, timeline, status, S02 |
| Annual report part 1 (passive + $I_h$), 2026-09-28 | `claude/TEEG_annual_report_2026_01_passive_properties.md` | almost every number of S03–S12 |
| Annual report part 2 (structural, $F$) | `claude/TEEG_annual_report_2026_02_structural_pipeline.md` | S14 ($F$) |
| $I_h$ staged plan v0.12 | `docs/project/claude/TEEG_Ih_fit_staged_plan.md` (main) | S11–S12 |
| Noise model doc | `docs/project/claude/TEEG_Stage7_noise_model_2026-09-28.md` (main) | S13 |
| Ih Fit README, CHANGELOG, SPEC, latest handoff | `Passive Features/HPC script/Ih Fit/{README.md, CHANGELOG_ih_fit.md, specs/SPEC.md, docs/TEEG_noise_fit_handoff_2026-10-28.md}` (main) | S13, S16, open points |
| Roadmaps | `specs/roadmap/allen-mtg-fit.yaml` (main); `specs/roadmap/diameter-pipeline.yaml` (branch `sci/diameter-pipeline`) | S15 |
| Decision log | `docs/project/TEEG_decisions_and_ideas_log.md`: main's copy, and the branch copy for D-117/D-118 | D-005 to D-016 ($I_h$ design), D-013 (morphology input), D-083 (gate first), D-095 to D-116 (noise fit), D-117/D-118 (diameter calibration) |
| Diameter workstream | branch `sci/diameter-pipeline`: `specs/SPEC.md`, `Passive Features/Diameter Re-measurement/docs/TEEG_diameter_phase2_colab_runbook_2026-10-06.md`, `docs/project/claude/TEEG_diameter_HPC_paths_reference.md` | S14 |
| Literature (full texts in the KB) | Eyal et al. 2016; Rich et al. 2021; Kalmbach et al. 2018; Moradi Chameh et al. 2021; the Allen Cell Types white paper | S08, S10. Quote numbers only from a full text actually read (FULL-TEXT rule) |

---

## 11. Rules that bind the pack

- **Project instructions**:
  - SOURCE PRIORITY, the FULL-TEXT rule, METADATA IS NOT DATA, TRANSPARENCY tags on every number;
  - jargon plus a plain explanation in the notes;
  - the user's notation preference: every symbol defined at first use, conditioning and quantifiers written out (e.g. $m_\infty(v)$ "for each fixed $v$").
- **What a slide must not say**:
  - that a real cell's $I_h$ parameters are known;
  - that the recovery gate passed;
  - that the diameter correction or the mesh $F$ is applied to any fit.
- **Provisional values** (gate tolerances, Block 9 proposals) are labelled "proposed" on the slide.
- **Documents**: the deck pack is a DOCUMENT UPKEEP doc kind (`claude/deck_pack/`). Every other project doc is outside this task. End the reply with the one line naming the docs written.
