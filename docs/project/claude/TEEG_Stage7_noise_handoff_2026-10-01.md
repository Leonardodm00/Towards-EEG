# TEEG -- Stage 7 noise model: handoff (2026-10-01)

| Date | Change |
|---|---|
| 2026-10-01 | Created for a new chat. State after the topic document of 2026-09-28 and the delivery of `ls_baseline_qc.py` as `ih_lsqc.patch` (applied, pushed and run: **not yet**, as of writing). |
| 2026-10-01 (same turn) | Two lines of the first write corrected before first use: §4.1 had quoted a stimulus amplitude and a sweep number read through a summarising fetch (removed: no number from such a fetch is used); §4.2's sign rule said a wander agrees in sign "about half the time", true only for a wander much faster than the windows' spacing (restated with that condition). |
| 2026-10-01 (later) | `ls_baseline_qc.py` pushed (`3dfa263`, the patch byte for byte) and run on L3_exc by the user; results in the new §4.6 (cluster output pasted 14:34, read-out run in the sandbox). §1 state and next steps, §3 (a), (c), (e), §4.5 and §5 updated. §4.2's sign rule **corrected** by Monte Carlo: under a stationary wander of any timescale the sign agreement is 0.49-0.53, so the sign test is strong evidence against a continuing creep and says nothing about the wander's timescale. |
| 2026-10-08 | Markers only, no text rewritten: D-028/D-029 (textbook autocorrelation through statsmodels; patch on `b355d8e`, pending push and the cluster gates); D-033 settles §3 (d)'s estimator family (our own indirect inference, no ABC); D-034 settles §3 (c) and §5's OU-or-tilt proposal: the variance at seconds is carried by the per-sweep tilt, calibrated from the SS window-mean semivariogram, and the SS noise windows are enlarged where the pulse transient has died out. §0's next free ID is now D-035. |
| 2026-10-08 (later) | The topic doc is now **v3**: body rewritten to the current approach at the user's request, superseded text removed; its own changelog lists what changed. Marked here: §0's section map, §2's pre-D-028 cohort numbers, "every cell" in §3 (e) and §4.6. Textbook cohort refit $\hat\theta_{\rm sim}$ = (0.0383 mV, 0.179, 0.0547 mV, 39.7 ms), $\tau_s$ 35.6-45.6 ms over five Monte Carlo draws; $h^*$ = 0.8 s recommended for D-034 (note there). Scripts: `TEEG_Stage7_noise_scripts_2026-10-08.zip`, sent to the user; not in the repo. |
| 2026-10-09 | Markers only. §0's next free ID is D-039 (the diameter chat logged D-035 to D-038 on 2026-10-08). The topic doc is now v3.2: §3.3.3 corrected (the measured resonant frequency is the impedance peak, below the ringing frequency of the autocorrelation). A handoff for the next theory chat (`TEEG_noise_theory_handoff_2026-10-09.md`, with `TEEG_noise_theory_checks_2026-10-09.zip`) was sent to the user as files; it is not in project knowledge, which is over its cap. |
| 2026-10-20 | Markers only (noise chat), no text rewritten. Evidence: the cluster gates of 2026-10-20 (Blocks 1-2 settled, 13/13; the regression gate PASS on the real cell 508282493) and the SS sweep structure read on two cells -- G12 on the cluster and the Google Drive copy of their archive files: each SS sweep is a train of 37 pulses, 200.5 ms onset to onset, and the stored windows are those that passed the writer's trace QC (decision log 2026-10-20 (later, 5)). Marked: §0's next free ID (D-068, I-003); §3 (c) and its update, §4.6's open paragraph and §7's trap (20 pulses, 200 ms, 4 s, lags to 3.8 s); §8's regression row. Added to §6: the gate logs and the archive's origin with its trace-QC files. |
| 2026-10-23 | Markers only (noise chat), no text rewritten. **D-076** (user, 12:08): per-sweep form (a) -- the fast pair and sigma_s per sweep, sigma_s shrunk towards the cell by DerSimonian-Laird, tau_s and the tilt per cell; t_SS(c) = max(80 ms, transient test); the subtracted quantity is the per-polarity mean window (per-sample mean), confirmed; the twins carry the real cell's own LS amplitudes; no QC anywhere. **Blocks 3 and 4 of `Ih Fit/specs/SPEC.md` built, smoke- and mutation-tested and delivered** as `block34_patches_2026-10-23.zip` (four patches on `main` `64fb8c5`, which carries the user's roadmap commit): `noise_windows.py`, `noise_cell_stats.py` (W1-W10 10/10), `noise_fit.py` (F1-F10 10/10; the per-cell fit of 2.3, measured at 16 s per binding evaluation and 0.10 s per Sigma_hat replicate on a 2-sweep cell). Two findings await the user (decision log 2026-10-23 (later, 2)) [corrected 2026-10-23 (later): that log row is now (later, 3) -- relabelled after a concurrent write of the diameter chat; the explanation with its checks is (later, 4)]: the transient test as written is noise-limited (`WindowConfig.transient_floor_z`, z = 3 recommended) [corrected 2026-10-23 (later): noise-limited at every L3_exc window count; z = 3 as implemented withdrawn, it still fails 4-13 % of noise-only cells] and sigma_b(c) of a 2-sweep cell rests on two tilt draws. Marked: §0's next free ID (D-077, I-005) [corrected 2026-10-23 (later): D-078, I-005]; §5's state line. |
| 2026-10-23 (later) | Markers only (noise chat), no text rewritten. The two findings explained to the user with checks by computation (`check_findings_2026-10-23.py` with its results, sent to the user, not in the repo; decision log 2026-10-23 (later, 4)): the transient test as written puts the SS window start at or beyond 190 ms in 56-90 % of noise-only cells at every L3_exc window count (K = 348-40), the z = 3 option still in 4-13 %; a per-bin null simulated at the cell's layout fails at most 0.3 % at z = 3.5; options put to the user, the assistant recommending t_SS = 80 ms for every cell with the test as a flag only; sigma_b_hat / sigma_b spans 0.00-1.95 (95 %) with 2 SS sweeps, 0.59-1.44 with 10, and moves theta_hat by at most 0.18 SD (linearised), so its error matters for the twins' tilt; partial pooling across cells recommended; the LS pre-window slopes stay excluded (§3 update (c)), correcting the assistant's listing of them as an option. Repo: commit `e75bdea` on `sci/noise-fit` (spec and comments only), with the next patch. Marked: §0's next free ID (D-078, I-005; D-077 is the diameter chat's); the 2026-10-23 row. |
| 2026-10-21 | Markers only (noise chat): **D-072** (user) -- the stored SS windows, trace-QC'd when archived, are the noise fit's data as written; no shape test anywhere in the noise fit, the twins included; **D-073** (user) -- each twin sweep carries noise calibrated on its own real sweep, not a per-cell average (form open). Marked: §0's next free ID (D-074, I-003); §2's generator-input sentence; §3 (g) (D-016 (d) settled); §6's archive row (the QC table checked on the cluster in all 18 cells). |
| 2026-10-26 | Markers only (noise chat), no text rewritten. **D-083** (user, 09:31): the six-parameter synthetic recovery runs on davinci (Stage 7) before any six-parameter fit of a real cell; roadmap S7 restated under D-082. **The tilt $\sigma_b(c)$ explained as a study document** (`TEEG_tilt_sigma_b_2026-10-26.md`, five figures, checks check01-check05; sent to the user, not in project knowledge): $h^*$ = 4 periods confirmed for the 120 ms windows of $t_{\rm SS}$ = 80 ms (read/match 0.987-0.989); a 2-sweep cell's spread is the Cramer-Rao limit of its two realised tilts; per-cell, cohort and DerSimonian-Laird partial pooling of $\log\hat\sigma_b^2$ compared on an 18-cell cohort, pooling recommended (decision log, D-034 note of 2026-10-26); statsmodels 0.15 does not clip the DL $\hat\tau^2$, so Block 4b must (D-076 clarified). Roadmap T8 back to open at the user's word (patch on `main`, not pushed). Marked: §0's next free ID (D-084, I-005). Next for this chat: the thorough explanation of the window-start rule (transient test) the user asked for. [done 2026-10-26; then D-095, 2026-10-28] |
| 2026-10-28 | Markers only (noise chat), no text rewritten. **D-095** (user, 09:55): the SS window start is read from the exponential decay of the mean windows, per polarity and per sweep over [-10, 200) ms; $t_{\rm SS}(c)$ where the decay falls within $\kappa_{\rm tr}\hat\sigma_{\rm SS}(c)$; no binning; the same mean window subtracted on $[t_{\rm SS}(c), 200)$ -- supersedes D-051 (iv)'s test and D-076 (ii)'s $t_{\rm tr}$; its open points are listed in the entry. Delivered: `noise_fit_patches_2026-10-28.zip` (five patches on `main` `64fb8c5`, `e75bdea` included; supersedes `block34_patches_2026-10-23.zip`) and `TEEG_window_start_checks_2026-10-28.zip`. **Chat changed:** the current state and the next steps are in `TEEG_noise_fit_handoff_2026-10-28.md`, sent to the user as a file and **not in project knowledge, which is full** (write refused 2026-10-28). Marked: §0's next free ID (D-096, I-005); the 2026-10-26 row's next step (done). [corrected 2026-10-28 (later): the patches are pushed (D-097); the handoff is in the repository, `Ih Fit/docs/`; next free IDs D-098, I-005] |
| 2026-10-28 (later) | Markers only (noise chat), no text rewritten. **D-097** (user, 10:24): the assistant pushes its own commits, and every document is also committed to the repository so any chat can read it. **Pushed** to `main`, now at `cd32dad`: the five noise-fit patches (Blocks 3-4 and three spec commits: `0dfe8bf`, `65c659b`, `76577d2`, `a95e618`, `033e3bf`), the two roadmap patches (`8d8d602`, `4930ba1`), `53b2713` (roadmap: D-095's revise marks on N3 and N4), `d2ffcb4` (documents), `cd32dad` (roadmap rendered); nothing is left for the user to apply. **Where the documents are now:** the current handoff, `Passive Features/HPC script/Ih Fit/docs/TEEG_noise_fit_handoff_2026-10-28.md` (read it first; in the repository only, project knowledge being full); the tilt study document, `Ih Fit/dev/2026-10-26_tilt_sigma_b/`; the window-start checks, `Ih Fit/dev/2026-10-28_window_start/`; byte copies of this handoff, the topic doc, the staged plan and the decision log under `docs/project/` (repository root, `main`). Roadmap allen-mtg-fit updated at the user's word: N3 and N4 pushed, with D-095's revise marks; page version 44. **Still to be answered by the user** (the user asked that this be written down): the tilt $\sigma_b(c)$ of a 2-sweep cell -- (a) per cell, (b) one cohort value, (c) DerSimonian-Laird partial pooling across cells, the assistant recommending (c) -- and D-095's open points. Marked: §0's next free ID (D-098, I-005) and its read-first list; the 2026-10-28 row. |
| 2026-10-28 (later, 2) | Markers only (noise chat), no text rewritten. **D-100** (user, about 18:37): "Let's do A and B for the 2 sweep cells." -- for the 14 L3_exc cells with two SS sweeps the twins' tilt SD is taken both per cell (option (a), D-034) and as one cohort value (option (b)); partial pooling (c), the assistant's recommendation, not taken. The assistant's reading, to confirm: two labelled variants of twins per 2-sweep cell, the cohort value the $S_{\rm eff}$-weighted mean of the unclipped per-cell estimates over the 18 cells (clipped at 0), each real cell's own fit unchanged. D-095's nine open points recapped to the user, still unanswered. The checks behind the day's explanations (check06-check11, the nugget figure, the overview) committed to `Ih Fit/dev/2026-10-28_tilt_explanation/`; SPEC §8, the current handoff (`Ih Fit/docs/TEEG_noise_fit_handoff_2026-10-28.md`) and the Ih Fit README and changelog marked. Marked: §0's next free ID (D-101, I-005) and its read-first row. [corrected 2026-10-28 (later, 3): D-100 was the assistant's misreading and is withdrawn; the tilt rule is D-102] |
| 2026-10-28 (later, 3) | Markers only (noise chat), no text rewritten. **D-102** (user, 19:35): "I think you have misuderstood ... For each cell draw out per each sweep a tilt. the twin will recieve the very same tilt as in teh real cell so that no statistics has to be computed on the tilt" -- each twin sweep carries the realised tilt of its own real sweep, read on an SS sweep as the least-squares slope of its window means on time (the assistant's proposal); no $\sigma_b(c)$ computed for the twins; **D-100 withdrawn**. Open: the Long Square sweeps' tilts (a 265 ms pre-window cannot give one, §4.6; proposal `delta_end_mV` / 6.89 s) and the per-cell fit's simulator. **D-103** (user, 19:35): D-095's open points closed -- the per-sample median window per sweep and polarity, the fitted decay's point value, the maximum over groups, one exponential with a free baseline on [5, 200) ms, the 80 ms floor kept and 80 ms also the fallback (flagged); open: a failed group (point 9) and the fallback's trigger. `check_kappa_tr_start.py` (how $\kappa_{\rm tr}$ moves the start) added to `Ih Fit/dev/2026-10-28_window_start/`. Marked: §0's next free ID (D-104, I-005) and its read-first row; the (later, 2) row. |
| 2026-10-28 (later, 4) | Markers only (noise chat), no text rewritten. **D-104** (user, 19:53): the 80 ms fallback applies when the window $[t_{\rm SS}(c), 200)$ ms would be shorter than 50 ms (a start later than 150 ms), and with no valid group; a group whose fit fails or that has fewer than 5 windows (PROVISIONAL) is flagged and left out. On the RC test cell at $\kappa_{\rm tr}$ = 0.1 the fallback applies from $\tau_m$ = 34.0 ms (113 MOhm) or $R_{\rm in}$ = 234 MOhm (28.1 ms) (`check_fallback_window.py`, `Ih Fit/dev/2026-10-28_window_start/`). Roadmap allen-mtg-fit updated at the user's word: notes and revise marks for D-102 to D-104 on N3, N4, N5 and T8, no status changed (`a945a7a`). Marked: §0's next free ID (D-105, I-005). |

**Basis tags.** **[src]** read from source; **[run]** executed, in the
assistant's sandbox unless marked *cluster*; **[user]** the user's decision or
report; **[KB]** project-knowledge full text; **[log]** the decision log's own
text; **[repo]** data repository, through a summarising fetch, no number used;
**[reasoning]** the assistant's own argument; **[open]** not decided or not
verified.

## 0. Read first

| Doc | What for |
|---|---|
| `Passive Features/HPC script/Ih Fit/docs/TEEG_noise_fit_handoff_2026-10-28.md` (repository, `main`; added 2026-10-28 (later)) | **read first**: the thread's current state and next steps since 2026-10-28 -- D-095 to implement, and the answers the user still owes (the tilt option for 2-sweep cells; D-095's open points) [updated 2026-10-28 (later, 2): the tilt option answered, D-100 (its reading to confirm); D-095's open points still owed] [corrected 2026-10-28 (later, 3): D-100 withdrawn; the tilt is D-102 (realised per-sweep tilts), D-095's points D-103] |
| `TEEG_decisions_and_ideas_log.md` (project **root**) | D-011..D-017 bind everything here; D-017 with its status notes 1-4 is the current state. **Not** `claude/TEEG_decisions_and_ideas_log.md`: a stale copy holding only D-001 and I-001. D-018 belongs to the diameter workstream (another chat), so the **next free ID is D-019** [corrected 2026-10-08: D-035] [corrected 2026-10-09: D-039, I-003; D-035 to D-038 belong to the diameter workstream -- re-read the log's last row before logging] [corrected 2026-10-20: D-068, I-003] [corrected 2026-10-21: D-074, I-003; D-068 to D-071 belong to the diameter workstream] [corrected 2026-10-23: D-077, I-005; D-074, D-075, I-003, I-004 belong to the diameter workstream, D-076 to this one] [corrected 2026-10-23 (later): D-078, I-005; D-077 belongs to the diameter workstream -- and a concurrent write of the two chats lost a row once, so re-read the log immediately before every write] [corrected 2026-10-26: D-084, I-005; D-078 to D-082 belong to the diameter workstream, D-083 to this one] [corrected 2026-10-28: D-096, I-005; D-084 to D-094 belong to the diameter workstream, D-095 to this one] [corrected 2026-10-28 (later): D-098, I-005; D-096 belongs to the diameter workstream, D-097 to this one] [corrected 2026-10-28 (later, 2): D-101, I-005; D-098 and D-099 belong to the diameter workstream, D-100 to this one] [corrected 2026-10-28 (later, 3): D-104, I-005; D-101 belongs to the diameter workstream, D-102 and D-103 to this one] [corrected 2026-10-28 (later, 4): D-105, I-005; D-104 to this one]. Decisions are never edited in place. |
| `claude/TEEG_Stage7_noise_model_2026-09-28.md` | the full argument, with notation table: the law §3.3; why a window statistic is not a parameter §3.4; inference §3.5; drift §3.6 (§3.6.7 = what settles creep versus wander); open points §3.7.3 and §5; per-cell statistics Appendix A; the scripts behind every number Appendix B **[corrected 2026-10-08: v3 map -- law §3.3, window statistics §3.4, inference §3.5, seconds scale and tilt §3.6, enlarged SS windows §3.7, per-cell calibration §3.8, open points §5]** |
| `claude/TEEG_Ih_fit_staged_plan.md` | stage structure and smoke-suite map (S1-S13, R1-R14) |
| repo `Passive Features/HPC script/Ih Fit/README.md`, `CHANGELOG_ih_fit.md` | how each script is run on the cluster; one row per delivery |

## 1. State on 2026-10-01

| Item | State | Basis |
|---|---|---|
| GitHub `main` | `2fd7643` "Ih Fit: D-016 spike screen, LS noise structure" | [run: `git ls-remote`, 2026-10-01] |
| Cluster clone | at `2fd7643`: the 2026-09-28 noise table (`noise_L3_v2.csv`) carries D-016's columns, which only that commit writes | [user; run, cluster 09-28] |
| Push/pull needed for the pipeline | **no**: no fitting, generator or noise-table code has changed since `2fd7643`; the only pending change is the diagnostic below | [run] |
| Stage 7 recovery gate | BUILT (D-011..D-016), never RUN | [log] |
| D-017 noise law | decided 2026-09-28; implementation NOT started; waits for open points (a)-(c); (d) leans to simulation | [user; log] |
| `ls_baseline_qc.py`, `smoke_ls_baseline_qc.py` | delivered 2026-10-01 as `ih_lsqc.patch` (4 files, 904 insertions, sha256 `c036727339b6...`); applies cleanly to `2fd7643`; smoke **10/10** on a fresh clone of `main`, all 10 mutations caught; the user's paste-back one-liners of §4.4 tested on a synthetic group. **Pending: apply, commit, push (laptop); pull, smoke, run (cluster).** | [run] |
| Scripts behind the topic doc's numbers | `TEEG_Stage7_noise_scripts_2026-09-28.zip` (14 scripts + README with expected outputs), sent to the user 2026-09-28; **not** in the repo, **not** in project knowledge | [run] |

**Update 2026-10-01 (later).**

| Item | State | Basis |
|---|---|---|
| GitHub `main` | `3dfa263` "Ih Fit: ls_baseline_qc -- LS baseline across the sweep, archive tail + Allen QC features (D-017 a,c); smoke L1-L10", on top of `2fd7643`; its four files are byte-identical to the delivered patch | [run: fresh clone, 2026-10-01] |
| Cluster clone | at `3dfa263` (the run used the new script) | [user; run, cluster] |
| `ls_baseline_qc.py` on L3_exc | run 2026-10-01 (~14:30): archive half complete, 104 sweeps in 18 cells; Allen half not run (network guard tripped, §4.5); outputs `lsqc_L3.csv`, `lsqc_L3_cells.csv`, `lsqc_L3.log` in the Ih Fit folder on the cluster | [run, cluster] |

~~**Next steps, in order.** (1) Get the user's `ls_baseline_qc.py` output
(§4.4) and read it per cell (§4.2), computing the archive-side null from the
real `tail_ms` first. (2) Put (a)-(d) of §3 to the user with that evidence;
log each answer as it comes (D-019 onward). (3) If (c) still needs it, write
the SS pre-pulse baseline series diagnostic. (4) Implement §5. (5) Run the
gate.~~ [superseded 2026-10-01 (later): step (1) done, §4.6]

**Next steps, in order (2026-10-01, later).** (1) Get the user's answers on
(a) (keep both cells; the evidence now supports it, §4.6) and on building the
SS pre-pulse baseline series diagnostic; log each answer as it comes (D-019
onward). (2) Write and deliver that diagnostic: it decides the shape of the
seconds-scale variance (§4.6), which the per-cell law needs. (3) Put (b) and
(d) to the user. (4) Implement §5. (5) Run the gate. The Allen half of
`ls_baseline_qc.py` is optional now (§4.5).

## 2. The decided law (D-017), compact

For a real cell $c$ with synthetic twin $c'$, let $\Delta t = 1/f_s$ (s) be the
sampling interval of one of $c$'s protocols (the twin is sampled at the same
interval; D-014, D-015), and let one sweep of $c'$ have samples at
$t_j = j\,\Delta t$ (s), $j = 0, \dots, n-1$, with mid-time $\bar t$ (s). The
injected noise (mV) is

$$ e_j = f_j + s_j + a + b\,(t_j - \bar t), \tag{D-017.1} $$

with $f$ a stationary AR(1) of SD $\sigma_f(c)$ (mV) and lag-1 coefficient
$\rho_f(c)$ at $\Delta t$; $s$ a stationary Ornstein-Uhlenbeck process of SD
$\sigma_s(c)$ (mV) and correlation time $\tau_s(c)$ (s; quoted in ms below),
independent of $f$;
and $a$ (mV), $b$ (mV/s) a per-sweep offset and tilt, drawn once per sweep and
independently across sweeps from laws calibrated on $c$ (open point (c)). For
each fixed $c$ and every lag $k \in \mathbb{N}$ (samples),

$$ \gamma_c(k) = \sigma_f(c)^2\,\rho_f(c)^{k} + \sigma_s(c)^2\,e^{-k\,\Delta t/\tau_s(c)} \quad (\mathrm{mV}^2). \tag{D-017.2} $$

$\theta_{\rm noise}(c) = (\sigma_f, \rho_f, \sigma_s, \tau_s)(c)$ is fitted to
$c$'s own window statistics; **no cohort law** is ever a generator input [updated 2026-10-21: nor, since D-073, a per-cell average:
each twin sweep's noise is calibrated on its own real sweep; which parameters per sweep and which per cell is open]; the
twin's Long Square onset is `ls_pre_ms`$(c)$ + 5 ms (270 ms on every L3_exc
cell) **[user, 2026-09-28; log]**.

**Cohort numbers (diagnostics only; L3_exc, 18 cells).** **[corrected 2026-10-08: pre-D-028 values; the textbook refit is in the topic doc §3.5.3]**

| Quantity | Value | Basis |
|---|---|---|
| LS pre-window `ls_pre_ms` | 265 ms in all 18 cells | [run, cluster 09-28] |
| AR(1) failure | last 132.5 ms of the LS pre-window: autocorrelation 0.448 at 1 ms, 0.244 at 10 ms; AR(1) implies 0.000 | [run, cluster 09-28] |
| closed-form fit $\hat\theta_{\rm cf}$ | $\sigma_f$ 0.0381 mV, $\rho_f$ 0.176, $\sigma_s$ 0.0506 mV, $\tau_s$ 36.3 ms; cost 1.88 against 215 for the best single AR(1) | [run] |
| simulated-medians fit $\hat\theta_{\rm sim}$ | 0.0381 mV, 0.179, 0.0560 mV, 37.3 ms; cost 0.69 (3.73 for $\hat\theta_{\rm cf}$ under the same objective) | [run] |
| bias of the closed form | ratio of expectations overshoots the autocorrelation estimator's Monte-Carlo mean by 0.008-0.05 and its median by 0.02-0.07 (ten statistics) | [run] |
| $\tau_s$ profile (closed-form cost) | 30, 11, 3.0 at 5, 10, 20 ms; 1.88 at 36.3 ms; 2.2, 2.8, 4.1 at 60, 100, 300 ms ($\sigma_s$ 0.099 mV at 300 ms): bounded below, loose above | [run] |
| SS residual | slow component supplies 25 %, 41 %, 51 %, 57 % of one SS trace's baseline-subtracted noise variance at 0.5, 5, 10, 15 ms after the pulse ($\hat\theta_{\rm cf}$) | [run] |
| trend share (share of pre-window variance a least-squares line explains) | 526785799: 0.58, 571430815: 0.64 (medians over 5 sweeps); cohort median 0.14 against its null 0.067 $\pm$ 0.017 ($P$ = 0.0004) | [run, cluster 09-28; run] |
| per-cell reproductions by stationary slow noise | 526785799: 1-2 % ($\sigma_f$ = 0.043 mV), 5-8 % ($\sigma_f$ lowered to 0.034-0.036 mV, last segment then 0.052-0.054 mV against the cell's 0.046): creep favoured, not decisively. 571430815: 7.5-12 % (correlation time 0.3-2 s): compatible with its own slow noise | [run] |
| locked share above the null's 95th percentile | 529878215 (0.186, above the 99th, 0.155), 571691416, 548425467, 571624532, 537313600 (all $K$ = 6): a weak sweep-locked component is not excluded in these | [run; cluster] |

## 3. Open points of D-017, in the order they block

| Point | Question | Assistant's recommendation | What settles it | State |
|---|---|---|---|---|
| (a) | 526785799 and 571430815 (high trend share): keep with a calibrated law, or exclude | keep both, flagged in the gate report; excluding them from the *real* fit is a separate QC decision | `ls_baseline_qc.py` (§4) | asked 2026-09-28, user has not answered |
| (b) | one law per cell for both protocols (SS pulses and LS sweeps from the same process, each through its own window) or D-014's per-protocol AR(1) pair for SS | one law (the cohort fit's form); the SS residual is not fast-only (§2 table) | user's call | not answered; D-014 (i) stands until then |
| (c) | where drift (tilt $b$) and everything slower than 265 ms are calibrated | `vm_delta_mv` plus the archive's own signed change (§4), **and** the SS pre-pulse baseline series (20 pulses 200 ms apart, about 4 s per sweep **[KB; src]**) [corrected 2026-10-20: in the archive's cells 37 pulses 200.5 ms apart onset to onset, about 7.2 s per sweep; data, decision log 2026-10-20 (later, 5)] | §4 for the first half; the SS series is **not written** | half delivered; **decided 2026-10-08, D-034** (tilt from the SS series) |
| (d) | estimator for the per-cell fit | simulated medians through `noise_calibration.py`'s own estimators, closed form only as a start; abcTau (BSD-3, GitHub only) as the slower option with posteriors | user's call | leans to simulation [log]; **family decided 2026-10-08, D-033** |
| (e) | $\tau_s$ loosely bounded above by windows of at most 265 ms; the twin's 1 s step extrapolates the law | bound it with the SS series of (c) | (c) | open |
| (f) | cells whose two protocols were sampled at different rates (none in L3_exc; other groups mix 50 and 200 kHz) | decide before any group other than L3_exc | -- | open |
| (g) | D-016 (d) per-sweep noise differences, D-016 (e) LJP level of the spike screen, D-015's stationarity across a response | unchanged | -- | open [2026-10-21: D-016 (d) settled by D-073 -- per-sweep calibration of the twins, form open] |

**Update 2026-10-01 (later), from the first run (§4.6).** (a) The two cells'
pre-window trends do not continue across the sweep: they are slow wander, not
creep, and both cells' tail changes lie inside their own wander scenarios. The
recommendation (keep both, flagged) now rests on data; the decision is still
the user's. (c) The per-sweep tilt $b$ must **not** be calibrated from
pre-window slopes (it would be too large by more than an order of magnitude);
the variance at seconds is measured by the LS tail (`delta_end_mV`, one lag of
6.89 s) and, once written, the SS pre-pulse series (lags 0.2-3.8 s [corrected 2026-10-20: 0.2005-7.218 s]), which
decides its shape. (e) Every cell **[corrected 2026-10-08: 17 or 18 of the 18, depending on the law]** carries more baseline change over 6.89 s than
the cohort law allows: a single 37 ms slow component is not enough for a twin
whose fit window runs 1 s past the pre-window.

## 4. `ls_baseline_qc.py`: creep or wander, for (a) and (c)

### 4.1 What it measures, and why

The fitter sees 265 ms before each Long Square step. Over 265 ms a steady
**creep** of the baseline and a slow stationary **wander** look the same; over
the 1 s step they do not -- a creep keeps going and tilts the whole step
response, which the six-parameter fit could read as I_h; a wander comes back
**[reasoning; topic doc §3.6.1-3.6.2]**. Records that span the whole sweep tell
them apart. The script reads two, per **hyperpolarising single** Long Square
sweep (the sweeps `noise_calibration.py` measures, loaded with its own
`load_for_noise` and read with its own `_pre_window_mask`) **[src]**:

| Column | Definition | Unit | Status |
|---|---|---|---|
| `tail_ms` | stored recording after the step ends, $t_{\rm last} - t_{\rm off}$ | ms | computed (archive) |
| `delta_end_mV` | mean(last 500 ms of the stored sweep) $-$ mean(pre-window), **signed**; NaN and `flag = tail_short` unless the end window lies entirely after the step | mV | computed (archive) |
| `creep_pred_mV` | pre-window slope $\times$ (centre of end window $-$ centre of pre-window): what a creep continuing at the pre-window's slope would give | mV | computed (archive) |
| `end_gap_ms`, `end_slope_mV_per_s` | where the end window starts after the step; its own slope (residual rebound) | ms; mV/s | computed (archive) |
| `allen_pre_vm_mv`, `allen_post_vm_mv` | the Allen's mean voltage over the 500 ms before the stimulus, and over the last 500 ms before the last non-NaN sample of the recording | mV, recorded (not LJP-corrected) | computed by the Allen (IPFX); pulled from the public API |
| `allen_vm_delta_mv` | $\lvert$`pre_vm_mv` $-$ `post_vm_mv`$\rvert$; the Allen sweep QC passes it below 1 mV | mV, $\ge 0$ | computed by the Allen **[src: IPFX 2.1.2 `current_clamp_sweep_qc_features`, `measure_vm_delta`, `epochs`; KB: white paper criterion]** |
| `allen_slow_noise_rms_mv` | SD (ddof 0) of the same 500 ms pre-stimulus window | mV | computed by the Allen |
| `allen_pre_offset_mV` | (pre-window mean $-$ LJP) $-$ `allen_pre_vm_mv`: the join check | mV | computed |

The archive voltages are LJP-corrected as recorded + `ljp_correction_mV`
(+14 mV for the Allen cells; the loader's constant reads "add to recorded Vm
to get true membrane V") **[src]**; the Allen fields are not. `vm_delta_mv` is
unsigned; the **signed** Allen change is `allen_post_vm_mv - allen_pre_vm_mv`
**[src: IPFX]**, computed in the paste-back line of §4.4. The query is the one
AllenSDK's `CellTypesApi.get_ephys_sweeps` sends,
`model::EphysSweep,rma::criteria,[specimen_id$eq<ID>],rma::options[num_rows$eq'all']`
on `http://api.brain-map.org/api/v2/data/query.json` **[src: AllenSDK 2.16.2;
smoke L4 asserts the URL]**, built with the standard library because the
cluster env has no AllenSDK; urllib's default opener honours `http_proxy`
**[run]**. Whether `vm_delta_mv` is populated for our sweeps: a summarising
fetch of the live API on 2026-10-01 returned the record of one hyperpolarising
Long Square sweep of 526785799 with the field filled in **[repo; no number
used]**. The same fetch, asked for a table, first put `peak_deflection` under
`vm_delta_mv`: **never take numbers from a summarising fetch of this API**;
the script reads the raw JSON.

### 4.2 How to read it, per cell

Each column above is one number computed from one sweep. The thresholds below
are properties of distributions (what a stationary wander or a continuing
creep would produce), computed by simulation in the topic doc, §3.6.7
**[reasoning; run]**.

| Cell | Wander null for `allen_vm_delta_mv` (two 500 ms windows, centres $\ge$ 1.5 s apart) | Continuing creep | Reading |
|---|---|---|---|
| 526785799 | median 0.15-0.19 mV, 90th percentile $\le$ 0.47 mV, essentially never > 1 mV | 0.7 mV/s gives $\ge$ 1.05 mV | **decisive**: well below 1 mV = the trend does not continue (wander, or a creep that saturates or reverses); near or above 1 mV, with signed change agreeing in sign with `creep_pred_mV`, = it continues |
| 571430815 | median 0.37-0.70 mV, 90th percentile 0.91-1.71 mV, > 1 mV in 7-34 % of sweeps | 2.4 mV/s gives $\ge$ 3.6 mV | only large values decide: up to about 2 mV fits a wander; well above 3 mV indicates a creep |

- **Sign and size.** For a creep, `delta_end_mV` (and the signed Allen
  change) share `creep_pred_mV`'s sign in every sweep and are of similar
  size. ~~A wander leaves the sizes unrelated; its signs agree with
  `creep_pred_mV` about half the time only if its correlation time is much
  shorter than the spacing between the windows (set by `tail_ms`), and more
  often if it is slower (note 2's 0.3-2 s scenarios for 571430815), because a
  slow wander keeps the pre-window's direction for a while. Sign agreement
  alone is therefore weak evidence; size against `creep_pred_mV` decides
  **[reasoning]**.~~ **[corrected 2026-10-01 (later)]** A wander leaves the
  sizes unrelated, and with the archive's windows its signs agree with
  `creep_pred_mV` in 0.49-0.53 of sweeps whatever its timescale (stationary
  OU, $\tau$ from 37 ms to 30 s, and a random walk; Monte Carlo, 20000 sweeps
  each), because the later change is dominated by the return towards the
  cell's mean level, which is unrelated to the local slope **[run]**. Sign
  agreement near 1 is therefore strong evidence for a continuing creep and
  agreement near 0.5 strong evidence against one, but the sign says nothing
  about the wander's timescale. `sign_agree` in the per-cell CSV counts
  agreements out of `n_valid`.
- ~~**The archive's own null is not computed yet [open].** The nulls above are
  for the Allen windows. `delta_end_mV` uses the 265 ms pre-window and a
  spacing set by `tail_ms`, so its null is a little wider; compute it from the
  real `tail_ms` before reading `delta_end_mV` against a threshold.~~
  [superseded 2026-10-01 (later): every tail is 6.000 s, the windows' centres
  are 6.89 s apart, and the archive-side nulls are in §4.6]
- **Rebound.** The end window sits after a hyperpolarising step; I_h
  deactivation and the rebound can bias its mean when the tail is short -- the
  case the Allen white paper flags **[KB]**. Read `end_gap_ms` and
  `end_slope_mV_per_s` beside `delta_end_mV`.
- **Join check.** The median $\lvert$`allen_pre_offset_mV`$\rvert$ should be a
  few tenths of a mV (the Allen's pre window is 500 ms and starts about 230 ms
  before our first stored sample, ours is 265 ms) **[reasoning]**. About 28 mV
  would mean the archive's LJP sign is the opposite of what the loader
  assumes: the archive half is unaffected, the join check is not.
- **Cohort.** The cohort's median trend share exceeds its null (§2), so look
  at the other 16 cells' `allen_vm_delta_median_mV` and `sign_agree` too:
  modest creeps common across the cohort, a slow noise slower than 36 ms, or
  both **[log]**.

### 4.3 Commands

Laptop (Git Bash), patch saved as `~/Downloads/ih_lsqc.patch`:

```
cd ~/Towards-EEG && git checkout main && git pull && git apply --check ~/Downloads/ih_lsqc.patch && git apply ~/Downloads/ih_lsqc.patch && git status --short
```

Expect, among your own untracked files: ` M .../Ih Fit/CHANGELOG_ih_fit.md`,
` M .../Ih Fit/README.md`, `?? .../Ih Fit/ls_baseline_qc.py`,
`?? .../Ih Fit/smoke_ls_baseline_qc.py`. Then:

```
cd ~/Towards-EEG && git add "Passive Features/HPC script/Ih Fit/ls_baseline_qc.py" "Passive Features/HPC script/Ih Fit/smoke_ls_baseline_qc.py" "Passive Features/HPC script/Ih Fit/README.md" "Passive Features/HPC script/Ih Fit/CHANGELOG_ih_fit.md" && git commit -m "Ih Fit: ls_baseline_qc -- LS baseline across the sweep, archive tail + Allen QC features (D-017 a,c); smoke L1-L10" && git push
```

Expect `4 files changed, 904 insertions(+)`.

Cluster, login node. `conda activate` is kept out of the `&&` chain because
on this cluster it can return non-zero while activating correctly (seen with
`spine_env`; `TEEG_HPC_paths_reference.md` §9) **[user; run, cluster]**:

```
conda activate prova; cd /davinci-1/home/ldellamea/TEEG/Towards-EEG && git pull && cd "Passive Features/HPC script/Ih Fit" && python smoke_ls_baseline_qc.py 2>&1 | tail -3
```

Expect the last line `SMOKE ls_baseline_qc: 10/10 passed`. Then, same folder
and shell:

```
module load proxy && python ls_baseline_qc.py --group-dir "/davinci-1/home/ldellamea/Human Neurons Fitting/L3_exc" --out lsqc_L3.csv --allen fetch > lsqc_L3.log 2>&1; echo "EXIT=$?"; tail -4 lsqc_L3.log
```

Expect `EXIT=0` and, at the end of the log:

```
[lsqc]  18/18 specimen_<id>  <K> sweep(s) | tail <a>-<b> ms | delta_end median ... | creep_pred median ... | Allen <K>/<K> matched
[lsqc] archive: <N> sweep(s) in 18 cell(s); tail after the step: min <x> ms, median <y> ms; end window (500 ms) usable in <n>/<N> sweeps
[lsqc] allen: mode fetch; <m>/<N> sweep(s) matched by sweep number; median |pre-window offset| <z> mV (LJP removed); cache allen_ephys_sweeps
[lsqc] wrote lsqc_L3.csv and lsqc_L3_cells.csv
```

Seconds per cell plus one HTTP request per cell; no simulation **[src]**.
Outputs land in the Ih Fit folder, untracked: `lsqc_L3.csv`,
`lsqc_L3_cells.csv`, `lsqc_L3.log`, `allen_ephys_sweeps/ephys_sweeps_<id>.json`.
Do not commit them.

### 4.4 What to paste back

```
tail -4 lsqc_L3.log; cat lsqc_L3_cells.csv
```

```
python -c "import pandas as pd; d=pd.read_csv('lsqc_L3.csv'); d['allen_signed_mV']=d['allen_post_vm_mv']-d['allen_pre_vm_mv']; c=['specimen_id','sweep_number','amplitude_pA','tail_ms','end_gap_ms','delta_end_mV','creep_pred_mV','end_slope_mV_per_s','allen_vm_delta_mv','allen_signed_mV','allen_pre_offset_mV','flag']; print(d[d.specimen_id.isin([526785799,571430815])][c].round(3).to_string(index=False))"
```

### 4.5 If the network or the archive half fails

| Symptom | Meaning | Do |
|---|---|---|
| `[lsqc] 2 consecutive network error(s): no more downloads ... Did you run 'module load proxy'?` | no route to `api.brain-map.org` | re-run after `module load proxy`; the archive half is complete either way |
| `allen: mode fetch (network disabled); 0/<N> matched` | the guard tripped | as above |
| still no network | -- | `python ls_baseline_qc.py --group-dir ... --out lsqc_L3.csv --print-urls`, open each URL in a browser, save under the printed name in `allen_ephys_sweeps/`, then re-run with `--allen cache` |
| `<n>/<N>` usable much below `<N>` | short tails (`flag = tail_short`) | the archive half cannot answer for those sweeps; the Allen half still can |
| `ARCHIVE ERROR` on a cell | unreadable `metadata.json` or `ls_sweeps.npz` | recorded, the run continues; paste the line |
| `mode fetch (network disabled); 0/104 matched` **even after `module load proxy`** (seen 2026-10-01 on `dvlogin02`) | the login node's proxy did not carry urllib's plain-HTTP request; cause not read yet | `grep -m1 "consecutive network" lsqc_L3.log; env \| grep -i _proxy`, then try HTTPS: `curl -sS -m 20 -o /dev/null -w "%{http_code}\n" "https://api.brain-map.org/api/v2/data/query.json?q=model::Specimen,rma::criteria,%5Bid\$eq526785799%5D"`; on `200`, re-run with `--allen-base-url https://api.brain-map.org`. Optional since §4.6: the archive half already measures what `vm_delta_mv` would, signed and on our own pre-window |

### 4.6 Results of the first run (L3_exc, 2026-10-01)

Inputs: the per-cell CSV (18 cells) and the per-sweep rows of 526785799 and
571430815 pasted by the user **[run, cluster]**; read-out
`lsqc_readout/readout.py` and `sign_check.py` in the assistant's sandbox
**[run]**. Each `delta_end_mV`, `creep_pred_mV` is a number computed from one
sweep; each threshold below is a property of a distribution under a stated
law, computed exactly for window means (Gaussian) and by Monte Carlo for
medians and signs.

| Finding | Numbers | Basis |
|---|---|---|
| geometry | every stored sweep holds 6.000 s after the step; end window starts 5.5 s after the step; window centres 6.89 s apart | [run, cluster] |
| no continuing creep, any cell | sign agreement 53/104 (two-sided binomial p = 0.92; 0.49-0.53 expected under any stationary wander, ~1 under a creep); 526785799: \|`creep_pred_mV`\| 4.8-6.2 mV in three sweeps against \|`delta_end_mV`\| $\le$ 0.42 mV; 571430815: \|`creep_pred_mV`\| 5.3-23.9 mV against 0.30-0.81 mV; slope of `delta_end_mV` on `creep_pred_mV` over those 10 sweeps 0.005 | [run, cluster; run] |
| the two cells are wander | P(5-sweep median \|`delta_end_mV`\| $\ge$ observed) under their own 09-28 wander scenarios: 0.23-0.70 (526785799, observed median 0.288 mV), 0.15-0.80 (571430815, 0.620 mV) | [run] |
| variance at seconds, every cell **[corrected 2026-10-08: 17 or 18 of 18]** | per-cell median \|`delta_end_mV`\| 0.048-0.620 mV, cohort median 0.128 mV; under the cohort laws that per-cell median has median 0.021-0.024 mV and 99th percentile 0.046-0.056 mV; 17 or 18 of 18 cells above it, depending on the law | [run, cluster; run] |
| where that variance can sit | an extra OU on top of $\hat\theta_{\rm sim}$ matching the cohort median needs $\sigma$ 0.142, 0.189, 0.293 mV at $\tau$ 3, 10, 30 s; at $\tau$ 0.3 or 1 s it would raise the cohort's 265 ms window SD from the measured 0.0594 to 0.098 or 0.072 mV. If it is stationary noise, it sits at $\tau \gtrsim$ 3 s | [run; reasoning] |
| why its shape matters | SD of the mean baseline error over the 1 s step (step mean $-$ pre-window mean): 0.031 mV from the 37 ms component; 0.049-0.070 mV from a seconds-scale OU matched to the 6.89 s change; 0.018 mV from a linear tilt matched to the same change | [run] |
| hint, not significant | 526785799: `delta_end_mV` rises with step size (Spearman 0.90, p = 0.04, 5 sweeps, uncorrected); amplitude and recording order are collinear by protocol (from $-110$ pA upwards in 20 pA steps **[KB]**), so a step-induced after-effect and a drift over the protocol cannot be separated; per-cell median `delta_end_mV` > 0 in 12/18 cells (p = 0.24) | [run, cluster; run] |

**What remains open.** The shape of the seconds-scale variance (OU of
seconds, random walk, or per-sweep tilt) and whether part of it is caused by
the step. The SS pre-pulse series constrains both: its variogram from 0.2 to
3.8 s [corrected 2026-10-20: 0.2005 to 7.218 s] separates a saturating OU, a random walk (variance growing with the lag)
and a tilt (variance growing with the square of the lag), at rest and without
a long step, and set beside the LS tail it bounds what the step adds
**[reasoning]**. The archive-side `delta_end_mV` of the depolarising subthreshold sweeps,
recorded after the hyperpolarising ones, would separate amplitude from order
**[reasoning]**; `ls_baseline_qc.py` reads hyperpolarising sweeps only.

## 5. Implementation of D-017, once (a)-(d) are settled

Specified in D-017 "What it implies" **[log]**; none of it written. [updated 2026-10-23: the noise-fit code lives in `Ih Fit/specs/SPEC.md` as Blocks 1-7 -- Blocks 1-2 on `main` (`e8ed84e`, `36ba658`), Blocks 3-4 (`noise_windows.py`, `noise_cell_stats.py`, `noise_fit.py`) delivered as patches on `main` `64fb8c5` and verified with `git am` on a fresh checkout, Block 4b (the per-sweep stage of D-076 (i)) and Blocks 5-7 (generator, driver and PBS, record) not yet written; the list below is the 2026-10-01 sketch, superseded by the SPEC where they differ.]

**Update 2026-10-01 (later), proposal, not decided.** §4.6 shows that the
law (D-017.2) with one slow component cannot carry the variance at seconds.
The per-cell fit should take, besides the window statistics, the LS tail
statistic (per-cell median of the absolute `delta_end_mV` over 6.89 s, from `lsqc_L3.csv`)
and the SS pre-pulse variogram as targets, and the seconds-scale part should
enter either as a second OU or as the per-sweep tilt $b$ of (D-017.1), the
form chosen from the SS variogram. **[decided 2026-10-08, D-034: the tilt]** Calibrating $b$ from pre-window slopes is
excluded by §4.6.

1. `noise_calibration.py`: a per-cell law fit, $\theta_{\rm noise}(c)$, by
   simulated medians of its own estimators (if (d) goes that way), written
   beside the existing table with its cost and flags.
2. Manifest columns under names that cannot collide with the Square
   Subthreshold columns of D-014 (`noise_ss_*`): e.g. `noise_fast_sigma_mV`,
   `noise_fast_rho_lag1`, `noise_slow_sigma_mV`, `noise_slow_tau_ms`, plus the
   drift-law columns that (c) defines; back-filled so an old manifest
   generates exactly what it generated before.
3. `synthetic_ground_truth.add_recording_noise`: the OU term, off for old
   manifests (bit-identical); the docstring's "electrode/seal drift"
   corrected (unsupported, not wrong; note 2 item (3)).
4. `gen_from_manifest`: the twin's `ls_onset_ms` = `ls_pre_ms` + 5 ms.
5. The two-component law is the DEFAULT for every cell in the noise table;
   single AR(1) survives only as back-fill and as a labelled comparison.
6. Smoke: closure (inject a known law, measure it with `noise_calibration.py`,
   fit it, recover it inside the Monte Carlo band at the real window lengths
   and sweep counts); an assertion of the default; D-015's queued check that
   the injected noise is stationary across the sweep.
7. Then the gate, from the Ih Fit folder: `qsub -v DRY_RUN=1
   submit_ih_recovery.sh`, a pilot (`MAX_CELLS=2,N_CALLS=20,N_INITIAL=10,
   RUN_TAG=pilot`; its verdict is not a verdict), then `qsub
   submit_ih_recovery.sh` (README, "Stage 7").

Delivery convention for this user: a git patch against the current `main`,
`git apply --check` on a fresh clone, smoke on that clone, ASCII-only Python
and LF-only shell, one chained command per step, expected output stated as
lines that must appear.

## 6. Paths and environment

| What | Where | Basis |
|---|---|---|
| laptop clone | `~/Towards-EEG` (Git Bash) | [user, 09-23] |
| cluster clone | `/davinci-1/home/ldellamea/TEEG/Towards-EEG` | [user, 09-23] |
| this code | `<clone>/Passive Features/HPC script/Ih Fit` | [user] |
| archive | `/davinci-1/home/ldellamea/Human Neurons Fitting/<GROUP>/specimen_<id>/` (`metadata.json`, `ls_sweeps.npz`, `reconstruction.swc`); L3_exc = 18 cells | [user; run, cluster] |
| run B (baseline) | `/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs/dt01ls60/<GROUP>/phase2_results.csv` (run A: `dt01ls60ph25`) | [user, 09-23] |
| conda env | `prova` (README "First run on the cluster"; `IH_ENV` default of both submit scripts) | [src] |
| network | `module load proxy` (sets `http_proxy`) | paths reference §9 [run, cluster] |
| noise tables | `noise_L3.csv` (09-23), `noise_L3_v2.csv` (09-28), written by `noise_calibration.py`; folder not recorded | [user; open] |
| gate logs (added 2026-10-20) | `~/ih_fit_gate_logs/<date_time>/summary.txt` plus one log per check (`$GATE_LOG_DIR` overrides), written by `cluster_gates_noise_fit.sh`, never in the repository; the run of 2026-10-20 is `/davinci-1/home/ldellamea/ih_fit_gate_logs/20261020_175303/` | [src; cluster] |
| archive origin (added 2026-10-20) | Google Drive folder `L3_exc` (the Colab writer's `ARCHIVE_DIR`, `MyDrive/Colab Notebooks/Allen_Intitute_Data/<GROUP>`), specimen folders written 2026-06-19; beside `ss_pulses.npz` and `metadata.json` each holds the trace-QC outputs `trace_qc_results.csv` (one row per pulse entering the QC), `trace_qc_summary.json`, `trace_qc_meanresid.csv`, `trace_qc_selfcal.csv` and figures. Whether the cluster copy also carries `trace_qc_results.csv` is **not checked** (G12 lists the npz and `metadata.json` only) [checked 2026-10-21: all 18 L3_exc cells carry it on the cluster, its rows = `n_pulses_raw` = `n_pulses_ampdur_qc`] | [data, Drive listing; cluster 2026-10-21] |

## 7. Traps

- **Two copies of two docs.** The live decision log is the root one (§0). The
  paths reference also exists twice (`TEEG_HPC_paths_reference.md` at root,
  newer, and `claude/TEEG_HPC_paths_reference.md`); both describe the H01
  pipeline and the `spine_env` env, not this code, whose paths are in §6 and
  in the README.
- **`noise_ss_*` means Square Subthreshold** (0.5 ms pulses, 20 per sweep,
  200 ms apart [corrected 2026-10-20: 37 per sweep, 200.5 ms onset to onset in
  the archive's cells; data]), not the Allen *Short Square* (3 ms pulses up to threshold)
  **[KB; note 2 item (1)]**.
- **Statistic is not parameter.** A window SD or autocorrelation of a slow
  component depends on the window length (demeaning hides slow variance) and
  the autocorrelation estimator is biased; the law is inferred through the
  estimators, never read off them (topic doc §3.4).
- **Status is not level.** `vm_delta_mv`, `delta_end_mV`, `creep_pred_mV` are
  numbers computed from one sweep; the wander and creep thresholds of §4.2 are
  properties of distributions. A per-cell median over 5 sweeps has its own,
  narrower null.
- **Summarising fetches** of `api.brain-map.org` mislabel fields (§4.1).

## 8. Older open items, outside the noise model

| Item | State | Basis |
|---|---|---|
| `regression_passive_identity.py` on a real cell (README's first-run step on the cluster) | result never reported [updated 2026-10-20: PASS on 508282493 as G11 of `cluster_gates_noise_fit.sh` -- loss max\|dL\| = 0 over 20 points, `x` and `fun` identical, ref 0.172357] | [open] [cluster, 2026-10-20] |
| `check_dep_sweeps.py` on L3_exc: depolarising sweeps the new screen reports as action potentials, to be inspected by eye | 7 such sweeps per the 09-28 session record; not re-verified here | [open] |
| Stage-7 tolerances (D-011) | the assistant's proposal, not a user decision; every per-cell error is written, so a tolerance changes without a re-run | [log] |
| cohort size 20; L2_exc as well; a higher-noise cohort | never justified by a power calculation; not decided | [log] |
| re-run Stage 7 on the diameter-corrected SWCs (D-013) | recommended; the SWCs depend on the diameter workstream (D-018) | [log] |
| D-012's real-data consistency check (run B's `noise_sigma_mV` between SS $\hat\sigma/\sqrt N$ and LS $\hat\sigma$) | never run | [log] |
