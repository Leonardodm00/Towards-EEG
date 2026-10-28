# TEEG -- noise-fit chat handoff (2026-10-28)

| Date | Change |
|---|---|
| 2026-10-28 | Created at the user's request when the noise-fit chat (Ih Fit Stage-7 noise law, `Ih Fit/specs/SPEC.md` Blocks 1-7) changed chat, right after **D-095** (the new SS window-start rule). The history of the thread stays in `claude/TEEG_Stage7_noise_handoff_2026-10-01.md`; this file holds the current state and the next steps. |
| 2026-10-28 (later) | **D-097** (user, 10:24): the assistant pushes its own work, and every document is also committed to the repository so any chat can read it. Everything of §1 pushed by the assistant (`main` at `cd32dad`); the roadmap updated at the user's word (D-095's revise marks on N3 and N4; page v44); this file committed at `Passive Features/HPC script/Ih Fit/docs/` (project knowledge is full: the repository is its only home). New section "First: the user's answers" (the user asked that the handoff say the tilt question is still theirs to answer). Superseded statements marked in §0, §1, §2, §3, §5, §6; next free IDs D-098, I-005. The commit-identity row no longer spells out the user's e-mail address (the repository is public; the identity is in the clone's git configuration). |
| 2026-10-28 (later, 2) | **D-100** (user, about 18:37): "Let's do A and B for the 2 sweep cells." -- for the 14 cells with two SS sweeps the twins' tilt SD is taken both per cell (option (a)) and as one cohort value (option (b)); partial pooling (c) not taken. Question 1 of "First: the user's answers" answered; the assistant's reading (D-100.1-.4) and the entry's open points (a)-(f) to confirm before Blocks 5-6 are specified. Question 2 (D-095's nine points) recapped to the user, still open. The checks behind the day's explanations committed to `Ih Fit/dev/2026-10-28_tilt_explanation/` (re-run there, logs reproduced); SPEC §8's tilt question marked. Marked: the top section, §0 (next free IDs D-101, I-005), §3 (the tilt row; Blocks 5-7), §5 (the new package). |
| 2026-10-28 (later, 3) | The user (19:35): "I think you have misunderstood" -- **D-100 withdrawn** (the assistant's misreading of "A and B"); **D-102**: each twin sweep carries the realised tilt of its own real sweep, no statistic of the tilt computed (open: the LS sweeps' tilts, the fit's simulator). **D-103**: D-095's points answered (median window, per sweep and polarity, point value, maximum, one exponential on [5, 200) ms, 80 ms floor and fallback, flagged); open: point 9 and the fallback's trigger. The formula for $\kappa_{\rm tr}$ shown, with `check_kappa_tr_start.py` (new, `Ih Fit/dev/2026-10-28_window_start/`). Marked: the top section, §0 (next free IDs D-104, I-005), §2 (the answers), §3, §4 (a row), §5. |
| 2026-10-28 (later, 4) | **D-104** (user, 19:53): the fallback applies when the window would be shorter than 50 ms (and with no valid group); a failed group, or one with fewer than 5 windows (PROVISIONAL), is flagged and left out. Roadmap allen-mtg-fit updated at the user's word (`a945a7a`: notes and revise marks for D-102 to D-104 on N3, N4, N5, T8). `check_fallback_window.py` added. Marked: the top section, §0 (next free IDs D-105, I-005), §2 (the answers), §4 (a row), §5. |

**Basis tags.** **[src]** read from source; **[run]** executed in the assistant's sandbox (*cluster* where marked);
**[user]** the user's decision or report; **[KB]** project knowledge, full text; **[log]** the decision log's text;
**[reasoning]** the assistant's argument, not checked; **[open]** not decided or not verified.

## First: the user's answers (ask before anything else)

The user asked (2026-10-28, 10:24) that this handoff say they still have to answer the open question from before. Ask both at the start of the new chat, one exchange, before writing any logic:

1. **The tilt $\sigma_b(c)$ of a 2-sweep cell** (14 of the 18 L3_exc cells): (a) per cell, as D-034 reads it; (b) one cohort value for all cells; (c) partial pooling across cells, DerSimonian-Laird on $\log\hat\sigma_b^2$ -- the assistant recommends (c). Put to the user on 2026-10-23 and 2026-10-26, unanswered; why it matters: at $\sigma_b$ = 0.1 mV/s a 2-sweep cell's $\hat\sigma_b/\sigma_b$ spans 0.00 to 1.96 or 2.04 (95 %, $K$ = 63 or 40; §4), so its error goes straight into the twins' tilt, while it moves the fitted law's coordinates $\hat u$ by at most 0.18 of their SD (linearised at $\theta_{\rm cf}$, the fixture layout) **[run; log: 2026-10-23 (later, 4), D-034 note of 2026-10-26]**. Study material: `Ih Fit/dev/2026-10-26_tilt_sigma_b/TEEG_tilt_sigma_b_2026-10-26.md` (§3.9 the three options); roadmap T8 is open because the user has not studied it yet. Whatever the answer, it goes into Block 4b (§3) and the decision log. [answered 2026-10-28 (later, 2): **D-100** -- (a) and (b) for the 2-sweep cells, (c) not taken. To confirm (the entry's open points): two labelled variants of twins per 2-sweep cell, or both values computed and the one the twins receive chosen later; the default, or a pass required under both; the cohort value pooled over all 18 cells ($S_{\rm eff}$-weighted mean of the unclipped estimates, clipped at 0) or over a subset; the multi-sweep cells under (a) only; each real cell's own fit unchanged] [corrected 2026-10-28 (later, 3): D-100 was the assistant's misreading and is withdrawn. The answer is **D-102**: each twin sweep carries the realised tilt of its own real sweep, $\hat b_{c,s}$, and no $\sigma_b(c)$ is computed for the twins. Still to ask: the Long Square sweeps' tilts (proposal: `delta_end_mV` / 6.89 s) and whether the per-cell fit's simulator uses the same $\hat b_{c,s}$ (proposal) or the data are detrended]
2. **D-095's nine open points** (§2's table): mean or median, the grouping, "statistically", how $t_{\rm SS}(c)$ is formed, the decay model, the 80 ms floor, the 190 ms fallback, $\kappa_{\rm tr}$, the flag for a failed fit. [answered 2026-10-28 (later, 3): **D-103** -- 1 median, 2 per $(s, p)$, 3 the point value, 4 the maximum, 5 as proposed (one exponential, free baseline, [5, 200) ms), 6 the 80 ms floor kept, 7 80 ms also the fallback, flagged, 8 the formula shown ($\kappa_{\rm tr}$ still 0.1, provisional); still open: 9 and the fallback's trigger] [updated 2026-10-28 (later, 4): 9 and the trigger answered by **D-104** -- a failed group flagged and left out; the fallback when the window would be shorter than 50 ms]

## 0. Read first

| Doc | What for |
|---|---|
| `TEEG_decisions_and_ideas_log.md` (project root) | **D-095** (this handoff's first task), and the decisions it touches: D-034, D-051, D-076; the noise fit's others: D-017, D-033, D-048 to D-050, D-056, D-072, D-073; the workflow: D-082 (roadmap; its hold of 2026-10-26, 16:48), D-083 (Stage 7 before any real-cell six-parameter fit), D-091 (a new SPEC block enters its roadmap in the same turn). **Next free IDs: D-096, I-005** [corrected 2026-10-28 (later): **D-098, I-005** -- D-096 is the diameter chat's (09:58), D-097 this chat's (push yourself; every document also in the repository)] [corrected 2026-10-28 (later, 2): **D-101, I-005** -- D-098 (15:01) and D-099 (17:07) the diameter chat's, D-100 this chat's (the tilt variants)] [corrected 2026-10-28 (later, 3): **D-104, I-005** -- D-101 the diameter chat's (18:17), D-102 (the realised tilts; D-100 withdrawn) and D-103 (D-095's points) this chat's] [corrected 2026-10-28 (later, 4): **D-105, I-005** -- D-104 this chat's (the fallback's trigger, a failed group)] -- the diameter chats logged D-084 to D-094 between 2026-10-26 and 2026-10-28: re-read the log's last changelog row immediately before every write |
| `Passive Features/HPC script/Ih Fit/specs/SPEC.md` on main + the five patches of §1 [corrected 2026-10-28 (later): all on `main` since the assistant's push] | the noise-fit spec; **§2.2 (window geometry, transient test) is stale since D-095** |
| `claude/TEEG_Stage7_noise_handoff_2026-10-01.md` | the thread's history (one changelog row per step), paths (§6), traps (§7) |
| `docs/project/README.md` (repository root; added 2026-10-28 (later)) | the byte copies of the project's documents in the repository (D-097): which are copied, which are not, and the rule (the project copy first, the repository copy in the same turn) |
| `claude/TEEG_Stage7_noise_model_2026-09-28.md` | the noise-model topic doc; §3.7 is the enlarged SS windows and the transient test, eq. (27) = $\bar r_{\rm odd}$ |
| `TEEG_tilt_sigma_b_2026-10-26.md` (in the user's `TEEG_tilt_sigma_b_2026-10-26.zip`, **not** in project knowledge) [corrected 2026-10-28 (later): in the repository, `Ih Fit/dev/2026-10-26_tilt_sigma_b/`] | the study document on the tilt $\sigma_b(c)$, five figures, checks check01-check05 |

## 1. Repository and delivery state

| Item | State | Basis |
|---|---|---|
| GitHub `main` | `64fb8c5` "roadmap: allen-mtg-fit created"; Blocks 1-2 (`e8ed84e`, `36ba658`) and the gate tools (`ca028ea`) are on it | [run: `git fetch`, 2026-10-28 10:0x] [corrected 2026-10-28 (later): `cd32dad`, pushed by the assistant -- `0dfe8bf`, `65c659b`, `76577d2` (Block 3), `a95e618` (Block 4), `033e3bf`, `8d8d602`, `4930ba1`, `53b2713` (roadmap: D-095 on N3, N4), `d2ffcb4` (documents), `cd32dad` (roadmap rendered); the copies of the log and the handoffs follow in the next commit] |
| Blocks 3-4 and three spec commits | **only as patches**: `noise_fit_patches_2026-10-28.zip`, five patches on `64fb8c5` -- 0001-0002 spec (SS sweep structure; D-072/D-073), 0003 Block 3 (`noise_windows.py`, `noise_cell_stats.py`, smoke W1-W10), 0004 Block 4 (`noise_fit.py`, smoke F1-F10), 0005 spec (the 2026-10-23 findings sharpened; spec and comments only) -- in the user's `Downloads` and sent in the chat; supersedes `block34_patches_2026-10-23.zip`. Verified: `git am` on a fresh checkout of `64fb8c5`, the folder equal to the assistant's branch file for file, QUICK gates 8/8, `smoke_noise_fit` 10/10. **Pending: the user applies and pushes** (commands in the zip's `APPLY.txt`) [corrected 2026-10-28 (later): pushed by the assistant (D-097); nothing to apply] | [run, 2026-10-28] |
| roadmap patches | `roadmap_t8_2026-10-26.zip`: 0001 S7 restated (D-083), N4b added, I4 done; 0002 T8 back to open (the tilt theory to study). On `64fb8c5`; applies cleanly after the five noise-fit patches, `roadmap.py check` ok. **Pending push** [corrected 2026-10-28 (later): pushed, `8d8d602`, `4930ba1`] | [run, 2026-10-28] |
| the assistant's pushes | refused twice by the session's safety classifier: deliver patches, never work around it [corrected 2026-10-28 (later): D-097 -- the user asked the assistant to push itself, and the pushes of 2026-10-28 went through; never work around a refusal, say so instead] | [log of this chat; user] |
| commits | authored as the user: the clone's local `user.name` and `user.email` are set to the user's identity (check with `git config user.name`; never write the address into a document, the repository is public); trailers as the session's reminder gives them | [user] |
| cluster | last FULL gate run 2026-10-20 (13/13, SETTLED, `main` at `ca028ea`); Blocks 3-4 never run on the cluster nor on a real cell | [run, cluster] |
| roadmap page | artifact `Fe6Zqjr5iq3XAQZfrTspqB`, collection `roadmaps`, doc `towards-eeg:allen-mtg-fit`: v27 written 2026-10-26 from `e0a0a36` (the two roadmap patches); the "Roadmap sync" task redraws it from `main` until the user pushes [corrected 2026-10-28 (later): v44 written by this chat from `cd32dad`; the sync task keeps it current from `main`] | [run] |

## 2. D-095 -- the SS window start, to implement

**The user's words (2026-10-28, 09:55).** Mean windows per polarity over [-10, 200) ms, "the mean (or median) values
across the $K_p$ windows"; on them, "at which point statistically the exponential response decay (from the mean
window) is within the $\kappa_{\rm tr}\hat\sigma_{\rm SS}(c)$. This will be our $t_{\rm SS}(c)$"; the real windows
cut to $[t_{\rm SS}(c), 200)$ minus the mean window on the same range "(just to be sure)"; "No binning, per polarity
and per sweep" **[user]**. Full entry with the assistant's reading: decision log D-095.

**The assistant's reading** (D-095.1-.3) **[log]**:

$$\bar x_{s,p}(t) = \frac{1}{K_{s,p}}\sum_{k\in(s,p)} x_k(t)\quad\text{for each } t\in[-10,200)\ {\rm ms},\qquad
\tilde x_k(t) = x_k(t) - \bar x_{s,p}(t),\ \ k\in(s,p),\ t\in[t_{\rm SS}(c),200)\ {\rm ms},$$

with $s$ the sweep, $p \in \{{\rm dep}, {\rm hyp}\}$, $K_{s,p}$ the stored windows of that sweep and polarity;
$t_{s,p}$ the time at which the fitted exponential decay of $\bar x_{s,p}$ falls within $\kappa_{\rm tr}\hat\sigma_{\rm
SS}(c)$; $t_{\rm SS}(c)$ formed from the $t_{s,p}$. $\hat\sigma_{\rm SS}(c)$ is unchanged: median over the cell's
windows of the SD (ddof 1) of each window's $[-10, 0)$ ms segment, on the stored voltage (`transient_start`, lines
284-286) **[src]**. The simulator applies the same $(s, p)$ grouping to its own windows.

**Ask these before writing any logic** (scientific-coding, clarify first; each is open in D-095):

| # | Question | Assistant's proposal | Why it matters |
|---|---|---|---|
| 1 | mean or median in $\bar x_{s,p}$ | mean | linear: the simulator mirrors it exactly; noise variance after it is $(1 - 1/K_{s,p})$ of the window's |
| 2 | grouping: per sweep and polarity for both the start and the subtraction (the quote also says "across the $K_p$ windows") | per $(s, p)$ for both | 10-19 windows per group in the archive's layouts against 20-175 per polarity [reasoning on the layout] |
| 3 | "statistically": the point value of the fitted decay, or a confidence bound of it (e.g. bootstrap over the $K_{s,p}$ windows, one-sided 95 %) | ask; the bound is the literal reading | the bound moves the start later |
| 4 | $t_{\rm SS}(c)$ from the $t_{s,p}$ | the maximum over groups | one window length per cell, which the per-cell medians and the binding function's $n_m$ need |
| 5 | the decay model, its fitted range, its baseline | one exponential, free baseline, fitted from about 5 ms to 200 ms | the pre-pulse level holds the previous opposite pulse's tail (D-095 comment (b)); real responses are not single exponentials (dendritic equalisation, $I_h$) |
| 6 | keep D-076 (ii)'s 80 ms floor | keep | its reason is the SQC selection leak (at most 0.09 per-cell SD from 80 ms), not the transient; it binds only for fast or small responses |
| 7 | keep D-051 (i)'s fallback at 190 ms (`ss_enlarged_none`) | keep the flag | with $\kappa_{\rm tr}$ = 0.1 the start passes 190 ms from $\tau_m \approx$ 46 ms at 113 MOhm (§4) |
| 8 | $\kappa_{\rm tr}$ (0.1, provisional since D-051) | ask, with §4's table | at 0.1 a typical cell's window is about 70 ms instead of 120 ms |
| 9 | a flag when a group's fit fails or has too few windows | flag and drop the group from the maximum | -- |

**Answers (the user, 2026-10-28, 19:35; D-103) [user].** 1: median (the per-sample median window, also the one subtracted); 2: yes; 3: the decay's point value; 4: the maximum over groups; 5: as proposed; 6: the 80 ms floor kept; 7: 80 ms is also the fallback, flagged (replacing 190 ms); 8: the formula, $t_{s,p} = \hat\tau_{s,p}\ln(\lvert\hat A_{s,p}\rvert/(\kappa_{\rm tr}\hat\sigma_{\rm SS}(c)))$ and $t_{\rm SS}(c) = \max\{80\ {\rm ms}, \max_{(s,p)} t_{s,p}\}$, shown; $\kappa_{\rm tr}$ stays 0.1. Open: 9, and the fallback's trigger (proposal: a start at or past 190 ms, or no valid group). In the plan below, `fallback_t_ms` = 80 and `t_ss_floor_ms` = 80; F2's expectation is that of a median-subtracted window (Monte Carlo). [updated 2026-10-28 (later, 4), **D-104** [user]: the fallback applies when the window would be shorter than 50 ms (a start later than 150 ms; also with no valid group, the assistant's reading), `min_window_ms` = 50; 9: a failed group, or one with fewer than `min_windows_per_group` = 5 windows (PROVISIONAL), is flagged and left out of the maximum. Still open: whether a dropped group's windows also leave the statistics.]

**Implementation plan, after the answers** (scientific-coding workflow; hpc-python-compat and hpc-git-delivery for the
patch):

| Where | Change |
|---|---|
| SPEC §2.2, Block 3 entry | rewrite the window-start paragraph for D-095 first; if written as a new block (e.g. "Block 3b"), it enters the roadmap in the same turn (`roadmap.py add block` under N3, D-091) |
| `noise_windows.py` | `transient_start` replaced (e.g. `decay_start(ss, cfg)` returning per-group fits, $t_{s,p}$, $t_{\rm SS}(c)$, flags); `subtract_mean_response` grouped by `sweep_number` and polarity; `WindowConfig`: `transient_bin_ms`, `transient_floor_z` retired, the fit's settings added, `kappa_tr` kept; `t_ss_floor_ms`, `fallback_t_ms` per answers 6-7 |
| `noise_fit.py` | `simulate_cell_windows` subtracts per $(s, p)$ (its `CellLayout.sweep_number` is there) |
| `noise_cell_stats.py` | unchanged in logic; the tilt's `window_mean_semivariogram` (same-sweep, same-polarity pairs) is untouched by D-095, since $\bar x_{s,p}$ cancels in those differences |
| smoke | W2 replaced (a planted exponential recovered; the crossing within one sample of $\tau\ln(A/(\kappa_{\rm tr}\hat\sigma_{\rm SS}))$; groups; flags); W3 zero mean per $(s, p)$; F2's expectation $\sigma_f^2(1 - 1/K_{s,p})$; mutations: grouping by polarity only, baseline from the pre-pulse level, mean of groups for the maximum |
| libraries | `scipy.optimize.curve_fit` (or `least_squares`) for the fit; `scipy.stats.bootstrap` if answer 3 is the bound; NumPy |
| columns | `noise_ss_enl_t0_ms` kept; per $(s, p)$ the fitted amplitude, time constant, baseline, $t_{s,p}$ |
| roadmap | N3's note still names the options of 2026-10-23: stale, but roadmap changes other than a new block wait for the user (D-082's hold) [corrected 2026-10-28 (later): updated at the user's word -- N3's and N4's notes name D-095, each carries a revise mark that clears when its block's code changes; a new block for D-095 enters with its SPEC entry (D-091); a theory topic for the window start is only proposed, never added unasked] |

## 3. Other open items of this thread

| Item | State | Basis |
|---|---|---|
| the tilt $\sigma_b(c)$ of a 2-sweep cell (14 of 18 cells): (a) per cell as D-034, (b) one cohort value, (c) partial pooling across cells by DerSimonian-Laird on $\log\hat\sigma_b^2$ | the user's choice pending; the assistant recommends (c) [2026-10-28 (later): **ask it first**, see the top] [answered 2026-10-28 (later, 2): **D-100** -- (a) and (b) for the 2-sweep cells; its reading to confirm] [corrected 2026-10-28 (later, 3): D-100 withdrawn (a misreading); **D-102** -- each twin sweep carries its real sweep's realised tilt] | [log: 2026-10-23 (later, 4), 2026-10-26 (later, 3), D-034 note of 2026-10-26] |
| roadmap T8 (the tilt's theory) | open: the user has not read it yet; study document delivered 2026-10-26 | [user] |
| Block 4b, the per-sweep stage (D-076 (i)) | not written; statsmodels 0.15's `combine_effects(method_re="dl")` does **not** clip $\hat\tau^2$ at zero, Block 4b must, and recompute the random-effects mean with the clipped value | [log: D-076 clarified 2026-10-26; run] |
| Blocks 5-7 (generator's law and per-sweep twins; driver, S0-S3 and PBS; the record) | not written [2026-10-28 (later, 2): D-100 adds to them -- the generator takes the tilt SD per variant (`tilt_variant` = `per_cell` or `cohort`, one manifest row and the same unit draws for both), and a cohort step after the 18 per-cell fits computes the cohort value; Blocks 3-4 must also export $\hat\sigma_b(c)^2_{\rm u}$ and the per-sweep pair counts] [corrected 2026-10-28 (later, 3): D-100's variants and cohort step are void; under D-102 the generator takes the per-sweep tilts $\hat b_{c,s}$, which Block 3 reads (least-squares slope of the window means, proposal)] | [log] |
| D-083 | Stage 7's six-parameter recovery on davinci (roadmap S7; SPEC Block 8, to specify) before any six-parameter fit of a real cell | [log] |
| real cells' pulse-to-pulse response variation ($cv$) | never measured; D-051 assumed 2 %; S3 is to measure it | [log] |
| the living handoff's older open points (its §3 (a), (e), (f), (g); §8) | unchanged | [KB] |

## 4. Numbers to carry (all at $\theta_{\rm cf}$ = (0.0380 mV, 0.175, 0.0508 mV, 36.6 ms), the archive's SS layout)

| Quantity | Value | Basis |
|---|---|---|
| D-051 (iv)'s test as written: SD of a 5 ms bin mean of $\bar r_{\rm odd}$ under noise alone, 100-200 ms | 0.30, 0.24, 0.15, 0.11 $\hat\sigma_{\rm SS}$ at $K$ = 40, 63, 199, 348; about $\sigma_s\sqrt{2/K}$; threshold 0.1 | [run: `check_findings_2026-10-23.py` A] |
| its fallback rate, noise alone | 90, 85, 68, 56 % ($\sigma_b$ = 0); 90, 86, 66, 51 % ($\sigma_b$ = 0.1 mV/s) | [run: A] |
| $\hat\sigma_{\rm SS}(c)$ | median 0.0402 mV, $\pm$1.1 % across cells ($K$ = 63); one 10 ms segment's expected variance $1.66\times10^{-3}$ mV$^2$, 87 % fast, 13 % slow (8.5 % of $\sigma_s^2$); a 0.1 mV/s tilt moves the median by less than 0.1 % | [run: `check_sigma_ss.py`] |
| RC fixture | 113 MOhm, 28.143 ms, $\pm$200 pA $\times$ 0.5 ms: peak 0.40 mV, 0.58 $\hat\sigma_{\rm SS}$ at 80 ms; L3_exc run-B group medians give $R_m C_m$ = 27.8 ms | [run; KB, annual report part 1 §6] |
| leak after the mean-window subtraction, gains $a_k = 1 + cv\,z_k$ | $\mathbb E[\hat\sigma_k^2]$ rises by $\delta_k = cv^2[(1 - 1/K_p)^2 q_k + K_p^{-2}\sum_{j\ne k} q_j]$, exact; per-cell medians move 0.57-0.64 of $\delta/(2\hat\sigma)$; at 80 ms: 0.019 $\pm$ 0.010 per-cell SD (fixture, $cv$ 0.2), 0.04 (5 $\times$ fixture, $cv$ 0.1), 0.46 (5 $\times$, $cv$ 0.3), 0.56 ($\tau_m$ 60 ms, 3 $\times$, $cv$ 0.3) | [run: `check_jitter_leak.py`] |
| D-095's start, noiseless RC, $\kappa_{\rm tr}$ = 0.1, point value of a fitted exponential with free baseline | 99, 130, 154, 170, 186, 201, 229 ms at $\tau_m$ = 20, 28.1, 35, 40, 45, 50, 60 ms (113 MOhm); 112, 130, 146, 157 ms at 60, 113, 200, 300 MOhm (28.1 ms); against the pre-pulse level instead: +3 ms at 28.1 ms, never within the window from 40 ms | [run: `check_d095_rc_start.py`] |
| D-103's start against $\kappa_{\rm tr}$ (the same RC test cell, 113 MOhm, 28.143 ms, $\hat\sigma_{\rm SS}$ 0.0402 mV; fitted exponential, free baseline, [5, 200) ms) | 110.3, 129.8, 149.3 ms at $\kappa_{\rm tr}$ = 0.2, 0.1, 0.05 (analytic 110.0, 129.5, 149.1); halving $\kappa_{\rm tr}$ delays the start by $\tau_m\ln 2$ = 19.51 ms; the start reaches 190 ms from $\tau_m$ = 61.4, 46.5, 38.1 ms | [run: `check_kappa_tr_start.py`] |
| D-104's fallback on the same RC test cell (start later than 150 ms, the window under 50 ms) | from $\tau_m$ = 34.0 ms at 113 MOhm or $R_{\rm in}$ = 234 MOhm at 28.1 ms ($\kappa_{\rm tr}$ = 0.1); 28.4 ms or 117 MOhm (0.05); 43.2 ms or 467 MOhm (0.2); the L3_exc run-B group medians give $R_m C_m$ = 27.8 ms | [run: `check_fallback_window.py`; KB: annual report part 1 §6] |
| SQC selection leak (the 80 ms floor's reason) | $-$0.17 to $-$0.19 per-cell SD at 40 ms; at most 0.09 from 80 ms; 0.08 at 120 ms | [src: SPEC §8] |
| tilt (T1) spread, $\sigma_b$ = 0.1 mV/s | 95 % range of $\hat\sigma_b/\sigma_b$: 0.00-1.96 or 2.04 (2 sweeps, $K$ = 63 or 40), 0.41-1.59 (6), 0.55-1.38 (10); below about 0.04-0.05 mV/s a 2-sweep cell cannot tell $\sigma_b(c)$ from zero | [run; log: D-034 note 2026-10-26] |

## 5. Files delivered

| File | Where | Content | State |
|---|---|---|---|
| `noise_fit_patches_2026-10-28.zip` | user's `Downloads`; sent 2026-10-28 | the five patches of §1, `APPLY.txt`, `SHA256SUMS` | to apply and push [corrected 2026-10-28 (later): pushed; do not apply] |
| `roadmap_t8_2026-10-26.zip` | user's `Downloads`; sent 2026-10-26 | two roadmap patches | to apply and push [corrected 2026-10-28 (later): pushed; do not apply] |
| `TEEG_window_start_checks_2026-10-28.zip` | user's `Downloads`; sent 2026-10-28 | every check of §4 with its output (failed runs kept), `README.txt`; scripts need `IH_FIT_DIR` set to an Ih Fit folder with Blocks 1-4 | reference [corrected 2026-10-28 (later): in the repository, `Ih Fit/dev/2026-10-28_window_start/`, where the scripts find the Ih Fit folder themselves] [updated 2026-10-28 (later, 3): `check_kappa_tr_start.py` with its output added there] [updated 2026-10-28 (later, 4): `check_fallback_window.py` with its output added there] |
| `TEEG_tilt_sigma_b_2026-10-26.zip` | user's `Downloads`; sent 2026-10-26 | the tilt study document, figures, scripts (its copy of `check_findings_2026-10-23.py` has the sandbox path hard-coded) | reference [corrected 2026-10-28 (later): in the repository, `Ih Fit/dev/2026-10-26_tilt_sigma_b/`, paths relative; `results/banks/` left out, regenerated by `make_banks.py`] |
| the checks of the day's explanations (added 2026-10-28 (later, 2)) | `Ih Fit/dev/2026-10-28_tilt_explanation/` | check06 (PM against DL inside option (c)), check07 (D-095's later start against the tilt: D-100's independence), check08-check11 (semivariogram, nugget, second-order stationarity), `fig_nugget.py`, the overview figure, `README.txt`; check06 and check07 need the tilt study's banks (`make_banks.py`) | reference; re-run there 2026-10-28, logs reproduced |
| this handoff | `Passive Features/HPC script/Ih Fit/docs/TEEG_noise_fit_handoff_2026-10-28.md` (added 2026-10-28 (later)) | the current state and next steps | the repository is its only home (project knowledge full) |
| copies of the project's documents | `docs/project/` at the repository root (added 2026-10-28 (later)) | the decision log, the living handoff, the noise-model topic doc, the staged plan; README with the rule | refreshed in the same turn as each write to project knowledge (D-097) |
| superseded | -- | `block34_patches_2026-10-23.zip`, `block3_patches_2026-10-23.zip`, `roadmap_d083_2026-10-26.zip` | do not apply |

## 6. Traps

- **The next free ID moves.** Two chats write the log; one concurrent write lost a row on 2026-10-23. Re-read the log
  immediately before every write and merge.
- **Three noise levels.** $\hat\sigma_{\rm SS}(c)$ (one 10 ms raw segment's SD, the threshold's unit) is not the SS
  statistic $\hat\sigma$ (the enlarged window after the subtraction), nor `TransientResult.floor_mV` (the per-sample SD
  of $\bar r_{\rm even}$, used only by `transient_floor_z`).
- **What $r_k$ was.** In D-051 (iv) and Block 3, $r_k$ is the whole window minus its own 10 ms pre-pulse mean; the
  10 ms enter only as the baseline level **[src]**. D-095 replaces the construction anyway.
- **The pre-pulse segment is not a baseline for a decay.** It holds the previous, opposite-polarity pulse's tail
  (consecutive windows overlap by 9.5 ms; 200.5 ms onset to onset) **[run: D-095 comment (b)]**.
- **Main has not moved.** As of 2026-10-28, `main` is `64fb8c5`: any new work starts from it plus the five noise-fit
  patches (plus the two roadmap patches); the user's laptop may already have them applied. [corrected 2026-10-28 (later):
  everything is on `main` (`cd32dad` and later); start from a fresh `git fetch`; the user's laptop may still hold the
  patches applied locally, which `git pull` reconciles only if the commits are identical -- the user applied none of them.]
- **The roadmap is on hold** except for new SPEC blocks (D-082's annotation of 2026-10-26, 16:48; D-091). [updated
  2026-10-28 (later): lifted by the user for the D-095 update only; the hold otherwise stands]
- **D-097: push yourself, and put every document in the repository, in the same turn.** The repository is public: no
  credentials, no personal data, no PDFs; the user's uploads only with their word. A project-knowledge write gets its byte
  copy under `docs/project/` in the same turn (re-read the project document first, as always).
- **The roadmap tool writes a bare `on:` key** in a revision entry, which PyYAML reads as the boolean true, so the
  flag's date is lost; quote it (`"on": <date>`) after `roadmap.py revise`, as done for N3 and N4 (`cd32dad`). The
  tool is the subproject-roadmap skill's: reported to the user, not patched in the repository.
- **Unshallow the clone before `roadmap.py status`** (`git fetch --unshallow origin`): a shallow clone turns `done`
  into `pushed` and puts a warning on the page.
