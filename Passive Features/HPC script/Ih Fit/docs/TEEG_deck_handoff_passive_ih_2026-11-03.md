# TEEG -- deck handoff: the optimisation of the passive + I_h model (2026-11-03)

| Date | Change |
|---|---|
| 2026-11-03 | Created by the noise-fit chat at the user's request, for the chat that will build the presentation deck on the optimisation of the passive + I_h model of human pyramidal cells. Status read from the roadmap (`roadmap.py status` at `854004c`); passive and I_h facts gathered from the documents and code named in §9 and spot-checked against them this session (numbers marked); nine figures made (§8). Lives in the repository (`Passive Features/HPC script/Ih Fit/docs/`), because project knowledge has been full since 2026-10-28 (`docs/project/README.md`): the write of a copy to project knowledge `claude/deck_pack/` was refused on 2026-11-03 (the project is over its maximum size). |
| 2026-11-03 (later) | The nine figures redesigned at the user's request ("much more visually appealing"): one script, `make_figures.py` v2, now builds all nine as HTML/CSS slides with SVG charts; the Mermaid sources are removed. §8: regenerate command corrected, fig01's kind, and a note on what v2 adds (numbers re-checked against §2-§6 and the code named there; the new 36 % / 64 % variance split computed this session). |

**Basis tags.** **[src]** read from source code or a repository document this session; **[run]** executed this
session; **[KB]** project knowledge, full text; **[log]** the decision log's text (`docs/project/TEEG_decisions_and_ideas_log.md`);
**[AR1]** project knowledge `claude/TEEG_annual_report_2026_01_passive_properties.md` (2026-09-28: its numbers were
computed in earlier chats from the cluster's CSVs and are reproduced, not recomputed); **[plan]**
`claude/TEEG_Ih_fit_staged_plan.md` v0.12 (2026-10-21, predates D-083); **[roadmap]** computed status at `854004c`;
**[reasoning]** the assistant's inference; **[open]** not decided or not verified.

## 0. What the deck chat is asked to build

A presentation on how the passive membrane parameters of human cortical pyramidal cells are fitted to the Allen
recordings, why the fit was extended with the h-current, how the extension will be validated before any real cell is
fitted, and where the work stands (theory and implementation). Audience and length were not specified by the user:
ask them. The storyline the evidence supports, in one line: *a passive-only fit works and is the current estimate of
record, but it returns a capacitance 2-3 times the published human value because it absorbs I_h; a six-parameter fit
models I_h instead, and a synthetic recovery gate -- whose twins need a fitted recording-noise law -- must pass before
it is used on real cells.* Every number below carries its source; §11 lists what must not be claimed.

## 1. Where we stand (roadmap allen-mtg-fit, computed 2026-11-03 at `854004c`)

Live page: "Subproject Roadmaps", https://claude.ai/artifact/Fe6Zqjr5iq3XAQZfrTspqB (collection `roadmaps`, doc
`towards-eeg:allen-mtg-fit`). Source of truth: `specs/roadmap/allen-mtg-fit.yaml`; rendered `specs/roadmap/allen-mtg-fit.md`.
Counts **[roadmap]**: implementation 8 done, 7 tested, 4 pushed, 7 planned; theory 5 closed, 4 studying, 8 open.
Figures: `fig01_pipeline_status.png` (the chain), `fig02_roadmap_board.png` (every stage and topic).

| Group | What it is | State **[roadmap]** | In plain words |
|---|---|---|---|
| G1 Data archive (Phase 0) | Colab writer: download, SWC repair, SS pulses, LS sweeps, trace QC | done | the data exist (L3_exc archive of 18 cells checked on the cluster, D-072) |
| G2 Passive fit (Phases 1-3) | loader, GP fit + gate (run B), Phase 2.5 + Phase 3, synthetic benchmark | done | the production 3-parameter fit ran on 68 cells; run B is the estimate of record |
| G3 I_h fitter (Stages 0-6) | regression harness, mechanisms, loader, ParamSpec, protocol, Phase 3, orchestrator | tested (Stage 0 done) | built and smoke-tested on davinci (2026-10-20); **no real cell fitted** |
| G4 Noise fit (SPEC Blocks) | Block 1 ACF, Block 2 law, Block 3 windows, Block 4 per-cell fit, tools | Blocks 1-4 pushed, tools done | per-cell noise-law fit written and smoke-tested in the sandbox, not yet run on real cells |
| | Block 9 ("4b") per-sweep stage; Blocks 5-7 (twins' noise, driver + S0-S3, record) | planned | Block 9's SPEC drafted (D-116), four open points with the user; 5-7 not specified |
| G5 Gate and campaign | Stage 7 as built (I12: tested); noise table (I13: done); S7 gate run, S8 campaign, S9 record | planned | the gate has **never run**; D-083: it runs before any real six-parameter fit |

| Theory group | Topics and state **[roadmap]** |
|---|---|
| TG1 Passive fit | T1 cable model, T2 recordings and archive, T3 BO + profile + bootstrap, T4 gate: **all closed** |
| TG2 Recording noise | T5 law, T6 statistics are not parameters, T7 indirect inference, T9 per-sweep shrinkage: **studying**; T8 tilt and semivariogram: open; T10 SS layout and trace-QC selection: closed |
| TG3 I_h fit | T11 kinetics and knobs, T12 rest balance, T13 protocol and loss, T14 identifiability and gate: **all open** |
| TG4 Not yet tackled | T15 temperature, T16 corrected morphologies, T17 comparing the arms: open |

Stale text on the roadmap: N4's summary still says "the tilt's SD alternated with the law" -- superseded by D-107
(every simulated sweep carries its real sweep's tilt; no tilt SD is fitted) **[log; src: SPEC 2.3 (f)]**.

## 2. Passive fit: model, data, loss, optimiser, uncertainty (theory T1-T4, closed)

### 2.1 The model

| Item | Value | Basis |
|---|---|---|
| fitted vector | $\theta = (C_m, R_m, R_a)$ in $\mu$F cm$^{-2}$, $\Omega$ cm$^2$, $\Omega$ cm; $g_{\rm pas} = 1/R_m$, $\tau_m = R_m C_m \cdot 10^{-3}$ ms | [AR1 §1] |
| search box | $C_m \in [0.3, 3.0]$, $R_m \in [10^3, 10^5]$, $R_a \in [50, 1000]$; optimiser coordinate $q = \log\theta$, uniform prior in $q$ | [AR1 §3.3; src: `param_spec.py`] |
| simulator | NEURON 9.0.1, `pas` only (no active conductances), IClamp at the soma, $e_{\rm pas}$ = the LJP-corrected $V_{\rm rest}$ (not fitted) | [AR1 §10; Biological_Passive_Fit_HPC_Guide.md §1] |
| morphology | Allen SWC via Import3d, Phase-0 repair (heal, minimum-spanning-tree stitch, re-root, electrical-viability check); axon replaced by the Hay stub, 2 cylinders of 30 $\mu$m $\times$ 1 $\mu$m | [AR1 §3.1-3.2] |
| spines | for each segment $s$ at path distance $d(s)$: $c_m(s) = C_m F(s)$, $g_{\rm pas}(s) = F(s)/R_m$, $F(s) = F$ if $d(s) \ge 60\ \mu$m else 1; $F$ = 1.9 for L2/L3 (Eyal 2016), an input, not fitted | [AR1 §3.2; KB: Eyal 2016] |
| time step | 0.025 ms for the brief pulses, 0.1 ms for the long steps (runs A/B); 0.1 ms "to be validated" | [AR1 §3.4; plan §12] |

What each protocol carries: the first milliseconds of the Square Subthreshold (SS) transient are capacitive and carry
$C_m$; the slow decay carries all three together; the Long Square (LS) plateau carries $R_{\rm in}$ and its charging
$\tau_m$ (and the sag of I_h). In the slow decay the sensitivities to $\log C_m$ and $\log R_m$ become collinear, and
$\partial R_{\rm in}/\partial C_m = 0$: only the early band separates $C_m$ from $R_m$ **[AR1 §4.1]**.

### 2.2 The data

| Item | Value | Basis |
|---|---|---|
| source | Allen Cell Types database, human cells with NWB recordings and 3-D reconstructions (158 such cells when Phase 0 was designed) | [AR1 §3.1; KB: phase1_conceptual_guide] |
| cohort of runs A/B | 68 cells in ten groups (L2-L6 x exc/inh); excitatory n: L2 10, L3 18, L4 6, L5 8, L6 5 | [AR1 §6] |
| SS protocol | 0.5 ms pulses at $\pm$200 pA; **each SS sweep is one 37-pulse train, 19 dep / 18 hyp, 200.5 ms onset to onset** (not "20 pulses at 200 ms" as AR1 §2 and the phase-1 guides say) | [log: 2026-10-20 (later, 5); SPEC §3, data inspected] |
| LS protocol | 1 s steps from $-110$ pA in 20 pA increments ($-220$ in 40 pA for some cells) | [plan §5] |
| LJP | +14 mV added once at load time | [AR1 §3.1] |
| archive | `specimen_<id>/` with `metadata.json`, `ss_pulses.npz`, `ls_sweeps.npz`, `reconstruction.swc`, `trace_qc_results.csv`; on davinci at `/davinci-1/home/ldellamea/Human Neurons Fitting/<GROUP>/` | [AR1 §3.1, §6; SPEC §3] |
| regions | the cells are not all MTG: the L2 candidate list also holds SFG, temporal and frontal cells (the roadmap's title says "MTG") | [subagent read of `Passive Allen Data/L2/candidates.csv`, not re-checked] |

### 2.3 Loss and optimiser

For each bundle $b$ (one SS pulse averaged per polarity, or one LS sweep) scored on its window $W_b$, with
$\mathrm{RMSD}_b(\theta \mid W_b)$ the baseline-subtracted root-mean-square deviation of model from data at $\theta$ and
$D_b$ the bundle's peak deflection **[AR1 §1, §3.4]**,

$$
\mathcal L(\theta) = \frac{1}{|\mathcal B|} \sum_{b \in \mathcal B} \frac{\mathrm{RMSD}_b(\theta \mid W_b)}{D_b},
\qquad
w(t \mid t_0, \tau_w) = e^{-(t - t_0)/\tau_w}\ \ (t \ge t_0)\ \text{on the SS residual},\ t_0 = 0.5\ \text{ms},\ \tau_w = 5\ \text{ms}.
$$

| Element | Run B (production passive fit) | Basis |
|---|---|---|
| training bundles | the SS hyperpolarising pulse (0.5-100 ms, time-weighted) and the 2 smallest hyperpolarising LS steps that pass the 12 mV deflection guard, scored on the first 60 ms after onset | [AR1 §3.4; plan §2-3] |
| optimiser | `skopt.gp_minimize`, 100 calls, 50 initial, acquisition `gp_hedge` (default) | [src: Biological Fit `submit_biological_fit.sh` N_CALLS=100, N_INITIAL=50; subagent read of the monolith] |
| gate (`_apply_absolute_gate`) | train and validation RMSD recomputed unweighted in mV at $\hat\theta$; in order: train > 2.0 mV failed; validation non-finite failed; validation $\le$ 0.2 mV good; ratio $\le$ 3 good; ratio $\le$ 10 to_refine; else failed | [src: Biological_Passive_Fit_HPC_Guide.md §8] |
| Phase 2.5 (run A) | $R_a$ fixed per group at the median profile argmin, $(C_m, R_m)$ refitted: **rejected** (§3) | [AR1 §3.5, §6] |
| Phase 3 | bootstrap, $B$ = 200, warm-started GP refits in a log-ball of radius 0.2, replicates with loss > 5x the MLE's dropped; percentile / normal / BCa intervals; GP-surrogate profile as an identifiability check | [AR1 §3.6; src: submit script BOOTSTRAP_N_CALLS=70] |

The profile likelihood (theory, `Bayesian Optimisation and Likelihood Profile.md` [KB]): under Gaussian noise the loss
is proportional to $-2\log L$; the profile minimises over the other parameters at each fixed value of one; it separates
identifiable, practically non-identifiable and structurally non-identifiable parameters.

### 2.4 The identifiability lesson (the strongest methods slide)

| Stage | Result | Basis |
|---|---|---|
| plain RMSD on 8 synthetic L2/3 morphologies, one truth ($C_m$ 1.0, $R_m$ 15000, $R_a$ 150, $\tau_m$ 15.0 ms) | median recovered/true: $\tau_m$ 1.09 (CV 8.9 %), $C_m$ 1.41, $R_m$ 0.78, $R_a$ 1.82; corr($\log C_m$, $\log R_m$) $= -0.972$: estimates slide along the iso-$\tau_m$ valley | [AR1 §4.1] |
| after the redesign (time-weighting, multi-protocol training, deflection guard), 30 L3 morphologies, human I_h injected | median $\lvert\log_{10}(\hat\theta/\theta)\rvert$: $C_m$ 0.050 (~12 %), $R_m$ 0.094 (~24 %), $R_a$ 0.062 (~15 %), $\tau_m$ 0.095 (~25 %); 27/30 cells have $R_m^{\rm fit} < R_m^{\rm true}$ -- the I_h fingerprint | [AR1 §4.2] |
| LS window 150 -> 60 ms, 12 paired cells | $R_m$ ratio 0.769 -> 0.848 (paired $p$ = 0.021, 10/12 improved), bias reduced 37 %; residual bias still significant (17/18, $p$ = 0.00005) | [AR1 §4.3] |
| Eyal 2016's six human L2/3 cells re-fitted | $C_m$ deviation $-1.8 \pm 10.9$ % (exp-weighted), every cell 0.34-0.53, never near 1.0 | [AR1 §5] |

Caveat that must travel with the before/after row: the comparison is confounded (different cells, different truth
design, I_h only in the after set); only the $\tau_m$ statistic compares like for like **[AR1 §4.2]**.

## 3. Passive fit: results to put on slides (figure `fig06_passive_results.png`)

| Quantity | Value | Basis |
|---|---|---|
| run B excitatory medians, $C_m$ ($\mu$F cm$^{-2}$) | L2 1.478, L3 1.062, L4 0.785, L5 0.797, L6 1.100 (n 10, 18, 6, 8, 5) | [AR1 §6] |
| run B excitatory medians, $R_m$ ($\Omega$ cm$^2$) | L2 17773, L3 26220, L4 44647, L5 51064, L6 42057 | [AR1 §6] |
| published human L2/3 reference | Eyal et al. 2016: $C_m = 0.47 \pm 0.03\ \mu$F cm$^{-2}$ (mean $\pm$ SD, n = 6 model fits); nucleated patches $0.51 \pm 0.11$ (n = 5); combined human $0.49 \pm 0.08$ (n = 11) against mouse $0.93 \pm 0.36$ (n = 11), $p$ = 0.0021 | [KB, full text: Eyal 2016, eLife 5:e16553, pp. 2-3; read this session] |
| reading | L2 ~3x, L3 ~2.3x the human value: the direction the benchmark predicts for a passive model fitted to I_h-bearing cells | [AR1 §6] |
| bootstrap widths (run B, 31 cells, 3-D) | median relative percentile width $C_m$ 0.131, $R_a$ 0.186, $R_m$ 0.592: $R_m$ is the least determined | [AR1 §6] |
| rails and intervals | 16 of 68 cells at the $R_a$ floor (50 $\Omega$ cm); BCa collapses to zero width in 34/99 (run A) and 41/93 (run B) intervals | [AR1 §3.6, §6] |
| Phase 2.5 verdict | degrades the training fit in 56/68 cells, doubles the $R_m$ rails, shifts $\tau_m$ by a median 20 %; L2 $C_m$ 1.478 -> 2.730; **run B is the estimate of record** | [AR1 §6] |
| noise in the residuals | lag-1 autocorrelation above 0.5 in 61 of 68 cells: an iid noise model overstates the information several-fold | [AR1 §6; log D-012] |

Not available in any source read: per-group medians of $R_a$ and $\tau_m$, the inhibitory groups' medians and n, gate
pass counts for runs A/B (they were presentation material of 2026-09-19, not in the repository) **[open]**.

## 4. The I_h extension: six parameters (theory T11-T14, open)

Why: three published I_h kinetics disagree by an order of magnitude at rest, so fixing one would bake that choice into
every $C_m$ (D-005); the user's decision is to let the data place and pace the curve and to gate it synthetically.

| Parameter | Unit | Coordinate | Box | Role | Basis |
|---|---|---|---|---|---|
| $C_m$, $R_m$, $R_a$ | as §2.1 | log | as §2.1 | passive | [src: `param_spec.py`] |
| $\bar g_h$ | S cm$^{-2}$ | log | $[10^{-6}, 10^{-3}]$ | I_h density, uniform on soma, basal, apical (axon stub excluded) | [src: `param_spec.py`; AR1 §7.2] |
| $\Delta v_h$ | mV | **linear** | $[-10, +10]$ | shift of the activation curve $m_\infty$ only | [src] |
| $\kappa_\tau$ | dimensionless | log | $[0.5, 2.0]$ | scale of the time constant $\tau^h$ | [src] |

For each segment $s$ and each fixed membrane potential $v$ **[AR1 eq. 7.1-7.2]**:

$$
m_\infty(v) = m_\infty^{\rm base}(v - \Delta v_{\rm base} - \Delta v_h), \qquad
\tau^h(v) = \kappa_\tau\, \tau^{h,\rm base}(v - \Delta v_{\rm base}), \qquad
e_{\rm pas}(s) = V_{\rm rest} + \frac{\bar g_h(s)\, m_\infty(V_{\rm rest})\,(V_{\rm rest} - E_h)}{g_{\rm pas}(s)},
$$

the last making $v \equiv V_{\rm rest}$ a stationary state of the whole cable (no seventh parameter).

| Mechanism | Source | Key constants | Basis |
|---|---|---|---|
| `Ih_human` (the base, D-007 C1) | Rich et al. 2021, Cereb Cortex, "L5 Human model" | $m_\infty = 1/(1 + e^{(v_m - v_h)/k})$, $v_h = -90.87$ mV, $k = 8.05$; $E_h = -49.85$ mV; default $\bar g_h = 5.14\cdot10^{-5}$ S cm$^{-2}$ | [src: `mod/Ih_human.mod`] |
| `Ih` (labelled alternative) | Kole et al. 2006 (rat L5), as in Hay et al. 2011; +20 mV reproduces Kalmbach et al. 2018 | $E_h = -45$ mV | [src via subagent: `mod/Ih.mod`] |

At $V_{\rm rest} = -73.5$ mV, computed this session with the fitter's own reference kinetics (`human_ih_params.py`):
$m_\infty$ = 0.104 / 0.026 / 0.166 and $\tau^h$ = 341 / 46 / 73 ms for Rich / Kole / Kole + Kalmbach; peak $\tau^h$ of
the human curve 343 ms; its formula falls below 1 ms below $-106.6$ mV (no floor) **[run; matches D-005 and AR1 §7.1]**.
Figure `fig04_ih_kinetics.png`.

**Protocol (D-006, D-007, D-010, D-016)** **[AR1 §7.3; plan §5]**: every guard that kept I_h out is removed (12 mV cap,
two-smallest-steps rule, 60/150 ms cut). Roles: train on the SS hyperpolarising pulses and $h_2 \dots h_{n-1}$ over the
whole step; validate on the spike-free depolarising steps $d_1 \dots d_k$ (I_h de-activation, never trained on), on
$h_1$ and the SS depolarising pulses; report only $h_n$ and every rebound window (post-inhibitory-rebound conductances
are not modelled). The depolarising-sweep spike screen is the Allen white paper's action-potential rule (D-016).

**Arms, at identical settings (dt 0.1 ms, GP 200 calls / 100 initial, $F$ = 1.9, Phase 2.5 off, Phase 3 on with
$B$ = 200)** **[AR1 §7.3; src: README, `run_ih_fit.py`]**:

| Arm | Parameters | Protocol | Why it exists |
|---|---|---|---|
| `baseline_runB` | 3 | run B's (2 smallest steps, 60 ms) | the estimate of record, reproduced |
| `passive_fullstep` | 3 | the D-006 protocol | the control that separates "the window and extra steps" from "I_h" |
| `ih6` | 6 | the D-006 protocol | the deliverable |
| `ih4` (Stage 7 only) | $C_m, R_m, R_a, \bar g_h$ | the D-006 protocol | kinetics fixed at the base |

Known identifiability risk **[plan §7, reasoning]**: the LS plateau constrains only $g_{\rm pas} + \bar g_h\, m_\infty(V_{\rm rest})$;
$R_m$ separates from $\bar g_h$ only through the sag fraction across step amplitudes and the SS pulses -- which is why
every amplitude is admitted and why the gate (§6) exists.

## 5. The recording-noise law and its fit (theory T5-T10; SPEC `Ih Fit/specs/SPEC.md`)

Why the twins need it: run B's noise column was measured on averaged bundles and no synthetic cohort had correlated
noise (D-012); on L3_exc the LS pre-window autocorrelation is 0.613 at 0.1 ms and 0.566 at 1 ms against AR(1)'s 0.134
and 0.000 **[AR1 §7.5; log D-016]**. A gate run on white or AR(1) twins would be optimistic (D-014).

**The law (D-017, SPEC §2.1)** for one sweep, samples $t_j = j\Delta t$, mid-time $\bar t$:
$e_j = e^{\rm fast}_j + e^{\rm slow}_j + a + b\,(t_j - \bar t)$, with $e^{\rm fast}$ an AR(1) of SD $\sigma_f$ and lag-1
coefficient $\rho_f$, $e^{\rm slow}$ an OU of SD $\sigma_s$ and time constant $\tau_s$ ($\rho_s = e^{-\Delta t/\tau_s}$);
for each fixed $\theta = (\sigma_f, \rho_f, \sigma_s, \tau_s)$ and every lag $k \ge 0$,
$\gamma(k \mid \theta) = \sigma_f^2 \rho_f^k + \sigma_s^2 e^{-k\Delta t/\tau_s}$. The tilt $b$ is not drawn: every twin
sweep and every simulated sweep inside the fit carries its own real sweep's tilt $\hat b(c, s)$ (D-102, D-107).
Cohort closed-form law $\theta_{\rm cf}$ = (0.0380 mV, 0.175, 0.0508 mV, 36.6 ms) **[src: SPEC]**. Figure `fig07_noise_law.png`.

| Step | What is done | Decisions | State |
|---|---|---|---|
| statistics | 15 per cell: SD and textbook autocorrelation at fixed lags of the SS enlarged windows (5) and LS pre-windows (10), medians over windows; textbook ACF through statsmodels | D-028, D-029, D-051 | Block 1, 3 pushed |
| SS windows | 37-pulse trains at the archive's positions; groups of < 5 windows dropped; each window's tilt offset removed; per-sample **median** window per sweep and polarity subtracted from $t_{\rm SS}(c)$ = max(80 ms, median of the groups' starts), each start from an exact least-squares exponential fit | D-095, D-103, D-104, D-106, D-110, D-111, D-114, D-115 | Block 3 pushed |
| tilts | SS: least-squares slope of window means at the 80 ms floor (one intercept per polarity); LS: line through the 265 ms pre-window and last 500 ms means | D-102, D-107, D-115 | Block 3 pushed |
| per-cell fit | indirect inference: $J_W(\theta \mid c) = (q_{\rm sim}(\theta) - \hat q(c))^\top W (q_{\rm sim}(\theta) - \hat q(c))$, $q_{\rm sim}$ Harrell-Davis medians of $M$ = 3000 simulated windows at the cell's layout (common random numbers), $W$ = inverse simulated covariance iterated to a fixed point | D-033, D-048 to D-051, D-056 | Block 4 pushed |
| per-sweep stage | each sweep's $(\sigma_f, \rho_f, \sigma_s)$ in one solve ($\tau_s$ held); $v(c,s)$ by a parametric bootstrap; DerSimonian-Laird per protocol, shrinkage | D-073, D-076, D-116 | Block 9 SPEC drafted |
| twins' generator, driver, record | each twin sweep with its real sweep's law and tilt; S0-S3 checks; PBS job | D-073, D-083 | Blocks 5-7 not specified |

Figure `fig08_noise_fit.png`. Open with the user (SPEC §8): Block 9's bootstrap size (proposal 50), the refits'
binding size (proposal 1000), failed refits, and the reading of "the cell's values" for a failed sweep. Measured cost
on the smoke fixture: one per-sweep solve 0.9 min (SS) / 4.0 min (LS) at $M$ = 3000 **[run: `Ih Fit/dev/2026-10-30_sweep_stage_cost/`]**.

## 6. The recovery gate (Stage 7) and the campaign (Stages 8-9)

| Element | Design | Basis |
|---|---|---|
| twins | real L3_exc archive morphologies (raw SWCs, $F$ = 1.9), up to 20 (`MAX_CELLS`) | [src: `submit_ih_recovery.sh`; log D-011] |
| truths | passive: $C_m$ 0.4-1.5, $\tau_m$ 3-40 ms, $R_a$ 100-500; I_h: $\bar g_h$ log-uniform [2e-5, 3e-4] S cm$^{-2}$, $\Delta v_h$ uniform [-5, 5] mV, $\kappa_\tau$ log-uniform [0.7, 1.4] | [src: `run_ih_recovery.py`, `submit_ih_recovery.sh`] |
| false-positive control | every 4th twin by index (fraction 0.25) has no I_h ($\bar g_h$ = 1e-6, $\Delta v_h$ = 0, $\kappa_\tau$ = 1) | [src: `synth_gt_grid.py`; log D-011 (ii)] |
| noise and protocol | D-083.2: each twin sweep carries its real sweep's fitted noise, and the real cell's own protocol (37-pulse SS trains, its LS amplitudes) -- **not what the code on disk generates yet** (the noise fit's Blocks 9, 5-7) | [log D-083; roadmap] |
| fits | 4 arms per twin (`baseline_runB`, `passive_fullstep`, `ih4`, `ih6`), 200 GP calls (100 initial), through `run_ih_fit.main()` | [src: `submit_ih_recovery.sh`] |
| error measure | $\varepsilon_i(c) = \log(\hat\theta_i(c)/\theta_i(c))$ for $C_m, R_m, R_a, \bar g_h, \kappa_\tau$; $\hat\theta_i(c) - \theta_i(c)$ (mV) for $\Delta v_h$; I_h axes over I_h-bearing twins only; rails counted, never dropped | [AR1 eq. 7.3; log D-011] |
| gate on `ih6` | median $\lvert\varepsilon\rvert \le \log 1.25$ for $C_m$ and $\bar g_h$; $\le$ 3 mV for $\Delta v_h$; $\le \log 1.5$ for $\kappa_\tau$; $\ge$ 80 % of the controls rail $\bar g_h$ at the low bound -- **the assistant's proposal, never agreed by the user** | [src: `ih_recovery_report.py`; AR1 §7.4] |
| on failure | a failing kinetic knob is frozen at its base value for the campaign; a failing $C_m$ or $\bar g_h$ cannot be frozen: the protocol must change | [log D-005, D-011, D-083.1] |
| order | D-083 (user, 2026-10-26): no real cell is fitted with six parameters, the one-cell dry run included, before Stage 7 has run on davinci and its verdict is in | [log D-083] |
| campaign (S8) | one diameter-corrected cell dry run, then the corrected set, then every excitatory group; three arms; per-cell parameters, gate columns, Phase-3 intervals; waits for the gate and for the corrected SWCs (D-013; none exist yet) | [plan §8; roadmap S8] |
| record (S9) | the decision naming the estimate of record; the guide updated | [roadmap S9] |

Commands (from `Passive Features/HPC script/Ih Fit` on davinci) **[src: README, `submit_ih_recovery.sh`]**:
`bash cluster_gates_noise_fit.sh` (prerequisite checks); `qsub -v DRY_RUN=1 submit_ih_recovery.sh`; pilot
`qsub -v MAX_CELLS=2,N_CALLS=20,N_INITIAL=10,RUN_TAG=pilot submit_ih_recovery.sh` (its verdict is not a verdict);
full `qsub submit_ih_recovery.sh`. Figure `fig09_recovery_gate.png`.

## 7. Implementation map (repository `Leonardodm00/Towards-EEG`, branch `main`)

| Path | What it holds |
|---|---|
| `Passive Features/Allen Institute Data/` | Phase-0 Colab writer (`save_alleninstitute_data (6).py`), trace QC |
| `Passive Features/Phase 1/`, `Phase 3/`, `Plot/` | the original phase guides (Phase 1 README = `phase1_function_reference.md`), Phase-3 README, publication-plot documentation |
| `Passive Features/HPC script/Biological Fit/` | the production passive fitter: `run_biological_fit.py` (orchestrator), `passive_fitting_hpc_fixed.py` (monolith), `passive_long_step_training.py` (loss), `submit_biological_fit.sh`, `submit_all_groups_biological.sh`, `Biological_Passive_Fit_HPC_Guide.md` |
| `Passive Features/HPC script/Synthetic Passive fit Test/` | the synthetic benchmark (`synthetic_ground_truth.py`, `synth_gt_grid.py`, `run_synth_benchmark.py`, `benchmark_usage_guide.md`, `CHANGELOG_human_ih_v70.md`) |
| `Passive Features/Eyal 2016 Adapter/` | ModelDB 195667 -> archive adapter for the Eyal replication |
| `Passive Features/HPC script/Ih Fit/` | the six-parameter fitter (edited copy of Biological Fit): `mod/Ih.mod`, `mod/Ih_human.mod`, `human_ih_params.py`, `ih_mechanism.py`, `param_spec.py`, `run_ih_fit.py` + `submit_ih_fit.sh`, `regression_passive_identity.py` (Stage 0, bit-for-bit), Stage 7 (`synth_gt_grid.py`, `gen_from_manifest.py`, `ih_recovery_report.py`, `run_ih_recovery.py` + `submit_ih_recovery.sh`), smoke suites `smoke_ih_fit.py` (12/12) and `smoke_ih_recovery.py` (14/14) |
| same folder, noise fit | `noise_stats.py` (Block 1), `noise_law.py` (2), `noise_windows.py`, `noise_cell_stats.py` (3), `noise_fit.py` (4), smoke `smoke_noise_*.py`, `cluster_gates_noise_fit.sh`, `specs/SPEC.md` |
| `specs/roadmap/` | `allen-mtg-fit.yaml` / `.md`, `roadmap.py` |
| cluster | davinci, PBS queue `cpu`, conda env `prova`; archives `/davinci-1/home/ldellamea/Human Neurons Fitting/`; passive outputs `.../pipeline_outputs/<RUN_TAG>/` (run B = `dt01ls60`); I_h outputs `.../pipeline_outputs_ih/`; Stage 7 `.../stage7_recovery/` |

Verification facts for an "engineering" slide: the Stage-0 regression gate passes bit-for-bit on the real cell
508282493 (max $\lvert\Delta L\rvert$ = 0 over 20 points) and after every stage; cluster gates 13/13 SETTLED on
2026-10-20 **[roadmap reasons; log]**.

## 8. Figures for the deck (`Ih Fit/docs/deck_passive_ih_2026-11-03/figures/`)

Raw URL pattern (public repository): `https://raw.githubusercontent.com/Leonardodm00/Towards-EEG/main/Passive%20Features/HPC%20script/Ih%20Fit/docs/deck_passive_ih_2026-11-03/figures/<file>`.
Regenerate all nine (v2): `python scripts/make_figures.py --status <roadmap status json>` (about 15 s; numpy,
Playwright's Chromium and the Inter font; `--only 4,7` for some). Each figure is a 1920 x 1080 slide rendered at 2x
(3840 x 2160 PNG); the charts are drawn as SVG from the same numbers as v1, following the dataviz rules (thin marks,
text in ink, legend plus direct labels). [corrected 2026-11-03 (later): v1 built fig02-04, 06, 07 with matplotlib and
fig01, 05, 08, 09 with `mmdc -i scripts/<name>.mmd ... -C scripts/_mermaid.css`; the user asked for much more visually
appealing figures, and v2 replaced both pipelines and removed the Mermaid sources.]

| File | Kind | The one claim it shows | Source of what is drawn |
|---|---|---|---|
| `fig01_pipeline_status.png` | schematic, status colours of the roadmap | the chain archive -> passive fit -> I_h fitter + noise fit -> gate -> campaign, and what is done / tested / pushed / planned | roadmap at `854004c` |
| `fig02_roadmap_board.png` | plotted from data | every implementation stage and theory topic with its state | `roadmap.py status --json` at `854004c` |
| `fig03_model_parameters.png` | schematic, not to scale | the cable model, where $F$ applies, and the six parameters with units, coordinates and boxes | `param_spec.py`; AR1 §3.2, §7.2 |
| `fig04_ih_kinetics.png` | plotted from formulas | the three published curves disagree at rest; the fitted knobs' boxes around the human base | `human_ih_params.py` [run] |
| `fig05_fit_loop.png` | schematic | data bundles -> NEURON -> loss -> GP optimiser -> $\hat\theta$ -> gate and Phase 3 | AR1 §3, §7.3 |
| `fig06_passive_results.png` | plotted from published-in-project numbers | run B's $C_m$ is 2-3x Eyal's in L2/L3; $R_m$ is the least determined parameter | AR1 §6; Eyal 2016 [KB] |
| `fig07_noise_law.png` | plotted (simulated) | the law's autocovariance (fast vs slow) and one simulated sweep with a tilt | `noise_law.py` at $\theta_{\rm cf}$ [run] |
| `fig08_noise_fit.png` | schematic | the per-cell indirect-inference loop and the per-sweep stage to the twins | SPEC §2.2-2.5 |
| `fig09_recovery_gate.png` | schematic | the Stage-7 design: truths, controls, arms, errors, proposed tolerances, outcomes | `run_ih_recovery.py`, `ih_recovery_report.py`, D-011, D-083 |

v2 keeps each figure's claim and adds, from the sources already listed: fig03 the four formula rows (membrane, spines,
I_h kinetics, rest balance; §2.1 and §4) and a drawn neuron (I_h halo, spines beyond 60 $\mu$m, electrode); fig05 the gate's
rules in their order (`_classify_fit`, guide §8) and the I_h arms' held-out bundles (§4); fig07 the split of the lag-0
variance, 36 % fast and 64 % slow, computed from $\theta_{\rm cf}$ **[run]**; fig08 schematic SS and LS sweeps with
their windows (SPEC §2.2, §2.4); fig09 the passive truth ranges in their boxes, the rejection draw of
$(C_m, R_m)$ on $\tau_m$ (`synth_gt_grid.py`) and each arm's parameter count (§4).

Existing figures in the repository (exist; not re-inspected for this handoff unless marked):

| File (under `Ih Fit/dev/`) | Shows | Use |
|---|---|---|
| `2026-10-30_offsets_and_fit/fig_fit_profile.png` | the exact decay fit's profile RSS($\tau$) for the SS window start (D-110) | current; made in this chat |
| `2026-10-16_covariance_weighting/bias_doc_2026-10-11/figures/fig01..fig05` | why a window statistic is not a parameter; bias against precision of the weighting | theory T6/T7 |
| `2026-10-26_tilt_sigma_b/figures/fig01..fig05`, `2026-10-28_tilt_explanation/fig01_overview.png`, `fig_nugget.png` | the tilt SD from the semivariogram | **legacy**: the pipeline no longer computes a tilt SD (D-102, D-107) |
| `2026-10-28_window_start/fig_transient_floor.png`, `noise_fit_decision_map.png` | the old transient test; a decision map of 2026-10-28 | **legacy / stale** (D-103 onwards) |

No passive-fit figure files exist in the repository; per AR1 and the Plot documentation they were generated on the
cluster or in Colab (corner plots, profiles, recovery plots) and their current location is not known **[open]**.

## 9. Documents to read (GitHub paths relative to the repository root; KB = project knowledge)

| Document | What for |
|---|---|
| KB `claude/TEEG_annual_report_2026_01_passive_properties.md` | the single best narrative with all passive numbers and the I_h design (§3-§7); note its superseded points in §11 |
| KB `claude/TEEG_annual_report_2026_00_overview.md` | timeline and maturity of the whole PhD year |
| KB `claude/TEEG_Ih_fit_staged_plan.md` (copy `docs/project/claude/`) | the I_h design, Stages 0-9, protocol, gate, open points (v0.12, predates D-083) |
| `docs/project/TEEG_decisions_and_ideas_log.md` (KB root copy) | D-005 to D-016 (I_h design), D-017 to D-116 (noise), D-083 (gate before real fits) |
| `Passive Features/HPC script/Biological Fit/Biological_Passive_Fit_HPC_Guide.md` | the production passive pipeline, the gate (§8), outputs |
| `Passive Features/HPC script/Ih Fit/README.md`, `CHANGELOG_ih_fit.md` | arms, commands, stage log |
| `Passive Features/HPC script/Ih Fit/specs/SPEC.md` | the noise fit: law, statistics, fit, tilts, per-sweep stage (§2.1-2.5), blocks |
| `Passive Features/HPC script/Ih Fit/docs/TEEG_noise_fit_handoff_2026-10-28.md` | the noise fit's current state and open points |
| `docs/project/claude/TEEG_Stage7_noise_model_2026-09-28.md` | the noise-model topic document (theory of the law and the statistics) |
| `Passive Features/HPC script/Ih Fit/dev/2026-10-16_covariance_weighting/bias_doc_2026-10-11/TEEG_bias_of_a_statistic_2026-10-11.md` | why statistics are not parameters (T6) |
| KB `Bayesian Optimisation and Likelihood Profile.md`, `phase1_conceptual_guide.md` (= `Passive Features/Phase 1/README.md`) | BO and profile theory; data model and protocols (user uploads: not copied to the repository) |
| `Passive Features/HPC script/Synthetic Passive fit Test/benchmark_usage_guide.md` | the synthetic benchmark and its plots |
| `specs/roadmap/allen-mtg-fit.md` | the rendered roadmap with every stage's summary and note |

## 10. Literature for the reference slide

| Reference | Used for | Access |
|---|---|---|
| Eyal et al. 2016, eLife 5:e16553 | the reference method, $F$ = 1.9 beyond 60 $\mu$m, human L2/3 $C_m \approx 0.5$ | KB full text (read this session for the $C_m$ values) |
| Eyal et al. 2018, Front Cell Neurosci 12:181 | the same cells as a passive skeleton | KB |
| Rich et al. 2021, Cereb Cortex | the human I_h kinetics (`Ih_human`), $E_h$, $\bar g_h$ anchor | KB |
| Kalmbach et al. 2018 | the +20 mV shift, uniform $10^{-4}$ S cm$^{-2}$, $e_{\rm pas}$ check | PMC (per the plan) |
| Kole et al. 2006; Hay et al. 2011 | the rodent `Ih` mechanism; the axon stub | Kole: PMC, not read; Hay via Rich Table 1 (per the plan) |
| Allen Cell Types technical white paper | protocols, LJP, 34 $^\circ$C, sag definition, spike rule | KB (`CellTypes_Ephys_Overview 1.pdf`) |
| Moradi Chameh et al. 2021 | expected sag sizes | KB main text (supplement not in KB) |
| DerSimonian and Laird 2015 | the random-effects pooling of the per-sweep stage | PubMed full text (per D-076) |
| Efron and Narasimhan, J Comput Graph Stat | the parametric bootstrap (the per-sweep variance) and BCa | KB full text |

## 11. What not to claim, and inconsistencies to settle before a slide

- That the Stage-7 gate ran, passed or failed, or that any real cell was fitted with six parameters (D-083; roadmap I11, S7).
- That run B's $C_m$ is the human membrane capacitance: it is an effective passive value with I_h absorbed, on raw arbours, with $F$ assumed **[AR1 §6, §9]**.
- That the gate's tolerances or the cohort size (20, no power calculation) are agreed: both are the assistant's proposals **[AR1 §9; log D-011]**.
- That the Phase-3 intervals are calibrated: SS-only resampling, warm-start ball, BCa collapse **[AR1 §3.6]**.
- That dt 0.1 ms is validated, that temperature is corrected (no Q10 in either `.mod`), or that interneurons are covered by the I_h fit **[plan §3, §12; AR1 §9]**.
- That the twins already carry the fitted noise: the generator on disk does not have the OU term or the per-sweep laws (noise-fit Blocks 9, 5-7 planned) **[roadmap; subagent read of `synthetic_ground_truth.py`]**.
- The SS protocol: 37-pulse trains, 200.5 ms onset to onset -- not "20 pulses at 200 ms" (AR1 §2, phase-1 guides) **[log; SPEC §3]**.
- The deep-layer spine factor: 1.5 (AR1, HPC guide), 1.6 (`submit_all_groups_biological.sh`) or 2.0 (older notes, Rich 2021) -- which one runs A/B used is not verified **[AR1 §3.2; subagent read]**.
- The passive validation window: "first 150 ms" in D-010, the plan and AR1; `DEFAULT_VALID_WINDOW_MS_AFTER_ONSET = 50.0` in the monolith **[src: `passive_fitting_hpc_fixed.py` l. 2825]** -- check before stating either.
- AR1 §7.4-7.5 describe the twins' noise as AR(1) per protocol (D-014); superseded by the law of D-017 and the per-sweep calibration (D-073, D-076, D-102, D-107, D-116) **[log]**.
- The figures `fig06`, `fig04` and `fig07` carry their sources in their footers; keep the footers or move them to the slide notes.

## 12. Suggested slide outline (a draft for the deck chat, not the user's)

1. Question and data: three numbers per cell type from somatic recordings of reconstructed human cells (§2.1-2.2; fig03).
2. The fit: loss, GP optimiser, gate, bootstrap and profile (§2.3; fig05).
3. What the data can identify: the iso-$\tau_m$ valley and the redesign (§2.4).
4. Passive results: run B and the 2-3x $C_m$ (§3; fig06).
5. Why I_h: the shunt looks like leak; three published curves disagree (§4; fig04).
6. The six-parameter model and protocol (§4; fig03).
7. Twins need real noise: the law and its fit (§5; fig07, fig08).
8. The recovery gate before any real fit (§6; fig09).
9. Where we stand and what comes next (§1; fig01, fig02).
