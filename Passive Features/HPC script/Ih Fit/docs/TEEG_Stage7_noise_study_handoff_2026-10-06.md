# Stage-7 noise model: study handoff (reading the topic document)

| Date | Change |
|---|---|
| 2026-10-06 | Created at the end of the study chat of 2026-10-05, for a new chat that carries on reading the topic document. Numbers checked against the topic document's §3.4-3.5 text and the live decision log on 2026-10-06 **[run: re-read]**. Written to the repo, not to project knowledge, because project knowledge was at 1,997,337 of 2,000,000 bytes on 2026-10-06 **[run: project_info]**. |

This is a **study** handoff: it says where the reading of the topic document
stands, what has already been explained and how, and which misreadings were
corrected. The **operational** state of Stage 7 (code, cluster runs, what to
build next) is in `claude/TEEG_Stage7_noise_handoff_2026-10-01.md`, not here.

## 1. What to load in the new chat, in order

| What | Where | Why |
|---|---|---|
| The topic document | project knowledge, `claude/TEEG_Stage7_noise_model_2026-09-28.md` (~97 KB; §1 notation, §2 glossary, §3.1-3.7 body, §4 summary, §5 open, §6 references, App. A per-cell statistics, App. B scripts) | the document under study |
| This handoff | repo, `Passive Features/HPC script/Ih Fit/docs/TEEG_Stage7_noise_study_handoff_2026-10-06.md` (or attached by the user) | where the study stands |
| Operational handoff, §4.6 | project knowledge, `claude/TEEG_Stage7_noise_handoff_2026-10-01.md` | the 2026-10-01 cluster results that supersede the topic document's §3.6.7 |
| Decision log, D-017 and its status notes 1-4 | project knowledge, **`TEEG_decisions_and_ideas_log.md` at the root**. The `claude/TEEG_decisions_and_ideas_log.md` copy is a stale duplicate that holds only D-001/I-001. | the decided law and every correction to its text |

The 2026-10-05 Q&A was the long tail of a chat that started on 2026-09-28.
Its transcript is in the user's chat history, but the new chat should not need
it: everything needed is below.

## 2. Where the reading stands

| Section of the topic document | State |
|---|---|
| §3.1-3.3 (context, measurements, the law (6)) | read in earlier sessions, when the document was written; not re-studied on 10-05 |
| §3.4.1 demeaning, eqs. (7a), (7b), (8) | **studied on 10-05 in depth** (§3 below) |
| §3.4.2 the autocorrelation is a biased ratio, eqs. (9), (10) | studied on 10-05 (autocovariance vs autocorrelation; nonlinearity) |
| §3.5.1 three routes, eqs. (11)-(13) | studied on 10-05 (why the direct fit fails, what indirect inference does) |
| **§3.5.2 closed-form fit, cost (14)** | **next** |
| §3.5.3 τ_s profile; §3.5.4 Monte Carlo check and the simulated-medians refit; §3.5.5 lineage | not yet studied |
| §3.6 drift (3.6.1-3.6.7) | not yet studied; **§3.6.7 is partly superseded** (§4 below) |
| §3.7 consequences for D-017; §4 summary; §5 open points; appendices | not yet studied |

## 3. What was settled on 2026-10-05

All answers were given in brief mode (Mode B). The notation is the topic
document's. For each fixed law θ, each fixed window length n ∈ ℕ (samples) and
each lag k or L ∈ {0, …, n−1}:

- x_1, …, x_n (mV): the samples of one window. **The document writes lowercase
  x_j both for the random variables (analytic level) and for the numbers in
  one stored window (computed level).** The split below is made explicit
  because that ambiguity caused most of the 10-05 questions.
- μ = E[x_j | θ] ∈ ℝ (mV): the process mean, a constant of the law, the same
  for every j. x̄ = n⁻¹ Σ_j x_j: the window mean computed from the n samples.
  u_j = x_j − x̄.
- γ(k | θ) = Cov(x_j, x_{j+k} | θ) (mV²); ρ(k | θ) = γ(k | θ)/γ(0 | θ)
  (dimensionless).
- S_0 = Σ_j u_j² (mV²); σ̂ = √(S_0/(n−1)) (eq. 2); S_L = Σ_{j=1}^{n−L} u_j u_{j+L};
  ρ̂(L) = [S_L/(n−L)] / [S_0/n] (eq. 3).

### 3.1 Two levels, one table

This table helped on 10-05. Use it again whenever a question mixes the two
levels.

| | analytic level: the law θ, over repeated independent windows | computed level: one stored window |
|---|---|---|
| a sample | random variable x_j | number x_j |
| mean | μ (constant, no hat) | x̄ (moves from window to window) |
| total wobble about μ | E[Σ_j (x_j − μ)² \| θ] = n γ(0 \| θ) | Σ_j (x_j − μ)², not computable (μ unknown) |
| wobble about the window's own mean | E[S_0 \| θ, n] | S_0 |
| wobble of the mean | n Var(x̄ \| θ, n) = E[n (x̄ − μ)² \| θ] | n (x̄ − μ)², not computable |
| variance estimate | E[σ̂² \| θ, n] | σ̂² |
| correlation at lag L | ρ(L \| θ) | ρ̂(L) |

The move from the right column to the left is "in expectation over repeated
independent windows" (in the data: over sweeps, §3.3 below). Indirect
inference (12) compares a right-column number (the measured median) with a
prediction of the right-column estimator's typical value under θ, never with a
left-column quantity.

### 3.2 The settled statements

| Question (10-05) | Settled answer |
|---|---|
| Autocovariance vs autocorrelation at a lag | γ(k \| θ) is a covariance in mV²; ρ(k \| θ) is that covariance divided by γ(0 \| θ), dimensionless. The estimators are S_L and ρ̂(L) (eq. 3). |
| Why do the means and medians of σ̂ and ρ̂ differ from what the expected sums give? | E[S_0] and E[S_L] are exact (eqs. 7b, 9), but σ̂ is a square root (Jensen: E[√Y] ≤ √E[Y]) and ρ̂ a ratio (E[A/B] ≠ E[A]/E[B]). Their sampling distributions are skewed, so the median differs from the mean as well. This is the second, smaller source of bias, after demeaning. |
| Why is demeaning the root of the biases? Is it why the direct fit fails? | Yes to both. Subtracting x̄ removes the part of a slow fluctuation that moves the window mean, so S_0 and S_L are deflated. The direct fit (11) compares the deflated ρ̂ with the process's undeflated ρ(L \| θ), so it returns too weak or too fast a slow component. Indirect inference (12) puts the same demeaning on both sides. |
| Eqs. (7a)-(7b) | (7a): Var(x̄ \| θ, n) = n⁻² Σ_{k=−(n−1)}^{n−1} (n − \|k\|) γ(\|k\| \| θ) (every pair of samples contributes its covariance). (7b): E[S_0 \| θ, n] = n γ(0 \| θ) − n Var(x̄ \| θ, n), and E[σ̂² \| θ, n] = E[S_0 \| θ, n]/(n−1). |
| "Window to window" of the same sweep? | No. "Window to window" means independent realizations of the window, which in the data are sweeps (LS: one pre-window per sweep). SS has 20 pulse windows per sweep, 200 ms apart; for the ~37 ms component they are effectively independent (e^{−200/37.3} ≈ 0.005). |
| Is μ the true window mean and x̄ the estimate from n samples? | Half right. μ is the true **process** mean, constant over the window and over windows; it is not a "window mean". x̄ is the window mean computed from the n samples. Var(x̄ \| θ, n) = E_θ[(x̄ − μ)²]. |
| Is that because we sample the process "in a Bayesian sense"? | Half right. The variance is that of the frequentist sampling distribution of x̄ given θ (repeat the recording, x̄ changes). The Bayesian object would be a posterior over θ; abcTau (Zeraati et al. 2022) is the Bayesian route, not this one. |
| Does μ account for the covariance and change over the window? | No. μ is a constant of the law; it carries no covariance information. The covariance enters through γ, i.e. through how far x̄ wanders from μ (7a). |
| Does n[γ(0) − Var(x̄)] assume iid samples? | No. E[Σ_j (x_j − μ)²] = n γ(0 \| θ) needs only linearity of expectation and stationarity; no independence. The correlations appear only in Var(x̄ \| θ, n), through (7a). |
| Why divide by n − 1? | Bessel's correction is exact only for uncorrelated samples. Then Var(x̄) = γ(0)/n, E[S_0] = (n−1) γ(0), and S_0/(n−1) is unbiased. With positive correlation, Var(x̄) > γ(0)/n, and S_0/(n−1) is biased low (eq. 7b). |
| "nγ(0) is the total wobble; nVar(x̄) = n(x−μ)²; Σ(x_j − x̄)² is the expected window variance" | Mixed levels (§3.1). The identity per window is Σ_j (x_j − μ)² = Σ_j (x_j − x̄)² + n (x̄ − μ)²; its expectation is n γ(0) = E[S_0] + n Var(x̄). S_0 = Σ_j (x_j − x̄)² is a per-window number, not an expected value, and n Var(x̄) = E[n (x̄ − μ)²], not n (x̄ − μ)². |
| Why multiply by n? | The shift x̄ − μ is the same at every one of the n samples, so it contributes n (x̄ − μ)² to the sum of squares. |
| Is a sweep one window? | For LS, yes: one pre-window (265 ms) per sweep. For SS, one sweep holds 20 pulse windows (10 ms each). |
| Averaging over sweeps with several windows: average nγ(0)? | No. n γ(0 \| θ) is already an expectation, a constant. What is averaged over windows is the per-window number Σ_j (x_j − μ)², and its average tends to n γ(0 \| θ). |
| How is Var(x̄) = γ(0)/n derived? | Only for uncorrelated samples: Var(Σ_j x_j) = Σ_j Var(x_j) = n γ(0); divide by n². In general the cross terms remain, and that is (7a). |

### 3.3 Misconceptions corrected on 10-05

The new chat should watch for these, because they recurred.

1. **Direction of the bias.** The user's summary said demeaning *inflates*
   the variance. It **deflates** it: E[S_0] = n γ(0) − n Var(x̄) < n γ(0).
2. **"Correlation drives slow components."** Positive long-lag correlation
   *is* the slow component: γ(k \| θ) > 0 at large k is what "slow" means. It
   neither drives one nor is driven by one.
3. **μ as a moving or covariance-carrying quantity.** μ is constant; see §3.2.
4. **iid assumption.** Not needed for n γ(0); see §3.2.
5. **Level mixing.** Per-window numbers vs expectations; see §3.1. This was the
   most frequent source of confusion.
6. **"Window to window" read as windows inside one sweep.** It means
   independent realizations (sweeps).

## 4. Statements in the topic document that are superseded

Do not teach these as current.

| Topic document | Superseded by | What changed |
|---|---|---|
| §3.6.7 (line ~1082), "Neither has been computed yet" (the LS tail test and the `vm_delta_mv` route) | operational handoff §4.6; D-017 status note 4 | `ls_baseline_qc.py` ran on L3_exc on 2026-10-01: **no continuing creep in any of the 18 cells** (sign agreement 53/104, p = 0.92). The baseline does change over 6.89 s by more than the 09-28 cohort law allows, in every cell. A slower component exists, and its shape is open. |
| §3.6.5, "526785799 -- a creep is the better reading, not decisively" | D-017 note 4 | Both 526785799 and 571430815 are slow wander, not creep. |
| Any reading that sign agreement is weak evidence against a slow wander (from the 10-01 work, D-017 note 3) | D-017 note 3, marked [corrected 2026-10-01 (later)] | A stationary wander of any timescale gives 0.49-0.53 sign agreement with these windows, so the sign test rejects a continuing creep. It says nothing about the wander's timescale. |
| D-017 text corrected by note 2: SS called "Short Square" | D-017 note 2 (1) | SS is the Allen **Square Subthreshold** protocol. The topic document already uses this. |

## 5. Numbers to keep at hand

All are L3_exc, 18 cells, from the topic document unless marked.

- Law (D-017.2, the document's (6)):
  γ(k | θ) = σ_f² ρ_f^k + σ_s² e^{−k Δt/τ_s}.
- θ̂_cf = (σ_f 0.0381 mV, ρ_f 0.176, σ_s 0.0506 mV, τ_s 36.3 ms), J^cf 1.88.
  The best single AR(1) reaches 215.4 (§3.5.2).
- θ̂_sim = (0.0381 mV, 0.179, 0.0560 mV, 37.3 ms), J^sim 0.69 (§3.5.4).
- τ_s profile J^cf: 29.8 / 11.0 / 3.0 / 1.96 / **1.88** / 1.95 / 2.20 / 2.82 / 4.14
  at 5 / 10 / 20 / 30 / **36.3** / 45 / 60 / 100 / 300 ms. Bounded well below,
  poorly above (§3.5.3).
- Visible share v(z) = 1 − (2/z²)(z − 1 + e^{−z}) (8): 8.6 / 33.6 / 59.8 / 76.4 %
  in the SS 10 ms / LS first 50 ms / last 132.5 ms / full 265 ms windows. Model
  σ̂ 0.0409 / 0.0481 / 0.0546 / 0.0584 mV against measured 0.0405 / 0.0465 /
  0.0551 / 0.0594 mV (§3.4.1).
- Demeaning bias of ρ̂ (last 132.5 ms, under θ̂_cf): true ρ(10 ms) 0.48 vs
  estimator median 0.23 (cohort measured 0.244). True ρ(100 ms) +0.04 vs median
  −0.11, 90 % range [−0.78, 0.26], so the 100 ms column is unusable as a
  target (§3.4.2).
- The ratio of expectations (10) overshoots the estimator's median by
  0.02-0.07 (worst 0.066, last segment, 10 ms) and its MC mean by 0.008-0.05.
  The SD closed form matches the estimator's mean to within 1.5 %, and the
  median is 1-3 % lower (§3.5.4; D-017 note 2 (5)).
- Fit design (§3.5.2): 14 cohort-median targets, ω_m 0.005 mV (SDs) and 0.05
  (autocorrelations), `scipy.optimize.least_squares`, six starts
  τ_s ∈ {2, 5, 10, 20, 50, 100} ms.
- 2026-10-01 tail results (operational handoff §4.6) **[run, cluster]**:
  - The per-cell median of |`delta_end_mV`| over 6.89 s is 0.048-0.620 mV
    (cohort median 0.128) against a cohort-law 99th percentile of
    0.046-0.056 mV.
  - An added OU component would need σ 0.142 / 0.189 / 0.293 mV at
    τ 3 / 10 / 30 s; τ ≤ 1 s is excluded by the 265 ms window SD.
  - SD of the step-baseline error: 0.031 mV from the 37 ms component,
    0.049-0.070 mV from a seconds-scale OU, 0.018 mV from a linear tilt.

Sources behind §3.4-3.5 (all read in full text in the 09-28 session):
Zeraati, Engel & Levina 2022, Nat Comput Sci, DOI 10.1038/s43588-022-00214-3
**[PubMed FT]**; Weltz, Laber & Volfovsky 2024, PMC12448677 **[PubMed FT]**.
(7a), (7b) and Bessel's correction are textbook.

## 6. How the user wants the explanations

- "Briefly" means Mode B: answer in the first sentence, then the displayed core
  equation, with the plain-language gloss alongside the jargon (project
  instruction).
- A confirmation question is a claim to check. "Half right" is said as half
  right, with the halves named; this happened four times on 10-05.
- What worked: the two-level table (§3.1), and the accounting identity
  Σ(x_j − μ)² = S_0 + n(x̄ − μ)² stated per window before taking
  expectations.
- The user's own summaries tend to fuse the analytic and computed levels and
  to reverse the sign of the demeaning bias. Check both before agreeing.

## 7. Pending items that belong to other chats (do not act on them while studying)

- **Next free decision ID is D-021.** D-019 was taken by the diameter chat
  (absorption coefficient μ). D-020 (diameter documents live in the repo) was
  decided on 2026-10-04 but is **not yet in the project log**: the write was
  refused for lack of space. The Stage-7 proposal "keep 526785799 and
  571430815 in the cohort, flagged" will therefore be D-021 if accepted.
- Still awaiting the user (from 2026-10-01): keep both cells (D-021); build
  `ss_baseline_series.py` (the SS pre-pulse variogram, 0.2-3.8 s); delete the
  stale duplicates `claude/TEEG_decisions_and_ideas_log.md` and
  `claude/TEEG_HPC_paths_reference.md`.
- Project knowledge is full (§ changelog). The diameter handoff of 2026-10-05
  calls `claude/TEEG_decisions_and_ideas_log.md` the live log. It is the
  stale duplicate; the live log is the root `TEEG_decisions_and_ideas_log.md`.
- Repo `main` on 2026-10-06 is `5a7dff1` (diameter docs). The Stage-7 code is
  unchanged since `3dfa263`, which is already pulled and run on the cluster
  **[run: git ls-remote, fresh clone]**.
