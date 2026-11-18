# Handoff: the joint fit of a node with its close neighbours (plan step F3) -- decisions D-161 to D-166 (clearance chat -> next chat)

| Date | Change |
|---|---|
| 2026-11-18 | Created at the user's request (16:22) for the chat that specifies and builds the joint fit. Branch `sci/diameter-pipeline` at `f6504b0` (code) and the commit that adds this file; decision log on `main` at `55ee935` (D-166 the last entry; next free IDs D-167, I-006). Nothing of the joint fit is specified in `specs/SPEC.md` or written in code yet. |
| 2026-11-18 (later) | D-167 [user, 16:31]: the weighting of the residuals is a parameter to tune; the readable planes are today's `select_planes` rule; a node without a close neighbour is the same fit; one pass, no repeat for now; nothing kept of the rounds' one-tube refinement. Section 2 (table), section 3, section 4 and section 8 (O1 to O3 closed) updated. Decision log on `main` at `68180f0`; next free IDs D-168, I-006. |
| 2026-11-18 (later, 2) | D-168 [user, 16:53]: the cap on free neighbours is a tunable parameter (O5 closed). Open and under discussion with the user: the neighbour's position fixed at the pilot's values or free in a narrow range (O4); what a "neighbour" is -- today one segment, proposed one chain (O6, O7). New check `tier2_sloppiness.py` (section 9) and a new open point O13 from it. Decision log on `main`: next free IDs D-169, I-006. |
| 2026-11-18 (later, 3) | 17:06, nothing decided (the user: "I'm still deciding"). The user's proposal for O13: fix a neighbour's absorption at a weighted mean over nearby nodes of its branch; check `tier2_fixed_mu.py` added (section 9). Correction to O4: today a neighbour sits at its TRACED position; the first stage's centre and depth are not stored in the features. |
| 2026-11-18 (later, 4) | D-169 [user, 17:38] changes what is built first: section 0 added (clear traces only, one round, no far-neighbour cleaning, Allen's diameter as the start, centre weights from Allen's diameter, near bumps removed). The joint fit with close neighbours (sections 2 to 7) stays decided and comes after. Section 8 now ends with the list of choices the user still has to make (8b); the user will answer them in the next chat. D-168 and D-169 added to section 2. Decision log on `main` at `30ed363`; next free IDs D-171, I-006. A second handoff exists for the defocus table on davinci (D-170): `TEEG_defocus_table_davinci_handoff_2026-11-18.md`. |
| 2026-11-18 (later, 5) | D-171 [user, 17:54]: A1 to A7 of section 8b answered (the weights' width and centre, the bump rule with multiplier 2 as a tunable parameter, no weight from the bump outwards, clear traces = Cell 19c's, the first stage gives every start but the diameter). A8 (the in-focus width meanwhile) still open: the user asked why 0.22. Decision log: next free IDs D-172, I-006. |
| 2026-11-18 (later, 6) | D-172 [user, 17:58]: the in-focus width stays 0.080 um for now (A8 closed). Decision log: next free IDs D-173, I-006. |
| 2026-11-18 (later, 7) | D-173 [user, 18:00]: in the first stage the diameter is fixed at Allen's value and the absorption and lateral centre are fitted (section 0's table, A7). Decision log: next free IDs D-174, I-006. |
| 2026-11-18 (later, 8) | Section 0's path is built and smoke-tested: commit `1bb9973` on `sci/diameter-pipeline` (SPEC Blocks 3, 5, 30, 32, 34, 36 amended, Block 39 new; notebook Cell 20). New section 0b: what was built, the commands, the knobs, what the synthetic checks show and do not show. The builder's choices are D-176 [orchestrator]. D-174 [user, the defocus-table chat, about 18:40] asks for the held diameter as the first stage's default for all its users: not in `1bb9973` (section 0b, last table row). Decision log: next free IDs D-177, I-006. |
| 2026-11-18 (later, 9) | The joint-fit chat (D-177 [user, 18:25, 18:52]: this chat refines the joint fit with close neighbours; the bumps are the 'Questions before logic' chat's). Section 13 added: four synthetic checks and the pilot's trace analysed (`checks/joint_2026-11-18/`), the chat's PROVISIONAL proposals for B1 to B6 (D-177.3 to D-177.11), the first test nodes. Section 6 corrected (where the own tube's heading comes from); section 12: the node spacing and the trace's depth scatter now measured. Decision log on `main`: next free IDs D-178, I-006. |
| 2026-11-18 (later, 10) | D-179 [user, 20:23]: of section 13.3, confirmed -- the per-chain rendering with summed shadows, rule (b) dropped, the first test nodes; read as confirmed, to check -- B1 and B2; open, with two figures (`checks/joint_2026-11-18/fig_attached_vs_free.png`, `fig_own_chain_options.png`) -- attached chains as neighbours, the own chain smoothed or raw. Section 13.2: the depth words for D-149's examples corrected. Decision log on `main`: next free IDs D-180, I-006. |

Basis tags: **[user]** said or decided by the user, with the time (Europe/Rome, 2026-11-18); **[log]** the decision
log on `main`; **[src]** read in the named file at `f6504b0`; **[run, sandbox]** executed in the handing-over
chat's sandbox, the script named; **[data]** read in files the user uploaded from Colab; **[orchestrator]** chosen by
the assistant, PROVISIONAL until the user confirms; **[reasoning]** the assistant's inference, not checked;
**[open]** neither decided nor verified.

## 0. Direction of 17:38 (D-169): what is built first

After the five clear nodes of different sizes (section 10, the last bullet) the user set the immediate work aside
from the joint fit with neighbours. Build this first; sections 2 to 7 come after it and keep their decisions.

| Point | Decided (D-169) | Today **[src, `f6504b0`]** |
|---|---|---|
| Nodes | Clear traces only, for now (D-169.9). | `clean_cell.py clear` picks nodes with no tier-2 neighbour and the least tier-1 light, optionally one per bin of Allen's diameter (`--d-bins`). The trace cannot see untraced structures: three of five such nodes were polluted (section 10). |
| Far neighbours | Their cleaning is skipped for now (D-169.7). | `clean_cell.py clean` always cleans tier 1; there is no switch to skip it. |
| Rounds | One (D-169.6); the user saw no change from round 1 to round 2. | `--rounds 1` exists. |
| Start of the diameter | Always Allen's value (D-169.2). | The work-list nodes start from the first stage's diameter and absorption (`clean_cell.py`, round 0, source "pilot"). |
| First stage | Its one-plane fit holds the diameter at Allen's value and fits the absorption and the lateral centre (D-173). | `measure_pilot` fits all three (SPEC section 2.2, Eq. (3)); no option to hold the diameter. |
| Weights | The residuals weighted by distance from the central value $v_c$ outwards, the width set by Allen's diameter: "an exponential lighlty larger than gaussian FWHM obtained from the Allen's diameter" (D-169.4). | Block 29: $w_n = e^{-\lvert v_n\rvert/\lambda}$, $\lambda$ = `measure.sgt_lambda_um` = 0.5 um for every node, centred on the line's origin. Block 30 (`fit_node`): no such weight. |
| Bumps | The near bumps that pollute a profile are removed; the example is node 2669 (D-169.5). | Nothing detects or removes a second dip on the line. |
| Own shadow | The node's own expected shadow is kept (D-169.8). | The one-tube model over the node's planes (`node_inputs.forward_factory`) and the same-node clearance product (`own_product`, D-156). |
| Close neighbour's absorption | Fixed in a node's joint fit; free when that neighbour is itself the working node (D-169.3). | Not built (the joint fit). |
| Geometry | The chain of D-164 is implemented (D-169.1). | One straight tube (`local_tube`). |

The handing-over chat's readings, PROVISIONAL, to confirm (D-169.10): the weight
$w_n = e^{-\lvert v_n - v_c\rvert/\lambda_i}$ with $\lambda_i = \kappa\,{\rm FWHM}_{{\rm A},i}$,
${\rm FWHM}_{{\rm A},i} = [(0.866\,D_{{\rm A},i})^2 + (2.355\,\sigma_{\rm r}(0))^2]^{1/2}$ (um), $D_{{\rm A},i}$
Allen's diameter of node $i$ (um), $\sigma_{\rm r}(0)$ the kernel's in-focus width (um), $\kappa$ = 1.25, $v_c$ the
fitted centre; this gives $\lambda_i$ = 0.54 um at $D_{\rm A}$ = 0.45 um and 1.32 um at 1.2 um with
$\sigma_{\rm r}(0)$ = 0.080 um **[reasoning: arithmetic]**. "Always Allen's" for the working node and for neighbours;
the first stage keeps what needs no diameter (the sharpest plane, the line's direction). Superseded for now:
D-167.1's default (no weight beyond $1/\sigma_i^2$), D-166.1's first step, D-162.4's default.

Evidence the user decided on **[data: upload of 17:25, Cell 19c; figures read by eye]**: Allen / first stage /
refined (um): 1013 0.449 / 3.200 / 2.168; 5353 0.454 / 0.803 / 0.774; 2669 0.788 / 2.981 / 2.588; 2620 0.811 /
2.056 / 1.913; 639 1.200 / 2.484 / 2.565. Node 2669: a deep narrow core at $v$ = +0.3 um and a second dip at
$v$ = -1.0 um; the one-tube model spans both and misses by about 25 gl. Nodes 1013 and 2620: blobs and a joining
branch on the line. Node 639: a flat-bottomed dip about 2 um wide at half depth; there Allen's 1.2 um is too thin
whatever the kernel **[reasoning]**. Which in-focus width that run used is not stated in the upload.

## 0b. Section 0's path as built (commit `1bb9973`, 2026-11-18)

Tags as above; **[run, sandbox]** here means executed in the building chat's sandbox on 2026-11-18. Nothing in this
section has run on Allen's data yet: the first run is the user's, notebook Cell 20.

| Point of section 0 | Built **[src, `1bb9973`]** |
|---|---|
| Nodes | Cell 20 reuses Cell 19c's examples file (`OUT/cleaning_sizes/examples_sizes_<S>.json`; nodes 1013, 5353, 2669, 2620, 639 on the pilot) or picks them with `clean_cell.py clear --d-bins`. |
| Far neighbours, rounds | `clean_cell.py clean --no-clean`: the example nodes are prepared (first stage, patch, backgrounds, line, noise) and recorded with `stopped` "not_cleaned"; no neighbour's node joins the work list, nothing is cleaned or refined (`clean_iteration.prepared_only`). The tiers are counted and recorded, not used. |
| First stage | `measure_pilot(..., d_fixed="allen")` -> `fit.fit_profile(..., d_fixed_um=2 r_Allen)` in every pass: $d$ held, $(\mu, v_0)$ fitted. Default of `--no-clean`; `--pilot-diameter free` restores the free fit. A cleaning run (no `--no-clean`) keeps the free fit. |
| Start | `node_inputs.pilot_start`: Allen's diameter, the first stage's $\mu$, centre and depth; the fit's starts are $d_0 \times$ `inference.multistart_factors` (or `--starts`). |
| Weights | `infer_nodes.py real` on a not-cleaned run: `node_inference.fit_node_recentred` with $w_n = e^{-\lvert v_n - v_c\rvert/\lambda_i}$, $\lambda_i$ = `inference.weight_kappa` (1.25) $\times$ `line_weights.allen_fwhm_um`$(D_{{\rm A},i}, \sigma_{\rm r}(0))$; $v_c$ = the fitted centre, re-centred between solves (at most `weight_irls_max_iter` = 3 solves, stop when $v_c$ moves $\le$ `weight_irls_tol_um` = 0.01 um). Covariance: the sandwich $A^{-1} B A^{-1}$ (the weights are not inverse variances). |
| Bumps | `line_weights.bump_mask`: on the mean dip over the planes used, smoothed by one line sample, a second dip whose prominence is $\ge$ `inference.bump_sd` (2) noise SDs, beyond the point where the central dip has fallen to `bump_half_depth` (0.5) of its depth; the samples from the lightest sample before it outwards get weight 0 in every plane. |
| Own shadow | Unchanged: the one-tube forward over the node's planes and the same-node clearance product at the estimate. |
| In-focus width | The run's `renderer.sigma_r0_um` (0.080 um; `--sigma-r0-um` on `clean` still changes it). |
| Close neighbour's absorption; chain | Not built (the joint fit; sections 2 to 7). |
| First stage's default for every user (D-174 [user, the defocus-table chat]) | **Not built.** `measure_pilot`'s default is still the free diameter; only `clean --no-clean` passes "allen". D-174.2 asks for a configuration field with the held diameter as the default (the estimator hash changes; Blocks 6, 10, 13 to 15, 17 follow). The davinci `nodes` and `calibration` steps of D-175 wait for that change. **[open]** |

Commands (Colab: Cell 20; here with the paths of that cell):

```
python3 scripts/clean_cell.py clean --specimen 529878215 --out OUT/cleaning_d169 --cache-dir CACHE <ALIGN> \
    --config-json CFG --examples OUT/cleaning_d169/examples_sizes_529878215.json --B 159.0 --mu 0.258 --no-clean
python3 scripts/infer_nodes.py real --specimen 529878215 --cleaning-dir OUT/cleaning_d169 --out OUT/inference_d169 \
    [--weights allen|none] [--bump-sd 2.0] [--weight-kappa 1.25] [--starts 1.0,1.6]
python3 scripts/show_focal.py --cleaning-dir OUT/cleaning_d169 --specimen 529878215 --inference-dir OUT/inference_d169
```

Outputs per node: `inference_<S>.csv` (now with `weights`, `lambda_um`, `v_cut_lo_um`, `v_cut_hi_um`, `n_solves`
before `seconds`), `inference_<S>.json` (`weights_mode`, `lambda_um`, `v_cut_um`, `bump_sd`, `n_solves`, `recentred`,
`weights`), `fit_node<id>.png` (orange: the weights; grey bands: samples of zero weight).

What the checks show **[run, sandbox]**:

| Check | Result |
|---|---|
| Synthetic cell (rendered 0.8 um, Allen 0.6 um), the smoke tests' fast configuration, `test_d169_path` | First stage: $d$ = 0.600 um (held), $\mu$ = 1.590 /um. Fit from 0.600 um: 0.8547 $\pm$ 0.0091 um, 1 solve, no cut. The same node unweighted: 0.8702 $\pm$ 0.0087 um. The fast configuration's forward (partition_vertical, 4 x 8 directions) is not the cell's renderer; neither number is a test of accuracy at 1 %. |
| Bump rule on noise alone, `checks/d169_2026-11-18/bump_false_positive_check.py` (white noise 2.5 gl, 500 draws) | On each plane's raw samples, a prominence of 2 noise SDs is reached in every profile (cut rate 1.00 with 1 to 9 planes). On the mean smoothed by one sample: 0.04 with 1 plane, 0.00 with 3, 5, 9. A second dip of depth 4 noise SDs in every plane is cut in 0.85 to 1.00 of the draws, one of 3 SDs in 0.37 to 0.55, one of 2 SDs in 0.00 to 0.19. |
| What the cut and the weights recover, `checks/d169_2026-11-18/bump_bias_check.py` (tube 1 um, noise-free, the second dip not in the model) | Second dip 1.3 um aside (SD 0.3 um, 60 gl): $d$ off by +5.3 % unweighted, +2.6 % cut, +3.5 % weights, +2.1 % both. 1.5 um aside (SD 0.6 um, 80 gl): +223 %, +18.7 %, +26.4 %, +15.8 %. **Not removed:** a second dip with no lighter sample between it and the central one (1.0 um aside, 60 gl): +96 % unweighted, +66 % with the weights, no cut. |
| Sandwich covariance, `test_smoke_node_inference.py::test_weights` (300 noisy sets, $\lambda$ = 0.6 um) | Coverage of $\hat d \pm$ SE 0.663 (nominal 0.683); mean SE 0.00637 um against the SD of $\hat d$ 0.00628 um. |

Limits the next chat should know:

- The bump rule sees a second dip only when a lighter sample separates it from the central one. A shoulder on the
  flank passes, and the weights alone leave most of its pull on $d$ (table above). A test of the profile's
  asymmetry would flag it; not built, not decided **[open]**.
- The cut is placed at the lightest sample between the two dips, where the second dip's near flank can still be
  tens of grey levels: what lies inside the cut still pulls $d$ up (the +15.8 % case).
- "The noise" in the rule is the SD of one line sample of one plane (the fit's `sigma_gl`), while the dip searched is
  a mean over planes: a structure present in one plane only is diluted by the number of planes used.
- With Allen's (thin) diameter held, the first stage's absorption comes out larger and can raise the first stage's
  `dark` flag (on a phantom rendered at 1.0 um with Allen's 0.6 um: $\mu$ 0.80 -> 2.25 /um, flag `dark`)
  **[run, sandbox: `test_pilot_fixed_diameter`]**. What a poor first-stage fit at Allen's diameter should flag is
  open (D-173).
- The fit's time on Allen's data under this path is not measured.
- No independent tester has run on `1bb9973` (written inline by the orchestrator at the user's "Implement now").

## 1. First actions in the next chat

1. Read this file, then in the decision log (`main`, `docs/project/TEEG_decisions_and_ideas_log.md`) the entries
   D-132, D-161, D-162, D-163, D-164, D-165, D-166, D-167 and the changelog rows of 2026-11-18. Read the last changelog
   rows again immediately before taking an ID: the noise-fit chat logs in the same file.
2. Build section 0 first. Before any logic (coding directive 3) get the user's answers to section 8b; the user said at 17:38 "update the handoff and i
   will reply in the other chat".
3. Write the SPEC entry of F3 (interfaces, shapes, units, oracles, tolerances), get the plan approved, then fan out
   (coding directives 5 to 9; the `scientific-coding` skill). Add the blocks to the roadmap
   (`specs/roadmap/diameter-pipeline.yaml`) in the turn their SPEC entries are written.
4. Ask the user for the outputs of notebook Cells 19b and 19c if they have not arrived (section 10): they bear on
   every diameter, the joint fit's included.

## 2. What is decided (the user's decisions of 2026-11-18, in force)

| ID | Time | Decided | By |
|---|---|---|---|
| D-161.1 | 14:25 | The joint fit of a node with its close neighbours (plan step F3, decided in principle by D-132.2 on 2026-11-13 and never built) is built now. | user |
| D-162.1 | 14:57 | In that fit all four features of a close neighbour are free: diameter, absorption, lateral shift, depth. Only its heading and tilt are held, computed from the trace. Read from the user's "for the rest it is perfect"; the user did not object when this reading was stated back at 15:05. | user (as read) |
| D-162.2 | 14:57 | The neighbour's fitted values are saved, with their standard errors, per pair (node, neighbour). | user |
| D-162.3 | 14:57 | "For now" they are working values only (the start of that node's own fit; its features where its shadow is subtracted from other nodes). Each node's delivered diameter is its own fit's. | user |
| D-162.4 | 14:57 | A parameter chooses the working values of the neighbouring nodes among Allen's, the pilot's (first stage) and the fitted ones. Default: the fitted ones. | user |
| D-162.5 | 14:57 | Where the chosen source has no value for a node, the next is used, in the order fitted, pilot, Allen. | orchestrator |
| D-164.1 | 15:13 | Tubes are built from one traced node to the next along the same branch, not as one straight tube. | user |
| D-164.2 | 15:13 | The chain runs beyond the adjacent nodes, out to the blur's reach: today's extent, `inference.tube_half_length_um` = 5 um each way. | user |
| D-164.3 | 15:13 | Within a node's fit its own stretch has one diameter and one absorption; only the geometry comes from the chain. | user |
| D-164.4 | 15:13 | A neighbour is tier 2 (fitted jointly) when the predicted bias of the node's diameter exceeds a tolerance; tier 1 (subtracted) otherwise. Formula in section 5. | user |
| D-164.5 | 15:13 | Assumed error of the neighbour's diameter 30 %, tolerance 1 %; rule (b) of today's tiers kept; the chain is a setting beside the straight tube until the two are compared. | orchestrator |
| D-166.1 | 16:22 | Order: the distant (tier-1) neighbours are cleaned first; the node and its close (tier-2) neighbours are fitted afterwards. | user |
| D-166.2 | 16:22 | That fit is one non-linear least-squares problem over the four features of the node and the four features of every close neighbour, optimised together. | user |
| D-166.3 | 16:22 | Its data are the planes of the node in which the node's shadow, with the close neighbours' shadows on it, is readable. | user |
| D-166.4 | 16:22 | The one-tube refinement with weights $w_n = e^{-\lvert v_n\rvert/\lambda}$ (today's Block 29, equation (2) of the 15:25 explanation) is not the form of this fit. The user: "I'm skeptical about using the w_n exponential weighting." | user |
| D-166.5 | 16:22 | The fit "MUST also implement the subtleties we've discussed": every row of this table from D-162 and D-164. | user |
| D-166.6 | 16:22 | The assistant's three readings, now settled by D-167.1 to D-167.3. | orchestrator, superseded |
| D-167.1 | 16:31 | The weighting of the residuals in the joint fit is a parameter to tune ("Keep as a parameter to tune"). | user |
| D-167.2 | 16:31 | The planes of the fit are those today's `select_planes` keeps: plane $k$ when $B_{i,k}\max_n(1 - \tau_{0,k}(v_n)) \ge$ `plane_snr_min` (1.0) $\times\ \sigma_i$, $\tau_0$ the node's own model at the start. | user |
| D-167.3 | 16:31 | A node without a close neighbour is the same fit with no neighbour term. | user |
| D-167.4 | 16:31 | Cleaning and joint fit are run once: "Repeat or not: for now not". | user |
| D-167.5 | 16:31 | Nothing of the rounds' one-tube refinement (Block 29) is kept in this path. The user: "I woudl say notthing what do you think?"; the assistant agreed at 16:40. | user (an inclination), assistant agreed |
| D-167.6 | 16:31 | The weighting's choices: none beyond $1/\sigma_i^2$ (default) or $w_n = e^{-\lvert v_n\rvert/\lambda}$ with $\lambda$ a parameter; the number of passes a parameter, default 1. | orchestrator |
| D-168 | 16:53 | The number of close neighbours fitted jointly is capped; the cap is a tunable parameter (default 3, the orchestrator's). | user |
| D-169 | 17:38 | Section 0. For the joint fit: the chain implemented (D-169.1); the diameter's start always Allen's (D-169.2), which makes Allen's the default of D-162.4's parameter; a close neighbour's absorption fixed, free when it is the working node (D-169.3). | user |
| D-171 | 17:54 | For section 0: the weights' width slightly above the FWHM of the Gaussian from Allen's diameter, re-centred while fitting; a bump = a second dip rising at least 2 noise SDs (multiplier tunable); the samples from it outwards get no weight; clear traces = Cell 19c's nodes; the first stage gives every start but the diameter. | user |
| D-172 | 17:58 | The kernel's in-focus width stays 0.080 um for now. | user |
| D-173 | 18:00 | In the first stage the diameter is fixed at Allen's value; the absorption and the lateral centre are fitted. | user |

Superseded on the way, do not implement: D-161.2's "the neighbour's position held" (by D-162.1); SPEC Block 30's
sentence "a joint hybrid scene with the tier-2 neighbours held at their features" (by D-161, D-162.1).

## 3. The order of the pipeline as decided (D-166.1)

1. First stage (exists): the pilot fit of each node at one plane gives start values.
2. Cleaning of the distant neighbours (exists: Blocks 25, 28, 32): the tier-1 neighbours' shadows, predicted from
   the working values (D-162.4), are added back as light on the node's patch.
3. Joint fit (to build): node and tier-2 neighbours fitted together on the node's cleaned line (section 4).
4. The fitted values are saved (D-162.2) and become working values (D-162.3, D-162.4).

One pass, no repeat for now (D-167.4); the number of passes is a parameter, default 1 (D-167.6). The rounds' one-tube refinement is not on this path (D-167.5): step 2 runs once, at the working values -- the first stage's where it converged, Allen's elsewhere (D-162.5) -- and the values saved in step 4 are read only when a repeat is switched on **[reasoning]**. What today's round loop does, for reference
**[src: `analysis/clean_iteration.py`]**: every round, for each work-list node, from the previous round's values
only (a Jacobi update): refit the background, subtract the tier-1 shadows, read the line, refine the node with the
one-tube weighted fit of Block 29, then update all nodes at once; stop when every diameter changed by less than
`cleaning.stop_rel_d` = 0.01 and no sample of an example's cleaned line by `sigma_gl`, or after `max_rounds` = 5.
After the rounds `scripts/infer_nodes.py` runs Block 30's per-node fit (one tube, unweighted, with a covariance).

## 4. The fit, in full notation (D-161, D-162.1, D-164.3, D-166.2 to D-166.6)

For a node $i$ with the set $Q_i$ of its tier-2 neighbours ($Q_i$ may be empty):

$$\big(\hat\theta_i, \{\hat\theta_q\}_{q \in Q_i}\big) = \arg\min_{\theta_i,\ \{\theta_q\}_{q \in Q_i}}\ \sum_{k \in K_i}\ \sum_{n=1}^{N_v} \frac{\big[I^{\rm clean}_{i,k}(v_n) - B_{i,k}\,\tau_k\big(v_n \mid \theta_i, \{\theta_q\}_{q \in Q_i};\ c_i, \{c_q, h_q\}_{q \in Q_i}\big)\big]^2}{\sigma_i^2}$$

- $\theta_i = (d_i, \mu_i, v_{0,i}, z_{{\rm ax},i})$: node $i$'s diameter (um), absorption coefficient (1/um), lateral
  shift along its measuring line (um) and axis depth (um). One $d_i$ and one $\mu_i$ for the whole own stretch
  (D-164.3).
- $\theta_q = (d_q, \mu_q, v_{0,q}, z_{{\rm ax},q})$ for each $q \in Q_i$: the neighbour's diameter (um), absorption
  (1/um), shift across its own axis (um) and depth (um). All free (D-162.1).
- $h_q$: the neighbour's heading and tilt (rad), computed from the trace and held (D-162.1). $c_i$, $c_q$: the
  chains of traced nodes that give the tubes' geometry (D-164.1, D-164.2), held.
- $K_i$: the planes of node $i$ in which its shadow is readable (D-166.3): those `select_planes` keeps (D-167.2). $v_n$,
  $n = 1, \dots, N_v$: the samples of the node's measuring line (um); $N_v$ = 53 today.
- $I^{\rm clean}_{i,k}(v_n)$ (grey levels, gl): the node's line in plane $k$ after the tier-1 cleaning (step 2 of
  section 3). $B_{i,k}$ (gl): the background of plane $k$ on the node's patch. $\sigma_i$ (gl): the node's noise SD.
- $\tau_k(\cdot)$ (dimensionless): the transmittance of node and close neighbours rendered together under the
  hybrid absorption (D-132.2; SPEC section 2.3 (H), Block 24), read on the line as the data are (the mean over the
  $2 n_{\rm avg} + 1$ parallel lines, pixel-integrated).
- Weighting: a parameter to tune (D-167.1). Default: no weight beyond $1/\sigma_i^2$, as written above; the other choice multiplies each term by $w_n = e^{-\lvert v_n\rvert/\lambda}$, $\lambda$ (um) a second parameter (D-167.6).
- Solver: non-linear least squares (D-166.2). Today's per-node fit uses `scipy.optimize.least_squares` (trf,
  bounds), several starts with the lowest cost kept, covariance $(J^\top J)^{-1}$ of the residuals divided by
  $\sigma_i$ **[src: `analysis/node_inference.py`]**.

Reading: one sum of squared differences between the cleaned line and the jointly rendered model, over the readable
planes and the line's samples; $4 + 4\lvert Q_i\rvert$ unknowns moved together at every iteration; nothing
alternates.

## 5. The tier rule: predicted bias (D-164.4, D-164.5)

For node $i$ and neighbour $q$:

$$\Delta\hat d_i(q) = \Big[\big(J_i^\top J_i\big)^{-1} J_i^\top\,\delta_{i,q}\Big]_d, \qquad \delta_{i,q}(k, n) = B_{i,k}\Big[S_{q,k}\big(v_n \mid (1 - \varepsilon)\,d_q, \mu_q; g_q\big) - S_{q,k}\big(v_n \mid d_q, \mu_q; g_q\big)\Big]$$

- $J_i$: the Jacobian of the node's own model profile $B_{i,k}\,\tau_k(v_n \mid \theta_i)$ (gl) with respect to
  $\theta_i$, one row per $(k, n)$, $k \in K_i$, at the current values.
- $S_{q,k}$ (dimensionless): the neighbour's shadow rendered alone, at its working values and traced geometry $g_q$.
  $\delta_{i,q}$ (gl): the error left on the line if $q$ is subtracted with its diameter off by the fraction
  $\varepsilon$.
- $q$ is tier 2 when $\lvert\Delta\hat d_i(q)\rvert / d_i > t$. $\varepsilon$ = 0.30, $t$ = 0.01 **[orchestrator]**.
- Kept beside it: rule (b) of today's tiers.

Today's rule, to be amended **[src: `analysis/cleaning.py::neighbour_tiers`]**: tier 2 when (a) the neighbour's
nearest point is within `close_lateral_um` = 2.0 um laterally and `close_defocus_um` = 0.84 um in depth of the node,
or (b) it shares no node with the own stretch and its projected footprint meets the own stretch's within
`tier_reach_um` = 3.0 um of the node, at any depth. Soma discs are tier 1.

## 6. Geometry and directions (D-162.1, D-164.1 to D-164.3)

| Object | Today **[src]** | Decided |
|---|---|---|
| The node's own tube | One straight cylinder along the heading and tilt of the first stage, `node_pipeline._pilot` (the total-least-squares line through the first stage's fitted centres of the nodes within +-`measure.line_fit_window_um`/2 = 2 um of the node along its branch) **[corrected 2026-11-18: the handoff said `candidate_search.trace_direction`, the line through the TRACED points; `node_inputs.local_tube` reads `pilot.theta_rad`, `pilot.phi_rad`]**, reaching at most `inference.tube_half_length_um` = 5 um each way; `node_inputs.local_tube`, `tube_extent`, `axis_extent`. A node nearer than `inference.extrusion_cap_margin_um` = 3 um to an end is flagged `near_end` and skipped. | Pieces from traced node to traced node along the branch, out to the same 5 um each way; one $d_i$, $\mu_i$ along them (D-164). The straight tube stays as a setting until compared (D-164.5). |
| A neighbour | Segments between consecutive traced nodes, each end with its node's diameter, absorption and offset; `node_features.featured_scene`, `FeaturedScene` (`seg_xyz`, `seg_d_um`, `seg_mu_per_um`, `seg_ids`). | The same pieces; in node $i$'s fit the neighbour takes one $\theta_q$ (section 8, O6 for what "one" covers). |
| Heading, tilt | From the trace, never fitted. | Unchanged (D-162.1). The tilt uses the traced depth; not checked against the image **[open]**. |

## 7. Working values (D-162.2 to D-162.5)

- Stored today in `NodeFeatures` (`node_id`, `d_um`, `mu_per_um`, `offset_um` (3), `source`, `flags`,
  `round_index`); `SOURCES` = ("allen", "pilot", "refined", "fine") **[src: `analysis/node_features.py`]**.
- Decided: one configuration parameter with three choices -- Allen's (twice the traced radius, the common
  absorption, no shift), the pilot's (first stage), the fitted ones -- default the fitted ones, with the fallback
  order fitted, pilot, Allen; the default asserted in the smoke suite.
- Decided: the neighbours' fitted values are written out per pair (node $i$, neighbour $q$), with standard errors.
- **[open]** the parameter's name and section; whether a fifth source name is added for values from a joint fit.

## 8. Open points to settle with the user before the SPEC entry

| # | Point | State | The handing-over chat's recommendation |
|---|---|---|---|
| O1 | Are cleaning and joint fit repeated? | **Closed, D-167.4:** one pass for now; the number of passes a parameter, default 1 (D-167.6). | -- |
| O2 | The rule for the readable planes. | **Closed, D-167.2:** today's `select_planes`, computed from the node's own model at the start. | -- |
| O3 | What becomes of Block 29's refinement. | **Closed, D-167.5:** nothing of it on this path; the first and only cleaning uses the first stage's values where it converged, Allen's elsewhere. The module and its tests stay until the joint fit is validated. | -- |
| O4 | The neighbour's shift and depth: free, free in a narrow range, or fixed. | D-162.1 made them free. At 16:53 the user leaned to fixing them at the pilot's values, or to a narrow range (depth: half the plane spacing, 0.14 um; lateral: one or two pixels, 0.11 to 0.23 um), fearing sloppy directions; the assistant's opinion was asked. Evidence: `tier2_sloppiness.py` and `tier2_neighbour_position.py` (section 9). | The narrow range about the pilot's values (the trace's where no pilot exists), both half-widths parameters with 0 = fixed, and a flag when a value ends at its bound. |
| O5 | How many neighbours are free at once, and what happens to the rest. | **Closed in part, D-168:** a cap, and the cap is a tunable parameter (default 3, the orchestrator's). **[open]** what happens to the neighbours beyond it. | Held at working values, the node flagged. |
| O6 | What a "neighbour" is, and what one $\theta_q$ covers. | Today **[src: `model/shadow.py::_segment_tube`, `neighbour_indices`; `analysis/cleaning.py::neighbour_tiers`]** a neighbour is ONE SEGMENT: a straight capped cylinder between two consecutive traced nodes, its diameter and absorption the means of its two ends, tiered on its own; a passing dendrite is several such neighbours. Proposed at 16:40, under discussion: a neighbour of the joint fit is ONE CHAIN, the run of connected segments of one branch within reach of the node's line. | One diameter, one absorption, one rigid shift and one depth per chain; chains do not share parameters with each other; the cap of O5 counts chains; the values are written back to the chain's traced node nearest the line. To confirm. |
| O7 | Neighbours attached to the own stretch (the same branch beyond +-5 um, daughters at a branch point). | Today tiered by rule (a) only. Under the predicted-bias rule many would be tier 2 **[reasoning]**. | Decide whether they share $d_i$, get their own $\theta_q$, or stay at working values. |
| O8 | How a chain's joints are rendered. | Block 24 assumes tubes that do not intersect; pieces meeting at a bend overlap on one side **[src: `model/scene_render.py` docstring]**. | Specify the joint (one union, no double absorption) and test it against a straight tube cut in two. |
| O9 | A renderer fast enough. | Block 35's fast line renderer is for one straight uniform tube. Block 24's scene renderer: 3.7 s for one two-tube render (9 planes, 116 x 324 samples, 16 x 32 directions); at 8 x 16 directions the 883 renders of `tier2_neighbour_check.py` (one and two tubes) took 627 s **[run, sandbox]**; the six-parameter fits there took 64 to 67 s each. | A line renderer for general pieces, built once and used for node and neighbours. |
| O10 | A node estimated several times (its own fit, and as a neighbour in others'). | D-162.3: the own fit is delivered. | Report the agreement of the estimates as a check; the rule for a conflict is open. |
| O11 | Cost and circularity of the tier rule: it needs $J_i$ and the neighbour's shadow at working values that may be wrong. | **[open]** | A cheap geometric pre-filter, then the rule; measure its time per pair. |
| O12 | The kernel's in-focus width (section 10). | Unresolved; affects every fitted diameter. | Read Cells 19b and 19c first. |
| O13 | A defocused neighbour's diameter and absorption are not separately determined by the node's line. | `tier2_sloppiness.py`: for neighbours 1.68 to 2.8 um out of focus the SE of $d_q$ is 12 % to 300 % of its value, while the node's own SE(d) stays within 9 % of its value without a neighbour. Bears on D-162.2 and D-162.3 (saving and reusing the neighbour's fitted values). | Fit them all the same (the node is not harmed), save them with their SE, and adopt a neighbour's fitted values as working values only when their SE is small; the threshold a parameter. |
| O14 | Where a neighbour's position comes from. | Today **[src: `scripts/clean_cell.py` (round 0), `node_features.featured_scene`]** the work-list nodes take the first stage's diameter and absorption only (source "pilot"); every node's position in the scene is the traced one plus its stored offset, and the offset is 0 until a refinement writes it. The first stage's centre and depth are used only as the start of that node's own refinement (`node_inputs.pilot_start`). | If the neighbour's position is to be "the pilot's" (the user, 16:53), the first stage's lateral offset and depth must be written into the features' offsets first. |
| O15 | The neighbour's absorption: free, or fixed. | The user's proposal of 17:06, not yet decided: fix $\mu_q$ at a weighted mean over the neighbour's node and the nodes near it on the same branch. `tier2_fixed_mu.py`: with $\mu_q$ fixed the conditioning returns to 7.4 to 9.3 and SE($d_q$) to 0.2 to 1.4 % in all five scenes; held 30 % too high it shifts the node's diameter by -2.4 to +0.9 % and the neighbour's by -12 to -14 %. | Fixed by default, the choice a parameter; the weights (by distance along the branch and by the first-stage fit's quality) and the window to settle. |

## 8b. Choices the user still has to make (to ask first; answers go to the decision log)

For section 0 (built first):

| # | Choice | Options, and the handing-over chat's suggestion |
|---|---|---|
| A1 | The weight law's constants. | **Answered, D-171.1:** the width slightly larger than the FWHM of the Gaussian from Allen's diameter, so that the weights concentrate about the dip's extreme. The factor $\kappa$ = 1.25 and the formula of section 0 are the handing-over chat's, PROVISIONAL (D-171.7); $\kappa$ a tunable parameter. |
| A2 | Where the weights are centred. | **Answered, D-171.2:** re-centred while fitting. |
| A3 | Whether the weights are a parameter. | Not answered at 17:54; D-167.1 (a parameter to tune) stands. |
| A4 | How a near bump is recognised. | **Answered, D-171.3:** a second dip separated from the central one by a rise of at least $m$ noise SDs, $m$ = 2, a tunable parameter. Searched in every used plane (the handing-over chat's, PROVISIONAL). |
| A5 | What "removed" means. | **Answered, D-171.4:** the samples from the bump outwards get no weight. |
| A6 | What a clear trace is. | **Answered, D-171.5:** the nodes notebook Cell 19c uses (`clean_cell.py clear --d-bins`, one clear node per bin of Allen's diameter). |
| A7 | The first stage's role. | **Answered, D-171.6 and D-173:** the first stage's one-plane fit runs with the diameter held at Allen's value and fits the absorption and the lateral centre; it gives the start of every parameter but the diameter (absorption, lateral centre, depth), plus the sharpest plane, the line and the background. PROVISIONAL (D-173.3): a setting for this path only; the calibration's node choice, the survey and the bias table keep the free-diameter first stage. Today **[src: `node_pipeline.measure_pilot`]** the first stage fits diameter, absorption and centre together. |
| A8 | The kernel's in-focus width while this runs. | **Answered, D-172:** 0.080 um ("Keep 0.08"); `--sigma-r0-um` kept for comparison runs. The blur under which Allen would be right is 0.20 to 0.23 um on the three thin clear nodes and 0.73 um on node 639 **[reasoning: arithmetic on dip widths read by eye]**. |

For the joint fit with close neighbours (after section 0; the joint-fit chat's proposals for B1 to B6 are in section 13, PROVISIONAL, D-177):

| # | Choice | Options, and the suggestion |
|---|---|---|
| B1 | A neighbour's position (O4, O14). | Fixed at the trace, or free in a narrow range about it (depth 0.14 um, lateral 0.23 um; 0 = fixed). The user leaned to fixed; `tier2_neighbour_position.py` shows about 10 % bias on the node per 0.2 um of error for a close in-focus neighbour. |
| B2 | The value a close neighbour's absorption is fixed at (O15). | The user's proposal of 17:06: a weighted mean over the neighbour's node and the nodes near it on the same branch. Needs: the window, the weights, and where the values come from now that the start is Allen's (the cell's common value, or absorptions fitted when those nodes were working nodes). |
| B3 | A neighbour is a chain (O6). | "Implement the chain" (D-169.1) is read as yes: one diameter, one fixed absorption and one position per chain; the cap counts chains. To confirm. |
| B4 | Same-branch neighbours and daughters at a branch point (O7). | Suggestion: the same branch beyond the own stretch takes the node's diameter; daughters held at working values, the node flagged. |
| B5 | Bends (O8). | Suggestion: the chain as one solid, a point inside any piece absorbing once; tested against a straight tube cut in two. |
| B6 | The renderer (O9). | Suggestion: a line renderer for chains of pieces, built before the fit. |
| B7 | A node estimated twice (O10). | Suggestion: the own fit delivered; the others tabulated with their difference in standard errors. |
| B8 | The tier rule's cost (O11). | Suggestion: a distance pre-filter (6 um), then the predicted-bias rule; its time measured before the plan. |
| B9 | When the far neighbours' cleaning and the repeat come back (D-169.7, D-167.4). | Both are off for now; say what brings them back. |

## 9. Evidence behind the decisions (synthetic checks; scripts and outputs in the repository)

All in `Passive Features/Diameter Re-measurement/checks/`, standalone, not pipeline code **[run, sandbox]**. Common
set-up unless stated: flat tubes, hybrid absorption, the pilot's kernel table (0.080, 0.1536, 0.3924, 0.5620 um at
0, 0.28, 0.56, 0.84 um; slope 0.79 beyond), 9 planes, 53 samples, B = 200 gl, noise-free data, so fitted minus true
is the bias.

| Script (folder `tier2_2026-11-18/`) | Result |
|---|---|
| `tier2_neighbour_check.py` | Node and neighbour 0.6 um, 0.5 /um. Fitted node diameter, truth 0.600 um, for a parallel neighbour 1.0 um aside / one crossing at 45 deg 0.84 um deeper: neighbour ignored 1.943 / 0.948 um; subtracted at its true features 0.599 / 0.557 um; subtracted with its diameter 30 % small 0.721 / 0.762 um; joint, neighbour held at truth 0.600 / 0.600 um; joint, held 30 % small 0.719 / 0.786 um; joint, its diameter and absorption free 0.600 / 0.600 um. Shadows fail to add by 0.12 / 5.79 gl. |
| `tier2_neighbour_se.py` | SE of the node's diameter at 1 gl noise: 0.77 % with no neighbour; 0.79 % / 0.87 % with the neighbour's diameter and absorption free. |
| `tier2_neighbour_position.py` | SE with the neighbour's position held / free: 0.79 / 0.79 %, 0.87 / 0.91 %, 0.84 / 0.84 %, 0.82 / 0.83 % (four scenes). Position held 0.2 um wrong shifts the node's diameter, to first order, by +10.8 % (lateral, parallel 1.0 um aside), +9.7 % (depth, crossing 0.84 um deeper), +1.6 % and +1.5 % (depth, neighbours 1.68 and 2.1 um deeper). Jacobian at the truth only. |
| `tier2_distance_sweep.py` | Bias when a parallel neighbour is subtracted with its diameter 30 % small, by lateral / depth offset (um): 1.0/0 +20.1 %; 1.5/0 +5.3 %; 2.0/0 +1.2 %; 3.0/0 +0.1 %; 0/0.84 +19.1 %; 0/1.68 +16.1 %; 0/2.8 +6.3 %; 0/4.2 +1.5 %; 1.0/1.68 +9.3 %; 2.0/1.68 +3.2 %. |
| `tier2_bias_predictor.py` | The formula of section 5 on the same points: +13.0, +5.2, +2.2, +0.4, +19.9, +15.3, +7.0, +3.1, +10.4, +4.4 %. Within about 2 points where the fitted bias is 1 to 10 %; too low where the bias is far above any tolerance. |
| `tier2_sloppiness.py` | Condition number of the Jacobian (columns scaled to unit norm) with the node's 4 parameters free / plus the neighbour's $d_q$, $\mu_q$ / plus its shift and depth: 7.3 / 8.4 / 8.5 (parallel 1.0 um aside, same depth); 6.8 / 9.9 / 10.2 (crossing 0.84 um deeper); 7.1 / 86.9 / 101.5 (parallel above, 1.68 um deeper); 7.2 / 1360 / 1529 (crossing 2.8 um deeper); 7.2 / 72.2 / 81.2 (2.0 um aside, 1.68 um deeper). The ill-conditioning comes with $d_q$, $\mu_q$ of a defocused neighbour (SE of $d_q$ 0.07 to 1.8 um), not with its position (SE of the shift 0.002 to 0.020 um, of the depth 0.004 to 0.049 um); SE of the node's own diameter 0.0046 to 0.0055 um throughout. Jacobian at the truth, 1 gl noise. |
| `tier2_fixed_mu.py` | The same five scenes with the neighbour's absorption fixed and its diameter, shift and depth free: condition number 7.6, 8.7, 9.0, 9.3, 7.4; SE of the neighbour's diameter 0.2, 0.5, 0.6, 1.4, 0.5 %; SE of the node's diameter 0.0047 to 0.0054 um. Absorption held 30 % too high, first order: the node's diameter +0.92, -2.43, +0.15, +0.01, +0.03 %; the neighbour's -14.0, -12.3, -13.5, -13.8, -13.4 %. |

Limits, all of them: one configuration per case, parallel or 45 deg tubes of one size, the model true by
construction, no noise. Not shown: that the joint fit converges from a trace error near 1 um; anything with more
than one neighbour; anything on real data.

## 10. State on the pilot, and the unresolved in-focus width (D-160, D-163, D-165)

- Pilot cell 529878215. The four earlier examples 8089, 4844, 7081, 7448 (D-149's, class `overlap`): refined
  diameters 2.44, 2.62, 3.80, 1.29 um against Allen's 0.553, 0.467, 0.446, 0.478 um, the fitted tubes wider than the
  dark bands **[data, 13:24]**. How many tier-2 neighbours they have is not known to the handing-over chat
  (`n_tier2` in `OUT/cleaning_worst/cleaning_529878215.json`).
- Three clear nodes (D-160: not near an end, no tier-2 neighbour, least tier-1 light): 5353, 3873, 4725 refined to
  0.774, 1.085, 1.033 um against Allen's 0.454, 0.536, 0.466 um (x 1.70, 2.02, 2.22) and the first stage's 0.803,
  1.003, 0.964 um; in the three planes shown the one-tube model follows the line, largest misfit about 4 gl
  **[data, 15:25; figures read by eye]**. Reading: the one-tube model holds where nothing else is on the line; the
  excess of the four earlier nodes belongs to their neighbours **[reasoning]**.
- Unresolved: `renderer.sigma_r0_um` = 0.080 um is configured, "never identified by the stacks" **[src:
  `config.py`, `calibration.kernel_from_growth`]**, and a clear node's diameter is confounded with it
  **[run, sandbox: `checks/kernel_2026-11-18/`]**: a true width of 0.22 / 0.30 um makes a 0.5 um tube fit as 0.81 /
  1.03 um with a misfit under 1 gl; the inflation falls with size (x 1.61, 1.26, 1.10, 1.04 at 0.5, 0.8, 1.2,
  2.0 um under 0.22 um). A halo of 10 to 30 % at 3 sigma: +1 to +15 %; the growth with defocus off by 0.5 to 2 x:
  -5 to +5 %.
- Built for it (D-165, `f6504b0`): `kernel.with_in_focus_width`, `clean_cell.py --sigma-r0-um`,
  `clean_cell.py clear --d-bins`, notebook Cells 19b (the clear nodes at 0.080, 0.18, 0.28 um) and 19c (one clear
  node per bin of Allen's diameter, edges 0.40, 0.45, 0.60, 0.80, 1.20, 5.0 um, at the same widths). Tested on the
  synthetic cell only. **Waiting for the user's Colab outputs**: the two tables, `ratio_by_size_529878215.png`, the
  `clear:` lines, the focal-plane figures.
- Cell 19c ran (upload of 17:25; one width, not stated which): the table and its reading are in section 0. Of the five clear nodes by size, one (5353) is clean, one (639) is a thick dendrite whose dip is about 2 um wide, three carry untraced structures **[data, figures read by eye]**. Cell 19b's table and the other widths: not received.
- D-163: a kept shadow sum changed by the images that left could come out at -1e-17 and stop the cleaning; fixed at
  `6ac81d2` (values below 0 by at most 1e-12 set to 0).

## 11. Code to start from

Paths under `Passive Features/Diameter Re-measurement/` **[src, `f6504b0`]**.

| File | What is there | For the joint fit |
|---|---|---|
| `src/allen_diameter/analysis/node_inference.py` | `fit_node(profiles_gl, B_gl, sigma_gl, forward, theta0, cfg, planes=None) -> NodeFit`; `select_planes(tau0, B_gl, sigma_gl, cfg)`. Four parameters; `forward` is a callable. | The interface to widen to $4 + 4\lvert Q_i\rvert$ parameters, or a new function beside it. |
| `scripts/infer_nodes.py` | The per-node fit of the example nodes after the rounds; a node with `n_tier2` > 0 is fitted alone and flagged `tier2_unmodelled`. | Where the joint fit replaces the lone fit. |
| `src/allen_diameter/model/scene_render.py` | Block 24: `render_scene(tubes, mus, z_planes, grid, rcfg, light_direction=+1, reduce=None)`, several tubes under the hybrid, 2-D planes. Slow (section 8, O9). | The reference for a faster line renderer of several pieces. |
| `src/allen_diameter/model/line_fast.py` | Block 35: `line_profiles` for one straight uniform tube; `SectionCache`. | Not usable for several pieces as it is. |
| `src/allen_diameter/analysis/cleaning.py` | `neighbour_tiers(fscene, own_ids, node_xyz, cfg)`; `clean_block`. | The tier rule to amend (section 5). |
| `src/allen_diameter/analysis/clean_iteration.py`, `coarse_fit.py` | The round loop; Block 29's `refine_node` (the weighted one-tube refinement). | Section 8, O1 and O3. |
| `src/allen_diameter/analysis/node_inputs.py` | `own_ids_within`, `tube_extent`, `axis_extent`, `local_tube`, `forward_factory`, `predict_factory`, `own_product`. | The own tube's geometry (section 6). |
| `src/allen_diameter/analysis/node_features.py` | `NodeFeatures`, `initial_features`, `with_rows`, `featured_scene`. | Working values (section 7). |
| `src/allen_diameter/analysis/candidate_search.py` | `trace_direction(branch, i, window_um)`. | Heading and tilt from the trace. |
| `scripts/clean_cell.py` | `survey`, `rank`, `clear`, `clean`; `--sigma-r0-um`; `clear --d-bins`. | The entry point the joint fit's run hangs on. |
| `notebooks/cleaning_pilot_2026-11-14.py` | Colab cells 0, 1, 16, 16b, 17, 17b, 18, 18b, 18c, 19, 19b, 19c. No line may start with a percent sign (Colab's `.py` export comments it out). | New cells for the joint fit. |

Tests at `f6504b0` **[run, sandbox]**: `test_smoke_pilot_run.py` 16 pass, 1 skip; `test_smoke_shadow.py` 8 pass,
1 skip; `test_smoke_cleaning.py` 6 pass, 1 skip; `test_smoke_clean_iteration.py` 9 pass, 1 skip;
`test_smoke_contamination.py` 8 pass, 1 skip; `test_smoke_render.py` 7 pass. Failing before and after this chat's
changes, listed as outdated in `specs/test_reports/2026-11-17_1754_incremental.md` (F2): four tests of
`tests/spec/test_spec_block25_shadow.py` and
`tests/spec/test_spec_block28_cleaning.py::test_boundary_block24_to_25_to_28[partition_vertical]`. The other smoke
and spec suites were not rerun on 2026-11-18. No tester run was made on 2026-11-18.

## 12. Not checked

- Nothing of this handoff's section 4 exists as code; no number in section 9 comes from real data.
- The joint fit with more than one neighbour, with a chain geometry, or started from a displaced trace.
- The node spacing of the pilot's trace (it decides how many pieces a chain has within 5 um). **[measured 2026-11-18, section 13: 1.15 um in the median, 10 pieces per own chain]**
- Whether the traced depth is good enough for the tilt. **[partly, section 13: the traced depth is smooth along a branch (a node within SD 0.04 um of the chord of its two neighbours); whether it is right, the trace cannot say]**
- The literature: three PubMed queries (the bright-field kernel; dendrite diameters by light against electron
  microscopy) and one project-knowledge query returned nothing usable on 2026-11-18; no claim here rests on a paper.

## 13. The joint-fit chat, 2026-11-18 from 18:25 (D-177)

Scope **[user, 18:25]**: this chat refines the joint fit of the working node with its close neighbours; the bumps
(D-169.5, D-171.3, D-171.4) are the 'Questions before logic' chat's. Scripts and outputs:
`checks/joint_2026-11-18/` (the weights-and-bump checks of the chat's first reply: `checks/weights_2026-11-18/`).

### 13.1 Synthetic checks **[run, sandbox]**

Flat tubes, hybrid absorption, the pilot's kernel table, 9 planes, 53 samples, B = 200 gl, noise-free.

| Script | Result |
|---|---|
| `b5_chain_joints.py` | A straight tube cut into 8 pieces of 1.18 um, one cut on the line. Pieces rendered together (Block 24, the ray history through every piece): the uncut tube, 0.000 gl. Each piece rendered alone and the shadows summed (Block 25's rule for a stretch): the dip 2.6, 6.1, 9.2 % too deep at d 0.45, 0.8, 1.2 um (mu 0.5 /um) and 2.4 % at d 0.8 um, mu 0.2 /um; a one-tube fit of it gives d -1.0, -1.3, -1.4, -0.5 % and mu +3.8, +7.6, +14.1, +2.7 %. |
| `b6_compose_rule.py` | Node and neighbour (0.6 um, 0.5 /um) rendered jointly, fitted with a composed model (node's four parameters and the neighbour's diameter free; its absorption and position at the truth). Sum of the separately rendered shadows, bias of the node's d / the neighbour's d: parallel 1.0 um aside 0.0 / -0.1 %; 0.7 um aside -0.6 / -0.5 %; 2.0 um aside and 1.68 um deeper 0.0 / -1.1 %; crossing at 45 deg over the node 0.84 um deeper -1.8 / -4.4 %, 0.84 um shallower +0.1 / +0.2 % (the node's mu 0.448 against 0.500 /um); parallel over the node 1.68 um deeper 0.0 / -2.4 %, shallower -0.2 / 0.0 %; crossing 2.8 um deeper 0.0 / -1.6 %. Product of the transmittances, the node's d: +0.3, +0.9, -0.4, -4.7, -2.5, -1.5, -2.1, -0.1 %. |
| (timing) | One 10 um tube on the line: Block 26 (the ray history at every sample) 5.4 s at 4 samples per pixel side, 26.5 s at 8; Block 35 0.30 s and 0.58 s. |
| `b5_chain_vs_straight.py` | A chain of 8 pieces of 1.15 um (d 0.6 um) read as the pipeline reads it (9 parallel lines), fitted with one straight tube: arc of radius 25 / 12 / 6 um, d -0.2 / -0.2 / +0.4 %; a kink of 10 / 20 / 45 deg at the node, +0.5 / +1.0 / +4.9 % (the kinked truths approximate at the joint: axial caps). |

Limits: one size and absorption in the last three rows, no noise, no real data; no exact reference for a bent chain.

### 13.2 The pilot's trace **[data: `reconstruction.raw.swc` of 529878215, the user's upload of 18:52]**

`trace_chain_stats.py`, `close_neighbour_nodes.py` (the pipeline's `cell.stretches`, `node_inputs.own_ids_within`,
`is_near_end`, `cleaning.neighbour_tiers`, default configuration; lengths and angles do not depend on the image
transform). The raw file carries the pipeline's node ids; the repaired `reconstruction.swc` renumbers the nodes.

- 8495 dendrite nodes on 119 stretches; 7821 not near an end. Node spacing 1.12 / 1.15 / 1.31 um (5th / 50th / 95th
  percentile); 10 pieces in a node's own chain (8 at the 5th percentile).
- The turn between a node's two pieces: 10.7 deg in the median (in the image plane), above 20 deg at 24.5 % of the
  nodes, above 45 deg at 4.5 %. A node lies off the chord of its two neighbours by SD 0.19 um laterally and 0.04 um
  in depth (independent trace errors of at most 0.16 um and 0.03 um per node, the dendrite's curvature included).
  Over the whole own chain (about 4.6 um each way) the trace leaves the line fitted to the traced nodes within +-2 um, taken through the node, by 1.7 um in the median.
- Today's tier rule: 24.7 % of the 7821 nodes have a tier-2 neighbour, 24.5 % by rule (b) alone (a free neighbour
  crossing the own stretch within 3 um, at any depth), whose depth distance is 15.3 um in the median (5th percentile
  2.8 um); 0.2 % by rule (a).
- A traced neighbour on the node's measuring line (a point of it within 0.61 um along the branch and 2.0 um across),
  by the depth distance of the nearest one:

| Within (um) | Nodes | Free / attached | Over the node / beside it |
|---|---|---|---|
| 0.84 | 17 (0.2 %) | 14 / 3 | 1 / 16 |
| 1.68 | 37 (0.5 %) | 26 / 11 | 8 / 29 |
| 2.8 | 127 (1.6 %) | 82 / 45 | 38 / 89 |
| 4.2 | 253 (3.2 %) | 135 / 118 | 66 / 187 |

  One chain at 251 of the 253 nodes. "Attached": on the node's own stretch beyond the own chain, or on a stretch
  sharing a branch point with it. The list: `close_neighbour_nodes_529878215.csv`.
- D-149's examples, the other chain's stage z minus the node's in brackets **[corrected 2026-11-18, later: the
  first version said "deeper" for the first two and "shallower" for the third, though all three lie on the same side;
  which sign is deeper along the light path is not claimed]**: 8089 an attached chain (a sibling branch from the same
  branch point) over the node (-2.0 um); 4844 an attached chain 1.15 um beside it (-2.1 um); 7081 a free chain 1.4 um
  beside it (-0.9 um); 7448 nothing on the line (tier 2 today through an axon crossing its stretch, +6.3 um). The clear nodes 5353, 3873, 4725, 1013, 2669, 2620, 639: nothing on
  the line at any depth.
- Not in these counts: the 674 nodes near a branch point or an end; untraced structures; other cells.

### 13.3 Proposals, PROVISIONAL, each for the user to confirm (D-177.3 to D-177.11)

State after D-179 [user, 20:23]: **confirmed** -- rows "B5, B6", "Tiers" and "Test nodes"; **read as confirmed, to
check** -- rows "B1" and "B2" (the user's "3) yes" read as the fourth point of the list put to them); **open, figures
delivered** -- rows "B4" and "Own chain"; not named in the answers -- row "B3".

| Choice | Proposal |
|---|---|
| B5, B6 | A chain's pieces share one ray history: each piece carries the cross-section history of an infinite tube of its direction (Block 35's method), clipped at the plane bisecting each joint. Node and neighbour chains rendered separately, their shadows summed (amends D-166.2's joint rendering); a neighbour over the node within 1.68 um in depth flags the node `overlap` (8 of the pilot's 7821 nodes). |
| Tiers | Rule (b) of today's tiers no longer makes tier 2 (amends D-164.5); the predicted-bias rule decides. |
| B4 | A close neighbour is one on the measuring line, attached or free: attached chains are 118 of the 253 cases. |
| B1 | The neighbour chain's position free in a narrow range about the trace (+-0.23 um across, +-0.14 um in depth; 0 = fixed), flagged at a bound. |
| B2 | Its absorption fixed at the mean of the first stage's absorptions (D-173) over the chain's nodes within 5 um of the line, weights $e^{-s/\ell}$ in the arc length $s$ from the point nearest the line, $\ell$ = 2.5 um; the cell's common value as the fallback. |
| B3 | One diameter, one fixed absorption, one shift and one depth per chain; the cap of D-168 counts chains. |
| Own chain | Through the trace's positions smoothed along the branch (a parameter, 0 = raw trace): a chain through the raw trace carries its kinks to the measuring line (above 20 deg at 24.5 % of the nodes, worth about 1 % of the diameter; above 45 deg at 4.5 %, about 5 %). The straight tube stays as the comparison. |
| Test nodes | Nodes 5720 to 5727 beside 7080 to 7087 (two basal stretches 1.2 to 1.8 um apart within 1.7 um in depth, each the other's neighbour; 7081 is one of D-149's examples), and 6746, 6747 (an axon 1.0 and 1.5 um beside them at their depth). |
