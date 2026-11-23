# TEEG -- the fit on clear nodes (D-169's path): state, pilot results, the kernel question -- handoff

| Date | Change |
|---|---|
| 2026-11-19 | Created at the user's request (22:02, "Now deliver a handoff for another chat") by the clearance chat ("Questions before logic"). Code on `sci/diameter-pipeline` at `7c05a01` plus this document's commit; decision log on `main`, next free IDs D-190, I-007 when written (re-read the log's last rows before taking one: other chats log too). Includes the user's correction of 22:02 ("the 2669 IS NOT fine in focus"), checked on the data in section 6. |
| 2026-11-19 (later) | The user, 22:17 (a new chat started from this handoff): "Let's first of all change the nodes we use" -- a cell that shows a set of judged clean nodes, from which the user selects four; the approach is refined after that. Built the same evening (D-194; SPEC Blocks 40, 41 and Block 36 amendment 18; notebook Cell 21): section 11, new. O2's state changed; nothing else in this document changed. Evidence: `tests/smoke/test_smoke_clearness.py`, `test_smoke_clearness_figures.py`, `checks/clear_gallery_2026-11-19/line_screen_pilot_check.out` **[run, sandbox]**. |
| 2026-11-20 | The first real run of Cell 21 (the user's upload, 16:03: the gallery folder and the preparation's records): no node judged clean under D-194; diagnosis and D-198 [orchestrator] (the verdict judges the measuring line only); section 11's last part, new; O2's state. Evidence: `checks/clear_gallery_2026-11-20/first_gallery_check.py` and `.out` **[run, sandbox]**, the run's pages **[data]**. |
| 2026-11-20 (later) | The user's four nodes and the first fit on them (D-199): notebook Cell 22; section 11.2, new. Evidence: Cell 22's code run on the synthetic cell **[run, sandbox]**. |
| 2026-11-20 (later, 2) | The user, 16:40: "could we settle this measurement problem of the diameter". Section 11.3, new: what the pilot's data pin and what they leave to the in-focus width; a proposal, not decided. Evidence: `checks/scale_2026-11-20/area_check.py` and `.out` **[run, sandbox]**; PubMed searches (abstracts only). |
| 2026-11-20 (later, 3) | The user, 16:59 and the answers after it: the in-focus width is fitted on thick nodes (D-201). Section 12, new; 11.3's proposal marked decided in part. Evidence: `kernel.with_kernel` and its smoke test (6 pass, 1 skip), the configuration's test (6 pass, 1 skip), Cell 23A run on the synthetic cell, the forward's time per evaluation measured in the sandbox. |
| 2026-11-20 (later, 4) | The user, 17:44: Cell 22's fit (WEIGHT_KAPPA 0.1 on D-199's four nodes) is dropped before it ran (D-202). Section 11.2 marked; the notebook's Cell 22 replaced by a one-line note. |
| 2026-11-20 (later, 5) | The first thick-node run (the user's upload, 18:09) and the user's "Maybe we have to account for a tilted background": section 12.1, new; D-203 (Block 42's start against a straight background). Evidence: `checks/tilt_2026-11-20/` on the upload; the current gallery script on the upload. |
| 2026-11-20 (later, 6) | The user, 18:24: estimate the background from a stripe close to the node, from 1.2 x Allen's radius per side, and carry it onto the line and into the fit; clear close neighbours first. Section 12.2, new: where the background comes from now, and how far the node's own dip reaches. Evidence: `checks/stripe_2026-11-20/` on the upload of 18:09. Not decided. |
| 2026-11-22 | The user, 2026-11-20 18:33 (refine the near-soma branches first; how far do diameter errors matter for the somatic passive/Ih fits?) and their answers (D-205): section 13, new. Evidence: `checks/diameter_reach_2026-11-22/` on the pilot's SWC (the user's upload of 2026-11-22). |
| 2026-11-22 (later) | The user, 14:53: the refinement is limited to the first 250 um from the soma, and on the pilot the soma comes first, from its images at focus in Colab (D-210; the log's D-206 to D-209 were taken by the twins chat meanwhile). Section 14, new. Evidence: the pilot's `reconstruction.swc` **[data inspected]**; SPEC Block 44's smoke test and notebook Cell 24 on the synthetic cell **[run, sandbox]**. |
| 2026-11-22 (later, 2) | The first real run of Cell 24 (the user's upload, 15:28: "the evaluation is wrong. The soma starts to be visualized at k = 107"): the automatic pick was a dendrite; the soma's plane is now chosen by eye (D-211). Section 14.1, new; section 14's description of the measure marked. Evidence: `soma_529878215.json` from the user's Drive **[data inspected]**; `checks/soma_focus_2026-11-22/` **[run, sandbox]**. |
| 2026-11-22 (later, 3) | The user, 16:05 ("Let's do it": the soma's outline per plane) and 16:25 (a larger crop with Allen's trace, D-213): sections 14.2 and 14.3, new. Evidence: the user's block and JSON from Drive and the SWC **[data inspected]**; `checks/soma_area_2026-11-22/`, `checks/soma_focus_2026-11-22/sphere_contours.py` **[run, sandbox]**. |
| 2026-11-22 (later, 4) | Wave 5 (D-215, D-216, D-218: the user's plan of 17:58 approved): section 14.4, new -- the soma's forward model and fit (SPEC Blocks 45-46, merged d1ee02f and e21dfcf), the orchestrator's provisional settings (D-219), the recovery check on simulated stacks (D-216.2). Evidence: the gates' reruns (test_smoke_soma_render.py 10 pass, test_smoke_soma_fit.py 9 pass, 1 skip), the joint noise-free fit, `checks/soma_recovery_2026-11-22/` (20f4ca1). The final check of the wave is running. |
| 2026-11-22 (later, 5) | The wave's final check came back (PASS WITH ISSUES) and was settled (D-220): section 14.4's "Final check" bullet. Evidence: the report, the reruns of the smoke suites and of the tester's spec tests against the fixed code (144d9c4). |
| 2026-11-23 | Wave 6 (D-222.5): section 14.5, new -- the corrected morphology (Block 47), the fit's figures (Block 48), the entry point `scripts/fit_soma.py` and notebook Cell 25. Evidence: the gates' reruns, the entry point's synthetic dry run. |
| 2026-11-23 (later) | Section 15, new: the kernel work of D-221/D-222 and its identifiability check (`checks/sgt_identifiability_2026-11-23/`); the wave-6 final check settled (D-223). |
| 2026-11-23 (later, 2) | Section 15.1, new [row added 2026-11-23, later, 3: it was written without one]: wave 7 merged -- the SGT kernel in the line renderer (Block 49), the fit's SGT option and fine-lattice error bars (Block 30 amendment 3), the width switch (Block 36 amendments 22-23); D-224, D-225, D-232. Evidence: `checks/sgt_psf_2d_2026-11-23/`, Block 49's and Block 30's reports, the synthetic run **[run, sandbox]**. |
| 2026-11-23 (later, 3) | Wave 7's final check came back (PASS WITH ISSUES) and was settled (D-233): section 15.2, new; 15.1's status bullet superseded and its synthetic times marked as p/16. Evidence: `specs/test_reports/2026-11-23_1355_incremental.md`, the tester's reproducers rerun on the fixes, the times measured at p/32 **[run, sandbox]**. |

Tags: **[user]** said by the user; **[log]** the decision log on `main` (`docs/project/TEEG_decisions_and_ideas_log.md`);
**[src]** read in the code at `7c05a01`; **[data]** read from the user's uploads, by eye or by a script named next to it;
**[run, sandbox]** executed in the clearance chat's sandbox, script under `checks/`; **[reasoning]** inference not
checked; **[open]** not decided. `<ws>` = `Passive Features/Diameter Re-measurement/`. Diameters in um, grey levels gl.

## 1. Where things are

| What | Where |
|---|---|
| Code, specification, checks | repository `Leonardodm00/Towards-EEG`, branch `sci/diameter-pipeline`, folder `<ws>` |
| Decision log | branch `main`, `docs/project/TEEG_decisions_and_ideas_log.md` |
| This path's specification | `specs/SPEC.md`: Block 39 (new) and the amendments Block 3 am. 2, Block 5 am. 2 and 3, Block 30 am. 2, Block 32 am. 3, Block 34 am. 2, Block 36 am. 16 and 17 |
| Notebook | `<ws>/notebooks/cleaning_pilot_2026-11-14.py`, Cell 20 (after Cells 0, 1, 16) |
| Checks behind this document | `<ws>/checks/d169_2026-11-18/` (the bump rule on synthetic data), `<ws>/checks/kernel_2026-11-18/` (kernel sensitivity, in-focus width), `<ws>/checks/kernel_tails_2026-11-19/` (in-focus width against heavier tails, real data), `<ws>/checks/clear_nodes_2026-11-19/` (the bump rule on node 2669; the dips' lateral drift) |
| Related handoffs | `<ws>/docs/TEEG_joint_fit_handoff_2026-11-18.md` (the joint fit F3, owned by the joint-fit chat, D-177; its section 0b holds the build details of this path); `<ws>/docs/TEEG_defocus_table_davinci_handoff_2026-11-18.md` (the pooled growth table on davinci; its point P4 is the in-focus width); `<ws>/docs/TEEG_same_node_clearance_handoff_2026-11-17.md` |
| The user's Colab folders | `OUT` = `/content/drive/MyDrive/diameters`; this path's runs in `OUT/cleaning_d169` (preparation) and `OUT/inference_d169` (fit; the second run overwrote the first); Cell 19c's nodes in `OUT/cleaning_sizes/examples_sizes_529878215.json`; image cache `/content/drive/MyDrive/allen_cache` **[data: the run logs]** |
| Pilot cell | specimen 529878215, SWC `H16.06.010.01.03.05.03_668702901_m.swc`, 12878 nodes, 521 planes **[data: run log]** |

| Commit | What |
|---|---|
| `1bb9973` | Block 39 and the clear-node path (D-169, D-171 to D-173; D-176) |
| `513ad9e` | the first stage holds the diameter at Allen's for every user (D-174, D-178) |
| `b10d5c2` | Cell 20's own imports (the first run stopped at its last loop on `glob`) |
| `d87d8b6` | the first run's results in the joint-fit handoff 0b; Cell 20's times measured |
| `7c05a01` | the kernel checks of 2026-11-19 |

## 2. Decisions in force

| ID | Says | Status |
|---|---|---|
| D-165 | the kernel's in-focus width is a tuning parameter (`clean_cell.py --sigma-r0-um`); the clear nodes are run at three widths (0.080, 0.18, 0.28 um) and in bins of Allen's diameter (0.40, 0.45, 0.60, 0.80, 1.20, 5.0 um) [user] | active; the three-width tables (Cells 19b, 19c) were never received |
| D-166, D-167 | the cleaning-rounds path: far neighbours cleaned first, then the joint fit; exponential residual weights refused there [user] | active; a cleaned run is fitted unweighted and uncut (D-176.4) |
| D-169 | for now: clear traces only, one round, no cleaning of the far neighbours, the node's own expected shadow kept, the diameter's start Allen's, residuals weighted from the centre with a width from Allen's diameter, near bumps removed [user] | active; built |
| D-171 | weights' width slightly above the FWHM expected from Allen's diameter; re-centred while fitting; a near bump rises at least 2 noise SDs (multiplier tunable); no weight from the bump outwards; clear traces = Cell 19c's nodes; the first stage gives every start but the diameter [user] | active; built |
| D-172 | the in-focus width stays at the configured 0.080 um for now; the option to run at another width is kept [user] | active |
| D-173, D-174, D-178 | the first stage holds the diameter at Allen's (twice the traced radius) and fits absorption and centre, for every user; `MeasureConfig.pilot_diameter` = "allen" (default) or "free" (labelled alternative) [user] | active; built (`513ad9e`) |
| D-176 | the builder's choices: the bump searched on the plane-averaged dip smoothed by one sample, beyond the central dip's half depth; re-centring in at most 3 solves to 0.01 um; sandwich covariance; weights and cut the defaults of a not-cleaned run only [orchestrator] | active, PROVISIONAL; D-176.1 misses node 2669's in-focus neighbour (section 6) |
| D-046, D-052, D-066, D-067 | heavier-tailed kernels: tried as a diagnostic in the one-plane fit (D-046); the skewed generalized t became the one-plane fit of the per-node chain of Blocks 12-13 (D-052); put into the renderer (D-066) and withdrawn the same day, the renderer stays Gaussian (D-067) [user] | active as stated |
| D-170, D-175, D-181 | the defocus table is pooled on davinci from all eligible cells; it measures the growth of the kernel's variance with defocus, not the in-focus width (P4); the near-field table read with a cubic spline (D-181, another chat) [user] | active; D-181 not built |
| D-177, D-179 | the joint fit with close neighbours (F3) belongs to the joint-fit chat [user] | active |

## 3. The path as built

Two commands per run (Colab Cell 20), on the nodes of Cell 19c **[src]**:

```
python3 scripts/clean_cell.py clean --specimen 529878215 --out OUT/cleaning_d169 --cache-dir CACHE <ALIGN> \
    <CLEAN_CFG> --examples OUT/cleaning_d169/examples_sizes_529878215.json --B 159.0 --mu 0.258 --no-clean \
    [--sigma-r0-um S0] [--pilot-diameter allen|free]
python3 scripts/infer_nodes.py real --specimen 529878215 --cleaning-dir OUT/cleaning_d169 --out OUT/inference_d169 \
    --weights allen --bump-sd 2.0 --weight-kappa 1.25 --starts 1.0,1.6 [--no-own] [--config-json FILE]
python3 scripts/show_focal.py --cleaning-dir OUT/cleaning_d169 --specimen 529878215 --inference-dir OUT/inference_d169
```

- **Preparation** (`--no-clean`): per example node the first stage (diameter held at Allen's, absorption $\mu$ and
  centre fitted, sharpest plane $k^*$, depth, direction, background), the patch, the nine planes $k^* \pm 4$
  ($\Delta z$ = 0.28 um), the 53-sample line (step 0.1144 um), the noise; nothing cleaned, no round
  (`clean_iteration.prepared_only`, record `stopped` = "not_cleaned").
- **Fit**: one tube, $\theta = (d, \mu, v_0, z_{\rm ax})$, the pipeline's forward (`node_inputs.forward_factory` ->
  `line_fast.line_profiles`, hybrid absorption, 8 x 16 ray directions, Gaussian kernel table with in-focus SD
  $\sigma_{\rm r}(0)$ = 0.080 um), starting at Allen's diameter; residuals weighted by
  $w_n = e^{-|v_n - v_c|/\lambda_i}$, $v_c$ the fitted centre, re-centred (`node_inference.fit_node_recentred`);
  $\lambda_i = \kappa\,[(\tfrac{\sqrt3}{2} D_{{\rm A},i})^2 + (2\sqrt{2\ln 2}\,\sigma_{\rm r}(0))^2]^{1/2}$ with
  $D_{{\rm A},i}$ Allen's diameter of node $i$ and $\kappa$ = `weight_kappa`; samples beyond a near bump get weight 0
  (`line_weights.bump_mask`); then the node's own expected shadow at the estimate.
- **Knobs**: in Cell 20 `WEIGHT_KAPPA` (1.25), `BUMP_SD` (2.0; 0 = off), `D169_WEIGHTS` ("allen"/"none"),
  `D169_STARTS` ("1.0,1.6"), `RERUN_D169_PREP`; at preparation `--sigma-r0-um`, `--pilot-diameter`; in the
  configuration's `inference` section, stored by the preparation and read by the fit: `z_bounds_um` (-1, 1),
  `v0_bounds_um` (-1, 1), `d_bounds_um` (0.1, 6), `mu_bounds_per_um` (0.01, 10), `plane_snr_min` 1.0,
  `bump_half_depth` 0.5, `bump_smooth_samples` 1.0, `weight_irls_max_iter` 3, `weight_irls_tol_um` 0.01,
  `max_nfev` 200 **[src: `config.py`, `infer_nodes.run_config`]**. The full table is in the clearance chat's answer to the
  user's message of 2026-11-19 15:35 and in the joint-fit handoff 0b.
- **Time**: the fit's cost grows with the diameter it reaches; per node 6 to 111 min in the first run, 3 to 49 min in
  the second **[data: run logs, section 4]**.

## 4. Pilot runs (529878215, Cell 19c's five nodes)

**[data: the user's uploads -- run 1 log and focal figure, 2026-11-19 14:57; run 2 folders `cleaning_d169` and
`inference_d169`, 20:48]**. Run 1: `WEIGHT_KAPPA` 1.25; run 2: 0.5; otherwise identical (in-focus SD 0.080 um, bump
cut 2 SDs, starts 1.0 and 1.6). Cell 19c: the cleaning rounds' one-tube refinement, 2026-11-18 (`focal_529878215.csv`
of that upload).

| Node | Allen $d$ | run 1 $\hat d \pm$ SE | run 1 $\chi^2/N$ | run 2 $\hat d \pm$ SE | run 2 $\chi^2/N$ | run 2 $\lambda$ | Cell 19c refined $d$ |
|---|---|---|---|---|---|---|---|
| 1013 | 0.449 | 1.692 $\pm$ 0.010 | 15.8 | 1.563 $\pm$ 0.017 | 6.10 | 0.216 | 2.168 |
| 5353 | 0.454 | 0.774 $\pm$ 0.011 | 4.43 | 0.769 $\pm$ 0.014 | 3.49 | 0.218 | 0.774 |
| 2669 | 0.788 | 2.728 $\pm$ 0.008 | 48.2 | 1.566 $\pm$ 0.008 | 38.9 | 0.354 | 2.588 |
| 2620 | 0.811 | 1.905 $\pm$ 0.006 | 11.6 | 1.911 $\pm$ 0.008 | 11.5 | 0.364 | 1.913 |
| 639 | 1.200 | 2.824 $\pm$ 0.007, depth at its bound | 49.8 | 2.629 $\pm$ 0.007 | 18.2 | 0.528 | 2.565 |

$\chi^2/N$: the weighted mean squared residual in noise units over the planes used. Fit times run 1: 46.7, 5.9, 86.1,
52.6, 111.0 min (305 min, plus 7.7 min preparation); run 2: 21.3, 2.7, 24.1, 26.9, 49.3 min (124 min). The bump cut
fired only on 1013 (samples below $v$ = -1.72 um dropped) in both runs. The first stage's $\mu$ at Allen's diameter:
1.082, 0.581, 0.813, 0.942, 1.241 /um; the fits' $\hat\mu$ (run 2): 0.152, 0.195, 0.228, 0.225, 0.312 /um.

## 5. What the profiles show, node by node

In-focus widths **[run, sandbox: `checks/kernel_tails_2026-11-19/dip_widths_check.py`, no model]**: full width at half
depth of the recorded dip, its deepest sample searched within max(0.5 um, 0.6 x Allen's diameter) of the line's
centre. Widest dip a tube of Allen's diameter renders in focus **[run, sandbox: `allen_width_bound_check.py`, the
pipeline's forward, $\mu$ 0.3 to 10 /um, kernels: the run's, in-focus SD 0.25 um, two heavy-tailed kernels at 0.080 um]**.

| Node | Narrowest recorded dip (plane, FWHM, depth) | Widest Allen-diameter dip, any kernel tried | What pollutes | Verdict |
|---|---|---|---|---|
| 5353 | k 135: 0.58 um, 23 gl | 0.50 (run's kernel), 0.58 (heaviest tail, nearly opaque only), 0.72 (SD 0.25 um) | nothing visible **[data: figures]** | the only clean node; its excess over Allen is the kernel-width question (section 7) |
| 2669 | k 259: 0.94 um, 88 gl | 1.01 | a second dip about 1.4 um to the left of the central one in the in-focus planes 258-260, 44-55 gl deep, a lighter point near -0.9 um between them; in five of the other six planes it merges and the dip is 2.2-2.5 um wide **[data: `bump_rule_2669_check.out`, `dip_widths_check.out`, the user's `fit_node2669.png`]** | **not clean in focus** -- [corrected 2026-11-19: the clearance chat had called it "compatible in focus" from the central dip's width alone; the user's correction of 22:02 stands] |
| 1013 | k 18: 1.10 um, 54 gl | 0.78 | a dark blob about 1.5 um wide on the node's edge at $k^*$, not followed by the trace **[data: focal figure]** | polluted |
| 2620 | k 200: 1.48 um, 89 gl | 1.04 | other processes join the node (a junction) **[data: focal figure]**; whether traced on uncounted branches or untraced: not checked | polluted |
| 639 | k 128: 2.05 um, 106 gl, flat-bottomed | 1.55 | a large dark structure 1.3-3 um to the left (profile never returns to background on that side) **[data: `fit_node639.png`]** | Allen's 1.2 um cannot make this dip under any kernel tried: Allen too thin here, or the band is two structures **[open]** |

Other observation, not followed up: the dips' half-depth midpoints move from plane to plane -- 5353 from +0.21 um
(k 132) to -0.02 um (k 139); slopes -0.05 to -0.11 um per um of depth across the five nodes; the one-tube model keeps
the axis at one lateral position **[run, sandbox: `checks/clear_nodes_2026-11-19/dip_centres_check.py`]**. 5353's
defocused dips are also asymmetric **[data: `fit_node5353.png`]**.

## 6. The near-bump rule missed 2669's in-focus neighbour

**[run, sandbox: `checks/clear_nodes_2026-11-19/bump_rule_2669_check.py` on the run-2 files]**, the rule of Block 39
with the run's settings ($m$ = 2, noise SD 1.64 gl, window = FWHM from Allen's diameter 0.708 um, half depth 0.5,
smoothing 1 sample, all nine planes used):

- **Plane by plane** (raw samples): planes 258, 259, 260 cut at $v$ = -0.80, -0.57, -0.69 um (the left neighbour
  removed); the other six planes give no cut.
- **As built** (the mean dip over the nine planes, D-176.1): no cut. In the mean the neighbour (46 gl at -1.26 um)
  sits on the central dip's flank with a saddle of 42 gl near -0.9 um, and the mean's central dip (70.6 gl) first falls
  to half its depth only at -1.72 um, beyond the neighbour, so the half-depth condition treats the neighbour as part of
  the central dip.
- Why the mean was chosen **[run, sandbox: `checks/d169_2026-11-18/bump_false_positive_check.out`]**: on one plane's
  raw samples a prominence of 2 noise SDs is reached by white noise alone in every profile (cut rate 1.00); on the mean
  smoothed by one sample 0.00-0.04.
- Options, none decided **[open]**: (a) cut on the in-focus planes only ($k^*$ and its two neighbours, smoothed),
  apply that cut to every plane; (b) per-plane cuts, each plane its own mask, on smoothed samples; (c) a lower
  half-depth fraction or none (the condition was added so that texture inside a wide flat-bottomed dip is not cut:
  cut rate 0.00 at 0.5 and 0.75, 0.04 at 0.9 in `checks/d169_2026-11-18/bump_bias_check.out`); (d) a test of the
  profile's asymmetry for shoulders that no lighter sample separates (D-176's open point (b)). Whichever is chosen,
  the false-positive rate on noise needs rechecking with `bump_false_positive_check.py`.

## 7. The kernel question: in-focus width against heavier tails

The user asked (20:48) whether a heavier-tailed kernel would fix the overestimates. **[run, sandbox:
`checks/kernel_tails_2026-11-19/kernel_tails_real_check.py`, node 5353, the pipeline's forward and fit, run 2's start
and planes, weights' width fixed at 0.218 um so that the rows compare; "unweighted" = every sample weight 1]**. The
heavy-tailed kernels: $K_\delta = (1-f)\,G_{\sigma_{\rm r}(\delta)} + f\,G_{h\,\sigma_{\rm r}(\delta)}$, a fraction $f$ of
the blur $h$ times wider than the table's at every defocus $\delta$, rendered as the mix of two renderings (the image is
linear in the kernel).

| Kernel | weighted: $\hat d$, $\chi^2/N$ | weighted, $d$ at Allen's 0.454: $\chi^2/N$ | unweighted: $\hat d$, $\chi^2/N$ | unweighted, $d$ at Allen's: $\chi^2/N$ |
|---|---|---|---|---|
| Gaussian, in-focus SD 0.080 (the run) | 0.769, 3.49 | 12.3 | 0.776, 4.56 | 6.28 |
| Gaussian, 0.15 | 0.713, 3.29 | 6.19 | 0.690, 4.51 | -- |
| Gaussian, 0.25 | 0.449, 3.89 | 3.89 | 0.329, 4.76 | 4.79 |
| $f$ 0.3, $h$ 3 | 0.800, 5.49 | 13.5 | 0.722, 5.13 | -- |
| $f$ 0.3, $h$ 6 | 0.784, 5.38 | 13.3 | 0.709, 4.79 | 5.75 |
| $f$ 0.5, $h$ 6 | 0.816, 8.85 | 16.0 | 0.645, 6.11 | 6.56 |

- Heavier tails at the configured in-focus width fit worse with either weighting and do not let Allen's diameter fit
  as well as a wider in-focus width does **[run, sandbox]**. Tails mostly add a shallow skirt; the fitted diameter is
  set by the dip's width at half depth, which they barely change (section 5's bound check: 0.50 -> 0.58 um only for a
  nearly opaque tube, while 5353's dip is 23 gl deep on a 140 gl background) **[reasoning, on these runs]**.
- A wider in-focus width brings $\hat d$ to Allen's (0.449 um at 0.25 um, weighted), but between 0.080 and 0.25 um
  $\chi^2/N$ changes by at most 0.6 while $\hat d$ moves from 0.77 to 0.45 um (0.33 um unweighted): the in-focus width
  and the diameter are not separated by this node -- the confound of D-165, now on real data. Both readings fit:
  in-focus SD near 0.25 um with Allen's diameter right, or smaller with the dendrite about 0.7 um and Allen's too thin
  **[run, sandbox; the reading is reasoning]**. The residuals are not white (background texture, see the figures), so
  the $\chi^2$ differences are not a calibrated test **[reasoning]**.
- A cap from P4's first option, on this node: 5353's narrowest dip (0.58 um at half depth) limits the in-focus SD to
  about 0.25 um (a Gaussian of FWHM 0.58 um has SD 0.246 um, and the tube's own width only adds) **[reasoning]**.
- History: October's runs found the same at a given width (a heavy-tailed kernel has a narrower core, so a smaller
  $d$) and that a free kernel width leaves a thin tube's $d$ unidentified (D-046, D-052) **[log]**.
- Not tested: power-law $|v|^{-2}$ tails (which the project's optics notes predict, mathematics section 3.3, quoted in
  I-002 **[log]**), a fitted tail fraction, tails that differ above and below focus, any node but 5353.

## 8. Open points (the user's to decide)

| # | Point | State |
|---|---|---|
| O1 | **The in-focus width (P4).** D-172 keeps 0.080 um; the davinci run does not measure it. Options put to the user on 2026-11-18, not decided: the narrowest in-focus dips of a cell as an upper bound; a fit on clean thick dendrites with the width free (width and diameter separate on thick tubes: in D-052's simulations a $\pm$5 % error in $d$ on 1.5 um tubes cost $\Delta\chi^2 \ge$ 20.8, the smallest a local search found **[log]**); a third, not yet proposed to the user: many clean thin nodes fitted with one shared width. | open; 5353 alone gives the cap of section 7 |
| O2 | **Clearness on the image.** The trace-based rule of Cell 19c (`clean_cell.py clear --d-bins`) admitted 1013, 2669, 2620; candidates: the per-plane bump test on the in-focus planes, the recorded in-focus width against the widest dip Allen's diameter can render (section 5), the fit's $\chi^2/N$ | [corrected 2026-11-19, later: was "open; nothing built"] a screen is built, PROVISIONAL (D-194 [orchestrator]; section 11): model-free, on the recorded image; it chooses the nodes shown to the user, who selects four. Its rule and thresholds are to confirm; it has not run on a real image [corrected 2026-11-20: it ran on 33 candidates and judged none clean; since D-198 it judges the measuring line only and passes 12 (section 11)] |
| O3 | **The bump rule** (section 6) | open; D-176.1 stays PROVISIONAL |
| O4 | **639**: Allen's 1.2 um against a 2.05 um flat-bottomed dip | open; look at the stack around the node |
| O5 | **Cost**: 124-305 min for five nodes on Colab; grows with the fitted diameter | open |
| O6 | **Defocused dips drift and are asymmetric** (section 5, last paragraph) | not followed up |
| O7 | Cells 19b/19c at three in-focus widths (D-165): the user's tables never arrived | open |

Suggested order, the clearance chat's opinion, not decided **[reasoning]**: O3 first (cheap; 2669 shows the miss);
then O2 to get several genuinely clean nodes per size bin; then O1 on those -- thick clean nodes for the width, thin
ones pooled -- and only then a rerun of Cell 20 with `--sigma-r0-um` at the settled width.

## 9. How to rerun the checks

The checks read the user's run folders (download `OUT/cleaning_d169` and `OUT/inference_d169` from Drive, unpack each
in its own empty folder) and the code folder; run from outside both, with `python3 -I`:

```
W="<clone>/Passive Features/Diameter Re-measurement"
python3 -I "$W/checks/kernel_tails_2026-11-19/dip_widths_check.py" CLEAN_DIR
python3 -I "$W/checks/kernel_tails_2026-11-19/allen_width_bound_check.py" "$W" CLEAN_DIR INFER_DIR      # ~2 min
python3 -I "$W/checks/kernel_tails_2026-11-19/kernel_tails_real_check.py" "$W" CLEAN_DIR INFER_DIR 5353 # ~25 min
python3 -I "$W/checks/kernel_tails_2026-11-19/kernel_tails_real_check.py" "$W" CLEAN_DIR INFER_DIR 5353 --unweighted
python3 -I "$W/checks/clear_nodes_2026-11-19/bump_rule_2669_check.py" "$W" CLEAN_DIR INFER_DIR
python3 -I "$W/checks/clear_nodes_2026-11-19/dip_centres_check.py" CLEAN_DIR
```

`kernel_tails_real_check.py` takes an optional comma-separated filter of variant names as its fifth argument; a comma
inside a name splits it. Each `.out` next to a script is the output of 2026-11-19 on run 2's files.

## 10. Not checked

- Nothing of this path has run on any cell but 529878215, and only on these five nodes.
- Run 1's fit folder was overwritten by run 2; run 1 is known from its log and focal figure only.
- The kernel checks used node 5353 only; the bound check uses the in-focus plane only.
- No independent tester has run on `1bb9973` or later commits. The smoke tests pass except
  `test_smoke_line_fast.py::test_reference`, which did not finish within the sandbox's time limit; six spec tests
  (Block 25 x4, Block 28 x1, Block 31 x1) failed before these changes and still do **[run, sandbox, 2026-11-18]**.
- Whether 2620's joining processes are traced (on branches the clear rule does not count) or untraced.

## 11. The gallery of image-judged clean nodes (Cell 21; D-194) [added 2026-11-19, later]

**[user]** 22:17: the five nodes of section 4 are no longer the working set; a cell shows a set of judged clean
nodes (the image in focus, the images across the planes, the measuring lines across the planes) and the user
selects four of them; the estimation and the removal of secondary dips are refined on those.

| What | Where |
|---|---|
| The screen (no model, no diameter) | `src/allen_diameter/analysis/clearness.py`, SPEC Block 40; thresholds in `ClearnessConfig` (section `clearness`, outside every signature) |
| The page per node and the contact sheet | `src/allen_diameter/plotting/clearness_figures.py`, SPEC Block 41 |
| Entry point | `scripts/show_clear_gallery.py`, SPEC Block 36 amendment 18 |
| Colab | notebook Cell 21: `clear --d-bins` (8 candidates per bin) -> `clean --no-clean` into `OUT/cleaning_gallery` -> the script into `OUT/cleaning_gallery/gallery` |
| Check on the pilot | `checks/clear_gallery_2026-11-19/line_screen_pilot_check.py` and `.out` |

- **The rule [orchestrator, PROVISIONAL].** On the smoothed dip $\tilde D(v_n) = B - I(v_n)$ of a line: the half-depth
  width $W$ and centre $c$; the skew $S$ = the largest shift of the midpoints at 0.75 and 0.25 of the depth from
  the half-depth midpoint, over $W$; the excess outside $E$ = the largest $|\tilde D|$ at least one $W$ from $c$, over
  the depth. Judged: $E \le 0.25$ and $S \le 0.15$ on the measuring line in the planes $k^* - 1 .. k^* + 1$ and on
  nine parallel lines 0.5 um apart, up to 2 um along the dendrite, in the plane $k^*$; along those nine lines the
  range over the median of the width at most 0.25, of the depth at most 0.30, and the range of the centre at most
  0.50 of the median width; the depth at $k^*$ at least 10 noise SDs. Score $Q$ = the largest ratio to its
  threshold; clean iff $Q \le 1$ and the depth holds. Full statement: SPEC Block 40.
- **On the five pilot nodes, the line part only [run, sandbox: the check above, on run 2's files]**: 5353 passes
  ($E$ 0.18, $S$ 0.06); 2669 ($E$ 0.63, $S$ 0.49), 1013 ($E$ 0.64, $S$ 0.89) and 639 ($E$ 0.67, no skew: the dip
  never returns on the left) fail; 2620 passes on the line ($E$ 0.15, $S$ 0.08) -- its pollution (section 5: a
  junction seen in the image) can only be caught by the lines along the dendrite, which need the patch
  (`OUT/cleaning_d169`, not in the sandbox). **Whether the part along the dendrite rejects 2620 is not checked.**
- **On synthetic patches [run, sandbox: `test_smoke_clearness.py`]**: a parallel neighbour, a crossing branch, a
  blob on the edge beside the line (0.83 of the depth), a taper and a bend are rejected; a blob of 0.42 of the depth
  on the edge passes; 200 noisy clean bands are all judged clean.
- **Not judged, reported in the table**: the half-depth width over Allen's diameter (`fwhm_over_allen`) -- a
  criterion on it would select the nodes on which Allen's diameter agrees; the planes beyond $k^* \pm 1$; the
  centre's drift from plane to plane (O6).
- **Not run** [corrected 2026-11-20: Cell 21 ran on Colab on 33 candidates, see below]: the screen on any real image; Cell 21 on Colab (its code ran here on the synthetic cell with
  stand-ins for the Colab variables). The preparation's time for 40 candidates (about an hour) is extrapolated
  from 7.7 min for five nodes, not measured.
- **Open, the user's**: the rule and each threshold (to confirm or change once the first gallery is seen); the
  four nodes; then O3 (the bump rule) and O1 (the in-focus width) on them.

### 11.1 The first real run (2026-11-20) and D-198 [added 2026-11-20]

**[data: the user's upload of 2026-11-20 16:03 -- `OUT/cleaning_gallery/gallery` and the preparation's
`cleaning_529878215.json`, `examples_gallery_529878215.json`, `features_529878215.csv`; the patches were not
uploaded]**. Cell 21 with the defaults: 33 candidates (8 in each of the bins 0.40, 0.45, 0.60, 0.80 um; 1 in 1.20 to
5.0 um, node 605: the trace-based rule found no other), **none judged clean**; 30 failed $U_W$, the width's range along the
dendrite, and on the other 3 (605, 1322, 7951) the width could not be formed on at least one line along it
[corrected 2026-11-20: first written "every node failed $U_W$"]; the eight pages shown were the best-scoring rejected ones.

- **Why [run, sandbox: `checks/clear_gallery_2026-11-20/first_gallery_check.py` on the run's stored features]**: at one
  plane a dendrite tilted in depth is in focus over a short stretch only. On 18 of 30 nodes the dip on the lines along
  it is narrowest within 0.5 um of the node (about 10 by chance: 3 of the 9 lines), and 2 um away it is a median 1.53
  times wider (0.95 to 4.51). The trace's tilt does not predict it (Spearman $\rho$ = 0.08, $p$ = 0.67, $n$ = 30 between
  $U_W$ and the trace's $\varphi$), so it cannot be corrected from the trace. **[data: the run's page of node 4725]**
  across its nine planes the in-focus stretch slides along the dendrite: the lower-left part sharp at k 169-170, the
  stretch through the node at k* = 173, a bouton about 1.5 um up the dendrite at k 176-177. Restricting the lines to
  $\pm$ 1 um and dropping the ranges still passed 2 nodes only: the lines' $E$ and $S$ are also raised by the
  background's texture against the shallower, defocused dips **[run, sandbox, the same stored features]**.
- **The measuring line's part agrees with the eye [reasoning, by eye on the eight pages; the orchestrator's reading,
  not the user's]**: 4725, 3883, 28 and 3984 pass it and look clean near the node; 2601 (a dark blob at the line's
  left), 2610 (a side structure), 6049 (blobs on both sides) and 2657 (blobs around a curved dendrite) fail it and
  look polluted. On the pilot it matched the user's reading on four of five nodes (section 11, above; 2620 passes the
  line although a junction joins it).
- **D-198 [orchestrator], superseding D-194.3 and D-194.4 as verdict criteria**: the verdict judges the measuring line
  on $k^* - 1 .. k^* + 1$ ($E \le 0.25$, $S \le 0.15$) and the depth floor; the lines along the dendrite are still
  measured and drawn, their values marked "(not judged)"; `--judge-along` (Cell 21: `GAL_JUDGE_ALONG = True`) restores
  D-194's rule. SPEC Block 40 amendment 1; Block 36 amendment 18 (the option).
- **On this run, recomputed from the stored features**, 12 judged clean: 3984, 1322 (bin 0.40); 4725, 3883, 3892,
  5353 (0.45); 28, 1865 (0.60); 2562, 2620, 2635 (0.80); 605 (1.20). All twelve are shown (at most 4 per bin).
  Of these, 2620 has the junction of section 5 and 28 a bend about 1.5 um along the dendrite; 605 is the only thick
  candidate, 637 (Allen 1.19 um) fails the line by $E$ 0.29.
- **To do on Colab**: Cell 1 (pull), then Cell 21: the preparation is reused, only the screen runs (seconds).
- **A lead for the refinement, not checked [reasoning]**: the forward model renders the line's read on the 3D tube
  with the trace's tilt; if the trace's tilt is wrong for a node, the modelled defocus across its nine planes is
  wrong too. The one sign so far is indirect (the $\rho$ above and node 4725's page, whose in-focus stretch moves
  about 2 um in depth over about 8 um along the dendrite, against a trace tilt of 6 degrees: a rough reading by eye).

### 11.2 The four nodes and the first fit on them (Cell 22; D-199) [added 2026-11-20, later] [updated 2026-11-20, 17:44: the fit dropped before it ran, D-202]

**[user]** 16:25: "Let's try the fit for the diameter of the nodes 3883 28 146 245 with a very narrow WEIGHT_KAPPA =
0.1"; asked which nodes "146" and "245" meant (they are the sharpest planes k* printed on the pages of 6049 and 2657,
and also the ids of two unscreened apical nodes), the user chose **6049 and 2657**. The four nodes: 3883 (Allen 0.529
um) and 28 (0.755), judged clean by D-198's verdict; 6049 (0.474, basal) and 2657 (0.793), not judged clean (line $E$
0.32 and 0.59, $S$ 0.55 and 0.34; blobs near the line on their pages).

- **Cell 22** (notebook): `clean --no-clean` of the four into `OUT/cleaning_fit4`, then `infer_nodes.py real` with
  `--weight-kappa 0.1` (the other settings as Cell 20: centre weights, bump cut at 2 noise SDs, starts 1.0 and 1.6)
  into `OUT/inference_fit4_k0.1` -- the folder named by $\kappa$, so a run at another $\kappa$ does not overwrite it (the
  pilot's second run overwrote the first, section 10) -- and `show_focal.py --out` into the same folder.
- **What $\kappa$ = 0.1 means here [run, sandbox: the formula of section 3 with $\sigma_{\rm r}(0)$ = 0.080 um]**:
  $\lambda$ = 0.0495, 0.0680, 0.0452, 0.0712 um for 3883, 28, 6049, 2657, i.e. 0.40 to 0.62 of a line sample (0.1144
  um); with the centre on a sample its neighbours weigh 0.08 to 0.20 and the next two 0.006 to 0.04; a plane's weights
  sum to 1.2 to 1.5 samples (4.0 to 6.3 at $\kappa$ 0.5, 9.9 to 15.1 at 1.25). The weights multiply the squared residuals
  (`node_inference`: residual $\sqrt{w}\,r/\sigma$) **[src]**. **[reasoning]** So the dips' widths barely enter; $d$ is set
  by how the central depth changes across the nine planes and trades against $\mu$ -- the confound with the in-focus
  width (section 7) applies.
- **On the synthetic cell [run, sandbox: Cell 22's code with stand-ins for the Colab variables; the smoke fixture's
  tube, true $d$ 0.80 um, its trace 0.60 um; the smoke test's coarse settings, 4 x 8 ray directions]**: $\hat d$ = 0.866
  $\pm$ 0.019 um at $\kappa$ 0.1, against 0.851 $\pm$ 0.002 (0.5), 0.855 $\pm$ 0.009 (1.25) and 0.869 $\pm$ 0.009
  (unweighted). The narrow weights add no offset of their own here and widen the standard error; the common excess of
  about 7 % over the truth is the fixture's under its coarse settings, not checked further.
- **Not run**: Cell 22 on Colab. Time not measured at this $\kappa$ (the pilot's second run, $\kappa$ 0.5: 3 to 49 min
  per node). [updated 2026-11-20, 17:44: dropped by the user before it ran (D-202); the notebook's Cell 22 is now a
  one-line note, its code in the history at `481abe4`. The choice of the four nodes, D-199.1, is not withdrawn.]

### 11.3 The scale problem: what the data pin, what they leave to the in-focus width [added 2026-11-20, later]

- **The weights do not move a clean thin node's diameter; the in-focus width does [data: sections 4 and 7]**. Node
  5353, all at $\sigma_{\rm r}(0)$ = 0.080 um: $\hat d$ = 0.774 ($\kappa$ 1.25), 0.769 ($\kappa$ 0.5), 0.776 (unweighted).
  At $\kappa$ 0.5: $\hat d$ = 0.769, 0.713, 0.449 um at $\sigma_{\rm r}(0)$ = 0.080, 0.15, 0.25 um ($\chi^2/N$ 3.49, 3.29,
  3.89): about $-2.6$ um of $d$ per um of $\sigma_{\rm r}(0)$ between 0.15 and 0.25. The weights move the polluted nodes
  (2669: 2.728 -> 1.566 um from $\kappa$ 1.25 to 0.5): they act on the neighbours, not on the scale.
- **The data pin the light the tube removes [run, sandbox: `checks/scale_2026-11-20/area_check.py`]**: with
  $a(\mu, d) = \int (1 - e^{-\mu c(v)})\,dv$, $c(v) = 2\sqrt{r^2 - v^2}$, 5353's first stage ($d$ held at Allen's 0.454
  um, $\mu$ 0.581 /um) and its fit (0.769 um, 0.195 /um) give $a$ = 0.0843 and 0.0852 um, 1 % apart; $\mu d^2$ = 0.120 and
  0.116 um. How that light splits into thickness and darkness is what the in-focus width decides.
- **The dip's area, which no normalized blur changes, is constant across the nine planes on deep isolated dips** --
  2620 coefficient of variation 1.4 %, 639 2.3 % -- drifts on polluted ones (1013 8.6 %, 2669 6.9 %), and on the faint
  5353 varies 1.8-fold (CV 19 %): its two wings differ by up to 4.8 gl, and 1.5 gl of background error over the 6 um
  line is as large as the dip's whole area (about 12 gl um). So the area is a blur-free observable on deep dips only
  **[same check]**.
- **Literature [PubMed; abstracts only, full texts not accessible through the connector]**: Streekstra and van Pelt
  2002 (Network 13:381-395, PMID 12222820): the PSF biases diameter estimates of neuronal tubes unless incorporated;
  Anthonisen et al. 2020 (Appl Opt 59:2914, doi 10.1364/AO.388265): sub-diffraction sizes from out-of-focus brightfield
  images by comparison with calibration objects of known size.
- **Proposed to the user, not decided** [updated 2026-11-20, later: the thick-node part decided, D-201; section 12]: calibrate $\sigma_{\rm r}(0)$ per cell on objects where it is identifiable --
  thick clean dendrites fitted with $\sigma_{\rm r}(0)$ free (D-052: $d$ identified on 1.5 um tubes), the thinnest
  traced processes as a near-PSF bound -- then fit the thin nodes at the calibrated value; first a simulation of how
  precisely $\sigma_{\rm r}(0)$ is recovered (target about $\pm$ 0.02 um, which gives $d$ to about $\pm$ 0.05 um at the
  slope above). It would change D-172 (the width held at 0.080 um).

## 12. The in-focus width fitted on thick nodes (D-201) [added 2026-11-20, later]

- **Decided [user, 16:59 and the answers after it]**: "a scan of 20 thick nodes with images only at focus and +-2
  planes around. I need a fast algorithm." The model is the pipeline's hybrid line renderer as it is ("do not use
  any 1D version"): `node_inputs.forward_factory` -> `line_fast.line_profiles`, the record's configuration, the
  kernel replaced through `kernel.with_kernel(renderer, sigma0, c)`. The parameters are
  $\theta = (d, \mu, v_0, z_{\rm ax}, \sigma_0, c)$; the kernel's variance at defocus $\delta$ is
  $\sigma_0^2 + c\,G(\delta)$, with $G(\delta) = \sigma_{\rm tab}(\delta)^2 - \sigma_{\rm tab}(0)^2$ and $\sigma_{\rm tab}$
  the configured table. In the Colab runs that table is Cell 7's per-cell calibrated one (Cell 16 writes it into
  `cleaning_config.json`), not the ideal-optics default. The background is $(b_{0k} + b_{1k} v)$ per plane, solved
  exactly for each $\theta$ (variable projection). The nodes have Allen's diameter >= 1.0 um, are spread out and lie
  on the planes $k^* \pm 2$. The fit makes one start, from the $k^*$ dip's features, with no multistart and no IRLS,
  and the nodes run in parallel. The settings of D-201.5 and D-201.6 are the orchestrator's (PROVISIONAL). The spec is
  SPEC Blocks 42 and 43 and Block 4 amendment 2.
- **Built [run, sandbox]**:
  - `kernel.with_kernel`: smoke test 6 pass, 1 skip; at $c = 1$ it equals `with_in_focus_width` bit for bit;
  - `config.SigmaCalibrationConfig`: outside every signature, the pinned hashes unchanged;
  - Blocks 42 and 43 as stubs (wave 4, to be written by two implementers once the user approves);
  - notebook Cell 23A: run on the synthetic cell, one node prepared on 5 planes (k* - 2 .. k* + 2), line +-4.46 um
    (79 samples of 0.1144 um), its gallery drawn.
- **Speed [run, sandbox, 5 planes, hybrid 8 x 16]**: one forward evaluation takes about 0.5 s at $d$ 0.8 um, 1.6 s at
  1.5 um and 4.2 s at 2.5 um; the per-slab loop dominates, so the section cache gains little. A whole fit is not timed
  yet. The bound: trf's `nfev` does not count the finite-difference Jacobian's 6 extra calls per iteration, so
  `max_nfev` 40 means at most about 280 forward evaluations per node; the expected number is not measured.
- **Next**:
  - the identifiability check of Block 42 (how well $\sigma_0$, $d$ and $c$ are pinned on 5 planes at $d$ 1.0,
    1.5, 2.5 um, and whether the line renderer biases $\hat\sigma_0$ against Block 24's 2-D renderer);
  - `scripts/calibrate_sigma.py` and Cell 23B;
  - the user's look at the 20 nodes (Cell 23A), dropping polluted ones;
  - then the thin nodes at the calibrated width, which would change D-172.
  If the check shows $\sigma_0$ and $c$ not separable on 5 planes, the approach goes back to the user (coding
  directive 7).

### 12.1 The first thick-node run and the tilted background (D-203) [added 2026-11-20, later]

- **What came out [data inspected: the user's upload, 18:09]**: `clear` found 53 thick candidates (Allen >= 1.0 um, not
  near an end, no tier-2 neighbour, >= 5 um from the soma) among all 8,495 ranked nodes; the 10 um spacing (3-D) kept
  7 (7, 599, 611, 629, 639, 663, 1802). Thick dendrites crowd near the soma.
- **The gallery ran code older than D-198** (its settings have no `judge_along`, which the current script always
  writes), so the lines along the dendrite were judged and 599 and 611 were rejected for `width_range`. The current
  script on the same preparation **[run, sandbox, `4823f23`]**: 7, 599, 611, 629 clean; 639 rejected (outside, skew);
  663 and 1802 rejected (the width against the flat background never closes). In Colab: rerun Cell 1 before Cell 23A.
- **The tilt [run, sandbox: `checks/tilt_2026-11-20/`]**: a straight background $L_k(v) = a_k + b_k v$ through the
  samples at $|v| \ge 3$ um on $k^* \pm 1$ has slopes of -0.3 to -0.2 gl/um (7), -1.5 to -1.3 (599), about +0.4 (611),
  about 0 (629), +3.0 to +3.2 (639), +7.5 to +8.0 (663) and -7.5 to -7.0 (1802). Measured against it, the screen's E
  halves on 7 and 599, the width of 663 and 1802 closes (1.5 to 3.4 um, against 4.2 to 4.7 um flat) but their E and S
  stay high, and 629's E rises from 0.24 to 0.28-0.30. The wings' scatter about the line is near the noise on 7, 599,
  611, 629 and 1802, and well above it on 639 (14 gl against 2.3) and 663 (8 to 9 gl against 5.7).
- **Where the tilt is handled**: the calibration fit's model already has $b_{0k} + b_{1k} v$ per plane (D-201.3);
  Block 42's start now measures the dip against the straight line (D-203, amendment 4); the clearness screen
  (Block 40) still measures against the flat $B_k$ (the user's to decide); the thin-node fit (Block 30) has one flat
  $B_k$ per plane, $I \approx B_k \tau_k$ -- to settle before the calibrated width is used on thin nodes.

### 12.2 The background's support near the node (the user's proposal, 18:24) [added 2026-11-20, later]

- **Proposed [user, 18:24]**: estimate the background's features always from the stripe, but closer to the node --
  "from like 1.20 * radius per side" -- so that it is faithful to the background that enters the measuring line and
  the fit; where traced nodes overlap or are close, a first round of clearing first. Not decided.
- **The background now [src: `node_inputs.model_background`, D-144.1]**: per plane, a plane
  $b_k(x, y) = b_{0k} + b_{xk} x + b_{yk} y$ fitted over the whole 20 x 20 um patch through the predicted shadow of every
  traced segment ($I \approx b_k (1 - S)$ on the pixels with predicted shadow $S \le 0.10$, outliers trimmed); only its
  value at the line's origin is kept, as the flat $B_k$ of the screen, of Block 42's first start and of Block 30's fit
  ($I \approx B_k \tau_k$). The plane's gradient is estimated and then dropped.
- **How far a thick node's own dip reaches [run, sandbox: `checks/stripe_2026-11-20/stripe_reach.py` on the upload
  of 18:09; nodes 7, 599, 611, 629, planes $k^* - 2 .. k^* + 2$; the dip against the straight background through
  $|v - c| \ge 3$ um]**: at $1.2\,r$ from the dip's centre ($r$ = Allen's radius) the profile is still 0.52 to 0.87 of
  the dip's depth below the background, on every plane and both sides; at $2\,r$ still 0.06 to 0.47; it stays below
  5 % only from 2.4 to 3.8 um (2.8 to 5.9 $r$). The half-depth width at $k^*$ is 1.24 to 1.58 times Allen's diameter
  (section 12.1), so $1.2\,r$ lies inside the half-depth points. Caveat: the reach is read against a background fitted
  beyond 3 um, which bounds it there.

## 13. How far from the soma diameter errors matter (D-205) [added 2026-11-22]

- **Asked [user, 2026-11-20 18:33]**: the refined morphologies feed the passive and Ih fits from somatic
  stimulation, so the near-soma branches come first; how far from the soma do wrong diameters still matter?
- **Answer [run, sandbox: `checks/diameter_reach_2026-11-22/`, its README]**: with every dendritic diameter beyond
  path distance x wrong by 20 % (either way), the passive fit's (Cm, Rm, Ra) move by less than 5 % only for
  x >= 250 um under Eyal et al. 2016's parameters (cell 060308; full refits confirm at 200 and 250 um), and for
  x >= 150 um with Rm and Cm forced to this cell's R_in and tau. Ra is set by the first 80-175 um, Rm by the
  most. Within 250 um lie the whole basal tree (to 218 um) and 59 % of the cell's membrane.
- **For the refinement order**: first the basal tree and the apical trunk and obliques to about 150 um, where
  every parameter moves; then to 250 um, which still moves Rm and Cm; the apical tuft beyond 250 um last.
- **Found on the way**: as reconstructed, the pilot's morphology cannot carry plausible passive parameters --
  Eyal's values give R_in 220 MOhm against the cell's 54.9 MOhm, and matching R_in and tau needs Cm 5.4 uF/cm2.
  Ih at rest, underestimated diameters and missing membrane are the candidates (not resolved).

## 14. The soma first: its view at focus (D-210; SPEC Block 44) [added 2026-11-22, later]

- **Asked [user, 2026-11-22 14:53]**: "let's focus on the first 250 um then from the soma. from the pilot cell let's
  evaluate the soma first. I need images at focus of the soma from colab". Logged as D-210 (D-206 to D-209 were
  taken by the twins chat while this was being written; the branch's files cite D-210 from `7108994`).
- **Scope [data inspected: the pilot's `reconstruction.swc`, path length along the SWC from the soma's centre]**:
  5509 of the 8496 dendrite nodes lie within 250 um (all 3141 basal, 2368 of 5355 apical), 3628 within 150 um;
  they carry 66.0 % of the dendritic shaft membrane with Allen's radii and no spine factor (section 13's 59 % is
  the whole cell's membrane with the spine factor F = 1.9 beyond 60 um). The soma is one SWC node, radius 5.97 um.
- **Built [SPEC Block 44; `b87e956`]**: `analysis/soma_view.py` (the traced soma's centre and radius, the crop,
  the planes, the focus measure -- the variance of `scipy.ndimage.gaussian_laplace` over the disk of the traced
  radius + 1 um, NaN on a plane not recorded -- and the sharpest plane, its argmax [corrected 2026-11-22: no plane
  is picked any more, section 14.1]); `plotting/soma_figures.py`
  (the sharpest plane with the traced circle, its centre and the primary dendrites' first nodes, beside the focus
  curve against depth; a montage, every second plane, 7 panels on each side); `scripts/show_soma.py`
  (`--synthetic` for a dry run); `SomaViewConfig`, PROVISIONAL (D-210.3). Smoke test 6 pass, 1 skip.
- **To run [Colab]**: Cells 0, 1 (it pulls the branch) and 16 (its survey is skipped when the examples file
  exists), then Cell 24. About 65 planes of 40 x 40 um are fetched once, into `CACHE_DIR`. Outputs in `OUT/soma/`:
  `soma_529878215_focus.png`, `soma_529878215_planes.png`, `soma_529878215.json` (`centre_um`, `radius_um`, `ks`,
  `focus`, `k_sharpest`, `z_sharpest_minus_soma_um`, ...) and `soma_529878215_block.npz` (the crop).
- **What to read in them**: whether the sharpest plane lies near the traced depth (`z_sharpest_minus_soma_um`);
  whether the soma's outline there matches the traced circle (radius 5.97 um); whether the focus curve has one
  clear peak -- a flat or double-peaked curve means something else in the disk (a dendrite crossing, the soma's
  own edges at two depths) drives the measure.
- **Not decided (D-210)**: how the soma is measured (outline, radius, area or volume) and whether its traced
  radius is replaced; whether D-201's thick nodes are taken only within 250 um. The 250 um path-distance cut in
  the pilot run's node selections (Block 36's examples, Block 40's candidates, Cell 23A's scan) is not coded yet.

### 14.1 The first real run: the automatic pick was a dendrite; the soma's plane by eye (D-211) [added 2026-11-22, later]

- **What happened [data inspected: `soma_529878215.json` on the user's Drive; the montage the user uploaded]**: the
  window held planes 85-149, all recorded; the traced depth is plane 117 (32.76 um). The measure peaked at k = 93
  (0.415; -6.72 um), where a basal dendrite is in sharp focus inside the traced disk; a second plateau, 0.33-0.345,
  at k = 107-112; under 0.28 above k = 117. The montage (every second plane, 7 panels a side of k = 93) ended at
  k = 107, where the user sees the soma begin.
- **Check [run, sandbox: `checks/soma_focus_2026-11-22/`]**: on a simulated stack (a smooth absorbing sphere of
  radius 6 um with a dendrite of 0.8, 1.6 or 2.4 um 6.7 um below it) the same measure peaks at the sphere's centre
  plane each time. So the failure is not the measure's for any soma: on this stack the soma's planes carry less fine
  detail than its in-focus neighbour, unlike a sharp-edged sphere. Why is not established.
- **Decided (D-211)**: the soma's plane, and the planes where it is seen, are chosen by eye; nothing is picked
  automatically until the soma's measurement is specified [user]. The window is now the traced radius + 6 um about
  the traced depth (k = 74..160 on the pilot [corrected 2026-11-22: k = 75..159, the window's ends rounding inward]), the montage spans it, every second plane aligned on the shown plane
  [orchestrator, PROVISIONAL]. Block 44 amendment 1, `c5a2bdf`.
- **To run [Colab]**: Cell 1, then Cell 24 as it is (no plane chosen: the focus figure shows the traced depth's
  plane, the montage the whole window); then set `SOMA_K` (the plane judged in focus) and `SOMA_K_RANGE` (the first
  and last plane where the soma is seen), and `MONTAGE_K = (first, last, step)` to narrow the montage, and run Cell 24
  again. The JSON records `k_traced`, `k_by_eye`, `k_range_by_eye`.
- **Open**: which plane is the soma's (the user's, from the new montage); how the soma is measured (D-210).

### 14.2 The soma's area: what the stack measures, and how much it matters [added 2026-11-22, later]

- **Asked [user, 16:05]**: "Let's do it" -- my proposal to trace the soma's outline on every plane of a range and sum
  it into a volume and a membrane area. **Corrected before any code**: on a simulated brightfield stack of a 6 um
  sphere the half-depth outline stays at about 6 um on every plane, even 8.4 um from the centre where the section is
  empty (`checks/soma_focus_2026-11-22/sphere_contours.py` **[run, sandbox]**). A plane shows the blurred silhouette of
  the whole soma; outlines give its width, not its height.
- **What the pilot's stack gives [data inspected: the block of the run of 15:54, the SWC]**: the silhouette at the
  sharp planes k = 111-121 covers 111-135 um2 (equivalent radius 5.95-6.55 um; Allen's 5.97); the stems' first nodes
  leave the soma between -4.5 and +2.4 um of its traced depth (6.9 um, stage units); spheroids within those bounds have
  327-539 um2 of membrane, 0.73-1.20 x Allen's sphere (448 um2) (`shape_areas.py`, closed form checked numerically).
- **How much it matters [run, sandbox: `soma_area_sensitivity.py`, the reach check's model]**: under the parameters
  matched to the cell (Cm 5.39 uF/cm2, Rm 3.8 kOhm cm2) a soma area x 0.75 moves the fitted Cm by +7 % (pulse) to
  +17 % (pulse + step), x 1.26 by -9 % to -13 %; under Eyal's parameters under 3 % (the soma is 1.45 % of the membrane
  the fit charges). A full refit at x 1.5 agrees with the linear step.
- **Open (the user's)**: (1) fit a soma model (a spheroid through the defocus model) to the stack, checked first on
  simulated soma stacks -- recommended; (2) bound the area and carry the range into the passive fit; (3) keep Allen's
  sphere.

### 14.3 Allen's trace over the soma view (D-213; Block 44 amendment 2) [added 2026-11-22, later]

- **Asked [user, 16:25]**: a larger crop centred on the soma, with Allen's reconstruction superimposed faintly at its
  traced diameters, the soma and the main branches.
- **Built [`soma_view.trace_segments`, `segment_mask`, `segments_at_depth`; the figures; `show_soma.py --half-um`;
  `SomaViewConfig.half_um` 40]**: every traced segment in the crop drawn as the disc of its traced radius swept along
  it -- outlined on every image (the axon dashed), filled where the trace puts it at the plane's depth; the soma's
  traced circle as before; the soma's own segments (from the soma node to the stems' first nodes) not drawn.
- **Checked on the real data [offline, the +-20 um block and the SWC]**: 358 traced segments in that crop (radii
  0.11-1.02 um, median 0.25); the trace follows the dendrites in the images. The traced axon runs about 10 um below the
  soma and crosses its disk in projection (hence dashed). At k = 117 the traced apical trunk (about 2 um wide) looks
  narrower than its dark band in the image -- a visual impression, not a measurement.
- **To run [Colab]**: Cell 1, then Cell 24 (`SOMA_HALF_UM` None = 40 um; planes already fetched at the old size are
  fetched again at the new one).

### 14.4 The soma's model and its fit (wave 5; SPEC Blocks 45-46; D-216, D-218, D-219) [added 2026-11-22, later]

- **Built [two implementers in parallel, merged after the gates]**: `model/soma_render.py` (Block 45) renders an
  ellipsoid soma -- centre (x0, y0, z0), in-plane semi-axes a, b at the angle phi, depth semi-axis c, absorption mu --
  with the partition absorption and the calibrated kernel, the far slope gamma replaceable per render (D-218);
  `analysis/soma_fit.py` (Block 46) fits theta = (x0, y0, z0, a, b, c, phi, mu, gamma) to the planes about the soma,
  each plane's grey level solved exactly (variable projection), the stems masked, and derives the membrane area S
  (Legendre's form), the volume and r_eq = sqrt(S / 4 pi) with their SEs. `far_slope` "fit" (default, D-218.1) or
  "fixed" (D-218.2, the configured slope; the user's far-slope calibration next week).
- **Provisional settings [orchestrator, D-219]**: the pixel grid itself (f = 1) and slabs of 0.14 um; the blur as the
  normalised sampled Gaussian (positive: tau <= 1; `checks/soma_fit_2026-11-22/check_transfer.py`); planes within
  10 um of the start, 0.56 um apart; a 13 um disk minus Allen's stems widened by 2 um; the start from the traced centre
  and radius with the shadow's orientation from its moments.
- **Cost [run, sandbox]**: one render of 36 planes of a 280 px crop 5.4 s, 380 MB; a fit 50-100 renders, 8-21 min on 2
  shared CPUs (about 3-4 s per render alone).
- **Recovery on simulated stacks [run, sandbox; `checks/soma_recovery_2026-11-22/`, D-216.2]**: the pilot's
  acquisition and kernel, the truth rendered finer than the fit (f 2, 0.07 um), noise 3 gl, the start offset as from
  Allen's trace (0.8, -0.6, +1.5 um; radius x 0.9).
  - Ellipsoids, c = 3, 4.5, 6 um: c within 0.009 um (0.3 %), S within 0.08 %, all within 2.2 SE: the fit's own
    discretisation does not bias it.
  - The far slope 20 % steeper than configured: fitted, recovered (0.9499 against 0.95) and c, S unbiased; held at the
    configured 0.79, c 0.74 um low (-16.5 %) and S 7.5 % low -- why gamma is fitted (D-218).
  - A pyramidal soma (the ellipsoid plus a cone toward the apical side, true area 10.6 % above the ellipsoid's): c right
    (+0.5 %), but S 5.1 % low and r_eq 2.6 % low -- the cone goes into the in-plane axes and the centre; chi2/dof 13.3
    against 9.1 for an ellipsoid, the cone's shadow in the residual of the in-focus planes. With masked stems as well:
    S 6.1 % low. These errors are model error, 100-400 times the SEs: the SE does not show them; chi2/dof and the
    residual's structure do.
- **Final check [independent tester, `specs/test_reports/2026-11-22_1905_incremental.md`]**: PASS WITH ISSUES, seven
  findings NOTE or MINOR, none changing a number on the default path; settled by the orchestrator (D-220, 144d9c4):
  `show_soma.py`'s refusals as SPEC says, the start's settings in `SomaFitConfig`, Block 44's plane window for Block 46,
  the "linear" continuation required by the soma fit, `calibrate_sigma.py` exiting 1 when no node is fitted. Its 101
  spec cases are on `tester/diameter-pipeline-2026-11-22_1905`, with the 14:40 run's, awaiting the user.
- **Next [pending the user]**: Block 47 (the corrected morphology: the
  equivalent sphere at the fitted centre, each stem's first node moved along its own direction onto the fitted surface,
  D-216.3) and Block 48 (the fit's figures) as wave 6, with `scripts/fit_soma.py` and notebook Cell 25 by the
  orchestrator; there is no entry point for the fit yet.

### 14.5 The fitted soma into the morphology; the fit's figures; Cell 25 (wave 6; SPEC Blocks 47-48; D-222) [added 2026-11-23]

- **Built [two implementers in parallel, merged after the gates]**: `analysis/soma_morphology.py` (Block 47): the SWC's
  soma node moved to the fitted centre with the radius of equal area, r_eq = sqrt(S / 4 pi) (D-216.3); each stem's
  first node moved along the stem onto the fitted surface; a stem whose next nodes lie inside the surface loses them and
  its first node goes where it leaves the surface; a first node inside that is a branch point or a leaf stays, flagged
  (D-222.5); `swc_io.edited_lines` writes the result in the original's format (Allen's four decimals kept byte for byte).
  `plotting/soma_fit_figures.py` (Block 48): data, model and residual per plane with the fitted outline, Allen's circle
  and the sphere of equal area; the parameters and the planes' scales and residuals by depth; the stems before and
  after.
- **Run it [Colab, Cell 25]**: after Cells 0, 1, 16 and 24; `SOMA_FIT_K` (default Cell 24's `SOMA_K`, else the traced
  depth), `FAR_SLOPE` "fit" (D-218.1) or "fixed" with `SLOPE0`. Outputs in `OUT/soma_fit`: `soma_fit_<S>_planes.png`,
  `_summary.png`, `_morphology.png`, `soma_fit_<S>.json` and the corrected `reconstruction_soma_<S>.swc`.
- **Dry run [run, sandbox]**: `fit_soma.py --synthetic` recovers a planted soma to 0.007 um (S +0.17 %, chi2/dof 9.07)
  in about 2 min. Two earlier dry runs went wrong for a reason in the dry run itself: the synthetic cell's image covers
  part of the crop and only planes -7..7, and composing it with a uniform background left a step the fit explained
  (S +9 %); the soma is now rendered on a uniform background alone.
- **Reading the real fit**: chi2/dof near the noise's level means the ellipsoid fits; a wedge in the in-focus residuals
  toward the apical side is a pyramidal soma's signature (S then about 5 % low, `checks/soma_recovery_2026-11-22/`).
- **Next**: the final check of wave 6; then the kernel wave (D-221, D-222: the switch to the calibrated sigma_0 and c;
  the SGT kernel in the node's joint fit), starting with a check that (beta, xi) separate from the diameter.

## 15. The kernel of the node fit: the switch to the calibrated sigma_0 and the SGT option (D-221, D-222) [added 2026-11-23]

- **Decided [user, D-221, D-222]**: sigma_0 stays 0.080 um (D-172) with a switch to Block 42's calibration (the weighted
  means of sigma_0 and c, `kernel.with_kernel`); the node's joint fit across planes (Block 30) gets the SGT kernel as an
  option -- the shape (beta, xi) fitted per node with zeta = 1, zeta fitted as a further option, the width the Gaussian
  law's FWHM at each slab-plane defocus; the Gaussian stays the default; the cleaning keeps it. Not built yet: the plan
  waits for the user's approval. [corrected 2026-11-23: approved at 07:43 (D-224), with the error bars on p/32 for both
  kernels; being built as wave 7, section 15.1]
- **Identifiability check [run, sandbox; `checks/sgt_identifiability_2026-11-23/`]**: a 1-D model of the node's line
  (partition absorption, the default kernel law, planes k* +- 4, noise 1.1 gl on 150 gl), validated against Block 35 to
  1.8e-10.
  - No degeneracy between d and the shape: freeing (beta, xi) multiplies SE(d) by 1.00-1.10, freeing zeta as well by
    1.01-1.13; the ill-conditioning lies inside the shape (beta against xi, correlation 0.97-1.00). Thin nodes (0.4 um)
    fix the shape loosely, but d stays unbiased. zeta is well determined and trades with v0, not with d. Expect xi = 0
    (a member of the family) in about half the fits of Gaussian-like nodes.
  - The Gaussian fit on non-Gaussian truths of the same FWHM biases d by +4 to +13 % on 0.4 um nodes, +0.7 to +2.9 % at
    0.8 um, at most 2.6 % in magnitude on thick nodes -- up to 6.5 SE on thin nodes and 32 SE on dark thick ones; its
    SE's 1-SE coverage falls to 0.03-0.25. The SGT fits recover d within 1.9 % at 0.4 um and 0.24 % above, with coverage
    0.66-0.72 at 1 SE.
  - Side finding: on Blocks 26/35's lattice (h = p/8) the derivative of the model with respect to d and v0 has not
    converged, so Block 30's SE(d) and corr(d, v0) carry errors of tens of percent on thin nodes (the values are fine);
    on a lattice of p/32 they converge.
  - Not tested: a width off the law (the in-focus-width confound), tilt, oblique headings, caps, neighbours, the hybrid
    absorption, background texture.

### 15.1 Wave 7: how the SGT kernel enters the line renderer, the error bars, the width switch (D-224, D-225) [added 2026-11-23]

- **The 2-D kernel [run, sandbox; `checks/sgt_psf_2d_2026-11-23/`; D-225.1]**: the kernel that is the SGT across the
  measuring line and the law's Gaussian along it agrees with D-222.1's isotropic PSF on flat tubes (at most 0.04 noise
  SD) but not on tilted ones (up to 1.9 noise SD); fitted with the shape free to isotropic data it moves d by -5 %
  (-1.2 SE) at 0.4 um and -3 % (-4.5 SE) at 0.8 um (shape (1.3, 0, 1), tilt 0.3 rad). So the SGT path renders the
  isotropic PSF (transfer h_hat(a |nu|)) by FFT on lattices coarsened with each pair's width, the pixel box's transfer
  exact (SPEC Block 49, `model/line_sgt.py`, `line_fast.line_profiles(..., kernel=SGTKernel(beta, xi, zeta))`); the
  Gaussian path (kernel None) is unchanged. The skew option stretches the isotropic PSF by zeta on one side of the
  measuring line and 1/zeta on the other; the kernel is truncated at 12 law SDs, its dropped mass reported
  (`line_sgt.truncated_mass`; about 1.6 % for the heavy-tailed (2, 1, 1)). Block 26 is not extended.
- **The fit [D-225.5]**: Block 30 amendment 3 -- the shape searched in log10 beta, xi, ln zeta within Block 12's bounds;
  xi = 0 (the generalized normal) is a converged end, not a bound; one start from the node's Gaussian estimate at the
  Gaussian member; under `--kernel sgt` the node's estimate is the SGT fit's, the Gaussian fit's beside it
  (`gauss_d_um`). The same-node clearance keeps the Gaussian fit (D-222.4).
- **The error bars [D-224.2, D-225.6]**: from central differences of the model at `inference.cov_fine_factor` = 32 with
  absolute steps (2e-3 um, 1e-3 /um, 2e-3 um, 2e-3 um, 1e-3, 1e-3, 1e-3); `cov_source` names the Jacobian used. Cost:
  Block 35's Gaussian path takes 3.5 s (flat 0.4 um) to 61 s (tilt 0.3 rad, 0.8 um) per warm call at p/32 against
  0.17 and 3.5 s at p/8 (hybrid, 8 x 16 directions, shared sandbox CPU): about nine such calls per Gaussian fit, so
  Cell 20's runs (6 to 111 min per node) roughly double; `--cov-fine-factor 0` (Cell 20's `COV_FINE_FACTOR`) gives the
  fit's own Jacobian back.
- **The width switch [D-225.7; SPEC Block 36 amendment 22]**: `--kernel-width fixed|calibrated --sigma-json
  sigma_<S>.json` in `clean_cell.py`, `infer_nodes.py`, `fit_soma.py`; `calibrate_sigma.py` now writes
  `summary_growth_scale`; "calibrated" re-anchors the configuration's base table with `kernel.with_kernel` at the two
  weighted means; `clean_cell.py` records the base renderer so that `infer_nodes.py`'s own switch does not compound c.
  Notebook: Cell 16's `KERNEL_WIDTH`, `SIGMA_JSON` (passed to Cells 20, 25, 26; Cell 23's thick-node preparation stays
  fixed); Cell 26 fits Cell 20's nodes with the SGT kernel beside the Gaussian.
- **Status** [updated 2026-11-23, 14:00; superseded 2026-11-23 by 15.2's status]: contract `9ea1abe`; Block 30 amendment 3 merged (`cf86785`), Block 49 merged
  (`5dbc9c1`; its one issue, oracle (iii) at the raised reach infeasible here, settled by D-232); the orchestrator's
  wiring (`0b20f32`, `bb650a7`); pushed at `16eb8be`; the independent final check running on that commit.
- **Cost, measured [run, sandbox; Block 49's report]**: per warm call (hybrid, 8 x 16 directions, planes -4..4),
  Gaussian / SGT zeta = 1 / SGT skew: 0.09-2.48 / 0.37-4.96 / 6.1-22.8 s at line_fine_factor 8; 1.3-44 / 2.4-105 /
  13-142 s at 32 (the error bars' lattice). The skew option is 5-10 times the Gaussian per call. On the synthetic
  cell (partition absorption, 4 x 8 directions) one node took 27 s for the Gaussian fit with error bars at p/16 and
  133 s with the SGT stage added [corrected 2026-11-23: both at p/16, not the default p/32; 15.2 gives the times at 32] (d 0.7968 +- 0.0018 um against the Gaussian's 0.7947 +- 0.0017; beta 2.09 +- 0.10,
  xi at its edge 0: the synthetic blur is Gaussian).

### 15.2 Wave 7's final check and its settlement (D-233) [added 2026-11-23, later]

- **The verdict [run, sandbox; `specs/test_reports/2026-11-23_1355_incremental.md`, an Opus tester on `16eb8be`]**:
  PASS WITH ISSUES. Every changed block meets its oracles against references written from the spec alone: Block 49
  within 1.3e-6 of the 1-D model at a raised reach (8 shapes, levels 0-2) and within 1.6e-5 of an exact model of the
  truncated kernel at the default reach, the fit box's corners included; the Gaussian member within 3.4e-10 of Block
  35's path, which is bit-identical to the base commit; Block 30's fine error bars equal to rule (D) by hand within
  1e-9; the SGT option's coverage 0.700 of 300 sets. No regression in the smoke suites or the earlier spec suites (the
  six stale cases of Blocks 25, 28 and 31 that SPEC names aside).
- **Fixed [D-233.1-5]**: the switch's recorded base renderer read back by `config_from_dict`'s rule (a configuration
  with Allen's JPEG tables failed under `infer_nodes.py --kernel-width`); the kernel-width smoke suite's script runner
  (section 6's command ran nothing); a failed SGT stage keeps the node's Gaussian record and its same-node clearance
  (the node's estimate stays the SGT's, written as failed); `clean_cell.config_sections` takes the switch's record as an
  argument (no module state); `fit_soma.py` closes its three figures.
- **Stated [D-233.6-12]**: `inference.cov_fine_factor` 32 by default; the covariance at an active bound is marginal over
  the bound parameter (the full inverse, then NaN rows and columns: Block 30's convention since its entry); without
  `--kernel-width`, `infer_nodes.py` takes the preparation's law, with "fixed" its base table; `validate()`'s
  constraints on the knobs; `sgt_ln_zeta_bounds` within +-ln 2.1 (Block 49's lattices do not move with zeta inside it);
  a missing `--sigma-json` refused with status 2; Block 49's oracle (viii) at the fit box's corners.
- **Time at the default p/32 [run, sandbox; the tester's end-to-end file under the pilot-run suite's fast configuration:
  partition absorption, 4 x 8 directions, one start, 9 planes; this sandbox]**: the Gaussian stage 68-93 s a synthetic
  node (10-33 s with `--cov-fine-factor 0`); with the SGT stage (zeta held) 351-499 s a node in all; the skew 769 s for
  one node with the fit's own Jacobian. On the two synthetic nodes the SGT fit ends with xi at its edge, beta 2.08-2.10,
  d 0.0020 um (about 1 SE) above the Gaussian fit's.
- **Left untested by the final check**: the real cell (SPEC section 7); the SGT path on a depth-dependent law with
  non-Gaussian members against an independent reference; the hybrid at 8 x 16 directions on the SGT path; the SGT stage
  on the clear-node path (`clean --no-clean`) end to end; production timings.
- **Status** [2026-11-23, later]: the tester's spec tests merged into `sci/diameter-pipeline` (`e0b1ab1`, D-224.3; its
  two strict xfails, F1 and F3, now plain tests); the fixes, the spec's statements and the report's decisions in `eda8fa7`;
  D-233 on `main` (`13f75f8`). The fixes are not yet re-checked by an independent tester: on the roadmap the blocks whose
  modules include `config.py` show as pushed until the next run, while Blocks 36 and 48 stay done there (their scripts'
  changes lie outside the roadmap's module map, which follows SPEC's Module lines).
