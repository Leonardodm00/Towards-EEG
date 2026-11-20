# TEEG -- the fit on clear nodes (D-169's path): state, pilot results, the kernel question -- handoff

| Date | Change |
|---|---|
| 2026-11-19 | Created at the user's request (22:02, "Now deliver a handoff for another chat") by the clearance chat ("Questions before logic"). Code on `sci/diameter-pipeline` at `7c05a01` plus this document's commit; decision log on `main`, next free IDs D-190, I-007 when written (re-read the log's last rows before taking one: other chats log too). Includes the user's correction of 22:02 ("the 2669 IS NOT fine in focus"), checked on the data in section 6. |
| 2026-11-19 (later) | The user, 22:17 (a new chat started from this handoff): "Let's first of all change the nodes we use" -- a cell that shows a set of judged clean nodes, from which the user selects four; the approach is refined after that. Built the same evening (D-194; SPEC Blocks 40, 41 and Block 36 amendment 18; notebook Cell 21): section 11, new. O2's state changed; nothing else in this document changed. Evidence: `tests/smoke/test_smoke_clearness.py`, `test_smoke_clearness_figures.py`, `checks/clear_gallery_2026-11-19/line_screen_pilot_check.out` **[run, sandbox]**. |

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
| O2 | **Clearness on the image.** The trace-based rule of Cell 19c (`clean_cell.py clear --d-bins`) admitted 1013, 2669, 2620; candidates: the per-plane bump test on the in-focus planes, the recorded in-focus width against the widest dip Allen's diameter can render (section 5), the fit's $\chi^2/N$ | [corrected 2026-11-19, later: was "open; nothing built"] a screen is built, PROVISIONAL (D-194 [orchestrator]; section 11): model-free, on the recorded image; it chooses the nodes shown to the user, who selects four. Its rule and thresholds are to confirm; it has not run on a real image |
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
- **Not run**: the screen on any real image; Cell 21 on Colab (its code ran here on the synthetic cell with
  stand-ins for the Colab variables). The preparation's time for 40 candidates (about an hour) is extrapolated
  from 7.7 min for five nodes, not measured.
- **Open, the user's**: the rule and each threshold (to confirm or change once the first gallery is seen); the
  four nodes; then O3 (the bump rule) and O1 (the in-focus width) on them.

