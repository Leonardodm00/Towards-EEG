# Implementation plan: neighbour cleaning and per-node inference (option 2) on the pilot cell (2026-11-13)

| Date | Change |
|---|---|
| 2026-11-13 | Created at the user's request (13:01) after D-129: cleaning first, then option 2, option 1 only if option 2 is not satisfactory; the pilot cell 529878215 in Colab. Code facts at `34a5196` on `sci/diameter-pipeline` **[src]**; the toy numbers of section 5 from this chat's checks **[run, sandbox]**. Nothing below is implemented. Roadmap: NC, NC1-NC4 (`0274ce8`). |
| 2026-11-13 (later) | Revised at the user's request (14:11) for the coding directives v2 of the same day: the main session plans and orchestrates, independent Opus agents write the blocks in parallel, Opus agents make the final check. New section "Fan-out: waves, blocks, files"; the steps regrouped into three waves of 5, 4 and 3 blocks (SPEC numbers 23-34, proposed); one file per block, so the iteration moves to `analysis/cleaning_iteration.py` and the recovery to `analysis/recovery.py`; a scene renderer added as a block of its own (the checks' generator, O11); scripts and notebook cells stay with the orchestrator. No step dropped; nothing implemented. |
| 2026-11-13 (later, 2) | The user's answers of 14:35 (D-130) entered in the table of choices: O1-O7 and O9-O11 settled, O8 after the survey, O12 at the judgment; the waves approved; the table route's production and inversion (S7-S8) paused while this branch is tried; theory topics T14 and T15 added. Wave 1's SPEC entries drafted (Blocks 23-27) with proposed tolerances, awaiting the user's agreement before the contract commit. Consequences flagged to the user: the hybrid's cost and its negative planes in dark stain (SPEC Block 21) now sit inside the per-node fit; a shadow removed from a neighbour in the node's light path is not additive. |

Tags: **[src]** read in the code at the commit above; **[SPEC]** `specs/SPEC.md`; **[log]** the decision log on `main`;
**[run, sandbox]** computed in the chat of 2026-11-13; **[user]** the user's words; **[proposal]** the assistant's, not
decided; **[open]** not decided.

## Scope and order (D-129)

- Problem **[user]**: a real node's measuring line carries the shadows of other branches, near in depth or laterally;
  the phantoms the table is built from are clean.
- Order **[user]**: (1) cleaning; (2) option 2, the node's features found by optimisation on the cleaned data; (3) the
  results judged; (4) option 1, the inversion table on cleaned lines, only if (3) is not satisfactory.
- Cleaning **[user]**: iterative; the neighbours' features start at Allen's diameters and are updated by a coarse fit.
- Where **[user]**: one cell, the one in Colab (529878215, the pilot cell of D-089).
- Evidence asked **[user]**: the patch images around the node and the measuring line, before and after the cleaning, on
  several example nodes.
- Not in this plan: option 1 (roadmap NC4), other cells, davinci, the switch to the hybrid absorption (D-084.2),
  structures that are not traced (other cells, untraced axon, spines): they stay with the background estimate.

## Fan-out: waves, blocks, files

Procedure: the `scientific-coding` skill, section 10. Orchestrator = the main session (Fable 5.2 on 2026-11-13);
implementers = one fresh general-purpose agent per block on Opus, each in its own worktree; final checker = the
integration tester on Opus. A block joins a wave when every interface it touches is fixed in SPEC.md, and no file
belongs to two blocks. Block numbers are proposed; they are assigned when the SPEC entries are written.

| Wave | Block | Step | Files owned by its agent (plus `tests/smoke/test_smoke_<name>.py`) | Consumes | Choices settled before its contract |
|---|---|---|---|---|---|
| 1 | 23 features table | C1 | `analysis/node_features.py`, `loading/features_io.py` | `plane_iteration.Scene` (exists) | O3 |
| 1 | 24 scene renderer | the checks' generator | `model/scene_render.py` | Blocks 2, 4 (exist) | O11 |
| 1 | 25 shadow image | C2 | `model/shadow.py` | Block 23's contract; Block 4 as oracle | O4, O5 |
| 1 | 26 line renderer | F1 | `model/line_render.py` | Block 4 as oracle; Block 5's profile sampler | O10 |
| 1 | 27 contamination survey | C0 | `analysis/contamination.py` | `calibration_scene` (exists) | O4, O5 |
| 2 | 28 cleaning operator | C3 | `analysis/cleaning.py` | Blocks 24, 25 | O1 |
| 2 | 29 coarse fit | C4 | `analysis/coarse_fit.py` | Blocks 24, 25 | O2 |
| 2 | 30 per-node fit | F2 | `analysis/node_inference.py` | Block 26 | O9 |
| 2 | 31 cleaning figures | C6 | `plotting/cleaning_figures.py` | the contracts of Blocks 25, 28 | -- |
| 3 | 32 the iteration | C5 | `analysis/cleaning_iteration.py` | Blocks 23, 25, 28, 29 | O6 |
| 3 | 33 recovery on phantoms | F4 | `analysis/recovery.py` | Blocks 24, 28, 30; Blocks 9, 21 (exist) | O12 (its numbers are reported, not gated) |
| 3 | 34 inference figures and report tables | E | `plotting/inference_figures.py` | the contract of Block 30 | -- |
| held | the joint fit with a close neighbour | F3 | `analysis/node_inference_joint.py` | Block 30 | O8, after the survey's run in Colab; inline or a wave of its own |

Why this grouping **[proposal]**: the line renderer's only oracle is Block 4, so it goes in wave 1 and the inference
code no longer waits for the cleaning code, only for cleaned data; the scene renderer goes first because the checks of
Blocks 25, 28, 29, 32 and 33 build their inputs with it; the survey reads only existing code, so its Colab run can
start after wave 1.

| Who | What |
|---|---|
| Orchestrator, inline, never an agent | the SPEC entries with their tolerances and the roadmap blocks (under NC1, NC2, NC3, in the turn each entry is written); `config.py` (`CleaningConfig`, `InferenceConfig`), package `__init__` files; the stubs and each wave's contract commit; `scripts/clean_cell.py`, `scripts/infer_nodes.py`; notebook Cells 16-18; the decision log; every issue an agent raises (directive 7); the gate and the merge of every block |
| Implementer agents (Opus) | one block each, its files only; both checks of the skill's section 6 on synthetic inputs; a report, or `STATUS: ISSUE` |
| Final checker (Opus) | the integration tester in the chat on each merged wave, scope from the skill's `scripts/changed_blocks.py` |
| The user | approves this plan and each wave's choices and tolerances before its contract commit; runs Colab: the survey after wave 1 (its shares settle O7 and O8), Cell 16 and Cell 17 after wave 3; the judgment (NC3). Asked during a wave only when the approach itself is in question |

Per wave, on `sci/diameter-pipeline`:

```
git commit -m "contract: wave <w>"        # stubs, config, smoke-test skeletons; SPEC entries complete
git worktree add ../wt-block-<n> -b wip/diameter-pipeline-block-<n> <contract commit>     # one per block; wip/ never pushed
# the wave's agents launched in one message: Agent tool, general-purpose, model "opus", the skill's brief
git diff --name-only sci/diameter-pipeline...wip/diameter-pipeline-block-<n>              # gate: its own files only
git merge-base --is-ancestor <contract commit> wip/diameter-pipeline-block-<n>            # gate: on the contract
# merge, then the final check on Opus; then SPEC statuses, the report committed, worktrees and wip/ branches removed
```

At the end of each wave the user gets, in this order: the change report (every change the orchestrator made to the
mathematics, units, conventions, estimators, validation targets or tolerances, or the statement that there was none);
the final check's verdict and how each finding was settled; each block with its code and the results obtained.

Limit: the sandbox cannot reach Allen's server **[SPEC, Block 16.3]**, so every agent's check and the final check run
on synthetic scenes; the checks on real data are the Colab runs and are named as skipped in each report.

## What exists and what each piece lacks for the cleaning

| Piece | Where **[src]** | What it does today | What the cleaning needs in addition |
|---|---|---|---|
| The cell's scene | `analysis/cell.cell_scene` -> `plane_iteration.Scene` | every dendrite segment's two ends (image frame, um), Allen's radii at both ends, the node ids, the soma discs | per-node features that change by round (diameter, absorption coefficient, fitted centre and depth); the scene holds Allen's radii only |
| A neighbour's predicted shadow | `analysis/calibration_scene.shadow_depths`, `isolation_breakdown`, `shadow_masks` (Block 10.S, D-101.2) | the dip (gl) of a flat tube of the segment's larger end radius, one absorption coefficient for the cell, one Gaussian at the segment end's defocus, at a lateral distance; used to mask pixels and to reject calibration nodes | an image of the shadow on the patch, plane by plane, from each segment's own features; it has never been subtracted |
| Patch and background | `calibration_scene.fetch_patch`, `planar_background`, `stripe_support` (+-25 um, `CalibrationConfig.patch_half_um`) | the square of pixels about a node over a plane range; a plane fitted through the far pixels | reuse as is; the cleaning's patch size is a choice (O5) |
| Renderer | `model/render.render_transmittance`, `synthetic_block` (Block 4) | one straight tube, slabs of 0.02 um, the kernel table, the camera chain | several tubes in one scene; the only multi-tube generator is the tests' provider, which multiplies separately blurred transmittances (`tests/smoke/test_smoke_node_pipeline.make_provider`) (O11) |
| First-stage fit | `node_pipeline.measure_pilot` (Block 5) | at the sharpest plane: a Gaussian-blurred tube fitted for diameter, absorption coefficient and centre; tilt from the line fit | a candidate for the coarse fit (O2); it reads a block from a provider, so a cleaned block can be served to it unchanged |
| Full chain | `node_pipeline.measure_node` (Block 13) | the plane iteration and the SGT fit | not used by this plan's first version |
| The pilot cell in Colab | `notebooks/phase2_colab.py`, Cells 0-15; `pilot/pilot_<specimen>.csv` | alignment (Cell 12); first-stage fits on the stretches of the chosen nodes (Cells 4a-4b) and on the survey's node sample (Cell 13); calibration (Cells 7, 15); `run_cell.py` on every dendrite node needs a table (Cell 10b) | new Cells 16 (cleaning), 17 (inference), 18 (the report). No first-stage fit of every node is produced by a cell today: C4 and C7 produce it. Not checked: which outputs the user's Drive holds |

Cost fact behind step F1 **[SPEC 13.5]**: one rendered and fitted realization takes 48.4 s on average with Block 4
(10 phantoms, sandbox), 35.8 s of it rendering. Block 4 cannot sit inside a per-node optimiser.

## Steps: the cleaning (roadmap NC1)

Each step is a SPEC block (numbers from 23 on; 20 is S3d, 22 is D-121's loop) written before its wave's contract commit
and entered in the roadmap under NC1 in the same turn; who writes it and in which wave is in the section "Fan-out:
waves, blocks, files" [corrected 2026-11-13 (later): was "becomes a SPEC block when it starts"]. Logic waits for the
choices of the section "Choices to settle before the logic".

| Step | What is built | Module (all new unless stated) | Check before real data | Output |
|---|---|---|---|---|
| **C0** contamination survey | for every dendrite node of the pilot cell: its traced neighbours' predicted shadow on its measuring line, relative to its own dip, with today's predictor and Allen's radii; classes: clean, diffuse, close and in focus, overlapping in projection | `analysis/contamination.py`, on `calibration_scene` | on synthetic scenes the index is 0 for an isolated tube and ranks scenes of growing pollution in order | a CSV, one row per node; the share of nodes per class; the example nodes (O7) |
| **C1** features table | per dendrite node and round: diameter, absorption coefficient, lateral offset, depth, their source (`allen`, `coarse`, `fine`), flags | `analysis/node_features.py`; `loading/features_io.py` | round trip exact; round 0 equals Allen's diameters and the trace | `features_<specimen>_round<t>.csv` |
| **C2** shadow image | for a target node's patch and planes: each neighbour segment's absorbed fraction as an image, from its end nodes' features and the calibrated kernel; segments as short straight tubes | `model/shadow.py` | against Block 4's render of the same tube: agreement fixed in the SPEC entry; zero for a segment beyond 6 kernel SDs; mass per plane equal across planes | array (planes, height, width), dimensionless |
| **C3** cleaning operator | the cleaned patch from the observed patch, the plane's background and the summed shadow (O1); pixels where a neighbour overlaps the node's own footprint in projection are flagged, not cleaned | `analysis/cleaning.py` | exact inverse on a synthetic scene built by the same rule; the flag fires on a tube directly above the node | cleaned block in the `fetch_zblock` contract, a flag mask |
| **C4** coarse fit | each node's features from its own cleaned line (O2) | `analysis/coarse_fit.py` or `measure_pilot` on a provider that serves the cleaned block | recovers the features of an isolated rendered tube within the tolerance of the SPEC entry | one features row |
| **C5** the iteration | the rounds of the section "The iteration, stated", the stop rule (O6), a record per round | `analysis/cleaning_iteration.py` [corrected 2026-11-13 (later): one file per block; was `analysis/cleaning.py`] | on synthetic scenes started from diameters 25 % too thin: the neighbours' features converge and the cleaned line equals the isolated node's line within tolerance, for diffuse and defocused neighbours; the close in-focus case is expected to end flagged (section "What the toy check says") | the features of every round; a convergence table |
| **C6** figures | per example node: the patch in several planes before, the predicted shadow, the patch after; the measuring line before and after with the first-stage fit; the change per round | `plotting/cleaning_figures.py` | runs on a synthetic scene; computes nothing scientific | PNG per node; a contact sheet |
| **C7** Colab Cell 16 | C0, then C5 and C6 on the pilot cell: the example nodes first, then every node | `scripts/clean_cell.py`; `notebooks/phase2_colab.py` Cell 16 (the orchestrator's, inline) | the dry run on a cached node | the files above in the run folder |

## The iteration, stated

Objects: $i$ a target node; $P_i$ its patch (pixels within the half-size of O5); $K_i$ its planes; $I_{i,k}(x,y)$ the
observed grey level of plane $k \in K_i$ at pixel $(x,y) \in P_i$ (measured); $B_{i,k}(x,y)$ that plane's background at
the pixel (the planar fit through the far pixels, gl); $Q_i$ the neighbour segments of node $i$ (O4); for a segment $q$,
$g_q$ its geometry (its two ends' positions and depths) and $\theta^{(t)}_q$ the features of its two end nodes at round
$t$ (diameter in um, absorption coefficient in 1/um); $\hat\sigma_{\rm r}$ the calibrated kernel law;
$s_{q,k}(x,y \mid \theta^{(t)}_q, g_q, \hat\sigma_{\rm r})$ the predicted absorbed fraction of segment $q$ in plane $k$
at $(x,y)$ (dimensionless, step C2).

Round 0: for every dendrite node $n$, $\theta^{(0)}_n$ = (twice Allen's radius, the cell's starting absorption
coefficient (O2)). For each round $t \ge 1$, each node $i$, each $k \in K_i$ and each $(x,y) \in P_i$:

$$I^{(t)}_{i,k}(x,y) = I_{i,k}(x,y) + B_{i,k}(x,y)\sum_{q \in Q_i} s_{q,k}\big(x,y \mid \theta^{(t-1)}_q, g_q, \hat\sigma_{\rm r}\big), \qquad \theta^{(t)}_i = \mathrm{coarse\ fit\ of\ node}\ i\ \mathrm{on}\ \{I^{(t)}_{i,k}\}_{k \in K_i}.$$

The first formula adds back, in grey levels, the light the neighbours are predicted to have removed; the second
refits the node on what is left. Every node's round-$t$ features are computed from the round-$(t-1)$ features of the
others, so the result does not depend on the order of the nodes and a round can run in parallel. It stops by O6. The
additive form is the renderer's own for tubes that do not overlap in projection (SPEC 2.3, Eqs. 5-6 are linear in
the absorbed light); the alternative divides the observed plane by the neighbours' predicted transmittance (O1).

## Steps: the inference, option 2 (roadmap NC2), and its judgment (NC3)

| Step | What is built | Module | Check before real data | Output |
|---|---|---|---|---|
| **F1** line renderer | the forward model on the measuring line only: the profiles across a straight tube in the planes $K_i$ for a given diameter, absorption coefficient, lateral offset, axis depth and tilt -- slabs of `dzeta_um`, each with its own kernel, the profile sampler's averaging and the pixel integration | `model/line_render.py` (custom: no library renders a depth-dependent blur) | against Block 4's full render read by `profiles.sample_profile`: agreement fixed in the SPEC entry, flat and tilted, in and off the plane; time per call reported | array (planes, samples), dimensionless |
| **F2** per-node fit | bounded least squares (`scipy.optimize.least_squares`, several starts) of F1 to the cleaned profiles of the planes $K_i$; parameters and data by O9; the noise level from the patch's far pixels; estimates, their covariance from the Jacobian, the residual against noise per plane, flags | `analysis/node_inference.py` | noise-free recovery to solver tolerance; the stated uncertainty covers the truth at the nominal rate on noisy phantoms | one row per node |
| **F3** close neighbours (O8) | for nodes of C0's class "close and in focus": either a flag, or the neighbour inside the model with its amplitude, position, depth and diameter free | `analysis/node_inference_joint.py` [corrected 2026-11-13 (later): its own file] | the toy scenes of the section below | decided after C0's shares |
| **F4** recovery on phantoms | phantoms through C3 and F2: isolated and with neighbours; rendered by Block 4, and by the generators that are not F1's model (the ray world, Block 9; the hybrid, Block 21) with the camera chain and noise; $\hat d/d$ and the coverage of the uncertainty by diameter, tilt and darkness | `analysis/recovery.py`; run by `scripts/infer_nodes.py phantoms` (the script is the orchestrator's) [corrected 2026-11-13 (later)] | this step is the check: a fit tested only on its own generator hides its model error | a table and figures; the acceptance numbers of O12 |
| **F5** Colab Cell 17 | F2 (and F3) on the pilot cell's example nodes, then every node, on the cleaned blocks of Cell 16 | `scripts/infer_nodes.py real`; notebook Cell 17 (the orchestrator's, inline) | the dry run on a cached node | `inference_<specimen>.csv`; per-node figures: data and fitted model, plane by plane |
| **E** the judgment (NC3) | a report for the user: residuals across planes, the diameters against Allen's and the first-stage fit's, along-branch consistency, the shares flagged, F4's recovery | `plotting/inference_figures.py`; notebook Cell 18 (the orchestrator's) | -- | the user's decision: option 2 stands, or NC4 starts |

## Choices to settle before the logic

| # | Choice | Options | Proposal | Status |
|---|---|---|---|---|
| O1 | how a shadow is removed | add back the absorbed light; divide by the neighbours' transmittance | add back (Eqs. 5-6); flag overlaps in projection | **[user, D-130]** the shadow is computed and removed wherever it falls, also directly above or below the node; no exclusion, the node keeps a label |
| O2 | the coarse fit, and the starting absorption coefficient | `measure_pilot` on the cleaned block; a two-parameter fit with the calibrated kernel at the trace's plane | a fit whose model is the shadow predictor's own, so that a neighbour's fitted features reproduce its observed shadow; start at the median of the first-stage fits | **[user, D-130]** start from the first-stage fit (`measure_pilot`), then refine to explain the observed shadow |
| O3 | what a round updates | diameter and absorption coefficient; also the lateral offset and depth | all four from round 1: position and depth errors mattered in the toy check | **[user, D-130]** all four |
| O4 | which segments are neighbours | other branches; the node's own branch beyond the fitted stretch; axon; soma | all traced dendrite segments outside the node's fitted stretch, and the soma as today's disc; axon if traced | **[user, D-130]** as proposed, the axon included where traced |
| O5 | patch half-size and planes | +-25 um (calibration); +-5 um (`block_half_um`); planes +-3 or +-6 | +-10 um and +-3 planes for the figures and the line; larger only if C0 shows shadows from farther | **[user, D-130]** the patch as proposed; no fixed plane range: as many planes as a neighbour's shadow can reach |
| O6 | stop rule | fixed rounds; a change threshold | every diameter changes by less than 1 % and every example line by less than the noise SD, or 5 rounds | **[user, D-130]** as proposed |
| O7 | example nodes | how many, how chosen | about 10, from C0's classes: 2 clean (controls), 3 diffuse, 3 close and in focus, 2 near the soma | **[user, D-130]** 20 nodes |
| O8 | close in-focus neighbours | flag and fill; joint fit | decide after C0's shares | **[open]** decided after the survey [user, D-130] |
| O9 | option 2's parameters and data | free: diameter, absorption coefficient, lateral offset, axis depth; tilt and heading from the line fit; planes +-3 | as listed | **[user, D-130]** as proposed, with as many planes as necessary |
| O10 | absorption in F1 | partition (default); hybrid | partition first; the hybrid enters through F4 as a generator | **[user, D-130]** the hybrid |
| O11 | the scene generator of the checks | the tests' product of transmittances; Eqs. 5-6 applied to several tubes at once | the second, built in C2/C3's tests; the first kept as a labelled comparison | **[user, D-130]** as proposed |
| O12 | "satisfactory" (NC3) | -- | the user's; candidates: F4's $\hat d/d$ within a stated band on independent generators; residuals at the noise level on the pilot cell; the share of flagged nodes | **[open]** |

Standing decisions the steps keep: phantoms go through exactly the real-data pipeline, the cleaning included (D-057
(iii)); every node whose fit is usable gets its own estimate (D-030, D-096); the flag `crossing` is evaluated on the
cleaned line **[proposal]**.

## What the toy check says about the risks

Toy **[run, sandbox]**: a 0.5 um node and a 1 um parallel neighbour, both flat, the pilot cell's kernel table, linear
absorption, no noise, one plane, the node read with its exact model. Fitted over true diameter of the node:

| Neighbour | no cleaning | subtracted, rough values | fitted with the node, depth free |
|---|---|---|---|
| 1 um aside, in focus | 0.45 | 0.69 | 1.00 |
| 1 um aside, 1.4 um out of focus | 1.06 | 1.01 | 1.00 |
| 2.5 um aside, in focus | 0.97 | 0.98 | 1.00 |
| directly above, 2 um | 1.18 | 1.09 | 1.00 |

"Rough values": one pixel off laterally, one plane off in depth and a diameter 25 % too thin, together. A joint fit
with the neighbour's depth held at its wrong value gave 0.92 in the first row. Scripts `check_neighbour_cleaning.py`
and `check_deconvolution_focal_plane.py`, sent as files in the chat of 2026-11-13; not in the repository.

| Risk | What measures it |
|---|---|
| a close in-focus neighbour is not removed well by subtraction | C0's share of such nodes; C5's synthetic scenes; F3 |
| Allen's diameters are too thin, so round 0 under-cleans | the change between rounds 0 and 1 in C5's record |
| trace position and depth errors misplace the shadow | O3; the before/after figures of C6 |
| shadows of tubes that overlap in projection do not add | C3's flag; its count in C0 |
| the fit trusts the kernel: any kernel error goes into the diameter | F4 with the generators that are not F1's model; the residuals of F5 |
| a node thinner than the blur: diameter and absorption coefficient trade off | F4 by diameter; the covariance of F2 |
| cost per node | F1's time per call; C7 and F5 report seconds per node |

Not verified here: nothing was run on Allen's data; the shares of polluted nodes on the pilot cell are unknown until C0.
