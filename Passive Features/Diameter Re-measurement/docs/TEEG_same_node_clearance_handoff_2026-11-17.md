# Handoff: same-node clearance -- the node's own shadow from the other planes, estimated per node on sub-planes and removed from each plane before the profile fit (diameter chat -> next chat)

| Date | Change |
|---|---|
| 2026-11-17 | Created at the user's request (D-151) for the chat that builds the same-node clearance. Branch `sci/diameter-pipeline` at `021869d` (code) and the commit that adds this file; decision log on `main` at `2cd9f89` (D-151). Nothing of the clearance is specified, built or computed. |
| 2026-11-17 (later) | The clearance chat (14:14 onward): section 12 added -- first numbers of the split $S = S^{\rm in} + S^{\rm out}$ and of the iteration's gain, from `checks/own_shadow_clearance_check.py` (output beside it) **[run, sandbox of the clearance chat]**; section 11's first item marked corrected. No decision, no SPEC entry, no pipeline code. |
| 2026-11-17 (later, 2) | The clearance chat, 17:05: **D-156** [user] -- the joint fit of all planes with the whole own shadow stays the estimator, the clearance becomes a product computed from its estimate every round, clear-then-fit is dropped without a comparison. Section 13 added; it supersedes section 6's correction "before the profile fit" and closes section 7's points (a) to (e). SPEC: Block 35 amendment 2, Blocks 37, 38, Block 36 amendment 11 (drafted, the contract commit that carries this row). The decision's ID is D-156 (written as D-155 until the push: the noise-fit chat took D-155 meanwhile). 17:20: the user approved the block plan and recalled that the fit comes after the neighbours' cleaning (D-156.6). |
| 2026-11-17 (later, 3) | The clearance chat: the wave of D-156 is built, merged and checked. Block 35 amendment 2 (`line_fast.line_shadow_split`), Block 37 (`analysis/own_clearance.py`), Block 38 (`plotting/own_clearance_figures.py`) by three Opus implementers; Block 36 amendment 11 (the entry points, `--no-own`, the pilot-run test, notebook Cells 17b and 18b) by the orchestrator; final check by an Opus tester, `specs/test_reports/2026-11-17_1754_incremental.md`: PASS WITH ISSUES (one MINOR, fixed; four NOTE; three spec gaps, closed), D-157 and D-158 [orchestrator]. Section 14 added. The theory document of the main route: `docs/TEEG_diameter_pipeline_theory_2026-11-17/`. Not run: the pilot cell in Colab (Allen's images do not reach the sandbox). |
| 2026-11-18 | The clearance chat: D-159 [user] -- the tester's spec tests merged into the branch, the three stale spec tests retired. New at the user's request: `scripts/show_focal.py` and notebook Cell 18c (each example node at its focal plane with the estimated tube's edges and Allen's diameter on the image; SPEC Block 31 amendment 3, Block 36 amendment 12). The pilot's first clearance figures were received (nodes 8089, 7448, 4844, 7081): section 15. |
| 2026-11-18 (later) | The clearance chat: Cell 18c ran on the pilot (`cleaning_worst`, the four `overlap` examples; refinement's estimates, Cell 18b not run there). Section 16 added: the figure's table and its reading. No decision. |
| 2026-11-18 (later, 2) | The clearance chat: D-160 [user] -- three clear nodes first. Built: `contamination.pick_clear_examples`, `clean_cell.py clear`, notebook Cell 19 (outputs in `OUT/cleaning_clear`). Waiting for the user's Colab run of Cells 16b and 19. |
| 2026-11-18 (later, 3) | The clearance chat, 13:41 to 16:22: the work moved to the joint fit of a node with its close neighbours (D-161 to D-166), to a fix of the cleaning (D-163) and to the kernel's in-focus width (D-165). This file is not updated for them: read `TEEG_joint_fit_handoff_2026-11-18.md` beside it. |

Basis tags: **[src]** read in the named file at `021869d`; **[data]** read in the files the user uploaded on
2026-11-17 at 12:40 (the Colab run of Cell 17b: `cleaning_529878215.json`, `examples_worst_529878215.json`, the
line and node figures); **[run, here]** executed in the handing-over chat's sandbox; **[user]** said or decided by
the user; **[log]** the decision log on `main`; **[reasoning]** the assistant's inference, not checked; **[open]**
neither decided nor verified.

Plainly: a recorded plane of Allen's stack sees a dendrite as the sum of the shadows of all its depth slices, each
blurred by the defocus kernel at its distance from that plane. For a node a few micrometres thick, the slices within
half a plane (+-0.14 um) of the focal plane hold only a small part of the absorbing path through the node's axis --
0.28/d of it, 7-14 % for d = 2-4 um **[reasoning, geometry]** -- and the rest lies at other planes, blurred by
0.15-1.7 um (section 5). The user wants this part, the node's own shadow from the other planes, estimated for every
node on fine sub-planes with the full kernel, added back to each plane before the profile fit, and re-estimated as
the diameter changes. The pipeline's forward models already compute every slice (Block 35, slabs of 0.02 um);
nothing yet separates them by plane or removes them.

## 1. First actions in the next chat

1. Read, in this order: this file; in the decision log (`main`, `docs/project/TEEG_decisions_and_ideas_log.md`)
   D-151 (the request), then D-150, D-149, D-148, D-144, D-138, D-137, D-132, D-130, D-129; for the kernel D-024
   (the configured far slope 0.79), D-026, D-080 (the table ends at 0.84 um), D-088.3 (too few far nodes), D-118's
   update of 2026-11-10 (the pilot's far fit) and D-120 (the pooled fits on davinci). In `specs/SPEC.md`: Block 4
   (the kernel, Eqs. 5-6), Block 26, Block 35 and its amendment 1, Block 29, Block 30, Block 32 and its amendments,
   Block 36 and its amendments 1-10, Block 28 (the sign of the cleaning).
2. Get the branch (in an existing clone `git fetch origin && git checkout sci/diameter-pipeline && git pull --ff-only`):
   ```
   git clone --branch sci/diameter-pipeline https://github.com/Leonardodm00/Towards-EEG.git && cd Towards-EEG && git log --oneline -1
   python3 specs/roadmap/roadmap.py status --yaml specs/roadmap/diameter-pipeline.yaml
   ```
   Expect `021869d` or a later commit (this file's).
3. Settle section 7's open points with the user before writing logic (coding directive 3): the band, the fit that
   takes the cleared planes and its model, the iteration, images or line. Then the SPEC entries (a Block 35 amendment
   for the split by depth, a block for the clearance, amendments of Blocks 32 and 36 for the rounds and the entry
   points), then propose the roadmap's entries to the user (nothing is added unasked).
4. Re-read the last changelog rows of the log before taking an ID: the noise-fit chat logs in parallel. Next free
   IDs at hand-off: **D-152, I-006** **[log]**.

## 2. What the user asked (2026-11-17), in force as D-151

The request (12:40), verbatim **[user]**: "Now the handoff for another chat that handles the same node clearance.
Given the defocus table (fitted from that cell data, the one in colab, remember that these are just test runs for
the algorithm in da vinci teh table will be refitted with the full amount of cells and nodes) estimate the
contribution of the same node's shadow coming from the other planes. I would also try to do the interpolation
across planes within the actually recorded ones (with a distance of 0.24 um)."

The answers to four questions that followed **[user]**:

| Question | Answer |
|---|---|
| the 0.24 um | "I was wrong. i intended the distance between planes so 0.28" |
| what the interpolation interpolates | "Use the full table (near and far fields) of defocusing. to not limit to the near filed. depending on the node we might need to interpolate past 1 um in both directions (if the node is considered to have the diameter of 2 um perfectly centered in the choosen focal plane). I need for each node a fine estimation of the shadow generated by the node itself across many subplanes (the distance in um is a chosen parameter for example i decide that we estimate the shadows across planes distant 0.1 um or less. This must be iteratively refined because the diameter across the iterations changes and so the shadows." |
| where the estimate is made | "In the other chat" -- the estimate is the next chat's work, not this file's |
| what the clearance delivers | "A correction": remove the node's own out-of-focus shadow from each plane before the profile fit |

## 2b. The same, as a list of requirements

| ID | Requirement | Basis |
|---|---|---|
| R1 | For every node, the node's own shadow at each of its planes, split by depth into sub-planes | D-151.2 [user] |
| R2 | Sub-plane spacing a parameter (um), e.g. 0.1 um or less | D-151.2 [user] |
| R3 | The full defocus kernel: the near table and its far-field continuation, over the node's whole depth, past 1 um on both sides when the node reaches there | D-151.3 [user] |
| R4 | A correction: the out-of-focus part added back to each plane before the profile fit | D-151.1 [user] |
| R5 | Re-estimated at every iteration from the current diameter | D-151.4 [user] |
| R6 | The pilot's kernel is a test value; davinci refits it on all cells and nodes; the code reads the kernel from the configuration | D-151.6 [user] |

## 3. Where things are

| What | Where | State at hand-off |
|---|---|---|
| Code | `Leonardodm00/Towards-EEG`, branch `sci/diameter-pipeline`, workstream `Passive Features/Diameter Re-measurement/` (`<ws>` below) | `021869d`, pushed **[src]** |
| Spec | `specs/SPEC.md` (repository root) | the latest amendments: Blocks 25 (7), 27 (2), 28 (2), 31 (2), 36 (10) at `97b40a4`, `4ed0793` (D-150) **[src]** |
| Decision log | `main`: `docs/project/TEEG_decisions_and_ideas_log.md` | D-151 at `2cd9f89` **[log]** |
| Plan of the cleaning and inference | `<ws>/docs/TEEG_diameter_cleaning_inference_plan_2026-11-13.md` | changelog row 2026-11-17 names this handoff **[src]** |
| Notebook | `<ws>/notebooks/cleaning_pilot_2026-11-14.py` (Colab, cells as `# %%`) | Cells 16b (rank), 17b (worse conditions, `OUT/cleaning_tier1`), 18b (fit); the user's 17 Nov run used an older Cell 17b (section 8) **[src; data]** |
| Roadmap | `specs/roadmap/diameter-pipeline.yaml` and `.md`; the page "Subproject Roadmaps" | no entry for the clearance **[src]** |
| Pilot cell | specimen 529878215; SWC on Drive `MyDrive/allen_cache/H16.06.010.01.03.05.03_668702901_m.swc`; Colab outputs under `MyDrive/diameters/` (`calibration/`, `cleaning/`, `cleaning_worst/`) | Allen's images reach only Colab: the sandbox's proxy refuses `api.brain-map.org` (403) **[run, earlier in the handing-over chat]** |
| The kernel used in Colab | `MyDrive/diameters/calibration/calibration_529878215.json` (Cell 7), assembled by the notebook's `cleaning_config()` into `MyDrive/diameters/cleaning/cleaning_config.json` | section 5 **[src; data]** |

## 4. The model the clearance acts on

Notation (symbols as in SPEC Blocks 4 and 35) **[src]**:

| Symbol | Meaning | Units, value |
|---|---|---|
| $i$ | a node of the work list | -- |
| $\theta = (d, \mu, v_0, z_{\rm ax})$ | the node's parameters: diameter, absorption coefficient, shift along the measuring axis $\hat y$ about the trace, axis depth | um, 1/um, um, um |
| $T_i(\theta)$ | the node's straight capped tube, `node_inputs.local_tube`: heading and tilt from the first-stage (pilot) fit, half-lengths $U_-, U_+$ along the axis (`U_axis_um`, 4.6-6.6 um on the work list **[data]**) | -- |
| $K_i$ | the node's recorded planes $k^*_i - 4, \dots, k^*_i + 4$ (`cleaning.plane_half_range` 4: 9 planes, +-1.12 um) | -- |
| $z_k = k\,\Delta z$ | depth of plane $k$; $\Delta z$ = `acquisition.dz_um` | um; 0.28 (46 cells checked, D-085.3) |
| $\zeta_j$, $\Delta\zeta$ | centre and thickness of depth slab $j$ (`geometry.slab_centres`); `renderer.dzeta_um` | um; 0.02 |
| $\Delta A_j(x, y \mid \theta)$ | light absorbed in slab $j$ at column $(x, y)$; `hybrid` in both fits (D-130.O10): $a_j\,h(x, y, \bar\zeta_j)$ with Block 21's ray histories | dimensionless |
| $\sigma_{\rm r}(\delta)$ | the defocus kernel's SD at defocus $\delta$, even in $\delta$ (section 5) | um |
| $G_\sigma$ | the normalised 2-D Gaussian of SD $\sigma$ | 1/um$^2$ |
| $B_k$ | plane $k$'s background, refitted every round (D-144) | gl; 89-96 on the work list **[data]** |
| $v_n$ | the measuring line's offsets, $[-3, 3]$ um in steps of 0.1144 um (53 samples; `measure.profile_half_um`, `profile_step_um`) | um |

The node's own shadow and its plane (Block 4 Eqs. 5-6 divided by $B$):
$$S_k(x, y \mid \theta) = \sum_j \big(\Delta A_j(\cdot \mid \theta) * G_{\sigma_{\rm r}(\zeta_j - z_k)}\big)(x, y), \qquad \tau_k(x, y \mid \theta) = 1 - S_k(x, y \mid \theta).$$
Reading: for plane $k$, every slab's absorbed light is blurred by the kernel at that slab's distance from the
plane, and the blurred slabs are summed; the plane's grey level is $I_k \approx B_k\,\tau_k$ plus the neighbours'
shadows and noise. The sum runs over every slab of the tube, in focus or not, so the node's own shadow from the
other planes is the part of $S_k$ from slabs away from $z_k$.

Where the code evaluates it on the measuring line: `line_fast._profiles` (`<ws>/src/allen_diameter/model/line_fast.py`,
lines 352-381 at `021869d`): for each slab $j$ and plane $k$, `deficit[k] += Ky @ (dA @ Kx.T)` with
`s = sig[j, k]` = $\sigma_{\rm r}(\zeta_j - z_k)$, a slab skipped for a plane when it lies farther than
`inference.line_reach_sigmas` (6) $\times \sigma$ from the line's pixels; then pixel means and `sample_profile`
**[src]**. The per-(slab, plane) term the split needs is already there: the band is a condition on
$|\zeta_j - z_k|$ inside that loop.

Where the own shadow enters today **[src]**:

| Place | What it does with the node's own shadow |
|---|---|
| `node_inputs.predict_factory` (the coarse refinement, Block 29, each round) | predict = $1 - $ `line_fast.line_profiles`($T_i(\theta)$): the whole $S_k$, every slab |
| `node_inputs.forward_factory` (the per-node fit, Block 30, Cell 18b) | forward = `line_fast.line_profiles`: the whole $\tau_k$ |
| `node_inputs.all_shadow` (the per-plane background model, D-144) | Block 25's shadow of every traced segment, the node's own stretch included, at the round's features |
| `shadow.shadow_image` (the neighbours' shadow, Block 25) | the node's own stretch (`own_ids_within`, $U$ = 5 um) is left out: Block 28 never touches the node's own tube |

## 5. The defocus kernel

Definition (`kernel.sigma_r`, `kernel_family = "gaussian_table"`) **[src]**: $\sigma_{\rm r}(\delta)$ is the linear
interpolation of the table at $|\delta| \le \delta_{\max}$ (its last knot) and
$\sigma_{\rm r}(\delta_{\max}) + \gamma\,(|\delta| - \delta_{\max})$ beyond (`kernel_continuation = "linear"`,
$\gamma$ = `kernel_continuation_slope`).

The pilot's kernel, as the 17 Nov run used it **[data: the run's configuration]**:

| Part | Value | Basis |
|---|---|---|
| table knots $\delta$ (um) | 0, 0.28, 0.56, 0.84 | Cell 7 on 529878215 |
| table $\sigma_{\rm r}$ (um) | 0.0800, 0.1536, 0.3924, 0.5620 | Cell 7 on 529878215; anchor 0.562 +- 0.049 um [log: D-118] |
| far slope $\gamma$ (um per um) | 0.79, the configured default | D-024 (Debye, $\cos^4$ weighting), not fitted: the notebook found no production far fit [src: `cleaning_config()`] |
| far-fit diagnostics on the pilot | 0.404 (Cell 7, 2026-11-02); 0.3971 +- 0.0302 (Cell 15, 2026-11-10, 2 far nodes, INSUFFICIENT under D-088.3) | [log] |

Beyond the table **[run, here: `kernel.sigma_r` with the run's table]**:

| $\lvert\delta\rvert$ (um) | 0.84 | 1.00 | 1.25 | 1.50 | 2.00 | 2.25 |
|---|---|---|---|---|---|---|
| $\sigma_{\rm r}$, $\gamma$ = 0.79 (um) | 0.562 | 0.688 | 0.886 | 1.083 | 1.478 | 1.676 |
| $\sigma_{\rm r}$, $\gamma$ = 0.397 (um) | 0.562 | 0.626 | 0.725 | 0.824 | 1.023 | 1.122 |

On davinci the pooled near and far fits over all the cells (Block 19, D-120) replace both the table and $\gamma$
**[log]**; the clearance reads `RendererConfig.kernel_table_delta_um`, `kernel_table_sigma_um`,
`kernel_continuation`, `kernel_continuation_slope` and hard-codes none of them (R6).

## 6. The task, in the notation above (D-151; the notation is the handing-over chat's, PROVISIONAL)

- Sub-planes: depths $\zeta_m = z_{\rm ax} + m\,h$, $m \in \mathbb Z$, spacing $h$ (um, the parameter of R2), over
  the tube's whole depth extent (a flat tube: $z_{\rm ax} \pm d/2$; a tilted one more).
- The in-focus band of plane $k$: the sub-planes with $|\zeta_m - z_k| \le \beta$, $\beta$ open (section 7 (a)).
- The node's own shadow from the other planes, at plane $k$:
  $$S^{\rm out}_k(x, y \mid \theta, \beta) = \sum_{m:\ |\zeta_m - z_k| > \beta} \big(\Delta A_m(\cdot \mid \theta) * G_{\sigma_{\rm r}(\zeta_m - z_k)}\big)(x, y), \qquad S^{\rm in}_k = S_k - S^{\rm out}_k.$$
  Reading: the same sum as $S_k$, restricted to the sub-planes outside plane $k$'s band; $S^{\rm in}_k$ is what is
  left, the shadow of the node's material at plane $k$ itself.
- The correction at iteration $t$, with the sign of Block 28's `clean_block` (light added back):
  $$I^{{\rm clr},(t)}_k(x, y) = I_k(x, y) + B_k\,S^{\rm out}_k\big(x, y \mid \hat\theta^{(t-1)}, \beta\big) \quad \text{(gl)}.$$
  Reading: plane $k$ gets back the light the node's out-of-band material took, as predicted from the previous
  estimate of the node; the profile fit of iteration $t$ then reads $I^{{\rm clr},(t)}$ and returns
  $\hat\theta^{(t)}$ (R4, R5).

## 7. Open points to settle with the user before writing logic

(a) **The band $\beta$.** $\pm\Delta z/2$ = +-0.14 um (the material nearer plane $k$ than any other recorded plane),
$\pm h/2$ (only the sub-plane at $z_k$), or another. It fixes what "the node's own shadow from the other planes"
means. **[open]**

(b) **The sub-plane spacing $h$.** The renderer already slices at $\Delta\zeta$ = 0.02 um (`renderer.dzeta_um`;
procedure s.3.6: at most 0.05 um) and blurs each slice at its own $\sigma_{\rm r}$, so $h$ can be $\Delta\zeta$
itself or a coarser grouping for speed; either way a check that $S^{\rm out}$ has converged in $h$ (the user's
"0.1 um or less"). **[open]**

(c) **Which fit takes the cleared planes, with which model.** Candidates: the first-stage per-plane fit (Blocks 5,
13: the pilot), the coarse refinement (Block 29, each round), the per-node fit (Block 30, Cell 18b). The refinement's
and the fit's forward models already contain $S^{\rm out}$ (Block 35 sums every slab), so fitting the cleared
planes with them counts it twice; with a cleared plane the model is $1 - S^{\rm in}_k(\theta, \beta)$
**[reasoning]**. A fixed point of "add back $S^{\rm out}$ at $\hat\theta^{(t-1)}$, fit $1 - S^{\rm in}$" is in
general not the full model's least-squares estimate: the fit holds $S^{\rm out}$ fixed and loses its derivative in
$\theta$ (a backfitting scheme) **[reasoning, Likely]**. Whether that is wanted, or the full fit stays the estimator
and the clearance serves other ends (the per-plane widths, the figures), is the user's call. **[open]**

(d) **The iteration.** Inside Block 32's Jacobi rounds (round $t$ uses round $t-1$'s features for the neighbours and
now for the node itself) or a loop nested per node; damping; the stop rule (Block 32 stops on `max_rel_d`); the bound
on rounds. Round 1 of the 17 Nov run changed $d$ by up to 24.6 % (`history.max_rel_d` 0.2457) **[data]**. **[open]**

(e) **Images or line.** On the patch's planes (as Block 28's `clean_block`, then the line is read), or on the
measuring line's profiles: `line_fast` gives the line at 0.44-3.63 s per cold call for $d$ = 0.3-2.0 um at half-length
5 um (Block 35's status, flat tube, 8 x 16 directions) and reuses the cross-section when only the centre moves
**[src]**. **[open]**

(f) **The far field.** $\gamma$ = 0.79 (configured) against 0.40 (diagnostic) moves $\sigma_{\rm r}$ by 0.06 um at
1 um and by 0.46 um at 2 um (section 5); the pilot's tests run on a far field that is not calibrated. **[data;
run, here]**

(g) **Validity of the line model.** `line_fast` holds only where no cap of the tube is within the blur's reach
(D-138.2: `line_reach_sigmas` $\times \max \sigma_{\rm r}$, plus a ray margin); the wide kernels of a thick node's
far slices make `ExtrusionInvalid` (flag and skip) likelier with $U_\pm$ = 4.6-6.6 um **[src; reasoning]**.

(h) **Light direction.** The absorbed light per slab is not even in depth (`transmitted_before` under
`partition_vertical`; the hybrid's ray histories likewise), so $S^{\rm out}$ from above and from below plane $k$
differ under an even kernel **[src; reasoning]**.

## 8. The cleaning's state at hand-off

The 17 Nov Colab run **[data]**: D-150's code, but the old Cell 17b (the survey's four examples 8089, 4844, 7081,
7448 with class `overlap`, folder `cleaning_worst`; Cell 16b's tier-1 ranking not run); round 1 done, round 2
running at download (`stopped: running`). After round 1 the cleaning added back under 0.1 gl on 8089's line, 1-2 gl
on 7081's and a 6-12 gl ramp on 7448's (the soma), the dips unchanged in shape. The user: "These are the results it
doens't seem to be much improved. Let's keep it however" -- D-150 kept (D-151's status note).

The ten work-list nodes after round 1 **[data: `cleaning_529878215.json`]**:

| node | example | Allen $d$ (um) | first-stage $\hat d$ (um) | $d$ (um) | $\mu$ (1/um) | $v_0$ (um) | $\Delta z$ of the axis (um) | planes |
|---|---|---|---|---|---|---|---|---|
| 4844 | yes | 0.467 | 3.14 | 2.652 | 0.0502 | -0.068 | +0.618 | 264-272 |
| 4924 | no | 0.469 | 5.65 | 4.454 | 0.0264 | +0.685 | -0.777 | 247-255 |
| 4925 | no | 0.468 | 3.12 | 2.444 | 0.0496 | -0.192 | -0.697 | 250-258 |
| 5719 | no | 0.500 | 4.17 | 3.604 | 0.0353 | -0.156 | +0.835 | 182-190 |
| 5720 | no | 0.499 | 3.54 | 2.669 | 0.0470 | +0.177 | +0.978 | 184-192 |
| 5721 | no | 0.498 | 3.78 | 3.225 | 0.0315 | +0.579 | +1.828 | 192-200 |
| 5722 | no | 0.496 | 3.36 | 2.980 | 0.0371 | +0.446 | +1.164 | 195-203 |
| 7081 | yes | 0.446 | 3.97 | 3.808 | 0.0432 | -0.894 | +0.736 | 194-202 |
| 7448 | yes | 0.478 | 1.50 | 1.288 | 0.1296 | -0.191 | +0.513 | 60-68 |
| 8089 | yes | 0.553 | 2.69 | 2.441 | 0.0751 | +0.140 | -0.472 | 104-112 |

Reading: the fits put these nodes at 1.3-4.5 um with $\mu$ = 0.026-0.13 /um, the first stage already at 1.5-5.7 um,
against Allen's 0.45-0.55 um; such a tube spans +-0.65 to +-2.2 um in depth, mostly beyond the table's 0.84 um, and
the refinement moved the axis by up to 1.83 um in depth from the trace (5721; the planes span +-1.12 um about the
first stage's sharpest plane, which lies within 0.13 um of plane $k^*$ on these nodes). Whether these diameters are
right is open. If they are
too large, $S^{\rm out}$ is too large with them, and the iteration of (d) needs a check against running away
**[reasoning]**.

## 9. Code to start from

All under `<ws>/` at `021869d` **[src]**.

| Module, function | What it gives | For the clearance |
|---|---|---|
| `src/allen_diameter/model/line_fast.py`: `line_profiles`, `_profiles`, `SectionCache`, `extrusion_valid`, `ExtrusionInvalid` | the node's $\tau_k(v_n)$ on its line, slabs of `dzeta_um`, the full kernel, hybrid; one cross-section carried along the tube (Block 35) | the slab loop (lines 352-381) is where $S^{\rm out}$ / $S^{\rm in}$ split by $\lvert\zeta_j - z_k\rvert$ |
| `src/allen_diameter/model/kernel.py`: `sigma_r(delta, rcfg)` | $\sigma_{\rm r}$ from the configuration | the full kernel (R3, R6) |
| `src/allen_diameter/model/render.py`: `render_transmittance`, `absorbed_fractions`, `slab_grid` | whole planes (Block 4) | the reference for any new line code, as Blocks 26 and 35 are tested |
| `src/allen_diameter/model/line_render.py`: `line_profiles` | Block 26, the slow reference of Block 35 | oracle |
| `src/allen_diameter/model/hybrid.py`: `slab_absorbed`, `ray_history` | the hybrid's absorbed light (Block 21) | $\Delta A_j$ |
| `src/allen_diameter/analysis/node_inputs.py`: `local_tube`, `predict_factory`, `forward_factory`, `build_node`, `own_ids_within`, `all_shadow`, `model_background` | the node's tube, line, planes, backgrounds | where a cleared predict/forward would be built |
| `src/allen_diameter/analysis/clean_iteration.py`: `round_once`, `run_rounds` | one Jacobi round per call: backgrounds, tiers, the neighbours' shadow, cleaning, profiles, refinement | where the clearance enters each round (d) |
| `src/allen_diameter/analysis/cleaning.py`: `clean_block` | $I + B\,S$ on a patch | the sign convention |
| `src/allen_diameter/analysis/coarse_fit.py`: `refine_node`; `src/allen_diameter/analysis/node_inference.py`: `fit_node` | the round's refinement; the per-node fit | the fits of (c) |
| `scripts/clean_cell.py` (`survey`, `rank`, `clean`), `scripts/infer_nodes.py` (`real`) | the entry points of the notebook's cells | the run's knobs and logs |
| `tests/smoke/test_smoke_line_fast.py`, `test_smoke_clean_iteration.py`, `test_smoke_pilot_run.py` | the tests to extend; the pilot run test drives `clean_cell.py` end to end on a synthetic cell | regression |

## 10. Decisions and rules in force

| ID or rule | What it imposes here |
|---|---|
| D-129, D-130 (O1-O12), D-132 | the cleaning's structure: tier-1 neighbours subtracted, tier 2 for a joint fit not built; the node's own stretch never in the neighbours' shadow |
| D-130.O10 | hybrid absorption in the per-node fit |
| D-137, D-138 | `line_fast`: one cross-section, 2e-4 against Block 26, flag and skip on `ExtrusionInvalid` |
| D-144 | the backgrounds refitted every round, from a model with every segment's shadow |
| D-148 | the coarse refinement's settings (`refine_line_fine_factor` 4, tolerances) |
| D-150 | the figures' scale, the tier-1 ranking, the neighbours' shadow kept across rounds -- kept by the user |
| D-151 | this work (sections 2, 2b) |
| Coding directives | SPEC entries before code; smoke test per algorithm with an oracle; verify twice; Opus implementers for parallel blocks, the orchestrator settles issues and logs `[orchestrator]` decisions; an Opus tester for the final check |
| Cluster | code that runs on davinci: ASCII source, LF endings (`hpc-python-compat`), shipped through git (`hpc-git-delivery`) |
| Documents | the log on `main` (read before acting, append at once); handoffs beside the code; every reply that writes a document ends with one line naming the documents and the commit |

## 11. Not checked

- Nothing of the clearance was computed: no $S^{\rm out}$, no share of the dip, no timing **[open]**. [corrected 2026-11-17 (later): first numbers on a flat tube in section 12; nothing on real profiles, no timing of a cleared fit]
- The run's log (the round times and the `shadow images ... reused, ... rendered` lines of D-150.3) was not
  uploaded, so D-150.3's speed-up is not seen on the pilot **[open]**.
- Cell 16b's tier-1 ranking has not run on the pilot **[open]**.
- The 7-14 % of the plain paragraph is the in-band share of the absorbing path through a flat tube's axis
  ($2\times0.14/d$), not of the blurred dip **[reasoning, geometry]**.

## 12. First numbers (the clearance chat, 2026-11-17; a diagnostic, not the clearance)

Script `<ws>/checks/own_shadow_clearance_check.py`, output `own_shadow_clearance_check.out` beside it **[run,
sandbox of the clearance chat]**. It splits the sum of `line_fast._profiles` by one condition on
$|\zeta_j - z_k|$ (the function's source rewritten in memory, the module untouched); $S = S^{\rm in} + S^{\rm out}$
holds to 2e-16.

Setting, all of it the assistant's choice for a first look, none decided: a flat capped tube (tilt 0, half-length
10 um, the cap test bypassed), hybrid at 8 x 16 directions, `line_fine_factor` 4, the pilot's table (section 5),
$\beta$ = 0.14 um, slabs of 0.02 um, the line $v_n$ = -2.97 to 2.97 um (53 samples, `n_avg` 0), planes $k$ = -4..4
at $z_k = k \times 0.28$ um, $B_k$ = 92 gl, $(d, \mu, z_{\rm ax})$ of three work-list nodes after round 1 (section 8).

The split with $\gamma$ = 0.79, over the planes that cut the tube ($|z_k - z_{\rm ax}| < d/2$):

| node | $d$ (um) | $\mu$ (1/um) | $\max_n S^{\rm in}_k(v_n)\,/\,S_k$ at the dip's peak | $B_k \max_n S^{\rm in}_k(v_n)$ (gl) | FWHM of $S_k(v_n)$ (um) | FWHM of $S^{\rm in}_k(v_n)$ (um) | chord of the tube at $z_k$ (um) |
|---|---|---|---|---|---|---|---|
| 4924 | 4.454 | 0.0264 | 0.073-0.078 | 0.88-0.94 | 3.66-4.12 | 2.29-4.35 | 2.33-4.45 |
| 8089 | 2.441 | 0.0751 | 0.130-0.142 | 2.35-2.68 | 2.06-2.29 | 1.14-2.29 | 1.30-2.43 |
| 7448 | 1.288 | 0.1296 | 0.153-0.275 | 2.37-4.55 | 0.92-1.14 | 0.46-1.14 | 0.43-1.28 |

Reading: with the band at half a plane spacing the clearance removes 72 to 93 % of the node's dip at its peak,
all of it predicted by the model at the previous estimate; what is left, $B_k S^{\rm in}_k$, is 0.9 to 4.5 gl
against a profile noise SD of 0.85 to 1.31 gl on the pilot (D-149.3). $S^{\rm in}_k(v_n)$ is an optical section:
its width follows the tube's chord at the depth of plane $k$ (widths read on the 0.1144 um samples) and it is zero
on the planes that miss the tube, whereas the width of the whole $S_k(v_n)$ changes little from plane to plane.
The far slope does not enter $S^{\rm in}$ (the band lies inside the table); $\gamma$ = 0.397 in place of 0.79
raises $\max_n S^{\rm out}_k(v_n)$ by 0 to 14 % on the planes that cut the tube (8089 at $k$ = +2: 0.1608 to
0.1755, that is 1.35 gl at $B_k$ = 92 gl, beside $B_k \max_n S^{\rm in}_k$ = 2.35 gl) and by up to 28 % on those
that miss it (7448 at $k$ = -4).

The iteration of section 6, linearised about a fixed point $\theta^*$ (noise-free, unweighted sum over the 9
planes and 53 samples, $\gamma$ = 0.79): with $J^{\rm in}$, $J^{\rm out}$ the derivatives of
$S^{\rm in}_k(v_n \mid \theta, \beta)$ and $S^{\rm out}_k(v_n \mid \theta, \beta)$ in $\theta = (d, \mu, v_0, z_{\rm ax})$
at $\theta^*$ (rows: planes times samples; central differences), "add back $S^{\rm out}$ at $\hat\theta^{(t-1)}$,
fit $1 - S^{\rm in}$ by least squares" gives
$$\hat\theta^{(t)} - \theta^* = M\,(\hat\theta^{(t-1)} - \theta^*), \qquad M = -\big(J^{{\rm in}\top} J^{\rm in}\big)^{-1} J^{{\rm in}\top} J^{\rm out},$$
and with a damping $\lambda$ (the new estimate moved only the share $\lambda$ of the way) the matrix is
$(1 - \lambda)\,I + \lambda M$. Reading: an error in the previous estimate comes back multiplied by $M$; the
iteration converges only if every eigenvalue of that matrix is below 1 in modulus.

| node | eigenvalues of $M$ | spectral radius, $\lambda$ = 1 | $\lambda$ = 0.5 | 0.3 | 0.25 | 0.2 | 0.1 |
|---|---|---|---|---|---|---|---|
| 8089 | -5.77, -2.33, -1.59, 0.10 | 5.77 | 2.38 | 1.03 | 0.78 | 0.82 | 0.91 |
| 7448 | -3.03, -1.59, -0.23, 0.42 | 3.03 | 1.02 | 0.83 | 0.86 | 0.88 | 0.94 |

Reading: undamped, the iteration diverges with alternating sign on both nodes (the out-of-band part is several
times the in-band part, so an overestimated $\mu$ or $d$ removes too much and the next fit swings the other way,
further); damped at 0.2 to 0.25 it converges by a factor of about 0.8 per round, some ten rounds per decimal
digit. At the fixed point the residual $I_k - B_k (1 - S_k(\cdot \mid \theta^*))$ is orthogonal to the columns of
$J^{\rm in}$, not to those of $J = J^{\rm in} + J^{\rm out}$ (the full model's least squares): under white noise
its standard errors are those of the full fit times 1.22, 1.17, 1.26, 1.79 (8089: $d$, $\mu$, $v_0$, $z_{\rm ax}$)
and 1.48, 1.38, 1.21, 1.82 (7448). The columns of $J^{\rm in}$ are less collinear than those of $J$ (condition
number at unit column norm 1.8 against 3.9 and 4.9).

Exact, for any $\theta$ **[reasoning, algebra]**: the cleared profile at $\hat\theta$ minus its own model is the
full model's residual,
$I_k + B_k S^{\rm out}_k(\cdot \mid \hat\theta, \beta) - B_k\big(1 - S^{\rm in}_k(\cdot \mid \hat\theta, \beta)\big) = I_k - B_k\big(1 - S_k(\cdot \mid \hat\theta)\big)$:
the cleared planes show the model's in-band section plus the misfit the full model already has.

Not covered: tilted tubes, the caps (section 7 (g)), the refinement's weights $w_n$, real profiles, a second band,
the noise's correlation along the line, any timing.

## 13. Decided on 2026-11-17, 17:05 (D-156) -- what is built

Section 6's correction acting before the fit, and its iteration, are superseded [corrected 2026-11-17 (later, 2)]:

| Point of section 7 | Outcome | Basis |
|---|---|---|
| (c) which fit, which model | no fit reads a cleared profile; Blocks 29 and 30 stay the estimator (joint least squares over the node's planes with the whole own shadow); clear-then-fit is not built, not even as a comparison | D-156.1, D-156.3 [user] |
| what the clearance is | a product computed from the estimate after the fit, every round of the cleaning and after each per-node fit: the split $S^{\rm in}_k$, $S^{\rm out}_k$, the cleared profile $I_k + B_k S^{\rm out}_k$, the chord per plane, figures | D-156.2 [user] |
| (a) the band | $\beta$ = 0.14 um, `CleaningConfig.own_band_half_um` | D-156.4 (the assistant's default, not objected to) |
| (b) the sub-planes | the split is computed on the renderer's slabs (0.02 um), a slab's share in a band being the fraction of its thickness inside; depth layers of `CleaningConfig.own_layer_um` (0.1 um) are kept for the example nodes only | D-156.4, D-156.5 [orchestrator] |
| (d) the iteration | none of its own: the product is recomputed from each round's estimate | D-156.2 |
| (e) images or line | the measuring line; cleared patch images not built | D-156.4 |
| (f) the far field | the kernel is read from the configuration; pilot runs at 0.79, with 0.397 as a labelled sensitivity run | D-156.4 |
| (g) the caps | `ExtrusionInvalid`: the node's product is flagged `extrusion_invalid`, nothing stops | SPEC Block 36, amendment 11 |

Blocks (SPEC, drafted 2026-11-17): Block 35 amendment 2 (`line_fast.line_shadow_split`, `OwnSplit`), Block 37
(`analysis/own_clearance.py`), Block 38 (`plotting/own_clearance_figures.py`), Block 36 amendment 11 (the
configuration, `node_inputs.own_split_factory`, the two entry points, the pilot-run test). Still open: whether the
pilot's fitted diameters are right -- the product is the check of the model on those nodes.

## 14. Built on 2026-11-17 -- how to run it, what it writes, what is next

Run (Colab, the pilot): the notebook's Cell 17b (`clean_cell.py clean`) and Cell 18b (`infer_nodes.py real`) as
before; the product is on by default, `--no-own` skips it; the features and the fits are the same either way
[run, sandbox: `test_own_product`, bit-identical on the synthetic cell].

| Output | Where | Content |
|---|---|---|
| `own` per work-list node | `cleaning_<S>.json`, `nodes[<id>]` | `status` (`ok`, `extrusion_invalid`, `not_refined`, `error`), `band_half_um`, and per plane `chord_um`, `in_peak_gl`, `out_peak_gl`, `in_share`, `residual_rms_gl` |
| arrays, example nodes | `cleaning_<S>_node<id>.npz` | `own_s_in`, `own_s_out`, `own_cleared`, `own_model_in`, `own_chord_um`, `own_layers_gl`, `own_layer_z` |
| figures, example nodes | `own_node<id>.png`, `own_node<id>_round<t>.png`, `own_layers_node<id>.png` | per plane the profile, the cleared profile, the in-band and whole models, the chord; the 0.1 um layers' shadows at plane $k^*$ |
| after each per-node fit | `inference_<S>.json` (`own`, status `not_fitted` when the fit failed), `own_<S>_node<id>.npz`, the same figures | at Block 30's estimate |

Reading a figure: if the model holds for the node, the cleared profile follows the in-band model, as wide as the
chord, and is flat on the planes that miss the fitted tube; the difference between the two is the joint fit's
residual (the handoff's section 12, the identity). A plane within $\beta$ of the tube's top or bottom has chord 0 and
a small in-band share (D-158.4).

Next, for the user's decision: run Cells 17b and 18b on the pilot and read the ten work-list nodes' figures against
the open question of section 8 (are the fitted 1.3 to 4.5 um right?); a sensitivity run with the far slope 0.397
(`--config-json`); whether the three stale spec tests (the report's F2) are retired or updated; whether the tester's
39 new spec tests (branch `tester/2026-11-17_1754`) come into the branch.

## 15. The pilot's first clearance figures (2026-11-18)

**[data: 16 PNGs of the user's Colab run, uploaded 2026-11-18 at 12:48 -- `own_node<id>.png`,
`own_node<id>_round1.png`, `own_node<id>_round2.png`, `own_layers_node<id>.png` for the nodes 8089, 7448, 4844, 7081
(class `overlap`, round 2); read by eye in the clearance chat, no array was sent]**

| node | fitted $d$ (um), $\mu$ (1/um), round 2 | what the figure shows |
|---|---|---|
| 8089 | 2.44, 0.0750 | at plane 108 the profile has a narrow core, about 0.6 um wide and down to about 69 gl, inside a broad dip; the whole model is smooth and misses the core by about 5 gl; the cleared profile shows a narrow dip of about 7 gl, not a flat section 2.4 um wide |
| 7448 | 1.288, 0.1296 | the dip is fitted well at planes 64 and 66; the cleared profile is irregular (side lobes of 3 to 5 gl on both sides: the soma's surroundings) |
| 4844 | 2.623, 0.0503 | broad asymmetric dips with shoulders at +1.5 to +2.5 um that the model does not have; the cleared profile is not a flat section |
| 7081 | 3.801, 0.0432 | at planes 196 and 198 the profile has two minima about 1.5 um apart with a rise between them; one tube of 3.8 um spans both |

Reading **[reasoning]**: on none of the four nodes does the cleared profile follow the in-band model, which by the
identity of section 12 is the joint fit's misfit shown plane by plane. On 8089 and 7081 the shape of the misfit (a
narrow core; two minima) is what a thinner structure, or two structures, inside the fitted width would give; whether
that is so needs the image itself -- hence Cell 18c -- and was not decided. These four are the survey's `overlap`
examples, the worst class by construction.

## 16. Cell 18c on the pilot (2026-11-18, 13:24): the fitted tube against the image

**[data: the user's Colab output of `scripts/show_focal.py --cleaning-dir .../cleaning_worst --specimen 529878215
--plane kstar --image before --zoom-um 6.0` (table) and its figure `focal_529878215.png`, pasted in the chat; the
figure read by eye: one plane, grey range the 1st to 99th percentile of the view]**

| node | plane $k^*$ | $z_{k^*} - \hat z_{\rm ax}$ (um) | Allen $d$ (um) | first stage (um) | refined $\hat d$ (um) | what the image shows |
|---|---|---|---|---|---|---|
| 8089 | 108 | +0.245 | 0.553 | 2.695 | 2.440 | one dark band roughly 1 um wide along the fitted direction, lying in the half of the fitted tube towards one edge; the other edge runs through light background about 1.5 um from the band; a grey surround around the band |
| 4844 | 268 | +0.926 | 0.467 | 3.141 | 2.623 | several dark blobs of different size, not one uniform band; the fitted axis is 0.93 um out of this plane |
| 7081 | 198 | +0.002 | 0.446 | 3.975 | 3.801 | a dark band about 2 um wide with a second dark structure joining it; the fitted axis lies about 0.9 um to one side of the trace, and one fitted edge runs through light background |
| 7448 | 64 | -0.504 | 0.478 | 1.498 | 1.288 | the darkest structure is elongated across the fitted direction, along the measuring line; the soma's dark mass in a corner |

Reading **[reasoning]**: on all four the fitted tube is wider than the dark band it is drawn on, and on none is the
object one straight uniform tube (a core with a grey surround; several structures; a heading that does not follow the
dark structure). With the profiles of section 15 (a narrow core inside a broad shallow dip) this says that the
one-tube fit follows the broad component. What the broad component is, is open: candidates are the dendrite's spines
(**[knowledge base: Benavides-Piccione et al. 2025, J Comp Neurol, Table 1, full text in project knowledge]** basal
dendrites of human temporal cortex: spine length 1.449 +- 0.012 um, mean +- SEM, 1.290 +- 0.046 spines per um), tier-2
neighbours left in the profile (D-132), or a real blur with wider tails than the Gaussian kernel. The fitted
coefficients (0.03 to 0.13 /um) are low, as a sparse absorber's would be. The four nodes are the survey's `overlap`
examples; nothing here says how isolated nodes behave. PubMed, "dendritic diameter light microscopy overestimation
biocytin reconstruction electron microscopy comparison": 0 results.
