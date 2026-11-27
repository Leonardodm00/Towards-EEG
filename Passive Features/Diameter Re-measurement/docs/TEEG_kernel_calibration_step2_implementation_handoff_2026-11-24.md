# Handoff: kernel calibration, step 2 -- the twins' least squares (roadmap NS1; procedure s.3.4, item 2) (defocus-table chat -> implementation chat)

| Date | Change |
|---|---|
| 2026-11-24 | Created at the user's request (16:36) for the chat that implements step 2 of the defocus kernel's calibration. Written on `sci/diameter-pipeline` at `7b4a9ab` (this file's commit follows it); decision log on `main` at `44b6e6a`. Nothing of step 2 is built (section 3). The defocus-table chat wrote no step-2 code; its study of step 2 is the T10 document (section 4.1), which is not in the repository, so sections 4 and 7 restate what the implementation needs from it. |
| 2026-11-24 (later) | Pushed at `786977b`, rebased onto `9c7e8bb`: the commits in between (Block 44 amendment 4, D-239) do not touch step 2's code, but they moved SPEC s.8, so its line numbers are dropped here. The log on `main` at `c7822ad`: D-174 and D-178 marked implemented (section 5); D-240 (the soma fit holds the far slope, uncalibrated) added to section 5 as a consumer of step 2's $\gamma$; next free IDs D-241, I-007. |
| 2026-11-27 | **D-261** [user] settles Q1: the user chooses the calibration nodes from the davinci export (the image at focus shows Allen's diameter) on its agreement with the image and on clean backgrounds; no diameter is set by eye (D-226.4 withdrawn); the chosen nodes go to steps 1 and 2 (D-230's order unchanged), and step 2's twins carry Allen's diameter as the true one: form B with the diameters held. Q4 gains a toy result of 2026-11-27: form B with every twin's depth fitted, started from the configured table and nothing from step 1, recovered the table and the depths on the faint toy and on the hybrid toy. Blocks 50 and 51 now exist in SPEC (wave 8). Log on `main` at `69fc18e`; next free IDs D-262, I-008. Branch at `e019d12` before this row. |
| 2026-11-27 (later) | **D-263** [user]: the near stage -- step 1 and step 2 on the near-set twins -- uses only the planes $\lvert k - k^*\rvert \le 3$, up to the third plane on each side of focus, the anchor's plane; farther planes only in the far stage. Already met by `offsets_planes` = 3 and D-229.2, now binding (sections 1, 4.3, 4.5 and 5). Log on `main` at `957bcdc`; next free IDs D-264, I-008. |
| 2026-11-27 (later, 2) | **D-264** [user]: D-230.3's alternation dropped; each stage runs once, near then far (sections 1, 4.5, 5, 7.5, 9 Q8 and 10). Open instead: the slope the near stage holds. Log on `main` at `330c9ad`; next free IDs D-265, I-008. |
| 2026-11-27 (later, 3) | **D-265** [user]: in the near stage the twins' slices past the anchor take the spline's extrapolation (the assistant's reading: D-181's not-a-knot spline in G, continued along its tangent), not the linear continuation with a held slope; so the near stage holds no slope and the renderer needs D-181's spline for it (sections 1, 4.5, 5, 8 and 9 Q13). Log on `main` at `ac66aec`; next free IDs D-266, I-008. |
| 2026-11-27 (later, 4) | D-265's scope confirmed by the user: near-field calibration only, where a third plane lies slightly past 0.84 um; the far stage and the tables keep $\gamma$. The tangent reading stays PROVISIONAL. Log on `main` at `b2752a7`. |
| 2026-11-27 (later, 5) | D-265's tangent confirmed by the user (10:45): past 0.84 um the near stage's spline is continued along its tangent in $G$, not by its last cubic piece. Log on `main` at `b5788ef`. |
| 2026-11-27 (18:06, the step-2 implementation chat) | The user answered section 9: **D-270** (the full stack as the forward model; noisy twins, each plane as its real plane; the hybrid in steps 1 and 2), **D-271** (each near twin's depth fitted, its absorption searched, in the near stage only; held in the slope's fit), **D-272** (the spline as the renderer's default between knots), **D-273** (weights, far planes from 4 in both fits, solver, a training/validation split, the nodes' delivery). New section 9a lists what is settled and what is still open. Check of this chat: `checks/step2_identifiability_2026-11-27/`. The log entries are committed (`ca2695e`) but not yet on `main`: the push to `main` was refused in this session and waits for the user. Branch at `aadac6a` before this row. |
| 2026-11-27 (evening, the step-2 implementation chat) | **D-274** [user] closes section 9a's open points: the absorption held at the pilot's value (no area term; Q3), a far twin's depth fitted on its own near planes with the knots held, $\sigma_{\rm r}(0)$ held with a fit compared on the validation set (Q2), the split by node 75/25, davinci with at most 100 cores (NS1 `needs_hpc: true`), the plan approved. The log is on `main` (`7ade7ff`, D-270 to D-276). Built: Block 4 amendment 4 (the spline reading as `kernel_family` \"gaussian_table_spline\", the continuation \"tangent\"; D-275), `far_min_k_offset` = 4, `Step2Config`, the SPEC contract of Blocks 52 to 56 (section 5, \"Kernel calibration, step 2\") and amendment 1 (D-276; `5962ca3`). Wave 1 (Blocks 52 to 56) is with its implementers. Section 9a's rows Q2 and Q3 and its list of open points are superseded by this row; section 10 by SPEC. |

**Status [2026-11-24]:** not started. D-228 to D-230 fix the estimator (form B, on core widths), its trial table and
its order. The twins' settings -- diameter, absorption, axis depth -- and their forward model are open (section 9) and
come before any logic. [updated 2026-11-27: the diameter is settled by D-261, Allen's taken as true; absorption, depth
and the forward model stay open.]

Basis tags: **[user]** said or decided by the user; **[log]** the decision log on `main`
(`docs/project/TEEG_decisions_and_ideas_log.md`, read at `44b6e6a`); **[src]** read on 2026-11-24 in the named file at
`7b4a9ab`; **[doc]** the T10 study document (section 4.1), with its rating where it matters (Certain / Likely /
Guessing); **[run, sandbox]** executed in the defocus-table chat's sandbox, script named; **[reasoning]** the
assistant's inference, not run; **[open]** neither decided nor verified; **[not read]** exists, not opened by this chat.

Paths are relative to `Passive Features/Diameter Re-measurement/`, except those starting with `specs/` or
`docs/project/` (repository root).

## 1. First actions in the next chat

1. Read this file; then, in the log on `main`, by ID: D-165, D-172, D-175, D-178, D-181, D-201, D-221, D-226 to D-231,
   D-261, D-263, D-264, D-265, I-006. Read the changelog's last rows again before taking an ID (several chats log there); at `c7822ad` the newest
   row gives the next free IDs as D-241 and I-007.
2. Ask the user:
   - whether D-231's code has reached `origin`: `git log origin/sci/diameter-pipeline --oneline --grep "D-231"`. At
     writing it had not (section 8, item 1);
   - for the T10 study document's zip, if its toys are wanted as test oracles (section 4.1).
3. Load the skills `scientific-coding` (SPEC entries before logic; a fan-out for the blocks ready together),
   `subproject-roadmap` (each new SPEC block enters the roadmap under NS1 in the turn its entry is written:
   `roadmap.py add block I<n> --block <n> --title ... --summary ... --parent NS1 --needs-theory T10`) and
   `project-decision-log`. For anything that runs on davinci, also `hpc-python-compat` and `hpc-git-delivery`. NS1 is
   `needs_hpc: false` in the roadmap; changing that is proposed to the user first.
4. Settle section 9 with the user before writing logic. Section 10's blocks are a proposal.

## 2. What the user asked

"Briefly did we start to implement this? if not write an handoff to implement the stage 2 in another chat"
**[user, 2026-11-24 16:36]**. "This" is step 2 of the defocus kernel's calibration: procedure s.3.4, item 2; roadmap
step NS1; theory topic T10. The T10 study document develops it, and D-228 to D-230 settle it in part.

## 3. Where step 2 stands (checked 2026-11-24)

| Where | What it says |
|---|---|
| Roadmap: `roadmap.py status --no-fetch` on `specs/roadmap/diameter-pipeline.yaml` at `7b4a9ab` | NS1 "Kernel calibration, step 2" is planned, with "no blocks defined yet"; `depends_on: [S3c]`, `needs_theory: [T10]`, `needs_hpc: false`. T10 is open. **[run, sandbox]** |
| `specs/SPEC.md` s.8, the item "Kernel calibration, step 2" | "Kernel calibration, step 2 (...): not built." **[src]** |
| Log | D-228 and D-229: "active; not built". D-230: "active; not built; D-230.3 to confirm". I-006: "idea; not scheduled". D-232 to D-239 concern other work. **[log]** |
| Code | Nothing named for step 2 under `src/allen_diameter/` or `scripts/`; the only "twin" hits are matplotlib's `twinx`. **[src]** |

## 4. What step 2 is

### 4.1 Sources

- **Procedure** `docs/TEEG_diameter_bias_table_procedure_2026-10-04.md` s.3.4, item 2. It tunes $\sigma_{\rm r}(\delta)$
  at each tabulated offset so that rendered thin-node phantoms reproduce the calibration nodes' real profiles plane by
  plane, with:
  - the calibration nodes' fitted diameter;
  - the full camera chain;
  - the same profile extraction;
  - least squares "on the profiles, or on their core widths". **[src]**
- **The T10 study document** `TEEG_diameter_kernel_calibration_step2_2026-11-18.md`, revised 2026-11-23 with the hybrid
  absorption as default (D-231). It was sent as `.md` + `.zip` in the defocus-table chat and is not in the repository.
  The zip holds `figures/` and `scripts/`, with checks 1 to 20 and two toys:
  - `toy_model.py`, the faint limit;
  - `hybrid_toy.py`, the document's Eq. (4).
  **[doc]**. The roadmap note on T10 at `origin` still names the 2026-11-18 version (section 11).

Equation numbers below are this file's; "doc Eq." refers to the study document. Where the document leaves a twin's
settings implicit in its index ($\omega^{\rm sim}_{i,k}(\theta)$), they are written out here
($\omega^{\rm sim}_{i,k}(\theta;\xi_i)$).

### 4.2 Step 1, built (SPEC Block 10)

`calibration.fit_growth` solves, over the $N$ near-set calibration nodes $i$ (doc Eq. 3; SPEC Block 10, Eq. 4):

$$\min_{\{c_i\},\ \{z_{{\rm ax},i}\},\ \{G_m\}}\ \sum_{i=1}^{N}\ \sum_{k \in K^{(1)}_i}\Big[c_i + G\big(z_{{\rm ax},i} - z_k\big) - \omega_{i,k}\Big]^2 . \tag{1}$$

Reading it **[src: `analysis/calibration.py`, `config.py`]**:

- $k^*_i$ is node $i$'s sharpest plane. $K^{(1)}_i$ is the set of planes $k \in \{k^*_i - 3, \dots, k^*_i + 3\}$ in
  which node $i$ has a finite core width; `offsets_planes` = 3.
- $z_k$ (um) is plane $k$'s depth; the planes are $\Delta z$ = 0.28 um apart.
- $\omega_{i,k}$ (um$^2$) is the squared width of a Gaussian fitted within about one FWHM of the core of node $i$'s dip
  in plane $k$ (`dip_statistics`, statistic `gaussian_core_width2`).
- $c_i$ (um$^2$) is a constant per node.
- $z_{{\rm ax},i}$ (um) is node $i$'s fitted axis depth, kept within `z_ax_bound_um` = 0.28 um of its sub-plane depth
  (`GrowthFit.z_ax_um`).
- $G(\delta) = \sigma_{\rm r}(\delta)^2 - \sigma_{\rm r}(0)^2$ (um$^2$) is the growth of the kernel's variance at
  defocus $\delta$, where $\delta = z_{{\rm ax},i} - z_k$ is object minus plane. It is represented by its knot values
  $G_m = G(\delta_m)$, $\delta_m = 0.28\,m$ um, read between knots by a cubic spline (`growth_interp` = "cubic").

`kernel_from_growth` turns $\hat G$ into the table $\sigma_{\rm r}(\delta_m) = [\sigma_{\rm r}(0)^2 + \hat G_m]^{1/2}$:

- cut at 0.84 um (`table_max_delta_um`);
- with $\sigma_{\rm r}(0)$ configured (`renderer.sigma_r0_um` = 0.080 um, "never identified by the stacks").

The far fit (Block 15, `far_field.fit_far_slope`) then fits the slope beyond 0.84 um. It holds the anchor
$\sigma_{\rm r}(0.84\ {\rm um})$, with a constant and an axis shift per far node.

Step 1 is biased on thick tubes, and pooling does not remove the bias (section 7.1).

### 4.3 The twins (doc Eq. 12)

**Twin $i$.** For calibration node $i$, the twin is the renderer's stack of a phantom tube with node $i$'s settings
$\xi_i$. Those settings are:

- diameter $\tilde d_i$ (um);
- absorption coefficient $\mu_i$ (1/um);
- axis depth $z_{{\rm ax},i}$;
- lateral centre, heading and tilt;
- background (gl);
- light direction;
- the camera chain.

**The trial table** $\theta = (\theta_1, \theta_2, \theta_3, \gamma) \in \Theta$ (D-228.1):

- $\theta_m$ (um) is the kernel's width at $\delta_m$ = 0.28, 0.56, 0.84 um, for $m$ = 1, 2, 3;
- $\gamma$ (um of width per um of defocus) is the slope of the linear continuation beyond 0.84 um;
- $\sigma_{\rm r}(0)$ is held at 0.080 um (D-221.1).

The twin is measured as its node was:

$$\omega^{\rm sim}_{i,k}(\theta;\,\xi_i) = S_k\big(R(\theta, \xi_i)\big), \qquad k \in K_i,\ \ i = 1, \dots, N. \tag{2}$$

Reading it:

- $R(\theta, \xi_i)$ is the stack, in grey levels, rendered under the kernel table (0.080, $\theta_1$, $\theta_2$,
  $\theta_3$) um at $\delta$ = (0, 0.28, 0.56, 0.84) um, continued with slope $\gamma$.
- $S_k$ is the measurement in plane $k$: the plane scan along the node's measuring line, then the core-width statistic.
  It is the same code with the same settings that measured the real node (section 8, item 2).
- $K_i$ is node $i$'s planes in step 2 (D-229.2):
  - for a near-set node, $k^*_i \pm 3$ (D-229.2; binding for steps 1 and 2 of the near stage, D-263);
  - for a far-set node, $4 \le |k - k^*_i| \le 16$. Its planes within $\pm 3$ stay out of the fit and serve as a
    validation set.

### 4.4 Form B, the decided estimator (doc Eq. 13; D-229.1)

$$\hat\theta^{(\rm B)} = \arg\min_{\theta \in \Theta}\ \sum_{i=1}^{N}\ \sum_{k \in K_i} \varpi\big(|k - k^*_i|\big)\,\Big[\omega_{i,k} - \omega^{\rm sim}_{i,k}(\theta;\,\xi_i)\Big]^2 . \tag{3}$$

Reading it:

- $\varpi \ge 0$ weights a plane by its distance from focus; its form and direction are open (D-229).
- $\Theta$ requires $\theta_1, \theta_2, \theta_3 > \sigma_{\rm r}(0)$ and $\gamma \ge 0$. The document's checks bounded the
  knots in [0.081, 2.0], [0.082, 2.0] and [0.083, 2.0] um.
- A plane with no finite core width is dropped.
- No per-node constant and no axis depth is fitted. Absolute core widths are compared, because each twin carries its
  own node's diameter and depth.

### 4.5 The order, decided (D-230)

1. **Near stage.** Step 1, then Eq. (3) on the near-set twins, gives $\theta_1, \theta_2, \theta_3$ with $\gamma$ held.
   Both steps use only the planes $|k - k^*_i| \le 3$, up to the anchor's plane on each side of focus (D-263). The
   twins' slices past 0.84 um take the spline through $\sigma_{\rm r}(0)$, $\theta_1$, $\theta_2$, $\theta_3$, extrapolated along
   its tangent in $G$, so the near stage holds no $\gamma$ (D-265).
2. **Far stage.** The far fit, then Eq. (3) on the far-set twins, gives $\gamma$ with the knots held.
3. **D-230.3** (the assistant's condition, PROVISIONAL until the user confirms): alternate the two stages until neither
   moves by more than a set fraction of its SE. This is block-coordinate descent toward the joint least squares of
   Eq. (3). The fraction and the maximum number of passes are open. [dropped 2026-11-27, D-264: no alternation; each
   stage runs once, near then far, and the table is the near knots with the far slope.]

### 4.6 Variants, not chosen (doc s.3.8)

**Form A** (doc Eq. 15) is a fixed-point iteration:

$$\theta^{(t+1)} = \theta^{(t)} + \big[T(\mathcal D^{\rm obs}) - T(\mathcal D^{\rm sim}(\theta^{(t)}))\big], \qquad \theta^{(0)} = T(\mathcal D^{\rm obs}),$$

- $T$ is step 1 (Eq. (1), then the table);
- $\mathcal D^{\rm obs}$ is the real nodes' scans, and $\mathcal D^{\rm sim}(\theta)$ the twins' scans under $\theta$.

Form A is blind to any error that shifts all of one twin's core widths alike, because $c_i$ absorbs it.

**Form C** (doc Eq. 16) minimises Eq. (3) over $\theta$ and every twin's diameter jointly:

- $\tilde d = (\tilde d_1, \dots, \tilde d_N) \in (0, \infty)^N$;
- the twin's core width is $\omega^{\rm sim}_{i,k}(\theta;\,\xi_i[\tilde d_i])$, where $\xi_i[\tilde d_i]$ is $\xi_i$
  with the diameter replaced by $\tilde d_i$.

**Form P** (doc Eq. 17; I-006):

$$\big(\hat\theta^{(\rm P)}, \hat d\big) = \arg\min_{\theta \in \Theta,\ \tilde d \in (0,\infty)^N}\ \sum_{i=1}^{N}\sum_{k \in K_i} \varpi\big(|k - k^*_i|\big)\,\frac{\big[\omega_{i,k} - \omega^{\rm sim}_{i,k}(\theta;\,\xi_i[\tilde d_i])\big]^2}{s_\omega^2} + \sum_{i=1}^{N} \frac{\big(\tilde d_i - d^{\rm ref}_i\big)^2}{\varrho_i^2} . \tag{4}$$

Reading it:

- $s_\omega$ (um$^2$) is the core widths' noise SD;
- $d^{\rm ref}_i$ (um) is twin $i$'s reference diameter;
- $\varrho_i$ (um) is the leash: as every $\varrho_i \to 0$ the problem becomes form B at $d^{\rm ref}$, and as every
  $\varrho_i \to \infty$ it becomes form C;
- in code, the penalty is $N$ extra residuals $(\tilde d_i - d^{\rm ref}_i)/\varrho_i$ appended to the weighted misfits
  **[reasoning]**.

### 4.7 Why form B can return the true table (doc s.3.7)

Three hypotheses:

- **H1.** A table $\theta_0$ exists at which the renderer images a tube as the microscope does, as far as the core
  widths go.
- **H2.** Each $\xi_i$ holds node $i$'s true settings.
- **H3.** The Jacobian $J_{(i,k),m} = \partial \omega^{\rm sim}_{i,k}(\theta;\xi_i)/\partial\theta_m$ has full column
  rank at $\theta_0$.

Under H1 to H3, without noise, $\theta_0$ is an isolated zero of Eq. (3) **[doc, Certain: argued there]**. If H1
fails, Eq. (3) returns the table under which the renderer reproduces the real core widths; the bias table needs that
table too, since it is computed by the same renderer **[doc, Likely]**. H2 is the hypothesis that fails in practice
(section 7).

## 5. Decisions that bind step 2 [log]

| ID | What it fixes for step 2 | State |
|---|---|---|
| D-165 | $\sigma_{\rm r}(0)$ is a parameter the runs can set; re-anchoring at a new width keeps the measured $G$ | active |
| D-172 | $\sigma_{\rm r}(0)$ = 0.080 um for now | active |
| D-221.1 | 0.080 um is the default, with a switch to Block 42's calibrated width (D-201). At `7b4a9ab` the switch is `--kernel-width fixed\|calibrated --sigma-json FILE` (`scripts/kernel_width.py`, `loading/sigma_io.py`, used by `clean_cell.py` and `fit_soma.py`) **[src]** | active |
| D-175 | The davinci table run. Each near node's fitted axis depth is in `calibration_pooled.json`. At most 100 one-core tasks at once, 48 h walltime. Its cell set is superseded by D-226 | active |
| D-178 | The first stage (the pilot fit) holds $d$ at Allen's diameter (twice the traced radius) and fits $\mu$ and the lateral centre. Twins built from the pilot record are therefore rendered at Allen's $d$ unless section 9 decides otherwise. In code since `513ad9e` (`measure.pilot_diameter` = "allen") **[src]** | active |
| D-181 | The renderer should read the near-field table between knots with step 1's cubic spline. Not built: `model/kernel.py::sigma_r` reads it by `np.interp`, linear in $\lvert\delta\rvert$ **[src]** | active; not built |
| D-201 | Block 42 fits $\sigma_0$ and a growth scale $c$ on thick nodes with the hybrid line renderer (`scripts/calibrate_sigma.py`) | active |
| D-226 | The user chooses the calibration nodes from per-node images of the passive + I_h fit's cells (in the assistant's reading the 18 L3_exc cells, PROVISIONAL) and sets by-eye diameters $d^{\rm eye}_i$ in Colab. What $d^{\rm eye}_i$ is used for is open [updated 2026-11-27: the by-eye diameter is withdrawn by D-261] | active; superseded in part by D-261 |
| D-227 | Beside each plane's image, the profile along the measuring line | active; not built |
| D-228 | $\theta = (\theta_1, \theta_2, \theta_3, \gamma)$, with $\sigma_{\rm r}(0)$ held; the far-set twins fix $\gamma$ | active; not built |
| D-229 | Form B on core widths for near and far twins; planes weighted by distance from focus; the far nodes' $\pm 3$ planes kept for validation; $\gamma$ always fitted | active; not built |
| D-230 | Near stage, then far stage; D-230.3's alternation PROVISIONAL [dropped by D-264] | active; not built; superseded in part by D-264 |
| D-264 | D-230.3 dropped: the stages are not alternated; each runs once, near then far. Open: the slope held in the near stage (0.79 configured, or a first far fit on step 1's anchor), now never revisited | active; not built |
| D-265 | In the near stage the twins' slices past the anchor (0.84 um) take the spline's extrapolation -- D-181's not-a-knot spline in $G$, continued along its tangent (the tangent confirmed by the user, 2026-11-27 10:45) -- not the linear continuation with a held $\gamma$; the far stage and the table keep $\gamma$ (scope confirmed by the user, 2026-11-27 10:41). On the pilot table the tangent's slope of $\sigma_{\rm r}$ is 0.417 against $\gamma$ = 0.79; the last cubic piece would make $\sigma_{\rm r}$ fall past 1.1 um | active; not built (needs D-181) |
| D-231 | The hybrid is the renderer's default absorption. Implemented at `2812d4d` in the defocus-table chat's sandbox; not on `origin` (section 8, item 1) | active |
| I-006 | Form P | idea; not used in step 2 under D-261 |
| D-261 | The user chooses the calibration nodes from the davinci export, whose image at focus shows Allen's diameter, on its agreement with the image and on clean backgrounds across planes; no diameter by eye; the chosen nodes go to steps 1 and 2; **step 2's twins carry Allen's diameter as the true one**, $\tilde d_i = D_{{\rm A},i}$: form B with the diameters held, forms C and P not used | active; not built |
| D-263 | The near stage -- step 1 and step 2 on the near-set twins -- uses only the planes $\lvert k - k^*_i\rvert \le 3$, up to the third plane on each side of focus, the anchor's plane (0.84 um); farther planes only in the far stage. Met by `offsets_planes` = 3 and D-229.2; `offsets_planes` is now a decision, not a tunable default | active; step 2 not built |
| D-240 | Not binding on step 2: the soma fit (Block 46) holds the far slope at the configured 0.79, which the entry calls uncalibrated. A calibrated $\gamma$ from step 2 is a candidate for it **[reasoning]** | active |

The log's status lines of D-174 and D-178 said "not implemented" although the changelog row of 2026-11-18 (later, 25)
records both implemented at `513ad9e`. They were marked "[updated 2026-11-24]" together with this file.

## 6. What exists to build on [src at `7b4a9ab`]

### 6.1 Measurement

| Piece | Where | Use in step 2 |
|---|---|---|
| `plane_scan(patch, node, branch, scene, cfg, keep_images=False)`, `dip_statistics(v, I, B_bar, mcfg, cal)` | `analysis/calibration.py` | $S_k$ of a near-set twin |
| `scan_node(branch, i, provider, cfg, ...)` | same | The pilot fit, then the scan. A twin needs the scan only; it takes its node's pilot values |
| `fit_growth(scans, cfg, cells=None)` -> `GrowthFit` (`c_um2`, `z_ax_um`, `g_um2`, `knots_um`, ...); `kernel_from_growth(growth, rcfg, sigma_r0_um=None, max_delta_um=None)`; `anchor_from_growth`, `bootstrap_sigma0` | same | Step 1, the near stage's start and its depths |
| `PILOT_RECORD_FIELDS` = (`k_star`, `z_sub_um`, `cx_um`, `cy_um`, `theta_rad`, `phi_rad`, `B_bar`, `mu_hat_per_um`, `d_hat_um`, `alpha_hat`, `fit_status`, `flags`, `reg_verdict`); `node_records(out, swc, cfg)` | same | Most of $\xi_i$, per node |
| `far_plane_scan(patch, node, branch, scene, cfg, ...)`, `far_plane_statistics`, `core_fit_iterative` | `analysis/far_field.py` | $S_k$ of a far-set twin (the far scan's iterative core fit) |
| `select_far_planes(scans, cfg)`, `fit_far_slope(sel, sigma0_um, cfg, ...)` -> `FarFit` (`gamma`, `c_um2`, `s_um` the per-node axis shift, ...) | same | The far stage's start |

### 6.2 Rendering

| Piece | Where | Note |
|---|---|---|
| `synthetic_block(tube, mu, ks, left, top, width, height, cfg, rng, pad_px, ...)` | `model/render.py` | The full stack. Dispatches on `cfg.renderer.absorption` (`ABSORPTIONS_RENDERED` = partition_vertical, linear, hybrid); the hybrid path takes Block 9's directions (`ray_world.directions`) |
| Block 21, the hybrid | `model/hybrid.py`; SPEC Block 21 | $\Delta A_j = a_j\,h(x, y, \bar\zeta_j)$ with $h(p) = \langle e^{-\mu\,\ell_{\rm back}(p;\hat s)}/s_z\rangle_W$ (see below) |
| Block 35, the fast line renderer | `model/line_fast.py` | The pixel-mean transmittance along the measuring line, $\tau^{\rm px}_k(v_n)$, for a straight capped tube, the hybrid included. The per-node fit uses it (Block 30's `fit_node`, run by `scripts/infer_nodes.py`, Block 36). A candidate forward model for twins (section 9, Q5) |
| `sigma_r(delta, rcfg)`, `with_in_focus_width(rcfg, s0)`, `with_kernel(rcfg, sigma_r0_um, growth_scale=1.0)` | `model/kernel.py` | The law; a trial table enters through the `RendererConfig` fields of section 6.4 |
| `calibration_phantom(...)`, `_one_synthetic(args)`, `run_synthetic(...)` | `scripts/calibrate_kernel.py` | The pattern "phantom -> render -> scan" for one tube, parallel over tubes with `multiprocessing.Pool` |
| `Draw`, `phantom_tube(draw, cfg)`, `phantom_branch(draw, cfg, rng)` | `analysis/phantoms.py` | Builds a twin's tube and branch from settings |

The symbols of the hybrid formula:

- $a_j$ is slab $j$'s absorbance in a column;
- $\bar\zeta_j$ is the midpoint depth of the column's part inside slab $j$;
- $\ell_{\rm back}(p;\hat s)$ is the path from point $p$ back to the tube's surface along direction $\hat s$, and $s_z$
  is $\hat s$'s $z$ component;
- $\langle\cdot\rangle_W$ averages over Block 9's condenser directions (16 x 32, NA 1.4, $n_{\rm oil}$ 1.515).

In the faint limit the hybrid's dip is $\langle 1/s_z\rangle_W$ = 1.4470 times the partition's at equal $\mu$
**[doc; log D-231]**.

### 6.3 Inputs on disk

- **Per-cell scans file** (`calibrate_kernel.py real`, and S3's calibration step). Its `meta` carries:
  - `config`, the full configuration the scans were made with;
  - `calibration`;
  - `nodes` (`node_records`: `node_id`, `allen_diameter_um`, `set`, `reasons`, `scanned`, `pilot`).

  The scans themselves are `PlaneScan`s (`ks`, `z_um`, `omega_um2`, ...), read by `calibration_io.load_scans`.
- **Calibration JSON** (`fit`). It writes the growth with `c_um2` and `z_ax_um` per node
  (`loading/calibration_io.py`, line 155).
- **The pooled run** (`scripts/survey_cells.py`'s pool step) writes `s3c/scans_pooled.json`,
  `calibration_pooled.json` and `far_fit_pooled.json`.
- **The Colab notebook's `cleaning_config`** (Cell 16) reads `calibration_<S>.json` (key `kernel`) and
  `far_fit_<S>.json` (`kernel.kernel_continuation_slope`). These are the formats a production kernel from step 2 should
  keep (D-170 handoff, s.3).

### 6.4 Configuration at `7b4a9ab`

- `RendererConfig.absorption` = `"partition_vertical"` (`config.py`, line 202). The hybrid default is D-231's
  unpushed commit.
- The renderer's default table is the ideal-Debye one, "NOT calibrated":
  - `kernel_table_delta_um` = (0.0, 0.14, 0.28, 0.42, 0.56, 0.84);
  - `kernel_table_sigma_um` = (0.080, 0.086, 0.122, 0.262, 0.438, 0.603).
- `kernel_continuation_slope` = 0.79 and `sigma_r0_um` = 0.080. `validate()` requires
  `kernel_table_sigma_um[0] == sigma_r0_um`.
- `CalibrationConfig`:
  - `offsets_planes` = 3;
  - `knot_step_um` = 0.28;
  - `z_ax_bound_um` = 0.28 (PROVISIONAL);
  - `growth_interp` = "cubic" (PROVISIONAL in the config's comment, the mark lifted by D-181.1);
  - `table_max_delta_um` = 0.84;
  - `far_offsets_planes` = 16;
  - `far_min_k_offset` = 5 (PROVISIONAL).

## 7. What the T10 study document found [doc; run, sandbox]

### 7.1 Step 1 is biased on rendered phantoms (check 20)

The setup:

- script `thick_near_check.py`, the copy with `--absorption`; `origin` has it without that option, at
  `checks/defocus_pool_2026-11-18/`;
- the pipeline's own renderer, pilot fit, scan and camera chain;
- 16 flat phantoms per range, $\alpha$ 0.15-0.4, tilt < 2 deg, the background fitted on a +-8 um patch;
- the truth is the pilot cell's table: 0.080, 0.1536, 0.3924, 0.5620 um at 0, 0.28, 0.56, 0.84 um, slope 0.79;
- run at `ed6cc63`.

The relative error of step 1's $\sigma_{\rm r}$:

| True $d$ (um) | Partition: 0.28 / 0.56 um; anchor 0.84 um | Hybrid: 0.28 / 0.56 um; anchor 0.84 um |
|---|---|---|
| 0.15-0.25 | -15.7 / -4.4 %; -10.6 % | -10.3 / -5.4 %; -4.9 % |
| 0.45-0.55 | -30.5 / -14.1 %; -4.7 % | -34.3 / -15.0 %; -4.1 % |
| 0.75-0.95 | +12.0 / -24.1 %; -7.8 % | +13.4 / -23.3 %; -7.9 % |

The cause, per doc s.3.3 **[doc, Likely]**:

- A tube's slices lie at different defocus, so a plane's dip blends blurs.
- The dip's variance is the slices' own spread plus the mean squared blur.
- The averaging excess, about $\tfrac12 G''\,{\rm Var}[\eta]$ (with $\eta$ a slice's depth offset from the axis), is
  constant per node, and so absorbed by $c_i$, only if $G$ is quadratic. The table's $G$ is not.
- Under the hybrid the blend also tilts toward the light, so $\hat z_{{\rm ax},i}$ sits near the blend's centroid, not
  at the axis.
- Pooling more nodes of the same kind leaves a bias untouched.

### 7.2 The faint toy (check 2; `check02_step2_toy.py`)

The setup **[run, sandbox]**:

- noise-free, 41 slices;
- the pipeline's core-width statistic and step 1;
- the same truth as check 20;
- 30 flat tubes per range, axis offsets $U(-0.14, 0.14)$ um, planes $k^* \pm 3$, seed 20261118;
- twins at the true axis depths, with diameters $f d_i$;
- $\sigma_{\rm r}(0)$ held at 0.080 um, $\gamma$ held at 0.79, $\varpi \equiv 1$;
- "recovered" means every knot within 0.002 um.

The relative errors at 0.28, 0.56 and 0.84 um:

| | Mid, $d \sim U(0.45, 0.55)$ um | Thick, $d \sim U(0.75, 0.95)$ um |
|---|---|---|
| Step 1 | -16.1, -13.4, -2.1 % | -10.6, -28.4, -9.9 % |
| A, twins at $d$ | 0.0, 0.0, 0.0 %; recovered in 9 updates | +39.6, -8.3, +5.5 %; not converged in 10 |
| B, twins at $d$ | 0.0, 0.0, 0.0 %; cond $J$ = 4 | 0.0, 0.0, 0.0 %; cond $J$ = 7 |
| C, diameters free, $\sigma_{\rm r}(0)$ right | 0.0, 0.0, 0.0 %; every $\hat d_i = d_i$ from both starts | the same |
| B, twins at 1.2 $d$ | -21.1, +3.2, -1.0 % | -47.3 (at the bound), -14.2, +9.2 % |
| B, twins at 0.55 $d$ | +25.7, -3.0, +0.8 % | +127.9, -11.5, +3.6 % |
| C, true $\sigma_{\rm r}(0)$ = 0.120 um, held 0.080 | -1.7, +0.4, -0.2 %; $\hat d_i/d_i - 1$ median +9.0 % | -2.9, +0.6, -0.4 %; +2.4 % |

### 7.3 The hybrid toy (check 19; `check19_formB_hybrid.py`)

The setup **[run, sandbox]**:

- doc Eq. (4) by the midpoint rule, 160 slices x 481 lateral points; this quadrature was checked against 240 x 961
  within 1.3e-4 relative on $\omega$;
- light +1, Block 9's 16 x 32 directions;
- no noise and no camera chain;
- 30 flat tubes, $d \sim U(0.45, 0.55)$ um, $\mu d \sim U(0.5, 1.5)$ per node, seed 20261123;
- form B with $\varpi \equiv 1$ and $\gamma$ held, started from step 1's table, twins at the true diameters.

| Twins | Relative errors at 0.28, 0.56, 0.84 um | cond $J$ | RMS misfit (um$^2$) | Evaluations |
|---|---|---|---|---|
| Step 1 (no twins) | -19.9, -14.0, -2.4 % | | | |
| True $\mu$, true depths | 0.0, 0.0, 0.0 %; recovered | 4.2 | 6.9e-11 | 66 |
| Rendered as faint, true depths | +5.8, -0.2, +0.7 % | 2.9 | 1.7e-2 | 52 |
| $\mu$ x 0.7 | +2.2, -0.4, +0.5 % | 3.5 | 4.8e-3 | 66 |
| $\mu$ x 1.3 | -2.6, +0.6, -0.6 % | 4.0 | 4.4e-3 | 43 |
| $\mu$ x 1.447 | -4.1, +0.9, -0.9 % | 3.8 | 6.5e-3 | 36 |
| True $\mu$, at step 1's depths (0.032 um toward the light on average) | -5.4, +4.5, -3.1 % | 3.9 | 1.6e-2 | 90 |

### 7.4 The slope (check 10)

Near-set twins respond to $\gamma$ only through the slices of their outer planes beyond 0.84 um **[run, sandbox]**:

- a relative change of $\gamma$ moves their core widths 0.108 (mid) and 0.138 (thick) times as much as the same
  relative change of $\theta_3$;
- adding $\gamma$ raises the condition number of the unit-free Jacobian from 10.4 to 30.1 (mid) and from 6.2 to 47.4
  (thick).

Hence the far-set twins (D-228.2). The alternation of D-230 is not tested: the toys have no far twins.

### 7.5 The cost (checks 5 and 5b; SPEC Block 21, finding (iii))

One near-set phantom (render, pilot fit and 7-plane scan, one sandbox core) at `ed6cc63` takes **[run, sandbox]**:

- 22.5 s under the partition;
- 89.3 s under the hybrid, 4.0 times as long.

The hybrid's render cost grows with the cross-section. With 8 x 16 directions the planes stay within 1.2e-3 of
32 x 64 at a quarter of the configured 16 x 32's cost **[log D-231; SPEC Block 21]**.

One evaluation of Eq. (3) renders every twin once, and a finite-difference Jacobian adds one render of every twin per
component of $\theta$. Form B on the hybrid toy took 36 to 90 evaluations, the Jacobian's included
**[run, sandbox]**.

Order of magnitude for near-set twins of about 0.5 um: about 1.75 $N$ core-hours **[doc, Guessing]**. Not included:

- davinci's speed per core (not measured);
- the far twins, with up to 26 planes each;
- the alternation [none since D-264].

### 7.6 What follows for the implementation [doc, Likely]

- **(a) Axis depth.** Twins placed at step 1's $\hat z_{{\rm ax},i}$ cost several percent at every knot under the hybrid
  (7.3). Either step 2 fits each twin's axis depth ($N$ more unknowns, each touching only its node's planes: a sparse
  Jacobian), or step 1's depth is corrected by the twin's own blend centroid. Neither was tested.
- **(b) Absorption.** The pilot record's $\hat\mu$ (`mu_hat_per_um`) comes from a partition-model fit (Block 3). A
  hybrid twin needs a $\mu$ read by a fit whose forward model is the hybrid. Up to the factor 1.447 apart in the faint
  limit is D-231's reasoning, not run **[log; Guessing]**. A wrong $\mu$ costs a few percent at the knot nearest focus
  (7.3).
- **(c) Diameter.** The diameter is the setting that decides step 2 most. A 20 % error costs form B about as much as
  skipping step 2 (7.2). Form C hands a wrong $\sigma_{\rm r}(0)$ to the diameters (7.2, last row), and its misfit is
  probably too small to reveal it **[doc, Guessing]**. Step 2 makes the table right given the diameters, or the
  diameters right given $\sigma_{\rm r}(0)$; from core widths alone it cannot do both. [2026-11-27: D-261 takes Allen's
  diameters as true and guards them by the user's choice of nodes whose Allen's diameter agrees with the image at
  focus; the residual risk is how fine that visual agreement is, not measured.]
- **(d) Slope.** $\gamma$ needs the far twins (7.4).

## 8. Pitfalls for the implementation [src at `7b4a9ab`; reasoning]

1. **D-231 is not on `origin`.** At `7b4a9ab` the absorption defaults to the partition, so a step-2 configuration that
   does not name the absorption renders the partition.
   - Name `"hybrid"` explicitly until D-231 is pushed.
   - D-231's code exists only as `2812d4d` (with `ed6cc63`, docs) on `6a39dcc`, in the defocus-table chat's sandbox.
     It needs a rebase onto `7b4a9ab` and a rerun of the smoke files.
   - If it never reaches `origin`, D-231.2 in the log lists what it changed: the default, `config.ABSORPTIONS`' order,
     and six smoke files set to the partition explicitly. It can be redone from that list.
2. **The scan reads the configured kernel**, so the twins need two configurations.
   - `calibration_scene` sizes the far-pixel margins and the shadow masks with `sigma_r(delta, cfg.renderer)` (lines
     104-136, 335-415, 553-575); `calibration.py` uses it for the patch's reach and the soma's blur (lines 265-269,
     540).
   - `far_plane_scan` takes its prior width $\sigma_p = \sigma_{\rm r}(|z_{\rm sub} - z_k|)$ from "the renderer's
     current law"; that width sets the iterative core fit's window and its bounds.
   - `_one_synthetic` renders and scans with one `cfg`. A twin rendered under $\theta$ and scanned with a configuration
     that carries $\theta$ would have an $S_k$ that changes with $\theta$, and "the same statistic on both sides"
     (doc s.3.6) would fail.
   - Keep two configurations: the render's, which carries $\theta$, and the scan's, which is the real scans' own
     (`meta.config` of the scans file) **[src; reasoning]**.
3. **The knot grid.** The trial table goes to the renderer as:
   - `kernel_table_delta_um` = (0, 0.28, 0.56, 0.84);
   - `kernel_table_sigma_um` = (0.080, $\theta_1$, $\theta_2$, $\theta_3$);
   - `kernel_continuation_slope` = $\gamma$, `kernel_continuation` = "linear".

   This is `kernel_from_growth`'s grid, not the configured default's six knots.
4. **Between knots the renderer is linear in $|\delta|$; step 1's $G$ is cubic** (D-181, not built). Step 2 is
   self-consistent, because $\theta$ enters the same renderer that later makes the bias table. If D-181 is built after
   step 2, the knots change meaning and step 2 reruns. [2026-11-27: under D-265 the near stage's twins need the spline
   and its tangent extrapolation past 0.84 um, so D-181 is built before step 2's near stage.]
5. **Dark stain.** The hybrid's planes go below zero in dark stain (SPEC Block 21, finding (i): for $d$ = 0.5 um the
   node plane's minimum transmittance is 0.036 at $\mu d$ = 1.5 and -0.057 at 2). The camera chain clips at 0 gl. A twin
   made with the line renderer, which has no camera chain, does not clip.
6. **The far fit has its own axis shift.** `fit_far_slope` fits a per-node axis shift $s_j$ and constant $c_j$ with
   the anchor held, so the axis-depth question (7.6 a) comes back in the far stage.
7. **"Twin" is overloaded.** The twins chat (D-182 to D-238, branch `sci/twins`, roadmap allen-mtg-fit) builds
   simulated cells for the passive + I_h fit. Its cap of 200 cores (D-238) is that submitter's, not this calibration's
   (D-175.3: 100).

## 9. To settle with the user before any logic

| Q | Question | Options and evidence |
|---|---|---|
| Q1 | The twins' diameter $\tilde d_i$ | **Settled by D-261 (2026-11-27): Allen's diameter, taken as the true one; form B with the diameters held.** The options were: Allen's ($D_{\rm A}$, the pilot record's since D-178); the by-eye $d^{\rm eye}_i$ (D-226, not built); the per-node fit's $\hat d$ (Block 30, under the configured kernel); free (form C); on a leash (form P, I-006: which $d^{\rm ref}_i$, which $\varrho_i$). Evidence in 7.2 and 7.6 (c) |
| Q2 | $\sigma_{\rm r}(0)$ in step 2 | Held at 0.080 um (D-221.1's default) or Block 42's calibrated width (the switch). This decides whether forms C and P hand $\sigma_{\rm r}(0)$'s error to the diameters (7.2, last row) |
| Q3 | The twins' $\mu_i$ | The pilot's partition-model $\hat\mu$ (7.6 b); the per-node fit under the hybrid (Block 30 via `infer_nodes.py`: `mu_hat_per_um`, `z_ax_hat_um`; it needs the cleaning run and uses the configured kernel, so step 2 would sit inside an outer iteration); a hybrid one-plane fit; or $\mu_i$ fitted per twin in step 2 |
| Q4 | The twins' axis depth | Step 1's $\hat z_{{\rm ax},i}$ as is (costs -5.4, +4.5, -3.1 %, 7.3); fitted per twin in step 2; or corrected by the blend's centroid (7.6 a). On 2026-11-27, form B with every twin's depth fitted (30 extra unknowns, sparse Jacobian), started from the configured ideal-Debye table and every depth at the nearest plane, recovered the table and the depths on the faint toy (both ranges, every start) and on the hybrid toy (check 19's truth; 270 evaluations including the Jacobian's) **[run, sandbox: `check_step2_without_step1.py`, `check_step2_without_step1_hybrid.py`, in the defocus-table chat]** |
| Q5 | The forward model | The full stack (`synthetic_block`, then `plane_scan` / `far_plane_scan`; the procedure's "full camera chain, same profile extraction"; 89.3 s per near phantom under the hybrid) or the fast line renderer (Block 35) with `dip_statistics` / `far_plane_statistics` (no 2-D background and no camera chain; its speed for this use not measured) **[open]** |
| Q6 | Noise in the twins | Noise-free, which gives a deterministic objective; or the camera chain's noise with common random numbers (a fixed seed per twin across $\theta$). The core fit's low-signal effects at faint far planes enter the real core widths and only noisy twins (doc s.3.6) |
| Q7 | The weights $\varpi$ | Open in D-229: their form and direction (e.g. the inverse of each plane's core-width noise variance); whether $k^*$ enters; the weight of far against near twins |
| Q8 | The far stage | The far twins' statistic (the assistant's reading of D-229.1: `far_plane_scan`'s core width, the far fit's own); first plane 4 (D-229.2) against the far fit's 5 (`far_min_k_offset`, PROVISIONAL); D-230.3's stop fraction and maximum number of passes [moot: D-264 drops the alternation; open instead, the slope held in the near stage]; the starting $\gamma$ (0.79, or a first far fit on step 1's anchor) |
| Q9 | The solver | `scipy.optimize.least_squares` (trf, the bounds of $\Theta$); a finite-difference Jacobian (the checks used `diff_step` 1e-3), with `jac_sparsity` when per-twin unknowns are added; each column of $J$ costs one render of every twin |
| Q10 | Where it runs | Colab or davinci. If davinci: how many cores (D-175.3's 100 were for the table run), and the roadmap's `needs_hpc` for NS1 |
| Q11 | Which nodes | D-226's chosen nodes (not built). Step 2 can be built and tested on phantoms first; the roadmap keeps the two decoupled |
| Q12 | Acceptance | The far validation planes (D-229.2): the criterion is open |
| Q13 | D-181 | Whether the renderer's spline reading is built before step 2 (item 4 of section 8). [2026-11-27: needed for the near stage at least, D-265: its twins read the spline and its tangent extrapolation past 0.84 um.] |

## 9a. Section 9 after the user's answers of 2026-11-27, 18:06 (D-270 to D-273)

| Q | Settled | By |
|---|---|---|
| Q1 | Allen's diameter, taken as true; form B with the diameters held | D-261 |
| Q2 | Not settled. The user would fit $\sigma_{\rm r}(0)$ in step 2 but does not want too many correlated unknowns; D-221.1 (held, switch `--kernel-width fixed\|calibrated --sigma-json FILE`) stands until confirmed. Evidence in D-273's "Why" | D-273.6 |
| Q3 | $\mu_i$ searched per near twin in the near stage, held in the far stage; PROVISIONAL until its statistic is agreed. Core widths alone cannot separate $\mu_i$ from the depth (correlation +1.00 on the proxy); the dip's area (`PlaneScan.area_gl_um`) is proposed as the second statistic | D-271.2 |
| Q4 | $z_{{\rm ax},i}$ fitted per near twin in the near stage, held in the far stage | D-271.1, D-271.3 |
| Q5 | The full stack: render, camera chain, `plane_scan` / `far_plane_scan` | D-270.1 |
| Q6 | Noisy twins, each plane with its real plane's noise; common random numbers across trial tables (the assistant's reading) | D-270.2 |
| Q7 | $\varpi(j) = 1/s_\omega(j)^2$ from noisy replicates, $\varpi \equiv 1$ selectable, the focus plane in; no far-against-near weight (D-264) | D-273.1 |
| Q8 | Far planes from 4 in step 2 and in the far fit (`far_min_k_offset` = 4); no alternation (D-264); the near stage holds no slope (D-265) | D-273.2 |
| Q9 | `least_squares` (trf), finite-difference Jacobian with sparsity, SEs clustered by node, no bootstrap for now | D-273.3 |
| Q10 | davinci follows from the full stack (the assistant's inference); cores and NS1's `needs_hpc` to confirm | D-270 |
| Q11 | Screening on davinci, the user selects and delivers per cell (specimen number) its nodes | D-261, D-273.5 |
| Q12 | A training/validation split of the chosen nodes, near and far both validated; on phantoms every knot within 0.002 um with the twins' noise off | D-273.4 |
| Q13 | The spline is the renderer's default between knots; the tangent continuation an option for the near stage | D-272, D-265 |

Still open, asked on 2026-11-27: (a) where a far-set twin's depth and absorption come from (far-set nodes never enter
the near stage); (b) the statistic that determines $\mu_i$; (c) whether $\sigma_{\rm r}(0)$ is fitted; (d) the split's
fraction, unit, stratification and seed; (e) the cores on davinci and NS1's `needs_hpc`. Section 10's breakdown is
superseded by the plan given to the user on 2026-11-27 (Block 4 amendment, then Blocks 52 onward); it is written here
once the user approves it.

## 10. A block breakdown (proposal, to agree with the user)

Each block, once agreed, gets a SPEC entry and joins the roadmap under NS1. At writing, the last SPEC block was 49
[updated 2026-11-27: Blocks 50 and 51 now exist, wave 8]; take the next free number when the entries are written, since another chat may have added blocks.

| Block | Module(s) | Delivers |
|---|---|---|
| Twin settings | `analysis/step2_twins.py` | $\xi_i$ per node, from the scans file's `nodes` and `PlaneScan`s, step 1's (or the far fit's) depths, and, when built, the user's list of chosen nodes. $\tilde d_i$ is the node record's `allen_diameter_um` (D-261). Q2 to Q4 become config fields |
| Twin forward model | `analysis/step2_forward.py` | $\omega^{\rm sim}_{i,k}(\theta;\xi_i)$ for all twins: $\theta$ -> `RendererConfig` (section 8, item 3); render (Q5); measure with the frozen scan configuration (section 8, item 2); parallel over twins |
| Estimator | `analysis/step2_fit.py` | Eq. (3) with $\varpi$ and the bounds; optional per-twin unknowns (depth, diameter, $\mu$); forms C and P as options; returns $\hat\theta$, cond $J$, the residuals by node and plane, and the per-twin fitted settings |
| Stages | same, or `analysis/step2_stages.py` | D-230 and D-264: the near stage, then the far stage, each once, no loop. Writes the production kernel in the `calibration_<S>.json` / `far_fit_<S>.json` format Cell 16 reads |
| Entry point, report, figures | `scripts/calibrate_kernel.py step2` (a new subcommand) and `plotting/` | Runs, JSON and report. Figure: real against twin core widths by plane, near and far, with the validation planes marked |

Candidate smoke-test oracles **[reasoning]**:

1. **Self-consistency.** "Real" nodes are twins rendered at a known $\theta_0$ and $\xi$; step 2 must return $\theta_0$
   within the solver's tolerance (noise-free), for the near stage and for the far stage.
2. **Scale invariance in the faint limit.** The core width is unchanged by $\mu$, within 2.2e-10 um$^2$ in doc check 4.
3. **The toys' numbers.** Check 19's row "true $\mu$, true depths" (recovered) and its "step 1's depths" row (the error
   reproduced in sign) as a regression, if the user uploads the zip.
4. **Determinism** under a fixed seed.
5. **Eq. 18's sign** at 0.28 um for twins at 1.2 $d$ and 0.55 $d$ (doc s.3.10).

## 11. Where things are, and what is not in the repository

| Item | Where | State |
|---|---|---|
| The T10 study document (md + zip with `figures/`, `scripts/`, `scripts/data/`) | Files sent in the defocus-table chat (2026-11-18; revised 2026-11-23) | Not in the repository; committing it needs the user's word |
| T10's roadmap note naming the revised document | Commit `e2fde39` on a local branch in the defocus-table chat's sandbox | Not pushed; `origin`'s note names the 2026-11-18 version |
| D-231's code | `2812d4d` (+ `ed6cc63`) on `6a39dcc`, in the defocus-table chat's sandbox | Not pushed (section 8, item 1) |
| The renderer check of check 20 | `checks/defocus_pool_2026-11-18/thick_near_check.py` (`ced566b`), without `--absorption` | On `origin` |
| The procedure | `docs/TEEG_diameter_bias_table_procedure_2026-10-04.md` s.3.4 | On `origin` |
| The davinci table run and D-226's change to it | `docs/TEEG_defocus_table_davinci_handoff_2026-11-18.md` (status note, s.7, s.9) | On `origin` |
| SPEC: Block 10 (s.5, line 574), Block 15 (690), Block 21 (831), Block 35 (1057), Block 42 (1274); s.8 "Open questions" | `specs/SPEC.md` | On `origin` |
