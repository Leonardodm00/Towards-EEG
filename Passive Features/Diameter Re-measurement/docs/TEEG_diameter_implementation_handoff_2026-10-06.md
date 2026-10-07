# Handoff: implementing the diameter pipeline (theory chat -> implementation chat)

| Date | Change |
|---|---|
| 2026-10-06 | v1. Written at the user's request (14:30 Europe/Rome): the whole diameter-estimation pipeline is implemented now, in a separate chat, while the theory study continues in the chat that wrote this file (decision D-021, below). For everything that is code, this file supersedes the "Next actions" of `TEEG_diameter_remeasurement_handoff_2026-10-05.md`; that file and the design handoff `handoff_diameter_remeasurement.md` stay the references for data access, method and equation numbers. Before commit, every number under "Findings" was re-run from the four scripts committed with this file (`checks/*.out`, identical to the original runs), and every cited parameter was re-read from its source document. |
| 2026-10-06 (later) | v1.1. The truncation check is now stated as procedure §3.10 writes it: frozen ($\gamma=0$) and proportional ($\gamma=\sigma_{\rm r}(0.84)/0.84\approx0.72$). The runs at $\gamma$ = 0.79 and 1.21 are labelled as a proposed addition. Two places are marked [corrected 2026-10-06]. Evidence: procedure §3.10, "Truncation" row, re-read. |
| 2026-10-06 (later, 2) | v1.2. Adds the method changes made after v1.1 -- D-025 (the stacks' noise is measured on clean background patches), D-026 (the kernel beyond ±0.84 µm is calibrated on dendrites just under 0.8 µm, $\gamma$ measured, Gaussian kept) and idea I-002 (a non-Gaussian kernel family) -- with their code consequences (section "Method changes after v1.1"). Their full text is in `docs/TEEG_decision_log_additions_2026-10-06.md`, because the project log could not be written (project knowledge full). New "Findings" row from `checks/calib_reach_check.py`, added with this version. Marks where the D-021 batch (D-022 to D-024) superseded provisional defaults of "Configuration", and corrects the stale statements about the log and the project-knowledge limit. Evidence: the project log re-read 2026-10-06 (D-001 to D-024, I-001); `config.py` on `sci/diameter-pipeline` (e908969) read; the new script run. |
| 2026-10-06 (later, 3) | v1.3. Adds D-027 (user, 18:12): the kernel calibration starts with flat branches only, near and far from focus; tilted branches come later, compared with the renderer's prediction. Rows added to "Decisions in force", "Method changes" and "Findings"; the calibration node-selection row marked. New script `checks/calib_width_check.py` (tilt and window effects on the width statistic); `checks/calib_reach_check.py` now also reports 0.5 µm nodes. Evidence: the user's message; both scripts run. |
| 2026-10-07 | v1.4. D-025 to D-027 and I-002 are now in the project log (merged 2026-10-07 by the theory chat): the "Decision-log facts" row, the D-025 row of "Decisions in force" and the "Admin left open" note are marked [corrected 2026-10-07, v1.4]; the size inference of v1.2 was not borne out. No method or configuration change. Evidence: `project_read` of the log 13:46 UTC and `project_write` 13:48 UTC (`replaced: true`), 2026-10-07. |
| 2026-10-07 (later) | v1.5. Adds **D-030** (user, 17:02): every real node gets its own corrected diameter whatever its optical depth; $\hat\alpha\le1.0$ leaves $\mathcal S$ for phantoms and real nodes, and the dark flag becomes a diagnostic column that no fill rule reads. Rows added to "Decisions in force" and "Method changes"; the "selection $\mathcal S$" and "dark-tube flag" rows of "Configuration" and the partition row of "Findings" marked [corrected 2026-10-07, v1.5]. Evidence: the user's message; the project log re-read (unchanged since 13:48 UTC) and written with D-030, 2026-10-07. |

Paths are relative to `Passive Features/Diameter Re-measurement/` in the repo
`Leonardodm00/Towards-EEG` unless they start with `claude/` (project knowledge).
"handoff Eq. n" = `docs/handoff_diameter_remeasurement.md`; "procedure Eq. (n)",
"mathematics Eq. (n)", "optics §n" = the three 2026-10-04 documents in `docs/`;
"(S n)" = equations derived in the theory chat, written out in this file only
(section "Equations S1-S10").

## Provenance of this file

| Claim class | Status |
|---|---|
| Repo state | **verified 2026-10-06** on a fresh clone: `main` = `a4e0739`. This folder holds `docs/`, `checks/`, `figures/`, `README.md` and **no pipeline code**. There is no `specs/` folder anywhere in the repo. |
| Image-access modules | **verified 2026-10-06**. The Drive connector shows only the 2026-09-22 versions in the `Allen Slices/Codes` folder (`allen_image_io.py` 15664 B, `allen_image_align.py` 7139 B, `allen_image_plot.py` 7807 B, `smoke_allen_image.py` 11739 B, `allen_image_measure.py` 5373 B, all modified 2026-09-22; no `robustness_registration.py`). The only known copies of the 2026-09-23 versions are the files delivered in the design chat (claude.ai chat `126e0cd6…`, 2026-09-23 -> 09-30, also packed in its `allen_slice_viewer.tar.gz`). The user may hold others locally; that was not checked. The project-knowledge snapshot `claude/allen_viewer_code_2026-09-23.md` named in both earlier handoffs is **not** in the project (project_info, 2026-10-06). |
| Allen API from the Claude sandbox | **blocked, re-tested 2026-10-06**: `http://api.brain-map.org` returns 403, and `https` is refused at CONNECT by the egress policy. All real-data code runs in Colab. |
| Allen facts for 529878215 (ids, sizes, pixel size, plane step) | from the design handoff (live queries before 2026-09-30); **not re-queried** |
| Decision-log facts | **read 2026-10-06** from project knowledge: the root-level `TEEG_decisions_and_ideas_log.md` holds D-001 to D-019 and I-001; `claude/TEEG_decisions_and_ideas_log.md` holds only D-001 and I-001 (last change 2026-09-19). **[corrected 2026-10-06, v1.2]** Since then the root log holds D-001 to D-024 and I-001 (re-read 2026-10-06); the `claude/` copy was deleted with the user's OK; D-025, D-026 and I-002 are pending in `docs/TEEG_decision_log_additions_2026-10-06.md`. **[corrected 2026-10-07, v1.4]** D-025 to D-027 and I-002 are now in the root log as well (merged 2026-10-07); it also holds D-028 and D-029 of the Ih Fit workstream, and the next free IDs are D-030 and I-003 |
| Repo visibility | **public** (GitHub repository listing, 2026-10-06), so Colab can clone without a token |
| Equations (S1)-(S10) | derived in the theory chat (2026-10-05/06). (S1)-(S4) and the ray-world chord were checked numerically **[run]**; (S9) and (S10) checked by quadrature **[run]** (scripts in `checks/`, see "Findings"). |
| Numbers under "Findings" | **[run]** with the *illustrative* ideal-Debye kernel or the ray world (geometric optics). No Allen data was used. Reproduced 2026-10-06 from the committed scripts; reference outputs `checks/*.out`. |
| Provisional defaults (section "Configuration") | the assistant's proposals, mostly the documents' own recommendations; **none confirmed by the user**. Confirm them in one batch before writing the logic they control. |
| Pillow can read and reuse JPEG quantization tables | **verified 2026-10-06** with Pillow 12.3.0 in the sandbox: `Image.open(f).quantization` returns the tables; re-encoding with `save(..., qtables=q)` gives back the same tables, and so does `quality="keep"` on an opened JPEG |

## How the two chats split the work

| Chat | Owns | Writes |
|---|---|---|
| theory chat (the one that wrote this file) | the method: `docs/` (procedure, mathematics, optics, notes) | doc commits on `main` (D-020); decision entries |
| implementation chat | the code: `src/`, `specs/`, `tests/`, `scripts/` (layout below) | commits on branch `sci/diameter-pipeline`; `specs/SPEC.md`; decision entries |

- A method change arrives as a doc commit plus a decision ID. The implementation chat pulls `main` at the start of each session and reads the changed sections (`git log --stat origin/main -- "Passive Features/Diameter Re-measurement/docs"`).
- By D-021, such a change should normally cost a configuration value or one block, not the architecture. If it cannot, change `specs/SPEC.md` first and name the blocks affected (scientific-coding skill, block protocol).

## D-021 (in the project log since 2026-10-06)

**[corrected 2026-10-06, v1.2]** v1's heading said "new; not yet in the
project decision log, which cannot be written: project knowledge is full".
D-020 and D-021 were appended to the root log the same day, after the stale
`claude/` copies were deleted with the user's OK.

**Date:** 2026-10-06, 14:30 **[user]**. **Binds:** the whole diameter-estimation
code (`src/`, `specs/SPEC.md`, the smoke suites); how open method choices enter
the code.

**Decision [user].** "In this chat I will continue to study the theory
underlying the process. However it is wise to save time that you start to
implement the whole pipeline for the diameter estimation [...] Is much faster
to implement slight changes while the entire code is already set rather than
waiting me to comprehend in deep everything (this takes time so in the
meantime let's implement it)." The pipeline is implemented now, end to end,
in a separate implementation chat, in parallel with the theory study.

**Statement.**
- Every choice the method has not settled enters the code as a named parameter of a configuration object. Each starts at a stated provisional default with its source written next to it.
- An alternative method (estimator variant, kernel family, absorption treatment) is a swappable component chosen by name from the configuration.
- Later theory results change a parameter value or a single block.
- **[assistant, under the project's coding directives 2-3 and the scientific-coding skill]** The provisional defaults are put to the user in one batch at the start of the implementation chat. Logic whose default is not confirmed is not written.

**Why.** Code that runs end to end on synthetic stacks exposes interface and
scale problems early. Changing a default is cheap once the structure exists.
The open questions (procedure §5) are mostly values, not structure.

**What it implies for code.**
- `config.py` holds every parameter in the section "Configuration" below.
- Each default carries a `# source:` comment, and `SPEC.md` cites the decision or document behind it.
- The non-deliverable variants are labelled comparisons in file names and figure titles (project-decision-log skill: "a default is a decision").

**What it does not decide.** Any of the open method choices themselves (D-018
(a), (c); D-019 (a), (b); kernel family; grid; selection rule; mean or median;
D5; D7), and where the code lives (section "Proposed layout").

**Status:** active. To be appended to the project log `TEEG_decisions_and_ideas_log.md`
(root-level copy; see "Admin") when space exists, together with D-020.
**[corrected 2026-10-06, v1.2]** Appended; the batch was answered by D-022 to
D-024.

## Decisions in force (read before coding anything they bind)

| ID | Where the full text is | One line | Open sub-points |
|---|---|---|---|
| D1-D7 | design handoff §Method/Decisions | D2: width read across the branch in the sharpest plane; D3: local 3-D path from several planes; D4: blurred-tube Beer-Lambert fit (amended by D-018, D-019); D6: compute in the global frame in µm, never fit a direction in px/plane units; **D5** (bias table + flag \|b-1\| > 0.2) and **D7** (start with 529878215) are "proposed, **confirm**" | D5, D7 -- **confirmed by D-024 (vi)** |
| D-013 | project log (root copy) | corrected morphologies = one SWC per specimen, archive format; `run_ih_fit.py --swc-dir` refuses a missing file | where the files live |
| D-018 | project log (root copy) | background $B$ of the fit fixed to a median $\bar B_i$ of the focal plane $k^*_i$ (D-018.1); $\sigma$ fixed | (a) region $\mathcal R_i$; (b) stained-fraction bias; (c) focus-score background -- **(a), (c) closed by D-023** |
| D-019 | project log (root copy) | fitted darkness is $\mu$ (µm⁻¹); $\alpha=\mu d/\cos\varphi_i$ with $\varphi_i$ from the line fit (D-019.1) | (a) $\mu$ per node or shared; (b) whether the vertical path $d/\cos\varphi_i$ holds for thick or steep branches (see "Findings": the partition and vertical rays); (c) one table per estimator variant -- **(a) closed by D-023 (per node)** |
| D-020 | this folder's `README.md` | documents live in `docs/` on `main`, updated by commit and push | — |
| D-021 | this file; project log | implement now; open choices as provisional config defaults, confirmed in one batch | — |
| D-022 | project log (root copy) | code in this folder on branch `sci/diameter-pipeline`; root `specs/SPEC.md`; the 2026-09-23 modules byte-identical; production table on davinci (PBS) | root `tests/smoke` redirect; PBS array layout |
| D-023 | project log (root copy) | $\mu$ per node; $\bar B_i$ over the block, traced path masked; focus score keeps the profile-ends median; $\sigma_{\rm fit}$ a study axis $\{0.080, 0.099, 0.125\}$ µm, deliverable 0.099 | the study set and the deliverable value |
| D-024 | project log (root copy) | partition renderer and linear continuation $\gamma=0.79$ "for now"; dark flag $\hat\alpha>1.0$; random phantom design over ranges, $U=10$ µm, $\varphi\in[0°,90°)$; no tilt exclusion; thin-plate spline; D5, D7 confirmed; jitter 0 until cell 13 | $d_{\max}$, $\mu$ range, flag threshold, replicates, jitter; **(i)'s $\gamma$ to be measured, D-026**; **(ii) superseded in part by D-030** |
| D-025 | `docs/TEEG_decision_log_additions_2026-10-06.md`; ~~pending in the project log~~ in the project log (root copy) since 2026-10-07 **[corrected 2026-10-07, v1.4]** | the stacks' noise is measured on clean background patches, in every plane, at several positions and in several stacks: second moment first, then its distribution | patch size and margin; level dependence; injection route |
| D-026 | same file | beyond ±0.84 µm the kernel is calibrated on dendrites just under 0.8 µm (Allen diameter), read beyond 0.84 µm; (D-026.1) with $\gamma$ fitted; Gaussian kept | thick-set selection; windows; "other terms"; joint or sequential fit |
| I-002 | same file | idea: a non-Gaussian kernel family with the Gaussian as a special case | to discuss |
| D-027 | same file | the calibration starts with flat branches only ($\hat\varphi_i\approx0$), near and far from focus; tilted branches later, compared with the renderer's prediction | tilt tolerance; tilt bins; nodes per bin |
| D-030 | project log (root copy) | every real node gets its own corrected diameter whatever its optical depth; $\hat\alpha\le1.0$ leaves $\mathcal S$ (phantoms and real nodes); `dark` is a label, not a fill trigger | the label's threshold; $\log\mu$ or $\hat\alpha$ as a regression axis; the renderer for dark nodes (D-024 (i) study) |

## Method changes after v1.1 (D-025, D-026, I-002, D-027, D-030)

The decisions above settle what v1.1 left provisional; these three change
what blocks 4, 10 and 11 must do. Full text, the assistant's comments and the
sources: `docs/TEEG_decision_log_additions_2026-10-06.md`. The code
consequences are proposals; names are fixed in `specs/SPEC.md`.

| ID | What changes | Code consequence |
|---|---|---|
| D-025 | the camera chain's noise is measured, not configured | Phase-II script, one row per clean background patch: stack, plane, position, size, mean $m$, detrended variance (D-025.1), autocovariance at 1-8 px in $x$ and $y$, plane-difference variance, clean-column criterion. `RendererConfig.noise_sd_gl` (3.0, provisional) gives way to a noise-model selector whose parameters come from that table (e.g. `"gaussian_after_chain"`, `"empirical_patches"`). A clean column has no stained structure on either side of the plane within about $3\sigma_{\rm r}(\delta)$. |
| D-026 | $\gamma$ of `kernel_continuation = "linear"` is measured | Block 10 takes a calibration-node table with a diameter class per node: thin ($\hat d\lesssim0.3$ µm) and thick (Allen diameter just under 0.8 µm, faint, $\hat\alpha\lesssim0.5$). Each class has its own plane window; the thick class is read beyond 0.84 µm, on both sides of the axis where possible. For the thick class the profile and the block widen to about $\pm3\sigma_{\rm r}$ ($\sigma_{\rm r}\approx$ 2-3.5 µm there), with a background trend term, and $z_{{\rm ax},i}$ comes from the node's own focus search instead of Eq. 4, because an axis-depth error mimics a change of the anchor $\sigma_{\rm r}(0.84)$. Output: the tabulated $\Delta\sigma^2$, its measured range (an element of $\mathcal C$) and $\hat\gamma$. Phase-I oracle: synthetic thick-node scans with a known $\gamma$ return it. |
| I-002 | nothing to code now | `kernel_family` is already a configuration name, so a new family can be added as one component |
| D-027 | both calibration sets start with flat nodes; tilt is studied later | `CalibrationConfig` gains a tilt tolerance for both sets (e.g. `phi_tol_deg`, provisional 5°) and a tilt-binned mode whose output is compared with the renderer's prediction from the flat calibration, labelled as a comparison; the width statistic is the core width (see "Findings") |
| D-030 | no real node is filled or left uncorrected because of its optical depth | Blocks 6-7: remove $\hat\alpha\le1.0$ from $\mathcal S$ for phantoms and real nodes; `alpha_dark_flag` only writes a `dark` column, which the fill rules do not read. Smoke check: a node with $\hat\alpha>1.0$ is corrected, not filled (`filled_from` empty), and has `dark = True`; the table build keeps phantom replicates with $\hat\alpha>1.0$ and reports its $\hat\alpha$ coverage. Dark nodes' corrections rest on the partition renderer where it is not validated ("Findings", partition row (4)); the D-024 (i) study, with `ray_world` (S6) as the comparison, is the check. |

## What "the whole pipeline" is

```
(A) REAL-DATA PATH, per dendrite node (Colab only)
    fetch block            allen_image_io.fetch_zblock  -> (block, ks, valid, frame)
    registration / snap    allen_image_align.registration_check  (cell 13; verdict, s*, dz*)
    Pass 1, node j         focus score F_jk, k*_j, sub-plane depth   (handoff Eqs. 1-2)
                           centre c_j                                (handoff Eq. 3)
    Pass 2, node i         line fit over window L -> t_i, theta_i, phi_i, y_hat_i  (handoff Eqs. 4-5)
    measure                profile I_i(v) in plane k*_i              (handoff Eq. 9)
                           B_bar_i                                   (D-018.1)
                           fit (d_hat_i, mu_hat_i, v0_hat_i)         (D-019.1)
    correct                d_tilde_i solves m_hat(d, phi_i | C) = d_hat_i   (procedure Eq. 1, §3.9)
                           flags, fill from neighbours, running median      (handoff step 6)
(B) SYNTHETIC PATH (sandbox for tests, Colab or sandbox for the table)
    phantom (d, phi, mu) + nuisances xi -> renderer (S1-S4, procedure Eqs. 5-6, camera chain)
    -> block array -> THE SAME per-node chain as (A), unchanged (procedure §3.7)
    -> d_hat_n -> b_hat(d, phi | C), tau_hat, SE   (procedure Eqs. 3, 7) -> table + C dictionary
(C) KERNEL CALIBRATION (Colab, real data): plane scans of thin faint flat nodes
    -> procedure Eq. 4 -> sigma_r(delta) on |delta| <= 0.84 um -> renderer kernel
(D) CELL LEVEL: all dendrite nodes of 529878215 -> per-node CSV -> corrected SWC (D-013 format)
    -> dendritic membrane-area ratio vs Allen -> Cm refit on davinci (Ih Fit, --swc-dir)
```

The per-node chain in (A) and (B) must be **one function** fed either by
`fetch_zblock` or by the renderer. That is what makes the table the bias of
the code actually used: the phantoms go through the real pipeline unchanged
(procedure §3.7), and the table is a property of the estimator (procedure §3.2).

## Proposed layout (confirm before scaffolding)

Precedent in this repo:
- Recent workstreams live in plain folders under `Passive Features/` (e.g. `HPC script/Ih Fit/`).
- The installable package `towards_eeg/` has a `passive/` subpackage whose docstring says passive fits are "managed separately".

Proposal: keep the code inside this workstream folder.

```
Passive Features/Diameter Re-measurement/
  specs/SPEC.md                 new (scientific-coding skill template); one entry per block
  src/
    allen_image_io.py           2026-09-23, byte-identical (sha256 below)
    allen_image_align.py        "
    allen_image_measure.py      "
    allen_image_plot.py         "
    allen_diameter/             new package
      config.py                 all parameters (section "Configuration"); dataclasses; # source: per default
      loading/   swc_io.py
      model/     geometry.py (S1-S5), tube_model.py (handoff Eqs. 10-11, D-019.1 forward),
                 kernel.py (sigma_r table + continuation), render.py (procedure Eqs. 5-6),
                 camera.py (pixel integration, grey mapping, noise, 8-bit, JPEG), ray_world.py (S6)
      analysis/  focus.py (Eqs. 1-2), path.py (Eqs. 3-5), background.py (D-018.1), fit.py (D-019.1),
                 node_pipeline.py (the shared per-node chain), phantoms.py, table.py (Eq. 7 + C),
                 invert.py (§3.9), fill.py, calibration.py (Eq. 4), cell.py (CSV, SWC, area)
      plotting/  figures.py
  scripts/       colab_bootstrap.py, run_node.py, run_cell.py, build_table.py, calibrate_kernel.py
  tests/smoke/   smoke_allen_image.py, robustness_registration.py (2026-09-23), test_smoke_<block>.py
  docs/ checks/ figures/        existing
```

- The four 2026-09-23 modules stay flat and byte-identical at first, so that their own suites stay valid (`smoke_allen_image.py` 20/20, `robustness_registration.py` 40/40 on 2026-09-30). Whether they import each other by flat module name is **not verified here**.
- D-018, D-019 and the design handoff (Next action 5) name the measurement code `allen_image_diameter.py` with `smoke_allen_image_diameter.py`. That was a plan, not a file. The package `allen_diameter/` replaces the single module, and the smoke tests become one per block. This is a naming proposal; record it in SPEC.md when confirmed.
- Alternative layout: `towards_eeg/passive/diameter/` in the installable package. It is not recommended because of the docstring above and the repo's S0 ledger rules for that tree, which were not read for this file.

**Colab bootstrap** (proposal; replaces copying files to Drive, which is how
Drive came to hold stale versions; the repo is public, so no token is needed
to clone):

```python
import os, sys, subprocess
# BRANCH exists only after the implementation chat has pushed it; until then use "main"
REPO, BRANCH = "/content/Towards-EEG", "sci/diameter-pipeline"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "50", "-b", BRANCH,
                    "https://github.com/Leonardodm00/Towards-EEG.git", REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=True)
SRC = os.path.join(REPO, "Passive Features", "Diameter Re-measurement", "src")
sys.path.insert(0, SRC)
import allen_image_io as aio, allen_image_align as aia, allen_diameter
# keep HttpFetcher's cache on Drive, as in cells 1-12: HttpFetcher(cache_dir=CACHE_DIR)
```

## Block plan (proposal; order chosen so an end-to-end run exists early)

Phase I runs entirely in the sandbox on synthetic data. Phase II needs Colab and real data.

| # | Block | Module(s) | Equations / source | Smoke oracles (at least one strong) |
|---|---|---|---|---|
| 0 | scaffolding; import the 2026-09-23 modules | `src/*.py`, `tests/smoke/` | design handoff §Code | sha256 prefixes match; their suites 20/20 and 40/40 |
| 1 | configuration | `config.py` | section "Configuration" | every default present with a source tag; defaults asserted (D-018: no fitted $B$; D-019: $\mu$ form; D-021) |
| 2 | geometry | `model/geometry.py` | (S1)-(S5) | 3-D membership test of handoff Eq. 6 vs (S3): 0 mismatches; slab chords vs brute force; $\sum_j a_j=\mu\ell(v)$ |
| 3 | fit forward model + fitter | `model/tube_model.py`, `analysis/fit.py` | handoff Eqs. 10-11, D-018.1, D-019.1 | noise-free recovery of $d$ at $\alpha\in\{0.3,3\}$, $d\in\{0.5,1,2\}$ µm (handoff "Numbers checked"); $\mu$ form vs $\alpha$ form give the same $\hat d$ (D-019 guard; `checks/mu_tie.py`); faint-limit dip area $=\bar B\mu\pi d^2/(4\cos\varphi)$ |
| 4 | renderer | `model/kernel.py`, `render.py`, `camera.py` | procedure Eqs. 5-6; (S4); FFT form below | conservation $\sum_j\Delta A_j=1-e^{-\sum a_j}$ (C1); one slab at $\delta=0$ with $K_0=g$ reproduces handoff Eq. 11 (C2); fine-grid rotation equivariance in $\theta$; convergence in $h_{\rm g}$, $\delta\zeta$, $U$; JPEG tables round trip |
| 5 | per-node chain | `analysis/focus.py`, `path.py`, `background.py`, `node_pipeline.py` | handoff Eqs. 1-5, 9; D-018.1 | recovers known $\theta,\varphi$; $\hat y\perp$ branch for any $\theta$; D6 guard (a pixel-unit fit inflates the tilt); flags fire for $\varphi\ge30°$ and for empty tissue (design handoff Next action 5); parabola vertex within $\pm\Delta z/2$; **first end-to-end gate** (procedure §3.10, first row): single-depth phantoms (all absorbance at $\delta=0$), $\varphi=0$, $d$ = 0.5-1 µm, with $\sigma_{\rm fit}^2\approx\sigma_{\rm r}(0)^2+p_{\rm x}^2/4$ ($p_{\rm x}^2/12$ from pixel integration plus $p_{\rm x}^2/6$ from bilinear interpolation, D-018 status note) give $\hat b\approx1$ |
| 6 | phantoms + table | `analysis/phantoms.py`, `table.py` | procedure §3.5, Eqs. 2-3, 7, §3.11 | seed determinism; nuisance laws; SE $=\hat\tau/\sqrt{N_{\mathcal S}}$ on a known-mean fixture; table refuses a mismatched $\mathcal C$ |
| 7 | inversion, flags, fill | `analysis/invert.py`, `fill.py` | procedure Eq. 1, §3.9; mathematics Eqs. 20-21 | root vs shortcut error $\approx-\beta(b-1)$; non-monotone and out-of-domain tables flagged; fill rules; after correction, synthetic stacks recover $d$ within ~10 % for $d\in\{0.5,1,2,3\}$ µm and $\varphi\le20°$ (design handoff Next action 5) |
| 8 | cell level | `analysis/cell.py`, `loading/swc_io.py` | handoff step 7; D-013 | SWC round trip byte-for-byte apart from radii; frustum area of known cylinders and cones |
| 9 | ray-world generator + §3.10 checks | `model/ray_world.py`, `scripts/` | (S6), chord formula | reproduces `checks/optics_points_check.py` self-checks; faint-limit match (S10) |
| 10 | kernel calibration | `analysis/calibration.py` | procedure Eq. 4, §3.4; mathematics §3.5; **[2026-10-06]** D-026 (thick set, measured $\gamma$) | synthetic plane scans with a known growth recovered up to the common shift (Phase I); synthetic thick-node scans with a known $\gamma$ return it (D-026); real nodes in Phase II |
| 11 | real-data runs (Phase II) | `scripts/` | design handoff Next actions 1-3, 7 | cell 13 on node 4505 + 5-10 stretches; camera-model calibration; $\hat\mu_i$ distribution; table; apply to 529878215 |

**Per-node CSV columns** (from D-018, D-019 and handoff step 7):
- identity and position: `node_id`, `type`, `x_um`, `y_um`, `z_um`, `path_um`;
- registration: `reg_verdict`, `s_star_um`, `dz_star_um`;
- focus and path: `k_star`, `z_sub_um`, `cx_um`, `cy_um`, `cz_um`, `theta_rad`, `phi_rad`;
- background: `B_bar`, `B_bar_region` (the rule used);
- fit: `d_hat_um`, `mu_hat_per_um`, `v0_hat_um`, `alpha_hat` (derived, diagnostics only), `fit_status`;
- correction: `b_hat`, `d_tilde_um`, `flags`, `filled_from`, `d_final_um`, `allen_radius_um`.

The corrected SWC keeps the archive format (D-013), with radius $=d_{\rm final}/2$ on dendrite nodes (types 3, 4) and everything else unchanged.

## Equations S1-S10 (written nowhere else)

Notation:
- Phantom: radius $r$, true diameter $d=2r$ (µm), tilt $\varphi\in[0,\pi/2)$, heading $\theta\in(-\pi,\pi]$, absorption coefficient $\mu\ge0$ (µm⁻¹), node point $c=(c_x,c_y,c_z)$.
- Frame: $(x,y)$ specimen-referred µm; $z$ in stage units, increasing **in the direction the light travels** (how Allen's plane index relates to that direction is **not verified**).
- Slabs: slab $j$ spans depths $[\zeta_j-\delta\zeta/2,\ \zeta_j+\delta\zeta/2]$. Output plane $k$ sits at depth $z_k$.

**(S1) Local frame.** For each fixed point $(x,y,z)\in\mathbb R^3$:
$$u=(x-c_x)\cos\theta+(y-c_y)\sin\theta,\qquad v=-(x-c_x)\sin\theta+(y-c_y)\cos\theta,\qquad w=z-c_z.$$

**(S2) Membership** (handoff Eq. 6 in these coordinates):
$$(x,y,z)\in\text{tube}\iff v^2+(w\cos\varphi-u\sin\varphi)^2\le r^2 .$$

**(S3) One vertical line.** For each fixed $(x,y)$, the vertical line is inside the tube if and only if $\lvert v\rvert\le r$ and
$$z\in[z_{\rm lo},z_{\rm hi}],\qquad z_{\rm lo/hi}(u,v)=c_z+u\tan\varphi\mp\tfrac12\ell(v),\qquad \ell(v)=\frac{2\sqrt{r^2-v^2}}{\cos\varphi}.$$
$\theta$ enters only through (S1). Rotating the image by $\pi$ about the optical axis maps the pixel grid onto itself, up to the sub-pixel offset that is drawn uniformly anyway. It also maps the tube $(\theta,\varphi)$ onto $(\theta+\pi,\varphi)$ and leaves $z$, and so the light direction, unchanged. Hence the draw $\theta\sim U[0,\pi)$ (procedure §3.5).

**(S4) Exact slab absorbance and column history.** For each fixed $(x,y)$ with $\lvert v\rvert\le r$:
$$a_j(x,y)=\mu\,\max\{0,\ \min(z_{\rm hi},\zeta_j+\tfrac{\delta\zeta}2)-\max(z_{\rm lo},\zeta_j-\tfrac{\delta\zeta}2)\},\qquad T_{<j}(x,y)=\exp\!\big(-\mu\,\max\{0,\ \min(z_{\rm hi},\zeta_j-\tfrac{\delta\zeta}2)-z_{\rm lo}\}\big).$$
Both are $0$ (respectively $1$) for $\lvert v\rvert>r$. Then $\Delta A_j=T_{<j}(1-e^{-a_j})$ (procedure Eq. 5) needs no cumulative sum, and $\sum_j a_j=\mu\ell(v)$ when the slabs tile the column.

**(S5) Depth reach.** For a tube cut at $\lvert u\rvert\le U$ (vertical end cuts, independent of the block and of $\theta$), and for each plane $k$, the farthest point of the tube from plane $k$ is at
$$\max_{\text{tube}}\lvert z-z_k\rvert=\lvert z_k-c_z\rvert+U\tan\varphi+\frac{r}{\cos\varphi},$$
and the slab centres $\zeta_j$ reach it to within $\delta\zeta/2$.

**FFT form of procedure Eq. 6 (Gaussian kernel).** For each plane $k$:
$$I_k=B\Big[1-\mathcal F^{-1}\Big\{\sum_j\mathcal F\{\Delta A_j\}(\mathbf f)\,e^{-2\pi^2\sigma_{\rm r}(\zeta_j-z_k)^2\lvert\mathbf f\rvert^2}\Big\}\Big]$$
- One forward FFT per slab, one inverse FFT per plane, on an array zero-padded to twice the grid, so the convolution is linear, not circular.
- The cost no longer grows with $\sigma_{\rm r}$.
- **[textbook: Fourier transform of a Gaussian; used in `checks/stack_geometry_check.py`]**

**(S6) Ray world (alternative generator for the end-to-end check).** In geometric optics, for each fixed plane $k$ and lateral point $\mathbf x$:
$$I^{\rm ray}_k(\mathbf x)=B\,\big\langle e^{-\mu L(\mathbf x,z_k,\hat s)}\big\rangle_{\hat s},$$
where $L$ is the length inside the tube of the line through $(\mathbf x,z_k)$ with direction $\hat s$. The average runs over directions with $(s_x,s_y)$ uniform on the disc of radius $s_m=\mathrm{NA}/n_{\rm oil}=0.924$ (evenly filled aperture, sine condition, index-matched).

The chord is analytic. With $q=P-c$, $\hat t=(\cos\varphi\cos\theta,\ \cos\varphi\sin\theta,\ \sin\varphi)$, define
$$A=1-(\hat s\cdot\hat t)^2,\qquad B'=q\cdot\hat s-(q\cdot\hat t)(\hat s\cdot\hat t),\qquad C=\lvert q\rvert^2-(q\cdot\hat t)^2-r^2 .$$
When $B'^2>AC$, the line is inside for $t\in\big[(-B'-\sqrt{B'^2-AC})/A,\ (-B'+\sqrt{B'^2-AC})/A\big]$, intersected with $\lvert u(t)\rvert\le U$.

Limits of the ray world:
- It has no diffraction, so its blur near focus is too wide. Use it to test the *treatment of absorption*, never for absolute $b$.
- It has no coherence.

**(S7)/(S8) What the ray world tests.** Telescoping (S6) along each ray and writing the renderer with the kernel of the same rays ($K_\delta$ = law of $\delta\,\mathbf t_{\hat s}$, $\mathbf t_{\hat s}=(s_x,s_y)/s_z$, $\mathbf x_j=\mathbf x+(\zeta_j-z_k)\mathbf t_{\hat s}$):
$$I^{\rm ray}_k=B\Big[1-\Big\langle\sum_j T^{\rm ray}_{<j}\big(1-e^{-a_j(\mathbf x_j)/\cos\vartheta}\big)\Big\rangle_{\hat s}\Big]\quad\text{vs}\quad I^{\rm P}_k=B\Big[1-\Big\langle\sum_j T_{<j}(\mathbf x_j)\big(1-e^{-a_j(\mathbf x_j)}\big)\Big\rangle_{\hat s}\Big].$$
Here $\vartheta$ is the polar angle of $\hat s$ ($\cos\vartheta=s_z$), and the left form is exact in the limit $\delta\zeta\to0$. The two forms differ only in where the history runs (along the ray vs along the vertical line through the crossing point) and in the path length ($a_j/\cos\vartheta$ vs $a_j$). For $J=1$ both histories equal 1; for faint stain both tend to 1.

**(S9) Opaque-limit leak of the partition** (flat tube; a ray entering at $v_e$ with lateral slope $s_v/s_z$; $h$ = height above the skin $z_{\rm lo}(v)$ of the column the ray is in):
$$\int\mu e^{-\mu h}\,dz\ \xrightarrow{\ \mu\to\infty\ }\ \frac{1}{1-z_{\rm lo}'(v_e)\,s_v/s_z}.$$
The integral runs along the ray, which is what the partition books to that ray. In the opaque limit it is concentrated where the ray crosses the lower skin, at rate $dh/dz=1-z_{\rm lo}'(v_e)\,s_v/s_z$ there.
- The value is $<1$ for every ray through the centre point with $s_v\ne0$, because there $z_{\rm lo}'(v_e)\,s_v<0$; it is 1 when $s_v=0$. Light leaks through an opaque tube.
- It is $>1$, more than the light the ray carries, when $z_{\rm lo}'(v_e)\,s_v/s_z>0$, i.e. the ray's lateral motion follows the skin's increase in $z$.
- Column sums are conserved (procedure Eq. 5 telescopes per vertical column).
- Checked: the cone average of this limit at the centre point of a flat $d=1$ µm tube is 0.691 (`focus_scan.py`), against a partition dip of 0.699 computed at $\mu d=50$ (`followup_checks.py`).

**(S10) Obliquity.**
$$\Big\langle\frac1{\cos\vartheta}\Big\rangle=\frac{2}{s_m^2}\Big(1-\sqrt{1-s_m^2}\Big)=1.447,\qquad\max\frac1{\cos\vartheta}=\frac1{\sqrt{1-s_m^2}}=2.62$$
for a fully open, evenly filled NA 1.4 condenser, index-matched (quadrature: 1.4470, `optics_points_check.py BC`).

What follows from it, with hypotheses:
- **Faint limit, flat tube.** A ray of polar angle $\vartheta$ through a tube that is invariant along $u$ has a chord whose integral over lateral position is $A/\cos\vartheta$, with $A=\pi d^2/4$ the cross-section **[reasoning]**. The blur preserves the dip's area. So the dip area under the full cone is $\langle1/\cos\vartheta\rangle$ times the vertical-ray area.
- **Vertical-ray renderer.** It reproduces that area only with $\mu_{\rm ph}=\langle1/\cos\vartheta\rangle\mu$ (matched in the ray world: 1.450, `optics_points_check.py matched`).
- **The fitted $\hat\mu_i$.** It equals $\langle1/\cos\vartheta\rangle\mu$ only if the fit's blur reproduces the profile's shape, so that $\hat d\approx d$. In the ray world, whose geometric blur is too wide near focus, the faint fit instead returns $\hat d/d=1.20$ and $\hat\mu/\mu=0.96$.
- **Either way.** $\hat\mu_i$ is an effective coefficient: comparable between nodes (D-019), but not a material constant. Matching the phantom $\mu$ through $\hat\mu$ (procedure §3.5) absorbs the factor.

## Configuration: every open choice as a parameter

All values are **provisional (assistant)** unless the Source column cites a decision. Confirm them in one batch.

**[corrected 2026-10-06, v1.2]** The batch was answered by D-022 to D-024,
and D-025 and D-026 followed. Where they differ from the tables below, the
decisions and `config.py` / `specs/SPEC.md` on `sci/diameter-pipeline` are
authoritative. Rows superseded so far, each also marked in place:
`sigma_fit_um` (a study axis, D-023), the selection $\mathcal S$ (no tilt
criterion, D-024 (iv)), `U_um` (10 µm, D-024 (iii)), `grid_d_um` and
`grid_phi_deg` (a random design over ranges, D-024 (iii)), the camera-chain
noise (measured, D-025), `kernel_continuation` and the kernel calibration
(thick set, measured $\gamma$, D-026).

**Acquisition (facts, not choices)**

| Parameter | Value | Source |
|---|---|---|
| `specimen_id` | 529878215 | D7 (proposed; confirm) |
| `res0_um` (pixel pitch) | 0.1144 | design handoff, Allen API (not re-queried) |
| `dz_um` (plane step, stage units) | 0.28 | design handoff (`scale_factor_z`) |

**Per-node measurement (path A, shared with B)**

| Parameter | Provisional default | Source / status |
|---|---|---|
| `node_step` | every SWC node (≈ 1.18 µm) | design handoff Next actions 4 |
| `block_half_um` | 5 (≈ 10 × 10 µm block) | design handoff |
| `planes_half` | 3, widened for steep pieces by $\lceil(L/2)\sin\varphi/\Delta z\rceil$ | design handoff; widening rule is the assistant's |
| `focus_smooth_px` | 1 | handoff Eq. 1 |
| `focus_bg_rule` | `"profile_ends_median"` ($\lvert v\rvert>1.5$ µm, handoff Eq. 1 as written) | D-018 (c) open |
| `subplane_depth` | on (handoff Eq. 2; middle of the plateau for flat tops) | handoff Eq. 2 "optional" |
| `line_fit_window_um` ($L$) | 4.0 | design handoff Next actions 4 |
| `profile_half_um`, `profile_step_um` | 3.0, 0.1144 | handoff Eq. 9 |
| `along_branch_avg_um` | 0.5 (on) | handoff Eq. 9 "optional"; the table absorbs it if phantoms use the same value |
| `bbar_region` | `"block_masked"`: node's block in plane $k^*_i$, pixels within Allen radius + 1 µm of any SWC segment masked | D-018 (a), assistant's recommendation |
| `sigma_fit_um` | 0.099 | procedure §3.3 heuristic budget; bracket handoff Eq. 12. **[2026-10-06]** a study axis $\{0.080, 0.099, 0.125\}$ µm, deliverable 0.099 (D-023) |
| `mu_mode` | `"per_node"`; `"shared"` as labelled comparison | D-019 (a): two-stage route, stage 1 |
| fit bounds | $d\in[0.05, 6]$ µm, $\mu\in[0, 20]$ µm⁻¹, $v_0\in[-1,1]$ µm; multi-start $d_0\times\{0.7,1,1.4\}$ | assistant |
| selection $\mathcal S$ | converged, no parameter at a bound, registration verdict "branch found" (the actual labels of `registration_check` are **not verified**), not steep ($\tan\varphi<0.58$, old tool's `STEEP_TAN`), not faint, no second dip in the window, not at the stack edge | procedure §3.2 (definition); flag details assistant. **[corrected 2026-10-06]** no tilt criterion (D-024 (iv)); dark flag $\hat\alpha>1.0$ added (D-024 (ii)). **[corrected 2026-10-07, v1.5]** $\hat\alpha\le1.0$ removed again (D-030): $\mathcal S$ has no optical-depth criterion |
| dark-tube flag | $\hat\alpha_i=\hat\mu_i\hat d_i/\cos\varphi_i>1.0$ | **new**, from "Findings" (4). The partition breaks between true $\mu d$ = 0.5 and 1.5 in the ray world (flat $d=1$ µm), where the fit's $\hat\alpha$ is 0.57 and 1.61. Threshold to be set on data. **[corrected 2026-10-07, v1.5]** A label only (column `dark`): it no longer removes nodes from $\mathcal S$ or triggers a fill (D-030). |

**Correction (procedure §3.9; handoff step 6)**

| Parameter | Provisional default | Source / status |
|---|---|---|
| inversion | `brentq` on $\hat m(\cdot,\varphi_i)$; interpolation linear in $\varphi$, linear in $\log d$ | procedure §3.9 |
| `bias_flag` | $\lvert\hat b-1\rvert>0.2$ | D5 (confirm) |
| `max_failure_rate` | 0.2 per grid point | assistant |
| fill | neighbours on the same branch; Allen radius if the whole stretch is flagged | design handoff |
| `median_window_nodes` | 3 | design handoff |

**Renderer (path B)**

| Parameter | Provisional default | Source / status |
|---|---|---|
| `kernel_family` | `"gaussian_table"`; `"empirical"` later | procedure §3.4 |
| `kernel_table` ($\delta$ → $\sigma_{\rm r}$, µm) | 0 → 0.080, 0.14 → 0.086, 0.28 → 0.122, 0.42 → 0.262, 0.56 → 0.438, 0.84 → 0.603 (ideal Debye, 550 nm; **not calibrated**) | mathematics §3.4; replaced by block 10 |
| `kernel_continuation` | linear beyond 0.84 µm with slope $\gamma=0.79$. All options are one family, $\sigma_{\rm r}(\delta)=\sigma_{\rm r}(0.84)+\gamma(\lvert\delta\rvert-0.84)$: `"frozen"` ($\gamma=0$) and `"proportional"` ($\gamma=\sigma_{\rm r}(0.84)/0.84$, i.e. 0.72 with the illustrative table) are procedure §3.10's two truncation rules, and proportional is also the rule of `checks/stack_geometry_check.py`; $\gamma$ = 0.79 and 1.21 are the textbook slopes | mathematics Eq. 17; procedure §3.10; see "Findings". **[2026-10-06]** 0.79 "for now" (D-024 (i)); to be measured on the thick calibration set (D-026) |
| `sigma_r0_um` | 0.080 (configured; never identified by the stacks) | procedure §3.4 |
| `absorption` | `"partition_vertical"` (procedure Eqs. 5-6); options `"linear"` (mathematics Eq. 11), `"ray_world"` (S6, alternative generator) | procedure §3.6; "Findings" |
| `light_direction` | +1 = light travels toward increasing plane index | **not verified**; matters only through the partition (dark tubes) |
| `h_g_um` (fine grid) | $p_{\rm x}/16$ for $d\le0.5$ µm, else $p_{\rm x}/8$ | "Findings" (convergence) |
| `dzeta_um` (slab thickness) | 0.02 | procedure §3.6 ($\le0.05$); convergence test |
| `U_um` (phantom half-length along $u$) | 8.0, cut at $\lvert u\rvert\le U$, tube drawn into the padding | "Findings"; convergence test, essential above 30°. **[corrected 2026-10-06]** 10 µm (D-024 (iii)) |
| `cross_section_aspect` | 1.0 (round) | procedure §3.5; squashed as a check |
| background $B$ | from real block medians | **needs real data** |
| camera chain | block-average to $p_{\rm x}$; black level and grey mapping **unknown**; Gaussian noise matched to the real background SD **after** the chain; 8-bit; JPEG with Allen's own tables read from fetched crops (`Image.open(f).quantization`) | procedure §3.6 step 6; Pillow API verified 2026-10-06. **[2026-10-06]** the noise is measured on clean background patches (D-025) |

**Phantoms and table (procedure §3.5, §3.8)**

| Parameter | Provisional default | Source / status |
|---|---|---|
| `grid_d_um` | 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0 | procedure §3.5 (proposal) |
| `grid_phi_deg` | 0, 5, 10, 15, 20, 25, 30, 40, 50, 60 | procedure §3.5 (proposal). **[corrected 2026-10-06]** both grids superseded by a random design over ranges, $\varphi\in[0°,90°)$ (D-024 (iii)) |
| phantom $\mu$ | matched to the real $\hat\mu_i$ distribution through the fitted statistic; until real data: 0.6 and 3.0 µm⁻¹ | procedure §3.5; D-019 test values |
| nuisances $\xi$ | $\theta\sim U[0,\pi)$; lateral offset uniform within a pixel; axis depth uniform on $[-\Delta z/2,\Delta z/2]$; noise per camera model | procedure §3.5 |
| `n_pilot`, `eps_mc` | 50 at a few grid corners; $N_{\mathcal S}=(\hat\tau/0.005)^2$ | procedure §3.8 (both "e.g.") |
| estimator | mean (procedure Eq. 2); median as labelled comparison | procedure §3.8 (open) |
| table file | `npz` with $\hat b$, $\hat\tau$, $N_{\mathcal S}$, failure rate, plus the $\mathcal C$ dictionary as JSON; correction refuses a mismatched $\mathcal C$ | procedure §3.11 |

**Kernel calibration (procedure §3.4)**

| Parameter | Provisional default | Source / status |
|---|---|---|
| node selection | $\hat d\lesssim0.3$ µm, faint, $\varphi\lesssim10°$, isolated | procedure §3.4. **[2026-10-06]** plus a thick set: Allen diameter just under 0.8 µm, faint (D-026); both sets flat ($\hat\varphi_i\approx0$) at first (D-027) |
| offsets | ±3 planes (±0.84 µm); trust beyond ±2 only after a residual check | procedure §3.4. **[2026-10-06]** thick set: planes beyond 0.84 µm (D-026) |
| statistic $\omega_{i,k}$ | squared Gaussian-core width of the dip (windowed $V$ as cross-check) | procedure §3.4 [corrected 2026-10-04] |
| origin convention | symmetric $\Delta\sigma^2(\delta)=\Delta\sigma^2(-\delta)$; signs fitted separately as a check | mathematics §3.5 |

## Findings of the theory chat that the code must accommodate

All are **[run]**. The scripts are in `checks/` (added with this file); run them from inside `checks/`. Their outputs, identical to the original runs, are in `checks/<script>.out`.

Common conditions, unless a row says otherwise:
- illustrative kernel: the ideal-Debye core table of mathematics §3.4, continued beyond 0.84 µm as each row states;
- noise-free, no camera chain: profiles are evaluated at the pipeline's sample points, with no pixel integration, noise, 8-bit or JPEG;
- fits by (D-019.1) with $\sigma_{\rm fit}=0.080$ µm and $\bar B=B$ (an oracle background).

These numbers show mechanisms and sensitivities. They are **not** values of $b$.

| Finding | Evidence | What the code must do |
|---|---|---|
| (S1)-(S4) are exact | 3-D membership vs (S3): 0 mismatches over 20 random tubes × 4000 rays. Slab chords vs brute force: max error 2.46e-6 µm, which is the brute-force sampling step. $\sum_j a_j-\mu\ell$: 1.4e-13. (`stack_geometry_check.py`) | use them as block-2 oracles |
| The heading matters only through the pixel grid | Same finite tube ($r=0.25$ µm, $\varphi=20°$, $\mu=1$ µm⁻¹, $U=3.5$ µm) at $\theta$ = 0/30/45°. Fine-grid profiles differ by at most 0.61% of the dip at $p_{\rm x}/8$ and 0.07% at $p_{\rm x}/16$ (discretisation). After pixel integration they differ by 2.6-3.0% of the dip, and refining the grid does not remove it: 3.04% at $p_{\rm x}/8$, 2.56% at $p_{\rm x}/16$ ($\theta=30°$). (`stack_geometry_check.py`) | draw $\theta$; check $\hat b$ binned by $\theta$ (procedure §3.10) |
| The fine grid $p_{\rm x}/8$ is not converged for thin tubes | Same tube, $\theta=0$: $p_{\rm x}/8$ and $p_{\rm x}/16$ differ by 2.56e-3 $B$, 0.7% of the dip | grid default above; convergence test |
| Phantom length matters at steep tilt | Setup: $r=0.25$ µm, $\mu=1$ µm⁻¹, node plane; kernel continued in proportion beyond 0.84 µm (slope 0.72); $h_{\rm g}=p_{\rm x}/4$; direct sum. Dip at the node for $U$ = 2 → 15 µm: 0.3910 → 0.3914 at 20°, 0.5319 → 0.5528 at 45°, 0.5569 → 0.5736 at 60°. A flank haze at $v$ = 2.5 µm appears (0.0144 / 0.0220 $B$ at 45° / 60°, $U$ = 15 µm) and is still growing at 15 µm. (`stack_geometry_check.py`) | `U_um` as config; cut by $\lvert u\rvert\le U$, never by the block |
| The kernel's far field moves $\hat d$ for steep or thick tubes | Setup: $\mu=1$ µm⁻¹, $U=6$ µm, profile in the plane through the node's axis depth. $\hat d/d$ with continuation $\gamma$ = 0 (frozen) / 0.79 / 1.21: <br>$d$ = 0.5 µm: 1.041 for all three at 0°; 1.062 / 1.063 / 1.066 at 20°; 1.266 / 1.334 / 1.323 at 45°; 1.444 / 1.464 / 1.415 at 60°. <br>$d$ = 2 µm: 1.126 / 1.128 / 1.130 at 0°; 1.138 / 1.154 / 1.166 at 20°. <br>Grid convergence checked at (0.5 µm, 45°) and (2 µm, 20°) under $\gamma=0.79$. (`optics_points_check.py A`, `followup_checks.py`) | continuation rule as config. Run procedure §3.10's truncation check as written: frozen ($\gamma=0$) and proportional ($\gamma=\sigma_{\rm r}(0.84)/0.84\approx0.72$), and flag where $\hat b$ moves. Proposed addition: $\gamma$ = 0.79 and 1.21, the textbook far-field slopes for the full NA 1.4 cone (mathematics Eq. 17). **[corrected 2026-10-06]** v1 said "with $\gamma\in[0.79,1.21]$, and frozen as a stress test", as if the procedure named those slopes. |
| The focus-search planes depend on the extrapolated kernel | Flat $d=2$ µm, $\mu=1$ µm⁻¹, centre dip (frozen / 0.79 / 1.21): 0.76 / 0.62 / 0.57 at $z_k=c_z+3\Delta z$ (away from the light), 0.86 / 0.82 / 0.80 at $c_z-3\Delta z$. At $d=0.5$ µm the three rules differ by at most 0.01. (`optics_points_check.py A`) | compare $k^*$ between real and synthetic stacks |
| Thin calibration nodes fade fast | $d=0.3$ µm, $\mu=0.6$ µm⁻¹, flat, planes on the light side, $\gamma=0.79$ beyond 3 planes: dip ÷ in-focus dip = 0.75, 0.29, 0.19, 0.14, 0.11 at 1-5 planes (`optics_points_check.py A`) | calibration window ±3 planes; read the real noise first |
| Faint thick nodes reach further; what a calibrated range covers (added v1.2) | Partition renderer, flat tube, planes on the light side, noise-free; reference = the centre dip of a 0.3 µm node at $\mu=0.6$ µm⁻¹, 3 planes out (0.0258 $B$, slope 0.79). A 0.8 µm node at $\mu=0.6$ µm⁻¹ stays at or above it out to plane 16 (4.48 µm, $\sigma_{\rm r}$ 3.48 µm) with $\gamma=0.79$ and plane 11 (3.08 µm, 3.31 µm) with $\gamma=1.21$; at $\mu=0.3$ µm⁻¹, planes 9 and 7. By (S5) with $U=10$ µm, the plane through the axis of a 0.5 µm phantom needs kernels beyond 0.84 µm from $\varphi\approx3.4°$ on; a calibrated range of 3.08-4.48 µm covers it up to 15.7-22.8°. (`calib_reach_check.py`, parts 1-2) | D-026: thick calibration set, windows of about $\pm3\sigma_{\rm r}$; steeper phantoms still use the extrapolated kernel |
| Tilt moves the second moment much more than the core width (added v1.3) | Thin node ($d=0.3$ µm, $\mu=0.6$ µm⁻¹), depth-only Gaussian kernel, planes 1-9 on the light side. Tilted / flat width: square root of the second moment over $\pm5\sigma_{\rm r}$ +4-5 % at 10°, +17-27 % at 20°, +32-59 % at 30°; Gaussian-core width within 1 %, 3 %, 6 %. Same profile (plane 1, 30°): second-moment ratio 1.32 over ±0.61 µm, 1.78 over ±1.0 µm; core ratio 1.054 over both. A 0.5 µm node at $\mu=0.6$ µm⁻¹ reaches 1.96 µm (slope 0.79) or 1.68 µm (1.21) in the reach test above. (`calib_width_check.py`; `calib_reach_check.py`) | D-027: calibrate on flat nodes, use the core width, compare tilted nodes with the renderer, not with flat numbers |
| **The partition fails qualitatively for dark, thick tubes** | **(1) In-focus centre gets lighter as the stain darkens.** Debye kernel ($\gamma=0.79$), flat $d=1$ µm: centre dip 0.831 / 0.788 / 0.723 at $\mu d$ = 3 / 10 / 50, while a vertical ray through the centre absorbs 0.950-1.000. For $d=2$ µm: 0.805 / 0.785 / 0.763. For $d=0.5$ µm it rises to 0.931 at $\mu d=10$, then falls to 0.910 at 50. (`followup_checks.py` §4) <br>**(2) The focus curve peaks toward the light.** Peak at plane −1 for $d=1$ µm, $\mu d=1.5$; at −3 for $d=2$ µm, $\mu d=2$; at 0 for the faint $d=2$ µm, $\mu d=0.2$. (`focus_scan.py`, ±6-plane scan, negative = light side) <br>**(3) The ray world shows neither artefact.** Its truth is symmetric: the dips at ±3Δz are equal to four decimals in every case run. With the true history, the opaque in-focus centre is black (Gv* 1.000 at $\mu d=50$). The partition's centre dip at $\mu d=50$ is 0.699, against 0.691 from (S9). (`optics_points_check.py BC, matched`; `followup_checks.py`; `focus_scan.py`) <br>**(4) $\hat\mu$ becomes unreachable.** With the kernel of the same rays and $\mu$ matched through $\hat\mu$ (procedure §3.5), no $\mu_{\rm ph}\in[0.5\mu,8\mu]$ lets the partition renderer reach the truth's $\hat\mu$ at $\mu d$ = 1.5 or 3 ($d=1$ µm, flat). It does reach it at $\mu d\le0.5$. (`optics_points_check.py matched`) | dark-tube flag; `absorption` and `light_direction` as config; ray-world generator for the end-to-end check. **[corrected 2026-10-07, v1.5]** the flag labels dark nodes, it does not exclude them (D-030), so this failure now enters their corrected diameters |
| Faint stain is fine; vertical rays only rescale $\mu$ | Ray world, $\mu$ matched through $\hat\mu$. <br>Partition renderer: $\mu_{\rm ph}/\mu$ = 1.450 at $\mu d=0.05$, against 1.447 from (S10). Its $\hat d/d$ differs from the truth's by −0.002 there and by −0.005 at $\mu d=0.5$. <br>Same rays with the true history but vertical path elements: +0.009 / +0.021 / +0.050 at $\mu d$ = 0.5 / 1.5 / 3 ($d=1$ µm, flat), and +0.026 at $d=0.5$ µm, 20°, $\mu d=0.5$. <br>The ray world's own $\hat d/d$ (1.20-1.32) is not a $b$: the ray world has no diffraction. (`optics_points_check.py matched`) | report $\hat\mu$ as effective; phantom $\mu$ matched via $\hat\mu$ |
| Coherence is not modelled anywhere | optics §3.6; the ray world has no coherence either | listed as a gap; nothing to code now |

## Validation (procedure §3.10) and where each check will live

| Check | Script (proposed) | Needs |
|---|---|---|
| consistency of the blurs (first end-to-end gate) | `tests/smoke/test_smoke_node_pipeline.py` | sandbox |
| $\sigma_{\rm r}(0)$ / $\lambda$ (0.066, 0.095 µm) | `scripts/build_table.py --reduced` | sandbox or Colab |
| $\sigma_{\rm fit}$ bracket | `scripts/build_table.py --reduced` + real refit | Colab |
| kernel shape (Gaussian vs empirical) | `scripts/build_table.py --kernel empirical` | after block 10 |
| truncation: frozen and proportional as in the procedure, plus $\gamma$ = 0.79, 1.21 (proposed); $k^*$ agreement. **[corrected 2026-10-06]** v1 listed only "$\gamma$ rules, frozen". | `scripts/build_table.py --continuation ...` | sandbox + real $k^*$ |
| above/below, depth dependence | `scripts/calibrate_kernel.py --split-sign --by-depth` | Colab |
| cross-section aspect | `scripts/build_table.py --aspect 0.6,1.0` | sandbox |
| $\mu$ at the 10th and 90th percentile | `scripts/build_table.py --mu ...` | real $\hat\mu_i$ |
| heading | table binned by $\theta$ | sandbox |
| real vs synthetic profiles | `scripts/compare_profiles.py` | Colab |
| end-to-end with an independent generator | `scripts/end_to_end.py --generator ray_world` | sandbox |

## Known gaps (facts nobody has yet)

- The 2026-09-23 image modules are not in the repo or on Drive (see Provenance).
- Allen's camera chain: black level, grey mapping, noise after JPEG, JPEG tables. The tables are readable from any fetched crop; the rest needs real blocks.
- Mounting medium and its index (optics §3.7); aperture-diaphragm setting, which sets the cone width and the coherence (optics §3.6); the light direction relative to Allen's plane index (listed as unverified in the 2026-10-05 handoff, "Open choices").
- Real $\hat\mu_i$ distribution (phantom $\mu$); uniformity of the stain along branches (D-019 (a)).
- Real noise level, and therefore the usable calibration range in planes. **[2026-10-06]** To be measured on clean background patches by D-025's Phase-II script (not yet written).
- Cell 13 (registration on node 4505 and on 5-10 stretches) has not been run (design handoff Next actions 1-3), so the snap radius is unknown.
- Older tools named in the design handoff (`allen_stack_radius_refit.py`, `allen_projection_radius_refit.py`, `README_stack_refit.md`): exist per that handoff; location **not checked**.

## First actions in the implementation chat

1. Clone and read:
   ```bash
   git clone https://github.com/Leonardodm00/Towards-EEG.git towards-eeg
   cd towards-eeg && git log -5 --format='%h %ci %s' -- "Passive Features/Diameter Re-measurement"
   ```
   Read this file, then the design handoff (§Method, §Math reference), procedure §3.3-3.11, and D-018/D-019 in the project log (root copy).
2. Load the `scientific-coding` and `project-decision-log` skills.
3. **Get the 2026-09-23 modules from the user** (attachments in the chat, or the design chat's delivered files). Verify them against the design handoff's table before committing:
   ```bash
   for f in allen_image_io.py allen_image_align.py allen_image_measure.py allen_image_plot.py smoke_allen_image.py robustness_registration.py; do printf "%s %s\n" "$(sha256sum $f | cut -c1-12)" $f; done
   # expect io dae5d24f7f2c, align 14dd899057a8, measure 4614cd949ba1, plot 3b9f1a07e544, smoke d7811c681b0f, robustness 18059c7109ee
   python smoke_allen_image.py          # expect 20/20 passed
   python robustness_registration.py    # expect 40/40 correct (takes > 2 min)
   ```
4. **Confirm in one batch** (AskUserQuestion; at most 4 questions per call, so two calls):
   - the layout;
   - the Colab bootstrap;
   - `mu_mode`, `bbar_region`, `focus_bg_rule`;
   - `sigma_fit_um`;
   - `absorption` and `kernel_continuation`;
   - `U_um` and the grid;
   - the dark-tube flag;
   - D5 and D7.

   Log each answer as a decision in the same turn.
5. Branch and scaffold:
   ```bash
   git checkout -b sci/diameter-pipeline
   ```
   Create `specs/SPEC.md` from the skill's template, entries `drafted`. The 2026-09-23 modules get entries marked `Confirmed: inferred`.
6. Blocks 0 → 9 in order, each with its smoke test, two-step verification and a SPEC entry. Push only when the user agrees, and offer to push at the end of each session.
7. Hand the user Colab cells for Phase II (block 11) once blocks 0-8 pass on synthetic stacks.

## Working rules (unchanged from earlier handoffs, plus the coding directives)

- **Answers:** brief and plain by default; full notation when maths is asked for; confirmation questions are claims to check; explanation modes A/B/C (v8).
- **Sources:** KB → PubMed (full text only for claims; no numbers from abstracts) → bioRxiv → data repositories (metadata is not data).
- **Code** follows the `scientific-coding` skill:
  - clarify before logic (via the D-021 batch);
  - libraries first (NumPy, SciPy `least_squares`, `brentq`, `ndimage`, `numpy.fft`; Pillow for JPEG);
  - loading, model, analysis and plotting kept separate;
  - parameters only in `config.py`;
  - randomness from a passed `numpy.random.Generator`;
  - one smoke test per new algorithm, with run instructions;
  - every script checked twice (re-read against SPEC, then executed);
  - code in blocks, never one dump.
- **Real data only in Colab.** The sandbox gets 403 from api.brain-map.org; `SyntheticFetcher` (2026-09-23) and the renderer are the offline inputs.
- **Docs (D-020):** method docs are changed by the theory chat; the implementation chat writes `specs/` and code docs. Every reply ends with one line naming the docs written.

## Admin left open

- **Project knowledge is full** (1,997,337 of 2,000,000 bytes, 2026-10-06). D-020 and D-021 cannot be appended to the project log until space is freed. Deletion needs the user's explicit OK. Candidates: the `claude/` copies of the procedure, mathematics and optics notes (stale per D-020), and `claude/TEEG_decisions_and_ideas_log.md`.
  **[corrected 2026-10-06, v1.2]** Resolved for D-020 to D-024: the user approved deleting those `claude/` copies, and the log was written. **Open again:** with project knowledge at 1,954,553 of 2,000,000 (`project_info`, 2026-10-06), the project refused the log with D-025, D-026 and I-002 (55,308 tokens). Any rewrite of the log, even unchanged, appears to exceed the free space **[inferred from those two numbers]**, so space must be freed first, with the user's OK. Until then the entries wait in `docs/TEEG_decision_log_additions_2026-10-06.md`.
  **[corrected 2026-10-07, v1.4]** Closed: D-025 to D-027 and I-002 were merged into the project log on 2026-10-07 (13:48 UTC), after the Ih Fit chat had rewritten the log the same day. The inference above was not borne out; whether space was freed in between was not checked.
- **[corrected 2026-10-06]** The 2026-10-05 handoff says D-018 and D-019 are in `claude/TEEG_decisions_and_ideas_log.md`. They are not: that copy holds only D-001 and I-001 (last change 2026-09-19). The live log with D-001 to D-019 is the root-level `TEEG_decisions_and_ideas_log.md`, so it must **not** be deleted, although the 2026-10-05 handoff lists it among the deletion candidates. D-020 and D-021 belong in it. The 2026-10-05 handoff carries the same correction marks as of this commit.
- The theory chat offered (2026-10-06) to push corrections to procedure §3.6 step 3 and mathematics §3.2, §3.5:
  - the light-direction dependence and the dark-node axis shift are artefacts of the partition, not physics in the ray limit;
  - add (S6)-(S9);
  - $\hat\mu$ is an effective coefficient.

  These are awaiting the user's OK. Until then, "Findings" above is the reference.
  **[2026-10-06, v1.2]** Also offered, also awaiting the user's OK: D-025 and D-026 into procedure §3.4 (calibration) and §3.6 step 6 (camera-chain noise).
