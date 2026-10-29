# TEEG decisions and ideas log

Living record of the modelling and implementation choices made in the
Towards-EEG project, and of ideas for parts not yet built. One entry per
decision; a decision is never edited in place — a later entry supersedes it
and both cross-reference each other. Retrieve by ID (`D-nnn`, `I-nnn`) or by
the "Binds" column.

| Date | Change |
|---|---|
| 2026-09-19 | Created. Logs D-001 (synaptic conductances are shaft-referenced and model-conditioned). Earlier session decisions (deliverable F variant, cap policy, fallback policy, three-vote defaults) are tabled in `TEEG_HPC_coding_handoff_2026-09-19.md` section 6 and are not yet backfilled here. |
| 2026-09-21 | Logs D-002 (morphological QC is a downstream filter, never a P1 gate) and D-003 (L2/L3 interneurons exported under both the SST and the PV/VIP passive set, into separate trees; choice downstream). Both were taken on the cluster's first real P1 runs; the handoff's section 6 carries the code consequences. |
| 2026-09-22 | Logs D-004 (one submission runs P1 -> P2 -> P3 for every cell, and the spine area of every cell is the mesh_beyond measurement). Taken after the arbour figure was found reporting the skeleton bracket as though it were the deliverable. The orchestration this requires does not exist yet; `TEEG_campaign_single_submission_handoff_2026-09-22.md` specifies it. |
| 2026-09-22 (later) | Logs D-005 (I_h in the passive fitter: six fitted parameters $(C_m, R_m, R_a, \bar g_h, \Delta v_h, \kappa_\tau)$, uniform density, analytic $e_{\rm pas}$ balance, GP 200/100) and D-006 (campaign protocol: no I_h-suppressing guards, full step window, depolarising validation, Phase 3 kept, run B as baseline, edited monolith copy in a new folder). Both taken as answers to the 14 questions of `TEEG_Ih_fit_staged_plan.md` v0.1; the v0.2 plan carries them. Five confirmations (C1--C5) still open. |
| 2026-09-22 (later, 2) | Logs D-007 (confirmations C1--C5 of D-005/D-006; every recorded long-square amplitude admitted). Stage 0--1 implementation of `TEEG_Ih_fit_staged_plan.md` started the same afternoon. |
| 2026-09-22 (later, 3) | Sign of $\Delta v_{\rm base}$ for the Kalmbach alternative corrected in D-005 and D-007 from $-20$ to $+20$ mV under the (D-005.1) convention; the decision itself (Kalmbach's shifted Kole as the labelled alternative) is unchanged. Caught by smoke check S5 of `Ih Fit/smoke_ih_fit.py` on the first build. Stage 0 harness PASS and Stage 1 smoke 6/6 in the assistant's sandbox. |
| 2026-09-22 (later, 4) | Logs D-008 (two implementation conventions of the I_h refactor). Stages 2-4 delivered: smoke 9/9, regression still bit-for-bit. |
| 2026-09-22 (later, 5) | Logs D-009 (the H01 campaign: the kappa fill stays and the uncovered fraction is reported rather than thresholded; `shaft_terminates` components are spines; the allocation is 200 CPUs at once with a 300 h walltime ceiling). **D-004's orchestration was implemented the same afternoon** -- `campaign.pbs`, `campaign_join.py`, `campaign_policy_report.py`, `campaign_queue_check.py`, smoke suite 9/9 in the assistant's replica, byte-verified into the laptop clone, pending the user's commit/push/pull. D-004's index status updated accordingly. This entry is D-009 and not D-008 because the I_h chat took D-008 for its own decision 24 minutes earlier; the campaign handoff's section 11 was renumbered to match. |
| 2026-09-22 (later, 6) | Logs **D-010** (two departures of the I_h orchestrator from `run_biological_fit.py` -- no per-cell $\tau_w$ sweep, and a validation gate windowed like the training side for the I_h arms -- plus the rule that no liveness check may certify a loss by `isfinite` alone). **Stages 5 and 6 delivered**: Phase 3 generalised to the `ParamSpec`, `run_ih_fit.py` and `submit_ih_fit.sh` written, smoke **11/11** and the regression gate still bit-for-bit, all re-verified from a clean extract of the delivery tarball (`nrnivmodl` from scratch, `bash -n`, submit dry-run). D-005, D-006, D-007 and D-008 index statuses updated from "NOT yet implemented" to implemented through Stage 6. The rule in D-010 (iii) came out of a defect in the assistant's own smoke suite, found by measurement. |
| 2026-09-22 (later, 7) | Logs **D-011** (Stage 7's cohort design and its two error measures: real archive morphologies, a one-quarter false-positive control with no I_h, run B's measured noise level, a log ratio for the five positive axes and a difference in mV for $\Delta v_h$, the I_h axes summarised over the I_h cells only, railed axes counted and never silently dropped). **Stage 7 delivered**: `synth_gt_grid.py` and `gen_from_manifest.py` extended, `ih_recovery_report.py` and `run_ih_recovery.py` written, `submit_ih_recovery.sh`, `smoke_ih_recovery.py` **9/9**, `smoke_ih_fit.py` still 11/11 and the regression gate still bit-for-bit, all re-verified from a clean extract. The gate itself has NOT been run: this records how it will be measured. Two defects found while writing its suite -- the manifest on disk was not canonical across re-runs, and the suite's own `except Exception` did not catch the `SystemExit` argparse raises, so a dead check printed no FAIL line. |
| 2026-09-23 | **Paths corrected in the I_h submit scripts, no decision changed.** Both had inherited `CODE_DIR="/davinci-1/home/ldellamea/Human Neurons Fitting/Ih Fit"` from the legacy passive-fit script. The user gave the real locations: laptop clone `~/Towards-EEG`, cluster clone `/davinci-1/home/ldellamea/TEEG/Towards-EEG`, and this code at `<clone>/Passive Features/HPC script/Ih Fit`. `CODE_DIR` is now `${PBS_O_WORKDIR:-$(pwd)}` -- submit from the folder -- with a refusal naming the exact `cd`; this also sidesteps `qsub -v`, which is comma-separated and cannot carry a path containing a space, as that one does. `ARCHIVE_ROOT` and `OUTPUT_BASE` became `IH_`-prefixed knobs (the archives are not in git: no `specimen_*` anywhere in the repo), and a bare `CODE`/`ROOT`/`ARCHIVE_ROOT`/`OUTPUT_BASE`/`CODE_DIR` is reported and ignored, per the `H01_*` doctrine of the paths reference sections 8.2 and 8.4. **Withdrawn:** the CRLF caution of the stage-6 handover -- the repo's own `.gitattributes` already pins `*.sh`, `*.py` and `*.mod` to `eol=lf`. |
| 2026-09-23 (later) | Logs **D-012** (Stage 7's recording noise is measured PER SWEEP on the real cells, with its lag-1 correlation and the real protocol constants; run B's `noise_sigma_mV` is not used as generator input -- **supersedes D-011 (iii)**) and **D-013** (C4 closed: the diameter-corrected morphologies arrive as one SWC per specimen in the archive's own format; tests use the raw SWCs). D-012 came out of tracing run B on the HPC at the user's request: run B is `/davinci-1/home/ldellamea/Human Neurons Fitting/pipeline_outputs/dt01ls60/<GROUP>/phase2_results.csv` (run A: `dt01ls60ph25`), and its `noise_sigma_mV` turned out to be a different object from the one the generator injects. Archive root `/davinci-1/home/ldellamea/Human Neurons Fitting` confirmed by the user. `noise_calibration.py` written; `smoke_ih_recovery.py` **12/12** (R10-R12 added, R8 now end to end through the measured path), `smoke_ih_fit.py` 11/11, regression gate bit-for-bit. D-007, D-008 and D-011 statuses updated. |
| 2026-09-23 (later, 2) | Logs **D-014** (Stage 7's synthetic cells are ACQUISITION TWINS with PER-PROTOCOL noise, and AR(1) adequacy is reported; supersedes D-012 (ii) and corrects D-012's implementation). Taken on the user's 'use also the short square with exponential weights as well': checked that the fit already trains the SS pulses with their exponential weight (it does, in every arm; now asserted at run time), and gave the SS pulses their own measured noise. Found and fixed in the same pass: every cohort had been generated at one cohort-median sampling rate while each cell's rho was measured at its own (the Allen archive mixes 200 and 50 kHz); and a reloaded noise table dropped every cell (NaN is truthy). `smoke_ih_recovery.py` **13/13**, `smoke_ih_fit.py` 11/11, regression bit-for-bit. |
| 2026-09-23 (later, 3) | Logs **D-015** (user, 17:07): the injected noise never carries the loss's exponential time-weight and its fast part is stationary; every synthetic cell is sampled at its own real cell's rate; the noise object is the single sweep's total pre-stimulus fluctuation. **Confirms D-014 (i) and (ii)**, which had been the assistant's reading and decision. No code change: all three were already the defaults. One smoke check (noise stationary across the sweep) is queued for the next delivery. |
| 2026-09-23 (later, 4) | Logs **D-016** (assistant, correcting its own Stage-2 code): the depolarising-sweep spike screen follows the Allen white paper's action-potential rule (10 kHz Bessel-filtered dV/dt >= 20 mV/ms, peak >= -30 mV and >= 2 mV above the crossing within 5 ms, plus the -20 mV catch-all; levels on the archive's LJP-corrected voltage, so 14 mV stricter than the Allen's labels); the raw-difference rule it replaces fired on recording noise alone, and every synthetic twin of the first cluster run lost its depolarising validation steps to it. The loader and the role assignment now say why a sweep was dropped. Adds the Long Square pre-window structure columns to the noise table, because the same run showed AR(1) does not fit the LS noise (acf 0.566 at 1 ms against 0.000) and three different causes fit that number. **Corrects the implementation of D-006 Q7** (intent unchanged); D-014's open point stands, now with its measurement. |
| 2026-09-28 | Logs **D-017** (user, 15:39): each synthetic twin carries its own real cell's two-component noise law -- fast AR(1) + slow Ornstein-Uhlenbeck + the per-sweep offset and tilt -- fitted to that cell's own window statistics, and its real Long Square pre-window (265 ms on L3_exc, not 95 ms); no cohort-level law. Answers D-014's open point and D-016's (a) and (c). Records two corrections: the LS pre-window is 265 ms in every L3_exc cell (D-016 had framed it as 0.5 s or 1 s), and D-011 quoted Moradi Chameh's L2&3 rebound depolarisation ($1.8 \pm 1.6$ mV) as the sag ($1.7 \pm 1.1$ mV). Open before any code: the two drifting cells (keep or exclude), one law for both protocols, and where drift is calibrated. |
| 2026-09-28 (later) | D-017 annotated, no decision changed: the lineage of its noise-law fit (Zeraati, Engel & Levina 2022; indirect inference) is recorded, and a Monte Carlo check shows that the closed-form shortcut used for the cohort fit overshoots the autocorrelation statistics by 0.02-0.07 at our window lengths. Refitted by simulated medians, the cohort law keeps $\tau_s \approx 37$ ms while $\sigma_s$ rises from 0.051 to 0.056 mV. D-017's open point (d) now leans to simulation. |
| 2026-09-28 (later, 2) | D-017 annotated again, no decision changed: the topic document `claude/TEEG_Stage7_noise_model_2026-09-28.md` was written and checked twice by independent agents. Recorded against D-017's text: SS is the Allen *Square Subthreshold* protocol, not *Short Square*; Why (i)'s reading of the quieter first segment and the electrode inference of "What it implies" were too strong; the two high-trend cells are drift candidates (526785799 favours a creep, not decisively; 571430815 is compatible with its own slow noise); the status note's "eleven" and "1-5 %" are ten and 1-3 %. Added for open point (c): the Allen per-sweep `vm_delta_mv` decides creep versus wander for 526785799 but only at large values for 571430815. |
| 2026-09-30 | Logs **D-018** (user, 16:16): in the diameter re-measurement workstream (Allen 63x stacks, feeding the corrected SWCs of D-013), the background level $B$ of the blurred-tube Beer-Lambert fit is FIXED to a median grey level of the focal plane and no longer fitted; $(d, \alpha, v_0)$ stay free and $\sigma$ stays fixed. Amends local decision D4 of that workstream's handoff (`handoff_diameter_remeasurement.md`, uploaded by the user in chat, not in project knowledge, not edited). Open: the median's region (whole plane or the node's block with the traced path masked -- the assistant recommends the block) and its stained-fraction bias. Its status note records a correction to that handoff's optics line (0.24 µm is the Airy first zero, not the FWHM) **[run]**. |
| 2026-10-01 | D-017 annotated a third time, no decision changed. **[user]** The Allen per-sweep QC query and a measurement of the archive's own Long Square sweeps go in one diagnostic: `Ih Fit/ls_baseline_qc.py` with `smoke_ls_baseline_qc.py` (10/10 on a fresh clone of `main` at `2fd7643`), delivered as a patch, not yet pushed or run. It answers note 2's "post-step length not verified" for the archive and gives open points (a) and (c) a signed, per-sweep test of creep against wander. Recorded with it: `vm_delta_mv` is unsigned, and the signed Allen change is `post_vm_mv - pre_vm_mv` (IPFX source). Handoff for the next chat: `claude/TEEG_Stage7_noise_handoff_2026-10-01.md`. Written twice the same turn: the second write removed from note 3 a stimulus amplitude read through a summarising fetch, and added that sign agreement alone is weak evidence against a slow wander. |
| 2026-10-01 (later) | D-017 annotated a fourth time, no decision changed: `ls_baseline_qc.py` was run on L3_exc (commit `3dfa263`, the delivered patch byte for byte). Archive half: in none of the 18 cells does the pre-window trend continue over the 6.89 s to the end of the stored sweep (sign agreement 53/104); 526785799 and 571430815 are slow wander, not creep. The baseline does change over that span by more than the cohort law allows, in every cell, so a slower component exists whose shape the SS pre-pulse series has to settle. Allen half not run (network). Note 3's remark that sign agreement is weak evidence is corrected by Monte Carlo. Evidence in status note 4. |
| 2026-10-01 (later, 2) | Logs **D-019** (user, 18:26): in the diameter fit (handoff Eq. 11, as amended by D-018) the darkness parameter is the stain's absorption coefficient $\mu$ (µm$^{-1}$), not the centre-line absorbance $\alpha$; the tilt $\varphi_i$ is not fitted but taken from the local line fit (handoff Eq. 5), so $\alpha = \mu d/\cos\varphi_i$ inside the model. Recorded with it **[reasoning; run]**: with $\mu$ free at every node the fit is a reparameterisation of D-018.2 and returns the same $\hat d_i$; $\mu$ changes $\hat d_i$ only if it is shared across nodes, which makes thin nodes robust to an error in $\sigma$ and sensitive to non-uniform stain instead. Per node or shared: open. Merged onto the project copy that carried this day's D-017 notes 3 and 4. |
| 2026-10-06 | Logs **D-020** (user, 2026-10-04 12:47; appended late because project knowledge was full): the documents of the diameter re-measurement workstream live in the Towards-EEG repository, `Passive Features/Diameter Re-measurement/docs/`, and are kept current there by commit and push; the assistant reads and cites them from the repository. Taken after the project reached its size limit and refused the optics document. Logs **D-021** (user, 2026-10-06 14:30): the whole diameter-estimation pipeline is implemented now, in a separate chat, in parallel with the theory study; every unsettled method choice enters the code as a named configuration parameter with a labelled provisional default, confirmed in one batch. Full text in `Passive Features/Diameter Re-measurement/docs/TEEG_diameter_implementation_handoff_2026-10-06.md`. Space was freed by deleting, with the user's OK, the stale `claude/` copies of the procedure, mathematics and optics-notes documents (the optics notes byte-identical to the repo copy; the other two are the v1 texts that the repo's v1.1/v1.2 contain unchanged) and `claude/TEEG_decisions_and_ideas_log.md` (D-001 and I-001 only, both in this file). |
| 2026-10-06 (later) | Logs **D-022** (user, ~15:05, the D-021 batch, part 1): where the code lives (workstream folder plus a root-level `specs/SPEC.md` whose Coverage names the folder), the 2026-09-23 image modules imported byte-identical (hashes verified; their suites 20/20 and 40/40 re-run), the production bias table on davinci via PBS. Logs **D-023** (user, ~15:08, part 2): $\mu$ per node (closes D-019 (a)); $\bar B_i$ over the node's block with the traced path masked (closes D-018 (a)); the focus score keeps its profile-ends median (closes D-018 (c)); $\sigma_{\rm fit}$ is a study axis -- tables, real-node fits and corrections are run at several in-focus blur values. Logs **D-024** (user, ~15:10 and ~15:12, parts 3-4): renderer = absorbed-light partition along vertical rays and linear kernel continuation with $\gamma = 0.79$, both "for now", the user studying both; dark-tube flag at $\hat\alpha > 1.0$; the phantom design is a random search over ranges, not a grid ($U = 10$ µm; $d$ on a temporary range until the Allen radius distribution is read; $\varphi$ over all tilts), with a smoothing thin-plate spline as the response estimator; every node gets a diameter whatever its tilt (the old 30° steep exclusion is withdrawn); D5 and D7 confirmed; node-position jitter a config knob at 0 until cell 13. |
| 2026-10-07 | Logs **D-028** (user, 09:41): the sample autocorrelation in `Ih Fit/` is the textbook estimator (both sums over the same $n$), computed by `statsmodels.tsa.stattools.acf` rather than by hand-written code; supersedes the "fitter's convention" (lag-$L$ sum over $n-L$ pairs) of D-012.1, D-014 and D-017. Taken after the user asked why the hand-written form had been used. Found while checking: the old function's reason for avoiding statsmodels was wrong (`adjusted=True` reproduces it to $4\times10^{-16}$), and the fitter's residual lag-1 estimate was already textbook while its noise lag-1 estimate was not. Change plan `claude/TEEG_acf_textbook_change_plan_2026-10-07.md`; nothing coded. D-012, D-014, D-017 index statuses annotated. |
| 2026-10-07 (later) | **Renumbered, no content changed:** the autocorrelation decision written at 09:47 as D-025 is **D-028**, because D-025, D-026, D-027 and I-002 were already taken by the diameter workstream and live in the repo file `Passive Features/Diameter Re-measurement/docs/TEEG_decision_log_additions_2026-10-06.md` (written there on 2026-10-06 while project knowledge was full; not yet merged here). Those IDs are reserved; the next free ID after this row is D-030. Logs **D-029** (user, 09:55): D-028's five open points closed -- `Biological Fit/` unchanged, no opt-in for the adjusted form, the 10-product rule kept, the wrapper in `Ih Fit/noise_stats.py`, statsmodels to be installed in the cluster env `prova`. Implementation handoff pushed to the repo: `Passive Features/HPC script/Ih Fit/docs/TEEG_acf_textbook_implementation_handoff_2026-10-07.md`. |
| 2026-10-07 (later, 2) | D-028 and D-029 **implemented**, no decision changed: one patch on `main` at `b355d8e`, written and verified in the implementation chat and delivered for the user's commit and push. `Ih Fit/noise_stats.py` (`sample_acf`, D-029 (iv)) carries the estimator; the fitter's two lag-1 estimates and `noise_calibration.lag_autocorr` call it; new `smoke_noise_stats.py` 6/6; in the sandbox smoke_ih_fit 12/12, smoke_ih_recovery 14/14, smoke_ls_baseline_qc 10/10 and the regression gate bit for bit, the counts of `b355d8e`. Status note added to D-028; D-028 and D-029 statuses and index rows updated. Not yet on the cluster: statsmodels (>= 0.12) is still to be installed in `prova`. |
| 2026-10-07 (later, 3) | Logs, appended late, the diameter workstream's **D-025** (user, 2026-10-06 17:01): the image noise of the real stacks is measured on clean background patches -- in every plane, at several positions per plane and in several stacks -- through its second moment, and its distribution is examined; **D-026** (user, 2026-10-06 17:01): the defocus-kernel calibration is extended beyond the ±0.84 µm that thin nodes reach, using dendrites whose Allen diameter is just under 0.8 µm, read on their planes beyond 0.84 µm; the near-focus law is continued there and adjusted, the kernel stays Gaussian for now, and D-024 (i)'s configured slope $\gamma$ becomes a measured one; **D-027** (user, 2026-10-06 18:12): the calibration starts with flat branches only, near and far from focus, and tilted branches come later, compared with the renderer's prediction; and idea **I-002** (user, 2026-10-06 17:01, "to discuss further"): a non-Gaussian kernel family with the Gaussian as a special case. Their text was written on 2026-10-06 to the repo file `Passive Features/Diameter Re-measurement/docs/TEEG_decision_log_additions_2026-10-06.md` when this log could not be written, and is merged here verbatim; the IDs had been reserved by the 2026-10-07 (later) row. The assistant's comments in the entries are open points. Numbers behind them: `checks/calib_reach_check.py` and `checks/calib_width_check.py` (repo, `main`). D-024's index status and status annotated. |
| 2026-10-07 (later, 4) | Logs **D-030** (user, 17:02): every real node gets its own diameter estimate whatever its optical depth; the removal of nodes with fitted $\hat\alpha > 1.0$ from $\mathcal S$ and their fill (D-024 (ii)) are withdrawn, and the threshold becomes a label. D-024's index status annotated and a note added under its status. Written by the theory chat after re-reading this log (unchanged since the 2026-10-07 (later, 3) merge). Next free IDs: D-031, I-003. |
| 2026-10-07 (later, 5) | Logs **D-031** (user, 18:01): the bias table also accounts for the estimated optical depth, as a third regression axis $\hat\alpha$ (settles D-030 comment (b)); and **D-032** (user, 18:01): tilt-specific tables are dropped, as kernel and as test, read by the theory chat as withdrawing D-027's later tilt-binned study (to be confirmed). Index statuses of D-024, D-027 and D-030 annotated. Written after re-reading this log (unchanged since the (later, 4) row). Next free IDs: D-033, I-003. |
| 2026-10-08 | Logs **D-033** (user, 09:56): the per-cell noise law of D-017 is fitted by our own indirect inference (least squares on the window statistics, point estimate), not by Zeraati et al.'s adaptive ABC (abcTau); settles the estimator-family half of D-017 (d). D-017's index status annotated. Written by the Stage-7 study chat after re-reading this log (unchanged since the (later, 5) row). Next free IDs: D-034, I-003. |
| 2026-10-08 (later) | Logs **D-034** (user, 11:19): the per-sweep multi-OU/BIC proposal is dropped; the variance at seconds is carried by the per-sweep tilt, its SD calibrated per cell from the semivariogram of the SS window means; the SS noise windows are enlarged where the pulse transient has died out. Settles D-017 (c) except the calibration lag; D-017's index status annotated. Written after re-reading this log (unchanged since the D-033 row). Next free IDs: D-035, I-003. |
| 2026-10-08 (later, 2) | D-034 annotated, no decision changed: Why (2)'s "its shape only weakly" is corrected (its 30 s comparison was not sized to the 6.89 s change; sized to it, the shapes separate with 17 sweeps); Why (1) is placed against the loss's own criterion; $h^*$ = 0.8 s recommended, not decided. Note under D-034, from the topic doc's v3. Written after re-reading this log (unchanged since the D-034 write). |
| 2026-10-08 (later, 3) | Logs **D-035** (user, 2026-10-07; logged late): the gradient energy of the profile is the focus rule of the diameter fit; the repository on `sci/diameter-pipeline` cites it as "D-030", which in this log is the optical-depth decision -- the ID collision is recorded in the entry. Logs **D-036** (user, 16:26): the plane-to-plane diagnostic evaluates the area under the measuring-line profile, plane by plane. Written by the diameter chat after re-reading this log (unchanged since the (later, 2) row). Next free IDs: D-037, I-003. |
| 2026-10-08 (later, 4) | Logs **D-037** (user, 17:07): the plane-to-plane diagnostics gain the entropy of the grey-level histogram along the measuring line and in a strip around it (notebook Cell 4f); the bin width, the strip's width and the pick rule are the assistant's provisional choices. Written by the diameter chat after re-reading this log (unchanged since the (later, 3) row). Next free IDs: D-038, I-003. |
| 2026-10-08 (later, 5) | D-037 annotated, no decision changed: the assistant's provisional pick of the entropy diagnostic is now each curve's global minimum (the dip rule stays as an option), after the trunk nodes 2 and 3 showed curves without a W. Logs **D-038** (user, 18:02): the measuring line is to be enlarged beyond $\pm 3$ um, to $\pm 4$ or $\pm 5$ um; the value, and whether Block 5's own line changes or only the diagnostics', are open. D-037's index status annotated. Written by the diameter chat after re-reading this log (unchanged since the (later, 4) row). Next free IDs: D-039, I-003. |
| 2026-10-09 | Logs **D-039** (user, 14:30): the entropy along a $\pm 5$ um line, and in its strip, is the plane-to-plane measure carried forward, and is tested next on thin dendrites against $k^*$; it settles D-038's value for the entropy. D-038's index status annotated. Written by the diameter chat after re-reading this log (unchanged since the (later, 5) row). Next free IDs: D-040, I-003. |
| 2026-10-09 (later) | Logs **D-040** (user, 16:19, with the answers to four questions): the gradient energy integrated over the whole measuring line is compared at $\pm 3$ um, $\pm 5$ um and on a line of full width $m \hat d$ ($m = 2$, a parameter), and on that last line against a min-max blend with the strip entropy weighted by a sigmoid of $\hat d$ and against the strip entropy alone (notebook Cells 4h and 4i). D-039's status annotated: Cell 4g ran on Allen's data. Written by the diameter chat after re-reading this log (unchanged since the 2026-10-09 row). Next free IDs: D-041, I-003. |
| 2026-10-09 (evening) | D-040's status updated, no decision changed: implemented and smoke-tested on `sci/diameter-pipeline` (commit `8a11f1e`, notebook Cells 4h and 4i), with two implementation choices and the dry runs on synthetic cells recorded under the entry; D-040's index status annotated. Written by the diameter chat after re-reading this log (unchanged since the (later) row). Next free IDs: D-041, I-003. |
| 2026-10-10 | Logs **D-041** (user, 2026-10-09 17:47 and the option chosen next): the blend's entropy is taken from the $\pm 5$ um strip, $G$ stays on the d-line, and the objective is written $J = w g - (1 - w) h$ with the entropy drawn as it is. D-040's status and index row annotated with the first run of Cells 4h and 4i on Allen's data. Written by the diameter chat after re-reading this log (unchanged since the (evening) row). Next free IDs: D-042, I-003. |
| 2026-10-10 (later) | Logs **D-042** (user, 15:30, and the option chosen next): the gradient energy's line is sized first from Allen's diameter, then from Block 5's fit at the plane each line picks, until a plane repeats or after 5 rounds (notebook Cell 4j). D-041's status and index row annotated with the re-run of Cell 4i on Allen's data (13:20 UTC). Written by the diameter chat after re-reading this log (unchanged since the 2026-10-10 row). Next free IDs: D-043, I-003. |
| 2026-10-12 | Logs **D-043** (user, 14:52): the Allen-guided iteration of D-042 is the method carried forward. D-042's status and index row annotated with the first run of Cell 4j on Allen's data (13 nodes, 12:43-12:45 UTC). Written by the diameter chat after re-reading this log (unchanged since the 2026-10-10 (later) row). Next free IDs: D-044, I-003. |
| 2026-10-12 (later) | Logs **D-044** (user, 15:03): the Allen-guided iteration's line multiplier is 3 (lines $\pm 1.5\, d$), the default of `--evaluation iterate` and of Cell 4j, which now writes `pilot/planeiter/bydiameter_m3/`; the d-line of Cells 4h and 4i keeps 2 (D-040). D-043's status and index row, and D-042's status, annotated. Written by the diameter chat after re-reading this log (unchanged since the 2026-10-12 row). Next free IDs: D-045, I-003. |
| 2026-10-12 (evening) | Logs **D-045** (user, 17:58, with the answers to four questions): notebook Cell 4k draws, for each node of Cell 4j's run, the plane the iteration chose with the line that chose it and the diameter fitted there, beside the profile and the fitted model, in an overview and one figure per node. Written by the diameter chat after re-reading this log (unchanged since the 2026-10-12 (later) row). Next free IDs: D-046, I-003. |
| 2026-10-13 | D-044's status and index row annotated, no decision changed: Cell 4j at $m$ = 3 has run on Allen's data (`pilot/planeiter/bydiameter_m3/`, files written 2026-10-13 12:25-12:27 UTC; Drive listing, metadata only, the files not yet read). Written by the diameter chat after re-reading this log (unchanged since the 2026-10-12 (evening) row). Next free IDs: D-046, I-003. |
| 2026-10-13 (later) | D-044's and D-045's statuses and index rows annotated, no decision changed: Cell 4k ran on Allen's data (13 nodes drawn, the fit redone equal to Cell 4j's d_iter on 13 of 13, no download), and its log gives the run at $m$ = 3 beside the one at $m$ = 2: the same plane on 12 of 13 nodes, node 8462 moved from 47 to 46. Written by the diameter chat after re-reading this log (unchanged since the 2026-10-13 row). Next free IDs: D-046, I-003. |
| 2026-10-13 (evening) | Logs **D-046** (user, 14:50, with the answers to six questions): notebook Cell 4l fits, at each node's plane of Cell 4j's run, Block 5's profile 100 ways -- ten blur kernels (scipy families, shapes fixed or fitted) x the width fixed at Block 5's or fitted x unweighted or with centre weights re-centred on the fit's own axis -- a diagnostic that changes no measurement. I-002's status annotated (part of it tried, on the fit only). Written by the diameter chat after re-reading this log (unchanged since the 2026-10-13 (later) row). Next free IDs: D-047, I-003. |
| 2026-10-14 | Logs **D-047** (user, 15:35, with the answers to three questions): notebook Cell 4m runs, for each of the 50 fit variants at the fitted width, its own Allen-guided iteration with Cell 4j's multiplier and round limit (inherited from Cell 4j's records), and draws per node a 50-row stack of the fitted edges and each variant's chosen plane with its line, the fitted dendrite's edges and d; a diagnostic that changes no measurement. D-046's status annotated (run on Allen's data). Written by the diameter chat after re-reading this log (unchanged since the 2026-10-13 (evening) row). Next free IDs: D-048, I-003. |
| 2026-10-15 | No decision changed. D-046: one sentence of its Why marked [corrected 2026-10-15] (a joint minimum over $v_0$ slides onto the background only where the fit has already merged a neighbour and stopped at the $v_0$ bound), and its status annotated with a Monte Carlo of the fitted-width fit on a thin tube; D-047's status annotated (how to read a small $d$ at $\mu$'s bound); both index rows annotated. Evidence: the checks behind the explanation of Cell 4m's objective (`check_objective.py`, C2, C3, C5) **[run, sandbox]**. Written by the diameter chat after re-reading this log (unchanged since the 2026-10-14 row). Next free IDs: D-048, I-003. |
| 2026-10-16 | Logs **D-048** (user, 09:29): the per-cell law fit weights its statistics with the inverse covariance of the statistics under the law, estimated by simulation and refined iteratively (first step from the real data's own weights, then $\hat\Sigma$ from $R$ simulated cells at the current $\hat\theta$, refit, until the objective plateaus); the sensitivity to a noise component the law does not capture is to be measured in silico during implementation. Settles the weights half of D-033's open (d). Written by the noise chat after re-reading this log (unchanged since the 2026-10-15 row). A handoff for the implementing chat, `claude/TEEG_Stage7_covariance_weighting_handoff_2026-10-16.md`, written the same turn. [corrected 2026-10-16: not in project knowledge -- the upload was refused because project knowledge is full; the handoff, with the sandbox checks under `checks/` and the bias document of 2026-10-11, was delivered to the user as a tarball for commit at `Passive Features/HPC script/Ih Fit/dev/2026-10-16_covariance_weighting/` (in the repo once pushed).] Next free IDs: D-049, I-003. |
| 2026-10-16 | No decision changed. D-048's changelog row, its Why and its Sources marked [corrected 2026-10-16]: the handoff `TEEG_Stage7_covariance_weighting_handoff_2026-10-16.md` is not in project knowledge (the write was refused, project knowledge full) and goes, with the sandbox checks and the bias document of 2026-10-11, to the repo at `Passive Features/HPC script/Ih Fit/dev/2026-10-16_covariance_weighting/`, delivered to the user as a tarball to commit and push (the assistant has no push access: the repo attach was denied on 2026-10-13). Evidence: the refusal of the project write this morning. Next free IDs: D-049, I-003. |
| 2026-10-16 | No decision changed. D-048's Why: the authors of the weighting paper marked [corrected 2026-10-16] (Kaczmarska, Isham & Northrop 2015, not "& Onof"; evidence: PubMed metadata of PMID 27563266). `claude/TEEG_Stage7_noise_model_2026-09-28.md` updated to v3.4 the same turn (new §3.5.6 on the D-048 weighting; §3.4.2 corrections carried from the bias document of 2026-10-11). Next free IDs: D-049, I-003. |
| 2026-10-16 (later) | Logs **D-049** (user, 10:16): the first-step weighting $W_0$ of the D-048 iteration is the diagonal of the cell's own window-level bootstrap variances, $W_0 = \mathrm{diag}(1/\widehat{\mathrm{SE}}_m^2)$, $\widehat{\mathrm{SE}}_m$ the SD of $B$ = 2000 bootstrap medians of the cell's $K_m$ per-window values of statistic $m$; chosen over the hand-set $\omega_m$ and over the diagonal of a cohort-law $\hat\Sigma$. Settles D-048's open point "how $W_0$ is derived by the real data"; D-048's index status annotated. Written by the D-048 implementing chat after re-reading this log (unchanged since the second 2026-10-16 row of the noise chat). Next free IDs: D-050, I-003. |
| 2026-10-16 (later) | No decision changed. **D-028 and D-029 statuses corrected:** the patch on `b355d8e` (`Ih Fit/noise_stats.py` with `sample_acf`, `smoke_noise_stats.py`, the two lag-1 estimates and `noise_calibration.lag_autocorr` rerouted) **was never applied anywhere** [user, 2026-10-16]. Evidence: the git archive of `main` taken on 2026-10-16 (`IhFit_main_2026-10-16.zip`) has no `noise_stats.py` and no `smoke_noise_stats.py`, its `CHANGELOG_ih_fit.md` ends at the 2026-10-01 row, and `noise_calibration.lag_autocorr` is still the pre-D-028 estimator (mean of the $n-L$ products over the mean square of $n$) [run, zip inspected]. Both return to "not implemented"; the D-048 implementation ships them as its first block, since the fit's estimator on data and simulations alike must be the textbook one (D-028, D-033). The 2026-10-07 (later, 2) row stands as the record of what was delivered. Next free IDs: D-050, I-003. |
| 2026-10-16 (later, 2) | Logs **D-050** (user, 10:30, answering three sub-questions): the Monte Carlo sizes and the stop rule of the D-048 iteration -- $R$ = 600 simulated cells per $\hat\Sigma_k$ while iterating and one final step at $R$ = 3000 that is what gets reported; $M$ = 3000 common-random-number windows per statistic in the binding function; the innovations of the $R$ cells and of the $M$ windows fixed per cell across iterations, so the iteration is deterministic; the plateau rule tests $\hat\theta$ only (max change 1 % in the fit coordinates), $J$ is reported, not tested; $k_{\max}$ = 6 as a tunable config parameter, a cell at the cap keeps its last iterate and is flagged `no_plateau`. Two brief explanations given with it ($R$ against $M$; how the iteration works), the $1/\sqrt M$ scaling of the binding function's Monte Carlo error checked by simulation (`check_M_scaling.py`, P1-P2 held, sent to the user). D-048's index status annotated. Written after re-reading this log (unchanged since the (later) rows). Next free IDs: D-051, I-003. |
| 2026-10-16 (later, 3) | Logs **D-051** (user, 10:50, answering four sub-questions, the recommended option each time): (i) the statistic set of the per-cell fit is $d$ = 15 -- the enlarged SS windows $[t_{\rm SS}(c), 200)$ ms of D-034 ($\hat\sigma$, $\hat\rho$ at 0.02, 0.1, 1, 10 ms) **replace** the 10 ms pre-pulse window, which is their last 10 ms; the ten LS statistics of §3.5.3 stay; fallback to the 14 of §3.5.3 where no enlarged window passes the transient test; (ii) **one law per cell for both protocols** (settles D-017 (b); supersedes D-014 (i) for every cell in the noise table); (iii) the tilt $b$ is **inside the fit's simulator**, $\sigma_b$ plugged in from (D-034.1) and alternated with $\hat\theta$ at every iteration, never a free parameter; (iv) each enlarged window has the **per-polarity mean response subtracted** before its statistics, the simulator doing the identical subtraction on its $K$ windows, so the operation is part of the estimator applied to both sides (D-033); $t_{\rm SS}(c)$ from the transient test stays as window start and flag; $\kappa_{\rm tr}$ = 0.1, 5 ms bins and $h^*$ = 0.8 s enter as labelled provisional config defaults (D-021 style). Recorded with it, a correction of the assistant's own figure in the question: a residual bounded by $0.1\,\hat\sigma_{\rm SS}$ at the window start and decaying contributes at most 0.03 % of the window variance, not 1 % (the 1 % is the bound for a residual that keeps its amplitude; `check_pulse_residual.py`, P1-P5 held). D-014, D-017, D-033, D-034 and D-048 index statuses annotated. Written after re-reading this log (unchanged since the (later, 2) row). Next free IDs: D-052, I-003. |
| 2026-10-16 (later, 4) | Logs **D-052** (user, 10:43, with the answers to three questions): the diameter measurement the bias table is built for -- per node the Allen-guided plane of D-042/D-044 with each round's line sized from the new fit, and as the fit the two-piece skewed generalized-$t$ kernel with its three shape parameters and its width fitted, centre-weighted at $\lambda$ = 0.5 um and re-centred by IRLS (D-046); real nodes and phantoms are measured by the same chain, and the error left is the table's to correct and report. Evidence recorded with it: the sandbox Monte Carlo and impostor search of 2026-10-15. Settles D-043 (a)'s first half, D-046 (a) and D-047's open point; D-043, D-046, D-047 and I-002 statuses and index rows annotated. The handoff for the table chat is in the repo (D-020), `docs/TEEG_diameter_correction_table_handoff_2026-10-16.md`. Written by the diameter chat after re-reading this log (changed since this chat last read it: D-048 to D-051 by the noise chats; merged, nothing of theirs edited). Next free IDs: D-053, I-003. |
| 2026-10-16 (later, 5) | No decision changed. D-050's Why (4) marked [corrected 2026-10-16]: its "about 1.2 ms on $\tau_s$" for the binding function's Monte Carlo error was scaled from the cohort fit of the topic doc §3.5.3, which used the hand-set $\omega_m$, and does not carry over to $\hat\Sigma^{-1}$, which weights the thin contrasts most. Evidence: `check_crn_determinism.py` (sent to the user 2026-10-16; $\tau_s$ alone free, the ten LS statistics, $K$ = 6, $R$ = 200, $M$ = 400, one simulated cell at $\tau_s$ = 39.74 ms, no tilt): with every draw frozen per seed the map $\hat\tau_k \mapsto \hat\tau_{k+1}$ is bit-identical when rebuilt from the same seed (P1) and meets the 1 % rule at $k$ = 1-2 for four seeds (P3), but the four seeds' fixed points span 34.46-40.15 ms under $\hat\Sigma^{-1}$ against 35.49-36.54 ms for the $W_0$ (D-049) fit of the same cell [run]. At D-050's sizes the seed-to-seed spread of $\hat\theta$ for fixed data is not measured; it joins the S0 proposal. D-050's index status annotated. Written by the D-048 implementing chat after re-reading this log (changed since its own (later, 3) write by the diameter chat's D-052; merged). Next free IDs: D-053, I-003. |
| 2026-10-16 (later, 6) | No decision changed. A note added under D-050's open optimiser point: the binding function $\mathbf q^{\rm sim}$, a median over $M$ simulated windows, is continuous but switches the window that holds the median as $\theta'$ moves, so `least_squares`' default finite-difference step (relative $\sqrt{\epsilon} \approx 1.5 \times 10^{-8}$ times $\max(1, |x|)$, SciPy 1.18.1 source) returns one window's slope; evidence `check_fd_jacobian_median.py` (sent to the user 2026-10-16) [run]. Written after re-reading this log (unchanged since the (later, 5) row). Next free IDs: D-053, I-003. |
| 2026-10-16 (later, 7) | No decision changed. D-052's Why: the thick-tube margins ($\Delta\chi^2_{\min} \ge 20.8$ at $\pm 5$ %, $\ge 67.6$ at $\pm 10$ %) marked [corrected 2026-10-16] -- they are the minima a local multi-start search found, so upper bounds, and say that no impostor was found; D-052's status annotated (handoff committed at `ea49181`; the per-solve `max_nfev` not settled, the two unconverged weighted fits having hit it); index row annotated. Evidence: an independent agent's fact-check of the handoff against the code and the result files, and draw 39 of the thick $t_4$ truth recomputed **[run, sandbox]**. Written by the diameter chat after re-reading this log (changed since its (later, 4) write by the noise chats' (later, 5) and (later, 6) rows; merged, nothing of theirs edited). Next free IDs: D-053, I-003. |
| 2026-10-16 (later, 8) | Logs **D-053** (user, about 13:05, the first card of the table chat: Q1, Q2 and Q4a of the handoff's §7): (i) the phantoms' radius is settled by a sensitivity check first -- the new chain run on rendered phantoms with the true radius, a fixed Allen-like one and a too-large one, read on $k^{\rm it}$ and $\hat d$ and shown as images of the in-focus plane with the measured diameters drawn -- then the true radius if nothing moves, else Allen-like draws; (ii) a fit leaves $\mathcal S$ when $d$, $\mu$, $v_0$ or $w$ is at a bound or the IRLS did not converge, the shape parameters at their box edges stay; (iii) the third table axis is $\ln\hat\alpha$. Settles D-052 (b), the coordinate half of D-031's open point and the procedure of D-043 (a)'s second half; D-031, D-043 and D-052 index rows annotated. Q4b (the axes' scaling) is explained, not yet decided. Written by the table chat after re-reading this log (unchanged since the (later, 7) row). Next free IDs: D-054, I-003. |
| 2026-10-18 | Logs **D-054** (user, about 11:15, the second card of the table chat: Q4b, Q4c, the failure-rate threshold and the per-solve budget): (i) the three table inputs are each divided by their design range before the thin-plate spline fit, cross-validation choosing the smoothing only; (ii) the local kernel's bandwidth in $\ln\hat\alpha$ is 0.3, PROVISIONAL; (iii) `max_failure_rate` stays 0.2 PROVISIONAL and is set on the davinci pilot's failure-rate curve by a later decision; (iv) the SGT fit's per-solve `max_nfev` is 3000, a `MeasureConfig` field. Settles the scaling half of D-031's open point and D-052 (h)'s cap; D-031, D-052 and D-053 index rows annotated. Two brief explanations given with it (why the axes are normalised; what the failure rate and `high_failure` are). Written by the table chat after re-reading this log (unchanged since the (later, 8) row). Next free IDs: D-055, I-003. |
| 2026-10-18 (later) | Logs **D-055** (user, 11:24, "Agree as written"): the in-silico sensitivity check of D-048 -- S0 closure, S1 the tilt as an approximation (seconds-scale OU twins), S2 four components the law lacks, S3 the 18 real L3_exc cells -- with every prediction fixed before any run (S0.1-S0.6, S1.1-S1.3, S2.1-S2.4, S3.1-S3.3), the twin cells, sizes and the cluster/sandbox split. Settles handoff §8 item 4 and D-048's "sensitivity measured in silico" design; D-048's index status annotated. Logs **D-056** (user, 11:24 and 11:31, three cards and one restated question): (i) the simulator reproduces each SS sweep as one continuous trace with the real pulse timing, cut at the real window positions (sharpens D-051 (iii), which said only that windows of a sweep share $b$); (ii) the spec of the noise-fit blocks is a new `Ih Fit/specs/SPEC.md`, the work on branch `sci/noise-fit` of a local git baseline built from the user's zip of `main`, delivered as a `git format-patch` series plus a tarball; (iii) the inner least squares: the binding function estimates the single-window median by the Harrell-Davis estimator (`scipy.stats.mstats.hdquantiles`) so that `scipy.optimize.least_squares` runs with its default finite-difference Jacobian, three starts (closed form, $\tau_s/2$, $2\tau_s$), best $J$ kept -- chosen over an own 0.02-step Jacobian after `check_fd_hd_median.py` (P1, P2 held; P3 failed in its favour) [run]. Settles D-050's open optimiser point; D-050 and D-051 index statuses annotated. Written by the D-048 implementing chat after re-reading this log (changed since its (later, 6) write by the table chat's D-053 and D-054; merged). Next free IDs: D-057, I-003. |
| 2026-10-18 (later, 2) | Logs **D-057** (user, about 11:40, the third card of the table chat: Q3, Q5 and the Q8-Q10 card): (i) `failure_rate_ignore` defaults to `()` -- every counted replicate enters the failure rate; (ii) D-023's $\sigma_{\rm fit}$ study is dropped: one production table and one real-node run at $\sigma_{\rm fit}$ = 0.099 um, `sigma_fit_study_um` and the `d_tilde_sigma_spread_um` column withdrawn (supersedes D-023 (iv)); (iii) **[user]** the rendered phantoms are measured by exactly the pipeline of the real data, the choice of the in-focus plane and the profile extent included -- D-052.4, until now the assistant's reading, is the user's decision. D-023, D-052 index rows annotated. Q7 and the specifics of Q8-Q10 ($d_{\max}$, `jitter_z_um`, the renderer's kernel) stay open: the user asked for a comprehensive explanation of how the tables are constructed first (given the same turn, equations inline from now on in this chat at the user's request, which amends handoff §11's PNG rule for the chat). Written by the table chat after re-reading this log (changed since its 2026-10-18 write by the noise chat's D-055 and D-056; merged, nothing of theirs edited). Next free IDs: D-058, I-003. |
| 2026-10-18 (later, 3) | No decision changed. D-052, D-053 and D-054 statuses and index rows annotated: **Block 12 implemented** on `sci/diameter-pipeline` (commit `bee1e0b`, local, not yet pushed) -- the skewed generalized-$t$ kernel (D-052.1) in `model/blur_kernels.py` (not added to `FAMILIES`, which stay the D-046 grid's), the fit at a plane with its IRLS (D-052.2) in the new `analysis/fit_sgt.py`, the `sgt_*` fields of `MeasureConfig`; D-053 (ii) is the fit's status rule; D-054 (iv)'s budget 3000. Evidence **[run, sandbox]**: `test_smoke_sgt_fit.py` 7 pass, the five D-052 Monte Carlo draws chosen (seed 20261017) reproduced bit for bit -- among them the weighted solve that stops at 3000 evaluations (status failed), a $\mu$ = 20 collapse (at_bound) and fits at $\xi$ = 0 (converged, `xi_lo` named) -- and the 23 other suites unchanged; estimator hash now `1bb3585b4a4f1373`. Recorded with it, open, for the user (SPEC §8): D-052.2's quadrature, $N$ = 848, converges only algebraically on the non-analytic members of the family (two-piece skew, non-even $\beta$, Laplace; observed orders 3.3, 2.6, 2.0); in the fitted regime ($w \ge$ 0.3 um, $d \le$ 5.5 um) it biases the IRLS $\hat d$ by at most 7.5e-4 um, at the box corner ($w$ 0.1 um, $d$ 6 um) the model errs by up to 2.1 grey levels. Written by the table chat after re-reading this log (unchanged since the (later, 2) row). Next free IDs: D-058, I-003. |
| 2026-10-18 (later, 4) | Logs **D-058** (user, 15:44, "use the method I said", the method stated at 13:57): each phantom's starting radius is $d_n/2 + \epsilon_n$, $\epsilon_n \sim \mathcal N(0, \sigma_r^2)$ -- supersedes D-053 (i)'s sensitivity check as the rule; $\sigma_r$, its scope (the line's start or all four places of handoff §6) and the rule for $r_n \le 0$ open. Logs **D-059** (user, 14:53 and 15:44): (i) the D-024 (iii) laws stay, with a check that the phantoms' $\hat\alpha$ covers the real cells'; (ii) $G$ independent cylinder draws over $(d, \varphi, \theta, \mu)$, each realized $k$ times with fresh nuisances, the per-draw estimate the mean ratio over its realizations in $\mathcal S$ -- supersedes D-024 (iii)'s one-realization design; $G$, $k$ (Q6) and the draw's $\hat\alpha$ coordinate open. Evidence recorded with them: `check_grouped_design.py` **[run, sandbox]** and the within-truth correlation of $\hat d$ and $\hat\alpha$ in the D-052 Monte Carlo **[data inspected]**. D-024 (note added), D-052 and D-053 (statuses) and their index rows annotated. Written by the diameter table chat after re-reading this log (unchanged since the (later, 3) row). Next free IDs: D-060, I-003. |
| 2026-10-18 (later, 5) | Logs **D-060** (user, 16:05, "Log A as the decision"): the table's spline is fitted through all $G k$ realizations in $\mathcal S$, each at its own $(\ln d_g, \varphi_g, \ln\hat\alpha_{g,r})$, the per-draw means $\bar y_g$ serving the per-draw scatter and the checks -- option (A) of D-059's open point on a draw's third coordinate; (B) draw means at the mean $\ln\hat\alpha$ and (C) the true $\alpha_g$ not taken. D-059's status and index row annotated. Written by the diameter table chat after re-reading this log (unchanged since the (later, 4) row). Next free IDs: D-061, I-003. |
| 2026-10-18 (later, 6) | No decision changed. Pushed at the user's request (16:10, "Push all the decisions done so far"): `origin/sci/diameter-pipeline` moved `ea49181..d643aaf` -- `bee1e0b` (Block 12) and `d643aaf` (SPEC records D-058 to D-060 in its section 4, annotates Block 6's design and radius for Block 14, corrects section 2.4's inversion sentence to the one-sign-change rule of Block 7, and lists the open points of D-058 to D-060 and the findings of the Steps 4-5 checks in its section 8; no code change). Evidence **[run, sandbox]**: before the push, every `.py` of the workstream pure ASCII, every `.sh`/`.pbs` LF-only, `compileall` of `src`, `scripts`, `tests` clean, estimator hash `1bb3585b4a4f1373`; after it, a fresh clone of the pushed branch has the tree of the tested working copy (`26b081f5`), the same byte checks and hash, `test_smoke_config` 6 pass 1 skip and `test_smoke_sgt_fit` 7 pass (the other suites passed on `bee1e0b`'s tree, the same code, 2026-10-18). D-052's status annotated. Written after re-reading this log (unchanged since the (later, 5) row). Next free IDs: D-061, I-003. |
| 2026-10-19 | No decision changed. D-057's Binds, its (D-057.2) and its implication (ii) marked [corrected 2026-10-19]: neither `scripts/build_table.py` nor `scripts/run_cell.py` has a `--sigma-fit` flag (read at `d643aaf`) **[src]**; what D-057 (ii) removes is `MeasureConfig.sigma_fit_study_um` with its `validate` check and its exclusion from `estimator_signature`, `config.with_sigma_fit`, the `d_tilde_sigma_spread_um` column of `analysis/cell.py`, notebook Cells 8 and 10c with the runbook, and the suites' uses of the study set (the 2026-10-19 handoff's §4). D-057's index status annotated. The table chat's handoff for the next chat is in the repo (D-020), `docs/TEEG_diameter_correction_table_handoff_2026-10-19.md`: the decisions D-053 to D-060 and what of them is coded, the open points (four for Block 13, sixteen for the user before Block 14), the plan of Blocks 13-14 and the gates. Pushed at the user's agreement ("Push all, with TPS doc") with `checks/table_2026-10-18/` (the checks behind D-054, D-059, D-060 and SPEC §8's Steps 4-5 findings) and `docs/TEEG_thin_plate_spline_in_3D/` (the study document of 2026-10-18): `origin/sci/diameter-pipeline` moved `d643aaf..84ea91f` (`6a8dbf4` the files and a SPEC §8 correction, `84ea91f` the handoff's clone URL; no code changed). Evidence **[run, sandbox]**: the 24 suites at `d643aaf` (151 pass, 0 fail, 21 skip); a fresh clone of `6a8dbf4` with the tested tree, clean byte checks and `compileall`, estimator hash `1bb3585b4a4f1373`. GitHub reports the repository moved to `Leonardodm00/Towards-EEG`; the lowercase URL redirects **[run]**. Written by the table chat after re-reading this log (unchanged since its (later, 6) row). Next free IDs: D-061, I-003. |
| 2026-10-19 (later) | No decision changed; repairs by the noise chat. (1) **Restored after a lost update:** the noise chat's 2026-10-18 row (written about 11:46 as "(later, 2)") and its annotations of D-028's and D-029's statuses and index rows are not in this log as read today: they were lost in a later write made from a copy read before them (the table chat's "(later, 2)" row, D-057, holds that slot; nothing of the table chat's is edited here). Restored as written then: **D-028 and D-029 implemented** as Block 1 of the noise-fit work, commit `b4da926` on branch `sci/noise-fit` of the local baseline (`9ed4da4` = `IhFit_main_2026-10-16.zip`), delivered to the user as `0001-block-1-textbook-sample-autocorrelation-through-nois.patch` (sha256 `133e4c906cad0fae...`), `git apply --check` and `git am` verified on a fresh extract of the zip; `Ih Fit/noise_stats.py` (`sample_acf`), `smoke_noise_stats.py` (N1-N6 6/6, mutation-tested), the fitter's two lag-1 sites, `noise_calibration.lag_autocorr`, the R8/R10 bands of `smoke_ih_recovery.py`; README, CHANGELOG; new `Ih Fit/specs/SPEC.md` (D-056 (ii)). Sandbox gates `smoke_ls_baseline_qc` 10/10, `smoke_ih_fit` 12/12, `smoke_ih_recovery` 14/14 **[run, sandbox, 2026-10-18; re-run 2026-10-19 on the same commit, all four green, `smoke_noise_stats` 6/6 included]**; `regression_passive_identity.py` NOT run (oracle folder and archive on the cluster) -- pending the user's apply, push, statsmodels in `prova` (D-029 (v)) and the cluster gates. (2) **A misplaced annotation:** D-056's index row carried "[updated 2026-10-18, later: (iv) implemented in Block 12, `bee1e0b`]"; D-056 has parts (i)-(iii) only, none implemented, and the annotation is D-054 (iv)'s (`MeasureConfig.sgt_max_nfev` = 3000, the diameter pipeline's Block 12, as D-054's own status records), whose index row lacked it. Struck through and marked [corrected 2026-10-19] on D-056's row, entered on D-054's. (3) D-056's open point on the SS sweep numbers annotated with D-017 (c)'s record that the archive writer stores the SS pulses in order with their sweep number. Written after re-reading this log (changed since the noise chat's 2026-10-18 write by the table chat's (later, 2) to (later, 6) rows and its 2026-10-19 row; merged). Next free IDs: D-061, I-003. |
| 2026-10-19 (later, 2) | No decision changed. **Block 2 of the noise fit implemented** (D-017, D-034, D-056 (i)): commit `1ab29f3` on `sci/noise-fit` (on `b4da926`), delivered as `0002-block-2-the-noise-law-simulator-from-unit-draws-and-.patch` (sha256 `a115a53132db643c...`) in `TEEG_noise_fit_blocks_1-2_2026-10-19.tar.gz` with the Block 1 patch; the two-patch series `git am`-verified on a fresh extract of `IhFit_main_2026-10-16.zip`, the applied files byte-identical to the tested tree. `Ih Fit/noise_law.py` (model layer): `NoiseLaw`, `autocovariance` (L3), `draw_units` / `simulate_traces` / `simulate_sweep` (both AR(1) recursions by `scipy.signal.lfilter` from unit draws, stationary start, per-trace tilt and offset, one continuous trace per sweep), the exact moments of a demeaned window (topic doc (7a), (7b), (9)), `closed_form_sd` / `closed_form_acf` ((10), the fit's start only), `visible_share` (8), `window_mean_cov` / `semivariogram` (eq. (25)'s $V(h \mid \theta, \sigma_b)$ computed exactly for the sampled law by lag counts over sample pairs -- an implementation refinement of D-034.1's closed form, recorded in SPEC 2.4; `semivariogram_continuous` = eq. (25) as written, the oracle). `smoke_noise_law.py` M1-M9 **9/9** **[run, sandbox]**: brute-force covariance moments to 1.7e-14, Monte Carlo within 3.5 SE, stationarity, window-mean covariance against brute force and Monte Carlo, discrete $V$ against eq. (25) to 1e-6, the tilt, contract and determinism, the visible share to 8e-8, and a regression against the 14 closed-form values of the topic doc's 3.5.3 at $\hat\theta_{\rm cf}$ (within 2.9e-5 mV and 9e-4); mutation-tested with five planted defects, each caught. No existing file changed; SPEC Block 2 entry, README and CHANGELOG rows. The repo attach with push access was denied again (classifier), so delivery stays by patch (D-056 (ii)). Next: Block 3 (`noise_windows.py`), which waits for the `ss_pulses.npz` key listing (D-056 open point). Written after re-reading this log (unchanged since the (later) row). Next free IDs: D-061, I-003. |
| 2026-10-20 | Logs **D-061** (user, 12:21): Block 13's implementation points -- one provider call over the pilot's block and $k_{\rm SWC} \pm 6$ plus one call per restack (a); the pilot and the iteration + SGT fit as separate functions, one `measure_node` for phantoms and real nodes, the diagnostics calling the pilot (c); the flags of $\mathcal S$ computed on the fitted plane $k^{\rm it}$ (d); Block 10's calibration nodes selected by Allen's diameters, thin branches for the near field (<= 0.84 um) and dendrites of about 0.8 um for the far field (D-026) (b, in the user's own form); the defocus kernel calibrated in Colab and loaded into the davinci configuration, where the draws, the table and its tests run (f). Logs **D-062** (user, 12:21): Block 14's values -- $\sigma_{\rm br}$ = 0.2 um in all four places, a redraw for $r \le 0$, per realization; $G \approx 10^5$ draws x $n_{\rm real}$ = 10; CV folds by draw; the smoothing grid extended to $10^4$; the failure count at the true $\alpha_g$; items 13 and 15 of the handoff's §5.2 dropped, item 14 closed (D-054 (iii) stands). Recorded with D-062 and raised to the user: at $N \approx 10^6$ rows the dense thin-plate spline of D-024 (v)/D-060 is infeasible (8 TB, about $2 \times 10^5$ h extrapolated from fits timed here), so the response estimator is to be decided before Block 14 (options (A)-(D), the assistant recommends local kernel regression on a grid); the redraw shifts the mean starting diameter of the thinnest phantoms (0.404 um at $d$ = 0.2 um). Evidence **[run, sandbox]**: `check_design_effect.py` (MC/formula 1.002 at $\rho$ = 0.2, $n$ = 10), `check_table_size.py`, `check_redraw_shift.py`, `fig_support_band.py` (the assistant's scratchpad, to be committed with Block 14). Index rows and statuses of D-024 (v), D-026, D-052, D-054, D-058, D-059, D-060 annotated. Written by the table chat after re-reading this log (changed since its 2026-10-19 write by the noise chat's (later) and (later, 2) rows; merged, nothing of theirs edited). Next free IDs: D-063, I-003. |
| 2026-10-20 (later) | Logs **D-063** (user, 16:11): at $N \approx 10^6$ rows the table's response estimator is local linear kernel regression on a grid with D-054's bandwidths (the deliverable), a penalized tensor-product B-spline the labelled comparison; the SE is the smoother's own with the draw-level (cluster) correction; `off_support` on the kernel mass at the root; the third axis becomes $\ln(\hat\mu\hat d)$ (amends D-053 (iii), retires D-054 (i)); supersedes D-024 (v)'s spline and the spline of D-060.1. Logs **D-064** (user, 16:11): Block 13's iteration reads a scene input (the cell's traced segments and soma) for the gradient energy's background, empty on a phantom (settles D-061 (e)). Logs **D-065** (user, 16:11): the ranges stay; the coverage check is a Colab survey of the real neurons' fitted optical depths across cells, per bin of $\hat d$; Allen's radii across cells set $d_{\max}$; the node jitter is set from the registration survey; Q7's pinned-kernel comparison not run. Logs **D-066** (user, 16:11): the renderer's defocus kernel becomes the SGT family (I-002 adopted), calibrated in Colab and used in both chains; recorded with it that every symmetric member is the line projection of a nonnegative isotropic PSF (inverse Abel transform, `check_abel_sgt.py`) and that the skew is not renderable isotropically. The assistant's push-back recorded in D-065: the Colab surveys and the calibration change the simulation hash, so the production launch should follow them (the davinci pilot need not). Grounding: PubMed 7 queries (one methods full text read, Li & Cao 2022, PMC9672641), bioRxiv category scan (nothing). Statuses and index rows of D-024, D-026, D-052, D-053, D-054, D-059, D-060, D-061, D-062 and I-002 annotated. Written after re-reading this log (unchanged since the 2026-10-20 row). Next free IDs: D-067, I-003. |
| 2026-10-20 (later, 2) | Logs **D-067** (user, 16:41): the Colab surveys and the calibration precede the production launch (D-065's recommendation, now the decision); the renderer stays Gaussian and the fit SGT -- **D-066 withdrawn** before any implementation, I-002 back to an idea (the Abel finding kept on record); only the local linear estimator is built (D-063.5's P-spline comparison dropped); Block 10's thresholds in Allen's diameter, thin <= 0.6 um and far field 0.6-0.8 um, PROVISIONAL, the other criteria from the pilot. The implementation plan redrafted for the user the same turn (to be committed as `docs/TEEG_diameter_table_implementation_plan_2026-10-20.md` with Block 13); the note for the theory chat rewritten (project knowledge full: delivered as a file). Index rows and statuses of D-026, D-061, D-063, D-065, D-066 and I-002 annotated. Written after re-reading this log (unchanged since the (later) row). Next free IDs: D-068, I-003. |
| 2026-10-20 (later, 3) | No decision changed (noise chat). (1) **Block 1 is on `main`**: `e8ed84e`, the user's push of the 2026-10-18 patch; `Ih Fit/` on `main` is byte-identical to the tested Block-1 tree, one dev script's executable bit aside **[src: a read-only clone of `main`, 2026-10-20]**. The laptop command of 2026-10-19 failed because the tarball was downloaded but never extracted -- the command omitted the `tar -xzf` step (the assistant's error) -- and would then have stopped at 0001, already applied. (2) **Delivered instead: one file, `TEEG_noise_fit_block2_tools_2026-10-20.patch`** (sha256 `ac735bf4...`), two commits: Block 2 (content unchanged since 2026-10-19; author set to the user, as for Block 1) and **the tools** -- `cluster_gates_noise_fit.sh` (G0-G12, one PASS/FAIL/SKIP line per check, a check passing only on the suite's own summary line, a GATES and a VERDICT line, logs outside the repository), `noise_fit_gates.py` (`bytes`, `compile`, `ss-pulses`) and `smoke_noise_fit_gates.py` (T1-T8 **8/8**, mutation-tested). Written into the user's `Downloads` through the desktop link (folder access granted for this session) and read back byte-identical. **[run, sandbox]** The series applied with `git am` on `main` at `e8ed84e`, its files byte-identical to the tested ones; the runner in FULL mode there, with `Biological Fit/` and `Synthetic Passive fit Test/` from `main` and a synthetic archive group: **13 passed, 0 failed, 0 skipped** -- `smoke_ih_recovery` 14/14 with R1 against the oracle folder, `regression_passive_identity` PASS (max |dL| = 0 over 20 points, x and fun identical) on a synthetic cell. (3) **D-056's sweep-number open point answered at source**: the archive writer (`Passive Features/Allen Institute Data/save_alleninstitute_data (6).py`, every copy of `_archive_one_cell` in it) stores `sweep_number` (int64, one per pulse) in Formats A and B, which the loader does not read; a sweep's windows in onset order, (-10, +200) ms around each onset; pulses failing the 200 +/- 30 pA, 0.5 +/- 0.5 ms QC or too close to a trace end dropped silently; onset times not stored; per its docstring, citing the Allen tech paper, the 20 SS pulses of one sweep alternate in polarity, 200 ms apart **[src]** [corrected 2026-10-20 (later, 5): 37 per sweep, 200.5 ms onset to onset, data]. Adjacency and the true onset spacing are recoverable from the exact sample overlap of consecutive windows (`noise_fit_gates.py ss-pulses`, checked on writer-like fixtures **[run]**); the files themselves are confirmed by G12 on the cluster. (4) **For Block 5**: the twins' generator gives every SS pulse its own sweep number and independent noise (a generated cell: 20 sweeps of one pulse, windows -10 to +200.5 ms) **[run; src]**, while D-017's per-sweep offset and tilt and D-056 (i)'s continuous sweeps imply SS twins with the real sweep structure; to settle with the user when Block 5 is specified (SPEC section 8). Pending: the user's `git am` and push; on the cluster `bash cluster_gates_noise_fit.sh`. Written after re-reading this log (changed since the noise chat's (later, 2) row of 2026-10-19 by the table chat's rows of 2026-10-20, D-061 to D-067; merged, nothing of theirs edited). Next free IDs: D-068, I-003. |
| 2026-10-20 (later, 4) | No decision changed (noise chat). **On the cluster, 17:43** (output pasted by the user) **[cluster]**: the clone at `/davinci-1/home/ldellamea/TEEG/Towards-EEG` fast-forwarded `3dfa263..e8ed84e`, so Block 1 is now there; **D-029 (v) holds**: `statsmodels` 0.14.6 is installed in `prova` (Python 3.12, numpy 2.4.4, scipy 1.17.1, pandas 3.0.2). `cluster_gates_noise_fit.sh` was not found because `origin/main` is still `e8ed84e` (checked with `git ls-remote`, 17:44): the laptop's `git am` and `git push` of `TEEG_noise_fit_block2_tools_2026-10-20.patch` have not reached GitHub. **[run, sandbox]** with statsmodels 0.14.6 in place of 0.15.0: `smoke_noise_stats` 6/6, `smoke_noise_law` 9/9, `smoke_noise_fit_gates` 8/8. Written after re-reading this log (unchanged since the (later, 3) row). Next free IDs: D-068, I-003. |
| 2026-10-20 (later, 5) | No decision changed (noise chat). (1) **Blocks 1-2 are settled on the cluster [cluster, output pasted 18:47]**: `cluster_gates_noise_fit.sh` FULL at `ca028ea` (the user's push of Block 2 `36ba658` and the tools), env `prova` (python 3.12.13, numpy 2.4.4, scipy 1.17.1, statsmodels 0.14.6, neuron 9.0.1, skopt 0.10.2): 13 passed, 0 failed, 0 skipped, VERDICT SETTLED (G5 6/6, G6 9/9, G8 12/12, G9 10/10, G10 14/14); G11 `regression_passive_identity` PASS on the real cell 508282493 (loss max\|dL\| = 0 over 20 points, `x` and `fun` identical, ref 0.172357; 1085 s). D-029 (v) holds on the cluster. Logs `~/ih_fit_gate_logs/20261020_175303/`. (2) **The SS sweep structure, data inspected [cluster: G12; Google Drive copy of the archive files]**: in 508282493 and 569844159 every SS sweep is one train of **37 pulses** (19 dep, 18 hyp, alternating, first and last dep), onset to onset **200.5 ms** (10025 samples at 50 kHz in every adjacent stored pair: 22/22 and 319/319, all of opposite polarity), about 7.2 s from first to last onset. `metadata.json['ss_extraction']` (Drive folder `L3_exc`, files written 2026-06-19): `n_pulses_raw` 74 and 370 (2 and 10 sweeps), `n_pulses_ampdur_qc` 74 and 370 (no amplitude/duration or boundary drop), `n_pulses_qc` 40 and 348; trace QC engine v2, alpha 0.5, B 500, scales 1-10 ms, 40 ms test window (self-calibration chunk 2000 samples), self-calibration off: 34/74 dropped (46 %; 33 `pathological`, 1 `spike`) and 22/370 (6 %). Every gap in the stored windows is a trace-QC drop, and the position of each stored window in its train is recovered exactly: in 508282493 from `trace_qc_results.csv` (its `passed` rows reproduce the G12 polarity strings of sweeps 92 and 93 exactly; the 74 rows split into two alternating 37-pulse trains only at row 37), in 569844159 by counting (the parity-minimal reconstruction accounts for all 22 drops). The Drive copy is taken for the cluster's on that match; the cluster files were not hashed. The Allen white paper (Oct 2017 v.5, appendix p. 15 **[KB]**) says 20 repetitions at 200 ms intervals: these sweeps disagree on the count (documentation vs data); 200 ms reads as the interval from pulse end to next onset (0.5 + 200 = 200.5 ms) **[reasoning]**. The 20 / 200 ms / 4 s statements in this log (row 2026-10-20 (later, 3), D-017 (c), D-017's note of 2026-09-28 (later, 2) item (1), D-017's later note on the SS series, D-034's Statement, D-056 (i)) are marked [corrected 2026-10-20] in place; so are `specs/SPEC.md` sections 2.4, 3, 8 (local commit on `sci/noise-fit`, delivered with Block 3) and the Stage-7 handoff. (3) **A selection the fit inherits [src; data; run, sandbox]**: the stored SS windows passed a test on the lag-1 structure of the post-pulse residual in [2.5, 42.5] ms, the timescales of the law's slow part. Check SQC (the repo's `trace_qc_v2.py`, last changed 2026-07-21, with the archive's configuration; Block-2 simulator; 2-sweep cells like 508282493, RC response 113 MOhm / 28.1 ms; enlarged-window statistics of D-051 of the kept windows against a random subset with the same per-polarity counts, which is what the fit's simulator reproduces): Gaussian noise of the law alone makes the QC drop 12.9 % (theta_cf, 200 cells) and 12.0 % (sigma_s 0.025 mV, 150 cells); the selection shifts the medians by -0.17 to -0.19 per-cell SD at t_SS = 40 ms (theta_cf; -0.06 to -0.12 at sigma_s 0.025), by at most 0.09 SD at 80 ms and 0.08 SD at 120 ms, and V_hat(0.802 s) by at most 0.07 SD **[run]**; with 10 sweeps the per-cell SD is about 2.2 times smaller, so the same shifts are about 0.4 SD at 40 ms and at most 0.2 SD from 80 ms **[reasoning]**. The larger shift of the kept windows against all windows (sigma_hat -0.06 to -0.37 SD) is the per-polarity mean subtraction over fewer windows, which the simulator reproduces with the cell's own counts (D-051 (iv)). Not covered: the 508282493 drop rate (46 %) is far above the noise-only 13 %, so most of its drops remove structure the law lacks (events, misfit, non-stationarity) **[reasoning]**; 569844159's 6 % is below it. Script `sqc_trace_qc_selection.py`, sent to the user 2026-10-20, not in the repo. (4) **Open, for the user before Block 3** (the assistant's recommendations): (a) window positions from the archive's own QC table and counts, the loader checking each cell and flagging a cell it cannot reconstruct exactly; (b) no trace QC in the binding function, with a floor t_SS(c) >= 80 ms and the excess drop rate per cell written as a flag; (c) for Block 5, the twins' SS sweeps as 37-pulse continuous sweeps keeping the real cell's stored positions. PubMed, two queries on selection or quality control inside simulation-based inference: one returned 48 records, not screened, the other none; nothing used. Next free IDs unchanged: D-068, I-003. |
| 2026-10-20 (later, 6) | No decision changed (table-implementation chat). **Block 13 implemented** on `sci/diameter-pipeline`, four local commits, not pushed (`b5191c8` 13.1, `e50abd9` 13.2, `70136ca` 13.3, `f38b1a6` 13.5): D-052.3 with D-052.2 inside `measure_node` (the pilot, the Allen-guided iteration on $K_j$, the SGT fit at $k^{\rm it}_j$, the flags at $k^{\rm it}_j$), D-061 (a), (c), (d), (e) and D-064; the pilot reads only Block 5's planes of the one block and equals Block 5 at `84ea91f` bit for bit; the estimator hash is `1654924e11020dd9` (simulation `a7fd8eb517841fd4`), so every table built before is refused **[run, sandbox: the 25 suites, 157 pass, 0 fail, 22 skip; `checks/block13_2026-10-20/check_13_3_pilot.py`]**. Annotated: D-052, D-061, D-064 (implemented), D-062 (a consequence of D-062.1 at the thin end, and the per-realization time). Open for the user: whether an iteration ending `cycle`, `max_rounds` or `no_plane` after round 1 fails $\mathcal S$ (implemented as not: `iter_status` a column; one line of Block 6's `reject_reasons`). Sources: `specs/SPEC.md` Block 13 and section 8 at `f38b1a6` **[src]**. |
| 2026-10-21 | Logs **D-068** (user, 09:32): an iteration that ends without converging keeps its fit and stays in $\mathcal S$ when the rest allows, with a flag `iter_<ending>` (a label, not a criterion); and **D-069** (user, 09:32): the iteration's first line holds at least 3 samples (option (b) for the thin end of D-062.1). **Block 13 pushed** at the user's agreement: `84ea91f..888452a` on `sci/diameter-pipeline`; a fresh clone of the pushed branch has the tested tree (`ff3ffc6`), a clean `compileall`, pure-ASCII `.py`, the hashes `1654924e11020dd9` / `a7fd8eb517841fd4` and `test_smoke_config` 6 pass, 1 skip **[run, sandbox]**; the 'local, not pushed' notes of D-052, D-061 and D-064 annotated, and D-062 (its thin end settled by D-069). D-068 and D-069 are to be implemented as Block 13.6. |
| 2026-10-21 (later) | Logs **D-070** (user, about 12:04, the question card after S2): the far-field calibration of D-026 becomes stage S2b, designed after Colab's Cell 7 counts the far-field nodes, the continuation beyond 0.84 um staying D-024 (i)'s configured $\gamma$ = 0.79 until then; S4's code is written while S3 runs. Status notes. **D-068 and D-069 implemented** (Block 13.6, `f4ac106`) and pushed at the user's agreement (`888452a..f4ac106` on `sci/diameter-pipeline`). **S2 of the 2026-10-20 plan implemented** (pushed at the user's agreement (`f4ac106..9c1329e` on `sci/diameter-pipeline`)): `1aed5d6` S2.1 -- Block 10's calibration nodes selected by Allen's diameter (D-061 (b), D-067.4): the near-field set scanned, the far-field set selected and counted but not scanned, its scan and the fit of (D-026.1) not being built (SPEC section 8; a gap of the plan, added there as S2b, proposed); `a879067` S2.2 -- D-057 (ii), the $\sigma_{\rm fit}$ study removed; `9c1329e` the plan's status. The estimator and simulation hashes are unchanged by S2 (`0508e7ff12f48728` / `bf65d525963d5f0f`). Evidence: the 25 suites; 27 of 27 mutants of the selection killed **[run, sandbox]**. Notes added to D-026, D-027, D-057, D-061, D-067, D-068, D-069 (entries and index). Next free ID: D-071. |
| 2026-10-21 (later, 2) | Logs **D-071** (user, 12:30): Block 14's open choices -- the bandwidth multiplier by cross-validation with folds by draw, the by-draw sandwich the one SE, the grid at half the chosen bandwidths, `mu_d_out_of_domain` a label -- and the far-field calibration (S2b) straight after Block 14 (amends D-070.1's timing). Statuses: D-030, D-057 (i), D-058, D-059, D-060, D-062, D-063 implemented by Blocks 14.1-14.6 on `sci/diameter-pipeline` (`89ca78a`, `e9682a5`, `779bec0`; local, not pushed), the 25 suites passing and 33 of 33 mutants of 14.3-14.6 killed **[run, sandbox]**; D-070.2 done. |
| 2026-10-21 (later, 3) | Logs **D-072** and **D-073** (user, 14:53; noise chat). D-072: the stored SS windows -- trace-QC'd when each cell was archived -- are the noise fit's data as written; no shape test is added anywhere in the noise fit, the twins included; each window's place in its 37-pulse train comes from the archive's own `trace_qc_results.csv`. Evidence **[cluster, 2026-10-21]**: all 18 L3_exc cells carry the table; `qc_rows` = `n_pulses_raw` = `n_pulses_ampdur_qc` = 37 x the number of SS sweeps (2 sweeps in 14 cells, 6 in one, 10 in three); 6-50 % of pulses dropped, median 27 %. D-073: each twin sweep is the twin of one real sweep and carries noise calibrated on that sweep, not a per-cell average; the per-sweep form is open, with the assistant's proposal from `sweep_identifiability.py` **[run, sandbox]** (one sweep identifies the fast pair to 1-3 %, the slow amplitude to 18-30 % with all four free and 6-13 % with the slow timescale held, the slow timescale only within a factor of about 2). D-016 (d) settled; D-017 and D-051 (ii) superseded in part and annotated, in the entries and the index. Open for the user: D-072's $t_{\rm SS} \ge 80$ ms floor and drop-rate label, D-073's comments (a)-(d). Next free IDs: D-074, I-003. |
| 2026-10-21 (later, 4) | Logs **D-074** (user, 14:47; the diameter implementation chat): S2b's far scan reads both sides of focus to +-16 planes with an SNR rule at 3, on a +-10 um line; the line-end baseline accepted with it has no samples beyond plane 9 ($\gamma$ = 0.79; plane 7 at 1.21), the assistant's correction of its own proposal, and a planar background from far pixels is proposed in its place; the far fit's design explained with checks run in the sandbox (renderer model) and not yet decided. **Block 14 pushed** at the user's agreement: `f63fe9f..825bef1` on `sci/diameter-pipeline` (`89ca78a`, `e9682a5`, `779bec0`, and `825bef1` the SPEC Block 15 draft); a fresh clone has the tested tree, estimator hash `0508e7ff12f48728`, simulation hash `sim-0ea5a6d2ec069c68` **[run, sandbox]**. Status notes on D-026, D-070, D-071 (entries and index). Next free IDs: D-075, I-003. |
| 2026-10-22 | Logs **I-003** (user, 15:09; the diameter implementation chat): the construction of the defocus kernel from a deep-learning perspective, taken up in a chat of its own; no decision changed (the renderer stays Gaussian, D-067.2; S2b's open points stay with the implementation chat, D-074). Its handoff, `TEEG_diameter_defocus_kernel_DL_handoff_2026-10-22.md`, is committed to `docs/` on `sci/diameter-pipeline` (local commit `b82b1ee`, not pushed); the project copy under `claude/` was refused (the project is at its size limit). Checked for it **[run, sandbox, 2026-10-22]**: at `825bef1` 25 of 25 smoke suites pass, hashes estimator `0508e7ff12f48728`, simulation `0ea5a6d2ec069c68`; the S2b node bootstrap's interval coverage recomputed (92 % at +-1.96 SE, 93 % at $t_{11}$, $J$ = 12). Next free IDs: D-075, I-004. |
| 2026-10-22 (later) | Logs **I-004** (user, about 16:20, the answer to the push card; the diameter implementation chat): the hybrid renderer of the 2026-10-09 renderers document -- each slice's absorption from the condenser's rays, then the defocus kernel -- whose optics the user studies first; added to the roadmap as theory stage T7 (open, `982186c`) and written to the roadmap page. **Pushed** at the user's agreement: `9112bf3..37b42ea` on `sci/diameter-pipeline` -- the DL kernel handoff (`b82b1ee`, `941a128`), merged with the remote's roadmap commit `9112bf3` (`3376e94`), T7 (`982186c`) and the handoff's update (`37b42ea`); a fresh clone has the tree of the working copy, a clean `compileall`, pure-ASCII `.py`, hashes estimator `0508e7ff12f48728` and simulation `0ea5a6d2ec069c68`, and `roadmap.py check` passes **[run, sandbox]**. Status notes on I-003 (entry and index). Next free IDs: D-075, I-005. |
| 2026-10-22 (later, 2) | Logs **D-075** (user, 17:36; the diameter implementation chat): whenever the near-field growth fit and the far-field law are computed, the run plots every node's per-plane width points with the fitted near curve and far line, and every off-focus plane's dip with its fitted Gaussian core; the layout is the assistant's reading, PROVISIONAL; the scans file must then keep each plane's profile and fit (today per-plane statistics only). Next free IDs: D-076, I-005. |
| 2026-10-23 | No decision changed (noise chat). **Roadmap of the fitting pipeline created** at the user's request (10:59): `specs/roadmap/allen-mtg-fit.yaml` with its rendering and the roadmap tool copies, on `main` -- delivered as the patch `0001-roadmap-allen-mtg-fit-created.patch` (commit `1d8d9ea` on `ca028ea` in the sandbox clone, applies on `origin/main`), not yet applied -- and its status document written to the Subproject Roadmaps page (`towards-eeg:allen-mtg-fit`, version 1); the sync task picks the yaml up once pushed. **Theory [user]**: the passive-fit topics T1-T4 closed ("everything about the passive fit is studied"), the noise topics T5-T9 studying, T10 (the archive's SS layout and the trace-QC selection) closed by D-072, the I_h topics T11-T14 open; **three topics proposed by the assistant as not yet tackled, DRAFT, for the user to keep, merge or drop**: T15 temperature and the transfer of kinetics (no Q10 in either .mod, plan s.3), T16 the corrected morphologies as the fit's input (D-013, F = 1.9), T17 comparing the campaign's arms. **Implementation, computed where the SPEC has an entry**: Blocks 1-2 `pushed` (on `main`, no night-tester report covers them), Blocks 3-7 and the gate run (proposed Block 8) `planned`; **the legacy stages carry a manual status with the evidence of the cluster runs of 2026-10-20/21 (set by hand, until the SPEC gets entries for them -- scientific-coding s.1.6, Confirmed: inferred)**: Phase-0 archive, Biological Fit Phases 1-3, the regression harness, the noise tables and the cluster gates `done`; the I_h fitter Stages 1-6 and Stage 7 as built `tested` (suites 12/12 and 14/14 on davinci, no real cell fitted, the gate never run); the synthetic passive benchmark `tested` (DRAFT: no cluster run recorded); the campaign and the record `planned`. Counts: implementation planned 8, pushed 2, tested 8, done 7; theory open 7, studying 5, closed 5. Next free IDs: D-076, I-005. |
| 2026-10-23 (later) | Logs **D-076** (user, 12:08; noise chat): the per-sweep form is option (a) (the fast pair and sigma_s per sweep, sigma_s shrunk towards the cell by the DerSimonian-Laird weight, tau_s and the tilt per cell); the SS noise window starts at max(80 ms, transient test), and the subtracted quantity is confirmed as the per-polarity mean window (the per-sample mean over windows, D-051.1), not each window's scalar mean; the twins carry the real cell's own LS amplitudes; no QC anywhere in the noise fit (restating D-072). D-072 and D-073 annotated. Block 3 (`noise_windows.py`, `noise_cell_stats.py`) started the same turn. Next free IDs: D-077, I-005. |
| 2026-10-23 (later, 2) | Logs **D-077** (user, 13:03; the diameter implementation chat): S2b's far statistic iterates its core fit, each pass's window sized on the previous pass's width, at most $N$ passes (10 the user's example), stopping when the FWHM changes by no more than one SE of itself (the form, the SE from the fit's residual RMS, and the `not_converged` status are the assistant's reading, PROVISIONAL); checked in the renderer's model: noise-free the two-pass width already sits at the asymptote (within $10^{-9}$), and at 1 gl the rule stops by pass 3 in all 3600 fits. D-074 and D-075 annotated. Written by the diameter chat after re-reading this log (changed since this chat last read it: the 2026-10-23 rows, D-076 and the D-072/D-073 annotations, by the noise chat; merged, nothing of theirs edited). Next free IDs: D-078, I-005. |
| 2026-10-23 (later, 3) | [restored 2026-10-23 (later, 4): written by the noise chat at about 13:10 as "(later, 2)" and lost when the diameter chat's concurrent write of D-077 (13:03) replaced the log; relabelled because D-077's row holds "(later, 2)"; text otherwise as written] No decision changed (noise chat). **Blocks 3 and 4 of `Ih Fit/specs/SPEC.md` implemented, smoke- and mutation-tested, delivered** as the patch series `block34_patches_2026-10-23.zip` (four patches on GitHub `main` `64fb8c5`: the two spec commits of 2026-10-20/21, `block 3: windows and statistics`, `block 4: the per-cell fit`; `git am` verified on a fresh checkout, QUICK gates 8/8 incl. the new `G6b`, `smoke_noise_fit` 10/10; the user's roadmap patch is on `main` as `64fb8c5`). Block 3: `noise_windows.py`, `noise_cell_stats.py`, `smoke_noise_windows.py` W1-W10 10/10, eight mutations caught. Block 4: `noise_fit.py`, `smoke_noise_fit.py` F1-F10 10/10 (about 100 s at reduced M, R), ten mutations caught; measured cost 16 s per q_sim evaluation at M = 3000 and 0.10 s per Sigma_hat replicate on a 2-sweep cell (a cell 1-3.5 h, a 10-sweep cell 2-5.5 h, one core). **Two findings put to the user, no decision yet [open]:** (i) D-051 (iv)'s transient test as written is noise-limited -- with K = 63 windows the odd part r_odd has a noise floor of 0.17 sigma_SS, above kappa_tr sigma_SS = 0.1 sigma_SS, so it never passes on noise alone; Block 3 reads the bin MEAN and offers `WindowConfig.transient_floor_z` (threshold at least z x the even part's spread; default None = as written; the assistant recommends z = 3) [corrected 2026-10-23 (later, 4): the 0.17 sigma_SS is the option's floor estimate (the per-sample SD of r_even) on one fixture realisation, not the SD of the test's statistic, which is 0.24 sigma_SS at K = 63; the test as written is noise-limited at every L3_exc window count, not only at K = 63; z = 3 as implemented is withdrawn -- it still sends 4-13 % of noise-only cells to `ss_enlarged_none`; see the row (later, 4)]; (ii) the per-cell tilt sigma_b(c) of D-034 rests, for a 2-sweep cell (14 of 18), on two draws of b: (T1) is exact on 200 replicate cells (1 %) but gives 0.17 mV/s for a planted 0.1 on the fixture; options listed in the SPEC's section 8 (keep per cell with its uncertainty, pool across cells, use the LS slopes). [corrected 2026-10-23 (later, 4): the LS pre-window slopes are not an option -- excluded by D-034's status (D-017 note 4); the assistant's error] **Assistant's choices recorded in SPEC 2.3 (a)-(h), flagged:** CRN by re-seeding, non-finite statistics dropped, bounds and finite-difference steps, `max_nfev` 15 trf iterations, three starts for the W_0 solve and a warm start after, J and q_sim reported at the final (theta_hat, sigma_b_hat) pair, the chi-square tail as a report field only, the cell-level mean window for the per-sweep statistics, every LS sweep's pre-window in q_hat. Next: Block 4b (`noise_sweep_fit.py`, D-076 (i) with DerSimonian-Laird), then Blocks 5-7. Next free IDs: D-077, I-005 [corrected 2026-10-23 (later, 4): D-078, I-005 -- D-077 was logged by the diameter chat at 13:03]. |
| 2026-10-23 (later, 4) | No decision changed (noise chat). **The two findings of Blocks 3-4 explained to the user (15:04) and checked by computation first** (standalone `check_findings_2026-10-23.py` with `results_2026-10-23.txt`, sent to the user; not in the repository; theta_cf, the archive's SS layout -- 37-pulse trains, 10025-sample period, stored (-10, +200) ms windows -- at K = 40, 63 (2 sweeps), 199 (6), 348 (10); 200-2000 simulated cells per case) **[run]**. **(i) Transient test (D-051 (iv), D-076 (ii)):** under noise alone the SD of a 5 ms bin mean of r_odd in [100, 200) ms is about sigma_s sqrt(2 / K) (the slow component, 5 ms << tau_s; x sqrt(2) from the pre-pulse baseline): simulated 0.30, 0.24, 0.15, 0.11 sigma_SS at K = 40, 63, 199, 348, at or above kappa_tr = 0.1 at every L3_exc window count (passing needs K >> 2 (sigma_s / kappa_tr sigma_SS)^2, about 320); as written the test puts t_SS >= 190 ms (`ss_enlarged_none`, D-051 (i)'s unbuilt fallback) in 90, 85, 68, 56 % of noise-only cells; the fixture's RC response (113 MOhm, 28.1 ms) falls below 0.1 sigma_SS only at 130 ms even without noise. The option `transient_floor_z` (z = 3, the floor read from r_even) still fails 4-13 % of noise-only cells: near the window end a window's samples correlate with the next, opposite-polarity window's pre-pulse baseline (slow component, 9.5 ms overlap), inflating r_odd's noise and deflating r_even's (per-bin SD at 195 ms 0.28 against 0.08 sigma_SS, K = 63, sigma_b = 0), and r_even carries the tilt's ramp -- **the assistant's z = 3 recommendation withdrawn**. A per-bin null simulated at the cell's layout and law (threshold max(kappa_tr sigma_SS, z x null SD)) fails at most 0.3 % at z = 3.5 and gives t_SS 80 ms (K = 63) and about 100 ms (K = 348) on the fixture's response. **Options put to the user [open]:** (a) D-051 as written (not viable), (b) the simulated per-bin null at z = 3.5, (c) t_SS = 80 ms for every cell with (b)'s test as a flag only -- **the assistant recommends (c)** (after D-051.1 an identical response is removed exactly; what leaks is pulse-to-pulse variation, which r_odd does not measure; D-051 (iv)'s P3: subtracted medians off by at most 0.022 SD with a 10 sigma_SS response at the window start and 2 % jitter, an assumed level that S3 measures). **(ii) Tilt (D-034.1, (T1)):** given the realised tilts b_1..b_S of the cell's S SS sweeps, E[sigma_b_hat^2 | b] = sum_s (N_s / N) b_s^2 (pairs of one sweep share its b), so sigma_b_hat^2 / sigma_b^2 is about chi^2_S / S: at sigma_b = 0.1 mV/s, t_SS = 80 ms, (T1) at the true law, the 95 % range of sigma_b_hat / sigma_b is 0.00-1.95 with 2 sweeps (clipped at 0 in 3.5-4.9 %), 0.42-1.56 with 6, 0.59-1.44 with 10; sigma_b_hat^2 unbiased (mean ratio 0.97-1.01); pooled over the cohort's 64 SS sweeps under a common sigma_b, 0.82-1.17. A sigma_b error of that size (0 or 0.17 for a cell at 0.1) moves u_hat by at most 0.18 of its SD (linearised at theta_cf, the fixture layout, Sigma_hat at R = 400, CRN at M = 3000): the stake is the twins' tilt -- its share of an LS twin's step-baseline error is sigma_b x 0.64 s. **Options put to the user [open]:** (a) per cell as D-034 says, with its interval, (b) one cohort sigma_b, (c) partial pooling across cells by DerSimonian-Laird as D-076.1 does across sweeps -- **the assistant recommends (c)**; the LS pre-window slopes are excluded (correction above). **Repository:** commit `e75bdea` on `sci/noise-fit` (SPEC 2.2 and section 8, `noise_windows` comments, `CHANGELOG_ih_fit.md`; no logic changed), to ship with the next patch. Written after re-reading this log, which the diameter chat had changed (D-077, the D-074/D-075 annotations; merged, nothing of theirs edited) and which had lost this chat's row of about 13:10, restored above as (later, 3). Next free IDs: D-078, I-005. |
| 2026-10-23 (later, 5) | Logs **D-078** (user, 17:13; the diameter implementation chat): the answers to questions 1-3 of the S2b answer of 2026-10-21 -- the background is the planar far-pixel fit of D-074.3, the line-end mean kept as its check; the far statistic is D-077's iterative core fit with $N$ = 10 and $\kappa$ = 1; the far fit is the recommended one (sequential with Block 10's anchor, both sides, axis free within +-0.28 um, $\lvert k - k^*\rvert \ge 5$, model weights by IRLS as coded, the predicted-dip SNR rule, the node-bootstrap SE with $t_{J-1}$ intervals plus the anchor term). Questions 4 (isolation and tilt) and 5 (the junction) still open. D-074 and D-077 annotated. Written by the diameter chat after re-reading this log (changed since this chat last wrote it: the noise chat's (later, 3) and (later, 4) rows, (later, 3) restored after this chat's D-077 write of 13:10 had replaced the log; merged, nothing of theirs edited). Next free IDs: D-079, I-005. |
| 2026-10-23 (later, 6) | Logs **D-079** (user, 17:39; the diameter implementation chat): a calibration node, near or far, needs a +-25 um background patch holding no traced point but its own branch (no other segment, branch point or soma); the near-field calibration uses the planar far-pixel background of D-078.1 too (the reading of "in any direction" and of the near-field extension are the assistant's, PROVISIONAL). Settles the isolation half of D-074's Q-D (question 4); its tilt half and the junction (question 5) still open. D-067, D-074 and D-078 annotated. Written after re-reading this log (unchanged since D-078's write unless stated in the rows above). Next free IDs: D-080, I-005. |
| 2026-10-23 (later, 7) | Logs **D-080** (user, 18:09; the diameter implementation chat): the junction -- every kernel table written for rendering ends at the anchor, 0.84 um (question 5 of the S2b answer, (a)); and the calibration's width-by-plane figures, near (planes $k^* \pm 3$) and far (from the anchor), all selected nodes in light grey with their median, plus one example node with the images of its planes, to judge whether a non-linear law is needed (layout readings PROVISIONAL). The user's message of 17:47 (cut short) confirms that D-079.1's isolation counts neighbours in depth too; the depth range stays open. D-074, D-075 and D-079 annotated. Written after re-reading this log. Next free IDs: D-081, I-005. |
| 2026-10-23 (later, 8) | Logs **D-081** (user, 18:30; the diameter implementation chat): the seven implementation questions of Blocks 10 and 15 answered -- isolation by the predicted shadow reaching the patch above 0.03 gl in some scanned plane, the node's own dendrite excluded; the planar far-pixel background in the near scan too, near fitted before far; tilt <= 5 deg for both sets (exposed); the core fit's passes exposed, unconverged planes dropped and counted; the figures as proposed (signed offset, raw and centred panels, residuals, the example node nearest the median with its patch images); the quadratic term renamed $\gamma_2$; the defaults (1000 bootstrap resamples, +-0.01 um anchor step, Block 10's $\hat\sigma_0$ bootstrap, the 0.84 um cut today) confirmed. Implementation ordered. D-074, D-077, D-079 and D-080 annotated. Written after re-reading this log. Next free IDs: D-082, I-005. |
| 2026-10-23 (later, 9) | No decision changed. **Blocks 10 and 15 implemented** per D-077 to D-081 (the diameter implementation chat, 20:30): commits `24eb687` (the patch scan, planar background, isolation and straightness, anchor SE, table cut; the far scan, statistic, selection, fit, SE and production kernel), `cd80081` (the figures of D-075 and D-080.2) and `f55c083` (Colab Cell 7 and runbook v22) on `sci/diameter-pipeline`, **not pushed** (the user decides) [updated 2026-10-26: pushed to `origin/sci/diameter-pipeline` at the user's request, `cc60f3c..dd901de`]; all 28 smoke suites pass, the estimator and simulation hashes unchanged. One PROVISIONAL refinement recorded under D-081: the straightness test uses the trace's own TLS line, not the pilot's. The roadmap re-rendered (`dd901de`) and written to the page (I10, I15 local; I15 no longer planned); T6 stays `studying` until the user closes it. Statuses of D-074, D-075, D-077 to D-081 annotated. Written after re-reading this log (identical to the (later, 8) version). [corrected 2026-10-23, 20:50: the four commits were re-authored under the Claude identity (the repository's stop hook, GitHub verification) before any push, so their hashes changed from `c374dce`, `31204ac`, `bb47fef`, `26960e8` to the ones now cited; the content is the same.] Next free IDs: D-082, I-005. |
| 2026-10-26 | No decision changed. **Blocks 10 and 15 pushed** at the user's request (09:01): `cc60f3c..dd901de` on `origin/sci/diameter-pipeline` (the four commits `24eb687`, `cd80081`, `f55c083`, `dd901de`). **Roadmap T6 (far-field calibration) closed** by the user on the Subproject Roadmaps page (08:59) and applied in the chat as the sync procedure's step 4 (`roadmap.py set T6 closed`, `c27e3a9` pushed, the page's request deleted); TG3 closed with it; on the page I15 moves from planned to pushed and I1, I10, I12, I13, I14 are pushed (no tester report covers them yet; the last report is 2026-10-07's). Statuses of D-074, D-075, D-077 to D-081 annotated. Written after re-reading this log (identical to the 2026-10-23 (later, 9) version). Next free IDs: D-082, I-005. |
| 2026-10-26 (later) | Logs **D-082** (user, 09:11; the diameter implementation chat): every step still to do goes on the roadmap as a step, general for now, and each step has a theory part of its own. Applied to the diameter-pipeline roadmap with `roadmap.py add step` -- the tool another session pushed at 09:19 (`c410c95`: steps name their theory group, blocks inherit it in blocked_by): S3 (S3a-S3d), S5, S6, S7, S8, S9 and, not scheduled, NS1-NS2, with the theory parts TS3 (TS3a-TS3d), TS5-TS9, TNS1 and T7 for NS2; DRAFT until the user confirms them (`d6d410d`, pushed; page version 41). Written after re-reading this log (identical to the 2026-10-26 version). Next free IDs: D-083, I-005. |
| 2026-10-26 (later, 2) | Logs **D-083** (user, 09:31; the noise-fit chat): the six-parameter synthetic recovery (Stage 7) runs on davinci before any fit of the six parameters on a real cell -- twins on each cell's own morphology, with its own fitted noise and its own stimulation protocol, given chosen values of the six parameters and fitted by the same fitter; the six-parameter version of the passive-only synthetic check, which the user reports having run. Roadmap allen-mtg-fit updated to match: S7 restated, S8's note orders the campaign (its one-cell dry run included) after S7, I4 (the passive benchmark) set to done on the user's statement, and N4b (Block 4b, the per-sweep stage of D-076 (i)) added with its theory part T9 under **D-082**; commit `8c671a2` in the noise chat's clone, delivered as a patch on `main` `64fb8c5` (not pushed: the user's push); page version 25. Written after re-reading this log (identical to the 2026-10-26 (later) version). Next free IDs: D-084, I-005. |
| 2026-10-26 (later, 3) | No decision changed (noise-fit chat). **The tilt $\sigma_b(c)$ explained to the user as a study document** (`TEEG_tilt_sigma_b_2026-10-26.md` with five figures and the standalone checks check01-check05, sent to the user, not in project knowledge), and roadmap stage **T8** set back to `open` at the user's word ("I do not think I have read the theory on this"; commit `e0a0a36` on top of `8c671a2`, both delivered as patches on `main` `64fb8c5`, not pushed; page version 27). Evidence for open choices only, recorded as a note on **D-034**: $h^*$ = 4 periods still carries the Long Square step's baseline error with the 120 ms windows of $t_{\rm SS}$ = 80 ms (read/match 0.987-0.989 for OUs of 3-30 s); a 2-sweep cell's $\hat\sigma_b(c)$ is at the Cramer-Rao limit of its two realised tilts; on an 18-cell cohort with the L3_exc sweep counts, partial pooling of $\log\hat\sigma_b(c)^2$ by DerSimonian-Laird is never worse than the better of per-cell and cohort (share of 2-sweep cells off by more than a factor 2) and never returns zero; the LS tail rejects one common variance at seconds. Clarification marked on **D-076**: statsmodels 0.15 does not clip the DL $\hat\tau^2$ itself (Block 4b must). Written after re-reading this log (identical to the (later, 2) version). Next free IDs: D-084, I-005. |
| 2026-10-26 (later, 4) | Logs **D-084** (user, 16:48; the diameter implementation chat): the hybrid absorption renderer of I-004 is promoted to be the renderer actually used, once built after its optics (T7); until then Block 4 keeps the partition of D-024 (i). I-004 and D-024 annotated. D-082 annotated: the roadmap changes it left open (G1's theory part, TG3 shared by G4 and G5) and any other roadmap change wait for the user, who is developing other changes to it (16:48). Written after re-reading this log and merging the noise-fit chat's rows of 09:31 and later (D-083 there). Next free IDs: D-085, I-005. |
| 2026-10-26 (later, 5) | Logs **D-085** (user, 17:09; the diameter implementation chat): the answers to the eleven questions on S3, the real-data programme in Colab -- five cells per neuron class and layer (all of a group with fewer); the alignment automatic; the plane spacing and pixel size checked first over all the cells; the node selection exposed (every node or the stratified sample); every figure plus the diameters by layer and class; the thick-node check dropped until the renderer is the right one; one camera setting; one calibration kernel per cell, with a check of whether a cell has enough clean dendrites; the final configuration waits for the hybrid. The user asked what $d_{\max}$ is for (answered in the chat; its rule still open). D-065 and D-084 annotated. Written after re-reading this log (identical to the (later, 4) version). Next free IDs: D-086, I-005. |
| 2026-10-26 (later, 6) | Logs **D-086** (user, 17:16; the diameter implementation chat): $d_{\max}$ is the maximum of Allen's dendrite $2r$ across S3's cells, not the 99.5th percentile proposed (D-085.7's open rule). D-085 and D-065 annotated; D-085's status records Block 16.1 built locally -- the cell pool, the acquisition check over the whole pool, the stratified selection (SPEC Block 16.1, smoke-tested, not pushed) -- with (D-085.1)'s readings as PROVISIONAL defaults. Written after re-reading this log (identical to the (later, 5) version). Next free IDs: D-087, I-005. |
| 2026-10-26 (later, 7) | Logs **D-087** and **D-088** (user, 17:59; the diameter implementation chat): sparsely spiny cells are a third class of S3's selection, beside exc and inh (settles D-085.1's open reading); one defocus kernel, calibrated on near- and far-field nodes pooled across several cells, hence one table (supersedes D-085.10's kernel per cell), tested for now on one pilot cell, with at least 10 and at most 30 calibration nodes per set. D-085 annotated. Also at 17:59 the user told this chat to push its verified S3 blocks to `sci/diameter-pipeline` without asking (a working rule of this chat, not a decision for other chats); Block 16.1 is on the remote (`27af07d..2fd8ffb`; the commit was `1dc0768` before its rebase onto another session's roadmap commit `27af07d`). Written after re-reading this log (identical to the (later, 6) version). Next free IDs: D-089, I-005. |
| 2026-10-27 | Logs **D-089** and **D-090** (user, 14:19; the diameter implementation chat): the defaults of the S3 blocks confirmed -- the set-aside ratio 2, the darkest-projection rule, the stratified default, the class name `sparsely_spiny`, the z0 flag at 1 um, 5 scans per set per cell, "per type" as per calibration set, 529878215 the pilot (D-089); what Allen's projections are, settled by a Colab check because the sandbox cannot reach Allen (D-090.1, SPEC Block 16.3, Cell 11b), and the pooled calibration's near and far points drawn together, coloured by cell, with their residuals from the pooled law per cell (D-090.2, SPEC Block 19.5). Status notes, no decision changed: D-085 (Blocks 16.1-19 on the remote), D-086 (Block 18), D-087 (Block 16.1 amended), D-088 (Block 19) implemented; evidence the commits and smoke tests named in each **[src; run, sandbox]**. |
| 2026-10-27 (later) | Logs **D-091** and **D-092** (user, 16:28; the diameter implementation chat): every new SPEC block enters its roadmap in the same turn, without being asked -- the hold of 2026-10-26, 16:48 lifted for this (D-091; applied: Blocks 16-19 as I16-I19, NS2 now HR, S3d after HR, `689c702`, `9b99b8b`, page version 55); "GO on with the hybrid renderer": built as SPEC Block 21, Block 4's absorption mode `hybrid` (Block 20 stays S3d), smoke-tested, the default left at the partition (D-092; `2572e15`). Findings for the user and T7 in D-092.3: the hybrid's planes go below zero in dark stain (from $\mu d \approx 2$ with the production kernel), it is closer to the ray world than the partition like for like up to $\mu d \approx 1.5$, and it costs 4 to 50 times the partition per phantom. Annotated: D-082 (the hold lifted for blocks), D-084 and I-004 (built, not switched), D-085, D-086, D-088 (the index's statuses brought up to their entries'). The index's rows for D-089 and D-090, missing from v19, are added. Written after re-reading this log (identical to the 2026-10-27 version). Next free IDs: D-093, I-005. |
| 2026-10-27 (later, 2) | No decision changed (the table theory chat). **Roadmap diameter-pipeline: T4 split into T4.1-T4.3**, the split the theory chat made on 2026-10-22 (its commits were never pushed), re-applied: T4.1 the table's design (T4's closed DRAFT carried: 2026-10-21, D-071), **T4.2 (Step 4: the grid value and its SE) closed** and **T4.3 (Step 5: the cubic read-out and the inversion) studying**, at the user's word (18:28: "The rest about the Stage 4 has been studied, update the roadmap"); T8 is now drawn blocked by T4. Commit `bfa23a1` on `a342769`, **not pushed** (this session's git proxy refused the push and the request for push access was refused), delivered to the user as a patch; page version 60, which the next Roadmap sync overwrites until the commit is pushed. The T4.3 study handoff (`TEEG_diameter_table_readout_inversion_study_handoff_2026-10-27.md`, with `table_theory_2026-10-27_full.zip`) sent to the user as files, not in project knowledge (that write was refused: the project is at its size limit). Recorded there: T4.2's open questions 4c (in part), 4d, 4e, 4f; the correction of the 2026-10-26 Stage 4 explanation's Eq. (5), whose factor $1 + (n_g - 1)\rho$ belongs inside the sum over draws. Written after re-reading this log (unchanged since the 2026-10-27 (later) row). Next free IDs: D-093, I-005. [updated 2026-10-28, 09:04: pushed at the user's word ("push by yourself") once push access was attached: rebased onto `eed549d` (D-093's three commits, no roadmap file among them) and pushed as `bae2ae1`; page version 64] |
| 2026-10-27 (later, 3) | Logs **D-093** (user, 19:08; the diameter implementation chat): the calibration nodes are not screened for straightness -- the calibration evaluates only the measuring line; flatness ($\varphi \le$ 5 deg) stays strict; if the test is restored, its tolerance is 2 (or 4) x Allen's radius. Supersedes D-081.3's straightness part. Applied: `CalibrationConfig.node_straight_check` False by default, `node_straight_tol_radii` 2 (`6a74735`; the evidence's wording corrected in `d46340c`, the blur-reach caveat in `eed549d`). D-081 annotated (index and status). D-085 annotated: (D-085.3)'s acquisition check run on Allen's data, every line the user pasted at dz 0.28 um and pixel 0.1144 um ("All share the same."). The roadmap gains no stage (Block 10 amended). Written after re-reading this log (unchanged since the (later, 2) row; merged onto it). Next free IDs: D-094, I-005. |
| 2026-10-28 | Logs **D-094** (user, 09:12; the diameter implementation chat): the calibration nodes are selected without the optical-depth criterion ($\hat\alpha \le 0.5$, `not_faint`) and without the flag `faint` of $\mathcal S$; $\mathcal S$ itself keeps the flag (the assistant's reading, to confirm). Applied as two switches, off by default (`CalibrationConfig.node_alpha_check`, `node_faint_flag_check`), `223a5a6`. D-026, D-067 and D-093 annotated. A check on rendered near-set phantoms is recorded under the entry: the fitted kernel moved by under 1 % at 0.84 um and under 2 % at 0.56 um between $\alpha$ = 0.1 and 2. The roadmap gains no stage (Block 10 amended). Written after re-reading this log (changed since v21 only by the other session's annotation of the (later, 2) row, 09:04; merged). Next free IDs: D-095, I-005. |
| 2026-10-28 (later) | Logs **D-095** (user, 09:55; the noise-fit chat): the SS window start is read from the exponential decay of the mean windows -- mean windows per polarity and per sweep over [-10, 200) ms, $t_{\rm SS}(c)$ where the decay falls within $\kappa_{\rm tr}\hat\sigma_{\rm SS}(c)$, no binning, and the mean window of the same sweep and polarity subtracted on $[t_{\rm SS}(c), 200)$; supersedes D-051 (iv)'s 5 ms-bin test of $\bar r_{\rm odd}$ and the per-polarity grouping of (D-051.1), and D-076 (ii)'s $t_{\rm tr}(c)$ with its cell-level mean window for the per-sweep statistics; the options (a)-(c) of 2026-10-23 closed, none taken; open points under the entry (mean or median, "statistically", how $t_{\rm SS}(c)$ is formed from the per-group times, the decay model and baseline, the 80 ms floor and the 190 ms fallback), to ask before Block 3 changes. Recorded with it **[run, sandbox]**: for a noiseless RC response at $\kappa_{\rm tr}$ = 0.1 the start is 130 ms for the fixture (28.1 ms, 113 MOhm; windows of about 70 ms instead of 120) and passes 190 ms from $\tau_m$ of about 46 ms (`check_d095_rc_start.py`); the decay must be measured against a fitted baseline, the pre-pulse level holding the previous pulse's tail; the 2026-10-26 leak law and the 2026-10-28 make-up of $\hat\sigma_{\rm SS}(c)$ (87 % fast noise). D-034, D-051 and D-076 annotated. No roadmap change (D-082's hold); N3's note is stale. Written after re-reading this log (changed since this chat's write of 2026-10-26 by the diameter chats' rows 2026-10-26 (later, 4) to 2026-10-28, D-084 to D-094; merged, nothing of theirs edited). Next free IDs: D-096, I-005. |
| 2026-10-28 (later, 2) | Logs **D-096** (user, 09:58; the diameter implementation chat): the flag `faint` leaves the selection $\mathcal S$ for phantoms and real nodes -- every node whose fit is in $\mathcal S$ is corrected, whatever its optical depth and its dip, and the table holds replicates of every dip depth; the flags of $\mathcal S$ enter the estimator signature (the assistant's choice), so both config hashes change (estimator `59678d4e2aedf2b6`, simulation `52aa73e479eb77cf`). Supersedes D-094.3. Implemented `cf2f91d` (the repository first called it D-095; renamed `76365a3`, the ID being the noise-fit chat's). D-030, D-053 and D-094 annotated. Written after re-reading this log (changed since v22 by the noise-fit chat's D-095 row and entry, 09:55; merged, nothing of theirs edited). Next free IDs: D-097, I-005. |
| 2026-10-28 (later, 3) | Logs **D-097** (user, 10:24; the noise-fit chat): the assistant pushes its own commits, and every document is also committed to the GitHub repository so any chat can consult it -- where, the assistant's reading: byte copies of the project-knowledge documents under `docs/project/` on `main` (the project copy first, the repository copy in the same turn), a workstream's handoffs in its `docs/`, study documents and check packages in its `dev/`; the user's uploads not copied without their word; the repository is public. Applied: the five noise-fit patches and the two roadmap patches pushed (`main` from `64fb8c5` to `4930ba1`), then `53b2713`, `d2ffcb4`, `cd32dad`; the copies of this log and of the living handoff follow in the next commit. Roadmap allen-mtg-fit updated at the user's word ("Update also the roadmap": D-082's hold lifted for this update): N3 and N4 pushed, each with D-095's revise mark; page version 44. D-082 and D-095 annotated. The user still has to answer the tilt option for 2-sweep cells (per cell, one cohort value, or partial pooling across cells; put to the user on 2026-10-23 and 2026-10-26), written into the noise-fit handoff at the user's request. Written after re-reading this log (changed since this chat's write of 2026-10-28 by the diameter chat's row 2026-10-28 (later, 2), D-096; merged, nothing of theirs edited). Next free IDs: D-098, I-005. |
| 2026-10-28 (later, 4) | Status notes from the Colab run of S3's Cells 11, 11b and 12 on Allen's data (the user's logs and figures of 11:16; the diameter implementation chat), no new decision: **D-085** (the pool, 158 cells; the acquisition check, 158 of 158 at dz 0.28 um and 0.1144 um/px; the selection, 51 cells in 15 groups; the alignment, 47 of 51 aligned, 4 failed on Allen's marker file, 5 set aside), **D-089** (.2: the darkest rule took the maximum-intensity xy projection in each of the 4 specimens that have one; .1: of the 5 cells set aside, 4 have shifts like the kept cells'), **D-090** (.1: on the pilot the minimum-intensity xy projection is the planes' minimum, r* 0.9941 at zero offset). The changes these suggest are put to the user, not made. Written after re-reading this log (changed since v23 by the noise-fit chat: D-097's row, index row and entry, notes on D-082 and D-095; merged, nothing of theirs edited). The repository copy of this version (D-097) is on `sci/diameter-pipeline`, `docs/project/`, until the user settles D-097 (d). Next free IDs: D-098, I-005. |
| 2026-10-28 (later, 5) | Logs **D-098** (user, 15:01; the diameter implementation chat): "Let's do this" to the assistant's proposals (a), (b), (d) after the Colab run of Cells 11-12 -- `fetch_swc` never takes Allen's marker file, the xy projection chosen by its type 'MinimumIntensityProjection - xy' (D-089.2's darkest rule the fallback), S3a's coverage recomputes $\mathcal S$ from the rows' columns -- with (c), the set-aside cells that look aligned run with `--include-set-aside`, and the user's request that Cell 7 draw each candidate's background patch with the neighbours behind `not_isolated`. Implemented and pushed (`c595bc0`, `4ef6d8e`, `cdba78a`, `28790c3`, `44e95e3`). D-089 (its .2 superseded) and D-090 annotated. Written after re-reading this log (unchanged since v24). The repository copy (D-097) on `sci/diameter-pipeline`, `docs/project/`. Next free IDs: D-099, I-005. |
| 2026-10-28 (later, 6) | Logs **D-099** (user, 17:07; the diameter implementation chat): "Blocks 13 and 14 are too slow. Let's change approach. Everything now on colab will run on a single pilot cell while the real runs will be on the DaVinci HPC. Adjust all the scripts and the pipeline" -- S3's per-cell survey (Cells 12-15) runs on davinci as three PBS jobs (SPEC Block 16.4: `survey_prepare.pbs`, the array `survey_cells.pbs` with one task per selected cell, `survey_pool.pbs`), and every Colab cell runs on the pilot 529878215 only (`cell_pool.py pick`, `survey_pilot/`, 3 nodes per bin). Implemented, smoke-tested and pushed the same evening (`5230474`, `d4559ba`, `ede27be`, `01a15d6`). D-085 and D-098 annotated. Written after re-reading this log (unchanged since the (later, 5) row). The repository copy (D-097) on `sci/diameter-pipeline`, `docs/project/`. Next free IDs: D-100, I-005. |
| 2026-10-28 (later, 7) | Logs **D-100** (user, about 18:37; the noise-fit chat): "Let's do A and B for the 2 sweep cells." -- for the 14 L3_exc cells with two SS sweeps the twins' tilt SD is taken both per cell (option (a), D-034's (D-034.1)) and as one cohort value (option (b)); partial pooling across cells (option (c)), the assistant's recommendation, is not taken. The assistant's reading, to confirm with the user: two labelled variants of twins per 2-sweep cell; the cohort value the $S_{\rm eff}$-weighted mean of the unclipped per-cell estimates over the 18 cells, clipped at 0; each real cell's own fit unchanged; open points (a)-(f) in the entry. D-034 annotated (entry and index). Written after re-reading this log (unchanged since the (later, 6) row). The repository copy (D-097) on `main`, `docs/project/`. Next free IDs: D-101, I-005. |
| 2026-10-28 (later, 8) | Logs **D-101** (user, 18:17 and the dialog's answers at about 18:40; the diameter implementation chat), after Cell 7's isolation overview on the pilot (70 candidates, 2 far-field, both shaded by the soma or a junction): flatness per set -- far $\varphi \le 15$ deg, near $\le 10$ deg --, and the calibration background's support becomes the **stripe** along the measuring line (a pixel is clear when no traced neighbour's predicted shadow at it exceeds 0.03 gl; the band along the line with most clear pixels, walked outward to the first shaded column, the node's own fitted cylinder removed; a plane usable with 500 clear pixels and 3 um of run per side), in place of the square's node-level isolation verdict. Built as SPEC Block 10.S the same evening (`03c43b2`; the notebooks and runbook v35 `6a261e4`). Why it was not S: the pilot table read off Drive -- flatness removes 26 of the 31 far-set nodes, S three, `bbar_few` none **[data]**. D-081 and D-078 annotated. Written after re-reading this log (changed since the (later, 6) row by the noise-fit chat's D-100, merged). The repository copy (D-097) on `sci/diameter-pipeline`, `docs/project/`. Next free IDs: D-102, I-005. |
| 2026-10-28 (later, 9) | Logs **D-102** and **D-103** (user, 19:35; the noise-fit chat). **D-102**: "I think you have misuderstood ... For each cell draw out per each sweep a tilt. the twin will recieve the very same tilt as in teh real cell so that no statistics has to be computed on the tilt" -- each twin sweep carries the realised tilt of its own real sweep, read on an SS sweep as the least-squares slope of its window means on time (the assistant's proposal; the LS sweeps' reading and the fit's simulator open), and no $\sigma_b(c)$ is computed for the twins; **D-100 withdrawn** as the assistant's misreading of "A and B". **D-103**: D-095's open points closed -- the per-sample median window per sweep and polarity for the start and the subtraction, the fitted decay's point value, $t_{\rm SS}(c)$ the maximum over groups with D-076 (ii)'s 80 ms floor, one exponential with a free baseline on [5, 200) ms, 80 ms also the fallback (flagged), replacing D-051 (i)'s 190 ms; open: point 9 (a failed group) and the fallback's trigger. D-034, D-051, D-076, D-095 and D-100 annotated (entries and index). Written after re-reading this log (changed since the (later, 7) row by the diameter chat's D-101, row (later, 8); merged, nothing of theirs edited). The repository copy (D-097) on `main`, `docs/project/`. Next free IDs: D-104, I-005. |
| 2026-10-28 (later, 10) | Logs **D-104** (user, 19:53; the noise-fit chat): "When the window has less than 50 ms size" -- the 80 ms fallback applies when the window $[t_{\rm SS}(c), 200)$ ms would be shorter than 50 ms (a start later than 150 ms), and also when no group is valid (the assistant's reading); "Flag and leave it out" -- a group whose fit fails or that has fewer than 5 windows (PROVISIONAL) is flagged and left out of the maximum; on the RC test cell at $\kappa_{\rm tr}$ = 0.1 the fallback applies from $\tau_m$ = 34.0 ms (113 MOhm) or $R_{\rm in}$ = 234 MOhm (28.1 ms). Roadmap allen-mtg-fit updated at the user's word ("Finally update the roadmap with the changes discussed here"): notes and revise marks for D-102 to D-104 on N3, N4, N5 and T8, no status changed (`a945a7a`). D-082, D-095 and D-103 annotated (entries and index). Written after re-reading this log (unchanged since the (later, 9) row). The repository copy (D-097) on `main`, `docs/project/`. Next free IDs: D-105, I-005. |
| 2026-10-28 (later, 11) | Logs **D-105** (user, 19:39; the diameter implementation chat): "given the far field nodes highlight the stripe used, expose also the parameters to shape the stripe's features ... Also recheck the previous implementation" -- the isolation figures show the stripe of every plane of the stage (panel (c); the far fit's planes lie 5-16 planes from $k^*$), Cell 7 exposes the stripe's existing settings as `CAL_KNOBS` (written to `calibration_config.json`, passed as `--config-json`; Cell 15 and davinci read the same file), and D-101's implementation was rechecked: the integration tester's report (F1, F5 fixed; F2 fixed as the assistant reads D-101.4, to confirm; F3, S4 open) and a run on the pilot's own reconstruction -- as built the stripe keeps 21 of the 138 near candidates and none of the 7 far ones, out-of-focus dendrites' predicted shadows covering the nodes; how they should enter the stripe is D-101 (d), open. D-101 and D-098 annotated (entries and index). Written after re-reading this log (changed since the (later, 8) row by the noise-fit chat's D-102 to D-104, rows (later, 9) and (later, 10); merged, nothing of theirs edited). The repository copy (D-097) on `sci/diameter-pipeline`, `docs/project/`. Next free IDs: D-106, I-005. |
| 2026-10-29 | Logs **D-106** (user, 14:17; the noise-fit chat): "Should a dropped group's windows also be left out of the noise statistics? Yes as you propose" -- a group with fewer than `min_windows_per_group` = 5 windows (PROVISIONAL) leaves the noise statistics and the layout handed to Block 4 as well as the start's maximum; a group dropped only for a failed fit keeps its windows. D-104 annotated (entry and index). Written after re-reading this log (changed since the (later, 10) row by the diameter chat's D-105, row (later, 11); merged, nothing of theirs edited). The repository copy (D-097) on `main`, `docs/project/`. Next free IDs: D-107, I-005. |
| 2026-10-29 (later) | Logs **D-107** (user, 14:41; the noise-fit chat), closing D-102's open points: a Long Square sweep's tilt is the line through a start-window mean and an end-window mean (the assistant's reading: the 265 ms pre-window and the last 500 ms of the stored sweep, `delta_end_mV` / 6.8875 s; error SD 0.0045 mV/s at $\theta_{\rm cf}$, `check_ls_tilt_reading.py`); the noise fit's simulator gives every simulated sweep its real sweep's tilt in every replicate, so the tilt SD, (T1) and the alternation leave the fit. D-102 annotated (entry and index). Written after re-reading this log (unchanged since the 2026-10-29 row). The repository copy (D-097) on `main`, `docs/project/`. Next free IDs: D-108, I-005. |
| 2026-10-29 (later, 2) | No decision changed (noise-fit chat). **The SS tilt reading checked** at the user's request (`check_ss_tilt_reading.py`, `Ih Fit/dev/2026-10-29_ls_tilt/`; note in D-107): D-103's median subtraction leaves the reading of (D-102.2) unchanged, and the later starts raise its RMS error from 0.0027-0.0036 mV/s (80 ms) to 0.0031-0.0041 (130 ms) and 0.0034-0.0044 (150 ms) at $\theta_{\rm cf}$ -- D-102 (c) and D-107 (a) answered. Roadmap allen-mtg-fit updated at the user's word ("Update the roadmap", 15:01): notes and revise marks for D-106 and D-107 on N2, N3, N4, N5 and T8, no status changed (`bee2abe`). D-082, D-102 and D-107 annotated (entries and index). Written after re-reading this log (unchanged since the 2026-10-29 (later) row). The repository copy (D-097) on `main`, `docs/project/`. Next free IDs: D-108, I-005. |
| 2026-09-19 (later) | Logs I-001 (human spine simulations from H01 geometry, two tiers) with the literature assessment made the same day, and its dependency on the P1 spine-statistics tables (handoff section 4.1). |

Basis tags: **[KB]** read from a project-knowledge full text; **[reasoning]**
the assistant's own derivation, hypotheses stated; **[user]** the user's
decision or report; **[open]** not decided or not verified.

## Entry format

Every decision entry has these fields, in this order: *Binds* (which stage,
file, table or parameter it constrains) — *Decision* (one sentence, the thing
to remember) — *Statement* (the precise form, with types and hypotheses) —
*Why* (the shortest argument that makes the decision non-arbitrary) — *What
it implies for code and tables* — *What it does not decide* — *Sources* —
*Status*. An idea entry has *Concerns*, *Idea*, *Why it might matter*, *What
would be needed*, *Status*.

## Index

| ID | Date | Binds | One line | Status |
|---|---|---|---|---|
| D-001 | 2026-09-19 | synaptic-parameter table (all cell types, all layers) | every synaptic peak conductance we enter is $G_{\rm syn}^{\rm shaft}$: effective, shaft-referenced, model-conditioned | active |
| D-002 | 2026-09-21 | P1 export, `p1_summary.csv`, P4 selection | QC verdicts (`pass` / `pass_low_confidence` / `fail`) are recorded by P1 and FILTERED at P4 / analysis; only `fail` withholds artefacts | active |
| D-003 | 2026-09-21 | `passive_params*.csv`, `p1_export.pbs` knobs, output trees `h01/p1_inh_SST`, `h01/p1_inh_PVVIP` | L2/L3 inh exported under BOTH passive sets (SST cm 1.0; PV/VIP cm 2.0), one tree each; which tree the EEG model uses is decided downstream | active |
| D-004 | 2026-09-22 | the campaign job scripts; what "a finished cell" means; which F is reported anywhere | one submission runs P1 -> P2 -> P3 per cell; the spine area and F of every cell are the `mesh_beyond` measurement, other variants are comparison only | active; orchestration IMPLEMENTED 2026-09-22 (`campaign.pbs`), pending push to the cluster |
| D-005 | 2026-09-22 | `Ih Fit/` mod files, `ih_mechanism.py`, `ParamSpec`, search box and GP budget of every I_h arm | six fitted parameters $(C_m, R_m, R_a, \bar g_h, \Delta v_h, \kappa_\tau)$; uniform $\bar g_h$ (Eyal law = comparison); $e_{\rm pas}$ solved so $V_{\rm rest}$ is stationary; 200/100 | active; IMPLEMENTED through Stage 7 on 2026-09-22, pending push; the recovery gate is BUILT (D-011) but has not yet been RUN |
| D-006 | 2026-09-22 | loader roles, loss windows, `run_ih_fit.py`, `submit_ih_fit.sh`, Phase 3, the comparison tables | no I_h-suppressing guards; train on $h_2..h_{n-1}$ on the step window; validate on small depolarising steps + $h_1$; $h_n$ and rebound report-only; Phase 2.5 off, Phase 3 on; baseline = run B (dt 0.1 ms, 60 ms); monolith copy edited in `Ih Fit/` with a 3-D regression gate | active; role table confirmed by D-007; IMPLEMENTED through Stage 7 on 2026-09-22, pending push; **the spike screen of Q7's implementation corrected by D-016** |
| D-007 | 2026-09-22 | same objects as D-005/D-006 | confirmations: base `Ih_human`; role table stands; dt 0.1 ms; $F=1.9$ for L2/L3 optimisation; Phase-3 100/30, B=200; every recorded hyperpolarising amplitude admitted | active; all six are the defaults of `run_ih_fit.py` / `submit_ih_fit.sh` as of Stage 6; morphology input contract closed by D-013 |
| D-008 | 2026-09-22 | `Ih Fit/` loader defaults, `param_spec.py`, every loss and refit call site | loader keeps legacy defaults (no-cap is opt-in, so the regression gate stays meaningful); every loss is variadic positional in spec order, `loss(*q)` | active; both conventions held through Stages 5-6 (the regression gate is still bit-for-bit) |
| D-009 | 2026-09-22 | `--min-coverage` and `qc_status` (what they are FOR); every table reporting an $F$ beside its mesh coverage; `campaign_policy_report.py`; `campaign.pbs` resources; `campaign_queue_check.py` defaults | the kappa fill stays and no cell is dropped -- the uncovered area fraction is REPORTED instead; `shaft_terminates` components are kept as spines; the allocation is 200 CPUs at once, 300 h walltime ceiling | active |
| D-010 | 2026-09-22 | `Ih Fit/run_ih_fit.py` ($\tau_w$ contract, `ArmConfig.gate_valid_via`), `submit_ih_fit.sh`, the `validation_status` column of every I_h arm, every liveness check in this pipeline | no per-cell $\tau_w$ sweep (a grid is refused, not collapsed); the I_h arms window the validation side of the gate like the training side; **no check may certify a loss by `isfinite` alone** -- it must be below the $10^6$ crash penalty and must vary | active |
| D-011 | 2026-09-22 | `Ih Fit/synth_gt_grid.py`, `run_ih_recovery.py`, `submit_ih_recovery.sh`, `ih_recovery_report.py`; every recovery number and the gate verdict | the Stage-7 cohort is drawn on the REAL morphologies, a quarter of it carries NO I_h as a false-positive control, the noise is run B's measured level; error is a log ratio on the five positive axes and a difference in mV on $\Delta v_h$; the I_h axes are summarised over the I_h cells only; railed axes are counted, never silently dropped | active; **(iii) superseded by D-012**; the gate has not yet been RUN; **Why (iii)'s sag number corrected by D-017** |
| D-012 | 2026-09-23 | `Ih Fit/noise_calibration.py`; the noise columns of a Stage-7 manifest (`noise_sigma_mV`, `noise_rho_lag1`, `noise_source`); `gen_from_manifest`; `run_ih_recovery.phase_noise` / `resolve_protocol`; `submit_ih_recovery.sh`; every Stage-7 verdict | each synthetic cell carries the PER-SWEEP noise $(\sigma, \rho)$ of the real cell whose morphology it borrows, read from single sweeps with the fitter's own estimator (Long Square by default); SS pulses per polarity and both sampling rates are taken from the real cells; run B's `noise_sigma_mV` (averaged bundles, mixed protocols) is not a generator input | active; supersedes D-011 (iii); **(ii) superseded and implementation of (iii) corrected by D-014**; the gate has not yet been RUN; **its law (D-012.2) replaced for measured cells, and its slow-noise open point taken up, by D-017**; **its estimator convention ((D-012.1), $r_k$) superseded by D-028** |
| D-013 | 2026-09-23 | `--swc-dir` of `run_ih_fit.py` / `load_cell_from_archive`; `SWC_DIR` of `submit_ih_fit.sh`; Stage 8's first scope | the diameter-corrected arbours arrive as one SWC per specimen, same format as the archive's `reconstruction.swc`; no scale table; every test until then runs on the raw archive SWCs | active; closes C4's open half (D-006, D-007, D-008) |
| D-014 | 2026-09-23 | `noise_calibration.py` (lookup modes, `acf_*`/`ar1_*`), manifest columns `noise_ss_*` and `acq_*`, `gen_from_manifest`, `generate_synthetic_cell(ss_noise=)`, `run_ih_recovery` (`--noise-protocol`, SS loss-weight flags), `submit_ih_recovery.sh` | each synthetic cell is its real cell's twin: generated at that cell's own sampling rates and SS pulse count, SS pulses with the SS-measured and LS sweeps with the LS-measured per-sweep noise; whether AR(1) reproduces the measured autocorrelation beyond lag 1 is reported | active; supersedes D-012 (ii); **(i) and (ii) confirmed by the user in D-015**; the gate has not yet been RUN; **its open point (AR(1) adequacy) answered NO on L3_exc, cause being measured -- D-016**; **a second, slow timescale added per cell -- D-017**; **its lag-$L$ estimator convention superseded by D-028**; **(i)'s per-protocol AR(1) pair superseded for every cell in the noise table by D-051 (one law per cell)** |
| D-015 | 2026-09-23 | the generator's noise law, the SS time-weight of the loss, the per-cell acquisition of every Stage-7 cohort | the exponential weight is the loss's alone, never the injected noise's; every synthetic cell at its own real cell's sampling rate; the noise object is the single sweep's total pre-stimulus fluctuation | active; confirms D-014 (i), (ii); nothing to change in code |
| D-016 | 2026-09-23 | `passive_fitting_hpc_fixed.sweep_spike_reason` / `sweep_has_spike` and the depolarising loop of `load_cell_from_archive` (`CellData.ls_dep_screen`); `assign_ls_roles`' warning; the LS pre-window structure columns of `noise_calibration.py`; `check_dep_sweeps.py` | a depolarising sweep is dropped from validation only for an action potential by the Allen white paper's rule (with a 5 ms time test and levels on LJP-corrected voltage), a peak above -20 mV, or an unusable trace -- never for noise; the loader records why; the LS pre-window is examined before any second noise timescale is modelled | active; corrects the implementation of D-006 Q7; adds the measurement D-014's open point needs; **(a) and (c) settled by D-017, which also corrects Why (iii)'s window-length framing (265 ms)** [(d) settled 2026-10-21 by D-073] |
| D-017 | 2026-09-28 | `synthetic_ground_truth.add_recording_noise` / `NoiseConfig` and the twin's `ls_onset_ms`; the noise columns of a Stage-7 manifest; `noise_calibration.py` (per-cell law fit); `smoke_ih_recovery.py`; every Stage-7 verdict | each twin's noise is its own real cell's fast AR(1) + slow OU (D-017.2) + per-sweep offset and tilt, fitted per cell to that cell's window statistics; no cohort law; the twin's LS pre-window is its cell's (265 ms on L3_exc) | active; answers D-014's open point and D-016 (a), (c); corrects D-016 Why (iii) and D-011 Why (iii); implementation waits for its open points (a)-(c); **its closed-form fit checked by Monte Carlo and found biased for the autocorrelations -- status note**; **five statements of its text corrected, drift evidence per cell and the `vm_delta_mv` route recorded -- status note 2**; **the creep-versus-wander diagnostic `ls_baseline_qc.py` delivered for (a) and (c) -- status note 3**; **run 2026-10-01: no creep in any cell, more variance at seconds than the cohort law -- status note 4**; **its autocorrelation convention superseded by D-028 (textbook, statsmodels)**; **(d)'s estimator family settled by D-033 (indirect inference, no ABC)**; **(c) settled by D-034 (per-sweep tilt calibrated from the SS series; enlarged SS windows)**; **(b) settled by D-051 (one law per cell, both protocols); (d)'s statistic set settled by D-051**; [updated 2026-10-19: the law's simulator and closed-form moments implemented as Block 2 of the noise fit, commit `1ab29f3` on `sci/noise-fit`; the generator's OU term (Block 5) and the per-cell fit (Blocks 3-4, 6) not yet] [updated 2026-10-20: for Block 5 -- the generator gives every SS pulse its own sweep; the SS twins need the real sweep structure (SPEC section 8), to settle with the user] [superseded in part 2026-10-21 by D-073] |
| D-018 | 2026-09-30 | diameter re-measurement from the Allen 63x stacks: Eq. 11 and local decision D4 of `handoff_diameter_remeasurement.md`; the planned `allen_image_diameter.py` and its smoke suite; the synthetic bias table $b(d, \varphi)$; the per-node CSV; the corrected SWCs of D-013 | the background $B$ of the blurred-tube fit is FIXED to a median grey level of the focal plane $k^*_i$ (D-018.1) and plugged into the fit; $(d, \alpha, v_0)$ fitted, $\sigma$ fixed | active; region of the median open (node's block with the path masked recommended); nothing coded; **$\alpha$ replaced by $\mu$ with $\varphi_i$ from Eq. 5 -- D-019**; **region closed by D-023 (block, path masked); the focus-score background (c) closed by D-023 (profile-ends median)** |
| D-019 | 2026-10-01 | the same objects as D-018: Eq. 11 / (D-018.2), the planned `allen_image_diameter.py` and its smoke suite, the per-node CSV, the synthetic table $b$ | the fitted darkness parameter is $\mu$ (absorption per µm), with $\alpha = \mu d / \cos\varphi_i$ and $\varphi_i$ from the local line fit (Eq. 5), never fitted in Eq. 11 | active; per node or shared across nodes OPEN (two-stage route recommended); nothing coded; **(a) closed by D-023: per node, stage 1 of the two-stage route** |
| D-020 | 2026-10-04 | where the diameter re-measurement workstream's documents live, how they are updated, and where the assistant reads them | canonical home is the repo folder `Passive Features/Diameter Re-measurement/docs/` on `main`; updates are committed and pushed there; the assistant refers to the repo copies | active |
| D-021 | 2026-10-06 | the whole diameter-estimation code (`src/`, `specs/SPEC.md`, the smoke suites); how open method choices enter the code | implement now, end to end, in a separate chat; every unsettled choice is a named config parameter with a labelled provisional default, confirmed in one batch; later theory results change a value or one block | active; the batch was answered the same afternoon (D-022 to D-024) |
| D-022 | 2026-10-06 | code layout and branch of the diameter pipeline; `specs/SPEC.md`; the six 2026-09-23 image modules; where the production bias table is computed | code in `Passive Features/Diameter Re-measurement/{src,scripts,tests}` on branch `sci/diameter-pipeline`; root-level `specs/SPEC.md` with Coverage = that folder; the 2026-09-23 modules imported byte-identical (sha256 verified, suites 20/20 and 40/40); production table on davinci via PBS | active |
| D-023 | 2026-10-06 | the estimator settings of the per-node fit: `mu_mode`, `bbar_region`, `focus_bg_rule`, `sigma_fit`; the configuration $\mathcal C$ of every bias table | $\mu$ per node; $\bar B_i$ = median of the node's block in $k^*_i$ with the traced path masked; focus score keeps the profile-ends median; $\sigma_{\rm fit}$ is a sensitivity axis: tables, real fits and corrections at several values, the deliverable at 0.099 µm (provisional) with the spread reported | active; closes D-019 (a), D-018 (a) and (c) [updated 2026-10-18: (iv), the $\sigma_{\rm fit}$ study, withdrawn by D-057 (ii); (i)-(iii) stand] |
| D-024 | 2026-10-06 | the renderer (`absorption`, `kernel_continuation`), the dark-tube flag, the phantom design (ranges, $U$, random search), the response estimator of the table, the tilt policy, D5/D7, node jitter | partition along vertical rays and linear continuation $\gamma = 0.79$ for now; flag $\hat\alpha > 1.0$; random $(d, \varphi, \ldots)$ draws over ranges with $U = 10$ µm, $\varphi \in [0°, 90°)$, $d$ on a temporary range until the Allen radius distribution is read; smoothing thin-plate spline for $\hat m$; every node is estimated whatever its tilt (no 30° exclusion); D5 and D7 confirmed; jitter knob at 0 until cell 13 | active; **(i)'s configured $\gamma$ to be measured -- D-026, on flat branches first -- D-027**; **(ii)'s removal of $\hat\alpha > 1.0$ from $\mathcal S$ withdrawn -- D-030**; **third table axis $\hat\alpha$ -- D-031** **(iii)'s one realization per draw: $G$ draws x $k$ realizations, laws unchanged -- D-059** [updated 2026-10-20: (v)'s dense thin-plate spline is infeasible at D-062's $N \approx 10^6$ rows (8 TB); the response estimator is to be decided before Block 14, D-062] [updated 2026-10-20, later: (v)'s response estimator superseded by D-063 (local linear kernel regression); the jitter of (vii) set from the registration survey, D-065.3] [updated 2026-10-26: (i)'s partition to be replaced by the hybrid once built, D-084] |
| D-025 | 2026-10-06 | the noise step of the renderer's camera chain (procedure §3.6 step 6), the nuisance "noise" (procedure §3.5), `RendererConfig.noise_sd_gl`, the configuration $\mathcal C$ of every table | the real stacks' noise is measured on clean background patches, in every plane, at several positions and in several stacks: its second moment (D-025.1) first, then its distribution | active; the assistant's comments (a)-(f) await the user |
| D-026 | 2026-10-06 | the kernel calibration (procedure §3.4, Eq. 4; block 10), `kernel_continuation` and its slope $\gamma$ (D-024 (i)), the choice of calibration nodes | beyond ±0.84 µm the kernel is calibrated on dendrites just under 0.8 µm (Allen diameter), read on their planes beyond 0.84 µm; the near law is continued (D-026.1) with $\gamma$ fitted, other terms only if the residuals need them; Gaussian for now | active; replaces D-024 (i)'s configured $\gamma$ once run; the assistant's comments (a)-(h) await the user; **refined by D-027 (flat nodes first)** [updated 2026-10-20: the calibration's nodes are selected by Allen's diameters, thin branches for the near field and about 0.8 um for the far field, D-061.4; Block 10 runs in Colab, its kernel table loaded into the davinci configuration, D-061.5] [updated 2026-10-20, later: the kernel is no longer Gaussian -- the SGT family in the renderer, D-066] [updated 2026-10-20, later 2: D-066 withdrawn by D-067 -- the kernel stays Gaussian, calibrated as planned on the nodes of D-061.4 / D-067.4] [updated 2026-10-28: (e)'s faint-node cut no longer applied to the calibration nodes, D-094 (a switch restores it); D-094's check covers the near set only] |
| D-027 | 2026-10-06 | the choice of calibration nodes (procedure §3.4 thin set; D-026 thick set); block 10; a later tilt study | the calibration starts with flat branches only ($\hat\varphi_i \approx 0$), near and far from focus; tilted branches come later and are compared with the renderer's prediction | active; refines D-026; the assistant's comments (a)-(d) await the user; **tilt-binned study withdrawn -- D-032** |
| D-028 | 2026-10-07 | `_estimate_noise`, `_estimate_residual_noise_at_mle`, `noise_calibration.lag_autocorr` and the noise-table columns built on it; D-017's per-cell fit; R10, R13, R14 | sample autocorrelation = textbook $\sum u_j u_{j+L} / \sum u_j^2$, computed by statsmodels `acf(adjusted=False)`; old values convert by $(n-L)/n$ | active; supersedes the estimator convention of D-012, D-014, D-017; ~~not implemented (change plan)~~ **implemented 2026-10-07 as a patch on `b355d8e`, pending push and the cluster gates (status note)** [corrected 2026-10-16: the patch was never applied; not in `main`; ships with the D-048 implementation] [updated 2026-10-18, restored 2026-10-19: implemented as Block 1, commit `b4da926` on `sci/noise-fit`, patch delivered; pending apply/push and the cluster gates] [updated 2026-10-20: on `main` as `e8ed84e`; the cluster gates pending]; **open points closed by D-029** |
| D-029 | 2026-10-07 | `Biological Fit/`; `Ih Fit/noise_stats.py`; the NaN rule of `sample_acf`; the cluster env `prova` | D-028's open points: oracle folder unchanged; no adjusted opt-in; NaN when fewer than 10 products; wrapper `noise_stats.sample_acf`; statsmodels installed in `prova` before the first run | active; ~~not implemented (handoff in the repo)~~ **implemented with D-028, 2026-10-07; (v) pending on the cluster** [corrected 2026-10-16: never applied; not in `main`; ships with the D-048 implementation] [updated 2026-10-18, restored 2026-10-19: (i)-(iv) implemented as Block 1 (`b4da926`, patch delivered); (v) still pending on the cluster] [updated 2026-10-20: on `main` as `e8ed84e`; (v) checked by G1 of `cluster_gates_noise_fit.sh`] [updated 2026-10-20, later 4: (v) holds -- statsmodels 0.14.6 in `prova` (cluster pip output, pasted by the user)] |
| D-030 | 2026-10-07 | D-024 (ii); the selection $\mathcal S$; `alpha_dark_flag`; flags and fill (procedure §3.9); blocks 6-7 | every real node gets its own corrected diameter whatever its optical depth; $\hat\alpha \le 1.0$ leaves $\mathcal S$ for phantoms and real nodes; `dark` becomes a label | active; supersedes D-024 (ii)'s exclusion and fill; the assistant's comments (a)-(c) await the user; **(b) settled by D-031** [updated 2026-10-28: carried from the optical depth to the dip by D-096 -- `faint` is a label too] |
| D-031 | 2026-10-07 | the table's regression axes; the response estimator; the inversion | the bias table also accounts for the estimated optical depth: $\hat m(d, \varphi, \hat\alpha \mid \mathcal C)$, inverted at each node's own $\hat\alpha_i$ | active; settles D-030 (b); amends D-024's axes [updated 2026-10-16: the coordinate is $\ln\hat\alpha$ -- D-053 (iii); the scaling, the third bandwidth and the budget still open] [updated 2026-10-18: scaling by design range and the third bandwidth 0.3 -- D-054 (i), (ii); the budget still open] |
| D-032 | 2026-10-07 | per-tilt tables; D-027's later tilt study; `CalibrationConfig` | tilt-specific tables are dropped, as kernel and as test | active; withdraws D-027's tilt-binned study (reading to be confirmed) |
| D-033 | 2026-10-08 | D-017 (d); the per-cell noise-law fit in `noise_calibration.py`; abcTau | per-cell $\theta_{\rm noise}$ is a least-squares point estimate on the window statistics (indirect inference); no ABC posterior, no abcTau | active; closed form vs simulated medians, statistic set and scales still open [updated 2026-10-16: scales settled by D-048/D-049/D-050 ($\hat\Sigma^{-1}$ by iteration); statistic set settled by D-051 (simulated medians, $d$ = 15)] |
| D-034 | 2026-10-08 | D-017 (c), (e); the tilt $b$ and `drift_sigma_mV`; the SS noise window of `noise_calibration.py`; D-033's statistic set | the variance at seconds is carried by the per-sweep tilt, $\sigma_b$ calibrated per cell from the SS window-mean semivariogram (D-034.1); no third OU, no per-sweep multi-OU/BIC; SS noise windows $[t_0, 200)$ ms after the previous pulse where the transient test passes | active; $h^*$ (0.8 s recommended, note) and the test threshold open [updated 2026-10-16: the enlarged windows join the statistic set, replacing the 10 ms window; the response is subtracted per polarity; $h^*$ = 0.8 s, $\kappa_{\rm tr}$ = 0.1 and 5 ms bins are labelled provisional defaults -- D-051] [note 2026-10-26: 4 periods also right for the 120 ms windows of $t_{\rm SS}$ = 80 ms; the 2-sweep spread of $\hat\sigma_b$ is the data's limit; per-cell, cohort and pooled $\sigma_b$ compared -- note in the entry] [updated 2026-10-28 (later, 7): for the 2-sweep cells per cell and one cohort value both, D-100] [updated 2026-10-28 (later, 9): the twins carry the realised per-sweep tilts, no $\sigma_b(c)$ for them -- D-102; D-100 withdrawn] |
| D-035 | 2026-10-07 | the focus rule of the diameter fit (procedure Eq. 1, handoff Eq. 1); `MeasureConfig.focus_rule`, `focus_grad_window`, `focus_grad_margin_um`; `analysis/focus.py`; the pilot's `k_star`, `k_star_depth` | the sharpest plane $k^*$ is the plane of maximum gradient energy of the profile; the dip depth is kept as the labelled comparison | active; logged 2026-10-08; cited in the repository as "D-030" (collision with this log's D-030); its window open on trunks |
| D-036 | 2026-10-08 | `scripts/plane_differences.py` (default evaluation), `focus.profile_areas`, notebook Cell 4e; not the focus rule | the plane-to-plane diagnostic evaluates the area under the measuring-line profile per plane and its change between consecutive planes | active; a diagnostic |
| D-037 | 2026-10-08 | `scripts/plane_differences.py --evaluation entropy`, `focus.plane_entropies`, `profiles.stripe_mask`, notebook Cell 4f; not the focus rule | the plane-to-plane diagnostics include the entropy of the grey-level histogram along the measuring line and in a strip around it | active; a diagnostic; its parameters provisional; pick = global minimum since 2026-10-08, evening (note under the entry) |
| D-038 | 2026-10-08 | the measuring line's half-length: the diagnostics' `--profile-half-um` (Cells 4e, 4f); Block 5's `measure.profile_half_um` and `block_half_um` if extended | the measuring line is enlarged beyond $\pm 3$ um, to $\pm 4$ or $\pm 5$ um | decided in principle [user]; $\pm 5$ um for the entropy (D-039); Block 5's line open |
| D-039 | 2026-10-09 | the entropy evaluation at `--profile-half-um 5` (D-037, D-038); `scripts/plane_differences.py --nodes thin`, notebook Cell 4g; not the focus rule | the entropy along a $\pm 5$ um line and in its strip is the plane-to-plane measure carried forward, tested next on thin dendrites against $k^*$ | active; the thin-dendrite test pending [updated 2026-10-09: run, see D-040] |
| D-040 | 2026-10-09 | `scripts/plane_differences.py --evaluation gradient` and `--evaluation blend`, `--nodes bydiameter`; notebook Cells 4h, 4i; not the focus rule (D-035) | the gradient energy over the whole measuring line is compared at $\pm 3$, $\pm 5$ um and at full width $m \hat d$ ($m = 2$), and on that line against a sigmoid-weighted min-max blend with the strip entropy and the strip entropy alone | active; implemented (commit `8a11f1e`, Cells 4h, 4i) [updated 2026-10-09, evening]; ~~not run on Allen's data~~ [corrected 2026-10-10: run on 13 nodes on 2026-10-09; Cell 4i's entropy strip and J's form changed by D-041] |
| D-041 | 2026-10-09 | `scripts/plane_differences.py --evaluation blend` (`--entropy-half-um`), `focus.blend_scores`, `figures.blend_figure`, `figures.picks_summary_figure`; notebook Cell 4i; not the focus rule (D-035) | the blend's entropy comes from the $\pm 5$ um strip, $G$ stays on the d-line; $J = w g - (1 - w) h$ with $h$ the min-max rescaled entropy (0 at its lowest), the entropy drawn as it is; the sigmoid unchanged | active; implemented (commit `be878dc`); ~~Cell 4i's re-run on Allen's data pending~~ [corrected 2026-10-10, later: re-run on 13 nodes at 13:20 UTC; within one plane of $k^*$ on the 11 non-trunk nodes: the $\pm 5$ um strip's entropy 2, the blend 7, $G$ on the d-line 10] |
| D-042 | 2026-10-10 | `scripts/plane_differences.py --evaluation iterate` (`--line-mult`, `--max-rounds`), `plane_differences.iterate_planes`, `refit_at_plane`, `gradient_iteration`, `figures.iteration_figure`; notebook Cell 4j; not the focus rule (D-035) | $G$ over the whole line, on a line sized first from Allen's diameter ($\pm m r_{\rm Allen}$), then from Block 5's fit at the plane the previous line picked ($\pm m \hat d / 2$), until a plane repeats or after 5 rounds | active; implemented (commit `2bcc8f4`); ~~not run on Allen's data~~ [corrected 2026-10-12: run on 13 nodes, all converged; carried forward by D-043] |
| D-043 | 2026-10-12 | the plane and the diameter carried forward from the focus diagnostics of D-036 to D-042 (`scripts/plane_differences.py --evaluation iterate`, Cell 4j); Block 5 (D-035) and the bias table only once (a) below is settled | the Allen-guided iteration of D-042 is the method used from now on: per node the last plane `k_iter` and the fit there, `d_iter_um` | active; where it enters the measurement open [updated 2026-10-12, later: $m$ = 3, D-044] [updated 2026-10-16: (a)'s first half settled by D-052 -- inside Block 5, every node, phantoms included; the phantoms' start radius open] [updated 2026-10-16, later: the start radius is decided by the sensitivity check of D-053 (i)] |
| D-044 | 2026-10-12 | `plane_differences.ITER_LINE_MULT` (3), the default of `gradient_iteration` and of `--evaluation iterate`; notebook Cell 4j (`LINE_MULT`, `pilot/planeiter/bydiameter_m3/`); not the d-line of Cells 4h and 4i (D-040: 2, `D_LINE_MULT`) | the iteration's lines are three diameters wide, $h = 3 d / 2$ (round 1: $\pm 3 r_{\rm Allen}$), and Gd is compared at the same $m$ | active; implemented (commit `941fc75`); ~~not yet run on Allen's data~~ [corrected 2026-10-13: run on Allen's data, `pilot/planeiter/bydiameter_m3/` written 12:25-12:27 UTC (Drive listing, metadata only)] [updated 2026-10-13, later: the same `k_iter` as at $m$ = 2 on 12 of 13 nodes; node 8462 47 to 46, d 2.38 to 2.73 um] |
| D-045 | 2026-10-12 | `scripts/iteration_fits.py`, `plane_differences.refit_at_plane(..., with_profile=True)`, `figures.iteration_fit_figure`, `figures.iteration_fits_overview`; notebook Cell 4k; a figure, no measurement | per node of a Cell 4j run, the plane `k_iter` with the line that chose it (the last round's, through the SWC node) and the diameter fitted there as a caliper between its edges, beside the profile and the model; an overview and one figure per node; by default the run at $m$ = 3 | active; implemented (commit `128fdce`); ~~not yet run on Allen's data~~ [corrected 2026-10-13: run on the 13 nodes of Cell 4j at $m$ = 3] |
| D-046 | 2026-10-13 | `model/blur_kernels.py`, `tube_model.model_profile_kernel`, `analysis/fit_variants.py`, `scripts/fit_variants.py`, `figures.fit_variants_node_figure`, `figures.fit_variants_summary_figure`; notebook Cell 4l; a diagnostic, no measurement; Block 5 keeps its Gaussian (D-026) | at Cell 4j's $k^{\rm it}$, Block 5's profile fitted with 10 blur kernels (Gaussian, Lorentzian, Voigt, Student, generalized normal, skew-normal; shapes fixed or fitted; each of unit area, FWHM $w$, peak at $v_0$) x $w$ fixed at 0.233 um or fitted x unweighted or weighted by $\exp(-|v - v_0| / \lambda)$, $\lambda$ = 0.25-2 um, re-centred on the fit's own $v_0$ (IRLS) | active; implemented (commit `4759c5f`); ~~not yet run on Allen's data~~ [corrected 2026-10-14: run on Allen's data, the user's figures of 2026-10-14]; one sentence of its Why [corrected 2026-10-15]; a Monte Carlo of the fitted width in its status [updated 2026-10-15] [updated 2026-10-16: (a) settled by D-052 -- no variant replaces Block 5's fit; D-052's does] |
| D-047 | 2026-10-14 | `scripts/variant_iteration.py`, `fit_variants.PlaneFits`, `plane_differences.gradient_iteration(..., fit_at=...)`, `figures.variant_stack_figure`, `figures.variant_planes_figure`; notebook Cell 4m; a diagnostic, no measurement | each of the 50 fitted-width variants (10 kernels x unweighted and $\lambda$ = 0.25-2 um) runs its own Allen-guided iteration with Cell 4j's $m$ and round limit read from its records (the user's "inherited"), each later line sized from the variant's own fit; Cell 4j's own iteration redone as the reference; per node a 50-row stack of the edges $v_0 \pm d/2$ and the image of each variant's plane with its line, the fitted dendrite's edges and $d$ | active; implemented (commit `9e3296c`); not yet run on Allen's data; how to read a small $d$ at $\mu$'s bound [updated 2026-10-15] [updated 2026-10-16: its open point settled by D-052] |
| D-048 | 2026-10-16 | D-033's open (d): the weights of the per-cell law fit in `Ih Fit/noise_calibration.py` (or the module beside it); the fit's convergence rule; the in-silico sensitivity check of the implementation | $W = \hat\Sigma^{-1}$, $\hat\Sigma$ the covariance of the statistic vector over $R$ cells simulated at the current $\hat\theta$, iterated from a first step weighted by the real data until the objective plateaus; sensitivity to an uncaptured noise component measured in silico during implementation | active; not implemented; **$W_0$ settled by D-049; $R$, $M$, the stop rule and $k_{\max}$ by D-050; the statistic set by D-051; the sensitivity check designed and its predictions fixed by D-055** |
| D-049 | 2026-10-16 | D-048 step (i): the first-step weighting $W_0$ of the per-cell law fit (`Ih Fit/noise_calibration.py` or the module beside it); the closure check S0; the `noise_fit_weighting` column | $W_0 = \mathrm{diag}(1/\widehat{\mathrm{SE}}_m^2)$, $\widehat{\mathrm{SE}}_m$ = SD of $B$ = 2000 window-level bootstrap medians of the cell's own $K_m$ per-window values; used only for $\hat\theta_0$, then $W_{k+1} = \hat\Sigma_k^{-1}$ (D-048); the hand-set $\omega_m$ stay the labelled comparison | active; not implemented |
| D-050 | 2026-10-16 | the D-048 iteration's Monte Carlo sizes and stop rule: `NoiseFitConfig` (`R_iter`, `R_final`, `M`, `tol_theta`, `k_max`), the per-cell seeds, the `noise_fit_iters` / `noise_fit_R` / flag columns; the closure check S0 | $R$ = 600 while iterating, one final step at $R$ = 3000 reported; $M$ = 3000 CRN windows per statistic; innovations fixed per cell across iterations; stop when $\max_i |\Delta q_i| \le 0.01$ in $(\log\sigma_f, \rho_f, \log\sigma_s, \log\tau_s)$, $J$ reported not tested; $k_{\max}$ = 6, tunable; at the cap keep the last iterate, flag `no_plateau` | active; not implemented [updated 2026-10-16: Why (4)'s 1.2 ms does not carry over to $\hat\Sigma^{-1}$; the seed-to-seed spread of $\hat\theta$ at these sizes is to be measured in S0] [updated 2026-10-18: optimiser settled by D-056 (iii)] |
| D-051 | 2026-10-16 | the statistic vector $\hat{\mathbf q}(c)$ and the estimator module of the per-cell fit; D-017 (b) and (d); D-014 (i); D-034 (ii) and its open points; the fit's simulator (binding function and $\hat\Sigma$); the SS loader; `NoiseFitConfig` (`kappa_tr`, `transient_bin_ms`, `h_star_s`) | $d$ = 15: enlarged SS $[t_{\rm SS}(c), 200)$ ms ($\hat\sigma$, $\hat\rho$ 0.02/0.1/1/10 ms) replacing the 10 ms window, plus the ten LS statistics; fallback $d$ = 14, flagged; one law per cell for SS and LS; the tilt in the simulator at $\hat\sigma_b$ from (D-034.1), alternated per iteration; the per-polarity mean response subtracted from each enlarged window on data and simulation alike; $\kappa_{\rm tr}$ 0.1, 5 ms bins, $h^*$ 0.8 s provisional | active; not implemented [updated 2026-10-18: (iii)'s simulated sweeps are continuous traces with the real pulse timing -- D-056 (i)] [(ii) superseded in part 2026-10-21 by D-073] [updated 2026-10-28 (later, 9): (i)'s fallback 190 ms replaced by 80 ms, flagged -- D-103] |
| D-052 | 2026-10-16 | Block 5's plane and final fit (`node_pipeline.measure_node`), so every real node's and every phantom's $\hat d$ and the bias table of D-024/D-031; the iteration's functions (to move from `scripts/plane_differences.py` into `src/`); `blur_kernels` (a new family), `fit_variants._Layout`; `MeasureConfig` and the estimator hash; $\mathcal S$'s bound rule | per node: the Allen-guided plane (start $2r$, lines $\pm 1.5 d$, $k_{\rm SWC} \pm 6$, at most 5 rounds), each round's $d$ and the final $\hat\theta$ from the skewed generalized-$t$ fit with $(d, \mu, v_0, w, \beta, \xi, \zeta)$ all fitted and centre weights $\lambda$ = 0.5 um re-centred by IRLS; real nodes and phantoms alike; the error left is the table's | active; not implemented [one sentence of its Why corrected 2026-10-16] [updated 2026-10-16, later: (b) settled by D-053 (ii); (a) by the check of D-053 (i)] [updated 2026-10-18: the per-solve budget settled by D-054 (iv)] [updated 2026-10-18, later: D-052.4 confirmed by the user -- D-057 (iii)] [updated 2026-10-18, later: (D-052.1)-(D-052.2) implemented, Block 12, commit `bee1e0b` (local); the plane rule inside Block 5 is Block 13, not yet] [updated 2026-10-18, later 4: (a), the phantoms' radius, settled by D-058 instead of D-053 (i)'s check] [updated 2026-10-20: Block 13's implementation points settled by D-061 ((a), (c), (d) as recommended; (b) by Allen's diameters)] [updated 2026-10-20, later: (f) settled by D-066 (the renderer's kernel is the SGT family); (g) not run for now, D-065.4] [updated 2026-10-20, later 6: (D-052.3) implemented with (D-052.2) inside the measurement, Block 13 (`measure_node` = the pilot, the iteration, `plane_iteration.sgt_at_plane` at $k^{\rm it}$; commits `b5191c8`-`70136ca`, local, not pushed)] [updated 2026-10-21: pushed at the user's agreement, `84ea91f..888452a` on `sci/diameter-pipeline`] |
| D-053 | 2026-10-16 | the phantoms' radius (`phantoms.phantom_branch`, the four places of the chain it enters); the bound rule of $\mathcal S$ (`phantoms.reject_reasons`, `fit_status`); the third regression axis of the table (`table.table_inputs`, `BiasTable`, `invert.invert_node`); Blocks 13-14 | (i) a sensitivity check on rendered phantoms decides the phantoms' radius (true $d/2$ if $k^{\rm it}$ and $\hat d$ do not move with the radius, else Allen-like draws), with in-focus images of the measured diameters; (ii) $\mathcal S$ fails on $d$, $\mu$, $v_0$ or $w$ at a bound or an unconverged IRLS, never on $\beta$, $\xi$, $\zeta$ at theirs; (iii) the third axis is $\ln\hat\alpha$ | active; not implemented; the check of (i) to run before Block 14  [updated 2026-10-18: Q4b, Q4c, the threshold and the budget settled by D-054] [updated 2026-10-18, later: (ii) implemented as Block 12's status rule, `bee1e0b`] [updated 2026-10-18, later 4: (i) superseded by D-058] [updated 2026-10-20, later: (iii)'s coordinate amended by D-063.1 -- the third axis is $\ln(\hat\mu\hat d)$] |
| D-054 | 2026-10-18 | `table.table_inputs` and `BiasTable` (the input scaling, stored with the table); `CorrectionConfig.tau_kernel_bandwidth` (three values), `max_failure_rate`; `MeasureConfig` (the new per-solve budget field, in the estimator hash); Blocks 12 and 14 | (i) each of $(\ln d, \varphi, \ln\hat\alpha)$ divided by its design range before the spline, CV on the smoothing only; (ii) bandwidth 0.3 in $\ln\hat\alpha$, PROVISIONAL; (iii) `max_failure_rate` 0.2 PROVISIONAL until the pilot's curve is read; (iv) `max_nfev` = 3000 per solve | active; not implemented [updated 2026-10-18, later: (iv) implemented in Block 12, `bee1e0b` -- entered on D-056's row by mistake, moved here 2026-10-19] [updated 2026-10-20: the smoothing grid extended to $10^4$ with a flag at either end, D-062.4; (iii) stands, item 14 closed] [updated 2026-10-20, later: (i)'s normalisation retired by D-063 (the estimator works in raw coordinates); (ii)'s bandwidths are now the estimator's, the third carried to $\ln(\hat\mu\hat d)$] |
| D-055 | 2026-10-18 | the in-silico sensitivity check of D-048: `sens_s0_closure.py`, `sens_s1_tilt.py`, `sens_s2_missing.py`, `sens_s3_real.py` (names provisional) and their read-out; the alarm threshold and $k_{\max}$ they calibrate | S0 closure on two real cells' window structures (537297470, 569844159), 150 twins each; S1 seconds-scale OU twins (0.142 mV, 3 s; 0.293 mV, 30 s), 100 each; S2 sweep-locked transient, +0.1/1/10 % scatter, SS response jitter 2/5 %, Bessel fast part, 100 each; S3 the 18 L3_exc cells; predictions S0.1-S3.3 fixed in the entry; pilot here at $R$ = 100, $M$ = 500, production on the cluster | active; not run |
| D-056 | 2026-10-18 | the noise-fit simulator's SS sweep structure; `Ih Fit/specs/SPEC.md`; branch `sci/noise-fit` and the delivery format; the inner optimiser of `noise_fit.py` (`NoiseFitConfig.n_starts`, the binding-function estimator) | continuous simulated SS sweeps with the real pulse timing, cut at the real windows; spec in `Ih Fit/specs/SPEC.md`, patch series + tarball from a local baseline of the zip; Harrell-Davis median in the binding function with SciPy's default finite-difference Jacobian, `method='trf'` with bounds, three starts (closed form, $\tau_s/2$, $2\tau_s$), best $J$ | active; not implemented ~~[updated 2026-10-18, later: (iv) implemented in Block 12, `bee1e0b`]~~ [corrected 2026-10-19: that annotation is D-054 (iv)'s (`sgt_max_nfev` = 3000, the diameter pipeline's Block 12), entered on this row by mistake; D-056 has parts (i)-(iii) only and none is implemented] [updated 2026-10-19, later 2: (i)'s simulator implemented as Block 2, `1ab29f3`; (ii) in force since Block 1; (iii) is Block 4's] [updated 2026-10-20: the sweep-number open point answered at source -- the archive writer stores `sweep_number` per pulse; the files to confirm by G12] |
| D-057 | 2026-10-18 | `CorrectionConfig.failure_rate_ignore`; `MeasureConfig.sigma_fit_study_um`, `scripts/build_table.py --sigma-fit`, `run_cell.py --sigma-fit`, the `d_tilde_sigma_spread_um` column, notebook Cell 8; procedure §3.7 as applied to Blocks 5-6 (`phantoms.run_replicate` through `measure_node`) | (i) `failure_rate_ignore = ()`; (ii) the $\sigma_{\rm fit}$ study of D-023 (iv) dropped -- one table, one real-node run, at 0.099 um; (iii) phantoms measured by exactly the real-data pipeline, plane choice and profile extent included [user] | active; not implemented; supersedes D-023 (iv) [corrected 2026-10-19: its Binds named two CLI flags that do not exist] [updated 2026-10-21, later: (ii) implemented, S2.2, `a879067`, pushed at the user's agreement (`f4ac106..9c1329e` on `sci/diameter-pipeline`)] |
| D-058 | 2026-10-18 | `phantoms.phantom_branch` (the radius a phantom's branch carries) and the places of the chain it enters (handoff §6); `PhantomConfig` (`radius_noise_sd_um`) and the simulation hash; D-053 (i) | each phantom starts from $r_n = d_n/2 + \epsilon_n$, $\epsilon_n \sim \mathcal N(0, \sigma_r^2)$ [user]; $\sigma_r$, the scope (line start or all four places) and the $r_n \le 0$ rule open | active; not implemented; supersedes D-053 (i) [updated 2026-10-20: settled by D-062.1 -- $\sigma_{\rm br}$ = 0.2 um, all four places, redraw for $r \le 0$, per realization; the sensitivity check dropped] |
| D-059 | 2026-10-18 | the phantom design of D-024 (iii) (`phantoms.draw_replicate`, `PhantomConfig`, the replicate rows, `build_table` chunking); what one element of the table is (`table.fit_table`); the failure-rate count; Q6 | the D-024 (iii) laws stay, with an $\hat\alpha$ coverage check against the real cells; $G$ independent draws of $(d, \varphi, \theta, \mu)$, each realized $k$ times with fresh nuisances, the per-draw estimate the mean ratio over its realizations in $\mathcal S$ [user]; $G$, $k$ and the draw's $\hat\alpha$ coordinate open | active; not implemented; supersedes D-024 (iii)'s one-realization design (laws unchanged) [updated 2026-10-18, later 5: a draw's third coordinate settled by D-060 (A)] [updated 2026-10-20: $G \approx 10^5$, $k$ = $n_{\rm real}$ = 10 and the failure count at the true $\alpha_g$, D-062; the coverage check's form still open] [updated 2026-10-20, later: the coverage check is D-065.1's Colab survey] |
| D-060 | 2026-10-18 | `table.fit_table`, `BiasTable`, `table.choose_smoothing` (which rows the spline sees, at which coordinates; its folds); the per-draw summaries of D-059; Block 14 | the spline through all $G k$ realizations in $\mathcal S$, each at its own $(\ln d_g, \varphi_g, \ln\hat\alpha_{g,r})$; the per-draw means $\bar y_g$ for the per-draw scatter and the checks [user]; CV folds by draw proposed, not decided | active; not implemented; settles D-059's third-coordinate point [updated 2026-10-20: folds by draw, D-062.3; at D-062's $N \approx 10^6$ the dense spline through every realization is infeasible -- the estimator is to be decided, D-062] [updated 2026-10-20, later: the spline through every realization superseded by D-063.2; the conditional target and the per-draw table stand] |
| D-061 | 2026-10-20 | Block 13 (`measure_node`: the provider call, the pilot/iteration split, the flags' plane; the Phase II diagnostics and their suites); Block 10's node selection (`calibration_reasons`, `node_dhat_max_um`); the kernel table's hand-over from Colab to the davinci configuration | (a) one provider call over the pilot's block and $k_{\rm SWC} \pm 6$ plus one per restack; (c) pilot and iteration + SGT fit as separate functions, one chain for phantoms and real nodes, the diagnostics call the pilot; (d) $\mathcal S$'s flags on the fitted plane $k^{\rm it}$; (b) Block 10 selects its nodes by Allen's diameters (thin branches for the near field, about 0.8 um for the far field, D-026); (f) the kernel is calibrated in Colab, the draws, the table and its tests run on davinci [user] | active; not implemented; (e) the iteration's background scene input, the Allen thresholds and the other criteria's values open [updated 2026-10-20, later: (e) settled by D-064] [updated 2026-10-20, later 2: (b)'s thresholds set PROVISIONAL by D-067.4] [updated 2026-10-20, later 6: (a), (c), (d), (e) implemented, Block 13 (commits `b5191c8`-`70136ca`, local, not pushed); the pilot reads only Block 5's planes of the one block, so it equals Block 5 at `84ea91f` bit for bit; (b) and (f) are the plan's S2 and S3] [updated 2026-10-21: pushed at the user's agreement, `84ea91f..888452a` on `sci/diameter-pipeline`] [updated 2026-10-21, later: (b) implemented as D-067.4 sets it, S2.1, `1aed5d6`, pushed at the user's agreement (`f4ac106..9c1329e` on `sci/diameter-pipeline`); the far-field set is selected but not scanned] |
| D-062 | 2026-10-20 | `PhantomConfig` (`radius_noise_sd_um`, `n_draws`, `n_realizations`), `phantom_branch`, the generator keys, the four places the radius enters; `choose_smoothing`'s folds, `spline_smoothing_grid`; the failure count's coordinate; `build_table` sizing; Block 14; the response estimator of D-024 (v)/D-060 (raised) | $\sigma_{\rm br}$ = 0.2 um, in all four places, redraw for $r \le 0$, per realization; $G \approx 10^5$ x $n_{\rm real}$ = 10; CV folds by draw; grid to $10^4$ with a flag at either end; every realization in the failure count at its true $\alpha_g$; items 13, 15 dropped, 14 closed [user] | active; not implemented; opens the response estimator at $N \approx 10^6$ (the dense spline infeasible: 8 TB, $\sim 2 \times 10^5$ h) [run] [updated 2026-10-20, later: the estimator settled by D-063] [updated 2026-10-20, later 6: a consequence of (D-062.1) found with Block 13 -- round 1's line has fewer than 3 samples, and the phantom ends `line_short` (outside $\mathcal S$), when $r_{\rm br} < 0.5\,\Delta v/m$ = 0.019 um: under the redraw, 5.0 % of the phantoms at $d$ = 0.2 um, 2.9 % at 0.4 um, 0.2 % at 1.0 um, while a real node starts from Allen's radius **[run, sandbox: `checks/block13_2026-10-20/check_line_short.py`]**; raised with the user for Block 14. The per-realization time at the defaults, 48.4 s mean on 10 phantoms (d 0.3-3.5 um, phi 0 and 45 deg; a lower bound for this design), puts $10^6$ realizations at $\ge 1.3 \times 10^4$ CPU-hours **[run, sandbox: `time_13_5.py`]**] [updated 2026-10-21: the thin end settled by **D-069** [user]: round 1's line floored at 3 samples, the start radius unchanged] |
| D-063 | 2026-10-20 | the response estimator (`response_estimator`, `table.fit_table`, `BiasTable`, `table_io`); the SE, $\hat\tau$, $N^{\rm loc}_{\rm eff}$, $\hat p_{\rm fail}$ of SPEC §2.4; Block 7's flags (`off_support`); the table's coordinates (`table_inputs`, `tau_kernel_bandwidth`); Block 14 | local linear kernel regression on a grid with D-054's bandwidths is the deliverable at $N \approx 10^6$; a penalized tensor-product B-spline the labelled comparison; the SE the smoother's own with the by-draw sandwich; `off_support` when the kernel mass at the root is below the rows' leave-one-out 5th percentile; the third axis $\ln(\hat\mu\hat d)$ in place of $\ln\hat\alpha$ [user] | active; not implemented; supersedes D-024 (v) and D-060.1's spline, amends D-053 (iii), retires D-054 (i) [updated 2026-10-20, later 2: (D-063.5)'s P-spline comparison dropped by D-067.3; the local linear estimator alone] |
| D-064 | 2026-10-20 | `measure_node` (a `scene` argument), the moved `block_background`, `survey.node_planes`, `cell`, `phantoms.run_replicate`; Block 13 | the iteration's background $B_{j,k}$ masks the cell's traced segments and soma passed in as a scene input, empty on a phantom; $\bar B_j(k)$ keeps its own-branch mask [user] | active; not implemented; settles D-061 (e) [updated 2026-10-20, later 6: implemented, Blocks 13.2-13.3 (`plane_iteration.Scene`, `cell.cell_scene`; commits `e50abd9`, `70136ca`, local)] [updated 2026-10-21: pushed at the user's agreement, `84ea91f..888452a` on `sci/diameter-pipeline`] |
| D-065 | 2026-10-20 | `PhantomConfig.d_range_um`, `mu_range_per_um` (kept), `jitter_xy_um`, `jitter_z_um`; the coverage check of D-059.1; cells 3b and 6 and a new survey cell of the notebook; Q7 | ranges kept; the coverage check is a Colab survey of the real neurons' $\hat\alpha_i$ and $\hat\mu_i\hat d_i$ across cells per bin of $\hat d$; Allen's radii across cells set $d_{\max}$; the jitter SDs from the registration survey; the pinned-kernel comparison not run [user]; the assistant recommends the surveys before the production launch | active; not implemented; values after the Colab runs [updated 2026-10-20, later 2: the recommended order is the decision, D-067.1] [updated 2026-10-26: "all the cells" is now 5 cells per class and layer, all when fewer; the kernel per cell; D-085] [updated 2026-10-26, later 6: $d_{\max}$ = the maximum of Allen's dendrite $2r$ across S3's cells, D-086] |
| D-066 | 2026-10-20 | Block 4's kernel (`kernel_family`, the kernel table, the FFT path), Block 10's calibration and Colab cell 7, gate 1, the simulation hash; D-026, D-052 (f), I-002 | the renderer's defocus kernel is the symmetric SGT family, its isotropic PSF the inverse Abel transform of the profile kernel (nonnegative for every member [run]), $(w, \beta, \xi)$ tabulated against defocus and calibrated in Colab on the D-061.4 nodes; the skew not rendered, read as a diagnostic; both chains share the family [user] | active; not implemented; adopts I-002; supersedes D-026's Gaussian "for now" and D-052 (f) [updated 2026-10-20, later 2: **withdrawn by D-067** before implementation; the Abel finding kept on record] |
| D-067 | 2026-10-20 | the schedule before the production launch; Block 4's kernel and Block 10 (D-026, D-027, I-002); `response_estimator`; `CalibrationConfig` (`node_allen_diameter_max_um`, `far_field_allen_diameter_um`) | Block 13 -> Colab (the real-cell survey, radii, registration, the Gaussian kernel's calibration) -> the final configuration -> Block 14 -> gate 2 -> davinci pilot -> production; the renderer stays Gaussian, the fit SGT (D-066 withdrawn); only the local linear estimator is built; thin <= 0.6 um and far field 0.6-0.8 um in Allen's diameter, PROVISIONAL, the other criteria from the pilot [user] | active; not implemented; supersedes D-066 and D-063.5 [updated 2026-10-21, later: (D-067.4) implemented, S2.1, `1aed5d6`, pushed at the user's agreement (`f4ac106..9c1329e` on `sci/diameter-pipeline`)] [updated 2026-10-23: the calibration nodes' criteria gain the patch isolation of D-079.1] [updated 2026-10-28: (D-067.4)'s faint criterion and, for the calibration nodes, the flag `faint` of "in $\mathcal S$" superseded by D-094 (both off by default)] |
| D-068 | 2026-10-21 | `node_pipeline.deliverable_flags` and `measure_node` (the flags of a node whose iteration does not converge); Block 6's `reject_reasons` and `SELECTION_FLAGS` (unchanged); the flags column of the replicate rows and the per-node CSV; Block 14's failure count | an iteration ending `cycle`, `max_rounds` or `no_plane` after round 1 keeps the fit of its last plane and stays in $\mathcal S$ when the fit's status and the other flags allow; every ending but `converged` adds the label `iter_<ending>` to the flags [user: "Keep but flag"] | active; not implemented; closes SPEC Block 13.3b's [to confirm] [updated 2026-10-21, later: implemented, Block 13.6, `f4ac106`, pushed] |
| D-069 | 2026-10-21 | `plane_iteration.iterate_planes` / `gradient_iteration` (round 1's line), `measure_node`, `MeasureConfig.iter_min_first_samples`, the estimator hash; the thin end of D-062.1's phantoms | round 1's line has half-length $\max(m d^{(0)}/2,\ h_{\min})$, $h_{\min} = \tfrac{n_{\min}-1}{2}\Delta v$, $n_{\min}$ = 3, so it holds at least 3 samples; later rounds and the start radius unchanged [user: "b"] | active; not implemented; settles the thin end of D-062.1 found on 2026-10-20 [updated 2026-10-21, later: implemented, Block 13.6, `f4ac106`, pushed] |
| D-070 | 2026-10-21 | the plan's S2b, S3 (d) and S4; Block 10's far-field set (`FAR_FIELD_NOT_BUILT`); `kernel_continuation_slope` until S2b | the far-field calibration of D-026 is stage S2b, designed after Colab's Cell 7 counts the far-field nodes, $\gamma$ = 0.79 configured until then; S4's code written while S3 runs [user] | active; S2b waits for Cell 7; S4 in progress [updated 2026-10-21, later 4: S4 done and pushed (`825bef1`); S2b's timing amended by D-071.5, its far scan's planes and line set by D-074] |
| D-071 | 2026-10-21 | `CorrectionConfig.bandwidth_cv_grid`, `table.choose_multiplier`, `table.grid_axes`, the table's SE (`se_method`), `invert.invert_node`'s labels; the plan's S2b and its order | Block 14's choices: the multiplier $c$ by cross-validation with folds by draw, the sandwich by draw the one SE, the grid's spacing at most $c\,h/2$ at the chosen $c$, `mu_d_out_of_domain` a label (clamped, not filled); the far-field calibration (S2b) straight after Block 14 | user [implemented, Blocks 14.3-14.6; pushed at the user's agreement, `f63fe9f..825bef1`; (D-071.5) in progress, D-074] |
| D-072 | 2026-10-21 | Block 3's SS window set and layout; Block 4's simulator; Block 5's twins; SPEC sections 3, 8 | the stored SS windows (trace-QC'd when archived) are the data as written; no shape test anywhere in the noise fit, the twins included; each window's place in its 37-pulse train from the archive's `trace_qc_results.csv` | active; user |
| D-073 | 2026-10-21 | D-017 and D-051 (ii) as generator inputs; D-016 (d); Blocks 3-5; the twins' protocol | each twin sweep is the twin of one real sweep and carries noise calibrated on that sweep, not a per-cell average; which parameters per sweep and which per cell is open (the assistant's proposal: the fast pair and the slow amplitude per sweep, the slow timescale and the tilt per cell) | active (intent), form open; user |
| D-074 | 2026-10-21 | SPEC Block 15 (Q-A to Q-D); `CalibrationConfig`'s far fields; the far scan and far fit of `analysis/calibration.py` (not built); `kernel_continuation_slope` | the far scan reads planes $k^*_j \pm 16$ on both sides with an SNR rule at 3, on a +-10 um line [user]; the line-end baseline accepted with it has no samples past plane 9, so a planar background from far pixels is proposed (the assistant's correction); the far fit (sequential, both sides, axis free, two-pass statistic, model weights) explained, not decided | active; (D-074.1)-(D-074.2) settled [user]; (D-074.3)-(D-074.4) and Q-D await the user [updated 2026-10-23: the far statistic's passes set by D-077 -- iterative, capped, stopped within one SE] [updated 2026-10-23, later: (D-074.3) and (D-074.4) settled by D-078 -- background (a), the iterative core fit with N = 10 and kappa = 1, fit (a); Q-D and the junction open] [updated 2026-10-23, later 2: Q-D's isolation half settled by D-079 -- no traced point but the node's own branch in the +-25 um patch, near and far nodes; the tilt half and the junction open] [updated 2026-10-23, later 3: the junction settled by D-080 (table cut at 0.84 um); only Q-D's tilt half open] [updated 2026-10-23, later 4: Q-D's tilt half settled by D-081.3 -- 5 deg both sets; nothing of this entry remains open] |
| D-075 | 2026-10-22 | Block 10's calibration outputs (`calibrate_kernel.py fit`, the scans file), Block 15 (not built), Colab Cell 7 | when the near- and far-field width laws are fitted, plot every node's per-plane width points with the fitted near curve and far line, and every off-focus plane's dip with its fitted Gaussian core [user]; layout PROVISIONAL | active; not implemented [updated 2026-10-23: D-075.2's windows are those of D-077's iterative core fit] [updated 2026-10-23, later: extended by D-080.2, the width-by-plane figures with the median and an example node's plane images] |
| D-076 | 2026-10-23 | D-073's form; D-072's floor; D-051 (i), (iv); Blocks 3-5 (`WindowConfig.t_ss_floor_ms`, the per-sweep stage of `noise_fit`, the generator's per-sweep laws, `run_ih_recovery`'s LS amplitudes) | per-sweep form (a): the fast pair and sigma_s per sweep, sigma_s shrunk towards the cell by DerSimonian-Laird, tau_s and the tilt per cell; t_SS(c) = max(80 ms, transient test), the subtracted quantity is the per-polarity mean window (per-sample mean), confirmed; the twins carry the real cell's own LS amplitudes; no QC anywhere | active; user [clarified 2026-10-26: statsmodels does not clip the DL $\hat\tau^2$; the clip and the clipped random-effects mean are Block 4b's to compute] [updated 2026-10-28 (later, 9): (ii)'s 80 ms floor kept; 80 ms is also the fallback -- D-103] |
| D-077 | 2026-10-23 | the far statistic of S2b (D-074's step 4; SPEC Block 15, not built); `CalibrationConfig`'s far fields; the far scan's plane statuses; D-075.2's panels | the two-pass core fit becomes iterative: pass $i+1$'s window $\pm$1 FWHM of pass $i$'s width, at most $N$ = 10 passes (the user's example), stopping at the first $i \ge 2$ with $\lvert\Delta{\rm FWHM}\rvert \le$ 1 SE of the FWHM (the fit's own SE, from its residual RMS; PROVISIONAL); no stop by $N$: `not_converged`, left out (PROVISIONAL) | active; not implemented (Block 15 not built) [updated 2026-10-23, later: $N$ = 10 and $\kappa$ = 1 set by D-078] [updated 2026-10-23, later 2: D-077.3 settled by D-081.4 -- the plane's own SE in the stop, unconverged planes dropped, N exposed] [implemented 2026-10-23: `24eb687`, `cd80081`, `f55c083`; not pushed [updated 2026-10-26: pushed]] |
| D-078 | 2026-10-23 | SPEC Block 15 (Q-B's background, Q-C); `CalibrationConfig`'s far fields; the far scan and far fit of `analysis/calibration.py` (not built); D-074.1's SNR reading, D-074.3, D-074.4; D-077.1-D-077.2 | the planar far-pixel background with the line-end mean as its check; D-077's iterative core fit with $N$ = 10, $\kappa$ = 1; the sequential far fit (anchor $\hat\sigma_0$, both sides, axis free +-0.28 um, $\lvert k - k^*\rvert \ge 5$, the predicted-dip SNR rule, model weights by IRLS) with ${\rm SE}^2 = {\rm SE}_{\rm boot}^2 + (\partial\hat\gamma/\partial\sigma_0)^2 {\rm SE}(\hat\sigma_0)^2$ and $t_{J-1}$ intervals [user] | active; not implemented (Block 15 not built); questions 4 and 5 open [updated 2026-10-23, later: D-078.1's patch isolated and its background used for the near field too, D-079] [implemented 2026-10-23: `24eb687`, `cd80081`, `f55c083`; not pushed [updated 2026-10-26: pushed]] |
| D-079 | 2026-10-23 | Block 10's node selection (`calibration_reasons`, near and far sets); Block 10's near scan background (PROVISIONAL reading); D-078.1's patch; D-074's Q-D (isolation); Colab Cell 7's counts | a calibration node's +-25 um background patch holds no traced point but its own branch, at any depth (PROVISIONAL reading); the near field is measured against the planar far-pixel background too (PROVISIONAL reading) [user] | active; not implemented (Block 10 to amend, Block 15 not built); readings to confirm [updated 2026-10-23, later: depth included, the user's message of 17:47 (D-080); the depth range open] [updated 2026-10-23, later 2: the readings settled by D-081.1-D-081.2 -- shadow-based depth rule, own dendrite excluded; planar background in the near scan] [implemented 2026-10-23: `24eb687`, `cd80081`, `f55c083`; not pushed [updated 2026-10-26: pushed]] |
| D-080 | 2026-10-23 | `kernel_from_growth`'s `max_delta_um` wherever a table is written for rendering (Block 10's `fit`, Block 15's output); the kernel table (simulation signature); D-075's figures; the calibration's plotting module; the scans file; Colab Cell 7 | the renderer's table ends at 0.84 um, so the continuation starts at the anchor; the width-by-plane figures of both stages: all nodes' $\hat\omega$ by plane in light grey with the median, and one example node with its plane images (layout PROVISIONAL) [user] | active; not implemented [updated 2026-10-23, later: the figures' layout settled by D-081.5] [implemented 2026-10-23: `24eb687`, `cd80081`, `f55c083`; not pushed [updated 2026-10-26: pushed]] |
| D-081 | 2026-10-23 | D-079.1's depth rule, D-079.2, D-077.3, D-080.2's layout, D-078.3's SE terms; SPEC Blocks 10 and 15; `CalibrationConfig`; Block 10's selection, near scan and output; Block 15; the plotting module; Colab Cell 7 | isolation by the predicted shadow (> 0.03 gl in a scanned plane), own dendrite excluded; planar background in the near scan, near then far; `node_phi_max_deg` = 5 both sets, straightness within $r_j$ (PROVISIONAL); passes exposed (`far_core_max_passes` 10, `far_core_stop_tol` 1), unconverged planes dropped; figures (a); $\gamma_2$; 1000 resamples, +-0.01 um anchor step, Block 10's $\hat\sigma_0$ bootstrap, the 0.84 um cut today [user] | active; implementation started [implemented 2026-10-23: `24eb687`, `cd80081`, `f55c083`; not pushed [updated 2026-10-26: pushed]; straightness against the trace's own TLS line, PROVISIONAL] [updated 2026-10-27, later 3: (D-081.3)'s straightness superseded by D-093 -- not tested by default, 2 radii when switched on] |
| D-082 | 2026-10-26 | the roadmaps' steps and their theory parts (`specs/roadmap/*.yaml`, `roadmap.py add step`); the diameter-pipeline roadmap's S3-S9 and NS1-NS2 | every step still to do is on the roadmap as a step, general for now (its blocks added when written); each step has a theory part of its own, its paired theory group or a named topic [user] | active; applied at `d6d410d`; open: G1 without a theory part, G4 and G5 sharing TG3 [updated 2026-10-26, 16:48: roadmap changes on hold until the user says so] [updated 2026-10-27: the hold of 16:48 lifted for blocks -- every new block enters the roadmap in the same turn, D-091] [updated 2026-10-28 (later, 10): the allen-mtg-fit roadmap updated at the user's word for D-102 to D-104, `a945a7a`] [updated 2026-10-29: updated again at the user's word for D-106 and D-107, `bee2abe`] |
| D-083 | 2026-10-26 | the order of work: Stage 7's gate run (roadmap allen-mtg-fit S7; SPEC Block 8, to specify) before any six-parameter fit of a real cell (S8, its one-cell dry run included); the twins' content (D-011, D-073, D-076); `run_ih_recovery.py` and `submit_ih_recovery.sh` on davinci; the roadmap's S7, S8, I4 | the six-parameter synthetic recovery runs on davinci before any real cell is fitted with the six parameters: twins on each cell's own morphology, with its own fitted noise and protocol, chosen truths, the same fitter; the six-parameter version of the passive-only check the user ran [user] | active; roadmap updated (`8c671a2`, patch pending the user's push) |
| D-084 | 2026-10-26 | Block 4's absorption (`RendererConfig.absorption`, today `partition_vertical`, D-024 (i)); I-004; the roadmap's NS2 and T7; the simulation hash and every table built on it (gate 2, the davinci pilot, the production); the thick-node check of S3a | the hybrid absorption renderer (I-004) is the renderer actually used once built; its optics (T7) first; the partition until then [user] | active; not implemented (T7 open) [updated 2026-10-26, later: S3d waits for the hybrid; the thick-node check dropped until then, D-085] [updated 2026-10-27: built as SPEC Block 21 and smoke-tested, the default not switched; its planes go below zero in dark stain, D-092] |
| D-085 | 2026-10-26 | S3 of the plan of 2026-10-20 (Blocks 16-20, to specify); the roadmap's S3a-S3d (on hold, D-082); D-065's "all the cells"; Cell 7's calibration per cell; D-084 | 5 cells per class (exc/inh) and layer, all when fewer; automatic alignment; dz and pixel size checked first over all cells; node selection exposed; every figure plus diameters by layer and class; no thick-node check; jitter by robust SDs pooled; one camera setting; one kernel per cell, pooling only if a cell lacks clean dendrites (checked); the final configuration waits for the hybrid [user] | active; not implemented; readings of (D-085.1), the d_max rule, the sufficiency thresholds and the per-cell tables open [updated 2026-10-26, later 6: Block 16.1 built locally (the pool, the acquisition check, the selection) with (D-085.1)'s readings as PROVISIONAL defaults -- spiny exc, aspiny inh, sparsely spiny left out, seed 20261026; the d_max rule settled by D-086] [updated 2026-10-26, later 7: sparsely spiny a third class, D-087; (D-085.10)'s kernel per cell superseded by D-088, one kernel pooled across cells] [updated 2026-10-27: Blocks 16.1-19 implemented and on the remote (the entry's status); S3d (Block 20) waits for the hybrid's switch, D-092] [updated 2026-10-27, later 3: (D-085.3)'s check run on Allen's data: the 47 lines the user pasted all read dz 0.28 um, pixel 0.1144 um ("All share the same.")] [updated 2026-10-28: Cells 11 and 12 run on Allen's data -- the pool 158 cells, the check 158 of 158 at dz 0.28 um and 0.1144 um/px, the selection 51 cells in 15 groups; 47 aligned, 4 failed (Allen's marker file fetched as the SWC), 5 set aside] |
| D-086 | 2026-10-26 | `PhantomConfig.d_range_um`'s upper end (Cell 8; the simulation hash); S3b (Block 18, to specify); D-065; D-085.7; D-024 (iii) | $d_{\max}$ = the maximum of Allen's dendrite $2r$ (SWC types 3, 4) across S3's cells, not the 99.5th percentile proposed [user] | active; not implemented (Block 18) [updated 2026-10-27: implemented, Block 18 (`83b400b`), the entry's status] |
| D-087 | 2026-10-26 | Block 16.1's class map (`SurveyConfig.class_map`) and selection groups; every per-class statistic and figure of S3a-S3d (D-085.5); D-085.1 | sparsely spiny cells are a third class, `sparsely_spiny`, beside exc (spiny) and inh (aspiny), with 5 cells per layer like the others [user] | active; Block 16.1 amended for it the same day |
| D-088 | 2026-10-26 | the defocus kernel's calibration (Blocks 10 and 15; Cell 7; S3c, Block 19); the production configuration and the number of tables; D-085.10 | one kernel, calibrated on near- and far-field nodes pooled across several cells, so one table; for now the test uses one pilot cell; each calibration set holds 10 to 30 nodes [user] | active; implemented 2026-10-26 (Block 19, `7064e2c`), smoke-tested; not run on Allen's data [updated 2026-10-27: the index's status brought up to the entry's] |
| D-089 | 2026-10-27 | the S3 blocks' PROVISIONAL defaults (Blocks 16.1-19; `SurveyConfig`, `CalibrationConfig`) | the set-aside ratio 2, the darkest-projection rule, the stratified default, `sparsely_spiny`, the z0 flag at 1 um, 5 scans per set per cell, "per type" as per calibration set, 529878215 the pilot -- confirmed [user] | active; implemented 2026-10-27 (comments and SPEC, `6b8ea64`) [index row added 2026-10-27, later] [updated 2026-10-28: on Allen's data (D-089.2) the darkest rule took the maximum-intensity xy projection in all 4 specimens that have one, and 508305938's alignment failed with it; (D-089.1) 4 of the 5 cells below 2 have shifts like the kept cells'; changes put to the user] [updated 2026-10-28, later: (D-089.2) superseded by D-098.2 -- the projection by its type, the darkest rule the fallback] |
| D-090 | 2026-10-27 | Block 16.2 (a)'s projection rule and the alignment of every S3 cell; S3c's pooled calibration; Cells 11b and 15; SPEC Blocks 16.3 and 19.5 | what Allen's projections are, settled by a Colab check (D-090.1); the pooled calibration's near and far points drawn together, coloured by cell, with their residuals from the pooled law (D-090.2) [user] | active; implemented 2026-10-27 (`6b8ea64`); not run on Allen's data [index row added 2026-10-27, later] [updated 2026-10-28: (D-090.1)'s check run on the pilot -- the minimum-intensity xy projection is the planes' minimum (r* 0.9941 at zero offset), the maximum-intensity one `unclear`] |
| D-091 | 2026-10-27 | every roadmap (`specs/roadmap/<name>.yaml`) and every new SPEC block; D-082's hold of 2026-10-26, 16:48; the diameter-pipeline roadmap's NS2 (now HR) and S3d | every new SPEC block enters its roadmap in the same turn it is written, without being asked; the hold lifted for this [user] | active; applied (`689c702`, `9b99b8b`; I21 with Block 21, `2572e15`) |
| D-092 | 2026-10-27 | Block 4's absorption (`RendererConfig.absorption`); D-084; I-004; the roadmap's HR (I21) and T7; S3d (D-085.11) | go on with the hybrid renderer [user]: built as SPEC Block 21, the mode `hybrid`; name, slab rule, Block 9's directions and the default left at the partition PROVISIONAL | active; implemented 2026-10-27 (`2572e15`), smoke-tested; the default not switched; its dark-stain planes go below zero (D-092.3) |
| D-093 | 2026-10-27 | Block 10.I (v): the near and far calibration sets (Blocks 10 and 15) and S3c's per-cell scans (Block 19.3); `CalibrationConfig.node_straight_check`, `node_straight_tol_radii`; D-081.3's straightness part; Colab Cell 7 | no straightness test for the calibration nodes (the calibration evaluates only the measuring line); flatness stays strict ($\varphi \le$ 5 deg); if restored, 2 (or 4) x $r_{\rm A}$ [user] | active; implemented 2026-10-27 (`6a74735`, `d46340c`, `eed549d`), smoke-tested; not re-run on Allen's data [updated 2026-10-28: (D-093.3)'s faintness and the flag `faint` of $\mathcal S$ no longer apply to the calibration nodes, D-094] |
| D-094 | 2026-10-28 | Block 10.I (i), (iv): the near and far calibration sets (Blocks 10 and 15), S3c's per-cell scans (Block 19.3), the CSV's `calibration_node`, `candidates_from_rows`; `CalibrationConfig.node_alpha_check`, `node_faint_flag_check`; D-067.4's faint criterion; Colab Cell 7 | the calibration nodes without the optical-depth criterion and without the flag `faint`; $\mathcal S$ itself keeps the flag [user; the last part the assistant's reading] | active; implemented 2026-10-28 (`223a5a6`), smoke-tested; not re-run on Allen's data [updated 2026-10-28: (D-094.3) superseded by D-096 -- $\mathcal S$ drops the flag `faint` too] |
| D-095 | 2026-10-28 | the SS window start $t_{\rm SS}(c)$ and the SS subtraction (D-051 (iv), (D-051.1); D-076 (ii)); Block 3 (`transient_start`, `subtract_mean_response`, `WindowConfig`), Block 4's simulator (`simulate_cell_windows`), smoke W2, W3, F2; SPEC §2.2; `noise_ss_enl_t0_ms` | mean windows per polarity and per sweep over [-10, 200) ms; $t_{\rm SS}(c)$ where their exponential decay falls within $\kappa_{\rm tr}\hat\sigma_{\rm SS}(c)$ ("statistically"); no binning; each enlarged window minus the mean window of its sweep and polarity on $[t_{\rm SS}(c), 200)$ [user]; mean or median, the reading of "statistically", the per-group combination, the fit model, the 80 ms floor and the fallback open | active; not implemented; supersedes D-051 (iv)'s test and (D-051.1)'s grouping, D-076 (ii)'s $t_{\rm tr}$ [updated 2026-10-28: revise marks on the roadmap's N3 and N4] [updated 2026-10-28 (later, 9): open points closed by D-103 -- median window, point value, maximum, one exponential on [5, 200) ms, 80 ms floor and fallback; point 9 and the trigger open] [updated 2026-10-28 (later, 10): the fallback's trigger and point 9 closed by D-104] |
| D-096 | 2026-10-28 | the selection $\mathcal S$ (Block 6, `config.SELECTION_FLAGS`), phantoms and real nodes; the bias table (Block 14); `correct_nodes` and the fill (Block 7, D5); the estimator and simulation signatures (Block 1); D-094.3; D-030 | the flag `faint` is a label, not a criterion of $\mathcal S$: every node whose fit is in $\mathcal S$ is corrected, whatever its optical depth and its dip [user]; the flags of $\mathcal S$ in the estimator signature [the assistant's choice] | active; implemented 2026-10-28 (`cf2f91d`, renumbered `76365a3`), smoke-tested; no table built |
| D-097 | 2026-10-28 | every chat's delivery: code pushed by the assistant (hpc-git-delivery's last step); every document also in the repository (`docs/project/` on `main` for the project-knowledge documents; handoffs in `<workstream>/docs/`, study documents and checks in `<workstream>/dev/`); DOCUMENT UPKEEP; the roadmaps' push | the assistant pushes its own commits; every document is also committed to the GitHub repository so any chat can read it [user]; where, the assistant's reading; the user's uploads not copied without their word; the repository is public | active; applied 2026-10-28 (noise-fit chat: `53b2713`, `d2ffcb4`, `cd32dad`) |
| D-098 | 2026-10-28 | `allen_image_io.fetch_swc` (Block 0); Block 16.2 (a)'s projection rule and 16.3 (a) (`projection_xy_type`); S3a's coverage (Block 17.3; `phantoms.row_reject_reasons`); Cell 7's outputs (Block 10.F (5); `scan_cell`'s `on_node`); the set-aside cells' use (D-089.1); D-089.2 | the marker file never taken; the xy projection by its type, the darkest the fallback; $\mathcal S$ recomputed from the rows' columns in the coverage; Cell 7 draws each candidate's background patch with the neighbours behind `not_isolated`; the set-aside cells that look aligned run with `--include-set-aside` [user; the figure's form the assistant's, PROVISIONAL] | active; implemented 2026-10-28 (`c595bc0`, `4ef6d8e`, `cdba78a`, `28790c3`, `44e95e3`), smoke-tested; not run on Allen's data [updated 2026-10-28 (later, 11): (D-098.4)'s figure gains panel (c), every plane's stripe, and the overview's $n/N$ -- D-105] |
| D-099 | 2026-10-28 | where S3's per-cell survey runs (Blocks 16.2, 17, 18, 19 over the selection: SPEC Block 16.4, `scripts/pbs/survey_*.pbs`); Colab's Cells 11-15 (the pilot only, `survey_pilot/`); the runner's `--task-index` and `--cache-per-cell`; `cell_pool.py pick` (Block 16.1 (e)); the roadmap's S3a-S3c (`needs_hpc`); D-085's "in Colab"; D-098.5's Cells 12b and 13b | the real runs on davinci, one PBS array task per selected cell plus a prepare and a pool job; Colab keeps every cell on the pilot [user]; the layout under the archive root, the proxy module and the 12 h walltime the assistant's, PROVISIONAL | active; implemented and pushed 2026-10-28, smoke-tested against stubs; not yet run on davinci |
| D-100 | 2026-10-28 | the tilt SD the twins of the 14 two-sweep L3_exc cells receive (Block 5; the tilt of (D-017.1)); D-034's (D-034.1) for those cells; a cohort step after the 18 per-cell fits (Block 6); Block 3's pair counts and Block 4's `TiltResult`; the Stage-7 gate's report (D-011); SPEC §8 (2026-10-23) | for the 2-sweep cells the twins' tilt SD both per cell (a) and as one cohort value (b); partial pooling (c) not taken [user]; two labelled variants of twins, the cohort value's form (the $S_{\rm eff}$-weighted mean of the unclipped estimates over the 18 cells, clipped at 0) and each cell's own fit unchanged -- the assistant's reading, to confirm | active; not implemented; its reading open [superseded 2026-10-28 (later, 9): withdrawn as the assistant's misreading -- D-102] |
| D-101 | 2026-10-28 | the calibration nodes' flatness (10.I (iii); `CalibrationConfig.node_phi_max_deg`, `node_phi_max_deg_far`); the background's support and the isolation verdict of both calibration sets (Blocks 10.P, 10.I (vi), 15.1; SPEC Block 10.S; `background_support`, `stripe_half_widths_um`, `stripe_min_pixels`, `stripe_min_side_um`); Cell 7 and S3c's per-cell scans; D-078.1's far mask and D-081.1's node verdict (kept as `"square"`); D-081.3's 5 deg | far $\varphi \le 15$ deg, near $\le 10$ deg; the stripe: a pixel is clear when no neighbour's predicted shadow at it exceeds 0.03 gl, the band along the measuring line with most clear pixels, walked out to the first shaded column, the own fitted cylinder removed, usable with 500 pixels and 3 um per side [user]; the $W$ grid, the walk, the own radius $\max(\hat d/2, r_A)$ and the near bound read as 10 deg the assistant's, PROVISIONAL | active; implemented and pushed 2026-10-28 (Block 10.S), smoke-tested; not yet run on Allen's data [updated 2026-10-28 (later, 11): rechecked -- the verdict's soma blur fixed (F1), the own cylinder on the fitted line (F2, the reading to confirm); on the pilot's own reconstruction as built 21 of 138 near candidates usable at $k^*$, none of 7 far; (c) answered by it, (d) open -- D-105] |
| D-102 | 2026-10-28 | the tilt in every twin (Block 5; (L1) = (D-017.1)); D-034's draw $b \sim \mathcal N(0, \sigma_b(c)^2)$ and (D-034.1) for the twins; D-100; Block 4's tilt alternation and (T1); Block 3's tilt reading; D-073; SPEC §2.4, §8 | each twin sweep carries the realised tilt of its own real sweep -- "the very same tilt as in the real cell", no statistic of the tilt computed [user]; read on an SS sweep as the least-squares slope of its window means on time, one intercept per polarity (the assistant's proposal); the LS sweeps' reading and the fit's simulator open; D-100 withdrawn as a misreading | active; not implemented; open (a)-(d) [updated 2026-10-29: (a), (b), (d) closed by D-107 -- the LS tilt through two window means; the simulator uses the real sweeps' tilts] [updated 2026-10-29 (later): (c) answered by the check noted in D-107 -- the median subtraction leaves the SS reading unchanged; the later starts raise its error by 15-23 %] |
| D-103 | 2026-10-28 | D-095's open points; D-076 (ii)'s 80 ms floor; D-051 (i)'s fallback (`fallback_t_ms`, `ss_enlarged_none`); Block 3 (`transient_start`, `subtract_mean_response`, `WindowConfig`), Block 4's simulator; smoke W2, W3, F2; SPEC §2.2 | the per-sample median window per sweep and polarity, for the start and the subtraction; one exponential with a free baseline on [5, 200) ms; $t_{s,p} = \hat\tau_{s,p}\ln(\lvert\hat A_{s,p}\rvert/(\kappa_{\rm tr}\hat\sigma_{\rm SS}(c)))$, the point value; $t_{\rm SS}(c) = \max\{80\ {\rm ms}, \max_{(s,p)} t_{s,p}\}$; 80 ms also the fallback, flagged [user]; the fallback's trigger, point 9 and $\kappa_{\rm tr}$ open | active; not implemented [updated 2026-10-28 (later, 10): (a) and (b) closed by D-104] |
| D-104 | 2026-10-28 | D-103's open points (a), (b); (D-103.3); Block 3's window start (`WindowConfig.fallback_t_ms`, `min_window_ms`, `min_windows_per_group`); the flags `ss_start_fallback` and `ss_group_dropped`; smoke W2; the roadmap allen-mtg-fit | the 80 ms fallback applies when the window $[t_{\rm SS}(c), 200)$ ms would be shorter than 50 ms; a group whose fit fails or has too few windows is flagged and left out [user]; also with no valid group, and the minimum of 5 windows (PROVISIONAL) -- the assistant's; the roadmap updated for D-102 to D-104 [user] | active; not implemented; open: a dropped group's windows in the statistics [updated 2026-10-29: open point (a) closed by D-106 -- the windows of a group with too few windows leave the statistics too] |
| D-105 | 2026-10-28 | the isolation figure of D-098.4 (SPEC Block 10.F (5): panel (c), the overview's $n/N$, `real`'s knob and far-set lines); Cell 7's `CAL_KNOBS` and `OUT/calibration/calibration_config.json` (both notebooks), Cell 15's `survey_config_s3c.json`, davinci's `DIAM_CONFIG_JSON`; SPEC Block 10.S as amended by the recheck (F1, F2, S1-S3, S5-S7) | the stripe of every plane shown, the far fit's planes emphasised; the stripe's existing settings and the flatness bounds exposed in Cell 7 and passed to every step [user]; the set of knobs, panel (c)'s form and F2's fitted cylinder -- the assistant's | active; implemented and pushed 2026-10-28; smoke-tested; open: D-101 (d) (how out-of-focus shadows enter the stripe), S4, F3, F2's reading, the far verdict on the fit's planes |
| D-106 | 2026-10-29 | D-104's open point (a); Block 3's statistics and the layout it hands to Block 4; Block 4's simulator; `WindowConfig.min_windows_per_group` | a group with too few windows (fewer than 5, PROVISIONAL) leaves the noise statistics too; a group dropped only for a failed fit keeps its windows [user] | active; not implemented |
| D-107 | 2026-10-29 | D-102's open points (a), (b), (d); Block 3's tilt readings; Block 4's simulator and fit (`simulate_cell_windows`, `fit_cell`, `tilt_from_semivariogram`); Block 2's `simulate_traces`; Block 5; SPEC §2.3, §2.4, §8 | a Long Square sweep's tilt is the line through a start-window mean and an end-window mean; every simulated sweep carries its real sweep's tilt in every replicate, so the tilt SD leaves the fit [user]; the windows (265 ms pre-window, last 500 ms) the assistant's reading | active; not implemented [updated 2026-10-29 (later): (a) answered by the check noted in the entry] |
| I-001 | 2026-09-19 | side project: spine-resolved human models from H01; closes D-001's open $R_{\rm neck}$ point and bounds D-001's validity regime | human spine simulations in two tiers — cable (NEURON) $R_{\rm neck}$ census, then electrodiffusion on selected geometries (own solver) | idea, user wants to pursue; not scheduled |
| I-002 | 2026-10-06 | the kernel family of the renderer and of the calibration (procedure §3.4) | a flexible kernel family with the Gaussian as a special case, its parameters fitted in the calibration | idea, to discuss further [user]; not scheduled [updated 2026-10-16: the fit's kernel is now a flexible family, D-052; the renderer's and the calibration's stay Gaussian] [updated 2026-10-20, later: adopted for the renderer and the calibration by D-066] [updated 2026-10-20, later 2: D-066 withdrawn by D-067; back to an idea, not scheduled] |
| I-003 | 2026-10-22 | the renderer's $\sigma_{\rm r}(\delta)$ and its construction (Blocks 4, 10, 15; procedure §3.4 step 2); D-067.2 and I-002 for any non-Gaussian kernel | the defocus kernel constructed with deep learning, in a chat of its own; handoff `docs/TEEG_diameter_defocus_kernel_DL_handoff_2026-10-22.md` | idea, the user pursues it in a chat of its own; not scheduled in the diameter plan [updated 2026-10-22, later: the handoff pushed, `9112bf3..37b42ea`] |
| I-004 | 2026-10-22 | Block 4's absorption (`partition_vertical`, D-024 (i)) and Block 9's ray world; the hybrid of the 2026-10-09 renderers document (§3.8-3.9, on `main`); roadmap T7 | the hybrid renderer -- each slice's absorption from the condenser's rays, then the defocus kernel -- whose optics the user studies first | idea; roadmap T7 open; not scheduled [updated 2026-10-26: adopted as the renderer actually used once built, D-084] [updated 2026-10-27: built as SPEC Block 21 (D-092), not yet the default] |

---

## D-001 — Synaptic conductances are shaft-referenced, effective, model-conditioned

**Date:** 2026-09-19. **Binds:** the per-pathway synaptic-parameter table
(pre-type × post-type × layer) that will be built for the microcircuit stage;
every place a peak conductance is assigned to a synapse on a shaft
compartment of an exported `.hoc` cell, including synapses that the H01 data
place on a spine head and that the pipeline redirects to the parent shaft.

**Decision [user].** Every synaptic peak conductance entered in the model is
treated as $G_{\rm syn}^{\rm shaft}$ — the effective conductance of a synapse
placed on the dendritic shaft compartment, conditional on the model in which
its value was obtained. No head-referenced conductance is entered directly.

**Statement.** Let $G_{\rm syn}^{\rm head} \in \mathbb{R}_{>0}$ (nS) be the peak
conductance of a synapse located on a spine head, $R_{\rm neck} \in
\mathbb{R}_{>0}$ (MΩ) the spine-neck resistance, $E_{\rm syn} \in \mathbb{R}$
(mV) the synaptic reversal potential, and $V_{\rm base}(t) \in \mathbb{R}$ (mV)
the membrane potential at the spine base (the shaft compartment) at time $t$.
In the linear, quasi-static regime, for one synapse, neglecting the head's own
membrane current, the current delivered to the shaft is

$$
I_{\rm shaft}(t) \;=\; \frac{G_{\rm syn}^{\rm head}}{1 + G_{\rm syn}^{\rm head} R_{\rm neck}}\,\bigl(E_{\rm syn} - V_{\rm base}(t)\bigr)
\;\equiv\; G_{\rm syn}^{\rm shaft}\,\bigl(E_{\rm syn} - V_{\rm base}(t)\bigr),
\tag{D-001.1}
$$

so that

$$
G_{\rm syn}^{\rm shaft} \;=\; \frac{G_{\rm syn}^{\rm head}}{1 + G_{\rm syn}^{\rm head} R_{\rm neck}} \;<\; G_{\rm syn}^{\rm head}.
\tag{D-001.2}
$$

$G_{\rm syn}^{\rm shaft}$ is what a fit of somatic PSP amplitude with synapses
placed on shaft compartments returns; it is a function of the fitting model
(morphology, $c_m$, $R_a$, spine factor $F$, number of contacts per connection
$N_{\rm syn}$, release probability $U$, placement rule), not a property of the
synapse alone. Two objects, two names: $G_{\rm syn}^{\rm head}$ (Eyal 2018's
$g_{\max}$) and $G_{\rm syn}^{\rm shaft}$ (Yao 2022, Guet-McCreight 2023, BBP
values) are never interchanged without (D-001.2). **[reasoning]**, textbook
circuit relation, hypotheses as stated.

**Why.** The pipeline removes spines from the exported morphology and carries
their membrane through $F$ (per-segment `cm *= F`, `g_pas *= F` beyond 60 µm).
Eyal et al. 2016 state the validity condition of that lumping explicitly: it
holds when current flows dendrite → spine (somatic stimulus; head and base
isopotential) and fails when current flows spine → dendrite (synaptic
activation), in which case "the stimulated spines must be modeled
individually" **[KB]**. Since we do not model them individually, the only
consistent place to put the neck's series effect is in the conductance, which
is what (D-001.2) does. With human L2/3 values ($G_{\rm syn}^{\rm head} =
0.63$–$0.88$ nS, $R_{\rm neck} = 50$–$80$ MΩ, Eyal 2018 **[KB]**) the current
reduction is 3–7 % **[reasoning]**; with a rat-like $R_{\rm neck} = 514$ MΩ it
would be ≈ 31 %. The *voltage* attenuation head → base ($1.61 \pm 0.93$,
Eyal 2018 **[KB]**) is a different quantity and is not absorbed by anything on
the shaft.

**What it implies for code and tables.**

1. The synaptic-parameter table carries, per entry, the columns
   `G_syn_shaft_nS`, `source`, `fitted_in_model` (morphology set, $c_m$, $R_a$,
   $F$, $N_{\rm syn}$, $U$, placement rule of the source fit), `refit_status`.
   A value whose `fitted_in_model` differs from ours is a **prior**, not a
   parameter.
2. Provenance classes of what is available in the knowledge base **[KB]**:
   (a) human, shaft-fitted, model-specific — Yao 2022 (five L2/3 pathways:
   Pyr→Pyr, Pyr→SST, SST→Pyr, PV→Pyr, Pyr→PV) and Guet-McCreight 2023 (L5
   Pyr→Pyr; SST→Pyr borrowed from Yao); (b) rodent, shaft-fitted — BBP
   (Markram 2015 Table S6; Borges 2022 Table 1), used by (a) for every other
   pathway; (c) human, **head-referenced** — Eyal 2018 AMPA $g_{\max}$
   $0.63 \pm 0.13$ / $0.88 \pm 0.70$ nS, NMDA $1.31$ nS, $\gamma = 0.077$
   mV⁻¹. Class (c) enters only after conversion by (D-001.2) with a stated
   $R_{\rm neck}$, and only as a prior.
3. Synapses that the H01 synapse tables place on a spine head and that the
   pipeline redirects to the shaft receive $G_{\rm syn}^{\rm shaft}$, never
   $G_{\rm syn}^{\rm head}$.
4. The NMDA gating factor $B(V) = 1/\bigl(1 + \exp(-\gamma V)\,[{\rm Mg}^{2+}]/n\bigr)$
   is evaluated on $V_{\rm base}(t)$ in our model, whereas Eyal 2018 fitted
   $\gamma$ on $V_{\rm head}(t)$. This is a known approximation, to be flagged
   in every table row that carries an NMDA component; no scalar conductance
   corrects it.

**What it does not decide [open].**

- The **refit procedure**: whether every human-fitted value is refitted in our
  own model against the same experimental PSP targets (Seeman 2018,
  Obermayer 2018, Komlósi 2012, Szegedi 2016, Testa-Silva 2010) as Yao and
  Guet-McCreight did, or whether their values are used as-is with the model
  mismatch accepted and documented. The assistant's recommendation on record
  is to refit; the user has not decided.
- The value of $R_{\rm neck}$ to use in (D-001.2) when converting class (c)
  values, and whether it is per-layer or per-spine (the H01 mesh gives neck
  geometry per spine) — I-001 Tier 1 is the planned route to close this.
- Whether pathways with no human data at all (L4, L6, INH→INH) take BBP
  values with a documented rodent tag or are excluded from EEG-level claims.

**Sources.** Eyal et al. 2016, eLife (Methods, "Incorporating dendritic spines
globally", validity condition of $F$) **[KB]**; Eyal et al. 2018, Front Cell
Neurosci (Methods "Model for Dendritic Spines", "Synaptic Inputs"; Results
Fig 3–4; Table S2) **[KB]**; Yao et al. 2022, Cell Reports (STAR Methods,
synaptic connectivity; synapse placement basal/apical/shaft) **[KB]**;
Guet-McCreight et al. 2023, Cereb Cortex (Methods, synaptic connectivity
models) **[KB]**; Markram et al. 2015, Cell (Fig 9–10, Table S6) **[KB]**;
Borges et al. 2022, Front Neuroinform (Table 1) **[KB]**. Equation (D-001.1):
textbook circuit theory, from memory, hypotheses stated.

**Status:** active. Supersedes nothing. Referenced by
`TEEG_HPC_coding_handoff_2026-09-19.md` section 6.

---

## D-002 — Morphological QC is a filter applied downstream, never a gate in P1

**Date:** 2026-09-21. **Binds:** `run_p1_export.py` (what is exported and
what is withheld), `p1_summary.csv` (what the filter needs), the P4 /
analysis selection criterion.

**Decision [user].** P1 exports every structurally exportable cell and
records its verdict; the high-confidence subset is selected afterwards by
filtering the recorded verdict, never by refusing the export.

**Statement.** For each cell $c$ in a population, P1 writes
$q(c) \in \{\text{pass}, \text{pass\_low\_confidence}, \text{fail}\}$ together
with the reason set $\mathcal{R}(c)$ (e.g. `soma_below_radius_floor`,
`name_geometry_disagree`, `soma_not_outlier`). Only $q(c) = \text{fail}$
(checks C1/C2, or no root) withholds the artefacts. A downstream stage
selects $S_\alpha = \{c : q(c) = \text{pass}\}$ or a criterion written in
terms of $\mathcal{R}(c)$; the selected set is recorded, the excluded cells
are excluded by criterion and never deleted.

**Why.** The first real export (1302789404) carried `pass_low_confidence`
because no node of its 89192-node skeleton reaches soma calibre (thickest
radius 1642 nm against the 2000 nm floor) **[run, cluster]**; whether such a
cell is a fragment or a skeletonisation artefact is a question about the
population, not about one cell, and the soma census answered it the same
day: 71.0% of the 3383 skeletons are `ok`, so the filter is a filter, not a
decimation **[run, cluster]**.

**What it implies.** `p1_summary.csv` carries `qc_status`, `reasons`,
`hoc_verdict`, `soma_diameter_um`, `gate_status`, `quarantined`; P4 gains
a written selection criterion; `soma_census.py` gives the per-population
pass rate before any run.

**What it does not decide [open].** The threshold itself (2000 nm floor,
outlier ratio 5.0); what to do when a population is decimated (L6_inh keeps
29 of 37); whether `geometry_disagrees` cells are recoverable (the
soma-to-candidate distance column, proposed, not written).

**Sources.** Handoff section 6 and 4.2; `soma_enforce.py` Q6 soft policy
**[src]**.

**Status:** active. Supersedes nothing.

---

## D-003 — Interneurons are exported under both passive variants, into separate trees

**Date:** 2026-09-21. **Binds:** `h01_code/passive_params.csv` (L2/L3 inh
rows deliberately blank), `passive_params_inh_SST.csv`,
`passive_params_inh_PVVIP.csv`, the `p1_export.pbs` knobs `PASSIVE_TABLE` /
`OUT_DIR`, the output trees `h01/p1_inh_SST/` and `h01/p1_inh_PVVIP/`, and
every downstream stage that builds a simulation from an exported
interneuron.

**Decision [user].** The L2/L3 inhibitory populations are exported twice,
once under the SST passive set and once under the PV/VIP set, each into
its own tree; which tree the EEG model uses is a downstream decision. The
cluster label stays `(layer, exc/inh)`; no per-cell PV/SST/VIP assignment
is made.

**Statement.** Let $c_m \in \mathbb{R}_{>0}$ (uF/cm$^2$) and $R_a \in
\mathbb{R}_{>0}$ (ohm cm) be the passive constants of a table row, $d \in
\mathbb{R}_{>0}$ (um) a section diameter and $f$ (Hz) the d_lambda
frequency. The exporter's segmentation uses
$$\lambda_f(d \mid c_m, R_a) = \tfrac{1}{2}\sqrt{\dfrac{d}{\pi f R_a c_m}},
\tag{D-003.1}$$
so for the same skeleton the two tables ($c_m = 1.0$ vs $2.0$, $R_a = 100$
in both) give $\lambda_f^{\rm PVVIP}/\lambda_f^{\rm SST} = 1/\sqrt{2}$ and
about 41% more segments per cell in the PV/VIP tree: the two trees are
different exports (different `lfpy_idx`), not relabellings. $c_m$ and $R_a$
are **not** written into the `.hoc` (`default_cell_factory(passive=False)`)
**[src]**; a simulation built from either tree must apply the $(c_m, R_a,
\lambda_f, d_\lambda)$ recorded in that tree's `_p1.json`.

**Why.** Yao et al. 2022 (Table S6) fit SST with $c_m = 1.0$ and PV, VIP
with $c_m = 2.0$ (n = 1 human neuron each; $R_a = 100$ fixed from Hay et
al. 2011, never fitted in human) **[KB]**; the H01 label carries no
subtype, so neither set can be assigned per cell. Exporting both keeps the
choice open at the cost of one extra array per inh population; choosing
one now would bake an unverifiable subtype assumption into every
interneuron's segmentation. Caveat carried in the PV/VIP table's
provenance: Guet-McCreight et al. 2023 set their PV $c_m = 2$ "to reproduce
membrane time constants (possibly due to errors in PV dendritic diameter
estimation)" **[KB]** -- a fit compensation, not a membrane measurement.

**What it implies.** One qsub per (population, variant); each record and
summary row names the table (`passive_table`, `passive_table_sha`); the
fingerprint hashes the values, not the file name; the variant tables are
inh-only, so an exc manifest against one is refused; `--summarise` runs
per tree. `Rm_qc` (the gate's Rm only) is $1/g_{\rm pas}$ of the SST model
(43103 ohm cm$^2$) and of the VIP model (38760; PV's 8475 would pass the
C4 window too).

**What it does not decide [open].** Which variant the microcircuit uses,
and by what criterion; whether a composition-weighted mixture (Yao: 5% SST,
7% PV, 8% VIP of L2/3 **[KB]**) is ever attempted; the L4-L6 inh rows of
both variant tables (blank, refused).

**Sources.** Yao et al. 2022, Cell Reports, Table S6 **[KB]**;
Guet-McCreight et al. 2023 **[KB]**; Hay et al. 2011 (via Yao) **[KB]**;
`morphology_exporter.default_cell_factory`, `alignment.py` lambda rule
**[src]**. Equation (D-003.1): NEURON's d_lambda rule, textbook, from
memory.

**Status:** active. Supersedes nothing; extends the "label is (layer,
exc/inh)" decision of 2026-09-20 (handoff section 4, item 1) rather than
reopening it.

---

## D-004 -- One submission runs the whole per-cell pipeline, and the spine area is the mesh

**Date:** 2026-09-22. **Binds:** the campaign job scripts (`p1_export.pbs`,
`spine_area_F.pbs`, and the orchestrator that does not yet exist); the
definition of a finished cell; every place an $F$ or a spine area is reported,
including `p1_summary.csv`, `spine_area_F_summary.csv`, `p1_plots.py` and any
downstream passive model.

**Decision [user].** Launching one job must run the whole pipeline -- P1, then
P2, then P3 -- for every cell of the population, and within each run the spine
area of each cell is computed from the **mesh**, `mesh_beyond`. The other
variants exist for comparison and are never the reported number.

**Statement.** For each cell $c$ in a population $P$ and one submission, the
pipeline must produce the complete artefact set: the aligned `.hoc` and its
tables (P1), the per-spine mesh areas (P2), and `phi_mesh` together with
$F^{\rm mesh\_beyond}(c)$ (P3, `--deliverable mesh_beyond`, which is already
that script's default **[src, verified 2026-09-22]**). The quantity reported
as $F$ for cell $c$, in any table, figure or model, is
$F^{\rm mesh\_beyond}(c)$. The skeleton-frustum value $F^{\rm skel}(c)$ is
retained as a comparison bracket and is never reported as $F$ without the
superscript.

**Why.** $F$ enters the model as a per-segment multiplier on $c_m$ and
$g_{\rm pas}$ beyond 60 um, so which spine-area model produced it is a
property of every simulated cell, not a reporting detail. Splitting the
pipeline into three separately-launched stages made the skeleton bracket the
only thing available at the end of a run, and it was then reported as $F$ --
in the P1 record, in `p1_summary.csv`, and in the first version of the arbour
figure. A pipeline whose default output is the fallback will keep producing
that outcome however carefully the fallback is labelled.

**What it implies for code and tables.**

1. An orchestration layer is needed; **none exists**. P1 is an array over
   manifest rows, P2 is an array over the shards of ONE cell, P3 is a manual
   per-cell command. See the handoff of 2026-09-22 for the design.
2. **P2 and P3 run once per CELL, not once per passive variant.** Spine area
   is geometry: it does not depend on $(c_m, R_a)$, so the two D-003
   interneuron trees (`p1_inh_SST`, `p1_inh_PVVIP`) share one mesh
   measurement, which lives in `h01/out/` and is not duplicated per tree.
3. `p1_summary.csv`'s `F_lit` column is P1's skeleton value and is never
   updated by P3. Either P3 writes the deliverable back into the P1 record, or
   every consumer joins `spine_area_F_summary.csv` on `cell_id`. Until one of
   those exists, that column is a standing trap.
4. `p1_plots.py` v1.2 already implements the reporting half: `--phi mesh` is
   the default and the arbour figure is skipped, with its reason, rather than
   silently drawn from the skeleton table.

**What it does not decide [open].**

- The shard count per cell, the walltime, and whether the orchestrator is one
  array per population or one per cell. All three follow from the per-spine
  measurement rate, which **has never been measured** -- the only figure on
  record (2.3 s) is a cold block read, not an ROI, and must not be used.
- What to do with a cell whose mesh coverage is poor (`fallback_area_frac`
  high): its `mesh_beyond` $F$ is then partly kappa-filled skeleton, i.e. a
  hybrid rather than a measurement. The threshold at which that is
  unacceptable is not set; `--min-coverage` defaults to 0.99.
- Whether P1 should be re-run once P2 exists so its own record carries the
  mesh $F$ (expensive, and unnecessary if item 3 is solved by a join).

**Sources.** The pipeline structure and the `--deliverable mesh_beyond`
default read from source 2026-09-22 (`merge_spine_area_F.build_parser`,
`h01_spine_area_F.DELIVERABLE_DEFAULT`, `phi_with_spine_areas`) **[src]**; the
decision itself **[user]**; the standing "spine area from the mesh, skeleton
only as fallback, shaft area always skeleton frustums" rule from the handoff
section 6 **[KB]**.

**Status:** active. **The orchestration was IMPLEMENTED on 2026-09-22** as
`h01_code/campaign.pbs` (one array, index = manifest row x shard, P1 on shard
0, P2 on every shard, P3 in the same task at `SHARDS=1` or as a dependent
`PHASE=merge` run) together with `campaign_join.py`, which closes item 3's
standing trap by the join route: smoke suite 9/9 in the assistant's replica,
byte-verified into the laptop clone, **pending the user's commit, push and
cluster pull**. Item 1's "none exists" is therefore superseded as of that
date; the as-built design and the two corrections the sketch needed are in
`TEEG_campaign_single_submission_handoff_2026-09-22.md` sections 5 and 11.
Of the three "what it does not decide" points, the coverage-threshold point
is closed by **D-009** (no threshold: the uncovered fraction is reported) and
the re-run point is moot (item 3 was solved by the join); the per-spine rate
remains unmeasured and still blocks the shard count and the walltime.
Supersedes nothing; makes explicit what the handoff's section 6 rule already
implied for the campaign's execution rather than only for its measurement
code.

---

## D-005 -- I_h enters the passive fitter with six fitted parameters, uniform density, analytic rest balance

**Date:** 2026-09-22. **Binds:** the new folder `Passive Features/HPC script/Ih
Fit/` (a copy of `Biological Fit/` that is edited; see D-006): `mod/Ih.mod`,
`mod/Ih_human.mod`, `ih_mechanism.py`, the `ParamSpec` that replaces
`PassiveSearchSpace`, the search box of every I_h fitting arm, the GP budget
of every arm, and every table that reports a fitted $\bar g_h$, $\Delta v_h$
or $\kappa_\tau$. Specified in `TEEG_Ih_fit_staged_plan.md` v0.2 §6.

**Decision [user].** The fitted vector is six-dimensional,
$\theta = (C_m, R_m, R_a, \bar g_h, \Delta v_h, \kappa_\tau)$ -- the passive
triple plus the I_h density, the position of its activation curve (the knob
that sets $m_\infty(V_{\rm rest})$) and a scale on its time constant; the
I_h density is uniform over the cell, with the Eyal/Hay apical law kept as a
selectable comparison; $\bar g_h$ is searched log-uniformly in
$[10^{-6}, 10^{-3}]$ S/cm² and the two kinetic knobs in boxes slightly wider
than the human literature; the leak reversal $e_{\rm pas}$ is shifted
analytically so that the Allen $V_{\rm rest}$ is an exact stationary state
(no free $e_{\rm pas}$); the GP budget is `n_calls = 200`, `n_initial = 100`
in every arm.

**Statement.** Let $m_\infty^{\rm base}(v) \in (0,1)$ and
$\tau^{h,{\rm base}}(v) \in \mathbb{R}_{>0}$ (ms), for each fixed $v$ (mV), be the
steady-state activation and time constant of the chosen base kinetics
(`Ih_human`, Rich 2021 Eq. 1; or `Ih`, Kole 2006, optionally with the
configuration shift $\Delta v_{\rm base} = +20$ mV that reproduces Kalmbach
2018 -- their text says "$-20$ mV" because the sign refers to the term inside
the rate argument, $v - 20$; in the convention of (D-005.1) that is $+20$).
[Sign corrected 2026-09-22 (later, 3); v0.2 of the plan and the first
writing of this entry said $-20$.] The fitted kinetics are, for each segment $s$ and fixed $v$,
$$m_\infty(v) = m_\infty^{\rm base}(v - \Delta v_{\rm base} - \Delta v_h), \qquad
\tau^{h}(v) = \kappa_\tau\, \tau^{h,{\rm base}}(v - \Delta v_{\rm base}),
\tag{D-005.1}$$
with $\Delta v_h \in [-10, +10]$ mV (linear), $\kappa_\tau \in [0.5, 2.0]$
(log), $\bar g_h \in [10^{-6}, 10^{-3}]$ S/cm² (log), and the passive boxes
unchanged. $\bar g_h(s) = \bar g_h$ on soma, basal and apical segments
(uniform law). For each segment $s$, with $g_{\rm pas}(s) = F(s)/R_m$ and
$F(s) \in \{1, F\}$ the spine multiplier,
$$e_{\rm pas}(s) = V_{\rm rest} + \frac{\bar g_h(s)\, m_\infty(V_{\rm rest})\,
(V_{\rm rest} - E_h)}{g_{\rm pas}(s)},
\tag{D-005.2}$$
which makes $v \equiv V_{\rm rest}$ a stationary state of the whole cable
(no axial current under a uniform $v$) with $m(s,0) = m_\infty(V_{\rm rest})$
**[reasoning]**. $e_{\rm pas}$ therefore has status *computed*, recomputed at
every loss evaluation. Reported alongside $\theta$: $m_\infty(V_{\rm rest})$,
$\tau^h(V_{\rm rest})$, $V_{1/2}$, `gh_rest_over_gpas`
$= \bar g_h m_\infty(V_{\rm rest}) R_m$, `e_pas_soma_mV`.

**Why.** The three published kinetics differ by an order of magnitude in what
they say about the regime our steps visit: at $V_{\rm rest} = -73.5$ mV,
$m_\infty$ = 0.026 (Kole), 0.166 (Kole with the Kalmbach shift), 0.104 (Rich) and $\tau^h$ = 46
/ 73 / 341 ms **[run, from the repository's reference implementations]**;
fixing either curve would bake one of those choices into every $C_m$. The
user's call is to let the data place the curve and set its speed, and to
gate that with the synthetic recovery test before trusting it. Uniform
density is the human precedent (Kalmbach 2018 **[PMC]**; Rich 2021 soma +
basal **[KB]**), is defined for cells without a traced apical, and avoids the
`nseg = 1` gradient-sampling issue. Shifting $e_{\rm pas}$ keeps $V_{\rm rest}$
a measured constraint instead of a seventh parameter; the check that makes
(D-005.2) credible is that with the Kalmbach-shifted Kole, $\bar g_h = 10^{-4}$ S/cm²,
$R_m = 30$ kΩ·cm² it returns $-87.7$ mV against Kalmbach's fitted
uniform-passive $e_{\rm pas} = -86.97$ mV **[PMC, run]**.

**What it implies for code and tables.** The six-parameter box is the default
of `run_ih_fit.py`; the 3-D passive fit is the labelled control arm; the
Eyal law and the wide $\kappa_\tau$ box $[0.25, 3]$ are opt-ins whose outputs
carry the law / box in the file name and table. Both `.mod` files gain RANGE
`vshift_minf` and `tau_scale` (and `Ih.mod` a RANGE `vshift` for
$\Delta v_{\rm base}$), defaulting to the published models. The smoke suite
asserts the defaults (S1), the uniform law (S2), the stationarity of
$V_{\rm rest}$ (S3) and the six-parameter synthetic recovery (S9). A parameter
that fails the Stage-7 recovery gate is frozen at its base value for the
campaign and the failure is recorded, not hidden.

**What it does not decide [open].** The base kinetics (proposal: `Ih_human`,
with `Ih` $-20$ mV as the labelled alternative) -- confirmation C1; the exact
widths of the $\Delta v_h$ and $\kappa_\tau$ boxes beyond "slightly wider than
the human literature" (the values above are the assistant's reading of that
instruction: human $V_{1/2}$ $-90.3$ / $-90.9$ mV, human voltage-clamp $\tau^h$
maxima 400--500 ms); whether the axon stub carries $\bar g_h$ under the uniform
law (proposal: no); the Phase-3 bootstrap budget in six dimensions
(confirmation C5); the Stage-7 recovery tolerances.

**Sources.** Rich et al. 2021, Cereb Cortex, Eq. 1, Table 1, Eq. 5, Table 3,
and the Fig. 6 text on voltage-clamp $\tau^h$ **[KB]**; Kalmbach et al. 2018,
Neuron, Methods "Biophysical model" **[PMC]**; Eyal et al. 2016, eLife, Eq. 4
**[KB]**; `Ih.mod`, `Ih_human.mod`, `human_ih_params.py` **[src]**; $m_\infty$,
$\tau^h$, $V_{1/2}$ and the (D-005.2) check **[run]**; the decision itself
**[user]**, answers to Q1--Q5 and Q11 of the plan v0.1 on 2026-09-22.

**Status:** active. **IMPLEMENTED through Stage 6 on 2026-09-22**: the
mechanism layer, `ParamSpec`, the six-parameter fit, Phase 3 over the generic
vector, and `run_ih_fit.py` / `submit_ih_fit.sh` with the box above as the
default -- smoke 11/11 and the 3-D regression gate still bit-for-bit in the
assistant's sandbox and again from a clean extract, **pending the user's push
and cluster pull**. C1 and C5 were closed by D-007; the confirmed base is
`Ih_human`. What is NOT yet done is the thing that makes the six parameters
believable: the Stage-7 synthetic recovery gate. Until it has run, a fitted
$\Delta v_h$ or $\kappa_\tau$ is a number the optimiser returned, not a
measurement. Supersedes the v0.1 plan's recommendation of a 4-D vector (never
a decision). Companion of D-006.

---

## D-006 -- I_h campaign protocol: no I_h-suppressing guards, full step window, depolarising validation, Phase 3 kept, run B as baseline, monolith copy edited

**Date:** 2026-09-22. **Binds:** `load_cell_from_archive` (which long-square
sweeps become bundles and in which role), `bundle_rmsd` and the loss windows,
`run_ih_fit.py`, `submit_ih_fit.sh`, the campaign arms, Phase 3 in the new
folder, and every comparison table between the passive and the I_h
estimates. Specified in `TEEG_Ih_fit_staged_plan.md` v0.2 §5 and §8.

**Decision [user].** (i) Every guard that kept I_h out of the passive loss is
removed: the 12 mV deflection cap, the "two smallest steps" rule, the 60 / 150
ms window cut. (ii) The larger hyperpolarising long steps are trained on, to
make the sag a large signal, but not the single strongest, because the
conductances that shape the post-inhibitory rebound are not modelled. (iii)
Validation is on the small, spike-free **depolarising** long squares, with
the weaker hyperpolarising steps also held out. (iv) Phase 2.5 is not used;
Phase 3 (bootstrap CIs) is used and must be generalised to the new vector.
(v) The baseline is the last passive run, dt 0.1 ms, 60 ms window ("run B").
(vi) Scope: first the cells with re-calculated arbour diameters (the diameter
work aiming at Eyal's $C_m$, $R_m$, $R_a$), then the whole dataset. (vii)
Mechanisms are compiled on the cluster in the new folder with an idempotent
`nrnivmodl` guard. (viii) The work lives in a new folder that holds an edited
copy of the monolith, with a bit-for-bit passive regression gate against the
untouched `Biological Fit/`.

**Statement.** For a cell, let $h_1, \dots, h_n$ be its hyperpolarising
subthreshold long-square bundles in increasing $\lvert I \rvert$ and
$d_1, \dots, d_k$ its spike-free depolarising ones. Training bundles: the SS
hyperpolarising pulses (as today) and $h_2, \dots, h_{n-1}$, each long step on
the window $[t_{\rm on}, t_{\rm off}]$. Gate-validation bundles: $d_1 \dots
d_k$, $h_1$, the SS depolarising pulses. Report-only residuals: $h_n$ on the
step window; every long step on $[t_{\rm off}, t_{\rm end}]$ (the rebound).
The loss keeps its form, $L(\theta) = B^{-1}\sum_b \mathrm{RMSD}_b(\theta \mid
W_b)/D_b$. Three arms per cell at identical settings (dt 0.1 ms, 200/100):
`baseline_runB` (read from run B's CSV), `passive_fullstep` ($\theta$ 3-D on
this protocol), `ih6` (D-005). **The role table is the assistant's
reconciliation of the user's answers to Q6, Q7 and Q12 and is [open] until
confirmed (C2).**

**Why.** With I_h in the model every device that kept it out of the loss
removes the very signal that identifies $\bar g_h$, $\Delta v_h$ and
$\kappa_\tau$; the amplitude dependence of the sag across steps is the
strongest single constraint on the density, and the archive already holds
every long-square sweep of every cell, both polarities **[src,
`save_alleninstitute_data (6).py` block C2, verified 2026-09-22]**, so the cap
and the two-step rule were the only things excluding them. The strongest step
and the rebound are withheld because Na and Ca_LVA conductances shape the
post-inhibitory rebound in human L5 (Rich 2021 **[KB]**) and are absent here;
their residuals are reported so the exclusion is checked rather than assumed.
Depolarising steps probe I_h *de*-activation, a regime never trained on,
which is what a validation set should be. The passive full-step control arm
exists because a loss improvement of the I_h arm over run B could otherwise
be the window and the admitted steps rather than I_h **[reasoning]**. Editing a
copy of the monolith rather than monkey-patching around it is the user's
call; the regression gate is what makes it safe.

**What it implies for code and tables.** The loader builds depolarising
spike-free bundles into a new `CellData.long_square_depolarising` field
(default empty) and admits hyperpolarising sweeps beyond 100 pA; the role
assignment replaces `split_train_validation_with_long_step`; `--ls-window
{step, sweep}` defaults to `step`; the old guards survive only as opt-ins for
reproducing run B; `--skip-phase2p5` defaults to on; `PARAM_NAMES` and the
positional `loss_fn(x[0], x[1], x[2])` in Phase 3 give way to the `ParamSpec`;
`regression_passive_identity.py` (3-D, run-B configuration, same seed) is
re-run after every stage; every output row names its arm, its window and the
role of each sweep.

**What it does not decide [open].** The exact role table (C2); whether the
I_h arms must inherit run B's dt 0.1 ms, which the code's own note said was
to be validated before use (C3); the input contract for the diameter-
corrected morphologies -- new SWC files or a scale table -- and whether $F$
stays 1.9 (C4); the Phase-3 budget (C5); a trough-voltage guard default
(knob exists, default off); interneurons (excluded: both mechanisms are
pyramidal-cell fits).

**Sources.** The archive contents and the loader filter **[src]**; Rich et
al. 2021 on post-inhibitory rebound conductances **[KB]**; Allen ephys
whitepaper on the long-square protocol **[KB]**; the decision itself
**[user]**, answers to Q6--Q10 and Q12--Q14 on 2026-09-22.

**Status:** active; the role table was confirmed by **D-007** (C2) and is no
longer a proposal. **IMPLEMENTED through Stage 6 on 2026-09-22**: the loader,
the role assignment (`assign_ls_roles`, audited per cell into `ls_roles.csv`),
the three window modes, the arms (`baseline_runB`, `passive_fullstep`, `ih6`),
Phase 2.5 off by default, Phase 3 over the `ParamSpec`, and the report-only
residuals written to `ls_diagnostics.csv` with the model's and the data's sag
fraction side by side -- so item (ii)'s exclusion is checked, as the entry
requires. Two implementation departures from `run_biological_fit.py` that this
entry did not anticipate are recorded in **D-010**. Pending the user's push
and cluster pull; the campaign itself (Stage 8) has not been run. Companion of
D-005. Supersedes the v0.1 plan's "keep the 12 mV cap, raise it to ~30 mV" and
"hold out the largest step" recommendations (never decisions). **The spike
screen that implements Q7 was corrected on 2026-09-23 -- see D-016**; Q7's
intent (spiking sweeps excluded by waveform) is unchanged.

---

## D-007 -- Confirmations closing the open points of D-005 and D-006

**Date:** 2026-09-22 (later). **Binds:** the same objects as D-005 and
D-006; closes their items C1--C5 and the "more amplitudes" reading of Q1.

**Decision [user].** (C1) Base kinetics `Ih_human` (Rich 2021); `Ih` with
$\Delta v_{\rm base} = +20$ mV (the Kalmbach 2018 shifted Kole; see the sign
note in D-005) is the labelled alternative.
(C2) The role table of D-006 stands: train on the SS hyperpolarising pulses
and $h_2 \dots h_{n-1}$; gate-validate on the spike-free depolarising steps,
$h_1$ and the SS depolarising pulses; $h_n$ and every rebound window are
report-only. (C3) dt = 0.1 ms in every arm, inherited from run B. (C4) $F$
is chosen per layer and type; **during the L2 and L3 optimisation $F = 1.9$**.
(C5) Phase-3 bootstrap budget 100/30 per replicate, $B = 200$. On the
identifiability burden of the six-parameter vector: **use more long-square
amplitudes** -- every hyperpolarising amplitude the archive holds is admitted
(no current cap), so $n$ is as large as the recording allows.

**Statement.** `--ih-mechanism Ih_human` is the default; `--ih-mechanism Ih
--vshift-base 20` the labelled alternative. `ls_max_amplitude_pA` in the
loader becomes unbounded by default (every sweep with
`detected_amplitude_pA < -5` and no spike is a candidate); the roles are then
assigned by amplitude rank as in D-006. With C2 confirmed, $h_1$ stays in
validation; admitting it to training is the opt-in `--ls-train-roles
all_hyp`, labelled. `F = 1.9` is the default of `run_ih_fit.py` for `L2_exc`, `L3_exc`;
other groups take the per-group table of `submit_all_groups_biological.sh`.
Phase 3: `--bootstrap-n-calls 100 --bootstrap-n-initial 30 --bootstrap-B 200`.

**Why.** The identifiability of $\Delta v_h$ rests on the amplitude
dependence of the sag fraction, so the number of distinct amplitudes in
training is the lever the user chose to pull rather than reducing the
parameter count **[user]**. The rest are confirmations of the D-005/D-006
proposals.

**What it implies.** Stage 2 (loader) removes the cap by default; Stage 6
sets the defaults above; the smoke suite asserts them (S7, and a new check
that `Ih_human` is the resolved default).

**What it does not decide [open].** The input contract for the
diameter-corrected morphologies (new SWC per specimen vs. a scale table) --
needed by Stage 2's `--swc-dir`, not by Stages 0--1; the Stage-7 tolerances;
the trough-voltage guard default (off).

**Sources.** **[user]**, 2026-09-22 16:06.

**Status:** active. Closes the [open] items C1--C5 of D-005 and D-006. All six
are in force as the defaults of the Stage-6 entrypoint as of 2026-09-22:
`--ih-mechanism Ih_human`, the role split with `--n-drop-weakest 1
--n-drop-strongest 1` (and `--train-all-hyp` as C2's labelled opt-in, which is
what "`--ls-train-roles all_hyp`" became), `--dt-brief-ms 0.1 --dt-long-ms
0.1`, `--F 1.9`, `--bootstrap-n-calls 100 --bootstrap-n-initial 30
--bootstrap-B 200`, and `ls_max_amplitude_pA = None` in the two I_h arms. The
morphology input contract (C4's other half) remains open and is the one thing
blocking Stage 8's first scope. **Closed 2026-09-23 by D-013**: one SWC per
specimen, in the archive's own format, through `--swc-dir`.

---

## D-008 -- Two implementation conventions of the I_h refactor

**Date:** 2026-09-22 (later, 4). **Binds:** `Ih Fit/passive_fitting_hpc_fixed.py`
(`load_cell_from_archive` defaults), `Ih Fit/param_spec.py`, every loss built
by `Ih Fit/passive_long_step_training.py`, and every future call site that
builds or evaluates a loss in this pipeline.

**Decision [assistant, under D-006 Q14's "edit a copy, keep a regression
gate"].** (i) The LOADER keeps its legacy defaults; admitting every
hyperpolarising amplitude and building the depolarising bundles is opt-in at
the call site (`ls_max_amplitude_pA=None, load_depolarising_ls=True`), and the
I_h orchestrator is what passes them. (ii) Every loss is **variadic positional
in the spec's axis order**, `loss(*q)`.

**Statement.** Let $\mathcal{S}$ be a `ParamSpec` with axes
$(a_1, \dots, a_p)$ and $q \in \mathbb{R}^p$ the optimiser coordinates, where
$q_i = \log(\theta_i)$ for a positive axis and $q_i = \theta_i$ for a linear
one (only $\Delta v_h$ is linear). Then every loss satisfies
$$L_{\mathcal{S}}(q_1, \dots, q_p) \in \mathbb{R}_{\ge 0},
\tag{D-008.1}$$
and for $\mathcal{S} = $ `PASSIVE_3D` this is exactly the legacy
$L(\log C_m, \log R_m, \log R_a)$, argument for argument.

**Why.** (i) D-007 says the campaign admits every amplitude; it does not say
the low-level loader function must change what it returns to a caller that
asks for nothing. Had the default changed, `regression_passive_identity.py`
would have loaded different bundles and its PASS would have meant nothing --
the gate would have been spent to buy a default. The campaign behaviour is
unchanged: `run_ih_fit.py` passes the no-cap arguments. (ii) A mapping-based
loss, `loss(theta_dict)`, would have been the obvious generalisation and would
have broken `_refit_from_bundles`'s `loss_fn(x[0], x[1], x[2])`,
`cm_profile_sweep.profile_cm`'s `loss(lcm, x[0], x[1])` and the regression
harness's own `loss(*q)` -- three call sites that the variadic form leaves
untouched while still making the arity a property of the spec.

**What it implies for code and tables.** `load_cell_from_archive(d)` returns
what it always returned; `CellData.long_square_depolarising` is empty unless
asked for. A new loss or refit helper takes `*q`, never three named
arguments. `param_spec.flatten_result_params(result)` is what writes a 6-D
theta to CSV, because `results_to_dataframe` emits a fixed column list and
would otherwise drop it silently.

**What it does not decide [open].** The morphology input contract for the
diameter-corrected arbours: `swc_dir` is implemented and resolves
`<swc_dir>/<id>.swc`, `<swc_dir>/specimen_<id>.swc` or
`<swc_dir>/specimen_<id>/reconstruction.swc`, and REFUSES rather than falling
back if none exists -- but whether the corrected morphologies will arrive as
SWC files at all, or as a per-cell scale table applied at build time, is
still the user's to say.

**Sources.** The three call sites and the fixed column list read from source
2026-09-22 **[src]**; the regression argument **[reasoning]**.

**Status:** active. Refines the implementation of D-006 and D-007; reverses
nothing. Both conventions held through Stages 5 and 6: the 3-D regression gate
is still bit-for-bit after the Phase-3 generalisation, and the variadic form
is what let `cm_profile_sweep` stay untouched -- which is also, exactly, why
the per-cell $\tau_w$ sweep could not be carried into the six-dimensional arms
(D-010 (i)). `flatten_result_params` is the writer of `phase2_results.csv`'s
theta columns, and it also emits the derived I_h quantities, so nothing else
may write them a second time under a prefix. The morphology-contract point of
"what it does not decide" is **closed by D-013** (2026-09-23): new SWC files,
which is exactly what `swc_dir` already resolves; no scale table is needed.

---

## D-009 -- The kappa fill stays and is reported, shaft_terminates are spines, and the campaign has 200 CPUs

**Date:** 2026-09-22 (later, 5). **Binds:** `--min-coverage` and `qc_status`
in `merge_spine_area_F.py` / `h01_spine_area_F.assemble_cell` (what they are
FOR, not their values); every table or figure that reports an $F$ beside its
mesh coverage; `campaign_policy_report.py`; the `select=` and `walltime`
lines of `campaign.pbs` and the defaults of `campaign_queue_check.py`. Closes
the coverage-threshold and `shaft_terminates` points of D-004's "what it does
not decide", and the queue-limit item of
`TEEG_campaign_single_submission_handoff_2026-09-22.md` section 10.

**Decision [user].** (i) The kappa fill is correct and stays: a spine that
the mesh did not measure keeps its skeleton area rescaled by kappa, no cell
is dropped and no coverage threshold is set -- what is required instead is
that the fraction of the reported spine area which did not come from the mesh
is reported alongside $F$. (ii) `shaft_terminates` components are treated as
spines ("use the spine's one"), i.e. kept with their kappa-filled area rather
than demoted out of the spine set. (iii) The campaign allocation is 200 CPUs
usable at once, with a walltime ceiling of 300 h.

**Statement.** For a cell $c$, let $S(c)$ be its spine set after the P0
partition, $M(c) \subseteq S(c)$ the spines measured on the mesh
(`measured_beyond`), and $A^{\rm used}(\sigma) \in \mathbb{R}_{\ge 0}$
(um$^2$) the area entered for spine $\sigma$ in the deliverable track -- the
mesh value for $\sigma \in M(c)$ and $\hat\kappa(A^{\rm skel}(\sigma))\,
A^{\rm skel}(\sigma)$ otherwise. (i) says $S(c)$ is never reduced: no
$\sigma$ is removed for being unmeasured, and $F^{\rm mesh\_beyond}(c)$ is
computed over all of $S(c)$, for every $c$, whatever its coverage. The
reported companion quantity is, for the segments beyond the cutoff
$d \ge 60$ um that $F_{\rm lit}$ sees,
$$\phi_{\rm fill}(c) = \frac{\sum_{\sigma \in S(c) \setminus M(c),\, d(\sigma) \ge 60}
A^{\rm used}(\sigma)}{\sum_{\sigma \in S(c),\, d(\sigma) \ge 60} A^{\rm used}(\sigma)}
\in [0, 1],
\tag{D-009.1}$$
reported per cell as `frac_area_filled_beyond` and, over a population $P$,
**pooled as a ratio of sums**, $\phi_{\rm fill}(P) = \sum_{c \in P}
{\rm num}(c) / \sum_{c \in P} {\rm den}(c)$, never as a mean of the
per-cell ratios -- the same convention `h01_spine_area_F.kappa_function` uses,
and for the same reason (a tiny cell with one unmeasured spine would
otherwise weigh as much as a large one). P3's own
`fallback_area_frac_deliverable` is the same construction over ALL spines of
the cell rather than only those beyond the cutoff; the two are different
quantities and are named apart. (ii) says
$\{\sigma : {\tt base\_verdict}(\sigma) = {\tt shaft\_terminates}\}
\subseteq S(c)$, which is what the code already does. (iii) with
`select=1:ncpus=2` the allocation runs $\lfloor 200/2 \rfloor = 100$ subjobs
concurrently, so a population of $N$ cells at $S$ shards takes
$\lceil NS/100 \rceil$ waves; the walltime ceiling bounds the per-subjob
request, not the campaign.

**Why.** (i) The fill is not a defect to be thresholded away but the stated
fallback policy of the pipeline (handoff section 6: "spine area from the
mesh, skeleton only as fallback"), and excluding cells would bias the
population toward well-imaged arbours without making any individual $F$ more
truthful. Reporting the fraction keeps the hybrid visible wherever $F$ is
used, which is what D-004's lesson demands, without inventing a cutoff no
measurement supports. (ii) A component whose union cross-sections never
separate from the dendrite is one the base finder could not resolve, not one
shown to be shaft; demoting it would silently remove real membrane from $F$
on the strength of a failed measurement. Keeping it puts it on the same
footing as any other unmeasured spine, where (D-009.1) already accounts for
it. (iii) **[user]**, reported allocation. All three **[user]**.

**What it implies for code and tables.** No measurement code changes.
`campaign_policy_report.py` v1.0 reports (D-009.1) per cell and pooled and
keeps `drop_unmeasured` / `drop_shaft_terminates` / `drop_not_reaching_box`
as labelled comparison BRACKETS -- never the reported $F$, and never a
licence to re-merge, being counterfactuals on recorded areas rather than
re-measurements. Its smoke suite asserts the bracket labels, the pooled
arithmetic (ratio of sums, checked against a fixture where the mean of ratios
differs) and that no threshold is derived. `campaign_queue_check.py` v1.1
separates the qstat limits (which can refuse a submission, and set its exit
status) from the CPU budget (which cannot, and only sets throughput), reading
`ncpus` and `walltime` from `campaign.pbs`'s own `#PBS -l` directives so the
two cannot drift; `--hours-per-task` turns the budget into a wall-clock and
CPU-hour estimate. `campaign.pbs` documents the allocation and that its 12 h
walltime is a per-subjob guess overridable with `qsub -l walltime=...` up to
300 h.

**What it does not decide [open].** The per-spine measurement rate, still
unmeasured, and therefore `SHARDS` and the walltime actually to request. What
value of (D-009.1) would make a cell unusable for a given downstream purpose
-- the number is reported, and any such rule is a later decision that cites
this one. Whether the reported fraction should also travel into
`campaign_F.csv`'s printed summary (it is already a column).

**Sources.** The decision itself **[user]**, 2026-09-22 17:30. The fill and
the `base_verdict` vocabulary read from source the same day
(`h01_spine_area_F.assemble_cell` tracks and `_track`, `h01_spine_base`
`base_verdict`) **[src]**. The pooled convention from
`h01_spine_area_F.kappa_function`'s own docstring **[src]**. The arithmetic
of (D-009.1) and the wave count **[reasoning]**, verified on fixtures
**[run]**.

**Status:** active. Closes the coverage-threshold and `shaft_terminates`
open points of D-004 and the queue-limit item of the campaign handoff's
section 10; supersedes nothing. Numbered D-009 rather than D-008 because the
I_h chat logged its own D-008 earlier the same afternoon.

---

## D-010 -- Two departures of the I_h orchestrator from `run_biological_fit.py`, and the crash-penalty rule for every liveness check

**Date:** 2026-09-22 (later, 6). **Binds:** `Ih Fit/run_ih_fit.py` (the
`--ss-tau-w-ms` contract and the `gate_valid_via` field of every `ArmConfig`),
`Ih Fit/submit_ih_fit.sh`, the `validation_status` column of every I_h-arm
`phase2_results.csv`, and every present and future check in this pipeline that
asserts a loss "works".

**Decision [assistant, under D-006 Q14 and D-007].** (i) The I_h orchestrator
does **not** run the two-pass auto-$\tau_w$ loop; `--ss-tau-w-ms` takes one
value and a comma-separated grid is refused rather than collapsed. (ii) For
the `passive_fullstep` and `ih6` arms the absolute-mV validation gate windows
the held-out bundles with the **same** convention as the training bundles;
`baseline_runB` keeps the legacy asymmetric early-window gate. (iii) No check
in this pipeline may certify a loss as working by `isfinite` alone: it must
assert the value is strictly **below** the builders' crash penalty and that it
**varies**.

**Statement.** (i) Let $\mathcal{S}$ be a `ParamSpec` with $p$ axes. The
per-cell $\tau_w$ selection of `run_biological_fit.py` calls
`cm_profile_sweep.profile_cm`, which for each fixed $\log C_m$ minimises over
exactly two nuisance coordinates, `loss(lcm, x[0], x[1])` **[src]**; it is
therefore defined only for $p = 3$ and raises for $p > 3$. Run B was
configured with `TAU_W_GRID_MS="5.0"`, a single-point grid, so its winner was
$\tau_w^\star = 5.0$ ms by construction and no selection took place
**[src]**. Hence for every arm here $\tau_w \equiv \tau_w^{\rm cli}$, written
to `tau_w_chosen_ms` with `tau_w_reason = "fixed_by_cli"`, and a grid of
length $> 1$ raises `SystemExit`.

(ii) For a bundle $b$ and the fit point $\hat\theta$, let
$\mathrm{RMSD}_b(\hat\theta \mid W)$ (mV) be the baseline-subtracted RMS
deviation over window $W$. `run_biological_fit.py` forms the gate from
$\{\mathrm{RMSD}_b(\hat\theta \mid W_b^{\rm arm})\}_{b \in {\rm train}}$ and
$\{\mathrm{RMSD}_b(\hat\theta \mid [t_{\rm on}, t_{\rm on} + 150\,{\rm ms}])\}_{b \in {\rm valid}}$,
the second window being documented in source as chosen "to minimise Ih
contamination in the validation signal" **[src]**. Under `gate_valid_via =
"same_window"` both sets use $W_b^{\rm arm}$, which for the I_h arms is the
whole step $[t_{\rm on}, t_{\rm off}]$. The classifier, its thresholds and the
fitted $\hat\theta$ are untouched; only the window of the validation side
changes. `gate_valid_via = "legacy_early"` restores the asymmetric form and is
the `baseline_runB` default.

(iii) Both loss builders wrap their body in `except Exception: return` $P$ with
$P = 10^{6}$ **[src]**. $P$ is finite, so for a loss $L$ evaluated at points
$q^{(1)}, \dots, q^{(k)}$ the predicate $\bigwedge_i |L(q^{(i)})| < \infty$ is
satisfied by a loss whose every simulation raised. A liveness check must
instead assert
$$\max_i L(q^{(i)}) < P \quad\text{and}\quad \max_i L(q^{(i)}) - \min_i L(q^{(i)}) > \varepsilon,
\tag{D-010.1}$$
with the probe points differing in at least one axis ($C_m$ is used).

**Why.** (i) Running a 3-D-only profiler inside a 6-D campaign would raise
per cell inside an orchestration loop that logs and continues, so the arm
would silently lose cells; collapsing a grid to its first entry would instead
report a $\tau_w$ that was never selected. Refusing is the only option that
neither hides a failure nor invents a provenance. Nothing is lost relative to
the baseline, because the baseline's grid had one point.

(ii) The held-out set exists to test the part of the response the fit was not
shown. Under D-006 that is the I_h regime: the depolarising steps probe
de-activation and $h_1$ probes the weak-stimulus end. A window explicitly
designed to suppress I_h contamination removes from the gate the thing the
gate is for, and it also puts the two sides of `valid_to_train_ratio_abs` on
different windows, so the ratio is not a ratio of comparable quantities
**[reasoning]**. `baseline_runB` keeps the legacy form because reproducing run
B, window for window, is that arm's only purpose.

(iii) Found by measurement, not by inspection: smoke check S6 built its NEURON
cell **before** `make_ih_archive`, whose generator builds its own cell and
clears `h.allsec()`, orphaning the first cell's `IClamp`. All eight loss
evaluations then raised `TypeError` inside `simulate()`, were swallowed, and
returned $10^6$ -- which `isfinite` accepts. The check had been reporting
"spec loss finite at 8 random theta" while exercising nothing. Verified
against the untouched `Biological Fit/` oracle that this was an artefact of
that fixture and not of the pipeline: the oracle's own non-parametric
bootstrap on the same synthetic cell keeps all 14 replicates with zero crashed
evaluations, and so does the production pattern (build, fit, destroy, build,
bootstrap) in the new folder **[run]**.

**What it implies for code and tables.** `run_ih_fit.py` exposes
`--ss-tau-w-ms` (scalar) and `--gate-valid-via {legacy_early, same_window}`;
`ArmConfig` carries `gate_valid_via` so the choice is a property of the named
arm and appears in the run header. A column comparison across arms must not
read `valid_rmsd_abs_mV` of `baseline_runB` against that of `ih6` as like for
like: they are RMSDs over different windows, and the comparison that is valid
is `passive_fullstep` against `ih6`, which share a protocol. `smoke_ih_fit.py`
S6, S11 and S12 and `run_ih_fit._assert_loss_live` all implement (D-010.1); a
new check that asserts only finiteness is a regression of this decision. For a
real cell the orchestrator's `_assert_loss_live` runs before the fit and names
the two usual causes in its error text (no `x86_64/` in the process working
directory; sections deleted by a later `build_neuron_model`).

**What it does not decide [open].** Whether a per-cell $\tau_w$ is worth
having in six dimensions at all -- it would need `profile_cm` generalised to
profile one named axis over $p - 1$ nuisance axes, which Stage 5 did for the
Phase-3 profile (`_other_axes_log_grid`) but not for `cm_profile_sweep`. The
morphology input contract (still D-007's and D-008's open point). The Stage-7
recovery tolerances.

**Sources.** `profile_cm`'s two-nuisance minimisation, run B's single-point
grid, `_rmsd_for_validation_bundle`'s window and its stated rationale, and the
two `return 1e6` penalties all read from source 2026-09-22 **[src]**. The
oracle comparison and the crash counts **[run]**. The reasoning on gate
symmetry **[reasoning]**.

**Status:** active. Refines the implementation of D-006 (iii) and (v) and of
D-007; reverses nothing. Numbered D-010 because D-009 was taken by the H01
campaign chat on the same afternoon.

---

## D-011 -- Stage 7's cohort: real morphologies, a false-positive control, measured noise, and the two error measures

**Date:** 2026-09-22 (later, 7). **Binds:** `Ih Fit/synth_gt_grid.py` (what a
Stage-7 manifest contains and how each column is drawn),
`Ih Fit/run_ih_recovery.py` and `submit_ih_recovery.sh` (the cohort's
defaults), `Ih Fit/ih_recovery_report.py` (every recovery number and the gate
verdict), and any later statement of the form "the six-parameter fit recovers
(or does not recover) X".

**Decision [user], 2026-09-22 18:40.** (i) The synthetic cohort is drawn on
the **real archive morphologies** -- `<ARCHIVE_ROOT>/<GROUP>/specimen_*/
reconstruction.swc`, the cells the campaign will actually fit -- not on a
ball-and-stick. (ii) A quarter of the cohort carries **no I_h**: a
**false-positive control**, fitted with the same six-parameter arm as the
rest. (iii) The recording noise injected into the synthetic cells is the
**median measured `noise_sigma_mV` of run B**, not the benchmark's nominal.

**Decision [assistant], forced by (i)-(iii).** (iv) The error of a parameter
is measured in the coordinate the parameter is searched in: a **log ratio**
for the five positive axes and a **difference in mV** for $\Delta v_h$, the
one linear axis. (v) The I_h axes are summarised over the **I_h-bearing cells
only**. (vi) A **railed** axis is counted and reported, never silently
dropped from a median.

**Statement.** Let $\mathcal{C}$ be the cohort, $\mathcal{C}_{\rm fp}
\subset \mathcal{C}$ the false-positive-control cells, and for a cell
$c \in \mathcal{C}$ and axis $i$ let $\theta_i(c) $ be the injected ground
truth and $\hat\theta_i(c)$ the estimate of a given arm.

(i) Each $c$ is generated on an SWC read from the archive, so the map from
parameters to predictions is the cell's own; $F = 1.9$ throughout (D-007 C4).

(ii) $\lvert \mathcal{C}_{\rm fp} \rvert / \lvert \mathcal{C} \rvert = 1/4$,
selected **by index** (every 4th cell) and therefore consuming no randomness,
so adding or removing the control cannot move any other cell's draw. For
$c \in \mathcal{C}_{\rm fp}$: $\bar g_h(c) = 10^{-6}$ S/cm$^2$ (the search-box
floor), $\Delta v_h(c) = 0$, $\kappa_\tau(c) = 1$.

(iii) With $\sigma_{\rm runB}$ the median of run B's per-cell
`noise_sigma_mV`, the generator's white-noise level is $\sigma_{\rm runB}$ and
its two slow components (per-sweep baseline offset, per-sweep drift) are
scaled by $\sigma_{\rm runB}/\sigma_{\rm default}$, so the *character* of the
noise is preserved while its level is measured.

(iv) For each fixed $c$, with $\mathcal{A}_{\log} = \{C_m, R_m, R_a,
\bar g_h, \kappa_\tau\}$ and $\mathcal{A}_{\rm lin} = \{\Delta v_h\}$,
$$
\varepsilon_i(c) =
\begin{cases}
\log\!\bigl(\hat\theta_i(c) / \theta_i(c)\bigr), & i \in \mathcal{A}_{\log}
  \ \text{(dimensionless)},\\[2pt]
\hat\theta_i(c) - \theta_i(c), & i \in \mathcal{A}_{\rm lin}\ \text{(mV)}.
\end{cases}
\tag{D-011.1}
$$
The gate compares $\operatorname{median}_{c}\lvert\varepsilon_i(c)\rvert$ to a
per-axis tolerance.

(v) For $i \in \{\bar g_h, \Delta v_h, \kappa_\tau\}$ the median in (D-011.1)
is taken over $\mathcal{C} \setminus \mathcal{C}_{\rm fp}$; likewise the
$C_m$-inflation read-out of the passive arms. $\mathcal{C}_{\rm fp}$ has its
own two statistics: the fraction of its cells whose $\hat{\bar g}_h$ sits on
the **low** bound, and $\lvert \varepsilon_{C_m} \rvert$ on those same cells.

(vi) An axis on a bound is recorded with the side it railed on; the rail
counts appear in every summary row, and excluding railed cells from a median
is an explicit opt-in (`--drop-railed`), off by default.

**Why.** (i) Identifiability of $\bar g_h$ and $\Delta v_h$ rests on the
amplitude dependence of the sag, and how much sag a step produces depends on
how much membrane sits far enough from the soma to be dominated by the h
conductance -- a property of the arbour. A uniform cable is the easiest
possible case for separating $R_m$ from $\bar g_h$, so a pass on
ball-and-stick would not license the campaign **[reasoning]**.

(ii) A recovery test alone cannot detect the failure mode that matters most
on real data: a six-parameter fit that always finds *some* I_h would recover
the truth on every I_h-bearing cell and still be wrong on every passive one,
inflating nothing in the test and everything in the campaign. The control
asks the complementary question -- and the read-out costs nothing, because
`rail_gbar` is already written per cell (D-008) **[reasoning]**.

(iii) The I_h signal in human L2/3 is a few mV on deflections of 10-20 mV
(Moradi Chameh et al. 2021, sag amplitude $1.8 \pm 1.6$ mV **[KB]**), so
whether it is recoverable at all is a statement about the noise. Generating at
a nominal cleaner than the bench would make the gate pass on data the campaign
will never see **[reasoning]**.

(iv) A log ratio is undefined for a quantity that takes both signs and passes
through zero, which $\Delta v_h$ does by construction (its zero is the
published activation curve, a modelling convention, not an origin). Measuring
each axis in the coordinate it is searched in also makes the tolerance a
tolerance in the optimiser's own metric. The smoke's fixture contains a cell
with $\Delta v_h = 0$ exactly, so a report that silently logged this axis
cannot produce the fixture's hand-computed answer **[run]**.

(v) On a cell with $\bar g_h$ at the floor, $\Delta v_h$ and $\kappa_\tau$
multiply a current that is not there: they are unidentifiable, and their
"error" is whatever the optimiser last touched. Including those cells would
move the median for a reason unrelated to recovery **[reasoning]**.

(vi) Dropping the railed cells is how a gate passes. The counts are the
evidence that it did not need to.

**What it implies for code and tables.** `synth_gt_grid.MANIFEST_COLUMNS`
gains `ih_regions`, `ih_vshift_base_mV`, `ih_dvh_mV`, `ih_kappa_tau`,
`is_fp_control`, and `load_manifest` back-fills them so a pre-Stage-7 manifest
still generates the ground truth it always generated. The kinetic draws come
from a FOURTH `SeedSequence` child, **appended**: `spawn(4)` returns the same
first three children as `spawn(3)`, so no earlier draw moves (asserted
column-by-column against the untouched `Synthetic Passive fit Test/` copy).
`noise_sigma_from_results` reads the level out of a real run's
`phase2_results.csv`; when it is not given, the run header prints
`*** BENCHMARK DEFAULT, NOT MEASURED ***`. `gen_from_manifest.build_proto`
now forwards `ls_dep_amplitudes_pA` -- **it did not before, so every archive
the benchmark ever generated had an empty depolarising Long Square set**, and
under the D-006 protocol those steps *are* the validation set. The gate is
applied to the `ih6` arm only; a failing kinetic knob is frozen per D-005
(`--fit-params` without it), and a failing $C_m$ or $\bar g_h$ is not, because
a smaller spec does not fix a protocol that cannot see the parameter.

**What it does not decide [open].** **The tolerances themselves.** The values
in force -- median $\lvert\varepsilon\rvert \le \log 1.25$ for $C_m$ and
$\bar g_h$, $\le 3$ mV for $\Delta v_h$, $\le \log 1.5$ for $\kappa_\tau$, and
$\ge 80\,\%$ low-rail on the control -- are the assistant's proposal carried
over from the plan, not a user decision. Every per-cell error is written to
`recovery_per_cell.csv`, so a different tolerance is re-applied without
re-running anything. Also open: the cohort size (20 by default, never
justified by a power calculation); whether the cohort should span L2_exc as
well as L3_exc; whether a second cohort at a higher noise level is worth the
cost; and the morphology input contract of D-007 C4, unchanged.

**Sources.** The decisions **[user]**, 2026-09-22 18:40. The sag amplitudes
Moradi Chameh et al. 2021 **[KB]**. `profile_cm`, `build_proto`'s missing
forward, `rail_flags` and the manifest schema read from source the same day
**[src]**. The fixture arithmetic and the RNG-stream invariance **[run]**.
The identifiability, control-design and measure arguments **[reasoning]**.

**Status:** active. Refines D-005's Stage-7 gate and D-006's protocol; adds
the false-positive control, which neither anticipated. The gate has **not yet
been run** -- this entry records how it will be measured, not what it found.
**Item (iii) is superseded by D-012 (2026-09-23)**: run B's `noise_sigma_mV`
is measured on AVERAGED bundles and mixes protocols, so it is not the per-sweep
noise the generator injects, and $\rho$ was never set. Items (i), (ii) and
(iv)-(vi) stand unchanged. **Why (iii)'s number is corrected by D-017
(2026-09-28)**: $1.8 \pm 1.6$ mV is Moradi Chameh's L2&3 rebound
depolarisation; the L2&3 sag is $1.7 \pm 1.1$ mV. The argument stands.

---

## D-012 -- Stage 7's recording noise is measured per sweep on the real cells, with its correlation; run B's noise column is not a generator input

**Date:** 2026-09-23. **Binds:** `Ih Fit/noise_calibration.py`; the noise
columns of a Stage-7 manifest (`noise_sigma_mV`, `noise_rho_lag1`,
`noise_source`, and the two slow-noise columns) written by
`synth_gt_grid.draw_manifest`; `gen_from_manifest.row_to_noise_kwargs` and
`build_proto`; `run_ih_recovery.phase_noise` and `resolve_protocol`; the noise
knobs of `submit_ih_recovery.sh`; every Stage-7 recovery number and verdict.
**Supersedes D-011 (iii).**

**Decision [assistant, implementing the user's D-011 (iii) intent].** The
user's instruction (2026-09-22) was to generate the synthetic cohort at the
noise the campaign's cells actually carry, "matched to run B's measured
noise"; on 2026-09-23 the user asked for run B to be traced on the HPC. Doing
that showed that run B's `noise_sigma_mV` column cannot serve: it is a
different object from the one the generator takes. So: (i) each synthetic
cell carries the **per-sweep** fast noise $(\sigma, \rho)$ of the real archive
cell whose morphology it borrows, read from that cell's **single** sweeps
with the fitter's own estimator; (ii) the Long Square measurement drives the
generator by default, the SS one is recorded beside it; (iii) the generated
protocol takes the real cells' SS pulses per polarity and both sampling rates;
(iv) a synthetic cell with no usable measurement falls back to the nominal
level and is labelled so, never silently. The user can overrule any of these;
the route through run B is gone from the code, not hidden behind a flag.

**Statement.** Let $c$ be a real archive cell, $p \in \{\mathrm{LS}, \mathrm{SS}\}$
a protocol, and $k = 1, \dots, K_p(c)$ its **single** sweeps of that protocol
(a Long Square bundle with `n_repeats_averaged` $= 1$; one Square Subthreshold
pulse). On the pre-stimulus window of sweep $k$ -- $t < 0$ for an SS pulse,
$[t_0, t_{\rm on} - 5\,{\rm ms})$ for an LS sweep [src, `_estimate_noise`] --
let $x_{k,1}, \dots, x_{k,n_k}$ (mV) be the samples,
$\bar x_k = n_k^{-1}\sum_j x_{k,j}$ and $u_{k,j} = x_{k,j} - \bar x_k$. The
fitter's estimator returns
$$
s_k = \Bigl(\tfrac{1}{n_k - 1}\textstyle\sum_{j=1}^{n_k} u_{k,j}^2\Bigr)^{1/2},
\qquad
r_k = \dfrac{\tfrac{1}{n_k - 1}\sum_{j=1}^{n_k - 1} u_{k,j}\,u_{k,j+1}}
             {\tfrac{1}{n_k}\sum_{j=1}^{n_k} u_{k,j}^2},
\tag{D-012.1}
$$
and the calibration records, for each fixed $c$ and $p$,
$\hat\sigma_p(c) = \operatorname{median}_k s_k$ (mV) and
$\hat\rho_p(c) = \operatorname{median}_k r_k$ (dimensionless), together with
the sampling rate, the number of sweeps, the SS pulses per polarity and the
median window length $n_k$ **[src]**. The generator injects into every sweep
of a synthetic cell $c'$ drawn on $c$'s morphology the stationary AR(1)
sequence
$$
e_1 \sim \mathcal{N}(0, \sigma^2), \qquad
e_j = \rho\, e_{j-1} + \eta_j, \quad
\eta_j \sim \mathcal{N}\bigl(0, \sigma^2 (1 - \rho^2)\bigr),
\tag{D-012.2}
$$
with $(\sigma, \rho) = (\hat\sigma_{\rm LS}(c), \hat\rho_{\rm LS}(c))$ by default
**[src, `add_recording_noise`]**, so that $\operatorname{Var}(e_j) = \sigma^2$ and
the lag-1 correlation is $\rho$ for every $j$.

The fitter's `PassiveFitResult.noise_sigma_mV` is instead $s$ of
(D-012.1) evaluated on each **averaged training bundle** and averaged over the
training bundles **[src]**. An SS bundle is the mean of $N$ independent pulses,
so its fast noise has standard deviation $\sigma / \sqrt{N}$ (textbook, from
memory); an LS bundle is one sweep. Run B's number is therefore a
protocol-dependent mixture that lies below the per-sweep $\sigma$ by up to a
factor $\sqrt{N}$ on its SS part, and feeding it in as the generator's
$\sigma$ would produce a cohort cleaner than the data. The lag-1 coefficient,
by contrast, is **unchanged** by averaging independent AR(1) sweeps that share
$\rho$ (the mean is again AR(1) with the same $\rho$ and variance $\sigma^2/N$)
**[reasoning]**, so run B's report that $\rho_{\rm lag1} > 0.5$ in 61 of its 68
cells **[run, computed 2026-09-19 from run B's CSVs]** does describe the
per-sweep noise, and it is what (ii) of the "Why" rests on.

**Why.** (i) **The object.** The generator's `NoiseConfig.sigma_mV` is the
fast noise of one sweep *before* averaging; the fitter's `noise_sigma_mV` is
measured *after* it and pooled across protocols. Two objects, two names
(Precision R4); the first must be measured as itself.
(ii) **The correlation.** No caller ever set `NoiseConfig.rho_lag1`, so every
synthetic cohort the benchmark has produced was white noise. For an AR(1)
sequence the variance of a mean over $n$ samples is approximately
$(\sigma^2/n)(1+\rho)/(1-\rho)$ for large $n$ (textbook, from memory; the
fitter's own docstring states the equivalent $(1-\rho)/(1+\rho)$
effective-sample reduction **[src]**): at $\rho = 0.7$ white noise overstates
the information in a window about 5.7-fold. A gate passed on white noise would
be a gate passed on data more informative than the bench's.
(iii) **The protocol.** $\rho$ is defined per sample, so the same process
sampled at another rate has another $\rho$; and the SS pulses per polarity set
how far the SS bundle's noise is averaged down. Both are therefore read from
the real cells and recorded with their source (`run_config.json` ->
`protocol`, precedence command line > measured > generator default).
(iv) **Long Square as the default.** The long steps are where $\bar g_h$,
$\Delta v_h$ and $\kappa_\tau$ are identified -- the parameters Stage 7 exists
to test. A second, measured reason: the estimator (D-012.1) demeans the
window, and $n_k - 1$ corrects that only for independent samples. On a short,
correlated window the mean absorbs part of the variance and $s_k$ reads low.
Monte Carlo of the exact statistic (median over sweeps of (D-012.1)) on the
law (D-012.2), $\sigma = 1$, 4000 replicates **[run]**: 16 SS pulses of 500
samples (10 ms at 50 kHz) at $\rho = 0.9$ give a mean $\hat\sigma/\sigma$ of
0.973 with a 0.1-99.9 % band of [0.884, 1.059]; 3 LS sweeps of 4750 samples
(95 ms at 50 kHz) give 0.997, band [0.936, 1.067]. These are the synthetic
fixture's window lengths; the real cells' are written to the noise table
(`n_pre_ls`, `n_pre_ss`) and have not yet been read.

**What it implies for code and tables.** `noise_calibration.py`:
`measure_specimen_noise`, `measure_group_noise`, `summarise_noise_table`,
`noise_lookup(protocol=)`, a stand-alone CLI (seconds per cell, no
simulation). `synth_gt_grid.draw_manifest(noise_table=, noise_rho_nominal=)`
assigns by the specimen id in the morphology path and **draws nothing from any
RNG stream to do it** -- every ground-truth column is identical with and without
the table (R11). `noise_sigma_from_results` and `--noise-from-results` are
removed. `gen_from_manifest` forwards `noise_rho_lag1` and both sampling rates;
a manifest written before this entry loads as white, nominal noise, i.e.
exactly what it generated before. `run_ih_recovery` gains step 0
(`noise_table.csv`) and `resolve_protocol`. `smoke_ih_recovery.py`: R10 closure
(inject a known $(\sigma, \rho)$, archive, read back) with its tolerance taken
from the Monte Carlo band at the window length and sweep count actually used,
because a hand-picked 6 % failed a correct estimator at $\rho = 0.9$; R11
per-cell assignment and stream invariance; R12 forwarding and precedence; R8
end to end through the measured path.

**What it does not decide [open].**
- **The slow noise is not measured.** The per-sweep DC offset and linear drift
  of a measured cell are the benchmark's nominal 0.05 mV / 0.10 mV scaled by
  $\hat\sigma(c)/0.05$ mV -- an extrapolation of the benchmark's *character*,
  not a measurement. Both are measurable from real sweeps (sweep-to-sweep
  baseline spread; pre- versus post-step baseline) and neither is
  implemented.
- **One noise law for both protocols.** The generator applies one
  $(\sigma, \rho)$ to the SS and LS sweeps of a cell; the table records both,
  and on real data they may differ (different sampling rates, filtering). With
  `NOISE_PROTOCOL=ls` the SS sweeps inherit the LS noise.
- **Spectrum beyond lag 1.** AR(1) matches the variance and the lag-1
  correlation only; line noise or a $1/f$ component is not reproduced.
- **The Allen test pulse.** The Phase-0 writer slices every sweep by the
  AllenSDK `index_range`, the same slice as the NWB loader, and the source
  comment says this clips the test pulse **[src]**; that this holds for every
  sweep is **not verified**. On the cluster, an LS $\hat\sigma$ far above the
  same cell's SS $\hat\sigma$ would be the signature of a test pulse inside
  the window.
- **Real-data consistency.** Per cell, run B's `noise_sigma_mV` should lie
  between the SS per-sweep $\hat\sigma/\sqrt{N}$ and the LS per-sweep
  $\hat\sigma$; this has not been checked (the table has not been run on the
  archive).

**Sources.** Estimator (D-012.1), generator law (D-012.2), the fitter's
`noise_sigma_mV` construction and docstring, the Phase-0 `index_range`
slicing: read from source 2026-09-23 **[src]**. Closure test and the Monte
Carlo bands **[run]**. Run B's $\rho$ census **[run, 2026-09-19]**. Averaging
invariance of $\rho$ **[reasoning]**. $\sigma/\sqrt{N}$ and the AR(1)
variance-of-the-mean factor: textbook, from memory. Run B's location
**[user-directed search of the 2026-09-19 conversation]**.

**Status:** active. Supersedes D-011 (iii); D-011's other items stand. The
gate has **not yet been run**. **Item (ii) is superseded by D-014**
(2026-09-23): the SS pulses now carry their own measured noise by default.
**The implementation of (iii) is corrected by D-014**: the protocol constants
were set once per cohort (median rate, median SS count) while each cell's
$\rho$ had been measured at its own rate; each synthetic cell is now generated
at its real twin's own acquisition. **The law (D-012.2) is replaced for every
measured cell by D-017's two-component law (2026-09-28)**, which also takes up
the open point "the slow noise is not measured". **The estimator convention of
$r_k$ in (D-012.1) is superseded by D-028 (2026-10-07)**: the lag-1 coefficient
is the textbook $\sum_j u_{k,j}u_{k,j+1}/\sum_j u_{k,j}^2$, computed by
statsmodels; $s_k$ is unchanged.

---

## D-013 -- The diameter-corrected morphologies arrive as one SWC per specimen, in the archive's own format

**Date:** 2026-09-23. **Binds:** `--swc-dir` of `Ih Fit/run_ih_fit.py` and
`swc_dir` of `load_cell_from_archive`; `SWC_DIR` of `submit_ih_fit.sh`; the
morphologies Stage 7 is run on; Stage 8's first scope. Closes the open half of
C4 (D-006, D-007) and D-008's open point.

**Decision [user], 2026-09-23.** "There will be a new swc file exactly in the
same format as the original one with the corrected diameters. For now for the
tests use the raw ones (already present)."

**Statement.** For each specimen id, the corrected morphology is one file in
the same SWC format as `<ARCHIVE_ROOT>/<GROUP>/specimen_<id>/reconstruction.swc`,
carrying the corrected diameters. It reaches the fitter through `--swc-dir`,
which resolves `<swc_dir>/<id>.swc`, `<swc_dir>/specimen_<id>.swc` or
`<swc_dir>/specimen_<id>/reconstruction.swc` and **refuses** a cell whose file
is absent rather than falling back to the raw arbour **[src]**. No per-cell
scale table is used. Until the files exist, every test -- the smoke suites,
the regression gate, and the Stage-7 recovery cohort -- runs on the raw
archive SWCs, i.e. without `--swc-dir`.

**Why.** It is the contract `swc_dir` already implements (D-008), so nothing
has to be built; the refusal keeps a missing corrected file from silently
producing a raw-arbour fit labelled as corrected.

**What it implies for code and tables.** No change to how morphologies are
read. One provenance gap closed the same day: `run_ih_fit.py` had printed
`swc_dir` to its log only, so a `phase2_results.csv` fitted on corrected
arbours would have been indistinguishable from one fitted on raw arbours.
Every row now carries `morphology_source` = `archive` or the `--swc-dir` path
(S12 asserts `archive` on a raw-SWC fit).

**What it does not decide [open].** Where the corrected files will live.
**Whether Stage 7 is re-run on them** before the corrected-scope campaign:
the gate's verdict is about the arbours it was run on, and corrected
diameters change how much membrane sits electrotonically far enough out to
sag, which is what D-011 (i) says identifiability depends on
**[reasoning]** -- the assistant's recommendation is to re-run it. $F$ stays
1.9 for L2/L3 (D-007 C4).

**Sources.** **[user]**, 2026-09-23; `swc_dir` resolution and refusal read
from source **[src]**.

**Status:** active. Closes C4 (D-006, D-007) and D-008's open point.

---

## D-014 -- Stage 7's synthetic cells are acquisition twins with per-protocol noise; AR(1) adequacy is reported, not assumed

**Date:** 2026-09-23 (later). **Binds:** `Ih Fit/noise_calibration.py`
(`noise_lookup` modes and keys, the `acf_*` / `ar1_*` columns);
`synth_gt_grid.py` (manifest columns `noise_ss_sigma_mV`, `noise_ss_rho_lag1`,
`noise_ss_source`, `acq_ss_n_repeats`, `acq_fs_ss_Hz`, `acq_fs_ls_Hz`);
`gen_from_manifest.py` (`row_to_ss_noise_kwargs`, `row_to_acquisition`,
`generate_group`); `synthetic_ground_truth.generate_synthetic_cell(ss_noise=)`;
`run_ih_recovery.py` (`--noise-protocol`, `apply_cli_acquisition`, the SS
loss-weight flags); `submit_ih_recovery.sh`. **Supersedes D-012 (ii)** and
**corrects D-012's implementation of (iii)**.

**Decision.** (i) **[user, 2026-09-23: "Use also the short square with
exponential weights as well"; assistant's reading]** The SS pulses of a
synthetic cell carry the SS-measured noise of its real twin, and its Long
Square sweeps the LS-measured noise (`--noise-protocol per_protocol`, the new
default); `ls` / `ss` keep D-012's one-law form. The request was read two
ways and both were acted on: as a check that the fit uses the SS pulses with
their exponential time-weight -- it does, in every arm and every recovery fit,
now asserted at run time -- and as an instruction to give the SS pulses their
own noise. The exponential weight belongs to the LOSS and is never put into
the noise: the generator injects stationary noise, as the recording is, and
the fitter then weights it exactly as it weights real data. (ii)
**[assistant, correcting its own D-012 code]** Each synthetic cell is
generated at its real twin's own sampling rates and SS pulse count; the
cohort-level protocol is only the fallback. (iii) **[assistant]** The noise
table reports whether AR(1) reproduces the measured autocorrelation beyond lag
1; nothing downstream acts on it.

**Statement.** For a real cell $c$ with synthetic twin $c'$, and
$\hat\sigma_p(c)$, $\hat\rho_p(c)$ the per-sweep estimates of (D-012.1) for
protocol $p \in \{\mathrm{LS}, \mathrm{SS}\}$:

- the Long Square sweeps of $c'$ carry the AR(1) law (D-012.2) with
  $(\sigma, \rho) = (\hat\sigma_{\rm LS}(c), \hat\rho_{\rm LS}(c))$; its SS
  pulses carry $(\hat\sigma_{\rm SS}(c), \hat\rho_{\rm SS}(c))$ when both are
  finite, and the LS pair otherwise (`noise_ss_source` = `measured` /
  `inherits`); the slow terms and the seed are the cell's, shared;
- $c'$ is sampled at the rates $f_{s,{\rm LS}}(c)$, $f_{s,{\rm SS}}(c)$ (Hz)
  at which $c$ was recorded, with $N_{\rm SS}(c)$ pulses per polarity; a value
  given on the command line overrides this for every cell (precedence command
  line > measured > default, per cell), and a forced rate that differs from
  a cell's own is logged with the number of cells affected;
- for each protocol and each lag $\tau \in \{0.1, 1\}$ ms, with
  $L = \mathrm{round}(\tau f_s)$ samples, the table records
  $\operatorname{median}_k \hat\rho_k(L)$ beside
  $\operatorname{median}_k \hat\rho_k(1)^L$, where $\hat\rho_k(L)$ is the
  lag-$L$ analogue of (D-012.1) on sweep $k$ (at $L = 1$ it is the fitter's
  own statistic, to machine precision, asserted by R10). A stationary AR(1)
  with lag-1 coefficient $\rho$ has autocorrelation $\rho^L$ at lag $L$
  (textbook, from memory), so the two agree in expectation on AR(1) noise
  and part when the noise has more than one correlation time.

**Why.** (i) **Per protocol.** What a pre-stimulus window measures depends on
its length: demeaning a window of duration $T$ removes most of the variance
of fluctuations much slower than $T$ **[reasoning]**. The SS residual is
baseline-subtracted over $(-10, 0)$ ms and weighted by
$w(t) = e^{-(t - 0.5\,{\rm ms})/5\,{\rm ms}}$ over $0.5$--$100$ ms **[src]**, so
it is judged almost entirely on its first $\sim$10--15 ms ($w$ is $e^{-2}$ at
10.5 ms and $e^{-4}$ at 20.5 ms); the LS residual spans the whole step, with
the baseline taken over the pre-step window **[src]**. Each protocol's own
pre-window therefore measures the noise at roughly the bandwidth its own
residual sees; one law for both would hand the SS pulses the long window's
slow variance, or the LS sweeps the short window's lack of it
**[reasoning]**. (ii) **The transplant.** $\rho$ is a correlation between
successive samples. The Allen recordings were "digitized at 200 KHz (before
2016) or 50 KHz (2016 and later)" after a 10 kHz Bessel filter
**[KB, Allen ephys whitepaper]**. For an Ornstein-Uhlenbeck process sampled
at interval $\Delta t$, $\rho = e^{-\Delta t / \tau_c}$ (textbook, from
memory), so one process gives $\rho_{200\,{\rm kHz}} = \rho_{50\,{\rm kHz}}^{1/4}$:
a $\rho$ carried to a different interval changes the correlation time by the
ratio of the intervals. D-012's code measured $\rho$ per cell but generated
every cell of a cohort at ONE rate, the cohort median; R8's own run log showed
two cells at 50 and 100 kHz producing a cohort rate of 75 kHz, at which
neither had been recorded **[run]**. Generating each twin at its own rate
transfers $(\sigma, \rho)$ exactly, with no model of how $\rho$ scales.
(iii) **AR(1) adequacy.** An AR(1) matched at lags 0 and 1 imposes a single
exponential correlation time. The neuronal-noise literature describes more
structure than that, qualitatively: somatic subthreshold voltage noise in rat
L4-5 pyramidal neurones in slices has a low-frequency part shaped by the Na+
conductance and a higher-frequency part dominated by synaptic activity
(Jacobson et al. 2005, **PubMed, abstract only**,
[DOI](https://doi.org/10.1113/jphysiol.2004.080903)); and the whole-cell
CURRENT noise of cultured rat hippocampal neurons, recorded with synaptic
blockers, has a spectrum with generalized $1/f$ behaviour, which becomes
dominated by a single Lorentzian only after TTX removes the Na+ channel
contribution (Diba et al. 2004, **PubMed, abstract only**,
[DOI](https://doi.org/10.1523/JNEUROSCI.1721-04.2004)). A single Lorentzian
is what AR(1) implies. Neither full text was retrievable
through the connector, so no number from either is used, and both are rat
preparations, not human slice: the diagnostic exists so the question is
answered by our own data, not transplanted. If slow correlated power exists
and the generator omits it, synthetic data are more informative than real
data, which is the optimistic direction for the gate **[reasoning]**.

**What it implies for code and tables.** The manifest columns above, all
back-filled so a pre-D-014 manifest generates one white nominal law at the
cohort protocol, as before. `noise_lookup` is CSV-safe: an empty `error` cell
reads back as NaN, which is truthy, and the previous test on the raw value
would have silently dropped every cell of a reloaded table
(`synth_gt_grid.py --noise-table`). `run_ih_recovery` forwards
`--ss-time-weight exp --ss-tau-w-ms 5.0 --ss-window-ms 0.5,100.0` explicitly.
Smoke: S12 and R8 assert the exponential SS weight on the orchestrator's own
loss builds; R10 the diagnostic's silence on AR(1) within its Monte Carlo band,
its lag-1 identity with the fitter, and the generator's legacy path bit for
bit; R11 the manifest columns and the CSV round trip (it fails with the old
predicate); R12 the mappings and per-cell precedence; R13 that the diagnostic
can fail (a hidden 5 ms component: 0.382 against an AR(1)-implied 0.004); R8
the closure real -> measure -> twin -> measure, at 50 and 100 kHz.

**What it does not decide [open].** Whether AR(1) is adequate for the real
cells: read `acf_*` against `ar1_*` in the L3_exc table. If the 1 ms column
sits well above its AR(1) value, the generator needs a second timescale (AR(1)
plus a slow OU, an AR(2), or a spectrum-matched generator) -- a new decision.
The slow terms (per-sweep offset, drift) are still the benchmark's nominal
values scaled by $\hat\sigma_{\rm LS}(c)/0.05$ mV. The Long Square amplitude
set is not twinned: the cohort uses one amplitude list, real cells differ.
The twin uses the hyperpolarising SS pulse count for both polarities.
**bioRxiv:** the connector has no keyword search; a category browse
(biophysics, last 30 days, 30 preprints) returned nothing on recording-noise
modelling.

**Sources.** Loss windows and weights, estimator, generator law, lookup code:
read from source 2026-09-23 **[src]**. Sampling rates and filter: Allen ephys
technical white paper **[KB]**. Jacobson et al. 2005 and Diba et al. 2004:
**PubMed, abstract only**, qualitative use only; full text not accessible
through the connector. R8 log, Monte Carlo bands, closure **[run]**. AR(1)
and OU autocorrelation: textbook, from memory. The decision (i) **[user]**,
as read by the assistant; (ii), (iii) **[assistant]**.

**Status:** active. Supersedes D-012 (ii); corrects the per-cohort protocol
constants of D-012's implementation; D-012 (i), (iii), (iv) stand. The gate
has **not yet been run**. **(i) and (ii) confirmed by the user on
2026-09-23 17:07 -- see D-015**, which records them as the user's. **Its open
point was answered on the cluster the same day: AR(1) does not reproduce the
LS autocorrelation of L3_exc (0.566 at 1 ms against 0.000); the cause is
being measured -- see D-016.** **Answered by D-017 (2026-09-28): a second,
slow timescale is added per cell; (i)'s per-protocol AR(1) pair stands until
D-017's open point (b) is settled.** **The lag-$L$ estimator of its Statement
(the analogue of (D-012.1)) is superseded by D-028 (2026-10-07).**

---

## D-015 -- The injected noise never carries the loss's time-weight and its fast part is stationary; every synthetic cell is sampled at its own real cell's rate

**Date:** 2026-09-23 (later, 3), 17:07. **Binds:** the generator's noise law
(`synthetic_ground_truth.add_recording_noise`, `NoiseConfig`); the SS
time-weight of the loss (`passive_long_step_training.exp_time_weight`,
`--ss-time-weight`, `--ss-tau-w-ms`); the per-cell acquisition of every
Stage-7 cohort (manifest `acq_*`, `gen_from_manifest.row_to_acquisition`);
the noise object `noise_calibration.py` measures. **Confirms D-014 (i) and
(ii)**, which were recorded as the assistant's reading and decision.

**Decision [user].** (i) The exponential time-weight belongs to the LOSS
alone: no component of the noise injected into a synthetic sweep carries it,
and the fast noise is stationary, as the recording's is taken to be at rest
(the per-sweep slow terms, a constant offset and a linear tilt, are not
weighted either) -- "about the decoupling of noise and the exponential
weighting is what I was thinking about". (ii) Every
synthetic cell is generated at its own real cell's sampling rate -- "use the
cells' own sampling rate". (iii) The noise object is the pre-stimulus
fluctuation of a SINGLE sweep, i.e. the total baseline fluctuation (recording
chain plus membrane), not an instrument-only noise -- "OK for both".

**Statement.** For an SS bundle the loss is the weighted RMSD
$\bigl(\sum_s w(t_s)\, d_s^2 / \sum_s w(t_s)\bigr)^{1/2}$ over
$t_s \in [0.5, 100]$ ms, with $d_s$ (mV) the model-minus-data difference after
each trace's own pre-window mean is removed and
$w(t) = e^{-(t - 0.5\,{\rm ms})/\tau_w}$, $\tau_w = 5$ ms **[src]**. The
module's own reading of $w$ as a heteroscedastic likelihood with noise
variance $\propto 1/w(t)$ **[src]** describes the ESTIMATOR, not the data.
The fast noise $e_j$ injected into every synthetic sweep is second-order
stationary for all $j$: $\mathbb{E}[e_j] = 0$ and
$\operatorname{Cov}(e_j, e_{j+L}) = \sigma^2 \rho^{L}$ for every lag $L$, with
$(\sigma, \rho)$ the per-protocol measurement of D-014, at the real cell's own
sampling interval $\Delta t = 1/f_s$. The fitter then applies $w$ to synthetic
and real sweeps identically.

**Why.** (i) The gate asks whether the fitter recovers the truth from data
that look like the recording. Shaping the synthetic noise by $1/w(t)$ would
build the loss's own weighting assumption into the data, so the fitter would
be tested on data made to satisfy its estimator -- circular **[reasoning]**.
(ii) $\rho$ is a correlation between successive samples, so it transfers only
at the interval it was measured at (D-014, Why (ii)). (iii) One pre-stimulus
window cannot separate instrument noise from membrane noise, and the fitter
sees their sum **[reasoning]**.

**What it implies for code and tables.** Nothing changes: all three are
already the defaults (D-014). `add_recording_noise` injects stationary AR(1)
plus the per-sweep slow terms; the weight is applied only inside
`bundle_rmsd`; each synthetic cell is generated at its twin's `acq_fs_*`,
and forcing a rate is an explicit opt-in (`SS_FS_HZ`, `LS_FS_HZ`), logged per
clash. S12 and R8 assert the weight in the loss; R8 asserts the twins'
rates. **Pending, for the next delivery:** a smoke check that asserts the
injected noise is stationary ACROSS the sweep (after the pulse as before it),
so no later change can move the weighting into the noise.

**What it does not decide [open].** Whether the REAL recording noise is
stationary across a response. The pre-stimulus window measures it at rest
only, and membrane noise can depend on voltage: in rat L4-5 pyramidal
neurones the somatic voltage noise grows with depolarisation (Jacobson et al.
2005, **PubMed, abstract only**,
[DOI](https://doi.org/10.1113/jphysiol.2004.080903)). Our steps are
hyperpolarising and the dependence there is not measured here. Whether AR(1)
is adequate (D-014, open).

**Sources.** **[user]**, 2026-09-23 17:07. The weighted RMSD and the
module's likelihood note, read from source **[src]**. Jacobson et al. 2005,
abstract only, qualitative. The circularity argument **[reasoning]**.

**Status:** active. Confirms D-014 (i) and (ii); D-014's other items and its
open points stand.

---

## D-016 -- The depolarising-sweep spike screen follows the Allen white paper's action-potential rule; the Long Square pre-window is examined before its noise is modelled

**Date:** 2026-09-23 (later, 4). **Binds:** `Ih Fit/passive_fitting_hpc_fixed.py`
(`sweep_spike_reason`, `sweep_has_spike`, `spike_events`,
`filtered_dvdt_mV_per_ms`, the depolarising loop of `load_cell_from_archive`,
`CellData.ls_dep_screen`); `passive_long_step_training.assign_ls_roles` (its
warning); `noise_calibration.py` (the LS pre-window structure columns); the new
`check_dep_sweeps.py`; S13, R8, R14. **Corrects the assistant's implementation
of D-006 Q7** ("spiking sweeps are excluded by waveform"); the intent of D-006
Q7 is unchanged. **Answers nothing in D-014's open point yet**: it adds the
measurement that point needs.

**Decision.** (i) **[assistant, correcting its own Stage-2 code]** A
depolarising Long Square sweep is "not subthreshold" when a sample is
non-finite, when it contains an action-potential event by the rule of
(D-016.1)--(D-016.3), or -- failing both -- when its voltage exceeds $-20$ mV
anywhere (a catch-all for a depolarisation too slow to be an event). The
previous rule, max $\lvert v_{j+1} - v_j \rvert / \Delta t > 10$ mV/ms on the
raw samples, is withdrawn. (ii) **[assistant]** The loader records, per cell,
how many depolarising sweeps it saw, how many were above the amplitude cap,
how many it rejected for each reason, and how many it kept; the role
assignment reports that cause instead of asking whether the loader was called
with `load_depolarising_ls=True`. (iii) **[assistant]** Before any second
noise timescale is added to the generator (D-014's open point), the noise
table reports where the Long Square pre-window variance sits. The generator is
not changed by this entry.

**Statement.** Let $v_j$ (mV), $j = 0, \dots, n-1$, be one depolarising sweep
as the archive stores it (LJP-corrected: recorded voltage plus
`ljp_correction_mV`, +14 mV for the Allen cells **[src]**), sampled at $f_s$
(Hz), $\Delta t = 1/f_s$, and $\tilde v$ the trace after a 4-pole Bessel
low-pass with cutoff $f_c = 10$ kHz applied forward and backward. The filtered
derivative is

$$ \dot v_j = (\tilde v_{j+1} - \tilde v_j)/\Delta t \quad (\text{mV/ms}), \qquad j = 0, \dots, n-2, \tag{D-016.1} $$

taken on $v$ unfiltered where $f_s \le 2 f_c$. A putative event is an index
$c$ with $\dot v_c < 20 \le \dot v_{c+1}$ mV/ms, a later one kept only if
$\dot v$ fell below 0 since the previous one; its peak $p$ is the index of the
largest $v$ from $c$ to the next event; two consecutive events with no
negative $\dot v$ between the first peak and the second crossing are merged
(first crossing, second peak); an event is kept if

$$ v_p \ge -30 \ \text{mV} \quad \text{and} \quad v_p - v_c \ge 2 \ \text{mV}, \tag{D-016.2} $$

and it is an action potential if (D-016.2) also holds with $p$ replaced by

$$ p_w = \operatorname*{arg\,max}_{c \,\le\, i \,\le\, \min(c + W,\ p)} v_i, \qquad W = \operatorname{round}(5\ \text{ms} \cdot f_s). \tag{D-016.3} $$

For the Long Square pre-window $x_k$ of single sweep $k$ of a cell
($k = 1, \dots, K$; the window the fitter reads, from the start of the stored
sweep to 5 ms before onset), the table records the medians over $k$ of: the SD
over the first 50 ms and over the last 500 ms (each at most half the window;
the lengths used are recorded); the lag-1 correlation $\hat\rho_{\rm late}$ of
the last segment and its autocorrelation at 0.1, 1, 10 and 100 ms beside the
AR(1) value $\hat\rho_{\rm late}^{L}$ ($L$ the lag in samples); the share of
$\operatorname{var}(x_k)$ that a least-squares line explains; and, once per
cell, the share that repeats across sweeps,

$$ s_{\rm lock} = \frac{\operatorname{var}_j\bigl(\bar x_j\bigr) - \bar v / K}{\bar v\,(1 - 1/K)}, \qquad \bar x_j = \frac{1}{K}\sum_k x_{k,j}, \quad \bar v = \frac{1}{K}\sum_k \operatorname{var}_j(x_{k,j}), \tag{D-016.4} $$

with each $x_k$ demeaned and cut to the shortest window, aligned at its first
sample, and $\operatorname{var}_j$ the variance over the sample index $j$.
Under $x_k = s + e_k$, with $s$ common to all sweeps and the $e_k$ independent
with a common variance, $\mathbb{E}[\operatorname{var}_j \bar x_j] = v_s +
v_e/K$ and $\mathbb{E}[\bar v] = v_s + v_e$, so (D-016.4) estimates
$v_s / (v_s + v_e)$ [reasoning], with $v_s$, $v_e$ the within-window
(demeaned) variances: 0 when nothing repeats, 1 when the sweeps are identical.
A drift with the same sign in every sweep also repeats, so $s_{\rm lock}$ is
read with the trend share and the early/late ratio, never alone.

**Why.** (i) **The old rule fired on noise.** For a stationary window with SD
$\sigma$ and lag-1 correlation $\rho$, the raw differences have SD
$\sigma\sqrt{2(1-\rho)}$ (textbook, from memory); at the L3_exc medians
($\sigma = 0.0594$ mV, $\rho = 0.668$, 50 kHz, **[run]** on the user's cluster,
2026-09-23) that is 2.42 mV/ms, and its maximum over one synthetic 1.3 s sweep
came out at 10.2 mV/ms **[run]**. Every synthetic twin in R8 lost its +20/+50
pA steps. In S13 the old rule rejects 7/8 noise-only subthreshold sweeps at
the L3_exc medians and 8/8 at 100 and 200 kHz; on its own, any bridge-balance
step of 0.2 mV or more at 50 kHz trips it (0.2 mV / 0.02 ms = 10 mV/ms,
[arithmetic]). Under the new rule no noise-only sweep is rejected, and the
largest filtered derivative of the noise is 4.7 mV/ms against the 20 mV/ms
threshold **[run]**. A bridge step larger than 1.41 mV at 50 kHz (1.45 mV at
200 kHz) does cross the filtered threshold **[run]**, and
(D-016.3) is what keeps it out: S13 puts a 3 mV step on a sweep that plateaus
at $-28$ mV, which (D-016.2) alone would call an action potential and
(D-016.3) does not. Whether real cells lost their depolarising steps is **not
yet known**: `check_dep_sweeps.py` answers it.

(ii) **Where the rule comes from, and where it departs.** "The voltage trace
was smoothed with a digital four-pole Bessel filter with a cutoff frequency of
10 kHz ... Putative action potential events were identified where the
derivative trace exceed 20 mV/ms and where the time derivative returned below
0 mV/ms between consecutive events ... If the time between the threshold and
the peak exceeded 2 ms, the putative action potential was rejected ... if the
difference between the threshold voltage and the peak was less than 2 mV, or
if the absolute peak was less than -30 mV, the putative action potential was
also rejected" **[KB, Allen Cell Types technical white paper, "Action
Potential Identification"]**. (D-016.1)--(D-016.2) are IPFX 2.1.2's
`calculate_dvdt`, `detect_putative_spikes`, `find_peak_indexes` and
`filter_putative_spikes` **[src]**, and agree with them event for event on
3000 random traces at 25--200 kHz (4336 events, 0 disagreements, **[run]**).
Two departures, both deliberate: (a) the time test (D-016.3) replaces IPFX's
`check_thresholds_and_peaks`, which re-estimates the threshold at 5 % of the
mean upstroke and re-searches the peak within twice the interval; its default
interval is 5 ms where the white paper says 2 ms -- **the two sources
disagree** -- and 5 ms is used; (b) the voltages compared with $-30$ and
$-20$ mV are the archive's LJP-corrected ones, while the Allen applied $-30$ mV
to the recorded voltage, so this screen's level test is 14 mV stricter than
the Allen's own spike labels: a fast event that reaches $-30$ mV of corrected
membrane potential does not belong in a passive + I_h validation set either
[reasoning]. IPFX is not imported: it pins `pynwb==3.1.2` and brings
`argschema`, `pg8000` and others into the cluster environment for four
numpy/scipy functions **[src]**. The Allen database's own spike counts would
be the natural cross-check; for specimen 508282493 its EphysSweep table,
read through a summarising web fetch, listed depolarising Long Square sweeps
under 100 pA with no spikes (**data repository, summarising fetch,
unverified**: two fetches agreed that such sweeps exist but not on which, so
no count is quoted).

(iii) **The LS noise is not AR(1), and why is not yet known.** The cluster
table: LS autocorrelation 0.613 at 0.1 ms and 0.566 at 1 ms against AR(1)'s
0.134 and 0.000; SS 0.155 and 0.036 against 0.002 and 0.000 **[run]**. Three
readings fit those numbers, and they call for different remedies: a transient
at the start of the stored sweep is excluded from the window, never
generated; stationary slow noise is generated as a second component; drift is
the generator's existing tilt term, calibrated. Every Allen data sweep begins
with a test pulse **[KB]**, and the archive stores each sweep from the start of
the NWB "Experiment" epoch, which AllenSDK documents as "the end of the test
pulse" (`get_sweep`'s `index_range`, **[src, AllenSDK 2.16.2]**, sliced so by
the Phase-0 writer **[src]**). How long after the pulse that is, is not
documented there, and it matters: IPFX's own experiment epoch starts 0.5 s
before the stimulus (`get_experiment_epoch`: stimulus start minus
`PRESTIM_STABILITY_EPOCH` = 0.5 s, **[src]**). If the stored LS sweep starts
there, the pre-window is about 0.5 s long, a test-pulse tail decaying in tens
of ms cannot reach it, and it covers the same samples as the Allen QC's
`slow_noise_rms_mv` (the RMS over the 500 ms before the stimulus,
`get_first_stability_epoch`, **[src]**); if it starts right after the pulse,
the window is about 1 s and a tail can sit at its start. With a 1 s window, a
0.5 mV, 20 ms tail at the start plus fast AR(1) noise of 0.040 mV,
$\rho = 0.25$, reproduces the table's four LS numbers ($\sigma$ 0.063 mV,
$\rho$ 0.70, 0.60 / 0.57 against 0.059, 0.67, 0.61 / 0.57); with a 0.5 s
window the same tail gives 0.079, 0.81, 0.74 / 0.71, which does not
**[run]**. `ls_pre_ms` settles the window length. On the other columns: with
no repeating component, $K$ = 5--6 sweeps of 500 ms carrying a slow OU of
30--300 ms give $s_{\rm lock}$ an SD of 0.03--0.06 and a 99th percentile of
0.09--0.20 over 300 draws per case **[run]**, so one cell's locked share is
informative above about 0.2 and a weaker transient shows only across the
cohort.

**What it implies for code and tables.** `DEFAULT_SPIKE_DVDT_MV_PER_MS` = 20
(was 10), new `DEFAULT_SPIKE_FILTER_KHZ` = 10, `DEFAULT_SPIKE_MIN_PEAK_MV` =
$-30$, `DEFAULT_SPIKE_MIN_HEIGHT_MV` = 2, `DEFAULT_SPIKE_MAX_INTERVAL_MS` = 5;
`sweep_has_spike_legacy` keeps the old rule for `check_dep_sweeps.py` only.
Noise table: `ls_pre_ms`, `ls_early_ms`, `ls_late_ms` (the segment lengths
used), `sigma_ls_early_mV`, `sigma_ls_late_mV`, `rho_ls_late`,
`ls_trend_share`, `ls_locked_share`, `acf_ls_late_*` / `ar1_ls_late_*`;
nothing downstream reads them, and a table written before them still
summarises (NaN). Smoke: S13 (noise at 50--200 kHz and bridge steps at rest
and on a $-28$ mV plateau are not spikes; the height rule and the $\dot v < 0$
rule decide noise-free cases as specified; action potentials peaking at +25
and $-25$ mV are, at 50 and 20 kHz, and are reported as action potentials,
not as the catch-all; the loader keeps +20/+50 pA and reports +200 pA as
capped or as one action potential; the role warning distinguishes "not
asked", "none in the archive" and "screened out"; `check_dep_sweeps` keeps
4/4 where the old rule keeps 0/4; the old rule, the missing time test and the
old reason order each make S13 FAIL); R8 asserts that all 8 role assignments
see the twins' +20/+50 pA steps; R14 separates a locked transient, a slow OU
and a drift, with margins from 40 seeds.

**What it does not decide [open].** (a) The generator's slow-noise model
(D-014's open point): waits for the new columns on L3_exc. (b) Whether the
fitter's own LS noise window and LS baseline window should start later than
the stored sweep, which a start-of-sweep transient would argue for. (c) The
twin's Long Square timing: the generator puts the step 100 ms into a 1.3 s
sweep, the Allen protocol 1.02 s into the recorded sweep (EphysSweep
`stimulus_start_time`, same unverified fetch), so a twin's LS pre-window
(95 ms) is shorter than its real cell's (`ls_pre_ms`). (d) Whether noise
differs between the sweeps of one cell, while the generator draws every sweep
of a cell from one law. [settled 2026-10-21 by D-073: per-sweep calibration] (e) Whether to place the level test at the Allen's
recorded-voltage $-30$ mV instead (departure (b) above): the user's call.

**Sources.** White paper: project KB, full text. IPFX 2.1.2 and AllenSDK
2.16.2: wheels from PyPI, read from source. Allen Brain Map API, EphysSweep
table, specimen 508282493: through a summarising web fetch, unverified, used
for no number. Simulations, smoke results, the IPFX agreement test and the
$s_{\rm lock}$ Monte Carlo **[run]**, 2026-09-23. An independent review of the
first version of this change (same day) found the reason order, the missing
merge and time tests, the LJP convention and the fetch provenance; all are
corrected above. PubMed searched 2026-09-23 for voltage-noise spectra of
pyramidal neurons in vitro and for coloured or autoregressive noise models in
electrophysiological parameter estimation: nothing that bears on this (the
only noise-spectrum sources remain Jacobson 2005 and Diba 2004 of D-014,
abstract only). bioRxiv: neuroscience category, last 30 days, 30 preprints:
nothing on recording noise or spike detection.

**Status:** active. Corrects the implementation of D-006 Q7 (intent
unchanged). D-014's open point stands, with its measurement now in place.
**Open points (a) and (c) are settled by D-017 (2026-09-28)**; Why (iii)'s
framing of the pre-window as about 0.5 s or about 1 s is corrected there: it
is 265 ms in every L3_exc cell, and (b) has no case on L3_exc.

---

## D-017 -- Each synthetic twin carries its own real cell's two-component noise law, fitted to that cell's window statistics, and its real Long Square timing

**Date:** 2026-09-28. **Binds:** the generator's noise law
(`synthetic_ground_truth.add_recording_noise`, `NoiseConfig`) and the twin's
Long Square timing (`ProtocolConfig.ls_onset_ms`, `gen_from_manifest`); the
noise columns of a Stage-7 manifest (`synth_gt_grid.draw_manifest`);
`noise_calibration.py`, which gains a per-cell law fit; `smoke_ih_recovery.py`;
every Stage-7 verdict. **Answers D-014's open point** (AR(1) is not adequate
on L3_exc; a second timescale is added) and **D-016's open points (a) and
(c)**; takes up D-012's open point "the slow noise is not measured".
**Replaces the AR(1) law (D-012.2) as the generator's law for every measured
cell.** Corrects D-016 Why (iii) and D-011 Why (iii) -- see "Corrections".

**Decision [user], 2026-09-28 15:39.** (i) "Per-cell two-component law ...
PERFECT": the noise injected into each synthetic twin is a fast AR(1)
component plus a slow Ornstein-Uhlenbeck (OU) component, with the generator's
existing per-sweep terms (a DC offset and a linear tilt) carrying drift; each
twin's parameters are calibrated on ITS OWN real cell by matching the window
statistics `noise_calibration.py` measures on that cell; and the twin's Long
Square pre-window is its real cell's (265 ms on L3_exc), not the generator's
95 ms. (ii) "No use the two compartment [component] law specifically for each
cell": there is no cohort-level law. The cohort fit of the same day
($\sigma_f$ = 0.0381 mV, $\rho_f$ = 0.176, $\sigma_s$ = 0.0506 mV,
$\tau_s$ = 36.3 ms) is a diagnostic of the cohort, never a generator input.

**Statement.** Let $c$ be a real archive cell with synthetic twin $c'$, and
$\Delta t = 1/f_s$ (s) the sampling interval of a protocol of $c$ (the twin
is sampled at the same interval, D-014, D-015). For one sweep of $c'$ with
samples at $t_j = j\,\Delta t$ (s), $j = 0, \dots, n-1$, and $\bar t$ (s) the
sweep's mid-time, the injected noise (mV) is

$$ e_j = f_j + s_j + a + b\,(t_j - \bar t), \tag{D-017.1} $$

where $f$ is the stationary AR(1) of (D-012.2) with parameters
$(\sigma_f(c), \rho_f(c))$ at $\Delta t$; $s$ is a stationary OU process with
SD $\sigma_s(c)$ (mV) and correlation time $\tau_s(c)$ (s; quoted in ms),
sampled at $\Delta t$ -- equivalently an AR(1) with coefficient
$\rho_s(c) = e^{-\Delta t/\tau_s(c)}$ (textbook, from memory) -- independent
of $f$; and $a$ (mV) and $b$ (mV/s) are drawn once per sweep, independently
across sweeps, from laws calibrated on $c$ (open point (c)). For each fixed
$c$ and every lag $k \in \mathbb{N}$ (samples), the autocovariance of $f + s$
is

$$ \gamma_c(k) = \sigma_f(c)^2\,\rho_f(c)^{k} + \sigma_s(c)^2\,e^{-k\,\Delta t/\tau_s(c)} \qquad (\mathrm{mV}^2). \tag{D-017.2} $$

For any law (D-017.2), the expected value of each statistic
`noise_calibration.py` computes on a window of $n$ samples -- the SD of the
demeaned window with ddof = 1 of (D-012.1), and the lag-$L$ autocorrelation
in the fitter's convention -- is a closed-form sum over
$\gamma_c(0), \dots, \gamma_c(n-1)$, taken as $\sqrt{\mathbb{E}[s^2]}$ for
the SD and as a ratio of expectations for the autocorrelation
(approximations that become exact as $n \to \infty$) **[reasoning, verified
by running it]**. $\theta_{\rm noise}(c) = (\sigma_f, \rho_f, \sigma_s,
\tau_s)(c)$ is the minimiser of the scaled squared distance between those
predictions and $c$'s measured per-sweep medians. The twin's Long Square
onset is placed so that the fitter reads the same pre-window as on $c$: at
`ls_pre_ms`$(c)$ + 5 ms from the start of the stored sweep, the fitter's LS
window being $[t_0, t_{\rm on} - 5\,{\rm ms})$ **[src]** -- 270 ms on every
L3_exc cell.

**Why.** (i) **Two timescales, not a transient.** On L3_exc the last
132.5 ms of the LS pre-window keep an autocorrelation of 0.448 at 1 ms and
0.244 at 10 ms, against 0.000 for AR(1); the first 50 ms are quieter than the
last 132.5 ms (SD 0.0465 against 0.0551 mV) and the locked share is 0.02
(cohort medians), so the correlated variance is neither a start-of-sweep
transient nor sweep-locked **[run, cluster 2026-09-28]**. Fitted to the 14
cohort-median statistics (SS 10 ms window: SD and autocorrelation at 0.02,
0.1, 1 ms; LS 265 ms: the same four; LS first 50 ms: SD; LS last 132.5 ms:
SD and autocorrelation at 0.02, 0.1, 1, 10 ms), (D-017.2) reaches a cost of
1.88 against 215 for the best single AR(1), from six starting $\tau_s$
between 2 and 100 ms that all converge to 36.3 ms **[run]**. The SD of a
demeaned window grows with its length as a slow component predicts: under
the fitted law 9 %, 34 %, 60 % and 76 % of $\sigma_s^2$ survives demeaning in
the 10, 50, 132.5 and 265 ms windows, giving SDs of 0.0409, 0.0481, 0.0546
and 0.0584 mV against 0.0405, 0.0465, 0.0551 and 0.0594 mV measured
**[run]**. (ii) **Per cell.** The cells differ exactly at the slow
timescales: late-segment autocorrelation at 1 ms from 0.23 to 0.88,
full-window LS SD from 0.049 to 0.228 mV **[run, cluster 2026-09-23 and
2026-09-28]**. A cohort law would give every twin the median cell's noise,
and the gate would test no cell under its own. (iii) **The timing.** The
fitter's LS baseline is the pre-window mean; how well that mean stands for
the level during the step depends on the window's length relative to
$\tau_s(c)$, so a twin read through a 95 ms window is tested under a
different baseline error from its cell's **[reasoning]**.

**What it implies for code and tables** (implementation not started; it
waits for (a)-(c) below). A per-cell law fit in `noise_calibration.py`
beside the existing table. Manifest columns for the law under names that
cannot be confused with the Short Square columns of D-014 -- `noise_ss_*`
already means Short Square -- e.g. `noise_fast_sigma_mV`,
`noise_fast_rho_lag1`, `noise_slow_sigma_mV`, `noise_slow_tau_ms` and the
drift columns. `add_recording_noise` gains the OU term, off for manifests
written before this entry so they generate what they always generated; the
twin's `ls_onset_ms` comes from `ls_pre_ms` + 5 ms. The per-cell
two-component law is the DEFAULT for every cell found in the noise table;
single AR(1) survives only as the back-fill of old manifests and as a
labelled comparison. Smoke: closure (inject a known law, measure it with
`noise_calibration.py`, fit it, recover it within the Monte Carlo band at the
real window lengths and sweep counts), an assertion of the default, and
D-015's queued check that the injected noise is stationary across the sweep.
The generator's docstring attributes the tilt to "electrode/seal drift"
**[src]**; the Allen cell-level QC caps electrode drift at 1 mV per 10 min of
recording **[KB]**, so a within-sweep creep of tenths of a mV per second is
not the electrode in a cell that passed it **[reasoning]**; the docstring is
corrected with the implementation.

**What it does not decide [open].**
- (a) **The two drifting cells.** 526785799 and 571430815 have LS trend
  shares (share of the pre-window variance a least-squares line explains) of
  0.58 and 0.64, medians over their 5 sweeps, against a cohort median of
  0.14 **[run, cluster 2026-09-28]**; under the cohort law a single 265 ms
  window exceeds 0.58 with probability 0.003 (median 0.06, 99th percentile
  0.50, 3000 draws) **[run]**. Whether they stay in the cohort with their
  drift calibrated or are excluded: asked 2026-09-28, pending. The
  assistant's recommendation: keep both, flagged in the gate report -- the
  gate exists to find out whether the fitter survives each cell's real noise.
  571430815 is also the noisiest cell at every slow timescale, so its excess
  may be its own slow noise rather than a creep; its own law decides.
- (b) **One law for both protocols.** Whether a cell's SS pulses and LS
  sweeps are drawn from the same law (D-017.2), each read at its own window
  -- the form of the cohort fit, and the assistant's reading of "matching the
  same window statistics" -- or the SS pulses keep D-014's own AR(1). D-014
  (i) stands until this is answered.
- (c) **Where drift is calibrated.** A 265 ms window cannot tell a linear
  creep from a fluctuation slower than itself; the two differ over the 1 s
  step, where one continues and the other wanders back **[reasoning]**.
  Candidates: the LS pre-window slopes, as the excess of their variance over
  what the cell's own law (D-017.2) produces; the SS sweeps, whose 20 pulses
  are 200 ms apart **[KB]** and which the archive stores in order with their
  sweep number and a (-10, +200) ms window each **[src]**, so that the 20
  pre-pulse baselines sample the slow baseline every 200 ms over about 4 s
  [corrected 2026-10-20: in the archive's cells 37 pulses per sweep, 200.5 ms
  onset to onset, about 7.2 s; data, changelog 2026-10-20 (later, 5)];
  the LS pre- versus post-step baseline, contaminated by the rebound. The
  assistant's recommendation: the SS series. Whether every stored sweep
  passed the Allen sweep-level QC (end-of-sweep voltage within 1 mV of the
  pre-stimulus voltage **[KB]**) is **not verified**: neither the archive
  writer nor AllenSDK's `get_ephys_sweeps` filters on it **[src]**.
- (d) The per-cell fit's statistic set, weights and estimator (analytic
  expectations as in the cohort fit, or simulated moments through
  `noise_calibration.py`'s own code).
- (e) **$\tau_s$ is loosely bounded above** by windows no longer than
  265 ms: profiling the cohort fit (the other three parameters refitted at
  each fixed $\tau_s$) gives cost 30, 11 and 3.0 at 5, 10 and 20 ms, the
  minimum 1.88 at 36.3 ms, and 2.2, 2.8 and 4.1 at 60, 100 and 300 ms, where
  $\sigma_s$ rises to 0.099 mV **[run]**. The cost is a scaled distance
  between medians, not a likelihood, so no interval is read from it. The
  twin's 1 s step extrapolates the law beyond the longest window that
  measured it; (c)'s SS series would bound it.
- (f) A cell whose two protocols were sampled at different rates (none in
  L3_exc; other groups mix 50 and 200 kHz, D-014): how the fast component
  transfers between them.
- (g) Unchanged: D-016 (d) and (e); D-015's open point (stationarity across a
  response).

**Corrections.** (1) **D-016 Why (iii)** framed the stored LS pre-window as
about 0.5 s or about 1 s and argued from that length. `ls_pre_ms` is 265 ms in
all 18 L3_exc cells **[run, cluster 2026-09-28]**: both framings were wrong,
the argument built on them is moot, and the start-of-sweep transient it was
designed to detect is absent (first 50 ms quieter than the last 132.5 ms;
locked share 0.02). D-016 (b) -- starting the fitter's LS noise and baseline
windows later -- has no case on L3_exc. (2) **D-011 Why (iii)** quotes
Moradi Chameh et al. 2021 as "sag amplitude $1.8 \pm 1.6$ mV". That is the
L2&3 **rebound depolarisation**; the L2&3 **sag** voltage amplitude is
$1.7 \pm 1.1$ mV ($n = 56$), measured with 600 ms steps from 0 to $-400$ pA,
and its ZD7288-sensitive part falls from $1.3 \pm 0.9$ to $0.2 \pm 0.3$ mV
($n = 13$) **[KB, full text, re-read 2026-09-28]**. D-011's argument does not
change; its number was misattributed.

**Sources.** **[user]**, 2026-09-28 15:39. The L3_exc noise tables
(`noise_L3.csv`, 2026-09-23; `noise_L3_v2.csv`, 2026-09-28), written by
`noise_calibration.py` on the archive and pasted by the user **[run,
cluster]**. The cohort fit, its $\tau_s$ profile, the visible-share arithmetic
and the trend-share Monte Carlo **[run, 2026-09-28]**. The estimator, the
generator, the manifest and the archive writer's SS storage; AllenSDK 2.16.2
`get_ephys_sweeps` **[src]**. The Allen white paper's cell- and sweep-level QC
criteria and SS protocol; Moradi Chameh et al. 2021 **[KB]**. PubMed,
2026-09-28: three conjunctive queries on whole-cell drift and pyramidal-cell
voltage noise returned nothing; "voltage noise pyramidal neurones" returned
81 records, of which Yaron-Jakoubovitch et al. 2008 was read in full text (rat
L4/5 slices at room temperature: somatic voltage-noise spectra flat up to an
average cut-off of 1.4 Hz at rest and falling steeply above it,
[DOI](https://doi.org/10.3389/neuro.03.003.2008)) -- used qualitatively only,
a rat preparation; Jacobson et al. 2005 is still abstract only through the
connector (a PMC record exists; its full text came back empty). OU as a
sampled AR(1): textbook, from memory.

**Status:** active. Implementation not started; waits for (a)-(c).
**[updated 2026-10-16: (b) settled by D-051 -- one law per cell for both protocols; (d) settled by D-033 (family), D-048/D-049/D-050 (weights) and D-051 (statistic set); (c) by D-034, with D-051's provisional $h^*$.]**
**[updated 2026-10-21: superseded in part by D-073 -- the per-cell law is no longer the generator input as such; each twin sweep's noise is calibrated on its own real sweep, the per-sweep form open.]**
**Note added 2026-09-28 (later), after the user asked where the fitting approach
comes from; no decision changes.** *Lineage.* (D-017.2) is, mathematically, the
two-timescale mixture of Ornstein-Uhlenbeck processes of Zeraati, Engel & Levina
2022 (PubMed full text, [DOI](https://doi.org/10.1038/s43588-022-00214-3)): an
AR(1) with coefficient $\rho_f \in (0,1)$ is an OU process sampled at $\Delta t$
(textbook, from memory). They show that the sample autocorrelation of a finite
series is biased, mainly through the deviation of the sample mean from the true
mean, and that its expected value is intractable for processes with several
timescales; they therefore fit timescales by making a generative model reproduce
the same summary statistic computed on synthetic data of the same duration and
number of trials (adaptive ABC; package abcTau, BSD-3-Clause, on GitHub, not on
PyPI **[src]**). Choosing the parameters under which the expected value of a
calibration statistic matches its observed value is indirect inference (Weltz,
Laber & Volfovsky 2024, PubMed full text, PMC12448677). The fit of this entry is
the assistant's own closed-form, least-squares shortcut of that approach, not a
published procedure. *Check.* The shortcut was not tested when this entry was
written. Under the fitted cohort law, the ratio of expectations overshoots the
median of the pipeline's autocorrelation estimator by 0.02 to 0.07 across the
eleven autocorrelation statistics (0.07 for the last 132.5 ms at a 10 ms lag); $\sqrt{\mathbb{E}[s^2]}$ matches the
mean of the SD estimator, whose median sits 1-5 % lower **[run]**. Refitted by
simulated medians (same 14 targets and scales; 600 SS and 300 LS windows with
common random numbers), the cohort law is $\sigma_f$ = 0.0381 mV, $\rho_f$ =
0.179, $\sigma_s$ = 0.0560 mV, $\tau_s$ = 37.3 ms, cost 0.69, against 3.73 for
the closed-form optimum under the same objective; a start at 80 ms stalls at
53.5 ms (cost 1.07) **[run]**. Open point (d) therefore leans to simulation
through `noise_calibration.py`'s own estimators, with the closed form as a
starting point only.

**Note added 2026-09-28 (later, 2), after the topic document and two independent
checks of it; no decision changes.** The full treatment is
`claude/TEEG_Stage7_noise_model_2026-09-28.md`. Five statements of this entry are
corrected here rather than edited in place:
(1) *What it implies* calls `noise_ss_*` "Short Square". SS is the Allen **Square
Subthreshold** protocol (0.5 ms pulses to $\pm 200$ pA, 20 per sweep, 200 ms
apart [corrected 2026-10-20: 37 per sweep, 200.5 ms onset to onset in the
archive's cells; data, changelog 2026-10-20 (later, 5)]); the Allen *Short Square* is 3 ms pulses stepped from 100 pA up to
action-potential threshold **[KB]**. The naming rule for the new manifest columns
stands.
(2) *Why (i)* reads the first 50 ms being quieter than the last 132.5 ms as
evidence against a start-of-sweep transient. The segments differ in length, and a
stationary slow component makes the shorter one quieter (9 % against 60 % of
$\sigma_s^2$ visible under the cohort law) **[run]**; the evidence is that the
stationary law fitted jointly to all 14 statistics reproduces the first segment's
SD (0.0466 mV predicted by simulated medians, 0.0465 measured) **[run]** -- a
consistency check on cohort medians. "Not sweep-locked" holds for the cohort
median only: under the cohort law the per-cell locked-share null for 265 ms
windows has a 95th percentile of 0.111 ($K$ = 5) and 0.094 ($K$ = 6), and five
cells, all with $K$ = 6, lie above it -- 529878215 (0.186, above the 99th
percentile, 0.155), 571691416, 548425467, 571624532, 537313600 **[run; cluster]**.
A weak sweep-locked component (a transient, or a creep of consistent sign) is
therefore not excluded in those cells; indicative, since slower laws widen the
null.
(3) *What it implies* infers from the Allen cell-level electrode-drift criterion
(final voltage within 1 mV of the zeroed electrode voltage per 10 min **[KB]**)
that a within-sweep creep "is not the electrode" in a cell that passed it. The
criterion bounds the NET offset over the recording: it excludes an electrode
creep of consistent sign over minutes, not within-sweep creeps of varying sign.
The generator docstring's "electrode/seal drift" is unsupported rather than wrong
**[reasoning]**.
(4) Open point (a) calls 526785799 and 571430815 "the two drifting cells" and
compares their 5-sweep medians with a single-window null. The per-cell null is
that of a 5-sweep median: mean 0.078, SD 0.064 under the cohort law, so the cells
sit 8-9 SD above it **[run]**. Against stationary slow noise scaled to each cell:
526785799 is reproduced in 1-2 % of simulated cells when the fast part takes the
whole early-segment SD ($\sigma_f$ = 0.043 mV), and in 5-8 % when $\sigma_f$ is
lowered to 0.034-0.036 mV so that the early segment matches, with the last
segment then louder than the cell's (0.052-0.054 against 0.046 mV) -- a creep is
the better reading, not decisively; 571430815 is reproduced in 7.5-12 %
(correlation time 0.3-2 s) -- compatible with its own slow noise **[run]**. They
are drift candidates. The cohort median trend share, 0.14, exceeds its null under
the cohort law (0.067 $\pm$ 0.017; probability 0.0004): either modest creeps are
common across the cohort or the slow noise is slower than 36 ms, or both.
(5) The first status note's "eleven autocorrelation statistics" are ten, and the
SD estimator's median sits 1-3 % (not 1-5 %) below its mean; the ratio of
expectations overshoots the autocorrelation estimator's Monte-Carlo mean by
0.008-0.05 and its median by 0.02-0.07 **[run]**.
*For open point (c).* The Allen per-sweep QC feature `vm_delta_mv` is
$\lvert$mean over the 500 ms before the stimulus $-$ mean over the last 500 ms of
the recording$\rvert$ **[src, IPFX 2.1.2]**, a direct measure of the net baseline
change across a sweep. The EphysSweep model has this field and AllenSDK's
`get_ephys_sweeps` queries those records **[src; repository, metadata only]**;
whether it is populated for our LS sweeps is **not verified**. With the two
windows' centres at least 1.5 s apart (post-step length not verified): a
526785799-sized creep (0.7 mV/s, a rough scale from medians) continuing across
the sweep gives 1.05 mV or more, against a stationary-wander null of median
0.15-0.19 mV and 90th percentile at most 0.47 mV -- decisive; for 571430815 the
wander null has median 0.37-0.70 mV, 90th percentile 0.91-1.71 mV and exceeds
1 mV in 7-34 % of sweeps, against at least 3.6 mV for a 2.4 mV/s creep -- only
large values decide **[run; reasoning]**. `slow_noise_rms_mv` (SD over the 500 ms
before the stimulus) bounds $\tau_s$ only weakly: per-sweep median 0.060 mV under
the cohort law against 0.065 mV under the $\tau_s$ = 300 ms profile law **[run]**.
The assistant's recommendation for (c) becomes `vm_delta_mv`, pulled on the
cluster, together with the SS pre-pulse series.
*For open point (b).* Under the cohort law the slow component supplies 25 %, 41 %,
51 % and 57 % of the baseline-subtracted noise variance of one SS trace at 0.5, 5,
10 and 15 ms after the pulse **[run]**: the SS residual is not fast noise only.
*Sources.* Scripts and outputs in `TEEG_Stage7_noise_scripts_2026-09-28.zip`
(delivered with the document) **[run]**; IPFX 2.1.2 `qc_feature_extractor`,
`epochs`, `qc_features`, `qc_criteria.json` and AllenSDK 2.16.2
`CellTypesApi.get_ephys_sweeps` **[src]**; Allen white paper **[KB]**; Allen
Brain Map API field list of three test-pulse records of specimen 508282493
**[repository, metadata only]**.

**Note added 2026-10-01, after the user asked for the `vm_delta_mv` route to be
built; no decision changes.**
*The tool [user], 2026-10-01.* The query of the Allen per-sweep QC features and a
measurement of the archive's own Long Square sweeps go in one script ("Yes, same
script"). Delivered as `Ih Fit/ls_baseline_qc.py`, diagnostic only (nothing
downstream reads it), with `smoke_ls_baseline_qc.py`: 10/10 on a fresh clone of
`main` at `2fd7643`, each check mutation-tested **[run]**; pending the user's push,
pull and run. Per hyperpolarising single Long Square sweep -- the sweeps
`noise_calibration.py` measures, read through its own loader and pre-window mask
**[src]** -- it writes (i) from the archive, with no network: the stored recording
after the step (`tail_ms`), which settles note 2's "post-step length not verified"
for the archive; the SIGNED change from the pre-window mean to the mean of the last
500 ms of the stored sweep (`delta_end_mV`, only when that window lies wholly after
the step); and, beside it, what a creep continuing at the pre-window's own slope
would give (`creep_pred_mV`); and (ii) with `--allen fetch`, the EphysSweep records
of the public API (the query AllenSDK's `get_ephys_sweeps` sends, built with the
standard library), joined by sweep number, with a check on the pre-window level.
*Recorded with it.* (1) `vm_delta_mv` = $\lvert$`pre_vm_mv` $-$ `post_vm_mv`$\rvert$,
with `pre_vm_mv` the mean over the 500 ms before the stimulus and `post_vm_mv` the
mean over the last 500 ms before the recording's last non-NaN sample **[src: IPFX
2.1.2 `current_clamp_sweep_qc_features`, `measure_vm_delta`,
`get_recording_epoch`]**; the signed Allen change `post_vm_mv - pre_vm_mv`
therefore comes from the same records, and the creep test compares its sign and
size with those of `creep_pred_mV` (~~sign alone is weak evidence: a wander slower
than the windows' spacing also keeps the pre-window's direction for a while
**[reasoning]**~~ **[corrected 2026-10-01 (later)]**: with these windows a stationary wander of
any timescale agrees in sign 0.49-0.53 of the time, so the sign is strong
evidence against a continuing creep and silent on the wander's timescale;
note 4). (2) A summarising fetch of the live API on 2026-10-01 returned
the record of one hyperpolarising Long Square sweep of 526785799 with
`vm_delta_mv` filled in **[repository, summarising fetch; no number used]**;
asked for a table, the same fetch first put `peak_deflection` under
`vm_delta_mv`, so no value from such a fetch is used; the script reads the raw
JSON. (3) The wander nulls of note 2
hold for the Allen windows (two 500 ms windows, centres at least 1.5 s apart) and
for one sweep; `delta_end_mV` uses the 265 ms pre-window and a spacing set by
`tail_ms`, so its null is a different distribution and has not been computed
**[open]**.
*Sources.* `ls_baseline_qc.py`, its smoke and the mutation tests **[run]**; IPFX
2.1.2 and AllenSDK 2.16.2 **[src]**; Allen Brain Map API **[repository,
summarising fetch]**. Handoff: `claude/TEEG_Stage7_noise_handoff_2026-10-01.md`.

**Note added 2026-10-01 (later), after the first cluster run of `ls_baseline_qc.py`
on L3_exc; no decision changes.**
*Run.* Commit `3dfa263` on `main`, whose four files are the delivered patch byte
for byte **[run]**; 18 cells, 104 hyperpolarising single Long Square sweeps.
Every stored sweep holds 6.000 s of recording after the step, so every end window
starts 5.5 s after the step and the two windows' centres are 6.89 s apart
**[run, cluster]**. The Allen half did not run: the network guard stopped the
downloads after two consecutive errors although `module load proxy` had been run
(`mode fetch (network disabled); 0/104 matched`); the error text in
`lsqc_L3.log` has not been read.
*(a) No creep.* A creep continuing at a sweep's own pre-window slope would move
the baseline by `creep_pred_mV` over the 6.89 s. Signs agree in 53 of 104 sweeps
(binomial, two-sided p = 0.92). With these windows a stationary wander gives a
sign agreement of 0.49-0.53 whatever its timescale (OU with $\tau$ from 37 ms to
30 s, and a random walk; Monte Carlo, 20000 sweeps each), because the later
change is dominated by the return towards the cell's mean level, which is
unrelated to the local slope; a continuing creep gives agreement near 1. The sign
test therefore rejects a continuing creep and says nothing about the wander's
timescale **[run]**. In the three sweeps of 526785799 with |`creep_pred_mV`| of
4.8-6.2 mV, |`delta_end_mV`| is at most 0.42 mV; in 571430815, |`creep_pred_mV`|
5.3-23.9 mV stands against |`delta_end_mV`| 0.30-0.81 mV; the least-squares slope
of `delta_end_mV` on `creep_pred_mV` over these 10 sweeps is 0.005 **[run,
cluster; run]**. Both cells' tail changes lie inside their own 09-28 wander
scenarios: the probability that the 5-sweep median of |`delta_end_mV`| reaches
the observed one is 0.23-0.70 for 526785799 (observed 0.288 mV) and 0.15-0.80
for 571430815 (0.620 mV) **[run]**. Their high trend shares are slow wander, not
creep, and a per-sweep tilt $b$ calibrated from pre-window slopes would be too
large by more than an order of magnitude **[run; reasoning]**.
*(c), (e) Variance at seconds.* The baseline does change over the 6.89 s. The
per-cell median of |`delta_end_mV`|, a number computed from each cell's 5-6
sweeps, is 0.048-0.620 mV, cohort median 0.128 mV **[run, cluster]**. Under the
cohort laws of 09-28 that per-cell median has a distribution with median
0.021-0.024 mV and 99th percentile 0.046-0.056 mV, and 17 or 18 of the 18 cells
lie above the 99th percentile, depending on the law **[run]**: a law fitted to
windows of at most 265 ms misses variance at seconds in every cell. Added to the
simulated-medians cohort law, an OU component reproducing the cohort median needs
$\sigma$ 0.142, 0.189 or 0.293 mV at $\tau$ 3, 10 or 30 s; at $\tau$ 0.3 or 1 s
it would raise the cohort's 265 ms window SD from the measured 0.0594 to 0.098 or
0.072 mV, so, if it is stationary membrane or recording noise, it sits at $\tau$
of about 3 s or more **[run; reasoning]**. One lag cannot fix its shape, and the
fit is sensitive to the shape: the SD of the mean baseline error over the 1 s step
(step mean minus pre-window mean) is 0.031 mV from the 37 ms component,
0.049-0.070 mV from a seconds-scale OU matched to the 6.89 s change, and
0.018 mV from a linear tilt matched to the same change **[run]**. The SS
pre-pulse series (20 baselines 200 ms apart per sweep, at rest **[KB; src]**)
[corrected 2026-10-20: 37 per sweep, 200.5 ms apart, lags 0.2005-7.218 s; data]
gives the variogram from 0.2 to 3.8 s and discriminates between them; it is
not written.
*Hint, not significant.* In 526785799 `delta_end_mV` rises with step size
(Spearman 0.90, p = 0.04, 5 sweeps, uncorrected). The protocol runs from
$-110$ pA upwards in 20 pA increments **[KB]**, so recording order and amplitude
are collinear, and a step-induced after-effect cannot be told from a drift over
the protocol. The per-cell median of `delta_end_mV` is positive in 12 of 18 cells
(p = 0.24) **[run]**.
*Searches, 2026-10-01.* PubMed: within-sweep drift of the membrane potential in
slice whole-cell recordings (0 records); slow afterdepolarisation after
hyperpolarisation in neocortical pyramidal cells (5 records, none on a
subthreshold hyperpolarising step at rest; none used). bioRxiv, neuroscience,
last 30 days (30 preprints): nothing relevant.
*Sources.* Output pasted by the user, 2026-10-01 14:34 **[run, cluster]**;
`lsqc_readout/readout.py` and `sign_check.py` (exact variances of window means
under OU laws; Monte Carlo of medians and signs) **[run]**; Allen white paper,
Long Square protocol **[KB]**.

**Note added 2026-10-07; no decision of this entry changes.** The
"lag-$L$ autocorrelation in the fitter's convention" of the Statement is
superseded by D-028: the statistics of the per-cell fit are computed with the
textbook estimator through statsmodels, on the data and on the simulations
alike. The cohort fits of the status notes used the old convention and are
recomputed as listed in `claude/TEEG_acf_textbook_change_plan_2026-10-07.md`.

---

## D-018 -- The background level of the blurred-tube diameter fit is fixed from a median of the focal plane, not fitted

**Date:** 2026-09-30. **Binds:** the diameter re-measurement workstream
(Allen 63x brightfield stacks, first specimen 529878215): Eq. 11 and local
decision D4 of its handoff `handoff_diameter_remeasurement.md` (uploaded by
the user in chat on 2026-09-30, not in project knowledge); the planned
`allen_image_diameter.py` and its smoke suite (nothing coded yet); the
synthetic bias table $b(d, \varphi)$ of that handoff's Eq. 13, which is built
by running synthetic stacks through the same fit; the per-node CSV; and,
downstream, every diameter written into the corrected SWCs of D-013.

**Decision [user], 2026-09-30 16:16.** "I would fix it. We can make a median
across the image in the focal plane." The background brightness $B$ leaves
the fitted vector of the blurred-tube Beer-Lambert model; its value is a
median of the grey levels of the focal plane in which the node is measured.

**Statement.** For a dendrite node $i$ with sharpest plane $k^*_i$ (handoff
Eq. 1), let $I_{k^*_i}(x, y)$ (grey levels) be the image of that plane,
$(x, y)$ (µm) in the global image frame, and $\mathcal{R}_i \subset
\mathbb{R}^2$ the region over which the median is taken (open point (a)).
The background is the number computed from the data
$$\bar B_i = \operatorname{median}_{(x, y) \in \mathcal{R}_i} I_{k^*_i}(x, y)
\quad (\text{grey levels}),
\tag{D-018.1}$$
plugged in for the model parameter $B$ of handoff Eq. 11. With $I_i(v_n)$ the
measured profile along the measuring axis $\hat y_i$ at positions $v_n$ (µm),
$s_d(v) = \sqrt{1 - (2v/d)^2}$ for $\lvert v \rvert \le d/2$ and 0 otherwise,
$T_{d, \alpha, v_0}(v) = \exp\bigl(-\alpha\, s_d(v - v_0)\bigr)$ the
Beer-Lambert transmittance of handoff Eqs. 10-11, and $g_\sigma$ a Gaussian of
standard deviation $\sigma$ (µm, fixed), the fit is, for each fixed $i$,
$$(\hat d_i, \hat\alpha_i, \hat v_{0,i}) = \operatorname*{arg\,min}_{d > 0,\
\alpha \ge 0,\ v_0 \in \mathbb{R}} \sum_n \Bigl[I_i(v_n) - \bar B_i\,
\bigl(T_{d, \alpha, v_0} * g_\sigma\bigr)(v_n)\Bigr]^2 .
\tag{D-018.2}$$
Status: $\bar B_i$ computed; $\hat d_i$, $\hat\alpha_i$, $\hat v_{0,i}$
computed (fitted); $\sigma$ configured.

**Why.** One free parameter fewer in a fit of about 50 samples whose
thin-process regime is already nearly degenerate between $d$ and $\sigma$
(handoff Eq. 12, which is why $\sigma$ is fixed) **[user; reasoning]**. The
trade-off, recorded when the choice was made: a fitted $B$ can absorb a flank
contaminant (a neighbouring process, a spine, haze), while a median ignores
one only if it is a small fraction of $\mathcal{R}_i$; and a fixed
$\bar B_i$ that is off cannot be absorbed by the fit, so the error moves into
$\hat d_i$ and $\hat\alpha_i$ **[reasoning, not simulated]**.

**What it implies for code and tables.** $\bar B_i$ is computed before the
fit and written per node in the per-node CSV, together with the definition of
$\mathcal{R}_i$ used. The fit exposes three parameters by default; a
fitted-$B$ variant, if kept at all, is an opt-in labelled as a comparison.
The synthetic bias table $b(d, \varphi)$ is built with the same rule applied
to the synthetic blocks, so that the correction matches the estimator. The
smoke suite asserts the default (no fitted $B$) and recovers a known
synthetic background.

**What it does not decide [open].**
- (a) **The region $\mathcal{R}_i$.** The user's words allow two readings:
  the whole plane $k^*_i$ (7605 × 9433 px, about 0.87 × 1.08 mm, 71.7 Mpx;
  the image module's `check_request_size` caps a returned image at 12 Mpx, so
  a full-resolution median of a whole plane needs tiling or a downsampled
  plane), or the block fetched around the node (about 10 × 10 µm,
  `fetch_zblock`). The assistant's recommendation: the node's block in plane
  $k^*_i$, with the pixels near the traced path masked out (e.g. within the
  Allen radius + 1 µm of any SWC segment in the block). A plane-wide number
  cannot follow slow changes of background across a millimetre field
  (illumination shading, local tissue density, out-of-focus stain)
  **[reasoning; not measured on the Allen stacks]**.
- (b) **The stained-fraction bias of a median.** If a fraction $f$ of the
  pixels of $\mathcal{R}_i$ are stained and all darker than the background
  median, (D-018.1) sits at the background's $(0.5 - f)/(1 - f)$ quantile:
  for Gaussian background noise of SD $s_B$ (grey levels) that is $-0.07$,
  $-0.14$, $-0.32$ and $-0.57\,s_B$ at $f$ = 0.05, 0.10, 0.20 and 0.30
  **[reasoning, arithmetic]**. Small for a thin branch in a 10 µm block; not
  small for a thick trunk, a crossing or a block near the soma, which is what
  the mask of (a) is for.
- (c) Whether the focus score's $B_{j,k}$ (handoff Eq. 1, the median of the
  profile ends in each plane) switches to the same rule.

**Sources.** **[user]**, 2026-09-30 16:16. Eqs. 1 and 10-13,
`check_request_size`, `fetch_zblock`: the user's handoff
`handoff_diameter_remeasurement.md` (2026-09-30). Plane size and pixel size:
Allen Brain Map API for specimen 529878215, as recorded in that handoff
(verified by live query in the earlier design chat, not re-queried). The
quantile arithmetic **[reasoning]**.

**Status:** active. Amends local decision D4 of the diameter handoff ($B$ no
longer fitted); the handoff file itself is the user's upload and has not been
edited. **Recorded in the same exchange, not a decision:** that handoff's
Eq. 12 line "σ ≈ 0.1 µm (FWHM ≈ 0.24 µm; textbook)" is wrong in its
derivation. 0.24 µm is the radius of the first dark ring of the Airy pattern,
$0.61\,\lambda/{\rm NA}$ at $\lambda$ = 0.55 µm and NA 1.4; the FWHM is
0.20 µm ($0.51\,\lambda/{\rm NA}$). The kernel acting across a straight
branch is the line-spread function, whose Gaussian-core $\sigma$ is 0.083 µm
(paraxial Airy) or 0.080 µm (scalar high-NA Debye integral), scaling with
$\lambda$. The effective $\sigma$ of the imaging chain comes back to about
0.10 µm only after pixel integration and the bilinear interpolation of
handoff Eq. 9 add $px^2/12 + px^2/6 = px^2/4 = 0.0033$ µm$^2$ (simulated
0.0034) **[run, 2026-09-30]**; JPEG at quality 75-95 added at most
0.0002 µm$^2$ on a synthetic line, and a 0.14 µm defocus 0.001 µm$^2$
**[run]**. **Amended 2026-10-01 by D-019:** the fitted darkness parameter of
(D-018.2) is $\mu$, not $\alpha$; $\bar B_i$ fixed and $\sigma$ fixed stand.
**Open points (a) and (c) closed 2026-10-06 by D-023** (block with the path
masked; the focus score keeps the profile-ends median); (b) is carried by the
mask and by the bias table, which is built with the same $\bar B$ rule.

---

## D-019 -- The diameter fit is parameterised by the stain's absorption coefficient, with the tilt taken from the local line fit

**Date:** 2026-10-01. **Binds:** the same objects as D-018: handoff Eq. 11 as
amended by (D-018.2); the planned `allen_image_diameter.py` and its smoke
suite; the per-node CSV; the synthetic bias table $b$, whose phantoms carry a
$\mu$. Nothing is coded.

**Decision [user], 2026-10-01 18:26.** "I think the fact that the whole alpha
is fitted is not correct. What should be fitted is the mu only. phi should be
derived by the locally fitted line." The darkness parameter of the fit is the
absorption coefficient $\mu$ of the stained cytoplasm; the tilt $\varphi_i$
enters as a number computed beforehand from the line fit through the centres
(handoff Eqs. 4-5) and is never a free parameter of Eq. 11.

**Statement.** For each fixed node $i$, let $\varphi_i \in [0, \pi/2)$ (rad)
be the tilt computed by handoff Eq. 5, $\bar B_i$ (grey levels) the background
of (D-018.1), $\sigma$ (µm) the fixed blur, $s_d(u) = \sqrt{1 - (2u/d)^2}$ for
$\lvert u \rvert \le d/2$ and 0 otherwise, and
$$T_{d, \mu, v_0}(v \mid \varphi_i) = \exp\!\Bigl(-\frac{\mu\, d}{\cos\varphi_i}\,
s_d(v - v_0)\Bigr),$$
$$(\hat d_i, \hat\mu_i, \hat v_{0,i}) = \operatorname*{arg\,min}_{d > 0,\ \mu \ge 0,\
v_0 \in \mathbb{R}} \sum_n \Bigl[I_i(v_n) - \bar B_i\,\bigl(T_{d, \mu, v_0}(\cdot
\mid \varphi_i) * g_\sigma\bigr)(v_n)\Bigr]^2 .
\tag{D-019.1}$$
Here $d$, $\mu$ (µm$^{-1}$) and $v_0$ are the model's true values and
$\hat d_i$, $\hat\mu_i$, $\hat v_{0,i}$ the numbers the fit returns. For each
fixed $\varphi_i < \pi/2$ the map $(d, \mu) \mapsto (d, \alpha = \mu d /
\cos\varphi_i)$ is a bijection of $\mathbb{R}_{>0} \times \mathbb{R}_{\ge 0}$
onto itself, so with $\mu$ free at every node (D-019.1) and (D-018.2) have the
same minimiser in $(d, v_0)$, with $\hat\mu_i = \hat\alpha_i \cos\varphi_i /
\hat d_i$ **[reasoning]**; in a noise-free test the two parameterisations
returned the same $\hat d_i$ to within 0.002 $d$ at every $(d, \mu, \sigma)$
tried but one, where they differed by 0.02 $d$ at a degenerate point
**[run]**. Per node, the change is in what is reported, not in what is
estimated.

**Why.** $\mu$ is a property of the stain, comparable between nodes; $\alpha$
mixes it with the node's own $d$ and $\varphi_i$. Reporting $\hat\mu_i$ turns
the uniform-stain assumption of Eq. 11 into something the data can test, and
it is the quantity a shared estimate would be built from **[user;
reasoning]**.

**What it implies for code and tables.** The per-node CSV carries
$\hat\mu_i$ and $\varphi_i$; $\hat\alpha_i$ is derived from them and kept for
diagnostics only. The fit's bounds are on $\mu$. The smoke suite asserts that
the $\mu$ form returns the same $\hat d_i$ as the $\alpha$ form on synthetic
profiles (a regression guard for the bijection) and that $\hat\mu_i$ recovers
a known synthetic $\mu$; $\varphi_i$ is an input of the fit function, so an
error in it enters only through $\cos\varphi_i$.

**What it does not decide [open].**
- (a) **Per node or shared.** If $\mu$ is shared across nodes (one value per
  cell or per branch, estimated beforehand), the darkness is no longer free
  and carries information on $d$: for a faint process the area of the dip is
  $\bar B_i\,\mu\,\pi d^2 / (4\cos\varphi_i)$, which fixes $d$ independently
  of $\sigma$, and $d \propto \mu^{-1/2}$ at fixed area **[reasoning]**.
  Noise-free test, $\sigma_{\rm true} = 0.10$ µm, $\varphi = 15°$, $\mu =
  0.6$ µm$^{-1}$, fitted with $\sigma = 0.125$ µm: $\hat d/d$ = 0.35, 0.88,
  0.98, 1.00 at $d$ = 0.3, 0.5, 1, 2 µm with $\mu$ free per node, against
  1.03, 1.015, 1.00, 1.00 with the true $\mu$ shared; at $\mu = 3$ µm$^{-1}$,
  0.52-0.54, 0.89, 0.98, 0.98 against 1.03, 1.01, 1.00, 1.00. A shared $\mu$
  20 % too high gives 0.91-0.96 $d$ at $\mu = 0.6$ and 0.92-0.99 $d$ at
  $\mu = 3$, with or without the $\sigma$ error **[run]**. Sharing trades
  sensitivity to $\sigma$ for sensitivity to a non-uniform stain. The
  assistant's recommendation: two stages -- fit $\mu$ per node first; read
  $\hat\mu_i$ against path distance, depth $z$ and branch on nodes with
  $\hat d_i \gtrsim 1$ µm and $\varphi_i \lesssim 20°$, where the table shows
  $\hat d$ hardly depends on $\sigma$; share $\hat\mu$ per branch or per cell
  only where it is uniform to about $\pm 10\,\%$ (about $\pm 5\,\%$ in $d$).
  Whether the biocytin-DAB stain is uniform along these dendrites is **not
  known** for the Allen stacks.
- (b) Whether the vertical path $d / \cos\varphi_i$ of handoff Eq. 10 holds
  for thick or steep branches whose vertical extent exceeds the depth of
  field; for those the measured darkness is not $\mu d / \cos\varphi_i$, and
  they are kept out of any shared estimate of $\mu$ **[reasoning]**.
- (c) The bias table $b$ belongs to an estimator: a per-node and a shared-$\mu$
  fit need their own tables, built with the same variant.

**Sources.** **[user]**, 2026-10-01 18:26. Handoff Eqs. 4-5 and 10-11 and
D-018. The bijection and the faint-limit area **[reasoning]**. The noise-free
comparison `mu_tie.py` (Eq. 11's own forward model, sampled at 0.1144 µm)
**[run, 2026-10-01]**. No literature search bears on a reparameterisation;
none was run for it.

**Status:** active. Amends D-018 (the darkness parameter); D-018's $\bar B_i$
and fixed $\sigma$ stand. **Open point (a) closed 2026-10-06 by D-023: per
node (stage 1 of the two-stage route); a shared $\mu$ is a labelled
comparison with its own table.**

---

## D-020 -- The diameter re-measurement documents live in the repository

**Date:** 2026-10-04, 12:47 (appended 2026-10-06; project knowledge was full
on 10-04 and 10-05). **Binds:** where the diameter re-measurement
workstream's documents live, how they are updated, and where the assistant
reads them.

**Decision [user].** The documents of the diameter re-measurement workstream
live in the Towards-EEG repository, `Passive Features/Diameter
Re-measurement/docs/`, and are kept current there by commit and push; the
assistant reads and cites them from the repository.

**Statement.** The canonical copy of every document of this workstream (the
design handoff, the procedure, the mathematics, the optics document, the
optics notes, the figures spec, the later handoffs) is the file in that
folder on `main`; a new document or an update is a commit pushed there.
Project-knowledge copies are not the reference.

**Why.** Taken after the project reached its size limit and refused the
optics document; the repository has no such limit and keeps the history.

**What it implies for code and tables.** The implementation chat (D-021)
pulls `main` at the start of each session and reads the changed docs
(`git log --stat origin/main -- "Passive Features/Diameter
Re-measurement/docs"`); equation numbers in code comments and in
`specs/SPEC.md` refer to the repo copies. On 2026-10-06 the stale
project-knowledge copies of the procedure, mathematics and optics-notes
documents were deleted with the user's OK after a check that the repository
holds their content (the optics notes byte-identical; the other two the v1
texts, which the repo's v1.1/v1.2 contain unchanged by their revision
records).

**What it does not decide [open].** Nothing about the documents' content.

**Sources.** **[user]**, 2026-10-04 12:47, recorded in the folder's
`README.md` the same day; the deletion check **[run, 2026-10-06]**.

**Status:** active.

---

## D-021 -- Implement the whole diameter pipeline now, with open choices as configuration

**Date:** 2026-10-06, 14:30. **Binds:** the whole diameter-estimation code
(`src/`, `specs/SPEC.md`, the smoke suites); how open method choices enter
the code. Full text: `Passive Features/Diameter
Re-measurement/docs/TEEG_diameter_implementation_handoff_2026-10-06.md`.

**Decision [user].** "In this chat I will continue to study the theory
underlying the process. However it is wise to save time that you start to
implement the whole pipeline for the diameter estimation [...] Is much faster
to implement slight changes while the entire code is already set rather than
waiting me to comprehend in deep everything." The pipeline is implemented
now, end to end, in a separate implementation chat, in parallel with the
theory study.

**Statement.** Every choice the method has not settled enters the code as a
named parameter of a configuration object, each at a stated provisional
default with its source written next to it; an alternative method
(estimator variant, kernel family, absorption treatment) is a swappable
component chosen by name from the configuration; later theory results change
a parameter value or a single block. **[assistant, under coding directives
2-3]** The provisional defaults are put to the user in one batch at the start
of the implementation chat; logic whose default is not confirmed is not
written.

**Why.** Code that runs end to end on synthetic stacks exposes interface and
scale problems early; changing a default is cheap once the structure exists;
the open questions (procedure §5) are mostly values, not structure.

**What it implies for code and tables.** `config.py` holds every parameter
of the handoff's "Configuration" section, each with a `# source:` comment;
`specs/SPEC.md` cites the decision or document behind every default; the
non-deliverable variants are labelled comparisons in file names and figure
titles.

**What it does not decide [open].** The open method choices themselves
(answered the same afternoon: D-022 to D-024) and where the code lives
(D-022).

**Sources.** **[user]**, 2026-10-06 14:30; the implementation handoff.

**Status:** active; the batch was answered in D-022 to D-024.

---

## D-022 -- Code layout, branch, the 2026-09-23 image modules, and where the production table is built

**Date:** 2026-10-06, about 15:05 (first card of the D-021 batch).
**Binds:** the layout of the diameter pipeline's code, spec and tests; the
branch; the six image modules the real-data path depends on; the host of
the production bias table; the project-knowledge cleanup.

**Decision [user].** (i) Layout: the code lives inside the workstream folder
-- `Passive Features/Diameter Re-measurement/src/` (the four 2026-09-23
image modules flat, plus the package `allen_diameter/`), `scripts/`,
`tests/smoke/` -- while the spec is the root-level `specs/SPEC.md` whose
Coverage section names that folder, so the scheduled integration tester
finds the repository. (ii) The modules come from the user's own copies: the
unsuffixed `allen_image_io.py`, `allen_image_align.py`,
`allen_image_measure.py`, `allen_image_plot.py`, `smoke_allen_image.py`,
`robustness_registration.py` in `~/Downloads` on the user's computer are the
2026-09-23 versions (sha256 prefixes dae5d24f7f2c, 14dd899057a8,
4614cd949ba1, 3b9f1a07e544, d7811c681b0f, 18059c7109ee, exactly the design
handoff's table; the files attached in chat with a `(1)` suffix were the
2026-09-22 versions). Their suites re-run in the sandbox: 20/20 and 40/40
(0/10 false positives, 0/30 misses, max |s* error| 0.10 µm). (iii) The
production bias table is computed on davinci via PBS (`hpc-git-delivery`,
`hpc-python-compat`); the sandbox builds reduced tables for tests. (iv)
Project-knowledge cleanup: "Delete but be sure everything lives in the repo,
otherwise push them in the repo" -- done as recorded under D-020.

**Statement.** Branch `sci/diameter-pipeline` off `main` (51d88ac). Block 0
commits the six modules byte-identical. `specs/SPEC.md` Coverage: `Passive
Features/Diameter Re-measurement/src/`, `.../scripts/`, `.../tests/`.

**Why.** (i) Repo precedent (workstreams live in plain folders under
`Passive Features/`) and the night tester's convention (root `specs/SPEC.md`,
Coverage for a subtree) are both satisfied; the installable package's
`passive/` says passive fits are "managed separately". (ii) Byte identity
keeps the validated suites valid. (iii) A full table is ~10^4 stacks of
minutes each in the sandbox; davinci has the CPUs.

**What it implies for code and tables.** `tests/smoke/` is inside the
workstream folder; the tester's `tests/smoke` expectation at the root is met
by a root-level `tests/smoke/` stub only if needed (open). The Colab
bootstrap clones the public repo on the branch, replacing Drive copies.

**What it does not decide [open].** Whether a root-level `tests/smoke/`
redirect is needed for the tester; the PBS array layout of the table build
(block 6).

**Sources.** **[user]**, 2026-10-06 (AskUserQuestion answers). Hashes and
suite outputs **[run, sandbox 2026-10-06]**.

**Status:** active.

---

## D-023 -- Estimator settings of the per-node fit, and the in-focus blur as a sensitivity axis

**Date:** 2026-10-06, about 15:08 (second card). **Binds:** `config.py`
fields `mu_mode`, `bbar_region`, `focus_bg_rule`, `sigma_fit_um`,
`sigma_fit_study_um`; the configuration $\mathcal C$ of every bias table
(procedure §3.3); the per-node CSV columns `mu_hat_per_um`, `B_bar`,
`B_bar_region`; the sensitivity study of procedure §3.10. Closes D-019 (a)
and D-018 (a), (c).

**Decision [user].** (i) `mu_mode = "per_node"`: the absorption coefficient
$\mu$ is fitted at every node (stage 1 of the two-stage route of D-019 (a));
a shared $\mu$ is a labelled comparison with its own table. (ii)
`bbar_region = "block_masked"`: $\bar B_i$ is the median of the node's fetched
block in plane $k^*_i$ with the pixels within (Allen radius + 1 µm) of any
SWC segment in the block masked out; phantoms use the same mask around their
known axis. (iii) `focus_bg_rule = "profile_ends_median"`: the focus score
$F_{j,k}$ keeps handoff Eq. 1 as written (median of the profile ends,
$|v| > 1.5$ µm, in each plane). (iv) $\sigma_{\rm fit}$ is a **study axis**:
"I would like to study also the sensitivity of the estimated (with the
correction applied) diameters to this value. Thus calibration tables as well
as diameter estimations must be done on different values of the in-focus
blur."

**Statement.** For each fixed $\sigma_{\rm fit} \in \Sigma$, with
$\Sigma = \{0.080, 0.099, 0.125\}$ µm (provisional set, assistant), the
pipeline builds its own table $\hat b(d, \varphi \mid \mathcal C_\sigma)$ --
$\sigma_{\rm fit}$ is an element of $\mathcal C$ -- fits every real node with
that $\sigma_{\rm fit}$, and corrects it with its own table, giving
$\tilde d_i(\sigma_{\rm fit})$. The deliverable column `d_final_um` uses
$\sigma_{\rm fit} = 0.099$ µm **[assistant, provisional: the heuristic
budget, notes §2]** and the per-node CSV carries
$\max_\Sigma \tilde d_i - \min_\Sigma \tilde d_i$ as `d_tilde_sigma_spread_um`,
a systematic-error column. Procedure §3.10: the spread should be ≈ 0 if the
simulator is right.

**Why.** (i)-(iii) are the documents' own recommendations; (i) keeps the fit
a reparameterisation (same $\hat d_i$) while making $\hat\mu_i$ reportable;
(ii) follows the background across the field and removes the stained-fraction
bias of D-018 (b); (iii) is local to the line and costs nothing. (iv) The
true in-focus core is not identified by the stacks (procedure §3.4), so the
dependence of the corrected output on it is a systematic error that has to
be measured, not assumed away **[user]**.

**What it implies for code and tables.** `config.py` carries
`sigma_fit_um` (deliverable) and `sigma_fit_study_um` (tuple); the table
file name and the $\mathcal C$ dictionary carry $\sigma_{\rm fit}$; the
correction refuses a table whose $\sigma_{\rm fit}$ differs from the fit's;
`scripts/build_table.py --sigma-fit` and `scripts/run_cell.py --sigma-fit`
loop over the study set; the smoke suite asserts the defaults and the
refusal.

**What it does not decide [open].** The study set $\Sigma$ and the
deliverable value (assistant's provisional choices); whether $\sigma_r(0)$
of the renderer is varied in step with $\sigma_{\rm fit}$ (procedure §3.10
treats them as two separate checks).

**Sources.** **[user]**, 2026-10-06; D-018, D-019; procedure §3.3, §3.4,
§3.10; optics notes §2.

**Status:** active. Closes D-019 (a), D-018 (a) and (c).

---

## D-024 -- Renderer defaults, the dark-tube flag, a random phantom design over ranges, the tilt policy, D5/D7, node jitter

**Date:** 2026-10-06, about 15:10 and 15:12 (third and fourth cards).
**Binds:** `config.py` fields `absorption`, `kernel_continuation`,
`alpha_dark_flag`, `U_um`, `d_range_um`, `phi_range_deg`, `mu_range_per_um`,
`n_replicates`, `response_estimator`, `jitter_xy_um`, `jitter_z_um`,
`bias_flag_threshold`; the selection event $\mathcal S$; the phantom
generator (block 6); the table estimator and inversion (blocks 6-7); the
fill rules; the local decisions D5 and D7 of the design handoff.

**Decision [user].** (i) `absorption = "partition_vertical"` (procedure
Eqs. 5-6) and `kernel_continuation = "linear"` with $\gamma = 0.79$
(mathematics Eq. 17, Debye weighting), both **"for now"**: the user will
study the partition's failure for dark thick tubes and the kernel's far
field and may replace either; `linear` (mathematics Eq. 11), `ray_world`
(S6), `frozen`, $\gamma = 1.21$ and `proportional` stay selectable by name.
(ii) Dark-tube flag: a node with fitted centre-line absorbance
$\hat\alpha_i = \hat\mu_i \hat d_i / \cos\varphi_i > 1.0$ is flagged and
excluded from $\mathcal S$ (provisional threshold, to be set on the real
$\hat\alpha_i$ distribution). (iii) Phantom design: **a random search over
ranges, not a grid** ("has better statistics"): $U = 10$ µm ("up to 10 half
length is ok"); $d$ drawn on a temporary range until a script run in Colab
(and on the davinci archive) reports the distribution of Allen's annotated
radii, which sets $d_{\max}$; $\varphi$ over all tilts "without redundancy".
(iv) **Every branch and node gets its diameter estimated whatever its
tilt**: the 30° steep exclusion of the old tool (`STEEP_TAN = 0.58`) and of
the design handoff's flag list is withdrawn as a selection criterion. (v)
Response estimator: a smoothing thin-plate spline (`scipy.interpolate.RBFInterpolator`,
`kernel="thin_plate_spline"`, smoothing chosen by cross-validation) for
$\hat m(d, \varphi \mid \mathcal C)$; local kernel weights for $\hat\tau$ and
the failure rate. (vi) D5 (table correction by inversion, flag
$|\hat b - 1| > 0.2$, fill from same-branch neighbours or Allen's radius) and
D7 (start with 529878215, a handful of branches, compare with Allen's radii,
then L2/L3) confirmed. (vii) Node-position jitter (lateral and depth offset
of the phantom's "SWC" nodes from the true axis) is a configuration knob at
amplitude 0 until cell 13 measures $s^*$ and $\Delta z^*$.

**Statement.** One replicate draws $(d, \varphi, \mu, \theta, \xi)$ with
$\log d \sim U[\log d_{\min}, \log d_{\max}]$ ($d_{\min} = 0.2$ µm,
$d_{\max} = 4.0$ µm provisional), $\varphi \sim U[0°, 90°)$,
$\theta \sim U[0, \pi)$ (the $\pi$ symmetry of (S3) removes the redundant
half), $\log\mu \sim U[\log 0.3, \log 3.0]$ µm$^{-1}$ (provisional, until the
real $\hat\mu_i$ distribution is matched through the fitted statistic,
procedure §3.5), and the nuisances $\xi$ of procedure §3.5 (sub-pixel
offset, axis depth in $[-\Delta z/2, \Delta z/2]$, noise, jitter). Every
replicate is stored with its draw and its outputs ($\hat d_n$, $\hat\mu_n$,
$\hat v_{0,n}$, $\hat\varphi_n$, $k^*_n$, status, flags), so the regression
axes ($\log d$, $\varphi$; optionally $\log\mu$ as a third axis, procedure
§3.5) are chosen after the fact without re-rendering. $\hat m$ is the
spline fitted to $\hat d_n$ over the retained replicates; the inversion
solves $\hat m(d, \varphi_i \mid \mathcal C) = \hat d_i$ by `brentq` on the
spline. The selection $\mathcal S$ = converged, no parameter at a bound,
registration on branch (real data), not empty tissue, no second dip in the
window, not at the stack edge, $\hat\alpha \le 1.0$ -- with no tilt
criterion; applied identically to phantoms and real nodes.

**Why.** (i) The documents' own proposals, kept while the theory chat looks
for better; keeping the alternatives selectable is what D-021 asks. (ii) In
the ray world the partition fails between true $\mu d$ = 0.5 and 1.5
(implementation handoff, Findings (4)); the fitted $\hat\alpha$ is the
quantity that is large there, whether through darkness, thickness or tilt.
(iii) A random design has no empty cells and supports a regression with
its own cross-validated smoothing; it also makes $\mu$ (and later $\theta$)
free axes at no extra cost. (iv) **[user]**: "regardless phi EVERY branch
and node must have its diameter estimated." Where the method came from: the
old `allen_stack_radius_refit.py` flag `STEEP_TAN = 0.58`, the design
handoff's step 6 ("flag steep pieces") and Next action 5 ("flags must fire
for φ ≥ 30°"), and handoff Eq. 13(b)'s geometric onset near 22°; it was a
quality flag that would have filled steep nodes from neighbours. With the
table covering $[0°, 90°)$ the data-driven flags (failure rate, domain,
monotonicity, $|\hat b - 1|$, dark) decide, not the tilt. (v) The user's
choice over local linear regression and a GP. (vi)-(vii) **[user]**.

**What it implies for code and tables.** Consequence to be aware of
**[assistant, from the handoff's own numbers]**: with D5's threshold 0.2
and $\hat d/d$ = 1.27-1.33 at 45° and 1.42-1.46 at 60° for $d = 0.5$ µm
(implementation handoff, Findings), thin steep nodes will be flagged by
$|\hat b - 1| > 0.2$ and filled; the threshold is `bias_flag_threshold` in
`config.py` and the table build reports the share of the $(d, \varphi)$
domain under it, so it can be revisited. At $\cos\varphi_i \to 0$ the
measuring axis $\hat y_i$ of handoff Eq. 5 is undefined and the model's
$\alpha = \mu d / \cos\varphi_i$ diverges; the code takes the heading from
the neighbouring nodes for $\varphi_i > 85°$ (config) and flags `vertical`.
The script for the Allen radius distribution (`scripts/allen_radius_distribution.py`)
reads an SWC file or the archive's `specimen_*/reconstruction.swc` tree and
reports percentiles and the maximum of the dendritic radii by type; its
Colab cell fetches the SWC with `allen_image_io.fetch_swc`.

**What it does not decide [open].** $d_{\max}$ (after the radius script);
the $\mu$ range and whether $\mu$ is a regression axis (after real
$\hat\mu_i$); the dark-flag threshold on data; the number of replicates
(set by the cross-validated spline's SE at the budget davinci allows); the
jitter amplitudes (after cell 13); whether D5's 0.2 stays once the share of
the domain it flags is known.

**Sources.** **[user]**, 2026-10-06 (AskUserQuestion answers, four cards).
Implementation handoff "Findings" (numbers **[run]** there). Design handoff
§Code, §Method step 6, Next action 5, Eq. 13(b). Procedure §3.2, §3.5,
§3.8-3.10. `scipy.interpolate.RBFInterpolator`: SciPy 1.18 API, from the
installed documentation **[src]**.

**Status:** active. Confirms D5 and D7 of the design handoff; withdraws the
steep exclusion of its step 6 / Next action 5 as a selection rule (it
survives as a reported diagnostic column, `steep`). [updated 2026-10-26: (i)'s partition absorption, "for now", is to be replaced by the hybrid renderer once built -- **D-084**]

**Note added 2026-10-07 (merged from the repo additions of 2026-10-06); no change to D-024's text.** (i)'s configured slope $\gamma$ is to be replaced by a value measured on dendrites just under 0.8 µm -- **D-026** -- starting with flat branches -- **D-027**; the noise of the camera chain is to be measured on clean background patches -- **D-025**; a non-Gaussian kernel is idea **I-002**.

**Note added 2026-10-07 (later, 4); no change to D-024's text.** (ii) is superseded in part by **D-030**: a real node with $\hat\alpha_i > 1.0$ is no longer removed from $\mathcal S$ or filled; it is estimated and corrected like every other node, and $\hat\alpha \le 1.0$ leaves $\mathcal S$ for phantoms too. The threshold survives as a label.

**Note added 2026-10-18 (later, 4); no change to D-024's text.** (iii)'s design is superseded in part by **D-059** [user]: $G$ independent draws of $(d, \varphi, \theta, \mu)$ over the same laws, each realized $k$ times with fresh nuisances $\xi$, the per-draw estimate the mean ratio of its realizations in $\mathcal S$. The laws and ranges are unchanged.

**Note added 2026-10-20; no change to D-024's text.** (v)'s smoothing thin-plate spline, fitted through every realization in $\mathcal S$ (D-060), is a dense $(N + 4) \times (N + 4)$ solve; at the design size of **D-062** ($G \approx 10^5$ draws x $n_{\rm real}$ = 10, $N \approx 10^6$ rows) that is 8 TB of memory and about $2 \times 10^5$ h extrapolated from fits timed in the sandbox (`check_table_size.py`, exponent 3.4) [run]. The response estimator is therefore to be decided again before Block 14 (options in D-062; local kernel regression on a grid recommended); the local kernel statistics of (v) are unaffected.

**Note added 2026-10-20 (later); no change to D-024's text.** (v)'s response estimator is superseded by **D-063** [user]: local linear kernel regression on a grid with the bandwidths of D-054 (ii), the thin-plate spline kept only as a selectable small-table option; (vii)'s jitter knobs are set from the registration survey by **D-065.3**; (i)'s kernel question moves to D-066 (the SGT family in the renderer).

---

## D-025 -- The image noise of the stacks is measured on clean background patches, in every plane, at several positions and in several stacks

**Date:** 2026-10-06, 17:01. **Binds:** the noise step of the renderer's
camera chain (procedure §3.6 step 6) and the nuisance draw "noise" (procedure
§3.5); the noise field of `config.py` on `sci/diameter-pipeline`
(`RendererConfig.noise_sd_gl`, 3.0 grey levels, provisional); the
configuration $\mathcal C$ of every bias table, which lists the noise
(procedure §3.3) and must be rebuilt when it changes (§3.11); a Phase-II
measurement script (Colab, real stacks).

**Decision [user].** "We will measure the in slice noise, taking a neat patch
(without any branch crossing the patch across the z stack from the condenser
up to the current focus plane) on the focus plane and calculating the noise.
This will be done across all the real images planes, across images and
withint the same plane across several position. We calculate the second
moment of such noise and try to understand whether there are specific
ditributions used to describe such noise."

**Statement.** For a real stack $s$, a plane $k$ of it and a patch $P$ (a set
of pixels of plane $k$; size open), let $I_{s,k}(x, y)$ (grey levels) be the
recorded image, $m_{s,k,P}$ its mean over $P$, and
$$v_{s,k,P} = \frac{1}{\lvert P\rvert - 1}\sum_{(x,y) \in P}\bigl(I_{s,k}(x,y) - m_{s,k,P}\bigr)^2 \quad (\text{grey levels}^2)
\tag{D-025.1}$$
its second central moment. $v_{s,k,P}$ is a number computed from the data
(computed level); it estimates the variance of the noise of the camera chain
at the grey level $m_{s,k,P}$ (analytic level) only if the noise is
stationary within $P$ and nothing else varies there (comment (b)). It is
computed in every plane $k$, at several patches per plane and in several
stacks $s$; its spread over $(s, k, P)$ tests whether the noise is stationary
across depth, field position and specimen, and the pixel distribution within
the patches is compared with candidate families. A patch qualifies only if no
stained structure crosses its column (the user's criterion; comment (a)
extends it).

**Why.** The renderer's injected noise must reproduce the real background
statistics after the whole camera chain (procedure §3.6 step 6)
**[KB-repo]**. $b$ is an average, over nuisance draws that include the
noise, of a non-linear fit's output conditional on the selection
$\mathcal S$ (procedure Eq. 2), so the noise level can move $b$ itself, not
only its scatter **[KB-repo; reasoning]**. The real noise level is not
known, and with it the usable calibration range in planes (implementation
handoff, "Known gaps") **[KB-repo]**.

**The assistant's comments, recorded with the decision; not yet answered
[reasoning unless tagged].**

- (a) **Both sides of the focal plane.** Plane $k$ records the blurred shadow
  of every stained slab of the section, below and above it: procedure
  Eq. (6) sums over all slabs $j$ **[KB-repo]**. A clean patch therefore
  needs no stained structure at any depth within a lateral margin of about
  $3\sigma_{\rm r}(\delta)$ around it, $\delta$ being that structure's depth
  distance from plane $k$, on the objective side as well as on the condenser
  side.
- (b) **Detrend first.** Illumination shading and the haze of distant
  stained structure vary across a patch; a plane (or low-order polynomial)
  fitted per patch and removed before (D-025.1) keeps them out of $v$.
- (c) **Exposure noise and static structure.** For the same patch in
  neighbouring planes, $\tfrac12\operatorname{Var}_P(I_{s,k} - I_{s,k+1})$
  estimates the part of the noise that is independent between exposures;
  parts common to both planes (sensor fixed pattern, dust in the optics,
  slowly varying tissue texture, possibly JPEG errors on nearly identical
  blocks) cancel in it. The single-plane $v$ is what the fit sees; the
  difference between the two is the static part.
- (d) **Level dependence and candidate laws.** For raw camera data the
  standard description is signal-dependent: a Poisson part from photon
  counting plus a Gaussian part for the remaining stationary disturbances,
  the noise SD a function of the expected pixel value, with clipping at
  under- and over-exposure (Foi et al. 2008, **PubMed, abstract only; full
  text not accessible**); the same mixed Poisson-Gaussian model is used for
  light-sheet fluorescence images (Julia et al. 2024, **PubMed full text**).
  The Allen images have passed a grey-level mapping, 8-bit quantisation and
  JPEG, so that law is not transplanted: $v$ is plotted against $m$ and the
  relation is read off the data, which needs patches darker than the
  background (e.g. uniform interiors of broad out-of-focus shadows, after
  detrending). Uniform 8-bit quantisation alone adds about 1/12 grey
  level$^2$ when the noise SD exceeds about one grey level (textbook, from
  memory).
- (e) **Spatial correlation.** JPEG codes 8 x 8 pixel blocks (textbook, from
  memory), and the profile of handoff Eq. 9 is bilinearly interpolated and
  optionally averaged over ±0.5 µm along the branch **[KB-repo]**, so
  correlated noise does not average down as white noise does. The
  autocovariance at lags of 1-8 px in $x$ and $y$ is measured beside $v$.
- (f) **How it enters the renderer.** Either a parametric noise injected
  before the synthetic 8-bit and JPEG steps and tuned until the background
  statistics after the chain match the measured ones (procedure §3.6 step 6)
  **[KB-repo]**, or real detrended background patches added after the
  synthetic chain (distribution and correlation kept without choosing a
  family; exact only at the background level, and the synthetic JPEG then
  acts on a noise-free image).

**What it implies for code and tables.** A Phase-II script writes one row per
patch: stack, plane, position, size, $m$, $v$ (detrended), the lag-1 to
lag-8 autocovariances in $x$ and $y$, the plane-difference variance of (c),
and the clean-column criterion used. `config.py` gains a noise-model selector
whose parameters take that table as their source and which replaces the
provisional `noise_sd_gl` (names, e.g. `"gaussian_after_chain"` and
`"empirical_patches"`, to be fixed in `specs/SPEC.md` by the implementation
chat).

**What it does not decide [open].** The patch size and the clean-column
margin; whether the noise model depends on the grey level; the injection
route of (f); whether the noise differs between specimens (then it is an
element of $\mathcal C$ per specimen, procedure §3.11).

**Sources.** **[user]**, 2026-10-06 17:01. Procedure Eq. 2, §3.3, §3.5, §3.6,
§3.11; handoff Eq. 9; implementation handoff "Known gaps" (repo, D-020). Julia
A. et al. 2024, *Sensors* 24(7):2053 (PMID 38610265, PMC11014158), PubMed full
text, [DOI](https://doi.org/10.3390/s24072053). Foi A. et al. 2008, *IEEE
Trans Image Process* 17(10):1737-54 (PMID 18784024), PubMed abstract only,
[DOI](https://doi.org/10.1109/TIP.2008.2001399). Searches 2026-10-06: PubMed,
11 queries (Poisson-Gaussian noise in microscopy and in camera raw data;
photon transfer; sCMOS pixel noise; shot and read noise; the generalized
Anscombe transform; JPEG and lossy compression in quantitative microscopy;
brightfield noise models; brightfield defocus PSF and deconvolution of
biocytin-filled dendrites), 0-4 records each; bioRxiv, bioengineering, last 30
days (30 preprints): nothing relevant. JPEG block size and the quantisation
variance: textbook, from memory.

**Status:** active. Comments (a)-(f) await the user.

---

## D-026 -- Beyond ±0.84 µm the defocus kernel is calibrated on dendrites just under 0.8 µm, continuing the near-focus law; the kernel stays Gaussian for now

**Date:** 2026-10-06, 17:01. **Binds:** the kernel calibration (procedure
§3.4, Eq. 4; block 10 of the implementation handoff; `CalibrationConfig` on
`sci/diameter-pipeline`); `RendererConfig.kernel_continuation` and
`kernel_continuation_slope` ($\gamma$; D-024 (i): `linear`, 0.79, "for
now"); the choice of calibration nodes; the renderer's $\sigma_{\rm
r}(\delta)$ for $\lvert\delta\rvert > 0.84$ µm.

**Decision [user].** "I do not actually understand much the concern with
thick branches. It harden the estimation of the calibration table but these
are valuable for the calibration table values past the 0.8 um limit set by the
thin branches.. For the calibartion table we take a node, center it, find the
measuring direction, calculate the second moment and then run toward smaller
values of z across the slabs (with the same second moment calculation). A thin
branch (that has a diameter less than the z distance between slabs) casts a
shadow that has a good SNR till the third slab. Afterward the shadow is too
faint. However we have a great prior information to add when we estimate the
tables' values for far (>0.8 ) slabs from the thick dendrites. What we can do
is to find dendrites whose Allen diameter  is estimated to be slighlty less
than 0.80 so that we can do the analysis from the slabs that are further in
depth than 0.8um. We extrapolate the value from the law under 0.8 and then try
to adjust it (with other terms or whatever) Keeping for now the gaussian
kernel assumption." In this log's vocabulary the user's "slabs" are the
recorded planes ($\Delta z = 0.28$ µm), not the renderer's slabs (procedure
§3.6).

**Statement.** Two calibration sets enter procedure Eq. (4). The thin set
(procedure §3.4: $\hat d \lesssim 0.3$ µm, faint, $\varphi \lesssim 10°$,
isolated) gives $\Delta\sigma^2(\delta)$ for $\lvert\delta\rvert \le 0.84$
µm. The thick set -- nodes whose Allen diameter is just below 0.8 µm -- is
read on its planes with $\lvert z_{{\rm ax},i} - z_k\rvert > 0.84$ µm. Beyond
0.84 µm the kernel width follows, for each fixed $\delta$ with
$\lvert\delta\rvert > 0.84$ µm,
$$\sigma_{\rm r}(\delta) = \sigma_{\rm r}(0.84\ \mu{\rm m}) + \gamma\,\bigl(\lvert\delta\rvert - 0.84\ \mu{\rm m}\bigr),
\tag{D-026.1}$$
the continuation of D-024 (i), with $\gamma$ fitted on the thick set instead
of configured, and further terms added only if the residuals require them.
$K_\delta$ stays a circular Gaussian of width $\sigma_{\rm r}(\delta)$; the
alternative is I-002. Levels (R8): $\sigma_{\rm r}(\delta)$ and $\gamma$ are
the model's quantities; the calibration returns estimates
$\hat\sigma_{\rm r}(\delta)$ and $\hat\gamma$, computed from the real stacks.

**Why.** A thin node's shadow fades within about three planes (**[user]**;
implementation handoff "Findings", "Thin calibration nodes fade fast"
**[run]**). A thicker faint node removes more light: in the faint limit the
dip keeps its area from plane to plane (mathematics Eq. 13a), and that area is
$\bar B\mu\pi d^2/(4\cos\varphi)$ (D-019 (a)), while its depth falls as
$1/\sigma_{\rm r}$ (Eq. 13b); so it stays measurable further from focus
**[KB-repo; reasoning; run, comment (c)]**. D-024 (i) left $\gamma$
configured "for now".

**The assistant's comments, recorded with the decision; not yet answered
[reasoning unless tagged].**

- (a) **The statistic.** Procedure §3.4 [corrected 2026-10-04] calibrates with
  the squared Gaussian-core width of the dip, fitted within about one FWHM,
  and keeps the windowed second moment only as a cross-check: every real
  line-spread function has $\lvert v\rvert^{-2}$ tails at every defocus
  (mathematics §3.3), so its full second moment is infinite and a windowed one
  is dominated by the tails -- in focus, ideal Debye, a core width of 0.080 µm
  against $\sqrt{V_W}$ = 0.240 µm at $W$ = 1.5 µm and 0.348 µm at $W$ = 3 µm
  **[run, `psf_check.py`, `doc_checks.py` C5, as cited in procedure §3.4]**.
  "Second moment" in the decision is read as that core width; the reading is
  not confirmed by the user.
- (b) **Eq. (4) holds for the thick set as written; what the far planes
  identify.** Its per-node constant $c_i$ absorbs the node's own width. In
  the faint limit, and where $\sigma_{\rm r}$ is linear in $\delta$ across the
  node's depth, the depth extent of a flat round node ($r = d/2$) adds only
  the constant $\gamma^2 r^2/4$ to the dip's variance (its absorbance along
  depth has a semicircle profile, of variance $r^2/4$), which $c_i$ also
  absorbs -- exact for the second moment, approximate for the core width.
  Under (D-026.1), with $x_{i,k} = \lvert \hat z_{{\rm ax},i} - z_k\rvert -
  0.84$ µm on one side of the axis and $s_i$ the error of the node's estimated
  axis depth $\hat z_{{\rm ax},i}$, $\omega_{i,k}$ is, up to a per-node
  constant, $2\gamma\,(\sigma_{\rm r}(0.84\ \mu{\rm m}) + \gamma s_i)\,x_{i,k}
  + \gamma^2 x_{i,k}^2$: an axis-depth error acts exactly like a change of the
  anchor $\sigma_{\rm r}(0.84\ \mu{\rm m})$. With $z_{{\rm ax},i}$ free per
  node, only the curvature $\gamma^2$ identifies $\gamma$; with
  $z_{{\rm ax},i}$ taken from the node's own focus search (handoff Eqs. 1-2),
  or with planes on both sides of the axis (there $s_i$ enters with opposite
  signs and the two linear coefficients average to $2\gamma\sigma_{\rm
  r}(0.84\ \mu{\rm m})$, if $K_\delta = K_{-\delta}$), the linear term
  identifies it too. **[reasoning; both algebraic statements checked
  numerically, `checks/calib_reach_check.py` part 3]**
- (c) **How far it reaches** **[run, `checks/calib_reach_check.py`;
  illustrative ideal-Debye kernel, partition renderer, flat tube, planes on
  the light side, noise-free; reference computed with slope 0.79]**: a 0.8 µm
  node at $\mu = 0.6$ µm$^{-1}$ ($\mu d = 0.48$) keeps a centre dip at least
  as deep as a 0.3 µm node's at 3 planes out to 16 planes (4.48 µm) if the far
  slope is 0.79 and 11 planes (3.08 µm) if it is 1.21; at $\mu = 0.3$
  µm$^{-1}$, 9 and 7 planes. The run assumes a far law itself, so the numbers
  show the gain, not the range on real data.
- (d) **Windows and isolation grow with it.** At those distances
  $\sigma_{\rm r} \approx$ 2-3.5 µm (same run): the profile (±3 µm, handoff
  Eq. 9) and the block (about 10 µm, implementation handoff `block_half_um`)
  must widen to about $\pm 3\sigma_{\rm r}$, and the clearance from other
  stained structure with them. The background needs a trend term (D-025
  comment (b)); the masked block median of D-023 is biased low once the
  shadow is wider than the mask.
- (e) **Faint thick nodes only** ($\hat\alpha \lesssim 0.5$). Procedure §3.4
  step 2 tunes $\sigma_{\rm r}(\delta)$ by matching rendered phantoms to the
  real profiles, and the default renderer (partition along vertical rays,
  D-024 (i)) fails qualitatively for dark thick tubes: the in-focus centre
  lightens as the stain darkens, the synthetic focus curve peaks toward the
  light, and no phantom $\mu$ reaches the truth's $\hat\mu$ at $\mu d$ = 1.5
  or 3, while one does at $\mu d \le 0.5$ (implementation handoff,
  "Findings" (1), (2), (4) **[run]**). Dark nodes would also break the faint
  limit of (b).
- (f) **Both sides of focus.** Stepping only toward smaller $z$ measures one
  sign of $\delta$. $K_\delta = K_{-\delta}$ holds for an index-matched system
  free of spherical aberration (optics §3.5) and fails under index mismatch
  (optics §3.7); procedure §3.10 ("Above/below") fits Eq. (4) per sign to
  test it. Both sides also pin the axis depth, (b).
- (g) **A ladder of diameters** (about 0.3, 0.5 and 0.8 µm) gives overlapping
  $\delta$ ranges, so each class's curve is checked against the next where
  they overlap, not only joined at 0.84 µm. A node's near planes, where
  $\sigma_{\rm r}$ is not linear across its depth, do not satisfy the constant
  of (b).
- (h) **What it covers** **[run, same script]**. With D-024's phantoms
  ($U = 10$ µm, $\varphi \in [0°, 90°)$), the farthest slab from the plane
  through the node's axis is at $U\tan\varphi + r/\cos\varphi$ ((S5) of the
  implementation handoff). For $d = 0.5$ µm the thin set's 0.84 µm covers
  tilts up to 3.4°, and a calibrated range of 3.08-4.48 µm covers up to
  15.7-22.8°. Steeper phantoms, and the planes of the focus search, which
  reach $\lvert z_k - c_z\rvert$ further, keep relying on (D-026.1) beyond
  the measured range.

**What it implies for code and tables.** Block 10 takes a calibration-node
table with a diameter class per node and a per-class plane window, fits
Eq. (4) with per-node $c_i$ and, for the thick set, $z_{{\rm ax},i}$ from the
node's own focus search (comment (b)), and writes the tabulated
$\Delta\sigma^2$, its measured range and $\hat\gamma$.
`kernel_continuation_slope` changes status from configured to measured, and
the measured range enters $\mathcal C$; the renderer reads $\sigma_{\rm
r}(\delta)$ from the calibrated table where it exists and from (D-026.1)
beyond. A Phase-I oracle for block 10: synthetic thick-node scans rendered
with a known $\gamma$ return it.

**What it does not decide [open].** The thick set's selection
(Allen-diameter band, faintness threshold, isolation radius); its profile
window and block size; the form of the "other terms"; whether the two sets
are fitted jointly or the far law after the near one; the anchor of
(D-026.1) if the residual check of procedure §3.4 trims the thin set's range
to ±2 planes.

**Sources.** **[user]**, 2026-10-06 17:01. Procedure §3.4, §3.6, §3.10;
mathematics §3.3, Eqs. 13a-13b, Eq. 17; optics §3.5, §3.7; handoff Eqs. 1-2,
9; implementation handoff "Findings", (S5) and `block_half_um` (repo, D-020);
D-019 (a), D-023, D-024 (this log). Comments (c), (h):
`checks/calib_reach_check.py` with `calib_reach_check.out`, run 2026-10-06
**[run]**. Comment (b): **[reasoning]**, derivation as written, checked by
part 3 of the same script.

**Status:** active; replaces D-024 (i)'s configured $\gamma$ once the
calibration has run. Comments (a)-(h) await the user. **Refined 2026-10-06
18:12 by D-027:** both calibration sets start with flat nodes; tilted nodes
come later, as a test against the renderer.

**Note added 2026-10-20; no change to D-026's text.** By **D-061.4** [user] the calibration's nodes are selected by Allen's diameters, not by the fit's $\hat d$: thin branches (Allen's diameter small) for the near field up to 0.84 um, dendrites whose Allen diameter is about 0.8 um for the far field; by **D-061.5** the calibration runs in Colab and its kernel table is loaded into the davinci configuration.

**Note added 2026-10-20 (later); no change to D-026's text.** "The kernel stays Gaussian for now" is superseded by **D-066** [user]: the renderer's defocus kernel is the symmetric SGT family, $(w, \beta, \xi)$ tabulated against defocus and calibrated on the D-061.4 nodes in Colab; the continuation question of D-024 (i) moves to that table.

**Note added 2026-10-20 (later, 2); no change to D-026's text.** D-066 is withdrawn by **D-067** [user]: the kernel stays Gaussian and its $\sigma_{\rm r}(\delta)$ table is calibrated as this entry and D-027 plan, on the nodes of D-061.4 with D-067.4's thresholds.

**Note added 2026-10-21; no change to D-026's text.** Since S2.1 (`1aed5d6`) Block 10 selects this entry's far-field set by D-067.4 ($0.6 < 2 r_{\rm A} \le 0.8$ um; in $\mathcal S$, flat and faint by the pilot) and Colab's Cell 7 counts it, but its scan beyond 0.84 um and the fit of (D-026.1) are not built: the open points above (the plane range and side, the profile and block extent, the background, near-then-far or joint, the axis depth, further terms, the isolation radius) are restated in SPEC section 8 and were put to the user on 2026-10-21.

**Note added 2026-10-21 (later 4); no change to D-026's text.** **D-074** [user] sets the far scan's planes ($k^*_j \pm 16$, both sides, comment (f)) and its line (+-10 um, comment (d)); the background of comment (d) is reopened there: a line-end baseline has no samples past plane 9 on that line, and a planar background from far pixels is proposed. The checks run for D-074 confirm comment (b) on the renderer's model -- with both sides and the axis free the far law returns $\gamma$ within 0.03 % noise-free -- and add two facts it did not state: the per-node constant of the core width is negative (-0.01 to -0.07 um$^2$, not $\gamma^2 r^2/4 + r^2/4$), and the axis the fit finds sits 0.009-0.022 um from the geometric one, growing with $\mu d$ **[run, sandbox]**.

**Note added 2026-10-28; no change to D-026's text.** By **D-094** [user] the calibration nodes are selected without comment (e)'s optical-depth cut ($\hat\alpha \le 0.5$; `CalibrationConfig.node_alpha_check` restores it) and without the flag `faint` of $\mathcal S$. D-094's check (`checks/d094_2026-10-28/`) covers the near set only: the far (thick) set, for which (e) argued the cut, is not tested.
---

## D-027 -- The defocus calibration starts with flat branches only, near and far from focus; tilted branches come later, as a test

**Date:** 2026-10-06, 18:12. **Binds:** the choice of calibration nodes in
procedure §3.4 (thin set, "$\varphi \lesssim 10°$") and in D-026 (thick set,
no tilt criterion); block 10 of the implementation handoff
(`CalibrationConfig` on `sci/diameter-pipeline`); a later tilt study.

**Decision [user].** "For the calibration of the defocus second moments I
would use initially only branches that have zero tilt angle for near and far
off focus layers. The if possible construct the same but finding braches with
several different tilt angles and studying if the second moments change in
the profile"

**Statement.** The first calibration of $\Delta\sigma^2(\delta)$ -- the thin
set for $\lvert\delta\rvert \le 0.84$ µm and the thick set of D-026 beyond --
uses only nodes whose tilt $\hat\varphi_i$ from the local line fit (handoff
Eq. 5) is zero within a tolerance $\varphi_{\rm tol}$ (open). Afterwards,
where enough nodes exist, the same plane scans are repeated on nodes binned by
tilt, and their width curves are compared.

**Why.** At $\varphi = 0$ a node's profile in plane $k$ holds a single
defocus, up to the node's own depth extent, which is what Eq. (4) and D-026
comment (b) assume **[KB-repo; reasoning]**.

**The assistant's comments, recorded with the decision; not yet answered
[reasoning unless tagged].**

- (a) **What a tilt study can show.** A tilted branch's profile at a node also
  collects light from the neighbouring stretches of the branch, which lie at
  other depths, so the profile changes with $\varphi$ even when the kernel
  depends on depth only. In the model **[run, `checks/calib_width_check.py`
  part 2: thin node, depth-only Gaussian kernel, planes 1-9]** the square root
  of the second moment over $\pm 5\sigma_{\rm r}$ grows by 4-5 % at 10°,
  17-27 % at 20° and 32-59 % at 30°, while the Gaussian-core width changes by
  at most 1 %, 3 % and 6 %. The comparison is therefore between the tilted
  measurements and the renderer's prediction from the flat calibration
  (procedure §3.10, "Real vs synthetic profiles"), not between tilted and flat
  numbers; a mismatch then means that the kernel is not depth-only or that the
  renderer misses something (oblique illumination, the partition).
- (b) **The statistic.** The second moment's sensitivity in (a) is the
  mechanism of D-026 comment (a): broad, faint contributions dominate it, so it
  also moves with the window. For the same profile (plane 1, 30°) its ratio is
  1.32 over $\pm 0.61$ µm and 1.78 over $\pm 1.0$ µm, while the core ratio is
  1.054 over both **[run, part 3]**. The core width is the statistic for the
  tilt study as well.
- (c) **"Zero" needs a tolerance.** At 5° the core width changes by at most
  0.3 % (the second moment by 0.9-1.2 %) in the same run, so a tolerance of
  about 5° costs little with the core width; it must also exceed the error of
  $\hat\varphi_i$ from the line fit, which is not measured.
- (d) **How many nodes.** Flat thick dendrites (Allen diameter just under
  0.8 µm, faint, isolated, $\hat\varphi_i \approx 0$) may be few in one cell;
  their count decides whether more specimens are needed **[open]**.

**What it implies for code and tables.** `CalibrationConfig` gains a tilt
tolerance applied to both sets (e.g. `phi_tol_deg`, provisional 5°) and a
tilt-binned mode for the later study, whose output is labelled as a comparison
with the renderer's prediction.

**What it does not decide [open].** $\varphi_{\rm tol}$; the tilt bins; the
number of nodes per bin.

**Sources.** **[user]**, 2026-10-06 18:12. Procedure §3.4, §3.10; handoff
Eq. 5; D-026 (this file). Comments (a)-(c): `checks/calib_width_check.py` with
`calib_width_check.out`, run 2026-10-06 **[run]**. Searches 2026-10-06:
PubMed, 3 queries (the second moment of a PSF with heavy tails and windowing;
tilted-fibre defocus PSF calibration in brightfield; orientation bias of
neurite diameters), 0 records each.

**Status:** active. Refines D-026 (adds a tilt criterion to both calibration
sets) and procedure §3.4's thin-set criterion ($\varphi \lesssim 10°$ becomes
$\hat\varphi_i \approx 0$). Comments (a)-(d) await the user. [updated 2026-10-21: S2.1 (`1aed5d6`) applies `node_phi_max_deg` = 10° (procedure §3.4) to both sets, the pilot's $\varphi_j$, as D-067.4 names it; $\varphi_{\rm tol}$ stays open (SPEC section 8)]

---

## D-028 -- The sample autocorrelation is the textbook estimator, computed by statsmodels

**Date:** 2026-10-07, 09:41. **Binds:** the lag-$L$ sample autocorrelation
everywhere in `Ih Fit/`: `passive_fitting_hpc_fixed._estimate_noise`
(`noise_rho_lag1`, the `is_correlated` flag, the $n_{\rm eff}$ warning) and
`_estimate_residual_noise_at_mle` (`residuals_rho_lag1`, the AR(1) of Phase 3's
parametric bootstrap); `noise_calibration.lag_autocorr`, `_acf_diagnostic`,
`_ls_structure_diagnostic` and every noise-table column built on them
(`noise_rho_lag1`, `noise_ss_rho_lag1`, `acf_*`, `ar1_*`, `rho_ls_late`); the
Stage-7 manifest columns fed by that table; D-017's per-cell law fit, on the
data and on the simulations; smoke R10, R13, R14; eq. (3) of
`claude/TEEG_Stage7_noise_model_2026-09-28.md`. **Supersedes the estimator
convention** of (D-012.1)'s $r_k$, of D-014's "lag-$L$ analogue of (D-012.1)"
and of D-017's "lag-$L$ autocorrelation in the fitter's convention"; nothing
else in those entries changes.

**Decision [user].** "As stated in the instruction you have to use libraries
that implement the tools, algorithms pipelines etc. when available. Use the
text book approach." The sample autocorrelation is the textbook (biased)
estimator, both sums divided by the same $n$, and it is computed by a library
call, not by hand-written code.

**Statement.** For a window $x_1, \dots, x_n$ (mV) with
$\bar x = n^{-1}\sum_j x_j$, $u_j = x_j - \bar x$, and each lag
$L \in \{1, \dots, n-1\}$ (samples),
$$\hat\rho_{\rm tb}(L) = \frac{\sum_{j=1}^{n-L} u_j\,u_{j+L}}{\sum_{j=1}^{n} u_j^2} \in [-1, 1],
\tag{D-028.1}$$
computed as `statsmodels.tsa.stattools.acf(x, adjusted=False, nlags=L_max)[L]`
(`adjusted=False` is the default), through one wrapper imported by both the
fitter and the calibration. It is a number computed from one window; it
estimates the law's $\rho(L \mid \theta)$ with the demeaning bias of the noise
model document §3.4.2, plus the shrinkage below. The convention it replaces,
$\hat\rho_{\rm adj}(L) = \bigl(\sum_{j=1}^{n-L} u_j u_{j+L}/(n-L)\bigr)\big/\bigl(\sum_{j=1}^{n} u_j^2/n\bigr)$,
satisfies, for every window and every lag,
$$\hat\rho_{\rm tb}(L) = \frac{n-L}{n}\,\hat\rho_{\rm adj}(L),
\tag{D-028.2}$$
so medians over windows of equal $n$ convert exactly. The SD estimator (NumPy
`std`, ddof = 1) is unchanged.

**Why.** The project's first coding directive: established libraries for
anything they provide **[user]**. The hand-written function's docstring
justified it by saying statsmodels would not reproduce the fitter's lag-1
value; that is true of the default only -- `acf(..., adjusted=True)`
reproduces the old convention to $4 \times 10^{-16}$ -- so the code duplicated
a library option **[run]**. The textbook form is bounded and gives a
positive-semidefinite sequence (textbook, from memory). Under indirect
inference (D-017) the convention is carried on both sides of every comparison,
so either form is admissible there; the user chose the textbook one.

**What it implies for code and tables.** As listed in
`claude/TEEG_acf_textbook_change_plan_2026-10-07.md`: one wrapper (proposed
`noise_stats.sample_acf`) replaces the two hand-written lag-1 lines and
`lag_autocorr`; R10 gains a library oracle and an assertion that the default
is textbook; R10's AR(1) band, R13 and R14 are re-run; the regression gate is
unaffected (it compares loss values, `result.x` and `result.fun`, and no
$\rho$ enters the loss **[src]**); the L3_exc noise tables, any manifest built
from them, and the noise model document's tables and cohort fits are
recomputed, with the change plan's table of (D-028.2) as an exact check;
statsmodels becomes a cluster dependency. Every `rho_lag1` written before the
change -- run B's included -- is $\hat\rho_{\rm adj}(1)$ and stays labelled so.

**What it does not decide [open].** Whether `Biological Fit/` (the regression
oracle, D-006 (viii)) changes too; whether the adjusted form survives as a
labelled opt-in for reproducing old tables; whether the rule "fewer than 10
products gives NaN" stays (proposed: yes); where the wrapper lives; whether
statsmodels is in the cluster environment (not verified); the other
hand-written routines found in the same audit (two AR(1) recursion loops that
`scipy.signal.lfilter` implements; `locked_share`, which has no library
equivalent), which need their own decision.

**Sources.** **[user]**, 2026-10-07 09:41. Code read from `main` on
2026-10-07 **[src]**. statsmodels 0.15.0 checked by running it: `acf` against
(D-028.1) by hand, $3 \times 10^{-16}$; `adjusted=True` against
`lag_autocorr`, $4 \times 10^{-16}$; (D-028.2) to six decimals **[run]**.
Boundedness and positive semidefiniteness: textbook, from memory.

**Status:** active. ~~Not implemented~~ **[corrected 2026-10-07 (later, 2)]** implemented as a patch, pending
push and the cluster gates (note below) **[corrected 2026-10-16: the patch was never applied anywhere (user); `main` of 2026-10-16 has no `noise_stats.py` and `lag_autocorr` is still the pre-D-028 estimator (zip inspected); not implemented -- ships as the first block of the D-048 implementation]** **[updated 2026-10-18, restored 2026-10-19: implemented as Block 1 of the noise-fit work, commit `b4da926` on `sci/noise-fit`, patch delivered to the user; smoke N1-N6 6/6 and the three sandbox suites green; pending apply, push, statsmodels in `prova` and the cluster gates incl. the regression gate]**; the change plan is
`claude/TEEG_acf_textbook_change_plan_2026-10-07.md`. Supersedes the estimator
convention of D-012 ((D-012.1), $r_k$), D-014 (Statement, third bullet) and
D-017 (Statement), whose other content stands. Written at 09:47 as "D-025" and renumbered
the same morning (that ID was already taken in the repo's additions file). **Open points
closed by D-029.**

**Note added 2026-10-07 (later, 2), after the implementation; no decision changes.**
*What was built [run].* One patch on `main` at `b355d8e` (implementation chat, from the
repo handoff), for the user to commit and push. New `Ih Fit/noise_stats.py`:
`sample_acf(x, lags)` returns `statsmodels.tsa.stattools.acf(x, adjusted=False,
nlags=max(lags), fft=True)` at the requested lags, with the NaN rules of D-029 (iii).
The fitter's `_estimate_noise` and `_estimate_residual_noise_at_mle` and
`noise_calibration.lag_autocorr` call it; R10's Monte Carlo bands in
`smoke_ih_recovery.py` compute their statistics with it; `README.md`,
`CHANGELOG_ih_fit.md` and the handoff are updated. Three additions to the handoff's
text, none changing a value: a 2-D window is refused; an install whose `acf` lacks the
`adjusted` keyword is refused at import (statsmodels 0.11.1 names it `unbiased`; 0.12.0,
0.13.5 and 0.14.1 have `adjusted`, read from their PyPI wheels); the result is read
through `.acf` should a later statsmodels return its announced `AcfResult`.
*Checks [run].* New `smoke_noise_stats.py` 6/6: the library's direct path at 55 416 lags
(worst $6.7 \times 10^{-16}$) and hand sums; (D-028.2) to $5.0 \times 10^{-16}$; the NaN
rules without a warning; $|\hat\rho_{\rm tb}| \le 1$ and a positive-semidefinite Toeplitz
matrix on windows where the old form leaves $[-1, 1]$; the textbook value at all three
call sites; byte safety. Nine mutants, each caught. Existing gates unchanged in count:
smoke_ih_fit 12/12, smoke_ih_recovery 14/14, smoke_ls_baseline_qc 10/10, regression gate
PASS bit for bit (fun 0.0796305 before and after). The calibration CLI on R10's three
synthetic cells, old code against new: every autocorrelation column equals $(n-L)/n$ times
the old one (the AR(1)-implied columns $((n-1)/n)^L$ times), worst relative error
$1.7 \times 10^{-13}$; every other column identical.
*Consequences recorded.* statsmodels >= 0.12 is now required by every script in
`Ih Fit/` (the monolith imports `noise_stats` at load time), so it is installed in `prova`
before the code is pulled where a job may start from it. A window of exactly 10 samples
now gives NaN at lag 1; a flat residual pool gives NaN instead of 0.0. No loss function
reads either estimate **[src]**, so no fitted parameter moves. The other copies of the
monolith outside `Ih Fit/` keep the old estimator: D-028 binds `Ih Fit/` only.
*Pending.* The user's commit and push; statsmodels in `prova`; the cluster gates
(handoff section 5); then `noise_L3_v3.csv` against the change plan's section 3.5 table,
the Stage-7 manifests, and the noise model document's autocorrelation tables.
*Sources.* The patch, its gates and the end-to-end comparison **[run]**;
`CHANGELOG_ih_fit.md` and the handoff in the repo, once pushed;
`claude/TEEG_acf_textbook_change_plan_2026-10-07.md`, updated the same turn.

## D-029 -- Confirmations closing D-028's open points

**Date:** 2026-10-07, 09:55. **Binds:** the same objects as D-028, and
`Passive Features/HPC script/Biological Fit/` (the regression oracle of
D-006 (viii)), the new module `Ih Fit/noise_stats.py`, and the cluster
environment `prova`.

**Decision [user].** Answers to D-028's five open points: "1) No need 2) No
3) Yes 4) Where you propose 5) I do not think so".

**Statement.** (i) `Biological Fit/` is not changed: it keeps its
hand-written lag-1 estimator, $\hat\rho_{\rm adj}(1)$, and stays the
byte-untouched oracle of the regression gate, which compares loss values,
`result.x` and `result.fun` and no lag-1 coefficient **[src]**. (ii) The
adjusted form does not survive as an opt-in: (D-028.1) is the only
autocorrelation estimator in `Ih Fit/`; a table written before D-028 is
compared with a new one through (D-028.2), never by re-running old code.
(iii) `sample_acf` returns NaN at every lag $L$ with $n - L < 10$, as
`lag_autocorr` did; a flat window or a non-finite sample also gives NaN, as
before (the user's answer kept the 10-product rule; keeping the other two
NaN cases is the assistant's, to leave every column's NaN pattern unchanged).
(iv) The wrapper is `noise_stats.sample_acf(x, lags)` in
`Passive Features/HPC script/Ih Fit/noise_stats.py`, imported by
`passive_fitting_hpc_fixed.py` and `noise_calibration.py`. (v) statsmodels is
believed absent from `prova` **[user]**; it is installed there before the
first cluster run of the changed code, and its presence and the textbook
convention are checked on the login node before any job is submitted.

**Why.** (i) The oracle exists to be unchanged (D-006 (viii)); its noise
columns enter no comparison. (ii) One estimator, one name: an opt-in would
be a second definition behind a flag, and (D-028.2) already converts exactly.
(iii) Keeps the NaN pattern of every existing column. (iv)-(v) **[user]**.

**What it implies for code and tables.** The handoff
`Passive Features/HPC script/Ih Fit/docs/TEEG_acf_textbook_implementation_handoff_2026-10-07.md`
(repo, `main`) carries the blocks, the tests and the cluster steps. A
`noise_rho_lag1` from `Biological Fit/` (run A, run B) and one from `Ih Fit/`
after D-028 differ by the factor $(n-1)/n$ and are labelled by folder.

**What it does not decide [open].** The other hand-written routines of the
D-028 audit (the two AR(1) recursion loops; `locked_share`); whether
`Ih Fit/` enters the Coverage of the root `specs/SPEC.md`.

**Sources.** **[user]**, 2026-10-07 09:55. The regression gate's comparison
**[src]**, `regression_passive_identity.py` on `main` (589a0de).

**Status:** active. Closes D-028's open points. ~~Not implemented.~~ **[corrected 2026-10-07
(later, 2)]** (i)-(iv) implemented with D-028 (its note of the same date) **[corrected 2026-10-16: never applied anywhere (user); not in `main`; ships with the D-048 implementation]** **[updated 2026-10-18, restored 2026-10-19: (i)-(iv) implemented as Block 1, `b4da926`]**; (v) pending on
the cluster -- the install-and-check command is in the `Ih Fit/` README, section
"Environment".

---

## D-030 -- Every real node gets its own diameter estimate, whatever its optical depth

**Date:** 2026-10-07, 17:02 (Europe/Rome). **Binds:** D-024 (ii) and the
selection event $\mathcal S$ (D-024 statement); `config.py` field
`alpha_dark_flag`; the table build, inversion, flags and fill (blocks 6-7 of
the implementation handoff; procedure §3.9); the output columns `flags`,
`filled_from`, `d_final_um`.

**Decision [user].** "This should not happen. Regardless the optical depth we
estimate the diameter of ALL the real nodes." Said of D-024 (ii) as the theory
chat had restated it the same afternoon: a node whose fitted optical depth
$\hat\alpha_i = \hat\mu_i \hat d_i / \cos\varphi_i$ exceeds 1.0 is flagged,
removed from $\mathcal S$ and, like every flagged node, filled from
same-branch neighbours or Allen's radius (procedure §3.9, D5).

**Statement.** For every real node $i$, whatever $\hat\alpha_i$, the reported
diameter is the node's own fit $\hat d_i$ corrected through the table, by
solving $\hat m(d, \varphi_i \mid \mathcal C) = \hat d_i$ (D-024 (v)-(vi)). A
large $\hat\alpha_i$ is no longer a reason to remove a node from $\mathcal S$
or to fill it. Because $\mathcal S$ is applied identically to phantoms and
real nodes (D-024), the condition $\hat\alpha \le 1.0$ leaves $\mathcal S$ for
both, so the table is fitted on replicates of every optical depth and
describes the nodes it corrects **[assistant: required for consistency]**.
With D-024 (iv) (no tilt criterion), neither optical depth nor tilt removes a
real node from the estimate. Levels (R8): $\alpha = \mu d/\cos\varphi$ is the
node's optical depth along the vertical ray through its axis (model level);
$\hat\alpha_i$ is its estimate from the fit (computed level).

**Why.** **[user]**: every real node must have its own diameter; D-024 (iv)
said it for tilt, D-030 says it for optical depth. At fixed stain $\mu$,
$\alpha$ grows with $d$ and with $1/\cos\varphi$ ($\mu d = 0.5$ gives
$\alpha = 1$ at $\varphi = 60°$), so a fill rule on $\hat\alpha$ would replace
the measurements of thick and steep dendrites first **[reasoning]**.

**What it implies for code and tables.**
- `config.py`: $\hat\alpha \le 1.0$ is removed from $\mathcal S$, for phantoms
  and real nodes; `alpha_dark_flag` only writes a diagnostic column `dark` in
  the output, which no fill rule reads (proposal, as D-024 did for `steep`).
- Smoke check: a node with $\hat\alpha > 1.0$ is corrected, not filled
  (`filled_from` empty, `d_final_um` its corrected value), and carries
  `dark = True`; the table build keeps phantom replicates with
  $\hat\alpha > 1.0$.
- The table build reports its coverage in $\hat\alpha$ next to its
  $(d, \varphi)$ coverage.

**The assistant's comments, recorded with the decision; not yet answered
[reasoning unless tagged].**
- (a) **What a dark node's correction rests on.** Near $\alpha \approx 1$ the
  node shadows itself: on the central ray it removes $1 - e^{-\alpha} = 0.63$
  of the light at $\alpha = 1$, where the linear rule says 1, and its far side
  receives $e^{-\alpha} = 0.37$ **[textbook; arithmetic run 2026-10-07]**. The
  default renderer (partition, D-024 (i)) counts the light that reaches each
  slice along the vertical line instead of along the illumination's oblique
  rays; that is exact for one slice and to first order in faint stain
  (implementation handoff (S7)/(S8); mathematics §3.2), and it fails for dark
  thick tubes: against the ray world no phantom $\mu$ reaches the truth's
  $\hat\mu$ at $\mu d$ = 1.5 or 3, while one does at $\mu d \le 0.5$
  (implementation handoff, Findings (4)) **[run]**. The incoherent image form
  $B(T * h_0)$ holds only to first order in absorbance (optics §3.6, textbook
  from memory there). The table corrects a dark node only as well as the
  renderer renders it, and $\hat\tau$ does not contain that model error; the
  D-024 (i) study of the partition's failure, with the `ray_world` option (S6)
  as the comparison for the treatment of absorption, is what would make dark
  nodes' corrected diameters trustworthy.
- (b) **$\mu$ as a regression axis.** With dark nodes corrected, the fit's
  bias at fixed $(d, \varphi)$ changes with darkness (the shadow saturates and
  the absorption concentrates on the side the light enters), so a table that
  averages over the phantoms' $\mu$ gives a dark node the correction of an
  average one. D-024's optional third axis ($\log\mu$, or $\hat\alpha$) is the
  remedy, to be decided after the real $\hat\mu_i$ (D-024 open point).
- (c) **The threshold labels; it no longer selects.** Its value (1.0,
  provisional) can be set on the real $\hat\alpha_i$ distribution without
  changing any estimate.

**What it does not decide [open].** Whether $\log\mu$ or $\hat\alpha$ becomes
a regression axis; whether the renderer for dark nodes changes (D-024 (i)
study); the other flags and fill rules of D5 ($|\hat b - 1| > 0.2$,
convergence, crossings, stack edge), which this entry does not touch; how dark
nodes are shown in summaries.

**Sources.** **[user]**, 2026-10-07 17:02. D-024 (this log); procedure §3.9;
mathematics §3.2; optics §3.6; implementation handoff (S6)-(S9), Findings
(1)-(4) and the dark-tube flag row (repo, D-020); arithmetic run 2026-10-07.
Searches 2026-10-07: PubMed, 5 queries (weak-object approximation in
bright-field; partially coherent imaging of absorbing specimens, twice;
transmission cross-coefficient; depth-variant PSF), returning 8, 0, 0, 1 and 3
records; the weak-object and coherence hits are electron-microscopy analogues
(abstract only), the PSF hits widefield fluorescence; no bright-field source
on dark absorbers. bioRxiv: no keyword search; a scan of 30 recent biophysics
preprints found nothing relevant.

**Status:** active. Supersedes D-024 (ii)'s removal of nodes with
$\hat\alpha > 1.0$ from $\mathcal S$ and the fill that followed from it; the
threshold survives as a label (comment (c)). Comments (a)-(c) await the user. [updated 2026-10-21, later 2: Block 14.1 (`89ca78a`, local): `dark` is no criterion of $\mathcal S$, for phantoms and real nodes (`phantoms.SELECTION_FLAGS`); a dark node is inverted and keeps the label, and Block 14.5's `correct_nodes` inverts it (checked) **[run, sandbox]**] [updated 2026-10-28: the same rule now holds for the dip -- **D-096** [user]: `faint` (a dip under 6 gl) is a label, not a criterion of $\mathcal S$, for phantoms and real nodes.]

---

## D-031 -- The bias table also accounts for the estimated optical depth: $\hat\alpha$ is its third axis

**Date:** 2026-10-07, 18:01 (Europe/Rome). **Binds:** the bias table's
regression axes (D-024 statement: "$\log d$, $\varphi$; optionally $\log\mu$ as
a third axis"); the response estimator (D-024 (v)); the inversion (D-024 (vi),
D5); `config.py` (the regression-axes field); blocks 6-7 of the
implementation handoff.

**Decision [user].** "I agree that the correction tables should also account
for the estimated optical depth." Answer to D-030 comment (b), which proposed
$\log\mu$ or $\hat\alpha$ as a third axis.

**Statement.** The response surface becomes
$\hat m(d, \varphi, \hat\alpha \mid \mathcal C)$, and the local spread
$\hat\tau$ and the failure rate are likewise indexed by $\hat\alpha$. It is
fitted over the phantom replicates in $\mathcal S$, each with its own
$\hat\alpha_n$ computed from its fit exactly as for real nodes (D-019.1). A
real node $i$ is corrected by solving
$\hat m(d, \varphi_i, \hat\alpha_i \mid \mathcal C) = \hat d_i$ for $d$ at its
own $\hat\alpha_i$. Levels (R8): the axis is the computed-level $\hat\alpha$,
which phantoms and real nodes both carry; no true $\alpha$ or $\mu$ is plugged
in.

**Why.** **[user]**. At fixed $(d, \varphi)$ the fit's bias changes with
darkness (the shadow saturates and the absorption concentrates on the side the
light enters), so a table that averages over the phantoms' $\mu$ would give a
dark node the correction of an average one (D-030 comment (b))
**[reasoning]**.

**What it implies for code and tables.**
- `config.py`: the regression axes become $(\log d, \varphi, \hat\alpha)$
  (names fixed in `specs/SPEC.md`); the thin-plate spline
  (`scipy.interpolate.RBFInterpolator`, which accepts three-dimensional inputs:
  checked with SciPy 1.18.1 in the sandbox, 2026-10-07 **[run]**) is fitted on
  axes scaled to comparable ranges **[reasoning]**.
- The phantom design must make $\hat\alpha_n$ cover the real $\hat\alpha_i$;
  the table reports its $\hat\alpha$ coverage (D-030), and a real node outside
  it is out of domain (D5's domain flag).
- More replicates for the same SE, since the spline has three axes (budget
  open, D-024).

**The assistant's comments, recorded with the decision; not yet answered
[reasoning].**
- (a) $\hat\alpha_i$ comes from the same fit as $\hat d_i$, so the correction
  is conditional on the node's $\hat\alpha_i$, held fixed while $d$ is solved
  for: a correction at given estimated optical depth, not a joint inversion
  for $(d, \mu)$.
- (b) This settles the axis half of D-024's open point "the $\mu$ range and
  whether $\mu$ is a regression axis"; the $\mu$ range stays open, since it
  sets the $\hat\alpha$ coverage.

**What it does not decide [open].** The axis scaling; $\hat\alpha$ or
$\log\hat\alpha$ as the coordinate; the replicate budget; how far outside the
phantoms' $\hat\alpha$ range a node may still be corrected.

**Sources.** **[user]**, 2026-10-07 18:01. D-019.1, D-024, D-030 (this log);
procedure §3.8 (Eq. 7) and §3.9 (repo, D-020); the SciPy check above
**[run]**.

**Status:** active. Settles D-030 comment (b); amends D-024's regression axes
($\hat\alpha$ added as the third axis, not $\log\mu$).

---

## D-032 -- Tilt-specific tables are dropped, as kernel and as test

**Date:** 2026-10-07, 18:01 (Europe/Rome). **Binds:** the per-tilt tables
discussed in this chat on 2026-10-07 (per-tilt kernel tables, or per-tilt
calibration tables used as a test of the depth-only kernel); D-027's later
tilt study; the tilt-binned mode of `CalibrationConfig` (implementation
handoff, D-027 row).

**Decision [user].** "Tilt-specific tables. Do not use it as a test. Just drop
this". Answer to the theory chat's advice of 2026-10-07 ("good as a test;
wrong as the kernel you render with").

**Statement.** No table of the defocus kernel or of calibration widths is
built per tilt bin, neither as the renderer's kernel nor as a test of the
depth-only kernel. The kernel stays one depth-only table (D-026, D-027), and
tilt enters only through the renderer's geometry. **[assistant's reading, to
be confirmed by the user]** This withdraws D-027's second half ("afterwards
... the same plane scans are repeated on nodes binned by tilt, and their
width curves are compared") and the tilt-binned mode of `CalibrationConfig`;
D-027's first half (calibrate on flat nodes, near and far from focus) stands.

**Why.** **[user]**. The argument against the kernel use stands on its own: a
kernel read off tilted branches already contains the light of the
neighbouring stretches, which the renderer adds again **[reasoning; run,
`checks/calib_width_check.py` part 2]**.

**What it implies for code and tables.** `CalibrationConfig` keeps the tilt
tolerance of D-027 (`phi_tol_deg`) and drops the tilt-binned mode.

**What it does not decide [open].** Whether any other check of the renderer's
tilt geometry against real tilted nodes is wanted.

**Sources.** **[user]**, 2026-10-07 18:01; D-027 (this log); this chat,
2026-10-07.

**Status:** active. Supersedes D-027's second half (the tilt-binned study),
on the reading above.

## D-033 -- The per-cell noise law is fitted by our own indirect inference, not by ABC

**Date:** 2026-10-08, 09:56. **Binds:** D-017's open point (d) (the per-cell
fit's statistic set, weights and estimator); the per-cell law fit planned in
`Ih Fit/noise_calibration.py`; any use of abcTau (Zeraati, Engel & Levina
2022) in `Ih Fit/`.

**Decision [user].** "Let's keep our approach." Said after the assistant
explained, on the user's question why Zeraati et al.'s method is not used,
that the law (D-017.2) already is their generative model and that what we do
not use is their adaptive-ABC inference (package abcTau).

**Statement.** For each real cell $c$, $\theta_{\rm noise}(c) =
(\sigma_f, \rho_f, \sigma_s, \tau_s)(c)$ is a point estimate: the minimiser
of the scaled squared distance between $c$'s measured window statistics and
the same statistics under law $\theta$ (indirect inference, eq. (12) of
`claude/TEEG_Stage7_noise_model_2026-09-28.md`). No approximate posterior is
sampled, and abcTau is not a dependency of `Ih Fit/`. The statistics are
computed with the pipeline's own estimators on data and model alike (the
autocorrelation through `noise_stats.sample_acf`, D-028).

**Why.** **[user]**. Recorded with it **[reasoning]**: abcTau would need a
custom generator (fast AR(1), slow OU, offset, tilt) and a custom summary
statistic (SDs and autocorrelations on four windows from two protocols,
medians over sweeps) before it could run on our model, and it is not on PyPI;
the generator of the twins needs one law per cell, which a point estimate
gives.

**What it implies for code and tables.** The per-cell fit is a least-squares
minimisation over $\theta$ in `noise_calibration.py` (or a module beside it),
with the expected statistics from the closed form or from simulation through
the pipeline's own estimators (still open, below). Its output columns carry
point values only.

**What it does not decide [open].** Within (d): closed-form expectations or
simulated medians (the 2026-09-28 note to D-017 leans to simulation, the
closed form as a starting point only); the statistic set and the scales
$\omega_m$. What a point estimate gives up, recorded so it is not forgotten:
no interval on $\tau_s$, which is loosely pinned from above (D-017 (e)), and
no formal test of one timescale against two. D-017's other open points.

**Sources.** **[user]**, 2026-10-08 09:56. Noise model document §3.5.1,
§3.5.5 **[KB]** (Zeraati et al. 2022 as read there in full text; abcTau
repository README at commit `72343a8` **[src]** as recorded there).

**Status:** active. Settles the estimator-family half of D-017 (d); the rest
of (d) stays open. **[updated 2026-10-16: the weights by D-048/D-049/D-050, the statistic set and simulated medians by D-051; nothing of (d) stays open.]**

---

## D-034 -- The variance at seconds is carried by the per-sweep tilt; SS noise windows are enlarged where the pulse transient has died out

**Date:** 2026-10-08, 11:19. **Binds:** D-017 (c) and (e); the tilt $b$ of law
(D-017.1) and the generator's `drift_sigma_mV` ($\sigma_\delta$); the SS noise window
of `noise_calibration.py`; D-033's statistic set.

**Decision [user].** "About the above let's drop my approach and let's use the
per-sweep tilt. However still used the enlarged window for the SS (only if the
successive windows are not influenced by the short transient". (i) The user's
proposal of the same day -- per-sweep fits of a sum of OU processes, their number
chosen by the elbow of a BIC curve -- is dropped. The law stays (D-017.1): fast
AR(1) + one slow OU + per-sweep offset $a$ and tilt $b$, no third OU; the variance
at seconds that windows of at most 265 ms miss (D-017 note 4) is carried by $b$.
(ii) The SS noise window is enlarged from the 10 ms before a pulse to the stretch
after the previous pulse, only where that pulse's response no longer reaches.

**Statement.** For each real cell $c$ and each generated trace of duration
$T_{\rm sw}$ (s), $b \sim \mathcal N(0, \sigma_b(c)^2)$ (mV/s), so
$\sigma_\delta = \sigma_b(c)\,T_{\rm sw}$ (mV). $\sigma_b(c)$ is estimated from
$\hat V_c(h)$, half the mean squared difference of two SS window means $h$ apart in
one sweep (computed level; $h$ a multiple of 0.2 s [corrected 2026-10-20: of
the onset period 0.2005 s; same polarity: even multiples; $h^*$ = 0.8 s is 4
periods, 0.802 s; data]):
$$\hat\sigma_b(c)^2 = 2\,\bigl[\hat V_c(h^*) - V(h^* \mid \hat\theta_c)\bigr]/h^{*2}, \tag{D-034.1}$$
with $V(h \mid \theta)$ the same quantity under the stationary part of (D-017.1)
(analytic level, closed form), evaluated at the cell's fitted law $\hat\theta_c$,
and $h^*$ the calibration lag (open). The mean of a line over a window is its value
at the window's centre, so the tilt adds exactly $\tfrac12\sigma_b^2h^2$. The SS
window before pulse $k+1$ is $[t_0(c), 200)$ ms after pulse $k$ -- the late part of
pulse $k$'s stored $(-10, +200)$ ms window -- with $t_0(c)$ the earliest start that
passes the transient test (comment (c)).

**Why.** **[user]**. Recorded with it **[run: this session's checks A-D and C2,
`verify_msg29.py`; cohort law $\hat\theta_{\rm sim}$]**: (1) A tilt's semivariogram
grows as $h^2$, a seconds-scale OU's at most as $h$, so a tilt matches the variance
at seconds at one lag only (D-017 note 4 rejected a continuing creep: signs 53/104).
Matched to the cohort's 6.89 s change (median 0.128 mV): $\sigma_b$ = 0.027 mV/s,
$\sigma_\delta$ = 0.20 mV over a 7.27 s LS sweep, and the step-mean baseline error
has SD 0.017 mV, against 0.049-0.070 mV under a seconds-scale OU matched to the same
change. Matched to that OU at the step instead: $\sigma_b$ = 0.077-0.110 mV/s,
predicting a 6.89 s median of 0.36-0.51 mV; the tilt's point error is then 0.43 of
the OU's (3 s) at the step onset and 1.14 at its end. (2) Per-cell SD of the median
textbook $\hat\rho$(10 ms): 0.08 from 5 LS pre-windows; 0.030-0.034 from 2 SS sweeps
and 0.009-0.012 from 17, for $t_0$ between 10 and 100 ms (response assumed removed
exactly). With 17 sweeps, $\hat V_c$(1 s) under seconds-scale OUs of 3 and 30 s
spans 0.0052-0.0072 and 0.0059-0.0085 mV$^2$ (10-90 %): the variance at seconds is
identified per cell, its shape only weakly.

**The assistant's comments, not yet answered [reasoning unless tagged].**
(a) **Calibration lag:** the step's lags, $h^*$ in 0.14-1.14 s from the pre-window
centre (step mean at 0.64 s), where the LS loss reads the baseline; the post-step
stretch, read only by the report-only rebound residuals, then carries too much
drift. Calibrating at 6.89 s instead makes the step's baseline error 3-4 times too
small. (b) $\sigma_\delta$ follows each generated trace's own $T_{\rm sw}$; using an
SS-calibrated $\sigma_b$ for LS sweeps rests on D-017 (b). (c) **Transient test**,
per cell: $\bar d(t) = \tfrac12[\bar r_{\rm dep}(t) - \bar r_{\rm hyp}(t)]$, the odd
part of the polarity-mean responses (each window minus its pre-pulse mean), is the
pulse response; $t_0$ is the earliest time after which $|\bar d|$ in 5 ms bins stays
below $\kappa\hat\sigma_{\rm SS}$ ($\kappa$ open, proposal 0.1), with the
across-pulse residual variance flat from $t_0$. Format A of `ss_pulses.npz` stores
no pulse times **[src]**: two stored windows are consecutive only if their 10 ms
overlap is identical within one sweep number. (d) $\sigma_a$ needs no calibration:
$a$ cancels in every residual and every window statistic **[src; topic doc
§3.6.2]**.

**What it does not decide [open].** $h^*$; $\kappa$ and the bin; whether the
enlarged windows' own statistics join D-033's statistic set (proposal: yes); D-017
(a), (b), the rest of (d); (e) beyond what the SS series bounds.

**Sources.** **[user]**, 2026-10-08 11:19. D-017 note 4 **[run, cluster]**; the
archive writer's `_extract_windows_around_pulses` and Format A **[src]**; SS
protocol, Allen white paper **[KB]**; the checks above **[run]**, in the assistant's
sandbox, not in the repo.

**Status:** active. Settles D-017 (c) except $h^*$; calibrating $b$ from pre-window
slopes stays excluded (D-017 note 4). **[updated 2026-10-16: the enlarged windows' statistics join D-033's set (D-051 (i)); the pulse response is subtracted per polarity before the statistics (D-051 (iv)); $h^*$ = 0.8 s, $\kappa_{\rm tr}$ = 0.1 and 5 ms bins are labelled provisional config defaults until S2 (D-051).]** **[updated 2026-10-28: comment (c)'s transient test -- the 5 ms bins of $\bar d$, the odd part -- is replaced by **D-095**: the start is read from the exponential decay of the mean windows per polarity and per sweep; $\kappa_{\rm tr}$ stays provisional.]** **[updated 2026-10-28 (later, 7): for the 14 cells with two SS sweeps the twins' tilt SD is taken both as (D-034.1) (variant (a)) and as one cohort value (variant (b)) -- **D-100**; the multi-sweep cells keep (D-034.1).]** **[updated 2026-10-28 (later, 9): the twins no longer draw $b$ from $\mathcal N(0, \sigma_b(c)^2)$: each twin sweep carries the realised tilt of its own real sweep, and no $\sigma_b(c)$ is computed for them -- **D-102**; the (later, 7) note's D-100 is withdrawn. The law -- the variance at seconds carried by a per-sweep tilt -- stands.]**

**Note added 2026-10-08 (later, 2), from the topic doc's v3 computations; no decision changes.**
Under the textbook cohort law $\hat\theta_{\rm sim}$ = (0.0383 mV, 0.179, 0.0547 mV, 39.7 ms),
with exact point moments **[run]**. *Why (2), last sentence, corrected:* its 30 s band was for an
OU with the 3 s one's $\sigma^2/\tau$, not sized to the 6.89 s change. Sized to it
($\sigma$ = 0.294 mV), the 17-sweep 10-90 % band of $\hat V_c$(1 s) is 0.0030-0.0043 mV$^2$
against 0.0051-0.0077 for 3 s, so with many sweeps the semivariogram resolves the timescale; with
2 sweeps the bands overlap. *Why (1) placed:* its numbers match the tilt to the step-mean error;
the LS loss reads the step-averaged squared error, matched by $\sigma_b$ = 0.083-0.119 mV/s (point
error 0.47-0.49 of the OU's at onset, 1.20-1.23 at the end; 6.89 s median 0.39-0.56 mV).
*Recommendation, not decided:* $h^*$ = 0.8 s with 190 ms windows, within 3 % of that match for
OUs of 3-30 s (0.6 s: 11-13 % more; 1.0 s: 12-13 % less). *Source:* topic doc v3 §3.6.4-3.6.5.

**Note added 2026-10-26, from the checks of the tilt study document (`TEEG_tilt_sigma_b_2026-10-26.md`, sent to the user, not in
project knowledge; standalone scripts check01-check05 on the Block 2-4 code); no decision changes.** *The lag, recomputed for the
windows in use* (D-076: $t_{\rm SS} \ge$ 80 ms, so 120 ms windows): with the variance at seconds taken as an OU of 3, 10 or 30 s, the
tilt read by (D-034.1) at 4 periods (0.802 s) is 0.987-0.989 of the tilt matched to the Long Square step's step-averaged baseline
error (0.969-0.974 with 190 ms windows; the numbers of the note above reproduced); the two coincide at 0.782-0.783 s; 2 and 6 periods
would give +37-40 % and -19-21 % **[run; formulas checked against `noise_law.semivariogram_continuous` and quadrature]**. A two-lag
form (4 minus 2 periods), free of $\hat\theta_c$, is as precise under a pure tilt but reads 0.80-0.83 of the match under such an OU
**[run]**. *Precision:* given the realised tilts, (D-034.1) estimates their pair-weighted mean square, so a 2-sweep cell's estimate rests
on two draws: 95 % range of $\hat\sigma_b/\sigma_b$ 0.00-1.96/2.04 (2 sweeps), 0.41-1.59 (6), 0.55-1.38 (10) at $\theta_{\rm cf}$ and
$\sigma_b$ = 0.1 mV/s, $\hat\sigma_b^2$ unbiased; per-sweep least-squares slopes recover each realised tilt to 3-4 % of $\sigma_b$, so
the spread is the Cramer-Rao limit of two draws, which (D-034.1) approaches within 9-30 % in variance; below about 0.04-0.05 mV/s a
2-sweep cell cannot tell $\sigma_b(c)$ from zero **[run]**. *Options for the user (SPEC §8), compared on an 18-cell cohort with the
L3_exc sweep counts:* per cell; one cohort value; partial pooling of $\log\hat\sigma_b(c)^2$ by DerSimonian-Laird (within-cell variances
trigamma$(S_{\rm eff}/2)$ from the layout, between-cell variance clipped at 0). With the log-SD $\omega$ of $\sigma_b(c)$ across cells at
0 / 0.25 / 0.5 / 1, the share of 2-sweep cells whose tilt SD is off by more than a factor 2 is 25 / 25 / 26 / 31 % per cell,
0 / 1 / 22 / 59 % cohort and 0 / 1 / 12 / 27 % pooled; pooling never returns zero (per cell 5-10 % zeros); by the RMS relative error,
which weighs overestimates more, per cell wins once $\omega \ge$ 0.5 **[run]**. The LS tail's per-cell medians reject one common variance
at seconds ($P$ = 1e-5 for their max/median under homogeneity) and suggest $\omega \approx$ 0.5 (0.2-0.9) **[run; from the three order
statistics of D-017 note 4 only]**. *Recommendation, not decided:* the pooled form. *Sources:* the study document §3.6-3.9 and its
scripts **[run]**; DerSimonian & Laird 2015 and Veroniki et al. 2016 **[PubMed FT]**.

---

## D-035 -- The sharpest plane $k^*$ is the plane of maximum gradient energy of the profile (logged late; the repository calls it "D-030")

**Date:** decided 2026-10-07 (afternoon; implemented in commit `597dd28`, 15:43), logged 2026-10-08. **Binds:** the focus
rule of the diameter fit (procedure Eq. 1, handoff Eq. 1); `MeasureConfig.focus_rule`, `focus_grad_window`,
`focus_grad_margin_um` on `sci/diameter-pipeline`; `analysis/focus.py`; the pilot's columns `k_star`, `k_star_depth`
and `focus_rule`; the default estimator hash (6f0397236447fe6f since then).

**Decision [user].** "Write the code to substitute the former method with the gradient energy and change also the
Colab cell if needed" (2026-10-07), after the diameter chat had compared the dip depth with sharpness measures of the
profile.

**Statement.** For node $j$ and plane $k$, with $\tilde I_{j,k}$ the profile along the node's measuring line smoothed by
a Gaussian of one sample and $B_{j,k}$ the median of the profile's ends (D-023):
$G_{j,k} = B_{j,k}^{-2}\int_{|v| \le h_j} (\partial \tilde I_{j,k}/\partial v)^2\,dv$ [1/um], with
$h_j = r_j + 0.5$ um and $r_j$ the node's radius (Allen's on real nodes, the true one on phantoms); the window rule and
its margin are PROVISIONAL (the assistant's). $k^*_j = \arg\max_k G_{j,k}$. The dip depth of handoff Eq. 1,
$F_{j,k} = -\ln(\min_v \tilde I_{j,k} / B_{j,k})$, is kept as the labelled comparison (`k_star_depth`).

**ID collision [assistant].** The repository cites this decision as "D-030" (config comments, `focus.py`, SPEC,
runbook, notebook, tests: about 65 mentions on 2026-10-08). This log's D-030, written at 17:02 the same day by another
chat, is a different decision (every real node is estimated whatever its optical depth). Until the repository's
references are renamed, its "D-030" means this entry.

**Why.** **[user; run, sandbox]** On rendered dark tubes the dip depth is pulled one or more planes toward the light
(the partition's focus shift), while the gradient energy over a window holding the edges finds the tube's own plane
(`test_smoke_focus.py`).

**What it implies for code and tables.** $k^*$ by $G$; $F$ recorded beside it; Cell 4c shows both curves; the
estimator hash changed, so tables and pilot rows made before it are refused.

**What it does not decide [open].** The window on thick trunks: there Allen's radius is far below the half-width of the
dendrite's edges (nodes 2 and 3 of specimen 529878215: $r$ 1.02-1.10 um, edges near $\pm 2.5$ um), the window holds no
edge, and $G$ picks at random. On a rendered faint 5 um tube $G$ fell within one plane of the centre in 1 of 8 noise
draws, with the window from the true radius in 4 of 8, and with that window and the profile averaged $\pm 2$ um along
the branch in 7 of 8; on such a tube the fitted diameter stays between 1.03 and 1.11 times the true one over $\pm 4$
planes **[run, sandbox, 2026-10-08]**. Whether to size the window from the measured profile is open.

**Sources.** **[user]** 2026-10-07; repository `sci/diameter-pipeline` (`597dd28`, later `d6e6695`) **[src]**; the
synthetic demo of 2026-10-08 **[run]**.

**Status:** active (decided 2026-10-07, logged 2026-10-08).

## D-036 -- The plane-to-plane diagnostic evaluates the area under the measuring-line profile, plane by plane

**Date:** 2026-10-08, 16:26 (Europe/Rome). **Binds:** `scripts/plane_differences.py` (its default evaluation),
`focus.profile_areas`, `plotting.figures.profile_area_figure`, the notebook's Cell 4e. Not the focus rule (D-035), not
any measured diameter.

**Decision [user].** "This is the way of evaluating: take the measuring line, do the bilinear interpolation and
construct the profile. Evaluate the area under the curve for a focal plane and for the subsequent, take the
difference. This after the rendering." It replaces the reading the assistant had implemented of the user's proposal
of 15:45 (the positive part of $I_{k+1} - I_k$ over the pixels of a 10 x 10 um square), which stays available as
`--evaluation image`.

**Statement.** In every plane $k$ of the node's block (the pilot's request, planes $k_{\rm SWC} \pm 6$), $I_k(v)$ is
the bilinear profile along the node's measuring line (through the SWC node, across its fitted heading,
$|v| \le 3$ um, step 0.1144 um, no along-branch averaging); $A_k = \int I_k(v)\,dv$ (trapezoid; grey levels x um);
$\Delta A_n = A_{k_n+1} - A_{k_n}$. The plane of the smallest $A_k$ is framed. Beside it **[assistant, PROVISIONAL]**:
the same areas with each plane divided by its own background level $B_k$ (the interquartile mean of the block's pixels
more than Allen's radius + 4 um from every traced dendrite and the soma, the margin shrinking next to the soma) and
multiplied by the mean of $B_k$, so that a change of a whole plane's brightness cancels.

**Why [assistant's findings].** (1) A blur that conserves light leaves $A_k$ unchanged while the dip stays inside the
line. On rendered synthetic tubes (planes $-6 \ldots 6$) $A_k$ is flat within 1.1 (0.5 um tube) and 4.7 (1.5 um tube)
grey levels x um against a noise SD of 2.0, so its minimum falls at random; for a faint 5 um tube, whose blurred edges
leave the $\pm 3$ um line, $A_k$ is V-shaped with its minimum at the centre plane without noise and within one plane of
it in 6 of 8 noise draws **[run, sandbox]**. (2) A change of a whole plane's brightness moves $A_k$: a 1 % change moves
it by 12.2, 11.1 and 3.9 grey levels x um on the three tubes **[run]**; on nodes 2 and 3 the pixel-difference run of
16:22 shows the whole 10 x 10 um square 0.92 and 1.13 grey levels darker from plane 117 to 118 (the mean of
$I_{118} - I_{117}$), against $-0.19$ to $+0.49$ for the other pairs **[data read: `planediff_529878215.json` on
Drive]**, so the dip of that run's curve at 117->118 marks a darkening, not a focus.

**What it implies for code and tables.** Cell 4e's default figure: the planes with the measuring line, the profiles,
$A_k$ as measured and normalised, $\Delta A$, and $B_k$ per plane.

**What it does not decide [open].** Whether any plane-to-plane measure replaces $k^*$ (D-035); the background margin;
whether the line should be longer for trunks.

**Sources.** **[user]** 2026-10-08 16:26; repository `sci/diameter-pipeline` (`d6e6695`) **[src]**;
`planediff_529878215.json` **[Drive, data read]**; the synthetic demo **[run]**.

**Status:** active; a diagnostic.

---

## D-037 -- The plane-to-plane diagnostics gain the entropy of the grey levels along the measuring line and in a strip around it

**Date:** 2026-10-08, 17:07 (Europe/Rome). **Binds:** `scripts/plane_differences.py --evaluation entropy`,
`focus.grey_histogram`, `focus.histogram_entropy`, `focus.plane_entropies`, `profiles.stripe_mask`,
`plotting.figures.entropy_figure`, the notebook's Cell 4f. Not the focus rule (D-035), not D-036's default
evaluation, not any measured diameter.

**Decision [user].** "Can we please implement also another tool, the entropy of the distribution of pixels in the
local image, along the measuring line and (an addition) within a stripe that is centered in the measuring line and is
wide enought for some statistics. DO both the synthetic and the one with real data. Antoher cell please."

**Statement.** In every plane $k$ of the node's block (the pilot's request, planes $k_{\rm SWC} \pm 6$), two sets of
grey levels: the 53 bilinear samples $I_k(v_m)$ along D-036's measuring line ($|v| \le 3$ um, step 0.1144 um), and the
pixels $p$ of the strip centred on that line, $|(\mathbf x_p - \mathbf o)\cdot\hat y| \le 3$ um across the branch and
$|(\mathbf x_p - \mathbf o)\cdot\hat e_u| \le s$ along it ($\mathbf x_p$ the pixel centre, $\mathbf o$ the SWC node,
$\hat y$ across the fitted heading, $\hat e_u$ along it). For each set,
$\hat H_k = -\sum_j \hat p_{k,j}\log_2\hat p_{k,j}$ (bits), with $\hat p_{k,j} = n_{k,j}/N$ the fraction of the set's
$N$ values in grey-level bin $j$. $\hat H_k$ is computed from the one image of plane $k$ (computed level); it is the
plug-in estimate of the entropy of the set's grey-level distribution (analytic level: a value drawn uniformly from the
set's positions, with its camera noise), and it reads low when $N$ is small against the number of occupied bins.
**[assistant, PROVISIONAL]**: bins of 1 grey level centred on the levels; $s$ = 1 um (917 pixels at 0.1144 um/px); the
framed plane of each curve is its dip, the minimum between its two largest maxima (`focus.difference_dip`), because the
curve is W-shaped (below).

**Why [assistant's findings].** (1) On rendered synthetic tubes (planes $-6 \ldots 6$, centre in plane 0), $\hat H_k$
is lowest in focus, rises on both sides and falls again far out. With the camera's noise and JPEG, the strip's dip is
within one plane of the centre in 8 of 8 noise draws for a 0.5 um and for a 1.5 um tube, the line's in 7 of 8 and 8 of
8; for a faint 5 um tube both curves are flat and the dips fall at random, as for every other score tried
**[run, sandbox]**. Reading **[reasoning]**: in focus most of the neighbourhood sits at the ground level and the tube
covers few pixels; the blur spreads its darkness over more pixels and more grey levels; far out the tube fades into
the noise. (2) Why the strip: on flat ground with 3 grey levels of noise the plug-in reads 0.25 bits low with an SD of
0.13 bits at $N$ = 53 (the line), and 0.02 bits low with an SD of 0.03 at $N$ = 917 (the strip) **[run]**. That the
direct estimate needs every value to occur many times, and changes with the binning chosen, is stated by Zbili & Rama
2021 (PubMed, full text read) **[literature]**. (3) A plane-wide offset by whole grey levels leaves $\hat H_k$
unchanged exactly (the histogram moves, its counts do not), unlike D-036's area; a plane made 2 % darker moved its
strip entropy by 0.0075 bits on the synthetic cell **[run]**.

**What it implies for code and tables.** Cell 4f's figure: the planes with the line and the strip's outline, the
histograms of both sets coloured by plane, and both entropy curves with their dips framed. Outputs
`planeentropy_<id>.png` and `planeentropy_<specimen>.json`. No measurement or table changes.

**What it does not decide [open].** Whether any plane-to-plane diagnostic replaces $k^*$ (D-035); the bin width, the
strip's width and the pick rule; whether the real trunks carry a focus signal the synthetic 5 um tube lacks (nodes 2
and 3 in Cell 4f will show it); what other neurites or the soma inside the strip do to its histogram (node 2 lies next
to the soma).

**Sources.** **[user]** 2026-10-08 17:07; repository `sci/diameter-pipeline` at `8cec811` **[src]**; the synthetic
demo and `tests/smoke/test_smoke_plane_entropy.py` **[run]**. Zbili M., Rama S. 2021, *Front Neuroinform*
([DOI](https://doi.org/10.3389/fninf.2021.596443); PubMed, full text read). Context, PubMed, abstract only: Firestone
L. et al. 1991, *Cytometry* ([DOI](https://doi.org/10.1002/cyto.990120302)) ranked the autofocus measures based on the
image's probability density function or histogram worst among those compared on microscope images -- the abstract
does not say whether entropy was among them; Ren Z., Chen N., Lam E.Y. 2016, *Appl Opt*
([DOI](https://doi.org/10.1364/AO.55.001040)) build a depth-from-focus map by entropy minimisation in optical
scanning holography, another modality; Sun Y., Duthaler S., Nelson B.J. 2004, *Microsc Res Tech*
([DOI](https://doi.org/10.1002/jemt.20118)) and Liu X.Y., Wang W.H., Sun Y. 2007, *J Microsc*
([DOI](https://doi.org/10.1111/j.1365-2818.2007.01779.x)) compare focus algorithms; no claim taken from them. Full
texts of Firestone 1991 and Sun 2004 not accessible: they would say how a histogram-entropy focus measure fares on
brightfield stacks. Searches 2026-10-08: PubMed, 14 queries (autofocus or focus measure with entropy, histogram,
comparison, whole-slide imaging; entropy-estimation bias); bioRxiv, one listing (the connector has no keyword search;
90 days of bioengineering, nothing on the topic).

**Status:** active; a diagnostic; its parameters provisional.

**Note (2026-10-08, evening) [assistant; no decision changed].** The framed plane of each curve is now its global
minimum over the valid planes (`--entropy-pick min`, the default); the dip rule stays as `--entropy-pick dip`. Reason
**[data read: `planeentropy_529878215.json` on Drive, the run of 2026-10-08 17:56 UTC]**: on the trunk nodes 2 and 3
(planes 113-125, line $\pm 3$ um) the curves have no W. Node 2's line entropy is lowest at plane 114 (2.90 bits); its
two largest maxima, planes 124 (3.97) and 119 (3.80), made the dip rule search 120-123 and take 121 (3.41). On the
synthetic tubes the global minimum needs a line longer than $\pm 3$ um for the 0.5 um tube (at $\pm 3$ um the strip's
minimum fell on the end plane in 7 of 8 noise draws, where the tube fades into the noise; at $\pm 4$ and $\pm 5$ um
both curves were within one plane of the centre in 8 of 8) **[run, sandbox]** -- the link to D-038. On nodes 2 and 3
the line entropy's minimum (114; 119) and D-036's smallest area (113, normalised 115; 118, normalised 119) agree with
each other, and the gradient energy's $k^*$ (123; 123) does not; a reference plane chosen by eye is what would rank
them. Commit `3084991`.

---

## D-038 -- The measuring line is enlarged beyond +-3 um, to +-4 or +-5 um

**Date:** 2026-10-08, 18:02 (Europe/Rome). **Binds:** the half-length $h$ of the measuring line: the diagnostics'
`--profile-half-um` (`scripts/plane_differences.py`; the notebook's Cells 4e and 4f); and, if extended, Block 5's
`measure.profile_half_um` (an estimator field: the fit and the focus scores; it changes the bias table's hash) with
`measure.block_half_um`.

**Decision [user].** "I also think that we have to slighlty enlarge the measuring line up to [-4,4] or [-5,5] um".

**Statement.** The measuring line through a node, across its fitted heading, covers $|v| \le h$ with $h$ = 4 or 5 um
instead of 3 um. Which value, and whether Block 5's own line changes or only the diagnostics', are open **[open]**.

**Why [assistant's findings].** (1) On the trunk nodes 2 and 3 the profile at $v = \pm 3$ um is still well below the
background: about 85 grey levels at both ends on node 3 and 70-80 on node 2, against $B_k$ of 96-98 **[figure read:
the user's Cell 4e figures of 2026-10-08; the profiles were not in the JSON then]**. (2) On the synthetic tubes the
5 um tube's profile in plane 0 ends at 77 grey levels at $\pm 3$ um and at the ground, 80, at $\pm 5$ um; with the
global-minimum pick the entropy finds the 0.5 um tube's centre plane within one plane in 8 of 8 noise draws at
$\pm 4$ and $\pm 5$ um, against 7 of 8 draws on an end plane for the strip at $\pm 3$ um **[run, sandbox]**. (3)
Against it: D-036's area works on a tube wider than the line because defocus pulls ground light into a line shorter
than the blurred tube; once the line holds the blurred tube its light is kept, and on the 5 um tube the area's range
over the planes falls from 9.0 to 5.0 to 2.5 grey levels x um at $\pm 3$, $\pm 4$ and $\pm 5$ um **[run]**.

**What it implies for code and tables.** The diagnostics take any $h$: the square around the node grows to $h$ + 2
pixels and is fetched when it outgrows the pilot's block, whose half-width is `block_half_um` + 1 pixel at least
(so $\pm 5$ um, outermost samples at 5.03 um, needs the wider square); Cells 4e and 4f compare $\pm 3$ and $\pm 5$
um; the records carry every plane's profile, so the profile ends can be read without the images. If Block 5's line
changes: a new estimator hash and a new bias table (davinci), and `block_half_um` at least $h$ plus room for the
along-branch average and the tilt.

**What it does not decide [open].** 4 or 5 um; Block 5's line or only the diagnostics'; whether the area evaluation
keeps its $\pm 3$ um line.

**Sources.** **[user]** 2026-10-08 18:02; repository `sci/diameter-pipeline` at `3084991` **[src]**; the synthetic
runs **[run]**; `planediff_529878215.json` and `planeentropy_529878215.json` **[Drive, data read]**; the user's
figures **[figure read]**.

**Status:** decided in principle **[user]**; the value and the scope open.

---

## D-039 -- The entropy along a +-5 um line is the measure carried forward, tested next on thin dendrites

**Date:** 2026-10-09, 14:30 (Europe/Rome). **Binds:** the entropy evaluation (D-037) with `--profile-half-um 5`
(settles D-038's value for it); `scripts/plane_differences.py --nodes thin`, the notebook's Cell 4g. Not the focus
rule of Block 5 (D-035).

**Decision [user].** "I would continue with the entropy calculated across the [-5 5 ] across the node. However we have
tested this on thick dendrites. let's try small ones because on thin ones the gradient energy or the other methods
seemd working well."

**Statement.** Of the plane-to-plane diagnostics (the pixel differences, D-036's area, D-037's entropy), the one
carried forward is the Shannon entropy $\hat H_k$ of the grey levels along the measuring line, $|v| \le 5$ um, and in
its strip, each curve picking its global minimum. Before any further use it is run on thin dendrites, where the
gradient energy's $k^*$ (D-035) and the dip depth agree, and its picked planes are compared with $k^*$ node by node.

**Why [user; assistant's findings].** **[user]** On thin dendrites the gradient energy and the other measures appear to
work; the entropy has been tried only on the trunks 2 and 3. **[assistant]** At $\pm 5$ um the entropy picks plane 114
on node 2 and 115 on node 3, line and strip alike, where $k^*$ takes 123 on both **[figure read: the labels of the
user's figures of 2026-10-09, 12:22 UTC]**; on the synthetic 0.5 um tube its global minimum at $\pm 5$ um fell within
one plane of the centre in 8 of 8 noise draws, line and strip **[run, sandbox]**.

**What it implies for code and tables.** `--nodes thin --pilot-csv` chooses the test nodes **[assistant,
PROVISIONAL]**: in S, Allen's $2r \le 0.6$ um, a fitted $\hat d \le 1.0$ um, $k^*$ equal to the dip depth's plane, not
steep; up to 3 per unbranched stretch, spread along it, 12 in all, spread over the stretches. 91 of the pilot's 393
nodes qualify **[data read: `pilot_529878215.csv`]**. A run over several nodes writes a summary CSV (each pick minus
$k^*$) and a summary figure, and logs how many picks fall within one plane of $k^*$. Cell 4g runs it. No measurement
or table changes.

**What it does not decide [open].** Whether the entropy replaces $k^*$ on thick nodes, on all nodes, or not at all;
Block 5's own line (D-038's scope); the selection thresholds; what to do where the entropy and $k^*$ part on thin
nodes.

**Sources.** **[user]** 2026-10-09 14:30; repository `sci/diameter-pipeline` at `50081a4` **[src]**; the user's figures
of 2026-10-09 **[figure read]**; `pilot_529878215.csv` **[Drive copy, data read]**; the synthetic runs **[run]**.

**Status:** active; the thin-dendrite test pending (Cell 4g). [updated 2026-10-09, 16:40] Cell 4g ran on Allen's data (the user's log of 15:40): on 12 thin nodes the line's entropy fell within one plane of $k^*$ on 3 and the strip's on 1 (median offsets 4.0 and 5.5 planes) **[user's log, read]**; the user judged $k^*$ right on thin dendrites and the strip entropy at $\pm 5$ um right on the trunks 2 and 3 **[user]**. What follows is D-040.

---

## D-040 -- Gradient energy over the whole line at +-3 um, +-5 um and at width 2 d_hat; a sigmoid min-max blend with the strip entropy

**Date:** 2026-10-09, 16:19 (Europe/Rome), with the user's answers to four questions at 16:3x. **Binds:**
`scripts/plane_differences.py --evaluation gradient` (Cell 4h) and `--evaluation blend` (Cell 4i), `--nodes
bydiameter`; the notebook's Cells 4h and 4i. Diagnostics only: not the focus rule of Block 5 (D-035), no measurement
or table changes.

**Decision [user].** "write a cell that compares the gradient energy with [-3 3] and [-5 5] and a third that chooses
the measuring line's width as multiple of the estimated d, let's say 2 times. DO this across dendrite nodes with
different diameter values. In a second cell compare across several dendrites of several dimensions the third variant
(with the measuring line's width a multiple of the estimated diameter) and the Min-max rescaling with a sigmoidal
diameter-dependent function as weights, and finally the entropy alone (stripe) across the same range as the first
method". Answers: the gradient energy integrates over the whole line; the line of the third variant is $[-\hat d,
\hat d]$, i.e. full width $m \hat d$ with the multiplier $m$ (2) a parameter; $d$ is the pilot's $\hat d$; the
sigmoid's midpoint and width are $d_0 = 1.5$ um, $s = 0.3$ um.

**Statement.** For node $j$ and plane $k$, with $\tilde I_{j,k}(v)$ the profile along node $j$'s measuring line (through
the SWC node, across its fitted heading, bilinear, smoothed by one sample as in Block 5) and $B_{j,k}$ the plane's
background (the interquartile mean of the block's pixels far from every traced dendrite and the soma; Cell 4e's, the
same for every variant): $G^{(h)}_{j,k} = B_{j,k}^{-2} \int_{|v| \le h} (\partial_v \tilde I_{j,k})^2\,dv$ over the
whole line, for $h = 3$ um, $h = 5$ um and $h = m \hat d_j / 2$; each variant picks $\arg\max_k G^{(h)}_{j,k}$. On the
line $h_j = m \hat d_j / 2$, with $H_{j,k}$ the entropy of the strip's grey levels ($|v| \le h_j$ across, $|u| \le 1$ um
along; D-037), the min-max rescaled scores $g_{j,k} = (G_{j,k} - \min_l G_{j,l})/(\max_l G_{j,l} - \min_l G_{j,l})$ and
$\eta_{j,k} = (\max_l H_{j,l} - H_{j,k})/(\max_l H_{j,l} - \min_l H_{j,l})$ (over the valid planes) are combined as
$J_{j,k} = w(\hat d_j)\, g_{j,k} + (1 - w(\hat d_j))\, \eta_{j,k}$ with $w(d) = 1/(1 + \exp((d - d_0)/s))$; the blend
picks $\arg\max_k J_{j,k}$, the entropy alone $\arg\min_k H_{j,k}$.

**Why [user; assistant's findings].** **[user]** On thin dendrites the gradient energy works, on the trunks the strip
entropy at $\pm 5$ um; the user proposed a threshold or a sigmoid-weighted blend (15:40). **[assistant]** The gradient
energy recomputed from the saved $\pm 5$ um profiles of 14 nodes reproduces $k^*$ on all 14; its trunk failure follows
the window, Allen's radius + 0.5 um (about 1.6 um on trunks of apparent half-width 2.5-2.9 um); with the window set
from the measured width it picks 113 and 116 on the trunks (the strip entropy 114 and 115) and $k^*$ on the 12 thin
nodes, in every variant tried that takes the width from the image with a window margin of at least 1 um (with 0.5 um
node 2 drifts to 118) **[run, scratch, 2026-10-09; Drive JSONs, data read]**. The assistant advised against a threshold on the uncorrected $\hat d$ (205 of the 337 nodes in S have
$\hat d > 1$ um, and the median Allen $2r$ is about 0.5 um in every $\hat d$ bin **[data read: pilot table]**) and
against a max-normalised blend (it gives the gradient energy's plane for any $w$ above 0.01-0.28 on these nodes).

**What it implies for code and tables.** Two new evaluations of `scripts/plane_differences.py` (`gradient`,
`blend`) with `--line-mult` ($m$), `--sigmoid-d0-um`, `--sigmoid-s-um`; `--nodes bydiameter --pilot-csv` chooses the
nodes **[assistant, PROVISIONAL]**: per bin of $\hat d$ (edges 0.8, 1.0, 1.5, 2.0, 3.0 um) up to 2 nodes spread over
the stretches, without steep, crossing or unconverged nodes, plus nodes 2 and 3 (the trunks judged by eye). Per-node
figures with the planes drawn without the line, a summary figure and a summary CSV, as in Cell 4g. No measurement,
estimator hash or table changes.

**What it does not decide [open].** The production focus rule (D-035's window on trunks); a threshold or a blend;
whether $\hat d$ (uncorrected, fitted at $k^*$) or a width measured from the image should set the line; the selection.

**Sources.** **[user]** 2026-10-09 16:19 and the answers; the assistant's analysis of 2026-10-09 (15:40-16:10)
**[run, scratch]**; `planeentropy_529878215.json` (`thin_line_5um`, `line_5um`) and `pilot_529878215.csv` **[Drive,
data read]**.

**Status:** active; Cells 4h and 4i not yet run on Allen's data. [updated 2026-10-09, evening] Implemented and smoke-tested on `sci/diameter-pipeline` (commit `8a11f1e`): `focus.plane_gradient_energies`, `sigmoid_weight`, `minmax_scores`, `blend_scores`; `scripts/plane_differences.py --evaluation gradient` and `--evaluation blend`, `--nodes bydiameter`; notebook Cells 4h and 4i (runbook v11); `test_smoke_plane_gradient.py` 8 pass, 45 of 47 mutants killed (2 equivalent), all 19 suites pass **[run, sandbox]**. Two implementation choices **[assistant]**: the d-line takes the node's $\hat d$ re-measured on its cached crops, which equals the pilot's for an unchanged configuration; a node without a usable d-line keeps the two fixed lines in Cell 4h and is skipped in Cell 4i. Dry runs on synthetic cells, pilot tables made from them **[run, sandbox]**: on a 0.8 um tube (Allen radius 0.3 um) the three lines, $k^*$ and the blend ($w$ = 0.90) pick the in-focus plane, while the strip entropy alone on the d-line ($\pm 0.85$ um) picks the end plane 6 planes away -- the strip is mostly dendrite, so its entropy is highest one plane either side of focus (about 6.1 bits), dips at focus (about 5.8) and is lowest on the end planes (about 3.3); on a 3.0 um tube with the same Allen radius, $k^*$ falls four planes from the axis (flagged `stack_edge`), the three lines agree and pick one or two planes from it, and the blend ($w$ = 0.02) follows the entropy to the plane next to the axis. Cells 4h and 4i not yet run on Allen's data. [updated 2026-10-10] Cells 4h and 4i ran on Allen's data on 2026-10-09, 13 nodes (both summary CSVs on Drive, written 15:38-15:40 UTC) **[Drive, data read]**: $G$ over the whole d-line stayed within one plane of $k^*$ on 10 of the 11 non-trunk nodes (node 8462: 2 planes) and moved the trunks off the stack edge, to 116 (node 3) and 113 (node 2), each one plane from the $\pm 5$ um entropy planes judged right there (115, 114); the fixed lines did as well on 10 of 11, each failing on a different node; the d-line strip's entropy had its minimum on the last plane or the one next to it on all 11 non-trunk nodes, so the blend, equal to $G$'s plane wherever $w \ge 1/2$, followed it wherever $w < 1/2$ (5 of 5 non-trunk nodes off by 3 to 11 planes; the trunks at 115 and 116). In the user's three figures the d-line entropy was highest at the plane $G$ picks or one plane from it. What follows is D-041.

## D-041 -- The blend's entropy on the +-5 um strip; the objective written J = w g - (1 - w) h

**Date:** 2026-10-09, 17:47 (Europe/Rome), the user's message, and the option the user chose next (logged
2026-10-10). **Binds:** `scripts/plane_differences.py --evaluation blend` (`--entropy-half-um`),
`focus.blend_scores`, `figures.blend_figure`, `figures.picks_summary_figure`; notebook Cell 4i. Diagnostics only:
not the focus rule of Block 5 (D-035), no measurement or table changes. Changes D-040's blend in two points: the
entropy's strip and the written form of $J$.

**Decision [user].** "THe Entropy chooses the best value across planes at its minimum while the Gradietn energy the
maximum. Thus the composite objective should normalize the gradient energy and take the maximum while for the
entorpy normalize it and take the minimum. The sigmoidal weights are ok. the metric thus enforces the plane with the
minimum of the entropy and maximum of the gradient energy" (17:47). Offered three ways on (fix the plot only; fix the
plot and take the entropy from the $\pm 5$ um strip; leave Cell 4i and decide the focus rule), the user chose "Plot
fix + entropy on $\pm 5$ um": $G$ stays on the d-line, the entropy comes from the $\pm 5$ um strip, the objective is
written $w g - (1 - w) h$ and the entropy is drawn un-inverted.

**Statement.** For node $j$ and plane $k$, with $G_{j,k}$ the gradient energy over the whole d-line
($h_{d,j} = m \hat d_j / 2$, $m$ = 2; D-040) and $H_{j,k}$ the entropy (bits) of the strip's grey levels,
$|v| \le h_H$ across and $|u| \le 1$ um along, $h_H$ = 5 um: over $V_j$, the planes where both are finite,
$g_{j,k} = (G_{j,k} - \min_{V_j} G_j)/(\max_{V_j} G_j - \min_{V_j} G_j)$ and
$h_{j,k} = (H_{j,k} - \min_{V_j} H_j)/(\max_{V_j} H_j - \min_{V_j} H_j)$, and
$J_{j,k} = w(\hat d_j)\, g_{j,k} - (1 - w(\hat d_j))\, h_{j,k}$, $w(d) = 1/(1 + \exp((d - d_0)/s))$, $d_0$ = 1.5 um,
$s$ = 0.3 um (unchanged); the blend picks $\arg\max_k J_{j,k}$, the entropy alone $\arg\min_k H_{j,k}$.

**Why [user; assistant's findings].** **[assistant]** D-040's form, $w g + (1 - w)(1 - h)$, equals
$[w g - (1 - w) h] + (1 - w)$; the last term is the same on every plane of a node, so the user's objective was
already the one computed, and the figure drew $1 - h$, which read as the entropy itself. **[Drive, data read: the two
summary CSVs of 2026-10-09]** The real issue was the strip: on the d-line the entropy's minimum fell on the last
plane or the one next to it on all 11 non-trunk nodes, and the blend followed it wherever $w < 1/2$ (D-040's status).
**[assistant, reasoning]** The d-line strip is about half dendrite, so focus spreads its pixels over more grey
levels; "lowest entropy in focus" needs the background to dominate the strip, as on the $\pm 5$ um strip where the
entropy was judged right on the trunks (D-039). Two autofocus comparisons ranked histogram-based focus measures
below derivative-based ones on their test images (Firestone et al. 1991, *Cytometry*, doi 10.1002/cyto.990120302;
Rudnaya et al. 2010, *J Microsc*, doi 10.1111/j.1365-2818.2010.03383.x) **[PubMed, abstract only]**: context, not
evidence for a strip around a dendrite.

**What it implies for code and tables.** `--entropy-half-um` (default 5) for the blend, `ENTROPY_HALF_UM` in Cell 4i;
the square around a node holds the strip (about 13 new crops per node, as in Cell 4g); the records carry `h_norm`,
`k_Hs` and `entropy_half_um` (were `eta`, `k_Hd`); Cell 4i writes to `pilot/planeblend/bydiameter_strip5um/`,
keeping the first run's `pilot/planeblend/bydiameter/`; the summary figure no longer inverts any curve. Implemented
on `sci/diameter-pipeline`, commit `be878dc`: `test_smoke_plane_gradient.py` 8 pass, 17 of 18 mutants killed (1
equivalent), all 19 suites pass; Cell 4i dry-run on synthetic cells: on a 0.8 um tube $G$, the $\pm 5$ um entropy
and the blend all pick the in-focus plane; on a 3.0 um tube with a 0.3 um Allen radius the $\pm 5$ um entropy picks
the plane next to the axis and the blend ($w$ = 0.02) follows it **[run, sandbox]**. No measurement, estimator hash
or table changes.

**What it does not decide [open].** The production focus rule (D-035's window on trunks): on the first real run $G$
over the whole d-line alone was within one plane of $k^*$ on 10 of 11 non-trunk nodes and within one plane of the
judged trunk planes, so whether a blend is needed at all is open; how the $\pm 5$ um entropy behaves on the nodes of
1.5-3 um, where it now decides (on Allen's thin nodes it fell within one plane of $k^*$ on 1 of 12, Cell 4g, where
$w$ is about 0.9); the selection of D-040.

**Sources.** **[user]** 2026-10-09 17:47 and the option chosen; **[Drive, data read]**
`planegrad_summary_529878215.csv`, `planeblend_summary_529878215.csv`; **[run, sandbox]** commit `be878dc`;
**[PubMed, abstract only]** the two DOIs above.

**Status:** active; implemented (commit `be878dc`); Cell 4i's re-run on Allen's data pending. [updated 2026-10-10, later] Re-run on Allen's data at 13:18-13:20 UTC, 13 nodes, into `pilot/planeblend/bydiameter_strip5um/` (the summary CSV read in full) **[Drive, data read]**: the $\pm 5$ um strip's entropy fell within one plane of $k^*$ on 2 of the 11 non-trunk nodes (3856, 684) and 2 to 6 planes from it, on either side, on the other 9; on the trunks it picked 116 (node 2) and 115 (node 3). The blend was within one plane of $k^*$ on 7 of 11: on the six with $w > 1/2$ and on 684 ($w$ = 0.21), and it followed the entropy on the four with $w < 0.3$ where the entropy was off (8482, 8462, 678, 8473: +4, +6, -5, +4). $G$ on the d-line was within one plane of $k^*$ on 10 of 11 (8462: -2) and picked 113 (node 2) and 116 (node 3), as on 2026-10-09. What follows is D-042.

## D-042 -- The gradient energy's line sized from Allen's diameter first, then from the fit at its plane

**Date:** 2026-10-10, 15:30 (Europe/Rome), the user's message, and the option the user chose next. **Binds:**
`scripts/plane_differences.py --evaluation iterate` (`--line-mult`, `--max-rounds`), `plane_differences.check_iteration`,
`iterate_planes`, `refit_at_plane`, `gradient_iteration`, `trajectory_text`, `node_stack(..., allen_line_mult)`,
`figures.iteration_figure`, `figures.iteration_style`, `figures.picks_summary_figure(..., note_key)`; notebook Cell 4j.
Diagnostics only: not the focus rule of Block 5 (D-035), no measurement or table changes.

**Decision [user].** "Now watching the results the Gd is the best one. However let's use as first guide the allen's
diameters. Let's do a cell where this approach is used." (15:30, ten minutes after the re-run of Cell 4i under D-041.)
Of the options then offered, the user chose "Allen first, then refine": the line $\pm m r_{\rm Allen}$, the plane of
the largest $G$, the fit $\hat d$ at that plane, the line $\pm m \hat d / 2$, a new plane, and so on until the plane
repeats (at most 5 rounds), with the trajectory reported.

**Statement.** For node $j$ with Allen's radius $r_{A,j}$, $m$ = 2 and at most $R$ = 5 rounds: $d^{(0)}_j = 2 r_{A,j}$;
in round $i = 1, \ldots, R$ the line of half-length $h_{j,i} = m\, d^{(i-1)}_j / 2$ through the SWC node, across the
fitted heading; $G^{(h_{j,i})}_{j,k} = B_{j,k}^{-2} \int (\partial_v \tilde I_{j,k})^2\, dv$ over the whole line, with
the background $B_{j,k}$ of D-040; and $k_{j,i} = \arg\max_k G^{(h_{j,i})}_{j,k}$ over $k_{\rm SWC} \pm 6$. If
$k_{j,i} \in \{k_{j,1}, \ldots, k_{j,i-1}\}$ the iteration stops -- `converged` when $k_{j,i} = k_{j,i-1}$, `cycle`
otherwise -- and the fit made there is reused; else $d^{(i)}_j = \hat d_j(k_{j,i})$, Block 5's final fit at plane
$k_{j,i}$ with every other input the pilot's (its centre, heading, tilt and offsets; $\bar B$ of that plane), so that
$\hat d_j(k^*_j) = \hat d_j$. Other endings: `fit_failed`, `no_plane`, `line_short`, `square_small`, `max_rounds`.
Reported per node: every round ($d$ in, $h$, $k$, $G$, the fit), the last plane `k_iter` and the fit there
`d_iter_um`, round 1's plane `k_allen`, and Cell 4h's d-line pick `k_Gd` for comparison.

**Why [user; assistant's findings].** **[user]** $G$ over the whole d-line was the best of the plane rules compared.
**[Drive, data read: `pilot/planeblend/bydiameter_strip5um/planeblend_summary_529878215.csv`, 13:20 UTC]** In the
re-run of Cell 4i that preceded the message, $G$ on the d-line was within one plane of $k^*$ on 10 of the 11 non-trunk
nodes, the blend on 7, the $\pm 5$ um strip's entropy on 2 (D-041's status). **[assistant, reasoning]** The d-line is
sized from the pilot's $\hat d$, which Block 5 fits at $k^*$, the plane under test; on the trunks, where $k^*$ sits at
the stack edge (123), that $\hat d$ (5.0 and 5.5 um) comes from a plane seven and ten planes from the ones $G$ then
picks (116, 113). Allen's diameter does not depend on the focus rule (about 0.5 um in every bin of $\hat d$, D-040;
2.0 and 2.2 um on the trunks); refining it by the fit at the plane its own line picks removes the dependence, where the
iteration settles. **[PubMed, abstract only]** An autofocus method for red-blood-cell images uses a focus window that
adapts to the image (Zhong et al. 2019, *Appl Opt*, doi 10.1364/AO.58.008031): context for sizing the window from the
object, not evidence for this iteration; no precedent for refitting the width at the chosen plane was found (PubMed,
seven queries; bioRxiv, a scan of 30 recent neuroscience preprints, no keyword search).

**Implementation choices [assistant, PROVISIONAL].** (i) "The plane repeats" is read as any earlier plane of the node's
own trajectory, so a cycle also ends it, reported as `cycle`. (ii) "Fit there" is Block 5's final fit with only the
plane changed (the pilot's centre, heading, tilt and offsets; $\bar B$ of that plane); at $k^*$ it returns the pilot's
$\hat d$ to the last bit. (iii) Round 1 counts among the 5. (iv) $G$'s line passes through the SWC node, as in Cells 4h
and 4i, the fit's through the pilot's centre, as in Block 5. (v) A fit at a bound sizes the next line like any other.
(vi) The square is made again, larger, when a later line does not fit (on real data from the cached block while the
line is shorter than about 4.7 um).

**What it implies for code and tables.** `--evaluation iterate` with `--line-mult` (2) and `--max-rounds` (5); Cell 4j
writes `pilot/planeiter/bydiameter/`. Implemented on `sci/diameter-pipeline`, commit `2bcc8f4`: `test_smoke_plane_iteration.py` 7 pass, 36 of 36 mutants killed (one only after the reference check moved to a plane whose background differs from $k^*$'s), all 20 suites pass; Cell 4j dry-run on two synthetic cells: on a 0.8 um tube every node converged in 2 rounds at the in-focus plane with the pilot's $\hat d$; on a 3.0 um tube (Allen radius 0.3 um) round 1 picked the end plane (-6) and fitted 2.7-2.8 um there, round 2 a plane one or two from the axis with 3.0-3.2 um, and round 3 repeated it (converged on every node, on the pilot d-line's plane), while $k^*$ stayed four planes off **[run, sandbox]**. No
measurement, estimator hash or table changes.

**What it does not decide [open].** Whether `k_iter` and `d_iter_um` should stand in for $k^*$ and $\hat d$ in Block 5
where the focus window fails (the trunks), which is the production focus rule (D-035); how the iteration behaves on
Allen's thick nodes, where round 1's line lies inside the dendrite; the readings marked PROVISIONAL above.

**Sources.** **[user]** 2026-10-10 15:30 and the option chosen; **[Drive, data read]**
`planeblend_summary_529878215.csv` (13:20 UTC); **[run, sandbox]** commit `2bcc8f4`; **[PubMed, abstract only]**
doi 10.1364/AO.58.008031.

**Status:** active; implemented (commit `2bcc8f4`); not run on Allen's data. [updated 2026-10-12] Cell 4j ran on Allen's data on 2026-10-12, 12:43-12:45 UTC, 13 nodes, into `pilot/planeiter/bydiameter/` (the summary CSV read in full) **[Drive, data read]**: every node converged, in 2 rounds (7 nodes), 3 (5) or 4 (node 3), with no cycle and no failed fit; `k_iter` equalled Cell 4h's `k_Gd` on all 13. Round 1 alone, on Allen's line, was within one plane of $k^*$ on all 11 non-trunk nodes, and the refinement kept 10 of them within one plane (8462: -2); on the trunks round 1 stayed next to $k^*$ (122 on node 3, 124 on node 2) and the refinement moved them to 115 and 113, against the planes of the $\pm 5$ um entropy judged right in D-039 (115, 114). `d_iter_um` equalled the pilot's $\hat d$ on the 8 nodes whose last plane is $k^*$ and differed on the other 5 (8462: 2.38 against 2.36 um; 678: 2.79 against 2.83; 8473: 2.81 against 3.11; node 3: 5.38 against 5.01; node 2: 6.00 against 5.50). On node 2 both outputs sit at a limit: the plane 113 is the first of the range scanned ($k_{\rm SWC} \pm 6$ = 113..125), and 6.00 um is the fit's upper bound (`fit_d_bounds_um` = (0.05, 6.0) um) **[src: `config.py`]**. What follows is D-043. [updated 2026-10-12, later] `--line-mult` defaults to 3 for `--evaluation iterate` by D-044, and Cell 4j writes `pilot/planeiter/bydiameter_m3/`; the run above stays in `pilot/planeiter/bydiameter/`.

## D-043 -- The Allen-guided iteration is the method carried forward

**Date:** 2026-10-12, 14:52 (Europe/Rome), after the user's five figures of Cell 4j's run on Allen's data (nodes 2,
8462, 678, 8473, 3). **Binds:** the choice of the plane and of the diameter among the focus diagnostics of D-036 to
D-042: from here on the method is D-042's iteration (`scripts/plane_differences.py --evaluation iterate`, notebook
Cell 4j), and the others -- the window rule $k^*$ of D-035, the fixed lines and the pilot's d-line of D-040, the blend
of D-040 and D-041, the entropy of D-037 and D-039 -- are labelled comparisons. Block 5's focus rule (D-035) and the
bias table are not changed by this entry; where the iteration enters them is open, (a) below.

**Decision [user].** "That's perfect. Let's use this last method." (14:52, of the iteration of D-042 as run in Cell 4j.)

**Statement.** For each dendrite node $j$, with Allen's radius $r_{A,j}$ (um) and $m$ = 2: $d^{(0)}_j = 2 r_{A,j}$;
round $i$ takes the line of half-length $h_{j,i} = m\, d^{(i-1)}_j / 2$, the plane
$k_{j,i} = \arg\max_{k \in K_j} G^{(h_{j,i})}_{j,k}$ over $K_j = \{k_{{\rm SWC},j} - 6, \ldots, k_{{\rm SWC},j} + 6\}$,
and $d^{(i)}_j = \hat d_j(k_{j,i})$, Block 5's fit at that plane with every other input the pilot's; it stops when a
plane repeats, after at most 5 rounds. The node's plane is $k^{\rm it}_j$, the last plane picked, and its diameter
$d^{\rm it}_j = \hat d_j(k^{\rm it}_j)$, both as D-042 defines them, its PROVISIONAL readings included.

**Why [user; data read].** **[user]** "That's perfect", of the five figures. **[Drive, data read:
`pilot/planeiter/bydiameter/planeiter_summary_529878215.csv`, 12:45 UTC]** All 13 nodes converged (D-042's status);
`k_iter` equals Cell 4h's d-line pick on all 13 and stays within one plane of $k^*$ on 10 of the 11 non-trunk nodes,
and it moves the trunks off the stack edge -- to 115 on node 3, the plane judged right in D-039, and to 113 on node 2,
one plane from it -- without the pilot's $\hat d$ sizing any line.

**What it implies for code and tables.** Nothing yet: no default of Block 5, no estimator hash and no table changes
with this entry. The iteration's outputs, `k_iter` and `d_iter_um`, are the quantities the next step carries; the
other diagnostics stay available as labelled comparisons.

**What it does not decide [open].** (a) Where the iteration enters the measurement: inside Block 5 for every node, in
place of $k^*$ -- then the bias table must measure its phantoms through the same rule, and the phantom chain hands
Block 5 the true radius $d/2$ (`analysis/phantoms.py`: `Branch.from_points(xyz, 0.5 * draw.d_um)` **[src]**) where a
real node hands it Allen's, so the phantoms' starting diameter would have to mimic Allen's -- or as a step after
Block 5 on the real nodes only. (b) The plane range: on node 2 the last plane, 113, is the first of the 13 planes
scanned. (c) The fit on the trunks: on node 2 $d^{\rm it}$ = 6.00 um is the fit's upper bound (`fit_d_bounds_um` =
(0.05, 6.0) um **[src: `config.py`]**), and the fit's profile spans $\pm$`profile_half_um` = $\pm 3$ um **[src]**
(D-038's open line length). (d) D-042's PROVISIONAL readings.

**Sources.** **[user]** 2026-10-12 14:52 and the five figures; **[Drive, data read]**
`planeiter_summary_529878215.csv` (12:45 UTC); **[src]** `config.py` (`fit_d_bounds_um`, `profile_half_um`),
`analysis/phantoms.py`.

**Status:** active; where it enters the measurement open, (a). [updated 2026-10-12, later] $m$ = 3 from D-044; the statement's $m$ = 2 is the value of the run D-043 judged (`pilot/planeiter/bydiameter/`). [updated 2026-10-16] (a)'s first half settled by D-052: the iteration enters the measurement inside Block 5, for every node, phantoms included (procedure §3.7), each round's $d$ from D-052's fit; the phantoms' starting radius, (a)'s second half, stays open (D-052 (a)).

---

## D-044 -- The Allen-guided iteration's lines are three diameters wide (m = 3)

**Date:** 2026-10-12, 15:03 (Europe/Rome), the user's reply to the summary of D-043's method. **Binds:** the line
multiplier $m$ of the Allen-guided iteration (D-042, D-043): `plane_differences.ITER_LINE_MULT` = 3, the default of
`gradient_iteration` and of `scripts/plane_differences.py --evaluation iterate` (`--line-mult` unset;
`run(line_mult=None)` takes each evaluation's own default); notebook Cell 4j (`LINE_MULT = 3.0`, outputs in
`pilot/planeiter/bydiameter_m3/`). The comparison line Gd of that evaluation is sized from the pilot's $\hat d$ at the
same $m$. Not bound: the d-line of the gradient and blend evaluations (D-040, Cells 4h and 4i), which keeps $m$ = 2
(`plane_differences.D_LINE_MULT`).

**Decision [user].** "Perfect set the multiplier to 3 now shall we do next?" (15:03.)

**Statement.** In the iteration of D-043, $m$ = 3: round $i$'s line through node $j$ has half-length
$h_{j,i} = 3\, d^{(i-1)}_j / 2$, so round 1's is $\pm 3 r_{A,j}$ and every later one $\pm 1.5\, \hat d_j(k_{j,i-1})$;
the rest of D-042 and D-043 is unchanged ($R$ = 5 rounds, $K_j = k_{{\rm SWC},j} \pm 6$, the fit at the picked plane,
the stopping rule). The comparison pick `k_Gd` is $\arg\max_{k \in K_j} G^{(h)}_{j,k}$ on the line of half-length
$h = 1.5\, \hat d_j$, with $\hat d_j$ the pilot's fit at $k^*_j$.

**Why [user; assistant's findings].** **[user]** The value 3, with no reason given. **[run, sandbox]** On the synthetic
tubes (Allen radius 0.3 um) $m$ = 3 ends on the plane of $m$ = 2 with the same diameter on every node tried: on the
0.8 um tube in 2 rounds at the in-focus plane (lines $\pm 0.9$ and $\pm 1.27$-1.28 um); on the 3.0 um tube in 3 rounds,
one or two planes from the axis, fit 3.0-3.2 um (lines $\pm 0.9$, $\pm 4.1$-4.2 and $\pm 4.6$-4.8 um, the square made
again twice per node); round 1, inside the 3.0 um tube at either $m$, took the other end plane on one node of the smoke
test (-5 for -6, their $G$ within 2 %). **[reasoning]** Once a line covers the blurred edges, lengthening it adds
background on both sides, which carries little gradient energy; so $m$ matters where the diameter sizing the line is
too small (round 1 on thick nodes) and, at the other end, where a long line reaches neighbouring neurites. From the
fits of the run at $m$ = 2 (D-042's status), the trunks' lines become about $\pm 8$-9 um (1.5 x 5.38 and 1.5 x 6.00
um) **[arithmetic on Drive data read]**.

**What it implies for code and tables.** Commit `941fc75` on `sci/diameter-pipeline`: the constants
`ITER_LINE_MULT` = 3 (this entry) and `D_LINE_MULT` = 2 (D-040), each the labelled default of its evaluations;
`--line-mult` unset resolves per evaluation; Cell 4j at $m$ = 3 writes `pilot/planeiter/bydiameter_m3/`, leaving the
run at $m$ = 2 in `pilot/planeiter/bydiameter/`; runbook v14. Checks: `test_smoke_plane_iteration.py` 7 pass (at $m$ = 2 and 3) and `test_smoke_plane_gradient.py` 8 pass; 9 of 9 mutants of the defaults killed; every suite re-run (20, no failure); Cell 4j at $m$ = 3 dry-run on the two synthetic tubes, the planes and fits of $m$ = 2 **[run, sandbox]**. No measurement, estimator hash or table
changes: the iteration is not inside Block 5 (D-043 (a)).

**What it does not decide [open].** Whether $m$ = 3 moves `k_iter` or `d_iter_um` on Allen's data (Cell 4j at $m$ = 3
not yet run); whether the trunks' longer lines reach neighbouring neurites; the $m$ of the phantoms if the iteration
enters Block 5 (follows D-043 (a)); D-043's (b)-(d).

**Sources.** **[user]** 2026-10-12 15:03; **[run, sandbox]** commit `941fc75`; **[Drive, data read]**
`pilot/planeiter/bydiameter/planeiter_summary_529878215.csv` (12:45 UTC), for the trunks' fits.

**Status:** active; implemented (commit `941fc75`); Cell 4j at $m$ = 3 not yet run on Allen's data. [updated 2026-10-13] Cell 4j at $m$ = 3 has run on Allen's data: `pilot/planeiter/bydiameter_m3/` created 2026-10-13 12:25 UTC, `planeiter_529878215.json` and `planeiter_summary_529878215.csv` written 12:27 UTC **[Drive listing, metadata only]**; the results are not yet read. [updated 2026-10-13, later] Cell 4k's log, which prints each node's `k_iter` and Cell 4j's `d_iter_um` at $m$ = 3, beside the summary of the run at $m$ = 2: the same plane on 12 of 13 nodes, and so the same diameter (the fit at a plane does not depend on $m$); node 8462 moved from plane 47 to 46 ($k^*$ 49) and its d from 2.38 to 2.73 um. The trunks end where they ended at $m$ = 2 (node 3 at 115, d 5.38 um; node 2 at 113, d 6.00 um, at the fit's bound) **[user's log of 2026-10-13, 14:36, read; the m = 2 summary CSV, data read]**.

---

## D-045 -- Cell 4k: each node's chosen plane with the line that chose it and the diameter fitted there

**Date:** 2026-10-12, 17:58 (Europe/Rome), the user's request, and the answers to four questions asked next.
**Binds:** `scripts/iteration_fits.py` (`read_records`, `selection_round`, `node_square`, `fit_overlay`, `run`),
`plane_differences.refit_at_plane(..., with_profile=True)`, `figures.iteration_fit_figure`,
`figures.iteration_fits_overview`; notebook Cell 4k. A figure: no measurement, estimator hash or table changes.

**Decision [user].** "Can you write a Cell 4k in which for each focal plane chosen with the method you plot the image
and the measuring line with the estimated diameter" (17:58). To the questions then asked: the line drawn is "the line
chosen to select the focal plane"; each node also shows the profile with the fitted model; an overview of every node
plus one figure per node; by default the run of Cell 4j at $m$ = 3 (`pilot/planeiter/bydiameter_m3/`).

**Statement.** For node $j$ of a Cell 4j record with a plane $k^{\rm it}_j$: the selection round $i^\ast_j$ is the last
round that picked a plane ($k_{j,i^\ast_j} = k^{\rm it}_j$; at convergence the repeat round, whose line is sized from
$d^{\rm it}_j$ itself) and $h^{\rm sel}_j$ its half-length; the line drawn is
$L_j = \{o_j + v\,\hat y_j : |v| \le h^{\rm sel}_j\}$, through the SWC node $o_j$ across the pilot's heading $\theta_j$,
$\hat y_j = (-\sin\theta_j, \cos\theta_j)$. The diameter is Block 5's fit redone at $k^{\rm it}_j$: $d^{\rm it}_j$ and
$\hat v_{0,j}(k^{\rm it}_j)$, measured along $\hat y_j$ from the pilot's fitted centre $c_j$; with
$s_j = (c_j - o_j)\cdot\hat y_j$ its edges lie on $L_j$ at $v^{\pm}_j = s_j + \hat v_{0,j}(k^{\rm it}_j) \pm d^{\rm it}_j/2$
(the caliper). The fit's own line, through $c_j$, is parallel to $L_j$ and $u_j = (c_j - o_j)\cdot\hat e_{u,j}$ apart
along the branch (reported per node). Each node's square is Cell 4j's last one, rebuilt, and the fit must give Cell
4j's `d_iter_um` to the last bit; a difference is printed in the node's notes.

**Why [user; assistant's choices and findings].** **[user]** To see, on the plane chosen, the line that chose it and
the diameter estimated there. **[assistant, PROVISIONAL]** The caliper is placed at the fitted edges' position across
the branch, carried onto $L_j$ along the branch, so it marks the dendrite whatever line the fit sampled; the profile
panel draws the fit's samples at $s_j + v_n$ on $L_j$'s coordinate. **[run, sandbox]** On the synthetic tubes, whose
axis passes through the SWC nodes, the caliper's middle lay within 0.004 um of the node; moving the fit's centre by
0.2 um across or 0.3 um along the branch moved it by less than 0.01 um. On the 3.0 um tube the profile panel showed
$\bar B$ at $k^{\rm it}$ 3.7 to 11.6 grey levels under the profile's plateau (nodes 2, 4, 6), and 0.7 to 3.6 when
Block 5's background mask was sized from the true radius instead of Allen's (0.3 um + the 1 um margin); a finding for
D-043 (c), not acted on here.

**What it implies for code and tables.** Commit `128fdce` on `sci/diameter-pipeline`: Cell 4k reads Cell 4j's
`planeiter_<specimen>.json` from `ITER_DIR` and writes `planefit_<id>.png`, `planefit_overview_<specimen>.png` and
`planefit_<specimen>.json` into `ITER_DIR/fits/`; runbook v15. Checks: `test_smoke_iteration_fits.py` 6 pass, 1 skip; 16 of 16 mutants of the new code killed (the model drawn with the pilot's background only after a check off k* was added); every suite re-run (21, no failure); Cells 4j and 4k ran from the notebook's own code on the two synthetic tubes: every node drawn, the fit redone equal to Cell 4j's d_iter, 0 downloads **[run, sandbox]**. No measurement, estimator hash or table
changes.

**What it does not decide [open].** Whether $\bar B$ on thick dendrites should come from a mask sized from the fitted
diameter rather than Allen's radius (the finding above; D-043 (c)); D-043's (a)-(d).

**Sources.** **[user]** 2026-10-12 17:58 and the four answers; **[run, sandbox]** commit `128fdce`.

**Status:** active; implemented (commit `128fdce`); not yet run on Allen's data. [updated 2026-10-13] Cell 4k ran on Allen's data on the run of Cell 4j at $m$ = 3: 13 nodes drawn, 0 skipped, the fit redone at `k_iter` equal to Cell 4j's `d_iter_um` on 13 of 13, every crop from the cache (0.7 min); node 2's fit is `at_bound`. The caliper's middle lies within 0.2 um of the SWC node on 9 nodes, and 0.30 to 0.77 um from it on 678, 2, 8473 and 8462 **[user's log of 2026-10-13, 14:36, read]**. The cell's display part did not run in that session: four lines quoted from inside `iteration_fits.fit_overlay` in the assistant's reply had been pasted after the cell's code (a NameError, no file affected).

---

## D-046 -- Cell 4l: fit variants at the iteration's plane, with other blur-kernel families and centre weights

**Date:** 2026-10-13, 14:50 (Europe/Rome), the user's request, and the answers to six questions asked next.
**Binds:** `model/blur_kernels.py`, `tube_model.model_profile_kernel`, `analysis/fit_variants.py` (`VariantSpec`,
`variant_grid`, `fit_unweighted`, `fit_weighted`, `fit_grid`), `scripts/fit_variants.py` (`node_inputs`,
`select_specs`, `run`), `figures.fit_variants_node_figure`, `figures.fit_variants_summary_figure`; notebook Cell 4l. A
diagnostic: no measurement, estimator hash or table changes; Block 5 keeps its Gaussian (D-026) and $\sigma_{\rm fit}$
(D-023).

**Decision [user].** "The thin dendrites are HUGELY over estimated. Can we do a crazy thinf. Taking the centered node
as the peak I would like that the fit gives, exponentially weights starting from the center. so that the curve peak
close to the center has more weight in the fit. Moreover I would also like to try (combined and not with the former
weighted fit) a different noise kernel (try different families) this is just to try out more thing while I study for
the construction of the correction map" (14:50, on Cell 4k's overview of the 13 nodes). To the questions then asked:
the weights centred on the fit's own centre $v_0$, re-centred at every iteration (IRLS); $\lambda$ scanned at 0.25,
0.5, 1 and 2 um beside the unweighted fit; the "noise kernel" is the blur kernel $g$ of the model; its width both fixed
at the FWHM of Block 5's Gaussian and fitted per node; shapes both fixed at representative members and fitted; the
fits made at Cell 4j's $k^{\rm it}$, the plane not re-chosen.

**Statement.** For node $j$, with Block 5's inputs at $k^{\rm it}_j$ (the samples $(v_n, I_n)$, the tilt $\phi_j$,
$\bar B_j$): $I_{\rm model}(v \mid d, \mu, v_0; g) = \bar B_j\,[1 - (d/2) \int_{-\pi/2}^{\pi/2} (1 - e^{-\alpha \cos\psi})
\cos\psi\; g(v - v_0 - (d/2) \sin\psi)\, d\psi]$, $\alpha = \mu d / \cos\phi_j$, with $g(z) = f_F(z / a + m_F(s); s) / a$,
$a = w / W_F(s)$: the unit density of family $F$ with its mode $m_F(s)$ moved to 0, FWHM $w$ (um) and shape $s$.
Families: Gaussian, Lorentzian, Voigt ($s = r$, the ratio of the Lorentzian component's FWHM to the Gaussian's),
Student ($\nu$), generalized normal ($\beta$), skew-normal ($a$). Ten kernels (Gaussian; Lorentzian; Voigt $r$ = 1;
Student $\nu$ = 3; generalized normal $\beta$ = 1 and $\beta$ = 4; Voigt, Student and generalized normal with the
shape fitted; skew-normal with the skew fitted) x $w$ fixed at $2\sqrt{2 \ln 2}\,\sigma_{\rm fit}$ = 0.233 um or
fitted in [0.1, 2] um x unweighted or weighted by $\omega_n = \exp(-|v_n - v_0| / \lambda)$,
$\lambda \in \{0.25, 0.5, 1, 2\}$ um: 100 variants per node. A weighted fit iterates
$\omega^{(t)}_n = \exp(-|v_n - v_0^{(t)}| / \lambda)$,
$\theta^{(t+1)} = \arg\min_\theta \sum_n \omega^{(t)}_n [(I_{\rm model}(v_n \mid \theta) - I_n) / \bar B_j]^2$ from
the unweighted fit of the same kernel and width, until $d$ and $v_0$ both move by at most 1e-4 um (or 30
iterations). The Gaussian at the fixed width, unweighted, is Block 5's fit ($d_{\rm base}$).

**Why [user; assistant's choices and findings].** **[user]** The thin dendrites look hugely overestimated on Cell
4k's overview; the variants are trials while the correction map is studied. **[assistant, PROVISIONAL]** Each kernel
is parameterised by its FWHM, so that at the fixed width every kernel has the half-maximum width of Block 5's Gaussian
and the comparison isolates the shape; each kernel's mode sits at $v_0$ (only the skew-normal's mode is not at 0 in
scipy's parameterisation); the fixed members ($r$ = 1, $\nu$ = 3, $\beta$ = 1 and 4), the ranges of the fitted widths
and shapes, the start points (Block 5's, then warm starts from neighbouring variants), and the re-weighting's tolerance
and cap are the assistant's. The fixed point is that of the user's rule, not a joint minimum over $v_0$, which would
let the weights slide onto the background. [corrected 2026-10-15: the first half stands -- at the rule's fixed point
$\theta^*$ the objective with the weights moving with $v_0$, $\Phi(\theta) = \tfrac12\sum_n \omega_n(v_0)\,r_n(\theta)^2$,
has $\partial\Phi/\partial v_0 = (2\lambda)^{-1}\sum_n \omega_n \operatorname{sgn}(v_n - v_0^*)\, r_n^2$, not 0 unless the
weighted squared residuals balance across $v_0^*$; the second half does not hold in general: on simulated profiles (this
entry's 0.6 um tube, another noise draw, at $\lambda$ = 0.25 and 0.5 um; the same tube with a fainter neighbour 1.8 um
away) the global joint minimum of $\Phi$ lay within 0.0011 um of $\theta^*$ in $v_0$ and 0.0055 um in $d$; it slid, to
the far $v_0$ bound, only where the fit had already merged a darker neighbour 1.3 um away and stopped at the $v_0$
bound **[run, sandbox, 2026-10-15]**] **[run, sandbox]** On synthetic profiles blurred by a Gaussian of FWHM
0.5 um (twice Block 5's), Block 5 gave 0.82 um for a 0.6 um tube and 1.47 um for a 1.4 um one; the Gaussian of fitted
width gave 0.70 and 1.39 um (FWHM 0.43 and 0.50 um); a 3.0 um tube came out near 3.0 um for every variant (one noise
draw each). On the 0.8 um synthetic cell (the renderer's blur), the heavy-tailed kernels at the fixed width gave
0.71-0.74 um against Block 5's 0.84-0.86 um: at a given FWHM a heavy-tailed kernel has a narrower core, so a smaller
$d$ does not by itself make a kernel better; the node figure marks, per width, the unweighted fit of least residual.
Where a darker neighbour merged with the dip 1 um away, the unweighted fit took both as one tube and the re-centred
weights at $\lambda$ = 0.25 um moved the fit onto the neighbour. Tripling the quadrature nodes moved a cusped
generalized normal's $d$ by at most 0.003 um. **[relation to I-002]** D-046 tries, on the fit only and as a
diagnostic, the kernel families I-002 proposed for the renderer and the calibration; its skew-normal with a fitted skew
gives, node by node, the asymmetry test I-002 (a) asks for. I-002 stays an idea.

**What it implies for code and tables.** Commit `4759c5f` on `sci/diameter-pipeline`: Cell 4l reads Cell 4j's
`planeiter_<specimen>.json` from `ITER_DIR` and writes `fitvar_<id>.png`, `fitvar_summary_<specimen>.png`,
`fitvar_<specimen>.csv` and `fitvar_<specimen>.json` into `ITER_DIR/fitvar/`; runbook v16. Checks: `test_smoke_fit_variants.py` 8 pass; 30 of 30 mutants of the new code each failed a check (26 at the first run; the re-weighting's stopping rule, the warm starts, the re-weighting's start and a skipped record's reason then got checks of their own); every suite re-run (22, no failure); Cells 4j, 4k and 4l ran from the notebook's own code on two synthetic cells (tubes of 0.8 and 3.0 um): 3 of 3 nodes fitted per cell, the baseline equal to Cell 4j's d_iter on 3 of 3, 0 downloads, 25-72 s per node; a clean clone of the branch from GitHub at `4759c5f` passed the new test 8 of 8 **[run, sandbox]**. No measurement,
estimator hash or table changes.

**What it does not decide [open].** (a) Whether any variant replaces Block 5's fit or enters the correction map. (b)
Whether the fitted widths, should they agree across the thin nodes, point at a blur wider than $\sigma_{\rm fit}$
(D-023) -- a question for the kernel calibration (D-026, Block 10) rather than for the fit. (c) I-002's questions.

**Sources.** **[user]** 2026-10-13 14:50 and the six answers; **[run, sandbox]** commit `4759c5f`; D-023, D-026,
I-002. Literature searched 2026-10-13: PubMed, iteratively re-weighted least squares in non-linear regression (7
records, applications in kinetics, PET and diffusion MRI, none the method's reference; not used), Voigt-profile
fitting (0 records; 46 with the phrase in title or abstract, not opened) and the line-spread function of a defocused
brightfield microscope (0 records, two queries); bioRxiv has no keyword search (one date-and-category query,
bioinformatics, 2026-07-01 to 2026-10-13: nothing relevant). The kernels are scipy's own densities; the Voigt's
parameterisation was checked numerically in the smoke test (the convolution of its two components by quadrature).

**Status:** active; implemented (commit `4759c5f`); not yet run on Allen's data. [updated 2026-10-14] Cell 4l ran on Allen's data: the user sent the figures of nodes 8497, 8496, 3851, 8481 and 8482 (2026-10-14, 15:35) and judged that it "works well" **[user]**. In those figures, at the fitted width the fits of nodes 8481, 8482 and 3851 put most of the dip into the kernel's width (FWHM 1-2 um) and gave $d$ of 0.2-0.6 um, many at a bound **[user's figures, read]**. [updated 2026-10-15] What a fitted width does to $d$ on a thin tube, simulated (this entry's 0.6 um tube under a Gaussian blur of FWHM 0.5 um, noise SD 2 grey levels, 40 draws): Block 5 0.825 +- 0.021 um; the Gaussian at a fitted width 0.525 +- 0.206 um (median 0.61 um), 12 of 40 stopped at $\mu$'s upper bound (20 /um) with $d$ 0.23-0.25 um, the other 28 at 0.33-0.76 um; with $\lambda$ = 0.25 um 0.506 +- 0.214 um, 14 of 40 at that bound. With the width free the objective is nearly flat along $d$ (on one draw its $\Delta\chi^2 \le 1$ interval is 4.7 times wider than at the fixed width) because $d$ trades against $w$, roughly along $\sigma_w^2 + d^2/16$ = const; a single draw (the 0.70 um above) does not show this spread **[run, sandbox, 2026-10-15]**. [updated 2026-10-16] (a) settled by D-052: none of the 100 variants replaces Block 5's fit; D-052's fit -- the three-shape skewed generalized $t$, not among them, its width fitted, $\lambda$ = 0.5 um -- does, and the variants stay diagnostics.

---

## D-047 -- Cell 4m: each fit variant's own Allen-guided iteration, with Cell 4j's line multiplier and round limit

**Date:** 2026-10-14, 15:35 (Europe/Rome), the user's request on Cell 4l's figures, with the answers to three questions.
**Binds:** `scripts/variant_iteration.py` (`iteration_params`, `NodeSquares`, `run_iteration`, `selection`,
`compare_reference`, `run`), `fit_variants.PlaneFits`, `plane_differences.gradient_iteration(..., fit_at=...)`,
`figures.variant_stack_figure`, `figures.variant_planes_figure`; notebook Cell 4m. A diagnostic: no measurement,
estimator hash or table changes; Cell 4j's iteration keeps Block 5's fit.

**Decision [user].** "it works well but I need two things: Let's focus on the following single table of variants that
include every kernel, width fitted, centre weighted and unweighted, edges v0 -+ d/2. I also want that this is
inherented." -- followed by Cell 4j's `LINE_MULT = 3.0` and `MAX_ROUNDS = 5` -- "Secondly I want the figure of the
fitted dendrite (multiple per clarity) at the focus plane and show for each fit the estimated d" (15:35). The answers
given with it: the table is "One 50-row caliper stack"; what inherits `LINE_MULT` and `MAX_ROUNDS` is "The way to
search for the estimated diameter and draw the measuring line given the line_mult"; the variants that drive the
iteration: "Each of the 50, own iteration".

**Statement.** For node $j$ of a Cell 4j record and each variant $V$ at the fitted width (10 kernels of D-046 x
unweighted and $\lambda \in \{0.25, 0.5, 1, 2\}$ um), Cell 4j's rule (D-042) with Cell 4j's own $m$ and $R$ read from
the record: $d^{(0)}_{j,V} = 2 r_{A,j}$; round $i$ draws the line of half-length $h_{j,V,i} = m\, d^{(i-1)}_{j,V} / 2$
through the SWC node and picks $k_{j,V,i} = \arg\max_k G^{(h)}_{j,k}$; a plane already picked ends it; else
$d^{(i)}_{j,V} = \hat d_V(k_{j,V,i})$, $V$'s fit to Block 5's inputs at that plane. Per $V$: `k_iter`, `d_iter`, the
selection line $\pm h^{\rm sel}$ (the last round that picked a plane) and the edges $s_j + \hat v_0 \pm d_{\rm iter}/2$
on it. The reference, Cell 4j's own iteration with Block 5's fit, is redone first through the same squares and must
match the record. Per node: a 50-row stack of the edges (the reference first) with the planes each variant's rounds
picked, and a grid of images, each variant's plane with its line, the fitted dendrite's two edges drawn along the
branch, and its $d$.

**Why [user; assistant's choices and findings].** **[user]** To narrow Cell 4l to one table of the fitted-width
variants, to let each variant choose its own plane and line the way Cell 4j does, and to see each fit on its dendrite.
**[assistant, PROVISIONAL]** $m$ and $R$ are read from each Cell 4j record rather than set in Cell 4m, so that the two
cannot disagree; the fits at a plane are shared by every variant that visits it; a variant whose line a square made
earlier already holds is sampled from the smallest such square (on real data a fetched square's JPEG pixels can
differ slightly from another fetch of the same place); the edges are drawn along the branch, not as a caliper on the
line, because on panels this small a caliper on the line hid the edges (seen on the first render); the panels share one
view; Cell 4j's own iteration is shown as a reference row and panel beside the 50 (not one of them). **[run, sandbox]**
On the synthetic tubes the reference redid Cell 4j's record to the last bit (planes, `k_iter`, `d_iter`, status,
restacks), and every variant ended on Cell 4j's plane: 0.8 um tube, $d$ 0.81-0.86 um, 19 s per node; 3.0 um tube,
$d$ 3.20-3.35 um, 2 planes fitted, 2 min per node.

**What it implies for code and tables.** Commit `9e3296c` on `sci/diameter-pipeline`: Cell 4m reads Cell 4j's
`planeiter_<specimen>.json` from `ITER_DIR` and writes `variter_stack_<id>.png`, `variter_planes_<id>.png`,
`variter_<specimen>.csv` and `variter_<specimen>.json` into `ITER_DIR/variter/`; runbook v17. Checks: `test_smoke_variant_iteration.py` 6 pass, 1 skip (`test_convergence`: nothing new iterates); 19 of 20 mutants of the new code each failed a check (the 20th, a record's planes with a gap accepted, is refused anyway by the square's own plane check); every suite re-run (23, no failure); Cells 4j to 4m ran from the notebook's own code on two synthetic cells (tubes of 0.8 and 3.0 um): Cell 4j's iteration redone identically on 3 of 3 nodes per cell, every variant ending on Cell 4j's plane, 0 downloads, 18-21 s per node on the 0.8 um tube and 2.4-3.8 min on the 3.0 um tube (other jobs running); a clean clone of the branch from GitHub at `9e3296c` passed the new test **[run, sandbox]**. No measurement,
estimator hash or table changes.

**What it does not decide [open].** Whether any variant's iteration replaces Cell 4j's (D-043 carries Block 5's fit
forward); what the variants' fitted widths say about the blur (D-046 (b)); D-043's (a)-(d).

**Sources.** **[user]** 2026-10-14 15:35, the three answers and the five figures of Cell 4l; **[run, sandbox]** commit
`9e3296c`; D-042, D-044, D-046.

**Status:** active; implemented (commit `9e3296c`); not yet run on Allen's data. [updated 2026-10-15] Reading its rows: all 50 variants fit the kernel's width, so on thin dendrites their $d$ is as weakly determined as D-046's update of 2026-10-15 records; in that simulation the smallest fitted $d$ (0.23-0.25 um) were the fits stopped at $\mu$'s upper bound, so on Allen's data a small `d_iter_um` with `mu_hi` (or `w_hi`) in `at_bound` deserves the same reading -- not yet checked on Allen's data **[run, sandbox; reasoning]**. [updated 2026-10-16] Its open point settled by D-052: the iteration that replaces Cell 4j's is the one driven by D-052's fit, not one of the 50 variants'.

---

## D-048 -- The per-cell law fit is weighted by the simulated inverse covariance of its statistics, refined iteratively from a first step weighted by the real data

**Date:** 2026-10-16, 09:29 (Europe/Rome). **Binds:** D-033's open point (d),
"the statistic set and the scales $\omega_m$": the weights only. The per-cell
law fit planned in `Ih Fit/noise_calibration.py` (or a module beside it); the
cost (12)/(13) of `claude/TEEG_Stage7_noise_model_2026-09-28.md` (eq. (14)
there, with the factor $\tfrac12$); the closure smoke of the Stage-7 handoff
§5 item 6; the implementing chat's check plan. Supersedes the hand-set
$\omega_m$ (0.005 mV for SDs, 0.05 for autocorrelations, "configured") as the
fit's default weighting; they survive as the labelled comparison.

**Decision [user].** "I understood the concept of the estimation of the
covariance under the model theta. We would have to check the behaviour with
an in silico simulation to understand how much a not captured (or only
approximated as in the case of the modelled linear creep) noise component
affects the estimation of the law (we will do this during the
implementation). Moreover the estimation is iterative in the sense that we do
an initial guess of theta starting from the weights derived by the real data,
and then iteratively refine the covariance matrix from the simulations R
until the objective plateau."

**Statement.** For each real cell $c$ with statistic vector
$\hat{\mathbf q}(c) \in \mathbb R^d$ (per-cell medians over its $K_m$ windows;
$d$ = 14 in the fit of §3.5.3, or D-034's enlarged set), and for a weighting
matrix $W \in \mathbb R^{d \times d}$ symmetric positive-definite,
$$\hat\theta_W(c) = \arg\min_{\theta \in \Theta}\;\bigl(\mathbf q(\theta) - \hat{\mathbf q}(c)\bigr)^{\!\top} W \bigl(\mathbf q(\theta) - \hat{\mathbf q}(c)\bigr), \qquad q_m(\theta) = \operatorname{med}(T_m \mid \theta, n_m), \tag{D-048.1}$$
the binding function by simulated medians (D-033). The iteration, $k = 0, 1, 2, \dots$:
(i) $W_0$ is built from the real data of cell $c$ alone; (ii) given $\hat\theta_k(c)$,
simulate $R$ cells from the law at $\hat\theta_k(c)$ with $c$'s own window
lengths $n_m$ and window counts $K_m$, through the pipeline's own estimators
(D-028), and set $\hat\Sigma_{k}(c)$ = the sample covariance of their $R$
statistic vectors, $W_{k+1} = \hat\Sigma_k(c)^{-1}$; (iii) refit
$\hat\theta_{k+1}(c) = \hat\theta_{W_{k+1}}(c)$; (iv) stop when the minimised
objective plateaus. The in-silico sensitivity check: twins generated with a
noise component the law (D-017.1) lacks, or only approximates (the linear
tilt of D-034 standing in for whatever carries the variance at seconds), are
fitted with this procedure, and the error in $\hat\theta$ and in its reported
precision is measured, before the fit is trusted on real cells.

**Why.** **[user]**. Recorded with it, from the noise chat of 2026-10-13/14
**[run, sandbox; checks in `TEEG_covariance_checks_2026-10-14.zip`; corrected 2026-10-16: also delivered for commit at `Ih Fit/dev/2026-10-16_covariance_weighting/checks/weights_1013/`]**: with
only $\tau_s$ free and the ten LS statistics ($K$ = 6), the robust SD of
$\hat\tau_s$ over 1500 simulated cells is 16.98 ms with the hand-set
$\omega_m$, 15.53 ms with the diagonal $1/\mathrm{SE}_m^2$ and 7.57 ms with
the full $\Sigma^{-1}$ (statistics within one window correlate at
0.993-0.997, so the diagonal double-counts them); the minimised cost under
$\Sigma^{-1}$ has mean 9.22 against $\chi^2_9$'s 9; a two-step $\hat\Sigma$
from $R$ = 600 cells at the first-step estimate gave 7.25 ms against the
oracle's 7.61 ms. Kaczmarska, Isham & Onof 2015 [corrected 2026-10-16: Kaczmarska, Isham & Northrop 2015, *Environmetrics* 26, 312-325, per the PubMed metadata of PMID 27563266] **[FT, PubMed]** state the
optimal weighting ($\Sigma^{-1}$, after Hansen) and describe the two-step
procedure. The sensitivity check is needed because a noise source the
simulator lacks always *understates* $\hat\Sigma$ along the contrasts it
affects, and $\Sigma^{-1}$ amplifies that: +1 % variance on each statistic
made the weight on the thinnest contrast 5.0 times too large and raised the
mean minimised cost from 8.9 to 18.8 ($\chi^2_9$'s 95 % point is 16.9),
while the robust SD of $\hat\tau_s$ went from 6.3 to 8.7 ms (oracle 7.8 ms)
and the precision reported from $\hat\Sigma$ did not move **[run, sandbox,
`check_missing_noise.py`]**.

**What it implies for code and tables.** $W = \hat\Sigma^{-1}$ by the iteration
is the DEFAULT of the per-cell fit; the hand-set $\omega_m$ and the diagonal
$1/\mathrm{SE}^2$ are opt-in comparisons, labelled in the output columns. The
fit writes, per cell: $\hat\theta$, the minimised objective at the last
iteration (to be read against $\chi^2_{d-p}$ as the misfit alarm), the
number of iterations, $R$, and $\kappa(\hat C)$ of the last correlation
matrix. $\hat\Sigma$ is estimated with the real cell's own $n_m$ and $K_m$,
never from one cell's own windows (rank $\le K-1 < d$). The closure smoke
asserts the default and recovers an injected law inside its Monte Carlo band.

**What it does not decide [open].** How $W_0$ is "derived by the real data":
the cell's own window-level SEs on the diagonal (the assistant's reading),
or the hand-set $\omega_m$ -- to be confirmed by the user. $R$; the plateau
criterion (tolerance on the objective and on $\hat\theta$, iteration cap);
whether the iteration converges for every cell; the statistic set
(D-034's enlarged SS windows); what to do when a cell's minimised objective
fails the $\chi^2$ reading; D-017's other open points.

**Sources.** **[user]**, 2026-10-16 09:29. Kaczmarska et al. 2015,
*Environmetrics*, PMC4975607 **[FT]**; noise model document §3.5
**[KB]**; the sandbox checks of 2026-10-13/14 **[run]**, not in the repo [corrected 2026-10-16: delivered to the user for commit under `Ih Fit/dev/2026-10-16_covariance_weighting/checks/`, with the handoff; in the repo once pushed].

**Status:** active; not implemented. Settles the weights half of D-033's
open (d); the statistic set stays open. **$W_0$ settled by D-049 (2026-10-16, later).**

---

## D-049 -- The first-step weighting $W_0$ is the diagonal of the cell's own window-level bootstrap variances

**Date:** 2026-10-16, 10:16 (Europe/Rome). **Binds:** D-048's open point "how
$W_0$ is derived by the real data": step (i) of the D-048 iteration in the
per-cell law fit (`Ih Fit/noise_calibration.py` or the module beside it); the
seed of the `noise_fit_weighting = sigma_inv` default; the closure check S0 of
the covariance-weighting handoff (§4). Not the weights of any later iteration
(D-048: $W_{k+1} = \hat\Sigma_k(c)^{-1}$).

**Decision [user].** Chosen among three options put by the assistant: "Per-cell
diagonal $1/\mathrm{SE}^2$" -- $W_0 = \mathrm{diag}(1/\widehat{\mathrm{SE}}_m^2)$,
$\widehat{\mathrm{SE}}_m$ from a window-level bootstrap ($B$ = 2000) of the cell's
own $K_m$ per-window values of statistic $m$. Rejected: the hand-set $\omega_m$
(0.005 mV for SDs, 0.05 for autocorrelations), and the diagonal of
$\hat\Sigma(\hat\theta_{\rm sim}; \mathbf n, \mathbf K)$ simulated at the cohort
law with the cell's own windows.

**Statement.** For a real cell $c$ and statistic $m = 1, \dots, d$, let
$T_m^{(1)}(c), \dots, T_m^{(K_m)}(c)$ be the values of statistic $m$ on the
$K_m$ windows of $c$ (computed level; their median is $\hat q_m(c)$, the fit's
target). Draw $B$ = 2000 resamples of size $K_m$ with replacement from
$\{T_m^{(k)}(c)\}_{k=1}^{K_m}$ -- the same resampled window indices for every $m$,
so one draw resamples whole windows -- take the median of each resample, and
let $\widehat{\mathrm{SE}}_m(c)$ be the SD (ddof 1) over the $B$ medians. Then
$$ W_0(c) = \mathrm{diag}\bigl(\widehat{\mathrm{SE}}_1(c)^{-2}, \dots, \widehat{\mathrm{SE}}_d(c)^{-2}\bigr), \qquad \hat\theta_0(c) = \arg\min_{\theta \in \Theta} \bigl(\mathbf q(\theta) - \hat{\mathbf q}(c)\bigr)^{\!\top} W_0(c) \bigl(\mathbf q(\theta) - \hat{\mathbf q}(c)\bigr), \tag{D-049.1} $$
with $\mathbf q(\theta)$ the binding function of (D-048.1). $W_0$ is used for
$\hat\theta_0(c)$ only; for $k \ge 0$, $W_{k+1} = \hat\Sigma_k(c)^{-1}$ (D-048).
The off-diagonal entries of the bootstrap covariance are never used (rank
$\le K_m - 1 < d$, D-048 "What it implies").

**Why.** **[user]**. Recorded with it: (1) the SS statistics have per-cell SEs
3-10 times smaller than the hand-set $\omega_m$ = 0.05 (per-cell median SS
$\hat\rho(1)$: 0.015 with 37 pulses, 0.005 with 348; topic doc §3.8.2 **[KB; run]**),
so the hand-set scale under-weights them by 10-100 in weight; a diagonal from
the cell's own windows gets the SS-against-LS scale right, with the LS
statistics' per-cell SEs (0.004 mV; 0.06-0.09 for the autocorrelations, $K$ = 5-6)
close to the hand-set values **[KB, §3.8.2]**. (2) The window-level bootstrap SE
of a median over $K$ = 6 windows is unbiased but spread 0.58-1.52 of the truth
**[run, `checks/boot_1013/check_boot_types.py`]**; tolerable for a seed, because
the seed does not survive the first iteration: the two-step from the hand-set
$\omega_m$ already matched the oracle (robust SD of $\hat\tau_s$ 7.25 against
7.61 ms; $R$ = 600) **[run, `checks/weights_1013/check_cov_source.py`]**.
(3) $B$ = 2000: the internal (resampling) RMS of a bootstrap SE is 0.019 of its
value at $B$ = 2000 **[run, `checks/boot_err_1014/check_boot_errors.py`]**, far
below the 0.58-1.52 sampling spread, so a larger $B$ buys nothing. (4) The
cohort-law diagonal would have seeded a per-cell fit with a cohort-level
object; D-017 (ii) forbids the cohort law as a generator input only, but the
user chose the cell's own data. **Caveat [reasoning].** A per-cell bootstrap
diagonal was never tested as a seed: the handoff's "diagonal 15.5 ms" is the
diagonal of the *oracle* $\Sigma$ (true SEs), and the two-step check started
from the hand-set $\omega_m$. S0 measures the seed (below).

**What it implies for code and tables.** The per-window statistic values
($K_m \times d$ per cell) are kept, not only their medians, so that one table
gives $\hat{\mathbf q}(c)$ and $W_0(c)$; the bootstrap draws come from a
`numpy.random.Generator` seeded from the config, one stream per cell. The
default `noise_fit_weighting = sigma_inv` means "$W_0$ by this entry, then
$\hat\Sigma^{-1}$" (D-048). Labelled comparisons, opt-in: `diag_se` ($W = W_0$,
no iteration) and `hand_omega`. Degenerate seed (**assistant's provisional
rule, to be confirmed**): if any $\widehat{\mathrm{SE}}_m(c)$ is not finite and
positive (a flat or NaN statistic), the cell is flagged `w0_degenerate` and
that statistic is dropped from $W_0$ (weight 0) for the seed only -- never given
infinite weight -- and the drop is written to the fit's flag column. Smoke:
$\widehat{\mathrm{SE}}_m$ on a planted $K \times d$ table equals an independent
recomputation; whole-window resampling (one index draw for all $m$) asserted; the
flag fires on a planted flat column. S0 runs, on the same twins, the full
iteration from this $W_0$ and from `hand_omega`, so the seed's effect on
$\hat\theta$ and on the iteration count is measured, not assumed.

**What it does not decide [open].** $R$, the plateau tolerances and $k_{\max}$
(handoff §8 item 2); the statistic set and D-034's enlarged SS windows (item 3);
S0-S3 and their predictions (item 4); the degenerate-seed rule above; whether
the LS SEs from $K$ = 5-6 windows should be floored; D-048's and D-017's other
open points.

**Sources.** **[user]**, 2026-10-16 10:16 (choice among three options). Sandbox
checks in the repo at `Passive Features/HPC script/Ih Fit/dev/2026-10-16_covariance_weighting/checks/`:
`check_cov_source.py` and its log (two-step from hand weights 7.25 ms, oracle
7.61 ms, "diag" = oracle SEs 15.61 ms, per-cell bootstrap covariance median
$\kappa$ $8.2 \times 10^6$), `check_boot_types.py` (window-level bootstrap spread),
`check_boot_errors.py` ($B$ = 2000) **[run, 2026-10-13/14; logs read 2026-10-16]**;
`claude/TEEG_Stage7_noise_model_2026-09-28.md` §3.8.2 **[KB]**; the
covariance-weighting handoff §1-§2 (repo, `dev/2026-10-16_covariance_weighting/`).

**Status:** active; not implemented. Settles D-048's $W_0$. **Its open points $R$, the plateau tolerances and $k_{\max}$ settled by D-050 (2026-10-16, later, 2).**

---

## D-050 -- Monte Carlo sizes and stop rule of the iterated fit: $R$ = 600 then 3000, $M$ = 3000, common random numbers per cell, a 1 % rule on $\hat\theta$ alone, $k_{\max}$ = 6 tunable

**Date:** 2026-10-16, 10:30 (Europe/Rome). **Binds:** D-048's open points "$R$;
the plateau criterion (tolerance on the objective and on $\hat\theta$,
iteration cap)" and D-033's open "$M$" (topic doc §3.5.3, §5 (d)): the
iteration of the per-cell law fit (`Ih Fit/noise_calibration.py` or the module
beside it), its config object, its seeds, the output columns `noise_fit_iters`,
`noise_fit_R` and the fit's flag column; the closure check S0.

**Decision [user]**, answering three sub-questions put by the assistant with
the options' evidence: (a) "The recommended one": $R$ = 600 while iterating,
one final step at $R$ = 3000, $M$ = 3000; (b) "theta only at 1 %"; (c) "Start
with $k_{\max}$ = 6 but make it tunable". The user asked, and received, a brief
explanation of $R$ against $M$ and of how the iteration works (transcript of
2026-10-16).

**Statement.** For cell $c$ with $d$ statistics, window lengths $n_m$ and
counts $K_m$, and fit coordinates $\mathbf q(\theta) \in \mathbb R^4$ -- here the
*coordinates* $q = (\log\sigma_f, \rho_f, \log\sigma_s, \log\tau_s)$, not the
statistics $\mathbf q(\theta)$ of (D-048.1); written $\mathbf u(\theta)$ below
to keep the two apart:
(i) **Binding function.** $q^{\rm sim}_m(\theta) = \operatorname{median}_{j=1..M} T_m(\mathbf x_j(\theta; \xi_j))$,
$M$ = 3000 windows of length $n_m$ per statistic, from Gaussian innovations
$\xi_1, \dots, \xi_M$ drawn once per cell from a `numpy.random.Generator` seeded
from the config and the cell id, and reused at every $\theta$ and every
iteration (common random numbers), so that $J_W(\theta)$ is a deterministic
function of $\theta$ for fixed $W$.
(ii) **Covariance.** $\hat\Sigma_k(c)$ is the sample covariance (ddof 1) of the
statistic vectors $\hat{\mathbf q}^{(r)}$, $r = 1..R$, of $R$ cells simulated at
$\hat\theta_k(c)$ with $c$'s own $n_m$ and $K_m$, each $\hat q^{(r)}_m$ the median
over its $K_m$ windows; $R$ = 600 for $k = 0, \dots, k_{\rm stop}-1$. The $R$
cells' innovations are drawn once per cell (a second stream) and reused at
every $k$.
(iii) **Stop rule.** With $\mathbf u_k = \mathbf u(\hat\theta_k)$,
$$ k_{\rm stop} = \min\{k \ge 1 : \max_{i} |u_{k,i} - u_{k-1,i}| \le 0.01\} \wedge k_{\max}, \qquad k_{\max} = 6 \text{ (config, default)}. \tag{D-050.1} $$
$J_k$ is written at every $k$ but not tested.
(iv) **Final step.** At $k_{\rm stop}$: $\hat\Sigma_{\rm fin}$ from $R$ = 3000
cells at $\hat\theta_{k_{\rm stop}}$ (a fresh stream, fixed per cell),
$W_{\rm fin} = \hat\Sigma_{\rm fin}^{-1}$, $\hat\theta(c) = \arg\min_\theta J_{W_{\rm fin}}(\theta)$,
$J(c) = J_{W_{\rm fin}}(\hat\theta(c))$, $\kappa(\hat C_{\rm fin})$; these are the
reported values. A cell with $k_{\rm stop} = k_{\max}$ by the cap is flagged
`no_plateau`, keeps its iterate and its final step, and is never forced or
dropped; its per-iteration trace $(\hat\theta_k, J_k)$ is written.

**Why.** **[user]**. Recorded with it: (1) $R$: the Monte Carlo error of
$\hat\Sigma^{-1}$ moves the cost by a factor in [0.76, 1.36] at $R$ = 600 and
[0.88, 1.14] at $R$ = 3000 ($m$ = 10) **[run, `check_cov_structure.py` A]**; an
error in $W$ costs precision and the $\chi^2$ reading of $J$, not the centre of
$\hat\theta$ at first order (medians 36.9-39.7 ms under four weightings against
the truth 39.7, robust SDs 17.1 to 7.3 ms **[run, `check_cov_source.log`]**), so
600 serves the iterations and 3000 the reported $J$. (2) $M$: the binding
function's Monte Carlo error moves $\hat\theta$ directly; at $M$ = 300 LS
windows the cohort $\hat\tau_s$ spread 35.6-45.6 ms over five CRN draws
**[run, topic doc §3.5.3]**; the error falls as $1/\sqrt M$ -- verified by
running it at $\hat\theta_{\rm sim}$ on the 265 ms window, 30 CRN draws per $M$:
SD of the $\hat\rho(1\,\mathrm{ms})$ binding value 0.0082 at $M$ = 300, 0.0024
at $M$ = 3000 (ratio 3.4; $\hat\sigma$: 3.8; $\sqrt{10}$ = 3.16 predicted, band
[2.4, 4.2]); at $M$ = 3000 that error is a twenty-fifth of the per-cell sampling
SD of the same statistic's $K$ = 6 median (0.059) **[run, `check_M_scaling.py`,
P1-P2 held; 73 s]**. (3) CRN across iterations makes the map
$\hat\theta_k \mapsto \hat\theta_{k+1}$ deterministic, so (D-050.1) tests a fixed
point and not Monte Carlo jitter **[reasoning; the smoke asserts determinism
under a fixed seed]**. (4) $\hat\theta$ alone: at a fixed point $\hat\Sigma$ and
hence $J$ are stationary too, so a $J$ test adds nothing; off a fixed point $J$
moves because $W$ moves, so a $J$ plateau could stop the loop early
**[reasoning]**. 1 % of $\tau_s$ (0.4 ms) is below the binding function's own
error (about 1.2 ms on $\tau_s$ by linear propagation, an inference) **[corrected 2026-10-16: that figure was scaled from the cohort fit under the hand-set $\omega_m$ (topic doc §3.5.3) and does not transfer to $W = \hat\Sigma^{-1}$ (R3); `check_crn_determinism.py` (sent to the user 2026-10-16; $\tau_s$ alone free, the ten LS statistics, $K$ = 6, $R$ = 200, $M$ = 400, one simulated cell at $\tau_s$ = 39.74 ms, no tilt): with every draw frozen per seed the map $\hat\tau_k \mapsto \hat\tau_{k+1}$ is bit-identical when rebuilt from the same seed (P1) and meets the 1 % rule at $k$ = 1-2 for four seeds (P3), but the four seeds' fixed points span 34.46-40.15 ms under $\hat\Sigma^{-1}$ against 35.49-36.54 ms for the $W_0$ (D-049) fit of the same cell; how large the spread is at $R$ = 600/3000, $M$ = 3000 is not measured -- S0 measures it]** and far
below the per-cell SD (7.6 ms), so the rule stops on the arithmetic's fixed
point, not on a statistical tolerance. (5) $k_{\max}$ = 6: the two-step
already captured most of the gain **[run]**, so the plateau should come at
$k$ = 2-3; a cell that does not plateau is a finding, not a failure to force.
Cost **[reasoning, from sandbox timings]**: one binding evaluation at $M$ = 3000
took about 1.7 s on two cores; about 6 min per iteration per cell at 200
evaluations; one job per cell on the cluster.

**What it implies for code and tables.** A config object (`NoiseFitConfig`
or equivalent) carries `R_iter = 600`, `R_final = 3000`, `M = 3000`,
`tol_theta = 0.01`, `k_max = 6`, `seed`; all are command-line knobs of the
fitting script with these defaults, and the defaults are asserted by the
smoke. Three random streams per cell, spawned from `(seed, specimen_id)`:
binding windows, iteration cells, final cells. Columns per cell:
`noise_fit_iters` ($k_{\rm stop}$), `noise_fit_R` (3000 for the reported
values), `noise_fit_J`, `noise_fit_J_pchi2`, `noise_fit_kappa_corr`, the flag
column with `no_plateau`; the trace $(\hat\theta_k, J_k, \kappa_k)$ in a side
file per group. Smoke: determinism (two runs, same seed, bit-identical
$\hat\theta$ and $J$); the stop rule on a planted sequence; the cap and the flag
on a planted non-converging map; the defaults.

**What it does not decide [open].** The statistic set and D-034's enlarged SS
windows (handoff §8 item 3); S0-S3 and their predictions (item 4); the
$\chi^2$ alarm threshold on $J$ (proposal 0.01, to be calibrated by S2); the
optimiser and its evaluation budget (proposal `scipy.optimize.least_squares`
on the residual vector $L(\mathbf q^{\rm sim}(\theta) - \hat{\mathbf q})$ with
$L^{\top}L = W$, multi-start from the closed form); the D-049 degenerate-seed
rule; D-017's other open points.
**[note 2026-10-16, for the optimiser point, no decision changed]** The inner problem at fixed $W$ is a weighted nonlinear least squares in $\theta'$; with $\hat\Sigma_k = CC^{\top}$ (Cholesky), $L = C^{-1}$. Its residual is a median over $M$ windows: continuous, but the window holding the median changes as $\theta'$ moves. `check_fd_jacobian_median.py` ($\tau_s$ alone varied, 265 ms LS window, $\hat\theta_{\rm sim}$, 12 seeds, central differences in $\log\tau_s$) **[run]**: at a step of $10^{-8}$ the same window held the median on both sides in 48 of 48 cases (P1), so the difference is that window's own slope; at $M$ = 3001 its across-seed SD was 2.9 ($\hat\sigma$; predicted $\ge 3$, failed narrowly) and 8.9 ($\hat\rho(1\,\mathrm{ms})$) times that at a step of 0.03 (P2), e.g. 0.053 against 0.006 on a slope of $-0.073$ for $\hat\rho(1\,\mathrm{ms})$; from $M$ = 301 to 3001 it fell to 0.45 ($\hat\sigma$; predicted $\ge 0.5$, failed) and 0.83 ($\hat\rho$) of its value (P3). SciPy 1.18.1's default relative steps are $\epsilon^{1/2}$ ('2-point') and $\epsilon^{1/3}$ ('3-point'), times $\max(1, |x|)$ **[src]**, both below the $10^{-4}$ at which the slope was still mostly one window's. Implication **[reasoning]**: the Jacobian step is part of the optimiser choice -- a few percent in the fit coordinates, through `diff_step` (which is relative to $|x|$ when set), or a smoothed binding function; the four-parameter Jacobian was not tested.

**Sources.** **[user]**, 2026-10-16 10:30. `check_cov_structure.py` A,
`check_cov_source.py`, `check_sigma_theta.py` and their logs in
`Ih Fit/dev/2026-10-16_covariance_weighting/checks/weights_1013/` **[run,
2026-10-14; read 2026-10-16]**; `check_M_scaling.py` with its log, sent to the
user 2026-10-16 **[run]**; topic doc §3.5.3 and §3.8.2 **[KB]**; the
covariance-weighting handoff §2 (repo). GMM weighting and its two-step form:
Kaczmarska et al. 2015 **[FT, per D-048]**; iterated GMM (Hansen, Heaton &
Yaron 1996) **[memory, not read; PubMed returned nothing usable, 2026-10-16]**.

**Status:** active; not implemented. Settles D-048's $R$, plateau criterion
and iteration cap, and D-033's $M$. **[updated 2026-10-18: the optimiser open point settled by D-056 (iii).]**

---

## D-051 -- The statistic set and the estimator of the per-cell fit: enlarged SS windows in place of the 10 ms window, one law per cell for both protocols, the tilt inside the simulator, the per-polarity mean response subtracted

**Date:** 2026-10-16, 10:50 (Europe/Rome). **Binds:** the statistic vector
$\hat{\mathbf q}(c)$ of the per-cell fit and the estimator module that computes
it on data and simulations alike (D-033); D-017 (b) and the statistic-set half
of D-017 (d); D-014 (i); D-034 (ii), its comments (b), (c) and its open points
($\kappa_{\rm tr}$, the bin, $h^*$, whether the enlarged windows join); the
fit's simulator (binding function and $\hat\Sigma$, D-048/D-050); the SS loader
of `noise_calibration.py`; the config object (`kappa_tr`, `transient_bin_ms`,
`h_star_s`).

**Decision [user]**, the recommended option to each of four sub-questions put
by the assistant: (i) "Enlarged SS replaces the 10 ms window"; (ii) "One law
per cell, both protocols"; (iii) "Tilt in the simulator, alternated with
(D-034.1)"; (iv) "Subtract the per-polarity mean response; threshold as flag",
with a request for an extensive explanation of (iv), given in the transcript.

**Statement.**
(i) **Set.** For cell $c$, $d$ = 15 statistics, each the median over its
windows: *SS enlarged* window $[t_{\rm SS}(c), 200)$ ms after each pulse (one
window per stored pulse, the late part of that pulse's stored $(-10, +200)$ ms
window): $\hat\sigma$, $\hat\rho$ at 0.02, 0.1, 1, 10 ms; *LS 265 ms*:
$\hat\sigma$, $\hat\rho$ at 0.02, 0.1, 1 ms; *LS first 50 ms*: $\hat\sigma$;
*LS last 132.5 ms*: $\hat\sigma$, $\hat\rho$ at 0.02, 0.1, 1, 10 ms. The 10 ms
pre-pulse window is not a separate target: it is the last 10 ms of the previous
pulse's enlarged window (consecutive stored windows overlap by those 10 ms
**[src, as recorded in the topic doc §3.7; not re-read this session]**). If no
enlarged window of $c$ passes the transient test ($t_{\rm SS}(c) \ge 190$ ms),
the cell falls back to the 14 statistics of §3.5.3 (the 10 ms window with
$\hat\sigma$, $\hat\rho$ at 0.02, 0.1, 1 ms) and is flagged `ss_enlarged_none`;
$d$ is written per cell.
(ii) **One law.** A single $\theta(c) = (\sigma_f, \rho_f, \sigma_s, \tau_s)(c)
\in \Theta$ is matched to all $d$ statistics of $c$; the twin's SS pulses and LS
sweeps are generated from it. D-014 (i)'s per-protocol AR(1) pair is superseded
for every cell in the noise table (the back-fill of pre-D-017 manifests is
unchanged). $J$ is written in total and split by protocol block, $J_{\rm SS}$
($d_{\rm SS}$ = 5) and $J_{\rm LS}$ ($d_{\rm LS}$ = 10), as components of one
joint fit, not as separate tests.
(iii) **Tilt.** The simulator of the binding function and of $\hat\Sigma$ draws
$b \sim \mathcal N(0, \hat\sigma_b(c)^2)$ per simulated sweep and adds
$b\,(t_j - \bar t)$ to every window of that sweep (D-017.1); the simulated SS
windows are grouped into sweeps as the real cell's are (pulses per sweep and
polarity from the archive's sweep numbers), so windows of one sweep share $b$.
$\hat\sigma_b(c)$ is plugged in from (D-034.1) at $h^*$: for the first binding
function, with $V(h^* \mid \hat\theta_{\rm cf}(c), 0)$ at the optimiser's
closed-form starting point; then, at every iteration $k \ge 0$ of D-050, recomputed
with $V(h^* \mid \hat\theta_k(c), 0)$ and used in the simulations of iteration
$k+1$ and of the final step. $\hat V_c(h^*)$ is computed from same-polarity
pairs of enlarged-window means within one sweep (so that the subtraction of
(iv), a common vector per polarity, cancels exactly in every difference). $\sigma_b$
is never a free parameter of $J$ (D-034). Since $\hat\sigma_{b,k}$ is a function
of $\hat\theta_k$, the $\theta$-only stop rule of D-050 tests the joint fixed
point; $\hat\sigma_{b,k}$ is written to the trace.
(iv) **Mean-response subtraction.** For each polarity $p \in \{\rm dep, hyp\}$
with $K_p$ enlarged windows $\mathbf x_1, \dots, \mathbf x_{K_p} \in \mathbb R^{n}$
on a common time grid from the pulse,
$$ \tilde{\mathbf x}_k = \mathbf x_k - \frac{1}{K_p}\sum_{j=1}^{K_p} \mathbf x_j, \qquad k = 1, \dots, K_p, \tag{D-051.1} $$
and every SS statistic of (i) is computed on $\tilde{\mathbf x}_k$. The
simulator applies (D-051.1) to its $K_p$ simulated windows of the same
polarity grouping before computing the same statistics, so the operation is
part of the estimator on both sides (D-033). $t_{\rm SS}(c)$ is the earliest 5 ms
bin boundary after which $|\bar r_{\rm odd}(t)|$ (topic doc eq. (27)) stays below
$\kappa_{\rm tr}\hat\sigma_{\rm SS}$ in every later bin, with $\hat\sigma_{\rm SS}$ the
cell's median 10 ms pre-pulse SD; it sets the window start and a flag. The
values $\kappa_{\rm tr}$ = 0.1, bin 5 ms, $h^*$ = 0.8 s are **labelled provisional
config defaults** (D-021 style), to be confirmed or changed after S2.

**Why.** **[user]**. Recorded with it. (i) The enlarged windows cut the
per-cell SD of the autocorrelation statistics 1.7-3.4-fold with 2 SS sweeps and
5-9-fold with 17 against the 5-6 LS pre-windows, and see 50-67 % of the slow
variance against 8 % in 10 ms **[KB, topic doc §3.7, §3.8.2]**; keeping the
nested 10 ms window as well would add statistics correlating at > 0.99 with the
enlarged ones and raise $\kappa(\hat C)$ for no information. (ii) Any set that
spans both protocols presumes one law; D-034 (b) already needs it for an
SS-calibrated $\sigma_b$ to serve the LS sweeps **[log]**; the cohort misfit at
SS 0.1-1 ms is about the fast part's shape, not its protocol **[KB §3.5.3]**, so
no evidence calls for per-protocol fast parts yet -- $J_{\rm SS}$ against
$J_{\rm LS}$ will show one if it exists. (iii) A tilt of $\sigma_b$ = 0.08-0.12
mV/s over a 190-265 ms window adds a demeaned variance $(\sigma_b T)^2/12$, 1-2.4 %
of the window variance, and a like share to every $\hat\rho$ **[reasoning,
arithmetic]** -- the order that raised the mean minimised cost from 8.9 to 18.8 in
`check_missing_noise.py` **[run]**; a simulator without the tilt would make the
$\chi^2$ alarm fire on every real cell for a known reason. (iv) Verified by running
it (`check_pulse_residual.py`, $\hat\theta_{\rm sim}$, 190 ms windows, $K$ = 40,
200 cells, common random numbers, P1-P5 all held; 8 s): a deterministic residual
$A\,\mathrm e^{-t/\tau_r}$ with $A = 0.1\,\hat\sigma_{\rm SS}$ contributes at
most 0.031 % of the window's demeaned variance (worst at $\tau_r \approx 0.31\,T$),
against 0.47 % for a residual that kept its amplitude -- **correcting the
assistant's "1 %" in the question** -- and biases no per-cell median by more than
0.009 of its across-cell SD without subtraction (P1, P2): the threshold case is
harmless. The subtraction earns its place elsewhere: with a response of
$10\,\hat\sigma_{\rm SS}$ at the window start and 2 % pulse-to-pulse amplitude
jitter (an assumed level), the unsubtracted medians are off by 9-34 SD and the
subtracted ones by at most 0.022 SD (P3), so (D-051.1) is what lets the window
start before the response has decayed to the noise floor; it removes a
deterministic component of either polarity symmetry, where (27) tests the odd
part only **[reasoning]**; and it is mirrored exactly on the simulation side: on
noise alone the residual windows have variance $(1 - 1/K_p)\gamma(0)$ and
equal-time correlation $-1/(K_p - 1)$ (measured 0.9748 against 0.975,
$-0.021$ against $-0.026$; P4), and with the deterministic response present the
subtracted statistics equal those of the noise alone to $3 \times 10^{-16}$ (P5).
The construction -- the trial-to-trial fluctuation around the mean response to
repeated identical stimuli as the "noise" -- is the standard one in trial-based
neurophysiology (Cohen & Kohn 2011, **[PubMed FT]**, [DOI](https://doi.org/10.1038/nn.2842);
qualitative use).

**What it implies for code and tables.** The estimator module gains:
`enlarged_ss_windows(cell, t_ss)` (loading, by sweep and polarity, on a common
grid), `transient_start(cell, kappa_tr, bin_ms)` ($t_{\rm SS}$ and flags),
`subtract_mean_response(windows_by_polarity)` ((D-051.1); called on data and
on simulations), `statistic_vector(...)` returning $\hat{\mathbf q}$ with its
$d$, $\mathbf n$, $\mathbf K$ and block labels. The simulator simulates sweeps
with the cell's structure (windows per sweep and polarity; $b$ shared within a
sweep; LS sweeps one window each). Config: `kappa_tr = 0.1`, `transient_bin_ms = 5`,
`h_star_s = 0.8`, each documented as provisional; the smoke asserts the
defaults. Columns: `noise_ss_enl_t0_ms` ($t_{\rm SS}$), `noise_fit_d`,
`noise_fit_J_ss`, `noise_fit_J_ls`, `noise_tilt_sigma_b_mV_per_s`, flag values
`ss_enlarged_none`, `transient_fail`. The manifest's law columns (D-017:
`noise_fast_sigma_mV`, `noise_fast_rho_lag1`, `noise_slow_sigma_mV`,
`noise_slow_tau_ms`) apply to both protocols of the twin; `noise_ss_*` of D-014
is back-fill only. Smoke: P4 and P5 as asserted checks; the fallback path;
$b$ shared within a simulated sweep; the overlap of the 10 ms window with the
enlarged window asserted on an archive cell.

**What it does not decide [open].** The final $\kappa_{\rm tr}$, bin and $h^*$
(after S2); the pulse order within a sweep (alternating or blocked; read from
the archive's sweep numbers, **not verified**), which fixes the same-polarity
lags available for $\hat V_c$; the pulse-to-pulse response variability of real
cells (to be measured in S3; 2 % was assumed); whether a slow drift of the
response amplitude across the recording calls for a per-sweep rather than
per-cell mean in (D-051.1); the $\chi^2$ alarm threshold; S0-S3 (item 4);
D-017 (a), (e), (f), (g).

**Sources.** **[user]**, 2026-10-16 10:50. Topic doc §3.4.1, §3.5.3, §3.7,
§3.8.2 **[KB]**; D-014, D-017, D-033, D-034 **[log]**; the archive's Format A
and the 10 ms overlap **[src, via the topic doc; not re-read this session]**;
`check_pulse_residual.py` with its log, sent to the user 2026-10-16 **[run]**;
`check_missing_noise.py` **[run, 2026-10-14]**; Cohen, M. R. & Kohn, A. (2011),
Measuring and interpreting neuronal correlations, *Nat Neurosci* 14, 811-819,
PMC3586814 **[PubMed FT; read 2026-10-16; qualitative]**.

**Status:** active; not implemented. Settles D-017 (b) and the statistic-set
half of D-017 (d); supersedes D-014 (i) for measured cells; settles D-034's
"whether the enlarged windows join" and fixes its $\kappa_{\rm tr}$, bin and
$h^*$ as provisional defaults. **[updated 2026-10-18: (iii)'s "windows of one sweep share $b$" sharpened by D-056 (i) -- each simulated SS sweep is one continuous trace with the real pulse timing, so the slow component's carry-over between adjacent windows ($e^{-t_{\rm SS}/\tau_s}$) is reproduced as well as $b$. Found on 2026-10-18 when the assistant's "independent windows" statement in the 3d explanation was challenged; the 2026-10-16 check `check_pulse_residual.py` used independent windows.]** **[updated 2026-10-21: (ii)'s "the twin's SS pulses and LS sweeps are generated from it" superseded by D-073 -- per-sweep calibration, form open; in the assistant's proposal the per-cell fit stays as the pooled stage. D-072: no shape test in the simulator.]** **[updated 2026-10-28: (iv)'s window-start test (5 ms bins of $\bar r_{\rm odd}$ against $\kappa_{\rm tr}\hat\sigma_{\rm SS}$) superseded by **D-095** -- $t_{\rm SS}(c)$ from the exponential decay of the mean windows, no binning -- and (D-051.1)'s group is now the sweep and polarity, on data and in the simulator alike; (i)'s fallback at 190 ms is an open point of D-095.]** **[updated 2026-10-28 (later, 9): (i)'s fallback at 190 ms (`ss_enlarged_none`) replaced by 80 ms, flagged (`ss_start_fallback`) -- **D-103**.]**

---

## D-052 -- The diameter measurement the bias table is built for: the Allen-guided plane and a three-shape kernel with its width fitted, centre-weighted at $\lambda$ = 0.5 um

**Date:** 2026-10-16, 10:43 (Europe/Rome), the user's statement, with the answers to three questions asked next.
**Binds:** Block 5's per-node chain (`analysis/node_pipeline.measure_node`: today the plane $k^*$ of D-035 and the
Gaussian fit `fit.fit_profile` at $\sigma_{\rm fit}$, D-019.1, D-023), so the measurement of every real node and of
every phantom (`analysis/phantoms.run_replicate`), hence the bias table of D-024 and D-031 and its inversion (Blocks
6-7); the Allen-guided iteration of D-042 to D-044 (`scripts/plane_differences.py`: `iterate_planes`,
`gradient_iteration`, `refit_at_plane`, `node_stack`, `line_profiles`, `block_background`); `model/blur_kernels.py`
and `analysis/fit_variants.py` (whose `_Layout` holds one shape parameter); `MeasureConfig` and
`config.signature_hash("estimator")`; the bound rule of $\mathcal S$ (D-024, as amended by D-030).

**Decision [user].** "Perfect but I would argue that the generalized fitted distributions actually behave better than
the gaussian. For what regards the errors we would have to leverage on the correction table that we are going to
implement in the next chat. So what has been settled in this chat is the way to choose the focal plane and the noise
kernel used and the fitting procedure" (10:43, on the assistant's Monte Carlo of 2026-10-15). To the three questions
then asked: the kernel, "3-shape family (SGT)" -- $\beta$, $\xi$, $\zeta$ and $w$ fitted per node; the weighting,
"Centre-weighted, λ = 0.5 µm" -- re-centred by IRLS as in D-046; the focal plane, "The settled fit's own" -- the
Allen-guided iteration with each round's line sized from this fit's $d$, as Cell 4m runs it per variant (D-047).

**Statement.** For a dendrite node $j$, real or the middle node of a phantom: $r_j$ is the radius its branch carries
(um; Allen's on a real node), $k_{{\rm SWC},j}$ the plane nearest its traced depth, $v_n$ ($n = 1, \ldots, N$) Block
5's profile offsets (um; $N$ = 53 at `profile_half_um` = 3 um and `profile_step_um` = 0.1144 um), $\varphi_j$ and the
measuring axis those of Block 5's pilot at $k^*_j$ (its centre, heading and tilt; D-042), $I_j(v_n; k)$ the profile
in plane $k$ and $\bar B_j(k)$ the D-018.1 background of plane $k$ (gl).

(D-052.1) **Kernel.** With $B$ the beta and $\Gamma$ the gamma function, for $\kappa = (\beta, \xi, \zeta)$ and every
$x \in \mathbb R$: $h_{\beta,\xi}(x) = \beta\,\xi^{1/\beta}\,[2B(1/\beta, 1/\xi)]^{-1}(1 + \xi|x|^\beta)^{-(1/\xi +
1/\beta)}$ for $\xi > 0$ and $h_{\beta,0}(x) = \beta\,[2\Gamma(1/\beta)]^{-1} e^{-|x|^\beta}$; the two-piece density
$f(x \mid \kappa) = 2(\zeta + 1/\zeta)^{-1} h_{\beta,\xi}(x/\zeta)$ for $x \ge 0$ and
$2(\zeta + 1/\zeta)^{-1} h_{\beta,\xi}(\zeta x)$ for $x < 0$ (unit integral, mode 0); its FWHM
$W(\kappa) = (\zeta + 1/\zeta)\,[(2^{\xi\beta/(\beta + \xi)} - 1)/\xi]^{1/\beta}$ for $\xi > 0$ and
$(\zeta + 1/\zeta)(\ln 2)^{1/\beta}$ for $\xi = 0$; and the kernel of FWHM $w$ (um),
$g_{w,\kappa}(z) = (W(\kappa)/w)\, f(W(\kappa)\, z / w \mid \kappa)$ for every $z \in \mathbb R$. Members: $(2, 0, 1)$
the Gaussian, $(1, 0, 1)$ the Laplace, $(2, 2/\nu, 1)$ Student's $t_\nu$ (so $(2, 2, 1)$ the Lorentzian),
$(\beta, 0, 1)$ the generalized normal, and for $\zeta \ne 1$ their two-piece skewed versions (right-to-left
half-width ratio $\zeta^2$).

(D-052.2) **Fit at plane $k$.** With $\theta = (d, \mu, v_0, w, \beta, \xi, \zeta)$ and $\alpha = \mu d / \cos\varphi_j$,
$I_{\rm model}(v \mid \theta; \varphi_j, \bar B_j(k)) = \bar B_j(k)\,[1 - \tfrac d2 \int_{-\pi/2}^{\pi/2}
(1 - e^{-\alpha\cos\psi}) \cos\psi\; g_{w,\kappa}(v - v_0 - \tfrac d2 \sin\psi)\, d\psi]$ (D-046's model,
`tube_model.model_profile_kernel`), and
$\hat\theta_j(k) = \arg\min_{\theta \in \Theta} \tfrac12 \sum_{n=1}^{N} \omega_n\,[(I_{\rm model}(v_n \mid \theta;
\varphi_j, \bar B_j(k)) - I_j(v_n; k)) / \bar B_j(k)]^2$ over $\Theta$: $d \in [0.05, 6]$ um, $\mu \in [0, 20]$ /um,
$v_0 \in [-1, 1]$ um, $w \in [0.1, 2]$ um, $\beta \in [1, 10]$, $\xi \in [0, 2]$,
$\zeta \in [e^{-0.7}, e^{0.7}] \approx [0.50, 2.01]$, searched in $(\log_{10}\beta, \xi, \ln\zeta)$. Weights: first
$\omega_n \equiv 1$ (the unweighted fit); then $\omega^{(t)}_n = \exp(-|v_n - \hat v_0^{(t)}| / \lambda)$,
$\lambda$ = 0.5 um, $\hat v_0^{(t)}$ the previous iterate's centre, until $d$ and $v_0$ both move by at most
$10^{-4}$ um, at most 30 iterations (D-046's IRLS). Solver `scipy.optimize.least_squares` (`trf`, `x_scale="jac"`,
`ftol = xtol = gtol` = `fit_tol` = $10^{-10}$); quadrature $N_q = \max(64, \lceil 6\,d_{\max}/\sigma_{\min}\rceil)$
= 848 Gauss-Legendre nodes ($d_{\max}$ = 6 um; $\sigma_{\min} = 0.1/(2\sqrt{2\ln 2})$ um, the Gaussian SD at the
smallest admissible $w$; `fit_variants.n_nodes_for`). Start points, PROVISIONAL (the sandbox check's): the Gaussian
fit of fitted width with each of the shapes $(2, 0, 1)$, $(1, 0, 1)$, $(5, 0, 1)$, $(2, 1, 1)$, and Block 5's
$(d, \mu, v_0)$ at $w$ = 0.7 um with the Gaussian shape; the least objective wins.

(D-052.3) **Plane.** $d^{(0)}_j = 2 r_j$; round $i = 1, \ldots, 5$ takes the line of half-length
$h_{j,i} = m\,d^{(i-1)}_j / 2$, $m$ = 3 (D-044), through the traced node across the pilot's heading, and
$k_{j,i} = \arg\max_{k \in K_j} G^{(h_{j,i})}_{j,k}$, with $G^{(h)}_{j,k}$ the gradient energy of plane $k$'s profile
over that whole line (D-040, D-042; `focus.plane_gradient_energies` with `block_background`'s $B_k$) and
$K_j = \{k_{{\rm SWC},j} - 6, \ldots, k_{{\rm SWC},j} + 6\}$; a plane already picked ends the iteration, else
$d^{(i)}_j = \hat d_j(k_{j,i})$ of (D-052.2). Output: $k^{\rm it}_j$, the last plane picked, and
$\hat\theta_j(k^{\rm it}_j)$ -- the node's $\hat d_j$, $\hat\mu_j$, $\hat v_{0,j}$, $\hat w_j$, $\hat\kappa_j$ -- with
$\hat\alpha_j = \hat\mu_j \hat d_j / \cos\varphi_j$.

(D-052.4) **Where it applies [assistant's reading, required by procedure §3.7].** The same chain measures the real
nodes and the phantoms, so the table is $\hat m(d, \varphi, \hat\alpha \mid \mathcal C)$ (D-031) of this estimator, and
a real node's $\hat d_j$ of (D-052.3) is the $\hat d_i$ its inversion solves for (D-024 (vi), D-030). The error left
in a node's $\hat d_j$ is the table's to correct in the mean and to report ($\hat\tau$, failure rate, flags)
**[user]**.

**Why [user; run, sandbox].** **[user]** The generalized fitted kernel behaves better than the Gaussian; the errors are
left to the correction table. **[run, sandbox, 2026-10-15]** Simulated profiles at Block 5's 53 offsets: tubes of 0.6
and 1.5 um, $\mu$ = 1 /um, $v_0$ = 0.1 um, $\varphi$ = 0.2 rad, $\bar B$ = 180 gl, blurred by kernels of FWHM 0.5 um
(the Gaussian, Student's $t_4$ = sgt(2, 0.5, 1), the generalized normal of $\beta$ = 4), Gaussian noise of SD 2 gl,
40 draws per truth; median $\pm$ SD of $\hat d$ (um), in brackets the fits stopped at $\mu$'s upper bound:

| truth | this fit ($\lambda$ = 0.5 um) | SGT, unweighted | Gaussian, $w$ fitted | Block 5 (Gaussian, $w$ 0.233 um) |
|---|---|---|---|---|
| 1.5 um, Gaussian | 1.498 ± 0.012 | 1.499 ± 0.012 | 1.499 ± 0.011 | 1.583 ± 0.007 ($t_{100}$) |
| 1.5 um, $t_4$ | 1.493 ± 0.015 | 1.494 ± 0.015 | 1.530 ± 0.014 | 1.639 ± 0.012 |
| 1.5 um, gen. normal 4 | 1.500 ± 0.012 | 1.499 ± 0.011 | 1.495 ± 0.011 | -- |
| 0.6 um, Gaussian | 0.579 ± 0.182 [8] | 0.564 ± 0.182 [8] | 0.624 ± 0.215 [14] | 0.830 ± 0.019 ($t_{100}$) |
| 0.6 um, $t_4$ | 0.539 ± 0.195 [16] | 0.511 ± 0.194 [15] | 0.231 ± 0.083 [39] | 0.914 ± 0.028 |

Block 5's column is the Student-only run's (its own draws, seed 20261016, truths $t_{100}$ and $t_4$). On thick tubes
the SGT removes the Gaussian's tail bias (1.530 to 1.493-1.494 um on the $t_4$ truth); on thin tubes it stops at
$\mu$'s upper bound (20 /um, $\hat d$ 0.23-0.25 um) less often than the Gaussian of fitted width (8 against 14, 15-16
against 39, of 40), but its SD stays 0.18-0.20 um on a 0.6 um tube. The impostor search says why: at 1 gl, 9 pairs
(truth, $d$) with $|d/d_{\rm true} - 1| \ge 0.2$ reach $\Delta\chi^2_{\min} < 1$ on the 0.6 um truths (17 at 2 gl),
while on the 1.5 um truths $\Delta\chi^2_{\min} \ge 20.8$ at $\pm 5$ % and $\ge 67.6$ at $\pm 10$ % for all five truth
kernels (the Laplace weakest) [corrected 2026-10-16: these are the smallest values a local multi-start search found, so upper bounds on the true minima -- no impostor was found below them, which does not exclude one; the thin-tube impostors, being found, are certain, and the thick tubes' identification rests on that search and on the Monte Carlo's SD of 0.011-0.015 um]; and a larger family can only add impostors, the minimum over a larger set being no
larger **[reasoning, exact]**. So the user's "behave better" holds against the Gaussian of fitted width -- no tail
bias on thick tubes, fewer collapses on thin ones -- and the thin-node spread is what the table has to report, since a
table removes bias, not scatter **[reasoning]**.

**What it implies for code and tables.**
- Two estimator changes in Block 5 -- the plane (the iteration of (D-052.3), after the pilot) and the final fit
  ((D-052.2)) -- each a set of new `MeasureConfig` fields, so `signature_hash("estimator")` changes and every table
  built before is refused (procedure §3.11, `BiasTable.check_estimator`); gate 2 (passed 2026-10-06 at estimator hash
  `535847535e83cf7d`, under the dip-depth focus rule) is re-run under the new hash.
- The iteration's functions move from `scripts/plane_differences.py` into `src/allen_diameter/analysis/`, `node_stack`
  decoupled from `survey.node_planes` (the real-data SWC machinery), so that `run_cell.py` and
  `phantoms.run_replicate` call one code path (procedure §3.7, "unchanged").
- `blur_kernels` gains the SGT family (reference implementation `sgt_kernel.py` of 2026-10-15) and
  `fit_variants._Layout` a shape vector. The decided fit is the default; the Gaussian at $\sigma_{\rm fit}$ at $k^*$
  (Block 5 today), the unweighted SGT and the Gaussian of fitted width are labelled comparisons.
- $\mathcal S$'s "no parameter at a bound" (D-024) cannot be applied as written to the shape parameters: $\xi = 0$ is
  the family's generalized-normal edge, where 20 of 40 unweighted and 26 of 40 weighted fits of the thick Gaussian
  truth ended **[run, sandbox]**; see (b).

**What it does not decide [open].** (a) The phantoms' radius: a real node starts the iteration at Allen's $2 r_j$, a
phantom at its true $d$ (`phantoms.phantom_branch`: `Branch.from_points(xyz, 0.5 * draw.d_um)` **[src]**); the same
radius sets the pilot's gradient-energy window (`focus_grad_margin_um`: "Allen's on real nodes, the true one on
phantoms" **[src: config.py]**), the D-018.1 mask and `block_background`'s mask -- D-043 (a)'s second half. (b) Which
bound hits fail $\mathcal S$: $\mu$ at 20 /um (the thin-node collapses), $w$ at a bound, the shape parameters at
theirs. (c) The plane range $K_j$ (D-043 (b)) and the fit's $\pm 3$ um profile on thick nodes (D-038, D-043 (c)).
(d) D-023's $\sigma_{\rm fit}$ axis, which no longer sets the final fit's kernel but still sets Block 5's pilot fit
(whose centre, heading and tilt the iteration uses) and the renderer's consistency check (procedure §3.10, first
row). (e) Whether the pilot's path pass (the centres of handoff Eq. 3, from Block 5's Gaussian fit at each node's
$k^*$) keeps the Gaussian. (f) The renderer's kernel, Gaussian (D-026); I-002 stays open for the renderer and the
calibration. (g) A thin-node variant with the kernel pinned from thick nodes, as a labelled comparison only (in the
sandbox, kernel from one thick node: 0.614 ± 0.052 and 0.604 ± 0.060 um on the 0.6 um Gaussian and $t_4$ truths,
0.607 ± 0.053 and 0.601 ± 0.058 um with $\lambda$ = 0.5 um, smallest 0.42 um **[run, sandbox]**). (h) The
PROVISIONAL readings: the start points, the IRLS cap, D-042's.

**Sources.** **[user]** 2026-10-16 10:43 and the three answers. **[run, sandbox, 2026-10-15]**
`identifiability_sgt.py` (seed 20261017: the impostor search at 1 and 2 gl, the Monte Carlo above),
`identifiability_check.py` (seed 20261016: the Student-only run, Block 5's column), `sgt_kernel.py` (unit integral,
FWHM and members checked numerically), `independent_model_check.py` (direct quadrature against
`model_profile_kernel`: within 6e-6 gl on three cases, 8e-4 gl on the cusped Laplace kernel at 1.5 um, over its 1e-4
gl tolerance, changing that $\Delta\chi^2$ by 0.002), `sgt_pinning_check.py`; sent to the user as
`identifiability_2026-10-15.zip` and `flexible_kernel_2026-10-15.zip`. **[src]** `sci/diameter-pipeline` at
`9e3296c`: `scripts/plane_differences.py`, `analysis/fit_variants.py`, `model/blur_kernels.py`,
`model/tube_model.py`, `analysis/node_pipeline.py`, `analysis/phantoms.py`, `config.py`. **[PubMed FT]** Simpson, M.
J. & Maclaren, O. J. (2023), a profile-likelihood workflow for identifiability, *PLoS Comput Biol*,
doi:10.1371/journal.pcbi.1011515, PMC10566698 (read 2026-10-15). **[PubMed, abstract only]** Raue, A. et al. (2009),
*Bioinformatics*, doi:10.1093/bioinformatics/btp358. The family's standard references (McDonald & Newey 1988;
Theodossiou 1998; Fernández & Steel 1998 for the two-piece skewing) are not in PubMed: named from memory, not
verified; the density is written out above and was checked numerically. PubMed "skewed generalized t": one record
(Lian et al. 2024; abstract only, not used); "generalized t distribution": 8 records, Kim et al. 2019 read in full, a
different family, not used. bioRxiv: two category queries (bioinformatics, biophysics; last 30 days), nothing
relevant.

**Status:** active; not implemented. Settles D-043 (a)'s first half (the iteration enters the measurement inside
Block 5, for every node, phantoms included), D-046 (a) (no D-046 variant replaces Block 5's fit; this fit, not among
its 100, does) and D-047's open point (the iteration that replaces Cell 4j's is the one this fit drives). The handoff
for the table chat is `Passive Features/Diameter Re-measurement/docs/TEEG_diameter_correction_table_handoff_2026-10-16.md`
(repo, D-020). [updated 2026-10-16] The handoff is committed (`ea49181` on `sci/diameter-pipeline`), with the checks behind this entry in `checks/estimator_2026-10-15/`. Not settled either: the fit's per-solve `max_nfev` -- the two unconverged weighted fits of the sandbox Monte Carlo (thick $t_4$ truth, draws 36 and 39) were solves that hit 3000, not the 30-solve cap **[run, sandbox: draw 39 recomputed 2026-10-16]**. [updated 2026-10-18, later] (D-052.1) and (D-052.2) are implemented as Block 12 on `sci/diameter-pipeline` (commit `bee1e0b`, local, not yet pushed [updated 2026-10-18, later 6: pushed, `origin/sci/diameter-pipeline` at `d643aaf`]): `model/blur_kernels.py` (the kernel, not one of `FAMILIES`), `analysis/fit_sgt.py` (`fit_plane`: the Gaussian starts of `fit_variants`, the unweighted fit, the IRLS), the `sgt_*` fields of `MeasureConfig`; the five Monte Carlo draws of the smoke test reproduced bit for bit **[run, sandbox]**. (D-052.3), the plane inside Block 5, is Block 13. Found in Block 12 and open: the quadrature rule of (D-052.2) converges algebraically on the non-analytic members (SPEC §8; $\hat d$ bias <= 7.5e-4 um in the fitted regime). [updated 2026-10-18, later 4] (a), the phantoms' radius, is settled by **D-058** ($d_n/2$ plus zero-mean normal noise), not by D-053 (i)'s check. [updated 2026-10-20] The implementation points of Block 13 are settled by **D-061**: one provider call plus one per restack, the pilot and the iteration as separate functions in one chain, the flags of $\mathcal S$ on $k^{\rm it}$, and Block 10's selection by Allen's diameters; (e), the iteration's background scene, awaits the user. [updated 2026-10-20, later] (f) is settled by **D-066** [user]: the renderer's kernel becomes the SGT family; (g), the pinned-kernel comparison, is not run for now (**D-065.4**). [updated 2026-10-20, later 6: (D-052.3) implemented with (D-052.2) inside the measurement, Block 13 (`measure_node` = the pilot, the iteration, `plane_iteration.sgt_at_plane` at $k^{\rm it}$; commits `b5191c8`-`70136ca`, local, not pushed)] [updated 2026-10-21: pushed at the user's agreement, `84ea91f..888452a` on `sci/diameter-pipeline`]

---

## D-053 -- The phantoms' radius is settled by a sensitivity check; the bound rule of $\mathcal S$ under the SGT fit; the third table axis is $\ln\hat\alpha$

**Date:** 2026-10-16, about 13:05 (Europe/Rome), the user's answers to the first card of the table chat (Q1, Q2 and Q4a of
`docs/TEEG_diameter_correction_table_handoff_2026-10-16.md` §7). **Binds:** (i) `analysis/phantoms.phantom_branch`
(`Branch.from_points(xyz, 0.5 * draw.d_um)`) and the four places of the chain the node's radius enters (the iteration's
start $d^{(0)}_j = 2 r_j$, the pilot's focus window $|v| \le r_j + 0.5$ um, the $\bar B_j(k)$ mask of D-018.1,
`block_background`'s mask; handoff §6), hence the simulation hash; (ii) the selection $\mathcal S$ of D-024 as amended
by D-030 (`phantoms.reject_reasons`, `SELECTION_FLAGS`, `fit_status`), for phantoms and real nodes alike; (iii) the
regression axes of D-031 (`table.table_inputs`, `BiasTable`, `invert.invert_node`, `config.validate`); Blocks 13-14 of
the handoff's §8.

**Decision [user].** (i) Q1: "DO (C) with also images that compares the diameters on the in focus image plane". (ii) Q2:
option (A), "d, mu, v0, w at a bound, or IRLS not converged". (iii) Q4a: "ln(alpha-hat)".

**Statement.**

(D-053.1) **The phantoms' radius.** Before Block 14, the new chain (D-052: pilot, Allen-guided iteration, SGT fit) is
run in the sandbox on rendered phantoms of $d \in \{0.5, 1, 2, 3\}$ um, each measured three times, with the radius its
branch carries set to $r \in \{d/2,\ 0.25\ \text{um},\ d\}$ (the true radius; a fixed Allen-like one, since Allen's $2r$
is about 0.5 um in every bin of $\hat d$ on 529878215, SPEC §8; one twice too large), the same radius used in all four
places it enters. Read per phantom: $k^{\rm it}_j$ and $\hat d_j$ against their SE over the replicates. If, at every
$d$, $k^{\rm it}$ is the same plane and $\hat d$ agrees within its SE across the three radii, the table's phantoms keep
the true radius $d/2$ (handoff Q1 (A)); otherwise each phantom draws an Allen-like radius from the specimen's Allen
radius distribution, independently of $d$, used in all four places, as a phantom draw inside the simulation hash
(Q1 (B)). **[user]** The check also produces, for each (phantom, radius), an image of the in-focus plane with the
measured diameter drawn on it (the measuring line through the node, the fitted edges $\hat v_0 \pm \hat d/2$ as in
Cell 4k, D-045), so that the three radii's results can be compared by eye on the same plane.

(D-053.2) **The bound rule of $\mathcal S$.** For a fit $\hat\theta_j(k^{\rm it}_j) = (\hat d, \hat\mu, \hat v_0, \hat w,
\hat\beta, \hat\xi, \hat\zeta)$ of (D-052.2): the fit fails $\mathcal S$ when $\hat d$, $\hat\mu$, $\hat v_0$ or
$\hat w$ lies at a bound of its box ($d \in [0.05, 6]$ um, $\mu \in [0, 20]$ /um, $v_0 \in [-1, 1]$ um,
$w \in [0.1, 2]$ um), or when the IRLS of (D-052.2) did not converge (its 30-solve cap, or a weighted solve that
failed, e.g. at `max_nfev`); a shape parameter at its box edge ($\beta \in [1, 10]$, $\xi \in [0, 2]$,
$\zeta \in [e^{-0.7}, e^{0.7}]$) does not fail $\mathcal S$. So the thin-node collapses ($\hat\mu$ = 20 /um,
$\hat d$ 0.23-0.25 um) count in the failure rate, and a real node that collapses is filled (D5). The rest of
$\mathcal S$ (D-024 as amended by D-030: registration on the branch, not `faint`, `crossing`, `stack_edge`,
`bbar_few`, `profile_nan`; never tilt, never `dark`) is unchanged. [updated 2026-10-28: `faint` leaves $\mathcal S$ too, **D-096**.]

(D-053.3) **The third axis.** The response surface of D-031 is fitted in the coordinates
$x = (\ln(d/\text{um}),\ \varphi/\text{rad},\ \ln\hat\alpha)$ with $\hat\alpha = \hat\mu\hat d/\cos\varphi$ the fitted
optical depth of the same node (computed level), $\hat\alpha > 0$ required (a fit with $\hat\mu$ = 0 is at a bound and
fails $\mathcal S$ by (D-053.2), so $\ln\hat\alpha$ is finite on every retained replicate); the coverage is still
reported in $\hat\alpha$ (D-030), and a real node is corrected at its own $\ln\hat\alpha_i$.

**Why [user; reasoning].** (i) **[user]** the check, and the images that make its verdict readable. **[reasoning]** The
radius enters the measurement in four places (handoff §6), and a phantom's exact radius is a systematic error the table
cannot see; the check measures whether it moves anything before the design is changed. (ii) **[user]** (A).
**[run, sandbox; D-052]** $\xi = 0$ is the generalized-normal edge of the family, where 20-26 of 40 thick Gaussian-truth
fits end, so D-024's literal rule would fail half the thick fits; a collapsed fit's $\hat d$ is set by the $\mu$ bound
almost whatever the true $d$, so keeping it (handoff Q2 (B)) would give the table a flat region the inversion cannot
use. Under (A) the sandbox Monte Carlo keeps 32 of 40 and 24 of 40 thin fits, nearly unbiased in the mean (0.603 and
0.597 um on a 0.6 um tube). (iii) **[user]**. **[reasoning]** $\hat\alpha$ spans decades (phantom $\mu d$ from 0.06 to
12, times $1/\cos\varphi$), and the thin-plate spline spaces its knots by Euclidean distance in the input coordinates,
so the log keeps the faint end from being compressed.

**What it implies for code and tables.**
- (i) A check script under the workstream (`checks/` or `scripts/`), run before Block 14, with its figures; its verdict
  decides whether `PhantomConfig` gains a radius draw (`radius_mode = "true" | "allen_draw"`, the draw from Cell 6's
  radius CSV) -- the default of Block 14 is set by that verdict and logged as its own decision.
- (ii) `phantoms.reject_reasons` reads which parameter is at a bound (the fit result carries a per-parameter
  `at_bound` vector, Block 12) and the IRLS status (`irls_ok`), and fails $\mathcal S$ on $d$, $\mu$, $v_0$, $w$ and on
  `irls_ok = False` only; a smoke check asserts that a fit with $\xi$ = 0 or $\beta$ = 1 stays in $\mathcal S$ and
  one with $\mu$ = 20 /um leaves it.
- (iii) `table.table_inputs(d, phi, alpha)` returns $(\ln d, \varphi, \ln\hat\alpha)$; the table file stores the
  $\hat\alpha$ range; `invert.invert_node(d_hat, phi, alpha_hat, table, cfg)` evaluates $\hat m$ at
  $\ln\hat\alpha_i$; `tau_kernel_bandwidth` gains a third value in $\ln\hat\alpha$ (its value open, Q4c).

**What it does not decide [open].** The axes' relative scaling (Q4b: range-normalised, or per-axis scales chosen by
cross-validation) -- explained in the same turn, the user's answer pending; the third bandwidth's value; the failure-rate
threshold `max_failure_rate` (0.2 PROVISIONAL), which (ii) makes bind on thin nodes; `failure_rate_ignore` (Q3); the
per-solve `max_nfev`; Q5-Q10 of the handoff; which Allen-like radius the check's middle case should use if 0.25 um
proves unrepresentative (Cell 6's distribution not yet read).

**Sources.** **[user]** 2026-10-16, about 13:05 (the first AskUserQuestion card). **[KB]** D-024, D-030, D-031, D-043,
D-052 (this log); `docs/TEEG_diameter_correction_table_handoff_2026-10-16.md` §4.4, §6, §7; `specs/SPEC.md` §2.2,
§2.4, §8. **[src]** `sci/diameter-pipeline` at `ea49181`: `analysis/phantoms.py`, `analysis/table.py`,
`analysis/invert.py`, `config.py`. **[run, sandbox]** the 23 smoke suites at `ea49181` (all pass, 2026-10-16).

**Status:** active; not implemented. Settles D-052 (b) and the procedure of D-052 (a) / D-043 (a)'s second half;
settles the coordinate half of D-031's open point. [updated 2026-10-18, later] (ii) is implemented as the status rule of Block 12's fit (`fit_sgt`, commit `bee1e0b`): `fit_status` is `at_bound` only for $d$, $\mu$, $v_0$, $w$, so Block 6's `fit_status == "converged"` applies it unchanged; (i) and (iii) not yet. [updated 2026-10-18, later 4] (i) is superseded by **D-058** [user]: each phantom's radius is $d_n/2$ plus zero-mean normal noise, and the sensitivity check is no longer the gate before Block 14 (whether it still runs as a diagnostic, with its images, is D-058's open point). [updated 2026-10-20, later] (iii)'s coordinate is amended by **D-063.1** [user]: the third axis is $\ln(\hat\mu\hat d)$, the tilt-free fitted optical depth, in place of $\ln\hat\alpha$; $\hat\alpha$ keeps the dark label (D-030).

---

## D-054 -- The table's inputs are range-normalised; the third bandwidth; the failure-rate threshold waits for the pilot; the SGT fit's per-solve budget is 3000

**Date:** 2026-10-18, about 11:15 (Europe/Rome), the user's answers to the second card of the table chat (handoff §7
Q4b and Q4c, the consequence raised under Q2, and the `max_nfev` row of §3.5). **Binds:** (i) `analysis/table.table_inputs`
and `BiasTable` (the scaling of the spline's inputs, stored in the table file with the ranges); (ii)
`CorrectionConfig.tau_kernel_bandwidth` (three values) and `config.validate`; (iii) `CorrectionConfig.max_failure_rate`
and the `high_failure` flag of `invert.invert_node`; (iv) `MeasureConfig` (a new field for the per-solve budget of the
D-052 fit, inside `signature_hash("estimator")`), Block 12's fit and Block 14's table.

**Decision [user].** (i) Q4b: "The option A" (each input divided by its design range). (ii) Q4c: "0.3 in ln alpha-hat
(Recommended, PROVISIONAL)". (iii) The threshold: "The recommended" (keep 0.2 PROVISIONAL; set it after the pilot).
(iv) `max_nfev`: "3000 per solve (Recommended)".

**Statement.**

(D-054.1) **Input scaling.** With $x_i = (\ln(d_i/\text{um}),\ \varphi_i/\text{rad},\ \ln\hat\alpha_i)$ the inputs of
retained replicate $i$ (D-053 (iii)) and $[x^{\min}_k, x^{\max}_k]$ the design range of coordinate $k$ ($\ln d$ and
$\varphi$ from `PhantomConfig.d_range_um`, `phi_range_deg`; $\ln\hat\alpha$ from the retained replicates' own minimum and
maximum, there being no configured range for a computed quantity), the spline of D-024 (v) is fitted on
$\tilde x_{ik} = (x_{ik} - x^{\min}_k)/(x^{\max}_k - x^{\min}_k) \in [0, 1]$, $k = 1, 2, 3$; the three
$(x^{\min}_k, x^{\max}_k)$ are stored in the table file and applied to every query, and cross-validation chooses the
smoothing $\lambda$ alone, over `spline_smoothing_grid` as today. The scaling is a table-fit setting (like
`spline_smoothing`), not an estimator setting: the estimator hash does not carry it.

(D-054.2) **Third bandwidth.** The Gaussian kernel weights of $\hat\tau$, SE, $N_{\rm eff}$ and the failure rate
(SPEC §2.4) use the bandwidths $(0.15$ in $\ln d$, $5°$ in $\varphi$, $0.3$ in $\ln\hat\alpha)$ in the *unscaled*
coordinates: `tau_kernel_bandwidth = (0.15, 5.0, 0.3)`, the third PROVISIONAL (assistant's value; about $\pm 35$ % in
$\hat\alpha$).

(D-054.3) **Failure-rate threshold.** `max_failure_rate` stays 0.2, PROVISIONAL. The davinci pilot (handoff §8.4
step 3, about 200 replicates) reports the local failure rate $\hat p_{\rm fail}(d, \varphi, \hat\alpha)$ by region; the
threshold is then set on that curve and logged as its own decision before production. Until then `invert_node` flags
`high_failure` where $\hat p_{\rm fail} > 0.2$ at the node's root, as today.

(D-054.4) **Per-solve budget.** Each `least_squares` solve of the D-052 fit -- the unweighted start solves and every
weighted IRLS solve -- has `max_nfev` = 3000, a `MeasureConfig` field (name fixed in SPEC.md, e.g.
`fit_sgt_max_nfev`), so it enters the estimator hash. A solve that stops at the budget (scipy status 0) marks the IRLS
unconverged (`irls_ok = False`) and the fit leaves $\mathcal S$ by D-053 (ii).

**Why [user; run, sandbox; reasoning].** (i) **[user]** A. **[reasoning; run, sandbox 2026-10-16,
`check_tps_scaling.py`]** The thin-plate kernel is a function of the Euclidean distance between inputs, so the axes'
relative units are part of the estimator: an isotropic rescale is equivalent to changing $\lambda$ (checked to
$2 \times 10^{-13}$), an anisotropic one is a different surface (the $\varphi$ axis $\times 10$ at fixed $\lambda$ moved
$\hat b$ by up to 0.055 on a fixture with $\tau$ = 0.03). Range normalisation gives every full sweep of an axis the same
distance and leaves one knob to CV; on the fixture ($b = 1 + 0.06/d + 0.15\sin^2\varphi + 0.05\ln\alpha$, $N$ = 600,
5 seeds) its RMS error against the truth was 0.0073-0.0100 against 0.0092-0.0106 raw, and CV-chosen per-axis scales
(option B) reached 0.0062-0.0087 at 25 times the fits -- kept as a possible labelled comparison, not chosen. (ii)
**[user]** the recommended value; **[reasoning]** a neighbourhood a few times wider than the fit's own scatter in
$\hat\alpha$ and narrow against the axis' span of about 5.5. (iii) **[user]**; **[reasoning]** under D-053 (ii) the
thin-phantom failure rate is of order 0.2-0.4 (8 and 16 of 40 at 0.6 um, handoff §4.4), so a threshold set now would
either flag most thin nodes or be chosen blind; the pilot measures the curve first. (iv) **[user]**; **[run, sandbox,
D-052]** every D-052 number was produced at 3000, and the only two unconverged weighted fits of the Monte Carlo were
solves that hit it; `fit_variants`' rule (700 for 7 parameters) would stop more often.

**What it implies for code and tables.**
- `table.table_inputs(d, phi, alpha)` returns the raw coordinates; `BiasTable` holds `x_min`, `x_max` (3 each) and
  scales inside `b_hat` / `_weights`; `table_io` writes and reads them; `fit_table` takes the $\ln\hat\alpha$ range from
  the retained rows. A smoke check asserts that the stored scaling reproduces $\hat b$ after a file round trip and that
  an isotropic rescale with $\lambda/c^2$ reproduces the fit (the C1 identity).
- `config.validate` requires three bandwidths; the kernel weights use the raw-coordinate bandwidths of (D-054.2).
- `max_failure_rate` unchanged in code; a pilot-reading step added to the HPC runbook (HPC paths reference §2).
- `MeasureConfig` gains the budget field with `# source: D-054`; the smoke suite asserts its default and that a solve at
  the budget yields `irls_ok = False`.

**What it does not decide [open].** The threshold's value (after the pilot); whether option B is run as a comparison on
the first table; the handoff's Q3, Q5-Q10 (asked next); the retained-row range for $\ln\hat\alpha$ versus a configured
one (assistant's choice above, PROVISIONAL); whether a solve stopped at the budget is retried (not now: the fit leaves
$\mathcal S$).

**Sources.** **[user]** 2026-10-18, about 11:15 (the second AskUserQuestion card). **[run, sandbox, 2026-10-16]**
`check_tps_scaling.py`, `check_tps_scaling_seeds.py` (sent to the user). **[src]** SciPy 1.18.1
`RBFInterpolator` docstring ("epsilon ... has the same effect as scaling the smoothing parameter" for the thin-plate
kernel); `sci/diameter-pipeline` at `ea49181`: `analysis/table.py`, `analysis/invert.py`, `config.py`. **[KB]** D-024,
D-031, D-052, D-053; handoff §3.5, §4.4, §7. PubMed 3 queries (thin-plate spline scaling / RBF input scaling), 0 records;
bioRxiv bioinformatics, 7 days, nothing relevant.

**Status:** active; not implemented. Settles the scaling half of D-031's open point and the `max_nfev` row of D-052 (h);
(iii) is a deferral with a named trigger, not a value. [updated 2026-10-18, later] (iv) implemented: `MeasureConfig.sgt_max_nfev` = 3000 (Block 12, commit `bee1e0b`); (i) and (ii) are Block 14's. [updated 2026-10-20] (D-062.4) extends `spline_smoothing_grid` to $10^4$ with a flag at either end; (iii) stands unchanged (handoff §5.2 item 14 closed by the user). [updated 2026-10-20, later] (i) is retired by **D-063** [user]: the estimator is a local linear kernel regression in raw coordinates, so no range normalisation; (ii)'s bandwidths become the estimator's own smoothing, the third carried to $\ln(\hat\mu\hat d)$ at 0.3, PROVISIONAL.

---

## D-055 -- The in-silico sensitivity check of the weighted fit: S0 closure, S1 the tilt as an approximation, S2 components the law lacks, S3 the real cells, with every prediction fixed before the runs

**Date:** 2026-10-18, 11:24 (Europe/Rome). **Binds:** D-048's "sensitivity to an
uncaptured noise component measured in silico during implementation" and the
closure smoke of the 2026-10-01 handoff §5 item 6; the four check scripts of the
implementation (names provisional: `sens_s0_closure.py`, `sens_s1_tilt.py`,
`sens_s2_missing.py`, `sens_s3_real.py`) and their read-out; what the $\chi^2$
alarm threshold (proposal 0.01) and $k_{\max}$ (D-050) are judged against.

**Decision [user].** "Agree as written", to the table put by the assistant on
2026-10-18 (handoff §8 item 4). The predictions below are fixed now and are not
to be adjusted after a run; a failed prediction is reported as such.

**Statement.** Twins carry the law (D-017.1 with the tilt of D-034) unless the
scenario says otherwise, and are fitted by the full procedure: D-049 seed,
D-050 iteration ($R$ = 600, $M$ = 3000, $\theta$-only 1 % rule, $k_{\max}$ = 6),
final step at $R$ = 3000, $d$ = 15 statistics of D-051, estimator D-051 (iv),
tilt alternation D-051 (iii), optimiser D-056. "SD" is the across-twin robust
SD $(q_{90} - q_{10})/2.563$ of S0 for the same parameter; "alarm" is
$\Pr(\chi^2_{11} > J) < 0.01$ at the final step ($d - p$ = 15 $-$ 4).
Production sizes run on the cluster, one PBS job per twin; a 20-twin pilot at
$R$ = 100, $M$ = 500 runs in the sandbox first and tests the wiring only, not the
predictions.

*S0, closure.* 150 twins at $\theta$ = $\hat\theta_{\rm sim}$ = (0.0383 mV, 0.179,
0.0547 mV, 39.7 ms), $\sigma_b$ = 0.10 mV/s, with the real $n_m$, $K_m$ and SS sweep
structure of two L3_exc cells: 537297470 (6 LS sweeps, 66 pulses) and 569844159
(6 LS sweeps, 348 pulses); 300 twins in all.
- S0.1 bias: $|\operatorname{median}\hat\theta_i - \theta_i| \le 0.25$ SD for $\sigma_f$, $\rho_f$, $\sigma_s$; $\le 0.5$ SD for $\tau_s$.
- S0.2 $J$: mean $J$ in $[0.8, 1.2] \times 11$; alarm rate $\le 3$ %.
- S0.3 convergence: the 1 % rule met at $k \le 3$ in $\ge 90$ % of twins; `no_plateau` $\le 2$ %.
- S0.4 gain: robust SD of $\hat\tau_s$ under `sigma_inv` $\le 0.7 \times$ under `hand_omega`, same twins.
- S0.5 Monte Carlo share: for 30 twins $\times$ 5 seeds, the seed-to-seed robust SD of $\hat\tau_s$ $\le 0.3 \times$ the across-twin SD.
- S0.6 seed $W_0$: the final $\hat\theta$ from the `diag_se` and the `hand_omega` seed within 0.1 SD in $\ge 90$ % of twins (same $\boldsymbol\xi$).
- Measured without prediction: per-parameter precision at 66 against 348 pulses; iterations; wall time per twin.

*S1, the tilt as an approximation.* 100 twins each carrying, in place of the
tilt, a seconds-scale OU of $(\sigma_3, \tau_3)$ = (0.142 mV, 3 s) and (0.293 mV,
30 s), both matched to the cohort's 6.89 s change (topic doc §3.6.3); fitted
with the law + tilt, $\hat\sigma_b$ from (D-034.1) at $h^*$ = 0.8 s on the twin's
own SS window means.
- S1.1 $\hat\sigma_s$ and $\hat\tau_s$ biased high by $\ge 1$ SD at 3 s; $\hat\sigma_f$, $\hat\rho_f$ within 0.5 SD.
- S1.2 alarm $\ge 50$ % at 3 s, $\ge 30$ % at 30 s.
- S1.3 the largest standardised residuals $(q^{\rm sim}_m - \hat q_m)/\mathrm{SE}_m$ at the optimum are in the window SDs and $\hat\rho(10\,\mathrm{ms})$, not at the 0.02-0.1 ms lags.
- Measured: $\hat\sigma_b$ against the matched 0.08-0.12 mV/s; the twin's implied step-mean baseline error.

*S2, components the law lacks*, 100 twins each: (a) a sweep-locked LS
transient, exponential from the sweep start, $\tau$ = 50 ms, amplitude set to a
locked share of 0.15 (the five flagged cells reach 0.10-0.19, D-017 note 2);
(b) independent extra scatter on each statistic, +0.1, 1 and 10 % variance (the
`check_missing_noise.py` model); (c) SS response jitter: amplitude
$10\,\hat\sigma_{\rm SS}$ at $t_{\rm SS}$, $\tau_r$ = 30 ms, pulse-to-pulse jitter 2 % and
5 %, subtraction (D-051 (iv)) on; (d) a non-exponential fast part: 4-pole
Bessel-filtered white noise (lag-1 0.81, topic doc §3.5.4) mixed with white
noise to the cohort's $\sigma_f$ and $\rho_f$.
- S2.1 (a): alarm $\ge 50$ %; $\hat\tau_s$ off by $\ge 1$ SD.
- S2.2 (b): alarm $\le 5$ % at 0.1 %, $\ge 50$ % at 1 %, $\approx 100$ % at 10 %; $\hat\tau_s$ bias $\le 0.5$ SD at 1 %.
- S2.3 (c): alarm $\le 5$ % and bias $\le 0.2$ SD at 2 % jitter; alarm $\ge 20$ % at 5 %.
- S2.4 (d): alarm $\approx 100$ %, driven by the SS 0.02-0.1 ms statistics (SE $\approx$ 0.003-0.005 against a shape miss $\approx$ 0.04); $\hat\rho_f$ absorbs part; $\hat\sigma_s$, $\hat\tau_s$ within 1 SD.
- Measured: which statistics carry each misfit.

*S3, the real cells* (cluster, after S0-S2): the 18 L3_exc cells.
- S3.1 alarm on $\ge 12$ of 18 cells, the largest residuals at the SS 0.1 ms autocorrelation -- the fast-shape misfit of the cohort fit (topic doc §3.5.3). **If S3.1 holds, the next decision is about the fast component's shape, not about the weighting.**
- S3.2 $\hat\tau_s \in [15, 100]$ ms for $\ge 15$ cells; 526785799 and 571430815 among the three largest $\hat\sigma_b$.
- S3.3 $J_{\rm SS} > J_{\rm LS}$ in $\ge 12$ cells.
- Measured: $t_{\rm SS}(c)$; the pulse order within a sweep; the response jitter; $\hat\sigma_b$ against the LS tail's $|\Delta_{\rm end}|$ (D-017 note 4).

**Why.** **[user]**. The scenarios come from the handoff §4 and from what the
checks of 2026-10-13/16 and 10-18 found: S0.4 and S0.5 from the gain (7.57
against 16.98 ms) and the seed spread (34.5-40.2 ms at $R$ = 200, $M$ = 400) of
`check_weights.py` and `check_crn_determinism.py` **[run]**; S1 from the OU
sizes of topic doc §3.6.3 **[KB]**; S2 (b) from `check_missing_noise.py`
(mean $J$ 8.9 to 18.8 at +1 %) **[run]**; S2 (c) from `check_pulse_residual.py`
(biases 9-34 SD unsubtracted, $\le 0.02$ SD subtracted at 2 %) **[run]**;
S2 (d) and S3.1 from the cohort misfit at SS 0.1-1 ms (0.109 against 0.153)
**[KB, §3.5.3]** and the SS SEs of §3.8.2 **[KB]**. Cost **[reasoning]**:
about 1100 fits at $\approx$ 30 min each, $\approx$ 3 h wall on 200 cores.

**What it implies for code and tables.** Four scripts, each with its
predictions in the docstring and a `PASS/FAIL` line per prediction; a shared
read-out that writes per-scenario tables (bias/SD, alarm rate, iterations,
residual loadings); the twin generator and the fit share one simulator (D-017
item 3, D-051 (iii), D-056 (i)); S2 (a)-(d) are extra generator options
labelled as misspecification, never defaults; S3 reads the archive on the
cluster.

**What it does not decide [open].** The alarm threshold (0.01 provisional,
judged by S2.2); $\kappa_{\rm tr}$, the bin and $h^*$ (D-051 provisional); what is
done if S3.1 holds (a decision after S3); the two high-trend cells (D-017 (a)).

**Sources.** **[user]**, 2026-10-18 11:24. Handoff §4 (repo,
`dev/2026-10-16_covariance_weighting/`); topic doc §3.5.3, §3.5.4, §3.6.3,
§3.8.2 **[KB]**; D-017 note 2, D-048-D-051 **[log]**; the sandbox checks named
above **[run]**.

**Status:** active; not run. Designs D-048's sensitivity check and settles
handoff §8 item 4.

---

## D-056 -- Implementation conventions of the noise fit: continuous simulated SS sweeps, a spec in `Ih Fit/specs/`, a patch series from the zip baseline, and the inner optimiser with a Harrell-Davis binding function

**Date:** 2026-10-18, 11:24 (three cards) and 11:31 (the optimiser, after the
question was restated with four options). **Binds:** D-051 (iii)'s simulator;
D-050's open optimiser point; where the spec of the noise-fit blocks lives
(D-022 covers the diameter workstream only); how the code is delivered (the
assistant has no push access).

**Decision [user].** (i) "Continuous sweep, real pulse timing"; (ii) "Ih
Fit/specs/SPEC.md + patch series"; (iii) "your recommended option is ok" --
the Harrell-Davis binding function with SciPy's default Jacobian and three
starts.

**Statement.**
(i) **Simulator.** For each real SS sweep of cell $c$ the simulator generates
one continuous trace of the sweep's own length at the cell's $\Delta t$, with
the law's fast and slow recursions run across the whole trace, one tilt $b$
for the trace, and the enlarged windows cut at the real pulse positions
(pulses 200 ms apart [corrected 2026-10-20: 200.5 ms, 37 per sweep; the stored
windows are a trace-QC-selected subset whose positions follow from
`trace_qc_results.csv`; data, changelog 2026-10-20 (later, 5)], from the archive's
sweep structure). The joint law of a
sweep's windows -- the slow component's carry-over $e^{-t_{\rm SS}/\tau_s}$
between adjacent windows and the shared $b$ -- is thereby reproduced exactly,
not approximated. LS sweeps are one window each. Sharpens D-051 (iii).
(ii) **Spec and delivery.** The blocks are specified in a new
`Passive Features/HPC script/Ih Fit/specs/SPEC.md` (scientific-coding template;
Coverage = the noise-fit modules and the lines they change); the root
`specs/SPEC.md` may point to it later (user's call). Work happens on branch
`sci/noise-fit` of a local git repository whose baseline commit is the
content of `IhFit_main_2026-10-16.zip` (a `git archive` of `main`), so that
`git format-patch` output applies to the user's `main`; delivery = the patch
series plus a tarball of the branch's files, `git apply --check` verified on a
fresh extract, ASCII-only `.py`, LF-only shell.
(iii) **Inner optimiser.** $q^{\rm sim}_m(\theta') = \mathrm{HD}_{1/2}\bigl(T_m(\mathbf x_j(\theta'; \xi_j)),\ j = 1..M\bigr)$,
the Harrell-Davis estimate of the median (`scipy.stats.mstats.hdquantiles`,
a fixed Beta-weighted average of the order statistics) over the $M$ CRN
windows -- the model-side estimator of $\operatorname{med}(T_m \mid \theta, n_m)$;
the cell's $\hat q_m(c)$ stays the sample median over its $K_m$ windows.
`scipy.optimize.least_squares(method='trf', bounds=...)` on the whitened
residual $C^{-1}(\mathbf q^{\rm sim}(\theta') - \hat{\mathbf q}(c))$, $\hat\Sigma = CC^{\top}$,
with its default finite-difference Jacobian; fit coordinates
$(\log\sigma_f, \rho_f, \log\sigma_s, \log\tau_s)$, bounds $\rho_f \in (0, 0.95)$ and
the positive parameters in log; three starts -- the closed-form
$\hat\theta_{\rm cf}(c)$ and $\hat\theta_{\rm cf}$ with $\tau_s/2$ and $2\tau_s$ -- the
lowest $J$ kept (`NoiseFitConfig.n_starts = 3`).

**Why.** **[user]**. Recorded with it: (i) the assistant's 2026-10-18 correction
that the enlarged windows of one sweep are not independent (correlation
$e^{-t_{\rm SS}/\tau_s}$: 0.78 at 10 ms, 0.08 at 100 ms, for $\tau_s$ = 39.7 ms
**[arithmetic]**); the mirroring argument of D-051 (iv) needs the same joint law
on both sides. (ii) The integration tester reads only the repository, so a
spec is the condition for testing (scientific-coding skill); the repo attach
was denied on 2026-10-13, so delivery stays by patch. (iii) With the sample
median, SciPy's default step gives one window's slope per Jacobian entry
(`check_fd_jacobian_median.py` **[run, 2026-10-16]**); with the Harrell-Davis
estimate, verified by running it (`check_fd_hd_median.py`, $M$ = 3001, 12 seeds,
$\tau_s$ alone varied **[run, 2026-10-18]**): the slope's seed-to-seed SD at the
default step fell from 0.00153 to 0.00033 ($\hat\sigma$) and from 0.053 to 0.0051
($\hat\rho(1\,\mathrm{ms})$), equal to the 0.03-step reference (0.00029, 0.0048), mean
slopes within 1 % of it; the HD and the sample median differ at $\tau_0$ by
$\le 0.1$ of the sample median's Monte Carlo SD (P1, P2 held; P3 -- HD still
$\ge 1.5\times$ noisier than the 0.03 step -- failed in the option's favour).
Three starts because the 2026-09-28 cohort fit stalled from one of four
(D-017 note 1) **[log]**. Alternatives declined: an own central-difference
Jacobian of step 0.02 (8 extra evaluations per Jacobian), Nelder-Mead (no
curvature, 150-300 evaluations), the default on the sample median with one
start.

**What it implies for code and tables.** `noise_law.simulate_sweep(theta,
sigma_b, n_samples, dt, xi)` returns one continuous trace; a `SweepStructure`
per cell (pulse count and positions per sweep, polarity per pulse, $t_{\rm SS}$)
read from the archive drives both the data windows and the simulated ones;
`noise_fit` uses `hdquantiles` for every binding-function entry (asserted in
the smoke against the sample median within its Monte Carlo band); starts and
bounds are config fields; the output records `noise_fit_start` (which start
won) and `noise_fit_nfev`. Smoke: the simulated windows of one sweep show the
predicted adjacent-window correlation; HD within band; three starts agree on a
planted twin.

**What it does not decide [open].** Whether `ss_pulses.npz` carries sweep
numbers at all -- the loader reads none (`passive_fitting_hpc_fixed.py`
l. 510-552 **[src]**); if it does not, the sweep structure must come from the
archive writer or from the pulse order, to be checked on the cluster
(command given to the user 2026-10-18) [updated 2026-10-19: D-017 (c) records, from the archive writer's source read on 2026-09-28, that the archive stores the SS pulses in order with their sweep number **[src, per D-017]**; where it is stored (`ss_pulses.npz` or the metadata) the listing decides] [updated 2026-10-20: answered at source -- `save_alleninstitute_data (6).py` writes `sweep_number` (int64, per pulse) into `ss_pulses.npz` in both formats and stores a sweep's windows in onset order, polarity alternating **[src]**; the files to confirm by G12 of `cluster_gates_noise_fit.sh`]; the pulse order within a sweep;
whether the root `specs/SPEC.md` Coverage points to the new file; the
`least_squares` tolerances (defaults until S0's pilot says otherwise).

**Sources.** **[user]**, 2026-10-18 11:24 and 11:31. `check_fd_jacobian_median.py`,
`check_fd_hd_median.py` with logs, sent to the user **[run]**; SciPy 1.18.1
`_numdiff` and `least_squares` **[src]**; `passive_fitting_hpc_fixed.py`
l. 510-552 **[src]**; D-017 note 1, D-022, D-050, D-051 **[log]**; the
scientific-coding and hpc-git-delivery skills.

**Status:** active; not implemented. Settles D-050's optimiser point and
sharpens D-051 (iii).

---

## D-057 -- Every counted replicate enters the failure rate; the $\sigma_{\rm fit}$ study is dropped; the phantoms go through exactly the real-data pipeline

**Date:** 2026-10-18, about 11:40 (Europe/Rome), the user's answers to the third card of the table chat (handoff §7
Q3, Q5, and the Q8-Q10 card, answered with a principle rather than a choice). **Binds:** (i)
`CorrectionConfig.failure_rate_ignore` and `table.fit_table`'s `counted` mask (SPEC §2.4, Block 6); (ii)
`MeasureConfig.sigma_fit_study_um`, `scripts/build_table.py --sigma-fit`, `scripts/run_cell.py --sigma-fit`, the
per-node column `d_tilde_sigma_spread_um`, notebook Cell 8 (the three study configurations), SPEC §2.4 and Block 8's
column list, procedure §3.10's $\sigma_{\rm fit}$ row [corrected 2026-10-19: neither `scripts/build_table.py` nor `scripts/run_cell.py` has a `--sigma-fit` flag at `d643aaf` **[src]**; the study runs through notebook Cell 8's configuration files (`config_sigma0.080.json`, `config_sigma0.125.json`) and Cell 10c, and the code it leaves is `config.with_sigma_fit`, `MeasureConfig.sigma_fit_study_um` with its `validate` check and its exclusion from `estimator_signature`, the `analysis/cell.py` column and the suites' uses (the 2026-10-19 handoff's §4)]; (iii) procedure §3.7 and D-052.4 as applied to
`phantoms.run_replicate` -> `node_pipeline.measure_node` (Blocks 5-6), i.e. the plane rule (D-052.3), the profile
extent (`profile_half_um`), the pilot, the background, the fit and the flags.

**Decision [user].** (i) Q3: "() -- count every replicate". (ii) Q5: "(C) Drop the study". (iii) On the Q8-Q10 card:
"After the rendering the exact same pipeline used on real data must be used on the phantom rendered data. this
includes the choice of the in focus plane the profile extent etc....".

**Statement.**

(D-057.1) **Failure rate.** `failure_rate_ignore = ()`: the local failure rate of SPEC §2.4 (with D-054's third
bandwidth) is the kernel-weighted share of replicates outside $\mathcal S$ among *all* replicates of the table,
$\hat p_{\rm fail}(d, \varphi, \hat\alpha) = \sum_n w_n [n \notin \mathcal S] / \sum_n w_n$ over every row $n$; no
screen removes a replicate from the count. The field stays (D-021: alternatives selectable by name) with the empty
default; `("dark",)` is inert under D-030 and is not a default anywhere.

(D-057.2) **$\sigma_{\rm fit}$.** D-023 (iv) is withdrawn: no table and no real-node run at $\sigma_{\rm fit} \in
\{0.080, 0.125\}$ um; the production table and the per-node CSV are computed at `sigma_fit_um` = 0.099 um
(PROVISIONAL, D-023) only; `sigma_fit_study_um`, the `--sigma-fit` loops [corrected 2026-10-19: the loops are the notebook's, Cells 8 and 10c; no script has such a flag], Cell 8's study configurations and the
`d_tilde_sigma_spread_um` column are removed from the code and the SPEC (not kept as options). $\sigma_{\rm fit}$
keeps its remaining roles under D-052: the pilot's Gaussian fit (centres, heading, tilt, D-052 (e)), the start
points of the SGT fit, and the renderer's consistency check (procedure §3.10, first row), and stays in the estimator
hash. D-023 (i)-(iii) stand.

(D-057.3) **One pipeline [user].** For every phantom replicate the measurement is `measure_node` exactly as it runs
on a real node: the same pilot ($k^*$ by D-035's gradient energy, the centres, the TLS direction, $\varphi_j$), the
same Allen-guided iteration over $K_j = k_{{\rm SWC},j} \pm 6$ with $m$ = 3 and at most 5 rounds (D-052.3), the same
profile offsets $v_n$ ($\pm$`profile_half_um` = $\pm 3$ um at `profile_step_um`), the same $\bar B_j(k)$ rule, the
same SGT fit with its IRLS (D-052.2), the same flags and the same $\mathcal S$ (D-053.2). Nothing of the chain is
switched, shortened or given the truth on a phantom; the only inputs that differ are the ones the real data supplies
and the phantom draws (the branch with its radius, the SWC depth $k_{{\rm SWC},j}$, the block). This elevates D-052.4
from the assistant's reading of procedure §3.7 to the user's decision.

**Why [user; reasoning].** (i) **[user]**; **[reasoning]** with `dark` a label (D-030) the old default rejected
nothing, so the configuration said something the code did not do; and a failure rate conditional on no screen is
the one a real node, which applies no screen to itself, is exposed to. (ii) **[user]**; **[reasoning]** under D-052
$\sigma_{\rm fit}$ no longer sets the final fit's kernel ($w$ and $\kappa$ are fitted per node), so the sensitivity the
study measured (the corrected $\tilde d$ against the assumed in-focus blur, D-023 Why (iv)) now runs only through
the pilot's centres and the SGT fit's starts, a second-order path; the renderer's $\sigma_{\rm r}(0)$ remains the
unidentified input and keeps its own check (procedure §3.10, second row). (iii) **[user]**; **[reasoning]** the table
is the bias of *this* estimator on *this* simulated microscope (procedure §3.2-3.3, mathematics §3.7): any step of
the chain that a phantom skips or is handed is a bias the table cannot see and the real nodes keep.

**What it implies for code and tables.**
- (i) `config.py`: `failure_rate_ignore: Tuple[str, ...] = ()` with `# source: D-057`; the smoke check of
  `test_smoke_table.py` that asserts the dark screen's effect becomes a check that `()` counts every row and that
  naming a screen still works; SPEC §2.4's failure-rate sentence rewritten.
- (ii) `config.py` loses `sigma_fit_study_um`; `config.estimator_signature` no longer excludes it; `build_table.py`
  and `run_cell.py` lose `--sigma-fit` [corrected 2026-10-19: they have none; `config.with_sigma_fit` and the notebook's Cell 10c go instead, and `test_smoke_invert.py` and `test_smoke_table.py` need another estimator field to build a mismatched configuration]; Block 8's CSV loses `d_tilde_sigma_spread_um`; Cell 8 of the notebook and the
  runbook are updated; SPEC §2.4, Block 6, Block 8 and §8 (the spread bullet) updated; `test_smoke_config.py`'s
  assertion of the study tuple removed. The default `sigma_fit_um` = 0.099 um is unchanged.
- (iii) Block 13's oracle: phantoms and real nodes share one function object (`measure_node`), no phantom-only
  branch inside it; `run_replicate` passes only `branch`, `provider`, `cfg` and the SWC-depth plane; a smoke check
  asserts that `measure_node` has no argument or configuration flag that a phantom sets differently from a real
  node except those three inputs.

**What it does not decide [open].** Q7 (thin nodes: the default only, or also the pinned-kernel comparison);
$d_{\max}$ of the phantom range (Q8: still the TEMPORARY 4 um until Cell 6's radius CSV is read); `jitter_z_um`
(Q9: 0 until $\Delta z^*$ is measured); the renderer's kernel shape (Q10, I-002); whether the renderer's
$\sigma_{\rm r}(0)$ check (procedure §3.10, second row) is run on the first table. These are restated to the user
after the explanation of the table's construction given the same turn.

**Sources.** **[user]** 2026-10-18, about 11:40 (the third AskUserQuestion card). **[KB]** D-021, D-023, D-030,
D-052, D-053, D-054; procedure §3.2, §3.3, §3.7, §3.10; mathematics §3.6-3.7; SPEC §2.4, Blocks 6-8. **[src]**
`sci/diameter-pipeline` at `ea49181`: `config.py` (l. 148), `analysis/table.py` (`fit_table`, l. 164-168),
`scripts/build_table.py`, `scripts/run_cell.py`.

**Status:** active; not implemented. (ii) supersedes D-023 (iv); (iii) confirms D-052.4. [updated 2026-10-21, later: (ii) implemented, S2.2 of the 2026-10-20 plan, `a879067`, pushed at the user's agreement (`f4ac106..9c1329e` on `sci/diameter-pipeline`): `MeasureConfig.sigma_fit_study_um` with its `validate` check and its exclusion from `estimator_signature`, `config.with_sigma_fit`, the per-node column `d_tilde_sigma_spread_um`, notebook Cell 8's study configurations and Cell 10c removed; the runbook (v19) and SPEC section 4, Blocks 1 and 8, section 8 updated; the suites set `sigma_fit_um` directly, and any positive value is now valid; the estimator and simulation hashes are unchanged, the study set having been outside the estimator signature **[run, sandbox]**. (i) is the plan's S4.] [updated 2026-10-21, later 2: (i) implemented, Block 14.1 (`89ca78a`, local): `failure_rate_ignore = ()`, every row in the failure count, which Block 14.3 places at the true coordinates (D-062.5)]

---

## D-058 -- The phantoms' starting radius is the true radius plus zero-mean normal noise

**Date:** 2026-10-18, 15:44 (Europe/Rome), the user's reply in the table chat ("I think you are overcomplicated thus
use the method I said"), the method stated at 13:57 ("use as starting d/2 the real cylinder's d plus some noise ...
derived from a zero mean normal distribution"). **Binds:** `analysis/phantoms.phantom_branch` (the radius each
phantom's branch carries, today `0.5 * draw.d_um`) and through it the places of the chain the node's radius enters
(handoff §6: the iteration's start $d^{(0)}_j = 2 r_j$, which sizes round 1's line, half-length $h_{j,1} = 3 r_j$ by
D-044; the pilot's focus window $|v| \le r_j + 0.5$ um; the $\bar B_j(k)$ mask of D-018.1; `block_background`'s
mask); `PhantomConfig` and the simulation hash; D-053 (i).

**Decision [user].** Each phantom's starting radius is its true radius plus noise drawn from a zero-mean normal
distribution; this replaces the sensitivity check of D-053 (i) as the way the phantoms' radius is chosen.

**Statement.**

(D-058.1) For phantom replicate $n$ (a realization of a draw, D-059), the radius its branch carries is
$$r_n = \frac{d_n}{2} + \epsilon_n, \qquad \epsilon_n \sim \mathcal N(0, \sigma_r^2),$$
$d_n$ the true diameter (um), $\sigma_r$ a configured SD (um; value open), $\epsilon_n$ drawn independently from the
replicate's generator. Real nodes keep Allen's radius $r_{{\rm A},j}$ as their start.

**Why [user; reasoning].** **[user]** the simpler method, preferred to Allen-like draws. **[reasoning]** The
assistant's reservation, recorded at the user's choice: real nodes start from Allen's radius, whose error is
systematic rather than zero-mean (Allen's $2r$ is about 0.5 um in every bin of $\hat d$ on 529878215, and on the
synthetic 3.0 um tube an Allen-like start five times too small ended the iteration one or two planes from the axis,
SPEC §8), so a table whose phantoms start within zero-mean noise of their true radius does not contain the effect of
that systematic start wherever it does not wash out after round 1 (from round 2 the line is resized from the fitted
$\hat d$, handoff (H6)).

**What it implies for code and tables.**
- `PhantomConfig` gains `radius_noise_sd_um` ($\sigma_r$, `# source: D-058`, PROVISIONAL until set), inside the
  simulation hash; `phantom_branch` uses $r_n$ instead of $d_n/2$; replicate rows carry `radius_um`.
- $\epsilon_n$ comes from a stream of its own (the assistant's proposal, e.g. `default_rng([seed, n, 1])`), so the
  draws of D-024 (iii) keep their streams.
- A smoke check tests the draw's law (zero mean, SD $\sigma_r$, Kolmogorov-Smirnov) and that the branch carries it.
- D-053 (i)'s check is no longer the gate before Block 14.

**What it does not decide [open].** $\sigma_r$; whether the noisy radius feeds all four places of handoff §6 or only
the iteration's start (the measuring line) -- D-053 (i) used one radius in all four; the rule for a draw with
$r_n \le 0$ (possible on thin phantoms, $d_n/2$ = 0.1 um at $d_{\min}$): a redraw or a floor; under D-059, whether
$\epsilon$ is drawn once per cylinder draw or per realization; whether D-053 (i)'s check, with its in-focus images,
still runs as a diagnostic.

**Sources.** **[user]** 2026-10-18, 13:57 and 15:44. **[KB]** D-018, D-044, D-052, D-053, D-057;
`docs/TEEG_diameter_correction_table_handoff_2026-10-16.md` §3.4 (H6), §6; `specs/SPEC.md` §8. **[src]**
`analysis/phantoms.py` at `bee1e0b` (`phantom_branch`, `_draw`).

**Status:** active; not implemented; supersedes D-053 (i). [updated 2026-10-20] Settled by **D-062.1** [user]: $\sigma_r$ ($\sigma_{\rm br}$ in the 2026-10-19 handoff) = 0.2 um; the radius enters all four places; a draw with $r_n \le 0$ is redrawn (the kept $\epsilon_n$ truncated below at $-d_n/2$: mean starting diameter 0.404 um at $d_n$ = 0.2 um, 0.515 at 0.4 um [run]); $\epsilon$ per realization; D-053 (i)'s check with its images is dropped. [updated 2026-10-21, later 2: implemented, Block 14.2 (`e9682a5`, local): the branch radius $d/2 + \epsilon$ per realization from `default_rng([seed, g, r, 1])`, redrawn until positive, carried at every node of the phantom's branch; the kept $\epsilon$ tested against the truncated normal (Kolmogorov-Smirnov) **[run, sandbox]**]

---

## D-059 -- The phantom design is $G$ independent cylinder draws over the D-024 ranges, each realized $k$ times; the per-draw estimate marginalizes the nuisances; the phantoms' $\hat\alpha$ is checked against the real cells'

**Date:** 2026-10-18, 14:53 and 15:44 (Europe/Rome), table chat, the user's replies to the assistant's answers on the
nuisance expectation (Step 1) and on repeated nuisance draws. **Binds:** the phantom design of D-024 (iii)
(`phantoms.draw_replicate`, `phantoms._draw`, `PhantomConfig`, the replicate rows of `table_io`,
`scripts/build_table.py` and `build_table.pbs` chunking); what one element of the table is (`table.fit_table`,
Blocks 6 and 14); the failure-rate count of D-057 (i); Q6's budget.

**Decision [user].** (i) 14:53: "we will keep the its true $d_n$ (log-uniform on [0.2, 4] um), $\varphi_n$ (uniform
on [0°, 90°)), heading $\theta_n$ (uniform on [0, π)) and $\mu_n$ (log-uniform on [0.3, 3] /um) fixed as ranges and
just do a check whether the range of values of the syntehtic alpha actually contain the ones observed in the real
cells." (ii) 15:44: "The elements in the table must have the nuisances marginalized away. Therefore set a k number of
replicates per draw. Moreover I will set a number of independent random draws across the 4 axis of the cylinder
'design space' ... Each drawn will be realized k times and the estimated value calculated."

**Statement.**

(D-059.1) **Ranges and coverage.** The laws of D-024 (iii) stay. The phantoms' $\hat\alpha$ is checked against the
$\hat\alpha_i$ of the real cells. **[open]** the check's form; the assistant's proposal: fitted against fitted, per
bin of $\hat d$ (equivalently the real $\hat\mu_i$ against the phantoms' $\hat\mu_n$ per bin), since a node can lie
inside the phantoms' overall $\hat\alpha$ range and outside it at its own size (the dark node of the same day's
explanation: $\hat\alpha$ = 2.03, inside the check phantoms' 0.07-845, outside their band at 0.5 um **[run,
sandbox]**).

(D-059.2) **Design.** $G$ independent cylinder draws $c_g = (d_g, \varphi_g, \theta_g, \mu_g)$, $g = 1, \dots, G$,
from the laws of (D-059.1); each draw is rendered and measured $k$ times, realization $r = 1, \dots, k$ carrying its
own nuisances $\xi_{g,r}$ (sub-pixel offset, axis depth, camera noise, node jitter). The per-draw estimate of the ratio
is the mean over its realizations in $\mathcal S$,
$$\bar y_g = \frac{1}{k_{\mathcal S,g}} \sum_{r:\,(g,r)\in\mathcal S} \frac{\hat d_{g,r}}{d_g},$$
$k_{\mathcal S,g}$ the number of them in $\mathcal S$, and the draw's failure share is $1 - k_{\mathcal S,g}/k$.
Supersedes D-024 (iii)'s one realization per draw (its laws stay).

**Why [user; run].** **[user]** each element of the table must have the nuisances marginalized. **[run, sandbox]** The
assistant's comparison, recorded with it: at an equal number of measured replicates, 300 draws x 5 realizations gave
a table SD 1.06 times that of 1500 independent draws when a draw's features off the table's axes (heading, $\mu$ at a
given $\hat\alpha$) add nothing, and 1.34 times (the design effect $1 + (k-1)\rho$ predicts 1.34) when they add an
offset with $\rho$ = 0.2 (`check_grouped_design.py`: the fixture and $\tau(d)$ of `check_table_se_coverage.py`, fixed
smoothing 10, 60 tables per arm), so $G$ and $k$ are to be chosen with the design effect in view.

**What it implies for code and tables.**
- `PhantomConfig` gains `n_draws` ($G$) and `n_realizations` ($k$); the draw's features from `default_rng([seed, g])`,
  realization $r$'s nuisances from `default_rng([seed, g, r])` (proposal), so any chunking reproduces them.
- Replicate rows gain `draw_index` and `realization_index`; the merge and the table group by draw.
- The failure rate of D-057 (i) still counts every realization; the per-draw failure share is reported.
- `build_table.py` and `build_table.pbs` chunk by draw.
- Smoke checks: draws reproducible per $(seed, g)$, realizations per $(seed, g, r)$; a draw's realizations share
  $c_g$ and differ in $\xi$.

**What it does not decide [open].** $G$ and $k$ (Q6, after Block 13's per-replicate time is measured). **The third
coordinate of a draw in the table:** a draw's $k$ realizations have $k$ different fitted $\hat\alpha_{g,r}$, and
within a draw the errors of $\hat d$ and $\hat\alpha$ are strongly anticorrelated on thin tubes -- in the D-052 Monte
Carlo (40 noise draws per truth) corr($\hat d$, $\hat\alpha$) = -0.94 and -0.90 at 0.6 um, with $\hat\alpha$'s
relative SD 37 % and 43 %, and -0.72 to -0.88 at 1.5 um, relative SD 2-3 % **[data inspected]** -- while a real node
is inverted at its single $\hat\alpha_i$ (D-053.3). Options: (A) the spline through all $G k$ realizations at their
own $\ln\hat\alpha$, the per-draw means $\bar y_g$ serving the per-draw scatter and the checks; (B) one point
$\bar y_g$ per draw at the mean $\ln\hat\alpha$ of its realizations; (C) one point per draw at its true $\ln\alpha_g$.
The assistant recommends (A). Also open: whether cross-validation keeps a draw's realizations in one fold (needed
under (A)); whether D-058's $\epsilon$ is per draw or per realization; the coverage check's form, and what follows if
it fails (a wider $\mu$ range, or a support flag at the real node).

**Sources.** **[user]** 2026-10-18, 14:53 and 15:44. **[KB]** D-024, D-031, D-053, D-057, D-058. **[run, sandbox]**
`check_grouped_design.py` (2026-10-18, sent to the user); `check_table_se_coverage.py` (the dark node). **[data
inspected]** `checks/estimator_2026-10-15/ident_sgt/identifiability_sgt.json` (`mc[*].raw.sgt_l050`: the IRLS SGT
fit; $\varphi$ = 0.2 rad, $\mu$ = 1 /um, noise 2 gl; $\mathcal S$ by D-053 (ii)). **[full text, PubMed]** the design
effect $1 + (m - 1)\rho$: "Intraclass Correlation in Cluster Randomized Controlled Trials: A Meta-Analysis of Studies
on Major Depression", DOI 10.1177/02537176241293194.

**Status:** active; not implemented; supersedes D-024 (iii)'s one realization per draw (the laws stay). [updated 2026-10-18, later 5] A draw's third coordinate is settled by **D-060** [user]: option (A), the spline through every realization in $\mathcal S$ at its own $\ln\hat\alpha$. [updated 2026-10-20] $G \approx 10^5$ and $k$ = $n_{\rm real}$ = 10 by **D-062.2** [user]; the failure count's third coordinate is the true $\alpha_g$ for every realization by **D-062.5**; the form of the coverage check (D-059.1) is still open, explained to the user on 2026-10-20. [updated 2026-10-20, later] (D-059.1)'s coverage check is **D-065.1** [user]: a Colab survey of the real neurons' fitted optical depths across cells, per bin of $\hat d$, against the phantoms' band; the ranges stay. [updated 2026-10-21, later 2: the design implemented, Block 14.2 (`e9682a5`, local): $G$ draws x $n_{\rm real}$ realizations from the keys `[seed, g]`, `[seed, g, r]`; the per-draw table in Block 14.4; the coverage check of (D-059.1) not built]

---

## D-060 -- The table is fitted through every realization at its own $\ln\hat\alpha$; the per-draw means serve the scatter and the checks

**Date:** 2026-10-18, 16:05 (Europe/Rome), table chat: "Log A as the decision", option (A) of D-059's open point on
the third coordinate of a draw. **Binds:** `table.fit_table` and `BiasTable` (which rows the spline sees, and at
which coordinates); `table.choose_smoothing` (its folds); the per-draw summaries of (D-059.2); Block 14.

**Decision [user].** Option (A): the thin-plate spline is fitted through all $G k$ realizations in $\mathcal S$,
each at its own coordinates, and the per-draw means $\bar y_g$ of (D-059.2) serve the per-draw scatter and the
checks.

**Statement.**

(D-060.1) The spline's data are the realizations in $\mathcal S$,
$\{(x_{g,r},\, y_{g,r}) : (g,r)\in\mathcal S\}$ with $x_{g,r} = (\ln d_g,\ \varphi_g,\ \ln\hat\alpha_{g,r})$ and
$y_{g,r} = \hat d_{g,r}/d_g$. The table $\hat b(x\mid\mathcal C,\mathcal S)$ therefore still estimates
$b(x\mid\mathcal C,\mathcal S) = \mathbb E[\hat D/d \mid d, \varphi, \hat A = a, \mathcal C, \mathcal S]$ of
(D-053.3), conditional on a single realization's fitted $\hat\alpha$, and a real node is inverted at its own
$\ln\hat\alpha_i$ as before.

(D-060.2) Per draw $g$: the mean ratio $\bar y_g$ of (D-059.2), the within-draw SD of the ratio over its
realizations in $\mathcal S$ (the scatter from the nuisances alone, at fixed cylinder features), $k_{\mathcal S,g}$ and
the failure share $1 - k_{\mathcal S,g}/k$; reported per draw and used by the table's checks.

**Why [user; data inspected; reasoning].** **[user]** (A), the assistant's recommendation. **[data inspected]** Within
a draw the errors of $\hat d$ and $\hat\alpha$ are strongly anticorrelated on thin tubes (corr -0.94 and -0.90 at
0.6 um, $\hat\alpha$'s relative SD 37 % and 43 %; -0.72 to -0.88 and 2-3 % at 1.5 um; D-052 Monte Carlo, 40 noise
draws per truth), so the table must condition on the single fitted $\hat\alpha$ that a real node carries.
**[reasoning]** (B), draw means at the mean $\ln\hat\alpha$, and (C), the true $\alpha_g$, condition on quantities a
real node does not have; under them a real node's single $\hat\alpha_i$ would be read on an axis defined otherwise.

**What it implies for code and tables.**
- `fit_table` fits every row in $\mathcal S$ (as today), the rows now carrying `draw_index`; it also returns the
  per-draw table of (D-060.2).
- The local statistics of SPEC §2.4 ($\hat\tau$, SE, $N^{\rm loc}_{\rm eff}$, $\hat p_{\rm fail}$) stay over
  realizations; $\hat\tau$ is then the total scatter at $x$ (within and between draws), the one a real node's error
  needs.
- Cross-validation: a draw's realizations share $(d_g, \varphi_g)$ and lie close in $\ln\hat\alpha$, so a fold that
  splits them scores the spline on near-duplicates of its training rows; the assistant proposes folds by draw (every
  realization of a draw in one fold) **[reasoning; to be checked in Block 14's smoke suite]**.
- Smoke checks: the spline sees $\sum_g k_{\mathcal S,g}$ rows; on a fixture with a known within-draw $\hat d$-$\hat\alpha$
  correlation, the table recovers $\mathbb E[\hat D/d \mid \hat A = a]$, not the draw-mean relation.

**What it does not decide [open].** The CV folds (by draw, proposed); $G$ and $k$ (Q6); whether D-058's $\epsilon$ is
drawn per draw or per realization; the failure rate's third coordinate for realizations without a usable $\hat\alpha$
(the assistant proposes the true $\alpha_g$ for every realization); the form of D-059.1's coverage check.

**Sources.** **[user]** 2026-10-18, 16:05. **[KB]** D-031, D-053, D-054, D-057, D-058, D-059. **[data inspected]**
`checks/estimator_2026-10-15/ident_sgt/identifiability_sgt.json` (`mc[*].raw.sgt_l050`, $\mathcal S$ by D-053 (ii)).

**Status:** active; not implemented; settles D-059's open point on a draw's third coordinate. [updated 2026-10-20] Folds by draw by **D-062.3** [user]. At D-062's $N \approx 10^6$ realizations the dense thin-plate spline through every realization cannot be fitted (8 TB; about $2 \times 10^5$ h extrapolated from fits timed in the sandbox, `check_table_size.py`); which estimator replaces the dense fit -- local kernel regression on a grid recommended -- is the user's decision before Block 14 (D-062). The conditional target of (D-060.1) is unchanged by that choice. [updated 2026-10-20, later] The spline of (D-060.1) is superseded by **D-063.2** [user] (local linear kernel regression through every realization in $\mathcal S$, at the coordinates of D-063.1); the conditional target, the per-draw table of (D-060.2) and the folds by draw (D-062.3) stand. [updated 2026-10-21, later 2: (D-060.1)'s conditional target and (D-060.2)'s per-draw table implemented with D-063's estimator, Blocks 14.3-14.6 (`779bec0`, local): `table.per_draw_table`, `draws_<hash>.csv`; on check_b_target's toy the table follows the conditional target, not the draws' mean relation (within 0.03 where the two differ by 0.10-0.13) **[run, sandbox]**]

---

## D-061 -- Block 13's implementation choices; Block 10 selects its calibration nodes by Allen's diameters; the kernel is calibrated in Colab and the table is built and tested on davinci

**Date:** 2026-10-20, 12:21 (Europe/Rome), the user's answers to section 3.A of the assistant's status report of
2026-10-19 (the four implementation points of `docs/TEEG_diameter_correction_table_handoff_2026-10-19.md` §5.1) and
a workflow statement given with them. **Binds:** Block 13 -- `analysis/node_pipeline.measure_node` and the new
iteration module (the provider call and the squares, the split into pilot and iteration + final fit, the plane the
flags of $\mathcal S$ describe, the pilot's columns of `NodeResult`); the Phase II diagnostics that re-measure a
node (`scripts/plane_differences.node_stack`, `analysis/survey.node_planes`) and their suites
(`test_smoke_plane_iteration`, `iteration_fits`, `fit_variants`, `variant_iteration`, `plane_gradient`); Block 10
(`analysis/calibration.calibration_reasons`, `scan_node`, `CalibrationConfig.node_dhat_max_um`) and D-026's
far-field extension; the workflow of the kernel table (`RendererConfig.kernel_table_delta_um`,
`kernel_table_sigma_um`) from Colab to the davinci configuration (D-022), and the HPC runbook.

**Decision [user].** (a) "A option"; (b) "For the defocus kernel we will use the Allen's proposed diameters. Remember
that we use a defocus kernel that is fitted on thin branches to calibrate the kernel in the near field (up to 0.84 um)
while we will use larger dendrites (approximately 0.8 um) to calibrate the further part of the kernel."; (c) "the
option A is ok"; (d) "A"; (f) "the defocus kernel is constructed on colab. Once done it is loaded in the DaVinci where
the 100000 or more draws (with relative realizations) will be done and the table constructed and tested."

**Statement.** Symbols as in the 2026-10-19 handoff §3: $j$ a node (real, or the middle node of a phantom),
$k_{{\rm SWC},j}$ the plane nearest its traced depth, $K_j = \{k_{{\rm SWC},j} - 6, \dots, k_{{\rm SWC},j} + 6\}$
(D-052.3), $k^*_j$ the pilot's plane (D-035), $k^{\rm it}_j$ the iteration's last plane, $r_{{\rm br},j}$ the radius
the node's branch carries (Allen's $r_{{\rm A},j}$ on a real node, D-058's on a phantom), $m$ = 3 the line
multiplier (D-044), $h_{j,i} = m\,d^{(i-1)}_j/2$ the half-length of round $i$'s line.

(D-061.1) **One provider call, plus one per restack.** `measure_node` requests, in one call, the pilot's block
(`block_half_um` about the window nodes) over the union of the pilot's plane range ($\pm$`planes_half` widened for
tilt) and $K_j$; a round whose line of half-length $h_{j,i}$ reaches within 2 px of the current square's edge makes
one further call for a larger square (the `restack` of `gradient_iteration`), on real data cut from the cached block
while the square lies inside it and fetched beyond it. The same calls are made for a phantom and for a real node
(D-057 (iii)). Option (B), one call sized for the largest line the box allows ($1.5 \times 6$ um + 2 px), is not taken.

(D-061.2) **Pilot and iteration are separate functions, one chain.** `measure_node` = the pilot (unchanged: $k^*_j$ by
D-035's gradient energy, the centres, the TLS direction, $\theta_j$, $\varphi_j$) followed by the iteration of
(D-052.3) from $d^{(0)}_j = 2\,r_{{\rm br},j}$ and the final fit `fit_sgt.fit_plane(v, I, phi, B_bar, cfg.measure).final`
at $k^{\rm it}_j$ with $\bar B_j(k^{\rm it}_j)$ the D-018.1 masked median of that plane. The pilot is a function of its
own, which `survey.node_planes` and the diagnostics call, so the Cell 4j-4m suites keep comparing the pilot with the
pilot and no diagnostic runs the iteration twice; phantoms and real nodes go through the one `measure_node`.

(D-061.3) **The flags describe the fitted plane.** `faint`, `crossing` (on the profile at $k^{\rm it}_j$),
`stack_edge` (when $k^{\rm it}_j$ is an end plane of $K_j$ or of the usable planes), `bbar_few` (from
$\bar B_j(k^{\rm it}_j)$) and `profile_nan` are computed on the plane and profile actually fitted, so that
$\mathcal S$ describes the fit that enters the table and the inversion; the pilot's values stay as labelled columns
(`k_star`, `d_pilot_um`, `alpha_pilot`, the pilot's status and flags).

(D-061.4) **Block 10 selects by Allen's diameters.** The calibration's node choice reads $2\,r_{{\rm A},j}$, not the
fit's $\hat d_j$: the near-field calibration (plane offsets up to 0.84 um, D-026) on thin branches selected by Allen's
diameter, and the far field (beyond 0.84 um) on dendrites whose Allen diameter is about 0.8 um (just under, D-026),
read on their planes beyond 0.84 um. The thresholds in Allen's diameter (today `node_dhat_max_um` = 0.3 um on
$\hat d$, PROVISIONAL) and whether the remaining criteria -- flat ($\varphi_j \le$ `node_phi_max_deg`), faint
($\hat\alpha_j \le$ `node_alpha_max`), in $\mathcal S$ -- read the pilot's or the final fit's values are open; the
assistant proposes the pilot's, so that no criterion of the calibration moves with the estimator being calibrated.

(D-061.5) **Where each part runs.** Block 10 (D-026, D-027) runs in Colab on Allen's stacks; the kernel table it
produces ($\sigma_{\rm r}(\delta)$ at the plane-offset knots, `kernel_table_delta_um` / `kernel_table_sigma_um`,
inside the simulation hash) is loaded into the davinci configuration JSON; the $G$ draws with their realizations
(D-062), the table build, the merge and the table's tests run on davinci (D-022).

**Why [user; reasoning].** (a) **[user]**; **[reasoning]** the call renders what the real data would fetch, and the
restack's extra calls are the cost the real nodes pay too. (b) **[user]**; **[reasoning]** the calibration's nodes
must not be chosen by a quantity that depends on the kernel being calibrated: $\hat d_j$ is a function of the fit's
kernel through the model (3''), so selecting on it closes a loop (kernel -> fit -> selection -> kernel), while
Allen's radius is the one size label independent of the kernel. (c) **[user]**; **[reasoning]** one code path without
a phantom/real switch (D-057 (iii)), and no second iteration inside a diagnostic. (d) **[user]**; **[reasoning]** the
table and the inversion are built on the fit at $k^{\rm it}_j$, so the selection event has to be evaluated there.
(f) **[user]**; **[reasoning]** Colab holds the image fetcher and the real stacks, davinci the CPUs; the kernel table
is a small configuration object, which is why it can cross.

**What it implies for code and tables.**
- Block 13 as the 2026-10-19 handoff §6.1 plans it, with (D-061.1)-(D-061.3) fixed: a new module under
  `src/allen_diameter/analysis/` holding the iteration, imported by `scripts/plane_differences.py`; the pilot as a
  public function; the new `NodeResult` columns; the iteration's fields in `MeasureConfig` (`# source:` D-042, D-044,
  D-052.3, D-061) and hence in the estimator hash; the Block 13 suite asserts the one-call-plus-restacks rule on a
  phantom whose line outgrows the square, that the pilot function and `measure_node` agree on the pilot's columns,
  and that the flags are those of the plane $k^{\rm it}_j$.
- Block 10: `calibration_reasons` reads Allen's diameter (new config fields, names fixed in SPEC, e.g.
  `node_allen_diameter_max_um` for the thin set and a far-field band for D-026's set); `scripts/calibrate_kernel.py`
  and the notebook's calibration cell follow; Block 10's SPEC entry and data flow updated.
- The HPC runbook (project doc `claude/TEEG_diameter_HPC_paths_reference.md` §2) gains the hand-over step: the
  Colab kernel table written into the davinci `--config-json`, checked by the simulation hash printed by every task.

**What it does not decide [open].** (e) The iteration's background mask inside `measure_node`: `block_background`
masks every traced dendrite segment of the block and the soma (`survey.node_planes`'s `segments`, `soma`), which
`measure_node` does not receive; the assistant proposes a scene input (the segment table and the soma discs,
empty for a phantom) and explained it in the same turn, the user's answer pending. The Allen-diameter thresholds of
(D-061.4) and which values its other criteria read. The PROVISIONAL iteration readings of D-042 and the 4.0 um
background margin (D-052 (h)). Whether the sandbox gate 2 (reduced) still runs before davinci: the assistant keeps it
as the software gate, the production table's tests being davinci's (D-061.5).

**Sources.** **[user]** 2026-10-20, 12:21. **[KB]** D-022, D-026, D-027, D-035, D-042 to D-044, D-052, D-057,
D-058; `docs/TEEG_diameter_correction_table_handoff_2026-10-19.md` §5.1, §6.1; `specs/SPEC.md` Blocks 5, 10, 12.
**[src]** `sci/diameter-pipeline` at `84ea91f`: `analysis/node_pipeline.py` (`measure_node` l. 206),
`scripts/plane_differences.py` (`node_stack` l. 148, `block_background` l. 270, `gradient_iteration` l. 547),
`analysis/survey.py` (`node_planes` l. 158), `analysis/calibration.py` (`calibration_reasons` l. 204),
`config.py` (`CalibrationConfig` l. 234). **[run, sandbox, 2026-10-19]** the 24 suites at `84ea91f` (151 pass, 0
fail, 21 skip), hashes `1bb3585b4a4f1373` / `e44ac0bd220f5214`.

**Status:** active; not implemented. Settles the handoff's §5.1 (a), (c), (d), and (b) in the user's own form
(selection by Allen's diameters rather than by the pilot's or the final fit's $\hat d$); adds the Colab -> davinci
workflow to D-022. Block 13 is next. [updated 2026-10-20, later] (e) is settled by **D-064** [user]: a scene input for the iteration's background mask. [updated 2026-10-20, later 2] (D-061.4)'s thresholds are set by **D-067.4** [user], PROVISIONAL: thin $2 r_{\rm A} \le$ 0.6 um, far field 0.6-0.8 um, the other criteria read from the pilot. [updated 2026-10-20, later 6: (a), (c), (d), (e) implemented, Block 13 (commits `b5191c8`-`70136ca`, local, not pushed); the pilot reads only Block 5's planes of the one block, so it equals Block 5 at `84ea91f` bit for bit; (b) and (f) are the plan's S2 and S3] [updated 2026-10-21: pushed at the user's agreement, `84ea91f..888452a` on `sci/diameter-pipeline`] [updated 2026-10-21, later: (b) implemented as D-067.4 sets it, S2.1 of the plan, `1aed5d6`, pushed at the user's agreement (`f4ac106..9c1329e` on `sci/diameter-pipeline`): `calibration.calibration_reasons(result, cfg, allen_diameter_um)` reads Allen's diameter $2 r_{{\rm br},j}$ (the branch's radius at the node) and the pilot's status, flags, $\varphi_j$ and $\hat\alpha_j$; the near-field set is scanned as before, the far-field set is selected and counted but not scanned -- its scan beyond 0.84 um and the fit of D-026.1 are not built (SPEC section 8)]

---

## D-062 -- Block 14's values and rules: $\sigma_{\rm br}$ = 0.2 um in all four places, redraw, per realization; $G \approx 10^5$ draws of $n_{\rm real}$ = 10; folds by draw; the grid extended; the failure count at the true $\alpha_g$; three points dropped

**Date:** 2026-10-20, 12:21 (Europe/Rome), the user's answers to section 3.B of the assistant's status report of
2026-10-19 (the 16 open points of `docs/TEEG_diameter_correction_table_handoff_2026-10-19.md` §5.2). **Binds:**
`PhantomConfig` (`radius_noise_sd_um`, `n_draws`, `n_realizations`), `phantoms.phantom_branch`, `_draw`,
`draw_replicate` and the generator keys; the four places the node's radius enters (the iteration's start, the
pilot's focus window, the $\bar B_j(k)$ mask, `block_background`'s mask; handoff-16 §6); `table.choose_smoothing`
(its folds) and `CorrectionConfig.spline_smoothing_grid`; the rows and coordinate of the local failure rate
$\hat p_{\rm fail}$ (SPEC §2.4, D-057 (i)); `scripts/build_table.py`, `scripts/pbs/build_table.pbs` and the davinci
sizing; Block 14 -- and, as a consequence recorded below, the response estimator of D-024 (v) / D-060.

**Decision [user].** (1) "let's define the sd of 0.2"; (2) "What you recommend" (the radius in all four places);
(3) "Redraw"; (4) "Per realization"; (5) "G will be much more approx 100'000 each of which with n_real = 10
realizations"; (6) "Folds By draw"; (7) "Enlarge the grid"; (8) "Perfect as you say" (the failure count's third
coordinate is the true $\alpha_g$); (13, 14, 15) "Drop them".

**Statement.** Notation of the 2026-10-19 handoff §3: $g = 1, \dots, G$ the cylinder draws, $r = 1, \dots, n_{\rm real}$
the realizations of a draw (D-059's $k$), $d_g$ (um), $\varphi_g$, $\theta_g$, $\mu_g$ (1/um) the draw's features,
$\alpha_g = \mu_g d_g / \cos\varphi_g$ its true optical depth (analytic level), $\hat\alpha_{g,r} = \hat\mu_{g,r}\hat d_{g,r}/\cos\varphi_{g,r}$
the fitted one (computed level), $r_{{\rm br},g,r}$ the radius the phantom's branch carries, $\sigma_{\rm br}$ its noise SD
(D-058's $\sigma_r$), $\lambda_s$ the spline's smoothing, $\mathcal S$ the selection event.

(D-062.1) **Radius.** $r_{{\rm br},g,r} = d_g/2 + \epsilon_{g,r}$ with $\epsilon_{g,r} \sim \mathcal N(0, \sigma_{\rm br}^2)$,
$\sigma_{\rm br}$ = 0.2 um (`radius_noise_sd_um`, `# source: D-062`); a draw with $r_{{\rm br},g,r} \le 0$ is redrawn,
so the kept $\epsilon_{g,r}$ follows the normal truncated below at $-d_g/2$; $\epsilon_{g,r}$ is drawn **per
realization**, from a stream of its own whose key cannot equal a realization's (the assistant's, e.g.
`default_rng([seed, g, r, 1])`), so a draw's realizations differ in their starting radius as in their other
nuisances (D-059); the one $r_{{\rm br},g,r}$ is used in **all four places** the radius enters, exactly as a real
node's single $r_{{\rm A},j}$ is; rows carry `radius_um`.

(D-062.2) **Design size.** $G \approx 10^5$ independent draws ("100,000 or more"), $n_{\rm real}$ = 10 realizations
each: $N = G\,n_{\rm real} \approx 10^6$ measured realizations; `n_draws`, `n_realizations` in `PhantomConfig`.

(D-062.3) **Cross-validation folds by draw:** every realization of draw $g$ lies in the same fold
(`spline_cv_folds` folds over the draws), so no fold is scored on near-copies of its training rows.

(D-062.4) **The smoothing grid is extended** beyond 100: `spline_smoothing_grid` gains $10^3$ and $10^4$ (the extent
the assistant's), and the table build flags a cross-validated choice at either end of the grid.

(D-062.5) **The failure count's third coordinate.** Every realization $(g, r)$ enters the local failure rate
$\hat p_{\rm fail}(x \mid \mathcal C) = \sum_n \varpi_n(x)\,[n \notin \mathcal S] / \sum_n \varpi_n(x)$ (D-057.1) at
$x^{\rm fail}_{g,r} = (\ln d_g,\ \varphi_g,\ \ln\alpha_g)$ -- the true $\alpha_g$, which every realization has whether
or not its fit produced a usable $\hat\alpha_{g,r}$ -- and a real node's rate is read at its own fitted
$\hat\alpha_i$: one level plugged in for the other, as the tilt axis already does with the line-fit $\varphi_i$
(procedure Eq. 2). The spline's rows keep D-060's $\ln\hat\alpha_{g,r}$: the two uses of the third axis differ and are
documented as such.

(D-062.6) **Dropped or closed.** Item 13, a per-point smoothing $\propto \hat\tau^2(x_n)$: not pursued. Item 15,
D-053 (i)'s sensitivity check with its in-focus images: not run (D-058 stands alone). Item 14 closed as an open
point: D-054 (iii) stands unchanged -- `max_failure_rate` 0.2 PROVISIONAL, set on the davinci pilot's failure-rate
curve by a later decision [the assistant's reading of "drop" for item 14; if the `high_failure` flag itself is meant
to go, that is a new decision].

**Why [user; run, sandbox; reasoning].** (1), (5), (7), (13-15) **[user]**. (2) **[user]**, the assistant's
recommendation: D-057 (iii) lets a phantom differ from a real node only by its inputs, and a real node's one radius
reaches all four places. (3) **[user]**; the redraw keeps every branch radius positive at the price of a positive
mean shift at the thin end: with $\sigma_{\rm br}$ = 0.2 um the rejection probability $\Phi(-d_g/(2\sigma_{\rm br}))$
is 0.309, 0.159, 0.067, 0.006 at $d_g$ = 0.2, 0.4, 0.6, 1.0 um and the mean starting diameter $2\,\mathbb E[r_{\rm br}]$
is 0.404, 0.515, 0.656, 1.007 um for those true diameters (closed form of the truncated normal, Monte Carlo within
4 SE) **[run, sandbox 2026-10-20: `check_redraw_shift.py`]** -- the thinnest phantoms start, on average, from a
line sized for a tube about twice their size, which is what Allen's systematic radius does to real thin nodes (D-058's
Why) **[reasoning]**. (4) **[user]**; the radius error is then one more nuisance that each element of the table
marginalizes (D-059). (6) **[user]**; **[run, sandbox: `check_tps_deep.py` T7, 2026-10-18]** on 300 draws x 5 with a
draw-level offset, plain folds chose $\lambda_s$ = 0.01-0.1 and folds by draw 10-100, the table's RMS error 0.0752
against 0.0295. (7) under folds by draw the choice reached the grid's end (100) in 2 of 6 seeds (T7). (8)
**[reasoning]** the true $\alpha_g$ is the one coordinate every realization has.

**What it implies for code and tables.**
- `PhantomConfig`: `radius_noise_sd_um` = 0.2, `n_draws`, `n_realizations` = 10 (all in the simulation hash;
  `n_replicates` retired); the keys `[seed, g]` (features), `[seed, g, r]` (nuisances), `[seed, g, r, 1]` (radius);
  the redraw loop with its count recorded; rows carry `draw_index`, `realization_index`, `radius_um`; the merge and
  the per-draw table group by draw; `build_table.py` / `build_table.pbs` chunk by draw.
- **The response estimator must be revisited before Block 14 [open; the user's].** $N \approx 10^6$ rows cannot be
  fitted by the dense thin-plate spline of D-024 (v) as D-060 states it: `RBFInterpolator` forms a dense
  $(N + 4) \times (N + 4)$ system, $8 N^2$ bytes = 8 TB at $N = 10^6$ (80 GB at the $G = 10^5$ per-draw means), and the
  solve time measured here between $N$ = 3000 and 6000 (1.7 s, 18.3 s, exponent 3.4; the table chat's T6: 0.27,
  2.85, 24.8 s at 1500, 3000, 6000, exponent 3.1) extrapolates to about $2 \times 10^5$ h at $N = 10^6$ and 80 h per
  fit at $10^5$, before any cross-validation **[run, sandbox 2026-10-20: `check_table_size.py`]**; SciPy's own
  docstring and the thin-plate document §3.12 (7) place the dense solve at a few thousand points **[KB]**. At this $N$
  the local kernel window of D-054 (ii) already holds about 5,400-5,600 units of kernel mass and an effective count
  $N^{\rm loc}_{\rm eff} \approx 1.5 \times 10^4$ at three query points of the design (2 x 10^5 draws scaled to 10^6)
  **[run: the same script]**, so the spline's pooling across the domain is no longer needed for precision. Options
  put to the user in the same turn: (A) local kernel regression (local linear) with D-054's bandwidths, evaluated on a
  grid and interpolated -- the SE of SPEC §2.4 becomes the estimator's own (closing item 10) and one knob set serves
  $\hat b$, $\hat\tau$ and $\hat p_{\rm fail}$; (B) a penalized tensor-product B-spline regression (linear in $N$);
  (C) kernel-binned grid means, then the thin-plate spline through the grid nodes; (D) a thinned subset for the
  spline. The assistant recommends (A). Until decided, D-024 (v) and D-060 stand as written and Block 14's table
  logic waits.
- Cost envelope **[reasoning, from KB numbers]**: rendering 10-140 s per replicate under today's chain (HPC paths
  reference §2) plus 1-5 SGT fits of 1.5-9 s per round (SPEC Block 12), say 60-200 s per realization, gives
  $1.7 \times 10^4$ to $5.6 \times 10^4$ CPU-h for $10^6$ realizations -- 3.5-12 days of wall time on 200 CPUs; the PBS
  array (`-J 0-99`, 12 h walltime) must be resized by draws per task from the time Block 13 measures; the rows
  are of order 1 GB.
- The design effect at $n_{\rm real}$ = 10 is $1 + 9\rho$, $\rho$ the within-draw correlation of the rows' errors
  (2.8 at $\rho$ = 0.2): the variance of a mean of $G\,n_{\rm real}$ rows is $\tau^2[1 + (n_{\rm real} - 1)\rho]/(G\,n_{\rm real})$,
  Monte Carlo over formula 1.002 at $\rho$ = 0.2 **[run, sandbox 2026-10-20: `check_design_effect.py`]**; a reported
  SE must carry the factor or be computed over draws.
- Smoke checks as the handoff §6.2 lists them, with the radius law tested as the truncated normal (KS on the kept
  draws against `scipy.stats.truncnorm`), the redraw count recorded, and the folds asserted to keep a draw whole.

**What it does not decide [open].** The response estimator at $N \approx 10^6$ (above); the SE the table reports
(item 10; resolved under option (A), open otherwise); the form of the $\hat\alpha$ coverage check (item 9), the
`off_support` flag (item 11) and the $\ln\hat\alpha$ normalisation range (item 12) -- explained in the same turn, the
user's answers pending; the PBS sizing and davinci's limits for an array of order $10^3$-$10^4$ tasks; Q7-Q10 of the
2026-10-16 handoff; the value "100,000 or more" is a floor, the exact $G$ to be set with the array.

**Sources.** **[user]** 2026-10-20, 12:21. **[KB]** D-024, D-053, D-054, D-057, D-058, D-059, D-060;
`docs/TEEG_diameter_correction_table_handoff_2026-10-19.md` §5.2-5.3, §6.2; `specs/SPEC.md` §2.4, §8;
`docs/TEEG_thin_plate_spline_in_3D/TEEG_thin_plate_spline_in_3D.md` §3.12 (7). **[run, sandbox, 2026-10-20]**
`check_redraw_shift.py`, `check_design_effect.py`, `check_table_size.py` (Python 3.13.16, NumPy 2.5.3, SciPy
1.18.1; in the assistant's scratchpad, to be committed under `checks/table_2026-10-20/` with Block 14). **[run,
sandbox, 2026-10-18]** `check_tps_deep.py` T6, T7 (`checks/table_2026-10-18/`). **[src]** `sci/diameter-pipeline`
at `84ea91f`: `analysis/phantoms.py`, `analysis/table.py`, `config.py`, `scripts/pbs/build_table.pbs`.

**Status:** active; not implemented. Settles D-058's open points (the value, the scope, the $r \le 0$ rule, per
realization), D-059's $G$ and $k$ (Q6) and the failure count's coordinate, D-060's folds; closes the handoff's §5.2
items 1-8 and 13-15. Opens: the response estimator at $N \approx 10^6$, which supersedes nothing until decided. [updated 2026-10-20, later] The response estimator is settled by **D-063** [user]; the push-back on the production's order is recorded in D-065. [updated 2026-10-20, later 6: a consequence of (D-062.1) found with Block 13 -- round 1's line has fewer than 3 samples, and the phantom ends `line_short` (outside $\mathcal S$), when $r_{\rm br} < 0.5\,\Delta v/m$ = 0.019 um: under the redraw, 5.0 % of the phantoms at $d$ = 0.2 um, 2.9 % at 0.4 um, 0.2 % at 1.0 um, while a real node starts from Allen's radius **[run, sandbox: `checks/block13_2026-10-20/check_line_short.py`]**; raised with the user for Block 14. The per-realization time at the defaults, 48.4 s mean on 10 phantoms (d 0.3-3.5 um, phi 0 and 45 deg; a lower bound for this design), puts $10^6$ realizations at $\ge 1.3 \times 10^4$ CPU-hours **[run, sandbox: `time_13_5.py`]**] [updated 2026-10-21: the thin end settled by **D-069** [user]: round 1's line floored at 3 samples, the start radius unchanged] [updated 2026-10-21, later 2: implemented -- (D-062.1)-(D-062.2) Block 14.2 (`e9682a5`); (D-062.3) both estimators' folds by draw, (D-062.4) the smoothing grid to $10^4$ with the edge flag, (D-062.5) the failure count at the true coordinates, Blocks 14.3-14.6 (`779bec0`); local, not pushed **[run, sandbox]**]

---

## D-063 -- The table at $N \approx 10^6$: local linear kernel regression on a grid is the deliverable, a penalized tensor-product B-spline the labelled comparison; the SE is the estimator's own with the draw-level correction; `off_support` on the kernel mass; the third axis becomes $\ln(\hat\mu\hat d)$

**Date:** 2026-10-20, 16:11 (Europe/Rome), the user's answers to section 1 and to points 10, 11 and 12 of the
assistant's reply of 2026-10-20 (12:21 round). **Binds:** the response estimator of D-024 (v) and D-060's "spline
through every realization" (`CorrectionConfig.response_estimator`, `analysis/table.fit_table`, `BiasTable`,
`choose_smoothing`, `loading/table_io`); the SE, $\hat\tau$, $N^{\rm loc}_{\rm eff}$ and $\hat p_{\rm fail}$ of SPEC
§2.4; the flags of Block 7 (`invert.invert_node`, a new `off_support`); the table's coordinates of D-053 (iii) and
D-054 (i)-(ii) (`table.table_inputs`, `tau_kernel_bandwidth`, the stored ranges); the per-draw table of D-060.2;
Block 14; the theory chat's study of the table.

**Decision [user].** Section 1 (the estimator): "Let's implement both option A and option B"; point 10: "We will
use A as replied above"; point 11: "As you say, be aware that we will not use anymore the spline approach as before
because of the huge dataset we will generate"; point 12: "As you say" (option (c), the coordinate $\ln(\hat\mu\hat d)$).

**Statement.** Notation of D-062: $g$ the draw, $r$ the realization, $x_{g,r} \in \mathbb R^3$ a row's coordinates,
$y_{g,r} = \hat d_{g,r}/d_g$ its ratio, $\mathcal S$ the selection event, $N_{\mathcal S} = \sum_g n_{\mathcal S,g}$ the
rows in $\mathcal S$; $\hat d$, $\hat\mu$, $\hat\varphi$ a fit's diameter (um), absorption coefficient (1/um) and
line-fit tilt (rad), computed level; $\varphi_g$ a phantom's true tilt (the second axis, `table_phi_axis = "true"`).

(D-063.1) **Coordinates.** The table's input is $x = (x_1, x_2, x_3) = (\ln(d/{\rm um}),\ \varphi/{\rm rad},\ \ln(\hat\mu\hat d))$,
the third coordinate the fitted, tilt-free optical depth $\hat\mu\hat d$ (dimensionless; $\hat\mu$ in 1/um, $\hat d$ in
um) in place of D-053 (iii)'s $\ln\hat\alpha = \ln(\hat\mu\hat d) - \ln\cos\hat\varphi$. A phantom row sits at
$x_{g,r} = (\ln d_g, \varphi_g, \ln(\hat\mu_{g,r}\hat d_{g,r}))$; a real node $i$ is inverted along the line
$(\cdot,\ \varphi_i,\ \ln(\hat\mu_i\hat d_i))$. The table then estimates
$b(x \mid \mathcal C, \mathcal S) = \mathbb E[\hat D/d \mid d, \varphi, \hat M \hat D = \hat\mu\hat d, \mathcal C, \mathcal S]$,
a conditional expectation on a quantity every node carries, as (D-060.1) was; the two conditionings coincide where the
measured tilt equals the true one and differ by the tilt estimate's error otherwise. The dark label stays on
$\hat\alpha$ (D-030); the failure count's third coordinate of (D-062.5) becomes the true $\ln(\mu_g d_g)$.
Amends D-053 (iii); D-054 (i)'s range normalisation is no longer needed (the estimator below works in raw
coordinates with per-axis bandwidths); D-054 (ii)'s third bandwidth, 0.3, carries over to $\ln(\hat\mu\hat d)$,
PROVISIONAL.

(D-063.2) **The deliverable estimator: local linear kernel regression.** With the Gaussian weights of SPEC §2.4,
$\varpi_n(x) = \exp\!\big(-\tfrac12 \sum_{k=1}^{3} (x_{n,k} - x_k)^2 / h_k^2\big)$, $h = (h_1, h_2, h_3)$ the bandwidths
in the raw coordinates (D-054 (ii): 0.15, 5 deg, 0.3), at every query point $x$:
$$(\hat a(x), \hat\beta(x)) = \arg\min_{a \in \mathbb R,\ \beta \in \mathbb R^3} \sum_{n \in \mathcal S} \varpi_n(x)\,\big[y_n - a - \beta^\top (x_n - x)\big]^2, \qquad \hat b(x \mid \mathcal C, \mathcal S) = \hat a(x),$$
a weighted least squares with four unknowns, so $\hat b(x) = \sum_n \ell_n(x)\, y_n$ is a linear smoother with
explicit weights $\ell_n(x)$. $\hat m(d, \varphi, \hat\mu\hat d \mid \mathcal C, \mathcal S) = d\,\hat b(x)$ as before.
The table is evaluated on a regular grid of the three coordinates (sizes PROVISIONAL, e.g. 48 x 19 x 36) and
stored as grid values; queries, the inversion included, read a tensor cubic interpolant of the grid
(`scipy.interpolate.RegularGridInterpolator`, `method="cubic"`; $C^2$), the root of $\hat m(\cdot, \varphi_i, \hat\mu_i\hat d_i) = \hat d_i$
found on it as Block 7 does today. The implementation is `response_estimator = "local_linear"` (the default; the
dense `"tps_spline"` stays selectable for small tables only and refuses $N_{\mathcal S}$ above a configured cap).
Open, the assistant's proposal: one scalar multiplier $c$ on the D-054 triple, chosen by cross-validation with folds
by draw (D-062.3) over a short grid, default $c$ = 1 -- one knob, as D-054 (i) had for the spline.

(D-063.3) **The SE, $\hat\tau$, $N^{\rm loc}_{\rm eff}$, $\hat p_{\rm fail}$.** $\hat\tau(x)$, $N^{\rm loc}_{\rm eff}(x)$ and
$\hat p_{\rm fail}(x \mid \mathcal C)$ stay the kernel statistics of SPEC §2.4 with the same $\varpi_n(x)$. The SE of
$\hat b(x)$ is the smoother's own variance with the draw-level (cluster) correction of D-062's design effect: the
assistant's form, PROVISIONAL, is the cluster-robust sandwich by draw,
$\widehat{\rm Var}(\hat b(x)) = \sum_g \big(\sum_{r:\,(g,r) \in \mathcal S} \ell_{g,r}(x)\, e_{g,r}\big)^2$ with residuals
$e_{g,r} = y_{g,r} - \hat b(x_{g,r})$, which carries the factor $1 + (n_{\rm real} - 1)\rho$ without estimating $\rho$;
$\widehat{\rm SE}(x) = \sqrt{\widehat{\rm Var}(\hat b(x))}$ is what the table reports (the $\hat\tau/\sqrt{N^{\rm loc}_{\rm eff}}$
of §2.4 stays as a column, labelled the independent-rows value). Settles the handoff's point 10.

(D-063.4) **`off_support`.** At the root $\tilde d_i$ of a real node, the kernel mass $M(x) = \sum_n \varpi_n(x)$ over
*all* rows (as the failure rate's denominator), at $x = (\ln\tilde d_i, \varphi_i, \ln(\hat\mu_i\hat d_i))$; the flag
fires when $M(x) < q_{0.05}\{M_{-n}(x_n)\}$, the 5th percentile (PROVISIONAL) of the rows' own leave-one-out masses
$M_{-n}(x_n) = M(x_n) - 1$, stored with the table; a flagged node is filled by D5, its root kept as a diagnostic
(`d_root_um`). Settles point 11.

(D-063.5) **The labelled comparison: penalized tensor-product B-splines (P-splines).** `response_estimator = "pspline"`:
cubic B-spline bases on each of the three axes (knots PROVISIONAL, e.g. 20, 12, 16), their tensor product, a
second-order difference penalty per axis with one smoothing parameter per axis or one shared, the penalized least
squares solved sparsely (the normal equations are of the basis size, not of $N$); the smoothing by grid search on
REML or GCV over an automatic interval (GCV can hold several local optima and undersmooths against REML). Evaluated
on the same grid and compared with (D-063.2) on the fixtures and on the production rows; never the deliverable.
Scheduled after the deliverable has passed gate 2.

**Why [user; run, sandbox; textbook; reasoning].** **[user]** both built, (A) the deliverable, the spline abandoned at
this size. **[run, sandbox 2026-10-20: `check_table_size.py`]** the dense spline of D-024 (v) needs $8N^2$ bytes =
8 TB and about $2 \times 10^5$ h at $N = 10^6$ (exponent 3.4 measured between $N$ = 3000 and 6000), while the local
kernel window at D-054's bandwidths already holds $N^{\rm loc}_{\rm eff} \approx 1.5 \times 10^4$ rows at
$N = 10^6$, so the spline's pooling across the domain buys nothing. **[textbook, from memory: Fan & Gijbels 1996,
local polynomial modelling]** the local linear fit is design-adaptive and free of the local constant's boundary and
density-gradient bias to first order (its bias is $O(h^2)$ in the curvature of $b$), which matters at the thin end
and at the band's edges (D-063.4); **[textbook, from memory: Liang & Zeger 1986]** the sandwich by cluster is the
standard variance for a linear estimator with correlated rows within independent clusters. **[reasoning]** (D-063.1):
the tilt is already the second axis, so a third coordinate carrying $-\ln\cos\hat\varphi$ repeats it and diverges at
steep tilts (the handoff's point 12: on 10 tables the 1st-99th percentile span of $\ln\hat\alpha$ was 50-77 % of its
min-max and 86-98 % of rows lay in the lower half of the normalised axis **[run, sandbox 2026-10-18:
`check09_lnalpha_span.py`]**); $\ln(\hat\mu\hat d)$ is bounded by $[\ln(\mu_{\min} d_{\min}), \ln(\mu_{\max} d_{\max})]$ up
to the fit's scatter, and it is the same computed quantity on a phantom row and on a real query, whereas
$\hat\alpha_{g,r}$ used the measured tilt while the second axis held the true one. (D-063.4): at fixed
$(\varphi_i, \hat\mu_i\hat d_i)$ the phantoms cover only $d \in [\hat\mu_i\hat d_i/\mu_{\max}, \hat\mu_i\hat d_i/\mu_{\min}]$, a
bracket of $\ln(\mu_{\max}/\mu_{\min})/\ln(d_{\max}/d_{\min})$ = 0.769 of the axis **[run, sandbox:
`check_table_size.py` (3)]**, and the C5 node was inverted outside it unflagged (mass 2.05 against a 5th percentile
of 3.06) **[run, sandbox 2026-10-18: `check_table_se_coverage.py`]**. (D-063.5): the P-spline is the natural
linear-in-$N$ global alternative; the smoothing-selection facts are read in full text **[PubMed FT: Li & Cao 2022,
Stat Comput, PMC9672641, doi:10.1007/s11222-022-10178-z -- penalized B-splines are banded, so a fit costs $O(nK)$
per smoothing value; GCV can have several local optima and tends to underestimate the smoothing parameter against
REML (they cite Reiss and Ogden for the theory); a grid search over a wide interval is recommended]**.

**What it implies for code and tables.**
- `table.table_inputs(d, phi, mu_hat, d_hat)` returns $(\ln d, \varphi, \ln(\hat\mu\hat d))$; the stored ranges
  become the grid's bounds; `tau_kernel_bandwidth = (0.15, 5.0, 0.3)` in raw coordinates (D-054 (ii)), `validate`
  requires three; `invert_node(d_hat, phi, mu_hat, table, cfg)`; `correct_nodes` passes `res.mu_hat_per_um` and
  `res.d_hat_um`; the per-node CSV adds `mu_d_hat`.
- `analysis/table.py`: `fit_table` builds the grid, the local linear weights by a binned or kd-tree kernel sum
  (compact support at 4 bandwidths, PROVISIONAL), the sandwich SE by draw, the kernel statistics, the LOO masses
  and their 5th percentile; `BiasTable` holds the grid arrays and the interpolant; `table_io` writes them. The
  P-spline comparison in its own module (`analysis/table_pspline.py`), the basis from `scipy.interpolate.BSpline.design_matrix`,
  sparse normal equations (`scipy.sparse`), REML/GCV by grid.
- Oracles (Block 14): on the 3-D fixture $b = 1 + 0.06/d + 0.15\sin^2\varphi + 0.05\ln(\mu d)$ with D-062's design,
  $\hat b$ within 4 SE of $b$ at every grid point in the band and the sandwich SE within 25 % of the across-table SD
  (60 tables); the local linear fit's weights sum to one and annihilate a linear $b$ exactly (1e-10); the
  interpolant reproduces the grid values; `off_support` fires on the C5 node and not on a node in the band's middle;
  the P-spline agrees with the local linear fit within 2 SE on the fixture.
- The thin-plate document of 2026-10-18 keeps its findings; they no longer describe the deliverable (the theory chat
  informed, 2026-10-20).

**What it does not decide [open].** The bandwidth multiplier's CV (proposal above); the grid sizes; the kernel's
truncation; the knots and penalty orders of the P-spline and whether REML or GCV chooses its smoothing; whether
(D-063.3)'s sandwich replaces or sits beside a bootstrap over draws; the `off_support` percentile (5 %, PROVISIONAL).

**Sources.** **[user]** 2026-10-20, 16:11. **[KB]** D-024, D-030, D-031, D-053, D-054, D-059, D-060, D-062; SPEC §2.4,
§8; `docs/TEEG_thin_plate_spline_in_3D/` §3.12. **[run, sandbox, 2026-10-20]** `check_table_size.py`,
`check_design_effect.py`, `fig_support_band.py`. **[PubMed FT]** Li, Z. & Cao, J. (2022), *Stat Comput* 33:1,
PMC9672641, doi:10.1007/s11222-022-10178-z (read in full). **[PubMed, searches]** "local linear kernel regression
bandwidth bias boundary": 1 record, an applied regression-discontinuity study, not used; "P-splines penalized
B-splines smoothing": 7 records, Li & Cao the one methods paper with full text; "cluster robust sandwich variance
... kernel regression" and "Abel inversion line spread function": 0 records. **[bioRxiv]** category scan,
bioinformatics, last 14 days, 30 titles: nothing on kernel regression, P-splines or PSF projection; the connector
has no keyword search. **[textbook, from memory]** Fan & Gijbels (1996); Liang & Zeger (1986); Eilers & Marx (1996)
for P-splines.

**Status:** active; not implemented. Supersedes D-024 (v)'s thin-plate spline as the response estimator and the
spline of (D-060.1) (the conditional target and the per-draw table of D-060 stand); amends D-053 (iii)'s coordinate
and retires D-054 (i)'s normalisation; settles points 10-12 of the 2026-10-19 handoff's §5.2 and D-062's open
estimator. Block 14. [updated 2026-10-20, later 2] (D-063.5), the P-spline comparison, is dropped by **D-067.3** [user]; the local linear estimator is the one built. [updated 2026-10-21, later 2: (D-063.1)-(D-063.4) implemented, Blocks 14.3-14.6 (`779bec0`, local): the coordinates, the local linear fit on a grid with D-071's multiplier, the sandwich SE (D-071.2), `off_support`; the grid arrays equal a brute-force evaluation to $10^{-12}$ and over 60 tables the sandwich SE matches the across-table SD **[run, sandbox]**]

---

## D-064 -- Block 13: the iteration's background mask reads a scene input (the cell's traced segments and soma), empty on a phantom

**Date:** 2026-10-20, 16:11 (Europe/Rome), the user's answer to point (e) of the assistant's reply of 2026-10-20 ("DO
as you suggest"). **Binds:** `analysis/node_pipeline.measure_node` (a new optional argument), the moved
`block_background` / `far_from_dendrites` of Block 13, `analysis/survey.node_planes` and `analysis/cell` (which build
the input on real data), `analysis/phantoms.run_replicate` (which builds it for a phantom), Block 13's oracle on
`tests/smoke/fixtures_cell.py`; D-061 (e).

**Decision [user].** The assistant's proposal (A): `measure_node` receives the scene -- the segment table and the
soma discs as `survey.node_planes` builds them -- and uses it for the gradient energy's background $B_{j,k}$;
$\bar B_j(k)$ keeps its own-branch mask.

**Statement.** For node $j$ and plane $k$ of $K_j$: $B_{j,k}$ (grey levels, measured), the background that normalises
the iteration's gradient energy $G^{(h)}_{j,k} = B_{j,k}^{-2}\int_{-h}^{h}(\partial_v \tilde I_j(v; k))^2\,dv$ (D-040,
D-042, D-052.3), is the interquartile mean of the pilot block's pixels farther than (the segment's larger end radius
+ margin) from every segment of the **scene** and farther than (soma radius + margin) from every soma disc, the margin
4.0 um shrinking through 3, 2, 1, 0.5 um until 5 % of the block's pixels remain (today's `block_background`,
PROVISIONAL readings of D-036/D-042). The scene is an input of `measure_node(branch, i, provider, cfg, reg=None,
scene=None)`: on real data the (n, 9) table of every dendrite segment whose bounding box meets the block (parent end,
child end, the two Allen radii, `in_stretch`) and the (m, 3) soma discs, built from the SWC by the caller exactly as
`survey.node_planes` builds them; on a phantom the tube's own branch segments and no soma. `scene=None` means the
node's own branch and no soma. $\bar B_j(k)$, the fit's background (D-018.1), keeps its mask on the node's own
branch (the SPEC §2.2 deviation recorded on 2026-10-16 stands as a separate question).

**Why [user; src; reasoning].** **[user]**. **[src, `84ea91f`]** `block_background` masks every traced segment and
the soma through `survey.node_planes`'s `segments` and `soma`; `measure_node` holds only its `Branch`. **[reasoning]**
$B_{j,k}$ is per plane: another dendrite or the soma in the block is dark in the planes where it is in focus, so
leaving its pixels in the mean lowers $B_{j,k}$ there, raises $G_{j,k}$ there, and the argmax can move toward the
other structure's plane; masking the own branch only would change $B_{j,k}$ against every recorded Cell 4j/4k/4m run
on Allen's cells and make real nodes and phantoms obey different rules. A scene is an input the real data supplies,
which D-057 (iii) allows to differ; a phantom, a single tube, has none.

**What it implies for code and tables.** The scene builder moves out of `survey.node_planes` into a function of
`analysis/cell.py` (or the Block 13 module) that `survey`, `cell.run_cell` and the calibration call; `run_replicate`
builds the phantom's scene from its branch; Block 13's oracle: on `fixtures_cell.py` the moved iteration with the old
fit reproduces the script's `gradient_iteration` byte for byte, which requires this input; a smoke check that
`scene=None` equals a scene of the own branch alone.

**What it does not decide [open].** The margin 4.0 um and its shrinking ladder (PROVISIONAL, D-036); whether
$\bar B_j(k)$'s mask should also read the scene (SPEC §2.2 says "any SWC segment in the block", the code masks the own
branch).

**Sources.** **[user]** 2026-10-20, 16:11. **[KB]** D-036, D-040, D-042, D-052, D-057, D-061. **[src]**
`scripts/plane_differences.py` l. 219-284, `analysis/survey.py` l. 158-215, `analysis/node_pipeline.py` l. 128-148,
206 at `84ea91f`.

**Status:** active; not implemented; settles D-061 (e). Block 13. [updated 2026-10-20, later 6: implemented, Blocks 13.2-13.3 (`plane_iteration.Scene`, `cell.cell_scene`; commits `e50abd9`, `70136ca`, local)] [updated 2026-10-21: pushed at the user's agreement, `84ea91f..888452a` on `sci/diameter-pipeline`]

---

## D-065 -- The real-cell programme in Colab: the fitted optical depths of the real neurons across cells against the phantoms' ranges, Allen's radii across cells for $d_{\max}$, the registration survey for the node jitter; no pinned-kernel comparison

**Date:** 2026-10-20, 16:11 (Europe/Rome), the user's answers to point 9 and to Q7, Q8, Q9 of point 16 of the
assistant's reply of 2026-10-20. **Binds:** the phantom laws and ranges of D-024 (iii)/D-059.1 (`PhantomConfig.d_range_um`,
`mu_range_per_um`); the coverage check of D-059.1 (its form); `PhantomConfig.jitter_xy_um`, `jitter_z_um` (D-024 (vii))
and the registration survey (`scripts/registration_survey.py`, notebook cell 3b); `scripts/allen_radius_distribution.py`
(notebook cell 6); `MeasureConfig.profile_half_um`, `block_half_um` as $d_{\max}$ moves; Q7 of the 2026-10-16
handoff; the simulation hash.

**Decision [user].** (9) "For now keep the ranges we have agreed on and while we run the 100000 draws I would do an
extensive analysis on the real neurons across cells in colab and draw the distributions of alpha values." (Q7) "do
not run for now the labelled comparison." (Q8) "On colab we will run through all the cells and construct the
distribution of all radius across cells so that this is settled." (Q9) "Let's implement this."

**Statement.**

(D-065.1) **Ranges and the coverage check.** The laws and ranges of D-024 (iii) stay ($d$ log-uniform on [0.2, 4] um
TEMPORARY, $\varphi$ uniform on [0, 90) deg, $\theta$ uniform on [0, pi), $\mu$ log-uniform on [0.3, 3] /um). The
coverage check of D-059.1 is an empirical survey in Colab: Block 13's chain run on the real neurons across cells, and
the distributions of the fitted optical depth $\hat\alpha_i = \hat\mu_i \hat d_i / \cos\varphi_i$ and, under D-063.1, of
$\hat\mu_i \hat d_i$ drawn per bin of $\hat d_i$ (the form of the assistant's point 9: the real values against the
phantoms' per bin of fitted diameter, both sides fitted by the same estimator), run while the davinci draws run
**[user]**. A real node outside the phantoms' band at its size is flagged by D-063.4's `off_support` (and
`alpha_out_of_domain` where the global range is left) and filled by D5; a systematic excess decides a wider
$\mu$ range by a later decision.

(D-065.2) **$d_{\max}$.** `allen_radius_distribution.py` (cell 6, SWC only) runs over all the cells of the archive in
Colab; the distribution of Allen's dendritic radii across cells sets $d_{\max}$ (D-024's open point), and with it
`profile_half_um` and `block_half_um` where the thickest dendrites need a longer line (D-038, Q8). The value is a
later decision; the procedure is this.

(D-065.3) **Node jitter.** D-024 (vii)'s knobs are set from the registration survey (cell 3b): the lateral SD
`jitter_xy_um` from the survey's lateral offsets $s^*$ and the depth SD `jitter_z_um` from its depth offsets
$\Delta z^*$, per node independent normal as coded (Block 6), the values a later decision after the survey has run
across cells; both inside the simulation hash.

(D-065.4) **Q7.** The pinned-kernel thin-node comparison (D-052 (g)) is not run now; the deliverable is D-052's fit
alone.

**Why [user; reasoning].** **[user]** throughout. **[reasoning]** (D-065.1): with the third axis a fitted quantity, the
only test of the phantoms' support is the real distribution of that quantity at each size; the survey needs Block
13's chain on the real stacks, which only Colab reaches. (D-065.2): the radii are in the SWCs, no images needed, so
the survey is cheap. (D-065.3): a phantom's SWC node sits on the true axis; a real tracing is off by $s^*$ laterally
and $\Delta z^*$ in depth, which moves the pilot's window and $K_j$ -- a nuisance the table must contain if the real
chain is exposed to it (D-057 (iii)).

**What it implies for code and tables.**
- A Colab cell (new, after Block 13) that runs `run_cell.py` without a table on every cell of the archive and
  draws, per bin of $\hat d$, the distributions of $\hat\alpha_i$ and $\hat\mu_i\hat d_i$ against the phantoms' band of
  the current ranges (the band's edges analytic: $\ln\mu_{\min} + \ln d$ and $\ln\mu_{\max} + \ln d$ in
  $\ln(\hat\mu\hat d)$), with the share of real nodes inside; the figures and a CSV under `MyDrive/diameters/survey/`.
- Cell 6 run over all cells; its summary read for $d_{\max}$; cell 3b run over all cells; its offsets' SDs read for
  the jitter; the three values entered in `config_production.json` before the production run, since all three are
  inside the simulation hash.
- **The assistant's recommendation, recorded, not decided:** the three Colab surveys and the kernel calibration of
  D-066 all change the simulation hash; a production of $10^6$ realizations started before them would be re-rendered
  if any of them moves a range or a knob. The davinci pilot (about 200 realizations: timing, failure-rate curve,
  D-054 (iii)) can start as soon as Block 14 passes gate 2; the production launch should wait for the surveys' values
  and the calibrated kernel. The Block 13 run on real cells that feeds (D-065.1) also provides D-066's calibration
  data and the real-data check of the 2026-10-16 handoff §7, so the order costs little.

**What it does not decide [open].** $d_{\max}$'s value and the line/block lengths that follow; the jitter SDs; whether
the jitter is independent per node or correlated along the branch (coded independent); the threshold of
(D-065.1)'s share that triggers a wider $\mu$ range; whether the survey draws $\hat\alpha_i$ or $\hat\mu_i\hat d_i$ as
the primary figure (both, proposed).

**Sources.** **[user]** 2026-10-20, 16:11. **[KB]** D-024 (iii), (vii), D-038, D-052 (g), D-057, D-059, D-063;
the 2026-10-16 handoff §7 (Q7-Q9); the 2026-10-19 handoff §5.2 items 9, 16. **[src]** `scripts/allen_radius_distribution.py`,
`scripts/registration_survey.py`, `config.py` (`jitter_xy_um`, `jitter_z_um`, `d_range_um`) at `84ea91f`.

**Status:** active; not implemented; settles D-059.1's check as a survey, D-024's $d_{\max}$ and jitter procedures,
and closes Q7 for now. [updated 2026-10-20, later 2] The recommended order is the decision, **D-067.1** [user]: the surveys and the calibration precede the production launch; the pilot may run earlier.

---

## D-066 -- The renderer's defocus kernel is no longer Gaussian: the SGT family, calibrated in Colab, in the synthetic chain as in the real one (I-002 adopted)

**Date:** 2026-10-20, 16:11 (Europe/Rome), the user's answer to Q10 of point 16 of the assistant's reply of 2026-10-20.
**Binds:** Block 4's renderer (`model/kernel.py` `sigma_r`, `model/render.py` `render_transmittance`, the FFT
backend's Gaussian split, `RendererConfig.kernel_family`, `kernel_table_delta_um`, `kernel_table_sigma_um`,
`kernel_continuation*`, `config.KERNEL_FAMILIES`); Block 10's calibration (`analysis/calibration.py`: the statistic,
`fit_growth`, `kernel_from_growth`) and its Colab cell 7; D-026, D-027, D-052 (f), I-002; gate 1 (Block 5 status)
and procedure §3.10's first row; the simulation hash; the real-data check of the 2026-10-16 handoff §7.

**Decision [user].** "The kernel is not anymore gaussian. Check it. We will obviously [use] this new kernel to both
real and synthetic chains." Read with D-052 and D-061.5: the defocus kernel of the renderer becomes the same
skewed generalized-$t$ family the real chain's fit uses (Block 12), its parameters calibrated on Allen's stacks in
Colab (Block 10, D-061.5) and loaded into the davinci configuration; I-002, "a non-Gaussian kernel family with the
Gaussian as a special case", becomes a decision.

**Checked [src, `84ea91f`].** At today's head the two chains differ: the renderer blurs each slab at signed defocus
$\delta$ with a circular Gaussian of SD $\sigma_{\rm r}(\delta)$ read from the table (`kernel_family = "gaussian_table"`;
`"empirical"` raises `NotImplementedError`, reserved for Block 10), while the real chain's final fit uses
$g_{w,\kappa}$ of (D-052.1) with $(w, \beta, \xi, \zeta)$ fitted per node. The renderer's Gaussian also enters the FFT
backend's split by $\sigma$ (`fft_split_sigma_um`, `fft_wrap_sigmas`, `far_grid_per_sigma`), the padding rule
(`pad_um` $\ge 3\sigma_{\rm r}$ of the farthest slab), gate 1's budget $\sigma_{\rm fit}^2 = \sigma_{\rm r}(0)^2 + p_{\rm x}^2/4$
and Block 10's growth statistic $G(\delta) = \sigma_{\rm r}(\delta)^2 - \sigma_{\rm r}(0)^2$ (a Gaussian core width).

**Statement.**

(D-066.1) **Family.** For a slab at defocus $\delta$ (um) the renderer's point-spread function is the isotropic 2-D
kernel $h_\delta(r)$, $r$ (um) the in-plane distance, whose line projection is the symmetric SGT profile kernel of
(D-052.1), $g_{w(\delta), \kappa(\delta)}$ with $\kappa(\delta) = (\beta(\delta), \xi(\delta), 1)$:
$$h_\delta(r) = -\frac1\pi \int_r^\infty \frac{g'_{w(\delta),\kappa(\delta)}(v)}{\sqrt{v^2 - r^2}}\,dv \qquad (\text{the inverse Abel transform; } \textstyle\int_{-\infty}^{\infty} h_\delta\big(\sqrt{u^2 + v^2}\big)\,du = g_{w(\delta),\kappa(\delta)}(v)),$$
so that a flat tube rendered with $h_\delta$ shows, across its axis, exactly the profile kernel the fit assumes.
$h_\delta \ge 0$ for every member, because every member is even and decreasing in $|v|$ (so $-g' \ge 0$ on $v > 0$);
checked numerically on the generalized normal at $\beta$ = 1, 1.5, 2, 3, 4, 6, 10 and the Student members at $\xi$ = 0.5,
2 (minimum of $h$ at least 0 to 1e-4 of its peak) **[run, sandbox 2026-10-20: `check_abel_sgt.py`]**. The Gaussian is
the member $(2, 0, 1)$, so today's table is the special case.

(D-066.2) **The skew is not rendered isotropically.** An isotropic PSF projects to an even profile kernel; a
two-piece skew ($\zeta \ne 1$) in the fit's kernel can only come from an asymmetric PSF with a direction fixed in the
image frame (oblique illumination), which projects differently onto measuring lines of different heading. The
renderer therefore uses $\zeta$ = 1; the fitted $\hat\zeta_j$ of the real nodes is a diagnostic read in the Colab
survey against the heading $\theta_j$ (a systematic $\hat\zeta$ with a consistent image-frame direction would make a
directional PSF a later decision).

(D-066.3) **Calibration (Block 10, Colab, D-061.5).** The kernel's defocus dependence is a table at the plane-offset
knots, as today's $\sigma_{\rm r}$ table: $(w(\delta_m), \beta(\delta_m), \xi(\delta_m))$ at $\delta_m$ = 0, 0.14, ...,
measured on the calibration nodes of D-061.4 (thin branches by Allen's diameter for $|\delta| \le$ 0.84 um, dendrites
of about 0.8 um beyond) by the D-052 fit at each plane of the scan (the per-plane $\hat w$, $\hat\beta$, $\hat\xi$ of
`fit_sgt.fit_plane` on the scan's profiles), the continuation beyond the last knot and the sub-plane shape open as
for $\sigma_{\rm r}$ (D-026). The calibrated table enters `RendererConfig` (new fields beside `kernel_table_sigma_um`,
inside the simulation hash) and is loaded into the davinci configuration (D-061.5).

(D-066.4) **Both chains.** The real chain's fit (Block 12) already uses the family with every parameter free per node;
the synthetic chain renders with the calibrated member per defocus. Phantoms and real nodes then share the kernel
family end to end (D-057 (iii) in the renderer's sense: the simulated microscope blurs as the real one is measured
to).

**Why [user; run, sandbox; reasoning].** **[user]**. **[reasoning]** the table corrects the bias of *this* estimator
on *this* simulated microscope (procedure §3.2-3.3); if the real blur is not Gaussian, phantoms blurred by a Gaussian
carry a kernel-shape bias the table cannot see and the real nodes keep (D-052's Why: the Gaussian's 1.530 against
1.493 um on the $t_4$ truth at 1.5 um); a fit whose kernel family contains the renderer's is the consistent pair.
**[run, sandbox]** the inverse Abel transform keeps every symmetric member a valid PSF, so the family change costs
no new positivity condition; the skew is the one member that has no isotropic source **[reasoning, exact]**.

**What it implies for code and tables.**
- Block 4: `KERNEL_FAMILIES` gains `"sgt_table"`; `model/kernel.py` returns, per slab, the tabulated radial PSF
  $h_\delta(r)$ (the inverse Abel transform computed once per knot and interpolated in $\delta$, PROVISIONAL), the
  renderer's FFT path convolving with a tabulated 2-D kernel instead of an analytic Gaussian (the Gaussian split by
  $\sigma$ replaced by a split by the kernel's second moment or by a direct per-knot convolution); the padding rule
  from the kernel's tail at the configured truncation; the `"gaussian_table"` family stays selectable (the
  comparison, and the member $(2, 0, 1)$ as the smoke oracle: `"sgt_table"` at the Gaussian shape reproduces
  `"gaussian_table"` to roundoff).
- Block 10: the plane scan's statistic becomes the per-plane SGT fit's $(\hat w, \hat\beta, \hat\xi)$ (and $\hat\zeta$
  as the diagnostic of D-066.2); `fit_growth` generalises from one width curve to the three curves, the node's
  $d$ still the nuisance it is today (the width-diameter trade-off of thin nodes, D-052's impostors, is the
  calibration's known weakness; the far-field nodes of D-026, thicker, constrain the tails); `kernel_from_growth`
  writes the new table; the Colab cell 7 and the hand-over of D-061.5.
- Gate 1 and procedure §3.10's first row restated for the family (the Gaussian budget no longer applies); the
  simulation hash changes; `checks/` gains the Abel check.
- Order: the calibration needs Block 13's chain on the real stacks (the SGT fit per plane), so it follows Block 13
  and precedes the production (the renderer's kernel is inside the simulation hash; D-065's recommendation).

**What it does not decide [open].** The parametrisation in $\delta$ (a table at the knots, as proposed, or a
parametric growth of $w$ with fixed shapes); the continuation beyond the last knot and the sub-plane shape; the
truncation and grid rules of the 2-D kernel; how the calibration separates the kernel's width from a thin node's
diameter (procedure §3.4 step 2's self-consistency loop, not built); whether $\xi(\delta)$ and $\beta(\delta)$ are
identifiable from the scans or must be held at in-focus values; what to do if the real $\hat\zeta$ shows a
consistent direction (D-066.2); the cost of the renderer with tabulated kernels (to be measured with Block 13's
timing, D-062).

**Sources.** **[user]** 2026-10-20, 16:11. **[KB]** D-024 (i), D-026, D-027, D-052, D-057, D-061, I-002; procedure
§3.2-3.4, §3.10; SPEC Blocks 4, 10, 12. **[src]** `model/kernel.py`, `model/render.py`, `config.py`
(`KERNEL_FAMILIES` l. 51, `RendererConfig` l. 172-174), `analysis/calibration.py` at `84ea91f`. **[run, sandbox,
2026-10-20]** `check_abel_sgt.py`. **[PubMed]** "inverse Abel transform point spread function line spread function
projection microscopy" and "Abel inversion line spread function point spread function": 0 records; the Abel
relation between an isotropic PSF and its line-spread function is **[textbook, from memory]**.

**Status:** active; not implemented; adopts I-002 for the renderer and the calibration; supersedes D-026's "the kernel
stays Gaussian for now" and D-052 (f); D-024 (i)'s `kernel_continuation` question moves to the new table's
continuation. Blocks 4 and 10, after Block 13. [updated 2026-10-20, later 2] **Withdrawn by D-067** [user, 16:41: "keep this in that way please without overengineering"], never implemented: the renderer stays Gaussian with the calibrated $\sigma_{\rm r}(\delta)$ table, the fit SGT. (D-066.1)'s Abel finding and (D-066.2)'s remark on the skew stay on record for a later decision; I-002 returns to an idea.

---

## D-067 -- The order of operations (Colab surveys and calibration before the production); the renderer stays Gaussian (D-066 withdrawn); the local linear estimator alone; Block 10's thresholds in Allen's diameter

**Date:** 2026-10-20, 16:41 (Europe/Rome), the user's answers to the three push-backs and the last open point of the
assistant's reply of 16:11. **Binds:** the schedule of D-061.5 / D-062.2 / D-065 (what runs before the production
launch); Block 4's kernel (`RendererConfig.kernel_family = "gaussian_table"`, the $\sigma_{\rm r}(\delta)$ table,
`config.KERNEL_FAMILIES`), Block 10 (D-026, D-027) and I-002; `CorrectionConfig.response_estimator` (D-063.5);
`CalibrationConfig` (new Allen-diameter fields, D-061.4).

**Decision [user].** (1) "Agreed on swapping the order." (2) On the two chains differing (the renderer Gaussian, the
fit SGT): "Let's keep this in that way please without overengineering." (3) "Let's just implement A." (4) Block 10's
thresholds in Allen's diameter as the assistant proposed: thin $\le$ 0.6 um, far field 0.6-0.8 um.

**Statement.**

(D-067.1) **Order.** Block 13 (the iteration and the SGT fit inside `measure_node`, D-052, D-061, D-064) -> in Colab,
on the real cells: Block 13's chain across cells (the survey of D-065.1 and the real-data check of the 2026-10-16
handoff §7), cell 6 (Allen's radii across cells, D-065.2) and cell 3b (the registration survey, D-065.3), and
Block 10's calibration of $\sigma_{\rm r}(\delta)$ on the nodes of (D-067.4) -> the production configuration (the
$\mu$ range confirmed or widened, $d_{\max}$ with `profile_half_um` and `block_half_um`, the jitter SDs, the
calibrated kernel table) -> Block 14 (D-062, D-063) -> gate 2 in the sandbox -> the davinci pilot (about 200
realizations: the per-realization time, the failure-rate curve for D-054 (iii)) -> the production of D-062.2 ->
merge and tests on davinci (D-061.5). The pilot may run as soon as Block 14 passes gate 2, on a provisional
configuration, since it measures time and failure rates; the production waits for the final configuration, every
part of which sits inside the simulation hash.

(D-067.2) **Kernels.** The renderer keeps its circular Gaussian defocus kernel of SD $\sigma_{\rm r}(\delta)$ read from
the table (`gaussian_table`), the table to be calibrated by Block 10 as D-026 and D-027 plan; the fit keeps the
SGT of D-052. **D-066 is withdrawn** before any implementation; I-002 returns to "idea, to discuss further, not
scheduled". Kept on record from D-066: every symmetric SGT member is the line projection of a nonnegative isotropic
PSF (inverse Abel transform; `check_abel_sgt.py`), the two-piece skew of none. What this accepts: a difference
between the real blur's shape and the Gaussian the phantoms are rendered with is a model error the table does not
contain; the real-data check (the fitted $\hat w$, $\hat\kappa$ of real nodes against rendered phantoms of the same
$\hat d$) reports it.

(D-067.3) **Estimator.** `response_estimator = "local_linear"` (D-063.2) is the one estimator built; D-063.5's
P-spline comparison is not built. `"tps_spline"` stays as the existing small-table code path for the fixtures and
suites (it refuses $N_{\mathcal S}$ above a configured cap), never the deliverable.

(D-067.4) **Block 10's selection, in Allen's diameter $2 r_{{\rm A},j}$ (D-061.4), PROVISIONAL values:** the thin set
$2 r_{{\rm A},j} \le$ 0.6 um for the near field (plane offsets up to 0.84 um); the far-field set
$0.6 < 2 r_{{\rm A},j} \le$ 0.8 um (D-026) for the offsets beyond 0.84 um; the remaining criteria -- flat
($\varphi_j \le$ `node_phi_max_deg`), faint ($\hat\alpha_j \le$ `node_alpha_max`), in $\mathcal S$ -- read the pilot's
values (`d_pilot_um`, `alpha_pilot`, the pilot's status and flags; D-061.2's columns). Config fields, names fixed in
SPEC: `node_allen_diameter_max_um` = 0.6, `far_field_allen_diameter_um` = (0.6, 0.8) (`# source: D-067`).

**Why [user; reasoning].** **[user]** throughout. **[reasoning]** (1) every Colab output changes the simulation hash,
so a production launched before them would be re-rendered (the assistant's recommendation of D-065, now the
decision). (2) the fit's kernel is free per node, so the measurement already adapts to the real blur's shape; the
renderer's Gaussian is the simplest microscope whose remaining mismatch the real-data check can measure; the
renderer and calibration rebuild D-066 would have needed were a second project. (3) one estimator, one code path.
(4) Allen's radius on specimen 529878215 is about 0.25 um in every bin of $\hat d$ (SPEC §8), so a thin set at
$2 r_{\rm A} \le$ 0.6 um admits the branches Allen traced as thin; the far-field band is D-026's "just under 0.8 um".

**What it implies for code and tables.**
- The implementation plan of 2026-10-20 (`docs/TEEG_diameter_table_implementation_plan_2026-10-20.md`, to be
  committed with Block 13) follows (D-067.1); the HPC runbook's production step gains the precondition "the final
  configuration".
- Block 4 unchanged; `KERNEL_FAMILIES` unchanged; Block 10's statistic and `fit_growth` unchanged; its
  `calibration_reasons` reads the two new fields and the pilot's columns.
- Block 14 builds `local_linear` only; no `table_pspline.py`.
- SPEC §4's binding list: D-066 struck as withdrawn, D-067 added.

**What it does not decide [open].** The thresholds' values beyond PROVISIONAL (set after the Colab calibration run);
whether `"tps_spline"` is removed once Block 14 is in; the renderer's kernel shape in the long run (I-002, idea).

**Sources.** **[user]** 2026-10-20, 16:41. **[KB]** D-026, D-027, D-052, D-054, D-061 to D-066, I-002; SPEC §8
(Allen's radii on 529878215). **[run, sandbox, 2026-10-20]** `check_abel_sgt.py` (the fact kept on record).

**Status:** active; not implemented [updated 2026-10-21, later: (D-067.4) implemented, S2.1, `1aed5d6`, pushed at the user's agreement (`f4ac106..9c1329e` on `sci/diameter-pipeline`) -- `CalibrationConfig.node_allen_diameter_max_um` = 0.6 um and `far_field_allen_diameter_um` = (0.6, 0.8) um with `# source: D-067`, `node_dhat_max_um` removed, the other criteria read from the pilot; 27 of 27 mutants of the selection killed **[run, sandbox]**]. Supersedes D-066 (withdrawn, never implemented) and D-063.5; makes D-065's
recommended order the decision; settles D-061.4's thresholds PROVISIONAL. [updated 2026-10-23: the calibration nodes of (D-067.4) must also pass the patch isolation of **D-079.1**.] [updated 2026-10-28: (D-067.4)'s faint criterion ($\hat\alpha_j \le$ `node_alpha_max`) and, for the calibration nodes only, the flag `faint` within "in $\mathcal S$" superseded by **D-094** [user]: both are switches, off by default; $\mathcal S$ itself is unchanged.]

---

## Ideas for future parts

Add entries as `I-nnn` with the fields in "Entry format"; an idea that becomes
a decision gets a `D-nnn` entry that cites the `I-nnn`.

---

## D-068 -- An iteration that ends without converging keeps its fit and stays in $\mathcal S$, flagged

**Date:** 2026-10-21, 09:32 (Europe/Rome), the user's answer to question 1 of the assistant's report of 2026-10-20
(Block 13 implemented). **Binds:** `analysis/node_pipeline.deliverable_flags` and `measure_node` (the flags of a node
whose Allen-guided iteration does not converge); Block 6's `phantoms.reject_reasons` and `SELECTION_FLAGS`, which stay
as they are; the `flags` column of the replicate rows (Block 6) and of the per-node CSV (Blocks 8, 11); Block 14's
failure count, in which these nodes count as retained; SPEC Block 13.3b's [to confirm].

**Decision [user].** "Keep but flag", of the two options put to the user: (A) not a criterion of $\mathcal S$, the
ending a column only (implemented on 2026-10-20 and recommended); (B) a criterion of $\mathcal S$.

**Statement.** For node $j$ (real, or a phantom's middle node) with the iteration's ending
$e_j \in$ `ITER_STATUSES` (D-042: `converged`, `cycle`, `max_rounds`, `fit_failed`, `no_plane`, `line_short`,
`square_small`), and $k^{\rm it}_j$ the last plane picked (D-052.3):

(D-068.1) **Kept.** When the iteration ends with a plane and a fit there -- $e_j$ = `cycle`, `max_rounds`, or `no_plane`
after round 1 -- the deliverable is the fit at $k^{\rm it}_j$, as for `converged`, and $e_j$ is not a criterion of
$\mathcal S$: the node is retained whenever its fit's status is `converged` and no flag of `SELECTION_FLAGS` is set
(Block 6's rule, unchanged).

(D-068.2) **Flagged.** Whenever $e_j \ne$ `converged`, `NodeResult.flags` gains the label `iter_<e_j>` (`iter_cycle`,
`iter_max_rounds`, `iter_no_plane`, `iter_fit_failed`, `iter_line_short`, `iter_square_small`), after Block 5's flags; the
label is not in `SELECTION_FLAGS`. The endings without a plane or without a fit there already fail $\mathcal S$ through
their status (`status:none`); for them the label names why **[assistant's reading of "flag": the user's question
concerned the endings with a plane; the label on the others costs nothing and keeps one rule]**. `iter_status` stays a
column.

**Why [user; reasoning].** **[user]** the choice. **[reasoning]** The table estimates the response of whatever the
chain returns, and phantoms and real nodes share the rule (D-057 (iii)), so a non-converged node's bias is inside the
table; removing such nodes would only raise the failure count. The label makes them countable: whether their ratio
$\hat d/d$ differs from the converged nodes' is a check on the per-draw table (D-060.2) once the production exists.
On Allen's data the 13 nodes of Cell 4j all converged (D-042's status) **[KB]**.

**What it implies for code and tables.** `deliverable_flags` takes the ending and appends the label; the Block 13
suite asserts the label for a non-converged ending, its absence for `converged`, and that `reject_reasons` does not
read it; SPEC Block 13.3b's [to confirm] closed. No configuration field: the estimator hash does not change for this
part (the flags are an output, not a setting).

**What it does not decide [open].** Whether the non-converged nodes are later removed or modelled apart if the check
above finds them biased differently; a per-ending treatment.

**Sources.** **[user]** 2026-10-21, 09:32. **[KB]** D-042, D-052.3, D-057 (iii), D-060.2, D-061.3. **[src]**
`analysis/node_pipeline.py`, `analysis/phantoms.py` at `888452a`.

**Status:** active; not implemented; closes SPEC Block 13.3b's [to confirm]. Block 13.6. [updated 2026-10-21, later: implemented, `f4ac106`, pushed at the user's agreement (`888452a..f4ac106`)]

---

## D-069 -- The iteration's first line holds at least 3 samples

**Date:** 2026-10-21, 09:32 (Europe/Rome), the user's answer to question 2 of the assistant's report of 2026-10-20.
**Binds:** `analysis/plane_iteration.iterate_planes` and `gradient_iteration` (round 1's line), `node_pipeline.measure_node`,
a new `MeasureConfig` field (`iter_min_first_samples`) and hence the estimator hash; the thin end of D-062.1's phantoms
and Block 14's failure count there.

**Decision [user].** "b", of three options for the thin end of D-062.1: (a) keep it, the phantoms counted as failures;
(b) floor the first line at 3 samples; (c) truncate the start radius at a positive floor.

**Statement.** Symbols of D-061 and D-044: $m$ = `iter_line_mult` = 3, $d^{(0)}_j = 2 r_{{\rm br},j}$ the start,
$\Delta v$ = `profile_step_um` (0.1144 um), $n_{\min}$ = `iter_min_first_samples` = 3. Round 1's line has half-length
$h_{j,1} = \max\left(m\, d^{(0)}_j/2,\ h_{\min}\right)$, $h_{\min} = \tfrac{n_{\min} - 1}{2}\,\Delta v$, so that it holds
$2\,{\rm round}(h_{j,1}/\Delta v) + 1 \ge n_{\min}$ samples; the line of every later round is unchanged,
$h_{j,i} = m\, d^{(i-1)}_j/2$ with $d^{(i-1)}_j$ the fit's, which the fit's lower bound $d \ge$ 0.05 um keeps at 3
samples or more at $m$ = 3 (${\rm round}(0.075/0.1144) = 1$) **[arithmetic]**. The start radius of D-062.1 is
unchanged ($d/2 + \epsilon$, redrawn until positive), and so is $r_{{\rm br},j}$ wherever else it enters (the pilot's
focus window, the two background masks). One rule for
phantoms and real nodes (D-057 (iii)).

**Why [user; data].** **[user]** the choice. **[run, sandbox, 2026-10-20]** Without the floor round 1's line has fewer
than 3 samples, and the iteration ends `line_short` with no plane, when $r_{\rm br} < 0.5\,\Delta v/m$ = 0.019 um: under
D-062.1's redraw 5.0 % of the phantoms at $d$ = 0.2 um, 2.9 % at 0.4 um, 0.2 % at 1.0 um
(`checks/block13_2026-10-20/check_line_short.py`), a failure no real node shares, since Allen's radii start the real
lines. With the floor such a phantom starts from the shortest line the gradient energy can be computed on; the round's
plane and the fits that follow decide its diameter, as for any other node.

**What it implies for code and tables.** `iterate_planes` takes the floor for round 1 (default none, so the Phase II
diagnostics of Cells 4j-4m and their checks against `84ea91f` are unchanged); `measure_node` passes
$h_{\min}$ from the configuration; `iter_min_first_samples` = 3 joins the estimator signature, so the estimator hash
changes again and every table before is refused; the Block 13 suite asserts the floor on a start too thin for 3
samples and its absence on a normal start.

**What it does not decide [open].** Options (a) and (c) are not taken; whether the diagnostics' iteration takes the
same floor (it matters only for Allen radii below 0.019 um).

**Sources.** **[user]** 2026-10-21, 09:32. **[KB]** D-042, D-044, D-057 (iii), D-062.1. **[run, sandbox]**
`checks/block13_2026-10-20/check_line_short.py` (+ `.log`) at `888452a`. **[src]** `analysis/plane_iteration.py`
(`iterate_planes`, `gradient_iteration`'s `line_short` rule) at `888452a`.

**Status:** active; not implemented; settles the thin end of D-062.1 found on 2026-10-20. Block 13.6. [updated 2026-10-21, later: implemented, `f4ac106`, pushed at the user's agreement (`888452a..f4ac106`)]

---

## D-070 -- The far-field calibration of D-026 becomes stage S2b, designed after Colab counts the far-field nodes; S4's code is written while S3 runs

**Date:** 2026-10-21, about 12:04 (Europe/Rome), the user's answers to the assistant's question card after S2 (the
implementation chat). **Binds:** the plan of 2026-10-20 (`docs/TEEG_diameter_table_implementation_plan_2026-10-20.md`
on `sci/diameter-pipeline`: its S2b row, S3 (d), S4); Block 10's far-field set (D-026, D-067.4;
`calibration.FAR_FIELD_NOT_BUILT`, SPEC Block 10 and section 8); `RendererConfig.kernel_continuation_slope` (D-024 (i)'s
configured $\gamma$); the order of D-067.1 as far as S4 is concerned.

**Decision [user].** "After Colab counts (Recommended)", of three options for the far-field calibration, which no stage
of the plan built: (A) design it after Colab's Cell 7 has counted the far-field nodes; (B) design it now, before
Cell 7 runs; (C) the near field only, keeping the configured slope. "Start S4 (Recommended)", of starting Block 14
while S3 runs in Colab or waiting for S3. (Also "Push S2", an action: `f4ac106..9c1329e`.)

**Statement.**

(D-070.1) **S2b.** The far-field calibration of D-026 -- the scan of the far-field set of D-067.4 ($0.6 < 2 r_{\rm A}
\le 0.8$ um; in $\mathcal S$, flat and faint by the pilot) on its planes with $|z_{{\rm ax},j} - z_k| > P\,\Delta z$ =
0.84 um, and the fit of (D-026.1) -- is a stage of its own, S2b. Its design (SPEC section 8: the plane range and side,
the line and block extent, the background, near-then-far or joint, the axis depth, further terms, the isolation
radius) is put to the user after Colab's Cell 7 has run on the S2 selection and reported, per cell, the far-field
calibration nodes (the log line, the scans file's `meta.nodes`, the pilot summary's `n_calibration_by_set`) and the
near-field growth fit. Until S2b exists, the renderer's continuation beyond the calibrated knots stays D-024 (i)'s
configured slope, $\gamma$ = 0.79.

(D-070.2) **S4 now.** Block 14 (S4) is written while S3 runs in Colab; its configuration values wait for S3, and S5
onwards wait for the final configuration (D-067.1 unchanged).

**Why [user; reasoning].** **[user]** both choices. **[reasoning]** D-027 (d): the number of flat thick nodes in a cell
decides whether more specimens are needed; and the far scan's plane range and window depend on how far the real
dips stay above the noise, which D-026 (c) estimated only with an illustrative kernel. A design made before those
numbers could serve nodes that do not exist in enough number. S4's code needs S3's values only as configuration
(the plan, section 2).

**What it implies for code and tables.** Nothing new to build for the counts (S2.1 reports them). The plan's S2b row
stays "proposed" until the design is agreed; S4 starts with its SPEC entry (block protocol).

**What it does not decide [open].** S2b's design; D-027's flat tolerance $\varphi_{\rm tol}$; whether more specimens
are needed for the far field.

**Sources.** **[user]** 2026-10-21, about 12:04 (the question card). **[KB]** D-024 (i), D-026, D-027, D-067; SPEC
section 8; the plan of 2026-10-20. **[src]** `sci/diameter-pipeline` at `9c1329e`.

**Status:** active; (D-070.1) waits for Cell 7 in Colab; (D-070.2) in progress. [updated 2026-10-21, later 2: (D-070.2) done, Block 14 written (14.1-14.6, local commits); (D-070.1)'s timing amended by **D-071.5**: S2b is designed now and its questions put to the user] [updated 2026-10-21, later 4: Block 14 pushed (`825bef1`); S2b's planes and line set by **D-074**, its background, statistic and fit open there]

---

## D-071 -- Block 14's open choices: the bandwidth multiplier by cross-validation with folds by draw, the by-draw sandwich the one SE, the grid at half the chosen bandwidths, `mu_d_out_of_domain` a label; the far-field calibration straight after Block 14

**Date:** 2026-10-21, 12:30 (Europe/Rome), the user's answers to the four questions of the assistant's message after
SPEC Block 14 was drafted (`f63fe9f`, its items marked [to confirm]), and the user's remark on the far-field
calibration. **Binds:** `CorrectionConfig.bandwidth_cv_grid`, `table.choose_multiplier` and `table.grid_axes`; the SE a
`BiasTable` reports (`se_hat`, `meta["se_method"]`); `invert.invert_node`'s flags (`LABEL_FLAGS`); the plan's S2b and
its place in the order (D-070.1).

**Decision [user].** "The four questions from my last message still decide the table code: 1) b 2) a 3) half
bandwidth but on the ones chosen by the cross validation 4) Just flag it"; "A gap I found: no stage of the plan builds
the far-field calibration [KB: D-026]"; "After block 14 straight to the far field calibration".

**Statement.** Notation of SPEC Block 14: $h = c\,(h_1, h_2, h_3)$, $(h_1, h_2, h_3)$ = `tau_kernel_bandwidth` =
(0.15, 5 deg, 0.3) in $(\ln d, \varphi, \ln(\hat\mu\hat d))$; $g$ a draw, $K$ = `spline_cv_folds`.

(D-071.1) **The multiplier $c$ by cross-validation** (answer 1, b): $K$-fold, folds by draw (draw $g$ in fold
$(g - 1) \bmod K$, D-062.3), over `bandwidth_cv_grid` = (0.5, 0.7071, 1, 1.4142, 2) (the values the assistant's,
PROVISIONAL); the score is the mean squared error of the held-out rows' ratios read from the table of the other folds;
a choice at either end of the grid is flagged (`cv_at_grid_edge`); a grid of one value fixes $c$.

(D-071.2) **The SE is the sandwich by draw** (answer 2, a): the cluster-robust variance of D-063.3,
$\widehat{\rm Var}(\hat b(x)) = \sum_g (\sum_r \ell_{g,r}(x) e_{g,r})^2$, the one SE the table reports; the
independent-rows value $\hat\tau/\sqrt{N^{\rm loc}_{\rm eff}}$ stays beside it, labelled; no bootstrap over draws.

(D-071.3) **The grid** (answer 3): regular axes with spacing at most $c\,h_k/2$ on axis $k$, $c$ the multiplier the
cross-validation chose, so the grid follows the bandwidths actually used ((41, 37, 37) points at $c$ = 1 on the
default ranges).

(D-071.4) **`mu_d_out_of_domain` is a label** (answer 4, "just flag it"): a real node whose $\hat\mu_i\hat d_i$ lies
outside the table's third axis keeps the flag and is inverted, not filled for it; the assistant's reading of how, PROVISIONAL:
$\ln(\hat\mu_i\hat d_i)$ clamped to the axis' nearest end.

(D-071.5) **Order.** The far-field calibration of D-026 (stage S2b) is taken up straight after Block 14, ahead of the
Colab counts on which D-070.1 had made its design wait; amends D-070.1's timing (its content stands).

**Why [user; run, sandbox; reasoning].** **[user]** throughout. **[reasoning]** (1) one multiplier keeps D-054 (ii)'s
proportions between the axes and leaves one knob to the data, as D-063.2 proposed; folds by draw because the
realizations of a draw are near-copies (D-062.3). (2) The sandwich carries the design effect $1 + (n_{\rm real} - 1)\rho$
without estimating $\rho$ (D-062's check). (3) A grid fixed before the choice would be too coarse at $c$ = 0.5 or
needlessly fine at $c$ = 2. **[run, sandbox 2026-10-21, `tests/smoke/test_smoke_table.py`]** the cross-validated
scores on 80,153 rows in $\mathcal S$ (20,000 draws x 5, the default ranges, $\tau^2 = 9 \times 10^{-4}$) were
9.17, 9.07, 9.03, 9.04, 9.12 x $10^{-4}$ at $c$ = 0.5, 0.7071, 1, 1.4142, 2 ($c$ = 1 chosen); on check_b_target's toy
(the $\hat d$ and $\hat\mu\hat d$ errors correlated) $c$ = 0.7071 was chosen at 20,000 draws; the noise of $\hat b$ over
the sandwich SE had SD 0.87-1.10 over five fixture seeds, and over 60 tables the sandwich SE matched the across-table SD
(0.80-1.20 by point) where the independent-rows SE fell short (0.56-0.93).

**What it implies for code and tables.** Implemented in Blocks 14.3-14.6 (SPEC Block 14): `bandwidth_cv_grid`,
`choose_multiplier` (the cross-validation's sums are the whole set's less the fold's, checked against tables fitted on
each fold's complement to $10^{-9}$), `grid_axes`, `sandwich_true` / `sandwich_pairs`, `LABEL_FLAGS`. The CV over five
multipliers is the merge's main cost: 131.5 s on 80,153 rows in $\mathcal S$ in the sandbox (2 cores; the table at $c$
13.7 s), linear in those rows, so about 27 min per $10^6$ -- the merge runs in a job, not on the login node
(`checks/table_2026-10-21/check_cv_timing.py`).

**What it does not decide [open].** The CV grid's values (PROVISIONAL; a choice at an end is flagged so that the grid
can be widened); S2b's design (the plane range and side, the profile and block, the background, near-then-far or joint,
the axis depth, further terms, the isolation radius; SPEC section 8), put to the user next; D-027's $\varphi_{\rm tol}$.

**Sources.** **[user]** 2026-10-21, 12:30. **[KB]** D-026, D-054, D-062, D-063, D-067, D-070; SPEC Block 14 and
section 8; the plan of 2026-10-20. **[run, sandbox]** `tests/smoke/test_smoke_table.py`, `test_smoke_invert.py`
(2026-10-21; Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1). **[src]** `sci/diameter-pipeline`, Blocks 14.1-14.6.

**Status:** active; (D-071.1)-(D-071.4) implemented, Blocks 14.3-14.6 (local commits `89ca78a` 14.1, `e9682a5` 14.2, `779bec0` 14.3-14.6, not pushed); (D-071.5)
in progress (S2b's design put to the user, 2026-10-21). [updated 2026-10-21, later 4: the four commits pushed at the user's agreement with the SPEC Block 15 draft, `f63fe9f..825bef1`; a fresh clone has the tested tree **[run, sandbox]**; (D-071.5): the far scan's planes and line set and its fit explained, D-074]

---

## D-072 -- The stored SS windows are the noise fit's data as the archive wrote them: no shape test is added anywhere in the noise fit, the twins included

**Date:** 2026-10-21, 14:53 (Europe/Rome), the user's answer to the
assistant's two questions of 2026-10-20 before Block 3. **Binds:** Block 3
(`noise_windows`: which SS windows enter and where each sits in its sweep);
Block 4's binding function and $\hat\Sigma$ (no QC step in the simulator);
Block 5 (no QC on the twins' SS windows); `Ih Fit/specs/SPEC.md` sections 3
and 8.

**Decision [user].** "The QC is done during the saving of the whole cell. so
the data on DaVinci and the Drive are already cleared. ... No more shape
tests."

**Statement.** For each real cell $c$ of the noise fit, the SS data are the
windows stored in its `ss_pulses.npz`: the pulses that passed the archive
writer's trace QC (engine v2, alpha 0.5) when the cell was archived; the
cluster and Google Drive hold the same files. No test on the response shape --
the writer's trace QC or any other selection on a window's content -- is
applied in the noise fit: not to the data, not to the windows the binding
function and $\hat\Sigma$ simulate (D-033, D-048 to D-051, D-056), not to the
twins' SS windows. Each stored window's position in its sweep's 37-pulse train
is read from the archive's own per-pulse table `trace_qc_results.csv` (its
`passed` rows are the stored windows, in order).

**Why.** **[user]**: the selection was made once, when the archive was
written; the noise fit takes its result as the data. Recorded with it:
(1) **[cluster, output pasted 2026-10-21 14:53]** all 18 L3_exc cells carry the
table, with `qc_rows` = `n_pulses_raw` = `n_pulses_ampdur_qc` = 37 x (number of
SS sweeps): 74 in 14 cells (2 sweeps), 222 in 571430815 (6), 370 in 569844159,
571483807, 571624532 (10); engine v2 and alpha 0.5 in every cell; stored 37-69
of 74, 199 of 222, 303-348 of 370 -- 6 to 50 % dropped, median 27 %. Every gap
is therefore a trace-QC drop and every layout can be rebuilt. (2) **[run,
sandbox; changelog 2026-10-20 (later, 5)]** SQC: the QC's selection through
Gaussian noise of the law shifts the D-051 SS statistics by -0.17 to -0.19
per-cell SD at $t_{\rm SS}$ = 40 ms and by at most 0.09 SD from 80 ms (2-sweep
cells; about 2.2 times more in SD units for a 10-sweep cell). (3)
**[reasoning]** 14 of the 18 cells drop more than the about 13 % that Gaussian
noise of the law alone makes the QC drop, so most drops removed structure the
law lacks; the windows left are the ones the law is meant to describe.

**What it implies for code and tables.** Block 3 reads the layout from
`ss_pulses.npz`, `trace_qc_results.csv` and `metadata.json['ss_extraction']`
and refuses a cell whose table, counts or polarity sequence disagree (no
fallback is needed on L3_exc). No QC module is imported by `noise_windows`,
`noise_fit` or the generator. The binding function simulates each real SS
sweep as one continuous 37-pulse trace (D-056 (i)) and cuts its windows where
the real cell has them.

**What it does not decide [open].** The floor $t_{\rm SS}(c) \ge 80$ ms (the
assistant's proposal from SQC: it keeps the selection's effect at or below
0.09 per-cell SD, against up to 0.4 SD in a 10-sweep cell whose transient test
gives 40 ms); whether the per-cell excess drop rate (drop rate minus the
noise-only rate) is written as a label (a label, never a criterion); the
twins' SS layout (D-073).

**Sources.** **[user]** 2026-10-21 14:53. **[cluster]** the per-cell check of
2026-10-21. **[data, Drive copy]** the two cells read on 2026-10-20. **[src]**
the writer's `_archive_one_cell`; `trace_qc_v2.run_trace_qc_for_cell_v2` (the
kept pulses in their input order). **[run, sandbox]** SQC,
`sqc_trace_qc_selection.py` (sent to the user 2026-10-20).

**Status:** active; not implemented (Block 3). [updated 2026-10-23: the floor t_SS(c) >= 80 ms set by **D-076** (ii); the excess-drop label not asked for -- the user: do not worry about the QC.]

---

## D-073 -- Each twin sweep carries noise calibrated on its own real sweep, not a per-cell average

**Date:** 2026-10-21, 14:53 (Europe/Rome), the same message. **Binds:** D-017
(each twin carries its own real cell's noise law) and D-051 (ii) (one law per
cell; the twin's SS pulses and LS sweeps generated from it), as generator
inputs; D-016 (d) (open since 2026-09-23: whether noise differs between the
sweeps of one cell while the generator draws every sweep of a cell from one
law); Blocks 3-5 (statistics kept per sweep; a per-sweep stage of the fit; a
per-sweep generator and manifest); the twins' protocol
(`gen_from_manifest.build_proto`, `run_ih_recovery --ls-hyp-amps
--ls-dep-amps`, `synthetic_ground_truth.simulate_ss_protocol`).

**Decision [user].** "I would like that the synthetic data generated sweeps
have the noise calibrated to each real relative sweep so that we do not have
to stick to an average noise per cell."

**Statement.** For each real cell $c$ and each sweep $s$ of $c$ that the twin
reproduces, the twin's sweep $s'$ is the twin of $s$ (same protocol; for Long
Square the same amplitude and place in the sequence; for Square Subthreshold
the same 37-pulse train), and the noise injected into $s'$ is calibrated on
the data of $s$, not on a summary over $c$'s sweeps. Settles D-016 (d) for
per-sweep noise. Supersedes D-017's and D-051 (ii)'s per-cell law *as the
generator input*; which parameters are estimated per sweep and which per cell
is not decided (comments).

**Why.** **[user]**: real sweeps differ; a twin built on one average law per
cell cannot reproduce that.

**The assistant's comments, not yet answered.** (a) **What one sweep can
identify [run, sandbox: `sweep_identifiability.py`, 2026-10-21].** Efficient
method-of-moments bound $\mathrm{Cov}(\hat u) \approx (J^\top \Sigma^{-1}
J)^{-1}$ at $\theta_{\rm cf}$ for $u = (\log\sigma_f, \rho_f, \log\sigma_s,
\log\tau_s)$, $J$ from Block 2's closed forms, $\Sigma$ the Monte Carlo
covariance of D-051's statistics (3000 simulated LS pre-windows, 1500 simulated
SS sweeps per window count). SD of the estimate from ONE sweep (for a log
parameter about its relative SD), all four free: LS sweep $\sigma_f$ 0.030,
$\rho_f$ 0.012, $\sigma_s$ 0.30, $\tau_s$ 0.72 (a factor of about 2); SS sweep
of 37 windows 0.010, 0.014, 0.18, 0.49; of 20 windows 0.010, 0.013, 0.23,
0.62; a whole cell of 5 LS and 2 SS sweeps 0.006, 0.005, 0.09, 0.23. With
$\tau_s$ held at the cell's value, $\sigma_s$ per sweep: 0.13 (LS), 0.06-0.07
(SS). The fast pair is identified per sweep, the slow amplitude only roughly,
the slow timescale not at all. (b) **Proposal [reasoning].** Per sweep
$\sigma_f(c,s)$, $\rho_f(c,s)$ and $\sigma_s(c,s)$; per cell $\tau_s(c)$ and
the tilt's $\sigma_b(c)$ (D-034: a spread across sweeps, not a property of one
sweep). $\sigma_s(c,s)$ partially pooled towards $\sigma_s(c)$ by a
random-effects estimate of the between-sweep spread (DerSimonian-Laird,
statsmodels' `combine_effects`), so a sweep whose data cannot tell keeps the
cell's value and a cell whose sweeps do not differ collapses to its own law.
(c) **Estimation [reasoning].** The per-cell fit of Blocks 3-4 as designed
(D-048 to D-051) gives $\theta(c)$ and $\sigma_b(c)$; then, per sweep, the same
indirect inference on that sweep's statistics for $(\sigma_f, \rho_f,
\sigma_s)$ with $\tau_s = \tau_s(c)$. (d) **Protocol twinning [src;
reasoning].** Needed for any per-sweep calibration: the twin's LS sweeps are
the real cell's (the role table's $h_1 \dots h_n$, $d_1 \dots d_k$, same
amplitudes), not the cohort set `--ls-hyp-amps` used today
(`run_ih_recovery.phase_generate`); its SS sweeps are 37-pulse continuous
sweeps, one per real SS sweep, written at the real cell's stored positions
(D-072: no QC on the twins).

**What it does not decide [open].** Comments (a)-(d); raw or partially pooled
per-sweep values; a per-sweep tilt (D-034 stays per cell unless decided); how
the per-sweep stage enters the recovery report.

**Sources.** **[user]** 2026-10-21 14:53. **[log]** D-012, D-016 (d), D-017,
D-034, D-051. **[src]** `run_ih_recovery.py` (the cohort's LS amplitudes),
`synthetic_ground_truth.simulate_ss_protocol`. **[run, sandbox]**
`sweep_identifiability.py` (Python 3.13, NumPy 2.5.3, SciPy 1.18.1,
statsmodels 0.15.0), sent to the user 2026-10-21.

**Status:** active (intent); form open; not implemented. [updated 2026-10-23: form settled by **D-076** -- option (a) of comment (b), comment (d) as proposed; (a) and (c) stand as the design behind it.]

---

## D-074 -- S2b's far scan: both sides of focus to +-16 planes with an SNR rule, on a +-10 um line (user); the line-end baseline that went with it does not exist beyond plane 9, and a planar far-pixel background is proposed; the far fit explained, not yet decided

**Date:** 2026-10-21, 14:47 (Europe/Rome), the user's answers to the question card posted at 14:34 after Block 14
(the implementation chat). **Binds:** SPEC Block 15 (drafted at `825bef1`: its [to confirm] items Q-A to Q-D);
`CalibrationConfig`'s far fields (`far_offsets_planes`, `far_min_snr`, `far_profile_half_um`, the far background rule,
`far_fit`); the far scan and far fit of `analysis/calibration.py` (not built); `RendererConfig.kernel_continuation_slope`
($\gamma$, configured 0.79 until S2b, D-070.1); D-026 (b)-(d), (f), (g); D-027 (c).

**Decision [user].** "Push all four (Recommended)" (an action: `f63fe9f..825bef1` on `sci/diameter-pipeline`); for the
far scan's planes "Both sides, ±16, SNR ≥ 3 (Recommended)"; for its line and background "±10 um, linear baseline
(Recommended)"; for the far fit "Before implementing the far field calibration explain me extensively how you would
proceed". The card's stated defaults for the far set -- "no other traced segment within the line's half-length, tilt
<= 5 deg" -- were not contested **[assistant's reading: PROVISIONAL, not confirmed]**.

**Statement.** Notation of SPEC Block 15: node $j$ of the far set, plane $k$, $\delta_{j,k} = z_{{\rm ax},j} - z_k$,
$x_{j,k} = |\delta_{j,k}| - \delta_0$, $\delta_0$ = 0.84 um.

(D-074.1) **Planes [user].** The far scan reads the planes $k^*_j - 16 \dots k^*_j + 16$ (4.48 um on both sides of
focus); a plane enters the far fit only when its dip passes an SNR rule at 3. The assistant's reading, PROVISIONAL: the
rule is applied to the predicted dip, $A_j / (\sqrt{2\pi}\,\sigma_{\rm p}) \ge 3\,{\rm RMS}_j$ ($A_j$ the node's dip
area from its planes 4-6, $\sigma_{\rm p}$ the prior width at the pilot's depth, ${\rm RMS}_j$ the node's median residual
RMS), not to each plane's observed amplitude, which biases $\hat\gamma$ by -1.1 to -1.7 % at 2 gl of profile noise
(Why).

(D-074.2) **Line [user].** The far scan's measuring line has half-length 10 um, a calibration setting (`MeasureConfig`
and the estimator hash unchanged).

(D-074.3) **The background that went with it does not exist past plane 9 [the assistant's correction of its own
proposal; for the user].** A baseline $B_0 + B_1 v$ fitted to the line's samples with $|v - v_{\rm peak}| \ge$ 2 FWHM has
at least 4 samples per side only up to plane 9 at $\gamma$ = 0.79 and plane 7 at $\gamma$ = 1.21 (a 0.7 um node,
$\mu d$ = 0.35, the renderer's model): the proposal sized the line against $3\sigma_{\rm r}$ (D-026 (d)), not against
2 FWHM $\approx 4.7\,\sigma_{\rm r}$ **[run, sandbox]**. Proposed instead: per plane, a planar background
$b_0 + b_x x + b_y y$ fitted (trimmed) to the pixels of a +-25 um square farther than $r_q + 3\sigma_{\rm r}(\delta_{q,k})$
from every traced segment $q$ and the soma (Block 13's `far_from_dendrites`, with a per-segment margin), evaluated on the
line; the line-end mean of the dip kept, where the zone exists, as the check of that background.

(D-074.4) **The far fit [open; explained 2026-10-21].** The assistant's recommendation, with its evidence put to the
user: sequential (the anchor $\hat\sigma_0 = \hat\sigma_{\rm r}(0.84)$ from Block 10), both sides, the axis free within
+-0.28 um of the pilot, the planes with $|k - k^*_j| \ge 5$ ($x \ge 0.4$ um $\ge r_j$), a two-pass core fit for the far
planes, weights from the far law's predicted width, the SE over nodes by the node bootstrap with $t_{J-1}$ intervals, plus the anchor's term; the renderer's
table cut at 0.84 um so that its continuation starts where (D-026.1) is anchored.

**Why [user; run, sandbox; reasoning].** **[user]** (D-074.1)-(D-074.2). **[run, sandbox, 2026-10-21; the renderer's
model -- Gaussian kernel, partition along vertical rays, an infinite flat tube, white noise, no camera chain]**:
Block 10's masked block median at the far planes reads 0.03 gl low at plane 4 and 2.5-2.6 gl low from plane 12 on (dips
4-5 gl deep there), so $\omega$ falls to 0.19 of itself at plane 16; the planar far-pixel fit is off by at most 0.003 gl;
a common background error of +0.1 gl raises the noise-free $\hat\gamma$ by 2.6-6.9 % and +0.05 gl the weighted Monte
Carlo's by 1.4 %; the far form of Eq. 4 fits noise-free core widths to $8 \times 10^{-3}$ um$^2$ and returns $\gamma$
within 0.03 %; over 200 simulated far sets of $J$ = 12 nodes (1 gl white profile noise) the recommended fit is unbiased
to 0.2 % with an SD of 0.5 %, a one-sided scan has 4.3-4.8 times its SD, a free $\sigma_0$ 3.7-4.1 times, an axis fixed
at the pilot (0.07 um SD, assumed) a bias of -0.6 to -0.9 %, and Block 10's statistic with its own weights biases by
-3.5 % and -38 % ($\gamma$ = 0.79, 1.21) through 165 and 647 planes where the fit locked onto noise; $\partial\hat\gamma /
\partial\sigma_0$ = -0.27 and -0.32 per um; the node bootstrap's SE is 0.83 of the true SD at $J$ = 12 (the jackknife's 0.88; +-1.96 SE intervals covered 92 % of 60 replicates); a tilt biases $\hat\gamma$ upward, +0.5-0.9 % at 5 deg and +1.9-3.0 % at 10 deg; at 2 gl of profile noise per sample the SNR rule on the observed amplitude biases $\hat\gamma$ by -1.1 % and -1.7 % (the planes kept at the edge of the reach read 5-37 % low), the same rule on the predicted dip by +0.07 % and -0.09 % (SD 1.5 % and 1.4 %), and at 1 gl the two agree within 0.2 %; Block
10's fit keeps a knot at 1.12 um, from which the renderer's continuation would start ($\sigma_{\rm r}(4.48)$ 4.881
against 5.006 um at $\gamma$ = 1.21).

**What it implies for code and tables.** SPEC Block 15: Q-A settled by (D-074.1); Q-B's line settled by (D-074.2), its
background reopened by (D-074.3); Q-C and Q-D wait for the user; the draft is rewritten once they are answered. The checks
(`farmodel.py`, `check_far_statistic.py`, `check_far_fit_mc.py`, `check_info_budget.py`, `check_junction.py`,
`check_worked_case.py`, `check_far_jackknife.py`, `check_far_tilt.py`, `check_far_selection.py`, `fig_far_line.py`,
`fig_far_fit.py`, with their logs) were run in the sandbox's scratchpad and sent to the user; they enter the repository under
`checks/s2b_2026-10-21/` with Block 15.

**What it does not decide [open].** The background rule (D-074.3); the far statistic; the fit, its weights and its SE
(D-074.4); isolation and tilt (Q-D: the per-plane neighbour rule proposed against the card's per-node rule; 5 deg and
straightness over +-10 um); the junction (cut at 0.84 um or anchor at the last knot).

**Sources.** **[user]** 2026-10-21, 14:47 (the card of 14:34). **[KB]** D-024 (i), D-025 (a)-(b), D-026 (a)-(h), D-027
(c)-(d), D-067.2, D-067.4, D-070, D-071; procedure §3.4; mathematics §3.3-3.5 and Eq. 17; SPEC Blocks 10, 13, 15 (draft).
**[src]** `analysis/calibration.py` (`dip_statistics`, `fit_growth`, `kernel_from_growth`), `model/kernel.py`
(`sigma_r`), `analysis/plane_iteration.py` (`far_from_dendrites`, `plane_background`), `scripts/calibrate_kernel.py`
(`run_fit`) at `825bef1`. **[run, sandbox]** the checks above (Python 3.13, NumPy 2.5.3, SciPy 1.18.1). **[PubMed]** 13
searches (defocus PSF width and geometric optics, brightfield defocus and neurite diameter, depth from defocus, background
plane fits, defocused PSF microscopy, linear PSF growth, widefield PSF z-stacks, Gaussian PSF approximation, symmetric
defocus, estimated-weight bias in weighted least squares, cluster bootstrap, two-step estimator SEs, linear defocus blur),
10 with no record; Maloberti et al. 2025, *Biomed Opt Express* 16(4):1310-26 (PMID 40322002, PMC12047720), full text read,
cited only as a non-numeric analogy (auxiliary planes on both sides of focus improve a joint PSF estimate in SMLM),
[DOI](https://doi.org/10.1364/BOE.551278). **[bioRxiv]** biophysics, last 30 days: nothing relevant.

**Status:** active; (D-074.1)-(D-074.2) settled [user]; (D-074.3)-(D-074.4) and Q-D await the user. [updated 2026-10-23: step 4's two-pass core fit made iterative by **D-077** (user); the rest of the far statistic as proposed.] [updated 2026-10-23, later: (D-074.3) and (D-074.4) settled by **D-078** (user) -- the planar far-pixel background, the iterative core fit with N = 10 and kappa = 1, the sequential fit as recommended; (D-074.1)'s predicted-dip SNR rule confirmed; Q-D (isolation, tilt) and the junction still await the user.] [updated 2026-10-23, later 2: the isolation half of Q-D settled by **D-079** (user): no traced point but the node's own branch within the +-25 um background patch, for near and far calibration nodes; the tilt half and the junction still await the user.] [updated 2026-10-23, later 3: the junction settled by **D-080** (user): the renderer's table cut at 0.84 um; of this entry's open points only the tilt half of Q-D remains.] [updated 2026-10-23, later 4: Q-D's tilt half settled by **D-081.3** (user): 5 deg for both sets, exposed; nothing of this entry remains open.] [implemented (D-074.1: `far_offsets_planes` = 16, `far_profile_half_um` = 10, the SNR rule in `far_field.select_far_planes`; D-074.2: `node_allen_diameter_max_um` and `far_field_allen_diameter_um` unchanged) 2026-10-23, 20:30: commits `24eb687` (Blocks 10 and 15), `cd80081` (the figures), `f55c083` (Cell 7, runbook v22) on `sci/diameter-pipeline`, not yet pushed [updated 2026-10-26: pushed to `origin/sci/diameter-pipeline` at the user's request, `cc60f3c..dd901de`]; the 28 smoke suites pass; the estimator and simulation hashes unchanged (`0508e7ff12f48728`, `0ea5a6d2ec069c68`); Phase II on Allen's data not run.]

---

## D-075 -- The kernel calibration draws its evidence: the width points with the fitted near curve and far line, and every off-focus plane's fitted Gaussian core

**Date:** 2026-10-22, 17:36 (Europe/Rome), the user's message in the diameter implementation chat. **Binds:**
Block 10's near-field calibration (`analysis/calibration.py`: `plane_scan`, `fit_growth`, `kernel_from_growth`;
`scripts/calibrate_kernel.py fit`; `loading/calibration_io.py`, whose scans file keeps per-plane statistics only);
Block 15's far fit (S2b, D-074; not built); Colab Cell 7.

**Decision [user].** "Take note that when such near and far field width linear approximation is calculated to plot
the estimated width points and the linear estimations as well as all the fitted gaussians models across the off
focus planes".

**Statement.** Whenever the near-field growth fit (Eq. 4) and the far-field law (D-026.1) are computed, on phantoms
or on Allen's stacks, the run also draws:

(D-075.1) **The width points and the fitted laws [user].** Every calibration node's per-plane width estimates
against defocus, with the fitted near curve and the fitted far line. The assistant's reading, PROVISIONAL: one
figure per run, each plane's width with its node's constant removed so that all nodes fall on one curve -- near
planes $(\hat\omega_{i,k} - \hat c_i + \sigma_{\rm r}(0)^2)^{1/2}$, far planes $(\hat\omega_{j,k} - \hat c_j)^{1/2}$
-- against $\lvert\delta\rvert$ from the fitted axis depth, the two sides and the near and far sets distinguished,
the planes the selection dropped shown and marked; over them the first-stage table
$\hat\sigma_{\rm r}(m\,\Delta z)$ and the far line $\hat\sigma_0 + \hat\gamma\,(\lvert\delta\rvert - \delta_0)$, the
anchor marked; beside it, per node, the raw $\hat\omega$ with that node's fitted curve.

(D-075.2) **Every fitted Gaussian [user].** For every calibration node, each off-focus plane's dip with the Gaussian
core fitted to it. The assistant's reading, PROVISIONAL: one figure per node, one panel per plane in plane order,
each with the dip $w(v_n)$, the fitted core drawn over its window (both passes' windows on the far planes),
$\hat\omega$, and the plane's status (kept, dropped by the SNR rule, at a bound, failed).

**Why [user; run, sandbox; reasoning].** **[user]**. **[reasoning]** The far slope rests on a few low-signal planes
per node, and the two failures the S2b checks found are visible at a glance in these figures and invisible in the
numbers: core fits locked onto noise (165 and 647 collapsed planes with Block 10's statistic at $\gamma$ = 0.79 and
1.21) and a background that pulls widths low (Block 10's masked median 2.5 gl low from plane 12)
**[run, sandbox, 2026-10-21]**.

**What it implies for code and tables.** The scans keep what the figures need: today `PlaneScan` and the scans file
hold only per-plane statistics (`calibration_io._SCAN_ARRAYS`), so each plane's profile $w(v_n)$, its window(s) and
the fitted $(\hat a, \hat v_c, \hat s)$ are added, or the figures are drawn during the scan. Plotting goes in a module
of its own, apart from the analysis (scientific-coding). `calibrate_kernel.py fit` and Block 15's far fit write the
figures beside the calibration file; Colab Cell 7 shows them. SPEC: Block 10 amended, and Block 15 rewritten with
it once D-074's open points are settled. No signature changes (the calibration settings are in none).

**What it does not decide [open].** The format, layout and names of the figures; whether every node is drawn or a
sample when there are many; the y-variable (the kernel width with the node constants removed, as proposed, or the
raw width); whether the DL route of I-003 must produce the same figures (proposed: yes, as its diagnostics).

**Sources.** **[user]** 2026-10-22, 17:36. **[KB]** D-026.1, D-074, I-003. **[src]** `analysis/calibration.py`,
`loading/calibration_io.py`, `scripts/calibrate_kernel.py` at `37b42ea`. **[run, sandbox]** the S2b checks
(2026-10-21).

**Status:** active; not implemented (Block 10's figures; Block 15 not built). [updated 2026-10-23: D-075.2's "both passes' windows" become the windows of D-077's iterative core fit; the panel draws the last pass's window, PROVISIONAL.] [updated 2026-10-23, later: extended by **D-080.2** (user): all nodes' widths by plane in light grey with their median, and one example node with the images of its planes, near and far.] [implemented (D-075.1: `kernel_law.png`; D-075.2: `gaussians/<set>_node_<id>.png`, the last pass's window) 2026-10-23, 20:30: commits `24eb687` (Blocks 10 and 15), `cd80081` (the figures), `f55c083` (Cell 7, runbook v22) on `sci/diameter-pipeline`, not yet pushed [updated 2026-10-26: pushed to `origin/sci/diameter-pipeline` at the user's request, `cc60f3c..dd901de`]; the 28 smoke suites pass; the estimator and simulation hashes unchanged (`0508e7ff12f48728`, `0ea5a6d2ec069c68`); Phase II on Allen's data not run.]

---

## D-076 -- The per-sweep form is option (a); the SS noise window starts at or after 80 ms; the twins' Long Square amplitudes are the real cell's own; no QC anywhere in the noise fit

**Date:** 2026-10-23, 12:08 (Europe/Rome), the user's answers to the three
questions of 2026-10-21 (restated with their context on 2026-10-21, 16:43)
and two remarks. **Binds:** D-073 (its open form and comments (a)-(d));
D-072's open floor; D-051 (i) (the window start) and (iv) (the subtraction,
confirmed); Blocks 3-5 of `Ih Fit/specs/SPEC.md` (`noise_windows`,
`noise_cell_stats`, `noise_fit`, the generator and `run_ih_recovery`'s LS
amplitudes); `WindowConfig.t_ss_floor_ms`.

**Decision [user].** "1. Per-sweep form: Keep option one. 2. Floor t_SS(c)
>= 80 ms. 80 ms is ok. however we have discussed that we would have taken the
mean windows (not the windows mean) in order to highlight the trend and
remove that per sample mean from the single windows. 3. each cell's own LS
amplitudes. ... in general do not worry about the QC. you have the data
already cleaned. and there is no need to replicate the QC on synthetic data."

**Statement.**
(i) **Per-sweep form (option (a) of D-073 comment (b)).** For each real cell
$c$ and each of its sweeps $s$: $\sigma_f(c,s)$ and $\rho_f(c,s)$ are
estimated on sweep $s$ alone; $\sigma_s(c,s)$ is estimated on sweep $s$ with
$\tau_s$ held at the cell's $\hat\tau_s(c)$ and then shrunk towards the cell
by the random-effects weight,
$$\log\tilde\sigma_s(c,s) = \hat\mu(c) + \frac{\hat\tau^2(c)}{\hat\tau^2(c) + v(c,s)}\bigl[\log\hat\sigma_s(c,s) - \hat\mu(c)\bigr], \tag{D-076.1}$$
with $\hat\mu(c)$ the random-effects mean of $\log\hat\sigma_s(c,\cdot)$ over
$c$'s sweeps, $\hat\tau^2(c) \ge 0$ the DerSimonian-Laird moment estimate of
the between-sweep variance of $\log\sigma_s$ (clipped at zero) and $v(c,s)$
the sampling variance of $\log\hat\sigma_s(c,s)$ from the per-sweep fit;
$\tau_s(c)$ and the tilt's $\sigma_b(c)$ stay per cell (D-034). The twin of
sweep $s$ draws its noise from $\tilde\theta(c,s) = (\hat\sigma_f(c,s),
\hat\rho_f(c,s), \tilde\sigma_s(c,s), \hat\tau_s(c))$ with
$b \sim \mathcal N(0, \hat\sigma_b(c)^2)$.
(ii) **Window start.** $t_{\rm SS}(c) = \max(80\ {\rm ms},\ t_{\rm tr}(c))$,
$t_{\rm tr}(c)$ the transient test's time (D-051 (iv), $\kappa_{\rm tr}$ = 0.1,
5 ms bins, provisional); the fallback `ss_enlarged_none` of D-051 (i) stands
when $t_{\rm SS}(c) \ge 190$ ms. **Confirmed [user]**: the quantity
subtracted from every SS window is the *mean window* of its polarity -- the
per-sample mean over that polarity's windows, a vector over time, (D-051.1)
-- which removes the pulse response (the trend); each window's own scalar
mean is then removed by the statistics themselves ($\hat\sigma$ and
$\hat\rho$ demean). For the per-sweep statistics the same cell-level mean
window is subtracted (the assistant's choice: more windows in the mean, the
same vector for every sweep, so the per-sweep statistics stay comparable);
the simulator subtracts the mean of its own simulated windows with the
cell's counts (D-051 (iv)).
(iii) **Twins' Long Square set.** The twin of cell $c$ has the LS sweeps of
$c$ -- the role table's $h_1 \dots h_n$ and $d_1 \dots d_k$, each at its own
amplitude, one twin sweep per real sweep -- in place of the cohort set
`--ls-hyp-amps` / `--ls-dep-amps` of `run_ih_recovery.py`; settles D-073
comment (d).
(iv) **No QC anywhere** (restates D-072): the archive's windows are the data
as written; no shape test or other selection on a window's content is run on
the data, in the fit's simulator or on the twins.

**Why.** **[user]**: (i) what one sweep can identify (D-073 comment (a),
`sweep_identifiability.py`); (ii) the selection's leak is at most 0.09
per-cell SD from 80 ms (SQC, changelog 2026-10-20 (later, 5)); (iii)
per-sweep calibration needs a real counterpart for every twin sweep; (iv) the
QC was done once, when the archive was written.

**What it implies for code and tables.** Block 3 keeps every statistic per
window, per sweep and per cell; `WindowConfig.t_ss_floor_ms = 80.0`; the
subtraction is the cell-level per-polarity mean window on data and on
simulations. Block 4 gains a per-sweep stage after the per-cell fit
((D-076.1); `statsmodels.stats.meta_analysis.combine_effects(method_re="dl")`,
whose $\hat\tau^2$ is clipped at zero, as its moment estimate can be
negative **[run, source read]** [clarified 2026-10-26: statsmodels 0.15 itself does not clip -- `combine_effects(...).tau2` is the raw
moment estimate, negative when the spread is below the sampling variances, and its `weights_re` and `mean_effect_re` then use
the negative value; the clip at zero and the random-effects mean with the clipped value are Block 4b's to compute
(`check04_pooling_options.py` of the tilt study document, 2026-10-26)]). Block 5 gives the generator the per-sweep
laws and the real cell's LS amplitude list; `run_ih_recovery.py`'s cohort
amplitudes become the fallback for a twin without a manifest list, labelled.
Columns per sweep: `noise_sweep_sigma_f_mV`, `noise_sweep_rho_f`,
`noise_sweep_sigma_s_raw_mV`, `noise_sweep_sigma_s_mV` (shrunk),
`noise_sweep_v_log_sigma_s`, `noise_sweep_w`; per cell `noise_cell_tau2_log_sigma_s`.

**What it does not decide [open].** Raw against shrunk values in the
report (both are written); a per-sweep tilt (none: D-034 per cell); how the
per-sweep stage enters the recovery report; whether depolarising LS sweeps'
pre-windows enter the per-cell statistics (Block 3 takes every LS sweep's
pre-window, hyperpolarising and depolarising, since D-073 needs both twins
-- the assistant's choice, flagged in the SPEC).

**Sources.** **[user]** 2026-10-23 12:08. **[log]** D-034, D-051, D-072,
D-073. **[run, sandbox]** `sweep_identifiability.py` (2026-10-21),
`shrink_check.py` (2026-10-23: shrinkage lowered the per-sweep RMS error of
$\log\sigma_s$ from 0.130 to 0.043-0.117 (LS) and from 0.056 to 0.038-0.056
(SS) over true between-sweep SDs of 0 to 0.20, 20000 cells of 6 LS + 2 SS
sweeps), `sqc_trace_qc_selection.py` (2026-10-20). **[PubMed, full text]**
DerSimonian and Laird 2015, [DOI](https://doi.org/10.1016/j.cct.2015.09.002).

**Status:** active; settles D-073's form and comments (b), (d) and D-072's
floor; Block 3 in progress (2026-10-23). **[updated 2026-10-28: (ii)'s $t_{\rm tr}(c)$ and the cell-level mean window for the per-sweep statistics superseded by **D-095** (mean windows per sweep and polarity; the start from their exponential decay); whether the 80 ms floor stays is an open point of D-095.]** **[updated 2026-10-28 (later, 9): (ii)'s 80 ms floor kept, and 80 ms is also the fallback, flagged -- **D-103**.]**

---

## D-077 -- S2b's far statistic iterates its core fit: each pass sizes its window on the previous pass's width, up to N passes, until the FWHM stops changing within its own uncertainty

**Date:** 2026-10-23, 13:03 (Europe/Rome), the user's message in the diameter implementation chat. **Binds:** the far
statistic of S2b (D-074; step 4 of the far procedure explained with it, "a two-pass core fit"; SPEC Block 15, not
built); `CalibrationConfig`'s far fields (Block 15); the far scan's plane statuses; D-075.2's per-plane figures. Not
Block 10's near statistic (`calibration.dip_statistics`).

**Decision [user].** "The 'Statistic. A two-pass core fit:' fit in my opinion should be noe just to pass but we should
make this iterative up to a maximum of N=10 (an example) iterates that stops when the FWHM reaches its asymptotic
value. This can be set by the RMS? or however within a standard deviation of its uncertainty".

**Statement.** For far node $j$ at plane $k$, with the dip $w_{j,k}(v) = \hat B_k({\rm line}(v)) - I_k(v)$ (gl) and the
prior width $\sigma_{\rm p} = \sigma_{\rm r}(\lvert z_{{\rm p},j} - z_k\rvert)$ of the renderer's current law: pass 1 fits
the core $a\,\exp(-(v - v_{\rm c})^2 / 2\varsigma^2)$ within $\pm{\rm FWHM}(\sigma_{\rm p})$ of the dip's minimum,
${\rm FWHM}(\varsigma) = 2\sqrt{2\ln 2}\,\varsigma$; pass $i+1$ fits it within $\pm{\rm FWHM}(\hat\varsigma^{(i)})$ of
$\hat v_{\rm c}^{(i)}$; the width stays bounded to $[0.25, 4]\,\sigma_{\rm p}$ and a fit at a bound ends the plane, as
proposed with D-074. ($\varsigma$ is the core SD written $\hat s$ in D-075, renamed so as not to clash with F2's axis
shifts $s_j$.)

(D-077.1) **The cap [user].** At most $N$ passes; $N$ = 10 is the user's example.

(D-077.2) **The stop [user; the form is the assistant's reading, PROVISIONAL].** The first pass $i \ge 2$ with
$\lvert{\rm FWHM}(\hat\varsigma^{(i)}) - {\rm FWHM}(\hat\varsigma^{(i-1)})\rvert \le \kappa\,{\rm SE}\bigl({\rm FWHM}(\hat\varsigma^{(i)})\bigr)$,
$\kappa$ = 1 (the user's "within a standard deviation of its uncertainty"); then $\hat\omega_{j,k} = (\hat\varsigma^{(i)})^2$.
Here ${\rm SE}({\rm FWHM}) = 2\sqrt{2\ln 2}\,{\rm SE}(\hat\varsigma)$, with ${\rm SE}(\hat\varsigma)$ the core fit's own
Gauss-Newton SE: the residual RMS times $[(J^\top J)^{-1}]_{\varsigma\varsigma}^{1/2}$, $J$ the fit's Jacobian. The
user's two readings, "set by the RMS" and "within a standard deviation of its uncertainty", are therefore one rule.

(D-077.3) **No stop by pass $N$ [assistant, PROVISIONAL].** The plane gets the status `not_converged`, is left out of
the far fit like a fit at a bound, and is counted.

**Why [user; run, sandbox; reasoning].** **[user]** The width is to be the fixed point of the window rule, not wherever
two passes leave it. **[run, sandbox, 2026-10-23]** `check_iterative_core.py` (the renderer's model: Gaussian kernel
law, infinite flat tube; $d$ = 0.7 um, $\mu d$ = 0.2-0.5, $\gamma$ = 0.79 and 1.21, axis on plane 0, background exact;
the loop checked against the S2b checks' `stat_s2`, two passes identical): noise-free, the iteration reaches its
asymptote in every case (planes 5-16), and the two-pass $\hat\omega$ is already within $10^{-9}$ of it (relative); with
1 gl of white noise per sample (200 draws in each of 18 cases, planes 8, 12 and 16), the one-SE rule stops at pass 2
in 3343 and at pass 3 in 257 of 3600 fits, never later, and the mean $\hat\omega$ moves by at most 0.1 % of the
asymptote from the two-pass value; at 2 gl, 1 fit of 3596 ran to pass 10 without stopping, and the means move by at
most 0.2 %. **[reasoning]** In the renderer's model the change is therefore a safeguard that costs about one pass; it
acts where a real dip's core is not the Gaussian the window assumes (another kernel shape, a neighbour inside the
window), where the window's size moves the fitted width -- not simulated here.

**What it implies for code and tables.** Block 15's far statistic: the two passes of the S2b draft become a loop with
two `CalibrationConfig` fields, the cap (default 10) and $\kappa$ (default 1.0), their names to be fixed in SPEC Block
15; the plane status gains `not_converged`; the scans file keeps each plane's number of passes, with D-075's profile
and fit; D-075.2's panel draws the last pass's window (PROVISIONAL). The S2b check scripts of 2026-10-21 keep their
two-pass S2: by the check above, their results carry over to the iterative statistic in the renderer's model. No
signature changes (the calibration settings are in none, D-075).

**What it does not decide [open].** $N$'s final value (10 is the user's example); $\kappa$ (1, as the user said; a
smaller $\kappa$ stops later and nearer the asymptote); which SE enters the stop (the plane's own fit, as proposed, or
F2's model SE); whether a plane at the cap is dropped (as proposed) or kept with a flag; a two-cycle between two
windows (none seen); D-074's other open points (D-074.3, D-074.4, Q-D, the fit and its weights).

**Sources.** **[user]** 2026-10-23, 13:03. **[KB]** D-074, D-075. **[src]** `check_far_fit_mc.stat_s2` and `_gauss_fit`
(the S2b check scripts, sandbox, 2026-10-21). **[run, sandbox]** `check_iterative_core.py` and its log, 2026-10-23,
sent to the user.

**Status:** active; not implemented (Block 15 not built). [updated 2026-10-23, later: (D-077.1)'s $N$ = 10 and (D-077.2)'s $\kappa$ = 1 confirmed by **D-078** (user); D-077.3 and the SE used in the stop stay PROVISIONAL.] [updated 2026-10-23, later 2: (D-077.3) settled by **D-081.4** (user): the stop uses the plane's own fit SE; a plane not stopped by pass N is dropped and counted; N exposed.] [implemented (`far_field.core_fit_iterative`: on a sampled Gaussian it returns $\hat\omega = s^2$ and stops at pass 2; with the stop disabled and $N$ = 2 it equals the two-pass statistic) 2026-10-23, 20:30: commits `24eb687` (Blocks 10 and 15), `cd80081` (the figures), `f55c083` (Cell 7, runbook v22) on `sci/diameter-pipeline`, not yet pushed [updated 2026-10-26: pushed to `origin/sci/diameter-pipeline` at the user's request, `cc60f3c..dd901de`]; the 28 smoke suites pass; the estimator and simulation hashes unchanged (`0508e7ff12f48728`, `0ea5a6d2ec069c68`); Phase II on Allen's data not run.]

---

## D-078 -- S2b's far calibration settled (user): the planar far-pixel background; the iterative core fit, N = 10 and kappa = 1; the sequential far fit with the anchor, model weights by IRLS, the predicted-dip SNR rule and a node-bootstrap SE plus the anchor term

**Date:** 2026-10-23, 17:13 (Europe/Rome), the user's answers to questions 1-3 of "Decisions for you" in the S2b answer
of 2026-10-21 (the diameter implementation chat). **Binds:** SPEC Block 15 (Q-B's background; Q-C, the statistic and
the fit); `CalibrationConfig`'s far fields (the far background rule, the far statistic's cap and tolerance, the far
fit's settings, its SE); the far scan and far fit of `analysis/calibration.py` (not built); D-074.1's SNR reading,
D-074.3, D-074.4; D-077.1-D-077.2. Not the junction, not isolation and tilt (questions 5 and 4, open).

**Decision [user].** "About the questions: 1. Background (my correction). (a) as you raccomend 2. Statistic at the far
planes. (a) The two-pass core fit, with Block 10's near scan unchanged. Perfect remember the N iterations instead of 2
as discussed beore. I would say N = 10 and k = 1 [3.] Fit. (a) Sequential, both sides, axis free within ±0.28 µm,
$\lvert k - k^*\rvert \ge 5$, model weights, the predicted-dip SNR rule, and an SE from the node bootstrap with $t_{J-1}$
intervals plus the anchor term (recommended). Perfect [...] Take note". The user's "k" is the tolerance $\kappa$ of
D-077.2 (the assistant's reading).

**Statement.** Notation of D-074 and D-077: far node $j$, plane $k$, $x_{j,k} = \lvert\delta_{j,k}\rvert - \delta_0$,
$\delta_0$ = 0.84 um.

(D-078.1) **Background [user; D-074.3's proposal].** In each plane $k$, $\hat B_k(x, y) = b_0 + b_x x + b_y y$ is fitted
by least squares to the pixels of a +-25 um square farther than $r_q + 3\sigma_{\rm r}(\delta_{q,k})$ from every traced
segment $q$ and from the soma ($\sigma_{\rm r}$ of the renderer's current law; Block 13's `far_from_dendrites` with a
per-segment margin; trimming against untraced debris an implementation detail, PROVISIONAL), and read on the line:
$w_{j,k}(v) = \hat B_k({\rm line}(v)) - I_k(v)$. The line-end mean of $w_{j,k}$ ($\lvert v - v_{\rm peak}\rvert \ge$
2 FWHM) is kept as the check of $\hat B_k$ in the planes where that zone exists.

(D-078.2) **Statistic [user].** At the far planes, the core fit of D-077 with the cap $N$ = 10 passes and the
tolerance $\kappa$ = 1 (both now the user's, no longer examples); Block 10's near statistic (`dip_statistics`)
unchanged. The stop stays on the FWHM as D-077.2 writes it (stopping on $\hat\omega$ gives the same pass in 99.5 % of
3600 simulated fits at 1 gl **[run, sandbox, 2026-10-23]**). D-077.3 (no stop by pass $N$: `not_converged`, left out)
stays PROVISIONAL.

(D-078.3) **Fit [user].** Sequential: $\sigma_0$ held at Block 10's anchor $\hat\sigma_0 = \hat\sigma_{\rm r}(0.84\
\mu{\rm m})$; both sides of focus; each node's axis shift $s_j$ free within +-0.28 um of the pilot; the planes with
$\lvert k - k^*_j\rvert \ge 5$; a plane enters when its predicted dip passes the SNR rule,
$A_j / (\sqrt{2\pi}\,\sigma_{\rm p}) \ge 3\,{\rm RMS}_j$ (D-074.1's PROVISIONAL reading, now the user's); model weights
$1/{\rm SE}_{{\rm m},j,k}^2$ applied by IRLS as coded -- an unweighted fit, then three refits with the weights frozen
at the previous round's estimates. **[assistant, 2026-10-23]** F2 written with ${\rm SE}_{\rm m}(\gamma, c, s)$ inside
one minimisation is a different estimator: minimised literally it lands +0.6 % and +0.8 % above the true $\gamma$
(0.79, 1.21; the coded IRLS -0.07 %) and, with $\gamma$ unbounded, its objective falls below its value at the solution
once $\gamma \gtrsim 20$ **[run, sandbox, `check_irls_vs_literal.py`]**; SPEC Block 15 states the iteration, not F2. The
SE: ${\rm SE}(\hat\gamma)^2 = {\rm SE}_{\rm boot}(\hat\gamma)^2 + (\partial\hat\gamma/\partial\sigma_0)^2\,
{\rm SE}(\hat\sigma_0)^2$, with ${\rm SE}_{\rm boot}$ from the node bootstrap (whole far nodes resampled with all their
planes) and intervals $\hat\gamma \pm t_{J-1,\,0.975}\,{\rm SE}(\hat\gamma)$; $\partial\hat\gamma/\partial\sigma_0$ by
refitting with the anchor shifted, PROVISIONAL; ${\rm SE}(\hat\sigma_0)$ from Block 10's own resampling of the near
nodes. The two terms add without a cross term because the near set ($D_{\rm A} \le$ 0.6 um) and the far set
(0.6-0.8 um, D-067.4) are different nodes **[reasoning]**.

**Why [user; run, sandbox; reasoning].** **[user]**. The evidence put to the user with D-074 (2026-10-21) and in the
explanations of 2026-10-23 **[run, sandbox; the renderer's model, $J$ = 12, 1 gl white profile noise]**: Block 10's
masked block median reads 2.5-2.6 gl low from plane 12, the planar far-pixel fit at most 0.003 gl off; Block 10's
statistic collapses on 165 and 647 kept planes ($\gamma$ = 0.79, 1.21), the two-pass fit on none, and the iterative fit
with $N$ = 10 and $\kappa$ = 1 stops by pass 3 in all 3600 test fits; the recommended fit is unbiased to 0.2 % with an
SD of 0.5 %, the anchor narrowing it 3.7-4.1 times ($\partial\hat\gamma/\partial\sigma_0$ = -0.27 and -0.32 per um);
the predicted-dip SNR rule removes the -1.1 and -1.7 % bias of the observed-amplitude rule at 2 gl. **[run]** The node
bootstrap's SE is 0.83 of the true SD at $J$ = 12, hence the $t_{J-1}$ intervals.

**What it implies for code and tables.** SPEC Block 15 is rewritten with D-074.1-D-074.2, D-077 and this entry once
questions 4 and 5 are answered (or drafted with them marked open). `CalibrationConfig` gains the far background's
square (25 um) and margin factor (3), the core fit's cap (10) and tolerance (1.0), the far fit's settings (axis bound
0.28 um, $\lvert k - k^*\rvert \ge 5$, SNR 3 on the predicted dip, 3 IRLS rounds) and the bootstrap's resample count;
the plane statuses gain `not_converged`. The checks of 2026-10-21 and 2026-10-23 enter the repository under
`checks/s2b_2026-10-21/` with Block 15. No signature changes (the calibration settings are in none, D-075); $\hat\gamma$
itself goes into `RendererConfig.kernel_continuation_slope`, in the simulation hash, when S2b runs.

**What it does not decide [open].** Isolation and tilt (question 4, D-074's Q-D: the per-plane neighbour rule against
the card's per-node rule; $\varphi \le 5°$ and straightness over +-10 um); the junction (question 5: the table cut at
0.84 um, recommended, or the far law anchored at the last knot); D-077's open points other than $N$ and $\kappa$ (which
SE enters the stop, a plane at the cap, two-cycles); the bootstrap's resample count; how $\partial\hat\gamma/\partial
\sigma_0$ and ${\rm SE}(\hat\sigma_0)$ are computed; the coverage of the bootstrap-SE $t_{J-1}$ intervals (only the
jackknife's was checked: 93 % at nominal 95 %, 60 replicates).

**Sources.** **[user]** 2026-10-23, 17:13. **[KB]** D-067.4, D-074, D-075, D-077. **[run, sandbox]** the S2b checks of
2026-10-21 (D-074's list); of 2026-10-23: `check_irls_vs_literal.py`, `check_iterative_core.py`,
`check_stop_variable.py`, `check_bg_offset_s2.py`, `check_pedestal_constant.py`, `fig_anchor_cloud.py`,
`fig_junction.py`, with their logs, sent to the user.

**Status:** active; settles D-074.3, D-074.4 (not its junction item) and D-074.1's SNR reading, and fixes D-077.1's
$N$ and D-077.2's $\kappa$; not implemented (Block 15 not built; SPEC Block 15 not yet rewritten). [updated 2026-10-23, later: D-078.1's patch must hold no traced point but node $j$'s own branch, and the near-field calibration uses the same background (**D-079**, readings PROVISIONAL).] [implemented (D-078.1: `calibration_scene.planar_background`, the line-end mean kept as `line_end_gl`; D-078.2: `far_field.core_fit_iterative`; D-078.3: `far_field.fit_far_slope` with IRLS, `bootstrap_gamma`, `anchor_sensitivity`, the $t_{J-1}$ interval in `far_fit_summary`; on rendered far phantoms $\hat\gamma$ within 1 % noise-free and 0.83 $\pm$ 0.017 against 0.79 with the camera chain, 4 nodes) 2026-10-23, 20:30: commits `24eb687` (Blocks 10 and 15), `cd80081` (the figures), `f55c083` (Cell 7, runbook v22) on `sci/diameter-pipeline`, not yet pushed [updated 2026-10-26: pushed to `origin/sci/diameter-pipeline` at the user's request, `cc60f3c..dd901de`]; the 28 smoke suites pass; the estimator and simulation hashes unchanged (`0508e7ff12f48728`, `0ea5a6d2ec069c68`); Phase II on Allen's data not run.] [updated 2026-10-28, D-101: (D-078.1)'s far mask is the default no more -- the background is fitted on the stripe of Block 10.S; the mask stays under `background_support = "square"`]

---

## D-079 -- The calibration nodes, near and far, are isolated within their background patch: no traced point in the +-25 um square but the node's own branch; the near-field calibration uses the planar far-pixel background too

**Date:** 2026-10-23, 17:39 (Europe/Rome), the user's message in the diameter implementation chat. **Binds:** Block
10's node selection (`calibration_reasons`, both the near set and the far set of D-067.4 / S2.1); Block 10's near scan
(`scan_node`, `plane_scan`: its background, today the masked block median $\bar B_k$ over the +-5 um block) under the
assistant's reading of (D-079.2); D-078.1's background patch; D-074's Q-D, its isolation half (question 4 of the S2b
answer); `CalibrationConfig`'s isolation and background fields; Colab Cell 7's counts of the calibration sets (D-070).

**Decision [user].** "1) 3e. The background must change. Perfect as you propose take note that in the patch of plane
extracted and to which the plane B_k(x,y) is fitted must have no neighbouring nodes in any direction except for the
queried j node. This regards the selection of the nodes used to calibrate both the near and far field. Use this
approach also for the near field calibration. Take note".

**Statement.** Notation of D-078: node $j$, plane $k$, $\hat B_k(x, y) = b_0 + b_x x + b_y y$ fitted to the far pixels
of the +-25 um square centred on node $j$.

(D-079.1) **Isolation [user; the reading of "in any direction" is the assistant's, PROVISIONAL].** A node $j$ enters a
calibration set, near or far, only if the +-25 um square centred on it -- the patch to which $\hat B_k$ is fitted --
contains no traced point other than those of node $j$'s own branch: no other segment of the cell, no branch point on
its own branch inside the square, and no soma. The assistant's reading: "in any direction" means all around the node in
the image plane, and a segment counts at any depth once its lateral projection enters the square, since its shadow
reaches the patch at some plane, the more widely the farther that plane is. The alternative reading counts only the
segments within the scanned depth range. The test is made on the trace (SWC), once per node.

(D-079.2) **The near field too [user; the assistant's reading, PROVISIONAL].** Block 10's near-field calibration
measures its dips against the same planar background as D-078.1, fitted to the far pixels of the isolated patch,
instead of its masked block median $\bar B_k$. The alternative reading extends only the isolation of (D-079.1) to the
near set.

**Why [user; run, sandbox; reasoning].** **[user]**. **[run, sandbox, 2026-10-21; the renderer's model, noise-free]**
The planar far-pixel fit is off by at most 0.003 gl, with an illumination trend and one quadrant of the square masked;
Block 10's masked block median reads 0.03 gl low at plane 4 and 2.5-2.6 gl low from plane 12 on. **[reasoning]** (i)
With no other segment in the square, the far pixels are the whole square less the node's own branch zone, so the planar
fit sees no neighbour's blur tails, which at the far planes reach $r_q + 3\sigma_{\rm r} \approx$ 10 um from a segment
(3 x 3.5 um at plane 16). (ii) One background definition in both stages: the anchor $\hat\sigma_0$ and the far widths
are then measured against the same background, so a background bias does not enter the near and far stages
differently, and the junction of 3f joins two quantities defined alike. (iii) The cost is fewer nodes: the rule is
stricter than the card's per-node 10 um rule and than the per-plane rule proposed on 2026-10-21; the far stage's SE
grows as $1/\sqrt J$.

**What it implies for code and tables.** Block 10's selection gains an isolation reason (a `calibration_reasons` entry,
its name to be fixed in SPEC Block 10; no isolation test exists there today **[src, `analysis/calibration.py` at
`cc60f3c`]**). Under (D-079.2), Block 10's near scan fetches the +-25 um patch and fits D-078.1's planar background
instead of the +-5 um block median, and its scans, the anchor and the kernel table are recomputed; the table is in the
simulation signature, so the simulation hash changes when Block 10 is re-run (the calibration settings themselves are
in no signature, D-075). SPEC Block 10 is amended and Block 15 written with it; Block 10's smoke tests follow. Colab Cell
7's counts of the near and far sets are re-run with the rule before Block 15's SE is judged.

**What it does not decide [open].** The depth range in which a segment counts (any depth, as read, or the scanned range
only); untraced structures, which a trace-based test cannot see (left to the planar fit's trimming, PROVISIONAL); the
tilt half of question 4 ($\varphi \le 5°$, straightness over +-10 um); the junction (question 5).

**Sources.** **[user]** 2026-10-23, 17:39. **[KB]** D-067.4, D-070, D-074 (Q-D), D-075, D-078. **[src]**
`analysis/calibration.py` (`calibration_reasons`, `scan_node`) at `cc60f3c`. **[run, sandbox]** the S2b check A3
(2026-10-21).

**Status:** active; not implemented (Block 10's selection and near background are implemented code to amend; Block 15
not built); settles the isolation half of D-074's Q-D; the readings of "in any direction" and of (D-079.2) await the
user's confirmation. [updated 2026-10-23, later: the user confirms that neighbours in depth count too (17:47, recorded with **D-080**); which depth range counts as nearby stays open.] [updated 2026-10-23, later 2: both readings settled by **D-081** (user): (D-079.1) counts a segment whose predicted shadow reaches the patch above 0.03 gl in a scanned plane, the node's own dendrite excluded (D-081.1); (D-079.2) the planar background in the near scan, near fitted before far (D-081.2).] [implemented (`calibration_scene.far_pixel_mask`, `planar_background`, `isolation_depth`; `calibration.trace_reasons`) 2026-10-23, 20:30: commits `24eb687` (Blocks 10 and 15), `cd80081` (the figures), `f55c083` (Cell 7, runbook v22) on `sci/diameter-pipeline`, not yet pushed [updated 2026-10-26: pushed to `origin/sci/diameter-pipeline` at the user's request, `cc60f3c..dd901de`]; the 28 smoke suites pass; the estimator and simulation hashes unchanged (`0508e7ff12f48728`, `0ea5a6d2ec069c68`); Phase II on Allen's data not run.]

---

## D-080 -- The junction: the renderer's table cut at the anchor, 0.84 um (user); the calibration's width-by-plane figures: every selected node's squared core widths across the planes in light grey with their median, and one example node with the images of its planes, near and far field (user)

**Date:** 2026-10-23, 18:09 (Europe/Rome), the user's answer to question 5 of the S2b answer of 2026-10-21 and the
figures asked with it (the diameter implementation chat). **Binds:** `calibration.kernel_from_growth`'s
`max_delta_um` wherever a table is written for rendering (`scripts/calibrate_kernel.py fit`, Block 10; Block 15's
output); `RendererConfig.kernel_table_delta_um` / `kernel_table_sigma_um` (in the simulation signature); D-074's
junction item; D-075's figures; the calibration's plotting module (not built); the scans file
(`loading/calibration_io.py`); Colab Cell 7.

**Decision [user].** "Junction. (a) Cut Block 10's table at 0.84 µm, so the renderer's continuation starts at the anchor
(recommended). Perfect. However as already said You will plot the estimated squared core width values across the
shallower and deeper planes from the focal plane (+-3) or the anchor for near field and far field calibration
respectively along with the image of the relative planes. So that I can judge if a non linear fit is needed. Moreover
you will do this for all the selected nodes and an example across them. the plot with all the nodes should have all
the nodes' values of the width at each plane in light grey and the mean or median in a color of your choice."

**Statement.**

(D-080.1) **Junction [user].** Every kernel table written for rendering ends at the anchor:
`kernel_from_growth(..., max_delta_um = 0.84)`, so the renderer's continuation is
$\sigma_{\rm r}(\delta) = \hat\sigma_0 + \gamma\,(\lvert\delta\rvert - \delta_0)$ for $\lvert\delta\rvert > \delta_0$ =
0.84 um, with $\gamma$ the configured 0.79 until S2b runs and $\hat\gamma$ after. Block 10's growth fit keeps its knots
up to $M h$ = 1.12 um for the fit itself; only the rendered table is cut. The assistant's reading, PROVISIONAL: the cut
applies from now on, to the table Block 10 writes today as well as to S2b's.

(D-080.2) **Width-by-plane figures [user; layout readings PROVISIONAL].** For each calibration stage -- the near field
on the planes $k^*_j - 3, \dots, k^*_j + 3$, shallower and deeper than the focal plane; the far field from the anchor's
planes outward on both sides -- the run draws: (i) all selected nodes, each node's $\hat\omega_{j,k}$ against the plane
in light grey, with their median per plane in one colour (the assistant's choice: the median, robust to a node gone
wrong; the colour is blue); (ii) one example node, its $\hat\omega_{j,k}$ against the plane, with the image of each of
its planes. The purpose [user]: to judge whether a non-linear law is needed. The assistant's readings: the x-axis is
the signed plane offset $k - k^*_j$, so that the median is taken over nodes at equal offsets; the fitted law (Block
10's curve, the far line) is drawn over the median as in D-075.1; a plane's image is its +-25 um background patch with
the line drawn on it. These figures come with D-075.1 (the width points and the fitted laws) and D-075.2 (every fitted
Gaussian), drawn by the plotting module from the scans.

**Why [user; run, sandbox; reasoning].** **[user]**. **[run, sandbox, 2026-10-21 and 2026-10-23]** Without the cut, a
measured $\hat\gamma$ = 1.21 would be continued from the table's last knot at 1.12 um, giving a far law parallel to the
measured one but 0.125 um lower: 9.8 % at 1.40 um, 2.5 % at 4.48 um (`check_junction.py`, `fig_junction.py`, the
pipeline's own `kernel_from_growth` and `sigma_r`). **[reasoning]** The median over nodes against the plane shows the
law's shape, with the nodes' scatter as its spread: a curvature beyond the linear far law, or a misfit near focus, is
seen directly. The plane images show what each width was read from: the isolation, the background, any neighbour.

**What it implies for code and tables.** `kernel_from_growth` is called with `max_delta_um = 0.84` in
`calibrate_kernel.py fit` and in Block 15's output; the kernel table, hence the simulation hash, changes when Block 10
is next run. The scans keep what these figures need -- each plane's $\hat\omega$ and status for every node, and the
example node's plane images (all nodes' images would be large: 33 planes of a +-25 um patch per far node) or the means
to fetch them again -- the format to be fixed in SPEC Blocks 10 and 15. The figures live in a plotting module of their
own (scientific-coding layout) and are written beside the calibration file; Colab Cell 7 shows them.

**What it does not decide [open].** The choice of the example node; the y-variable (the raw $\hat\omega$, or with each
node's constant removed so that all nodes fall on one curve); a residual panel; the crop and contrast of the plane
images; whether the quadratic comparison term of Block 15's draft is drawn.

**Note on D-079.1 [user, 17:47; message cut short].** "In any direction: I mean across the image plane and the depth
also because nearby dendrites at shallower or deeper depths could potentially". The depth is included, as read; which
depth range counts as "nearby" stays open.

**Sources.** **[user]** 2026-10-23, 18:09 and 17:47. **[KB]** D-074, D-075, D-077, D-078, D-079. **[src]**
`analysis/calibration.py` (`kernel_from_growth`), `model/kernel.py` (`sigma_r`) at `cc60f3c`. **[run, sandbox]**
`check_junction.py` (2026-10-21), `fig_junction.py` (2026-10-23).

**Status:** active; not implemented. Settles D-074's junction; with D-078 and D-079, of the S2b questions only the
tilt half of question 4 remains open. [updated 2026-10-23, later: (D-080.2)'s layout settled by **D-081.5** (user): signed offset, raw and centred panels, residuals, the example node nearest the median.] [implemented (D-080.1: `kernel_from_growth(..., max_delta_um=table_max_delta_um)` in `fit` and in the far fit's production kernel; D-080.2: `plotting/calibration_figures.py` with `analysis/calibration_report.py`) 2026-10-23, 20:30: commits `24eb687` (Blocks 10 and 15), `cd80081` (the figures), `f55c083` (Cell 7, runbook v22) on `sci/diameter-pipeline`, not yet pushed [updated 2026-10-26: pushed to `origin/sci/diameter-pipeline` at the user's request, `cc60f3c..dd901de`]; the 28 smoke suites pass; the estimator and simulation hashes unchanged (`0508e7ff12f48728`, `0ea5a6d2ec069c68`); Phase II on Allen's data not run.]

---

## D-081 -- The implementation choices of Blocks 10 and 15 settled (user): isolation by the predicted shadow, the node's own dendrite excluded; the planar background in the near scan, near fitted before far; tilt 5 deg for both sets and the passes of the core fit as exposed parameters; the figures as proposed; gamma_2 for the quadratic term; the defaults confirmed

**Date:** 2026-10-23, 18:30 (Europe/Rome), the user's answers to the seven questions of 18:20 (the diameter
implementation chat), with the order to implement Blocks 10 and 15 and update the roadmap. **Binds:** D-079.1 (the
depth rule), D-079.2, D-077.3, D-080.2, D-078.3's SE terms; SPEC Blocks 10 and 15; `CalibrationConfig`
(`node_phi_max_deg`, the isolation, straightness, core-fit and bootstrap fields); Block 10's selection, near scan and
output; Block 15; the calibration's plotting module; Colab Cell 7.

**Decision [user].** "1) The b choice. Be aware that same dendrite nodes will fall within the patch thus these should
be not regarded. 2) The option a). about '. Then the anchor and the far widths are measured against one background
definition.' yes but in successive moments first near field then the far field 3) Both far and near sets must have the
tilt lower than 5°. however expose the parameter 4) otpion a) However make the number of passes an exposed parameter
5) Option a) as you reccomend 6) ok as you say 7) 7. Defaults to confirm or change. perfect all. Now implement all we
discussed and finally update the roadtable".

**Statement.**

(D-081.1) **Isolation in depth (settles D-079.1's open range) [user].** A traced segment $q$ of the cell counts as a
neighbour of calibration node $j$ when its shadow, predicted with the renderer's current law, reaches node $j$'s
$\pm$25 um background patch in some scanned plane $k$ of the stage with a depth above $\varepsilon_{\rm iso}$ = 0.03 gl:
in each scanned plane, the dip of a flat tube of radius $r_q$ and the pilot's $\hat\mu$ at defocus $\delta_{q,k}$
(the node's $\hat\mu$, PROVISIONAL), of width $\sigma_{\rm r}(\delta_{q,k})$, evaluated at the segment's lateral
distance from the patch's nearest edge; a positive value anywhere in the patch fails the node (reason `not_isolated`).
The points of node $j$'s own dendrite inside the patch -- its own branch (the stretch, Block 8) -- are not neighbours;
a branch point of its own branch inside the patch, and the soma, are. The threshold is a `CalibrationConfig` field.

(D-081.2) **Near background, in order (settles D-079.2) [user].** Block 10's near scan measures its dips against the
planar far-pixel background of D-078.1 (the $\pm$25 um patch, the far pixels farther than $r_q + 3\sigma_{\rm r}
(\delta_{q,k})$ from every traced segment and the soma, a least-squares plane), replacing the masked block median
$\bar B_k$. The near field is fitted first, giving $\hat\sigma_0$ and the table; the far field is scanned and fitted
after, against the same background definition (sequential, D-078.3).

(D-081.3) **Tilt (settles the tilt half of D-074's Q-D) [user].** Both sets require $\varphi_j \le$
`node_phi_max_deg`, default 5 deg (was 10). Straightness, the assistant's proposal of 18:20 accepted with (3),
PROVISIONAL: the traced centreline within $\pm$10 um of the node along the branch stays within $r_j$ of the node's
straight line (reason `not_straight`; the tolerance `node_straight_tol_radii` = 1, exposed).

(D-081.4) **The core fit's passes (settles D-077.3) [user].** The stop uses the plane's own fit SE (D-077.2); a plane
that has not stopped by pass $N$ is dropped from the far fit and counted (`not_converged`); $N$ is the exposed
`CalibrationConfig` field `far_core_max_passes`, default 10, with `far_core_stop_tol` ($\kappa$), default 1.0.

(D-081.5) **Figures (settles D-080.2's layout) [user].** The x-axis is the signed plane offset $k - k^*_j$ with the
median over nodes at each offset; two panels, the raw $\hat\omega_{j,k}$ and the per-node-centred
$\hat\omega_{j,k} - \hat c_j$; the fitted law drawn over the median with a residual panel below; the example node is
the one nearest the median; its plane images are the $\pm$25 um patch with the line, the core-fit window and the
far-pixel mask overlaid.

(D-081.6) **Symbols [user].** Block 15's quadratic comparison coefficient is $\gamma_2$
($\sigma_{\rm r} = \sigma_0 + \gamma x + \gamma_2 x^2$); $\kappa$ is D-077's stopping tolerance only.

(D-081.7) **Defaults [user].** The node bootstrap uses 1000 resamples in both stages (`bootstrap_resamples`);
$\partial\hat\gamma/\partial\sigma_0$ by a central difference of $\pm$0.01 um on the anchor
(`anchor_sensitivity_step_um`); ${\rm SE}(\hat\sigma_0)$ from the node bootstrap Block 10 gains; the 0.84 um table cut
(D-080.1) applies to today's configuration with $\gamma$ = 0.79 as well.

**Why [user; reasoning].** **[user]**. **[reasoning]** (D-081.1) ties isolation to what 3e needs -- a background right
to a few hundredths of a grey level -- instead of a fixed radius, and excludes the node's own dendrite, which is in
every patch by construction. (D-081.2) gives the anchor and the far widths one background definition, so a background
bias enters both stages alike; the order keeps the far stage conditional on the near one (D-078.3). (D-081.3) at 5 deg
the tilt bias on $\hat\gamma$ is +0.5-0.9 % (D-074's Why), at 10 deg +1.9-3.0 %.

**What it implies for code and tables.** SPEC Block 10 amended and Block 15 rewritten (this turn); `CalibrationConfig`
gains `patch_half_um` (25), `far_pixel_sigmas` (3), `isolation_depth_gl` (0.03), `node_straight_tol_radii` (1),
`far_offsets_planes` (16), `far_min_k_offset` (5), `far_min_snr` (3), `far_core_max_passes` (10), `far_core_stop_tol`
(1.0), `far_core_width_bounds` ((0.25, 4)), `far_irls_rounds` (3), `bootstrap_resamples` (1000),
`anchor_sensitivity_step_um` (0.01), `table_max_delta_um` (0.84); `node_phi_max_deg` default 5; the scans keep each
plane's profile, background, window and fit (D-075); `kernel_from_growth` is called with `max_delta_um` =
`table_max_delta_um`; the figures in `plotting/calibration_figures.py`. The estimator hash is unchanged (no calibration
field is in a signature); the simulation hash changes when a table is next written.

**What it does not decide [open].** The pilot's $\hat\mu$ as the neighbour's absorption (PROVISIONAL); the straightness
tolerance's value; the example node's choice when several tie; everything SPEC Block 15 marks PROVISIONAL after this
turn.

**Sources.** **[user]** 2026-10-23, 18:30. **[KB]** D-074 to D-080.

**Status:** active; implementation started the same turn (Blocks 10 and 15 on `sci/diameter-pipeline`). [implemented 2026-10-23, 20:30: commits `24eb687` (Blocks 10 and 15), `cd80081` (the figures), `f55c083` (Cell 7, runbook v22) on `sci/diameter-pipeline`, not yet pushed [updated 2026-10-26: pushed to `origin/sci/diameter-pipeline` at the user's request, `cc60f3c..dd901de`]; the 28 smoke suites pass; the estimator and simulation hashes unchanged (`0508e7ff12f48728`, `0ea5a6d2ec069c68`); Phase II on Allen's data not run.] [refinement 2026-10-23, 20:30, PROVISIONAL: (D-081.3)'s straightness is implemented against the trace's own total-least-squares line through its points within $\pm$10 um of the node (`calibration_scene.straightness_excess`), not the pilot's line, because the pilot's heading error of about 1 deg moves a 10 um line by 0.13 um and flagged straight 0.2 um phantoms (`not_straight` at $r_j$ = 0.1 um); the tolerance stays one $r_j$ (`node_straight_tol_radii` = 1). Two further readings, PROVISIONAL: the isolation test's predicted shadow uses the renderer's current law (the near table continued with today's slope) at the neighbour's defocus, with the soma as a disc of its own radius; a plane whose core fit ends at a width bound is dropped like an unconverged one (`at_bound`). To confirm or change.] [updated 2026-10-26: the roadmap's theory stage T6 (far-field calibration) closed by the user on the Subproject Roadmaps page (08:59), applied in the chat (`c27e3a9`); no question was open under it after D-074 and D-077 to D-081.]
[updated 2026-10-27, later 3: (D-081.3)'s straightness part superseded by **D-093** (user, 19:08): the calibration nodes are not screened for straightness by default (`node_straight_check` False), the tolerance 2 radii when the test is switched on (was 1); flatness ($\varphi \le$ 5 deg) and the isolation test unchanged. On Cell 7's first run on the pilot 529878215 none of the 68 candidates was a calibration node; the 57 lines the user pasted all failed straightness at one radius, 46 of them isolation too.] [updated 2026-10-28, D-101: (D-081.3)'s 5 deg becomes 10 deg near / 15 deg far; (D-081.1)'s node verdict is the stripe's, the any-shadow rule kept under `background_support = "square"`]

---

## D-082 -- Every step still to do is on the roadmap, general for now, and each step has a theory part of its own (user)

**Date:** 2026-10-26, 09:11 (Europe/Rome), the diameter implementation chat. **Binds:** the roadmaps of the project
(`specs/roadmap/<name>.yaml`): each implementation step names its theory part in `needs_theory`; the diameter-pipeline
roadmap's steps added on 2026-10-26.

**Decision [user].** "add to the roadmap all the other steps to do (these steps can be also very general for now, for
example there could be a step for the inversion therefore put it in the roadmap in the implementation) each step has
its own theory part remember this".

**Statement.**

(D-082.1) Every step still to do enters the roadmap's implementation graph as a step, general for now; its blocks are
added when they are written (`roadmap.py add block`), and it is drawn planned until the first one arrives.

(D-082.2) Each step has a theory part of its own: the theory group paired with it (`roadmap.py add step` creates TS<k>
and writes `needs_theory: [TS<k>]` on the step), or, where the theory already exists as a topic, that topic.

**Why [user; reasoning].** **[user]** as quoted. **[reasoning]** The pairing shows on the page the theory to study
before each step, and a block placed under a step inherits that theory in its `blocked_by` (the tool at `c410c95`,
pushed 2026-10-26 09:19 by another session).

**What it implies.** The diameter-pipeline roadmap at `d6d410d` (pushed; page version 41), DRAFT until the user
confirms it, from the plan of 2026-10-20 (S3-S8), D-065, D-067 and D-013: S3 Phase II on Allen's data (S3a real-cell
survey, S3b simulator inputs, S3c kernel calibration on Allen's data, S3d final configuration), S5 gate 2, S6 davinci
pilot, S7 production table and its acceptance tests, S8 inversion of the real cells, S9 corrected SWCs into the passive
refit; not scheduled: NS1 kernel calibration step 2, NS2 hybrid absorption renderer. Theory parts: TS3 (TS3a-TS3d),
TS5-TS9, TNS1, and T7 for NS2. A first draft of the same steps as blocks with manual statuses (I16-I27, T8-T18) was
replaced by this form before any push, once the tool had changed.

**What it does not decide [open].** Whether G1 (foundations), which has no theory part, gets one; whether G4 and G5,
which share TG3 since `c410c95`, get theory groups of their own; the order and the content of each theory part.

**Sources.** **[user]** 2026-10-26, 09:11. **[KB]** the implementation plan of 2026-10-20, D-065, D-067, D-013; the
roadmap tool at `c410c95`.

**Status:** active; applied to the diameter-pipeline roadmap (`d6d410d`). [updated 2026-10-26, 16:48: the questions on G1 and on TG3 shared by G4 and G5, and any other change to the roadmap, wait until the user says so -- other changes to it are being developed (the user)] [updated 2026-10-27, 16:28: the hold is lifted for blocks -- every new SPEC block enters its roadmap in the same turn, without being asked, **D-091** (the user); G1's and TG3's questions stay open] [updated 2026-10-28 (later, 3): the allen-mtg-fit roadmap updated at the user's word ("Update also the roadmap", 10:24) -- D-095's notes and revise marks on N3 and N4 (`53b2713`, `cd32dad`; D-097's row)] [updated 2026-10-28 (later, 10): the allen-mtg-fit roadmap updated at the user's word ("Finally update the roadmap with the changes discussed here", 19:53) -- notes and revise marks for D-102 to D-104 on N3, N4, N5 and T8, no status changed (`a945a7a`; **D-104**)] [updated 2026-10-29: the allen-mtg-fit roadmap updated again at the user's word ("Update the roadmap", 15:01) -- notes and revise marks for D-106 and D-107 on N2, N3, N4, N5 and T8, no status changed (`bee2abe`)]


---

## D-083 -- The six-parameter synthetic recovery runs on davinci before any fit of the six parameters on a real cell (user)

**Date:** 2026-10-26, 09:31 (Europe/Rome), the noise-fit chat. **Binds:** the order of work: Stage 7's recovery-gate
run (roadmap allen-mtg-fit S7; SPEC Block 8, to specify) before Stage 8's campaign on real cells (S8), its one-cell dry
run included; what the twins of that run carry (D-011, D-017, D-073, D-076); `run_ih_recovery.py` and
`submit_ih_recovery.sh`, run on davinci; the roadmap's S7, S8 and I4.

**Decision [user].** "Before doing any fit on real cells i would like to do in da vinci a recover of the 6 parameters
(passive and iH) from synthetic twins (with the relative fitted noise and morphologies) where the 6 parameters has been
chosen and the same stimulation protocol applied. It's the same synthetic check on the optimization and fitting
algorithm that I've done with the passive parameters only. Add this to the implementation roadmap please".

**Statement.**

(D-083.1) **Order.** No real cell is fitted with the six-parameter arm -- $\theta = (C_m, R_m, R_a, \bar g_h, \Delta v_h,
\kappa_\tau)$, D-005 -- before the synthetic recovery of Stage 7 has run on davinci and its verdict is in. The verdict is
D-011's gate on the `ih6` arm; a kinetic knob that fails it is frozen at its base value for the campaign and a failing
$C_m$ or $\bar g_h$ is not (D-005, D-011).

(D-083.2) **The twins.** Each twin is built on one real L3_exc cell's archive morphology (D-011 (i)) and carries that
cell's fitted recording noise -- each twin sweep calibrated on its own real sweep (D-073, D-076 (i)) -- and that cell's
stimulation protocol: its SS sweeps as 37-pulse trains at the stored positions and its own Long Square amplitudes
(D-076 (iii)). The six parameters of every twin are chosen by the Stage-7 manifest, a quarter of the cohort with no
I_h as the false-positive control (D-011 (ii)), and every twin is fitted by the same fitter and optimiser as a real
cell (`run_ih_recovery.py` through `run_ih_fit.main()`, D-011).

(D-083.3) **Precedent.** It is the six-parameter version of the passive-only synthetic check of the optimiser and the
fitting algorithm, which the user reports having run (the synthetic passive benchmark, roadmap I4).

**Why [user; log].** **[user]** The same check of the optimisation and the fitting algorithm that the passive fit passed,
now with I_h. **[log]** D-005's status: until the gate has run, "a fitted $\Delta v_h$ or $\kappa_\tau$ is a number the
optimiser returned, not a measurement".

**What it implies.** The roadmap allen-mtg-fit, commit `8c671a2` (the noise chat's clone; delivered as a patch on
`main` `64fb8c5`; the Subproject Roadmaps page at version 25): S7 restated in these terms, its theory part T14 (D-082);
S8's note: no six-parameter fit of a real cell before S7, the dry run included; I4's manual status `done` on the user's
statement (it was `tested`, DRAFT: no run was recorded); N4b, Block 4b (the per-sweep stage of D-076 (i)), added as a
step with its theory part T9 under D-082, and N5 and N6 made to depend on it. **[reasoning, flagged]** "Any fit on real
cells" is read as any fit of the six parameters: the noise-law fit on the real recordings (Block 6, roadmap N6) still
runs before S7, because it supplies the twins' noise, and run B's passive fits already exist (roadmap I2).

**What it does not decide [open].** The gate's tolerances (the assistant's proposal, unowned, D-011); the cohort size
(20 by default, no power calculation) and whether L2_exc joins it; raw against corrected SWCs for the twins (D-013:
every test runs on the raw SWCs until the corrected ones arrive; re-running Stage 7 on the corrected ones is
recommended); the SPEC entry of the gate run (Block 8); where the results of the passive-only benchmark are recorded.
[corrected 2026-10-26, same turn: "which arms besides `ih6` are run on the twins" was listed here as open; the staged
plan already fixes them -- four fits per twin, `baseline_runB`, `passive_fullstep`, `ih4` and `ih6`, the gate on
`ih6` (plan s.8, Stage 7) **[KB]**.]

**Sources.** **[user]** 2026-10-26, 09:31. **[log]** D-005, D-011, D-013, D-073, D-076, D-082. **[src]** the roadmap
at `8c671a2`.

**Status:** active; the roadmap updated (patch pending the user's push).
---

## D-084 -- The hybrid absorption renderer is promoted to be the renderer actually used (user)

**Date:** 2026-10-26, 16:48 (Europe/Rome), the diameter implementation chat. **Binds:** Block 4's absorption
(`RendererConfig.absorption`, today `partition_vertical`, D-024 (i) "for now"); idea I-004; the diameter-pipeline
roadmap's NS2 (not scheduled) and T7 (open); the simulation hash and every table built on it -- gate 2 (S5), the
davinci pilot (S6), the production table (S7); the thick-node check of the real-cell survey (S3a).

**Decision [user].** "Take note to promote the Hybrid absorption rendered as the one actually used".

**Statement.**

(D-084.1) The hybrid renderer of I-004 -- each slice's absorbed light taken from the condenser's rays (NA 1.4,
$n_{\rm oil}$ 1.515, averaged over directions), then every image plane rendered by applying the defocus kernel to the
slices -- is the renderer actually used, as Block 4's absorption mode, once it is built.

(D-084.2) Its optics come first (roadmap T7, open, I-004); until the hybrid is built and tested, Block 4 keeps the
partition (`partition_vertical`, D-024 (i)) and the code built today runs with it.

**Why [user].** **[user]** as quoted.

**What it implies [reasoning].** The absorption mode is inside the simulation hash, so a table rendered with the
partition (gate 2, the davinci pilot, a production) is not the table of the hybrid and would be rendered again; the
thick-node check of S3a (real against rendered phantoms of the same $\hat d$) and any band of phantoms' fitted
$\hat\mu\hat d$ are to be redone with the hybrid; the roadmap's NS2 leaves "not scheduled" when the user allows roadmap
changes (on hold since 16:48, D-082's annotation). Whether the kernel calibrated in S3c carries over unchanged depends
on what the hybrid's per-slice absorption does with defocus: if the rays only change how much each slice absorbs, the
calibrated $\sigma_{\rm r}(\delta)$ stays the defocus blur; if they also spread a slice's shadow with defocus, the kernel
is refitted under the hybrid from the same scans -- a question for T7.

**What it does not decide [open].** When the hybrid is built relative to S5 and S6; whether gate 2 and the pilot run
first with the partition; the kernel question above; the hybrid's own parameters.

**Sources.** **[user]** 2026-10-26, 16:48. **[KB]** I-004, D-024 (i), D-067.2, the roadmap's T7 and NS2.

**Status:** active; not implemented (T7 open). [updated 2026-10-27: built as SPEC Block 21 (`2572e15`), Block 4's absorption mode `hybrid`, and smoke-tested; the default left at the partition, so D-084.2 still holds until the user switches it -- the hybrid's planes go below zero in dark stain and it costs 4 to 50 times the partition per phantom (**D-092**, D-092.3); on the kernel question above: in the faint limit the hybrid books $1.447\,a_j$ uniformly, the partition's faint booking at $1.447\mu$, so a kernel calibrated through the partition's model on faint nodes carries over with $\hat\mu$ rescaled **[reasoning]**; in dark stain still open]

---

## D-085 -- S3, the real-data programme in Colab, settled (user): five cells per class and layer; automatic alignment; the plane spacing and pixel size checked first over all the cells; node selection exposed; every figure plus the diameters by layer and class; no thick-node check; jitter pooled; one camera setting; one calibration kernel per cell, checked for enough clean dendrites; the final configuration waits for the hybrid

**Date:** 2026-10-26, 17:09 (Europe/Rome), the user's answers to the eleven questions of 16:48 (the diameter
implementation chat). **Binds:** S3 of the plan of 2026-10-20 and the blocks that build it (16-20, to specify, SPEC
first); the roadmap's S3a-S3d (changes on hold, D-082); D-065 (which cells "all the cells" means); D-067.1; Cell 7's
calibration, now per cell; D-084 (the final configuration waits for the hybrid).

**Decision [user].** "1) do the statistics on colab. 5 cells for each neuron class (exc / inh ) and layer If there
aren't enough cells take all. 2) Automatic 3) First of all control for all the cells in colab if they share the same
plane spacing and pixel size 4) Both options, expose the parameter so I can choose one or the other 5) All the figures,
also the distribution of diameters per layer and type. 6) We do not have the correct renderer we stilll have to discuss
the hybrid absorption and implement the defocusing kernel. we need to do these. Drop this part 7) What's d_max for? Be
brief 8) yes 9) I think all cells have been captured with teh same camera thus use the same settings across cells 10)
one cell one calibration kernel. We will drop this ONLY if there aren't enough neat and clear dendrite to construct the
near and far field calibrations. Add this check in the pipeline 11) Wait for the hybrid. Perfect for the rest."

**Statement.**

(D-085.1) **Cells.** The S3 statistics run in Colab on 5 cells per neuron class (excitatory, inhibitory) and cortical
layer, and on all the cells of a class-and-layer group that has fewer than 5 [user]. Readings to confirm,
PROVISIONAL: the pool is the human cells with a reconstruction in Allen's database (as Phase 0 lists them); the class
is read from Allen's dendrite type (spiny excitatory, aspiny inhibitory; sparsely spiny to be placed by the user); the
5 of a larger group are drawn at random with a recorded seed.

(D-085.2) **Alignment.** Each cell's global alignment (SWC to image: shift and flip, no rotation) is computed
automatically -- Block 0's `align_translation` on the cell's minimum-intensity projection at downsample 4, as cell 11
did for 529878215 -- with an overlay figure and the score per cell; a cell below a score threshold is set aside for the
user to look at [user; the method as proposed].

(D-085.3) **Plane spacing and pixel size first.** Before anything else, a Colab check over all the cells of the pool
reads each cell's plane spacing (`NeuronReconstruction.scale_factor_z`) and pixel size and reports whether they are
shared; the configuration's are 0.28 um and 0.1144 um/px, from 529878215. What follows for cells that differ is
decided on that report [user].

(D-085.4) **S3a's nodes.** Both modes, chosen by a parameter: every dendrite node of the cell, or the stratified
sample (up to $N$ nodes per bin of Allen's $2r$, spread over the cell's stretches; $N$ and the bins exposed,
proposal $N$ = 20 and the bins 0-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.2, 1.2-2, > 2 um) [user].

(D-085.5) **S3a's figures.** All of them: $\hat\alpha_i$ and $\hat\mu_i\hat d_i$ per bin of $\hat d_i$ against the
phantoms' band (the analytic edges $\ln\mu_{\min} + \ln d$, $\ln\mu_{\max} + \ln d$ of D-065, and the phantoms' own
fitted values once rendered draws exist), the share of real nodes inside with the pass rule fixed before the run
($\ge$ 95 % inside in every bin with $\ge$ 20 nodes), and the distributions of the diameters per layer and per class
[user]; a failing bin is reported, and widening the $\mu$ range stays a later decision (D-065.1).

(D-085.6) **No thick-node check now.** The comparison of thick real nodes' SGT shape with rendered phantoms is dropped:
the renderer is not the right one yet -- the hybrid absorption is still to be discussed and the defocus kernel still to
be implemented [user] (D-084).

(D-085.7) **$d_{\max}$.** The user asked what it is for; answered in the chat (17:09): the upper end of the phantoms'
true-diameter range, which bounds the correction (a node beyond the table's reach is `out_of_domain` and filled),
sizes the line and the block, and spreads the draws. Its rule (the 99.5th percentile of Allen's $2r$ over the cells,
proposed) is open.

(D-085.8) **Jitter.** `jitter_xy_um` and `jitter_z_um` are the robust SDs ($1.4826 \times$ MAD) of the lateral offsets
$s^*$ and the depth offsets $\Delta z^*$ over the registration survey's ON nodes, pooled across the cells, with the
per-cell values reported; the nodes of a cell from `list_stretches`' suggestion [user].

(D-085.9) **Camera.** One setting for all the cells, captured with the same camera by the user's account: the most
common JPEG table set, the median background and the median noise SD over the cells (the noise SD an upper bound, as
now) [user].

(D-085.10) **One kernel per cell.** Each cell is calibrated on its own near-field and far-field nodes (Blocks 10 and 15
per cell). Pooling across cells is considered only if a cell lacks enough clean dendrites for its near and far
calibrations; the pipeline checks this per cell and reports it, the minimum counts exposed (proposal 5 near-field and
5 far-field calibration nodes, PROVISIONAL) [user].

(D-085.11) **The final configuration waits for the hybrid** (D-084): S3d is completed once the hybrid renderer exists
[user].

(D-085.12) **The rest as proposed** [user]: runs resumable per cell (a cell already done is skipped); outputs under
`MyDrive/diameters/survey/<specimen>/`, the pooled tables and figures under `MyDrive/diameters/survey/`; the crop cache
`MyDrive/allen_cache`.

**Why [user; reasoning].** **[user]** throughout. **[reasoning]** (D-085.3) the table is rendered at one plane spacing
and pixel size, so a cell that differs is not covered by it; (D-085.10) the kernel may differ from cell to cell (depth
in the slice, tissue), which one pooled kernel would hide.

**What it implies.** The blocks, SPEC first: 16 the cell pool and its stratified selection, the plane-spacing and
pixel-size check over all cells, the per-cell alignment, the resumable runner; 17 the real-cell survey (S3a) without
the thick-node check; 18 the simulator inputs across cells (S3b); 19 the per-cell kernel calibration with its
sufficiency check (S3c); 20 the collector of the suggestions (S3d), completed after the hybrid. **[reasoning]** The
kernel is inside the simulation hash, so one kernel per cell implies one table per cell, unless the per-cell kernels
are compared and one table is built where they agree within their SEs -- put to the user.

**What it does not decide [open].** The readings of (D-085.1); $d_{\max}$'s rule; the sufficiency thresholds; one table
per cell or one table where the kernels agree; what is done with cells whose plane spacing or pixel size differ.

**Sources.** **[user]** 2026-10-26, 17:09. **[KB]** D-065, D-067, D-084, the plan of 2026-10-20 (S3).

**Status:** active; not implemented. [updated 2026-10-26, later 6: 16.1 built locally -- the cell pool, the acquisition
check, the stratified selection, SPEC Block 16.1 -- with (D-085.1)'s readings as PROVISIONAL defaults; (D-085.7)'s
rule settled by D-086] [updated 2026-10-26, later 7: (D-085.1)'s sparsely spiny cells a third class, D-087; (D-085.10) superseded by D-088] [updated 2026-10-27: Blocks 16.1-19 built and on the remote, `2fd8ffb`..`7064e2c` (the plan row `826eb02`), smoke-tested in the sandbox, not run on Allen's data; (D-085.2)'s projection rule confirmed (D-089.2) and checked by Block 16.3 (D-090.1); (D-085.11) holds: S3d waits for the hybrid]
[updated 2026-10-27, later 3: (D-085.3)'s check run on Allen's data (Cell 11, `check`): the 47 lines the user pasted (the first one cut, so 46 specimen ids seen, the pilot 529878215 among them) all read dz 0.28 um, pixel 0.1144 um, 276 to 651 planes; the user: "All share the same." The check's summary line was not pasted, so the pool's size and any failed cells are not recorded here. No cell seen differs, so (D-085.3)'s rule for cells that differ has nothing to act on.]
[updated 2026-10-28, the user's logs of 11:16 (Colab, the S3 run file of 2026-10-27): Cell 11 -- the pool holds 158 human cells with a reconstruction (2333 rows read in one query): 111 exc, 38 inh and 9 `sparsely_spiny` over 15 class-layer groups; (D-085.3)'s check covered all 158 (6.6 min), 0 failed, every one at dz 0.28 um and 0.1144 um/px (168 to 660 planes); the selection (D-085.1) is 51 cells (seed 20261026, at most 5 per group), none left out. Cell 12 -- (D-085.2)'s alignment (29.7 min, 10 to 184 s per aligned cell): 47 cells aligned, 42 kept and 5 set aside (D-089.1); 4 failed before aligning (569095789, 508279351, 596036453, 525018757) because `allen_image_io.fetch_swc` downloaded Allen's marker file (`*_marker_m.swc`, which passes its `.swc` suffix test; its comma-separated lines fail `read_swc`) instead of the reconstruction **[user's log, read; src]**. Cell 13 (S3a, D-085.4) was running. A fix of the SWC fetch is put to the user, not made.] [updated 2026-10-28, D-099: the per-cell survey runs on davinci, Colab keeps the pilot only; the programme itself is unchanged]

---

## D-086 -- $d_{\max}$ is the maximum of Allen's dendrite diameters across S3's cells (user)

**Date:** 2026-10-26, 17:16 (Europe/Rome), the user's answer after the explanation of what $d_{\max}$ is for (the
diameter implementation chat). **Binds:** the upper end of `PhantomConfig.d_range_um` (the production configuration,
Cell 8; in the simulation hash, so every table built after it); S3b, Block 18 (to specify), which computes it; D-065
("Allen's radii across cells set $d_{\max}$"); D-085.7 (its rule left open); D-024 (iii).

**Decision [user].** "Use the maximum. Start implementing" -- $d_{\max}$ is the maximum of Allen's $2r$ across the
cells, not the 99.5th percentile proposed in D-085.7.

**Statement.** Let $\mathcal C$ be the set of S3's cells (Block 16.1's selection) and, for each cell $c \in \mathcal C$,
$\mathcal N_c$ the set of its SWC nodes of types 3 and 4 (basal and apical dendrite, the pooling of
`scripts/allen_radius_distribution.py`), with $r_{{\rm A},i}$ (um) node $i$'s radius in Allen's reconstruction. Then
$$d_{\max} = \max_{c \in \mathcal C}\ \max_{i \in \mathcal N_c}\ 2\,r_{{\rm A},i}\quad(\mathrm{um}),$$
the upper end of `phantom.d_range_um` $= (d_{\min}, d_{\max})$: the largest of the per-cell `dend_p100` of the radius
script run over the cells of $\mathcal C$.

**Why [user; reasoning].** **[user]** the choice. **[reasoning]** With the maximum, the phantoms' true-diameter range
covers every Allen dendrite diameter of S3's cells (Allen's $2r$ standing in for the true diameter, which is not known);
a percentile would leave the thickest nodes beyond the range by construction, and beyond the table's reach a node is
`out_of_domain` and filled (D-085.7). The cost: a single node -- a thick trunk near the soma, or a tracing artefact --
sets the range for every cell, and a fixed number of draws spread over a wider range puts fewer of them per unit of
diameter where most nodes are.

**What it implies for code and tables.** S3b (Block 18) computes $d_{\max}$ this way and reports the cell and the node
that set it, with the per-cell maxima and the pooled p99 and p99.5 beside it as comparisons [reasoning: a proposal for
Block 18's SPEC]; Cell 8's `phantom.d_range_um` takes it when the user writes it into `DECIDED` (D-065). Nothing in
Block 16.1 changes.

**What it does not decide [open].** $d_{\min}$; whether nodes are screened for tracing artefacts before the maximum (one
outlier sets it; to be read on S3b's report); whether the maximum runs over S3's selection only or over every pool cell
that passes the acquisition check (read here as S3's selection, PROVISIONAL); the soma and the axon are not dendrite
nodes and are not counted.

**Sources.** **[user]** 2026-10-26, 17:16. **[KB]** D-024 (iii), D-065, D-085.7. **[src]**
`scripts/allen_radius_distribution.py` (`--types` default 3,4; `dend_p100`).

**Status:** active; not implemented (Block 18, to specify). Settles D-085.7's open rule, superseding its proposal (the
99.5th percentile). [updated 2026-10-27: implemented in Block 18 (`s3_inputs.d_max`: the maximum over the cells' dendrite nodes, with the cell and node that set it, the per-cell maxima and the pooled p99 and p99.5 beside it), on the remote at `83b400b`; S3's selection is the set of cells, as read; not run on Allen's data]

---

## D-087 -- Sparsely spiny cells are a third class of S3's selection (user)

**Date:** 2026-10-26, 17:59 (Europe/Rome), the user's answer to the open points of the Block 16.1 report (the diameter
implementation chat). **Binds:** Block 16.1's class map (`SurveyConfig.class_map`) and the groups of the stratified
selection; every per-class statistic and figure of S3a-S3d (D-085.5: "per layer and class"); D-085.1, whose reading
left the sparsely spiny cells "to be placed by the user".

**Decision [user].** "Sparsely spiny keep as a third class".

**Statement.** The class of a cell is read from Allen's dendrite type (exact match, case-insensitive): spiny -> `exc`,
aspiny -> `inh`, sparsely spiny -> `sparsely_spiny`; a cell without a dendrite type has no class and is not selected.
The selection draws 5 cells per (class, layer) for each of the three classes, all of a smaller group (D-085.1). The
names `exc` and `inh` for spiny and aspiny remain the reading of D-085.1 (PROVISIONAL); the name `sparsely_spiny` is the
assistant's.

**Why [user; reasoning].** **[user]** the choice. **[reasoning]** The sparsely spiny cells are assigned to neither
excitatory nor inhibitory here, so a class of their own keeps them in the survey without a guess at their identity.

**What it implies for code and tables.** `SurveyConfig.class_map` gains ("sparsely spiny", "sparsely_spiny") as a
default; Block 16.1's oracles change accordingly (the canned sparsely spiny group is now selected); S3a's per-class
figures show three classes.

**What it does not decide [open].** Whether the third class's nodes are pooled with the others in the coverage
statistics of D-085.5 (read as pooled, like exc and inh, PROVISIONAL).

**Sources.** **[user]** 2026-10-26, 17:59. **[KB]** D-085.1. **[src]** Block 16.1 (`2fd8ffb`).

**Status:** active; Block 16.1 is amended for it the same day (SPEC Block 16.1, its status line). [updated 2026-10-27: implemented and on the remote, `39d6ef9`; the name `sparsely_spiny` confirmed, D-089.4; the pooling of its nodes in the coverage, as read, unchanged]

---

## D-088 -- One defocus kernel, calibrated on nodes pooled across cells; the test on one pilot cell; 10 to 30 nodes per calibration set (user)

**Date:** 2026-10-26, 17:59 (Europe/Rome), the user's answers to the open points of the Block 16.1 report (the
diameter implementation chat). **Binds:** the defocus kernel's calibration -- Blocks 10 (near field) and 15 (far
field), `scripts/calibrate_kernel.py`, notebook Cell 7 -- and S3c (Block 19, to specify); the production
configuration's kernel (`renderer.kernel_table_*`, `kernel_continuation_slope`; in the simulation hash) and so the
number of tables; D-085.10, which it supersedes.

**Decision [user].** "I do not know how fast is the entire pipeline for the table. For now I would say a single kernel
whose calibration for the defocusing kernel is constructed across nodes belonging to several cells. For now for the
test we will use data from a single pilot cell", and, on the minimum number of calibration nodes: "Do maximum 30 per
type and minimum 10".

**Statement.**

(D-088.1) **One kernel.** One defocus kernel $\sigma_{\rm r}(\delta)$ -- Block 10's near table to $\delta_0$ = 0.84 um
continued linearly with Block 15's far slope $\hat\gamma$ (D-080.1) -- for all cells, calibrated on the near-field and
far-field calibration nodes pooled across several of S3's cells; so one production configuration and one table, not
one kernel and one table per cell [user].

(D-088.2) **The test now.** For now the calibration is tested on the nodes of a single pilot cell [user]; read as
specimen 529878215, Phase II's pilot cell, through Cell 7 as it stands (PROVISIONAL reading).

(D-088.3) **10 to 30 nodes per set.** Each calibration set -- near field ($D_{\rm A} \le$ 0.6 um) and far field
($0.6 < D_{\rm A} \le 0.8$ um), D-067.4 -- holds at least 10 and at most 30 nodes [user; "per type" read as per
calibration set, PROVISIONAL]. Below 10 the set is insufficient: the fit still runs as a diagnostic and says so, and
no production kernel is offered from it (the assistant's proposal, PROVISIONAL). Above 30 qualifying nodes, 30 are
scanned, taken in a recorded seeded order and, when several cells contribute, in turn across the cells (the
assistant's proposal, PROVISIONAL).

**Why [user; reasoning].** **[user]** the cost of the whole table pipeline is not known yet, and one table per cell
would multiply it. **[reasoning]** Pooling averages the kernel over the cells' depths in the slice and their tissue;
the per-cell differences D-085.10 meant to keep can still be read from the pooled fit's per-node residuals grouped by
cell. The cap bounds the far scans' cost (33 planes of a +-25 um patch per far node); the minimum bounds the
bootstrap's small-sample error (the node bootstrap read 0.83 of the true SD at $J$ = 12, D-074's Why).

**What it implies for code and tables.** `CalibrationConfig` gains the minimum and the maximum per set (10, 30; in
no signature, like the rest of `CalibrationConfig`). `calibrate_kernel.py real` stops scanning a set once it holds 30
scanned nodes, taking the candidates in a seeded order; `fit` and `far-fit` label a set with fewer than 10 nodes
insufficient and withhold the production kernel. Block 19 pools the scans of several cells: node ids are qualified by
the specimen (two cells can share a node id, and the fits carry one nuisance parameter per node), and the cap is
filled in turn across the cells. One table follows (S4's job, unchanged).

**What it does not decide [open].** Which and how many cells feed the pooled calibration; whether the bootstrap
resamples cells rather than nodes once several cells contribute (nodes of one cell are not independent, a reasoning
point); whether per-cell kernels are fitted as a check of the pooling; the reading of "per type" (if it meant per
neuron class, the counts apply per class instead).

**Sources.** **[user]** 2026-10-26, 17:59. **[KB]** D-067.4, D-074, D-080.1, D-085.10. **[src]**
`scripts/calibrate_kernel.py` (`real`, `fit`, `far-fit`), `analysis/calibration.py` (`scan_cell`, `scan_node`).

**Status:** active; not implemented. Supersedes D-085.10 (one kernel per cell, pooled only when a cell lacks clean
dendrites; the per-cell sufficiency check with 5 + 5 nodes, PROVISIONAL); answers the open point "one table per cell
or one table where the kernels agree": one table. [updated 2026-10-27: implemented in Block 19, on the remote at `7064e2c` -- Cell 7 stops at 30 scanned nodes per set in a seeded order and withholds the production kernel below 10 per set; the step `calibration` (5 scans per set per cell) and `pool --what calibration` (round-robin over the cells to 30 per set, qualified ids); the readings confirmed, D-089.6-8; the points by cell and their residuals, D-090.2 (Block 19.5); not run on Allen's data]

---

## D-089 -- The defaults of S3's blocks confirmed (user): the set-aside ratio 2, the darkest-projection rule, the stratified default, `sparsely_spiny`, the z0 flag at 1 um, 5 scans per set per cell, "per type" as per calibration set, 529878215 the pilot

**Date:** 2026-10-27, 14:19 (Europe/Rome), the user's answer to the S3 report of 2026-10-26 (the diameter implementation
chat). **Binds:** `SurveyConfig.align_min_score_ratio` (Block 16.2 (e)); the projection rule of Block 16.2 (a);
`SurveyConfig.nodes_mode` (Block 17.1); the class name of D-087; `SurveyConfig.inputs_z0_flag_um` (Block 18);
`SurveyConfig.calib_scans_per_cell` (Block 19.3); the readings of D-088.2 and D-088.3. None is in a signature.

**Decision [user].** "Ok for the default" -- to the report's list "Defaults I chose that need your OK" (2026-10-26).

**Statement.** Confirmed as the defaults [user]:

(D-089.1) A cell is set aside when its alignment's score ratio $R$ is below 2 (`align_min_score_ratio`; Block 16.2 (c)-(e)).

(D-089.2) Where a cell has several xy projections (SubImages other than Primary with the planes' size), the darkest --
the lowest median grey level at level 4 -- is its minimum-intensity projection. What the projections are is checked
by Block 16.3 (D-090.1).

(D-089.3) `nodes_mode` = `stratified` by default -- up to 20 nodes per bin of Allen's $2r$ -- with `all`, every dendrite
node, the alternative (D-085.4).

(D-089.4) The third class of D-087 is named `sparsely_spiny`.

(D-089.5) A cell whose median depth offset $\Delta z^*$ (registration survey) exceeds 1 um is flagged `z0`
(`inputs_z0_flag_um`).

(D-089.6) Each cell contributes at most 5 scans per calibration set to the pool (`calib_scans_per_cell`); so a set
reaches the minimum of 10 only from 2 cells, and the cap of 30 only from 6.

(D-089.7) "Do maximum 30 per type and minimum 10" (D-088.3) counts per calibration set -- near field and far field --
not per neuron class.

(D-089.8) The pilot cell of D-088.2's test is specimen 529878215, Phase II's pilot.

**Why [user; reasoning].** **[user]** the confirmation. **[reasoning]** These were the assistant's proposals, marked
PROVISIONAL in the code and the SPEC; the list did not hold the other PROVISIONAL readings and values -- D-085.1's pool
and its exc/inh names for spiny/aspiny, the selection seed, the null's size and tiles, the fine stage, the tolerances,
the seeded orders, the round-robin of the pool, the rule below the minimum (diagnostics, no production kernel) -- so
those stay PROVISIONAL.

**What it implies for code and tables.** The `# source:` comments of `config.py` and the SPEC entries of Blocks 16-19
read "confirmed, D-089"; no value and no hash changes (estimator `0508e7ff12f48728`, simulation `0ea5a6d2ec069c68`).

**What it does not decide [open].** The readings and values left PROVISIONAL above.

**Sources.** **[user]** 2026-10-27, 14:19. **[KB]** D-085, D-087, D-088; the report of 2026-10-26 (its list of
defaults).

**Status:** active; implemented 2026-10-27 (comments and SPEC, `6b8ea64`). [updated 2026-10-28, the Colab run of Cells 11b and 12 (the user's logs and figures of 11:16): (D-089.2) each of the 52 specimens listed (the 51 selected and the pilot) has a `'MinimumIntensityProjection - xy'`; 4 also have a `'MaximumIntensityProjection - xy'` (508305938, 508279351, 525018757 and the pilot 529878215), and in each of those 4 the darkest at level 4 was the maximum's; Cell 12 aligned 508305938 on it and failed (y-flipped, shift (-6, 1606) full-res px, ratio 1.03, set aside; the overlay shows the skeleton below the cell); 508279351 and 525018757 failed before aligning (D-085's note). On the pilot the minimum's xy is the planes' minimum (D-090.1's check, below), so the rule's premise -- the minimum-intensity projection is the darkest -- fails on Allen's data wherever a maximum-intensity one exists **[user's logs, read]**. (D-089.1) R spans 1.03 to 12.08 over the 47 aligned cells; 5 are below 2: 508305938 and four whose shifts, (0, 2), (2, 0), (2, 2) and (2, 2) full-res px, are like the 42 kept cells' (all within 4 px but 558211203's (170, -104) px) -- in the overlays the user sent, the skeleton lies on the visible dendrites of 576084692, 596887423 and 611598198, and on the soma's halo of 571464200, whose dendrites are too faint there to judge **[user's figures, read: the assistant's reading]**. Changes to (D-089.1) and (D-089.2) are put to the user, not made.] [updated 2026-10-28, later: **D-098** [user, 15:01]: (D-089.2) is superseded by D-098.2 -- the xy projection is the SubImage of type 'MinimumIntensityProjection - xy', the darkest rule only when none has that type; (D-089.1)'s threshold stays 2, and the three set-aside cells that look aligned (and 571464200 after the user's look) run with `--include-set-aside` (D-098.5).]

---

## D-090 -- The two open points of the S3 report (user): what Allen's projections are is settled by a Colab check; the pooled calibration's near- and far-field points are drawn together, coloured by cell, to judge the discrepancies among cells

**Date:** 2026-10-27, 14:19 (Europe/Rome), the user's answer to the report's open points (the diameter implementation
chat). **Binds:** Block 16.2 (a)'s projection rule (D-089.2) and so the alignment of every S3 cell; S3c's pooled
calibration (Block 19, D-088.1); notebook Cells 11b and 15; the SPEC's Blocks 16.3 and 19.5.

**Decision [user].** "About the still open questions 1)understand this by yourself. If not possible write a colabr do
function and cell to run on colabrodo 2) plot the near and far field points toghether but with a cell base color code
so that we can evaluate discrepancies among cells".

**Statement.**

(D-090.1) **What Allen's projections are**: worked out from the sources at hand, otherwise by a function and a cell
run in Colab [user]. From the repository **[KB-repo: `docs/handoff_diameter_remeasurement.md`, the table of Allen data
access, from live API queries of the design chat before a compaction, not re-queried]**: specimen 529878215 has four
projections, named there min-xy (546843114, 7607 x 9435 px), min-yz (546843119, 1333 x 9435), max-xy (550303812) and
max-yz (550303817), and Primary planes of 7605 x 9433 px; their `image_type` strings were not recorded. Not possible
here: the sandbox cannot reach api.brain-map.org (403), and a WebFetch of the RMA SubImage query waits for the user's
approval (withdrawn unanswered on 2026-10-27). So Block 16.3, run as Cell 11b: for every selected cell, the SubImages
that are not Primary with every field the API returns, the xy candidates of Block 16.2 (a) and the one its darkest
rule takes; for the pilot, each xy projection against the pixelwise minimum, mean, median and maximum over its usable
planes on a 512 x 512 px (59 um) window around the soma -- the best integer offset within +-3 px, the correlation
$r^*$, the least-squares line and its residual, and the verdict `matches <op>` when $r^* \ge 0.9$ and the next
operator's line leaves at least 4 times the best's unexplained variance (the assistant's rule, PROVISIONAL; a lead in
$r$ does not decide: on synthetic stacks the planes' mean correlates with their minimum at 0.95-0.97 while its line
leaves 4 times the residual **[run, sandbox]**).

(D-090.2) **The pooled calibration by cell** [user]: the near-field and far-field points of the kernel law (Block
10.F's kernel SDs against $|\hat\delta|$) drawn together and coloured by cell, with each point's residual from the
pooled law -- the near table to 0.84 um, then the far line -- and per cell and set the median and quartiles, in a
figure and a CSV (Block 19.5). Up to three cells one panel coloured by cell; beyond three, one panel per cell over
the pool in grey (the assistant's: three hues are the most a scatter keeps apart for colour-blind readers, the dataviz
skill's validated palette).

**Why [user; reasoning].** **[user]** as quoted. **[reasoning]** (D-090.1) Cell 12's alignment rests on the darkest
projection being the minimum over the planes and on the projection's pixel grid being the planes'; the check tests both
(the offset is the projection's pixel minus the planes'). (D-090.2) One pooled kernel (D-088.1) averages over the
cells; a cell whose blur differs from the pool's shows as residuals of one sign, which colour by cell makes visible
without a per-cell fit.

**What it implies for code and tables.** Block 16.3: `analysis/projection_check.py`, `scripts/projection_check.py`,
`allen_image_io.subimage_rows` and `images_frame` (`list_images` split in two, its table unchanged), Cell 11b,
`survey/projections/`. Block 19.5: `calibration_report.law_points`, `pooled_sigma_law`, `kernel_law_by_cell`,
`calibration_pool.cell_of`, two figures in `s3c/figures/` and `s3c/kernel_law_by_cell.csv` (Cell 15). No hash
changes. A finding of the smoke test, reported with the figure: the near residuals against the production table carry,
for every cell alike, the difference between the near fit's cubic growth and the table's linear interpolation between
knots 0.28 um apart (up to 0.075 um on a synthetic growth) **[run, sandbox]**.

**What it does not decide [open].** What follows if no projection matches the minimum or the offset is not (0, 0) (on
the report); whether a cell that stands out leaves the pool; a per-cell fit of the kernel and a bootstrap over cells
(D-088's open points) stay unbuilt.

**Sources.** **[user]** 2026-10-27, 14:19. **[KB-repo]** `docs/handoff_diameter_remeasurement.md`. **[run, sandbox]**
`tests/smoke/test_smoke_projection_check.py`, `tests/smoke/test_smoke_calibration_by_cell.py`; the dataviz skill's
`validate_palette.js` on the three cell hues.

**Status:** active; implemented 2026-10-27 (`6b8ea64`); not run on Allen's data (Colab Cells 11b and 15). [updated 2026-10-28: (D-090.1) run on Allen's data (Cell 11b, the user's log and figure of 11:16; 7.0 min, 579 crops): 52 specimens listed, each with `'MinimumIntensityProjection - xy'` and `'- yz'`, 4 also with `'MaximumIntensityProjection - xy'` and `'- yz'` (the strings as the API gives them). On the pilot 529878215, a 512 px window at (4199, 5325) over 521 planes: `'MinimumIntensityProjection - xy'` (546843114) matches the planes' pixelwise minimum, r* 0.9941 at offset (0, 0) (mean 0.7040, median 0.5376, max 0.4743); `'MaximumIntensityProjection - xy'` (550303812) is `unclear`: its best r is negative for every operator (min -0.4870, mean -0.3395, median -0.2415, max -0.3080), each at an edge of the +-3 px search **[user's log and figure, read]**; in the figure its neurites are bright on a dark, speckled ground -- the assistant's reading: an inverted, contrast-stretched image, which is why it is the darkest at level 4 (D-089.2's note) **[reasoning]**. Checked on the pilot only, in one window. (D-090.2), Cell 15, not run.] [updated 2026-10-28, later: the check's result applied by **D-098.2** -- Block 16.2 (a) takes the projection by its type.]

---

## D-091 -- Every new SPEC block enters its roadmap in the same turn, without being asked (user)

**Date:** 2026-10-27, 16:28 (Europe/Rome), the diameter implementation chat. **Binds:** every roadmap of the project
(`specs/roadmap/<name>.yaml`, its rendered md and the Subproject Roadmaps page) and every SPEC block written from now on;
D-082's annotation of 2026-10-26, 16:48 (the hold on roadmap changes); the diameter-pipeline roadmap's NS2 (now HR) and
S3d.

**Decision [user].** "You have implemented the Sa,b,c but hte roadmap is not updated, you should do this automatically".

**Statement.**

(D-091.1) A new SPEC block enters its roadmap in the same turn it is written: `roadmap.py add block` under its step,
then `status`, `render`, the commit with the block's and the page's document -- without waiting to be asked. The hold of
2026-10-26, 16:48 (D-082's annotation) is lifted for this; D-082's own open questions (G1's theory part, TG3 shared by
G4 and G5) stay open.

(D-091.2) Applied at once: Blocks 16-19 entered as I16 (under S3), I17 (S3a), I18 (S3b) and I19 (S3c); the hybrid
renderer left "not scheduled" -- NS2 became HR, a step of its own -- and S3d depends on HR (D-085.11) (`689c702`,
`9b99b8b` on `sci/diameter-pipeline`; page version 55). Block 21 entered as I21 under HR with its commit (`2572e15`).
Theory statuses stay the user's (T7 open).

**Why [user; reasoning].** **[user]** as quoted. **[reasoning]** The page is drawn from the repository's yaml: a block
that is not in it does not show, however far it is built -- S3's four blocks were pushed but drawn as nothing, and I16
showed `planned` until its SPEC entry carried a plain `Modules:` line the tool parses. The skill's rule that an
implementation change needs nothing written holds for blocks already in the yaml; a new block needs `add block`.

**What it implies.** Each block's commit carries its roadmap entry; the md is rendered after the push (its colours read
the remote); the page is written in the same turn.

**What it does not decide [open].** D-082's open questions; whether the subproject-roadmap skill should say this for
every subproject (a skill change is the user's to save).

**Sources.** **[user]** 2026-10-27, 16:28. **[src]** `689c702`, `9b99b8b`, `2572e15` on `sci/diameter-pipeline`; the
page's version 55.

**Status:** active; applied (I16-I19, HR, I21).

---

## D-092 -- Go on with the hybrid renderer (user): built as SPEC Block 21, the default left at the partition; its planes go below zero in dark stain

**Date:** 2026-10-27, 16:28 (Europe/Rome), the diameter implementation chat. **Binds:** Block 4's absorption
(`RendererConfig.absorption`); D-084; I-004; the roadmap's HR (I21) and T7; S3d (Block 20, D-085.11); the simulation
hash of any configuration that selects the hybrid.

**Decision [user].** "GO on with the hybrid renderer".

**Statement.**

(D-092.1) The hybrid is built as SPEC Block 21 (Block 20 stays S3d's, the plan of 2026-10-20): Block 4's absorption
mode `hybrid` (`model/hybrid.py`). For each slab $j$ and column $(x, y)$,
$\Delta A^{\rm hyb}_j(x, y) = a_j(x, y)\,h\big(x, y, \bar\zeta_j(x, y)\big)$ with $a_j$ the slab's exact absorbance (S4),
$\bar\zeta_j$ the middle of the slab's inside part and $h(p) = \langle e^{-\mu\ell_{\rm back}(p;\hat s)}/s_z\rangle_W$ the
cone's average of each ray's own history times its path per unit depth ($\ell_{\rm back}$ the length inside the tube of
the ray that reaches $p$, before $p$; $W$ the evenly filled NA 1.4 cone, $n_{\rm oil}$ 1.515); then every plane is
rendered with the kernel as before -- the renderers document's Eqs. 13-14 (2026-10-09, s.3.8, on `main`) [user: to
build it; the formulation the document's].

(D-092.2) The assistant's choices, PROVISIONAL until the user confirms them: the name `hybrid`; Block 9's directions
(`ray_na`, `ray_n_oil`, `ray_n_rho` 16, `ray_n_psi` 32: one condenser for the ray world and the hybrid, no new field,
so the default's hashes stay estimator `0508e7ff12f48728` and simulation `0ea5a6d2ec069c68`; a configuration with the
hybrid has simulation hash `5e5b04a2f154e1c8`); the slab rule (the middle of the slab's inside part times its exact
$a_j$: the document's midpoint rule inside the tube, and the faint limit $1.447\,a_j$ exact at a column's ends); each
slab computed once per render and served to every plane; the default left at `partition_vertical` -- D-084.2 ("until
the hybrid is built and tested"), and the findings below are the user's to weigh first.

(D-092.3) Findings, for the user and T7 **[run, sandbox: `checks/hybrid_2026-10-27/`, infinite flat tubes in 1-D,
light +1, 16 x 32 directions]**: (a) the hybrid conserves the light per direction -- its booking over a tube equals the
ray world's total shadow (to 6e-10 at $\mu d$ = 3 and 1.7e-7 at 50, per unit length of a flat 1 um tube) -- but not per
column: the centre column of that tube books 1.03 times its own light at $\mu d$ = 1.5 and 1.18 at 3, the oblique rays
entering through the side walls being booked where they are absorbed. (b) So it is not positivity-preserving: with the
production Gaussian kernel, which is narrower than the cone beyond about 0.1 um (the document's (15)) and does not
spread that light back out, the node plane's minimum of $\tau = I/B$ is -0.057 / 0.005 / 0.042 / 0.040 at $\mu d$ = 2
($d$ = 0.5 / 1 / 2 / 3 um) and -0.130 / -0.096 / -0.034 / -0.013 at 3, and the planes on the light's entry side go
below zero earlier (-0.062 at $d$ = 1 um, $\mu d$ = 2, 0.28 um before the axis); the partition's planes go below zero
too, from $\mu d$ = 5 off focus. (c) Like for like -- the cone's own kernel weighted $W^*$ against the exact ray world
without post blur, profile level, no fit -- its largest $|\tau - \tau^{\rm ray}|$ is 0.009 / 0.060 / 0.105 / 0.227 /
0.436 at $\mu d$ = 0.3 / 1 / 1.5 / 3 / 10 against the partition's 0.085 / 0.195 / 0.254 / 0.365 / 0.499 at the same
$\mu$, its dip area within 0.4 % of the ray world's up to $\mu d$ = 3; with that kernel too it goes below zero from
$\mu d$ = 3 on the entry side and at 10 in focus: the light the document saw moved from the centre lands near the
tube's edges and overshoots there. (d) Cost, one sandbox core, an 87 x 87 px block with 17 px padding at $p_{\rm x}/8$,
7 planes: 11.8 s against the partition's 2.8 s at $d$ = 0.3 um, 95.8 against 6.0 s at 1 um, 432 against 8.6 s at 2 um
(flat), growing as $d^2$; at 8 x 16 directions the planes stay within 1.2e-3 of 32 x 64 at a quarter of the cost.

**Why [user; reasoning].** **[user]** as quoted, after D-084 had promoted the hybrid to the renderer actually used once
built. **[reasoning]** (D-092.2) Reusing Block 9's cone keeps one model of the condenser and leaves every other
configuration's hash where it was; the slab rule is the document's Eq. 13 written with (S4)'s exact overlaps. (D-092.3)
The phantoms' design reaches $\mu d$ = 12 ($\mu \le$ 3/um, $d \le$ 4 um), so the dark failure is inside the table's
range; whether real nodes are there is what S3a's coverage (Cell 13) will show.

**What it implies.** Code: `model/hybrid.py`, the mode in `model/render.py` (its slabs cached per render),
`config.ABSORPTIONS`; the prototype `checks/hybrid_absorption_check.py` and its output copied from `main`;
`tests/smoke/test_smoke_hybrid.py` (7 pass; the prototype's six centre dips reproduced to its print); commit
`2572e15`. Nothing changes for any table or Colab run: the default stays the partition. S3d still waits (D-085.11), now
on the user's choice among SPEC section 8's options.

**What it does not decide [open].** (a) Whether the hybrid becomes the default as it is (with the phantoms' $\mu d$
limited to where it stays positive), after a variant that conserves light per column or per line (T7), or not at all
(the partition stays); (b) when the default switches; (c) a cheaper quadrature for the production (a field of the
hybrid's own, 8 x 16) and other speed-ups; (d) the kernel under the hybrid in dark stain (D-084's question).

**Sources.** **[user]** 2026-10-27, 16:28. **[doc]** the renderers document of 2026-10-09, s.3.4-3.9 (on `main`).
**[src]** `2572e15` on `sci/diameter-pipeline`. **[run, sandbox]** `tests/smoke/test_smoke_hybrid.py`,
`checks/hybrid_2026-10-27/` (`profile_survey`, `cone_survey`, `directions_planes`, `cost`).

**Status:** active; implemented 2026-10-27 (`2572e15`), smoke-tested; the default not switched.

---

## D-093 -- The calibration nodes are not screened for straightness; flatness stays strict; if the test returns, 2 (or 4) radii (user)

**Date:** 2026-10-27, 19:08 (Europe/Rome), the diameter implementation chat, after Cell 7's first run on the pilot
529878215 had rejected every candidate. **Binds:** Block 10.I (v), hence the near and the far calibration sets
(Blocks 10 and 15) and S3c's per-cell calibration scans (Block 19.3, which select with the same
`calibration.scan_cell`); `CalibrationConfig.node_straight_check` (new), `node_straight_tol_radii`; D-081.3's
straightness part (superseded); Colab Cell 7; the runbook (v30).

**Decision [user].** "Relax the straigthness. We need just flatness to be strict. The straigthness can be removed
because we will evaluate onlu the measuring line. Otherwise relax the straightness as 2 or 4 × r_A."

**Statement.**

(D-093.1) **No straightness test by default.** A calibration node is no longer rejected as `not_straight`. The test
of 10.I (v) stays in the code behind `CalibrationConfig.node_straight_check` (bool, default False). When it is
switched on, node $j$ (Allen's radius $r_{{\rm A},j} > 0$ at the node, um, from the reconstruction) fails if
$e_j > c\,r_{{\rm A},j}$. Here $e_j = \max_i \lVert(\mathbf I - \hat{\mathbf t}\hat{\mathbf t}^{\top})(\mathbf p_i -
\bar{\mathbf p})\rVert$ over the branch's traced points $\mathbf p_i$ (um, 3-D) with $|s_i - s_j| \le h$ along the
trace, $\bar{\mathbf p}$ is their mean and $\hat{\mathbf t}$ their total-least-squares direction
(`calibration_scene.straightness_excess`). The half-window is $h$ = `node_straight_half_um` = 10 um, unchanged; the
tolerance is $c$ = `node_straight_tol_radii` [user].

(D-093.2) **The tolerance when the test is on** is $c$ = 2, was 1 (D-081.3). The user offered 2 or 4; taking 2 as
the default is the assistant's choice, PROVISIONAL, and 4 is one configuration value away.

(D-093.3) **Unchanged** [user: "We need just flatness to be strict"]: flatness $\varphi_j \le$ `node_phi_max_deg` = 5
deg for both sets (D-081.3); faintness $\hat\alpha_j \le 0.5$; membership of $\mathcal S$; Allen's diameter bounds of
the two sets (D-067.4); the isolation test (a predicted shadow deeper than 0.03 gl inside the $\pm$25 um patch in a
scanned plane fails the node, D-081.1).

**Why [user; reasoning].** **[user]** as quoted. The user had reported (18:58) that none of the pilot's 68 candidates
was a calibration node. Of those 68 lines the user pasted 57 (the first one cut), and all 57 carry `not_straight` at
one radius: 46 with `not_isolated` too, 11 with `not_straight` alone (nodes 3985, 8472, 8497, 3972, 3980, 3976, 3999,
3973, 3984, 8473, 4000). The other 11 lines were not seen.

**[reasoning]** The calibration's statistic is the width of the dip's Gaussian core on the profile across the
measuring line at the node. The trace away from the node enters that profile only through the defocus blur, which
takes in the tube within about $\pm 3\sigma_{\rm r}(\delta)$ along the branch. **[run, sandbox: `model.kernel.sigma_r`
on the default configuration]** That reach is about $\pm$1.8 um at the near set's outermost planes ($\sigma_{\rm r}$(0.84
um) = 0.603 um in the default table, ideal Debye, not calibrated). At the far set's offset of 4.48 um it is about
$\pm$10 um ($\sigma_{\rm r}$ = 3.48 um with the default continuation slope 0.79; 3.22 to 5.01 um for slopes 0.72 to
1.21). So the user's argument holds fully for the near set and only in part for the far set: a bend within about 10 um
of a far node still enters its profiles.

**What it implies.** Code: `config.py` (the switch, $c$ = 2, validated as a bool); `calibration.trace_reasons` (the test
only with the switch on); `tests/smoke/test_smoke_calibration.py`, which checks:
- the defaults are off and 2;
- with the switch on at 1 radius, the old assertions still hold;
- at 1, 2 and 4 radii, `not_straight` exactly when the excess exceeds $c\,r$.

Three mutants were killed: always test, never test, and the tolerance hard-coded to 1. The commit is `6a74735`; the
evidence's wording was corrected in `d46340c` and the blur-reach caveat in `eed549d` (SPEC section 8). The estimator
and simulation hashes are unchanged (`0508e7ff12f48728`, `0ea5a6d2ec069c68`): the calibration fields are not part of
them.

Colab: re-run Cells 0, 1 and 7. The 11 nodes that failed straightness alone would now be scanned. To restore the test,
pass a config JSON `{"calibration": {"node_straight_check": true}}` (with `"node_straight_tol_radii": 4.0` for 4
radii) to `calibrate_kernel.py real` as `--config-json`. Each scans file records the calibration settings it ran
with.

**What it does not decide [open].**
- (a) Whether the far set keeps a relaxed test (2 or 4 radii) while the near set has none. This is the far-field caveat
  above, in SPEC section 8.
- (b) The isolation test. It is unchanged, and it failed 46 of the 57 lines seen.
- (c) Whether either set now reaches D-088's 10 to 30 nodes per calibration set. The pasted lines do not say which
  set each node belongs to, so this is known only when Cell 7 is re-run.
- (d) Whether a restored test's half-window should follow $\pm 3\sigma_{\rm r}(\delta)$ rather than a fixed 10 um
  **[reasoning]**.

**Sources.** **[user]** 2026-10-27, 18:58 (Cell 7's output, 57 of the 68 lines) and 19:08 (the decision). **[src]**
`6a74735`, `d46340c`, `eed549d` on `sci/diameter-pipeline`. **[run, sandbox]** Suites run, none failing:
- `test_smoke_calibration.py`: 8 pass;
- `test_smoke_phase2.py`: 4 pass, 3 skip;
- `test_smoke_far_field.py`: 6 pass, 1 skip;
- `test_smoke_config.py`, `_calibration_scene`, `_s3c`, `_calibration_figures`, `_calibration_by_cell`: 6 pass,
  1 skip each;
- the three mutants, all killed;
- `model.kernel.sigma_r` on the default configuration.

**Status:** active; implemented 2026-10-27 (`6a74735`, with `d46340c` and `eed549d`), smoke-tested; not re-run on
Allen's data (Cell 7). [updated 2026-10-28: of (D-093.3)'s unchanged criteria, faintness ($\hat\alpha_j \le 0.5$) and, for the calibration nodes, the flag `faint` of $\mathcal S$ are dropped by **D-094** [user]; flatness, Allen's diameter bounds and isolation stay.]

---

## D-094 -- The calibration nodes are selected without the faint criteria: no optical-depth cut, and the flag faint not counted (user)

**Date:** 2026-10-28, 09:12 (Europe/Rome), the diameter implementation chat. **Binds:** Block 10.I (i) and (iv), hence
the near and the far calibration sets (Blocks 10 and 15), S3c's per-cell calibration scans (Block 19.3), the pilot
CSV's `calibration_node` / `calibration_set` (`survey.pilot_row`) and the CSV pre-filter `candidates_from_rows`
(Cell 7, the run file's pre-flight); `CalibrationConfig.node_alpha_check`, `node_faint_flag_check` (new); D-067.4's
faint criterion and, for the calibration nodes only, its "in $\mathcal S$" as far as the flag `faint`; D-093.3's list
of what stays; D-026 (e).

**Decision [user].** "Faint: the pilot's fitted optical depth is α̂ ≤ 0.5 (provisional). This is not the `faint` flag
above, which removes nodes whose dip is under 6 grey levels. Remove both faint" (the first two sentences quote the
assistant's list of the candidate criteria of 2026-10-27, 19:02).

**Statement.**

(D-094.1) **No optical-depth criterion.** A calibration node is no longer rejected as `not_faint`. Criterion (iv),
$\hat\alpha_j \le$ `node_alpha_max` (0.5, PROVISIONAL) on the pilot's fitted optical depth $\hat\alpha_j$, runs only
when `CalibrationConfig.node_alpha_check` is true; the default is false [user].

(D-094.2) **The flag `faint` is not counted.** The flag (a dip under `measure.faint_min_dip_gl` = 6 gl on the pilot's
plane) no longer fails a calibration node unless `CalibrationConfig.node_faint_flag_check` is true; the default is
false. The other reasons of $\mathcal S$ still fail it: the fit's status, the registration, `crossing`, `stack_edge`,
`bbar_few`, `profile_nan` [user]. The code: `calibration.selection_flags(cfg)`, `calibration.selection_reasons(result,
cfg)`; `candidates_from_rows` reads the same flags.

(D-094.3) **$\mathcal S$ itself is unchanged.** Block 6's rule (`reject_reasons`), the bias table, the per-node
deliverable and the CSV's `in_S` and `reject` keep the flag. This is the assistant's reading of "Remove both faint",
said to the user before the work began: the request came within the list of the calibration's candidate criteria. The
other reading -- the flag dropped from $\mathcal S$ -- would change which phantoms and real nodes enter the table and
the inversion, so a new table. To confirm.

**Why [user; reasoning; run].** **[user]** as quoted, the day after Cell 7 found no calibration node on the pilot
(D-093). **[reasoning]** The cut $\hat\alpha \le 0.5$ was there for the faint limit: the calibration's identities hold
for a dip that is the kernel applied to the projected absorbance (D-026 (b)), and D-026 (e) asked for faint thick
nodes because the partition renderer fails for dark thick tubes. **[run, sandbox:
`checks/d094_2026-10-28/dark_node_growth.py`]** On noise-free flat phantoms of the near set ($d$ = 0.3 and 0.5 um),
rendered with the default table on paired geometries and fitted by Block 10's growth fit, the fitted kernel moved
little with the optical depth $\alpha = \mu d$ (model level):
- under the partition, five phantoms per ($d$, $\alpha$), between $\alpha$ = 0.1 and 2: $\hat\sigma_{\rm r}$(0.84 um)
  by less than 1 % (0.597-0.604 um), $\hat\sigma_{\rm r}$(0.56 um) by at most 1.7 %;
- under Block 21's hybrid, $d$ = 0.3 um, four phantoms, $\alpha$ = 0.1, 1, 1.5: by 0.7 % and 1.0 %;
- at the one-plane knot, whose growth is 0.0085 um$^2$ in the table and already within Phase I's bias, $\hat G$(0.28
  um) moved by up to 0.005 um$^2$ at $d$ = 0.5 um ($\hat\sigma_{\rm r}$ 0.106 -> 0.081 um).

A by-product of the same run, independent of $\alpha$: at $\alpha$ = 0.1 the fitted growth sits below Phase I's oracle (the projected faint-limit growth of a tube of that $d$) by 0.020 um$^2$ ($d$ = 0.3 um) and 0.047 um$^2$
($d$ = 0.5 um) at 0.56 um, four to eleven times the largest $\alpha$ effect there; the first-stage table at two planes
comes out 0.389 and 0.357 um against the configured 0.438 um. Within the near set, a node's thickness moves the
first-stage table far more than its darkness; procedure s.3.4's step 2 (not built) is where that is meant to be
absorbed (SPEC section 8).

These are statements about the two renderers, not about the microscope: whether either images a dark tube as the
microscope does is T7's question (the partition's profiles differ from the ray world's by up to about 0.2-0.25 in
$\tau$ at $\mu d$ = 1-1.5, D-092.3 (c)).

**What it implies.** Code: `config.py` (the two switches, validated as bools); `calibration.criteria_reasons` (iv only
with the switch), `selection_flags`, `selection_reasons`, `calibration_reasons`; `calibrate_kernel.candidates_from_rows`;
`tests/smoke/test_smoke_calibration.py` (the old rule kept with both switches on, the new default, each switch alone,
$\mathcal S$'s own rule unchanged, the CSV rule on 18 rows under both configurations). Commit `223a5a6`. The estimator
and simulation hashes are unchanged (`0508e7ff12f48728`, `0ea5a6d2ec069c68`). Colab: re-run Cells 0, 1 and 7; the run
file's pre-flight prints the new candidate count by set from the pilot CSV (no download). To restore either test, pass
`{"calibration": {"node_alpha_check": true}}` or `{"calibration": {"node_faint_flag_check": true}}` to
`calibrate_kernel.py real` as `--config-json`.

**What it does not decide [open].**
- (a) Whether the far set keeps $\hat\alpha \le 0.5$: D-026 (e) argued the cut there, and the check above covers the
  near set only.
- (b) The faint-flagged nodes' noise: their defocused planes fall to a few noise SDs, and whether their core fits add
  only scatter or also a low-SNR bias to the growth is not measured **[reasoning]**.
- (c) Whether $\mathcal S$ itself drops the flag (D-094.3).
- (d) How many calibration nodes the pilot now gives: Cell 7's re-run, with the isolation test unchanged (it failed 46
  of the 57 lines seen on 2026-10-27, D-093).

**Sources.** **[user]** 2026-10-28, 09:12. **[src]** `223a5a6` on `sci/diameter-pipeline`. **[run, sandbox]** all 36 smoke suites (`test_smoke_calibration.py` 8 pass with the new checks);
six mutants killed (each switch ignored or inverted, the CSV filter's old flags, the reasons unfiltered); `checks/d094_2026-10-28/` (`dark_node_growth.out`, `dark_node_growth_hybrid.out`). **[KB]** D-026 (b), (e);
D-067.4; D-092.3 (c); D-093.

**Status:** active; implemented 2026-10-28 (`223a5a6`), smoke-tested; not re-run on Allen's data (Cell 7). [updated 2026-10-28: (D-094.3), the assistant's reading that $\mathcal S$ kept the flag `faint`, is superseded by **D-096** [user]: $\mathcal S$ drops the flag for the table and the per-node diameters. D-094's calibration switches keep their meaning; `node_faint_flag_check` now adds the flag for the calibration only.]

---

## D-095 -- The SS window start is read from the exponential decay of the mean windows: mean windows per polarity and per sweep over [-10, 200) ms, $t_{\rm SS}(c)$ where the decay falls within $\kappa_{\rm tr}\hat\sigma_{\rm SS}(c)$, no binning, and the same mean window subtracted on $[t_{\rm SS}(c), 200)$ (user)

**Date:** 2026-10-28, 09:55 (Europe/Rome), the noise-fit chat. **Binds:** D-051 (iv)'s window-start test (the 5 ms
bins of $\bar r_{\rm odd}$, topic doc eq. (27)) and the grouping of (D-051.1); D-076 (ii) (its $t_{\rm tr}(c)$, and
the assistant's choice of the cell-level mean window for the per-sweep statistics); D-034 (ii) and comment (c); the
options (a)-(c) of 2026-10-23 (later, 4); `Ih Fit/specs/SPEC.md` §2.2 and Block 3 (`noise_windows.transient_start`,
`subtract_mean_response`, `WindowConfig.kappa_tr`, `transient_bin_ms`, `transient_floor_z`, `t_ss_floor_ms`,
`fallback_t_ms`); Block 4's simulator (`noise_fit.simulate_cell_windows`, its subtraction); the smoke checks W2, W3
and F2; the column `noise_ss_enl_t0_ms`.

**Decision [user].** "rk is calcualted only on the first 10ms ad hten we use it to calculate the mean r_depo and
r_hypo. What you have to do is the following thing which will settle everything. Calculate the mean windows (per
pulse polarity) from -10 to 200 ms. this means that we have a window that spans -10 200ms whose samples are the mean
(or median) values acorss the K_p windows. whith these windows (always per polarity divided) we do 2 things. first we
calculate at which point statistically the exponantial response decay (from the mean window) is within the k_tr *
sigmahat_SS(c). This will be our t_SS(c). then the cut real traces which will span [t_SS(c) , 200) will be subtraced
the mean window witin the same range (just to be sure). No binning, per polarity and per sweep." It answers, by
replacing it, the window-start question put on 2026-10-23 (options (a)-(c), none taken), after the explanation of the
rule as written (2026-10-26) and of $\hat\sigma_{\rm SS}(c)$ (2026-10-28, 09:41).

**Statement** (the assistant's reading of the quote; every point marked **[open]** is to be confirmed with the user
before Block 3 is changed).

(D-095.1) *Mean windows.* For each real cell $c$, each SS sweep $s$ of $c$ and each polarity $p \in \{{\rm dep},
{\rm hyp}\}$, with $K_{s,p}$ stored windows $x_k$, $k \in (s,p)$, on the common grid $t \in [-10, 200)$ ms from the
pulse onset,
$$\bar x_{s,p}(t) = \frac{1}{K_{s,p}}\sum_{k \in (s,p)} x_k(t) \qquad \text{for each } t \in [-10, 200)\ {\rm ms}, \tag{D-095.1}$$
the per-sample mean; the per-sample median across the $K_{s,p}$ windows is the user's alternative **[open]**. The
grouping is per polarity and per sweep ("per polarity and per sweep"); the quote also says "across the $K_p$
windows", the per-polarity count of the cell **[open: the assistant reads the last sentence as fixing the grouping
to $(s, p)$]**.

(D-095.2) *Window start.* On each $\bar x_{s,p}$ the pulse response's decay is modelled as an exponential, and
$t_{s,p}$ is the time from the onset at which that decay falls within $\kappa_{\rm tr}\hat\sigma_{\rm SS}(c)$ --
with $\hat\sigma_{\rm SS}(c)$ unchanged (the median over the cell's windows of the SD, ddof 1, of each window's
$[-10, 0)$ ms segment, as explained on 2026-10-28) and $\kappa_{\rm tr}$ = 0.1 still the provisional default of
D-051. "Statistically" **[open]**: the point value of the fitted decay, or the time at which a confidence bound of it
(for example a bootstrap over the $K_{s,p}$ windows) falls within the tolerance. $t_{\rm SS}(c)$ is formed from the
$t_{s,p}$ **[open: the assistant proposes their maximum, so that each cell keeps one window length, which the per-cell
medians of D-051 (i) and the binding function's $n_m$ need]**. No 5 ms bins and no odd part $\bar r_{\rm odd}$.

(D-095.3) *Subtraction.* Every enlarged window is cut to $[t_{\rm SS}(c), 200)$ ms and has the mean window of its own
sweep and polarity subtracted on the same range,
$$\tilde x_k(t) = x_k(t) - \bar x_{s,p}(t), \qquad k \in (s,p),\ t \in [t_{\rm SS}(c), 200)\ {\rm ms}, \tag{D-095.2}$$
which is (D-051.1) with the group $(s, p)$ in place of the cell's polarity $p$ ("just to be sure": it removes whatever
mean response remains). The fit's simulator applies the same grouping to its own simulated windows, so the operation
stays part of the estimator on both sides (D-051 (iv), D-033).

**Why [user; reasoning].** **[user]** as quoted: "which will settle everything". **[reasoning]** The rule as written
is noise-limited: under noise alone a 5 ms bin mean of $\bar r_{\rm odd}$ has an SD of 0.30, 0.24, 0.15, 0.11
$\hat\sigma_{\rm SS}$ at $K$ = 40, 63, 199, 348 against a threshold of 0.1 $\hat\sigma_{\rm SS}$, and the
last-exceedance rule sends 56-90 % of noise-only cells to the fallback (`check_findings_2026-10-23.py` A,
$\theta_{\rm cf}$) **[run]**; $\hat\sigma_{\rm SS}(c)$ is 87 % fast noise in expectation, while the bin means' noise
is the slow component's wander between windows (`check_sigma_ss.py`) **[run]**. A fitted decay is compared with the
tolerance as a smooth curve rather than as one bin's noisy mean, and a start where the mean response is already within
$\kappa_{\rm tr}\hat\sigma_{\rm SS}(c)$ bounds what the subtraction could let through, whatever the pulse-to-pulse
variation: under gains $a_k = 1 + cv\,z_k$ the subtraction leaves, in expectation, $\delta_k \approx cv^2(1 -
1/K_{s,p})$ times the response's demeaned mean square in the window added to each $\hat\sigma_k^2$ (verified exact in
$\hat\sigma^2$, the per-cell medians moving 0.57-0.64 of the first-order value; 0.46 per-cell SD at $cv$ = 0.3 for a
response of 2.7 $\hat\sigma_{\rm SS}$ at 80 ms, `check_jitter_leak.py`) **[run, 2026-10-26]**.

**What it implies for code and tables.**
- Block 3: `transient_start` is replaced by a function that builds (D-095.1), fits the decay per $(s, p)$ and returns
  the $t_{s,p}$, the fit parameters, $t_{\rm SS}(c)$ and flags; `subtract_mean_response` groups by sweep and polarity
  (`SSWindows.sweep_number` is there); `WindowConfig.transient_bin_ms` and `transient_floor_z` are retired, the fit's
  settings join it (model, fitted range, the "statistically" reading), `kappa_tr` stays (0.1, provisional),
  `t_ss_floor_ms` and `fallback_t_ms` wait for the open points. W2 is replaced by tests of the fit and of the crossing
  (a planted exponential recovered; the crossing within one sample of $\tau\ln(A/(\kappa_{\rm tr}\hat\sigma_{\rm
  SS}))$); W3 checks the zero mean per $(s, p)$.
- Block 4: `simulate_cell_windows` subtracts the mean of its simulated windows per $(s, p)$; F2's expectation becomes
  $\sigma_f^2(1 - 1/K_{s,p})$.
- The tilt (D-034.1) is untouched: $\hat V_c(h^*)$ uses same-sweep, same-polarity pairs, in which $\bar x_{s,p}$
  cancels exactly (D-051 (iii)).
- `Ih Fit/specs/SPEC.md` §2.2 (window geometry, transient test) is stale from this entry and is rewritten before the
  code (scientific-coding workflow); the rule's open points are asked first.
- Columns: `noise_ss_enl_t0_ms` keeps its name; per $(s, p)$ the fitted amplitude, time constant, baseline and
  $t_{s,p}$ are written.
- Roadmap allen-mtg-fit: N3's note (the options of 2026-10-23) is stale; changes to the roadmap wait for the user
  (D-082's hold of 2026-10-26, 16:48), except a new SPEC block, which enters with its commit (D-091).

**The assistant's comments, not yet answered [reasoning unless tagged].**
(a) *Where the start falls.* For a noiseless RC response of the SS train (37 alternating pulses of $\pm$200 pA $\times$
0.5 ms), $\kappa_{\rm tr}$ = 0.1 and $\hat\sigma_{\rm SS}$ = 0.0402 mV, read as the point value of a fitted single
exponential with a free baseline: $t_{\rm SS}$ = 99, 130, 154, 170, 186, 201, 229 ms for $\tau_m$ = 20, 28.1, 35, 40,
45, 50, 60 ms at 113 MOhm, and 112, 130, 146, 157 ms for $R_{\rm in}$ = 60, 113, 200, 300 MOhm at $\tau_m$ = 28.1 ms
(`check_d095_rc_start.py`, agreeing within 0.5 ms with $\tau_m\ln(A/(\kappa_{\rm tr}\hat\sigma_{\rm SS}))$, $A =
V_0/(1 + e^{-200.5\,{\rm ms}/\tau_m})$, $V_0 = Q R_{\rm in}/\tau_m$) **[run]**. The fixture's 28.1 ms is close to
the L3_exc run-B group medians, $R_m C_m$ = 26220 Ohm cm$^2$ $\times$ 1.062 uF/cm$^2$ = 27.8 ms **[KB, annual report
part 1 §6; arithmetic on medians, the median of per-cell $\tau_m$ not read]**. So a typical cell gets windows of
about 70 ms instead of the 120 ms of the 80 ms floor, and from $\tau_m$ of about 46 ms (113 MOhm) the start passes 190
ms (the fallback); a confidence bound moves it later still.
(b) *The decay's baseline.* Measured against the mean window's own pre-pulse level, the same response crosses 3 ms
later at 28.1 ms and never within the window for $\tau_m \ge$ 40 ms (113 MOhm): the pre-pulse segment holds the
previous, opposite-polarity pulse's tail, which stays in the window as a constant offset **[run]**. The baseline is
best fitted (or the train's tails modelled), not taken from $[-10, 0)$ ms.
(c) *The model.* For an RC membrane the train's window is exactly $b + A e^{-t/\tau_m}$ **[run]**; a real cell's is
not -- dendritic equalisation in the first milliseconds, the $I_h$ component later (the reason this fit exists) -- so
the fitted range and model decide the extrapolated tail **[not checked]**.
(d) *Per sweep.* A group holds about 10-19 windows in the archive's layouts (37-pulse trains, 19 dep and 18 hyp,
minus the QC drops) **[reasoning on the layout; per-cell counts not re-read]**, against 20-175 per polarity: each
subtracted window keeps $(1 - 1/K_{s,p})$ of its noise variance and the windows of a group correlate at $-1/(K_{s,p} -
1)$ (as D-051's P4 at $K_p$); mirrored by the simulator, so no bias, but the statistics are those of the subtracted
windows.
(e) *What remains of D-076 (ii).* Its 80 ms floor rests on the SQC selection leak (at most 0.09 per-cell SD from 80
ms), not on the transient; under (a) the new start is above it for every RC case with $\tau_m \ge$ 20 ms at 113 MOhm,
so it binds only for fast or small responses **[open: kept or dropped]**.

**What it does not decide [open].** Mean or median in (D-095.1); the reading of "statistically"; how $t_{\rm SS}(c)$
is formed from the $t_{s,p}$; the decay model, its fitted range and its baseline; whether D-076 (ii)'s 80 ms floor and
D-051 (i)'s fallback at 190 ms stay; $\kappa_{\rm tr}$'s final value (provisional since D-051); the flag when a fit
fails or a group has too few windows.

**Sources.** **[user]** 2026-10-28, 09:55. **[log]** D-034, D-051, D-076, the rows 2026-10-23 (later, 4) and
2026-10-26 (later, 3). **[src]** `noise_windows.transient_start` and `subtract_mean_response`, SPEC §2.2 (read
2026-10-26 and 2026-10-28, `sci/noise-fit` at `e75bdea`). **[run, sandbox]** `check_findings_2026-10-23.py` (A, D),
`check_jitter_leak.py` (2026-10-26), `check_sigma_ss.py` and `check_d095_rc_start.py` (2026-10-28), standalone, sent
to the user in `TEEG_window_start_checks_2026-10-28.zip`, not in the repository. **[KB]** annual report part 1 §6
(run-B medians).

**Status:** active; not implemented. Supersedes D-051 (iv)'s window-start test (the 5 ms bins of $\bar r_{\rm odd}$)
and the per-polarity grouping of (D-051.1) (now per sweep and polarity); supersedes D-076 (ii)'s $t_{\rm tr}(c)$ and
its cell-level mean window for the per-sweep statistics; leaves D-076 (ii)'s 80 ms floor and D-051 (i)'s fallback as
open points. The options (a)-(c) of 2026-10-23 are closed, none taken. [updated 2026-10-28 (later, 3): the roadmap's N3 and N4 carry its revise marks (`53b2713`); its checks are in the repository, `Ih Fit/dev/2026-10-28_window_start/` (D-097)] [updated 2026-10-28 (later, 9): open points closed by **D-103** -- the per-sample median window (not the mean), per sweep and polarity, the fitted decay's point value, the maximum over groups, one exponential with a free baseline on [5, 200) ms, the 80 ms floor kept and 80 ms the fallback; open: point 9 (a failed group) and the fallback's trigger] [updated 2026-10-28 (later, 10): the fallback's trigger (a window shorter than 50 ms) and point 9 (a failed group flagged and left out) closed by **D-104**]

---

## D-096 -- The flag faint leaves the selection S: every node whose fit is in S is corrected, whatever its optical depth and its dip; the flags of S enter the estimator signature (user)

**Date:** 2026-10-28, 09:58 (Europe/Rome), the diameter implementation chat. **ID note:** first numbered D-095 in the repository; that ID had been taken at 09:55 by the noise-fit chat (the SS window start), so this is D-096 -- the commit messages of `cf2f91d` and `095c5bf` keep the old number, the repository's text was renamed in `76365a3`. **Binds:** the selection $\mathcal S$
(Block 6, `SELECTION_FLAGS`, now in `config.py`), for phantoms and real nodes alike; the bias table (Block 14: which
replicates enter); the inversion and the fill (Block 7, `correct_nodes`, D5); the estimator and simulation signatures
(Block 1); D-094.3 (superseded); D-030 (extended from the optical depth to the dip); D-053.2's list of what stays in
$\mathcal S$; Block 10's `selection_flags` (D-094's switch keeps its meaning).

**Decision [user].** "Be aware that we have to correct ALL the nodes regadless the optical depth and the dip. So we need
to drop entirely the faint selection criteria for the bias table." Said in answer to the assistant's message of 09:17,
which had read D-094 as the calibration's selection only, with $\mathcal S$ keeping the flag ("If you meant S too,
tell me: that changes the estimator and needs a new table").

**Statement.**

(D-096.1) **`faint` is a label, not a criterion of $\mathcal S$.** The flag `faint` (a dip under
`measure.faint_min_dip_gl` = 6 gl, PROVISIONAL, on the fitted plane, D-061.3) no longer fails $\mathcal S$:
`SELECTION_FLAGS` = (`crossing`, `stack_edge`, `bbar_few`, `profile_nan`), for phantoms and real nodes alike, since
$\mathcal S$ is applied identically to both (D-024). The flag stays in the node's flags as a label, as `dark` since
D-030 [user].

(D-096.2) **What follows.** The table is fitted on phantom replicates of every dip depth. Every real node whose fit is in
$\mathcal S$ is inverted, whatever its $\hat\alpha$ (D-030) and its dip. A node whose fit fails $\mathcal S$ by its status
(no convergence, a bound, D-053.2), by its registration or by one of the remaining flags still goes to the fill (D5):
it has no $\hat d$ to correct, or a profile that is not one tube **[reasoning]**.

(D-096.3) **The rule of $\mathcal S$ is part of the estimator.** The flags of $\mathcal S$ enter the estimator signature
(`"selection": {"flags": [...]}`), so a table built, or a replicate row rendered, under another $\mathcal S$ is refused
(`check_estimator`; the merge's `sim_hash`). The hashes move:
- estimator `0508e7ff12f48728` -> `59678d4e2aedf2b6`;
- simulation `0ea5a6d2ec069c68` -> `52aa73e479eb77cf`;
- with the hybrid absorption, simulation `5e5b04a2f154e1c8` -> `7a1dff452223c9fa`.

This is the assistant's choice, to confirm; no production table existed.

(D-096.4) **What it supersedes.** D-094.3 (the assistant's reading that $\mathcal S$ kept the flag). D-094's calibration
switch `node_faint_flag_check` keeps its meaning: true puts the flag back among the calibration's reasons only (first
among the flags, as before).

**Why [user; reasoning; KB].** **[user]** as quoted: D-030's argument -- every real node gets its own diameter -- carried
from the optical depth to the dip. **[KB, D-030]** $\mathcal S$ is applied identically to phantoms and real nodes, so if
faint real nodes are corrected, the table must hold faint phantoms: it describes the nodes it corrects. **[KB,
mathematics document 2026-10-04, Eqs. 8-11]** The flag was a contrast criterion (a dip against the camera noise), not the
faint-limit assumption: the per-node fit uses the full Beer-Lambert transmittance $e^{-\mu\ell}$, so keeping shallow
dips breaks no assumption of the fit. **[reasoning]** What changes is noise: their fits scatter more, and the table
fitted with them carries that scatter into its SE (D-071's by-draw SE).

**What it implies.** Code: `config.SELECTION_FLAGS` (moved from `analysis/phantoms.py`, which re-exports it) and the
estimator signature's `"selection"`; `calibration.selection_flags` and `selection_reasons` (the flag `faint` added, first
among the flags, only with `node_faint_flag_check`); `invert.correct_nodes` unchanged in code (it reads
`reject_reasons`). Tests (config: the signature's keys, the flags of S without faint and dark, another S changes both hashes; invert: a faint node inverted, a crossing one filled, a table built under the old S refused; calibration: S without faint, the switch puts faint first; hybrid: the pinned hashes). Commit `cf2f91d`. Colab: no cell changes. New runs of Cells 4b and 13 write the new
`in_S`; the pilot CSV of 2026-10-07 keeps its old `in_S` column, and Cell 7 recomputes its candidates from the flags.
davinci: the table job's rows carry the new `sim_hash`; nothing has run there.

**What it does not decide [open].**
- (a) `faint_min_dip_gl` (6 gl, PROVISIONAL) now only labels: whether it keeps a role (a diagnostic, a stratum of the
  table's report) is open.
- (b) How the table's SE and the inversion behave at low contrast, where the faint replicates now enter **[reasoning]**:
  to be read on the table's coverage report once a table is built.
- (c) Whether the flags of $\mathcal S$ belong in the estimator signature (D-096.3, the assistant's choice).

**Sources.** **[user]** 2026-10-28, 09:58. **[src]** `cf2f91d` on `sci/diameter-pipeline`. **[run, sandbox]** all 36 smoke suites;
five mutants killed (faint back in S, the selection not signed, the switch ignored or always on, faint put last). **[KB]** D-024, D-030, D-053.2, D-061.3, D-094; the mathematics document of 2026-10-04, §3.1-3.3.

**Status:** active; implemented 2026-10-28 (`cf2f91d`), smoke-tested; no table built.

---

## D-097 -- The assistant pushes its own commits, and every document is also committed to the GitHub repository so that any chat can read it (user)

**Date:** 2026-10-28, 10:24 (Europe/Rome), the noise-fit chat. **Binds:** how every chat of the project delivers
its work -- code (the last step of the hpc-git-delivery workflow: the assistant's own push instead of patches for the
user to apply) and documents (each document a chat writes is committed to the repository too); the project's DOCUMENT
UPKEEP (a repository copy beside each write to project knowledge); the roadmaps' "push when the user agrees"
(subproject-roadmap skill); the noise-fit handoffs' rule "the assistant's pushes are refused: deliver patches"
(2026-10-19, 2026-10-23).

**Decision [user].** "Always push by youself, every docuemnt also must be in the github repository so any chat can
consult it. Update also the roadmap. in the handoff write also that I have to reply to the other open question from
before". Said after the noise-fit chat had delivered Blocks 3-4 as patches (`noise_fit_patches_2026-10-28.zip`), its
own earlier pushes having been refused, and after project knowledge had refused a new document (the noise-fit handoff
of 2026-10-28: project knowledge full).

**Statement.**

(D-097.1) *Push.* The assistant commits and pushes its own work to `Leonardodm00/Towards-EEG` in the turn the work is
done, once the verification gates of the scientific-coding and hpc-git-delivery skills have passed; a patch for the
user to apply is no longer the delivery route [user]. Commits stay authored as the user (the clone's local git
identity), with the session's trailers, as before.

(D-097.2) *Documents.* Every document a chat writes is also committed to the repository, so that any chat can read it
[user]. Where -- the assistant's reading [open, (a)]:
- each project-knowledge document is copied byte for byte to `docs/project/` on `main`, at its project path
  (`docs/project/claude/<name>` for `claude/<name>`, `docs/project/<name>` for a root document), in the same turn as
  every write to it; the project copy stays the source of truth while it exists; a document that project knowledge
  refuses (full since 2026-10-28) lives only in the repository, and its own changelog says so;
- a workstream's handoffs, study documents and check packages go beside its code, as the noise fit's already did:
  `<workstream>/docs/` for handoffs, `<workstream>/dev/<date>_<topic>/` for a study document or a check package with
  its scripts and logs, regenerable bulk data left out with the command that regenerates it;
- `docs/project/README.md` says what is copied and what is not.

(D-097.3) *What is not copied without the user's word* [reasoning; DOCUMENT UPKEEP]. The files the user uploaded
(`phase1_function_reference.md`, `phase1_conceptual_guide.md`, `phase2_patch.md`, `Bayesian Optimisation and
Likelihood Profile.md`, the three .docx files); the project's PDFs are published papers and are never copied. The
repository is public (GitHub, read 2026-10-28) [run], so every copied document is public: no credentials and no
personal data in a document's text -- the commit identity stays in git's configuration.

**Applied (2026-10-28, the noise-fit chat).** Pushed to `main`: first the five noise-fit patches (`0dfe8bf`,
`65c659b`, `76577d2` Block 3, `a95e618` Block 4, `033e3bf`) and the two roadmap patches (`8d8d602`, `4930ba1`); then
`53b2713` (roadmap allen-mtg-fit: D-095's notes and revise marks on N3 and N4, the tool updated to the skill's
version), `d2ffcb4` (the tilt study document in `Ih Fit/dev/2026-10-26_tilt_sigma_b/`; the window-start checks in
`Ih Fit/dev/2026-10-28_window_start/`; `docs/project/` with its README, the noise-model topic document and the staged
plan; the Ih Fit README and changelog) and `cd32dad` (the roadmap rendered; the revisions' `on` key quoted). Roadmap
page written (version 44). The copies of this log and of the living handoff, and the new handoff
`Ih Fit/docs/TEEG_noise_fit_handoff_2026-10-28.md`, follow in the commit after this write (a log cannot name the
commit that carries it).

**Why [user; reasoning].** **[user]** "so any chat can consult it". **[reasoning]** Project knowledge is full (a new
document refused on 2026-10-28), so a document written now reaches the next chat only through the repository; a chat
with the repository attached reads it there at any size, and the documents sit at the same address as the code they
describe.

**What it does not decide [open].**
(a) The copies' place and rule (`docs/project/` on `main`; the project copy first, the repository copy in the same
turn) -- the assistant's reading.
(b) Whether the other chats' documents are copied now, all at once, or each when its chat next writes it (the README
lists them as not yet copied).
(c) Whether the user's uploads are copied.
(d) A workstream on a branch of its own (`sci/diameter-pipeline`): its project documents copied to `main`'s
`docs/project/` (the assistant's proposal), its handoffs and study documents on its own branch beside its code.

**Sources.** **[user]** 2026-10-28, 10:24. **[run]** git (the pushes above, 2026-10-28); the session's GitHub
repository listing (visibility public, 2026-10-28). **[log]** D-082, D-091; the noise-fit handoffs (pushes refused on
2026-10-19 and 2026-10-23).

**Status:** active; applied by the noise-fit chat on 2026-10-28 (commits above); open points (a)-(d).

---

## D-098 -- After the Colab run of S3's Cells 11 to 12 (user): `fetch_swc` never takes the marker file; the xy projection chosen by its type; S3a's coverage recomputes $\mathcal S$; Cell 7 draws each candidate's background patch; the set-aside cells that look aligned run with `--include-set-aside`

**Date:** 2026-10-28, 15:01 (Europe/Rome), the user's answer to the assistant's report on the Colab run of Cells 11,
11b and 12 (the diameter implementation chat). **Binds:** `allen_image_io.fetch_swc` (SPEC Block 0); the projection
rule of Block 16.2 (a) and the report of 16.3 (a) (`SurveyConfig.projection_xy_type`); S3a's pooled coverage (Block
17.3; `phantoms.row_reject_reasons`); Cell 7's outputs (Block 10.F (5); `calibration.scan_cell`'s `on_node`); how four
set-aside cells are used (D-089.1); D-089.2's rule, superseded and kept as the fallback.

**Decision [user].** The user quoted the assistant's suggestion -- "My suggestion is to do (a), (b) and (d) now, each
specced, smoke-tested and pushed. Then do (c) with `--include-set-aside` once you've looked at 571464200 at full
resolution." -- and answered "Let's do this." Then: "Moreover for the cell 7 slightly change it. as you evaluate each
candidate also plot the image of the patch that will be evaluated for the Background (the one that triggers the
non_isolated flag)". The proposals (a)-(e) were the assistant's reply of 11:30 to the user's logs and figures of 11:16.

**Statement.**

(D-098.1) *The SWC.* `fetch_swc` never takes a well_known_file whose name contains "marker"; among the others the
reconstruction's type id (303941301) first, then a `.swc` path [user: proposal (a)]. Four of the 51 selected cells
(569095789, 508279351, 596036453, 525018757) list their marker file, named `*_marker_m.swc`, before the reconstruction,
and failed Cell 12 on its comma-separated lines **[the user's log, read]**.

(D-098.2) *The xy projection.* Block 16.2 (a) takes the SubImage of type 'MinimumIntensityProjection - xy'
(`projection_xy_type`; without case and outer spaces; the darkest of several of that type); the darkest of all only
when no candidate has that type -- D-089.2's rule, now the fallback [user: proposal (b)]. Why: on the pilot the
minimum's xy is the planes' pixelwise minimum (r* 0.9941 at zero offset), every one of the 52 specimens listed has
one, and the 'MaximumIntensityProjection - xy' of the 4 specimens that have one is an inverted, contrast-stretched
image, the darker of the two, which the darkest rule took -- misaligning 508305938 **[the user's logs and figures;
the inversion the assistant's reading]**. The alignment record names the rule (`projection.rule`), the overlay the
projection, Cell 11b's report adds `chosen` beside `darkest`.

(D-098.3) *S in the coverage.* S3a's pooled coverage reads $\mathcal S$ from each per-node row's `fit_status`,
`reg_verdict` and `flags` under the flags of $\mathcal S$ of the code that pools, not from the stored `in_S`, which
carries the rule of the code that wrote the CSV; the stored count is reported beside it [user: proposal (d)]. So the
Cell 13 run started on 2026-10-28 counts its faint nodes in $\mathcal S$ (D-096) if its clone predates D-096
(`estimator_hash` `0508e7ff12f48728` in `pilot_summary_<id>.json`), without measuring again; the measurement itself is
the same before and after D-093 to D-096.

(D-098.4) *Cell 7's isolation figure.* As `calibrate_kernel.py real` evaluates each candidate it draws the patch its
background is fitted on and the traced neighbours whose predicted shadows decide `not_isolated` [user]. The figure's
form is the assistant's reading [PROVISIONAL]: the +-25 um patch at the pilot's sharpest plane $k^*$ (one crop more per
candidate not scanned); panel (a) the traced segments -- the node's own stretch blue, a neighbour orange when its
predicted shadow $\max_k D_{q,k}$ over the stage's planes exceeds `isolation_depth_gl` (0.03 gl), grey below; panel (b)
the far pixels and the fitted background plane; the verdict and the deepest shadow in the title; overview pages of 36
thumbnails; `--isolation-figures DIR`, `--no-isolation-figures`. The numbers drawn are those the selection used
(`isolation_breakdown`, of which `isolation_depth` is the deepest; the far mask and plane of `patch_planes` over the
whole stage).

(D-098.5) *(c), the set-aside cells that look aligned.* 576084692 (R 1.66), 596887423 (1.89) and 611598198 (1.14) are
run with `--include-set-aside` on their automatic alignment, and 571464200 (1.79) after the user's look at full
resolution; D-089.1's threshold stays 2 [user]. The run file's Cell 13b holds the command (`ACCEPTED`).

**Why [user; data; reasoning].** **[user]** the agreement and the request. **[data inspected: the user's Colab logs and
figures of 2026-10-28, 11:16]** the marker-file failures, the projections' types and the check on the pilot, the
darkest rule's four picks, the set-aside cells' shifts and overlays. **[reasoning]** (D-098.3): a stored `in_S` depends
on the code that wrote it, the stored columns do not. (D-098.4): 46 of the 57 candidates the user pasted on 2026-10-27
were `not_isolated`; the figure shows which traced neighbour makes each so -- including, as the synthetic cell shows,
a segment joining the node's stretch to the soma or to another stretch at a branch point, which counts as a neighbour
**[run, sandbox]**.

**What it implies for code and tables.** Implemented and pushed to `sci/diameter-pipeline`: `c595bc0` (D-098.1),
`4ef6d8e` (D-098.2), `cdba78a` (D-098.3), `28790c3` (D-098.4), `44e95e3` (the run file's Cells 12b and 13b, the
notebooks' Cell 7, runbook v33). No signature changes (estimator `59678d4e2aedf2b6`, simulation `52aa73e479eb77cf`).
Colab: Cells 0 and 1 again (the pull), after the running Cell 13 has finished or been stopped; then Cell 12b
(508305938 re-aligned), Cell 13 (it aligns the four cells now that their SWC is the reconstruction, and measures them),
Cell 13b, and Cell 7 for the figures.

**What it does not decide [open].** (a) D-089.1's threshold (unchanged; the four cells by the flag). (b) The isolation
rule itself -- 0.03 gl, and whether a segment joining the stretch to the soma or to another stretch should count as a
neighbour -- which the figures are for judging. (c) 558211203, kept with a shift of (170, -104) px after a server warning
at level 4: its overlay is to be looked at before its nodes are pooled. (d) Whether 571464200 joins (D-098.5).

**Sources.** **[user]** 2026-10-28, 15:01. **[data]** the user's logs and figures of 2026-10-28, 11:16 (Cells 11, 11b,
12). **[run, sandbox]** `test_smoke_swc_fetch.py`, `test_smoke_cell_alignment.py`, `test_smoke_projection_check.py`,
`test_smoke_s3a.py`, `test_smoke_isolation_view.py` and the suites they touch; 5 + 6 + 7 + 8 mutants, each killed
(SPEC Blocks 0, 16.2, 17, 10).

**Status:** active; (D-098.1)-(D-098.4) implemented and pushed 2026-10-28, smoke-tested, not yet run on Allen's data;
(D-098.5) waits for the user's look at 571464200 and a Colab run. [updated 2026-10-28, D-099: (D-098.5)'s Cells 12b and 13b are knobs of the davinci array job (`DIAM_CELLS`, `DIAM_FORCE`, `DIAM_INCLUDE_SET_ASIDE`)] [updated 2026-10-28 (later, 11): (D-098.4)'s figure, under the stripe of D-101, gains a panel (c) -- the stripe of every plane of the stage, the far fit's planes emphasised -- and the overview's labels the count $n/N$ of the fit's planes with a usable stripe -- D-105.1]

---

## D-099 -- S3's per-cell survey runs on davinci, one PBS array task per selected cell; Colab runs every cell on the pilot only (user)

**Date:** 2026-10-28, 17:07 (Europe/Rome), the diameter implementation chat, while Colab's Cell 13 of the run started
that morning was still measuring. **Binds:** where the per-cell steps of S3 run (Blocks 16.2, 17.1-17.2, 18.1-18.3,
19.3 over the selection; SPEC Block 16.4); Colab's Cells 11-15 of both notebooks (`notebooks/phase2_colab.*`,
`notebooks/s3_run_2026-10-27.*`) and the runbook; `scripts/survey_cells.py run` (`--task-index`, `--cache-per-cell`;
Block 16.2 (g)); `scripts/cell_pool.py pick` (Block 16.1 (e)); the roadmap's S3a-S3c (`needs_hpc: true`, the jobs
`diam_survey`, `diam_survey_pool`); D-085's framing of S3 as "the real-data programme in Colab"; D-098.5's Cells 12b
and 13b.

**Decision [user].** "Blocks 13 and 14 are too slow. Let's change approach. Everything now on colab will run on a single
pilot cell while the real runs will be on the DaVinci HPC. Adjust all the scripts and the pipeline."

**Statement.**

(D-099.1) *The survey on davinci.* The per-cell survey of S3 -- the alignment (Block 16.2), the node sample (17), the
radius, registration and camera inputs (18) and the calibration scans (19.3) of every selected cell -- runs on davinci
as PBS jobs, with the same scripts Colab ran: `scripts/pbs/survey_prepare.pbs` (one job: Cell 11's pool, acquisition
check and selection over the whole pool, Cell 11b's projection listing with the pilot's test), the array
`scripts/pbs/survey_cells.pbs` (task $i$ runs row $i$ of `selection.csv`, 0-based, through `survey_cells.py run
--task-index i --cache-per-cell`; one task per selected cell, 51 on 2026-10-28; resumable by the step markers) and
`scripts/pbs/survey_pool.pbs` (the status table, the pooled coverage, inputs and calibration, and one archive
`results_<RUN_TAG>.tgz` of what is read off the cluster) [user: "the real runs will be on the DaVinci HPC"; the split
into three jobs and one task per cell the assistant's]. The runner gains `task_index` (one row of the selection,
exclusive with `--cells`, refused outside the selection) and `source_for` / `--cache-per-cell` (each cell's Allen
source over `<cache>/<specimen>/`, so no two tasks write one file).

(D-099.2) *Colab on the pilot.* Every Colab cell runs on the pilot 529878215 only [user: "Everything now on colab will
run on a single pilot cell"]: Cells 11-15 work in `survey_pilot/` (`survey/` keeps the run of 2026-10-28 over 51 cells
untouched), Cell 11 builds the one-cell selection with `cell_pool.py pick --cells 529878215` (Block 16.1 (e); the pilot
is not in the stratified selection) and checks its acquisition on that one-row file, Cell 13 measures 3 nodes per bin
of Allen's $2r$ (`nodes_per_bin` 3 in `survey_config.json`; at most 18 nodes, about 10 min at the 30 s per node of
2026-10-28; PROVISIONAL, the assistant's size) so that the chain and the pools are exercised end to end in minutes;
the numbers that decide (the mu range, $d_{\max}$, the jitter, the camera, the kernel) are the davinci run's. The
davinci array keeps the default 20 per bin. Cells 12b and 13b of the run file become knobs of the array job
(`DIAM_CELLS=508305938,DIAM_FORCE=align`; `DIAM_CELLS=...,DIAM_INCLUDE_SET_ASIDE=1`; a single job without `-J`).

(D-099.3) *The reading of "Blocks 13 and 14".* The notebook's Cells 13 and 14 (S3a and S3b), the slow ones (about an
hour per cell of 120 nodes, measured on 2026-10-28), not SPEC Blocks 13 and 14 [the assistant's reading, stated to the
user; D-098's entry records the times].

(D-099.4) *The provisional choices [assistant, PROVISIONAL].* (i) The compute nodes reach `api.brain-map.org` through
`module load proxy` (the H01 reference, section 9 **[KB]**; the same line the H01 campaign uses); every job that needs
the network runs one RMA query first and exits 6 if it fails, so a blocked network costs a minute, not a queue wait.
(ii) The layout under the archive root of D-012, as the table job's: `.../Human Neurons Fitting/diameter_survey/<RUN_TAG>/`
and `.../Human Neurons Fitting/allen_cache/<specimen>/`. (iii) One task per cell, 12 h walltime, 1 CPU and 4 GB per
task (Colab's hour per cell with margin). (iv) The array marks the roadmap's S3a and the pool job S3c (hpc markers).

**Why [user; data; reasoning].** **[user]** the decision and its reason, "too slow". **[data inspected: the user's Colab
run of 2026-10-28]** Cell 12 took 29.7 min for 51 cells; Cell 13's first cells about an hour each, so one to two days
for 42 cells in a Colab session that disconnects -- against 51 cells in parallel on davinci, an hour or two of wall
time once queued. **[reasoning]** per-cell crop caches and one task per cell remove every shared write; the probe
before the work turns the one unverified assumption (the proxy's reach) into a one-minute failure; the pilot in Colab
keeps a place where the user sees each step's output and figures with the code of the day, which the cluster's logs
do not give.

**What it implies for code and tables.** On `sci/diameter-pipeline`, pushed 2026-10-28 evening: `5230474` the
runner's options and `pick` (SPEC 16.1 (e), 16.2 (g)); `d4559ba` Block 16.4 with `scripts/pbs/survey_common.sh`, the
three job scripts, `tests/smoke/test_smoke_survey_pbs.py` (4 pass, 3 skip; 7 mutants killed) and the SPEC; `ede27be`
the notebooks' Cells 11-15 on the pilot and runbook v34 with the davinci section; `01a15d6` the roadmap's S3 title,
S3a-S3c `needs_hpc` and jobs. No signature changes (estimator `59678d4e2aedf2b6`, simulation `52aa73e479eb77cf`).
Order on davinci: the byte checks, `DIAM_DRYRUN=1`, the prepare job, one probe task (`DIAM_STEPS=align`, no
`-J`), the array, the pool; the archive read off the cluster.

**What it does not decide [open].** (a) Whether the Colab Cell 13 still running on 2026-10-28 is stopped or left to
finish: its results in `survey/` stay readable either way, and the davinci run starts from its own `selection.csv`
(the same seed, so the same 51 cells). (b) Whether Drive's crop cache is copied to davinci: not planned; the tasks fetch
their crops again (one cache per cell). (c) The proxy's reach to Allen's server, the time per cell and the queue's
behaviour with 51 concurrent fetchers: the first run's. (d) `nodes_per_bin` 3 for the Colab pilot. (e) The set-aside
cells and 571464200 (D-098.5): the same question, now asked of the array's knobs.

**Sources.** **[user]** 2026-10-28, 17:07. **[KB]** the H01 reference's `module load proxy` (section 9), the davinci
conventions of the paths references (the conda hook, `#PBS -k eo`, the reserved names). **[data]** the user's Colab
logs of 2026-10-28. **[run, sandbox]** `test_smoke_survey_pbs.py`, `test_smoke_cell_pool.py`,
`test_smoke_cell_alignment.py` and the S3 suites; the mutants of Block 16.4.

**Status:** active; implemented and pushed 2026-10-28; smoke-tested against stubs; nothing yet run on davinci.

---

## D-100 -- For the cells with two SS sweeps, the twins' tilt SD is taken both per cell and as one cohort value; partial pooling is not taken (user)

**Date:** 2026-10-28, about 18:37 (Europe/Rome), the noise-fit chat. **Binds:** D-034's per-cell $\sigma_b(c)$
(D-034.1) for the 14 L3_exc cells with two SS sweeps; the tilt SD a twin's generator receives (Block 5,
`synthetic_ground_truth`; the tilt $b$ of law (D-017.1)); a cohort step after the 18 per-cell fits (Block 6, the driver
and its PBS jobs); Block 3's pair counts (`noise_cell_stats.window_mean_semivariogram`) and Block 4's tilt output
(`noise_fit.TiltResult`, the column `noise_sigma_b_mV_per_s`); the Stage-7 gate's report (`run_ih_recovery.py`,
`ih_recovery_report.py`; D-011); `Ih Fit/specs/SPEC.md` §8 (the tilt question of 2026-10-23); the options (a)-(c) of the
tilt study document's §3.9; roadmap T8.

**Decision [user].** "Let's do A and B for the 2 sweep cells." Said in answer to the assistant's account, after the
explanation of the tilt and of the semivariogram (2026-10-28), of what was left for the user to decide on
$\sigma_b(c)$: (a) per cell, as D-034 reads it; (b) one cohort value for all cells; (c) partial pooling across cells,
DerSimonian-Laird on $\log\hat\sigma_b(c)^2$ -- the assistant's recommendation (D-034's note of 2026-10-26; the
noise-fit handoff of 2026-10-28, question 1). Option (c) is not taken.

**Statement** (the assistant's reading of the quote; every point marked **[open]** is to be confirmed with the user
before Blocks 5 and 6 are specified).

*Notation.* $\mathcal R$ is the set of the 18 real L3_exc cells and $\mathcal R_2 \subset \mathcal R$ the 14 with two
SS sweeps (of the other four, one has 6 and three have 10) **[data: the archive's sweep counts, as the tilt study
document's §1 reads them]**; $\mathcal R$ is not D-011's synthetic cohort $\mathcal C$. For a cell $c$, $S(c)$ is its
number of SS sweeps, $N_s(c) \in \mathbb N$ the number of same-sweep, same-polarity window-mean pairs $h^*$ apart in its
sweep $s$ ($h^*$ = 4 onset periods, 0.802 s; D-034.1) and $N(c) = \sum_s N_s(c)$.

(D-100.1) *Two tilt variants for the 2-sweep cells.* For each $c \in \mathcal R_2$ the twin is generated, and the
gate's per-cell errors (D-011 (iv)) computed, once under each of
$$\tilde\sigma_b^{(a)}(c) = \hat\sigma_b(c), \qquad \tilde\sigma_b^{(b)}(c) = \hat\sigma_{b,\rm coh} \qquad ({\rm mV/s}), \tag{D-100.1}$$
with $\hat\sigma_b(c) \in \mathbb R_{\ge 0}$ the cell's (D-034.1) clipped at 0, as Block 4 computes it
(`tilt_from_semivariogram`) **[src]**, and $\hat\sigma_{b,\rm coh} \in \mathbb R_{\ge 0}$ one value for the whole of
$\mathcal R$ (D-100.2). Both are numbers computed from the recorded sweeps (computed level), each put in place of the
SD of the law the cell's tilts are drawn from (analytic level); every twin sweep then draws its own tilt
$b \sim \mathcal N(0, \tilde\sigma_b(c)^2)$, independently across sweeps (D-017.1). The cells of
$\mathcal R \setminus \mathcal R_2$ keep variant (a) only, as D-034 says ("for the 2 sweep cells"). *Variant*, not
*arm*: an arm is already a configuration of the I_h fit (D-005, D-011).

(D-100.2) *The cohort value*, in the assistant's form -- the one check04 of the tilt study compared
(`tilt_common.pool_cohort`) **[src]**:
$$\hat\sigma_{b,\rm coh}^2 = \max\Bigl\{0,\ \frac{\sum_{c \in \mathcal R}\nu_c\,\hat\sigma_b(c)^2_{\rm u}}{\sum_{c \in \mathcal R}\nu_c}\Bigr\}, \qquad \hat\sigma_b(c)^2_{\rm u} = \frac{2\,\bigl[\hat V_c(h^*) - V(h^* \mid \hat\theta_c, 0)\bigr]}{h^{*2}}, \qquad \nu_c = S_{\rm eff}(c) = \frac{N(c)^2}{\sum_s N_s(c)^2}, \tag{D-100.2}$$
with $\hat\sigma_b(c)^2_{\rm u} \in \mathbb R$ ((mV/s)$^2$) the unclipped (D-034.1) and $\nu_c \in [1, S(c)]$ the
effective number of tilt draws behind $\hat V_c(h^*)$ (the study document's eq. (12)). The unclipped values are
averaged because each is unbiased for $\sigma_b(c)^2$ over the law of the tilts when the floor $V(h^* \mid \theta, 0)$
is taken at the true law (study document §3.4; check03 (b): mean ratio 0.997-1.027 in every layout simulated)
**[run]**, so the mean stays unbiased for a common $\sigma_b^2$; the weights $\nu_c$ are then inverse-variance weights,
since the variance of $\hat\sigma_b(c)^2_{\rm u}$ is close to $2\sigma_b^4/\nu_c$ when the tilt draws dominate it (95-98
% of it in every layout, study document §3.7) **[run; reasoning]**. With one common $\sigma_b$ = 0.1 mV/s this form
puts $\tilde\sigma_b^{(b)}/\sigma_b$ within 0.85-1.14 (5-95 %) **[run: check04, $\omega$ = 0]**. **[open: pooled over
all of $\mathcal R$, as here; over the four multi-sweep cells only; or over $\mathcal R_2$ only.]**

(D-100.3) *Where the variants act.* Only in the value the twins' generator receives. Inside each real cell's own fit
(Block 4: $\hat\sigma_b(c)$ alternated with $\hat\theta_c$, the tilt of the SS simulator) the cell keeps its own
$\hat\sigma_b(c)$, as now; a $\sigma_b$ error of the two-draw size (0 or 0.17 mV/s in the simulator for a cell at 0.1)
moves $\hat u$ by at most 0.18 of its SD (linearised at $\theta_{\rm cf}$, the fixture layout; SPEC §8) **[run,
2026-10-23]**, so the fitted law would change little under variant (b) **[open: the assistant's reading]**.

(D-100.4) *What the gate reports.* For each cell of $\mathcal R_2$, its per-cell errors under both variants, side by
side and labelled; for the other cells, under (a). **[open: whether one variant is the default -- the verdict -- and
the other a labelled comparison, or a 2-sweep cell passes only if it passes under both.]**

**Why [user; run; reasoning].** **[user]** as quoted. **[run: check04 of the tilt study, re-run 2026-10-28 with
identical output; 18 cells with the L3_exc layouts, $\theta_{\rm cf}$ in every cell, $\sigma_b(c) =
0.1\,\mathrm e^{\omega z_c}$ mV/s with $z_c \sim \mathcal N(0, 1)$ and $\omega$ the log-SD of $\sigma_b$ across cells,
(T1) at the true law, 4000 cohorts]** For a 2-sweep cell the two variants fail in different ways. Per cell errs low:
median $\tilde\sigma_b/\sigma_b$ 0.83 at every $\omega$ from 0 to 1, a zero tilt in 4.6-10.4 % of cells, more than a
factor 2 off in 25.1, 25.4, 25.7, 30.5 % at $\omega$ = 0, 0.25, 0.5, 1. The cohort value is right when the cells share
one $\sigma_b$ and errs high once they differ: median ratio 0.99, 1.05, 1.22, 2.00, more than a factor 2 off in 0.0,
1.4, 22.3, 59.4 %. At $\sigma_b$ = 0.1 mV/s the tilt is a twin's larger baseline error at the Long Square step's lags
(step-mean SD $0.6375\,\tilde\sigma_b(c)$ s = 0.064 mV, against 0.031 mV from the 37 ms component), and a twin cleaner
than its cell makes the gate optimistic for that cell (study document §3.8) **[doc]**; the gate run under both variants
shows, cell by cell, how much its verdict rests on the twin's tilt **[reasoning]**. *Not waiting for D-095* **[run:
`check07_tss_window.py`, 2026-10-28; the two 2-sweep reference layouts ($K$ = 40, 63), $\theta_{\rm cf}$, 2000 cells
per layout, the same traces cut at both starts]**: a window start of 130 ms instead of 80 ms -- D-095's rule on the RC
fixture at $\kappa_{\rm tr}$ = 0.1 -- raises a 2-sweep cell's detection limit (the 95th percentile of $\hat\sigma_b$
under $\sigma_b$ = 0) from 0.0478 to 0.0564 mV/s ($K$ = 40) and from 0.0381 to 0.0445 mV/s ($K$ = 63), and moves the
2.5/50/97.5 % points of $\hat\sigma_b/\sigma_b$ at $\sigma_b$ = 0.1 mV/s by at most 0.015; the variants and their
comparison do not depend on D-095's open points.

**What it implies for code and tables.**
- Blocks 3-4: (D-100.2) needs each cell's $\hat\sigma_b(c)^2_{\rm u}$ and $\nu_c$. `TiltResult` holds `sigma_b`
  (clipped), `V_model`, `V_hat` and `h_star`, from which $\hat\sigma_b(c)^2_{\rm u}$ follows; `window_mean_semivariogram`
  returns $\hat V_c(h^*)$ and the total pair count $N(c)$ only, not the $N_s(c)$ **[src]**. Proposal: the per-sweep
  counts returned too, and the noise table gains `noise_sigma_b2_unclipped` ((mV/s)$^2$) and `noise_tilt_s_eff`.
- A cohort step after the 18 per-cell fits, which run as independent jobs (Block 6): it reads their rows, computes
  (D-100.2) and writes $\hat\sigma_{b,\rm coh}$ with the cells and weights it used -- a pure function in the analysis
  layer, smoke-tested on a planted cohort, run by a PBS job that depends on the per-cell jobs **[proposal]**.
- Block 5 (the twins' generator): the tilt SD is an input per variant; a twin's manifest row and every output carry
  its variant (`tilt_variant` = `per_cell` | `cohort`), so the two are never pooled by accident; the two variants of
  one cell share the manifest row and every unit draw except the tilt's scale (Block 2's `simulate_traces` scales one
  fixed `xi_tilt` by `sigma_b`), so their difference is the tilt's effect alone **[proposal; src: Block 2's API]**.
- No variant is the default for a 2-sweep cell until (b) below is answered: the gate report shows both, labelled; the
  multi-sweep cells' default stays (a), D-034 **[reasoning: a default is a decision]**.
- The Stage-7 recovery fits of the 14 cells double (D-083: on davinci) **[reasoning]**.
- Block 4's fit and Block 4b (D-076 (i), $\sigma_s$ per sweep): unchanged (D-100.3); Block 4b's clip of the DL
  $\hat\tau^2$ still stands (D-076).
- SPEC §8's tilt question of 2026-10-23: answered for the 2-sweep cells by this entry (marked there). Roadmap T8 (the
  tilt's theory) and N4's note: left for the user (D-082's hold) **[log]**.

**What it does not decide [open].**
(a) The reading of "do A and B": two variants of twins per 2-sweep cell (D-100.1), or both values computed and
reported and the one the twins receive chosen later, on the real cells' $\hat\sigma_b(c)$ once Block 3 has run on the
archive.
(b) The default (D-100.4): one variant as the verdict and the other a labelled comparison, or a pass required under
both.
(c) The cohort value's cells and form (D-100.2).
(d) Whether the four multi-sweep cells also get variant (b), for a like-for-like comparison across $\mathcal R$.
(e) Zero tilts in variant (a): a cell with $\hat\sigma_b(c)^2_{\rm u} \le 0$ (`tilt_clipped`) gets twins without a
tilt -- kept, as D-034 is implemented (the assistant's proposal: variant (b) covers that case), or floored.
(f) Whether a cell's own fit is re-run with the cohort value under variant (b) (D-100.3).

**Sources.** **[user]** 2026-10-28, about 18:37. **[log]** D-005, D-011, D-017, D-034 (with its note of 2026-10-26),
D-076, D-082, D-083, D-095; the noise-fit handoff of 2026-10-28 (question 1). **[doc]** the tilt study document,
`Passive Features/HPC script/Ih Fit/dev/2026-10-26_tilt_sigma_b/` (§1, §3.4, §3.7-3.9). **[src]**
`noise_fit.tilt_from_semivariogram` and `TiltResult`, `noise_cell_stats.window_mean_semivariogram`,
`noise_law.simulate_traces`, the study's `tilt_common.pool_cohort` and `s_eff` (`main` at `b82c189`). **[run,
sandbox]** check03 and check04 of the tilt study (re-run 2026-10-28, identical); `check07_tss_window.py` (2026-10-28,
`Ih Fit/dev/2026-10-28_tilt_explanation/`); `check_findings_2026-10-23.py` (C), through SPEC §8.

**Status:** active; not implemented; its reading open, (a)-(f). Extends D-034 for the 2-sweep cells -- its (D-034.1)
stays as variant (a) and for the other cells; nothing superseded. Answers, for the 2-sweep cells, the question put to
the user on 2026-10-23, 2026-10-26 and 2026-10-28 (SPEC §8); option (c), the assistant's recommendation, not taken. **[superseded 2026-10-28 (later, 9): withdrawn -- the assistant's misreading of "A and B" (the user, 19:35: "I think you have misunderstood"); replaced by **D-102**: each twin sweep carries its real sweep's realised tilt, no statistic of the tilt computed.]**

---

## D-101 -- The calibration nodes' flatness per set (far 15 deg, near 10 deg), and the background fitted on a stripe along the measuring line that stops at the first predicted shadow, the own cylinder removed (user)

**Date:** 2026-10-28, 18:17 (Europe/Rome) and the dialog's four answers (about 18:40), the diameter implementation
chat, on Cell 7's isolation overview of the pilot 529878215. **Binds:** the calibration criteria of Block 10.I
((iii) flatness, (vi) isolation) for both sets; the background's support of every scanned plane (Blocks 10.P and
15.1) -- SPEC Block 10.S; `CalibrationConfig.node_phi_max_deg` (now the near set's, 10), `node_phi_max_deg_far` (15),
`background_support`, `stripe_half_widths_um`, `stripe_min_pixels`, `stripe_min_side_um`; Cell 7 of both notebooks
and S3c's per-cell scans (`survey_cells.py`'s calibration step, the davinci array's); the isolation figures of
D-098.4. Supersedes D-078.1's $3\sigma$ far mask and D-081.1's node-level verdict as the default (both kept under
`background_support = "square"`); amends D-081.3's 5 deg.

**Decision [user].** 18:17: "This is the output of the candidates. Only 2 far field. We need to select more points, S
is too strict. and also change approach for the far field nodes background estimation. What about using a stripe that
extends up to where it is clear in the isolation sense. So for near field nodes find such a stripe that maximizes the
number of pixels of background in isolation." The dialog's answers: flatness "Far <= 15 while keep the near lower than
10"; the clear-pixel rule, the stripe's shape and the acceptance as recommended, "What I would do also is to remove the
fitted node's cylinder from the stripe so that it doesn't bias the calculation".

**Statement.**

(D-101.1) *Flatness per set.* A far-set node ($0.6 < D_{\rm A} \le 0.8$ um) is flat when $\varphi_j \le 15$ deg, a
near-set node when $\varphi_j \le 10$ deg [user; the near bound read as 10 deg from "lower than 10" -- the assistant's
reading, to correct if 5 was meant]. Was 5 deg for both (D-081.3).

(D-101.2) *The clear pixel.* In plane $k$ a pixel $p$ of the patch is shaded by a traced segment $q$ when
${\rm dist}(p, q) \le \rho_\tau(q, k)$, $\rho_\tau$ the lateral distance at which $q$'s predicted shadow (10.I's
$D_{q,k}$, with the pilot's $\bar B_j$ and $\hat\mu_j$) falls to $\tau$ = `isolation_depth_gl` = 0.03 gl; the soma as
a disc of its radius. This replaces the $3\sigma$ margin of 10.P's far mask [user, the recommended option]: one rule,
reaching farther than $3\sigma$ for a dark neighbour and less for a faint one.

(D-101.3) *The stripe.* In the measuring line's frame ($v$ along the line, $u$ along the dendrite), for each
half-width $W \in$ `stripe_half_widths_um` = (1, 2, 4, 8, 12) um the band $|u| \le W$ is walked outward from the node on
each side of the line, past the columns the node's own cylinder shades, to the first column a neighbour (or the own
branch again) shades or the patch's edge; the $W$ with most clear pixels wins; one stripe per plane of the stage
(the far planes' shadows are wider) [user, the recommended option]. The plane's background is 10.P's trimmed
least-squares plane fitted on the stripe.

(D-101.4) *Acceptance and the own cylinder.* A plane is usable with at least `stripe_min_pixels` = 500 clear pixels
in its stripe and at least `stripe_min_side_um` = 3 um of clear run on each side (the profile's half-length); the node's
own fitted cylinder -- radius $\max(\hat d_j/2, r_{\rm Allen})$ with the pilot's fitted $\hat d_j$, Allen's radius
when $\hat d_j$ is not finite -- is removed from the stripe like any neighbour's shadow, so that its own tails bias no
background [user: "remove the fitted node's cylinder from the stripe"]. The node's verdict before any fetch is the
stripe at the plane nearest the pilot's depth: `not_isolated:too_few` or `not_isolated:side_short`.

(D-101.5) *What stays.* $\tau$ = 0.03 gl (D-081.1's value); the $\pm 25$ um patch (D-078.1); the planar background
and its trimming (10.P); every other criterion of 10.I; the square rule and the plain `not_isolated` under
`background_support = "square"`. Without a finite $\bar B_j$ and $\hat\mu_j$ (a phantom's node) the margins fall back
to 10.P's geometric $r + 3\sigma$ (the assistant's, so that the phantom checks keep their meaning).

**Why [user; data; reasoning].** **[user]** the request and the four answers. **[data inspected: `pilot_529878215.csv`,
393 nodes, read off Drive 2026-10-28]** of the 31 nodes with Allen's $2r$ in (0.6, 0.8] um (two stretches, 677-691
and 7694-7709) 28 are in $\mathcal S$ and 26 fail flatness alone at 5 deg; `bbar_few` is set on none (unmasked
fraction 0.65-0.82); at 10 / 15 / 20 / 30 deg the far set would hold 6 / 7 / 14 / 24 before isolation, the near set
138 / 183 / 213 / 256 (68 at 5 deg). **[the user's figures, read]** the two far survivors, 7702 and 689, are shaded by
the soma (76.7 gl) and a junction (36.6 gl); near candidates with shadows of 0.04-0.3 gl -- far below the 5.9 gl
median background noise SD of the pilot's nodes -- were rejected by the any-shadow rule. **[reasoning]** the measuring
line is across the dendrite, so tilt does not foreshorten the width; it enters through the $\pm 0.5$ um along-branch
average (a depth smear of 0.17 um at 20 deg, under one plane spacing) and the plane iteration, so 15 deg for the far
set, whose planes sit 1.4-4.5 um from focus, costs little; the stripe keeps the background local to the fitted line
(a plane over 50 um is already an approximation to a blotchy background) and turns a neighbour from a veto into a
boundary. The quadrature of $D_{q,k}$ (16 Gauss-Legendre nodes) ripples at the $10^{-3}$ level when the blur is narrow
against the tube, which broke the monotone root search on the first attempt: the rule now takes up to 512 nodes and
the root is bracketed by a scan first **[run, sandbox]**.

**What it implies for code and tables.** On `sci/diameter-pipeline`: `03c43b2` (the block) and `6a261e4` (the notebooks, runbook v35), pushed 2026-10-28 evening. SPEC Block 10.S (and 10.I (iii), (vi) amended); `calibration_scene.shadow_depths`,
`shadow_margin`, `shadow_masks`, `stripe_support`, `patch_frame`; `calibration.support_masks`, `patch_planes` (six-tuple:
the stripe infos), `stripe_reasons`, `criteria_reasons` per set; `PlaneScan.stripe` and its JSON; the isolation figures'
panel (b) and the overview's orange rule; `calibrate_kernel.py real`'s summary line; `tests/smoke/test_smoke_stripe.py`
(6 pass, 1 skip; 7 of 8 mutants killed, one equivalent); the suites `calibration`, `isolation_view` (the square checks
under `"square"`), `far_field`, `s3c` adapted and passing; runbook v35; the notebooks' Cell 7 texts. No signature
changes: the hashes stay estimator `59678d4e2aedf2b6`, simulation `52aa73e479eb77cf`. Colab: Cells 0, 1 and 7 again
(the figures show the stripes). davinci: the array's calibration step uses the stripe once the clone is pulled.

**What it does not decide [open].** (a) The near bound (10 deg, the reading above). (b) The $W$ grid, 500 pixels, 3 um,
the own radius rule and the one-plane verdict -- PROVISIONAL, to judge on Cell 7's figures. (c) How many far-set nodes
the stripe admits on the pilot and across the 51 cells (Cell 7 again; the davinci run). (d) Whether $\tau$ = 0.03 gl
stays once the stripe makes it a boundary rather than a veto (a larger $\tau$ was offered and not taken). (e) The
near-set bound and the far-set bound on the per-cell stratified samples of S3c.

**Sources.** **[user]** 2026-10-28, 18:17 and the dialog. **[data]** `pilot_529878215.csv` (Drive, 2026-10-07), read in
full; the user's two overview pages of 2026-10-28, 18:17. **[run, sandbox]** `test_smoke_stripe.py` and the calibration
suites; the mutants of Block 10.S. **[reasoning]** the tilt and the locality arguments above.

**Status:** active; implemented and pushed 2026-10-28; smoke-tested; not yet run on Allen's data. [updated 2026-10-28 (later, 11), **D-105**: rechecked -- the integration tester's report on `e32d03b` (`specs/test_reports/2026-10-28_1955_incremental.md`): the node's verdict shaded the soma at the in-focus blur while the scan used the stage's (F1, fixed at `39ae156`); the own cylinder was centred on the trace while the walk starts on the fitted line (F2, fixed as the union of both -- the assistant's reading of "the fitted node's cylinder", to confirm). On the pilot's own reconstruction **[run, sandbox: `checks/d105_2026-10-28/`]** open point (c) has an answer before Cell 7 runs: as built, 21 of the 138 near candidates have a usable stripe at $k^*$ and none of the 7 far candidates has one in any of the far fit's planes, because the predicted shadows of out-of-focus dendrites exceed 0.03 gl over the nodes themselves (89 near nodes lie under at least one with $\sigma_{\rm r} \ge$ 2 um). Open point (d) is now the question: $\tau$ = 0.3 / 1 gl keeps 92 / 117 near nodes with a residual on the line of median 0.24 / 0.35 gl, but no far node under 3.5 gl -- D-105.]

---

## D-102 -- Each twin sweep carries the realised tilt of its own real sweep; no statistic of the tilt is computed; D-100 withdrawn as a misreading (user)

**Date:** 2026-10-28, 19:35 (Europe/Rome), the noise-fit chat. **Binds:** the tilt of law (L1) = (D-017.1) in every
twin (Block 5, `synthetic_ground_truth`): D-034's draw $b \sim \mathcal N(0, \sigma_b(c)^2)$, the generator's
`drift_sigma_mV` and the calibration (D-034.1); **D-100** (withdrawn); the per-cell fit's tilt (Block 4: the
alternation of $\hat\sigma_b(c)$ with $\hat\theta_c$, `tilt_from_semivariogram`, the simulator's `sigma_b`); Block 3's
window-mean semivariogram (`window_mean_semivariogram`, used only by (D-034.1)); D-073 (each twin sweep the twin of one
real sweep); the column `noise_sigma_b_mV_per_s`; `Ih Fit/specs/SPEC.md` §2.4 and §8; the tilt study document's options
(a)-(c) (§3.9); roadmap T8.

**Decision [user].** "I think you have misuderstood. Let's do in a simpler way. For each cell draw out per each sweep a
tilt. the twin will recieve the very same tilt as in teh real cell so that no statistics has to be computed on the
tilt." Said about D-100, which had read the user's "Let's do A and B for the 2 sweep cells" (about 18:37) as options (a)
and (b) of the tilt study's §3.9 -- two variants of twins, one with the per-cell SD and one with a cohort SD. That
reading was the assistant's and is withdrawn.

**Statement** (the assistant's reading of the quote; every point marked **[open]** is to be confirmed with the user
before code is written).

(D-102.1) *The tilt is copied, not drawn.* For each real cell $c$ -- all 18 L3_exc cells -- and each of its sweeps $s$,
the realised tilt $b_{c,s}$ (mV/s; one draw of the law, fixed in the recording) is estimated by $\hat b_{c,s}$ (computed
level; D-102.2, D-102.3), and the twin of sweep $s$ (D-073) carries that estimate: its injected noise is
$$e_j = e^{\rm fast}_j + e^{\rm slow}_j + a + \hat b_{c,s}\,(t_j - \bar t_s) \qquad ({\rm mV}), \tag{D-102.1}$$
with $e^{\rm fast}$, $e^{\rm slow}$ the fast AR(1) and the slow OU of the cell's law $\theta_c$ and $a$ the offset, as in
(L1); $t_j$ (s) the twin sweep's sample times and $\bar t_s$ (s) its mid-time. $\hat b_{c,s}$ is one number per sweep,
the same in every replicate of the twin: the tilt is no longer a random variable of the twins' generator, and
$\sigma_b(c)$ is not computed for the twins -- neither (D-034.1), nor D-100's cohort value, nor partial pooling [user:
"no statistics has to be computed on the tilt"].

(D-102.2) *Reading $\hat b_{c,s}$ on an SS sweep* -- the assistant's proposal: the ordinary least-squares slope of the
sweep's window means on their centre times, with one intercept per polarity,
$$\bigl(\hat b_{c,s}, \hat\alpha_{\rm dep}, \hat\alpha_{\rm hyp}\bigr) = \arg\min_{b,\,\alpha_{\rm dep},\,\alpha_{\rm hyp}}\ \sum_{k \in s}\Bigl(\langle\tilde x_k\rangle - \alpha_{p(k)} - b\,\bigl(t^{\rm c}_k - \bar t_s\bigr)\Bigr)^2, \tag{D-102.2}$$
with $\langle\tilde x_k\rangle$ (mV) the average of window $k$ over $[t_{\rm SS}(c), 200)$ ms after the subtraction of its
group's median window (D-103.1), $t^{\rm c}_k$ (s) the window's centre time in the sweep and $p(k)$ its polarity. In the
tilt study's check03 (f) -- $\theta_{\rm cf}$, the 2-, 6- and 10-sweep reference layouts, $\sigma_b$ = 0.1 mV/s,
$t_{\rm SS}$ = 80 ms, the mean-window subtraction -- this recovered each realised tilt with an RMS error of 0.0027-0.0036
mV/s, 3-4 % of $\sigma_b$ **[run]**. With D-103's later starts and the median subtraction it is not checked **[open]**.

(D-102.3) *Reading $\hat b_{c,s}$ on a Long Square sweep* **[open]**. The twins' LS sweeps -- the 1 s steps the I_h fit
reads -- need their real sweeps' tilts too, and a 265 ms pre-window cannot give one: pre-window slopes would be too
large by more than an order of magnitude (D-017 note 4; the living handoff §4.6) **[log]**. The assistant's proposal:
$\hat b_{c,s} = \Delta_{{\rm end},c,s}/(6.89\ {\rm s})$, with $\Delta_{{\rm end},c,s}$ (mV) = `delta_end_mV` of
`ls_baseline_qc.py`, the end-window mean minus the pre-window mean, centres 6.89 s apart **[src; log]** -- under the
hypothesis that the step leaves no after-effect in the end window, 5.5-6 s after it (the living handoff §4.2's rebound
caveat; the uncorrected hint of a rise with step size in 526785799, §4.6) **[log]**. Alternatives: LS twins without a
tilt; a tilt taken from the cell's SS sweeps.

(D-102.4) *Inside the per-cell noise fit* **[open]**. The assistant's proposal: the binding function's simulator gives
each simulated sweep the same $\hat b_{c,s}$ as its real sweep, in every replicate, so the fit's simulation mirrors the
data (as it does for the subtraction), and (T1) with its alternation (SPEC §2.3) leaves the fit. The alternative:
detrend each real sweep by $\hat b_{c,s}$ and simulate no tilt.

**Why [user; run; reasoning].** **[user]** "so that no statistics has to be computed on the tilt". **[run: the tilt
study, check03 (b) and (f)]** The realised tilts are observed almost exactly (3-4 % of $\sigma_b$ in the configuration
run), while their SD is not: with two SS sweeps, $\hat\sigma_b(c)/\sigma_b$ spans 0.00-2.04 (95 %, $K$ = 40), the
two-draw limit of the data. Copying the tilts uses the well-determined quantity and never needs the ill-determined one.
**[reasoning]** The gate then tests the I_h fit on each cell's own drift -- the one the real fit of that cell faces --
rather than on a typical drift for it; the twin's tilt is one realisation, not resampled, so the gate's result for a
cell is conditional on it. The law of D-034 -- the variance at seconds carried by a per-sweep tilt -- is unchanged; what
it does not carry (a non-linear wander at seconds, if any) the twins do not carry either.

**What it implies for code and tables.**
- Block 5: the generator takes per-sweep tilts ($\hat b_{c,s}$, mV/s, keyed by sweep number) in place of $\sigma_b(c)$;
  `drift_sigma_mV` and the draw of `xi_tilt` are not used for the twins.
- Block 3: a function returning $\hat b_{c,s}$ and its SE for each SS sweep (D-102.2); written per sweep (the per-sweep
  table of D-073), e.g. `noise_tilt_mV_per_s`.
- Block 4: if (D-102.4) is confirmed, (T1) (`tilt_from_semivariogram`), the window-mean semivariogram's use and the
  alternation leave the fit; whether `noise_sigma_b_mV_per_s` stays as a report-only diagnostic is open.
- D-100's implications (two variants, the cohort step, the variant labels, the exports for the cohort value) are void.
- SPEC §2.4 and §8 are marked; they are rewritten with the block that implements (D-102.2).

**What it does not decide [open].** (a) The LS sweeps' tilts (D-102.3). (b) The fit's simulator (D-102.4). (c) The
precision of (D-102.2) with D-103's starts and the median subtraction (not checked). (d) Whether (T1) stays as a
report-only diagnostic.

**Sources.** **[user]** 2026-10-28, 19:35. **[log]** D-017 (with note 4), D-034, D-073, D-100; the living handoff
§4.2, §4.6. **[doc]** the tilt study document (`Ih Fit/dev/2026-10-26_tilt_sigma_b/`), §3.7 and §3.9. **[run, sandbox]**
check03 (b), (f) of the tilt study (re-run 2026-10-28, same output). **[src]** `ls_baseline_qc.py` (`delta_end_mV`),
`noise_fit.tilt_from_semivariogram` (`main` at `fefdee9`).

**Status:** active; not implemented; open (a)-(d). Supersedes **D-100** (a misreading) and, for the twins, D-034's draw
of $b$ from $\mathcal N(0, \sigma_b(c)^2)$ with its calibration (D-034.1); D-034's law stands. **[updated 2026-10-29: open points (a), (b) and (d) closed by **D-107** -- a Long Square sweep's tilt is the line through a start-window mean and an end-window mean; every simulated sweep of the fit carries its real sweep's tilt, so the tilt SD, (T1) and the alternation leave the fit; (c) stays open.]** **[updated 2026-10-29 (later): (c) answered by the check noted in **D-107**: the median subtraction leaves the SS reading unchanged; the later starts raise its RMS error by 15-23 %.]**

---

## D-103 -- Confirmations closing D-095's open points: the median window per sweep and polarity, the fitted decay's point value, the maximum over groups, one exponential with a free baseline on [5, 200) ms, 80 ms as floor and as fallback (user)

**Date:** 2026-10-28, 19:35 (Europe/Rome), the noise-fit chat. **Binds:** D-095's open points; D-076 (ii)'s 80 ms floor;
D-051 (i)'s fallback at 190 ms (`fallback_t_ms`, the flag `ss_enlarged_none`); Block 3 (`noise_windows.transient_start`,
`subtract_mean_response`, `WindowConfig`); Block 4's simulator (`noise_fit.simulate_cell_windows`); the smoke checks
W2, W3 and F2; `Ih Fit/specs/SPEC.md` §2.2; the column `noise_ss_enl_t0_ms`.

**Decision [user].** The answers to the nine open points of D-095, as put to the user on 2026-10-28: "1) median 2) yes
3) I would say the decay's point value 4) Maximum over groups 5) perfect as you suggest 6) keep this aslo as the
fallback but flag 7) 80 ms is the fall back 8) Can you briefly show me the formula". Point 9 (a group whose fit fails or
has too few windows) was not answered.

**Statement.** For each real cell $c$, each SS sweep $s$ and each polarity $p \in \{{\rm dep}, {\rm hyp}\}$, with the
$K_{s,p}$ stored windows $x_k(t)$ (mV), $k \in (s,p)$, on $t \in [-10, 200)$ ms from the pulse onset:

(D-103.1) *The median window* [user: 1, 2]:
$$m_{s,p}(t) = \operatorname*{med}_{k \in (s,p)}\, x_k(t) \qquad \text{for each } t \in [-10, 200)\ {\rm ms}, \tag{D-103.1}$$
the per-sample median over the group's windows (computed level). It replaces D-095's mean $\bar x_{s,p}(t)$ in both of
its uses: the start (D-103.2) and the subtraction, $\tilde x_k(t) = x_k(t) - m_{s,p}(t)$ for $t \in [t_{\rm SS}(c),
200)$ ms [the assistant's reading: D-095 uses one window for both]. Both are per $(s, p)$ [user: "yes"]. The fit's
simulator applies the same per-sample median to its own windows, per $(s, p)$, so the operation stays part of the
estimator on both sides (D-051 (iv), D-033).

(D-103.2) *The start of a group* [user: 3, 5]. A single exponential with a free baseline is fitted to $m_{s,p}(t)$ on
$t \in [5, 200)$ ms ("perfect as you suggest"; the 5 ms is the assistant's "about 5 ms" made exact, PROVISIONAL),
$$m_{s,p}(t) \approx \hat B_{s,p} + \hat A_{s,p}\,e^{-t/\hat\tau_{s,p}},$$
with $\hat B_{s,p}$, $\hat A_{s,p}$ (mV) and $\hat\tau_{s,p} > 0$ (ms) the fitted values, and the group's start is the
point value at which the fitted decay falls within the tolerance [user: "the decay's point value"], the solution of
$\lvert\hat A_{s,p}\rvert\,e^{-t/\hat\tau_{s,p}} = \kappa_{\rm tr}\,\hat\sigma_{\rm SS}(c)$:
$$t_{s,p} = \hat\tau_{s,p}\,\ln\frac{\lvert\hat A_{s,p}\rvert}{\kappa_{\rm tr}\,\hat\sigma_{\rm SS}(c)} \quad \text{if } \lvert\hat A_{s,p}\rvert > \kappa_{\rm tr}\,\hat\sigma_{\rm SS}(c), \qquad t_{s,p} = 0 \ \text{otherwise}, \tag{D-103.2}$$
with $\hat\sigma_{\rm SS}(c)$ (mV) unchanged (D-095.2) and $\kappa_{\rm tr} > 0$ dimensionless, 0.1, provisional since
D-051.

(D-103.3) *The start of the cell* [user: 4, 6, 7]:
$$t_{\rm SS}(c) = \max\Bigl\{80\ {\rm ms},\ \max_{(s,p)}\, t_{s,p}\Bigr\}, \tag{D-103.3}$$
the maximum over the cell's valid groups, with D-076 (ii)'s 80 ms floor kept [user: "keep this"]; 80 ms is also the
fallback [user: "80 ms is the fall back"], replacing D-051 (i)'s 190 ms, and is flagged [user: "but flag"]. Flags (the
assistant's names): `ss_start_floor` when the floor binds, `ss_start_fallback` when the fallback is used (replacing
`ss_enlarged_none`). **[open: the fallback's trigger -- the assistant's proposal: $\max_{(s,p)} t_{s,p} \ge$ 190 ms
(under 10 ms of window left, D-051 (i)'s trigger) or no valid group.]**

(D-103.4) *What $\kappa_{\rm tr}$ does* (point 8: the formula shown to the user) **[run: `check_kappa_tr_start.py`,
2026-10-28; the noiseless RC test cell of `check_d095_rc_start.py`, 113 MOhm, $\tau_m$ = 28.143 ms, $\hat\sigma_{\rm
SS}$ = 0.0402 mV; P1-P3 PASS]**: by (D-103.2), halving $\kappa_{\rm tr}$ delays every group's start by
$\hat\tau_{s,p}\ln 2$ (19.51 ms at 28.1 ms); the test cell starts at 110.3, 129.8 and 149.3 ms for $\kappa_{\rm tr}$ =
0.2, 0.1 and 0.05, and its start reaches 190 ms from $\tau_m$ = 61.4, 46.5 and 38.1 ms. $\kappa_{\rm tr}$ stays 0.1,
provisional.

**Why [user; run; reasoning].** **[user]** as quoted. **[reasoning]** A point value needs no resampling layer between
the fit and the start. The 80 ms fallback keeps a full window for every cell instead of the 10 ms of D-051 (i): what an
early start lets through after the subtraction is the pulse-to-pulse variation of the response, which D-095's Why
bounds for the mean subtraction (`check_jitter_leak.py`, 2026-10-26) **[run]**; for the median subtraction it is not
checked.

**What it implies for code and tables.**
- Block 3: `transient_start` replaced by the decay fit per $(s, p)$ on the median window (`scipy.optimize.curve_fit`,
  as in `check_d095_rc_start.py`), returning $\hat A_{s,p}$, $\hat\tau_{s,p}$, $\hat B_{s,p}$, $t_{s,p}$,
  $t_{\rm SS}(c)$ and the flags; `subtract_mean_response` subtracts the per-$(s, p)$ median window (`numpy.median`
  along the window axis); `WindowConfig`: `transient_bin_ms` and `transient_floor_z` retired, `fit_start_ms` = 5
  (PROVISIONAL), `t_ss_floor_ms` = 80, `fallback_t_ms` = 80 (was 190), `kappa_tr` = 0.1.
- Block 4: `simulate_cell_windows` subtracts the per-$(s, p)$ median of its simulated windows; F2's expectation is no
  longer $\sigma_f^2(1 - 1/K_{s,p})$ but that of a median-subtracted window, taken by Monte Carlo in the smoke test
  [proposal].
- Smoke: W2 replaced (a planted exponential recovered; the start within one sample of (D-103.2); the floor, the
  fallback and their flags); W3: zero median per $(s, p)$ after the subtraction.
- Columns: `noise_ss_enl_t0_ms` = $t_{\rm SS}(c)$; per $(s, p)$: $\hat A_{s,p}$, $\hat\tau_{s,p}$, $\hat B_{s,p}$,
  $t_{s,p}$; the flags.
- SPEC §2.2 rewritten for D-095 with these answers before the code (scientific-coding workflow); D-102's per-sweep
  tilts are read on the windows after this subtraction (D-102.2).

**What it does not decide [open].** (a) Point 9: a group whose fit fails, returns $\hat\tau_{s,p} \le 0$ or has too
few windows -- the assistant's proposal: flagged and dropped from the maximum; the minimum $K_{s,p}$ not set. (b) The
fallback's trigger (D-103.3). (c) $\kappa_{\rm tr}$'s final value. (d) The fit's starting values and bounds; the 5 ms
start (PROVISIONAL).

**Sources.** **[user]** 2026-10-28, 19:35. **[log]** D-033, D-051, D-076, D-095. **[run, sandbox]**
`check_d095_rc_start.py` (re-run 2026-10-28, identical) and `check_kappa_tr_start.py` (2026-10-28), both in
`Ih Fit/dev/2026-10-28_window_start/`; `check_jitter_leak.py` (2026-10-26).

**Status:** active; not implemented. Closes D-095's open points except (a)-(d); supersedes D-095's mean window (now the
median), D-051 (i)'s fallback at 190 ms (now 80 ms, flagged) and its flag `ss_enlarged_none`; keeps D-076 (ii)'s 80 ms
floor. **[updated 2026-10-28 (later, 10): (a) and (b) closed by **D-104** -- a failed group or one with fewer than 5 windows (PROVISIONAL) is flagged and left out; the fallback applies when the window would be shorter than 50 ms, or with no valid group.]**

---

## D-104 -- The SS window-start fallback applies when the window would be shorter than 50 ms; a group whose fit fails or has too few windows is flagged and left out (user)

**Date:** 2026-10-28, 19:53 (Europe/Rome), the noise-fit chat. **Binds:** D-103's open points (a) and (b); (D-103.3);
Block 3's window start (`WindowConfig.fallback_t_ms`, a new `min_window_ms` and `min_windows_per_group`); the flag
`ss_start_fallback` and a per-group flag; the smoke check W2; the roadmap allen-mtg-fit (D-082's hold).

**Decision [user].** "1) When the fallback applies.: When the window has less than 50 ms size 2) Flag and leave it out",
the answers to the two points D-103 left open: the fallback's trigger, and point 9 (a group whose fit fails or has too
few windows). With them: "Finally update the roadmap with the changes discussed here".

**Statement.**

(D-104.1) *The trigger* [user]. With $t^\star(c) = \max\{80\ {\rm ms},\ \max_{(s,p)\ {\rm valid}} t_{s,p}\}$ the start of
(D-103.3),
$$t_{\rm SS}(c) = \begin{cases} t^\star(c) & \text{if } 200\ {\rm ms} - t^\star(c) \ge 50\ {\rm ms},\\ 80\ {\rm ms},\ \text{flagged } \texttt{ss\_start\_fallback} & \text{otherwise,} \end{cases} \tag{D-104.1}$$
so the fallback applies when the window $[t_{\rm SS}(c), 200)$ ms would be shorter than 50 ms, a start later than 150 ms;
and also when no group is valid (D-104.2) [the assistant's reading: the maximum then has no member].

(D-104.2) *A failed group* [user: "Flag and leave it out"]. A group $(s,p)$ is left out of the maximum, and flagged
(`ss_group_dropped:<sweep>:<polarity>:<reason>`, the assistant's name), when its fit does not converge, returns a
non-finite or non-positive $\hat\tau_{s,p}$, or when $K_{s,p}$ is below `min_windows_per_group` = 5 (the assistant's
value, PROVISIONAL).

(D-104.3) *Where the fallback starts to bite* **[run: `check_fallback_window.py`, 2026-10-28; the noiseless RC test
cell of `check_d095_rc_start.py`, $\hat\sigma_{\rm SS}$ = 0.0402 mV; P1 PASS, re-run identical]**: at $\kappa_{\rm tr}$
= 0.1 the start passes 150 ms from $\tau_m$ = 34.0 ms at 113 MOhm, or from $R_{\rm in}$ = 234 MOhm at $\tau_m$ = 28.1 ms;
at $\kappa_{\rm tr}$ = 0.05, from 28.4 ms or 117 MOhm; at 0.2, from 43.2 ms or 467 MOhm. The L3_exc run-B group medians
give $R_m C_m$ = 27.8 ms **[KB: annual report part 1 §6; arithmetic on medians, the per-cell $\tau_m$ not read]**: at
$\kappa_{\rm tr}$ = 0.1 a cell somewhat slower, or of higher input resistance, than that falls back to 80 ms
**[reasoning]**.

(D-104.4) *The roadmap* [user]. Updated at the user's word, D-082's hold lifted for this update: notes for D-102 to
D-104 on N3, N4, N5 and T8; revise marks D-103 and D-102 on N3 and N4, D-102 on N5, and on T8 from N5; no status
changed (`a945a7a`).

**Why [user; run].** **[user]** as quoted. **[run]** (D-104.3) shows where the 50 ms limit falls for the test cell:
the number to keep in mind when $\kappa_{\rm tr}$ is settled, since a smaller $\kappa_{\rm tr}$ sends more cells to the
fallback.

**What it implies for code and tables.** `WindowConfig`: `min_window_ms` = 50 [user], `fallback_t_ms` = 80,
`t_ss_floor_ms` = 80, `min_windows_per_group` = 5 (PROVISIONAL); the start computed on the real windows only (the
simulator cuts its windows at the real cell's $t_{\rm SS}(c)$). W2 adds: a planted decay whose start passes 150 ms
falls back to 80 ms with the flag, one at 149 ms keeps a 51 ms window; a group with a failed fit or too few windows is
left out of the maximum with its flag. The flags go to the noise table's `noise_flags`.

**What it does not decide [open].** (a) Whether a dropped group's windows also leave the statistics -- the assistant's
proposal: yes when $K_{s,p}$ < 5 (a group of one window is identically zero after its own median is subtracted), no
when only its fit failed; the simulator then leaves out the same windows. (b) The value of `min_windows_per_group`
(5, PROVISIONAL). (c) $\kappa_{\rm tr}$ (0.1, provisional since D-051).

**Sources.** **[user]** 2026-10-28, 19:53. **[log]** D-051, D-076, D-082, D-095, D-103. **[run, sandbox]**
`check_fallback_window.py` (2026-10-28, `Ih Fit/dev/2026-10-28_window_start/`). **[KB]** annual report part 1 §6 (run-B
medians). **[src]** the roadmap commit `a945a7a`.

**Status:** active; not implemented. Closes D-103's open points (a) and (b), with (a) above left; with D-103, D-095's
open points are answered except $\kappa_{\rm tr}$'s final value. **[updated 2026-10-29: (a) closed by **D-106** -- a group with fewer than 5 windows (PROVISIONAL) leaves the noise statistics too; a group dropped only for a failed fit keeps its windows.]**

---

## D-105 -- The stripe used in every plane is drawn, the stripe's settings are exposed in Cell 7, and D-101's implementation rechecked (user)

**Date:** 2026-10-28, 19:39 (Europe/Rome), the diameter implementation chat, on the report of D-101's
implementation. **Binds:** the isolation figure of D-098.4 (SPEC Block 10.F (5)): panel (c), the overview's labels,
`calibrate_kernel.py real`'s lines `calibration knobs: ...` and `far-set candidates with a usable stripe ...`; Cell 7
of both notebooks (`CAL_KNOBS`, `OUT/calibration/calibration_config.json` passed as `--config-json` to `real`, `fit`,
`far-fit` and `figures`), Cell 15 (`survey_pilot/survey_config_s3c.json`, `REDO`), the davinci survey's
`DIAM_CONFIG_JSON`; SPEC Block 10.S as amended by the recheck. Amends D-098.4's figure; changes no rule of D-101.

**Decision [user].** Quoting the report's sentence on the figures, then: "given the far field nodes highlight the
stripe used, expose also the parameters to shape the stripe's features. Also recheck the previous implementation".

**Statement.**

(D-105.1) *The stripe used, plane by plane.* The isolation view carries the stripe of every plane of the stage as
`patch_planes` computed it, and the figure draws it as panel (c): above, each plane's clear pixels against
`stripe_min_pixels`; below, its runs along the measuring line beyond the own cylinder, one bar per side as wide as the
plane's $W$, dark where the plane's stripe is usable and orange where not, a tick where a side reaches
`stripe_min_side_um`, the planes the set's fit does not use faint -- the far fit uses $|k - k^*| \ge$
`far_min_k_offset` = 5, the dip's area $|k - k^*|$ = 4-6. The overview's label ends with $n/N$, the fit's planes with a
usable stripe; `real` prints how many far candidates have at least `far_min_planes_per_node` = 6 of them. Nothing in
the selection changes [the assistant's reading of "highlight the stripe used": for a far node the stripes the far
fit uses are those of planes 5-16 from $k^*$, while panel (b) shows $k^*$'s].

(D-105.2) *The knobs.* Cell 7 begins with `CAL_KNOBS`, the configuration's existing fields that shape the stripe --
`background_support`, `isolation_depth_gl` ($\tau$), `stripe_half_widths_um`, `stripe_min_pixels`,
`stripe_min_side_um`, `patch_half_um` -- and the two flatness bounds, at their defaults; the cell writes them as a
`calibration` section to `OUT/calibration/calibration_config.json` and passes that file to the four steps, and `real`
prints the values it ran with. Cell 15 merges the section into its own file; on davinci the same file goes as
`DIAM_CONFIG_JSON` to the array and the pool job [the set of knobs is the assistant's: every existing field that
shapes the stripe, no new one; a smoke check fails if the notebooks' values leave the defaults].

(D-105.3) *The recheck.* An integration tester run on `e32d03b` (PASS WITH ISSUES,
`specs/test_reports/2026-10-28_1955_incremental.md`) and a run of the masks on the pilot's own reconstruction.
Fixed at `39ae156`: F1 (MAJOR; the verdict now shades the soma at the stage's blur, so it equals the scan's stripe at
$k^*$), F5 (a help text); F2 as the assistant reads D-101.4 -- the own cylinder is the trace's own segments and their
copy moved onto the pilot's fitted centre; F4 -- the margin search stays custom, now vectorized over (segment, plane)
pairs, the reason stated (SciPy's vectorized root finder needs SciPy >= 1.15). S1-S3 and S5-S7 written into the SPEC.
The triage column of the report holds these as proposals awaiting the user.

**Why [user; run; reasoning].** **[user]** the request. **[run, sandbox: `checks/d105_2026-10-28/`, the pilot's SWC
(8496 segments) and the pilot table's centres, $k^*$, $\bar B_j$, $\hat\mu_j$, $\hat d_j$; the masks and the stripe
only, no image]** as built ($\tau$ = 0.03 gl) 21 of the 138 near candidates have a usable stripe at $k^*$ (91
`too_few`, 26 `side_short`) and none of the 7 far candidates has a usable stripe in any of its 24 fit planes; over the
node itself 89 near candidates lie under a neighbour whose predicted shadow exceeds 0.03 gl with
$\sigma_{\rm r}(\delta_{q,k}) \ge$ 2 um, 10 under a junction segment, 3 under a sharp one. With $\tau$ = 0.3 / 1 / 3 gl
the near set keeps 92 / 117 / 127, and the predicted error the neighbours' summed shadow leaves on the $\pm$3 um line
after a plane fitted on the stripe has median 0.24 / 0.35 / 0.45 gl over the usable nodes (90th percentile 0.82 / 2.1 /
3.8 gl; 0.10 and 0.13 gl as built); the far set keeps 0 / 4 / 6 nodes with at least 6 usable fit planes, their error on
the $\pm$10 um line 3.5-14 gl per node, against the pilot's median background noise SD of 5.9 gl (D-101). Leaving the
shadows with $\sigma_{\rm r} \ge$ 2 um to the plane at 0.03 gl keeps 82 near nodes (their error median 0.37 gl). The
performance found on the same scene: the first version's masks took 30-50 s per plane for a near node; now 1-14 s per
node. **[reasoning]** a neighbour $\delta$ off the plane spreads its shadow over $\sigma_{\rm r}(\delta) \approx$ 0.76-0.79 $\delta$
for $\delta \ge$ 2 um (the renderer's current law **[run, sandbox]**), and the reach of a fixed per-pixel $\tau$ grows in proportion (2.1-2.6 $\delta$ for a 0.3 um
tube at $\bar B$ = 200 gl, $\hat\mu$ = 1 /um), so in a dendritic field a few tens of um deep almost every node lies under
some out-of-focus dendrite's 0.03 gl contour; those shadows are smooth and mostly absorbed by the planar background,
which is what $\tau$ was not built to recognise.

**What it implies for code and tables.** On `sci/diameter-pipeline`: `39ae156` (the recheck; the tester's report),
`93a5fde` (panel (c), the knobs, the notebooks, the SPEC), `a4d8b17` (runbook v36), `732e303` (the checks),
`fcbaf1c` (their references), `ff81622` (the roadmap md); pushed, a fresh clone of `a4d8b17` passing the 40 smoke
suites (248 pass, 38 skip) and the image checks **[run, sandbox]**. No signature changes: estimator `59678d4e2aedf2b6`, simulation `52aa73e479eb77cf`. Colab: Cells 0, 1 and 7 (and 15)
again; an open notebook needs Cell 7's new code from GitHub (the knobs live in the cell). davinci: nothing until the
knob file is passed.

**What it does not decide [open].** (a) How out-of-focus shadows enter the stripe -- D-101 (d): raise $\tau$ (a knob
now), leave the shadows with $\sigma_{\rm r} \ge \sigma_{\rm cap}$ to the plane under a bound on the predicted error on
the line (a new rule), or keep the rule as built and take the far set from the survey's cells. (b) S4: the far set's
minimum side, 3 um or the far line's 10 um. (c) F2's reading of the fitted cylinder. (d) F3: the pilot's tilt error on
0.2 um nodes (up to 4.5 deg on the tester's phantoms) against the 10 deg bound. (e) Whether the far set's verdict
before the fetch should count the fit's usable planes (at least 6) instead of $k^*$'s stripe -- panel (c) shows
both.

**Sources.** **[user]** 2026-10-28, 19:39. **[run, sandbox]** the tester's report; `test_smoke_stripe.py` (7 pass,
1 skip), `test_smoke_isolation_view.py` (6 pass, 1 skip; 7 mutants killed); `checks/d105_2026-10-28/`
(`summarize.out`). **[data]** D-101's noise SD. **[reasoning]** the blur-reach argument above.

**Status:** active; implemented and pushed 2026-10-28; smoke-tested; open (a)-(e).

---

## D-106 -- A group with too few windows leaves the noise statistics as well; a group dropped only for a failed fit keeps its windows (user)

**Date:** 2026-10-29, 14:17 (Europe/Rome), the noise-fit chat. **Binds:** D-104's open point (a); (D-104.2)'s dropped
groups; Block 3's statistics (per window, per sweep and per cell: `noise_cell_stats`) and the layout it hands to Block 4
(`noise_fit.CellLayout.from_windows`); Block 4's simulator (`simulate_cell_windows`); `WindowConfig.min_windows_per_group`.

**Decision [user].** "Should a dropped group's windows also be left out of the noise statistics? Yes as you propose",
the answer to D-104's open point (a), where the assistant had proposed: yes when $K_{s,p}$ < 5, no when only the group's
fit failed.

**Statement.**

(D-106.1) *Too few windows.* For a group $(s,p)$ of an SS sweep $s$ and polarity $p$ with $K_{s,p}$ below
`min_windows_per_group` (5, PROVISIONAL, D-104.2), its windows leave the noise statistics as well as the start's
maximum: they enter no per-window, per-sweep or per-cell statistic, nor $\hat V_c(h^*)$ while (T1) is computed, nor the
tilt reading of (D-102.2). The layout handed to Block 4 leaves them out too, so the simulator's replicate windows are
cut at the same positions and the operation stays part of the estimator on both sides (D-051 (iv), D-033). The flag
`ss_group_dropped:<sweep>:<polarity>:few_windows` records it.

(D-106.2) *A failed fit only.* A group dropped from the maximum because its fit failed (no convergence, a non-finite or
non-positive $\hat\tau_{s,p}$) keeps its windows in the statistics, each minus its group's median window $m_{s,p}(t)$
(D-103.1) on $[t_{\rm SS}(c), 200)$ ms; the flag `ss_group_dropped:<sweep>:<polarity>:fit` records it.

**Why [user; reasoning].** **[user]** as quoted. **[reasoning]** After its own median is subtracted, a group of one
window is identically zero, and a group of two keeps only half of each window's difference from the other: the
statistics of such groups say little about the noise. A failed decay fit says nothing about the windows' noise, so
their windows are kept.

**What it implies for code and tables.** Block 3 removes the windows of the small groups before the statistics and
before the layout is built (`SSWindows` filtered; `K` per sweep and polarity recomputed); the per-cell window count
and every per-cell median exclude them. Smoke: a group of 3 windows is absent from the statistics and from the layout,
with its flag; a group whose fit fails keeps its windows, with its flag.

**What it does not decide [open].** The value of `min_windows_per_group` (5, PROVISIONAL); $\kappa_{\rm tr}$ (0.1,
provisional since D-051).

**Sources.** **[user]** 2026-10-29, 14:17. **[log]** D-033, D-051, D-102, D-103, D-104.

**Status:** active; not implemented. Closes D-104's open point (a).

---

## D-107 -- Confirmations closing D-102's open points: a Long Square sweep's tilt is the line through a start-window mean and an end-window mean; the noise fit's simulator gives every simulated sweep its real sweep's tilt (user)

**Date:** 2026-10-29, 14:41 (Europe/Rome), the noise-fit chat. **Binds:** D-102's open points (a), (b) and (d);
(D-102.3), (D-102.4); Block 3 (the tilt readings, SS and LS); Block 4 (`simulate_cell_windows`, `binding_function`,
`simulated_covariance`, `fit_cell`: the tilt alternation, `tilt_from_semivariogram`, the flags `tilt_clipped` and
`tilt_no_pairs`); Block 2's `simulate_traces` (a fixed tilt per trace); Block 5's generator; `Ih Fit/specs/SPEC.md`
§2.3 (f)-(h), §2.4 and §8; the columns `noise_sigma_b_mV_per_s` and the per-sweep tilt.

**Decision [user].** "1) Tilts of the Long Square sweeps: Take a mean window at the start, a mean window at the end and
define the tilt as teh one that passes thorugh these two measn 2) Tilts in the noise fit's simulator: every simulated
sweep gets the same $\hat b_{c,s}$ as its real sweep, in every replicate PERFECT". The answers to D-102's two questions,
restated to the user at 14:17.

**Statement.**

(D-107.1) *The tilt of a Long Square sweep* [user]. For each LS sweep $s$ of cell $c$,
$$\hat b_{c,s} = \frac{\bar y^{\rm end}_{c,s} - \bar y^{\rm pre}_{c,s}}{\Delta t_{\rm c}} \qquad ({\rm mV/s}), \tag{D-107.1}$$
the slope of the line through the two window means, with $\bar y^{\rm pre}_{c,s}$ (mV) the sweep's mean over its
pre-window, $\bar y^{\rm end}_{c,s}$ (mV) its mean over the last 500 ms of the stored sweep, and $\Delta t_{\rm c}$ (s)
the distance between the two windows' centres [the assistant's reading of "a mean window at the start, a mean window at
the end": the windows of `ls_baseline_qc.py`'s `delta_end_mV`, so that $\hat b_{c,s}$ = `delta_end_mV` /
$\Delta t_{\rm c}$]. On the archive's L3_exc LS geometry -- pre-window [0, 265) ms, step from 270 ms for 1 s, 6.000 s
stored after it -- $\Delta t_{\rm c}$ = 6.8875 s **[src; log: the living handoff §4.6]**. Under (L1), the mean of a line
over a window being its value at the window's centre, $\hat b_{c,s} = b_{c,s} + (\bar e^{\rm end}_{c,s} - \bar e^{\rm
pre}_{c,s})/\Delta t_{\rm c}$, with $\bar e$ the window means of the stationary part: unbiased, under the hypothesis
that the step leaves no after-effect in the end window, 5.5-6 s after it (the living handoff §4.2's rebound caveat; the
uncorrected hint of a rise with step size in 526785799, §4.6) **[reasoning; log]**. At $\theta_{\rm cf}$ the error's SD
is 0.0045 mV/s -- 4.5 % of a tilt SD of 0.1 mV/s, 17 % of 0.027 mV/s -- from Block 2's exact window-mean moments,
matched by 2000 simulated sweeps (0.00449 against 0.00451 mV/s) **[run: `check_ls_tilt_reading.py`, 2026-10-29]**; the
SS reading of (D-102.2) erred by 0.0027-0.0036 mV/s RMS in its check's configuration **[run: check03 (f) of the tilt
study]**.

(D-107.2) *The fit's simulator* [user: "PERFECT"]. In every replicate of the binding function and of $\hat\Sigma$, each
simulated sweep -- SS and LS -- carries the tilt of its real sweep, $\hat b_{c,s}(t_j - \bar t_s)$, a fixed number, not a
draw. The tilt SD leaves the fit: (T1), `tilt_from_semivariogram` and the alternation of $\hat\sigma_b(c)$ with
$\hat\theta_c$ (SPEC §2.3 (f)-(h)) are not computed [the assistant's proposal, confirmed]; (T1) is not kept as a
report-only diagnostic either [the assistant's reading of D-102: "no statistics has to be computed on the tilt"].

**Why [user; run; reasoning].** **[user]** as quoted. **[run]** (D-107.1): the line through two window means 6.9 s apart
reads an LS sweep's tilt to about 0.0045 mV/s at $\theta_{\rm cf}$, of the order of the SS reading. **[reasoning]** With
the same tilts in the simulator, the fit's simulated statistics carry the real sweeps' tilts exactly, as its subtraction
mirrors the data's; nothing about the tilt is estimated inside the fit.

**What it implies for code and tables.**
- Block 3: the LS reading (D-107.1) beside the SS reading (D-102.2), one tilt per sweep and protocol, written to the
  per-sweep table (e.g. `noise_tilt_mV_per_s`); `ls_baseline_qc.py`'s `delta_end_mV` is its oracle in the smoke test.
- Block 2: a fixed tilt per trace (mV/s) as an input of `simulate_traces`, which today forms `sigma_b * xi_tilt`
  [proposal: a new keyword argument, the old pair kept].
- Block 4: `simulate_cell_windows` takes the per-sweep tilts in place of `sigma_b`; `fit_cell` loses the alternation;
  smoke F5 retired, F7's tilt assertions replaced; `noise_sigma_b_mV_per_s` and the flags `tilt_clipped`, `tilt_no_pairs`
  retired.
- Block 5: the generator takes the per-sweep tilts of both protocols.
- SPEC §2.3, §2.4 and §8 rewritten with the blocks (marked now).

**What it does not decide [open].** (a) The precision of the SS reading (D-102.2) with D-103's starts and the median
subtraction (not checked). (b) Whether the LS end window stays at 500 ms (`END_WINDOW_MS`, the Allen convention) or is
made longer.

**Sources.** **[user]** 2026-10-29, 14:41. **[log]** D-017 (note 4), D-034, D-102, D-103; the living handoff §4.2 and
§4.6. **[src]** `ls_baseline_qc.py` (`END_WINDOW_MS` = 500, `delta_end_mV`), `noise_law.var_window_mean`,
`window_mean_cov`, `simulate_traces` (`main` at `c7a587a`). **[run, sandbox]** `check_ls_tilt_reading.py` (2026-10-29,
`Ih Fit/dev/2026-10-29_ls_tilt/`; P1, P2 PASS, re-run identical); check03 (f) of the tilt study.

**Status:** active; not implemented. Closes D-102's open points (a), (b) and (d); its (c) stays open as (a) above. **[updated 2026-10-29 (later): (a) answered by the note below.]**

**Note added 2026-10-29 (later, 2), from `check_ss_tilt_reading.py` (`Ih Fit/dev/2026-10-29_ls_tilt/`), run at the
user's request; no decision changes.** *What was checked:* the SS reading of (D-102.2) under D-103's per-sweep,
per-polarity median subtraction and its later window starts. *Configuration:* $\theta_{\rm cf}$, $\sigma_b$ = 0.1
mV/s, continuous sweeps from the tilt study's simulator (Block 2 underneath), the study's four reference layouts (2
sweeps with $K$ = 40 and 63, 6 with 199, 10 with 348), 2000 or 400 simulated cells, starts 80, 130 and 150 ms
**[run]**. *Results:* (i) the subtraction does not change the reading -- no subtraction, the per-polarity mean of
D-051.1 and the per-$(s,p)$ median give slopes equal to $2\times10^{-16}$ mV/s, since each subtracted window shifts
every window of its group by one constant and a group is one polarity of one sweep, which the polarity intercept
absorbs **[run; reasoning]**; (ii) the RMS of $\hat b_{c,s} - b_{c,s}$ is 0.0036, 0.0028, 0.0028, 0.0027 mV/s at
80 ms (check03 (f) reproduced), 0.0041, 0.0032, 0.0032, 0.0031 at 130 ms and 0.0044, 0.0034, 0.0034, 0.0034 at 150
ms for the four layouts -- a factor 1.15 and 1.23 from the shorter windows' noisier means -- each within 3 SE of
the exact value from the law's window-mean covariances **[run]**. The error does not depend on $\sigma_b$ (the
reading is linear and the tilt exactly linear); at the later starts it is 3-4 % of a tilt SD of 0.1 mV/s and 12-16 % of 0.027 mV/s
**[run; arithmetic]**. *Not covered:* pulse-to-pulse variation of the response, which enters each window's mean
and so the reading **[reasoning]**; the real cells' laws.

---

## I-001 — Human spine simulations from H01 geometry (two tiers)

**Date:** 2026-09-19. **Concerns:** the spine→shaft link that the main pipeline
removes (spines pruned, membrane carried by $F$, synapses redirected to the
shaft, conductances shaft-referenced by D-001); the value and validity of
$R_{\rm neck}$ for human spines; a possible standalone result.

**Idea [user].** Reconstruct several H01 spines of different geometries,
together with a portion of the attached shaft and with the cytosolic
"hindrance" elements (organelles) so that the tortuosity — or at least its
lower bound — can be derived; implement the electrodiffusion laws; simulate
synaptic activity of different magnitudes and frequencies. The user proposed
NEURON for all of it.

**Assessment (2026-09-19; literature searched that day).** Worth pursuing as a
bounded side project with a defined deliverable, in two tiers, and not all of
it in NEURON.

*What only a spine-resolved model can answer.* (i) The **human** distribution
of $R_{\rm neck}\in\mathbb{R}_{>0}$ (MΩ, one scalar per spine) from real,
non-cylindrical necks; (ii) the regime — conductance × frequency × head volume
— in which the scalar conversion of D-001,
$G_{\rm syn}^{\rm shaft}=G_{\rm syn}^{\rm head}/(1+G_{\rm syn}^{\rm head}R_{\rm neck})$,
stops holding because ion concentrations in the head change; (iii) the shift
of $R_{\rm neck}$ due to organelle occlusion of the neck lumen.

*What it cannot answer alone.* $R_a$ (Ω cm, the cytoplasmic resistivity of the
cable equation). The hindrance factor $\lambda^2/\alpha$ (tortuosity
$\lambda$, free-volume fraction $\alpha$, both dimensionless) multiplies a
free-solution resistivity that must still be assumed, and under the
insulating-obstacle hypothesis it is a **lower bound** on the multiplier
(Gardner-Medwin; established earlier in the same chat) **[reasoning +
textbook]**. Geometry in, a *range* of $R_{\rm neck}$ out, never a point value.

*State of the art (PubMed / bioRxiv, searched 2026-09-19).*

| Work | What it did | Left open for us |
|---|---|---|
| Ofer, Shapira & Segev 2026 — **bioRxiv PREPRINT, not peer-reviewed; published-version lookup 2026-09-19: none** — [DOI](https://doi.org/10.64898/2026.02.16.706119); full text read via PMC | "Biophysics of connectomics": 2,074 EM-reconstructed **mouse L6** spines (6×6×30 nm); section-by-section $R_{\rm neck}=\sum_i R_a\,\ell_i/A_i$ over virtual cross-sections ($\ell_i$ section length, $A_i$ cross-sectional area); the average-diameter cylinder underestimates $R_{\rm neck}$ by ~50 % (median 150 MΩ, central 99 % range 2–918 MΩ under their assumed $R_a$); NEURON 8.2 **cable** models, no electrodiffusion; organelles cited as occluding 7–27 % of the neck lumen, not modelled; fixation-artefact caveat discussed | the human case; electrodiffusion; organelles as measured obstacles |
| Lagache, Jayant & Yuste 2019 — [DOI](https://doi.org/10.1007/s10827-019-00725-5); full text read | coarse-grained Poisson–Nernst–Planck, head + cylindrical neck, one cation and one anion with equal diffusivity; head concentration rises up to ~90 % during an EPSP; $R_{\rm neck}$ *decreases* with concentration; trains ≥ 50 Hz predicted to facilitate | symmetric two-ion electrolyte |
| Eberhardt 2024 — [DOI](https://doi.org/10.1007/s10827-024-00864-4); full text read | cable equation + diffusion currents for Na⁺, K⁺, Cl⁻ with distinct diffusion constants $D_k$; explicit finite-difference solver, code released; with physiological composition $R_{\rm neck}$ *slightly increases* (sign opposite to Lagache; the disagreement is traced to the equal-diffusivity / high-Cl⁻ assumption); head depolarisation boosted up to 45 %; NMDAR current up to ~25 %; voltage-divider estimates of $R_{\rm neck}$ biased upward | our geometry, our ion composition, trains |
| Paragot, Krell & Guerrier 2026 — [DOI](https://doi.org/10.3389/fncel.2026.1740414); **abstract only** | full PNP with a DDFV finite-volume scheme on 2-D spine/branch geometries; a spine near a large branch behaves as isolated | 3-D, organelles |
| Qian & Sejnowski 1990 — [DOI](https://doi.org/10.1073/pnas.87.20.8145); **abstract only**; Holcman & Yuste 2015 — [DOI](https://doi.org/10.1038/nrn4022); **abstract only** | classical and review statements that cable theory fails in spines because concentrations change | — |

**Disagreement between sources, to carry into Tier 2:** the *sign* of the
$R_{\rm neck}$ change during an EPSP depends on the assumed electrolyte
(Lagache: decrease; Eberhardt: slight increase with physiological Na⁺/K⁺/Cl⁻
and distinct $D_k$). Our model must use physiological composition and
species-specific diffusion constants, and report the sign as a result.

Two PubMed queries for direct *human* spine-neck electrical measurements
returned nothing. The only human value in the knowledge base is Eyal 2018's
$R_{\rm neck}=50$–$80$ MΩ **[KB]**, an average-cylinder estimate (neck
0.25 µm × 1.35 µm, $R_a=200$–$300$ Ω cm) — the method the preprint shows
underestimates $R_{\rm neck}$ in mouse. That gap is the opening.

*NEURON.* The cable tier is NEURON's home ground (the preprint used it). The
electrodiffusion tier is not: NEURON's `rxd` does Fickian reaction–diffusion
coupled to membrane currents and, per McDougal et al. 2022
([DOI](https://doi.org/10.3389/fninf.2022.847108); full text, relevant passages
read), "we do not consider electrodiffusion … we intend to add support". The
current release's documentation was not checked (fetch blocked that day)
**[open]**. Practical route: Eberhardt's 1-D scheme (head + segmented neck +
shaft stub, three ion species) is a few hundred lines and answers every
*electrical* question; full 3-D PNP (FEniCS, or a DDFV code) only if organelles
are to be explicit internal boundaries rather than a per-section area
reduction.

*Plan.*

1. **Tier 1 — cheap, in NEURON/pandas, feeds TEEG directly.** From the
   per-spine node tables written by P1 (handoff section 4.1: nodes, radii,
   head/neck class, base, synapses) compute the section-by-section
   $R_{\rm neck}$ for every H01 spine, per layer and type, with a
   lumen-occlusion column where the spine apparatus is visible. Output: the
   human counterpart of the preprint's distribution, and the per-spine
   conversion factor $1/(1+G_{\rm syn}^{\rm head}R_{\rm neck})$ that closes
   D-001's open point "which $R_{\rm neck}$ for converting class-(c) values".
   Uncertainty carried as a range over $R_a$.
2. **Tier 2 — own solver, not NEURON.** ~10 geometries chosen by
   $R_{\rm neck}$ quantile, with and without occlusion; AMPA conductances
   0.5–3 nS; trains 10–200 Hz; physiological Na⁺/K⁺/Cl⁻ with species-specific
   $D_k$. Output: the boundary of the linear regime — where D-001's
   shaft-referenced scalar needs a frequency- or history-dependent correction
   — and a testable human-specific hypothesis: with larger heads and thicker
   necks (Eyal 2018 **[KB]**) the same synaptic current changes head
   concentration less, so human spines should sit deeper in the linear regime
   than mouse. A hypothesis to test, not a result (no mouse number is
   transplanted).
3. **Not worth doing:** deriving $R_a$ from tortuosity alone; PNP inside
   NEURON.

*Caveats that travel with the plan.* Whether H01 ships organelle
segmentations could not be confirmed from the H01 paper in the knowledge base
(its methods describe subcompartment classes and spine detection, not
organelles) **[open]** — mitochondria / spine apparatus may have to be
segmented from the released EM volume; resected human tissue carries the same
fixation-shrinkage question the preprint raises for neck constrictions, so a
shrinkage sensitivity analysis belongs in Tier 1.

**What would be needed.** (a) The P1 spine-statistics tables (handoff §4.1)
with the `spine_id` join contract — Tier 1 is blocked without them; (b) a
decision on the $R_a$ range to report; (c) for Tier 2, a solver choice
(Eberhardt-style 1-D by default) and its smoke test against the published
cases; (d) organelle segmentation source; (e) a shrinkage-correction policy.

**Relation to decisions.** Tier 1 closes D-001's open point on $R_{\rm neck}$;
Tier 2 defines the validity regime of D-001. If Tier 2 shows the linear regime
fails at physiological rates for a class of spines, D-001 gets a superseding
entry.

**Sources.** Project KB — Eyal 2016 (validity condition of $F$), Eyal 2018
(human spine geometry), Shapson-Coe et al. 2024 (H01 methods). PubMed full
text — Lagache 2019, Eberhardt 2024, McDougal 2022 (excerpts). bioRxiv
preprint, full text via PMC — Ofer 2026 (published version: none as of
2026-09-19). PubMed abstract only, flagged — Paragot 2026, Qian & Sejnowski
1990, Holcman & Yuste 2015. Searches run 2026-09-19: PubMed ×6, bioRxiv
published-lookup ×1, KB ×1.

**Status:** idea; user wants to pursue; not scheduled; depends on handoff §4.1.

---

## I-002 -- A non-Gaussian defocus kernel, with the Gaussian as a special case

**Date:** 2026-10-06, 17:01. **Concerns:** the kernel family of the renderer
and of the calibration (procedure §3.4, "Two kernel families"); D-026 keeps
the Gaussian for now.

**Idea [user].** "For now I see that we have assumed the defocus kernel as
being Gaussian which might not be the case. So in estimating the kernel and
the calibration table  we can also account for a more flexible family
distribution (that has as a specific case the Gaussian distribution but can
also be shaped into skewed ones) and fit also its parameters given the
estimated second moment. It's pretty difficult so maybe we might need to
discuss this further. Take note of this for the things to discuss furhter."

**Why it might matter.** The real line-spread function differs from a
Gaussian in two known ways: $\lvert v\rvert^{-2}$ tails at every defocus,
which dominate its second moment near focus (mathematics §3.3), and, away
from focus, a shape that broadens, flattens on top and grows a rim
(mathematics §3.4) **[KB-repo]**. A Gaussian matched to one statistic misses
the rest (procedure §3.4).

**First comments of the assistant, for the discussion [reasoning unless
tagged].**

- (a) **Symmetric across the branch, asymmetric in depth.** For a
  rotationally symmetric PSF the profile across a straight branch is even in
  $v$, so a kernel skewed in $v$ would need off-axis aberrations or oblique
  illumination -- testable as the skewness of averaged thin-node profiles.
  What can be asymmetric is the dependence on $\delta$: the defocused PSF is
  the same above and below focus in an index-matched system free of spherical
  aberration (optics §3.5) and differs under index mismatch (optics §3.7;
  procedure §3.10, "Above/below") **[KB-repo]**. The natural generalisation is
  then a family symmetric in $v$, with a shape parameter for the tails and the
  flatness of the top, allowed to differ for $\delta > 0$ and $\delta < 0$.
- (b) **Identifiability.** One second moment fixes one parameter; a shape
  parameter needs the profile's shape -- procedure §3.4 step 2 already fits
  whole per-plane profiles -- or further statistics (two window widths, a
  fourth moment). A heavy-tailed member has no finite second moment, so the
  window becomes part of the definition.
- (c) **Two alternatives to weigh against it.** The empirical kernel already
  in procedure §3.4 (averaged real profiles, after four corrections), and a
  physical PSF model whose few parameters (mounting index, depth, condenser
  aperture) generate the shape. For transmitted-light brightfield stacks of
  rat cortical tissue in Mowiol imaged with an oil objective, Oberlaender et
  al. 2009 measured the aberration function of microscope and tissue with a
  Shack-Hartmann sensor, report the tissue's index as homogeneous and the
  spherical aberration as relatively low, and deconvolve with a model PSF
  built from refraction between embedding and immersion medium and 3-D
  diffraction at the objective's pupil, for reconstructing biocytin-stained
  dendrites and axons (**PubMed, abstract only; full text not accessible**,
  [DOI](https://doi.org/10.1111/j.1365-2818.2009.03118.x)). The Allen
  mounting medium and its index are not known (implementation handoff,
  "Known gaps"), so that result is not transplanted.
- (d) **A literature lead, not yet usable.** 3-D localisation microscopy
  reads a single fluorophore's axial position from the shape of its
  astigmatic PSF (Huang et al. 2008, **PubMed, abstract only**: the PMC
  record returned no body text, [DOI](https://doi.org/10.1126/science.1153529)).
  Doing so needs a calibrated relation between PSF shape and defocus
  **[reasoning]**; whether that paper's calibration curve carries asymmetric
  terms usable as D-026's "other terms" is **from memory, not verified**.

**What would be needed.** Averaged thin-node profiles from block 10 (symmetry
and tail shape); the full texts of Oberlaender 2009 and Huang 2008 (with its
supplementary material), by upload; a literature search on PSF models and
kernel families when the discussion opens.

**Sources.** **[user]**, 2026-10-06 17:01. Mathematics §3.3, §3.4; procedure
§3.4, §3.10; optics §3.5, §3.7; implementation handoff "Known gaps" (repo,
D-020). Oberlaender M. et al. 2009, *J Microsc* (PMID 19220694) and Huang B.
et al. 2008, *Science* (PMID 18174397, PMC2633023): PubMed, abstract only.
Searches 2026-10-06: PubMed, 3 queries (3-D localisation by astigmatism; PSF
width versus defocus calibration; defocus blur from edge or line-spread
width), returning 2, 0 and 0 records; plus the brightfield PSF query of D-025.

**Status:** idea, to discuss further **[user]**; not scheduled. [updated 2026-10-13] Part of it tried as a diagnostic on the fit, not on the renderer or the calibration: D-046 (Cell 4l) fits Block 5's profiles with Lorentzian, Voigt, Student, generalized-normal and skew-normal kernels, shapes fixed or fitted; the idea itself stays open. [updated 2026-10-16] The fit's kernel is now a flexible family with the Gaussian as a member (D-052: the skewed generalized $t$, three shape parameters fitted per node); the renderer and the calibration keep the Gaussian (D-026), which is what this idea concerns, so it stays open.

**Note added 2026-10-20 (later).** Adopted by **D-066** [user] for the renderer and the calibration: the SGT family of D-052 in both chains.

**Note added 2026-10-20 (later, 2).** D-066 withdrawn by **D-067** [user]; back to an idea, not scheduled. Kept from D-066: every symmetric SGT member is the line projection of a nonnegative isotropic PSF, the skew of none (`check_abel_sgt.py`, 2026-10-20).

---

## I-003 -- The defocus kernel constructed with deep learning, in a chat of its own

**Date:** 2026-10-22, 15:09 (Europe/Rome), the user's request in the diameter implementation chat.
**Concerns:** the renderer's defocus kernel $\sigma_{\rm r}(\delta)$ (Block 4) and its construction from
Allen's stacks -- Block 10's near field (procedure Eq. 4), S2b's far field ((D-026.1), D-074), procedure
§3.4's step 2 (not built); the kernel family (D-067.2, I-002).

**Idea [user].** "Can you please write down an handoff for another chat where I tackle this defocus kernel
construction from a deep learning perspective? THanks" -- the user takes up, in a chat of its own, the
construction of the defocus kernel with deep-learning methods.

**What it changes now.** No decision. The renderer's kernel stays the circular Gaussian of D-067.2; S2b's
open points (D-074.3, D-074.4, Q-D) stay with the implementation chat; a learned kernel enters the pipeline
only through `RendererConfig`'s kernel fields (in the simulation hash) or, if non-Gaussian, through a new
kernel family, which reopens I-002 and D-066 and is the user's decision **[KB; src]**.

**First comments of the assistant [reasoning unless tagged].** In the handoff
(`docs/TEEG_diameter_defocus_kernel_DL_handoff_2026-10-22.md` on `sci/diameter-pipeline`, local commit
`b82b1ee`): six scope questions for the user before any logic; five candidate formulations, none decided --
a learned far-plane background (as BGnet), a learned per-plane width statistic, procedure §3.4's step 2 as
differentiable analysis by synthesis (as uiPSF), amortised neural posterior estimation (as SNPE), a
non-Gaussian kernel (reopens I-002); what width scans cannot identify ($\sigma_{\rm r}(0)$, sub-plane
resolution, Eq. 4's additive constant and common shift), with a prior-sensitivity test for any learned
estimator; the classical baseline on the renderer's model (S2b checks: bias +0.01 % and -0.18 %, SD 0.0039
and 0.0064 at $\gamma$ = 0.79 and 1.21, $J$ = 12, 1 gl) **[run, sandbox, 2026-10-21]**; acceptance tests,
the first of which -- how much a kernel error moves $\hat d/d$ -- has never been run and sets the accuracy
that matters **[open]**.

**What would be needed.** Real calibration profiles (Colab Cell 7 has not run, and the scans file keeps
per-plane statistics only, so a profile export); the S2b checks (the zip of 2026-10-21, not in the
repository); PyTorch (not installed in the sandbox); a GPU (Colab or davinci, not checked).

**Sources.** **[user]** 2026-10-22, 15:09. **[KB]** D-024 (i), D-025, D-026, D-061, D-066, D-067, D-070,
D-074, I-002. **[src]** `825bef1`. **[PubMed, full text]** uiPSF (Liu et al. 2024,
[DOI](https://doi.org/10.1038/s41592-024-02282-x)), BGnet (Möckl et al. 2020,
[DOI](https://doi.org/10.1073/pnas.1916219117)), SNPE (Gonçalves et al. 2020,
[DOI](https://doi.org/10.7554/eLife.56261)); abstract-only and preprint leads listed in the handoff, §12.

**Status:** idea; the user pursues it in a chat of its own **[user]**; not scheduled in the diameter plan
(D-067.1's order unchanged). [updated 2026-10-22, later: the handoff pushed at the user's agreement, `9112bf3..37b42ea` (`b82b1ee` is on the remote, merged with the roadmap commit); see also I-004]

---

## I-004 -- The hybrid renderer: each slice's absorption from the condenser's rays, then the defocus kernel; its optics studied first

**Date:** 2026-10-22, about 16:20 (Europe/Rome), the user's answer to the push card in the diameter
implementation chat. **Concerns:** Block 4's absorption (`partition_vertical`, D-024 (i) "for now") and
Block 9's ray world; the hybrid of `TEEG_diameter_renderers_partition_ray_world_hybrid_2026-10-09.md`
§3.8-3.9 (on `main`), which that document calls "the user's proposal of 2026-10-09; it is not yet logged as
a decision or an idea"; the kernel the hybrid applies (D-067.2, I-003); the roadmap's theory stage T7.

**Idea [user].** "Moreover in the new roadmap add that I need to study all the optics theory to construct
the hybrid plane absorption to which the render then applies the defocusing kernel and so on." The
assistant's reading: the user will study the optics needed to build the hybrid renderer of 2026-10-09 --
each slice's absorbed light computed along the condenser's rays (NA 1.4, $n_{\rm oil}$ 1.515) and averaged
over the directions arriving at each point (that document's Eq. 13), then every image plane rendered by
applying the defocus kernel to the slices' absorbed light (its Eq. 14) -- and what follows from it in the
chain.

**What exists [doc].** The renderers document (the theory chat, 2026-10-09, on `main`; it "records no
decision"): the partition has the right blur and, in dark stain, the wrong bookkeeping; the ray world has the
right bookkeeping and, lacking diffraction, the wrong blur near focus; the hybrid takes absorption from the
rays and blur from the kernel, drops the cone's own spread, and averages over directions before spreading;
with the kernel that faint nodes would calibrate in a ray-world microscope it equals the ray world at first
order, and near the opaque limit its centre is too light, as the partition's is (§3.8-3.9 and its findings).

**What it changes now.** Nothing in the code or the plan: Block 4 renders with the partition (D-024 (i)),
the kernel stays Gaussian (D-067.2); a hybrid would be a new `absorption` value of Block 4, inside the
simulation hash, with a SPEC block of its own and the user's decision. Recorded as the roadmap's theory
stage T7, status open, depending on T3, needed by no implementation stage yet (`982186c`) **[src]**.

**What would be needed.** The optics of the cone and the sine condition (optics v1.5 §3.3-3.7 and
`TEEG_diameter_sine_condition_and_W_2026-10-12.md`, both on `main`); the renderers document §3.4-3.9 and its
open points (§5); then a SPEC block for the new absorption mode (Block 4's modes, Block 9's direction
quadrature).

**Sources.** **[user]** 2026-10-22, about 16:20. **[doc]** the renderers document of 2026-10-09 (§3.8-3.9,
the findings of its abstract); optics v1.5 (on `main`). **[KB]** D-024 (i), D-067.2, I-002, I-003.
**[src]** `specs/roadmap/diameter-pipeline.yaml` at `37b42ea`.

**Status:** idea; the user studies its optics first (roadmap T7, open) **[user]**; not scheduled in the
diameter plan. [updated 2026-10-26, 16:48: promoted by **D-084** (user) to be the renderer actually used once built; its optics (T7) first] [updated 2026-10-27: built as SPEC Block 21 (**D-092**), smoke-tested, not yet the default; T7 still open]
