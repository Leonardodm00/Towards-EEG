# docs/project -- copies of the claude.ai project's documents (D-097)

The Towards-EEG project on claude.ai keeps its documents in project knowledge, where its chats read
them. Since 2026-10-28 every document is also kept in this repository, so that any chat can read it
here (D-097, the user: "every docuemnt also must be in the github repository so any chat can consult
it"). This folder holds the copies of the project-knowledge documents, at the path each has there:
`claude/<name>` for the documents Claude wrote under `claude/`, the folder's root for the others.

Rules (the assistant's reading of D-097, recorded there as open point (a)):

- **The project copy is the source of truth while it exists.** A chat that writes a project document
  writes it here too, in the same turn, byte for byte (one write per document per turn, as the
  project's DOCUMENT UPKEEP section asks). A copy here that differs from project knowledge is stale,
  not a fork: re-read the project document, merge, write both.
- **Project knowledge has been full since 2026-10-28** (a new document was refused). A document that
  cannot be written there lives only in the repository, and the row of its own changelog says so.
- **A workstream's own documents stay beside its code**: handoffs in `<workstream>/docs/`, study
  documents and check packages in `<workstream>/dev/<date>_<topic>/` (for the I_h fit,
  `Passive Features/HPC script/Ih Fit/docs/` and `.../Ih Fit/dev/`).
- **This copy lives on `main`.** A workstream on a branch of its own copies its project documents to
  `main`'s `docs/project/` (the assistant's proposal, D-097 open point (d)).
- **The repository is public.** Everything here is public: no credentials, no personal data, no
  published papers.

## Copied

| Path here | In project knowledge | Copied | Workstream |
|---|---|---|---|
| `TEEG_decisions_and_ideas_log.md` | root | 2026-10-28, with D-097 | all chats |
| `claude/TEEG_Stage7_noise_model_2026-09-28.md` | `claude/` | 2026-10-28 (v3.4 of 2026-10-16) | I_h fit, Stage-7 noise |
| `claude/TEEG_Ih_fit_staged_plan.md` | `claude/` | 2026-10-28 (v0.12 of 2026-10-21) | I_h fit |
| `claude/TEEG_Stage7_noise_handoff_2026-10-01.md` | `claude/` | 2026-10-28, with its D-097 row | I_h fit, Stage-7 noise |

## Not copied yet

Documents Claude wrote, to be copied by the chat that next writes each, or all at once on the user's
word:

- `claude/`: `TEEG_acf_textbook_change_plan_2026-10-07.md`, `TEEG_annual_report_2026_00_overview.md`,
  `TEEG_annual_report_2026_01_passive_properties.md`, `TEEG_annual_report_2026_02_structural_pipeline.md`,
  `TEEG_campaign_single_submission_handoff_2026-09-22.md`, `TEEG_diameter_HPC_paths_reference.md`,
  `TEEG_HPC_coding_handoff_2026-09-19.md`, `TEEG_alignment_metadata_reference.md`,
  `TEEG_HPC_paths_reference.md`;
- root: `TEEG_HPC_paths_reference.md`, `TEEG_cylinder_calibration_HPC_and_Colab_reference.md`,
  `TEEG_S1_operational_reference_v1.md`.

Not copied without the user's word (files the user uploaded, DOCUMENT UPKEEP): `phase1_function_reference.md`,
`phase1_conceptual_guide.md`, `phase2_patch.md`, `Bayesian Optimisation and Likelihood Profile.md`,
`figure1_passive_pipeline.docx`, `passive_properties_summary.docx`, `Supplementary materials of Human
Cortical Pyramidal Neurons From Spines to Spikes via Models.docx`; the project's PDFs are published papers
and are never copied here.
