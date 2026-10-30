# docs/project on `sci/diameter-pipeline` -- copies made by the diameter chat (D-097)

D-097 (the user, 2026-10-28): every document a chat writes is also committed to this repository, so that any chat
can read it. `main`'s `docs/project/` holds the copies of the claude.ai project's documents and the rules
(`docs/project/README.md` there): the project copy is the source of truth, and a chat that writes a project document
writes the repository copy in the same turn, byte for byte.

Where a workstream on a branch of its own puts its copies is D-097's open point (d); `main`'s README proposes `main`.
Until the user settles it, the diameter chat commits the copies of the project documents it writes here, on its own
branch, at the paths they have in project knowledge:

| Path here | In project knowledge | Copied |
|---|---|---|
| `claude/TEEG_diameter_HPC_paths_reference.md` | `claude/` | 2026-10-30, its v18 -- D-115.4 to D-115.6 on the remote (v17: Cell 7's Colab run of 2026-10-30, D-112's code); **only here**: project knowledge refused the write (its tool needs a whole-file read in one call, beyond the session's read limit), so project knowledge still holds v16 (v16: D-112: the near set searched over the whole cell too, in Cell 7, Cell 15 and the davinci array; v15: D-108, D-109, tau 0.3 gl, Cell 7's far search and `far_search_<sp>.csv`; v14: D-105, the recheck, Cell 7's knob file, Cell 15's merged file; v13: D-101, the stripe; v12: D-099, the survey jobs, section 2b; v11: D-098 on the remote; first copied at v10) |
| `TEEG_decisions_and_ideas_log.md` | root | 2026-10-30, after the noise-fit chat's D-114 (08:21), with the status notes of Cell 7's Colab run of 2026-10-30 (D-088, D-108, D-109, D-112) and **D-115** (the answers to its report), then D-115's status and D-115.6's corrected reading (the row 2026-10-30 (later, 3)); **only here**: project knowledge refused the write (the same limit), so it holds D-114 without these notes (v31: D-112, D-113, on the noise-fit chat's D-110, D-111 and its rows (later, 4) and (later, 5); v30: D-108, D-109; v29: D-105; v28: D-101; v26: D-099; v25: D-098; v24: the status notes on D-085, D-089, D-090) |

A copy here that differs from project knowledge, or from `main`'s copy of the same document, is stale, not a fork:
re-read the project document and merge.
