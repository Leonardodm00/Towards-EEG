# Checks of 2026-10-30 for D-115 (the answers to the Cell 7 report)

## D-115.6: does Block 10.C's search depend on the cell's transform beyond the sampling grid?

Why: Cell 15 (S3's calibration step on the pilot) reuses Cell 7's search when both were made from the same trace
under the same settings (SPEC 10.C.R). The cell's transform is not in the reuse key: S3's alignment of the pilot
(Cell 12, automatic) never equals Cell 7's design alignment exactly, and under the transform the trace moves rigidly.
This check measures how much a small transform change moves the far search's result.

Run (from the workstream directory; the pilot 529878215's SWC and its pilot table, the pipeline's own
`candidate_search.search` and `choose`, the pilot table's medians in S as B_bar and mu_hat, default configuration):

    python3 -I checks/d115_2026-10-30/search_transform_check.py run SWC PILOT_CSV far_T0.json 0 0 0 far
    python3 -I checks/d115_2026-10-30/search_transform_check.py run SWC PILOT_CSV far_T1.json 2.3 -1.7 0.1 far
    python3 -I checks/d115_2026-10-30/search_transform_check.py compare far_T0.json far_T1.json

SWC: `H16.06.010.01.03.05.03_668702901_m.swc` (Allen's reconstruction of 529878215); PILOT_CSV: `pilot_529878215.csv`
(Cell 6's table of the Colab run). T0 = Cell 7's transform (no shift, no flip, z0 0); T1 = a shift of (2.3, -1.7) px
and z0 = 0.1 um, the size of offset an automatic alignment leaves.

Result (`compare.out`): 723 far-range records under each transform; 16 reasons differ (borderline nodes among
`not_isolated:side_short`, `not_isolated:too_few`, `few_planes` and passing); `n_pixels_k` differs for 130; 19 chosen
under each, 15 in common; the other 4 are replaced by the adjacent node of the same stretch, 1.1-2.3 um away
(1417/1416, 1422/1421, 2672/2674, 4628/4629: stretch and position checked on the SWC). 2.7 min per run.

Reading: the transform changes which node of a stretch the spacing rule keeps and flips borderline verdicts, through
the pixel and plane grid; it does not change which stretches are chosen. The reused search is Cell 7's, with Cell 7's
transform, B_bar and mu_hat; each chosen node is measured and judged again on its own pilot in Cell 15.
