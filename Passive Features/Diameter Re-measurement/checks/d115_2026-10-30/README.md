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

## D-115.3: the goodness of fit's kappa_g on rendered phantoms (SPEC 10.G)

Why: a scanned plane is dropped (`poor_fit`) when its core fit's residual RMS exceeds kappa_g times its
background's noise SD on the support (1.4826 MAD of the support pixels' residuals about the fitted plane). The user
decided the rule ("Residual vs noise") and that kappa_g is set on rendered phantoms with a neighbouring dip.

Run (from the workstream directory; the pipeline's own `calibration.scan_node` on each phantom, default
configuration; seed 20261030):

    python3 -I checks/d115_2026-10-30/gof_kappa_check.py checks/d115_2026-10-30/gof_near.json 30 30 2 near
    python3 -I checks/d115_2026-10-30/gof_kappa_check.py checks/d115_2026-10-30/gof_wide.json 0 0 2 near 20
    python3 -I checks/d115_2026-10-30/gof_kappa_summary.py checks/d115_2026-10-30/gof_near.json 1.0 1.5 2.0
    python3 -I checks/d115_2026-10-30/gof_kappa_summary.py checks/d115_2026-10-30/gof_wide.json

(76 and 46 min on two sandbox cores; both runs scanned with the code before the rule, so every plane's ratio is
recorded.) Outputs: `gof_near.json`, `gof_wide.json`, `gof_near_summary.out`, `gof_wide_summary.out`.

Result:
- clean thin phantoms (30; d 0.15-0.25 um, alpha 0.15-0.4; 210 planes): g = RMS / SD_bg median 0.47, 99th percentile
  1.151, largest 1.27, unrelated to the core's amplitude (rank correlation 0.07). The rule (99th percentile, rounded
  up to 0.1) gives kappa_g = 1.2: 1.0 % of these planes dropped, on 2 nodes.
- composites (30; an untraced parallel neighbour, same d and alpha, 0.5-1.5 um away, +-0.5 um in depth): 8 rejected
  by the pilot (`crossing`), 22 scanned (154 planes): g median 0.73, 99th 5.3. At 1.2: 31 % of their planes dropped,
  on 19 of 22 nodes; 43 of 105 planes with the neighbour within 0.8 um, 3 of 35 at 0.8-1.2 um, 2 of 14 beyond.
- wide clean phantoms (20 drawn, 14 scanned; d 0.6-2.5 um, alpha 0.15-0.6, traced with Allen's 2r = 0.5 um, as wide
  as the pilot's near nodes measure; 98 planes): g median 1.24, 99th 5.3 -- 0.97 below 1 um, 1.63 from 1.6 um. A
  Gaussian core fitted within one FWHM misfits a wide tube's flat-bottomed dip in proportion to its depth: at 1.2
  about half of these planes would be dropped, and no single kappa_g separates them from the composites.
- the pilot's Colab scans of 2026-10-30 (no SD_bg stored; the pilot's background noise SD, 5.9 gl, in its place):
  kappa_g = 1.2 would drop 11 of 206 near planes and none of 43 far planes. The real near dips' RMS / 5.9 gl has
  median 0.49 and 90th percentile 0.97: they fit like the thin phantoms, not the wide ones.

Reading: kappa_g = 1.2 (PROVISIONAL) catches a neighbour within about 0.8 um in a third to a half of the planes and
leaves clean thin dips alone; if Cell 7's next run drops far more than about 5 % of the near planes (its
`goodness of fit` line), the wide-dip caveat applies and the statistic needs a depth-relative form -- the user's call.
