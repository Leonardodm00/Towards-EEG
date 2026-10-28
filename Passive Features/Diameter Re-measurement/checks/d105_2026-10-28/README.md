# Checks behind D-105 (the stripe on the pilot's own reconstruction), 2026-10-28

Not pipeline code: the runs behind SPEC Block 10.S's status of 2026-10-28 (later) and the decision log's D-105.
They compute, for the D-101 candidates of the pilot 529878215, the shadow masks and the stripe of Block 10.S on
the pilot's own traced scene -- no image is read. Run from anywhere; the scripts import the workstream's `src/`
(the directory two levels up). Inputs, both on Drive and not in the repository:

- the pilot's SWC, `MyDrive/allen_cache/H16.06.010.01.03.05.03_668702901_m.swc` (8496 segments in the scene);
- Cell 4b's pilot table, `MyDrive/diameters/pilot/pilot_529878215.csv` (2026-10-07; each node's centre, heading,
  $k^*$, $\bar B_j$, $\hat\mu_j$, $\hat d_j$ are read from it, as Cell 7 reads them).

`ids.txt`: line 1 the 7 far-set candidates, line 2 the 138 near-set candidates (the D-101 rule on the pilot table:
in S, flat at 10 deg near / 15 deg far).

| command | what it computes | time (sandbox, one core) |
|---|---|---|
| `python tau_check.py --swc SWC --pilot-csv CSV --out tau_near.jsonl $(sed -n 2p ids.txt)` | per node, at $k^*$ (and, for the far set, at the 24 planes with $5 \le \lvert k - k^*\rvert \le 16$): the stripe as built -- every predicted shadow bounds it -- at $\tau$ = 0.03, 0.3, 1 and 3 gl, and the largest $\lvert$residual$\rvert$ on the measuring line ($\pm$3 um near, $\pm$10 um far, 0.25 um steps) of the summed predicted shadow of every neighbour and the soma after a least-squares plane fitted on the stripe (1 pixel in 36) | ~3 min near, ~6 min far |
| `python cap_check.py --swc SWC --pilot-csv CSV --out cap_near.jsonl $(sed -n 2p ids.txt)` | per node: what lies over its centre (the neighbours whose predicted shadow at the centre exceeds 0.03 gl, by kind: soma, junction, broad with $\sigma_{\rm r}(\delta_{q,k}) \ge$ 2 um, sharp) and the stripe at 0.03 gl when the shadows with $\sigma_{\rm r} \ge$ 2, 3 or 5 um are left out of the masks, with the residual their summed shadow leaves on the line | ~20 min near, ~7 min far |
| `python summarize.py --tau-near tau_near.jsonl --tau-far tau_far.jsonl --cap-near cap_near.jsonl --cap-far cap_far.jsonl` | the tables below (`summarize.out`) | seconds |

The four `.jsonl` files are the runs of 2026-10-28 (evening) on the branch's code of `39ae156` (the masks of
Block 10.S after the recheck); a spot check with the committed scripts reproduced three nodes' lines exactly.
At $\tau$ = 0.03 gl `tau_check.py` agrees with a first check that called `calibration.support_masks` at $k^*$ on 44
of the near nodes: the same reasons, the pixel counts within 2 % (that check is not kept).

What `summarize.out` shows:

- As built ($\tau$ = 0.03 gl): 21 of the 138 near candidates have a usable stripe at $k^*$ (91 `too_few`, 26
  `side_short`); none of the 7 far candidates has a usable stripe in any of the far fit's planes.
- Over the node itself, 89 near candidates lie under at least one neighbour whose predicted shadow exceeds 0.03 gl
  with a blur of 2 um or more, 10 under a junction segment, 3 under a sharp neighbour; the summed predicted shadow
  at the centre has median 4.0 gl.
- $\tau$ = 0.3 / 1 / 3 gl: 92 / 117 / 127 near candidates usable; the residual on the line over the usable ones has
  median 0.24 / 0.35 / 0.45 gl (90th percentile 0.82 / 2.1 / 3.8 gl; at 0.03 gl, 0.10 and 0.13 gl). The far set:
  0 / 4 / 6 candidates with at least 6 usable fit planes, their residual on the $\pm$10 um line 3.5-14 gl per node
  (median over the planes).
- Leaving the shadows with $\sigma_{\rm r} \ge$ 2 um to the plane at 0.03 gl: 82 near candidates usable, the broad
  shadows' residual on the line median 0.37 gl.

The scale: the pilot's median background noise SD is 5.9 gl (D-101's data).

Not shown, and why:

- The images: these are the model's predicted shadows (10.I's $D_{q,k}$ with each node's own $\bar B_j$ and
  $\hat\mu_j$ applied to every neighbour, the renderer's current blur law), not grey levels read on Allen's planes.
  What a residual of a given size does to the fitted width is not computed here.
- The plane is a plain least-squares fit on a subsample of the stripe, not 10.P's trimmed fit.
- The trace's registration offsets are not applied (the SWC in the image frame, no shift or flip, as the pilot ran).
