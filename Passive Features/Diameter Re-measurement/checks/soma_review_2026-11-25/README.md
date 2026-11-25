# Soma fit: checks prompted by the independent review, 2026-11-25

Standalone checks (not pipeline code) run while discussing the review handoff
`TEEG_soma_dendrite_approach_review_handoff_2026-11-25.md` (the user's upload; not in the repository) with the user.
Simulated data only. Python 3 with the repository's `src/` (set `TEEG_W` to the workstream folder if moved).

`sensitivity.py OUT.json` -- three simulated somas (in-plane axes 7.0 x 5.6 um with c 2.5 and 4.5 um; 8.08 x 6.52 um
with c 7.26 um; 2 mu c = 0.87; centre 1.8 um above the traced depth), the pilot's acquisition and **the pilot's own
blur table** (0.080, 0.1536, 0.3924, 0.5620 um at 0, 0.28, 0.56, 0.84 um; D-170), far slope 0.79 held, 35 planes,
background 110 gl, noise SD 3 gl. For five fit configurations (region radius 5.97, 10 or 13 um; each plane's brightness
B_k re-solved, as today, or held at the value measured outside the soma) it gives, at the truth: the noise SE of each
parameter from the residuals' Jacobian; and the bias after one Gauss-Newton step when the data come from a world the
model does not hold (far slope 0.95 or 0.60; in-focus blur SD 0.258 um; a black level of 10 gl). The bias is
linearised: a sensitivity, not a fit's result. Cross-check of the linearisation: c 4.5 um, 13 um region, B_k re-solved,
slope 0.95 gives c -17.9 %, S -7.6 %; the full fit of handoff 14.4 gave -16.5 % and -7.5 %.

Results (`sensitivity.out`; run in the sandbox, 2026-11-25):

| Question | Result |
|---|---|
| Noise | SE of c 0.06-0.35 %, of S under 0.09 %, in every configuration: noise is not what limits the soma's area |
| The held far slope | world 0.95 against 0.79: c -4.5 to -59 %, S -1.3 to -18 %; world 0.60: c +7.9 to +90 %, S +2.9 to +28 %; worst for the flat soma (c 2.5 um), least on the D-241 region with B_k re-solved |
| Holding B_k (the review's proposal) | SE of c smaller by a factor 1.0-1.7; the slope's bias about twice as large on the 5.97 um region (c 4.5 um: -22.2 against -10.5 %), unchanged on 10-13 um regions |
| The region without the soma's edge (8.08 x 6.52 um soma, 5.97 um disc) | a, b still identified: SE 0.11 and 0.09 %, largest bias over the four worlds 1.4 and 2.1 % |
| In-plane axes in general | largest bias of a, b over the four worlds and all configurations 3.9 % |
| In-focus blur 0.258 against 0.080 um | c at most -4.0 %, S at most -1.3 % |
| Black level 10 gl | c at most 1.9 %, S at most 1.0 % |
| A thresholded outline for a, b | the in-focus plane's half-depth contour encloses 0.89, 0.94, 0.97 of pi a b at c 2.5, 4.5, 7.26 um: it depends on c |

What it does not show: anything about the pilot's stack; the effect of structure the model lacks (processes crossing
the soma, a non-Gaussian far blur, phase contrast, a tilted illumination cone).

`figures_revised.py sensitivity.json OUT_DIR` -- the three figures of the explanation of the revised approach (2026-11-25, 14:12):
fig5 the approach (schematic), fig6 the bias of c and of a against the true far slope (from `sensitivity.json`), fig7 S against
c with an illustrative carried range. The revised approach is proposed, not decided (handoff section 9).
