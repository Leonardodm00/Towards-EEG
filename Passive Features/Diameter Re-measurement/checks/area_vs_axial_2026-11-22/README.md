# Area or diameter? and how far apart Allen's nodes are (2026-11-22)

The user (16:53): "the compartment area is what determines the quality of the fit", and the plan to measure the
diameters within 250 um, sampling between successive nodes.

- `area_vs_axial.py <reconstruction.swc>` **[run, sandbox]** (the reach check's model and linearised fit shift,
  `../diameter_reach_2026-11-22/`): within path distance 250 um of soma(0.5) (5003 of 8490 dendritic sections), every
  diameter x 1.2 against the membrane alone x 1.2 (cm and g_pas: the same area load, the axial resistance unchanged).
  Fitted shifts (pulse fit; `area_vs_axial.out` also has pulse + step):

  | parameters | perturbation | dCm | dRm | dRa |
  |---|---|---|---|---|
  | Eyal 2016 060308 | diameters x 1.2 | -15 % | +14 % | +37 % |
  | Eyal 2016 060308 | membrane x 1.2 | -15 % | +16 % | +2 % |
  | matched to R_in, tau | diameters x 1.2 | -10 % | +13 % | +42 % |
  | matched to R_in, tau | membrane x 1.2 | -5 % | +11 % | +5 % |

  The area mostly sets the fitted Cm and Rm; the diameter's axial part sets Ra, and in the matched regime half of
  Cm's shift. Measuring diameters, not areas alone, is what the refinement needs. About 2 min a parameter set.
- `node_spacing.py <reconstruction.swc>`: Allen's dendritic nodes within 250 um are 1.15 um apart (median; 10-90 %
  1.13-1.24 um; max 2.55 um; 0.7 % of the segments longer than 2 um) (`node_spacing.out`).
