# Soma area: what the brightfield stack measures, and how much it matters (2026-11-22; D-210, D-211)

- `../soma_focus_2026-11-22/sphere_contours.py` **[run, sandbox]**: on a simulated brightfield stack of a 6 um sphere
  the half-depth outline stays at about 6 um on every plane, even 8.4 um from the centre where the section is empty:
  a plane shows the blurred silhouette of the whole soma, not its section. Outlines give the width, not the height.
- `outline_rules_prototype.py` (exploration, not pipeline code) on the pilot's block (`soma_529878215_block.npz`, the
  user's Drive): the steepest-rise and the half-depth outlines agree on k = 111-121 (111-135 um2 per plane;
  equivalent radius 5.95-6.55 um against Allen's 5.97) and trace only the blur above k ~ 123
  (`outline_rules_prototype_529878215.png`).
- The stems' first nodes leave the soma between -4.5 and +2.4 um of its traced depth (6.9 um; the SWC).
- `shape_areas.py`: spheroids within those bounds have 327-539 um2 of membrane, 0.73-1.20 x Allen's sphere (448 um2);
  closed form checked against a numerical surface integral (`shape_areas.out`).
- `soma_area_sensitivity.py <swc> <out> eyal|matched [--refit S]` (the reach check's model,
  `../diameter_reach_2026-11-22/`): the soma's membrane area times s, the linearised shift of the fitted Cm, Rm, Ra.
  Eyal's parameters: the soma is 1.45 % of the membrane the fit charges; s 0.5-2 moves Cm and Rm by at most 5 %, Ra by
  at most 10 %. Parameters matched to the cell's R_in and tau (Cm 5.39 uF/cm2, Rm 3.8 kOhm cm2): s 0.75 moves Cm by +7 %
  (pulse fit) to +17 % (pulse + step), s 1.26 by -9 % to -13 %. A full refit at s 1.5 agrees with the linear step
  (`run_eyal.out`, `run_matched.out`, `soma_area_*.json`). About 5 min (eyal with the refit), 1.5 min (matched).
