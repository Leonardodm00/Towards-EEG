"""Arithmetic behind the literature-based range for the soma's depth semi-axis c (2026-11-25, D-248).
Standalone check, not pipeline code. Run: python3 literature_c_range.py
Inputs, each read in a full text this session:
  Benavides-Piccione et al. 2024 (Cereb Cortex 34:bhae180, Table 2 and Results): human CA1 soma volume
    5041 +- 273.1 um^3 and area of the maximum perimeter 350 +- 9.7 um^2; mouse CA1 1288 +- 97.52 um^3 and
    137 +- 3.0 um^2; human temporal L3a (BA20, BA21) area of the maximum perimeter 244 +- 5.1 and 253 +- 7.5 um^2
    (mean +- SEM). Their neocortical soma VOLUMES are only in Fig. 3A / Supplementary Table 1A: not read.
  Mohan et al. 2015 (Cereb Cortex 26:3608): 350 um slices, 140.7 um after mounting (median), z-shrinkage
    63 +- 10 %.
  The pilot (handoff 14.2, data inspected in an earlier chat): silhouette at the sharp planes 111-135 um^2;
    the stems leave the soma between -4.5 and +2.4 um of the traced depth.
Assumptions (mine): an ellipsoid whose largest cross-section lies in the image plane, so A = pi a b and
V = 4/3 pi a b c, hence c = 3 V / (4 A); ratios of group means, not means of per-cell ratios; the soma shrinks
in depth like the slice."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.environ.get("TEEG_W", "/home/claude/towards-eeg/Passive Features/Diameter Re-measurement"), "src"))
from allen_diameter.analysis.soma_fit import ellipsoid_area

lit = {"human CA1": (5041.0, 350.0), "mouse CA1": (1288.0, 137.0)}
for k, (V, A) in lit.items():
    c = 3 * V / (4 * A); r = np.sqrt(A / np.pi)
    print("%-9s c = 3V/4A = %.2f um; in-plane radius sqrt(A/pi) = %.2f um; c/r = %.3f; sphere of r: V = %.0f um^3 (reported %.0f)"
          % (k, c, r, c / r, 4 / 3 * np.pi * r ** 3, V))
for k, A in (("BA20", 244.0), ("BA21", 253.0)):
    print("%s in-plane radius sqrt(A/pi) = %.2f um" % (k, np.sqrt(A / np.pi)))
ratio = (0.86, 1.07)          # c/r: 0.84 x 1.02 (if the confocal z distension of the same lab applies) .. 1.07
r_pilot = np.sqrt(np.array([111.0, 135.0]) / np.pi)
print("pilot in-plane radius %.2f - %.2f um" % tuple(r_pilot))
c_live = (ratio[0] * r_pilot[0], ratio[1] * r_pilot[1])
print("living c = (c/r) x r: %.2f - %.2f um" % c_live)
k = (1 - 0.63 - 0.10, 1 - 0.63 + 0.10)        # mounted / original thickness, Mohan's 63 +- 10 %
print("mounted/original thickness %.2f - %.2f (central %.2f; medians 140.7/350 = %.3f)" % (k[0], k[1], 0.37, 140.7 / 350))
print("stack c = living c x (mounted/original): %.2f - %.2f um (central %.2f - %.2f)"
      % (c_live[0] * k[0], c_live[1] * k[1], c_live[0] * 0.37, c_live[1] * 0.40))
print("stems' exit span 6.9 um -> c <= %.2f um if they leave at the surface" % (6.9 / 2))
for r in r_pilot:
    print("living S at r = %.2f: c/r 0.86 -> %.0f, 1.00 -> %.0f, 1.07 -> %.0f um^2 (Allen's sphere 447.9)"
          % (r, ellipsoid_area(r, r, 0.86 * r), 4 * np.pi * r * r, ellipsoid_area(r, r, 1.07 * r)))
