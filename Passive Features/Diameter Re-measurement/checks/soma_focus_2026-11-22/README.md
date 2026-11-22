# Check: does a dendrite in focus inside the soma's disk outweigh the soma's own focus? (2026-11-22, D-211)

Block 44's first real run picked plane k = 93 of the pilot 529878215, where a basal dendrite is in sharp focus inside
the traced soma's disk, not the soma (the user sees the soma begin at k = 107). The measure was the variance over the
disk of the Laplacian of Gaussian (1 px); D-211 made the soma's plane a choice by eye and kept the measure only as a
diagnostic (the fine-detail energy).

`check_detail_energy.py` asks whether the failure is the measure's for any soma. It renders a brightfield stack of a
smooth absorbing sphere (radius 6 um, mu 0.258 /um, Beer-Lambert, Gaussian defocus of SD sqrt(0.1^2 + (0.5 delta)^2)
um per slab) with a horizontal dendrite 6.7 um below the sphere's centre, crossing its disk, and computes the measure
on every plane.

    python3 check_detail_energy.py              # dendrite 0.8 um
    python3 check_detail_energy.py d_um=1.6     # any CFG key can be overridden

Result **[run, sandbox]** (outputs `check_detail_energy_d*.out`): for a dendrite of 0.8, 1.6 or 2.4 um the measure
peaks at the sphere's centre plane every time (about 1.2 there, 0.17-0.25 at the dendrite's plane). A sharp-edged
sphere would have been found; the pilot's soma was not, so its planes carry less fine detail than its in-focus
neighbour. Why is not established. Standalone check, not pipeline code; about 40 s a run.
