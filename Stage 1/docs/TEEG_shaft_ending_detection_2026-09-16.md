# Shaft-ending detection in the mesh base finder

A fourth, mesh-only vote on whether a spine-labelled component is a spine.
Added 2026-09-16 after cell 1302789404 sigma 3588 passed every existing gate
and was visibly the end of a dendrite. Supplements
`TEEG_unified_spine_pipeline_HPC_handoff_2026-09-14.md`; nothing there is
retracted.

| Date | Change |
|---|---|
| 2026-09-16 | Created. `h01_spine_base` v1.0 -> v1.1, `h01_spine_area_F` v1.5 -> v1.6. Records the `find_base` defect, the angle sweep that motivated the fallback, and one open item on the 3D colouring. |

## 1. The defect

`find_base` took the base to be the first cross-section station whose loop
does NOT touch the cutout box, on the reasoning that the dendrite traverses
the ROI and therefore every section inside it reaches a box face.

That fails when the section planes never cut the dendrite lengthwise. The
planes are perpendicular to the SPINE axis, so a component COLLINEAR with its
dendrite -- a continuation or a terminal ending -- is cut transversely, as a
disc, and the disc never reaches the box. Station 0, the base node itself, was
then already "clear", so:

- the base was placed at s = 0;
- every triangle of the component counted as spine;
- `A_beyond` silently became the whole mesh;
- no error, no flag, nothing in the summary.

On sigma 3588 the profile printed `measured base 0 nm` and the run looked
clean. The component is a full-calibre dendrite tip: the section's equivalent
radius starts at 250 nm and holds for 200 nm, against a skeleton `r_shaft` of
199 nm, then declines smoothly to a 145 nm minimum at 0.58 um and rises to
175 nm. A terminal swelling, not a head on a neck.

## 2. Why the skeleton votes did not catch it

Calibre, collinearity and the taper test all read the skeleton only. A
terminal swelling has a distal radius maximum exactly as a head does, so the
taper test votes SPINE on it. NOT VERIFIED for sigma 3588 specifically: the
run that produced it used the two-observable rule, so its rho/cos/taper
verdicts were never printed. The reasoning is the mechanism, not a measurement
of that component.

The mesh has an observable the skeleton does not: whether the union
cross-sections ever separate the component from the dendrite.

## 3. The fix

Three parts, in `h01_spine_base.find_base`.

**Anchor after the last shaft station, not on the first clear one.** If any
station touches the box, the base is the first clear station AFTER the last
touching one. This alone removes the station-0 default.

**Area-drop fallback when no station touches the box.** Necessary because box
contact also fails for genuine spines leaving at a shallow angle. Measured on
phantoms, shaft r = 300 nm, neck r = 70 nm, head r = 250 nm, departure angle
theta from the shaft AXIS:

| theta | stations touching the box | max consecutive area drop |
|---|---|---|
| 90 deg | 12 | 13.2x |
| 70 deg | 14 | (not recorded) |
| 60 deg | 5 | 33.6x |
| 45 deg | 0 | 26.5x |
| 35 deg | 0 | 23.2x |
| 25 deg | 0 | 1.4x |
| terminal ending | 0 | 1.4x |

So below ~45 deg box contact is gone but the NECK still shows as a large drop
in section area between consecutive stations, while a taper declines about
1.4x. `DROP_MIN = 3.0` separates them with a wide margin on both sides. The
base is then the first station whose preceding station is >= 3x larger.

**`ShaftTerminates` when neither rule fires.** A subclass of `BaseError`.
Raised rather than guessed.

## 4. What it costs

Below about 30 deg a genuine spine has no drop either -- the plane never
separates neck from dendrite -- so it is flagged too. A false positive, and
an acknowledged one: flagged components are kappa-filled from their skeleton
area and carry the verdict, so nothing is silently miscounted. They are
visible in the union 3D view for a human call.

## 5. New fields

Per spine, in the ledger and `cell{id}_spines.csv`:

| field | meaning |
|---|---|
| `base_method` | `box_contact` or `area_drop` -- which rule placed the base |
| `base_verdict` | `ok`, `shaft_terminates`, or `error` |
| `shaft_reaches_box` | does the connected shaft context touch a cutout face (mask-level, no mesh) |

Per cell, in `spine_area_F_summary.csv`:
`n_shaft_terminates`, `n_base_by_area_drop`, `n_shaft_not_reaching_box`.

A flagged component keeps `A_beyond = NaN`, so it never enters the measured
side of kappa.

`h01_spine_base.shaft_reaches_box(roi)` is a standalone mask-level test, cheap
enough to run on every spine in the campaign without building a union mesh.

## 6. Verification

`smoke_test_h01_spine_area_F.py`, T18, on a new terminating-dendrite phantom
(a smoothly tapering dendrite ending in a bulb, the spine-labelled component
being its tip):

- `measure_base` raises `ShaftTerminates` at 1.38x, does NOT return station 0;
- `shaft_reaches_box` distinguishes the shaft crossing the box from the
  component's sections never doing so -- two different facts that an earlier
  version of this test conflated;
- the batch records the verdict and keeps the spine `ok`, with `A_beyond` NaN;
- the radial phantom still finds its base at 305 nm by `box_contact`;
- a 45 deg spine gets its base by `area_drop` at 26.5x, near 506 nm, against
  r/sin(45) = 424 nm.

Suite: 119 checks, 0 failed, 2 skipped (plotly, absent in the sandbox). Run
twice, the second time in a clean copy of the folder.

NOT verified: the fix has never run on a real cell. The pilot 30 of cell
1302789404 predate it and need re-measuring -- `require_keys` will do that
automatically once `base_method` is added to the list.

## 7. Open item

**The union 3D figure still colours a flagged component as if it were a
spine.** When there is no base plane the cell falls back to the CYLINDER rule
for colouring, so on sigma 3588 the legend reads `spine 0.853 um2`,
`rind 0.031 um2` -- everything beyond r_shaft = 199 nm drawn blue, for an
object the profile has just declared not a spine. The measurement is
unaffected (`A_beyond` is NaN and the summary counts it), but the figure is
the artefact a human uses to judge the call, so it should not contradict the
verdict.

Fix, roughly ten lines in `h01_spine_area_F_figures.union_mesh_3d` and the
union-view cell: when `base_verdict != "ok"`, do not split spine from rind at
all -- label the whole component one region named for the verdict, and put the
verdict in the figure title.

## 8. Consequence for the unified pipeline

This vote needs the segmentation, so it can only be cast at P2, after P1 has
already fixed the partition. **The mesh can veto a spine the skeleton
accepted**, and on sigma 3588 it did. The P0 decision (route every partition
call site through `demote_shaft_continuations_three_vote`) does not change,
but P1's partition is no longer the last word on what counts as a spine, and
P4 should reconcile the two rather than assume they agree.
