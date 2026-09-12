# Spine-area campaign on Da-Vinci

Calibrated dendritic-spine membrane area from the H01 segmentation, substituted
into Stage 1's phi table, giving a mesh-based `F` per cell.

`h01_spine_area_F v1.4` | `run_spine_area_F v1.0` | `merge_spine_area_F v1.0`

---

## 1. What is in this bundle, and what is missing

**Included** (the full import closure of the runner, minus the figure stack):

| file | role |
|---|---|
| `h01_spine_area_F.py` | area, rind, base frustum, kappa, F |
| `h01_spine_batch.py` | the analysis mesh (14 Taubin iterations) |
| `h01_spine_roi.py` | per-spine cutout, Voronoi split, bridging |
| `h01_spine_geometry.py` | smoothing, skeleton helpers |
| `h01_area_calibration.py` | g table load / fold / lookup |
| `h01_radius_census.py` | imported by `h01_spine_roi` |
| `sma_run.py`, `s0_ingest.py`, `shaft_continuation.py` | Stage 0/1 glue |
| `run_spine_area_F.py` | one array task |
| `merge_spine_area_F.py` | union the shards, assemble F |
| `spine_area_F.pbs` | PBS Pro array job |
| `smoke_test_h01_spine_area_F.py` | 91 checks (2 skip without the figure module) |
| `smoke_test_hpc_runner.py` | 20 checks, runner + merger end to end |
| `g_table_cyl_2deg.npz` + `.json` | the v7 cylinder calibration |

**NOT included -- you must copy these from the Stage 1 Drive folder into
`stage1/`:**

```
spine_density.py  spine_labeller.py  spine_geometry.py  morphology_exporter.py
```

They are Stage 1's own modules. The attribution gate exists precisely to check
this code against the real `spine_density`, so a stub would defeat the point.
`run_spine_area_F.py` refuses to start if they are absent or if the label
vocabulary falls back to `sma_run`'s literals.

Figures (`h01_spine_area_F_figures.py`, `h01_spine_roi_figures.py`) are
deliberately absent: they need matplotlib, plotly and IPython, and the campaign
draws none. Pilot QC figures stay in Colab.

---

## 2. Layout on the cluster

```
TEEG/Spines/
  h01_code/                     <- this bundle (the git repo)
      *.py  *.pbs  README.md
      stage1/                   <- the four Stage 1 modules (see above)
      logs/                     <- mkdir this; PBS writes here
  h01/                          <- the campaign root, NOT in git
      neurons/neuron_1302789404.csv   (and the other three cells)
      g_table_cyl_2deg.npz + .json
      out/                      <- created automatically
```

`--root` points at `h01/`, the job's `CODE` at `h01_code/`. Keeping data out of
the repo keeps `git status` clean between runs.

---

## 3. Before submitting anything

```
cd h01_code && python3 smoke_test_h01_spine_area_F.py && python3 smoke_test_hpc_runner.py
```

Expect `91 checks passed, 0 failed` (or 82 with 2 skipped if the figure module
is absent) and `20 checks passed, 0 failed`. Both are fully offline: no
network, no Drive, no Stage 1 folder needed.

Then a dry run, which prepares, shards and reports without touching the
network or writing a ledger:

```
python3 run_spine_area_F.py --root ../h01 --cell 1302789404 --task 0 --ntasks 40 --dry-run
```

It prints the g-table hash and axes, the spine count, and the parameter
fingerprint. Every shard must print the SAME fingerprint; the merge refuses to
combine shards that disagree.

---

## 4. Staged submission

**Stage 1 -- one task, 300 spines.** Confirms the network path and gives the
per-spine rate:

```
mkdir -p logs && qsub -v CELL=1302789404,SUBSET=300,NTASKS=1 spine_area_F.pbs
```

Change `#PBS -J 0-39` to `#PBS -J 0-0` for this, or submit without `-J`.

**Stage 2 -- the full cell, 40 tasks:**

```
qsub -v CELL=1302789404 spine_area_F.pbs
```

**Stage 3 -- merge:**

```
python3 merge_spine_area_F.py --root ../h01 --cell 1302789404 --ntasks 40
```

Add `--allow-missing` only after deciding an incomplete answer is acceptable;
by default a dead shard stops the merge rather than silently biasing F.

**Stage 4 -- the other cells:** same two commands with a different `CELL`.

`NTASKS` in the `qsub -v` list must equal the array width in the `#PBS -J`
line. Change both together, or the shards will not tile the spine list.

---

## 5. Resuming

Every shard writes its ledger atomically every 25 spines. Re-submitting the
same task skips spines already measured, so a walltime kill costs at most 25
spines. Records written by an older module version are re-measured
automatically (`require_keys`), so the junction columns cannot come back
half-filled.

---

## 6. Outputs, for Stage 1

In `h01/out/`:

| file | content |
|---|---|
| `neuron_{id}_phi_mesh.csv` | Stage 1's phi schema, `spine_area_um2` replaced; `spine_area_skel_um2` keeps the original |
| `cell{id}_spines.csv` | per spine: A_mesh, A_rind, A_skel, base frustum, class fractions |
| `cell{id}_kappa_function.csv` | kappa in quantile bins of A_skel |
| `spine_area_F_summary.csv` | one row per cell: F variants, kappa variants, gate, coverage |

Read `F_lit_mesh`. `F_lit_mesh_norind` removes the Voronoi rind, which the
shaft frustum double-counts; `F_lit_skel` is the Stage 1 baseline on the same
partition. When `F_lit_mesh` and `F_lit_mesh_skelfill` converge, coverage is
high enough that F is measured rather than extrapolated.

---

## 7. Options worth knowing

| flag | default | note |
|---|---|---|
| `--subset N` | all spines | size-stratified sample of the whole cell, then sharded |
| `--min-spine-value` | off | demote spines below `--min-spine-metric` to shaft, before phi |
| `--axial-window-nm` | none | cap on how far from the base a triangle may count as rind |
| `--roi-cache DIR` | none | reuse CELL 12 ROIs instead of refetching |
| `--kappa-min-per-bin` | 25 (merge) | below ~25 per bin the kappa curve fits noise |
| `--no-shaft-stub-fix` | off | must match how Stage 1 built its own phi |
