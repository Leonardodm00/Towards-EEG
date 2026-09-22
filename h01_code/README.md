# H01 campaign on Da-Vinci: P1 export and P2 spine area

Two array jobs live here. **P1** (`run_p1_export.py`, section 8) partitions,
exports, aligns and gates one cell per array task, headless, from a manifest
of (cell, layer, exc/inh). **P2** (`run_spine_area_F.py`, sections 3-7) is the
calibrated dendritic-spine membrane area from the H01 segmentation, substituted
into Stage 1's phi table, giving a mesh-based `F` per cell. **`campaign.pbs`
(section 11) runs P1 -> P2 -> P3 for every cell of a population from ONE
submission** (decision D-004): that is the way to launch a population; the
two stage scripts remain for re-running one stage by hand.

`run_p1_export v1.0` (rev 1.0.1) | `p1_spine_stats v1.0` | `p1_hoc_audit v1.0` | `build_p1_manifest v1.1` | `soma_census v1.0`
`h01_spine_area_F v1.7` | `run_spine_area_F v1.3` | `merge_spine_area_F v1.1` | `h01_spine_base v1.2`

---

## 1. What is in this bundle, and what is missing

**Included** (the full import closure of both runners, minus the figure stack):

| file | role |
|---|---|
| `run_p1_export.py` | P1: one array task = one cell through `alignment.align_and_export`, plus the controls, audits and tables of section 8 |
| `build_p1_manifest.py` | writes one manifest per population, checked against the layer's alignment bank, whose directory it searches (8.1) |
| `p1_spine_stats.py` | the per-spine-node and per-spine tables (handoff section 4.1) |
| `p1_hoc_audit.py` | structural `.hoc` audit, NEURON validation subprocess, quarantine (ports of notebook CELL 6a) |
| `passive_params.csv` | cm, Ra, gate Rm per (layer, cell_type); **L2/L3 exc filled, every inh row blank** |
| `passive_params_inh_SST.csv`, `passive_params_inh_PVVIP.csv` | the two interneuron variants (cm 1.0 / 2.0), inh-only; decision D-003, section 8.2 |
| `p1_export.pbs` | PBS Pro array job for P1, one submission per population |
| `soma_census.py` | soma-radius census over the campaign skeletons, without running P1: how many cells carry a soma that passes `soma_enforce`'s completeness gate (section 9) |
| `smoke_test_soma_census.py` | 25 checks, incl. the two cells `soma_enforce` documents by name |
| `p1_plots.py` | v1.2. figures from a P1 tree: aligned skeleton before/after pruning, raw skeleton vs d_lambda compartments with exc/inh synapses, spine and arbour statistics per cell, population summary, paired SST vs PV/VIP (section 10) |
| `smoke_test_p1_plots.py` | 50 checks: loaders and transforms against a real fixture export, every figure rendered on Agg, the LFPy path through a fake cell |
| `smoke_test_p1_export.py` | 113 checks: bank discovery, the passive variants, tables on a hand-labelled fixture, the real export end to end, audit, refusals, the job script |
| `h01_spine_area_F.py` | area, rind, base frustum, kappa, F |
| `h01_spine_batch.py` | the analysis mesh (14 Taubin iterations) |
| `h01_spine_roi.py` | per-spine cutout, Voronoi split, bridging |
| `h01_spine_geometry.py` | smoothing, skeleton helpers |
| `h01_area_calibration.py` | g table load / fold / lookup |
| `h01_radius_census.py` | imported by `h01_spine_roi` |
| `sma_run.py`, `s0_ingest.py`, `shaft_continuation.py` | Stage 0/1 glue |
| `run_spine_area_F.py` | one array task |
| `merge_spine_area_F.py` | union the shards, assemble F |
| `campaign.pbs` | **the population launcher**: one PBS array runs P1, P2 and P3 per cell (index = row x shard); D-004, section 11 |
| `campaign_join.py` | one table per (cell, passive tree) with `F_lit_skel_p1` AND `F_lit_deliverable` side by side, never confused (section 11.3) |
| `smoke_test_campaign_join.py` | 22 checks: the join, NaN where no P3, both inh trees to one mesh F, refusals |
| `spine_area_F.pbs` | PBS Pro array job; reads the g table from `h01_code`, activation block copied from `run_smoke_tests.sh` |
| `probe_net.pbs` | compute-node network diagnostic (absolute interpreter path) |
| `run_smoke_tests.sh` | runs every `smoke_test_*.py`, on the login node or via `qsub` |
| `stage1_link.sh` | assembles `stage1/` as symlinks to the canonical Stage 1 modules |
| `smoke_test_h01_spine_area_F.py` | 112 checks (4 skip without the figure module) |
| `smoke_test_hpc_runner.py` | 98 checks: runner + merger end to end, section B: the job scripts parsed and run against a fixture with a stub conda, section C: `campaign.pbs` at non-trivial (N, S) with the argv of all three stages asserted |
| `smoke_test_p0_partition.py`, `smoke_test_p3_assemble.py` | 9 and 14 checks |
| `g_table_cyl_2deg.npz` + `.json` | the v7 cylinder calibration |

**Not copied -- linked.** The Stage 1 modules (`spine_density`,
`spine_labeller`, `morphology_exporter`, `alignment`, `hoc_qc`,
`spine_geometry`, ...) live in `towards_eeg/structure/` and `Stage 1/` of this
repo; `stage1_link.sh` fills `stage1/` with symlinks to them (once per clone,
and after any `git pull` that adds a module; `run_smoke_tests.sh` does it for
you). The attribution gate exists precisely to check this code against the
real `spine_density`, so a copy or a stub would defeat the point. Both runners
refuse to start if they are absent; `run_spine_area_F.py` also refuses if the
label vocabulary falls back to `sma_run`'s literals.

Figures (`h01_spine_area_F_figures.py`, `h01_spine_roi_figures.py`) are
deliberately absent: they need matplotlib, plotly and IPython, and the campaign
draws none. Pilot QC figures stay in Colab.

---

## 2. Layout on the cluster

```
TEEG/Towards-EEG/                 <- the git repo, branch main
  h01_code/                       <- H01_CODE: this directory
      *.py  *.pbs  *.sh  README.md
      g_table_cyl_2deg.npz + .json   <- the calibration table lives HERE
      stage1/                     <- symlink farm, built by stage1_link.sh (gitignored)
      logs/                       <- mkdir this; PBS writes here
  h01/                            <- H01_ROOT: the campaign root, gitignored
      neurons/neuron_<id>.csv
      synapses/neuron_<id>_synapses.csv
      neurons/alignment_metadata_L{2,3,4,5,6}.csv  <- the banks live beside
                                      the skeletons (the extraction script
                                      writes them there); searched, not assumed
      p1/manifests/<layer>_<type>.csv   <- written by build_p1_manifest.py
      p1/<cell_id>/               <- P1 output, one directory per cell
      p1/p1_summary.csv           <- written by run_p1_export.py --summarise
      out/                        <- P2 output, created by the first shard
```

The path knobs are `H01_ROOT` and `H01_CODE` (`qsub -v H01_ROOT=...`); the bare
`ROOT` / `CODE` are names the login shell exports for another project and every
script here reports and ignores them. `spine_area_F.pbs` passes
`--g-table "$H01_CODE/g_table_cyl_2deg.npz"` explicitly, because the runner's
own default (`<root>/g_table_cyl_2deg.npz`) points into `h01/`, where the table
is not. Keeping data out of the repo keeps `git status` clean between runs.

---

## 3. Before submitting anything

```
cd h01_code && bash run_smoke_tests.sh
```

Expect `passed 8/8` and `ALL SUITES PASSED`. It activates `spine_env`
(`H01_ENV=other_env` to choose another; a stale `ENV_NAME` exported by the
login shell is reported and ignored, like `CODE` and `ROOT`), checks and
repairs the `stage1/` symlink farm, then runs every `smoke_test_*.py`. All
suites are offline: no network, no bucket. The report header must say
`env    spine_env`; if it names another env, the wrong interpreter is running
and skimage / cloudvolume will be missing.

Then a dry run, which prepares, shards and reports without touching the
network or writing a ledger:

```
python3 run_spine_area_F.py --root ../h01 --stage1-dir stage1 --g-table g_table_cyl_2deg.npz --cell 1302789404 --task 0 --ntasks 40 --dry-run
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

---

## 8. P1: partition, export, align and gate (one cell per array task)

`run_p1_export.py` is a wrapper over `alignment.align_and_export`, which is
already the whole single-cell pipeline (export steps 1-8 with the three-vote
demotion at 4b, alignment at 9, the propagation gate at 12 with staging, the
synapse snap and the C-09 emitter at 14). The driver supplies what the Colab
notebook supplies by hand and a cluster cannot: which cell (a manifest row by
array index), which constants (cm, Ra, gate Rm from `passive_params.csv` by
(layer, cell_type)), and the controls around the call: rigidity control,
structural `.hoc` audit with quarantine, arbour angle, the section-4.1 spine
tables, a parameter fingerprint and one JSON record per cell.

Standing decisions are baked in and fingerprinted: `demote_continuations=True`
at the default thresholds, `cap_tips=True, cap_h_um=0.1` (the notebook's
`CAP_TIPS=False` is superseded), propagation gate ON, synapse redirect ON
whenever the synapse CSV exists, cm shaft-referenced, F never folded in.

**8.1 The label, and the bank.** A cell carries only `(layer, cell_type)`
with `layer in {L2..L6}` and `cell_type in {exc, inh}`. Nothing finer exists
on the cluster. The campaign is ten populations, one manifest and one array
job each, and each layer has its own bank `alignment_metadata_L<n>.csv`
(column `neuron_id`).

The bank's directory is searched, not assumed: `<root>/neurons/`, then
`<root>/alignment/`, then `<root>/`. `neurons/` comes first because that is
where the extraction script puts it -- `Alignment Metadata/Usage.py` saves
into its own `input_dir`, the folder of `neuron_<id>.csv`. `--bank <file>` or
`--bank-dir <dir>` overrides the search; the same filename in two searched
places is a warning naming both, with the first used. Whatever path is found
is what the manifest's `alignment_metadata` column carries, so the driver
follows the builder rather than a second hard-coded guess.

The bank is used for two different things, and only one of them is this
label: `build_p1_manifest.py` reads the `neuron_id` column to decide which
layer a cell belongs to, while `alignment.neighbourhood_rotation` aligns each
exported cell by the rotation-group mean of its **k = 3 spatially nearest
bank rows** (`--k-neighbors`). A cell therefore does not need to be in the
bank to be exported -- it needs bank members near it in space.

**8.2 The passive tables, and what cm is for here.** `cm` and `Ra` are not
written into the `.hoc` -- the exported geometry is passive-free. They fix
the **lfpy_idx segmentation** (nseg per section by the lambda_f rule) and
therefore every compartment index in `mapped_synapses.csv`; the simulation
must rebuild each cell with the same `(cm, Ra, lambda_f, d_lambda)` or the
mapped synapses land in the wrong compartments. `Rm_qc` is the gate's Rm
only.

Three tables, one row per `(layer, cell_type)` with columns `cm_uF_cm2,
Ra_ohm_cm, Rm_qc_ohm_cm2, cm_reference, source, provenance`:

| table | rows filled | values |
|---|---|---|
| `passive_params.csv` | `L2 exc`, `L3 exc` | cm 0.50, Ra 268.5, Rm_qc 24000 (Eyal 2016 / Deitcher 2017); **every `inh` row deliberately blank** |
| `passive_params_inh_SST.csv` | `L2 inh`, `L3 inh` | cm **1.0**, Ra 100, Rm_qc 43103 (Yao 2022 SST) |
| `passive_params_inh_PVVIP.csv` | `L2 inh`, `L3 inh` | cm **2.0**, Ra 100, Rm_qc 38760 (Yao 2022 PV/VIP; VIP's g_pas) |

A blank row is **refused before any export**, so a population cannot run on a
placeholder, and the variant tables are **inh-only**, so an exc manifest
pointed at one is refused with `no row for (L2, exc)` rather than exported
into the wrong tree.

**Why two inh tables (decision D-003).** The label is two-level, so every
interneuron in a layer gets one `cm`, yet Yao 2022's SST model has cm 1.0 and
its PV/VIP models 2.0 -- and the authors of the L5 analogue describe their
PV cm=2 as a fit compensation for dendritic-diameter errors, not a
measurement. Rather than choose now, **interneurons are exported under both
sets, into separate trees, and the choice is made downstream.** The record
and `p1_summary.csv` name the table (`passive_table`); the fingerprint hashes
the values, so the two trees never resume into each other, and an
identically-filled table under another name does resume.

**8.3 Build the manifest** (one per population; paths inside it are relative
to `H01_ROOT`, POSIX separators, so it travels between laptop and cluster):

```
python3 build_p1_manifest.py --root ../h01 --layer L3 --cell-type exc \
    --ids 1302789404,1317492596,1333261412,1376890291 \
    --out ../h01/p1/manifests/L3_exc.csv
```

It prints the bank it used and where it found it, then the `#PBS -J` width.
It refuses an id with no `neurons/neuron_<id>.csv`, and an id absent from the
layer's bank unless `--allow-unbanked` (then `layer_source` is `manifest`, not
`bank`). A missing synapse CSV is allowed: the column is left empty, the cell
is exported without the redirect, and the builder says so. `--ids-file` reads
one id per line (what `Save nids` writes). If it refuses with `alignment bank
... not found`, the message lists every directory it looked in -- pass
`--bank-dir` with the right one rather than moving the file.

**8.4 Dry run, then submit.** The dry run resolves the manifest row, the
passive constants, the inputs (with their hashes) and the fingerprint, and
exports nothing:

```
python3 run_p1_export.py --root ../h01 --manifest ../h01/p1/manifests/L3_exc.csv --task 0 --dry-run
```

Then, from `h01_code`, one submission per population with its own name, log
and width (these override the `#PBS` defaults in the script):

```
mkdir -p logs
qsub -N p1_L3exc -o logs/p1_L3exc.log -J 0-3 -v MANIFEST=../h01/p1/manifests/L3_exc.csv p1_export.pbs
```

`qsub -v` knobs: `MANIFEST` (required; relative paths resolve against
`H01_CODE`), `PASSIVE_TABLE` (default `passive_params.csv`; relative against
`H01_CODE`), `OUT_DIR` (default `p1`; relative against `H01_ROOT`),
`H01_ROOT`, `H01_CODE`, `H01_ENV`, and `DRY_RUN=1`, `FORCE=1`,
`NO_RIGIDITY=1`, `NO_NEURON_VALIDATE=1`. The bare `ROOT` / `CODE` /
`ENV_NAME` are reported and ignored, as everywhere else here.

An inhibitory population is two submissions, one per variant, each into its
own tree:

```
qsub -N p1_L2inh_SST   -o logs/p1_L2inh_SST.log   -J 0-217 \
     -v MANIFEST=../h01/p1/manifests/L2_inh.csv,PASSIVE_TABLE=passive_params_inh_SST.csv,OUT_DIR=p1_inh_SST \
     p1_export.pbs
qsub -N p1_L2inh_PVVIP -o logs/p1_L2inh_PVVIP.log -J 0-217 \
     -v MANIFEST=../h01/p1/manifests/L2_inh.csv,PASSIVE_TABLE=passive_params_inh_PVVIP.csv,OUT_DIR=p1_inh_PVVIP \
     p1_export.pbs
```

Excitatory populations use the defaults and land in `p1/`. The layout is
therefore `h01/p1/` (exc), `h01/p1_inh_SST/`, `h01/p1_inh_PVVIP/`, each with
its own `p1_summary.csv` (`--summarise --out-dir <tree>`).

**8.5 After the array:**

```
python3 run_p1_export.py --root ../h01 --summarise --manifest ../h01/p1/manifests/L3_exc.csv
python3 run_p1_export.py --root ../h01 --out-dir ../h01/p1_inh_SST --summarise --manifest ../h01/p1/manifests/L2_inh.csv
```

folds every `<tree>/*/neuron_*_p1.json` into `<tree>/p1_summary.csv` (one
row per cell: status, qc_status, gate_status, hoc_verdict, rigidity,
quarantine, n_sections, F variants, spine and synapse counts, passive
constants and the table they came from, fingerprint) and, with `--manifest`, lists the manifest cells that have no
record and exits 1 -- an array narrower than the manifest or a dead task is
caught here, not discovered later.

**8.6 Outputs**, in `h01/p1/<cell_id>/`:

| file | writer | content |
|---|---|---|
| `neuron_<id>_aligned.hoc`, `_phi.csv`, `_segment_map.csv`, `_section_table.csv`, `_spine_bases.csv`, `_synapses.csv`, `_provenance.json`, `_alignment.json`, `_mapped_synapses.csv` | `align_and_export` | the Stage 1 export, committed only on a gate pass |
| `neuron_<id>_spine_nodes.csv` | `p1_spine_stats` | one row per spine node (`spine_part` head/neck, raw nm, aligned base) |
| `neuron_<id>_spine_stats.csv` | `p1_spine_stats` | one row per spine, keyed `root_id`: head/neck geometry, base and beyond-shaft lengths, R_neck, path distance, A_skel, section, synapse count, demotion vote |
| `neuron_<id>_hoc_validation.json` | `p1_hoc_audit` | structural audit + NEURON validation (or `neuron_available: false`) |
| `neuron_<id>_p1.json` | the driver | the record: manifest row, passive constants, fingerprint, result summary, timings |

A cell the gate rejects gets only the `_p1.json` (status `ok`, `qc_status`
`fail`, nothing else written; that verdict is final on rerun unless
`--force`). A cell the structural audit rejects is moved whole into
`<out>/<cell_id>/_quarantine/` and the record says so. A cell whose synapse
CSV is empty after the direction filter is exported without the redirect and
recorded `pass_low_confidence`. A cell with a record whose fingerprint matches
is skipped (`resumed`); a record with `status: error` is always reprocessed.
The per-spine key is the root node id: `root_node_id` here, `root_id` in P3's
`cell<id>_spines.csv`; join on that. The `spine_id` string
(`"<cell_id>:<root_node_id>"`) is written here as a convenience column only.

**8.7 Not yet written:** P4 (the bank) and the eight blank passive rows.

---

## 9. Soma census: which cells carry a complete soma

`soma_enforce.py` gates on the soma's radius, and that test is a
**cell-completeness** check, not a formality: its own docstring records
`neuron_606394351` (soma radius 331.9 nm) as "a truncated arbour fragment
with a promoted root" against the intact `neuron_794820508` at 5325.5 nm.
A cell below the floor still exports -- decision D-002 keeps the flag and
filters downstream -- but you want to know how many there are before
committing to ten populations.

```
python3 soma_census.py --neurons-dir ../h01/neurons --stage1-dir stage1 --limit 50
python3 soma_census.py --neurons-dir ../h01/neurons --stage1-dir stage1 --out ../h01/soma_census.csv
```

It reads six columns per skeleton and runs no simulation, so it is I/O bound;
it calls `soma_enforce.identify_soma_by_geometry` itself, with the module's
thresholds passed explicitly, rather than restating the test. One row per
cell, and a verdict:

| verdict | meaning |
|---|---|
| `ok` | the root passes the 2000 nm floor and is the thickest node -- high confidence |
| `below_floor` | a soma by name, too thin to be one |
| `geometry_disagrees` | a thicker node exists elsewhere in the cell |
| `both` | below the floor AND the geometry disagrees |
| `no_root`, `no_r_column`, `missing:...`, `unreadable` | the skeleton could not be read as a tree |

The summary prints the percentage that would survive a high-confidence
filter, plus root-diameter quantiles. That percentage is the number that
decides whether D-002's downstream filter is a filter or a decimation.

## 10. Looking at the results: `p1_plots.py`

Everything below reads only what P1 wrote under `<out_dir>/<cell_id>/` plus
the raw skeleton, and never re-runs the exporter. Figures land in `--fig-dir`
as PNG at **220 dpi** (`--dpi`), named `<tree>_<cell>_<figure>[_<proj>].png`,
in the ALIGNED frame (micrometres, soma at the origin, +z the cortical axis
of the bank).

**Projection.** `--proj xz` (default), `yz`, `xy` flatten the aligned frame;
**`--proj 3d`** draws the arbour in three dimensions on a cubic box -- both
panels share one cube of the largest range, so no axis is silently stretched
-- viewed from `--elev` / `--azim` (default 18 deg, -70 deg).

**Mark sizes** live in one `STYLE` dict and are tuned with two knobs:
`--lw-scale` multiplies every skeleton and compartment line width, and
`--syn-size` sets the synapse marker **diameter in points** (default 2.2; the
pruned-spine triangle keeps its 1.6 ratio to it). Note matplotlib's `s` is an
area in points squared, which the code squares for you -- passing `--syn-size 9`
would give a very large marker, not v1.0's.

```
python3 p1_plots.py --root ../h01 --out-dir ../h01/p1_inh_SST --cell 489469961 --fig-dir ../h01/figures
python3 p1_plots.py --root ../h01 --out-dir ../h01/p1 --cell 1317492596 --cell 1333261412 --fig-dir ../h01/figures --proj 3d --dpi 300
python3 p1_plots.py --root ../h01 --out-dir ../h01/p1_inh_SST --population --compare ../h01/p1_inh_PVVIP --fig-dir ../h01/figures
```

| figure | what it shows | reads |
|---|---|---|
| `pruning` | the aligned skeleton with the pruned spine nodes in violet next to the pruned skeleton alone (demoted continuations stay, as in the export) | `neurons/neuron_<id>.csv`, `_alignment.json` (the cell's own `soma_pos_nm`, `mean_matrix`), `_spine_nodes.csv` |
| `downsampling` | the raw skeleton next to the LFPy compartments the d_lambda rule produced, width ~ diameter, alternating shades so every compartment is visible; each incoming synapse at its aligned position, exc orange / inh blue, hollow triangle if it sat on a pruned spine, joined by a thin line to the compartment its `lfpy_idx` names (from the anchor when redirected) | `_aligned.hoc` rebuilt through `alignment.default_cell_factory` with the record's `result.segmentation`, `_mapped_synapses.csv`. **Needs NEURON + LFPy (spine_env)**; skipped with a message otherwise. The rebuilt compartment count is checked against the record's `totnsegs` and a mismatch is a WARNING -- it means the (cm, Ra, lambda_f, d_lambda) did not reproduce the export |
| `spines` | neck radius (min over neck nodes), head radius (max), neck length, `R_neck` from skeleton frustums at Ra = 100 (log axis when the spread earns it), spine density per um of shaft along the path distance (60 um F gate marked), synapses per spine split by partition source | `_spine_stats.csv`, `_phi.csv` |
| `arbour` | per path-distance bin: shaft cable, length-weighted mean shaft diameter, shaft + spine membrane area, and their ratio (the local `A_spine / A_shaft` that `F` integrates beyond 60 um). **Which spine-area model it draws is named in every panel and in the file name** -- see below | `<root>/out/neuron_<id>_phi_mesh.csv` when it exists, else `_phi.csv`; `_p1.json` |
| `population` | from `p1_summary.csv`: `F_lit`, spines vs cable, soma diameter, arbour angle from +z, qc verdicts, compartments per cell | `p1_summary.csv` (run `--summarise` first) |
| `population_vs_<tree>` | with `--compare`: the same, but the last two panels are paired by cell: `totnsegs` tree A vs tree B against the identity and the sqrt(2) line, and the `qc_status` cross-tab -- the D-003 comparison | both trees' `p1_summary.csv` |

Bin width for the profiles is `--bin-um` (20). The palette is fixed, not
cycled: exc `#eb6834`, inh `#2a78d6`, pruned spine `#4a3aa7`, shaft grey.

`compartments_from_cell` handles LFPy >= 2.2 (`cell.x` of shape
`(totnsegs, 2)`) and the older `xstart / xend` layout; the cluster's LFPy
2.3.7 was not exercised from the sandbox (no NEURON there), so the first run
on the cluster is where that branch is confirmed.

### 10.1 Which phi the arbour figure draws -- mesh, or the skeleton fallback

Two tables carry the name `phi`, and they are not interchangeable:

| file | written by | `spine_area_um2` is | its F |
|---|---|---|---|
| `<out_dir>/<id>/neuron_<id>_phi.csv` | P1 | **skeleton frustums** -- the FALLBACK model | `F_lit` in the record, i.e. `F_skel` |
| `<root>/out/neuron_<id>_phi_mesh.csv` | P3 | **the mesh measurement** (`h01_spine_area_F.phi_with_spine_areas`, variant `mesh_beyond`); the value it replaced is kept as `spine_area_skel_um2` | the deliverable F |

The standing decision is **spine area from the mesh, skeleton only as a
fallback; shaft area ALWAYS from skeleton frustums** -- `phi_with_spine_areas`
swaps one column and leaves `shaft_area_um2`, `d_from_um` and the row set
untouched. So the two left-hand panels of the figure (cable, shaft diameter)
are identical either way, and only the two right-hand ones change.

**`--phi mesh` is the default**, because F is reported from the `mesh_beyond`
variant by decision and the other variants exist for comparison only. With no
P3 table the arbour figure is **skipped, with the reason printed** -- the other
figures are still written. `--phi auto` permits the skeleton bracket as a
fallback and `--phi skel` forces it; both label it a fallback, in the file
name and on every panel.
The choice is written into the file name (`..._arbour_mesh.png` /
`..._arbour_skel.png`), into the title, into the stacked-bar legend and into
the ratio panel's axis label, and the fallback is labelled a fallback rather
than presented as the spine area. F in the title is computed **from the phi
actually drawn**, through `spine_density.cell_f_beyond_cutoff` -- the same
function the exporter calls, never a reimplementation; on P1's table it
reproduces the record's `F_lit` exactly (check M6).

**Today no cell has a mesh measurement**: `h01/out/` has never been created,
so with the default the arbour figure is skipped for every cell and you must
pass `--phi skel` explicitly to see the skeleton bracket. The mesh path is
wired and tested against a synthetic P3 table (checks M3-M8), not against real
P2 output.

The population figure's `F_lit` panel is P1's skeleton F for the same reason,
and its axis says so. The deliverable F is `F_lit_deliverable` in P3's
`spine_area_F_summary.csv`, never this column.

---

## 11. The campaign: one submission per population (`campaign.pbs`, D-004)

**Decision D-004 (2026-09-22):** launching one job must run the whole
pipeline -- P1, then P2, then P3 -- for every cell of the population, and the
spine area of every cell is the **mesh** measurement, `mesh_beyond`. The other
variants are comparison only. Running the stages as three separate
submissions is how the skeleton bracket ended up reported as F; this script
removes that path.

### 11.1 How the array is laid out

`campaign.pbs` decomposes `PBS_ARRAY_INDEX` over (manifest row, shard):

```
N = manifest rows, S = SHARDS (default 1)       qsub -J 0-(N*S-1)
ROW   = index / S     the cell (0-based manifest row = P1 --task)
SHARD = index % S     the P2 shard of that cell (= P2 --task, --ntasks S)
```

| stage | runs on | argv it receives |
|---|---|---|
| P1 `run_p1_export.py` | shard 0 of every cell | `--task ROW --manifest M --passive-table --out-dir`, `--stage1-dir $H01_CODE/stage1` |
| P2 `run_spine_area_F.py` | every shard | `--cell <row's cell_id> --task SHARD --ntasks S`, `--stage1-dir` and `--g-table` under `$H01_CODE` |
| P3 `merge_spine_area_F.py` | the same task when S = 1; a dependent array with `PHASE=merge` when S > 1 | `--cell --ntasks S --deliverable mesh_beyond`, `--stage1-dir` and `--g-table` under `$H01_CODE` (its `resolve()` needs both -- the handoff sketch omitted `--g-table` and would have died there) |

The cell id of every stage is read from the same manifest row, so P1's
`--task` and P2/P3's `--cell` cannot drift. `module load proxy` runs once,
between P1 and P2. A P1 error stops that cell before P2 (exit 1, cell
named); a P2 error stops it before P3, so no merge ever runs on a dead
shard. `--allow-missing` is never passed. All three stages resume, so
re-submitting an index after a walltime kill is nearly free.

**Start with S = 1** (one task per cell, everything in it, no dependent
job). Move to S > 1 only if section 11.4's arithmetic says a cell does not
fit the 12 h walltime.

### 11.2 Submitting

Per population, from `h01_code`, with its own `-N`, `-o` and `-J`:

```
python3 build_p1_manifest.py --root ../h01 --layer L3 --cell-type exc --ids-file populations/L3_exc.txt --allow-unbanked --out ../h01/p1/manifests/L3_exc.csv   # prints -J 0-536
qsub -N camp_L3exc -o logs/camp_L3exc.log -J 0-536 -v MANIFEST=../h01/p1/manifests/L3_exc.csv campaign.pbs
```

With `SHARDS=4` the array is `-J 0-2147` and the merge follows as a
dependent array of the same script:

```
qsub -N camp_L3exc  -o logs/camp_L3exc.log  -J 0-2147 -v MANIFEST=../h01/p1/manifests/L3_exc.csv,SHARDS=4 campaign.pbs
qsub -N merge_L3exc -o logs/merge_L3exc.log -J 0-536  -W depend=afterokarray:<jobid[]> -v MANIFEST=../h01/p1/manifests/L3_exc.csv,SHARDS=4,PHASE=merge campaign.pbs
```

**Interneurons (D-003 x D-004).** Spine area is geometry and does not depend
on (cm, Ra), so P2/P3 run **once per cell** into `h01/out/`, whichever P1
tree they were launched with. Submit ONE variant through `campaign.pbs` and
the other through `p1_export.pbs` (P1 only):

```
qsub -N camp_L2inh_SST -o logs/camp_L2inh_SST.log -J 0-217 -v MANIFEST=../h01/p1/manifests/L2_inh.csv,PASSIVE_TABLE=passive_params_inh_SST.csv,OUT_DIR=p1_inh_SST campaign.pbs
qsub -N p1_L2inh_PVVIP -o logs/p1_L2inh_PVVIP.log -J 0-217 -v MANIFEST=../h01/p1/manifests/L2_inh.csv,PASSIVE_TABLE=passive_params_inh_PVVIP.csv,OUT_DIR=p1_inh_PVVIP p1_export.pbs
```

Knobs: `MANIFEST` (required), `SHARDS`, `PHASE=all|merge`, `PASSIVE_TABLE`,
`OUT_DIR` (P1 only), `DRY_RUN=1` (P1 and P2 dry-run, P3 skipped),
`FORCE=1`, `NO_RIGIDITY=1`, `NO_NEURON_VALIDATE=1` (P1 only), `SUBSET`,
`MIN_SPINE_VALUE`, `NO_MEASURE_BASE` (P2 **and** P3 -- fingerprint knobs),
`MIN_COVERAGE` (P3), plus `H01_ROOT`, `H01_CODE`, `H01_ENV`. A task whose
row is past the manifest refuses and prints the `-J` width to use.

Dry run first, one index in the foreground (no queue slot):

```
PBS_ARRAY_INDEX=0 DRY_RUN=1 MANIFEST=../h01/p1/manifests/L3_exc.csv bash campaign.pbs
```

Expect the banner `index 0 -> row 0 shard 0/1 | cell 1302789404`, then
`=== P1 export`, `DRY RUN: inputs and constants resolved`, `=== P2 measure`,
the P2 dry-run shard plan with g-table sha `b4daf2f6815d`, and
`=== P3 merge SKIPPED (DRY_RUN)`.

### 11.3 Which F is which, and the join

| file | written by | its F |
|---|---|---|
| `<tree>/<id>/neuron_<id>_phi.csv`, `..._p1.json`, `<tree>/p1_summary.csv` | P1 | `F_lit` = **skeleton frustums**, the comparison bracket; never updated by P3 |
| `out/neuron_<id>_phi_mesh.csv`, `out/spine_area_F_summary.csv` | P3 | `F_lit_deliverable` (`deliverable_variant = mesh_beyond`) -- **the F** |

`p1_summary.csv`'s `F_lit` will always be the skeleton. So after the array:

```
python3 run_p1_export.py --root ../h01 --summarise --manifest ../h01/p1/manifests/L3_exc.csv    # per tree: --out-dir ../h01/p1_inh_SST
python3 campaign_join.py --root ../h01 --require-complete
```

`campaign_join.py` writes `out/campaign_F.csv`, one row per (cell, P1 tree),
with `F_lit_skel_p1` and `F_lit_deliverable` side by side and `p3_present`
saying which cells have a mesh F. Where P3 is absent `F_lit_deliverable` is
NaN -- it is never filled from the skeleton. A P3 row whose variant is not
`mesh_beyond` is refused outright (`--deliverable` to change the expectation,
never to mix). `--require-complete` exits 1 listing the cells still without
a P3 row. An interneuron's one P3 row joins to both of its trees.

### 11.4 Sizing S and the walltime (still open)

Seconds per spine on a warm ROI cache has never been measured; the 2.3 s on
record is a cold block read. Measure it with one cell first:

```
qsub -N p2_probe -o logs/p2_probe.log -J 0-0 -v MANIFEST=../h01/p1/manifests/L3_exc.csv,SUBSET=300 campaign.pbs
```

Then per-cell wall time = spines x rate / S, against the 12 h walltime;
`p1_summary.csv`'s `n_spines` gives the spine counts. Check the queue caps
(`qstat -Qf cpu | grep -i max`) before the first `N*S` array.

What the orchestrator does NOT decide (D-004, open): the coverage policy for
a cell whose `fallback_area_frac_deliverable` is high (a hybrid, not a
measurement; `--min-coverage` 0.99 only flags it), and the
`shaft_terminates` policy. Both live in P3's numbers, not in the launch.
