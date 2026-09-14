# TEEG unified spine pipeline on davinci-1: handoff

Plan and current state for merging realignment, spine identification, the
shaft-continuation correction, and calibrated mesh spine-area into ONE pipeline
that runs on the cluster over the whole L3 bank. Companions:
`TEEG_S1_operational_reference_v1.md` (Stage 1 as of 26 Aug),
`TEEG_cylinder_calibration_HPC_and_Colab_reference.md` (the g table),
`TEEG_spine_mesh_cylinder_calibration_handoff_2026-09-10.md`.

| Date | Change |
|---|---|
| 2026-09-14 | Created. Records the compute-node network result (which CONTRADICTS the cylinder-calibration reference), `morphology_exporter-1.2.0` with the three-vote correction, `h01_spine_area_F v1.5`, and the partition-divergence problem that must be fixed before any campaign run. |

## 0. The one-paragraph version

Everything up to and including F currently exists as two Colab notebooks that
build their own partitions independently and therefore disagree about which
components are spines. The unified pipeline makes the partition once, per cell,
on the cluster, and everything downstream -- the `.hoc`, phi, F_skel, the
per-spine mesh areas, F_mesh -- is computed from that one partition. Compute
nodes CAN reach the H01 bucket (measured 2026-09-14), so no fetch/compute split
is forced and the whole bank is one array dimension.

```
P1 partition (per cell, seconds, no network)
     raw skeleton -> steps 1-3 -> labeller -> THREE-VOTE demotion -> soma
     -> phi, F_skel -> prune -> align -> .hoc + sk table + fingerprint
P2 measure (per spine, network + compute, array-parallel)
     seg cutout -> Voronoi/bridge -> mesh -> classify -> g -> A_mesh,
     rind, measured base -> shard ledger
P3 assemble (per cell)  ledgers -> attribution gate -> phi_mesh -> F variants
P4 bank (once)          per-layer kappa, F_skel vs F_mesh, F into hoc metadata
```

## 1. Scope of what was actually verified

Verified by running it in the 2026-09-14 conversation:

- compute-node network and a real CloudVolume read (section 2);
- `morphology_exporter-1.2.0` step 4b on fixtures, and the CELL 6 loop body
  against a stubbed notebook namespace (42/42 on the spine-check suite, plus
  the project's own 203/203 `run_all_tests.py`);
- the HPC runner and merger end to end against a stub network (21/21);
- `h01_spine_area_F v1.5` and `h01_spine_base v1.0` (103 checks, 1 skip; the
  skip is plotly, absent in the sandbox);
- bank cell 15543554616 is shaft-intact after pruning (0 missing, 0 extra, 0
  diameter mismatches, cable 4456.090 um matching to 9e-13 um).

NOT verified, named here so nobody assumes coverage:

- no cell has ever been run through the cluster pipeline end to end -- P1 as
  bundled does not yet exist as a headless driver (section 4);
- `alignment.py`, `hoc_qc.py`, `synapse_redirect_audit.py`, `soma_enforce.py`,
  `spine_labeller.py` source was never read in this work; behaviour is
  described from call sites and outputs only;
- the three-vote rule has run on FIXTURES only, never on a real cell. Every
  number attributed to it below is a prediction, not a measurement;
- per-spine timing on real H01 data from the cluster. The 2.3 s figure is one
  cold 64x64x8 block, not an ROI.

## 2. Compute nodes have outbound network: correction to project knowledge

`TEEG_cylinder_calibration_HPC_and_Colab_reference.md` section 1 states that
compute nodes have no outbound network and that `cloud-volume` is therefore
deliberately absent from the environment. That is no longer true, and the
campaign design depends on it.

Measured 2026-09-14, PBS batch job `1716150.pbsserver01` on `dvnode001`:

```
PROBE module proxy    : loaded          http_proxy=http://10.17.16.110:8080/
PROBE curl bucket     : HTTP 200
PROBE curl c3 info    : HTTP 200  (778 bytes)
PROBE cv read         : OK  shape (64, 64, 8)  dtype uint64  5 distinct ids  2.3 s
```

`module load proxy` is required and is already in `probe_net.pbs` and
`spine_area_F.pbs`. Without it the job has no proxy variables and every fetch
fails.

`cloud-volume 12.14.4` was installed into `spine_env` with pip and did NOT move
numpy/scipy/skimage/pandas. Post-install versions, read from the probe:

| package | version |
|---|---|
| numpy | 2.5.3 |
| scipy | 1.18.0 |
| scikit-image | 0.26.0 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| cloud-volume | 12.14.4 |

## 3. THE blocker: five partition call sites, three different rules

The partition -- which nodes are spine, which are shaft -- is the decision that
splits A_spine from A_shaft. F is a ratio of the two, so a different partition
is a different F, and the per-spine sigma ids refer to different objects.

Call sites, verified by grep on 2026-09-14:

| Where | File:line | Rule today |
|---|---|---|
| Stage 1 export | `morphology_exporter.py:524` step 4b | THREE-VOTE (1.2.0) |
| Stage 1 CELL 6c rebuild | `_label_like_exporter` in `CELL_6c_complete.py` | steps 1-3 + 5 only, NO 4b |
| Mesh notebook CELL 5 | `Spine_Mesh_Analysis_cells_v2.py:223` | two-observable |
| Mesh SAF cell | `COLAB_CELL_spine_area_F.py:83` | two-observable |
| HPC runner | `run_spine_area_F.py:137` | two-observable |

Consequences, measured:

- cell 1302789404, mesh notebook CELL 5, two-observable:
  `demoted 824 component(s), 5912 node(s), spine nodes 51287 -> 45375`;
- cell 15543554616, two-observable scored 312 of 2023 roots, worth
  `F_lit 1.6384 -> 1.4953`, dF = -0.143.

The three-vote rule demotes strictly fewer (the taper test holds back anything
with a distal radius maximum; the 150 nm floor holds back sub-voxel stubs), so
F_skel rises relative to those numbers. By how much is UNMEASURED.

CELL 6c's rebuild is caught by its own gate -- it compares its spine-node count
against `records` and prints `WARNING ... partitions disagree` -- but it still
produces wrong statistics when that fires. Fix: add step 4b to
`_label_like_exporter`, between the labeller call and the final
`classify_frame`.

### The single entry point

`morphology_exporter.demote_shaft_continuations_three_vote(df, ...)`, verified
present at `morphology_exporter.py:524`:

```python
demote_shaft_continuations_three_vote(
    df, rho_shaft_min=None, cos_shaft_min=None, min_len_nm=None,
    bulge_min=None, peak_frac_min=None, require_taper=True,
    annotation_column="annotated_type")   # -> (df, report)
```

Defaults: `rho_shaft_min` and `cos_shaft_min` fall through to
`shaft_continuation` (0.50 and 0.70); `MIN_LEN_NM = 150.0`,
`BULGE_MIN = 1.25`, `PEAK_FRAC_MIN = 0.30` from `continuation_inspect`.
`require_taper=False` reduces it to the two-observable rule plus the floor.

Needs `shaft_continuation.py` AND `continuation_inspect.py` importable beside
it; raises a named ImportError rather than exporting uncorrected.

Every one of the five call sites must route through this function before any
campaign run. That is P0.

### The taper vote, since it is the new part

For a component with root rho, take its LONGEST path outward, ordered by path
distance s from the root, with radius r(s). Let k = argmax_s r(s). Then

    bulge        = r(s_k) / min_{s <= s_k} r(s)
    s_peak_frac  = s_k / s_max

The component is SPINE-LIKE when `bulge >= BULGE_MIN` and
`s_peak_frac >= PEAK_FRAC_MIN`, BRANCH-LIKE otherwise, and UNDECIDABLE when
`s_max < MIN_LEN_NM`. Rationale: a dendritic branch thins monotonically; a
spine has a thin neck and then a head, so its radius has a distal maximum. Two
path nodes suffice -- root and tip already distinguish a rise from a taper.
Undecidable components are KEPT as spines, the conservative direction.

## 4. Stage plan

| Stage | Scope | Exists? | Gate |
|---|---|---|---|
| P0 partition unification | code only | NO -- five call sites (section 3) | all five route through `demote_shaft_continuations_three_vote`; one cell gives the same spine-node count in Stage 1 and the mesh notebook |
| P1 partition + export | per cell, seconds, offline | NO as a headless driver; the logic exists in `CELL_6_complete.py` | `check_pruned_hoc` A-C pass; section D reports ~0 shaft-like remaining; fingerprint written |
| P2 measure | per spine, network+compute, array | YES -- `run_spine_area_F.py` (needs the section 3 fix) | fingerprint match across shards; failures < 1%; box ~ 0; seam < 4 nm |
| P3 assemble | per cell | YES -- `merge_spine_area_F.py` | attribution gate 0.00; coverage ~100%; `F_lit mesh` and `F_lit mesh_skelfill` converge |
| P4 bank | once | NO | one row per cell; F provenance into the hoc metadata |

Dependency chaining: P1 array -> `-W depend=afterok` -> P2 array -> P3 -> P4.
P2-P4 re-run alone whenever the measurement method changes; P1 only when the
partition does.

### P1: what the headless driver must do

Port of `CELL_6_complete.py` minus Colab. Order is load-bearing:

1. `s0_ingest.write_s0_table` -> node table
2. `node_classify.extract_synapse_frame` / `classify_frame` /
   `resolve_mislabelled_nodes` (exporter steps 1-3)
3. `spine_labeller.label_dendritic_spines_robust` at
   `morphology_exporter.SPINE_LENGTH_THRESHOLD_NM` = 4000 nm
4. step 4b, the three-vote demotion
5. `classify_frame` again (MANDATORY: the labeller overwrites
   `annotated_type` and leaves `compartment_class` stale)
6. `soma_enforce.enforce_soma`
7. `build_phi` x3 (nocap / disc / cap) -> F_skel
8. prune, align, write `.hoc`
9. `h01_spine_area_F.skeleton_spine_table` -> the `sk` table P2 needs
10. write the parameter fingerprint

Simplest implementation: call `morphology_exporter.export_neuron(...,
demote_continuations=True, continuation_kw=...)` and let it do 1-8, then add 9
and 10. That reuses the tested path rather than reimplementing it.

Open: whether P1 runs the propagation gate (`hoc_qc`) and the synapse redirect.
Both need decisions -- see section 7.

### P2: commands

Already written and tested. After the section 3 fix:

```
cd /davinci-1/home/ldellamea/TEEG/Towards-EEG/h01_code && python3 run_spine_area_F.py --root ../h01 --cell 1302789404 --task 0 --ntasks 40 --dry-run
```

```
mkdir -p logs && qsub -v CELL=1302789404 spine_area_F.pbs
```

The array width in `#PBS -J` and `NTASKS` in the `qsub -v` list must match, or
the shards will not tile the spine list.

Flags worth knowing: `--measure-base` (union-mesh base, the `A_beyond` track),
`--subset N` (size-stratified sample), `--min-spine-value` (protrusion floor),
`--roi-cache DIR`, `--axial-window-nm`.

### P3: commands

```
python3 merge_spine_area_F.py --root ../h01 --cell 1302789404 --ntasks 40
```

Refuses shards whose parameter fingerprint disagrees. `--allow-missing` only
after deciding an incomplete answer is acceptable.

## 5. Locations, environment, inventory

| | |
|---|---|
| Repo on cluster | `/davinci-1/home/ldellamea/TEEG/Towards-EEG` (cloned 2026-09-14) |
| Code | `.../Towards-EEG/h01_code` -- NOT `TEEG/Spines`, which is the old cylinder-sweep repo |
| Branch | `h01-spine-campaign` |
| Campaign root | `../h01` (`neurons/`, `out/`, the g table) -- DOES NOT EXIST YET |
| Conda base | `/archive/apps/miniconda/miniconda3/py312_2` |
| Env | `spine_env` |
| Login node | `dvlogin02` |
| Scheduler | PBS Pro. `-J 0-39`, `$PBS_ARRAY_INDEX`. No `sbatch`. |

Module versions, read from source 2026-09-14:

| Module | Version | Home |
|---|---|---|
| `morphology_exporter` | 1.2.0 | Stage 1 |
| `spine_density` | 1.3.0 | Stage 1 |
| `shaft_continuation` | 1.1.0 | Spine Mesh Analysis (needed by Stage 1 too) |
| `continuation_inspect` | v1.0 | Stage 1 |
| `check_pruned_hoc` | v1.1 | Stage 1 |
| `h01_spine_area_F` | v1.5 | both |
| `h01_spine_base` | v1.0 | both |
| `h01_spine_roi`, `h01_spine_batch`, `h01_spine_geometry` | v1.0 | mesh |
| `h01_area_calibration` | v1.1 | both |
| `run_spine_area_F`, `merge_spine_area_F` | v1.0 | HPC only |

Calibration table in use: `g_table_cyl_2deg.npz`, sha256[:12] `b4daf2f6815d`,
28000 cylinders over 40 files, g in [1.0014, 1.7689], area-weighted mean
1.0415, axes theta 0..90 (46 points) and phi 0..44 (23 points).
`h01_spine_area_F` v1.5 accepts both that axis convention and
`fundamental_grid`'s (phi 0..45); v1.4 and earlier rejected the former.

Missing from the cluster as of 2026-09-14: the four Stage 1 modules in
`h01_code/stage1/`, the neuron CSVs, `alignment_metadata_L3.csv`, and the
synapse CSVs if the redirect is wanted.

## 6. Current numbers, and what they are conditional on

Cell 1302789404, 2026-09-14 run, TWO-OBSERVABLE partition, 28 of 5627 spines
measured (0.5% coverage, so F_mesh is 100% kappa-extrapolated):

| quantity | value |
|---|---|
| F_lit skel | 1.5761 |
| F_lit mesh | 1.7795 |
| F_lit mesh_norind | 1.7475 |
| F_lit mesh_beyond | 1.6515 |
| kappa raw / norind / nobase / both / beyond | 1.353 / 1.297 / 1.889 / 1.811 / 1.131 |
| Voronoi rind | 2.2% of A_mesh (median), max 37.4% |
| base frustum | 28.4% of A_skel (pooled) |
| s_base - r_shaft | median +71 nm, p90 abs 220 nm |

`F_lit mesh_beyond` is the column to read. A_beyond is the membrane past the
measured base plane -- the spine's own surface -- and its fill factor
kappa_beyond = 1.131 is self-consistent regardless of what A_skel includes.

Two interpretations settled in that conversation:

- the base frustum, NOT the Voronoi rind, is the dominant junction error. Over
  a quarter of what the skeleton calls spine area is a slab at shaft radius.
  The rind is only 2.2%, so the Voronoi bisector sits close to the shaft
  surface on real spines -- better behaved than the phantom suggested.
- `s_base - r_shaft > 0` is EXPECTED GEOMETRY, not a radius bug. For a spine
  leaving at angle alpha to the shaft axis,
  `s_base = r_shaft / sin(alpha) >= r_shaft` for all alpha in (0, pi/2], with
  equality only at alpha = pi/2. A median +71 nm on r ~ 300 nm implies
  sin(alpha) ~ 0.81, alpha ~ 54 deg, which is ordinary. An earlier reading of
  this as a systematic radius underestimate was WRONG and is corrected here
  rather than silently dropped.

What F is still conditional on, at any coverage:

A_shaft is skeleton frustums, always. `build_phi` computes `shaft_area_um2`
from node radii and nothing in the mesh pipeline touches it;
`phi_with_spine_areas` explicitly preserves that column. So F is half-measured
and half-modelled however many spines are run. This is a thing to state when
quoting the number, not a reason to change the plan. A mesh-based shaft audit
is cheap later -- the shaft-context mask is already in every ROI -- but is out
of scope here.

Literature anchor (project knowledge, Eyal et al. 2016, full text): human L3,
40 y temporal `F = 2.39 +/- 0.63` per dendrite; 85 y temporal 1.87-1.89;
adopted value 1.9. H01 is a 45 y temporal donor, so the 40 y row is the
age-matched comparison.

## 7. Open decisions, needed before P1 can be written

These are genuine forks, not details. Each changes what the pipeline does.

1. Which cells? `alignment_metadata_L3.csv` carries 74 references; the mesh
   notebook's `ALL_CELLS` lists 4. Does "all neurons" mean the 74, or a
   different set? This sets the campaign size (at ~5000 spines per cell, 74
   cells is roughly 370k spines).
2. Which F is the deliverable? The exporter builds three (`nocap`, `disc`,
   `cap` at `CAP_H_UM`). Stage 1 currently runs `CAP_TIPS=True, CAP_H_UM=0.1`,
   and every mesh comparison so far was on the NOCAP basis. Mixing them is how
   the 1.638-vs-1.664 confusion arose.
3. Does P1 run the propagation gate and the synapse redirect? Both are in the
   Colab CELL 6 path. The gate needs NEURON + LFPy in `spine_env` (present in
   Colab; UNVERIFIED on the cluster). The redirect needs the synapse CSVs on
   the cluster. Dropping either makes P1 simpler, but the `.hoc` bank it
   produces is then not the same artefact Stage 1 produces.
4. Does Colab keep a role? The interactive figures (union profiles, 3D
   galleries, the continuation inspector) only exist there. Assumed yes, for
   QC on a handful of cells, with the cluster doing production -- but if the
   intent is to retire the notebooks, P1-P4 need figure output.
5. ROI store, keep or drop? Design A removes the need. The argument for
   keeping it is re-measurement: the method changed twice in one week
   (v1.4 -> v1.5), and a store makes re-measurement pure compute. Cost
   estimate (NOT measured): 50-100 kB per spine, so 20-40 GB for 74 cells.
6. Three-vote thresholds: accept the defaults? `MIN_LEN_NM=150`,
   `BULGE_MIN=1.25`, `PEAK_FRAC_MIN=0.30` were chosen from one gallery of four
   spines plus fixture reasoning. Running CELL 6e on a real cell before the
   campaign would put a number on them.

## 8. Troubleshooting index

| Symptom | Cause | Fix |
|---|---|---|
| every fetch fails on a compute node | `module load proxy` missing | it is in the shipped `.pbs`; check the job script was not edited |
| `ModuleNotFoundError: shaft_continuation` | it lives in Spine Mesh Analysis, not Stage 1 | copy it beside `morphology_exporter.py`; it needs only `spine_labeller` |
| `demote_continuations=True` raises ImportError | `continuation_inspect.py` absent | copy it, or pass `continuation_kw={'require_taper': False}` |
| merge refuses: "shards built with different parameters" | one shard ran with different flags or a different CSV | re-run those tasks with the same flags, or delete their ledgers |
| `smoke_test_h01_spine_area_F` dies on `import matplotlib` | pre-2026-09-14 bundle | the import is guarded in the current one |
| CELL 6 resumes a cell instead of correcting it | checkpoint predates the correction | current CELL 6 detects this and reprocesses; older ones need `rm "$OUTPUT_DIR"/neuron_*_alignment.json` |
| CELL 6c prints "partitions disagree" | the rebuild lacks step 4b | section 3 |
| calibration table rejected: "theta/phi axes are not the 2.0-degree fold grid" | v1.4 and earlier only accepted `fundamental_grid` axes | update `h01_spine_area_F` to v1.5 |
| `g max >= 2` on load | v6 table with the radial-projection artefact | rebuild from the v7 sweep |
| rigidity reports `*** MOVED ***` on every cell | the unaligned control got different export kwargs | both calls must share one kwargs dict -- `_EXPORT_KW` in `CELL_6_complete.py` |

## 9. Deferred, deliberately

- kappa bootstrap. BCa on the pooled ratio-of-sums over measured spines,
  paired. To be run as post-processing AFTER the campaign, not before.
- per-dendrite F (mean +/- SD over dendrites >= 10 um, segments >= 60 um), for
  a like-for-like comparison with Eyal's per-dendrite spread. Computable from
  `neuron_{id}_phi_mesh.csv`.
- mesh-based A_shaft, per section 6.
- staged 300-spine run: dropped, the campaign runs everything.
- base-vs-radius inspection: closed by the obliquity argument in section 6.
