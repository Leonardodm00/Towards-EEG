# Diameter pipeline, Phase II: Colab runbook

| Date | Change |
|---|---|
| 2026-10-06 | v1. Written with Block 11 (branch `sci/diameter-pipeline`). Every script below was run in the sandbox on a synthetic cell served through `allen_image_io.fetch_zblock`; none has run on Allen's data (the sandbox gets 403 from api.brain-map.org). |

This runbook covers the real-data steps, in order. Each cell names the line that must **appear** in its output. A missing line means the cell did not get as far as the code that prints it. Each step lists the decisions it feeds; those stay yours. Nothing in these scripts changes the configuration on its own.

Paths: `REPO` is the clone in `/content`. Drive holds data only: the HttpFetcher cache (`CACHE_DIR`, the one cells 1-12 use) and the outputs (`OUT`). Specimen 529878215 has no shift and no flip (design handoff, "Coordinate transform"), and its plane index is z / 0.28 um (`--z0 0`).

## Cell 0: Drive and constants

```python
from google.colab import drive; drive.mount("/content/drive")
SPECIMEN = 529878215
CACHE_DIR = "/content/drive/MyDrive/allen_cache"          # the cache of cells 1-12
OUT = "/content/drive/MyDrive/diameters"
SHIFT_X, SHIFT_Y, FLIP_H, Z0 = 0.0, 0.0, None, 0.0          # global alignment (cells 1-12)
```

## Cell 1: bootstrap (clone or update, `sys.path`)

```python
import os, subprocess, sys
REPO, BRANCH = "/content/Towards-EEG", "sci/diameter-pipeline"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "50", "-b", BRANCH,
                    "https://github.com/Leonardodm00/Towards-EEG.git", REPO], check=True)
sys.path.insert(0, os.path.join(REPO, "Passive Features", "Diameter Re-measurement", "scripts"))
import colab_bootstrap
ENV = colab_bootstrap.bootstrap(REPO, BRANCH)
WS, SCRIPTS = ENV["workstream"], ENV["scripts"]
def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=WS, capture_output=True, text=True)
    print(r.stdout[-6000:] or r.stderr[-6000:])
```

Expect: `[diameter] commit <hash> on sci/diameter-pipeline`. The hash should match the branch head on GitHub.

## Cell 2: smoke tests in Colab (about one minute)

```python
for t in ("test_smoke_config", "test_smoke_geometry", "test_smoke_fit"):
    run(f"tests/smoke/{t}.py")
```

Expect `-- ... 0 fail, 0 error, 0 todo` three times.

## Cell 3: registration on node 4505 and 5-10 stretches ("cell 13")

```python
NODES = "4505"          # then add one node on each of 5-10 stretches: basal, oblique, trunk, tuft
run("scripts/registration_survey.py", "--specimen", str(SPECIMEN), "--nodes", NODES, "--cache-dir", CACHE_DIR,
    "--out-dir", f"{OUT}/registration", "--figures", "--shift-x", str(SHIFT_X), "--shift-y", str(SHIFT_Y),
    "--z0", str(Z0))
```

Expect, for each node: `node 4505: ON THE PROCESS (lateral offset ...)` (or ALONGSIDE / FAR / NOT ON). Then a summary with `"verdicts"`. Figures go in `{OUT}/registration/figures/`.

This step feeds:
- the snap radius (design handoff §Method, step 2);
- the jitter amplitudes `jitter_xy_um` and `jitter_z_um`, which D-024 (vii) keeps at 0 until now. The summary's `suggested_jitter_um` is a suggestion, not a decision.

The file `registration_<specimen>.json` is what `--registration-json` takes below. A node whose category is not ON is dropped from the selection S [corrected 2026-10-06: before Block 11, every registered node would have been dropped].

## Cell 4: pilot measurement, no table

```python
run("scripts/run_node.py", "--specimen", str(SPECIMEN), "--nodes", NODES, "--cache-dir", CACHE_DIR,
    "--out-dir", f"{OUT}/pilot", "--registration-json", f"{OUT}/registration/registration_{SPECIMEN}.json",
    "--figures", "--background", "--shift-x", str(SHIFT_X), "--shift-y", str(SHIFT_Y), "--z0", str(Z0))
```

Every node of the stretches through `NODES` is measured with the same chain that builds the table. Expect `stretch of N nodes from node ... measured`, then the summary JSON with `"n_in_S"`. Outputs:
- `pilot_<specimen>.csv`;
- `pilot_summary_<specimen>.json`;
- `figures/node_<id>.png`, showing the focus curve and the profile with the fitted model.

This step feeds:
- the phantom μ range: `suggested_phantom_mu_range_per_um`, the 10th-90th percentile of μ̂ over the nodes in S (procedure §3.10);
- the dark-flag threshold: `alpha_hat` percentiles and `dark_share_of_converged`;
- the number of calibration nodes, `n_calibration_nodes` (Cell 7).

## Cell 5: camera-chain inputs

```python
run("scripts/camera_calibration.py", "--specimen", str(SPECIMEN), "--cache-dir", CACHE_DIR,
    "--out-dir", f"{OUT}/camera", "--pilot-summary", f"{OUT}/pilot/pilot_summary_{SPECIMEN}.json")
```

Expect `"renderer": {...}` with three fields:
- `jpeg_qtables_file`: Allen's tables, read from the cached crops. `table_sets` should show a single set.
- `background_B_gl`: the median of the masked block medians.
- `noise_sd_gl`: the injected SD whose post-JPEG flat-field SD matches the real background's. JPEG removes much of white noise, so the background SD read on the images is smaller than the injected SD. This value is an upper bound, because tissue texture adds to the real background SD.

Black level and gain cannot be identified from these crops. They stay as configured.

## Cell 6: Allen's radius distribution (sets d_max)

```python
run("scripts/allen_radius_distribution.py", "--fetch", str(SPECIMEN), "--cache-dir", CACHE_DIR,
    "--out", f"{OUT}/radius_{SPECIMEN}.csv")
```

This step feeds `phantom.d_range_um`, which is temporary until now (D-024 (iii), SPEC §8).

## Cell 7: kernel calibration, first stage (diagnostic)

```python
run("scripts/calibrate_kernel.py", "real", "--specimen", str(SPECIMEN),
    "--nodes-csv", f"{OUT}/pilot/pilot_{SPECIMEN}.csv", "--cache-dir", CACHE_DIR,
    "--out", f"{OUT}/calibration/scans_{SPECIMEN}.json", "--shift-x", str(SHIFT_X), "--shift-y", str(SHIFT_Y),
    "--z0", str(Z0), "--registration-json", f"{OUT}/registration/registration_{SPECIMEN}.json")
run("scripts/calibrate_kernel.py", "fit", "--scans", f"{OUT}/calibration/scans_{SPECIMEN}.json",
    "--out", f"{OUT}/calibration/calibration_{SPECIMEN}.json")
```

Expect `growth fit (gaussian_core_width2, symmetric, cubic): N nodes`, the table of knots, and `residual RMS by |k - k*|`. You need at least two thin, faint, flat nodes (d̂ ≤ 0.3 µm, φ ≤ 10°, α̂ ≤ 0.5). Measure more stretches in Cell 4 if there are fewer.

The first-stage table is **not** a calibrated kernel. On rendered thin phantoms its growth is about 0.01 µm² low within two planes, and the table came out non-monotone near focus (SPEC Block 10). Read it next to the default table and keep the default for the production table until step 2 of procedure §3.4 exists.

## Cell 8: the production configuration (your decisions)

Copy the default configuration and change only the fields you have decided:
- `d_range_um` (from Cell 6);
- `mu_range_per_um` (from Cell 4);
- the camera fields (from Cell 5);
- `jitter_xy_um` and `jitter_z_um` (from Cell 3).

Record each choice as a decision before building the table on it.

```python
import json, dataclasses, allen_diameter.config as C
cfg = C.default_config()
# example only -- the values are yours:
# cfg = dataclasses.replace(cfg, phantom=dataclasses.replace(cfg.phantom, d_range_um=(0.2, 4.0)))
cfg.validate()
open(f"{OUT}/config_production.json", "w").write(cfg.to_json())
print(cfg.signature_hash("estimator"))
```

The estimator hash names the table. A real fit refuses a table with a different hash (procedure §3.11).

## Cell 9: the production table (davinci, not Colab)

The table is built with `scripts/pbs/build_table.pbs`: 100 tasks of 20 replicates, one core and 4 GB each, then the merge on the login node. Copy `config_production.json` to the cluster and pass it as `DIAM_CONFIG_JSON`. The commands are in SPEC §6.

Probe first:

```
qsub -J 0-1 -v DIAM_N_TOTAL=2,DIAM_N_TASKS=2,RUN_TAG=probe,DIAM_CONFIG_JSON=<path> scripts/pbs/build_table.pbs
```

Each replicate takes 10-140 s (sandbox timing).

## Cell 10: apply to the cell

```python
run("scripts/run_cell.py", "--specimen", str(SPECIMEN), "--table", f"{OUT}/tables/bias_table_<hash>",
    "--out-dir", f"{OUT}/cell", "--cache-dir", CACHE_DIR, "--config-json", f"{OUT}/config_production.json",
    "--registration-json", f"{OUT}/registration/registration_{SPECIMEN}.json",
    "--shift-x", str(SHIFT_X), "--shift-y", str(SHIFT_Y), "--z0", str(Z0))
```

Outputs:
- `nodes_<specimen>.csv`;
- `specimen_<specimen>/reconstruction.swc`, with radius = d_final / 2 on dendrite nodes and every other byte unchanged (D-013);
- `summary_<specimen>.json`, with the membrane-area ratio against Allen's radii.

Run it on a few stretches first, with `--nodes`.
