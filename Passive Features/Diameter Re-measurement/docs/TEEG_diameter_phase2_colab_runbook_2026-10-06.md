# Diameter pipeline, Phase II: Colab runbook

| Date | Change |
|---|---|
| 2026-10-06 | v1. Written with Block 11 (branch `sci/diameter-pipeline`). Every script below was run in the sandbox on a synthetic cell served through `allen_image_io.fetch_zblock`; none has run on Allen's data (the sandbox gets 403 from api.brain-map.org). |
| 2026-10-07 | v2. The cells now exist as a notebook, `notebooks/phase2_colab.ipynb`, with the same numbers; the code below is the notebook's. Corrections: `CACHE_DIR` is not the cache of cells 1-12 (Drive folder listing); Cell 5 suggests `renderer.jpeg_qtables`, the tables themselves, because the old `jpeg_qtables_file` was read by nothing (SPEC Blocks 1, 4, 11); Cell 6 writes its per-node CSV to Drive (it went into the clone). Added: Cell 3a, the node list (`scripts/list_stretches.py`); Cell 4 split into a one-stretch timing run and the full pilot; a streaming `run()` that stops "Run all" on a failure; Cell 8 through `config.with_overrides`, writing one configuration per in-focus blur of D-023; Cell 9's Pillow check and the merge with the same configuration, which the merge now enforces (every replicate row carries `sim_hash`); Cell 10's $\sigma_{\rm fit}$ runs. Evidence: the notebook's offline cells ran in the sandbox against a stub `google.colab`, and every command it builds parsed with its script's own parser **[run, sandbox]**; the cells that reach api.brain-map.org have still not run anywhere. |
| 2026-10-07 (later) | v3. The focus rule is the gradient energy (D-030): Cell 2 runs `test_smoke_focus` too; Cells 4b/4c describe and show both focus curves, the summary's `k_star_vs_dip_depth` and the nodes where the two rules differ; the default estimator hash is now 6f0397236447fe6f. Evidence: `test_smoke_focus.py` 7 pass and every suite of the workstream re-run in the sandbox **[run, sandbox]**; the cells that reach api.brain-map.org have still not run here. |
| 2026-10-07 (later, 2) | v4. Added Cell 4d: plane montages of chosen nodes with Allen's reconstruction drawn on them (`scripts/node_planes.py`, SPEC Block 11); each node is measured again as in Cell 4b, so its crops come from the cache. Evidence: `test_smoke_phase2.py` 4 pass, 3 skip with the montage checks (the re-measurement equals the pilot's and makes the same request; the image extent agrees with the profile sampler to 1e-6 grey levels; the drawn trace lies on the rendered tube within 0.05 um), nine mutants of the new code killed, every suite re-run **[run, sandbox]**; not yet run on Allen's data. |

This runbook covers the real-data steps, in order. Each cell names the line that must **appear** in its output. A missing line means the cell did not get as far as the code that prints it. Each step lists the decisions it feeds; those stay yours. Nothing in these scripts changes the configuration on its own.

**The notebook** [added 2026-10-07]: `Passive Features/Diameter Re-measurement/notebooks/phase2_colab.ipynb` on branch `sci/diameter-pipeline`. In Colab: File > Open notebook > GitHub, repository `Leonardodm00/Towards-EEG`, branch `sci/diameter-pipeline`; or upload the file. Its code is the code below; if the two ever differ, the notebook is the one that was checked. In a new session run Cells 0, 1 and 3a first: they define the paths, `run()` and the node list `NODES`.

Paths: `REPO` is the clone in `/content`. Drive holds data only: the HttpFetcher cache (`CACHE_DIR`, ~~the one cells 1-12 use~~) and the outputs (`OUT`). [corrected 2026-10-07: cells 1-12 cached their crops in `MyDrive/Colab Notebooks/Allen Slices/cache` and the SWC in `Allen Slices/SWC` (Drive listing, 2026-10-07). Phase II keeps its own cache, `MyDrive/allen_cache`, which a run on 2026-10-06 already created (it holds the SWC `H16.06.010.01.03.05.03_668702901_m.swc`). The two are kept apart because Cell 5 reads the JPEG tables of every crop in the cache, and the montage crops of cells 1-12 were fetched at other downsample levels.] Specimen 529878215 has no shift and no flip (design handoff, "Coordinate transform"), and its plane index is z / 0.28 um (`--z0 0`).

## Cell 0: Drive and constants

```python
from google.colab import drive
drive.mount("/content/drive")

import os
SPECIMEN = 529878215
# Phase II keeps its own image cache. Cells 1-12 of the viewer notebook cached their crops in
# "MyDrive/Colab Notebooks/Allen Slices/cache" (montage crops at several downsample levels), and
# Cell 5 reads the JPEG tables of every crop in the cache, so the two caches are kept apart.
CACHE_DIR = "/content/drive/MyDrive/allen_cache"
OUT = "/content/drive/MyDrive/diameters"
# global alignment of 529878215 (cells 1-12, design handoff): no shift, no flip; plane index = z / 0.28 um
SHIFT_X, SHIFT_Y, FLIP_H, Z0 = 0.0, 0.0, None, 0.0
REG_JSON = "%s/registration/registration_%d.json" % (OUT, SPECIMEN)
for d in (CACHE_DIR, OUT):
    os.makedirs(d, exist_ok=True)
print("[cell 0] specimen %d | cache %s | outputs %s" % (SPECIMEN, CACHE_DIR, OUT))
```

Expect: `[cell 0] specimen 529878215 | cache ... | outputs ...`.

## Cell 1: bootstrap (clone or update, `sys.path`) and `run()`

```python
import importlib, os, shlex, subprocess, sys, time
REPO, BRANCH = "/content/Towards-EEG", "sci/diameter-pipeline"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "50", "-b", BRANCH,
                    "https://github.com/Leonardodm00/Towards-EEG.git", REPO], check=True)
else:      # a clone made earlier in this runtime: update it before importing anything from it
    for step in (["fetch", "--depth", "50", "origin", BRANCH], ["checkout", BRANCH], ["merge", "--ff-only", "FETCH_HEAD"]):
        subprocess.run(["git", "-C", REPO] + step, check=True)
sys.path.insert(0, os.path.join(REPO, "Passive Features", "Diameter Re-measurement", "scripts"))
import colab_bootstrap
colab_bootstrap = importlib.reload(colab_bootstrap)   # this pull's version, if an older one was imported
ENV = colab_bootstrap.bootstrap(REPO, BRANCH)      # fetch + fast-forward, sys.path, versions, commit
WS = ENV["workstream"]


def run(script, *args, check=True, every=200):
    """Run a workstream script in a fresh Python, stream its output, raise if it fails (check=True)."""
    cmd = [sys.executable, "-u", script] + [str(a) for a in args]
    print("$ python " + " ".join(shlex.quote(c) for c in cmd[2:]), flush=True)
    t0, fetched = time.time(), 0
    p = subprocess.Popen(cmd, cwd=WS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in p.stdout:
            if line.startswith("  fetched id="):        # allen_image_io.HttpFetcher: one line per crop
                fetched += 1
                if fetched % every == 0:
                    print("  ... %d crops downloaded" % fetched, flush=True)
                continue
            print(line, end="", flush=True)
        rc = p.wait()
    except KeyboardInterrupt:                           # stopping the cell stops the script too
        p.terminate()
        p.wait()
        raise
    print("[run] exit %d after %.1f min; %d crops downloaded" % (rc, (time.time() - t0) / 60, fetched), flush=True)
    if rc != 0 and check:
        raise RuntimeError("%s exited with %d: the traceback is above" % (script, rc))
    return rc


ALIGN = ["--shift-x", SHIFT_X, "--shift-y", SHIFT_Y, "--z0", Z0] + ([] if FLIP_H is None else ["--flip-h", FLIP_H])
```

Expect: `[diameter] commit <hash> on sci/diameter-pipeline`. The hash should match the branch head on GitHub.

[corrected 2026-10-07: v1's `run()` captured the output and printed stdout, or stderr only when stdout was empty, so a traceback after any printed line was hidden, and a failing script did not stop the notebook. The new `run()` streams both, raises on a non-zero exit, and counts the fetcher's one-line-per-crop messages instead of printing them. v1 also imported `colab_bootstrap` from an existing clone before pulling, so a clone older than the bootstrap failed at the import (seen in the sandbox on a clone at `bd8b2b6`); an existing clone is now fast-forwarded first.]

## Cell 2: smoke tests in Colab (about 1 min)

```python
for t in ("test_smoke_config", "test_smoke_geometry", "test_smoke_fit", "test_smoke_camera_tables",
          "test_smoke_stretches", "test_smoke_focus"):
    run("tests/smoke/%s.py" % t)
```

Expect `-- ... 0 fail, 0 error, 0 todo` six times. `test_smoke_camera_tables` checks that Allen's JPEG tables reach the renderer and that this Pillow orders them as the configuration does (8.3 or newer). `test_smoke_focus` [added 2026-10-07] checks the focus rule of D-030 on closed forms and on rendered planes.

## Cell 3: registration on node 4505 and 5-10 stretches ("cell 13")

### Cell 3a: the node list [added 2026-10-07]

```python
import json
run("scripts/list_stretches.py", "--specimen", SPECIMEN, "--cache-dir", CACHE_DIR, "--out-dir", OUT + "/stretches",
    "--include", 4505, "--per-type", 3, "--min-nodes", 10)
with open("%s/stretches/stretches_%d.json" % (OUT, SPECIMEN)) as f:
    NODES = json.load(f)["nodes_arg"]
# NODES = "4505,1234,5678"        # your own choice, one node id per stretch
print("NODES =", NODES)
```

The script lists every unbranched dendrite stretch: type, nodes, length, path distance from the soma, branch order and Allen's median diameter. It suggests node 4505 first, then, for basal and for apical, 3 stretches of at least 10 nodes spread over path distance, each by its middle node. Spreading over path distance reaches the trunk (proximal) and the tuft (distal); it is a starting point, not a classification. Expect the counts per type, the chosen stretches, `the pilot measures every node of these stretches: N nodes` and `NODES = "4505,..."`. The table is `stretches/stretches_<specimen>.csv` (the notebook displays it). The pilot (Cell 4b) measures every node of every stretch in `NODES`, so N sets its cost.

### Cell 3b: registration

```python
run("scripts/registration_survey.py", "--specimen", SPECIMEN, "--nodes", NODES, "--cache-dir", CACHE_DIR,
    "--out-dir", OUT + "/registration", "--figures", *ALIGN)
```

Expect, for each node: `node 4505: ON THE PROCESS (lateral offset ...), s* ... um, dz* ... um, p ..., coverage ...` (or ALONGSIDE / FAR / NOT ON). Then a summary with `"verdicts"`. Figures go in `{OUT}/registration/figures/registration_<node>.png`.

This step feeds:
- the snap radius (design handoff §Method, step 2);
- the jitter amplitudes `jitter_xy_um` and `jitter_z_um`, which D-024 (vii) keeps at 0 until now. The summary's `suggested_jitter_um` is a suggestion, not a decision.

The file `registration_<specimen>.json` is what `--registration-json` takes below (`REG_JSON`). A node whose category is not ON is dropped from the selection S [corrected 2026-10-06: before Block 11, every registered node would have been dropped].

## Cell 4: pilot measurement, no table

### Cell 4a: node 4505's stretch, to time one stretch [added 2026-10-07]

```python
run("scripts/run_node.py", "--specimen", SPECIMEN, "--nodes", 4505, "--cache-dir", CACHE_DIR,
    "--out-dir", OUT + "/pilot", "--registration-json", REG_JSON, "--figures", "--background", *ALIGN)
```

Each node fetches one crop per plane: 7 planes for a flat node (3 on each side of its own), up to 23 for a steep one (`planes_half` and `planes_widen_for_tilt` in the configuration), plus one each for the figure and the background. The time per crop from Colab has not been measured: multiply the minutes printed here by the N of Cell 3a to budget Cell 4b.

### Cell 4b: every stretch of `NODES`

```python
run("scripts/run_node.py", "--specimen", SPECIMEN, "--nodes", NODES, "--cache-dir", CACHE_DIR,
    "--out-dir", OUT + "/pilot", "--registration-json", REG_JSON, "--figures", "--background", *ALIGN)
```

Every node of the stretches through `NODES` is measured with the same chain that builds the table; node 4505's stretch comes back from the cache. Expect `stretch of N nodes from node ... measured`, then the summary JSON with `"n_in_S"`. Outputs (overwriting Cell 4a's):
- `pilot_<specimen>.csv`;
- `pilot_summary_<specimen>.json`;
- `figures/node_<id>.png`, showing the focus curves and the profile with the fitted model.

The focus rule [changed 2026-10-07, D-030]: the sharpest plane $k^*$ is the plane of maximum gradient energy of the profile, $G = B^{-2}\int_{|v| \le r + 0.5\,\mu\mathrm{m}} (d\tilde I/dv)^2\,dv$, with $r$ Allen's radius of the node; the dip depth of the design handoff's Eq. 1 is computed on the same planes and recorded as `k_star_depth`. Each figure draws both curves against the plane index, with $k^*$ (red), the dip depth's choice (grey) and the SWC's own plane (dotted).

The notebook's Cell 4c prints the summary's key fields, among them `focus_rule gradient_energy`, `k_star_vs_dip_depth` (how many nodes the two rules put in different planes) and `estimator_hash 6f0397236447fe6f`; then the table of the nodes where the rules differ, and the figures, those nodes first (or the ids in `SHOW_NODES`). A pilot written before D-030 has no `k_star_depth`: re-run Cell 4b.

This step feeds:
- the phantom μ range: `suggested_phantom_mu_range_per_um`, the 10th-90th percentile of μ̂ over the nodes in S (procedure §3.10);
- the dark-flag threshold: `alpha_hat` percentiles and `dark_share_of_converged`;
- the number of calibration nodes, `n_calibration_nodes` (Cell 7).

### Cell 4d: the planes of a node, with Allen's reconstruction drawn on them [added 2026-10-07]

```python
# Cell 4d: the planes of a node with Allen's reconstruction drawn on them (run Cell 4c first)
import os
from IPython.display import Image, display
PLANE_NODES = (list(SHOW_NODES) + [int(n) for n in moved["node_id"] if int(n) not in SHOW_NODES])[:6]
if not PLANE_NODES:       # the two focus rules agree on every node: the first measured nodes instead
    PLANE_NODES = [int(n) for n in rows.loc[rows["z_sub_um"].notna(), "node_id"][:3]]
run("scripts/node_planes.py", "--specimen", SPECIMEN, "--nodes", ",".join(str(n) for n in PLANE_NODES),
    "--cache-dir", CACHE_DIR, "--pilot-csv", "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN),
    "--out-dir", OUT + "/pilot/planes", *ALIGN)
for n in PLANE_NODES:
    p = "%s/pilot/planes/planes_%d.png" % (OUT, n)
    if os.path.exists(p):
        print(os.path.basename(p))
        display(Image(filename=p))
```

Run after Cell 4c: it takes the nodes of `SHOW_NODES`, then those of Cell 4c's table, at most 6 (if the two focus rules agree everywhere, the first three measured nodes). One figure per node, one panel per plane the focus rule scored (the SWC's own plane $\pm 3$, widened for tilt), all on one grey scale. On each panel: Allen's traced centre lines with their $\pm r$ outline (the SWC frustum), cyan for the measured stretch and amber for the other dendrites in the block, fainter the farther a segment's depth is from the plane's; the SWC node and the fit's profile line (black); in plane $k^*$ (red frame) the fitted edges (red ticks); the dip depth's plane in a grey dashed frame. Each panel title gives the plane, its depth minus the node's, and the two focus scores $G$ and $F$.

Each node is measured again exactly as in Cell 4b, so its crops come from the cache. Expect one `[planes] node N: ... | same as the pilot` line per node, then `[planes] wrote N montages (0 skipped) ...; crops: ... from the cache, 0 downloaded`. `DIFFERS from the pilot` means the pilot was measured with another configuration or code version: re-run Cell 4b.

Use it to judge, node by node, three things the numbers of Cell 4c cannot show: whether Allen's trace sits on the dendrite in the image (a lateral offset is a registration matter, Cell 3b); whether $k^*$ is the plane where the dendrite's edges are sharpest; and whether another neurite crosses the profile line near $k^*$ (D-030 open point on overlaps).

## Cell 5: camera-chain inputs

```python
run("scripts/camera_calibration.py", "--specimen", SPECIMEN, "--cache-dir", CACHE_DIR, "--out-dir", OUT + "/camera",
    "--pilot-summary", "%s/pilot/pilot_summary_%d.json" % (OUT, SPECIMEN))
```

Expect `"renderer": {...}` with three fields:
- ~~`jpeg_qtables_file`~~ `jpeg_qtables`: Allen's tables, read from the cached crops. `table_sets` should show a single set, and `pillow_version` 8.3 or newer. [corrected 2026-10-07: the old field held a file path that the renderer never read, so a table built on it would have used quality 85; the tables themselves now go into the configuration, and `jpeg_qtables_<specimen>.json` stays as a record.]
- `background_B_gl`: the median of the masked block medians.
- `noise_sd_gl`: the injected SD whose post-JPEG flat-field SD matches the real background's. JPEG removes much of white noise, so the background SD read on the images is smaller than the injected SD. This value is an upper bound, because tissue texture adds to the real background SD.

Black level and gain cannot be identified from these crops. They stay as configured. If `n_tables` is 2, the crops are colour JPEGs: the synthetic planes are grey and use the first (luminance) table only (SPEC §8).

## Cell 6: Allen's radius distribution (sets d_max)

```python
os.makedirs(OUT + "/radius", exist_ok=True)
run("scripts/allen_radius_distribution.py", "--fetch", SPECIMEN, "--cache-dir", CACHE_DIR,
    "--out", "%s/radius/radius_summary_%d.csv" % (OUT, SPECIMEN),
    "--out-nodes", "%s/radius/radius_nodes_%d.csv" % (OUT, SPECIMEN))
```

Expect `pooled dendrite diameter (um): p0=..., ..., p100=...`, a text histogram and `wrote ...`. [corrected 2026-10-07: v1 passed only `--out`, so the per-node CSV went to the default `allen_radius_nodes.csv` inside the clone, lost with the runtime.]

This step feeds `phantom.d_range_um`, which is temporary until now (D-024 (iii), SPEC §8).

## Cell 7: kernel calibration, first stage (diagnostic)

```python
os.makedirs(OUT + "/calibration", exist_ok=True)
SCANS = "%s/calibration/scans_%d.json" % (OUT, SPECIMEN)
rc = run("scripts/calibrate_kernel.py", "real", "--specimen", SPECIMEN, "--nodes-csv",
         "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN), "--cache-dir", CACHE_DIR, "--out", SCANS,
         "--registration-json", REG_JSON, *ALIGN, check=False)
if rc == 0:
    run("scripts/calibrate_kernel.py", "fit", "--scans", SCANS,
        "--out", "%s/calibration/calibration_%d.json" % (OUT, SPECIMEN), check=False)
```

Expect `N candidate nodes`, then `growth fit (gaussian_core_width2, symmetric, cubic): N nodes`, the table of knots, and `residual RMS by |k - k*|`. You need at least two thin, faint, flat nodes (d̂ ≤ 0.3 µm, φ ≤ 10°, α̂ ≤ 0.5). Measure more stretches in Cell 4 if there are fewer. A failure here does not stop the notebook (`check=False`): the step is a diagnostic.

The first-stage table is **not** a calibrated kernel. On rendered thin phantoms its growth is about 0.01 µm² low within two planes of focus, and the table came out non-monotone near focus (SPEC Block 10). Read it next to the default table and keep the default for the production table until step 2 of procedure §3.4 exists.

## Cell 8: the production configuration (your decisions)

Copy the default configuration and change only the fields you have decided:
- `d_range_um` (from Cell 6);
- `mu_range_per_um` (from Cell 4);
- the camera fields (from Cell 5);
- `jitter_xy_um` and `jitter_z_um` (from Cell 3).

Record each choice as a decision before building the table on it.

```python
import importlib, json
import pandas as pd
import allen_diameter.config as C
C = importlib.reload(C)                      # the code of this session's pull


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


REG = load("%s/registration/registration_summary_%d.json" % (OUT, SPECIMEN))
PILOT = load("%s/pilot/pilot_summary_%d.json" % (OUT, SPECIMEN))
CAM = load("%s/camera/camera_%d.json" % (OUT, SPECIMEN))
try:
    RAD = pd.read_csv("%s/radius/radius_summary_%d.csv" % (OUT, SPECIMEN)).iloc[0].to_dict()
except FileNotFoundError:
    RAD = {}
base = C.default_config()
cam = CAM.get("renderer", {})
print("suggested by the runs                                      [current default]")
print("  d range      Allen dendrite diameter p99 %s um, max %s um   [%s]"
      % (RAD.get("dend_p99"), RAD.get("dend_p100"), base.phantom.d_range_um))
print("  mu range     %s   [%s]" % (PILOT.get("suggested_phantom_mu_range_per_um"), base.phantom.mu_range_per_um))
print("  jitter       %s   [%s, %s]" % (REG.get("suggested_jitter_um"), base.phantom.jitter_xy_um, base.phantom.jitter_z_um))
print("  background   %s   [%s]" % (cam.get("background_B_gl"), base.renderer.background_B_gl))
print("  noise SD     %s   [%s]" % (cam.get("noise_sd_gl"), base.renderer.noise_sd_gl))
print("  JPEG tables  %s table set(s) in the cache, Pillow %s   [none: quality %d]"
      % (len(CAM.get("table_sets", [])), CAM.get("pillow_version"), base.renderer.jpeg_quality))

DECIDED = {
    # "phantom.d_range_um": (0.2, 4.0),                                         # Cell 6 sets d_max (D-024 iii)
    # "phantom.mu_range_per_um": PILOT["suggested_phantom_mu_range_per_um"],   # Cell 4 (procedure s.3.10)
    # "phantom.jitter_xy_um": REG["suggested_jitter_um"]["jitter_xy_um"],      # Cell 3 (D-024 vii)
    # "phantom.jitter_z_um": REG["suggested_jitter_um"]["jitter_z_um"],
    # "renderer.background_B_gl": cam["background_B_gl"],                      # Cell 5
    # "renderer.noise_sd_gl": cam["noise_sd_gl"],                              # Cell 5 (an upper bound)
    # "renderer.jpeg_qtables": cam["jpeg_qtables"],                            # Cell 5
}
cfg = C.with_overrides(base, DECIDED)
PENDING = ("phantom.d_range_um", "phantom.mu_range_per_um", "phantom.jitter_xy_um", "phantom.jitter_z_um",
           "renderer.background_B_gl", "renderer.noise_sd_gl", "renderer.jpeg_qtables")
left = [k for k in PENDING if k not in DECIDED]
print("\nstill at the provisional default: %s" % (", ".join(left) if left else "none"))
for s in cfg.measure.sigma_fit_study_um:           # D-023: one table per in-focus blur
    c = C.with_sigma_fit(cfg, s)
    name = "config_production.json" if s == cfg.measure.sigma_fit_um else "config_sigma%.3f.json" % s
    with open("%s/%s" % (OUT, name), "w") as f:
        f.write(c.to_json())
    print("%-24s sigma_fit %.3f um -> table bias_table_%s" % (name, s, c.signature_hash("estimator")))
```

`config.with_overrides` [added 2026-10-07] checks every name, validates the result, and puts each value in the field's declared form: a JSON list becomes the tuple the configuration holds, and an integer given for a float field becomes a float, so the same value always gives the same hash. The cell writes one configuration per in-focus blur of the D-023 study: `config_production.json` (the deliverable, σ_fit = 0.099 µm), `config_sigma0.080.json` and `config_sigma0.125.json`. Expect three lines `... -> table bias_table_<hash>`.

The estimator hash names the table. A real fit refuses a table with a different hash (procedure §3.11).

## Cell 9: the production table (davinci, not Colab)

The table is built with `scripts/pbs/build_table.pbs`: 100 tasks of 20 replicates, one core and 4 GB each, then the merge on the login node. Copy `config_production.json` to the cluster, to a path **without spaces** (for example `$HOME/diam/config_production.json`; `qsub -v` takes a comma-separated list, and a value with a space is fragile there), and pass it as `DIAM_CONFIG_JSON`. The commands are in SPEC §6 and in the paths reference (project knowledge, `claude/TEEG_diameter_HPC_paths_reference.md`).

Update the code and check Pillow first [added 2026-10-07: 8.3 or newer, the order of the JPEG tables]:

```
cd "/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/Diameter Re-measurement" && git fetch origin && git checkout sci/diameter-pipeline && git pull --ff-only && conda activate spine_env && python -c "import PIL; print('Pillow', PIL.__version__)" && python tests/smoke/test_smoke_config.py && python tests/smoke/test_smoke_camera_tables.py
```

Probe first:

```
qsub -J 0-1 -v DIAM_N_TOTAL=2,DIAM_N_TASKS=2,RUN_TAG=probe,DIAM_CONFIG_JSON=<path> scripts/pbs/build_table.pbs
```

Each replicate takes 10-140 s (sandbox timing). `DIAM_N_TOTAL` (default 2000) sets the number of replicates; `phantom.n_replicates` in the configuration is a record only.

Merge with the **same** configuration [added 2026-10-07]. Without `--config-json` the merge assumes the default configuration. When the estimator hashes agree, the row files are then found, and before 2026-10-07 the default ranges and signature would have been written into the table. Since 2026-10-07 every row records the configuration it was rendered under (`sim_hash`), and the merge stops with `merge refused` when it is given another one (SPEC Block 6):

```
python scripts/build_table.py merge --out-dir "/davinci-1/home/ldellamea/Human Neurons Fitting/diameter_tables/v1" --config-json $HOME/diam/config_production.json
```

Copy `bias_table_<hash>.npz` and `bias_table_<hash>.json` to `MyDrive/diameters/tables/`. For the σ_fit study, repeat with `RUN_TAG=v1_s0080` and `config_sigma0.080.json`, then `RUN_TAG=v1_s0125` and `config_sigma0.125.json`.

## Cell 10: apply to the cell

```python
import importlib, json
import allen_diameter.config as C
C = importlib.reload(C)
CFG = OUT + "/config_production.json"
with open(CFG) as f:
    H = C.config_from_dict(json.load(f)).signature_hash("estimator")
TABLE = "%s/tables/bias_table_%s" % (OUT, H)
missing = [TABLE + e for e in (".npz", ".json") if not os.path.exists(TABLE + e)]
if missing:
    raise FileNotFoundError("copy these from davinci first (Cell 9, step 5): %s" % missing)
run("scripts/run_cell.py", "--specimen", SPECIMEN, "--table", TABLE, "--out-dir", OUT + "/cell",
    "--cache-dir", CACHE_DIR, "--config-json", CFG, "--registration-json", REG_JSON, "--nodes", NODES, *ALIGN)
```

Outputs:
- `nodes_<specimen>.csv`;
- `specimen_<specimen>/reconstruction.swc`, with radius = d_final / 2 on dendrite nodes and every other byte unchanged (D-013);
- `summary_<specimen>.json`, with the membrane-area ratio against Allen's radii.

Run it on a few stretches first, with `--nodes` (above); the notebook's Cell 10b then runs every dendrite node (the same command without `--nodes`), and Cell 10c runs each σ_fit study table that is on Drive into `cell_sigma<value>/`. No script yet joins the three runs into the `d_tilde_sigma_spread_um` column (SPEC §8): compare the three `nodes_<specimen>.csv` by `node_id`.
