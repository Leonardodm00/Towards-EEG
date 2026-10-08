# Diameter pipeline, Phase II (Colab), specimen 529878215: the cells of notebooks/phase2_colab.ipynb as plain
# Python. Paste each block between two '# ===== CELL' banners into its own Colab cell and run them in order.
# Cell 9 runs on davinci, not in Colab: its commands are in the comment block before Cell 10.
# Same code as the notebook and the runbook (2026-10-08). Pure ASCII.

# ====================================================================================================
# ===== CELL 0
# ====================================================================================================
# DIAMETER PIPELINE, PHASE II (COLAB): SPECIMEN 529878215
#
# The Colab cells of the runbook `docs/TEEG_diameter_phase2_colab_runbook_2026-10-06.md`, with the same cell
# numbers. Code is cloned fresh from GitHub (branch `sci/diameter-pipeline`) at every session; Drive holds
# only data: the image cache and the outputs.
#
# How to run. Top to bottom. Each cell names the line that must appear in its output: a missing line means the
# cell stopped before the code that prints it. A script that fails raises, so "Run all" stops before a cell
# that needs its files. In a new session run Cells 0, 1 and 3a first: they define the paths, the `run()`
# helper and the node list `NODES`.
#
# Decisions stay yours. Two cells need you: 3a (which stretches) and 8 (the production configuration). Nothing
# in these scripts changes the configuration by itself.
#
# Time. Each measured node fetches one crop per plane from Allen's server: 7 planes for a flat node (3 on each
# side of its own), up to 23 for a steep one (from the configuration), plus one each for the figure and the
# background. The time per crop has not been measured from Colab, so Cell 4a times one stretch first. Re-runs
# are fast: every crop is cached on Drive.
# CELL 0: DRIVE AND CONSTANTS
# Expect `[cell 0] specimen 529878215 | cache ... | outputs ...`.
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

# ====================================================================================================
# ===== CELL 1
# ====================================================================================================
# CELL 1: CODE (CLONE OR UPDATE), `RUN()` HELPER
# Expect `[diameter] commit <hash> on sci/diameter-pipeline`; the hash should be the branch head on GitHub.
#
# `run()` starts each script in a fresh Python, so the code that runs is the code just pulled. It streams the
# output and counts the image fetcher's one-line-per-crop messages instead of printing them.
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

# ====================================================================================================
# ===== CELL 2
# ====================================================================================================
# CELL 2: SMOKE TESTS ON THIS RUNTIME (ABOUT 1 MIN)
# Expect `-- ... 0 fail, 0 error, 0 todo ...` six times. `test_smoke_camera_tables` checks that Allen's JPEG
# tables reach the renderer and that this Pillow orders them as the tables in the configuration (8.3 or
# newer). `test_smoke_focus` checks the focus rule of D-030 on closed forms and on rendered planes.
for t in ("test_smoke_config", "test_smoke_geometry", "test_smoke_fit", "test_smoke_camera_tables",
          "test_smoke_stretches", "test_smoke_focus"):
    run("tests/smoke/%s.py" % t)

# ====================================================================================================
# ===== CELL 3a
# ====================================================================================================
# CELL 3A: THE NODE LIST
# `list_stretches.py` reads the SWC and lists every unbranched dendrite stretch: type, nodes, length, path
# distance from the soma, branch order and Allen's median diameter. It then suggests one node per stretch:
# node 4505 first, then, for basal and for apical, 3 stretches of at least 10 nodes spread over path distance.
# Spreading reaches the trunk (proximal) and the tuft (distal); the suggestion is a starting point, not a
# classification.
#
# Expect the per-type counts, the chosen stretches, `the pilot measures every node of these stretches: N
# nodes` and `NODES = "4505,..."`. Edit `NODES` to choose other stretches; the table is in the next cell. The
# pilot (Cell 4b) measures every node of every stretch in `NODES`, so N sets its cost.
import json
run("scripts/list_stretches.py", "--specimen", SPECIMEN, "--cache-dir", CACHE_DIR, "--out-dir", OUT + "/stretches",
    "--include", 4505, "--per-type", 3, "--min-nodes", 10)
with open("%s/stretches/stretches_%d.json" % (OUT, SPECIMEN)) as f:
    NODES = json.load(f)["nodes_arg"]
# NODES = "4505,1234,5678"        # your own choice, one node id per stretch
print("NODES =", NODES)

# ====================================================================================================
# ===== CELL 3a (optional: the stretch table)
# ====================================================================================================
# (optional) the stretch table, by type and path distance from the soma
import pandas as pd
pd.set_option("display.max_rows", 400)
tab = pd.read_csv("%s/stretches/stretches_%d.csv" % (OUT, SPECIMEN))
display(tab.sort_values(["type", "path_start_um"]).reset_index(drop=True))

# ====================================================================================================
# ===== CELL 3b
# ====================================================================================================
# CELL 3B: REGISTRATION ON THE NODE LIST ("CELL 13")
# For each node, `registration_check` of the 2026-09-23 modules on a block around the traced path. Expect one
# line per node: `node 4505: ON THE PROCESS (lateral offset ...), s* ... um, dz* ... um, p ..., coverage ...`
# (or ALONGSIDE / FAR / NOT ON), then the summary with `"verdicts"`.
#
# Feeds the snap radius (design handoff, Method step 2) and the jitter amplitudes `phantom.jitter_xy_um`,
# `jitter_z_um`, which D-024 (vii) keeps at 0 until now: `suggested_jitter_um` is a suggestion only. A node
# whose category is not ON leaves the selection S.
run("scripts/registration_survey.py", "--specimen", SPECIMEN, "--nodes", NODES, "--cache-dir", CACHE_DIR,
    "--out-dir", OUT + "/registration", "--figures", *ALIGN)

# ====================================================================================================
# ===== CELL 3b (optional: the registration figures)
# ====================================================================================================
# (optional) the registration figures
import glob
from IPython.display import Image, display
for p in sorted(glob.glob(OUT + "/registration/figures/registration_*.png")):
    print(os.path.basename(p))
    display(Image(filename=p))

# ====================================================================================================
# ===== CELL 4a
# ====================================================================================================
# CELL 4A: PILOT ON NODE 4505'S STRETCH (TIMING)
# The per-node chain of the bias table (Block 5) on every node of the stretch holding node 4505, no table.
# Expect `stretch of N nodes from node ... measured`, the summary JSON with `"n_in_S"`, then the minutes.
# Multiply by the N of Cell 3a to budget Cell 4b.
run("scripts/run_node.py", "--specimen", SPECIMEN, "--nodes", 4505, "--cache-dir", CACHE_DIR,
    "--out-dir", OUT + "/pilot", "--registration-json", REG_JSON, "--figures", "--background", *ALIGN)

# ====================================================================================================
# ===== CELL 4b
# ====================================================================================================
# CELL 4B: PILOT ON EVERY STRETCH OF `NODES`
# Same expectations, one `stretch of ... measured` line per stretch. Node 4505's stretch comes back from the
# cache. The outputs of Cell 4a are overwritten: `pilot_<specimen>.csv`, `pilot_summary_<specimen>.json`,
# `figures/node_<id>.png`.
#
# Feeds the phantom mu range (`suggested_phantom_mu_range_per_um`, the 10th-90th percentile of mu over the
# nodes in S, procedure s.3.10), the dark-flag threshold (alpha percentiles, `dark_share_of_converged`) and
# the number of calibration nodes (Cell 7).
#
# Since D-030 (2026-10-07) the sharpest plane k* is the plane of maximum gradient energy of the profile,
# G = B^-2 * integral over |v| <= r + 0.5 um of (dI~/dv)^2 dv; the dip depth, the old rule, is computed on the
# same planes and recorded as `k_star_depth`. Each figure draws both curves against the plane index, with k*
# (red), the dip depth's choice (grey) and the SWC's own plane (dotted). Cell 4c lists the nodes where the two
# rules differ and shows their figures first; put node ids in `SHOW_NODES` to see others first.
run("scripts/run_node.py", "--specimen", SPECIMEN, "--nodes", NODES, "--cache-dir", CACHE_DIR,
    "--out-dir", OUT + "/pilot", "--registration-json", REG_JSON, "--figures", "--background", *ALIGN)

# ====================================================================================================
# ===== CELL 4c
# ====================================================================================================
# CELL 4C: WHAT THE PILOT SAYS, AND THE NODE FIGURES
# Expect the summary fields, among them `focus_rule gradient_energy`, `k_star_vs_dip_depth {...}` and
# `estimator_hash 6f0397236447fe6f` (the default configuration since D-030), then `N nodes where the
# gradient energy (k_star) and the dip depth (k_star_depth) pick different planes`, their table, and the
# figures.
# Cell 4c: what the pilot says, and the node figures (focus curves; profile with the fitted model)
# Re-run Cell 4b after pulling: a pilot written before D-030 has no k_star_depth column.
import glob, json, os
import pandas as pd
from IPython.display import Image, display
with open("%s/pilot/pilot_summary_%d.json" % (OUT, SPECIMEN)) as f:
    PILOT = json.load(f)
for k in ("n_nodes", "n_in_S", "reject_counts", "suggested_phantom_mu_range_per_um", "dark_share_of_converged",
          "n_calibration_nodes", "background", "focus_rule", "k_star_vs_dip_depth", "estimator_hash"):
    print("%-36s %s" % (k, PILOT.get(k)))
for k in ("d_hat_um", "mu_hat_per_um", "alpha_hat", "d_hat_over_allen_d", "phi_deg"):
    print("%-20s %s" % (k, {q: round(v, 3) for q, v in PILOT.get(k, {}).items()}))
rows = pd.read_csv("%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN))
moved = rows[rows["z_sub_um"].notna() & (rows["k_star"] != rows["k_star_depth"])]
print(len(moved), "nodes where the gradient energy (k_star) and the dip depth (k_star_depth) pick different planes")
display(moved[["node_id", "k_star", "k_star_depth", "z_um", "d_hat_um", "allen_radius_um", "in_S", "flags"]].head(30))
SHOW_NODES = []          # node ids whose figures come first, e.g. [8441]
ids = list(SHOW_NODES) + [int(n) for n in moved["node_id"] if int(n) not in SHOW_NODES]
figs = [p for p in (OUT + "/pilot/figures/node_%d.png" % n for n in ids) if os.path.exists(p)][:12]
figs += [p for p in sorted(glob.glob(OUT + "/pilot/figures/node_*.png")) if p not in figs][:max(0, 6 - len(figs))]
print(len(figs), "node figures:")
for p in figs:
    print(os.path.basename(p))
    display(Image(filename=p))

# ====================================================================================================
# ===== CELL 4d
# ====================================================================================================
# CELL 4D: THE PLANES OF A NODE, WITH ALLEN'S RECONSTRUCTION DRAWN ON THEM
# Run after Cell 4c: it takes the nodes of `SHOW_NODES`, then those of Cell 4c's table, at most 6 (if the two
# focus rules agree everywhere, the first three measured nodes). One figure per node, one panel per plane the
# focus rule scored (the SWC's own plane +-3, widened for tilt), all on one grey scale. On each panel: Allen's
# traced centre lines with their +-r outline (the SWC frustum), cyan for the measured stretch and amber for
# the other dendrites in the block, fainter the farther a segment's depth is from the plane's; the SWC node
# and the fit's profile line (black); in plane k* (red frame) the fitted edges (red ticks); the dip depth's
# plane in a grey dashed frame. Each panel title gives the plane, its depth minus the node's, and the two
# focus scores G and F.
#
# Each node is measured again exactly as in Cell 4b, so its crops come from the cache. Expect one `[planes]
# node N: ... | same as the pilot` line per node, then `[planes] wrote N montages (0 skipped) ...; crops: ...
# from the cache, 0 downloaded`. `DIFFERS from the pilot` means the pilot was measured with another
# configuration or code version: re-run Cell 4b.
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

# ====================================================================================================
# ===== CELL 4e
# ====================================================================================================
# CELL 4E: CONSECUTIVE-PLANE DIFFERENCES AROUND A NODE
# The proposal of 2026-10-08, as a diagnostic that changes no measurement. For each node, the planes k_SWC - 6
# to k_SWC + 6 of its block, cropped to the 10 x 10 um square about the node. Between each pair of consecutive
# planes the difference D = I_(k+1) - I_k is formed, its negative values are set to zero, and S+ is the mean
# of what remains over the pixels of the square (grey levels per pixel); S-, the mean of the negative part, is
# the same measure taken from the other end of the stack. The figure shows the planes (frames: k* red, the dip
# depth's plane grey dashed, the SWC plane dotted), the positive part of each difference (blue), and the two
# curves, with the dip of S+ between its two largest maxima.
#
# Each node is located as the pilot measures it, so the pilot's planes come from the cache; the planes beyond
# them are fetched, one crop each. Expect one `[planediff] node N: planes ... (0 missing), ... pixels; dip of
# S+ at a->b; k* ..., dip-depth plane ..., SWC plane ...` line per node, then `[planediff] wrote N figures (0
# skipped) ...`. Add `"--band-um", 3.5` to the command to count only the pixels within 3.5 um of the traced
# stretch.
#
# What to expect (synthetic tubes, SPEC Block 11): the dip falls on a pair next to the tube's centre for a 0.5
# um and a 1.5 um tube, with or without the camera's noise; for a faint 5 um tube it does only without noise,
# because the change between its planes is smaller than the noise. Read a trunk's curve with that in mind.
# Cell 4e: consecutive-plane differences around a node (needs Cells 0 and 1 only)
import os
from IPython.display import Image, display
DIFF_NODES = [2, 3]       # node ids; 2 and 3 are trunk nodes of the pilot
run("scripts/plane_differences.py", "--specimen", SPECIMEN, "--nodes", ",".join(str(n) for n in DIFF_NODES),
    "--planes-half", 6, "--cache-dir", CACHE_DIR, "--out-dir", OUT + "/pilot/planediff", *ALIGN)
for n in DIFF_NODES:
    p = "%s/pilot/planediff/planediff_%d.png" % (OUT, n)
    if os.path.exists(p):
        print(os.path.basename(p))
        display(Image(filename=p))

# ====================================================================================================
# ===== CELL 5
# ====================================================================================================
# CELL 5: CAMERA-CHAIN INPUTS
# Expect `"renderer": {` holding three suggestions:
# - `jpeg_qtables`: Allen's quantization tables read from the cached crops, the tables themselves, ready for
# the configuration. `table_sets` should show one set, and `pillow_version` 8.3 or newer. Corrected
# 2026-10-07: the old field `jpeg_qtables_file` was never read by the renderer.
# - `background_B_gl`: the median of the masked block medians.
# - `noise_sd_gl`: the injected SD whose post-JPEG flat-field SD matches the real background's. JPEG removes
# much of white noise, so the SD read on the images is smaller than the injected one. This value is an upper
# bound, because tissue texture adds to the real background SD.
#
# Black level and gain cannot be identified from these crops; they stay as configured.
run("scripts/camera_calibration.py", "--specimen", SPECIMEN, "--cache-dir", CACHE_DIR, "--out-dir", OUT + "/camera",
    "--pilot-summary", "%s/pilot/pilot_summary_%d.json" % (OUT, SPECIMEN))

# ====================================================================================================
# ===== CELL 6
# ====================================================================================================
# CELL 6: ALLEN'S RADIUS DISTRIBUTION (SETS D_MAX)
# Expect `pooled dendrite diameter (um): p0=..., ..., p100=...`, a text histogram, `wrote ...`. Feeds
# `phantom.d_range_um`, temporary until now (D-024 (iii)).
os.makedirs(OUT + "/radius", exist_ok=True)
run("scripts/allen_radius_distribution.py", "--fetch", SPECIMEN, "--cache-dir", CACHE_DIR,
    "--out", "%s/radius/radius_summary_%d.csv" % (OUT, SPECIMEN),
    "--out-nodes", "%s/radius/radius_nodes_%d.csv" % (OUT, SPECIMEN))

# ====================================================================================================
# ===== CELL 7
# ====================================================================================================
# CELL 7: KERNEL CALIBRATION, FIRST STAGE (DIAGNOSTIC ONLY)
# The calibration nodes of the pilot (thin, faint, flat:  d <= 0.3 um, phi <= 10 deg, alpha <= 0.5) are
# re-measured and scanned plane by plane, then the growth of the core width is fitted. Expect `N candidate
# nodes`, `growth fit (gaussian_core_width2, symmetric, cubic): N nodes`, the knot table and `residual RMS by
# |k - k*|`. Fewer than two candidates: measure more stretches in Cell 4b. A failure here does not stop the
# notebook.
#
# The first-stage table is not a calibrated kernel: on rendered thin phantoms its growth is about 0.01 um^2
# low within two planes, and it came out non-monotone near focus (SPEC Block 10). Keep the default kernel for
# the production table until step 2 of procedure s.3.4 exists.
os.makedirs(OUT + "/calibration", exist_ok=True)
SCANS = "%s/calibration/scans_%d.json" % (OUT, SPECIMEN)
rc = run("scripts/calibrate_kernel.py", "real", "--specimen", SPECIMEN, "--nodes-csv",
         "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN), "--cache-dir", CACHE_DIR, "--out", SCANS,
         "--registration-json", REG_JSON, *ALIGN, check=False)
if rc == 0:
    run("scripts/calibrate_kernel.py", "fit", "--scans", SCANS,
        "--out", "%s/calibration/calibration_%d.json" % (OUT, SPECIMEN), check=False)

# ====================================================================================================
# ===== CELL 8
# ====================================================================================================
# CELL 8: THE PRODUCTION CONFIGURATION (YOUR DECISIONS)
# The cell prints what Cells 3-6 suggest next to the current defaults. Uncomment and set, in `DECIDED`, only
# what you have decided; record each choice as a decision before the table is built on it. `with_overrides`
# checks every name and value, and turns JSON lists into the tuples the configuration holds.
#
# It writes one configuration per in-focus blur of the D-023 study: `config_production.json` (the deliverable,
# sigma_fit = 0.099 um) and `config_sigma0.080.json`, `config_sigma0.125.json`. Expect three lines `... ->
# table bias_table_<hash>`: the hash names the table, and a real fit refuses a table with another hash
# (procedure s.3.11).
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

# ====================================================================================================
# ===== CELL 9: DAVINCI, NOT COLAB (nothing to paste here)
# ====================================================================================================
# CELL 9: THE PRODUCTION TABLE ON DAVINCI (NOT COLAB)
# `scripts/pbs/build_table.pbs`: 100 tasks of 20 replicates, one core and 4 GB each; then the merge on the
# login node.
#
# 1. Copy `MyDrive/diameters/config_production.json` (and the two `config_sigma*.json`) to davinci into a
# folder whose path has no spaces, e.g. `~/diam/`. `qsub -v` takes a comma-separated list, and a value with a
# space is fragile there.
# 2. Update the code, check Pillow (8.3 or newer: the JPEG tables' order) and run the two new suites:
#     cd "/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/Diameter Re-measurement" && git fetch origin && git checkout sci/diameter-pipeline && git pull --ff-only && conda activate spine_env && python -c "import PIL; print('Pillow', PIL.__version__)" && python tests/smoke/test_smoke_config.py && python tests/smoke/test_smoke_camera_tables.py
# 3. Dry run, two-replicate probe, then the full array (from the same directory):
#     qsub -v DIAM_DRYRUN=1,RUN_TAG=v1,DIAM_CONFIG_JSON=$HOME/diam/config_production.json scripts/pbs/build_table.pbs
#     qsub -J 0-1 -v DIAM_N_TOTAL=2,DIAM_N_TASKS=2,RUN_TAG=probe,DIAM_CONFIG_JSON=$HOME/diam/config_production.json scripts/pbs/build_table.pbs
#     qsub -v RUN_TAG=v1,DIAM_CONFIG_JSON=$HOME/diam/config_production.json scripts/pbs/build_table.pbs
# Expect in every `.o` file in `$HOME`: `[diam-table] env    spine_env`, a `Pillow` version line,
# `[diam-table] task <i> done`. `DIAM_N_TOTAL` (default 2000) sets the number of replicates;
# `phantom.n_replicates` in the configuration is a record only.
#
# 4. Merge with the same configuration, with `spine_env` active. Every row records the configuration it was
# rendered under, and the merge stops with `merge refused` when it is given another one; without
# `--config-json` it assumes the default configuration:
#     python scripts/build_table.py merge --out-dir "/davinci-1/home/ldellamea/Human Neurons Fitting/diameter_tables/v1" --config-json $HOME/diam/config_production.json
# Expect `... rows from 100 files, <n> in S; smoothing <s> -> .../bias_table_<hash>.npz/.json`, with the hash
# Cell 8 printed for `config_production.json`.
#
# 5. Copy `bias_table_<hash>.npz` and `bias_table_<hash>.json` to `MyDrive/diameters/tables/`.
#
# The sigma_fit study (D-023): repeat 3-5 with `RUN_TAG=v1_s0080` and `config_sigma0.080.json`, then
# `RUN_TAG=v1_s0125` and `config_sigma0.125.json`.

# ====================================================================================================
# ===== CELL 10
# ====================================================================================================
# CELL 10: APPLY THE TABLE TO THE CELL
# First the stretches of `NODES`, then every dendrite node. Expect `stretch of N nodes from node ... done` per
# stretch, then the summary with `area_ratio`. Outputs in `diameters/cell/`: `nodes_<specimen>.csv`;
# `specimen_<specimen>/reconstruction.swc`, radius = d_final/2 on dendrite nodes and every other byte
# unchanged (D-013); `summary_<specimen>.json`.
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

# ====================================================================================================
# ===== CELL 10b
# ====================================================================================================
# Cell 10b: every dendrite node of the cell (long: every node fetches its own crops)
run("scripts/run_cell.py", "--specimen", SPECIMEN, "--table", TABLE, "--out-dir", OUT + "/cell",
    "--cache-dir", CACHE_DIR, "--config-json", CFG, "--registration-json", REG_JSON, *ALIGN)

# ====================================================================================================
# ===== CELL 10c
# ====================================================================================================
# Cell 10c: the sigma_fit study (D-023), one run per study table that is on Drive; each writes its own folder.
# The per-node spread across the three runs (column d_tilde_sigma_spread_um) is not filled by any script yet:
# compare the three nodes_<specimen>.csv by node_id.
for s in (0.080, 0.125):
    cfg_path = "%s/config_sigma%.3f.json" % (OUT, s)
    with open(cfg_path) as f:
        h = C.config_from_dict(json.load(f)).signature_hash("estimator")
    table = "%s/tables/bias_table_%s" % (OUT, h)
    if not os.path.exists(table + ".npz"):
        print("sigma_fit %.3f: bias_table_%s is not on Drive yet; skipped" % (s, h))
        continue
    run("scripts/run_cell.py", "--specimen", SPECIMEN, "--table", table, "--out-dir", "%s/cell_sigma%.3f" % (OUT, s),
        "--cache-dir", CACHE_DIR, "--config-json", cfg_path, "--registration-json", REG_JSON, *ALIGN)

# ====================================================================================================
# ===== CELL end of session (optional)
# ====================================================================================================
# WHAT TO SEND BACK
# - Cell 3b: the per-node verdict lines and `registration_summary_529878215.json`.
# - Cell 4b/4c: `pilot_summary_529878215.json` (with `k_star_vs_dip_depth`), the minutes per stretch, the
# table of nodes where the two focus rules differ, and two or three node figures (node 8441's among them).
# - Cell 4d: the `[planes]` lines, and the montages of two or three nodes: one where the two focus rules
# differ, one trunk node (large Allen radius).
# - Cell 4e: the `[planediff]` lines and the figures of nodes 2 and 3.
# - Cell 5: the `"renderer"` block without the tables, `table_sets`, `pillow_version`, `notes`.
# - Cell 6: the `pooled dendrite diameter` line.
# - Cell 7: the growth-fit report.
#
# These decide the open items of SPEC section 8: d_max, the mu range, the jitter amplitudes, the camera fields
# and the dark-flag threshold.
# At the end of a session: write everything to Drive and unmount (the next cell would need Cell 0 again)
drive.flush_and_unmount()
