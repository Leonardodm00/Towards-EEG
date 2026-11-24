# -*- coding: ascii -*-
# TEEG diameter pipeline -- the Colab run of the cleaning and the per-node fit on the pilot 529878215 (2026-11-14)
#
# Jupytext "percent" format: every "# %%" starts a cell. Paste the cells into a Colab notebook in order, or convert:
#   pip install jupytext && jupytext --to ipynb cleaning_pilot_2026-11-14.py
# Cells 0 and 1 are the setup cells of notebooks/s3_run_2026-10-27.py (Drive, clone, the run() helper); Cells 16, 17
# and 18 are new (SPEC Block 36; D-129, D-130, D-132, D-134, D-137, D-138, D-141). They need the outputs of Cell 7
# (OUT/calibration/calibration_<S>.json: the calibrated kernel). Each cell says what to send back.

# %% [markdown]
# ## Cells 0 and 1: Drive, the repository, the run() helper (as in the S3 notebook)

# %%
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


# %% [markdown]
# ## Cell 1: code (clone or update) and the `run()` helper
# Expect `[diameter] commit <hash> on sci/diameter-pipeline`; the hash should be the branch head on GitHub.

# %%
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

# %% [markdown]
# ## Cell 16: the configuration of the run and the contamination survey (no image is read; about 3 s per dendrite node
# on the pilot, so about 7 h for all 8496 nodes)
#
# `SURVEY_MAX_NODES`: survey a seeded random sample of this many nodes (None: all). The outputs are rewritten every 100
# nodes and when the cell is stopped, so a stopped survey still leaves a usable table and example file.
#
# Knobs: `B_GL`, `MU` (Cell 7 of 2026-11-10 printed 159.0 gl and 0.258 /um for the pilot); `KERNEL_SLOPE` (None: the
# far fit's slope if it is a production kernel, else the configured 0.79 beyond the table's last depth);
# `KERNEL_WIDTH` (D-221.1, D-222.3; SPEC Block 36 amendment 22): "fixed" keeps the kernel's in-focus width at the
# configured 0.080 um; "calibrated" re-anchors the kernel at Cell 23B's inverse-variance weighted means of sigma0 and of
# the growth scale c (`SIGMA_JSON`, Cell 23B's sigma_<S>.json). The switch reaches the fits of Cells 20, 25 and 26
# (`KW_ARGS`); the thick nodes' preparation of Cell 23 stays fixed, so that the calibration is not re-anchored on itself.
# Output to send back: the line `kernel used: ...`, the count per class, and the list of example nodes.

# %%
import json
B_GL, MU = 159.0, 0.258          # gl, 1/um: the pilot's medians (Cell 7, 2026-11-10)
KERNEL_SLOPE = None              # um of SD per um of defocus beyond the table; None = far fit if production, else configured
SURVEY_MAX_NODES = 600          # a random sample (about 30 min); None = all 8496 nodes (about 7 h)
KERNEL_WIDTH = "fixed"           # "fixed" (0.080 um in focus, D-172) or "calibrated" (Cell 23B's means, D-222.3)
SIGMA_JSON = "%s/cleaning_thick/sigma/sigma_%d.json" % (OUT, SPECIMEN)   # Cell 23B's output, read when "calibrated"
KW_ARGS = ["--kernel-width", KERNEL_WIDTH] + (["--sigma-json", SIGMA_JSON] if KERNEL_WIDTH == "calibrated" else [])
CLEAN = OUT + "/cleaning"
INFER = OUT + "/inference"
os.makedirs(CLEAN, exist_ok=True)
CALIB = "%s/calibration/calibration_%d.json" % (OUT, SPECIMEN)
FARFIT = "%s/calibration/far_fit_%d.json" % (OUT, SPECIMEN)
CFG_CAL = OUT + "/calibration/calibration_config.json"
CFG_CLEAN = CLEAN + "/cleaning_config.json"


def cleaning_config(calib_path, farfit_path, cal_knobs_path, slope=None):
    """The run's configuration as nested sections: the calibrated kernel of Cell 7 (its table; the continuation's
    slope from the far fit when that holds a production kernel, or `slope`), Cell 7's calibration knobs when the
    file exists. Returns (dict, a line saying which kernel is used)."""
    cfg, words = {}, []
    if os.path.exists(cal_knobs_path):
        with open(cal_knobs_path) as f:
            cfg.update(json.load(f))
    with open(calib_path) as f:
        cal = json.load(f)
    k = cal.get("kernel")
    if not k:
        raise RuntimeError("no kernel table in %s (%s): run Cell 7 first" % (calib_path, cal.get("kernel_error", "")))
    ren = {"kernel_table_delta_um": k["kernel_table_delta_um"], "kernel_table_sigma_um": k["kernel_table_sigma_um"],
           "sigma_r0_um": k["sigma_r0_um"]}
    words.append("table %s um at %s um (Cell 7)" % (["%.3f" % s for s in ren["kernel_table_sigma_um"]],
                                                    ["%.2f" % d for d in ren["kernel_table_delta_um"]]))
    far = None
    if os.path.exists(farfit_path):
        with open(farfit_path) as f:
            far = json.load(f).get("kernel")
    if slope is not None:
        ren["kernel_continuation_slope"] = float(slope)
        words.append("continuation slope %.4f (KERNEL_SLOPE)" % float(slope))
    elif far and "kernel_continuation_slope" in far:
        ren["kernel_continuation_slope"] = float(far["kernel_continuation_slope"])
        words.append("continuation slope %.4f (far fit, production)" % ren["kernel_continuation_slope"])
    else:
        words.append("continuation slope: the configured one (no production far fit)")
    cfg["renderer"] = dict(cfg.get("renderer", {}), **ren)
    return cfg, "kernel used: " + "; ".join(words)


cfg_d, line = cleaning_config(CALIB, FARFIT, CFG_CAL, KERNEL_SLOPE)
with open(CFG_CLEAN, "w") as f:
    json.dump(cfg_d, f, indent=1)
print("[cell 16] " + line)
CLEAN_CFG = ["--config-json", CFG_CLEAN]
EXAMPLES = "%s/examples_%d.json" % (CLEAN, SPECIMEN)
RESURVEY = False                # True: survey again although an examples file exists
if RESURVEY or not os.path.exists(EXAMPLES):
    run("scripts/clean_cell.py", "survey", "--specimen", SPECIMEN, "--out", CLEAN, "--cache-dir", CACHE_DIR, *ALIGN,
        *CLEAN_CFG, "--B", B_GL, "--mu", MU, *([] if SURVEY_MAX_NODES is None else ["--max-nodes", SURVEY_MAX_NODES]))
else:
    print("[cell 16] survey skipped: %s exists (set RESURVEY = True to survey again)" % EXAMPLES)
with open(EXAMPLES) as f:
    ex = json.load(f)
print("[cell 16] example nodes:", ex.get("example_ids"), "| classes:", ex.get("classes"))
print("[cell 16] surveyed %d of %d dendrite nodes (%s); counts:" % (ex["n_surveyed"], ex["n_dendrite_nodes"], ex["sample"]),
      ex["counts"])

# %% [markdown]
# ## Cell 17: the cleaning rounds on the example nodes and their main neighbours (downloads patches; of the order of
# 2 to 4 min per node and round on the synthetic cell, so 2 to 3 h at the first run's settings -- not measured on the pilot)
#
# Each round refits every node's background through the shadow model (D-144) before it cleans.
#
# Knobs: `ROUNDS` (at most 5, D-130.O6), `MAX_WORK` (how many nodes are refined), `WORK_PEAK_GL` (a neighbour joins
# the work list when its predicted shadow on an example's patch reaches this). The run can be stopped and started
# again: the image crops stay in the cache.
# Output to send back: each node's line (pixel SD x factor, the backgrounds, `kept the previous backgrounds` if any),
# the lines of each round (`max relative d change`, `max line change`), the stop reason, the
# dropped nodes with their reasons, `cleaning_rounds.png` and the `cleaning_node<id>.png` figures.

# %%
ROUNDS, MAX_WORK, WORK_PEAK_GL = 1, 5, 3.0      # a first test. More rounds help only when MAX_WORK exceeds the
# number of examples (then neighbours are refined too); every round is saved as it ends, so stopping keeps the last one
run("scripts/clean_cell.py", "clean", "--specimen", SPECIMEN, "--out", CLEAN, "--cache-dir", CACHE_DIR, *ALIGN,
    *CLEAN_CFG, "--examples", EXAMPLES, "--B", B_GL, "--mu", MU, "--rounds", ROUNDS, "--max-work", MAX_WORK,
    "--work-peak-gl", WORK_PEAK_GL)
import glob
from IPython.display import Image, display
for path in [CLEAN + "/cleaning_rounds.png"] + sorted(glob.glob(CLEAN + "/cleaning_node*.png")):
    if os.path.exists(path):
        print(path)
        display(Image(filename=path))

# %% [markdown]
# ## Cell 18: the per-node fit on the cleaned lines of the example nodes (no download; about 1 h)
#
# Reads Cell 17's folder only. A node with a close or overlapping neighbour is fitted alone and flagged
# `tier2_unmodelled` (the joint fit of D-132.2 is not built yet).
# Output to send back: the line per node, the table below, `diameters.png`, `chi2.png` and the `fit_node<id>.png` figures.

# %%
os.makedirs(INFER, exist_ok=True)
STARTS = "1.0"     # d0 x these: one start at the cleaning's refined values (3x faster); "0.6,1.0,1.6" for three
run("scripts/infer_nodes.py", "real", "--specimen", SPECIMEN, "--cleaning-dir", CLEAN, "--out", INFER, "--starts", STARTS)
import pandas as pd
res = pd.read_csv("%s/inference_%d.csv" % (INFER, SPECIMEN))
display(res)
for path in [INFER + "/diameters.png", INFER + "/chi2.png"] + sorted(glob.glob(INFER + "/fit_node*.png")):
    if os.path.exists(path):
        print(path)
        display(Image(filename=path))

# %% [markdown]
# ## Cell 16b: rank every dendrite node by its worst tier-1 neighbour (D-150.2; geometry only, about 10 min)
#
# Run Cells 0, 1 and 16 first. For each dendrite node: the neighbours the cleaning subtracts (tier 1, D-132), each
# one's predicted dip along the node's measuring line (the survey's predictor, Allen's radii, MU), and the largest
# structured part of one such dip (its maximum minus its minimum along the line: a shoulder or a second dip, not a
# uniform rise). The examples for Cell 17b are the highest-ranked nodes that are not near_end, whose tier-2
# neighbours (left in by design) put at most 1 gl on the line, 10 um apart. Outputs in CLEAN:
# `rank_tier1_<S>.csv` and `examples_tier1_<S>.json`. Output to send back: the log's last lines.

# %%
import json
import pandas as pd
from IPython.display import Image, display
RANK_CSV = "%s/rank_tier1_%d.csv" % (CLEAN, SPECIMEN)
EXAMPLES_T1 = "%s/examples_tier1_%d.json" % (CLEAN, SPECIMEN)
RERANK = False                                         # True: recompute even if the files exist
if RERANK or not (os.path.exists(RANK_CSV) and os.path.exists(EXAMPLES_T1)):
    run("scripts/clean_cell.py", "rank", "--specimen", SPECIMEN, "--out", CLEAN, "--cache-dir", CACHE_DIR, *ALIGN,
        *CLEAN_CFG, "--B", B_GL, "--mu", MU)
rank = pd.read_csv(RANK_CSV)
with open(EXAMPLES_T1) as f:
    ex_t1 = json.load(f)
cols = ["node_id", "worst1_range_gl", "worst1_peak_gl", "worst1_lateral_um", "worst1_defocus_um", "worst2_peak_gl",
        "n_tier2", "own_dip_gl", "near_end"]
print("examples (tier-1 ranking):", ex_t1["example_ids"])
display(rank.sort_values("worst1_range_gl", ascending=False)[cols].head(15))

# %% [markdown]
# ## Cell 17b: the cleaning in worse conditions -- the tier-1 ranking's examples, with their neighbours refined
# (D-149, D-150; about 2-3 h for 4 examples and 12 work-list nodes)
#
# Example nodes: Cell 16b's (`examples_tier1_<S>.json`), or your own list in WORST_IDS. The neighbours' nodes join
# the work list by their predicted shadow on an example's measuring line (D-149.2), at least WORK_PEAK_GL_W, the
# strongest first, up to MAX_WORK_W nodes in all. Round 1 cleans every node with round 0's features (the neighbours
# at Allen's diameter and mu MU) and refines them; round 2 cleans the examples with the refined neighbours. Each
# round is saved as it ends. The shadows are kept between rounds (D-150.3): a round renders only the neighbours
# that changed or that differ between the background's and the cleaning's sums, and the log says how many after
# each round (`shadow images ... reused, ... rendered`). Outputs in OUT/cleaning_tier1 (Cell 17's folder and
# OUT/cleaning_worst, the run of 2026-11-16 on 8089, 4844, 7081, 7448, are untouched): per example
# `cleaning_line_node<id>.png` (the line before, after round 1 and after round 2, and the light each round added
# back), `cleaning_node<id>_round1.png` and `cleaning_node<id>_round2.png` (before and after on one grey scale,
# the predicted shadow on its own, D-150.1). Output to send back: the log (the `work list:` lines above all) and
# the figures.
#
# Same-node clearance (D-156; SPEC Blocks 35 amendment 2, 37, 38): after each round the node's own shadow at the
# round's estimate is split by depth. `own_node<id>.png`: per plane the cleaned profile, the cleared profile (the
# out-of-band part of the own shadow added back), the in-band model and the whole model, and the fitted tube's
# chord as a bar; `own_layers_node<id>.png`: the shadow of each 0.1 um depth layer at the plane k*. It is a
# product: the features and the fits are the same with `--no-own`. Per node, `cleaning_<S>.json` carries `own`.

# %%
WORST_IDS = None                                       # None: Cell 16b's examples; or a list of node ids
ROUNDS_W, MAX_WORK_W, WORK_PEAK_GL_W = 2, 12, 1.0      # MAX_WORK_W counts the examples too
CLEAN_W = OUT + "/cleaning_tier1"                       # D-149's run stays in OUT/cleaning_worst
os.makedirs(CLEAN_W, exist_ok=True)
with open(EXAMPLES_T1) as f:
    exw = json.load(f)
if WORST_IDS is not None:
    exw["example_ids"] = list(WORST_IDS)
    exw["classes"] = {str(i): "chosen" for i in WORST_IDS}
EXAMPLES_W = "%s/examples_worst_%d.json" % (CLEAN_W, SPECIMEN)
with open(EXAMPLES_W, "w") as f:
    json.dump(exw, f, indent=1)
ids_w = [int(i) for i in exw["example_ids"]]
print("[cell 17b] examples:", ids_w)
run("scripts/clean_cell.py", "clean", "--specimen", SPECIMEN, "--out", CLEAN_W, "--cache-dir", CACHE_DIR, *ALIGN,
    *CLEAN_CFG, "--examples", EXAMPLES_W, "--B", B_GL, "--mu", MU, "--rounds", ROUNDS_W, "--max-work", MAX_WORK_W,
    "--work-peak-gl", WORK_PEAK_GL_W)
for nid in ids_w:
    for name in ("cleaning_line_node%d.png", "cleaning_node%d_round1.png", "cleaning_node%d_round2.png",
                 "own_node%d.png", "own_layers_node%d.png"):          # the last two: the same-node clearance (D-156)
        path = CLEAN_W + "/" + name % nid
        if os.path.exists(path):
            print(path)
            display(Image(filename=path))

# %% [markdown]
# ## Cell 18b (optional): the per-node fit on Cell 17b's cleaned lines (no download; about 1 h for the four examples)
#
# Outputs in OUT/inference_tier1 (OUT/inference_worst untouched). Output to send back: the line per node, the table, `chi2.png` and the `fit_node<id>.png` figures.
# After each fit the same-node clearance at the fitted parameters: `own_node<id>.png`, `own_layers_node<id>.png`, `own_<S>_node<id>.npz` (D-156).

# %%
INFER_W = OUT + "/inference_tier1"
os.makedirs(INFER_W, exist_ok=True)
run("scripts/infer_nodes.py", "real", "--specimen", SPECIMEN, "--cleaning-dir", CLEAN_W, "--out", INFER_W,
    "--starts", "1.0")
import glob
res_w = pd.read_csv("%s/inference_%d.csv" % (INFER_W, SPECIMEN))
display(res_w)
for path in ([INFER_W + "/diameters.png", INFER_W + "/chi2.png"] + sorted(glob.glob(INFER_W + "/fit_node*.png"))
             + sorted(glob.glob(INFER_W + "/own_node*.png")) + sorted(glob.glob(INFER_W + "/own_layers_node*.png"))):
    if os.path.exists(path):
        print(path)
        display(Image(filename=path))

# %% [markdown]
# ## Cell 18c: each example node at its focal plane, with the estimated diameter drawn on the image (no download; seconds)
#
# For every example node of the cleaning run (the nodes of `WORST_IDS` in Cell 17b; images are kept for the example
# nodes only): the node's patch at the first stage's sharpest plane k*, the two edges of the tube the joint fit
# estimated (solid: the per-node fit of Cell 18b when it has run, else the last round's refinement of Cell 17b) and
# Allen's diameter about the traced position (dashed). SPEC Block 31 amendment 3, Block 36 amendment 12. It reads only
# the files Cells 17b and 18b wrote. Output to send back: `focal_<S>.png` and the table.

# %%
FOCAL_NODES = None          # None: the run's example nodes; or a string of ids, e.g. "8089,7448,4844,7081"
FOCAL_PLANE = "kstar"       # "kstar": the first stage's sharpest plane; "axis": the recorded plane nearest the fitted axis
FOCAL_IMAGE = "before"      # "before": the patch as recorded; "after": after the neighbours' cleaning
FOCAL_ZOOM_UM = 6.0         # half-width of the view about the fitted axis (um)
FOCAL_DIR = next((d for d in (OUT + "/cleaning_tier1", OUT + "/cleaning_worst")
                  if os.path.exists("%s/cleaning_%d.json" % (d, SPECIMEN))), None)
assert FOCAL_DIR is not None, "no cleaning_%d.json under %s/cleaning_tier1 or cleaning_worst: run Cell 17b first" % (SPECIMEN, OUT)
FOCAL_INF = FOCAL_DIR.replace("/cleaning_", "/inference_")
focal_args = ["--cleaning-dir", FOCAL_DIR, "--specimen", SPECIMEN, "--plane", FOCAL_PLANE, "--image", FOCAL_IMAGE,
              "--zoom-um", FOCAL_ZOOM_UM]
if os.path.exists("%s/inference_%d.json" % (FOCAL_INF, SPECIMEN)):
    focal_args += ["--inference-dir", FOCAL_INF]         # Cell 18b has run: its per-node fits are drawn
if FOCAL_NODES:
    focal_args += ["--nodes", FOCAL_NODES]
run("scripts/show_focal.py", *focal_args)
display(Image(filename="%s/focal_%d.png" % (FOCAL_DIR, SPECIMEN)))
display(pd.read_csv("%s/focal_%d.csv" % (FOCAL_DIR, SPECIMEN)))

# %% [markdown]
# ## Cell 19: three clear nodes -- the one-tube model where no neighbour is on the line (D-160; about 1 h)
#
# Run Cells 0, 1, 16 and 16b first (16b's ranking is geometry only, about 10 min, and is reused if it exists).
# `clear` picks from the ranking the nodes that are not near a branch's end, have no tier-2 neighbour (none close or
# overlapping, D-132), lie at least MIN_SOMA_UM from the soma, and have the least predicted light of a tier-1
# neighbour on their line, 10 um apart. `clean` then runs on these nodes alone (two rounds) into OUT/cleaning_clear,
# and `show_focal.py` draws each at its focal plane with the fitted tube's edges (Cell 18c's figure). If the solid
# edges sit on the dendrite's edges here, the one-tube model holds where nothing else is on the line. Output to send
# back: the `clear:` lines of the log, `focal_<S>.png`, the table and the `own_node<id>.png` figures.

# %%
N_CLEAR, MIN_SOMA_UM = 3, 20.0
CLEAR_IDS = None                                       # None: the pick of `clear`; or your own list of node ids
CLEAN_C = OUT + "/cleaning_clear"
os.makedirs(CLEAN_C, exist_ok=True)
assert os.path.exists("%s/rank_tier1_%d.csv" % (CLEAN, SPECIMEN)), "run Cell 16b first (the tier-1 ranking)"
run("scripts/clean_cell.py", "clear", "--specimen", SPECIMEN, "--out", CLEAN, *CLEAN_CFG, "--n", N_CLEAR,
    "--min-soma-um", MIN_SOMA_UM)
with open("%s/examples_clear_%d.json" % (CLEAN, SPECIMEN)) as f:
    exc = json.load(f)
if CLEAR_IDS is not None:
    exc["example_ids"] = list(CLEAR_IDS)
    exc["classes"] = {str(i): "chosen" for i in CLEAR_IDS}
EXAMPLES_C = "%s/examples_clear_%d.json" % (CLEAN_C, SPECIMEN)
with open(EXAMPLES_C, "w") as f:
    json.dump(exc, f, indent=1)
ids_c = [int(i) for i in exc["example_ids"]]
print("[cell 19] clear nodes:", ids_c)
run("scripts/clean_cell.py", "clean", "--specimen", SPECIMEN, "--out", CLEAN_C, "--cache-dir", CACHE_DIR, *ALIGN,
    *CLEAN_CFG, "--examples", EXAMPLES_C, "--B", B_GL, "--mu", MU, "--rounds", 2, "--max-work", len(ids_c),
    "--work-peak-gl", 1.0)
run("scripts/show_focal.py", "--cleaning-dir", CLEAN_C, "--specimen", SPECIMEN, "--zoom-um", 6.0)
display(Image(filename="%s/focal_%d.png" % (CLEAN_C, SPECIMEN)))
display(pd.read_csv("%s/focal_%d.csv" % (CLEAN_C, SPECIMEN)))
with open("%s/cleaning_%d.json" % (CLEAN_C, SPECIMEN)) as f:
    rec_c = json.load(f)
for nid in ids_c:
    node_c = rec_c["nodes"].get(str(nid), {})
    print("node %d: tier-1 neighbours %s, tier-2 neighbours %s, refined d %s um" % (   # no line may start with a
        nid, node_c.get("n_tier1"), node_c.get("n_tier2"), node_c.get("d_um")))          # percent sign: Colab's .py export comments it out
    for name in ("own_node%d.png", "cleaning_line_node%d.png"):
        path = CLEAN_C + "/" + name % nid
        if os.path.exists(path):
            print(path)
            display(Image(filename=path))

# %% [markdown]
# ## Cell 19b: the clear nodes of Cell 19 at three in-focus widths of the kernel (D-165; about 5 min per width)
#
# The kernel's in-focus width sigma_r(0) is configured (0.080 um), never measured by the stacks, and a node's fitted
# diameter is confounded with it. `--sigma-r0-um` re-anchors the kernel table at another width, keeping its measured
# growth with defocus (SPEC Block 4 amendment 1, Block 36 amendment 14). Here the three nodes of Cell 19 are cleaned
# and refined once per width of SIGMA_R0_LIST, each run in its own folder OUT/cleaning_clear_s0_<nm>. The table
# gives, per node and width: the refined diameter, its ratio to Allen's, the coefficient, and the misfit (the RMS
# over the planes of the line's residual after the one-tube fit, gl) beside the node's noise SD. A width the data
# reject shows as a misfit clearly above the noise; widths the data cannot tell apart show the same misfit.
# Run Cells 0, 1, 16, 16b and 19 first. Output to send back: the table and the three `focal_<S>.png` figures.

# %%
import numpy as np
SIGMA_R0_LIST = [0.080, 0.18, 0.28]                    # um: the configured width and two larger ones
RERUN_S0 = False                                       # True: run a width again although its folder holds a result


def clean_at_widths(examples_path, tag, widths, rounds=2):
    """The nodes of examples_path cleaned once per in-focus width, each run in OUT/cleaning_<tag>_s0_<nm>.
    Returns a table with one row per node and width."""
    with open(examples_path) as f:
        ex_w = json.load(f)
    ids = [int(i) for i in ex_w["example_ids"]]
    rows = []
    for s0 in widths:
        out_w = "%s/cleaning_%s_s0_%03d" % (OUT, tag, int(round(1000 * s0)))
        os.makedirs(out_w, exist_ok=True)
        rec_path = "%s/cleaning_%d.json" % (out_w, SPECIMEN)
        if RERUN_S0 or not os.path.exists(rec_path):
            run("scripts/clean_cell.py", "clean", "--specimen", SPECIMEN, "--out", out_w, "--cache-dir", CACHE_DIR,
                *ALIGN, *CLEAN_CFG, "--examples", examples_path, "--B", B_GL, "--mu", MU, "--rounds", rounds,
                "--max-work", len(ids), "--work-peak-gl", 1.0, "--sigma-r0-um", s0)
            run("scripts/show_focal.py", "--cleaning-dir", out_w, "--specimen", SPECIMEN, "--zoom-um", 6.0)
        with open(rec_path) as f:
            rec_w = json.load(f)
        print("in-focus width %.3f um -> %s" % (s0, out_w))
        focal_w = "%s/focal_%d.png" % (out_w, SPECIMEN)
        if os.path.exists(focal_w):
            display(Image(filename=focal_w))
        for nid in ids:
            node_w = rec_w["nodes"].get(str(nid))
            if node_w is None:
                continue
            rms = (node_w.get("own") or {}).get("residual_rms_gl")
            misfit = float(np.sqrt(np.mean(np.square(rms)))) if rms else float("nan")
            rows.append(dict(node_id=nid, sigma_r0_um=s0, d_allen_um=node_w.get("d_allen_um"), d_um=node_w.get("d_um"),
                             mu_per_um=node_w.get("mu_per_um"), misfit_gl=misfit, noise_gl=node_w.get("sigma_gl"),
                             status=node_w.get("status")))
    tab = pd.DataFrame(rows)
    tab["d_over_allen"] = tab["d_um"] / tab["d_allen_um"]
    return tab


def show_widths(tab):
    """The table of clean_at_widths, one line per node and one column per width, the nodes by Allen's diameter."""
    wide = tab.pivot(index="node_id", columns="sigma_r0_um", values=["d_um", "d_over_allen", "mu_per_um", "misfit_gl"])
    wide.columns = ["%s at %.3f" % (name, s0) for name, s0 in wide.columns]      # one level: quantity at width (um)
    first = tab.drop_duplicates("node_id").set_index("node_id")[["d_allen_um", "noise_gl"]]
    display(first.join(wide.round(3)).sort_values("d_allen_um"))


EXAMPLES_C = "%s/cleaning_clear/examples_clear_%d.json" % (OUT, SPECIMEN)
assert os.path.exists(EXAMPLES_C), "run Cell 19 first (its three clear nodes)"
tab_c = clean_at_widths(EXAMPLES_C, "clear", SIGMA_R0_LIST)
show_widths(tab_c)

# %% [markdown]
# ## Cell 19c: clear nodes of different sizes, at the same three widths (D-165; about 10 min per width)
#
# `clear --d-bins` picks N_PER_BIN clear nodes per bin of Allen's diameter (twice the traced radius: the only size
# known before the measurement; SPEC Block 36 amendment 15). On the pilot Allen's diameters run from 0.43 to 2.2 um
# with 792, 6703, 724, 217 and 57 dendrite nodes in the bins below. The nodes are then cleaned and refined at each
# width of SIGMA_R0_LIST into OUT/cleaning_sizes_s0_<nm> (Cell 19b's function: run that cell first). What to read:
# if the in-focus blur is wider than configured, the ratio refined / Allen falls toward 1 as the dendrite gets
# thicker (on synthetic tubes under a true width of 0.22 um: x 1.6 at 0.5 um, x 1.3 at 0.8 um, x 1.1 at 1.2 um);
# a ratio that stays the same at every size points elsewhere. Thick nodes also separate blur from diameter best,
# so their misfit is the one to compare across the widths.
# Output to send back: the `clear:` lines of the log, the table, the figure and the `focal_<S>.png` figures.

# %%
import matplotlib.pyplot as plt
D_BINS = "0.40,0.45,0.60,0.80,1.20,5.0"                # um, edges of the bins of Allen's diameter
N_PER_BIN, MIN_SOMA_Z_UM = 1, 20.0
CLEAN_Z = OUT + "/cleaning_sizes"
os.makedirs(CLEAN_Z, exist_ok=True)
run("scripts/clean_cell.py", "clear", "--specimen", SPECIMEN, "--out", CLEAN, "--cache-dir", CACHE_DIR, *ALIGN,
    *CLEAN_CFG, "--n", N_PER_BIN, "--min-soma-um", MIN_SOMA_Z_UM, "--d-bins", D_BINS)
with open("%s/examples_clear_%d.json" % (CLEAN, SPECIMEN)) as f:
    exz = json.load(f)
EXAMPLES_Z = "%s/examples_sizes_%d.json" % (CLEAN_Z, SPECIMEN)
with open(EXAMPLES_Z, "w") as f:
    json.dump(exz, f, indent=1)
print("[cell 19c] nodes by size:", {i: round(s["d_allen_um"], 3) for i, s in exz["sizes"].items()})
tab_z = clean_at_widths(EXAMPLES_Z, "sizes", SIGMA_R0_LIST)
show_widths(tab_z)
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for s0, g in tab_z.groupby("sigma_r0_um"):
    g = g.sort_values("d_allen_um")
    ax.plot(g["d_allen_um"], g["d_over_allen"], "o-", label="in-focus width %.3f um" % s0)
ax.axhline(1.0, color="0.5", lw=0.8)
ax.set_xlabel("Allen's diameter (um)")
ax.set_ylabel("refined diameter / Allen's")
ax.legend()
fig.savefig("%s/ratio_by_size_%d.png" % (CLEAN_Z, SPECIMEN), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Cell 20: the clear-node fit -- no cleaning, one pass, the first stage held at Allen's diameter, centre weights, bump cut (D-169, D-171 to D-173)
#
# Run after Cells 0, 1 and 16 (Cell 16 skips its survey when its examples file exists). The nodes are Cell 19c's
# (clear nodes by size; its examples file is reused when it exists, else they are picked here with the same rule,
# which needs Cell 16b's ranking in OUT/cleaning). Two steps (SPEC Block 36 amendments 16 and 17):
#   1. `clean --no-clean`: per node the first stage with the diameter held at Allen's (absorption, centre, depth,
#      sharpest plane, direction and background from it), the patch, the line as recorded. No neighbour is cleaned,
#      no round is run. Downloads the patches when they are not cached; about a minute per node.
#   2. `infer_nodes.py real`: the multi-plane one-tube fit started at Allen's diameter, its residuals weighted by
#      exp(-|v - v_c| / lambda) about the fitted centre, lambda = WEIGHT_KAPPA x FWHM expected from Allen's diameter
#      (FWHM^2 = (0.866 D_Allen)^2 + (2.355 sigma_r(0))^2, sigma_r(0) = 0.080 um), re-centred while fitting; a second
#      dip beside the central one that rises at least BUMP_SD noise SDs gives no weight to the samples from it
#      outwards. Then the node's own expected shadow (same-node clearance) at the estimate.
# Knobs: WEIGHT_KAPPA, BUMP_SD (0 = no cut), D169_WEIGHTS ("none" = unweighted), D169_STARTS; COV_FINE_FACTOR (D-224.2;
# SPEC Block 36 amendment 23): the error bars from the model at line_fine_factor 32 (p/32, where they converge) -- about
# nine renders 20 times the fit's cost each, so it adds roughly the fit's own time again per node; 0 = the fit's own
# Jacobian, as before (faster, its SE(d) off by tens of percent on thin nodes). Cell 16's KERNEL_WIDTH applies.
# Time (measured on the pilot, 2026-11-19, Colab): preparation 7.7 min; the fit 305 min for the five nodes, from
# 6 min (5353, fitted 0.77 um) to 111 min (639, fitted 2.82 um) -- it grows with the diameter the fit reaches.
# [corrected 2026-11-19: the earlier estimate here, "several minutes to a quarter of an hour per node", was wrong]
# The fit figures are written to OUT/inference_d169/fit_node<id>.png by the fit itself: to look at them again,
# run only the last loop of this cell, not the cell.
# Output to send back: the lines `node <id>: weights ...` and `node <id>: d ...` of the log, the table, the
# `focal_<S>.png` figure and the `fit_node<id>.png` figures (orange line: the weights; grey: samples with no weight).

# %%
import glob, json, os, shutil       # the cell's own imports: it can follow Cell 16 directly
import pandas as pd
from IPython.display import Image, display
WEIGHT_KAPPA = 1.25                    # lambda in units of the FWHM from Allen's diameter (D-171.1; slightly > 1)
BUMP_SD = 2.0                          # a near bump rises at least this many noise SDs (D-171.3); 0 = no cut
D169_WEIGHTS = "allen"                 # "allen": the centre weights; "none": unweighted (the bump cut still applies)
D169_STARTS = "1.0,1.6"                # the fit's starts, x Allen's diameter; "1.0" alone drops one start (time saved not measured)
COV_FINE_FACTOR = 32                   # the error bars' lattice (D-224.2); 0 = the fit's own Jacobian (faster)
D169_BINS, D169_N_PER_BIN, D169_MIN_SOMA_UM = "0.40,0.45,0.60,0.80,1.20,5.0", 1, 20.0
RERUN_D169_PREP = False                # True: prepare the nodes again although the folder holds them
CLEAN_D, INFER_D = OUT + "/cleaning_d169", OUT + "/inference_d169"
os.makedirs(CLEAN_D, exist_ok=True)
os.makedirs(INFER_D, exist_ok=True)
EXAMPLES_D = "%s/examples_sizes_%d.json" % (CLEAN_D, SPECIMEN)
EX_19C = "%s/cleaning_sizes/examples_sizes_%d.json" % (OUT, SPECIMEN)
if not os.path.exists(EXAMPLES_D):
    if os.path.exists(EX_19C):                         # the clear traces are Cell 19c's nodes (D-171.5)
        shutil.copyfile(EX_19C, EXAMPLES_D)
    else:
        run("scripts/clean_cell.py", "clear", "--specimen", SPECIMEN, "--out", CLEAN, "--cache-dir", CACHE_DIR, *ALIGN,
            *CLEAN_CFG, "--n", D169_N_PER_BIN, "--min-soma-um", D169_MIN_SOMA_UM, "--d-bins", D169_BINS)
        shutil.copyfile("%s/examples_clear_%d.json" % (CLEAN, SPECIMEN), EXAMPLES_D)
with open(EXAMPLES_D) as f:
    print("[cell 20] nodes:", json.load(f)["example_ids"])
if RERUN_D169_PREP or not os.path.exists("%s/cleaning_%d.json" % (CLEAN_D, SPECIMEN)):
    run("scripts/clean_cell.py", "clean", "--specimen", SPECIMEN, "--out", CLEAN_D, "--cache-dir", CACHE_DIR, *ALIGN,
        *CLEAN_CFG, "--examples", EXAMPLES_D, "--B", B_GL, "--mu", MU, "--no-clean")
run("scripts/infer_nodes.py", "real", "--specimen", SPECIMEN, "--cleaning-dir", CLEAN_D, "--out", INFER_D,
    "--weights", D169_WEIGHTS, "--bump-sd", BUMP_SD, "--weight-kappa", WEIGHT_KAPPA, "--starts", D169_STARTS,
    "--cov-fine-factor", COV_FINE_FACTOR, *KW_ARGS)
run("scripts/show_focal.py", "--cleaning-dir", CLEAN_D, "--specimen", SPECIMEN, "--inference-dir", INFER_D,
    "--zoom-um", 6.0)
res_d = pd.read_csv("%s/inference_%d.csv" % (INFER_D, SPECIMEN))
res_d["d_over_allen"] = res_d["d_hat_um"] / res_d["d_allen_um"]
display(res_d[["node_id", "d_allen_um", "d_hat_um", "se_d_um", "d_over_allen", "mu_hat_per_um", "v0_hat_um",
               "z_ax_hat_um", "chi2_mean", "n_planes_used", "lambda_um", "v_cut_lo_um", "v_cut_hi_um", "n_solves",
               "status", "at_bound", "flags"]].round(4))
display(Image(filename="%s/focal_%d.png" % (CLEAN_D, SPECIMEN)))
for path in sorted(glob.glob(INFER_D + "/fit_node*.png")):
    print(path)
    display(Image(filename=path))

# %% [markdown]
# ## Cell 21: a gallery of the nodes the image judges clean -- look, then pick four (D-194; about 1.5 min per candidate)
#
# Run after Cells 0, 1 and 16; needs Cell 16b's ranking in OUT/cleaning (as Cells 19 to 20). Three steps (SPEC
# Blocks 40, 41 and Block 36 amendment 18):
#   1. `clear --d-bins`: GAL_N_PER_BIN candidates per bin of Allen's diameter by the trace-based rule (no traced
#      neighbour on the line). That rule cannot see what is not traced: on the pilot it admitted 1013, 2669, 2620, 639.
#   2. `clean --no-clean`: per candidate the first stage (diameter held at Allen's), the patch, the nine planes, the
#      line as recorded. Nothing is cleaned, nothing is fitted. Downloads the patches (5 nodes took 7.7 min on
#      2026-11-19, so about an hour for 40: an extrapolation, not measured). Kept in OUT/cleaning_gallery and reused.
#   3. `show_clear_gallery.py`: the screen on the recorded image only (no model, no diameter). A node is judged
#      clean when its dip on the measuring line, in the planes k* - 1 .. k* + 1, is one symmetric dip that returns
#      to the background (D-198, 2026-11-20). Nine parallel lines up to 2 um along the dendrite are measured and
#      drawn but not judged: on the first run they rejected all 33 nodes, because a dendrite tilted in depth is in
#      focus over a short stretch only (GAL_JUDGE_ALONG = True judges them again). Seconds; rerun it alone with
#      other thresholds.
# Per shown node one page: the patch at k* with the measuring line (dotted), the lines along the dendrite and
# Allen's diameter (dashed); the profiles on those lines; the measuring line on the nine planes overlaid; the
# screen's numbers against their thresholds; then the patch at each of the nine planes on one grey scale, and the
# measuring line's profile at each plane on one axis (blue bar: its half-depth width; light-blue bar: Allen's).
# Shown: the nodes judged clean, the thinnest bin first, at most GAL_MAX_PER_BIN per bin; when fewer than
# GAL_FILL_TO are shown, the best-scoring REJECTED nodes follow, titled "NOT judged clean" with the reason (the
# thresholds are provisional: the rejected ones show which threshold to move).
# Knobs: GAL_BINS, GAL_N_PER_BIN, GAL_MAX_PER_BIN, GAL_FILL_TO, GAL_SHOW_IDS (your own list of node ids instead
# of the rule), GAL_ZOOM_UM, GAL_THRESHOLDS (any of outside-max 0.25, skew-max 0.15, width-range-max 0.25,
# depth-range-max 0.30, centre-range-max 0.50, min-depth-sd 10, along-half-um 2.0, along-step-um 0.5,
# focus-half-planes 1), GAL_JUDGE_ALONG, RERUN_GAL_PREP (True after changing GAL_BINS or GAL_N_PER_BIN).
# Output to send back: the four node ids you choose, the printed table, and `gallery_sheet_<S>.png`.

# %%
import json, os, shutil             # the cell's own imports: it can follow Cell 16 directly
import pandas as pd
from IPython.display import Image, display
GAL_BINS = "0.40,0.45,0.60,0.80,1.20,5.0"     # um, edges of the bins of Allen's diameter (Cell 19c's)
GAL_N_PER_BIN, GAL_MIN_SOMA_UM = 8, 20.0      # candidates per bin by the trace-based rule; their least distance from the soma
GAL_MAX_PER_BIN = 4                           # judged-clean nodes shown per bin at most (0: all)
GAL_FILL_TO = 8                               # fewer shown than this: the best-scoring rejected nodes follow, labelled (0: never)
GAL_SHOW_IDS = None                           # None: the rule above; or a list of node ids to show whatever their verdict
GAL_ZOOM_UM = 5.0                             # the image panels show +- this about the line (um)
GAL_THRESHOLDS = {}                           # e.g. {"outside-max": 0.30, "width-range-max": 0.35}; {} = the defaults
GAL_JUDGE_ALONG = False                       # True: the lines along the dendrite enter the verdict too (D-194's rule)
RERUN_GAL_PREP = False                        # True: pick and prepare the candidates again
CLEAN_G = OUT + "/cleaning_gallery"
GAL_OUT = CLEAN_G + "/gallery"
os.makedirs(CLEAN_G, exist_ok=True)
EXAMPLES_G = "%s/examples_gallery_%d.json" % (CLEAN_G, SPECIMEN)
if RERUN_GAL_PREP or not os.path.exists(EXAMPLES_G):
    assert os.path.exists("%s/rank_tier1_%d.csv" % (CLEAN, SPECIMEN)), "run Cell 16b first (the tier-1 ranking)"
    run("scripts/clean_cell.py", "clear", "--specimen", SPECIMEN, "--out", CLEAN, "--cache-dir", CACHE_DIR, *ALIGN,
        *CLEAN_CFG, "--n", GAL_N_PER_BIN, "--min-soma-um", GAL_MIN_SOMA_UM, "--d-bins", GAL_BINS)
    shutil.copyfile("%s/examples_clear_%d.json" % (CLEAN, SPECIMEN), EXAMPLES_G)
with open(EXAMPLES_G) as f:
    ids_g = [int(i) for i in json.load(f)["example_ids"]]
print("[cell 21] %d candidates:" % len(ids_g), ids_g)
if RERUN_GAL_PREP or not os.path.exists("%s/cleaning_%d.json" % (CLEAN_G, SPECIMEN)):
    run("scripts/clean_cell.py", "clean", "--specimen", SPECIMEN, "--out", CLEAN_G, "--cache-dir", CACHE_DIR, *ALIGN,
        *CLEAN_CFG, "--examples", EXAMPLES_G, "--B", B_GL, "--mu", MU, "--no-clean")
if os.path.isdir(GAL_OUT):                    # the figures of an earlier screen do not mix with this one's
    shutil.rmtree(GAL_OUT)
gal_args = ["--cleaning-dir", CLEAN_G, "--specimen", SPECIMEN, "--out", GAL_OUT, "--d-bins", GAL_BINS,
            "--max-per-bin", GAL_MAX_PER_BIN, "--fill-to", GAL_FILL_TO, "--zoom-um", GAL_ZOOM_UM]
for name, value in GAL_THRESHOLDS.items():
    gal_args += ["--" + name, value]
if GAL_JUDGE_ALONG:
    gal_args += ["--judge-along"]
if GAL_SHOW_IDS is not None:
    gal_args += ["--nodes", ",".join(str(int(i)) for i in GAL_SHOW_IDS)]
run("scripts/show_clear_gallery.py", *gal_args)
tab_g = pd.read_csv("%s/gallery_%d.csv" % (GAL_OUT, SPECIMEN))
display(tab_g[["node_id", "d_allen_um", "clean", "score", "reasons", "k_star", "depth_gl", "depth_sd", "fwhm_kstar_um",
               "fwhm_over_allen", "outside_focus", "skew_focus", "outside_along", "skew_along", "width_range",
               "depth_range", "centre_range", "shown"]].round(3))
sheet_g = "%s/gallery_sheet_%d.png" % (GAL_OUT, SPECIMEN)
if os.path.exists(sheet_g):
    display(Image(filename=sheet_g))
for name in tab_g.loc[tab_g["shown"], "figure"]:           # in the order shown: judged clean by bin, then rejected
    print(GAL_OUT + "/" + name)
    display(Image(filename=GAL_OUT + "/" + name))
print("[cell 21] judged clean:", tab_g.loc[tab_g["clean"], "node_id"].tolist())

# %% [markdown]
# ## Cell 22: dropped (D-202)
#
# The clear-node fit on D-199's four nodes (3883, 28, 6049, 2657) with WEIGHT_KAPPA 0.1 was dropped by the user on
# 2026-11-20, 17:44, before it ran (D-202). Its code is in the history at commit 481abe4.

# %% [markdown]
# ## Cell 23 (A): the scan of the thick nodes for the in-focus width -- images at k* and +-2 planes (D-201)
#
# Run after Cells 0, 1 and 16; needs Cell 16b's ranking in OUT/cleaning. The in-focus width sigma_r(0) is to be
# fitted on thick nodes, where it separates from the diameter (D-201; SPEC Blocks 42, 43); this part chooses and
# prepares them and shows them, with the scripts that exist (the fit is part B, built next):
#   1. `clear --d-bins 1.0,50`: up to THICK_N nodes with Allen's diameter >= 1.0 um, at least THICK_MIN_SOMA_UM from
#      the soma, no tier-2 neighbour, at least 10 um apart (3-D), the least traced light on the line first. The
#      ranking may hold only a sample of the cell (Cell 16's SURVEY_MAX_NODES): fewer than THICK_N can come out.
#   2. `clean --no-clean` into OUT/cleaning_thick with the planes k* - 2 .. k* + 2 only (cleaning.plane_half_range 2)
#      and the measuring line +-4.5 um (measure.profile_half_um; D-201.6), so that a thick tube's defocused dip
#      still reaches the background on both sides. Its time per node is not measured on real data (as Cell 21's
#      preparation: the node's image block and its neighbours' shadows at 5 planes).
#   3. The gallery of Cell 21 on all of them, judged or not (pages: the patch at each of the 5 planes, the line's
#      profile at each plane). Seconds.
# Output to send back: the printed table and `gallery_sheet_<S>.png`; say which nodes look clean.

# %%
import json, os, shutil       # the cell's own imports: it can follow Cell 16 directly
import pandas as pd
from IPython.display import Image, display
THICK_BINS = "1.0,50"         # um: one bin, Allen's diameter >= 1.0 um (D-201.4)
THICK_N = 20                  # nodes at most (D-201.4)
THICK_MIN_SOMA_UM = 5.0       # um: least distance from the soma in the image plane (D-201.4)
THICK_HALF_PLANES = 2         # the planes k* - 2 .. k* + 2 (D-201 [user])
THICK_LINE_HALF_UM = 4.5      # um: the measuring line's half-length (D-201.6; Cell 16's runs use 3.0)
RERUN_THICK_PREP = False      # True: choose and prepare the nodes again
CLEAN_T = OUT + "/cleaning_thick"
GAL_T = CLEAN_T + "/gallery"
os.makedirs(CLEAN_T, exist_ok=True)
with open(CFG_CLEAN) as f:                   # the run's configuration (Cell 16), with the two changes above
    cfg_t = json.load(f)
cfg_t.setdefault("cleaning", {})["plane_half_range"] = THICK_HALF_PLANES
cfg_t.setdefault("measure", {})["profile_half_um"] = THICK_LINE_HALF_UM
CFG_THICK = CLEAN_T + "/thick_config.json"
with open(CFG_THICK, "w") as f:
    json.dump(cfg_t, f, indent=1)
THICK_CFG = ["--config-json", CFG_THICK]
EXAMPLES_T = "%s/examples_thick_%d.json" % (CLEAN_T, SPECIMEN)
if RERUN_THICK_PREP or not os.path.exists(EXAMPLES_T):
    assert os.path.exists("%s/rank_tier1_%d.csv" % (CLEAN, SPECIMEN)), "run Cell 16b first (the tier-1 ranking)"
    run("scripts/clean_cell.py", "clear", "--specimen", SPECIMEN, "--out", CLEAN, "--cache-dir", CACHE_DIR, *ALIGN,
        *CLEAN_CFG, "--n", THICK_N, "--min-soma-um", THICK_MIN_SOMA_UM, "--d-bins", THICK_BINS)
    shutil.copyfile("%s/examples_clear_%d.json" % (CLEAN, SPECIMEN), EXAMPLES_T)
with open(EXAMPLES_T) as f:
    ids_t = [int(i) for i in json.load(f)["example_ids"]]
print("[cell 23A] %d thick nodes:" % len(ids_t), ids_t)
if RERUN_THICK_PREP or not os.path.exists("%s/cleaning_%d.json" % (CLEAN_T, SPECIMEN)):
    run("scripts/clean_cell.py", "clean", "--specimen", SPECIMEN, "--out", CLEAN_T, "--cache-dir", CACHE_DIR, *ALIGN,
        *THICK_CFG, "--examples", EXAMPLES_T, "--B", B_GL, "--mu", MU, "--no-clean")
if os.path.isdir(GAL_T):
    shutil.rmtree(GAL_T)
run("scripts/show_clear_gallery.py", "--cleaning-dir", CLEAN_T, "--specimen", SPECIMEN, "--out", GAL_T,
    "--d-bins", THICK_BINS, "--max-per-bin", 0, "--fill-to", THICK_N, "--zoom-um", 6.0)
tab_t = pd.read_csv("%s/gallery_%d.csv" % (GAL_T, SPECIMEN))
display(tab_t[["node_id", "d_allen_um", "clean", "score", "reasons", "k_star", "depth_gl", "depth_sd", "fwhm_kstar_um",
               "fwhm_over_allen", "outside_focus", "skew_focus", "shown"]].round(3))
display(Image(filename="%s/gallery_sheet_%d.png" % (GAL_T, SPECIMEN)))
for name in tab_t.loc[tab_t["shown"], "figure"]:
    print(GAL_T + "/" + name)
    display(Image(filename=GAL_T + "/" + name))

# %% [markdown]
# ## Cell 23 (B): the in-focus width fitted on the thick nodes (D-201; SPEC Blocks 42, 43; Block 36 amendment 19)
#
# Run after Cell 23A (its preparation in OUT/cleaning_thick and its gallery); in a new runtime, after Cells 0, 1 and
# 16 first (Cell 1 pulls the branch: this cell needs Blocks 42 and 43). For each node `scripts/calibrate_sigma.py`
# fits the planes k* - 2 .. k* + 2 with the pipeline's hybrid line renderer as it is, its kernel re-anchored at a
# fitted in-focus width sigma0 with a fitted growth scale c, and a straight background per plane solved exactly,
# from one start. The nodes: those the gallery of Cell 23A judged clean (D-198's verdict), or your own list in
# SIGMA_NODES. Time per node [measured in the sandbox on synthetic nodes]: 26 s at d 1.0 um, 59 s at 1.5 um, 208 s
# at 2.5 um, the nodes in parallel on the runtime's CPUs. On the pilot in Colab node 7 (fitted 2.87 um) took 17.7 min,
# 68 evaluations of about 15 s [the user's run, 2026-11-23]: Colab runs these renders 2-3 times slower than the sandbox.
# Its sigma_<S>.json is what Cell 16's KERNEL_WIDTH = "calibrated" reads (D-222.3; SPEC Block 36 amendment 22): the
# inverse-variance weighted means of sigma0 and of the growth scale c, printed last. A file written before wave 7 has
# no summary of c and the switch refuses it: run this cell again with the current branch.
# Output to send back: the printed table and the two summaries, sigma_summary_<S>.png and the node pages
# (OUT/cleaning_thick/sigma).

# %%
import os, json               # the cell's own imports: it can follow Cell 23A or Cell 16 directly
import pandas as pd
from IPython.display import Image, display
SIGMA_NODES = None            # None: the nodes the gallery of Cell 23A judged clean; or a list of node ids
SIGMA_WORKERS = 0             # 0: all CPUs of the runtime
CLEAN_T = OUT + "/cleaning_thick"
GAL_T = CLEAN_T + "/gallery"
SIG_T = CLEAN_T + "/sigma"
if SIGMA_NODES is None:
    gal = pd.read_csv("%s/gallery_%d.csv" % (GAL_T, SPECIMEN))
    is_clean = gal["clean"].map(lambda x: str(x).strip().lower() == "true")
    ids_s = [int(i) for i in gal.loc[is_clean, "node_id"]]
else:
    ids_s = [int(i) for i in SIGMA_NODES]
assert ids_s, "no node to fit: the gallery judged none clean -- give SIGMA_NODES"
print("[cell 23B] fitting %d nodes:" % len(ids_s), ids_s)
run("scripts/calibrate_sigma.py", "--cleaning-dir", CLEAN_T, "--specimen", SPECIMEN, "--out", SIG_T,
    "--nodes", ",".join(str(i) for i in ids_s), "--workers", SIGMA_WORKERS)
tab_s = pd.read_csv("%s/sigma_%d.csv" % (SIG_T, SPECIMEN))
display(tab_s[["node_id", "d_allen_um", "status", "ok", "d_um", "se_d_um", "sigma0_um", "se_sigma0_um", "growth_scale",
               "se_growth_scale", "chi2_per_dof", "n_eval", "seconds", "at_bound"]].round(4))
with open("%s/sigma_%d.json" % (SIG_T, SPECIMEN)) as f:
    sig = json.load(f)
s0_sum, c_sum = sig["summary"], sig.get("summary_growth_scale") or {}
print("summary of sigma0 (um):", s0_sum)
print("summary of the growth scale c:", c_sum or "absent (calibrate_sigma.py from before wave 7)")
f4 = lambda x: "%.4f" % x if isinstance(x, (int, float)) else str(x)   # None (n = 0) printed as it is
print("[cell 23B] for Cell 16's KERNEL_WIDTH = 'calibrated': sigma0 %s +- %s um (n %s), c %s +- %s (n %s)"
      % (f4(s0_sum.get("ivw_mean_um")), f4(s0_sum.get("ivw_se_um")), s0_sum.get("n"), f4(c_sum.get("ivw_mean")),
         f4(c_sum.get("ivw_se")), c_sum.get("n")))
display(Image(filename="%s/sigma_summary_%d.png" % (SIG_T, SPECIMEN)))
for nid in ids_s:
    page = "%s/sigma_node%d.png" % (SIG_T, nid)
    if os.path.exists(page):
        print(page)
        display(Image(filename=page))

# %% [markdown]
# ## Cell 24: the soma, plane by plane; its plane chosen by eye; Allen's trace over it (D-210, D-211, D-213; Block 44)
#
# Run after Cells 0, 1 and 16 (Cell 1 pulls the branch). The traced soma (the SWC's type-1 node, placed in the image
# frame with ALIGN's transform) is cropped at +-SOMA_HALF_UM (40 um by default; it was 20) from every plane within its
# traced radius + 6 um of its depth (on the pilot k = 75..159; the traced depth is k = 117). Allen's reconstruction is
# drawn faintly over every image (D-213): each traced dendrite segment at its traced diameter (blue outline; the axon
# dashed), filled where the trace places it at that plane's depth; the soma is its traced circle (dashed yellow). Nothing is picked automatically (D-211): the first run's
# automatic pick, k = 93, was a basal dendrite in focus inside the soma's disk. The montage shows every second plane
# of the whole window; once you have looked at it, set SOMA_K (the plane where you judge the soma in focus) and
# SOMA_K_RANGE (the first and last plane where you see it) and run the cell again: the focus figure then shows your
# plane, and both are recorded in soma_<S>.json. MONTAGE_K = (first, last, step) narrows the montage, e.g.
# (105, 145, 1) for every plane from 105 to 145. The right panel of the focus figure, the fine-detail energy inside
# the traced disk, counts every structure (dendrites included): a diagnostic, not a focus estimate. Planes already
# fetched come from CACHE_DIR. Output to send back: soma_<S>_focus.png, soma_<S>_planes.png and soma_<S>.json.

# %%
from IPython.display import Image, display   # the cell's own imports
SOMA_K = None         # the plane where you judge the soma in focus, e.g. 115; None: the traced depth's plane is shown
SOMA_K_RANGE = None   # (first, last) plane where you see the soma, e.g. (107, 140); None: not set
MONTAGE_K = None      # (first, last, step), e.g. (105, 145, 1); None: the whole window, every second plane
SOMA_HALF_UM = None   # the crop's half-size in um, e.g. 60; None: 40 (the configuration's soma_view.half_um)
SOMA_OUT = OUT + "/soma"
choice = (([] if SOMA_K is None else ["--soma-k", SOMA_K])
          + ([] if SOMA_K_RANGE is None else ["--soma-k-range", *SOMA_K_RANGE])
          + ([] if MONTAGE_K is None else ["--montage-k", *MONTAGE_K])
          + ([] if SOMA_HALF_UM is None else ["--half-um", SOMA_HALF_UM]))
run("scripts/show_soma.py", "--specimen", SPECIMEN, "--cache-dir", CACHE_DIR, *ALIGN, *CLEAN_CFG, "--out", SOMA_OUT,
    *choice)
for name in ("soma_%d_focus.png" % SPECIMEN, "soma_%d_planes.png" % SPECIMEN):
    print(SOMA_OUT + "/" + name)
    display(Image(filename=SOMA_OUT + "/" + name))



# %% [markdown]
# ## Cell 24 (b): the soma's planes one at a time -- a slider, nothing drawn over them; the plane you pick becomes SOMA_K (D-211, D-239; Block 44 amendment 4)
#
# Run after Cells 0, 1 and 16 (Cell 1 pulls the branch); Cell 24 need not run first. The same crop and window as Cell
# 24 (show_soma.py, which also rewrites Cell 24's figures, not shown here; planes already fetched come from CACHE_DIR),
# shown one plane at a time with nothing of Allen's reconstruction drawn over them, centred on Allen's traced soma.
# Drag the slider, or click its handle and step with the arrow keys, or press play to run through focus. CONTRAST
# "fixed" shows every plane on one grey range, so the soma's darkening and sharpening compare across planes; "per
# plane" stretches each plane to its own range (the toggle under the slider switches between them). The button sets
# SOMA_K to the plane shown: Cell 25 starts from it when its SOMA_FIT_K is None. Running Cell 24 again resets SOMA_K
# to its own line, so type the plane there if you rerun it. VIEW_HALF_UM crops the view about the traced soma (None:
# the whole crop, +-SOMA_HALF_UM), ZOOM enlarges it in whole pixels; SOMA_Z_MARGIN_UM widens the window beyond the
# traced radius + 6 um (on the pilot k = 75..159; more planes to fetch the first time). Nothing to send back: tell me
# the plane you picked.

# %%
from IPython.display import display   # the cell's own imports
from allen_diameter.loading import soma_io
from allen_diameter.plotting import plane_viewer
VIEW_HALF_UM = 20        # the view's half-size about the traced soma (um); None: the whole crop
ZOOM = 2                 # whole-pixel enlargement of the view
CONTRAST = "fixed"       # "fixed": one grey range for every plane; "per plane": each plane its own
SOMA_Z_MARGIN_UM = None  # the window: the planes within the traced radius + this of the traced depth (um); None: 6
SOMA_OUT = OUT + "/soma"
run("scripts/show_soma.py", "--specimen", SPECIMEN, "--cache-dir", CACHE_DIR, *ALIGN, *CLEAN_CFG, "--out", SOMA_OUT,
    *([] if SOMA_Z_MARGIN_UM is None else ["--z-margin-um", SOMA_Z_MARGIN_UM]),
    *([] if globals().get("SOMA_HALF_UM") is None else ["--half-um", SOMA_HALF_UM]))
soma24b = soma_io.read_soma_view(SOMA_OUT, SPECIMEN)


def use_plane(k):
    """The viewer's button: the plane shown becomes SOMA_K."""
    globals()["SOMA_K"] = int(k)
    return "SOMA_K = %d: Cell 25 starts from this plane (with its SOMA_FIT_K None)" % k


viewer24b = plane_viewer.PlaneViewer(
    soma24b.block, soma24b.ks, soma24b.valid, left=soma24b.left, top=soma24b.top, res0_um=soma24b.res0_um,
    dz_um=soma24b.dz_um, centre_xy_um=soma24b.centre_um[:2], view_half_um=VIEW_HALF_UM, k_ref=soma24b.k_traced,
    title="specimen %s: the soma's planes, centred on Allen's traced soma (k %d: the traced depth's plane)"
          % (SPECIMEN, soma24b.k_traced),
    zoom=ZOOM, contrast=CONTRAST, k_start=globals().get("SOMA_K"), on_pick=use_plane,
    pick_label="use this plane as SOMA_K")
display(viewer24b.widget)

# %% [markdown]
# ## Cell 25: the soma fitted with an ellipsoid; the corrected SWC (D-215, D-216, D-218, D-222, D-240; Blocks 45-48)
#
# Run after Cells 0, 1, 16 and 24 or 24 (b) (Cell 1 pulls the branch). The soma is fitted with an ellipsoid -- centre,
# three semi-axes, in-plane orientation, absorption -- rendered with the partition absorption and Cell 16's kernel, on
# the planes within 10 um of the start, Allen's traced processes masked (D-216). The start is Allen's traced soma; its
# depth is SOMA_FIT_K (default: the SOMA_K of Cell 24 or 24 (b), else the traced depth). The kernel's far slope is not
# fitted (D-240, superseding D-218.1): it is held at SLOPE0 (None: the configured 0.79, uncalibrated; the far-slope
# calibration's value when there is one). A fit took 8-21 min in the sandbox (36 planes of a 280 px
# crop, 50-100 renders); Colab ran Cell 23B's renders 2-3 times slower, so allow up to about an hour. The fitted soma enters the
# corrected SWC as the sphere of equal membrane area at the fitted centre, each stem's first node moved onto the fitted
# surface (D-216.3, D-222.5). Output to send back: soma_fit_<S>_planes.png, _summary.png, _morphology.png and
# soma_fit_<S>.json; the corrected SWC is reconstruction_soma_<S>.swc. On the simulated stacks (D-216.2) chi2/dof near
# the noise's level means an ellipsoid fits; a wedge in the in-focus residuals toward the apical side is the
# pyramidal soma's signature (the area then about 5 % low).

# %%
from IPython.display import Image, display   # the cell's own imports
SOMA_FIT_K = None     # the start plane, e.g. 115; None: Cell 24's SOMA_K, else the traced depth
SLOPE0 = None         # the far slope's held value (um/um, D-240); None: the configured renderer.kernel_continuation_slope
SOMA_FIT_OUT = OUT + "/soma_fit"
if "KW_ARGS" not in globals():        # a Cell 16 from before wave 7: the kernel's width fixed, the default (D-221.1)
    KW_ARGS = ["--kernel-width", "fixed"]
print("[cell 25] kernel width:", " ".join(str(x) for x in KW_ARGS))
k_start = SOMA_FIT_K if SOMA_FIT_K is not None else globals().get("SOMA_K")
choice = (([] if k_start is None else ["--soma-k", k_start]) + ["--far-slope", "fixed"]
          + ([] if SLOPE0 is None else ["--slope0", SLOPE0]))
run("scripts/fit_soma.py", "--specimen", SPECIMEN, "--cache-dir", CACHE_DIR, *ALIGN, *CLEAN_CFG, "--out", SOMA_FIT_OUT,
    *choice, *KW_ARGS)       # Cell 16's KERNEL_WIDTH (Block 36 amendment 22)
for name in ("soma_fit_%d_planes.png" % SPECIMEN, "soma_fit_%d_summary.png" % SPECIMEN,
             "soma_fit_%d_morphology.png" % SPECIMEN):
    print(SOMA_FIT_OUT + "/" + name)
    display(Image(filename=SOMA_FIT_OUT + "/" + name))

# %% [markdown]
# ## Cell 26: the per-node fit with the SGT kernel -- Cell 20's nodes, the Gaussian fit and the SGT fit side by side (D-221, D-222, D-224, D-225; SPEC Block 49, Block 30 amendment 3, Block 36 amendment 23)
#
# Run after Cells 0, 1 and 16, once Cell 20 has prepared its nodes (OUT/cleaning_d169, read here; no download): Cell 20
# need not run again in this runtime, its knobs taking Cell 20's values when they are not defined. Per node the
# Gaussian fit of Cell 20 is run again, then the fit with the skewed generalized-t kernel (D-052.1) started from its
# estimate at the Gaussian member: the kernel's FWHM at each slab-plane defocus is the Gaussian law's (D-222.2), its
# shape (beta: the core, xi: the tails) is fitted per node, the skew zeta held at 1 (SGT_SKEW "off": the isotropic PSF;
# "fit" frees it, as an asymmetry along the node's measuring line). The node's estimate is the SGT fit's; the Gaussian
# fit's d is beside it (gauss_d_um). If the SGT fit fails, the node's status is "failed" (flag sgt_error when it
# raised) and its Gaussian fit and same-node clearance are kept (D-233.3). The error bars of both fits come from the
# model at line_fine_factor COV_FINE_FACTOR (D-224.2; 0 = the fit's own Jacobian). Cell 16's KERNEL_WIDTH applies.
# Time: Cell 20's Gaussian fits took 6 min (5353, 0.77 um) to 111 min (639, 2.82 um) on the pilot. With this cell's
# settings (hybrid, 8 x 16 directions, two starts, centre weights, p/32 error bars) a 0.8 um synthetic node took 624 s
# against 107 s for the Gaussian fit alone, 5.9 times [run, sandbox, 2026-11-23]: the Gaussian fit's error bars add
# about the fit's time again, the SGT stage about 400 s. Carried over to 5353, about 35 min -- an estimate, not
# measured on the pilot [corrected 2026-11-23, 18:25: the earlier "1.5 to 3.5 h" came from the test suite's fast
# configuration, which makes the Gaussian fit unusually cheap]. All five of Cell 20's nodes would take of the order of
# a day, beyond a Colab session: the cell starts with 5353 alone (SGT_NODES; "5353,2620" adds a node; None: all).
# COV_FINE_FACTOR = 0 drops the error bars' renders for a first look at the shape.
# Output to send back: the lines `node <id>: d ...` of the log, the table, and the `fit_node<id>.png` figures.

# %%
import glob, os                       # the cell's own imports
import pandas as pd
from IPython.display import Image, display
SGT_SKEW = "off"                      # "off": zeta = 1 (the isotropic PSF, D-222.1); "fit": zeta fitted too
COV_FINE_FACTOR = 32                  # the error bars' lattice (D-224.2); 0 = the fit's own Jacobian (faster)
SGT_NODES = "5353"                    # the thinnest of Cell 20's nodes first; "5353,2620" adds one; None: all of them
CLEAN_D, INFER_S = OUT + "/cleaning_d169", OUT + "/inference_sgt"
if "KW_ARGS" not in globals():        # a Cell 16 from before wave 7: the kernel's width fixed, the default (D-221.1)
    KW_ARGS = ["--kernel-width", "fixed"]
print("[cell 26] kernel width:", " ".join(str(x) for x in KW_ARGS), "| nodes:", SGT_NODES or "all of Cell 20's")
os.makedirs(INFER_S, exist_ok=True)
run("scripts/infer_nodes.py", "real", "--specimen", SPECIMEN, "--cleaning-dir", CLEAN_D, "--out", INFER_S,
    "--weights", globals().get("D169_WEIGHTS", "allen"), "--bump-sd", globals().get("BUMP_SD", 2.0),   # Cell 20's
    "--weight-kappa", globals().get("WEIGHT_KAPPA", 1.25), "--starts", globals().get("D169_STARTS", "1.0,1.6"),
    "--kernel", "sgt", "--sgt-skew", SGT_SKEW, "--cov-fine-factor", COV_FINE_FACTOR, *KW_ARGS,
    *([] if SGT_NODES is None else ["--nodes", SGT_NODES]))
res_s = pd.read_csv("%s/inference_%d.csv" % (INFER_S, SPECIMEN))
res_s["sgt_over_gauss"] = res_s["d_hat_um"] / res_s["gauss_d_um"]
display(res_s[["node_id", "d_allen_um", "gauss_d_um", "gauss_se_d_um", "d_hat_um", "se_d_um", "sgt_over_gauss",
               "beta", "se_beta", "xi", "se_xi", "zeta", "truncated_mass", "chi2_mean", "gauss_chi2_mean", "status",
               "gauss_status", "at_bound", "flags", "cov_source", "seconds"]].round(4))
for path in sorted(glob.glob(INFER_S + "/fit_node*.png")):
    print(path)
    display(Image(filename=path))

# %% [markdown]
# ## Cell 26 (b): the fitted tube on the node's in-focus image (scripts/show_focal.py; SPEC Block 36, amendment 12)
#
# Cell 26's estimate drawn on the node's patch at the first stage's sharpest plane k* (FOCAL26_PLANE = "axis": the
# recorded plane nearest the fitted axis depth): the estimated diameter's two edges (solid), Allen's diameter about the
# trace (dashed), the measuring line. Reads the files Cells 20 and 26 wrote; seconds. Cell 20's Gaussian fits of the
# five nodes are already drawn the same way in OUT/cleaning_d169/focal_<S>.png. On 5353 the Gaussian and the SGT
# diameters differ by 0.003 um, far below a pixel.

# %%
import pandas as pd
from IPython.display import Image, display
FOCAL26_NODES = "5353"         # Cell 26's nodes, comma-separated; None: all of them
FOCAL26_PLANE = "kstar"        # or "axis"
INF26 = OUT + "/inference_sgt"
png26 = "%s/focal_%d.png" % (INF26, SPECIMEN)
run("scripts/show_focal.py", "--cleaning-dir", OUT + "/cleaning_d169", "--specimen", SPECIMEN, "--inference-dir", INF26,
    "--plane", FOCAL26_PLANE, "--zoom-um", 4.0, "--out", png26, *(["--nodes", FOCAL26_NODES] if FOCAL26_NODES else []))
display(Image(filename=png26))
display(pd.read_csv("%s/focal_%d.csv" % (INF26, SPECIMEN)))
