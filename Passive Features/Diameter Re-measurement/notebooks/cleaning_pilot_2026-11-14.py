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
# far fit's slope if it is a production kernel, else the configured 0.79 beyond the table's last depth).
# Output to send back: the line `kernel used: ...`, the count per class, and the list of example nodes.

# %%
import json
B_GL, MU = 159.0, 0.258          # gl, 1/um: the pilot's medians (Cell 7, 2026-11-10)
KERNEL_SLOPE = None              # um of SD per um of defocus beyond the table; None = far fit if production, else configured
SURVEY_MAX_NODES = 600          # a random sample (about 30 min); None = all 8496 nodes (about 7 h)
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
