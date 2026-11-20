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
# Knobs: WEIGHT_KAPPA, BUMP_SD (0 = no cut), D169_WEIGHTS ("none" = unweighted), D169_STARTS.
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
    "--weights", D169_WEIGHTS, "--bump-sd", BUMP_SD, "--weight-kappa", WEIGHT_KAPPA, "--starts", D169_STARTS)
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
# ## Cell 22: the clear-node fit on the four chosen nodes, with a weight width of the user's choosing (D-199)
#
# Run after Cells 0, 1 and 16. The user's four nodes of 2026-11-20 (D-199): 3883 and 28, judged clean by Cell 21's
# measuring-line verdict (D-198), and 6049 and 2657, not judged clean (blobs near the line). The same two steps as
# Cell 20 (D-169, D-171 to D-173), on these nodes only:
#   1. `clean --no-clean` into OUT/cleaning_fit4: per node the first stage at Allen's diameter, the patch, the nine
#      planes, the line as recorded; nothing cleaned. About a minute per node (reused when the folder holds these
#      nodes; RERUN_FIT4_PREP = True prepares them again).
#   2. `infer_nodes.py real`: the multi-plane one-tube fit started at Allen's diameter, residuals weighted by
#      w_n = exp(-|v_n - v_c| / lambda) about the fitted centre v_c, lambda = WEIGHT_KAPPA x FWHM expected from
#      Allen's diameter (FWHM^2 = (0.866 D_Allen)^2 + (2.355 sigma_r(0))^2, sigma_r(0) = 0.080 um), re-centred while
#      fitting; the near-bump cut at BUMP_SD noise SDs. The weights multiply the squared residuals.
# WEIGHT_KAPPA = 0.1 is the user's trial (2026-11-20). D-171.1's default is 1.25 (Cell 20's first run), the pilot's
# second run 0.5. At 0.1, lambda is 0.045 to 0.071 um for these four nodes (Allen 0.47 to 0.79 um), 0.40 to 0.62 of
# a line sample (0.1144 um): with the centre on a sample, its two neighbours weigh 0.08 to 0.20 and the next two
# 0.006 to 0.04 -- the weights of a plane sum to 1.2 to 1.5 samples, against 4.0 to 6.3 at 0.5 and 9.9 to 15.1 at
# 1.25. So the width of each dip barely enters; the diameter is set mainly by how the central depth changes across
# the nine planes, and it trades against the absorption mu.
# The fit folder carries the weight width in its name (OUT/inference_fit4_k<WEIGHT_KAPPA>), so a run at another
# WEIGHT_KAPPA is written beside this one instead of over it; the focal figure goes there too.
# Time: not measured at this width. The pilot's second run (WEIGHT_KAPPA 0.5) took 3 to 49 min per node; it grows
# with the diameter the fit reaches.
# Output to send back: the lines `node <id>: weights ...` and `node <id>: d ...` of the log, the table, the
# focal figure and the `fit_node<id>.png` figures (the folder OUT/inference_fit4_k<WEIGHT_KAPPA> as a zip is enough).

# %%
import glob, json, os                # the cell's own imports: it can follow Cell 16 directly
import pandas as pd
from IPython.display import Image, display
FIT4_IDS = [3883, 28, 6049, 2657]     # the user's four nodes (D-199)
WEIGHT_KAPPA = 0.1                    # lambda in units of the FWHM from Allen's diameter; the user's trial (D-171.1's default 1.25)
BUMP_SD = 2.0                         # a near bump rises at least this many noise SDs (D-171.3); 0 = no cut
FIT4_WEIGHTS = "allen"                # "allen": the centre weights; "none": unweighted (the bump cut still applies)
FIT4_STARTS = "1.0,1.6"               # the fit's starts, x Allen's diameter
RERUN_FIT4_PREP = False               # True: prepare the nodes again although the folder holds them
CLEAN_F4 = OUT + "/cleaning_fit4"
INFER_F4 = "%s/inference_fit4_k%g" % (OUT, WEIGHT_KAPPA)
os.makedirs(CLEAN_F4, exist_ok=True)
os.makedirs(INFER_F4, exist_ok=True)
EXAMPLES_F4 = "%s/examples_fit4_%d.json" % (CLEAN_F4, SPECIMEN)
REC_F4 = "%s/cleaning_%d.json" % (CLEAN_F4, SPECIMEN)
prepared = []
if os.path.exists(REC_F4):
    with open(REC_F4) as f:
        prepared = [int(i) for i in json.load(f).get("examples_requested", [])]
if RERUN_FIT4_PREP or sorted(prepared) != sorted(FIT4_IDS):
    with open(EXAMPLES_F4, "w") as f:
        json.dump({"example_ids": [int(i) for i in FIT4_IDS]}, f)
    run("scripts/clean_cell.py", "clean", "--specimen", SPECIMEN, "--out", CLEAN_F4, "--cache-dir", CACHE_DIR, *ALIGN,
        *CLEAN_CFG, "--examples", EXAMPLES_F4, "--B", B_GL, "--mu", MU, "--no-clean")
with open(REC_F4) as f:
    rec_f4 = json.load(f)
print("[cell 22] nodes prepared:", rec_f4["example_ids"], "| dropped:", rec_f4.get("dropped") or "none")
run("scripts/infer_nodes.py", "real", "--specimen", SPECIMEN, "--cleaning-dir", CLEAN_F4, "--out", INFER_F4,
    "--weights", FIT4_WEIGHTS, "--bump-sd", BUMP_SD, "--weight-kappa", WEIGHT_KAPPA, "--starts", FIT4_STARTS)
FOCAL_F4 = "%s/focal_%d.png" % (INFER_F4, SPECIMEN)
run("scripts/show_focal.py", "--cleaning-dir", CLEAN_F4, "--specimen", SPECIMEN, "--inference-dir", INFER_F4,
    "--zoom-um", 6.0, "--out", FOCAL_F4)
res_f4 = pd.read_csv("%s/inference_%d.csv" % (INFER_F4, SPECIMEN))
res_f4["d_over_allen"] = res_f4["d_hat_um"] / res_f4["d_allen_um"]
display(res_f4[["node_id", "d_allen_um", "d_hat_um", "se_d_um", "d_over_allen", "mu_hat_per_um", "v0_hat_um",
                "z_ax_hat_um", "chi2_mean", "n_planes_used", "lambda_um", "v_cut_lo_um", "v_cut_hi_um", "n_solves",
                "status", "at_bound", "flags"]].round(4))
display(Image(filename=FOCAL_F4))
for path in sorted(glob.glob(INFER_F4 + "/fit_node*.png")):
    print(path)
    display(Image(filename=path))

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
