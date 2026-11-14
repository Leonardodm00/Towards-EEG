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
# ## Cell 16: the configuration of the run and the contamination survey (no image is read; about 1 s per dendrite node,
# so of the order of an hour for a cell of a few thousand nodes -- measured on the 14-node synthetic cell only)
#
# Knobs: `B_GL`, `MU` (Cell 7 of 2026-11-10 printed 159.0 gl and 0.258 /um for the pilot); `KERNEL_SLOPE` (None: the
# far fit's slope if it is a production kernel, else the configured 0.79 beyond the table's last depth).
# Output to send back: the line `kernel used: ...`, the count per class, and the list of example nodes.

# %%
import json
B_GL, MU = 159.0, 0.258          # gl, 1/um: the pilot's medians (Cell 7, 2026-11-10)
KERNEL_SLOPE = None              # um of SD per um of defocus beyond the table; None = far fit if production, else configured
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
run("scripts/clean_cell.py", "survey", "--specimen", SPECIMEN, "--out", CLEAN, "--cache-dir", CACHE_DIR, *ALIGN,
    *CLEAN_CFG, "--B", B_GL, "--mu", MU)
EXAMPLES = "%s/examples_%d.json" % (CLEAN, SPECIMEN)
with open(EXAMPLES) as f:
    ex = json.load(f)
print("[cell 16] example nodes:", ex.get("example_ids"), "| classes:", ex.get("classes"))

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
ROUNDS, MAX_WORK, WORK_PEAK_GL = 2, 20, 3.0     # a first run; then 3 to 5 rounds and up to 60 nodes
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
run("scripts/infer_nodes.py", "real", "--specimen", SPECIMEN, "--cleaning-dir", CLEAN, "--out", INFER)
import pandas as pd
res = pd.read_csv("%s/inference_%d.csv" % (INFER, SPECIMEN))
display(res)
for path in [INFER + "/diameters.png", INFER + "/chi2.png"] + sorted(glob.glob(INFER + "/fit_node*.png")):
    if os.path.exists(path):
        print(path)
        display(Image(filename=path))
