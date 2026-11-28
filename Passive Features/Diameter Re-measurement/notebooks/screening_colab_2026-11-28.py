# -*- coding: ascii -*-
# TEEG diameter pipeline -- the screening run in Colab (D-283): the sheets of the apt calibration nodes of the 68 cells
#
# Jupytext "percent" format: every "# %%" starts a cell, "# %% [markdown]" a text cell. Open the twin
# screening_colab_2026-11-28.ipynb in Colab, or convert this file: pip install jupytext && jupytext --to ipynb <file>.
# Why Colab: davinci's proxy refuses Allen's server (the decision log, D-099's notes of 2026-11-28), so the run of the
# davinci handoff's section 10 (D-226, D-278 to D-281) runs here: the same cells, steps and configuration. Each cell's
# text says what must appear in its output and what to send back.

# %% [markdown]
# ## Before you start
# - Runtime: a CPU runtime is enough. Drive needs room for the outputs, one folder per cell under
#   `MyDrive/diameters/survey_screen1`; the probe (Cell 3) prints the size of one cell's sheets -- about 68 times that.
# - Order: Cells 0 and 1 in every new session; Cell 2 once; Cell 3 (the probe on the pilot) and a look at its sheets;
#   Cell 4 (every cell, several at a time); Cell 5 (status) whenever you want the table; Cell 6 at the end.
# - Everything is resumable: after a disconnection run Cells 0 and 1, then the cell that was running. Finished steps
#   are skipped; a cell stopped in the middle of a step does that step again.
# - Keep the browser tab open while Cell 4 runs: Colab stops a runtime nobody is looking at.
# - Times measured in Colab in October against Allen's server: the acquisition check 6.6 min for 158 cells; the step
#   `nodes` about an hour per cell of 120 nodes. The probe measures this run's time for one cell.

# %% [markdown]
# ## Cell 0: Drive and constants
# Expect `[cell 0] outputs .../survey_screen1 | cache /content/allen_cache_screen1 | workers 4 | cpus <n>`.

# %%
from google.colab import drive
drive.mount("/content/drive")

import os
OUT = "/content/drive/MyDrive/diameters"
SURVEY = OUT + "/survey_screen1"            # the run's folder on Drive: one folder per cell, resumable (D-283)
CACHE = "/content/allen_cache_screen1"      # the crop cache, one folder per cell, on the runtime's own disk (D-284)
SEL = SURVEY + "/selection.csv"
PILOT = 529878215
STEPS = "align,nodes,calibration,screening"
WORKERS = 4                                  # cells processed at the same time in Cell 4 (D-284)
for d in (SURVEY, CACHE):
    os.makedirs(d, exist_ok=True)
print("[cell 0] outputs %s | cache %s | workers %d | cpus %d" % (SURVEY, CACHE, WORKERS, os.cpu_count()))

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
CFG = os.path.join(WS, "configs", "survey_screen1.json")    # calib_scans_per_cell lifted: every apt node is scanned


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


# %% [markdown]
# ## Cell 2: the 68 cells -- Allen's pool, the acquisition check, the pick (D-281)
# The cells are `configs/screen1_cells.csv`: every `specimen_<id>` under `L<n>_exc` and `L<n>_inh` of the archive on
# davinci, from your listing of 2026-11-28. The check reads every pool cell's plane spacing and pixel size (about
# 7 min; resumable). The pick stops if one of the 68 is not in Allen's pool or was imaged at other settings than the
# pipeline assumes (0.28 um between planes, 0.1144 um per pixel): then send its last line and stop here.
# Expect `pool: <n> Homo Sapiens cells ...`, `check: <n> cells, 0 failed; ... <n> match the configuration ...`,
# `pick: 68 cells -> .../selection.csv`, then
# `[cell 2] 68 cells: L2_exc 10, L2_inh 1, L3_exc 18, L3_inh 11, L4_exc 6, L4_inh 2, L5_exc 8, L5_inh 5, L6_exc 5, L6_inh 2`.

# %%
import csv
with open(os.path.join(WS, "configs", "screen1_cells.csv")) as f:
    CELLS = [(r["group"], int(r["specimen_id"])) for r in csv.DictReader(f)]
run("scripts/cell_pool.py", "pool", "--out", SURVEY + "/pool.csv")
run("scripts/cell_pool.py", "check", "--pool", SURVEY + "/pool.csv", "--out", SURVEY + "/acquisition.csv")
run("scripts/cell_pool.py", "pick", "--pool", SURVEY + "/pool.csv", "--cells", ",".join(str(i) for _, i in CELLS),
    "--acquisition", SURVEY + "/acquisition.csv", "--require-match", "--out", SEL)
groups = {}
for g, _ in CELLS:
    groups[g] = groups.get(g, 0) + 1
print("[cell 2] %d cells: %s" % (len(CELLS), ", ".join("%s %d" % kv for kv in sorted(groups.items()))))

# %% [markdown]
# ## Cell 3: the probe -- the pilot 529878215, every step, then two of its sheets
# The pilot first: its earlier steps are known to work in Colab, so a failure here is the new step's. Expect, with the
# seconds each took, `[1/1] 529878215 align: ...`, `[1/1] 529878215 nodes: ...`,
# `[1/1] 529878215 calibration: ... near-field ... far-field scans`, `[1/1] 529878215 screening: ... sheets`; then
# `[cell 3] <n> sheets (<a> near, <b> far), <x> MB, <t> min` and one near and one far sheet.
# Send the four step lines and the `[cell 3]` line: they say how long the 68 cells take and how many to run at once.

# %%
import glob
from IPython.display import Image, display
t0 = time.time()
run("scripts/survey_cells.py", "run", "--selection", SEL, "--out-dir", SURVEY, "--cache-dir", CACHE,
    "--cache-per-cell", "--steps", STEPS, "--config-json", CFG, "--cells", PILOT)
sheets = sorted(glob.glob("%s/%d/screening/*.png" % (SURVEY, PILOT)))
near = [p for p in sheets if os.path.basename(p).startswith("near_")]
far = [p for p in sheets if os.path.basename(p).startswith("far_")]
mb = sum(os.path.getsize(p) for p in glob.glob("%s/%d/screening/*" % (SURVEY, PILOT))) / 1e6
print("[cell 3] %d sheets (%d near, %d far), %.1f MB, %.1f min"
      % (len(sheets), len(near), len(far), mb, (time.time() - t0) / 60))
for p in near[:1] + far[:1]:
    display(Image(filename=p))

# %% [markdown]
# ## Cell 4: every cell, WORKERS at a time (resumable)
# `scripts/survey_parallel.py` gives the cells not yet done to WORKERS processes, each with its log in
# `survey_screen1/logs/`. Every 10 minutes a line
# `[parallel] <h> h | of 68 cells: <d> done, <s> set aside, <f> failed, <n> to go | workers running <r>/<W>`.
# Stopping the cell stops the workers; running it again (after Cells 0 and 1 in a new session) resumes. At the end
# expect `[parallel] finished: ... ; <W> of <W> workers exited 0`. A failed cell has its traceback in
# `survey_screen1/<id>/errors.log`, and running the cell again retries it. A cell set aside at the alignment has no
# sheets: send me the list (Cell 5), you decide whether it runs anyway.

# %%
run("scripts/survey_parallel.py", "--workers", WORKERS, "--selection", SEL, "--out-dir", SURVEY,
    "--cache-dir", CACHE, "--steps", STEPS, "--config-json", CFG, "--poll-s", 600)

# %% [markdown]
# ## Cell 5: status -- one row per cell
# Each step `done`, `pending`, `failed` or `skipped:set_aside`, from the markers on Drive; also written to
# `survey_screen1/status.csv`. Expect `status: 68 cells -> ...`.

# %%
run("scripts/survey_cells.py", "status", "--out-dir", SURVEY, "--selection", SEL)

# %% [markdown]
# ## Cell 6: the candidates in one file, and the scans files for davinci
# `survey_screen1/candidates_all.csv` holds every cell's apt nodes (the header `specimen_id,node_id` once): delete the
# rows you do not want and save what is left as `chosen_nodes.csv` (D-261.2, D-275.3). `diameters/screen1_scans.tgz`
# holds the selection and each cell's scans file, what step 2 reads on davinci (D-283.2): download it from Drive, copy
# it to davinci as a binary file (scp, MobaXterm's browser), and unpack it in
# `/davinci-1/home/ldellamea/Human Neurons Fitting/diameter_survey/` (it creates `screen1/`).
# Expect `[cell 6] candidates of <c> cells, <m> nodes -> ...` and `[cell 6] <k> scans files -> ...`.

# %%
import tarfile
rows, n_cells = [], 0
for path in sorted(glob.glob(SURVEY + "/*/screening/candidates_*.csv")):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    rows += lines[1:]                                  # the file's own header dropped
    n_cells += 1
with open(SURVEY + "/candidates_all.csv", "w") as f:
    f.write("specimen_id,node_id\n" + "".join(r + "\n" for r in rows))
print("[cell 6] candidates of %d cells, %d nodes -> %s" % (n_cells, len(rows), SURVEY + "/candidates_all.csv"))
scans = sorted(glob.glob(SURVEY + "/*/calibration/scans_*.json"))
with tarfile.open(OUT + "/screen1_scans.tgz", "w:gz") as t:
    t.add(SEL, arcname="screen1/selection.csv")
    for p in scans:
        t.add(p, arcname="screen1/" + os.path.relpath(p, SURVEY))
print("[cell 6] %d scans files -> %s" % (len(scans), OUT + "/screen1_scans.tgz"))
