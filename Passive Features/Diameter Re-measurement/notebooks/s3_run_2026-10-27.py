# -*- coding: ascii -*-
# TEEG diameter pipeline -- the Colab run of 2026-10-27: Cell 7 on the pilot, then S3 (Cells 11, 11b, 12, 13, 14, 15)
#
# Jupytext "percent" format: every "# %%" starts a cell, "# %% [markdown]" a text cell. Paste the cells into a Colab
# notebook in order, or convert the file: pip install jupytext && jupytext --to ipynb s3_run_2026-10-27.py
# The code is the notebook's own (notebooks/phase2_colab.py on sci/diameter-pipeline, the same cell numbers); two
# additions: a pre-flight before Cell 7, and Cell 7 passes Cell 3b's registration file only if it exists and shows
# its figures. Each cell's text says what must appear in its output and what to send back.

# %% [markdown]
# ## Before you start
# - Runtime: a CPU runtime is enough. Drive must hold `MyDrive/allen_cache` (the crop cache) and `MyDrive/diameters`.
# - Order: Cells 0 and 1 in every new session, then the pre-flight and Cell 7, then 11, 11b, 12, 13, 14, 15.
# - Every S3 step is resumable: after a disconnection run Cells 0 and 1, then the same cell again; finished cells and
#   steps are skipped.
# - Nothing here changes the configuration; Cell 8 (not in this file) is where you write the decisions.
# - Times against Allen's server are not measured: the slowest parts are Cell 11's check over the whole pool, Cell 11b's
#   ~520 crops, and Cell 13 (up to 120 nodes per cell).

# %% [markdown]
# ## Cell 0: Drive and constants
# Expect `[cell 0] specimen 529878215 | cache ... | outputs ...`.

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
# ## Pre-flight for Cell 7 (no download)
# Cell 7 selects its calibration nodes from the pilot's table (Cell 4b, `pilot/pilot_529878215.csv`) and uses Cell 3b's
# registration offsets. On Drive both exist from 2026-10-07/08 (the pilot of the chain before Block 13: Cell 7 reads such
# a table as the pilot's, D-061.2, and re-measures every candidate with today's chain before scanning it).
# Expect two `[preflight]` lines with the files' dates, then `pilot rows 393; columns missing: none; candidates n by set
# {'near': a, 'far': b}`. If a column is missing or a set has very few candidates, run Cells 3a and 4b of the notebook
# first (Cell 4b re-measures every node of the stretches: long).

# %%
# The inputs Cell 7 reads, and the calibration candidates it will consider (the repository's own rule), in a fresh
# Python like run(): no download.
import subprocess
PREFLIGHT = r"""
import os, sys, time
sys.path[:0] = ["src", "scripts"]
import calibrate_kernel
from allen_diameter.analysis import calibration
from allen_diameter.config import default_config
from allen_diameter.loading import table_io
out, sp, reg = sys.argv[1], int(sys.argv[2]), sys.argv[3]
pilot = "%s/pilot/pilot_%d.csv" % (out, sp)
for name, path in (("pilot CSV (Cell 4b)", pilot), ("registration (Cell 3b)", reg)):
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(os.path.getmtime(path))) if os.path.exists(path) else "MISSING"
    print("[preflight] %-22s %s" % (name, stamp))
if os.path.exists(pilot):
    cfg = default_config()
    rows = table_io.read_rows([pilot])
    need = ("node_id", "fit_status", "flags", "phi_rad", "alpha_hat", "allen_radius_um")
    missing = [c for c in need if rows and c not in rows[0]]
    cand = calibrate_kernel.candidates_from_rows(rows, cfg)
    by = {}
    for r in rows:
        if int(r["node_id"]) in cand:
            s = calibration.calibration_set(2.0 * float(r["allen_radius_um"]), cfg)
            by[s] = by.get(s, 0) + 1
    print("[preflight] pilot rows %d; columns missing: %s; candidates %d by set %s" % (
        len(rows), missing or "none", len(cand), by))
    print("[preflight] a production kernel needs at least 10 scanned nodes per set (D-088.3); the scan drops some "
          "(not straight, not isolated)")
"""
subprocess.run([sys.executable, "-c", PREFLIGHT, OUT, str(SPECIMEN), REG_JSON], cwd=WS, check=False)


# %% [markdown]
# ## Cell 7: the kernel calibration on the pilot -- near field (Block 10), far field (Block 15), figures
# The test on one pilot cell (D-088.2, 529878215 confirmed in D-089.8). Calibration nodes chosen by Allen's diameter:
# near field 2r <= 0.6 um scanned over k* +- 3, far field 0.6-0.8 um over k* +- 16, each on a +-25 um patch per plane;
# a node must be in S, flat (phi <= 5 deg), faint, straight and isolated. At most 30 scanned nodes per set (seeded
# order; the rest `set_full`), at least 10 for a production kernel (D-088.3).
#
# Expect from `real`: `N candidate nodes`, one line per node (`scanned (near field)`, `scanned (far field)` or `not a
# calibration node (...)`), then the counts and reasons. From `fit`: `growth fit ...: N nodes`, the knot table, the
# table cut at 0.84 um and `anchor sigma_0_hat = ... um, SE ...` -- or `INSUFFICIENT: J near-field nodes < 10`. From
# `far-fit`: `far fit: gamma_hat = ... +- ...`, the per-sign and quadratic comparisons and `production kernel: ...` (or
# `no production kernel: insufficient ...`). Then the figures. A failure here does not stop the next cells.
#
# Send back: the `real` counts and reasons line, the `fit` and `far-fit` reports, and the five figures.

# %%
os.makedirs(OUT + "/calibration/figures", exist_ok=True)
PILOT_CSV = "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN)
SCANS = "%s/calibration/scans_%d.json" % (OUT, SPECIMEN)
CALIB = "%s/calibration/calibration_%d.json" % (OUT, SPECIMEN)
FARFIT = "%s/calibration/far_fit_%d.json" % (OUT, SPECIMEN)
PATCHES = "%s/calibration/patches_%d.npz" % (OUT, SPECIMEN)
REG_ARGS = ["--registration-json", REG_JSON] if os.path.exists(REG_JSON) else []   # Cell 3b's offsets, if run
rc = run("scripts/calibrate_kernel.py", "real", "--specimen", SPECIMEN, "--nodes-csv", PILOT_CSV,
         "--cache-dir", CACHE_DIR, "--out", SCANS, *REG_ARGS, *ALIGN, check=False)
if rc == 0:
    rc = run("scripts/calibrate_kernel.py", "fit", "--scans", SCANS, "--out", CALIB, check=False)
if rc == 0:
    run("scripts/calibrate_kernel.py", "far-fit", "--scans", SCANS, "--calibration", CALIB, "--out", FARFIT, check=False)
    fig_args = ["--scans", SCANS, "--calibration", CALIB, "--out-dir", OUT + "/calibration/figures"]
    if os.path.exists(FARFIT):
        fig_args += ["--far-fit", FARFIT]
    if os.path.exists(PATCHES):
        fig_args += ["--patches", PATCHES]
    run("scripts/calibrate_kernel.py", "figures", *fig_args, check=False)
from IPython.display import Image, display
for name in ("width_by_plane_near", "width_by_plane_far", "kernel_law", "example_near", "example_far"):
    path = "%s/calibration/figures/%s.png" % (OUT, name)
    if os.path.exists(path):
        display(Image(filename=path))


# %% [markdown]
# ## Cell 11: S3 step 0 -- the cell pool, the acquisition check, the selection (Block 16.1)
# `pool`: every human cell with a reconstruction (`pool: N Homo Sapiens cells ...`, the counts by class and layer).
# `check`, over the whole pool (D-085.3): each cell's plane spacing and pixel size against 0.28 um and 0.1144 um/px,
# one line per cell, then `check: N cells, k failed; ...; m match the configuration`. Resumable. `select`: 5 cells per
# class (exc, inh, sparsely_spiny) and layer, all of a smaller group, seeded (`select: G groups, S cells selected ...`).
# REQUIRE_MATCH = True keeps only the cells that match the configuration -- yours to change after reading the check.
# A failure that names missing fields prints the keys Allen's rows hold; a warning `... without an order` means the
# pool needed two pages: write {"survey": {"pool_page_rows": 50000}} to a JSON file and pass it as --config-json.
#
# Send back: the `pool:` lines, the `check:` summary line with `survey/acquisition.json`, the `select:` lines.

# %%
SURVEY = OUT + "/survey"
os.makedirs(SURVEY, exist_ok=True)
run("scripts/cell_pool.py", "pool", "--out", SURVEY + "/pool.csv")
run("scripts/cell_pool.py", "check", "--pool", SURVEY + "/pool.csv", "--out", SURVEY + "/acquisition.csv")
REQUIRE_MATCH = True          # decide after reading the check (D-085.3)
run("scripts/cell_pool.py", "select", "--pool", SURVEY + "/pool.csv", "--acquisition", SURVEY + "/acquisition.csv",
    *(["--require-match"] if REQUIRE_MATCH else []), "--out", SURVEY + "/selection.csv")


# %% [markdown]
# ## Cell 11b: what Allen's projections are (Block 16.3, D-090.1)
# For every selected cell: the SubImages that are not Primary planes, with every field Allen returns, the planes'
# count and size, the xy projections Cell 12 chooses among and the one its darkest rule takes. For the pilot: each xy
# projection against the minimum, mean, median and maximum over its ~520 planes on a 59 um window around the soma --
# the best pixel offset, r*, the verdict (`matches min` expected for one of them). About 520 crops for the pilot.
#
# Expect `[proj] <id>: N non-Primary SubImages: ...` and `... darkest at level 4: <id>` per cell, then for the pilot
# `[proj] 529878215: <id> '<type>': matches min -- r* min 0.99.. at (+a, +b); mean ...`.
#
# Send back: `survey/projections/projection_types.csv`, `projections_529878215.json`, `projection_check_529878215.png`.

# %%
SURVEY = OUT + "/survey"
SEL = SURVEY + "/selection.csv"
run("scripts/projection_check.py", *(["--selection", SEL] if os.path.exists(SEL) else ["--specimens", SPECIMEN]),
    "--check", SPECIMEN, "--cache-dir", CACHE_DIR, "--out-dir", SURVEY + "/projections")
from IPython.display import Image, display
FIG = "%s/projections/projection_check_%d.png" % (SURVEY, SPECIMEN)
if os.path.exists(FIG):
    display(Image(filename=FIG))


# %% [markdown]
# ## Cell 12: S3 step 1 -- every selected cell's global alignment (Block 16.2)
# The minimum-intensity xy projection at level 4, `align_translation` with both flips, the score ratio R against
# block-shuffled images, a fine stage at level 1. One line per cell, `[i/S] <id> align: shift (x, y) px, ..., ratio R`
# or `SET ASIDE (reasons)` (R < 2, or a search on its window's edge; D-089.1). A set-aside cell's overlay is shown: to
# align it by hand, write `survey/<id>/align/alignment_manual.json` with {"shift_x_px": .., "shift_y_px": ..,
# "flip_h_px": null, "z0_um": 0.0} and run again with "--cells", "<id>", "--force", "align".
#
# Send back: the `survey:` summary line, `survey/status.csv`, the overlays of the set-aside cells.

# %%
SURVEY = OUT + "/survey"
run("scripts/survey_cells.py", "run", "--selection", SURVEY + "/selection.csv", "--out-dir", SURVEY,
    "--cache-dir", CACHE_DIR, "--steps", "align")
run("scripts/survey_cells.py", "status", "--out-dir", SURVEY, "--selection", SURVEY + "/selection.csv")
import glob, json as _json
from IPython.display import Image, display
for path in sorted(glob.glob(SURVEY + "/*/align/alignment_*.json")):
    with open(path) as f:
        rec = _json.load(f)
    if rec.get("set_aside"):
        print("set aside: %s %s" % (rec["specimen_id"], rec["reasons"]))
        display(Image(filename=path.replace("alignment_", "overlay_").replace(".json", ".png")))


# %% [markdown]
# ## Cell 13: S3a -- the node survey and the coverage of the phantoms' band (Block 17)
# Per aligned cell: the node sample (NODES_MODE "stratified", up to 20 nodes per bin of Allen's 2r, the default
# confirmed in D-089.3; or "all") measured by Block 13's chain. Then per bin of d_hat the share of nodes in S whose mu_hat
# lies in the phantoms' range, the pass rule (>= 95 % in every bin with >= 20 nodes), the diameters by layer and class.
#
# Expect one line per cell `[i/S] <id> nodes: stratified: n sampled, n measured, k in S (t s)`, then `coverage: ...`
# per bin and `coverage: PASS ...` or `coverage: FAIL in <bins> ...`. Resumable.
#
# Send back: the coverage lines, `survey/s3a/coverage.json`, `diameters.csv` and the four figures.

# %%
import json
SURVEY = OUT + "/survey"
NODES_MODE = "stratified"      # or "all" (D-085.4)
PHANTOM_ROWS = []              # e.g. [OUT + "/tables/rows_<hash>_0_1000.csv"]
CFG_S3 = SURVEY + "/survey_config.json"
with open(CFG_S3, "w") as f:
    json.dump({"survey": {"nodes_mode": NODES_MODE}}, f)
run("scripts/survey_cells.py", "run", "--selection", SURVEY + "/selection.csv", "--out-dir", SURVEY,
    "--cache-dir", CACHE_DIR, "--steps", "align,nodes", "--config-json", CFG_S3)
run("scripts/survey_cells.py", "pool", "--out-dir", SURVEY, "--selection", SURVEY + "/selection.csv",
    "--what", "coverage", "--config-json", CFG_S3, *(["--phantom-rows"] + PHANTOM_ROWS if PHANTOM_ROWS else []))
from IPython.display import Image, display
for name in ("coverage_mud", "coverage_share", "coverage_alpha", "diameters"):
    if os.path.exists("%s/s3a/%s.png" % (SURVEY, name)):
        display(Image(filename="%s/s3a/%s.png" % (SURVEY, name)))


# %% [markdown]
# ## Cell 14: S3b -- d_max, the node jitter, the camera across cells (Block 18)
# Per aligned cell: Allen's dendrite diameters, the registration survey on its suggested stretches, the camera inputs
# from the nodes' background. Then the suggestions: d_max = the largest Allen 2r over the cells (D-086), the jitter = the
# robust SDs of the ON nodes' offsets (D-085.8), one camera setting (D-085.9). A cell flagged z0 has a median depth offset
# above 1 um (D-089.5). These are suggestions for Cell 8 only.
#
# Expect `inputs: d_max ... um (cell, node; p99, p99.5); jitter xy ... um, z ... um; background ... gl, noise SD ... gl`
# and `inputs: suggestions only (Cell 8 decides)`.
#
# Send back: the `inputs:` lines, `survey/s3b/inputs.json`, `inputs_per_cell.csv`, `inputs.png`.

# %%
import json
SURVEY = OUT + "/survey"
CFG_S3 = SURVEY + "/survey_config.json"            # Cell 13's (the node mode); written here if Cell 13 did not run
if not os.path.exists(CFG_S3):
    with open(CFG_S3, "w") as f:
        json.dump({"survey": {"nodes_mode": "stratified"}}, f)
run("scripts/survey_cells.py", "run", "--selection", SURVEY + "/selection.csv", "--out-dir", SURVEY,
    "--cache-dir", CACHE_DIR, "--steps", "align,nodes,radius,registration,camera", "--config-json", CFG_S3)
run("scripts/survey_cells.py", "pool", "--out-dir", SURVEY, "--selection", SURVEY + "/selection.csv",
    "--what", "inputs", "--config-json", CFG_S3)
with open(SURVEY + "/s3b/inputs.json") as f:
    INPUTS = json.load(f)["suggested"]
print({k: v for k, v in INPUTS.items() if k != "jpeg_qtables"})
from IPython.display import Image, display
display(Image(filename=SURVEY + "/s3b/inputs.png"))


# %% [markdown]
# ## Cell 15: S3c -- one defocus kernel calibrated on nodes pooled across cells (Block 19)
# Per aligned cell: its calibration candidates scanned in a seeded order, at most 5 per set (D-089.6). Then per set the
# cells' scans taken in turn across the cells up to 30, at least 10 for the production kernel, and Cell 7's fit, far-fit
# and figures on the pooled scans: one kernel (D-088.1). With two cells or more, the near and far points by cell and
# their residuals from the pooled law (D-090.2): `by cell: specimen <id> near field: n nodes, m planes used; median
# residual ...`, `survey/s3c/kernel_law_by_cell.csv`, `kernel_law_by_cell.png`, `kernel_law_residuals_by_cell.png`.
#
# Expect per cell `[i/S] <id> calibration: n candidates: a near-field, b far-field scans; ...`, then `pooled near field:
# k of n scans taken (at most 30) from c cells {...} -- sufficient` (or `INSUFFICIENT`), the same for the far field,
# the fit and far-fit reports, the `by cell:` lines and `pool calibration -> .../survey/s3c`.
#
# Send back: the pooled lines, the fit and far-fit lines, the `by cell:` lines and the figures.

# %%
import json
SURVEY = OUT + "/survey"
CFG_S3 = SURVEY + "/survey_config.json"            # Cell 13's (the node mode); written here if Cell 13 did not run
if not os.path.exists(CFG_S3):
    with open(CFG_S3, "w") as f:
        json.dump({"survey": {"nodes_mode": "stratified"}}, f)
run("scripts/survey_cells.py", "run", "--selection", SURVEY + "/selection.csv", "--out-dir", SURVEY,
    "--cache-dir", CACHE_DIR, "--steps", "align,nodes,calibration", "--config-json", CFG_S3)
run("scripts/survey_cells.py", "pool", "--out-dir", SURVEY, "--selection", SURVEY + "/selection.csv",
    "--what", "calibration", "--config-json", CFG_S3)
from IPython.display import Image, display
for name in ("width_by_plane_near", "width_by_plane_far", "kernel_law", "kernel_law_by_cell",
             "kernel_law_residuals_by_cell"):
    if os.path.exists("%s/s3c/figures/%s.png" % (SURVEY, name)):
        display(Image(filename="%s/s3c/figures/%s.png" % (SURVEY, name)))


# %% [markdown]
# ## End of session (optional)
# Write everything to Drive and unmount.

# %%
drive.flush_and_unmount()

