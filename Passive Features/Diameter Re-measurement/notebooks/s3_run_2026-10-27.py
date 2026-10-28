# -*- coding: ascii -*-
# TEEG diameter pipeline -- the Colab run of 2026-10-27: Cell 7 on the pilot, then S3 (Cells 11, 11b, 12, 13, 14, 15)
#
# Jupytext "percent" format: every "# %%" starts a cell, "# %% [markdown]" a text cell. Paste the cells into a Colab
# notebook in order, or convert the file: pip install jupytext && jupytext --to ipynb s3_run_2026-10-27.py
# The code is the notebook's own (notebooks/phase2_colab.py on sci/diameter-pipeline, the same cell numbers); two
# additions: a pre-flight before Cell 7, and Cell 7 passes Cell 3b's registration file only if it exists and shows
# its figures. Each cell's text says what must appear in its output and what to send back.
# 2026-10-28 (D-098): Cell 7 also draws each candidate's isolation figure; Cells 12b and 13b added (508305938 aligned
# again on its minimum projection; the set-aside cells that look aligned run with --include-set-aside).
# 2026-10-28, later (D-099): Colab runs every cell on the pilot 529878215 only -- Cells 11-15 in survey_pilot/, Cell 13
# with 3 nodes per bin -- and the survey over the 51 selected cells runs on davinci (scripts/pbs/survey_*.pbs; runbook,
# "S3 on davinci"); Cells 12b and 13b are knobs of the array job now.

# %% [markdown]
# ## Before you start
# - Runtime: a CPU runtime is enough. Drive must hold `MyDrive/allen_cache` (the crop cache) and `MyDrive/diameters`.
# - Order: Cells 0 and 1 in every new session, then the pre-flight and Cell 7, then 11, 11b, 12, 13, 14, 15 -- all on
#   the pilot since D-099 (the title line says "on the pilot").
# - Every S3 step is resumable: after a disconnection run Cells 0 and 1, then the same cell again; finished cells and
#   steps are skipped.
# - Nothing here changes the configuration; Cell 8 (not in this file) is where you write the decisions.
# - Times against Allen's server, measured on 2026-10-28 over the 51 selected cells: Cell 11's check 6.6 min for 158
#   cells, Cell 11b 7.0 min, Cell 12 29.7 min; Cell 13 about an hour per cell of 120 nodes -- too slow for Colab, so
#   since D-099 Colab runs the pilot only (Cell 13 with 18 nodes, about 10 min) and the 51 cells run on davinci.

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
          "(not isolated; neither straightness (D-093) nor the optical depth and the flag faint (D-094) is tested)")
"""
subprocess.run([sys.executable, "-c", PREFLIGHT, OUT, str(SPECIMEN), REG_JSON], cwd=WS, check=False)


# %% [markdown]
# ## Cell 7: the kernel calibration on the pilot -- near field (Block 10), far field (Block 15), figures
# The test on one pilot cell (D-088.2, 529878215 confirmed in D-089.8). Calibration nodes chosen by Allen's diameter:
# near field 2r <= 0.6 um scanned over k* +- 3, far field 0.6-0.8 um over k* +- 16, each on a +-25 um patch per plane;
# a node must be in S (the flag faint aside, D-094), flat (phi <= 5 deg) and isolated -- not screened for
# straightness since D-093 (the calibration evaluates only the measuring line) nor for its optical depth since
# D-094 (the user: "Remove both faint"). At most 30 scanned nodes per set (seeded
# order; the rest `set_full`), at least 10 for a production kernel (D-088.3).
#
# Expect from `real`: `N candidate nodes`, one line per node (`scanned (near field)`, `scanned (far field)` or `not a
# calibration node (...)`), then the counts and reasons. From `fit`: `growth fit ...: N nodes`, the knot table, the
# table cut at 0.84 um and `anchor sigma_0_hat = ... um, SE ...` -- or `INSUFFICIENT: J near-field nodes < 10`. From
# `far-fit`: `far fit: gamma_hat = ... +- ...`, the per-sign and quadratic comparisons and `production kernel: ...` (or
# `no production kernel: insufficient ...`). Then the figures. A failure here does not stop the next cells.
#
# [2026-10-28, D-098.4] As `real` evaluates each candidate it draws the patch its background is fitted on (+-25 um,
# plane k*): (a) the traced segments -- the node's own branch blue, a neighbour orange when its predicted shadow in the
# patch exceeds 0.03 gl in some plane (what makes `not_isolated`), grey below -- and (b) the far pixels with the fitted
# background plane. `real` ends with `isolation figures: N candidates (k with a neighbour's predicted shadow above
# 0.03 gl) -> .../calibration/figures/isolation`; one PNG per candidate (`<set>_node_<id>.png`) and overview pages
# (`isolation_overview_<n>.png`, shown below). Costs one crop per candidate that is not scanned.
#
# Send back: the `real` counts and reasons line, the `fit` and `far-fit` reports, the five figures, and the isolation
# overview pages (or the per-node figures of the candidates you want to discuss).

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
import glob
for path in sorted(glob.glob(OUT + "/calibration/figures/isolation/isolation_overview_*.png")):   # D-098.4
    display(Image(filename=path))


# %% [markdown]
# ## Cell 11: S3 step 0 on the pilot -- the pool, the pilot's selection, its acquisition check (Block 16.1)
# Since 2026-10-28 (D-099) every Colab cell runs on the pilot 529878215 only, in its own folder `survey_pilot/`
# (`survey/` keeps the run of 2026-10-28 over 51 cells). The run over the selected cells is on davinci since
# 2026-10-28 (D-099; runbook, section "S3 on davinci"): `scripts/pbs/survey_prepare.pbs`, the array
# `scripts/pbs/survey_cells.pbs` (one task per cell), `scripts/pbs/survey_pool.pbs`.
#
# `pool`: every human cell with a reconstruction (`pool: N Homo Sapiens cells ...`; 0.1 min). `pick`: the pilot's
# selection record from the pool (Block 16.1 (e); `pick: 1 cells -> .../survey_pilot/selection.csv`). `check` on that
# one-row file: its plane spacing and pixel size against 0.28 um and 0.1144 um/px (`check: 1 cells, 0 failed; ...;
# 1 match the configuration`). The stratified selection of 51 cells (`select`) is the prepare job's on davinci.
#
# Send back: the `pick:` line and the `check:` summary line.

# %%
SURVEY = OUT + "/survey_pilot"     # the pilot's own folder (D-099); survey/ keeps the run of 2026-10-28 over 51 cells
os.makedirs(SURVEY, exist_ok=True)
run("scripts/cell_pool.py", "pool", "--out", SURVEY + "/pool.csv")
run("scripts/cell_pool.py", "pick", "--pool", SURVEY + "/pool.csv", "--cells", SPECIMEN, "--out", SURVEY + "/selection.csv")
run("scripts/cell_pool.py", "check", "--pool", SURVEY + "/selection.csv", "--out", SURVEY + "/acquisition.csv")


# %% [markdown]
# ## Cell 11b: what Allen's projections are, on the pilot (Block 16.3, D-090.1)
# The pilot's SubImages that are not Primary planes, with every field Allen returns, the planes' count and size, the xy
# projections Cell 12 chooses among and the one its rule takes; then each xy projection against the minimum, mean,
# median and maximum over its ~520 planes on a 59 um window around the soma -- the best pixel offset, r*, the verdict
# (`matches min` expected for one of them). The ~520 crops are in Drive's cache since the run of 2026-10-28, so this
# takes a minute; the listing over the selected cells is the prepare job's on davinci.
#
# Expect `[proj] 529878215: N non-Primary SubImages: ...`, `... darkest at level 4: <id>` and one
# `[proj] 529878215: <id> '<type>': matches min -- r* min 0.99.. at (+a, +b); mean ...` line per xy projection.
#
# Send back: `survey_pilot/projections/projection_types.csv`, `projections_529878215.json`, `projection_check_529878215.png`.

# %%
SURVEY = OUT + "/survey_pilot"
SEL = SURVEY + "/selection.csv"
run("scripts/projection_check.py", *(["--selection", SEL] if os.path.exists(SEL) else ["--specimens", SPECIMEN]),
    "--check", SPECIMEN, "--cache-dir", CACHE_DIR, "--out-dir", SURVEY + "/projections")
from IPython.display import Image, display
FIG = "%s/projections/projection_check_%d.png" % (SURVEY, SPECIMEN)
if os.path.exists(FIG):
    display(Image(filename=FIG))


# %% [markdown]
# ## Cell 12: S3 step 1 on the pilot -- its global alignment (Block 16.2)
# The xy projection of type 'MinimumIntensityProjection - xy' at level 4 (D-098.2; the darkest only when no SubImage has
# that type), `align_translation` with both flips, the score ratio R against block-shuffled images, a fine stage at
# level 1. The pilot's design alignment is no shift and no flip (Cell 0), so expect
# `[1/1] 529878215 align: shift (x, y) px` with x and y near 0, `no flip`, `ratio R` well above 2, and the overlay with the
# skeleton on the dendrites. `SET ASIDE` here means the alignment code, not the cell, needs a look: send the overlay.
# The 51 cells' alignments are the array's `align` step on davinci; a cell to align again there is
# `qsub -v RUN_TAG=v1,DIAM_CELLS=<id>,DIAM_FORCE=align scripts/pbs/survey_cells.pbs`.
#
# Send back: the `align:` line and the overlay.

# %%
SURVEY = OUT + "/survey_pilot"
run("scripts/survey_cells.py", "run", "--selection", SURVEY + "/selection.csv", "--out-dir", SURVEY,
    "--cache-dir", CACHE_DIR, "--steps", "align")
run("scripts/survey_cells.py", "status", "--out-dir", SURVEY, "--selection", SURVEY + "/selection.csv")
from IPython.display import Image, display
display(Image(filename="%s/%d/align/overlay_%d.png" % (SURVEY, SPECIMEN, SPECIMEN)))


# %% [markdown]
# ## Cells 12b and 13b (2026-10-28, morning): retired to davinci (D-099)
# What they did is a knob of the array job now: 508305938 aligned again on its minimum projection (D-098.2) is
# `qsub -v RUN_TAG=v1,DIAM_CELLS=508305938,DIAM_FORCE=align scripts/pbs/survey_cells.pbs`; the set-aside cells whose
# overlays look aligned (576084692, 596887423, 611598198, and 571464200 after your look at full resolution; D-098.5) run
# with `qsub -v RUN_TAG=v1,DIAM_CELLS=576084692,596887423,611598198,DIAM_INCLUDE_SET_ASIDE=1 \
# scripts/pbs/survey_cells.pbs`, then `survey_pool.pbs` pools the coverage again. Both without -J: one job on the
# named cells.

# %% [markdown]
# ## Cell 13: S3a on the pilot -- the node survey and the coverage of the phantoms' band (Block 17)
# The pilot's node sample (NODES_MODE "stratified"; NODES_PER_BIN = 3 per bin of Allen's 2r, the Colab pilot's size since
# D-099, PROVISIONAL -- at most 18 nodes, about 10 min at the 30 s per node measured on 2026-10-28; the davinci array keeps
# the default 20) measured by Block 13's chain; then the coverage over these nodes: per bin of d_hat the share of nodes in
# S whose mu_hat lies in the phantoms' range, the pass rule, the diameters by layer and class. With 18 nodes every bin is
# `too_few` (the rule counts bins of 20 or more): the cell checks the chain and the pool end to end, the numbers that
# decide the mu range are the davinci run's (`survey_pool.pbs`, `s3a/`).
#
# Expect `[1/1] 529878215 nodes: stratified: n sampled, n measured, k in S (t s)` with n <= 18, then `coverage: S
# recomputed ...`, one `coverage:` line per bin and `coverage: PASS ...` or `coverage: FAIL ...` (or every bin too_few).
# Resumable: a run with another NODES_PER_BIN needs "--force", "nodes".
#
# Send back: the `nodes:` line and the coverage lines.

# %%
import json
SURVEY = OUT + "/survey_pilot"
NODES_MODE = "stratified"      # or "all" (D-085.4)
NODES_PER_BIN = 3              # the Colab pilot's size (D-099, PROVISIONAL): <= 3 x 6 bins = 18 nodes; davinci keeps 20
PHANTOM_ROWS = []              # e.g. [OUT + "/tables/rows_<hash>_0_1000.csv"]
CFG_S3 = SURVEY + "/survey_config.json"
with open(CFG_S3, "w") as f:
    json.dump({"survey": {"nodes_mode": NODES_MODE, "nodes_per_bin": NODES_PER_BIN}}, f)
run("scripts/survey_cells.py", "run", "--selection", SURVEY + "/selection.csv", "--out-dir", SURVEY,
    "--cache-dir", CACHE_DIR, "--steps", "align,nodes", "--config-json", CFG_S3)
run("scripts/survey_cells.py", "pool", "--out-dir", SURVEY, "--selection", SURVEY + "/selection.csv",
    "--what", "coverage", "--config-json", CFG_S3, *(["--phantom-rows"] + PHANTOM_ROWS if PHANTOM_ROWS else []))
from IPython.display import Image, display
for name in ("coverage_mud", "coverage_share", "coverage_alpha", "diameters"):
    if os.path.exists("%s/s3a/%s.png" % (SURVEY, name)):
        display(Image(filename="%s/s3a/%s.png" % (SURVEY, name)))


# %% [markdown]
# ## Cell 14: S3b on the pilot -- d_max, the node jitter, the camera (Block 18)
# On the pilot's node sample of Cell 13: Allen's dendrite diameters, the registration survey on its suggested stretches,
# the camera inputs from the nodes' background; then the suggestions over this one cell: d_max = the largest Allen 2r
# (D-086), the jitter = the robust SDs of the ON nodes' offsets (D-085.8), one camera setting (D-085.9). On one cell of
# 18 nodes these are a check of the steps, not the suggestions for Cell 8: those come from the davinci run
# (`survey_pool.pbs`, `s3b/inputs.json`).
#
# Expect one `radius:`, `registration:` and `camera:` line for the pilot, then `inputs: d_max ... um (cell, node; p99,
# p99.5); jitter xy ... um, z ... um; background ... gl, noise SD ... gl` and `inputs: suggestions only (Cell 8 decides)`.
#
# Send back: the `inputs:` lines and `survey_pilot/s3b/inputs.png`.

# %%
import json
SURVEY = OUT + "/survey_pilot"
CFG_S3 = SURVEY + "/survey_config.json"     # Cell 13's (node mode, the pilot's nodes per bin); written if Cell 13 did not run
if not os.path.exists(CFG_S3):
    with open(CFG_S3, "w") as f:
        json.dump({"survey": {"nodes_mode": "stratified", "nodes_per_bin": 3}}, f)
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
# ## Cell 15: S3c on the pilot -- the kernel calibration step and the pool on one cell (Block 19)
# The pilot's calibration candidates among Cell 13's nodes, scanned in a seeded order, at most 5 per set (D-089.6); then
# the pool over this one cell: per set the scans taken (up to 30, at least 10 for a production kernel) and Cell 7's fit,
# far-fit and figures on them. With 18 nodes a set is usually `INSUFFICIENT` (fewer than 10 scans): the cell checks the
# step and the pool; the pooled kernel across cells is the davinci run's (`survey_pool.pbs`, `s3c/`), and Cell 7 remains
# the pilot's full calibration (D-088.2).
#
# Expect `[1/1] 529878215 calibration: n candidates: a near-field, b far-field scans; ...`, then `pooled near field: k of
# n scans taken (at most 30) from 1 cells {...} -- sufficient` or `INSUFFICIENT`, the same for the far field, the fit and
# far-fit reports when a set is sufficient, and `pool calibration -> .../survey_pilot/s3c`.
#
# Send back: the `calibration:` line and the pooled lines.

# %%
import json
SURVEY = OUT + "/survey_pilot"
CFG_S3 = SURVEY + "/survey_config.json"     # Cell 13's (node mode, the pilot's nodes per bin); written if Cell 13 did not run
if not os.path.exists(CFG_S3):
    with open(CFG_S3, "w") as f:
        json.dump({"survey": {"nodes_mode": "stratified", "nodes_per_bin": 3}}, f)
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

