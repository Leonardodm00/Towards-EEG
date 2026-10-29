# Diameter pipeline, Phase II (Colab), specimen 529878215: the cells of notebooks/phase2_colab.ipynb as plain
# Python. Paste each block between two '# ===== CELL' banners into its own Colab cell and run them in order.
# Cell 9 runs on davinci, not in Colab: its commands are in the comment block before Cell 10. Since 2026-10-28
# (D-099) Cells 11-15 run on the pilot only (survey_pilot/); the survey over the selected cells is davinci's
# (scripts/pbs/survey_prepare.pbs, survey_cells.pbs, survey_pool.pbs; runbook, "S3 on davinci").
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
# CELL 4E: PLANE-TO-PLANE EVALUATION AROUND A NODE
# The evaluation proposed on 2026-10-08, as a diagnostic that changes no measurement. For each node, the
# planes k_SWC - 6 to k_SWC + 6 of its block. In every plane, the profile I_k(v) along the node's measuring
# line (the line the focus scores use: through the SWC node, across its fitted heading, |v| <= h, bilinear),
# the area under it A_k = int I_k dv, and the difference A_(k+1) - A_k between consecutive planes. Beside it,
# the same areas with each plane divided by its own background level B_k (the interquartile mean of the
# block's pixels more than Allen's radius + 4 um from every traced dendrite and the soma) and multiplied by
# the planes' mean, so that a change of a whole plane's brightness drops out. The figure shows the planes with
# the measuring line (frames: smallest area blue, smallest normalised area violet, k* red, the dip depth's
# plane grey dashed, the SWC plane dotted), the profiles, the two area curves, their differences, and B_k.
#
# `LINE_HALF_UMS` runs the evaluation once per half-length h of the line, each into its own folder
# (`pilot/planediff/line_3um`, `line_5um`) [added 2026-10-08, evening: the user's proposal to enlarge the
# line]. The pipeline's line is +-3 um. The square around the node grows to hold a longer line; at +-5 um it
# reaches past the pilot's block, so its planes are fetched again (13 new crops per node). The JSON also holds
# every plane's profile (`v_um`, `profiles`).
#
# Expect, for each h, one `[planediff] node N: planes ... (0 missing); smallest area at k ..., normalised k
# ...; background ... gl; k* ..., dip-depth plane ..., SWC plane ...` line per node, then `[planediff] wrote N
# figures (0 skipped) ...`. Add `"--evaluation", "image"` to the command for the pixel-difference version (the
# positive part of I_(k+1) - I_k over the 10 x 10 um square).
#
# What to expect (synthetic tubes, SPEC Block 11): blur moves light but does not remove it, so the area of a
# profile does not change while the dip stays inside the line; for a 0.5 um or a 1.5 um tube the smallest area
# falls at random. For a tube wider than the line, defocus pulls background light into the line, so the area
# is smallest in focus: for a 5 um tube and a +-3 um line, within one plane of the centre in 6 of 8 noise
# draws. A line long enough to hold the blurred tube keeps all its light, and that signal fades: on the 5 um
# tube the area's range shrinks from 9.0 to 5.0 to 2.5 grey levels x um at +-3, +-4 and +-5 um. A step in B_k
# (a whole plane brighter or darker) moves the smallest raw area onto that plane; the normalised curve does
# not follow it.
# Cell 4e: plane-to-plane evaluation around a node (needs Cells 0 and 1 only)
import os
from IPython.display import Image, display
DIFF_NODES = [2, 3]       # node ids; 2 and 3 are trunk nodes of the pilot
LINE_HALF_UMS = [3, 5]    # half-lengths of the measuring line to compare, um (the pipeline's line is +-3 um)
for h in LINE_HALF_UMS:
    out = "%s/pilot/planediff/line_%gum" % (OUT, h)
    run("scripts/plane_differences.py", "--specimen", SPECIMEN, "--nodes", ",".join(str(n) for n in DIFF_NODES),
        "--planes-half", 6, "--profile-half-um", h, "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
    for n in DIFF_NODES:
        p = "%s/planediff_%d.png" % (out, n)
        if os.path.exists(p):
            print("line +-%g um: %s" % (h, os.path.basename(p)))
            display(Image(filename=p))

# ====================================================================================================
# ===== CELL 4f
# ====================================================================================================
# CELL 4F: ENTROPY OF THE GREY LEVELS AROUND A NODE
# The user's proposal of 2026-10-08 (17:07), as a diagnostic that changes no measurement. For each node, the
# planes k_SWC - 6 to k_SWC + 6 of its block, and in every plane the Shannon entropy H (bits) of the
# grey-level histogram, one bin per grey level, of two sets of values: the bilinear samples along the node's
# measuring line (the line of Cell 4e, |v| <= h: 53 samples at h = 3 um, 89 at 5 um), and the pixels of a
# strip centred on that line, as long as the line across the dendrite and 2 um along it (`--stripe-half-um`
# sets the half-width along it). The figure shows the planes with the line and the strip's outline (frames:
# the plane the line's entropy picks blue, the strip's amber, k* red, the dip depth's plane grey dashed, the
# SWC plane dotted), the histograms of both sets coloured by plane, and the two entropy curves.
#
# Each curve picks its global minimum (`--entropy-pick min`) [changed 2026-10-08, evening: on nodes 2 and 3
# the curves have no W shape, and the earlier rule, the minimum between the two largest maxima
# (`--entropy-pick dip`), skipped node 2's lowest plane]. `LINE_HALF_UMS` runs the evaluation once per
# half-length, as in Cell 4e, into `pilot/planeentropy/line_3um` and `line_5um`.
#
# Expect, for each h, one `[planediff] node N: planes ... (0 missing); lowest entropy at k ... along the line
# (53 samples), k ... in the strip (... pixels); k* ..., dip-depth plane ..., SWC plane ...` line per node (89
# samples and about 1500 pixels at 5 um), then `[planediff] wrote N figures (0 skipped) ...`. The files are
# `planeentropy_<id>.png` and `planeentropy_529878215.json`.
#
# What to expect (synthetic tubes, SPEC Block 11): the entropy is lowest in focus, where most of the
# neighbourhood sits at the ground level and the tube covers few pixels; it rises on both sides as the blur
# spreads the tube's darkness over more pixels and more grey levels, and falls again far out as a thin tube
# fades into the noise. With a +-3 um line that last fall can put the strip's lowest entropy on an end plane
# (0.5 um tube, 7 of 8 noise draws); with +-4 or +-5 um the global minima of both curves fall within one plane
# of the centre in 8 of 8 draws for the 0.5 um tube, and for the 1.5 um tube in 8 of 8 (strip) and 7 or 8 of 8
# (line). For a faint 5 um tube both curves stay flat at every length, as every other score does. A whole
# plane brighter or darker by whole grey levels keeps its entropy (the area of Cell 4e moves with it). With
# the earlier dip rule and a +-3 um line (the default until 2026-10-08, evening), the strip's pick fell within
# one plane of the centre in 8 of 8 noise draws for the 0.5 um and the 1.5 um tube, and the line's in 7 and 8
# of 8. The line's 53 samples give a lower and noisier entropy than the strip's pixels: on flat ground with 3
# grey levels of noise, 0.25 bits low with a spread of 0.13 bits, against 0.02 and 0.03.
# Cell 4f: entropy of the grey levels around a node (needs Cells 0 and 1 only)
import os
from IPython.display import Image, display
ENTROPY_NODES = [2, 3]    # node ids; 2 and 3 are trunk nodes of the pilot
LINE_HALF_UMS = [3, 5]    # half-lengths of the measuring line (and of the strip across the dendrite), um
for h in LINE_HALF_UMS:
    out = "%s/pilot/planeentropy/line_%gum" % (OUT, h)
    run("scripts/plane_differences.py", "--specimen", SPECIMEN, "--nodes", ",".join(str(n) for n in ENTROPY_NODES),
        "--evaluation", "entropy", "--planes-half", 6, "--profile-half-um", h, "--stripe-half-um", 1.0,
        "--entropy-pick", "min", "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
    for n in ENTROPY_NODES:
        p = "%s/planeentropy_%d.png" % (out, n)
        if os.path.exists(p):
            print("line +-%g um: %s" % (h, os.path.basename(p)))
            display(Image(filename=p))

# ====================================================================================================
# ===== CELL 4g
# ====================================================================================================
# CELL 4G: THE ENTROPY ON THIN DENDRITES, LINE +-5 UM
# The test asked for on 2026-10-09 (14:30): the entropy along a +-5 um line, which picked planes 114 and 115
# on the trunk nodes 2 and 3 where k* picked 123, is run on thin dendrites, where the gradient energy is
# thought to work. The nodes come from the pilot's table (`--nodes thin`): in S, Allen's 2r <= 0.6 um, a
# fitted d <= 1.0 um, k* equal to the dip depth's plane (the pipeline's two focus rules agree, so k* is a
# credible reference), not steep; up to 3 per unbranched stretch, spread along it, and 12 in all, spread over
# the stretches. The thresholds are the assistant's provisional choice; `THIN` sets them.
#
# Output in `pilot/planeentropy/thin_line_5um/`: one figure per node, as in Cell 4f;
# `planeentropy_summary_529878215.png`, one panel per node with both entropy curves rescaled 0-1 against k -
# k_SWC, k*, the dip depth's plane, the SWC plane and the two picks; and `planeentropy_summary_529878215.csv`,
# per node the type, Allen's 2r, the fitted d, the planes and each pick minus k*, printed by the cell. [added
# 2026-10-09, afternoon: the user's request] The cell then shows each node's figure, in the summary's order,
# with the planes drawn without the measuring line and the strip's outline (`--hide-line`), so that the focus
# can be judged by eye; the node is at the centre of every panel and the line runs across the dendrite through
# it. Cells 4e and 4f still draw the line; `--hide-line` works there too.
#
# Expect `[planediff] thin nodes (...): <ids>`, one `[planediff] node N: ...` line per node, then `[planediff]
# summary over N nodes: k_h_line within 1 plane of k* in a of N (median |diff| ... planes); k_h_strip within 1
# plane of k* in b of N (...)` and `[planediff] wrote N figures (0 skipped) ...`; about 13 new crops per node
# (the +-5 um square).
#
# What to expect (synthetic tubes, SPEC Block 11): with a +-5 um line the global minimum of both curves fell
# within one plane of the centre in 8 of 8 noise draws for a 0.5 um tube. If the thin nodes agree with k* too,
# the entropy works on both kinds of node; where it does not, the panels show how it parts from k* (a
# neighbouring neurite inside the line or the strip is the first thing to look for).
# Cell 4g: the entropy on thin dendrites, line +-5 um (needs Cells 0 and 1 and the pilot's CSV of Cell 4b)
import json
import os
import pandas as pd
from IPython.display import Image, display
THIN = dict(max_nodes=12, per_stretch=3, max_2r_um=0.6, max_dhat_um=1.0)   # the choice of thin reference nodes
out = OUT + "/pilot/planeentropy/thin_line_5um"
run("scripts/plane_differences.py", "--specimen", SPECIMEN, "--nodes", "thin",
    "--pilot-csv", "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN), "--max-nodes", THIN["max_nodes"],
    "--per-stretch", THIN["per_stretch"], "--thin-max-2r-um", THIN["max_2r_um"],
    "--thin-max-dhat-um", THIN["max_dhat_um"], "--evaluation", "entropy", "--planes-half", 6,
    "--profile-half-um", 5, "--stripe-half-um", 1.0, "--entropy-pick", "min", "--hide-line",
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
summ = "%s/planeentropy_summary_%d.csv" % (out, SPECIMEN)
if os.path.exists(summ):
    print(pd.read_csv(summ).to_string(index=False))
p = "%s/planeentropy_summary_%d.png" % (out, SPECIMEN)
if os.path.exists(p):
    display(Image(filename=p))
recs = "%s/planeentropy_%d.json" % (out, SPECIMEN)
if os.path.exists(recs):
    with open(recs) as f:
        nodes = [r["node_id"] for r in json.load(f) if "skipped" not in r]
    for n in nodes:       # each node's figure, as in Cell 4f; its planes drawn without the line (--hide-line)
        p = "%s/planeentropy_%d.png" % (out, n)
        if os.path.exists(p):
            print(os.path.basename(p))
            display(Image(filename=p))

# ====================================================================================================
# ===== CELL 4h
# ====================================================================================================
# CELL 4H: THE GRADIENT ENERGY OVER THE WHOLE LINE, ON THREE LINES, ACROSS DIAMETERS
# The comparison asked for on 2026-10-09 (16:19), a diagnostic that changes no measurement (D-040). On the
# trunk nodes 2 and 3 the pipeline's focus rule, k*, integrates (dI~_k/dv)^2 only over |v| <= r_Allen + 0.5
# um, about 1.6 um, while the fitted diameter d_hat there is about 5 um: the window ends inside the dark core
# and sees neither edge. Here the gradient energy of plane k, G_k = B_k^-2 * integral of (dI~_k/dv)^2 dv, is
# integrated over the whole of each of three lines through the node, across the dendrite: +-3 um, +-5 um, and
# the d-line +-m d_hat / 2, sized from the pilot's fitted diameter d_hat of the node (full width m d_hat;
# `LINE_MULT` sets m, and 2 gives the line [-d_hat, d_hat]). I~_k is the profile along the line, smoothed by
# one sample; B_k is the plane's background, the interquartile mean of the block's pixels far from every
# traced dendrite (as in Cell 4e), the same for the three lines. Each line picks the plane of its largest G_k.
#
# The nodes come from the pilot's table (`--nodes bydiameter`): the rows with a converged fit, a finite z_sub,
# a finite positive d_hat, not steep and without the `crossing` flag, split by d_hat into the bins (0, 0.8],
# (0.8, 1.0], (1.0, 1.5], (1.5, 2.0], (2.0, 3.0] and (3.0, inf) um; up to 2 per bin, spread along the
# stretches; then the trunk nodes 2 and 3, whatever their row says; all ordered by d_hat. The bins and the
# counts are the assistant's provisional choice; `BYDIAM` sets them. Each node is measured again on its cached
# crops, so the d_hat of its d-line equals the pilot's for an unchanged configuration.
#
# Output in `pilot/planegrad/bydiameter/`: one figure per node, `planegrad_<id>.png`: the planes on one grey
# scale without any line, framed at each line's pick (blue G3, violet G5, aqua Gd) and at k* (red), the dip
# depth's plane (grey, dashed) and the SWC plane (black, dotted); below, every plane's profile along the
# longest line with each line's extent dashed, each line's G_k relative to its own maximum with its pick, and
# B_k. `planegrad_summary_529878215.png`: one panel per node, the three curves rescaled 0-1 against k - k_SWC,
# with the picks and each pick minus k* in the panel's title; `planegrad_summary_529878215.csv`: per node the
# type, Allen's 2r, d_hat, the d-line's half-length, the planes and each pick minus k*, printed by the cell.
# The cell then shows each node's figure, in order of d_hat.
#
# Expect `[planediff] nodes by diameter (...): <id> (<d_hat> um), ...`, one `[planediff] node N: planes ... (0
# missing); d_hat ... um; G picks: +-3 um k ..., +-5 um k ..., +-... um (d-line) k ...; k* ..., dip-depth
# plane ..., SWC plane ...` line per node, then `[planediff] summary over N nodes: k_G3 within 1 plane of k*
# in a of N (median |diff| ... planes); k_G5 ...; k_Gd ...` and `[planediff] wrote N figures (0 skipped) ...`.
# A node whose d-line would hold fewer than 3 samples keeps the two fixed lines, and its line says `no d-line:
# ...`.
#
# What to expect (synthetic tubes, SPEC Block 11): on a 0.8 um tube (Allen radius 0.3 um) the three lines and
# k* pick the in-focus plane on every node; on a 3.0 um tube with the same Allen radius, so that k*'s window
# ends 0.8 um from the axis, k* falls four planes from the tube's axis (flagged `stack_edge`), while the three
# lines agree with each other on every node and pick a plane one or two planes (0.28-0.56 um) from the axis,
# where G_k stays within 3 % of its maximum over two planes; d_hat there is about 2.7 um, fitted at k*. On
# Allen's data, if the d-line agrees with +-3 um on the thin nodes and with +-5 um on the thick ones, one rule
# sized from d_hat covers both kinds of node with no threshold. A wrong d_hat (a neighbouring neurite inside
# the fit) gives a wrong d-line; the profile panel shows its extent against the dendrite.
# Cell 4h: the gradient energy over the whole line, on three lines, across diameters (needs Cells 0 and 1 and the pilot's CSV of Cell 4b)
import json
import os
import pandas as pd
from IPython.display import Image, display
BYDIAM = dict(dhat_bins="0.8,1.0,1.5,2.0,3.0", per_bin=2, add_nodes="2,3")   # the choice of nodes by d_hat
GRAD_LINES_UM = "3,5"   # half-lengths of the fixed lines, um
LINE_MULT = 2.0         # the d-line's full width in units of d_hat (2: the line [-d_hat, d_hat])
out = OUT + "/pilot/planegrad/bydiameter"
run("scripts/plane_differences.py", "--specimen", SPECIMEN, "--nodes", "bydiameter",
    "--pilot-csv", "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN), "--dhat-bins", BYDIAM["dhat_bins"],
    "--per-bin", BYDIAM["per_bin"], "--add-nodes", BYDIAM["add_nodes"], "--evaluation", "gradient",
    "--grad-lines-um", GRAD_LINES_UM, "--line-mult", LINE_MULT, "--planes-half", 6,
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
summ = "%s/planegrad_summary_%d.csv" % (out, SPECIMEN)
if os.path.exists(summ):
    print(pd.read_csv(summ).to_string(index=False))
p = "%s/planegrad_summary_%d.png" % (out, SPECIMEN)
if os.path.exists(p):
    display(Image(filename=p))
recs = "%s/planegrad_%d.json" % (out, SPECIMEN)
if os.path.exists(recs):
    with open(recs) as f:
        nodes = [r["node_id"] for r in json.load(f) if "skipped" not in r]
    for n in nodes:       # each node's figure, in order of d_hat; its planes drawn without any line
        p = "%s/planegrad_%d.png" % (out, n)
        if os.path.exists(p):
            print(os.path.basename(p))
            display(Image(filename=p))

# ====================================================================================================
# ===== CELL 4i
# ====================================================================================================
# CELL 4I: G ON THE D-LINE, THE ENTROPY OF THE +-5 UM STRIP AND THEIR BLEND, ACROSS DIAMETERS
# The second comparison of 2026-10-09 (16:19, D-040), changed on 2026-10-10 (D-041), on the nodes of Cell 4h:
# three ways to pick the plane. (i) G_k alone, over the whole d-line +-m d_hat / 2 (`LINE_MULT` = m = 2), as
# in Cell 4h. (ii) The strip's entropy alone, H_k: the Shannon entropy of the grey levels of the pixels within
# |v| <= `ENTROPY_HALF_UM` (5 um) across the dendrite and |u| <= `STRIPE_HALF_UM` (1 um) along it, the plane
# of its lowest value (as in Cells 4f and 4g). (iii) Their blend, in the user's form (D-041): over the planes
# where both curves are finite, each is min-max rescaled, g_k = (G_k - min G) / (max G - min G) (1 at the
# largest G) and h_k = (H_k - min H) / (max H - min H) (0 at the lowest H), and J_k = w g_k - (1 - w) h_k,
# largest where G is high and H low, with the weight of G falling with the diameter, w(d_hat) = 1 / (1 +
# exp((d_hat - d0) / s)), d0 = 1.5 um and s = 0.3 um (`SIGMOID`): w is 1/2 at d_hat = d0, 0.97 at 0.5 um and
# 0.03 at 2.5 um. The blend picks the plane of its largest J_k; a flat curve scores 0 on every plane and
# leaves the choice to the other.
#
# Why the entropy's strip is +-5 um and not the d-line (D-041): in the first run (2026-10-09, the entropy on
# the d-line, outputs kept in `pilot/planeblend/bydiameter/`) the entropy was highest in focus, and its
# minimum fell on the last plane or the one next to it on all 11 non-trunk nodes, so wherever w < 1/2 the
# blend followed it away from focus; on the trunks, where the d-line is itself about +-5 um, it agreed with
# the planes judged right. The form J_k = w g_k - (1 - w) h_k replaces w g_k + (1 - w)(1 - h_k), which differs
# from it by 1 - w, the same on every plane of a node, so that form picked the same planes.
#
# Output in `pilot/planeblend/bydiameter_strip5um/`: one figure per node, `planeblend_<id>.png`: the planes
# without any line, framed at the three picks (aqua Gd, green J, yellow Hs) and at the pipeline's planes;
# below, every plane's profile along the +-5 um line with the d-line's extent dashed, g_k and h_k with their
# picks (h drawn as it is: its pick is its lowest point), and J_k with its pick and w, on the axis -(1 - w) ..
# w that J spans. `planeblend_summary_529878215.png`: one panel per node with the three curves rescaled 0-1
# and drawn as they are (the entropy's pick at the bottom), w in each title;
# `planeblend_summary_529878215.csv`: per node d_hat, the d-line's half-length, w, the planes and each pick
# minus k* (`k_Gd`, `k_blend`, `k_Hs`), printed by the cell. The cell then shows each node's figure, in order
# of d_hat.
#
# Expect the node list of Cell 4h, one `[planediff] node N: planes ... (0 missing); d_hat ... um; G on the
# d-line +-... um (... samples), entropy on the strip +-5.0 um (... pixels); w ...; G k ..., blend k ...,
# strip entropy k ...; k* ..., dip-depth plane ..., SWC plane ...` line per node, then `[planediff] summary
# over N nodes: k_Gd within 1 plane of k* in a of N (...); k_blend ...; k_Hs ...` and `[planediff] wrote N
# figures (0 skipped) ...`; about 13 new crops per node (the +-5 um square, as in Cell 4g). A node whose
# d-line would hold fewer than 3 samples is skipped, with the reason.
#
# What to expect (synthetic tubes, SPEC Block 11): on a 0.8 um tube w = 0.90, and G_k, the +-5 um strip's
# entropy and the blend all pick the in-focus plane on every node; on a 3.0 um tube whose Allen radius is 0.3
# um, so that k* falls four planes from the axis, w = 0.02: the +-5 um entropy picks the plane next to the
# axis on every node and the blend follows it, while G_k alone picks one or two planes from the axis. On
# Allen's data the +-5 um strip's entropy fell within one plane of k* on 1 of 12 thin nodes in Cell 4g, but
# there w is about 0.9 and G_k decides; what this run adds is the nodes of 1.5-3 um, where w < 1/2 and the +-5
# um entropy decides.
# Cell 4i: G on the d-line, the entropy of the +-5 um strip and their blend, across diameters (needs Cells 0 and 1 and the pilot's CSV of Cell 4b)
import json
import os
import pandas as pd
from IPython.display import Image, display
BYDIAM = dict(dhat_bins="0.8,1.0,1.5,2.0,3.0", per_bin=2, add_nodes="2,3")   # the same nodes as Cell 4h
LINE_MULT = 2.0                        # G's line: full width LINE_MULT x d_hat, as in Cell 4h
ENTROPY_HALF_UM = 5.0                  # the entropy's strip: +-5 um across the dendrite (D-041) ...
STRIPE_HALF_UM = 1.0                   # ... and +-1 um along it (as in Cells 4f and 4g)
SIGMOID = dict(d0_um=1.5, s_um=0.3)    # the weight of G: w(d_hat) = 1 / (1 + exp((d_hat - d0) / s))
out = OUT + "/pilot/planeblend/bydiameter_strip%gum" % ENTROPY_HALF_UM
run("scripts/plane_differences.py", "--specimen", SPECIMEN, "--nodes", "bydiameter",
    "--pilot-csv", "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN), "--dhat-bins", BYDIAM["dhat_bins"],
    "--per-bin", BYDIAM["per_bin"], "--add-nodes", BYDIAM["add_nodes"], "--evaluation", "blend",
    "--line-mult", LINE_MULT, "--entropy-half-um", ENTROPY_HALF_UM, "--stripe-half-um", STRIPE_HALF_UM,
    "--sigmoid-d0-um", SIGMOID["d0_um"], "--sigmoid-s-um", SIGMOID["s_um"], "--planes-half", 6,
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
summ = "%s/planeblend_summary_%d.csv" % (out, SPECIMEN)
if os.path.exists(summ):
    print(pd.read_csv(summ).to_string(index=False))
p = "%s/planeblend_summary_%d.png" % (out, SPECIMEN)
if os.path.exists(p):
    display(Image(filename=p))
recs = "%s/planeblend_%d.json" % (out, SPECIMEN)
if os.path.exists(recs):
    with open(recs) as f:
        nodes = [r["node_id"] for r in json.load(f) if "skipped" not in r]
    for n in nodes:       # each node's figure, in order of d_hat; its planes drawn without any line
        p = "%s/planeblend_%d.png" % (out, n)
        if os.path.exists(p):
            print(os.path.basename(p))
            display(Image(filename=p))

# ====================================================================================================
# ===== CELL 4j
# ====================================================================================================
# CELL 4J: G ON LINES SIZED FROM ALLEN'S DIAMETER, THEN FROM THE FIT AT THE PLANE EACH PICKS
# The user's request of 2026-10-10 (15:30), after the runs of Cells 4h and 4i on Allen's data (Cell 4i's
# re-run of 13:20 UTC included): G_k over the whole d-line was judged the best of the plane rules, but its
# line is sized from the pilot's d_hat, which is itself fitted at k*, the plane under test. Here the first
# line comes from Allen's diameter instead, and the diameter is then refined (D-042, the option "Allen first,
# then refine"; the method carried forward, D-043). Round 1: the line +-m r_Allen through the node, across the
# dendrite (full width m times Allen's 2r; `LINE_MULT` = m = 3, the user's choice of 2026-10-12 after the run
# at m = 2, D-044), and the plane k_1 of its largest G_k = B_k^-2 * integral of (dI~_k/dv)^2 dv, with the
# background B_k of Cell 4h. Block 5's fit at plane k_1 (the pipeline's own fit with only the plane changed:
# the pilot's centre, heading, tilt and offsets, and B_bar of that plane; at k* it returns the pilot's d_hat
# to the last digit) gives d_1. Round 2 takes the line +-m d_1 / 2 and its plane k_2, the fit there gives d_2,
# and so on. The iteration stops when a plane comes up again (`converged` when it is the previous round's
# plane, `cycle` when it is an earlier one; the fit made there is reused), after `MAX_ROUNDS` = 5 rounds
# (`max_rounds`), or when a fit fails (`fit_failed`). That "the plane repeats" covers any earlier plane, and
# what the fit holds fixed, are the assistant's reading of the option, PROVISIONAL.
#
# The nodes are Cell 4h's (`--nodes bydiameter`, `BYDIAM`), so the three cells compare node by node. The
# square around each node holds round 1's line and the line sized from the pilot's d_hat at the same m, and is
# made again, larger, when a later round's line does not fit; up to about +-4.7 um it is cut from the pilot's
# cached block, beyond that it is fetched as new crops. From the fits of the run at m = 2, only the trunks'
# lines pass that limit at m = 3 (nodes 2 and 3, about +-8-9 um); theirs passed it at m = 2 too.
#
# Output in `pilot/planeiter/bydiameter_m3/` (the folder carries m, so the run at m = 2 stays in
# `pilot/planeiter/bydiameter/`): one figure per node, `planeiter_<id>.png`: the planes without any line,
# framed at the iteration's last plane (blue, it), round 1's plane (yellow, dashed, A), the pick of the line
# sized from the pilot's d_hat at the same m (aqua, Gd; at m = 2 it is Cell 4h's d-line) and the pipeline's
# planes; below, every plane's profile along the longest line used, with round 1's, the last round's and the
# pilot d-line's extents dashed; each round's G_k relative to its own maximum with its pick (round 1 yellow,
# the later rounds blue, darker = later; the pilot d-line aqua); and the trajectory: the diameter sizing each
# line (Allen's 2r, then the fit at each round's plane, labelled with the plane; hollow: a plane picked again)
# against the pilot's d_hat (red, dashed). `planeiter_summary_529878215.png`: one panel per node with round
# 1's curve, the last round's and the pilot d-line's, rescaled 0-1, each pick minus k* and the iteration's
# ending in the title; `planeiter_summary_529878215.csv`: per node the type, Allen's 2r, d_hat, k*, the three
# planes (`k_allen`, `k_iter`, `k_Gd`) and each minus k*, `d_iter_um` (the fit at the last plane),
# `iter_status`, `n_rounds`, `n_restacks` and the trajectory, which the cell prints one line per node. The
# cell then shows each node's figure, in order of d_hat.
#
# Expect the node list of Cell 4h, one `[planediff] node N: planes ... (0 missing); Allen 2r ... um; start d
# ... um | r1 +-... um k ... d ... um | r2 ... | converged; the pilot's d-line +-... um k ...; k* ...,
# dip-depth plane ..., SWC plane ...` line per node, then `[planediff] summary over N nodes: k_allen within 1
# plane of k* in a of N (...); k_iter ...; k_Gd ...`, `[planediff] iteration endings: converged ...` and
# `[planediff] wrote N figures (0 skipped) ...`. A node without Allen's radius is skipped, with the reason.
#
# What to expect (synthetic tubes, SPEC Block 11, at m = 3): on a 0.8 um tube with Allen's radius 0.3 um,
# round 1's line (+-0.9 um) already reaches the tube's edges: every node picks the in-focus plane in round 1,
# the fit there is the pilot's d_hat (0.84-0.86 um), and round 2 picks the same plane (converged in 2 rounds).
# On a 3.0 um tube with the same Allen radius, round 1's line (+-0.9 um) lies inside the tube and picks the
# end plane of the range (-6), where the blurred profile fits 2.7-2.8 um; round 2's line (+-4.1-4.2 um) picks
# a plane one or two from the axis (-1 or -2), the fit there gives 3.0-3.2 um, and round 3 picks it again
# (converged in 3 rounds on every node, the square made again twice): the iteration recovers from an Allen
# diameter five times too small and ends on the plane the pilot's d-line picks, while k* stays four planes
# from the axis. On both tubes m = 3 ends on the plane of m = 2 with the same diameter; only the lines are
# longer. On Allen's data (the run at m = 2) Allen's 2r is 0.45-0.76 um on the 11 nodes of the bins and
# 2.0-2.2 um on the trunks, so round 1 tests a start that knows little of the fit: on the thick nodes its line
# lies inside the dendrite, as on the 3.0 um tube. What to look at, node by node: whether `k_iter` and
# `d_iter_um` move from the run at m = 2 (`pilot/planeiter/bydiameter/`: all 13 nodes converged, `k_iter`
# equal to `k_Gd` on all 13 and within one plane of k* on 10 of the 11 nodes of the bins); whether `k_iter`
# still agrees with `k_Gd`, now both at m = 3; and the trunks (nodes 2 and 3): at m = 2 they ended on 113 and
# 115, beside the planes 114 and 115 where the +-5 um entropy was judged right (D-039), node 2 on the range's
# end plane (113 of 113..125) with d at the fit's 6 um bound; the longer lines may now reach neighbouring
# neurites.
# Cell 4j: G on lines sized from Allen's diameter, then from the fit at the plane each picks (needs Cells 0 and 1 and the pilot's CSV of Cell 4b)
import json
import os
import pandas as pd
from IPython.display import Image, display
BYDIAM = dict(dhat_bins="0.8,1.0,1.5,2.0,3.0", per_bin=2, add_nodes="2,3")   # the same nodes as Cells 4h and 4i
LINE_MULT = 3.0     # each line's full width in units of the diameter sizing it (3: +-1.5 d, D-044; round 1: Allen's 2r)
MAX_ROUNDS = 5      # at most this many rounds, round 1 included; the iteration stops earlier when a plane repeats
out = OUT + "/pilot/planeiter/bydiameter_m%g" % LINE_MULT    # bydiameter_m3; the run at m = 2 stays in bydiameter
run("scripts/plane_differences.py", "--specimen", SPECIMEN, "--nodes", "bydiameter",
    "--pilot-csv", "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN), "--dhat-bins", BYDIAM["dhat_bins"],
    "--per-bin", BYDIAM["per_bin"], "--add-nodes", BYDIAM["add_nodes"], "--evaluation", "iterate",
    "--line-mult", LINE_MULT, "--max-rounds", MAX_ROUNDS, "--planes-half", 6,
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
summ = "%s/planeiter_summary_%d.csv" % (out, SPECIMEN)
if os.path.exists(summ):
    tab = pd.read_csv(summ)
    print(tab.drop(columns=["trajectory"]).to_string(index=False))
    for n, t in zip(tab["node_id"], tab["trajectory"]):    # per node: each round's line, its plane, the fit there
        print("node %d: %s" % (n, t))
p = "%s/planeiter_summary_%d.png" % (out, SPECIMEN)
if os.path.exists(p):
    display(Image(filename=p))
recs = "%s/planeiter_%d.json" % (out, SPECIMEN)
if os.path.exists(recs):
    with open(recs) as f:
        nodes = [r["node_id"] for r in json.load(f) if "skipped" not in r]
    for n in nodes:       # each node's figure, in order of d_hat; its planes drawn without any line
        p = "%s/planeiter_%d.png" % (out, n)
        if os.path.exists(p):
            print(os.path.basename(p))
            display(Image(filename=p))

# ====================================================================================================
# ===== CELL 4k
# ====================================================================================================
# CELL 4K: THE PLANE EACH NODE'S ITERATION CHOSE, WITH THE LINE THAT CHOSE IT AND THE DIAMETER FITTED THERE
# The user's request of 2026-10-12 (17:58): for each node, the plane the Allen-guided iteration chose (k_iter,
# Cell 4j), drawn with the line that chose it and the diameter fitted there (D-045). The line is the one of
# the last round that picked a plane -- at convergence the repeat round, sized from d_iter itself: +-m d_iter
# / 2 through the SWC node, across the pilot's heading (m = 3, D-044). The diameter is Block 5's fit redone at
# k_iter; its two edges, v0_hat +- d_iter / 2 from the pilot's fitted centre, are drawn on that line as a
# caliper (blue). The fit's own line passes through the fitted centre, parallel to the drawn one; how far
# apart the two lie along the branch is printed per node (`u_um`).
#
# Nothing is measured anew: the cell reads Cell 4j's `planeiter_529878215.json` in `ITER_DIR`, rebuilds each
# node's last square from the cached crops, and redoes the fit at k_iter only to draw it. The fit must give
# Cell 4j's `d_iter_um` to the last digit and the square Cell 4j's half-width; a difference is printed in the
# node's notes, not hidden. Set `ITER_DIR` to `OUT + "/pilot/planeiter/bydiameter"` to draw the run at m = 2.
#
# Output in `ITER_DIR/fits/`: `planefit_overview_529878215.png`, every node's plane with the line (black) and
# the caliper (blue), in Cell 4j's order; one figure per node, `planefit_<id>.png`: on the left the plane
# k_iter on its own grey scale with the line, the caliper, the SWC node (+) and a scale bar; on the right,
# against the position v along the line (0 at the SWC node), the plane's profile along the line (grey), the
# fit's profile (black dots: 9 parallel lines averaged along the branch through the fitted centre, over the
# fit's +-3 um window, shaded), the fitted model (blue), B_bar (dashed) and the edges (blue verticals, d
# between them); `planefit_529878215.json`: per node k_iter, k*, the fit (`d_refit_um`, `fit_status`,
# `v0_hat_um`), `s_um` and `u_um`, the caliper's ends (`edges_v_um` along the line, `edges_xy_um` in the
# image), the line (`line_xy_um`, `sel_round`, `sel_half_um`) and the notes. The cell prints one line per node
# and shows the overview, then each node's figure.
#
# Expect one `[planefit] node N: k_iter K (k* K*); line +-H um (round R); d D um, edges at v -E and +E um from
# the SWC node` line per node, then `[planefit] 13 nodes drawn, 0 skipped; the fit redone at k_iter equals
# Cell 4j's d_iter on 13 of 13` and `[planefit] files in ...; crops: ... from the cache, 0 downloaded`: the
# crops are those Cell 4j fetched.
#
# What to expect (synthetic tubes, SPEC Block 11): on a 0.8 um tube the caliper lies on the tube, its middle
# within 0.004 um of the SWC node (the tube's axis passes through it), with d 0.84-0.86 um; on a 3.0 um tube d
# is 3.03-3.24 um and the profile panel shows B_bar 4 to 12 grey levels under the profile's plateau, so the
# model cannot reach it: B_bar comes from the block's pixels outside Allen's radius + 1 um (0.3 + 1 um), which
# on this tube still holds its edges. What to look at, node by node: whether the caliper spans the dendrite as
# you see it in the plane; on the trunks (nodes 2 and 3), whether the edges lie inside the fit's +-3 um window
# (at d = 6 um they reach its ends) and where B_bar lies against the plateau.
# Cell 4k: the plane each node's iteration chose, with the line that chose it and the diameter fitted there (needs Cells 0 and 1 and Cell 4j's output)
import json
import os
from IPython.display import Image, display
ITER_DIR = OUT + "/pilot/planeiter/bydiameter_m3"   # Cell 4j's run at m = 3 (D-044); ".../bydiameter" for its run at m = 2
out = ITER_DIR + "/fits"
run("scripts/iteration_fits.py", "--specimen", SPECIMEN, "--iter-json", "%s/planeiter_%d.json" % (ITER_DIR, SPECIMEN),
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
p = "%s/planefit_overview_%d.png" % (out, SPECIMEN)
if os.path.exists(p):
    display(Image(filename=p))
recs = "%s/planefit_%d.json" % (out, SPECIMEN)
if os.path.exists(recs):
    with open(recs) as f:
        drawn = [r for r in json.load(f) if "skipped" not in r]
    for r in drawn:       # each node's figure, in Cell 4j's order (the pilot's d_hat)
        print("node %d: k_iter %d (k* %d), d %.2f um, %s%s" % (r["node_id"], r["k_iter"], r["k_star"], r["d_refit_um"],
                                                             r["fit_status"], "".join("; " + n for n in r["notes"])))
        display(Image(filename=r["png"]))

# ====================================================================================================
# ===== CELL 4l
# ====================================================================================================
# CELL 4L: FIT VARIANTS AT THE PLANE EACH NODE'S ITERATION CHOSE: OTHER BLUR KERNELS AND CENTRE WEIGHTS
# The user's request of 2026-10-13 (14:50), on seeing Cell 4k: the thin dendrites look hugely overestimated,
# so try a fit whose weights fall exponentially from the dip's centre, and other families of the blur kernel,
# with and without the weights (D-046). A diagnostic: nothing in the measurement changes, Block 5's fit keeps
# its Gaussian (D-026). At each node's plane k_iter (Cell 4j, not re-chosen) the profile Block 5 fitted there
# is fitted again 100 ways: 10 kernels (Gaussian, Lorentzian, Voigt r = 1, Student nu = 3, generalized normal
# beta = 1 and 4, and the Voigt, Student and generalized normal with their shape fitted, and a skew-normal
# with its skew fitted), each at the width of Block 5's Gaussian (FWHM 0.233 um) and with the width fitted
# (0.1-2 um), each unweighted and with the weights exp(-|v - v0| / lambda), lambda = 0.25, 0.5, 1, 2 um. The
# weights follow the fit's own centre v0: fit, re-centre the weights on the new v0, fit again, until v0 and d
# move by less than 1e-4 um (at most 30 rounds). Every kernel has unit area and its peak at v0; its width is
# its FWHM, so a kernel and the Gaussian compared at the same width have the same half-maximum width.
#
# Nothing is fetched: the cell reads Cell 4j's `planeiter_529878215.json` in `ITER_DIR` and rebuilds each
# node's square from the cached crops. The Gaussian at the fixed width, unweighted, is Block 5's fit itself
# and must give Cell 4j's `d_iter_um`; a difference is printed in the node's notes. To try fewer variants, add
# for instance `"--families", "lorentz,gennorm"`, `"--lambdas", "0.5"` or `"--width-modes", "fixed"` to the
# `run(...)` call (the Gaussian is always kept); `"--nodes", "8497,3856"` for some nodes only.
#
# Output in `ITER_DIR/fitvar/`: `fitvar_summary_529878215.png`, for every kernel and width mode d / d_base
# against d_base (Block 5's d), one series per weighting, hollow where a fit is flagged; one figure per node,
# `fitvar_<id>.png`: three columns of the profile at k_iter against v (um from the fitted centre) with the
# fits: every kernel unweighted at the fixed width, every kernel unweighted at a fitted width, the Gaussian at
# the fixed width unweighted and for each lambda (with the weights drawn below); above each column the fits'
# edges v0 +- d/2 as stacked calipers with their d; below, d of all 100 variants as a table coloured by d /
# d_base (blue: smaller than Block 5's), flagged * at a bound, x failed, ~ weights not settled in 30 rounds;
# `fitvar_529878215.csv`, one row per node and variant (`d_um`, `fwhm_um`, `shape`, `v0_um`, `status`,
# `ratio`, the residual sums with and without the weights, ...) and `fitvar_529878215.json` with the profiles.
# The cell shows the summary, a table of eight variants' d per node, then each node's figure.
#
# Expect one `[fitvar] node N: k_iter K (k* K*); Block 5 d D um (converged); Gaussian centre-weighted d ...
# (lambda 0.25/0.5/1/2 um); unweighted kernels: fixed width d A-B, fitted width d C-E, the Gaussian's fitted
# FWHM W um; smallest d ...; 100 variants, F flagged; T s` line per node, then `[fitvar] 13 nodes fitted, 0
# skipped; 100 variants per node; the baseline equals Cell 4j's d_iter on 13 of 13` and `[fitvar] files in
# ...; crops: ... from the cache, 0 downloaded`. It takes 25-75 s per node on one CPU of the sandbox (the
# longer on thick dendrites), some 10-15 minutes for the 13 nodes.
#
# What to expect (synthetic profiles, SPEC Block 11; one noise draw each, so indicative): when the true blur
# is a Gaussian of FWHM 0.5 um, twice Block 5's, Block 5 gives 0.82 um for a 0.6 um tube and 1.47 um for a 1.4
# um one; the Gaussian of fitted width gives 0.70 and 1.39 um (its FWHM 0.43 and 0.50 um), the heavy-tailed
# kernels at the fixed width (Lorentzian, generalized normal beta = 1) 0.67-0.69 um on the thin tube, and the
# weights change d by a few hundredths on a clean single dip; the 3.0 um tube is near 3.0 um for every
# variant. Shapes often run to a bound (a Student nu of 100 or a Voigt r of 0.01 is the Gaussian again; a skew
# of 10 at the fixed width widens the kernel on one side): the flag says so. Where a darker neighbour merges
# with the dip, the unweighted fit takes both as one tube and the re-centred weights at lambda = 0.25 um can
# move onto the neighbour: check v0 in the calipers. What to look at: on the thin nodes, which variants bring
# d down and whether their curves still follow the profile; whether the fitted widths agree across nodes (a
# common width would point at the blur rather than at the dendrites); on the trunks, whether the variants
# agree.
# Cell 4l: fit variants at the plane each node's iteration chose: other blur kernels and centre weights (needs Cells 0 and 1 and Cell 4j's output; 25-75 s per node)
import json
import os
import pandas as pd
from IPython.display import Image, display
ITER_DIR = OUT + "/pilot/planeiter/bydiameter_m3"   # Cell 4j's run at m = 3 (D-044)
out = ITER_DIR + "/fitvar"
run("scripts/fit_variants.py", "--specimen", SPECIMEN, "--iter-json", "%s/planeiter_%d.json" % (ITER_DIR, SPECIMEN),
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
p = "%s/fitvar_summary_%d.png" % (out, SPECIMEN)
if os.path.exists(p):
    display(Image(filename=p))
p = "%s/fitvar_%d.csv" % (out, SPECIMEN)
if os.path.exists(p):           # d (um) of a few variants per node, in Cell 4j's order; the CSV holds all 100
    t = pd.read_csv(p)
    keys = ["gauss|w fixed|none", "gauss|w fixed|lam=0.25", "gauss|w fixed|lam=0.5", "gauss|w fixed|lam=1",
            "gauss|w fixed|lam=2", "gauss|w fitted|none", "lorentz|w fixed|none", "gennorm beta=1|w fixed|none"]
    keys = [k for k in keys if k in set(t["variant"])]
    tab = t[t["variant"].isin(keys)].pivot(index="node_id", columns="variant", values="d_um")
    display(tab.loc[list(dict.fromkeys(t["node_id"])), keys].round(2))
p = "%s/fitvar_%d.json" % (out, SPECIMEN)
if os.path.exists(p):
    with open(p) as f:
        done = [r for r in json.load(f) if "skipped" not in r]
    for r in done:       # each node's figure, in Cell 4j's order (the pilot's d_hat)
        print("node %d: k_iter %d (k* %d), Block 5's d %.2f um%s" % (r["node_id"], r["k_iter"], r["k_star"],
                                                                   r["d_base_um"], "".join("; " + n for n in r["notes"])))
        display(Image(filename=r["png"]))

# ====================================================================================================
# ===== CELL 4m
# ====================================================================================================
# CELL 4M: EACH FIT VARIANT'S OWN ALLEN-GUIDED ITERATION, ITS EDGES AND ITS PLANE
# The user's request of 2026-10-14 (15:35), on the figures of Cell 4l, and the answers to three questions
# (D-047): one table per node of the variants at the fitted width -- every kernel, unweighted and
# centre-weighted, as their edges v0 +- d/2 (a 50-row stack); each of the 50 runs its own Allen-guided
# iteration, the way Cell 4j searches the plane and sizes the measuring line, with Cell 4j's `LINE_MULT` and
# `MAX_ROUNDS` inherited (read from its records, so the two never disagree); and the figure of the fitted
# dendrite at each variant's own plane with its d. For a variant V: round 1's line is +-m r_A (from Allen's 2
# r_A), as in Cell 4j; each later line is +-m d / 2 with d V's own fit at the plane the previous line picked
# (Block 5's inputs there: the profile through the pilot's fitted centre, the tilt, that plane's B_bar); the
# plane is the one of the largest gradient energy over the line; the iteration stops when a plane repeats or
# after `MAX_ROUNDS`. Cell 4j's own iteration (Block 5's fit) is redone first with the same machinery, as the
# reference.
#
# Nothing is measured anew: the cell reads Cell 4j's `planeiter_529878215.json` in `ITER_DIR` and rebuilds
# each node's squares as Cell 4j made them, from the cached crops. A variant whose d is larger than Block 5's
# can need a longer line than Cell 4j ever drew, and so a larger square: on the trunks some new crops may be
# fetched. The reference must give Cell 4j's planes and `d_iter_um` to the last digit; a difference is printed
# in the node's notes. To try fewer variants, add for instance `"--families", "lorentz,gennorm"` or
# `"--lambdas", "0.5"` to the `run(...)` call (the Gaussian is always kept); `"--nodes", "8497,3856"` for some
# nodes only; `"--width-modes", "fixed,fitted"` adds the fixed-width variants (100 then).
#
# Output in `ITER_DIR/variter/`: per node `variter_stack_<id>.png`, one row per variant (Cell 4j's own first,
# in ink): its label and d, its edges v0 +- d/2 (um from the pilot's fitted centre) at the plane its own
# iteration chose, and the planes its rounds picked (grey) ending on that plane (filled), against k* (red) and
# the SWC plane (dotted); and `variter_planes_<id>.png`, each variant's chosen plane with the line that chose
# it (black, through the SWC node +) and the fitted dendrite's two edges drawn along the branch (the kernel's
# colour), titled with the plane and d: Cell 4j's own on top, then one row per kernel and one column per
# weighting, every panel on the same view; `variter_529878215.csv` (one row per node and variant: `k_iter`,
# `d_iter_um`, `iter_status`, `planes`, the fit's `v0_um`, `fwhm_um`, `shape`, `ratio_ref` = d / the
# reference's, ...) and `variter_529878215.json` (with each variant's trajectory, edges and line). The cell
# shows a table of d per node for Cell 4j's own and each kernel unweighted, then each node's two figures.
#
# Expect one `[variter] node N: Cell 4j redone k_iter K d D um (converged); 50 variants: k_iter K1 xn1, ...;
# d_iter A-B um; unweighted ...; P planes fitted, S squares; T s` line per node, then `[variter] 13 nodes, 0
# skipped; 50 variants per node, each with its own iteration; Cell 4j's iteration redone identically on 13 of
# 13` and `[variter] files in ...; crops: ...`. The fits at a plane are shared by every variant that visits
# it, so the time grows with the number of planes visited: about 20 s per plane on a thin node and 1 minute on
# a thick one (one CPU of the sandbox); expect some 10-40 minutes for the 13 nodes.
#
# What to expect (synthetic tubes, SPEC Block 11): there, every variant's iteration ends on Cell 4j's plane
# (0.8 um tube: d 0.81-0.86 um; 3.0 um tube: d 3.20-3.35 um), and the reference redoes Cell 4j exactly. On
# Allen's data the variants whose fitted width takes up most of the dip (Cell 4l: nodes 8481, 8482 and 3851
# gave d of 0.2-0.6 um with widths of 1-2 um) size short lines, which may pick other planes. What to look at:
# in the stack, the rows whose filled plane is not Cell 4j's; in the images, whether each variant's edges
# follow the dendrite at the plane it chose.
# Cell 4m: each fit variant's own Allen-guided iteration, its edges and its plane (needs Cells 0 and 1 and Cell 4j's output)
import json
import os
import pandas as pd
from IPython.display import Image, display
ITER_DIR = OUT + "/pilot/planeiter/bydiameter_m3"   # Cell 4j's run at m = 3: its LINE_MULT and MAX_ROUNDS are inherited from its records
out = ITER_DIR + "/variter"
run("scripts/variant_iteration.py", "--specimen", SPECIMEN, "--iter-json", "%s/planeiter_%d.json" % (ITER_DIR, SPECIMEN),
    "--cache-dir", CACHE_DIR, "--out-dir", out, *ALIGN)
p = "%s/variter_%d.csv" % (out, SPECIMEN)
if os.path.exists(p):           # d_iter (um) per node: Cell 4j's own (Block 5's fit) and each kernel unweighted
    t = pd.read_csv(p)
    u = t[t["weight"] == "none"]
    tab = u.pivot(index="node_id", columns="kernel", values="d_iter_um")
    display(tab.loc[list(dict.fromkeys(t["node_id"])), list(dict.fromkeys(u["kernel"]))].round(2))
p = "%s/variter_%d.json" % (out, SPECIMEN)
if os.path.exists(p):
    with open(p) as f:
        done = [r for r in json.load(f) if "skipped" not in r]
    for r in done:       # each node's edge stack and planes, in Cell 4j's order (the pilot's d_hat)
        d = r["d_iter_cell4j_um"]
        print("node %d: Cell 4j's k_iter %s, d_iter %s um%s" % (r["node_id"], r["k_iter_cell4j"],
                                                              "n/a" if d is None else "%.2f" % d,
                                                              "".join("; " + n for n in r["notes"])))
        display(Image(filename=r["png_stack"]))
        display(Image(filename=r["png_planes"]))

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
# CELL 7: KERNEL CALIBRATION -- NEAR FIELD (BLOCK 10), FAR FIELD (BLOCK 15), FIGURES
# The calibration nodes of the pilot are chosen by Allen's diameter D_A = 2 r_A, not by the fitted d_hat (D-061.4,
# D-067.4; thresholds PROVISIONAL): the near-field set, D_A <= 0.6 um, is scanned plane by plane over k* +- 3
# (offsets up to 0.84 um); the far-field set, 0.6 < D_A <= 0.8 um, over k* +- 16 (offsets to 4.5 um, D-026,
# D-078). Both must be in S and flat by the pilot's columns -- phi <= 10 deg for the near set, <= 15 deg for the
# far set since 2026-10-28 (D-101.1; was 5 deg for both, D-081.3) -- since D-094 with no optical-depth cut and
# the flag faint (a dip under 6 gl) not counted (`calibration.node_alpha_check` restores alpha <= 0.5,
# `calibration.node_faint_flag_check` the flag) -- and then, on the trace, usable: since D-101 (2026-10-28) the
# background of each plane is fitted on the STRIPE along the measuring line -- the band (half-width 1-12 um along
# the dendrite, the one with most pixels) walked outward on each side of the node to the first pixel column a
# traced neighbour's predicted shadow reaches above 0.3 gl (D-108, 2026-10-29; 0.03 gl before), the node's own
# fitted cylinder removed -- and a
# node is `not_isolated:too_few` / `not_isolated:side_short` when the stripe at k* has fewer than 500 pixels or
# under 3 um of clear run on a side (the old square rule, D-078.1 / D-081.1, is `calibration.background_support`
# = "square"); not screened for straightness since D-093 (the calibration evaluates only the measuring line;
# `calibration.node_straight_check` restores it at 2 radii). Each scan reads a +-25 um patch per plane, fits a
# planar background to its stripe and fits the dip's Gaussian core iteratively (up to 10 passes, stop at 1 SE of
# the width; D-077, D-078.2). `real` prints `N candidate nodes`, one line per node (`scanned (near field)`,
# `scanned (far field)` or `not a calibration node (...)` with the reasons), `... candidates scanned: n near-
# field, m far-field; reasons {...}` and the example nodes whose patch images are kept (`patches -> ...`).
# `fit` (near field): `growth fit (gaussian_core_width2, symmetric, cubic): N nodes`, the knot table, the
# residual RMS by |k - k*|, the table cut at 0.84 um (D-080.1) and `anchor sigma_0_hat = ... um, SE ... um
# (node bootstrap, ...)` (D-081.7). `far-fit` (far field, needs the anchor): `far planes: J nodes kept ...`,
# `far fit: gamma_hat = ... +- ... (bootstrap ..., anchor term ...), t_{J-1} interval [...]`, the per-sign and
# the quadratic comparison (gamma_2, labelled comparison, D-081.6), and `production kernel: table to 0.84 um,
# continuation linear with slope ...` -- the value for `renderer.kernel_continuation_slope` in Cell 8, yours to
# decide. `figures` writes `width_by_plane_near.png` / `_far.png` (every node light grey, the median in colour,
# the fitted law), `example_near.png` / `_far.png` (the node nearest the median with its plane images) and
# `kernel_law.png` into OUT/calibration/figures (D-080.2, D-081.5): judge there whether the straight line
# holds before you set the slope. Fewer than two near-field nodes stop `fit`; fewer than two far-field nodes
# with 6 kept planes each stop `far-fit`: measure more stretches in Cell 4b. With the 0.3 grey-level isolation
# rule the sets may be small on a dense cell: the `reasons {...}` counts say which rule removed the nodes. A
# failure here does not stop the notebook.
# [2026-10-26, D-088.3, Block 19] Each set holds at most 30 scanned nodes (the candidates taken in a seeded order;
# the rest `set_full`, not measured) and needs at least 10: below, `fit` prints `INSUFFICIENT: J near-field nodes
# < 10 (D-088.3) ...`, `far-fit` writes no production kernel (`no production kernel: insufficient ...`) and the fits
# are diagnostics. One kernel for every cell (D-088.1): this cell is the test on the pilot cell (D-088.2); Cell 15
# pools the nodes across the survey's cells.
#
# The first-stage table is not a calibrated kernel on its own: on rendered thin phantoms its growth is about
# 0.01 um^2 low within two planes (SPEC Block 10). Keep the default kernel for the production table until you
# have read the width-by-plane figures and the far fit.
# [2026-10-28, D-098.4] As `real` evaluates each candidate it draws the patch its background is fitted on (+-25 um,
# plane k*): the traced segments -- the node's own branch blue, a neighbour orange when its predicted shadow in the
# patch exceeds 0.3 gl in some plane (they bound the stripe), grey below -- and (b) the stripe the background is
# fitted on with its width, runs and pixel count (D-101), into OUT/calibration/figures/isolation/<set>_node_<id>.png
# (orange thumbnails in the overview: the stripe at k* is not usable), with overview pages
# isolation_overview_<n>.png (shown below); `real` ends with `isolation figures: N candidates (k whose stripe at k* is
# not usable: fewer than 500 px or a side under 3.0 um) -> ...`. One crop more per candidate that is not scanned.
#
# [2026-10-28, later] The stripe's knobs: the code starts with `CAL_KNOBS`, the stripe's parameters at the
# configuration's defaults -- `background_support`, tau (`isolation_depth_gl`), the W grid, the minimum pixel count
# and side run, the patch's half-side -- and the two flatness bounds; the cell writes them to
# `OUT/calibration/calibration_config.json`, passes the file to `real`, `fit`, `far-fit` and `figures`, and `real`
# prints `calibration knobs: ...` with the values it ran with. Each isolation figure has a panel (c): every plane's
# stripe -- its clear pixels, and its runs along the measuring line beyond the own cylinder -- dark where usable,
# orange where not, faint for the planes the set's fit does not use (the far fit uses |k - k*| >= 5; the dip's area
# |k - k*| = 4-6, half-faint). The overview's labels end with n/N, the fit's planes whose stripe is usable, and `real`
# adds `far-set candidates with a usable stripe in at least 6 of the far fit's planes (|k - k*| >= 5): m of n`.
#
# [2026-10-29, D-108, D-109] tau is 0.3 gl (the stripe kept 21 of the pilot's 138 near candidates at 0.03 gl, 92 at
# 0.3 gl, in the sandbox's run on the pilot's own reconstruction). The far set is no longer the pilot table's (Cell
# 4b's stretches hold 31 far-range nodes, on two crowded stretches): before fetching anything `real` judges every
# dendrite node of the cell with Allen's 2r in (0.6, 0.8] um from the trace -- the trace's tilt (<= 15 deg), the
# stripe at the trace's plane, and at least 6 usable stripes among the far fit's 24 planes (|k - k*| >= 5), with the
# pilot table's median background and absorption -- keeps the passing nodes at least 5 um apart along their stretch,
# and then measures and scans them like any candidate. It prints `candidate search: the cell's typical background ...`
# (once, for both searches), a progress line every 500 nodes, and `far search over the cell: N far-range nodes; a flat
# by the trace (phi <= 15 deg); b with a usable stripe at the trace's plane; c with at least 6 usable far-fit planes;
# m chosen, at least 5 um apart (t min) -> .../far_search_529878215.csv` (one row per far-range node and why it
# failed). On the pilot, in the sandbox, the search took 5.1 min over its 723 far-range nodes (502 flat by the trace,
# 90 usable at the trace's plane, 74 with at least 6 usable far-fit planes, 19 chosen), before the first fetch; Colab
# should take about as long, then the 19 are measured and scanned (about 46 crops each). `calibration.far_candidates =
# "table"` restores the table's far candidates.
#
# [2026-10-29, D-112] The near set is searched over the whole cell as well (the user: "the whole cell's nodes must be
# evaluated for both near and far field calibration"): every dendrite node with Allen's 2r <= 0.6 um, judged from the
# trace -- the tilt (<= 10 deg), the stripe at the trace's plane, and at least 3 usable stripes among the trace's
# plane +- 3 (the pairs +-1, +-2, +-3 in turn, stopping once 3 are usable) -- kept at least 5 um apart along its
# stretch (the most clear pixels at the trace's plane first), then measured and scanned like any candidate, at most 30
# scanned per set (the others counted in one line, `not measured, their set already holding 30 scans (set_full):
# ...`). Before the far search it prints `near search: n of N near-range nodes judged (t min)` every 500 nodes and
# `near search over the cell: N near-range nodes; a flat by the trace (phi <= 10 deg); b with a usable stripe at the
# trace's plane; c with at least 3 usable planes of the trace's plane +- 3; m chosen, at least 5 um apart (t min) ->
# .../near_search_529878215.csv`. On the pilot, in the sandbox: 7500 near-range nodes, 3856 flat by the trace, 1960
# usable at the trace's plane, 1947 with at least 3 usable planes, 380 chosen on 48 stretches, in 15.4 min; with the
# far search (5.1 min) Cell 7 spends about 20 min before its first download. `calibration.near_candidates = "table"`
# restores the table's near candidates.

# The stripe's knobs (D-101, D-108) and the candidates' search over the whole cell (D-109, D-112): the configuration's
# calibration.* fields, here at their defaults. Change one and run the cell again: the file below goes to real, fit,
# far-fit and figures, and real prints `calibration knobs: ...` with the values it ran with. Cell 15 and the davinci
# array read the same file when it is passed (runbook, "the stripe's knobs").
CAL_KNOBS = {
    "calibration.background_support": "stripe",   # "stripe" (D-101) or "square" (before 2026-10-28: the 3-sigma far mask)
    "calibration.isolation_depth_gl": 0.3,        # tau (gl): a pixel is shaded where a traced segment's predicted shadow exceeds it (D-108)
    "calibration.stripe_half_widths_um": [1.0, 2.0, 4.0, 8.0, 12.0],   # the band's half-widths W tried (um); most clear pixels wins
    "calibration.stripe_min_pixels": 500,         # a plane's stripe is usable with at least this many clear pixels ...
    "calibration.stripe_min_side_um": 3.0,        # ... and at least this clear run (um) on each side of the own cylinder
    "calibration.patch_half_um": 25.0,            # the patch's half-side (um): the runs end at its edge (larger: bigger crops)
    "calibration.node_phi_max_deg": 10.0,         # flatness, near set (deg)
    "calibration.node_phi_max_deg_far": 15.0,     # flatness, far set (deg)
    "calibration.near_candidates": "cell",        # near set: "cell" found over the whole cell from the trace (D-112), "table"
    "calibration.near_search_spacing_um": 5.0,    # the near candidates kept at least this far apart along their stretch (um)
    "calibration.far_candidates": "cell",         # far set: "cell" found over the whole cell from the trace (D-109), "table"
    "calibration.far_search_spacing_um": 5.0,     # the far candidates kept at least this far apart along their stretch (um)
}
import json
assert all(k.startswith("calibration.") for k in CAL_KNOBS), "CAL_KNOBS: calibration.* fields only"
os.makedirs(OUT + "/calibration/figures", exist_ok=True)
CFG_CAL = OUT + "/calibration/calibration_config.json"
with open(CFG_CAL, "w") as f:
    json.dump({"calibration": {k.split(".", 1)[1]: v for k, v in CAL_KNOBS.items()}}, f, indent=1)
CAL_CFG = ["--config-json", CFG_CAL]
SCANS = "%s/calibration/scans_%d.json" % (OUT, SPECIMEN)
CALIB = "%s/calibration/calibration_%d.json" % (OUT, SPECIMEN)
FARFIT = "%s/calibration/far_fit_%d.json" % (OUT, SPECIMEN)
PATCHES = "%s/calibration/patches_%d.npz" % (OUT, SPECIMEN)
rc = run("scripts/calibrate_kernel.py", "real", "--specimen", SPECIMEN, "--nodes-csv",
         "%s/pilot/pilot_%d.csv" % (OUT, SPECIMEN), "--cache-dir", CACHE_DIR, "--out", SCANS,
         "--registration-json", REG_JSON, *ALIGN, *CAL_CFG, check=False)
if rc == 0:
    rc = run("scripts/calibrate_kernel.py", "fit", "--scans", SCANS, "--out", CALIB, *CAL_CFG, check=False)
if rc == 0:
    run("scripts/calibrate_kernel.py", "far-fit", "--scans", SCANS, "--calibration", CALIB, "--out", FARFIT, *CAL_CFG,
        check=False)
    fig_args = ["--scans", SCANS, "--calibration", CALIB, "--out-dir", OUT + "/calibration/figures"] + CAL_CFG
    if os.path.exists(FARFIT):
        fig_args += ["--far-fit", FARFIT]
    if os.path.exists(PATCHES):
        fig_args += ["--patches", PATCHES]
    run("scripts/calibrate_kernel.py", "figures", *fig_args, check=False)
import glob
from IPython.display import Image, display
for path in sorted(glob.glob(OUT + "/calibration/figures/isolation/isolation_overview_*.png")):   # D-098.4
    display(Image(filename=path))

# ====================================================================================================
# ===== CELL 8
# ====================================================================================================
# CELL 8: THE PRODUCTION CONFIGURATION (YOUR DECISIONS)
# The cell prints what Cells 3-6 suggest next to the current defaults. Uncomment and set, in `DECIDED`, only
# what you have decided; record each choice as a decision before the table is built on it. `with_overrides`
# checks every name and value, and turns JSON lists into the tuples the configuration holds.
#
# It writes one configuration, `config_production.json` (sigma_fit = 0.099 um; the sigma_fit study of D-023 is
# dropped, D-057 (ii)). Expect one line `config_production.json ... -> table bias_table_<hash>`: the hash names
# the table, and a real fit refuses a table with another hash (procedure s.3.11).
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
with open("%s/config_production.json" % OUT, "w") as f:      # one configuration, sigma_fit 0.099 um (D-057 ii)
    f.write(cfg.to_json())
print("config_production.json   sigma_fit %.3f um -> table bias_table_%s"
      % (cfg.measure.sigma_fit_um, cfg.signature_hash("estimator")))

# ====================================================================================================
# ===== CELL 9: DAVINCI, NOT COLAB (nothing to paste here)
# ====================================================================================================
# CELL 9: THE PRODUCTION TABLE ON DAVINCI (NOT COLAB)
# `scripts/pbs/build_table.pbs`: an array of tasks, each rendering a chunk of draws with every realization (Block
# 14.2: G draws x phantom.n_realizations), one core and 4 GB each; the array's size and walltime come from the
# davinci pilot (S6). Then the merge on the login node.
#
# 1. Copy `MyDrive/diameters/config_production.json` to davinci into a
# folder whose path has no spaces, e.g. `~/diam/`. `qsub -v` takes a comma-separated list, and a value with a
# space is fragile there.
# 2. Update the code, check Pillow (8.3 or newer: the JPEG tables' order) and run the two new suites:
#     cd "/davinci-1/home/ldellamea/TEEG/Towards-EEG/Passive Features/Diameter Re-measurement" && git fetch origin && git checkout sci/diameter-pipeline && git pull --ff-only && conda activate spine_env && python -c "import PIL; print('Pillow', PIL.__version__)" && python tests/smoke/test_smoke_config.py && python tests/smoke/test_smoke_camera_tables.py
# 3. Dry run, two-replicate probe, then the full array (from the same directory):
#     qsub -v DIAM_DRYRUN=1,RUN_TAG=v1,DIAM_CONFIG_JSON=$HOME/diam/config_production.json scripts/pbs/build_table.pbs
#     qsub -J 0-1 -v DIAM_N_DRAWS=2,DIAM_N_TASKS=2,RUN_TAG=probe,DIAM_CONFIG_JSON=$HOME/diam/config_production.json scripts/pbs/build_table.pbs
#     qsub -J 0-<T-1> -v RUN_TAG=v1,DIAM_N_DRAWS=<G>,DIAM_N_TASKS=<T>,DIAM_CONFIG_JSON=$HOME/diam/config_production.json scripts/pbs/build_table.pbs
# Expect in every `.o` file in `$HOME`: `[diam-table] env    spine_env`, a `Pillow` version line,
# `[diam-table] task <i> done`. `DIAM_N_DRAWS` (default 100000) sets the draws of a submission and
# `DIAM_DRAW_OFFSET` where it starts; every draw is rendered `phantom.n_realizations` times; `phantom.n_draws` in the
# configuration is a record only.
#
# 4. Merge with the same configuration, with `spine_env` active, in a job or an interactive session, not on the
# login node: at 10^6 rows the fit takes of order 20 min and a few hundred MB (Block 14.4). Every row records the
# configuration it was rendered under, and the merge stops with `merge refused` when it is given another one, or
# when a (draw, realization) occurs twice; without `--config-json` it assumes the default configuration:
#     python scripts/build_table.py merge --out-dir "/davinci-1/home/ldellamea/Human Neurons Fitting/diameter_tables/v1" --config-json $HOME/diam/config_production.json
# Expect `<N> rows of <G> draws from <T> files (phantom.n_draws records <G>)`, a `note:` line for any draw with
# fewer than `n_realizations` rows, then `<n> in S; local_linear, multiplier <c>, grid ... ->
# .../bias_table_<hash>.npz/.json and draws_<hash>.csv`, with the hash Cell 8 printed for `config_production.json`;
# a `WARNING` line means the cross-validated multiplier sits at an end of `bandwidth_cv_grid`.
#
# 5. Copy `bias_table_<hash>.npz`, `bias_table_<hash>.json` and `draws_<hash>.csv` (the per-draw table,
# D-060.2) to `MyDrive/diameters/tables/`.

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
# ===== CELL 11
# ====================================================================================================
# Cell 11: S3 step 0 on the pilot -- the pool, the pilot's selection, its acquisition check (Block 16.1)
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
SURVEY = OUT + "/survey_pilot"     # the pilot's own folder (D-099); survey/ keeps the run of 2026-10-28 over 51 cells
os.makedirs(SURVEY, exist_ok=True)
run("scripts/cell_pool.py", "pool", "--out", SURVEY + "/pool.csv")
run("scripts/cell_pool.py", "pick", "--pool", SURVEY + "/pool.csv", "--cells", SPECIMEN, "--out", SURVEY + "/selection.csv")
run("scripts/cell_pool.py", "check", "--pool", SURVEY + "/selection.csv", "--out", SURVEY + "/acquisition.csv")

# ====================================================================================================
# ===== CELL 11b
# ====================================================================================================
# Cell 11b: what Allen's projections are, on the pilot (Block 16.3, D-090.1)
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
SURVEY = OUT + "/survey_pilot"
SEL = SURVEY + "/selection.csv"
run("scripts/projection_check.py", *(["--selection", SEL] if os.path.exists(SEL) else ["--specimens", SPECIMEN]),
    "--check", SPECIMEN, "--cache-dir", CACHE_DIR, "--out-dir", SURVEY + "/projections")
from IPython.display import Image, display
FIG = "%s/projections/projection_check_%d.png" % (SURVEY, SPECIMEN)
if os.path.exists(FIG):
    display(Image(filename=FIG))

# ====================================================================================================
# ===== CELL 12
# ====================================================================================================
# Cell 12: S3 step 1 on the pilot -- its global alignment (Block 16.2)
# The xy projection of type 'MinimumIntensityProjection - xy' at level 4 (D-098.2; the darkest only when no SubImage has
# that type), `align_translation` with both flips, the score ratio R against block-shuffled images, a fine stage at
# level 1. The pilot's design alignment is no shift and no flip (Cell 0), so expect
# `[1/1] 529878215 align: shift (x, y) px` with x and y near 0, `no flip`, `ratio R` well above 2, and the overlay with the
# skeleton on the dendrites. `SET ASIDE` here means the alignment code, not the cell, needs a look: send the overlay.
# The 51 cells' alignments are the array's `align` step on davinci; a cell to align again there is
# `qsub -v RUN_TAG=v1,DIAM_CELLS=<id>,DIAM_FORCE=align scripts/pbs/survey_cells.pbs`.
#
# Send back: the `align:` line and the overlay.
SURVEY = OUT + "/survey_pilot"
run("scripts/survey_cells.py", "run", "--selection", SURVEY + "/selection.csv", "--out-dir", SURVEY,
    "--cache-dir", CACHE_DIR, "--steps", "align")
run("scripts/survey_cells.py", "status", "--out-dir", SURVEY, "--selection", SURVEY + "/selection.csv")
from IPython.display import Image, display
display(Image(filename="%s/%d/align/overlay_%d.png" % (SURVEY, SPECIMEN, SPECIMEN)))

# ====================================================================================================
# ===== CELL 13
# ====================================================================================================
# Cell 13: S3a on the pilot -- the node survey and the coverage of the phantoms' band (Block 17)
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

# ====================================================================================================
# ===== CELL 14
# ====================================================================================================
# Cell 14: S3b on the pilot -- d_max, the node jitter, the camera (Block 18)
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

# ====================================================================================================
# ===== CELL 15
# ====================================================================================================
# Cell 15: S3c on the pilot -- the kernel calibration step and the pool on one cell (Block 19)
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
#
# Cell 7's stripe knobs (`calibration/calibration_config.json`), when Cell 7 wrote them, are merged into this cell's
# own file `survey_config_s3c.json` and printed (`[cell 15] calibration knobs: ...`); the step runs once per cell, so
# after a change of the knobs set `REDO = ["--force", "calibration"]` [2026-10-28, later].
#
# [2026-10-29, D-112] The calibration step takes the pilot's candidates from Block 10.C's search over the whole cell,
# as Cell 7 does (Cell 13's nodes give only the typical background and absorption): it prints the `candidate search`
# line and both searches' lines, takes about as long as Cell 7's search before its first download, and writes
# `survey_pilot/529878215/calibration/near_search_529878215.csv` and `far_search_529878215.csv`; at most 5 scans per
# set, as before.
import json
SURVEY = OUT + "/survey_pilot"
CFG_S3 = SURVEY + "/survey_config.json"     # Cell 13's (node mode, the pilot's nodes per bin); written if Cell 13 did not run
if not os.path.exists(CFG_S3):
    with open(CFG_S3, "w") as f:
        json.dump({"survey": {"nodes_mode": "stratified", "nodes_per_bin": 3}}, f)
# Cell 7's stripe knobs, when Cell 7 wrote them, join Cell 13's settings in this cell's own file
CFG_CAL = OUT + "/calibration/calibration_config.json"
CFG_S3C = SURVEY + "/survey_config_s3c.json"
with open(CFG_S3) as f:
    cfg_s3c = json.load(f)
if os.path.exists(CFG_CAL):
    with open(CFG_CAL) as f:
        cfg_s3c["calibration"] = json.load(f)["calibration"]
with open(CFG_S3C, "w") as f:
    json.dump(cfg_s3c, f, indent=1)
print("[cell 15] calibration knobs:", cfg_s3c.get("calibration", "the defaults"))
REDO = []    # ["--force", "calibration"] after a change of Cell 7's knobs: a done step is not run again otherwise
run("scripts/survey_cells.py", "run", "--selection", SURVEY + "/selection.csv", "--out-dir", SURVEY,
    "--cache-dir", CACHE_DIR, "--steps", "align,nodes,calibration", "--config-json", CFG_S3C, *REDO)
run("scripts/survey_cells.py", "pool", "--out-dir", SURVEY, "--selection", SURVEY + "/selection.csv",
    "--what", "calibration", "--config-json", CFG_S3C)
from IPython.display import Image, display
for name in ("width_by_plane_near", "width_by_plane_far", "kernel_law", "kernel_law_by_cell",
             "kernel_law_residuals_by_cell"):
    if os.path.exists("%s/s3c/figures/%s.png" % (SURVEY, name)):
        display(Image(filename="%s/s3c/figures/%s.png" % (SURVEY, name)))

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
# - Cells 4e and 4f: the `[planediff]` lines and the figures of nodes 2 and 3 at both line lengths; the
# JSONs now carry the profiles, so the files themselves are enough.
# - Cell 4g: the `[planediff]` lines (the thin nodes and the summary line) and the summary figure.
# - Cell 4g (2026-10-09, afternoon): the figures of the nodes where a pick lies more than one plane from k*.
# - Cells 4h and 4i (2026-10-09, evening): the `[planediff]` lines (the nodes by diameter and the summary
# lines), both summary CSVs and figures, and the figures of the trunk nodes and of the nodes where the picks
# part; the JSONs carry the curves and the profiles, so the files themselves are enough.
# - Cell 4i (2026-10-10, the entropy on the +-5 um strip): the `[planediff]` summary line, the summary CSV and
# figure, and the figures of the nodes with d_hat between 1.5 and 3.5 um, where the entropy decides.
# - Cell 4j (the Allen-guided iteration; at m = 3 since 2026-10-12, D-044, in
# `pilot/planeiter/bydiameter_m3/`): the `[planediff]` lines (the trajectories and the endings), the summary
# CSV and figure, and the figures of the trunks and of the nodes whose last plane differs from the run at m =
# 2 or from the Gd pick; the JSON carries every round, so the files themselves are enough.
# - Cell 4k (2026-10-12, the planes the iteration chose): the `[planefit]` lines and the overview, and the
# figures of the trunks and of any node with a note; the JSON in `fits/` carries every caliper.
# - Cell 4l (2026-10-13, the fit variants): the `[fitvar]` lines, the summary figure, the table of eight
# variants, and the figures of the thin nodes (d_base under 1 um) and of the trunks; the CSV in `fitvar/`
# holds all 100 variants per node.
# - Cell 4m (2026-10-14, each variant's own iteration): the `[variter]` lines, the table of d per node, and
# the two figures of the nodes where some variant ends on another plane than Cell 4j's; the CSV in `variter/`
# holds every variant's trajectory.
# - Cell 5: the `"renderer"` block without the tables, `table_sets`, `pillow_version`, `notes`.
# - Cell 6: the `pooled dendrite diameter` line.
# - Cell 7: the growth-fit report, the far fit (gamma_hat and its interval) and the width-by-plane figures.
# - Cells 11-15 on the pilot (since D-099, survey_pilot/): Cell 11 the `pick:` and `check:` lines; Cell 11b
# survey_pilot/projections/projection_types.csv, projections_529878215.json and projection_check_529878215.png;
# Cell 12 the `align:` line and the overlay; Cell 13 the `nodes:` line and the coverage lines; Cell 14 the `inputs:`
# lines and inputs.png; Cell 15 the `calibration:` line and the pooled lines.
# - The survey over the selected cells (davinci, D-099): results_<RUN_TAG>.tgz of survey_pool.pbs -- status.csv,
# s3a/ (coverage.json, diameters.csv, the four figures), s3b/ (inputs.json, inputs_per_cell.csv, inputs.png), s3c/
# (the pooled lines in the job's log, kernel_law_by_cell.csv and the figures), every cell's align/ and nodes/.
#
# These decide the open items of SPEC section 8: d_max, the mu range, the jitter amplitudes, the camera fields
# and the dark-flag threshold.
# At the end of a session: write everything to Drive and unmount (the next cell would need Cell 0 again)
drive.flush_and_unmount()
