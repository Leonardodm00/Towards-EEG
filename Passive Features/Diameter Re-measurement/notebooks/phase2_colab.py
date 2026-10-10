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
# - Cells 4e and 4f: the `[planediff]` lines and the figures of nodes 2 and 3 at both line lengths; the
# JSONs now carry the profiles, so the files themselves are enough.
# - Cell 4g: the `[planediff]` lines (the thin nodes and the summary line) and the summary figure.
# - Cell 4g (2026-10-09, afternoon): the figures of the nodes where a pick lies more than one plane from k*.
# - Cells 4h and 4i (2026-10-09, evening): the `[planediff]` lines (the nodes by diameter and the summary
# lines), both summary CSVs and figures, and the figures of the trunk nodes and of the nodes where the picks
# part; the JSONs carry the curves and the profiles, so the files themselves are enough.
# - Cell 4i (2026-10-10, the entropy on the +-5 um strip): the `[planediff]` summary line, the summary CSV and
# figure, and the figures of the nodes with d_hat between 1.5 and 3.5 um, where the entropy decides.
# - Cell 5: the `"renderer"` block without the tables, `table_sets`, `pillow_version`, `notes`.
# - Cell 6: the `pooled dendrite diameter` line.
# - Cell 7: the growth-fit report.
#
# These decide the open items of SPEC section 8: d_max, the mu range, the jitter amplitudes, the camera fields
# and the dark-flag threshold.
# At the end of a session: write everything to Drive and unmount (the next cell would need Cell 0 again)
drive.flush_and_unmount()
