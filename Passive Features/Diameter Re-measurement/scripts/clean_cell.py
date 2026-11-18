#!/usr/bin/env python3
"""The pilot run of the cleaning on one cell -- Block 36 in specs/SPEC.md (plan steps C0, C5-C6;
D-129, D-130, D-132, D-136, D-138).

  survey  Block 27 on every dendrite node of the cell (no image is read): each node's
          contamination index and class on its own stretch (D-136.1), and near_end
          (D-138.2) -> contamination_<S>.csv; the example nodes drawn by
          contamination.pick_examples with cleaning.example_counts, seeded, never a
          near_end node (D-138.3) -> examples_<S>.json; a count per class printed
  rank    Block 27, amendment 2 (D-150.2) on every dendrite node (no image is read): each
          node's worst tier-1 neighbour -- one the cleaning subtracts -- by the structured
          part of its predicted dip on the measuring line, with the worst tier-2 peak ->
          rank_tier1_<S>.csv; the example nodes for a test of the cleaning
          (contamination.pick_tier1_examples) -> examples_tier1_<S>.json, which clean reads
  clean   the work list (node_inputs.work_list: the example nodes and the nodes of their
          strong tier-1 neighbours); the pilot of every work-list node (Block 5's
          measure_pilot; a node whose pilot or inputs fail is dropped and logged); round 0 =
          Allen's diameters and mu everywhere, the work-list nodes at their pilot's
          (d_hat, mu_hat), source "pilot" (D-130.O2); the rounds of Block 32
          (clean_iteration.run_rounds, predict = node_inputs.predict_factory, D-138.4) ->
          features_<S>.csv (Block 23), cleaning_<S>.json, cleaning_<S>_node<id>.npz per
          example node, cleaning_node<id>.png per example node, cleaning_rounds.png

Every step prints one line with the clock time and the seconds since the start; a node
that fails is dropped with its reason and counted, never the whole run.

--synthetic runs either command on the smoke tests' synthetic cell (tests/smoke/
fixtures_cell.py: a soma and one straight flat dendrite rendered through the camera chain,
served through fetch_zblock) without the network: a dry run.

Examples
    # Colab, on the pilot cell (the transform from the global alignment, Cell 12):
    python scripts/clean_cell.py survey --specimen 529878215 --B 210 --mu 0.9 \\
        --shift-x 12.3 --shift-y -4.1 --cache-dir /content/drive/MyDrive/allen_cache --out OUT/cleaning
    python scripts/clean_cell.py clean --specimen 529878215 --examples OUT/cleaning/examples_529878215.json \\
        --cache-dir /content/drive/MyDrive/allen_cache --out OUT/cleaning
    # dry run, no network:
    python scripts/clean_cell.py survey --synthetic --specimen 0 --B 210 --mu 1.0 --out /tmp/c
    python scripts/clean_cell.py rank --synthetic --specimen 0 --B 210 --mu 1.0 --out /tmp/c
    python scripts/clean_cell.py clean --synthetic --specimen 0 --examples /tmp/c/examples_0.json --out /tmp/c

Pure ASCII, LF, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import signal
import sys
import tempfile
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
SRC = os.path.join(WS, "src")
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from allen_diameter.analysis import cell, clean_iteration, contamination, node_features  # noqa: E402
from allen_diameter.analysis import node_inputs  # noqa: E402
from allen_diameter.analysis.node_pipeline import measure_pilot  # noqa: E402
from allen_diameter.loading import features_io, swc_io, table_io  # noqa: E402
from allen_diameter.model import kernel, line_fast, shadow  # noqa: E402
from build_table import load_config  # noqa: E402

# the synthetic cell of --synthetic (tests/smoke/fixtures_cell.py): 14 nodes 1.18 um apart, so
# that the middle nodes are not near_end at the default margins; the rendered tube's d and mu
SYNTH_N_NODES = 14
SYNTH_D_TRUE_UM = 0.8
SYNTH_MU_PER_UM = 1.0
SYNTH_ALLEN_RADIUS_UM = 0.3
DPI = 110

CONTAMINATION_COLUMNS = ("node_id", "own_dip_gl", "index", "class", "n_neighbours", "worst_child_id",
                         "worst_lateral_um", "worst_defocus_um", "worst_share", "soma_lateral_um", "near_end")


# ---------------------------------------------------------------- logging ---

class Log(object):
    """One line per step: clock time, seconds since the start, the message."""

    def __init__(self):
        self.t0 = time.time()

    def __call__(self, msg):
        now = time.time()
        print("[%s +%7.1f s] %s" % (time.strftime("%H:%M:%S", time.localtime(now)), now - self.t0, msg), flush=True)


# ------------------------------------------------------------------ files ---

def jsonable(x):
    """numpy -> python, non-finite floats -> None (strict JSON)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def write_json(path, obj):
    tmp = path + ".tmp"                      # written whole, then renamed: a kill never leaves half a file (F4)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(jsonable(obj), f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)
    return path


def _cell_text(v):
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return str(v)


def write_csv(path, rows, columns):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([_cell_text(r.get(c, "")) for c in columns])
    os.replace(tmp, path)
    return path


def own_record(status, cfg, clr=None, detail=""):
    """A node's `own` entry of the JSON (Block 36, amendment 11): the same-node clearance's status, the band
    and, when it was computed, the per-plane chord, peaks, in-band share and residual RMS (D-156.2)."""
    rec = dict(status=str(status), band_half_um=float(cfg.cleaning.own_band_half_um))
    if detail:
        rec["detail"] = str(detail)
    if clr is not None:
        rec.update(chord_um=clr.chord_um, in_peak_gl=clr.in_peak_gl, out_peak_gl=clr.out_peak_gl,
                   in_share=clr.in_share, residual_rms_gl=clr.residual_rms_gl)
    return rec


def own_arrays(split, clr):
    """The example node's own_* arrays for its NPZ (amendment 11); the layers' arrays are empty without layers."""
    out = dict(own_s_in=split.s_in, own_s_out=split.s_out, own_cleared=clr.cleared_gl, own_model_in=clr.model_in_gl,
               own_chord_um=clr.chord_um)
    has = clr.layers_gl is not None and split.layer_z is not None
    out["own_layers_gl"] = np.asarray(clr.layers_gl) if has else np.zeros((0,) + split.s_in.shape)
    out["own_layer_z"] = np.asarray(split.layer_z, dtype=float) if has else np.zeros(0)
    return out


def own_figures(out_dir, nid, node, profiles_gl, B_gl, split, clr, v_axis_um, meta, k_star, cfg, dpi, round_index=None):
    """own_node<id>.png (and own_node<id>_round<t>.png with round_index), own_layers_node<id>.png at the plane
    k_star when the split carries layers (Block 38); -> the list of files written."""
    import matplotlib.pyplot as plt
    from allen_diameter.plotting import own_clearance_figures as OF
    band = float(cfg.cleaning.own_band_half_um)
    meta = dict(meta, band_half_um=band)
    ks = [int(k) for k in node.ks]
    planes = [k for k in (k_star - 2, k_star, k_star + 2) if k in ks] or None
    fig = OF.figure_own(node.v, node.ks, profiles_gl, B_gl, clr, meta, planes=planes, v0_um=float(v_axis_um))
    files = [OF.save(fig, os.path.join(out_dir, "own_node%d.png" % nid), dpi)]
    if round_index is not None:
        files.append(OF.save(fig, os.path.join(out_dir, "own_node%d_round%d.png" % (nid, int(round_index))), dpi))
    plt.close(fig)
    if clr.layers_gl is not None and split.layer_z is not None and len(split.layer_z) > 0 and k_star in ks:
        fig = OF.figure_layers(node.v, split.layer_z, clr.layers_gl, node.ks, int(k_star), node.planes_z, band, meta)
        files.append(OF.save(fig, os.path.join(out_dir, "own_layers_node%d.png" % nid), dpi))
        plt.close(fig)
    return files


def config_sections(cfg):
    """The configuration's cleaning and inference sections (D-136.5) and the full signature."""
    return dict(cleaning=dataclasses.asdict(cfg.cleaning), inference=dataclasses.asdict(cfg.inference),
                config=json.loads(cfg.to_json()))


# ------------------------------------------------------------------- cell ---

def transform_of(a, default=None):
    """keyword arguments of cell.to_image_um from --shift-x/--shift-y/--flip-h/--z0, each falling back to
    `default` (a dict as written in examples_<S>.json) and then to the identity."""
    d = dict(default or {})
    sx = a.shift_x if a.shift_x is not None else float((d.get("shift_full_px") or (0.0, 0.0))[0])
    sy = a.shift_y if a.shift_y is not None else float((d.get("shift_full_px") or (0.0, 0.0))[1])
    fh = a.flip_h if a.flip_h is not None else d.get("flip_y_full_h")
    z0 = a.z0 if a.z0 is not None else float(d.get("z0_um", 0.0))
    return dict(shift_full_px=(float(sx), float(sy)), flip_y_full_h=None if fh is None else float(fh), z0_um=float(z0))


def load_cell(a, cfg, log, need_images=True):
    """(swc, provider or None, label). --synthetic: the smoke tests' synthetic cell, no network."""
    if a.synthetic:
        sys.path.insert(0, os.path.join(WS, "tests", "smoke"))
        import run_cell
        from fixtures_cell import synthetic_cell
        with tempfile.TemporaryDirectory(prefix="clean_cell_synthetic_") as tmp:    # the SWC is read in memory
            swc, fetcher, planes, _path = synthetic_cell(tmp, getattr(a, "cfg_data", None) or cfg,
                                                         d_true=SYNTH_D_TRUE_UM, mu=SYNTH_MU_PER_UM,
                                                         n_nodes=SYNTH_N_NODES, allen_radius=SYNTH_ALLEN_RADIUS_UM)
        log("synthetic cell: %d SWC nodes, the dendrite rendered with d %.3f um, mu %.3f /um (no network)"
            % (len(swc.ids), SYNTH_D_TRUE_UM, SYNTH_MU_PER_UM))
        return swc, run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um), "synthetic"
    import allen_image_io as aio
    path = a.swc or aio.fetch_swc(int(a.specimen), a.cache_dir)
    swc = swc_io.read_swc(path)
    log("SWC %s: %d nodes" % (path, len(swc.ids)))
    if not need_images:
        return swc, None, "real"
    import run_cell
    planes = aio.plane_table(aio.list_images(int(a.specimen)))
    log("plane table: %d planes" % len(planes))
    return swc, run_cell.real_provider(aio.HttpFetcher(cache_dir=a.cache_dir), planes, cfg.acquisition.res0_um), "real"


def soma_z(swc, cfg, transform):
    """The soma's depth in the image frame (um): the mean of the type-1 nodes' z after the transform; None
    without a soma node."""
    som = np.asarray(swc.types) == 1
    if not som.any():
        return None
    xyz = cell.to_image_um(swc.xyz, cfg.acquisition.res0_um, **transform)
    return float(np.mean(xyz[som, 2]))


# ----------------------------------------------------------------- survey ---

class _Stop(Exception):
    """Raised by the SIGTERM handler: the notebook's run() helper terminates the script when its cell is stopped."""


def _on_sigterm(signum, frame):
    raise _Stop()


def _survey_outputs(a, cfg, log, rows, single, failed, label, transform, n_target, n_dendrite, partial):
    """Write contamination_<S>.csv and examples_<S>.json from the rows surveyed so far (a checkpoint when
    partial). The example nodes are drawn from these rows only."""
    csv_path = write_csv(os.path.join(a.out, "contamination_%s.csv" % a.specimen), rows, CONTAMINATION_COLUMNS)
    eligible = [r for r in rows if not r["near_end"]]
    rng = np.random.default_rng(int(a.seed if a.seed is not None else cfg.phantom.seed))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        examples = contamination.pick_examples(eligible, tuple(cfg.cleaning.example_counts), rng)
    by_id = {int(r["node_id"]): r for r in rows}
    counts = {c: sum(1 for r in rows if r["class"] == c) for c in contamination.CLASSES}
    counts_eligible = {c: sum(1 for r in eligible if r["class"] == c) for c in contamination.CLASSES}
    rec = dict(specimen=str(a.specimen), source=label, example_ids=examples,
               classes={str(i): by_id[i]["class"] for i in examples}, B_gl=float(a.B), mu_per_um=float(a.mu),
               seed=int(a.seed if a.seed is not None else cfg.phantom.seed),
               example_counts=list(cfg.cleaning.example_counts), example_keys=list(contamination.EXAMPLE_KEYS),
               transform=dict(shift_full_px=list(transform["shift_full_px"]),
                              flip_y_full_h=transform["flip_y_full_h"], z0_um=transform["z0_um"]),
               swc=a.swc or "", n_surveyed=len(rows), n_near_end=len(rows) - len(eligible),
               n_single_node_stretches=single, failed=failed, counts=counts, counts_not_near_end=counts_eligible,
               n_target=int(n_target), n_dendrite_nodes=int(n_dendrite), partial=bool(partial),
               sample="all dendrite nodes" if n_target == n_dendrite else
                      "a seeded random sample of %d of %d dendrite nodes" % (n_target, n_dendrite),
               **config_sections(cfg))
    ex_path = write_json(os.path.join(a.out, "examples_%s.json" % a.specimen), rec)
    return csv_path, ex_path, examples, counts, counts_eligible, caught


def run_survey(a, cfg, log):
    """The survey visits the dendrite nodes in a seeded random order, so the nodes surveyed at any moment are a
    random sample of the cell; --max-nodes stops after that many. The outputs are rewritten every
    --checkpoint-every nodes and when the script is stopped (SIGINT, or SIGTERM from the notebook's run()):
    a stopped survey leaves a usable contamination table and example file of the nodes done."""
    os.makedirs(a.out, exist_ok=True)
    swc, _provider, label = load_cell(a, cfg, log, need_images=False)
    transform = transform_of(a)
    res0 = float(cfg.acquisition.res0_um)
    xyz = cell.to_image_um(swc.xyz, res0, **transform)
    scene = cell.cell_scene(swc, tuple(cfg.cleaning.neighbour_swc_types), res0, transform)
    log("scene: %d segments (types %s), %d soma discs; B %.4g gl, mu %.4g /um"
        % (scene.seg_xyz.shape[0], tuple(cfg.cleaning.neighbour_swc_types), scene.soma.shape[0], a.B, a.mu))
    rows, single, failed = [], 0, []
    runs = cell.stretches(swc, tuple(cfg.acquisition.dendrite_swc_types))
    branches = []
    for run in runs:
        ob = node_inputs.own_branch(swc, run, xyz)
        if ob is None:
            single += 1
            log("node %d: a one-node stretch, not surveyed (no own direction; near_end)" % int(swc.ids[run[0]]))
            continue
        branches.append(ob[0])
    pairs = [(b, t) for b in branches for t in range(len(b.ids))]
    n_dendrite = len(pairs)
    order = np.random.default_rng(int(a.seed if a.seed is not None else cfg.phantom.seed) + 1).permutation(n_dendrite)
    if a.max_nodes is not None:
        order = order[:max(0, int(a.max_nodes))]
    n_target = len(order)
    log("survey of %s (seeded random order; outputs rewritten every %d nodes and when stopped)"
        % ("all %d dendrite nodes" % n_dendrite if n_target == n_dendrite
           else "%d of %d dendrite nodes" % (n_target, n_dendrite), a.checkpoint_every))
    old = signal.signal(signal.SIGTERM, _on_sigterm)
    stopped, t_last = False, time.time()
    try:
        for j in order:
            branch, t = pairs[int(j)]
            nid = int(branch.ids[t])
            try:
                row = contamination.node_contamination(branch, t, scene, a.B, a.mu, cfg)
            except (KeyboardInterrupt, _Stop):
                raise
            except Exception as exc:          # one node's failure never stops the survey
                failed.append(nid)
                log("node %d: survey failed (%s: %s), dropped" % (nid, type(exc).__name__, exc))
                continue
            row["near_end"] = node_inputs.is_near_end(branch, t, cfg)
            rows.append(row)
            if len(rows) % max(1, int(a.checkpoint_every)) == 0:
                _survey_outputs(a, cfg, log, rows, single, failed, label, transform, n_target, n_dendrite, True)
            if len(rows) % 200 == 0 or time.time() - t_last > 60:
                t_last = time.time()
                log("%d of %d dendrite nodes surveyed" % (len(rows), n_target))
    except (KeyboardInterrupt, _Stop):
        stopped = True
        log("survey stopped after %d of %d nodes: writing the outputs of the nodes done" % (len(rows), n_target))
    finally:
        signal.signal(signal.SIGTERM, old)
    csv_path, ex_path, examples, counts, counts_eligible, caught = _survey_outputs(
        a, cfg, log, rows, single, failed, label, transform, n_target, n_dendrite, stopped)
    for w in caught:
        log(str(w.message))
    for c in contamination.CLASSES:
        print("class %-15s %6d nodes (%d not near_end)" % (c, counts[c], counts_eligible[c]), flush=True)
    log("survey: %d nodes%s, %d near_end, %d one-node stretches skipped, %d failed -> %s; %d example nodes %s -> %s"
        % (len(rows), " (stopped early)" if stopped else "", len(rows) - sum(counts_eligible.values()), single,
           len(failed), csv_path, len(examples), examples, ex_path))
    return 0


# ------------------------------------------------------------------- rank ---

RANK_COLUMNS = tuple(contamination.TIER1_COLUMNS) + ("near_end",)


def _rank_outputs(a, cfg, rows, label, transform, n_target, n_dendrite, partial, failed, single):
    """rank_tier1_<S>.csv (every node ranked so far) and examples_tier1_<S>.json (pick_tier1_examples' nodes, in
    the examples file's format that clean reads), from the rows so far (a checkpoint when partial)."""
    csv_path = write_csv(os.path.join(a.out, "rank_tier1_%s.csv" % a.specimen), rows, RANK_COLUMNS)
    c = cfg.cleaning
    examples = contamination.pick_tier1_examples(rows, c.rank_examples, c.rank_max_tier2_gl, c.rank_min_separation_um)
    by_id = {int(r["node_id"]): r for r in rows}
    rec = dict(specimen=str(a.specimen), source=label, example_ids=examples,
               classes={str(i): "tier1" for i in examples}, B_gl=float(a.B), mu_per_um=float(a.mu),
               transform=dict(shift_full_px=list(transform["shift_full_px"]),
                              flip_y_full_h=transform["flip_y_full_h"], z0_um=transform["z0_um"]),
               swc=a.swc or "", n_ranked=len(rows), n_target=int(n_target), n_dendrite_nodes=int(n_dendrite),
               n_single_node_stretches=single, failed=failed, partial=bool(partial),
               sample="all dendrite nodes" if n_target == n_dendrite else
                      "a seeded random sample of %d of %d dendrite nodes" % (n_target, n_dendrite),
               ranking={str(i): {k: by_id[i][k] for k in RANK_COLUMNS} for i in examples},
               **config_sections(cfg))
    ex_path = write_json(os.path.join(a.out, "examples_tier1_%s.json" % a.specimen), rec)
    return csv_path, ex_path, examples


def _truthy(x):
    return str(x).strip().lower() in ("1", "true", "yes")


def parse_d_bins(text):
    """--d-bins "e0,e1,...,eJ" -> [e0, ..., eJ] (um), or None for an empty text. ValueError unless there are at
    least two edges, all finite, > 0 and strictly increasing (Block 36, amendment 15)."""
    if not str(text).strip():
        return None
    try:
        edges = [float(x) for x in str(text).split(",")]
    except ValueError:
        raise ValueError("--d-bins must be numbers separated by commas (um), got %r" % (text,))
    if len(edges) < 2 or not all(math.isfinite(e) and e > 0 for e in edges) \
            or any(b <= a_ for a_, b in zip(edges[:-1], edges[1:])):
        raise ValueError("--d-bins needs at least two finite edges > 0, strictly increasing (um), got %r" % (text,))
    return edges


def run_clear(a, cfg, log):
    """examples_clear_<S>.json: the clearest nodes of rank's table (Block 36, amendment 13) -- not near_end, no
    tier-2 neighbour, at least --min-soma-um from the soma, the least tier-1 light on the line first,
    cleaning.rank_min_separation_um apart. Reads rank's two outputs in --out; no image, no network."""
    rank_csv = os.path.join(a.out, "rank_tier1_%s.csv" % a.specimen)
    ex_t1 = os.path.join(a.out, "examples_tier1_%s.json" % a.specimen)
    for path in (rank_csv, ex_t1):
        if not os.path.exists(path):
            log("clear: %s is missing: run `clean_cell.py rank` first" % path)
            return 1
    with open(rank_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:                                         # the CSV's text back to numbers
        for k in list(r):
            if k == "near_end":
                r[k] = _truthy(r[k])
            elif k in ("node_id", "n_tier2", "n_evaluated", "worst1_child_id"):
                r[k] = int(float(r[k])) if str(r[k]).strip() not in ("", "nan") else -1
            else:
                r[k] = float(r[k]) if str(r[k]).strip() != "" else float("nan")
    sep = float(cfg.cleaning.rank_min_separation_um)
    by_id = {int(r["node_id"]): r for r in rows}
    bins = parse_d_bins(getattr(a, "d_bins", ""))
    sizes = {}
    if bins is None:
        n_cand = len(contamination.pick_clear_examples(rows, len(rows), 0.0, a.min_soma_um))
        picked = contamination.pick_clear_examples(rows, a.n, sep, a.min_soma_um)
    else:        # amendment 15 (D-165): --n nodes per bin of Allen's diameter (twice the traced radius)
        swc, _provider, _label = load_cell(a, cfg, log, need_images=False)
        d_allen = {int(i): 2.0 * float(r) for i, r in zip(np.asarray(swc.ids), np.asarray(swc.radius, dtype=float))}
        n_cand, picked, pos = 0, [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sub = [r for r in rows if lo <= d_allen.get(int(r["node_id"]), float("nan")) < hi]
            n_bin = len(contamination.pick_clear_examples(sub, len(sub), 0.0, a.min_soma_um))
            got = contamination.pick_clear_examples(sub, a.n, sep, a.min_soma_um, taken_xyz=pos)
            n_cand += n_bin
            log("clear: Allen's diameter in [%.3g, %.3g) um: %d ranked nodes, %d qualify, picked %s"
                % (lo, hi, len(sub), n_bin, got if got else "none (the bin is skipped)"))
            for i in got:
                picked.append(i)
                pos.append([float(by_id[i][k]) for k in ("x_um", "y_um", "z_um")])
                sizes[str(i)] = dict(d_allen_um=d_allen[i], bin_um=[float(lo), float(hi)])
    if not picked:
        log("clear: no node of %d qualifies (not near_end, no tier-2 neighbour, >= %.1f um from the soma%s)"
            % (len(rows), a.min_soma_um, "" if bins is None else ", in the bins of --d-bins"))
        return 1
    with open(ex_t1) as f:
        rec = json.load(f)
    clear = dict(n=int(a.n), min_soma_um=float(a.min_soma_um), min_separation_um=sep, n_candidates=n_cand)
    if bins is not None:
        clear = dict(clear, d_bins_um=[float(e) for e in bins], n_per_bin=int(a.n))
    rec.update(example_ids=picked, classes={str(i): "clear" for i in picked},
               ranking={str(i): {k: by_id[i][k] for k in RANK_COLUMNS} for i in picked}, clear=clear)
    if bins is not None:
        rec["sizes"] = sizes
    out = write_json(os.path.join(a.out, "examples_clear_%s.json" % a.specimen), rec)
    log("clear: %d of %d ranked nodes qualify; picked %s -> %s" % (n_cand, len(rows), picked, out))
    print("node    own dip (gl)   tier-1 peak (gl)   tier-1 lateral (um)   soma (um)   x, y, z (um)%s"
          % ("" if bins is None else "   Allen d (um)"), flush=True)
    for i in picked:
        r = by_id[i]
        print("%-7d %-14.2f %-18.3f %-21.2f %-11.1f %.1f, %.1f, %.1f%s"
              % (i, float(r["own_dip_gl"]), float(r["worst1_peak_gl"]), float(r["worst1_lateral_um"]),
                 float(r["soma_lateral_um"]), float(r["x_um"]), float(r["y_um"]), float(r["z_um"]),
                 "" if bins is None else "   %.3f" % sizes[str(i)]["d_allen_um"]), flush=True)
    return 0


def run_rank(a, cfg, log):
    """Block 27, amendment 2 (D-150.2): every dendrite node (or a seeded random sample, --max-nodes) ranked by its
    worst tier-1 neighbour's structured dip on its measuring line, at round 0 (Allen's diameters, --mu); the
    example nodes for a test of the cleaning are the highest not near_end, with tier-2 neighbours under
    cleaning.rank_max_tier2_gl on the line, cleaning.rank_min_separation_um apart. Geometry only: no image."""
    os.makedirs(a.out, exist_ok=True)
    swc, _provider, label = load_cell(a, cfg, log, need_images=False)
    transform = transform_of(a)
    res0 = float(cfg.acquisition.res0_um)
    xyz = cell.to_image_um(swc.xyz, res0, **transform)
    types = tuple(cfg.cleaning.neighbour_swc_types)
    scene = cell.cell_scene(swc, types, res0, transform)
    fscene = node_features.featured_scene(scene, node_features.initial_features(swc, types, a.mu))
    log("scene: %d segments, %d soma discs; B %.4g gl, mu %.4g /um; bound %.3g gl, examples %d (tier 2 on the line "
        "<= %.3g gl, %.3g um apart)" % (scene.seg_xyz.shape[0], scene.soma.shape[0], a.B, a.mu,
                                         cfg.cleaning.rank_bound_gl, cfg.cleaning.rank_examples,
                                         cfg.cleaning.rank_max_tier2_gl, cfg.cleaning.rank_min_separation_um))
    branches, single = [], 0
    for run in cell.stretches(swc, tuple(cfg.acquisition.dendrite_swc_types)):
        ob = node_inputs.own_branch(swc, run, xyz)
        if ob is None:
            single += 1
            continue
        branches.append(ob[0])
    pairs = [(b, t) for b in branches for t in range(len(b.ids))]
    n_dendrite = len(pairs)
    order = np.random.default_rng(int(a.seed if a.seed is not None else cfg.phantom.seed) + 1).permutation(n_dendrite)
    if a.max_nodes is not None:
        order = order[:max(0, int(a.max_nodes))]
    n_target = len(order)
    log("ranking %s" % ("all %d dendrite nodes" % n_dendrite if n_target == n_dendrite
                        else "%d of %d dendrite nodes (seeded random sample)" % (n_target, n_dendrite)))
    rows, failed = [], []
    U = float(cfg.inference.tube_half_length_um)
    old = signal.signal(signal.SIGTERM, _on_sigterm)
    stopped, t_last = False, time.time()
    try:
        for j in order:
            branch, t = pairs[int(j)]
            nid = int(branch.ids[t])
            try:
                own = node_inputs.own_ids_within(branch, t, U)[0]
                row = contamination.tier1_contamination(fscene, branch, t, own, a.B, cfg)
            except (KeyboardInterrupt, _Stop):
                raise
            except Exception as exc:          # one node's failure never stops the ranking
                failed.append(nid)
                log("node %d: ranking failed (%s: %s), dropped" % (nid, type(exc).__name__, exc))
                continue
            row["near_end"] = node_inputs.is_near_end(branch, t, cfg)
            rows.append(row)
            if time.time() - t_last > 60:
                t_last = time.time()
                log("%d of %d dendrite nodes ranked" % (len(rows), n_target))
    except (KeyboardInterrupt, _Stop):
        stopped = True
        log("ranking stopped after %d of %d nodes: writing the outputs of the nodes done" % (len(rows), n_target))
    finally:
        signal.signal(signal.SIGTERM, old)
    csv_path, ex_path, examples = _rank_outputs(a, cfg, rows, label, transform, n_target, n_dendrite, stopped,
                                                failed, single)
    by_id = {int(r["node_id"]): r for r in rows}
    top = sorted(rows, key=lambda r: (-float(r["worst1_range_gl"]), int(r["node_id"])))[:10]
    print("the ten largest tier-1 structured dips on the line (gl): " + ", ".join(
        "%d %.2f%s" % (int(r["node_id"]), float(r["worst1_range_gl"]), " (near_end)" if r["near_end"] else
                       " (tier 2 %.1f)" % float(r["worst2_peak_gl"]) if float(r["worst2_peak_gl"]) >
                       cfg.cleaning.rank_max_tier2_gl else "") for r in top), flush=True)
    for i in examples:
        r = by_id[i]
        log("example %d: worst tier-1 neighbour (child %d) %.2f gl structured, peak %.2f gl, %.2f um aside, %.2f um "
            "out of focus; tier-2 peak %.2f gl; own dip %.1f gl" % (
                i, int(r["worst1_child_id"]), r["worst1_range_gl"], r["worst1_peak_gl"], r["worst1_lateral_um"],
                r["worst1_defocus_um"], r["worst2_peak_gl"], r["own_dip_gl"]))
    log("rank: %d nodes%s, %d failed -> %s; %d example nodes %s -> %s" % (
        len(rows), " (stopped early)" if stopped else "", len(failed), csv_path, len(examples), examples, ex_path))
    return 0


# ------------------------------------------------------------------ clean ---

class NodeClock(object):
    """Times each node of each round through the two callables Block 32 calls per node (shadow_fn first,
    predict_factory after the cleaning): a node's line is printed when the next node starts or the run
    ends; a round's line when its last node's line is printed."""

    def __init__(self, n_nodes, log):
        self.n, self.log = int(n_nodes), log
        self.k = 0                     # nodes started
        self.cur = None                # (round, node_id, t_start, t_predict, nfev)
        self.round_t0 = None
        self.calls = 0
        self.bg_seconds = None

    def node_start(self, nid=None):
        now = time.time()
        self.finish(now)
        if self.k % self.n == 0:
            self.round_t0 = now
        self.k += 1
        self.cur = [1 + (self.k - 1) // self.n, None if nid is None else int(nid), now, None]
        self.calls = 0
        self.bg_seconds = None

    def node_predict(self, nid):
        if self.cur is not None:
            self.cur[1], self.cur[3] = int(nid), time.time()

    def finish(self, now=None):
        if self.cur is None:
            return
        now = time.time() if now is None else now
        rnd, nid, t0, tp = self.cur
        tp = now if tp is None else tp
        bg = "" if self.bg_seconds is None else "background refit %.1f s, " % self.bg_seconds
        self.log("round %d node %s: %sshadow and cleaning %.1f s, refinement %.1f s (%d model calls), %.1f s"
                 % (rnd, nid, bg, tp - t0 - (self.bg_seconds or 0.0), now - tp, self.calls, now - t0))
        if self.k % self.n == 0:
            self.log("round %d: %d nodes in %.1f s" % (rnd, self.n, now - self.round_t0))
        self.cur = None


def _clip_start(theta0, cfg):
    """theta0 = (d, mu, v_t, dz) clipped into Block 29's refine box (refine_node refuses a start outside it);
    returns (theta0, clipped?)."""
    c = cfg.cleaning
    lo = np.array([c.refine_d_bounds_um[0], c.refine_mu_bounds_per_um[0], c.refine_shift_bounds_um[0],
                   c.refine_dz_bounds_um[0]], dtype=float)
    hi = np.array([c.refine_d_bounds_um[1], c.refine_mu_bounds_per_um[1], c.refine_shift_bounds_um[1],
                   c.refine_dz_bounds_um[1]], dtype=float)
    th = np.asarray(theta0, dtype=float)
    if not np.all(np.isfinite(th)):
        return th, False
    out = np.clip(th, lo, hi)
    return out, bool(np.any(out != th))


def _float(x):
    """A CSV value as float; NaN when missing or not a number."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def pilot_fields(p):
    names = ("k_star", "z_sub_um", "cx_um", "cy_um", "theta_rad", "phi_rad", "B_bar", "d_hat_um", "mu_hat_per_um",
             "v0_hat_um", "alpha_hat", "fit_status")
    out = {n: getattr(p, n) for n in names}
    out["flags"] = list(p.flags)
    return out


def _cache_logger(cfg, log):
    """log_cache(label): the neighbours' images reused and rendered since the last call, and what the cache of
    shadow.shadow_image keeps (Block 25, amendment 7, D-150.3)."""
    seen = {"hits": 0, "misses": 0}

    def log_cache(label):
        ci = shadow.cache_info()
        log("%s: shadow images %d reused, %d rendered; %d patches kept, %.0f MB (budget %g MB)"
            % (label, ci["hits"] - seen["hits"], ci["misses"] - seen["misses"], ci["patches"], ci["mb"],
               float(cfg.cleaning.shadow_cache_mb)))
        seen.update(hits=ci["hits"], misses=ci["misses"])
    return log_cache


def run_clean(a, cfg, log):
    os.makedirs(a.out, exist_ok=True)
    shadow.cache_clear()                 # what shadow_image keeps across rounds starts empty (D-150.3)
    with open(a.examples, encoding="utf-8") as f:
        ex = json.load(f)
    examples = [int(i) for i in ex["example_ids"]]
    classes = {int(k): v for k, v in ex.get("classes", {}).items()}
    B, B_src = (float(a.B), "--B") if a.B is not None else (float(ex["B_gl"]), "examples file")
    mu, mu_src = (float(a.mu), "--mu") if a.mu is not None else (float(ex["mu_per_um"]), "examples file")
    if a.nodes_csv:
        prow = [cell.pilot_row_view(r) for r in table_io.read_rows([a.nodes_csv])]
        bb = np.array([_float(r.get("B_bar")) for r in prow], dtype=float)
        mm = np.array([_float(r.get("mu_hat_per_um")) for r in prow], dtype=float)
        if a.B is None and np.any(np.isfinite(bb) & (bb > 0)):
            B, B_src = float(np.median(bb[np.isfinite(bb) & (bb > 0)])), "median B_bar of %s" % a.nodes_csv
        if a.mu is None and np.any(np.isfinite(mm) & (mm > 0)):
            mu, mu_src = float(np.median(mm[np.isfinite(mm) & (mm > 0)])), "median mu_hat of %s" % a.nodes_csv
    transform = transform_of(a, ex.get("transform"))
    cl = cfg.cleaning
    repl = {}
    if a.rounds is not None:
        repl["max_rounds"] = int(a.rounds)
    if a.work_peak_gl is not None:
        repl["work_peak_min_gl"] = float(a.work_peak_gl)
    if a.max_work is not None:
        repl["work_max_nodes"] = int(a.max_work)
    swc, provider, label = load_cell(a, cfg, log)
    zs = soma_z(swc, cfg, transform)
    if zs is not None:
        repl["soma_z_um"] = zs                     # CleaningConfig.soma_z_um: set by the script (Block 25)
    cfg = dataclasses.replace(cfg, cleaning=dataclasses.replace(cl, **repl))
    cfg.validate()
    log("B %.4g gl (%s), mu %.4g /um (%s); rounds <= %d, work peak >= %.3g gl on an example's %s, work list <= %d "
        "nodes; soma z %s um; fit absorption %s, %d x %d directions; shadow absorption %s"
        % (B, B_src, mu, mu_src, cfg.cleaning.max_rounds, cfg.cleaning.work_peak_min_gl, cfg.cleaning.work_rank,
           cfg.cleaning.work_max_nodes, "%.3f" % zs if zs is not None else "none", cfg.inference.absorption,
           cfg.inference.ray_n_rho, cfg.inference.ray_n_psi, cfg.cleaning.shadow_absorption))
    res0 = float(cfg.acquisition.res0_um)
    types = tuple(cfg.cleaning.neighbour_swc_types)
    xyz = cell.to_image_um(swc.xyz, res0, **transform)
    scene = cell.cell_scene(swc, types, res0, transform)
    features0 = node_features.initial_features(swc, types, mu)
    fscene0 = node_features.featured_scene(scene, features0)
    log("scene: %d segments, %d soma discs; %d nodes in the features table" % (scene.seg_xyz.shape[0],
                                                                                scene.soma.shape[0], len(features0)))
    wpeaks = {}
    no_clean = bool(getattr(a, "no_clean", False))        # amendment 16 (D-169.7): the examples only, nothing cleaned
    pilot_d = getattr(a, "pilot_diameter", None) or ("allen" if no_clean else "free")
    if no_clean:
        work = [int(i) for i in examples]
        log("--no-clean (D-169): the work list is the %d example nodes; no neighbour is cleaned, no round is run; "
            "the first stage's diameter is %s" % (len(work), "held at Allen's (D-173)" if pilot_d == "allen"
                                                  else "fitted"))
    else:
        work = node_inputs.work_list(examples, fscene0, swc, cfg, cfg.cleaning.work_peak_min_gl, B, peaks=wpeaks)
    cut = [i for i in examples if i not in work]
    if cut:                                         # amendment 2, F5: more examples than work_max_nodes
        log("work list capped at %d nodes: %d example nodes cut %s" % (cfg.cleaning.work_max_nodes, len(cut), cut))
    n_ex = len(examples) - len(cut)
    log("work list: %d nodes (%d examples, %d neighbours' nodes)" % (len(work), n_ex, len(work) - n_ex))
    if wpeaks:                                      # amendment 9 (D-149.2): what joined, on whose line, and what not
        joined = [e for e in work if e in wpeaks]
        left_out = sorted((e for e in wpeaks if e not in work), key=lambda e: (-wpeaks[e][0], e))
        log("work list: neighbours' nodes by their largest predicted shadow on an example's %s, gl (example): %s"
            % (cfg.cleaning.work_rank, ", ".join("%d %.2f (%d)" % (e, wpeaks[e][0], wpeaks[e][1]) for e in joined)))
        if left_out:
            log("work list: %d more nodes reach %.3g gl but are cut by the cap of %d: %s%s"
                % (len(left_out), cfg.cleaning.work_peak_min_gl, cfg.cleaning.work_max_nodes,
                   ", ".join("%d %.2f (%d)" % (e, wpeaks[e][0], wpeaks[e][1]) for e in left_out[:15]),
                   " ..." if len(left_out) > 15 else ""))

    stretch_of = {}
    for run in cell.stretches(swc, tuple(cfg.acquisition.dendrite_swc_types)):
        for t, r in enumerate(run):
            stretch_of[int(swc.ids[r])] = (run, t)
    nodes, infos, pilots, halfs, dropped = [], {}, {}, {}, {}
    # Pass 1: the pilot of every work-list node; round 0's features = Allen's and --mu, the work-list nodes at their
    # pilot's (d, mu) (the entry; D-130.O2); the first backgrounds are fitted on that scene (F3; D-145.3)
    prepared = []
    for nid in work:
        t0 = time.time()
        try:
            run, t = stretch_of[nid]
            full, off = cell.stretch_branch(swc, run, xyz)
            ob = node_inputs.own_branch(swc, run, xyz)
            if ob is None:
                raise ValueError("a one-node stretch")
            pilot = measure_pilot(full, off + t, provider, cfg, d_fixed="allen" if pilot_d == "allen" else None)
        except Exception as exc:                     # one node's failure never stops the run
            dropped[nid] = "%s: %s" % (type(exc).__name__, exc)
            log("node %d: dropped (%s) in %.1f s" % (nid, dropped[nid], time.time() - t0))
            continue
        prepared.append((nid, ob[0], t, pilot, time.time() - t0))
    rows0 = [dict(node_id=nid, d_um=float(p.d_hat_um), mu_per_um=float(p.mu_hat_per_um), source="pilot")
             for nid, _b, _t, p, _dt in prepared
             if math.isfinite(float(p.d_hat_um)) and float(p.d_hat_um) > 0
             and math.isfinite(float(p.mu_hat_per_um)) and float(p.mu_hat_per_um) > 0]
    if rows0:
        features0 = node_features.with_rows(features0, rows0, 0)
    fscene0 = node_features.featured_scene(scene, features0)
    log("round 0: %d of %d work-list nodes at their pilot's (d, mu); the first backgrounds on that scene"
        % (len(rows0), len(work)))
    # Pass 2: each node's patch, backgrounds, line and noise
    for nid, own_branch_, t, pilot, dt in prepared:
        t0 = time.time() - dt
        try:
            start, clipped = _clip_start(node_inputs.pilot_start(pilot, own_branch_.xyz_um[t]), cfg)
            own = node_inputs.own_ids_within(own_branch_, t, cfg.inference.tube_half_length_um)[0]   # amendment 3
            node, info = node_inputs.build_node(own_branch_, t, pilot, provider, scene, cfg, own, fscene0,
                                                theta0=start)          # own ids within U (amendment 2, F1)
        except Exception as exc:                     # one node's failure never stops the run
            dropped[nid] = "%s: %s" % (type(exc).__name__, exc)
            log("node %d: dropped (%s) in %.1f s" % (nid, dropped[nid], time.time() - t0))
            continue
        if node is None:
            dropped[nid] = "%s: %s" % (info["reason"], info.get("detail", ""))
            log("node %d: dropped (%s) in %.1f s" % (nid, dropped[nid], time.time() - t0))
            continue
        if not (math.isfinite(info["sigma_gl"]) and info["sigma_gl"] > 0):
            dropped[nid] = "no_noise_sd: the profile's noise SD %r" % (info["sigma_gl"],)
            log("node %d: dropped (%s) in %.1f s" % (nid, dropped[nid], time.time() - t0))
            continue
        info.update(pilot=pilot_fields(pilot), start_clipped=clipped, theta0=start,
                    d_allen_um=2.0 * float(own_branch_.radius_um[t]))
        nodes.append(node)
        infos[nid], pilots[nid], halfs[nid] = info, pilot, info["U_axis_um"]
        log("node %d: pilot d %.3f um, mu %.3f /um, k* %d (%s)%s; %d planes %d..%d, B %.1f..%.1f gl, noise SD %.2f "
            "gl (pixel %.2f gl x %.3f), %d own nodes, tube -%.2f..+%.2f um, support %s; %.1f s"
            % (nid, pilot.d_hat_um, pilot.mu_hat_per_um, pilot.k_star, pilot.fit_status,
               " start clipped into the refine box" if clipped else "", node.ks.size, node.ks[0], node.ks[-1],
               node.B_gl.min(), node.B_gl.max(), info["sigma_gl"], info["sigma_pixel_gl"],
               info["noise_factor"], len(node.own_ids), info["U_axis_um"][0], info["U_axis_um"][1],
               ("%.2f..%.2f" % (min(info["support_frac"]), max(info["support_frac"]))) if "support_frac" in info
               else cfg.cleaning.background_support, time.time() - t0))
    kept_examples = [i for i in examples if i in infos]
    base = dict(specimen=str(a.specimen), source=label, examples_file=os.path.abspath(a.examples),
                example_ids=kept_examples, examples_requested=examples, examples_cut=cut, work_list=work, dropped=dropped,
                pilot_diameter=pilot_d, no_clean=no_clean,
                B_gl=B, B_source=B_src, mu_per_um=mu, mu_source=mu_src,
                transform=dict(shift_full_px=list(transform["shift_full_px"]), flip_y_full_h=transform["flip_y_full_h"],
                               z0_um=transform["z0_um"]), **config_sections(cfg))
    json_path = os.path.join(a.out, "cleaning_%s.json" % a.specimen)
    if not nodes:
        write_json(json_path, dict(base, stopped="no_nodes", history=[], rounds=0, nodes={}))
        log("no work-list node could be prepared (%d dropped) -> %s" % (len(dropped), json_path))
        return 1
    sigma = float(np.median([infos[n.node_id]["sigma_gl"] for n in nodes]))
    log("%d nodes prepared (%d dropped), %d example nodes; sigma_gl %.3f gl (median of the nodes' profile SDs)"
        % (len(nodes), len(dropped), len(kept_examples), sigma))
    log_cache = _cache_logger(cfg, log)
    log_cache("round 0 (work list and preparation)")

    clock = NodeClock(len(nodes), log)
    cache = line_fast.SectionCache()
    own_on = not getattr(a, "no_own", False)             # the same-node clearance product (amendment 11)
    factory = node_inputs.predict_factory(pilots, halfs, cfg, cache)

    def timed_factory(node):
        clock.node_predict(node.node_id)
        predict = factory(node)

        def counted(d, mu_, v0, dz):
            clock.calls += 1
            return predict(d, mu_, v0, dz)
        return counted

    model_bg = cfg.cleaning.background_support == "model"
    bg_fn = node_inputs.background_factory(cfg, log=log) if model_bg else None   # amendment 4, D-144

    def timed_background(node, F):
        clock.node_start(getattr(node, "node_id", None))      # named from the start: an error before the refinement
        t_b = time.time()
        out = bg_fn(node, F)
        clock.bg_seconds = time.time() - t_b
        return out

    def timed_shadow(fscene, own_ids, left, top, width, height, planes_z, cfg_, B_gl, include):
        if not model_bg:
            clock.node_start()
        return shadow.shadow_image(fscene, own_ids, left, top, width, height, planes_z, cfg_, B_gl=B_gl,
                                   include=include).S

    line_rounds = {}                                # example id -> {round: cleaned line}, for the figures (D-149.4)

    def write_outputs(res):
        """The run's files from an IterationResult: after every round (D-148.4) and at the end."""
        feats = res.features
        t_now = len(res.history)
        pos = feats.index_of()
        own = {}
        feat_path = os.path.join(a.out, "features_%s.csv" % a.specimen)
        features_io.write_features(feat_path, feats)
        per_node = {}
        for n in nodes:
            nid = n.node_id
            k = pos[nid]
            tiers = np.asarray(res.tiers[nid])
            fit = dict(res.fits[nid])
            rec = dict(infos[nid], status=fit.get("status"), refine=fit, flags=str(feats.flags[k]),
                       is_example=nid in kept_examples, d_um=float(feats.d_um[k]), mu_per_um=float(feats.mu_per_um[k]),
                       offset_um=feats.offset_um[k], source=str(feats.source[k]),
                       n_tier1=int(np.sum(tiers == 1)), n_tier2=int(np.sum(tiers == 2)))
            rec["class"] = classes.get(nid, "")
            rec["B_gl_last"] = np.asarray(res.B_gl.get(nid, n.B_gl))
            if model_bg:                                   # the backgrounds per round (amendment 4)
                rec["background_rounds"] = [dict(round=r + 1, **h) for r, h in enumerate(bg_fn.history.get(nid, []))]
            if own_on:                                     # the same-node clearance (amendment 11, D-156.2)
                own[nid] = None
                if str(feats.source[k]) != "refined" or fit.get("status") == "failed":
                    rec["own"] = own_record("not_refined", cfg)
                else:
                    try:
                        off = np.asarray(feats.offset_um[k], dtype=float)
                        th = (float(feats.d_um[k]), float(feats.mu_per_um[k]),
                              float(off[0] * n.y_hat[0] + off[1] * n.y_hat[1]), float(off[2]))
                        B_now = np.asarray(res.B_gl.get(nid, n.B_gl), dtype=float)
                        st, split, clr, v_axis = node_inputs.own_product(
                            n, pilots[nid], halfs[nid], th, res.profiles_after[nid], B_now, cfg, cache,
                            layers=nid in kept_examples)
                        rec["own"] = own_record(st, cfg, clr)
                        if st == node_inputs.OWN_OK:
                            own[nid] = (split, clr, v_axis, B_now)
                    except Exception as exc:               # the product never stops the rounds
                        rec["own"] = own_record("error", cfg, detail="%s: %s" % (type(exc).__name__, exc))
                        log("node %d: no same-node clearance (%s: %s)" % (nid, type(exc).__name__, exc))
            per_node[str(nid)] = rec
        write_json(json_path, dict(base, history=res.history, stopped=res.stopped, rounds=res.rounds, sigma_gl=sigma,
                                   nodes=per_node,
                                   counts=dict(work=len(work), prepared=len(nodes), dropped=len(dropped),
                                               tier1=int(sum(v["n_tier1"] for v in per_node.values())),
                                               tier2=int(sum(v["n_tier2"] for v in per_node.values())))))
        log("wrote %s, %s" % (feat_path, json_path))

        from allen_image_io import CropFrame
        from allen_diameter.plotting import cleaning_figures as CF
        for n in nodes:
            nid = n.node_id
            if nid not in kept_examples:
                continue
            r = float(np.max(np.abs(n.v)))
            line_xy = np.array([n.origin_xy - r * n.y_hat, n.origin_xy + r * n.y_hat])
            line_rounds.setdefault(nid, {})[t_now] = np.array(res.profiles_after[nid], dtype=float)
            t_saved = sorted(line_rounds[nid])
            npz = os.path.join(a.out, "cleaning_%s_node%d.npz" % (a.specimen, nid))
            np.savez(npz, block=np.asarray(n.block), shadow_gl=res.shadow[nid], cleaned=res.cleaned[nid], ks=n.ks,
                     B_gl=np.asarray(res.B_gl.get(nid, n.B_gl)), B0_gl=n.B_gl, v=n.v, profiles_before=res.profiles_before[nid], profiles_after=res.profiles_after[nid],
                     profiles_after_rounds=np.stack([line_rounds[nid][t] for t in t_saved]),
                     rounds_saved=np.asarray(t_saved, dtype=np.int64),
                     origin_xy=n.origin_xy, y_hat=n.y_hat, e_u=n.e_u, n_avg=n.n_avg, step=n.step, line_xy=line_xy,
                     left=n.left, top=n.top, res0_um=res0, planes_z=n.planes_z, node_xyz=n.node_xyz,
                     own_ids=np.asarray(n.own_ids, dtype=np.int64),
                     **(own_arrays(own[nid][0], own[nid][1]) if own.get(nid) else {}))
            k_star = int(pilots[nid].k_star)
            planes = [k for k in (k_star - 2, k_star, k_star + 2) if k in set(int(x) for x in n.ks)]
            meta = dict(node_id=nid, round=res.rounds, features=dict(d_um=float(feats.d_um[pos[nid]]),
                                                                        mu_per_um=float(feats.mu_per_um[pos[nid]])))
            if nid in classes:
                meta["class"] = classes[nid]
            try:
                if res.rounds < 1:                      # --no-clean: nothing was cleaned, no figure of it
                    raise RuntimeError("nothing cleaned")
                import matplotlib.pyplot as plt
                fig = CF.figure_node(n.block, res.shadow[nid], res.cleaned[nid], n.ks, CropFrame(n.left, n.top, 0, res0),
                                     line_xy, n.v, res.profiles_before[nid], res.profiles_after[nid], meta, planes=planes)
                png = CF.save(fig, os.path.join(a.out, "cleaning_node%d.png" % nid), DPI)
                CF.save(fig, os.path.join(a.out, "cleaning_node%d_round%d.png" % (nid, t_now)), DPI)   # D-149.4
                plt.close(fig)
                fig = CF.figure_line_rounds(n.v, n.ks, res.profiles_before[nid], line_rounds[nid],
                                            dict(meta, round="1-%d" % t_now if t_now > 1 else 1), planes=planes)
                png_l = CF.save(fig, os.path.join(a.out, "cleaning_line_node%d.png" % nid), DPI)
                plt.close(fig)
                png = "%s, %s" % (png, png_l)
            except Exception as exc:
                png = "no figure (%s: %s)" % (type(exc).__name__, exc)
            if own.get(nid):
                try:
                    split, clr, v_axis, B_now = own[nid]
                    own_meta = dict(node_id=nid, round=t_now, d_um=float(feats.d_um[pos[nid]]),
                                    mu_per_um=float(feats.mu_per_um[pos[nid]]))
                    if nid in classes:
                        own_meta["class"] = classes[nid]
                    files = own_figures(a.out, nid, n, res.profiles_after[nid], B_now, split, clr, v_axis,
                                        own_meta, k_star, cfg, DPI, round_index=t_now)
                    png = "%s, %s" % (png, ", ".join(files))
                except Exception as exc:
                    png = "%s, no own figure (%s: %s)" % (png, type(exc).__name__, exc)
            log("example node %d: %s, %s" % (nid, npz, png))
        if not res.history:                              # --no-clean: no round to draw
            return ""
        fig = CF.figure_rounds(res.history, cfg)
        png = CF.save(fig, os.path.join(a.out, "cleaning_rounds.png"), DPI)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return png

    saved = {"round": 0}

    def on_round(res):
        h = res.history[-1]
        log("round %d: max relative d change %.4g, largest example line change %.4g gl, %d refined, %d failed, "
            "%d at a bound" % (h["round"], h["max_rel_d"], h["max_line_change_gl"], h["n_refined"], h["n_failed"],
                               h["n_at_bound"]))
        log_cache("round %d" % h["round"])
        try:
            write_outputs(res)
            saved["round"] = int(h["round"])
            log("round %d saved (%s)" % (h["round"], res.stopped))
        except Exception as exc:                       # a failed save never stops the rounds
            log("round %d: the outputs could not be written (%s: %s)" % (h["round"], type(exc).__name__, exc))

    if no_clean:                                         # amendment 16 (D-169.7): the record, nothing cleaned
        res = clean_iteration.prepared_only(features0, scene, nodes, cfg, keep_ids=kept_examples)
        write_outputs(res)
        clock.finish()
        log("--no-clean: %d example nodes recorded with their lines as read (stopped %s); done"
            % (len(kept_examples), res.stopped))
        return 0
    t_run = time.time()
    try:
        res = clean_iteration.run_rounds(features0, scene, nodes, cfg, sigma, timed_factory,
                                         example_ids=kept_examples, shadow_fn=timed_shadow,
                                         background_fn=timed_background if model_bg else None, on_round=on_round)
    except Exception as exc:
        clock.finish()
        if saved["round"] > 0:
            log("the rounds stopped on an error (%s: %s); round %d stays saved in %s"
                % (type(exc).__name__, exc, saved["round"], json_path))
            return 1
        write_json(json_path, dict(base, stopped="error", error="%s: %s" % (type(exc).__name__, exc), history=[],
                                   rounds=0, nodes={str(k): v for k, v in infos.items()}))
        log("the rounds stopped on an error (%s: %s) -> %s" % (type(exc).__name__, exc, json_path))
        return 1
    clock.finish()
    log("rounds: %d, stopped %s, in %.1f s; %d cross-sections cached (%d hits, %d misses)"
        % (res.rounds, res.stopped, time.time() - t_run, len(cache), cache.hits, cache.misses))
    log("wrote the outputs of round %d; done" % res.rounds)
    return 0


# ------------------------------------------------------------------- main ---

def _common(p):
    p.add_argument("--specimen", required=True, help="the specimen id (any label with --synthetic); in file names")
    p.add_argument("--out", required=True, help="the output folder")
    p.add_argument("--swc", default="", help="a local SWC file instead of Allen's (real cell)")
    p.add_argument("--cache-dir", default="allen_cache", help="allen_image_io's cache (real cell)")
    p.add_argument("--shift-x", type=float, default=None, help="SWC -> image transform: full-resolution px (Cell 12)")
    p.add_argument("--shift-y", type=float, default=None)
    p.add_argument("--flip-h", type=float, default=None, help="full-resolution image height when y is flipped")
    p.add_argument("--z0", type=float, default=None, help="um subtracted from the SWC z")
    p.add_argument("--config-json", default="", help="a configuration JSON (config_from_dict); default: the defaults")
    p.add_argument("--synthetic", action="store_true", help="the smoke tests' synthetic cell, no network (dry run)")
    p.add_argument("--sigma-r0-um", type=float, default=None, help="the kernel's in-focus width (um): the table "
                   "re-anchored at it, its growth with defocus kept (Block 36, amendment 14; default: the "
                   "configuration's renderer.sigma_r0_um)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey", help="the contamination survey and the example nodes (Block 27)")
    _common(s)
    s.add_argument("--B", type=float, required=True, help="the background, gl (e.g. the pilot table's median B_bar)")
    s.add_argument("--mu", type=float, required=True, help="the coefficient, 1/um (e.g. the pilot table's median mu_hat)")
    s.add_argument("--seed", type=int, default=None, help="pick_examples' seed; default phantom.seed")
    s.add_argument("--max-nodes", type=int, default=None, help="survey a seeded random sample of this many dendrite "
                   "nodes (default: all; about 3 s per node on the pilot)")
    s.add_argument("--checkpoint-every", type=int, default=100, help="rewrite the outputs every this many nodes")
    k = sub.add_parser("rank", help="rank the dendrite nodes by their worst tier-1 neighbour on the line (D-150.2)")
    _common(k)
    k.add_argument("--B", type=float, required=True, help="the background, gl (as survey)")
    k.add_argument("--mu", type=float, required=True, help="the coefficient, 1/um (as survey)")
    k.add_argument("--seed", type=int, default=None, help="the sample's seed; default phantom.seed")
    k.add_argument("--max-nodes", type=int, default=None, help="rank a seeded random sample of this many dendrite "
                   "nodes (default: all)")
    e = sub.add_parser("clear", help="pick the clearest nodes of rank's table (Block 36, amendment 13)")
    _common(e)
    e.add_argument("--n", type=int, default=3, help="how many nodes")
    e.add_argument("--min-soma-um", type=float, default=20.0, help="least distance from the soma in the image plane (um)")
    e.add_argument("--d-bins", default="", help="edges (um) of bins of Allen's diameter, e.g. 0.4,0.6,0.8,1.2,5: "
                   "--n nodes per bin (amendment 15); default: no bins")
    c = sub.add_parser("clean", help="the cleaning rounds on the work list (Blocks 25-32)")
    _common(c)
    c.add_argument("--examples", required=True, help="examples_<S>.json of survey")
    c.add_argument("--nodes-csv", default="", help="a per-node table of the first stage (run_cell.py's nodes_<S>.csv "
                   "or the pilot table): the medians of its pilots' B_bar and mu_hat replace the examples file's B, mu")
    c.add_argument("--B", type=float, default=None, help="gl; default: --nodes-csv's median, else the examples file's")
    c.add_argument("--mu", type=float, default=None, help="1/um, round 0's coefficient; default as --B")
    c.add_argument("--rounds", type=int, default=None, help="cleaning.max_rounds")
    c.add_argument("--work-peak-gl", type=float, default=None, help="cleaning.work_peak_min_gl")
    c.add_argument("--max-work", type=int, default=None, help="cleaning.work_max_nodes")
    c.add_argument("--no-own", action="store_true", help="skip the same-node clearance product (Block 36, "
                   "amendment 11); the features and the fits are the same either way")
    c.add_argument("--no-clean", action="store_true", help="the clear-node path (Block 36, amendment 16; D-169): "
                   "prepare the example nodes only and record their lines as read; no neighbour is cleaned, no "
                   "round is run. Then infer_nodes.py real fits them")
    c.add_argument("--pilot-diameter", choices=("allen", "free"), default=None, help="the first stage's diameter: "
                   "allen = held at Allen's, the absorption and the centre fitted (D-173); free = fitted. Default: "
                   "allen with --no-clean, free otherwise")
    a = ap.parse_args(argv)
    log = Log()
    cfg = load_config(a.config_json)
    log("clean_cell.py %s, specimen %s%s" % (a.cmd, a.specimen, " (synthetic)" if a.synthetic else ""))
    a.cfg_data = cfg                                # the synthetic cell is rendered with the file's kernel
    if a.sigma_r0_um is not None:                   # amendment 14 (D-165): the model's in-focus width, a tuning knob
        ren = kernel.with_in_focus_width(cfg.renderer, a.sigma_r0_um)
        log("kernel: in-focus width %.4g um (--sigma-r0-um; the configuration's %.4g um), table %s um at %s um"
            % (ren.sigma_r0_um, cfg.renderer.sigma_r0_um, ["%.4f" % s for s in ren.kernel_table_sigma_um],
               ["%.2f" % d for d in ren.kernel_table_delta_um]))
        cfg = dataclasses.replace(cfg, renderer=ren)
        cfg.validate()
    if a.cmd == "survey":
        return run_survey(a, cfg, log)
    if a.cmd == "rank":
        return run_rank(a, cfg, log)
    if a.cmd == "clear":
        return run_clear(a, cfg, log)
    return run_clean(a, cfg, log)


if __name__ == "__main__":
    sys.exit(main())
