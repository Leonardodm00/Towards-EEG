#!/usr/bin/env python3
"""The pilot run of the cleaning on one cell -- Block 36 in specs/SPEC.md (plan steps C0, C5-C6;
D-129, D-130, D-132, D-136, D-138).

  survey  Block 27 on every dendrite node of the cell (no image is read): each node's
          contamination index and class on its own stretch (D-136.1), and near_end
          (D-138.2) -> contamination_<S>.csv; the example nodes drawn by
          contamination.pick_examples with cleaning.example_counts, seeded, never a
          near_end node (D-138.3) -> examples_<S>.json; a count per class printed
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
from allen_diameter.model import line_fast, shadow  # noqa: E402
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
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(jsonable(obj), f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")
    return path


def _cell_text(v):
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return str(v)


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([_cell_text(r.get(c, "")) for c in columns])
    return path


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
            swc, fetcher, planes, _path = synthetic_cell(tmp, cfg, d_true=SYNTH_D_TRUE_UM, mu=SYNTH_MU_PER_UM,
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

def run_survey(a, cfg, log):
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
    n_nodes = sum(len(r) for r in runs)
    t_last = time.time()
    for run in runs:
        ob = node_inputs.own_branch(swc, run, xyz)
        if ob is None:
            single += 1
            log("node %d: a one-node stretch, not surveyed (no own direction; near_end)" % int(swc.ids[run[0]]))
            continue
        branch = ob[0]
        for t in range(len(run)):
            nid = int(branch.ids[t])
            try:
                row = contamination.node_contamination(branch, t, scene, a.B, a.mu, cfg)
            except Exception as exc:          # one node's failure never stops the survey
                failed.append(nid)
                log("node %d: survey failed (%s: %s), dropped" % (nid, type(exc).__name__, exc))
                continue
            row["near_end"] = node_inputs.is_near_end(branch, t, cfg)
            rows.append(row)
            if len(rows) % 200 == 0 or time.time() - t_last > 60:
                t_last = time.time()
                log("%d of %d dendrite nodes surveyed" % (len(rows), n_nodes))
    csv_path = write_csv(os.path.join(a.out, "contamination_%s.csv" % a.specimen), rows, CONTAMINATION_COLUMNS)
    eligible = [r for r in rows if not r["near_end"]]
    rng = np.random.default_rng(int(a.seed if a.seed is not None else cfg.phantom.seed))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        examples = contamination.pick_examples(eligible, tuple(cfg.cleaning.example_counts), rng)
    for w in caught:
        log(str(w.message))
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
               **config_sections(cfg))
    ex_path = write_json(os.path.join(a.out, "examples_%s.json" % a.specimen), rec)
    for c in contamination.CLASSES:
        print("class %-15s %6d nodes (%d not near_end)" % (c, counts[c], counts_eligible[c]), flush=True)
    log("survey: %d nodes, %d near_end, %d one-node stretches skipped, %d failed -> %s; %d example nodes %s -> %s"
        % (len(rows), len(rows) - len(eligible), single, len(failed), csv_path, len(examples), examples, ex_path))
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

    def node_start(self):
        now = time.time()
        self.finish(now)
        if self.k % self.n == 0:
            self.round_t0 = now
        self.k += 1
        self.cur = [1 + (self.k - 1) // self.n, None, now, None]
        self.calls = 0

    def node_predict(self, nid):
        if self.cur is not None:
            self.cur[1], self.cur[3] = int(nid), time.time()

    def finish(self, now=None):
        if self.cur is None:
            return
        now = time.time() if now is None else now
        rnd, nid, t0, tp = self.cur
        tp = now if tp is None else tp
        self.log("round %d node %s: shadow and cleaning %.1f s, refinement %.1f s (%d model calls), %.1f s"
                 % (rnd, nid, tp - t0, now - tp, self.calls, now - t0))
        if self.k % self.n == 0:
            self.log("round %d: %d nodes in %.1f s" % (rnd, self.n, now - self.round_t0))
        self.cur = None


def _clip_start(theta0, cfg):
    """theta0 = (d, mu, 0, 0) clipped into Block 29's refine box (refine_node refuses a start outside it);
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


def run_clean(a, cfg, log):
    os.makedirs(a.out, exist_ok=True)
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
    log("B %.4g gl (%s), mu %.4g /um (%s); rounds <= %d, work peak >= %.3g gl, work list <= %d nodes; soma z %s um; "
        "fit absorption %s, %d x %d directions; shadow absorption %s"
        % (B, B_src, mu, mu_src, cfg.cleaning.max_rounds, cfg.cleaning.work_peak_min_gl, cfg.cleaning.work_max_nodes,
           "%.3f" % zs if zs is not None else "none", cfg.inference.absorption, cfg.inference.ray_n_rho,
           cfg.inference.ray_n_psi, cfg.cleaning.shadow_absorption))
    res0 = float(cfg.acquisition.res0_um)
    types = tuple(cfg.cleaning.neighbour_swc_types)
    xyz = cell.to_image_um(swc.xyz, res0, **transform)
    scene = cell.cell_scene(swc, types, res0, transform)
    features0 = node_features.initial_features(swc, types, mu)
    fscene0 = node_features.featured_scene(scene, features0)
    log("scene: %d segments, %d soma discs; %d nodes in the features table" % (scene.seg_xyz.shape[0],
                                                                                scene.soma.shape[0], len(features0)))
    work = node_inputs.work_list(examples, fscene0, swc, cfg, cfg.cleaning.work_peak_min_gl, B)
    log("work list: %d nodes (%d examples, %d neighbours' nodes)" % (len(work), len(examples), len(work) - len(examples)))

    stretch_of = {}
    for run in cell.stretches(swc, tuple(cfg.acquisition.dendrite_swc_types)):
        for t, r in enumerate(run):
            stretch_of[int(swc.ids[r])] = (run, t)
    nodes, infos, pilots, halfs, dropped = [], {}, {}, {}, {}
    for nid in work:
        t0 = time.time()
        try:
            run, t = stretch_of[nid]
            full, off = cell.stretch_branch(swc, run, xyz)
            ob = node_inputs.own_branch(swc, run, xyz)
            if ob is None:
                raise ValueError("a one-node stretch")
            own_branch_, _ = ob
            pilot = measure_pilot(full, off + t, provider, cfg)
            start, clipped = _clip_start((pilot.d_hat_um, pilot.mu_hat_per_um, 0.0, 0.0), cfg)
            node, info = node_inputs.build_node(own_branch_, t, pilot, provider, scene, cfg,
                                                tuple(int(i) for i in own_branch_.ids), theta0=start)
        except Exception as exc:                     # one node's failure never stops the run
            dropped[nid] = "%s: %s" % (type(exc).__name__, exc)
            log("node %d: dropped (%s) in %.1f s" % (nid, dropped[nid], time.time() - t0))
            continue
        if node is None:
            dropped[nid] = "%s: %s" % (info["reason"], info.get("detail", ""))
            log("node %d: dropped (%s) in %.1f s" % (nid, dropped[nid], time.time() - t0))
            continue
        if not (math.isfinite(info["sigma_gl"]) and info["sigma_gl"] > 0):
            dropped[nid] = "no_noise_sd: the planes' noise SD %r" % (info["sigma_gl"],)
            log("node %d: dropped (%s) in %.1f s" % (nid, dropped[nid], time.time() - t0))
            continue
        info.update(pilot=pilot_fields(pilot), start_clipped=clipped, theta0=start,
                    d_allen_um=2.0 * float(own_branch_.radius_um[t]))
        nodes.append(node)
        infos[nid], pilots[nid], halfs[nid] = info, pilot, info["half_length_um"]
        log("node %d: pilot d %.3f um, mu %.3f /um, k* %d (%s)%s; %d planes %d..%d, B %.1f..%.1f gl, noise SD %.2f "
            "gl, tube half-length %.2f um; %.1f s"
            % (nid, pilot.d_hat_um, pilot.mu_hat_per_um, pilot.k_star, pilot.fit_status,
               " start clipped into the refine box" if clipped else "", node.ks.size, node.ks[0], node.ks[-1],
               node.B_gl.min(), node.B_gl.max(), info["sigma_gl"], info["half_length_um"], time.time() - t0))
    kept_examples = [i for i in examples if i in infos]
    base = dict(specimen=str(a.specimen), source=label, examples_file=os.path.abspath(a.examples),
                example_ids=kept_examples, examples_requested=examples, work_list=work, dropped=dropped,
                B_gl=B, B_source=B_src, mu_per_um=mu, mu_source=mu_src,
                transform=dict(shift_full_px=list(transform["shift_full_px"]), flip_y_full_h=transform["flip_y_full_h"],
                               z0_um=transform["z0_um"]), **config_sections(cfg))
    json_path = os.path.join(a.out, "cleaning_%s.json" % a.specimen)
    if not nodes:
        write_json(json_path, dict(base, stopped="no_nodes", history=[], rounds=0, nodes={}))
        log("no work-list node could be prepared (%d dropped) -> %s" % (len(dropped), json_path))
        return 1
    rows0 = [dict(node_id=n.node_id, d_um=float(pilots[n.node_id].d_hat_um),
                  mu_per_um=float(pilots[n.node_id].mu_hat_per_um), source="pilot") for n in nodes]
    features0 = node_features.with_rows(features0, rows0, 0)
    sigma = float(np.median([infos[n.node_id]["sigma_gl"] for n in nodes]))
    log("%d nodes prepared (%d dropped), %d example nodes; sigma_gl %.3f gl (median of the nodes')"
        % (len(nodes), len(dropped), len(kept_examples), sigma))

    clock = NodeClock(len(nodes), log)
    cache = line_fast.SectionCache()
    factory = node_inputs.predict_factory(pilots, halfs, cfg, cache)

    def timed_factory(node):
        clock.node_predict(node.node_id)
        predict = factory(node)

        def counted(d, mu_, v0, dz):
            clock.calls += 1
            return predict(d, mu_, v0, dz)
        return counted

    def timed_shadow(fscene, own_ids, left, top, width, height, planes_z, cfg_, B_gl, include):
        clock.node_start()
        return shadow.shadow_image(fscene, own_ids, left, top, width, height, planes_z, cfg_, B_gl=B_gl,
                                   include=include).S

    t_run = time.time()
    try:
        res = clean_iteration.run_rounds(features0, scene, nodes, cfg, sigma, timed_factory,
                                         example_ids=kept_examples, shadow_fn=timed_shadow)
    except Exception as exc:
        clock.finish()
        write_json(json_path, dict(base, stopped="error", error="%s: %s" % (type(exc).__name__, exc), history=[],
                                   rounds=0, nodes={str(k): v for k, v in infos.items()}))
        log("the rounds stopped on an error (%s: %s) -> %s" % (type(exc).__name__, exc, json_path))
        return 1
    clock.finish()
    for h in res.history:
        log("round %d: max relative d change %.4g, largest example line change %.4g gl, %d refined, %d failed, "
            "%d at a bound" % (h["round"], h["max_rel_d"], h["max_line_change_gl"], h["n_refined"], h["n_failed"],
                               h["n_at_bound"]))
    log("rounds: %d, stopped %s, in %.1f s; %d cross-sections cached (%d hits, %d misses)"
        % (res.rounds, res.stopped, time.time() - t_run, len(cache), cache.hits, cache.misses))

    feats = res.features
    pos = feats.index_of()
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
        npz = os.path.join(a.out, "cleaning_%s_node%d.npz" % (a.specimen, nid))
        np.savez(npz, block=np.asarray(n.block), shadow_gl=res.shadow[nid], cleaned=res.cleaned[nid], ks=n.ks,
                 B_gl=n.B_gl, v=n.v, profiles_before=res.profiles_before[nid], profiles_after=res.profiles_after[nid],
                 origin_xy=n.origin_xy, y_hat=n.y_hat, e_u=n.e_u, n_avg=n.n_avg, step=n.step, line_xy=line_xy,
                 left=n.left, top=n.top, res0_um=res0, planes_z=n.planes_z, node_xyz=n.node_xyz,
                 own_ids=np.asarray(n.own_ids, dtype=np.int64))
        k_star = int(pilots[nid].k_star)
        planes = [k for k in (k_star - 2, k_star, k_star + 2) if k in set(int(x) for x in n.ks)]
        meta = dict(node_id=nid, round=res.rounds, features=dict(d_um=float(feats.d_um[pos[nid]]),
                                                                    mu_per_um=float(feats.mu_per_um[pos[nid]])))
        if nid in classes:
            meta["class"] = classes[nid]
        try:
            fig = CF.figure_node(n.block, res.shadow[nid], res.cleaned[nid], n.ks, CropFrame(n.left, n.top, 0, res0),
                                 line_xy, n.v, res.profiles_before[nid], res.profiles_after[nid], meta, planes=planes)
            png = CF.save(fig, os.path.join(a.out, "cleaning_node%d.png" % nid), DPI)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as exc:
            png = "no figure (%s: %s)" % (type(exc).__name__, exc)
        log("example node %d: %s, %s" % (nid, npz, png))
    fig = CF.figure_rounds(res.history, cfg)
    png = CF.save(fig, os.path.join(a.out, "cleaning_rounds.png"), DPI)
    import matplotlib.pyplot as plt
    plt.close(fig)
    log("wrote %s; done" % png)
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey", help="the contamination survey and the example nodes (Block 27)")
    _common(s)
    s.add_argument("--B", type=float, required=True, help="the background, gl (e.g. the pilot table's median B_bar)")
    s.add_argument("--mu", type=float, required=True, help="the coefficient, 1/um (e.g. the pilot table's median mu_hat)")
    s.add_argument("--seed", type=int, default=None, help="pick_examples' seed; default phantom.seed")
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
    a = ap.parse_args(argv)
    log = Log()
    cfg = load_config(a.config_json)
    log("clean_cell.py %s, specimen %s%s" % (a.cmd, a.specimen, " (synthetic)" if a.synthetic else ""))
    if a.cmd == "survey":
        return run_survey(a, cfg, log)
    return run_clean(a, cfg, log)


if __name__ == "__main__":
    sys.exit(main())
