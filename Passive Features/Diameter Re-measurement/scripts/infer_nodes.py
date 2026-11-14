#!/usr/bin/env python3
"""The per-node fit of the pilot run -- Block 36 in specs/SPEC.md (plan step F5; D-129.1, D-130.O9,
D-132.2, D-137, D-138).

  real    for each example node of cleaning_<S>.json (scripts/clean_cell.py clean): Block 30's
          fit_node of node_inputs.forward_factory (Block 35's fast line renderer, the
          absorption inference.absorption) on the node's cleaned profiles (the last round's,
          cleaning_<S>_node<id>.npz), sigma = the node's noise SD, the start (d, mu, v0,
          z_t + dz) from features_<S>.csv (shifts about the trace, amendment 1), the planes by select_planes; a node with at least
          one tier-2 neighbour is fitted alone and flagged tier2_unmodelled (the joint fit of
          D-132.2 is plan step F3, not built) -> inference_<S>.csv, inference_<S>.json,
          fit_node<id>.png, diameters.png, chi2.png in --out

The configuration is the cleaning run's (the full signature and the two sections stored in
cleaning_<S>.json) unless --config-json is given. Every node prints one line with the clock
time; a node whose fit fails is reported and counted, never the whole run.
--synthetic is accepted for symmetry with clean_cell.py: this step reads only files.

Example
    python scripts/infer_nodes.py real --specimen 529878215 --cleaning-dir OUT/cleaning --out OUT/inference

Pure ASCII, LF, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import types as _types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from allen_diameter.analysis import node_inference, node_inputs  # noqa: E402
from allen_diameter.analysis.clean_iteration import NodeData  # noqa: E402
from allen_diameter.config import config_from_dict  # noqa: E402
from allen_diameter.loading import features_io  # noqa: E402
from allen_diameter.model import line_fast  # noqa: E402
from build_table import load_config  # noqa: E402
from clean_cell import Log, write_csv, write_json  # noqa: E402

DPI = 110
FLAG_TIER2 = "tier2_unmodelled"
INFERENCE_COLUMNS = ("node_id", "class", "d_allen_um", "d_pilot_um", "d_refined_um", "d_hat_um", "se_d_um",
                     "mu_hat_per_um", "se_mu_per_um", "v0_hat_um", "se_v0_um", "z_ax_hat_um", "se_z_ax_um", "cost",
                     "chi2_mean", "planes_used", "n_planes_used", "status", "at_bound", "flags", "seconds")


def run_config(rec, config_json, starts):
    """The fit's configuration: --config-json, else the cleaning run's (its full signature with its cleaning and
    inference sections); --starts replaces inference.multistart_factors."""
    if config_json:
        cfg = load_config(config_json)
    else:
        d = dict(rec["config"])
        d["cleaning"], d["inference"] = rec["cleaning"], rec["inference"]
        cfg = config_from_dict(d)
    if starts:
        f = tuple(float(x) for x in starts.split(","))
        cfg = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, multistart_factors=f))
    cfg.validate()
    return cfg


def node_from_npz(nid, z, theta0):
    """The NodeData of an example node from its cleaning_<S>_node<id>.npz (the observed block, the line)."""
    return NodeData(node_id=int(nid), own_ids=tuple(int(i) for i in z["own_ids"]), node_xyz=z["node_xyz"],
                    block=z["block"], B_gl=z["B_gl"], ks=z["ks"], planes_z=z["planes_z"], left=int(z["left"]),
                    top=int(z["top"]), origin_xy=z["origin_xy"], y_hat=z["y_hat"], e_u=z["e_u"], v=z["v"],
                    n_avg=int(z["n_avg"]), step=float(z["step"]), theta0=theta0)


def fit_start(source, d_um, mu_per_um, offset_um, pilot, node_xyz):
    """(theta0 = (d, mu, v0, z_ax), which) of Block 30's fit, shifts about the trace (amendment 1, D-141):
    a "refined" features row starts at its (d, mu) and its offset (v0 = offset . y_hat, z_ax = z_t +
    offset_z); any other row at node_inputs.pilot_start, the first-stage fit about the trace, z_ax = z_t +
    (z_sub - z_t) (amendment 2, F4)."""
    z_t = float(np.asarray(node_xyz, dtype=float)[2])
    if source == "refined":
        th = float(pilot.theta_rad)
        off = np.asarray(offset_um, dtype=float)
        v0 = float(off[0] * (-math.sin(th)) + off[1] * math.cos(th))
        return np.array([float(d_um), float(mu_per_um), v0, z_t + float(off[2])]), "features"
    st = node_inputs.pilot_start(pilot, node_xyz)
    return np.array([st[0], st[1], st[2], z_t + st[3]]), "pilot_start"


def _num(x):
    return float("nan") if x is None else float(x)


def run_real(a, log):
    rec_path = os.path.join(a.cleaning_dir, "cleaning_%s.json" % a.specimen)
    with open(rec_path, encoding="utf-8") as f:
        rec = json.load(f)
    cfg = run_config(rec, a.config_json, a.starts)
    feats = features_io.read_features(os.path.join(a.cleaning_dir, "features_%s.csv" % a.specimen))
    pos = feats.index_of()
    os.makedirs(a.out, exist_ok=True)
    ids = [int(i) for i in rec.get("example_ids", [])]
    log("cleaning run %s: %d example nodes, stopped %s after %d rounds; fit absorption %s, %d x %d directions, starts %s"
        % (rec_path, len(ids), rec.get("stopped"), int(rec.get("rounds", 0)), cfg.inference.absorption,
           cfg.inference.ray_n_rho, cfg.inference.ray_n_psi, tuple(cfg.inference.multistart_factors)))
    from allen_diameter.plotting import inference_figures as IF
    import matplotlib.pyplot as plt
    cache = line_fast.SectionCache()
    rows, per_node, n_failed = [], {}, 0
    for nid in ids:
        t0 = time.time()
        info = rec["nodes"][str(nid)]
        pil = info["pilot"]
        pilot = _types.SimpleNamespace(cx_um=_num(pil["cx_um"]), cy_um=_num(pil["cy_um"]),
                                       theta_rad=_num(pil["theta_rad"]), phi_rad=_num(pil["phi_rad"]),
                                       z_sub_um=_num(pil["z_sub_um"]), d_hat_um=_num(pil.get("d_hat_um")),
                                       mu_hat_per_um=_num(pil.get("mu_hat_per_um")))
        row = {"node_id": nid, "class": info.get("class", ""), "d_allen_um": _num(info.get("d_allen_um")),
               "d_pilot_um": _num(pil.get("d_hat_um")), "status": "failed", "at_bound": "", "planes_used": "",
               "n_planes_used": 0}
        flags = [f for f in str(info.get("flags", "")).split(";") if f]
        if int(info.get("n_tier2", 0)) > 0:
            flags.append(FLAG_TIER2)
        try:
            k = pos[nid]
            off = feats.offset_um[k]
            row["d_refined_um"] = float(feats.d_um[k])
            with np.load(os.path.join(a.cleaning_dir, "cleaning_%s_node%d.npz" % (a.specimen, nid))) as z:
                node = node_from_npz(nid, z, None)
                prof = np.array(z["profiles_after"], dtype=float)
            theta0, row["start"] = fit_start(str(feats.source[k]), float(feats.d_um[k]), float(feats.mu_per_um[k]),
                                             off, pilot, node.node_xyz)
            sigma = float(info["sigma_gl"])
            forward = node_inputs.forward_factory(pilot, float(info["half_length_um"]), node, cfg, cache)
            fit = node_inference.fit_node(prof, node.B_gl, sigma, forward, theta0, cfg)
        except Exception as exc:                      # one node's failure never stops the run
            n_failed += 1
            row.update(flags=";".join(flags + ["fit_error"]), seconds=time.time() - t0)
            rows.append(row)
            per_node[str(nid)] = dict(error="%s: %s" % (type(exc).__name__, exc))
            log("node %d: fit not run (%s: %s), %.1f s" % (nid, type(exc).__name__, exc, time.time() - t0))
            continue
        th, se = fit.theta, fit.se
        used = [int(kk) for kk, u in zip(node.ks, fit.planes) if u]
        row.update(d_hat_um=th[0], se_d_um=se[0], mu_hat_per_um=th[1], se_mu_per_um=se[1], v0_hat_um=th[2],
                   se_v0_um=se[2], z_ax_hat_um=th[3], se_z_ax_um=se[3], cost=fit.cost,
                   chi2_mean=float(np.nanmean(fit.chi2_per_plane)) if np.any(np.isfinite(fit.chi2_per_plane))
                   else float("nan"), planes_used=";".join(str(x) for x in used), n_planes_used=len(used),
                   status=fit.status, at_bound=";".join(fit.at_bound), flags=";".join(flags),
                   seconds=time.time() - t0)
        rows.append(row)
        n_failed += fit.status == "failed"
        per_node[str(nid)] = dict(start=row["start"], chi2_per_plane=fit.chi2_per_plane, ks=node.ks, planes=fit.planes, theta=th,
                                  cov=fit.cov, se=se, theta0=theta0, sigma_gl=sigma, n_starts=fit.n_starts,
                                  nfev=fit.nfev, status=fit.status)
        log("node %d: d %.4f +- %.4f um (Allen %.3f, pilot %.3f, refined %.3f), mu %.4g /um, v0 %.3f um, z_ax %.3f um, "
            "%d planes, mean chi2/N %.3g, %s%s; %d evaluations, %.1f s"
            % (nid, th[0], se[0], row["d_allen_um"], row["d_pilot_um"], row["d_refined_um"], th[1], th[2], th[3],
               len(used), row["chi2_mean"], fit.status, (" [" + row["flags"] + "]") if row["flags"] else "",
               fit.nfev, row["seconds"]))
        if fit.status != "failed" and np.all(np.isfinite(th)):
            try:
                model = node.B_gl[:, None] * forward(*th)
                fig = IF.figure_fit(prof, model, node.v, node.ks, sigma, fit,
                                    dict(node_id=nid, **({"class": row["class"]} if row["class"] else {})))
                IF.save(fig, os.path.join(a.out, "fit_node%d.png" % nid), DPI)
                plt.close(fig)
            except Exception as exc:
                log("node %d: no figure (%s: %s)" % (nid, type(exc).__name__, exc))
    csv_path = write_csv(os.path.join(a.out, "inference_%s.csv" % a.specimen), rows, INFERENCE_COLUMNS)
    json_path = write_json(os.path.join(a.out, "inference_%s.json" % a.specimen),
                           dict(specimen=str(a.specimen), cleaning_dir=os.path.abspath(a.cleaning_dir),
                                cleaning=dataclasses.asdict(cfg.cleaning), inference=dataclasses.asdict(cfg.inference),
                                config=json.loads(cfg.to_json()), nodes=per_node, n_failed=int(n_failed)))
    fitted = [r for r in rows if math.isfinite(_num(r.get("d_hat_um")))]
    if fitted:
        drows = [dict(node_id=r["node_id"], d_allen_um=r["d_allen_um"], d_fit_um=r["d_hat_um"], se_um=r["se_d_um"],
                      status=r["status"], d_pilot_um=r["d_pilot_um"], **({"class": r["class"]} if r["class"] else {}))
                 for r in fitted]
        fig = IF.figure_diameters(drows, cfg)
        IF.save(fig, os.path.join(a.out, "diameters.png"), DPI)
        plt.close(fig)
        crow = [dict(node_id=int(k), chi2_per_plane=np.asarray(v["chi2_per_plane"], dtype=float),
                     ks=np.asarray(v["ks"])) for k, v in per_node.items() if "chi2_per_plane" in v]
        fig = IF.figure_chi2(crow)
        IF.save(fig, os.path.join(a.out, "chi2.png"), DPI)
        plt.close(fig)
    log("%d nodes fitted, %d failed -> %s, %s%s" % (len(rows) - n_failed, n_failed, csv_path, json_path,
                                                      ", diameters.png, chi2.png" if fitted else " (no figure)"))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("real", help="the per-node fit of the cleaning run's example nodes")
    r.add_argument("--specimen", required=True, help="the specimen id of the cleaning run (its file names)")
    r.add_argument("--cleaning-dir", required=True, help="the --out folder of clean_cell.py clean")
    r.add_argument("--out", required=True, help="the output folder")
    r.add_argument("--starts", default="", help="comma-separated d0 factors, replacing inference.multistart_factors")
    r.add_argument("--config-json", default="", help="a configuration JSON; default: the cleaning run's")
    r.add_argument("--synthetic", action="store_true", help="accepted for symmetry: this step reads only files")
    a = ap.parse_args(argv)
    log = Log()
    log("infer_nodes.py real, specimen %s%s" % (a.specimen, " (synthetic)" if a.synthetic else ""))
    return run_real(a, log)


if __name__ == "__main__":
    sys.exit(main())
