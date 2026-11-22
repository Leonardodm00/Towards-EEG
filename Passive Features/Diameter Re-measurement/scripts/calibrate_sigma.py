"""The in-focus width calibrated on thick nodes -- the entry point (SPEC Block 36, amendment 19; D-201).

    python scripts/calibrate_sigma.py --cleaning-dir DIR --specimen S [--out DIR/sigma] [--nodes 7,599,...]
                                      [--workers N] [--config-json FILE]

For each example node of a preparation made by `clean_cell.py clean --no-clean` (cleaning_<S>.json and its
cleaning_<S>_node<id>.npz files): Block 42's fit (analysis.sigma_calibration) on the planes k* - half_planes ..
k* + half_planes of the recorded profiles (`profiles_before`), with the run's configuration (infer_nodes.run_config:
the record's own, or --config-json), one node per worker process, BLAS threads 1; Block 43's figures
(plotting.sigma_figures). Writes into --out: sigma_<S>.csv (one row per node), sigma_<S>.json (the fits, the
summary of the nodes that converged with neither sigma0 nor c at a bound), sigma_node<id>.png, sigma_summary_<S>.png.
Pure ASCII, LF, for the cluster.
"""
from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per worker process
    os.environ.setdefault(_var, "1")

import argparse            # noqa: E402
import dataclasses         # noqa: E402
import json                # noqa: E402
import math                # noqa: E402
import sys                 # noqa: E402
import time                # noqa: E402
import types as _types     # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed   # noqa: E402

import numpy as np         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from allen_diameter.analysis import sigma_calibration as SC   # noqa: E402
from allen_diameter.model import line_fast                    # noqa: E402

DPI = 110                  # the figures' resolution (as the gallery's)
COLUMNS = ("node_id", "d_allen_um", "k_star", "n_planes", "status", "ok", "d_um", "se_d_um", "mu_per_um",
           "se_mu_per_um", "v0_um", "z_ax_um", "sigma0_um", "se_sigma0_um", "growth_scale", "se_growth_scale",
           "chi2", "dof", "chi2_per_dof", "n_eval", "at_bound", "seconds", "message")


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def load_record(cleaning_dir, specimen):
    with open(os.path.join(cleaning_dir, "cleaning_%s.json" % specimen)) as f:
        return json.load(f)


def pilot_of(info):
    """The first stage's fit of a node's record entry, as infer_nodes.py builds it."""
    pil = info["pilot"]
    return _types.SimpleNamespace(cx_um=_num(pil["cx_um"]), cy_um=_num(pil["cy_um"]), theta_rad=_num(pil["theta_rad"]),
                                  phi_rad=_num(pil["phi_rad"]), z_sub_um=_num(pil["z_sub_um"]),
                                  d_hat_um=_num(pil.get("d_hat_um")), mu_hat_per_um=_num(pil.get("mu_hat_per_um")))


def fit_one(cleaning_dir, specimen, nid, config_json, out_dir):
    """Block 42's fit of one node and its Block 43 page. -> (row dict, arrays for the summary). Runs in a worker."""
    import infer_nodes                                   # the script's helpers: run_config, node_from_npz
    from allen_diameter.plotting import sigma_figures as SF
    from allen_diameter.plotting.figures import image_extent_um
    rec = load_record(cleaning_dir, specimen)
    cfg = infer_nodes.run_config(rec, config_json, None)
    scfg = cfg.sigma_calibration
    info = rec["nodes"][str(nid)]
    pilot = pilot_of(info)
    k_star = int(info["pilot"]["k_star"])
    with np.load(os.path.join(cleaning_dir, "cleaning_%s_node%d.npz" % (specimen, nid)), allow_pickle=False) as z:
        node = infer_nodes.node_from_npz(nid, z, None)
        prof = np.array(z["profiles_before"], dtype=float)
        res0 = float(z["res0_um"]) if "res0_um" in z.files else float(cfg.acquisition.res0_um)
    ks = np.asarray(node.ks).astype(int)
    sel = SC.select_planes(ks, k_star, scfg.half_planes)
    node_s = dataclasses.replace(node, ks=np.asarray(node.ks)[sel], planes_z=np.asarray(node.planes_z)[sel],
                                 block=np.asarray(node.block)[sel], B_gl=np.asarray(node.B_gl)[sel])
    prof_s = prof[sel]
    ks_s = ks[sel]
    k_index = int(np.argmin(np.abs(ks_s - k_star)))
    fwd = SC.forward_with_kernel(pilot, tuple(float(x) for x in info["U_axis_um"]), node_s, cfg, line_fast.SectionCache())
    theta0 = SC.start_theta(prof_s, node_s.B_gl, np.asarray(node_s.v, dtype=float), k_index, pilot, node.node_xyz, scfg)
    fit = SC.fit_sigma_node(prof_s, float(info["sigma_gl"]), np.asarray(node_s.v, dtype=float), fwd, theta0, scfg)
    th, se_s = fit.theta, fit.se_scaled
    ok = bool(fit.status == "converged" and not ({"sigma0_um", "growth_scale"} & set(fit.at_bound)))
    row = dict(node_id=int(nid), d_allen_um=_num(info.get("d_allen_um")), k_star=k_star, n_planes=int(sel.size),
               status=fit.status, ok=ok, d_um=th[0], se_d_um=se_s[0], mu_per_um=th[1], se_mu_per_um=se_s[1],
               v0_um=th[2], z_ax_um=th[3], sigma0_um=th[4], se_sigma0_um=se_s[4], growth_scale=th[5],
               se_growth_scale=se_s[5], chi2=fit.chi2, dof=fit.dof,
               chi2_per_dof=(fit.chi2 / fit.dof) if fit.dof > 0 else float("nan"), n_eval=fit.n_eval,
               at_bound=" ".join(fit.at_bound), seconds=fit.seconds, message=fit.message)
    v = np.asarray(node_s.v, dtype=float)
    background = (fit.b[:, 0:1] + fit.b[:, 1:2] * v[None, :]) if np.all(np.isfinite(fit.b)) else None
    frame = _types.SimpleNamespace(left=int(node.left), top=int(node.top), res_um_px=res0, downsample=0)
    block = np.asarray(node_s.block, dtype=float)
    text = "\n".join([
        "status     %s%s" % (fit.status, "" if ok else "  (not in the summary)"),
        "d          %.3f +- %.3f um" % (th[0], se_s[0]),
        "Allen d    %.3f um" % row["d_allen_um"],
        "mu         %.3f +- %.3f /um" % (th[1], se_s[1]),
        "sigma0     %.4f +- %.4f um" % (th[4], se_s[4]),
        "c          %.3f +- %.3f" % (th[5], se_s[5]),
        "v0, z_ax   %.3f, %.3f um" % (th[2], th[3]),
        "chi2/dof   %.3f (%d samples)" % (row["chi2_per_dof"], fit.n_res),
        "evals      %d in %.0f s" % (fit.n_eval, fit.seconds),
        "at bound   %s" % (row["at_bound"] or "none")])
    node_map = dict(images=block, extent=image_extent_um(frame, block.shape), ks=ks_s, k_index=k_index,
                    origin_xy=np.asarray(node.origin_xy, dtype=float), y_hat=np.asarray(node.y_hat, dtype=float),
                    v_um=v, profiles_gl=prof_s, model_gl=fit.model_gl, background_gl=background,
                    title="node %d  |  Allen %.3f um  |  in-focus width fitted with the hybrid as it is (D-201)"
                          % (nid, row["d_allen_um"]), text=text)
    try:
        fig = SF.figure_node_fit(node_map)
        SF.save(fig, os.path.join(out_dir, "sigma_node%d.png" % nid), DPI)
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ValueError as exc:                     # a failed fit's NaN model is drawn as gaps; a refusal is reported
        row["message"] = row["message"] + "; figure: %s" % exc
    depth = th[3] if math.isfinite(th[3]) else float(np.asarray(node.node_xyz, dtype=float)[2])
    return row, dict(depth_um=depth)


def write_csv(path, rows):
    with open(path, "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for r in rows:
            out = []
            for c in COLUMNS:
                x = r[c]
                if isinstance(x, float):
                    out.append("%.6g" % x)
                else:
                    out.append('"%s"' % str(x).replace('"', "'") if isinstance(x, str) and ("," in x or " " in x)
                               else str(x))
            f.write(",".join(out) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="the in-focus width fitted on thick nodes (SPEC Block 36 am. 19)")
    ap.add_argument("--cleaning-dir", required=True)
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--out", default=None, help="default: <cleaning-dir>/sigma")
    ap.add_argument("--nodes", default="", help="comma-separated node ids; default: the preparation's example nodes")
    ap.add_argument("--workers", type=int, default=None, help="default: sigma_calibration.workers (0 = all CPUs)")
    ap.add_argument("--config-json", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    out = a.out or os.path.join(a.cleaning_dir, "sigma")
    os.makedirs(out, exist_ok=True)
    rec = load_record(a.cleaning_dir, a.specimen)
    import infer_nodes
    cfg = infer_nodes.run_config(rec, a.config_json, None)
    scfg = cfg.sigma_calibration
    prepared = [int(i) for i in rec.get("example_ids", [])]
    ids = [int(s) for s in a.nodes.split(",") if s.strip()] if a.nodes else prepared
    ids = list(dict.fromkeys(ids))                          # each node once, in the order given
    missing = [i for i in ids if str(i) not in rec["nodes"] or not os.path.exists(
        os.path.join(a.cleaning_dir, "cleaning_%s_node%d.npz" % (a.specimen, i)))]
    if missing:
        print("calibrate_sigma: nodes not prepared in %s: %s" % (a.cleaning_dir, missing))
        return 1
    workers = a.workers if a.workers is not None else int(scfg.workers)
    workers = max(1, min(len(ids), workers if workers > 0 else (os.cpu_count() or 1)))
    print("calibrate_sigma: %d nodes %s, planes k* +- %d, %d worker(s); start sigma0 %.3f um, c %.2f; max_nfev %d"
          % (len(ids), ids, scfg.half_planes, workers, scfg.sigma0_start_um, scfg.growth_scale_start, scfg.max_nfev),
          flush=True)
    rows, extra = {}, {}
    if workers == 1:
        for nid in ids:
            rows[nid], extra[nid] = fit_one(a.cleaning_dir, a.specimen, nid, a.config_json, out)
            _report(rows[nid], t0)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fit_one, a.cleaning_dir, a.specimen, nid, a.config_json, out): nid for nid in ids}
            for fu in as_completed(futs):
                nid = futs[fu]
                rows[nid], extra[nid] = fu.result()
                _report(rows[nid], t0)
    ordered = [rows[i] for i in ids]
    sig = np.array([r["sigma0_um"] for r in ordered], dtype=float)
    se = np.array([r["se_sigma0_um"] for r in ordered], dtype=float)
    ok = np.array([bool(r["ok"]) and math.isfinite(r["sigma0_um"]) and math.isfinite(r["se_sigma0_um"])
                   and r["se_sigma0_um"] > 0 for r in ordered], dtype=bool)
    summary = SC.summarize(np.where(ok, sig, np.nan), np.where(ok, se, np.nan), ok)
    write_csv(os.path.join(out, "sigma_%s.csv" % a.specimen), ordered)
    with open(os.path.join(out, "sigma_%s.json" % a.specimen), "w") as f:
        json.dump(dict(specimen=a.specimen, cleaning_dir=a.cleaning_dir, nodes=ids, config=dataclasses.asdict(scfg),
                       summary=summary, fits=ordered, seconds=time.time() - t0), f, indent=1, default=float)
    from allen_diameter.plotting import sigma_figures as SF
    srows = [dict(node_id=r["node_id"], sigma0_um=r["sigma0_um"], se_sigma0_um=r["se_sigma0_um"],
                  growth_scale=r["growth_scale"], se_growth_scale=r["se_growth_scale"], d_hat_um=r["d_um"],
                  d_allen_um=r["d_allen_um"], depth_um=extra[r["node_id"]]["depth_um"], ok=bool(ok[k]))
             for k, r in enumerate(ordered)]
    fig = SF.figure_summary(srows, summary)
    SF.save(fig, os.path.join(out, "sigma_summary_%s.png" % a.specimen), DPI)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("summary over %d node(s) that converged, sigma0 and c inside their bounds: median %.4f um (MAD %.4f), "
          "weighted mean %.4f +- %.4f um, chi2_cons %.2f on %s dof; %.0f s"
          % (summary["n"], summary["median_um"], summary["mad_um"], summary["ivw_mean_um"], summary["ivw_se_um"],
             summary["chi2_cons"], summary["dof"], time.time() - t0))
    print("wrote %s/sigma_%s.csv, .json, sigma_summary_%s.png and %d node pages" % (out, a.specimen, a.specimen, len(ids)))
    return 0


def _report(r, t0):
    print("[%6.0f s] node %d: %s, d %.3f +- %.3f um (Allen %.3f), sigma0 %.4f +- %.4f um, c %.3f +- %.3f, "
          "chi2/dof %.2f, %d evaluations in %.0f s%s"
          % (time.time() - t0, r["node_id"], r["status"], r["d_um"], r["se_d_um"], r["d_allen_um"], r["sigma0_um"],
             r["se_sigma0_um"], r["growth_scale"], r["se_growth_scale"], r["chi2_per_dof"], r["n_eval"], r["seconds"],
             "" if r["ok"] else "  (not in the summary: %s%s)" % (r["status"], (", at bound " + r["at_bound"])
                                                                if r["at_bound"] else "")), flush=True)


if __name__ == "__main__":
    sys.exit(main())
