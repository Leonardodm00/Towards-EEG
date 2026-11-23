"""The in-focus width calibrated on thick nodes -- the entry point (SPEC Block 36, amendment 19; D-201).

    python scripts/calibrate_sigma.py --cleaning-dir DIR --specimen S [--out DIR/sigma] [--nodes 7,599,...]
                                      [--workers N] [--config-json FILE]

For each example node of a preparation made by `clean_cell.py clean --no-clean` (cleaning_<S>.json and its
cleaning_<S>_node<id>.npz files): Block 42's fit (analysis.sigma_calibration) on the planes k* - half_planes ..
k* + half_planes of the recorded profiles (`profiles_before`), with the run's configuration (the preparation's own,
infer_nodes.run_config; a --config-json file's sections are merged over it field by field: SPEC Block 36 amendment
20, F5), one node per worker process, BLAS threads 1; Block 43's figures (plotting.sigma_figures). The log and every
page name the forward model used. A node that raises is recorded (status "error", its message) and the run goes on;
the CSV and the JSON are rewritten after every node (amendment 20, F4). Writes into --out: sigma_<S>.csv (one row per
node: `ok` = converged with neither sigma0 nor c at a bound, `in_summary` = ok with a finite sigma0 and a finite SE
> 0, S3), sigma_<S>.json (strict JSON, NaN written as null, F3: the fits, the summary of the in_summary nodes),
sigma_node<id>.png, sigma_summary_<S>.png. Pure ASCII, LF, for the cluster.
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
COLUMNS = ("node_id", "d_allen_um", "k_star", "n_planes", "status", "ok", "in_summary", "d_um", "se_d_um", "mu_per_um",
           "se_mu_per_um", "v0_um", "z_ax_um", "sigma0_um", "se_sigma0_um", "growth_scale", "se_growth_scale",
           "chi2", "dof", "chi2_per_dof", "n_eval", "at_bound", "seconds", "message")


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def run_configuration(rec, config_json):
    """The run's configuration: the preparation's own (infer_nodes.run_config without a file) with, when config_json
    is given, the file's sections merged over it field by field, so that a file holding only `sigma_calibration`
    leaves the forward model as prepared (SPEC Block 36 amendment 20; final check of wave 4, F5)."""
    import dataclasses as _dc
    import infer_nodes
    from allen_diameter.config import config_from_dict
    base = infer_nodes.run_config(rec, None, None)
    if not config_json:
        return base
    with open(config_json) as f:
        over = json.load(f)
    d = _dc.asdict(base)
    for sec, val in over.items():
        if sec not in d:
            raise ValueError("--config-json: unknown section %r" % (sec,))
        d[sec] = dict(d[sec], **val) if isinstance(val, dict) else val
    cfg = config_from_dict(d)
    cfg.validate()
    return cfg


def forward_model_text(cfg):
    """The forward model a fit uses, in words, for the log and the pages."""
    i, r = cfg.inference, cfg.renderer
    return "%s line model, %d x %d directions, kernel sigma_r(0) %.4f um" % (i.absorption, int(i.ray_n_rho),
                                                                          int(i.ray_n_psi), float(r.sigma_r0_um))


def _json_safe(x):
    """x with every non-finite float replaced by None and numpy scalars made Python ones (strict JSON, F3)."""
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def error_row(rec, nid, exc):
    """The CSV/JSON row of a node whose fit raised (F4): status "error", the exception in `message`."""
    info = rec.get("nodes", {}).get(str(nid), {})
    nan = float("nan")
    row = {c: nan for c in COLUMNS}
    row.update(node_id=int(nid), d_allen_um=_num(info.get("d_allen_um")),
               k_star=_num((info.get("pilot") or {}).get("k_star")), n_planes=0, status="error", ok=False,
               in_summary=False, dof=0, n_eval=0, at_bound="", message="%s: %s" % (type(exc).__name__, exc))
    return row


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
    import infer_nodes                                   # the script's helpers: node_from_npz
    from allen_diameter.plotting import sigma_figures as SF
    from allen_diameter.plotting.figures import image_extent_um
    rec = load_record(cleaning_dir, specimen)
    cfg = run_configuration(rec, config_json)
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
    in_sum = bool(ok and math.isfinite(th[4]) and math.isfinite(se_s[4]) and se_s[4] > 0)      # S3
    row = dict(node_id=int(nid), d_allen_um=_num(info.get("d_allen_um")), k_star=k_star, n_planes=int(sel.size),
               status=fit.status, ok=ok, in_summary=in_sum, d_um=th[0], se_d_um=se_s[0], mu_per_um=th[1], se_mu_per_um=se_s[1],
               v0_um=th[2], z_ax_um=th[3], sigma0_um=th[4], se_sigma0_um=se_s[4], growth_scale=th[5],
               se_growth_scale=se_s[5], chi2=fit.chi2, dof=fit.dof,
               chi2_per_dof=(fit.chi2 / fit.dof) if fit.dof > 0 else float("nan"), n_eval=fit.n_eval,
               at_bound=" ".join(fit.at_bound), seconds=fit.seconds, message=fit.message)
    v = np.asarray(node_s.v, dtype=float)
    background = (fit.b[:, 0:1] + fit.b[:, 1:2] * v[None, :]) if np.all(np.isfinite(fit.b)) else None
    frame = _types.SimpleNamespace(left=int(node.left), top=int(node.top), res_um_px=res0, downsample=0)
    block = np.asarray(node_s.block, dtype=float)
    text = "\n".join([
        "status     %s%s" % (fit.status, "" if in_sum else "  (not in the summary)"),
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
                    title="node %d  |  Allen %.3f um  |  in-focus width fitted with the %s (D-201)"
                          % (nid, row["d_allen_um"], forward_model_text(cfg)), text=text)
    try:
        fig = SF.figure_node_fit(node_map)
        SF.save(fig, os.path.join(out_dir, "sigma_node%d.png" % nid), DPI)
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ValueError as exc:                     # a failed fit's NaN model is drawn as gaps; a refusal is reported
        row["message"] = row["message"] + "; figure: %s" % exc
    depth = th[3] if math.isfinite(th[3]) else float(np.asarray(node.node_xyz, dtype=float)[2])
    return row, dict(depth_um=depth)


def write_outputs(out, a, ids, rows, scfg, summary, t0):
    """sigma_<S>.csv and sigma_<S>.json of the nodes done so far, in the order of ids (strict JSON, F3); with the
    final summary, the growth scale's beside sigma0's (Block 36 amendment 22, D-222.3: the switch to the calibrated
    kernel reads both), computed here from the rows so that the signature stays the one amendment 20 tested."""
    done = [rows[i] for i in ids if i in rows]
    summary_growth_scale = None if summary is None else growth_scale_summary(done)
    write_csv(os.path.join(out, "sigma_%s.csv" % a.specimen), done)
    with open(os.path.join(out, "sigma_%s.json" % a.specimen), "w") as f:
        json.dump(_json_safe(dict(specimen=a.specimen, cleaning_dir=a.cleaning_dir, nodes=ids,
                                  config=dataclasses.asdict(scfg), summary=summary,
                                  summary_growth_scale=summary_growth_scale, fits=done,
                                  complete=summary is not None, seconds=time.time() - t0)),
                  f, indent=1, allow_nan=False)


def growth_scale_summary(ordered):
    """The growth scale c's summary (Block 36 amendment 22): sigma_calibration.summarize over the nodes ok (converged,
    neither sigma0 nor c at a bound) with a finite c and a finite SE(c) > 0, its keys without the unit."""
    c = np.array([_num(r.get("growth_scale")) for r in ordered], dtype=float)
    se = np.array([_num(r.get("se_growth_scale")) for r in ordered], dtype=float)
    ok = np.array([bool(r.get("ok")) for r in ordered], dtype=bool)
    with np.errstate(invalid="ignore"):
        ok = ok & np.isfinite(c) & np.isfinite(se) & (se > 0)
    s = SC.summarize(np.where(ok, c, np.nan), np.where(ok, se, np.nan), ok)
    return dict(n=s["n"], median=s["median_um"], mad=s["mad_um"], ivw_mean=s["ivw_mean_um"], ivw_se=s["ivw_se_um"],
                chi2_cons=s["chi2_cons"], dof=s["dof"])


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
    cfg = run_configuration(rec, a.config_json)
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
    print("calibrate_sigma: forward model: %s%s" % (forward_model_text(cfg), "" if not a.config_json else
                                                   " (the preparation's configuration with %s merged over it)"
                                                   % a.config_json), flush=True)
    rows, extra = {}, {}

    def record(nid, result_or_exc):
        if isinstance(result_or_exc, BaseException):
            rows[nid], extra[nid] = error_row(rec, nid, result_or_exc), dict(depth_um=float("nan"))
            print("[%6.0f s] node %d: error -- %s (recorded; the run goes on)" % (time.time() - t0, nid,
                                                                                 rows[nid]["message"]), flush=True)
        else:
            rows[nid], extra[nid] = result_or_exc
            _report(rows[nid], t0)
        write_outputs(out, a, ids, rows, scfg, None, t0)        # as it goes (F4)

    if workers == 1:
        for nid in ids:
            try:
                res = fit_one(a.cleaning_dir, a.specimen, nid, a.config_json, out)
            except Exception as exc:                            # one node's failure is recorded, not fatal (F4)
                res = exc
            record(nid, res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fit_one, a.cleaning_dir, a.specimen, nid, a.config_json, out): nid for nid in ids}
            for fu in as_completed(futs):
                try:
                    res = fu.result()
                except Exception as exc:
                    res = exc
                record(futs[fu], res)
    ordered = [rows[i] for i in ids]
    sig = np.array([r["sigma0_um"] for r in ordered], dtype=float)
    se = np.array([r["se_sigma0_um"] for r in ordered], dtype=float)
    ok = np.array([bool(r["in_summary"]) for r in ordered], dtype=bool)
    summary = SC.summarize(np.where(ok, sig, np.nan), np.where(ok, se, np.nan), ok)
    summary_c = growth_scale_summary(ordered)
    write_outputs(out, a, ids, rows, scfg, summary, t0)
    from allen_diameter.plotting import sigma_figures as SF
    srows = [dict(node_id=r["node_id"], sigma0_um=r["sigma0_um"], se_sigma0_um=r["se_sigma0_um"],
                  growth_scale=r["growth_scale"], se_growth_scale=r["se_growth_scale"], d_hat_um=r["d_um"],
                  d_allen_um=r["d_allen_um"], depth_um=extra[r["node_id"]]["depth_um"], ok=bool(ok[k]))
             for k, r in enumerate(ordered) if r["status"] != "error"]
    if srows:
        fig = SF.figure_summary(srows, summary)
        SF.save(fig, os.path.join(out, "sigma_summary_%s.png" % a.specimen), DPI)
        import matplotlib.pyplot as plt
        plt.close(fig)
    else:
        print("calibrate_sigma: every node failed; no summary figure")
    print("summary over %d node(s) that converged, sigma0 and c inside their bounds: median %.4f um (MAD %.4f), "
          "weighted mean %.4f +- %.4f um, chi2_cons %.2f on %s dof; %.0f s"
          % (summary["n"], summary["median_um"], summary["mad_um"], summary["ivw_mean_um"], summary["ivw_se_um"],
             summary["chi2_cons"], summary["dof"], time.time() - t0))
    print("growth scale c over %d node(s) (ok, a finite c and SE): median %.4f (MAD %.4f), weighted mean %.4f +- %.4f, "
          "chi2_cons %.2f on %s dof -- with sigma0's weighted mean, the values --kernel-width calibrated applies "
          "(D-222.3)" % (summary_c["n"], summary_c["median"], summary_c["mad"], summary_c["ivw_mean"],
                         summary_c["ivw_se"], summary_c["chi2_cons"], summary_c["dof"]))
    # Block 36 amendment 21 (D-220.3): name what was written; no node fitted -> exit 1
    failed = [str(r["node_id"]) for r in ordered if r["status"] == "error"]
    n_err = len(failed)
    n_pages = len(ordered) - n_err
    written = "%s/sigma_%s.csv, .json" % (out, a.specimen) + (", sigma_summary_%s.png" % a.specimen if srows else "")
    print("wrote %s and %d node page(s)%s" % (written, n_pages, "; %d node(s) failed: %s (status \"error\" in the CSV)"
                                               % (n_err, ", ".join(failed)) if n_err else ""))   # D-223.3: ids named
    if n_pages == 0:
        print("calibrate_sigma: no node was fitted; see the status and message columns of %s/sigma_%s.csv"
              % (out, a.specimen))
        return 1
    return 0


def _report(r, t0):
    print("[%6.0f s] node %d: %s, d %.3f +- %.3f um (Allen %.3f), sigma0 %.4f +- %.4f um, c %.3f +- %.3f, "
          "chi2/dof %.2f, %d evaluations in %.0f s%s"
          % (time.time() - t0, r["node_id"], r["status"], r["d_um"], r["se_d_um"], r["d_allen_um"], r["sigma0_um"],
             r["se_sigma0_um"], r["growth_scale"], r["se_growth_scale"], r["chi2_per_dof"], r["n_eval"], r["seconds"],
             "" if r["in_summary"] else "  (not in the summary: %s%s%s)"
             % (r["status"], (", at bound " + r["at_bound"]) if r["at_bound"] else "",
                ", SE not finite" if (r["ok"] and not r["in_summary"]) else "")), flush=True)


if __name__ == "__main__":
    sys.exit(main())
