"""Cell 4k (D-045, the user's request of 2026-10-12, 17:58): for each node of a
Cell 4j run (scripts/plane_differences.py --evaluation iterate), the plane the
Allen-guided iteration chose, k_iter, with the line that chose it and the
diameter fitted there. A figure: no measurement changes.

For node j: the selection round i* is the last round of the trajectory that
picked a plane (k_(j,i*) = k_iter; at convergence the repeat round, whose line
is sized from d_iter itself), h_sel its half-length. The line that chose the
plane is L_j = {o_j + v y_hat : |v| <= h_sel}, through the SWC node o_j across
the pilot's heading theta_j, y_hat = (-sin theta_j, cos theta_j) (the user's
answer: "the line chosen to select the focal plane"). The diameter is Block
5's fit at k_iter (plane_differences.refit_at_plane): d_iter and v0_hat, the
latter measured along y_hat from the pilot's fitted centre c_j; with
s_j = (c_j - o_j) . y_hat its edges lie on L_j at v = s_j + v0_hat -+ d_iter / 2
(the caliper). The fit's own line, through c_j, is parallel to L_j and
u_j = (c_j - o_j) . e_u apart along the branch, e_u = (cos theta_j, sin
theta_j); both offsets are reported.

Per node the square is Cell 4j's last one, rebuilt (node_square) and the fit is
redone at k_iter (fit_overlay); a square or a diameter that differs from the
record is reported in the node's notes. Writes planefit_<id>.png (the plane
with L_j and the caliper, beside the profiles and the fitted model),
planefit_overview_<specimen>.png and planefit_<specimen>.json.

    python scripts/iteration_fits.py --specimen 529878215 \
        --iter-json OUT/pilot/planeiter/bydiameter_m3/planeiter_529878215.json \
        --out-dir OUT/pilot/planeiter/bydiameter_m3/fits --cache-dir CACHE

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plane_differences as PD  # noqa: E402
from allen_diameter.model import tube_model  # noqa: E402
from build_table import load_config  # noqa: E402

PREFIX = "planefit"


# ---------------------------------------------------------------- loading ---

def read_records(path):
    """The records of Cell 4j's planeiter_<specimen>.json, in their order (the
    pilot's d_hat). ValueError for a file that is not a list of records of the
    iterate evaluation."""
    with open(path) as f:
        recs = json.load(f)
    if not isinstance(recs, list) or not all(isinstance(r, dict) and "node_id" in r for r in recs):
        raise ValueError("%s: not a list of node records" % path)
    other = [int(r["node_id"]) for r in recs if "skipped" not in r and r.get("evaluation") != "iterate"]
    if other:
        raise ValueError("%s: records not of --evaluation iterate (nodes %s)" % (path, other))
    return recs


# --------------------------------------------------------------- analysis ---

def selection_round(rec):
    """The round whose pick is the record's k_iter: the last round of the
    trajectory that picked a plane (iterate_planes' k_final). None when the
    record has no k_iter; ValueError when that round does not hold it."""
    k = rec.get("k_iter")
    if k is None:
        return None
    picked = [x for x in rec.get("rounds") or [] if x.get("k") is not None]
    if not picked or int(picked[-1]["k"]) != int(k):
        raise ValueError("node %d: the last round that picked a plane does not hold k_iter %s"
                         % (int(rec["node_id"]), k))
    return picked[-1]


def node_square(swc, provider, cfg, rec, transform=None, half_um=None):
    """Cell 4j's last square of the node, rebuilt: plane_differences.node_stack
    with the record's planes (k_SWC - ks[0] on each side), the longest line it
    used (profile_half_um) and its multiplier for the Allen line and the
    pilot's d-line. The squares of Cell 4j only grow and each holds every line
    before it, so this gives the last one's half-width and the same crops (from
    the cache). half_um must be Cell 4j's --half-um (it passes none). Returns
    (st, notes): a square of another half-width is noted, not refused.
    ValueError for a record whose planes are not k_SWC +- n, or when the
    rebuilt square has other planes."""
    ks = [int(k) for k in rec["ks"]]
    k_swc = int(rec["k_swc"])
    n = k_swc - ks[0]
    if n < 0 or ks != list(range(k_swc - n, k_swc + n + 1)):
        raise ValueError("node %d: the record's planes %s are not k_SWC %d +- n" % (int(rec["node_id"]), ks, k_swc))
    m = float(rec["line_mult"])
    h = rec.get("profile_half_um")
    st = PD.node_stack(swc, provider, cfg, int(rec["node_id"]), transform, n, half_um, 0.0,
                       None if h is None else float(h), m, m)
    if [int(k) for k in st["ks"]] != ks:
        raise ValueError("node %d: the square's planes %s, the record's %s" % (int(rec["node_id"]),
                                                                              [int(k) for k in st["ks"]], ks))
    notes = []
    if abs(float(st["square_half_um"]) - float(rec["half_um"])) > 1e-9:
        notes.append("square +-%.3f um, Cell 4j's +-%.3f um" % (float(st["square_half_um"]), float(rec["half_um"])))
    return st, notes


def fit_overlay(st, cfg, rec):
    """What one node's figure draws (module docstring), from node_square's st
    and Cell 4j's record: the plane k_iter of the square (plane, extent), the
    SWC node o, y_hat and e_u, the line that chose the plane (h_sel, sel_round,
    line_xy (2, 2) um) with the profile of k_iter along it (v_sel, prof_sel),
    Block 5's fit redone at k_iter (fit: refit_at_plane with its profile; d,
    v0), s_um and u_um, the caliper (edges_v (2,) on L_j, edges_xy (2, 2) um;
    None without a finite fit), the fitted model at the fit's offsets (model;
    None without a fit) and notes (the fit differing from the record's
    d_iter_um). ValueError without k_iter, or when k_iter is not a valid plane
    of the square."""
    nid = int(rec["node_id"])
    rnd = selection_round(rec)
    if rnd is None:
        raise ValueError("node %d: no plane chosen (iteration %s)" % (nid, rec.get("iter_status")))
    k = int(rec["k_iter"])
    ks = np.asarray(st["ks"])
    j = k - int(ks[0])
    if not (0 <= j < ks.size and int(ks[j]) == k and bool(st["valid"][j])):
        raise ValueError("node %d: plane %d is not a valid plane of the square" % (nid, k))
    h = float(rnd["half_um"])
    _h, v_sel, y_hat, e_u, prof = PD.line_profiles(st, cfg, h)
    o = np.asarray(st["o"], dtype=float)
    r = st["result"]
    c = np.array([float(r.cx_um), float(r.cy_um)])
    s_um, u_um = float(np.dot(c - o, y_hat)), float(np.dot(c - o, e_u))
    fit = PD.refit_at_plane(st, cfg, k, with_profile=True)
    d, v0 = float(fit["d_hat_um"]), float(fit["v0_hat_um"])
    notes = []
    d_rec = rec.get("d_iter_um")
    d_rec = float("nan") if d_rec is None else float(d_rec)
    if not (d == d_rec or (math.isnan(d) and math.isnan(d_rec))):
        notes.append("the fit redone here gives d %s um, Cell 4j's %s um"
                     % ("%.4f" % d if math.isfinite(d) else "n/a", "%.4f" % d_rec if math.isfinite(d_rec) else "n/a"))
    edges_v = edges_xy = model = None
    if math.isfinite(d) and math.isfinite(v0):
        edges_v = np.array([s_um + v0 - 0.5 * d, s_um + v0 + 0.5 * d])
        edges_xy = o[None, :] + edges_v[:, None] * y_hat[None, :]
        model = tube_model.model_profile(fit["v_um"], d, fit["alpha_hat"], v0, cfg.measure.sigma_fit_um,
                                         fit["B_bar"], fit["n_nodes"])
    return dict(k=k, j=j, plane=np.asarray(st["stack"][j], dtype=float), extent=st["extent"], o=o, y_hat=y_hat,
                e_u=e_u, h_sel=h, sel_round=int(rnd["round"]), line_xy=np.array([o - h * y_hat, o + h * y_hat]),
                v_sel=v_sel, prof_sel=prof[j], s_um=s_um, u_um=u_um, fit=fit, d=d, v0=v0, edges_v=edges_v,
                edges_xy=edges_xy, model=model, notes=notes)


# ----------------------------------------------------------------- wiring ---

def _f(x, fmt="%.2f"):
    return fmt % x if x is not None and math.isfinite(float(x)) else "n/a"


def _signed(x):
    """+-x.xx um, without a negative zero."""
    return "%+.2f" % (0.0 if abs(x) < 0.005 else x)


def _fit_label(ov):
    st = ov["fit"]["fit_status"]
    return "d %s um%s" % (_f(ov["d"]), "" if st == "converged" else " (%s)" % st)


def run(swc, provider, cfg, records, out_dir, specimen, transform=None, half_um=None, nodes=None, ncol=5, dpi=110,
        log=print):
    """One figure per node with a plane chosen, the overview and the JSON (the
    module docstring); nodes: ids to draw (default every node of the
    records). A record skipped by Cell 4j, without k_iter, or refused by
    node_square is listed with its reason. Returns the per-node records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from allen_diameter.plotting import figures as fg
    os.makedirs(out_dir, exist_ok=True)
    want = None if not nodes else {int(x) for x in nodes}
    out, panels = [], []
    for rec in records:
        nid = int(rec["node_id"])
        if want is not None and nid not in want:
            continue
        if "skipped" in rec or rec.get("k_iter") is None:
            why = ("Cell 4j: " + str(rec["skipped"])) if "skipped" in rec else (
                "no plane chosen (iteration %s)" % rec.get("iter_status"))
            log("[planefit] node %d skipped: %s" % (nid, why))
            out.append(dict(node_id=nid, skipped=why))
            continue
        try:
            st, notes = node_square(swc, provider, cfg, rec, transform, half_um)
            ov = fit_overlay(st, cfg, rec)
        except ValueError as e:
            log("[planefit] node %d skipped: %s" % (nid, e))
            out.append(dict(node_id=nid, skipped=str(e)))
            continue
        notes += ov["notes"]
        r = st["result"]
        head = ("node %d: plane k_iter %d (k* %d, SWC plane %d); %s, the pilot's d_hat %s um at k*, Allen 2r %s um"
                % (nid, ov["k"], int(r.k_star), int(st["k_swc"]), _fit_label(ov), _f(rec.get("d_hat_um")),
                   _f(2.0 * float(rec["allen_radius_um"]))))
        sub = ("the line that chose k_iter: round %d of %d (%s), +-%.2f um through the SWC node, m = %g; the fit's "
               "own line, through the fitted centre, %s um along the branch from it%s"
               % (ov["sel_round"], int(rec["n_rounds"]), rec.get("iter_status"), ov["h_sel"],
                  float(rec["line_mult"]), _signed(ov["u_um"]), "; " + "; ".join(notes) if notes else ""))
        block = dict(label=head, sublabel=sub, plane=ov["plane"], extent=ov["extent"], o=ov["o"], e_u=ov["e_u"],
                     line_xy=ov["line_xy"], edges_xy=ov["edges_xy"], edges_v=ov["edges_v"], d=ov["d"],
                     v_sel=ov["v_sel"], prof_sel=ov["prof_sel"], s_um=ov["s_um"],
                     v_fit=None if ov["fit"]["v_um"] is None else np.asarray(ov["fit"]["v_um"]) + ov["s_um"],
                     prof_fit=ov["fit"]["profile"], model=ov["model"], B_bar=ov["fit"]["B_bar"],
                     fit_half_um=float(cfg.measure.profile_half_um))
        fig = fg.iteration_fit_figure(block, "The plane the iteration chose, the line that chose it and the diameter "
                                             "fitted there, specimen %s" % specimen)
        png = os.path.join(out_dir, "%s_%d.png" % (PREFIX, nid))
        fig.savefig(png, dpi=dpi)
        plt.close(fig)
        panels.append(dict(label="node %d: k_iter %d (k* %d)\n%s" % (nid, ov["k"], int(r.k_star),
                                                                        _fit_label(ov)),
                           plane=ov["plane"], extent=ov["extent"], o=ov["o"], e_u=ov["e_u"], line_xy=ov["line_xy"],
                           edges_xy=ov["edges_xy"]))
        ev = None if ov["edges_v"] is None else [float(x) for x in ov["edges_v"]]
        out.append(dict(node_id=nid, k_iter=ov["k"], k_star=int(r.k_star), k_swc=int(st["k_swc"]),
                        d_iter_um=rec.get("d_iter_um"), d_refit_um=ov["d"], fit_status=ov["fit"]["fit_status"],
                        v0_hat_um=ov["v0"], s_um=ov["s_um"], u_um=ov["u_um"], edges_v_um=ev,
                        edges_xy_um=None if ov["edges_xy"] is None else ov["edges_xy"].tolist(),
                        line_xy_um=ov["line_xy"].tolist(), sel_round=ov["sel_round"], sel_half_um=ov["h_sel"],
                        line_mult=float(rec["line_mult"]), iter_status=rec.get("iter_status"),
                        d_hat_um=rec.get("d_hat_um"), allen_2r_um=2.0 * float(rec["allen_radius_um"]),
                        square_half_um=float(st["square_half_um"]), notes=notes, png=png))
        log("[planefit] node %d: k_iter %d (k* %d); line +-%.2f um (round %d); %s, edges at v %s and %s um from the "
            "SWC node%s" % (nid, ov["k"], int(r.k_star), ov["h_sel"], ov["sel_round"], _fit_label(ov),
                            "n/a" if ev is None else _signed(ev[0]), "n/a" if ev is None else _signed(ev[1]),
                            "; " + "; ".join(notes) if notes else ""))
    if panels:
        fig = fg.iteration_fits_overview(panels, "The plane each node's iteration chose, with the line that chose it "
                                                 "and the diameter fitted there, specimen %s" % specimen, ncol=ncol)
        fig.savefig(os.path.join(out_dir, "%s_overview_%s.png" % (PREFIX, specimen)), dpi=dpi)
        plt.close(fig)
    with open(os.path.join(out_dir, "%s_%s.json" % (PREFIX, specimen)), "w") as f:
        json.dump(out, f, indent=1)
    done = [x for x in out if "skipped" not in x]
    same = sum(1 for x in done if not any(n.startswith("the fit redone") for n in x["notes"]))
    log("[planefit] %d nodes drawn, %d skipped; the fit redone at k_iter equals Cell 4j's d_iter on %d of %d"
        % (len(done), len(out) - len(done), same, len(done)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--iter-json", required=True, help="Cell 4j's planeiter_<specimen>.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nodes", default="", help="comma-separated node ids (default: every node of the record)")
    ap.add_argument("--ncol", type=int, default=5, help="columns of the overview")
    ap.add_argument("--half-um", type=float, default=None,
                    help="Cell 4j's --half-um, if it was given one (Cell 4j passes none)")
    ap.add_argument("--swc", default="")
    ap.add_argument("--cache-dir", default="allen_cache")
    ap.add_argument("--shift-x", type=float, default=0.0, help="full-res px (global alignment), as in Cell 4j")
    ap.add_argument("--shift-y", type=float, default=0.0)
    ap.add_argument("--flip-h", type=float, default=None)
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    if a.ncol < 1:
        ap.error("--ncol must be >= 1")
    try:
        nodes = [int(x) for x in a.nodes.split(",") if x.strip()]
    except ValueError as e:
        ap.error("--nodes takes comma-separated node ids: %s" % e)
    cfg = load_config(a.config_json)
    records = read_records(a.iter_json)
    import allen_image_io as aio
    import run_cell
    from allen_diameter.loading import swc_io
    swc = swc_io.read_swc(a.swc or aio.fetch_swc(int(a.specimen), a.cache_dir))
    fetcher = aio.HttpFetcher(cache_dir=a.cache_dir)
    planes = aio.plane_table(aio.list_images(int(a.specimen)))
    provider = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
    t0 = time.perf_counter()
    run(swc, provider, cfg, records, a.out_dir, a.specimen,
        dict(shift_full_px=(a.shift_x, a.shift_y), flip_y_full_h=a.flip_h, z0_um=a.z0), a.half_um, nodes, a.ncol,
        log=lambda m: print(m, flush=True))
    print("[planefit] files in %s; crops: %d from the cache, %d downloaded (%.1f MB); %.1f min"
          % (a.out_dir, fetcher.n_cache_hits, fetcher.n_requests, fetcher.bytes_downloaded / 1e6,
             (time.perf_counter() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
