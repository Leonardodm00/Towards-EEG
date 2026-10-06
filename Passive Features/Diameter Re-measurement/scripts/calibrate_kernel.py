#!/usr/bin/env python3
"""Calibrate the defocus kernel from plane scans -- Block 10 in specs/SPEC.md
(procedure Eq. 4, s.3.4; mathematics Eq. 18, s.3.5).

  synthetic  Phase I: render N thin, faint, flat phantoms with the
             configuration's kernel, measure each (Block 5) and scan its
             planes k* - P .. k* + P along the fixed line -> scans JSON
  real       Phase II (Colab: api.brain-map.org): the candidates of a
             previous run_cell.py table (converged, in S, thin, flat, faint)
             are re-measured and scanned on Allen's planes -> scans JSON
  fit        Eq. (4) over the scans; the first-stage Gaussian table
             sigma_r(delta_m) = sqrt(sigma_r(0)^2 + G(delta_m)), sigma_r(0)
             configured (not identified) -> calibration JSON, and a report:
             the knots, the residual RMS by plane offset (trust beyond
             +-trust_planes only after reading it) and the area diagnostic

The first-stage table is a starting point, not a calibrated kernel: on
rendered thin phantoms the core-width growth is low by about 0.01 um^2 within
two planes of focus (Block 10 status), and procedure s.3.4 step 2 -- tuning
sigma_r so that rendered phantoms reproduce the real profiles plane by plane
-- is not implemented.

Examples
    python scripts/calibrate_kernel.py synthetic --n 12 --seed 5 --out cal/scans_syn.json --workers 2
    python scripts/calibrate_kernel.py fit --scans cal/scans_syn.json --out cal/calibration_syn.json
    # Colab, after run_cell.py wrote nodes_529878215.csv:
    python scripts/calibrate_kernel.py real --specimen 529878215 --nodes-csv out/nodes_529878215.csv \
        --shift-x 12.3 --shift-y -4.1 --cache-dir /content/drive/MyDrive/allen_cache --out cal/scans_529878215.json

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from allen_diameter.analysis import calibration, phantoms  # noqa: E402
from allen_diameter.loading import calibration_io, table_io  # noqa: E402
from allen_diameter.model import render  # noqa: E402
from build_table import load_config  # noqa: E402


def _pair(text):
    a, b = (float(x) for x in text.split(","))
    return a, b


def calibration_phantom(cfg, seed, index, d_range, alpha_range, phi_max_deg):
    """(Draw, generator) of a thin, faint, flat phantom: log d uniform on d_range, alpha = mu d / cos(phi)
    uniform on alpha_range, phi uniform on [0, phi_max), theta on [0, pi), the sub-pixel offset and
    the axis depth as in the table's design; replicate n draws from default_rng([seed, n])."""
    rng = np.random.default_rng([int(seed), int(index)])
    d = math.exp(rng.uniform(math.log(d_range[0]), math.log(d_range[1])))
    phi = math.radians(rng.uniform(0.0, phi_max_deg))
    theta = rng.uniform(0.0, math.pi)
    alpha = rng.uniform(*alpha_range)
    mu = alpha * math.cos(phi) / d
    p, dz = cfg.acquisition.res0_um, cfg.acquisition.dz_um
    cx, cy = rng.uniform(-0.5 * p, 0.5 * p, 2)
    cz = rng.uniform(-0.5 * dz, 0.5 * dz)
    return phantoms.Draw(int(index), int(seed), d, phi, theta, mu, float(cx), float(cy), float(cz)), rng


def _one_synthetic(args):
    cfg, seed, index, d_range, alpha_range, phi_max = args
    draw, rng = calibration_phantom(cfg, seed, index, d_range, alpha_range, phi_max)
    tube = phantoms.phantom_tube(draw, cfg)
    branch = phantoms.phantom_branch(draw, cfg, rng)
    pad = int(math.ceil(cfg.renderer.pad_um / cfg.acquisition.res0_um))

    def provider(left, top, width, height, k_lo, k_hi):
        return render.synthetic_block(tube, draw.mu_per_um, np.arange(k_lo, k_hi + 1), left, top, width, height,
                                      cfg, rng, pad)

    res, scan, reasons = calibration.scan_node(branch, cfg.phantom.nodes_each_way, provider, cfg)
    if scan is not None:                 # every phantom's measured node is node nodes_each_way: name it by index
        scan = dataclasses.replace(scan, node_id=int(index))
    return draw, res, scan, reasons


def run_synthetic(cfg, n, seed, out, d_range, alpha_range, phi_max, workers=1):
    jobs = [(cfg, seed, i, d_range, alpha_range, phi_max) for i in range(int(n))]
    if workers > 1:
        import multiprocessing
        with multiprocessing.Pool(workers) as pool:
            results = list(pool.imap(_one_synthetic, jobs, chunksize=1))
    else:
        results = [_one_synthetic(j) for j in jobs]
    scans = [s for _, _, s, _ in results if s is not None]
    meta = dict(source="synthetic", seed=int(seed), n=int(n), d_range_um=list(d_range), alpha_range=list(alpha_range),
                phi_max_deg=phi_max, config=json.loads(cfg.to_json()),
                truth=[dict(index=dr.index, d_um=dr.d_um, mu_per_um=dr.mu_per_um, phi_rad=dr.phi_rad, cz_um=dr.cz_um,
                            scanned=s is not None, reasons=";".join(rs)) for dr, _, s, rs in results])
    calibration_io.save_scans(scans, out, meta)
    for dr, res, s, rs in results:
        print("phantom %d: d %.3f um, alpha %.2f -> d_hat %.3f, %s" % (
            dr.index, dr.d_um, dr.mu_per_um * dr.d_um / math.cos(dr.phi_rad), res.d_hat_um,
            "scanned" if s is not None else "not a calibration node (%s)" % ";".join(rs)))
    print("%d of %d phantoms scanned -> %s" % (len(scans), len(results), out))
    return out


def candidates_from_rows(rows, cfg):
    """Node ids of a run_cell.py table that can be calibration nodes (the rule of calibration_reasons on the
    CSV columns); they are re-measured before the scan, so this only saves fetches."""
    cal = cfg.calibration
    out = set()
    for r in rows:
        flags = [f for f in str(r.get("flags", "")).split(";") if f]
        try:
            ok = (r["fit_status"] == "converged" and not any(f in phantoms.SELECTION_FLAGS for f in flags)
                  and float(r["d_hat_um"]) <= cal.node_dhat_max_um
                  and float(r["phi_rad"]) <= math.radians(cal.node_phi_max_deg)
                  and float(r["alpha_hat"]) <= cal.node_alpha_max)
        except (KeyError, ValueError, TypeError):
            ok = False
        if ok:
            out.add(int(r["node_id"]))
    return out


def run_real(cfg, a):
    import allen_image_io as aio
    import run_cell
    from allen_diameter.loading import swc_io
    swc = swc_io.read_swc(a.swc or aio.fetch_swc(int(a.specimen), a.cache_dir))
    planes = aio.plane_table(aio.list_images(int(a.specimen)))
    provider = run_cell.real_provider(aio.HttpFetcher(cache_dir=a.cache_dir), planes, cfg.acquisition.res0_um)
    cand = candidates_from_rows(table_io.read_rows([a.nodes_csv]), cfg)
    regs = {}
    if a.registration_json:
        with open(a.registration_json) as f:
            regs = {int(k): v for k, v in json.load(f).items()}
    print("%d candidate nodes" % len(cand), flush=True)
    t0 = time.perf_counter()
    out = calibration.scan_cell(swc, provider, cfg, dict(shift_full_px=(a.shift_x, a.shift_y), flip_y_full_h=a.flip_h,
                                                        z0_um=a.z0), regs, cand, log=lambda m: print(m, flush=True))
    scans = [s for _, _, s, _ in out if s is not None]
    calibration_io.save_scans(scans, a.out, dict(source="real", specimen=str(a.specimen), n_candidates=len(cand),
                                                 config=json.loads(cfg.to_json())))
    print("%d of %d candidates scanned in %.1f min -> %s" % (len(scans), len(out), (time.perf_counter() - t0) / 60, a.out))
    return a.out


def run_fit(cfg, scans_path, out, sigma_r0=None):
    scans, meta = calibration_io.load_scans(scans_path)
    try:
        g = calibration.fit_growth(scans, cfg)
    except ValueError as exc:
        sys.exit("fit: %s (%d scans in %s)" % (exc, len(scans), scans_path))
    kernel, err = None, ""
    try:
        kernel = calibration.kernel_from_growth(g, cfg.renderer, sigma_r0)
    except ValueError as exc:
        err = str(exc)
    ratio = calibration.area_ratio_by_offset(scans)
    calibration_io.save_calibration(out, g, kernel, err, ratio,
                                    dict(scans=scans_path, scans_meta_source=meta.get("source", ""),
                                         config=json.loads(cfg.to_json())))
    trust = (cfg.calibration.trust_planes + 0.5) * cfg.acquisition.dz_um
    print("growth fit (%s, %s, %s): %d nodes, %d dropped, cost %.3g, %s"
          % (g.statistic, g.convention, g.interp, g.node_ids.size, len(g.dropped_nodes), g.cost, g.message))
    print("  delta_um   G_um2      n_obs(interval to the next knot)  trusted")
    for m, (x, val) in enumerate(zip(g.knots_um, g.g_um2)):
        n = "%5d" % int(g.n_obs_per_interval[m]) if m < g.n_obs_per_interval.size else "    -"
        print("  %+7.3f  %9.5f  %s  %s" % (x, val, n, "yes" if abs(x) <= trust + 1e-9 else "after the residual check"))
    print("  residual RMS by |k - k*|: %s" % ", ".join("%d: %.4f" % kv for kv in g.rms_by_offset().items()))
    print("  A_W(k) / A_W(k*): %s" % ", ".join("%+d: %.3f" % kv for kv in ratio.items()))
    if kernel is not None:
        mono = bool(np.all(np.diff(kernel.kernel_table_sigma_um) > 0))
        print("  first-stage table: delta %s" % ", ".join("%.2f" % x for x in kernel.kernel_table_delta_um))
        print("                     sigma %s  (sigma_r(0) = %.3f configured; %s)"
              % (", ".join("%.3f" % x for x in kernel.kernel_table_sigma_um), kernel.sigma_r0_um,
                 "increasing" if mono else "NOT increasing: do not render with it before step 2"))
    else:
        print("  no first-stage table: %s" % err)
    print("-> %s" % out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--d-range", type=_pair, default=(0.15, 0.25),
                   help="true d, um, comma-separated (thin flat tubes read 5-25 %% wide, and the node must "
                        "read d_hat <= node_dhat_max_um)")
    s.add_argument("--alpha-range", type=_pair, default=(0.15, 0.4))
    s.add_argument("--phi-max", type=float, default=5.0, help="deg")
    s.add_argument("--workers", type=int, default=1)
    r = sub.add_parser("real")
    r.add_argument("--specimen", required=True)
    r.add_argument("--nodes-csv", required=True, help="nodes_<specimen>.csv of run_cell.py")
    r.add_argument("--swc", default="")
    r.add_argument("--cache-dir", default="allen_cache")
    r.add_argument("--shift-x", type=float, default=0.0)
    r.add_argument("--shift-y", type=float, default=0.0)
    r.add_argument("--flip-h", type=float, default=None)
    r.add_argument("--z0", type=float, default=0.0)
    r.add_argument("--registration-json", default="")
    f = sub.add_parser("fit")
    f.add_argument("--scans", required=True)
    f.add_argument("--sigma-r0", type=float, default=None, help="um; default: the configuration's sigma_r0_um")
    for q in (s, r, f):
        q.add_argument("--out", required=True)
        q.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config_json)
    if a.cmd == "synthetic":
        run_synthetic(cfg, a.n, a.seed, a.out, a.d_range, a.alpha_range, a.phi_max, a.workers)
    elif a.cmd == "real":
        run_real(cfg, a)
    else:
        run_fit(cfg, a.scans, a.out, a.sigma_r0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
