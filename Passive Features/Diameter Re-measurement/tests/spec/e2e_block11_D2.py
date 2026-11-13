"""End-to-end run of Block 11's entry points (integration tester run D2, step 6,
2026-11-13): SPEC section 6's Block 11 commands -- the registration survey,
the pilot (run_node.py), the plane montage (node_planes.py), the plane
evaluations (plane_differences.py: entropy, blend, iterate = Cell 4j),
iteration_fits.py (Cell 4k), fit_variants.py (Cell 4l), variant_iteration.py
(Cell 4m) and list_stretches.py -- each through its own main(), on the
synthetic cell of tests/smoke/fixtures_cell.py served in place of Allen's
server (SPEC section 7's substitute: allen_image_io.HttpFetcher, list_images
and plane_table patched to serve the rendered planes). `check` verifies the
files, the spec's columns, finiteness, the hand-offs between the cells, and
that two runs are identical (timings and paths aside). Not collected by
pytest.

usage (from `Passive Features/Diameter Re-measurement/`):
    python tests/spec/e2e_block11_D2.py run OUT1 && python tests/spec/e2e_block11_D2.py run OUT2
    python tests/spec/e2e_block11_D2.py check OUT1 OUT2
Add --full-grid to `run` for the full 100 / 50 variant grids (minutes per node).
Pure ASCII.
"""
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("MPLBACKEND", "Agg")


def run(out, full=False):
    import allen_image_io as aio
    import fixtures_cell
    from build_table import load_config
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    cfg = load_config("")
    t0 = time.perf_counter()
    os.makedirs(os.path.join(out, "cell"), exist_ok=True)
    swc, fetcher, planes, swc_path = fixtures_cell.synthetic_cell(os.path.join(out, "cell"), cfg, n_nodes=4)
    fetcher.n_requests, fetcher.n_cache_hits, fetcher.bytes_downloaded = 0, 0, 0
    aio.HttpFetcher = lambda cache_dir=None, verbose=True: fetcher
    aio.list_images = lambda specimen: "stub"
    aio.plane_table = lambda images: planes
    log = open(os.path.join(out, "drive.log"), "w")


    def step(name, fn, argv):
        t = time.perf_counter()
        rc = fn(argv)
        log.write("%s rc=%s %.1f s argv=%s\n" % (name, rc, time.perf_counter() - t, argv))
        log.flush()
        print("[drive] %s rc=%s %.1f s" % (name, rc, time.perf_counter() - t), flush=True)
        return rc


    log.write("render %.1f s\n" % (time.perf_counter() - t0))
    import run_node  # noqa: E402
    import node_planes  # noqa: E402
    import plane_differences  # noqa: E402
    import iteration_fits  # noqa: E402
    import fit_variants  # noqa: E402
    import variant_iteration  # noqa: E402
    import list_stretches  # noqa: E402

    import registration_survey  # noqa: E402

    S = "999"
    step("registration_survey", registration_survey.main,
         ["--specimen", S, "--nodes", "3,4", "--out-dir", os.path.join(out, "reg"), "--swc", swc_path])
    step("run_node (Cell 4b, pilot)", run_node.main,
         ["--specimen", S, "--nodes", "4", "--out-dir", os.path.join(out, "pilot"), "--swc", swc_path, "--figures",
          "--background", "--registration-json", os.path.join(out, "reg", "registration_%s.json" % S)])
    step("node_planes (Cell 4d)", node_planes.main,
         ["--specimen", S, "--nodes", "4", "--pilot-csv", os.path.join(out, "pilot", "pilot_%s.csv" % S), "--out-dir",
          os.path.join(out, "planes"), "--swc", swc_path])
    step("plane_differences iterate (Cell 4j)", plane_differences.main,
         ["--specimen", S, "--nodes", "4", "--out-dir", os.path.join(out, "planeiter"), "--evaluation", "iterate",
          "--swc", swc_path])
    for ev in ("entropy", "blend"):
        step("plane_differences %s (Cells 4f, 4i)" % ev, plane_differences.main,
             ["--specimen", S, "--nodes", "4", "--out-dir", os.path.join(out, "plane_" + ev), "--evaluation", ev,
              "--swc", swc_path])
    iter_json = os.path.join(out, "planeiter", "planeiter_%s.json" % S)
    step("iteration_fits (Cell 4k)", iteration_fits.main,
         ["--specimen", S, "--iter-json", iter_json, "--out-dir", os.path.join(out, "fits"), "--swc", swc_path])
    fv = ["--specimen", S, "--iter-json", iter_json, "--out-dir", os.path.join(out, "fitvar"), "--swc", swc_path]
    if not full:
        fv += ["--families", "lorentz,skewnorm", "--lambdas", "0.5"]
    step("fit_variants (Cell 4l)", fit_variants.main, fv)
    vi = ["--specimen", S, "--iter-json", iter_json, "--out-dir", os.path.join(out, "variter"), "--swc", swc_path]
    if not full:
        vi += ["--families", "lorentz", "--lambdas", "0.5"]
    step("variant_iteration (Cell 4m)", variant_iteration.main, vi)
    step("list_stretches", list_stretches.main,
         ["--specimen", S, "--out-dir", os.path.join(out, "stretches"), "--swc", swc_path, "--include", "4"])
    log.write("total %.1f s\n" % (time.perf_counter() - t0))
    log.close()


def check(a, b):
    S = "999"
    problems, notes = [], []


    def need(cond, msg):
        if not cond:
            problems.append(msg)


    def rows(path):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))


    def fnum(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")


    FILES = ["pilot/pilot_%s.csv" % S, "pilot/pilot_summary_%s.json" % S, "planes/planes_4.png", "planes/planes_%s.json" % S,
             "planeiter/planeiter_%s.json" % S, "planeiter/planeiter_4.png",
             "fits/planefit_4.png", "fits/planefit_%s.json" % S, "fits/planefit_overview_%s.png" % S,
             "fitvar/fitvar_%s.csv" % S, "fitvar/fitvar_%s.json" % S, "fitvar/fitvar_4.png", "fitvar/fitvar_summary_%s.png" % S,
             "variter/variter_%s.csv" % S, "variter/variter_%s.json" % S, "variter/variter_stack_4.png",
             "variter/variter_planes_4.png", "stretches/stretches_%s.csv" % S, "stretches/stretches_%s.json" % S,
             "reg/registration_%s.json" % S, "reg/registration_summary_%s.json" % S,
             "plane_entropy/planeentropy_%s.json" % S, "plane_entropy/planeentropy_4.png",
             "plane_blend/planeblend_%s.json" % S, "plane_blend/planeblend_4.png"]
    for d in (a, b):
        for f in FILES:
            p = os.path.join(d, f)
            need(os.path.isfile(p) and os.path.getsize(p) > 0, "missing or empty: %s" % p)
        with open(os.path.join(d, "drive.log")) as f:
            for line in f:
                if " rc=" in line and " rc=0 " not in line:
                    problems.append("%s: %s" % (d, line.strip()))

    # ---- the pilot (Block 11 pilot_row / summarize; SPEC Block 11 'Pilot')
    pr = rows(os.path.join(a, "pilot/pilot_%s.csv" % S))
    need(len(pr) == 4, "pilot rows %d, expected the 4 nodes of node 4's stretch" % len(pr))
    cols = set(pr[0])
    for c in ("node_id", "k_star", "z_sub_um", "d_hat_um", "mu_hat_per_um", "alpha_hat", "fit_status", "flags",
              "allen_radius_um", "sigma_fit_um", "in_S", "reject", "calibration_node", "calibration_set", "focus_rule",
              "k_star_depth", "k_iter", "iter_status", "d_pilot_um", "pilot_status", "pilot_flags", "in_S_pilot",
              "reject_pilot", "bg_median_gl", "bg_robust_sd_gl", "bg_clipped_sd_gl", "bg_unmasked_frac"):
        need(c in cols, "pilot CSV lacks %s" % c)
    for r in pr:
        if r["in_S"] == "True":
            for c in ("d_hat_um", "mu_hat_per_um", "alpha_hat", "phi_rad", "B_bar"):
                need(math.isfinite(fnum(r[c])), "pilot node %s in S with a non-finite %s" % (r["node_id"], c))
            need(0.05 <= fnum(r["d_hat_um"]) <= 6.0, "d_hat outside the fit box: %s" % r["d_hat_um"])
    with open(os.path.join(a, "pilot/pilot_summary_%s.json" % S)) as f:
        sm = json.load(f)
    for k in ("n_nodes", "n_in_S", "d_hat_um", "mu_hat_per_um", "alpha_hat", "d_hat_over_allen_d", "phi_deg",
              "dark_share_of_converged", "n_calibration_nodes", "n_calibration_by_set", "focus_rule", "k_star_vs_dip_depth",
              "estimator_hash", "background"):
        need(k in sm, "pilot summary lacks %s" % k)
    notes.append("pilot: %d nodes, %d in S, d_hat p50 %.3f um (true 0.8 um), d/2rA p50 %.3f, calibration %s"
                 % (sm["n_nodes"], sm["n_in_S"], sm["d_hat_um"].get("p50", float("nan")),
                    sm["d_hat_over_allen_d"].get("p50", float("nan")), sm["n_calibration_by_set"]))

    # ---- the registration survey -> the pilot (Block 11's own hand-off, --registration-json)
    with open(os.path.join(a, "reg/registration_%s.json" % S)) as f:
        rg = json.load(f)
    need(set(rg) == {"3", "4"}, "registration JSON nodes %s" % sorted(rg))
    for nid, v in rg.items():
        need({"verdict", "lateral_offset_um", "z_offset_um"} <= set(v), "registration JSON node %s keys %s" % (nid, sorted(v)))
        need(str(v["verdict"]).startswith("ON THE PROCESS"), "registration verdict of node %s: %r" % (nid, v["verdict"]))
        need(abs(fnum(v["lateral_offset_um"])) <= 0.2 and abs(fnum(v["z_offset_um"])) <= 0.3,
             "registration offsets of node %s: %r %r (SPEC: |s*| <= 0.2, |dz*| <= 0.3 on the cell's own path)"
             % (nid, v["lateral_offset_um"], v["z_offset_um"]))
    for r in pr:
        if r["node_id"] in rg:
            need(r["reg_verdict"] == rg[r["node_id"]]["verdict"], "pilot row %s reg_verdict %r" % (r["node_id"], r["reg_verdict"]))
    notes.append("registration: " + "; ".join("%s %s s* %.3f dz* %.3f" % (k, v["verdict"][:14], fnum(v["lateral_offset_um"]),
                                                                        fnum(v["z_offset_um"])) for k, v in sorted(rg.items())))

    # ---- the montage
    with open(os.path.join(a, "planes/planes_%s.json" % S)) as f:
        pl = json.load(f)
    need(pl[0]["pilot"] == "same", "node_planes: the re-measurement differs from the pilot row: %r" % pl[0].get("pilot"))

    # ---- Cell 4j, 4k
    with open(os.path.join(a, "planeiter/planeiter_%s.json" % S)) as f:
        it = json.load(f)
    rec = [r for r in it if int(r["node_id"]) == 4][0]
    need(rec.get("k_iter") is not None and math.isfinite(fnum(rec.get("d_iter_um"))), "Cell 4j node 4: no k_iter or d_iter")
    notes.append("Cell 4j node 4: %s in %s rounds, k_iter %s, d_iter %.4f um, k* %s"
                 % (rec.get("iter_status"), rec.get("n_rounds"), rec.get("k_iter"), fnum(rec.get("d_iter_um")),
                    rec.get("k_star")))
    with open(os.path.join(a, "fits/planefit_%s.json" % S)) as f:
        pf = json.load(f)
    p4 = [r for r in pf if int(r["node_id"]) == 4][0]
    need(p4.get("d_refit_um") == rec.get("d_iter_um"), "Cell 4k: d_refit %r != Cell 4j d_iter %r" % (p4.get("d_refit_um"),
                                                                                                  rec.get("d_iter_um")))
    need(not p4.get("notes"), "Cell 4k notes: %r" % p4.get("notes"))

    # ---- Cell 4l (SPEC Block 11 'Fit variants')
    fv = rows(os.path.join(a, "fitvar/fitvar_%s.csv" % S))
    spec_cols = ["node_id", "k_iter", "k_star", "k_swc", "d_iter_um", "variant", "family", "kernel", "shape_mode",
                 "width_mode", "weight", "lam_um", "d_um", "mu_per_um", "v0_um", "fwhm_um", "shape", "alpha_hat", "status",
                 "at_bound", "rss_w_gl2", "rss_gl2", "n_irls", "irls_ok", "nfev", "n_starts", "best_start", "n_nodes",
                 "d_base_um", "ratio"]
    need(set(spec_cols) <= set(fv[0]), "fitvar CSV lacks %s" % sorted(set(spec_cols) - set(fv[0])))
    with open(os.path.join(a, "fitvar/fitvar_%s.json" % S)) as f:
        fj = json.load(f)
    n4 = [r for r in fj if int(r["node_id"]) == 4][0]
    for k in ("d_block5_um", "d_base_um", "phi_rad", "B_bar", "v_um", "profile", "variants", "notes", "png", "seconds"):
        need(k in n4, "fitvar JSON lacks %s" % k)
    need(len(fv) == len(n4["variants"]), "fitvar CSV rows %d vs JSON variants %d" % (len(fv), len(n4["variants"])))
    need(len(n4["v_um"]) == len(n4["profile"]) == 53, "fitvar profile length %d (53 expected at +-3 um, 0.1144 um)"
         % len(n4["v_um"]))
    need(abs(n4["d_base_um"] - fnum(rec["d_iter_um"])) <= 1e-6, "fitvar baseline %r vs Cell 4j d_iter %r"
         % (n4["d_base_um"], rec["d_iter_um"]))
    need(not n4["notes"], "fitvar notes: %r" % n4["notes"])
    for r in fv:
        if r["status"] != "failed":
            need(math.isfinite(fnum(r["d_um"])) and fnum(r["d_um"]) > 0, "fitvar %s d %s" % (r["variant"], r["d_um"]))
            need(abs(fnum(r["ratio"]) - fnum(r["d_um"]) / fnum(r["d_base_um"])) <= 1e-12 * max(1.0, fnum(r["ratio"])),
                 "fitvar ratio %s" % r["variant"])
    notes.append("Cell 4l node 4: %d variants, d %.3f-%.3f um, statuses %s"
                 % (len(fv), min(fnum(r["d_um"]) for r in fv), max(fnum(r["d_um"]) for r in fv),
                    sorted({r["status"] for r in fv})))

    # ---- Cell 4m (SPEC Block 11 'Each fit variant's own Allen-guided iteration')
    vi = rows(os.path.join(a, "variter/variter_%s.csv" % S))
    spec_cols = ["node_id", "k_swc", "k_star", "line_mult", "max_rounds", "allen_2r_um", "d_iter_cell4j_um", "k_iter_cell4j",
                 "variant", "family", "kernel", "shape_mode", "width_mode", "weight", "lam_um", "k_allen", "k_iter",
                 "d_iter_um", "iter_status", "n_rounds", "planes", "v0_um", "mu_per_um", "fwhm_um", "shape", "fit_status",
                 "at_bound", "n_irls", "irls_ok", "B_bar", "sel_round", "sel_half_um", "s_um", "n_restacks", "ratio_ref"]
    need(set(spec_cols) <= set(vi[0]), "variter CSV lacks %s" % sorted(set(spec_cols) - set(vi[0])))
    need(set(vi[0]) <= set(spec_cols), "variter CSV has extra columns %s" % sorted(set(vi[0]) - set(spec_cols)))
    need(vi[0]["variant"].startswith("Block 5"), "variter: the reference is not the first row (%s)" % vi[0]["variant"])
    need(vi[0]["k_iter"] == str(rec["k_iter"]) and fnum(vi[0]["d_iter_um"]) == fnum(rec["d_iter_um"]),
         "variter reference k_iter/d_iter %s/%s vs Cell 4j %s/%s" % (vi[0]["k_iter"], vi[0]["d_iter_um"], rec["k_iter"],
                                                                     rec["d_iter_um"]))
    with open(os.path.join(a, "variter/variter_%s.json" % S)) as f:
        vj = json.load(f)
    v4 = [r for r in vj if int(r["node_id"]) == 4][0]
    need(not v4["notes"], "variter notes: %r" % v4["notes"])
    for r in vi:
        if math.isfinite(fnum(r["d_iter_um"])):
            need(abs(fnum(r["ratio_ref"]) - fnum(r["d_iter_um"]) / fnum(vi[0]["d_iter_um"])) <= 1e-12,
                 "ratio_ref %s" % r["variant"])
    notes.append("Cell 4m node 4: %d rows; k_iter %s; statuses %s; d %s"
                 % (len(vi), sorted({r["k_iter"] for r in vi}), sorted({r["iter_status"] for r in vi}),
                    ", ".join("%.3f" % fnum(r["d_iter_um"]) for r in vi)))

    # ---- determinism: run b equals run a (timing and paths aside)
    DROP = {"seconds", "png", "png_stack", "png_planes", "path", "out_dir", "csv"}


    def scrub(x):
        if isinstance(x, dict):
            return {k: scrub(v) for k, v in x.items() if k not in DROP}
        if isinstance(x, list):
            return [scrub(v) for v in x]
        if isinstance(x, str) and (x.startswith(a) or x.startswith(b)):
            return "<path>"
        return x


    for f in FILES:
        pa, pb = os.path.join(a, f), os.path.join(b, f)
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            continue
        if f.endswith(".csv"):
            ra, rb = rows(pa), rows(pb)
            need(scrub(ra) == scrub(rb), "determinism: %s differs between the two runs" % f)
        elif f.endswith(".json"):
            with open(pa) as fa, open(pb) as fb:
                ja, jb = json.load(fa), json.load(fb)
            if isinstance(ja, dict):
                ja.pop("seconds", None), jb.pop("seconds", None)
            need(scrub(ja) == scrub(jb), "determinism: %s differs between the two runs" % f)

    print("NOTES")
    for n in notes:
        print("  " + n)
    print("PROBLEMS (%d)" % len(problems))
    for p in problems:
        print("  " + p)
    return problems


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(sys.argv[2], "--full-grid" in sys.argv)
    elif len(sys.argv) == 4 and sys.argv[1] == "check":
        sys.exit(1 if check(sys.argv[2], sys.argv[3]) else 0)
    else:
        sys.exit(__doc__)
