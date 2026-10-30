"""D-115.6 check: does Block 10.C's far search depend on the cell's transform beyond the pixel grid?

The far search (candidate_search.search + choose, the pipeline's code) on the pilot 529878215's trace with the pilot
table's medians as B_bar and mu_hat, under one transform given on the command line. Run twice (Cell 7's design
alignment, and a shifted one like S3's automatic alignment) and compare the outputs with --compare.

Usage (from the workstream directory):
    python3 -I search_transform_check.py run SWC PILOT_CSV OUT_JSON SHIFT_X SHIFT_Y Z0 [near|far]
    python3 -I search_transform_check.py compare A_JSON B_JSON
"""
import json
import os
import sys
import time

sys.path[:0] = [os.path.abspath("src"), os.path.abspath("scripts")]


def run(swc_path, pilot_csv, out_json, sx, sy, z0, which="far"):
    from allen_diameter.analysis import candidate_search
    from allen_diameter.config import default_config
    from allen_diameter.loading import swc_io, table_io
    cfg = default_config()
    swc = swc_io.read_swc(swc_path)
    B_bar, mu = candidate_search.typical_background(table_io.read_rows([pilot_csv]), cfg)
    tr = dict(shift_full_px=(float(sx), float(sy)), flip_y_full_h=None, z0_um=float(z0))
    t0 = time.perf_counter()
    recs = candidate_search.search(swc, cfg, which, tr, B_bar, mu, log=lambda m: print(m, flush=True))
    chosen = candidate_search.choose(recs, cfg, which)
    out = dict(transform=tr, B_bar=B_bar, mu=mu, which=which, minutes=(time.perf_counter() - t0) / 60,
               chosen=[int(c) for c in chosen],
               records={str(int(r["node_id"])): dict(reason=str(r["reason"]), k_trace=int(r["k_trace"]),
                                                     n_pixels_k=r.get("n_pixels_k"), n_fit_usable=r.get("n_fit_usable"))
                        for r in recs})
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("%s: %d records, %d chosen, %.1f min" % (out_json, len(recs), len(chosen), out["minutes"]), flush=True)


def compare(a_json, b_json):
    a, b = json.load(open(a_json)), json.load(open(b_json))
    print("A transform", a["transform"], "B transform", b["transform"])
    ra, rb = a["records"], b["records"]
    print("records: A %d, B %d, same node ids: %s" % (len(ra), len(rb), set(ra) == set(rb)))
    diff_reason = [n for n in ra if n in rb and ra[n]["reason"] != rb[n]["reason"]]
    diff_pix = [n for n in ra if n in rb and ra[n]["n_pixels_k"] != rb[n]["n_pixels_k"]]
    print("reason differs: %d %s" % (len(diff_reason), [(n, ra[n]["reason"], rb[n]["reason"]) for n in diff_reason[:20]]))
    print("n_pixels_k differs: %d of %d" % (len(diff_pix), len(ra)))
    ca, cb = set(a["chosen"]), set(b["chosen"])
    print("chosen: A %d, B %d, common %d, only A %s, only B %s" % (len(ca), len(cb), len(ca & cb), sorted(ca - cb),
                                                                    sorted(cb - ca)))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:])
    else:
        compare(*sys.argv[2:4])
