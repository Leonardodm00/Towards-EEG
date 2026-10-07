#!/usr/bin/env python3
"""Camera-chain inputs from real Allen crops -- Block 11 in specs/SPEC.md
(procedure s.3.6 step 6; impl-handoff "Known gaps").

  JPEG tables   read from the crops an HttpFetcher cached (bytes as served):
                the distinct table sets and their counts; the most common set
                is suggested as RendererConfig.jpeg_qtables (the tables
                themselves, so the configuration JSON is self-contained on
                davinci) and written to jpeg_qtables_<specimen>.json for
                reference [corrected 2026-10-07: it was suggested as
                jpeg_qtables_file, a path the renderer never read]
  background    from a run_node.py --background summary: the median over
                nodes of the masked block median (-> background_B_gl), and
                the injected noise SD whose post-chain SD on a flat field
                matches the median clipped SD of the real background pixels
                (analysis.camera_fit; JPEG with the tables found, else the
                configured quality): the background SD is read AFTER JPEG,
                which removes much of white noise, so it is not the injected
                SD itself (-> noise_sd_gl; an UPPER bound, since tissue
                texture adds to the real background SD)

Black level and gain are not identified by brightfield crops (no dark frame);
they stay as configured. Writes camera_<specimen>.json: the suggested renderer
fields, the evidence, and the notes. Nothing is applied automatically: copy
the fields into the configuration JSON of the table build once reviewed.

Example (Colab)
    python scripts/camera_calibration.py --specimen 529878215 --cache-dir /content/drive/MyDrive/allen_cache \
        --pilot-summary /content/drive/MyDrive/diameters/pilot/pilot_summary_529878215.json \
        --out-dir /content/drive/MyDrive/diameters/camera

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from allen_diameter.analysis import camera_fit  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import jpeg_tables  # noqa: E402


def run(cache_dir, out_dir, specimen, pilot_summary=None, max_files=500, cfg=None, seed=20261006):
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(cache_dir, "*.img")))[:int(max_files)]
    groups = jpeg_tables.collect_qtables(paths)
    out = dict(specimen=str(specimen), n_files=len(paths), renderer={}, notes=[],
               table_sets=[dict(count=n, first=os.path.basename(p), n_tables=None if t is None else len(t))
                           for t, n, p in groups])
    cfg = default_config() if cfg is None else cfg
    real = [(t, n) for t, n, _ in groups if t is not None]
    if real:
        import PIL
        qpath = os.path.join(out_dir, "jpeg_qtables_%s.json" % specimen)
        jpeg_tables.save_qtables(real[0][0], qpath)
        out["renderer"]["jpeg_qtables"] = [list(t) for t in real[0][0]]
        out.update(jpeg_qtables_file=qpath, pillow_version=str(PIL.__version__))
        if len(real) > 1:
            out["notes"].append("%d distinct table sets: the most common (%d of %d files) was written"
                                % (len(real), real[0][1], sum(n for _, n in real)))
    else:
        out["notes"].append("no JPEG tables found in %s" % cache_dir)
    if pilot_summary:
        with open(pilot_summary) as f:
            bg = json.load(f).get("background", {})
        b, sd = bg.get("B_bar_gl", {}), bg.get("clipped_sd_gl", {})
        if "p50" in b:
            out["renderer"]["background_B_gl"] = b["p50"]
            out["background_B_gl_percentiles"] = b
        if "p50" in sd:
            rc = cfg.renderer
            if "background_B_gl" in out["renderer"]:
                import dataclasses
                rc = dataclasses.replace(rc, background_B_gl=float(out["renderer"]["background_B_gl"]))
            q = real[0][0] if real else None
            est, grid, curve = camera_fit.noise_for_post_chain_sd(sd["p50"], rc, seed, q)
            out["clipped_sd_gl_percentiles"] = sd
            out["noise_match"] = dict(target_post_chain_sd=sd["p50"], injected_sd_grid=[float(x) for x in grid],
                                      post_chain_sd=[float(x) for x in curve],
                                      jpeg="Allen tables" if q is not None else "quality %d" % rc.jpeg_quality)
            if est == est:
                out["renderer"]["noise_sd_gl"] = est
                out["notes"].append("noise_sd_gl matched through the camera chain (%s) to the median clipped SD "
                                    "of the background, %.3f grey levels after the chain; an upper bound (tissue "
                                    "texture adds to the real background SD)" % (out["noise_match"]["jpeg"], sd["p50"]))
            else:
                out["notes"].append("the background SD %.3f lies outside the simulated range; noise_sd_gl not "
                                    "suggested" % sd["p50"])
    out["notes"].append("black_level_gl and gain are not identified by brightfield crops; they stay configured")
    with open(os.path.join(out_dir, "camera_%s.json" % specimen), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pilot-summary", default="")
    ap.add_argument("--max-files", type=int, default=500)
    ap.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    sys.path.insert(0, HERE)
    from build_table import load_config
    out = run(a.cache_dir, a.out_dir, a.specimen, a.pilot_summary or None, a.max_files, load_config(a.config_json))
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
