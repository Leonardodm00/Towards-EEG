"""Check by computation (2026-11-18, D-170, handoff section 6): does Block 10's near fit (Eq. 4) recover the
growth of the kernel's variance G(delta) when the near-set nodes are as thick as the pilot's clear "thin" nodes
were fitted (0.77-1.09 um), rather than 0.15-0.25 um as the calibration's own phantoms are?

Truth: the renderer's kernel is the pilot's measured table, sigma_r = 0.080, 0.1536, 0.3924, 0.5620 um at
delta = 0, 0.28, 0.56, 0.84 um (continued with slope 0.79), so G(delta_m) = sigma_r(delta_m)^2 - 0.080^2 =
0.01719, 0.14758, 0.30944 um^2 at 0.28, 0.56, 0.84 um. For each range of true diameters, N flat phantoms
(calibrate_kernel.py synthetic: log d uniform on the range, alpha 0.15-0.4, tilt < 2 deg, the camera chain of the
default configuration, absorption partition_vertical, the background fitted on a +-8 um patch instead of +-25 um
to save time) are measured, scanned on k* +- 3 and fitted by calibrate_kernel.py fit; the fitted G and table are
compared with the truth. For the thick range the near set's Allen-diameter bound is raised to 1.0 um so the
phantoms enter the near set (on Allen's data such nodes enter with Allen's 2r <= 0.6 um while being thicker).

Run from the workstream folder (Passive Features/Diameter Re-measurement):
    python3 /home/claude/work/thick_check/thick_near_check.py --n 16 --workers 1
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH_DELTA = (0.28, 0.56, 0.84)
TRUTH_SIGMA = {0.0: 0.080, 0.28: 0.1536, 0.56: 0.3924, 0.84: 0.5620}
RANGES = (("thin", "0.15,0.25", "k_thin.json"), ("mid", "0.45,0.55", "k_thin.json"),
          ("thick", "0.75,0.95", "k_wide.json"))


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        raise SystemExit("failed: %s" % " ".join(cmd))
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    g_true = {d: TRUTH_SIGMA[d] ** 2 - TRUTH_SIGMA[0.0] ** 2 for d in TRUTH_DELTA}
    print("truth G (um^2): " + ", ".join("%.2f: %.5f" % (d, g_true[d]) for d in TRUTH_DELTA))
    for i, (name, rng_txt, cfg_name) in enumerate(RANGES):
        if a.only and name not in a.only.split(","):
            continue
        cfg = os.path.join(HERE, cfg_name)
        scans = os.path.join(HERE, "scans_%s.json" % name)
        calib = os.path.join(HERE, "calibration_%s.json" % name)
        out = run([sys.executable, "scripts/calibrate_kernel.py", "synthetic", "--set", "near", "--n", str(a.n),
                   "--seed", str(a.seed + i), "--d-range", rng_txt, "--out", scans, "--config-json", cfg,
                   "--workers", str(a.workers)])
        dhat = [ln for ln in out.splitlines() if ln.startswith("phantom ")]
        out = run([sys.executable, "scripts/calibrate_kernel.py", "fit", "--scans", scans, "--out", calib,
                   "--config-json", cfg])
        with open(calib) as f:
            c = json.load(f)
        kern = c.get("kernel") or {}
        tab = dict(zip(kern.get("kernel_table_delta_um", []), kern.get("kernel_table_sigma_um", [])))
        gro = c.get("growth") or {}
        print("== %s: true d %s um, %d phantoms" % (name, rng_txt, a.n))
        for ln in dhat[:a.n]:
            print("   " + ln)
        print("   growth keys: %s" % sorted(gro.keys())[:12])
        for d in TRUTH_DELTA:
            s = tab.get(d)
            if s is None:
                continue
            g = s * s - TRUTH_SIGMA[0.0] ** 2
            print("   delta %.2f um: sigma_r fitted %.4f (true %.4f, %+.1f %%); G fitted %.5f (true %.5f, %+.1f %%)"
                  % (d, s, TRUTH_SIGMA[d], 100 * (s / TRUTH_SIGMA[d] - 1), g, g_true[d], 100 * (g / g_true[d] - 1)))
        anc = c.get("anchor") or {}
        print("   anchor %s um, SE %s um" % (anc.get("sigma0_um"), anc.get("se_um")))
        for ln in out.splitlines():
            if "nodes," in ln and "growth fit" in ln:
                print("   " + ln.strip()[:200])


if __name__ == "__main__":
    main()
