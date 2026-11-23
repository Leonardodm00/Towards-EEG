"""The calibrated kernel's values, read from the in-focus width calibration's output -- Block 36, amendment 22
(D-221.1, D-222.3, D-225.7).

read_calibrated_kernel(path) reads sigma_<S>.json, written by scripts/calibrate_sigma.py (Block 42's fits and their
summaries; strict JSON), and returns the values the switch --kernel-width calibrated applies through
kernel.with_kernel: the inverse-variance weighted mean of sigma_0 over the nodes in its summary (`summary.ivw_mean_um`)
and that of the growth scale c (`summary_growth_scale.ivw_mean`), with their SEs, their node counts and the file.
ValueError for a file that is not a finished calibration (`complete` false), that has no growth scale summary (written
before amendment 22: run calibrate_sigma.py again), whose summaries have n = 0, or whose means and SEs are not finite
(means > 0, SEs >= 0).

Pure ASCII, for the cluster. Parsing and validation only.
"""
from __future__ import annotations

import json
import math
import os


def _number(d, key, what):
    try:
        x = float(d[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError("%s: the summary has no number %r" % (what, key))
    return x


def read_calibrated_kernel(path):
    """-> dict(sigma0_um, se_sigma0_um, n_sigma0, growth_scale, se_growth_scale, n_growth_scale, file, specimen)."""
    path = os.path.abspath(str(path))
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict) or not d.get("complete", False):
        raise ValueError("%s is not a finished calibration (complete is not true): let calibrate_sigma.py end" % path)
    s, c = d.get("summary"), d.get("summary_growth_scale")
    if not isinstance(s, dict):
        raise ValueError("%s has no summary of sigma0" % path)
    if not isinstance(c, dict):
        raise ValueError("%s has no summary_growth_scale (written before Block 36 amendment 22): run "
                         "calibrate_sigma.py again" % path)
    n0, nc = int(_number(s, "n", path)), int(_number(c, "n", path))
    if n0 < 1 or nc < 1:
        raise ValueError("%s: no node in the summary (n = %d for sigma0, %d for c)" % (path, n0, nc))
    out = dict(sigma0_um=_number(s, "ivw_mean_um", path), se_sigma0_um=_number(s, "ivw_se_um", path), n_sigma0=n0,
               growth_scale=_number(c, "ivw_mean", path), se_growth_scale=_number(c, "ivw_se", path),
               n_growth_scale=nc, file=path, specimen=d.get("specimen"))
    for k in ("sigma0_um", "growth_scale"):
        if not (math.isfinite(out[k]) and out[k] > 0):
            raise ValueError("%s: the weighted mean %s = %r is not finite and > 0" % (path, k, out[k]))
    for k in ("se_sigma0_um", "se_growth_scale"):
        if not (math.isfinite(out[k]) and out[k] >= 0):
            raise ValueError("%s: %s = %r is not finite and >= 0" % (path, k, out[k]))
    return out
