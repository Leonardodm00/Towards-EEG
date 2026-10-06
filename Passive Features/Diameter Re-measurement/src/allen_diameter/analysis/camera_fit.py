"""Matching the camera chain's injected noise to a background SD measured
after the chain -- Block 11 in specs/SPEC.md (procedure s.3.6 step 6).

The renderer injects Gaussian noise of SD noise_sd_gl before rounding and
JPEG (model/camera.py). JPEG removes much of white noise, so the SD read on a
real background is not the injected SD: on the default chain (B = 210,
quality 85) an injected SD of 3 grey levels reads about 1.5 after it. The
match runs the camera chain itself on flat fields: the post-chain SD as a
function of the injected SD on a grid, then the inverse at the measured SD.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..model import camera


def clipped_sd(values, k=5.0):
    """SD (ddof 1) of the values within k robust SDs of their median (robust SD
    = 1.4826 MAD, at least 1 grey level, so integer data with MAD 0 keep their
    spread). The same statistic must be read on real and simulated backgrounds."""
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    med = float(np.median(x))
    rsd = max(1.0, 1.4826 * float(np.median(np.abs(x - med))))
    keep = x[np.abs(x - med) <= k * rsd]
    return float(np.std(keep, ddof=1)) if keep.size >= 2 else float("nan")


def post_chain_sd(noise_sd_gl, rcfg, rng, qtables=None, size=256):
    """clipped_sd of a flat field (tau = 1) after the camera chain with the injected SD noise_sd_gl."""
    rc = replace(rcfg, noise_sd_gl=float(noise_sd_gl))
    return clipped_sd(camera.camera_chain(np.ones((int(size), int(size))), rc, rng, qtables))


def noise_for_post_chain_sd(target_sd, rcfg, seed, qtables=None, grid=None, size=256):
    """The injected SD whose post-chain clipped SD equals target_sd: the curve
    post_chain_sd(grid) (one generator per grid point, default_rng([seed, i])),
    made non-decreasing by its running maximum, inverted by linear
    interpolation. Returns (noise_sd_gl or NaN when target_sd lies outside the
    curve's range, grid, curve)."""
    grid = np.linspace(0.0, 10.0, 41) if grid is None else np.asarray(grid, dtype=float)
    curve = np.array([post_chain_sd(s, rcfg, np.random.default_rng([int(seed), i]), qtables, size)
                      for i, s in enumerate(grid)])
    env = np.maximum.accumulate(np.nan_to_num(curve, nan=0.0))
    keep = np.r_[True, np.diff(env) > 0]          # strictly increasing points only
    xs, ys = env[keep], grid[keep]
    t = float(target_sd)
    if not (xs.size >= 2 and xs[0] <= t <= xs[-1]):
        return float("nan"), grid, curve
    return float(np.interp(t, xs, ys)), grid, curve
