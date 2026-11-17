"""Diagnostic for the same-node clearance (D-151): first numbers, before any SPEC entry. NOT pipeline code.

The node's own shadow on its measuring line, S_k(v_n | theta), is split into the part from the slabs within
beta of plane k (S_in) and the rest (S_out) by one condition on |zeta_j - z_k| inside the slab loop of
line_fast._profiles (the function's source is rewritten in memory; the module on disk is untouched).

Part 1 (split): per plane, the peaks of S, S_in, S_out, the share S_in / S at the dip's peak, B * S_in in
grey levels, the full widths at half maximum of S and S_in beside the tube's chord at that depth.
Part 2 (gain): the iteration 'add back B * S_out(theta_prev), then fit 1 - S_in(theta) by least squares',
linearised about a fixed point theta*: delta_t = M delta_(t-1), M = -(J_in^T J_in)^-1 J_in^T J_out, with
J_x = d S_x / d theta by central differences (theta = d, mu, v0, z_ax); its spectral radius, the same with
a damping lam ((1 - lam) I + lam M), and the standard errors of the fixed point against the full model's
least squares under white noise.

Setting: a flat capped tube (tilt 0, half-length 10 um, the cap test bypassed), hybrid absorption at the
inference's directions, line_fine_factor 4, the pilot's kernel table (Cell 7, cell 529878215) with the far
slope given, beta = 0.14 um, the line v = -26..26 x 0.1144 um with n_avg 0, planes k = -4..4, unweighted
sums. Nodes: (d, mu, axis depth) of 8089, 7448, 4924 after round 1 of the 17 Nov run.

Run (from the code folder "Passive Features/Diameter Re-measurement", about 4 minutes on one core):
    python3 checks/own_shadow_clearance_check.py
Keep this file pure ASCII.
"""
from __future__ import annotations

import dataclasses
import inspect
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import line_fast as LF  # noqa: E402

BETA_UM = 0.14
B_GL = 92.0
NOISE_GL = 1.0   # about the pilot's profile noise SD (0.85 to 1.31 gl, D-149.3)
NODES = (("8089", 2.441, 0.0751, -0.472), ("7448", 1.288, 0.1296, 0.513), ("4924", 4.454, 0.0264, -0.777))
TABLE_DELTA = (0.0, 0.28, 0.56, 0.84)
TABLE_SIGMA = (0.0800, 0.1536, 0.3924, 0.5620)


def _split_function():
    src = inspect.getsource(LF._profiles)
    old = "            deficit[k] += Ky @ (dA @ Kx.T)"
    if src.count(old) != 1:
        raise RuntimeError("line_fast._profiles changed: the slab loop's accumulation line was not found once")
    new = ("            if _BAND is None or (abs(zj - float(z_planes[k])) <= _BAND) == _INSIDE:\n"
           "                deficit[k] += Ky @ (dA @ Kx.T)")
    ns = dict(LF.__dict__)
    exec(src.replace(old, new).replace("def _profiles(", "def _profiles_split("), ns)
    return ns


NS = _split_function()


def make_cfg(gamma):
    base = default_config()
    return dataclasses.replace(
        base,
        renderer=dataclasses.replace(base.renderer, kernel_table_delta_um=TABLE_DELTA, kernel_table_sigma_um=TABLE_SIGMA,
                                     sigma_r0_um=TABLE_SIGMA[0], kernel_continuation_slope=gamma),
        inference=dataclasses.replace(base.inference, line_fine_factor=4))


def shadow(theta, cfg, band, inside, geom, cache):
    """S (n_planes, n_v), dimensionless: band None the whole shadow; else the slabs inside / outside the band."""
    d, mu, v0, z = theta
    o, yh, eu, v, ks = geom
    tube = G.Tube((o[0], o[1] + v0, z), d / 2, 0.0, 0.0, 1.0, 10.0, "axial")
    NS["_BAND"], NS["_INSIDE"] = band, inside
    tau = NS["_profiles_split"](tube, mu, ks, o, yh, eu, v, 0, cfg.acquisition.res0_um, cfg, "hybrid", cache, check=False)
    return 1.0 - tau


def fwhm(v, s):
    pk = float(s.max())
    if pk <= 0:
        return float("nan")
    above = np.flatnonzero(s >= 0.5 * pk)
    return float(v[above[-1]] - v[above[0]])


def part_split(geom):
    o, yh, eu, v, ks = geom
    for gamma in (0.79, 0.397):
        cfg, cache = make_cfg(gamma), LF.SectionCache()
        dz_pl = cfg.acquisition.dz_um
        for name, d, mu, z in NODES:
            th = (d, mu, 0.0, z)
            S = shadow(th, cfg, None, True, geom, cache)
            Sin = shadow(th, cfg, BETA_UM, True, geom, cache)
            Sout = shadow(th, cfg, BETA_UM, False, geom, cache)
            err = float(np.abs(S - Sin - Sout).max())
            print("\nSPLIT gamma %.3f node %s d %.3f um mu %.4f /um axis %+.3f um  max|S - S_in - S_out| %.1e" % (gamma, name, d, mu, z, err))
            print("   k  z_k-z_ax  peak_S  peak_S_in  peak_S_out  S_in/S  B*S_in_gl  FWHM_S  FWHM_S_in  chord_um")
            for i, k in enumerate(ks):
                n = int(np.argmax(S[i]))
                off = k * dz_pl - z
                chord = 2 * math.sqrt(max((d / 2) ** 2 - off ** 2, 0.0))
                print("  %+d   %+.3f   %.4f   %.4f     %.4f     %.3f    %.2f      %.2f    %.2f       %.2f"
                      % (k, off, S[i, n], Sin[i].max(), Sout[i].max(), Sin[i, n] / S[i, n], B_GL * Sin[i].max(),
                         fwhm(v, S[i]), fwhm(v, Sin[i]), chord))


def part_gain(geom):
    cfg, cache = make_cfg(0.79), LF.SectionCache()
    for name, d, mu, z in NODES[:2]:
        th = np.array([d, mu, 0.0, z])
        steps = np.array([0.02, 0.02 * mu, 0.02, 0.02])
        Jin, Jout = [], []
        for a in range(4):
            e = np.zeros(4)
            e[a] = steps[a]
            Jin.append((shadow(th + e, cfg, BETA_UM, True, geom, cache) - shadow(th - e, cfg, BETA_UM, True, geom, cache)).ravel() / (2 * steps[a]))
            Jout.append((shadow(th + e, cfg, BETA_UM, False, geom, cache) - shadow(th - e, cfg, BETA_UM, False, geom, cache)).ravel() / (2 * steps[a]))
        Jin, Jout = np.array(Jin).T, np.array(Jout).T
        J = Jin + Jout
        M = -np.linalg.solve(Jin.T @ Jin, Jin.T @ Jout)
        ev = np.linalg.eigvals(M)
        print("\nGAIN node %s theta* (d um, mu /um, v0 um, z_ax um) = %s" % (name, th))
        print("  eigenvalues of M: %s  spectral radius %.2f" % (np.array2string(np.sort(ev.real), precision=2), np.abs(ev).max()))
        for lam in (1.0, 0.5, 0.3, 0.25, 0.2, 0.15, 0.1):
            rho = np.abs(np.linalg.eigvals((1 - lam) * np.eye(4) + lam * M)).max()
            print("  damping %.2f: spectral radius %.2f" % (lam, rho))
        se_full = np.sqrt(np.diag(np.linalg.inv(J.T @ J))) * NOISE_GL / B_GL
        A = np.linalg.inv(Jin.T @ J) @ Jin.T
        se_fp = np.sqrt(np.diag(A @ A.T)) * NOISE_GL / B_GL
        print("  SE at B %.0f gl, white noise SD %.1f gl per sample, 9 planes x 53 samples:" % (B_GL, NOISE_GL))
        print("    full model's least squares   %s" % np.array2string(se_full, precision=4))
        print("    fixed point of the clearance %s  ratio %s" % (np.array2string(se_fp, precision=4), np.array2string(se_fp / se_full, precision=2)))
        cn = lambda X: np.linalg.cond(X / np.linalg.norm(X, axis=0))  # noqa: E731
        print("  condition number, columns at unit norm: J_in %.1f, J_in + J_out %.1f" % (cn(Jin), cn(J)))


def main():
    p = default_config().acquisition.res0_um
    geom = (np.array([437 * p, 211 * p]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.arange(-26, 27) * 0.1144, np.arange(-4, 5))
    part_split(geom)
    part_gain(geom)


if __name__ == "__main__":
    main()
