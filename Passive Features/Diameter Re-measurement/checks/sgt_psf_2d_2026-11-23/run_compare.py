"""Wave 7 design check: the isotropic SGT PSF against the SGT-across / Gaussian-along kernel on Block 35's line read.

Usage: python3 run_compare.py <src dir> <out.json> [validate|compare|bias]
"""
import dataclasses
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
from allen_diameter.config import DiameterConfig  # noqa: E402
from allen_diameter.model import blur_kernels as bk, geometry, line_fast  # noqa: E402

import fft_line as F  # noqa: E402

NOISE_TAU = 1.1 / 150.0      # the identifiability check's sigma / B (gl / gl)
SHAPES = {"heavy": (2.0, 0.5, 1.0), "b13": (1.3, 0.0, 1.0), "flat4": (4.0, 0.0, 1.0), "tails1": (2.0, 1.0, 1.0)}


def config(f=4, dzeta=0.04):
    c = DiameterConfig()
    return dataclasses.replace(c, inference=dataclasses.replace(c.inference, line_fine_factor=f),
                               renderer=dataclasses.replace(c.renderer, dzeta_um=dzeta))


def line(theta):
    return (-math.sin(theta), math.cos(theta)), (math.cos(theta), math.sin(theta))


def setup(d, phi, theta, half_length, mu=1.0, v0=0.04, z_ax=0.1):
    tube = geometry.Tube((v0 * -math.sin(theta), v0 * math.cos(theta), z_ax), 0.5 * d, phi, theta, 1.0,
                         half_length, "axial")
    return tube, mu


def common_args(cfg, theta):
    p = cfg.acquisition.res0_um
    yh, eu = line(theta)
    return np.arange(-4, 5), (0.0, 0.0), yh, eu, np.arange(-26, 27) * p, 4, p


def validate(src_out):
    cfg = config()
    rows = []
    for d, phi, theta, L in ((0.8, 0.0, 0.0, 8.0), (0.8, 0.0, 0.52, 8.0), (0.4, 0.3, 0.52, 3.2)):
        tube, mu = setup(d, phi, theta, L)
        ks, o, yh, eu, v, n_avg, step = common_args(cfg, theta)
        t0 = time.perf_counter()
        ref = line_fast._profiles(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, "partition_vertical",
                                  line_fast.SectionCache(), check=False)
        t1 = time.perf_counter()
        got, used = F.render(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, F.gauss_transfer(), pad_um=9.0,
                             sigma_cut_um=10.0)
        t2 = time.perf_counter()
        rows.append(dict(d=d, phi=phi, theta=theta, half_length=L, max_abs_diff=float(np.max(np.abs(got - ref))),
                         min_tau=float(ref.min()), pairs=used, s_line_fast=t1 - t0, s_fft=t2 - t1))
        print(rows[-1], flush=True)
    return rows


def compare(cases, shapes, sigma_cut=1.3, pad=8.0):
    cfg = config()
    tables = {}
    for name in shapes:
        b, x, z = SHAPES[name]
        tables[name] = (F.unit_cf_table(lambda xx: bk.sgt_unit_pdf(xx, b, x, z)), bk.sgt_unit_fwhm(b, x, z))
    rows = []
    for d, phi, theta in cases:
        tube, mu = setup(d, phi, theta, 8.0)
        ks, o, yh, eu, v, n_avg, step = common_args(cfg, theta)
        t0 = time.perf_counter()
        full = line_fast._profiles(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, "partition_vertical",
                                   line_fast.SectionCache(), check=False)
        tg, used = F.render(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, F.gauss_transfer(), pad, sigma_cut)
        for name in shapes:
            table, W = tables[name]
            ti, _ = F.render(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, F.iso_transfer(table, W), pad, sigma_cut)
            ts, _ = F.render(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, F.sep_transfer(table, W, yh, eu), pad,
                             sigma_cut)
            diff = ti - ts
            rows.append(dict(d=d, phi=phi, theta=theta, shape=name,
                             max_iso_sep_noise=float(np.max(np.abs(diff)) / NOISE_TAU),
                             rms_iso_sep_noise=float(np.sqrt(np.mean(diff ** 2)) / NOISE_TAU),
                             max_iso_gauss_noise=float(np.max(np.abs(ti - tg)) / NOISE_TAU),
                             excluded_pairs_max_noise=float(np.max(np.abs(full - tg)) / NOISE_TAU),
                             pairs_used=used, seconds=time.perf_counter() - t0))
            print(json.dumps(rows[-1]), flush=True)
    return rows


if __name__ == "__main__":
    out, mode = sys.argv[2], sys.argv[3]
    if mode == "validate":
        res = validate(out)
    else:
        cases = [(d, phi, th) for d in (0.4, 0.8, 1.5) for phi in (0.0, 0.3, 0.6) for th in (0.0, 0.52)]
        res = compare(cases, ("heavy", "b13", "flat4", "tails1"))
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
