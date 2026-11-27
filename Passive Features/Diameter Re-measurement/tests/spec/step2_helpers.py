"""Fixtures for the spec-derived tests of step 2 (Blocks 52 to 56; tester, 2026-11-27). Written from specs/SPEC.md
("Kernel calibration, step 2"): synthetic PlaneScans and node records in the scans file's format, built by hand;
nothing here computes an expected value with the code under test.

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter import config as C  # noqa: E402
from allen_diameter.analysis.calibration import PlaneScan  # noqa: E402


def plane_scan(node_id, set_="near", k_star=10, P=None, z_sub=None, seed=0, nan_plane=None, bad_plane=None):
    """A synthetic PlaneScan over k* - P .. k* + P (P = 3 near, 16 far) with arbitrary finite values."""
    P = (3 if set_ == "near" else 16) if P is None else P
    rng = np.random.default_rng([seed, node_id])
    ks = np.arange(k_star - P, k_star + P + 1)
    n = ks.size
    om = 0.01 + 0.02 * np.abs(ks - k_star) ** 2 + rng.uniform(0, 0.001, n)
    status = ["ok"] * n
    if nan_plane is not None:
        om[nan_plane] = np.nan
    if bad_plane is not None:
        status[bad_plane] = "few_samples"
        om[bad_plane] = np.nan
    z_sub = 0.28 * k_star + 0.03 if z_sub is None else z_sub
    return PlaneScan(node_id=int(node_id), d_hat_um=0.5, phi_rad=0.05, alpha_hat=0.3, k_star=int(k_star),
                     z_sub_um=float(z_sub), ks=ks, z_um=ks * 0.28, omega_um2=om, vw_um2=om * 1.1,
                     area_gl_um=rng.uniform(5, 6, n), depth_gl=rng.uniform(10, 12, n),
                     B_bar=rng.uniform(200, 210, n), status=tuple(status), set=set_,
                     se_omega_um2=rng.uniform(1e-4, 2e-4, n), bg_sd_gl=rng.uniform(2, 3, n))


def pilot(k_star=10, z_sub=None, **kw):
    p = dict(k_star=int(k_star), z_sub_um=0.28 * k_star + 0.03 if z_sub is None else z_sub, cx_um=12.5, cy_um=-3.25,
             theta_rad=0.4, phi_rad=0.05, B_bar=205.0, mu_hat_per_um=0.6, d_hat_um=0.55, alpha_hat=0.33,
             fit_status="converged", flags=[], reg_verdict="on branch")
    p.update(kw)
    return p


def record(node_id, set_="near", d=0.5, **pkw):
    return dict(node_id=int(node_id), allen_diameter_um=d, set=set_, reasons="", scanned=True, pilot=pilot(**pkw))


def scans_meta(nodes, cfg=None):
    cfg = C.default_config() if cfg is None else cfg
    return dict(source="real", specimen="529878215", config=json.loads(cfg.to_json()),
                calibration=dataclasses.asdict(cfg.calibration), nodes=nodes)
