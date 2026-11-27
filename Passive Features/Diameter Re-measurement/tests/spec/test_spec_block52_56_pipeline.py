"""End-to-end and boundary tests of step 2's wave 1 (Blocks 52 -> 53 -> 54 -> 55 -> 56; tester, 2026-11-27). There is
no entry point yet (SPEC section 6, Blocks 57 to 60 not built): the chain is driven as the common definitions describe
the near stage, on the minimal case the tester derived from the SPEC:

  "real" data  two flat near-set phantoms (d 0.40 and 0.50 um) rendered by render.synthetic_block under a known
               spline table (the truth, theta_true) with the default camera noise, and scanned by
               calibration.scan_node (Block 10; the producer of the scans file)
  Block 52     the scans file written by calibration_io.save_scans with the meta of calibrate_kernel.py real
               (config, calibration, nodes from calibration.pilot_record), the chosen nodes as a CSV; read back
  Block 53     build_twins, split_twins, fit_planes "near"
  Block 54/55  per twin and trial table: render_twin then twin_statistics (2 replicates)
  Block 56     fit_shared_local: shared theta (3), one local z_ax per twin, Step2Config's steps and bounds

Absorption "partition_vertical" in the truth and through Step2Config.absorption (the swappability check of the
directive audit; the hybrid costs 5 times more per render). Patch +-8 um.

Oracles: every output finite and of the shape and unit the SPEC states; the model a deterministic function of its
unknowns (common random numbers: two evaluations bit for bit, and the whole fit twice bit for bit); the boundary
contracts (scan -> Twin fields, TwinStack.ks = the scan's stage planes, CropFrame from TwinStack); the objective lower
at the truth than at a start 40 % wide; the fit's estimate within its bounds and its objective no higher than the
start's.

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import step2_helpers as H  # noqa: F401  (sys.path)
from allen_diameter import config as C
from allen_diameter.analysis import calibration as CAL
from allen_diameter.analysis import step2_fit as F
from allen_diameter.analysis import step2_measure as M
from allen_diameter.analysis import step2_render as SR
from allen_diameter.analysis import step2_twins as T
from allen_diameter.analysis.node_pipeline import Branch
from allen_diameter.loading import calibration_io
from allen_diameter.loading import step2_io as IO
from allen_diameter.model import geometry, render

THETA_TRUE = (0.13, 0.36, 0.58)
CELL = 529878215
_CACHE = {}


def _truth_cfg():
    c = C.default_config()
    c = dataclasses.replace(c, calibration=dataclasses.replace(c.calibration, patch_half_um=8.0))
    s2 = dataclasses.replace(c.step2, absorption="partition_vertical", noise_replicates=2, max_nfev=3)
    c = dataclasses.replace(c, step2=s2)
    r = SR.trial_renderer(c.renderer, c.renderer.sigma_r0_um, THETA_TRUE, None, s2, "near")
    return dataclasses.replace(c, renderer=r)


def _real(tmp_path_factory):
    if "real" in _CACHE:
        return _CACHE["real"]
    cfg = _truth_cfg()
    rng = np.random.default_rng(20261127)
    scans, recs, truth = [], [], {}
    for nid, (d, mu, c, theta) in {101: (0.40, 1.0, (0.02, -0.03, 0.05), 0.7),
                                   202: (0.50, 0.8, (60.01, 40.02, 0.31), 2.1)}.items():
        tube = geometry.Tube(c, d / 2, 0.0, theta, 1.0, 40.0, cfg.renderer.end_cut)
        t = geometry.axis_direction(0.0, theta)
        br = Branch.from_points(np.asarray(c)[None, :] + np.arange(-12, 13)[:, None] * 1.0 * t[None, :], d / 2)
        pad = int(math.ceil(cfg.renderer.pad_um / cfg.acquisition.res0_um))

        def provider(left, top, width, height, k_lo, k_hi, tube=tube, mu=mu):
            return render.synthetic_block(tube, mu, np.arange(k_lo, k_hi + 1), left, top, width, height, cfg, rng,
                                          pad)
        res, scan, reasons = CAL.scan_node(br, 12, provider, cfg)
        assert scan is not None and scan.set == "near", reasons
        scan = dataclasses.replace(scan, node_id=nid)
        scans.append(scan)
        recs.append(dict(node_id=nid, allen_diameter_um=d, set="near", reasons="", scanned=True,
                         pilot=CAL.pilot_record(res)))
        truth[nid] = dict(d=d, mu=mu, z=c[2])
    out = tmp_path_factory.mktemp("step2")
    sp = out / ("scans_%d.json" % CELL)
    calibration_io.save_scans(scans, str(sp), dict(source="real", specimen=str(CELL), config=json.loads(cfg.to_json()),
                                                   calibration=dataclasses.asdict(cfg.calibration), nodes=recs))
    cp = out / "chosen.csv"
    cp.write_text("specimen_id,node_id\n%d,202\n%d,101\n" % (CELL, CELL))
    _CACHE["real"] = (cfg, sp, cp, scans, truth)
    return _CACHE["real"]


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    cfg, sp, cp, scans, truth = _real(tmp_path_factory)
    chosen = IO.read_chosen_nodes(cp)
    assert chosen == {CELL: (101, 202)}
    cs = IO.load_cell(CELL, sp)
    twins = T.build_twins(CELL, cs.scans, cs.nodes, chosen[CELL])
    twins = T.split_twins(twins, 0.0, cfg.step2.split_seed)
    return cfg, cs, twins, truth


def _model_factory(cfg, cs, twins, masks):
    s2 = cfg.step2
    s0 = float(cs.config.renderer.sigma_r0_um)

    def model(shared, local):
        out = []
        r = SR.trial_renderer(cs.config.renderer, s0, tuple(shared), None, s2, "near")
        for i, t in enumerate(twins):
            st = SR.render_twin(t, float(local[i, 0]), r, cs.config, s2)
            stats = M.twin_statistics(t, st, cs.config, s2)
            out.append(stats["omega_um2"][masks[i]])
        return np.concatenate(out)
    return model


def test_boundaries_52_to_55(chain):
    cfg, cs, twins, truth = chain
    assert cs.config.signature_hash("full") == cfg.signature_hash("full")
    assert cs.config.renderer.kernel_family == "gaussian_table_spline"
    by = {s.node_id: s for s in cs.scans}
    for t in twins:
        s = by[t.node_id]
        assert t.d_um == truth[t.node_id]["d"] and t.set == "near" and t.role == "train"
        assert_array_equal(t.ks, s.ks) and t.k_star == s.k_star and t.z_sub_um == s.z_sub_um
        assert_array_equal(t.omega_um2, s.omega_um2)
        st = SR.render_twin(t, t.z_sub_um, cs.config.renderer, cs.config, cfg.step2)
        # TwinStack.ks are the scan's stage planes, so measure_twin accepts them; the patch equals the scan's
        assert_array_equal(st.ks, CAL.stage_planes(t, cs.config, "near"))
        sc = M.measure_twin(t, st, cs.config, cfg.step2, noise=False)
        assert sc.ks.tolist() == s.ks.tolist() and sc.set == "near"
        assert np.isfinite(sc.B_bar).all()
        # the twin's background reads back the real scan's per plane (the camera mapping, 0.5 gl)
        assert np.nanmax(np.abs(sc.B_bar - s.B_bar)) <= 0.5


def test_end_to_end_near_stage(chain):
    cfg, cs, twins, truth = chain
    s2 = cfg.step2
    masks = [T.fit_planes(t, s2, "near") for t in twins]
    assert all(m.sum() >= 5 for m in masks)
    y = np.concatenate([t.omega_um2[m] for t, m in zip(twins, masks)])
    group = np.concatenate([np.full(m.sum(), i) for i, m in enumerate(masks)])
    model = _model_factory(cfg, cs, twins, masks)
    s0 = float(cs.config.renderer.sigma_r0_um)
    z_true = np.array([[truth[t.node_id]["z"]] for t in twins])
    # determinism (common random numbers): the model is a function of its unknowns
    a1 = model(np.array(THETA_TRUE), z_true)
    a2 = model(np.array(THETA_TRUE), z_true)
    assert_array_equal(a1, a2)
    assert a1.shape == y.shape and np.isfinite(a1).all()
    start = np.array(THETA_TRUE) * 1.4
    zs = np.array([[t.z_sub_um] for t in twins])
    b = model(start, zs)
    # at the start some planes have no core width in either replicate (NaN by the rule of twin_statistics): Block 56
    # drops them for the whole fit; compare the objectives over the planes finite at both points
    both = np.isfinite(a1) & np.isfinite(b)
    S_true, S_start = float(np.sum((y - a1)[both] ** 2)), float(np.sum((y - b)[both] ** 2))
    print("\nobjective at the truth %.3e, at the start %.3e over %d of %d planes; start NaN at %s;\n"
          "real omega %s\nreal - twin at the truth: %s"
          % (S_true, S_start, both.sum(), y.size, np.flatnonzero(~np.isfinite(b)).tolist(), np.round(y, 4).tolist(),
             np.round(y - a1, 5).tolist()))
    assert S_true < S_start
    # amendment 3 (tester, 2026-11-27 22:11; SPEC Block 55, F2 of the 1904 report): the twin's scan runs without
    # Block 10.G's goodness-of-fit screen, so every plane the real scan kept has a twin core width at the truth and
    # at the start (2 replicates, ceil(2/2) = 1 finite needed): no plane, in particular no in-focus plane, is
    # dropped by the start
    assert np.isfinite(a1).all() and np.isfinite(b).all(), np.flatnonzero(~np.isfinite(b)).tolist()
    lo = np.full(3, s0 + s2.knot_min_margin_um)
    hi = np.full(3, s2.knot_max_um)
    lb, ub = zs - s2.z_ax_bound_um, zs + s2.z_ax_bound_um
    kw = dict(local0=zs, local_bounds=(lb, ub), diff_step_shared=np.full(3, s2.diff_step_knot_um),
              diff_step_local=np.array([s2.diff_step_z_um]), max_nfev=s2.max_nfev)
    r = F.fit_shared_local(model, y, np.ones_like(y), group, start, (lo, hi), **kw)
    print("fit: shared %s (truth %s), local %s (truth %s), cost %.3e, nfev %d, calls %d, status %d, se_cluster %s"
          % (r.shared, THETA_TRUE, r.local.ravel(), z_true.ravel(), r.cost, r.nfev, r.n_model_calls, r.status,
             r.se_cluster))
    assert r.shared.shape == (3,) and r.local.shape == (2, 1)
    assert np.all(r.shared >= lo) and np.all(r.shared <= hi)
    assert np.all(r.local >= lb) and np.all(r.local <= ub)
    assert np.all(np.isfinite(r.residual[r.kept])) and r.kept.sum() == both.sum()
    assert r.cost <= 0.5 * S_start + 1e-15
    assert r.jac.shape == (r.kept.sum(), 3 + 2)
    r2 = F.fit_shared_local(model, y, np.ones_like(y), group, start, (lo, hi), **kw)
    assert_array_equal(r.shared, r2.shared)
    assert_array_equal(r.local, r2.local)
    assert r.n_model_calls == r2.n_model_calls
    _CACHE["fit"] = r


def test_far_boundary_54_to_55_and_fit_planes():
    # Block 53 -> 54 -> 55 for a far twin: trial_renderer "far" (linear, gamma), render_twin over the far stage's
    # planes (k* +- far_offsets_planes), measure_twin through far_plane_scan; fit_planes "far" selects
    # 4 <= |k - k*| <= 16, the same lower edge as Block 15's far_min_k_offset (amendment)
    c = C.default_config()
    c = dataclasses.replace(c, calibration=dataclasses.replace(c.calibration, patch_half_um=10.5))
    s2 = dataclasses.replace(c.step2, absorption="partition_vertical")
    P = c.calibration.far_offsets_planes
    ks = np.arange(5 - P, 5 + P + 1)
    n = ks.size
    z = np.zeros(n)
    tw = T.Twin(cell_id=1, node_id=9, set="far", role="train", d_um=0.7, mu_per_um=0.6, k_star=5, z_sub_um=1.42,
                cx_um=3.01, cy_um=-2.02, theta_rad=0.3, phi_rad=0.02, B_pilot_gl=200.0, d_pilot_um=0.7,
                alpha_pilot=0.42, ks=ks, B_gl=z + 200.0, noise_sd_gl=z + 1.5, omega_um2=z + 0.5, se_omega_um2=z,
                area_gl_um=z, status=("ok",) * n)
    r = SR.trial_renderer(c.renderer, c.renderer.sigma_r0_um, (0.13, 0.36, 0.58), 0.8, s2, "far")
    st = SR.render_twin(tw, 1.42, r, c, s2)
    assert st.tau.shape[0] == 2 * P + 1 == 33
    sc = M.measure_twin(tw, st, c, s2)
    assert sc.set == "far" and sc.ks.tolist() == ks.tolist()
    fp = T.fit_planes(tw, s2, "far")
    j = np.abs(ks - 5)
    assert_array_equal(fp, (j >= c.calibration.far_min_k_offset) & (j <= P))
    ok = np.array([s == "ok" for s in sc.status])
    print("\nfar twin statuses:", sc.status)
    assert ok[fp].sum() >= 6
