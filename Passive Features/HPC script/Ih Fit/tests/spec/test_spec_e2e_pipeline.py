"""End to end (step 6 of the tester's procedure): Block 6's driver
(run_noise_fit.py) is not written (SPEC Block 6: drafted), so the pipeline is
run through the per-cell chain Block 6 specifies (D-123.1) on the public API:

    load_ss_windows -> ss_noise_windows -> load_ls_prewindows ->
    CellLayout.from_cell -> CellTarget.from_cell -> fit_cell -> fit_sweeps

on the tester's minimal writer-like cell (2 SS sweeps, 3 LS sweeps, planted
law theta_cf and tilts), at reduced sizes, twice with the same seeds. Checks:
it finishes; every output exists, is finite where the spec says so, has the
stated shapes and units; the two runs are identical bit for bit (seconds
excluded); the law lands near the planted one (loose: reduced M and R).
"""
import json
import math

import numpy as np
import pytest

import _fixtures as fx
import noise_fit as nf
import noise_sweep_fit as nsf
import noise_windows as nw

SIZES = dict(M_binding=60, R_iter=40, R_final=60, B_boot=400, k_max=2, max_nfev=6)


def _run_once(d):
    ss = nw.load_ss_windows(d)
    nwin = nw.ss_noise_windows(ss)
    ls = nw.load_ls_prewindows(d)
    layout = nf.CellLayout.from_cell(nwin, ls)
    target = nf.CellTarget.from_cell(nwin, ls, layout)
    cfg = nf.NoiseFitConfig(**SIZES)
    cell = nf.fit_cell(target, layout, cfg)
    stage = nsf.fit_sweeps(cell, target, layout, None, nsf.SweepStageConfig(B_v=3, M_refit=30))
    return nwin, layout, target, cell, stage


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k != "seconds"}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


@pytest.mark.slow
def test_e2e_minimal_cell_twice(tmp_path):
    d = fx.write_cell(str(tmp_path), specimen_id=555, ss_tilts=(0.1, -0.05),
                      drop=[(0, 3), (0, 4), (1, 10), (1, 11), (1, 12)],
                      ls=((60, -50.0), (61, 30.0), (63, -90.0)), ls_tilts=(0.05, -0.02, 0.08), seed=4242)
    a = _run_once(d)
    b = _run_once(d)
    nwin, layout, target, cell, stage = a
    # --- Block 3 outputs
    assert nwin.X_tilde.shape == (69, layout.m) and np.all(np.isfinite(nwin.X_tilde))
    assert 80.0 <= nwin.start.t_ss_ms < 200.0 and np.isfinite(target.q_hat).all() and target.q_hat.shape == (15,)
    # --- Block 4 outputs: units mV, --, mV, s; finite report fields
    law = cell.law
    assert 0.01 < law.sigma_f < 0.2 and 0.0 < law.rho_f < 1.0 and 0.005 < law.sigma_s < 0.5 and 1e-3 < law.tau_s < 1.0
    for v in (cell.J, cell.p_value, cell.kappa):
        assert math.isfinite(v)
    assert cell.Sigma.shape == (15, 15) and cell.dof == 11 and cell.n_solves == len(cell.history)
    assert sorted(cell.per_sweep_cov) == [31, 35, 60, 61, 63]
    # loose recovery (reduced M, R): within the F7 window of SPEC Block 4 (0.04, 0.03, 0.4, 1.0) in u
    du = np.abs(nf.law_to_u(law) - nf.law_to_u(nf.NoiseLaw(*fx.THETA_CF)))
    assert np.all(du <= np.array([0.04, 0.03, 0.4, 1.0])), du
    # --- Block 9 outputs
    assert [(r["protocol"], r["sweep"]) for r in stage.rows] == [("ss", 31), ("ss", 35), ("ls", 60), ("ls", 61), ("ls", 63)]
    for r in stage.rows:
        tw = stage.twin_laws[(r["protocol"], r["sweep"])]
        assert tw.tau_s == law.tau_s
        if not r["failed"]:
            assert math.isfinite(r["v_log_sigma_s"]) and r["v_log_sigma_s"] > 0
            assert tw.sigma_s == pytest.approx(r["sigma_s_tilde_mV"], rel=1e-15)
            assert (tw.sigma_f, tw.rho_f) == (r["sigma_f_mV"], r["rho_f"])
        else:
            assert tw == law
    tab = stage.table_rows()
    assert len(tab) == 5 and all(set(t) >= {"noise_sweep_sigma_s_mV", "noise_sweep_w", "noise_tilt_mV_per_s"} for t in tab)
    # --- JSON-ready and reproducible bit for bit
    ja = json.dumps(_strip(cell.to_dict()), allow_nan=False, sort_keys=True)
    jb = json.dumps(_strip(b[3].to_dict()), allow_nan=False, sort_keys=True)
    assert ja == jb
    sa = json.dumps(_strip(stage.to_dict()), allow_nan=False, sort_keys=True)
    sb = json.dumps(_strip(b[4].to_dict()), allow_nan=False, sort_keys=True)
    assert sa == sb
    np.testing.assert_array_equal(nwin.X_tilde, b[0].X_tilde)
