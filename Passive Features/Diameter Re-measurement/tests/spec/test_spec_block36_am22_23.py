"""Spec-derived tests of Block 36 amendments 22 and 23 (wave 7; D-221, D-222, D-224, D-225) -- integration tester,
2026-11-23 13:55: the kernel's in-focus width, fixed or calibrated (scripts/kernel_width.py, loading/sigma_io.py,
scripts/calibrate_sigma.py's growth-scale summary, the switch in clean_cell.py, infer_nodes.py, fit_soma.py), and the
SGT stage with the fine-lattice error bars (node_inputs.forward_factory / forward_cov_factory, infer_nodes.py's
outputs). Expected values from the spec's formulas written out (tests/spec/wave7_refs.py), from line_fast called by
hand where the spec defines a factory as that call, or from a Gaussian run of the same script where the spec says the
SGT run keeps the Gaussian's numbers; never from the function under test itself.

Run: python3 -m pytest -q tests/spec/test_spec_block36_am22_23.py   (the end-to-end part about 10-15 min)
Pure ASCII.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import nc_helpers as H
import wave7_refs as R
from allen_diameter import config as C
from allen_diameter.analysis import node_inputs
from allen_diameter.model import line_fast, line_sgt

WS = H.WS
SCRIPTS = WS / "scripts"
FAST = {"cleaning": {"shadow_absorption": "partition_vertical", "shadow_fine_factor": 2, "soma_mu_per_um": 0.0,
                     "example_counts": [2, 0, 0, 0, 0], "max_rounds": 1, "refine_max_nfev": 40},
        "inference": {"absorption": "partition_vertical", "ray_n_rho": 4, "ray_n_psi": 8,
                      "multistart_factors": [1.0], "max_nfev": 60}}


def sigma_json(path, s0=0.12, se0=0.004, n0=5, c=1.3, sec=0.05, nc=5, complete=True, drop_c=False, **over):
    """A calibration file of calibrate_sigma.py's form (Block 36 amendment 22), written by hand."""
    d = dict(specimen="7", complete=complete,
             summary=dict(n=n0, median_um=s0, mad_um=0.01, ivw_mean_um=s0, ivw_se_um=se0, chi2_cons=1.0, dof=n0 - 1),
             summary_growth_scale=dict(n=nc, median=c, mad=0.1, ivw_mean=c, ivw_se=sec, chi2_cons=1.0, dof=nc - 1))
    if drop_c:
        d.pop("summary_growth_scale")
    for k, v in over.items():
        sec_name, key = k.split("__")
        d[sec_name][key] = v
    with open(path, "w") as f:
        json.dump(d, f)
    return str(path)


def _run(args, check=True):
    r = subprocess.run([sys.executable] + [str(a) for a in args], cwd=str(WS), capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg"))
    if check and r.returncode != 0:
        raise AssertionError("%s failed (%d):\n%s\n%s" % (args, r.returncode, r.stdout[-3000:], r.stderr[-3000:]))
    return r


# ------------------------------------------------- the switch (amendment 22) ---

def test_fixed_keeps_the_configuration():
    """B22.2: "fixed" (and the default None) leaves the configuration as it was [exact]; the record names the mode,
    the configuration's sigma_r(0), c = 1 and the base renderer."""
    import kernel_width
    cfg = C.default_config()
    for mode in (None, "fixed"):
        out, rec = kernel_width.apply(cfg, mode, "")
        assert out == cfg and out.renderer == cfg.renderer
        assert rec["mode"] == "fixed" and rec["sigma0_um"] == cfg.renderer.sigma_r0_um and rec["growth_scale"] == 1.0
        assert rec["base_renderer"] == dataclasses.asdict(cfg.renderer)


def test_calibrated_table_by_hand(tmp_path):
    """B22.2 / B22.4: "calibrated" gives the table of Block 4 amendment 2 at sigma_0 = summary.ivw_mean_um and
    c = summary_growth_scale.ivw_mean written out by hand -- sigma'(d)^2 = c (sigma(d)^2 - sigma(0)^2) + sigma_0^2,
    sigma'(0) = sigma_0, the linear slope times sqrt(c) [1e-12] -- and records the values, their SEs and counts, the
    file and the base renderer; the base is the configuration's table before any re-anchoring (base=...)."""
    import kernel_width
    cfg = C.default_config()
    js = sigma_json(tmp_path / "s.json", s0=0.123, se0=0.006, n0=7, c=1.37, sec=0.08, nc=6)
    out, rec = kernel_width.apply(cfg, "calibrated", js)
    r0 = cfg.renderer
    ref, s0, slope = R.with_kernel_table(r0.kernel_table_delta_um, r0.kernel_table_sigma_um,
                                         r0.kernel_continuation_slope, 0.123, 1.37)
    assert_allclose(out.renderer.kernel_table_sigma_um, ref, rtol=1e-12)
    assert out.renderer.sigma_r0_um == s0 == out.renderer.kernel_table_sigma_um[0]
    assert_allclose(out.renderer.kernel_continuation_slope, slope, rtol=1e-12)
    assert tuple(out.renderer.kernel_table_delta_um) == tuple(r0.kernel_table_delta_um)
    assert dataclasses.replace(out, renderer=r0) == cfg                 # nothing else changed
    for k, v in dict(mode="calibrated", sigma0_um=0.123, se_sigma0_um=0.006, n_sigma0=7, growth_scale=1.37,
                     se_growth_scale=0.08, n_growth_scale=6).items():
        assert rec[k] == v, k
    assert os.path.abspath(rec["file"]) == os.path.abspath(js)
    assert rec["base_renderer"] == dataclasses.asdict(r0)
    # re-anchoring an already re-anchored renderer from the base gives the base's re-anchoring, not a compounded one
    out2, _ = kernel_width.apply(out, "calibrated", js, base=dataclasses.asdict(r0))
    assert_allclose(out2.renderer.kernel_table_sigma_um, ref, rtol=1e-12)


@pytest.mark.parametrize("kw", [dict(complete=False), dict(n0=0), dict(nc=0), dict(drop_c=True),
                                dict(summary__ivw_mean_um=None), dict(summary_growth_scale__ivw_mean=None),
                                dict(summary__ivw_mean_um=0.0), dict(summary_growth_scale__ivw_mean=-1.0),
                                dict(summary__ivw_mean_um=float("inf")), dict(summary_growth_scale__ivw_mean=float("nan"))])
def test_calibrated_refusals(tmp_path, kw):
    """B22.3: refused with ValueError (the scripts turn it into an argparse error): a file whose complete is false,
    whose summaries have n = 0 or lack the growth scale's, whose means are not finite and > 0."""
    import kernel_width
    js = sigma_json(tmp_path / "s.json", **kw)
    with pytest.raises(ValueError):
        kernel_width.apply(C.default_config(), "calibrated", js)


def test_calibrated_without_file_refused():
    import kernel_width
    with pytest.raises(ValueError):
        kernel_width.apply(C.default_config(), "calibrated", "")
    with pytest.raises(ValueError):
        kernel_width.apply(C.default_config(), "sideways", "")


def test_clean_cell_switch_refusals(tmp_path):
    """B22.3 through clean_cell.py: "calibrated" with --sigma-r0-um, "calibrated" without --sigma-json and with a file
    whose complete is false end with an argparse error (status 2) naming the problem, before any work."""
    js = sigma_json(tmp_path / "s.json")
    bad = sigma_json(tmp_path / "bad.json", complete=False)
    sc = SCRIPTS / "clean_cell.py"
    base = [sc, "survey", "--synthetic", "--specimen", "7", "--B", "210", "--mu", "1.0", "--out", tmp_path / "o"]
    r1 = _run(base + ["--kernel-width", "calibrated", "--sigma-json", js, "--sigma-r0-um", "0.1"], check=False)
    r2 = _run(base + ["--kernel-width", "calibrated"], check=False)
    r3 = _run(base + ["--kernel-width", "calibrated", "--sigma-json", bad], check=False)
    for r, word in ((r1, "exclusive"), (r2, "sigma-json"), (r3, "complete")):
        assert r.returncode == 2 and word in r.stderr, (r.returncode, r.stderr[-500:])
    assert not (tmp_path / "o" / "contamination_7.csv").exists()


# ------------------------------------- calibrate_sigma's growth-scale summary ---

def test_growth_scale_summary_by_hand():
    """B22.1: summary_growth_scale = summarize's statistics over the nodes ok with a finite c and a finite SE(c) > 0,
    keys n, median, mad, ivw_mean, ivw_se, chi2_cons, dof (no unit) -- recomputed by hand [1e-12]."""
    import calibrate_sigma
    rows = [dict(node_id=1, growth_scale=1.2, se_growth_scale=0.1, ok=True),
            dict(node_id=2, growth_scale=0.9, se_growth_scale=0.05, ok=True),
            dict(node_id=3, growth_scale=1.5, se_growth_scale=0.2, ok=True),
            dict(node_id=4, growth_scale=2.0, se_growth_scale=0.1, ok=False),          # not ok
            dict(node_id=5, growth_scale=float("nan"), se_growth_scale=0.1, ok=True),   # c not finite
            dict(node_id=6, growth_scale=1.1, se_growth_scale=0.0, ok=True),            # SE 0
            dict(node_id=7, growth_scale=1.1, se_growth_scale=None, ok=True),           # SE missing
            dict(node_id=8, growth_scale=1.0, se_growth_scale=0.07, ok=True)]
    got = calibrate_sigma.growth_scale_summary(rows)
    assert set(got) == {"n", "median", "mad", "ivw_mean", "ivw_se", "chi2_cons", "dof"}
    keep = [0, 1, 2, 7]
    ref = R.summary_ref([rows[i]["growth_scale"] for i in keep], [rows[i]["se_growth_scale"] for i in keep],
                        [True] * 4)
    for k in ("median", "mad", "ivw_mean", "ivw_se", "chi2_cons"):
        assert_allclose(got[k], ref[k], rtol=1e-12, err_msg=k)
    assert int(got["n"]) == 4 and int(got["dof"]) == 3


def test_growth_scale_summary_reaches_the_switch(tmp_path):
    """Boundary Block 42 -> calibrate_sigma -> sigma_io -> the switch: write_outputs' final file holds `summary` and
    `summary_growth_scale` (strict JSON), and read_calibrated_kernel reads sigma_0 from summary.ivw_mean_um and c from
    summary_growth_scale.ivw_mean of that file; the file written before the end (summary None) has complete false
    and is refused."""
    import calibrate_sigma
    from allen_diameter.loading import sigma_io
    def row(**kw):
        r = {c: float("nan") for c in calibrate_sigma.COLUMNS}
        r.update(at_bound="", message="", n_planes=5, k_star=0, dof=10, n_eval=40)
        r.update(kw)
        return r
    rows = {1: row(node_id=1, status="converged", sigma0_um=0.11, se_sigma0_um=0.01, growth_scale=1.2,
                   se_growth_scale=0.1, ok=True, in_summary=True),
            2: row(node_id=2, status="converged", sigma0_um=0.13, se_sigma0_um=0.02, growth_scale=0.9,
                   se_growth_scale=0.05, ok=True, in_summary=True)}
    a = argparse.Namespace(specimen="7", cleaning_dir=str(tmp_path))
    scfg = C.default_config().sigma_calibration
    from allen_diameter.analysis import sigma_calibration as SC
    summ = SC.summarize(np.array([0.11, 0.13]), np.array([0.01, 0.02]), np.array([True, True]))
    calibrate_sigma.write_outputs(str(tmp_path), a, [1, 2], rows, scfg, None, 0.0)
    with pytest.raises(ValueError):
        sigma_io.read_calibrated_kernel(tmp_path / "sigma_7.json")
    calibrate_sigma.write_outputs(str(tmp_path), a, [1, 2], rows, scfg, summ, 0.0)
    d = json.load(open(tmp_path / "sigma_7.json"))
    assert d["complete"] is True and "summary_growth_scale" in d
    vals = sigma_io.read_calibrated_kernel(tmp_path / "sigma_7.json")
    ref = R.summary_ref([1.2, 0.9], [0.1, 0.05], [True, True])
    assert_allclose(vals["sigma0_um"], summ["ivw_mean_um"], rtol=1e-15)
    assert_allclose(vals["growth_scale"], ref["ivw_mean"], rtol=1e-12)
    assert vals["n_sigma0"] == 2 and vals["n_growth_scale"] == 2


# ---------------------------------------------- the synthetic preparation ---

@pytest.fixture(scope="module")
def prep(tmp_path_factory):
    """survey + clean (1 round) on the synthetic cell with the fast configuration (the pilot-run suite's, with two
    example nodes); and a second preparation run with --kernel-width calibrated (sigma_0 0.10 um, c 1.25)."""
    base = tmp_path_factory.mktemp("am22_23")
    cfgj = base / "fast.json"
    cfgj.write_text(json.dumps(FAST))
    sc = SCRIPTS / "clean_cell.py"
    P = base / "prep"
    _run([sc, "survey", "--synthetic", "--specimen", "7", "--B", "210", "--mu", "1.0", "--out", P,
          "--config-json", cfgj])
    _run([sc, "clean", "--synthetic", "--specimen", "7", "--examples", P / "examples_7.json", "--out", P,
          "--rounds", "1", "--config-json", cfgj])
    js = sigma_json(base / "cal.json", s0=0.10, se0=0.003, n0=4, c=1.25, sec=0.04, nc=4)
    Q = base / "prep_cal"
    _run([sc, "survey", "--synthetic", "--specimen", "7", "--B", "210", "--mu", "1.0", "--out", Q,
          "--config-json", cfgj])
    oq = _run([sc, "clean", "--synthetic", "--specimen", "7", "--examples", Q / "examples_7.json", "--out", Q,
               "--rounds", "1", "--config-json", cfgj, "--kernel-width", "calibrated", "--sigma-json", js])
    return dict(base=base, P=P, Q=Q, cfgj=cfgj, cal=js, log_cal=oq.stdout + oq.stderr)


def _rec(d):
    return json.load(open(d / "cleaning_7.json"))


def test_clean_records_the_switch(prep):
    """B22.4: the preparation's record carries kernel_width (mode, values, SEs, n, file, base renderer, table) and
    its configuration holds the re-anchored table [by hand, 1e-12]; the fixed preparation records "fixed"; the log
    names the values."""
    rq, rp = _rec(prep["Q"]), _rec(prep["P"])
    kw = rq["kernel_width"]
    assert kw["mode"] == "calibrated" and kw["sigma0_um"] == 0.10 and kw["growth_scale"] == 1.25
    assert kw["n_sigma0"] == 4 and kw["n_growth_scale"] == 4 and kw["se_sigma0_um"] == 0.003
    base = C.default_config().renderer
    assert kw["base_renderer"]["kernel_table_sigma_um"] == list(base.kernel_table_sigma_um)
    ref, s0, slope = R.with_kernel_table(base.kernel_table_delta_um, base.kernel_table_sigma_um,
                                         base.kernel_continuation_slope, 0.10, 1.25)
    assert_allclose(rq["config"]["renderer"]["kernel_table_sigma_um"], ref, rtol=1e-12)
    assert_allclose(rq["config"]["renderer"]["kernel_continuation_slope"], slope, rtol=1e-12)
    assert rp["kernel_width"]["mode"] == "fixed"
    assert "calibrated" in prep["log_cal"] and "0.1000" in prep["log_cal"]


def _ns(**kw):
    d = dict(config_json="", kernel_width=None, sigma_json="", kernel="", sgt_skew="", cov_fine_factor=None)
    d.update(kw)
    return argparse.Namespace(**d)


def test_infer_switch_reanchors_the_base(prep, tmp_path):
    """B22.5: infer_nodes inherits the preparation's law (no --kernel-width: the preparation's table and record);
    its own "calibrated" re-anchors the preparation's recorded base table, also when the preparation was itself
    calibrated (no compounding: the table equals Block 4 amendment 2 of the base by hand [1e-12]); "fixed" given
    there returns to the base table; the override is logged."""
    import infer_nodes
    js2 = sigma_json(tmp_path / "b.json", s0=0.14, c=0.8)
    base = C.default_config().renderer
    ref2, s02, slope2 = R.with_kernel_table(base.kernel_table_delta_um, base.kernel_table_sigma_um,
                                            base.kernel_continuation_slope, 0.14, 0.8)
    for d in (prep["P"], prep["Q"]):
        rec = _rec(d)
        cfg0 = infer_nodes.run_config(rec, "", None)
        logs = []
        cfg, kw = infer_nodes.kernel_switches(_ns(), rec, cfg0, logs.append)
        assert cfg.renderer == cfg0.renderer and kw == rec["kernel_width"]
        cfg, kw = infer_nodes.kernel_switches(_ns(kernel_width="calibrated", sigma_json=js2), rec, cfg0, logs.append)
        assert_allclose(cfg.renderer.kernel_table_sigma_um, ref2, rtol=1e-12)
        assert_allclose(cfg.renderer.kernel_continuation_slope, slope2, rtol=1e-12)
        assert cfg.renderer.sigma_r0_um == 0.14 and kw["mode"] == "calibrated"
        assert any("overrides" in m for m in logs)
        cfg, kw = infer_nodes.kernel_switches(_ns(kernel_width="fixed"), rec, cfg0, logs.append)
        assert tuple(cfg.renderer.kernel_table_sigma_um) == tuple(base.kernel_table_sigma_um)
    rq = _rec(prep["Q"])
    cfgq = infer_nodes.run_config(rq, "", None)
    assert cfgq.renderer.sigma_r0_um == 0.10                       # inherited: the preparation's calibrated law


def test_infer_kernel_switches_replace_fields(prep):
    """B23.6: --kernel, --sgt-skew and --cov-fine-factor replace inference.kernel_family, sgt_skew and
    cov_fine_factor; the defaults are the configuration's."""
    import infer_nodes
    rec = _rec(prep["P"])
    cfg0 = infer_nodes.run_config(rec, "", None)
    cfg, _ = infer_nodes.kernel_switches(_ns(kernel="sgt", sgt_skew="fit", cov_fine_factor=0), rec, cfg0, lambda m: 0)
    assert (cfg.inference.kernel_family, cfg.inference.sgt_skew, cfg.inference.cov_fine_factor) == ("sgt", "fit", 0)
    cfg, _ = infer_nodes.kernel_switches(_ns(), rec, cfg0, lambda m: 0)
    assert (cfg.inference.kernel_family, cfg.inference.sgt_skew, cfg.inference.cov_fine_factor) == \
        (cfg0.inference.kernel_family, cfg0.inference.sgt_skew, cfg0.inference.cov_fine_factor)


# ------------------------------------------------ the factories (amendment 23) ---

def _node_and_pilot(prep):
    import types
    import infer_nodes
    rec = _rec(prep["P"])
    nid = int(rec["example_ids"][0])
    info = rec["nodes"][str(nid)]
    pil = info["pilot"]
    pilot = types.SimpleNamespace(cx_um=pil["cx_um"], cy_um=pil["cy_um"], theta_rad=pil["theta_rad"],
                                  phi_rad=pil["phi_rad"], z_sub_um=pil["z_sub_um"], d_hat_um=pil.get("d_hat_um"),
                                  mu_hat_per_um=pil.get("mu_hat_per_um"))
    with np.load(prep["P"] / ("cleaning_7_node%d.npz" % nid)) as z:
        node = infer_nodes.node_from_npz(nid, z, None)
    half = tuple(float(x) for x in info["U_axis_um"])
    return infer_nodes.run_config(rec, "", None), node, pilot, half


def test_forward_factories_by_hand(prep):
    """B23.1 / B23.2: forward_factory(..., "sgt")(d, mu, v0, z_ax, beta, xi, zeta) is line_fast.line_profiles of
    local_tube(pilot, trace, d, v0, z_ax - z_t, U-, U+) with kernel=SGTKernel(beta, xi, zeta) [exact]; "gaussian" the
    same without a kernel [exact]; forward_cov_factory the same model with line_fine_factor = cov_fine_factor [exact]
    and None at 0; an unknown family is refused."""
    cfg, node, pilot, half = _node_and_pilot(prep)
    th = (0.85, 0.9, 0.04, float(node.node_xyz[2]) + 0.05)
    shape = (1.6, 0.4, 1.2)
    t = node_inputs.local_tube(pilot, np.asarray(node.node_xyz, float), th[0], th[2], th[3] - float(node.node_xyz[2]),
                               half[0], half[1])
    args = (node.ks, node.origin_xy, node.y_hat, node.e_u, node.v, int(node.n_avg), float(node.step))
    f7 = node_inputs.forward_factory(pilot, half, node, cfg, line_fast.SectionCache(), "sgt")
    ref7 = line_fast.line_profiles(t, th[1], *args, cfg, kernel=line_sgt.SGTKernel(*shape))
    assert_array_equal(f7(*th, *shape), ref7)
    f4 = node_inputs.forward_factory(pilot, half, node, cfg, line_fast.SectionCache())
    assert_array_equal(f4(*th), line_fast.line_profiles(t, th[1], *args, cfg))
    cfg16 = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, cov_fine_factor=16))
    fc = node_inputs.forward_cov_factory(pilot, half, node, cfg16, line_fast.SectionCache(), "sgt")
    cfg_f = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, line_fine_factor=16))
    assert_array_equal(fc(*th, *shape), line_fast.line_profiles(t, th[1], *args, cfg_f,
                                                                 kernel=line_sgt.SGTKernel(*shape)))
    cfg0 = dataclasses.replace(cfg, inference=dataclasses.replace(cfg.inference, cov_fine_factor=0))
    assert node_inputs.forward_cov_factory(pilot, half, node, cfg0, line_fast.SectionCache(), "sgt") is None
    with pytest.raises(ValueError):
        node_inputs.forward_factory(pilot, half, node, cfg, line_fast.SectionCache(), "laplace")


def test_forward_factory_sgt_nan_on_extrusion_invalid(prep):
    """B23.1: a tube whose caps are within extrusion_cap_margin_um of the line gives NaN (as the Gaussian forward)."""
    cfg, node, pilot, half = _node_and_pilot(prep)
    th = (0.85, 0.9, 0.04, float(node.node_xyz[2]))
    short = (0.6, 0.6)
    f7 = node_inputs.forward_factory(pilot, short, node, cfg, line_fast.SectionCache(), "sgt")
    f4 = node_inputs.forward_factory(pilot, short, node, cfg, line_fast.SectionCache())
    a, b = f7(*th, 2.0, 0.0, 1.0), f4(*th)
    assert a.shape == (len(node.ks), len(node.v)) and np.all(np.isnan(a)) and np.all(np.isnan(b))


# ------------------------------------------------ infer_nodes end to end ---

INF = SCRIPTS / "infer_nodes.py"


@pytest.fixture(scope="module")
def runs(prep):
    """infer_nodes.py real --synthetic on the fixed preparation: gaussian; sgt (twice, for determinism); sgt with the
    skew fitted, the fit's own error bars and one node; gaussian with the fit's own error bars."""
    P = prep["P"]
    out = {}
    for name, extra in (("g", ["--kernel", "gaussian"]), ("s", ["--kernel", "sgt"]), ("s2", ["--kernel", "sgt"]),
                        ("g0", ["--kernel", "gaussian", "--cov-fine-factor", "0"])):
        o = P / ("inf_" + name)
        r = _run([INF, "real", "--synthetic", "--specimen", "7", "--cleaning-dir", P, "--out", o] + extra)
        out[name] = dict(dir=o, log=r.stdout + r.stderr)
    nid = str(_rec(P)["example_ids"][0])
    o = P / "inf_sf"
    r = _run([INF, "real", "--synthetic", "--specimen", "7", "--cleaning-dir", P, "--out", o, "--kernel", "sgt",
              "--sgt-skew", "fit", "--cov-fine-factor", "0", "--nodes", nid])
    out["sf"] = dict(dir=o, log=r.stdout + r.stderr, nid=nid)
    return out


def _csv(d):
    with open(d / "inference_7.csv") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd)


def _json(d):
    return json.load(open(d / "inference_7.json"))


def _f(x):
    return float("nan") if x in ("", None) else float(x)


def test_csv_columns(runs):
    """B23.3: the columns after `flags` are kernel, beta, se_beta, xi, se_xi, zeta, se_zeta, truncated_mass,
    cov_source, gauss_d_um, gauss_se_d_um, gauss_status, gauss_chi2_mean; amendment 16's five columns stay just
    before `seconds`, the last."""
    cols, _rows = _csv(runs["s"]["dir"])
    i = cols.index("flags")
    assert cols[i + 1:i + 14] == ["kernel", "beta", "se_beta", "xi", "se_xi", "zeta", "se_zeta", "truncated_mass",
                                  "cov_source", "gauss_d_um", "gauss_se_d_um", "gauss_status", "gauss_chi2_mean"]
    assert cols[-6:] == ["weights", "lambda_um", "v_cut_lo_um", "v_cut_hi_um", "n_solves", "seconds"]


def test_gaussian_run(runs):
    """B23.4 / B23.6: under "gaussian" the estimate columns equal the Gaussian fit's (gauss_*) [exact]; kernel
    "gaussian"; the error bars from the model on the fine lattice (cov_source "fine" at the default cov_fine_factor)
    and from the fit's own Jacobian with --cov-fine-factor 0 ("fit"), the estimate the same either way [exact]."""
    _c, rows = _csv(runs["g"]["dir"])
    _c, rows0 = _csv(runs["g0"]["dir"])
    assert rows and len(rows) == len(rows0)
    for r, r0 in zip(rows, rows0):
        assert r["kernel"] == "gaussian" and r["cov_source"] == "fine" and r0["cov_source"] == "fit"
        assert r["d_hat_um"] == r["gauss_d_um"] and r["se_d_um"] == r["gauss_se_d_um"]
        assert r["status"] == r["gauss_status"] and r["beta"] == "" and r["truncated_mass"] == ""
        assert r["d_hat_um"] == r0["d_hat_um"] and r["mu_hat_per_um"] == r0["mu_hat_per_um"]
        assert math.isfinite(_f(r["se_d_um"])) and _f(r["se_d_um"]) > 0
    js = _json(runs["g"]["dir"])
    for nid, rec in js["nodes"].items():
        assert rec["kernel"] == "gaussian" and rec["shape"] is None and rec["param_names"] == ["d", "mu", "v0", "z"]


def test_sgt_run(runs):
    """B23.4, B23.5, B23.8: under "sgt" the estimate columns are the SGT fit's (the JSON's theta, se, shape,
    se_shape, status), the gauss_* columns equal the Gaussian run's estimate [exact], the JSON node record carries
    kernel, cov_source, param_names (six, zeta held), cov_search (6 x 6), shape, se_shape and the Gaussian record
    `gauss` equal to the Gaussian run's node record; truncated_mass is the dropped fraction at the fitted shape
    (independent Abel quadrature, R/a = 12 W(kappa)/FWHM_PER_SD) [1e-8]."""
    _c, rs = _csv(runs["s"]["dir"])
    _c, rg = _csv(runs["g"]["dir"])
    js, jg = _json(runs["s"]["dir"]), _json(runs["g"]["dir"])
    assert [r["node_id"] for r in rs] == [r["node_id"] for r in rg]
    for r, g in zip(rs, rg):
        nid = r["node_id"]
        rec, recg = js["nodes"][nid], jg["nodes"][nid]
        assert r["kernel"] == "sgt"
        assert r["gauss_d_um"] == g["d_hat_um"] and r["gauss_se_d_um"] == g["se_d_um"]
        assert r["gauss_status"] == g["status"]
        assert_allclose(rec["gauss"]["theta"], recg["theta"], rtol=0, atol=0)
        assert_allclose(rec["gauss"]["se"], recg["se"], rtol=0, atol=0)
        assert rec["kernel"] == "sgt" and rec["param_names"] == ["d", "mu", "v0", "z", "log10_beta", "xi"]
        assert np.asarray(rec["cov_search"], dtype=float).shape == (6, 6)
        assert_allclose(_f(r["d_hat_um"]), rec["theta"][0], rtol=1e-12)
        assert_allclose(_f(r["se_d_um"]), rec["se"][0], rtol=1e-12)
        sh = [_f(r["beta"]), _f(r["xi"]), _f(r["zeta"])]
        assert_allclose(sh, rec["shape"], rtol=1e-12)
        assert sh[2] == 1.0 and math.isnan(_f(r["se_zeta"]))
        assert r["status"] == rec["status"]
        ra = 12.0 * R.unit_fwhm(*sh) / R.FWHM_PER_SD
        assert abs(_f(r["truncated_mass"]) - R.tail_mass_2d(sh[0], sh[1], ra)) <= 1e-8
        print("node %s: Gaussian d %s, SGT d %s (beta %.3f, xi %.3g), %s" % (nid, g["d_hat_um"], r["d_hat_um"],
                                                                           sh[0], sh[1], r["status"]))


def test_sgt_skew_fit_one_node(runs):
    """B23.6: --nodes runs the listed node alone; --sgt-skew fit adds ln_zeta to the search (seven names, zeta
    reported with its SE); --cov-fine-factor 0 gives the fit's own error bars ("fit")."""
    _c, rows = _csv(runs["sf"]["dir"])
    js = _json(runs["sf"]["dir"])
    assert [r["node_id"] for r in rows] == [runs["sf"]["nid"]] and list(js["nodes"]) == [runs["sf"]["nid"]]
    rec = js["nodes"][runs["sf"]["nid"]]
    assert rec["param_names"] == ["d", "mu", "v0", "z", "log10_beta", "xi", "ln_zeta"]
    assert rows[0]["cov_source"] == "fit" and rec["cov_source"] == "fit"
    assert np.asarray(rec["cov_search"], dtype=float).shape == (7, 7)


def test_unknown_node_refused(prep):
    """B23.6: --nodes with an id that is not among the preparation's example nodes is refused (non-zero status)."""
    r = _run([INF, "real", "--synthetic", "--specimen", "7", "--cleaning-dir", prep["P"], "--out",
              prep["base"] / "inf_bad", "--nodes", "999999"], check=False)
    assert r.returncode != 0 and "999999" in (r.stdout + r.stderr)


def test_own_product_keeps_the_gaussian_estimate(runs):
    """B23.7 (D-222.4): the same-node clearance product of an SGT run is computed at the Gaussian estimate: its arrays
    and its JSON record equal the Gaussian run's [exact]."""
    gd, sd = runs["g"]["dir"], runs["s"]["dir"]
    files = sorted(p.name for p in gd.glob("own_7_node*.npz"))
    assert files and files == sorted(p.name for p in sd.glob("own_7_node*.npz"))
    for name in files:
        with np.load(gd / name) as a, np.load(sd / name) as b:
            assert sorted(a.files) == sorted(b.files)
            for k in a.files:
                assert_array_equal(a[k], b[k], err_msg="%s %s" % (name, k))
    jg, js = _json(gd), _json(sd)
    for nid in jg["nodes"]:
        assert jg["nodes"][nid]["own"] == js["nodes"][nid]["own"]


def test_sgt_runs_deterministic(runs):
    """Step 6: two SGT runs give the same CSV (but the seconds column) and the same JSON node records (but the
    timing)."""
    c1, r1 = _csv(runs["s"]["dir"])
    c2, r2 = _csv(runs["s2"]["dir"])
    assert c1 == c2
    for a, b in zip(r1, r2):
        a.pop("seconds")
        b.pop("seconds")
        assert a == b
    j1, j2 = _json(runs["s"]["dir"]), _json(runs["s2"]["dir"])
    assert j1["nodes"] == j2["nodes"]


def test_outputs_exist_and_log_names_the_kernel(runs):
    """B23.4: the per-node figures and the two summary figures exist; the node's log line names the kernel's shape;
    the JSON records the width switch."""
    d = runs["s"]["dir"]
    js = _json(d)
    for nid in js["nodes"]:
        assert (d / ("fit_node%s.png" % nid)).stat().st_size > 0
    assert (d / "diameters.png").stat().st_size > 0 and (d / "chi2.png").stat().st_size > 0
    assert "SGT beta" in runs["s"]["log"] and js["kernel_width"]["mode"] == "fixed"


# ------------------------------------------------------------ findings' reproducers ---

# F1, a strict xfail in the tester's run of 2026-11-23 13:55, settled by D-233 (the recorded base renderer is read back
# by config.config_from_dict's rule): a plain test since (the orchestrator, 2026-11-23)
def test_switch_with_jpeg_tables_in_the_preparation(tmp_path):
    """Block 36 amendment 22 ("its own --kernel-width re-anchors the preparation's recorded base table") on a
    preparation whose renderer carries Allen's JPEG tables (Block 4: renderer.jpeg_qtables, the tables themselves):
    the record as clean_cell.py writes it (JSON) must give back the same renderer under "fixed" and Block 4
    amendment 2 of it under "calibrated"."""
    import kernel_width
    cfg = C.with_overrides(C.default_config(), {"renderer.jpeg_qtables": [list(range(1, 65))]})
    _, rec = kernel_width.apply(cfg, "fixed", "")
    base = json.loads(json.dumps(rec))["base_renderer"]          # as infer_nodes reads it from cleaning_<S>.json
    out, _ = kernel_width.apply(cfg, "fixed", "", base=base)
    assert out.renderer == cfg.renderer
    js = sigma_json(tmp_path / "s.json")
    out, _ = kernel_width.apply(cfg, "calibrated", js, base=base)
    assert out.renderer.jpeg_qtables == cfg.renderer.jpeg_qtables


# F3 (named F2 in the xfail's reason at the time), a strict xfail in the tester's run of 2026-11-23 13:55, settled by
# D-233 (a failed SGT stage keeps the Gaussian record and the clearance): a plain test since (the orchestrator, 2026-11-23)
def test_failed_sgt_stage_keeps_the_gaussian_record(prep, monkeypatch, tmp_path):
    """Block 36 amendment 23 (iii): the node's estimate is the configured kernel's fit, the Gaussian beside it; the
    same-node clearance keeps the Gaussian estimate. An SGT stage whose model is not finite (stand-in: the "sgt"
    forward returns NaN, as Block 30 then reports a failed fit with NaN shape) must leave the node's Gaussian record
    (`gauss`) and its clearance product (status "ok") in the JSON, with the SGT fit's status "failed"."""
    import infer_nodes
    real = node_inputs.forward_factory

    def patched(pilot, half, node, cfg, cache, kernel_family="gaussian"):
        f = real(pilot, half, node, cfg, cache, kernel_family)
        if kernel_family != "sgt":
            return f
        return lambda *a: np.full((len(node.ks), len(node.v)), np.nan)
    monkeypatch.setattr(node_inputs, "forward_factory", patched)
    nid = str(_rec(prep["P"])["example_ids"][0])
    out = tmp_path / "inf_fail"
    infer_nodes.main(["real", "--synthetic", "--specimen", "7", "--cleaning-dir", str(prep["P"]), "--out", str(out),
                      "--kernel", "sgt", "--cov-fine-factor", "0", "--nodes", nid])
    rec = json.load(open(out / "inference_7.json"))["nodes"][nid]
    assert "gauss" in rec and rec.get("status") == "failed"
    assert rec["own"]["status"] == "ok"


def test_calibrate_sigma_end_to_end_growth_summary_and_failed_ids(prep, tmp_path):
    """B22.1 through the script, and the D-223 settlement's F3 (Block 36 amendment 21): calibrate_sigma.py on a copy of
    the synthetic preparation with node 10's file corrupted ends 0 (one node fitted); sigma_<S>.json holds `summary` and
    `summary_growth_scale` (keys n, median, mad, ivw_mean, ivw_se, chi2_cons, dof) equal to Block 42's statistics
    recomputed by hand from the file's own fits [1e-12]; the growth scale's summary is printed; the last line names
    the failed node's id."""
    import shutil
    d = tmp_path / "prep_copy"
    shutil.copytree(prep["P"], d, ignore=shutil.ignore_patterns("inf_*"))
    (d / "cleaning_7_node10.npz").write_bytes(b"not an npz")
    r = _run([SCRIPTS / "calibrate_sigma.py", "--cleaning-dir", d, "--specimen", "7", "--nodes", "8,10",
              "--workers", "1"])
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    assert "growth scale c over" in r.stdout
    assert "1 node(s) failed: 10" in lines[-1], lines[-1]
    js = json.load(open(d / "sigma" / "sigma_7.json"))
    assert js["complete"] is True
    g = js["summary_growth_scale"]
    assert set(g) == {"n", "median", "mad", "ivw_mean", "ivw_se", "chi2_cons", "dof"}
    fits = js["fits"]
    keep = [f for f in fits if f.get("ok") and f.get("growth_scale") is not None and f.get("se_growth_scale")
            and f["se_growth_scale"] > 0]
    ref = R.summary_ref([f["growth_scale"] for f in keep], [f["se_growth_scale"] for f in keep], [True] * len(keep))
    assert int(g["n"]) == ref["n"]
    for k in ("median", "mad", "ivw_mean", "ivw_se"):
        if ref["n"]:
            assert_allclose(g[k], ref[k], rtol=1e-12, err_msg=k)
    print("fits:", [(f["node_id"], f["status"], f.get("growth_scale")) for f in fits], "summary_growth_scale:", g)
