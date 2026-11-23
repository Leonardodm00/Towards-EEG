"""Spec-derived end-to-end tests of the soma fit's entry point, scripts/fit_soma.py (SPEC section 6, Cell 25; Blocks
44-48; D-216.3, D-218, D-220.2, D-222.5) -- tester 2026-11-23.

Oracles (SPEC section 6's row and Blocks 46-48):
  - `--synthetic` finishes with status 0 and writes soma_fit_<S>.json, _planes.png, _summary.png, _morphology.png and
    reconstruction_soma_<S>.swc; the JSON parses as strict JSON (no NaN tokens) and its numbers are finite where the fit
    converged;
  - the known ellipsoid is recovered: every fitted parameter within 5 SE of the truth; chi2/dof near the noise variance
    (noise SD 3 gl, then rounded to whole grey levels: 9 + 1/12 gl^2; its SD over ~1.3e6 samples is 0.011, band +-0.1);
  - derived quantities: S(a, b, c) against an independent area, V = 4/3 pi a b c, r_eq = sqrt(S / 4 pi), SE(r_eq) =
    SE(S) / (8 pi r_eq) and SE(V) by the delta method on the (a, b, c) block of the covariance (section 6);
  - the corrected SWC (Block 47 through Block 8): the soma node at the fitted centre in the SWC frame with r_eq, every
    stem's first node on the fitted surface (to the file's 4 decimals), every other data line byte for byte;
  - the crop: soma_fit.region_radius_um + crop_margin_um about the traced centre (Block 44's crop_box), the planes
    within plane_half_um of the start depth (Block 44's plane_range); --soma-k K sets the start depth to K dz;
  - --far-slope fixed: theta[8] = slope0 exactly, its SE null, eight names fitted;
  - a kernel continuation that is not "linear" and a negative --slope0 are refused (argparse error, status 2);
  - the same seed gives the same results (determinism);
  - boundary Block 44 -> entry point: the mask holds every pixel within (traced radius + mask_widen_um) of a traced
    segment that is not soma-rooted ("the trace's mask (+ mask_widen_um)"), segments outside the crop included;
  - boundary Block 47 -> Block 48 in the entry point: the morphology figure's `next` holds the stem's next nodes of
    the corrected morphology (after a walk, P_m and on).
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import soma_spec_helpers as H
from allen_diameter.analysis import cell, soma_view as SV
from allen_diameter.analysis import soma_morphology as SM
from allen_diameter.config import default_config
from allen_diameter.loading import swc_io

import fit_soma  # noqa: E402  (scripts/ on the path)

P, DZ = 0.1144, 0.28
REDUCED = {"soma_fit": {"plane_half_um": 4.0, "region_radius_um": 8.0, "plane_step": 3}}


def _run(out, *extra, config=None):
    argv = ["--specimen", "7", "--synthetic", "--out", out] + list(extra)
    if config is not None:
        cj = os.path.join(out + "_cfg.json")
        with open(cj, "w") as f:
            json.dump(config, f)
        argv += ["--config-json", cj]
    try:
        rc = fit_soma.main(argv)
    except SystemExit as exc:
        return int(exc.code), None
    with open(os.path.join(out, "soma_fit_7.json")) as f:
        text = f.read()
    rec = json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError("non-strict JSON: " + c)))
    return rc, rec


@pytest.fixture(scope="module")
def default_run():
    out = os.path.join(tempfile.mkdtemp(prefix="tester_fit_soma_"), "run")
    rc, rec = _run(out)
    return rc, rec, out


def _no_seconds(d):
    if isinstance(d, dict):
        return {k: _no_seconds(v) for k, v in d.items() if k != "seconds"}
    if isinstance(d, list):
        return [_no_seconds(v) for v in d]
    return d


def test_default_run_outputs_and_recovery(default_run):
    rc, rec, out = default_run
    assert rc == 0
    for name in ("soma_fit_7.json", "soma_fit_7_planes.png", "soma_fit_7_summary.png", "soma_fit_7_morphology.png",
                 "reconstruction_soma_7.swc"):
        assert os.path.getsize(os.path.join(out, name)) > 0, name
    fit = rec["fit"]
    assert fit["status"] == "converged" and fit["at_bound"] == []
    th, se = np.array(fit["theta"], float), np.array(fit["se"], float)
    truth = np.array(rec["synthetic_truth"], float)
    assert np.all(np.isfinite(th)) and np.all(np.isfinite(se)) and np.all(se > 0)
    assert np.all(np.abs(th - truth) <= 5 * se), (th - truth) / se
    chi2_dof = fit["chi2"] / fit["dof"]
    assert abs(chi2_dof - (9.0 + 1.0 / 12.0)) < 0.1, chi2_dof
    # the crop and the planes (Block 44's rules with the fit's settings)
    q = rec["settings"]
    left, top, w, h = SV.crop_box(rec["traced"]["centre_um"][:2], q["region_radius_um"] + q["crop_margin_um"], P)
    assert (rec["crop"]["left"], rec["crop"]["top"], rec["crop"]["width"], rec["crop"]["height"]) == (left, top, w, h)
    win = H.ref_window(rec["start"]["centre_um"][2], q["plane_half_um"], DZ, np.arange(-100, 101))
    assert (rec["crop"]["k_lo"], rec["crop"]["k_hi"]) == (win[0], win[-1])
    # derived quantities
    a, b, c = th[3:6]
    S = H.ref_area(a, b, c)
    der = rec["derived"]
    assert_allclose(der["area_um2"], S, rtol=1e-8)
    assert_allclose(der["volume_um3"], 4 / 3 * math.pi * a * b * c, rtol=1e-12)
    assert_allclose(der["r_eq_um"], math.sqrt(S / (4 * math.pi)), rtol=1e-8)
    assert_allclose(der["se_r_eq_um"], der["se_area_um2"] / (8 * math.pi * der["r_eq_um"]), rtol=1e-12)
    cov = np.array(fit["cov"], float)
    g = 4 / 3 * math.pi * np.array([b * c, a * c, a * b])
    assert_allclose(der["se_volume_um3"], math.sqrt(g @ cov[3:6, 3:6] @ g), rtol=1e-10)
    assert_allclose(se, np.sqrt(np.diag(cov)), rtol=1e-12)


def test_default_run_corrected_swc(default_run):
    rc, rec, out = default_run
    th = np.array(rec["fit"]["theta"], float)
    r_eq = rec["derived"]["r_eq_um"]
    tr = rec["transform"]
    new = swc_io.read_swc(os.path.join(out, "reconstruction_soma_7.swc"))
    # the input SWC: the synthetic cell's own (fixtures_cell), rebuilt the way the script reads it
    import clean_cell                                           # noqa: F401
    from fixtures_cell import synthetic_cell
    with tempfile.TemporaryDirectory() as tmp:
        old, _f, _p, path = synthetic_cell(tmp, default_config(), d_true=clean_cell.SYNTH_D_TRUE_UM,
                                           mu=clean_cell.SYNTH_MU_PER_UM, n_nodes=clean_cell.SYNTH_N_NODES,
                                           allen_radius=clean_cell.SYNTH_ALLEN_RADIUS_UM)
        old_text = open(path).read()
    new_text = open(os.path.join(out, "reconstruction_soma_7.swc")).read()
    soma = int(np.flatnonzero(new.types == 1)[0])
    centre_swc = np.array([(th[0] / P - tr["shift_full_px"][0]) * P, (th[1] / P - tr["shift_full_px"][1]) * P,
                           th[2] + tr["z0_um"]])
    assert tr["flip_y_full_h"] is None
    assert_allclose(new.xyz[soma], centre_swc, rtol=0, atol=0.5e-4 + 1e-12)     # 4 decimals written
    assert_allclose(new.radius[soma], r_eq, rtol=0, atol=0.5e-4 + 1e-12)
    stems = rec["stems"]
    assert len(stems) >= 1
    moved_ids = {s["id"] for s in stems if s["flag"] == "moved"}
    for s in stems:
        k = int(np.flatnonzero(new.ids == s["id"])[0])
        assert_allclose(new.xyz[k], s["new_um"], atol=0.5e-4 + 1e-12)
        if s["id"] in moved_ids:
            q = H.ref_q(H.ref_to_image(new.xyz[k][None, :], P, tr["shift_full_px"], None, tr["z0_um"]), th)[0]
            assert abs(q) < 1e-4, q                              # on the surface to the written precision
    # every other data line byte for byte (Allen's 4 decimals; the synthetic file writes 4 decimals)
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    assert len(old_lines) == len(new_lines)
    for lo, ln in zip(old_lines, new_lines):
        tok = lo.split()
        if not tok or tok[0].startswith("#"):
            assert lo == ln
        elif int(tok[0]) not in moved_ids and int(tok[1]) != 1:
            assert lo == ln, (lo, ln)


def test_reduced_runs_soma_k_fixed_slope_and_determinism(tmp_path):
    rc1, r1 = _run(str(tmp_path / "a"), "--soma-k", "2", "--far-slope", "fixed", "--slope0", "0.79", config=REDUCED)
    rc2, r2 = _run(str(tmp_path / "b"), "--soma-k", "2", "--far-slope", "fixed", "--slope0", "0.79", config=REDUCED)
    assert rc1 == 0 and rc2 == 0
    assert _no_seconds(r1) == _no_seconds(_no_seconds(r2) | {"swc_out": r1["swc_out"]})
    assert open(r1["swc_out"]).read() == open(r2["swc_out"]).read()
    fit = r1["fit"]
    assert fit["theta"][8] == 0.79 and fit["se"][8] is None
    assert fit["fitted"] == list(fit["params"][:8])
    assert r1["start"]["centre_um"][2] == 2 * DZ and r1["start"]["source"] == "plane 2 (by eye)"
    win = H.ref_window(2 * DZ, 4.0, DZ, np.arange(-100, 101))
    assert (r1["crop"]["k_lo"], r1["crop"]["k_hi"]) == (win[0], win[-1])


def test_transform_round_trip_with_flip(tmp_path):
    tr = ["--shift-x", "250.5", "--shift-y", "-120.25", "--flip-h", "4000", "--z0", "1.5"]
    rc, rec = _run(str(tmp_path / "f"), *tr, config=REDUCED)
    assert rc == 0
    th = np.array(rec["fit"]["theta"], float)
    new = swc_io.read_swc(rec["swc_out"])
    soma = int(np.flatnonzero(new.types == 1)[0])
    # the fitted centre (image frame) mapped back to the SWC frame by the inverse of the spec's map
    x = (th[0] / P - 250.5) * P
    y = (4000.0 - (th[1] / P + 120.25)) * P
    z = th[2] + 1.5
    assert_allclose(new.xyz[soma], [x, y, z], atol=0.5e-4 + 1e-9)
    assert_allclose(H.ref_to_image(new.xyz[soma][None, :], P, (250.5, -120.25), 4000.0, 1.5)[0], th[:3],
                    atol=1e-4)


@pytest.mark.parametrize("extra, config", [
    ((), {"renderer": {"kernel_continuation": "frozen"}}),
    ((), {"renderer": {"kernel_continuation": "proportional"}}),
    (("--slope0", "-1"), None),
    (("--slope0", "nan"), None),
])
def test_refusals(tmp_path, extra, config):
    rc, rec = _run(str(tmp_path / "x"), *extra, config=config)
    assert rc == 2 and rec is None


# 2026-11-23 (00:27) report F1, fixed by D-223.1: a strict xfail until then, a plain test since D-224.3.
def test_mask_holds_widened_segments_outside_the_crop():
    """The trace's mask (+ mask_widen_um): a traced dendrite running 0.9 um outside the crop's edge, radius 0.5 um,
    reaches 1.6 um into the crop once widened by 2 um -- into the fit's 13 um disk (crop half 14 um)."""
    q = default_config().soma_fit
    half = q.region_radius_um + q.crop_margin_um
    left, top, W, Hh = SV.crop_box((0.0, 0.0), half, P)
    xs, r, widen = (left + W) * P + 0.9, 0.5, q.mask_widen_um
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "c.swc")
        with open(fn, "w") as f:
            f.write("# t\n1 1 0 0 0 4 -1\n2 3 %.4f -8.0 0.0 %.4f 1\n3 3 %.4f 8.0 0.0 %.4f 2\n" % (xs, r, xs, r))
        swc = swc_io.read_swc(fn)
    box = (left * P, (left + W) * P, top * P, (top + Hh) * P)
    m, _n = fit_soma._mask(swc.xyz, swc, swc.parent_index(), np.asarray(swc.types), box, (Hh, W), left, top, P, widen)
    X = (left + np.arange(W)) * P
    ref = np.broadcast_to(np.abs(X[None, :] - xs) <= r + widen, (Hh, W)) & \
        np.broadcast_to((((top + np.arange(Hh)) * P) >= -8.0)[:, None] & (((top + np.arange(Hh)) * P) <= 8.0)[:, None],
                        (Hh, W))
    region = SV.disk_mask((Hh, W), left, top, P, (0.0, 0.0), q.region_radius_um)
    assert (ref & region).sum() > 0                      # the widened dendrite reaches the fit's region
    assert_array_equal(m & region, ref & region)


# 2026-11-23 (00:27) report F2, fixed by D-223.2: a strict xfail until then, a plain test since D-224.3.
def test_morphology_view_next_nodes_after_a_walk():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "c.swc")
        with open(fn, "w") as f:
            f.write("# t\n1 1 0 0 0 3 -1\n2 3 1 0 0 0.5 1\n3 3 2.5 0 0 0.5 2\n4 3 4 0 0 0.5 3\n5 3 6 0 0 0.5 4\n"
                    "6 3 8 0 0 0.5 5\n")
        swc = swc_io.read_swc(fn)
    th = np.array([0, 0, 0, 5.0, 5.0, 5.0, 0.0, 0.1, 0.79])
    tr = dict(shift_full_px=(0.0, 0.0), flip_y_full_h=None, z0_um=0.0)
    edit = SM.correct_soma(swc, th[:8], 4.0, 1.0, **tr)
    assert edit.stems[0]["removed_ids"] == (3, 4)                   # P2, P3 removed; P4 (id 5) re-parented to P1
    xyz = cell.to_image_um(swc.xyz, 1.0, **tr)
    top, _side = fit_soma._morphology_views(swc, edit, xyz, th, np.zeros(3), 3.0, 4.0, 1.0, tr)
    nxt = np.asarray(top["stems"][0]["next"])                        # (2, m): the corrected stem goes on at P4, P5
    assert nxt.shape == (2, 2), nxt.shape
    assert_allclose(nxt, [[6.0, 8.0], [0.0, 0.0]])
