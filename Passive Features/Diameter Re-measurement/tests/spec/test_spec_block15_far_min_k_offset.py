"""Spec-derived tests of Block 15's amendment (CalibrationConfig.far_min_k_offset = 4, D-273.2; tester,
2026-11-27 22:11). Oracles from SPEC.md Block 15 (15.3, the plane selection) and Block 10.C (the far candidates'
fit planes): K_j = {k : |k - k*_j| >= far_min_k_offset, status ok, predicted dip >= far_min_snr x RMS_j}; with every
plane ok and a large SNR, the far fit keeps the 2 x (far_offsets_planes - far_min_k_offset + 1) = 26 planes of
4 <= |k - k*| <= 16, which are exactly Step2Config.far_planes' band (the far stage, common definitions). Scans
are built by hand (PlaneScan in the far format), so nothing here computes the expectation with the code.

Pure ASCII.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import step2_helpers as H
from allen_diameter import config as C
from allen_diameter.analysis import far_field as FF
from allen_diameter.analysis import step2_twins as T


def _far_scan(node_id, k_star=20, P=16, bad=()):
    sc = H.plane_scan(node_id, "far", k_star=k_star, P=P)
    n = sc.ks.size
    status = list(sc.status)
    om = np.asarray(sc.omega_um2, float).copy()
    for q in bad:
        status[q] = "few_samples"
        om[q] = np.nan
    # amplitude 50 gl, RMS 0.1 gl: the predicted dip is hundreds of RMS in every plane (SNR rule passes)
    return dataclasses.replace(sc, omega_um2=om, status=tuple(status), amp_gl=np.full(n, 50.0), rms_gl=np.full(n, 0.1))


def test_far_min_k_offset_is_4_and_26_planes():
    cfg = C.default_config()
    cal = cfg.calibration
    assert cal.far_min_k_offset == 4 and cal.far_offsets_planes == 16
    sel = FF.select_far_planes([_far_scan(1), _far_scan(2, k_star=7)], cfg)
    assert sel.dropped == {}
    assert sel.n_planes_per_node.tolist() == [26, 26]
    for j, ks in ((0, 20), (1, 7)):
        k = np.sort(sel.obs_k[sel.obs_node == j])
        expect = np.array([x for x in range(ks - 16, ks + 17) if 4 <= abs(x - ks) <= 16])
        assert expect.size == 26
        np.testing.assert_array_equal(k, expect)


def test_offset_3_never_enters_and_a_bad_plane_is_left_out():
    cfg = C.default_config()
    sc = _far_scan(3, bad=(16 + 4, 16 - 5))            # |o| = 4 and 5 not ok
    sel = FF.select_far_planes([sc], cfg)
    off = np.abs(sel.obs_k - 20)
    assert off.min() >= 4 and 4 in off.tolist()        # o = -4 still kept, o = +4 removed
    assert sel.obs_k.size == 24 and int(sel.n_planes_per_node[0]) == 24


def test_far_stage_band_equals_block15_band():
    cfg = C.default_config()
    s2 = cfg.step2
    sc = _far_scan(4)
    tw = T.build_twins(1, [sc], {4: H.record(4, "far", d=0.8, k_star=20, z_sub=sc.z_sub_um)}, [4])[0]
    m = T.fit_planes(tw, s2, "far")
    sel = FF.select_far_planes([sc], cfg)
    np.testing.assert_array_equal(np.asarray(tw.ks)[m], np.sort(sel.obs_k))
