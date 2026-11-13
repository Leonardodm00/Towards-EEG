"""Spec-derived tests of Block 18 (S3b: d_max, the node jitter, the camera
suggestions) -- written by the integration tester (run D5, 2026-11-13) from
specs/SPEC.md, not from the code. Expected values: numpy (percentile,
median), scipy.stats.median_abs_deviation, collections.Counter, or by hand.

Run (from the workstream directory):
    MPLBACKEND=Agg python -m pytest tests/spec/test_spec_block18_s3b.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import spec_helpers_D5 as H

from allen_diameter.analysis import s3_inputs as S3I

NAN = float("nan")


def _rad():
    return [dict(specimen_id=30, node_ids=[1, 2, 3, 4], D_allen=[0.5, 2.4, 1.0, NAN]),
            dict(specimen_id=10, node_ids=[7, 8], D_allen=[3.1, 0.2]),
            dict(specimen_id=20, node_ids=[5, 6, 9], D_allen=[1.0, 2.9, 3.0])]


def test_d_max_largest_with_cell_node_and_comparisons():
    # 18.4: max over cells of max D_A, with its cell and node; pooled p99 and p99.5 (numpy.percentile); per cell max
    out = S3I.d_max(_rad())
    assert out["value_um"] == 3.1 and out["specimen_id"] == 10 and out["node_id"] == 7
    allD = np.array([0.5, 2.4, 1.0, 3.1, 0.2, 1.0, 2.9, 3.0])
    assert_allclose((out["p99_um"], out["p99_5_um"]), np.percentile(allD, [99.0, 99.5]), rtol=1e-12)
    per = {r["specimen_id"]: (r["max_um"], r["node_id"]) for r in out["per_cell"]}
    assert per == {10: (3.1, 7), 20: (3.0, 9), 30: (2.4, 2)}


def test_d_max_order_invariance_under_a_tie():
    # 18 oracle (c): "the cells' order changes nothing" -- two cells sharing the largest 2r
    cells = [dict(specimen_id=41, node_ids=[1, 2], D_allen=[1.0, 4.0]),
             dict(specimen_id=40, node_ids=[3, 4], D_allen=[4.0, 0.5])]
    a, b = S3I.d_max(cells), S3I.d_max(list(reversed(cells)))
    assert a["value_um"] == b["value_um"] == 4.0
    assert (a["specimen_id"], a["node_id"]) == (b["specimen_id"], b["node_id"]), (a["specimen_id"], b["specimen_id"])


def test_robust_sd_reference():
    rng = np.random.default_rng(1)
    for x in (rng.normal(0.2, 0.7, 101), rng.standard_t(2, 40), np.r_[rng.normal(size=30), [NAN, np.inf]]):
        got = S3I.robust_sd(x)
        assert_allclose(got, H.ref_robust_sd(x), rtol=1e-12)
        # scipy's scale="normal" is 1/Phi^-1(3/4) = 1.4826022..., the spec's 1.4826: relative gap 1.5e-6
        ref = stats.median_abs_deviation(x[np.isfinite(x)], scale="normal")
        assert_allclose(got, ref, rtol=2e-6)
    assert math.isnan(S3I.robust_sd([])) and math.isnan(S3I.robust_sd([NAN]))


def test_jitter_pooled_per_cell_and_z0_flag():
    # 18.4: over ON nodes, pooled jitter_xy = 1.4826 median|s - median s|, jitter_z the same of dz; per cell the
    # same and the median; flag z0 iff |median dz| > 1.0 (1.0 itself not flagged); no ON node: NaN, pooled from others
    cells = [dict(specimen_id=2, s_star=[0.1, -0.2, 0.05, 0.3, -0.1], dz_star=[1.5, 1.2, 1.9, 1.5, 1.4]),
             dict(specimen_id=1, s_star=[0.0, 0.4, -0.3], dz_star=[1.0, 0.8, 1.2]),
             dict(specimen_id=3, s_star=[], dz_star=[])]
    out = S3I.jitter(cells, 1.0)
    s_all = [0.1, -0.2, 0.05, 0.3, -0.1, 0.0, 0.4, -0.3]
    z_all = [1.5, 1.2, 1.9, 1.5, 1.4, 1.0, 0.8, 1.2]
    assert_allclose((out["jitter_xy_um"], out["jitter_z_um"]), (H.ref_robust_sd(s_all), H.ref_robust_sd(z_all)),
                    rtol=1e-12)
    per = {r["specimen_id"]: r for r in out["per_cell"]}
    assert per[2]["z0_flag"] and not per[1]["z0_flag"]                  # median 1.5 > 1; median 1.0 not > 1
    assert per[2]["median_dz_um"] == 1.5 and per[1]["median_dz_um"] == 1.0
    assert_allclose(per[2]["sd_xy_um"], H.ref_robust_sd(cells[0]["s_star"]), rtol=1e-12)
    assert per[3]["sd_xy_um"] is None or math.isnan(per[3]["sd_xy_um"])
    assert out["z0_flagged"] == [2]


def test_jitter_order_invariance():
    cells = [dict(specimen_id=k, s_star=list(np.random.default_rng(k).normal(0, 0.3, 7)),
                  dz_star=list(np.random.default_rng(k + 9).normal(0.5, 0.5, 7))) for k in range(4)]
    a, b = S3I.jitter(cells, 1.0), S3I.jitter(list(reversed(cells)), 1.0)
    assert a == b


def test_camera_most_common_set_and_medians():
    # 18.4: the most common JPEG table set (by the tables' identity) and its cells, the medians of B and noise
    t1, t2 = [[1] * 64], [[2] * 64, [3] * 64]
    cells = [dict(specimen_id=5, background_B_gl=210.0, noise_sd_gl=2.0, jpeg_qtables=t1),
             dict(specimen_id=6, background_B_gl=200.0, noise_sd_gl=None, jpeg_qtables=t2),
             dict(specimen_id=7, background_B_gl=None, noise_sd_gl=3.0, jpeg_qtables=[list(t) for t in t1]),
             dict(specimen_id=8, background_B_gl=190.0, noise_sd_gl=4.0, jpeg_qtables=None)]
    out = S3I.camera(cells)
    assert out["jpeg_qtables"] == t1 and sorted(out["jpeg_cells"]) == [5, 7] and out["n_table_sets"] == 2
    assert out["background_B_gl"] == float(np.median([210.0, 200.0, 190.0]))
    assert out["noise_sd_gl"] == float(np.median([2.0, 3.0, 4.0]))
    rev = S3I.camera(list(reversed(cells)))
    assert rev["jpeg_qtables"] == out["jpeg_qtables"] and rev["background_B_gl"] == out["background_B_gl"]


def test_camera_tie_between_table_sets_independent_of_order():
    t1, t2 = [[1] * 64], [[2] * 64]
    cells = [dict(specimen_id=9, background_B_gl=1.0, noise_sd_gl=1.0, jpeg_qtables=t1),
             dict(specimen_id=4, background_B_gl=1.0, noise_sd_gl=1.0, jpeg_qtables=t2)]
    assert S3I.camera(cells)["jpeg_qtables"] == S3I.camera(list(reversed(cells)))["jpeg_qtables"]
