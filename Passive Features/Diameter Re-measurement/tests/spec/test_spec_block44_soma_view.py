"""Spec-derived tests, Block 44 (the soma's view) amendments 1-3 (D-211, D-213, D-220) -- tester 2026-11-23.

Oracles (SPEC Block 44):
  amendment 3 (i)  show_soma.py refuses (argparse error, exit status 2) --soma-k-range LO HI unless LO <= HI and both
                   are recorded planes of the window, and --montage-k LO HI STEP unless LO and HI lie in the window; the
                   montage skips the planes between them that are not recorded;
  amendment 3 (ii) crop_box refuses a non-positive half; plane_range takes half >= 0 (0: the plane at the depth only)
                   and refuses half < 0;
  amendment 1 (iii) montage_planes on a window k = 75..159, k_ref 117, step 2: 43 planes, 117 among them; a range
                   105..145 step 1: 41;
  public API: crop_box's corner rounded down and side ceil(2 half / res0) + 1; plane_range the k with |k dz - z| <= half;
  amendment 2 (iii) segment_mask of a straight segment of constant radius r and length L covers 2 r L + pi r^2 [3 %].
The synthetic cell (--synthetic) records planes -7..7; the window of its soma (r 4 um, z_margin 6 um) is -35..35.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

import soma_spec_helpers as H
from allen_diameter.analysis import soma_view as SV

import show_soma  # noqa: E402  (scripts/ on the path)

RECORDED = list(range(-7, 8))


def _run(tmp_path, *extra):
    out = str(tmp_path / "out")
    try:
        rc = show_soma.main(["--specimen", "7", "--synthetic", "--out", out] + list(extra))
    except SystemExit as exc:
        return int(exc.code), None
    with open(os.path.join(out, "soma_7.json")) as f:
        return rc, json.load(f)


@pytest.mark.parametrize("argv", [["--soma-k-range", "-7", "8"],      # HI in the window but not recorded
                                  ["--soma-k-range", "-20", "0"],     # LO not recorded
                                  ["--soma-k-range", "3", "2"],       # LO > HI
                                  ["--montage-k", "-36", "0", "1"],   # LO outside the window -35..35
                                  ["--montage-k", "0", "36", "1"],    # HI outside
                                  ["--montage-k", "0", "5", "0"]])    # step < 1
def test_amendment3_refusals(tmp_path, argv):
    rc, rec = _run(tmp_path, *argv)
    assert rc == 2 and rec is None


def test_amendment3_accepted_ranges(tmp_path):
    rc, rec = _run(tmp_path, "--soma-k-range", "-7", "7", "--montage-k", "-35", "35", "3")
    assert rc == 0
    assert rec["k_range_by_eye"] == [-7, 7]
    assert rec["ks"] == list(range(-35, 36))
    k_ref = rec["k_shown"]
    exp = [k for k in RECORDED if (k - k_ref) % 3 == 0]            # unrecorded planes skipped
    assert rec["montage_ks"] == exp


def test_crop_box_and_plane_range():
    p = 0.1144
    for half in (0.0, -1.0):
        with pytest.raises(ValueError):
            SV.crop_box((3.0, 4.0), half, p)
    left, top, w, h = SV.crop_box((3.0, -4.0), 14.0, p)
    assert (left, top) == (math.floor((3.0 - 14.0) / p), math.floor((-4.0 - 14.0) / p))
    assert w == h == math.ceil(28.0 / p) + 1
    assert SV.plane_range(20 * 0.28, 0.0, 0.28) == (20, 20)       # half 0: the plane at the depth only
    with pytest.raises(ValueError):
        SV.plane_range(0.0, -0.01, 0.28)
    for z, half, dz in ((0.0, 10.0, 0.28), (0.1, 2.8, 0.28), (117 * 0.28, 4 + 6, 0.28), (-1.3, 0.56, 0.28)):
        ks = np.arange(-200, 600)
        win = H.ref_window(z, half, dz, ks)
        assert SV.plane_range(z, half, dz) == (win[0], win[-1])


def test_montage_planes_numbers():
    ks = np.arange(75, 160)
    v = np.ones(ks.size, bool)
    idx = SV.montage_planes(ks, v, 117, 2)
    assert len(idx) == 43 and 117 in ks[idx].tolist()
    idx = SV.montage_planes(ks, v, 117, 1, 105, 145)
    assert len(idx) == 41


def test_segment_mask_area():
    p, r, L = 0.1144, 0.6, 9.0
    left, top, W, H_ = -60, -60, 121, 121
    m = SV.segment_mask((H_, W), left, top, p, np.array([[-4.0, -1.0]]), np.array([[-4.0 + L * 0.8, -1.0 + L * 0.6]]),
                        np.array([r]), np.array([r]))
    area = m.sum() * p * p
    assert abs(area / (2 * r * L + math.pi * r * r) - 1) < 0.03
