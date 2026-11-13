"""Spec-derived tests of Block 16 (S3 across cells: 16.1 the pool, the
acquisition check and the selection; 16.2 the alignment and the runner; 16.3
the projection check; 16.4 the survey jobs) -- written by the integration
tester (run D5, 2026-11-13) from specs/SPEC.md, not from the code. Every
expected value comes from spec_helpers_D5 (numpy, zlib, brute force) or by hand.

Run (from the workstream directory):
    MPLBACKEND=Agg python -m pytest tests/spec/test_spec_block16_survey.py -q
"""
from __future__ import annotations

import dataclasses
import math
import os
import random
import subprocess
import warnings
import zlib

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import spec_helpers_D5 as H

import allen_image_align as aia
import allen_image_io as aio
import survey_cells as SC
from allen_diameter.analysis import cell as cellmod
from allen_diameter.analysis import cell_alignment as CA
from allen_diameter.analysis import cell_selection as CS
from allen_diameter.analysis import projection_check as PC
from allen_diameter.config import default_config, with_overrides
from allen_diameter.loading import cell_pool as CP
from allen_diameter.loading import survey_io

NAN = float("nan")
RHO0 = 0.1144


# =============================================================== 16.1 ======

def _raw(sid, species="Homo Sapiens", layer="3", dtype="spiny", recon="full", **kw):
    row = {"specimen__id": sid, "donor__species": species, "structure__layer": layer, "tag__dendrite_type": dtype,
           "nr__reconstruction_type": recon, "structure_parent__acronym": "MTG", "donor__id": 9,
           "specimen__hemisphere": "left", "csl__normalized_depth": 0.4}
    row.update(kw)
    return row


def test_pool_paging_pages_until_short_and_warns_after_one():
    # 16.1 (a): pages of page_rows until a page comes back short; a second page warns (RuntimeWarning)
    for n_rows, page, n_calls, warn in ((25, 10, 3, True), (20, 10, 3, True), (5, 10, 1, False), (0, 10, 1, False)):
        rows = [_raw(i) for i in range(n_rows)]
        calls = []

        def api(criteria, num_rows=200, start_row=0):
            calls.append((num_rows, start_row))
            return rows[start_row:start_row + num_rows]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            got = CP.fetch_specimen_details(api=api, page_rows=page, max_pages=50)
        assert len(got) == n_rows and len(calls) == n_calls, (n_rows, calls)
        assert [c[1] for c in calls] == [page * k for k in range(n_calls)]
        assert any(issubclass(x.category, RuntimeWarning) for x in w) == warn, (n_rows, [str(x.message) for x in w])


def test_pool_paging_never_more_than_max_pages():
    # 16.1 (a): "at most 50 pages" -- an API that always fills the page is not read past max_pages
    calls = []

    def api(criteria, num_rows=200, start_row=0):
        calls.append(start_row)
        return [_raw(start_row + i) for i in range(num_rows)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            CP.fetch_specimen_details(api=api, page_rows=4, max_pages=3)
        except RuntimeError:
            pass
    assert len(calls) <= 3, calls


def test_pool_filter_by_spec():
    # 16.1 (a): species == "Homo Sapiens" ignoring case; reconstruction type present (None, "", none/null/nan absent);
    # one record per specimen, the first row of a repeated id
    rows = [_raw(5), _raw(3, species="homo sapiens"), _raw(4, species="HOMO SAPIENS", recon="dendrite-only"),
            _raw(6, species="Mus musculus"), _raw(7, recon=None), _raw(8, recon=""), _raw(9, recon="NULL"),
            _raw(10, recon="nan"), _raw(11, recon="None"), _raw(5, layer="6a", dtype="aspiny")]
    recs = CP.pool_records(rows)
    assert sorted(r["specimen_id"] for r in recs) == [3, 4, 5]
    r5 = [r for r in recs if r["specimen_id"] == 5][0]
    assert r5["layer"] == "3" and r5["dendrite_type"] == "spiny"            # the first row of id 5
    assert isinstance(r5["specimen_id"], int) and isinstance(r5["normalized_depth"], float)


def test_pool_sdk_names_give_same_records():
    raw = [_raw(5), _raw(3, layer="5", dtype="aspiny")]
    sdk = [{"id": r["specimen__id"], "species": r["donor__species"], "structure_layer_name": r["structure__layer"],
            "dendrite_type": r["tag__dendrite_type"], "reconstruction_type": r["nr__reconstruction_type"],
            "structure_area_abbrev": "MTG", "donor_id": 9, "structure_hemisphere": "left", "normalized_depth": 0.4}
           for r in raw]
    assert CP.pool_records(raw) == CP.pool_records(sdk)


def test_pool_required_field_missing_raises_naming_keys():
    row = _raw(5)
    del row["tag__dendrite_type"]
    with pytest.raises(ValueError) as e:
        CP.pool_records([row])
    assert "donor__species" in str(e.value) and "specimen__id" in str(e.value)


def test_pool_absent_depth_is_nan():
    r = CP.normalize_row(_raw(5, csl__normalized_depth=None))
    assert math.isnan(r["normalized_depth"])


def test_neuron_class_exact_case_insensitive():
    cm = default_config().survey.class_map
    cases = {"spiny": "exc", "Spiny": "exc", "aspiny": "inh", "ASPINY": "inh", "sparsely spiny": "sparsely_spiny",
             "Sparsely Spiny": "sparsely_spiny", "unknown": None, None: None, "spin": None}
    for t, want in cases.items():
        assert CS.neuron_class(t, cm) == want, t


def _acq(sid=1, dz=0.28, px=0.1144, n=4, err=""):
    return dict(specimen_id=sid, dz_um=dz, scale_x=1.0, scale_y=1.0, pixel_um=px,
                pixel_values="" if isinstance(px, float) and math.isnan(px) else "%g" % px, n_planes=n,
                n_planes_failed=0, width_px=1000.0, height_px=2000.0, error=err)


def test_matches_acquisition_tolerance_and_reasons():
    # 16.1 (c): |dz_c - 0.28| <= 1e-3 0.28 and |p_c - 0.1144| <= 1e-3 0.1144; NaN, error, no plane never match
    cfg = default_config()
    inside, outside = 1 + 0.9e-3, 1 + 1.1e-3          # just inside / outside the relative tolerance 1e-3
    assert CS.matches_acquisition(_acq(dz=0.28 * inside, px=0.1144 / inside), cfg) == (True, "")
    assert CS.matches_acquisition(_acq(dz=0.28 * outside), cfg) == (False, "dz")
    assert CS.matches_acquisition(_acq(px=0.1144 / outside), cfg) == (False, "pixel")
    assert CS.matches_acquisition(_acq(dz=0.27, px=0.2288), cfg) == (False, "dz;pixel")
    assert CS.matches_acquisition(_acq(dz=NAN), cfg)[1] == "dz"
    assert CS.matches_acquisition(_acq(px=NAN), cfg)[1] == "pixel"
    assert CS.matches_acquisition(_acq(err="images: boom"), cfg) == (False, "error")
    assert CS.matches_acquisition(_acq(n=0), cfg) == (False, "no_planes")


def test_acquisition_summary_by_hand():
    cfg = default_config()
    recs = [_acq(1), _acq(2), _acq(3, dz=0.5), _acq(4, err="x"), _acq(5, px=NAN)]
    recs[4]["pixel_values"] = "0.1144;0.2288"
    s = CS.acquisition_summary(recs, cfg)
    assert {"n_cells", "n_failed", "dz_um_counts", "pixel_um_counts", "n_match", "mismatch", "configuration"} <= set(s)
    assert s["n_cells"] == 5 and s["n_failed"] == 1 and s["n_match"] == 2
    assert s["dz_um_counts"] == {"0.2800": 4, "0.5000": 1}
    assert s["pixel_um_counts"]["0.1144"] == 4 and s["pixel_um_counts"]["multiple: 0.1144;0.2288"] == 1
    assert s["mismatch"] == {"3": "dz", "4": "error", "5": "pixel"}


def _pool(groups, seed=0):
    """groups: {(dendrite_type, layer): n}; distinct ids, deliberately unsorted."""
    rng = random.Random(seed)
    pool, nid = [], 1000
    for (dt, layer), n in groups.items():
        for _ in range(n):
            nid += rng.randint(1, 50)
            pool.append(dict(specimen_id=nid, species="Homo Sapiens", layer=layer, dendrite_type=dt,
                             reconstruction_type="full", area="MTG", donor_id="1", hemisphere="left",
                             normalized_depth=0.5))
    rng.shuffle(pool)
    return pool


def _cfg_seed(seed, per_group=5):
    cfg = default_config()
    return dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, selection_seed=seed,
                                                               cells_per_group=per_group))


def test_select_equals_reference_at_two_seeds():
    # 16.1 (d) by an independent statement: sorted ids, default_rng([seed, crc32(key)]).permutation, first n
    groups = {("spiny", "3"): 9, ("aspiny", "3"): 1, ("spiny", "5"): 12, ("sparsely spiny", "4"): 5,
              ("aspiny", "6a"): 6}
    pool = _pool(groups)
    cls = {"spiny": "exc", "aspiny": "inh", "sparsely spiny": "sparsely_spiny"}
    for seed in (20261026, 17):
        sel = CS.select_cells(pool, _cfg_seed(seed))
        got = {(r["group"], r["specimen_id"], r["rank"]) for r in sel.selected}
        want = set()
        for (dt, layer), n in groups.items():
            key = "%s|%s" % (cls[dt], layer)
            ids = [p["specimen_id"] for p in pool if p["dendrite_type"] == dt and p["layer"] == layer]
            want |= {(key, sid, rank) for sid, rank in H.ref_select(ids, seed, key, 5)}
            assert sel.groups[key] == (n, min(n, 5)), (key, sel.groups[key])
        assert got == want, seed
        assert all(r["neuron_class"] == r["group"].split("|")[0] for r in sel.selected)


def test_select_invariants_order_and_other_groups():
    groups = {("spiny", "3"): 9, ("aspiny", "5"): 7}
    pool = _pool(groups, seed=1)
    cfg = default_config()
    base = {r["specimen_id"] for r in CS.select_cells(pool, cfg).selected if r["group"] == "exc|3"}
    shuffled = list(reversed(pool))
    assert {r["specimen_id"] for r in CS.select_cells(shuffled, cfg).selected if r["group"] == "exc|3"} == base
    only = [p for p in pool if p["dendrite_type"] == "spiny"]
    assert {r["specimen_id"] for r in CS.select_cells(only, cfg).selected} == base
    more = pool + _pool({("aspiny", "5"): 4}, seed=99)
    assert {r["specimen_id"] for r in CS.select_cells(more, cfg).selected if r["group"] == "exc|3"} == base


def test_select_exclusions_and_require_match():
    cfg = default_config()
    pool = _pool({("spiny", "3"): 4}, seed=2)
    ids = sorted(p["specimen_id"] for p in pool)
    pool += [dict(pool[0], specimen_id=1, dendrite_type="weird"), dict(pool[0], specimen_id=2, layer=None),
             dict(pool[0], specimen_id=3, dendrite_type=None)]
    acq = [_acq(ids[0]), _acq(ids[1], dz=0.3), _acq(ids[2], err="x")]          # ids[3] has no record
    sel = CS.select_cells(pool, cfg, acq, require_match=True)
    assert sel.excluded == {1: "no_class", 2: "no_layer", 3: "no_class", ids[1]: "acquisition_mismatch:dz",
                            ids[2]: "acquisition_mismatch:error", ids[3]: "acquisition_missing"}
    assert [r["specimen_id"] for r in sel.selected] == [ids[0]] and sel.groups == {"exc|3": (1, 1)}
    with pytest.raises(ValueError):
        CS.select_cells(pool, cfg, None, require_match=True)


def test_select_determinism_and_seed_effect():
    pool = _pool({("spiny", "3"): 11, ("aspiny", "4"): 3}, seed=3)
    a = CS.select_cells(pool, _cfg_seed(20261026))
    b = CS.select_cells(pool, _cfg_seed(20261026))
    assert a == b
    c = CS.select_cells(pool, _cfg_seed(5))
    big = lambda s: [r["specimen_id"] for r in s.selected if r["group"] == "exc|3"]    # noqa: E731
    small = lambda s: sorted(r["specimen_id"] for r in s.selected if r["group"] == "inh|4")   # noqa: E731
    assert big(a) != big(c) and small(a) == small(c)


def test_pick_order_rank_and_refusals():
    cfg = default_config()
    pool = _pool({("spiny", "3"): 3, ("aspiny", "5"): 2}, seed=4)
    pool.append(dict(pool[0], specimen_id=7, dendrite_type="weird"))
    pool.append(dict(pool[0], specimen_id=8, layer=""))
    ids = [p["specimen_id"] for p in pool if p["specimen_id"] not in (7, 8)]
    pick = [ids[3], ids[0], ids[1]]
    sel = CS.pick_cells(pool, cfg, pick)
    assert [r["specimen_id"] for r in sel.selected] == pick and all(r["rank"] == 0 for r in sel.selected)
    assert sel.excluded == {}
    for bad in ([123456789], [7], [8]):
        with pytest.raises(ValueError) as e:
            CS.pick_cells(pool, cfg, bad)
        assert str(bad[0]) in str(e.value)
    with pytest.raises(ValueError):
        CS.pick_cells(pool, cfg, [ids[0]], [_acq(ids[0], dz=0.5)], require_match=True)


def test_tables_round_trip_none_nan_and_types(tmp_path):
    rows = [dict(specimen_id=5, species="Homo Sapiens", layer="2/3", dendrite_type="spiny", reconstruction_type="full",
                 area=None, donor_id="12,3", hemisphere='le"ft', normalized_depth=0.1 + 0.2),
            dict(specimen_id=6, species="Homo Sapiens", layer="5", dendrite_type="aspiny", reconstruction_type="full",
                 area="MTG", donor_id=None, hemisphere=None, normalized_depth=NAN)]
    p = str(tmp_path / "pool.csv")
    CP.write_table(p, rows, CP.POOL_COLUMNS)
    back = CP.read_table(p, CP.POOL_TYPES)
    assert back[0]["normalized_depth"] == 0.1 + 0.2                    # repr: the float exactly
    assert back[0]["area"] is None and back[0]["donor_id"] == "12,3" and back[0]["hemisphere"] == 'le"ft'
    assert math.isnan(back[1]["normalized_depth"]) and back[1]["donor_id"] is None
    assert not os.path.exists(p + ".tmp")
    with pytest.raises(ValueError):
        CP.write_table(p, [dict(rows[0], extra=1)], CP.POOL_COLUMNS)


def test_tables_numpy_float_round_trip(tmp_path):
    # Public API: write_table writes NaN as "nan" and a float so that read_table gets it back; a numpy float64
    # (what pandas and numpy reductions return) is a float too
    p = str(tmp_path / "acq.csv")
    rec = dict(_acq(5), dz_um=np.float64(0.28), width_px=np.float64(1000.5))
    CP.write_table(p, [rec], CP.ACQUISITION_COLUMNS)
    back = CP.read_table(p, CP.ACQUISITION_TYPES)
    assert back[0]["dz_um"] == 0.28 and back[0]["width_px"] == 1000.5


def test_survey_section_in_no_signature():
    cfg = default_config()
    alt = with_overrides(cfg, {"survey.nodes_per_bin": 7, "survey.selection_seed": 1, "survey.coverage_pass_share": 0.5,
                               "survey.projection_xy_type": "X", "survey.inputs_z0_flag_um": 3.0})
    for which in ("estimator", "simulation", "full"):
        assert cfg.signature_hash(which) == alt.signature_hash(which), which


@pytest.mark.parametrize("override", [{"survey.projection_xy_type": "  "}, {"survey.nodes_mode": "some"},
                                      {"survey.coverage_pass_share": 1.5}, {"survey.acquisition_rel_tol": -1.0},
                                      {"survey.coverage_phantom_quantiles": [0.9, 0.1]},
                                      {"survey.inputs_z0_flag_um": 0.0}, {"survey.align_null_n": 5}])
def test_survey_validate_refuses(override):
    with pytest.raises(ValueError):
        with_overrides(default_config(), override)


def _api_for_acquisition(dz=0.28, resolutions=(0.1144,) * 4, failed=(False, False, False, True)):
    planes = [dict(id=500 + k, image_type="Primary", section_number=10 + k, width=1000 + 2 * k, height=2000 - 2 * k,
                   resolution=res, data_set_id=77, failed=fl) for k, (res, fl) in enumerate(zip(resolutions, failed))]
    projs = [dict(id=900, image_type="MinimumIntensityProjection - xy", section_number=None, width=1002,
                  height=2002, resolution=0.1144, data_set_id=None, failed=False)]

    def api(criteria, num_rows=200, start_row=0, timeout=60):
        if criteria.startswith("model::NeuronReconstruction"):
            return [dict(scale_factor_x=1.0, scale_factor_y=1.0, scale_factor_z=dz)]
        if criteria.startswith("model::Specimen"):
            return [dict(id=5, data_sets=[dict(id=77)])]
        if "data_set[id$eq77]" in criteria:
            return planes[start_row:start_row + num_rows]
        if "specimen_id$eq" in criteria:
            return projs + [planes[0]]                    # a plane listed twice: deduplicated by id
        raise AssertionError(criteria)
    return api


def test_acquisition_record_through_substituted_api(monkeypatch):
    # 16.1 (c): dz = scale_factor_z, the usable planes' count, the failed count, the shared resolution, medians
    monkeypatch.setattr(aio, "api_query", _api_for_acquisition())
    r = CP.acquisition_record(5)
    assert r["error"] == "" and r["dz_um"] == 0.28 and r["pixel_um"] == 0.1144
    assert r["n_planes"] == 3 and r["n_planes_failed"] == 1
    assert r["width_px"] == float(np.median([1000, 1002, 1004])) and r["height_px"] == float(np.median([2000, 1998, 1996]))
    monkeypatch.setattr(aio, "api_query", _api_for_acquisition(resolutions=(0.1144, 0.1144, 0.2288, 0.1144)))
    r2 = CP.acquisition_record(5)
    assert math.isnan(r2["pixel_um"]) and r2["pixel_values"] == "0.1144;0.2288"
    assert CS.matches_acquisition(r2, default_config()) == (False, "pixel")


# =============================================================== 16.2 ======

def test_full_transform_feeds_cell_to_image_um():
    # 16.2 notation: X = x / rho0 + dx, Y = H - y / rho0 + dy (flip) or y / rho0 + dy; z - z0; to_image_um = px x rho0
    xyz = np.array([[10.0, 20.0, 3.0], [0.5, 100.25, -1.0]])
    for flip in (None, 4000.0):
        rec = dict(shift_x_px=12.5, shift_y_px=-3.0, flip_h_px=flip, z0_um=0.7)
        t = CA.full_transform(rec)
        assert set(t) == {"shift_full_px", "flip_y_full_h", "z0_um"}
        got = cellmod.to_image_um(xyz, RHO0, **t)
        X = xyz[:, 0] / RHO0 + 12.5
        Y = (flip - xyz[:, 1] / RHO0 if flip else xyz[:, 1] / RHO0) - 3.0
        assert_allclose(got[:, 0], X * RHO0, rtol=0, atol=1e-9)          # um; float64 round-off only
        assert_allclose(got[:, 1], Y * RHO0, rtol=0, atol=1e-9)
        assert_allclose(got[:, 2], xyz[:, 2] - 0.7, rtol=0, atol=1e-12)


def _images():
    rows = [dict(id=1, image_type="MinimumIntensityProjection - xy", width=1015, height=2030),
            dict(id=2, image_type="MaximumIntensityProjection - xy", width=990, height=1990),
            dict(id=3, image_type="MinimumIntensityProjection - yz", width=180, height=2000),
            dict(id=4, image_type="Projection", width=1025, height=2000)]
    rows += [dict(id=10 + k, image_type="Primary", width=w, height=2000, section_number=k, failed=f)
             for k, (w, f) in enumerate(((1000, False), (1000, False), (1002, False), (998, False), (5000, True)))]
    for r in rows:
        r.setdefault("section_number", None)
        r.setdefault("failed", False)
        r.update(resolution=RHO0, data_set_id=None)
    df = pd.DataFrame(rows)
    df["is_plane"] = df["image_type"] == "Primary"
    return df


def test_xy_projections_within_two_percent_of_usable_planes():
    # 16.2 (a): width and height each within 2 % of the usable Primary planes' medians (1000 x 2000)
    got = sorted(int(i) for i in CA.xy_projections(_images(), 0.02)["id"])
    assert got == [1, 2]


def test_choose_projection_type_then_darkest():
    a = [np.full((4, 4), v) for v in (100.0, 50.0, 80.0, 60.0)]
    T = "MinimumIntensityProjection - xy"
    k, rule, meds = CA.choose_projection(["  minimumintensityprojection - XY ", "X", T, "Y"], a, T)
    assert (k, rule) == (2, "type") and meds == [100.0, 50.0, 80.0, 60.0]       # typed: the darker of 0 and 2
    k, rule, _ = CA.choose_projection(["A", "B", "C", "D"], a, T)
    assert (k, rule) == (1, "darkest")
    with pytest.raises(ValueError):
        CA.choose_projection(["A", "B"], a, T)


def test_search_score_equals_brute_force():
    rng = np.random.default_rng(5)
    D = rng.random((37, 52))
    m1 = rng.random((37, 52)) > 0.9
    m2 = np.zeros_like(m1)
    m2[10:14, 30:41] = True
    for M in (0, 3, 7):
        got = CA.search_score(D, [m1, m2], M)
        # rtol 1e-7: search_score casts the mask to float32 before fftconvolve, whose spectrum scipy.fft then
        # computes in single precision (relative error ~3e-8 measured here; 0 with a float64 mask) -- finding F11
        assert_allclose(got, H.ref_search_score(D, [m1, m2], M), rtol=1e-7)


def test_search_score_window_larger_than_half_the_array():
    # 16.2 (c) defines the score for every |dx|, |dy| <= M; the alignment's own M never exceeds rows // 2 - 1,
    # but the public function takes M as given
    rng = np.random.default_rng(6)
    D = rng.random((12, 16))
    m = np.zeros((12, 16), bool)
    m[2:4, 3:9] = True
    M = 7                                                     # > rows // 2 = 6
    assert_allclose(CA.search_score(D, [m], M), H.ref_search_score(D, [m], M), rtol=1e-7)   # float32: see above


def test_block_shuffle_permutes_tiles_keeps_margins():
    rng = np.random.default_rng(7)
    D = rng.random((37, 45))
    D0 = D.copy()
    b = 8
    out = CA.block_shuffle(D, b, np.random.default_rng(1))
    assert_array_equal(D, D0)                                          # input not mutated
    nh, nw = 37 // b, 45 // b
    assert_array_equal(out[nh * b:, :], D[nh * b:, :])
    assert_array_equal(out[:, nw * b:], D[:, nw * b:])

    def tiles(A):
        return sorted(tuple(A[i * b:(i + 1) * b, j * b:(j + 1) * b].ravel()) for i in range(nh) for j in range(nw))
    assert tiles(out) == tiles(D)
    assert not np.array_equal(out, D)
    assert_array_equal(out, CA.block_shuffle(D, b, np.random.default_rng(1)))


def test_fine_search_equals_brute_force():
    rng = np.random.default_rng(8)
    D = rng.random((31, 43))
    mask = np.zeros((31, 43), bool)
    rr = np.arange(3, 28)
    mask[rr, (rr * 1.3).astype(int)] = True
    mask[0:2, 40:43] = True                                    # pixels that leave the crop for some shifts
    res = CA.fine_search(D, mask, 4)
    a, b, S, S00, _ = H.ref_fine_search(D, mask, 4)
    assert (res[0], res[1]) == (a, b)
    assert_allclose(res[2], S, rtol=1e-12)
    assert_allclose(res[3], S00, rtol=1e-12)


def _line_swc():
    # Block 0's table (um): three dendrites of type 3 from a soma at (130, 110) um, inside a 160 x 120 px array
    # at 1.83 um/px for shifts of +-10 px and both flips
    rows = [(1, 1, 130.0, 110.0, 0.0, 5.0, -1)]
    nid = 2
    for (x0, y0, ang, n) in ((130.0, 110.0, 0.4, 25), (130.0, 110.0, 2.6, 20), (130.0, 110.0, 4.4, 18)):
        par = 1
        for k in range(1, n + 1):
            rows.append((nid, 3, x0 + 4.0 * k * math.cos(ang), y0 + 4.0 * k * math.sin(ang), 0.0, 0.5, par))
            par = nid
            nid += 1
    return pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"])


def _painted(swc, shape, rho, dx, dy, flip, seed=0):
    """The skeleton drawn 1 px wide (dark on bright, small noise) at the true array shift: the mean darkness under
    the mask then peaks at that shift alone (a wider line makes +-1 px a near-tie decided by the noise)."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, 200.0)
    m = aia.render_skeleton_mask(swc, shape, rho, (dx, dy), shape[0] if flip else None, (3, 4))
    img[m] = 60.0
    return np.clip(img + rng.normal(0, 3, shape), 0, 255)


def test_coarse_alignment_known_shift_window_and_scaling():
    # 16.2 (b): the array shift, full-res = f_c x, H = f_c rows when flipped, M = min(floor(250/rho_c),
    # rows/2 - 1, cols/2 - 1), at_border iff max(|dx|, |dy|) >= M - 1
    cfg = default_config()
    f_c = 16
    rho_c = f_c * RHO0
    swc = _line_swc()
    for (dx0, dy0, flip) in ((7, -4, False), (-5, 9, True)):
        P = _painted(swc, (120, 160), rho_c, dx0, dy0, flip)
        co = CA.coarse_alignment(swc, P, rho_c, f_c, cfg.survey)
        M = min(int(math.floor(250.0 / rho_c)), 120 // 2 - 1, 160 // 2 - 1)
        assert co["half_window_px"] == M == 59
        assert (co["dx_px"], co["dy_px"], co["flip"]) == (dx0, dy0, flip), co
        assert (co["shift_x_px"], co["shift_y_px"]) == (f_c * dx0, f_c * dy0)
        assert co["flip_h_px"] == (f_c * 120 if flip else None)
        assert co["at_border"] == (max(abs(dx0), abs(dy0)) >= M - 1)


@pytest.mark.parametrize("R,at_c,at_f,aside", [(2.0, False, False, False), (1.9999, False, False, True),
                                              (NAN, False, False, True), (None, False, False, True),
                                              (5.0, True, False, True), (5.0, False, True, True)])
def test_verdict_rule(R, at_c, at_f, aside):
    # 16.2 (e): set aside when R < 2.0 or NaN, or at_border, or fine_at_border
    rec = dict(score_ratio=R, coarse=dict(at_border=at_c), fine=dict(at_border=at_f))
    got, reasons = CA.verdict(rec, default_config().survey)
    assert got == aside and (len(reasons) > 0) == aside


# ---- the runner (16.2 (g)), with substituted step functions -------------

def _selection(tmp, sids):
    rows = [dict(specimen_id=s, species="Homo Sapiens", layer="3", dendrite_type="spiny", reconstruction_type="full",
                 area="MTG", donor_id="9", hemisphere="left", normalized_depth=0.3, neuron_class="exc",
                 group="exc|3", rank=i) for i, s in enumerate(sids)]
    path = os.path.join(tmp, "selection.csv")
    CP.write_table(path, rows, CP.SELECTION_COLUMNS)
    return path


@pytest.fixture
def fake_steps(monkeypatch):
    calls = []

    def align(ctx, cfg):
        calls.append(("align", ctx.sid))
        ctx.data["alignment"] = dict(set_aside=(ctx.sid == 2), shift_x_px=0.0, shift_y_px=0.0, flip_h_px=None,
                                     z0_um=0.0)
        ctx.save()
        return {}

    def nodes(ctx, cfg):
        calls.append(("nodes", ctx.sid))
        if ctx.sid == 3:
            raise RuntimeError("boom")
        return {}
    monkeypatch.setitem(SC.STEPS, "align", align)
    monkeypatch.setitem(SC.STEPS, "nodes", nodes)
    monkeypatch.setitem(SC.SAY, "align", str)
    monkeypatch.setitem(SC.SAY, "nodes", str)
    return calls


def test_runner_resume_failure_set_aside(tmp_path, fake_steps):
    cfg = default_config()
    sel = _selection(str(tmp_path), [1, 2, 3])
    out = str(tmp_path / "survey")
    rows = {r["specimen_id"]: r for r in SC.run(cfg, sel, out, None, ["align", "nodes"], log=lambda m: None)}
    assert rows[1]["align"] == rows[1]["nodes"] == "done"
    assert rows[2]["nodes"] == "skipped:set_aside"
    assert rows[3]["nodes"] == "failed" and "RuntimeError" in rows[3]["error"]
    for sid, step, exists in ((1, "nodes", True), (2, "nodes", False), (3, "nodes", False), (3, "align", True)):
        assert survey_io.CellDir(out, sid).done(step) == exists, (sid, step)
    mk = survey_io.read_marker(survey_io.CellDir(out, 1), "align")
    assert {"summary", "seconds", "config_hash", "finished_utc"} <= set(mk)
    fake_steps.clear()
    SC.run(cfg, sel, out, None, ["align", "nodes"], log=lambda m: None)
    assert fake_steps == [("nodes", 3)]                               # done steps skipped, the failure retried
    fake_steps.clear()
    SC.run(cfg, sel, out, None, ["align", "nodes"], include_set_aside=True, log=lambda m: None)
    assert ("nodes", 2) in fake_steps and ("nodes", 1) not in fake_steps


def test_runner_task_index(tmp_path, fake_steps):
    cfg = default_config()
    sel = _selection(str(tmp_path), [11, 12, 13])
    out = str(tmp_path / "survey")
    rows = SC.run(cfg, sel, out, None, ["align"], task_index=1, log=lambda m: None)
    assert [r["specimen_id"] for r in rows] == [12] and fake_steps == [("align", 12)]
    for bad in (3, -1):
        with pytest.raises(ValueError):
            SC.run(cfg, sel, out, None, ["align"], task_index=bad, log=lambda m: None)
    with pytest.raises(ValueError):
        SC.run(cfg, sel, out, None, ["align"], task_index=0, cells=[11], log=lambda m: None)


def test_runner_status_table_one_row_per_cell(tmp_path, fake_steps):
    # 16.2 (g): <out>/status.csv, one row per cell -- also after a run on one cell of the selection
    cfg = default_config()
    sel = _selection(str(tmp_path), [11, 12, 13])
    out = str(tmp_path / "survey")
    SC.run(cfg, sel, out, None, ["align"], log=lambda m: None)
    SC.run(cfg, sel, out, None, ["align"], task_index=2, log=lambda m: None)
    st = CP.read_table(os.path.join(out, "status.csv"), {"specimen_id": "int"})
    assert sorted(r["specimen_id"] for r in st) == [11, 12, 13], st


def test_runner_steps_in_the_order_given(tmp_path, fake_steps):
    # 16.2 (g): "the named steps, in the order the step list gives them"
    cfg = default_config()
    sel = _selection(str(tmp_path), [21])
    out = str(tmp_path / "survey")
    survey_io.CellDir(out, 21).save_context(dict(alignment=dict(set_aside=False)))
    SC.run(cfg, sel, out, None, ["nodes", "align"], log=lambda m: None)
    assert fake_steps == [("nodes", 21), ("align", 21)], fake_steps


# =============================================================== 16.3 ======

def test_plane_operators_reference():
    rng = np.random.default_rng(9)
    st = rng.integers(0, 256, (6, 7, 9)).astype(np.uint8)
    valid = np.array([True, False, True, True, False, True])
    ops = PC.plane_operators(st, valid)
    s = st[valid].astype(float)
    for op, ref in (("min", s.min(0)), ("max", s.max(0)), ("mean", s.mean(0)), ("median", np.median(s, 0))):
        assert ops[op].dtype == np.float64 and ops[op].shape == (7, 9)
        assert_array_equal(ops[op], ref)                         # exact: small integers and their means
    with pytest.raises(ValueError):
        PC.plane_operators(st, [True] + [False] * 5)


def test_shift_match_known_offset_and_reference():
    rng = np.random.default_rng(10)
    s, Hh, Ww = 3, 20, 24
    O = rng.random((Hh, Ww)) * 100
    for (a0, b0) in ((1, 1), (2, 0), (-3, 2)):
        P_big = rng.random((Hh + 2 * s, Ww + 2 * s)) * 100
        P_big[s + b0:s + b0 + Hh, s + a0:s + a0 + Ww] = 3.0 + 2.0 * O
        m = PC.shift_match(P_big, O, s)
        assert (m["shift_x_px"], m["shift_y_px"]) == (a0, b0)
        assert_allclose([m["r"], m["slope"], m["intercept"], m["rms_gl"]], [1.0, 2.0, 3.0, 0.0], atol=1e-9)
        crop = PC.aligned_crop(P_big, a0, b0, s)
        assert_array_equal(crop, 3.0 + 2.0 * O)
    # a noisy case against numpy.corrcoef and numpy.polyfit
    P_big = rng.random((Hh + 2 * s, Ww + 2 * s)) * 100
    P_big[s + 1:s + 1 + Hh, s - 2:s - 2 + Ww] += 0.8 * O
    m = PC.shift_match(P_big, O, s)
    r, a, b = H.ref_shift_search(P_big, O, s)
    assert (m["shift_x_px"], m["shift_y_px"]) == (a, b)
    assert_allclose(m["r"], r, rtol=1e-10)
    p = P_big[s + b:s + b + Hh, s + a:s + a + Ww].ravel()
    slope, icpt = np.polyfit(O.ravel(), p, 1)
    assert_allclose([m["slope"], m["intercept"]], [slope, icpt], rtol=1e-9)
    assert_allclose(m["rms_gl"], np.sqrt(np.mean((p - icpt - slope * O.ravel()) ** 2)), rtol=1e-9)


def test_classify_ratio_rule_by_hand():
    by = lambda **r: {k: dict(r=v) for k, v in r.items()}       # noqa: E731
    best, m, q, v = PC.classify(by(min=0.99, mean=0.96), 0.9, 4.0)
    assert_allclose(q, (1 - 0.96 ** 2) / (1 - 0.99 ** 2), rtol=1e-12)   # 3.94: below 4
    assert best == "min" and v.startswith("unclear")
    best, m, q, v = PC.classify(by(min=0.995, mean=0.96, max=NAN), 0.9, 4.0)
    assert v == "matches min"
    assert_allclose(q, (1 - 0.96 ** 2) / (1 - 0.995 ** 2), rtol=1e-12)          # 7.86
    assert_allclose(m, 0.995 - 0.96, rtol=1e-12)
    best, m, q, v = PC.classify(by(max=0.95), 0.9, 4.0)
    assert v == "matches max" and q == math.inf
    best, m, q, v = PC.classify(by(min=0.89, mean=0.1), 0.9, 4.0)
    assert v.startswith("unclear")
    assert PC.classify(by(min=NAN), 0.9, 4.0)[3] == "unclear: no finite r"
    assert PC.classify(by(min=1.0, mean=0.5), 0.9, 4.0)[2] == math.inf


def test_place_window_by_hand():
    for (cx, cy, h, s, X, Y) in ((500.4, 600.5, 256, 3, 7605, 9433), (10.0, 10.0, 256, 3, 7605, 9433),
                                 (7600.0, 9430.0, 256, 3, 7605, 9433), (300.5, 260.49, 256, 3, 518, 600)):
        assert PC.place_window(cx, cy, h, s, X, Y) == H.ref_place_window(cx, cy, h, s, X, Y)
    with pytest.raises(ValueError):
        PC.place_window(0, 0, 256, 3, 517, 9000)                   # X < 2 (h + s) = 518


def test_soma_center_rows_and_transform():
    swc = pd.DataFrame([(1, 1, 10.0, 20.0, 0.0, 5.0, -1), (2, 1, 30.0, 40.0, 0.0, 5.0, -1), (3, 1, 99.0, 99.0, 0, 5, 1),
                        (4, 3, 50.0, 50.0, 0.0, 1.0, 1)], columns=["id", "type", "x", "y", "z", "r", "parent"])
    cx, cy = PC.soma_center(swc, RHO0)
    assert_allclose((cx, cy), (20.0 / RHO0, 30.0 / RHO0), rtol=1e-12)
    cx, cy = PC.soma_center(swc, RHO0, (5.0, -2.0), 1000.0)
    assert_allclose((cx, cy), (20.0 / RHO0 + 5.0, 1000.0 - 30.0 / RHO0 - 2.0), rtol=1e-12)
    only_child = swc[swc["id"].isin([3, 4])]
    assert_allclose(PC.soma_center(only_child, RHO0), (99.0 / RHO0, 99.0 / RHO0), rtol=1e-12)
    with pytest.raises(ValueError):
        PC.soma_center(swc[swc["type"] == 3], RHO0)


def test_list_images_equals_images_frame_of_subimage_rows(monkeypatch):
    # 16.3 (a): subimage_rows = both queries deduplicated by id keeping the first, every field verbatim;
    # list_images(sid) = images_frame(subimage_rows(sid)): the columns of list_images, projections first, then the
    # planes by section_number (list_images' docstring, Block 0)
    monkeypatch.setattr(aio, "api_query", _api_for_acquisition())
    rows = aio.subimage_rows(5)
    assert [r["id"] for r in rows] == [500, 501, 502, 503, 900]          # data-set rows first, plane 500 once
    a, b = aio.list_images(5), aio.images_frame(rows)
    pd.testing.assert_frame_equal(a, b)
    assert {"id", "image_type", "section_number", "width", "height", "resolution", "data_set_id", "failed"} <= set(a)
    assert list(a["id"]) == [900, 500, 501, 502, 503]


# =============================================================== 16.4 ======

PBS = ("survey_common.sh", "survey_prepare.pbs", "survey_cells.pbs", "survey_pool.pbs")


@pytest.mark.parametrize("name", PBS)
def test_pbs_ascii_lf_bash_n_headers(name):
    path = H.WS / "scripts" / "pbs" / name
    data = path.read_bytes()
    assert all(b < 128 for b in data), "non-ASCII byte"
    assert b"\r" not in data, "CR in the file"
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
    text = data.decode("ascii")
    if name.endswith(".pbs"):
        assert "#PBS -N " in text and "#PBS -k eo" in text
        assert ("#PBS -J 0-50" in text) == (name == "survey_cells.pbs")
        assert "ROADMAP_" in text or name == "survey_prepare.pbs"


@pytest.mark.parametrize("env,code,needle", [
    (dict(PBS_ARRAY_INDEX="1"), 0, "--task-index 1"),
    (dict(PBS_ARRAY_INDEX="3"), 4, "FATAL"),                                   # a task beyond the 3-row selection
    (dict(PBS_ARRAY_INDEX="2", DIAM_N_TASKS="2"), 4, "disagree"),              # -J against DIAM_N_TASKS
    (dict(PBS_ARRAY_INDEX="1", DIAM_CELLS="2,3"), 0, "nothing to do"),         # DIAM_CELLS: task 0 only
    (dict(PBS_ARRAY_INDEX="0", DIAM_CELLS="2,3", DIAM_FORCE="align"), 0, "--cells 2,3 --force align")])
def test_pbs_array_dry_run(tmp_path, env, code, needle):
    # 16.4 (b): task i runs --task-index i --cache-per-cell over the selection; DIAM_DRYRUN=1 prints, runs nothing
    surv = tmp_path / "survey"
    surv.mkdir()
    (surv / "selection.csv").write_text("specimen_id,group\n1,exc|3\n2,exc|3\n3,exc|5\n")
    e = dict(PATH=os.environ["PATH"], HOME=str(tmp_path), DIAM_DRYRUN="1", RUN_TAG="t", DIAM_CODE=str(H.WS),
             DIAM_SURVEY=str(surv), DIAM_CACHE=str(tmp_path / "cache"), **env)
    r = subprocess.run(["bash", "scripts/pbs/survey_cells.pbs"], cwd=str(H.WS), env=e, capture_output=True, text=True)
    assert r.returncode == code, r.stdout + r.stderr
    assert needle in r.stdout + r.stderr
    if code == 0 and "DRYRUN" in r.stdout:
        assert "--cache-per-cell" in r.stdout and "--out-dir %s" % surv in r.stdout
