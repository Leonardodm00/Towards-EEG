"""Smoke test for the check of what Allen's projections are -- Block 16.3 in
specs/SPEC.md (D-090.1; 2026-10-27).

A synthetic stack: 24 planes of 150 x 166 px, seven dark tubes each in focus
at its own plane and blurred away from it, on a bright noisy field. Its
exact pixelwise minimum and maximum are placed in images 2 px larger than
the planes with the content offset by (+1, +1) px, as Allen's projections
are 2 px larger than its planes. A synthetic source serves the SubImage rows
(with a field the code does not know), the SWC (a soma at the centre) and
the crops (allen_image_io.SyntheticFetcher per image; optionally as JPEG).

Checks
    test_known_answer   check_projection: the minimum image matches min at
                        offset (+1, +1) with r* >= 0.999 (exact integers: r =
                        1 to roundoff), the maximum image matches max; served
                        as JPEG (quality 85), still matches min at (+1, +1)
    test_reference      r* equals numpy.corrcoef on the shifted crop, the line
                        numpy.polyfit, the RMS the residual's (a noisy
                        projection, so r < 1); plane_operators equals the
                        operators written out plane by plane
    test_convergence    not a discretisation: skipped
    test_invariants     an affine change of the projection leaves r* and the
                        offset unchanged; permuting the planes leaves the
                        operators unchanged
    test_contract       the result's keys; images_frame equals the table the
                        former list_images built; subimage_rows through a
                        substituted api_query keeps every field and the first
                        of a repeated id; the script end to end: the JSON (the
                        unknown field verbatim), the CSV rows, the figure
    test_determinism    two runs give the same JSON (but the seconds)
    test_edge_cases     fewer than 2 valid planes refused; a missing section
                        excluded through valid; a constant projection gives
                        NaN and unclear; the window moved inside at an edge, a
                        too-small image refused; --check none lists only; no
                        xy candidate listed, not failed; no soma refused

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_projection_check.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import importlib.metadata
import io
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import allen_image_io as aio  # noqa: E402
import projection_check as PCS  # noqa: E402
from allen_diameter.analysis import projection_check as PC  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import cell_pool, survey_io  # noqa: E402

SEED = 20261027
REPORT_PACKAGES = ("numpy", "scipy", "pandas", "matplotlib", "Pillow")
RHO0 = 0.1144
HP, WP, NPL = 150, 166, 24            # the planes (rows, cols) and their number
SID = 777
H, S = 32, 3                          # the test window's half-side and margin (px)


# --------------------------------------------------------------- fixtures ---

def synthetic_stack(seed=SEED):
    """(NPL, HP, WP) uint8: dark tubes, each darkest and sharpest at its own plane, on 200 gl with noise SD 3."""
    rng = np.random.default_rng(seed)
    planes = np.full((NPL, HP, WP), 200.0)
    yy, xx = np.mgrid[0:HP, 0:WP]
    for _ in range(7):
        x0, y0, th = rng.uniform(15, WP - 15), rng.uniform(15, HP - 15), rng.uniform(0, math.pi)
        dist = np.abs(-(xx - x0) * math.sin(th) + (yy - y0) * math.cos(th))
        k_t = int(rng.integers(2, NPL - 2))
        for k in range(NPL):
            width = 1.5 + 0.6 * abs(k - k_t)                          # px: the shadow widens away from focus
            depth = 110.0 * math.exp(-0.5 * ((k - k_t) / 3.0) ** 2) + 8.0
            planes[k] -= depth * np.exp(-0.5 * (dist / width) ** 2)
    planes += rng.normal(0.0, 3.0, planes.shape)
    return np.clip(np.round(planes), 0, 255).astype(np.uint8)


def projection_image(op_img, fill=200, dx=1, dy=1):
    """The operator's image in an image 2 px larger, its content at offset (dx, dy), each in 0..2."""
    P = np.full((HP + 2, WP + 2), fill, dtype=np.uint8)
    P[dy:dy + HP, dx:dx + WP] = np.clip(np.round(op_img), 0, 255).astype(np.uint8)
    return P


def jpeg(arr, quality=85):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="JPEG", quality=quality)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("L"))


class MultiFetcher(aio.ImageFetcher):
    def __init__(self, by_id, as_jpeg=False):
        self.by_id, self.as_jpeg, self.n_requests = by_id, as_jpeg, 0

    def get(self, image_id, left, top, width, height, downsample=0):
        self.n_requests += 1
        img = self.by_id[int(image_id)].get(image_id, left, top, width, height, downsample)
        return jpeg(img) if self.as_jpeg else img


def window(stack_ops):
    left, top = PC.place_window(WP / 2.0, HP / 2.0, H, S, WP, HP)
    return {k: v[top:top + 2 * H, left:left + 2 * H] for k, v in stack_ops.items()}, left, top


def big_crop(P, left, top):
    """The projection's crop over the window widened by S, in the projection's own pixels."""
    return P[top - S:top + 2 * H + S, left - S:left + 2 * H + S]


class Source:
    """A synthetic Allen source: SubImage rows, an SWC with its soma at the planes' centre, the crops."""

    def __init__(self, tmp, as_jpeg=False, with_xy=True, soma=True, missing_section=None):
        self.stack = synthetic_stack()
        ops = PC.plane_operators(self.stack)
        by_id, rows = {}, []
        if with_xy:
            for iid, op in ((502, "max"), (501, "min")):              # the maximum first: the order must not matter
                by_id[iid] = aio.SyntheticFetcher(projection_image(ops[op]))
                kind = "MaximumIntensityProjection - xy" if op == "max" else "MinimumIntensityProjection - xy"
                rows.append(dict(id=iid, image_type=kind, section_number=None, width=WP + 2, height=HP + 2,
                                 resolution=RHO0, data_set_id=None, failed=False, storage_directory="/proj/%d/" % iid))
        by_id[503] = aio.SyntheticFetcher(np.full((HP + 2, 40), 200, dtype=np.uint8))
        rows.append(dict(id=503, image_type="MinimumIntensityProjection - yz", section_number=None, width=40,
                         height=HP + 2, resolution=RHO0,
                         data_set_id=None, failed=False, storage_directory="/proj/503/"))
        for k in range(NPL):
            if missing_section is not None and k == missing_section:
                continue
            by_id[1000 + k] = aio.SyntheticFetcher(self.stack[k])
            rows.append(dict(id=1000 + k, image_type="Primary", section_number=10 + k, width=WP, height=HP,
                             resolution=RHO0, data_set_id=99, failed=False))
        self.rows, self.fetcher = rows, MultiFetcher(by_id, as_jpeg)
        self.swc = os.path.join(tmp, "cell.swc")
        with open(self.swc, "w") as f:
            if soma:
                f.write("1 1 %.4f %.4f 20.0 5.0 -1\n" % (WP / 2.0 * RHO0, HP / 2.0 * RHO0))
            f.write("2 3 %.4f %.4f 20.0 0.5 %d\n" % (WP / 2.0 * RHO0 + 3.0, HP / 2.0 * RHO0, 1 if soma else -1))

    def subimage_rows(self, sid):
        return copy.deepcopy(self.rows)

    def swc_path(self, sid):
        return self.swc


def small_cfg():
    cfg = default_config()
    return dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, projcheck_half_px=H, projcheck_max_shift_px=S))


def run_quiet(*args, **kw):
    lines = []
    out = PCS.run(*args, log=lines.append, **kw)
    return out, lines


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    stack = synthetic_stack()
    ops_full = PC.plane_operators(stack)
    ops, left, top = window(ops_full)
    for op in ("min", "max"):
        res = PC.check_projection(big_crop(projection_image(ops_full[op]), left, top), ops, S, 0.9, 4.0)
        m = res["by_operator"][op]
        assert res["verdict"] == "matches %s" % op and res["best"] == op, res["verdict"]
        assert (m["shift_x_px"], m["shift_y_px"]) == (1, 1), m
        # the operator's integers placed exactly: r = 1 up to float64 roundoff, and the next operator's line leaves
        # orders of magnitude more residual (the min's next is the mean, r about 0.97 on this stack)
        assert m["r"] >= 0.999 and m["rms_gl"] <= 1e-6, m
        assert res["ratio"] >= 1e6, res["ratio"]
    # an asymmetric offset: (+2, 0) is reported as (+2, 0), not (0, +2)
    res = PC.check_projection(big_crop(projection_image(ops_full["min"], dx=2, dy=0), left, top), ops, S, 0.9, 4.0)
    m = res["by_operator"]["min"]
    assert res["verdict"] == "matches min" and (m["shift_x_px"], m["shift_y_px"]) == (2, 0), m
    # served as JPEG (the server's compression stand-in): the minimum of the original planes, compressed, against
    # the minimum of the compressed planes
    jstack = np.stack([jpeg(p) for p in stack])
    jops, _, _ = window(PC.plane_operators(jstack))
    res = PC.check_projection(big_crop(jpeg(projection_image(ops_full["min"])), left, top), jops, S, 0.9, 4.0)
    m = res["by_operator"]["min"]
    assert res["verdict"] == "matches min" and (m["shift_x_px"], m["shift_y_px"]) == (1, 1), (res["verdict"], m)
    # JPEG on both sides, measured on 9 synthetic stacks (3 shapes x 3 seeds, 2026-10-27): r* of the min 0.997-0.999,
    # the ratio 12-31, the min's line RMS 2.3-2.6 gl against the mean's 8-13 gl; the bounds leave room
    assert m["r"] >= 0.99 and res["ratio"] >= 8.0 and m["rms_gl"] <= 4.0, (m, res["ratio"])
    assert res["margin"] < 0.05, "the margin in r alone would not separate min from mean here: %r" % res["margin"]


def test_reference():
    stack = synthetic_stack()
    ops_full = PC.plane_operators(stack)
    ops, left, top = window(ops_full)
    rng = np.random.default_rng(SEED + 1)
    P = np.clip(projection_image(ops_full["min"]).astype(float) + rng.normal(0, 6.0, (HP + 2, WP + 2)), 0, 255)
    Pb = big_crop(P, left, top)
    m = PC.shift_match(Pb, ops["min"], S)
    crop = PC.aligned_crop(Pb, m["shift_x_px"], m["shift_y_px"], S)
    r_ref = np.corrcoef(crop.ravel(), ops["min"].ravel())[0, 1]
    assert m["r"] < 0.999 and abs(m["r"] - r_ref) <= 1e-12, (m["r"], r_ref)
    slope, intercept = np.polyfit(ops["min"].ravel(), crop.ravel(), 1)
    assert abs(m["slope"] - slope) <= 1e-9 and abs(m["intercept"] - intercept) <= 1e-7, (m, slope, intercept)
    rms = math.sqrt(np.mean((crop.ravel() - intercept - slope * ops["min"].ravel()) ** 2))
    assert abs(m["rms_gl"] - rms) <= 1e-9
    # aligned_crop at (a, b) = (2, 0) is the projection's window shifted 2 px along x
    Pa = big_crop(projection_image(ops_full["min"], dx=2, dy=0), left, top)
    assert np.array_equal(PC.aligned_crop(Pa, 2, 0, S), Pa[S:S + 2 * H, S + 2:S + 2 + 2 * H])
    assert np.array_equal(PC.aligned_crop(Pa, 2, 0, S).astype(float), ops["min"])
    # the soma's pixel under Block 0's transform (no shift, no flip; then a shift and a flip)
    import allen_image_plot as aip
    with tempfile.TemporaryDirectory() as tmp:
        swc = aip.read_swc(Source(tmp).swc)
    cx, cy = PC.soma_center(swc, RHO0)
    assert abs(cx - WP / 2.0) <= 1e-9 and abs(cy - HP / 2.0) <= 1e-9, (cx, cy)
    cx, cy = PC.soma_center(swc, RHO0, (10.0, -4.0), 1000.0)
    assert abs(cx - (WP / 2.0 + 10.0)) <= 1e-9 and abs(cy - (1000.0 - HP / 2.0 - 4.0)) <= 1e-9, (cx, cy)
    # every offset's r against numpy.corrcoef: the reported one is the largest
    rs = [np.corrcoef(PC.aligned_crop(Pb, a, b, S).ravel(), ops["min"].ravel())[0, 1]
          for b in range(-S, S + 1) for a in range(-S, S + 1)]
    assert abs(max(rs) - m["r"]) <= 1e-12
    # the operators written out plane by plane
    f = stack.astype(float)
    srt = np.sort(f, axis=0)
    ref = dict(min=functools.reduce(np.minimum, list(f)), max=functools.reduce(np.maximum, list(f)),
               mean=sum(list(f)) / NPL, median=0.5 * (srt[NPL // 2 - 1] + srt[NPL // 2]))
    for op in PC.OPERATORS:
        assert np.max(np.abs(ops_full[op] - ref[op])) <= 1e-9, op


def test_convergence():
    raise unittest.SkipTest("no discretisation: a correlation over a pixel window")


def test_invariants():
    stack = synthetic_stack()
    ops_full = PC.plane_operators(stack)
    ops, left, top = window(ops_full)
    rng = np.random.default_rng(SEED + 2)
    P = projection_image(ops_full["min"]).astype(float) + rng.normal(0, 6.0, (HP + 2, WP + 2))
    a = PC.shift_match(big_crop(P, left, top), ops["mean"], S)
    b = PC.shift_match(big_crop(7.0 + 0.5 * P, left, top), ops["mean"], S)
    assert abs(a["r"] - b["r"]) <= 1e-12 and (a["shift_x_px"], a["shift_y_px"]) == (b["shift_x_px"], b["shift_y_px"])
    perm = rng.permutation(NPL)
    ops_p = PC.plane_operators(stack[perm])
    for op in PC.OPERATORS:
        assert np.array_equal(ops_p[op], ops_full[op]), op


def _old_list_images_frame(rows):
    """The table list_images built before Block 16.3 split it (its body, verbatim but the queries)."""
    df = pd.DataFrame(rows).drop_duplicates(subset="id")
    keep = ["id", "image_type", "section_number", "width", "height", "resolution", "data_set_id", "failed"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["is_plane"] = df["image_type"] == "Primary"
    return df.sort_values(["is_plane", "section_number"]).reset_index(drop=True)


def test_contract():
    with tempfile.TemporaryDirectory() as tmp:
        src = Source(tmp)
        ops, left, top = window(PC.plane_operators(src.stack))
        full = PC.plane_operators(src.stack)
        res = PC.check_projection(big_crop(projection_image(full["max"]), left, top), ops, S, 0.9, 4.0)
        assert set(res) == {"by_operator", "best", "margin", "ratio", "verdict"}
        assert tuple(res["by_operator"]) == PC.OPERATORS
        assert all(set(v) == {"r", "shift_x_px", "shift_y_px", "slope", "intercept", "rms_gl"}
                   for v in res["by_operator"].values())
        rows = src.subimage_rows(SID) + [dict(src.rows[0])]                    # a repeated id
        pd.testing.assert_frame_equal(aio.images_frame(rows), _old_list_images_frame(rows))
        # subimage_rows through a substituted api_query: every field kept, the first of a repeated id
        canned = {"Specimen": [dict(id=SID, data_sets=[dict(id=99)])],
                  "ds": [dict(r) for r in src.rows if r["image_type"] == "Primary"],
                  "sp": [dict(r) for r in src.rows if r["image_type"] != "Primary"] + [dict(src.rows[-1], width=1)]}

        def fake_api(criteria, num_rows=200, start_row=0, timeout=60):
            if criteria.startswith("model::Specimen"):
                return canned["Specimen"]
            if "data_set[id$eq99]" in criteria:
                return canned["ds"] if start_row == 0 else []
            return canned["sp"]
        real = aio.api_query
        aio.api_query = fake_api
        try:
            got = aio.subimage_rows(SID)
            frame = aio.list_images(SID)
        finally:
            aio.api_query = real
        ids = [r["id"] for r in got]
        assert len(ids) == len(set(ids)) == len(src.rows), (len(ids), len(src.rows))
        last = next(r for r in got if r["id"] == src.rows[-1]["id"])
        assert last["width"] == WP, "the first of a repeated id is kept"
        assert next(r for r in got if r["id"] == 501)["storage_directory"] == "/proj/501/"
        pd.testing.assert_frame_equal(frame, aio.images_frame(got))
        # the script end to end
        out = os.path.join(tmp, "out")
        recs, lines = run_quiet(small_cfg(), [SID], [SID], src, out)
        rec = survey_io.read_json(os.path.join(out, "projections_%d.json" % SID))
        assert {"specimen_id", "subimages", "planes", "candidates", "darkest", "chosen", "window", "n_planes_used",
                "checks", "seconds"} <= set(rec), sorted(rec)
        assert rec["chosen"] == dict(id=501, rule="type", image_type="MinimumIntensityProjection - xy"), rec["chosen"]
        assert sorted(r["id"] for r in rec["subimages"]) == [501, 502, 503]
        assert {r["id"]: r["storage_directory"] for r in rec["subimages"]}[502] == "/proj/502/"
        assert sorted(rec["candidates"]) == [501, 502] and rec["darkest"]["id"] == 501, rec["darkest"]
        assert rec["planes"] == dict(n_usable=NPL, n_failed=0, width_px=float(WP), height_px=float(HP)), rec["planes"]
        assert rec["n_planes_used"] == NPL and rec["window"]["center_source"] == "soma (no shift, no flip)"
        by = {c["id"]: c for c in rec["checks"]}
        assert by[501]["verdict"] == "matches min" and by[502]["verdict"] == "matches max", [c["verdict"] for c in rec["checks"]]
        assert (by[501]["by_operator"]["min"]["shift_x_px"], by[501]["by_operator"]["min"]["shift_y_px"]) == (1, 1)
        tab = cell_pool.read_table(os.path.join(out, "projection_types.csv"), PCS.TYPES_TYPES)
        assert [r["id"] for r in tab] == [r["id"] for r in rec["subimages"]]
        t = {r["id"]: r for r in tab}
        assert (t[501]["xy_candidate"], t[501]["darkest"], t[501]["chosen"], t[501]["best"]) == ("yes", "yes", "yes", "min")
        assert (t[502]["xy_candidate"], t[502]["darkest"], t[502]["chosen"], t[502]["best"]) == ("yes", "no", "no", "max")
        assert t[503]["xy_candidate"] == "no" and t[503]["verdict"] in ("", None) and math.isnan(t[503]["r"])
        png = os.path.join(out, "projection_check_%d.png" % SID)
        from PIL import Image
        assert os.path.getsize(png) > 10000 and Image.open(png).size[0] > 500
        assert any("matches min" in ln for ln in lines) and any("darkest at level 4: 501" in ln for ln in lines), lines
        assert any("chosen (by type): 501" in ln for ln in lines), lines


def _strip(path):
    rec = survey_io.read_json(path)
    rec.pop("seconds")
    return json.dumps(rec, sort_keys=True)


def test_determinism():
    with tempfile.TemporaryDirectory() as tmp:
        src = Source(tmp)
        a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
        run_quiet(small_cfg(), [SID], [SID], src, a)
        run_quiet(small_cfg(), [SID], [SID], src, b)
        name = "projections_%d.json" % SID
        assert _strip(os.path.join(a, name)) == _strip(os.path.join(b, name))
        with open(os.path.join(a, "projection_types.csv")) as fa, open(os.path.join(b, "projection_types.csv")) as fb:
            assert fa.read() == fb.read()


def test_edge_cases():
    stack = synthetic_stack()
    for bad in (stack[:1], ):
        try:
            PC.plane_operators(bad)
            raise AssertionError("one plane accepted")
        except ValueError:
            pass
    try:
        PC.plane_operators(stack, valid=np.arange(NPL) == 3)
        raise AssertionError("one valid plane accepted")
    except ValueError:
        pass
    ops, left, top = window(PC.plane_operators(stack))
    res = PC.check_projection(np.full((2 * H + 2 * S, 2 * H + 2 * S), 200.0), ops, S, 0.9, 4.0)
    assert res["best"] is None and res["verdict"] == "unclear: no finite r" and math.isnan(res["ratio"]) and all(
        math.isnan(v["r"]) and v["shift_x_px"] is None for v in res["by_operator"].values()), res
    # the window moved inside at the edges; an image too small refused
    assert PC.place_window(0.0, 0.0, H, S, WP, HP) == (S, S)
    assert PC.place_window(1e9, 1e9, H, S, WP, HP) == (WP - 2 * H - S, HP - 2 * H - S)
    for w, h in ((2 * (H + S) - 1, HP), (WP, 2 * (H + S) - 1)):
        try:
            PC.place_window(10.0, 10.0, H, S, w, h)
            raise AssertionError("a window larger than the image accepted")
        except ValueError:
            pass
    with tempfile.TemporaryDirectory() as tmp:
        cfg = small_cfg()
        # a missing section: white in fetch_zblock's block, excluded through valid
        src = Source(tmp, missing_section=11)
        rec, planes, cands = PCS.list_specimen(src, SID, cfg)
        ops_m, crops = PCS.check_specimen(src, SID, rec, planes, cands, cfg)
        assert rec["n_planes_used"] == NPL - 1 and float(ops_m["max"].max()) < 255.0, (rec["n_planes_used"],
                                                                                   ops_m["max"].max())
        # a failed plane: counted, not used
        src_f = Source(tmp)
        src_f.rows = [dict(r, failed=True) if r["id"] == 1005 else r for r in src_f.rows]
        rec_f, planes_f, _ = PCS.list_specimen(src_f, SID, cfg)
        assert (rec_f["planes"]["n_usable"], rec_f["planes"]["n_failed"]) == (NPL - 1, 1), rec_f["planes"]
        assert 1005 not in set(int(i) for i in planes_f["id"])
        # --check none: listed only
        out = os.path.join(tmp, "none")
        run_quiet(cfg, [SID], [], Source(tmp), out)
        rec = survey_io.read_json(os.path.join(out, "projections_%d.json" % SID))
        assert "checks" not in rec and "window" not in rec and rec["darkest"]["id"] == 501
        assert not os.path.exists(os.path.join(out, "projection_check_%d.png" % SID))
        # no xy candidate: listed and checked without failing
        out = os.path.join(tmp, "noxy")
        run_quiet(cfg, [SID], [SID], Source(tmp, with_xy=False), out)
        rec = survey_io.read_json(os.path.join(out, "projections_%d.json" % SID))
        assert rec["candidates"] == [] and rec["darkest"] is None and rec["checks"] == [] and "no xy" in rec["note"]
        assert rec["chosen"] is None
        # no soma and no --center: refused, naming --center
        try:
            run_quiet(cfg, [SID], [SID], Source(tmp, soma=False), os.path.join(tmp, "nosoma"))
            raise AssertionError("a check without a soma accepted")
        except ValueError as exc:
            assert "--center" in str(exc), exc
        # ... and with a given centre it runs
        recs, _ = run_quiet(cfg, [SID], [SID], Source(tmp, soma=False), os.path.join(tmp, "given"),
                            center=(WP / 2.0, HP / 2.0))
        assert recs[SID]["window"]["center_source"] == "given"
    assert PCS._int_list("none") == [] and PCS._int_list("1, 2") == [1, 2] and PCS._int_list("") == []
    try:
        PCS.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
    print(_environment())
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, lines[0] if lines else ""))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
