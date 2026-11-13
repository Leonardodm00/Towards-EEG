"""Spec-derived tests of Block 0 (the image modules; D-098.1's choice of the
SWC file) -- integration tester, run D1, 2026-11-13. Block 0's entries are
`Confirmed: inferred`; expected values come from SPEC Block 0, section 3's
image-block contract and hashlib, never from the code under test. No network:
`api_query` and `requests` are replaced for the duration of each test.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block0_image_io.py -q
"""
from __future__ import annotations

import hashlib
import itertools
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

import spec_helpers_D1 as H

import allen_image_io as aio

WS = H.WS
SWC = H.SWC_TYPE_ID
RECON = b"# reconstruction\n1 1 0.0 0.0 0.0 5.0 -1\n2 3 1.0 0.0 0.0 0.5 1\n"
MARK = b"##x,y,z,radius,shape\n1,2,3,4,5\n"


def wkf(i, name, t=None, folder="/external/mousecelltypes/prod%d"):
    f = {"id": i, "path": (folder % i) + "/" + name}
    if t is not None:
        f["well_known_file_type_id"] = t
    return f


# --------------------------------------------------------------- O0.1 -----

FIVE = [wkf(11, "c_marker_m.swc", 486584244), wkf(12, "c_m.swc", SWC),
        wkf(13, "c_m.png", 303941302), wkf(14, "c_other.swc"), wkf(15, "C_MARKER.swc", SWC)]


def test_choice_equals_the_rule_over_every_order():
    for perm in itertools.permutations(FIVE):
        got = aio.choose_swc_file(list(perm))
        exp = H.ref_choose_swc_file(list(perm))
        assert (got is None and exp is None) or got["id"] == exp["id"], (perm, got, exp)
        assert got["id"] == 12  # the typed reconstruction, whatever the order


@pytest.mark.parametrize("files,expected", [
    # the type decides before the suffix, the type id given as text
    ([wkf(1, "a.swc"), wkf(2, "b_m.swc", str(SWC))], 2),
    # no typed file: the first .swc path, markers dropped in any case
    ([wkf(1, "x_Marker_m.swc"), wkf(2, "x.png"), wkf(3, "x_m.swc"), wkf(4, "y.swc")], 3),
    # only the basename is tested for "marker": a folder named so is no marker
    ([wkf(7, "r_m.swc", SWC, folder="/external/marker_cells/prod%d")], 7),
    # a typed file without a .swc path is taken
    ([wkf(5, "recon.txt", SWC)], 5),
    # nothing usable
    ([wkf(1, "a_marker.swc", SWC), wkf(2, "b.png")], None),
    ([], None),
])
def test_choice_cases(files, expected):
    got = aio.choose_swc_file(files)
    exp = H.ref_choose_swc_file(files)
    assert (exp is None and expected is None) or exp["id"] == expected  # the reference itself
    if expected is None:
        assert got is None
    else:
        assert got["id"] == expected


# --------------------------------------------------------------- O0.2 -----

class _Resp:
    def __init__(self, content):
        self.content = content
        self.status_code = 200

    def raise_for_status(self):
        return None


class _FakeRequests(types.ModuleType):
    def __init__(self, contents):
        super().__init__("requests")
        self.contents = contents
        self.urls = []

    def get(self, url, *a, **k):
        self.urls.append(url)
        return _Resp(self.contents[int(url.rstrip("/").split("/")[-1])])


@pytest.fixture
def served(monkeypatch):
    def _serve(files, reply=True):
        contents = {int(f["id"]): (MARK if "marker" in f["path"].lower() else RECON) for f in files}
        fake = _FakeRequests(contents)

        def api_query(criteria, num_rows=200, start_row=0, timeout=60):
            return [dict(id=1, specimen_id=1, well_known_files=list(files))] if reply else []

        monkeypatch.setattr(aio, "api_query", api_query)
        monkeypatch.setitem(sys.modules, "requests", fake)
        return fake
    return _serve


def test_fetch_swc_downloads_the_chosen_file_and_reuses_the_cache(served, tmp_path):
    files = [wkf(599474140, "H_599474134_marker_m.swc", 486584244),
             wkf(599474136, "H_599474134_m.swc", SWC)]
    fake = served(files)
    p = aio.fetch_swc(569095789, str(tmp_path))
    assert os.path.dirname(os.path.abspath(p)) == os.path.abspath(str(tmp_path))
    assert os.path.basename(p) == "H_599474134_m.swc"
    assert len(fake.urls) == 1 and fake.urls[0].endswith("/599474136")
    with open(p, "rb") as fh:
        assert fh.read() == RECON
    # cached: no new request
    p2 = aio.fetch_swc(569095789, str(tmp_path))
    assert p2 == p and len(fake.urls) == 1
    # overwrite fetches again
    aio.fetch_swc(569095789, str(tmp_path), overwrite=True)
    assert len(fake.urls) == 2


@pytest.mark.parametrize("files,reply", [
    ([wkf(1, "a_marker_m.swc", SWC), wkf(2, "a_MARKER.swc")], True),
    ([wkf(3, "a.png", 303941302)], True),
    ([], True),
    ([wkf(4, "a.swc", SWC)], False),
])
def test_fetch_swc_refuses_without_a_reconstruction(served, tmp_path, files, reply):
    served(files, reply=reply)
    with pytest.raises(RuntimeError) as ei:
        aio.fetch_swc(1, str(tmp_path))
    if reply:
        for f in files:   # "RuntimeError naming every file"
            assert os.path.basename(f["path"]) in str(ei.value) or str(f["id"]) in str(ei.value)


def test_cached_marker_is_never_returned(served, tmp_path):
    files = [wkf(21, "z_marker_m.swc", SWC), wkf(22, "z_m.swc", SWC)]
    (tmp_path / "z_marker_m.swc").write_bytes(MARK)
    served(files)
    p = aio.fetch_swc(2, str(tmp_path))
    assert os.path.basename(p) == "z_m.swc"


# --------------------------------------------------------------- O0.3 -----

@pytest.mark.parametrize("rel,prefix", [
    ("src/allen_image_io.py", "44f6fbd01369"),
    ("src/allen_image_align.py", "14dd899057a8"),
    ("src/allen_image_measure.py", "4614cd949ba1"),
    ("src/allen_image_plot.py", "3b9f1a07e544"),
    ("tests/smoke/smoke_allen_image.py", "d7811c681b0f"),
    ("tests/smoke/robustness_registration.py", "18059c7109ee"),
])
def test_sha256_prefixes(rel, prefix):
    data = (WS / rel).read_bytes()
    assert hashlib.sha256(data).hexdigest()[:12] == prefix


# --------------------------------------------------------------- O0.4 -----

class _Planes:
    """A fetcher serving one full-resolution image per plane id."""

    def __init__(self, images):
        self.images = images
        self.calls = []

    def get(self, image_id, left, top, width, height, downsample=0):
        self.calls.append((image_id, left, top, width, height, downsample))
        return self.images[int(image_id)][top:top + height, left:left + width].copy()


def test_fetch_zblock_contract_and_missing_planes():
    rng = np.random.default_rng(1)
    imgs = {k: rng.integers(0, 200, size=(60, 80), dtype=np.uint8) for k in (3, 4, 6)}
    planes = pd.DataFrame({"plane_index": [3, 4, 6], "id": [3, 4, 6]})
    block, ks, valid, frame = aio.fetch_zblock(_Planes(imgs), planes, 3, 7, 10, 5, 30, 20, 0.1144)
    assert block.dtype == np.uint8 and block.shape == (5, 20, 30)
    assert list(np.asarray(ks)) == [3, 4, 5, 6, 7]
    assert list(np.asarray(valid, dtype=bool)) == [True, True, False, True, False]
    for j, k in enumerate([3, 4, 5, 6, 7]):
        if k in imgs:
            np.testing.assert_array_equal(block[j], imgs[k][5:25, 10:40])
        else:
            assert np.all(block[j] == 255)  # missing planes white
    assert (frame.left, frame.top, frame.downsample) == (10, 5, 0)
    assert frame.res0_um_px == 0.1144
