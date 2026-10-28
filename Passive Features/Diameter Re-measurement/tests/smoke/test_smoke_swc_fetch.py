"""Smoke test for allen_image_io.fetch_swc's choice of file -- Block 0 in
specs/SPEC.md, amended by D-098.1 (2026-10-28): the reconstruction, never
Allen's marker file.

In Colab on 2026-10-28 four of the 51 cells of the S3 selection failed their
alignment because fetch_swc took the first well_known_file whose path ends in
.swc, and those specimens list a marker file named *_marker_m.swc first; its
comma-separated lines are not an SWC. No network here: api_query is replaced
by a canned NeuronReconstruction reply and `requests` by a fake that serves
the files' bytes by well_known_file id.

Checks
    test_known_answer   the four cells' pattern (the marker first, with a .swc
                        path; the reconstruction second, of the SWC type):
                        the reconstruction is downloaded from its own id,
                        written under its own name, and read_swc parses it;
                        the type id given as a string works too
    test_reference      choose_swc_file against an independent statement of
                        the rule, over every order of five files
    test_convergence    not a discretisation: skipped
    test_invariants     the order of the files never changes the choice when
                        one reconstruction is listed; "MARKER" in any case is
                        excluded
    test_contract       the path returned lies in out_dir under the
                        reconstruction's name; a cached reconstruction is
                        reused without a request, overwrite=True fetches it
                        again; the marker's own lines fail read_swc (why the
                        old choice failed)
    test_determinism    two fetches into two folders give identical bytes
    test_edge_cases     only marker files, no files, no reconstruction: a
                        RuntimeError naming the files; a typed file without
                        a .swc path is taken; a .swc path without a type is
                        taken; a cached marker file is never returned

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_swc_fetch.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import contextlib
import importlib.metadata
import itertools
import os
import platform
import sys
import tempfile
import time
import traceback
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for _p in (SRC, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import allen_image_io as aio  # noqa: E402
from allen_diameter.loading import swc_io  # noqa: E402

SEED = 20261028
REPORT_PACKAGES = ("numpy", "pandas")
SWC_TYPE = aio.SWC_FILE_TYPE_ID
MARKER_TYPE = 486584244          # any id other than the SWC type: the marker's own type is not relied on

RECON_TEXT = ("# a reconstruction\n"
              "1 1 0.0 0.0 0.0 4.0 -1\n"
              "2 3 6.0 0.0 0.0 0.3 1\n"
              "3 3 7.2 0.1 0.0 0.3 2\n"
              "4 3 8.4 0.2 0.0 0.3 3\n")
MARKER_TEXT = ("##x,y,z,radius,shape,name,comment, color_r,color_g,color_b\n"
               "558.1164,321.2352,1.4,0.0,1,Marker,0,255,0,0\n"
               "570.0011,330.4100,2.8,0.0,1,Marker,0,255,0,0\n")


# --------------------------------------------------------------- helpers ---

def wkf(i, path, type_id):
    f = dict(id=i, path="/external/mousecelltypes/prod%d/%s" % (i, path))
    if type_id is not None:
        f["well_known_file_type_id"] = type_id
    return f


def four_cells_files():
    """The pattern of 569095789, 508279351, 596036453, 525018757: the marker first, then the reconstruction."""
    return [wkf(599474140, "H17.06.004.11.05.04_599474134_marker_m.swc", MARKER_TYPE),
            wkf(599474136, "H17.06.004.11.05.04_599474134_m.swc", SWC_TYPE),
            wkf(599474138, "H17.06.004.11.05.04_599474134_m.png", 303941302)]


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None


class FakeRequests(types.ModuleType):
    """requests.get(url, timeout) serving the bytes of each well_known_file id; the URLs are kept."""

    def __init__(self, contents):
        super().__init__("requests")
        self.contents, self.urls = contents, []

    def get(self, url, timeout=None):
        self.urls.append(url)
        return FakeResponse(self.contents[int(url.rstrip("/").split("/")[-1])])


@contextlib.contextmanager
def served(files, contents, reply=True):
    """aio.api_query -> one NeuronReconstruction holding `files` (none when reply is False); requests -> fake."""
    fake = FakeRequests(contents)
    old_query, old_requests = aio.api_query, sys.modules.get("requests")

    def api_query(criteria, num_rows=200, start_row=0, timeout=60):
        assert "NeuronReconstruction" in criteria and "well_known_files" in criteria, criteria
        return [dict(id=1, specimen_id=1, well_known_files=list(files))] if reply else []
    aio.api_query = api_query
    sys.modules["requests"] = fake
    try:
        yield fake
    finally:
        aio.api_query = old_query
        if old_requests is None:
            sys.modules.pop("requests", None)
        else:
            sys.modules["requests"] = old_requests


def contents_for(files):
    out = {}
    for f in files:
        name = os.path.basename(f["path"])
        out[int(f["id"])] = (MARKER_TEXT if "marker" in name.lower() else RECON_TEXT).encode("ascii")
    return out


def rule(files):
    """The rule written out independently: drop the marker files; the first SWC-typed one, else the first .swc."""
    keep = []
    for f in files:
        base = f["path"].split("/")[-1]
        if base.lower().find("marker") >= 0:
            continue
        keep.append(f)
    for f in keep:
        t = f.get("well_known_file_type_id")
        if t is not None and str(t).strip() == str(SWC_TYPE):
            return f
    for f in keep:
        if f["path"].lower()[-4:] == ".swc":
            return f
    return None


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    files = four_cells_files()
    with tempfile.TemporaryDirectory() as tmp, served(files, contents_for(files)) as fake:
        path = aio.fetch_swc(569095789, tmp)
        assert os.path.basename(path) == "H17.06.004.11.05.04_599474134_m.swc", path
        assert fake.urls == ["%s/well_known_file_download/%d" % (aio.API, 599474136)], fake.urls
        with open(path, "rb") as fh:
            assert fh.read() == RECON_TEXT.encode("ascii")
        swc = swc_io.read_swc(path)
        assert len(swc.ids) == 4 and list(swc.ids) == [1, 2, 3, 4], swc.ids
    as_text = [dict(f, well_known_file_type_id=str(f["well_known_file_type_id"])) for f in files]
    assert aio.choose_swc_file(as_text)["id"] == 599474136
    # the type decides before the suffix, also when the API writes it as text: an untyped .swc listed first loses
    assert aio.choose_swc_file([wkf(71, "h_other.swc", None), wkf(72, "h_m.swc", str(SWC_TYPE))])["id"] == 72
    # the rule before D-098.1 (the first path ending in .swc) took the marker here
    old = [f for f in files if f["path"].lower().endswith(".swc")][0]
    assert old["id"] == 599474140 and aio.choose_swc_file(files)["id"] != old["id"]


def test_reference():
    base = [wkf(11, "a_marker_m.swc", MARKER_TYPE), wkf(12, "a_m.swc", SWC_TYPE), wkf(13, "a_m.png", 303941302),
            wkf(14, "a.marker", None), wkf(15, "b_m.swc", None)]
    n = 0
    for perm in itertools.permutations(base):
        got, want = aio.choose_swc_file(list(perm)), rule(list(perm))
        assert got is want, ([f["id"] for f in perm], got and got["id"], want and want["id"])
        assert got["id"] == 12          # the typed reconstruction whatever the order
        n += 1
    assert n == 120
    untyped = [wkf(21, "c_marker_m.swc", None), wkf(22, "c_m.swc", None), wkf(23, "d_m.swc", None)]
    for perm in itertools.permutations(untyped):
        assert aio.choose_swc_file(list(perm)) is rule(list(perm))


def test_convergence():
    raise unittest.SkipTest("a choice among files: no discretisation, nothing converges")


def test_invariants():
    files = four_cells_files()
    for perm in itertools.permutations(files):
        assert aio.choose_swc_file(list(perm))["id"] == 599474136
    for name in ("x_MARKER_m.swc", "x_Marker_m.swc", "x_marker.swc", "MARKER.SWC"):
        assert aio.choose_swc_file([wkf(31, name, SWC_TYPE)]) is None, name


def test_contract():
    files = four_cells_files()
    with tempfile.TemporaryDirectory() as tmp:
        with served(files, contents_for(files)) as fake:
            path = aio.fetch_swc(508279351, tmp)
            assert os.path.dirname(os.path.abspath(path)) == os.path.abspath(tmp) and len(fake.urls) == 1
        with served(files, contents_for(files)) as fake:
            again = aio.fetch_swc(508279351, tmp)              # cached: no request
            assert again == path and fake.urls == [], fake.urls
        with served(files, contents_for(files)) as fake:
            aio.fetch_swc(508279351, tmp, overwrite=True)      # overwrite: fetched again
            assert len(fake.urls) == 1, fake.urls
        marker = os.path.join(tmp, "marker_m.swc")
        with open(marker, "w") as fh:
            fh.write(MARKER_TEXT)
        try:
            swc_io.read_swc(marker)
        except ValueError:
            pass
        else:
            raise AssertionError("the marker's comma-separated lines parsed as an SWC")


def test_determinism():
    files = four_cells_files()
    got = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp, served(files, contents_for(files)):
            with open(aio.fetch_swc(596036453, tmp), "rb") as fh:
                got.append(fh.read())
    assert got[0] == got[1]


def test_edge_cases():
    only_markers = [wkf(41, "e_marker_m.swc", SWC_TYPE), wkf(42, "e.marker", None), wkf(43, "e_m.png", 303941302)]
    for files, reply in ((only_markers, True), ([], True), (four_cells_files(), False)):
        with tempfile.TemporaryDirectory() as tmp, served(files, contents_for(files), reply=reply) as fake:
            try:
                aio.fetch_swc(525018757, tmp)
            except RuntimeError as exc:
                msg = str(exc)
                assert fake.urls == [], fake.urls
                if reply:
                    assert all(os.path.basename(f["path"]) in msg for f in files), msg
            else:
                raise AssertionError("no error for %r" % [f["path"] for f in files])
    typed_no_suffix = [wkf(51, "f_marker_m.swc", None), wkf(52, "f_reconstruction.txt", SWC_TYPE)]
    assert aio.choose_swc_file(typed_no_suffix)["id"] == 52
    suffix_no_type = [wkf(61, "g_marker_m.swc", None), wkf(62, "g_m.png", None), wkf(63, "g_m.swc", None)]
    assert aio.choose_swc_file(suffix_no_type)["id"] == 63
    files = four_cells_files()
    with tempfile.TemporaryDirectory() as tmp:
        cached_marker = os.path.join(tmp, "H17.06.004.11.05.04_599474134_marker_m.swc")
        with open(cached_marker, "w") as fh:
            fh.write(MARKER_TEXT)                     # what the run before D-098.1 left in the cache
        with served(files, contents_for(files)) as fake:
            path = aio.fetch_swc(569095789, tmp)
        assert path != cached_marker and len(fake.urls) == 1 and swc_io.read_swc(path).ids.size == 4


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
