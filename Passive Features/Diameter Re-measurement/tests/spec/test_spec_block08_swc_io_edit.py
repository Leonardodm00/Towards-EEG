"""Spec-derived tests, Block 8 amendment (2026-11-22; D-222.5; wave 6) and Block 47's SWC editor -- tester 2026-11-23.

Oracles (SPEC Block 8, amendment; Block 47 "Public API" and T5; D-013):
  - write_swc with radius alone: the bytes of the file with only token 6 of each data line replaced (the token's own
    decimals, at least MIN_DECIMALS = 4); with nothing given: the input bytes [exact];
  - edited_lines(swc, xyz, radius, parent, keep): the data lines of the kept rows with x, y, z (tokens 3-5), the radius
    (6) and the parent (7) replaced where given -- floats with the token's decimals (>= 4), the parent as an integer --,
    the removed rows' lines dropped, every other line verbatim (comments, blank lines, separators, line endings) [exact,
    against an independent tokeniser];
  - a field given is rewritten on every kept line, so Allen's 4-decimal tokens of unchanged values come back byte for byte;
  - an edit round-trips through read_swc (the new xyz to the written decimals, radius, parents; removed ids absent);
  - ValueError for shapes that disagree, a non-finite coordinate, a radius <= 0, a kept row whose parent is removed, a
    parent id absent from the kept rows (-1 allowed);
  - no mutation of the SWC object; read_swc's arrays unchanged by the amendment (Block 8's data contract, section 3).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import soma_spec_helpers as H  # noqa: F401  (paths)
from allen_diameter.loading import swc_io

# A file with every layout the writer must keep: a header, a blank line, tabs and double spaces, CRLF on one line,
# trailing spaces, 2 and 6 decimals, integer tokens, an inline comment after the 7 fields, no final newline.
TEXT = ("# tester's cell\n"
        "#  second comment line\r\n"
        "1 1 0.0000 0.0000 0.0000 5.0000 -1\n"
        "\n"
        "2\t3  1.2345\t2.3456 3.4567  0.5000 1\r\n"
        "3 3 2.00 3.00 4.00 0.25 2   \n"
        "4 3 1.123456 2.123456 3.123456 0.123456 3\n"
        "5 4 7 8 9 1 4 # inline comment\n"
        "6 3 1.0 2.0 3.0 0.5 5")


@pytest.fixture()
def swc(tmp_path):
    p = tmp_path / "cell.swc"
    p.write_bytes(TEXT.encode("ascii"))
    return swc_io.read_swc(str(p))


def test_read_swc_contract_unchanged(swc):
    # section 3: ids, types, parent (N,) int; xyz (N, 3), radius (N,) float64; comments kept verbatim
    assert_array_equal(swc.ids, [1, 2, 3, 4, 5, 6])
    assert_array_equal(swc.types, [1, 3, 3, 3, 4, 3])
    assert_array_equal(swc.parent, [-1, 1, 2, 3, 4, 5])
    assert swc.xyz.shape == (6, 3) and swc.xyz.dtype == np.float64 and swc.radius.dtype == np.float64
    assert_array_equal(swc.xyz[3], [1.123456, 2.123456, 3.123456])
    assert_array_equal(swc.radius, [5.0, 0.5, 0.25, 0.123456, 1.0, 0.5])
    assert "".join(swc.lines) == TEXT


def test_write_without_edits_is_byte_identical(swc, tmp_path):
    out = tmp_path / "o.swc"
    swc_io.write_swc(swc, str(out))
    assert out.read_bytes() == TEXT.encode("ascii")


def test_radius_alone_changes_only_token_6(swc, tmp_path):
    r = np.array([4.5, 0.61, 0.333333, 0.2, 1.25, 0.75])
    exp = H.ref_edit_text(TEXT, radius=r)
    out = tmp_path / "o.swc"
    swc_io.write_swc(swc, str(out), radius=r)
    assert out.read_bytes().decode("ascii") == exp
    # edited_lines with radius alone gives the same lines (Block 47: "with radius alone the bytes of
    # rewrite_radius_lines, as before")
    assert "".join(swc_io.edited_lines(swc, radius=r)) == exp
    # the decimals: 2-decimal token -> 4 (MIN_DECIMALS), 6-decimal token kept at 6
    lines = out.read_text().splitlines()
    assert lines[5].split()[5] == "0.3333" and lines[6].split()[5] == "0.200000"


def test_edit_all_fields_against_independent_tokeniser(swc):
    xyz = swc.xyz.copy()
    xyz[1] = [10.5, -3.25, 0.000049]        # moved node: -> 10.5000 -3.2500 0.0000
    xyz[5] = [1.23456789, 2.0, -7.5]        # 1-decimal tokens: 4 decimals
    radius = swc.radius.copy()
    radius[0] = 3.14159
    parent = swc.parent.copy()
    keep = np.ones(6, dtype=bool)
    keep[3] = False                          # node 4 removed
    parent[4] = 3                            # its child (node 5) re-parented to node 3
    got = "".join(swc_io.edited_lines(swc, xyz=xyz, radius=radius, parent=parent, keep=keep))
    exp = H.ref_edit_text(TEXT, xyz=xyz, radius=radius, parent=parent, keep=keep)
    assert got == exp, (got, exp)
    # spot checks written by hand
    assert "2\t3  10.5000\t-3.2500 0.0000  0.5000 1\r\n" in got
    assert "5 4 7.0000 8.0000 9.0000 1.0000 3 # inline comment\n" in got
    assert got.endswith("6 3 1.2346 2.0000 -7.5000 0.5000 5")            # no newline added
    assert "1.123456" not in got                                          # node 4's line dropped
    assert got.startswith("# tester's cell\n#  second comment line\r\n")


def test_allen_four_decimals_unchanged_values_kept_byte_for_byte(tmp_path):
    rows = [(1, 1, 412.3456, 388.1234, 25.0001, 7.1234, -1), (2, 3, 415.0, 390.5, 26.25, 0.8765, 1),
            (3, 3, 418.9999, 391.0001, 26.5, 0.6543, 2), (4, 2, 409.0, 385.25, 24.75, 0.3, 1)]
    text = H.make_swc_text(rows)
    p = tmp_path / "a.swc"
    p.write_text(text)
    s = swc_io.read_swc(str(p))
    got = "".join(swc_io.edited_lines(s, xyz=s.xyz, radius=s.radius, parent=s.parent,
                                      keep=np.ones(len(s), dtype=bool)))
    assert got == text


def test_round_trip_through_read_swc(swc, tmp_path):
    xyz = swc.xyz.copy()
    xyz[2] = [-1.23456, 7.65432, 0.5]
    parent = swc.parent.copy()
    keep = np.array([True, True, True, False, True, True])
    parent[4] = 3
    radius = swc.radius.copy()
    radius[2] = 0.987654
    out = tmp_path / "e.swc"
    swc_io.write_swc(swc, str(out), xyz=xyz, radius=radius, parent=parent, keep=keep)
    back = swc_io.read_swc(str(out))
    assert_array_equal(back.ids, [1, 2, 3, 5, 6])                         # node 4 absent
    assert_array_equal(back.parent, [-1, 1, 2, 3, 5])
    # node 3's tokens had 2 decimals -> 4 written: the values to 0.5e-4 (the written precision)
    np.testing.assert_allclose(back.xyz[2], [-1.2346, 7.6543, 0.5], rtol=0, atol=0.5e-4 + 1e-12)
    np.testing.assert_allclose(back.radius[2], 0.9877, rtol=0, atol=1e-12)
    # every row not edited keeps its values exactly
    for k_in, k_out in ((0, 0), (1, 1), (4, 3), (5, 4)):
        assert_array_equal(back.xyz[k_out], np.round(swc.xyz[k_in], 4))


@pytest.mark.parametrize("kw, what", [
    (dict(xyz=np.zeros((6, 2))), "xyz shape"),
    (dict(xyz=np.zeros((5, 3))), "xyz rows"),
    (dict(radius=np.ones(5)), "radius shape"),
    (dict(parent=np.zeros(7, dtype=int)), "parent shape"),
    (dict(xyz=np.where(np.arange(18).reshape(6, 3) == 4, np.nan, 1.0)), "non-finite coordinate"),
    (dict(xyz=np.where(np.arange(18).reshape(6, 3) == 4, np.inf, 1.0)), "infinite coordinate"),
    (dict(radius=np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])), "radius 0"),
    (dict(radius=np.array([1.0, 1.0, -0.1, 1.0, 1.0, 1.0])), "radius < 0"),
    (dict(radius=np.array([1.0, 1.0, np.nan, 1.0, 1.0, 1.0])), "radius NaN"),
    (dict(keep=np.array([True, True, False, True, True, True])), "kept row whose parent is removed"),
    (dict(parent=np.array([-1, 1, 2, 3, 99, 5])), "parent id absent"),
    (dict(parent=np.array([-1, 1, 2, 3, 4, 4]), keep=np.array([True, True, True, False, True, True])),
     "kept rows 5 and 6 under a removed parent"),
])
def test_refusals(swc, kw, what):
    with pytest.raises(ValueError):
        swc_io.edited_lines(swc, **kw)


def test_minus_one_parent_allowed_and_no_mutation(swc):
    snap = copy.deepcopy(swc)
    parent = swc.parent.copy()
    parent[5] = -1                       # a second root is allowed (-1)
    lines = swc_io.edited_lines(swc, parent=parent)
    assert lines[-1] == "6 3 1.0 2.0 3.0 0.5 -1"
    for name in ("ids", "types", "xyz", "radius", "parent"):
        assert_array_equal(getattr(swc, name), getattr(snap, name))
    assert swc.lines == snap.lines and swc.data_rows == snap.data_rows
