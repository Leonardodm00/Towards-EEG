"""Block 8 (SWC I/O, stretches, membrane area, CSV columns): oracles from SPEC.md Block 8 and section 3."""
import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from allen_diameter.loading import swc_io
from allen_diameter.analysis import cell

# soma 1; basal 2-3-4 then branch at 4 into (5, 6) and (7); apical 8-9 from the soma; axon 10 from soma
TEXT = ("# header comment\r\n"
        "#   id type x y z r parent\r\n"
        "1 1 0.0 0.0 0.0 5.0 -1\r\n"
        "2 3 0.0 10.0 0.0 1.0 1\r\n"
        "3\t3   0.0 13.0 0.0 0.80 2\r\n"
        "4 3 0.0 17.0 0.0 0.6000 3\r\n"
        "5 3 3.0 21.0 0.0 0.5 4\r\n"
        "6 3 6.0 25.0 0.0 0.5 5\r\n"
        "7 3 -3.0 21.0 0.0 0.4 4\r\n"
        "8 4 0.0 -10.0 0.0 2.0 1\r\n"
        "9 4 0.0 -14.0 3.0 1.5 8\r\n"
        "10 2 10.0 0.0 0.0 0.3 1\r\n"
        "\r\n")


@pytest.fixture
def swc_path(tmp_path):
    p = tmp_path / "c.swc"
    p.write_bytes(TEXT.encode("ascii"))
    return str(p)


def test_parse_and_byte_identity(swc_path, tmp_path):
    s = swc_io.read_swc(swc_path)
    assert list(s.ids) == list(range(1, 11)) and list(s.types) == [1, 3, 3, 3, 3, 3, 3, 4, 4, 2]
    assert_allclose(s.xyz[8], [0, -14, 3]) and s.radius[2] == 0.8 and s.parent[0] == -1
    out = tmp_path / "o.swc"
    swc_io.write_swc(s, str(out))
    assert out.read_bytes() == TEXT.encode("ascii")
    new = s.radius * 1.5
    swc_io.write_swc(s, str(out), new)
    a, b = TEXT.encode().split(b"\r\n"), out.read_bytes().split(b"\r\n")
    assert len(a) == len(b)
    for la, lb in zip(a, b):
        ta, tb = la.split(), lb.split()
        if la.startswith(b"#") or not ta:
            assert la == lb
            continue
        assert ta[:5] == tb[:5] and ta[6:] == tb[6:]
        dec = max(len(ta[5].split(b".")[1]), 4)
        assert len(tb[5].split(b".")[1]) == dec
        # separators preserved: line with the token replaced
        assert lb == la.replace(ta[5], tb[5], 1)
    assert_allclose(swc_io.read_swc(str(out)).radius, new, rtol=1e-4)


def test_refusals(tmp_path, swc_path):
    p = tmp_path / "bad.swc"
    p.write_text("1 1 0 0 0 1\n")
    with pytest.raises(ValueError):
        swc_io.read_swc(str(p))
    s = swc_io.read_swc(swc_path)
    for bad in (np.zeros(10), np.full(10, np.nan), np.ones(9)):
        with pytest.raises(ValueError):
            swc_io.rewrite_radius_lines(s, bad)


def test_frustum_and_dendrite_area(swc_path):
    s = swc_io.read_swc(swc_path)
    fa = swc_io.frustum_areas_um2(s)
    # node 3 to 2: frustum r 0.8, 1.0, h 3
    assert fa[2] == pytest.approx(math.pi * 1.8 * math.sqrt(9 + 0.04), rel=1e-12)
    r, p = s.radius, s.parent_index()
    ref = 0.0
    for i in range(10):
        if s.types[i] not in (3, 4) or p[i] < 0:
            continue
        h = np.linalg.norm(s.xyz[i] - s.xyz[p[i]])
        rp = r[p[i]] if s.types[p[i]] in (3, 4) else r[i]          # soma link: cylinder of own radius
        ref += math.pi * (r[i] + rp) * math.sqrt(h * h + (r[i] - rp) ** 2)
    assert cell.dendrite_area_um2(s) == pytest.approx(ref, rel=1e-12)
    # cylinders only double exactly; here: ratio of the hand formula at 2 r and at r
    r2, ref2 = 2 * r, 0.0
    for i in range(10):
        if s.types[i] not in (3, 4) or p[i] < 0:
            continue
        h = np.linalg.norm(s.xyz[i] - s.xyz[p[i]])
        rp = r2[p[i]] if s.types[p[i]] in (3, 4) else r2[i]
        ref2 += math.pi * (r2[i] + rp) * math.sqrt(h * h + (r2[i] - rp) ** 2)
    assert cell.area_ratio(s, r2) == pytest.approx(ref2 / ref, rel=1e-12)


def test_stretches(swc_path):
    s = swc_io.read_swc(swc_path)
    got = sorted([list(s.ids[r]) for r in cell.stretches(s)])
    assert got == sorted([[2, 3, 4], [5, 6], [7], [8, 9]])
    dend = set(int(i) for i in s.ids[np.isin(s.types, (3, 4))])
    flat = [int(i) for r in cell.stretches(s) for i in s.ids[r]]
    assert sorted(flat) == sorted(dend)                  # partition
    xyz = s.xyz.copy()
    br, off = cell.stretch_branch(s, cell.stretches(s)[[list(s.ids[r]) for r in cell.stretches(s)].index([7])], xyz)
    assert list(br.ids) == [4, 7] and off == 1            # one-node stretch takes its parent
    br, off = cell.stretch_branch(s, np.array([1, 2, 3]), xyz)   # starts at a soma child: nothing prepended
    assert list(br.ids) == [2, 3, 4] and off == 0
    br, off = cell.stretch_branch(s, np.array([1]), xyz)         # a one-node stretch takes its parent (soma)
    assert list(br.ids) == [1, 2] and off == 1


def test_to_image_um_matches_alignment_module():
    from allen_image_align import swc_to_full_px
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 500, (20, 3))
    df = pd.DataFrame(dict(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2]))
    for flip in (None, 9000.0):
        x, y = swc_to_full_px(df, 0.1144, (12.5, -7.0), flip)
        got = cell.to_image_um(xyz, 0.1144, (12.5, -7.0), flip, 3.0)
        assert_allclose(got[:, 0], x * 0.1144, rtol=1e-14)
        assert_allclose(got[:, 1], y * 0.1144, rtol=1e-14)
        assert_allclose(got[:, 2], xyz[:, 2] - 3.0, rtol=1e-14)


def test_csv_columns():
    spec = ("node_id, type, x_um, y_um, z_um, path_um, reg_verdict, s_star_um, dz_star_um, k_star, z_sub_um, cx_um, "
            "cy_um, cz_um, theta_rad, phi_rad, steep, vertical, B_bar, B_bar_region, d_hat_um, mu_hat_per_um, "
            "v0_hat_um, alpha_hat, fit_status, b_hat, d_tilde_um, flags, filled_from, d_final_um, allen_radius_um, "
            "sigma_fit_um, d_tilde_sigma_spread_um").split(", ")
    assert list(cell.CSV_COLUMNS) == spec
