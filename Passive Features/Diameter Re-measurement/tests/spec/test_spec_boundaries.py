"""Block boundaries (procedure step 7): what a producer returns against what its consumer expects.
One rendered replicate (about 10-30 s on one core) feeds the table, the inversion and the CSV row."""
import dataclasses
import math

import numpy as np
import pytest

from allen_diameter.config import default_config
from allen_diameter.analysis import cell, invert, phantoms as PH, table as TB
from allen_diameter.loading import table_io

BASE = default_config()
CFG = dataclasses.replace(BASE, phantom=dataclasses.replace(BASE.phantom, d_range_um=(0.6, 0.7),
                                                            phi_range_deg=(0.0, 5.0), mu_range_per_um=(0.5, 0.6)))


@pytest.fixture(scope="module")
def row():
    return PH.run_replicate(CFG, 99, 0)


def test_replicate_row_feeds_fit_table(row, tmp_path):
    for key in ("d_um", "phi_rad", "meas_phi_rad", "ratio", "in_S", "reject", "seed", "index", "flags"):
        assert key in row
    assert row["ratio"] == pytest.approx(row["d_hat_um"] / row["d_um"], rel=1e-15)
    assert isinstance(row["in_S"], bool)
    # units: true tilt in rad within the configured degree range; measured tilt rad in [0, pi/2]
    assert 0 <= row["phi_rad"] < math.radians(5.0) and 0 <= row["meas_phi_rad"] <= math.pi / 2
    # phantom planes are numbered about 0 (z_k = k dz, axis depth within dz/2 of plane 0)
    assert abs(row["k_star"]) <= 2
    # CSV round trip keeps types the table needs
    p = str(tmp_path / "r.csv")
    table_io.write_rows([row], p)
    back = table_io.read_rows(p)[0]
    assert back["in_S"] is row["in_S"] and back["ratio"] == pytest.approx(row["ratio"], rel=1e-15)
    assert back["reject"] == row["reject"] or (back["reject"] == "" and row["reject"] == "")


def test_table_axis_units_shared_by_invert():
    rng = np.random.default_rng(0)
    d = np.exp(rng.uniform(math.log(0.2), math.log(4.0), 300))
    phi = np.radians(rng.uniform(0, 90, 300))
    rows = [dict(d_um=d[i], phi_rad=phi[i], ratio=1.1, in_S=True, reject="") for i in range(300)]
    T = TB.fit_table(rows, BASE)
    assert np.allclose(T.X_kept[:, 0], np.log(d)) and np.allclose(T.X_kept[:, 1], phi)
    inv = invert.invert_node(1.1, math.radians(30), T, BASE)
    assert inv.d_tilde_um == pytest.approx(1.0, rel=1e-3)


def test_node_row_columns(row):
    # a NodeResult rebuilt from a measured phantom feeds node_row; columns in Block 8 order
    draw, rng = PH.draw_replicate(CFG, 99, 0)
    from allen_diameter.analysis.node_pipeline import measure_node
    from allen_diameter.model import render
    tube = PH.phantom_tube(draw, CFG)
    br = PH.phantom_branch(draw, CFG, rng)

    def prov(left, top, width, height, k_lo, k_hi):
        return render.synthetic_block(tube, draw.mu_per_um, np.arange(k_lo, k_hi + 1), left, top, width, height,
                                      CFG, rng, int(math.ceil(CFG.renderer.pad_um / CFG.acquisition.res0_um)))
    res = measure_node(br, CFG.phantom.nodes_each_way, prov, CFG)
    assert res.d_hat_um == row["d_hat_um"]                       # same seed, same replicate: bit-identical
    inv = invert.Inversion(float("nan"), float("nan"), float("nan"), ("out_of_domain",))
    r = cell.node_row(res, inv, 1.0, "allen", 0.3, CFG.measure.sigma_fit_um)
    assert list(r) == list(cell.CSV_COLUMNS)
    assert r["flags"].split(";")[-1] == "out_of_domain"
