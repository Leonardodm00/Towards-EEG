"""Smoke test for soma_enforce (stage S1.2).

Runs standalone (python3 test_soma_enforce.py) or under pytest. Areas are
recomputed here from first principles (4*pi*r^2 and the exporter's two-point
z-extent convention pi*diam*L) rather than taken from the module.

Coverage
  S1  geometry test passes on an intact cell, fails on the measured fragment
  S2  I-18: agreement -> pass ; disagreement -> pass_low_confidence (SOFT, Q6)
  S3  multi-node collapse conserves sphere area exactly and reparents children
  S4  soma_area_um2 equals BOTH 4*pi*r^2 and the emitted section's pi*diam*L
  S5  no root -> fail ; nothing else is ever fail
  S6  reroot leaves a valid tree with the intended root
  S7  origin tolerance only fires on an aligned frame
  S8  plot_fn is called on the pathological path and never otherwise
"""

import math

import numpy as np
import pandas as pd

import node_classify as nc
import soma_enforce as se


COLS = ["id", "p", "x", "y", "z", "r", "annotated_type"]


def _mk(rows):
    return pd.DataFrame(rows, columns=COLS)


def _intact(soma_r=5325.5):
    """One soma at the origin, two dendrites, an axon. Measured proportions."""
    return _mk([
        (0, -1,     0.0,    0.0, 0.0, soma_r, "Soma"),
        (1,  0,  1000.0,    0.0, 0.0,  300.0, "Dendrite"),
        (2,  1,  2000.0,    0.0, 0.0,  280.0, "Dendrite"),
        (3,  0,     0.0, 1000.0, 0.0,  140.0, "Dendrite"),
        (4,  3,     0.0, 2000.0, 0.0,  130.0, "Dendrite"),
        (5,  0, -1000.0,    0.0, 0.0,  150.0, "Axon"),
    ])


def _fragment():
    """The neuron_606394351 case: a 'Soma' of 331.9 nm radius."""
    return _intact(soma_r=331.9)


# --------------------------------------------------------------------------- #
def test_S1_geometry_intact_vs_fragment():
    ok = nc.classify_frame(_intact())
    g = se.identify_soma_by_geometry(ok)
    assert g["candidate_id"] == 0, g
    assert g["radius_above_floor"] is True, g
    assert g["radius_is_outlier"] is True, g
    assert g["passed"] is True, g
    # ratio recomputed independently: 5325.5 / median(300,280,140,130,150)
    med = float(np.median([300.0, 280.0, 140.0, 130.0, 150.0]))
    assert abs(g["radius_ratio"] - 5325.5 / med) < 1e-9, (g["radius_ratio"], med)

    frag = nc.classify_frame(_fragment())
    gf = se.identify_soma_by_geometry(frag)
    assert gf["radius_above_floor"] is False, gf
    assert gf["radius_is_outlier"] is False, gf
    assert gf["passed"] is False, gf


def test_S2_i18_agreement_is_soft():
    ok = nc.classify_frame(_intact())
    out, rep = se.enforce_soma(ok, nid="ok", verbose=False)
    assert rep["qc_status"] == se.QC_PASS, rep
    assert rep["reasons"] == [], rep
    assert rep["soma_id"] == 0

    # disagreement: the node CALLED soma is tiny, a dendrite node is huge
    bad = _intact(soma_r=100.0)
    bad.loc[bad["id"] == 2, "r"] = 6000.0
    bad = nc.classify_frame(bad)
    out2, rep2 = se.enforce_soma(bad, nid="bad", verbose=False)
    assert rep2["qc_status"] == se.QC_LOW, rep2       # SOFT, per Q6
    assert "name_geometry_disagree" in rep2["reasons"], rep2
    assert rep2["by_geometry"]["candidate_id"] == 2, rep2
    assert rep2["soma_id"] == 0, "the topological root stays the soma"

    # the fragment is flagged but still usable
    frag = nc.classify_frame(_fragment())
    out3, rep3 = se.enforce_soma(frag, nid="frag", verbose=False)
    assert rep3["qc_status"] == se.QC_LOW, rep3
    assert "soma_below_radius_floor" in rep3["reasons"], rep3
    assert "soma_not_outlier" in rep3["reasons"], rep3
    assert rep3["qc_status"] != se.QC_FAIL


def test_S3_collapse_conserves_area():
    df = _intact()
    # three soma nodes in a chain, with a dendrite hanging off the last one
    df.loc[len(df)] = (6, 0, 500.0, 0.0, 0.0, 4000.0, "Soma")
    df.loc[len(df)] = (7, 6, 900.0, 0.0, 0.0, 3000.0, "Soma")
    df.loc[len(df)] = (8, 7, 1500.0, 0.0, 0.0, 200.0, "Dendrite")
    df = nc.classify_frame(df)

    radii = [5325.5, 4000.0, 3000.0]
    expect_area = sum(4.0 * math.pi * (r / 1000.0) ** 2 for r in radii)
    expect_req = math.sqrt(sum(r ** 2 for r in radii))

    out, info = se.collapse_soma_nodes(df, [0, 6, 7])
    assert info["collapsed"] is True
    assert info["kept_id"] == 0, info
    assert abs(info["r_equivalent_nm"] - expect_req) < 1e-9
    assert abs(info["area_preserved_um2"] - expect_area) < 1e-9
    assert len(out) == len(df) - 2
    # node 8's parent was 7, which was removed -> reparented onto 0
    assert int(out.loc[out["id"] == 8, "p"].iloc[0]) == 0, out
    # no dangling parents anywhere
    ids = set(out["id"])
    assert all((p == -1) or (p in ids) for p in out["p"]), out

    # and through the entry point the flag is raised, softly
    out2, rep = se.enforce_soma(df, nid="multi", verbose=False)
    assert "multiple_soma_nodes" in rep["reasons"], rep
    assert rep["qc_status"] == se.QC_LOW


def test_S4_soma_area_convention():
    r_nm = 5325.5
    r_um = r_nm / 1000.0
    sphere = 4.0 * math.pi * r_um * r_um
    assert abs(se.soma_area_um2(r_nm) - sphere) < 1e-12
    # the emitted section is two points at z-r and z+r with diam = 2r,
    # so its lateral area is pi * diam * L
    L = 2.0 * r_um
    diam = 2.0 * r_um
    assert abs(math.pi * diam * L - sphere) < 1e-12, "convention broken"


def test_S5_only_missing_root_is_fatal():
    df = nc.classify_frame(_intact())
    noroot = df.copy()
    noroot.loc[noroot["id"] == 0, "p"] = 99
    out, rep = se.enforce_soma(noroot, nid="noroot", verbose=False)
    assert rep["qc_status"] == se.QC_FAIL, rep
    assert "no_root" in rep["reasons"]

    out2, rep2 = se.enforce_soma(pd.DataFrame(columns=COLS), verbose=False)
    assert rep2["qc_status"] == se.QC_FAIL and "empty_frame" in rep2["reasons"]

    # two roots is soft, not fatal
    two = df.copy()
    two.loc[two["id"] == 5, "p"] = -1
    out3, rep3 = se.enforce_soma(two, nid="two", verbose=False)
    assert rep3["qc_status"] == se.QC_LOW and "multiple_roots" in rep3["reasons"]

    # a cell with no soma-labelled node at all is soft too
    nosoma = _intact()
    nosoma.loc[nosoma["id"] == 0, "annotated_type"] = "Dendrite"
    nosoma = nc.classify_frame(nosoma)
    out4, rep4 = se.enforce_soma(nosoma, nid="nosoma", verbose=False)
    assert rep4["qc_status"] == se.QC_LOW, rep4
    assert "no_soma_labelled_node" in rep4["reasons"], rep4


def test_S6_reroot_valid_tree():
    df = nc.classify_frame(_intact())
    out = se._reroot(df, 2)
    roots = out.loc[out["p"] == -1, "id"].tolist()
    assert roots == [2], roots
    ids = set(out["id"])
    assert all((p == -1) or (p in ids) for p in out["p"])
    # still a tree: walking up from every node reaches the root without a cycle
    par = dict(zip(out["id"], out["p"]))
    for n in ids:
        cur, steps = n, 0
        while par[cur] != -1:
            cur = par[cur]
            steps += 1
            assert steps <= len(ids), "cycle introduced by reroot"
        assert cur == 2


def test_S7_origin_tolerance():
    aligned = nc.classify_frame(_intact())          # soma already at (0,0,0)
    g = se.identify_soma_by_geometry(aligned, origin_tolerance_nm=100.0)
    assert g["near_origin"] is True and g["passed"] is True

    raw = _intact()
    for c in ("x", "y", "z"):
        raw[c] = raw[c] + 2.7e6                     # raw H01 coordinates
    raw = nc.classify_frame(raw)
    g2 = se.identify_soma_by_geometry(raw, origin_tolerance_nm=100.0)
    assert g2["near_origin"] is False and g2["passed"] is False
    # with the test disabled (pre-alignment use) it passes again
    g3 = se.identify_soma_by_geometry(raw, origin_tolerance_nm=None)
    assert g3["near_origin"] is None and g3["passed"] is True

    out, rep = se.enforce_soma(raw, nid="raw", origin_tolerance_nm=100.0,
                               verbose=False)
    assert "soma_off_origin" in rep["reasons"], rep


def test_S8_plot_hook():
    calls = []

    def fake_plot(df, report, path):
        calls.append((len(df), path))

    ok = nc.classify_frame(_intact())
    se.enforce_soma(ok, plot_fn=fake_plot, plot_path="x.html", verbose=False)
    assert calls == [], "plot must not fire on a healthy cell"

    df = _intact()
    df.loc[len(df)] = (6, 0, 500.0, 0.0, 0.0, 4000.0, "Soma")
    df = nc.classify_frame(df)
    out, rep = se.enforce_soma(df, plot_fn=fake_plot, plot_path="y.html",
                               verbose=False)
    assert len(calls) == 1, calls
    assert rep["collapse_plot_path"] == "y.html", rep

    # a plot failure must not take the pipeline down
    def boom(df_, report_, path_):
        raise RuntimeError("no display")

    out2, rep2 = se.enforce_soma(df, plot_fn=boom, plot_path="z.html",
                                 verbose=False)
    assert "collapse_plot_error" in rep2, rep2
    assert rep2["qc_status"] == se.QC_LOW


# --------------------------------------------------------------------------- #
def _run_all():
    tests = [
        ("S1 geometry: intact vs fragment", test_S1_geometry_intact_vs_fragment),
        ("S2 I-18 agreement, soft QC", test_S2_i18_agreement_is_soft),
        ("S3 collapse conserves sphere area", test_S3_collapse_conserves_area),
        ("S4 soma area convention", test_S4_soma_area_convention),
        ("S5 only a missing root is fatal", test_S5_only_missing_root_is_fatal),
        ("S6 reroot yields a valid tree", test_S6_reroot_valid_tree),
        ("S7 origin tolerance", test_S7_origin_tolerance),
        ("S8 plot hook fires only when pathological", test_S8_plot_hook),
    ]
    n_pass = 0
    for name, fn in tests:
        try:
            fn()
            print("[PASS] %s" % name)
            n_pass += 1
        except AssertionError as e:
            print("[FAIL] %s -- %s" % (name, e))
        except Exception as e:                      # noqa: BLE001
            print("[ERROR] %s -- %r" % (name, e))
    print("-" * 62)
    print("%d / %d passed" % (n_pass, len(tests)))
    return n_pass == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
