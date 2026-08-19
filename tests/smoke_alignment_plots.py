"""Smoke test for alignment_plots.

Runs standalone (python3 smoke_alignment_plots.py) or under pytest. Uses the
Agg backend, writes into a temp dir and deletes it, so it needs no display and
leaves nothing behind.

Plotting code is easy to write and easy to get silently wrong, so these tests
check SEMANTICS wherever a figure exposes them -- that the geometry helpers
return the right vectors, that the raw panel really is re-centred, that the
rigidity panel flags a non-zero delta -- and fall back to "it produced a valid
non-empty artefact" only for the purely cosmetic parts.

Coverage
  G1  arbour_direction recovers a known planted direction
  G2  angle_from_z_deg on the cardinal cases, and NaN passthrough
  G3  edge extraction: one segment per non-root node, parent -> child order
  G4  subsampling is deterministic and respects the cap
  G5  unit scaling nm -> um, and a bad unit string is rejected
  G6  arbour_comparison: a plotly Figure, two scenes, raw panel re-centred
  G7  arbour_static and depth_profile produce non-trivial mpl figures
  G8  neighbourhood marks exactly the chosen references
  G9  batch_quality and orientation_consistency survive NaNs and one cell
  G10 rigidity flags a non-zero delta in the title, and is silent when zero
  G11 save_figure dispatches on backend and writes a real file
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

import alignment_plots as ap               # noqa: E402


COLS = ["id", "p", "x", "y", "z", "compartment_class"]


def _cell(direction=(0.0, 0.0, 1.0), n=6, step=10000.0, units="nm"):
    """Soma at the origin plus a straight dendrite along `direction` (nm)."""
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    rows = [(0, -1, 0.0, 0.0, 0.0, "soma")]
    for i in range(1, n):
        p = u * step * i
        rows.append((i, i - 1, p[0], p[1], p[2], "dend"))
    rows.append((n, 0, -step, 0.0, 0.0, "axon"))
    return pd.DataFrame(rows, columns=COLS)


# --------------------------------------------------------------------------- #
def test_G1_arbour_direction():
    for want in ([0, 0, 1], [1, 0, 0], [1, 1, 1], [0, -1, 0]):
        df = _cell(want)
        got = ap.arbour_direction(df)
        exp = np.asarray(want, float) / np.linalg.norm(want)
        assert got is not None
        assert np.allclose(got, exp, atol=1e-9), (want, got)
        assert abs(np.linalg.norm(got) - 1.0) < 1e-12

    # no dendrite at all -> None, not a crash
    only_axon = pd.DataFrame([(0, -1, 0.0, 0.0, 0.0, "soma"),
                              (1, 0, 100.0, 0.0, 0.0, "axon")], columns=COLS)
    assert ap.arbour_direction(only_axon) is None
    assert ap.arbour_direction(pd.DataFrame(columns=COLS)) is None


def test_G1b_direction_follows_distal_tips_not_bulk():
    """The direction must come from the DISTAL tips, not from every node.

    A pyramidal cell carries a dense proximal basal skirt and a sparse apical
    trunk. Averaging over all nodes lets the numerous proximal nodes outvote
    the trunk and returns the wrong axis; the `calculate_z_alignment_math`
    construction takes the centre of mass of the most distal tips precisely to
    avoid that. This fixture makes the two answers point in different
    directions, so the shortcut cannot pass.
    """
    rows = [(0, -1, 0.0, 0.0, 0.0, "soma")]
    nid = 1
    # 80 proximal skirt nodes along -x. Numerous, so they dominate a naive mean.
    for i in range(80):
        ang = 2 * np.pi * i / 80.0
        rows.append((nid, 0, -50000.0, 3000.0 * np.cos(ang),
                     3000.0 * np.sin(ang), "dend"))
        nid += 1
    # 20 trunk nodes along +z, further out. They must be at least 10% of the
    # nodes or a 90th-percentile cut cannot isolate them.
    prev = 0
    for r in np.linspace(60000.0, 100000.0, 20):
        rows.append((nid, prev, 0.0, 0.0, r, "dend"))
        prev = nid
        nid += 1
    df = pd.DataFrame(rows, columns=COLS)

    got = ap.arbour_direction(df)
    assert got is not None
    assert got[2] > 0.95, (
        "direction must follow the distal trunk to +z, got %r. Averaging over "
        "ALL nodes instead of the distal tips produces this failure." % (got,))
    assert abs(ap.angle_from_z_deg(got)) < 15.0

    # confirm the fixture really does discriminate: the all-node centre of mass
    # points somewhere else entirely
    P = df[df["compartment_class"] == "dend"][["x", "y", "z"]].to_numpy(float)
    bulk = P.mean(axis=0)
    bulk = bulk / np.linalg.norm(bulk)
    assert ap.angle_from_z_deg(bulk) > 45.0, (
        "fixture is not discriminating; bulk direction is %r" % (bulk,))


def test_G2_angle_from_z():
    assert abs(ap.angle_from_z_deg(np.array([0.0, 0.0, 1.0]))) < 1e-9
    assert abs(ap.angle_from_z_deg(np.array([0.0, 0.0, -1.0])) - 180.0) < 1e-9
    assert abs(ap.angle_from_z_deg(np.array([1.0, 0.0, 0.0])) - 90.0) < 1e-9
    assert abs(ap.angle_from_z_deg(np.array([0.0, 1.0, 1.0]) / np.sqrt(2))
               - 45.0) < 1e-9
    assert np.isnan(ap.angle_from_z_deg(None))


def test_G3_edges():
    df = _cell()
    segs, cls = ap._edges(df)
    assert len(segs) == len(df) - 1, "one segment per non-root node"
    assert segs.shape[1:] == (2, 3)
    assert len(cls) == len(segs)
    # segment 0 runs soma -> first dendrite node, in that order
    assert np.allclose(segs[0][0], [0.0, 0.0, 0.0])
    assert np.allclose(segs[0][1], [0.0, 0.0, 10000.0])
    assert "axon" in set(cls) and "dend" in set(cls)

    # a dangling parent is skipped rather than raising
    broken = df.copy()
    broken.loc[broken["id"] == 3, "p"] = 999
    segs2, _ = ap._edges(broken)
    assert len(segs2) == len(segs) - 1


def test_G4_subsample():
    df = _cell(n=50)
    segs, cls = ap._edges(df)
    s1, c1 = ap._subsample(segs, cls, 10)
    s2, c2 = ap._subsample(segs, cls, 10)
    assert len(s1) <= 10 and len(s1) == len(c1)
    assert np.array_equal(s1, s2), "subsampling is not deterministic"
    s3, _ = ap._subsample(segs, cls, None)
    assert len(s3) == len(segs), "None must mean no cap"
    s4, _ = ap._subsample(segs, cls, 10 ** 9)
    assert len(s4) == len(segs)


def test_G5_scaling():
    df = _cell()
    um = ap._scale(df, "nm")
    assert abs(float(um.loc[1, "z"]) - 10.0) < 1e-12
    assert abs(float(df.loc[1, "z"]) - 10000.0) < 1e-9, "input mutated"
    same = ap._scale(df, "um")
    assert abs(float(same.loc[1, "z"]) - 10000.0) < 1e-9
    try:
        ap._scale(df, "furlongs")
    except ValueError:
        pass
    else:
        raise AssertionError("a bad unit string was accepted")


def test_G6_arbour_comparison():
    raw = _cell([1.0, 1.0, 0.3])
    for c, off in zip("xyz", (2.7e6, 5.3e5, 5.7e4)):
        raw[c] = raw[c] + off                       # MICrONS-like coordinates
    ali = _cell([0.0, 0.0, 1.0])

    fig = ap.arbour_comparison(raw, ali, nid=42)
    assert hasattr(fig, "write_html"), "not a plotly Figure"
    assert len(fig.data) >= 4, "expected traces for both panels"
    scenes = {t.scene for t in fig.data if hasattr(t, "scene")}
    assert len(scenes) == 2, scenes

    # the raw panel must be RE-CENTRED, or the two panels are incomparable
    left = np.concatenate([np.asarray(t.x, float) for t in fig.data
                           if getattr(t, "scene", "scene") == "scene"])
    left = left[np.isfinite(left)]
    assert np.abs(left).max() < 1e3, (
        "raw panel was not centred on its soma; max |x| = %g" % np.abs(left).max())
    assert np.isnan(np.concatenate([np.asarray(t.x, float) for t in fig.data
                                    if t.mode == "lines"])).any(), \
        "line traces must use NaN separators between segments"


def test_G7_static_and_depth():
    ali = _cell([0.0, 0.0, 1.0], n=30)
    raw = _cell([1.0, 0.0, 0.2], n=30)

    fig = ap.arbour_static(raw, ali, nid=7)
    assert len(fig.axes) == 2
    plt.close(fig)

    fig = ap.depth_profile(ali, nid=7)
    assert len(fig.axes) == 2
    # the cumulative panel must report a fraction, and this cell is all +z
    titles = " ".join(a.get_title() for a in fig.axes)
    assert "z > 0" in titles, titles
    assert "100.0%" in titles, titles
    plt.close(fig)

    flipped = _cell([0.0, 0.0, -1.0], n=30)
    fig = ap.depth_profile(flipped, nid=8)
    titles = " ".join(a.get_title() for a in fig.axes)
    assert "0.0%" in titles, "an upside-down cell must report 0% above z=0"
    plt.close(fig)


def test_G8_neighbourhood():
    md = pd.DataFrame({
        "neuron_id": range(6),
        "soma_x": np.linspace(0, 5e5, 6),
        "soma_y": np.zeros(6), "soma_z": np.zeros(6),
    })
    diag = {"neighbour_row_indices": [0, 1, 2], "pairwise_angle_deg_max": 54.5}
    fig = ap.neighbourhood(np.array([0.0, 0.0, 0.0]), md, diag, nid=1)
    assert len(fig.axes) == 2
    assert "54.5" in (fig._suptitle.get_text() if fig._suptitle else "")
    # exactly the 3 chosen references are ringed on the right-hand panel
    marked = [c for c in fig.axes[1].collections if len(c.get_offsets()) == 3]
    assert marked, "the k chosen references were not marked"
    plt.close(fig)


def test_G9_batch_panels():
    recs = []
    for i, ang in enumerate([2.0, 8.0, 15.0, float("nan")]):
        recs.append({
            "nid": 100 + i, "qc_status": "pass", "n_sections": 1000 + i,
            "angle_from_z_deg": ang,
            "alignment": {"neighbour_distance_um": [10.0 + i, 50.0, 90.0],
                          "pairwise_angle_deg_max": 5.0 * i,
                          "det": 1.0, "orthonormality_error": 1e-15},
        })
    fig = ap.batch_quality(recs); assert len(fig.axes) == 4; plt.close(fig)
    fig = ap.orientation_consistency(recs)
    assert "n = 3" in fig._suptitle.get_text(), fig._suptitle.get_text()
    plt.close(fig)

    # a single cell must not blow up
    fig = ap.batch_quality(recs[:1]); plt.close(fig)
    fig = ap.orientation_consistency(recs[:1]); plt.close(fig)

    # an upside-down cell is counted and named in the title
    recs[0]["angle_from_z_deg"] = 175.0
    fig = ap.orientation_consistency(recs)
    assert "1 cell(s) beyond 90 deg" in fig._suptitle.get_text()
    plt.close(fig)


def test_G10_rigidity_flags_movement():
    base = {"n_sections": 1785, "n_branches": 1760, "f_implied": 1.5285,
            "F_lit": 1.8948, "A_shaft_um2": 20360.6, "A_spine_um2": 10761.1}
    clean = [{"nid": 1, "unaligned": dict(base), "aligned": dict(base)}]
    fig = ap.rigidity(clean)
    assert "NON-ZERO" not in fig.axes[0].get_title()
    plt.close(fig)

    moved = dict(base); moved["A_shaft_um2"] = 20360.7
    dirty = [{"nid": 1, "unaligned": dict(base), "aligned": moved}]
    fig = ap.rigidity(dirty)
    assert "NON-ZERO DELTA" in fig.axes[0].get_title(), \
        "a moved quantity was not flagged"
    plt.close(fig)


def test_G11_save_dispatch():
    tmp = tempfile.mkdtemp()
    try:
        fig = ap.arbour_static(_cell(), _cell(), nid=1)
        p = ap.save_figure(fig, os.path.join(tmp, "a", "static.png"))
        assert p.endswith(".png") and os.path.getsize(p) > 1000

        fig = ap.arbour_static(_cell(), _cell(), nid=1)
        p2 = ap.save_figure(fig, os.path.join(tmp, "noext"))
        assert p2.endswith(".png") and os.path.isfile(p2)

        figp = ap.arbour_comparison(_cell(), _cell(), nid=1)
        p3 = ap.save_figure(figp, os.path.join(tmp, "inter.png"))
        assert p3.endswith(".html"), "a plotly figure must be written as html"
        assert os.path.getsize(p3) > 1000
        assert "plotly" in open(p3, "r", encoding="utf-8").read()[:5000].lower()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
def _run_all():
    tests = [
        ("G1  arbour_direction recovers a planted vector", test_G1_arbour_direction),
        ("G1b direction follows distal tips, not bulk", test_G1b_direction_follows_distal_tips_not_bulk),
        ("G2  angle_from_z_deg cardinal cases", test_G2_angle_from_z),
        ("G3  edge extraction", test_G3_edges),
        ("G4  subsampling deterministic and capped", test_G4_subsample),
        ("G5  unit scaling and validation", test_G5_scaling),
        ("G6  arbour_comparison re-centres the raw panel", test_G6_arbour_comparison),
        ("G7  static arbour and depth profile", test_G7_static_and_depth),
        ("G8  neighbourhood marks the chosen k", test_G8_neighbourhood),
        ("G9  batch panels survive NaN and n=1", test_G9_batch_panels),
        ("G10 rigidity flags a non-zero delta", test_G10_rigidity_flags_movement),
        ("G11 save_figure dispatches on backend", test_G11_save_dispatch),
    ]
    n = 0
    for name, fn in tests:
        try:
            fn()
            print("[PASS] %s" % name)
            n += 1
        except AssertionError as e:
            print("[FAIL] %s -- %s" % (name, e))
        except Exception as e:                          # noqa: BLE001
            print("[ERROR] %s -- %r" % (name, e))
    print("-" * 62)
    print("%d / %d passed" % (n, len(tests)))
    return n == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
