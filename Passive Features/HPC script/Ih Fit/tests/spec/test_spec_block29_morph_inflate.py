"""Spec-derived tests of Block 29 (SPEC 2.11, D-268): morph_inflate.py, the
build on an inflated morphology (twin_build.build_twin's sim_swc_path) and the
run's options and record (run_twin_recovery.py). Integration tester,
2026-11-27 (incremental run on 8e1fbdb, base 718c778).

Oracles, written from SPEC 2.11 and the Block 29 entry before the code was read:
- the path length of a node is the sum of the Euclidean distances along its
  parent chain to the soma's root node (computed here by hand on trees with
  3-4-5 / 5-12-13 geometry, and by scipy.sparse.csgraph on a random tree);
- radius * f on types 1, 3, 4 when path <= L (closed end), * 1 beyond and on
  every other type; ids, types, coordinates, parents unchanged; radii to 10
  significant digits (relative error <= 5e-10); one comment line added;
- refusals: factor not finite and > 0, L not finite and >= 0, dst is src, a
  duplicated id, a parent that is not a node, a cycle (ValueError);
- plan: the two options recorded only when f != 1; refused (the entry's
  Refused, exit code 3 like every other plan refusal) otherwise;
- end to end: the archive stores the ORIGINAL reconstruction, byte for byte;
  the twin's traces equal those of a plain twin of a real cell whose
  reconstruction is the inflated one (written here by an independent writer);
  the noise is unchanged (noisy minus noise-free equal in both runs); the
  record of provenance.sim_morphology; reuse only under the same inflation.
"""
import contextlib
import csv
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import morph_inflate as MI  # noqa: E402

CODE_DIR = Path(__file__).resolve().parents[2]


# ============================================================================ helpers (independent of the code)

def _write(path, rows, header=("# hand tree",)):
    path = Path(path)
    lines = list(header) + ["%d %d %r %r %r %r %d" % (int(r[0]), int(r[1]), float(r[2]), float(r[3]), float(r[4]),
                                                    float(r[5]), int(r[6])) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


# A hand tree, listed OUT OF ORDER (children before parents) with non-contiguous ids and 3-D steps whose lengths
# are integers (3-4-5, 5-12-13, 2-3-6-7): the path lengths below are exact in floating point.
#   id 10 soma root (0,0,0) r 8
#   id 20 basal  (3,4,0)    parent 10  -> 5
#   id 21 basal  (3,4,12)   parent 20  -> 5 + 12 = 17
#   id 22 basal  (3,4,12)+(2,3,6)=(5,7,18) parent 21 -> 17 + 7 = 24
#   id 30 apical (0,0,-13)  parent 10 -> 13
#   id 31 apical (5,12,-13) parent 30 -> 13 + 13 = 26
#   id 40 axon   (0,-5,0)   parent 10 -> 5
#   id 50 type 7 (custom)   (0,0,5) parent 10 -> 5
#   id 60 soma   (0,0,1)    parent 10 -> 1   (a second soma node)
HAND = np.array([
    [22, 3, 5, 7, 18, 0.7, 21],
    [31, 4, 5, 12, -13, 1.25, 30],
    [20, 3, 3, 4, 0, 1.0, 10],
    [10, 1, 0, 0, 0, 8.0, -1],
    [21, 3, 3, 4, 12, 0.9, 20],
    [30, 4, 0, 0, -13, 1.5, 10],
    [40, 2, 0, -5, 0, 0.4, 10],
    [50, 7, 0, 0, 5, 0.6, 10],
    [60, 1, 0, 0, 1, 7.0, 10],
], dtype=float)
HAND_PATH = {22: 24.0, 31: 26.0, 20: 5.0, 10: 0.0, 21: 17.0, 30: 13.0, 40: 5.0, 50: 5.0, 60: 1.0}


def _expected_factor(rows, path, f, L, types=(1, 3, 4)):
    typ = rows[:, 1].astype(int)
    return np.where(np.isin(typ, types) & (path <= L), f, 1.0)


# ============================================================================ pure: path length

def test_path_length_hand_tree_out_of_order():
    d = MI.path_length_from_root(HAND)
    want = np.array([HAND_PATH[int(i)] for i in HAND[:, 0]])
    assert d.shape == (HAND.shape[0],) and d.dtype == np.float64
    # exact: every step length is an integer computed from integer coordinates
    assert_allclose(d, want, rtol=0, atol=1e-12)


def test_path_length_random_tree_against_scipy_csgraph():
    """Reference: scipy.sparse.csgraph.dijkstra on the tree's undirected edge graph (in a tree the shortest path
    from the root is the parent chain)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    rng = np.random.default_rng(29)
    n = 400
    par = np.array([-1] + [int(rng.integers(0, k)) for k in range(1, n)])
    xyz = rng.normal(scale=20.0, size=(n, 3))
    ids = rng.permutation(np.arange(1000, 1000 + 7 * n, 7))[:n]       # non-contiguous, shuffled ids
    rows = np.column_stack([ids, np.full(n, 3.0), xyz, np.ones(n), np.where(par < 0, -1, ids[np.maximum(par, 0)])])
    rows[0, 1] = 1.0
    perm = rng.permutation(n)
    rows = rows[perm]
    w = np.linalg.norm(xyz[1:] - xyz[par[1:]], axis=1)
    g = coo_matrix((w, (np.arange(1, n), par[1:])), shape=(n, n))
    ref = dijkstra(g, directed=False, indices=0)
    d = MI.path_length_from_root(rows)
    # tolerance: float64 sums of up to n terms of size ~30 um; 1e-9 um relative is far above roundoff
    assert_allclose(d, ref[perm], rtol=1e-12, atol=1e-9)


def test_path_length_long_chain_is_not_quadratic():
    """A 20000-node unbranched chain listed leaf first: linear-time bookkeeping expected (a few seconds at most)."""
    import time
    n = 20000
    ids = np.arange(1, n + 1)
    rows = np.column_stack([ids, np.full(n, 3.0), ids.astype(float), np.zeros(n), np.zeros(n), np.ones(n),
                            ids - 1])
    rows[0, 6] = -1
    t0 = time.perf_counter()
    d = MI.path_length_from_root(rows[::-1].copy())
    dt = time.perf_counter() - t0
    assert_allclose(d[::-1], np.arange(n, dtype=float), rtol=0, atol=1e-9)
    assert dt < 20.0, dt


# ============================================================================ pure: the copy

@pytest.mark.parametrize("L", [0.0, 1.0, 5.0, 13.0, 17.0, 24.0, 25.0, 1e9])
def test_inflate_step_types_closed_end(tmp_path, L):
    f = 1.5
    src = _write(tmp_path / "src.swc", HAND)
    before = src.read_bytes()
    dst = tmp_path / "dst.swc"
    c = MI.inflate_swc(src, dst, factor=f, max_path_um=L)
    out, head = MI.read_swc(dst)
    path = np.array([HAND_PATH[int(i)] for i in HAND[:, 0]])
    fac = _expected_factor(HAND, path, f, L)
    # tolerance: radii written with 10 significant digits -> relative error <= 5e-10
    assert_allclose(out[:, 5], HAND[:, 5] * fac, rtol=6e-10, atol=0)
    # ids, types, coordinates, parents, row order unchanged (the hand values have few digits: exact)
    assert_array_equal(out[:, [0, 1, 2, 3, 4, 6]], HAND[:, [0, 1, 2, 3, 4, 6]])
    assert src.read_bytes() == before
    assert head[:1] == ["# hand tree"] and len(head) == 2 and head[1].startswith("#")
    assert c["n_nodes"] == HAND.shape[0] and c["n_inflated"] == int((fac != 1.0).sum())
    assert c["factor"] == f and c["max_path_um"] == L
    nbt = {int(k): int(v) for k, v in c["n_by_type"].items() if int(v) > 0}
    want = {}
    for t in (1, 3, 4):
        k = int(((HAND[:, 1] == t) & (fac != 1.0)).sum())
        if k:
            want[t] = k
    assert nbt == want
    # the axon (2) and the custom type (7) are never touched
    assert np.all(out[np.isin(HAND[:, 1], (2, 7)), 5] == HAND[np.isin(HAND[:, 1], (2, 7)), 5])


def test_inflate_factor_one_is_identity_and_boundary_node_included(tmp_path):
    src = _write(tmp_path / "src.swc", HAND)
    MI.inflate_swc(src, tmp_path / "one.swc", factor=1.0, max_path_um=300.0)
    out, _ = MI.read_swc(tmp_path / "one.swc")
    assert_array_equal(out, HAND)
    # node 21 is at exactly 17 um: included at L = 17, excluded at the next float below
    MI.inflate_swc(src, tmp_path / "a.swc", factor=2.0, max_path_um=17.0)
    MI.inflate_swc(src, tmp_path / "b.swc", factor=2.0, max_path_um=float(np.nextafter(17.0, 0.0)))
    a, _ = MI.read_swc(tmp_path / "a.swc")
    b, _ = MI.read_swc(tmp_path / "b.swc")
    k = int(np.where(HAND[:, 0] == 21)[0][0])
    assert a[k, 5] == 2 * HAND[k, 5] and b[k, 5] == HAND[k, 5]


def test_inflate_coordinates_with_many_digits_are_copied(tmp_path):
    """'everything else copied': coordinates and radii given with 15-17 significant digits (an SWC exported from
    float64) survive the copy. Tolerance: none stated for coordinates; we accept relative 1e-12."""
    rows = np.array([[1, 1, 0.123456789012345, 1234.56789012345, -0.000123456789012345, 5.123456789012345, -1],
                     [2, 3, 10.987654321098765, 2.0, 3.0, 0.987654321098765, 1]])
    src = _write(tmp_path / "s.swc", rows)
    MI.inflate_swc(src, tmp_path / "d.swc", factor=1.0, max_path_um=1e6)
    out, _ = MI.read_swc(tmp_path / "d.swc")
    assert_allclose(out[:, 2:5], rows[:, 2:5], rtol=1e-12, atol=0)


@pytest.mark.parametrize("kw", [dict(factor=0.0, max_path_um=1.0), dict(factor=-1.5, max_path_um=1.0),
                                dict(factor=float("nan"), max_path_um=1.0), dict(factor=float("inf"), max_path_um=1.0),
                                dict(factor=1.5, max_path_um=-1e-9), dict(factor=1.5, max_path_um=float("nan")),
                                dict(factor=1.5, max_path_um=float("inf"))])
def test_inflate_refuses_bad_numbers(tmp_path, kw):
    src = _write(tmp_path / "src.swc", HAND)
    with pytest.raises(ValueError):
        MI.inflate_swc(src, tmp_path / "x.swc", **kw)
    assert not (tmp_path / "x.swc").exists()


def test_inflate_refuses_dst_is_src_in_any_spelling(tmp_path):
    src = _write(tmp_path / "src.swc", HAND)
    before = src.read_bytes()
    (tmp_path / "sub").mkdir()
    link = tmp_path / "link.swc"
    link.symlink_to(src)
    for dst in (src, tmp_path / "sub" / ".." / "src.swc", link):
        with pytest.raises(ValueError):
            MI.inflate_swc(src, dst, factor=1.5, max_path_um=10.0)
    assert src.read_bytes() == before


@pytest.mark.parametrize("bad", ["dup", "orphan", "cycle", "self"])
def test_inflate_refuses_bad_trees(tmp_path, bad):
    rows = HAND.copy()
    if bad == "dup":
        rows[0, 0] = 20
    elif bad == "orphan":
        rows[0, 6] = 999
    elif bad == "cycle":                       # 20 -> 21 -> 20, detached from the root
        rows[rows[:, 0] == 20, 6] = 21
    else:
        rows[rows[:, 0] == 31, 6] = 31
    src = _write(tmp_path / "src.swc", rows)
    with pytest.raises(ValueError):
        MI.inflate_swc(src, tmp_path / "x.swc", factor=1.5, max_path_um=10.0)
    assert not (tmp_path / "x.swc").exists()


# ============================================================================ the plan's options

pytest.importorskip("neuron")
pytest.importorskip("skopt")

import run_twin_recovery as rtr    # noqa: E402
from test_spec_block8_twin_recovery import group  # noqa: E402,F401  (the fixture group of Block 8's tests)

SMALL29 = ["--n-per-configuration", "1", "--n-no-noise", "1", "--n-calls", "2", "--n-initial", "2",
           "--refine-max-nfev", "1", "--refine-starts", "1", "--refine-maxfev", "2", "--seed", "4"]
F29, L29 = 1.5, 300.0
INFL = ["--twin-diam-factor", str(F29), "--twin-diam-max-path-um", str(L29)]


def _plan(gdir, ndir, out, *extra):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = rtr.main(["plan", "--group-dir", str(gdir), "--noise-dir", str(ndir), "--out", str(out)]
                        + SMALL29 + list(extra))
    return code, err.getvalue()


@pytest.mark.parametrize("extra", [["--twin-diam-factor", "0"], ["--twin-diam-factor", "-1.5"],
                                   ["--twin-diam-factor", "nan"], ["--twin-diam-factor", "inf"],
                                   ["--twin-diam-factor", "1.5", "--twin-diam-max-path-um", "-1"],
                                   ["--twin-diam-factor", "1.5", "--twin-diam-max-path-um", "nan"]])
def test_plan_refuses_bad_inflation_with_exit_3(group, tmp_path, extra):
    """SPEC Block 29: 'Refused: factor not finite and > 0, path not finite and >= 0'. Every other refusal of
    `plan` on its option values is Refused(EXIT_REFUSED = 3) or (EXIT_USAGE = 2); a Python traceback is not."""
    out = tmp_path / "r"
    try:
        code, err = _plan(group["gdir"], group["ndir"], out, *extra)
    except TypeError as exc:                 # the refusal itself crashed
        pytest.fail("plan %s raised %s: %s (expected a refusal with exit 2 or 3)" % (extra, type(exc).__name__, exc))
    assert code in (2, 3), (code, err)
    assert not (out / "manifest.csv").exists()


def test_plan_options_recorded_only_when_inflated(group, tmp_path):
    c0, _ = _plan(group["gdir"], group["ndir"], tmp_path / "a")
    c1, _ = _plan(group["gdir"], group["ndir"], tmp_path / "b", "--twin-diam-factor", "1",
                  "--twin-diam-max-path-um", "50")
    c2, _ = _plan(group["gdir"], group["ndir"], tmp_path / "c", *INFL)
    assert (c0, c1, c2) == (0, 0, 0)
    oa = json.loads((tmp_path / "a" / "run_config.json").read_text())["options"]
    ob = json.loads((tmp_path / "b" / "run_config.json").read_text())["options"]
    oc = json.loads((tmp_path / "c" / "run_config.json").read_text())["options"]
    strip = lambda o: {k: v for k, v in o.items() if k != "out"}  # noqa: E731
    assert "twin_diam_factor" not in oa and "twin_diam_max_path_um" not in oa
    assert strip(ob) == strip(oa)                       # factor 1: the record of a plan made before Block 29
    assert oc["twin_diam_factor"] == F29 and oc["twin_diam_max_path_um"] == L29
    assert {k: v for k, v in strip(oc).items() if not k.startswith("twin_diam")} == strip(oa)
    # the manifest does not depend on the inflation (same seeds, same twins)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "a" / "manifest.csv"),
                                  pd.read_csv(tmp_path / "c" / "manifest.csv"))
    # a plan without the options on an inflated run differs -> refused without --force; the same plan is a no-op
    assert _plan(group["gdir"], group["ndir"], tmp_path / "c")[0] == 3
    assert _plan(group["gdir"], group["ndir"], tmp_path / "c", *INFL)[0] == 0
    assert rtr.twin_inflation(oa) is None and rtr.twin_inflation(ob) is None
    assert rtr.twin_inflation(oc) == dict(factor=F29, max_path_um=L29)


# ============================================================================ end to end

def _independent_inflated_swc(src, dst, f, L):
    """An inflated copy written WITHOUT morph_inflate: BFS from the parent -1 node over the parent links."""
    rows = np.loadtxt(src, comments="#", ndmin=2)
    ids = rows[:, 0].astype(int)
    pos = {i: k for k, i in enumerate(ids)}
    path = np.zeros(len(ids))
    children = {}
    for k, p in enumerate(rows[:, 6].astype(int)):
        children.setdefault(p, []).append(k)
    order, queue = [], list(children[-1])
    while queue:
        k = queue.pop(0)
        order.append(k)
        p = int(rows[k, 6])
        if p >= 0:
            q = pos[p]
            path[k] = path[q] + np.sqrt(np.sum((rows[k, 2:5] - rows[q, 2:5]) ** 2))
        queue.extend(children.get(int(ids[k]), []))
    assert len(order) == len(ids)
    fac = _expected_factor(rows, path, f, L)
    with open(dst, "w") as fh:
        fh.write("# independent inflated copy\n")
        for k in range(len(ids)):
            r = rows[k]
            fh.write("%d %d %.17g %.17g %.17g %.17g %d\n" % (int(r[0]), int(r[1]), r[2], r[3], r[4], r[5] * fac[k],
                                                           int(r[6])))
    return fac


@pytest.fixture(scope="module")
def e2e29(group, tmp_path_factory):
    """Three runs with the same seeds: plain; inflated; plain on a copy of the group whose real cells'
    reconstruction.swc is the independently inflated one ('reference'). In each, the full twin (noisy) and its
    full_no_noise partner are built and fitted at the smallest budget."""
    root = tmp_path_factory.mktemp("e29")
    gref = root / "refgroup" / "G"          # the group name is the directory's basename
    shutil.copytree(group["gdir"], gref)
    facs = {}
    for d in sorted(gref.glob("specimen_*")):
        facs[d.name] = _independent_inflated_swc(group["gdir"] / d.name / "reconstruction.swc",
                                                 d / "reconstruction.swc", F29, L29)
    runs = {}
    for key, gdir, extra in (("plain", group["gdir"], ()), ("infl", group["gdir"], INFL), ("ref", gref, ())):
        out = root / key
        code, err = _plan(gdir, group["ndir"], out, *extra)
        assert code == 0, err
        man = pd.read_csv(out / "manifest.csv")
        tid = {str(c): int(t) for c, t in zip(man["configuration"], man["specimen_id"])}
        codes = {c: rtr.main(["twin", "--out", str(out), "--twin", str(tid[c])])
                 for c in ("full", "full_no_noise")}
        runs[key] = dict(out=out, man=man, tid=tid, codes=codes)
    return dict(runs=runs, facs=facs, gref=gref, root=root)


def _sdir(run, conf):
    t = run["tid"][conf]
    return run["out"] / "twins" / ("twin_%d" % t) / ("specimen_%d" % t)


def _arrays(spec_dir):
    ss = np.load(spec_dir / "ss_pulses.npz", allow_pickle=False)
    ls = np.load(spec_dir / "ls_sweeps.npz", allow_pickle=False)
    return {("ss", k): np.asarray(ss[k]) for k in ss.files} | {("ls", k): np.asarray(ls[k]) for k in ls.files}


def test_e2e_runs_finish(e2e29):
    for key, run in e2e29["runs"].items():
        assert run["codes"] == {"full": 0, "full_no_noise": 0}, (key, run["codes"])
        for c in ("full", "full_no_noise"):
            st = json.loads((run["out"] / "status" / ("twin_%d.json" % run["tid"][c])).read_text())
            assert st["status"] == "ok", (key, c, st)


def test_e2e_archive_keeps_the_original_reconstruction(e2e29, group):
    run = e2e29["runs"]["infl"]
    man = run["man"].set_index("specimen_id")
    for c in ("full", "full_no_noise"):
        real = int(man.loc[run["tid"][c], "twin_of_specimen_id"])
        orig = (group["gdir"] / ("specimen_%d" % real) / "reconstruction.swc").read_bytes()
        assert (_sdir(run, c) / "reconstruction.swc").read_bytes() == orig


def test_e2e_sim_morphology_file_and_record(e2e29, group):
    run = e2e29["runs"]["infl"]
    man = run["man"].set_index("specimen_id")
    for c in ("full", "full_no_noise"):
        t = run["tid"][c]
        real = int(man.loc[t, "twin_of_specimen_id"])
        sim = run["out"] / "twins" / ("twin_%d" % t) / ("sim_morphology_%d.swc" % t)
        assert sim.is_file()
        orig = np.loadtxt(group["gdir"] / ("specimen_%d" % real) / "reconstruction.swc", comments="#", ndmin=2)
        got = np.loadtxt(sim, comments="#", ndmin=2)
        fac = e2e29["facs"]["specimen_%d" % real]
        assert_array_equal(got[:, [0, 1, 2, 3, 4, 6]], orig[:, [0, 1, 2, 3, 4, 6]])
        assert_allclose(got[:, 5], orig[:, 5] * fac, rtol=6e-10, atol=0)   # 10 significant digits
        tw = json.loads((_sdir(run, c) / "metadata.json").read_text())["twin"]
        rec = tw["provenance"]["sim_morphology"]
        assert rec == dict(factor=F29, max_path_um=L29, n_nodes=int(orig.shape[0]),
                           n_inflated=int((fac != 1.0).sum()), file=sim.name), rec
    for c in ("full", "full_no_noise"):
        tw = json.loads((_sdir(e2e29["runs"]["plain"], c) / "metadata.json").read_text())["twin"]
        assert "sim_morphology" not in (tw.get("provenance") or {})


def test_e2e_inflated_twin_equals_plain_twin_of_an_inflated_cell(e2e29):
    """The inflated run's traces are those of a plain run whose real cells carry the inflated reconstruction
    (written independently): same truth, stimuli and noise. Tolerance 1e-12 mV: the two NEURON runs read the
    same radii (the reference file holds f * r to 17 digits; morph_inflate's to 10 -- the fixture's radii 1.5,
    1.8, 15 are exact in both)."""
    ri, rr = e2e29["runs"]["infl"], e2e29["runs"]["ref"]
    pd.testing.assert_frame_equal(ri["man"].drop(columns=["real_swc_path", "swc"], errors="ignore"),
                                  rr["man"].drop(columns=["real_swc_path", "swc"], errors="ignore"))
    for c in ("full", "full_no_noise"):
        a, b = _arrays(_sdir(ri, c)), _arrays(_sdir(rr, c))
        assert a.keys() == b.keys()
        for k in a:
            assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, k
            if a[k].dtype.kind == "f":
                assert np.all(np.isfinite(a[k])), k
                assert_allclose(a[k], b[k], rtol=0, atol=1e-12, err_msg=str(k))
            else:
                assert_array_equal(a[k], b[k], err_msg=str(k))
        mi = json.loads((_sdir(ri, c) / "metadata.json").read_text())
        mr = json.loads((_sdir(rr, c) / "metadata.json").read_text())
        for k in ("rin_MOhm", "tau_ms", "sag_ratio", "v_rest_mV"):
            assert_allclose(mi[k], mr[k], rtol=1e-12, atol=0, err_msg=k)


def test_e2e_noise_unchanged_by_the_inflation(e2e29):
    """2.11: 'the noise ... unchanged': (full - full_no_noise) is the same in the plain and the inflated run.
    Tolerance 1e-9 mV: the difference of two traces of ~-70 mV carries roundoff of a few 1e-14 mV."""
    rp, ri = e2e29["runs"]["plain"], e2e29["runs"]["infl"]
    assert rp["tid"] == ri["tid"]
    np_, nn_ = _arrays(_sdir(rp, "full")), _arrays(_sdir(rp, "full_no_noise"))
    ip_, in_ = _arrays(_sdir(ri, "full")), _arrays(_sdir(ri, "full_no_noise"))
    vkeys = [k for k in np_ if (k[0] == "ss" and k[1] == "v") or (k[0] == "ls" and k[1].startswith("v_"))]
    assert vkeys
    moved = 0.0
    for k in vkeys:
        assert_allclose(ip_[k] - in_[k], np_[k] - nn_[k], rtol=0, atol=1e-9, err_msg=str(k))
        moved = max(moved, float(np.max(np.abs(ip_[k] - np_[k]))))
    assert moved > 1e-3                      # and the responses did change (a no-op inflation would pass above)


def test_e2e_fit_reads_the_original_morphology(e2e29):
    run = e2e29["runs"]["infl"]
    for c in ("full", "full_no_noise"):
        fit = pd.read_csv(run["out"] / "fits" / ("twin_%d" % run["tid"][c]) / "phase2_results.csv")
        assert str(fit["morphology_source"].iloc[0]) == "archive"
        argv = json.loads((run["out"] / "fits" / ("twin_%d" % run["tid"][c]) / "fit_argv.json").read_text())
        flat = json.dumps(argv)
        assert "sim_morphology" not in flat and "--swc-dir" not in flat


def test_e2e_reuse_and_rebuild_rules(e2e29, group):
    """Re-running `twin` on the finished inflated twin keeps it; a re-plan (--force) at another factor makes the
    twin job fail with the reason; `twin --force` rebuilds it under the new record."""
    out = e2e29["root"] / "reuse"
    code, err = _plan(group["gdir"], group["ndir"], out, *INFL)
    assert code == 0, err
    tid = e2e29["runs"]["infl"]["tid"]["full_no_noise"]
    assert rtr.main(["twin", "--out", str(out), "--twin", str(tid)]) == 0
    meta0 = (out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid) / "metadata.json").read_text()
    assert rtr.main(["twin", "--out", str(out), "--twin", str(tid)]) == 0
    st = json.loads((out / "status" / ("twin_%d.json" % tid)).read_text())
    assert st["status"] == "ok" and st["built"] is False
    code, err = _plan(group["gdir"], group["ndir"], out, "--twin-diam-factor", "2.0", "--force")
    assert code == 0, err
    # the fit options did not change, so a finished fit would be kept: the archive check must still refuse
    assert rtr.main(["twin", "--out", str(out), "--twin", str(tid)]) == 1
    st = json.loads((out / "status" / ("twin_%d.json" % tid)).read_text())
    assert st["status"] == "build_failed" and "inflation" in st["reason"], st
    assert (out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid) / "metadata.json").read_text() == meta0
    assert rtr.main(["twin", "--out", str(out), "--twin", str(tid), "--force"]) == 0
    rec = json.loads((out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid) / "metadata.json").read_text())
    assert rec["twin"]["provenance"]["sim_morphology"]["factor"] == 2.0
    # back to no inflation: the inflated archive is refused
    code, err = _plan(group["gdir"], group["ndir"], out, "--force")
    assert code == 0, err
    assert rtr.main(["twin", "--out", str(out), "--twin", str(tid)]) == 1


def test_e2e_two_inflated_runs_are_identical(e2e29, group):
    out = e2e29["root"] / "infl2"
    code, err = _plan(group["gdir"], group["ndir"], out, *INFL)
    assert code == 0, err
    ri = e2e29["runs"]["infl"]
    tid = ri["tid"]["full"]
    assert rtr.main(["twin", "--out", str(out), "--twin", str(tid)]) == 0
    a = _arrays(_sdir(ri, "full"))
    b = _arrays(out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid))
    for k in a:
        assert_array_equal(a[k], b[k], err_msg=str(k))
    assert (ri["out"] / "twins" / ("twin_%d" % tid) / ("sim_morphology_%d.swc" % tid)).read_bytes() == \
        (out / "twins" / ("twin_%d" % tid) / ("sim_morphology_%d.swc" % tid)).read_bytes()


def test_e2e_sensitivity_builds_the_inflated_twin(e2e29, group):
    """Block 28's `sensitivity` builds a twin 'as twin does': on an inflated run a twin built first by
    `sensitivity` carries the inflation record and the original reconstruction."""
    out = e2e29["root"] / "sens"
    code, err = _plan(group["gdir"], group["ndir"], out, *INFL)
    assert code == 0, err
    man = pd.read_csv(out / "manifest.csv")
    tid = int(man.loc[man["configuration"] == "active_fit", "specimen_id"].iloc[0])
    real = int(man.loc[man["configuration"] == "active_fit", "twin_of_specimen_id"].iloc[0])
    rc = rtr.main(["sensitivity", "--run-dir", str(out), "--twin", str(tid), "--n-noise", "3"])
    assert rc == 0
    sd = out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid)
    rec = json.loads((sd / "metadata.json").read_text())["twin"]["provenance"]["sim_morphology"]
    assert rec["factor"] == F29 and rec["max_path_um"] == L29
    assert (sd / "reconstruction.swc").read_bytes() == \
        (group["gdir"] / ("specimen_%d" % real) / "reconstruction.swc").read_bytes()
    ident = out / "sensitivity" / ("twin_%d" % tid) / "identifiability.csv"
    assert ident.is_file()
    df = pd.read_csv(ident)
    sd_cols = [c for c in df.columns if c.startswith("sd_lin_")]
    assert sd_cols and np.all(np.isfinite(df[sd_cols].to_numpy(float)))


def test_e2e_merge_on_the_inflated_run(e2e29):
    out = e2e29["runs"]["infl"]["out"]
    assert rtr.main(["merge", "--out", str(out), "--no-plot"]) == 0
    st = pd.read_csv(out / "twins_status.csv")
    assert set(st["status"]) >= {"ok"}


def test_build_twin_refuses_a_missing_simulation_morphology(e2e29, tmp_path):
    """Block 29: twin_build.build_twin(..., sim_swc_path=<missing>) raises TwinBuildError and writes nothing."""
    import dataclasses
    import run_ih_fit as R
    import twin_build as TB
    import twin_manifest as TM
    import twin_table as TT
    from twin_config import TwinConfig
    run = rtr.load_run(e2e29["runs"]["infl"]["out"])
    row = run["manifest"].set_index("specimen_id", drop=False).loc[e2e29["runs"]["infl"]["tid"]["full"]]
    opts = run["options"]
    spec = TM.twin_spec_of(row, F=float(opts["F"]))
    argv = rtr.fit_argv(opts, tmp_path, tmp_path / "fit", str(row["arm"]))
    ih_spec, pspec = rtr.twin_model(R._parse_args(argv), str(row["configuration"]))
    table = TT.read_twin_sweep_table(run["paths"]["table"])
    real_dir = Path(run["config"]["group_dir"]) / ("specimen_%d" % int(row["twin_of_specimen_id"]))
    tcfg = dataclasses.replace(TwinConfig(), base_seed=int(opts["base_seed"]))
    gdir = tmp_path / "g"
    with pytest.raises(TB.TwinBuildError):
        TB.build_twin(spec, real_dir, table, gdir, ih_spec=ih_spec, apply_theta=pspec.apply, cfg=tcfg,
                      sim_swc_path=tmp_path / "nope.swc")
    assert not (gdir / ("specimen_%d" % spec.twin_id)).exists()
