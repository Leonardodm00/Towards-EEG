"""Spec-derived tests of Block 17 (S3a: the node sample, the coverage of the
phantoms' band, the diameters by layer and class) -- written by the
integration tester (run D5, 2026-11-13) from specs/SPEC.md, not from the
code. Expected values: numpy (quantile, log), spec_helpers_D5, or by hand.

Run (from the workstream directory):
    MPLBACKEND=Agg python -m pytest tests/spec/test_spec_block17_s3a.py -q
"""
from __future__ import annotations

import dataclasses
import math
import random

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spec_helpers_D5 as H

from allen_diameter.analysis import coverage as COV
from allen_diameter.analysis import s3_nodes as S3N
from allen_diameter.config import default_config
from allen_diameter.loading import swc_io

NAN = float("nan")
MU_LO, MU_HI = 0.3, 3.0          # phantom.mu_range_per_um (D-065.1); checked below against the configuration


def cfg_survey(**kw):
    cfg = default_config()
    return dataclasses.replace(cfg, survey=dataclasses.replace(cfg.survey, **kw))


def test_configuration_values_used_here():
    cfg = default_config()
    assert tuple(cfg.phantom.mu_range_per_um) == (MU_LO, MU_HI)
    assert cfg.survey.nodes_allen_bins_um == (0.4, 0.6, 0.8, 1.2, 2.0) and cfg.survey.nodes_per_bin == 20
    assert cfg.survey.coverage_dhat_bins_um == (0.2, 0.4, 0.6, 0.8, 1.2, 2.0, 4.0)
    assert (cfg.survey.coverage_min_nodes, cfg.survey.coverage_pass_share) == (20, 0.95)


# ================================================================ 17.1 =====

def test_bins_of_allen_diameter_edges_and_bad_radii(tmp_path):
    # 17.1: bins [e_b, e_b+1) of D_A = 2 r with edges (0, 0.4, 0.6, 0.8, 1.2, 2, inf); a non-finite or negative
    # radius belongs to no bin
    radii = [0.0, 0.1999, 0.2, 0.3, 0.6, 1.0, 5.0, -0.1, NAN, math.inf]     # 4 decimals: the SWC writer's format
    path, br = H.star_swc(str(tmp_path / "a.swc"), [[r] for r in radii])
    swc = swc_io.read_swc(path)
    ids, counts = S3N.stratified_nodes(swc, default_config(), 7)
    want = {}
    for r in radii:
        b = H.ref_allen_bin(r, (0.4, 0.6, 0.8, 1.2, 2.0))
        if b is not None:
            want[b] = want.get(b, 0) + 1
    assert {b: c[0] for b, c in counts.items()} == want == {0: 2, 1: 1, 2: 1, 4: 1, 5: 2}
    bad = {br[7][0], br[8][0], br[9][0]}
    assert not (set(ids) & bad) and len(ids) == len(set(ids)) == 7


def _bin2_swc(tmp_path, lens=(10, 3, 7, 1), extra=True, seed=0):
    """Branches whose nodes lie in bin 3 (D_A in [0.8, 1.2): r = 0.5), interleaved with bin-0 nodes (r = 0.1)."""
    rng = random.Random(seed)
    radii = []
    for n in lens:
        rs = [0.5] * n + ([0.1] * rng.randint(0, 3) if extra else [])
        rng.shuffle(rs)
        radii.append(rs)
    path, br = H.star_swc(str(tmp_path / ("b%d.swc" % seed)), radii)
    return path, br, radii


def test_stratified_equals_reference_and_round_robin(tmp_path):
    path, br, radii = _bin2_swc(tmp_path)
    swc = swc_io.read_swc(path)
    r_of = {nid: r for ids, rs in zip(br, radii) for nid, r in zip(ids, rs)}
    for sid, seed in ((7, 20261026), (529878215, 3)):
        cfg = cfg_survey(selection_seed=seed)
        ids, counts = S3N.stratified_nodes(swc, cfg, sid)
        ref_ids, ref_counts = H.ref_stratified(br, r_of, (0.4, 0.6, 0.8, 1.2, 2.0), seed, sid, 20)
        assert ids == ref_ids and {b: tuple(c) for b, c in counts.items()} == ref_counts
        b2 = [i for i in ids if r_of[i] == 0.5]
        assert tuple(counts[3]) == (21, 20) and len(b2) == 20 and len(set(ids)) == len(ids)
        taken = [sum(1 for i in b2 if i in set(run)) for run in br]
        avail = [sum(1 for i in run if r_of[i] == 0.5) for run in br]
        for s in range(len(br)):                       # round-robin: an unexhausted stretch is never behind by 2
            if taken[s] < avail[s]:
                assert all(taken[t] <= taken[s] + 1 for t in range(len(br))), (taken, avail)


def test_stratified_determinism_and_seed(tmp_path):
    path, br, radii = _bin2_swc(tmp_path, lens=(12, 9, 6), seed=1)
    swc = swc_io.read_swc(path)
    a = S3N.stratified_nodes(swc, default_config(), 11)
    assert a == S3N.stratified_nodes(swc, default_config(), 11)
    b = S3N.stratified_nodes(swc, cfg_survey(selection_seed=1), 11)
    r_of = {nid: r for ids, rs in zip(br, radii) for nid, r in zip(ids, rs)}
    big = lambda res: sorted(i for i in res[0] if r_of[i] == 0.5)              # noqa: E731  27 > 20 available
    small = lambda res: sorted(i for i in res[0] if r_of[i] == 0.1)            # noqa: E731  all taken
    assert big(a) != big(b) and small(a) == small(b)


def test_all_nodes_dendrite_types_only(tmp_path):
    path, br = H.star_swc(str(tmp_path / "t.swc"), [[0.5, 0.5], [0.5], [0.4, 0.4, 0.4], [0.3]],
                          types=[3, 2, 4, 1])
    swc = swc_io.read_swc(path)
    want = sorted(br[0] + br[2])
    assert sorted(S3N.all_nodes(swc, default_config())) == want
    ids, counts = S3N.sample_nodes(swc, cfg_survey(nodes_mode="all"), 3)
    assert sorted(ids) == want


# ================================================================ 17.3 =====

def _row(nid, d, mu, alpha=None, status="converged", flags="", reg="", r_allen=0.3, in_S=True, reject=""):
    return dict(node_id=nid, d_hat_um=d, mu_hat_per_um=mu, alpha_hat=mu * d if alpha is None else alpha,
                fit_status=status, flags=flags, reg_verdict=reg, allen_radius_um=r_allen, in_S=in_S, reject=reject)


def _cell(sid, cls="exc", layer="3"):
    return dict(specimen_id=sid, neuron_class=cls, layer=layer)


def _hand_rows():
    """Two cells. d in [0.4, 0.6): 20 nodes, 1 above the band (mu 5); [0.6, 0.8): 20 nodes, 2 below (mu 0.1);
    [1.2, 2.0): 19 nodes, all above (mu 4); plus nodes outside S that must change nothing."""
    rng = np.random.default_rng(3)
    a, b = [], []
    nid = 1
    for k in range(20):
        mu = 5.0 if k == 0 else float(rng.uniform(0.5, 2.5))
        (a if k % 2 else b).append(_row(nid, float(rng.uniform(0.41, 0.59)), mu))
        nid += 1
    for k in range(20):
        mu = 0.1 if k < 2 else float(rng.uniform(0.5, 2.5))
        a.append(_row(nid, float(rng.uniform(0.61, 0.79)), mu))
        nid += 1
    for k in range(19):
        b.append(_row(nid, float(rng.uniform(1.3, 1.9)), 4.0))
        nid += 1
    a.append(_row(nid, 0.5, 9.0, flags="crossing"))                       # outside S: ignored
    b.append(_row(nid + 1, 0.7, 0.01, status="at_bound"))                 # outside S: ignored
    return [(_cell(1), a), (_cell(2, "inh", "5"), b)]


def _by_lo(bins):
    return {round(b["lo"], 6): b for b in bins}


def test_coverage_by_hand_shares_quantiles_and_rule():
    cfg = default_config()
    rbc = _hand_rows()
    nodes = COV.node_table(rbc)
    bins = _by_lo(COV.coverage_table(nodes, cfg))
    assert sorted(bins) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.2, 2.0, 4.0]
    assert math.isinf(bins[4.0]["hi"]) and bins[0.0]["hi"] == 0.2
    b4, b6, b12 = bins[0.4], bins[0.6], bins[1.2]
    assert (b4["n"], b6["n"], b12["n"]) == (20, 20, 19)
    assert (b4["n_cells"], b6["n_cells"], b12["n_cells"]) == (2, 1, 1)
    assert b4["share_inside"] == 0.95 and b4["share_above"] == 0.05 and b4["status"] == "pass"
    assert b6["share_inside"] == 0.9 and b6["share_below"] == 0.1 and b6["status"] == "fail"
    assert b12["share_above"] == 1.0 and b12["status"] == "too_few"
    # quantiles: numpy.quantile (linear) of ln(mu d) and ln(alpha) over the bin's nodes in S
    sel = [r for _, rows in rbc for r in rows if 0.4 <= r["d_hat_um"] < 0.6 and not r["flags"]
           and r["fit_status"] == "converged"]
    want = np.quantile(np.log([r["mu_hat_per_um"] * r["d_hat_um"] for r in sel]), (0.05, 0.25, 0.5, 0.75, 0.95))
    assert_allclose(b4["q_ln_mud"], want, rtol=1e-12)
    want_a = np.quantile(np.log([r["alpha_hat"] for r in sel]), (0.05, 0.25, 0.5, 0.75, 0.95))
    assert_allclose(b4["q_ln_alpha"], want_a, rtol=1e-12)
    v = COV.verdict(list(bins.values()), cfg)
    assert v["passed"] is False and [bins_lo for bins_lo in v["failing"]] == [b6["bin"]]
    assert b12["bin"] in v["too_few"]


def test_band_holds_exactly_at_the_mu_range_ends():
    # 17.3: the analytic band "holds exactly when mu_min <= mu_hat <= mu_max", so a node at mu_hat = mu_min or
    # mu_max is inside at every d_hat
    cfg = default_config()
    d = np.linspace(0.21, 3.9, 400)
    rows = [_row(k, float(x), mu) for k, (x, mu) in enumerate([(x, MU_LO) for x in d] + [(x, MU_HI) for x in d])]
    nodes = COV.node_table([(_cell(1), rows)])
    out = {"below": 0, "above": 0}
    for b in COV.coverage_table(nodes, cfg):
        out["below"] += int(round(b["share_below"] * b["n"])) if b["n"] else 0
        out["above"] += int(round(b["share_above"] * b["n"])) if b["n"] else 0
    assert out == {"below": 0, "above": 0}, "%s of %d nodes on the band's edges classed outside" % (out, len(rows))


def test_phantom_band_quantiles_and_no_phantoms():
    cfg = default_config()
    rng = np.random.default_rng(4)
    pd_ = np.r_[rng.uniform(0.41, 0.59, 25), rng.uniform(0.61, 0.79, 19)]
    pm = np.exp(rng.normal(-0.5, 0.6, pd_.size))
    nodes = COV.node_table(_hand_rows())
    bins = _by_lo(COV.coverage_table(nodes, cfg, dict(d_hat=pd_, mud=pm)))
    lo, hi = np.quantile(np.log(pm[:25]), (0.005, 0.995))
    assert_allclose((bins[0.4]["phantom_lo"], bins[0.4]["phantom_hi"]), (lo, hi), rtol=1e-12)
    real = [n for n in nodes if n["in_S"] and 0.4 <= n["d_hat"] < 0.6]
    lnm = np.log([n["mu_hat"] * n["d_hat"] for n in real])
    assert_allclose(bins[0.4]["share_inside_phantom"], np.mean((lnm >= lo) & (lnm <= hi)), rtol=1e-12)
    assert math.isnan(bins[0.6]["share_inside_phantom"]) and bins[0.6]["status_phantom"] == "no_phantoms"


def test_node_table_recomputes_S_from_columns():
    # D-098.3 / D-096 / D-030: faint and dark are labels; crossing, stack_edge, bbar_few, profile_nan, a status
    # other than converged, a registration category other than ON fail S; reasons in reject_reasons' order
    rows = [_row(1, 0.5, 1.0, flags="faint", in_S=False, reject="faint"),
            _row(2, 0.5, 1.0, flags="crossing;faint", in_S=False, reject="faint;crossing"),
            _row(3, 0.5, 1.0, status="at_bound", flags="profile_nan;stack_edge"),
            _row(4, 0.5, 1.0, reg="ALONGSIDE: a ridge +0.40 um to the side -- snap needed"),
            _row(5, 0.5, 1.0, reg="ON THE PROCESS (lateral offset +0.10 um)", flags="dark;steep;iter_max_rounds"),
            _row(6, 0.5, 1.0, flags=NAN, reg=NAN),
            dict(node_id=7, d_hat_um=0.5, mu_hat_per_um=1.0, alpha_hat=0.5, allen_radius_um=0.3, in_S="False",
                 reject="faint")]
    nodes = {n["node_id"]: n for n in COV.node_table([(_cell(1), rows)])}
    assert nodes[1]["in_S"] and nodes[1]["reject"] == "" and not nodes[1]["in_S_stored"]
    assert not nodes[2]["in_S"] and nodes[2]["reject"] == "crossing"
    assert nodes[3]["reject"] == "status:at_bound;stack_edge;profile_nan"
    assert nodes[4]["reject"] == "registration:ALONGSIDE"
    assert nodes[5]["in_S"] and nodes[6]["in_S"]
    assert nodes[7]["s_source"] == "stored" and nodes[7]["in_S"] is False and nodes[7]["reject"] == "faint"
    assert all(nodes[k]["s_source"] == "recomputed" for k in range(1, 7))
    s = COV.selection_summary(list(nodes.values()))
    assert (s["n_rows"], s["n_recomputed"], s["n_stored_used"], s["n_now_in_S"], s["n_now_out_of_S"]) == (7, 6, 1, 1, 2)


def test_invariants_row_order_and_nodes_outside_S():
    cfg = default_config()
    rbc = _hand_rows()
    base = COV.coverage_table(COV.node_table(rbc), cfg)
    rng = random.Random(5)
    shuffled = [(c, rng.sample(rows, len(rows))) for c, rows in reversed(rbc)]
    extra = [(c, rows + [_row(900 + i, 0.5, 9.0, flags="bbar_few") for i in range(5)]) for c, rows in rbc]
    for other in (shuffled, extra):
        got = COV.coverage_table(COV.node_table(other), cfg)
        for x, y in zip(base, got):
            for k in ("n", "n_cells", "share_inside", "share_below", "share_above", "status"):
                assert x[k] == y[k] or (x[k] != x[k] and y[k] != y[k]), (k, x, y)
            assert_allclose(x["q_ln_mud"], y["q_ln_mud"], rtol=1e-12)


def test_diameter_table_by_hand():
    rows1 = [_row(1, 0.5, 1.0, r_allen=0.2), _row(2, 0.7, 1.0, r_allen=0.4), _row(3, 0.9, 1.0, flags="crossing",
                                                                                 r_allen=0.5)]
    rows2 = [_row(4, 1.5, 1.0, r_allen=0.6), _row(5, NAN, 1.0, status="none", r_allen=0.7)]
    nodes = COV.node_table([(_cell(1, "exc", "3"), rows1), (_cell(2, "inh", "3"), rows2)])
    tab = {(r["neuron_class"], r["layer"]): r for r in COV.diameter_table(nodes)}
    assert set(tab) == {("exc", "3"), ("inh", "3"), ("exc", "all"), ("inh", "all"), ("all", "3")}
    Q = (0.05, 0.25, 0.5, 0.75, 0.95)
    assert_allclose(tab[("exc", "3")]["q_d_hat"], np.quantile([0.5, 0.7], Q), rtol=1e-12)        # in S only
    assert_allclose(tab[("exc", "3")]["q_D_allen"], np.quantile([0.4, 0.8, 1.0], Q), rtol=1e-12)  # every node
    assert_allclose(tab[("all", "3")]["q_d_hat"], np.quantile([0.5, 0.7, 1.5], Q), rtol=1e-12)
    assert tab[("all", "3")]["n_cells"] == 2 and tab[("all", "3")]["n"] == 5
    assert_allclose(tab[("inh", "all")]["q_D_allen"], np.quantile([1.2, 1.4], Q), rtol=1e-12)


def test_outside_counts_by_reason():
    rows = [_row(1, 0.5, 1.0, flags="crossing;stack_edge"), _row(2, 0.5, 1.0, status="failed"),
            _row(3, 0.5, 1.0, flags="crossing"), _row(4, 0.5, 1.0)]
    out = COV.outside_counts(COV.node_table([(_cell(1), rows)]))
    assert out == {"crossing": 2, "stack_edge": 1, "status:failed": 1}
