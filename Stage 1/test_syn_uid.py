"""test_syn_uid -- smoke test for syn_uid.

RUN
---
    python3 test_syn_uid.py
    python3 test_syn_uid.py raw_synapses.csv neuron_skeleton.csv

The two-argument form additionally runs collapse_report on real data and
prints how many synapses the nearest-node mapping loses for that cell.

Test list:
  1  identifiers are unique and stable across repeated calls
  2  identifiers survive row shuffling when there are no duplicate keys
     (the property the whole module exists for)
  3  identifiers survive filtering: dropping rows does not change the
     identifiers of the rows that remain
  4  duplicate coordinate+type rows get distinct identifiers, and the report
     says how many identifiers are order-dependent
  5  identifiers depend on nid, on coordinates, and on type -- changing any
     one of them changes the digest
  6  a CSV round trip does not change any identifier (float formatting is
     defeated by rounding to integer nm)
  7  refusing to reissue: calling twice on the same frame raises
  8  voxel scaling is applied (voxel input and pre-scaled nm input agree)
  9  collapse_report counts a hand-built collision correctly
 10  mapped-frame variant flags itself as identifying a node, not a synapse
"""

import io
import sys

import numpy as np
import pandas as pd

import syn_uid as su


FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s   %s" % (name, detail))
        FAILURES.append(name)


def raw_frame(n=12, seed=0):
    """A synthetic raw H01-style export in voxel coordinates."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "location_x": rng.randint(0, 100000, n).astype(float),
        "location_y": rng.randint(0, 100000, n).astype(float),
        "location_z": rng.randint(0, 3000, n).astype(float),
        "synapse_type": rng.choice([1, 2], n),
        "direction": ["incoming"] * n,
        "partner_id": rng.randint(0, 10 ** 12, n),
    })


def test_unique_and_stable():
    print("\n[1] unique and stable across repeated calls")
    df = raw_frame()
    a, ra = su.assign_syn_uid(df, nid=12345)
    b, rb = su.assign_syn_uid(df, nid=12345)
    check("unique", ra["uid_is_unique"])
    check("n_rows preserved", ra["n_rows"] == len(df))
    check("repeat call identical",
          list(a["syn_uid"]) == list(b["syn_uid"]))
    check("no duplicate keys in this fixture",
          ra["n_keys_with_multiplicity"] == 0,
          "got %d" % ra["n_keys_with_multiplicity"])


def test_shuffle_invariance():
    print("\n[2] invariant under row shuffling")
    df = raw_frame(seed=1)
    a, _ = su.assign_syn_uid(df, nid=7)
    sh = df.sample(frac=1.0, random_state=42)
    b, _ = su.assign_syn_uid(sh, nid=7)
    m = dict(zip(b["partner_id"], b["syn_uid"]))
    same = all(m[p] == u for p, u in zip(a["partner_id"], a["syn_uid"]))
    check("same synapse -> same uid after shuffle", same)


def test_filter_invariance():
    print("\n[3] invariant under filtering")
    df = raw_frame(seed=2)
    a, _ = su.assign_syn_uid(df, nid=7)
    sub = df.iloc[[0, 3, 5, 9]]
    b, _ = su.assign_syn_uid(sub, nid=7)
    expect = [a["syn_uid"].iloc[i] for i in (0, 3, 5, 9)]
    check("surviving rows keep their uid", list(b["syn_uid"]) == expect,
          "%s vs %s" % (list(b["syn_uid"]), expect))


def test_duplicates():
    print("\n[4] duplicate rows get distinct uids and are reported")
    df = raw_frame(n=5, seed=3)
    dup = pd.concat([df, df.iloc[[1]], df.iloc[[1]]], ignore_index=True)
    out, rep = su.assign_syn_uid(dup, nid=7)
    check("all uids distinct", rep["uid_is_unique"])
    check("one key has multiplicity", rep["n_keys_with_multiplicity"] == 1,
          "got %d" % rep["n_keys_with_multiplicity"])
    check("max multiplicity is 3", rep["max_multiplicity"] == 3,
          "got %d" % rep["max_multiplicity"])
    check("order-dependent rows counted", rep["n_rows_order_dependent"] == 3,
          "got %d" % rep["n_rows_order_dependent"])


def test_key_sensitivity():
    print("\n[5] uid changes when any key component changes")
    df = raw_frame(n=4, seed=4)
    base, _ = su.assign_syn_uid(df, nid=7)
    other_nid, _ = su.assign_syn_uid(df, nid=8)
    check("nid changes the digest",
          base["syn_uid"].iloc[0] != other_nid["syn_uid"].iloc[0])
    moved = df.copy()
    moved.loc[0, "location_x"] = moved.loc[0, "location_x"] + 1.0
    out, _ = su.assign_syn_uid(moved, nid=7)
    check("coordinate changes the digest",
          base["syn_uid"].iloc[0] != out["syn_uid"].iloc[0])
    check("other rows unaffected",
          list(base["syn_uid"])[1:] == list(out["syn_uid"])[1:])
    retyped = df.copy()
    retyped.loc[0, "synapse_type"] = 99
    out2, _ = su.assign_syn_uid(retyped, nid=7)
    check("type changes the digest",
          base["syn_uid"].iloc[0] != out2["syn_uid"].iloc[0])
    # sub-nanometre jitter must NOT change it: that is what rounding buys
    jitter = df.copy()
    jitter.loc[0, "location_x"] = jitter.loc[0, "location_x"] + 1e-9
    out3, _ = su.assign_syn_uid(jitter, nid=7)
    check("sub-nm jitter does not change the digest",
          base["syn_uid"].iloc[0] == out3["syn_uid"].iloc[0])


def test_csv_roundtrip():
    print("\n[6] CSV round trip preserves every uid")
    df = raw_frame(seed=5)
    a, _ = su.assign_syn_uid(df, nid=7)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    df2 = pd.read_csv(buf)
    b, _ = su.assign_syn_uid(df2, nid=7)
    check("identical after round trip", list(a["syn_uid"]) == list(b["syn_uid"]))


def test_refuse_reissue():
    print("\n[7] refuses to reissue over an existing column")
    df = raw_frame(n=3, seed=6)
    a, _ = su.assign_syn_uid(df, nid=7)
    try:
        su.assign_syn_uid(a, nid=7)
        check("raises on reissue", False, "no exception")
    except ValueError:
        check("raises on reissue", True)


def test_voxel_scaling():
    print("\n[8] voxel scaling is applied")
    df = raw_frame(n=6, seed=7)
    a, _ = su.assign_syn_uid(df, nid=7, voxel_scale=(8.0, 8.0, 33.0))
    nm = df.copy()
    nm["location_x"] = nm["location_x"] * 8.0
    nm["location_y"] = nm["location_y"] * 8.0
    nm["location_z"] = nm["location_z"] * 33.0
    b, _ = su.assign_syn_uid(nm, nid=7, voxel_scale=(1.0, 1.0, 1.0))
    check("voxel and pre-scaled nm agree",
          list(a["syn_uid"]) == list(b["syn_uid"]))
    c, _ = su.assign_syn_uid(df, nid=7, voxel_scale=(1.0, 1.0, 1.0))
    check("wrong scale gives different uids",
          list(a["syn_uid"]) != list(c["syn_uid"]))


def test_collapse_report():
    print("\n[9] collapse_report counts a hand-built collision")
    skel = pd.DataFrame({
        "id": [0, 1, 2],
        "p": [-1, 0, 1],
        "x": [0.0, 1000.0, 2000.0],
        "y": [0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0],
    })
    # four synapses: two land on node 1 with DIFFERENT types (discordant),
    # one on node 1 again (same type as one of them), one on node 2
    syn = pd.DataFrame({
        "location_x": [125.0, 126.0, 124.0, 250.0],
        "location_y": [0.0, 0.0, 0.0, 0.0],
        "location_z": [0.0, 0.0, 0.0, 0.0],
        "synapse_type": [1, 2, 1, 2],
        "direction": ["incoming"] * 4,
    })
    rep = su.collapse_report(syn, skel, voxel_scale=(8.0, 8.0, 33.0))
    check("4 synapses", rep["n_synapses"] == 4, "got %d" % rep["n_synapses"])
    check("2 distinct nodes", rep["n_distinct_nodes"] == 2,
          "got %d" % rep["n_distinct_nodes"])
    check("2 synapses lost", rep["n_synapses_lost"] == 2,
          "got %d" % rep["n_synapses_lost"])
    check("1 node discordant", rep["n_nodes_discordant"] == 1,
          "got %d" % rep["n_nodes_discordant"])
    check("3 synapses on the discordant node",
          rep["n_synapses_discordant"] == 3,
          "got %d" % rep["n_synapses_discordant"])


def test_mapped_variant():
    print("\n[10] mapped-frame variant flags what it identifies")
    mapped = pd.DataFrame({
        "nid": [111, 111, 111],
        "node_id": [4, 9, 12],
        "x": [1.5, 2.5, 3.5],
        "y": [0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0],
        "synapse_label": ["exc_syn", "inh_syn", "exc_syn"],
    })
    out, rep = su.assign_syn_uid_mapped(mapped)
    check("uid column added", "syn_uid" in out.columns)
    check("unique", rep["uid_is_unique"])
    check("nid inferred from the frame", rep["nid"] == 111,
          "got %r" % rep["nid"])
    check("identifies == node", rep.get("identifies") == "node")
    check("carries a warning", "warning" in rep)


def run_on_real(raw_path, skel_path):
    print("\n[real] %s  x  %s" % (raw_path, skel_path))
    raw = pd.read_csv(raw_path)
    skel = pd.read_csv(skel_path)
    print("      raw columns: %r" % list(raw.columns))
    out, rep = su.assign_syn_uid(raw, nid=skel.get("nid", pd.Series([None]))
                                 .iloc[0] if "nid" in skel.columns else "NA")
    for k in sorted(rep):
        print("      %-28s %s" % (k, rep[k]))
    crep = su.collapse_report(raw, skel)
    print("      -- collapse --")
    for k in sorted(crep):
        v = crep[k]
        print("      %-28s %s" % (k, ("%.6g" % v) if isinstance(v, float)
                                  else v))


def main():
    print("syn_uid smoke test  (%s)" % su.MODULE_VERSION)
    test_unique_and_stable()
    test_shuffle_invariance()
    test_filter_invariance()
    test_duplicates()
    test_key_sensitivity()
    test_csv_roundtrip()
    test_refuse_reissue()
    test_voxel_scaling()
    test_collapse_report()
    test_mapped_variant()

    if len(sys.argv) >= 3:
        run_on_real(sys.argv[1], sys.argv[2])

    print("\n%s" % ("-" * 62))
    if FAILURES:
        print("FAILED %d check(s): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
