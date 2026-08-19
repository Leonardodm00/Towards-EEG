"""test_truncation_flag -- smoke test for truncation_flag.

RUN
---
    python3 test_truncation_flag.py
    python3 test_truncation_flag.py neuron_A_spines.csv neuron_B_spines.csv ...

The multi-argument form pools the given labelled frames with pool_axis_bounds,
runs build_truncation_table on each, and prints the per-cell summary plus the
pooled z_range_fraction_of_slab sanity check.

Test list:
  1  a real tapering ending is NOT flagged by taper alone
  2  an abrupt cut (no taper) IS flagged by taper alone
  3  a stub too short to judge is taper_determinate == False, not silently
     scored either way
  4  the reference-radius interpolation matches a hand-computed value on a
     two-segment neck
  5  pool_axis_bounds recovers the true min/max on a synthetic multi-cell set
  6  pool_axis_bounds excludes axon nodes by default and includes them only
     when asked
  7  flag_by_boundary finds a tip near the pooled z bound and correctly
     reports the per-axis distances
  8  combine_flags: taper=True, z=False -> basis 'taper'
  9  combine_flags: taper=False (determinate, real taper), z=True -> basis
     is still driven correctly (NOT flagged, since taper is determinate and
     says "real ending" -- boundary evidence does not override a determinate
     taper verdict of "not cut")
 10  combine_flags: taper=None (indeterminate), z=True -> basis 'z_boundary',
     is_truncated True
 11  combine_flags: taper=None, z=False -> basis 'unresolved', is_truncated
     False (absence of evidence is not evidence of truncation)
 12  x/y proximity never drives is_truncated on its own
 13  cell_truncation_summary counts and median are self-consistent
 14  determinism under row shuffling
 15  empty input (no dendrite tips) returns an empty table cleanly
"""

import math
import sys

import numpy as np
import pandas as pd

import spine_density as sd
import truncation_flag as tf


FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s   %s" % (name, detail))
        FAILURES.append(name)


def approx(a, b, rtol=1e-9, atol=0.0):
    return abs(a - b) <= atol + rtol * abs(b)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
def make_frame(rows):
    return pd.DataFrame(rows, columns=["id", "p", "x", "y", "z", "r",
                                       "annotated_type"])


def straight_dendrite(n_segments=6, spacing_um=2.0, radius_um=0.5,
                      start_id=0, base_xyz=(0.0, 0.0, 0.0), axis=(1, 0, 0),
                      soma=True):
    """A soma (optional) followed by a straight run of n_segments dendrite
    nodes. Returns (rows, next_free_id, tip_id).
    """
    ux, uy, uz = axis
    x0, y0, z0 = base_xyz
    rows = []
    prev = None
    nid = start_id
    if soma:
        rows.append((nid, -1, x0, y0, z0, 5.0, "soma"))
        prev = nid
        nid += 1
        d0 = 1
    else:
        rows.append((nid, -1, x0, y0, z0, radius_um, "dendrite"))
        prev = nid
        nid += 1
        d0 = 1
    for k in range(1, n_segments + 1):
        dist = d0 * 0 + k * spacing_um if False else k * spacing_um
        rows.append((nid, prev, x0 + ux * dist, y0 + uy * dist,
                    z0 + uz * dist, radius_um, "dendrite"))
        prev = nid
        nid += 1
    tip_id = prev
    return rows, nid, tip_id


def apply_taper(rows, tip_id, taper_r_um):
    """Overwrite the tip node's own radius (last dendrite node) to simulate a
    real tapering ending."""
    out = []
    for row in rows:
        if row[0] == tip_id:
            row = (row[0], row[1], row[2], row[3], row[4], taper_r_um, row[6])
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
def test_real_ending_not_flagged():
    print("\n[1] real tapering ending is not flagged by taper")
    rows, nid, tip = straight_dendrite(n_segments=6, spacing_um=2.0,
                                       radius_um=0.5)
    rows = apply_taper(rows, tip, taper_r_um=0.08)   # 0.08 / 0.5 = 0.16
    df = make_frame(rows)
    taper = tf.build_taper_table(df, input_units="um")
    row = taper[taper["tip_node_id"] == tip].iloc[0]
    check("taper_determinate", bool(row["taper_determinate"]))
    check("taper_ratio well below threshold",
          row["taper_ratio"] < tf.DEFAULT_TAPER_RATIO_THRESHOLD,
          "got %.3f" % row["taper_ratio"])
    flagged = tf.flag_by_taper(taper)
    fr = flagged[flagged["tip_node_id"] == tip].iloc[0]
    check("taper_flag_truncated is False", fr["taper_flag_truncated"] == False,
          "got %r" % fr["taper_flag_truncated"])


def test_abrupt_cut_flagged():
    print("\n[2] abrupt cut (no taper) is flagged")
    rows, nid, tip = straight_dendrite(n_segments=6, spacing_um=2.0,
                                       radius_um=0.5)
    # no taper applied: tip radius == shaft radius throughout
    df = make_frame(rows)
    taper = tf.build_taper_table(df, input_units="um")
    flagged = tf.flag_by_taper(taper)
    row = flagged[flagged["tip_node_id"] == tip].iloc[0]
    check("taper_ratio approx 1", approx(row["taper_ratio"], 1.0, rtol=1e-9),
          "got %.6f" % row["taper_ratio"])
    check("taper_flag_truncated is True", row["taper_flag_truncated"] == True,
          "got %r" % row["taper_flag_truncated"])


def test_short_stub_indeterminate():
    print("\n[3] short stub is indeterminate, not silently scored")
    rows, nid, tip = straight_dendrite(n_segments=1, spacing_um=0.5,
                                       radius_um=0.5)
    df = make_frame(rows)
    taper = tf.build_taper_table(
        df, input_units="um", reference_path_um=tf.DEFAULT_REFERENCE_PATH_UM)
    row = taper[taper["tip_node_id"] == tip].iloc[0]
    check("path_length_available < min",
          row["path_length_available_um"] < tf.DEFAULT_MIN_PATH_FOR_TAPER_UM,
          "got %.3f" % row["path_length_available_um"])
    check("taper_determinate is False", not bool(row["taper_determinate"]))
    flagged = tf.flag_by_taper(taper)
    fr = flagged[flagged["tip_node_id"] == tip].iloc[0]
    check("taper_flag_truncated is pd.NA",
          fr["taper_flag_truncated"] is pd.NA,
          "got %r" % fr["taper_flag_truncated"])


def test_reference_interpolation():
    print("\n[4] reference-radius interpolation matches hand computation")
    # soma(0) -> 1 (2um, r=0.40) -> 2 (2um, r=0.30) -> 3 (2um, r=0.20, tip)
    rows = [
        (0, -1, 0.0, 0.0, 0.0, 5.0, "soma"),
        (1, 0, 2.0, 0.0, 0.0, 0.40, "dendrite"),
        (2, 1, 4.0, 0.0, 0.0, 0.30, "dendrite"),
        (3, 2, 6.0, 0.0, 0.0, 0.20, "dendrite"),
    ]
    df = make_frame(rows)
    # reference_path_um = 3.0: from tip (node 3, dist 0) back through node 2
    # (dist 2.0, r=0.30) then 1.0 further into the 2->1 segment (r 0.30->0.40)
    # linear interp at 1.0/2.0 of the way -> r = 0.30 + 0.5*(0.40-0.30) = 0.35
    taper = tf.build_taper_table(df, input_units="um", reference_path_um=3.0)
    row = taper[taper["tip_node_id"] == 3].iloc[0]
    check("r_ref == 0.35 (hand-computed)", approx(row["r_ref_um"], 0.35,
                                                  rtol=1e-9),
          "got %.6f" % row["r_ref_um"])
    check("path_length_available == 3.0",
          approx(row["path_length_available_um"], 3.0, rtol=1e-9),
          "got %.6f" % row["path_length_available_um"])


def test_pool_axis_bounds_recovers_extent():
    print("\n[5] pool_axis_bounds recovers the true min/max")
    frames = []
    extents = []
    for k, ax in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        rows, _, tip = straight_dendrite(
            n_segments=5, spacing_um=10.0, radius_um=0.3,
            base_xyz=(0.0, 0.0, 0.0), axis=ax)
        frames.append((k, make_frame(rows)))
    bounds = tf.pool_axis_bounds(frames, input_units="um")
    check("x max == 50.0", approx(bounds["x"][1], 50.0, rtol=1e-9),
          "got %r" % (bounds["x"],))
    check("y max == 50.0", approx(bounds["y"][1], 50.0, rtol=1e-9),
          "got %r" % (bounds["y"],))
    check("z max == 50.0", approx(bounds["z"][1], 50.0, rtol=1e-9),
          "got %r" % (bounds["z"],))
    check("x/y/z min == 0.0 (soma at origin)",
          all(approx(bounds[a][0], 0.0, atol=1e-9) for a in "xyz"))
    check("n_cells == 3", bounds["n_cells"] == 3)
    check("z_range_fraction_of_slab computed",
          approx(bounds["z_range_fraction_of_slab"],
                 50.0 / tf.H01_SLAB_THICKNESS_UM, rtol=1e-9))


def test_pool_axis_bounds_axon_exclusion():
    print("\n[6] pool_axis_bounds excludes axon nodes by default")
    rows, nid, _ = straight_dendrite(n_segments=3, spacing_um=5.0,
                                     radius_um=0.3)
    # add a long axon reaching far past the dendrite
    rows.append((nid, 0, 500.0, 0.0, 0.0, 0.1, "axon"))
    df = make_frame(rows)
    b_default = tf.pool_axis_bounds([(0, df)], input_units="um")
    check("axon excluded by default: x max == 15.0",
          approx(b_default["x"][1], 15.0, rtol=1e-9),
          "got %r" % (b_default["x"],))
    b_incl = tf.pool_axis_bounds([(0, df)], input_units="um",
                                 include_axon=True)
    check("axon included when asked: x max == 500.0",
          approx(b_incl["x"][1], 500.0, rtol=1e-9),
          "got %r" % (b_incl["x"],))


def test_boundary_flag():
    print("\n[7] flag_by_boundary finds a tip near the pooled z bound")
    rows, nid, tip = straight_dendrite(
        n_segments=3, spacing_um=10.0, radius_um=0.3, axis=(0, 0, 1))
    df = make_frame(rows)
    taper = tf.build_taper_table(df, input_units="um")
    bounds = {"x": (-5.0, 5.0), "y": (-5.0, 5.0), "z": (0.0, 32.0)}
    out = tf.flag_by_boundary(taper, bounds, margin_um=5.0)
    row = out[out["tip_node_id"] == tip].iloc[0]
    check("dist_to_z_bound == 2.0 (tip at z=30, bound at 32)",
          approx(row["dist_to_z_bound_um"], 2.0, rtol=1e-9),
          "got %.3f" % row["dist_to_z_bound_um"])
    check("z_boundary_flag_truncated True", bool(row["z_boundary_flag_truncated"]))
    check("x_boundary_near True (tip at x=0, bounds +/-5, margin 5)",
          bool(row["x_boundary_near"]))


def _combo_row(taper_flag, z_flag):
    """Build a one-row frame with the two flag columns pre-set, for exercising
    combine_flags directly without going through the full pipeline."""
    df = pd.DataFrame({
        "tip_node_id": [0],
        "d_from_soma_um": [10.0],
        "taper_flag_truncated": pd.array([taper_flag], dtype="boolean"),
        "z_boundary_flag_truncated": [z_flag],
        "x_boundary_near": [False],
        "y_boundary_near": [False],
    })
    return df


def test_combine_taper_true_z_false():
    print("\n[8] combine: taper=True, z=False -> basis 'taper'")
    out = tf.combine_flags(_combo_row(True, False))
    check("is_truncated True", bool(out["is_truncated"].iloc[0]))
    check("basis == 'taper'", out["truncation_basis"].iloc[0] == "taper",
          "got %r" % out["truncation_basis"].iloc[0])


def test_combine_taper_false_z_true():
    print("\n[9] combine: taper=False (determinate), z=True -> NOT overridden")
    out = tf.combine_flags(_combo_row(False, True))
    check("is_truncated False (determinate taper wins)",
          not bool(out["is_truncated"].iloc[0]))
    check("basis == 'none'", out["truncation_basis"].iloc[0] == "none",
          "got %r" % out["truncation_basis"].iloc[0])


def test_combine_indeterminate_z_true():
    print("\n[10] combine: taper=None, z=True -> basis 'z_boundary'")
    out = tf.combine_flags(_combo_row(pd.NA, True))
    check("is_truncated True", bool(out["is_truncated"].iloc[0]))
    check("basis == 'z_boundary'",
          out["truncation_basis"].iloc[0] == "z_boundary",
          "got %r" % out["truncation_basis"].iloc[0])


def test_combine_indeterminate_z_false():
    print("\n[11] combine: taper=None, z=False -> basis 'unresolved', not flagged")
    out = tf.combine_flags(_combo_row(pd.NA, False))
    check("is_truncated False", not bool(out["is_truncated"].iloc[0]))
    check("basis == 'unresolved'",
          out["truncation_basis"].iloc[0] == "unresolved",
          "got %r" % out["truncation_basis"].iloc[0])


def test_xy_never_drives_alone():
    print("\n[12] x/y proximity never drives is_truncated alone")
    rows, nid, tip = straight_dendrite(
        n_segments=6, spacing_um=2.0, radius_um=0.5, axis=(1, 0, 0))
    rows = apply_taper(rows, tip, taper_r_um=0.05)  # real, tapering ending
    df = make_frame(rows)
    taper = tf.build_taper_table(df, input_units="um")
    taper = tf.flag_by_taper(taper)
    # bounds put the tip RIGHT at the x edge but far from z edge
    bounds = {"x": (0.0, 12.0), "y": (-100.0, 100.0), "z": (-100.0, 100.0)}
    taper = tf.flag_by_boundary(taper, bounds, margin_um=5.0)
    out = tf.combine_flags(taper)
    row = out[out["tip_node_id"] == tip].iloc[0]
    check("x_boundary_near True", bool(row["x_boundary_near"]))
    check("z_boundary_flag_truncated False",
          not bool(row["z_boundary_flag_truncated"]))
    check("is_truncated False despite x proximity",
          not bool(row["is_truncated"]),
          "basis=%r" % row["truncation_basis"])


def test_summary_consistency():
    print("\n[13] cell_truncation_summary is self-consistent")
    rows = []
    nid_ctr = [0]
    all_rows = [(0, -1, 0.0, 0.0, 0.0, 5.0, "soma")]
    nid_ctr[0] = 1
    tips = []
    for k, taper_amt in enumerate([0.5, 0.08, 0.5]):
        # three independent branches off the soma
        prev = 0
        base = nid_ctr[0]
        for j in range(1, 4):
            r = 0.5 if j < 3 else taper_amt
            all_rows.append((nid_ctr[0], prev, k * 20.0 + j * 2.0, 0.0, 0.0,
                            r, "dendrite"))
            prev = nid_ctr[0]
            nid_ctr[0] += 1
        tips.append(prev)
    df = make_frame(all_rows)
    bounds = {"x": (-1000, 1000), "y": (-1000, 1000), "z": (-1000, 1000)}
    out = tf.build_truncation_table(df, bounds, input_units="um")
    summ = tf.cell_truncation_summary(out)
    check("n_tips == 3", summ["n_tips"] == 3, "got %d" % summ["n_tips"])
    check("frac_truncated consistent",
          approx(summ["frac_truncated"],
                 summ["n_truncated"] / float(summ["n_tips"]), rtol=1e-9))
    check("basis_counts sums to n_tips",
          sum(summ["basis_counts"].values()) == summ["n_tips"],
          "got %r" % summ["basis_counts"])


def test_determinism():
    print("\n[14] determinism under input row shuffling")
    rows, nid, tip = straight_dendrite(n_segments=6, spacing_um=2.0,
                                       radius_um=0.5)
    df = make_frame(rows)
    a = tf.build_taper_table(df, input_units="um")
    b = tf.build_taper_table(df.sample(frac=1.0, random_state=3),
                             input_units="um")
    check("same tip set", set(a["tip_node_id"]) == set(b["tip_node_id"]))
    am = a.set_index("tip_node_id")["taper_ratio"]
    bm = b.set_index("tip_node_id")["taper_ratio"]
    check("same taper_ratio per tip",
          np.allclose(am.sort_index().values, bm.sort_index().values))


def test_empty_input():
    print("\n[15] empty input (no dendrite tips)")
    rows = [(0, -1, 0.0, 0.0, 0.0, 5.0, "soma")]
    df = make_frame(rows)
    out = tf.build_taper_table(df, input_units="um")
    check("no tips -> empty frame", len(out) == 0, "got %d" % len(out))
    check("empty frame has expected columns",
          "taper_ratio" in out.columns and "tip_node_id" in out.columns)
    check("cell_truncation_summary handles it",
          tf.cell_truncation_summary(
              tf.combine_flags(tf.flag_by_boundary(
                  tf.flag_by_taper(out),
                  {"x": (0, 1), "y": (0, 1), "z": (0, 1)})))["n_tips"] == 0)


# --------------------------------------------------------------------------- #
# Real-data harness                                                            #
# --------------------------------------------------------------------------- #
def run_on_real(paths):
    print("\n[real] pooling %d frame(s)" % len(paths))
    frames = [(p, pd.read_csv(p)) for p in paths]
    bounds = tf.pool_axis_bounds(frames, input_units="nm")
    print("      pooled n_cells=%d n_nodes=%d" % (bounds["n_cells"],
                                                   bounds["n_nodes"]))
    print("      x %s  y %s  z %s (um)" % (
        tuple(round(v, 1) for v in bounds["x"]),
        tuple(round(v, 1) for v in bounds["y"]),
        tuple(round(v, 1) for v in bounds["z"])))
    print("      z_range_fraction_of_slab = %.3f  (near 1.0 means the "
          "pooled sample spans close to the full 170 um H01 depth)"
          % bounds["z_range_fraction_of_slab"])
    for nid, df in frames:
        out = tf.build_truncation_table(df, bounds, nid=nid,
                                        input_units="nm")
        summ = tf.cell_truncation_summary(out)
        print("      %-40s %s" % (nid, {k: v for k, v in summ.items()
                                        if k != "basis_counts"}))
        print("      %-40s basis_counts=%s" % ("", summ.get("basis_counts")))


def main():
    print("truncation_flag smoke test  (%s, against %s)"
          % (tf.MODULE_VERSION, sd.MODULE_VERSION))
    test_real_ending_not_flagged()
    test_abrupt_cut_flagged()
    test_short_stub_indeterminate()
    test_reference_interpolation()
    test_pool_axis_bounds_recovers_extent()
    test_pool_axis_bounds_axon_exclusion()
    test_boundary_flag()
    test_combine_taper_true_z_false()
    test_combine_taper_false_z_true()
    test_combine_indeterminate_z_true()
    test_combine_indeterminate_z_false()
    test_xy_never_drives_alone()
    test_summary_consistency()
    test_determinism()
    test_empty_input()

    if len(sys.argv) > 1:
        run_on_real(sys.argv[1:])

    print("\n%s" % ("-" * 62))
    if FAILURES:
        print("FAILED %d check(s): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
