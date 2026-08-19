"""test_spine_geometry -- smoke test for spine_geometry.

RUN
---
    python3 test_spine_geometry.py                  # synthetic fixtures only
    python3 test_spine_geometry.py neuron_123.csv   # + a real labelled frame

Exit code 0 means every check passed. Any failure prints the offending
quantities and exits 1.

WHAT IS ACTUALLY BEING TESTED
-----------------------------
The point of a smoke test for this module is that the axial-resistance formula
has a closed form only because of a cancellation (Eq. G1), so a plausible-looking
wrong implementation -- mean radius, mean of 1/r^2, geometric mean -- gives
answers of the right order of magnitude and is easy to miss by inspection. Tests
2 and 3 therefore check against NUMERICAL INTEGRATION of the same integral, not
against a hand-computed number, so they fail if the closed form is wrong rather
than if the arithmetic is mistyped.

Test list:
  1  uniform cylinder neck reproduces Eq. (4), 4 rho L / (pi d^2)
  2  tapered neck matches numerical integration of rho dx / (pi r(x)^2)
  3  multi-frustum neck matches numerical integration end to end
  4  uniform radius scaling: G(s) == G(1) / s^2 exactly
  5  area decomposition is exhaustive and agrees with spine_density
  6  branched spine: n_tips == 2, g_min <= g_primary <= g_max, primary tip is
     the one with the larger head
  7  no-neck spine is flagged, not silently reported as R = 0
  8  Eyal-like reference spine lands in the published 50-80 MOhm band at
     rho_a = 200-300 Ohm cm
  9  determinism: two builds of the same frame are byte-identical
 10  empty input returns an empty frame with the right columns
 11  missing 'r' column: every node falls back and the flag says so
 12  additive radius offsets move G in the right direction by the right amount

CONVENTION IN THE FIXTURES
--------------------------
Coordinates are built in um and passed with input_units='um' so the fixture
geometry is readable. Real H01 frames are nm; test 13 checks the two agree.
"""

import math
import sys

import numpy as np
import pandas as pd

import spine_density as sd
import spine_geometry as sg


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
# Fixture builders                                                             #
# --------------------------------------------------------------------------- #
def make_frame(rows):
    """rows: list of (id, p, x, y, z, r, annotated_type) in um."""
    return pd.DataFrame(rows, columns=["id", "p", "x", "y", "z", "r",
                                       "annotated_type"])


def straight_shaft(n=4, spacing=10.0, radius=0.5):
    """A soma at the origin plus n dendrite nodes along +x."""
    rows = [(0, -1, 0.0, 0.0, 0.0, 5.0, "soma")]
    for i in range(1, n + 1):
        rows.append((i, i - 1, i * spacing, 0.0, 0.0, radius, "dendrite"))
    return rows


def add_spine(rows, base_id, start_id, neck_segments, head_segments,
              offset=(0.0, 1.0, 0.0)):
    """Append a spine hanging off base_id along +y.

    neck_segments : list of (length_um, radius_at_far_end_um)
    head_segments : same, appended after the neck
    The radius at the near end of the first neck segment is the radius of the
    first neck node itself (spine_density's frustum convention uses the two node
    radii, and the base node is a shaft node), so the first segment runs from
    the SHAFT radius to the first neck radius. Tests that need an exactly
    uniform neck therefore start the neck with a zero-length node at the shaft.
    """
    base = [r for r in rows if r[0] == base_id][0]
    x0, y0, z0 = base[2], base[3], base[4]
    nid = start_id
    dist = 0.0
    prev = base_id
    ux, uy, uz = offset
    norm = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / norm, uy / norm, uz / norm
    for (ln, rad) in neck_segments:
        dist += ln
        rows.append((nid, prev, x0 + ux * dist, y0 + uy * dist, z0 + uz * dist,
                     rad, "neck"))
        prev = nid
        nid += 1
    for (ln, rad) in head_segments:
        dist += ln
        rows.append((nid, prev, x0 + ux * dist, y0 + uy * dist, z0 + uz * dist,
                     rad, "head"))
        prev = nid
        nid += 1
    return rows, nid


# numpy renamed trapz -> trapezoid in 2.0; Colab may still be on 1.x
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def numeric_axial_factor_um(r1, r2, length, n=200001):
    """Numerically integrate L/(pi r(x)^2) with a linear radius taper."""
    x = np.linspace(0.0, length, n)
    r = r1 + (r2 - r1) * x / length
    return float(_TRAPZ(1.0 / (math.pi * r ** 2), x))


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
def test_uniform_cylinder():
    print("\n[1] uniform cylinder neck reproduces Eq. (4)")
    L, d = 1.35, 0.25
    r = d / 2.0
    rows = straight_shaft()
    # zero-length first node at the shaft, at neck radius, so the neck is
    # genuinely uniform from the base outward
    rows.append((100, 2, 20.0, 0.0, 0.0, r, "neck"))
    rows.append((101, 100, 20.0, L, 0.0, r, "neck"))
    rows.append((102, 101, 20.0, L + 0.5, 0.0, 0.4, "head"))
    df = make_frame(rows)
    out = sg.build_spine_geometry(df, nid=1, input_units="um",
                                  radius_offsets_um=())
    check("one spine found", len(out) == 1, "got %d" % len(out))
    row = out.iloc[0]
    check("L_neck == 1.35", approx(row["L_neck_um"], L, rtol=1e-12),
          "got %.12g" % row["L_neck_um"])
    for rho in (100.0, 200.0, 300.0):
        expect = 4.0 * rho * (L * 1e-4) / (math.pi * (d * 1e-4) ** 2)
        got = sg.neck_resistance_ohm(row["g_per_cm"], rho)
        check("Eq.(4) at rho=%g  (%.4g Ohm)" % (rho, expect),
              approx(float(got), expect, rtol=1e-10),
              "got %.10g expected %.10g" % (got, expect))
    check("d_neck_equiv == 0.25", approx(row["d_neck_equiv_um"], d, rtol=1e-10),
          "got %.10g" % row["d_neck_equiv_um"])


def test_tapered_single():
    print("\n[2] tapered neck matches numerical integration")
    L, r1, r2 = 1.2, 0.15, 0.06
    rows = straight_shaft()
    rows.append((100, 2, 20.0, 0.0, 0.0, r1, "neck"))
    rows.append((101, 100, 20.0, L, 0.0, r2, "neck"))
    rows.append((102, 101, 20.0, L + 0.4, 0.0, 0.35, "head"))
    df = make_frame(rows)
    out = sg.build_spine_geometry(df, nid=1, input_units="um",
                                  radius_offsets_um=())
    g_um = out.iloc[0]["g_per_cm"] / sg.UM_PER_CM
    ref = numeric_axial_factor_um(r1, r2, L)
    check("closed form == numerical integral", approx(g_um, ref, rtol=1e-6),
          "closed %.10g numeric %.10g" % (g_um, ref))
    # a mean-radius implementation would give this instead -- must NOT match
    rbar = 0.5 * (r1 + r2)
    wrong = L / (math.pi * rbar ** 2)
    check("mean-radius form is measurably different",
          abs(g_um - wrong) / ref > 0.05,
          "closed %.6g mean-radius %.6g (too close to distinguish)"
          % (g_um, wrong))


def test_tapered_multi():
    print("\n[3] multi-frustum neck matches numerical integration")
    segs = [(0.4, 0.14), (0.3, 0.09), (0.5, 0.05), (0.2, 0.07)]
    r0 = 0.18
    rows = straight_shaft()
    rows.append((100, 2, 20.0, 0.0, 0.0, r0, "neck"))
    nid = 101
    y = 0.0
    prev_r = r0
    ref = 0.0
    prev = 100
    for (ln, rad) in segs:
        y += ln
        rows.append((nid, prev, 20.0, y, 0.0, rad, "neck"))
        ref += numeric_axial_factor_um(prev_r, rad, ln)
        prev, prev_r = nid, rad
        nid += 1
    rows.append((nid, prev, 20.0, y + 0.4, 0.0, 0.32, "head"))
    df = make_frame(rows)
    out = sg.build_spine_geometry(df, nid=1, input_units="um",
                                  radius_offsets_um=())
    g_um = out.iloc[0]["g_per_cm"] / sg.UM_PER_CM
    check("series sum == numerical integral", approx(g_um, ref, rtol=1e-6),
          "closed %.10g numeric %.10g" % (g_um, ref))
    check("L_neck == sum of segment lengths",
          approx(out.iloc[0]["L_neck_um"], sum(s[0] for s in segs), rtol=1e-12),
          "got %.10g" % out.iloc[0]["L_neck_um"])


def test_radius_scaling():
    print("\n[4] uniform radius scaling: G(s) == G(1)/s^2")
    rows = straight_shaft()
    rows.append((100, 2, 20.0, 0.0, 0.0, 0.12, "neck"))
    rows.append((101, 100, 20.0, 0.8, 0.0, 0.07, "neck"))
    rows.append((102, 101, 20.0, 1.2, 0.0, 0.3, "head"))
    df = make_frame(rows)
    g1 = sg.build_spine_geometry(df, input_units="um",
                                 radius_offsets_um=()).iloc[0]["g_per_cm"]
    for s in (0.5, 2.0, 3.0):
        df_s = df.copy()
        df_s["r"] = df_s["r"] * s
        gs = sg.build_spine_geometry(df_s, input_units="um",
                                     radius_offsets_um=()).iloc[0]["g_per_cm"]
        check("s=%.1f" % s, approx(gs, g1 / (s * s), rtol=1e-10),
              "got %.10g expected %.10g" % (gs, g1 / (s * s)))


def test_area_agreement():
    print("\n[5] area decomposition and agreement with spine_density")
    rows = straight_shaft()
    rows, nid = add_spine(rows, 2, 100, [(0.6, 0.10), (0.6, 0.08)],
                          [(0.5, 0.30)])
    rows, nid = add_spine(rows, 3, nid, [(0.9, 0.09)], [(0.4, 0.28)],
                          offset=(0.0, 0.0, 1.0))
    df = make_frame(rows)
    out = sg.build_spine_geometry(df, input_units="um", radius_offsets_um=())
    check("two spines", len(out) == 2, "got %d" % len(out))
    parts = out["A_neck_um2"] + out["A_head_um2"] + out["A_other_um2"]
    check("A parts sum to A_spine",
          np.allclose(parts.values, out["A_spine_um2"].values, atol=1e-12))
    node, children, root = sd._prepare_nodes(df, sd.SHAFT_REGEX,
                                             sd.SPINE_LABELS,
                                             sd.DEFAULT_RADIUS_NM, "um")
    seg_area, dropped = sd._attribute_spine_area(node, children)
    total_sd = sum(seg_area.values()) + dropped
    check("total area == spine_density attribution",
          approx(float(out["A_spine_um2"].sum()), float(total_sd), rtol=1e-9),
          "here %.10g there %.10g" % (out["A_spine_um2"].sum(), total_sd))
    phi = sd.build_phi(df, input_units="um")
    check("total area == phi spine_area column",
          approx(float(out["A_spine_um2"].sum()),
                 float(phi["spine_area_um2"].sum()), rtol=1e-9),
          "here %.10g phi %.10g" % (out["A_spine_um2"].sum(),
                                    phi["spine_area_um2"].sum()))
    merged, rep = sg.attach_to_phi(out, phi)
    check("every spine joins to a phi segment", rep["n_unmatched"] == 0,
          "unmatched %d" % rep["n_unmatched"])


def test_branched_spine():
    print("\n[6] branched spine: two tips, primary is the larger head")
    rows = straight_shaft()
    # shared neck 100 -> 101, then two heads
    rows.append((100, 2, 20.0, 0.0, 0.0, 0.10, "neck"))
    rows.append((101, 100, 20.0, 0.8, 0.0, 0.08, "neck"))
    rows.append((102, 101, 20.0, 1.2, 0.3, 0.20, "head"))   # smaller head
    rows.append((103, 101, 20.0, 1.2, -0.3, 0.45, "head"))  # larger head
    df = make_frame(rows)
    out = sg.build_spine_geometry(df, input_units="um", radius_offsets_um=())
    check("one spine (shared root)", len(out) == 1, "got %d" % len(out))
    row = out.iloc[0]
    check("n_tips == 2", row["n_tips"] == 2, "got %s" % row["n_tips"])
    check("primary tip is node 103 (larger head)", row["primary_tip_id"] == 103,
          "got %s" % row["primary_tip_id"])
    check("g_min <= g <= g_max",
          row["g_per_cm_min"] <= row["g_per_cm"] <= row["g_per_cm_max"])


def test_no_neck_flagged():
    print("\n[7] no-neck spine is flagged, not reported as R = 0")
    rows = straight_shaft()
    rows.append((100, 2, 20.0, 0.0, 0.0, 0.30, "head"))
    rows.append((101, 100, 20.0, 0.5, 0.0, 0.30, "head"))
    df = make_frame(rows)
    out = sg.build_spine_geometry(df, input_units="um", radius_offsets_um=())
    row = out.iloc[0]
    check("has_neck is False", not bool(row["has_neck"]))
    check("g_per_cm == 0", row["g_per_cm"] == 0.0, "got %g" % row["g_per_cm"])
    check("d_neck_equiv is nan", math.isnan(row["d_neck_equiv_um"]))
    summ = sg.cell_spine_summary(out)
    check("summary excludes it from R stats", summ["n_with_neck"] == 0,
          "n_with_neck=%d" % summ["n_with_neck"])
    check("frac_with_neck == 0", summ["frac_with_neck"] == 0.0)


def test_eyal_band():
    print("\n[8] Eyal-like spine lands in the published 50-80 MOhm band")
    L, d = 1.35, 0.25
    r = d / 2.0
    rows = straight_shaft()
    rows.append((100, 2, 20.0, 0.0, 0.0, r, "neck"))
    rows.append((101, 100, 20.0, L, 0.0, r, "neck"))
    # head area 2.8 um^2 -> sphere-equivalent diameter sqrt(2.8/pi) = 0.944 um
    rows.append((102, 101, 20.0, L + 0.5, 0.0, 0.55, "head"))
    df = make_frame(rows)
    out = sg.build_spine_geometry(df, input_units="um", radius_offsets_um=())
    g = out.iloc[0]["g_per_cm"]
    r200 = float(sg.neck_resistance_mohm(g, 200.0))
    r300 = float(sg.neck_resistance_mohm(g, 300.0))
    print("      R(200 Ohm cm) = %.1f MOhm ; R(300 Ohm cm) = %.1f MOhm"
          % (r200, r300))
    check("R(200) in [45, 65] MOhm", 45.0 <= r200 <= 65.0, "got %.2f" % r200)
    check("R(300) in [70, 95] MOhm", 70.0 <= r300 <= 95.0, "got %.2f" % r300)
    # the envelope endpoints quoted in the literature
    r_lo = float(sg.neck_resistance_mohm(
        sg.axial_factor_to_per_cm(
            sg.frustum_axial_factor_um(0.15, 0.15, 1.35)), 100.0))
    r_hi = float(sg.neck_resistance_mohm(
        sg.axial_factor_to_per_cm(
            sg.frustum_axial_factor_um(0.10, 0.10, 1.344)), 300.0))
    print("      envelope: %.1f MOhm (rho=100, d=0.30) to %.1f MOhm "
          "(rho=300, d=0.20)" % (r_lo, r_hi))
    check("low endpoint ~ 19 MOhm", 18.0 <= r_lo <= 20.0, "got %.2f" % r_lo)
    check("high endpoint ~ 128 MOhm", 125.0 <= r_hi <= 131.0,
          "got %.2f" % r_hi)


def test_determinism():
    print("\n[9] determinism")
    rows = straight_shaft()
    rows, nid = add_spine(rows, 2, 100, [(0.6, 0.10)], [(0.5, 0.30)])
    rows, nid = add_spine(rows, 3, nid, [(0.9, 0.09)], [(0.4, 0.28)],
                          offset=(0.0, 0.0, 1.0))
    rows, nid = add_spine(rows, 1, nid, [(0.7, 0.11)], [(0.45, 0.26)],
                          offset=(0.0, -1.0, 0.0))
    df = make_frame(rows)
    a = sg.build_spine_geometry(df, input_units="um")
    b = sg.build_spine_geometry(df.sample(frac=1.0, random_state=0),
                                input_units="um")
    check("row order stable under input shuffle",
          list(a["spine_root_id"]) == list(b["spine_root_id"]),
          "%s vs %s" % (list(a["spine_root_id"]), list(b["spine_root_id"])))
    check("values identical under input shuffle",
          np.allclose(a["g_per_cm"].values, b["g_per_cm"].values, atol=0,
                      rtol=0))


def test_empty_and_missing_radius():
    print("\n[10,11] empty input and missing radius column")
    rows = straight_shaft()
    df = make_frame(rows)
    out = sg.build_spine_geometry(df, input_units="um")
    check("no spines -> empty frame", len(out) == 0, "got %d" % len(out))
    check("empty frame still has spine_root_id column",
          "spine_root_id" in out.columns)
    check("cell_spine_summary handles empty",
          sg.cell_spine_summary(out)["n_spines"] == 0)

    rows2 = straight_shaft()
    rows2.append((100, 2, 20.0, 0.0, 0.0, 0.1, "neck"))
    rows2.append((101, 100, 20.0, 0.8, 0.0, 0.3, "head"))
    df2 = make_frame(rows2).drop(columns=["r"])
    out2 = sg.build_spine_geometry(df2, input_units="um",
                                   default_radius_nm=50.0)
    check("missing r -> all nodes at fallback",
          out2.iloc[0]["radius_default_frac"] == 1.0,
          "got %g" % out2.iloc[0]["radius_default_frac"])
    summ = sg.cell_spine_summary(out2)
    check("summary reports the fallback fraction",
          summ["frac_nodes_at_default_radius"] == 1.0,
          "got %g" % summ["frac_nodes_at_default_radius"])


def test_radius_offsets():
    print("\n[12] additive radius offsets")
    rows = straight_shaft()
    rows.append((100, 2, 20.0, 0.0, 0.0, 0.125, "neck"))
    rows.append((101, 100, 20.0, 1.0, 0.0, 0.125, "neck"))
    rows.append((102, 101, 20.0, 1.5, 0.0, 0.4, "head"))
    df = make_frame(rows)
    out = sg.build_spine_geometry(df, input_units="um",
                                  radius_offsets_um=(-0.025, 0.025))
    row = out.iloc[0]
    base = row["g_per_cm"]
    minus = row["g_per_cm_off0"]
    plus = row["g_per_cm_off1"]
    check("negative offset raises G", minus > base,
          "%.6g vs %.6g" % (minus, base))
    check("positive offset lowers G", plus < base,
          "%.6g vs %.6g" % (plus, base))
    # uniform cylinder: G scales as 1/r^2, so the ratios are exact
    check("minus offset ratio == (0.125/0.100)^2",
          approx(minus / base, (0.125 / 0.100) ** 2, rtol=1e-10),
          "got %.10g" % (minus / base))
    check("plus offset ratio == (0.125/0.150)^2",
          approx(plus / base, (0.125 / 0.150) ** 2, rtol=1e-10),
          "got %.10g" % (plus / base))


def test_radius_report_integration():
    print("\n[14] radius_report integration into cell_spine_summary")
    rows = straight_shaft()
    rows.append((100, 2, 20.0, 0.0, 0.0, 0.12, "neck"))
    rows.append((101, 100, 20.0, 0.9, 0.0, 0.09, "neck"))
    rows.append((102, 101, 20.0, 1.4, 0.0, 0.33, "head"))
    df = make_frame(rows)
    out = sg.build_spine_geometry(df, input_units="um", radius_offsets_um=())

    # without a report, the summary must not claim trustworthiness either way
    plain = sg.cell_spine_summary(out)
    check("no radius_report -> no resistance_trustworthy key",
          "resistance_trustworthy" not in plain)

    # a clean report
    good = {"radius_suspect": False, "flat_radius": False,
            "default_dominated": False, "frac_at_default_r": 0.0}
    s_good = sg.cell_spine_summary(out, radius_report=good)
    check("clean report -> trustworthy True",
          s_good["resistance_trustworthy"] is True)
    check("report fields copied under radius_ prefix",
          s_good["radius_flat_radius"] is False)

    # a suspect report must flip it, regardless of the spine geometry
    bad = {"radius_suspect": True, "flat_radius": True,
           "default_dominated": True, "frac_at_default_r": 1.0}
    s_bad = sg.cell_spine_summary(out, radius_report=bad)
    check("suspect report -> trustworthy False",
          s_bad["resistance_trustworthy"] is False)

    # and it must survive the empty-frame path too
    empty = sg.build_spine_geometry(make_frame(straight_shaft()),
                                    input_units="um")
    s_empty = sg.cell_spine_summary(empty, radius_report=bad)
    check("empty frame still carries the radius verdict",
          s_empty["resistance_trustworthy"] is False)


def test_head_shape_bracket():
    """The lateral-vs-sphere bracket must expose a blob-like head.

    This is the diagnostic that decides whether A_head_um2 is a measurement
    or a lower bound. A head the skeleton traverses fully (path = 2 x radius)
    must give lateral == sphere; a head collapsed toward a point must give a
    ratio that falls in proportion to the path length.
    """
    print("\n[15] head shape bracket: lateral vs sphere")
    r_head = 0.479                      # Eyal-sized head: 4 pi r^2 = 2.88 um2

    def build(path_um):
        rows = straight_shaft()
        rows.append((100, 2, 20.0, 0.0, 0.0, 0.125, "neck"))
        rows.append((101, 100, 20.0, 1.0, 0.0, 0.125, "neck"))
        # head entered at node 102 and, if path_um > 0, exited at 103
        rows.append((102, 101, 20.0, 1.0 + 1e-6, 0.0, r_head, "head"))
        if path_um > 0:
            rows.append((103, 102, 20.0, 1.0 + path_um, 0.0, r_head, "head"))
        return sg.build_spine_geometry(make_frame(rows), input_units="um",
                                       radius_offsets_um=()).iloc[0]

    full = build(2.0 * r_head)          # traversed fully
    check("sphere area is 4 pi r^2",
          approx(full["A_head_sphere_um2"], 4 * math.pi * r_head ** 2,
                 rtol=1e-9),
          "got %.6g" % full["A_head_sphere_um2"])
    # NOTE the ratio carries a constant offset from the neck->head transition
    # segment: radius jumps from neck to head over ~zero length, so its slant
    # is large and it contributes a real "shoulder" annulus of membrane. That
    # is genuine area, not an artefact, so the right test is on the INCREMENT
    # with path length, which isolates the within-head lateral contribution.
    shoulder = full["head_lateral_over_sphere"] - 1.0
    check("full traverse: lateral/sphere = 1 + shoulder, shoulder in (0, 0.4)",
          0.0 < shoulder < 0.4,
          "got ratio %.4f -> shoulder %.4f"
          % (full["head_lateral_over_sphere"], shoulder))
    check("full traverse: path/radius ~ 2",
          abs(full["head_path_over_radius"] - 2.0) < 0.02,
          "got %.4f" % full["head_path_over_radius"])

    half = build(1.0 * r_head)
    check("halving the path halves the within-head lateral area",
          abs((full["head_lateral_over_sphere"] - shoulder)
              - 2.0 * (half["head_lateral_over_sphere"] - shoulder)) < 0.03,
          "full %.4f half %.4f shoulder %.4f"
          % (full["head_lateral_over_sphere"],
             half["head_lateral_over_sphere"], shoulder))

    # the ratio must FALL with path length -- monotone, not incidental
    ratios = [build(f * r_head)["head_lateral_over_sphere"]
              for f in (2.0, 1.0, 0.5, 0.25)]
    check("ratio falls monotonically with path length",
          all(ratios[i] > ratios[i + 1] for i in range(len(ratios) - 1)),
          "got %s" % np.round(ratios, 4))

    # a spine with no head at all must not fabricate a bracket
    rows = straight_shaft()
    rows.append((100, 2, 20.0, 0.0, 0.0, 0.12, "neck"))
    rows.append((101, 100, 20.0, 0.9, 0.0, 0.09, "neck"))
    noh = sg.build_spine_geometry(make_frame(rows), input_units="um",
                                  radius_offsets_um=()).iloc[0]
    check("no head -> sphere area is nan",
          math.isnan(noh["A_head_sphere_um2"]))
    check("no head -> ratio is nan",
          math.isnan(noh["head_lateral_over_sphere"]))

    summ = sg.cell_spine_summary(
        sg.build_spine_geometry(make_frame(rows), input_units="um",
                                radius_offsets_um=()))
    check("summary carries the bracket keys",
          "head_lateral_over_sphere_median" in summ
          and "A_head_sphere_median_um2" in summ)


def test_units_equivalence():
    print("\n[13] nm and um input give the same geometry")
    rows = straight_shaft()
    rows.append((100, 2, 20.0, 0.0, 0.0, 0.12, "neck"))
    rows.append((101, 100, 20.0, 0.9, 0.0, 0.09, "neck"))
    rows.append((102, 101, 20.0, 1.4, 0.0, 0.33, "head"))
    df_um = make_frame(rows)
    df_nm = df_um.copy()
    for c in ("x", "y", "z", "r"):
        df_nm[c] = df_nm[c] * 1000.0
    a = sg.build_spine_geometry(df_um, input_units="um", radius_offsets_um=())
    b = sg.build_spine_geometry(df_nm, input_units="nm", radius_offsets_um=())
    for col in ("L_neck_um", "A_head_um2", "A_neck_um2", "g_per_cm",
                "d_neck_equiv_um", "d_base_um"):
        check("%s agrees across units" % col,
              approx(float(a.iloc[0][col]), float(b.iloc[0][col]), rtol=1e-9),
              "um %.10g nm %.10g" % (a.iloc[0][col], b.iloc[0][col]))


# --------------------------------------------------------------------------- #
# Real-data harness                                                            #
# --------------------------------------------------------------------------- #
def run_on_real(path):
    """Run on a real labelled frame and print the diagnostics that matter.

    Nothing here asserts: on real data the interesting output is the shape of
    the distributions and the data-quality flags, not a pass/fail.
    """
    print("\n[real] %s" % path)
    df = pd.read_csv(path)
    print("      %d nodes, labels: %s" % (
        len(df), sorted(set(str(t).lower()
                            for t in df["annotated_type"].unique()))[:12]))
    if "r" not in df.columns:
        print("      WARNING: no 'r' column -- every radius will be the "
              "fallback and no resistance below is meaningful")
    out = sg.build_spine_geometry(df, nid=None, input_units="nm")
    summ = sg.cell_spine_summary(out)
    for k in sorted(summ):
        v = summ[k]
        print("      %-38s %s" % (k, ("%.6g" % v) if isinstance(v, float)
                                  else v))
    if len(out):
        wn = out[out["has_neck"]]
        if len(wn):
            print("      L_neck_um    quantiles 5/25/50/75/95: %s"
                  % np.round(np.percentile(wn["L_neck_um"],
                                           [5, 25, 50, 75, 95]), 4))
            print("      d_neck_equiv quantiles 5/25/50/75/95: %s"
                  % np.round(np.percentile(wn["d_neck_equiv_um"],
                                           [5, 25, 50, 75, 95]), 4))
            for rho in (100.0, 200.0, 300.0, 400.0):
                r = sg.neck_resistance_mohm(wn["g_per_cm"].values, rho)
                print("      R_neck MOhm at rho=%3.0f  5/50/95: "
                      "%8.2f %8.2f %8.2f"
                      % (rho, np.percentile(r, 5), np.percentile(r, 50),
                         np.percentile(r, 95)))
    return out


# --------------------------------------------------------------------------- #
def main():
    print("spine_geometry smoke test  (%s, against %s)"
          % (sg.MODULE_VERSION, sd.MODULE_VERSION))
    test_uniform_cylinder()
    test_tapered_single()
    test_tapered_multi()
    test_radius_scaling()
    test_area_agreement()
    test_branched_spine()
    test_no_neck_flagged()
    test_eyal_band()
    test_determinism()
    test_empty_and_missing_radius()
    test_radius_offsets()
    test_radius_report_integration()
    test_head_shape_bracket()
    test_units_equivalence()

    for path in sys.argv[1:]:
        run_on_real(path)

    print("\n%s" % ("-" * 62))
    if FAILURES:
        print("FAILED %d check(s): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
