"""test_s1_plots -- smoke test for s1_plots.

RUN
---
    python3 test_s1_plots.py
    python3 test_s1_plots.py --save /tmp/figs      # also write PNGs

A plotting smoke test cannot check that a figure is CORRECT -- only a human
looking at it can. What it can check is that every function returns a real
Figure on realistic input, survives the degenerate inputs that occur in
practice (no spines, no necks, all-NaN columns, a single cell), draws the
number of axes it claims to, and does not mutate its input. Those are the
failures that would otherwise surface as a broken notebook cell at the end of
a long bank run.

Test list:
  1  every function returns a matplotlib Figure on realistic input
  2  each returns the documented number of axes
  3  empty input is handled without raising, for all six
  4  spines with no labelled neck: resistance figures degrade gracefully
  5  all-NaN d_neck_equiv (every spine neckless) does not raise
  6  a single cell works as well as a dict of cells
  7  a bare DataFrame is accepted as well as a {nid: frame} mapping
  8  input frames are not mutated
  9  kappa is computed correctly: hand-check one grid cell against
     1 / (1 + g R), which is the only real arithmetic in the module
 10  the suspect-radii path colours and titles differently
 11  missing optional columns do not raise
"""

import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

import s1_plots as s0p                 # noqa: E402


FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s   %s" % (name, detail))
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
def make_spine_frame(n=400, seed=0, with_neck_frac=0.9, g_scale=1.0):
    """A spine geometry frame shaped like spine_geometry.build_spine_geometry."""
    rng = np.random.RandomState(seed)
    has_neck = rng.rand(n) < with_neck_frac
    l_neck = np.where(has_neck, np.abs(rng.normal(1.34, 0.5, n)), 0.0)
    # G [1/cm] for a 0.25 um neck of that length: 1e4 * L / (pi r1 r2)
    r = np.abs(rng.normal(0.125, 0.03, n)) + 0.02
    g = np.where(has_neck, 1e4 * l_neck / (np.pi * r * r), 0.0) * g_scale
    d_eq = np.where(has_neck & (g > 0),
                    np.sqrt(4.0 * l_neck / (np.pi * np.maximum(g, 1e-12) / 1e4)),
                    np.nan)
    return pd.DataFrame({
        "nid": ["c0"] * n,
        "spine_root_id": np.arange(n),
        "L_neck_um": l_neck,
        "A_head_um2": np.abs(rng.normal(2.88, 1.37, n)),
        "A_neck_um2": np.abs(rng.normal(0.6, 0.2, n)),
        "A_spine_um2": np.abs(rng.normal(3.5, 1.4, n)),
        "g_per_cm": g,
        "has_neck": has_neck,
        "neck_r_mean_um": np.where(has_neck, r, np.nan),
        "d_neck_equiv_um": d_eq,
        "d_base_um": np.abs(rng.normal(95.0, 45.0, n)),
    })


def make_trunc_frame(n=200, seed=1, bimodal=True):
    rng = np.random.RandomState(seed)
    if bimodal:
        ratio = np.concatenate([rng.normal(0.25, 0.08, n // 2),
                                rng.normal(0.95, 0.05, n - n // 2)])
    else:
        ratio = rng.normal(0.6, 0.2, n)
    ratio = np.clip(ratio, 0.01, 1.5)
    det = rng.rand(n) < 0.85
    zb = np.abs(rng.normal(40.0, 30.0, n))
    taper_flag = pd.array(np.where(det, ratio >= 0.7, None), dtype="boolean")
    zflag = zb <= 10.0
    # pandas BooleanArray: .isna() returns a plain ndarray, .fillna() a
    # BooleanArray. Normalise both to ndarray[bool] before combining.
    tf_true = np.asarray(taper_flag.fillna(False), dtype=bool)
    tf_na = np.asarray(taper_flag.isna(), dtype=bool)
    is_trunc = tf_true | (tf_na & zflag)
    basis = np.where(tf_true & zflag, "both",
             np.where(tf_true, "taper",
              np.where(tf_na & zflag, "z_boundary",
               np.where(tf_na, "unresolved", "none"))))
    return pd.DataFrame({
        "nid": ["c0"] * n,
        "tip_node_id": np.arange(n),
        "taper_ratio": ratio,
        "taper_determinate": det,
        "taper_flag_truncated": taper_flag,
        "dist_to_z_bound_um": zb,
        "z_boundary_flag_truncated": zflag,
        "d_from_soma_um": np.abs(rng.normal(180.0, 70.0, n)),
        "is_truncated": is_trunc,
        "truncation_basis": basis,
    })


def make_s1_df(n=8, seed=2):
    rng = np.random.RandomState(seed)
    d = {
        "nid": ["c%d" % i for i in range(n)],
        "n_spines": rng.randint(200, 1600, n),
        "frac_with_neck": rng.uniform(0.4, 1.0, n),
        "frac_truncated": rng.uniform(0.0, 0.4, n),
        "resistance_trustworthy": [True] * (n - 2) + [False, False],
    }
    for rho in (100.0, 200.0, 300.0, 400.0):
        d["R_neck_MOhm_rho%d_median" % int(rho)] = rng.uniform(
            20, 90, n) * rho / 200.0
    return pd.DataFrame(d)


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
ALL_FIGS = {}


def test_returns_figures():
    print("\n[1,2] every function returns a Figure with the documented axes")
    sf = {"c0": make_spine_frame(), "c1": make_spine_frame(seed=5)}
    tf = {"c0": make_trunc_frame(), "c1": make_trunc_frame(seed=6)}
    s0 = make_s1_df()

    cases = [
        ("neck_geometry", s0p.neck_geometry(sf), 4),
        ("neck_resistance_sweep", s0p.neck_resistance_sweep(sf), 2),
        ("attenuation_factor", s0p.attenuation_factor(sf), 2),
        ("spine_distance_profile", s0p.spine_distance_profile(sf), 2),
        ("truncation_diagnostics", s0p.truncation_diagnostics(tf), 4),
        ("s1_batch", s0p.s1_batch(s0)), 
    ]
    # last one has no axis count declared; handle uniformly
    for case in cases:
        name, fig = case[0], case[1]
        n_expect = case[2] if len(case) > 2 else 4
        ALL_FIGS[name] = fig
        check("%s returns a Figure" % name,
              isinstance(fig, matplotlib.figure.Figure),
              "got %r" % type(fig))
        n_axes = len([a for a in fig.axes
                      if not getattr(a, "_colorbar", None)])
        check("%s has >= %d axes" % (name, n_expect), n_axes >= n_expect,
              "got %d" % n_axes)


def test_empty_inputs():
    print("\n[3] empty input is handled without raising")
    empty_spine = pd.DataFrame(columns=[
        "nid", "L_neck_um", "d_neck_equiv_um", "A_head_um2", "has_neck",
        "g_per_cm", "A_spine_um2", "d_base_um", "neck_r_mean_um"])
    empty_trunc = pd.DataFrame(columns=[
        "nid", "taper_ratio", "taper_determinate", "truncation_basis",
        "dist_to_z_bound_um", "d_from_soma_um", "is_truncated"])
    for name, fn, arg in (
            ("neck_geometry", s0p.neck_geometry, {"e": empty_spine}),
            ("neck_resistance_sweep", s0p.neck_resistance_sweep,
             {"e": empty_spine}),
            ("attenuation_factor", s0p.attenuation_factor,
             {"e": empty_spine}),
            ("spine_distance_profile", s0p.spine_distance_profile,
             {"e": empty_spine}),
            ("truncation_diagnostics", s0p.truncation_diagnostics,
             {"e": empty_trunc}),
            ("s1_batch", s0p.s1_batch, pd.DataFrame())):
        try:
            fig = fn(arg)
            check("%s on empty input" % name,
                  isinstance(fig, matplotlib.figure.Figure))
            plt.close(fig)
        except Exception as exc:                        # noqa: BLE001
            check("%s on empty input" % name, False,
                  "%s: %s" % (type(exc).__name__, exc))


def test_no_necks():
    print("\n[4,5] spines with no labelled neck at all")
    sf = make_spine_frame(n=120, with_neck_frac=0.0)
    check("all has_neck False", not sf["has_neck"].any())
    check("all d_neck_equiv NaN", sf["d_neck_equiv_um"].isna().all())
    for name, fn in (("neck_geometry", s0p.neck_geometry),
                     ("neck_resistance_sweep", s0p.neck_resistance_sweep),
                     ("attenuation_factor", s0p.attenuation_factor)):
        try:
            fig = fn({"c0": sf})
            check("%s with zero necks" % name,
                  isinstance(fig, matplotlib.figure.Figure))
            plt.close(fig)
        except Exception as exc:                        # noqa: BLE001
            check("%s with zero necks" % name, False,
                  "%s: %s" % (type(exc).__name__, exc))


def test_single_cell_and_bare_frame():
    print("\n[6,7] single cell, and a bare DataFrame instead of a mapping")
    sf = make_spine_frame(n=200)
    f1 = s0p.neck_resistance_sweep({"only": sf})
    f2 = s0p.neck_resistance_sweep(sf)          # bare frame
    check("mapping form works", isinstance(f1, matplotlib.figure.Figure))
    check("bare DataFrame works", isinstance(f2, matplotlib.figure.Figure))
    plt.close(f1); plt.close(f2)


def test_no_mutation():
    print("\n[8] input frames are not mutated")
    sf = make_spine_frame(n=150)
    before = sf.copy(deep=True)
    for fn in (s0p.neck_geometry, s0p.neck_resistance_sweep,
               s0p.attenuation_factor, s0p.spine_distance_profile):
        plt.close(fn({"c0": sf}))
    check("spine frame unchanged", sf.equals(before))

    tf = make_trunc_frame(n=120)
    before_t = tf.copy(deep=True)
    plt.close(s0p.truncation_diagnostics({"c0": tf}))
    check("truncation frame unchanged", tf.equals(before_t))


def test_kappa_arithmetic():
    """The only real arithmetic in the module: verify one grid cell by hand."""
    print("\n[9] kappa grid matches 1 / (1 + g R) by hand")
    # one spine, known G, so the median IS the value
    g_per_cm = 275033.0                    # Eyal-like: 1.35 um x 0.25 um neck
    sf = pd.DataFrame({
        "nid": ["c0"], "g_per_cm": [g_per_cm], "has_neck": [True],
        "L_neck_um": [1.35], "A_head_um2": [2.88], "A_spine_um2": [3.5],
        "d_base_um": [100.0], "neck_r_mean_um": [0.125],
        "d_neck_equiv_um": [0.25]})
    rho, g_nS = 200.0, 1.0
    fig = s0p.attenuation_factor(sf, rho_a_ohm_cm=(rho,), g_syn_nS=(g_nS,))
    # recover the annotated value from the heatmap text
    texts = [t.get_text() for t in fig.axes[0].texts]
    r_ohm = g_per_cm * rho
    expect = 1.0 / (1.0 + g_nS * 1e-9 * r_ohm)
    check("R_neck is ~55 MOhm as expected", abs(r_ohm / 1e6 - 55.0) < 1.0,
          "got %.2f MOhm" % (r_ohm / 1e6))
    check("heatmap annotates kappa = %.3f" % expect,
          any(abs(float(t) - expect) < 5e-4 for t in texts if t),
          "texts %r expected %.4f" % (texts, expect))
    plt.close(fig)


def test_suspect_radii_path():
    print("\n[10] suspect-radii cells are drawn differently")
    s0 = make_s1_df(n=6)
    fig = s0p.s1_batch(s0)
    title = fig._suptitle.get_text() if fig._suptitle else ""
    check("suptitle names the suspect count", "SUSPECT RADII" in title,
          "got %r" % title)
    bars = fig.axes[0].patches
    reds = sum(1 for b in bars
               if matplotlib.colors.to_hex(b.get_facecolor()) == "#d62728")
    check("2 bars drawn in the warning colour", reds == 2, "got %d" % reds)
    plt.close(fig)

    # and with no suspect cells, the title should not mention it
    s0_ok = s0.copy(); s0_ok["resistance_trustworthy"] = True
    fig2 = s0p.s1_batch(s0_ok)
    t2 = fig2._suptitle.get_text() if fig2._suptitle else ""
    check("clean bank title is quiet", "SUSPECT" not in t2, "got %r" % t2)
    plt.close(fig2)


def test_missing_optional_columns():
    print("\n[11] missing optional columns do not raise")
    sf = make_spine_frame(n=100).drop(columns=["neck_r_mean_um"])
    try:
        plt.close(s0p.neck_geometry({"c0": sf}))
        check("neck_geometry without neck_r_mean_um", True)
    except Exception as exc:                            # noqa: BLE001
        check("neck_geometry without neck_r_mean_um", False,
              "%s: %s" % (type(exc).__name__, exc))

    tf = make_trunc_frame(n=80).drop(columns=["truncation_basis",
                                              "dist_to_z_bound_um"])
    try:
        plt.close(s0p.truncation_diagnostics({"c0": tf}))
        check("truncation_diagnostics without optional cols", True)
    except Exception as exc:                            # noqa: BLE001
        check("truncation_diagnostics without optional cols", False,
              "%s: %s" % (type(exc).__name__, exc))

    # a required column missing must raise a NAMED error, not a KeyError
    bad = make_spine_frame(n=50).drop(columns=["g_per_cm"])
    try:
        s0p.neck_resistance_sweep({"c0": bad})
        check("missing required column raises", False, "no exception")
    except ValueError as exc:
        check("missing required column raises ValueError naming it",
              "g_per_cm" in str(exc), "got %s" % exc)
    except Exception as exc:                            # noqa: BLE001
        check("missing required column raises ValueError", False,
              "got %s" % type(exc).__name__)


def save_all(outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    for name, fig in ALL_FIGS.items():
        p = os.path.join(outdir, "%s.png" % name)
        fig.savefig(p, dpi=110, bbox_inches="tight")
        print("  wrote %s" % p)


def main():
    print("s1_plots smoke test  (%s)" % s0p.MODULE_VERSION)
    test_returns_figures()
    test_empty_inputs()
    test_no_necks()
    test_single_cell_and_bare_frame()
    test_no_mutation()
    test_kappa_arithmetic()
    test_suspect_radii_path()
    test_missing_optional_columns()

    if "--save" in sys.argv:
        i = sys.argv.index("--save")
        save_all(sys.argv[i + 1] if len(sys.argv) > i + 1 else "./figs")

    for fig in ALL_FIGS.values():
        plt.close(fig)

    print("\n%s" % ("-" * 62))
    if FAILURES:
        print("FAILED %d check(s): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
