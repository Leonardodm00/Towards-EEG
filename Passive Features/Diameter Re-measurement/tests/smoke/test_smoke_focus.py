"""Smoke test for the focus scores -- Block 5 in specs/SPEC.md (D-030; handoff Eqs. 1-2).

The gradient energy of a profile,
    G = B^-2 * integral over |v| <= h of (dI~/dv)^2 dv   [1/um],
is the focus rule (D-030); the dip depth of handoff Eq. 1,
    F = -ln(min I~ / B),
is kept beside it as the labelled comparison.

Checks
    test_known_answer   a sampled Gaussian dip I/B = 1 - D exp(-v^2 / (2 w^2)),
                        smoothing off: G over the whole profile equals
                        sqrt(pi) D^2 / (2 w), and over |v| <= h equals
                        (D^2 / w) [sqrt(pi)/2 erf(h/w) - (h/w) exp(-h^2/w^2)],
                        both within the central-difference error; F equals
                        -ln(1 - D) to roundoff; the radius window is r + margin
    test_reference      G and F against loops written out by hand (central
                        differences, trapezoid, Gaussian weights)
    test_convergence    G of the sampled dip converges to the closed form at
                        order 2 in the step (central differences)
    test_invariants     G and F do not change when the whole profile is scaled
                        (a camera gain); mirror symmetry; G >= 0 and G = 0 on a
                        flat profile; at a fixed dip area G w^3 is constant
                        (faint limit); plane_scores holds every rule of
                        config.FOCUS_RULES
    test_contract       floats; NaN for a non-finite sample or B <= 0; the
                        radius window refuses a missing radius
    test_determinism    a pure function: equal inputs, equal outputs
    test_edge_cases     rendered planes (Block 4, default kernel, noise and
                        JPEG): a dark isolated tube (d 1.5 um, mu 1/um), whose
                        dip depth is pulled one plane toward the light (the
                        partition's focus shift), is found at its own plane by
                        G; a thin tube (d 0.5 um) under a thick one (d 2 um)
                        crossing 1.4 um deeper at 60 deg is found within one
                        plane by G over the radius window, while G over the
                        whole profile and the dip depth go 2 or more planes
                        toward the thick one (the scene discriminates)

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_focus.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for p in (SRC, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from allen_diameter import config as C  # noqa: E402
from allen_diameter.analysis import focus as FO  # noqa: E402
from allen_diameter.analysis import profiles as PR  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402

SEED = 20261007
REPORT_PACKAGES = ("numpy", "scipy")


def measure(**changes):
    """MeasureConfig of the defaults with some fields replaced (validated through a DiameterConfig)."""
    base = C.default_config()
    cfg = dataclasses.replace(base, measure=dataclasses.replace(base.measure, **changes))
    cfg.validate()
    return cfg.measure


def gauss_dip(v, B, D, w, v0=0.0):
    return B * (1.0 - D * np.exp(-0.5 * ((v - v0) / w) ** 2))


def g_closed(D, w, h=math.inf):
    """Gradient energy of the continuous dip over |v| <= h (1/um)."""
    if math.isinf(h):
        return math.sqrt(math.pi) * D ** 2 / (2.0 * w)
    a = h / w
    return D ** 2 / w * (0.5 * math.sqrt(math.pi) * math.erf(a) - a * math.exp(-a * a))


def fine_v(step, half=3.0):
    n = int(round(half / step))
    return np.arange(-n, n + 1) * step


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    m = measure(focus_smooth_px=0.0, focus_grad_window="whole_profile")
    B, D, w, step = 180.0, 0.3, 0.2, 0.002
    v = fine_v(step)
    I = gauss_dip(v, B, D, w)
    Gw, Bw = FO.gradient_energy(I, v, m)
    # B: the median of |v| > 1.5 um, where the dip is below exp(-28) of D -> B to 1e-12 relative
    assert abs(Bw - B) <= 1e-12 * B, Bw
    # central differences underestimate G by (h^2 / 3) int (f'')^2 / int (f')^2 = h^2 / (2 w^2) relative
    # for this dip (test_convergence measures the constant); at h = 0.002, w = 0.2: 5.0e-5
    rel = Gw / g_closed(D, w) - 1.0
    assert -6e-5 <= rel <= -4e-5, ("whole profile", rel)
    for h in (0.12, 0.2, 0.3, 0.5):    # a window cutting the dip (margin >= one profile step): the partial integral
        m_h = measure(focus_smooth_px=0.0, focus_grad_window="radius_margin", focus_grad_margin_um=h)
        Gh = FO.gradient_energy(I, v, m_h, radius_um=0.0)[0]
        # the same central-difference error, over part of the dip: 1.6e-5 to 8.2e-5 observed at these h
        assert abs(Gh / g_closed(D, w, h) - 1.0) <= 1.5e-4, (h, Gh, g_closed(D, w, h))
    # the radius window is r + margin
    m_r = measure(focus_grad_window="radius_margin", focus_grad_margin_um=0.5)
    assert FO.grad_half_width(v, m_r, 0.25) == 0.75
    assert FO.grad_half_width(v, measure(focus_grad_window="whole_profile"), 0.25) == float(np.max(np.abs(v)))
    # dip depth, smoothing off, a sample at v = 0: F = -ln(1 - D)
    F, I_min, B_F = FO.dip_depth(I, v, m)
    assert abs(F + math.log(1.0 - D)) <= 1e-12 and abs(I_min - B * (1.0 - D)) <= 1e-9, (F, I_min)
    # best_plane picks the gradient-energy maximum among planes (k* rule, D-030)
    z = np.arange(-3, 4) * 0.28
    scores = [FO.gradient_energy(gauss_dip(v, B, D, ww), v, m)[0] for ww in (0.5, 0.35, 0.25, 0.2, 0.25, 0.35, 0.5)]
    assert FO.best_plane(scores, z, C.default_config().measure, 0.28)[0] == 3


def test_reference():
    m = C.default_config().measure          # smoothing 1 sample, radius window 0.5 um
    rng = np.random.default_rng(SEED)
    v = PR.profile_offsets(m)
    I = gauss_dip(v, 170.0, 0.4, 0.15, 0.06) + rng.normal(0.0, 2.0, v.size)
    r = 0.3
    # Gaussian weights of s = 1 sample (scipy truncates at 4 s: 9 taps), edges by the nearest sample
    s, half = m.focus_smooth_px, int(4.0 * m.focus_smooth_px + 0.5)
    taps = [math.exp(-0.5 * (t / s) ** 2) for t in range(-half, half + 1)]
    tot = sum(taps)
    n = v.size
    sm = [sum(taps[t + half] * I[min(max(i + t, 0), n - 1)] for t in range(-half, half + 1)) / tot for i in range(n)]
    dv = [v[i + 1] - v[i] for i in range(n - 1)]
    g = [(sm[1] - sm[0]) / dv[0]] + [(sm[i + 1] - sm[i - 1]) / (v[i + 1] - v[i - 1]) for i in range(1, n - 1)] \
        + [(sm[n - 1] - sm[n - 2]) / dv[-1]]
    idx = [i for i in range(n) if abs(v[i]) <= r + m.focus_grad_margin_um + 1e-9]
    integral = sum(0.5 * (g[a] ** 2 + g[b] ** 2) * (v[b] - v[a]) for a, b in zip(idx[:-1], idx[1:]))
    ends = sorted(I[i] for i in range(n) if abs(v[i]) > m.focus_bg_ends_um)
    k = len(ends)
    B = ends[k // 2] if k % 2 else 0.5 * (ends[k // 2 - 1] + ends[k // 2])
    G_hand = integral / B ** 2
    G_lib, B_lib = FO.gradient_energy(I, v, m, radius_um=r)
    assert abs(B_lib - B) <= 1e-12 * B and abs(G_lib - G_hand) <= 1e-12 * G_hand, (G_lib, G_hand)
    F_lib = FO.dip_depth(I, v, m)[0]
    assert abs(F_lib + math.log(min(sm) / B)) <= 1e-12, (F_lib, -math.log(min(sm) / B))


def test_convergence():
    m = measure(focus_smooth_px=0.0, focus_grad_window="whole_profile")
    D, w = 0.3, 0.2
    steps = np.array([0.04, 0.02, 0.01, 0.005])
    err = np.array([abs(FO.gradient_energy(gauss_dip(fine_v(h), 1.0, D, w), fine_v(h), m)[0] / g_closed(D, w) - 1.0)
                    for h in steps])
    order = np.log(err[:-1] / err[1:]) / np.log(2.0)
    # central differences: order 2; the trapezoid of a smooth integrand that vanishes at both ends is
    # spectrally accurate, so it does not lower the order
    assert np.all((order > 1.9) & (order < 2.1)), (err, order)
    const = err[-1] / steps[-1] ** 2 * w ** 2
    assert abs(const - 0.5) <= 0.01, const      # (h^2 / 3) * (3 / (2 w^2)): the constant of test_known_answer


def test_invariants():
    m = C.default_config().measure
    rng = np.random.default_rng(SEED)
    v = PR.profile_offsets(m)
    I = gauss_dip(v, 150.0, 0.35, 0.2, -0.05) + rng.normal(0.0, 1.5, v.size)
    G0, F0 = FO.gradient_energy(I, v, m, radius_um=0.4)[0], FO.dip_depth(I, v, m)[0]
    for gain in (0.37, 2.5):            # a camera gain scales profile and B alike
        assert abs(FO.gradient_energy(gain * I, v, m, radius_um=0.4)[0] / G0 - 1.0) <= 1e-12
        assert abs(FO.dip_depth(gain * I, v, m)[0] - F0) <= 1e-12
    # mirror symmetry (v is symmetric about 0)
    assert abs(FO.gradient_energy(I[::-1], v, m, radius_um=0.4)[0] / G0 - 1.0) <= 1e-12
    assert G0 > 0 and FO.gradient_energy(np.full(v.size, 140.0), v, m, radius_um=0.4)[0] <= 1e-24  # roundoff of the weights
    # faint limit, fixed area A = sqrt(2 pi) D w: G w^3 is constant (fine sampling, no smoothing)
    mf = measure(focus_smooth_px=0.0, focus_grad_window="whole_profile")
    vf = fine_v(0.002)
    A = 0.06
    gw3 = [FO.gradient_energy(gauss_dip(vf, 1.0, A / (math.sqrt(2 * math.pi) * w), w), vf, mf)[0] * w ** 3
           for w in (0.15, 0.2, 0.3, 0.45)]
    assert max(gw3) / min(gw3) - 1.0 <= 1e-3, gw3      # the discretisation error, h^2 / (2 w^2) <= 9e-5 here
    assert abs(gw3[0] - A ** 2 / (4.0 * math.sqrt(math.pi))) <= 1e-3 * gw3[0], gw3[0]
    # one score per rule, under the configuration's names
    assert set(FO.plane_scores(I, v, m, radius_um=0.4)) == set(C.FOCUS_RULES)


def test_contract():
    m = C.default_config().measure
    v = PR.profile_offsets(m)
    I = gauss_dip(v, 150.0, 0.3, 0.2)
    G_, B_ = FO.gradient_energy(I, v, m, radius_um=0.3)
    assert isinstance(G_, float) and isinstance(B_, float) and math.isfinite(G_)
    F, I_min, B2 = FO.dip_depth(I, v, m)
    assert all(isinstance(x, float) for x in (F, I_min, B2))
    bad = I.copy()
    bad[5] = np.nan
    assert math.isnan(FO.gradient_energy(bad, v, m, radius_um=0.3)[0]) and math.isnan(FO.dip_depth(bad, v, m)[0])
    m_bb = measure(focus_bg_rule="same_as_bbar")
    assert math.isnan(FO.gradient_energy(I, v, m_bb, B_override=0.0, radius_um=0.3)[0])
    assert math.isnan(FO.gradient_energy(I, v, m_bb, B_override=None, radius_um=0.3)[0])
    for r in (None, float("nan"), -0.1):
        try:
            FO.gradient_energy(I, v, m, radius_um=r)
        except ValueError:
            continue
        raise AssertionError("radius_margin accepted radius %r" % (r,))
    for vv in (v[::-1], v[:2]):
        try:
            FO.gradient_energy(I[:vv.size], vv, m, radius_um=0.3)
        except ValueError:
            continue
        raise AssertionError("gradient_energy accepted a bad v")


def test_determinism():
    m = C.default_config().measure
    v = PR.profile_offsets(m)
    I = gauss_dip(v, 150.0, 0.3, 0.2) + np.random.default_rng(SEED).normal(0, 2.0, v.size)
    a = FO.plane_scores(I, v, m, radius_um=0.3)
    b = FO.plane_scores(I.copy(), v.copy(), m, radius_um=0.3)
    assert a == b, (a, b)


# rendered planes ------------------------------------------------------------

def _scene_picks(tubes, mus, radius, cfg, seed, th, o=(0.03, -0.02), nk=4):
    """k* of each rule over planes -3..3 around plane 0, profiles along the true line through o."""
    import test_smoke_node_pipeline as T      # the shared synthetic provider (Block 4 renderer + camera)
    p = cfg.acquisition.res0_um
    left, top = int(math.floor((o[0] - 4.0) / p)), int(math.floor((o[1] - 4.0) / p))
    width = int(math.ceil(8.0 / p))
    block, ks, valid, frame = T.make_provider(tubes, mus, cfg, seed)(left, top, width, width, -nk, nk)
    v = PR.profile_offsets(cfg.measure)
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    keep = np.abs(ks) <= 3
    picks = {}
    for name, mcfg in (("radius", cfg.measure),
                       ("whole", dataclasses.replace(cfg.measure, focus_grad_window="whole_profile"))):
        scores = [FO.plane_scores(PR.sample_profile(block[i], frame, np.array(o), y_hat, e_u, v), v, mcfg,
                                  radius_um=radius) for i in range(ks.size)]
        G_ = np.array([s["gradient_energy"] for s in scores])
        picks[name] = int(ks[keep][int(np.nanargmax(G_[keep]))])
        if name == "radius":
            D_ = np.array([s["dip_depth"] for s in scores])
            picks["depth"] = int(ks[keep][int(np.nanargmax(D_[keep]))])
    return picks


def test_edge_cases():
    import test_smoke_node_pipeline as T
    cfg = T.small_cfg()
    th = 0.5
    c1 = np.array([0.03, -0.02, 0.05])        # the tube centre sits in plane 0 (z 0.05 um, dz 0.28 um)
    # a dark isolated tube: the dip depth is pulled toward the light; G finds the tube's own plane
    for seed in (1, 2):
        pk = _scene_picks([G.Tube(tuple(c1), 0.75, 0.0, th, 1.0, 6.0, "axial")], [1.0], 0.75, cfg, seed, th)
        assert pk["radius"] == 0 and pk["whole"] == 0, ("dark tube", seed, pk)
        assert pk["depth"] != 0, ("dark tube: the dip depth no longer shifts, the scene does not discriminate", pk)
    # a thin tube under a thick one crossing 1.4 um deeper at 60 deg
    thin = G.Tube(tuple(c1), 0.25, 0.0, th, 1.0, 6.0, "axial")
    thick = G.Tube(tuple(c1 + np.array([0.0, 0.0, 1.4])), 1.0, 0.0, th + math.radians(60.0), 1.0, 6.0, "axial")
    for seed in (1, 2):
        pk = _scene_picks([thin, thick], [1.0, 1.0], 0.25, cfg, seed, th)
        assert abs(pk["radius"]) <= 1, ("crossing: G over the radius window", seed, pk)
        assert pk["whole"] >= 2 and pk["depth"] >= 2, ("crossing: the scene no longer discriminates", seed, pk)


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
    print(_environment())
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, headline))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
