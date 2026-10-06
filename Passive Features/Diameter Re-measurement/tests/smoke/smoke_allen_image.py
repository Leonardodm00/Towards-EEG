"""
smoke_allen_image.py -- offline self-test for the Allen slice viewer.

Runs with NO network: SyntheticFetcher stands in for the image service, so the
crop arithmetic, the coordinate frame, the FWHM and the SWC overlay are all
exercised on data whose ground truth is known by construction.

Run:      python smoke_allen_image.py
Expect:   "7/7 passed"  and exit status 0.

What each check would catch if it failed
  1 crop arithmetic      -- an off-by-one or a transposed axis in fetch_crop
  2 downsample scaling   -- res_um_px not tracking the pyramid factor
  3 frame round-trip     -- to_array_xy / to_full_xy disagreeing
  4 FWHM on a known bar  -- a wrong half-max definition or interpolation
  5 faint / clipped      -- NaN not returned where the profile is unusable
  6 SWC overlay          -- skeleton landing on the wrong pixels
  7 plotting runs        -- a rendering path that raises on real inputs
"""
import math
import sys

import numpy as np
from scipy import ndimage

import allen_image_align as aia
import allen_image_io as aio
import allen_image_measure as aim

RES0 = 0.1144                      # um/px, the Allen 63x value
PASS, FAIL = "PASS", "FAIL"


def _synthetic_field(H=900, W=1200, bar_um=1.0, y_bar=400, rng_seed=0):
    """Bright field with one horizontal dark bar of known width, plus noise."""
    rng = np.random.default_rng(rng_seed)
    img = np.full((H, W), 200.0, dtype=np.float64)
    half_px = 0.5 * bar_um / RES0
    ys = np.arange(H)
    mask = np.abs(ys - y_bar) <= half_px
    img[mask, :] = 40.0
    img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255)
    return img.astype(np.uint8)


def check_1_crop(results):
    full = _synthetic_field()
    fetch = aio.SyntheticFetcher(full)
    img, frame = aio.fetch_crop(fetch, 1, left=300, top=350, width=200, height=120,
                                downsample=0, res0_um_px=RES0)
    ok = img.shape == (120, 200) and np.array_equal(img, full[350:470, 300:500])
    results.append(("1 crop arithmetic: exact region returned", ok))


def check_2_downsample(results):
    full = _synthetic_field()
    fetch = aio.SyntheticFetcher(full)
    img, frame = aio.fetch_crop(fetch, 1, 0, 0, 512, 512, downsample=2, res0_um_px=RES0)
    ok = (img.shape == (128, 128)
          and abs(frame.res_um_px - RES0 * 4) < 1e-12
          and frame.factor == 4)
    results.append(("2 downsample: shape /4 and res_um_px x4", ok))


def check_3_frame(results):
    frame = aio.CropFrame(left=300, top=350, downsample=2, res0_um_px=RES0)
    xf, yf = 1234.0, 987.0
    c, r = frame.to_array_xy(xf, yf)
    xb, yb = frame.to_full_xy(c, r)
    ok = abs(xb - xf) < 1e-9 and abs(yb - yf) < 1e-9 and abs(c - (1234 - 300) / 4) < 1e-9
    results.append(("3 CropFrame round-trip full <-> array", ok))


def check_4_fwhm(results):
    detail = []
    ok = True
    for bar_um in (0.6, 1.0, 2.0):
        full = _synthetic_field(bar_um=bar_um)
        fetch = aio.SyntheticFetcher(full)
        img, frame = aio.fetch_crop(fetch, 1, 500, 340, 200, 120, 0, RES0)
        centre = (100.0, 400 - 340.0)               # (col, row) of the bar centre
        t, v = aim.perpendicular_profile(img, centre, (1.0, 0.0), frame.res_um_px,
                                         half_len_um=3.0)
        w, info = aim.fwhm_um(t, v)
        good = np.isfinite(w) and abs(w - bar_um) / bar_um < 0.12
        ok &= good
        detail.append("%.1f->%.2f" % (bar_um, w))
    results.append(("4 FWHM recovers 0.6/1.0/2.0 um within 12%% (%s)"
                    % ", ".join(detail), ok))


def check_5_unusable(results):
    # (a) faint: bar only 4 levels darker than background
    img = np.full((60, 60), 200, dtype=np.uint8)
    img[28:32, :] = 196
    t, v = aim.perpendicular_profile(img, (30.0, 30.0), (1.0, 0.0), RES0, 3.0)
    w_faint, info_faint = aim.fwhm_um(t, v)
    # (b) clipped: dark everywhere in the window, never returns to background
    img2 = np.full((60, 60), 40, dtype=np.uint8)
    img2[0, :] = 40
    t2, v2 = aim.perpendicular_profile(img2, (30.0, 30.0), (1.0, 0.0), RES0, 3.0)
    w_clip, info_clip = aim.fwhm_um(t2, v2)
    ok = (not np.isfinite(w_faint)) and "faint" in info_faint["reason"] and (not np.isfinite(w_clip))
    results.append(("5 unusable profiles return NaN with a stated reason", ok))


def check_6_overlay(results):
    """A two-node SWC whose segment is known to cross a given array pixel."""
    import pandas as pd
    frame = aio.CropFrame(left=1000, top=2000, downsample=1, res0_um_px=RES0)
    # node at full-res px (1100, 2100) -> array (col,row) = (50, 50) at ds=1
    x_um = 1100 * RES0
    y_um = 2100 * RES0
    swc = pd.DataFrame([(1, 1, x_um, y_um, 0.0, 5.0, -1),
                        (2, 3, x_um + 10 * RES0, y_um, 0.0, 0.5, 1)],
                       columns=["id", "type", "x", "y", "z", "r", "parent"])
    col, row = frame.to_array_xy(swc["x"].values / RES0, swc["y"].values / RES0)
    ok = abs(col[0] - 50.0) < 1e-6 and abs(row[0] - 50.0) < 1e-6 and abs(col[1] - 55.0) < 1e-6
    results.append(("6 SWC microns -> array pixels lands where expected", ok))


def check_7_plotting(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import allen_image_plot as aip
    try:
        full = _synthetic_field(bar_um=1.0)
        fetch = aio.SyntheticFetcher(full)
        img, frame = aio.fetch_crop(fetch, 1, 500, 340, 200, 120, 0, RES0)
        ax = aip.show_image(img, frame.res_um_px, title="synthetic")
        t, v = aim.perpendicular_profile(img, (100.0, 60.0), (1.0, 0.0), frame.res_um_px, 3.0)
        w, info = aim.fwhm_um(t, v)
        aip.plot_profile(t, v, info, w, title="profile")
        big, _ = aio.fetch_crop(fetch, 1, 0, 0, 400, 400, 2, RES0)
        aip.montage([big, big, big], frame.res_um_px * 4, titles=["a", "b", "c"], ncols=3)
        plt.close("all")
        ok = True
    except Exception as e:                                        # noqa: BLE001
        print("    plotting raised:", type(e).__name__, e)
        ok = False
    results.append(("7 show_image / plot_profile / montage all render", ok))


def check_8_dark_centre(results):
    """A single dark blob off-centre must be found, not the image centre."""
    img = np.full((400, 600), 210, dtype=np.uint8)
    img[300:340, 120:180] = 30          # blob centred at (col,row) = (150, 320)
    c, r = aim.centre_of_dark_mass(img, percentile=99.0)
    ok = abs(c - 150) < 4 and abs(r - 320) < 4
    results.append(("8 centre_of_dark_mass finds an off-centre blob (got %.0f,%.0f)"
                    % (c, r), ok))


def check_9_size_guard(results):
    """The guard must measure the RETURNED image, not the full-res footprint.

    Regression test for a real failure: a 7607 x 9435 projection requested at
    downsample 4 returns 475 x 589 and must be ALLOWED; the same region at
    downsample 0 returns 71.8 Mpx and must be REFUSED.
    """
    W, H = 7607, 9435
    ok = True
    try:
        rows, cols = aio.check_request_size(W, H, 4)      # must pass
        ok &= (rows, cols) == (H // 16, W // 16)
    except ValueError as e:
        print("    downsample 4 wrongly refused:", e)
        ok = False
    try:
        aio.check_request_size(W, H, 0)                   # must refuse
        print("    downsample 0 wrongly allowed (71.8 Mpx)")
        ok = False
    except ValueError:
        pass
    results.append(("9 size guard: whole projection allowed at ds=4, refused at ds=0", ok))


def check_10_whole_overview(results):
    """End-to-end: fetch_whole on a projection-sized image must succeed and
    carry the right micron scale -- the exact call that failed in Colab."""
    full = _synthetic_field(H=1200, W=1000)
    fetch = aio.SyntheticFetcher(full)
    row = dict(id=1, width=1000, height=1200, resolution=RES0)
    img, frame = aio.fetch_whole(fetch, row, 4)
    ok = (img.shape == (1200 // 16, 1000 // 16)
          and abs(frame.res_um_px - RES0 * 16) < 1e-12)
    results.append(("10 fetch_whole at ds=4 returns the reduced image and scale", ok))



def _skeleton_swc(n_branch=5, length_um=120.0, step_um=4.0, x0_um=60.0, y0_um=70.0,
                  seed=3):
    """A small radial tree in microns, with a soma node at its centre."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = [(1, 1, x0_um, y0_um, 0.0, 5.0, -1)]
    nid = 2
    for b in range(n_branch):
        a = 2 * math.pi * b / n_branch
        par, x, y = 1, x0_um, y0_um
        for _ in range(int(length_um / step_um)):
            a += rng.normal(0, 0.12)
            x += step_um * math.cos(a)
            y += step_um * math.sin(a)
            rows.append((nid, 3, x, y, 0.0, 0.5, par))
            par = nid
            nid += 1
    return pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"])


def check_11_align_recovers_shift(results):
    """Paint a skeleton into an image at a KNOWN offset, then recover it."""
    res = 0.5                       # um/px, an overview-like scale
    rows, cols = 400, 400
    swc = _skeleton_swc()
    true_dx, true_dy = 37.0, -23.0
    mask = aia.render_skeleton_mask(swc, (rows, cols), res, (true_dx, true_dy))
    img = np.full((rows, cols), 205, dtype=np.float64)
    img[ndimage.binary_dilation(mask, iterations=1)] = 45
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)

    out = aia.align_translation(swc, img, res, max_shift_um=120.0)
    ok = (abs(out["dx_px"] - true_dx) <= 2 and abs(out["dy_px"] - true_dy) <= 2
          and out["flip_y"] is False)
    results.append(("11 align recovers a known shift (%.0f,%.0f -> %.0f,%.0f), no flip"
                    % (true_dx, true_dy, out["dx_px"], out["dy_px"]), ok))


def check_12_align_detects_flip(results):
    """Same tree painted y-flipped: the flipped hypothesis must win."""
    res = 0.5
    rows, cols = 400, 400
    swc = _skeleton_swc(seed=7)
    mask = aia.render_skeleton_mask(swc, (rows, cols), res, (10.0, 5.0), flip_y_rows=rows)
    img = np.full((rows, cols), 205, dtype=np.float64)
    img[ndimage.binary_dilation(mask, iterations=1)] = 45
    img = np.clip(img + np.random.default_rng(1).normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    out = aia.align_translation(swc, img, res, max_shift_um=120.0)
    ok = out["flip_y"] is True and out["score_flip"] > 2 * max(out["score_noflip"], 1e-6)
    results.append(("12 align picks the y-flip when the image is flipped "
                    "(flip %.1f vs noflip %.1f)" % (out["score_flip"], out["score_noflip"]), ok))


def check_13_frame_diagnosis(results):
    """diagnose_frame must separate 'fits' from 'bigger than the image'."""
    swc = _skeleton_swc(x0_um=400.0, y0_um=500.0, length_um=300.0)
    good = aia.diagnose_frame(swc, 0.1144, 7607, 9435)       # fits comfortably
    bad = aia.diagnose_frame(swc, 0.1144, 200, 200)          # image far too small
    ok = ("image microns" in good["verdict"]) and ("LARGER" in bad["verdict"])
    results.append(("13 diagnose_frame separates in-frame from out-of-scale", ok))



# ---------------------------------------------------------------------------
# registration-check tests: a synthetic stack with one wavy 1.0 um process at a
# known depth, cluttered with dark specks at random depths, plus noise.
# ---------------------------------------------------------------------------
_RC = {}


def _rc_stack(seed=11, amp=25.0):
    key = (seed, amp)
    if key in _RC:
        return _RC[key]
    import pandas as pd
    res, dz = RES0, 0.28
    NZ, H, W = 60, 760, 540
    rng = np.random.default_rng(seed)
    # centreline (array px) and its depth (um): a gentle wave, slight z slope
    t = np.arange(50.0, 490.0, 0.25)
    cl_c = t
    cl_r = 220.0 + amp * np.sin(t / 60.0)
    cl_z = 8.0 + 2.0 * (t - t[0]) / (t[-1] - t[0])
    mask = np.zeros((H, W), bool)
    mask[np.round(cl_r).astype(int), np.round(cl_c).astype(int)] = True
    dist, (ir, ic) = ndimage.distance_transform_edt(~mask, return_indices=True)
    zmap_idx = np.full((H, W), np.nan)
    lut = {}
    for rr_, cc_, zz_ in zip(np.round(cl_r).astype(int), np.round(cl_c).astype(int), cl_z):
        lut[(rr_, cc_)] = zz_
    near_z = np.vectorize(lambda a, b: lut.get((a, b), np.nan))(ir, ic)
    tube = (dist * res <= 0.5).astype(np.float32)                    # d = 1.0 um
    tube = ndimage.gaussian_filter(tube, 0.30 / 2.355 / res)         # lateral PSF
    vol = np.empty((NZ, H, W), np.float32)
    sig_z = 0.6
    specks = [(rng.integers(0, H), rng.integers(0, W), rng.uniform(0, NZ * dz),
               rng.uniform(3, 8)) for _ in range(60)]
    yy, xx = np.mgrid[0:H, 0:W]
    speck_img = []
    for (sr, sc, sz, rad) in specks:
        speck_img.append((((yy - sr) ** 2 + (xx - sc) ** 2) <= rad ** 2, sz))
    for k in range(NZ):
        w = np.exp(-((k * dz - near_z) ** 2) / (2 * sig_z ** 2))
        dark = 160.0 * tube * np.nan_to_num(w)
        for m_, sz in speck_img:
            dark[m_] = np.maximum(dark[m_], 110.0 * np.exp(-((k * dz - sz) ** 2) / (2 * sig_z ** 2)))
        vol[k] = 205.0 - dark
    vol = np.clip(vol + rng.normal(0, 3, vol.shape), 0, 255).astype(np.uint8)
    # SWC nodes every ~1.2 um along the centreline
    step = int(round(1.2 / res / 0.25))
    sel = np.arange(0, len(t), step)
    rows = []
    for n, i in enumerate(sel):
        rows.append((n + 1, 3, cl_c[i] * res, cl_r[i] * res, cl_z[i], 0.5, n if n else -1))
    swc = pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"])
    swc.loc[0, "type"] = 1
    _RC[key] = dict(vol=vol, swc=swc, dz=dz, NZ=NZ, H=H, W=W,
                     frame=aio.CropFrame(0, 0, 0, res), ks=np.arange(NZ),
                     valid=np.ones(NZ, bool))
    return _RC[key]


def _rc_run(swc_variant, seed=11, null_seed=0, amp=25.0):
    R = _rc_stack(seed, amp)
    mid = int(swc_variant["id"].iloc[len(swc_variant) // 2])
    path = aia.path_through_node(swc_variant, mid, n_each_way=15)
    return aia.registration_check(path, R["vol"], R["ks"], R["valid"], R["frame"], R["dz"],
                                  node_id=mid, n_null=60, seed=null_seed)


def check_14_path_stops_at_branch(results):
    import pandas as pd
    # soma 1 -> 2 -> 3 -> 4 (branch) -> {5 -> 6, 7 -> 8}
    rows = [(1, 1, 0, 0, 0, 5, -1), (2, 3, 1, 0, 0, .5, 1), (3, 3, 2, 0, 0, .5, 2),
            (4, 3, 3, 0, 0, .5, 3), (5, 3, 4, 1, 0, .5, 4), (6, 3, 5, 1, 0, .5, 5),
            (7, 3, 4, -1, 0, .5, 4), (8, 3, 5, -1, 0, .5, 7)]
    swc = pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "r", "parent"])
    p_mid = list(aia.path_through_node(swc, 3, 10)["id"])      # 2,3,4 : stops AT branch 4
    p_dist = list(aia.path_through_node(swc, 6, 10)["id"])     # 4,5,6 : starts AT branch 4
    ok = p_mid == [2, 3, 4] and p_dist == [4, 5, 6]
    results.append(("14 path_through_node stops at soma and branch points (%s, %s)"
                    % (p_mid, p_dist), ok))


def check_15_registration_on(results):
    R = _rc_stack()
    chk = _rc_run(R["swc"])
    ok = (chk["verdict"].startswith("ON THE PROCESS") and abs(chk["lateral_offset_um"]) <= 0.3
          and chk["p_value"] <= 0.05 and chk["coverage"] >= 0.7 and abs(chk["z_offset_um"]) <= 0.6)
    results.append(("15 path on its process -> ON (s*=%+.2f, p=%.3f, cov=%.0f%%, dz=%+.2f)"
                    % (chk["lateral_offset_um"], chk["p_value"], 100 * chk["coverage"],
                       chk["z_offset_um"]), ok))


def check_16_registration_alongside(results):
    """Displace every node 1.5 um along its local normal: s* must come back as -1.5."""
    R = _rc_stack()
    sw = R["swc"].copy()
    res = RES0
    x, y = sw["x"].values / res, sw["y"].values / res
    tc = np.gradient(x); tr = np.gradient(y)
    nrm = np.hypot(tc, tr); tc, tr = tc / nrm, tr / nrm
    d = 1.5 / res
    sw["x"] = (x + d * (-tr)) * res
    sw["y"] = (y + d * tc) * res
    chk = _rc_run(sw)
    ok = chk["verdict"].startswith("ALONGSIDE") and abs(chk["lateral_offset_um"] + 1.5) <= 0.3
    results.append(("16 path displaced +1.5 um -> ALONGSIDE, s* = %+.2f (expect -1.50)"
                    % chk["lateral_offset_um"], ok))


def check_17_registration_z_offset(results):
    R = _rc_stack()
    sw = R["swc"].copy()
    sw["z"] = sw["z"] + 2.0
    chk = _rc_run(sw)
    ok = abs(chk["z_offset_um"] + 2.0) <= 0.6 and chk["verdict"].startswith("ON THE PROCESS")
    results.append(("17 SWC z +2.0 um -> focus peak at %+.2f um (expect -2.0)"
                    % chk["z_offset_um"], ok))


def check_18_registration_empty(results):
    R = _rc_stack()
    sw = R["swc"].copy()
    sw["y"] = sw["y"] + 300 * RES0          # same shape, dropped into empty tissue
    chk = _rc_run(sw)
    ok = chk["verdict"].startswith("NOT ON")
    results.append(("18 path over empty tissue -> NOT ON (p=%.3f, coverage %.0f%%)"
                    % (chk["p_value"], 100 * chk["coverage"]), ok))


def check_19_plan_block(results):
    R = _rc_stack()
    path = aia.path_through_node(R["swc"], int(R["swc"]["id"].iloc[10]), 15)
    plan = aia.plan_path_block(path, RES0, R["dz"], margin_um=20.0, dz_max_um=4.0,
                               k_min=0, k_max=10 ** 6)
    x, y = aia.swc_to_full_px(path, RES0)
    k = path["z"].values / R["dz"]
    m = 20.0 / RES0
    ok = (x.min() - plan["left"] >= m - 1 and plan["left"] + plan["width"] - x.max() >= m - 1
          and y.min() - plan["top"] >= m - 1 and plan["top"] + plan["height"] - y.max() >= m - 1
          and plan["k_lo"] <= k.min() - 4.0 / R["dz"] and plan["k_hi"] >= k.max() + 4.0 / R["dz"])
    results.append(("19 plan_path_block covers path + 20 um margin + 4 um in z", ok))


def check_20_registration_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import allen_image_plot as aip
    try:
        chk = _rc_run(_rc_stack()["swc"])
        aip.plot_registration_check(chk, RES0)
        plt.close("all")
        ok = True
    except Exception as e:                                        # noqa: BLE001
        print("    plot_registration_check raised:", type(e).__name__, e)
        ok = False
    results.append(("20 plot_registration_check renders", ok))


def main():
    results = []
    for fn in (check_1_crop, check_2_downsample, check_3_frame, check_4_fwhm,
               check_5_unusable, check_6_overlay, check_7_plotting,
               check_8_dark_centre, check_9_size_guard, check_10_whole_overview,
               check_11_align_recovers_shift, check_12_align_detects_flip,
               check_13_frame_diagnosis, check_14_path_stops_at_branch,
               check_15_registration_on, check_16_registration_alongside,
               check_17_registration_z_offset, check_18_registration_empty,
               check_19_plan_block, check_20_registration_figure):
        fn(results)
    print()
    for name, ok in results:
        print("[%s] %s" % (PASS if ok else FAIL, name))
    n_ok = sum(1 for _, ok in results if ok)
    print("\n%d/%d passed" % (n_ok, len(results)))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
