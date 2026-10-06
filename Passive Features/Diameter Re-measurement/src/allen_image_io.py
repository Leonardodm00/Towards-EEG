"""
allen_image_io.py -- data access for Allen Cell Types images. NOTHING ELSE.

No plotting, no measurement: this module only answers "what images exist for
this specimen" and "give me these pixels". That separation is what lets the
whole stack be tested without a network (see smoke_allen_image.py, which swaps
HttpFetcher for SyntheticFetcher and exercises identical code paths).

Coordinate convention -- ASSUMPTION, verify it once per environment
-------------------------------------------------------------------
The brain-map image service is called as

    /api/v2/image_download/<image_id>?downsample=D&left=L&top=T&width=W&height=H

and this module ASSUMES L, T, W, H are in FULL-RESOLUTION pixels while the
returned image is reduced by 2**D, i.e. the returned array is about
(H / 2**D, W / 2**D). This was NOT verified from the writing environment
(api.brain-map.org is outside its egress allowlist). Call
`verify_coordinate_convention()` once from a networked machine; it fetches one
small region at two downsample levels and tells you whether the assumption
holds, with the correction to apply if it does not.
"""
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

API = "http://api.brain-map.org/api/v2"
RETRIES = 4
RETRY_SLEEP_S = 2.0
MAX_RETURNED_PX = 12_000_000    # cap on the size of the image the server SENDS BACK


def check_request_size(width, height, downsample):
    """Guard on the RETURNED image, not the full-resolution footprint.

    The region (width, height) is given in full-res pixels but the server
    returns it reduced by 2**downsample, so the quantity that actually has to
    be transferred and held in memory is (width*height) / 4**downsample.
    Guarding the footprint instead refuses whole-image overviews that are
    entirely reasonable once downsampled -- a 7607 x 9435 projection at
    downsample 4 comes back as 475 x 589.

    Returns the expected returned shape (rows, cols); raises ValueError if the
    request is genuinely too large.
    """
    f = 2 ** int(downsample)
    out_rows, out_cols = int(height) // f, int(width) // f
    if out_rows * out_cols > MAX_RETURNED_PX:
        raise ValueError(
            "request would return %d x %d = %.1f Mpx, over MAX_RETURNED_PX=%.1f Mpx; "
            "raise downsample (each step divides the returned pixels by 4) or "
            "shrink the region" % (out_cols, out_rows, out_rows * out_cols / 1e6,
                                   MAX_RETURNED_PX / 1e6))
    return out_rows, out_cols


# ------------------------------------------------------------------ metadata
def api_query(criteria, num_rows=200, start_row=0, timeout=60):
    """Raw RMA query -> list of dicts. Raises on a failed query."""
    import requests
    url = ("%s/data/query.json?criteria=%s,rma::options[num_rows$eq%d][start_row$eq%d]"
           % (API, criteria, num_rows, start_row))
    last = None
    for k in range(RETRIES):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            if not j.get("success", False):
                raise RuntimeError("RMA refused the query: %s" % j.get("msg"))
            return j["msg"]
        except Exception as e:                                   # noqa: BLE001
            last = e
            if k < RETRIES - 1:
                time.sleep(RETRY_SLEEP_S * (k + 1))
    raise RuntimeError("api_query failed after %d attempts: %s" % (RETRIES, last))


def list_images(specimen_id):
    """Every SubImage of a specimen: the 4 projections and the Primary z planes.

    Returns a DataFrame sorted with projections first, then planes by
    section_number. Columns: id, image_type, section_number, width, height,
    resolution (um/px at downsample 0), data_set_id, failed.
    """
    rows = []
    for ds in _data_set_ids(specimen_id):
        start = 0
        while True:
            msg = api_query("model::SubImage,rma::criteria,data_set[id$eq%d]" % ds,
                            num_rows=1000, start_row=start)
            rows.extend(msg)
            start += len(msg)
            if not msg or len(msg) < 1000:
                break
    # projections live on the specimen, not always on a data_set
    rows.extend(api_query("model::SubImage,rma::criteria,[specimen_id$eq%d]" % specimen_id,
                          num_rows=100))
    if not rows:
        raise RuntimeError("no SubImage rows for specimen %d" % specimen_id)
    df = pd.DataFrame(rows).drop_duplicates(subset="id")
    keep = ["id", "image_type", "section_number", "width", "height", "resolution",
            "data_set_id", "failed"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["is_plane"] = df["image_type"] == "Primary"
    return df.sort_values(["is_plane", "section_number"]).reset_index(drop=True)


def _data_set_ids(specimen_id):
    msg = api_query("model::Specimen,rma::criteria,[id$eq%d],rma::include,data_sets"
                    % specimen_id, num_rows=10)
    if not msg:
        return []
    return [d["id"] for d in msg[0].get("data_sets", [])]


def plane_table(df):
    """Just the usable z planes, with a 0-based plane index anchored so that
    plane_index = section_number - min(section_number). Missing sections leave
    gaps in plane_index on purpose, so plane_index * dz is a true z distance."""
    p = df[df["image_type"] == "Primary"].copy()
    if "failed" in p.columns:
        p = p[~p["failed"].fillna(False).astype(bool)]
    if p.empty:
        raise RuntimeError("no usable Primary planes")
    p["plane_index"] = p["section_number"] - int(p["section_number"].min())
    return p.reset_index(drop=True)


def reconstruction_info(specimen_id):
    """Morphometrics + scale factors (dz lives in scale_factor_z)."""
    msg = api_query("model::NeuronReconstruction,rma::criteria,[specimen_id$eq%d]"
                    % specimen_id, num_rows=10)
    return msg[0] if msg else {}


# ------------------------------------------------------------------- frames
@dataclass(frozen=True)
class CropFrame:
    """Maps full-resolution image pixels <-> indices in a fetched crop array.

    left, top   : full-res pixel coordinates of the crop's top-left corner
    downsample  : the crop was reduced by factor f = 2**downsample
    res0_um_px  : um per pixel at downsample 0
    """
    left: int
    top: int
    downsample: int
    res0_um_px: float

    @property
    def factor(self):
        return 2 ** self.downsample

    @property
    def res_um_px(self):
        """um per pixel OF THIS ARRAY (not of the full-res image)."""
        return self.res0_um_px * self.factor

    def to_array_xy(self, x_full, y_full):
        """Full-res pixel -> (col, row) in the crop array. Floats, not rounded."""
        f = self.factor
        return (np.asarray(x_full) - self.left) / f, (np.asarray(y_full) - self.top) / f

    def to_full_xy(self, col, row):
        f = self.factor
        return np.asarray(col) * f + self.left, np.asarray(row) * f + self.top


# ------------------------------------------------------------------ fetchers
class ImageFetcher:
    """Interface: .get(image_id, left, top, width, height, downsample) -> uint8 (H, W).

    left/top/width/height are FULL-RESOLUTION pixels (see the module docstring)."""

    def get(self, image_id, left, top, width, height, downsample=0):
        raise NotImplementedError


class HttpFetcher(ImageFetcher):
    """Real fetcher, with an on-disk cache of the bytes exactly as served."""

    def __init__(self, cache_dir="allen_cache", verbose=True):
        self.cache_dir = cache_dir
        self.verbose = verbose
        os.makedirs(cache_dir, exist_ok=True)
        self.n_requests = 0
        self.n_cache_hits = 0
        self.bytes_downloaded = 0

    def _key(self, image_id, left, top, width, height, downsample):
        s = "%d_%d_%d_%d_%d_%d" % (image_id, left, top, width, height, downsample)
        return os.path.join(self.cache_dir, "%s_%s.img" % (image_id, hashlib.md5(s.encode()).hexdigest()[:12]))

    def get(self, image_id, left, top, width, height, downsample=0):
        from PIL import Image
        check_request_size(width, height, downsample)
        path = self._key(image_id, left, top, width, height, downsample)
        if os.path.exists(path):
            self.n_cache_hits += 1
            with open(path, "rb") as f:
                return np.asarray(Image.open(io.BytesIO(f.read())).convert("L"))
        data = self._download(image_id, left, top, width, height, downsample)
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)                     # stored as served; never re-encoded
        os.replace(tmp, path)
        return np.asarray(Image.open(io.BytesIO(data)).convert("L"))

    def _download(self, image_id, left, top, width, height, downsample):
        import requests
        url = ("%s/image_download/%d?downsample=%d&left=%d&top=%d&width=%d&height=%d"
               % (API, image_id, downsample, left, top, width, height))
        last = None
        for k in range(RETRIES):
            try:
                r = requests.get(url, timeout=180)
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                if not ctype.startswith("image"):
                    raise RuntimeError("server returned %s, not an image: %s"
                                       % (ctype, r.text[:200]))
                self.n_requests += 1
                self.bytes_downloaded += len(r.content)
                if self.verbose:
                    print("  fetched id=%d ds=%d %dx%d px (%.0f kB)"
                          % (image_id, downsample, width, height, len(r.content) / 1e3))
                return r.content
            except Exception as e:                               # noqa: BLE001
                last = e
                if k < RETRIES - 1:
                    time.sleep(RETRY_SLEEP_S * (k + 1))
        raise RuntimeError("image_download failed after %d attempts (%s): %s"
                           % (RETRIES, url, last))


class SyntheticFetcher(ImageFetcher):
    """Offline stand-in backed by an in-memory full-resolution array, so the
    tests exercise the same crop/downsample arithmetic as the real path."""

    def __init__(self, full_image):
        self.full = np.asarray(full_image)
        self.n_requests = 0

    def get(self, image_id, left, top, width, height, downsample=0):
        self.n_requests += 1
        H, W = self.full.shape
        out = np.full((height, width), 255, dtype=np.uint8)
        sx0, sy0 = max(0, left), max(0, top)
        sx1, sy1 = min(W, left + width), min(H, top + height)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = self.full[sy0:sy1, sx0:sx1]
        f = 2 ** downsample
        if f > 1:                     # box-average, matching a pyramid reduction
            h2, w2 = (out.shape[0] // f) * f, (out.shape[1] // f) * f
            out = out[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))
        return out.astype(np.uint8)


# --------------------------------------------------------------- convenience
def fetch_crop(fetcher, image_id, left, top, width, height, downsample, res0_um_px,
               warn_on_shape=True):
    """One region -> (array, CropFrame). The frame is what every later
    coordinate conversion goes through; never recompute it by hand.

    If the server silently caps a large request, the returned array is smaller
    than asked for and every micron figure derived from it would be wrong by
    that factor. This warns rather than failing, because a small rounding
    difference is normal.
    """
    exp_rows, exp_cols = check_request_size(width, height, downsample)
    img = fetcher.get(image_id, int(left), int(top), int(width), int(height), int(downsample))
    if warn_on_shape:
        got_rows, got_cols = img.shape[0], img.shape[1]
        if abs(got_rows - exp_rows) > 2 or abs(got_cols - exp_cols) > 2:
            print("[warn] asked for %dx%d at downsample %d (expected %dx%d back) but got "
                  "%dx%d -- the server may be capping the request; microns per pixel "
                  "would be wrong. Re-check with a smaller region."
                  % (width, height, downsample, exp_cols, exp_rows, got_cols, got_rows))
    return img, CropFrame(int(left), int(top), int(downsample), float(res0_um_px))


def fetch_whole(fetcher, row, downsample):
    """Whole image at a reduced resolution, in ONE request. `row` is a record
    from list_images (needs id, width, height, resolution)."""
    W, H = int(row["width"]), int(row["height"])
    return fetch_crop(fetcher, int(row["id"]), 0, 0, W, H, downsample, float(row["resolution"]))


def verify_coordinate_convention(fetcher, row, probe_px=512):
    """Fetch the same full-res region at downsample 0 and 2 and report which
    convention the server actually uses. Run this ONCE on a networked machine."""
    left = max(0, int(row["width"]) // 2 - probe_px // 2)
    top = max(0, int(row["height"]) // 2 - probe_px // 2)
    a = fetcher.get(int(row["id"]), left, top, probe_px, probe_px, 0)
    b = fetcher.get(int(row["id"]), left, top, probe_px, probe_px, 2)
    ratio = a.shape[0] / max(b.shape[0], 1)
    verdict = ("full-res coords (as assumed)" if abs(ratio - 4) < 0.6 else
               "DOWNSAMPLED coords -- divide left/top/width/height by 2**downsample"
               if abs(ratio - 1) < 0.2 else "UNEXPECTED -- inspect manually")
    return dict(shape_ds0=a.shape, shape_ds2=b.shape, height_ratio=ratio, verdict=verdict)


# ------------------------------------------------------------------- SWC
SWC_FILE_TYPE_ID = 303941301        # well_known_file_type for the reconstruction


def fetch_swc(specimen_id, out_dir, overwrite=False):
    """Download the Allen reconstruction SWC for a specimen. Returns the path.

    The file is attached to the NeuronReconstruction as a well_known_file and
    served from /api/v2/well_known_file_download/<id>. Selection is by the .swc
    path suffix, with the type id as a fallback, because the same
    NeuronReconstruction also carries a .marker file and a summary .png.
    """
    import requests
    os.makedirs(out_dir, exist_ok=True)
    msg = api_query("model::NeuronReconstruction,rma::criteria,[specimen_id$eq%d],"
                    "rma::include,well_known_files" % specimen_id, num_rows=10)
    if not msg:
        raise RuntimeError("no NeuronReconstruction for specimen %d" % specimen_id)
    files = msg[0].get("well_known_files", [])
    cand = [f for f in files if str(f.get("path", "")).lower().endswith(".swc")]
    if not cand:
        cand = [f for f in files if f.get("well_known_file_type_id") == SWC_FILE_TYPE_ID]
    if not cand:
        raise RuntimeError("no .swc among the %d well_known_files for specimen %d: %s"
                           % (len(files), specimen_id,
                              [os.path.basename(str(f.get("path"))) for f in files]))
    wkf = cand[0]
    name = os.path.basename(str(wkf["path"]))
    dest = os.path.join(out_dir, name)
    if os.path.exists(dest) and not overwrite:
        print("cached: %s" % dest)
        return dest
    url = "%s/well_known_file_download/%d" % (API, int(wkf["id"]))
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, dest)
    print("downloaded %s (%.0f kB) from well_known_file %d"
          % (name, len(r.content) / 1e3, int(wkf["id"])))
    return dest


# ---------------------------------------------------------------- z block
def fetch_zblock(fetcher, planes, k_lo, k_hi, left, top, width, height, res0_um_px):
    """The same full-resolution xy crop from every plane with plane_index in
    [k_lo, k_hi], stacked into (n_planes, height, width).

    Returns (block, ks, valid, frame):
      block : uint8 array; a missing section is left white (255)
      ks    : plane_index of each block slice -- block[j] is plane ks[j]
      valid : False where the section is missing, so a gap is never mistaken
              for an empty plane
      frame : CropFrame of the xy crop (downsample 0)
    """
    by_k = {int(k): int(i) for k, i in zip(planes["plane_index"], planes["id"])}
    ks = np.arange(int(k_lo), int(k_hi) + 1)
    block = np.full((len(ks), int(height), int(width)), 255, dtype=np.uint8)
    valid = np.zeros(len(ks), dtype=bool)
    for j, k in enumerate(ks):
        if int(k) not in by_k:
            continue
        img, _ = fetch_crop(fetcher, by_k[int(k)], left, top, width, height, 0, res0_um_px,
                            warn_on_shape=False)
        h, w = min(img.shape[0], int(height)), min(img.shape[1], int(width))
        block[j, :h, :w] = img[:h, :w]
        valid[j] = True
    return block, ks, valid, CropFrame(int(left), int(top), 0, float(res0_um_px))
