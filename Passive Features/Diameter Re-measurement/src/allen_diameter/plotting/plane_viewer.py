"""A crop's planes one at a time, with a slider and nothing drawn over them -- SPEC Block 44 amendment 4 (D-239).

The notebook's Cell 24 (b): the user runs through the planes of the soma's crop and picks, by eye, the plane where the
soma is in focus (D-211); the pick is handed to a callback (the cell sets SOMA_K with it). Display only: grey levels
mapped linearly onto the screen's 0..255, no measurement. PIL is imported by plane_png and ipywidgets by
PlaneViewer, both notebook-side dependencies that the pipeline never imports. Pure ASCII, LF.

Conventions (soma_view's): pixel column j of a crop whose first column is `left` has its centre at x = (left + j)
res0_um, row i at y = (top + i) res0_um; plane k lies at depth k dz_um.
"""
from __future__ import annotations

import io
import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

CONTRASTS = ("fixed", "per plane")      # one grey range for every plane / each plane its own
PLACEHOLDER_GREY = 128                  # the image shown for a plane that was not recorded


def view_slices(shape, left, top, res0_um, centre_xy_um=None, half_um=None) -> Tuple[slice, slice]:
    """(rows, cols): the slices of a crop of shape (H, W) holding the pixels whose centre lies within half_um of
    centre_xy_um in both x and y (the square view about the centre), clipped to the crop. half_um None: the whole
    crop. ValueError for a half_um that is not finite and > 0, a centre that is not finite, or a view that misses
    the crop."""
    H, W = int(shape[0]), int(shape[1])
    if half_um is None:
        return slice(0, H), slice(0, W)
    h, p = float(half_um), float(res0_um)
    if not (math.isfinite(h) and h > 0 and math.isfinite(p) and p > 0):
        raise ValueError("view_slices needs half_um and res0_um finite and > 0")
    if centre_xy_um is None:
        raise ValueError("view_slices needs a centre for a view of half_um %.3g" % h)
    cx, cy = (float(v) for v in np.asarray(centre_xy_um, dtype=float).reshape(-1)[:2])
    if not (math.isfinite(cx) and math.isfinite(cy)):
        raise ValueError("view_slices needs a finite centre")

    def span(c, first, n):
        lo = max(int(math.ceil((c - h) / p - first - 1e-9)), 0)
        hi = min(int(math.floor((c + h) / p - first + 1e-9)), n - 1)
        if hi < lo:
            raise ValueError("the view (+-%.3g um about %.3f um) misses the crop" % (h, c))
        return slice(lo, hi + 1)

    return span(cy, int(top), H), span(cx, int(left), W)


def grey_ranges(block, valid, contrast="fixed", percentiles=(0.5, 99.5)) -> np.ndarray:
    """(n_planes, 2) float: the grey range (lo, hi) each plane is shown with. "fixed": the percentiles of the pixels
    of every recorded plane together, the same row for each plane (darkening and sharpening compare across planes);
    "per plane": each recorded plane's own percentiles. NaN rows for planes not recorded; a flat range gets
    hi = lo + 1. ValueError for an unknown contrast, percentiles not 0 <= p_lo < p_hi <= 100, a block that is not
    (n_planes, H, W) with n_planes = len(valid), or no recorded plane."""
    b = np.asarray(block)
    v = np.asarray(valid, dtype=bool).reshape(-1)
    if contrast not in CONTRASTS:
        raise ValueError("contrast must be one of %s, not %r" % (CONTRASTS, contrast))
    p_lo, p_hi = (float(x) for x in percentiles)
    if not (0.0 <= p_lo < p_hi <= 100.0):
        raise ValueError("percentiles must satisfy 0 <= p_lo < p_hi <= 100")
    if b.ndim != 3 or b.shape[0] != len(v):
        raise ValueError("block must be (n_planes, H, W) with n_planes = len(valid)")
    if not np.any(v):
        raise ValueError("no recorded plane")
    out = np.full((len(v), 2), np.nan)
    if contrast == "fixed":
        lo, hi = np.percentile(b[v].astype(float), (p_lo, p_hi))
        out[v] = (lo, hi if hi > lo else lo + 1.0)
    else:
        for i in np.flatnonzero(v):
            lo, hi = np.percentile(b[i].astype(float), (p_lo, p_hi))
            out[i] = (lo, hi if hi > lo else lo + 1.0)
    return out


def plane_png(image, lo, hi, zoom=1) -> bytes:
    """PNG bytes (8-bit grey) of one plane: grey level g shown as rint(255 (g - lo) / (hi - lo)) clipped to 0..255,
    each pixel repeated zoom x zoom times (whole pixels, no interpolation). ValueError for an image that is not 2-D,
    lo or hi not finite, hi <= lo, or zoom not an integer >= 1."""
    from PIL import Image
    a = np.asarray(image, dtype=float)
    if a.ndim != 2:
        raise ValueError("plane_png needs a 2-D image")
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError("plane_png needs finite lo < hi")
    if isinstance(zoom, bool) or int(zoom) != zoom or int(zoom) < 1:
        raise ValueError("zoom must be an integer >= 1")
    u = np.clip(np.rint(255.0 * (a - lo) / (hi - lo)), 0, 255).astype(np.uint8)
    z = int(zoom)
    if z > 1:
        u = np.repeat(np.repeat(u, z, axis=0), z, axis=1)
    buf = io.BytesIO()
    Image.fromarray(u, mode="L").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


class PlaneViewer:
    """The planes of a crop one at a time: a slider over the planes (and a play button), the plane's depth, the
    contrast ("fixed" or "per plane"), the image with nothing drawn over it, and a button handing the shown plane to
    on_pick. Display it with IPython.display.display(viewer.widget).

    block (n_planes, H, W) grey levels; ks (n_planes,) increasing plane indices; valid (n_planes,) bool; left, top
    (px) the crop's corner; res0_um (um/px); dz_um (um per plane). centre_xy_um and view_half_um: the square view
    about the centre (None: the whole crop). k_ref: a plane the depths are given from (the traced depth's), shown in
    the info line. k_start: the first plane shown when it is a recorded plane of the slider (else k_ref, else the
    middle recorded plane; a k_start that is not used is said in the status line). on_pick(k) is called with the
    shown plane when the button is pressed; a str it returns is shown in the status line, an exception's message too.
    ValueError for arrays that disagree, ks not increasing, no recorded plane, an unknown contrast or a bad zoom."""

    def __init__(self, block, ks, valid, *, left, top, res0_um, dz_um, centre_xy_um=None, view_half_um=None,
                 k_ref=None, title="", zoom=2, contrast="fixed", k_start=None,
                 on_pick: Optional[Callable[[int], object]] = None, pick_label="pick this plane",
                 percentiles: Sequence[float] = (0.5, 99.5), play_interval_ms=200):
        import ipywidgets as W
        b = np.asarray(block)
        self.ks = np.asarray(ks).astype(int).reshape(-1)
        self.valid = np.asarray(valid, dtype=bool).reshape(-1)
        if b.ndim != 3 or b.shape[0] != len(self.ks) or len(self.valid) != len(self.ks):
            raise ValueError("block must be (n_planes, H, W) with n_planes = len(ks) = len(valid)")
        if len(self.ks) > 1 and np.any(np.diff(self.ks) <= 0):
            raise ValueError("ks must be increasing")
        if not np.any(self.valid):
            raise ValueError("no recorded plane")
        if contrast not in CONTRASTS:
            raise ValueError("contrast must be one of %s, not %r" % (CONTRASTS, contrast))
        if isinstance(zoom, bool) or int(zoom) != zoom or int(zoom) < 1:
            raise ValueError("zoom must be an integer >= 1")
        rows, cols = view_slices(b.shape[1:], left, top, res0_um, centre_xy_um, view_half_um)
        self.view = b[:, rows, cols]
        self.zoom, self.dz_um, self.res0_um = int(zoom), float(dz_um), float(res0_um)
        self.percentiles = tuple(float(x) for x in percentiles)
        self.k_ref = None if k_ref is None else int(k_ref)
        self.on_pick = on_pick
        self._index = {int(k): i for i, k in enumerate(self.ks)}
        self._ranges = {}
        self._png = {}
        rec = self.ks[self.valid]
        note = ""
        if k_start is not None and self._recorded(int(k_start)):
            k0 = int(k_start)
        else:
            if k_start is not None:
                note = "plane %s is not a recorded plane of this window (%d..%d): not shown first" % (
                    k_start, rec[0], rec[-1])
            k0 = self.k_ref if self.k_ref is not None and self._recorded(self.k_ref) else int(rec[len(rec) // 2])
        h, w = self.view.shape[1:]
        head = ("%splanes %d..%d (%d recorded); view %d x %d px (%.1f x %.1f &micro;m), 1 px = %.4f &micro;m, "
                "shown x%d; nothing drawn over the planes" % (str(title) + "<br>" if title else "", rec[0], rec[-1],
                                                               len(rec), w, h, (w - 1) * self.res0_um,
                                                               (h - 1) * self.res0_um, self.res0_um, self.zoom))
        self.title = W.HTML(value=head)
        self.slider = W.IntSlider(value=k0, min=int(rec[0]), max=int(rec[-1]), step=1, description="plane k",
                                  continuous_update=True, layout=W.Layout(width="70%"))
        self.play = W.Play(value=k0, min=int(rec[0]), max=int(rec[-1]), step=1, interval=int(play_interval_ms))
        W.jslink((self.play, "value"), (self.slider, "value"))
        self.contrast = W.ToggleButtons(options=list(CONTRASTS), value=contrast, description="contrast")
        self.info = W.HTML()
        self.status = W.HTML(value=note)
        self.image = W.Image(format="png")
        self.pick = W.Button(description=str(pick_label), button_style="primary", layout=W.Layout(width="auto"))
        self.slider.observe(lambda change: self._show(), names="value")
        self.contrast.observe(lambda change: self._show(), names="value")
        self.pick.on_click(lambda button: self._picked())
        controls = [self.contrast] + ([self.pick] if on_pick is not None else [])
        self.widget = W.VBox([self.title, W.HBox([self.play, self.slider]), self.info, W.HBox(controls), self.status,
                              self.image])
        self._show()

    def _recorded(self, k):
        i = self._index.get(int(k))
        return i is not None and bool(self.valid[i])

    @property
    def k(self) -> int:
        """The plane shown."""
        return int(self.slider.value)

    def png(self, k, contrast) -> bytes:
        """The PNG shown for plane k under the contrast (cached); a uniform grey image for a plane not recorded."""
        key = (int(k), str(contrast))
        if key not in self._png:
            if not self._recorded(k):
                h, w = self.view.shape[1:]
                img, lo, hi = np.full((h, w), float(PLACEHOLDER_GREY)), 0.0, 255.0
            else:
                if contrast not in self._ranges:
                    self._ranges[contrast] = grey_ranges(self.view, self.valid, contrast, self.percentiles)
                i = self._index[int(k)]
                img = self.view[i]
                lo, hi = self._ranges[contrast][i]
            self._png[key] = plane_png(img, lo, hi, self.zoom)
        return self._png[key]

    def _show(self):
        k = self.k
        z = k * self.dz_um
        ref = "" if self.k_ref is None else ("; z - z(k %d) = %+.2f &micro;m" % (self.k_ref,
                                                                                (k - self.k_ref) * self.dz_um))
        rec = "" if self._recorded(k) else " -- <b>not recorded</b>"
        self.info.value = "plane k = <b>%d</b>; z = %.2f &micro;m%s%s" % (k, z, ref, rec)
        self.image.value = self.png(k, self.contrast.value)

    def _picked(self):
        k = self.k
        if not self._recorded(k):
            self.status.value = "plane %d was not recorded: not picked" % k
            return
        try:
            msg = self.on_pick(k)
        except Exception as exc:        # shown, never swallowed: a widget callback's error is otherwise invisible
            self.status.value = "<b>on_pick failed</b> for plane %d: %s: %s" % (k, type(exc).__name__, exc)
            return
        self.status.value = msg if isinstance(msg, str) else "picked plane %d" % k
