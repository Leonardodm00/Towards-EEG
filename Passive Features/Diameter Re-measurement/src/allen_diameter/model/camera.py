"""Camera chain of the synthetic stacks -- Block 4 in specs/SPEC.md
(procedure s.3.6 step 6).

In this order, after the renderer's transmittance tau = I / B has been
pixel-integrated (render.pixel_integrate):

    signal = black_level_gl + gain * background_B_gl * tau_px      grey mapping
    signal += N(0, noise_sd_gl^2)                                    noise
    q = clip(rint(signal), 0, 2^bit_depth - 1)                       quantisation
    q = JPEG decode(JPEG encode(q))                                  if jpeg

The background grey level of the image (tau = 1, before noise) is therefore
black_level_gl + gain * background_B_gl. Black level and gain are NOT
VERIFIED for Allen's chain (impl-handoff, Known gaps); the injected noise is
meant to be matched to the real background SD measured AFTER the chain.
No bilinear interpolation here: the per-node chain applies it when it reads
a profile (handoff Eq. 9).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import io

import numpy as np


def grey_mapping(tau_px, rcfg):
    """black_level_gl + gain * background_B_gl * tau_px (grey levels, float64)."""
    return rcfg.black_level_gl + rcfg.gain * rcfg.background_B_gl * np.asarray(tau_px, dtype=float)


def quantize(signal, bit_depth):
    """Round to the nearest integer (half to even) and clip to [0, 2^bit_depth - 1];
    uint8 for 8 bit, uint16 for 16 bit."""
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16, got %r" % (bit_depth,))
    s = np.asarray(signal, dtype=float)
    if not np.all(np.isfinite(s)):
        raise ValueError("non-finite signal")
    top = float(2 ** bit_depth - 1)
    return np.clip(np.rint(s), 0.0, top).astype(np.uint8 if bit_depth == 8 else np.uint16)


def jpeg_roundtrip(img8, quality=None, qtables=None):
    """Encode a 2-D uint8 image as greyscale JPEG with Pillow and decode it.

    qtables: quantization tables as Pillow takes them (a list of 64-integer
    lists, or the dict Image.open(f).quantization returns); when given,
    quality is ignored. Returns uint8 of the same shape.
    """
    from PIL import Image   # Pillow is needed only when JPEG is on

    a = np.asarray(img8)
    if a.ndim != 2 or a.dtype != np.uint8:
        raise ValueError("jpeg_roundtrip needs a 2-D uint8 array, got %s %r" % (a.dtype, a.shape))
    buf = io.BytesIO()
    image = Image.fromarray(np.ascontiguousarray(a))
    if qtables is not None:
        from ..loading.jpeg_tables import check_pillow_table_order   # a version check, no file I/O
        check_pillow_table_order()
        image.save(buf, format="JPEG", qtables=qtables)
    else:
        if quality is None or not (0 < int(quality) <= 100):
            raise ValueError("jpeg_roundtrip needs a quality in (0, 100] or qtables")
        image.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as decoded:
        return np.array(decoded.convert("L"), dtype=np.uint8)


def camera_chain(tau_px, rcfg, rng, qtables=None):
    """Grey mapping, noise, quantisation and (if rcfg.jpeg) JPEG, per plane.

    tau_px: (n_planes, H, W) or (H, W) pixel-integrated transmittance.
    rng: numpy.random.Generator for the noise (used when noise_sd_gl > 0).
    qtables: Allen's quantization tables (loading.jpeg_tables); None means
    rcfg.jpeg_qtables when that is non-empty, else rcfg.jpeg_quality
    [corrected 2026-10-07: None always meant jpeg_quality, and no table build
    passed tables]. Returns uint8 (uint16 when bit_depth = 16 and no JPEG) of
    the shape of tau_px.
    """
    if qtables is None and rcfg.jpeg_qtables:
        qtables = [list(t) for t in rcfg.jpeg_qtables]
    signal = grey_mapping(tau_px, rcfg)
    if rcfg.noise_sd_gl > 0:
        signal = signal + rng.normal(0.0, rcfg.noise_sd_gl, size=signal.shape)
    q = quantize(signal, rcfg.bit_depth)
    if rcfg.jpeg:
        planes = q.reshape((-1,) + q.shape[-2:])
        out = np.empty_like(planes)
        for k in range(planes.shape[0]):
            out[k] = jpeg_roundtrip(planes[k], quality=rcfg.jpeg_quality, qtables=qtables)
        q = out.reshape(q.shape)
    return q
