"""Part C: how pixel integration, bilinear interpolation and JPEG widen a thin line.

A straight dark line with a Gaussian cross-profile (sigma0) is rendered on the
Allen pixel grid (0.1144 um) with pixel-area integration (16x16 supersampling),
quantised to 8 bit, optionally JPEG-compressed (PIL, greyscale), then read along
a perpendicular line at 0.1144 um steps with bilinear interpolation
(scipy.ndimage.map_coordinates, order=1), as in handoff Eq. 9. A Gaussian dip is
fitted to each profile; sigma_eff^2 - sigma0^2 is the added variance.
Random line angle and random sub-pixel offsets; 400 profiles per condition.
"""
import io
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit

PX = 0.1144
SIG0 = 0.080       # in-focus LSF sigma at 550 nm, NA 1.4 (Part A, Debye)
BG, DEPTH = 210.0, 70.0   # grey levels: background, dip depth (illustrative)
N_PIX = 64
SUPER = 16
rng = np.random.default_rng(1)


def render(theta, off_um):
    """Pixel-integrated image of a line through the image centre + offset."""
    n = N_PIX * SUPER
    c = (np.arange(n) + 0.5) / SUPER * PX          # sub-sample centres (um)
    X, Y = np.meshgrid(c, c, indexing="xy")
    x0 = y0 = N_PIX * PX / 2
    nx, ny = -np.sin(theta), np.cos(theta)         # unit normal to the line
    dist = (X - x0) * nx + (Y - y0) * ny - off_um
    img = BG - DEPTH * np.exp(-dist ** 2 / (2 * SIG0 ** 2))
    img = img.reshape(N_PIX, SUPER, N_PIX, SUPER).mean(axis=(1, 3))  # pixel area
    return img, (x0, y0, nx, ny)


def to8(img):
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def jpeg(img8, q):
    buf = io.BytesIO()
    Image.fromarray(img8, mode="L").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf), dtype=float)


def profile(img, geom, off_um, start_frac):
    x0, y0, nx, ny = geom
    v = (np.arange(-26, 27) + start_frac) * PX     # ~ +-3 um, sub-pixel start
    xs = x0 + (v + off_um) * nx
    ys = y0 + (v + off_um) * ny
    # pixel centre of index i sits at (i + 0.5) * PX
    cols = xs / PX - 0.5
    rows = ys / PX - 0.5
    return v, map_coordinates(img, [rows, cols], order=1, mode="nearest")


def dip(v, a, v0, s, b):
    return b - a * np.exp(-(v - v0) ** 2 / (2 * s ** 2))


def run(q=None, noise=0.0, n=400):
    s2 = []
    for _ in range(n):
        th = rng.uniform(0, np.pi)
        off = rng.uniform(-PX, PX)
        img, geom = render(th, off)
        if noise > 0:
            img = img + rng.normal(0, noise, img.shape)
        img8 = to8(img)
        img_r = jpeg(img8, q) if q else img8.astype(float)
        v, p = profile(img_r, geom, off, rng.uniform(0, 1))
        popt, _ = curve_fit(dip, v, p, p0=[DEPTH, 0.0, 0.1, BG], maxfev=5000)
        s2.append(popt[2] ** 2)
    s2 = np.array(s2)
    return np.mean(s2) - SIG0 ** 2, np.std(s2) / np.sqrt(n), np.sqrt(np.mean(s2))


print("px^2/12 = %.5f  px^2/6 = %.5f  px^2/4 = %.5f um^2" % (PX**2/12, PX**2/6, PX**2/4))
print("%-22s %12s %10s %14s" % ("condition", "added var", "+- se", "sigma_eff (um)"))
for label, q, noise in [("pixel+interp, 8 bit", None, 0.0),
                        ("  + JPEG q95", 95, 0.0),
                        ("  + JPEG q85", 85, 0.0),
                        ("  + JPEG q75", 75, 0.0),
                        ("  + JPEG q50", 50, 0.0),
                        ("noise 3 GL, no JPEG", None, 3.0),
                        ("noise 3 GL, JPEG q75", 75, 3.0)]:
    a, se, s = run(q, noise)
    print("%-22s %12.5f %10.5f %14.4f" % (label, a, se, s))
