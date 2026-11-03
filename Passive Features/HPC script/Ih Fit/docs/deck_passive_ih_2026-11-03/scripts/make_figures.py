"""make_figures.py -- the nine deck figures (v2, 2026-11-03): passive + I_h optimisation.

Each figure is a 1920 x 1080 HTML/CSS slide rendered to PNG at 2x (3840 x 2160) with Playwright's
Chromium. Charts are drawn here as SVG from the numbers below (no plotting library), so every label
is set in Inter with real sub- and superscripts. Content is unchanged from v1 (the deck handoff's
section 8 lists each figure's claim and source); v2 is the visual redesign asked for on 2026-11-03.

    python make_figures.py --status <roadmap.py status --json output>            # all nine, ~20 s
    python make_figures.py --status <json> --only 4,7                           # some of them

Needs numpy, playwright (Python, with its Chromium) and the Inter font (Pillow, if present, recompresses the PNGs
losslessly); reads human_ih_params.py and noise_law.py from the Ih Fit folder two levels up. Standalone, not
pipeline code. ASCII, LF.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IH_FIT = HERE.parents[2]
OUT = HERE.parent / "figures"
sys.path.insert(0, str(IH_FIT))
import human_ih_params as hip                        # noqa: E402
import noise_law as nl                               # noqa: E402

# ----------------------------------------------------------------------------- tokens
INK, INK2, INK3, MUTE = "#0F172A", "#334155", "#64748B", "#94A3B8"
GRID, AXIS = "#E9EDF2", "#CBD5E1"
# categorical slots 1-3 of the dataviz reference palette, in their fixed order; grey = de-emphasis
BLUE, ORANGE, AQUA, GREY = "#2A78D6", "#EB6834", "#1BAF7A", "#C9D2DE"
BLUE_T, ORANGE_T, AQUA_T = "#E8F1FC", "#FDEEE6", "#E1F5EE"
STATUS = {   # pill background, text, dot (the dots are the roadmap tool's own colours)
    "done": ("#E6F4EA", "#1E7B34", "#2D9A3D"), "closed": ("#E6F4EA", "#1E7B34", "#2D9A3D"),
    "tested": ("#E0F2F0", "#17695F", "#2A9D8F"), "pushed": ("#FBF1D5", "#8A6510", "#E0B33A"),
    "planned": ("#FDE7E7", "#B42318", "#D62828"), "studying": ("#E5EEFF", "#2457C5", "#3A86FF"),
    "open": ("#EEF2F6", "#556070", "#8D99AE"), "local": ("#FDEBD8", "#9A4D00", "#F77F00"),
    "failing": ("#FBE4EC", "#9B1442", "#B5174E"), "running": ("#EFE6FD", "#5B21B6", "#8338EC"),
    "testing": ("#E5EEFF", "#2457C5", "#3A86FF"),
}
IMPL_ORDER = ["planned", "local", "failing", "pushed", "testing", "tested", "running", "done"]
THETA_CF = nl.NoiseLaw(0.0380, 0.175, 0.0508, 0.0366)    # (mV, -, mV, s): the cohort closed-form law
V_REST = -73.5                                            # mV, LJP-corrected (D-005)
W, H = 1920, 1080
TICK = "font-size:13.5px;color:%s;font-variant-numeric:tabular-nums" % INK3
AXL = "font-size:15px;color:%s;font-weight:600" % INK2

HAT_SVG = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 5'>"
           "<path d='M0.6,4.4 L5,0.8 L9.4,4.4' fill='none' stroke='%230F172A' stroke-width='1.1'/></svg>")
HAT_SVG_W = HAT_SVG.replace("%230F172A", "%23FFFFFF")
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { width: 1920px; height: 1080px; font-family: Inter, sans-serif; color: %(INK)s;
       -webkit-font-smoothing: antialiased; }
.page { position: relative; width: 1920px; height: 1080px; overflow: hidden;
        background: radial-gradient(1100px 520px at 92%% -8%%, #E3ECFB 0%%, rgba(227,236,251,0) 62%%),
                    radial-gradient(900px 500px at -5%% 105%%, #FBEADF 0%%, rgba(251,234,223,0) 60%%),
                    linear-gradient(180deg, #FBFAF7 0%%, #F3F2EE 100%%); }
.head { position: absolute; left: 72px; top: 50px; right: 72px; }
.kicker { font-size: 15px; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; color: %(BLUE)s; }
h1 { font-family: 'Inter Display', Inter, sans-serif; font-size: 48px; font-weight: 800; letter-spacing: -.02em;
     line-height: 1.08; margin-top: 10px; color: %(INK)s; }
.sub { font-size: 21px; color: #475569; margin-top: 12px; line-height: 1.4; max-width: 1700px; }
.card { position: absolute; background: #FFFFFF; border-radius: 20px; border: 1px solid #ECEAE4;
        box-shadow: 0 1px 2px rgba(15,23,42,.05), 0 12px 32px rgba(15,23,42,.07); }
.card.pad { padding: 22px 26px; }
.label { font-size: 13px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: %(INK3)s; }
.cap { font-size: 16px; font-weight: 700; color: %(INK2)s; }
.ctitle { font-size: 22px; font-weight: 700; color: %(INK)s; letter-spacing: -.01em; line-height: 1.2; }
.ctext { font-size: 16.5px; color: %(INK2)s; line-height: 1.45; }
.small { font-size: 14.5px; color: %(INK3)s; line-height: 1.4; }
.muted { color: %(INK3)s; }
.pill { display: inline-flex; align-items: center; gap: 8px; padding: 5px 13px 5px 11px; border-radius: 999px;
        font-size: 14px; font-weight: 700; white-space: nowrap; }
.pill i { width: 9px; height: 9px; border-radius: 50%%; display: inline-block; }
.foot { position: absolute; left: 72px; right: 72px; bottom: 24px; font-size: 13.5px; color: #94A3B8; line-height: 1.4; }
.chip { display: inline-block; padding: 4px 10px; border-radius: 8px; background: #F1F5F9; color: %(INK2)s;
        font-size: 14px; font-weight: 600; white-space: nowrap; }
.legend { display: flex; gap: 26px; align-items: center; font-size: 15.5px; color: %(INK2)s; font-weight: 600; }
.legend > span { display: inline-flex; align-items: center; gap: 9px; white-space: nowrap; }
.ss { display: inline-flex; flex-direction: column; vertical-align: 0.42em; font-size: 64%%; line-height: 1.0;
      margin-left: 1px; text-align: left; }
.wrap { white-space: normal; }
.sw { display: inline-block; width: 24px; height: 4px; border-radius: 2px; }
.swa { display: inline-block; width: 22px; height: 14px; border-radius: 4px; }
.hat { position: relative; display: inline-block; }
.hat::after { content: ""; position: absolute; left: 18%%; right: 12%%; top: -.06em; height: .32em;
              background: no-repeat center/100%% 100%% url("%(HAT)s"); }
.hat.hatw::after { background-image: url("%(HATW)s"); }
.row { display: flex; justify-content: space-between; align-items: center; gap: 14px; }
sub, sup { font-size: 70%%; line-height: 0; }
svg.layer { position: absolute; left: 0; top: 0; width: 1920px; height: 1080px; overflow: visible;
            pointer-events: none; }
""" % dict(INK=INK, INK2=INK2, INK3=INK3, BLUE=BLUE, HAT=HAT_SVG, HATW=HAT_SVG_W)

DEFS = ('<svg width="0" height="0" style="position:absolute"><defs>'
        + "".join('<marker id="%s" viewBox="0 0 10 10" refX="8.6" refY="5" markerWidth="13" markerHeight="13" '
                  'markerUnits="userSpaceOnUse" orient="auto"><path d="M0.5,0.8 L9.5,5 L0.5,9.2 z" fill="%s"/></marker>'
                  % (k, c) for k, c in (("ah", MUTE), ("ahb", BLUE), ("aho", ORANGE), ("ahk", INK2)))
        + '</defs></svg>')


# ----------------------------------------------------------------------------- page helpers
def pill(status, text=None):
    bg, fg, dot = STATUS.get(status, ("#EEE", "#333", "#999"))
    return ('<span class="pill" style="background:%s;color:%s"><i style="background:%s"></i>%s</span>'
            % (bg, fg, dot, text or status))


def card(x, y, w, h, inner="", extra=""):
    return ('<div class="card pad" style="left:%dpx;top:%dpx;width:%dpx;height:%dpx;%s">%s</div>'
            % (x, y, w, h, extra, inner))


def at(x, y, html, anchor="tl", style=""):
    """An absolutely positioned HTML snippet; anchor = which of its points sits at (x, y)."""
    tr = {"tl": "none", "c": "translate(-50%,-50%)", "l": "translate(0,-50%)", "r": "translate(-100%,-50%)",
          "b": "translate(-50%,-100%)", "t": "translate(-50%,0)", "bl": "translate(0,-100%)",
          "br": "translate(-100%,-100%)", "tr": "translate(-100%,0)"}[anchor]
    return ('<div style="position:absolute;left:%.1fpx;top:%.1fpx;transform:%s;white-space:nowrap;%s">%s</div>'
            % (x, y, tr, style, html))


def layer(*elements):
    return '<svg class="layer" viewBox="0 0 1920 1080">%s</svg>' % "".join(elements)


def path(d, color=MUTE, width=2.5, dash=None, marker="ah", opacity=1.0):
    da = ' stroke-dasharray="%s"' % dash if dash else ""
    mk = ' marker-end="url(#%s)"' % marker if marker else ""
    return ('<path d="%s" fill="none" stroke="%s" stroke-width="%s" stroke-linecap="round" stroke-linejoin="round" '
            'opacity="%s"%s%s/>' % (d, color, width, opacity, da, mk))


def hcurve(x1, y1, x2, y2, **kw):
    mx = (x1 + x2) / 2.0
    return path("M%.1f,%.1f C%.1f,%.1f %.1f,%.1f %.1f,%.1f" % (x1, y1, mx, y1, mx, y2, x2, y2), **kw)


def vcurve(x1, y1, x2, y2, **kw):
    my = (y1 + y2) / 2.0
    return path("M%.1f,%.1f C%.1f,%.1f %.1f,%.1f %.1f,%.1f" % (x1, y1, x1, my, x2, my, x2, y2), **kw)


def page(kicker, title, sub, body, foot=""):
    return ("<!doctype html><html><head><meta charset='utf-8'><style>%s</style></head><body><div class='page'>%s"
            "<div class='head'><div class='kicker'>%s</div><h1>%s</h1><div class='sub'>%s</div></div>%s"
            "<div class='foot'>%s</div></div></body></html>" % (CSS, DEFS, kicker, title, sub, body, foot))


def supsub(sup, sub):
    """A superscript stacked over a subscript, e.g. e^fast_j."""
    return '<span class="ss"><span>%s</span><span>%s</span></span>' % (sup, sub)


def flow(*steps):
    """Chips joined by arrows, for a short sequence of steps."""
    arrow = '<span style="color:%s;font-size:18px;font-weight:600">&#8594;</span>' % MUTE
    return arrow.join('<span class="chip" style="font-size:15.5px;padding:7px 12px">%s</span>' % s for s in steps)


def num(v, fmt="%g"):
    return (fmt % v).replace("-", "&#8722;")


def legend(items, x, y, gap=26):
    """items: (kind, colour, html) with kind 'line', 'area' or 'dot'; the text stays in ink."""
    out = []
    for kind, colr, html in items:
        if kind == "line":
            sw = '<i class="sw" style="background:%s"></i>' % colr
        elif kind == "area":
            sw = '<i class="swa" style="background:%s;opacity:.28"></i>' % colr
        else:
            sw = '<i style="width:11px;height:11px;border-radius:50%%;background:%s;display:inline-block"></i>' % colr
        out.append("<span>%s<span>%s</span></span>" % (sw, html))
    return at(x, y, '<div class="legend" style="gap:%dpx">%s</div>' % (gap, "".join(out)))


# ----------------------------------------------------------------------------- SVG charts
class Chart:
    """A plot area at page coordinates (x, y, w, h): marks in one SVG layer, all text in HTML."""
    _n = 0

    def __init__(self, x, y, w, h, xlim, ylim, xlog=False):
        Chart._n += 1
        self.cid = "clip%d" % Chart._n
        self.x, self.y, self.w, self.h = x, y, w, h
        self.xlim, self.ylim, self.xlog = xlim, ylim, xlog
        self.under, self.clipped, self.over, self.text = [], [], [], []

    def X(self, v):
        if self.xlog:
            a, b = math.log10(self.xlim[0]), math.log10(self.xlim[1])
            return self.x + (math.log10(v) - a) / (b - a) * self.w
        return self.x + (v - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w

    def Y(self, v):
        return self.y + self.h - (v - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

    def ygrid(self, ticks, fmt="%g"):
        for t in ticks:
            yy = self.Y(t)
            self.under.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>'
                              % (self.x, yy, self.x + self.w, yy, AXIS if t == self.ylim[0] else GRID))
            self.text.append(at(self.x - 12, yy, num(t, fmt) if isinstance(fmt, str) else fmt(t), "r", TICK))

    def xgrid(self, ticks):
        for t in ticks:
            xx = self.X(t)
            self.under.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>'
                              % (xx, self.y, xx, self.y + self.h, GRID))

    def xticks(self, ticks, fmt="%g", off=22):
        for t in ticks:
            self.text.append(at(self.X(t), self.y + self.h + off, num(t, fmt) if isinstance(fmt, str) else fmt(t),
                                "c", TICK))

    def xlabel(self, html, off=52):
        self.text.append(at(self.x + self.w / 2.0, self.y + self.h + off, html, "c", AXL))

    def ylabel(self, html, off=66):
        self.text.append('<div style="position:absolute;left:%.1fpx;top:%.1fpx;transform:translate(-50%%,-50%%) '
                         'rotate(-90deg);white-space:nowrap;%s">%s</div>' % (self.x - off, self.y + self.h / 2.0,
                                                                            AXL, html))

    def _pts(self, xs, ys):
        return " ".join("%.1f,%.1f" % (self.X(a), self.Y(b)) for a, b in zip(xs, ys))

    def line(self, xs, ys, color, width=2.5):
        self.clipped.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="%s" stroke-linejoin="round" '
                            'stroke-linecap="round"/>' % (self._pts(xs, ys), color, width))

    def area(self, xs, y0, y1, color, opacity=0.10):
        pts = self._pts(xs, y1) + " " + self._pts(xs[::-1], y0[::-1])
        self.clipped.append('<polygon points="%s" fill="%s" opacity="%s" stroke="none"/>' % (pts, color, opacity))

    def vline(self, v, color=MUTE, width=1.25):
        xx = self.X(v)
        self.under.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%s"/>'
                          % (xx, self.y, xx, self.y + self.h, color, width))

    def seg(self, x0, y0, x1, y1, color=INK, width=2.0):     # data coordinates
        self.over.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%s" '
                         'stroke-linecap="round"/>' % (self.X(x0), self.Y(y0), self.X(x1), self.Y(y1), color, width))

    def dot(self, xv, yv, color, r=5.5):
        self.over.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" stroke="#FFFFFF" stroke-width="2"/>'
                         % (self.X(xv), self.Y(yv), r, color))

    def bar(self, xv, v, width, color, r=4.0):
        """A column from the baseline: square at the baseline, 4 px rounded data-end."""
        x0, x1 = self.X(xv) - width / 2.0, self.X(xv) + width / 2.0
        yb, yt = self.Y(self.ylim[0]), self.Y(v)
        r = min(r, (yb - yt) / 2.0)
        self.over.append('<path d="M%.1f,%.1f L%.1f,%.1f Q%.1f,%.1f %.1f,%.1f L%.1f,%.1f Q%.1f,%.1f %.1f,%.1f '
                         'L%.1f,%.1f Z" fill="%s"/>' % (x0, yb, x0, yt + r, x0, yt, x0 + r, yt, x1 - r, yt, x1, yt,
                                                        x1, yt + r, x1, yb, color))

    def leader(self, p0, p1, color=MUTE):                    # page coordinates
        self.over.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.25"/>'
                         % (p0[0], p0[1], p1[0], p1[1], color))

    def note(self, xv, yv, html, anchor="l", style=""):
        self.text.append(at(self.X(xv), self.Y(yv), html, anchor, "font-size:14.5px;color:%s;%s" % (INK2, style)))

    def html(self):
        clip = ('<defs><clipPath id="%s"><rect x="%.1f" y="%.1f" width="%.1f" height="%.1f"/></clipPath></defs>'
                % (self.cid, self.x, self.y - 2, self.w, self.h + 4))
        return (layer(clip, "".join(self.under), '<g clip-path="url(#%s)">%s</g>' % (self.cid, "".join(self.clipped)),
                      "".join(self.over)) + "".join(self.text))


# ----------------------------------------------------------------------------- pictograms (schematic)
def _poly(xs, ys, col, width=2.2):
    return ('<polyline points="%s" fill="none" stroke="%s" stroke-width="%s" stroke-linejoin="round" '
            'stroke-linecap="round"/>' % (" ".join("%.1f,%.1f" % p for p in zip(xs, ys)), col, width))


def mini_ss(w=150, h=44, col=BLUE):
    """A brief hyperpolarising pulse response: fast fall, slow recovery."""
    xs = np.linspace(0, 1, 70)
    ys = np.where(xs < 0.12, 0.0, np.exp(-(xs - 0.12) / 0.20) * (1 - np.exp(-(xs - 0.12) / 0.015)))
    ys = ys / ys.max()
    return '<svg width="%d" height="%d">%s</svg>' % (w, h, _poly(4 + xs * (w - 8), 6 + ys * (h - 12), col))


def mini_ls(w=150, h=44, col=ORANGE):
    """A hyperpolarising step with sag and rebound."""
    xs = np.linspace(0, 1, 90)
    on, off = 0.15, 0.82
    ys = np.zeros_like(xs)
    m = (xs >= on) & (xs < off)
    t = xs[m] - on
    ys[m] = -(1 - np.exp(-t / 0.04)) * (0.72 + 0.28 * np.exp(-t / 0.16))
    r = xs >= off
    tr = xs[r] - off
    ys[r] = ys[m][-1] * np.exp(-tr / 0.04) + 0.10 * (1 - np.exp(-tr / 0.03)) * np.exp(-tr / 0.12)
    return '<svg width="%d" height="%d">%s</svg>' % (w, h, _poly(4 + xs * (w - 8), 8 - ys * (h - 18), col))


def picto_ss_train(w, h, rng):
    """SS sweep: alternating pulse responses on noise, the analysed window after each pulse shaded."""
    n_p, per = 5, 1.0 / 5
    xs = np.linspace(0, 1, 900)
    ys = 0.05 * np.convolve(rng.standard_normal(xs.size), np.ones(9) / 3.0, mode="same")
    shades = []
    for k in range(n_p):
        t0 = k * per + 0.01
        s = 1 if k % 2 == 0 else -1
        tt = xs - t0
        ys += np.where(tt > 0, s * 0.9 * np.exp(-tt / 0.03) * (1 - np.exp(-tt / 0.004)), 0.0)
        shades.append((t0 + 0.40 * per, t0 + per - 0.012))
    rects = "".join('<rect x="%.1f" y="0" width="%.1f" height="%d" rx="4" fill="%s"/>'
                    % (a * w, (b - a) * w, h, BLUE_T) for a, b in shades)
    return '<svg width="%d" height="%d">%s%s</svg>' % (w, h, rects, _poly(xs * w, h / 2.0 - ys * h * 0.42, INK3, 1.4))


def picto_ls_sweep(w, h, rng):
    """LS sweep (time not to scale): pre-window, the 1 s step, the long tail and its end window."""
    xs = np.linspace(0, 1, 900)
    on, off = 0.30, 0.55
    base = np.zeros_like(xs)
    m = (xs >= on) & (xs < off)
    t = xs[m] - on
    base[m] = -(1 - np.exp(-t / 0.02)) * (0.80 + 0.20 * np.exp(-t / 0.08))
    r = xs >= off
    base[r] = base[m][-1] * np.exp(-(xs[r] - off) / 0.02)
    ys = base + 0.05 * np.convolve(rng.standard_normal(xs.size), np.ones(9) / 3.0, mode="same")
    pre, end = (0.02, on - 0.025), (0.80, 0.995)
    rects = ('<rect x="%.1f" y="0" width="%.1f" height="%d" rx="4" fill="%s"/>'
             '<rect x="%.1f" y="0" width="%.1f" height="%d" rx="4" fill="%s"/>'
             % (pre[0] * w, (pre[1] - pre[0]) * w, h, BLUE_T, end[0] * w, (end[1] - end[0]) * w, h, AQUA_T))
    brk = ('<path d="M%.1f,%.1f l6,-10 M%.1f,%.1f l6,-10" stroke="%s" stroke-width="1.6"/>'
           % (0.68 * w, h * 0.40, 0.68 * w + 6, h * 0.40, INK3))
    return ('<svg width="%d" height="%d">%s%s%s</svg>'
            % (w, h, rects, _poly(xs * w, h * 0.30 - ys * h * 0.62, INK3, 1.4), brk))


def picto_noise(w, h, trace):
    xs = np.linspace(0, 1, trace.size)
    yy = trace / max(1e-12, np.abs(trace).max())
    shades = "".join('<rect x="%.1f" y="0" width="%.1f" height="%d" rx="4" fill="%s"/>'
                     % ((k * 0.2 + 0.09) * w, 0.10 * w, h, BLUE_T) for k in range(5))
    return '<svg width="%d" height="%d">%s%s</svg>' % (w, h, shades, _poly(xs * w, h / 2.0 - yy * h * 0.42, ORANGE, 1.6))


def load_status(path_json):
    d = json.loads(Path(path_json).read_text())
    impl, theo = d["implementation"], d["theory"]
    return d, {s["id"]: s for s in impl}, impl, theo


# ============================================================================= fig01
def fig01(d, imp):
    st = lambda i: imp[i]["status"]  # noqa: E731
    noise14 = min((st(i) for i in ("N1", "N2", "N3", "N4")), key=IMPL_ORDER.index)
    col = '<div class="label" style="position:absolute;left:%dpx;top:%dpx">%s</div>'
    b = [col % (72, 238, "1 &middot; Data"), col % (452, 238, "2 &middot; Passive fit"),
         col % (922, 238, "3 &middot; Six-parameter fit and its noise"), col % (1422, 238, "4 &middot; Validate, then use")]
    b.append(card(72, 270, 320, 330,
                  '<div class="ctitle">Allen human cells</div><div class="small" style="margin-top:4px">'
                  'Phase-0 archive, written on Colab</div>'
                  '<div style="margin-top:14px">%s</div><div class="ctext" style="margin-top:-2px">Square Subthreshold '
                  'pulse trains</div><div style="margin-top:8px">%s</div><div class="ctext" style="margin-top:-2px">'
                  'Long Square steps, 1 s</div><div class="ctext" style="margin-top:10px">+ repaired 3-D '
                  'reconstruction</div><div style="margin-top:14px">%s</div>' % (mini_ss(), mini_ls(), pill(st("I0")))))
    rows = [("Phase 1", "NEURON model of the reconstructed cell; spines folded in by F", "I1"),
            ("Phase 2", "Gaussian-process fit + absolute-mV gate &middot; run B, 68 cells: <b>the estimate of "
                        "record</b>", "I2"),
            ("Phase 3", "bootstrap intervals (B = 200) + likelihood profile", "I3")]
    y = 270
    for t, s, i in rows:
        b.append(card(452, y, 410, 128, '<div class="row"><div class="ctitle">%s</div>%s</div>'
                                        '<div class="ctext" style="margin-top:8px">%s</div>' % (t, pill(st(i)), s)))
        y += 146
    b.append(at(460, 712, 'Synthetic benchmark validated the loss (%s).<br/>Phase 2.5 (R<sub>a</sub> fixed) was tried and '
                          '<b>rejected</b>.' % st("I4"), style="font-size:15px;color:%s;line-height:1.45" % INK3))
    b.append(card(922, 270, 440, 190,
                  '<div class="row"><div class="ctitle">I<sub>h</sub> fitter &middot; 6 parameters</div>%s</div>'
                  '<div class="ctext" style="margin-top:8px">Stages 0-6: regression gate, mechanisms, loader, '
                  'parameter vector, protocol, Phase 3, orchestrator.</div><div class="ctext" style="margin-top:8px;'
                  'color:#B42318;font-weight:600">No real cell fitted yet.</div>' % pill(st("G3")),
                  "border-top:5px solid %s" % ORANGE))
    nrow = '<div class="row" style="margin-top:12px"><div class="ctext">%s</div>%s</div>'
    b.append(card(922, 482, 440, 278,
                  '<div class="ctitle">Recording-noise fit</div><div class="small" style="margin-top:2px">'
                  'the law each synthetic twin must carry</div>'
                  + nrow % ("Blocks 1-4 &middot; statistics, law,<br/>windows, per-cell fit", pill(noise14))
                  + nrow % ("Block 9 &middot; per-sweep stage", pill(st("N4b"), "planned &middot; SPEC drafted"))
                  + nrow % ("Blocks 5-7 &middot; twins, driver,<br/>record", pill(st("N5"))),
                  "border-top:5px solid %s" % AQUA))
    b.append(card(1422, 270, 426, 222,
                  '<div class="row"><div class="ctitle">Stage 7 &middot; recovery gate</div>%s</div>'
                  '<div class="ctext" style="margin-top:8px">Synthetic twins with known parameters, fitted on davinci. '
                  'Built and tested; <b>never run</b>.</div><div class="ctext" style="margin-top:10px;padding:8px 12px;'
                  'background:#FDF2F2;border-radius:10px;color:#9B1C1C;font-size:15px"><b>D-083</b>: its verdict comes '
                  'before any real six-parameter fit.</div>' % pill(st("S7")), "border:2px solid #F3B4B4"))
    b.append(card(1422, 514, 426, 128,
                  '<div class="row"><div class="ctitle">Stage 8 &middot; campaign</div>%s</div><div class="ctext" '
                  'style="margin-top:8px">three arms on the real cells; also waits for diameter-corrected '
                  'arbours</div>' % pill(st("S8"))))
    b.append(card(1422, 664, 426, 96, '<div class="row"><div class="ctitle">Stage 9 &middot; record</div>%s</div>'
                                      % pill(st("S9"))))
    b.append(layer(hcurve(392, 435, 450, 334), hcurve(862, 480, 920, 365, color=BLUE, marker="ahb"),
                   hcurve(1362, 365, 1420, 380), hcurve(1362, 620, 1420, 420),
                   path("M1635,494 L1635,512"), path("M1635,644 L1635,662")))
    b.append(card(72, 800, 1776, 146,
                  '<div class="label">The storyline</div><div class="ctext" style="font-size:19px;margin-top:10px;'
                  'line-height:1.5">The passive fit works and is the estimate of record, but it returns a capacitance '
                  '<b>2-3 times</b> the published human value because it absorbs the h-current. The six-parameter fit '
                  'models I<sub>h</sub> instead &mdash; and is trusted only after a synthetic recovery gate whose twins '
                  'carry each real sweep&rsquo;s fitted noise.</div>'))
    b.append(at(72, 974, '<div style="display:flex;gap:12px;align-items:center"><span class="label" style="margin-right:6px">'
                         'Status</span>%s</div>' % "".join(pill(s) for s in ("done", "tested", "pushed", "planned"))))
    return page("Roadmap allen-mtg-fit &middot; where we stand", "From a passive fit to a validated six-parameter fit",
                "Status computed from the repository by roadmap.py at %s (%s)." % (d["commit_short"],
                                                                                d["commit_date"][:10]),
                "".join(b), "<span style='position:absolute;right:0'>Sources: specs/roadmap/allen-mtg-fit.yaml; annual "
                            "report part 1 &sect;6; decision D-083</span>")


# ============================================================================= fig02
SHORT_IMPL = {"G1": "Data archive", "G2": "Passive fit", "G3": "I_h fitter", "G4": "Noise fit",
              "G5": "Gate and campaign", "I0": "Phase-0 writer", "I1": "Loader", "I2": "Phase 2 GP fit",
              "I3": "Phase 2.5 + 3", "I4": "Synthetic benchmark", "I5": "S0 regression", "I6": "S1 mechanisms",
              "I7": "S2 loader", "I8": "S3 ParamSpec", "I9": "S4 protocol", "I10": "S5 Phase 3",
              "I11": "S6 orchestrator", "N1": "B1 autocorrelation", "N2": "B2 noise law", "NT": "Cluster gates",
              "N3": "B3 windows", "N4": "B4 per-cell fit", "N4b": "B9 per-sweep", "N5": "B5 twins' law",
              "N6": "B6 driver + PBS", "N7": "B7 record", "I12": "Stage 7 as built", "I13": "Noise table",
              "S7": "Stage 7 gate run", "S8": "Stage 8 campaign", "S9": "Stage 9 record"}
SHORT_THEO = {"TG1": "Passive fit", "TG2": "Recording noise", "TG3": "I_h fit", "TG4": "Not yet tackled",
              "T1": "Cable model", "T2": "Recordings, archive", "T3": "BO, profile, CIs", "T4": "Validation gate",
              "T5": "Noise law", "T6": "Stats vs parameters", "T7": "Indirect inference",
              "T8": "Tilt, semivariogram", "T9": "Per-sweep shrinkage", "T10": "SS layout, QC",
              "T11": "I_h kinetics", "T12": "Rest balance", "T13": "Protocol and loss",
              "T14": "Identifiability", "T15": "Temperature", "T16": "Corrected arbours",
              "T17": "Comparing the arms"}


def tile(sid, title, status, w):
    bg, fg, dot = STATUS[status]
    return ('<div style="width:%dpx;height:56px;border-radius:12px;background:%s;padding:7px 12px;'
            'border:1px solid rgba(0,0,0,.04)"><div style="display:flex;align-items:center;gap:7px">'
            '<i style="width:8px;height:8px;border-radius:50%%;background:%s;display:inline-block"></i>'
            '<span style="font-size:12.5px;font-weight:800;color:%s">%s</span><span style="font-size:12px;'
            'font-weight:700;color:%s;margin-left:auto">%s</span></div><div style="font-size:15px;'
            'font-weight:600;color:%s;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">%s</div>'
            '</div>' % (w, bg, dot, fg, sid, fg, status, INK, title.replace("I_h", "I<sub>h</sub>")))


def fig02(d, impl_list, theo_list):
    def section(stages, short, x, y, w, tile_w):
        out, yy = [], y
        for g in [s for s in stages if not s.get("parent")]:
            members = [s for s in stages if s.get("parent") == g["id"]]
            out.append(at(x, yy, '<div style="display:flex;align-items:center;gap:12px"><span class="ctitle" '
                                 'style="font-size:18px">%s</span>%s</div>'
                          % (short[g["id"]].replace("I_h", "I<sub>h</sub>"), pill(g["status"]))))
            yy += 38
            per_row = max(1, int((w + 10) // (tile_w + 10)))
            for i in range(0, len(members), per_row):
                out.append(at(x, yy, '<div style="display:flex;gap:10px">%s</div>'
                              % "".join(tile(s["id"], short[s["id"]], s["status"], tile_w)
                                        for s in members[i:i + per_row])))
                yy += 64
            yy += 10
        return "".join(out), yy

    left, y_l = section(impl_list, SHORT_IMPL, 102, 300, 1010, 194)
    right, y_r = section(theo_list, SHORT_THEO, 1238, 300, 590, 188)
    bottom = max(y_l, y_r) + 8
    ci, ct = d["counts"]["implementation"], d["counts"]["theory"]
    body = (card(72, 236, 1072, bottom - 236, '<div class="label">Implementation &middot; computed from git, SPEC.md, '
                                              'the tester reports and the cluster markers</div>')
            + card(1208, 236, 640, bottom - 236, '<div class="label">Theory &middot; set by the user</div>')
            + left + right)
    sub = ("Implementation: %d done, %d tested, %d pushed, %d planned &nbsp;&middot;&nbsp; theory: %d closed, "
           "%d studying, %d open" % (ci["done"], ci["tested"], ci["pushed"], ci["planned"], ct["closed"],
                                      ct["studying"], ct["open"]))
    return page("Roadmap allen-mtg-fit &middot; computed %s at %s" % (d["commit_date"][:10], d["commit_short"]),
                "Every stage and every topic, by state", sub, body,
                "Live page: Subproject Roadmaps (claude.ai artifact Fe6Zqjr5iq3XAQZfrTspqB). Short names; the full "
                "titles are in specs/roadmap/allen-mtg-fit.md.")


# ============================================================================= fig03
def _bez(pts, n):
    p = np.array(pts, dtype=float)
    u = np.linspace(0, 1, n)[:, None]
    if len(p) == 3:
        return (1 - u) ** 2 * p[0] + 2 * (1 - u) * u * p[1] + u ** 2 * p[2]
    return (1 - u) ** 3 * p[0] + 3 * (1 - u) ** 2 * u * p[1] + 3 * (1 - u) * u ** 2 * p[2] + u ** 3 * p[3]


def neuron(ox, oy, rng):
    """Schematic pyramidal cell in a 600 x 640 box at page (ox, oy); soma centre at box (300, 420)."""
    S, R60 = np.array([300.0, 420.0]), 100.0
    dend = "#4B5A70"
    branches = [  # Bezier control points (box coordinates), start width, end width
        ([(300, 382), (298, 300), (305, 200), (302, 112)], 8.0, 4.2),            # apical trunk
        ([(302, 112), (282, 72), (232, 34)], 3.6, 1.8), ([(302, 112), (306, 62), (292, 14)], 3.6, 1.8),
        ([(302, 112), (330, 72), (372, 30)], 3.6, 1.8), ([(302, 112), (350, 96), (424, 74)], 3.4, 1.7),
        ([(300, 332), (252, 302), (186, 290)], 3.6, 1.8), ([(301, 292), (352, 264), (424, 252)], 3.6, 1.8),
        ([(303, 214), (262, 188), (206, 152)], 3.2, 1.6), ([(304, 196), (346, 172), (398, 132)], 3.2, 1.6),
        ([(272, 446), (212, 468), (118, 470)], 5.2, 2.0), ([(328, 446), (390, 470), (482, 480)], 5.2, 2.0),
        ([(284, 456), (240, 520), (188, 592)], 5.0, 2.0), ([(316, 456), (362, 516), (424, 586)], 5.0, 2.0),
    ]
    halo, segs, spines = [], [], []
    for pts, w0, w1 in branches:
        c = _bez(pts, 60) + [ox, oy]
        halo.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="%.1f" stroke-linecap="round" '
                    'stroke-linejoin="round"/>' % (" ".join("%.1f,%.1f" % tuple(q) for q in c), ORANGE, w0 + 13))
        for k in range(len(c) - 1):
            wk = w0 + (w1 - w0) * (k + 0.5) / (len(c) - 1)
            segs.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%.2f" '
                        'stroke-linecap="round"/>' % (c[k][0], c[k][1], c[k + 1][0], c[k + 1][1], dend, wk))
        # spines every ~11 px of arc length, only beyond 60 um of the soma
        f = _bez(pts, 400)
        s_cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(f, axis=0).T))])
        side = 1
        for s0 in np.arange(6.0, s_cum[-1] - 3.0, 11.0):
            j = min(max(int(np.searchsorted(s_cum, s0)), 1), len(f) - 1)
            p = f[j]
            if np.hypot(*(p - S)) < R60:
                continue
            tvec = (f[j] - f[j - 1]) / max(1e-9, np.hypot(*(f[j] - f[j - 1])))
            nvec = np.array([-tvec[1], tvec[0]]) * side
            ang = rng.uniform(-0.45, 0.45)
            nv = np.array([nvec[0] * math.cos(ang) - nvec[1] * math.sin(ang),
                           nvec[0] * math.sin(ang) + nvec[1] * math.cos(ang)])
            ln = rng.uniform(5.5, 8.0)
            a = p + nvec * 1.5 + [ox, oy]
            b = p + nv * (1.5 + ln) + [ox, oy]
            spines.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.4"/>'
                          '<circle cx="%.1f" cy="%.1f" r="2.4" fill="%s"/>' % (a[0], a[1], b[0], b[1], BLUE, b[0], b[1],
                                                                              BLUE))
            side = -side
    sx, sy = ox + 300, oy + 420
    soma_d = ("M%d,%d C%d,%d %d,%d %d,%d C%d,%d %d,%d %d,%d C%d,%d %d,%d %d,%d C%d,%d %d,%d %d,%d Z"
              % (sx, sy - 46, sx + 9, sy - 32, sx + 30, sy - 4, sx + 33, sy + 15, sx + 36, sy + 30, sx + 20, sy + 40,
                 sx, sy + 40, sx - 20, sy + 40, sx - 36, sy + 30, sx - 33, sy + 15, sx - 30, sy - 4, sx - 9, sy - 32,
                 sx, sy - 46))
    halo.append('<path d="%s" fill="%s" stroke="%s" stroke-width="16" stroke-linejoin="round"/>'
                % (soma_d, ORANGE, ORANGE))
    soma = ('<defs><radialGradient id="sg" cx="38%%" cy="30%%" r="75%%"><stop offset="0%%" stop-color="#F8FAFC"/>'
            '<stop offset="100%%" stop-color="#CBD5E1"/></radialGradient></defs>'
            '<path d="%s" fill="url(#sg)" stroke="#475569" stroke-width="2.4"/>' % soma_d)
    axon = ('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#8A97A8" stroke-width="2.6" stroke-linecap="round"/>'
            '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#8A97A8" stroke-width="2.6" stroke-linecap="round"/>'
            % (sx, sy + 42, sx, sy + 96, sx, sy + 104, sx, sy + 158))
    ring = ('<circle cx="%d" cy="%d" r="%d" fill="none" stroke="#94A3B8" stroke-width="1.5" stroke-dasharray="5 6"/>'
            % (sx, sy, R60))
    tip, back = np.array([sx - 30.0, sy + 6.0]), np.array([ox + 14.0, oy + 336.0])   # glass pipette
    dv = (back - tip) / np.hypot(*(back - tip))
    nv = np.array([-dv[1], dv[0]])
    poly = [tip + nv * 1.6, back + nv * 12, back - nv * 12, tip - nv * 1.6]
    pip = ('<polygon points="%s" fill="#DBEAFE" stroke="#93C5FD" stroke-width="1.4" opacity=".95"/>'
           % " ".join("%.1f,%.1f" % tuple(q) for q in poly))
    halo_g = '<g opacity=".17">%s</g>' % "".join(halo)       # one opacity for the group: no build-up at overlaps
    return halo_g + ring + "".join(segs) + "".join(spines) + soma + axon + pip, (sx, sy), R60


def fig03():
    rng = np.random.default_rng(3)
    ox, oy = 122, 268
    art, (sx, sy), r60 = neuron(ox, oy, rng)
    a40 = math.radians(-38)
    labels = (at(sx + r60 * math.cos(a40) + 8, sy + r60 * math.sin(a40) - 6, "60 &#956;m", "bl",
                 "font-size:14px;font-weight:700;color:%s" % INK3)
              + at(sx, sy + 166, "axon stub, 2 &#215; 30 &#956;m", "t", "font-size:14px;color:%s" % INK3)
              + at(ox + 14, oy + 318, "somatic electrode", "bl", "font-size:14px;font-weight:600;color:%s" % INK3))
    spine_glyph = ('<svg width="26" height="16"><line x1="1" y1="12" x2="25" y2="12" stroke="#4B5A70" stroke-width="2.4" '
                   'stroke-linecap="round"/><line x1="8" y1="11" x2="8" y2="4" stroke="%s" stroke-width="1.4"/>'
                   '<circle cx="8" cy="3.5" r="2.4" fill="%s"/><line x1="18" y1="11" x2="18" y2="4" stroke="%s" '
                   'stroke-width="1.4"/><circle cx="18" cy="3.5" r="2.4" fill="%s"/></svg>' % (BLUE, BLUE, BLUE, BLUE))
    leg = ('<div style="display:flex;flex-direction:column;gap:9px" class="ctext">'
           '<div style="display:flex;align-items:center;gap:12px"><i class="swa" style="background:%s;opacity:.32;'
           'width:26px"></i><span>I<sub>h</sub>: one <i>g&#772;</i><sub>h</sub> on soma, basal and apical dendrites'
           '</span></div><div style="display:flex;align-items:center;gap:12px">%s<span>spines beyond 60 &#956;m, '
           'folded into <i>c</i><sub>m</sub> and <i>g</i><sub>pas</sub> by <i>F</i></span></div>'
           '<div style="display:flex;align-items:center;gap:12px"><span style="display:inline-flex;gap:2px">%s%s</span>'
           '<span>SS pulses and LS steps at the soma</span></div></div>'
           % (ORANGE, spine_glyph, mini_ss(40, 20), mini_ls(44, 20)))
    left = card(72, 236, 700, 790) + layer(art) + labels + at(104, 912, leg)
    pcard = ('<div style="background:#FFFFFF;border:1px solid #ECEAE4;border-radius:16px;padding:16px 18px 14px;'
             'border-left:6px solid %s;height:128px;box-shadow:0 6px 18px rgba(15,23,42,.05)">'
             '<div style="display:flex;align-items:baseline;gap:10px"><span style="font-size:34px;font-weight:800;'
             'color:%s">%s</span><span style="font-size:16px;font-weight:600;color:%s">%s</span></div>'
             '<div style="margin-top:9px;display:flex;gap:8px;flex-wrap:wrap"><span class="chip">%s</span>'
             '<span class="chip">%s</span><span class="chip" style="background:%s;color:%s">%s</span></div></div>')
    P = [("<i>C</i><sub>m</sub>", "membrane capacitance", "&#956;F/cm&#178;", "0.3 &#8211; 3.0", "log", BLUE),
         ("<i>R</i><sub>m</sub>", "membrane resistance", "&#937;&#183;cm&#178;", "10&#179; &#8211; 10&#8309;", "log",
          BLUE),
         ("<i>R</i><sub>a</sub>", "axial resistivity", "&#937;&#183;cm", "50 &#8211; 1000", "log", BLUE),
         ("<i>g&#772;</i><sub>h</sub>", "I<sub>h</sub> density", "S/cm&#178;", "10&#8315;&#8310; &#8211; 10&#8315;&#179;",
          "log", ORANGE),
         ("&#916;<i>v</i><sub>h</sub>", "shift of <i>m</i><sub>&#8734;</sub>", "mV", "&#8722;10 &#8211; +10", "linear",
          ORANGE),
         ("<i>&#954;</i><sub>&#964;</sub>", "scale of <i>&#964;</i><sup>h</sup>", "no unit", "0.5 &#8211; 2.0", "log",
          ORANGE)]
    cells = []
    for sym, name, unit, box, coord, colr in P:
        tint, tcol = (BLUE_T, "#1D5FB0") if colr == BLUE else (ORANGE_T, "#B4461A")
        cells.append(pcard % (colr, INK, sym, INK2, name, unit, "box " + box, tint, tcol, coord))
    grid = ('<div style="position:absolute;left:800px;top:240px;width:1048px">'
            '<div class="label" style="margin-bottom:12px">Passive &middot; fitted in every arm</div>'
            '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px">%s</div>'
            '<div class="label" style="margin:24px 0 12px">h-current &middot; added by the ih6 arm</div>'
            '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px">%s</div></div>'
            % ("".join(cells[:3]), "".join(cells[3:])))
    frow = ('<div style="display:grid;grid-template-columns:150px 1fr;column-gap:20px;align-items:baseline;'
            'padding:12px 0;border-top:1px solid #EEF0F3"><div class="cap">%s</div><div><div style="font-size:22px;'
            'color:%s">%s</div><div class="small" style="margin-top:5px">%s</div></div></div>')
    nocase = '<span style="text-transform:none;letter-spacing:0">%s</span>'
    formulas = card(800, 604, 1048, 422,
                    '<div class="label">How they enter the cable &middot; for each segment %s and each potential %s</div>'
                    '<div style="margin-top:8px">' % (nocase % "<i>s</i>", nocase % "<i>v</i>")
                    + frow % ("Membrane", INK, "<i>g</i><sub>pas</sub> = 1 / <i>R</i><sub>m</sub>,&nbsp;&nbsp; "
                              "<i>&#964;</i><sub>m</sub> = <i>R</i><sub>m</sub> <i>C</i><sub>m</sub>",
                              "NEURON, IClamp at the soma; in the passive fit <i>e</i><sub>pas</sub> = the LJP-corrected "
                              "<i>V</i><sub>rest</sub>, not fitted")
                    + frow % ("Spines", INK, "<i>c</i><sub>m</sub>(<i>s</i>) = <i>C</i><sub>m</sub> <i>F</i>(<i>s</i>),"
                              "&nbsp;&nbsp; <i>g</i><sub>pas</sub>(<i>s</i>) = <i>F</i>(<i>s</i>) / <i>R</i><sub>m</sub>",
                              "<i>F</i>(<i>s</i>) = <i>F</i> at path distance &#8805; 60 &#956;m, else 1; <i>F</i> = 1.9 "
                              "for L2/L3 (Eyal 2016): an input, not fitted")
                    + frow % ("I<sub>h</sub> kinetics", INK,
                              "<i>m</i><sub>&#8734;</sub>(<i>v</i>) = <i>m</i><sub>&#8734;</sub><sup>base</sup>"
                              "(<i>v</i> &#8722; &#916;<i>v</i><sub>base</sub> &#8722; &#916;<i>v</i><sub>h</sub>),"
                              "&nbsp;&nbsp; <i>&#964;</i><sup>h</sup>(<i>v</i>) = <i>&#954;</i><sub>&#964;</sub> "
                              "<i>&#964;</i><sup>h,base</sup>(<i>v</i> &#8722; &#916;<i>v</i><sub>base</sub>)",
                              "base: Rich et al. 2021, human (&#916;<i>v</i><sub>base</sub> = 0); &#916;<i>v</i><sub>h"
                              "</sub> moves the activation only")
                    + frow % ("Rest balance", INK,
                              "<i>e</i><sub>pas</sub>(<i>s</i>) = <i>V</i><sub>rest</sub> + <i>g&#772;</i><sub>h</sub>"
                              "(<i>s</i>) <i>m</i><sub>&#8734;</sub>(<i>V</i><sub>rest</sub>) (<i>V</i><sub>rest</sub>"
                              " &#8722; <i>E</i><sub>h</sub>) / <i>g</i><sub>pas</sub>(<i>s</i>)",
                              "the leak battery moves so that the model rests at the cell&rsquo;s own <i>V</i><sub>rest"
                              "</sub> for every trial <i>g&#772;</i><sub>h</sub>: not a seventh parameter")
                    + "</div>")
    return page("The model", "A reconstructed cable, a passive membrane and a uniform h-current",
                "Three passive parameters in the production fit; three more for the h-current in the six-parameter "
                "arm. The optimiser works in q = log &#952; (&#916;<i>v</i><sub>h</sub> linear), uniform in q.",
                left + grid + formulas,
                "Sources: Ih Fit param_spec.py (boxes); annual report part 1 &sect;3.2, &sect;7.2 (eq. 7.1-7.2); "
                "mod/Ih_human.mod. Neuron drawn schematically, not to scale.")


# ============================================================================= fig04
CURVES = [("Rich et al. 2021 &middot; human, the base", "Rich (human)", BLUE, dict(mechanism="Ih_human")),
          ("Kole et al. 2006 &middot; rat", "Kole (rat)", ORANGE, dict(mechanism="Ih")),
          ("Kole 2006 + Kalmbach 2018 shift (+20 mV)", "Kole + Kalmbach", AQUA, dict(mechanism="Ih", vshift_mV=20.0))]


def fig04():
    v = np.linspace(-131.0, -40.0, 456)
    body = [legend([("line", c[2], c[0]) for c in CURVES] + [("area", BLUE, "fitted box around the base")], 72, 240)]
    charts = []
    for k, which in enumerate(("m", "t")):
        cx0 = 72 + k * 896
        body.append(card(cx0, 284, 880, 556))
        body.append(at(cx0 + 28, 306, ("Activation &nbsp;<span style='font-weight:500;color:%s'><i>m</i><sub>&#8734;</sub>"
                                       "(<i>v</i>)</span>" if which == "m" else
                                       "Time constant &nbsp;<span style='font-weight:500;color:%s'><i>&#964;</i><sup>h"
                                       "</sup>(<i>v</i>)</span>") % INK2, style="font-size:19px;font-weight:700"))
        ch = Chart(cx0 + 104, 362, 740, 386, (-131.0, -40.0), (0.0, 1.0) if which == "m" else (0.0, 750.0))
        if which == "m":
            ch.ygrid([0, 0.2, 0.4, 0.6, 0.8, 1.0], "%.1f")
            lo = [hip.minf(x, "Ih_human", vshift_minf_mV=-10.0) for x in v]
            hi = [hip.minf(x, "Ih_human", vshift_minf_mV=10.0) for x in v]
        else:
            ch.ygrid([0, 150, 300, 450, 600, 750], "%d")
            lo = [hip.mtau_ms(x, "Ih_human", tau_scale=0.5) for x in v]
            hi = [hip.mtau_ms(x, "Ih_human", tau_scale=2.0) for x in v]
        ch.area(v, np.array(lo), np.array(hi), BLUE, 0.11)
        ch.vline(V_REST, "#A9B4C2", 1.25)
        for _, _, colr, kw in CURVES[::-1]:
            y = [hip.minf(x, **kw) if which == "m" else hip.mtau_ms(x, **kw) for x in v]
            ch.line(v, y, colr, 2.5)
        for _, _, colr, kw in CURVES:
            ch.dot(V_REST, hip.minf(V_REST, **kw) if which == "m" else hip.mtau_ms(V_REST, **kw), colr)
        ch.xticks([-130, -110, -90, -70, -50], "%d")
        ch.xlabel("membrane potential <i>v</i> (mV, LJP-corrected)")
        ch.text.append(at(ch.X(V_REST) + 8, ch.y + 4, "<i>V</i><sub>rest</sub> &#8722;73.5 mV", "tl",
                          "font-size:14px;font-weight:600;color:%s" % INK3))
        if which == "m":
            ch.ylabel("steady-state activation")
            # direct labels with leaders to the dots at rest (the curves converge elsewhere)
            order = [2, 0, 1]
            for j, yl in zip(order, [0.43, 0.335, 0.24]):
                _, lab, colr, kw = CURVES[j]
                yd = hip.minf(V_REST, **kw)
                p0 = (ch.X(-63.0), ch.Y(yl))
                ch.leader((p0[0] - 4, p0[1]), (ch.X(V_REST) + 6, ch.Y(yd) - 4))
                ch.text.append(at(p0[0], p0[1], '<span style="display:inline-flex;align-items:center;gap:8px">'
                                                 '<i class="sw" style="background:%s;width:16px"></i>%s</span>'
                                  % (colr, lab), "l", "font-size:14.5px;font-weight:600;color:%s" % INK2))
            ch.note(-129.0, 0.03, "shaded: the base shifted by<br/>&#916;<i>v</i><sub>h</sub> &#8712; [&#8722;10, +10] mV",
                    "bl", "color:%s;line-height:1.35" % INK3)
        else:
            ch.ylabel("milliseconds")
            ch.note(-64.0, 640.0, "shaded: the base scaled by<br/><i>&#954;</i><sub>&#964;</sub> &#8712; [0.5, 2]",
                    "l", "color:%s;line-height:1.35" % INK3)
            ch.leader((ch.X(-64.5), ch.Y(625.0)), (ch.X(-69.5), ch.Y(470.0)))
        charts.append(ch.html())
    stats = []
    for name, _, colr, kw in CURVES:
        m, t = hip.minf(V_REST, **kw), hip.mtau_ms(V_REST, **kw)
        stats.append('<div style="flex:1;background:#fff;border:1px solid #ECEAE4;border-radius:16px;padding:14px 20px;'
                     'box-shadow:0 6px 18px rgba(15,23,42,.05)"><div style="display:flex;align-items:center;gap:10px;'
                     'font-size:15px;font-weight:600;color:%s"><i class="sw" style="background:%s"></i>%s</div>'
                     '<div style="margin-top:6px;font-size:29px;font-weight:700;color:%s"><i>m</i><sub>&#8734;</sub> '
                     '%.3f <span style="color:#CBD5E1;font-weight:400">&nbsp;|&nbsp;</span> <i>&#964;</i><sup>h</sup> '
                     '%.0f ms</div></div>' % (INK2, colr, name, INK, m, t))
    body.append(at(72, 862, '<span class="cap">At rest, <i>V</i><sub>rest</sub> = &#8722;73.5 mV: '
                            '<span style="font-weight:500;color:%s">a 6-fold spread in <i>m</i><sub>&#8734;</sub>, '
                            '7-fold in <i>&#964;</i><sup>h</sup></span></span>' % INK3))
    body.append(at(72, 892, '<div style="display:flex;gap:16px;width:1776px">%s</div>' % "".join(stats)))
    return page("Why fit the kinetics", "Three published I<sub>h</sub> curves disagree where the cell sits",
                "Fixing one would bake its choice into every C<sub>m</sub>; the fit moves the human curve "
                "(&#916;<i>v</i><sub>h</sub>) and scales its speed (<i>&#954;</i><sub>&#964;</sub>) inside the shaded "
                "boxes.", "".join(body) + "".join(charts),
                "Computed with the fitter's reference kinetics (Ih Fit human_ih_params.py); values match decision "
                "D-005. Below about &#8722;107 mV the published human &#964;<sup>h</sup> formula falls under 1 ms "
                "(no floor).")


# ============================================================================= fig05
def fig05():
    cx, cy, rx, ry, hw, hh, gap = 930.0, 545.0, 290.0, 205.0, 118.0, 45.0, 12.0
    nodes = [("&#952; trial", "3 passive, or 6 with I<sub>h</sub>", (cx, cy - ry), BLUE),
             ("NEURON", "simulate the reconstructed cell", (cx + rx, cy), BLUE),
             ("Loss <i>L</i>(&#952;)", "distance to the data", (cx, cy + ry), ORANGE),
             ("GP surrogate", "a model of <i>L</i> over &#952;", (cx - rx, cy), BLUE)]
    b = []
    for title, sub, (x, y), colr in nodes:
        b.append(at(x, y, '<div class="wrap" style="width:%dpx;height:%dpx;background:#fff;border-radius:18px;border:2px solid %s;'
                          'box-shadow:0 10px 26px rgba(15,23,42,.08);text-align:center;padding-top:15px">'
                          '<div class="ctitle" style="font-size:21px">%s</div><div class="small" style="margin-top:4px;'
                          'color:%s">%s</div></div>' % (2 * hw, 2 * hh, colr, title, INK2, sub), "c"))
    a_out = math.degrees(math.acos((hw + gap) / rx))           # leaves a top/bottom node through its side
    a_in = math.degrees(math.asin((hh + gap) / ry))            # enters a left/right node through its top/bottom
    arcs = []
    for a0, a1 in [(-a_out, -a_in), (a_in, a_out), (180 - a_out, 180 - a_in), (180 + a_in, 180 + a_out)]:
        x0, y0 = cx + rx * math.cos(math.radians(a0)), cy + ry * math.sin(math.radians(a0))
        x1, y1 = cx + rx * math.cos(math.radians(a1)), cy + ry * math.sin(math.radians(a1))
        arcs.append(path("M%.1f,%.1f A%.1f,%.1f 0 0 1 %.1f,%.1f" % (x0, y0, rx, ry, x1, y1), BLUE, 2.5,
                         marker="ahb"))
    for ang, text, anc in ((-40, "run the cell", "bl"), (40, "compare with the data", "tl"),
                           (140, "update the surrogate", "tr"), (220, "acquisition: next &#952;", "br")):
        x = cx + (rx + 22) * math.cos(math.radians(ang))
        y = cy + (ry + 22) * math.sin(math.radians(ang))
        b.append(at(x, y, text, anc, "font-size:15px;font-weight:600;color:%s" % INK3))
    b.append(at(cx, cy, '<div style="text-align:center"><div class="label">skopt gp_minimize</div>'
                        '<div class="ctext" style="font-size:15.5px;margin-top:8px">run B: 100 calls, 50 initial<br/>'
                        'I<sub>h</sub> arms: 200 calls, 100 initial</div></div>', "c"))
    data = card(72, 250, 410, 610,
                '<div class="label">Data bundles of one real cell</div>'
                '<div style="margin-top:18px">%s</div><div class="ctitle" style="font-size:18px;margin-top:4px">'
                'SS pulse, hyperpolarising</div><div class="ctext" style="font-size:15.5px;margin-top:4px">averaged per '
                'polarity; scored on 0.5-100 ms with the weight e<sup>&#8722;(<i>t</i> &#8722; 0.5 ms) / 5 ms</sup></div>'
                '<div style="margin-top:22px">%s</div><div class="ctitle" style="font-size:18px;margin-top:4px">LS steps, '
                '1 s</div><div class="ctext" style="font-size:15.5px;margin-top:4px">run B: the 2 smallest, first 60 ms'
                '<br/>I<sub>h</sub> arms: <i>h</i><sub>2</sub> &#8230; <i>h</i><sub><i>n</i>&#8722;1</sub>, the whole '
                'step</div><div class="ctitle" style="font-size:18px;margin-top:22px">Held out for the gate</div>'
                '<div class="ctext" style="font-size:15.5px;margin-top:4px">I<sub>h</sub> arms: the spike-free '
                'depolarising steps <i>d</i><sub>1</sub> &#8230; <i>d</i><sub><i>k</i></sub>, <i>h</i><sub>1</sub> and '
                'the SS depolarising pulses</div><div class="small" style="margin-top:14px">each bundle <i>b</i> is '
                'scored on its own window <i>W</i><sub><i>b</i></sub></div>' % (mini_ss(240, 60), mini_ls(240, 60)))
    y_top = cy - ry - 12
    best = at(1110, y_top, '<div style="width:224px;text-align:center;background:%s;color:#fff;border-radius:999px;'
                           'padding:8px 0;font-size:15px;font-weight:700">best <span class="hat hatw">&#952;</span> after '
                           'the budget</div>' % INK, "l")
    gate = card(1440, 250, 408, 290,
                '<div class="label">Gate on held-out bundles</div><div class="small" style="margin-top:6px">train and '
                'validation RMSD recomputed in mV, read in this order</div><div style="margin-top:12px;display:grid;'
                'grid-template-columns:auto 1fr;column-gap:12px;row-gap:7px;align-items:center;font-size:15px;color:%s">'
                '%s<span>train &gt; 2 mV</span>%s<span>validation &#8804; 0.2 mV</span>%s<span>validation / train '
                '&#8804; 3</span>%s<span>validation / train &#8804; 10</span>%s<span>otherwise</span></div>'
                % (INK2, pill("planned", "failed"), pill("done", "good"), pill("done", "good"),
                   pill("pushed", "to refine"), pill("planned", "failed")))
    ph3 = card(1440, 566, 408, 234,
               '<div class="label">Phase 3 &middot; uncertainty</div><div class="ctitle" style="font-size:19px;'
               'margin-top:12px">Bootstrap, B = 200</div><div class="ctext" style="font-size:15.5px">warm-started refits '
               '&#8594; percentile and BCa intervals</div><div class="ctitle" style="font-size:19px;margin-top:14px">'
               'Profile likelihood</div><div class="ctext" style="font-size:15.5px">is each axis identifiable?</div>')
    ws = [hcurve(482, 730, cx - hw - 4, cy + ry + 26, color=MUTE),
          path("M%.1f,%.1f L1106,%.1f" % (cx + hw + 4, y_top, y_top)),
          path("M1336,%.1f C1400,%.1f 1400,395 1436,395" % (y_top, y_top)),
          path("M1336,%.1f C1410,%.1f 1404,683 1436,683" % (y_top, y_top))]
    b.append(layer(*(arcs + ws)))
    b.append(at(498, 722, "data", "bl", "font-size:14px;font-weight:600;color:%s" % INK3))
    loss = card(500, 858, 860, 128,
                '<div style="display:flex;align-items:center;gap:22px"><div class="label">The loss</div>'
                '<div style="font-size:25px"><i>L</i>(&#952;) = |<i>B</i>|<sup>&#8722;1</sup> &#931;<sub><i>b</i></sub> '
                'RMSD<sub><i>b</i></sub>(&#952; | <i>W</i><sub><i>b</i></sub>) / <i>D</i><sub><i>b</i></sub></div></div>'
                '<div class="ctext" style="font-size:15px;margin-top:8px">mean over the bundles <i>b</i> &#8712; <i>B</i>; '
                'each RMSD on its window <i>W</i><sub><i>b</i></sub> and divided by the peak deflection '
                '<i>D</i><sub><i>b</i></sub>, so the sub-mV pulse is not drowned by the steps</div>')
    return page("How a cell is fitted", "One loop: simulate, score, let a Gaussian process choose the next guess",
                "The same loop fits the three passive parameters (run B) and the six of the I<sub>h</sub> arms; only the "
                "bundles, the windows and the budget change.", data + "".join(b) + best + gate + ph3 + loss,
                "Sources: annual report part 1 &sect;3.3-3.6, &sect;7.3; Biological Fit guide &sect;8 (the gate, "
                "_classify_fit); submit_biological_fit.sh and run_ih_fit.py (budgets).")


# ============================================================================= fig06
def fig06():
    layers_ = ["L2", "L3", "L4", "L5", "L6"]
    n = [10, 18, 6, 8, 5]
    cm = [1.478, 1.062, 0.785, 0.797, 1.100]
    rm = [17773, 26220, 44647, 51064, 42057]
    eyal, eyal_sd = 0.47, 0.03
    tile_ = ('<div style="flex:1;padding:0 30px;%s"><div class="cap" style="font-weight:600;color:%s">%s</div>'
             '<div style="font-size:46px;font-weight:700;letter-spacing:-.01em;color:%s;margin-top:4px">%s</div>'
             '<div class="small" style="margin-top:2px">%s</div></div>')
    sep = "border-left:1px solid #EEF0F3"
    tiles = ('<div style="display:flex">'
             + tile_ % ("", INK2, "Cells fitted, run B", INK, "68", "ten groups (L2-L6, exc and inh), free R<sub>a</sub>")
             + tile_ % (sep, INK2, "C<sub>m</sub> over the human L2/3 value", INK,
                        "%.1f&#215; &nbsp;and&nbsp; %.1f&#215;" % (cm[0] / eyal, cm[1] / eyal),
                        "L2 and L3 medians against Eyal 2016&rsquo;s 0.47 &#956;F/cm&#178;")
             + tile_ % (sep, INK2, "R<sub>m</sub>: relative width of the 95 % interval", INK, "0.59",
                        "the least determined of the three (C<sub>m</sub> 0.13, R<sub>a</sub> 0.19)") + "</div>")
    body = [card(72, 236, 1776, 136, "", "padding:22px 0"), at(72, 258, '<div style="width:1776px">%s</div>' % tiles)]
    value_style = "font-size:16px;font-weight:700;color:%s" % INK
    cat_style = "font-size:15px;font-weight:700;color:%s" % INK2
    # C_m by layer
    body.append(card(72, 396, 704, 610))
    body.append(at(100, 418, "C<sub>m</sub> by layer &nbsp;<span style='font-weight:500;color:%s'>excitatory medians, "
                             "&#956;F/cm&#178;</span>" % INK3, style="font-size:19px;font-weight:700"))
    body.append(legend([("line", INK, "human L2/3, Eyal 2016: 0.47 &#177; 0.03")], 100, 458))
    ch = Chart(172, 508, 568, 400, (-0.5, 4.5), (0.0, 1.6))
    ch.ygrid([0, 0.4, 0.8, 1.2, 1.6], "%.1f")
    for i, (c, k) in enumerate(zip(cm, n)):
        ch.bar(i, c, 24, BLUE if i < 2 else GREY)
        ch.text.append(at(ch.X(i), ch.Y(c) - 16, "%.2f" % c, "c", value_style))
        ch.text.append(at(ch.X(i), ch.y + ch.h + 22, layers_[i], "c", cat_style))
        ch.text.append(at(ch.X(i), ch.y + ch.h + 44, "n = %d" % k, "c", TICK))
    ch.over.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" opacity=".10"/>'
                   % (ch.X(-0.45), ch.Y(eyal + eyal_sd), ch.X(1.45) - ch.X(-0.45),
                      ch.Y(eyal - eyal_sd) - ch.Y(eyal + eyal_sd), INK))
    ch.seg(-0.45, eyal, 1.45, eyal, INK, 2.0)
    body.append(ch.html())
    # R_m by layer
    body.append(card(792, 396, 560, 610))
    body.append(at(820, 418, "R<sub>m</sub> by layer &nbsp;<span style='font-weight:500;color:%s'>medians, "
                             "k&#937;&#183;cm&#178;</span>" % INK3, style="font-size:19px;font-weight:700"))
    ch = Chart(870, 508, 450, 400, (-0.5, 4.5), (0.0, 60.0))
    ch.ygrid([0, 15, 30, 45, 60], "%d")
    for i, r in enumerate(rm):
        ch.bar(i, r / 1000.0, 24, BLUE)
        ch.text.append(at(ch.X(i), ch.Y(r / 1000.0) - 16, "%.1f" % (r / 1000.0), "c", value_style))
        ch.text.append(at(ch.X(i), ch.y + ch.h + 22, layers_[i], "c", cat_style))
    body.append(ch.html())
    # bootstrap widths
    body.append(card(1368, 396, 480, 610))
    body.append(at(1396, 418, "Bootstrap 95 % interval", style="font-size:19px;font-weight:700"))
    body.append(at(1396, 450, "median relative width, 31 cells", style="font-size:15px;color:%s" % INK3))
    ch = Chart(1446, 508, 370, 400, (-0.5, 2.5), (0.0, 0.6))
    ch.ygrid([0, 0.2, 0.4, 0.6], "%.1f")
    for i, (nm, v_) in enumerate((("C<sub>m</sub>", 0.131), ("R<sub>a</sub>", 0.186), ("R<sub>m</sub>", 0.592))):
        ch.bar(i, v_, 24, BLUE if i == 2 else GREY)
        ch.text.append(at(ch.X(i), ch.Y(v_) - 16, "%.3f" % v_, "c", value_style))
        ch.text.append(at(ch.X(i), ch.y + ch.h + 22, nm, "c", cat_style))
    body.append(ch.html())
    return page("Passive fit &middot; results", "The passive-only fit returns 2-3 times the human capacitance",
                "Run B is the estimate of record; its C<sub>m</sub> is an effective value that absorbs the h-current into "
                "the passive skeleton &mdash; the reason for the six-parameter fit.", "".join(body),
                "Sources: annual report part 1 &sect;6 (medians from the cluster's phase2_results.csv, run B = dt01ls60; "
                "inhibitory groups not shown); Eyal et al. 2016, eLife 5:e16553, p. 2 (mean &#177; SD, n = 6).")


# ============================================================================= fig07
def noise_sweep(seed=20261103, b=0.1):
    """One 1 s sweep at 50 kHz under theta_cf: all terms, and the slow part + tilt on the same draws."""
    dt, n = 1.0 / 50000.0, 50000
    rng = np.random.default_rng(seed)
    units = nl.draw_units(rng, 1, n)
    tot = nl.simulate_traces(THETA_CF, dt, units.fast, units.slow, tilt_mV_per_s=np.array([b]))[0]
    slow_law = nl.NoiseLaw(0.0, THETA_CF.rho_f, THETA_CF.sigma_s, THETA_CF.tau_s)
    slow = nl.simulate_traces(slow_law, dt, units.fast, units.slow, tilt_mV_per_s=np.array([b]))[0]
    return dt, np.arange(n) * dt, tot, slow, b


def fig07():
    th = THETA_CF
    vf, vs = th.sigma_f ** 2, th.sigma_s ** 2
    share_f = vf / (vf + vs)
    term = ('<div style="flex:%s;border-radius:16px;padding:13px 18px;background:#fff;border:1px solid #ECEAE4;'
            'border-top:5px solid %s;box-shadow:0 6px 18px rgba(15,23,42,.05)"><div style="font-size:19px;color:%s">%s'
            '</div><div class="small" style="margin-top:5px;color:%s">%s</div></div>')
    plus = '<div style="font-size:30px;font-weight:300;color:#94A3B8;align-self:center">+</div>'
    eq = ('<div style="position:absolute;left:72px;top:246px;width:1776px;display:flex;gap:12px;align-items:stretch">'
          '<div style="align-self:center;font-size:32px;font-weight:600;margin-right:6px"><i>e</i><sub><i>j</i></sub> ='
          '</div>'
          + term % (1.25, BLUE, INK, "<i>e</i>%s &nbsp;<span class='small'>fast, AR(1)</span>"
                    % supsub("fast", "<i>j</i>"), INK2,
                    "&#963;<sub>f</sub> = 0.038 mV, &#961;<sub>f</sub> = 0.175 per 20 &#956;s sample<br/>"
                    "gone within ~0.1 ms") + plus
          + term % (1.25, ORANGE, INK, "<i>e</i>%s &nbsp;<span class='small'>slow, Ornstein-Uhlenbeck</span>"
                    % supsub("slow", "<i>j</i>"), INK2,
                    "&#963;<sub>s</sub> = 0.051 mV, &#964;<sub>s</sub> = 36.6 ms<br/>what a 265 ms window still sees")
          + plus
          + term % (0.9, GREY, INK, "<i>a</i> &nbsp;<span class='small'>offset</span>", INK2,
                    "one per sweep<br/>invisible to every demeaned statistic") + plus
          + term % (1.1, AQUA, INK, "<i>b</i> (<i>t</i><sub><i>j</i></sub> &#8722; <i>t&#772;</i>) &nbsp;"
                    "<span class='small'>tilt</span>", INK2, "<i class='hat'>b</i>(<i>c</i>, <i>s</i>) read on the "
                                                            "real sweep<br/>and copied into its twin, not drawn")
          + '</div>')
    body = [eq]
    # left: where the variance sits, and how fast each part forgets
    body.append(card(72, 416, 660, 590))
    body.append(at(100, 438, "Autocovariance by component", style="font-size:19px;font-weight:700"))
    body.append(at(100, 474, '<div style="display:flex;align-items:center;gap:14px;font-size:14.5px;color:%s">'
                             '<span>variance at lag 0</span><div style="display:flex;width:300px;height:14px;gap:2px">'
                             '<div style="flex:%.3f;background:%s;border-radius:4px 0 0 4px"></div><div style="flex:%.3f;'
                             'background:%s;border-radius:0 4px 4px 0"></div></div><span><b style="color:%s">%.0f %%</b> '
                             'fast, <b style="color:%s">%.0f %%</b> slow</span></div>'
                   % (INK2, share_f, BLUE, 1 - share_f, ORANGE, INK, 100 * share_f, INK, 100 * (1 - share_f))))
    body.append(legend([("line", BLUE, "fast AR(1), stacked on the slow part"), ("line", ORANGE, "slow OU")], 100, 508))
    dt = 1.0 / 50000.0
    lags = np.unique(np.round(np.geomspace(1, 50000, 500)).astype(int))
    t_ms = lags * dt * 1e3
    fast = 1e3 * vf * th.rho_f ** lags
    slow = 1e3 * vs * np.exp(-lags * dt / th.tau_s)
    ch = Chart(174, 560, 524, 336, (t_ms[0], 1000.0), (0.0, 3.0), xlog=True)
    ch.ygrid([0, 1, 2, 3], "%d")
    ch.xgrid([0.1, 1, 10, 100, 1000])
    ch.area(t_ms, np.zeros_like(slow), slow, ORANGE, 0.12)
    ch.area(t_ms, slow, slow + fast, BLUE, 0.22)
    ch.line(t_ms, slow, ORANGE, 2.5)
    keep = fast > 1e-3 * fast[0]                              # the fast edge only where the wedge exists
    ch.line(t_ms[keep], (slow + fast)[keep], BLUE, 2.5)
    ch.vline(th.tau_s * 1e3, "#A9B4C2", 1.25)
    ch.text.append(at(ch.X(th.tau_s * 1e3) + 8, ch.y + 4, "<i>&#964;</i><sub>s</sub> = 36.6 ms", "tl",
                      "font-size:14px;font-weight:600;color:%s" % INK3))
    ch.xticks([0.1, 1, 10, 100, 1000], lambda t: ("%g" % t))
    ch.xlabel("lag (ms), log scale; first point = one 20 &#956;s sample")
    ch.ylabel("10<sup>&#8722;3</sup> mV&#178;", 52)
    body.append(ch.html())
    # right: one simulated sweep
    dt, t, tot, slow_tr, bb = noise_sweep()
    body.append(card(752, 416, 1096, 590))
    body.append(at(780, 438, "One simulated sweep &nbsp;<span style='font-weight:500;color:%s'>1 s at 50 kHz, "
                             "&#952;<sub>cf</sub>, <i>b</i> = 0.1 mV/s</span>" % INK3,
                   style="font-size:19px;font-weight:700"))
    body.append(legend([("area", "#8C9AAE", "all terms"), ("line", ORANGE, "slow part + tilt"),
                        ("line", AQUA, "tilt <i>b</i> (<i>t</i> &#8722; <i>t&#772;</i>)")], 780, 476))
    ch = Chart(850, 530, 960, 366, (0.0, 1.0), (-0.3, 0.3))
    ch.ygrid([-0.3, -0.15, 0, 0.15, 0.3], "%.2f")
    edges = np.linspace(0, t.size, 961).astype(int)
    tb = np.array([t[a:b_].mean() for a, b_ in zip(edges[:-1], edges[1:])])
    lo = np.array([tot[a:b_].min() for a, b_ in zip(edges[:-1], edges[1:])])
    hi = np.array([tot[a:b_].max() for a, b_ in zip(edges[:-1], edges[1:])])
    ch.area(tb, lo, hi, "#8C9AAE", 0.45)
    ch.line(t[::25], slow_tr[::25], ORANGE, 2.0)
    ch.line(np.array([0.0, 1.0]), bb * (np.array([0.0, 1.0]) - t.mean()), AQUA, 2.5)
    ch.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], "%.1f")
    ch.xlabel("time in the sweep (s)")
    ch.ylabel("mV", 64)
    body.append(ch.html())
    return page("Recording noise", "Each twin sweep must carry the noise of its own real sweep",
                "A fast AR(1) part, a slow Ornstein-Uhlenbeck part and a per-sweep tilt: on the L3_exc cells the Long "
                "Square noise is far from white (autocorrelation 0.566 at 1 ms, where AR(1) gives 0.000).",
                "".join(body),
                "For one sweep, samples <i>j</i> at <i>t</i><sub><i>j</i></sub> = <i>j</i>&#916;<i>t</i>, mid-time "
                "<i>t&#772;</i>; &#952;<sub>cf</sub> = the cohort closed-form law (L3_exc). Computed with the "
                "repository's noise_law.py (Block 2), seed 20261103. Sources: SPEC &sect;2.1; D-016, D-017, D-102, "
                "D-107; annual report part 1 &sect;7.5.")


# ============================================================================= fig08
def fig08(imp):
    st = lambda i: imp[i]["status"]  # noqa: E731
    rng = np.random.default_rng(8)
    _, _, _, slow_tr, _ = noise_sweep(seed=7, b=0.0)
    pic_noise = slow_tr[:12000:30] + 0.012 * np.random.default_rng(9).standard_normal(400)
    box = ('<div class="wrap" style="width:%dpx;height:%dpx;background:#fff;border-radius:18px;border:2px solid %s;'
           'box-shadow:0 10px 26px rgba(15,23,42,.07);padding:16px 20px">%s</div>')
    b = [card(72, 250, 470, 452,
              '<div class="cap">The real cell <i>c</i></div>'
              '<div style="margin-top:12px">%s</div><div class="small" style="color:%s;margin-top:2px"><b>SS sweep</b>: '
              '37 pulses, 200.5 ms apart; each window [<i>t</i><sub>SS</sub>(<i>c</i>), 200) ms after a pulse, its tilt '
              'offset removed, the median window of its sweep and polarity subtracted</div>'
              '<div style="margin-top:14px">%s</div><div class="small" style="color:%s;margin-top:2px"><b>LS sweep</b>: '
              'the 265 ms pre-window before the step; the last 500 ms read the tilt only</div>'
              % (picto_ss_train(418, 100, rng), INK2, picto_ls_sweep(418, 100, rng), INK2)),
         card(1378, 250, 470, 452,
              '<div class="cap">The simulator</div><div style="font-size:21px;margin-top:12px">law &#952; = '
              '(&#963;<sub>f</sub>, &#961;<sub>f</sub>, &#963;<sub>s</sub>, &#964;<sub>s</sub>)</div><div class="small" '
              'style="color:%s;margin-top:4px">+ each sweep&rsquo;s real tilt <i class="hat">b</i>(<i>c</i>, <i>s</i>), '
              'fixed in every replicate</div><div style="margin-top:16px">%s</div><div class="small" style="color:%s;'
              'margin-top:4px">continuous sweeps at the cell&rsquo;s own layout, cut and processed exactly like the '
              'data, with common random numbers across &#952;</div>' % (INK2, picto_noise(418, 150, pic_noise), INK2))]
    b.append(at(590, 300, box % (340, 142, BLUE,
                                 '<div style="font-size:21px;font-weight:700"><i class="hat">q</i>(<i>c</i>) &#8712; '
                                 '&#8477;<sup>15</sup></div><div class="small" style="margin-top:6px;color:%s">medians over '
                                 'the windows of SD and textbook autocorrelation at fixed lags</div>' % INK2)))
    b.append(at(990, 300, box % (340, 142, ORANGE,
                                 '<div style="font-size:21px;font-weight:700"><i>q</i><sub>sim</sub>(&#952;) &#8712; '
                                 '&#8477;<sup>15</sup></div><div class="small" style="margin-top:6px;color:%s">the same '
                                 'statistics, Harrell-Davis medians over <i>M</i> = 3000 simulated windows</div>' % INK2)))
    b.append(at(700, 522, box % (520, 136, INK,
                                 '<div style="font-size:24px"><i>J</i>(&#952;) = (<i>q</i><sub>sim</sub> &#8722; '
                                 '<i class="hat">q</i>)<sup>T</sup> <i>W</i> (<i>q</i><sub>sim</sub> &#8722; <i class="hat">'
                                 'q</i>)</div><div class="small" style="margin-top:8px;color:%s"><i>W</i> = inverse of the '
                                 'simulated covariance, re-estimated at each new <span class="hat">&#952;</span> until it '
                                 'settles</div>' % INK2)))
    ws = [path("M542,371 L586,371", BLUE, 2.5, marker="ahb"), path("M1378,371 L1334,371", ORANGE, 2.5, marker="aho"),
          vcurve(760, 442, 850, 518, color=BLUE, marker="ahb"), vcurve(1160, 442, 1070, 518, color=ORANGE, marker="aho"),
          path("M1220,610 C1290,610 1300,640 1374,640", MUTE, 2.2, dash="6 6")]
    b.append(layer(*ws))
    b.append(at(1232, 662, "least-squares step<br/>updates &#952;", "tl",
                "font-size:13.5px;font-weight:600;color:%s;line-height:1.35" % INK3))
    steps = [("Per-cell fit <span class='hat'>&#952;</span>(<i>c</i>)", "Blocks 1-4", st("N4"), None),
             ("Per-sweep stage", "Block 9: one solve per sweep, bootstrap variance, pooling per protocol", st("N4b"),
              "planned &middot; SPEC drafted"),
             ("Twins&rsquo; noise", "Block 5: each twin sweep gets its real sweep&rsquo;s law and tilt", st("N5"), None),
             ("Recovery gate", "Stage 7 on davinci", st("S7"), None)]
    x0 = 72
    track, arrows = [], []
    for (t_, s_, st_, lab), w_ in zip(steps, [360, 520, 480, 304]):
        track.append(card(x0, 772, w_, 170, '<div class="ctitle" style="font-size:19px">%s</div><div class="ctext" '
                                             'style="font-size:15px;margin-top:8px">%s</div><div style="margin-top:12px">'
                                             '%s</div>' % (t_, s_, pill(st_, lab))))
        if x0 + w_ < 1840:
            arrows.append(path("M%d,857 L%d,857" % (x0 + w_ + 6, x0 + w_ + 30)))
        x0 += w_ + 36
    b.append(at(72, 738, "From the cell&rsquo;s law to the twins", style="font-size:13px;font-weight:700;"
                                                                         "letter-spacing:.14em;text-transform:uppercase;"
                                                                         "color:%s" % INK3))
    return page("Recording noise &middot; the fit", "Indirect inference: match the cell&rsquo;s statistics with "
                                                     "simulated ones",
                "A window statistic is not a parameter, so the law is fitted through the very estimators applied to "
                "the data, on simulated copies of the cell&rsquo;s own sweeps.",
                "".join(b) + "".join(track) + layer(*arrows),
                "Sources: Ih Fit specs/SPEC.md &sect;2.1-2.5; decisions D-033, D-048 to D-051, D-056, D-095 to D-116. "
                "Pictograms schematic, not to scale.")


# ============================================================================= fig09
def fig09():
    dots = []
    for i in range(20):
        r, c = divmod(i, 5)
        x, y = 32 + c * 66, 32 + r * 66
        if i % 4 == 3:          # synth_gt_grid: fp = (arange(n) % every) == (every - 1), every = 4
            dots.append('<circle cx="%d" cy="%d" r="23" fill="#F8FAFC" stroke="#94A3B8" stroke-width="2.2" '
                        'stroke-dasharray="5 4"/>' % (x, y))
        else:
            dots.append('<circle cx="%d" cy="%d" r="23" fill="%s"/><circle cx="%d" cy="%d" r="8.5" fill="#fff"/>'
                        % (x, y, ORANGE, x, y))
    grid = '<svg width="332" height="266">%s</svg>' % "".join(dots)

    def range_bar(label, lo_box, hi_box, lo_t, hi_t, log, truths, box, colr):
        f = (lambda q: math.log(q)) if log else (lambda q: q)
        a, b_ = f(lo_box), f(hi_box)
        p0, p1 = (f(lo_t) - a) / (b_ - a) * 100, (f(hi_t) - a) / (b_ - a) * 100
        return ('<div style="margin-top:13px"><div class="row"><span style="font-size:17px;color:%s">%s</span>'
                '<span class="small">truths %s</span></div><div style="position:relative;height:10px;border-radius:5px;'
                'background:#EEF2F6;margin-top:6px"><div style="position:absolute;left:%.1f%%;width:%.1f%%;top:0;'
                'bottom:0;border-radius:5px;background:%s"></div></div><div class="row small" style="margin-top:3px;'
                'font-size:12.5px"><span>%s</span><span>fit box (%s)</span><span>%s</span></div></div>'
                % (INK, label, truths, p0, p1 - p0, colr, box[0], "log" if log else "linear", box[1]))
    truths = (range_bar("<i>g&#772;</i><sub>h</sub>", 1e-6, 1e-3, 2e-5, 3e-4, True,
                        "2&#183;10&#8315;&#8309; &#8211; 3&#183;10&#8315;&#8308; S/cm&#178;",
                        ("10&#8315;&#8310;", "10&#8315;&#179;"), ORANGE)
              + range_bar("&#916;<i>v</i><sub>h</sub>", -10, 10, -5, 5, False, "&#8722;5 &#8211; +5 mV",
                          ("&#8722;10", "+10"), ORANGE)
              + range_bar("<i>&#954;</i><sub>&#964;</sub>", 0.5, 2.0, 0.7, 1.4, True, "0.7 &#8211; 1.4",
                          ("0.5", "2.0"), ORANGE)
              + range_bar("<i>C</i><sub>m</sub>", 0.3, 3.0, 0.4, 1.5, True, "0.4 &#8211; 1.5 &#956;F/cm&#178;",
                          ("0.3", "3.0"), BLUE)
              + range_bar("<i>R</i><sub>a</sub>", 50.0, 1000.0, 100.0, 500.0, True, "100 &#8211; 500 &#937;&#183;cm",
                          ("50", "1000"), BLUE))
    check = ('<div style="display:flex;gap:12px;align-items:flex-start"><div style="width:24px;height:24px;'
             'border-radius:7px;border:2px solid #CBD5E1;flex:none;margin-top:1px"></div><div class="ctext" '
             'style="font-size:16px">%s</div></div>')
    arms = "".join('<div class="chip row" style="font-size:16px;padding:9px 12px;%s"><span>%s</span>'
                   '<span style="font-size:13.5px;font-weight:600;opacity:.8">%s</span></div>' % (s_, a, k)
                   for a, k, s_ in (("baseline_runB", "3 parameters", ""), ("passive_fullstep", "3 parameters", ""),
                                    ("ih4", "4: kinetics at the base", ""),
                                    ("<b>ih6</b>", "6 &middot; the gated arm", "background:%s;color:#B4461A"
                                                                               % ORANGE_T)))
    checks = "".join(check % c_ for c_ in (
        "median |&#949;| &#8804; log 1.25 for <i>C</i><sub>m</sub> and <i>g&#772;</i><sub>h</sub>",
        "median |&#949;| &#8804; 3 mV for &#916;<i>v</i><sub>h</sub>",
        "median |&#949;| &#8804; log 1.5 for <i>&#954;</i><sub>&#964;</sub>",
        "&#8805; 80 % of the controls rail <i>g&#772;</i><sub>h</sub> at the low bound"))
    b = [card(72, 250, 380, 456,
              '<div class="label">Up to 20 twins</div><div class="ctext" style="margin-top:6px">each on a real L3_exc '
              'morphology (raw SWC, <i>F</i> = 1.9)</div><div style="margin-top:14px">%s</div>'
              '<div style="display:flex;flex-direction:column;gap:6px;margin-top:4px" class="small">'
              '<span style="display:flex;align-items:center;gap:10px;color:%s"><i style="width:14px;height:14px;'
              'border-radius:50%%;background:%s;display:inline-block;flex:none"></i><span>I<sub>h</sub>-bearing twin'
              '</span></span><span style="display:flex;align-items:center;gap:10px;color:%s"><i style="width:14px;'
              'height:14px;border-radius:50%%;border:2px dashed #94A3B8;display:inline-block;box-sizing:border-box;'
              'flex:none"></i><span>control, every 4th: <i>g&#772;</i><sub>h</sub> at the box floor</span></span></div>'
              % (grid, INK2, ORANGE, INK2)),
         card(476, 250, 470, 456,
              '<div class="label">Known truths, inside the fit boxes</div>%s<div class="small wrap" style="margin-top:12px">'
              '<i>R</i><sub>m</sub>: (<i>C</i><sub>m</sub>, <i>R</i><sub>m</sub>) drawn by rejection until '
              '<i>&#964;</i><sub>m</sub> = <i>R</i><sub>m</sub><i>C</i><sub>m</sub> lies in 3-40 ms</div>' % truths),
         card(970, 250, 380, 456,
              '<div class="label">Fitted in four arms</div><div style="display:flex;flex-direction:column;gap:10px;'
              'margin-top:16px">%s</div><div class="ctext" style="margin-top:18px;font-size:15px">each at 200 GP calls '
              '(100 initial), through the production driver run_ih_fit</div>' % arms),
         card(1374, 250, 474, 456,
              '<div class="row"><div class="label">Gate on ih6</div>%s</div><div style="margin-top:16px;display:flex;'
              'flex-direction:column;gap:13px">%s</div><div class="small" style="margin-top:14px">&#949; per axis: '
              'the log ratio to the truth, in mV for &#916;<i>v</i><sub>h</sub>; railed axes counted, never dropped'
              '</div><div class="small" style="margin-top:12px;padding:9px 12px;background:#FDF2F2;border-radius:10px;'
              'color:#9B1C1C">D-083: run only once each twin sweep carries its real sweep&rsquo;s fitted noise and the '
              'real cell&rsquo;s own protocol</div>' % (pill("pushed", "proposed &middot; not agreed"), checks)),
         card(72, 780, 870, 176,
              '<div style="display:flex;align-items:center;gap:14px">%s<div class="ctitle" style="font-size:20px">'
              'Stage 8: the campaign on real cells</div></div><div style="display:flex;align-items:center;gap:10px;'
              'margin-top:22px">%s</div>' % (pill("done", "pass"), flow("one diameter-corrected cell",
                                                                        "the corrected set", "every excitatory group"))),
         card(978, 780, 870, 176,
              '<div style="display:flex;align-items:center;gap:14px">%s<div class="ctitle" style="font-size:20px">'
              'Freeze or change</div></div><div style="display:flex;flex-direction:column;gap:10px;margin-top:16px">'
              '<div style="display:flex;align-items:center;gap:10px">%s</div><div style="display:flex;'
              'align-items:center;gap:10px">%s</div></div>'
              % (pill("planned", "fail"), flow("a kinetic knob fails", "it is frozen at its base value"),
                 flow("<i>C</i><sub>m</sub> or <i>g&#772;</i><sub>h</sub> fails", "the protocol must change")))]
    ws = [path("M452,478 L472,478"), path("M946,478 L966,478"), path("M1350,478 L1370,478"),
          path("M1611,706 L1611,736 Q1611,742 1605,742 L513,742 Q507,742 507,748 L507,776"),
          path("M1413,742 L1413,776")]
    b.append(layer(*ws))
    return page("Stage 7 &middot; the recovery gate", "Before any real six-parameter fit: recover a known truth",
                "Synthetic twins with chosen parameters, fitted exactly as a real cell would be; a quarter carry no "
                "I<sub>h</sub> worth the name, to catch a fit that always finds some. Built and smoke-tested; never run.",
                "".join(b),
                "Sources: Ih Fit run_ih_recovery.py, submit_ih_recovery.sh, synth_gt_grid.py (controls: "
                "g&#772;<sub>h</sub> = 10&#8315;&#8310; S/cm&#178;, &#916;v<sub>h</sub> = 0, &#954;<sub>&#964;</sub> = 1), "
                "ih_recovery_report.py; decisions D-005, D-011, D-083. Tolerances: the assistant's proposal.")


# ============================================================================= main
def render(pages):
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
        for name, html in pages:
            pg.set_content(html, wait_until="load")
            pg.evaluate("document.fonts.ready")
            pg.wait_for_timeout(150)
            pg.screenshot(path=str(OUT / name), clip={"x": 0, "y": 0, "width": W, "height": H})
            try:                                    # lossless re-compression, about 40 % smaller
                from PIL import Image
                Image.open(OUT / name).save(OUT / name, "PNG", optimize=True)
            except ImportError:
                pass
            print("wrote", OUT / name)
        browser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", required=True, help="a roadmap.py status --json output for allen-mtg-fit")
    ap.add_argument("--only", default="", help="comma-separated figure numbers, e.g. 1,4")
    args = ap.parse_args()
    d, imp, impl_list, theo_list = load_status(args.status)
    builders = {
        1: ("fig01_pipeline_status.png", lambda: fig01(d, imp)),
        2: ("fig02_roadmap_board.png", lambda: fig02(d, impl_list, theo_list)),
        3: ("fig03_model_parameters.png", fig03),
        4: ("fig04_ih_kinetics.png", fig04),
        5: ("fig05_fit_loop.png", fig05),
        6: ("fig06_passive_results.png", fig06),
        7: ("fig07_noise_law.png", fig07),
        8: ("fig08_noise_fit.png", lambda: fig08(imp)),
        9: ("fig09_recovery_gate.png", fig09),
    }
    want = [int(x) for x in args.only.split(",") if x.strip()] or sorted(builders)
    render([(builders[k][0], builders[k][1]()) for k in want])
    for name, _, _, kw in CURVES:
        print("at V_rest %.1f mV: %-44s m_inf %.3f  tau_h %.1f ms"
              % (V_REST, name.replace("&middot;", "-"), hip.minf(V_REST, **kw), hip.mtau_ms(V_REST, **kw)))
    vf, vs = THETA_CF.sigma_f ** 2, THETA_CF.sigma_s ** 2
    print("theta_cf: fast share of the lag-0 variance %.3f; C_m L2, L3 over 0.47: %.2f, %.2f"
          % (vf / (vf + vs), 1.478 / 0.47, 1.062 / 0.47))


if __name__ == "__main__":
    main()
