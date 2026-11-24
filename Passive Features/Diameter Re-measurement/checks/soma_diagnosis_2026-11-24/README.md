# Soma fit diagnosis, 2026-11-24 (the gallery chat)

Standalone checks behind handoff 14.6 and `docs/TEEG_soma_fit_handoff_2026-11-24.md`; not pipeline code. They read
the pilot's soma-fit outputs of the user's Cell 25 run of 2026-11-24 (uploaded at 14:15, free far slope):
`soma_fit_529878215.json` and the corrected SWC `reconstruction_soma_529878215.swc` -- not in the repository (the
user's Drive / Colab OUT/soma_fit); pass their paths as arguments. Python 3 with the repository's `src/`.

| Script | Arguments | What it shows | Result on the pilot (run in the sandbox) |
|---|---|---|---|
| `depth_signal.py` | JSON | how much the planes change when only c, or only the far slope, changes | c 15 -> 7.26 um at constant mu c: 1.1-4.1 gl RMS per plane (median 2.2) over the 13 um disc; slope 0.367 -> 0.79: 4.8-5.8 gl |
| `tradeoff.py` | JSON | a compact soma imitating the fitted one by re-fitting z0 and the slope (about 9 min) | c 7.26: z0 28.39 um, slope 0.513, 2.40 gl RMS; c 10: z0 28.45, slope 0.490, 1.34 gl |
| `region_makeup.py` | JSON SWC | the fit region's composition (first rebuild, without the stems' inner stretch: superseded by `fig_region.py`) | 8610 px; 72 / 11 / 17 % (corrected: see `fig_region.py`) |
| `region_signal.py` | JSON SWC | the compact-versus-tall difference on the region's own pixels | c 7.26: 2.14 gl RMS; c 10: 1.18 gl |
| `sensitivity.py` | JSON SWC | linear response of c, z0 to background patterns (a bright wedge, a tilt) | at most 0.10 um of c per gl: not the cause |
| `fig_region.py` | JSON SWC OUT_PNG | figure: the region R by category (the stems' inner stretch restored from the JSON) | 7486 px (run 7644): 82 % outside the outline, 12 % edge, 6 % inside; 94 % of the outline masked |
| `region_new_check.py` | JSON SWC | the depth signal on the D-241 region (Allen's radius, branches only) against the old one | compact vs tall, plane brightness re-solved: 2.3 gl median (new) against 1.4 (old) |
| `explain_figures.py` | JSON CSV OUT_DIR | the explanation figures fig1 (soma area, depth not pinned) and fig2 (SGT kernels) | figures only |

The kernel-width checks of the same day are in `../sgt_width_2026-11-24/` (`fwhm_vs_sd.py`, `core_fit_proxy.py`).
