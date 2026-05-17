# `phase3_publication_plots_v2.py` — Technical Reference

**Module purpose.** Produce the six publication-quality figures that
summarise a single-cell Phase-3 passive-parameter fit, reading
exclusively from a v4 `ReplotBundle` (the lightweight pickle produced
by `phase3_persistence_v4.save_replot_bundle`). The module is a
drop-in replacement for the v1 `phase3_publication_plots` and
preserves its public API; four of the six functions have, however,
been **re-defined to compute statistically distinct quantities** from
what the v1 names suggest, because the v1 versions required objects
that v4 does not persist. Each redefinition is documented inline and
in §4 below.

The module has no NEURON, no `skopt`, no `sklearn` dependency at
import time — just `numpy`, `scipy`, `matplotlib`, and (optional)
`plotly`. It is therefore safe to import in headless replot
notebooks, on cluster nodes without NEURON, and in CI.

---

## 1 · Bundle contract

Every function reads the same minimal set of fields. Inputs are
validated by attribute access, not by isinstance, so any duck-typed
object exposing these attributes will work — useful for unit tests
with synthetic bundles.

| Attribute                                   | Type                              | Used by                                                                               |
| ------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------- |
| `bundle.specimen_id`                        | `int`                             | titles, output paths                                                                  |
| `bundle.bootstrap_result.samples_log`       | `(N, 3) ndarray`                  | corner, 3D cloud, RMSD heatmap, density heatmap, 3D loss surface                      |
| `bundle.bootstrap_result.samples`           | `(N, 3) ndarray` (physical units) | scatter overlays                                                                      |
| `bundle.bootstrap_result.rmsds`             | `(N,) ndarray`                    | RMSD heatmap, 3D loss surface                                                         |
| `bundle.bootstrap_result.mle_log`           | `(3,) sequence`                   | every plot                                                                            |
| `bundle.bootstrap_result.mle_physical`      | `(3,) sequence`                   | every plot                                                                            |
| `bundle.bootstrap_result.mle_rmsd`          | `float`                           | 3D loss surface (optional; `NaN` is tolerated)                                        |
| `bundle.bootstrap_result.cov_log`           | `(3, 3) ndarray`                  | Gaussian-ellipse overlays                                                             |
| `bundle.bootstrap_result.ci_percentile`     | `dict[str, (float, float)]`       | every plot's CI overlay                                                               |
| `bundle.bootstrap_result.ci_bca`            | `dict[str, (float, float)]`       | every plot's CI overlay                                                               |
| `bundle.bootstrap_result.ci_normal`         | `dict[str, (float, float)]`       | every plot's CI overlay                                                               |
| `bundle.bootstrap_result.B_requested`       | `int`                             | corner subtitle                                                                       |
| `bundle.bootstrap_result.n_kept`            | `int`                             | corner subtitle, scatter legend                                                       |
| `bundle.bootstrap_result.bootstrap_mode`    | `str`                             | corner subtitle                                                                       |
| `bundle.gp_diagnostic_result.per_parameter` | `dict[str, _PerParamProfile]`     | 1-D GP profile                                                                        |
| &nbsp;&nbsp;`.grid_physical`                | `(M,) ndarray`                    | x-axis of profile                                                                     |
| &nbsp;&nbsp;`.profile_mean`                 | `(M,) ndarray`                    | µ_GP curve                                                                            |
| &nbsp;&nbsp;`.profile_std`                  | `(M,) ndarray`                    | σ_GP envelope                                                                         |
| &nbsp;&nbsp;`.mle_value`                    | `float`                           | MLE vertical line                                                                     |
| &nbsp;&nbsp;`.gp_trustworthy`               | `bool`                            | footer-note flag                                                                      |
| `bundle.skopt_dimensions`                   | `list[_SkoptDimLite]`             | heatmap log-space bounds (looked up by `.name`, not by index — order does not matter) |

**What is deliberately not in the bundle.** The live `skopt`
`OptimizeResult`, including its trained `GaussianProcessRegressor` and
the associated `Space` transformer, are **not** persisted. They held a
closure over the live NEURON cell at fit time, and pickling that
closure was the root cause of the 43 MB v3 corruption bug.
Consequently, `gp.predict(...)` cannot be evaluated at replot time —
which is why four of the six plots have been redefined (§4).

---

## 2 · Public API

```python
plot_3d_bootstrap_cloud(bundle, **opts)        # plotly,  HTML
plot_corner(bundle, **opts)                    # mpl,     PDF + PNG
plot_gp_heatmap_pair(bundle, p_x, p_y, **opts) # mpl,     PDF + PNG
plot_gp_sigma_heatmap_pair(bundle, p_x, p_y, **opts)   # mpl,     PDF + PNG
plot_gp_profile_family(bundle, parameter, **opts)      # mpl,     PDF + PNG
plot_3d_loss_surface(bundle, p_x, p_y, **opts) # plotly,  HTML

render_all_publication_figures(bundle, **opts) # orchestrator
```

All `mpl` functions return `(fig, axes)`; all `plotly` functions
return a `go.Figure`. Saving is controlled by `out_dir` (every
function) plus `save_pdf` / `save_png` / `save_html` (plot-type
dependent).

---

## 3 · Theme system

Three pre-defined themes live in the module-level `THEMES` dict:

| Name            | bg        | fg        | Use case                                                                                                   |
| --------------- | --------- | --------- | ---------------------------------------------------------------------------------------------------------- |
| `"light"`       | white     | dark-grey | print figures, journal submissions                                                                         |
| `"dark"`        | near-black | near-white | beamer/keynote slides on a dark deck                                                                       |
| `"colourblind"` | white     | dark-grey | colour-vision-deficient-safe palette (Wong 2011 / Okabe-Ito); also use this for the canonical thesis figs |

The themes encode every colour the renderer uses: background, foreground,
grid, scatter, MLE marker, heatmap colormaps (mpl and plotly variants), CI
colours per kind, ellipse colour, GP-mean line colour, surface colormap,
text size, and title size. A custom theme is built by instantiating
`Theme(...)` directly and passing the instance into `theme=...` instead of
a string.

`_apply_mpl_theme(fig, theme)` is called at the end of each mpl plot to
recolour every spine, axis, label, and tick. It is idempotent.

---

## 4 · Function-by-function reference

The six figures are listed here in the order in which
`render_all_publication_figures` produces them. Each entry states (a)
what the figure shows, (b) the relevant maths, and (c) whether the v2
implementation matches v1.

### 4.1 `plot_3d_bootstrap_cloud` — *unchanged from v1*

Interactive Plotly 3D scatter of the bootstrap cloud in
`(log Cm, log Rm, log Ra)`-space, with:

- **Axis-wall projections** — the cloud projected onto each of the three
  bounding walls of the scene, plotted at lower opacity so the eye can
  triangulate the 3D position of each point. Controlled by
  `projection_mode ∈ {"a", "b", "c", "none"}` where `"a"` = scatter
  shadows only, `"b"` = ellipse + CI shadows only, `"c"` = both, `"none"`
  = neither.
- **2D Gaussian confidence ellipses** on each wall, computed from
  `bootstrap_result.cov_log` evaluated at the MLE; level controlled by
  `contour_level` (default 0.95, i.e. χ²(2) at 95 %).
- **CI rectangles** on each wall, built from the cross-product of
  one-dimensional bootstrap CIs (`ci_kind="percentile"` by default;
  switch to `"bca"` or `"normal"` as needed).

Same implementation as v1 — this function never touched the live GP.

### 4.2 `plot_corner` — *unchanged from v1, but visually polished in this version*

Standard corner / triangle plot in physical units, log-scaled axes:

- **Diagonal** — 1D Gaussian KDE of the marginal bootstrap distribution
  for each parameter, with MLE and the requested CIs overlaid as
  vertical lines.
- **Lower triangle** — 2D Gaussian KDE of the corresponding parameter
  pair, scatter of the bootstrap samples, MLE star, Gaussian
  confidence ellipse (at `ellipse_level`), and CI cross-hair lines.

**Publication polish applied in this version:**

1. **Single figure-level legend** placed in the empty upper-right
   triangle. Per-panel `ax.legend()` calls have been removed entirely;
   the proxy artists list captures MLE marker, 1D KDE colour,
   Gaussian-ellipse line, and one entry per requested CI kind.
2. **KDE grid extended to the Gaussian-ellipse extent**: in v1, if the
   Gaussian ellipse extended beyond the bootstrap convex hull (which
   happens whenever the bootstrap is narrow but cov_log is wider, e.g.
   for low B), the coloured KDE region terminated inside the ellipse,
   giving the false impression that the density vanished there.
   `_kde_2d_on_grid` now accepts an `extra_bounds_log` argument; the
   corner plot computes the ellipse first, derives its log-space
   bounding box, and passes it in. The KDE grid then covers
   `union(padded data range, ellipse bounding box)` so the colour
   field reaches the ellipse.
3. **Robust inner-tick-label suppression.** Inner panels — i.e.
   non-bottom-row x-ticks and non-leftmost-column y-ticks — must not
   draw their tick labels (they collide with the neighbouring panel
   on logarithmic axes). The v1 idiom `ax.set_yticklabels([])` is
   silently undone by matplotlib whenever a subsequent
   `tick_params(...)` call regenerates the labels on a log-scale axis,
   which happens inside `_apply_mpl_theme`. The v2 implementation runs
   a *final pass* over the panel grid **after** both
   `_apply_mpl_theme(fig, th)` and `fig.subplots_adjust(...)` have
   returned, calling
   `plt.setp(ax.get_xticklabels(), visible=False)` and the y-axis
   counterpart. Because no further mpl call touches the tick artists,
   the visibility setting survives into the rendered PDF/PNG.
4. **Two-line title.** The main title is now
   `"Bootstrap distribution of passive parameters — cell <id>"`; the
   B / kept / mode metadata moves to a small italic subcaption via
   `fig.text(0.5, 0.965, ...)`, which keeps the main title clean
   without losing the information.

The panel grid is `3 × 3` GridSpec with `hspace = wspace = 0.14`, which is
large enough to keep the tick marks from each panel out of its
neighbour's plotting area.

### 4.3 `plot_gp_heatmap_pair` — *redefined from v1*

| | v1 | v2 |
| --- | --- | --- |
| **Quantity shown** | μ_GP(p_x, p_y) with third parameter pinned at MLE | RMSD(p_x, p_y) interpolated from bootstrap refits, third parameter marginalised |
| **Source** | Live `gp.predict(grid)` | `scipy.interpolate.griddata(samples_log[:, [ix, iy]], rmsds, X, Y, method="linear")` |
| **Region of definition** | The full log-bounds rectangle | The convex hull of the bootstrap samples; outside it, NaN (rendered as the colormap's `bad` colour) |

**Why the change is faithful but not identical.** Both quantities are
visualisations of the loss surface on a 2D parameter pair, but they
condition differently on the third parameter. v1 *slices* (fixes the
third parameter); v2 *marginalises* (lets the third parameter take
the value it took in each bootstrap refit). For well-identified
problems where the three parameters are weakly correlated, the two
look very similar in the high-density region. For problems with
strong correlation along the third axis, they diverge — and v2 is the
honest visualisation given what v4 saved.

**Publication polish in v2.** Title is now a two-line publication
title with LaTeX math:

```
Loss surface on (C_m, R_a)  — cell <id>
RMSD interpolated from bootstrap (R_m marginalised)
```

The third-parameter name is computed dynamically from `PARAM_NAMES`
minus `{p_x, p_y}`, so the title is always correct.

### 4.4 `plot_gp_sigma_heatmap_pair` — *redefined from v1*

| | v1 | v2 |
| --- | --- | --- |
| **Quantity shown** | σ_GP(p_x, p_y), the GP's posterior std-dev of the loss surface | KDE of the bootstrap *samples* in (p_x, p_y) log-space — i.e. parameter density, not loss-surface uncertainty |
| **Source** | Live `gp.predict(grid, return_std=True)` | `scipy.stats.gaussian_kde(samples_log[:, [ix, iy]].T)` evaluated on a regular grid |

**Caveat.** σ_GP and the bootstrap-density KDE are conceptually
different. σ_GP measured "how unsure is the GP that the loss surface
takes value f at point x?". The bootstrap-density KDE measures "how
often did our re-estimator visit (p_x, p_y) under perturbation of the
data?". In the limit of asymptotic identifiability the two are linked
(both reflect the curvature of the loss surface at the MLE), but they
are not the same quantity, and you should describe this plot in a
caption as a parameter density, never as a loss-surface uncertainty.

**Publication polish in v2.** Title becomes:

```
Bootstrap parameter density on (C_m, R_a)  — cell <id>
2D Gaussian KDE in log-space (R_m marginalised)
```

### 4.5 `plot_gp_profile_family` — *redefined from v1*

| | v1 | v2 |
| --- | --- | --- |
| **Quantity shown** | A family of µ_GP(`parameter`) curves, one per quantile of `family_param` | The single 1D GP profile that v4 persists in `bundle.gp_diagnostic_result.per_parameter[parameter]`, with the matching σ-envelope |
| **Source** | Live `gp.predict` along a 1D slice for each pinned `family_param` | `pp.grid_physical`, `pp.profile_mean ± k · pp.profile_std` |

The v1 *family* of curves cannot be reconstructed: every member of
the family required re-querying the live GP at a different pinned
location, and v4 does not save the GP. The v2 implementation accepts
the v1 keywords `family_param` and `percentiles` for signature
compatibility but emits a `UserWarning` when `family_param` is passed
(`percentiles` is silently ignored — it has no observable effect on
the output).

**Publication polish in v2 (this is what the user explicitly asked
for in the polish pass).**

1. **Bootstrap histogram overlay removed.** The v1 implementation
   used `ax.twinx()` to put a density histogram of the bootstrap
   `parameter` marginal under the profile. The histogram competed
   for the reader's attention without adding information that the
   corner plot does not already provide more clearly. The twin
   axis is gone. `show_bootstrap_hist` and `hist_bins` are accepted
   for signature compatibility but silently ignored (no warning —
   they are not a misuse, just deprecated).
2. **Title polished** to a single line:
   `"1D loss profile along <param> — cell <id>"`. The
   "GP ✓ / GP ✗ MISMATCH" internal diagnostic flag has been moved
   out of the title.
3. **Trust-flag footer.** When `pp.gp_trustworthy is False`, an
   italic footer annotation is added via `fig.text(0.5, 0.005, ...)`
   that reads:

   > *Note: precomputed GP profile disagreed with reference
   > simulations at validation — interpret envelope with caution.*

   When the flag is `True`, nothing is drawn — the figure stays
   clean.
4. **Legend cleaned**: `frameon=False`, top-right corner of the
   panel, no numeric MLE value in the legend label (just "MLE"),
   no `(precomputed)` annotation on the µ_GP entry. CI labels are
   deduplicated against `seen_ci` so each kind appears only once
   even though the lower and upper limits are drawn as separate
   `axvline` calls.

### 4.6 `plot_3d_loss_surface` — *redefined from v1*

| | v1 | v2 |
| --- | --- | --- |
| **Quantity shown** | Continuous GP-mean surface over (p_x, p_y) with bootstrap cloud hovering at their refit RMSDs | Only the bootstrap cloud, plotted in 3D with z = stored refit RMSD |
| **Source** | Live `gp.predict(meshgrid)` | `bundle.bootstrap_result.rmsds` |

No continuous fit is implied or drawn. The MLE is added as a diamond
at z = `bundle.bootstrap_result.mle_rmsd` if that field is finite.
The camera defaults to `eye=(1.6, -1.6, 1.0)`, which gives a clear
view of the surface from below and to the right.

---

## 5 · Orchestrator: `render_all_publication_figures`

```python
render_all_publication_figures(
    bundle, *,
    root_dir=None,                  # base output directory
    themes=("light",),              # one or more theme names
    show_ci=("percentile",),        # which CIs to overlay
    projection_mode="c",            # 3D cloud projection style
    verbose=True,                   # print one line per saved file
) -> Path                           # returns the per-cell base dir
```

Output layout under the returned base:

```
<root_dir>/cell_<id>/publication/
  ├── <theme1>/
  │   ├── corner.pdf
  │   ├── corner.png
  │   ├── 3d_bootstrap_cloud.html
  │   ├── gp_mean_heatmap_Cm_Rm.{pdf,png}
  │   ├── gp_mean_heatmap_Cm_Ra.{pdf,png}
  │   ├── gp_mean_heatmap_Rm_Ra.{pdf,png}
  │   ├── gp_sigma_heatmap_Cm_Rm.{pdf,png}
  │   ├── gp_sigma_heatmap_Cm_Ra.{pdf,png}
  │   ├── gp_sigma_heatmap_Rm_Ra.{pdf,png}
  │   ├── profile_Cm.{pdf,png}
  │   ├── profile_Rm.{pdf,png}
  │   ├── profile_Ra.{pdf,png}
  │   ├── 3d_loss_surface_Cm_Rm.html
  │   ├── 3d_loss_surface_Cm_Ra.html
  │   └── 3d_loss_surface_Rm_Ra.html
  └── <theme2>/   (same structure)
```

The function closes each mpl figure with `plt.close(fig)` immediately
after writing it to disk, so 18 figures × N themes does not blow up
memory.

---

## 6 · Usage

### 6.1 Single cell, single theme, single figure

```python
from passive_result_plot import load_replot_bundle
import phase3_publication_plots_v2 as ppub

bundle = load_replot_bundle(
    "/.../Passive_opt_out/cell_531520401/replot"
)

# Just the corner plot, saved to disk:
fig, _ = ppub.plot_corner(
    bundle,
    show_ci=("percentile", "bca"),
    out_dir="./out",
    filename_base="corner",
)
```

### 6.2 Full publication suite for one cell, multiple themes

```python
ppub.render_all_publication_figures(
    bundle,
    root_dir="/.../publication_output",
    themes=("light", "colourblind"),
    show_ci=("percentile", "bca"),
)
```

### 6.3 Loop over every cell

```python
from pathlib import Path

for cell_dir in sorted(Path("/.../Passive_opt_out").glob("cell_*")):
    rep = cell_dir / "replot"
    if not (rep / "core.pkl").exists():
        continue
    bundle = load_replot_bundle(rep)
    ppub.render_all_publication_figures(
        bundle,
        root_dir="/.../publication_output",
        themes=("light",),
    )
```

### 6.4 Embedding a heatmap into an existing figure

All `plot_gp_*` functions accept an `ax=...` argument. Pass a pre-made
matplotlib `Axes` and the function will render into it instead of
creating its own figure:

```python
fig, ax = plt.subplots(figsize=(7, 5))
ppub.plot_gp_heatmap_pair(bundle, "Cm", "Ra", ax=ax)
fig.savefig("custom_layout.pdf")
```

---

## 7 · Statistical caveats — read before pasting in a thesis

The four redefined plots (§§4.3 – 4.6) show statistically distinct
quantities from their v1 namesakes. The differences matter at the
caption-writing stage:

1. **`plot_gp_heatmap_pair` is a marginal RMSD map, not a slice.**
   The third parameter is not pinned. For any thesis caption, write
   *"refit RMSD interpolated from bootstrap, with the third parameter
   marginalised over its bootstrap distribution"*, **not** *"GP-mean
   loss surface at fixed Rm"*.

2. **`plot_gp_sigma_heatmap_pair` is a parameter density, not a
   loss-surface uncertainty.** Write *"2D KDE of the bootstrap
   parameter samples in log-space"*, **not** *"GP posterior
   standard deviation"*.

3. **`plot_gp_profile_family` shows one curve, not a family.** The
   family of curves cannot be reconstructed from v4 data. Write
   *"1D loss profile along Cm with the GP posterior σ-envelope,
   evaluated from the precomputed profile saved at fit time"*.

4. **`plot_3d_loss_surface` has no continuous surface.** Write *"3D
   scatter of bootstrap refits, coloured by RMSD"*, **not** *"GP-mean
   loss surface"*.

If you need the exact v1 quantities for any of these plots, the
correct fix is to extend `phase3_persistence_v4` (i.e. `_v5`) to
pre-compute the GP slices, the GP σ-slices, the family of GP
profiles, and the GP surface, and to persist them as plain numpy
arrays in `bundle.extras["gp_surfaces"]`. The replot functions can
then read these arrays instead of querying a live GP. The cost is one
re-run of Phase 3 per cell.

---

## 8 · Dependencies

| Package      | Minimum tested version | Why                                                        |
| ------------ | ---------------------- | ---------------------------------------------------------- |
| `numpy`      | 1.24                   | Everything; `np.meshgrid`, `np.linalg.cholesky`            |
| `scipy`      | 1.10                   | `gaussian_kde`, `chi2`, `interpolate.griddata`             |
| `matplotlib` | 3.7                    | `GridSpec`, `mpl.colormaps.get_cmap(...).copy()` (3.7 API) |
| `plotly`     | 5.18 (optional)        | 3D interactive plots; module degrades gracefully without it |

There is no NEURON, no `skopt`, no `sklearn` dependency.
`matplotlib.colormaps.get_cmap(...)` is the post-3.7 API; if you are
stuck on matplotlib 3.5–3.6 you will see a deprecation warning but
the code still runs.

---

## 9 · Known limitations

1. **B is too small.** With B ≤ 30 bootstrap replicates, the
   `griddata("linear", ...)` interpolation in
   `_bootstrap_rmsd_slice` has too few scattered points and produces
   visibly tessellated triangles. Either raise B in Phase 3, or set
   `interp_method="cubic"` in the call. With B ≥ 200 the surface is
   smooth.

2. **`gp_trustworthy = False` cells.** These cells passed Phase-3
   fitting but the precomputed GP profile failed validation against
   reference NEURON simulations. The σ-envelope on
   `plot_gp_profile_family` should not be reported as a quantitative
   uncertainty for these cells. The footer note in the figure makes
   this visible at-a-glance.

3. **CI dictionaries may be partial.** When a particular CI kind
   (`percentile`, `bca`, `normal`) failed to compute for a given
   parameter, that entry is silently absent from the dict and is
   skipped at draw time. There is no warning. This is intentional —
   the v1 code did the same — but means a missing CI line in the
   output is not necessarily a bug.
