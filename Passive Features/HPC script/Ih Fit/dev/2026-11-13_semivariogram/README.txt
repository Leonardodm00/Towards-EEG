TEEG -- the semivariogram, a brief study document (2026-11-13) -- standalone, not pipeline code

Written by the noise-fit chat when the user asked for "a brief document about the semivariogram". The
document is TEEG_semivariogram_2026-11-13.md (topic-document rules: notation, glossary, body, references;
every claim tagged with its source, key claims rated Certain / Likely / Guessing). Nothing here is
imported by the Ih Fit code or runs in a job; the scripts import the repository's noise_law (Block 2)
for predictions only (environment variable TEEG_IH_FIT, else the Ih Fit folder three levels above).

What it covers: the definition and its relation to the autocovariance; nugget, sill and range; the
shapes of white noise, an OU, a random walk and a random tilt; the classical estimator against the
autocovariance with the record's own mean; the semivariogram of window means (the floor at the SS lags)
and the Allan variance as a special case; how D-034 read the per-sweep tilt from the SS window means,
and where that stands since D-102 / D-107 (the pipeline computes no semivariogram; roadmap T8 open).
The tilt SD's precision and pooling are in ../2026-10-26_tilt_sigma_b/ (the tilt study).

Run, from scripts/ (numpy, scipy, matplotlib):
    python check01_generic.py                 (~3 s)   -> ../results/check01_generic.log (by tee)
    python check02_window_means.py            (~9 s, ~0.5 GB) -> ../results/check02_window_means.json
    python fig01_shapes.py                    (~2 s)   -> ../figures/fig01_shapes.png
    python fig02_window_mean_semivariogram.py (~5 s)   -> ../figures/fig02_window_mean_semivariogram.png
                                                          (reads check02's json: run check02 first)

| script | output | what it shows |
|---|---|---|
| check01_generic.py | results/check01_generic.log | 25/25 PASS: white noise = nugget; OU + nugget = eta2 + sigma2 (1 - e^-h/tau); random walk q h / 2; random slope eta2 + sigma_b^2 h^2 / 2; offsets cancel (5e-16); the own-mean autocovariance's exact expectation (= noise_law.expected_sl, 9e-15) and its bias on short records; the random walk's sample autocovariance grows with record length, the semivariogram does not (seed 20261113) |
| check02_window_means.py | results/check02_window_means.{log,json} | 16/16 PASS at the SS layout and theta_cf: the estimate over same-sweep, same-polarity pairs = noise_law.semivariogram at all 18 even lags, with and without a tilt; the floor flat to 9.3e-5 (= Var of one 120 ms window mean, 1.1125e-3 mV^2); the tilt term exact; (D-034.1) reads the realised tilts' mean square; Allan variance = V(h = T); OU excess slopes vs the tilt's 2; the averaging shift h / (h - T/3) (seed 20261114) |
| fig01_shapes.py | figures/fig01_shapes.png, results/fig01_shapes.log | the four shapes, formulas against simulated estimates (32 dots within 1.9 SE; seed 20261115) |
| fig02_window_mean_semivariogram.py | figures/fig02_window_mean_semivariogram.png, results/fig02_window_mean_semivariogram.log | the window-mean semivariogram at the SS lags: floor, + tilt, + OU 3 s, + OU 30 s; the excess log-log |
| viz_style.py | -- | shared plotting style (dataviz reference palette) |

Pure ASCII, LF (the scripts).
