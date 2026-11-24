# The SGT's width rule, 2026-11-24 (the gallery chat)

Standalone checks behind D-242 (the SGT dropped): `python3 fwhm_vs_sd.py`, `python3 core_fit_proxy.py` (no inputs).

| Script | What it shows | Result |
|---|---|---|
| `fwhm_vs_sd.py` | FWHM per SD of SGT members; where the SD is infinite (xi >= beta/2) | SD-matched FWHM = 0.68x (beta 2, xi 0.5) to 1.46x (beta 10, xi 0) the law's; node 5353's shape 1.28x |
| `core_fit_proxy.py` | the calibration's Gaussian-core fit (Block 10) run on SGT blur: does 2.355 s equal the SGT's FWHM? | exact only for the Gaussian; 0.77-0.96 for node 5353's shape, 1.21-1.36 for beta 1, 0.55-0.94 for a flat top (FWHM 0.3-1.5 um) |
