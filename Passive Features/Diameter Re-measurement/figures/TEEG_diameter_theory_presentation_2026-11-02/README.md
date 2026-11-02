# Figure pack: the theory of the diameter pipeline (2026-11-02)

| Date | Change |
|---|---|
| 2026-11-02 | v1, with the handoff `TEEG_diameter_theory_presentation_handoff_2026-11-02.md` (diameter theory chat). |

The handoff is the index. Its section 14 lists every figure with its id, topic, content, origin and caveats; its section 15 lists the figures still to make.

| Folder | Ids | What | In the repository |
|---|---|---|---|
| `figures/A_this_chat_optics/` | A1-A6 | Pupil, spot, defocus, Koehler and coherence, transfer functions, tube images by condenser setting, pupil-function cases | This folder (after the patch) |
| `figures/B_this_chat_theory/` | B1-B5 | T1 measuring-line geometry; T2 tube forward model; T5 SGT kernel family; the defocus-kernel law (configured, pilot, priors); T4.3 inversion (illustration) | This folder (after the patch) |
| `figures/R_repo_main/` (zip only) | R1-R14 | Copies of the repository's figures on `main` (sine-condition figures, microscope schematic, direction sines, ray world vs partition SVG, four interactive HTML figures) and `previews/` PNGs of the HTML and SVG | `Passive Features/Diameter Re-measurement/figures/` and `docs/` on `main` |
| `figures/S_superseded_redraw/` (zip only) | S1-S6 | The support-band figure (superseded coordinate) and the thin-plate-spline figures (superseded estimator): redraw or use as history only | `checks/table_2026-10-20/` and `docs/TEEG_thin_plate_spline_in_3D/figures/` on `sci/diameter-pipeline` |
| `scripts/` | -- | Physics/maths modules, figure scripts and checks (standalone, not pipeline code; UTF-8) | This folder |
| `checks/` | -- | Outputs of the checks, re-run on 2026-11-02 | This folder |

Regenerate (from `scripts/`):

```
python3 check_optics_claims.py && python3 make_figures.py && python3 make_pupil_figure.py   # A1-A6; writes here
python3 check_conjugate_planes.py; python3 check_diffraction.py
python3 check_theory_figures.py --repo <clone of Towards-EEG at sci/diameter-pipeline>     # 21 of 24 PASS
python3 make_theory_figures.py ../figures/B_this_chat_theory                                # B1-B5 (byte-identical rerun)
python3 render_previews.py <folder with the .html/.svg> <output folder>                     # needs Playwright
```

Check results:
- **Optics, `check_optics_claims.py`.** D3, C4 and C6 failed. They are the original claims, kept as they came out; D3b, C4b and C6b test the corrected statements and pass.
- **`check_conjugate_planes.py`.** W1 failed and W1b passes, on the same rule.
- **`check_theory_figures.py`.** P1, P2 and K3 failed and are kept; P1b, P2b and K3b test the diagnosed causes and pass:
  - P1: the reference convolution grid's h^1.5 error;
  - P2: cancellation in I1 - L1 at large alpha;
  - K3: denormal tail values.
