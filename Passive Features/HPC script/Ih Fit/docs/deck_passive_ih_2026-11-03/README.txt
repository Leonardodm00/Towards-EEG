docs/deck_passive_ih_2026-11-03 -- figures for the deck on the passive + I_h optimisation (2026-11-03)
====================================================================================================

Read first: ../TEEG_deck_handoff_passive_ih_2026-11-03.md (section 8 indexes these figures: the one claim each
shows, whether it is plotted or schematic, and the source of what it draws).

figures/   nine PNGs, 3840 x 2160 (a 1920 x 1080 slide at 2x): fig01_pipeline_status, fig02_roadmap_board,
           fig03_model_parameters, fig04_ih_kinetics, fig05_fit_loop, fig06_passive_results, fig07_noise_law,
           fig08_noise_fit, fig09_recovery_gate
scripts/   make_figures.py (v2): builds all nine as HTML/CSS slides, charts drawn as SVG from the numbers in the
           script and from the Ih Fit folder two levels up (human_ih_params.py, noise_law.py); renders them with
           Playwright's Chromium. Needs numpy, playwright (Python, with its Chromium) and the Inter font.

    cd scripts
    python make_figures.py --status <output of: python specs/roadmap/roadmap.py status --yaml
                                     specs/roadmap/allen-mtg-fit.yaml --json <file>>
    python make_figures.py --status <file> --only 4,7        # some of them

The roadmap figures (fig01, fig02, the pills of fig08) show the state at commit 854004c (2026-11-02); regenerate
them before a talk.

History: v1 (2026-11-03) used matplotlib for fig02-04, 06, 07 and Mermaid (*.mmd) for fig01, 05, 08, 09; v2
(2026-11-03, later) is the redesign the user asked for, with the same claims; the Mermaid sources were removed.
