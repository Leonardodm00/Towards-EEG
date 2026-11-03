docs/deck_passive_ih_2026-11-03 -- figures for the deck on the passive + I_h optimisation (2026-11-03)
====================================================================================================

Read first: ../TEEG_deck_handoff_passive_ih_2026-11-03.md (section 8 indexes these figures: the one claim each
shows, whether it is plotted or schematic, and the source of what it draws).

figures/   nine PNGs: fig01_pipeline_status, fig02_roadmap_board, fig03_model_parameters, fig04_ih_kinetics,
           fig05_fit_loop, fig06_passive_results, fig07_noise_law, fig08_noise_fit, fig09_recovery_gate
scripts/   make_figures.py (fig02, fig03, fig04, fig06, fig07; numpy + matplotlib + the Ih Fit folder two levels up)
           and the Mermaid sources of fig01, fig05, fig08, fig09 (*.mmd, with _style.mmd and _mermaid.css)

    cd scripts
    python make_figures.py --status <output of: python specs/roadmap/roadmap.py status --yaml
                                     specs/roadmap/allen-mtg-fit.yaml --json <file>>
    mmdc -i fig01_pipeline_status.mmd -o ../figures/fig01_pipeline_status.png -w 2000 -b white -s 1.5 -C _mermaid.css

The roadmap figures show the state at commit 854004c (2026-11-02); regenerate them before a talk.
