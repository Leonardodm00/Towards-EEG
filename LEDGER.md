# Towards-EEG -- S0.0 Reconciliation Ledger

**Stage:** S0.0 (roadmap rev 3, section 3.2)  
**Generated:** 2026-07-21 12:56:54 UTC by `s0_ledger` v1.0.0  
**Snapshot:** `github.com/Leonardodm00/Towards-EEG@main`, 2026-07-19  
**Tree root:** `<repository root>`  
**Rows:** 205 (201 repository files + 4 local working files)

This ledger is the evidence base for the byte-identity rule (handoff
brief section 3). Every file in the reorganised tree must be (1) byte-identical
to an identified ancestor, (2) the output of a stated reproducible mechanical
transform of one, or (3) new infrastructure nothing pre-existing depends on.
Each row below records which, with the hash that proves it.

Regenerate with:

```bash
python3 tools/build_ledger.py --root . --spec tools/ancestors.json \
        --local-dir <dir containing the four working files> --out-dir .
```

## 0. Exit test

**PASS** -- every row carries a final verdict; every declared
relationship agrees with the measured hashes.

## 1. Summary

### By verdict

| verdict      | files  |
|--------------|--------|
| discard      | 22     |
| keep         | 13     |
| new          | 58     |
| retain       | 112    |

### By scope

| scope          | files  |
|----------------|--------|
| colab          | 60     |
| infrastructure | 58     |
| orchestrator   | 18     |
| passive        | 69     |

### By acting sub-step

| stage    | files  |
|----------|--------|
| -        | 112    |
| S0.0     | 11     |
| S0.1     | 4      |
| S0.2     | 11     |
| S0.2c    | 3      |
| S0.3     | 9      |
| S0.4     | 21     |
| S0.5     | 12     |
| S0.7     | 22     |

### Python surface at pre-s0

- python files: **114**
- failing `ast.parse`: **0**
- containing bytes >= 0x80: **0**
- non-ASCII and no PEP 263 cookie: **0**
- containing CRLF: **0**
- raising SyntaxWarning (latent, e.g. invalid escape sequences): **6**

## 2. Working-branch reconciliation (S0.1)

The four local files are committed BEFORE the `pre-s0` tag, so that
ancestry is verifiable by `git` rather than by assertion. Relationship
`working_branch_edit` marks a hand edit that is NOT a mechanical
transform; it is admissible only because the commit makes the file its
own ancestor for every later sub-step.

| file                                   | relationship         | T   | ancestor                                             | sha256(pre-s0) |
|----------------------------------------|----------------------|-----|------------------------------------------------------|----------------|
| Connectomics/Calculate Relative Subpop pct/Usage_example.py | mechanical           | T6  | Connectomics/Calculate Relative Subpop % /Usage_example.py | 7f95739f273f   |
| Connectomics/Calculate Relative Subpop pct/calculate_bbp_relative_presences.py | mechanical           | T6  | Connectomics/Calculate Relative Subpop % /calculate_bbp_relative_presences.py | d10121eb3f7e   |
| Utility_fun.py                         | identical            | T0  | HybridLFPy Tweaked/Utility_function.py               | 9d3d29c8a997   |
| hybrid_sim_evoked_with_EEG_CHANGED.py  | identical            | T0  | HybridLFPy Tweaked/hybrid_sim_evoked_with_EEG_multiMorph.py | 6c3ed490ba14   |
| params_evoked_with_EEG_CHANGED.py      | identical            | T0  | HybridLFPy Tweaked/params_evoked_with_EEG_multiMorph.py | 927ee3d9cc0d   |
| population_CHANGED.py                  | identical            | T0  | HybridLFPy Tweaked/Population_multiMorph.py          | 43df1344230e   |
| towards_eeg/connectome/connectivity_buildup.py | mechanical           | T6  | Classes/Connectomics/connectivity_buildup (9).py     | e62667fa6e57   |
| towards_eeg/connectome/connectomics.py | mechanical           | T6  | Classes/Connectomics/Class.py                        | 2f58a6ae79a3   |
| towards_eeg/connectome/macropopulation.py | mechanical           | T6  | HybridLFPy Tweaked/Extract macro-population data/Extract_macropopulation.py | 0bfb98e308b9   |
| towards_eeg/connectome/usage_example.py | mechanical           | T6  | HybridLFPy Tweaked/Extract macro-population data/Usage_example.py | dc4828bb65d4   |
| towards_eeg/hybrid/README.md           | mechanical           | T6  | HybridLFPy Tweaked/README.md                         | 8183e1b9cd57   |
| towards_eeg/hybrid/driver.py           | mechanical           | T6  | HybridLFPy Tweaked/hybrid_sim_evoked_with_EEG_multiMorph.py | 566b25332991   |
| towards_eeg/hybrid/params.py           | mechanical           | T6  | HybridLFPy Tweaked/params_evoked_with_EEG_multiMorph.py | 927ee3d9cc0d   |
| towards_eeg/hybrid/population.py       | mechanical           | T6  | HybridLFPy Tweaked/Population_multiMorph.py          | 35dbec718a35   |
| towards_eeg/hybrid/utility.py          | mechanical           | T6  | HybridLFPy Tweaked/Utility_function.py               | ec55afe6c0c1   |

- **Connectomics/Calculate Relative Subpop pct/Usage_example.py** -- colab scope; stays in the repository, outside the installed package; moved by T6 (git mv), bytes unchanged
- **Connectomics/Calculate Relative Subpop pct/calculate_bbp_relative_presences.py** -- colab scope; stays in the repository, outside the installed package; moved by T6 (git mv), bytes unchanged
- **Utility_fun.py** -- TEEG_01 rev 2 section 3.3: ancestor plus three new functions (insert_mechanisms, get_gIhbar_L5_apical, get_gCa_HVA_apical); keep. Carries defects U1-U4, deferred to S3; bytes equal
- **hybrid_sim_evoked_with_EEG_CHANGED.py** -- TEEG_01 rev 2 section 3.3: identical modulo a trailing newline; keep; bytes equal
- **params_evoked_with_EEG_CHANGED.py** -- TEEG_01 rev 2 section 3.3: byte-identical to the snapshot; keep; bytes equal
- **population_CHANGED.py** -- TEEG_01 rev 2 section 3.3: ancestor plus 7 lines, an empty tonic-inhibition placeholder comment in cellsim; keep. Not a mechanical transform; legitimate because S0.1 commits it, making it the ancestor for everything downstream; bytes equal
- **towards_eeg/connectome/connectivity_buildup.py** -- orchestrator source; enters the installed package; moved by T6 (git mv), bytes unchanged
- **towards_eeg/connectome/connectomics.py** -- orchestrator source; enters the installed package; moved by T6 (git mv), bytes unchanged
- **towards_eeg/connectome/macropopulation.py** -- orchestrator source; enters the installed package; moved by T6 (git mv), bytes unchanged
- **towards_eeg/connectome/usage_example.py** -- orchestrator source; enters the installed package; moved by T6 (git mv), bytes unchanged
- **towards_eeg/hybrid/README.md** -- orchestrator source; enters the installed package; moved by T6 (git mv), bytes unchanged
- **towards_eeg/hybrid/driver.py** -- ancestor of a working-branch file; superseded at S0.1, moved at S0.4; moved by T6 (git mv), bytes unchanged
- **towards_eeg/hybrid/params.py** -- ancestor of a working-branch file; superseded at S0.1, moved at S0.4; moved by T6 (git mv), bytes unchanged
- **towards_eeg/hybrid/population.py** -- ancestor of a working-branch file; superseded at S0.1, moved at S0.4; moved by T6 (git mv), bytes unchanged
- **towards_eeg/hybrid/utility.py** -- ancestor of a working-branch file; superseded at S0.1, moved at S0.4; moved by T6 (git mv), bytes unchanged

## 3. Discards

22 files. Payload discards move to the separate repository per D-8;
no history is rewritten.

| path                                                           | stage  | sha256(pre-s0) | bytes      |
|----------------------------------------------------------------|--------|----------------|------------|
| Connectomics/Construct_ADJ/Microcircuit Assembly Documentation.pdf | S0.7   | dec69a3bdff9   | 101049     |
| Connectomics/Generate Point Microcolumn/Population Generation Documentation (1).pdf | S0.7   | be7956c4b022   | 103270     |
| Morphology Compliance/Morphology Selection Documentation.pdf   | S0.7   | 2acac69e5047   | 86184      |
| Nuova cartella compressa.zip                                   | S0.7   | 9da5198cce44   | 7712035    |
| Spanning Trees/Neuronal Mapping Documentation.pdf              | S0.7   | 14240ebdd440   | 53832      |
| Spanning tree overlap and synapse identification/Peters Rule Connectivity Engine Documentation.pdf | S0.7   | a44a1660a019   | 83827      |
| Spanning tree overlap and synapse identification/Synaptic Overlap Debugger Documentation.pdf | S0.7   | 7a7a3b5a8d8a   | 68572      |
| Synaptic Placement/Neural Simulation Documentation.pdf         | S0.7   | d5300eb50e13   | 43562      |
| Classes/Connectomics/Connectomics Class Overview.pdf           | S0.7   | 3cea1baed2e4   | 73846      |
| HybridLFPy Tweaked/Extract macro-population data/Project README.pdf | S0.7   | 95a6e04fd1bd   | 54432      |
| Passive Features/HPC script/passive_fit.e1099100               | S0.7   | 856f6afb4ea7   | 86458      |
| Passive Features/HPC script/passive_fit.e1099156               | S0.7   | 3c03fe908847   | 5271       |
| Passive Features/HPC script/passive_fit.e1099215               | S0.7   | 87501536b336   | 19419      |
| Passive Features/HPC script/passive_fit.o1099100               | S0.7   | 2c2f8bc9b1e7   | 16377      |
| Passive Features/HPC script/passive_fit.o1099156               | S0.7   | ef17cc6efc8e   | 2321       |
| Passive Features/HPC script/passive_fit.o1099215               | S0.7   | c69560156a4e   | 10497      |
| Passive Features/Passive Allen Data/L2/specimen_528706755.zip  | S0.7   | 579910a148da   | 16782700   |
| Passive Features/Passive Allen Data/L2/specimen_537204107.zip  | S0.7   | 01bff4f88db8   | 25756947   |
| Passive Features/Passive Allen Data/L2/specimen_614659629.zip  | S0.7   | e8ceaafe3743   | 19472705   |
| Passive Features/Passive Allen Data/L2/specimen_616647103.zip  | S0.7   | f4219ebd7149   | 7331096    |
| Passive Features/Phase 2/phase2_patch.pdf                      | S0.7   | 2d488e5e1300   | 1355690    |
| Passive Features/Phase 2/phase2_technical.pdf                  | S0.7   | 03d8016e1289   | 4049654    |

## 4. Defect O7 register -- shadowed module-level definitions

At module scope the LAST definition of a name wins, unconditionally. In
a notebook the winner depends on cell execution order, which version
control does not record. Once these files are renamed and moved at S0.4
this table is the only surviving record of which definition ran.

**114 shadowed pairs across 11 files.**

| file                                                     | name                               | defined at       | effective  | dead         |
|----------------------------------------------------------|------------------------------------|------------------|------------|--------------|
| Dendrite F score/morpholgy_pathways (6).py               | calculate_dendrite_xy_anisotropy   | L25, L376        | L376       | L25          |
| Dendrite F score/morpholgy_pathways (6).py               | export_neuron_to_hoc               | L842, L1331      | L1331      | L842         |
| Dendrite F score/morpholgy_pathways (6).py               | align_neurons_to_neighborhood      | L973, L1448      | L1448      | L973         |
| Dendrite F score/morpholgy_pathways (6).py               | label_dendritic_spines_robust      | L1653, L2353     | L2353      | L1653        |
| Dendrite F score/morpholgy_pathways (6).py               | plot_color_coded_neurons           | L1826, L2444     | L2444      | L1826        |
| Dendrite F score/morpholgy_pathways (6).py               | generate_smooth_tube               | L2573, L2783     | L2783      | L2573        |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _simulate_square_subthreshold      | L368, L7299, L12664 | L12664     | L368, L7299  |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | IncompleteDataError                | L1595, L7327, L12692 | L12692     | L1595, L7327 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | SweepBundle                        | L1613, L7343, L12708 | L12708     | L1613, L7343 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | CellData                           | L1632, L7362, L12727 | L12727     | L1632, L7362 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | PassiveSearchSpace                 | L1663, L7393, L12758 | L12758     | L1663, L7393 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | OptimiserInputs                    | L1712, L7442, L12807 | L12807     | L1712, L7442 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | list_human_cells_with_morphology   | L1727, L7456, L12821 | L12821     | L1727, L7456 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _to_pA_seconds                     | L1872, L7600, L12972 | L12972     | L1872, L7600 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _select_square_subthreshold        | L1913, L7641, L13013 | L13013     | L1913, L7641 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _select_long_square_subthreshold   | L1933, L7661, L13033 | L13033     | L1933, L7661 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _detect_step_amplitude             | L1949, L7677, L13049 | L13049     | L1949, L7677 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _detect_pulses_in_current          | L1980, L7708, L13080 | L13080     | L1980, L7708 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _extract_windows_around_pulses     | L2021, L7749, L13121 | L13121     | L2021, L7749 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _build_subthreshold_bundles        | L2063, L7790, L13162 | L13162     | L2063, L7790 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _build_bundles_from_group          | L2187, L7914, L13286 | L13286     | L2187, L7914 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | load_allen_data                    | L2250, L7976, L13348 | L13348     | L2250, L7976 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | PassiveCell                        | L2456, L8181, L13553 | L13553     | L2456, L8181 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | build_neuron_model                 | L2604, L8329, L13701 | L13701     | L2604, L8329 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | prepare_optimiser_inputs           | L2620, L8344, L13716 | L13716     | L2620, L8344 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | load_complete_cells                | L2727, L8450, L13822 | L13822     | L2727, L8450 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _NumpyEncoder                      | L3189, L8506, L13878 | L13878     | L3189, L8506 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | download_allen_archive             | L3211, L8524, L13896 | L13896     | L3211, L8524 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _count_swc_roots                   | L3401, L8710, L14164 | L14164     | L3401, L8710 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _archive_one_cell                  | L3438, L11325, L16806 | L16806     | L3438, L11325 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _reconstruct_cell_data_for_smoke_test | L3902, L8747, L14201 | L14201     | L3902, L8747 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | run_checks                         | L4471, L8951, L14405 | L14405     | L4471, L8951 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | RepairConfig                       | L4965, L8973, L14427 | L14427     | L4965, L8973 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | RepairStep                         | L4986, L8994, L14448 | L14448     | L4986, L8994 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | RepairDiagnostic                   | L4994, L9002, L14456 | L14456     | L4994, L9002 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | RepairResult                       | L5017, L9025, L14479 | L14479     | L5017, L9025 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | load_node_table                    | L5047, L9055, L14509 | L14509     | L5047, L9055 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | save_node_table                    | L5066, L9074, L14528 | L14528     | L5066, L9074 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _build_graph                       | L5077, L9085, L14539 | L14539     | L5077, L9085 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _path_length                       | L5090, L9098, L14552 | L14552     | L5090, L9098 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _preclean                          | L5104, L9112, L14566 | L14566     | L5104, L9112 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _normalize_soma                    | L5144, L9152, L14606 | L14606     | L5144, L9152 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _heal_components                   | L5204, L9212, L14666 | L14666     | L5204, L9212 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _reroot_table                      | L5285, L9293, L14747 | L14747     | L5285, L9293 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _fix_z_jumps                       | L5314, L9322, L14776 | L14776     | L5314, L9322 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _fix_soma_radius                   | L5340, L9348, L14802 | L14802     | L5340, L9348 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | diagnose                           | L5357, L9365, L14819 | L14819     | L5357, L9365 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | MorphologyRepair                   | L5422, L9430, L14884 | L14884     | L5422, L9430 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | repair_and_verify                  | L5489, L9497, L14951 | L14951     | L5489, L9497 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _seg_xyz                           | L5601, L10500, L15954 | L15954     | L5601, L10500 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | SomaGatedRepairConfig              | L6148, L9625, L15079 | L15079     | L6148, L9625 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | RepairOutcome                      | L6156, L9633, L15087 | L15087     | L6156, L9633 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _component_type                    | L6177, L9654, L15108 | L15108     | L6177, L9654 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _heal_stage1_soma_gated            | L6184, L9661, L15115 | L15115     | L6184, L9661 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _detect_soma_flagged               | L6258, L9735, L15189 | L15189     | L6258, L9735 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | repair_table_for_import            | L6268, L9745, L15199 | L15199     | L6268, L9745 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | repair_swc_file                    | L6302, L9779, L15233 | L15233     | L6302, L9779 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | import3d_swc_read_repaired         | L6317, L9794, L15248 | L15248     | L6317, L9794 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | make_repaired_build_fn             | L6342, L9819, L15273 | L15273     | L6342, L9819 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | load_allen_swc                     | L6378, L9855, L15309 | L15309     | L6378, L9855 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _segments_by_type                  | L6414, L9891, L15345 | L15345     | L6414, L9891 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _orphan_root_markers               | L6431, L9908, L15362 | L15362     | L6431, L9908 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | figure_raw_vs_repaired             | L6440, L9917, L15371 | L15371     | L6440, L9917 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | summarize_repair                   | L6517, L9994, L15448 | L15448     | L6517, L9994 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _summary_text                      | L6560, L10037, L15491 | L15491     | L6560, L10037 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | run                                | L6583, L10060, L15514 | L15514     | L6583, L10060 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | ElectricalViabilityConfig          | L6631, L10169, L15623 | L15623     | L6631, L10169 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | ElectricalViabilityResult          | L6658, L10225, L15679 | L15679     | L6658, L10225 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | electrical_viability_check         | L6850, L10607, L16061 | L16061     | L6850, L10607 |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | count_swc_roots                    | L10284, L15738   | L15738     | L10284       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | classify_reached                   | L10314, L15768   | L15768     | L10314       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | is_exploding                       | L10325, L15779   | L15779     | L10325       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | attenuation_spearman               | L10332, L15786   | L15786     | L10332       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | fit_lambda_um                      | L10354, L15808   | L15808     | L10354       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | fit_tau_ms                         | L10375, L15829   | L15829     | L10375       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _get_h                             | L10399, L15853   | L15853     | L10399       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _clear_neuron                      | L10411, L15865   | L15865     | L10411       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _categorise                        | L10418, L15872   | L15872     | L10418       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _default_build                     | L10435, L15889   | L15889     | L10435       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _apply_dlambda                     | L10466, L15920   | L15920     | L10466       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _lambda_f_manual                   | L10483, L15937   | L15937     | L10483       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _is_terminal                       | L10515, L15969   | L15969     | L10515       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _select_sites                      | L10522, L15976   | L15976     | L10522       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _simulate_soma_step                | L10541, L15995   | L15995     | L10541       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _make_phase0_repair_cfg            | L10826, L16280   | L16280     | L10826       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _make_phase0_ev_cfg                | L10839, L16293   | L16293     | L10839       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | plot_electrical_triage             | L10856, L16310   | L16310     | L10856       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | plot_membrane_traces               | L10925, L16379   | L16379     | L10925       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _run_static_triage_if_available    | L11076, L16530   | L16530     | L11076       |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py | _phase0_archive_swc                | L11132, L16586   | L16586     | L11132       |
| Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py | _interp_to_grid                    | L2462, L4694     | L4694      | L2462        |
| Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py | _simulate_square_subthreshold      | L2505, L4706     | L4706      | L2505        |
| Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py | _interp_to_grid                    | L2462, L4694     | L4694      | L2462        |
| Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py | _simulate_square_subthreshold      | L2505, L4706     | L4706      | L2505        |
| Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py | _interp_to_grid                    | L2462, L4694     | L4694      | L2462        |
| Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py | _simulate_square_subthreshold      | L2505, L4706     | L4706      | L2505        |
| Passive Features/HPC script/passive_fitting_hpc_fixed.py | _interp_to_grid                    | L2462, L3963     | L3963      | L2462        |
| Passive Features/HPC script/passive_fitting_hpc_fixed.py | _simulate_square_subthreshold      | L2505, L3975     | L3975      | L2505        |
| Passive Features/Plot/passive_result_plot (5).py         | SweepBundle                        | L81, L288        | L288       | L81          |
| Passive Features/Plot/passive_result_plot (5).py         | CellData                           | L97, L304        | L304       | L97          |
| Passive Features/Plot/passive_result_plot (5).py         | PassiveSearchSpace                 | L113, L320       | L320       | L113         |
| Passive Features/Plot/passive_result_plot (5).py         | OptimiserInputs                    | L132, L339       | L339       | L132         |
| Passive Features/Plot/passive_result_plot (5).py         | IncompleteDataError                | L149, L356       | L356       | L149         |
| Passive Features/Plot/passive_result_plot (5).py         | PassiveFitResult                   | L161, L368       | L368       | L161         |
| Passive Features/phase1fittingcolab (2).py               | _interp_to_grid                    | L1938, L3399     | L3399      | L1938        |
| Passive Features/phase1fittingcolab (2).py               | _simulate_square_subthreshold      | L1981, L3411     | L3411      | L1981        |
| Rec_Utility.py                                           | plot_soma_skeleton                 | L1019, L1961     | L1961      | L1019        |
| Rec_Utility.py                                           | find_stable_soma_centroid          | L1113, L2055     | L2055      | L1113        |
| Synaptic Placement/synaptic_placement (9).py             | map_abstract_synapses_to_segments  | L1613, L2541     | L2541      | L1613        |
| Synaptic Placement/synaptic_placement (9).py             | cell_MorphSelect                   | L1688, L2616     | L2616      | L1688        |
| Synaptic Placement/synaptic_placement (9).py             | debug_snapping_accuracy_3d         | L1959, L2713     | L2713      | L1959        |
| final_get_comparments_h10.py                             | plot_soma_skeleton                 | L873, L2564      | L2564      | L873         |
| final_get_comparments_h10.py                             | find_stable_soma_centroid          | L967, L2658      | L2658      | L967         |
| final_get_comparments_h10.py                             | plot_merged_neuron                 | L1486, L1742     | L1742      | L1486        |

## 5. Duplicate top-level class names

Recorded for awareness. Smoke-test assertion 6 acts on this at S0.9 and
must be scoped, or it can never pass while the passive sub-project and
the Colab scripts remain in the tree.

**39 names defined in more than one file.**

| class                        | copies | locations                                                                                        |
|------------------------------|--------|--------------------------------------------------------------------------------------------------|
| BootstrapCIResult            | 6      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:4745; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:4745; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:4745; Passive Features/HPC script/passive_fitting_hpc_fixed.py:4014; Passive Features/Plot/passive_result_plot (5).py:767; Passive Features/phase1fittingcolab (2).py:3450 |
| CellData                     | 10     | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:1632; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:7362; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:12727; Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:649; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:649; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:649; Passive Features/HPC script/passive_fitting_hpc_fixed.py:649; Passive Features/Plot/passive_result_plot (5).py:97; Passive Features/Plot/passive_result_plot (5).py:304; Passive Features/phase1fittingcolab (2).py:157 |
| CellQCResult                 | 2      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:11829; Passive Features/Allen Institute Data/trace_qc (1).py:161 |
| CellSweepInput               | 2      | Passive Features/HPC script/Biological Fit/cm_profile_sweep.py:318; Passive Features/HPC script/Synthetic Passive fit Test/cm_profile_sweep.py:318 |
| CmProfile                    | 2      | Passive Features/HPC script/Biological Fit/cm_profile_sweep.py:88; Passive Features/HPC script/Synthetic Passive fit Test/cm_profile_sweep.py:88 |
| Connectomics                 | 3      | towards_eeg/connectome/connectomics.py:6; towards_eeg/connectome/usage_example.py:6; towards_eeg/hybrid/params.py:1212 |
| ElectricalViabilityConfig    | 3      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:6631; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:10169; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:15623 |
| ElectricalViabilityResult    | 3      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:6658; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:10225; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:15679 |
| GpDiagnosticPerParameter     | 6      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:4787; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:4787; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:4787; Passive Features/HPC script/passive_fitting_hpc_fixed.py:4056; Passive Features/Plot/passive_result_plot (5).py:809; Passive Features/phase1fittingcolab (2).py:3492 |
| GpDiagnosticResult           | 6      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:4802; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:4802; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:4802; Passive Features/HPC script/passive_fitting_hpc_fixed.py:4071; Passive Features/Plot/passive_result_plot (5).py:824; Passive Features/phase1fittingcolab (2).py:3507 |
| IncompleteDataError          | 10     | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:1595; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:7327; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:12692; Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:612; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:612; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:612; Passive Features/HPC script/passive_fitting_hpc_fixed.py:612; Passive Features/Plot/passive_result_plot (5).py:149; Passive Features/Plot/passive_result_plot (5).py:356; Passive Features/phase1fittingcolab (2).py:120 |
| MorphoPath                   | 2      | Parames evoked with EEG/Params_evoked_with_EEG_multiMorph.py:1119; towards_eeg/hybrid/params.py:2369 |
| MorphologyRepair             | 3      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:5422; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:9430; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:14884 |
| OptimiserInputs              | 10     | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:1712; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:7442; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:12807; Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:729; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:729; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:729; Passive Features/HPC script/passive_fitting_hpc_fixed.py:729; Passive Features/Plot/passive_result_plot (5).py:132; Passive Features/Plot/passive_result_plot (5).py:339; Passive Features/phase1fittingcolab (2).py:237 |
| PassiveCell                  | 8      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:2456; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:8181; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:13553; Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:1473; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:1473; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:1473; Passive Features/HPC script/passive_fitting_hpc_fixed.py:1473; Passive Features/phase1fittingcolab (2).py:981 |
| PassiveFitResult             | 8      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:150; Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:2287; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:2287; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:2287; Passive Features/HPC script/passive_fitting_hpc_fixed.py:2287; Passive Features/Plot/passive_result_plot (5).py:161; Passive Features/Plot/passive_result_plot (5).py:368; Passive Features/phase1fittingcolab (2).py:1763 |
| PassiveSearchSpace           | 10     | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:1663; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:7393; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:12758; Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:680; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:680; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:680; Passive Features/HPC script/passive_fitting_hpc_fixed.py:680; Passive Features/Plot/passive_result_plot (5).py:113; Passive Features/Plot/passive_result_plot (5).py:320; Passive Features/phase1fittingcolab (2).py:188 |
| Phase2p5CellResult           | 3      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:3416; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:3416; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:3416 |
| Phase2p5GroupResult          | 3      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:3461; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:3461; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:3461 |
| Phase3Result                 | 6      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:4811; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:4811; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:4811; Passive Features/HPC script/passive_fitting_hpc_fixed.py:4080; Passive Features/Plot/passive_result_plot (5).py:833; Passive Features/phase1fittingcolab (2).py:3516 |
| Population                   | 2      | Population/Population_multiMorph.py:919; towards_eeg/hybrid/population.py:1138                   |
| PopulationSuper              | 2      | Population/Population_multiMorph.py:41; towards_eeg/hybrid/population.py:45                      |
| PulseQCResult                | 2      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:11806; Passive Features/Allen Institute Data/trace_qc (1).py:138 |
| QCConfig                     | 2      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:11759; Passive Features/Allen Institute Data/trace_qc (1).py:91 |
| RepairConfig                 | 3      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:4965; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:8973; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:14427 |
| RepairDiagnostic             | 3      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:4994; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:9002; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:14456 |
| RepairOutcome                | 3      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:6156; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:9633; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:15087 |
| RepairResult                 | 3      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:5017; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:9025; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:14479 |
| RepairStep                   | 3      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:4986; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:8994; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:14448 |
| ReplotBundle                 | 6      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:6295; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:6295; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:6295; Passive Features/HPC script/passive_fitting_hpc_fixed.py:5482; Passive Features/Plot/passive_result_plot (5).py:2209; Passive Features/phase1fittingcolab (2).py:4892 |
| SomaGatedRepairConfig        | 3      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:6148; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:9625; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:15079 |
| SweepBundle                  | 10     | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:1613; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:7343; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:12708; Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:630; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:630; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:630; Passive Features/HPC script/passive_fitting_hpc_fixed.py:630; Passive Features/Plot/passive_result_plot (5).py:81; Passive Features/Plot/passive_result_plot (5).py:288; Passive Features/phase1fittingcolab (2).py:138 |
| TopoPopulation               | 2      | Population/Population_multiMorph.py:1931; towards_eeg/hybrid/population.py:1708                  |
| _NumpyEncoder                | 3      | Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:3189; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:8506; Passive Features/Allen Institute Data/save_alleninstitute_data (6).py:13878 |
| _SkoptDimLite                | 6      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:6278; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:6278; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:6278; Passive Features/HPC script/passive_fitting_hpc_fixed.py:5465; Passive Features/Plot/passive_result_plot (5).py:2192; Passive Features/phase1fittingcolab (2).py:4875 |
| _TrainBundleLite             | 6      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:6257; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:6257; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:6257; Passive Features/HPC script/passive_fitting_hpc_fixed.py:5444; Passive Features/Plot/passive_result_plot (5).py:2171; Passive Features/phase1fittingcolab (2).py:4854 |
| general_params               | 2      | Parames evoked with EEG/Params_evoked_with_EEG_multiMorph.py:178; towards_eeg/hybrid/params.py:178 |
| multicompartment_params      | 2      | Parames evoked with EEG/Params_evoked_with_EEG_multiMorph.py:744; towards_eeg/hybrid/params.py:824 |
| point_neuron_network_params  | 2      | Parames evoked with EEG/Params_evoked_with_EEG_multiMorph.py:445; towards_eeg/hybrid/params.py:525 |

## 6. Full ledger

Hash-chain columns: `pre` is measured now. `s02`, `s03`, `s04` are
filled by rerunning this tool after each of those sub-steps. A dash means
the sub-step has not run yet. S0.4 must leave `s04` equal to `s03`:
that equality IS the exit test for the move.

| path                                                                         | scope         | stage  | verdict  | pre           | s02  | s03  | s04  |
|------------------------------------------------------------------------------|---------------|--------|----------|---------------|------|------|------|
| .gitignore                                                                   | colab         | -      | retain   | 36e51240be00  | -    | 36e  | 36e  |
| Align Morphologies/Alignment.py                                              | colab         | -      | retain   | 153229364d93  | 6aa  | d99  | d99  |
| Align Morphologies/Alignment_IdentifySpines.py                               | colab         | -      | retain   | 00ea21aaec70  | 00e  | c34  | c34  |
| Align Morphologies/README.md                                                 | colab         | -      | retain   | 6abfac74ce73  | 6ab  | 6ab  | 6ab  |
| Align Morphologies/Usage example.py                                          | colab         | -      | retain   | 2f310fe50363  | 2f3  | 2f3  | 2f3  |
| Alignment Metadata/Extract_metadata.py                                       | colab         | -      | retain   | 1a462fe0a7c1  | 1a4  | fb3  | fb3  |
| Alignment Metadata/README.md                                                 | colab         | -      | retain   | 3a9ff3627f65  | 3a9  | 3a9  | 3a9  |
| Alignment Metadata/Usage.py                                                  | colab         | -      | retain   | 96d6e6a72c39  | 96d  | 78e  | 78e  |
| Connectomics/Calculate Relative Subpop pct/Usage_example.py                  | colab         | -      | retain   | 7f95739f273f  | 7f9  | 36d  | 36d  |
| Connectomics/Calculate Relative Subpop pct/calculate_bbp_relative_presences.py | colab         | -      | retain   | d10121eb3f7e  | d10  | d10  | d10  |
| Connectomics/Construct_ADJ/Construct_ADJ.py                                  | colab         | -      | retain   | 550f1d89a21e  | 550  | 550  | 550  |
| Connectomics/Construct_ADJ/Microcircuit Assembly Documentation.pdf           | colab         | S0.7   | discard  | dec69a3bdff9  | dec  | dec  | dec  |
| Connectomics/Construct_ADJ/Plotting_function                                 | colab         | -      | retain   | 1af3295bfe7a  | 1af  | 1af  | 1af  |
| Connectomics/Construct_ADJ/Usage_example.py                                  | colab         | -      | retain   | 5aebc98a68f6  | 5ae  | 5ae  | 5ae  |
| Connectomics/Full_impementation                                              | colab         | -      | retain   | e5f5fdef748d  | e5f  | e5f  | e5f  |
| Connectomics/Generate Point Microcolumn/Population Generation Documentation (1).pdf | colab         | S0.7   | discard  | be7956c4b022  | be7  | be7  | be7  |
| Connectomics/Generate Point Microcolumn/Usage_example.py                     | colab         | -      | retain   | 4c1226dcdce8  | 4c1  | 4c1  | 4c1  |
| Connectomics/Generate Point Microcolumn/generate_microcolumn_cells.py        | colab         | -      | retain   | d2ce64546d55  | d2c  | d2c  | d2c  |
| Dendrite F score/README.md                                                   | colab         | -      | retain   | f2eae84356fd  | f2e  | f2e  | f2e  |
| Dendrite F score/morpholgy_pathways (6).py                                   | colab         | -      | retain   | bd8cf4b341be  | bd8  | 4a0  | 4a0  |
| Fetch Synapses/Fetch_MappedSyn.py                                            | colab         | -      | retain   | 81b9fef26511  | 81b  | 81b  | 81b  |
| Fetch Synapses/README.md                                                     | colab         | -      | retain   | d98bf970cbb7  | d98  | d98  | d98  |
| Fetch Synapses/Usage_Example.py                                              | colab         | -      | retain   | eeb2aa856814  | eeb  | bdd  | bdd  |
| Final Implementation/README.md                                               | colab         | -      | retain   | 15b82ff6e8cd  | 15b  | 15b  | 15b  |
| Jitter Neuron/README.md                                                      | colab         | -      | retain   | a09b36fc6e2a  | a09  | a09  | a09  |
| Jitter Neuron/jitter_neurons (1).py                                          | colab         | -      | retain   | aaf2bbe87751  | aaf  | 5b5  | 5b5  |
| Morphology Compliance/Compliance_Check.py                                    | colab         | -      | retain   | babeb0342594  | 431  | 35c  | 35c  |
| Morphology Compliance/Morphology Selection Documentation.pdf                 | colab         | S0.7   | discard  | 2acac69e5047  | 2ac  | 2ac  | 2ac  |
| Morphology Compliance/Usage example                                          | colab         | -      | retain   | 2f366ba474ee  | 2f3  | 2f3  | 2f3  |
| Nuova cartella compressa.zip                                                 | colab         | S0.7   | discard  | 9da5198cce44  | 9da  | 9da  | 9da  |
| Point Neuronal Network/Utility Functions/HDF5_Builder.py                     | colab         | -      | retain   | 2bc3d3328419  | 2bc  | 2bc  | 2bc  |
| README.md                                                                    | colab         | -      | retain   | 648aec5644fb  | 648  | 648  | 648  |
| Rec_Utility.py                                                               | colab         | -      | retain   | f13de0459687  | f13  | e97  | e97  |
| Save nids/Usage_example.py                                                   | colab         | -      | retain   | 7f3657845a84  | 7f3  | 7f3  | 7f3  |
| Save nids/save_indicies.py                                                   | colab         | -      | retain   | ea8eb268821e  | ea8  | ea8  | ea8  |
| Spanning Trees/Calculate_field.py                                            | colab         | -      | retain   | a481c7f3cdca  | a48  | 027  | 027  |
| Spanning Trees/Neuronal Mapping Documentation.pdf                            | colab         | S0.7   | discard  | 14240ebdd440  | 142  | 142  | 142  |
| Spanning Trees/Save.py                                                       | colab         | -      | retain   | 5c8b29eaeef0  | 5c8  | 14d  | 14d  |
| Spanning Trees/Usage_example.py                                              | colab         | -      | retain   | 5e7a16b72697  | 5e7  | 5e7  | 5e7  |
| Spanning tree overlap and synapse identification/Peters Rule Connectivity Engine Documentation.pdf | colab         | S0.7   | discard  | a44a1660a019  | a44  | a44  | a44  |
| Spanning tree overlap and synapse identification/Synaptic Overlap Debugger Documentation.pdf | colab         | S0.7   | discard  | 7a7a3b5a8d8a  | 7a7  | 7a7  | 7a7  |
| Spanning tree overlap and synapse identification/Usage_Example.py            | colab         | -      | retain   | 0d5e17d101cf  | 0d5  | 0d5  | 0d5  |
| Spanning tree overlap and synapse identification/calculate_synaptic_overlap.py | colab         | -      | retain   | 88206f979839  | 882  | 358  | 358  |
| Spine Detection/DEBUG/Generate_smooth_surface.py                             | colab         | -      | retain   | c792fbfc66e4  | c79  | 5bb  | 5bb  |
| Spine Detection/DEBUG/Label_spines.py                                        | colab         | -      | retain   | 7f30cf1710ed  | 7f3  | 6a6  | 6a6  |
| Spine Detection/DEBUG/README.md                                              | colab         | -      | retain   | 2d17a2e83686  | 2d1  | 2d1  | 2d1  |
| Spine Detection/DEBUG/Segment_spine.py                                       | colab         | -      | retain   | 3c63163f6c5e  | 3c6  | d47  | d47  |
| Spine Detection/Label_Dendritic_Spines.py                                    | colab         | -      | retain   | 2c56299fde3a  | 2c5  | d9e  | d9e  |
| Spine Detection/README.md                                                    | colab         | -      | retain   | 21224f75dad4  | 212  | 212  | 212  |
| Synapse Retrival/Assess_quality.py                                           | colab         | -      | retain   | 5e7569a8ef88  | 5e7  | 7c2  | 7c2  |
| Synapse Retrival/Extract_synapses.py                                         | colab         | -      | retain   | 2ebf1ef9d730  | 8aa  | 5c6  | 5c6  |
| Synapse Retrival/README.md                                                   | colab         | -      | retain   | d0354e2a0a56  | d03  | d03  | d03  |
| Synapse Retrival/map_synapses_to_segments.py                                 | colab         | -      | retain   | 27b225da3a49  | 27b  | fad  | fad  |
| Synaptic Placement/Neural Simulation Documentation.pdf                       | colab         | S0.7   | discard  | d5300eb50e13  | d53  | d53  | d53  |
| Synaptic Placement/synaptic_placement (9).py                                 | colab         | -      | retain   | a068ccaba84d  | 799  | 445  | 445  |
| TRANSCRIPTOMICS_PROTEOMICS/README.md                                         | colab         | -      | retain   | e2f89bc4b9d2  | e2f  | e2f  | e2f  |
| TRANSCRIPTOMICS_PROTEOMICS/transcriptomics_proteomics (1).py                 | colab         | -      | retain   | 24f7f063a7db  | 24f  | 24f  | 24f  |
| automated_reconstruction.py                                                  | colab         | -      | retain   | 463690b33638  | e3e  | 16d  | 16d  |
| final_get_comparments_h10.py                                                 | colab         | -      | retain   | f967b32cd895  | ed4  | 92d  | 92d  |
| utility.py                                                                   | colab         | -      | retain   | a20f39eedd3b  | 2ab  | 51e  | 51e  |
| .gitattributes                                                               | infrastructure | S0.3   | new      | 21a67317bef6  | -    | -    | 21a  |
| LEDGER.md                                                                    | infrastructure | S0.0   | new      | 5075a5ab0a05  | 0bf  | eaa  | 587  |
| ledger.csv                                                                   | infrastructure | S0.0   | new      | 8f2951bdd603  | 32b  | 851  | 9d7  |
| pyproject.toml                                                               | infrastructure | S0.4   | new      | 377f9664b589  | -    | -    | 210  |
| tools/ancestors.json                                                         | infrastructure | S0.0   | new      | 3a0672db37f8  | 2a3  | 2a3  | 8a0  |
| tools/apply_moves.py                                                         | infrastructure | S0.2c  | new      | 5615213c62b1  | -    | 5f3  | 561  |
| tools/build_ledger.py                                                        | infrastructure | S0.0   | new      | afb19b68ac9a  | 0a3  | 2e3  | ebf  |
| tools/path_moves.json                                                        | infrastructure | S0.2c  | new      | baf9c7926395  | -    | 6e8  | baf  |
| tools/phase_hashes.json                                                      | infrastructure | S0.2   | new      | e0797061508d  | -    | ab1  | e00  |
| tools/s0_ledger/__init__.py                                                  | infrastructure | S0.0   | new      | 7ec4151c2327  | 7ec  | 7ec  | 7ec  |
| tools/s0_ledger/analyse.py                                                   | infrastructure | S0.0   | new      | 8cb68f8e03ad  | a1f  | 336  | 7d0  |
| tools/s0_ledger/render.py                                                    | infrastructure | S0.0   | new      | 5ae8d7abf0ce  | 5ae  | 610  | 610  |
| tools/s0_ledger/scan.py                                                      | infrastructure | S0.0   | new      | 6d8862fa2ea6  | 6d8  | 6d8  | 6d8  |
| tools/s0_paths.py                                                            | infrastructure | S0.4   | new      | 929ef005d98b  | -    | -    | 929  |
| tools/s0_transform/S02_discard.sh                                            | infrastructure | S0.2   | new      | 9e5b9e16be5d  | -    | 9e5  | 9e5  |
| tools/s0_transform/__init__.py                                               | infrastructure | S0.2   | new      | e9e6c067a8ae  | -    | e9e  | e9e  |
| tools/s0_transform/asciify.py                                                | infrastructure | S0.3   | new      | e2967d7b04e1  | -    | -    | e29  |
| tools/s0_transform/decolab.py                                                | infrastructure | S0.2   | new      | 68f7db5fc32d  | -    | 68f  | 68f  |
| tools/s0_transform/s02_colab_commands.json                                   | infrastructure | S0.2   | new      | d5c039844cbc  | -    | d5c  | d5c  |
| tools/s0_transform/s02_exit_scope.json                                       | infrastructure | S0.2   | new      | e7bc2d61c3a6  | -    | e7b  | e7b  |
| tools/s0_transform/s02_targets.json                                          | infrastructure | S0.2   | new      | 77796b350ad3  | -    | 777  | 777  |
| tools/s0_transform/s02_transform_log.json                                    | infrastructure | S0.2   | new      | 43f9086474d1  | -    | 43f  | 43f  |
| tools/s0_transform/s03_exit_scope.json                                       | infrastructure | S0.3   | new      | 85dde24f2176  | -    | -    | 85d  |
| tools/s0_transform/s03_targets.json                                          | infrastructure | S0.3   | new      | 073092c2968b  | -    | -    | 073  |
| tools/s0_transform/s03_transform_log.json                                    | infrastructure | S0.3   | new      | e1dc45bfda62  | -    | -    | e1d  |
| tools/s0_transform/s03_translit_map.json                                     | infrastructure | S0.3   | new      | 5f0fd4bf6e2b  | -    | -    | 5f0  |
| tools/s0_transform/s04_exit_scope.json                                       | infrastructure | S0.0   | new      | 43f188d585c7  | -    | -    | 2ba  |
| tools/s0_transform/s05_exit_scope.json                                       | infrastructure | S0.5   | new      | ec3d7b5b4a53  | -    | -    | -    |
| tools/stamp_phase.py                                                         | infrastructure | S0.2   | new      | 492dac01b50a  | -    | 7ac  | 7ac  |
| tools/test_s0_2_exit.py                                                      | infrastructure | S0.2   | new      | dc29ec7d7c87  | -    | dc2  | dc2  |
| tools/test_s0_3_exit.py                                                      | infrastructure | S0.3   | new      | 17df58d01aab  | -    | -    | 17d  |
| tools/test_s0_4_exit.py                                                      | infrastructure | S0.4   | new      | ddf9ba47b8a6  | -    | -    | 2d8  |
| tools/test_s0_4_smoke.py                                                     | infrastructure | S0.0   | new      | 7a8d6f644cff  | -    | -    | 565  |
| tools/test_s0_5_exit.py                                                      | infrastructure | S0.5   | new      | b41fdf00ac88  | -    | -    | -    |
| tools/test_s0_5_smoke.py                                                     | infrastructure | S0.5   | new      | 8d10fc9624e5  | -    | -    | -    |
| tools/test_s0_asciify_smoke.py                                               | infrastructure | S0.3   | new      | 64f7aebbfd75  | -    | -    | 64f  |
| tools/test_s0_chain.py                                                       | infrastructure | S0.3   | new      | be648dc4f3be  | -    | -    | a88  |
| tools/test_s0_decolab_smoke.py                                               | infrastructure | S0.2   | new      | 49b18e35dbb5  | -    | 49b  | 49b  |
| tools/test_s0_ledger_smoke.py                                                | infrastructure | S0.0   | new      | 57eae0d12bc9  | 57e  | 57e  | 57e  |
| tools/test_s0_moves_smoke.py                                                 | infrastructure | S0.2c  | new      | c0e46e478ac0  | -    | 42e  | c18  |
| towards_eeg/__init__.py                                                      | infrastructure | S0.4   | new      | 53378de1aac3  | -    | -    | 533  |
| towards_eeg/config/__init__.py                                               | infrastructure | S0.4   | new      | edbe5540e130  | -    | -    | edb  |
| towards_eeg/connectome/__init__.py                                           | infrastructure | S0.4   | new      | 4b53382c2d9e  | -    | -    | 4b5  |
| towards_eeg/cosim/__init__.py                                                | infrastructure | S0.4   | new      | 69c430578222  | -    | -    | 69c  |
| towards_eeg/hybrid/__init__.py                                               | infrastructure | S0.4   | new      | a16a45a3ec5b  | -    | -    | a16  |
| towards_eeg/io/__init__.py                                                   | infrastructure | S0.4   | new      | d678cb676da1  | -    | -    | 0aa  |
| towards_eeg/io/geometry.py                                                   | infrastructure | S0.5   | new      | a33cf6906644  | -    | -    | -    |
| towards_eeg/io/invariants.py                                                 | infrastructure | S0.5   | new      | b45c66152e47  | -    | -    | -    |
| towards_eeg/io/schema.py                                                     | infrastructure | S0.5   | new      | c67c878b9889  | -    | -    | -    |
| towards_eeg/io/schemas/C08_morphology_bank.json                              | infrastructure | S0.5   | new      | 613bb0403aaa  | -    | -    | -    |
| towards_eeg/io/schemas/C09_observed_synapses.json                            | infrastructure | S0.5   | new      | adb16e556f93  | -    | -    | -    |
| towards_eeg/io/schemas/C14_passive_parameters.json                           | infrastructure | S0.5   | new      | 41d7a55b20eb  | -    | -    | -    |
| towards_eeg/io/schemas/C15_mechanism_spec.json                               | infrastructure | S0.5   | new      | c287c9febd4f  | -    | -    | -    |
| towards_eeg/io/schemas/alphabets.json                                        | infrastructure | S0.5   | new      | a8332981d5cb  | -    | -    | -    |
| towards_eeg/io/validate.py                                                   | infrastructure | S0.5   | new      | 4aedc7dabc05  | -    | -    | -    |
| towards_eeg/passive/__init__.py                                              | infrastructure | S0.4   | new      | a06facd2152f  | -    | -    | a06  |
| towards_eeg/pointnet/__init__.py                                             | infrastructure | S0.4   | new      | 5be4441efb25  | -    | -    | 5be  |
| towards_eeg/structure/__init__.py                                            | infrastructure | S0.4   | new      | 835cb1cb771f  | -    | -    | 835  |
| Classes/Connectomics/Connectomics Class Overview.pdf                         | orchestrator  | S0.7   | discard  | 3cea1baed2e4  | 3ce  | 3ce  | 3ce  |
| HybridLFPy Tweaked/Extract macro-population data/Project README.pdf          | orchestrator  | S0.7   | discard  | 95a6e04fd1bd  | 95a  | 95a  | 95a  |
| Parames evoked with EEG/Params_evoked_with_EEG_multiMorph.py                 | orchestrator  | -      | retain   | de9169ec67d8  | de9  | de9  | de9  |
| Population/Population_multiMorph.py                                          | orchestrator  | -      | retain   | 78ce9475d76c  | 78c  | 78c  | 78c  |
| Population/README.md                                                         | orchestrator  | -      | retain   | ca11825113c5  | ca1  | ca1  | ca1  |
| Utility_fun.py                                                               | orchestrator  | S0.1   | keep     | 9d3d29c8a997  | -    | -    | -    |
| hybrid_sim_evoked_with_EEG_CHANGED.py                                        | orchestrator  | S0.1   | keep     | 6c3ed490ba14  | -    | -    | -    |
| params_evoked_with_EEG_CHANGED.py                                            | orchestrator  | S0.1   | keep     | 927ee3d9cc0d  | -    | -    | -    |
| population_CHANGED.py                                                        | orchestrator  | S0.1   | keep     | 43df1344230e  | -    | -    | -    |
| towards_eeg/connectome/connectivity_buildup.py                               | orchestrator  | S0.4   | keep     | e62667fa6e57  | e62  | 1a4  | 1a4  |
| towards_eeg/connectome/connectomics.py                                       | orchestrator  | S0.4   | keep     | 2f58a6ae79a3  | 2f5  | 5c2  | 5c2  |
| towards_eeg/connectome/macropopulation.py                                    | orchestrator  | S0.4   | keep     | 0bfb98e308b9  | 0bf  | 0bf  | 0bf  |
| towards_eeg/connectome/usage_example.py                                      | orchestrator  | S0.4   | keep     | dc4828bb65d4  | dc4  | 6b3  | 6b3  |
| towards_eeg/hybrid/README.md                                                 | orchestrator  | S0.4   | keep     | 8183e1b9cd57  | 818  | 818  | 818  |
| towards_eeg/hybrid/driver.py                                                 | orchestrator  | S0.4   | keep     | 566b25332991  | 6c3  | c19  | c19  |
| towards_eeg/hybrid/params.py                                                 | orchestrator  | S0.4   | keep     | 927ee3d9cc0d  | 927  | 1fa  | 1fa  |
| towards_eeg/hybrid/population.py                                             | orchestrator  | S0.4   | keep     | 35dbec718a35  | 43d  | 829  | 829  |
| towards_eeg/hybrid/utility.py                                                | orchestrator  | S0.4   | keep     | ec55afe6c0c1  | 9d3  | 9da  | 9da  |
| Passive Features/Allen Institute Data/README.md                              | passive       | -      | retain   | 954aa51240c9  | 954  | 954  | 954  |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py        | passive       | -      | retain   | 9d8ccf9f5698  | 84b  | 0fd  | 0fd  |
| Passive Features/Allen Institute Data/trace_qc (1).py                        | passive       | -      | retain   | 5d983eb7cc91  | 5d9  | a88  | a88  |
| Passive Features/Allen Institute Data/trace_qc_v2.py                         | passive       | -      | retain   | 3baf0064e73f  | 3ba  | d21  | d21  |
| Passive Features/HPC script/Biological Fit/Biological_Passive_Fit_HPC_Guide.md | passive       | -      | retain   | d5294b571c9b  | d52  | d52  | d52  |
| Passive Features/HPC script/Biological Fit/README.md                         | passive       | -      | retain   | 1867dde1be4f  | 186  | 186  | 186  |
| Passive Features/HPC script/Biological Fit/cm_profile_sweep.py               | passive       | -      | retain   | 236b3827d6fd  | 236  | 236  | 236  |
| Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py      | passive       | -      | retain   | 6c95c3accb42  | 6c9  | caf  | caf  |
| Passive Features/HPC script/Biological Fit/passive_long_step_training.py     | passive       | -      | retain   | 845c4459dbbb  | 845  | e19  | e19  |
| Passive Features/HPC script/Biological Fit/run_biological_fit.py             | passive       | -      | retain   | 97f67b587144  | 97f  | 97f  | 97f  |
| Passive Features/HPC script/Biological Fit/smoke_run_biological_fit.py       | passive       | -      | retain   | 9f2f46440e03  | 9f2  | 9f2  | 9f2  |
| Passive Features/HPC script/Biological Fit/submit_all_groups_biological.sh   | passive       | -      | retain   | 8c3fd53e2abe  | 8c3  | 8c3  | 8c3  |
| Passive Features/HPC script/Biological Fit/submit_biological_fit.sh          | passive       | -      | retain   | da07d3283977  | da0  | da0  | da0  |
| Passive Features/HPC script/Error_Log.txt                                    | passive       | -      | retain   | 042dc06898fa  | 042  | 042  | 042  |
| Passive Features/HPC script/Multiple Sweeps with phase 2.5/README.md         | passive       | -      | retain   | 6e924b9543d3  | 6e9  | 6e9  | 6e9  |
| Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py | passive       | -      | retain   | 6c95c3accb42  | 6c9  | caf  | caf  |
| Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed_description.md | passive       | -      | retain   | ea52844dec91  | ea5  | ea5  | ea5  |
| Passive Features/HPC script/Multiple nodes/README.md                         | passive       | -      | retain   | cd1425d8e2bc  | cd1  | cd1  | cd1  |
| Passive Features/HPC script/Multiple nodes/submit_all_groups.sh              | passive       | -      | retain   | 57ddbfda6712  | 57d  | fac  | fac  |
| Passive Features/HPC script/Multiple nodes/submit_passive_fit.sh             | passive       | -      | retain   | 94cf1cd080ab  | 94c  | 68d  | 68d  |
| Passive Features/HPC script/README.md                                        | passive       | -      | retain   | b3662fa0410a  | b36  | b36  | b36  |
| Passive Features/HPC script/Synthetic Passive fit Test/Ih.mod                | passive       | -      | retain   | c989a36d9730  | c98  | c98  | c98  |
| Passive Features/HPC script/Synthetic Passive fit Test/README.md             | passive       | -      | retain   | 8a29244b39ce  | 8a2  | 8a2  | 8a2  |
| Passive Features/HPC script/Synthetic Passive fit Test/aggregate_synth_results.py | passive       | -      | retain   | d995c1b46af7  | d99  | ecf  | ecf  |
| Passive Features/HPC script/Synthetic Passive fit Test/benchmark_usage_guide.md | passive       | -      | retain   | 9dfd962d4bc6  | 9df  | 9df  | 9df  |
| Passive Features/HPC script/Synthetic Passive fit Test/cm_profile_sweep.py   | passive       | -      | retain   | 236b3827d6fd  | 236  | 236  | 236  |
| Passive Features/HPC script/Synthetic Passive fit Test/gen_from_manifest.py  | passive       | -      | retain   | bdf4f6d72683  | bdf  | 8fa  | 8fa  |
| Passive Features/HPC script/Synthetic Passive fit Test/kv.mod                | passive       | -      | retain   | c97ebc7abff9  | c97  | c97  | c97  |
| Passive Features/HPC script/Synthetic Passive fit Test/na.mod                | passive       | -      | retain   | e8fda3334266  | e8f  | e8f  | e8f  |
| Passive Features/HPC script/Synthetic Passive fit Test/passive_consistency_diagnostic.py | passive       | -      | retain   | 844c92272e11  | 844  | 844  | 844  |
| Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py | passive       | -      | retain   | 6c95c3accb42  | 6c9  | caf  | caf  |
| Passive Features/HPC script/Synthetic Passive fit Test/passive_long_step_training.py | passive       | -      | retain   | 845c4459dbbb  | 845  | e19  | e19  |
| Passive Features/HPC script/Synthetic Passive fit Test/run_synth_benchmark.py | passive       | -      | retain   | 2f62abb5249f  | 2f6  | 84a  | 84a  |
| Passive Features/HPC script/Synthetic Passive fit Test/smoke_aggregate_synth_results.py | passive       | -      | retain   | 2d27a88355ca  | 2d2  | 2d2  | 2d2  |
| Passive Features/HPC script/Synthetic Passive fit Test/smoke_gen_from_manifest.py | passive       | -      | retain   | eb9f612cdc94  | eb9  | eb9  | eb9  |
| Passive Features/HPC script/Synthetic Passive fit Test/smoke_run_synth_benchmark.py | passive       | -      | retain   | 769ea0d8f7d2  | 769  | 769  | 769  |
| Passive Features/HPC script/Synthetic Passive fit Test/smoke_synth_gt_grid.py | passive       | -      | retain   | e8787fd8fe28  | e87  | e87  | e87  |
| Passive Features/HPC script/Synthetic Passive fit Test/submit_all_cohorts.sh | passive       | -      | retain   | 1bca07408152  | 1bc  | 1bc  | 1bc  |
| Passive Features/HPC script/Synthetic Passive fit Test/submit_synth_benchmark.sh | passive       | -      | retain   | 0bf477ce6ab6  | 0bf  | 2ee  | 2ee  |
| Passive Features/HPC script/Synthetic Passive fit Test/synth_gt_grid.py      | passive       | -      | retain   | 495d2c4eb558  | 495  | 495  | 495  |
| Passive Features/HPC script/Synthetic Passive fit Test/synthetic_ground_truth.py | passive       | -      | retain   | fd1bebcaba5b  | fd1  | fd1  | fd1  |
| Passive Features/HPC script/passive_fit.e1099100                             | passive       | S0.7   | discard  | 856f6afb4ea7  | 856  | 856  | 856  |
| Passive Features/HPC script/passive_fit.e1099156                             | passive       | S0.7   | discard  | 3c03fe908847  | 3c0  | 3c0  | 3c0  |
| Passive Features/HPC script/passive_fit.e1099215                             | passive       | S0.7   | discard  | 87501536b336  | 875  | 875  | 875  |
| Passive Features/HPC script/passive_fit.o1099100                             | passive       | S0.7   | discard  | 2c2f8bc9b1e7  | 2c2  | 2c2  | 2c2  |
| Passive Features/HPC script/passive_fit.o1099156                             | passive       | S0.7   | discard  | ef17cc6efc8e  | ef1  | ef1  | ef1  |
| Passive Features/HPC script/passive_fit.o1099215                             | passive       | S0.7   | discard  | c69560156a4e  | c69  | c69  | c69  |
| Passive Features/HPC script/passive_fitting_hpc_fixed.py                     | passive       | -      | retain   | 366381884ca2  | 366  | dc1  | dc1  |
| Passive Features/HPC script/requirements.txt                                 | passive       | -      | retain   | f0fbc7ed8c52  | f0f  | f0f  | f0f  |
| Passive Features/HPC script/submit_passive_fit.sh                            | passive       | -      | retain   | aaf476707ddc  | aaf  | d26  | d26  |
| Passive Features/HPC script/test_pickle_roundtrip.py                         | passive       | -      | retain   | 4f9e385f9bad  | 4f9  | 4f9  | 4f9  |
| Passive Features/Passive Allen Data/L2/README.md                             | passive       | -      | retain   | 3bb5a022d1b9  | 3bb  | 3bb  | 3bb  |
| Passive Features/Passive Allen Data/L2/candidates.csv                        | passive       | -      | retain   | 8dc7a8e93c70  | 8dc  | 8dc  | 8dc  |
| Passive Features/Passive Allen Data/L2/manifest.json                         | passive       | -      | retain   | bb58c6136861  | bb5  | bb5  | bb5  |
| Passive Features/Passive Allen Data/L2/specimen_528706755.zip                | passive       | S0.7   | discard  | 579910a148da  | 579  | 579  | 579  |
| Passive Features/Passive Allen Data/L2/specimen_537204107.zip                | passive       | S0.7   | discard  | 01bff4f88db8  | 01b  | 01b  | 01b  |
| Passive Features/Passive Allen Data/L2/specimen_614659629.zip                | passive       | S0.7   | discard  | e8ceaafe3743  | e8c  | e8c  | e8c  |
| Passive Features/Passive Allen Data/L2/specimen_616647103.zip                | passive       | S0.7   | discard  | f4219ebd7149  | f42  | f42  | f42  |
| Passive Features/Phase 1/README.md                                           | passive       | -      | retain   | 99e89cb13b59  | 99e  | 99e  | 99e  |
| Passive Features/Phase 2/aaa.py                                              | passive       | -      | retain   | 17e682f060b5  | 17e  | 17e  | 17e  |
| Passive Features/Phase 2/phase2_patch.pdf                                    | passive       | S0.7   | discard  | 2d488e5e1300  | 2d4  | 2d4  | 2d4  |
| Passive Features/Phase 2/phase2_technical.pdf                                | passive       | S0.7   | discard  | 03d8016e1289  | 03d  | 03d  | 03d  |
| Passive Features/Phase 3/README.md                                           | passive       | -      | retain   | 4f9b9c11f112  | 4f9  | 4f9  | 4f9  |
| Passive Features/Plot/X                                                      | passive       | -      | retain   | 10c7dbb397fc  | 10c  | 10c  | 10c  |
| Passive Features/Plot/passive_result_plot (5).py                             | passive       | -      | retain   | a6b24bc1a3a6  | 5ad  | 947  | 947  |
| Passive Features/Plot/phase3_publication_plots_v2_documentation.md           | passive       | -      | retain   | c366f5dc493d  | c36  | c36  | c36  |
| Passive Features/node_test.py                                                | passive       | -      | retain   | f1c9580c2604  | f1c  | f1c  | f1c  |
| Passive Features/phase1fittingcolab (2).py                                   | passive       | -      | retain   | b47f1f9e552e  | 739  | 242  | 242  |
| Passive Features/submit_node_test.sh                                         | passive       | -      | retain   | f2329cb6a4be  | f23  | f23  | f23  |

Full detail, including rationale for every row, is in `ledger.csv`.
