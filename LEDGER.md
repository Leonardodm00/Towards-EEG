# Towards-EEG -- S0.0 Reconciliation Ledger

**Stage:** S0.0 (roadmap rev 3, section 3.2)  
**Generated:** 2026-07-20 15:48:09 UTC by `s0_ledger` v1.0.0  
**Snapshot:** `github.com/Leonardodm00/Towards-EEG@main`, 2026-07-19  
**Tree root:** `/home/claude/gitrepo`  
**Rows:** 156 (152 repository files + 4 local working files)

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
| discard      | 23     |
| keep         | 16     |
| new          | 9      |
| retain       | 108    |

### By scope

| scope          | files  |
|----------------|--------|
| colab          | 59     |
| infrastructure | 9      |
| orchestrator   | 19     |
| passive        | 69     |

### By acting sub-step

| stage    | files  |
|----------|--------|
| -        | 98     |
| S0.0     | 9      |
| S0.1     | 4      |
| S0.2     | 11     |
| S0.4     | 12     |
| S0.7     | 22     |

### Python surface at pre-s0

- python files: **86**
- failing `ast.parse`: **11**
- containing bytes >= 0x80: **48**
- non-ASCII and no PEP 263 cookie: **24**
- containing CRLF: **6**
- raising SyntaxWarning (latent, e.g. invalid escape sequences): **6**

## 2. Working-branch reconciliation (S0.1)

The four local files are committed BEFORE the `pre-s0` tag, so that
ancestry is verifiable by `git` rather than by assertion. Relationship
`working_branch_edit` marks a hand edit that is NOT a mechanical
transform; it is admissible only because the commit makes the file its
own ancestor for every later sub-step.

| file                                   | relationship         | T   | ancestor                                             | sha256(pre-s0) |
|----------------------------------------|----------------------|-----|------------------------------------------------------|----------------|
| Utility_fun.py                         | working_branch_edit  | -   | HybridLFPy Tweaked/Utility_function.py               | 9d3d29c8a997   |
| hybrid_sim_evoked_with_EEG_CHANGED.py  | mechanical           | T1  | HybridLFPy Tweaked/hybrid_sim_evoked_with_EEG_multiMorph.py | 6c3ed490ba14   |
| params_evoked_with_EEG_CHANGED.py      | identical            | T0  | HybridLFPy Tweaked/params_evoked_with_EEG_multiMorph.py | 927ee3d9cc0d   |
| population_CHANGED.py                  | working_branch_edit  | -   | HybridLFPy Tweaked/Population_multiMorph.py          | 43df1344230e   |

- **Utility_fun.py** -- TEEG_01 rev 2 section 3.3: ancestor plus three new functions (insert_mechanisms, get_gIhbar_L5_apical, get_gCa_HVA_apical); keep. Carries defects U1-U4, deferred to S3; content differs beyond line endings; ancestor LF, local CRLF (748)
- **hybrid_sim_evoked_with_EEG_CHANGED.py** -- TEEG_01 rev 2 section 3.3: identical modulo a trailing newline; keep; trailing newline only (line endings may also differ)
- **params_evoked_with_EEG_CHANGED.py** -- TEEG_01 rev 2 section 3.3: byte-identical to the snapshot; keep; bytes equal
- **population_CHANGED.py** -- TEEG_01 rev 2 section 3.3: ancestor plus 7 lines, an empty tonic-inhibition placeholder comment in cellsim; keep. Not a mechanical transform; legitimate because S0.1 commits it, making it the ancestor for everything downstream; content differs beyond line endings; ancestor LF, local CRLF (1784)

## 3. Discards

23 files. Payload discards move to the separate repository per D-8;
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
| Population/population_CHANGED.py                               | S0.2   | 872519706aef   | 82782      |
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

**16 shadowed pairs across 6 files.**

| file                                                     | name                               | defined at       | effective  | dead         |
|----------------------------------------------------------|------------------------------------|------------------|------------|--------------|
| Dendrite F score/morpholgy_pathways (6).py               | calculate_dendrite_xy_anisotropy   | L25, L376        | L376       | L25          |
| Dendrite F score/morpholgy_pathways (6).py               | export_neuron_to_hoc               | L842, L1331      | L1331      | L842         |
| Dendrite F score/morpholgy_pathways (6).py               | align_neurons_to_neighborhood      | L973, L1448      | L1448      | L973         |
| Dendrite F score/morpholgy_pathways (6).py               | label_dendritic_spines_robust      | L1653, L2353     | L2353      | L1653        |
| Dendrite F score/morpholgy_pathways (6).py               | plot_color_coded_neurons           | L1826, L2444     | L2444      | L1826        |
| Dendrite F score/morpholgy_pathways (6).py               | generate_smooth_tube               | L2573, L2783     | L2783      | L2573        |
| Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py | _interp_to_grid                    | L2462, L4694     | L4694      | L2462        |
| Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py | _simulate_square_subthreshold      | L2505, L4706     | L4706      | L2505        |
| Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py | _interp_to_grid                    | L2462, L4694     | L4694      | L2462        |
| Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py | _simulate_square_subthreshold      | L2505, L4706     | L4706      | L2505        |
| Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py | _interp_to_grid                    | L2462, L4694     | L4694      | L2462        |
| Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py | _simulate_square_subthreshold      | L2505, L4706     | L4706      | L2505        |
| Passive Features/HPC script/passive_fitting_hpc_fixed.py | _interp_to_grid                    | L2462, L3963     | L3963      | L2462        |
| Passive Features/HPC script/passive_fitting_hpc_fixed.py | _simulate_square_subthreshold      | L2505, L3975     | L3975      | L2505        |
| Rec_Utility.py                                           | plot_soma_skeleton                 | L1019, L1961     | L1961      | L1019        |
| Rec_Utility.py                                           | find_stable_soma_centroid          | L1113, L2055     | L2055      | L1113        |

## 5. Duplicate top-level class names

Recorded for awareness. Smoke-test assertion 6 acts on this at S0.9 and
must be scoped, or it can never pass while the passive sub-project and
the Colab scripts remain in the tree.

**26 names defined in more than one file.**

| class                        | copies | locations                                                                                        |
|------------------------------|--------|--------------------------------------------------------------------------------------------------|
| BootstrapCIResult            | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:4745; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:4745; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:4745; Passive Features/HPC script/passive_fitting_hpc_fixed.py:4014 |
| CellData                     | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:649; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:649; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:649; Passive Features/HPC script/passive_fitting_hpc_fixed.py:649 |
| CellSweepInput               | 2      | Passive Features/HPC script/Biological Fit/cm_profile_sweep.py:318; Passive Features/HPC script/Synthetic Passive fit Test/cm_profile_sweep.py:318 |
| CmProfile                    | 2      | Passive Features/HPC script/Biological Fit/cm_profile_sweep.py:88; Passive Features/HPC script/Synthetic Passive fit Test/cm_profile_sweep.py:88 |
| Connectomics                 | 3      | Classes/Connectomics/Class.py:6; HybridLFPy Tweaked/Extract macro-population data/Usage_example.py:6; HybridLFPy Tweaked/params_evoked_with_EEG_multiMorph.py:1212 |
| GpDiagnosticPerParameter     | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:4787; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:4787; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:4787; Passive Features/HPC script/passive_fitting_hpc_fixed.py:4056 |
| GpDiagnosticResult           | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:4802; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:4802; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:4802; Passive Features/HPC script/passive_fitting_hpc_fixed.py:4071 |
| IncompleteDataError          | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:612; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:612; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:612; Passive Features/HPC script/passive_fitting_hpc_fixed.py:612 |
| MorphoPath                   | 2      | HybridLFPy Tweaked/params_evoked_with_EEG_multiMorph.py:2369; Parames evoked with EEG/Params_evoked_with_EEG_multiMorph.py:1119 |
| OptimiserInputs              | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:729; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:729; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:729; Passive Features/HPC script/passive_fitting_hpc_fixed.py:729 |
| PassiveCell                  | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:1473; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:1473; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:1473; Passive Features/HPC script/passive_fitting_hpc_fixed.py:1473 |
| PassiveFitResult             | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:2287; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:2287; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:2287; Passive Features/HPC script/passive_fitting_hpc_fixed.py:2287 |
| PassiveSearchSpace           | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:680; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:680; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:680; Passive Features/HPC script/passive_fitting_hpc_fixed.py:680 |
| Phase2p5CellResult           | 3      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:3416; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:3416; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:3416 |
| Phase2p5GroupResult          | 3      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:3461; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:3461; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:3461 |
| Phase3Result                 | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:4811; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:4811; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:4811; Passive Features/HPC script/passive_fitting_hpc_fixed.py:4080 |
| Population                   | 2      | HybridLFPy Tweaked/Population_multiMorph.py:1138; Population/Population_multiMorph.py:919        |
| PopulationSuper              | 2      | HybridLFPy Tweaked/Population_multiMorph.py:45; Population/Population_multiMorph.py:41           |
| ReplotBundle                 | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:6295; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:6295; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:6295; Passive Features/HPC script/passive_fitting_hpc_fixed.py:5482 |
| SweepBundle                  | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:630; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:630; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:630; Passive Features/HPC script/passive_fitting_hpc_fixed.py:630 |
| TopoPopulation               | 2      | HybridLFPy Tweaked/Population_multiMorph.py:1701; Population/Population_multiMorph.py:1931       |
| _SkoptDimLite                | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:6278; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:6278; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:6278; Passive Features/HPC script/passive_fitting_hpc_fixed.py:5465 |
| _TrainBundleLite             | 4      | Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py:6257; Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py:6257; Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py:6257; Passive Features/HPC script/passive_fitting_hpc_fixed.py:5444 |
| general_params               | 2      | HybridLFPy Tweaked/params_evoked_with_EEG_multiMorph.py:178; Parames evoked with EEG/Params_evoked_with_EEG_multiMorph.py:178 |
| multicompartment_params      | 2      | HybridLFPy Tweaked/params_evoked_with_EEG_multiMorph.py:824; Parames evoked with EEG/Params_evoked_with_EEG_multiMorph.py:744 |
| point_neuron_network_params  | 2      | HybridLFPy Tweaked/params_evoked_with_EEG_multiMorph.py:525; Parames evoked with EEG/Params_evoked_with_EEG_multiMorph.py:445 |

## 6. Full ledger

Hash-chain columns: `pre` is measured now. `s02`, `s03`, `s04` are
filled by rerunning this tool after each of those sub-steps. A dash means
the sub-step has not run yet. S0.4 must leave `s04` equal to `s03`:
that equality IS the exit test for the move.

| path                                                                         | scope         | stage  | verdict  | pre           | s02  | s03  | s04  |
|------------------------------------------------------------------------------|---------------|--------|----------|---------------|------|------|------|
| Align Morphologies/Alignment.py                                              | colab         | S0.2   | retain   | 153229364d93  | -    | -    | -    |
| Align Morphologies/Alignment_IdentifySpines.py                               | colab         | -      | retain   | 00ea21aaec70  | -    | -    | -    |
| Align Morphologies/README.md                                                 | colab         | -      | retain   | 6abfac74ce73  | -    | -    | -    |
| Align Morphologies/Usage example.py                                          | colab         | -      | retain   | 2f310fe50363  | -    | -    | -    |
| Alignment Metadata/Extract_metadata.py                                       | colab         | -      | retain   | 1a462fe0a7c1  | -    | -    | -    |
| Alignment Metadata/README.md                                                 | colab         | -      | retain   | 3a9ff3627f65  | -    | -    | -    |
| Alignment Metadata/Usage.py                                                  | colab         | -      | retain   | 96d6e6a72c39  | -    | -    | -    |
| Connectomics/Calculate Relative Subpop % /Usage_example.py                   | colab         | -      | retain   | 7f95739f273f  | -    | -    | -    |
| Connectomics/Calculate Relative Subpop % /calculate_bbp_relative_presences.py | colab         | -      | retain   | d10121eb3f7e  | -    | -    | -    |
| Connectomics/Construct_ADJ/Construct_ADJ.py                                  | colab         | -      | retain   | 550f1d89a21e  | -    | -    | -    |
| Connectomics/Construct_ADJ/Microcircuit Assembly Documentation.pdf           | colab         | S0.7   | discard  | dec69a3bdff9  | -    | -    | -    |
| Connectomics/Construct_ADJ/Plotting_function                                 | colab         | -      | retain   | 1af3295bfe7a  | -    | -    | -    |
| Connectomics/Construct_ADJ/Usage_example.py                                  | colab         | -      | retain   | 5aebc98a68f6  | -    | -    | -    |
| Connectomics/Full_impementation                                              | colab         | -      | retain   | e5f5fdef748d  | -    | -    | -    |
| Connectomics/Generate Point Microcolumn/Population Generation Documentation (1).pdf | colab         | S0.7   | discard  | be7956c4b022  | -    | -    | -    |
| Connectomics/Generate Point Microcolumn/Usage_example.py                     | colab         | -      | retain   | 4c1226dcdce8  | -    | -    | -    |
| Connectomics/Generate Point Microcolumn/generate_microcolumn_cells.py        | colab         | -      | retain   | d2ce64546d55  | -    | -    | -    |
| Dendrite F score/README.md                                                   | colab         | -      | retain   | f2eae84356fd  | -    | -    | -    |
| Dendrite F score/morpholgy_pathways (6).py                                   | colab         | -      | retain   | bd8cf4b341be  | -    | -    | -    |
| Fetch Synapses/Fetch_MappedSyn.py                                            | colab         | -      | retain   | 81b9fef26511  | -    | -    | -    |
| Fetch Synapses/README.md                                                     | colab         | -      | retain   | d98bf970cbb7  | -    | -    | -    |
| Fetch Synapses/Usage_Example.py                                              | colab         | -      | retain   | eeb2aa856814  | -    | -    | -    |
| Final Implementation/README.md                                               | colab         | -      | retain   | 15b82ff6e8cd  | -    | -    | -    |
| Jitter Neuron/README.md                                                      | colab         | -      | retain   | a09b36fc6e2a  | -    | -    | -    |
| Jitter Neuron/jitter_neurons (1).py                                          | colab         | -      | retain   | aaf2bbe87751  | -    | -    | -    |
| Morphology Compliance/Compliance_Check.py                                    | colab         | S0.2   | retain   | babeb0342594  | -    | -    | -    |
| Morphology Compliance/Morphology Selection Documentation.pdf                 | colab         | S0.7   | discard  | 2acac69e5047  | -    | -    | -    |
| Morphology Compliance/Usage example                                          | colab         | -      | retain   | 2f366ba474ee  | -    | -    | -    |
| Nuova cartella compressa.zip                                                 | colab         | S0.7   | discard  | 9da5198cce44  | -    | -    | -    |
| Point Neuronal Network/Utility Functions/HDF5_Builder.py                     | colab         | -      | retain   | 2bc3d3328419  | -    | -    | -    |
| README.md                                                                    | colab         | -      | retain   | 648aec5644fb  | -    | -    | -    |
| Rec_Utility.py                                                               | colab         | -      | retain   | f13de0459687  | -    | -    | -    |
| Save nids/Usage_example.py                                                   | colab         | -      | retain   | 7f3657845a84  | -    | -    | -    |
| Save nids/save_indicies.py                                                   | colab         | -      | retain   | ea8eb268821e  | -    | -    | -    |
| Spanning Trees/Calculate_field.py                                            | colab         | -      | retain   | a481c7f3cdca  | -    | -    | -    |
| Spanning Trees/Neuronal Mapping Documentation.pdf                            | colab         | S0.7   | discard  | 14240ebdd440  | -    | -    | -    |
| Spanning Trees/Save.py                                                       | colab         | -      | retain   | 5c8b29eaeef0  | -    | -    | -    |
| Spanning Trees/Usage_example.py                                              | colab         | -      | retain   | 5e7a16b72697  | -    | -    | -    |
| Spanning tree overlap and synapse identification/Peters Rule Connectivity Engine Documentation.pdf | colab         | S0.7   | discard  | a44a1660a019  | -    | -    | -    |
| Spanning tree overlap and synapse identification/Synaptic Overlap Debugger Documentation.pdf | colab         | S0.7   | discard  | 7a7a3b5a8d8a  | -    | -    | -    |
| Spanning tree overlap and synapse identification/Usage_Example.py            | colab         | -      | retain   | 0d5e17d101cf  | -    | -    | -    |
| Spanning tree overlap and synapse identification/calculate_synaptic_overlap.py | colab         | -      | retain   | 88206f979839  | -    | -    | -    |
| Spine Detection/DEBUG/Generate_smooth_surface.py                             | colab         | -      | retain   | c792fbfc66e4  | -    | -    | -    |
| Spine Detection/DEBUG/Label_spines.py                                        | colab         | -      | retain   | 7f30cf1710ed  | -    | -    | -    |
| Spine Detection/DEBUG/README.md                                              | colab         | -      | retain   | 2d17a2e83686  | -    | -    | -    |
| Spine Detection/DEBUG/Segment_spine.py                                       | colab         | -      | retain   | 3c63163f6c5e  | -    | -    | -    |
| Spine Detection/Label_Dendritic_Spines.py                                    | colab         | -      | retain   | 2c56299fde3a  | -    | -    | -    |
| Spine Detection/README.md                                                    | colab         | -      | retain   | 21224f75dad4  | -    | -    | -    |
| Synapse Retrival/Assess_quality.py                                           | colab         | -      | retain   | 5e7569a8ef88  | -    | -    | -    |
| Synapse Retrival/Extract_synapses.py                                         | colab         | S0.2   | retain   | 2ebf1ef9d730  | -    | -    | -    |
| Synapse Retrival/README.md                                                   | colab         | -      | retain   | d0354e2a0a56  | -    | -    | -    |
| Synapse Retrival/map_synapses_to_segments.py                                 | colab         | -      | retain   | 27b225da3a49  | -    | -    | -    |
| Synaptic Placement/Neural Simulation Documentation.pdf                       | colab         | S0.7   | discard  | d5300eb50e13  | -    | -    | -    |
| Synaptic Placement/synaptic_placement (9).py                                 | colab         | S0.2   | retain   | a068ccaba84d  | -    | -    | -    |
| TRANSCRIPTOMICS_PROTEOMICS/README.md                                         | colab         | -      | retain   | e2f89bc4b9d2  | -    | -    | -    |
| TRANSCRIPTOMICS_PROTEOMICS/transcriptomics_proteomics (1).py                 | colab         | -      | retain   | 24f7f063a7db  | -    | -    | -    |
| automated_reconstruction.py                                                  | colab         | S0.2   | retain   | 463690b33638  | -    | -    | -    |
| final_get_comparments_h10.py                                                 | colab         | S0.2   | retain   | f967b32cd895  | -    | -    | -    |
| utility.py                                                                   | colab         | S0.2   | retain   | a20f39eedd3b  | -    | -    | -    |
| LEDGER.md                                                                    | infrastructure | S0.0   | new      | 5075a5ab0a05  | -    | -    | -    |
| ledger.csv                                                                   | infrastructure | S0.0   | new      | 8f2951bdd603  | -    | -    | -    |
| tools/ancestors.json                                                         | infrastructure | S0.0   | new      | 3a0672db37f8  | -    | -    | -    |
| tools/build_ledger.py                                                        | infrastructure | S0.0   | new      | afb19b68ac9a  | -    | -    | -    |
| tools/s0_ledger/__init__.py                                                  | infrastructure | S0.0   | new      | 7ec4151c2327  | -    | -    | -    |
| tools/s0_ledger/analyse.py                                                   | infrastructure | S0.0   | new      | 8cb68f8e03ad  | -    | -    | -    |
| tools/s0_ledger/render.py                                                    | infrastructure | S0.0   | new      | 5ae8d7abf0ce  | -    | -    | -    |
| tools/s0_ledger/scan.py                                                      | infrastructure | S0.0   | new      | 6d8862fa2ea6  | -    | -    | -    |
| tools/test_s0_ledger_smoke.py                                                | infrastructure | S0.0   | new      | 57eae0d12bc9  | -    | -    | -    |
| Classes/Connectomics/Class.py                                                | orchestrator  | S0.4   | keep     | 2f58a6ae79a3  | -    | -    | -    |
| Classes/Connectomics/Connectomics Class Overview.pdf                         | orchestrator  | S0.7   | discard  | 3cea1baed2e4  | -    | -    | -    |
| Classes/Connectomics/connectivity_buildup (9).py                             | orchestrator  | S0.4   | keep     | e62667fa6e57  | -    | -    | -    |
| HybridLFPy Tweaked/Extract macro-population data/Extract_macropopulation.py  | orchestrator  | S0.4   | keep     | 0bfb98e308b9  | -    | -    | -    |
| HybridLFPy Tweaked/Extract macro-population data/Project README.pdf          | orchestrator  | S0.7   | discard  | 95a6e04fd1bd  | -    | -    | -    |
| HybridLFPy Tweaked/Extract macro-population data/Usage_example.py            | orchestrator  | S0.4   | keep     | dc4828bb65d4  | -    | -    | -    |
| HybridLFPy Tweaked/Population_multiMorph.py                                  | orchestrator  | S0.4   | keep     | 35dbec718a35  | -    | -    | -    |
| HybridLFPy Tweaked/README.md                                                 | orchestrator  | S0.4   | keep     | 8183e1b9cd57  | -    | -    | -    |
| HybridLFPy Tweaked/Utility_function.py                                       | orchestrator  | S0.4   | keep     | ec55afe6c0c1  | -    | -    | -    |
| HybridLFPy Tweaked/hybrid_sim_evoked_with_EEG_multiMorph.py                  | orchestrator  | S0.4   | keep     | 566b25332991  | -    | -    | -    |
| HybridLFPy Tweaked/params_evoked_with_EEG_multiMorph.py                      | orchestrator  | S0.4   | keep     | 927ee3d9cc0d  | -    | -    | -    |
| Parames evoked with EEG/Params_evoked_with_EEG_multiMorph.py                 | orchestrator  | S0.4   | keep     | de9169ec67d8  | -    | -    | -    |
| Population/Population_multiMorph.py                                          | orchestrator  | S0.4   | keep     | 78ce9475d76c  | -    | -    | -    |
| Population/README.md                                                         | orchestrator  | S0.4   | keep     | ca11825113c5  | -    | -    | -    |
| Population/population_CHANGED.py                                             | orchestrator  | S0.2   | discard  | 872519706aef  | -    | -    | -    |
| Utility_fun.py                                                               | orchestrator  | S0.1   | keep     | 9d3d29c8a997  | -    | -    | -    |
| hybrid_sim_evoked_with_EEG_CHANGED.py                                        | orchestrator  | S0.1   | keep     | 6c3ed490ba14  | -    | -    | -    |
| params_evoked_with_EEG_CHANGED.py                                            | orchestrator  | S0.1   | keep     | 927ee3d9cc0d  | -    | -    | -    |
| population_CHANGED.py                                                        | orchestrator  | S0.1   | keep     | 43df1344230e  | -    | -    | -    |
| Passive Features/Allen Institute Data/README.md                              | passive       | -      | retain   | 954aa51240c9  | -    | -    | -    |
| Passive Features/Allen Institute Data/save_alleninstitute_data (6).py        | passive       | S0.2   | retain   | 9d8ccf9f5698  | -    | -    | -    |
| Passive Features/Allen Institute Data/trace_qc (1).py                        | passive       | -      | retain   | 5d983eb7cc91  | -    | -    | -    |
| Passive Features/Allen Institute Data/trace_qc_v2.py                         | passive       | -      | retain   | 3baf0064e73f  | -    | -    | -    |
| Passive Features/HPC script/Biological Fit/Biological_Passive_Fit_HPC_Guide.md | passive       | -      | retain   | d5294b571c9b  | -    | -    | -    |
| Passive Features/HPC script/Biological Fit/README.md                         | passive       | -      | retain   | 1867dde1be4f  | -    | -    | -    |
| Passive Features/HPC script/Biological Fit/cm_profile_sweep.py               | passive       | -      | retain   | 236b3827d6fd  | -    | -    | -    |
| Passive Features/HPC script/Biological Fit/passive_fitting_hpc_fixed.py      | passive       | -      | retain   | 6c95c3accb42  | -    | -    | -    |
| Passive Features/HPC script/Biological Fit/passive_long_step_training.py     | passive       | -      | retain   | 845c4459dbbb  | -    | -    | -    |
| Passive Features/HPC script/Biological Fit/run_biological_fit.py             | passive       | -      | retain   | 97f67b587144  | -    | -    | -    |
| Passive Features/HPC script/Biological Fit/smoke_run_biological_fit.py       | passive       | -      | retain   | 9f2f46440e03  | -    | -    | -    |
| Passive Features/HPC script/Biological Fit/submit_all_groups_biological.sh   | passive       | -      | retain   | 8c3fd53e2abe  | -    | -    | -    |
| Passive Features/HPC script/Biological Fit/submit_biological_fit.sh          | passive       | -      | retain   | da07d3283977  | -    | -    | -    |
| Passive Features/HPC script/Error_Log.txt                                    | passive       | -      | retain   | 042dc06898fa  | -    | -    | -    |
| Passive Features/HPC script/Multiple Sweeps with phase 2.5/README.md         | passive       | -      | retain   | 6e924b9543d3  | -    | -    | -    |
| Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed.py | passive       | -      | retain   | 6c95c3accb42  | -    | -    | -    |
| Passive Features/HPC script/Multiple Sweeps with phase 2.5/passive_fitting_hpc_fixed_description.md | passive       | -      | retain   | ea52844dec91  | -    | -    | -    |
| Passive Features/HPC script/Multiple nodes/README.md                         | passive       | -      | retain   | cd1425d8e2bc  | -    | -    | -    |
| Passive Features/HPC script/Multiple nodes/submit_all_groups.sh              | passive       | -      | retain   | 57ddbfda6712  | -    | -    | -    |
| Passive Features/HPC script/Multiple nodes/submit_passive_fit.sh             | passive       | -      | retain   | 94cf1cd080ab  | -    | -    | -    |
| Passive Features/HPC script/README.md                                        | passive       | -      | retain   | b3662fa0410a  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/Ih.mod                | passive       | -      | retain   | c989a36d9730  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/README.md             | passive       | -      | retain   | 8a29244b39ce  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/aggregate_synth_results.py | passive       | -      | retain   | d995c1b46af7  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/benchmark_usage_guide.md | passive       | -      | retain   | 9dfd962d4bc6  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/cm_profile_sweep.py   | passive       | -      | retain   | 236b3827d6fd  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/gen_from_manifest.py  | passive       | -      | retain   | bdf4f6d72683  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/kv.mod                | passive       | -      | retain   | c97ebc7abff9  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/na.mod                | passive       | -      | retain   | e8fda3334266  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/passive_consistency_diagnostic.py | passive       | -      | retain   | 844c92272e11  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/passive_fitting_hpc_fixed.py | passive       | -      | retain   | 6c95c3accb42  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/passive_long_step_training.py | passive       | -      | retain   | 845c4459dbbb  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/run_synth_benchmark.py | passive       | -      | retain   | 2f62abb5249f  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/smoke_aggregate_synth_results.py | passive       | -      | retain   | 2d27a88355ca  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/smoke_gen_from_manifest.py | passive       | -      | retain   | eb9f612cdc94  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/smoke_run_synth_benchmark.py | passive       | -      | retain   | 769ea0d8f7d2  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/smoke_synth_gt_grid.py | passive       | -      | retain   | e8787fd8fe28  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/submit_all_cohorts.sh | passive       | -      | retain   | 1bca07408152  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/submit_synth_benchmark.sh | passive       | -      | retain   | 0bf477ce6ab6  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/synth_gt_grid.py      | passive       | -      | retain   | 495d2c4eb558  | -    | -    | -    |
| Passive Features/HPC script/Synthetic Passive fit Test/synthetic_ground_truth.py | passive       | -      | retain   | fd1bebcaba5b  | -    | -    | -    |
| Passive Features/HPC script/passive_fit.e1099100                             | passive       | S0.7   | discard  | 856f6afb4ea7  | -    | -    | -    |
| Passive Features/HPC script/passive_fit.e1099156                             | passive       | S0.7   | discard  | 3c03fe908847  | -    | -    | -    |
| Passive Features/HPC script/passive_fit.e1099215                             | passive       | S0.7   | discard  | 87501536b336  | -    | -    | -    |
| Passive Features/HPC script/passive_fit.o1099100                             | passive       | S0.7   | discard  | 2c2f8bc9b1e7  | -    | -    | -    |
| Passive Features/HPC script/passive_fit.o1099156                             | passive       | S0.7   | discard  | ef17cc6efc8e  | -    | -    | -    |
| Passive Features/HPC script/passive_fit.o1099215                             | passive       | S0.7   | discard  | c69560156a4e  | -    | -    | -    |
| Passive Features/HPC script/passive_fitting_hpc_fixed.py                     | passive       | -      | retain   | 366381884ca2  | -    | -    | -    |
| Passive Features/HPC script/requirements.txt                                 | passive       | -      | retain   | f0fbc7ed8c52  | -    | -    | -    |
| Passive Features/HPC script/submit_passive_fit.sh                            | passive       | -      | retain   | aaf476707ddc  | -    | -    | -    |
| Passive Features/HPC script/test_pickle_roundtrip.py                         | passive       | -      | retain   | 4f9e385f9bad  | -    | -    | -    |
| Passive Features/Passive Allen Data/L2/README.md                             | passive       | -      | retain   | 3bb5a022d1b9  | -    | -    | -    |
| Passive Features/Passive Allen Data/L2/candidates.csv                        | passive       | -      | retain   | 8dc7a8e93c70  | -    | -    | -    |
| Passive Features/Passive Allen Data/L2/manifest.json                         | passive       | -      | retain   | bb58c6136861  | -    | -    | -    |
| Passive Features/Passive Allen Data/L2/specimen_528706755.zip                | passive       | S0.7   | discard  | 579910a148da  | -    | -    | -    |
| Passive Features/Passive Allen Data/L2/specimen_537204107.zip                | passive       | S0.7   | discard  | 01bff4f88db8  | -    | -    | -    |
| Passive Features/Passive Allen Data/L2/specimen_614659629.zip                | passive       | S0.7   | discard  | e8ceaafe3743  | -    | -    | -    |
| Passive Features/Passive Allen Data/L2/specimen_616647103.zip                | passive       | S0.7   | discard  | f4219ebd7149  | -    | -    | -    |
| Passive Features/Phase 1/README.md                                           | passive       | -      | retain   | 99e89cb13b59  | -    | -    | -    |
| Passive Features/Phase 2/aaa.py                                              | passive       | -      | retain   | 17e682f060b5  | -    | -    | -    |
| Passive Features/Phase 2/phase2_patch.pdf                                    | passive       | S0.7   | discard  | 2d488e5e1300  | -    | -    | -    |
| Passive Features/Phase 2/phase2_technical.pdf                                | passive       | S0.7   | discard  | 03d8016e1289  | -    | -    | -    |
| Passive Features/Phase 3/README.md                                           | passive       | -      | retain   | 4f9b9c11f112  | -    | -    | -    |
| Passive Features/Plot/X                                                      | passive       | -      | retain   | 10c7dbb397fc  | -    | -    | -    |
| Passive Features/Plot/passive_result_plot (5).py                             | passive       | S0.2   | retain   | a6b24bc1a3a6  | -    | -    | -    |
| Passive Features/Plot/phase3_publication_plots_v2_documentation.md           | passive       | -      | retain   | c366f5dc493d  | -    | -    | -    |
| Passive Features/node_test.py                                                | passive       | -      | retain   | f1c9580c2604  | -    | -    | -    |
| Passive Features/phase1fittingcolab (2).py                                   | passive       | S0.2   | retain   | b47f1f9e552e  | -    | -    | -    |
| Passive Features/submit_node_test.sh                                         | passive       | -      | retain   | f2329cb6a4be  | -    | -    | -    |

Full detail, including rationale for every row, is in `ledger.csv`.
