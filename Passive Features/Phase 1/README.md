# Phase 1 Function Reference
## `phase1_data_loader.py` — Human Cortical Passive-Property Pipeline

**Human Cortical Microcircuit Modelling Project** | Internal documentation

---

## Overview and purpose

Phase 1 is a data-preparation and model-construction layer that sits between raw experimental databases and the Bayesian optimiser in Phase 2. Its job is deceptively simple to state — *give the optimiser a cell morphology, an experimental voltage trace to fit, and a second trace to validate against* — but the path from Allen Institute recordings to a NEURON simulation ready for Gaussian-process optimisation involves a long chain of biophysical, numerical, and software-engineering decisions, every one of which has a scientific reason behind it. This document explains those decisions function by function.

A useful mental model before reading: think of Phase 1 as building two parallel representations of the same neuron. One representation lives in the experimental world: voltage traces measured in a slice with a patch-clamp electrode. The other lives in the simulation world: a multi-compartmental NEURON model whose geometry comes from the same cell's 3-D reconstruction. Phase 2's job is then to find the three cable parameters (Cm, Rm, Ra) that make the simulated cell's voltage response match the experimental one as closely as possible. Phase 1 makes both representations and connects them so Phase 2 can do the comparison cleanly.

---

## Data model — the dataclasses

Before discussing functions it is worth understanding the data structures they pass around, because the design choices in the dataclasses constrain everything else.

### `SweepBundle`

A `SweepBundle` is the fundamental unit of experimental data in Phase 1. It represents one **averaged voltage transient** measured at the soma in response to one category of stimulus — a single averaged pulse in the case of Square Subthreshold, or a single averaged current step in the case of Long Square.

Every `SweepBundle` carries six arrays and several scalar annotations. The most important fields are `t` (time in seconds from pulse onset), `v_mV` (LJP-corrected somatic membrane potential in millivolts), and `i_pA` (injected current in picoamperes). The `polarity` field is either `"dep"` or `"hyp"`, recording whether this bundle's stimulus was depolarising or hyperpolarising — a distinction that matters because hyperpolarising pulses activate less Ih and produce cleaner passive transients. The `amplitude_pA` field stores the baseline-corrected step amplitude (not the raw sweep metadata, which is often None in Allen's database), and `n_repeats_averaged` records how many raw pulses or sweeps were averaged into this bundle, giving downstream code a sense of the statistical confidence of the trace.

The reason for building this intermediate dataclass rather than passing NumPy arrays directly is reproducibility: a `SweepBundle` is self-describing. Six months after a fit, you can read the `sweep_numbers` field and know exactly which Allen sweeps contributed, and check `n_repeats_averaged` to judge the SNR of the bundle.

### `CellData`

`CellData` is the complete data package for one Allen specimen. It holds the SWC morphology path, two lists of `SweepBundle` objects (one for Square Subthreshold training data, one for Long Square validation data), and three reference scalars (Rin, τm, Vrest) read from Allen's Cell Feature Summary. The scalar features were computed by Allen's own vetted feature-extraction code, so they serve as the ground-truth validation targets even when the Long Square waveform is unavailable.

Separating the two sweep lists into named fields (`square_subthreshold` vs. `long_square_subthreshold`) rather than a single generic list makes the purpose of each bundle explicit. Square Subthreshold bundles are short-timescale capacitive transients used for fitting. Long Square bundles are long-timescale step responses used for post-fit validation of Rin and τm. Mixing them into a single list would invite silent errors in Phase 2.

### `PassiveSearchSpace` and `OptimiserInputs`

`PassiveSearchSpace` encodes the three-dimensional parameter space the Bayesian optimiser will search. Its bounds come directly from the project pipeline document: Cm ∈ [0.3, 3.0] µF/cm², Rm ∈ [1 000, 100 000] Ω·cm², Ra ∈ [50, 1 000] Ω·cm. The prior for Rm is `log-uniform` rather than uniform because Rm spans two orders of magnitude and its effect on the somatic transient is highly non-linear — concentrating more evaluations in the lower-Rm region where most human neurons live makes the optimiser significantly more efficient.

`OptimiserInputs` bundles the search space dimensions, the training bundles, the validation bundle, and the reference scalars into a single object that Phase 2 can consume without needing to know anything about how the data was loaded. This clean handoff is deliberate: if Phase 2 is eventually run on an HPC cluster where the Allen data is not accessible, the `OptimiserInputs` object is completely self-contained.

### `IncompleteDataError`

This is a custom exception (not just a `ValueError`) because callers need to distinguish "this cell has bad data" from "there is a bug in the code". Catching `IncompleteDataError` in the batch loader is a normal, expected control-flow event; catching `ValueError` or `RuntimeError` is not. The exception also carries the partially-constructed `CellData` object as an attribute, so a caller that catches it can still log what was found before giving up.

---

## Cell discovery — `list_human_cells_with_morphology()`

### What it does

This function queries the Allen Cell Types Database and returns a Pandas DataFrame whose rows are candidate specimens. It applies five filters in sequence: species (human only), morphological reconstruction (required), cortical layer (optional), dendrite type (spiny/aspiny, optional), and protocol availability (Square Subthreshold and Long Square, optional). An optional sixth filter applies Patch-seq transcriptomic subtype (SST/PV/VIP) if a mapping CSV is supplied.

### Why it is structured this way

The most important design decision is the order of filters. Layer and dendrite-type filtering happens first, using fast in-memory operations on the DataFrame. Protocol availability filtering comes last because it requires one network round-trip per candidate cell (one call to `ctc.get_ephys_sweeps()`). Checking all 158 human cells for protocol availability before filtering by layer would make 158 network calls; checking only the 13 layer-4-spiny cells makes 13. This makes the function usable interactively without painful waiting.

The protocol availability check works by scanning the `stimulus_name` field of each sweep and testing whether any name contains both "Square" and "Subthreshold" but not "Long Square". This string-based matching was chosen over a more precise duration-or-amplitude check because Allen has used at least three different name strings for the same 0.5 ms subthreshold protocol across SDK releases ("Square - 0.5ms Subthreshold", "Square Subthreshold", "Short Square - Subthreshold"), making string matching more robust than trying to filter by nominal duration.

The Patch-seq subtype overlay is deliberately optional, loaded from a user-supplied CSV rather than queried automatically, because the Allen Cell Types API does not expose transcriptomic identities (SST/PV/VIP) directly for human cells. Human tissue comes from surgical resection, not from Cre-driver lines, so Cre-line metadata is meaningless. Genetic identity is only available for cells that received simultaneous Patch-seq processing, which is a subset. Cells without Patch-seq data are returned with `subtype = "unknown"` rather than being silently dropped — you never want to discard real experimental data just because a metadata field is absent.

The `n_cells` parameter truncates the output DataFrame. Its purpose is practical: during Colab development, iterating over all 7 layer-6-spiny cells takes long enough to interrupt the thinking flow. Setting `n_cells=3` lets you test the full pipeline quickly.

---

## The internal unit normaliser — `_to_pA_seconds()`

This small helper converts Allen sweep metadata amplitude and duration values to a consistent (pA, seconds) representation. It exists because Allen's ephys_sweeps metadata has been returned in two different unit systems across SDK releases: Amperes and seconds (SI), and picoamperes and milliseconds (legacy). The heuristic for detecting which convention is in use is: if `|amplitude| < 1×10⁻³`, the value is in Amperes (a 200 pA stimulus would be `2×10⁻¹⁰ A`, well below the threshold); otherwise it is already in pA. The duration heuristic is symmetric: if the numerical value is below 10, it is in seconds; above 10, it is in milliseconds.

Crucially, the function returns `(NaN, NaN)` for any `None`, non-numeric, or already-NaN input, rather than raising an exception. This is the correct contract for a normalisation helper: it tells the caller "this metadata is missing" and lets the caller decide what to do. Every call site checks `np.isnan()` on the result before using it. The alternative — raising an exception — would have caused the entire sweep-selection loop to abort the first time any test sweep (which always has `stimulus_amplitude = None`) appeared in the metadata.

---

## Square Subthreshold sweep selection — `_select_square_subthreshold()`

### What it does

Returns every sweep whose `stimulus_name` contains both "Square" and "Subthreshold" but not "Long Square". That is all it does — no amplitude or duration filtering at the metadata level.

### Why the design is so minimal

This minimalism is the result of an important structural discovery about Allen's protocol. The Square Subthreshold protocol delivers all 20 brief pulses **inside a single sweep** (see Allen Technical White Paper, Appendix p.15: *"0.5 ms square current injections to +/- 200 pA, repeated 20 times (200 ms intervals). N/A (single sweep)"*). The sweep's metadata `stimulus_duration` therefore describes the total recording duration (~4–7 s), not the duration of any individual pulse. Similarly, `stimulus_amplitude` for a sweep that contains alternating +200 pA and −200 pA pulses is either undefined or approximately zero. Any amplitude- or duration-based filter applied to the per-sweep metadata would therefore reject every Square Subthreshold sweep that exists — which is exactly the bug that was present in earlier versions of this code and caused `cd.square_subthreshold` to always be empty.

The correct approach is to identify candidate sweeps by name (robust, version-independent) and then detect the individual pulses by inspecting the actual current waveform. That is what `_detect_pulses_in_current()` and `_build_subthreshold_bundles()` do.

---

## Long Square sweep selection — `_select_long_square_subthreshold()`

This function returns every sweep whose name contains "Long Square", with no amplitude filtering. Unlike Square Subthreshold, each Long Square sweep does carry a single commanded amplitude — but `stimulus_amplitude` in the metadata is frequently `None` even for Long Square sweeps in many Allen SDK releases. The amplitude is therefore measured from the actual NWB waveform by `_detect_step_amplitude()` during loading.

---

## Step detection — `_detect_step_amplitude()`

### What it does

Given a full Long Square current waveform in pA, this function finds the commanded step amplitude. It subtracts a baseline (the median of the first 5% of the trace, which is guaranteed by Allen's QC to be pre-stimulus time), thresholds on `|delta| > 5 pA`, finds the first and last samples above threshold, and returns the median current within that window. If no region of minimum duration 100 ms is found, it returns NaN.

### Why measure amplitude from the waveform

The earlier approach of averaging the middle 50% of the sweep suffered from a dilution problem: a 1-second step inside a 7-second recording occupies less than 15% of the trace, so the middle-50% mean is dominated by silent baseline periods and gives an amplitude estimate roughly one-third the true step size. For a genuine −10 pA step, the diluted estimate would be approximately −3 pA, which fell below the −5 pA detection cutoff and caused the step to be silently discarded. The waveform-based detector has no such limitation: it finds where the step actually is, then measures amplitude only inside that region, giving exact results at every amplitude tested.

The 5-second detection threshold (the minimum step duration) guards against accidentally detecting the short test pulse that Allen injects at the beginning of every data sweep to monitor series resistance. At 50 kHz sampling, 100 ms corresponds to 5,000 samples — far longer than any test pulse but shorter than the 1-second Long Square step.

---

## Brief pulse detection — `_detect_pulses_in_current()`

### What it does

Given the full current waveform from one Square Subthreshold sweep, this function returns a list of `(start_idx, end_idx, peak_pA_signed)` tuples — one per detected 0.5 ms pulse. It subtracts a baseline, thresholds at 50 pA (a comfortable margin below the nominal 200 pA pulse), finds contiguous above-threshold regions, rejects any shorter than 0.1 ms (guards against noise spikes), and records the signed peak amplitude within each region.

### Why baseline-subtract before thresholding

Allen applies a bias current throughout the recording to maintain the resting membrane potential. This bias current can be anywhere from zero to tens of pA. Without baseline subtraction, a +40 pA bias current would raise the apparent baseline and cause a −200 pA pulse to appear as −160 pA relative to zero — which would still clear the threshold, but a −50 pA bias current combined with a +200 pA depolarising pulse would give +150 pA apparent, possibly distorting the polarity detection. Subtracting the baseline as the first step makes the detection threshold scale-invariant relative to whatever holding current the experimenter was using.

---

## Window extraction — `_extract_windows_around_pulses()`

### What it does

For each detected pulse, this function cuts a window from 10 ms before the pulse onset to 200 ms after it, producing a `(t, v_mV, i_pA)` tuple where `t = 0` is exactly the pulse onset. Pulses whose windows would extend beyond the trace boundaries are silently skipped.

### Why 200 ms post-onset

The Eyal 2016 fitting window is 1–100 ms post-stimulus. The window is extended to 200 ms here to give Phase 2 flexibility: it can use 1–100 ms for fitting (matching Eyal exactly) while having the additional 100 ms of recovery for visual inspection or for computing the steady-state voltage. The 10 ms pre-pulse baseline is stored so Phase 2 can zero each trace to its own pre-pulse mean, which is more stable than relying on the LJP-corrected absolute voltage.

---

## Square Subthreshold bundle builder — `_build_subthreshold_bundles()`

### What it does and why it is the most complex function in Phase 1

This function is the heart of the Square Subthreshold data pipeline. It accepts a list of candidate sweeps, loads each from the NWB file, detects every brief pulse in the current waveform, applies a QC filter to each detected pulse (checking that its amplitude is within ±30 pA of 200 pA and its duration is within ±0.5 ms of 0.5 ms), classifies each passing pulse as depolarising or hyperpolarising based on the sign of its peak amplitude, groups all pulses by polarity across all sweeps, and then partitions and averages within each polarity group.

The reason for this multi-stage pipeline rather than simply averaging whole sweeps is that the Square Subthreshold protocol packs multiple pulses of opposite polarity into a single sweep. There is no per-sweep concept of "this sweep was depolarising" — every sweep contains both. The only way to separate the two polarities is by inspecting the current waveform pulse by pulse and classifying each one individually.

The `n_avg_groups` parameter controls how many separate averaged bundles are produced per polarity. With `n_avg_groups=1` (the default) all available pulses are averaged into one high-SNR trace. With `n_avg_groups=K`, the pool of pulses is split into K equal partitions and each is averaged independently, yielding K bundles. This is the bootstrap mechanism for Phase 2's parameter uncertainty estimation: K bootstrapped traces give K independent RMSD measurements, from which the empirical standard deviation of (Cm, Rm, Ra) can be estimated at negligible additional computational cost.

The per-pulse QC filter is important because `_select_square_subthreshold()` matches sweeps by name only. In rare cases, a sweep name containing "Square" and "Subthreshold" might correspond to a protocol with different parameters — a different amplitude or a different pulse duration recorded in a Core 2 session. The per-pulse QC catches these cases at the waveform level and excludes them, ensuring that only genuine 0.5 ms ±200 pA pulses contribute to the fit target.

---

## Long Square bundle builder — `_build_bundles_from_group()`

This function handles the simpler case where each sweep carries a single sustained current step. It loads all specified sweeps, uses `index_range` to exclude the test pulse at the start of each sweep, averages the voltage and current traces across sweeps, and detects the onset and amplitude of the step from the averaged current waveform. The result is a single `SweepBundle` with the full step-response trace from pre-step baseline through the 1-second step and into the post-step recovery.

The use of `index_range` (from the sweep dict) to clip each sweep is essential: Allen prepends a short test pulse to every data sweep for series-resistance monitoring, and including it in the average would contaminate the baseline estimate and offset the detected step onset.

---

## Main data loader — `load_allen_data()`

### What it does

This is the top-level function that orchestrates everything else for one specimen. It proceeds in five stages: metadata retrieval, morphology download, Square Subthreshold loading, Long Square loading, and Cell Feature Summary retrieval.

### Morphology download and SWC path

The function calls `ctc.get_reconstruction(specimen_id, file_name=str(swc_file))` rather than calling it without arguments and then trying to find the cached file path. This is because `CellTypesCache` does not expose its manifest path as a public attribute — an earlier version of the code tried `ctc.manifest_file` and crashed with `AttributeError`. Supplying an explicit `file_name` means we choose where the SWC lands and always know the path, regardless of AllenSDK's internal caching conventions.

### The LJP correction

Allen records voltages uncorrected for the −14 mV liquid junction potential (LJP) between the potassium gluconate internal solution and the ACSF bath (documented in Allen Cell Types Technical White Paper, v.5). All voltages in `CellData` are shifted by +14 mV at load time, bringing them into the LJP-corrected convention used by every published human neuron model (Eyal 2016, Rich 2021, Yao 2021). This correction is applied once, at the moment of loading, rather than scattered through downstream code. Applying it late — or inconsistently — would introduce a systematic −14 mV offset into e_pas in NEURON, causing the model's resting potential to be mismatched from the experimental one and making the RMSD meaningless.

### Long Square multi-amplitude loading

The function collects all hyperpolarising subthreshold sweeps, detects their amplitudes from the waveforms, groups them by amplitude (rounding to the nearest 10 pA to account for small biological variability in commanded levels), and builds one `SweepBundle` per distinct amplitude. The resulting list is sorted by increasing absolute amplitude so that `long_square_subthreshold[0]` is always the smallest step.

This design was chosen specifically to address the Ih contamination problem: larger hyperpolarising steps progressively deactivate more Ih, introducing an amplitude-dependent sag that a purely passive NEURON model cannot reproduce. By keeping all amplitudes, Phase 2 has the flexibility to use the smallest available step (least Ih activation, cleanest passive τm estimate) for its τm validation target, while using all amplitudes together for a linear Rin fit that mirrors Allen's own Rin computation methodology.

### The `require_complete_data` gate

When `require_complete_data=True` (the default), the function raises `IncompleteDataError` if either the Square Subthreshold or Long Square bundle list is empty. The scientific rationale is that a cell without Square Subthreshold data has no training target for the passive fit, and a cell without Long Square data has no held-out validation target. Allowing such cells to propagate to Phase 2 would produce fits with no meaningful quality control. Failing loudly at the end of loading — rather than returning a `CellData` with empty lists that silently fails later — makes the error surfaced immediately, at the exact cell that caused it, with a message naming which protocol was missing.

---

## Batch loader — `load_complete_cells()`

This convenience function iterates through the rows of a `list_human_cells_with_morphology()` DataFrame, calls `load_allen_data()` on each, catches `IncompleteDataError` and any unexpected exceptions, prints a one-line note for each dropped cell, and returns a list containing only the specimens that yielded complete data. The `max_cells` parameter performs an early exit once the desired number of complete cells is found, avoiding unnecessary downloads. Sharing a single pre-built `ctc` object across all cells in the loop avoids re-downloading the Allen cell metadata table on every call.

The typical Phase 2 entry point is:

```python
candidates = list_human_cells_with_morphology(layer="6", dendrite_type="spiny")
cells = load_complete_cells(candidates, max_cells=10)
opt_inputs = [prepare_optimiser_inputs(cd) for cd in cells]
```

---

## NEURON model construction — `PassiveCell` and `build_neuron_model()`

### Overview

`PassiveCell` builds a NEURON compartmental model from an Allen SWC morphology file and prepares it for repeated passive-parameter evaluation by the optimiser. The class is designed so that instantiation (which is expensive) happens once, and parameter evaluation (which must be fast, because gp_minimize calls it hundreds of times) is a cheap in-place update.

### SWC import — `_import_swc()`

NEURON's `Import3d_SWC_read` + `Import3d_GUI.instantiate(None)` reads the SWC file and creates NEURON sections in the running simulation. The sections are then categorised by name into soma, basal dendrites (`dend`), apical dendrites (`apic`), and axon using `_categorise_sections()`. This categorisation drives all subsequent per-compartment decisions.

### Axon replacement — `_replace_axon_hay_stub()`

The original axon reconstruction from the Allen SWC file is a short stump — typically just the axon initial segment visible under the microscope after slicing. This truncated axon is not biologically representative, and including it unchanged would make the model's somatic input impedance depend on a poorly-reconstructed structure. The standard solution (Hay et al. 2011, PLoS Comput Biol; used in essentially every published multi-compartmental human neuron model since) is to delete the original axon and replace it with two uniform cylindrical sections each 30 µm long and 1 µm in diameter, connected at `soma(1.0)`. This "Hay stub" provides a standardised axon that contributes a small but consistent capacitive load to the soma, making models from different cells comparable.

### Passive mechanism insertion — `_insert_passive()`

The NEURON built-in `pas` mechanism is a parallel RC circuit with conductance `g_pas` (S/cm²) and reversal potential `e_pas` (mV). Inserting it in every section gives each segment two membrane properties: the passive leak current and the membrane capacitance `cm`. These are the parameters the optimiser will vary.

### The F-factor multiplier — `_precompute_F_per_segment()`

The spine-area correction factor F accounts for the fact that dendritic spines contribute approximately 50% of total dendritic membrane area but are not represented in the SWC reconstruction (which traces only the dendritic shaft). Following Eyal et al. (2016), the effective membrane area of any dendritic segment more than 60 µm from the soma is increased by the factor F:

```
F = (shaft area + total spine area) / shaft area
```

In the NEURON model, this is implemented by multiplying `cm` by F and multiplying `g_pas` by F (equivalently, dividing Rm by F) for those segments. The cutoff of 60 µm reflects the observation (Benavides-Piccione et al. 2013) that spine density is very low in the proximal 60 µm of the dendritic tree.

The multiplier table is computed once at construction time by `_precompute_F_per_segment()` and stored in a dict keyed by `(section_name, segment_x)`. This means that when `set_passive()` is called inside the optimiser loop, looking up the multiplier for each segment takes microseconds rather than requiring a path-distance computation. For a cell with thousands of segments and potentially thousands of optimiser evaluations, this precomputation is essential for keeping the total fitting time manageable.

Aspiny compartments (soma, axon stub) receive multiplier = 1.0. Interneurons (aspiny throughout) receive F = 1.0 for all dendritic segments; for them the F parameter passed to `build_neuron_model()` should simply be set to 1.0.

### `set_passive(Cm, Rm, Ra)` — the inner loop

This method is called once per gp_minimize evaluation. It applies the three cable parameters uniformly across all compartments, automatically scaling dendritic segments by their precomputed F multiplier:

- `cm = Cm × F_multiplier` in each segment
- `g_pas = (1/Rm) × F_multiplier` in each segment
- `Ra = Ra` (uniform; spine correction does not affect axial resistance)

Because F is already baked into the per-segment lookup table, `set_passive()` contains no branching logic and no distance computations — just a loop over all sections and segments. The method does not set `e_pas`; that is handled separately by `set_e_pas(cd.v_rest_mV)` before each simulation, because the resting potential is a fixed experimental measurement, not a fitted parameter.

### `simulate(stim_amp_pA, stim_delay_ms, stim_dur_ms, tstop_ms, v_init_mV, dt_ms)`

This method runs one NEURON simulation with an IClamp at the soma centre and returns `(t_ms, v_mV)` arrays from the soma. NEURON's IClamp expects amplitude in nA; the conversion from pA (`amp_nA = amp_pA × 1e-3`) is done internally. The simulation is initialised with `h.v_init = v_rest_mV` to match the experimental pre-pulse potential, so the RMSD is computed on the correct baseline.

---

## Optimiser input assembly — `prepare_optimiser_inputs()`

### What it does

This function takes a `CellData` and returns an `OptimiserInputs` object that Phase 2 can consume without knowing anything about Allen Institute data formats or NEURON.

The `fit_target` parameter controls which Square Subthreshold bundle goes into `train_bundles`. The default is `"hyp"` — hyperpolarising sweeps only. This is not arbitrary: Eyal et al. (2016) showed that including Ih (which is partially active at rest) during passive fitting biases the estimated Cm upward from ~0.45 µF/cm² toward ~0.76 µF/cm². Hyperpolarising pulses deactivate some of the resting Ih rather than activating more, so the capacitive transient is slightly cleaner. The depolarising sweep (`"dep"`) or both (`"both"`) can be selected explicitly if the user wants to reproduce Eyal's original approach or use both polarities jointly.

The `long_square_validation_amp` parameter selects which Long Square bundle goes into `validation_bundles`. By default it picks `long_square_subthreshold[0]`, which is the smallest available amplitude (stored that way by `load_allen_data()`). Smaller amplitudes keep the membrane closer to Vrest, minimising Ih deactivation and giving the cleanest passive τm estimate. A larger amplitude (e.g. −90 pA) can be requested explicitly for higher SNR at the cost of more Ih contamination.

The `PassiveSearchSpace.as_skopt_dimensions()` method returns a list of three `Real` objects that scikit-optimize understands. Rm gets a `log-uniform` prior rather than uniform because at equal linear spacing, the Gaussian-process surrogate wastes most evaluations in the 50 000–100 000 Ω·cm² range while severely under-sampling the biologically realistic 10 000–30 000 Ω·cm² region.

---

## Visualisation — `plot_example_traces()`

This function produces a 2 × 2 matplotlib figure with voltage traces on the top row and current traces on the bottom row, Square Subthreshold in the left column and Long Square in the right column. It overlays the depolarising (red) and hyperpolarising (blue) averaged Square Subthreshold bundles, and draws the Vrest reference line as a dotted grey horizontal on both voltage panels.

The function exists primarily as a sanity-check tool: before running a 300-evaluation Gaussian-process optimisation on a cell, it is always worth visually confirming that the data loaded correctly, the LJP correction brought the baseline to a physiologically plausible voltage, the capacitive transient in the Square Subthreshold trace has the expected shape, and the Long Square step shows clean exponential recovery (with or without Ih sag, depending on the cell). A glance at the figure takes ten seconds and can save hours of debugging later.

If the cell has no Long Square data (because it failed the amplitude detection), the Long Square panels display "no Long Square data" in grey text rather than raising an exception. The same graceful handling applies if Square Subthreshold bundles are missing for one polarity.

---

## Complete call sequence

For reference, the expected order of calls in a typical Phase 2 preparation session is:

1. `list_human_cells_with_morphology(layer, dendrite_type, n_cells)` → `candidates` DataFrame
2. `load_complete_cells(candidates, max_cells=N)` → `List[CellData]`
3. For each `cd` in the list: `plot_example_traces(cd)` to visually confirm data quality
4. For each `cd`: `build_neuron_model(cd.swc_path, F=F_layer)` → `PassiveCell`
5. For each `cd`: `prepare_optimiser_inputs(cd, fit_target="hyp")` → `OptimiserInputs`
6. Pass each `(PassiveCell, OptimiserInputs)` pair to Phase 2's `gp_minimize` loop

---

## Design principles

Three principles run through every function in Phase 1.

**Fail loudly on bad data, return NaN for missing metadata.** Allen's database is large and maintained by many people across many years. Metadata fields are inconsistently populated, unit conventions have changed, and protocol names are unstable. Where a missing value represents normal database variability (e.g. `stimulus_amplitude = None` for a test pulse), the code returns NaN and continues. Where missing data means the cell is scientifically unusable for our purpose (no Square Subthreshold sweeps, no Long Square sweeps), the code raises a named exception. The distinction matters: `NaN` means "skip this sweep", `IncompleteDataError` means "skip this cell".

**Detect from waveforms, not from metadata.** Both the Square Subthreshold pulse polarity and the Long Square step amplitude are determined by inspecting the actual current waveform, not by trusting the per-sweep metadata fields. This robustness cost a significant debugging effort but means the code will continue to work correctly even if Allen changes their metadata conventions again.

**Precompute everything that is constant across optimiser evaluations.** The F-factor table, the section categorisation, and the morphology import all happen once at `PassiveCell` construction. The inner loop (`set_passive()` + `simulate()`) does only what must be done per evaluation. For a 300-evaluation Gaussian-process fit, this distinction can reduce wall time from hours to minutes on a laptop.

---

## Key references

- **Eyal G et al. (2016)** eLife 5:e16553 — passive-fit methodology; Cm ≈ 0.5 µF/cm² in human L2/3; F factor; nucleated-patch validation.
- **Hay E et al. (2011)** PLoS Comput Biol 7:e1002107 — axon-stub convention; parameter bounds for Rm and Ra.
- **Benavides-Piccione R et al. (2013)** Cereb Cortex 23:1798–1810 — F factor values (F_basal ≈ 1.89, F_apical ≈ 1.87 for 85-year-old human L2/3); spine-proximal cutoff of 60 µm.
- **Shapson-Coe A et al. (2024)** Science 384:eadk4858 — H01 EM connectome; layer-specific spine densities for L4 and L6.
- **Allen Cell Types Database Technical White Paper v5 (2017)** — Square Subthreshold protocol specification; Long Square protocol; −14 mV LJP; QC criteria.
- **Internal pipeline document** `passive_properties_summary.docx` — parameter bounds; fitting window; F application convention; fallback strategies.
