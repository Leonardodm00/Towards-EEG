# S1 modules

Four modules plus a plotting module and a test runner. **None of them changes
any existing output.** `spine_geometry`, `truncation_flag` and `s1_plots` are
pure reads; `syn_uid` adds one column to the synapse frame. Nothing in the
current export path depends on any of them.

## Files -- ALL of these go in CODE_DIR

Put every file below in the same folder as `spine_density.py`, `alignment.py`
and the rest, i.e.

    /content/drive/MyDrive/Colab Notebooks/New algorithms/Stage 1

| file | what it is |
|---|---|
| `spine_geometry.py` | per-spine geometry and neck axial conductance factor G(sigma) |
| `truncation_flag.py` | dendrite-tip EM-cut evidence: terminal taper + boundary proximity |
| `syn_uid.py` | stable content-addressed synapse identifiers + node-collapse audit |
| `s1_plots.py` | the six figures |
| `run_all_tests.py` | one-call gate: transfer safety, imports, compile, all four suites |
| `test_spine_geometry.py` | 53 checks |
| `test_syn_uid.py` | 29 checks |
| `test_truncation_flag.py` | 40 checks |
| `test_s1_plots.py` | 35 checks |
| `colab_s1_cells.py` | the notebook cells to paste (not a script to run) |
| `ancestors_s1_patch.json` | patch for the S0.0 ledger spec |

The four test files are needed by `run_all_tests.py`. If any are absent it
exits **4** and prints a directory listing plus closest-name matches.

## Run everything

```bash
python3 run_all_tests.py
```

Expected: `157 checks passed, 0 failed, across 4 suite(s) run` and `ALL GREEN`.

Exit codes: **0** green, **1** a real test failure, **3** a dependency
(`spine_density`, `node_classify`) is not importable, **4** a test file is
absent. Never 2 -- the interpreter itself uses 2 for "can't open file".

## Notebook cells

`colab_s1_cells.py` carries them in order:

    CELL 1   EDIT     add the new filenames to the upload/module list
    CELL 3b  NEW      after CELL 3 -- runs run_all_tests.py
    CELL 6   EDIT     three lines in the loop body (syn_uid)
    CELL 6c  NEW      after CELL 6 -- spine geometry + truncation
    CELL 8b  NEW      after CELL 8 -- the six figures
    CELL 9   REPLACE  full rewrite of the summary/manifest cell

CELL 6c reads `frames`, which CELL 6 leaves behind. If the kernel restarted,
re-run CELLs 4, 5 and 6 before it -- CELL 6c now says so explicitly instead
of raising a bare NameError.

## Run on real data

```bash
python3 test_spine_geometry.py neuron_4683651279_spines.csv
python3 test_truncation_flag.py neuron_A_spines.csv neuron_B_spines.csv ...
python3 test_syn_uid.py neuron_X_synapses_raw.csv neuron_X.csv
python3 test_s1_plots.py --save ./figs
```

## Before quoting any resistance

Check `phi_pipeline_colab.radius_report(df)["radius_suspect"]` first. It is the
authoritative radius gate and is stricter than anything in `spine_geometry`:
it assesses flatness on non-soma nodes only, so a cell whose entire dendrite
sits at the 50 nm fallback cannot pass on the strength of a distinct soma
radius. Pass it through:

```python
rq = phi_pipeline_colab.radius_report(df)
summ = sg.cell_spine_summary(spine_df, radius_report=rq)
# summ["resistance_trustworthy"] is False whenever radius_suspect is True
```

If that is False, every R_neck in the cell is a function of neck LENGTH alone
and means nothing -- and it will still land inside the published range, which
is exactly why the flag exists. Panel 4 of the `neck_geometry` figure shows
the same thing visually.

Also check `frac_with_neck`: the labeller assigns everything to 'head' when a
spine path has fewer than 3 distinct nodes, and those spines get G = 0. They
are excluded from the resistance statistics rather than counted as
zero-resistance, so a low fraction means the quantiles rest on a minority.

## The six figures, and what to look for

| figure | read it for |
|---|---|
| `s1_neck_geometry` | panel 4 a spike at 0.05 um -> stop, radii are synthetic |
| `s1_spine_profile` | peak far from ~90 um -> suspect the labeller |
| `s1_neck_resistance` | the SPREAD across rho_a -> how much is assumption |
| `s1_attenuation` | median kappa near 1.000 -> assumption A2 is close to exact |
| `s1_truncation` | panel 1 bimodal -> the taper threshold is calibratable |
| `s1_batch` | red bars -> those cells are unusable |

## Design note: why x/y is not trusted as a boundary

The H01 block is ~3 mm x 2 mm in the imaging plane but only ~170 um deep along
the sectioning axis (Shapson-Coe et al. 2024). Z is the same order of
magnitude as a dendritic arbour, so pooled z-extent is a reasonable proxy for
the slab boundary. X and Y are not: a subpopulation occupies a small local
patch of a millimetre-scale plane, so a cell reaching past its neighbours is
almost certainly reaching into untouched tissue, not out of the dataset.
`truncation_flag` reports x/y proximity as a diagnostic but never lets it
drive `is_truncated` on its own.

## Ledger (S0.0)

`ancestors_s1_patch.json` carries the `new_infrastructure` entries plus the
one-line `render.py` change (append `sha256_post_s1` to `CSV_COLUMNS`).
Validated against the live spec: no path collisions, merged spec loads through
`build_ledger.load_spec`.

The `--local-dir` working set is the four HybridLFPy S0.1 working copies named
in `ancestors.json["working_branch"]`, which live on your machine:

```bash
python3 tools/build_ledger.py --root . --spec tools/ancestors.json \
        --local-dir <dir with those four files> --out-dir .
```
