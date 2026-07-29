# Patch set: rodent -> human h-current and human passive priors

Target: `Passive Features/HPC script/Synthetic Passive fit Test/`
(mirror of `/davinci-1/home/ldellamea/Human Neurons Fitting/Synthetic Test`)

Every patch is minimal and backwards compatible: with no new CLI flags the
pipeline reproduces the current v60ls behaviour exactly.

New files to drop in (provided separately):

| File | Goes to |
|---|---|
| `Ih_human.mod` | `mod/` (next to `Ih.mod`) |
| `human_ih_params.py` | code dir |
| `smoke_test_ih_human.py` | code dir |

---

## Patch 1 -- `synthetic_ground_truth.py` : make the mechanism selectable

### 1a. `IhConfig` gains a `mechanism` field

FIND (around line 98):

```python
@dataclass
class IhConfig:
    """Hay et al. 2011 / Kole et al. 2006 h-current (the model Eyal 2018 used).
```

REPLACE the whole dataclass body's field block. Specifically, FIND:

```python
    gIhbar_S_cm2: float = 5e-5
    ehcn_mV: float = -45.0
    distribution: str = "uniform"            # "uniform" | "hay_exponential"
    regions: Tuple[str, ...] = ("soma", "dend", "apic")
```

REPLACE WITH:

```python
    gIhbar_S_cm2: float = 5e-5
    ehcn_mV: float = -45.0
    distribution: str = "uniform"            # "uniform" | "hay_exponential"
    regions: Tuple[str, ...] = ("soma", "dend", "apic")
    # NMODL SUFFIX of the h-current mechanism to insert.
    #   "Ih"       -> Kole et al. 2006 rat L5 kinetics (Hay 2011, unaltered)
    #   "Ih_human" -> Rich et al. 2021 human L5 kinetics (Ih_human.mod)
    # Default "Ih" keeps every existing manifest reproducing bit-for-bit.
    mechanism: str = "Ih"
```

### 1b. `_insert_ih` becomes mechanism-agnostic

FIND (around line 305):

```python
        for sec in secs:
            sec.insert("Ih")
            is_apic = sec in self.apic
            for seg in sec:
                if (ih.distribution == "hay_exponential" and is_apic and d_max > 0):
                    d = h.distance(seg.x, sec=sec)
                    factor = -0.8696 + 2.0870 * math.exp(3.6161 * d / d_max)
                    seg.Ih.gIhbar = float(ih.gIhbar_S_cm2) * max(factor, 0.0)
                else:
                    seg.Ih.gIhbar = float(ih.gIhbar_S_cm2)
                seg.Ih.ehcn = float(ih.ehcn_mV)
```

REPLACE WITH:

```python
        mech = str(getattr(ih, "mechanism", "Ih") or "Ih")
        for sec in secs:
            sec.insert(mech)
            is_apic = sec in self.apic
            for seg in sec:
                mobj = getattr(seg, mech)
                if (ih.distribution == "hay_exponential" and is_apic and d_max > 0):
                    d = h.distance(seg.x, sec=sec)
                    factor = -0.8696 + 2.0870 * math.exp(3.6161 * d / d_max)
                    mobj.gIhbar = float(ih.gIhbar_S_cm2) * max(factor, 0.0)
                else:
                    mobj.gIhbar = float(ih.gIhbar_S_cm2)
                mobj.ehcn = float(ih.ehcn_mV)
```

### 1c. Lengthen the I_h settling time (REQUIRED with human kinetics)

The human mTau peaks at ~343 ms, versus ~78 ms for the rodent model. The
current 3000 ms settle is only ~8.7 human time constants, and the resting
state is what every downstream trace is initialised from.

FIND (in `measure_rin_tau_sag`, around line 522):

```python
    settle = 3000.0 if has_dyn else 600.0     # I_h is slow (~100s of ms); settle long
```

REPLACE WITH:

```python
    # I_h is slow. Rodent (Kole 2006) mTau peaks at ~78 ms; human (Rich 2021)
    # mTau peaks at ~343 ms near -74 mV. 6000 ms is >17 human time constants.
    settle = 6000.0 if has_dyn else 600.0
```

---

## Patch 2 -- `gen_from_manifest.py` : carry the kinetics through

FIND (around line 49):

```python
    if use_ih:
        ih = dict(
            gIhbar_S_cm2=float(row["ih_gihbar_S_cm2"]),
            ehcn_mV=float(row["ih_ehcn_mV"]),
            distribution=str(row["ih_dist"]),
        )
```

REPLACE WITH:

```python
    if use_ih:
        ih = dict(
            gIhbar_S_cm2=float(row["ih_gihbar_S_cm2"]),
            ehcn_mV=float(row["ih_ehcn_mV"]),
            distribution=str(row["ih_dist"]),
        )
        # Tolerate manifests written before the human-kinetics column existed:
        # absent column => rodent Kole/Hay mechanism, i.e. legacy behaviour.
        try:
            mech = str(row["ih_kinetics"]).strip()
        except (KeyError, IndexError, TypeError):
            mech = ""
        ih["mechanism"] = mech if mech else "Ih"
```

---

## Patch 3 -- `synth_gt_grid.py` : manifest column, defaults, CLI

### 3a. Add the column

FIND:

```python
    "use_ih", "ih_gihbar_S_cm2", "ih_ehcn_mV", "ih_dist",
```

REPLACE WITH:

```python
    "use_ih", "ih_gihbar_S_cm2", "ih_ehcn_mV", "ih_dist", "ih_kinetics",
```

### 3b. Add human physiological defaults for Ra

FIND:

```python
PHY_TAU_LO_DEFAULT: float = 3.0    # ms
PHY_TAU_HI_DEFAULT: float = 40.0   # ms
```

REPLACE WITH:

```python
PHY_TAU_LO_DEFAULT: float = 3.0    # ms
PHY_TAU_HI_DEFAULT: float = 40.0   # ms

# Ra physiological window for HUMAN L2/3-L3 pyramidal neurons.
# Eyal et al. 2016 (eLife 5:e16553) model fits, n=6: Ra 203-384 Ohm*cm,
# mean 268.5 +/- 30.0. Kalmbach et al. 2018 human deep-L3, uniform-passive
# variant: Ra = 350.24 Ohm*cm (per Rich et al. 2021 Table 3).
# The FITTER box stays (50, 1000) Ohm*cm; this only narrows the GROUND-TRUTH
# draw, so Ra recovery is still tested against a wide search box.
PHY_RA_LO_DEFAULT: float = 100.0   # Ohm*cm
PHY_RA_HI_DEFAULT: float = 500.0   # Ohm*cm
```

### 3c. `make_manifest` signature: new keyword arguments

FIND:

```python
    ih_gihbar_nominal_S_cm2: float = 2e-4,
    ih_gihbar_cv: float = 0.5,
    ih_ehcn_mV: float = -45.0,
    ih_dist: str = "hay_exponential",
```

REPLACE WITH:

```python
    ih_gihbar_nominal_S_cm2: float = 2e-4,
    ih_gihbar_cv: float = 0.5,
    ih_ehcn_mV: float = -45.0,
    ih_dist: str = "hay_exponential",
    ih_kinetics: str = "Ih",          # "Ih" (rodent) | "Ih_human" (Rich 2021)
    ra_phys_lo: Optional[float] = None,
    ra_phys_hi: Optional[float] = None,
```

### 3d. Write the column

FIND:

```python
            ih_dist=(str(ih_dist) if use_ih else ""),
```

REPLACE WITH:

```python
            ih_dist=(str(ih_dist) if use_ih else ""),
            ih_kinetics=(str(ih_kinetics) if use_ih else ""),
```

### 3e. Constrain the Ra draw

FIND:

```python
        ra_cohort = _logU(rng_gt, box.ra_bounds, n_cohorts)
```

and

```python
        ra = _logU(rng_gt, box.ra_bounds, n)
```

In BOTH places replace `box.ra_bounds` with `_ra_draw_bounds`, and insert
immediately before the first of them:

```python
    # Ra draw window: physiological if given, else the full fitter box.
    _ra_lo = float(ra_phys_lo) if ra_phys_lo is not None else box.ra_bounds[0]
    _ra_hi = float(ra_phys_hi) if ra_phys_hi is not None else box.ra_bounds[1]
    if _ra_lo < box.ra_bounds[0] - 1e-9 or _ra_hi > box.ra_bounds[1] + 1e-9:
        raise ValueError(
            "Ra physiological window [{}, {}] escapes the fitter box "
            "[{}, {}]; assert_bounds_match_phase1 would then be violated."
            .format(_ra_lo, _ra_hi, box.ra_bounds[0], box.ra_bounds[1]))
    _ra_draw_bounds = (_ra_lo, _ra_hi)
```

### 3f. Loader must not choke on old manifests

CORRECTED. `load_manifest` raises `KeyError` on any column of
`MANIFEST_COLUMNS` that is absent, and that check runs BEFORE the dtype
coercions. So the backfill has to happen ABOVE the check, not below it, or
every pre-existing manifest (`manifest_v60ls.csv`) becomes unloadable --
including on the login node, where `submit_all_cohorts.sh` calls
`load_manifest` just to read the cohort labels.

FIND:

```python
def load_manifest(path: Union[Path, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    miss = [c for c in MANIFEST_COLUMNS if c not in df.columns]
```

REPLACE WITH:

```python
def load_manifest(path: Union[Path, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Backfill columns introduced after some manifests were written. This MUST
    # run before the strict schema check below, otherwise every pre-v70
    # manifest raises KeyError. Absent ih_kinetics => legacy rodent "Ih".
    if "ih_kinetics" not in df.columns:
        df["ih_kinetics"] = ""
    miss = [c for c in MANIFEST_COLUMNS if c not in df.columns]
```

THEN, separately, FIND:

```python
    df["ih_dist"] = df["ih_dist"].fillna("").astype(str)
```

REPLACE WITH:

```python
    df["ih_dist"] = df["ih_dist"].fillna("").astype(str)
    df["ih_kinetics"] = df["ih_kinetics"].fillna("").astype(str)
```

### 3g. CLI flags

FIND:

```python
    ap.add_argument("--ih-dist", default="hay_exponential")
```

REPLACE WITH:

```python
    ap.add_argument("--ih-dist", default="hay_exponential")
    ap.add_argument("--ih-kinetics", default="Ih",
                    choices=["Ih", "Ih_human"],
                    help="h-current NMODL mechanism: 'Ih' = Kole 2006 rat L5 "
                         "(Hay 2011 unaltered); 'Ih_human' = Rich et al. 2021 "
                         "human L5 fitted kinetics (default: Ih)")
    ap.add_argument("--ra-phys-lo", type=float, default=None,
                    help="lower Ra bound [Ohm*cm] for the ground-truth draw "
                         "(human L2/3-L3: try {})".format(PHY_RA_LO_DEFAULT))
    ap.add_argument("--ra-phys-hi", type=float, default=None,
                    help="upper Ra bound [Ohm*cm] for the ground-truth draw "
                         "(human L2/3-L3: try {})".format(PHY_RA_HI_DEFAULT))
```

Then pass all three through to `make_manifest(...)` in `main()`:

```python
        ih_kinetics=args.ih_kinetics,
        ra_phys_lo=args.ra_phys_lo,
        ra_phys_hi=args.ra_phys_hi,
```

---

## Patch 4 -- `submit_all_cohorts.sh` : human parameter block

FIND:

```bash
IH_GIHBAR=2e-4
IH_GIHBAR_CV=0.5
IH_EHCN=-45.0
IH_DIST="hay_exponential"
```

REPLACE WITH:

```bash
# --- h-current: HUMAN parameters (v4) --------------------------------------
# Kinetics  : Rich et al. 2021, Cereb Cortex 31(2):845-872 (human L5 fit).
#             Peak mTau ~343 ms vs ~78 ms for the Kole 2006 rat model that
#             Ih.mod implements. Cross-layer transfer to L3 is licensed by
#             Moradi Chameh et al. 2021, Nat Commun 12:2497, Suppl. Fig. 5b:
#             human L2&3 and L5 I_h time constants are indistinguishable
#             (p >= 0.9999), only the amplitude differs.
# Density   : Kalmbach et al. 2018 human deep-L3, gIh = 1e-4 S/cm2 UNIFORM
#             (exact value tabulated in Rich et al. 2021 Table 3). This is
#             the layer-matched density for an L3_exc morphology pool.
#             The previous 2e-4 + hay_exponential was the RAT L5 setting.
# ehcn      : -49.85 mV, Rich et al. 2021 Table 1 (rodent value was -45.0).
IH_KINETICS="Ih_human"
IH_GIHBAR=1e-4
IH_GIHBAR_CV=0.5
IH_EHCN=-49.85
IH_DIST="uniform"
```

FIND (in the manifest ARGS array):

```bash
        --ih-dist          "$IH_DIST"
```

REPLACE WITH:

```bash
        --ih-dist          "$IH_DIST"
        --ih-kinetics      "$IH_KINETICS"
        --ra-phys-lo       "$RA_PHYS_LO"
        --ra-phys-hi       "$RA_PHYS_HI"
```

and add near the Cm/tau block:

```bash
# Ra ground-truth draw window (Eyal 2016 human L2/3: 203-384 Ohm*cm;
# Kalmbach 2018 human L3 uniform fit: 350 Ohm*cm). The FITTER box is
# unchanged at [50, 1000], so Ra recovery is still a real test.
RA_PHYS_LO=100.0
RA_PHYS_HI=500.0
```

Also change the manifest / output names so the human run does not collide
with the rodent one:

```bash
MANIFEST="$CODE_DIR/manifest_v70human.csv"
```

and in `submit_synth_benchmark.sh`:

```bash
MANIFEST="$CODE_DIR/manifest_v70human.csv"
ARCHIVE_ROOT="$CODE_DIR/synthetic_archive_v70human"
OUTPUT_ROOT="$CODE_DIR/synthetic_out_v70human"
```

---

## Patch 5 -- `submit_synth_benchmark.sh` : force one recompile

This is the step that silently breaks everything if skipped. The existing
guard is:

```bash
if [ -d "$MOD_DIR" ] && [ ! -x "$CODE_DIR/x86_64/special" ]; then
```

`x86_64/special` already exists from the v60ls runs, so `nrnivmodl` will NOT
run, `Ih_human` will not exist, and every job dies with
`argument not a density mechanism name`.

Run ONCE on the login node after copying `Ih_human.mod` into `mod/`:

```bash
cd "/davinci-1/home/ldellamea/Human Neurons Fitting/Synthetic Test"
conda activate prova
rm -rf x86_64
nrnivmodl mod
ls -l x86_64/special          # must exist and be executable
```

Optionally make the guard self-healing by replacing the condition with a
mechanism-level check:

```bash
NEED_COMPILE=0
[ ! -x "$CODE_DIR/x86_64/special" ] && NEED_COMPILE=1
# recompile if any .mod is newer than the built binary
if [ -x "$CODE_DIR/x86_64/special" ]; then
    for m in "$MOD_DIR"/*.mod; do
        [ "$m" -nt "$CODE_DIR/x86_64/special" ] && NEED_COMPILE=1
    done
fi
if [ -d "$MOD_DIR" ] && [ "$NEED_COMPILE" = "1" ]; then
```

---

## Verification order

```bash
# 1. mechanism transcription, no NEURON needed
python smoke_test_ih_human.py

# 2. after nrnivmodl
python smoke_test_ih_human.py --with-neuron

# 3. existing smoke tests must still pass (backwards compatibility)
python smoke_synth_gt_grid.py
python smoke_gen_from_manifest.py

# 4. build the new manifest and eyeball the realised summary
bash submit_all_cohorts.sh        # will stop after manifest if you Ctrl-C

# 5. ONE cohort first, not the whole fan-out
qsub -v GROUP=cohort_0000 submit_synth_benchmark.sh
```

Step 5 matters: check the per-cell generation line
`[synthetic] specimen ... -> Vrest=... Rin=... tau_m=... sag=...`
lands in the human range before spending the full fan-out.
Target windows from the literature (see `human_ih_params.py` for citations):

| Quantity | Human L2/3-L3 reference | Source |
|---|---|---|
| R_in at RMP | 83 +/- 38 MOhm (L2&3, n=56); 79.4 +/- 21.4 (L3c, n=15) | Moradi Chameh 2021 Fig. 1c |
| R_in at -65 mV, I_h intact | 48.9 +/- 4.5 MOhm (deep L3) | Kalmbach 2018 |
| tau_m (measured, I_h intact) | 13.7 +/- 7.1 ms (L2&3); 17.1 +/- 5.7 ms (L3c) | Moradi Chameh 2021 Fig. 1d |
| sag ratio | 0.07 +/- 0.04 (L2&3); 0.08 +/- 0.01 (L3c) | Moradi Chameh 2021 Suppl. Fig. 2a |
