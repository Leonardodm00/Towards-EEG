# -*- coding: utf-8 -*-
"""
synth_gt_grid.py  (v2: cohort-shared Ra + per-cell I_h / noise jitter)
=====================================================================

Phase 1 of the da Vinci synthetic passive-fit benchmark.

Build a *reproducible manifest* of synthetic cells. Each cell is one
(morphology x ground-truth) instance, organised into COHORTS that double as PBS
groups (``$GROUP = cohort_k``) -- one cohort per job, mirroring the real-data
``submit_passive_fit.sh`` dispatch, and matching the unit Phase 2.5 operates on.

This module is PURE (numpy / pandas only): no NEURON, no skopt, no I/O beyond
CSV. Generation (Phase 2) reads a manifest row -> GroundTruthParams + NoiseConfig
+ IhConfig; aggregation (Phase 3) reads the manifest to join injected truth onto
recovered fits. ONE seeded manifest keeps the whole benchmark reproducible.

Generative hierarchy (why it is shaped this way)
------------------------------------------------
COHORT-level (shared by all cells in a cohort = a PBS group = a Phase-2.5 unit):
    Ra      axial resistivity   [Ohm*cm]   -- Phase 2.5 borrows strength on the
            premise that Ra is shared across a cohort; drawing it per cohort
            makes that premise TRUE, so the cohort-median Ra that Phase 2.5 fixes
            has a real ground-truth target. (ra_mode='per_cell' overrides this
            for an Ra-misspecification stress test; then Ra recovery is NOT
            interpretable and Cm/Rm pick up extra bias -- by design.)
CELL-level (the per-cell variability of a realistic cohort):
    Cm      specific membrane capacitance   [uF/cm^2]   log-uniform over the box
    Rm      specific membrane resistance     [Ohm*cm^2]  log-uniform over the box
    gIhbar  I_h maximal conductance          [S/cm^2]    log-normal jitter
    sigma   recording-noise level            [mV]        log-normal jitter
    morphology, noise_seed

Draws that span the fitter's search box (log-uniform on each axis) are exactly
"randomly drawn from the searched parameter space". I_h / noise jitter use a
log-normal with median at the nominal and CV exactly as requested:
    x = nominal * exp(N(0, s)),   s = sqrt(ln(1 + CV^2))   =>   median=nominal,
    CV(x)=CV, x>0 always.

Notation (units carried throughout; the 1e-3 in tau_m is written, not absorbed)
------------------------------------------------------------------------------
    tau_m = Rm * Cm * 1e-3   [ms]   (Ohm*uF -> ms)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# --- bounds MIRRORED from phase1_data_loader.py (keep in sync) ---------------
DEFAULT_CM_BOUNDS: Tuple[float, float] = (0.3, 3.0)            # uF/cm^2
DEFAULT_RM_BOUNDS: Tuple[float, float] = (1_000.0, 100_000.0)  # Ohm*cm^2
DEFAULT_RA_BOUNDS: Tuple[float, float] = (50.0, 1_000.0)      # Ohm*cm

SYNTH_ID_BASE: int = 900_000_000   # synthetic specimen IDs never collide w/ Allen

MANIFEST_COLUMNS: List[str] = [
    "specimen_id", "group", "cohort", "morph_name", "swc", "draw_idx",
    "cm_true", "rm_true", "ra_true", "ra_mode", "tau_m_true_ms",
    "e_pas_mV", "F",
    "use_ih", "ih_gihbar_S_cm2", "ih_ehcn_mV", "ih_dist",
    "noise_sigma_mV", "noise_baseline_sigma_mV", "noise_drift_sigma_mV",
    "noise_seed",
]

_GENERIC_STEMS = {"reconstruction", "morphology", "morph", "cell"}


# ===========================================================================
#  Small pure helpers
# ===========================================================================
def _unique_morph_names(swcs: Sequence[Path]) -> List[str]:
    """Stable, UNIQUE display name per morphology. The Allen layout
    ``specimen_<id>/reconstruction.swc`` makes every stem collide ('reconstruction'),
    so use the parent dir for generic stems, the stem otherwise, then disambiguate
    any residual collision by the pool index."""
    raw = [(s.parent.name if s.stem.lower() in _GENERIC_STEMS else s.stem)
           for s in swcs]
    out: List[str] = []
    for i, name in enumerate(raw):
        out.append(f"{name}__{i}" if raw.count(name) > 1 else name)
    return out


def _logU(rng: np.random.Generator, lo_hi: Tuple[float, float], n: int) -> np.ndarray:
    """n log-uniform draws over [lo, hi] (physical units)."""
    lo, hi = lo_hi
    if not (0.0 < lo < hi):
        raise ValueError(f"bad bounds {lo_hi}: need 0 < lo < hi")
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))


def _lognormal_jitter(rng: np.random.Generator, nominal: float, cv: float,
                      n: int) -> np.ndarray:
    """n positive draws with median=nominal and coefficient of variation=cv.
    cv<=0 -> constant (no jitter)."""
    if nominal <= 0.0:
        raise ValueError(f"nominal must be > 0, got {nominal}")
    if cv <= 0.0:
        return np.full(n, float(nominal))
    s = np.sqrt(np.log(1.0 + cv * cv))            # CV(lognormal) = cv exactly
    return float(nominal) * np.exp(rng.normal(0.0, s, size=n))


@dataclass(frozen=True)
class SearchBox:
    """Log-uniform draw box, in physical units. Defaults == the fitter's box."""
    cm_bounds: Tuple[float, float] = DEFAULT_CM_BOUNDS
    rm_bounds: Tuple[float, float] = DEFAULT_RM_BOUNDS
    ra_bounds: Tuple[float, float] = DEFAULT_RA_BOUNDS


def assert_bounds_match_phase1(box: SearchBox = SearchBox()) -> None:
    """Best-effort guard: if phase1_data_loader is importable, assert our mirrored
    bounds equal the fitter's. Skipped silently if that module (or its NEURON/skopt
    deps) cannot be imported -- this file must stay NEURON-free."""
    try:
        import phase1_data_loader as p1  # noqa: WPS433 (optional import by design)
    except Exception:
        return
    for name, ours, theirs in (
        ("Cm", box.cm_bounds, p1.DEFAULT_CM_BOUNDS),
        ("Rm", box.rm_bounds, p1.DEFAULT_RM_BOUNDS),
        ("Ra", box.ra_bounds, p1.DEFAULT_RA_BOUNDS),
    ):
        if tuple(map(float, ours)) != tuple(map(float, theirs)):
            raise AssertionError(
                f"{name} bounds drifted from phase1_data_loader: "
                f"synth_gt_grid={ours} vs phase1={theirs}. A draw outside the fit "
                f"box can never be recovered.")


# ===========================================================================
#  Manifest builder
# ===========================================================================
def draw_manifest(
    swc_paths: Sequence[Union[Path, str]],
    *,
    draws_per_morph: int = 1,
    seed: int = 0,
    cells_per_cohort: int = 10,
    box: SearchBox = SearchBox(),
    ra_mode: str = "per_cohort",
    e_pas_mV: float = -70.0,
    F: float = 1.9,
    max_cells: Optional[int] = None,
    noise_seed_base: Optional[int] = None,
    # --- I_h (ON by default now: per-cell maximal-conductance variability) ---
    use_ih: bool = True,
    ih_gihbar_nominal_S_cm2: float = 2e-4,
    ih_gihbar_cv: float = 0.5,
    ih_ehcn_mV: float = -45.0,
    ih_dist: str = "hay_exponential",
    # --- recording noise (per-cell level jitter via one 'noisiness' factor) ---
    noise_sigma_nominal_mV: float = 0.05,
    noise_baseline_nominal_mV: float = 0.05,
    noise_drift_nominal_mV: float = 0.10,
    noise_cv: float = 0.3,
    id_base: int = SYNTH_ID_BASE,
) -> pd.DataFrame:
    """Build the seeded synthetic-cell manifest with cohort-shared Ra and
    per-cell I_h / noise jitter.

    Reproducibility & stream isolation
    ----------------------------------
    A SeedSequence(seed) spawns TWO independent streams: rng_gt draws the
    identifiability targets (Cm, Rm, Ra) and rng_var draws the per-cell nuisance
    variability (I_h gbar, noise level). So toggling use_ih / changing any CV
    never shifts the (Cm, Rm, Ra) draws -- the GT is stable under nuisance edits.

    ra_mode
    -------
    'per_cohort' (default): one Ra per cohort, shared -> Phase 2.5's premise holds.
    'per_cell'             : independent Ra per cell -> Ra-misspecification stress
                             test (Ra recovery not interpretable; Cm/Rm biased).
    """
    if draws_per_morph < 1:
        raise ValueError(f"draws_per_morph must be >= 1, got {draws_per_morph}")
    if cells_per_cohort < 1:
        raise ValueError(f"cells_per_cohort must be >= 1, got {cells_per_cohort}")
    if ra_mode not in ("per_cohort", "per_cell"):
        raise ValueError(f"ra_mode must be 'per_cohort'|'per_cell', got {ra_mode!r}")
    swcs = [Path(s) for s in swc_paths]
    if not swcs:
        raise ValueError("swc_paths is empty -- no morphologies to draw on.")
    assert_bounds_match_phase1(box)

    n_morph = len(swcs)
    n = n_morph * int(draws_per_morph)
    if max_cells is not None:
        n = min(n, int(max_cells))
    if n == 0:
        raise ValueError("zero cells after applying max_cells.")

    cohort_of = np.arange(n) // int(cells_per_cohort)
    n_cohorts = int(cohort_of.max()) + 1

    # --- two independent streams: GT targets vs nuisance variability ----------
    ss = np.random.SeedSequence(int(seed))
    rng_gt, rng_var = (np.random.default_rng(s) for s in ss.spawn(2))

    cm = _logU(rng_gt, box.cm_bounds, n)
    rm = _logU(rng_gt, box.rm_bounds, n)
    if ra_mode == "per_cohort":
        ra_cohort = _logU(rng_gt, box.ra_bounds, n_cohorts)
        ra = ra_cohort[cohort_of]
    else:
        ra = _logU(rng_gt, box.ra_bounds, n)

    # nuisance draws ALWAYS consumed in fixed order (I_h then noise) so the noise
    # stream position is independent of use_ih:
    ih_raw = _lognormal_jitter(rng_var, ih_gihbar_nominal_S_cm2, ih_gihbar_cv, n)
    noise_factor = _lognormal_jitter(rng_var, 1.0, noise_cv, n)  # per-cell scalar

    nsb = int(seed if noise_seed_base is None else noise_seed_base)
    morph_names = _unique_morph_names(swcs)

    rows: List[dict] = []
    for i in range(n):
        m = i // int(draws_per_morph)
        swc = swcs[m]
        f = float(noise_factor[i])
        rows.append(dict(
            specimen_id=int(id_base) + i,
            group=f"cohort_{int(cohort_of[i]):04d}",
            cohort=int(cohort_of[i]),
            morph_name=morph_names[m],
            swc=str(swc),
            draw_idx=i % int(draws_per_morph),
            cm_true=float(cm[i]), rm_true=float(rm[i]), ra_true=float(ra[i]),
            ra_mode=ra_mode,
            tau_m_true_ms=float(rm[i]) * float(cm[i]) * 1e-3,
            e_pas_mV=float(e_pas_mV), F=float(F),
            use_ih=bool(use_ih),
            ih_gihbar_S_cm2=(float(ih_raw[i]) if use_ih else np.nan),
            ih_ehcn_mV=(float(ih_ehcn_mV) if use_ih else np.nan),
            ih_dist=(str(ih_dist) if use_ih else ""),
            noise_sigma_mV=float(noise_sigma_nominal_mV) * f,
            noise_baseline_sigma_mV=float(noise_baseline_nominal_mV) * f,
            noise_drift_sigma_mV=float(noise_drift_nominal_mV) * f,
            noise_seed=nsb + i,
        ))
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    # --- invariants (fail loud) ----------------------------------------------
    if df["specimen_id"].duplicated().any():
        raise AssertionError("duplicate specimen_id in manifest.")
    for col, (lo, hi) in (("cm_true", box.cm_bounds),
                          ("rm_true", box.rm_bounds),
                          ("ra_true", box.ra_bounds)):
        v = df[col].to_numpy()
        if not np.all((v >= lo - 1e-9) & (v <= hi + 1e-9)):
            raise AssertionError(f"{col} drew outside the search box {(lo, hi)}.")
    if ra_mode == "per_cohort":
        per = df.groupby("cohort")["ra_true"].nunique()
        if not (per == 1).all():
            raise AssertionError("per_cohort Ra not constant within every cohort.")
    return df


# ===========================================================================
#  I/O + per-job slicing
# ===========================================================================
def save_manifest(df: pd.DataFrame, path: Union[Path, str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=MANIFEST_COLUMNS).to_csv(path, index=False)


def load_manifest(path: Union[Path, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    miss = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if miss:
        raise KeyError(f"manifest {path} missing column(s) {miss}; "
                       f"available={sorted(df.columns)}")
    for c in ("specimen_id", "cohort", "noise_seed"):
        df[c] = df[c].astype(int)
    df["use_ih"] = df["use_ih"].astype(bool)
    df["ih_dist"] = df["ih_dist"].fillna("").astype(str)
    df["ra_mode"] = df["ra_mode"].astype(str)
    return df.reindex(columns=MANIFEST_COLUMNS)


def list_groups(df: pd.DataFrame) -> List[str]:
    return sorted(df["group"].unique().tolist())


def group_of(df: pd.DataFrame, group: str) -> pd.DataFrame:
    sub = df[df["group"] == group]
    if sub.empty:
        raise KeyError(f"group {group!r} not in manifest; "
                       f"known groups={list_groups(df)}")
    return sub.reset_index(drop=True)


# ===========================================================================
#  CLI
# ===========================================================================
def _resolve_swcs(morph_root: str, morph_glob: str) -> List[Path]:
    swcs = sorted(Path(morph_root).glob(morph_glob))
    if not swcs:
        raise FileNotFoundError(
            f"no morphologies at {morph_root}/{morph_glob}. Check MORPH_ROOT / "
            f"MORPH_GLOB (e.g. 'specimen_*/reconstruction.swc' or '*.swc').")
    return swcs


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build the seeded synthetic-cell manifest (v2).")
    ap.add_argument("--morph-root", required=True)
    ap.add_argument("--morph-glob", default="specimen_*/reconstruction.swc")
    ap.add_argument("--out", required=True, help="manifest CSV path")
    ap.add_argument("--draws-per-morph", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells-per-cohort", type=int, default=10)
    ap.add_argument("--ra-mode", choices=["per_cohort", "per_cell"], default="per_cohort")
    ap.add_argument("--e-pas", type=float, default=-70.0)
    ap.add_argument("--F", type=float, default=1.9)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--no-ih", action="store_true", help="disable I_h (passive baseline)")
    ap.add_argument("--ih-gihbar", type=float, default=2e-4)
    ap.add_argument("--ih-gihbar-cv", type=float, default=0.5)
    ap.add_argument("--ih-ehcn", type=float, default=-45.0)
    ap.add_argument("--ih-dist", default="hay_exponential")
    ap.add_argument("--noise-sigma", type=float, default=0.05)
    ap.add_argument("--noise-baseline", type=float, default=0.05)
    ap.add_argument("--noise-drift", type=float, default=0.10)
    ap.add_argument("--noise-cv", type=float, default=0.3)
    args = ap.parse_args(argv)

    swcs = _resolve_swcs(args.morph_root, args.morph_glob)
    df = draw_manifest(
        swcs, draws_per_morph=args.draws_per_morph, seed=args.seed,
        cells_per_cohort=args.cells_per_cohort, ra_mode=args.ra_mode,
        e_pas_mV=args.e_pas, F=args.F, max_cells=args.max_cells,
        use_ih=not args.no_ih,
        ih_gihbar_nominal_S_cm2=args.ih_gihbar, ih_gihbar_cv=args.ih_gihbar_cv,
        ih_ehcn_mV=args.ih_ehcn, ih_dist=args.ih_dist,
        noise_sigma_nominal_mV=args.noise_sigma,
        noise_baseline_nominal_mV=args.noise_baseline,
        noise_drift_nominal_mV=args.noise_drift, noise_cv=args.noise_cv,
    )
    save_manifest(df, args.out)
    groups = list_groups(df)
    print(f"[manifest] {len(df)} cells | {len(swcs)} morphologies x "
          f"{args.draws_per_morph} draw(s) | {len(groups)} cohort(s) of "
          f"<= {args.cells_per_cohort} | ra_mode={args.ra_mode} | "
          f"I_h={'on' if not args.no_ih else 'off'} (gbar cv={args.ih_gihbar_cv}) | "
          f"noise cv={args.noise_cv} | seed={args.seed}")
    print(f"[manifest] groups: {groups}")
    print(f"[manifest] wrote {args.out}")


if __name__ == "__main__":
    main()
