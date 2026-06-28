# -*- coding: utf-8 -*-
"""
synth_gt_grid.py  (v3: constrained physiological priors)
=========================================================

Phase 1 of the da Vinci synthetic passive-fit benchmark.

Build a *reproducible manifest* of synthetic cells. Each cell is one
(morphology x ground-truth) instance, organised into COHORTS that double as
PBS groups ($GROUP = cohort_k) -- one cohort per job, mirroring the real-data
submit_passive_fit.sh dispatch, and matching the unit Phase 2.5 operates on.

Changes vs v2
-------------
1. Physiological (Cm, tau_m) rejection filter in _draw_cm_rm_constrained.
   Draws (Cm, Rm) pairs by batch rejection until tau_m = Rm*Cm*1e-3 [ms]
   AND Cm both land inside user-specified physiological windows. Individual
   draws remain within box.cm_bounds / box.rm_bounds, so the fitter search
   box and assert_bounds_match_phase1 are unaffected.

2. Three independent RNG streams (previously two):
     rng_gt  -- Ra only         (stable across Cm/Rm constraint changes)
     rng_rej -- Cm, Rm          (rejection-filtered; new stream)
     rng_var -- I_h gbar, noise (unchanged)
   Ra stream isolation means cohort-median Ra and Phase-2.5 targets are
   reproducible even when the Cm/Rm constraint is tightened or relaxed.

3. Realized-distribution summary printed after manifest is built.

Notation (units carried; the 1e-3 in tau_m is written, not absorbed)
---------------------------------------------------------------------
    tau_m = Rm * Cm * 1e-3   [ms]   (Ohm*uF -> ms)
    cm    specific membrane capacitance   [uF/cm^2]
    Rm    specific membrane resistance     [Ohm*cm^2]
    Ra    axial resistivity               [Ohm*cm]
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# --- bounds MIRRORED from phase1_data_loader.py (keep in sync) ---------------
DEFAULT_CM_BOUNDS: Tuple[float, float] = (0.3, 3.0)            # uF/cm^2
DEFAULT_RM_BOUNDS: Tuple[float, float] = (1_000.0, 100_000.0)  # Ohm*cm^2
DEFAULT_RA_BOUNDS: Tuple[float, float] = (50.0, 1_000.0)       # Ohm*cm

# Physiological defaults (human L2/3-L3 pyramidal, Eyal 2016 + Allen data)
# Cm: human L2/3 ~0.45 uF/cm^2 (Eyal), standard ~1.0; window covers both.
# tau_m: physiological range for L3 pyramidal neurons fitted by this pipeline.
PHY_CM_LO_DEFAULT:  float = 0.4    # uF/cm^2
PHY_CM_HI_DEFAULT:  float = 1.5    # uF/cm^2
PHY_TAU_LO_DEFAULT: float = 3.0    # ms
PHY_TAU_HI_DEFAULT: float = 40.0   # ms

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
    """Stable, UNIQUE display name per morphology."""
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
    s = np.sqrt(np.log(1.0 + cv * cv))
    return float(nominal) * np.exp(rng.normal(0.0, s, size=n))


def _draw_cm_rm_constrained(
    rng: np.random.Generator,
    n: int,
    box: "SearchBox",
    cm_lo: Optional[float] = None,
    cm_hi: Optional[float] = None,
    tau_lo_ms: Optional[float] = None,
    tau_hi_ms: Optional[float] = None,
    batch_factor: int = 12,
    max_rounds: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw n (Cm [uF/cm^2], Rm [Ohm*cm^2]) pairs satisfying optional constraints:

        cm_lo  <= Cm            <= cm_hi     [uF/cm^2]
        tau_lo <= Rm * Cm * 1e-3 <= tau_hi   [ms]

    Individual draws remain within box.cm_bounds and box.rm_bounds, so
    assert_bounds_match_phase1 and the per-column invariant checks both pass.

    When all four constraint args are None the function falls back to
    unconstrained log-uniform draws, preserving v2 behaviour exactly.

    Acceptance rate ~32% for the physiological defaults, so batch_factor=12
    fills n slots in a single round (~3.8 accepted per draw on average).
    """
    unconstrained = (cm_lo is None and cm_hi is None
                     and tau_lo_ms is None and tau_hi_ms is None)
    if unconstrained:
        cm = _logU(rng, box.cm_bounds, n)
        rm = _logU(rng, box.rm_bounds, n)
        return cm, rm

    # Apply defaults on any unspecified side (no tightening)
    _cm_lo  = float(cm_lo)     if cm_lo     is not None else box.cm_bounds[0]
    _cm_hi  = float(cm_hi)     if cm_hi     is not None else box.cm_bounds[1]
    _tau_lo = float(tau_lo_ms) if tau_lo_ms is not None else 0.0
    _tau_hi = float(tau_hi_ms) if tau_hi_ms is not None else float("inf")

    if _cm_lo < box.cm_bounds[0] - 1e-9 or _cm_hi > box.cm_bounds[1] + 1e-9:
        raise ValueError(
            "Cm physiological constraint [{}, {}] uF/cm^2 extends outside "
            "the search box [{}, {}]. The constraint must be a sub-interval "
            "of the search box so that all injected truths are recoverable.".format(
                _cm_lo, _cm_hi, box.cm_bounds[0], box.cm_bounds[1]))

    # Minimum Rm that can produce tau_m >= tau_lo for the widest Cm in range:
    # tau_lo = Rm * cm_hi * 1e-3  =>  Rm_min_possible = tau_lo / (cm_hi * 1e-3)
    # If that exceeds box.rm_bounds[1], no solution exists.
    rm_min_possible = _tau_lo / (_cm_hi * 1e-3)
    if rm_min_possible > box.rm_bounds[1] + 1e-9:
        raise ValueError(
            "tau_lo_ms={} with cm_hi={} requires Rm >= {:.0f} Ohm*cm^2, "
            "which exceeds box.rm_bounds[1]={}. Relax tau_lo or cm_hi.".format(
                _tau_lo, _cm_hi, rm_min_possible, box.rm_bounds[1]))

    cm_ok = np.empty(n)
    rm_ok = np.empty(n)
    n_filled = 0
    batch = max(int(n * batch_factor), 500)

    for _round in range(max_rounds):
        _cm  = np.exp(rng.uniform(np.log(box.cm_bounds[0]),
                                  np.log(box.cm_bounds[1]), batch))
        _rm  = np.exp(rng.uniform(np.log(box.rm_bounds[0]),
                                  np.log(box.rm_bounds[1]), batch))
        _tau = _rm * _cm * 1e-3
        mask = ((_cm  >= _cm_lo)  & (_cm  <= _cm_hi)
                & (_tau >= _tau_lo) & (_tau <= _tau_hi))
        take = min(int(mask.sum()), n - n_filled)
        cm_ok[n_filled:n_filled + take] = _cm[mask][:take]
        rm_ok[n_filled:n_filled + take] = _rm[mask][:take]
        n_filled += take
        if n_filled >= n:
            break

    if n_filled < n:
        raise RuntimeError(
            "Rejection sampler: only {}/{} (Cm, Rm) pairs accepted after {} "
            "rounds. Constraints: Cm in [{}, {}] uF/cm^2, "
            "tau_m in [{}, {}] ms. Check feasibility against the search "
            "box Cm=[{}, {}], Rm=[{}, {}].".format(
                n_filled, n, max_rounds,
                _cm_lo, _cm_hi, _tau_lo, _tau_hi,
                box.cm_bounds[0], box.cm_bounds[1],
                box.rm_bounds[0], box.rm_bounds[1]))

    return cm_ok, rm_ok


@dataclass(frozen=True)
class SearchBox:
    """Log-uniform draw box, in physical units. Defaults == the fitter's box."""
    cm_bounds: Tuple[float, float] = DEFAULT_CM_BOUNDS
    rm_bounds: Tuple[float, float] = DEFAULT_RM_BOUNDS
    ra_bounds: Tuple[float, float] = DEFAULT_RA_BOUNDS


def assert_bounds_match_phase1(box: SearchBox = SearchBox()) -> None:
    """Best-effort guard: if phase1_data_loader is importable, assert our mirrored
    bounds equal the fitter's. Skipped silently if the module cannot be imported."""
    try:
        import phase1_data_loader as p1  # noqa: WPS433
    except Exception:
        return
    for name, ours, theirs in (
        ("Cm", box.cm_bounds, p1.DEFAULT_CM_BOUNDS),
        ("Rm", box.rm_bounds, p1.DEFAULT_RM_BOUNDS),
        ("Ra", box.ra_bounds, p1.DEFAULT_RA_BOUNDS),
    ):
        if tuple(map(float, ours)) != tuple(map(float, theirs)):
            raise AssertionError(
                "{} bounds drifted from phase1_data_loader: "
                "synth_gt_grid={} vs phase1={}. A draw outside the fit "
                "box can never be recovered.".format(name, ours, theirs))


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
    # Physiological constraints on (Cm, tau_m = Rm*Cm*1e-3).
    # None = unconstrained (v2 behaviour). Set both lo AND hi to activate.
    cm_phys_lo: Optional[float] = None,   # uF/cm^2  lower bound on Cm
    cm_phys_hi: Optional[float] = None,   # uF/cm^2  upper bound on Cm
    tau_lo_ms:  Optional[float] = None,   # ms        lower bound on tau_m
    tau_hi_ms:  Optional[float] = None,   # ms        upper bound on tau_m
    # I_h (ON by default: per-cell maximal-conductance variability)
    use_ih: bool = True,
    ih_gihbar_nominal_S_cm2: float = 2e-4,
    ih_gihbar_cv: float = 0.5,
    ih_ehcn_mV: float = -45.0,
    ih_dist: str = "hay_exponential",
    # Recording noise (per-cell level jitter via one 'noisiness' factor)
    noise_sigma_nominal_mV: float = 0.05,
    noise_baseline_nominal_mV: float = 0.05,
    noise_drift_nominal_mV: float = 0.10,
    noise_cv: float = 0.3,
    id_base: int = SYNTH_ID_BASE,
) -> pd.DataFrame:
    """Build the seeded synthetic-cell manifest (v3).

    Reproducibility and stream isolation
    -------------------------------------
    A SeedSequence(seed) spawns THREE independent streams:

        rng_gt  draws Ra only.
                Stable across Cm/Rm constraint changes: cohort-median Ra
                and Phase-2.5 targets remain reproducible even if you
                tighten or relax the physiological Cm/tau_m window.

        rng_rej draws (Cm, Rm) pairs via rejection sampling.
                Constrained to [cm_phys_lo, cm_phys_hi] x tau_m-window
                when those args are set; unconstrained log-uniform otherwise.

        rng_var draws I_h gbar and noise factors.
                Stable across ALL GT/constraint changes (toggling use_ih
                or changing CV never shifts any other stream).

    ra_mode
    -------
    'per_cohort' (default): one Ra per cohort, shared.
    'per_cell'             : independent Ra per cell (stress test).
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

    # --- three independent streams -------------------------------------------
    # IMPORTANT: spawn order determines stream identity. Do not reorder.
    #   child 0 -> rng_gt  (Ra)
    #   child 1 -> rng_rej (Cm, Rm)
    #   child 2 -> rng_var (I_h, noise)
    ss = np.random.SeedSequence(int(seed))
    rng_gt, rng_rej, rng_var = (np.random.default_rng(s) for s in ss.spawn(3))

    # --- Ra (rng_gt exclusive) -----------------------------------------------
    if ra_mode == "per_cohort":
        ra_cohort = _logU(rng_gt, box.ra_bounds, n_cohorts)
        ra = ra_cohort[cohort_of]
    else:
        ra = _logU(rng_gt, box.ra_bounds, n)

    # --- Cm, Rm (rng_rej, with optional physiological rejection) -------------
    cm, rm = _draw_cm_rm_constrained(
        rng_rej, n, box,
        cm_lo=cm_phys_lo, cm_hi=cm_phys_hi,
        tau_lo_ms=tau_lo_ms, tau_hi_ms=tau_hi_ms,
    )

    # --- nuisance draws (rng_var; order fixed: I_h first, then noise) --------
    ih_raw = _lognormal_jitter(rng_var, ih_gihbar_nominal_S_cm2, ih_gihbar_cv, n)
    noise_factor = _lognormal_jitter(rng_var, 1.0, noise_cv, n)

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

    # --- realized-distribution summary (ALWAYS print; catches silent failures) -
    _print_realized_summary(df, cm_phys_lo, cm_phys_hi, tau_lo_ms, tau_hi_ms)

    return df


def _print_realized_summary(
    df: pd.DataFrame,
    cm_phys_lo: Optional[float],
    cm_phys_hi: Optional[float],
    tau_lo_ms: Optional[float],
    tau_hi_ms: Optional[float],
) -> None:
    """Print percentile summary of realized GT distributions to stdout."""
    cm_v  = df["cm_true"].to_numpy()
    rm_v  = df["rm_true"].to_numpy()
    ra_v  = df["ra_true"].to_numpy()
    tau_v = df["tau_m_true_ms"].to_numpy()

    def _pct(v):
        return np.percentile(v, [0, 25, 50, 75, 100])

    def _fmt_pct(label, v, fmt=".3f"):
        p = _pct(v)
        return ("[manifest] {:8s}: "
                "min={:{f}}  p25={:{f}}  p50={:{f}}  "
                "p75={:{f}}  max={:{f}}".format(
                    label, *p, f=fmt))

    print("[manifest] --- realized GT distributions ---")
    print(_fmt_pct("Cm[uF/cm2]", cm_v,  ".3f"))
    print(_fmt_pct("Rm[Ohm cm2]", rm_v, ".0f"))
    print(_fmt_pct("Ra[Ohm cm]", ra_v,  ".1f"))
    print(_fmt_pct("tau_m[ms]", tau_v,  ".2f"))

    # Flag any violation of the requested constraints (should never trigger
    # if _draw_cm_rm_constrained works correctly, but cheap to confirm).
    n = len(df)
    violations = 0
    if cm_phys_lo is not None:
        bad = int((cm_v < cm_phys_lo - 1e-9).sum())
        if bad:
            print("[manifest] WARN: {} cells have Cm < cm_phys_lo={}".format(
                bad, cm_phys_lo))
            violations += bad
    if cm_phys_hi is not None:
        bad = int((cm_v > cm_phys_hi + 1e-9).sum())
        if bad:
            print("[manifest] WARN: {} cells have Cm > cm_phys_hi={}".format(
                bad, cm_phys_hi))
            violations += bad
    if tau_lo_ms is not None:
        bad = int((tau_v < tau_lo_ms - 1e-9).sum())
        if bad:
            print("[manifest] WARN: {} cells have tau_m < tau_lo_ms={}".format(
                bad, tau_lo_ms))
            violations += bad
    if tau_hi_ms is not None:
        bad = int((tau_v > tau_hi_ms + 1e-9).sum())
        if bad:
            print("[manifest] WARN: {} cells have tau_m > tau_hi_ms={}".format(
                bad, tau_hi_ms))
            violations += bad
    if violations == 0 and any(x is not None for x in
                                [cm_phys_lo, cm_phys_hi, tau_lo_ms, tau_hi_ms]):
        print("[manifest] constraint check: all {} cells within "
              "physiological window -- OK".format(n))


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
            "no morphologies at {}/{}. Check MORPH_ROOT / MORPH_GLOB "
            "(e.g. 'specimen_*/reconstruction.swc' or '*.swc').".format(
                morph_root, morph_glob))
    return swcs


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Build the seeded synthetic-cell manifest (v3).")
    ap.add_argument("--morph-root", required=True)
    ap.add_argument("--morph-glob", default="specimen_*/reconstruction.swc")
    ap.add_argument("--out", required=True, help="manifest CSV path")
    ap.add_argument("--draws-per-morph", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells-per-cohort", type=int, default=10)
    ap.add_argument("--ra-mode", choices=["per_cohort", "per_cell"],
                    default="per_cohort")
    ap.add_argument("--e-pas", type=float, default=-70.0)
    ap.add_argument("--F", type=float, default=1.9)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--no-ih", action="store_true",
                    help="disable I_h (passive baseline)")
    ap.add_argument("--ih-gihbar", type=float, default=2e-4)
    ap.add_argument("--ih-gihbar-cv", type=float, default=0.5)
    ap.add_argument("--ih-ehcn", type=float, default=-45.0)
    ap.add_argument("--ih-dist", default="hay_exponential")
    ap.add_argument("--noise-sigma", type=float, default=0.05)
    ap.add_argument("--noise-baseline", type=float, default=0.05)
    ap.add_argument("--noise-drift", type=float, default=0.10)
    ap.add_argument("--noise-cv", type=float, default=0.3)
    # --- physiological prior constraints (v3) --------------------------------
    ap.add_argument("--cm-phys-lo", type=float, default=PHY_CM_LO_DEFAULT,
                    help="lower Cm bound [uF/cm^2] for rejection filter "
                         "(default: {})".format(PHY_CM_LO_DEFAULT))
    ap.add_argument("--cm-phys-hi", type=float, default=PHY_CM_HI_DEFAULT,
                    help="upper Cm bound [uF/cm^2] for rejection filter "
                         "(default: {})".format(PHY_CM_HI_DEFAULT))
    ap.add_argument("--tau-lo-ms", type=float, default=PHY_TAU_LO_DEFAULT,
                    help="lower tau_m bound [ms] for rejection filter "
                         "(default: {})".format(PHY_TAU_LO_DEFAULT))
    ap.add_argument("--tau-hi-ms", type=float, default=PHY_TAU_HI_DEFAULT,
                    help="upper tau_m bound [ms] for rejection filter "
                         "(default: {})".format(PHY_TAU_HI_DEFAULT))
    ap.add_argument("--no-phy-filter", action="store_true",
                    help="disable physiological rejection filter (v2 behaviour)")
    args = ap.parse_args(argv)

    swcs = _resolve_swcs(args.morph_root, args.morph_glob)

    cm_lo = cm_hi = tau_lo = tau_hi = None
    if not args.no_phy_filter:
        cm_lo, cm_hi = args.cm_phys_lo, args.cm_phys_hi
        tau_lo, tau_hi = args.tau_lo_ms, args.tau_hi_ms

    df = draw_manifest(
        swcs,
        draws_per_morph=args.draws_per_morph, seed=args.seed,
        cells_per_cohort=args.cells_per_cohort, ra_mode=args.ra_mode,
        e_pas_mV=args.e_pas, F=args.F, max_cells=args.max_cells,
        cm_phys_lo=cm_lo, cm_phys_hi=cm_hi,
        tau_lo_ms=tau_lo, tau_hi_ms=tau_hi,
        use_ih=not args.no_ih,
        ih_gihbar_nominal_S_cm2=args.ih_gihbar, ih_gihbar_cv=args.ih_gihbar_cv,
        ih_ehcn_mV=args.ih_ehcn, ih_dist=args.ih_dist,
        noise_sigma_nominal_mV=args.noise_sigma,
        noise_baseline_nominal_mV=args.noise_baseline,
        noise_drift_nominal_mV=args.noise_drift, noise_cv=args.noise_cv,
    )
    save_manifest(df, args.out)
    groups = list_groups(df)
    print("[manifest] {} cells | {} morphologies x {} draw(s) | {} cohort(s) "
          "of <= {} | ra_mode={} | I_h={} (gbar cv={}) | noise cv={} | "
          "seed={}".format(
              len(df), len(swcs), args.draws_per_morph, len(groups),
              args.cells_per_cohort, args.ra_mode,
              "on" if not args.no_ih else "off", args.ih_gihbar_cv,
              args.noise_cv, args.seed))
    if not args.no_phy_filter:
        print("[manifest] physiological filter: "
              "Cm in [{}, {}] uF/cm^2  tau_m in [{}, {}] ms".format(
                  cm_lo, cm_hi, tau_lo, tau_hi))
    else:
        print("[manifest] physiological filter: OFF (v2 unconstrained mode)")
    print(f"[manifest] groups: {groups}")
    print(f"[manifest] wrote {args.out}")


if __name__ == "__main__":
    main()
