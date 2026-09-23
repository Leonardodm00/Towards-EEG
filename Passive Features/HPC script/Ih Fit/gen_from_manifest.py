# -*- coding: utf-8 -*-
"""
gen_from_manifest.py
====================

Phase 2 (generation stage) of the da Vinci synthetic benchmark.

Turn one cohort's manifest rows into Phase-0 archives on the compute node, using
the project's own ``synthetic_ground_truth`` generator. Each archive is then
loadable by the monolith's ``load_cells_from_archive`` exactly like real data.

Separation of concerns (so the mapping is testable WITHOUT NEURON)
-----------------------------------------------------------------
    row_to_gt_kwargs / row_to_noise_kwargs / use_builder_factory
        PURE functions: manifest row -> plain kwargs dicts / bool. No NEURON,
        no synthetic_ground_truth import. Unit-tested in smoke_gen_from_manifest.py.
    build_gt / build_noise / build_proto
        Construct the real synthetic_ground_truth dataclasses from those kwargs.
        ``sgt`` (the synthetic_ground_truth module) is injected, not imported at
        top level, so importing THIS module never pulls in NEURON.
    generate_group
        Orchestrates generation for one cohort (NEURON-side). Injects ``sgt`` and
        ``mono`` (the monolith, for build_neuron_model). No fitting, no plotting.

Factory choice (mirrors the Colab):
    passive cell (use_ih == False, not active)  -> mono.build_neuron_model
    I_h / active present                        -> None (the bundled cell that
                                                   can host I_h / active mechs)
"""

import dataclasses
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


# ===========================================================================
#  PURE mapping: manifest row -> generation kwargs  (NEURON-free, testable)
# ===========================================================================
def _get(row, key: str, default: float) -> float:
    """row[key] as a float, or `default` when the column is absent, NaN or
    unparseable. Manifests predate several columns and must keep generating
    the ground truth they always generated."""
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return float(default)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if not np.isfinite(f) else f


def _get_str(row, key: str, default: str) -> str:
    """row[key] as a stripped string, or `default` when absent/NaN/blank."""
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    if v is None:
        return default
    t = str(v).strip()
    return default if t.lower() in ("", "nan", "none") else t


def row_to_gt_kwargs(row) -> Dict:
    """Manifest row -> GroundTruthParams kwargs (as a plain dict).

    ``ih`` is a nested kwargs dict (or None) so this stays import-free; build_gt
    turns it into an IhConfig. Cm/Rm/Ra carry their manifest units verbatim."""
    use_ih = bool(row["use_ih"])
    ih = None
    if use_ih:
        ih = dict(
            gIhbar_S_cm2=float(row["ih_gihbar_S_cm2"]),
            ehcn_mV=float(row["ih_ehcn_mV"]),
            distribution=str(row["ih_dist"]),
        )
        # --- Stage 7 (D-005): the two FITTED kinetic knobs as ground truth,
        # plus the configuration shift that is not fitted, plus where the
        # mechanism goes. A manifest written before Stage 7 has none of these
        # columns, so each falls back to the value that reproduces the
        # published model exactly -- the ground truth that manifest always
        # generated. `_get` never raises on a missing column, because a row
        # is a pandas Series in production and a plain dict in the smoke.
        ih["vshift_mV"] = _get(row, "ih_vshift_base_mV", 0.0)
        ih["vshift_minf_mV"] = _get(row, "ih_dvh_mV", 0.0)
        ih["tau_scale"] = _get(row, "ih_kappa_tau", 1.0)
        regions = _get_str(row, "ih_regions", "")
        if regions:
            ih["regions"] = tuple(r.strip() for r in regions.split(",")
                                  if r.strip())
        # NMODL SUFFIX of the h-current mechanism. Tolerate manifests written
        # before this column existed: absent/blank -> rodent "Ih" (legacy).
        try:
            mech = str(row["ih_kinetics"]).strip()
        except (KeyError, IndexError, TypeError):
            mech = ""
        if mech.lower() in ("", "nan", "none"):
            mech = "Ih"
        ih["mechanism"] = mech
    return dict(
        cm_uF_cm2=float(row["cm_true"]),
        rm_Ohm_cm2=float(row["rm_true"]),
        ra_Ohm_cm=float(row["ra_true"]),
        e_pas_mV=float(row["e_pas_mV"]),
        spine_factor_F=float(row["F"]),
        active=False,                    # active channels not driven from manifest
        ih=ih,
    )


def row_to_noise_kwargs(row) -> Dict:
    """Manifest row -> NoiseConfig kwargs (per-cell levels + seed).

    `rho_lag1` is the AR(1) coefficient of the per-sweep fast noise. No
    manifest set it before the noise calibration, so every earlier cohort had
    WHITE noise; a row without the column still gets 0.0, i.e. exactly what
    it always generated."""
    return dict(
        sigma_mV=float(row["noise_sigma_mV"]),
        rho_lag1=_get(row, "noise_rho_lag1", 0.0),
        baseline_sigma_mV=float(row["noise_baseline_sigma_mV"]),
        drift_sigma_mV=float(row["noise_drift_sigma_mV"]),
        seed=int(row["noise_seed"]),
    )


def row_to_ss_noise_kwargs(row) -> Optional[Dict]:
    """NoiseConfig kwargs for the Square Subthreshold pulses ALONE, or None.

    None means the SS pulses carry the same noise as the Long Square sweeps
    (row_to_noise_kwargs) -- every manifest written before D-014, and every
    cell whose real twin had no usable SS measurement. Otherwise the fast
    per-sweep level and lag-1 correlation are the SS pulses' own
    (`noise_ss_sigma_mV`, `noise_ss_rho_lag1`), while the slow terms and the
    seed stay the cell's: a DC offset or a drift is a property of the
    recording, whereas the fast noise depends on how each protocol was
    sampled and on the bandwidth its own window sees."""
    sig = _get(row, "noise_ss_sigma_mV", float("nan"))
    rho = _get(row, "noise_ss_rho_lag1", float("nan"))
    if not (np.isfinite(sig) and np.isfinite(rho)):
        return None
    kw = row_to_noise_kwargs(row)
    kw["sigma_mV"] = float(sig)
    kw["rho_lag1"] = float(rho)
    return kw


def row_to_acquisition(row) -> Dict[str, float]:
    """The per-cell ACQUISITION TWIN (D-014): SS pulses per polarity and the
    two sampling rates of the real cell whose morphology and noise this
    synthetic cell borrows. Only finite, positive entries are returned; an
    absent or NaN column leaves the cohort protocol in force for that item.

    Why per cell: rho_lag1 is a correlation between SUCCESSIVE SAMPLES, so it
    is defined only together with the sampling interval. The Allen archive
    mixes cells digitised at 200 kHz (before 2016) and at 50 kHz (2016 and
    later) [Allen ephys whitepaper, project KB]; a rho measured on 5-us
    samples and injected into 20-us samples describes noise whose
    correlation time is four times too long. Generating each synthetic cell
    at its twin's own rates makes (sigma, rho) transfer exactly, with no
    model of how rho would change with the rate. The SS pulse count sets how
    far the SS bundle's noise is averaged down, so it is per cell too."""
    out: Dict[str, float] = {}
    n = _get(row, "acq_ss_n_repeats", float("nan"))
    if np.isfinite(n) and n >= 1:
        out["ss_n_repeats"] = int(round(n))
    for col, key in (("acq_fs_ss_Hz", "ss_sampling_rate_Hz"),
                     ("acq_fs_ls_Hz", "ls_sampling_rate_Hz")):
        v = _get(row, col, float("nan"))
        if np.isfinite(v) and v > 0:
            out[key] = float(v)
    return out


def use_builder_factory(row) -> bool:
    """True -> generate with mono.build_neuron_model (pure-passive cell); False ->
    use the generator's bundled cell (needed to host I_h / active mechanisms)."""
    return not bool(row["use_ih"])


# ===========================================================================
#  NEURON-side constructors (sgt = synthetic_ground_truth module, injected)
# ===========================================================================
def build_gt(sgt, gt_kwargs: Dict):
    """Build a GroundTruthParams from row_to_gt_kwargs output."""
    ih = None
    if gt_kwargs.get("ih") is not None:
        ih = sgt.IhConfig(**gt_kwargs["ih"])
    kw = {k: v for k, v in gt_kwargs.items() if k != "ih"}
    return sgt.GroundTruthParams(ih=ih, **kw)


def build_noise(sgt, noise_kwargs: Dict):
    """Build a NoiseConfig, keeping only fields THIS installed version supports
    (older copies lack baseline_sigma_mV / drift_sigma_mV; mirror the Colab's
    defensive construction so generation never crashes on an old generator)."""
    ok = {f.name for f in dataclasses.fields(sgt.NoiseConfig)}
    dropped = [k for k in noise_kwargs if k not in ok]
    if dropped:
        print(f"[gen][WARN] synthetic_ground_truth.NoiseConfig lacks {dropped}; "
              f"continuing without them (bundle-mean signal is unaffected).")
    return sgt.NoiseConfig(**{k: v for k, v in noise_kwargs.items() if k in ok})


def build_proto(sgt, *, ss_n_repeats: int,
                ls_hyp_amplitudes_pA: Sequence[float],
                ls_dep_amplitudes_pA: Sequence[float] = (),
                ss_sampling_rate_Hz: Optional[float] = None,
                ls_sampling_rate_Hz: Optional[float] = None):
    """ProtocolConfig for one cohort.

    `ls_dep_amplitudes_pA` was NOT forwarded before Stage 7, so every
    generated archive had an empty depolarising Long Square set. Under the
    D-006 protocol those steps ARE the validation set, so a cohort generated
    without them would leave the I_h arms with nothing held out and a
    validation RMSD computed over the brief pulses alone. It defaults to ()
    so every pre-Stage-7 caller generates exactly what it generated before.

    The two sampling rates default to None = ProtocolConfig's own (50 kHz SS,
    20 kHz LS). They matter once the noise is measured: the lag-1
    correlation of a filtered recording depends on the sampling rate, so a
    rho read off real sweeps at one rate and injected at another is not the
    same noise. The measured rates travel with the measured rho.
    """
    kw = dict(
        ss_n_repeats=int(ss_n_repeats),
        ls_hyp_amplitudes_pA=tuple(float(a) for a in ls_hyp_amplitudes_pA),
        ls_dep_amplitudes_pA=tuple(float(a) for a in ls_dep_amplitudes_pA))
    if ss_sampling_rate_Hz is not None:
        kw["ss_sampling_rate_Hz"] = float(ss_sampling_rate_Hz)
    if ls_sampling_rate_Hz is not None:
        kw["ls_sampling_rate_Hz"] = float(ls_sampling_rate_Hz)
    return sgt.ProtocolConfig(**kw)


# ===========================================================================
#  Orchestration: generate one cohort's archives
# ===========================================================================
def generate_group(
    group_df: pd.DataFrame,
    archive_dir: Union[Path, str],
    *,
    sgt,                         # synthetic_ground_truth module
    mono,                        # passive_fitting_hpc_fixed module (build_neuron_model)
    ss_n_repeats: int = 30,
    ls_hyp_amplitudes_pA: Sequence[float] = (-10., -30., -50., -70., -90.),
    ls_dep_amplitudes_pA: Sequence[float] = (),
    ss_sampling_rate_Hz: Optional[float] = None,
    ls_sampling_rate_Hz: Optional[float] = None,
    clear_fn: Optional[Callable[[], None]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate + write a Phase-0 archive for every cell in ``group_df``.

    One cell at a time (optional ``clear_fn`` wipes NEURON sections between cells,
    matching the rest of the pipeline). Returns a small per-cell meta DataFrame
    (specimen_id, measured rin_MOhm, sag_ratio, injected tau_m) for the report;
    the archives themselves go to ``archive_dir/specimen_<sid>/``.
    """
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    cohort_kw = dict(ss_n_repeats=ss_n_repeats,
                     ls_hyp_amplitudes_pA=ls_hyp_amplitudes_pA,
                     ls_dep_amplitudes_pA=ls_dep_amplitudes_pA,
                     ss_sampling_rate_Hz=ss_sampling_rate_Hz,
                     ls_sampling_rate_Hz=ls_sampling_rate_Hz)
    proto = build_proto(sgt, **cohort_kw)

    meta: List[dict] = []
    for _, row in group_df.iterrows():
        if clear_fn is not None:
            clear_fn()
        sid = int(row["specimen_id"])
        gt = build_gt(sgt, row_to_gt_kwargs(row))
        noise = build_noise(sgt, row_to_noise_kwargs(row))
        ss_kw = row_to_ss_noise_kwargs(row)
        acq = row_to_acquisition(row)
        # the cohort protocol, with this cell's own acquisition where known
        cell_proto = build_proto(sgt, **{**cohort_kw, **acq}) if acq else proto
        factory = mono.build_neuron_model if use_builder_factory(row) else None
        gen_kw = dict(proto=cell_proto, noise=noise, specimen_id=sid,
                      passive_cell_factory=factory, verbose=verbose)
        if ss_kw is not None:
            # only when present, so an older generator without the keyword
            # still generates every legacy manifest
            gen_kw["ss_noise"] = build_noise(sgt, ss_kw)
        try:
            syn = sgt.generate_synthetic_cell(Path(row["swc"]), gt, **gen_kw)
            sgt.write_archive_cell(syn, archive_dir / f"specimen_{sid}",
                                   verbose=verbose)
            meta.append(dict(
                specimen_id=sid,
                rin_MOhm_true=float(getattr(syn, "rin_MOhm", np.nan)),
                sag_ratio_true=float(getattr(syn, "sag_ratio", np.nan)),
                tau_m_true_ms=float(row["tau_m_true_ms"]),
                ss_n_repeats=int(cell_proto.ss_n_repeats),
                fs_ss_Hz=float(cell_proto.ss_sampling_rate_Hz),
                fs_ls_Hz=float(cell_proto.ls_sampling_rate_Hz),
                ss_noise_own=bool(ss_kw is not None),
                ok=True, reason="",
            ))
        except Exception as exc:  # noqa: BLE001 -- record, keep going
            if verbose:
                print(f"[gen] specimen {sid} FAILED: {type(exc).__name__}: {exc}")
            meta.append(dict(specimen_id=sid, rin_MOhm_true=np.nan,
                             sag_ratio_true=np.nan,
                             tau_m_true_ms=float(row["tau_m_true_ms"]),
                             ok=False, reason=f"{type(exc).__name__}: {exc}"))
    if clear_fn is not None:
        clear_fn()

    meta_df = pd.DataFrame(meta)
    meta_df.to_csv(archive_dir / "generated_meta.csv", index=False)
    if verbose:
        n_ok = int(meta_df["ok"].sum())
        print(f"[gen] cohort -> {n_ok}/{len(meta_df)} archives written under "
              f"{archive_dir}")
    return meta_df
