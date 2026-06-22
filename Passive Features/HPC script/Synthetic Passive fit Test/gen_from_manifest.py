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
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ===========================================================================
#  PURE mapping: manifest row -> generation kwargs  (NEURON-free, testable)
# ===========================================================================
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
    """Manifest row -> NoiseConfig kwargs (per-cell jittered levels + seed)."""
    return dict(
        sigma_mV=float(row["noise_sigma_mV"]),
        baseline_sigma_mV=float(row["noise_baseline_sigma_mV"]),
        drift_sigma_mV=float(row["noise_drift_sigma_mV"]),
        seed=int(row["noise_seed"]),
    )


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


def build_proto(sgt, *, ss_n_repeats: int, ls_hyp_amplitudes_pA: Sequence[float]):
    return sgt.ProtocolConfig(ss_n_repeats=int(ss_n_repeats),
                              ls_hyp_amplitudes_pA=tuple(float(a) for a in ls_hyp_amplitudes_pA))


# ===========================================================================
#  Orchestration: generate one cohort's archives
# ===========================================================================
def generate_group(
    group_df: pd.DataFrame,
    archive_dir: Path | str,
    *,
    sgt,                         # synthetic_ground_truth module
    mono,                        # passive_fitting_hpc_fixed module (build_neuron_model)
    ss_n_repeats: int = 30,
    ls_hyp_amplitudes_pA: Sequence[float] = (-10., -30., -50., -70., -90.),
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
    proto = build_proto(sgt, ss_n_repeats=ss_n_repeats,
                        ls_hyp_amplitudes_pA=ls_hyp_amplitudes_pA)

    meta: List[dict] = []
    for _, row in group_df.iterrows():
        if clear_fn is not None:
            clear_fn()
        sid = int(row["specimen_id"])
        gt = build_gt(sgt, row_to_gt_kwargs(row))
        noise = build_noise(sgt, row_to_noise_kwargs(row))
        factory = mono.build_neuron_model if use_builder_factory(row) else None
        try:
            syn = sgt.generate_synthetic_cell(
                Path(row["swc"]), gt, proto=proto, noise=noise,
                specimen_id=sid, passive_cell_factory=factory, verbose=verbose)
            sgt.write_archive_cell(syn, archive_dir / f"specimen_{sid}",
                                   verbose=verbose)
            meta.append(dict(
                specimen_id=sid,
                rin_MOhm_true=float(getattr(syn, "rin_MOhm", np.nan)),
                sag_ratio_true=float(getattr(syn, "sag_ratio", np.nan)),
                tau_m_true_ms=float(row["tau_m_true_ms"]),
                ok=True, reason="",
            ))
        except Exception as exc:  # noqa: BLE001 — record, keep going
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
