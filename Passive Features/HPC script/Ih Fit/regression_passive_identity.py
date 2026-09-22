"""regression_passive_identity.py -- Stage 0 gate of the I_h fitting pipeline.

Proves that the monolith in THIS folder (`Ih Fit/`, the copy that later stages
edit) still reproduces the untouched `Biological Fit/` monolith on the passive
3-D path: same cell, same seed, same run-B loss configuration, identical loss
values at random parameters and identical gp_minimize output. It is re-run
after every stage of TEEG_Ih_fit_staged_plan.md; a difference means the
refactor changed the passive path, which is the one thing it must not do.

Run (either machine with NEURON + skopt):
    python regression_passive_identity.py --synthetic            # self-made archive cell
    python regression_passive_identity.py --archive-cell <ARCHIVE_ROOT>/<GROUP>/specimen_<id>

Expected last line:  regression_passive_identity: PASS (loss max|dL| = 0.00e+00; x identical; fun identical)

Run-B configuration (D-006 Q9, [user]): dt 0.1 ms on both protocols, long-step
window 60 ms after onset, the 2 smallest hyperpolarising long steps, 12 mV
deflection cap, r_in_target='peak', 'relative' weighting, SS window
(0.5, 100) ms with the exponential time-weight tau_w = 5 ms, F = 1.9.
Override any of them with the flags below to reproduce another run.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_synthetic_archive(new_dir: Path, out_root: Path, specimen_id: int = 900000001) -> Path:
    """A passive ball-and-stick cell written in the Phase-0 archive layout,
    which load_cell_from_archive reads unchanged (synthetic_ground_truth.py)."""
    sgt = load_module(new_dir / "synthetic_ground_truth.py", "sgt_for_regression")
    swc = sgt.write_ball_and_stick_swc(out_root / "bas.swc", soma_r_um=10.0,
                                       dend_len_um=400.0, dend_r_um=1.0,
                                       apic_len_um=800.0, apic_r_um=1.2, step_um=20.0)
    gt = sgt.GroundTruthParams(cm_uF_cm2=0.9, rm_Ohm_cm2=30000.0, ra_Ohm_cm=200.0,
                               e_pas_mV=-72.0, spine_factor_F=1.9)
    proto = sgt.ProtocolConfig(ss_n_repeats=10,
                               ls_hyp_amplitudes_pA=(-10.0, -30.0, -50.0, -70.0, -90.0))
    noise = sgt.NoiseConfig(sigma_mV=0.10, seed=1)
    syn = sgt.generate_synthetic_cell(swc, gt, proto=proto, noise=noise,
                                      specimen_id=specimen_id, verbose=True)
    spec_dir = out_root / ("specimen_%d" % specimen_id)
    sgt.write_archive_cell(syn, spec_dir, verbose=True)
    sgt._clear_neuron_sections()
    return spec_dir


def run_one(code_dir: Path, tag: str, specimen_dir: Path, args, thetas_log: np.ndarray):
    """Load one copy of the pipeline, fit the cell, evaluate the loss on the
    random parameter set. Returns (loss values, result.x, result.fun)."""
    mono = load_module(code_dir / "passive_fitting_hpc_fixed.py", "mono_" + tag)
    plst = load_module(code_dir / "passive_long_step_training.py", "plst_" + tag)
    plst.integrate_long_step(
        mono, n_long_train=args.n_long_train,
        max_ls_train_deflection_mV=args.ls_deflection_cap,
        r_in_target=args.r_in_target, ls_window_ms_after_onset=args.ls_window_ms,
        weighting=args.weighting, ss_window_ms=tuple(args.ss_window_ms),
        ss_time_weight=args.ss_time_weight, ss_tau_w_ms=args.ss_tau_w_ms,
        dt_brief_ms=args.dt_brief_ms, dt_long_ms=args.dt_long_ms, verbose=False)
    cd = mono.load_cell_from_archive(specimen_dir, n_avg_groups=1, verbose=False)
    oi = mono.prepare_optimiser_inputs(cd, fit_target="hyp")
    cell = mono.build_neuron_model(cd.swc_path, F=args.F)
    mono.assert_dt_enforced(cell, dt_values_ms=(args.dt_brief_ms, args.dt_long_ms), verbose=False)
    loss = mono._build_loss_function(cell=cell, train_bundles=oi.train_bundles,
                                     v_rest_mV=oi.v_rest_mV, train_window_ms=oi.train_window_ms)
    lvals = np.array([loss(*map(float, q)) for q in thetas_log], dtype=float)
    x = fun = None
    if not args.loss_only:
        fr = mono.fit_one_cell(cell, cd, oi, F=args.F, n_calls=args.n_calls,
                               n_initial=args.n_initial, seed=args.seed, verbose=False)
        if fr.gp_result is None:
            raise RuntimeError("[%s] fit failed: %s" % (tag, fr.error_message))
        x = np.array(fr.gp_result.x, dtype=float); fun = float(fr.gp_result.fun)
    cell.destroy()
    return lvals, x, fun


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", default=str(HERE.parent / "Biological Fit"))
    ap.add_argument("--new-dir", default=str(HERE))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--archive-cell", help="a Phase-0 specimen_<id> directory")
    src.add_argument("--synthetic", action="store_true", help="generate a passive synthetic archive cell")
    ap.add_argument("--n-calls", type=int, default=12)
    ap.add_argument("--n-initial", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-random", type=int, default=20, help="random log-theta points for the loss check")
    ap.add_argument("--loss-only", action="store_true", help="skip gp_minimize (fast)")
    ap.add_argument("--F", type=float, default=1.9)
    # run-B configuration (D-006 Q9)
    ap.add_argument("--dt-brief-ms", type=float, default=0.1)
    ap.add_argument("--dt-long-ms", type=float, default=0.1)
    ap.add_argument("--ls-window-ms", type=float, default=60.0)
    ap.add_argument("--n-long-train", type=int, default=2)
    ap.add_argument("--ls-deflection-cap", type=float, default=12.0)
    ap.add_argument("--r-in-target", default="peak")
    ap.add_argument("--weighting", default="relative")
    ap.add_argument("--ss-window-ms", type=float, nargs=2, default=(0.5, 100.0))
    ap.add_argument("--ss-time-weight", default="exp")
    ap.add_argument("--ss-tau-w-ms", type=float, default=5.0)
    args = ap.parse_args()

    ref_dir, new_dir = Path(args.ref_dir).resolve(), Path(args.new_dir).resolve()
    for d in (ref_dir, new_dir):
        if not (d / "passive_fitting_hpc_fixed.py").exists():
            sys.exit("no passive_fitting_hpc_fixed.py in %s" % d)
    if args.synthetic:
        out_root = Path(tempfile.mkdtemp(prefix="regression_synth_"))
        specimen_dir = make_synthetic_archive(new_dir, out_root)
    else:
        specimen_dir = Path(args.archive_cell).resolve()
    print("[regression] ref=%s\n[regression] new=%s\n[regression] cell=%s" % (ref_dir, new_dir, specimen_dir))

    rng = np.random.default_rng(args.seed + 12345)
    lo = np.log([0.3, 1e3, 50.0]); hi = np.log([3.0, 1e5, 1000.0])
    thetas_log = lo + rng.random((args.n_random, 3)) * (hi - lo)

    t0 = time.perf_counter()
    l_ref, x_ref, f_ref = run_one(ref_dir, "ref", specimen_dir, args, thetas_log)
    l_new, x_new, f_new = run_one(new_dir, "new", specimen_dir, args, thetas_log)
    dl = float(np.max(np.abs(l_ref - l_new)))
    ok = dl <= 1e-12 and np.all(np.isfinite(l_ref))
    msg = "loss max|dL| = %.2e over %d points" % (dl, args.n_random)
    if not args.loss_only:
        same_x = np.array_equal(x_ref, x_new); same_f = (f_ref == f_new)
        ok = ok and same_x and same_f
        msg += "; x %s; fun %s (ref %.6g)" % ("identical" if same_x else "DIFFERENT %s vs %s" % (x_ref, x_new),
                                              "identical" if same_f else "DIFFERENT %.12g vs %.12g" % (f_ref, f_new), f_ref)
    print("[regression] %.1f s" % (time.perf_counter() - t0))
    print("regression_passive_identity: %s (%s)" % ("PASS" if ok else "FAIL", msg))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
