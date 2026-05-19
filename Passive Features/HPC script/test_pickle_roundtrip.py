"""
test_pickle_roundtrip.py

Smoke test for the pickling fix in passive_fitting_hpc_final.py.
Run this BEFORE launching the parallel HPC job. It reproduces, in a
single process, exactly the operation that multiprocessing.Pool
performs on the return trip from worker to parent.

Usage on the HPC (adjust the path as needed):
    python test_pickle_roundtrip.py

Expected output on success:
    [smoke] running fit_one_cell on specimen XYZ ...
    [smoke] sanitised gp_result.specs
    [smoke] pickle round-trip OK (NNN bytes)
    [smoke] unpickled result.cm_uF_per_cm2 = 2.110...
    [smoke] GP model preserved: True
    [smoke] PASS

If you see a MaybeEncodingError, AttributeError, or PicklingError,
the patch has NOT been applied correctly to passive_fitting_hpc_final.py.
"""

import sys
import pickle
import traceback

# --- 1. Import the (patched) script as a module --------------------------
# NB: this assumes you launch this test from the same directory as
# passive_fitting_hpc_final.py. Adjust sys.path if you keep it elsewhere.
import passive_fitting_hpc_final as P


def main() -> int:
    # --- 2. Load Phase-1 outputs the same way the main script does ------
    # You must adapt these three lines to whatever loader you use at the
    # bottom of passive_fitting_hpc_final.py (Phase 1). The point is to
    # get ONE (cell_data, opt_inputs) pair into memory.
    cells_data, opt_inputs = P.load_archive_minimal_one_cell()  # <-- ADAPT

    cd  = cells_data[0]
    oi  = opt_inputs[0]

    print(f"[smoke] running fit_one_cell on specimen {cd.specimen_id} ...")

    # --- 3. Run the worker code path manually --------------------------
    # We deliberately use the exact same call chain as _fit_worker so the
    # test catches the same closure embedding the pool would have caught.
    cell = P.build_neuron_model(cd.swc_path, F=1.9)
    r    = P.fit_one_cell(cell, cd, oi, F=1.9, n_calls=10, n_initial=10)

    # --- 4. Apply the same sanitation _fit_worker now does -------------
    r.neuron_cell = None
    P._sanitize_gp_result_for_pickle(r.gp_result)
    print("[smoke] sanitised gp_result.specs")

    # --- 5. Round-trip through pickle ----------------------------------
    try:
        blob = pickle.dumps(r, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"[smoke] FAIL at pickle.dumps: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    print(f"[smoke] pickle round-trip OK ({len(blob):,} bytes)")

    try:
        r2 = pickle.loads(blob)
    except Exception as e:
        print(f"[smoke] FAIL at pickle.loads: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # --- 6. Spot-check that the important Phase-3 state survived -------
    print(f"[smoke] unpickled result.cm_uF_per_cm2 = {r2.cm_uF_per_cm2:.4f}")
    gp_models_preserved = (
        r2.gp_result is not None
        and getattr(r2.gp_result, "models", None) is not None
        and len(r2.gp_result.models) > 0
    )
    print(f"[smoke] GP model preserved: {gp_models_preserved}")
    space_preserved = (
        r2.gp_result is not None
        and getattr(r2.gp_result, "space", None) is not None
    )
    print(f"[smoke] GP space preserved: {space_preserved}")

    if not gp_models_preserved or not space_preserved:
        print("[smoke] FAIL: Phase-3 state was stripped accidentally")
        return 1

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
