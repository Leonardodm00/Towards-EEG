"""hoc_qc -- stage S1 gate: does the emitted .hoc actually conduct?

WHAT THIS MODULE ESTABLISHES
----------------------------
An emitted morphology is not usable merely because it parses and loads. It must
be ONE electrically connected tree, in which a somatic current step produces a
voltage transient that reaches every compartment and decays away from the
injection site. This module runs that test on an LFPy.Cell and returns a
verdict. It writes nothing, plots nothing and reads no files.

WHY NOT A VOLTAGE THRESHOLD
---------------------------
The obvious test -- "every branch must reach X mV" -- is wrong for these cells
and would reject healthy ones. Deitcher et al. 2017 (Cereb Cortex 27:5398-5414,
doi:10.1093/cercor/bhx226) measured steady-state dendrite-to-soma voltage
attenuation factors of 54.64 and 20.20 on two exemplar human L2/3 pyramidal
cells, with a mean dendritic cable length of 0.99 +/- 0.24, and explicitly
predict that active mechanisms are needed to compensate for it. In a PASSIVE
model a distal tip therefore legitimately sees a very small fraction of the
somatic deflection, and that is biology, not a defect.

The criteria are therefore STRUCTURAL, not calibrational:

  C1 FINITE     no NaN and no inf anywhere in vmem. Catches an unintegrable
                model (degenerate geometry, zero diameter, zero length).
  C2 CONNECTED  every compartment deflects by a strictly non-zero amount. A
                section not attached to the tree sits at e_pas forever and
                gives exactly zero. This is the criterion that earns its keep:
                the exporter's 'connect child(0), parent(1)' convention plus
                re-decomposition can orphan a section silently, and nothing
                else in the pipeline would notice.
  C3 MONOTONE   peak |dV| does not INCREASE away from the soma, within a loose
                tolerance, along every root-to-tip path. For a passive tree
                driven at one point this holds exactly; a violation means an
                inverted or mis-parented connection.
  C4 SOMA       the somatic deflection lies inside a generous sanity window.
                Catches an absurd soma area, not a mis-calibrated one.

QC POLICY (decision Q6): SOFT except where the model is structurally impossible.

  C1 or C2 violated -> 'fail'
        the morphology is not one integrable tree. Nothing is committed.
  C3 or C4 violated -> 'pass_low_confidence'
        suspicious, with a machine-readable reason. Not a stop.

STIMULUS, AND WHY THESE NUMBERS
-------------------------------
A current STEP, not a steady-state solve, because the thing being verified is
that a transient propagates. Defaults: 0.1 nA for 100 ms after a 5 ms baseline,
integrated at dt = 0.25 ms to tstop = 110 ms.

  amplitude   0.1 nA against a somatic input resistance near 47.68 +/- 15.26
              MOhm (Deitcher et al. 2017, same source) puts the somatic
              deflection around 5 mV -- large enough to be far above solver
              noise at every compartment, small enough to be meaningless as a
              perturbation.
  duration    100 ms is about 8 tau_m at the default Rm below, so the transient
              is settled to better than 0.1 percent and the peak is effectively
              the steady state.
  dt          0.25 ms rather than the LFPy default 2^-4 ms purely for memory:
              vmem is (totnsegs x n_samples) float64, and on a cell with 20000
              segments this is about 70 MB at 0.25 ms against 280 MB at the
              default. The model is passive, so the coarser step costs nothing
              that matters to these criteria.

SEGMENTATION IS UNCHANGED BY THIS MODULE
----------------------------------------
The flattened lfpy_idx space is fixed by (cm, Ra, lambda_f, d_lambda,
nsegs_method). Inserting the 'pas' mechanism does not change nseg. The cell
built by passive_cell_factory below therefore has the SAME index space as
alignment.default_cell_factory would produce for the same tuple, which is what
makes it safe for align_and_export to build ONE cell, gate on it here, and then
hand the same object to snap_synapses.

O7 HYGIENE
----------
build_segment_index and build_section_tree are IMPORTED from
synapse_redirect_audit, never reimplemented. They are covered by that module's
tests R1 and R4.

ASCII only, LF only, no top-level side effects.
"""

import numpy as np

import synapse_redirect_audit as sra


MODULE_VERSION = "hoc_qc-1.0.0"

QC_PASS = "pass"
QC_LOW = "pass_low_confidence"
QC_FAIL = "fail"

# Rm = tau_m / cm. Human L2/3 comparators from the project literature:
# tau_m = 12.03 +/- 1.79 ms (Deitcher et al. 2017) with cm = 0.5 uF/cm^2
# (Eyal et al. 2016) give Rm = 12.03e-3 / 0.5e-6 = 24060 Ohm cm^2, rounded here.
# The corresponding g_pas is 1 / Rm = 4.167e-5 S/cm^2.
DEFAULT_RM_OHM_CM2 = 24000.0
DEFAULT_E_PAS_MV = -70.0

DEFAULT_AMP_NA = 0.1
DEFAULT_DELAY_MS = 5.0
DEFAULT_DUR_MS = 100.0
DEFAULT_DT_MS = 0.25
DEFAULT_TSTOP_MS = 110.0

# C2: a truly detached section returns EXACTLY e_pas, so any positive floor
# separates it from an attached compartment. Kept far below solver noise.
CONNECTED_TOL_MV = 1e-9

# C3: loose by instruction. A child may exceed its parent by 5 percent plus
# 1 uV before it is called a violation.
MONOTONE_RTOL = 0.05
MONOTONE_ATOL_MV = 1e-3

# C4: generous by design. This window rejects absurdity, not miscalibration.
SOMA_DV_MIN_MV = 0.01
SOMA_DV_MAX_MV = 200.0


# --------------------------------------------------------------------------- #
#  1. Building the passive cell                                                #
# --------------------------------------------------------------------------- #
def passive_cell_factory(hoc_path, cm, Ra, Rm=DEFAULT_RM_OHM_CM2,
                         e_pas=DEFAULT_E_PAS_MV, lambda_f=100.0, d_lambda=0.1,
                         nsegs_method="lambda_f", dt=DEFAULT_DT_MS,
                         tstart=0.0, tstop=DEFAULT_TSTOP_MS, v_init=None,
                         custom_fun=None, custom_fun_args=None):
    """Instantiate the LFPy cell this module simulates.

    `cm`, `Ra` and `Rm` are all REQUIRED in spirit: cm and Ra are positional
    because they fix the segmentation (see the module docstring), and Rm is
    defaulted only because no stage upstream of S5 produces it.

    Rm is the specific membrane resistance in Ohm cm^2; g_pas = 1 / Rm.

    In LFPy 2.3.7 `passive_parameters` carries ONLY g_pas and e_pas; Ra and cm
    are separate top-level kwargs and are silently ignored if nested. They are
    passed top-level here, exactly as alignment.default_cell_factory does.

    v_init defaults to e_pas so the cell starts at rest and the only deflection
    in vmem is the one this module injects.
    """
    # Guards BEFORE the import, deliberately: a bad parameter must be a
    # ValueError on any machine, not an ImportError on one without NEURON.
    if float(Rm) <= 0.0:
        raise ValueError("Rm must be positive, got %r" % (Rm,))
    if float(cm) <= 0.0 or float(Ra) <= 0.0:
        raise ValueError("cm and Ra must be positive, got %r, %r" % (cm, Ra))

    import LFPy                                    # deferred: no NEURON in tests

    g_pas = 1.0 / float(Rm)
    if v_init is None:
        v_init = float(e_pas)

    kwargs = dict(morphology=hoc_path,
                  passive=True,
                  passive_parameters=dict(g_pas=float(g_pas),
                                          e_pas=float(e_pas)),
                  Ra=float(Ra), cm=float(cm),
                  nsegs_method=nsegs_method, lambda_f=float(lambda_f),
                  d_lambda=float(d_lambda),
                  dt=float(dt), tstart=float(tstart), tstop=float(tstop),
                  v_init=float(v_init), delete_sections=True)
    if custom_fun is not None:
        kwargs["custom_fun"] = custom_fun
        kwargs["custom_fun_args"] = custom_fun_args or [{}]
    return LFPy.Cell(**kwargs)


def soma_index(cell):
    """Flattened index of a somatic compartment, with a documented fallback.

    The exporter makes the soma the topology root (I-15) and LFPy flattens in
    allseclist order, so index 0 is the soma on any file this pipeline emits.
    cell.get_idx('soma') is preferred anyway, because relying on the ordering
    would couple this module to the exporter's emission order.
    """
    getter = getattr(cell, "get_idx", None)
    if callable(getter):
        try:
            idx = np.atleast_1d(np.asarray(getter("soma")))
            if idx.size:
                return int(idx[0])
        except (KeyError, TypeError, ValueError):
            pass
    return 0


def _default_stim_factory(cell, idx, amp_nA, delay_ms, dur_ms):
    import LFPy                                    # deferred: no NEURON in tests
    return LFPy.StimIntElectrode(cell=cell, idx=int(idx), pptype="IClamp",
                                 amp=float(amp_nA), dur=float(dur_ms),
                                 delay=float(delay_ms), record_current=False)


def simulate_step(cell, amp_nA=DEFAULT_AMP_NA, delay_ms=DEFAULT_DELAY_MS,
                  dur_ms=DEFAULT_DUR_MS, idx=None, stim_factory=None):
    """Inject a somatic current step and record vmem in EVERY compartment.

    Mutates `cell`: after this returns, cell.vmem is (totnsegs x n_samples) and
    cell.tvec is (n_samples,). Returns the stimulus object so the caller can
    keep it alive for the lifetime of the cell.

    `stim_factory` exists so the smoke test can substitute a stub and assert the
    wiring without NEURON.
    """
    if idx is None:
        idx = soma_index(cell)
    factory = stim_factory or _default_stim_factory
    stim = factory(cell, idx=int(idx), amp_nA=float(amp_nA),
                   delay_ms=float(delay_ms), dur_ms=float(dur_ms))
    cell.simulate(rec_vmem=True)
    return stim


# --------------------------------------------------------------------------- #
#  2. The four criteria, each testable on plain arrays                         #
# --------------------------------------------------------------------------- #
def peak_deflection(vmem, tvec, baseline_end_ms):
    """Per-compartment baseline and peak absolute deflection from it.

    The baseline is the LAST sample strictly before `baseline_end_ms`, taken
    per compartment rather than assumed equal to e_pas, so a cell that was not
    initialised exactly at rest is still measured against its own start.

    Returns (v0, dv), both length totnsegs, dv >= 0, in mV.
    """
    v = np.asarray(vmem, dtype=float)
    t = np.asarray(tvec, dtype=float)
    if v.ndim != 2:
        raise ValueError("vmem must be 2-D (totnsegs x n_samples), got %r"
                         % (v.shape,))
    if v.shape[1] != t.shape[0]:
        raise ValueError("vmem has %d samples but tvec has %d"
                         % (v.shape[1], t.shape[0]))
    pre = t < float(baseline_end_ms)
    if not bool(pre.any()):
        raise ValueError("no sample before baseline_end_ms=%r; the stimulus "
                         "delay must exceed at least one dt" % (baseline_end_ms,))
    v0 = v[:, pre][:, -1]
    dv = np.abs(v - v0[:, None]).max(axis=1)
    return v0, dv


def check_finite(vmem):
    """C1. Returns (ok, report)."""
    v = np.asarray(vmem, dtype=float)
    bad = ~np.isfinite(v)
    n_bad = int(bad.sum())
    rep = {"n_nonfinite": n_bad,
           "n_compartments_affected": int(bad.any(axis=1).sum()) if v.size else 0}
    if n_bad:
        first = np.argwhere(bad)[0]
        rep["first_bad_compartment"] = int(first[0])
        rep["first_bad_sample"] = int(first[1])
    return (n_bad == 0), rep


def check_connected(dv, tol=CONNECTED_TOL_MV):
    """C2. Returns (ok, report). A detached section gives dv exactly 0."""
    d = np.asarray(dv, dtype=float)
    dead = np.nonzero(~(d > float(tol)))[0]
    rep = {"tol_mV": float(tol),
           "n_zero_deflection": int(dead.size),
           "zero_deflection_idx": [int(i) for i in dead[:20]],
           "dv_min_mV": float(d.min()) if d.size else float("nan"),
           "dv_max_mV": float(d.max()) if d.size else float("nan")}
    return (dead.size == 0), rep


def section_slices(sec_of_idx):
    """Contiguous [start, stop) flattened-index range of each section.

    LFPy flattens as `for sec in allseclist: for seg in sec`, so every section
    owns one contiguous run. This asserts that rather than assuming it.
    """
    s = np.asarray(sec_of_idx, dtype=int)
    out = {}
    i, n = 0, s.size
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        k = int(s[i])
        if k in out:
            raise ValueError("section %d owns a non-contiguous index run" % k)
        out[k] = (i, j)
        i = j
    return out


def check_monotone(dv, sec_of_idx, sec_names, parent_of,
                   rtol=MONOTONE_RTOL, atol=MONOTONE_ATOL_MV):
    """C3. Peak |dV| must not increase away from the soma.

    Tested locally in two places, which together are equivalent to global
    monotonicity along every root-to-tip path and cost O(totnsegs):

      within  segment m+1 against segment m inside one section, proximal to
              distal, which is LFPy's within-section ordering;
      across  the proximal segment of a child section against the distal
              segment of its parent.

    A violation is recorded when dv_distal > dv_proximal * (1 + rtol) + atol.

    Sections present in the cell but absent from `parent_of` are counted and
    reported rather than raising: a name mismatch between the .hoc and the
    section table is a real problem, but it is reported softly per Q6.
    """
    d = np.asarray(dv, dtype=float)
    slices = section_slices(sec_of_idx)
    name_to_k = {}
    for k, nm in enumerate(sec_names):
        name_to_k[str(nm)] = k

    viol = []
    unknown = []

    for k, (a, b) in sorted(slices.items()):
        seg = d[a:b]
        for m in range(seg.size - 1):
            if seg[m + 1] > seg[m] * (1.0 + rtol) + atol:
                viol.append({"kind": "within", "section": str(sec_names[k]),
                             "seg_proximal": m, "seg_distal": m + 1,
                             "dv_proximal_mV": float(seg[m]),
                             "dv_distal_mV": float(seg[m + 1])})

    for k, nm in enumerate(sec_names):
        nm = str(nm)
        if nm not in parent_of:
            unknown.append(nm)
            continue
        pnm = parent_of.get(nm)
        if pnm is None:
            continue
        pk = name_to_k.get(str(pnm))
        if pk is None or pk not in slices or k not in slices:
            unknown.append(nm)
            continue
        pa, pb = slices[pk]
        ca, cb = slices[k]
        dv_par = float(d[pb - 1])
        dv_chi = float(d[ca])
        if dv_chi > dv_par * (1.0 + rtol) + atol:
            viol.append({"kind": "across", "section": nm, "parent": str(pnm),
                         "dv_parent_distal_mV": dv_par,
                         "dv_child_proximal_mV": dv_chi})

    def _excess(v):
        if v["kind"] == "within":
            p, c = v["dv_proximal_mV"], v["dv_distal_mV"]
        else:
            p, c = v["dv_parent_distal_mV"], v["dv_child_proximal_mV"]
        return c - (p * (1.0 + rtol) + atol)

    viol.sort(key=_excess, reverse=True)
    rep = {"rtol": float(rtol), "atol_mV": float(atol),
           "n_violations": len(viol),
           "worst_violations": viol[:10],
           "n_unknown_sections": len(unknown),
           "unknown_sections": unknown[:10]}
    return (len(viol) == 0), rep


def check_soma(dv, idx_soma, lo_mV=SOMA_DV_MIN_MV, hi_mV=SOMA_DV_MAX_MV):
    """C4. The somatic deflection is inside a generous sanity window."""
    d = np.asarray(dv, dtype=float)
    if not (0 <= int(idx_soma) < d.size):
        raise IndexError("soma index %r outside 0..%d" % (idx_soma, d.size - 1))
    val = float(d[int(idx_soma)])
    ok = bool(float(lo_mV) <= val <= float(hi_mV))
    return ok, {"idx_soma": int(idx_soma), "dv_soma_mV": val,
                "window_mV": [float(lo_mV), float(hi_mV)]}


# --------------------------------------------------------------------------- #
#  3. The gate                                                                 #
# --------------------------------------------------------------------------- #
def check_propagation(cell, parent_of, baseline_end_ms=DEFAULT_DELAY_MS,
                      idx_soma=None, connected_tol_mV=CONNECTED_TOL_MV,
                      monotone_rtol=MONOTONE_RTOL,
                      monotone_atol_mV=MONOTONE_ATOL_MV,
                      soma_lo_mV=SOMA_DV_MIN_MV, soma_hi_mV=SOMA_DV_MAX_MV):
    """Run C1-C4 on an ALREADY SIMULATED cell and return the verdict.

    `cell` must carry vmem and tvec, i.e. simulate_step has been called.
    `parent_of` is the section parent map from
    synapse_redirect_audit.build_section_tree(section_table).

    Returns a dict with qc_status, reasons (machine-readable strings) and the
    per-criterion reports. Never raises on a bad cell; raises only on a caller
    error such as an unsimulated cell.
    """
    vmem = getattr(cell, "vmem", None)
    tvec = getattr(cell, "tvec", None)
    if vmem is None or tvec is None:
        raise ValueError("cell has no vmem/tvec: call simulate_step first")

    sec_of_idx, sec_names, sec_length_um = sra.build_segment_index(cell)

    res = {"module_version": MODULE_VERSION,
           "qc_status": QC_PASS, "reasons": [],
           "totnsegs": int(len(sec_of_idx)),
           "n_sections": int(len(sec_names)),
           "cable_length_um": float(sum(sec_length_um.values()))}

    ok1, rep1 = check_finite(vmem)
    res["C1_finite"] = rep1
    if not ok1:
        res["qc_status"] = QC_FAIL
        res["reasons"].append("nonfinite_vmem:%d" % rep1["n_nonfinite"])
        return res                       # dv is meaningless once vmem is broken

    v0, dv = peak_deflection(vmem, tvec, baseline_end_ms)
    res["dv_min_mV"] = float(dv.min())
    res["dv_max_mV"] = float(dv.max())

    ok2, rep2 = check_connected(dv, tol=connected_tol_mV)
    res["C2_connected"] = rep2
    if not ok2:
        res["qc_status"] = QC_FAIL
        res["reasons"].append("zero_deflection_compartments:%d"
                              % rep2["n_zero_deflection"])

    ok3, rep3 = check_monotone(dv, sec_of_idx, sec_names, parent_of,
                               rtol=monotone_rtol, atol=monotone_atol_mV)
    res["C3_monotone"] = rep3
    if not ok3 and res["qc_status"] != QC_FAIL:
        res["qc_status"] = QC_LOW
    if not ok3:
        res["reasons"].append("monotonicity_violations:%d" % rep3["n_violations"])
    if rep3["n_unknown_sections"]:
        if res["qc_status"] == QC_PASS:
            res["qc_status"] = QC_LOW
        res["reasons"].append("sections_absent_from_section_table:%d"
                              % rep3["n_unknown_sections"])

    if idx_soma is None:
        idx_soma = soma_index(cell)
    ok4, rep4 = check_soma(dv, idx_soma, lo_mV=soma_lo_mV, hi_mV=soma_hi_mV)
    res["C4_soma"] = rep4
    if not ok4:
        if res["qc_status"] == QC_PASS:
            res["qc_status"] = QC_LOW
        res["reasons"].append("soma_deflection_out_of_window:%.4g"
                              % rep4["dv_soma_mV"])

    return res


def gate_hoc(hoc_path, section_table, cm, Ra, Rm=DEFAULT_RM_OHM_CM2,
             e_pas=DEFAULT_E_PAS_MV, lambda_f=100.0, d_lambda=0.1,
             nsegs_method="lambda_f", amp_nA=DEFAULT_AMP_NA,
             delay_ms=DEFAULT_DELAY_MS, dur_ms=DEFAULT_DUR_MS,
             dt=DEFAULT_DT_MS, tstop=DEFAULT_TSTOP_MS,
             cell_factory=None, stim_factory=None, keep_cell=True,
             **check_kwargs):
    """Build, simulate, judge. Returns (report, cell) with cell=None if released.

    This is the one call align_and_export makes. `keep_cell=True` hands the
    LIVE cell back so the caller can reuse it for snap_synapses instead of
    paying for a second instantiation; the caller then owns its lifetime.
    """
    factory = cell_factory or passive_cell_factory
    cell = factory(hoc_path, cm=cm, Ra=Ra, Rm=Rm, e_pas=e_pas,
                   lambda_f=lambda_f, d_lambda=d_lambda,
                   nsegs_method=nsegs_method, dt=dt, tstop=tstop)
    try:
        simulate_step(cell, amp_nA=amp_nA, delay_ms=delay_ms, dur_ms=dur_ms,
                      stim_factory=stim_factory)
        parent_of = sra.build_section_tree(section_table)
        rep = check_propagation(cell, parent_of, baseline_end_ms=delay_ms,
                                **check_kwargs)
    except Exception as exc:                                   # noqa: BLE001
        # A raise here is itself a verdict: the model could not be built,
        # integrated or reconciled with its own section table. Soft per Q6 in
        # form -- a report, not a traceback -- but 'fail' in substance, so
        # nothing downstream is committed.
        _release_cell(cell)
        rep = {"module_version": MODULE_VERSION, "qc_status": QC_FAIL,
               "reasons": ["qc_raised:%s" % type(exc).__name__],
               "error": str(exc)}
        _stamp_settings(rep, cm, Ra, Rm, e_pas, amp_nA, delay_ms, dur_ms,
                        dt, tstop)
        return rep, None

    _stamp_settings(rep, cm, Ra, Rm, e_pas, amp_nA, delay_ms, dur_ms, dt, tstop)
    if keep_cell:
        return rep, cell
    _release_cell(cell)
    return rep, None


def _stamp_settings(rep, cm, Ra, Rm, e_pas, amp_nA, delay_ms, dur_ms, dt, tstop):
    """Record what was run, so a verdict is reproducible from the report alone."""
    rep["stimulus"] = {"amp_nA": float(amp_nA), "delay_ms": float(delay_ms),
                       "dur_ms": float(dur_ms), "dt_ms": float(dt),
                       "tstop_ms": float(tstop)}
    rep["passive"] = {"cm_uF_cm2": float(cm), "Ra_ohm_cm": float(Ra),
                      "Rm_ohm_cm2": float(Rm),
                      "g_pas_S_cm2": 1.0 / float(Rm), "e_pas_mV": float(e_pas)}
    return rep


def _release_cell(cell):
    """Best-effort teardown, mirroring alignment.align_and_export's own."""
    closer = getattr(cell, "__del__", None)
    if callable(closer):
        try:
            closer()
        except Exception:                                      # noqa: BLE001
            pass
