#!/usr/bin/env python3
"""
test_s0_5_smoke.py -- correctness harness for towards_eeg/io/.

    python3 tools/test_s0_5_smoke.py --root .

WHY THIS EXISTS
---------------
At S0.5 there is no bank, no manifest and no instance, so every invariant
skips on every real invocation. Six functions that have only ever returned
SKIPPED are indistinguishable from six functions that are wrong.

So every clause of every invariant is exercised HERE, in both directions:
a synthetic instance in which all fourteen clauses pass, and one mutation per
clause that violates it in exactly the way TEEG_02 section 3.3 says that
invariant catches. A mutation that survives is a check that does not work.

The fixtures are synthetic by necessity and by decision. TEEG_00 section 1 is
explicit that the morphology .hoc files are not attached to an S0 chat, so
I-15 and I-18 are written against the abstract geometry record of
towards_eeg/io/geometry.py and tested against records built in Python. No
morphology file enters S0.

Standard library only, ASCII source, Python 3.8+.
"""

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from towards_eeg.io import geometry as G           # noqa: E402
from towards_eeg.io import invariants as INV       # noqa: E402
from towards_eeg.io import schema as SCH           # noqa: E402
from towards_eeg.io import validate as VAL         # noqa: E402

FOUR_PAIRS = [["soma", "none"], ["axon", "none"],
              ["dend", "apical"], ["dend", "basal"]]


# ---------------------------------------------------------------------------
# the fixture in which every clause passes
# ---------------------------------------------------------------------------

def good_geometry(nid):
    """One soma of somatic calibre at the alignment origin, four thin neurites.

    Diameters follow the measured case in TEEG_02 C8.1: the soma of
    489469961 is 11.22 um against a median of 0.40 um over all sections.
    """
    soma = G.Section("s0", "soma[0]", 11.22, (0.0, 0.0, 0.0), None)
    kids = [G.Section("d%d" % i, "apic_dend[%d]" % i, 0.40,
                      (10.0 * (i + 1), 0.0, 0.0), "s0") for i in range(4)]
    return G.Geometry(nid, [soma] + kids, alignment_origin_um=(0.0, 0.0, 0.0))


def good_instance():
    return {
        "c08_rows": [
            {"nid": 1, "qc_status": "pass", "spine_status": "pruned",
             "spine_area_density": "phi/1.json",
             "section_vocabulary": copy.deepcopy(FOUR_PAIRS)},
            {"nid": 2, "qc_status": "pass_low_confidence",
             "spine_status": "pruned", "spine_area_density": "phi/2.json",
             "section_vocabulary": copy.deepcopy(FOUR_PAIRS)},
        ],
        "c14_rows": [
            {"layer": "L5", "gtype": "PC", "cm": 1.0, "rm": 20000.0,
             "ra": 100.0, "cm_reference": "shaft", "source": "fitted",
             "provenance": "fit-2026-07"},
        ],
        "c15_rows": [
            {"layer": "L5", "gtype": "PC", "mechanism_mode": "active",
             "section_key": ["dend", "apical"], "mechanism": "Ih",
             "parameter": "gbar", "value": "ih_sigmoid_L5",
             "source": "literature", "provenance": "C15.3"},
        ],
        "f_applied_at_build_time": True,
        "spike_config": {"dt_grid_ms": 0.125, "t_stop_ms": 1000.0},
        # 2^-3 ms grid; the bound is 2^(24-3) = 2097152 ms. Every sampled time
        # below is an exact binary32 value.
        "spike_times_ms": [0.0, 0.125, 0.25, 512.0, 1000.0],
        "geometries": {1: good_geometry(1), 2: good_geometry(2)},
    }


def clause_status(inst, iid, clause):
    rep = VAL.validate_instance(inst)
    for c in rep.by_id(iid).clauses:
        if c.clause == clause:
            return c.status, c.reason
    raise KeyError("no clause %r on %s" % (clause, iid))


def all_clause_statuses(inst):
    rep = VAL.validate_instance(inst)
    return dict(((r.iid, c.clause), c.status)
                for r in rep.results for c in r.clauses)


# ---------------------------------------------------------------------------
# one mutation per clause -- each must turn exactly that clause FAIL
# ---------------------------------------------------------------------------

def m_i13c_domain(inst):
    # The mechanism row targets (dend, apical); remove it from nid 1's
    # vocabulary. The mechanism is then applied to nothing, silently.
    inst["c08_rows"][0]["section_vocabulary"] = [
        p for p in inst["c08_rows"][0]["section_vocabulary"]
        if tuple(p) != ("dend", "apical")]
    return "I-13c", "mech_dict_domain_within_vocabulary"


def m_i13c_passive(inst):
    inst["c15_rows"][0]["mechanism_mode"] = "passive"    # C15.2
    return "I-13c", "passive_mode_implies_empty"


def m_i14_dt(inst):
    inst["spike_config"]["dt_grid_ms"] = 0.1            # not 2^-k
    return "I-14", "dt_grid_is_negative_power_of_two"


def m_i14_tstop(inst):
    inst["spike_config"]["dt_grid_ms"] = 2.0 ** -20     # bound becomes 16 ms
    return "I-14", "t_stop_within_binary32_range"


def m_i14_times(inst):
    inst["spike_times_ms"] = list(inst["spike_times_ms"]) + [0.1]
    return "I-14", "sampled_times_round_trip"


def m_i15_two_somata(inst):
    g = inst["geometries"][1]
    g.sections.append(G.Section("s1", "soma[1]", 11.0, (500.0, 0.0, 0.0), "s0"))
    return "I-15", "exactly_one_somatic_calibre_section"


def m_i15_not_root(inst):
    # The 489469961 failure: a somatic-calibre section that is not the root.
    g = inst["geometries"][1]
    g.sections[0].parent = "d0"
    g.sections[1].parent = None
    return "I-15", "somatic_section_is_root"


def m_i16_synapse_label(inst):
    # Defect O8 at the level of the bank: a synapse type used as a section
    # label.
    inst["c08_rows"][0]["section_vocabulary"].append(["exc_syn", "none"])
    return "I-16", "no_synapse_label_in_vocabulary"


def m_i16_outside_c81(inst):
    # (dend, none) violates the C8.1 biconditional without being a synapse
    # label -- a different failure, which must be reported as a different one.
    inst["c08_rows"][0]["section_vocabulary"].append(["dend", "none"])
    return "I-16", "vocabulary_within_C8_1"


def m_i17_phi_lost(inst):
    # Spine OMISSION masquerading as spine pruning: the area is simply gone.
    inst["c08_rows"][0]["spine_area_density"] = ""
    return "I-17", "phi_present_where_pruned"


def m_i17_effective_cm(inst):
    # An effective cm applied together with a derived F double-counts the
    # spine correction by a factor of roughly 1.7 to 2.4.
    inst["c14_rows"][0]["cm_reference"] = "effective"
    return "I-17", "cm_is_shaft_referenced_where_F_applied"


def m_i18_no_geometry_id(inst):
    g = inst["geometries"][1]
    for sec in g.sections:
        sec.diameter_um = 1.0                       # no section of calibre
    return "I-18", "geometry_identification_exists"


def m_i18_name_lost(inst):
    # The exact O8 signature: the soma emitted under a synapse label, so no
    # section is name-identifiable at all.
    inst["geometries"][1].sections[0].name = "exc_syn[0]"
    return "I-18", "name_identification_is_unique"


def m_i18_disagree(inst):
    # Name says one section, diameter and position say another.
    g = inst["geometries"][1]
    g.sections[0].name = "apic_dend[99]"
    g.sections[1].name = "soma[0]"
    return "I-18", "identifications_agree"


MUTATIONS = [m_i13c_domain, m_i13c_passive, m_i14_dt, m_i14_tstop, m_i14_times,
             m_i15_two_somata, m_i15_not_root, m_i16_synapse_label,
             m_i16_outside_c81, m_i17_phi_lost, m_i17_effective_cm,
             m_i18_no_geometry_id, m_i18_name_lost, m_i18_disagree]


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_00_baseline_every_clause_passes():
    st = all_clause_statuses(good_instance())
    bad = sorted(k for k, v in st.items() if v != INV.PASS)
    if bad:
        return False, "fixture is not clean: %r" % bad[:5]
    return True, "all %d clause(s) of all %d invariant(s) pass on the fixture" \
                 % (len(st), len(INV.INVARIANTS))


def check_01_baseline_report_is_ok():
    rep = VAL.validate_instance(good_instance())
    if not rep.ok:
        return False, "a fully satisfied instance is not ok: %r" % rep.failed
    if rep.skipped:
        return False, "clauses skipped on a complete instance: %r" % rep.skipped
    return True, "ok=True, %d clause(s) ran, none skipped" % len(rep.checked)


def check_02_empty_instance_is_not_ok():
    """The load-bearing property: an unvalidated instance must not read as
    valid. `ok` meaning 'nothing failed' would be True here."""
    rep = VAL.validate_instance({})
    if rep.ok:
        return False, "MUTATION SURVIVED: an empty instance reports ok=True"
    if len(rep.skipped) != len(INV.INVARIANTS):
        return False, "not every invariant skipped: %r" % rep.skipped
    if rep.checked:
        return False, "clauses ran on an empty instance: %r" % rep.checked
    return True, ("all %d invariant(s) skipped, 0 clause(s) ran, ok=False"
                  % len(INV.INVARIANTS))


def check_03_every_skip_states_its_reason():
    rep = VAL.validate_instance({})
    silent = [(r.iid, c.clause) for r in rep.results for c in r.clauses
              if c.status == INV.SKIPPED and not c.reason]
    if silent:
        return False, "skips with no stated reason: %r" % silent
    unnamed = [(r.iid, c.clause) for r in rep.results for c in r.clauses
               if c.status == INV.SKIPPED and "absent" not in c.reason
               and "cannot" not in c.reason]
    if unnamed:
        return False, "skips that do not name the missing datum: %r" % unnamed
    return True, "every skip names the datum it wanted"


def check_04_absent_manifest_skips_gracefully():
    """TEEG_00 section 4: skip gracefully on an absent manifest."""
    tmp = tempfile.mkdtemp(prefix="s05_")
    try:
        rep = VAL.validate_instance(instance_path=tmp)
        if rep.ok:
            return False, "an absent manifest produced ok=True"
        if len(rep.skipped) != len(INV.INVARIANTS):
            return False, "not every invariant skipped: %r" % rep.skipped
        with open(os.path.join(tmp, VAL.MANIFEST_FILENAME), "w",
                  encoding="ascii", newline="\n") as fh:
            json.dump({"c08_rows": [
                {"nid": 1, "qc_status": "pass", "spine_status": "retained",
                 "section_vocabulary": FOUR_PAIRS}]}, fh)
        rep2 = VAL.validate_instance(instance_path=tmp)
        if rep2.by_id("I-16").status != INV.PASS:
            return False, "a present manifest did not feed I-16: %s" \
                          % rep2.by_id("I-16").status
        return True, ("absent manifest -> all skipped, ok=False; present "
                      "manifest -> I-16 evaluated")
    finally:
        shutil.rmtree(tmp)


def check_05_partial_data_does_not_round_up_to_pass():
    """I-14's first two clauses are checkable from the config alone. An
    instance with the config but no spike times must NOT report I-14 as
    passing: two of three conjuncts is not the invariant."""
    inst = {"spike_config": {"dt_grid_ms": 0.125, "t_stop_ms": 1000.0}}
    rep = VAL.validate_instance(inst)
    r = rep.by_id("I-14")
    st = dict((c.clause, c.status) for c in r.clauses)
    if st.get("dt_grid_is_negative_power_of_two") != INV.PASS:
        return False, "clause 1 did not run on config-only data"
    if st.get("sampled_times_round_trip") != INV.SKIPPED:
        return False, "clause 3 did not skip without spike times"
    if r.status != INV.SKIPPED:
        return False, ("MUTATION SURVIVED: I-14 reports %s having verified "
                       "two of three conjuncts" % r.status)
    return True, "two conjuncts verified, one skipped, invariant reports SKIPPED"


def check_06_a_raising_invariant_becomes_a_failure():
    """An invariant that raises must not take the report down with it, and
    must not vanish from it either."""
    def boom(data):
        raise RuntimeError("synthetic")
    victim = INV.INVARIANTS[0]
    saved = victim.check
    try:
        victim.check = boom
        rep = VAL.validate_instance(good_instance())
        r = rep.by_id(victim.iid)
        if r.status != INV.FAIL:
            return False, "a raising invariant reported %s" % r.status
        if rep.ok:
            return False, "the report is ok despite a raising invariant"
        others = [x.iid for x in rep.results if x.iid != victim.iid
                  and x.status == INV.PASS]
        if len(others) != len(INV.INVARIANTS) - 1:
            return False, "the other invariants did not survive: %r" % others
    finally:
        victim.check = saved
    return True, "raised -> FAIL on that invariant, the other %d still ran" \
                 % (len(INV.INVARIANTS) - 1)


def check_07_register_matches_the_declared_scope():
    expected = ("I-13c", "I-14", "I-15", "I-16", "I-17", "I-18")
    if INV.INVARIANT_IDS != expected:
        return False, "register is %r, expected %r" % (INV.INVARIANT_IDS, expected)
    if "I-19" in INV.INVARIANT_IDS:
        return False, "I-19 is registered; it does not exist (Doc 5 E-7)"
    missing = [inv.iid for inv in INV.INVARIANTS
               if not inv.statement or not inv.catches or not inv.requires]
    if missing:
        return False, "invariants without a statement, a catch or a "\
                      "requirement: %r" % missing
    return True, "six invariants, each with a statement, a catch and declared "\
                 "data requirements; I-19 absent"


def check_08_schema_self_consistency_is_clean():
    p = SCH.check_schema_self_consistency()
    return (not p), ("the five pinned schema files are internally consistent"
                     if not p else "%r" % p[:3])


def check_09_schema_checker_catches_a_broken_C8_1():
    """MUTATION: the C8.1 biconditional, the qc_only flag on f_implied, and a
    callable marked implemented must each be refused."""
    import towards_eeg.io.schema as S
    real = S._read
    cases = []

    def patched(kind):
        def _read(filename):
            doc = copy.deepcopy(real(filename))
            if kind == "c81" and filename == "alphabets.json":
                doc["section_vocabulary"]["pairs"][0]["dom"] = "apical"
            if kind == "qc" and filename.startswith("C08"):
                for f in doc["fields"]:
                    if f["name"] == "f_implied":
                        f["qc_only"] = False
            if kind == "impl" and filename.startswith("C15"):
                doc["callable_registry"][0]["implemented"] = True
            if kind == "substr" and filename == "alphabets.json":
                doc["section_vocabulary"]["pairs"][2]["get_idx_substrings"] = ["soma"]
            return doc
        return _read

    try:
        for kind in ("c81", "qc", "impl", "substr"):
            S._read = patched(kind)
            if not S.check_schema_self_consistency():
                cases.append(kind)
    finally:
        S._read = real
    if cases:
        return False, "MUTATION SURVIVED: schema checker accepted %r" % cases
    return True, "4 broken schemas refused, unbroken schema accepted"


def check_10_geometry_record_rejects_malformed_input():
    bad = []
    try:
        G.Section("s", "soma[0]", 0.0, (0, 0, 0), None); bad.append("zero diameter")
    except G.GeometryError:
        pass
    try:
        G.Section("s", "soma[0]", 1.0, (0, 0), None); bad.append("2d position")
    except G.GeometryError:
        pass
    try:
        G.Geometry(1, []); bad.append("no sections")
    except G.GeometryError:
        pass
    try:
        s = G.Section("a", "soma[0]", 1.0, (0, 0, 0), "nope")
        G.Geometry(1, [s]); bad.append("dangling parent")
    except G.GeometryError:
        pass
    try:
        s = G.Section("a", "soma[0]", 1.0, (0, 0, 0), None)
        G.Geometry(1, [s, s]); bad.append("duplicate sid")
    except G.GeometryError:
        pass
    if bad:
        return False, "MUTATION SURVIVED: accepted %r" % bad
    return True, "5 malformed geometry inputs refused"


def check_11_calibre_threshold_is_a_parameter_not_a_constant():
    """The somatic-calibre ratio is declared, not baked in: a bank whose
    diameter distribution disagrees must produce an argument, not a silent
    reclassification."""
    inst = good_instance()
    st = clause_status(inst, "I-15", "exactly_one_somatic_calibre_section")
    if st[0] != INV.PASS:
        return False, "fixture fails at the default ratio"
    inst["somatic_calibre_ratio"] = 100.0        # nothing is somatic now
    st2 = clause_status(inst, "I-15", "exactly_one_somatic_calibre_section")
    if st2[0] != INV.FAIL:
        return False, "the ratio is not honoured by the invariant"
    return True, "the threshold is read from the instance, default %g" \
                 % G.SOMATIC_CALIBRE_RATIO


CHECKS = [check_00_baseline_every_clause_passes,
          check_01_baseline_report_is_ok,
          check_02_empty_instance_is_not_ok,
          check_03_every_skip_states_its_reason,
          check_04_absent_manifest_skips_gracefully,
          check_05_partial_data_does_not_round_up_to_pass,
          check_06_a_raising_invariant_becomes_a_failure,
          check_07_register_matches_the_declared_scope,
          check_08_schema_self_consistency_is_clean,
          check_09_schema_checker_catches_a_broken_C8_1,
          check_10_geometry_record_rejects_malformed_input,
          check_11_calibre_threshold_is_a_parameter_not_a_constant]


def run_mutations():
    out = []
    for mut in MUTATIONS:
        inst = good_instance()
        try:
            iid, clause = mut(inst)
            st = all_clause_statuses(inst)
            got = st.get((iid, clause))
            if got is None:
                ok, msg = False, "no clause %r on %s" % (clause, iid)
            elif got != INV.FAIL:
                ok, msg = False, ("MUTATION SURVIVED: %s.%s reports %s"
                                  % (iid, clause, got))
            else:
                collateral = sorted(k for k, v in st.items()
                                    if v != INV.PASS and k != (iid, clause))
                ok = True
                msg = "caught by %s.%s%s" % (
                    iid, clause,
                    "" if not collateral
                    else " (also non-pass: %s)" % ", ".join(
                        "%s.%s" % k for k in collateral[:3]))
        except Exception as exc:                            # noqa: BLE001
            ok, msg = False, "raised %s: %s" % (type(exc).__name__, exc)
        out.append((mut.__name__, ok, msg))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Correctness harness for io/.")
    ap.add_argument("--root", default=".")
    ap.parse_args(argv)

    results = []
    for c in CHECKS:
        try:
            ok, msg = c()
        except Exception as exc:                            # noqa: BLE001
            ok, msg = False, "raised %s: %s" % (type(exc).__name__, exc)
        results.append((c.__name__, ok, msg))
    results.extend(run_mutations())

    width = max(len(n) for n, _, _ in results)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, msg in results:
        print("%-*s  %s   %s" % (width, name, "PASS" if ok else "FAIL", msg))
    print("-" * (width + 40))
    print("%d/%d checks passed" % (n_ok, len(results)))
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
