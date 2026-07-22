#!/usr/bin/env python3
"""
invariants.py -- the six cross-contract invariants assigned to S0.5.

    I-13c  for all (l,g), for all assignable nu: dom(D_{l,g}) subset varsigma(nu)
    I-14   dt_grid = 2^-k; T_stop <= 2^(24-k) ms; float64(float32(t)) == t
    I-15   for all nu: exactly one section of somatic calibre exists AND is the
           topology root
    I-16   for all nu: varsigma(nu) intersect Sigma_syn is empty
    I-17   phi present for every nu with spine_status 'pruned'; and
           cm_reference == 'shaft' wherever F is applied at build time
    I-18   for all nu: the somatic section is identified by diameter AND
           position, not by name, and the two identifications agree

TEEG_00 section 4 says "I-13c...I-19". I-19 DOES NOT EXIST: TEEG_02 section
3.3's table ends at I-18, and Doc 5 E-7 settled the scope at six. The twelve
remaining invariants I-1...I-12 are NOT registered here, by decision on
21 July 2026: registering twelve unimplemented entries would be machinery with
no test to prove it right, and the honest signal is that this module holds six.

WHY EVERY CHECK RETURNS CLAUSES
-------------------------------
Four of the six are conjunctions. If a conjunction reports one status, a
conjunct that could not be checked is indistinguishable from one that passed,
and the invariant reports PASS having verified part of itself. That is the
failure family recorded as G-5 and trap T-11 in Doc 9. So each check returns
one ClauseResult per conjunct, and the invariant's status is the worst of
them: FAIL if any clause failed, else SKIPPED if any clause could not run,
else PASS.

WHAT IS AND IS NOT CHECKED AT S0.5
----------------------------------
There is no bank, no manifest and no instance at S0.5, so on any real
invocation every invariant here reports SKIPPED naming the datum it wanted.
That is the designed behaviour, not a gap. What makes it safe is that each
skip carries a declared reason, and that tools/test_s0_5_smoke.py exercises
every clause of every invariant against SYNTHETIC data in both directions:
one fixture that satisfies it, one that violates it in exactly the way
TEEG_02 section 3.3 says it catches.

Standard library only, ASCII source, Python 3.8+.
"""

import math
import struct

from .geometry import Geometry, SOMATIC_CALIBRE_RATIO
from .schema import load_alphabets, section_vocabulary_pairs

__all__ = ["PASS", "FAIL", "SKIPPED", "ClauseResult", "Invariant",
           "INVARIANTS", "INVARIANT_IDS", "worst"]

PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"

# Worst-first: the status of a conjunction is the worst of its conjuncts.
_SEVERITY = {FAIL: 2, SKIPPED: 1, PASS: 0}


def worst(statuses):
    """The status of a conjunction, given its conjuncts' statuses."""
    statuses = list(statuses)
    if not statuses:
        return SKIPPED
    return max(statuses, key=lambda s: _SEVERITY[s])


class ClauseResult(object):
    """One conjunct of one invariant."""

    __slots__ = ("clause", "status", "reason", "detail")

    def __init__(self, clause, status, reason, detail=None):
        if status not in _SEVERITY:
            raise ValueError("unknown status %r" % status)
        self.clause = clause
        self.status = status
        self.reason = reason
        self.detail = detail if detail is not None else []

    def __repr__(self):
        return "ClauseResult(%r, %s, %r)" % (self.clause, self.status, self.reason)


class Invariant(object):
    """One registered invariant: its statement, what it catches, what it needs."""

    __slots__ = ("iid", "statement", "catches", "contracts", "requires", "check")

    def __init__(self, iid, statement, catches, contracts, requires, check):
        self.iid = iid
        self.statement = statement
        self.catches = catches
        self.contracts = contracts
        self.requires = tuple(requires)
        self.check = check

    def __repr__(self):
        return "Invariant(%s)" % self.iid


def _missing(data, keys):
    return [k for k in keys if data.get(k) is None]


def _skip_all(clauses, missing):
    reason = "absent from the instance: %s" % ", ".join(missing)
    return [ClauseResult(c, SKIPPED, reason) for c in clauses]


# ---------------------------------------------------------------------------
# I-13c -- R6, the silent no-op
# ---------------------------------------------------------------------------

def check_i13c(data):
    """dom(D_{l,g}) subset varsigma(nu), for each fixed (l,g) and each
    assignable nu.

    TEEG_02 calls I-13c the only CATEGORY check in the table: the only
    assertion catching a failure that would otherwise emit no error at all.
    A mechanism specified for a section label the morphology does not carry
    is applied to nothing, silently, and the cell simulates without complaint.
    """
    clauses = ["mech_dict_domain_within_vocabulary", "passive_mode_implies_empty"]
    missing = _missing(data, ("c15_rows", "c08_rows"))
    if missing:
        return _skip_all(clauses, missing)

    c15 = data["c15_rows"]
    c08 = data["c08_rows"]

    # "Assignable" nu: declared by the instance if it knows, otherwise every
    # bank row that passed QC. Declared rather than assumed, because silently
    # including qc_status == 'fail' would make the invariant stricter than the
    # project intends and silently excluding rows would make it weaker.
    if data.get("assignable_nids") is not None:
        assignable = set(data["assignable_nids"])
    else:
        assignable = set(r["nid"] for r in c08
                         if r.get("qc_status") in ("pass", "pass_low_confidence"))

    vocab_by_nid = dict((r["nid"], set(tuple(p) for p in r["section_vocabulary"]))
                        for r in c08)

    dom = {}
    for row in c15:
        dom.setdefault((row["layer"], row["gtype"]), set()).add(
            tuple(row["section_key"]))

    violations = []
    for (layer, gtype), keys in sorted(dom.items()):
        for nid in sorted(assignable):
            vocab = vocab_by_nid.get(nid)
            if vocab is None:
                violations.append("(%s,%s): nid %r is assignable but has no "
                                  "C-08 row" % (layer, gtype, nid))
                continue
            orphan = keys - vocab
            if orphan:
                violations.append(
                    "(%s,%s) specifies %s for nid %r, whose vocabulary is %s "
                    "-- the mechanism would be applied to nothing"
                    % (layer, gtype, sorted(orphan), nid, sorted(vocab)))
    out = [ClauseResult(
        clauses[0], FAIL if violations else PASS,
        "%d violation(s) of R6" % len(violations) if violations
        else "every mech_dict domain lies within the vocabulary of every "
             "assignable morphology (%d class(es) x %d morphology(s))"
             % (len(dom), len(assignable)),
        violations[:10])]

    # C15.2: mechanism_mode == 'passive' implies D is empty for all (l,g).
    passive_rows = [r for r in c15 if r.get("mechanism_mode") == "passive"]
    out.append(ClauseResult(
        clauses[1], FAIL if passive_rows else PASS,
        "%d mechanism row(s) declare mechanism_mode 'passive'; C15.2 requires "
        "the mech_dict to be empty in passive mode" % len(passive_rows)
        if passive_rows else "no mechanism row is declared in passive mode (C15.2)"))
    return out


# ---------------------------------------------------------------------------
# I-14 -- C-12 precision loss and collisions
# ---------------------------------------------------------------------------

def _is_negative_power_of_two(value):
    """True if value == 2^-k for some integer k >= 0."""
    if not (value > 0) or not math.isfinite(value):
        return False
    m, e = math.frexp(value)
    return m == 0.5 and e <= 1


def _round_trips_through_float32(t):
    """float64(float32(t)) == t, exactly."""
    return struct.unpack("f", struct.pack("f", t))[0] == t


def check_i14(data):
    """dt_grid = 2^-k; T_stop <= 2^(24-k) ms; sampled times representable.

    The first two clauses are properties of two configuration numbers and are
    checkable with no data at all. The third needs actual spike times, so an
    instance carrying only the config gets PASS, PASS, SKIPPED -- and the
    invariant as a whole reports SKIPPED, because it was not fully verified.
    """
    clauses = ["dt_grid_is_negative_power_of_two", "t_stop_within_binary32_range",
               "sampled_times_round_trip"]
    out = []

    missing = _missing(data, ("spike_config",))
    if missing:
        out.extend(_skip_all(clauses[:2], missing))
        k = None
    else:
        cfg = data["spike_config"]
        dt = cfg.get("dt_grid_ms")
        t_stop = cfg.get("t_stop_ms")
        ok_dt = dt is not None and _is_negative_power_of_two(dt)
        out.append(ClauseResult(
            clauses[0], PASS if ok_dt else FAIL,
            "dt_grid = %r ms = 2^%d" % (dt, round(math.log2(dt))) if ok_dt
            else "dt_grid = %r is not 2^-k for integer k >= 0" % (dt,)))
        k = round(-math.log2(dt)) if ok_dt else None
        if k is None or t_stop is None:
            out.append(ClauseResult(
                clauses[1], SKIPPED,
                "cannot bound T_stop without a valid dt_grid and t_stop_ms"))
        else:
            bound = float(2 ** (24 - k))
            ok = t_stop <= bound
            out.append(ClauseResult(
                clauses[1], PASS if ok else FAIL,
                "T_stop = %r ms <= 2^(24-%d) = %r ms" % (t_stop, k, bound) if ok
                else "T_stop = %r ms exceeds 2^(24-%d) = %r ms; spike times "
                     "beyond it are not representable in binary32 and distinct "
                     "times would collide" % (t_stop, k, bound)))

    times = data.get("spike_times_ms")
    if times is None:
        out.append(ClauseResult(
            clauses[2], SKIPPED,
            "absent from the instance: spike_times_ms -- the first two "
            "clauses bound the grid, this one checks the actual samples"))
    else:
        bad = [t for t in times if not _round_trips_through_float32(t)]
        out.append(ClauseResult(
            clauses[2], FAIL if bad else PASS,
            "%d of %d sampled time(s) do not survive float64(float32(t))"
            % (len(bad), len(list(times))) if bad
            else "all %d sampled time(s) round-trip through binary32 exactly"
                 % len(list(times)),
            [repr(t) for t in bad[:10]]))
    return out


# ---------------------------------------------------------------------------
# I-15 -- the 489469961 root failure
# ---------------------------------------------------------------------------

def check_i15(data):
    """Exactly one section of somatic calibre exists, and it is the topology root."""
    clauses = ["exactly_one_somatic_calibre_section", "somatic_section_is_root"]
    missing = _missing(data, ("geometries",))
    if missing:
        return _skip_all(clauses, missing)

    ratio = data.get("somatic_calibre_ratio", SOMATIC_CALIBRE_RATIO)
    geoms = data["geometries"]
    wrong_count, not_root = [], []
    for nid, geom in sorted(geoms.items()):
        if not isinstance(geom, Geometry):
            wrong_count.append("nid %r is not a Geometry record" % (nid,))
            continue
        fat = geom.somatic_calibre_sections(ratio=ratio)
        if len(fat) != 1:
            wrong_count.append(
                "nid %r has %d section(s) of somatic calibre (>= %g x median "
                "diameter %g um): %r"
                % (nid, len(fat), ratio, geom.median_diameter_um(),
                   [s.name for s in fat[:5]]))
            continue
        if fat[0].parent is not None:
            not_root.append("nid %r: somatic section %r has parent %r, so it "
                            "is not the topology root"
                            % (nid, fat[0].name, fat[0].parent))
    n = len(geoms)
    return [
        ClauseResult(clauses[0], FAIL if wrong_count else PASS,
                     "%d morphology(s) do not have exactly one somatic-calibre "
                     "section" % len(wrong_count) if wrong_count
                     else "all %d morphology(s) have exactly one" % n,
                     wrong_count[:10]),
        ClauseResult(clauses[1], FAIL if not_root else PASS,
                     "%d morphology(s) whose somatic section is not the root"
                     % len(not_root) if not_root
                     else "the somatic section is the topology root in all %d" % n,
                     not_root[:10]),
    ]


# ---------------------------------------------------------------------------
# I-16 -- regression to synapse-as-section (defect O8)
# ---------------------------------------------------------------------------

def check_i16(data):
    """varsigma(nu) intersect Sigma_syn is empty, for all nu.

    Stated in TEEG_02 as a set intersection. Implemented as two clauses,
    because a vocabulary can fail this in two distinct ways: by carrying a
    synapse label, and by carrying anything else outside the four admissible
    pairs of C8.1. Reporting them separately says which.
    """
    clauses = ["no_synapse_label_in_vocabulary", "vocabulary_within_C8_1"]
    missing = _missing(data, ("c08_rows",))
    if missing:
        return _skip_all(clauses, missing)

    alphabets = load_alphabets()
    sigma_syn = set(alphabets["sigma_syn"]["values"])
    admissible = section_vocabulary_pairs(alphabets)

    leaked, outside = [], []
    for row in data["c08_rows"]:
        vocab = set(tuple(p) for p in row["section_vocabulary"])
        for pair in sorted(vocab):
            if set(pair) & sigma_syn:
                leaked.append("nid %r carries %r as a SECTION LABEL; synapse "
                              "type is a per-synapse attribute of C-09 (O8)"
                              % (row["nid"], pair))
            elif pair not in admissible:
                outside.append("nid %r carries %r, outside the four admissible "
                               "pairs of C8.1" % (row["nid"], pair))
    n = len(data["c08_rows"])
    return [
        ClauseResult(clauses[0], FAIL if leaked else PASS,
                     "%d synapse label(s) found in section vocabularies"
                     % len(leaked) if leaked
                     else "no synapse label appears as a section label in any "
                          "of %d bank row(s)" % n, leaked[:10]),
        ClauseResult(clauses[1], FAIL if outside else PASS,
                     "%d label(s) outside C8.1" % len(outside) if outside
                     else "every vocabulary lies within the four pairs of C8.1",
                     outside[:10]),
    ]


# ---------------------------------------------------------------------------
# I-17 -- spine-area loss; F double-counting
# ---------------------------------------------------------------------------

def check_i17(data):
    """phi present wherever spines were pruned; cm_reference == 'shaft'
    wherever F is applied at build time.

    Two independent failures with one number between them. Losing phi loses
    the spine area outright (spine OMISSION rather than spine PRUNING).
    Applying an effective cm together with a derived F double-counts by a
    factor of roughly 1.7 to 2.4, and the two capacitances are
    indistinguishable by inspection.
    """
    clauses = ["phi_present_where_pruned", "cm_is_shaft_referenced_where_F_applied"]
    out = []

    missing = _missing(data, ("c08_rows",))
    if missing:
        out.extend(_skip_all(clauses[:1], missing))
    else:
        lost = [r["nid"] for r in data["c08_rows"]
                if r.get("spine_status") == "pruned"
                and not r.get("spine_area_density")]
        pruned = [r for r in data["c08_rows"]
                  if r.get("spine_status") == "pruned"]
        out.append(ClauseResult(
            clauses[0], FAIL if lost else PASS,
            "%d pruned morphology(s) carry no spine_area_density; their spine "
            "area is lost, which is spine OMISSION, not pruning" % len(lost)
            if lost else "phi is present for all %d pruned morphology(s)"
                         % len(pruned),
            [repr(n) for n in lost[:10]]))

    missing = _missing(data, ("c14_rows", "f_applied_at_build_time"))
    if missing:
        out.extend(_skip_all(clauses[1:], missing))
    else:
        if not data["f_applied_at_build_time"]:
            out.append(ClauseResult(
                clauses[1], PASS,
                "F is not applied at build time in this instance, so the "
                "cm_reference precondition does not bind"))
        else:
            bad = ["(%s,%s) cm_reference=%r" % (r["layer"], r["gtype"],
                                                r.get("cm_reference"))
                   for r in data["c14_rows"]
                   if r.get("cm_reference") != "shaft"]
            out.append(ClauseResult(
                clauses[1], FAIL if bad else PASS,
                "%d passive class(es) are not shaft-referenced while F is "
                "applied at build time; the spine correction is counted twice"
                % len(bad) if bad
                else "all %d passive class(es) are shaft-referenced"
                     % len(data["c14_rows"]), bad[:10]))
    return out


# ---------------------------------------------------------------------------
# I-18 -- O8, label precedence
# ---------------------------------------------------------------------------

def check_i18(data):
    """The somatic section is identified by diameter and position, and the
    name-based and geometry-based identifications agree.

    TEEG_02 section 3.3: I-18 is DELIBERATELY REDUNDANT with I-15, because
    the failure mode is that one of the two identifications is wrong. A
    validator that collapses them has removed the point, so both are
    registered and both are checked.
    """
    clauses = ["geometry_identification_exists", "name_identification_is_unique",
               "identifications_agree"]
    missing = _missing(data, ("geometries",))
    if missing:
        return _skip_all(clauses, missing)

    ratio = data.get("somatic_calibre_ratio", SOMATIC_CALIBRE_RATIO)
    token = data.get("soma_name_token", "soma")
    geoms = data["geometries"]

    no_geom, bad_name, disagree = [], [], []
    for nid, geom in sorted(geoms.items()):
        by_geom = geom.identify_soma_by_geometry(ratio=ratio)
        if by_geom is None:
            no_geom.append("nid %r: no section of somatic calibre, so the "
                           "geometry-based identification is empty" % (nid,))
            continue
        by_name = geom.identify_soma_by_name(token=token)
        if len(by_name) != 1:
            bad_name.append(
                "nid %r: %d section(s) named with %r. Zero is the O8 signature "
                "-- the soma was emitted under a synapse label and lost its "
                "class" % (nid, len(by_name), token))
            continue
        if by_name[0].sid != by_geom.sid:
            disagree.append(
                "nid %r: name says %r, geometry says %r (diameter %g um, %g um "
                "from the alignment origin)"
                % (nid, by_name[0].name, by_geom.name, by_geom.diameter_um,
                   by_geom.distance_from(geom.alignment_origin_um)))
    n = len(geoms)
    return [
        ClauseResult(clauses[0], FAIL if no_geom else PASS,
                     "%d morphology(s) have no geometry-based identification"
                     % len(no_geom) if no_geom
                     else "a somatic section is identifiable by diameter and "
                          "position in all %d morphology(s)" % n, no_geom[:10]),
        ClauseResult(clauses[1], FAIL if bad_name else PASS,
                     "%d morphology(s) do not have exactly one section named "
                     "with %r" % (len(bad_name), token) if bad_name
                     else "exactly one section is name-identified in all %d" % n,
                     bad_name[:10]),
        ClauseResult(clauses[2], FAIL if disagree else PASS,
                     "%d morphology(s) where the two identifications disagree"
                     % len(disagree) if disagree
                     else "the two independent identifications agree in all %d" % n,
                     disagree[:10]),
    ]


# ---------------------------------------------------------------------------
# the register
# ---------------------------------------------------------------------------

INVARIANTS = (
    Invariant(
        "I-13c",
        "for all (l,g), for all assignable nu: dom(D_{l,g}) subset varsigma(nu)",
        "R6 violation -- the silent no-op",
        ("C-08", "C-15"),
        ("c15_rows", "c08_rows"),
        check_i13c),
    Invariant(
        "I-14",
        "dt_grid = 2^-k; T_stop <= 2^(24-k) ms; float64(float32(t)) == t",
        "C-12 precision loss and collisions",
        ("C-12",),
        ("spike_config", "spike_times_ms"),
        check_i14),
    Invariant(
        "I-15",
        "for all nu: exactly one section of somatic calibre exists and is the "
        "topology root",
        "the 489469961 root failure",
        ("C-08",),
        ("geometries",),
        check_i15),
    Invariant(
        "I-16",
        "for all nu: varsigma(nu) intersect Sigma_syn is empty",
        "regression to synapse-as-section",
        ("C-08",),
        ("c08_rows",),
        check_i16),
    Invariant(
        "I-17",
        "phi present for every nu with spine_status 'pruned'; and "
        "cm_reference == 'shaft' wherever F is applied at build time",
        "spine-area loss; F double-counting",
        ("C-08", "C-14"),
        ("c08_rows", "c14_rows", "f_applied_at_build_time"),
        check_i17),
    Invariant(
        "I-18",
        "for all nu: the somatic section is identified by diameter and "
        "position, not by name, and the two identifications agree",
        "O8, label precedence",
        ("C-08",),
        ("geometries",),
        check_i18),
)

INVARIANT_IDS = tuple(inv.iid for inv in INVARIANTS)
