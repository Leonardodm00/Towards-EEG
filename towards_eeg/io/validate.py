#!/usr/bin/env python3
"""
validate.py -- validate_instance() and the report it returns.

WHAT THIS ESTABLISHES
---------------------
One entry point that runs the six registered invariants over an instance and
returns a structured report. TEEG_00 section 4 requires it to "skip gracefully
on an absent manifest", and at S0.5 there is no manifest anywhere, so skipping
is the NORMAL path rather than an edge case.

That makes the shape of the return value the whole design. A boolean would
report True for an instance in which nothing was checked. So:

  * every invariant appears in the report, always, whether or not it ran;
  * every clause of every invariant appears, with its own status;
  * every skip carries a reason naming the datum it wanted;
  * `report.ok` is True only if something was actually verified and nothing
    failed. An all-skipped report is NOT ok.

The last point is the one that matters. `ok` meaning "nothing failed" would be
True for an empty instance, and every caller would then treat an unvalidated
instance as a valid one.

SCOPE
-----
Six invariants: I-13c, I-14, I-15, I-16, I-17, I-18. TEEG_00 section 4 says
"I-13c...I-19"; I-19 does not exist (TEEG_02 section 3.3's table ends at I-18,
settled in Doc 5 E-7). I-1 through I-12 are deliberately not registered --
see invariants.py.

Standard library only, ASCII source, Python 3.8+.
"""

import json
import os

from .invariants import (FAIL, INVARIANTS, INVARIANT_IDS, PASS, SKIPPED,
                         ClauseResult, worst)

__all__ = ["InvariantReport", "InstanceReport", "validate_instance",
           "load_instance", "MANIFEST_FILENAME"]

MANIFEST_FILENAME = "manifest.json"


class InvariantReport(object):
    """The outcome of one invariant: a status, and one entry per clause."""

    __slots__ = ("iid", "statement", "catches", "contracts", "clauses", "status")

    def __init__(self, invariant, clauses):
        self.iid = invariant.iid
        self.statement = invariant.statement
        self.catches = invariant.catches
        self.contracts = invariant.contracts
        self.clauses = list(clauses)
        self.status = worst(c.status for c in self.clauses)

    @property
    def reasons(self):
        return [(c.clause, c.status, c.reason) for c in self.clauses]

    def __repr__(self):
        return "InvariantReport(%s, %s, %d clause(s))" % (
            self.iid, self.status, len(self.clauses))


class InstanceReport(object):
    """The outcome of validating one instance against all six invariants."""

    __slots__ = ("results", "instance_path", "available", "missing")

    def __init__(self, results, instance_path=None, available=(), missing=()):
        self.results = list(results)
        self.instance_path = instance_path
        self.available = tuple(sorted(available))
        self.missing = tuple(sorted(missing))

    def __iter__(self):
        return iter(self.results)

    def by_id(self, iid):
        for r in self.results:
            if r.iid == iid:
                return r
        raise KeyError("no such invariant in this report: %r" % iid)

    def counts(self):
        out = {PASS: 0, FAIL: 0, SKIPPED: 0}
        for r in self.results:
            out[r.status] += 1
        return out

    @property
    def failed(self):
        return [r.iid for r in self.results if r.status == FAIL]

    @property
    def skipped(self):
        return [r.iid for r in self.results if r.status == SKIPPED]

    @property
    def checked(self):
        """Clauses that actually ran, across all invariants."""
        return [(r.iid, c.clause) for r in self.results for c in r.clauses
                if c.status in (PASS, FAIL)]

    @property
    def ok(self):
        """True only if something was verified and nothing failed.

        An all-skipped report is NOT ok. Returning True for an instance in
        which no clause ran is how an unvalidated instance gets treated as a
        valid one, which is the failure this whole module is shaped to
        prevent.
        """
        return bool(self.checked) and not self.failed

    def format(self, verbose=False):
        lines = []
        width = max(len(i) for i in INVARIANT_IDS)
        for r in self.results:
            lines.append("%-*s  %-7s  %s" % (width, r.iid, r.status,
                                             r.catches))
            for c in r.clauses:
                if verbose or c.status != PASS:
                    lines.append("      %-8s %-42s %s"
                                 % (c.status, c.clause, c.reason))
                    for d in c.detail[:5]:
                        lines.append("               - %s" % d)
        counts = self.counts()
        lines.append("-" * 72)
        lines.append("%d pass, %d fail, %d skipped; %d clause(s) actually ran; "
                     "ok=%s" % (counts[PASS], counts[FAIL], counts[SKIPPED],
                                len(self.checked), self.ok))
        return "\n".join(lines)


def _empty_instance():
    return {}


def load_instance(instance_path):
    """Read an instance manifest, or return an empty instance if absent.

    Graceful absence is required by TEEG_00 section 4 and is the normal case
    at S0.5. It is NOT silent: the returned dict is empty, every invariant
    then skips with a reason naming what it wanted, and `report.ok` is False.

    S0 populates nothing, so this reads a manifest and does not construct one,
    does not read a .hoc, and does not touch the bank.
    """
    if instance_path is None:
        return _empty_instance(), None
    manifest = os.path.join(instance_path, MANIFEST_FILENAME)
    if not os.path.isfile(manifest):
        return _empty_instance(), manifest
    with open(manifest, "r", encoding="utf-8") as fh:
        return json.load(fh), manifest


def validate_instance(instance=None, instance_path=None):
    """Run the six registered invariants over an instance.

    Parameters
    ----------
    instance : dict or None
        The instance data. Recognised keys are the union of every
        `Invariant.requires` in the register; any key may be absent, and an
        absent key makes the invariants needing it skip with a stated reason.
    instance_path : str or None
        A frozen-instance directory. If given and `instance` is None, its
        manifest.json is read; if the file is absent, an empty instance is
        used and every invariant skips.

    Returns
    -------
    InstanceReport
    """
    resolved_manifest = None
    if instance is None:
        instance, resolved_manifest = load_instance(instance_path)
    if instance is None:
        instance = _empty_instance()

    wanted = set()
    for inv in INVARIANTS:
        wanted.update(inv.requires)
    available = set(k for k in wanted if instance.get(k) is not None)

    results = []
    for inv in INVARIANTS:
        try:
            clauses = inv.check(instance)
        except Exception as exc:                            # noqa: BLE001
            # An invariant that raises must not take the report down with it:
            # the other five still have something to say, and a raised
            # exception is a FAIL of that invariant, not of validation.
            clauses = [ClauseResult("check_raised", FAIL,
                                    "%s: %s" % (type(exc).__name__, exc))]
        if not clauses:
            clauses = [ClauseResult("no_clauses", FAIL,
                                    "the check returned no clauses; an "
                                    "invariant that reports nothing has not "
                                    "been evaluated")]
        results.append(InvariantReport(inv, clauses))

    return InstanceReport(results,
                          instance_path=resolved_manifest or instance_path,
                          available=available,
                          missing=wanted - available)
