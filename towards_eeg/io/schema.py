#!/usr/bin/env python3
"""
schema.py -- load and self-check the pinned contract schemas.

WHAT THIS ESTABLISHES
---------------------
The schemas in `schemas/` are data, not code: they are diffable against the
tables in TEEG_02 section 3.2 almost line for line, which is the property that
makes "does the implementation match the contract?" an answerable question
rather than a code review. This module is the thin layer that loads them,
checks they are internally consistent, and exposes the alphabets as Python
objects.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It validates SCHEMAS, not instances. Nothing here reads a bank, a manifest or
a morphology. S0 populates nothing (TEEG_00 section 4), and a loader that
quietly grew the ability to read data would be a defect fix wearing a schema's
clothes.

It also does not depend on `jsonschema` or any third party. pyproject.toml
declares no dependencies until B-4 is answered and S0.8 generates the
lockfiles from a real pip freeze, so everything here is standard library.

Standard library only, ASCII source, Python 3.8+.
"""

import json
import os

__all__ = [
    "SCHEMA_IDS", "ALPHABET_NAMES",
    "schema_dir", "load_alphabets", "load_schema", "load_all",
    "section_vocabulary_pairs", "check_schema_self_consistency",
    "SchemaError",
]


class SchemaError(ValueError):
    """A pinned schema is malformed or internally inconsistent."""


# The five files, named so that a missing one is a loud KeyError rather than a
# quietly shorter iteration (trap T-11).
SCHEMA_IDS = ("C-08", "C-09", "C-14", "C-15")

_FILENAMES = {
    "C-08": "C08_morphology_bank.json",
    "C-09": "C09_observed_synapses.json",
    "C-14": "C14_passive_parameters.json",
    "C-15": "C15_mechanism_spec.json",
}

ALPHABET_NAMES = ("sigma_cls", "sigma_dom", "sigma_syn", "mechanism_mode",
                  "cm_reference", "spine_status", "layer_source", "qc_status",
                  "parameter_source")


def schema_dir():
    """Directory holding the pinned schema files.

    Package-relative, resolved from __file__. This is NOT a configuration
    path and is deliberately not routed through config/paths.yaml: it locates
    the package's own data inside itself, and a package that had to be told
    where its own schemas live would be misconfigured rather than flexible.
    Declared in tools/s0_transform/s05_exit_scope.json so that S0.6's
    "no path literal outside config/" sweep does not have to guess.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas")


def _read(filename):
    path = os.path.join(schema_dir(), filename)
    if not os.path.isfile(path):
        raise SchemaError("pinned schema file is missing: %s" % filename)
    with open(path, "r", encoding="ascii") as fh:
        return json.load(fh)


def load_alphabets():
    """The fixed alphabets and enumerated vocabularies of TEEG_02."""
    return _read("alphabets.json")


def load_schema(contract_id):
    """One contract schema, by contract id (e.g. 'C-08')."""
    if contract_id not in _FILENAMES:
        raise SchemaError("no pinned schema for %r; known: %r"
                          % (contract_id, list(SCHEMA_IDS)))
    return _read(_FILENAMES[contract_id])


def load_all():
    """{contract_id: schema} for every pinned contract, plus alphabets."""
    out = dict((cid, load_schema(cid)) for cid in SCHEMA_IDS)
    out["alphabets"] = load_alphabets()
    return out


def section_vocabulary_pairs(alphabets=None):
    """The four admissible (class, domain) pairs of C8.1, as a frozenset.

    Returned as a frozenset of tuples because varsigma(nu) is a SET of pairs
    (TEEG_02 section 1), and because set containment is what invariants I-13c
    and I-16 are written in terms of.
    """
    a = alphabets if alphabets is not None else load_alphabets()
    return frozenset((p["cls"], p["dom"])
                     for p in a["section_vocabulary"]["pairs"])


def _enum_values(alphabets, name):
    if name == "section_vocabulary":
        return section_vocabulary_pairs(alphabets)
    if name not in alphabets:
        raise SchemaError("field refers to unknown alphabet %r" % name)
    return frozenset(alphabets[name]["values"])


def check_schema_self_consistency():
    """Check the pinned schemas against themselves and against TEEG_02.

    Returns a list of problem strings; empty means consistent. This is the
    schema-level analogue of the ledger's exit test: the schemas are the
    authority for everything downstream, so they get checked before anything
    is allowed to consume them.
    """
    problems = []
    alphabets = load_alphabets()

    # -- C8.1's biconditional, checked rather than trusted ------------------
    pairs = alphabets["section_vocabulary"]["pairs"]
    if len(pairs) != 4:
        problems.append("C8.1 declares %d pairs, expected exactly 4" % len(pairs))
    seen = set()
    for p in pairs:
        key = (p["cls"], p["dom"])
        if key in seen:
            problems.append("duplicate section-vocabulary pair %r" % (key,))
        seen.add(key)
        if p["cls"] not in alphabets["sigma_cls"]["values"]:
            problems.append("pair %r has a class outside Sigma_cls" % (key,))
        if p["dom"] not in alphabets["sigma_dom"]["values"]:
            problems.append("pair %r has a domain outside Sigma_dom" % (key,))
        # C8.1: sigma_dom == 'none' if and only if sigma_cls in {soma, axon}
        lhs = (p["dom"] == "none")
        rhs = (p["cls"] in ("soma", "axon"))
        if lhs != rhs:
            problems.append("pair %r violates the C8.1 biconditional" % (key,))
        for sub in p["get_idx_substrings"]:
            if sub not in p["section_array"]:
                problems.append(
                    "pair %r: get_idx substring %r does not occur in the "
                    "section array name %r, so LFPy.Cell.get_idx would not "
                    "select it" % (key, sub, p["section_array"]))

    # -- the synapse alphabet must not leak into the section vocabulary -----
    # This is defect O8 asserted at the level of the SCHEMA, before any bank
    # exists to violate it. I-16 is the same statement about a bank.
    syn = set(alphabets["sigma_syn"]["values"])
    if syn & set(p["cls"] for p in pairs):
        problems.append("Sigma_syn leaked into the compartment classes (O8)")
    excluded = alphabets["section_vocabulary"]["excluded_by_decision"]
    if set(excluded["synapse_type"]["labels"]) != syn:
        problems.append("the excluded synapse labels do not match Sigma_syn")

    # -- every field's declared enum must exist -----------------------------
    for cid in SCHEMA_IDS:
        sch = load_schema(cid)
        if sch.get("contract_id") != cid:
            problems.append("%s: file declares contract_id %r"
                            % (cid, sch.get("contract_id")))
        names = set()
        for fld in sch["fields"]:
            for req in ("name", "type", "units", "required"):
                if req not in fld:
                    problems.append("%s field %r lacks %r"
                                    % (cid, fld.get("name"), req))
            if fld["name"] in names:
                problems.append("%s declares field %r twice" % (cid, fld["name"]))
            names.add(fld["name"])
            for key in ("enum", "element_enum"):
                if key in fld:
                    try:
                        _enum_values(alphabets, fld[key])
                    except SchemaError as exc:
                        problems.append("%s.%s: %s" % (cid, fld["name"], exc))
        for k in sch.get("key") or []:
            if k not in names:
                problems.append("%s: key field %r is not among its fields"
                                % (cid, k))

    # -- C-14 and C-15 are keyed identically, deliberately ------------------
    if load_schema("C-14")["key"] != load_schema("C-15")["key"]:
        problems.append("C-14 and C-15 must be keyed identically "
                        "(TEEG_02 C-14 header); they are not")

    # -- C-15's callable registry -------------------------------------------
    vocab = section_vocabulary_pairs(alphabets)
    for entry in load_schema("C-15")["callable_registry"]:
        for pair in entry["applies_to"]:
            if tuple(pair) not in vocab:
                problems.append("callable %r applies to %r, which is not an "
                                "admissible section pair"
                                % (entry["name"], tuple(pair)))
        if entry.get("implemented", False):
            problems.append(
                "callable %r is marked implemented; C-15's implementation is "
                "deferred to S3 by closed decision" % entry["name"])

    # -- f_implied must stay marked QC-only ---------------------------------
    # TEEG_02 C-08: "QC only, never a multiplier". The multiplier is F(nu, s),
    # derived per segment from phi at build time (R7). Losing this flag is how
    # a cell-level scalar quietly becomes a per-segment correction.
    c08 = load_schema("C-08")
    fi = [f for f in c08["fields"] if f["name"] == "f_implied"]
    if not fi:
        problems.append("C-08 has no f_implied field")
    elif not fi[0].get("qc_only", False):
        problems.append("C-08.f_implied is not marked qc_only")
    if any(f["name"] == "F" or f["name"].lower() == "f_segment"
           for f in c08["fields"]):
        problems.append("C-08 stores F per segment; rule R7 forbids it")

    return problems
