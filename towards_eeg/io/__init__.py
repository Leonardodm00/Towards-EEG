"""
Contract readers and validate_instance().

Pins the schemas of C-08 (morphology bank), C-09 (observed synapses),
C-14 (passive parameters) and C-15 (active mechanisms) from
TEEG_02_interface_contracts_v3.md, and registers the six cross-contract
invariants assigned to S0.5: I-13c, I-14, I-15, I-16, I-17, I-18.

SCHEMAS ONLY. S0 populates nothing: nothing here reads a bank, builds an
instance, or touches a morphology. validate_instance() skips gracefully --
and visibly, with a stated reason per clause -- when the data it needs is
absent, which at S0.5 is always.
"""

from .geometry import Geometry, GeometryError, Section, SOMATIC_CALIBRE_RATIO
from .invariants import FAIL, INVARIANT_IDS, INVARIANTS, PASS, SKIPPED
from .schema import (SCHEMA_IDS, SchemaError, check_schema_self_consistency,
                     load_all, load_alphabets, load_schema,
                     section_vocabulary_pairs)
from .validate import (InstanceReport, InvariantReport, load_instance,
                       validate_instance)

__all__ = [
    "SCHEMA_IDS", "INVARIANT_IDS", "INVARIANTS",
    "PASS", "FAIL", "SKIPPED",
    "load_schema", "load_alphabets", "load_all", "section_vocabulary_pairs",
    "check_schema_self_consistency", "SchemaError",
    "Section", "Geometry", "GeometryError", "SOMATIC_CALIBRE_RATIO",
    "validate_instance", "load_instance", "InstanceReport", "InvariantReport",
]
