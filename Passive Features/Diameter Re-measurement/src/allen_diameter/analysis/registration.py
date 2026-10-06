"""Reading the verdicts of the 2026-09-23 registration check -- Blocks 6, 8
and 11 in specs/SPEC.md.

allen_image_align.registration_check returns its verdict as a sentence whose
first words name the category: "ON THE PROCESS (lateral offset +0.00 um)",
"ALONGSIDE: a ridge ... -- snap needed before measuring", "FAR: nearest
matching ridge ...", "NOT ON A VISIBLE PROCESS (...)", optionally followed by
" [and focus is ... probably a DIFFERENT process]". Everything downstream
(the selection S, the surveys) compares categories, never sentences
[corrected 2026-10-06: the selection compared the sentence with "ON" and so
would have rejected every registered real node].

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

CATEGORIES = ("ON", "ALONGSIDE", "FAR", "NOT_ON")
_PREFIXES = (("ON THE PROCESS", "ON"), ("ALONGSIDE", "ALONGSIDE"), ("FAR", "FAR"), ("NOT ON", "NOT_ON"))


def registration_category(verdict):
    """'' when there is no verdict (phantoms; no registration run), one of
    CATEGORIES for a registration_check sentence or a bare category name, and
    'UNKNOWN' for anything else."""
    v = str(verdict or "").strip().upper()
    if not v:
        return ""
    for prefix, cat in _PREFIXES:
        if v.startswith(prefix):
            return cat
    return v if v in CATEGORIES else "UNKNOWN"


def different_process(verdict):
    """True when the sentence carries the module's 'probably a DIFFERENT process' warning."""
    return "DIFFERENT PROCESS" in str(verdict or "").upper()
