#!/usr/bin/env python3
"""
s0_paths.py -- resolve a path recorded before a T6 move to where it lives now.

WHY THIS EXISTS
---------------
Every transform log written before S0.4 is keyed by the path the file had at
the time: `tools/s0_transform/s02_transform_log.json` names
"HybridLFPy Tweaked/Utility_function.py", and after S0.4 that file is at
"towards_eeg/hybrid/utility.py". A harness that opens the logged path either
crashes with FileNotFoundError -- which is loud and therefore harmless -- or,
much worse, quietly filters the file out of its own check and reports PASS
over a smaller set than it claims to cover.

Doc 7 s3 states the general form: an exemption that only skips is an
exemption that hides. The same is true of a lookup that only misses. So this
module exists rather than three copies of `if path in ledger`.

There is exactly ONE implementation of the remapping, in apply_moves.py.
This module wraps it with a root and a cached move list; it deliberately does
not reimplement it, because two implementations of the same rule drift.

USAGE
-----
    from s0_paths import Resolver
    R = Resolver(root)
    R.current("HybridLFPy Tweaked/Utility_function.py")
    # -> "towards_eeg/hybrid/utility.py"
    R.original("towards_eeg/hybrid/utility.py")
    # -> "HybridLFPy Tweaked/Utility_function.py"

Standard library only, ASCII source, Python 3.8+.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from apply_moves import remap, unmap  # noqa: E402


class Resolver(object):
    """Path resolution across the declared T6 moves for one repository root."""

    def __init__(self, root, moves_path=None):
        self.root = root
        path = moves_path or os.path.join(root, "tools", "path_moves.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                self.moves = json.load(fh).get("moves", [])
        else:
            self.moves = []

    def current(self, path):
        """A path as recorded pre-move -> where that file lives now."""
        new_path, _ = remap(path, self.moves)
        return new_path

    def original(self, path):
        """The inverse: a current path -> where it lived before any move."""
        old_path, _ = unmap(path, self.moves)
        return old_path

    def full(self, path):
        """Absolute filesystem path of a pre-move path, after resolution."""
        return os.path.join(self.root, self.current(path))

    def moved(self, path):
        """True if this pre-move path was relocated by a declared move."""
        _, was = remap(path, self.moves)
        return was
