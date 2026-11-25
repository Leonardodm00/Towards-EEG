"""trace_sensitivity.py -- the stored sensitivities of a cell at a point
(SPEC section 2.10 (d), Block 26).

No NEURON import: the model enters through an evaluator (Block 23's
build_trace_evaluator, or any callable of the same form).
Contract stubs: signatures and shapes as the Block 26 entry of specs/SPEC.md.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["compute_sensitivities", "save_sensitivities", "load_sensitivities"]


def compute_sensitivities(evaluator: Callable, q, bounds: Sequence[Tuple[float, float]], *,
                          h: float = 1e-3, scheme: str = "central") -> Dict[str, object]:
    """q, h, scheme, traces0 (list of (n_t_b,)), S (list of (n_t_b, p), mV
    per unit of q_i), n_points."""
    raise NotImplementedError("Block 26")


def save_sensitivities(path, sens: Dict[str, object], *, t_list: Sequence[np.ndarray],
                       names: Sequence[str], meta: Optional[dict] = None) -> None:
    """One .npz (no pickle)."""
    raise NotImplementedError("Block 26")


def load_sensitivities(path) -> Dict[str, object]:
    """The dict save_sensitivities wrote (plus t_list, names, meta)."""
    raise NotImplementedError("Block 26")
