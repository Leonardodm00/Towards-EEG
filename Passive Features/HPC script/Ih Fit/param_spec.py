"""param_spec.py -- the fitted parameter vector of the I_h pipeline.

Stage 3 of TEEG_Ih_fit_staged_plan.md; decisions D-005 (six parameters,
boxes, uniform density, analytic rest balance) and D-007 (confirmations).

Why this exists
---------------
The passive pipeline hard-codes three parameters in roughly fifty places:
`result.x[0..2]`, `params["Cm"]`, `loss_fn(x[0], x[1], x[2])`,
`PARAM_NAMES = ("Cm","Rm","Ra")`. A `ParamSpec` is the single object that
owns "which parameters are being fitted, in which order, in which
coordinates, and how they are applied to a cell", so that the 3-D passive
fit and the 6-D I_h fit are two configurations of one code path rather than
two code paths.

The coordinate convention, unchanged from the passive pipeline
------------------------------------------------------------
The optimiser works in q-space. For a POSITIVE parameter p (every passive
one, plus gbar and kappa_tau) the coordinate is q = log(p) (natural log), so
a uniform prior on q is log-uniform on p -- the correct non-informative prior
for a scale parameter. For a parameter that may be zero or negative (dv_h,
a voltage shift) the coordinate IS the parameter, q = p. Each axis carries
its own flag, so `to_physical` is the only place the distinction lives.

For the three passive axes in that order this reproduces
`PassiveSearchSpace.as_skopt_dimensions()` exactly -- the same Real objects
with the same bounds, names and priors -- which is what makes
`regression_passive_identity.py` a meaningful gate after this refactor.

The loss calling convention
---------------------------
Every loss built by this pipeline is VARIADIC POSITIONAL in spec order:

    loss(*q)        q = (q_1, ..., q_p) in the spec's axis order

For the 3-D passive spec that is `loss(cm_log, rm_log, ra_log)`, i.e. the
legacy signature, unchanged. This is deliberate: it keeps every existing
call site (`_refit_from_bundles`'s `loss_fn(*x)`, the profile sweep's
`loss(lcm, x[0], x[1])`) working while making the arity a property of the
spec rather than of the code.

Notation (units carried explicitly)
-----------------------------------
    Cm   [uF/cm^2]   Rm [Ohm*cm^2]   Ra [Ohm*cm]
    gbar [S/cm^2]    dv_h [mV]       kappa_tau [dimensionless]
    q    [mixed: log of a positive parameter, or the parameter itself]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:                                   # skopt is absent in NEURON-free unit tests
    from skopt.space import Real
except Exception:                      # pragma: no cover - exercised on HPC only
    Real = None

__all__ = [
    "ParamAxis", "ParamSpec", "flatten_result_params",
    "PASSIVE_3D", "make_passive_spec", "make_ih_spec",
    "DEFAULT_GBAR_BOUNDS", "DEFAULT_DVH_BOUNDS", "DEFAULT_KAPPA_BOUNDS",
    "PASSIVE_NAMES", "IH_NAMES",
]

# Passive boxes: unchanged from DEFAULT_*_BOUNDS of the monolith.
DEFAULT_CM_BOUNDS = (0.3, 3.0)             # uF/cm^2
DEFAULT_RM_BOUNDS = (1_000.0, 100_000.0)   # Ohm*cm^2
DEFAULT_RA_BOUNDS = (50.0, 1_000.0)        # Ohm*cm

# I_h boxes (D-005 / D-007: "slightly wider than the human literature").
#   gbar   : anchors 5.14e-5 (Rich 2021 human L5 soma+basal), 1e-4 (Kalmbach
#            2018 human deep L3, uniform; = Eyal's 0.1 mS/cm^2), 2e-4 (Hay rat
#            L5; = Eyal's 0.2). A decade of margin each side. At the floor I_h
#            carries ~0.3 % of the resting conductance, i.e. numerically
#            passive, which is how the passive limit is reached in log space.
#   dv_h   : human V_1/2 is -90.9 mV (Rich) / -90.3 mV (Kalmbach-shifted
#            Kole); +-10 mV spans m_inf(V_rest) from 0.032 to 0.286 on the
#            Rich base.
#   kappa  : human voltage-clamp tau maxima are 400-500 ms and the Rich model
#            peaks at ~343 ms; [0.5, 2] spans peak tau 171-685 ms.
DEFAULT_GBAR_BOUNDS = (1e-6, 1e-3)         # S/cm^2
DEFAULT_DVH_BOUNDS = (-10.0, 10.0)         # mV
DEFAULT_KAPPA_BOUNDS = (0.5, 2.0)          # dimensionless

PASSIVE_NAMES: Tuple[str, ...] = ("Cm", "Rm", "Ra")
IH_NAMES: Tuple[str, ...] = ("gbar", "dv_h", "kappa_tau")


@dataclass(frozen=True)
class ParamAxis:
    """One fitted parameter: its name, its box in PHYSICAL units, and whether
    the optimiser sees log(p) or p."""
    name: str
    lo: float
    hi: float
    log: bool = True
    unit: str = ""

    def __post_init__(self) -> None:
        if not (self.hi > self.lo):
            raise ValueError("axis %r: hi must exceed lo, got (%r, %r)"
                             % (self.name, self.lo, self.hi))
        if self.log and not (self.lo > 0.0):
            raise ValueError("axis %r is log-scaled so lo must be > 0, got %r"
                             % (self.name, self.lo))

    # -- coordinate transforms, both directions -------------------------
    def to_q(self, p: float) -> float:
        return float(np.log(p)) if self.log else float(p)

    def to_physical(self, q: float) -> float:
        return float(np.exp(q)) if self.log else float(q)

    def q_bounds(self) -> Tuple[float, float]:
        return (self.to_q(self.lo), self.to_q(self.hi))


class ParamSpec:
    """An ordered set of ParamAxis plus the rule for applying a parameter
    vector to a NEURON cell.

    `apply_fn(cell, theta, v_rest_mV)` is what makes a spec more than a box:
    it is the only place that knows the ORDER in which the model must be
    updated. For the I_h spec that order is set_passive -> set_ih ->
    balance_e_pas, because the analytic rest balance (plan Eq. 11.3) depends
    on g_pas (hence Rm and F) and on the I_h density and shift.
    """

    def __init__(self, axes: Sequence[ParamAxis],
                 apply_fn: Callable[[Any, Mapping[str, float], float], None],
                 *, label: str = "") -> None:
        self.axes: Tuple[ParamAxis, ...] = tuple(axes)
        if len({a.name for a in self.axes}) != len(self.axes):
            raise ValueError("duplicate axis names in %s" % (self.names,))
        self._apply_fn = apply_fn
        self.label = label or "+".join(self.names)

    # -- basic accessors -------------------------------------------------
    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(a.name for a in self.axes)

    @property
    def n(self) -> int:
        return len(self.axes)

    def __len__(self) -> int:
        return len(self.axes)

    def __repr__(self) -> str:
        return "ParamSpec(%s)" % ", ".join(
            "%s[%g,%g]%s" % (a.name, a.lo, a.hi, "" if a.log else " lin")
            for a in self.axes)

    def axis(self, name: str) -> ParamAxis:
        for a in self.axes:
            if a.name == name:
                return a
        raise KeyError("no axis %r in %s" % (name, self.names))

    def index(self, name: str) -> int:
        return self.names.index(name)

    def has(self, name: str) -> bool:
        return name in self.names

    # -- coordinate transforms ------------------------------------------
    def to_physical(self, q: Sequence[float]) -> Dict[str, float]:
        """q-vector (spec order) -> {name: physical value}."""
        q = list(q)
        if len(q) != self.n:
            raise ValueError("expected %d coordinates for %s, got %d"
                             % (self.n, self.names, len(q)))
        return {a.name: a.to_physical(v) for a, v in zip(self.axes, q)}

    def to_q(self, theta: Mapping[str, float]) -> List[float]:
        """{name: physical value} -> q-vector in spec order."""
        missing = [a.name for a in self.axes if a.name not in theta]
        if missing:
            raise KeyError("theta is missing %s" % missing)
        return [a.to_q(float(theta[a.name])) for a in self.axes]

    def clip_to_box(self, theta: Mapping[str, float]) -> Dict[str, float]:
        """Physical theta clipped into the box (for a starting point that a
        neighbouring fit produced under different bounds)."""
        return {a.name: float(min(max(float(theta[a.name]), a.lo), a.hi))
                for a in self.axes}

    def rail_flags(self, theta: Mapping[str, float], *,
                   rtol: float = 1e-3) -> Dict[str, bool]:
        """Per axis: is the fitted value sitting on a bound? Computed in
        q-space so a log axis is judged on relative, not absolute, distance."""
        out: Dict[str, bool] = {}
        for a in self.axes:
            qlo, qhi = a.q_bounds()
            q = a.to_q(float(theta[a.name]))
            span = max(qhi - qlo, 1e-12)
            out[a.name] = bool((q - qlo) / span < rtol or (qhi - q) / span < rtol)
        return out

    # -- optimiser interface ---------------------------------------------
    def as_skopt_dimensions(self) -> List[Any]:
        """skopt dimensions in q-space, named for use_named_args.

        For the three passive axes this is identical to the monolith's
        PassiveSearchSpace.as_skopt_dimensions(): Real(log(lo), log(hi),
        prior='uniform', name=...) in the order Cm, Rm, Ra.
        """
        if Real is None:                       # pragma: no cover
            raise ImportError("scikit-optimize is required for as_skopt_dimensions()")
        dims = []
        for a in self.axes:
            qlo, qhi = a.q_bounds()
            dims.append(Real(qlo, qhi, prior="uniform", name=a.name))
        return dims

    # -- the model update -------------------------------------------------
    def apply(self, cell: Any, theta: Mapping[str, float],
              v_rest_mV: float) -> None:
        """Apply a PHYSICAL theta to `cell`. See the class docstring on order."""
        self._apply_fn(cell, theta, float(v_rest_mV))

    def apply_q(self, cell: Any, q: Sequence[float], v_rest_mV: float) -> Dict[str, float]:
        theta = self.to_physical(q)
        self.apply(cell, theta, v_rest_mV)
        return theta


# ---------------------------------------------------------------------------
#  The two concrete specs
# ---------------------------------------------------------------------------
def _apply_passive(cell: Any, theta: Mapping[str, float], v_rest_mV: float) -> None:
    """Exactly what the legacy loss did, in the legacy order."""
    cell.set_passive(Cm=float(theta["Cm"]), Rm=float(theta["Rm"]),
                     Ra=float(theta["Ra"]))
    cell.set_e_pas(float(v_rest_mV))


def _apply_ih(cell: Any, theta: Mapping[str, float], v_rest_mV: float) -> None:
    """Passive constants, then the I_h knobs, then the analytic rest balance.

    The balance MUST be last: e_pas(s) depends on g_pas(s) (so on Rm and the
    spine factor F) and on gbar * m_inf(V_rest) (so on gbar and dv_h).
    """
    import ih_mechanism as _ih                 # local: keeps this module importable
    cell.set_passive(Cm=float(theta["Cm"]), Rm=float(theta["Rm"]),
                     Ra=float(theta["Ra"]))
    _ih.set_ih(cell, float(theta["gbar"]),
               float(theta.get("dv_h", 0.0)),
               float(theta.get("kappa_tau", 1.0)))
    _ih.balance_e_pas(cell, float(v_rest_mV))


def make_passive_spec(cm_bounds: Tuple[float, float] = DEFAULT_CM_BOUNDS,
                      rm_bounds: Tuple[float, float] = DEFAULT_RM_BOUNDS,
                      ra_bounds: Tuple[float, float] = DEFAULT_RA_BOUNDS) -> ParamSpec:
    """The 3-D passive spec. Byte-compatible with PassiveSearchSpace."""
    return ParamSpec(
        [ParamAxis("Cm", cm_bounds[0], cm_bounds[1], True, "uF/cm^2"),
         ParamAxis("Rm", rm_bounds[0], rm_bounds[1], True, "Ohm*cm^2"),
         ParamAxis("Ra", ra_bounds[0], ra_bounds[1], True, "Ohm*cm")],
        _apply_passive, label="passive3")


def make_ih_spec(cm_bounds: Tuple[float, float] = DEFAULT_CM_BOUNDS,
                 rm_bounds: Tuple[float, float] = DEFAULT_RM_BOUNDS,
                 ra_bounds: Tuple[float, float] = DEFAULT_RA_BOUNDS,
                 gbar_bounds: Tuple[float, float] = DEFAULT_GBAR_BOUNDS,
                 dvh_bounds: Optional[Tuple[float, float]] = DEFAULT_DVH_BOUNDS,
                 kappa_bounds: Optional[Tuple[float, float]] = DEFAULT_KAPPA_BOUNDS,
                 ) -> ParamSpec:
    """The I_h spec: 4-D, 5-D or 6-D depending on which kinetic knobs are free.

    Pass `dvh_bounds=None` and/or `kappa_bounds=None` to FREEZE that knob at
    its base value -- which is what the Stage-7 recovery gate prescribes for a
    parameter that fails it (D-005: "frozen at its base value for the
    campaign and the failure recorded", never silently carried).
    """
    axes = [ParamAxis("Cm", cm_bounds[0], cm_bounds[1], True, "uF/cm^2"),
            ParamAxis("Rm", rm_bounds[0], rm_bounds[1], True, "Ohm*cm^2"),
            ParamAxis("Ra", ra_bounds[0], ra_bounds[1], True, "Ohm*cm"),
            ParamAxis("gbar", gbar_bounds[0], gbar_bounds[1], True, "S/cm^2")]
    if dvh_bounds is not None:
        axes.append(ParamAxis("dv_h", dvh_bounds[0], dvh_bounds[1], False, "mV"))
    if kappa_bounds is not None:
        axes.append(ParamAxis("kappa_tau", kappa_bounds[0], kappa_bounds[1], True, ""))
    return ParamSpec(axes, _apply_ih, label="ih%d" % len(axes))


#: The default spec of every legacy call site.
PASSIVE_3D: ParamSpec = make_passive_spec()


# ---------------------------------------------------------------------------
#  Serialisation helper
# ---------------------------------------------------------------------------
def flatten_result_params(result: Any, *, prefix: str = "") -> Dict[str, float]:
    """Flat, CSV-safe columns for one fit result's generic parameter vector.

    `results_to_dataframe` writes a FIXED column list, so a six-parameter
    theta would otherwise be dropped on the way to phase2_results.csv -- the
    failure mode of writing a number nobody can read back. This returns

        {"<name>": value, "<name>_sigma": value, ...}  plus "param_names"

    for every axis of the result's own spec, so the orchestrator can do
    `row.update(flatten_result_params(r))` and get exactly the axes that
    were fitted, named as the spec names them. A 3-D result yields Cm, Rm,
    Ra (duplicating the legacy columns, harmlessly and explicitly).

    `rail_<name>` marks an axis sitting on its bound, which for gbar is the
    difference between "the data wanted no I_h" and "the box was too narrow"
    and must never be read off the value alone.
    """
    out: Dict[str, float] = {}
    params = dict(getattr(result, "params", {}) or {})
    sigmas = dict(getattr(result, "sigmas_by_name", {}) or {})
    spec = getattr(result, "param_spec", None)
    names = list(spec.names) if spec is not None else sorted(params)
    for n in names:
        if n in params:
            out[prefix + n] = float(params[n])
        key = "%s_sigma" % n.lower()
        if key in sigmas:
            out[prefix + n + "_sigma"] = float(sigmas[key])
    if spec is not None and params:
        try:
            for n, flag in spec.rail_flags(params).items():
                out[prefix + "rail_" + n] = bool(flag)
        except Exception:                      # pragma: no cover
            pass
    out[prefix + "param_names"] = "|".join(names)
    for k, v in (getattr(result, "ih_summary", {}) or {}).items():
        out[prefix + k] = v
    return out
