"""Configuration of the diameter pipeline -- Block 1 in specs/SPEC.md.

Every parameter of the pipeline lives here, in frozen dataclasses, with a
``# source:`` comment naming the decision, document or status behind its
default (decision D-021). Nothing in model/ or analysis/ reads a number that
is not passed in from one of these objects.

Provenance tags used in the ``# source:`` comments:
    D-nnn            a decision in TEEG_decisions_and_ideas_log.md (binding)
    D1..D7           a local decision of docs/handoff_diameter_remeasurement.md
    handoff          docs/handoff_diameter_remeasurement.md (design handoff)
    impl-handoff     docs/TEEG_diameter_implementation_handoff_2026-10-06.md
    procedure s.n    docs/TEEG_diameter_bias_table_procedure_2026-10-04.md
    mathematics s.n  docs/TEEG_diameter_bias_table_mathematics_2026-10-04.md
    notes s.n        docs/TEEG_diameter_optics_notes.md
    PROVISIONAL      the assistant's value, confirmed only as a default
    NOT VERIFIED     a fact nobody has checked yet (impl-handoff, Known gaps)

The configuration C of a bias table (procedure s.3.3) is the pair
(estimator settings, simulator settings). ``estimator_signature()`` is the
part a real-node fit must share with the table it is corrected with; the
correction code refuses a table whose estimator signature differs from the
fit's (procedure s.3.11). ``full_signature()`` adds the simulator settings
and is stored next to the table.

Units: lengths in um, angles in degrees in this file (converted to radians by
the code that uses them), grey levels for intensities, mu in 1/um. The depth
axis z is in stage units (procedure s.1, Conventions).

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Tuple

# ----------------------------------------------------------------- choices --
# Named alternatives. The first entry of each tuple is the deliverable
# default; every other entry is a labelled comparison (project-decision-log
# skill: "a default is a decision").
MU_MODES = ("per_node", "shared_branch", "shared_cell")        # D-023 / D-019 (a)
BBAR_REGIONS = ("block_masked", "block", "plane_downsampled")  # D-023 / D-018 (a)
FOCUS_BG_RULES = ("profile_ends_median", "same_as_bbar")       # D-023 / D-018 (c)
ABSORPTIONS = ("partition_vertical", "linear", "ray_world")    # D-024 / procedure s.3.6
KERNEL_FAMILIES = ("gaussian_table", "empirical")              # procedure s.3.4
KERNEL_CONTINUATIONS = ("linear", "frozen", "proportional")    # D-024 / mathematics Eq. 17
RESPONSE_ESTIMATORS = ("tps_spline", "local_linear", "grid_mean")  # D-024
PHANTOM_DESIGNS = ("random", "grid")                           # D-024
END_CUTS = ("axial", "vertical")                               # Block 2 (provisional default, see RendererConfig.end_cut)
FIT_START_RULES = ("profile", "fixed")                         # Block 3 (provisional default, see MeasureConfig.fit_start_rule)
RENDER_BACKENDS = ("fft", "direct")                            # Block 4: fft = impl-handoff FFT form; direct = reference
TABLE_STATISTICS = ("mean", "median")                          # procedure s.3.8
FILL_POLICIES = ("same_branch_then_allen", "allen_only", "none")  # D5 / handoff step 6


@dataclass(frozen=True)
class AcquisitionConfig:
    """Facts about the acquisition, not choices (impl-handoff, Configuration)."""

    specimen_id: int = 529878215      # source: D7, confirmed by D-024 (start with this cell)
    res0_um: float = 0.1144           # source: handoff, Allen API (pixel pitch; not re-queried)
    dz_um: float = 0.28               # source: handoff, NeuronReconstruction.scale_factor_z
    light_direction: int = +1         # source: NOT VERIFIED; +1 = light travels toward increasing plane index (matters only through the partition, dark tubes)
    dendrite_swc_types: Tuple[int, ...] = (3, 4)  # source: handoff step 1 (3 basal, 4 apical)


@dataclass(frozen=True)
class MeasureConfig:
    """Per-node measurement (real path A, shared with synthetic path B).

    Everything here is an ESTIMATOR setting and therefore part of the
    configuration C the bias table is conditional on (procedure s.3.3).
    """

    # -- blocks and planes (handoff step 1; design handoff Next actions 4)
    node_step: str = "every_node"     # source: handoff Next actions 4 (every SWC node, ~1.18 um)
    block_half_um: float = 5.0        # source: handoff step 1 (~10 x 10 um block)
    planes_half: int = 3              # source: handoff step 1 (+-3 planes around the node)
    planes_widen_for_tilt: bool = True  # source: impl-handoff, PROVISIONAL (widen by ceil((L/2) sin(phi) / dz) for steep pieces)
    # -- focus score, handoff Eq. 1-2
    focus_smooth_px: float = 1.0      # source: handoff Eq. 1 (Gaussian weights, s ~ 1 px)
    focus_bg_rule: str = "profile_ends_median"  # source: D-023 (closes D-018 (c))
    focus_bg_ends_um: float = 1.5     # source: handoff Eq. 1 (|v| > 1.5 um)
    subplane_depth: bool = True       # source: handoff Eq. 2 ("optional"), PROVISIONAL on
    focus_plateau_rel: float = 0.02   # source: PROVISIONAL (Block 5): planes with F >= (1 - this) F_max next to k* form the plateau; 3 or more -> its middle (handoff Eq. 2 note)
    # -- path, handoff Eqs. 3-5
    line_fit_window_um: float = 4.0   # source: handoff Next actions 4 (L = 4 um)
    direction_redraws: int = 1        # source: handoff step 3 (raw SWC direction, then one redraw)
    # -- profile, handoff Eq. 9
    profile_half_um: float = 3.0      # source: handoff Eq. 9 (v in [-3, 3] um)
    profile_step_um: float = 0.1144   # source: handoff Eq. 9 (~ one pixel)
    along_branch_avg_um: float = 0.5  # source: handoff Eq. 9 ("optional"), PROVISIONAL on; the table absorbs it
    # -- background, D-018
    bbar_region: str = "block_masked"  # source: D-023 (closes D-018 (a))
    bbar_mask_margin_um: float = 1.0  # source: D-018 (a) recommendation: Allen radius + 1 um around any SWC segment
    bbar_min_unmasked_frac: float = 0.2  # source: PROVISIONAL (Block 5): fewer unmasked block pixels than this fraction flags 'bbar_few'
    # -- fit, D-018.2 / D-019.1
    sigma_fit_um: float = 0.099       # source: D-023, deliverable value, PROVISIONAL (notes s.2 heuristic budget)
    sigma_fit_study_um: Tuple[float, ...] = (0.080, 0.099, 0.125)  # source: D-023 (study axis), PROVISIONAL set = handoff Eq. 12 bracket + budget
    mu_mode: str = "per_node"         # source: D-023 (closes D-019 (a))
    fit_params: Tuple[str, ...] = ("d", "mu", "v0")  # source: D-018 (no fitted B) + D-019 (mu, not alpha)
    fit_d_bounds_um: Tuple[float, float] = (0.05, 6.0)     # source: impl-handoff, PROVISIONAL
    fit_mu_bounds_per_um: Tuple[float, float] = (0.0, 20.0)  # source: impl-handoff, PROVISIONAL
    fit_v0_bounds_um: Tuple[float, float] = (-1.0, 1.0)    # source: impl-handoff, PROVISIONAL
    fit_multistart_factors: Tuple[float, ...] = (0.7, 1.0, 1.4)  # source: impl-handoff, PROVISIONAL (d0 x factors)
    fit_start_rule: str = "profile"   # source: PROVISIONAL (Block 3): d0 = half-depth width of the dip with the blur FWHM removed in quadrature; "fixed" = fit_d0_um
    fit_d0_um: float = 1.0            # source: PROVISIONAL: d0 of the "fixed" rule, and the fallback of "profile" when the half-depth width is undefined
    fit_tol: float = 1e-10            # source: PROVISIONAL: ftol = xtol = gtol of scipy.optimize.least_squares (noise-free recovery 1e-13 observed)
    fit_max_nfev: int = 300           # source: scipy default for method trf (100 x 3 parameters)
    fit_at_bound_rel_tol: float = 1e-6  # source: PROVISIONAL: a parameter within this fraction of its bound range of a bound is reported at_bound
    fit_quad_min_nodes: int = 64      # source: PROVISIONAL; Gauss-Legendre nodes of the model quadrature (Block 3), at least this many
    fit_quad_nodes_per_sigma: float = 6.0  # source: PROVISIONAL; and N >= this x d_hi / sigma_fit (error < 1e-13 B at 3.4 per sigma, Block 3 smoke test)
    # -- selection S and flags (procedure s.3.2; D-024)
    alpha_dark_flag: float = 1.0      # source: D-024 (impl-handoff Findings (4)); threshold to be set on real alpha_hat
    steep_tan_diagnostic: float = 0.58  # source: handoff, old tool STEEP_TAN; DIAGNOSTIC COLUMN ONLY, never a selection rule (D-024)
    phi_vertical_deg: float = 85.0    # source: PROVISIONAL (D-024): above this the heading is taken from the neighbours and the node is flagged 'vertical'
    second_dip_rel: float = 0.5       # source: PROVISIONAL: a second minimum deeper than this fraction of the main dip flags 'crossing'
    second_dip_min_sep_um: float = 1.0  # source: PROVISIONAL (Block 5): ... at least this far from the main minimum
    edge_margin_planes: int = 1       # source: PROVISIONAL: k* within this many planes of the stack ends flags 'stack_edge'
    faint_min_dip_gl: float = 6.0     # source: PROVISIONAL: dip depth (grey levels) below which the node is flagged 'faint'


@dataclass(frozen=True)
class CorrectionConfig:
    """Bias-table estimation, inversion, flags and fill (procedure s.3.8-3.9; D5)."""

    response_estimator: str = "tps_spline"   # source: D-024 (smoothing thin-plate spline for m_hat)
    spline_smoothing: str = "cv"             # source: D-024; "cv" = k-fold cross-validation, or a float
    spline_cv_folds: int = 5                 # source: PROVISIONAL
    spline_smoothing_grid: Tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)  # source: PROVISIONAL (Block 6): RBFInterpolator smoothing values tried by k-fold CV
    tau_kernel_bandwidth: Tuple[float, float] = (0.15, 5.0)  # source: PROVISIONAL (log d, deg) for the local tau_hat and failure rate
    table_statistic: str = "mean"            # source: procedure Eq. 2 (mean); median = labelled comparison
    bias_flag_threshold: float = 0.2         # source: D5, confirmed by D-024 (|b_hat - 1| > 0.2)
    max_failure_rate: float = 0.2            # source: impl-handoff, PROVISIONAL (local failure rate above which the region is flagged)
    inversion_method: str = "brentq"         # source: procedure s.3.9
    fill_policy: str = "same_branch_then_allen"  # source: D5 / handoff step 6, confirmed by D-024
    median_window_nodes: int = 3             # source: handoff Next actions 4 (spine median window)


@dataclass(frozen=True)
class RendererConfig:
    """Synthetic-stack renderer (procedure s.3.6; impl-handoff S1-S6)."""

    kernel_family: str = "gaussian_table"  # source: procedure s.3.4; "empirical" after block 10
    kernel_table_delta_um: Tuple[float, ...] = (0.0, 0.14, 0.28, 0.42, 0.56, 0.84)  # source: mathematics s.3.4 (ideal Debye, 550 nm; NOT calibrated)
    kernel_table_sigma_um: Tuple[float, ...] = (0.080, 0.086, 0.122, 0.262, 0.438, 0.603)  # source: mathematics s.3.4 (ideal Debye, 550 nm; NOT calibrated)
    kernel_continuation: str = "linear"  # source: D-024 ("for now")
    kernel_continuation_slope: float = 0.79  # source: D-024 / mathematics Eq. 17 (Debye cos^4 weighting); 1.21 = uniform disc, 0.72 = proportional
    sigma_r0_um: float = 0.080         # source: procedure s.3.4 (configured; never identified by the stacks)
    absorption: str = "partition_vertical"  # source: D-024 ("for now"); procedure Eqs. 5-6
    h_g_um_thin: float = 0.1144 / 16.0  # source: impl-handoff Findings (grid convergence): p_x/16 for d <= 0.5 um
    h_g_um_thick: float = 0.1144 / 8.0  # source: impl-handoff Findings: p_x/8 above 0.5 um
    h_g_switch_d_um: float = 0.5       # source: impl-handoff Findings
    dzeta_um: float = 0.02             # source: procedure s.3.6 (<= 0.05 um); impl-handoff; convergence test
    U_um: float = 10.0                 # source: D-024 ("up to 10 half length is ok"); the tube is cut by end_cut, never by the block
    end_cut: str = "axial"             # source: PROVISIONAL (D-024 'half length'): "axial" = caps perpendicular to the axis at |s| <= U_um (bounded depth span for every phi < 90 deg); "vertical" = |u| <= U_um (impl-handoff (S5), its Findings; depth span 2 U tan(phi) diverges as phi -> 90 deg)
    cross_section_aspect: float = 1.0  # source: procedure s.3.5 (round); a value k != 1 squashes a round tube along global z by k (mounting shrinkage); checked at bracketing k
    pad_um: float = 2.0                # source: PROVISIONAL: padding of the fine grid beyond the block, >= 3 sigma_r of the farthest slab
    backend: str = "fft"               # source: PROVISIONAL: "fft" (impl-handoff FFT form) or "direct" (reference, slow)
    fft_wrap_sigmas: float = 6.0       # source: PROVISIONAL (Block 4): zero padding >= this x sigma_max / h, so periodic images sit >= 6 sigma away (wrap error < exp(-18))
    direct_truncate: float = 8.0       # source: PROVISIONAL (Block 4): truncate of scipy.ndimage.gaussian_filter in the reference backend (tail mass < 1e-15)
    fft_split_sigma_um: float = 0.5    # source: PROVISIONAL (Block 4): (slab, plane) pairs with a wider kernel go to the far-field path (coarse output grid, exact object); 0.5 was faster than 1.0 on a steep and a flat case
    far_grid_per_sigma: float = 8.0    # source: PROVISIONAL (Block 4): far-field output spacing <= smallest far sigma / this (cubic interpolation error ~ (1/8)^4 / 384)
    # -- camera chain (procedure s.3.6 step 6)
    background_B_gl: float = 210.0     # source: PROVISIONAL (blur_chain_check.py illustrative); NEEDS REAL DATA
    black_level_gl: float = 0.0        # source: NOT VERIFIED (Allen camera chain unknown)
    gain: float = 1.0                  # source: NOT VERIFIED (grey mapping unknown)
    noise_sd_gl: float = 3.0           # source: PROVISIONAL (blur_chain_check.py); matched to the real background SD AFTER the chain
    bit_depth: int = 8                 # source: handoff (8-bit JPEG crops)
    jpeg: bool = True                  # source: procedure s.3.6 step 6
    jpeg_quality: int = 85             # source: PROVISIONAL; replaced by Allen's own tables read from fetched crops (Image.open(f).quantization)
    jpeg_qtables_file: str = ""        # source: Phase II: path of a JSON with Allen's quantization tables; "" = use jpeg_quality


@dataclass(frozen=True)
class PhantomConfig:
    """Phantom design and nuisance draws (procedure s.3.5; D-024)."""

    design: str = "random"             # source: D-024 (random search over ranges, not a grid)
    n_replicates: int = 2000           # source: PROVISIONAL; set by the spline's SE at the davinci budget
    d_range_um: Tuple[float, float] = (0.2, 4.0)  # source: D-024, TEMPORARY until scripts/allen_radius_distribution.py sets d_max
    d_log_uniform: bool = True         # source: PROVISIONAL (b changes fastest at small d)
    phi_range_deg: Tuple[float, float] = (0.0, 90.0)  # source: D-024 ("all the angles without redundancy"); upper end exclusive
    theta_range_rad: Tuple[float, float] = (0.0, math.pi)  # source: procedure s.3.5 / (S3): theta ~ U[0, pi)
    mu_range_per_um: Tuple[float, float] = (0.3, 3.0)  # source: PROVISIONAL (brackets the D-019 test values 0.6, 3.0); matched to real mu_hat later
    mu_log_uniform: bool = True        # source: PROVISIONAL
    subpixel_offset: bool = True       # source: procedure s.3.5 (uniform within one pixel)
    axis_depth_jitter: bool = True     # source: procedure s.3.5 (uniform on [-dz/2, dz/2])
    jitter_xy_um: float = 0.0          # source: D-024 (0 until cell 13 measures s*)
    jitter_z_um: float = 0.0           # source: D-024 (0 until cell 13 measures dz*)
    nodes_each_way: int = 6            # source: PROVISIONAL: phantom "SWC" nodes on each side of the measured one (covers L/2 + margin at node_step)
    phantom_node_step_um: float = 1.18  # source: handoff morphometrics (total_length / n_nodes)
    seed: int = 20261006               # source: PROVISIONAL (every draw from numpy.random.default_rng(seed))
    # grid design (labelled comparison only)
    grid_d_um: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)  # source: procedure s.3.5 (proposal)
    grid_phi_deg: Tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0)  # source: procedure s.3.5 (proposal)


@dataclass(frozen=True)
class CalibrationConfig:
    """Defocus-kernel calibration on thin real nodes (procedure s.3.4, Eq. 4)."""

    node_dhat_max_um: float = 0.3     # source: procedure s.3.4 (thin)
    node_phi_max_deg: float = 10.0    # source: procedure s.3.4 (flat)
    offsets_planes: int = 3           # source: procedure s.3.4 (+-3 planes = +-0.84 um)
    trust_planes: int = 2             # source: procedure s.3.4 (trust beyond +-2 only after a residual check)
    statistic: str = "gaussian_core_width2"  # source: procedure s.3.4 [corrected 2026-10-04]; windowed V as cross-check
    origin_convention: str = "symmetric"     # source: mathematics s.3.5 (identified up to a common shift)
    core_fit_fwhm_factor: float = 1.0        # source: procedure s.3.4 (Gaussian fitted within about one FWHM)


@dataclass(frozen=True)
class DiameterConfig:
    """The whole configuration. Build with ``DiameterConfig()`` for the
    deliverable defaults, or ``dataclasses.replace`` for a labelled variant."""

    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    phantom: PhantomConfig = field(default_factory=PhantomConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # ------------------------------------------------------------ checks --
    def validate(self) -> None:
        """Raise ValueError on a name or range that the code does not accept."""
        m, c, r, p = self.measure, self.correction, self.renderer, self.phantom
        _check_in(m.mu_mode, MU_MODES, "measure.mu_mode")
        _check_in(m.bbar_region, BBAR_REGIONS, "measure.bbar_region")
        _check_in(m.focus_bg_rule, FOCUS_BG_RULES, "measure.focus_bg_rule")
        _check_in(r.absorption, ABSORPTIONS, "renderer.absorption")
        _check_in(r.kernel_family, KERNEL_FAMILIES, "renderer.kernel_family")
        _check_in(r.kernel_continuation, KERNEL_CONTINUATIONS, "renderer.kernel_continuation")
        _check_in(c.response_estimator, RESPONSE_ESTIMATORS, "correction.response_estimator")
        _check_in(c.table_statistic, TABLE_STATISTICS, "correction.table_statistic")
        _check_in(c.fill_policy, FILL_POLICIES, "correction.fill_policy")
        _check_in(p.design, PHANTOM_DESIGNS, "phantom.design")
        _check_in(r.end_cut, END_CUTS, "renderer.end_cut")
        _check_in(r.backend, RENDER_BACKENDS, "renderer.backend")
        if r.kernel_table_sigma_um[0] != r.sigma_r0_um:
            raise ValueError("renderer.sigma_r0_um must equal kernel_table_sigma_um[0] (one number, two names)")
        if any(not (s > 0) for s in r.kernel_table_sigma_um) or not (r.kernel_continuation_slope >= 0):
            raise ValueError("renderer kernel table: sigma must be > 0 and the continuation slope >= 0")
        if not (r.fft_wrap_sigmas > 0 and r.direct_truncate > 0 and r.fft_split_sigma_um > 0 and r.far_grid_per_sigma >= 2):
            raise ValueError("renderer.fft_wrap_sigmas, direct_truncate, fft_split_sigma_um must be > 0 and far_grid_per_sigma >= 2")
        for name, h in (("renderer.h_g_um_thin", r.h_g_um_thin), ("renderer.h_g_um_thick", r.h_g_um_thick)):
            f = self.acquisition.res0_um / h if h > 0 else 0.0
            if not (h > 0) or round(f) < 1 or abs(f - round(f)) > 1e-9 * f:
                raise ValueError("%s must divide res0_um into a whole number of samples" % name)
        if r.bit_depth not in (8, 16) or (r.jpeg and r.bit_depth != 8):
            raise ValueError("renderer.bit_depth must be 8 or 16, and 8 when jpeg is on")
        if not (0 < r.jpeg_quality <= 100) or r.noise_sd_gl < 0 or not (r.gain > 0) or not (r.background_B_gl > 0):
            raise ValueError("renderer camera settings out of range")
        if not (r.cross_section_aspect > 0):
            raise ValueError("renderer.cross_section_aspect must be > 0")
        if m.fit_quad_min_nodes < 8 or not (m.fit_quad_nodes_per_sigma > 0):
            raise ValueError("measure.fit_quad_* out of range")
        _check_in(m.fit_start_rule, FIT_START_RULES, "measure.fit_start_rule")
        if not (0 <= m.focus_plateau_rel < 1 and 0 <= m.bbar_min_unmasked_frac <= 1 and m.second_dip_min_sep_um >= 0
                and 0 < m.second_dip_rel <= 1 and m.planes_half >= 1 and m.direction_redraws >= 0
                and m.line_fit_window_um > 0 and m.profile_step_um > 0 and m.profile_half_um > m.profile_step_um
                and m.along_branch_avg_um >= 0 and m.focus_smooth_px >= 0 and m.block_half_um > 0):
            raise ValueError("measure: per-node chain settings out of range")
        if not (m.fit_d_bounds_um[0] <= m.fit_d0_um <= m.fit_d_bounds_um[1]):
            raise ValueError("measure.fit_d0_um must lie within fit_d_bounds_um")
        if not (0 < m.fit_tol < 1e-3) or m.fit_max_nfev < 1 or not (0 < m.fit_at_bound_rel_tol < 0.5):
            raise ValueError("measure.fit_tol / fit_max_nfev / fit_at_bound_rel_tol out of range")
        if len(m.fit_multistart_factors) < 1 or any(not (f > 0) for f in m.fit_multistart_factors):
            raise ValueError("measure.fit_multistart_factors must be positive and non-empty")
        if m.fit_d_bounds_um[0] <= 0 or m.fit_mu_bounds_per_um[0] < 0:
            raise ValueError("measure fit bounds: d must be > 0 and mu >= 0")
        if m.fit_params != ("d", "mu", "v0"):
            raise ValueError("measure.fit_params must be ('d', 'mu', 'v0'): B is fixed (D-018) "
                             "and the darkness parameter is mu (D-019); got %r" % (m.fit_params,))
        if m.sigma_fit_um not in m.sigma_fit_study_um:
            raise ValueError("measure.sigma_fit_um (%g) must be one of sigma_fit_study_um %r (D-023)"
                             % (m.sigma_fit_um, m.sigma_fit_study_um))
        if len(r.kernel_table_delta_um) != len(r.kernel_table_sigma_um):
            raise ValueError("renderer kernel table: delta and sigma lengths differ")
        if any(b <= a for a, b in zip(r.kernel_table_delta_um[:-1], r.kernel_table_delta_um[1:])):
            raise ValueError("renderer.kernel_table_delta_um must increase strictly")
        if r.kernel_table_delta_um[0] != 0.0:
            raise ValueError("renderer.kernel_table_delta_um must start at 0")
        for name, lo, hi in (("phantom.d_range_um",) + p.d_range_um,
                             ("phantom.phi_range_deg",) + p.phi_range_deg,
                             ("phantom.mu_range_per_um",) + p.mu_range_per_um,
                             ("measure.fit_d_bounds_um",) + m.fit_d_bounds_um,
                             ("measure.fit_mu_bounds_per_um",) + m.fit_mu_bounds_per_um,
                             ("measure.fit_v0_bounds_um",) + m.fit_v0_bounds_um):
            if not (lo < hi):
                raise ValueError("%s: need lo < hi, got (%g, %g)" % (name, lo, hi))
        if not (0.0 <= p.phi_range_deg[0] and p.phi_range_deg[1] <= 90.0):
            raise ValueError("phantom.phi_range_deg must lie within [0, 90]")
        if p.d_range_um[0] <= 0 or p.mu_range_per_um[0] < 0:
            raise ValueError("phantom ranges: d must be > 0 and mu >= 0")
        for name, v in (("renderer.dzeta_um", r.dzeta_um), ("renderer.U_um", r.U_um),
                        ("renderer.h_g_um_thin", r.h_g_um_thin), ("renderer.h_g_um_thick", r.h_g_um_thick),
                        ("measure.sigma_fit_um", m.sigma_fit_um), ("renderer.sigma_r0_um", r.sigma_r0_um),
                        ("acquisition.res0_um", self.acquisition.res0_um), ("acquisition.dz_um", self.acquisition.dz_um)):
            if not (v > 0):
                raise ValueError("%s must be > 0, got %r" % (name, v))
        if self.acquisition.light_direction not in (+1, -1):
            raise ValueError("acquisition.light_direction must be +1 or -1")
        if c.spline_cv_folds < 2 or len(c.spline_smoothing_grid) < 1 or any(not (x >= 0) for x in c.spline_smoothing_grid) \
                or any(not (b > 0) for b in c.tau_kernel_bandwidth) or len(c.tau_kernel_bandwidth) != 2:
            raise ValueError("correction: spline CV folds >= 2, smoothing values >= 0, two positive bandwidths")
        if p.n_replicates < 1 or p.nodes_each_way < 1 or not (p.phantom_node_step_um > 0) or p.jitter_xy_um < 0 or p.jitter_z_um < 0:
            raise ValueError("phantom: n_replicates, nodes_each_way >= 1, node step > 0, jitter >= 0")
        if not (0 < c.bias_flag_threshold) or not (0 <= c.max_failure_rate <= 1):
            raise ValueError("correction thresholds out of range")

    # -------------------------------------------------------- signatures --
    def estimator_signature(self) -> Dict[str, Any]:
        """The estimator part of C (procedure s.3.3, 'shared by real and
        synthetic runs'): a real fit and the table that corrects it must agree
        on every key here."""
        d = {"acquisition": _asdict(self.acquisition), "measure": _asdict(self.measure)}
        d["measure"].pop("sigma_fit_study_um", None)   # the study set is not an estimator setting; sigma_fit_um is
        return d

    def full_signature(self) -> Dict[str, Any]:
        """Estimator + simulator settings; stored with every table."""
        d = self.estimator_signature()
        d["renderer"] = _asdict(self.renderer)
        d["phantom"] = _asdict(self.phantom)
        d["correction"] = _asdict(self.correction)
        return d

    def signature_hash(self, which: str = "estimator") -> str:
        """sha256[:16] of the canonical JSON of a signature; used in file names."""
        sig = self.estimator_signature() if which == "estimator" else self.full_signature()
        return hashlib.sha256(canonical_json(sig).encode("ascii")).hexdigest()[:16]

    def to_json(self) -> str:
        return canonical_json(self.full_signature())


# ----------------------------------------------------------------- helpers --
def _check_in(value: str, allowed: Tuple[str, ...], name: str) -> None:
    if value not in allowed:
        raise ValueError("%s = %r is not one of %r" % (name, value, allowed))


def _asdict(obj: Any) -> Dict[str, Any]:
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, tuples as lists."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_from_dict(d: Dict[str, Any]) -> DiameterConfig:
    """Rebuild a DiameterConfig from ``full_signature()`` (tuples restored)."""
    kwargs = {}
    for f in fields(DiameterConfig):
        sub = d.get(f.name, {})
        cls = f.default_factory  # type: ignore[attr-defined]
        tuple_fields = {g.name for g in fields(cls) if str(g.type).startswith("Tuple")}
        kwargs[f.name] = cls(**{k: (tuple(v) if k in tuple_fields and isinstance(v, list) else v)
                                for k, v in sub.items()})
    cfg = DiameterConfig(**kwargs)
    cfg.validate()
    return cfg


def default_config() -> DiameterConfig:
    cfg = DiameterConfig()
    cfg.validate()
    return cfg


def with_sigma_fit(cfg: DiameterConfig, sigma_fit_um: float) -> DiameterConfig:
    """The study variant of D-023: the same configuration at another in-focus
    blur. The value must be in the study set, so a stray value cannot enter."""
    new = dataclasses.replace(cfg, measure=dataclasses.replace(cfg.measure, sigma_fit_um=float(sigma_fit_um)))
    new.validate()
    return new
