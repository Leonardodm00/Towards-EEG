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
FOCUS_RULES = ("gradient_energy", "dip_depth")                 # D-030: gradient energy; dip depth = handoff Eq. 1, comparison
FOCUS_GRAD_WINDOWS = ("radius_margin", "whole_profile")        # D-030 (provisional default, see MeasureConfig.focus_grad_window)
ABSORPTIONS = ("partition_vertical", "linear", "ray_world")    # D-024 / procedure s.3.6
KERNEL_FAMILIES = ("gaussian_table", "empirical")              # procedure s.3.4
KERNEL_CONTINUATIONS = ("linear", "frozen", "proportional")    # D-024 / mathematics Eq. 17
RESPONSE_ESTIMATORS = ("local_linear", "tps_spline", "grid_mean")  # D-024, D-063.2, D-067.3 (grid_mean: not built)
PHANTOM_DESIGNS = ("random", "grid")                           # D-024
END_CUTS = ("axial", "vertical")                               # Block 2 (provisional default, see RendererConfig.end_cut)
TABLE_PHI_AXES = ("true", "measured")                          # Block 7 (procedure Eq. 2 / mathematics s.3.7)
FIT_START_RULES = ("profile", "fixed")                         # Block 3 (provisional default, see MeasureConfig.fit_start_rule)
RENDER_BACKENDS = ("fft", "direct")                            # Block 4: fft = impl-handoff FFT form; direct = reference
TABLE_STATISTICS = ("mean", "median")                          # procedure s.3.8
FILL_POLICIES = ("same_branch_then_allen", "allen_only", "none")  # D5 / handoff step 6
CALIBRATION_STATISTICS = ("gaussian_core_width2", "windowed_variance")  # procedure s.3.4 [corrected 2026-10-04]: core width drives
ORIGIN_CONVENTIONS = ("symmetric", "min_origin")               # mathematics s.3.5: the growth is identified up to a common shift
GROWTH_INTERPS = ("cubic", "linear")                           # Block 10 (provisional default, see CalibrationConfig.growth_interp)


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
    # -- focus score, D-030 (handoff Eq. 1 = the "dip_depth" comparison) and handoff Eq. 2
    focus_rule: str = "gradient_energy"  # source: D-030 (user, 2026-10-07): gradient energy of the profile replaces the dip depth of handoff Eq. 1
    focus_grad_window: str = "radius_margin"  # source: PROVISIONAL (D-030, assistant): integrate over |v| <= node radius + focus_grad_margin_um; "whole_profile" = every sample
    focus_grad_margin_um: float = 0.5  # source: PROVISIONAL (D-030): margin beyond the node radius (Allen's on real nodes, the true one on phantoms)
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
    sigma_fit_um: float = 0.099       # source: D-023, deliverable value, PROVISIONAL (notes s.2 heuristic budget); the one value, D-057 (ii) dropped the study
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
    # -- the fit at a plane, D-052.2 (Block 12): skewed generalized-t kernel, w and (beta, xi, zeta) fitted, IRLS
    sgt_w_bounds_um: Tuple[float, float] = (0.1, 2.0)  # source: D-052.2 (the kernel's FWHM w, um; = fit_variants.FITTED_FWHM_BOUNDS_UM of D-046)
    sgt_beta_bounds: Tuple[float, float] = (1.0, 10.0)  # source: D-052.2 (searched in log10 beta)
    sgt_xi_bounds: Tuple[float, float] = (0.0, 2.0)  # source: D-052.2 (xi = 0 is the generalized-normal edge, a member)
    sgt_ln_zeta_bounds: Tuple[float, float] = (-0.7, 0.7)  # source: D-052.2 (zeta in [e^-0.7, e^0.7] = [0.497, 2.014], searched in ln zeta)
    sgt_lambda_um: float = 0.5        # source: D-052 (user, 2026-10-16): centre weights exp(-|v - v0| / lambda), re-centred by IRLS
    sgt_irls_tol_um: float = 1e-4     # source: D-052.2 / D-046: the IRLS stops when d and v0 both move by at most this (um)
    sgt_irls_max_iter: int = 30       # source: D-052.2 / D-046, PROVISIONAL (D-052 (h)): at most this many weighted solves
    sgt_max_nfev: int = 3000          # source: D-054 (iv) (user, 2026-10-18): max_nfev of every least_squares solve of the SGT fit
    sgt_start_shapes: Tuple[Tuple[float, float, float], ...] = ((2.0, 0.0, 1.0), (1.0, 0.0, 1.0), (5.0, 0.0, 1.0), (2.0, 1.0, 1.0))  # source: D-052.2, PROVISIONAL (the sandbox check's): (beta, xi, zeta) started from the Gaussian of fitted width
    sgt_start_w_um: float = 0.7       # source: D-052.2, PROVISIONAL: the width of the start from Block 5's (d, mu, v0)
    sgt_base_start_shape: Tuple[float, float, float] = (2.0, 0.0, 1.0)  # source: D-052.2, PROVISIONAL: the shape of that start (Gaussian)
    # -- the Allen-guided plane iteration, the deliverable's plane (D-042, D-044, D-052.3; Block 13)
    iter_line_mult: float = 3.0       # source: D-044 (user, 2026-10-12): round i's line is +-iter_line_mult d_(i-1) / 2 (+-1.5 d)
    iter_max_rounds: int = 5          # source: D-042 (user, 2026-10-10), PROVISIONAL reading (D-052 (h)): at most this many rounds
    iter_planes_half: int = 6         # source: D-052.3 / D-042: the planes K = k_SWC +- this
    iter_bg_margin_um: float = 4.0    # source: D-036, PROVISIONAL (D-052 (h)): B_k from pixels farther than Allen's radius + this from every traced segment and the soma
    iter_min_first_samples: int = 3   # source: D-069 (user, 2026-10-21): round 1's line holds at least this many samples (its half-length at least (n - 1) / 2 profile steps)
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

    response_estimator: str = "local_linear"  # source: D-063 (D-063.2), D-067.3: local linear kernel regression on a grid [was "tps_spline", D-024]
    spline_smoothing: str = "cv"             # source: D-024; "cv" = k-fold cross-validation, or a float (tps_spline only)
    spline_cv_folds: int = 5                 # source: PROVISIONAL; the folds are over the draws, draw g in fold (g - 1) mod K (D-062.3)
    spline_smoothing_grid: Tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)  # source: PROVISIONAL (Block 6), extended by 1e3, 1e4 (D-062.4): RBFInterpolator smoothing values tried by k-fold CV (tps_spline only)
    tau_kernel_bandwidth: Tuple[float, float, float] = (0.15, 5.0, 0.3)  # source: D-054 (ii), D-063.1, PROVISIONAL: (h1, h2, h3) in ln d, deg, ln(mu_hat d_hat), the kernel of the estimator and of its statistics
    bandwidth_cv_grid: Tuple[float, ...] = (0.5, 0.7071, 1.0, 1.4142, 2.0)  # source: D-071 (D-071.1): the multiplier c of tau_kernel_bandwidth, chosen by k-fold CV with folds by draw; values PROVISIONAL; one value fixes c
    min_neff: float = 10.0                   # source: PROVISIONAL (Block 14.4): the support floor, N_eff of the rows in S below which a grid point holds no estimate
    loo_mass_percentile: float = 5.0         # source: D-063 (D-063.4), PROVISIONAL: off_support below this percentile of the rows' leave-one-out kernel masses
    tps_max_rows: int = 3000                 # source: D-063.2, D-067.3: the dense tps_spline refuses more rows in S (value PROVISIONAL)
    table_statistic: str = "mean"            # source: procedure Eq. 2 (mean); median = labelled comparison
    bias_flag_threshold: float = 0.2         # source: D5, confirmed by D-024 (|b_hat - 1| > 0.2)
    max_failure_rate: float = 0.2            # source: impl-handoff, PROVISIONAL (local failure rate above which the region is flagged)
    failure_rate_ignore: Tuple[str, ...] = ()  # source: D-057 (D-057.1): every row enters the failure count; a screen named here still removes the rows it rejected [was ("dark",), PROVISIONAL, gate 2 2026-10-06]
    inversion_method: str = "brentq"         # source: procedure s.3.9
    inversion_grid_points: int = 256         # source: PROVISIONAL (Block 7): log-spaced diameters on which m_hat - d_hat is bracketed
    table_phi_axis: str = "true"             # source: procedure Eq. 2 (true tilt); "measured" = index phantoms by their line-fit tilt (mathematics s.3.7), labelled comparison
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
    h_g_um_thin: float = 0.1144 / 8.0   # source: PROVISIONAL (Block 4, 2026-10-06): d_hat after pixel integration agrees to 1e-5 at p_x/8, /16, /32 (d 0.3-0.5 um, phi 5-20 deg) and to 0.7 % at d 0.3 um, phi 45 deg (b = 1.87, flagged); impl-handoff Findings proposed p_x/16 from fine-grid profiles [corrected 2026-10-06]
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
    max_fft_accumulator_mb: float = 256.0  # source: PROVISIONAL (Block 4): spectral accumulators per plane chunk; a 2-worker run was OOM-killed at 3.5 GB per worker without it
    ray_na: float = 1.4                # source: impl-handoff (S6): condenser NA (evenly filled, sine condition)
    ray_n_oil: float = 1.515           # source: impl-handoff (S6): immersion index (index-matched)
    ray_n_rho: int = 16                # source: PROVISIONAL (Block 9): Gauss-Legendre nodes in rho^2 (checks/optics_points_check.py used 48)
    ray_n_psi: int = 32                # source: PROVISIONAL (Block 9): azimuths (the check script used 96)
    ray_post_sigma_um: float = 0.08    # source: checks/optics_points_check.py (post blur standing in for diffraction near focus)
    # -- camera chain (procedure s.3.6 step 6)
    background_B_gl: float = 210.0     # source: PROVISIONAL (blur_chain_check.py illustrative); NEEDS REAL DATA
    black_level_gl: float = 0.0        # source: NOT VERIFIED (Allen camera chain unknown)
    gain: float = 1.0                  # source: NOT VERIFIED (grey mapping unknown)
    noise_sd_gl: float = 3.0           # source: PROVISIONAL (blur_chain_check.py); matched to the real background SD AFTER the chain
    bit_depth: int = 8                 # source: handoff (8-bit JPEG crops)
    jpeg: bool = True                  # source: procedure s.3.6 step 6
    jpeg_quality: int = 85             # source: PROVISIONAL; replaced by Allen's own tables read from fetched crops (Image.open(f).quantization)
    jpeg_qtables: Tuple[Tuple[int, ...], ...] = ()  # source: Phase II (scripts/camera_calibration.py): Allen's JPEG quantization tables, 64 integers each in natural order (Pillow >= 8.3); () = use jpeg_quality [corrected 2026-10-07: was jpeg_qtables_file, a path nothing read]


@dataclass(frozen=True)
class PhantomConfig:
    """Phantom design and nuisance draws (procedure s.3.5; D-024)."""

    design: str = "random"             # source: D-024 (random search over ranges, not a grid)
    n_draws: int = 100000              # source: D-062 (D-062.2): G, "100,000 or more"; a record of the production's size (the run's draw range sets what is rendered) [replaces n_replicates]
    n_realizations: int = 10           # source: D-062 (D-062.2): n_real realizations of every draw (D-059.2)
    radius_noise_sd_um: float = 0.2    # source: D-062 (D-062.1): the branch radius d/2 + eps, eps ~ N(0, sd^2) redrawn until positive, per realization (D-058)
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

    node_allen_diameter_max_um: float = 0.6  # source: D-067 (D-067.4, D-061.4), PROVISIONAL: the near-field (thin) set, Allen's diameter 2 r_A <= this (um), scanned up to offsets_planes
    far_field_allen_diameter_um: Tuple[float, float] = (0.6, 0.8)  # source: D-067 (D-067.4, D-026), PROVISIONAL: the far-field set, lo < 2 r_A <= hi (um), for the offsets beyond offsets_planes (its scan is not built)
    node_phi_max_deg: float = 5.0     # source: D-081.3 [user, 2026-10-23]: flat, both sets, <= 5 deg (was 10, procedure s.3.4); exposed
    offsets_planes: int = 3           # source: procedure s.3.4 (+-3 planes = +-0.84 um)
    trust_planes: int = 2             # source: procedure s.3.4 (trust beyond +-2 only after a residual check)
    statistic: str = "gaussian_core_width2"  # source: procedure s.3.4 [corrected 2026-10-04]; windowed V as cross-check
    origin_convention: str = "symmetric"     # source: mathematics s.3.5 (identified up to a common shift)
    core_fit_fwhm_factor: float = 1.0        # source: procedure s.3.4 (Gaussian fitted within about one FWHM): samples with |v - v_peak| <= this x FWHM
    node_alpha_max: float = 0.5              # source: PROVISIONAL (Block 10): "faint" calibration node, alpha_hat <= this (the dark flag is 1.0)
    window_half_um: float = 1.5              # source: PROVISIONAL (Block 10): half-window of V_W and A_W about the core centre (the focus score's ends rule)
    min_window_samples: int = 5              # source: PROVISIONAL (Block 10): fewest samples in the core-fit window (3 parameters)
    knot_step_um: float = 0.28               # source: procedure s.3.4 ("tabulated per plane offset"); Block 10: knots finer than dz are not identified (a zig-zag of period 2 x step is absorbed by the c_i)
    z_ax_bound_um: float = 0.28              # source: PROVISIONAL (Block 10): the fitted axis depth stays within this of the node's sub-plane depth
    min_planes_per_node: int = 3             # source: PROVISIONAL (Block 10): a node enters the growth fit with at least this many finite omega
    growth_interp: str = "cubic"             # source: PROVISIONAL (Block 10): between the plane-offset knots; linear made G(1 plane) < 0 on rendered thin phantoms (convex curve)
    # 2026-10-23, D-078 to D-081 (Block 10.P, 10.I, 10.A): the patch, its planar background, isolation, straightness,
    # the anchor's SE and the table cut -- both calibration sets. None of these is in a signature.
    patch_half_um: float = 25.0              # source: D-078.1 [user]: the +-25 um square about the pilot's centre, fetched over the stage's planes; the background is fitted on its far pixels
    far_pixel_sigmas: float = 3.0            # source: D-078.1 [user]: a far pixel lies beyond r_q + this x sigma_r(delta_q,k) of every traced segment q and of the soma
    background_min_far_frac: float = 0.05    # source: PROVISIONAL (Block 10.P): fewer far pixels than this fraction of the patch -> plane status bg_few
    background_trim_sigmas: float = 3.0      # source: PROVISIONAL (Block 10.P): the plane is refitted once without the pixels beyond this x 1.4826 MAD of the first fit's residuals (untraced debris)
    isolation_depth_gl: float = 0.03         # source: D-081.1 [user]: a traced neighbour (not the node's own branch) whose predicted shadow reaches the patch deeper than this (gl) in a scanned plane fails the node (not_isolated)
    node_straight_half_um: float = 10.0      # source: D-081.3 (proposal accepted), PROVISIONAL: the trace within +-this (um) along the branch must stay near the node's straight line
    node_straight_tol_radii: float = 1.0     # source: D-081.3, PROVISIONAL: ... within this x (Allen's diameter / 2) of the line, else not_straight
    bootstrap_resamples: int = 1000          # source: D-081.7 [user]: node resamples for the SE of the anchor (10.A) and of the far slope (15.5)
    table_max_delta_um: float = 0.84         # source: D-080.1 [user]: every table written for rendering ends here, so the renderer's linear continuation starts at the anchor
    # Block 15, the far-field calibration (D-074, D-077, D-078, D-081)
    far_offsets_planes: int = 16             # source: D-074.1 [user]: the far scan reads k* +- this (4.48 um on both sides of focus)
    far_profile_half_um: float = 10.0        # source: D-074.2 [user]: half-length of the far line (um); a calibration setting, not MeasureConfig's
    far_min_snr: float = 3.0                 # source: D-074.1 [user]: a plane enters the far fit when its PREDICTED dip A_j / (sqrt(2 pi) sigma_p) >= this x RMS_j (D-078.3)
    far_min_k_offset: int = 5                # source: PROVISIONAL (D-026 (g)): planes with |k - k*| >= this enter the far fit (x >= 0.4 um >= r_j)
    far_area_offsets: Tuple[int, int] = (4, 6)  # source: PROVISIONAL (S2b checks): the node's dip area A_j is the median of a s sqrt(2 pi) over the ok planes with |k - k*| in this range
    far_min_planes_per_node: int = 6         # source: PROVISIONAL: a far node enters the fit with at least this many selected planes
    far_core_max_passes: int = 10            # source: D-078.2 / D-081.4 [user]: N, the cap on the iterative core fit's passes (exposed)
    far_core_stop_tol: float = 1.0           # source: D-078.2 [user]: kappa, the stop |ds| <= kappa SE(s) between passes
    far_core_width_bounds: Tuple[float, float] = (0.25, 4.0)  # source: PROVISIONAL (S2b checks): the core SD bounded to this x sigma_p; a fit at a bound ends the plane
    far_line_end_fwhm_factor: float = 2.0    # source: PROVISIONAL (D-078.1's check): the line-end zone |v - v_c| >= this x FWHM
    far_line_end_min_samples: int = 4        # source: PROVISIONAL: samples per side the line-end zone needs
    far_irls_rounds: int = 3                 # source: D-078.3 [user; the S2b checks' code]: refits with the model weights frozen at the previous round
    far_gamma_max: float = 5.0               # source: PROVISIONAL: upper bound of gamma in the far fit; a fit at it fails
    anchor_sensitivity_step_um: float = 0.01  # source: D-081.7 [user]: half-step of the central difference d gamma_hat / d sigma_0


@dataclass(frozen=True)
class SurveyConfig:
    """S3 across cells (Block 16; D-065, D-085): the cell pool, the acquisition
    check and the stratified selection (16.1); each cell's global alignment
    (16.2). None of these is in a signature."""

    species: str = "Homo Sapiens"            # source: D-085.1; the Allen SDK's CellTypesApi.HUMAN, the pool of the passive-fit pipeline
    cells_per_group: int = 5                 # source: D-085.1 [user]: 5 cells per neuron class and layer, all of a smaller group
    class_map: Tuple[Tuple[str, str], ...] = (("spiny", "exc"), ("aspiny", "inh"), ("sparsely spiny", "sparsely_spiny"))  # source: D-085.1 (spiny exc, aspiny inh: PROVISIONAL reading), D-087 [user]: sparsely spiny a third class (its name the assistant's); an unmapped or missing type is left out
    selection_seed: int = 20261026           # source: PROVISIONAL: seed of the per-group permutations (Block 16.1 (d))
    acquisition_rel_tol: float = 1e-3        # source: PROVISIONAL: relative tolerance of a cell's plane spacing and pixel size against acquisition (D-085.3)
    pool_page_rows: int = 10000              # source: PROVISIONAL: rows per page of the ApiCellTypesSpecimenDetail query; large, so the model comes in one page (the SDK asks num_rows 'all'; pages without an order may skip rows)
    # Block 16.2, each cell's global alignment (D-085.2): coarse on the minimum-intensity projection, then fine
    align_downsample: int = 4                # source: D-085.2 [user]: the projection's downsample level for the coarse alignment (1.83 um/px), as cell 11 did for 529878215
    align_max_shift_um: float = 250.0        # source: PROVISIONAL: half-window of the coarse search (allen_image_align.align_translation's default)
    align_null_n: int = 50                   # source: PROVISIONAL: block-shuffled darkness maps in the null of the score ratio
    align_null_block_um: float = 30.0        # source: PROVISIONAL: side of the shuffled tiles (um; 16 px at level 4)
    align_min_score_ratio: float = 2.0       # source: PROVISIONAL: a cell whose coarse score is below this multiple of the null's median is set aside (D-085.2)
    align_fine_downsample: int = 1           # source: PROVISIONAL (the assistant's fine stage): the finest level tried for the fine crop (0.23 um/px)
    align_fine_margin_um: float = 20.0       # source: PROVISIONAL: margin of the fine crop around the skeleton's bounding box (um)
    align_fine_max_shift_um: float = 5.0     # source: PROVISIONAL: half-window of the fine direct search (um)
    align_fine_ridge_um: float = 6.0         # source: PROVISIONAL: widest dark feature kept by the fine stage's ridge darkness (um; the soma is removed)
    align_z0_um: float = 0.0                 # source: PROVISIONAL: depth of plane index 0 (um), as 529878215 (plane index = z / dz); checked by S3b's registration offsets
    projection_size_rel_tol: float = 0.02    # source: PROVISIONAL: an xy projection's width and height lie within this of the planes' median


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
    survey: SurveyConfig = field(default_factory=SurveyConfig)

    # ------------------------------------------------------------ checks --
    def validate(self) -> None:
        """Raise ValueError on a name or range that the code does not accept."""
        m, c, r, p = self.measure, self.correction, self.renderer, self.phantom
        _check_in(m.mu_mode, MU_MODES, "measure.mu_mode")
        _check_in(m.bbar_region, BBAR_REGIONS, "measure.bbar_region")
        _check_in(m.focus_bg_rule, FOCUS_BG_RULES, "measure.focus_bg_rule")
        _check_in(m.focus_rule, FOCUS_RULES, "measure.focus_rule")
        _check_in(m.focus_grad_window, FOCUS_GRAD_WINDOWS, "measure.focus_grad_window")
        if not (m.focus_grad_margin_um >= m.profile_step_um):
            raise ValueError("measure.focus_grad_margin_um must be >= profile_step_um, so that the window holds "
                             "at least three samples")
        _check_in(r.absorption, ABSORPTIONS, "renderer.absorption")
        _check_in(r.kernel_family, KERNEL_FAMILIES, "renderer.kernel_family")
        _check_in(r.kernel_continuation, KERNEL_CONTINUATIONS, "renderer.kernel_continuation")
        _check_in(c.response_estimator, RESPONSE_ESTIMATORS, "correction.response_estimator")
        _check_in(c.table_statistic, TABLE_STATISTICS, "correction.table_statistic")
        _check_in(c.fill_policy, FILL_POLICIES, "correction.fill_policy")
        _check_in(c.table_phi_axis, TABLE_PHI_AXES, "correction.table_phi_axis")
        if c.inversion_method != "brentq" or c.inversion_grid_points < 8 or c.median_window_nodes < 1:
            raise ValueError("correction: inversion_method brentq, inversion_grid_points >= 8, median_window_nodes >= 1")
        _check_in(p.design, PHANTOM_DESIGNS, "phantom.design")
        _check_in(r.end_cut, END_CUTS, "renderer.end_cut")
        _check_in(r.backend, RENDER_BACKENDS, "renderer.backend")
        if r.kernel_table_sigma_um[0] != r.sigma_r0_um:
            raise ValueError("renderer.sigma_r0_um must equal kernel_table_sigma_um[0] (one number, two names)")
        if any(not (s > 0) for s in r.kernel_table_sigma_um) or not (r.kernel_continuation_slope >= 0):
            raise ValueError("renderer kernel table: sigma must be > 0 and the continuation slope >= 0")
        if not (r.fft_wrap_sigmas > 0 and r.direct_truncate > 0 and r.fft_split_sigma_um > 0 and r.far_grid_per_sigma >= 2
                and r.max_fft_accumulator_mb > 0 and 0 < r.ray_na < r.ray_n_oil and r.ray_n_rho >= 1
                and r.ray_n_psi >= 1 and r.ray_post_sigma_um >= 0):
            raise ValueError("renderer: fft_wrap_sigmas, direct_truncate, fft_split_sigma_um, max_fft_accumulator_mb must be > 0, "
                             "far_grid_per_sigma >= 2, 0 < ray_na < ray_n_oil, ray_n_rho and ray_n_psi >= 1, ray_post_sigma_um >= 0")
        for name, h in (("renderer.h_g_um_thin", r.h_g_um_thin), ("renderer.h_g_um_thick", r.h_g_um_thick)):
            f = self.acquisition.res0_um / h if h > 0 else 0.0
            if not (h > 0) or round(f) < 1 or abs(f - round(f)) > 1e-9 * f:
                raise ValueError("%s must divide res0_um into a whole number of samples" % name)
        if r.bit_depth not in (8, 16) or (r.jpeg and r.bit_depth != 8):
            raise ValueError("renderer.bit_depth must be 8 or 16, and 8 when jpeg is on")
        if not (0 < r.jpeg_quality <= 100) or r.noise_sd_gl < 0 or not (r.gain > 0) or not (r.background_B_gl > 0):
            raise ValueError("renderer camera settings out of range")
        _check_qtables(r.jpeg_qtables)
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
        _check_sgt(m)
        _check_iteration(m)
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
        cal = self.calibration
        _check_in(cal.statistic, CALIBRATION_STATISTICS, "calibration.statistic")
        _check_in(cal.origin_convention, ORIGIN_CONVENTIONS, "calibration.origin_convention")
        _check_in(cal.growth_interp, GROWTH_INTERPS, "calibration.growth_interp")
        thin, band = cal.node_allen_diameter_max_um, tuple(cal.far_field_allen_diameter_um)
        if not (len(band) == 2 and all(math.isfinite(x) for x in (thin,) + band) and 0 < thin <= band[0] < band[1]):
            raise ValueError("calibration: 0 < node_allen_diameter_max_um <= far_field_allen_diameter_um[0] < "
                             "far_field_allen_diameter_um[1], all finite (D-067.4: two disjoint sets of Allen's "
                             "diameter)")
        if not (0 <= cal.node_phi_max_deg <= 90 and cal.offsets_planes >= 1
                and 0 <= cal.trust_planes <= cal.offsets_planes and cal.core_fit_fwhm_factor > 0
                and cal.node_alpha_max > 0 and cal.window_half_um > 0 and cal.min_window_samples >= 4
                and cal.knot_step_um >= self.acquisition.dz_um - 1e-12 and cal.z_ax_bound_um > 0
                and cal.min_planes_per_node >= 3):
            raise ValueError("calibration settings out of range (knot_step_um must be >= dz_um: finer knots "
                             "are not identified by plane scans)")
        wb, ao = tuple(cal.far_core_width_bounds), tuple(cal.far_area_offsets)
        if not (cal.patch_half_um > 0 and cal.far_pixel_sigmas >= 0 and 0 < cal.background_min_far_frac <= 1
                and cal.background_trim_sigmas > 0 and cal.isolation_depth_gl >= 0 and cal.node_straight_half_um > 0
                and cal.node_straight_tol_radii > 0 and cal.bootstrap_resamples >= 2 and cal.table_max_delta_um > 0
                and cal.far_offsets_planes >= cal.offsets_planes and cal.far_profile_half_um > 0
                and cal.far_min_snr >= 0 and 1 <= cal.far_min_k_offset <= cal.far_offsets_planes
                and len(ao) == 2 and 1 <= ao[0] <= ao[1] <= cal.far_offsets_planes and cal.far_min_planes_per_node >= 2
                and cal.far_core_max_passes >= 1 and cal.far_core_stop_tol >= 0
                and len(wb) == 2 and 0 < wb[0] < 1 < wb[1] and cal.far_line_end_fwhm_factor > 0
                and cal.far_line_end_min_samples >= 1 and cal.far_irls_rounds >= 0 and cal.far_gamma_max > 0
                and cal.anchor_sensitivity_step_um > 0):
            raise ValueError("calibration: patch, background, isolation, straightness, bootstrap, table cut or far-field "
                             "settings out of range (Block 10.P/10.I/10.A, Block 15; D-078 to D-081)")
        if abs(cal.table_max_delta_um - round(cal.table_max_delta_um / cal.knot_step_um) * cal.knot_step_um) > 1e-9:
            raise ValueError("calibration.table_max_delta_um must be a multiple of knot_step_um (the table ends on a knot)")
        sv = self.survey
        types = [str(t).strip().lower() for t, _ in sv.class_map] if all(len(x) == 2 for x in sv.class_map) else None
        if not (isinstance(sv.species, str) and sv.species.strip() and sv.cells_per_group >= 1
                and types is not None and all(types) and len(set(types)) == len(types)
                and all(isinstance(c, str) and c.strip() for _, c in sv.class_map)
                and sv.acquisition_rel_tol >= 0 and math.isfinite(sv.acquisition_rel_tol) and sv.pool_page_rows >= 1):
            raise ValueError("survey: species, cells_per_group >= 1, class_map (unique dendrite types -> non-empty "
                             "classes), acquisition_rel_tol >= 0 or pool_page_rows >= 1 out of range (Block 16.1, D-085)")
        if not (0 <= sv.align_fine_downsample < sv.align_downsample <= 6 and sv.align_max_shift_um > 0
                and sv.align_null_n >= 10 and sv.align_null_block_um > 0
                and sv.align_min_score_ratio > 0 and sv.align_fine_margin_um >= 0 and sv.align_fine_max_shift_um > 0
                and sv.align_fine_ridge_um > 0 and math.isfinite(sv.align_z0_um)
                and 0 < sv.projection_size_rel_tol < 1):
            raise ValueError("survey: 0 <= align_fine_downsample < align_downsample <= 6, align_max_shift_um > 0, "
                             "align_null_n >= 10, align_null_block_um > 0, align_min_score_ratio > 0, fine "
                             "margin >= 0, fine shift and ridge > 0, align_z0_um finite, 0 < projection_size_rel_tol "
                             "< 1 (Block 16.2, D-085.2)")
        for name, v in (("renderer.dzeta_um", r.dzeta_um), ("renderer.U_um", r.U_um),
                        ("renderer.h_g_um_thin", r.h_g_um_thin), ("renderer.h_g_um_thick", r.h_g_um_thick),
                        ("measure.sigma_fit_um", m.sigma_fit_um), ("renderer.sigma_r0_um", r.sigma_r0_um),
                        ("acquisition.res0_um", self.acquisition.res0_um), ("acquisition.dz_um", self.acquisition.dz_um)):
            if not (v > 0):
                raise ValueError("%s must be > 0, got %r" % (name, v))
        if self.acquisition.light_direction not in (+1, -1):
            raise ValueError("acquisition.light_direction must be +1 or -1")
        if c.spline_cv_folds < 2 or len(c.spline_smoothing_grid) < 1 or any(not (x >= 0) for x in c.spline_smoothing_grid):
            raise ValueError("correction: spline CV folds >= 2 and smoothing values >= 0")
        if len(c.tau_kernel_bandwidth) != 3 or not all(math.isfinite(b) and b > 0 for b in c.tau_kernel_bandwidth):
            raise ValueError("correction.tau_kernel_bandwidth needs three positive finite values (ln d, deg, "
                             "ln(mu_hat d_hat); D-063.1), got %r: a configuration written before Block 14.3 has two "
                             "-- make it again" % (c.tau_kernel_bandwidth,))
        if len(c.bandwidth_cv_grid) < 1 or not all(math.isfinite(x) and x > 0 for x in c.bandwidth_cv_grid):
            raise ValueError("correction.bandwidth_cv_grid needs at least one positive finite multiplier")
        if not (math.isfinite(c.min_neff) and c.min_neff >= 2) or not (0 <= c.loo_mass_percentile <= 100) \
                or not (isinstance(c.tps_max_rows, int) and not isinstance(c.tps_max_rows, bool) and c.tps_max_rows >= 1):
            raise ValueError("correction: min_neff finite and >= 2, loo_mass_percentile in [0, 100], tps_max_rows an "
                             "integer >= 1")
        if p.n_draws < 1 or p.n_realizations < 1 or p.nodes_each_way < 1 or not (p.phantom_node_step_um > 0) \
                or p.jitter_xy_um < 0 or p.jitter_z_um < 0 or not (math.isfinite(p.radius_noise_sd_um)
                                                                   and p.radius_noise_sd_um >= 0):
            raise ValueError("phantom: n_draws, n_realizations, nodes_each_way >= 1, node step > 0, jitter >= 0, "
                             "radius_noise_sd_um finite and >= 0")
        if not all(isinstance(x, str) and x for x in c.failure_rate_ignore):
            raise ValueError("correction.failure_rate_ignore must hold non-empty reason names (e.g. 'dark')")
        if not (0 < c.bias_flag_threshold) or not (0 <= c.max_failure_rate <= 1):
            raise ValueError("correction thresholds out of range")

    # -------------------------------------------------------- signatures --
    def estimator_signature(self) -> Dict[str, Any]:
        """The estimator part of C (procedure s.3.3, 'shared by real and
        synthetic runs'): a real fit and the table that corrects it must agree
        on every key here."""
        return {"acquisition": _asdict(self.acquisition), "measure": _asdict(self.measure)}

    def full_signature(self) -> Dict[str, Any]:
        """Estimator + simulator settings; stored with every table."""
        d = self.estimator_signature()
        d["renderer"] = _asdict(self.renderer)
        d["phantom"] = _asdict(self.phantom)
        d["correction"] = _asdict(self.correction)
        return d

    def simulation_signature(self) -> Dict[str, Any]:
        """What a replicate's rendering and measurement depend on: the full
        signature without the correction settings, which only the merge reads.
        Every replicate row carries its hash, and the merge refuses rows
        rendered under another one (2026-10-07)."""
        d = self.full_signature()
        d.pop("correction")
        return d

    def signature_hash(self, which: str = "estimator") -> str:
        """sha256[:16] of the canonical JSON of a signature ("estimator",
        "simulation" or "full"); used in file names and row provenance."""
        sigs = {"estimator": self.estimator_signature, "simulation": self.simulation_signature,
                "full": self.full_signature}
        if which not in sigs:
            raise ValueError("signature %r is not one of %s" % (which, sorted(sigs)))
        return hashlib.sha256(canonical_json(sigs[which]()).encode("ascii")).hexdigest()[:16]

    def to_json(self) -> str:
        return canonical_json(self.full_signature())


# ----------------------------------------------------------------- helpers --
def _check_in(value: str, allowed: Tuple[str, ...], name: str) -> None:
    if value not in allowed:
        raise ValueError("%s = %r is not one of %r" % (name, value, allowed))


def _check_sgt(m: "MeasureConfig") -> None:
    """The SGT fit's settings (D-052.2, Block 12): ordered finite bounds with
    0 < w_lo, 0 < beta_lo and 0 <= xi_lo; positive lambda and IRLS tolerance;
    a cap and a budget of at least 1; at least one start shape; every start
    shape (beta, xi, zeta) and the start width inside the box."""
    (w_lo, w_hi), (b_lo, b_hi), (x_lo, x_hi), (z_lo, z_hi) = (m.sgt_w_bounds_um, m.sgt_beta_bounds,
                                                            m.sgt_xi_bounds, m.sgt_ln_zeta_bounds)
    edges = (w_lo, w_hi, b_lo, b_hi, x_lo, x_hi, z_lo, z_hi)
    if not all(math.isfinite(e) for e in edges) or not (0 < w_lo < w_hi and 0 < b_lo < b_hi and 0 <= x_lo < x_hi
                                                         and z_lo < z_hi):
        raise ValueError("measure.sgt_*_bounds: need finite 0 < w_lo < w_hi, 0 < beta_lo < beta_hi, "
                         "0 <= xi_lo < xi_hi and ln_zeta_lo < ln_zeta_hi")
    if not (m.sgt_lambda_um > 0 and m.sgt_irls_tol_um > 0) or m.sgt_irls_max_iter < 1 or m.sgt_max_nfev < 1:
        raise ValueError("measure: sgt_lambda_um and sgt_irls_tol_um must be > 0, sgt_irls_max_iter and "
                         "sgt_max_nfev >= 1")
    shapes = tuple(m.sgt_start_shapes) + (m.sgt_base_start_shape,)
    ok = len(m.sgt_start_shapes) >= 1 and all(len(s) == 3 for s in shapes)
    if ok:
        ok = all(b_lo <= s[0] <= b_hi and x_lo <= s[1] <= x_hi and s[2] > 0 and z_lo <= math.log(s[2]) <= z_hi
                 for s in shapes)
    if not ok:
        raise ValueError("measure.sgt_start_shapes / sgt_base_start_shape: at least one (beta, xi, zeta), each "
                         "inside sgt_beta_bounds, sgt_xi_bounds and exp(sgt_ln_zeta_bounds)")
    if not (w_lo <= m.sgt_start_w_um <= w_hi):
        raise ValueError("measure.sgt_start_w_um must lie within sgt_w_bounds_um")


def _check_iteration(m: "MeasureConfig") -> None:
    """The iteration's settings (D-042, D-044, D-052.3, D-069, Block 13): a
    finite line multiplier > 0; a round limit and a plane half-range that are
    integers >= 1 (5 and 5.0 pass; 5.5, True and "5" do not, as
    plane_iteration.check_iteration); a finite background margin >= 0; the
    first line's sample floor an odd integer >= 3."""
    def whole(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and float(x).is_integer()
    mult = m.iter_line_mult
    if not (isinstance(mult, (int, float)) and not isinstance(mult, bool) and math.isfinite(mult) and mult > 0):
        raise ValueError("measure.iter_line_mult must be finite and > 0, got %r" % (mult,))
    if not (whole(m.iter_max_rounds) and m.iter_max_rounds >= 1 and whole(m.iter_planes_half)
            and m.iter_planes_half >= 1):
        raise ValueError("measure.iter_max_rounds and iter_planes_half must be integers >= 1, got %r and %r"
                         % (m.iter_max_rounds, m.iter_planes_half))
    margin = m.iter_bg_margin_um
    if not (isinstance(margin, (int, float)) and not isinstance(margin, bool) and math.isfinite(margin)
            and margin >= 0):
        raise ValueError("measure.iter_bg_margin_um must be finite and >= 0, got %r" % (margin,))
    n = m.iter_min_first_samples
    if not (whole(n) and n >= 3 and int(n) % 2 == 1):
        raise ValueError("measure.iter_min_first_samples must be an odd integer >= 3, got %r" % (n,))


def _check_qtables(tables: Any) -> None:
    """renderer.jpeg_qtables: () or 1-4 tuples of 64 integers in [1, 65535].
    Tuples, not lists: the configuration is frozen and must stay hashable."""
    if not isinstance(tables, tuple) or len(tables) > 4:
        raise ValueError("renderer.jpeg_qtables must be a tuple of at most 4 tables (got %s); build it with "
                         "tuple(tuple(t) for t in tables)" % type(tables).__name__)
    for t in tables:
        if not (isinstance(t, tuple) and len(t) == 64
                and all(isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= 65535 for x in t)):
            raise ValueError("renderer.jpeg_qtables: every table is a tuple of 64 integers in [1, 65535]")


def _asdict(obj: Any) -> Dict[str, Any]:
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def _tuplify(value: Any) -> Any:
    """JSON lists back to tuples, nested lists included (renderer.jpeg_qtables)."""
    return tuple(_tuplify(v) for v in value) if isinstance(value, list) else value


# keys of older configuration files that no longer exist, with why and what replaced them
_LEGACY_KEYS = {
    ("renderer", "jpeg_qtables_file"): "it never was by the renderer: put the tables themselves in "
                                        "renderer.jpeg_qtables (camera_<specimen>.json, 'renderer')",
    ("phantom", "n_replicates"): "retired by D-062 (Block 14.2): the design is phantom.n_draws draws x "
                                 "phantom.n_realizations realizations; make the configuration again",
}


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, tuples as lists."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_from_dict(d: Dict[str, Any]) -> DiameterConfig:
    """Rebuild a DiameterConfig from ``full_signature()`` (tuples restored,
    nested ones included). A legacy key is dropped when it is empty (it had
    no effect) and refused otherwise, with why [2026-10-07: renderer.jpeg_qtables_file
    was a path the renderer never read; 2026-10-21: phantom.n_replicates, Block 14.2]."""
    kwargs = {}
    for f in fields(DiameterConfig):
        sub = dict(d.get(f.name, {}))
        for (section, key), why in _LEGACY_KEYS.items():
            if section == f.name and key in sub:
                if sub.pop(key):
                    raise ValueError("%s.%s is no longer read (%s)" % (section, key, why))
        cls = f.default_factory  # type: ignore[attr-defined]
        tuple_fields = {g.name for g in fields(cls) if str(g.type).startswith("Tuple")}
        kwargs[f.name] = cls(**{k: (_tuplify(v) if k in tuple_fields else v) for k, v in sub.items()})
    cfg = DiameterConfig(**kwargs)
    cfg.validate()
    return cfg


def default_config() -> DiameterConfig:
    cfg = DiameterConfig()
    cfg.validate()
    return cfg


def _coerce(annotation: str, value: Any) -> Any:
    """An override in the field's declared form: lists to tuples (nested ones
    included), and an integer given for a float field to a float, so that 0 and
    0.0 give one configuration and one hash."""
    value = _tuplify(list(value)) if isinstance(value, tuple) else _tuplify(value)

    def num(x: Any) -> Any:
        return float(x) if isinstance(x, int) and not isinstance(x, bool) else x
    if annotation == "float":
        return num(value)
    if annotation.startswith("Tuple[float") and isinstance(value, tuple):
        return tuple(num(x) for x in value)
    if annotation.startswith("Tuple[Tuple[float") and isinstance(value, tuple):   # measure.sgt_start_shapes
        return tuple(tuple(num(x) for x in row) if isinstance(row, tuple) else row for row in value)
    return value


def with_overrides(cfg: DiameterConfig, overrides: Dict[str, Any]) -> DiameterConfig:
    """A copy of cfg with dotted fields replaced, for example
    {"phantom.d_range_um": [0.2, 6.0], "renderer.jpeg_qtables": [[...64...]]}:
    values are put in the field's declared form (_coerce), an unknown section or
    field raises ValueError naming it, and the result is validated. Used by the
    Colab notebook to write the production configuration (2026-10-07)."""
    sections = [f.name for f in fields(DiameterConfig)]
    new = cfg
    for key, value in overrides.items():
        section, _, name = str(key).partition(".")
        if section not in sections or not name:
            raise ValueError("override %r: expected 'section.field' with a section in %s" % (key, sections))
        sub = getattr(new, section)
        declared = {f.name: str(f.type) for f in fields(sub)}
        if name not in declared:
            raise ValueError("override %r: %s has no field %r" % (key, type(sub).__name__, name))
        new = dataclasses.replace(new, **{section: dataclasses.replace(sub, **{name: _coerce(declared[name], value)})})
    new.validate()
    return new
