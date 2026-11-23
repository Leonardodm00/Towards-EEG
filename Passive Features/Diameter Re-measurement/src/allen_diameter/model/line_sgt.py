"""The SGT point-spread function in the fast line renderer -- Block 49 in specs/SPEC.md (wave 7; D-221, D-222, D-224,
D-225).

The quantity is Block 35's (model/line_fast.py): for a capped straight tube, mu, plane indices ks and Block 5's
measuring line, the value tau_k(v_n) that profiles.sample_profile reads from the pixel means of plane k,

    tau_k = 1 - read( pixel means of sum_j Delta A_j * K_jk ),

with another kernel K_jk in place of Block 4's Gaussian. Slab j (centre zeta_j) and plane k (depth z_k = k dz):
delta_jk = zeta_j - z_k; the law's SD sigma_jk = sigma_r(delta_jk) (Block 4, the configured table); the SGT's FWHM
w_jk = 2 sqrt(2 ln 2) sigma_jk (D-222.2); the shape kappa = (beta, xi, zeta) (D-052.1); the core scale
a_jk = w_jk / W(kappa), W of (H3) (blur_kernels.sgt_unit_fwhm). h is the symmetric core density of (H1) at unit
scale (blur_kernels.sgt_unit_pdf(x, beta, xi, 1)), hhat(t) = int h(x) exp(-2 pi i t x) dx its characteristic
function (real, even).

- zeta = 1 (D-222.1, D-066.1): K_jk is the isotropic PSF whose line projection is g_{w_jk,(beta,xi,1)}; its 2-D
  Fourier transform is hhat(a_jk |nu|) (projection-slice theorem). P_jk(r) denotes its radial profile.
- zeta != 1 (inference.sgt_skew "fit"; D-225.3): with u = x . y_hat (u > 0 towards increasing v) and s = x . e_u,
      K_jk(x) = (2 / (zeta + 1/zeta)) P_jk(sqrt((u / zeta)^2 + s^2))     (u >= 0)
      K_jk(x) = (2 / (zeta + 1/zeta)) P_jk(sqrt((zeta u)^2 + s^2))       (u < 0),
  whose line projection across y_hat, int K_jk(u y_hat + s e_u) ds, is the two-piece SGT g_{w_jk,kappa}(u) of
  (H2)-(H3); zeta = 1 gives the isotropic kernel.
- Truncation (D-225.4): the kernel is applied within R_jk = inference.sgt_reach_sigmas x sigma_jk of its centre
  in the radial variable of P_jk (before the stretch), and its mass beyond is dropped, not renormalised. R_jk / a_jk
  = sgt_reach_sigmas x W(kappa) / (2 sqrt(2 ln 2)) depends on kappa only, so the truncated kernel scales with the
  width and tau is smooth in kappa. The dropped fraction is the isotropic kernel's 2-D mass beyond that radius (the
  stretch keeps the fraction): truncated_mass.
- Slabs, absorption, the hybrid's cross-section, the pixel means (the mean over the pixel square of the blurred
  field, each fine-lattice sample a point mass h^2 Delta A_j) and the read are Block 35's.

Pure ASCII, for the cluster (hpc-python-compat). Contract stubs (wave 7): the implementer of Block 49 fills them.
"""
from __future__ import annotations

import dataclasses
import math

from . import blur_kernels


@dataclasses.dataclass(frozen=True)
class SGTKernel:
    """The SGT shape kappa = (beta, xi, zeta) of Block 49's kernel (D-052.1); ValueError outside the family's domain
    (blur_kernels.check_sgt_shape: beta > 0, xi >= 0, zeta > 0, all finite)."""

    beta: float
    xi: float
    zeta: float = 1.0

    def __post_init__(self):
        b, x, z = blur_kernels.check_sgt_shape(self.beta, self.xi, self.zeta)
        object.__setattr__(self, "beta", b)
        object.__setattr__(self, "xi", x)
        object.__setattr__(self, "zeta", z)


FWHM_PER_SD = 2.0 * math.sqrt(2.0 * math.log(2.0))     # a Gaussian's FWHM over its SD (D-222.2)


def sgt_tail_mass(beta, xi, r_over_a):
    """The isotropic unit kernel's 2-D mass beyond radius r (core-scale units, r_over_a >= 0):
    1 - int_0^r P_1(rho) 2 pi rho drho, P_1 the radial profile of the isotropic PSF whose line projection is h of
    (H1) (zeta = 1). -> float in [0, 1]. Closed forms for the oracles: the Gaussian member (2, 0) gives exp(-r^2);
    the Laplace member (1, 0) gives r K_1(r)."""
    raise NotImplementedError("Block 49 (wave 7)")


def truncated_mass(kernel, cfg):
    """The fraction of every pair's kernel dropped by the truncation (D-225.4): sgt_tail_mass(beta, xi, R / a) with
    R / a = cfg.inference.sgt_reach_sigmas x W(kappa) / FWHM_PER_SD. kernel an SGTKernel; cfg a DiameterConfig."""
    raise NotImplementedError("Block 49 (wave 7)")


def profiles(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption, cache, kernel):
    """tau (n_planes, n_v) of line_fast.line_profiles with the SGT kernel `kernel` (an SGTKernel); the arguments
    already validated by line_fast (absorption a name, cache a line_fast.SectionCache)."""
    raise NotImplementedError("Block 49 (wave 7)")
