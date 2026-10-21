"""D-062.1: with r_br = d/2 + eps, eps ~ N(0, s^2) redrawn until r_br > 0, the kept eps follows the normal
truncated below at -d/2, whose mean is s*phi(a)/(1-Phi(a)) with a = -d/(2s) (closed form, textbook).
Report P(reject) and the mean starting radius/diameter at s = 0.2 um for the design's thin end; check the
closed form against 2e6 draws (tolerance: 4 SE of the Monte Carlo mean)."""
import numpy as np
from scipy.stats import norm

S, SEED = 0.2, 20261020
rng = np.random.default_rng(SEED)
for d in (0.2, 0.3, 0.4, 0.6, 1.0):
    a = -0.5 * d / S
    p_rej = norm.cdf(a)
    shift = S * norm.pdf(a) / (1.0 - norm.cdf(a))
    eps = rng.normal(0.0, S, 2_000_000)
    kept = eps[eps > -0.5 * d]
    mc = kept.mean()
    se = kept.std(ddof=1) / np.sqrt(kept.size)
    ok = abs(mc - shift) < 4 * se
    print("d=%.1f um: P(r_br<=0)=%.3f  mean shift of r_br: closed form %.4f um, MC %.4f um (%s); "
          "mean start diameter 2 E[r_br] = %.3f um (true %.1f)" % (d, p_rej, shift, mc, "PASS" if ok else "FAIL", 2 * (0.5 * d + shift), d))
