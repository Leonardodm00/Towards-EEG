"""D-SFP.2 check: (1) the continuous Gaussian transfer exp(-2 pi^2 s^2 nu^2) on a sample grid gives a kernel with
negative lobes when s is below about one sample; (2) the Poisson-summed transfer of the normalised sampled Gaussian
equals that kernel's DFT; (3) a step blurred with each: the continuous transfer overshoots, the sampled one does not."""
import math
import numpy as np

N, h = 4096, 1.0
nu = np.fft.fftfreq(N, d=h)

def transfer(s, nu, h):
    M = max(1, math.ceil(1.2 * h / s))
    m = np.arange(-M, M + 1)
    num = np.exp(-2 * math.pi ** 2 * s ** 2 * (nu[..., None] - m / h) ** 2).sum(-1)
    den = np.exp(-2 * math.pi ** 2 * s ** 2 * (m / h) ** 2).sum()
    return num / den

worst = 0.0
for s in (0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 20.0):
    n = np.arange(N); n = np.minimum(n, N - n)
    w = np.exp(-(n * h) ** 2 / (2 * s ** 2)); w /= w.sum()
    dft = np.fft.fft(w).real
    err = np.max(np.abs(dft - transfer(s, nu, h)))
    worst = max(worst, err)
    k_cont = np.fft.ifft(np.exp(-2 * math.pi ** 2 * s ** 2 * nu ** 2)).real
    step = (np.arange(N) < N // 2).astype(float)
    over_c = np.max(np.fft.ifft(np.fft.fft(step) * np.exp(-2 * math.pi ** 2 * s ** 2 * nu ** 2)).real) - 1
    over_s = np.max(np.fft.ifft(np.fft.fft(step) * transfer(s, nu, h)).real) - 1
    print("s/h=%5.2f  |DFT - Poisson| %.1e   continuous kernel min %+.2e   step overshoot: continuous %+.2e, sampled %+.2e"
          % (s / h, err, k_cont.min(), over_c, over_s))
print("worst |DFT - Poisson| over s/h in [0.2, 20]: %.1e" % worst)
