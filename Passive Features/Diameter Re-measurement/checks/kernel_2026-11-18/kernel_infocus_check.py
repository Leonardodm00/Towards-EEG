#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-18: the in-focus kernel width sigma_r(0) is configured (0.080 um),
never identified by the stacks. If the TRUE in-focus width is larger -- the same amount of variance added at every
defocus, sigma_true(delta)^2 = sigma_T(delta)^2 + (s0^2 - 0.080^2) -- how far does the fitted diameter of a clear
node move when the fit keeps the table T? Set-up and fit of kernel_sensitivity_check.py.

Run:  python3 kernel_infocus_check.py <src>
"""
import importlib.util
import os
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("ks", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 "kernel_sensitivity_check.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
S0 = (0.12, 0.16, 0.22, 0.30)
TRUTHS = ((0.5, 0.8), (0.5, 0.4))
for d, mu in TRUTHS:
    th = np.array([d, mu, 0.0, 0.0])
    print("\ntruth d %.2f um, mu %.2f /um" % (d, mu), flush=True)
    for s0 in S0:
        extra = s0 ** 2 - m.SIGMA_T[0] ** 2
        sig = [np.sqrt(s ** 2 + extra) for s in m.SIGMA_T]
        data = m.B_GL * m.tau(th, m.cfg_for(sig, m.SLOPE), "s%.2f" % s0)
        x, rms = m.fit(data, th * [1.5, 0.7, 1, 1])
        print("  true in-focus sigma %.2f um (table 0.08):  fitted d %.3f um = %.2f x true, mu %.3f, rms %.2f gl "
              "(dip in focus %.1f gl)" % (s0, x[0], x[0] / d, x[1], rms, m.B_GL - data.min()), flush=True)
