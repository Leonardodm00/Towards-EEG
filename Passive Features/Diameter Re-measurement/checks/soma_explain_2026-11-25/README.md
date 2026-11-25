# Soma fit explanation, 2026-11-25

Standalone checks and figure code behind the explanation of the soma fit given to the user on 2026-11-25 (handoff
`docs/TEEG_soma_fit_handoff_2026-11-24.md`, section 1); not pipeline code. Simulated data only: a pilot-like ellipsoid
soma rendered with Block 45's renderer under the default configuration (`common.CONFIG`: traced depth 32.76 um and
radius 5.97 um as on the pilot; truth a 7.0, b 5.6, c 5.0 um, z0 30.96 um, mu 0.087 /um, chosen; background 110 gl,
noise SD 3 gl, seed 529). Python 3 with the repository's `src/` (set `TEEG_W` to the workstream folder if moved).

| Script | Arguments | What it shows | Result (run in the sandbox, 2026-11-25) |
|---|---|---|---|
| `recovery_d241.py` | OUT_JSON | the fit as it stands (far slope held, D-240; region the disc of Allen's radius, D-241; crop 14 um, D-243) recovers a simulated soma | converged, nothing at a bound, 72 evaluations; a, b, c within 0.06 %, z0 within 0.0004 um, S 430.8 against 430.5 um^2 (+0.07 %), chi2/dof 9.07 (noise 3 gl); `recovery_d241.out` |
| `figures.py` | OUT_DIR | fig1 the forward model and the blur law; fig2 a compact and a tall soma of equal 2 mu c on five planes; fig3 S against c, Legendre's form against numerical quadrature; fig4 the region and the crop | `figures_numbers.json`: tall (c 10) minus compact (c 5) 3.2-9.8 gl RMS on R per plane (nothing re-fitted); Legendre = quadrature to 4e-16; S linear in c beyond 5 um within 2 %, 64 um^2 per um at a 8.08, b 6.52 |

The recovery shows what the fit does when its model is true (ellipsoid, Gaussian blur law with the slope it holds, no
unmodelled structure); it says nothing about whether that model holds on the pilot's stack.
