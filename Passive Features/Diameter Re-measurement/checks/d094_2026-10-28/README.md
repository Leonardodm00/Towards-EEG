# Check behind D-094 (calibration nodes without the faint criteria), 2026-10-28

Not pipeline code: the numerical check behind SPEC section 8's D-094 item. Run from the
workstream directory (`Passive Features/Diameter Re-measurement/`), sandbox, one core. The
script imports `tests/smoke/test_smoke_calibration.py` for Phase I's rendered flat phantom and its
scan (`scan_phantom`), so the check and the smoke test render and scan the same way. Each `.out` is
the script's output.

| command | what it computes | time |
|---|---|---|
| `python checks/d094_2026-10-28/dark_node_growth.py` | Block 10's near growth fit (Eq. 4) on noise-free flat phantoms of $d$ = 0.3 and 0.5 um, optical depth $\alpha = \mu d$ = 0.1, 0.25, 0.5, 1, 1.5, 2, five phantoms per ($d$, $\alpha$) on the same geometries, partition renderer (the default) with the default table: $\hat G$ and $\hat\sigma_{\rm r}$ at 0.28, 0.56, 0.84 um, and $\hat\sigma_{\rm r}$(0.84 um) against $\alpha$ = 0.1's | ~5 min |
| `python checks/d094_2026-10-28/dark_node_growth.py --absorption hybrid --diameters 0.3 --alphas 0.1,1,1.5 --nodes 4` | the same under Block 21's hybrid absorption, $d$ = 0.3 um, four phantoms | ~10 min |

What the two `.out` files show: between $\alpha$ = 0.1 and 2 the fitted
$\hat\sigma_{\rm r}$(0.84 um) moves by less than 1 % (partition, both diameters) and by 0.7 % up to
$\alpha$ = 1.5 (hybrid); $\hat\sigma_{\rm r}$(0.56 um) by at most 1.7 % (partition) and 1.0 % (hybrid).
The one-plane knot is the exception in relative terms: the table's growth there is only 0.0085 um$^2$
(already within Phase I's bias), and $\hat G$(0.28 um) moves by up to 0.005 um$^2$ at $d$ = 0.5 um.

The row `proj` of each diameter is Phase I's oracle on the same design: the growth fit of ideal Eq. (4)
data whose curve is the faint tube's expected growth (`semicircle_growth`). At $\alpha$ = 0.1 the fitted
growth sits below it by 0.020 um$^2$ ($d$ = 0.3 um) and 0.047 um$^2$ ($d$ = 0.5 um) at 0.56 um, four to
eleven times the largest $\alpha$ effect there (0.005 and 0.004 um$^2$): the first-stage table at two planes is 0.389 and 0.357 um against the configured
0.438 um. That is Phase I's finding (a tube's depth extent mixes defocus values, and the core width of the
mixture grows more slowly) at larger $d$; the first stage needs procedure s.3.4's step 2 (not built).

Not shown, and why:
- Whether either renderer images a dark tube as the microscope does (T7) is not tested here.
- The far set (Block 15) is not covered; D-026 (e) put the cut for dark thick tubes there.
- The flag `faint` is about noise, and these phantoms are noise-free.
