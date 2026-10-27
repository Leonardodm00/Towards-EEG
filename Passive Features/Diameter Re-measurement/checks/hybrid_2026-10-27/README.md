# Checks of the hybrid absorption (SPEC Block 21), 2026-10-27

Not pipeline code: surveys behind Block 21's findings. Each script takes the
path of `src/` as its only argument; run from the workstream directory
(`Passive Features/Diameter Re-measurement/`), sandbox, one core. Each `.out`
is that script's output.

| script | what it computes | command | time |
|---|---|---|---|
| `profile_survey.py` | minimum of $\tau = I/B$ across an infinite flat tube (1-D: the 2-D Gaussian blur of a tube along x is a 1-D blur across it), at the node plane and at $\pm 0.28$, $\pm 0.56$ um, for $d$ = 0.5, 1, 2, 3 um and $\mu d$ = 0.3-10: the hybrid and the partition with the production Gaussian kernel, the ray world with and without its 0.08 um post blur | `python checks/hybrid_2026-10-27/profile_survey.py src` | ~15 min |
| `cone_survey.py` | the like-for-like comparison (renderers document s.3.7, s.3.9 tests 1-2 at profile level, no fit): the hybrid and the partition blurred by the cone's own kernel weighted $W^*$, against the exact ray world without post blur; minimum of $\tau$, largest $|\tau - \tau^{\rm ray}|$, dip area against the ray world's | `python checks/hybrid_2026-10-27/cone_survey.py src` | ~15 min |
| `directions_planes.py` | rendered planes (Block 4 with `absorption = "hybrid"`) at 8 x 16 and 16 x 32 directions against 32 x 64, flat and 40-degree 1 um tubes, $\mu$ = 1 and 3 /um | `python checks/hybrid_2026-10-27/directions_planes.py src` | ~3 min |
| `cost.py` | wall time of one phantom block (87 x 87 px, 17 px padding, $h = p_{\rm x}/8$, 7 planes) with the partition and with the hybrid | `python checks/hybrid_2026-10-27/cost.py src` | ~15 min |

Read with Block 21's findings in `specs/SPEC.md`: the hybrid's planes go below
zero in dark stain with the production kernel (from $\mu d \approx 2$ at the
node plane) and, beyond $\mu d \approx 3$, with the cone's kernel too; like for
like it is much closer to the ray world than the partition up to
$\mu d \approx 1.5$; it costs 4 to 50 times the partition per phantom. The
timings in `cost.out` were taken while another check ran on the second core
(the 45-degree case).
