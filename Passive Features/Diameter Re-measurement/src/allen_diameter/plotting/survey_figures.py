"""The S3 survey's figures -- Blocks 16.2 (f) and 17 in specs/SPEC.md
(D-085.2, D-085.5). Plotting only: every array, frame and transform comes
prepared; nothing scientific is computed here.

  alignment_overlay  the projection at the coarse level with the skeleton under
                     the final transform, and the fine crop with the skeleton
                     under the coarse (grey) and the final (orange) transforms

Colours: the palette of the project's figures. Pure ASCII, for the cluster
(hpc-python-compat).
"""
from __future__ import annotations

import allen_image_plot as aip

GREY, INK, BLUE, ORANGE = "#a9a8a2", "#0b0b0b", "#2a78d6", "#eb6834"


def alignment_overlay(swc, images, rec, title=""):
    """Block 16.2 (f). swc: Block 0's SWC table; images: the second return of
    cell_alignment.align_record; rec: the alignment record."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    final = (float(rec["shift_x_px"]), float(rec["shift_y_px"]))
    flip = rec.get("flip_h_px")
    cf, ff = images["coarse_frame"], images["fine_frame"]
    aip.show_image(images["coarse"], cf.res_um_px, ax=axes[0], robust=True)
    aip.overlay_swc(axes[0], swc, cf, final, flip, color=ORANGE, lw=0.6)
    axes[0].set_title("projection, level %d (%.2f um/px)" % (cf.downsample, cf.res_um_px), fontsize=10)
    aip.show_image(images["fine"], ff.res_um_px, ax=axes[1], robust=True)
    ct = images["coarse_transform"]
    aip.overlay_swc(axes[1], swc, ff, ct["shift_full_px"], ct["flip_y_full_h"], color=GREY, lw=0.6, alpha=0.8)
    aip.overlay_swc(axes[1], swc, ff, final, flip, color=ORANGE, lw=0.6)
    fine = rec["fine"]
    axes[1].set_title("fine crop, level %d (%.2f um/px): coarse grey, final orange; fine step (%+d, %+d) px"
                      % (ff.downsample, ff.res_um_px, fine["dx_px"], fine["dy_px"]), fontsize=10)
    R = rec.get("score_ratio")
    fig.suptitle("%s  shift (%.1f, %.1f) full-res px, %s, score ratio %s -- %s" % (
        title, final[0], final[1], "y-flipped (H %.0f px)" % flip if flip is not None else "no flip",
        "%.2f" % R if R is not None else "n/a",
        "SET ASIDE (%s)" % ", ".join(rec["reasons"]) if rec.get("set_aside") else "accepted"), fontsize=11)
    fig.tight_layout()
    return fig
