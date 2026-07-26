"""phi_pipeline_colab -- Drive-facing driver for the spine-area density phi.

Towards-EEG stage S1.3, data-generation side. This module runs in COLAB against
the Drive database, in the same style as compute_F_factors in
morpholgy_pathways__6_.py: batch-label the requested neurons, loop per cell with
[WARN] guards, aggregate into a per-cell table plus a summary, persist to disk,
return (df_out, summary).

WHAT IT PRODUCES (per run, under output_path)
    neuron_<nid>_phi.csv          phi(nu, b, d) for one cell, one row per shaft
                                  segment                              [per cell]
    <output_filename>.csv         per-cell table (f_implied, areas, QC)
    <output_filename>.pkl         full results incl. summary and profiles
    <output_filename>_psi_profile.csv
                                  population psi-vs-distance profile (S1.7 input)
    <output_filename>_provenance.json
                                  phi_id: module version, labeler identity and
                                  source hash, threshold, and the three design
                                  decisions, so any phi CSV can be traced to the
                                  exact code and settings that made it

SEPARATION OF CONCERNS
    computation  -> spine_density.py   (no I/O, no plotting)
    Drive I/O + orchestration -> this module, sections 1-4
    visualisation -> this module, section 5, separate functions that take
                     DataFrames and never touch disk or recompute anything

WHY THE LABELLER IS INJECTED, NOT IMPORTED
    morpholgy_pathways__6_.py contains TWO shadowed definitions of
    label_dendritic_spines_robust (L1653, threshold 5000 nm, head/neck aware;
    L2353, threshold 3000 nm, no head/neck). That is one of the O7 shadowed
    pairs. Vendoring a copy here would create a THIRD definition. Instead the
    function is passed in (or resolved from the notebook globals) and its
    identity plus source hash are recorded in the provenance file, so which
    definition ran is a matter of record rather than of guesswork.

Pure ASCII source. Importable outside Colab (the Drive mount is inside a
function, and google.colab is imported lazily), so the smoke test can run
anywhere.
"""

import hashlib
import inspect
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

import spine_density as sd


# --------------------------------------------------------------------------- #
# 0. Configuration -- paths match the notebook's Drive layout                  #
# --------------------------------------------------------------------------- #
DEFAULT_INPUT_DIR = ("/content/drive/MyDrive/Colab Notebooks/"
                     "Reconstructed neurons")
DEFAULT_PHI_DIR = "/content/drive/MyDrive/Colab Notebooks/Spine Density Phi"
DEFAULT_SPINES_DIR = "/content/drive/MyDrive/Colab Notebooks/Spine Labels"

# Confirmed spine-identification threshold for this project (S1.3).
SPINE_LENGTH_THRESHOLD_NM = 4000.0


def mount_drive(force_remount=True):
    """Mount Google Drive, exactly as the notebook does. No-op off Colab."""
    try:
        from google.colab import drive  # noqa: F401  (lazy, Colab-only)
    except ImportError:
        print("[WARN] google.colab not available -- not mounting Drive. "
              "Set input_dir/output_path to local paths.")
        return False
    from google.colab import drive
    drive.mount("/content/drive", force_remount=force_remount)
    return True


# --------------------------------------------------------------------------- #
# 1. Discovery and QC                                                          #
# --------------------------------------------------------------------------- #
def discover_neuron_ids(input_dir=DEFAULT_INPUT_DIR, verbose=True):
    """List neuron ids for which 'neuron_<id>.csv' exists in input_dir.

    Matches only the plain skeleton files: files with an extra suffix
    (neuron_<id>_spines.csv, neuron_<id>_synapses.csv, ...) are excluded.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError("input_dir does not exist: %s" % input_dir)
    ids = []
    for fname in sorted(os.listdir(input_dir)):
        if not (fname.startswith("neuron_") and fname.endswith(".csv")):
            continue
        stem = fname[len("neuron_"):-len(".csv")]
        if stem and "_" not in stem:
            ids.append(int(stem) if stem.isdigit() else stem)
    if verbose:
        print("[OK] Discovered %d neuron skeleton files in %s"
              % (len(ids), input_dir))
    return ids


def radius_report(df, default_radius_nm=sd.DEFAULT_RADIUS_NM,
                  default_dominated_frac=0.99):
    """Diagnose the 'r' column -- the single biggest risk to phi's validity.

    Both label_dendritic_spines_robust and _compute_F_for_neuron silently fall
    back to a FLAT 50 nm radius when 'r' is absent. Spine head and neck radii
    drive the membrane area, so if radii are absent or constant, phi and
    f_implied are geometric artefacts rather than measurements.

    IMPORTANT: flatness is assessed on NON-SOMA nodes only. The soma frequently
    carries a distinct radius even when every dendritic node sits at the
    fallback, so a whole-cell uniqueness test would report 2 distinct radii and
    silently pass a cell whose entire dendrite is synthetic.

    Returns a dict; the field to check before trusting any density number is
    'radius_suspect' (true if the dendritic radii are constant OR overwhelmingly
    at the default fallback value).
    """
    out = {
        "has_r_column": bool("r" in df.columns),
        "n_nodes": int(len(df)),
        "n_nodes_nonsoma": 0,
        "r_median_nm": float("nan"),
        "r_min_nm": float("nan"),
        "r_max_nm": float("nan"),
        "n_unique_r": 0,
        "frac_at_default_r": float("nan"),
        "flat_radius": True,
        "default_dominated": True,
        "radius_suspect": True,
    }
    if not out["has_r_column"]:
        return out

    labels = df["annotated_type"].astype(str)
    is_soma = labels.str.contains("soma", case=False, na=False, regex=False)
    sub = df.loc[~is_soma]
    if len(sub) == 0:                      # degenerate: fall back to all nodes
        sub = df
    r = pd.to_numeric(sub["r"], errors="coerce").dropna().values
    out["n_nodes_nonsoma"] = int(len(r))
    if len(r) == 0:
        return out

    out["r_median_nm"] = float(np.median(r))
    out["r_min_nm"] = float(np.min(r))
    out["r_max_nm"] = float(np.max(r))
    out["n_unique_r"] = int(len(np.unique(np.round(r, 6))))
    out["frac_at_default_r"] = float(
        np.count_nonzero(np.isclose(r, default_radius_nm)) / len(r))
    out["flat_radius"] = bool(out["n_unique_r"] <= 1)
    out["default_dominated"] = bool(
        out["frac_at_default_r"] >= default_dominated_frac)
    out["radius_suspect"] = bool(out["flat_radius"] or out["default_dominated"])
    return out


def spine_node_counts(df, spine_labels=sd.SPINE_LABELS):
    """Count spine nodes and spine roots directly from the labelled frame.

    Independent of spine_density's area attribution, so a disagreement between
    these counts and the phi output is informative rather than tautological.
    """
    spine_set = {s.lower() for s in spine_labels}
    labels = df["annotated_type"].astype(str).str.lower()
    is_spine = labels.isin(spine_set)
    parent_of = dict(zip(df["id"].values, df["p"].values))
    spine_ids = set(df.loc[is_spine, "id"].values)
    n_roots = 0
    for nid in spine_ids:
        if parent_of.get(nid, -1) not in spine_ids:
            n_roots += 1
    return {"n_spine_nodes": int(len(spine_ids)), "n_spine_roots": int(n_roots)}


# --------------------------------------------------------------------------- #
# 2. Labeller resolution and provenance (the phi_id)                           #
# --------------------------------------------------------------------------- #
def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _callable_name(fn):
    """Best available name for any callable.

    Plain functions carry __qualname__, but a functools.partial, a callable
    class instance or some decorated functions do not, and instances never
    inherit their class's __qualname__. Falling back to '?' would silently
    destroy the record of WHICH labeller ran, which is the whole point of the
    provenance file (the two shadowed O7 definitions), so resolve through
    several fallbacks and only ever give up to a repr.
    """
    for obj in (fn, type(fn)):
        for attr in ("__qualname__", "__name__"):
            name = getattr(obj, attr, None)
            if isinstance(name, str) and name:
                return name
    inner = getattr(fn, "func", None)          # functools.partial
    if inner is not None:
        return _callable_name(inner)
    return repr(fn)


def _resolve_labeller(label_fn=None):
    """Return the spine labeller, preferring an explicitly passed function.

    Falls back to 'label_dendritic_spines_robust' in the notebook globals
    (__main__), which is where a Colab cell definition lives.
    """
    if label_fn is not None:
        return label_fn
    import __main__
    fn = getattr(__main__, "label_dendritic_spines_robust", None)
    if fn is None:
        raise RuntimeError(
            "No spine labeller found. Run the notebook cell that defines "
            "label_dendritic_spines_robust first, or pass it explicitly: "
            "compute_phi_factors(..., label_fn=label_dendritic_spines_robust). "
            "Note there are two shadowed definitions (L1653 and L2353) -- pass "
            "the one you intend to use.")
    return fn


def _labeller_provenance(fn):
    """Identity + source hash of the labeller actually used (O7 bookkeeping)."""
    prov = {
        "qualname": _callable_name(fn),
        "module": getattr(fn, "__module__", "unknown"),
        "source_file": None,
        "first_line": None,
        "source_sha256": None,
    }
    try:
        prov["source_file"] = inspect.getsourcefile(fn)
        src, first_line = inspect.getsourcelines(fn)
        prov["first_line"] = int(first_line)
        prov["source_sha256"] = _sha256_text("".join(src))
    except (OSError, TypeError):
        pass  # Colab cell sources are not always retrievable
    return prov


def _module_provenance():
    prov = {"module_version": sd.MODULE_VERSION, "source_sha256": None}
    try:
        with open(inspect.getsourcefile(sd), "r", encoding="utf-8") as fh:
            prov["source_sha256"] = _sha256_text(fh.read())
    except (OSError, TypeError):
        pass
    return prov


# --------------------------------------------------------------------------- #
# 3. Population entry point                                                    #
# --------------------------------------------------------------------------- #
def compute_phi_factors(target_ids,
                        output_path=DEFAULT_PHI_DIR,
                        output_filename="phi_bank",
                        input_dir=DEFAULT_INPUT_DIR,
                        spine_length_threshold_nm=SPINE_LENGTH_THRESHOLD_NM,
                        label_fn=None,
                        labeller_kwargs=None,
                        input_units="nm",
                        bin_width_um=10.0,
                        literature_cutoff_um=60.0,
                        cutoff_by="d_from_um",
                        save_per_cell_phi=True,
                        spines_output_dir=None,
                        verbose=True):
    """Build and persist phi(nu, b, d) for every neuron in target_ids.

    Parameters
    ----------
    target_ids : list
        Neuron ids. Files 'neuron_<id>.csv' must exist in input_dir.
    output_path : str
        Directory for all outputs (created if absent).
    output_filename : str
        Base name for the run-level files, e.g. 'phi_bank_L23_pyr'.
    input_dir : str
        Directory holding 'neuron_<id>.csv'.
    spine_length_threshold_nm : float
        Subtree-length threshold below which a dendritic side branch is
        reclassified as a spine. Project setting: 4000 nm.
    label_fn : callable or None
        The spine labeller. If None, resolved from the notebook globals.
    labeller_kwargs : dict or None
        Extra keyword arguments forwarded to the labeller (e.g.
        {'smoothing_sigma': 2.0}); the L2353 variant accepts fewer.
    input_units : {'nm', 'um'}
        Units of the coordinates and radii in the CSVs.
    bin_width_um : float
        Bin width for the population psi-vs-distance profile.
    literature_cutoff_um : float
        Proximal cutoff used ONLY for the literature-comparison column F_lit
        (spine_density.cell_f_beyond_cutoff). Default 60.0 matches Eyal et al.
        (2016/2018) and the Benavides-Piccione region studies (Htemp, Hcing,
        MCA1, HCA1); note the SAME sources use 30.0 for mouse. This does NOT
        affect phi itself, which is stored with no cutoff (S1.3 decision) --
        it only affects this one reporting column, computed by integrating the
        already-built, cutoff-free phi.
    cutoff_by : {'d_from_um', 'd_to_um', 'mid'}
        See spine_density.cell_f_beyond_cutoff.
    save_per_cell_phi : bool
        Write neuron_<nid>_phi.csv per cell.
    spines_output_dir : str or None
        If set, the labeller also writes neuron_<nid>_spines.csv there.

    Returns
    -------
    (df_out, summary)
        df_out  : pandas.DataFrame indexed by neuron_id
        summary : dict, also persisted in the .pkl and the provenance JSON
    """
    os.makedirs(output_path, exist_ok=True)
    labeller = _resolve_labeller(label_fn)
    labeller_kwargs = dict(labeller_kwargs or {})

    # ------------------------------------------------------------------- #
    # 1. Load and label every neuron in one batch call                     #
    # ------------------------------------------------------------------- #
    if verbose:
        print(" Loading and labelling %d neurons from %s ..."
              % (len(target_ids), input_dir))
        print("   labeller: %s  (threshold %.0f nm)"
              % (_callable_name(labeller), spine_length_threshold_nm))
    call_kwargs = dict(
        input_dir=input_dir,
        spine_length_threshold_nm=spine_length_threshold_nm,
    )
    if spines_output_dir is not None:
        call_kwargs["output_dir"] = spines_output_dir
    call_kwargs.update(labeller_kwargs)
    neuron_dict = labeller(list(target_ids), **call_kwargs)

    # ------------------------------------------------------------------- #
    # 2. Build phi per neuron                                              #
    # ------------------------------------------------------------------- #
    per_cell = {}
    phi_frames = []
    for nid in target_ids:
        if nid not in neuron_dict:
            if verbose:
                print("[WARN]  Neuron %s: no labelled DataFrame returned. "
                      "Skipping." % nid)
            continue
        df_lab = neuron_dict[nid]
        try:
            phi_df = sd.build_phi(df_lab, nid=nid, input_units=input_units,
                                  self_check=True)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print("[WARN]  Neuron %s: phi computation failed (%s). "
                      "Skipping." % (nid, e))
            continue

        rq = radius_report(df_lab)
        sc = spine_node_counts(df_lab)
        f_implied = sd.cell_f_implied_from_phi(phi_df)
        # Literature-comparable F: cutoff applied to BOTH numerator and
        # denominator, matching Eyal et al. (2016/2018) and Benavides-Piccione
        # et al. -- NOT the same quantity as f_implied. See
        # spine_density.cell_f_beyond_cutoff for why the two differ.
        f_lit = sd.cell_f_beyond_cutoff(phi_df, cutoff_um=literature_cutoff_um,
                                        by=cutoff_by)

        phi_path = ""
        if save_per_cell_phi:
            phi_path = os.path.join(output_path, "neuron_%s_phi.csv" % nid)
            sd.save_phi(phi_df, phi_path)

        rec = {
            "f_implied": f_implied,
            "F_lit": f_lit["F"],
            "F_lit_cutoff_um": literature_cutoff_um,
            "F_lit_frac_shaft_beyond_cutoff": f_lit["frac_shaft_area_included"],
            "A_shaft_um2": float(phi_df["shaft_area_um2"].sum()),
            "A_spine_um2": float(phi_df["spine_area_um2"].sum()),
            "n_shaft_segments": int(len(phi_df)),
            "n_branches": int(phi_df["branch_id"].nunique()) if len(phi_df) else 0,
            "max_path_distance_um": float(phi_df["d_to_um"].max())
            if len(phi_df) else float("nan"),
            "max_seg_len_um": float(phi_df["seg_len_um"].max())
            if len(phi_df) else float("nan"),
            "dropped_spine_area_um2":
                float(phi_df.attrs.get("dropped_spine_area_um2", 0.0)),
            "n_spine_nodes": sc["n_spine_nodes"],
            "n_spine_roots": sc["n_spine_roots"],
            "has_r_column": rq["has_r_column"],
            "r_median_nm": rq["r_median_nm"],
            "n_unique_r": rq["n_unique_r"],
            "frac_at_default_r": rq["frac_at_default_r"],
            "flat_radius": rq["flat_radius"],
            "default_dominated": rq["default_dominated"],
            "radius_suspect": rq["radius_suspect"],
            "phi_path": phi_path,
        }
        per_cell[nid] = rec
        phi_frames.append(phi_df)

        if verbose:
            warn = "  [!] RADII SUSPECT -- phi not trustworthy" \
                if rq["radius_suspect"] else ""
            print("   * neuron %s: f_implied = %.3f  F_lit(>=%gum) = %.3f  "
                  "(shaft segs=%d, branches=%d, spine roots=%d)%s"
                  % (nid, f_implied, literature_cutoff_um, f_lit["F"],
                     rec["n_shaft_segments"], rec["n_branches"],
                     rec["n_spine_roots"], warn))

    # ------------------------------------------------------------------- #
    # 3. Aggregate                                                         #
    # ------------------------------------------------------------------- #
    df_out = pd.DataFrame.from_dict(per_cell, orient="index")
    df_out.index.name = "neuron_id"

    def _pop_stats(col):
        if len(df_out) == 0:
            return float("nan"), float("nan"), float("nan"), 0
        vals = df_out[col].dropna().values
        if len(vals) == 0:
            return float("nan"), float("nan"), float("nan"), 0
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sem = std / float(np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        return mean, std, sem, int(len(vals))

    f_mean, f_std, f_sem, f_n = _pop_stats("f_implied")
    flit_mean, flit_std, flit_sem, flit_n = _pop_stats("F_lit")

    # Population psi-vs-distance profile: pooled, area-weighted across cells.
    if phi_frames:
        pooled = pd.concat(phi_frames, ignore_index=True)
        psi_profile = sd.psi_vs_distance(pooled, bin_width_um=bin_width_um)
    else:
        psi_profile = sd.psi_vs_distance(pd.DataFrame())

    summary = {
        "f_implied_mean": f_mean,
        "f_implied_std": f_std,
        "f_implied_sem": f_sem,
        # F_lit is the quantity comparable to published F values (Eyal et al.
        # 2016/2018 human L2/3 temporal cortex, F=1.9; Benavides-Piccione et
        # al. MCA1/HCA1, F~2) -- f_implied is NOT, by construction (S1.3: no
        # proximal cutoff). Use F_lit_* for any literature comparison in S1.7.
        "F_lit_mean": flit_mean,
        "F_lit_std": flit_std,
        "F_lit_sem": flit_sem,
        "F_lit_cutoff_um": literature_cutoff_um,
        "F_lit_cutoff_by": cutoff_by,
        "n_cells": f_n,
        "n_cells_requested": len(target_ids),
        "n_cells_radius_suspect": int(df_out["radius_suspect"].sum())
        if len(df_out) else 0,
        "spine_length_threshold_nm": spine_length_threshold_nm,
        "spine_labels_used": list(sd.SPINE_LABELS),
        "shaft_regex": sd.SHAFT_REGEX,
        "input_units": input_units,
        "bin_width_um": bin_width_um,
        # The three S1.3 design decisions, recorded so a phi CSV is self-describing
        "proximal_cutoff_applied": False,
        "spine_area_attribution": "base_segment",
        "phi_representation": "piecewise_constant_per_shaft_segment",
    }

    # ------------------------------------------------------------------- #
    # 4. Persist                                                           #
    # ------------------------------------------------------------------- #
    csv_path = os.path.join(output_path, "%s.csv" % output_filename)
    pkl_path = os.path.join(output_path, "%s.pkl" % output_filename)
    prof_path = os.path.join(output_path, "%s_psi_profile.csv" % output_filename)
    prov_path = os.path.join(output_path, "%s_provenance.json" % output_filename)

    df_out.to_csv(csv_path)
    psi_profile.to_csv(prof_path, index=False)
    with open(pkl_path, "wb") as fh:
        pickle.dump({"per_cell": per_cell,
                     "summary": summary,
                     "per_cell_table": df_out,
                     "psi_profile": psi_profile}, fh)

    provenance = {
        "phi_id": "%s|%s|thr=%.0fnm" % (
            sd.MODULE_VERSION,
            _callable_name(labeller),
            spine_length_threshold_nm),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spine_density_module": _module_provenance(),
        "labeller": _labeller_provenance(labeller),
        "input_dir": input_dir,
        "output_path": output_path,
        "target_ids": [str(t) for t in target_ids],
        "summary": summary,
    }
    with open(prov_path, "w", encoding="ascii") as fh:
        json.dump(provenance, fh, indent=2, default=str)

    if verbose:
        print("\n[OK] Population f_implied (n=%d/%d): %.3f +/- %.3f "
              "(mean +/- SD)" % (f_n, len(target_ids), f_mean, f_std))
        print("[OK] Population F_lit (>=%gum, literature-comparable, n=%d/%d): "
              "%.3f +/- %.3f (mean +/- SD)  [cf. Eyal et al. 2016/2018 F=1.9; "
              "Benavides-Piccione et al. MCA1/HCA1 F~2]"
              % (literature_cutoff_um, flit_n, len(target_ids), flit_mean,
                 flit_std))
        if summary["n_cells_radius_suspect"]:
            print("[WARN] %d/%d cells have SUSPECT radii (constant, or >=99%% at "
                  "the 50 nm fallback): their phi reflects the fallback, not "
                  "measured geometry."
                  % (summary["n_cells_radius_suspect"], f_n))
        print("   Saved per-cell table  -> %s" % csv_path)
        print("   Saved psi profile     -> %s" % prof_path)
        print("   Saved full results    -> %s" % pkl_path)
        print("   Saved provenance      -> %s" % prov_path)

    return df_out, summary


# --------------------------------------------------------------------------- #
# 5. VISUALISATION -- kept separate from computation and from I/O              #
#    These take DataFrames only. They never recompute and never write CSVs.    #
# --------------------------------------------------------------------------- #
def plot_psi_profile(psi_profile, ax=None, label=None, show_counts=False):
    """Plot the dimensionless spine density psi against path distance.

    This is the figure compared to published distance-resolved human spine
    densities in S1.7. Expects the output of spine_density.psi_vs_distance.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.2))
    d = psi_profile["d_mid_um"].values
    y = psi_profile["psi_mean"].values
    ax.plot(d, y, marker="o", markersize=3, linewidth=1.4, label=label)
    ax.axvline(60.0, linestyle="--", linewidth=1.0, color="grey")
    ax.annotate("60 um (expected emergent onset)", xy=(60.0, ax.get_ylim()[1]),
                xytext=(4, -12), textcoords="offset points",
                fontsize=8, color="grey")
    ax.set_xlabel("path distance from soma d [um]")
    ax.set_ylabel("psi = spine area / shaft area [dimensionless]")
    ax.set_title("Spine-area density profile")
    if show_counts:
        ax2 = ax.twinx()
        ax2.bar(d, psi_profile["n_segments"].values,
                width=0.8 * (d[1] - d[0] if len(d) > 1 else 1.0),
                alpha=0.15, color="grey")
        ax2.set_ylabel("segments per bin")
    if label:
        ax.legend(frameon=False, fontsize=9)
    return ax


def plot_phi_scatter(phi_df, ax=None, max_points=20000):
    """Per-segment phi against path distance -- the raw, unbinned view."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.2))
    sub = phi_df
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=0)
    ax.scatter(sub["d_from_um"].values, sub["phi_um"].values,
               s=3, alpha=0.25, edgecolors="none")
    ax.set_xlabel("path distance from soma d [um]")
    ax.set_ylabel("phi [um^2 per um]")
    ax.set_title("Per-segment spine-area density")
    return ax
