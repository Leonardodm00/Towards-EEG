"""The cell pool of S3 and each cell's acquisition record -- Block 16.1 in
specs/SPEC.md (D-065, D-085).

  fetch_specimen_details   the rows of Allen's ApiCellTypesSpecimenDetail, paged
  pool_records             the human cells with a reconstruction, as canonical records
  acquisition_record       one cell's plane spacing, pixel size and planes (Allen API)
  write_table, read_table  the CSV files of the pool, the acquisition check and the selection

The raw field names are the ones the Allen SDK reads from this model
(allensdk 2.16.2, api/queries/cell_types_api.py: filter_cells_api and
simplify_cells_api, read 2026-10-26); the live API is not reachable from the
sandbox, so they are not checked against it. Each canonical field is read
from the first present of its candidate keys, the raw API name first, then
the SDK's simplified name, so a CSV exported with the SDK reads the same way.
A required field absent from a row raises ValueError naming the keys the row
holds.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import csv
import math
import os
import warnings

import allen_image_io as aio

# canonical name, candidate keys (raw API name first), required, type
FIELDS = (
    ("specimen_id", ("specimen__id", "id", "specimen_id"), True, "int"),
    ("species", ("donor__species", "species"), True, "str"),
    ("layer", ("structure__layer", "structure_layer_name", "layer"), True, "str"),
    ("dendrite_type", ("tag__dendrite_type", "dendrite_type"), True, "str"),
    ("reconstruction_type", ("nr__reconstruction_type", "reconstruction_type"), True, "str"),
    ("area", ("structure_parent__acronym", "structure_area_abbrev", "area"), False, "str"),
    ("donor_id", ("donor__id", "donor_id"), False, "str"),
    ("hemisphere", ("specimen__hemisphere", "structure_hemisphere", "hemisphere"), False, "str"),
    ("normalized_depth", ("csl__normalized_depth", "normalized_depth"), False, "float"),
)
POOL_TYPES = {name: kind for name, _keys, _req, kind in FIELDS}
POOL_COLUMNS = tuple(POOL_TYPES)
ACQUISITION_TYPES = {"specimen_id": "int", "dz_um": "float", "scale_x": "float", "scale_y": "float",
                     "pixel_um": "float", "pixel_values": "str", "n_planes": "int", "n_planes_failed": "int",
                     "width_px": "float", "height_px": "float", "error": "str"}
ACQUISITION_COLUMNS = tuple(ACQUISITION_TYPES)
SELECTION_TYPES = dict(POOL_TYPES, neuron_class="str", group="str", rank="int")
SELECTION_COLUMNS = tuple(SELECTION_TYPES)
SPECIMEN_DETAIL_MODEL = "model::ApiCellTypesSpecimenDetail"


def _absent(v):
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == "" or v.strip().lower() in ("none", "null", "nan")
    return isinstance(v, float) and math.isnan(v)


def _convert(v, kind):
    """A value in the field's type; absent values are None (int, str) or NaN (float)."""
    if _absent(v):
        return float("nan") if kind == "float" else None
    if kind == "int":
        f = float(v)
        if not (math.isfinite(f) and f == int(f)):
            raise ValueError("not an integer: %r" % (v,))
        return int(f)
    if kind == "float":
        return float(v)
    return str(v).strip()


def normalize_row(row):
    """One API (or SDK, or CSV) row as a canonical record (FIELDS)."""
    rec = {}
    for name, keys, required, kind in FIELDS:
        key = next((k for k in keys if k in row), None)
        if key is None and required:
            raise ValueError("pool row without %s (looked for %s); the row holds: %s"
                             % (name, ", ".join(keys), ", ".join(sorted(str(k) for k in row))))
        rec[name] = _convert(row[key] if key is not None else None, kind)
    return rec


def fetch_specimen_details(api=None, page_rows=10000, max_pages=50):
    """Every row of ApiCellTypesSpecimenDetail (both species), paged: page p asks
    for page_rows rows from row p * page_rows, until a page comes back short.
    The Allen SDK reads this model in one query (num_rows 'all',
    CellTypesApi.list_cells_api); api_query takes a number, so page_rows is
    large enough for one page, and a second page raises a RuntimeWarning: the
    query names no order, and pages of an unordered query may skip or repeat
    rows (a repeated id is caught by pool_records, a skipped one is not).
    api: a callable like allen_image_io.api_query (criteria, num_rows, start_row)."""
    api = aio.api_query if api is None else api
    rows = []
    for page in range(int(max_pages)):
        msg = api(SPECIMEN_DETAIL_MODEL, num_rows=int(page_rows), start_row=page * int(page_rows))
        rows.extend(msg)
        if len(msg) < int(page_rows):
            if page > 0:
                warnings.warn("fetch_specimen_details: %d rows came in %d pages of %d without an order; rows may "
                              "have been skipped -- raise survey.pool_page_rows above %d"
                              % (len(rows), page + 1, int(page_rows), len(rows)), RuntimeWarning, stacklevel=2)
            return rows
    raise RuntimeError("fetch_specimen_details: more than %d pages of %d rows; raise max_pages"
                       % (max_pages, page_rows))


def pool_records(rows, species="Homo Sapiens", require_reconstruction=True):
    """The canonical records of the rows whose species equals `species`
    (case-insensitive) and, with require_reconstruction, whose reconstruction
    type is present -- the filter of the Allen SDK's filter_cells_api; one
    record per specimen (the first row of a repeated id), sorted by
    specimen_id."""
    want = str(species).strip().lower()
    out = {}
    for row in rows:
        rec = normalize_row(row)
        if rec["species"] is None or rec["species"].lower() != want:
            continue
        if require_reconstruction and rec["reconstruction_type"] is None:
            continue
        if rec["specimen_id"] is None:
            raise ValueError("a %s row with a reconstruction has no specimen id: %r" % (species, row))
        out.setdefault(rec["specimen_id"], rec)
    return [out[k] for k in sorted(out)]


def _float(v):
    try:
        return float("nan") if _absent(v) else float(v)
    except (TypeError, ValueError):
        return float("nan")


def acquisition_record(specimen_id):
    """One cell's acquisition record (Block 16.1 (c)) from allen_image_io:
    reconstruction_info (scale_factor_x, _y, _z; the plane spacing dz is
    scale_factor_z) and list_images + plane_table (the usable Primary planes,
    their resolution in um/px at downsample 0, width and height). A query that
    fails is recorded in `error` -- the survey goes on to the next cell."""
    nan = float("nan")
    rec = dict(specimen_id=int(specimen_id), dz_um=nan, scale_x=nan, scale_y=nan, pixel_um=nan, pixel_values="",
               n_planes=0, n_planes_failed=0, width_px=nan, height_px=nan, error="")
    errors = []
    try:
        info = aio.reconstruction_info(int(specimen_id))
        if not info:
            errors.append("reconstruction: no NeuronReconstruction row")
        else:
            rec["dz_um"] = _float(info.get("scale_factor_z"))
            rec["scale_x"] = _float(info.get("scale_factor_x"))
            rec["scale_y"] = _float(info.get("scale_factor_y"))
    except (RuntimeError, KeyError, ValueError, TypeError) as exc:   # api_query wraps network errors in RuntimeError
        errors.append("reconstruction: %s" % exc)
    try:
        df = aio.list_images(int(specimen_id))
        prim = df[df["image_type"] == "Primary"]
        if "failed" in prim.columns:
            rec["n_planes_failed"] = int(prim["failed"].fillna(False).astype(bool).sum())
        planes = aio.plane_table(df)
        res = sorted({round(float(x), 6) for x in planes["resolution"] if math.isfinite(_float(x))})
        rec["pixel_values"] = ";".join("%g" % x for x in res)
        rec["pixel_um"] = res[0] if len(res) == 1 else nan
        rec["n_planes"] = int(len(planes))
        rec["width_px"] = float(planes["width"].median()) if "width" in planes.columns else nan
        rec["height_px"] = float(planes["height"].median()) if "height" in planes.columns else nan
    except (RuntimeError, KeyError, ValueError, TypeError) as exc:   # no SubImage rows, no usable plane, a column missing
        errors.append("images: %s" % exc)
    rec["error"] = "; ".join(errors)
    return rec


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "nan" if math.isnan(v) else repr(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def write_table(path, rows, columns):
    """A CSV with exactly `columns` (a row with another key raises ValueError),
    written to path + '.tmp' then moved into place, so an interrupted run never
    leaves half a file. None is written empty, NaN as 'nan'."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), restval="", extrasaction="raise", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: _cell(v) for k, v in row.items()})
    os.replace(tmp, path)


def read_table(path, types):
    """The rows of a CSV written by write_table, each value parsed by its
    column's type in `types` (int, float, str; a column not in `types` stays a
    string); an empty cell is None, or NaN for a float column."""
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: _convert(v, types.get(k, "str")) for k, v in row.items()} for row in csv.DictReader(f)]
