from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .local_tools import bmf_tools, filters, mesh_tools, readers


DOMAIN = "__sv_domain__"


def table_format(spec: dict):
    return readers().TableFormat(**spec.get("format", {}))


def header(spec: dict) -> list[str]:
    if Path(spec["path"]).suffix.lower() == ".bmf":
        return list(load_bmf(spec["path"]).columns)
    return readers().read_header(spec["path"], spec.get("sheet") or None, fmt=table_format(spec))


def load_bmf(path: str) -> pd.DataFrame:
    result = bmf_tools().load_bmf_table(path)
    frame = result["dataframe"]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(
            f"BMF variant could not be decoded by the local reader ({result['reader_mode']}). "
            "Export a flagged CSV with centroid and per-block DX/DY/DZ columns from Vulcan."
        )
    if result.get("row_count") != len(frame):
        raise ValueError("BMF reader returned an incomplete table; export to CSV.")
    return frame


def chunks(spec: dict, columns=None, text_columns=(), progress=None):
    if Path(spec["path"]).suffix.lower() == ".bmf":
        frame = load_bmf(spec["path"])
        for start in range(0, len(frame), 100_000):
            yield frame.iloc[start:start + 100_000] if columns is None else frame.iloc[start:start + 100_000][columns]
    else:
        yield from readers().iter_table_chunks(
            spec["path"], usecols=columns, text_cols=list(text_columns),
            sheet=spec.get("sheet") or None, fmt=table_format(spec), chunksize=100_000,
            progress=progress,
        )


def xyz_columns(spec: dict, columns: list[str], sample=False) -> list[str]:
    from .local_tools import local_module

    detect = local_module("srkreliability.flagging", "SRKReliability").detect_sample_xyz if sample else (
        local_module("srkreliability.blockstats", "SRKReliability").detect_xyz_columns
    )
    xyz = spec.get("xyz") or detect(columns)
    if not xyz or len(xyz) != 3 or len(set(xyz)) != 3 or any(c not in columns for c in xyz):
        raise ValueError("Choose three distinct X/Y/Z columns; automatic detection failed.")
    return list(xyz)


def numeric(frame: pd.DataFrame, columns) -> np.ndarray:
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def assign_blocks(samples: pd.DataFrame, xyz: list[str], spec: dict, needed: list[str], log, typed_cols=()):
    from .local_tools import local_module

    cols = header(spec)
    bxyz = xyz_columns(spec, cols)
    size_cols = spec.get("size_cols") or local_module(
        "srkreliability.blockstats", "SRKReliability"
    ).detect_size_columns(cols)
    constant = spec.get("block_size")
    if not size_cols and not constant:
        meta = readers().read_leapfrog_metadata(spec["path"])
        if any(f > 1 for f in meta.get("subblock_factors", [])):
            raise ValueError("Subblocked models require explicit per-row dimension columns.")
        constant = meta.get("parent_block_size")
    if not size_cols and (constant is None or len(constant) != 3):
        raise ValueError("Set DX/DY/DZ columns or an explicit regular block size; sizes are never guessed.")
    if size_cols and (len(size_cols) != 3 or any(c not in cols for c in size_cols)):
        raise ValueError("Three valid block dimension columns are required.")
    if any(c not in cols for c in needed):
        raise ValueError(f"Missing block attributes: {set(needed) - set(cols)}")
    points = numeric(samples, xyz)
    tree = cKDTree(points)
    sample_min, sample_max = points.min(axis=0), points.max(axis=0)
    hits = np.zeros(len(samples), dtype=np.int64)
    values = {c: np.full(len(samples), None, dtype=object) for c in needed}
    usecols = list(dict.fromkeys(bxyz + list(size_cols or []) + needed))
    rows = 0
    for frame in chunks(spec, usecols, text_columns=[c for c in needed if c not in typed_cols],
                        progress=lambda fraction, message: log(f"Reading blocks ({fraction:.0%}): {message}")):
        centers = numeric(frame, bxyz)
        sizes = numeric(frame, size_cols) if size_cols else np.tile(np.asarray(constant, float), (len(frame), 1))
        if not np.isfinite(centers).all() or not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("Block coordinates/dimensions contain invalid values; fix the source.")
        plausible = np.all((centers + sizes / 2 >= sample_min) &
                           (centers - sizes / 2 <= sample_max), axis=1)
        indices = np.flatnonzero(plausible)
        neighbors = tree.query_ball_point(centers[indices], np.linalg.norm(sizes[indices], axis=1) / 2)
        attributes_array = frame[needed].to_numpy(object)
        # Half-open boxes give shared faces deterministic upper-side ownership.
        for block_index, found in zip(indices, neighbors):
            center, size, attributes = centers[block_index], sizes[block_index], attributes_array[block_index]
            candidates = np.asarray(found, dtype=int)
            if not len(candidates):
                continue
            delta = points[candidates] - center
            selected = candidates[np.all((delta >= -size / 2) & (delta < size / 2), axis=1)]
            hits[selected] += 1
            if np.any(hits[selected] > 1):
                raise ValueError("Overlapping blocks contain the same sample. Resolve block overlap before modelling.")
            for column, value in zip(needed, attributes):
                values[column][selected] = value
        rows += len(frame)
        log(f"Flagged against {rows:,} block rows")
    result = samples.copy()
    for col in needed:
        # Infer numeric dtype for expression fields, exactly as the local reader does.
        result["BM:" + col] = pd.Series(values[col]).infer_objects() if col in typed_cols else values[col]
    result[DOMAIN] = result["BM:" + spec["domain"]]
    log(f"Block assignment: {(hits > 0).sum():,} matched; {(hits == 0).sum():,} outside")
    return result


def load_solids(specs: list[dict], repair: str, log) -> list:
    tools = mesh_tools()
    meshes = []
    names = [str(s["name"]).strip() for s in specs]
    if not names or any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError("Solid domain names must be nonempty and unique.")
    for spec in specs:
        log(f"Loading solid {spec['name']}")
        mesh = tools.load_mesh(Path(spec["path"]))
        if mesh.is_watertight and mesh.is_winding_consistent and mesh.volume < 0:
            mesh.invert()
            log(f"{spec['name']}: reversed global inward winding; geometry unchanged")
        if repair == "meshops":
            mesh = tools.heal_mesh(
                mesh, intended_mesh_type="solid", allow_residual_alpha_wrap_fallback=False,
                status_callback=log,
            )
        if mesh.is_empty or not np.isfinite(mesh.vertices).all():
            raise ValueError(f"Invalid mesh: {spec['name']}")
        if repair != "voxel" and not mesh.is_volume:
            raise ValueError(
                f"{spec['name']} is not a consistently oriented closed volume. "
                "Use MeshOps repair or explicitly choose approximate voxel repair."
            )
        meshes.append(mesh)
    return meshes


def coherent_voxels(meshes, pitch: float, max_cells: int, log):
    """One world grid, one owner. Solid order is explicit conflict priority."""
    if not np.isfinite(pitch) or pitch <= 0 or max_cells <= 0:
        raise ValueError("Voxel pitch and cell budget must be positive.")
    lower = np.floor(np.min([m.bounds[0] for m in meshes], axis=0) / pitch) * pitch
    upper = np.max([m.bounds[1] for m in meshes], axis=0)
    shape = np.ceil((upper - lower) / pitch).astype(np.int64) + 3
    if np.prod(shape.astype(object)) > max_cells:
        raise ValueError("Voxel bounding grid exceeds cell budget. Increase pitch or budget.")
    owners = {}
    conflicts = 0
    for index, mesh in enumerate(meshes):
        shifted = mesh.copy()
        shifted.apply_translation(-lower)
        # trimesh voxel centres are multiples of pitch; translating first gives
        # every solid the same lattice even when their bounding boxes differ.
        voxel = shifted.voxelized(pitch).fill()
        keys = np.rint(voxel.points / pitch).astype(np.int64)
        for row in keys:
            key = tuple(row)
            if key in owners:
                conflicts += 1
            else:
                owners[key] = index
        if len(owners) > max_cells:
            raise ValueError("Voxel occupancy exceeds cell budget.")
    log(f"APPROXIMATE voxel repair: {len(owners):,} occupied cells; {conflicts:,} conflicts assigned to first solid")
    return lower, owners, conflicts


def assign_solids(samples, xyz, specs, settings, log):
    mode = settings.get("repair", "none")
    if mode not in ("none", "meshops", "voxel"):
        raise ValueError("Repair must be none, meshops or voxel.")
    meshes = load_solids(specs, mode, log)
    points = numeric(samples, xyz)
    labels = np.full(len(samples), None, dtype=object)
    audit = {}
    if mode == "voxel":
        pitch = float(settings.get("voxel_pitch", 5))
        origin, owners, conflicts = coherent_voxels(meshes, pitch, int(settings.get("max_voxels", 2_000_000)), log)
        keys = np.floor((points - origin) / pitch + 0.5).astype(np.int64)
        for i, key in enumerate(keys):
            owner = owners.get(tuple(key))
            if owner is not None:
                labels[i] = str(specs[owner]["name"])
        audit = {"mode": "approximate_shared_voxels", "pitch": pitch, "conflicts": conflicts,
                 "priority": [s["name"] for s in specs], "occupied_cells": len(owners)}
    else:
        for spec, mesh in zip(specs, meshes):
            inside = mesh.contains(points)
            if np.any(inside & pd.notna(labels)):
                raise ValueError("Solid domains overlap at sample locations; resolve the overlap or use explicit voxel priority.")
            labels[inside] = str(spec["name"])
        audit = {"mode": mode, "overlap_check": "sample locations only"}
    result = samples.copy()
    result[DOMAIN] = labels
    log(f"Solid assignment: {pd.notna(labels).sum():,} matched; {pd.isna(labels).sum():,} outside")
    return result, audit


def prepare(config: dict, log):
    sample_spec = config["samples"]
    log("Inspecting sample header and file format")
    columns = header(sample_spec)
    if DOMAIN in columns or any(c.startswith("BM:") for c in columns):
        raise ValueError("Sample columns use reserved __sv_domain__ / BM: names.")
    xyz = xyz_columns(sample_spec, columns, sample=True)
    value = sample_spec["value"]
    if value not in columns:
        raise ValueError(f"Sample value column not found: {value}")
    drilling_columns = [sample_spec[k] for k in ("hole_id", "from", "to") if sample_spec.get(k)]
    if bool(sample_spec.get("from")) != bool(sample_spec.get("to")):
        raise ValueError("Map both From and To, or leave both blank.")
    if sample_spec.get("from") and not sample_spec.get("hole_id"):
        raise ValueError("Hole ID is required when From/To are mapped.")
    if len(set(drilling_columns)) != len(drilling_columns):
        raise ValueError("Hole ID, From and To must use distinct columns.")
    if any(c not in columns for c in drilling_columns):
        raise ValueError("Mapped drilling columns are missing from the sample file.")
    helper = filters()
    sample_filters = helper.normalize_filters(config.get("filters", []))
    for spec in sample_filters:
        if spec["type"] == "expression":
            spec["expression"] = helper.wrap_bm_refs(spec["expression"])
    # Read only columns required for modelling and filters, preserving categories.
    required = list(dict.fromkeys(xyz + [value] + drilling_columns + [
        f for f in helper.filter_fields(sample_filters, columns) if not f.startswith("BM:")
    ]))
    typed_sample = helper.expression_columns(sample_filters, columns)
    text_columns = [f for f in helper.text_fields(sample_filters)
                    if not f.startswith("BM:") and f not in typed_sample]
    if sample_spec.get("hole_id"):
        text_columns = list(dict.fromkeys(text_columns + [sample_spec["hole_id"]]))
    frames = []
    log("Reading sample data")
    for frame in chunks(sample_spec, required, text_columns=text_columns,
                        progress=lambda fraction, message: log(f"Reading samples ({fraction:.0%}): {message}")):
        frames.append(frame)
        log(f"Read sample chunk: {len(frame):,} rows")
    if not frames:
        raise ValueError("Sample file is empty.")
    samples = pd.concat(frames, ignore_index=True)
    original = len(samples)
    valid = np.isfinite(numeric(samples, xyz + [value])).all(axis=1)
    log(f"Excluded {(~valid).sum():,} samples with missing/nonfinite coordinates or values")
    samples = samples.loc[valid].reset_index(drop=True)
    if samples.empty:
        raise ValueError("No samples have valid coordinates and values.")
    samples[value] = pd.to_numeric(samples[value])
    domains = config["domains"]
    if domains["kind"] == "blocks":
        log("Preparing block model assignment")
        needed = list(dict.fromkeys([domains["domain"]] + helper.bm_fields(sample_filters)))
        expression_fields = helper.expression_columns(sample_filters, ["BM:" + c for c in needed])
        typed_cols = [c for c in needed if "BM:" + c in expression_fields]
        samples = assign_blocks(samples, xyz, domains, needed, log, typed_cols)
        audit = {"mode": "axis_aligned_half_open_blocks"}
    elif domains["kind"] == "solids":
        if helper.bm_fields(sample_filters):
            raise ValueError("BM: filters require a block model, not solid domains.")
        samples, audit = assign_solids(samples, xyz, domains["solids"], domains, log)
    else:
        raise ValueError("Domain kind must be blocks or solids.")
    helper.check_filters(sample_filters, samples)
    log("Applying sample and flagged-block filters")
    mask = helper.filter_mask(samples, sample_filters)
    audit["sample_counts"] = {
        "input": original, "invalid_coordinates_or_values": int((~valid).sum()),
        "outside_domains": int(samples[DOMAIN].isna().sum()), "rejected_by_filters": int((~mask).sum()),
    }
    samples = samples.loc[mask & samples[DOMAIN].notna()].copy()
    samples[DOMAIN] = samples[DOMAIN].map(helper.norm_token)
    samples = samples.loc[samples[DOMAIN] != ""].reset_index(drop=True)
    if samples.empty:
        raise ValueError("No samples remain after filters and domain assignment.")
    if sample_spec.get("hole_id"):
        ids = samples[sample_spec["hole_id"]]
        if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
            raise ValueError("Retained samples have missing Hole ID values.")
        if sample_spec.get("from"):
            intervals = numeric(samples, [sample_spec["from"], sample_spec["to"]])
            if (not np.isfinite(intervals).all() or np.any(intervals[:, 0] < 0)
                    or np.any(intervals[:, 1] <= intervals[:, 0])):
                raise ValueError("Retained samples require finite From/To with 0 <= From < To.")
    audit["sample_counts"]["retained"] = len(samples)
    log(f"Retained {len(samples):,} of {original:,} input samples")
    return samples, xyz, value, audit
