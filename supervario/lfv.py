import copy
import json
import math
import os
import tempfile
from pathlib import Path


def validate(model: dict) -> dict:
    if model.get("content_type") != "com.seequent.elements.variogram":
        raise ValueError("Not a Seequent variogram LFV.")
    if str(model.get("content_version")) != "2":
        raise ValueError("Only reference LFV content_version 2 is supported.")

    def number(value, label, positive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be numeric.")
        if not math.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(f"Invalid {label}: {value}")

    for field in ("nugget", "sill", "data_variance"):
        number(model[field], field, positive=field != "nugget")
    structures = model["structures"]
    if not 1 <= len(structures) <= 2:
        raise ValueError("Use one or two structures for Leapfrog Edge.")
    for structure in structures:
        if structure["structure_type"] not in ("spherical", "linear", "spheroidal"):
            raise ValueError(f"Unsupported LFV structure: {structure['structure_type']}")
        number(structure["contribution"], "contribution")
        anisotropy = structure["anisotropy"]
        for key in ("major", "semi_major", "minor"):
            number(anisotropy["ranges"][key], key, positive=True)
        rotation = anisotropy["rotation"]
        for key, limit in (("dip_direction", 360), ("dip", 90), ("pitch", 180)):
            number(rotation[key], key)
            if rotation[key] > limit:
                raise ValueError(f"{key} exceeds {limit}.")
    total = model["nugget"] + sum(s["contribution"] for s in structures)
    if not math.isclose(total, model["sill"], rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("Total sill must equal nugget plus structure contributions.")
    return model


def load(path: str | Path) -> dict:
    with open(path, encoding="utf-8-sig") as handle:
        return validate(json.load(handle))


def protect_output(path: str | Path, inputs=()) -> Path:
    path = Path(path).resolve()
    for source in inputs:
        if source and Path(source).resolve() == path:
            raise ValueError(f"Refusing to overwrite input: {path}")
        if source and path.exists() and Path(source).exists() and os.path.samefile(path, source):
            raise ValueError(f"Output aliases an input: {path}")
    return path


def write_json(path: str | Path, data, inputs=()):
    path = protect_output(path, inputs)
    payload = json.dumps(data, indent=2, allow_nan=False, ensure_ascii=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save(path: str | Path, model: dict, inputs=()):
    write_json(path, validate(copy.deepcopy(model)), inputs)
