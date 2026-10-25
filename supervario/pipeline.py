import copy
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp

from . import __version__, lfv
from .data import DOMAIN, numeric, prepare
from .variography import analyse, validate_settings
from .recommendations import evidence, refresh
from .workbook import recommendation_workbook


def default_config() -> dict:
    return {
        "samples": {"path": "", "xyz": [], "value": "", "sheet": "",
                    "format": {"sep": "", "decimal": "", "header_line": 0}},
        "domains": {"kind": "blocks", "path": "", "domain": "", "xyz": [], "size_cols": [],
                    "sheet": "", "format": {"sep": "", "decimal": "", "header_line": 0},
                    "solids": [], "repair": "none", "voxel_pitch": 5.0, "max_voxels": 2_000_000},
        "filters": [],
        "variogram": {"bins": 12, "lag": 0, "max_distance": 0, "angle_tolerance": 30, "in_plane_step": 15,
                      "azimuth_step": 15, "plunge_step": 30,
                      "min_pairs": 10, "max_samples": 5000, "max_pairs": 300_000,
                      "structures": 1, "seed": 1729, "rotation": None, "trend_file": ""},
        "boundaries": {"distance": 25.0, "min_samples": 20, "approved_pairs": []},
        "output": "output",
    }


def input_paths(config, include_projects=True):
    paths = [config["samples"]["path"]]
    domains = config["domains"]
    if domains["kind"] == "blocks":
        paths.append(domains["path"])
    else:
        paths.extend(s["path"] for s in domains["solids"])
    if config.get("variogram", {}).get("trend_file"):
        paths.append(config["variogram"]["trend_file"])
    return paths + (list(config.get("_protected_inputs", [])) if include_projects else [])


def load_config(path):
    import json

    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)
    # Resolve paths relative to the configuration, not the shell working directory.
    for spec in [config["samples"], config["domains"]] + config["domains"].get("solids", []):
        if spec.get("path"):
            spec["path"] = str((path.parent / spec["path"]).resolve())
    config["output"] = str((path.parent / config.get("output", "output")).resolve())
    if config.get("variogram", {}).get("trend_file"):
        config["variogram"]["trend_file"] = str((path.parent / config["variogram"]["trend_file"]).resolve())
    config["_protected_inputs"] = [str(path)]
    return config


def contacts(samples, xyz, value, settings):
    radius = float(settings.get("distance", 25))
    minimum = int(settings.get("min_samples", 20))
    if not np.isfinite(radius) or radius <= 0 or minimum < 2:
        raise ValueError("Contact distance must be positive; min_samples must be at least 2.")
    groups = {str(k): g for k, g in samples.groupby(DOMAIN, sort=True)}
    result = []
    names = list(groups)
    for i, a in enumerate(names):
        ga = groups[a]
        pa = numeric(ga, xyz)
        va = ga[value].to_numpy(float)
        for b in names[i + 1:]:
            gb = groups[b]
            pb = numeric(gb, xyz)
            vb = gb[value].to_numpy(float)
            da = cKDTree(pb).query(pa, k=1)[0]
            db = cKDTree(pa).query(pb, k=1)[0]
            aa, bb = va[da <= radius], vb[db <= radius]
            if not len(aa) or not len(bb):
                continue
            pooled = np.sqrt((np.var(aa) + np.var(bb)) / 2)
            difference = abs(float(np.mean(aa) - np.mean(bb)))
            effect = difference / pooled if pooled > 0 else (0.0 if difference == 0 else None)
            ks = float(ks_2samp(aa, bb).statistic)
            recommend = min(len(aa), len(bb)) >= minimum and effect is not None and effect <= 0.25 and ks <= 0.2
            bins = []
            for sign, d, v in ((-1, da, va), (1, db, vb)):
                for lo in np.linspace(0, radius, 6)[:-1]:
                    take = (d > lo if lo else d >= 0) & (d <= lo + radius / 5)
                    bins.append({"distance": float(sign * (lo + radius / 10)), "count": int(take.sum()),
                                 "mean": float(np.mean(v[take])) if take.any() else None})
            result.append({
                "a": a, "b": b, "n_a": len(aa), "n_b": len(bb), "standardised_mean_difference": effect,
                "ks_statistic": ks, "recommend_review": bool(recommend), "distance": radius,
                "profile": bins,
                "method": "Nearest opposite-domain sample distance proxy; not geometric boundary distance or proof of equivalence",
            })
    return result


def run(config: dict, log=print):
    config = copy.deepcopy(config)
    validate_settings(config.get("variogram", {}))
    samples, xyz, value, domain_audit = prepare(config, log)
    boundary_settings = config.get("boundaries", {})
    log("Analysing cross-domain contact sample support")
    contact_report = contacts(samples, xyz, value, boundary_settings)
    radius = float(boundary_settings.get("distance", 25))
    approvals = set()
    names = set(samples[DOMAIN])
    for pair in boundary_settings.get("approved_pairs", []):
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair).issubset(names):
            raise ValueError(f"Invalid approved soft-boundary domain pair: {pair}")
        approvals.add(frozenset(pair))
    results, failures = [], []
    for name, group in samples.groupby(DOMAIN, sort=True):
        log(f"Building variogram for {name}")
        source = group
        borrowed = []
        tree = cKDTree(numeric(group, xyz))
        for pair in approvals:
            if name in pair:
                neighbour = next(iter(pair - {name}))
                other = samples.loc[samples[DOMAIN] == neighbour]
                selected = tree.query(numeric(other, xyz))[0] <= radius
                if selected.any():
                    borrowed.append(other.loc[selected])
        if borrowed:
            source = pd.concat([group] + borrowed, ignore_index=True)
        try:
            drilling = None
            sample_spec = config["samples"]
            if sample_spec.get("hole_id"):
                drilling = {"hole_id": source[sample_spec["hole_id"]].astype(str).str.strip().to_numpy()}
                if sample_spec.get("from"):
                    drilling["intervals"] = numeric(source, [sample_spec["from"], sample_spec["to"]])
            result = analyse(numeric(source, xyz), source[value].to_numpy(float), value, name,
                             config.get("variogram", {}), log, drilling=drilling)
        except ValueError as exc:
            # A failed domain remains an explicit failed domain, never an empty LFV.
            failures.append({"domain": str(name), "error": str(exc)})
            log(f"FAILED {name}: {exc}")
            continue
        result["hard_domain_samples"] = len(group)
        result["borrowed_samples"] = len(source) - len(group)
        if borrowed:
            result["fit"]["warnings"].append("Approved soft-boundary sharing uses distance to target-domain samples, not surfaces.")
        log(f"Assessing estimator search neighbourhoods for {name}")
        result["neighbourhood_evidence"] = evidence(numeric(source, xyz), numeric(group, xyz))
        refresh(result)
        results.append(result)
    report = {
        "application": "SuperVario", "version": __version__, "config": config,
        "domain_assignment": domain_audit, "filtered_samples": len(samples),
        "contacts": contact_report, "results": results, "failures": failures,
        "scientific_notes": [
            "Candidate models require geological review and validation in Leapfrog Edge.",
            "Coordinates must share one Cartesian coordinate system and linear unit.",
            "Classical Matheron semivariance; population variance (ddof=0); no implicit compositing, detrending or declustering.",
            "Contact recommendations are descriptive heuristics, not statistical evidence of boundary continuity.",
            "Pairs are unordered, bidirectional cones; lag bins are [k*lag, (k+1)*lag), plotted at mean pair distance.",
        ],
    }
    if not results:
        details = "; ".join(f"{f['domain']}: {f['error']}" for f in failures)
        raise ValueError("No domain produced a model. " + details)
    return report


def result_filename(domain):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", domain).strip("._")[:65] or "domain"
    return f"variogram_{stem}.lfv"


def result_filenames(domains):
    names = [result_filename(domain) for domain in domains]
    reserved = {name.casefold() for name in names}
    used = set()
    allocated = {}
    for index in sorted(range(len(domains)), key=lambda i: (domains[i].casefold(), domains[i])):
        name = names[index]
        if name.casefold() in used:
            stem = Path(name).stem
            suffix = 2
            while (f"{stem}_{suffix}.lfv".casefold() in reserved or
                   f"{stem}_{suffix}.lfv".casefold() in used):
                suffix += 1
            name = f"{stem}_{suffix}.lfv"
        used.add(name.casefold())
        allocated[index] = name
    return [allocated[i] for i in range(len(domains))]


def export_report(report, directory, overwrite=False):
    import json

    report = copy.deepcopy(report)
    for result in report["results"]:
        refresh(result)
    json.dumps(report, allow_nan=False)
    directory = Path(directory).resolve()
    inputs = input_paths(report["config"])
    names = result_filenames([r["domain"] for r in report["results"]])
    targets = [(directory / name, r["model"]) for name, r in zip(names, report["results"])]
    report_path = directory / "analysis.json"
    advice_path = directory / "estimator_recommendations.json"
    workbook_path = directory / "estimator_recommendations.xlsx"
    # Preflight the whole set before changing any destination.
    for path, model in targets:
        lfv.protect_output(path, inputs)
        lfv.validate(model)
    for path in [p for p, _ in targets] + [workbook_path, advice_path, report_path]:
        lfv.protect_output(path, inputs)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Choose another folder or explicitly allow overwrite.")
    workbook_data = recommendation_workbook(report["results"])
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".xlsx")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(workbook_data)
        os.replace(temporary, workbook_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    for path, model in targets:
        lfv.save(path, model, inputs)
    lfv.write_json(advice_path, {
        "application": report["application"], "version": report["version"],
        "note": "Advisory settings, not part of the LFV schema; target-block validation required.",
        "domains": [{"domain": r["domain"], **r["estimator_recommendations"]} for r in report["results"]],
    }, inputs)
    lfv.write_json(report_path, report, inputs)
    return [str(p) for p, _ in targets] + [str(workbook_path), str(advice_path), str(report_path)]
