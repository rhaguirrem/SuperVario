"""Transparent starting settings, not a kriging neighbourhood optimisation."""

import copy

import numpy as np
from scipy.spatial import cKDTree

from .lfv import validate
from .variography import AXES, rotation_basis


SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)
EXTENDED_SCALES = (3.0, 4.0)
LEAPFROG_HELP = "https://help.seequent.com/Geo/5.1/en-GB/Content/estimation/estimators.htm"


def leapfrog_parameters(advice):
    constraints = advice["constraints"]
    search = {
        "Minimum samples": constraints["minimum_samples"],
        "Maximum samples": constraints["maximum_samples"],
        "Outlier Restriction": {
            "enabled": False, "Mode": None, "Distance (% of search)": None, "Value threshold": None,
            "status": "Off for baseline. Select Clamp or Discard and thresholds only after domain-specific review; "
                      "this restricts distant high-value inputs, not output values.",
        },
        "Sector Search": {
            "enabled": False, "Mode": "Octant",
            "Maximum samples per sector": constraints["sector_search"]["maximum_samples_per_sector"],
            "Maximum empty sectors": constraints["sector_search"]["maximum_empty_sectors"],
            "status": "Off for baseline. Test Octant with these candidate limits against unrestricted search. "
                      "Quadrant is an alternative. Sector occupancy is not included in support fractions.",
        },
        "Drillhole Limit": {
            "enabled": False, "Maximum samples per drillhole": None,
            "Apply drillhole limit per sector": None,
            "status": "Not inferred without drillhole IDs/composite support. Choose and validate limits and the "
                      "per-sector option before enabling; sample counts are not independent drillhole counts.",
        },
    }
    parameters = {
        "source": LEAPFROG_HELP,
        "status": "Manual-entry guidance, not an importable Leapfrog estimator definition",
        "Interpolant": {
            "Kriging type": "Ordinary", "Mean value": None,
            "Mean value guidance": "Not used for Ordinary. Simple requires a justified stationary domain mean; "
                                   "do not copy 100 or substitute the unvalidated arithmetic sample mean.",
            "Discretisation": {"X": None, "Y": None, "Z": None,
                               "status": "For blocks, determine X/Y/Z from target block dimensions and convergence tests. "
                                         "Use 1/1/1 only for point kriging; evaluation on points uses point kriging. "
                                         "Test e.g. 2/2/2, 3/3/3, 4/4/4; do not copy 5/5/2 without validation."},
            "Variogram Model": advice["variogram_name"],
        },
        "Value Clipping": {
            "Clip input values": False, "Lower bound": None, "Upper bound": None,
            "status": "Off for baseline; set bounds only from domain-specific input-value review. "
                      "Separate from Search > Outlier Restriction.",
        },
        "Search": search,
        "Outputs": {
            "always": ["Value", "Status"],
            "select": ["NS", "MinD", "AvgD", "ND", "KM", "KV", "KE", "SoR", "Sum", "SumN"],
            "status": "Inspect by domain and pass; MinD and AvgD are Cartesian distances. "
                      "Outputs are calculated by Leapfrog, not by this application.",
        },
        "passes": [],
    }
    for p in advice["passes"]:
        axes = p["semi_axes"]
        rotation = p["rotation"]
        parameters["passes"].append({
            "pass": p["pass"], "apply_to": p["apply_to"], "adoption_status": p["adoption_status"],
            "Ellipsoid": {
                "Maximum": axes["major"], "Intermed.": axes["semi_major"], "Minimum": axes["minor"],
                "Dip": rotation["dip"], "Dip Azimuth": rotation["dip_direction"], "Pitch": rotation["pitch"],
                "Variable Orientation": False,
                "units": "Input coordinate units; centre-to-boundary ranges, not diameters",
                "axis_mapping": "Maximum = major, Intermed. = semi-major, Minimum = minor. "
                                "Preserve axis identity and orientation; do not independently sort ranges.",
                "status": "Fixed orientation from the reference variogram structure. Search ellipsoid is distinct "
                          "from the variogram ellipsoid; Set to the variogram only copies its base geometry.",
            },
            "Search": {**copy.deepcopy(search),
                       "Minimum samples": p["minimum_samples"], "Maximum samples": p["maximum_samples"]},
        })
    return parameters


def evidence(source_points, target_points, seed=1729):
    source = np.unique(np.asarray(source_points, float), axis=0)
    target = np.unique(np.asarray(target_points, float), axis=0)
    rng = np.random.default_rng(seed)
    source_count, target_count = len(source), len(target)
    if source_count > 5000:
        source = source[rng.choice(source_count, 5000, replace=False)]
    if target_count > 1000:
        target = target[rng.choice(target_count, 1000, replace=False)]
    return {
        "source_unique_locations": source.tolist(), "probe_locations": target.tolist(),
        "available_unique_sources": source_count, "available_unique_probes": target_count,
        "sampled_sources": len(source) < source_count,
        "method": "Unique-coordinate hard-domain sample probes; approved borrowed samples may supply neighbours; zero-distance locations excluded",
    }


def recommend(result):
    output = _recommend(result)
    output["pass_guidance"] = pass_guidance(output)
    output["leapfrog_parameters"] = leapfrog_parameters(output)
    return output


def pass_guidance(advice):
    passes = advice["passes"]
    if not passes:
        return ["Additional passes cannot be assessed: supported search dimensions and spatial evidence are required. "
                "Recalculate/review the variogram and sample locations before selecting passes."]
    reference = advice["reference"]
    axes = " / ".join(f"{reference['base_semi_axes'][axis]:.6g}" for axis in AXES)
    policy = advice["selection_policy"]
    goal = policy["sample_probe_support_goal"]
    increment = policy["minimum_incremental_support"]
    required = policy["required_other_unique_locations"]
    lines = [
        f"Reference: Structure {reference['structure_index'] + 1} ({reference['rule']}). "
        f"Base semi-axes, major / semi-major / minor: {axes}; capped at observed maximum distance when needed.",
        f"Selection goal: at least {required} other unique sample locations at {goal:.0%} of sample probes. "
        f"Retain a larger candidate only if support increases by at least {increment * 100:g} percentage point(s) "
        "over the last retained pass; stop when the goal is reached.",
    ]
    previous = 0.0
    for p in passes:
        fraction = p["sample_probe_support_fraction"]
        reason = ("Smallest tested search retained as the starting candidate."
                  if p["pass"] == 1 else
                  f"Retained for a {(fraction - previous) * 100:.2f} percentage-point support gain "
                  "over the preceding retained pass; apply only to blocks unestimated by earlier passes.")
        lines.append(f"Pass {p['pass']}: {p['range_multiplier']:g}x reference semi-axes; "
                     f"{fraction:.1%} sample-probe support. {reason}")
        previous = fraction
    if previous >= goal:
        lines.append(
            f"No additional pass is justified by the current sample-probe selection rule: "
            f"Pass {passes[-1]['pass']} reaches the {goal:.0%} goal. "
            "This does not prove target-block coverage; inspect actual unestimated blocks and constraints before "
            "deciding whether any further pass is needed.")
    else:
        lines.append(
            f"The retained passes reach only {previous:.1%} sample-probe support, below the {goal:.0%} goal. "
            "Any tested larger candidates not retained failed the minimum incremental-support rule. "
            "No further pass is justified by the tested evidence; review sparse areas, drilling, boundaries and "
            "model validity rather than force-filling or extending the search indefinitely.")
    if any(p["beyond_base_ranges"] for p in passes):
        lines.append("Searches above 1x exceed the reference base ranges; review extrapolative estimates separately.")
    if any(p["conditional_extension"] for p in passes):
        lines.append("Retained 3x/4x extensions are conditional exploratory tests, requiring target-block evidence "
                     "and geological approval, not approved production passes.")
    if advice.get("orientation_status") == "major_direction_unresolved":
        lines.append("The major direction is unresolved: all pass geometries remain exploratory, "
                     "even when the sample-support goal is met.")
    diagnostics = advice["support_diagnostics"]
    lines.append(f"Evidence: {diagnostics['probes_checked']} sample probes and "
                 f"{diagnostics['sources_checked']} source locations checked. "
                 "When source locations are capped, support counts are lower bounds.")
    lines.append("Support is measured at sample locations, not target blocks, before sector/drillhole limits. "
                 "Validate later passes on actual remaining blocks; preserve hard boundaries, retain pass identity "
                 "and do not interpret pass number as resource classification.")
    return lines


def _recommend(result):
    model = validate(result["model"])
    warnings = [
        "Starting settings only, not an optimised estimator or resource classification.",
        "Sample-location support is not target-block coverage; validate on the actual estimation grid.",
        "Neighbour counts precede sector/drillhole limits and do not establish independent drillhole support.",
        "Search distances below are centre-to-boundary semi-axis lengths in input coordinate units, not full diameters.",
    ]
    output = {
        "status": "provisional_requires_target_block_validation",
        "primary_estimator": "ordinary_kriging",
        "variogram_name": model["metadata"]["name"],
        "warnings": warnings,
        "passes": [],
        "validation": [
            "Run search neighbourhood sensitivity tests on actual target blocks; inspect unestimated volume by domain and pass.",
            "Use spatially separated or leave-one-drillhole-out validation, not random interval splits; check bias and conditional bias.",
            "Compare OK, IDW and nearest-neighbour global means, distributions and swath plots at comparable support.",
            "Inspect kriging variance (KV), kriging efficiency (KE), sum of negative weights (SumN), NS, MinD, AvgD and Status.",
            "Test block discretisation convergence (for example 2x2x2, 3x3x3 and 4x4x4); block dimensions/support are needed before choosing.",
            "Validate sector occupancy and drillhole caps with actual hole IDs and sampling/composite lengths.",
            "Apply later passes only to blocks unestimated in earlier passes; retain pass identity and review late-pass estimates separately.",
            "Keep geological hard boundaries unless sharing has been explicitly approved; do not infer classification from pass number.",
        ],
        "comparisons": {
            "idw": {"exponent_start": 2, "exponents_to_test": [1, 2, 3],
                    "search": "Compare on the same supported neighbourhoods and boundary policy as OK; tune exponent through validation.",
                    "anisotropic_distance": "Check separately: changing the IDW ellipsoid changes weights as well as the search."},
            "nearest_neighbour": {"purpose": "Baseline for global/local bias and spatial patterns, not an automatic replacement for OK.",
                                  "search": "Compare isotropic versus geological anisotropic nearest selection; match boundaries and estimated extent."},
        },
        "constraints": {
            "minimum_samples": 8, "maximum_samples": 24,
            "sector_search": {"candidate": "Octant", "maximum_samples_per_sector": 3,
                              "maximum_empty_sectors": 4,
                              "status": "Sensitivity-test candidate only; occupancy is not included in the support calculation."},
            "drillhole_limit": {"value": None, "status": "Not inferred without drillhole IDs and sample-support information."},
            "clipping_and_outlier_restriction": "No automatic thresholds; select from geological and grade-distribution review.",
            "discretisation": {"value": None, "status": "Not inferred without target block dimensions and convergence tests; 1x1x1 is point kriging."},
        },
        "selection_policy": {"candidate_scales": list(SCALES), "required_other_unique_locations": 8,
                             "sample_probe_support_goal": 0.9, "minimum_incremental_support": 0.01,
                             "default_maximum_candidates": 5,
                             "conditional_extended_scales": list(EXTENDED_SCALES),
                             "maximum_candidates_with_conditional_extensions": 7},
    }
    orientation_status = result.get("orientation_diagnostics", {}).get("status", "not_assessed")
    output["orientation_status"] = orientation_status
    if orientation_status == "major_direction_unresolved":
        output["status"] = "exploratory_only_major_direction_unresolved"
        warnings.append("Search ellipsoids use an unresolved major direction; do not adopt them for production estimation without resolving the orientation.")
    if model["nugget"] / model["sill"] >= 0.5:
        warnings.append("Nugget is at least half of total sill: weak short-range continuity; compare OK carefully with IDW/NN and review support.")
    warnings.extend(result.get("fit", {}).get("warnings", []))
    usable = [(i, s) for i, s in enumerate(model["structures"])
              if s["structure_type"] == "spherical" and s["contribution"] > 0]
    if not usable or any(s["structure_type"] != "spherical" and s["contribution"] > 0 for s in model["structures"]):
        output["status"] = "search_ranges_not_inferred"
        warnings.append("Finite spherical ranges are required; no search dimensions inferred from linear/spheroidal models.")
        return output
    index, structure = max(usable, key=lambda item: np.mean(
        np.log([item[1]["anisotropy"]["ranges"][axis] for axis in AXES])))
    ranges = np.array([structure["anisotropy"]["ranges"][axis] for axis in AXES])
    observed = result.get("max_distance")
    if observed is not None:
        if np.any(ranges > observed):
            warnings.append("Reference ranges exceed observed distance: base semi-axes capped at observed max distance before pass scaling.")
        ranges = np.minimum(ranges, observed)
    rotation = structure["anisotropy"]["rotation"]
    output["reference"] = {
        "structure_index": index, "rule": "Positive-contribution spherical structure with largest geometric-mean range",
        "base_semi_axes": dict(zip(AXES, map(float, ranges))), "rotation": copy.deepcopy(rotation),
    }
    support = result.get("neighbourhood_evidence")
    if not support:
        output["status"] = "support_evidence_unavailable"
        warnings.append("No saved spatial evidence: recalculate from samples before recommending pass count.")
        return output
    source = np.asarray(support["source_unique_locations"], float)
    probes = np.asarray(support["probe_locations"], float)
    if source.ndim != 2 or probes.ndim != 2 or source.shape[1] != 3 or probes.shape[1] != 3:
        raise ValueError("Invalid saved neighbourhood coordinates.")
    if not len(source) or not len(probes) or not np.isfinite(source).all() or not np.isfinite(probes).all():
        raise ValueError("Neighbourhood coordinates must be nonempty and finite.")
    origin = source[0]
    basis = rotation_basis(rotation)
    transformed_source = ((source - origin) @ basis) / ranges
    transformed_probe = ((probes - origin) @ basis) / ranges
    tree = cKDTree(transformed_source)
    # If a probe's own location is in the capped source set, exclude it.
    own = tree.query_ball_point(transformed_probe, 0.0, return_length=True)
    diagnostics = []
    previous = None
    for scale in SCALES + EXTENDED_SCALES:
        if scale in EXTENDED_SCALES and previous is not None and previous >= 0.9:
            break
        counts = tree.query_ball_point(transformed_probe, scale, return_length=True) - own
        fraction = float(np.mean(counts >= 8))
        increment = fraction if previous is None else fraction - previous
        diagnostics.append({
            "scale": scale, "fraction_probes_with_at_least_8_other_unique_locations": fraction,
            "increment_from_previous_candidate": increment,
            "neighbour_count_p10_p50_p90": np.percentile(counts, [10, 50, 90]).tolist(),
            "conditional_extension": scale in EXTENDED_SCALES,
        })
        previous = fraction
    output["support_diagnostics"] = {
        "method": support["method"], "probes_checked": len(probes), "sources_checked": len(source),
        "available_unique_sources": support["available_unique_sources"],
        "available_unique_probes": support["available_unique_probes"],
        "candidate_support": diagnostics,
    }
    if support["sampled_sources"]:
        warnings.append("Source locations were capped at 5000; support fractions are lower-bound counts, not full-data coverage.")
    selected = []
    last_fraction = 0.0
    for diagnostic in diagnostics:
        fraction = diagnostic["fraction_probes_with_at_least_8_other_unique_locations"]
        if not selected or fraction - last_fraction >= 0.01:
            selected.append(diagnostic)
            last_fraction = fraction
        if last_fraction >= 0.9:
            break
    for number, diagnostic in enumerate(selected, 1):
        scale = diagnostic["scale"]
        output["passes"].append({
            "pass": number, "range_multiplier": scale,
            "semi_axes": dict(zip(AXES, map(float, ranges * scale))),
            "full_axis_lengths": dict(zip(AXES, map(float, ranges * scale * 2))),
            "rotation": copy.deepcopy(rotation), "minimum_samples": 8, "maximum_samples": 24,
            "sample_probe_support_fraction": diagnostic["fraction_probes_with_at_least_8_other_unique_locations"],
            "apply_to": "All eligible domain blocks" if number == 1 else "Only blocks unestimated by earlier passes",
            "beyond_base_ranges": scale > 1,
            "conditional_extension": diagnostic["conditional_extension"],
            "adoption_status": ("Exploratory only; requires target-block evidence and geological approval before use"
                                if diagnostic["conditional_extension"] else "Starting candidate; requires validation"),
        })
    output["recommended_pass_count"] = len(selected)
    output["default_pass_count"] = sum(not p["conditional_extension"] for p in output["passes"])
    output["conditional_additional_pass_count"] = sum(p["conditional_extension"] for p in output["passes"])
    if output["conditional_additional_pass_count"]:
        warnings.append(
            "Sparse support prompted conditional 3x/4x search candidates beyond the default five. "
            "These are sensitivity-test proposals only: target-block evidence and geological approval are required; "
            "no domain expansion or sample sharing is authorised.")
    if last_fraction < 0.9:
        warnings.append(
            "Candidate search did not reach 90% sample-probe support. Do not force-fill: review sparse areas, "
            "drilling, domains and model validity rather than extending the search indefinitely.")
    if any(p["beyond_base_ranges"] for p in output["passes"]):
        warnings.append("Later ellipsoids exceed base continuity ranges; flag extrapolative estimates separately.")
    return output


def refresh(result):
    result["estimator_recommendations"] = recommend(result)


def format_recommendations(result):
    advice = result.get("estimator_recommendations")
    if advice is None:
        return "Estimator recommendations unavailable; recalculate this domain."
    parameters = leapfrog_parameters(advice)
    interpolant = parameters["Interpolant"]
    lines = [f"Ordinary kriging candidate | Status: {advice['status']}",
             f"Candidate passes: {advice.get('default_pass_count', 0)} default + "
             f"{advice.get('conditional_additional_pass_count', 0)} conditional exploratory",
             "Leapfrog Edge manual-entry settings (not an estimator import file).",
             "Semi-axes are centre-to-boundary distances, NOT diameters. Units match coordinates.",
             "", "INTERPOLANT",
             f"Kriging type: {interpolant['Kriging type']}",
             f"Mean value: not applicable. {interpolant['Mean value guidance']}",
             f"Discretisation X / Y / Z: not determined. {interpolant['Discretisation']['status']}",
             f"Variogram Model: {interpolant['Variogram Model']}",
             "", "VALUE CLIPPING",
             "Clip input values: Off. Lower bound / Upper bound: not selected.",
             parameters["Value Clipping"]["status"], "", "PASS SELECTION AND ADDITIONAL-PASS GUIDANCE",
             *pass_guidance(advice), "", "ELLIPSOID AND SEARCH BY PASS"]
    if not advice["passes"]:
        lines.append("No supported pass dimensions available; recalculate/review the variogram and spatial evidence.")
    for p, entry in zip(advice["passes"], parameters["passes"]):
        ellipsoid = entry["Ellipsoid"]
        search = entry["Search"]
        dims = " / ".join(f"{ellipsoid[a]:.6g}" for a in ("Maximum", "Intermed.", "Minimum"))
        label = " CONDITIONAL EXPLORATORY" if p["conditional_extension"] else ""
        lines += [f"Pass {p['pass']}{label}: {entry['apply_to']}",
                  f"  Ellipsoid ranges - Maximum / Intermed. / Minimum: {dims}",
                  f"  Directions - Dip: {ellipsoid['Dip']:.6g}; Dip Azimuth: {ellipsoid['Dip Azimuth']:.6g}; "
                  f"Pitch: {ellipsoid['Pitch']:.6g}",
                  "  Variable Orientation: Off (fixed fitted orientation).",
                  f"  Search - Minimum samples: {search['Minimum samples']}; Maximum samples: {search['Maximum samples']}",
                  f"  Sample-probe support before sector/drillhole limits: {p['sample_probe_support_fraction']:.1%}",
                  f"  Adoption: {entry['adoption_status']}"]
    lines += ["", "SEARCH OPTIONS (all passes)",
              "Outlier Restriction: Off. Clamp / Discard, Distance (% of search), Value threshold: not selected.",
              parameters["Search"]["Outlier Restriction"]["status"],
              "Sector Search: Off for baseline. Sensitivity test: Octant; Maximum samples per sector: "
              f"{parameters['Search']['Sector Search']['Maximum samples per sector']}; Maximum empty sectors: "
              f"{parameters['Search']['Sector Search']['Maximum empty sectors']}.",
              parameters["Search"]["Sector Search"]["status"],
              "Drillhole Limit: Off pending IDs/support review. Maximum samples per drillhole: not determined. "
              "Apply drillhole limit per sector: not determined.",
              parameters["Search"]["Drillhole Limit"]["status"],
              "", "OUTPUTS",
              "Always included: " + ", ".join(parameters["Outputs"]["always"]),
              "Select for validation: " + ", ".join(parameters["Outputs"]["select"]),
              parameters["Outputs"]["status"],
              "", "ELLIPSOID MAPPING",
              "Maximum = major; Intermed. = semi-major; Minimum = minor. Do not sort these values independently.",
              "Search ellipsoids differ from the variogram ellipsoid; each pass uses its scaled ranges.",
              "", "Comparison: IDW exponent 2 (test 1, 2, 3); nearest neighbour as a bias baseline.",
              "Leapfrog field reference: " + LEAPFROG_HELP,
              "", "WARNINGS:"] + advice["warnings"] + ["", "VALIDATE:"] + advice["validation"]
    return "\n".join(lines)
