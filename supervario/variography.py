from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .lfv import validate


AXES = ("major", "semi_major", "minor")


@dataclass
class Pairs:
    delta: np.ndarray
    distance: np.ndarray
    gamma: np.ndarray
    possible: int
    drawn: int
    duplicates: int


def make_pairs(points, values, max_distance: float, budget: int, seed: int) -> Pairs:
    n = len(points)
    total = n * (n - 1) // 2
    if n < 2 or budget < 1 or not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("Pair generation needs samples, a positive distance and pair budget.")
    drawn = min(total, budget)
    if total <= budget:
        left, right = np.triu_indices(n, 1)
    else:
        rng = np.random.default_rng(seed)
        # Uniformly sample unordered pairs without allocating the full N*N matrix.
        ranks = rng.choice(total, size=drawn, replace=False)
        offsets = np.arange(n, dtype=np.int64)
        offsets = offsets * (2 * n - offsets - 1) // 2
        left = np.searchsorted(offsets, ranks, side="right") - 1
        right = left + 1 + ranks - offsets[left]
    delta = points[right] - points[left]
    distance = np.linalg.norm(delta, axis=1)
    duplicates = int(np.count_nonzero(distance == 0))
    valid = (distance > 0) & (distance <= max_distance)
    return Pairs(delta[valid], distance[valid],
                 0.5 * np.square(values[right[valid]] - values[left[valid]]), total, drawn, duplicates)


def direction(azimuth: float, plunge: float) -> np.ndarray:
    az, pl = np.deg2rad([azimuth, plunge])
    return np.array([np.sin(az) * np.cos(pl), np.cos(az) * np.cos(pl), -np.sin(pl)])


def rotation_basis(rotation: dict) -> np.ndarray:
    az, dip, pitch = np.deg2rad([rotation[k] for k in ("dip_direction", "dip", "pitch")])
    strike = np.array([-np.cos(az), np.sin(az), 0])
    down = np.array([np.sin(az) * np.cos(dip), np.cos(az) * np.cos(dip), -np.sin(dip)])
    major = np.cos(pitch) * strike + np.sin(pitch) * down
    semi = -np.sin(pitch) * strike + np.cos(pitch) * down
    return np.column_stack((major, semi, np.cross(major, semi)))


def basis_rotation(basis: np.ndarray) -> dict:
    normal = basis[:, 2].copy()
    if normal[2] < 0:
        normal *= -1
    dip = np.arccos(np.clip(normal[2], -1, 1))
    az = np.arctan2(normal[0], normal[1]) if np.sin(dip) > 1e-9 else 0.0
    strike = np.array([-np.cos(az), np.sin(az), 0])
    down = np.array([np.sin(az) * np.cos(dip), np.cos(az) * np.cos(dip), -np.sin(dip)])
    pitch = np.arctan2(np.dot(basis[:, 0], down), np.dot(basis[:, 0], strike))
    return {"dip_direction": float(np.rad2deg(az) % 360), "dip": float(np.rad2deg(dip)),
            "pitch": float(np.rad2deg(pitch) % 180)}


def experiment(pairs: Pairs, vector, lag: float, bins: int, tolerance: float, bandwidth=None):
    if not 0 < tolerance <= 90 or lag <= 0 or bins < 1:
        raise ValueError("Invalid lag/bins/angular tolerance.")
    selected = np.ones(len(pairs.distance), dtype=bool)
    if vector is not None:
        along = np.abs(pairs.delta @ vector)
        selected &= along >= pairs.distance * np.cos(np.deg2rad(tolerance)) - 1e-12
        if bandwidth is not None:
            if not np.isfinite(bandwidth) or bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            selected &= np.sqrt(np.maximum(0, pairs.distance ** 2 - along ** 2)) <= bandwidth
    indices = np.floor(pairs.distance / lag).astype(int)
    selected &= indices < bins
    idx = indices[selected]
    count = np.bincount(idx, minlength=bins)
    distance = np.bincount(idx, weights=pairs.distance[selected], minlength=bins)
    gamma = np.bincount(idx, weights=pairs.gamma[selected], minlength=bins)
    return {
        "distance": [float(distance[i] / count[i]) if count[i] else (i + 0.5) * lag for i in range(bins)],
        "gamma": [float(gamma[i] / count[i]) if count[i] else None for i in range(bins)],
        "count": count.tolist(),
    }


def rank_directions(pairs, vectors, lag, bins, tolerance, min_pairs, variance, bandwidth=None,
                    comparison_bins=None):
    experiments = [experiment(pairs, v, lag, bins, tolerance, bandwidth) for v in vectors]
    counts = np.array([e["count"] for e in experiments])
    supported = counts >= min_pairs
    if comparison_bins is None:
        # The same early bins are used for every direction, not each direction's
        # own first populated bins. Use bins supported by at least half of
        # candidates that have three supported bins.
        eligible = supported.sum(axis=1) >= 3
        common = np.flatnonzero(supported[eligible].mean(axis=0) >= 0.5) if eligible.any() else np.array([], dtype=int)
        comparison_bins = common[:max(2, int(np.ceil(bins / 3)))].tolist()
    extents = [int(max(np.flatnonzero(row), default=-1) + 1) for row in supported]
    reference_extent = max(extents, default=0)
    reference_pairs = max((int(row.sum()) for row in counts), default=1)
    rankings = []
    for index, exp in enumerate(experiments):
        usable = supported[index]
        near = [b for b in comparison_bins if usable[b]]
        comparable = len(comparison_bins) >= 2 and len(near) == len(comparison_bins)
        coverage = extents[index] / max(reference_extent, 1)
        occupancy = float(usable.sum() / max(reference_extent, 1))
        support = float(counts[index].sum() / max(reference_pairs, 1))
        score = None
        if usable.sum() >= 3 and comparable:
            # Equal lag-bin weight prevents one densely sampled distance from
            # dominating the continuity comparison.
            continuity = float(np.mean([exp["gamma"][b] for b in near]) / variance)
            score = float(continuity + 0.5 * (1 - coverage) + 0.25 * (1 - occupancy) + 0.25 * (1 - np.sqrt(support)))
        rankings.append({"index": index, "score": score, "supported_bins": int(usable.sum()),
                         "supported_pairs": int(counts[index][usable].sum()),
                         "coverage_fraction": coverage, "bin_occupancy": occupancy,
                         "relative_pair_support": support, "comparable": comparable})
    return rankings, comparison_bins


def search_basis(pairs, lag, bins, tolerance, min_pairs, variance, extra_candidates=(),
                 bandwidth=None, diagnostics=None, seed=1729, azimuth_step=15, plunge_step=30):
    validate_settings({"azimuth_step": azimuth_step, "plunge_step": plunge_step})
    def select(vectors):
        rankings, common = rank_directions(pairs, vectors, lag, bins, tolerance, min_pairs, variance, bandwidth)
        valid = [r for r in rankings if r["score"] is not None]
        if not valid:
            raise ValueError("Major direction unresolved: insufficient pair support at comparable lag distances. "
                             "Increase pair budget, adjust lag/tolerance or supply a geological manual direction.")
        best = min(valid, key=lambda r: r["score"])
        return vectors[best["index"]], rankings, common, best

    positive = list(range(int(plunge_step), 90, int(plunge_step)))
    plunges = [-p for p in reversed(positive)] + [0] + positive
    candidates = [direction(az, pl) for pl in plunges for az in range(0, 180, int(azimuth_step))]
    candidates.append(direction(0, 90))
    candidates.extend(extra_candidates)
    major, rankings, common, best = select(candidates)
    reference = np.array([0.0, 0.0, 1.0]) if abs(major[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(major, reference)
    u /= np.linalg.norm(u)
    v = np.cross(major, u)
    perpendicular = [np.cos(a) * u + np.sin(a) * v for a in np.deg2rad(np.arange(0, 180, azimuth_step))]
    semi, _, _, _ = select(perpendicular)
    if diagnostics is not None:
        rng = np.random.default_rng(seed)
        winners = []
        for _ in range(4):
            indices = rng.choice(len(pairs.distance), max(1, len(pairs.distance) // 2), replace=False)
            subset = Pairs(pairs.delta[indices], pairs.distance[indices], pairs.gamma[indices],
                           pairs.possible, len(indices), 0)
            trial, _ = rank_directions(subset, candidates, lag, bins, tolerance,
                                      max(1, int(np.ceil(min_pairs / 2))), variance, bandwidth, common)
            valid = [r for r in trial if r["score"] is not None]
            winners.append(candidates[min(valid, key=lambda r: r["score"])["index"]].tolist() if valid else None)
        diagnostics.update({
            "method": "Common-lag continuity plus distance coverage, bin occupancy and relative pair-support penalties",
            "comparison_bin_indices": common, "rankings": rankings,
            "candidate_vectors": [v.tolist() for v in candidates],
            "selected_search_major": major.tolist(), "resampled_major_vectors": winners,
            "selected_support": best, "resampling_seed": seed,
            "azimuth_step_degrees": int(azimuth_step), "plunge_step_degrees": int(plunge_step),
            "stability_method": "Four reproducible half-pair subsamples without replacement; pairs share samples, so this is not independent sample/drillhole validation",
        })
    return np.column_stack((major, semi, np.cross(major, semi)))


def promising_frames(pairs, diagnostics, lag, bins, tolerance, min_pairs, variance,
                     bandwidth=None, azimuth_step=15, log=lambda message: None):
    vectors = np.array(diagnostics["candidate_vectors"])
    ranked = sorted((r for r in diagnostics["rankings"] if r["score"] is not None),
                    key=lambda r: r["score"])
    seeds = []
    for rank in ranked:
        vector = vectors[rank["index"]]
        if all(abs(vector @ previous) < np.cos(np.deg2rad(25)) for previous in seeds):
            seeds.append(vector)
        if len(seeds) == 4:
            break
    pool = []
    for index, major in enumerate(seeds):
        log(f"Screening alternative 3D frames from direction {index + 1}/{len(seeds)}")
        reference = np.array([0., 0, 1]) if abs(major[2]) < 0.9 else np.array([1., 0, 0])
        u = np.cross(major, reference)
        u /= np.linalg.norm(u)
        v = np.cross(major, u)
        semis = [np.cos(a) * u + np.sin(a) * v for a in np.deg2rad(np.arange(0, 180, azimuth_step))]
        ranks, _ = rank_directions(pairs, semis, lag, bins, tolerance, min_pairs, variance, bandwidth)
        valid = sorted((r for r in ranks if r["score"] is not None), key=lambda r: r["score"])
        chosen = []
        for rank in valid:
            semi = semis[rank["index"]]
            if any(abs(semi @ previous) >= np.cos(np.deg2rad(25)) for previous in chosen):
                continue
            chosen.append(semi)
            frame = np.column_stack((major, semi, np.cross(major, semi)))
            axis_ranks, _ = rank_directions(
                pairs, list(frame.T), lag, bins, tolerance, min_pairs, variance, bandwidth)
            if all(r["score"] is not None for r in axis_ranks):
                pool.append((sum(r["score"] for r in axis_ranks), frame))
            if len(chosen) == 2:
                break
    # Retain geometrically distinct triads, then test every named-axis
    # assignment so the shortest direction need not remain the minor.
    triads = []
    for _, frame in sorted(pool, key=lambda item: item[0]):
        if any(np.all(np.max(abs(previous.T @ frame), axis=0) > np.cos(np.deg2rad(10)))
               for previous in triads):
            continue
        triads.append(frame)
        if len(triads) == 4:
            break
    frames = []
    for triad in triads:
        for order in permutations(range(3)):
            frame = triad[:, order].copy()
            frame[:, 2] = np.cross(frame[:, 0], frame[:, 1])
            frames.append(frame)
    diagnostics["frame_shortlist"] = {
        "seed_directions": [v.tolist() for v in seeds], "screened_triads": len(pool),
        "retained_triads": len(triads), "ordered_frames": len(frames),
        "policy": "Up to four major seeds separated by 25 degrees, two separated perpendicular directions per seed; "
                  "up to four distinct supported triads ranked by summed direction scores; all six axis assignments",
    }
    return frames


def spherical(h):
    h = np.asarray(h)
    return np.where(h < 1, 1.5 * h - 0.5 * h ** 3, 1.0)


def assess_orientation(diagnostics, final_basis, experiments, bins, min_pairs):
    major = final_basis[:, 0]
    refinement = diagnostics.get("plane_refinement", {})
    joint_comparison = refinement.get("status") in ("rotated_frame_selected", "alternative_3d_frame_selected")
    winners = (refinement["resampled_major_vectors"] if joint_comparison
               else diagnostics["resampled_major_vectors"])
    deviations = [
        float(np.degrees(np.arccos(np.clip(abs(np.dot(major, vector)), 0, 1))))
        if vector is not None else None for vector in winners
    ]
    agreement = sum(angle is not None and angle <= 30 for angle in deviations) / len(deviations)
    normal_deviations = []
    if joint_comparison:
        normal_deviations = [
            float(np.degrees(np.arccos(np.clip(abs(np.dot(final_basis[:, 2], vector)), 0, 1))))
            if vector is not None else None for vector in refinement["resampled_plane_normals"]
        ]
    normal_agreement = (sum(a is not None and a <= 30 for a in normal_deviations) / len(normal_deviations)
                        if normal_deviations else None)
    supported = np.array([e["count"] for e in experiments]) >= min_pairs
    extents = [int(max(np.flatnonzero(row), default=-1) + 1) for row in supported]
    coverage = extents[0] / max(max(extents), 1)
    alternatives = []
    aligned = []
    comparisons = ([(r["major_vector"], r) for r in refinement["candidates"] if r.get("eligible")]
                   if joint_comparison else
                   zip(diagnostics.get("candidate_vectors", []), diagnostics.get("rankings", [])))
    for candidate, rank in comparisons:
        if rank["score"] is None:
            continue
        angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(major, candidate)), 0, 1))))
        if joint_comparison:
            normal_angle = float(np.degrees(np.arccos(np.clip(
                abs(np.dot(final_basis[:, 2], rank["plane_normal"])), 0, 1))))
            angle = max(angle, normal_angle)
        (aligned if angle <= 30 else alternatives).append(rank["score"])
    margin = min(alternatives) - min(aligned) if aligned and alternatives else None
    required_margin = 0.01 if joint_comparison else 0.1
    ambiguous = margin is not None and margin < required_margin
    resolved = (agreement >= 0.75 and coverage >= 0.5 and
                supported[0].sum() >= max(3, int(np.ceil(bins / 3))) and not ambiguous and
                (normal_agreement is None or normal_agreement >= 0.75))
    diagnostics.update({
        "status": "provisional_supported_direction" if resolved else "major_direction_unresolved",
        "final_major_vector": major.tolist(), "half_pair_angular_deviations_degrees": deviations,
        "fraction_trials_within_30_degrees": agreement, "final_major_distance_coverage": coverage,
        "score_margin_to_direction_over_30_degrees_away": margin, "competing_directions_ambiguous": ambiguous,
        "assessment_method": refinement["stability_method"] if joint_comparison else diagnostics.get("stability_method"),
        "half_pair_plane_normal_deviations_degrees": normal_deviations,
        "fraction_trials_plane_normal_within_30_degrees": normal_agreement,
        "resolution_policy": f"At least 3/4 half-pair winners within 30 degrees (major and, for selected alternative frames, plane normal), at least half directional distance coverage, at least max(3, ceil(bins/3)) supported major bins, and score margin >= {required_margin} to eligible alternatives over 30 degrees away when available",
        "geological_validation": "Not performed; a finite shortlisted 3D frame search is not a global orientation optimum. Even a stable candidate requires geological and independent sample/drillhole review.",
    })


def evaluate(model: dict, distance, axis: str) -> np.ndarray:
    h = np.asarray(distance, dtype=float)
    result = np.where(h > 0, model["nugget"], 0.0)
    reference_rotation = model["structures"][0]["anisotropy"].get("rotation")
    reference_direction = rotation_basis(reference_rotation)[:, AXES.index(axis)] if reference_rotation else None
    for structure in model["structures"]:
        anisotropy = structure["anisotropy"]
        if reference_direction is not None and anisotropy.get("rotation"):
            local_direction = rotation_basis(anisotropy["rotation"]).T @ reference_direction
            ranges = np.array([anisotropy["ranges"][k] for k in AXES])
            ratio = h * np.linalg.norm(local_direction / ranges)
        else:
            ratio = h / anisotropy["ranges"][axis]
        if structure["structure_type"] == "spherical":
            curve = spherical(ratio)
        elif structure["structure_type"] == "linear":
            curve = ratio
        else:
            raise ValueError("Spheroidal LFVs can be round-tripped, but their proprietary curve is not approximated.")
        result += structure["contribution"] * curve
    return result


class FitConvergenceError(ValueError):
    pass


def fit(experiments, variance, rotation, attribute, domain, lag, max_distance, count, min_pairs,
        fixed_nugget=None):
    if fixed_nugget is not None and (
            not np.isfinite(fixed_nugget) or fixed_nugget < 0):
        raise ValueError("Fixed nugget must be finite and nonnegative.")
    if count not in (1, 2):
        raise ValueError("Select one or two spherical structures.")
    observations = []
    for axis, exp in enumerate(experiments):
        supported = [i for i, c in enumerate(exp["count"]) if c >= min_pairs and exp["gamma"][i] is not None]
        if len(supported) < 3:
            raise ValueError(f"{AXES[axis]} has fewer than three supported lag bins.")
        h = np.array([exp["distance"][i] for i in supported])
        y = np.array([exp["gamma"][i] for i in supported]) / variance
        weights = np.sqrt(np.array([exp["count"][i] for i in supported], float))
        observations.append((h, y, weights))
    total_weight = np.sqrt(sum(np.sum(weight ** 2) for _, _, weight in observations))
    observations = [(h, y, weight / total_weight) for h, y, weight in observations]
    lower_range = lag * 0.1
    caps = np.array([min(max_distance, h.max()) for h, _, _ in observations])
    if np.any(caps <= lower_range):
        raise ValueError("Supported directional distances are too short to constrain ranges.")

    def decode_ranges(params):
        fractions = params[1 + count:].reshape(count, 3)
        ranges = np.empty_like(fractions)
        for axis in range(3):
            ceiling = caps[axis] if axis == 0 else np.minimum(caps[axis], ranges[:, axis - 1])
            ranges[:, axis] = lower_range + fractions[:, axis] * (ceiling - lower_range)
        return ranges

    def residual(params):
        if fixed_nugget is not None:
            params = np.r_[fixed_nugget / variance, params]
        pieces = []
        ranges = decode_ranges(params)
        for axis, (h, y, weight) in enumerate(observations):
            pred = np.full(len(h), params[0])
            for s in range(count):
                pred += params[1 + s] * spherical(h / ranges[s, axis])
            pieces.append((pred - y) * weight)
        return np.concatenate(pieces)

    lower = np.zeros(1 + count + count * 3)
    upper = np.array([3.0] * (1 + count) + [1.0] * (count * 3))
    if fixed_nugget is not None:
        lower, upper = lower[1:], upper[1:]
    attempts = []
    for scale in (0.3, 0.65, 0.95):
        fractions = np.array([[scale * (s + 1) / count, 0.7, 0.6] for s in range(count)])
        initial = np.r_[0.1, [0.9 / count] * count, fractions.ravel()]
        if fixed_nugget is not None:
            initial = initial[1:]
        solved = least_squares(residual, initial, bounds=(lower, upper), max_nfev=2000, x_scale="jac")
        if solved.success and np.isfinite(solved.x).all():
            attempts.append(solved)
    if not attempts:
        raise FitConvergenceError("Variogram fitting did not converge; review experimental curves and settings.")
    solved = min(attempts, key=lambda s: np.dot(s.fun, s.fun))
    parameters = solved.x if fixed_nugget is None else np.r_[fixed_nugget / variance, solved.x]
    ranges = decode_ranges(parameters)
    structures = [{
        "contribution": float(parameters[1 + s] * variance),
        "anisotropy": {"ranges": dict(zip(AXES, map(float, ranges[s]))), "rotation": dict(rotation)},
        "structure_type": "spherical", "parameters": {},
    } for s in range(count)]
    structures.sort(key=lambda s: sum(s["anisotropy"]["ranges"].values()))
    nugget = float(parameters[0] * variance) if fixed_nugget is None else float(fixed_nugget)
    model = {
        "content_type": "com.seequent.elements.variogram", "content_version": "2",
        "metadata": {"name": f"Variogram {attribute} {domain}", "description": "SuperVario candidate; geological review required",
                     "modelling_space": "data", "domain": str(domain), "attribute": attribute,
                     "extensions": {"leapfrog": {"sill_type": "normalised"}}},
        "nugget": nugget, "sill": nugget + sum(s["contribution"] for s in structures),
        "data_variance": float(variance), "is_rotation_fixed": True, "structures": structures,
    }
    warnings = []
    range_limited = bool(np.any(ranges >= caps * 0.99))
    if range_limited:
        warnings.append("Range reached the supported-distance limit: the sill/range is not resolved by the data. "
                        "Automatic extrapolation was prevented; extend sampling/lag support or review the model.")
    if np.any(solved.active_mask[:count + (fixed_nugget is None)]):
        warnings.append("A nugget/contribution reached its fitting bound; review model complexity and data support.")
    if np.any(ranges <= lower_range * 1.01):
        warnings.append("A fitted range is below practical lag resolution; separate nugget and short-range structure may be unidentifiable.")
    warnings.append(
        "Nugget is jointly fitted, not independently measured at zero separation; "
        "inspect short-distance support before interpreting it as measurement error or microscale variability."
        if fixed_nugget is None else
        "Nugget fixed by explicit downhole review, not independently measured at zero separation; "
        "contributions and ranges refitted with orientation unchanged.")
    return validate(model), {"weighted_rmse_normalised": float(np.sqrt(np.sum(solved.fun ** 2))),
                             "fitting_warnings": list(warnings),
                             "warnings": warnings, "converged": True,
                             "direction_weighting": "Global pair-count weights; no equal-total-weight normalisation per axis",
                             "range_order_enforced": True, "shared_orientation": True,
                             "range_caps": dict(zip(AXES, map(float, caps))),
                             "nearest_supported_lag_distance": dict(zip(AXES, [float(h.min()) for h, _, _ in observations])),
                             "range_support_limited": range_limited}


def fit_axis_assignments(experiments, variance, rotation, attribute, domain, lag, max_distance,
                         count, min_pairs, automatic=True, log=lambda message: None):
    basis = rotation_basis(rotation)
    candidates = []
    orders = list(permutations(range(3))) if automatic else [(0, 1, 2)]
    for order in orders:
        log(f"{domain}: constrained fit with axes " + " / ".join(AXES[i] for i in order))
        candidate_basis = basis[:, order].copy()
        candidate_basis[:, 2] = np.cross(candidate_basis[:, 0], candidate_basis[:, 1])
        candidate_rotation = basis_rotation(candidate_basis) if automatic else dict(rotation)
        ordered_experiments = [experiments[i] for i in order]
        model, info = fit(ordered_experiments, variance, candidate_rotation, attribute, domain,
                          lag, max_distance, count, min_pairs)
        info["axis_assignment_from_search"] = [AXES[i] for i in order]
        candidates.append((model, info, ordered_experiments))
    best = min(candidates, key=lambda candidate: candidate[1]["weighted_rmse_normalised"])
    best[1]["axis_assignments_tested"] = len(candidates)
    return best


def supported_plane_range(model, experiments, min_pairs):
    total = 0.0
    for structure in model["structures"]:
        contribution = structure["contribution"]
        if contribution < 0.1 * model["sill"]:
            continue
        ranges = [structure["anisotropy"]["ranges"][axis] for axis in AXES[:2]]
        for exp, radius in zip(experiments[:2], ranges):
            distances = np.array([h for h, n in zip(exp["distance"], exp["count"]) if n >= min_pairs])
            # A longer fitted range earns no reward without observations of
            # both its rising part and its sill, away from the distance bound.
            if (np.count_nonzero(distances < radius) < 3 or
                    np.count_nonzero(distances >= radius * 1.1) < 2):
                break
        else:
            total += contribution / model["sill"] * float(np.sqrt(np.prod(ranges)))
    return total


def refine_plane(pairs, baseline, variance, attribute, domain, lag, bins, tolerance,
                 max_distance, count, min_pairs, bandwidth=None, seed=1729, log=lambda message: None,
                 in_plane_step=15, alternative_frames=()):
    validate_settings({"in_plane_step": in_plane_step})
    model, info, experiments = baseline
    basis = rotation_basis(model["structures"][0]["anisotropy"]["rotation"])
    rng = np.random.default_rng(seed + 1)
    indices = rng.permutation(len(pairs.distance))
    split = len(indices) * 2 // 3

    def subset(indices):
        return Pairs(pairs.delta[indices], pairs.distance[indices], pairs.gamma[indices],
                     pairs.possible, len(indices), 0)

    training, validation = subset(indices[:split]), subset(indices[split:])
    train_min = max(1, int(np.ceil(min_pairs * 2 / 3)))
    test_min = max(1, int(np.ceil(min_pairs / 3)))
    candidates = []
    diagnostics = {
        "method": "Joint orthogonal-frame comparison: in-plane rotations and shortlisted alternative 3D frames",
        "in_plane_step_degrees": int(in_plane_step),
        "policy": f"{int(in_plane_step)}-degree rotations; resolved contribution-weighted plane range rewarded by up to 0.1 normalised RMSE; "
                  "at least 5% range gain, held-out RMSE increase <= 0.02, and >= 80% baseline pair support required",
        "validation": "Deterministic 2/3 training and 1/3 held-out pairs; shared samples mean this is sensitivity evidence, not independent validation",
        "candidates": [], "seed": seed + 1,
    }
    proposals = []
    for angle in range(0, 90, int(in_plane_step)):
        theta = np.deg2rad(angle)
        rotated = np.column_stack((np.cos(theta) * basis[:, 0] + np.sin(theta) * basis[:, 1],
                                   -np.sin(theta) * basis[:, 0] + np.cos(theta) * basis[:, 1], basis[:, 2]))
        proposals.append((angle, rotated, "in_plane"))
    for frame in alternative_frames:
        proposals.append((None, frame, "alternative_3d"))
    seen = []
    for angle, rotated, source in proposals:
        log(f"{domain}: comparing {source} frame ({angle if angle is not None else 'alternative normal'})")
        for swapped in ((False, True) if source == "in_plane" else (False,)):
            if angle == 0 and swapped:
                continue
            frame = rotated[:, [1, 0, 2]].copy() if swapped else rotated.copy()
            frame[:, 2] = np.cross(frame[:, 0], frame[:, 1])
            if any(np.all(abs(np.diag(previous.T @ frame)) > 1 - 1e-10) for previous in seen):
                continue
            seen.append(frame)
            rotation = basis_rotation(frame)
            full = [experiment(pairs, frame[:, i], lag, bins, tolerance, bandwidth) for i in range(3)]
            train = [experiment(training, frame[:, i], lag, bins, tolerance, bandwidth) for i in range(3)]
            test = [experiment(validation, frame[:, i], lag, bins, tolerance, bandwidth) for i in range(3)]
            record = {"angle_degrees": angle, "plane_axes_swapped": swapped, "rotation": rotation,
                      "source": source, "plane_normal": frame[:, 2].tolist()}
            diagnostics["candidates"].append(record)
            if any(sum(n >= threshold for n in e["count"]) < 3
                   for exps, threshold in ((full, min_pairs), (train, train_min), (test, test_min))
                   for e in exps):
                record["status"] = "insufficient_training_or_validation_support"
                continue
            try:
                candidate = baseline if angle == 0 else (*fit(
                    full, variance, rotation, attribute, domain, lag, max_distance, count, min_pairs), full)
                trained, _ = fit(train, variance, rotation, attribute, domain, lag, max_distance, count, train_min)
            except FitConvergenceError as exc:
                record.update({"status": "fit_did_not_converge", "error": str(exc)})
                log(f"{domain}: rejected frame {angle} degrees (swapped={swapped}): {exc}")
                continue
            record.update({"status": "compared",
                           "supported_plane_range": supported_plane_range(candidate[0], full, min_pairs),
                           "major_vector": frame[:, 0].tolist()})
            candidates.append((candidate, trained, test, record))
    if not candidates or candidates[0][3]["angle_degrees"] != 0:
        diagnostics["status"] = "baseline_validation_support_insufficient"
        info["plane_refinement"] = diagnostics
        log(f"{domain}: retained original frame: baseline training/validation support or convergence insufficient")
        return baseline

    # Every comparison uses the same lag indices on each axis. Otherwise
    # dropping difficult long-distance bins could look like improved prediction.
    common = [[i for i in range(bins) if candidates[0][2][axis]["count"][i] >= test_min]
              for axis in range(3)]
    comparable = []
    for candidate in candidates:
        if any(candidate[2][axis]["count"][i] < test_min for axis in range(3) for i in common[axis]):
            candidate[3].update({"status": "incomplete_baseline_validation_window",
                                 "rejection_reasons": ["missing_baseline_comparison_lags"]})
        else:
            comparable.append(candidate)
    candidates = comparable
    diagnostics["comparison_bin_indices"] = common
    if any(len(indices) < 3 for indices in common):
        diagnostics["status"] = "common_validation_distances_insufficient"
        info["plane_refinement"] = diagnostics
        log(f"{domain}: retained original frame: fewer than three common validation lags per axis")
        return baseline

    def prediction_error(trained, test, comparison=None, threshold=test_min):
        error, support = 0.0, 0
        for axis, indices in enumerate(common if comparison is None else comparison):
            exp = test[axis]
            h = np.array([exp["distance"][i] for i in indices])
            y = np.array([exp["gamma"][i] for i in indices])
            n = np.array([exp["count"][i] for i in indices])
            if any(n < threshold):
                return None, 0
            error += float(np.sum(n * ((evaluate(trained, h, AXES[axis]) - y) / variance) ** 2))
            support += int(n.sum())
        return float(np.sqrt(error / support)), support

    for candidate, trained, test, record in candidates:
        error, support = prediction_error(trained, test)
        comparable_test = [dict(exp, count=[n if i in common[axis] else 0 for i, n in enumerate(exp["count"])])
                           for axis, exp in enumerate(test)]
        full_range = record["supported_plane_range"]
        record.update({"held_out_rmse_normalised": error, "comparison_pairs": support,
                       "full_data_supported_plane_range": full_range,
                       "supported_plane_range": min(full_range, supported_plane_range(
                           candidate[0], comparable_test, test_min))})
    base = candidates[0][3]
    largest = max(c[3]["supported_plane_range"] for c in candidates)

    def score(record, error):
        return error + 0.1 * (1 - record["supported_plane_range"] / max(largest, lag))

    for _, _, _, record in candidates:
        record["score"] = score(record, record["held_out_rmse_normalised"])
        reasons = []
        if record is not base:
            if record["supported_plane_range"] < max(lag, base["supported_plane_range"] * 1.05):
                reasons.append("insufficient_gain_in_supported_structured_range")
            if record["held_out_rmse_normalised"] > base["held_out_rmse_normalised"] + 0.02:
                reasons.append("held_out_prediction_deteriorated")
            if record["comparison_pairs"] < base["comparison_pairs"] * 0.8:
                reasons.append("insufficient_pair_support_relative_to_baseline")
        record.update({"eligible": not reasons, "rejection_reasons": reasons})
    eligible = [c for c in candidates if c[3]["eligible"]]
    chosen = min(eligible, key=lambda c: c[3]["score"])
    selected_status = ("alternative_3d_frame_selected" if chosen[3]["source"] == "alternative_3d"
                       else "rotated_frame_selected")
    diagnostics.update({"status": selected_status if chosen is not candidates[0] else "baseline_retained",
                        "selected_candidate_index": diagnostics["candidates"].index(chosen[3]),
                        "baseline_supported_plane_range": base["supported_plane_range"],
                        "selected_supported_plane_range": chosen[3]["supported_plane_range"],
                        "resampled_major_vectors": [],
                        "resampled_plane_normals": [],
                        "stability_method": "Four half-held-out-pair re-rankings of fixed training fits; not independent refits or sample/drillhole validation"})
    for _ in range(4):
        trial = subset(rng.choice(indices[split:], max(1, len(indices[split:]) // 2), replace=False))
        trial_candidates = []
        for candidate, trained, _, record in eligible:
            frame = rotation_basis(candidate[0]["structures"][0]["anisotropy"]["rotation"])
            test = [experiment(trial, frame[:, i], lag, bins, tolerance, bandwidth) for i in range(3)]
            trial_candidates.append((trained, test, record))
        trial_min = max(1, int(np.ceil(test_min / 2)))
        trial_common = [[i for i in common[axis] if all(
            c[1][axis]["count"][i] >= trial_min for c in trial_candidates)] for axis in range(3)]
        if any(len(indices) < max(3, int(np.ceil(len(common[axis]) / 2)))
               for axis, indices in enumerate(trial_common)):
            diagnostics["resampled_major_vectors"].append(None)
            diagnostics["resampled_plane_normals"].append(None)
            continue
        winners = []
        for trained, test, record in trial_candidates:
            error, _ = prediction_error(trained, test, trial_common, trial_min)
            if error is not None:
                winners.append((score(record, error), record["major_vector"], record["plane_normal"]))
        diagnostics["resampled_major_vectors"].append(min(winners, key=lambda item: item[0])[1] if winners else None)
        diagnostics["resampled_plane_normals"].append(min(winners, key=lambda item: item[0])[2] if winners else None)
    chosen[0][1]["axis_assignments_tested"] = info["axis_assignments_tested"]
    chosen[0][1]["plane_refinement"] = diagnostics
    log(f"{domain}: plane comparison: {diagnostics['status']}")
    return chosen[0]


def validate_settings(settings):
    for key, label, default, minimum, maximum, integer in (
        ("bins", "Lag bins", 12, 3, None, True),
        ("min_pairs", "Minimum pairs per bin", 10, 1, None, True),
        ("max_samples", "Sample cap per domain", 5000, 12, None, True),
        ("max_pairs", "Pair cap per domain", 300000, 1, None, True),
        ("structures", "Spherical structures", 1, 1, 2, True),
        ("angle_tolerance", "Cone half-angle", 30, 0, 90, False),
        ("in_plane_step", "In-plane search step", 15, 5, 90, True),
        ("azimuth_step", "Azimuth search step", 15, 5, 90, True),
        ("plunge_step", "Plunge search step", 30, 5, 90, True),
        ("lag", "Lag distance", 0, 0, None, False),
        ("max_distance", "Maximum distance", 0, 0, None, False),
        ("seed", "Random seed", 1729, 0, None, True),
    ):
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{label} must be a finite number.")
        if integer and value != int(value):
            raise ValueError(f"{label} must be a whole number.")
        if value < minimum or (maximum is not None and value > maximum) or (key == "angle_tolerance" and value == 0):
            if key == "angle_tolerance":
                raise ValueError("Cone half-angle must be greater than 0 and at most 90 degrees.")
            bounds = f"{minimum} to {maximum}" if maximum is not None else f"at least {minimum}"
            raise ValueError(f"{label} must be {bounds}. Only Lag distance and Maximum distance accept 0 for automatic selection.")
    for key, default, label in (("in_plane_step", 15, "In-plane"), ("azimuth_step", 15, "Azimuth"),
                                ("plunge_step", 30, "Plunge")):
        if settings.get(key, default) % 5:
            raise ValueError(f"{label} search step must be a multiple of 5 degrees (5 to 90).")
    bandwidth = settings.get("bandwidth")
    if bandwidth is not None:
        if isinstance(bandwidth, bool) or not isinstance(bandwidth, (int, float)) or not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("Bandwidth must be a finite positive number.")


def downhole_diagnostic(values, drilling, budget=300000, seed=1729, min_pairs=10):
    if drilling is None or "intervals" not in drilling:
        return {"status": "unavailable", "reason": "Map Hole ID, From and To to calculate downhole diagnostics."}
    holes = np.asarray(drilling["hole_id"])
    intervals = np.asarray(drilling["intervals"], dtype=float)
    depths = intervals.mean(axis=1)
    lag = float(np.median(intervals[:, 1] - intervals[:, 0]))
    groups = [np.flatnonzero(holes == hole) for hole in np.unique(holes)]
    sizes = np.array([len(group) * (len(group) - 1) // 2 for group in groups], dtype=np.int64)
    cumulative = np.cumsum(sizes)
    total = int(sizes.sum())
    if total == 0:
        return {"status": "unavailable", "reason": "No same-hole sample pairs are available."}
    ranks = np.sort(np.random.default_rng(seed).choice(total, min(total, budget), replace=False))
    counts = np.zeros(12, dtype=np.int64)
    distances = np.zeros(12)
    sums = np.zeros(12)
    hole_counts = np.zeros(12, dtype=np.int64)
    per_hole = []
    overlapping = False
    for index, group in enumerate(groups):
        ordered = intervals[group][np.argsort(intervals[group, 0])]
        if len(ordered) > 1 and np.any(ordered[1:, 0] < np.maximum.accumulate(ordered[:-1, 1]) - 1e-9):
            overlapping = True
        start = int(cumulative[index] - sizes[index])
        local = ranks[(ranks >= start) & (ranks < cumulative[index])] - start
        if not len(local):
            continue
        n = len(group)
        offsets = np.arange(n, dtype=np.int64)
        offsets = offsets * (2 * n - offsets - 1) // 2
        left = np.searchsorted(offsets, local, side="right") - 1
        right = left + 1 + local - offsets[left]
        left, right = group[left], group[right]
        separation = np.abs(depths[right] - depths[left])
        # Centred depth windows; zero-depth pairs are excluded.
        bins = np.floor(separation / lag + 0.5).astype(int) - 1
        valid = (separation > 0) & (bins >= 0) & (bins < 12)
        bins = bins[valid]
        count = np.bincount(bins, minlength=12)
        counts += count
        hole_counts += count > 0
        hole_distance = np.bincount(bins, weights=separation[valid], minlength=12)
        hole_gamma = np.bincount(bins, weights=0.5 * (values[right[valid]] - values[left[valid]]) ** 2,
                                 minlength=12)
        distances += hole_distance
        sums += hole_gamma
        per_hole.append({"count": count.tolist(), "distance_sum": hole_distance.tolist(),
                         "gamma_sum": hole_gamma.tolist()})
    supported = (counts >= min_pairs) & (hole_counts >= 2)
    return {
        "status": "available" if supported.any() else "insufficient_support",
        "lag": lag, "bins": 12, "possible_pairs": total, "drawn_pairs": len(ranks),
        "distance": [float(distances[i] / counts[i]) if counts[i] else (i + 1) * lag for i in range(12)],
        "gamma": [float(sums[i] / counts[i]) if counts[i] else None for i in range(12)],
        "count": counts.tolist(), "hole_count": hole_counts.tolist(), "supported": supported.tolist(),
        "hole_bin_statistics": per_hole, "overlapping_intervals": overlapping,
        "interval_length_min_max": [float(np.min(intervals[:, 1] - intervals[:, 0])),
                                    float(np.max(intervals[:, 1] - intervals[:, 0]))],
        "method": "Same-hole interval-midpoint depth separation; centred bins [(k+0.5)*lag, (k+1.5)*lag).",
        "limitations": [
            "Lag is median interval length; 12 depth bins. No compositing, detrending or overlap correction.",
            "Holes may have different orientations; no directional model curve is fitted.",
            "The first experimental value is not a measured nugget; review extrapolation towards zero.",
            "Supported bins require minimum pairs and at least two contributing holes; pairs are not independent.",
        ],
    }


def estimate_downhole_nugget(diagnostic, variance, min_pairs=10):
    output = {
        "status": "not_estimated",
        "method": "Weighted linear extrapolation of first three contiguous downhole bins",
        "policy": "At least 30 pairs/bin and three common holes with three pairs/bin; no hole supplies >50% "
                  "of pairs. Nonnegative intercept/slope, residual <=5% variance, window and leave-one-hole "
                  "intercept deviations <=10% variance. No overlaps or interval-length ratio >2.",
        "limitations": "Window/leave-one-hole sensitivity is not a confidence interval. Sub-interval structure "
                        "and mixed hole orientations remain unresolved; geological review is required.",
    }

    def reject(reason):
        output["reason"] = reason
        return output

    if diagnostic["status"] != "available":
        return reject("Supported downhole evidence unavailable; original directional fit retained.")
    if not diagnostic.get("hole_bin_statistics"):
        return reject("Per-hole evidence unavailable; recalculate before automatic estimation.")
    if diagnostic["overlapping_intervals"]:
        return reject("Overlapping intervals prevent automatic nugget estimation.")
    smallest, largest = diagnostic["interval_length_min_max"]
    if largest > 2 * smallest:
        return reject("Mixed interval lengths exceed a 2:1 ratio; review/composite sample support.")
    counts = np.asarray(diagnostic["count"], float)
    holes = np.asarray(diagnostic["hole_count"])
    if np.any(counts[:3] < max(30, min_pairs)) or np.any(holes[:3] < 3):
        return reject("First three contiguous depth bins require at least minimum pairs (floor 30) and three holes.")
    h = np.asarray(diagnostic["distance"], float)
    gamma = np.array([np.nan if v is None else v for v in diagnostic["gamma"]])
    if h[0] > 1.5 * diagnostic["lag"] or np.any(np.diff(h[:3]) <= 0):
        return reject("Near-origin depth coverage is inadequate.")
    statistics = diagnostic["hole_bin_statistics"]
    common = [entry for entry in statistics if min(entry["count"][:3]) >= 3]
    if len(common) < 3:
        return reject("Fewer than three common holes supply three pairs in each early bin.")
    if any(np.any(np.asarray(entry["count"][:3]) / counts[:3] > 0.5) for entry in statistics):
        return reject("One hole dominates an early depth bin (>50% of pairs).")

    def extrapolate(x, y, weight):
        design = np.column_stack((np.ones(len(x)), x))
        weight = np.sqrt(weight / weight.sum())
        parameters = np.linalg.lstsq(design * weight[:, None], y * weight, rcond=None)[0]
        residual = float(np.sqrt(np.sum(((design @ parameters - y) * weight) ** 2)))
        return float(parameters[0]), float(parameters[1]), residual

    intercept, slope, residual = extrapolate(h[:3], gamma[:3], counts[:3])
    output.update(candidate_nugget=intercept, slope=slope, residual_rmse=residual,
                  window_bins=[0, 1, 2], common_holes=len(common))
    if intercept < 0 or intercept > min(gamma[:3]) or slope < 0 or residual > 0.05 * variance:
        return reject("Early-bin intercept/slope or linearity checks failed; no clamping was applied.")
    estimates = [intercept]
    for size in (4, 5):
        if np.all(counts[:size] >= max(30, min_pairs)) and np.all(holes[:size] >= 3):
            estimate, _, _ = extrapolate(h[:size], gamma[:size], counts[:size])
            estimates.append(estimate)
    for entry in statistics:
        own = np.asarray(entry["count"][:3])
        if not own.any():
            continue
        remaining = counts[:3] - own
        x = (h[:3] * counts[:3] - entry["distance_sum"][:3]) / remaining
        y = (gamma[:3] * counts[:3] - entry["gamma_sum"][:3]) / remaining
        estimate, _, _ = extrapolate(x, y, remaining)
        estimates.append(estimate)
    output["sensitivity_nugget_min_max"] = [float(min(estimates)), float(max(estimates))]
    if min(estimates) < 0 or max(abs(value - intercept) for value in estimates) > 0.1 * variance:
        return reject("Extrapolated intercept is unstable across short-lag windows or leave-one-hole checks.")
    output.update(status="estimated", nugget=intercept,
                  reason="Early-bin support, linearity and intercept sensitivity checks passed.")
    return output


def automatic_downhole_refit(model, info, experiments, diagnostic, lag, distance, min_pairs, log):
    estimate = estimate_downhole_nugget(diagnostic, model["data_variance"], min_pairs)
    diagnostic["automatic_nugget"] = estimate
    if estimate["status"] != "estimated":
        log("Automatic downhole nugget not applied: " + estimate["reason"])
        return model, info
    rotation = model["structures"][0]["anisotropy"]["rotation"]
    try:
        candidate, fitted = fit(
            experiments, model["data_variance"], rotation, model["metadata"].get("attribute", ""),
            model["metadata"].get("domain", ""), lag, distance, len(model["structures"]),
            min_pairs, fixed_nugget=estimate["nugget"])
    except FitConvergenceError as exc:
        estimate.update(status="refit_failed", reason=str(exc))
        log("Automatic downhole refit failed; original fit retained: " + str(exc))
        return model, info
    estimate["original_directional_rmse"] = info["weighted_rmse_normalised"]
    estimate["fixed_directional_rmse"] = fitted["weighted_rmse_normalised"]
    if fitted["weighted_rmse_normalised"] > info["weighted_rmse_normalised"] + 0.02:
        estimate.update(status="refit_rejected",
                        reason="Fixed-nugget directional RMSE deteriorated by more than 0.02 normalised units.")
        log("Automatic downhole refit not applied: " + estimate["reason"])
        return model, info
    candidate["metadata"] = model["metadata"].copy()
    prior_fit = dict(info)
    context = [w for w in info["warnings"] if w not in info.get("fitting_warnings", [])]
    fitted["warnings"] = [
        w.replace("Nugget fixed by explicit downhole review", "Nugget fixed by automatic downhole extrapolation")
        for w in fitted["warnings"]
    ]
    fitted["fitting_warnings"] = list(fitted["warnings"])
    fitted["warnings"] += context
    fitted.update(
        fixed_nugget=estimate["nugget"], orientation_reassessed=False,
        nugget_policy="Automatic downhole extrapolation; fixed-nugget refit at retained orientation.",
        automatic_downhole_refit={"previous_model": model, "previous_fit": prior_fit})
    if "plane_refinement" in info:
        fitted["plane_refinement"] = dict(info["plane_refinement"], scope="Orientation selection before fixed-nugget refit")
    estimate.update(status="applied", reason="Extrapolation checks passed and fixed-nugget refit passed directional RMSE guard.")
    log(f"Automatic downhole nugget {estimate['nugget']:.8g} applied; contributions and ranges refitted.")
    return candidate, fitted


def downhole_review(diagnostic, nugget):
    if diagnostic["status"] == "unavailable":
        return diagnostic["reason"]
    supported = [g for g, ok in zip(diagnostic["gamma"], diagnostic["supported"]) if ok]
    if not supported:
        return "Downhole support is insufficient: no bin meets minimum pairs and two-hole support."
    if supported[0] < nugget:
        return ("The first supported downhole semivariance is below the model nugget. "
                "Review short-range structure, sample support and normalization before explicitly editing the nugget.")
    return ("Downhole evidence is available for short-range/nugget review; it does not independently "
            "determine a zero-distance nugget or validate the directional model.")


def spacing_lag(points, drilling, pairs, distance, bins, min_pairs, tolerance=30, rotation=None, bandwidth=None):
    """Choose the finest supported candidate near between-hole spacing."""
    info = {
        "method": "spacing_informed_candidate_search",
        "limitations": [
            "Heuristic spacing/support selection, not a statistically unique optimum or directional validation.",
            "Nearest different-hole sample spacing is not collar spacing or independent drillhole-pair support.",
            "One lag distance per domain; review downhole/short-range behaviour separately.",
        ],
        "candidates": [],
    }
    holes = np.asarray(drilling["hole_id"])
    info["holes_checked"] = int(len(np.unique(holes)))
    info["samples_checked"] = len(points)
    if info["holes_checked"] < 2:
        info["reason"] = "Fewer than two holes: between-hole spacing unavailable; geometric default retained."
        return distance / bins, bins, info
    tree = cKDTree(points)
    unresolved = np.arange(len(points))
    nearest = np.full(len(points), np.nan)
    k = min(8, len(points))
    while len(unresolved):
        pending = []
        for start in range(0, len(unresolved), 128):
            batch = unresolved[start:start + 128]
            d, indices = tree.query(points[batch], k=k)
            other = (holes[indices] != holes[batch, None]) & (d > 0)
            found = other.any(axis=1)
            nearest[batch[found]] = np.min(np.where(other[found], d[found], np.inf), axis=1)
            pending.extend(batch[~found])
        unresolved = np.asarray(pending, dtype=int)
        if k == len(points):
            break
        k = min(k * 2, len(points))
    valid = nearest[np.isfinite(nearest)]
    if not len(valid):
        info["reason"] = "No positive different-hole spacing: geometric default retained."
        return distance / bins, bins, info
    spacing = float(np.median(valid))
    info["different_hole_spacing_p10_p50_p90"] = np.percentile(valid, [10, 50, 90]).tolist()
    support = 0.0
    if "intervals" in drilling:
        intervals = np.asarray(drilling["intervals"])
        lengths = intervals[:, 1] - intervals[:, 0]
        support = float(np.median(lengths))
        info["interval_length_p10_p50_p90"] = np.percentile(lengths, [10, 50, 90]).tolist()
        along = []
        for hole in np.unique(holes):
            midpoints = np.sort(intervals[holes == hole].mean(axis=1))
            along.extend(np.diff(midpoints)[np.diff(midpoints) > 0].tolist())
        if along:
            info["downhole_midpoint_spacing_median"] = float(np.median(along))
        info["limitations"].append("Median interval length constrains candidates; no compositing is performed.")
    else:
        info["limitations"].append("From/To not mapped: sample interval support is unknown.")
    basis = (rotation_basis(rotation) if rotation is not None else
             np.linalg.eigh(np.cov(points, rowvar=False))[1])
    info["directional_screen"] = ("Manual orientation" if rotation is not None else
                                  "Principal coordinate axes; final automatic directions may differ")
    selected = None
    for target in sorted(set(max(support, spacing * scale) for scale in (0.5, 1.0, 2.0))):
        count = min(1000, max(3, int(np.ceil(distance / target))))
        lag = distance / count
        if lag < support:
            continue
        exp = experiment(pairs, None, lag, count, 90)
        supported = int(np.count_nonzero(np.asarray(exp["count"]) >= min_pairs))
        directional = [
            int(np.count_nonzero(np.asarray(
                experiment(pairs, basis[:, axis], lag, count, tolerance, bandwidth)["count"]) >= min_pairs))
            for axis in range(3)
        ]
        acceptable = supported >= max(3, int(np.ceil(count / 2))) and min(directional) >= 3
        info["candidates"].append({
            "lag": lag, "bins": count, "supported_bins": supported,
            "directional_supported_bins": directional, "accepted": acceptable,
        })
        if selected is None and acceptable:
            selected = (lag, count)
    if selected is None:
        info["reason"] = "No spacing candidate meets pooled/directional pair support: geometric default retained."
        return distance / bins, bins, info
    info["reason"] = ("Finest spacing candidate with at least half of pooled bins and three bins on each "
                      "screening direction meeting minimum pairs; maximum distance preserved by adjusting bin count.")
    return selected[0], selected[1], info


def analyse(points, values, attribute, domain, settings, log, drilling=None):
    settings = dict(settings)
    validate_settings(settings)
    max_samples = int(settings.get("max_samples", 5000))
    budget = int(settings.get("max_pairs", 300_000))
    bins = int(settings.get("bins", 12))
    min_pairs = int(settings.get("min_pairs", 10))
    tolerance = float(settings.get("angle_tolerance", 30))
    seed = int(settings.get("seed", 1729))
    structures = int(settings.get("structures", 1))
    if len(points) < 12:
        raise ValueError("At least 12 filtered samples are required per domain.")
    variance = float(np.var(values, ddof=0))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Domain values have zero or invalid variance.")
    original = len(points)
    if drilling is not None:
        holes = np.asarray(drilling["hole_id"])
        if holes.shape != (original,) or any(h is None or not str(h).strip() for h in holes):
            raise ValueError("Drilling metadata requires one nonblank Hole ID per sample.")
        if "intervals" in drilling:
            intervals = np.asarray(drilling["intervals"], dtype=float)
            if (intervals.shape != (original, 2) or not np.isfinite(intervals).all()
                    or np.any(intervals[:, 0] < 0) or np.any(intervals[:, 1] <= intervals[:, 0])):
                raise ValueError("Drilling intervals require one finite 0 <= From < To pair per sample.")
    if original > max_samples:
        selected = np.random.default_rng(seed).choice(original, max_samples, replace=False)
        points, values = points[selected], values[selected]
        if drilling is not None:
            drilling = {key: np.asarray(array)[selected] for key, array in drilling.items()}
        log(f"{domain}: deterministic uniform sample reduction {original:,} -> {max_samples:,}")
    distance = float(settings.get("max_distance") or (np.linalg.norm(np.ptp(points, axis=0)) * 0.5))
    lag = float(settings.get("lag") or distance / bins)
    if not np.isfinite(lag) or lag <= 0 or not np.isfinite(distance) or distance <= 0:
        raise ValueError("Lag and maximum distance must be finite and positive.")
    distance = min(distance, lag * bins)
    log(f"{domain}: drawing up to {budget:,} sample pairs")
    pairs = make_pairs(points, values, distance, budget, seed)
    if len(pairs.distance) < min_pairs * 3:
        raise ValueError("Too few pairs within the maximum distance.")
    lag_info = {"method": "manual" if settings.get("lag") else "geometric_default"}
    if not settings.get("lag") and drilling is not None:
        lag, bins, lag_info = spacing_lag(
            points, drilling, pairs, distance, bins, min_pairs, tolerance,
            settings.get("rotation"), settings.get("bandwidth"))
        log(f"{domain}: automatic lag distance {lag:.8g}, {bins} bins. {lag_info['reason']}")
    lag_info.update(lag=lag, bins=bins, max_distance=distance, samples_reduced=original > len(points))
    trend_info = None
    orientation_info = {"status": "manual_direction_not_statistically_validated"}
    extra_candidates = []
    if settings.get("trend_file"):
        from .trends import DEPRECATION_MESSAGE
        trend_info = {"status": "deprecated_ignored", "source": settings["trend_file"],
                      "message": DEPRECATION_MESSAGE}
        log(f"{domain}: {DEPRECATION_MESSAGE}")
    if settings.get("rotation") is not None:
        rotation = dict(settings["rotation"])
        basis = rotation_basis(rotation)
    else:
        log(f"{domain}: searching for continuity directions")
        basis = search_basis(pairs, lag, bins, tolerance, min_pairs, variance, extra_candidates,
                             bandwidth=settings.get("bandwidth"), diagnostics=orientation_info, seed=seed,
                             azimuth_step=settings.get("azimuth_step", 15), plunge_step=settings.get("plunge_step", 30))
        rotation = basis_rotation(basis)
        basis = rotation_basis(rotation)
    experiments = [experiment(pairs, basis[:, i], lag, bins, tolerance, settings.get("bandwidth")) for i in range(3)]
    log(f"{domain}: fitting {structures} spherical structure(s) to directional variograms")
    model, fit_info, experiments = fit_axis_assignments(
        experiments, variance, rotation, attribute, domain, lag, distance, structures, min_pairs,
        automatic=settings.get("rotation") is None, log=log)
    if settings.get("rotation") is None:
        frames = promising_frames(
            pairs, orientation_info, lag, bins, tolerance, min_pairs, variance, settings.get("bandwidth"),
            settings.get("azimuth_step", 15), log)
        model, fit_info, experiments = refine_plane(
            pairs, (model, fit_info, experiments), variance, attribute, domain, lag, bins, tolerance,
            distance, structures, min_pairs, settings.get("bandwidth"), seed, log,
            in_plane_step=settings.get("in_plane_step", 15), alternative_frames=frames)
        orientation_info["plane_refinement"] = fit_info["plane_refinement"]
    rotation = dict(model["structures"][0]["anisotropy"]["rotation"])
    basis = rotation_basis(rotation)
    log(f"{domain}: calculating radial plot")
    polar_angles = np.arange(0, 360, 15)
    radial = [experiment(pairs, np.cos(a) * basis[:, 0] + np.sin(a) * basis[:, 1],
                         lag, bins, tolerance, settings.get("bandwidth"))
              for a in np.deg2rad(polar_angles)]
    downhole = downhole_diagnostic(values, drilling, budget, seed, min_pairs)
    model, fit_info = automatic_downhole_refit(
        model, fit_info, experiments, downhole, lag, distance, min_pairs, log)
    warnings = fit_info["warnings"]
    if downhole["status"] != "unavailable":
        automatic = downhole["automatic_nugget"]
        warnings.append("Automatic downhole nugget: " + automatic["status"] + ". " + automatic["reason"])
    downhole["review"] = downhole_review(downhole, model["nugget"])
    if downhole["status"] != "unavailable":
        warnings.append(downhole["review"])
    if lag_info["method"] == "spacing_informed_candidate_search":
        warnings.extend(lag_info["limitations"])
        if "retained" in lag_info["reason"]:
            warnings.append(lag_info["reason"])
    if trend_info:
        warnings.append(trend_info["message"])
    if settings.get("rotation") is None:
        assess_orientation(orientation_info, basis, experiments, bins, min_pairs)
        if orientation_info["status"] == "major_direction_unresolved":
            warnings.append("Major direction unresolved: common-distance support and/or half-pair stability is insufficient. "
                            "The displayed orientation is an exploratory candidate, not an established major axis.")
    if pairs.drawn < pairs.possible:
        warnings.append("Experimental variograms use a uniform random subset of unordered pairs.")
    if original > len(points):
        warnings.append("Uniform sample reduction was applied; clustered data are not declustered.")
    if pairs.duplicates:
        warnings.append(f"{pairs.duplicates} drawn zero-distance pairs excluded; inspect duplicate samples.")
    if settings.get("rotation") is None:
        warnings.append("Automatic orientation is a coarse continuity search, not geological validation.")
    return {
        "domain": str(domain), "model": model, "experiments": dict(zip(AXES, experiments)),
        "omnidirectional": experiment(pairs, None, lag, bins, 90),
        "radial": {"angles": polar_angles.tolist(), "gamma": [e["gamma"] for e in radial],
                   "rotation": dict(rotation)},
        "sample_count": original, "used_samples": len(points), "variance": variance,
        "pairs": {"possible": pairs.possible, "drawn": pairs.drawn, "within_distance": len(pairs.distance),
                  "zero_distance_drawn": pairs.duplicates},
        "lag": lag, "lag_selection": lag_info, "downhole": downhole,
        "fitting_settings": {"min_pairs": min_pairs},
        "max_distance": distance, "fit": fit_info, "string_theory": trend_info,
        "orientation_diagnostics": orientation_info,
    }
