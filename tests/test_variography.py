import copy
from pathlib import Path

import numpy as np
import pytest

from supervario import lfv
from supervario.variography import (
    AXES, Pairs, analyse, basis_rotation, direction, evaluate, experiment, fit,
    make_pairs, rotation_basis, search_basis,
    spacing_lag,
    downhole_diagnostic, downhole_review,
    estimate_downhole_nugget, automatic_downhole_refit, FitConvergenceError,
)


REFERENCE = Path(__file__).resolve().parents[1] / "Reference" / "Variogram Model RQDpc UGB01.lfv"


def supported_linear_downhole():
    h = np.arange(1, 13, dtype=float)
    gamma = 0.2 + 0.05 * h
    return {
        "status": "available", "lag": 1, "distance": h.tolist(), "gamma": gamma.tolist(),
        "count": [100] * 12, "hole_count": [5] * 12, "supported": [True] * 12,
        "overlapping_intervals": False, "interval_length_min_max": [1, 1],
        "hole_bin_statistics": [
            {"count": [20] * 12, "distance_sum": (20 * h).tolist(),
             "gamma_sum": (20 * gamma).tolist()} for _ in range(5)],
        "limitations": [],
    }


def test_automatic_downhole_extrapolation_recovers_known_intercept_without_mutation():
    diagnostic = supported_linear_downhole()
    original = copy.deepcopy(diagnostic)
    estimated = estimate_downhole_nugget(diagnostic, 1)
    assert estimated["status"] == "estimated"
    assert estimated["nugget"] == pytest.approx(0.2)
    assert estimated["slope"] == pytest.approx(0.05)
    assert estimated["common_holes"] == 5
    np.testing.assert_allclose(estimated["sensitivity_nugget_min_max"], [0.2, 0.2])
    assert diagnostic == original


@pytest.mark.parametrize("change,reason", [
    ({"status": "unavailable"}, "unavailable"),
    ({"overlapping_intervals": True}, "Overlapping"),
    ({"interval_length_min_max": [1, 3]}, "Mixed"),
    ({"count": [0] + [100] * 11}, "First three"),
    ({"hole_count": [2] * 12}, "First three"),
    ({"hole_bin_statistics": []}, "Per-hole"),
    ({"gamma": [0.4, 0.3, 0.2] + [0.5] * 9}, "intercept/slope"),
    ({"gamma": [0.1, 0.4, 0.7] + [0.8] * 9}, "intercept/slope"),
    ({"gamma": [0.25, 0.3, 0.35, 2., 3.] + [3.] * 7}, "unstable"),
])
def test_automatic_downhole_rejects_insufficient_or_unstable_evidence(change, reason):
    diagnostic = supported_linear_downhole()
    diagnostic.update(change)
    estimated = estimate_downhole_nugget(diagnostic, 1)
    assert estimated["status"] == "not_estimated"
    assert reason in estimated["reason"]


def test_automatic_downhole_rejects_dominant_hole():
    diagnostic = supported_linear_downhole()
    diagnostic["hole_bin_statistics"][0]["count"] = [60] * 12
    assert "dominates" in estimate_downhole_nugget(diagnostic, 1)["reason"]


def test_automatic_downhole_refit_is_exact_and_failure_preserves_original(monkeypatch):
    import supervario.variography as variography
    h = np.linspace(2, 100, 30)
    experiments = [{"distance": h.tolist(), "gamma": (0.2 + 0.8 * variography.spherical(h / r)).tolist(),
                    "count": [100] * len(h)} for r in (80, 50, 20)]
    model, info = fit(experiments, 1, {"dip_direction": 0, "dip": 20, "pitch": 0},
                      "grade", "A", 4, 100, 1, 10)
    info["warnings"].append("Orientation unresolved")
    original = copy.deepcopy((model, info))
    diagnostic = supported_linear_downhole()
    changed, fitted = automatic_downhole_refit(model, info, experiments, diagnostic, 4, 100, 10, lambda _: None)
    assert diagnostic["automatic_nugget"]["status"] == "applied"
    assert changed["nugget"] == diagnostic["automatic_nugget"]["nugget"]
    assert fitted["fixed_nugget"] == changed["nugget"]
    assert "Orientation unresolved" in fitted["warnings"]
    assert fitted["automatic_downhole_refit"]["previous_model"] == original[0]
    assert (model, info) == original

    def fail(*args, **kwargs):
        raise FitConvergenceError("did not converge")

    monkeypatch.setattr(variography, "fit", fail)
    diagnostic = supported_linear_downhole()
    assert automatic_downhole_refit(model, info, experiments, diagnostic, 4, 100, 10, lambda _: None) == original
    assert diagnostic["automatic_nugget"]["status"] == "refit_failed"


def test_automatic_downhole_refit_rejects_directional_deterioration():
    import supervario.variography as variography
    h = np.linspace(2, 100, 30)
    experiments = [{"distance": h.tolist(), "gamma": (0.01 + 0.1 * variography.spherical(h / r)).tolist(),
                    "count": [100] * len(h)} for r in (80, 50, 20)]
    model, info = fit(experiments, 1, {"dip_direction": 0, "dip": 20, "pitch": 0},
                      "grade", "A", 4, 100, 1, 10)
    original = copy.deepcopy((model, info))
    diagnostic = supported_linear_downhole()
    assert automatic_downhole_refit(model, info, experiments, diagnostic, 4, 100, 10, lambda _: None) == original
    assert diagnostic["automatic_nugget"]["status"] == "refit_rejected"

def test_calculation_applies_automatic_nugget_and_records_baseline(monkeypatch):
    import supervario.variography as variography
    rng = np.random.default_rng(52)
    points = rng.uniform(0, 100, (400, 3))
    values = np.sin(points[:, 0] / 20) + rng.normal(0, 0.5, 400)
    settings = {"rotation": {"dip_direction": 0, "dip": 20, "pitch": 0}, "min_pairs": 3,
                "max_distance": 60}
    baseline = analyse(points, values, "grade", "A", settings, lambda _: None)
    nugget = baseline["model"]["nugget"]
    diagnostic = supported_linear_downhole()
    x = np.asarray(diagnostic["distance"])
    y = nugget + 0.01 * x
    diagnostic["gamma"] = y.tolist()
    for entry in diagnostic["hole_bin_statistics"]:
        entry["gamma_sum"] = (20 * y).tolist()
    monkeypatch.setattr(variography, "downhole_diagnostic", lambda *args: copy.deepcopy(diagnostic))
    result = analyse(points, values, "grade", "A", settings, lambda _: None)
    assert result["downhole"]["automatic_nugget"]["status"] == "applied"
    assert result["model"]["nugget"] == pytest.approx(nugget)
    assert result["fit"]["automatic_downhole_refit"]["previous_model"] == baseline["model"]
    assert result["experiments"] == baseline["experiments"]
    assert result["orientation_diagnostics"] == baseline["orientation_diagnostics"]
    assert any("automatic downhole" in w for w in result["fit"]["warnings"])
    import json
    json.dumps(result, allow_nan=False)

@pytest.mark.parametrize("nugget", [0, 0.2, 4])
def test_fixed_nugget_fit_optimises_structures_without_changing_nugget(nugget):
    h = np.linspace(2, 100, 30)
    experiments = [{"distance": h.tolist(), "gamma": (0.2 + 0.8 * np.minimum(h / r, 1)).tolist(),
                    "count": [100] * len(h)} for r in (80, 50, 20)]
    model, info = fit(experiments, 1, {"dip_direction": 30, "dip": 20, "pitch": 0},
                      "grade", "A", 4, 100, 2, 10, fixed_nugget=nugget)
    assert model["nugget"] == nugget
    assert info["converged"]
    assert np.isfinite(info["weighted_rmse_normalised"])
    for s in model["structures"]:
        ranges = s["anisotropy"]["ranges"]
        assert ranges["major"] >= ranges["semi_major"] >= ranges["minor"] > 0
        assert max(ranges.values()) <= 100
    assert not any("jointly fitted" in w for w in info["warnings"])
    assert model["sill"] == pytest.approx(nugget + sum(s["contribution"] for s in model["structures"]))

def test_downhole_uses_same_hole_depths_not_xyz_and_counts_holes():
    drilling = {"hole_id": np.array(["A", "A", "A", "B", "B", "B"]),
                "intervals": np.array([[0, 1], [1, 2], [2, 3]] * 2)}
    values = np.array([0, 2, 4, 100, 102, 104], float)
    result = downhole_diagnostic(values, drilling, min_pairs=1)
    assert result["status"] == "available"
    assert result["possible_pairs"] == result["drawn_pairs"] == 6
    assert result["count"][:2] == [4, 2]
    assert result["hole_count"][:2] == [2, 2]
    assert result["gamma"][:2] == [2, 8]
    assert result["distance"][:2] == [1, 2]
    assert result["gamma"][2] is None
    assert "below the model nugget" in downhole_review(result, 3)
    assert "does not independently" in downhole_review(result, 1)
    assert downhole_diagnostic(values, drilling, budget=3, seed=7) == (
        downhole_diagnostic(values, drilling, budget=3, seed=7))


def test_downhole_missing_metadata_single_hole_and_zero_depth_pairs():
    assert downhole_diagnostic(np.zeros(2), None)["status"] == "unavailable"
    drilling = {"hole_id": np.array(["A", "A", "A"]),
                "intervals": np.array([[0, 1], [0, 1], [1, 2]])}
    result = downhole_diagnostic(np.array([0., 1., 2.]), drilling, min_pairs=1)
    assert result["count"][0] == 2
    assert result["hole_count"][0] == 1
    assert result["status"] == "insufficient_support"
    assert "insufficient" in downhole_review(result, 1)
    drilling["hole_id"] = np.array(["A", "B", "C"])
    assert downhole_diagnostic(np.zeros(3), drilling)["status"] == "unavailable"

def drilling_grid():
    points = np.array([[x, 0, z] for x in (0, 10, 20, 30) for z in range(100)], float)
    drilling = {"hole_id": np.repeat(["001", "002", "003", "004"], 100),
                "intervals": np.tile(np.column_stack((np.arange(100), np.arange(100) + 1)), (4, 1))}
    pairs = make_pairs(points, np.sin(points[:, 2]), 50, 100000, 42)
    return points, drilling, pairs


def test_spacing_lag_uses_other_holes_not_dense_downhole_intervals():
    points, drilling, pairs = drilling_grid()
    lag, bins, info = spacing_lag(points, drilling, pairs, 50, 12, 10, tolerance=90)
    assert lag == 5
    assert bins == 10
    assert lag * bins == 50
    assert info["different_hole_spacing_p10_p50_p90"] == [10, 10, 10]
    assert info["interval_length_p10_p50_p90"] == [1, 1, 1]
    assert info["downhole_midpoint_spacing_median"] == 1
    assert info["candidates"][0]["accepted"]
    drilling["intervals"][:, 1] = drilling["intervals"][:, 0] + 8
    lag, _, _ = spacing_lag(points, drilling, pairs, 50, 12, 10, tolerance=90)
    assert lag >= 8


def test_spacing_lag_fallbacks_are_explicit_and_hole_only_is_supported():
    points, drilling, pairs = drilling_grid()
    drilling.pop("intervals")
    assert spacing_lag(points, drilling, pairs, 50, 12, 10, tolerance=90)[0] == 5
    lag, bins, info = spacing_lag(points, drilling, pairs, 50, 12, 1000000)
    assert lag == 50 / 12 and bins == 12
    assert "geometric default retained" in info["reason"]
    drilling["hole_id"][:] = "001"
    assert "Fewer than two holes" in spacing_lag(points, drilling, pairs, 50, 12, 10)[2]["reason"]

def test_spacing_lag_rejects_candidates_without_directional_support():
    points, drilling, pairs = drilling_grid()
    lag, bins, info = spacing_lag(points, drilling, pairs, 50, 12, 10)
    assert lag == 50 / 12 and bins == 12
    assert all(not candidate["accepted"] for candidate in info["candidates"])
    assert any(0 in candidate["directional_supported_bins"] for candidate in info["candidates"])


def test_spacing_lag_end_to_end_preserves_manual_override_and_metadata_alignment():
    points, drilling, _ = drilling_grid()
    points = np.random.default_rng(42).uniform(0, 50, points.shape)
    settings = {"rotation": {"dip_direction": 0, "dip": 0, "pitch": 0},
                "max_distance": 50, "min_pairs": 1, "max_samples": 200}
    result = analyse(points, np.sin(points[:, 2]), "grade", "A", settings, lambda _: None, drilling)
    assert result["lag_selection"]["method"] == "spacing_informed_candidate_search"
    assert result["lag_selection"]["samples_checked"] == 200
    assert result["lag_selection"]["samples_reduced"]
    assert len(result["radial"]["gamma"][0]) == result["lag_selection"]["bins"]
    assert result["max_distance"] == 50
    settings["lag"] = 4
    manual = analyse(points, np.sin(points[:, 2]), "grade", "A", settings, lambda _: None, drilling)
    assert manual["lag"] == 4
    assert manual["lag_selection"]["method"] == "manual"
    assert manual["max_distance"] == 48


def test_reference_round_trip_preserves_all_values(tmp_path):
    model = lfv.load(REFERENCE)
    destination = tmp_path / "roundtrip.lfv"
    lfv.save(destination, model)
    assert lfv.load(destination) == model
    assert model["structures"][0]["anisotropy"]["ranges"]["major"] == 54.0
    assert len(model["structures"]) == 2


@pytest.mark.parametrize("rotation", [
    {"dip_direction": 105.52, "dip": 57.52, "pitch": 95.58},
    {"dip_direction": 0, "dip": 0, "pitch": 25},
    {"dip_direction": 359, "dip": 90, "pitch": 0},
    {"dip_direction": 190, "dip": 40, "pitch": 140},
])
def test_rotation_round_trip_matches_axial_directions(rotation):
    basis = rotation_basis(rotation)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    recovered = rotation_basis(basis_rotation(basis))
    np.testing.assert_allclose(np.abs(np.diag(basis.T @ recovered)), np.ones(3), atol=1e-12)


@pytest.mark.parametrize("axis,expected_azimuth,expected_plunge", [(0, 116, 57), (1, 19, 5), (2, 286, 32)])
def test_reference_direction_agrees_with_screenshot_labels(axis, expected_azimuth, expected_plunge):
    rotation = lfv.load(REFERENCE)["structures"][0]["anisotropy"]["rotation"]
    vector = rotation_basis(rotation)[:, axis]
    if vector[2] > 0:
        vector *= -1
    az = np.degrees(np.arctan2(vector[0], vector[1])) % 360
    plunge = np.degrees(np.arcsin(-vector[2]))
    assert abs(az - expected_azimuth) < 1
    assert abs(plunge - expected_plunge) < 1


def test_exact_pairs_match_matheron_definition():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], float)
    pairs = make_pairs(points, np.array([0, 2, 4]), 10, 100, 42)
    assert pairs.possible == pairs.drawn == 3
    np.testing.assert_allclose(pairs.distance, [1, 2, np.sqrt(5)])
    np.testing.assert_allclose(pairs.gamma, [2, 8, 2])


def test_sampled_pairs_are_unique_unordered_and_reproducible():
    x = np.arange(100)
    points = np.column_stack([x, x ** 2, np.zeros(100)])
    a = make_pairs(points, x, 1e8, 300, 42)
    b = make_pairs(points, x, 1e8, 300, 42)
    assert a.drawn == len(a.distance) == 300
    assert len(np.unique(a.delta, axis=0)) == 300
    assert np.all(a.delta[:, 0] > 0)
    np.testing.assert_array_equal(a.delta, b.delta)
    np.testing.assert_array_equal(a.gamma, b.gamma)


def test_zero_distance_pairs_are_reported_and_excluded():
    pairs = make_pairs(np.zeros((3, 3)), np.array([1, 2, 3]), 1, 100, 42)
    assert pairs.duplicates == 3
    assert len(pairs.distance) == 0


def test_directional_cone_lag_means_and_bandwidth():
    delta = np.array([[1, 0, 0], [-2, 0, 0], [0, 1, 0], [2, 1, 0]], float)
    pairs = Pairs(delta, np.linalg.norm(delta, axis=1), np.array([1, 4, 9, 16]), 4, 4, 0)
    exp = experiment(pairs, np.array([1, 0, 0]), 1.5, 3, 30, bandwidth=0.5)
    assert exp["count"] == [1, 1, 0]
    assert exp["gamma"] == [1, 4, None]
    assert exp["distance"][:2] == [1, 2]


def test_search_selects_continuity_not_geometry_frequency():
    distances = np.arange(1, 13, dtype=float)
    delta = np.vstack([np.column_stack([distances, distances * 0, distances * 0]),
                       np.column_stack([distances * 0, distances, distances * 0]),
                       np.column_stack([distances * 0, distances * 0, distances])])
    pairs = Pairs(delta, np.tile(distances, 3), np.repeat([0.1, 0.5, 1.0], 12), 36, 36, 0)
    basis = search_basis(pairs, 2.5, 5, 5, 1, 1)
    assert abs(np.dot(basis[:, 0], direction(90, 0))) > 0.99


def test_direction_ranking_does_not_reward_short_sparse_low_semivariance():
    from supervario.variography import rank_directions
    distances = np.r_[np.tile(np.arange(0.5, 3, 1), 10),
                       np.tile(np.arange(0.5, 9, 1), 100)]
    delta = np.zeros((len(distances), 3))
    delta[:30, 0] = distances[:30]
    delta[30:, 1] = distances[30:]
    pairs = Pairs(delta, distances, np.r_[np.full(30, 0.05), np.full(900, 0.3)], 930, 930, 0)
    rankings, common = rank_directions(pairs, [np.array([1, 0, 0]), np.array([0, 1, 0])],
                                      1, 9, 5, 10, 1)
    assert common == [0, 1, 2]
    assert rankings[0]["coverage_fraction"] == pytest.approx(1 / 3)
    assert rankings[0]["score"] > rankings[1]["score"]


def test_direction_ranking_uses_common_bins_not_each_directions_first_bins():
    from supervario.variography import rank_directions
    distances = np.r_[np.tile([0.5, 1.5, 2.5], 10), np.tile([3.5, 4.5, 5.5], 10)]
    delta = np.zeros((60, 3))
    delta[:30, 0] = distances[:30]
    delta[30:, 1] = distances[30:]
    pairs = Pairs(delta, distances, np.ones(60), 60, 60, 0)
    rankings, common = rank_directions(pairs, [np.array([1, 0, 0]), np.array([0, 1, 0])],
                                      1, 6, 5, 10, 1)
    assert common == [0, 1]
    assert rankings[1]["score"] is None


def test_pair_resampling_direction_diagnostics_are_reproducible():
    distances = np.tile(np.arange(0.5, 12, 1), 20)
    delta = np.vstack([np.column_stack([distances, distances * 0, distances * 0]),
                       np.column_stack([distances * 0, distances, distances * 0]),
                       np.column_stack([distances * 0, distances * 0, distances])])
    pairs = Pairs(delta, np.tile(distances, 3), np.repeat([0.1, 0.5, 1.0], len(distances)), 720, 720, 0)
    first, second = {}, {}
    search_basis(pairs, 1, 12, 5, 10, 1, diagnostics=first)
    search_basis(pairs, 1, 12, 5, 10, 1, diagnostics=second)
    assert first == second
    assert len(first["resampled_major_vectors"]) == 4


def test_unstable_or_short_major_direction_is_reported_unresolved():
    from supervario.variography import assess_orientation
    diagnostics = {"resampled_major_vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1], None]}
    experiments = [{"count": [100] * 3 + [0] * 9}, {"count": [100] * 12}, {"count": [100] * 12}]
    assess_orientation(diagnostics, np.eye(3), experiments, 12, 10)
    assert diagnostics["status"] == "major_direction_unresolved"
    assert diagnostics["fraction_trials_within_30_degrees"] == 0.25
    assert diagnostics["final_major_distance_coverage"] == 0.25


def test_supported_stable_direction_remains_provisional():
    from supervario.variography import assess_orientation
    diagnostics = {"resampled_major_vectors": [[1, 0, 0]] * 4}
    assess_orientation(diagnostics, np.eye(3), [{"count": [100] * 12}] * 3, 12, 10)
    assert diagnostics["status"] == "provisional_supported_direction"


def test_competing_equal_continuity_directions_are_not_declared_resolved():
    from supervario.variography import assess_orientation
    diagnostics = {"resampled_major_vectors": [[1, 0, 0]] * 4,
                   "candidate_vectors": [[1, 0, 0], [0, 1, 0]],
                   "rankings": [{"score": 0.2}, {"score": 0.21}]}
    assess_orientation(diagnostics, np.eye(3), [{"count": [100] * 12}] * 3, 12, 10)
    assert diagnostics["status"] == "major_direction_unresolved"
    assert diagnostics["competing_directions_ambiguous"]


def test_sparse_axis_has_less_total_influence_on_joint_fit():
    h = np.linspace(1, 100, 30)
    truth = {"nugget": 0.2, "structures": [{"contribution": 0.8, "structure_type": "spherical",
             "anisotropy": {"ranges": dict(zip(AXES, [60, 40, 20]))}}]}
    dense = [{"distance": h.tolist(), "gamma": evaluate(truth, h, axis).tolist(), "count": [10000] * len(h)}
             for axis in AXES]
    # A sparse, biased major curve should not have equal total weight to the
    # other two curves merely because there are three directions.
    sparse = copy.deepcopy(dense)
    sparse[0]["gamma"] = (np.array(sparse[0]["gamma"]) + 0.5).tolist()
    sparse[0]["count"] = [10] * len(h)
    model, info = fit(sparse, 1, {"dip_direction": 0, "dip": 0, "pitch": 0}, "grade", "A", 4, 100, 1, 10)
    assert model["nugget"] == pytest.approx(0.2, abs=0.02)
    assert "Global" in info["direction_weighting"]


def plane_compromise_pairs(rotated_pairs_per_bin=300, unresolved=False):
    h = np.arange(2.5, 100, 5)
    vectors = [np.array([1., 0, 0]), np.array([0., 1, 0]),
               np.array([1., 1, 0]) / np.sqrt(2), np.array([-1., 1, 0]) / np.sqrt(2),
               np.array([0., 0, 1])]
    delta, gamma = [], []
    for i, (vector, radius) in enumerate(zip(vectors, [25, 20, 70, 45, 10])):
        n = rotated_pairs_per_bin if i in (2, 3) else (20 if i == 4 else 300)
        distances = np.repeat(h, n)
        nugget = 0.35 if i in (2, 3) else 0.2
        curve = nugget + (1 - nugget) * (1.5 * np.minimum(distances / radius, 1) -
                                        0.5 * np.minimum(distances / radius, 1) ** 3)
        if unresolved and i in (2, 3):
            curve = 0.2 + distances / 200
        delta.append(distances[:, None] * vector)
        gamma.append(curve)
    delta = np.vstack(delta)
    return Pairs(delta, np.linalg.norm(delta, axis=1), np.concatenate(gamma), len(delta), len(delta), 0)


@pytest.mark.parametrize("step", [5, 15, 25, 90])
def test_plane_search_uses_configured_step(step):
    from supervario.variography import fit_axis_assignments, refine_plane
    pairs = plane_compromise_pairs()
    experiments = [experiment(pairs, np.eye(3)[:, i], 5, 20, 5) for i in range(3)]
    baseline = fit_axis_assignments(experiments, 1, basis_rotation(np.eye(3)), "grade", "A", 5, 100, 1, 10)
    result = refine_plane(pairs, baseline, 1, "grade", "A", 5, 20, 5, 100, 1, 10, in_plane_step=step)
    diagnostics = result[1]["plane_refinement"]
    assert diagnostics["in_plane_step_degrees"] == step
    assert sorted({r["angle_degrees"] for r in diagnostics["candidates"]}) == list(range(0, 90, step))
    assert len(diagnostics["candidates"]) == 2 * len(range(0, 90, step)) - 1


@pytest.mark.parametrize("key,default", [("azimuth_step", 15), ("plunge_step", 30)])
def test_3d_search_grid_uses_configured_increments(key, default):
    pairs = plane_compromise_pairs()
    diagnostics = {}
    settings = {key: 5}
    search_basis(pairs, 5, 20, 10, 10, 1, diagnostics=diagnostics, **settings)
    assert diagnostics[key + "_degrees"] == 5
    azimuth = settings.get("azimuth_step", 15)
    plunge = settings.get("plunge_step", 30)
    expected_plunges = 1 + 2 * len(range(plunge, 90, plunge))
    assert len(diagnostics["candidate_vectors"]) == len(range(0, 180, azimuth)) * expected_plunges + 1


def test_promising_3d_frames_have_distinct_normals_and_are_reproducible():
    from supervario.variography import promising_frames
    rng = np.random.default_rng(19)
    delta = rng.normal(size=(12000, 3))
    delta /= np.linalg.norm(delta, axis=1)[:, None]
    distances = rng.uniform(0.1, 100, len(delta))
    delta *= distances[:, None]
    pairs = Pairs(delta, distances, np.full(len(delta), 0.5), len(delta), len(delta), 0)
    diagnostics = {}
    search_basis(pairs, 5, 20, 30, 5, 1, diagnostics=diagnostics)
    frames = promising_frames(pairs, diagnostics, 5, 20, 30, 5, 1)
    repeated = promising_frames(pairs, diagnostics, 5, 20, 30, 5, 1)
    np.testing.assert_array_equal(frames, repeated)
    assert 6 <= len(frames) <= 24
    for frame in frames:
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
        assert np.linalg.det(frame) == pytest.approx(1)
    assert any(abs(frames[0][:, 2] @ f[:, 2]) < 0.9 for f in frames)


@pytest.mark.parametrize("structures", [1, 2])
def test_joint_3d_comparison_can_choose_a_different_plane_normal(structures):
    from supervario.variography import fit_axis_assignments, refine_plane, assess_orientation
    alternative = rotation_basis({"dip_direction": 135, "dip": 45, "pitch": 30})
    distances = np.repeat(np.arange(2.5, 100, 5), 300)
    delta, gamma = [], []
    for frame, ranges, nugget in [(np.eye(3), [25, 20, 10], 0.2),
                                   (alternative, [70, 45, 15], 0.35)]:
        truth = {"nugget": nugget, "structures": [{"contribution": 1 - nugget,
                 "structure_type": "spherical", "anisotropy": {"ranges": dict(zip(AXES, ranges))}}]}
        for axis, name in enumerate(AXES):
            delta.append(distances[:, None] * frame[:, axis])
            gamma.append(evaluate(truth, distances, name))
    delta = np.vstack(delta)
    pairs = Pairs(delta, np.linalg.norm(delta, axis=1), np.concatenate(gamma), len(delta), len(delta), 0)
    exps = [experiment(pairs, np.eye(3)[:, i], 5, 20, 5) for i in range(3)]
    baseline = fit_axis_assignments(exps, 1, basis_rotation(np.eye(3)), "grade", "A", 5, 100, structures, 10)
    chosen = refine_plane(pairs, baseline, 1, "grade", "A", 5, 20, 5, 100, structures, 10,
                          alternative_frames=[alternative])
    diagnostics = chosen[1]["plane_refinement"]
    assert diagnostics["status"] == "alternative_3d_frame_selected"
    final = rotation_basis(chosen[0]["structures"][0]["anisotropy"]["rotation"])
    np.testing.assert_allclose(abs(np.diag(alternative.T @ final)), 1, atol=1e-12)
    assert abs(final[:, 2] @ np.array([0, 0, 1])) < 0.9
    assert diagnostics["selected_supported_plane_range"] > diagnostics["baseline_supported_plane_range"]
    assessment = {"plane_refinement": diagnostics, "resampled_major_vectors": [[1, 0, 0]] * 4}
    assess_orientation(assessment, final, chosen[2], 20, 10)
    assert assessment["fraction_trials_plane_normal_within_30_degrees"] == 1
    unstable = copy.deepcopy(diagnostics)
    unstable["resampled_plane_normals"] = [final[:, 1].tolist()] * 4
    assessment = {"plane_refinement": unstable, "resampled_major_vectors": [[1, 0, 0]] * 4}
    assess_orientation(assessment, final, chosen[2], 20, 10)
    assert assessment["fraction_trials_within_30_degrees"] == 1
    assert assessment["status"] == "major_direction_unresolved"


@pytest.mark.parametrize("step", [0, 4, 7, 12.5, 95, True, float("nan"), "5"])
@pytest.mark.parametrize("key", ["in_plane_step", "azimuth_step", "plunge_step"])
def test_invalid_plane_search_steps_are_rejected(step, key):
    from supervario.variography import validate_settings
    with pytest.raises(ValueError, match="search step"):
        validate_settings({key: step})


@pytest.mark.parametrize("rotated_pairs_per_bin,unresolved,should_rotate", [
    (300, False, True), (3, False, False), (300, True, False),
])
@pytest.mark.parametrize("structures", [1, 2])
def test_joint_plane_comparison_prefers_only_supported_resolved_longer_ranges(
        rotated_pairs_per_bin, unresolved, should_rotate, structures):
    from supervario.variography import fit_axis_assignments, refine_plane
    pairs = plane_compromise_pairs(rotated_pairs_per_bin, unresolved)
    basis = np.eye(3)
    experiments = [experiment(pairs, basis[:, i], 5, 20, 5) for i in range(3)]
    baseline = fit_axis_assignments(experiments, 1, basis_rotation(basis), "grade", "A", 5, 100, structures, 10)
    selected = refine_plane(pairs, baseline, 1, "grade", "A", 5, 20, 5, 100, structures, 10)
    model, info, exps = selected
    diagnostic = info["plane_refinement"]
    assert (diagnostic["status"] == "rotated_frame_selected") == should_rotate
    final_basis = rotation_basis(model["structures"][0]["anisotropy"]["rotation"])
    assert abs(final_basis[:, 2] @ basis[:, 2]) == pytest.approx(1)
    np.testing.assert_allclose(final_basis.T @ final_basis, np.eye(3), atol=1e-12)
    assert len(model["structures"]) == structures
    for structure in model["structures"]:
        ranges = structure["anisotropy"]["ranges"]
        assert ranges["major"] >= ranges["semi_major"] >= ranges["minor"] > 0
    if should_rotate:
        assert abs(final_basis[:, 0] @ (np.array([1, 1, 0]) / np.sqrt(2))) > 0.99
        assert diagnostic["selected_supported_plane_range"] > diagnostic["baseline_supported_plane_range"] * 1.5
        chosen = diagnostic["candidates"][diagnostic["selected_candidate_index"]]
        base = diagnostic["candidates"][0]
        assert chosen["held_out_rmse_normalised"] <= base["held_out_rmse_normalised"] + 0.02
        # The compromise is not just minimising semivariance at the nearest lag.
        assert exps[1]["gamma"][0] > experiments[1]["gamma"][0]
        assert len(diagnostic["resampled_major_vectors"]) == 4
        from supervario.variography import assess_orientation
        assessment = {"resampled_major_vectors": [[1, 0, 0]] * 4, "plane_refinement": diagnostic}
        assess_orientation(assessment, final_basis, exps, 20, 10)
        assert assessment["status"] == "provisional_supported_direction"
        assert assessment["fraction_trials_within_30_degrees"] == 1


def test_plane_refinement_rejects_long_ranges_with_deteriorating_held_out_predictions():
    from supervario.variography import fit_axis_assignments, refine_plane
    pairs = plane_compromise_pairs()
    indices = np.random.default_rng(1730).permutation(len(pairs.distance))
    held_out = indices[len(indices) * 2 // 3:]
    oblique = (abs(pairs.delta[held_out, 0]) > 0) & (abs(pairs.delta[held_out, 1]) > 0)
    pairs.gamma[held_out[oblique]] += 0.6
    experiments = [experiment(pairs, np.eye(3)[:, i], 5, 20, 5) for i in range(3)]
    baseline = fit_axis_assignments(experiments, 1, basis_rotation(np.eye(3)), "grade", "A", 5, 100, 1, 10)
    selected = refine_plane(pairs, baseline, 1, "grade", "A", 5, 20, 5, 100, 1, 10)
    diagnostics = selected[1]["plane_refinement"]
    assert diagnostics["status"] == "baseline_retained"
    oblique_candidates = [r for r in diagnostics["candidates"] if r["angle_degrees"] == 45]
    assert all("held_out_prediction_deteriorated" in r["rejection_reasons"] for r in oblique_candidates)


def test_supported_plane_range_ignores_negligible_or_unbracketed_nested_structures():
    from supervario.variography import supported_plane_range
    exps = [{"distance": list(range(5, 101, 5)), "count": [100] * 20}] * 3
    structure = {"contribution": 0.8, "anisotropy": {"ranges": dict(zip(AXES, [60, 40, 20]))}}
    model = {"sill": 1, "structures": [structure]}
    expected = 0.8 * np.sqrt(60 * 40)
    assert supported_plane_range(model, exps, 10) == pytest.approx(expected)
    negligible = copy.deepcopy(structure)
    negligible["contribution"] = 0.05
    model["structures"].append(negligible)
    assert supported_plane_range(model, exps, 10) == pytest.approx(expected)
    unresolved = copy.deepcopy(structure)
    unresolved["contribution"] = 0.1
    unresolved["anisotropy"]["ranges"]["major"] = 100
    model["structures"].append(unresolved)
    assert supported_plane_range(model, exps, 10) == pytest.approx(expected)
    exps = [dict(e, count=[100] * 5 + [0] * 15) for e in exps]
    assert supported_plane_range(model, exps, 10) == 0


def test_plane_comparison_is_reproducible_and_preserves_baseline_on_nonconvergence(monkeypatch):
    from supervario import variography
    pairs = plane_compromise_pairs()
    experiments = [experiment(pairs, np.eye(3)[:, i], 5, 20, 5) for i in range(3)]
    baseline = variography.fit_axis_assignments(
        experiments, 1, basis_rotation(np.eye(3)), "grade", "A", 5, 100, 1, 10)
    first = variography.refine_plane(pairs, baseline, 1, "grade", "A", 5, 20, 5, 100, 1, 10)
    second = variography.refine_plane(pairs, baseline, 1, "grade", "A", 5, 20, 5, 100, 1, 10)
    assert first == second

    def fail_fit(*args):
        raise variography.FitConvergenceError("No convergence")

    monkeypatch.setattr(variography, "fit", fail_fit)
    logs = []
    fallback = variography.refine_plane(
        pairs, baseline, 1, "grade", "A", 5, 20, 5, 100, 1, 10, log=logs.append)
    assert fallback[0] == baseline[0]
    assert fallback[1]["plane_refinement"]["status"] == "baseline_validation_support_insufficient"
    assert any("No convergence" in message for message in logs)


def test_fit_recovers_known_spherical_parameters():
    h = np.linspace(2, 160, 30)
    rotation = {"dip_direction": 0, "dip": 0, "pitch": 0}
    truth = {"nugget": 0.2, "structures": [{"contribution": 0.8, "structure_type": "spherical",
             "anisotropy": {"ranges": dict(zip(("major", "semi_major", "minor"), [80, 40, 20]))}}]}
    experiments = [{"distance": h.tolist(), "gamma": evaluate(truth, h, axis).tolist(), "count": [100] * len(h)}
                   for axis in ("major", "semi_major", "minor")]
    model, info = fit(experiments, 1, rotation, "grade", "A", 5, 160, 1, 10)
    assert info["converged"]
    assert model["nugget"] == pytest.approx(0.2, abs=1e-5)
    assert model["structures"][0]["contribution"] == pytest.approx(0.8, abs=1e-5)
    np.testing.assert_allclose(list(model["structures"][0]["anisotropy"]["ranges"].values()), [80, 40, 20], atol=1e-3)


def test_nested_fit_recovers_known_curves():
    h = np.linspace(1, 140, 40)
    truth = {"nugget": 0.15, "structures": [
        {"contribution": 0.35, "structure_type": "spherical", "anisotropy": {
            "ranges": dict(zip(("major", "semi_major", "minor"), [30, 20, 10]))}},
        {"contribution": 0.5, "structure_type": "spherical", "anisotropy": {
            "ranges": dict(zip(("major", "semi_major", "minor"), [100, 70, 40]))}},
    ]}
    experiments = [{"distance": h.tolist(), "gamma": evaluate(truth, h, axis).tolist(), "count": [200] * len(h)}
                   for axis in ("major", "semi_major", "minor")]
    model, info = fit(experiments, 1, {"dip_direction": 0, "dip": 0, "pitch": 0}, "grade", "A", 4, 140, 2, 10)
    assert len(model["structures"]) == 2
    assert info["weighted_rmse_normalised"] < 1e-5
    for axis in ("major", "semi_major", "minor"):
        np.testing.assert_allclose(evaluate(model, h, axis), evaluate(truth, h, axis), atol=1e-4)


def test_nested_model_evaluation_honours_independent_rotations():
    base = {"dip_direction": 0, "dip": 0, "pitch": 0}
    rotated = {"dip_direction": 0, "dip": 0, "pitch": 90}
    model = {"nugget": 0, "structures": [
        {"contribution": 0, "structure_type": "spherical", "anisotropy": {
            "ranges": {"major": 100, "semi_major": 20, "minor": 10}, "rotation": base}},
        {"contribution": 1, "structure_type": "spherical", "anisotropy": {
            "ranges": {"major": 100, "semi_major": 20, "minor": 10}, "rotation": rotated}},
    ]}
    # Rotating the second structure by 90 degrees makes the reference major
    # direction its semi-major direction, with range 20 rather than 100.
    assert evaluate(model, np.array([20]), "major")[0] == pytest.approx(1)


def test_automatic_assignment_refits_swapped_axes_without_mislabelled_experiments():
    from supervario.variography import fit_axis_assignments
    h = np.linspace(1, 160, 40)
    truth = {"nugget": 0.2, "structures": [{"contribution": 0.8, "structure_type": "spherical",
             "anisotropy": {"ranges": dict(zip(AXES, [20, 80, 40]))}}]}
    experiments = [{"distance": h.tolist(), "gamma": evaluate(truth, h, axis).tolist(), "count": [100] * len(h)}
                   for axis in AXES]
    rotation = {"dip_direction": 105, "dip": 57, "pitch": 95}
    model, info, reordered = fit_axis_assignments(experiments, 1, rotation, "grade", "A", 4, 160, 1, 10)
    assert info["axis_assignments_tested"] == 6
    assert info["axis_assignment_from_search"] == ["semi_major", "minor", "major"]
    fitted_basis = rotation_basis(model["structures"][0]["anisotropy"]["rotation"])
    original_basis = rotation_basis(rotation)
    np.testing.assert_allclose(np.abs(np.diag(original_basis[:, [1, 2, 0]].T @ fitted_basis)), 1, atol=1e-10)
    assert reordered == [experiments[1], experiments[2], experiments[0]]
    np.testing.assert_allclose(list(model["structures"][0]["anisotropy"]["ranges"].values()), [80, 40, 20], atol=0.01)


def test_order_and_observed_distance_constraints_hold_for_crossed_nested_ranges():
    from supervario.variography import AXES
    h = np.linspace(2, 100, 30)
    truth = {"nugget": 0.2, "structures": [
        {"contribution": 0.4, "structure_type": "spherical",
         "anisotropy": {"ranges": dict(zip(AXES, [10, 60, 30]))}},
        {"contribution": 0.4, "structure_type": "spherical",
         "anisotropy": {"ranges": dict(zip(AXES, [500, 150, 20]))}},
    ]}
    experiments = [{"distance": h.tolist(), "gamma": evaluate(truth, h, axis).tolist(), "count": [100] * len(h)}
                   for axis in AXES]
    model, info = fit(experiments, 1, {"dip_direction": 0, "dip": 0, "pitch": 0}, "grade", "A", 4, 100, 2, 10)
    for structure in model["structures"]:
        major, semi, minor = [structure["anisotropy"]["ranges"][a] for a in AXES]
        assert 100 >= major >= semi >= minor > 0
    assert info["range_order_enforced"]
    assert info["shared_orientation"]


def test_unresolved_long_ranges_are_constrained_and_remain_flagged():
    h = np.linspace(1, 100, 30)
    experiments = [{"distance": h.tolist(), "gamma": (0.1 + h / 100).tolist(), "count": [100] * len(h)}
                   for _ in AXES]
    model, info = fit(experiments, 1, {"dip_direction": 0, "dip": 0, "pitch": 0},
                      "grade", "A", 4, 500, 1, 10)
    assert info["range_caps"] == dict.fromkeys(AXES, 100)
    assert all(r <= 100 for r in model["structures"][0]["anisotropy"]["ranges"].values())
    assert info["range_support_limited"]
    assert any("not resolved" in warning for warning in info["warnings"])


def test_variogram_rejects_constant_values():
    with pytest.raises(ValueError, match="zero"):
        analyse(np.random.default_rng(0).random((20, 3)), np.ones(20), "grade", "A", {}, lambda _: None)


@pytest.mark.parametrize("field,value", [("nugget", -1), ("sill", float("nan")), ("data_variance", 0)])
def test_lfv_rejects_invalid_parameters(field, value):
    model = copy.deepcopy(lfv.load(REFERENCE))
    model[field] = value
    with pytest.raises(ValueError):
        lfv.validate(model)
