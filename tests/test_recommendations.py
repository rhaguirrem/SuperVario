import copy

import numpy as np
import pytest

from supervario.recommendations import evidence, format_recommendations, recommend
from supervario.variography import rotation_basis


def result_with_ranges(ranges=(10, 10, 10)):
    return {
        "domain": "A", "max_distance": 1000,
        "model": {
            "content_type": "com.seequent.elements.variogram", "content_version": "2",
            "metadata": {"name": "Variogram grade A"},
            "nugget": 0.1, "sill": 1, "data_variance": 1,
            "structures": [{
                "contribution": 0.9, "structure_type": "spherical", "parameters": {},
                "anisotropy": {"ranges": dict(zip(("major", "semi_major", "minor"), ranges)),
                               "rotation": {"dip_direction": 0, "dip": 0, "pitch": 0}},
            }],
        },
    }


def test_dense_support_recommends_one_pass_excluding_self():
    points = np.random.default_rng(42).uniform(-1, 1, (30, 3))
    result = result_with_ranges()
    result["neighbourhood_evidence"] = evidence(points, points)
    advice = recommend(result)
    assert advice["recommended_pass_count"] == 1
    assert advice["passes"][0]["semi_axes"] == {"major": 5, "semi_major": 5, "minor": 5}
    assert advice["passes"][0]["full_axis_lengths"] == {"major": 10, "semi_major": 10, "minor": 10}
    assert advice["passes"][0]["sample_probe_support_fraction"] == 1
    assert advice["support_diagnostics"]["candidate_support"][0]["neighbour_count_p10_p50_p90"] == [29, 29, 29]
    guidance = "\n".join(advice["pass_guidance"])
    assert "Pass 1: 0.5x" in guidance
    assert "100.0% sample-probe support" in guidance
    assert "No additional pass is justified" in guidance
    assert "does not prove target-block coverage" in guidance


def test_leapfrog_parameters_map_tabs_axes_and_directions_without_sorting():
    import json
    result = result_with_ranges((10, 20, 3))
    rotation = {"dip_direction": 67.01, "dip": 57.36, "pitch": 114.019}
    result["model"]["structures"][0]["anisotropy"]["rotation"] = rotation
    result["neighbourhood_evidence"] = evidence(np.random.default_rng(42).random((30, 3)),
                                                np.zeros((1, 3)))
    original = copy.deepcopy(result)
    advice = recommend(result)
    mapped = advice["leapfrog_parameters"]
    assert mapped["Interpolant"]["Kriging type"] == "Ordinary"
    assert mapped["Interpolant"]["Mean value"] is None
    assert mapped["Interpolant"]["Variogram Model"] == result["model"]["metadata"]["name"]
    assert mapped["Interpolant"]["Discretisation"]["X"] is None
    assert mapped["Interpolant"]["Discretisation"]["Y"] is None
    assert mapped["Interpolant"]["Discretisation"]["Z"] is None
    for p, entry in zip(advice["passes"], mapped["passes"]):
        e = entry["Ellipsoid"]
        assert [e[k] for k in ("Maximum", "Intermed.", "Minimum")] == [
            p["semi_axes"][k] for k in ("major", "semi_major", "minor")]
        assert e["Maximum"] < e["Intermed."]
        assert (e["Dip"], e["Dip Azimuth"], e["Pitch"]) == (57.36, 67.01, 114.019)
        assert e["Variable Orientation"] is False
        assert entry["Search"]["Minimum samples"] == p["minimum_samples"] == 8
        assert entry["Search"]["Maximum samples"] == p["maximum_samples"] == 24
    assert mapped["Outputs"]["always"] == ["Value", "Status"]
    assert mapped["Outputs"]["select"] == ["NS", "MinD", "AvgD", "ND", "KM", "KV", "KE", "SoR", "Sum", "SumN"]
    assert mapped["Value Clipping"]["Clip input values"] is False
    assert mapped["Value Clipping"]["Lower bound"] is None
    search = mapped["Search"]
    assert search["Outlier Restriction"]["enabled"] is False
    assert search["Outlier Restriction"]["Distance (% of search)"] is None
    assert search["Outlier Restriction"]["Value threshold"] is None
    assert search["Sector Search"]["enabled"] is False
    assert search["Sector Search"]["Maximum samples per sector"] == 3
    assert search["Sector Search"]["Maximum empty sectors"] == 4
    assert search["Drillhole Limit"]["Maximum samples per drillhole"] is None
    assert search["Drillhole Limit"]["Apply drillhole limit per sector"] is None
    assert result == original
    json.dumps(advice, allow_nan=False)


def test_leapfrog_display_uses_tabs_and_handles_legacy_recommendation_reports():
    result = result_with_ranges()
    result["neighbourhood_evidence"] = evidence(np.zeros((1, 3)), np.zeros((1, 3)))
    advice = recommend(result)
    result["estimator_recommendations"] = advice
    text = format_recommendations(result)
    for label in ("INTERPOLANT", "VALUE CLIPPING", "ELLIPSOID AND SEARCH BY PASS", "OUTPUTS",
                  "Maximum / Intermed. / Minimum", "Dip Azimuth", "Minimum samples: 8",
                  "Maximum samples: 24", "Apply drillhole limit per sector", "SoR", "SumN"):
        assert label in text
    advice.pop("leapfrog_parameters")
    assert format_recommendations(result) == text
    advice.pop("pass_guidance")
    assert format_recommendations(result) == text


def test_leapfrog_unavailable_passes_do_not_fabricate_ellipsoid_dimensions():
    result = result_with_ranges()
    advice = recommend(result)
    assert advice["leapfrog_parameters"]["passes"] == []
    result["estimator_recommendations"] = advice
    assert "No supported pass dimensions" in format_recommendations(result)
    result["model"]["structures"][0]["structure_type"] = "linear"
    advice = recommend(result)
    assert advice["leapfrog_parameters"]["passes"] == []
    assert advice["status"] == "search_ranges_not_inferred"
    assert "cannot be assessed" in advice["pass_guidance"][0]


@pytest.mark.parametrize("status", [
    "major_direction_unresolved", "manual_direction_not_statistically_validated",
    "provisional_supported_direction",
])
def test_orientation_status_is_preserved_and_unresolved_ellipsoids_are_exploratory(status):
    points = np.random.default_rng(42).uniform(-1, 1, (30, 3))
    result = result_with_ranges()
    original_model = copy.deepcopy(result["model"])
    result["orientation_diagnostics"] = {"status": status}
    result["neighbourhood_evidence"] = evidence(points, points)
    advice = recommend(result)
    assert advice["orientation_status"] == status
    assert advice["recommended_pass_count"] == 1
    assert result["model"] == original_model
    if status == "major_direction_unresolved":
        assert advice["status"] == "exploratory_only_major_direction_unresolved"
        assert any("do not adopt them for production" in w for w in advice["warnings"])
    else:
        assert advice["status"] == "provisional_requires_target_block_validation"
    result["estimator_recommendations"] = advice
    assert advice["status"] in format_recommendations(result)


def test_duplicates_do_not_manufacture_neighbour_support():
    points = np.zeros((50, 3))
    result = result_with_ranges()
    result["neighbourhood_evidence"] = evidence(points, points)
    advice = recommend(result)
    assert advice["support_diagnostics"]["sources_checked"] == 1
    assert advice["passes"][0]["sample_probe_support_fraction"] == 0
    assert all(c["neighbour_count_p10_p50_p90"] == [0, 0, 0]
               for c in advice["support_diagnostics"]["candidate_support"])
    assert any("did not reach 90%" in warning for warning in advice["warnings"])
    assert "below the 90% goal" in "\n".join(advice["pass_guidance"])
    assert "failed the minimum incremental-support rule" in "\n".join(advice["pass_guidance"])


def test_support_matches_brute_force_rotated_ellipsoid():
    points = np.random.default_rng(5).normal(size=(60, 3)) * 8
    result = result_with_ranges((20, 10, 4))
    rotation = {"dip_direction": 105, "dip": 57, "pitch": 95}
    result["model"]["structures"][0]["anisotropy"]["rotation"] = rotation
    result["neighbourhood_evidence"] = evidence(points, points[:15])
    advice = recommend(result)
    basis = rotation_basis(rotation)
    distances = np.linalg.norm(((points[None, :, :] - points[:15, None, :]) @ basis) / [20, 10, 4], axis=2)
    for diagnostic in advice["support_diagnostics"]["candidate_support"]:
        counts = ((distances <= diagnostic["scale"]) & (distances > 0)).sum(axis=1)
        assert diagnostic["fraction_probes_with_at_least_8_other_unique_locations"] == pytest.approx(
            np.mean(counts >= 8))
    fractions = [p["sample_probe_support_fraction"] for p in advice["passes"]]
    assert fractions == sorted(fractions)
    assert 1 <= advice["recommended_pass_count"] <= 5
    assert all(p["rotation"] == rotation for p in advice["passes"])


def test_sparse_support_retains_multiple_passes_only_when_they_add_support():
    rng = np.random.default_rng(50)
    points = rng.uniform(-15, 15, (200, 3))
    result = result_with_ranges()
    result["neighbourhood_evidence"] = evidence(points, points)
    advice = recommend(result)
    assert 1 < advice["recommended_pass_count"] <= 5
    fractions = [p["sample_probe_support_fraction"] for p in advice["passes"]]
    assert all(b - a >= 0.01 for a, b in zip(fractions, fractions[1:]))
    assert all(p["apply_to"].startswith("Only blocks") for p in advice["passes"][1:])
    for p in advice["passes"][1:]:
        assert any(f"Pass {p['pass']}:" in line and "percentage-point support gain" in line
                   for line in advice["pass_guidance"])


def test_nested_reference_keeps_one_real_structure_orientation():
    result = result_with_ranges((10, 20, 30))
    other = copy.deepcopy(result["model"]["structures"][0])
    other["contribution"] = 0.5
    other["anisotropy"]["ranges"] = {"major": 100, "semi_major": 30, "minor": 40}
    other["anisotropy"]["rotation"]["dip"] = 35
    result["model"]["structures"][0]["contribution"] = 0.4
    result["model"]["structures"].append(other)
    result["neighbourhood_evidence"] = evidence(np.zeros((1, 3)), np.zeros((1, 3)))
    advice = recommend(result)
    assert advice["reference"]["structure_index"] == 1
    assert advice["reference"]["rotation"]["dip"] == 35
    assert advice["reference"]["base_semi_axes"] == other["anisotropy"]["ranges"]
    assert advice["pass_guidance"][0].startswith("Reference: Structure 2")


def test_linear_or_no_spatial_evidence_does_not_fabricate_passes():
    result = result_with_ranges()
    assert recommend(result)["status"] == "support_evidence_unavailable"
    result["model"]["structures"][0]["structure_type"] = "linear"
    advice = recommend(result)
    assert advice["status"] == "search_ranges_not_inferred"
    assert advice["passes"] == []


def test_sampling_evidence_is_bounded_reproducible_and_labelled():
    points = np.random.default_rng(1).random((6000, 3))
    first = evidence(points, points)
    assert first == evidence(points, points)
    assert len(first["source_unique_locations"]) == 5000
    assert len(first["probe_locations"]) == 1000
    result = result_with_ranges()
    result["neighbourhood_evidence"] = first
    advice = recommend(result)
    assert any("lower-bound" in w for w in advice["warnings"])


def test_observed_distance_caps_base_axes_and_flags_high_nugget():
    result = result_with_ranges((100, 200, 20))
    result["max_distance"] = 50
    result["model"]["nugget"] = 0.6
    result["model"]["structures"][0]["contribution"] = 0.4
    result["neighbourhood_evidence"] = evidence(np.zeros((1, 3)), np.zeros((1, 3)))
    advice = recommend(result)
    assert advice["reference"]["base_semi_axes"] == {"major": 50, "semi_major": 50, "minor": 20}
    assert any("Nugget" in w for w in advice["warnings"])
    result["estimator_recommendations"] = advice
    assert "NOT diameters" in format_recommendations(result)


def test_neighbourhood_evidence_must_be_finite():
    result = result_with_ranges()
    result["neighbourhood_evidence"] = {
        "source_unique_locations": [[float("nan"), 0, 0]], "probe_locations": [[0, 0, 0]],
    }
    with pytest.raises(ValueError, match="finite"):
        recommend(result)


def test_sparse_context_can_propose_more_than_five_conditionally():
    probes = np.array([[i * 1000, 0, 0] for i in range(10)], dtype=float)
    sources = []
    radii = [4, 6, 9, 14, 19, 25, 35, 60, 60, 60]
    for probe, radius in zip(probes, radii):
        for angle in np.arange(8) * np.pi / 4:
            sources.append(probe + [radius * np.cos(angle), radius * np.sin(angle), 0])
    result = result_with_ranges()
    result["neighbourhood_evidence"] = evidence(np.array(sources), probes)
    advice = recommend(result)
    assert advice["default_pass_count"] == 5
    assert advice["conditional_additional_pass_count"] == 2
    assert advice["recommended_pass_count"] == 7
    assert all("Exploratory" in p["adoption_status"] for p in advice["passes"][5:])
    assert advice["passes"][-1]["semi_axes"]["major"] == 40
    guidance = "\n".join(advice["pass_guidance"])
    assert "3x/4x extensions are conditional exploratory" in guidance
    assert "above 1x exceed" in guidance
