"""Label-centroid secants, numerical adjacency, and eligibility."""

import pytest
import torch

from dalg.analysis.label_tangent_alignment import estimate_label_mean_tangents, segment_assignment_weights
from dalg.analysis.label_tangent_alignment import (
    estimate_label_tangents,
    combine_crossfit_scores, component_alignment_diagnostics, make_crossfit_folds,
    reference_agreement, score_pc1_alignment, shuffled_reference_labels,
    summarize_paired_alignment,
)


def test_ten_year_centroids_pool_prompts_not_year_means():
    labels = torch.tensor([1000, 1009, 1009, 1010, 1019, 1030])
    values = torch.tensor([[0., 0.], [3., 0.], [6., 0.], [6., 4.], [6., 4.], [9., 8.]])
    result = estimate_label_tangents(values, labels, block_size=10)
    assert result["block_size"] == result["label_step"] == 10
    assert result["years"].tolist() == [1000, 1010, 1030]
    assert result["counts"].tolist() == [3, 2, 1]
    torch.testing.assert_close(result["means"], torch.tensor([[3., 0.], [6., 4.], [9., 8.]], dtype=torch.float64))
    assert result["endpoint_indices"].tolist() == [[0, 1]]
    torch.testing.assert_close(result["tangents"], torch.tensor([[.6, .8]], dtype=torch.float64))
    assert result["exclusions"]["missing_next_year"] == 1


def test_block_boundaries_use_floor_and_keep_partial_blocks():
    from dalg.analysis.label_tangent_alignment import label_block_starts
    labels = torch.tensor([-1, 0, 9, 10, 19, 20])
    assert label_block_starts(labels, 10).tolist() == [-10, 0, 0, 10, 10, 20]
    result = estimate_label_tangents(labels[:, None].double(), labels, block_size=10)
    assert result["counts"].tolist() == [1, 2, 2, 1]
    assert result["segment_years"].tolist() == [-10, 0, 10]




@pytest.mark.parametrize("template_id", ["chosen", "other", "absent", None])
def test_centroids_use_all_prompts_independently_of_template_selection(template_id):
    labels = torch.tensor([2001, 2000, 2001, 2000, 2001])
    values = torch.tensor([[6., 8.], [0., 0.], [1., 0.], [0., 0.], [2., 4.]])
    templates = ["other", "chosen", "chosen", "other", "third"]
    before = values.clone()
    result = estimate_label_tangents(values, labels, templates, template_id=template_id, block_size=1)
    assert result["counts"].tolist() == [2, 3]
    torch.testing.assert_close(result["means"], torch.tensor([[0., 0.], [3., 4.]], dtype=torch.float64))
    assert result["endpoint_indices"].tolist() == [[0, 1]]
    assert result["segment_years"].tolist() == [2000]
    torch.testing.assert_close(result["tangents"], torch.tensor([[.6, .8]], dtype=torch.float64))
    assert "reference_rows" not in result and "endpoint_rows" not in result
    torch.testing.assert_close(values, before)


def test_centroids_include_singletons_without_bridging_missing_years():
    labels = torch.tensor([2000, 2002, 2003, 2004, 2005, 2006])
    values = torch.tensor([[0.], [1.], [2.], [3.], [3.], [4.]])
    result = estimate_label_tangents(values, labels, ["a", "a", "b", "a", "a", "a"], template_id="a", block_size=1)
    assert result["means"].shape == (6, 1)
    assert result["segment_years"].tolist() == [2002, 2003, 2005]
    assert result["endpoint_indices"].tolist() == [[1, 2], [2, 3], [4, 5]]
    assert result["exclusions"] == dict(missing_next_year=1, insufficient_samples=0, degenerate_difference=1)


@pytest.mark.parametrize("labels", [[2000], [2000, 2002], [2000, 2001]])
def test_centroid_empty_segments_preserve_shapes(labels):
    result = estimate_label_tangents(torch.zeros((len(labels), 3)), torch.tensor(labels),
                                     ["a"] * len(labels), template_id="a", block_size=1)
    assert result["tangents"].shape == (0, 3)
    assert result["endpoint_indices"].shape == (0, 2)


def test_centroid_degeneracy_uses_endpoint_norms():
    for options in (dict(atol=1., rtol=0.), dict(atol=0., rtol=.01)):
        result = estimate_label_tangents(torch.tensor([[100.], [101.]]), torch.tensor([2000, 2001]),
                                        ["a", "a"], template_id="a", block_size=1, **options)
        assert result["exclusions"]["degenerate_difference"] == 1


def test_unsorted_labels_produce_group_means_and_forward_unit_secants():
    labels = torch.tensor([2001, 2000, 2001, 2000, 2001])
    values = torch.tensor([[3., 3.], [0., -1.], [3., 4.], [0., 1.], [3., 5.]])
    before = values.clone()
    result = estimate_label_mean_tangents(values, labels)
    assert result["years"].tolist() == [2000, 2001]
    assert result["counts"].tolist() == [2, 3]
    torch.testing.assert_close(result["means"], torch.tensor([[0., 0.], [3., 4.]], dtype=torch.float64))
    torch.testing.assert_close(result["tangents"], torch.tensor([[.6, .8]], dtype=torch.float64))
    assert result["segment_years"].tolist() == [2000]
    torch.testing.assert_close(values, before)
    assert labels.tolist() == [2001, 2000, 2001, 2000, 2001]


def test_exclusions_follow_adjacency_then_support_then_degeneracy():
    labels = torch.tensor([2000, 2002, 2003, 2003, 2004, 2004, 2005, 2005])
    values = torch.zeros((8, 2), dtype=torch.float64)
    values[-2:, 0] = 1
    result = estimate_label_mean_tangents(values, labels)
    assert result["segment_years"].tolist() == [2004]
    assert result["exclusions"] == dict(
        missing_next_year=1, insufficient_samples=1, degenerate_difference=1,
    )
    assert sum(result["exclusions"].values()) + len(result["segment_years"]) == len(result["years"]) - 1


@pytest.mark.parametrize("labels", [[2000], [2000, 2000], [2000, 2002], [2000, 2001]])
def test_empty_segment_population_retains_tensor_shapes(labels):
    result = estimate_label_mean_tangents(torch.zeros((len(labels), 3)), torch.tensor(labels))
    assert result["segment_years"].shape == (0,)
    assert result["segment_years"].dtype == torch.int64
    assert result["tangents"].shape == (0, 3)
    assert result["tangents"].dtype == torch.float64
    assert result["means"].shape == (len(set(labels)), 3)


def test_degeneracy_uses_absolute_and_relative_tolerances():
    labels = torch.tensor([2000, 2001])
    values = torch.tensor([[100., 0.], [101., 0.]], dtype=torch.float64)
    for options in (dict(atol=1., rtol=0.), dict(atol=0., rtol=.01)):
        result = estimate_label_mean_tangents(values, labels, min_samples=1, **options)
        assert result["exclusions"]["degenerate_difference"] == 1
    result = estimate_label_mean_tangents(values, labels, min_samples=1, atol=0., rtol=.001)
    assert result["segment_years"].tolist() == [2000]


@pytest.mark.parametrize("options", [dict(min_samples=0), dict(min_samples=1.5),
                                      dict(atol=-1.), dict(rtol=float("nan"))])
def test_invalid_thresholds_are_rejected(options):
    with pytest.raises(ValueError):
        estimate_label_mean_tangents(torch.zeros((2, 2)), torch.tensor([2000, 2001]), **options)


@pytest.mark.parametrize("values,labels", [
    (torch.zeros((0, 2)), torch.empty(0, dtype=torch.int64)),
    (torch.zeros((2, 0)), torch.tensor([2000, 2001])),
    (torch.zeros((2, 2)), torch.tensor([2000])),
    (torch.zeros((2, 2)), torch.tensor([2000., 2001.])),
    (torch.tensor([[float("nan")], [1.]]), torch.tensor([2000, 2001])),
])
def test_invalid_data_are_rejected(values, labels):
    with pytest.raises(ValueError):
        estimate_label_mean_tangents(values, labels)


@pytest.mark.parametrize("n_components", [2, 3, 8])
def test_segment_weights_count_repeated_ids_and_balance_unequal_endpoints(n_components):
    labels = torch.tensor([2000, 2000, 2001, 2001, 2001, 2001])
    assignments = torch.tensor([0, 1, 1, 1, 1, 1])
    segments = torch.tensor([2000])
    before = [t.clone() for t in (labels, assignments, segments)]
    weights = segment_assignment_weights(labels, assignments, segments, n_components)
    expected = torch.zeros((1, n_components), dtype=torch.float64)
    expected[0, :2] = torch.tensor([.25, .75])
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(weights.sum(1), torch.ones(1, dtype=torch.float64))
    for actual, original in zip((labels, assignments, segments), before):
        torch.testing.assert_close(actual, original)
    duplicated = segment_assignment_weights(
        torch.cat((labels, labels[:2])), torch.cat((assignments, assignments[:2])), segments, n_components,
    )
    torch.testing.assert_close(duplicated, weights)


def test_segment_weights_preserve_requested_order_with_missing_intermediate_years():
    labels = torch.tensor([2004, 2000, 2003, 2001, 2004])
    assignments = torch.tensor([2, 0, 1, 1, 2])
    result = segment_assignment_weights(labels, assignments, torch.tensor([2003, 2000]), 4)
    expected = torch.tensor([[0., .5, .5, 0.], [.5, .5, 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(
        segment_assignment_weights(labels.flip(0), assignments.flip(0), torch.tensor([2003, 2000]), 4), result,
    )


@pytest.mark.parametrize("segment", [1999, 2001, 2002, 2004, 2005])
def test_segment_weights_reject_missing_endpoint_without_bridging_gaps(segment):
    with pytest.raises(ValueError, match="endpoint"):
        segment_assignment_weights(torch.tensor([2000, 2001, 2003, 2004]),
                                   torch.zeros(4, dtype=torch.int64), torch.tensor([segment]), 2)


def test_segment_weights_empty_populations():
    empty = torch.empty(0, dtype=torch.int64)
    result = segment_assignment_weights(empty, empty, empty, 3)
    assert result.shape == (0, 3) and result.dtype == torch.float64
    with pytest.raises(ValueError, match="endpoint"):
        segment_assignment_weights(empty, empty, torch.tensor([2000]), 3)


@pytest.mark.parametrize("assignments,n_components", [([-1, 0], 2), ([0, 2], 2), ([0, 0], 0)])
def test_segment_weights_reject_invalid_component_ids_or_capacity(assignments, n_components):
    with pytest.raises(ValueError):
        segment_assignment_weights(torch.tensor([2000, 2001]), torch.tensor(assignments),
                                   torch.tensor([2000]), n_components)


def test_segment_weights_reject_duplicate_segment_starts():
    with pytest.raises(ValueError, match="distinct"):
        segment_assignment_weights(torch.tensor([2000, 2001]), torch.tensor([0, 0]),
                                   torch.tensor([2000, 2000]), 1)


def test_alignment_masks_undefined_pcs_without_renormalizing_primary_score():
    tangents = torch.eye(2, dtype=torch.float64)
    pc1 = torch.tensor([[1., 0.], [float("nan"), float("nan")]], dtype=torch.float64)
    weights = torch.tensor([[.25, .75], [.75, .25]], dtype=torch.float64)
    eligible = torch.tensor([True, False])
    result = score_pc1_alignment(tangents, pc1, eligible, weights)
    torch.testing.assert_close(result["segment_scores"], torch.tensor([.25, 0.], dtype=torch.float64))
    assert result["score"] == .125
    assert result["conditional_score"] == .25
    assert result["unidentified_weight"] == .5
    assert torch.isfinite(result["contributions"]).all()
    all_invalid = score_pc1_alignment(tangents, pc1, torch.tensor([False, False]), weights)
    assert all_invalid["score"] == 0
    assert all_invalid["conditional_score"] is None
    assert all_invalid["unidentified_weight"] == 1


def test_alignment_perfect_orthogonal_sign_and_rotation_invariance():
    tangents = torch.eye(2, dtype=torch.float64)
    pc1 = torch.eye(2, dtype=torch.float64)
    weights = torch.eye(2, dtype=torch.float64)
    valid = torch.tensor([True, True])
    perfect = score_pc1_alignment(tangents, pc1, valid, weights)
    assert perfect["score"] == 1
    assert score_pc1_alignment(tangents, pc1.flip(0), valid, weights)["score"] == 0
    rotation = torch.tensor([[.6, -.8], [.8, .6]], dtype=torch.float64)
    rotated = score_pc1_alignment(tangents @ rotation, -pc1 @ rotation, valid, weights)
    torch.testing.assert_close(rotated["alignment"], perfect["alignment"])
    angle = score_pc1_alignment(torch.tensor([[.6, .8]], dtype=torch.float64), pc1,
                                valid, torch.tensor([[.5, .5]], dtype=torch.float64))
    assert angle["score"] == pytest.approx(.5)


@pytest.mark.parametrize("pc1,weights", [
    ([[2., 0.]], [[1.]]),
    ([[float("nan"), 0.]], [[1.]]),
    ([[1., 0.]], [[.5]]),
    ([[1., 0.]], [[-1.]]),
])
def test_alignment_rejects_invalid_geometry_or_weights(pc1, weights):
    with pytest.raises(ValueError):
        score_pc1_alignment(torch.tensor([[1., 0.]], dtype=torch.float64),
                            torch.tensor(pc1, dtype=torch.float64), torch.tensor([True]),
                            torch.tensor(weights, dtype=torch.float64))


@pytest.mark.parametrize("seed", [0, 7, 42, 100])
@pytest.mark.parametrize("use_templates", [True, False])
def test_crossfit_partitions_balance_years_and_template_strata(seed, use_templates):
    years = torch.tensor([2000] * 7 + [2001] * 5 + [2002] + [2003] * 9)
    templates = ["a", "a", "a", "b", "c", "d", "e", "a", "a", "b", "c", "d", "a"] + ["f"] * 9
    state = torch.random.get_rng_state().clone()
    a, b = make_crossfit_folds(years, templates if use_templates else None, seed=seed)
    repeat = make_crossfit_folds(years, templates if use_templates else None, seed=seed)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(a, repeat[0]) and torch.equal(b, repeat[1])
    assert not set(a.tolist()) & set(b.tolist())
    assert torch.cat((a, b)).sort().values.tolist() == list(range(len(years)))
    assert a.tolist() == sorted(a.tolist()) and b.tolist() == sorted(b.tolist())
    assert abs(len(a) - len(b)) <= 1
    for year in years.unique().tolist():
        assert abs(int((years[a] == year).sum()) - int((years[b] == year).sum())) <= 1
        if use_templates:
            for template in set(templates):
                counts = [sum(int(years[i]) == year and templates[i] == template for i in fold.tolist())
                          for fold in (a, b)]
                assert abs(counts[0] - counts[1]) <= 1


def test_singleton_templates_are_distributed_across_folds():
    years = torch.tensor([2000] * 10 + [2001] * 8)
    templates = [str(i) for i in range(len(years))]
    a, b = make_crossfit_folds(years, templates, seed=42)
    assert len(a) == len(b) == 9
    assert (years[a] == 2000).sum() == (years[b] == 2000).sum() == 5


def test_fold_aggregation_pools_eligible_mass_and_handles_zero_mass():
    folds = [dict(segment_scores=torch.tensor([.25], dtype=torch.float64),
                  eligible_mass=torch.tensor([.25], dtype=torch.float64)),
             dict(segment_scores=torch.tensor([0.], dtype=torch.float64),
                  eligible_mass=torch.tensor([.75], dtype=torch.float64))]
    result = combine_crossfit_scores(folds)
    assert result["score"] == .125 and result["conditional_score"] == .25
    zero = dict(segment_scores=torch.zeros(1, dtype=torch.float64),
                eligible_mass=torch.zeros(1, dtype=torch.float64))
    result = combine_crossfit_scores([folds[0], zero])
    assert result["score"] == .125 and result["conditional_score"] == 1.
    assert combine_crossfit_scores([zero, zero])["conditional_score"] is None
    with pytest.raises(ValueError, match="Exactly two"):
        combine_crossfit_scores([zero])


def test_reference_agreement_is_sign_invariant_and_can_be_empty():
    a = torch.tensor([[1., 0.], [1., 0.]], dtype=torch.float64)
    b = torch.tensor([[-1., 0.], [0., 1.]], dtype=torch.float64)
    torch.testing.assert_close(reference_agreement(a, b), torch.tensor([1., 0.], dtype=torch.float64))
    assert reference_agreement(a[:0], b[:0]).shape == (0,)
    with pytest.raises(ValueError, match="unit"):
        reference_agreement(a * 2, b)


def test_shuffling_preserves_label_counts_inputs_and_global_rng():
    years = torch.tensor([2000, 2000, 2001, 2002, 2002, 2002])
    before = years.clone()
    state = torch.random.get_rng_state().clone()
    shuffled = shuffled_reference_labels(years, seed=9)
    assert torch.equal(shuffled.sort().values, years.sort().values)
    assert torch.equal(shuffled, shuffled_reference_labels(years, seed=9))
    assert torch.equal(years, before) and torch.equal(state, torch.random.get_rng_state())
    assert shuffled_reference_labels(years[:0], seed=9).shape == (0,)


def test_component_diagnostics_keep_distinct_mass_denominators_and_undefined_values():
    ids = torch.tensor([0, 0, 0, 1])
    weights = torch.tensor([[.5, .5, 0.], [.25, .75, 0.]], dtype=torch.float64)
    alignment = torch.tensor([[1., 0., 0.], [0., 0., 0.]], dtype=torch.float64)
    result = component_alignment_diagnostics(ids, weights, alignment,
        torch.tensor([True, False, True]), torch.tensor([1, 0, 1]))
    torch.testing.assert_close(result["assignment_mass"], torch.tensor([.75, .25, 0.], dtype=torch.float64))
    torch.testing.assert_close(result["segment_mass"], torch.tensor([.375, .625, 0.], dtype=torch.float64))
    torch.testing.assert_close(result["contribution"], torch.tensor([.25, 0., 0.], dtype=torch.float64))
    assert result["conditional_alignment"][0] == pytest.approx(2 / 3)
    assert torch.isnan(result["conditional_alignment"][1:]).all()


@pytest.mark.parametrize("hddc,kmeans,word", [
    ([.8, .6], [.3, .2], "consistent"),
    ([.8, .1], [.3, .2], "restricted"),
    ([.1, .2], [.3, .2], "No HDDC advantage"),
])
def test_paired_summary_reports_descriptive_outcomes_and_midpoint_median(hddc, kmeans, word):
    years = torch.tensor([2000, 2003])
    h, k = torch.tensor(hddc, dtype=torch.float64), torch.tensor(kmeans, dtype=torch.float64)
    result = summarize_paired_alignment(years, h, k)
    assert result["median_difference"] == pytest.approx(result["mean_difference"])
    assert result["positive_fraction"] + result["negative_fraction"] + result["tie_fraction"] == 1
    assert word in result["summary"]
    reverse = summarize_paired_alignment(years, k, h)
    assert result["mean_difference"] == pytest.approx(-reverse["mean_difference"])


def test_paired_summary_tie_tolerance():
    result = summarize_paired_alignment(torch.tensor([2000]), torch.tensor([.5 + 1e-13], dtype=torch.float64),
                                        torch.tensor([.5], dtype=torch.float64))
    assert result["tie_fraction"] == 1
