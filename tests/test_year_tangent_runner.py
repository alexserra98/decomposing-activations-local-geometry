"""Standalone years evaluator loading adapters and report serialization."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from dalg.models.adaptive_q.mfa_hddc import MFA_HDDC
from dalg.models.kmeans import KMeans


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/temporary/evaluate_year_tangent_alignment.py"
    spec = importlib.util.spec_from_file_location("year_tangent_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numeric_label_mapping_preserves_integer_years_and_floors_temperature_bins(runner):
    years = [dict(labels=dict(year=y)) for y in (1000.0, 1001, 2000.0)]
    assert runner.evaluation_labels(years, "year").tolist() == [1000, 1001, 2000]
    temperatures = [dict(labels=dict(fahrenheit=t)) for t in (-30.05, -30., -.01, 0., .99, 1.)]
    assert runner.evaluation_labels(temperatures, "fahrenheit", 1.).tolist() == [-31, -30, -1, 0, 0, 1]
    with pytest.raises(ValueError, match="explicit bin_width"):
        runner.evaluation_labels(temperatures, "fahrenheit")
    with pytest.raises(ValueError, match="finite and positive"):
        runner.evaluation_labels(temperatures, "fahrenheit", 0.)


def test_geometry_adapter_preserves_native_assignments_and_rank_zero(runner):
    points = torch.tensor([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
    centers = torch.tensor([[-1.5, 0.], [1.5, 0.]])
    kmeans = KMeans.from_centroids(centers)
    kmeans.compute_pcs(points)
    hddc = MFA_HDDC(centers, rank=1, isotropic_psi=True,
                    init_directions=torch.tensor([[[1.], [0.]], [[0.], [1.]]]))
    hddc.rank_mask[1] = False
    before = {kind: {k: v.clone() for k, v in model.state_dict().items()}
              for kind, model in {"kmeans": kmeans, "hddc": hddc}.items()}
    result = runner.model_geometry_and_assignments(
        {"kmeans": kmeans, "hddc": hddc}, points, "cpu", batch_size=2)
    torch.testing.assert_close(result["kmeans"]["assignments"], kmeans.predict(points))
    torch.testing.assert_close(result["hddc"]["assignments"], hddc.responsibilities(points).argmax(1))
    assert result["hddc"]["identifiable"].tolist() == [True, False]
    assert result["hddc"]["ranks"].tolist() == [1, 0]
    for kind, model in {"kmeans": kmeans, "hddc": hddc}.items():
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[kind][key])


def test_common_segments_use_labels_and_preserve_reference_order(runner):
    references = [dict(segment_years=torch.tensor([2003, 2000]),
                       tangents=torch.tensor([[0., 1.], [1., 0.]])),
                  dict(segment_years=torch.tensor([2000, 2002, 2003]))]
    segments = runner._common_segments(references)
    assert segments.tolist() == [2000, 2003]
    torch.testing.assert_close(runner._aligned_tangents(references[0], segments), torch.eye(2))


def test_prompt_distribution_retains_fold_and_template_counts(runner):
    records = [dict(labels=dict(year=2000, template_id=t)) for t in ["a", "a", "b"]]
    rows = runner.prompt_distribution(records, (torch.tensor([0, 2]), torch.tensor([1])))
    assert rows == [dict(fold="A", year=2000, template="a", count=1),
                    dict(fold="A", year=2000, template="b", count=1),
                    dict(fold="B", year=2000, template="a", count=1)]


def test_report_serialization_retains_undefined_components_and_refuses_overwrite(runner, tmp_path):
    component = dict(rank=torch.tensor([1, 0]), identifiable=torch.tensor([True, False]),
                     assignment_mass=torch.tensor([.5, .5]), segment_mass=torch.tensor([.5, .5]),
                     contribution=torch.tensor([.5, 0.]),
                     conditional_alignment=torch.tensor([1., float("nan")]))
    method = dict(score=.5, segment_scores=torch.tensor([.5]), eligible_mass=torch.tensor([.5]),
                  conditional_score=1., unidentified_weight=.5, eligible_component_fraction=.5,
                  negative_control_score=None, matched_real_score=None,
                  exclusion_reasons=["", "zero_rank"],
                  folds=[dict(components=component), dict(components=component)])
    report = dict(segment_years=torch.tensor([2000]), reference_agreement=torch.tensor([1.]),
                  methods=dict(hddc=method, kmeans=method), comparison=dict(summary="No difference."),
                  prompt_distribution=[dict(fold="A", year=2000, template="a", count=2)])
    output = tmp_path / "report"
    runner.write_reports(report, output)
    saved = json.loads((output / "alignment.json").read_text())
    assert saved["methods"]["hddc"]["folds"][0]["components"]["conditional_alignment"] == [1., None]
    assert len(list(output.glob("*.csv"))) == 4
    assert len(list(output.glob("*.html"))) == 4
    with pytest.raises(FileExistsError):
        runner.write_reports(report, output)


def test_crossfit_uses_opposite_folds_and_shared_segments(runner, monkeypatch, tmp_path):
    years = torch.cat((torch.arange(2000, 2004).repeat_interleave(8), torch.tensor([2010, 2010])))
    n = len(years)
    values = torch.stack((torch.arange(n).double(), torch.zeros(n, dtype=torch.float64)), dim=1)
    base = dict(ranks=torch.ones(n, dtype=torch.int64), identifiable=torch.ones(n, dtype=torch.bool),
                exclusion_reasons=[""] * n, assignments=torch.arange(n))
    geometries = dict(hddc=dict(base, pc1=torch.tensor([[1., 0.]], dtype=torch.float64).repeat(n, 1)),
                      kmeans=dict(base, pc1=torch.tensor([[0., 1.]], dtype=torch.float64).repeat(n, 1)))
    references, evaluation_ids = [], []
    estimate = runner.metric.estimate_label_mean_tangents
    weights = runner.metric.segment_assignment_weights

    def track_reference(x, labels, **kwargs):
        references.append((x[:, 0].long().clone(), labels.clone()))
        return estimate(x, labels, **kwargs)

    def track_weights(labels, assignments, segments, components, **kwargs):
        evaluation_ids.append(assignments.clone())
        return weights(labels, assignments, segments, components, **kwargs)

    monkeypatch.setattr(runner.metric, "estimate_label_mean_tangents", track_reference)
    monkeypatch.setattr(runner.metric, "segment_assignment_weights", track_weights)
    templates = [str(i) for i in range(n)]
    report = runner.evaluate_crossfit(values, years, templates, geometries, seed=42)
    a, b = report["fold_rows"]
    assert report["segment_years"].tolist() == [2000, 2001, 2002]
    assert report["methods"]["hddc"]["score"] == pytest.approx(1.)
    assert report["methods"]["kmeans"]["score"] == pytest.approx(0.)
    assert report["comparison"]["mean_difference"] == pytest.approx(1.)
    torch.testing.assert_close(report["reference_agreement"], torch.ones(3, dtype=torch.float64))
    for i, fold in enumerate((a, b)):
        assert torch.equal(references[i][0], fold)
        assert torch.equal(references[i + 2][0], fold)
        assert torch.equal(references[i + 2][1].sort().values, years[fold].sort().values)
    for i, ids in enumerate(evaluation_ids):
        assert torch.equal(ids, (b, a)[i % 2])
        assert not set(ids.tolist()) & set(references[i % 2][0].tolist())
    for method in report["methods"].values():
        for fold in method["folds"]:
            assert fold["components"]["contribution"].sum() == pytest.approx(fold["score"])
    assert set(report["negative_control"]["segment_years"].tolist()) <= {2000, 2001, 2002}
    records = [dict(labels=dict(year=int(y), template_id=t)) for y, t in zip(years, templates)]
    report["prompt_distribution"] = runner.prompt_distribution(records, (a, b))
    runner.write_reports(report, tmp_path / "comparison")
    saved = json.loads((tmp_path / "comparison/alignment.json").read_text())
    assert saved["comparison"]["mean_difference"] == pytest.approx(1.)


def test_crossfit_rejects_empty_shared_population(runner):
    with pytest.raises(ValueError, match="No valid consecutive-year segments"):
        runner.evaluate_crossfit(torch.zeros((4, 2), dtype=torch.float64),
                                 torch.tensor([2000, 2000, 2001, 2001]), None, {})


def test_centroid_evaluation_uses_all_prompts_for_reference_weights_and_control(runner, tmp_path, monkeypatch):
    years = torch.tensor([2000, 2000, 2001, 2001, 2001])
    values = torch.tensor([[0., 0.], [100., 30.], [1., 0.], [100., 40.], [200., 80.]])
    templates = ["a", "b", "a", "b", "c"]
    geometry = dict(pc1=torch.eye(2, dtype=torch.float64), ranks=torch.ones(2, dtype=torch.int64),
                    identifiable=torch.ones(2, dtype=torch.bool), exclusion_reasons=["", ""],
                    assignments=torch.tensor([0, 1, 0, 1, 1]))
    shuffled_labels = []
    shuffle = runner.metric.shuffled_reference_labels

    def track_shuffle(labels, **kwargs):
        shuffled_labels.append(labels.clone())
        return shuffle(labels, **kwargs)

    monkeypatch.setattr(runner.metric, "shuffled_reference_labels", track_shuffle)
    report = runner.evaluate_template_alignment(values, years, templates,
                                                dict(hddc=geometry, kmeans=geometry), template_id="a", block_size=1)
    # Centroid difference is (151/3, 25); component weights are (5/12, 7/12).
    expected = (5 * 151**2 + 7 * 75**2) / (12 * (151**2 + 75**2))
    assert report["methods"]["hddc"]["score"] == pytest.approx(expected)
    assert report["reference"][0]["endpoint_indices"].tolist() == [[0, 1]]
    assert report["reference"][0]["counts"].tolist() == [2, 3]
    assert torch.equal(shuffled_labels[0], years)
    assert report["reference_agreement"] is None
    assert report["settings"]["mode"] == "label_centroids"
    assert "template_id" not in report["settings"]
    assert report["fold_rows"][0].tolist() == list(range(5))
    assert len(report["methods"]["hddc"]["folds"]) == 1
    records = [dict(labels=dict(year=int(y), template_id=t)) for y, t in zip(years, templates)]
    report["prompt_distribution"] = runner.prompt_distribution(records, report["fold_rows"])
    runner.write_reports(report, tmp_path / "selected")
    saved = json.loads((tmp_path / "selected/alignment.json").read_text())
    assert saved["reference_agreement"] is None
    assert not (tmp_path / "selected/reference_reliability.html").exists()
    assert "year-conditioned mean activation" in (tmp_path / "selected/summary.txt").read_text()
    other = runner.evaluate_template_alignment(values, years, templates,
                                               dict(hddc=geometry, kmeans=geometry), template_id="missing", block_size=1)
    for field in ("score", "negative_control_score"):
        assert other["methods"]["hddc"][field] == report["methods"]["hddc"][field]


@pytest.mark.parametrize("bin_width", [1., .5])
def test_default_block_evaluation_uses_all_block_assignments(runner, tmp_path, bin_width):
    years = torch.tensor([1000, 1099, 1099, 1100, 1199])
    values = torch.tensor([[0., 0.], [3., 0.], [6., 0.], [6., 4.], [6., 4.]])
    geometry = dict(pc1=torch.eye(2, dtype=torch.float64), ranks=torch.ones(2, dtype=torch.int64),
                    identifiable=torch.ones(2, dtype=torch.bool), exclusion_reasons=["", ""],
                    assignments=torch.tensor([0, 1, 1, 0, 1]))
    report = runner.evaluate_template_alignment(values, years, ["a"] * 5,
                                                dict(hddc=geometry, kmeans=geometry), template_id="a")
    # Block means (3,0) -> (6,4); weights (5/12,7/12), including 1099 and 1199.
    assert report["methods"]["hddc"]["score"] == pytest.approx((5 * .36 + 7 * .64) / 12)
    assert report["reference"][0]["counts"].tolist() == [3, 2]
    assert report["settings"]["block_size"] == 100
    records = [dict(labels=dict(year=int(y), template_id="a")) for y in years]
    report["provenance"] = dict(label_field="year", bin_width=bin_width)
    report["prompt_distribution"] = runner.prompt_distribution(records, report["fold_rows"])
    runner.write_reports(report, tmp_path / "blocks")
    import csv
    with (tmp_path / "blocks/segments.csv").open() as f:
        row = next(csv.DictReader(f))
    assert float(row["segment_end"]) == 1100 * bin_width
    assert f"successive {100 * bin_width:g}-unit year blocks" in (tmp_path / "blocks/summary.txt").read_text()
    assert f"step = {100 * bin_width:g}" in (tmp_path / "blocks/alignment_by_year.html").read_text()


def test_temperature_intervals_pool_prompts_and_skip_empty_intervals(runner):
    temperatures = [-30.05, -.01, 0., 9.99, 9.99, 10., 19.99, 40.]
    records = [dict(labels=dict(fahrenheit=t)) for t in temperatures]
    labels = runner.evaluation_labels(records, "fahrenheit", bin_width=1.)
    blocks = runner.metric.label_block_starts(labels, 10)
    assert blocks.tolist() == [-40, -10, 0, 0, 0, 10, 10, 40]
    values = torch.tensor([[-3., 0.], [-1., 0.], [0., 0.], [3., 0.],
                           [6., 0.], [6., 4.], [6., 4.], [9., 4.]])
    reference = runner.metric.estimate_label_tangents(values, labels, block_size=10)
    assert reference["years"].tolist() == [-40, -10, 0, 10, 40]
    assert reference["counts"].tolist() == [1, 1, 3, 2, 1]
    assert reference["segment_years"].tolist() == [-10, 0]
    assert reference["endpoint_indices"].tolist() == [[1, 2], [2, 3]]
    assert reference["exclusions"]["missing_next_year"] == 2
    torch.testing.assert_close(reference["means"][2:4], torch.tensor([[3., 0.], [6., 4.]], dtype=torch.float64))
    geometry = dict(pc1=torch.eye(2, dtype=torch.float64), ranks=torch.ones(2, dtype=torch.int64),
                    identifiable=torch.ones(2, dtype=torch.bool), exclusion_reasons=["", ""],
                    assignments=torch.tensor([0, 0, 0, 1, 1, 0, 1, 1]))
    report = runner.evaluate_template_alignment(values, labels, None,
        dict(hddc=geometry, kmeans=geometry), template_id=None, block_size=10)
    # The [0,10) -> [10,20) secant has direction (.6,.8) and weights (5/12,7/12).
    assert float(report["methods"]["hddc"]["segment_scores"][1]) == pytest.approx((5 * .36 + 7 * .64) / 12)


def test_original_space_alignment_is_preserved_by_display_projection(runner):
    from dalg.analysis.activation_projection import project_alignment_inputs

    values = torch.tensor([[0., 0., 0., 0.], [1., 0., 0., 1.]], dtype=torch.float64)
    labels = torch.tensor([1000, 1100])
    geometry = dict(pc1=torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64),
                    ranks=torch.tensor([1]), identifiable=torch.tensor([True]),
                    exclusion_reasons=[""], assignments=torch.tensor([0, 0]))
    geometries = dict(hddc=geometry, kmeans=geometry)
    original = runner.evaluate_template_alignment(values, labels, None, geometries, template_id=None)
    projection = dict(mean=torch.zeros(4, dtype=torch.float64), basis=torch.eye(4, dtype=torch.float64)[:3])
    display_values, display_geometry = project_alignment_inputs(values, geometries, projection)
    projected = runner.evaluate_template_alignment(display_values, labels, None, display_geometry, template_id=None)
    for kind in geometries:
        assert original["methods"][kind]["score"] == pytest.approx(.5)
        assert original["methods"][kind]["negative_control_score"] == pytest.approx(.5)
        assert projected["methods"][kind]["score"] == pytest.approx(1.)
        assert geometries[kind]["pc1"].shape[1] == 4
    reference = runner.metric.estimate_label_tangents(values, labels)
    display_means, _ = project_alignment_inputs(reference["means"], {}, projection)
    torch.testing.assert_close(display_means, display_values)
    assert reference["tangents"].shape[1] == 4


def test_crossfit_missing_control_population_is_reported_as_null(runner, monkeypatch):
    years = torch.tensor([2000] * 4 + [2001] * 4)
    values = torch.stack(((years - 2000).double(), torch.zeros(8, dtype=torch.float64)), dim=1)
    geometry = dict(pc1=torch.tensor([[1., 0.]], dtype=torch.float64), ranks=torch.tensor([1]),
                    identifiable=torch.tensor([True]), exclusion_reasons=[""],
                    assignments=torch.zeros(8, dtype=torch.int64))

    # A balanced permutation gives identical reference means at both shuffled years.
    def balanced_permutation(labels, *, seed):
        assert labels.tolist() == [2000, 2000, 2001, 2001]
        return labels[torch.tensor([0, 2, 1, 3])]

    monkeypatch.setattr(runner.metric, "shuffled_reference_labels", balanced_permutation)
    report = runner.evaluate_crossfit(values, years, None, dict(hddc=geometry, kmeans=geometry))
    assert report["negative_control"]["segment_years"].numel() == 0
    assert report["negative_control"]["excluded_from_primary"] == 1
    for method in report["methods"].values():
        assert method["score"] == 1
        assert method["negative_control_score"] is None
        assert method["matched_real_score"] is None
