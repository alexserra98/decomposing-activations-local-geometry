"""Interactive model selection, actual secant endpoints, and Gaussian hover scores."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2


@pytest.mark.parametrize("label_field", ["year", "fahrenheit"])
def test_plot_uses_component_scores_and_switches_gaussian_groups(label_field):
    path = Path(__file__).resolve().parents[1] / "scripts/temporary/plot_label_tangent_alignment.py"
    spec = importlib.util.spec_from_file_location("plot_alignment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    diagnostics = dict(conditional_alignment=np.array([.75, np.nan]),
                       contribution=np.array([.3, 0.]), segment_mass=np.array([.4, 0.]),
                       assignment_mass=np.array([.5, .5]), rank=np.array([1, 0]),
                       identifiable=np.array([True, False]))
    gaussian = dict(centers=np.array([[0., 0., 0.], [2., 0., 0.]]),
                    covariances=np.array([np.diag([4., 1., 1.]), 2 * np.eye(3)]), valid=np.array([True, True]))
    direction = np.array([1., 1., 0.]) / np.sqrt(2)
    comparison = dict(gaussians=gaussian, components=diagnostics,
                      pc1=np.array([direction, np.zeros(3)]))
    points = np.array([[1., 2., 3.], [4., 5., 6.]])
    centroids = np.array([[2., 3., 4.], [5., 6., 7.]])
    temperature = label_field == "fahrenheit"
    figure = module.make_alignment_figure(
        points, [-.01, 9.99] if temperature else [2000, 2001],
        None if temperature else ["a", "a"], ["First prompt", "Second prompt"],
        dict(means=centroids, years=[-10, 0] if temperature else [2000, 2001], counts=[2, 3],
             block_size=10 if temperature else 1,
             endpoint_indices=np.array([[0, 1]]), segment_years=[-10] if temperature else [2000]),
        {"HDDC | K=2": comparison, "KMeans | K=2": comparison}, label_field=label_field)
    if temperature:
        assert "Temperature: -0.01 °F" in figure.data[0].text[0]
        assert "Temperature: 9.99 °F" in figure.data[0].text[1]
        assert "Template" not in figure.data[0].text[0]
        assert "[-10, 0) °F" in figure.data[1].text[0]
        assert "All prompts" in figure.data[1].text[0]
        assert "[-10, 0) °F → [0, 10) °F" in figure.data[2].text[0]
    else:
        assert "Year: 2000<br>Template: a" in figure.data[0].text[0]
        assert "Label centroid: 2000–2000" in figure.data[1].text[0]
    np.testing.assert_allclose(figure.data[1].x, centroids[:, 0])
    np.testing.assert_allclose(figure.data[2].x[:2], centroids[:, 0])
    assert figure.data[1].name == "Label centroids"
    assert "Prompts: 2" in figure.data[1].text[0]
    assert figure.data[2].name == "Centroid secants"
    surfaces = [trace for trace in figure.data if trace.type == "surface"]
    assert len(surfaces) == 4
    assert "Component 0" in surfaces[0].hovertemplate and "0.7500" in surfaces[0].hovertemplate
    assert "undefined" in surfaces[1].hovertemplate
    assert surfaces[0].visible and not surfaces[2].visible
    buttons = figure.layout.updatemenus[0].buttons
    assert len(buttons) == 2
    assert list(buttons[1].args[0]["visible"]) == [True] * 3 + [False] * 4 + [True] * 4
    axes = [trace for trace in figure.data if trace.name == "Evaluated PC1"]
    assert len(axes) == 2
    endpoints = np.column_stack((axes[0].x, axes[0].y, axes[0].z))
    assert endpoints.shape == (3, 3)  # Only the identifiable component has a line.
    delta = endpoints[1] - endpoints[0]
    np.testing.assert_allclose(delta / np.linalg.norm(delta), direction)
    np.testing.assert_allclose(endpoints[:2].mean(0), gaussian["centers"][0])
    for point in endpoints[:2]:
        np.testing.assert_allclose(point @ np.linalg.solve(gaussian["covariances"][0], point),
                                   chi2.ppf(.5, df=3))
    assert "Component 0" in axes[0].text[0] and "0.7500" in axes[0].text[0]
    # Plotly serialization validates the interactive artifact's trace schema.
    assert "Component 0" in figure.to_json()

    # A null display projection must not erase the original-space alignment.
    hidden = dict(comparison, pc1=np.zeros((2, 3)), pc1_visible=np.array([False, False]))
    full_space = module.make_alignment_figure(
        points, [2000, 2001], None, ["First", "Second"],
        dict(means=centroids, years=[2000, 2001], counts=[2, 3],
             endpoint_indices=np.array([[0, 1]])),
        {"HDDC": hidden}, alignment_space="original 4096D")
    pc1_trace = next(t for t in full_space.data if t.name == "Evaluated PC1")
    assert len(pc1_trace.x) == 0
    surface = next(t for t in full_space.data if t.type == "surface")
    assert "0.7500" in surface.hovertemplate
    assert "Alignment space: original 4096D" in surface.hovertemplate
    assert "original 4096D" in full_space.layout.updatemenus[0].buttons[0].args[1]["title"]["text"]
