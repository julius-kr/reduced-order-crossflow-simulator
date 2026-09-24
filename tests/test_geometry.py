import numpy as np
import pytest

from crossflow_simulator import geometry


def test_polygon_checks():
    assert geometry.validate_polygon([(0, 0), (1, 0), (0, 1)]) == []
    codes = {i.code for i in geometry.validate_polygon([(0, 0), (1, 0)])}
    assert "too_few_vertices" in codes
    codes = {i.code for i in geometry.validate_polygon([(0, 0), (1, 0), (1, 0), (0, 1)])}
    assert "repeated_point" in codes
    codes = {i.code for i in geometry.validate_polygon([(0, 0), (1, 1), (1, 0), (0, 1)])}
    assert "self_intersection" in codes
    codes = {i.code for i in geometry.validate_polygon([(0, 0), (np.nan, 0), (0, 1)])}
    assert "non_finite" in codes


def test_closing_vertex_is_dropped_and_winding_normalized():
    v = geometry.as_vertices([(0, 0), (0, 1), (1, 0), (0, 0)])
    assert len(v) == 3
    assert geometry.signed_area(v) < 0
    assert geometry.signed_area(geometry.ensure_ccw(v)) > 0


def test_points_in_polygon_and_concave_shapes():
    concave = np.array([(0, 0), (4, 0), (4, 4), (2, 1), (0, 4)], dtype=float)
    inside = geometry.points_in_polygon(np.array([1.0, 2.0, 3.0]), np.array([0.5, 2.0, 0.5]), concave)
    assert inside.tolist() == [True, False, True]


def test_raster_fraction_matches_area():
    tri = np.array([(0.1, 0.1), (0.9, 0.1), (0.1, 0.7)])
    edges = np.linspace(0, 1, 101)
    fraction = geometry.raster_fraction(tri, edges, edges, subsamples=8)
    area = fraction.sum() * (edges[1] - edges[0]) ** 2
    assert area == pytest.approx(0.5 * 0.8 * 0.6, rel=2e-3)


def test_directed_edges_have_outward_normals():
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for edge in geometry.directed_edges(square):
        midpoint = 0.5 * (np.array(edge["start"]) + np.array(edge["end"]))
        outward = midpoint + 1e-3 * np.array(edge["normal"])
        assert not geometry.points_in_polygon(np.array([outward[0]]), np.array([outward[1]]), np.array(square))[0]


def test_polygon_distance():
    a = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float)
    b = a + np.array([3.0, 0.0])
    assert geometry.polygon_distance(a, b) == pytest.approx(2.0)
    assert geometry.polygon_distance(a, a + 0.5) == 0.0


def test_stable_hash_ignores_float_noise():
    assert geometry.stable_hash({"x": 0.1 + 0.2}) == geometry.stable_hash({"x": 0.3})
    assert geometry.stable_hash({"x": 0.3}) != geometry.stable_hash({"x": 0.31})
