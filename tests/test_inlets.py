"""Explicit inlet topology and flux-constrained pattern partitions."""
import numpy as np
import pytest

from crossflow_simulator.config import FluidProperties, InitialPattern
from crossflow_simulator.inlets import AmbiguousLayoutError, infer_pattern_from_points, partition_pattern

from conftest import make_config


def grid_config(rates, partition="auto"):
    cfg = make_config(rates=rates, layout="grid", pattern=InitialPattern(layout="grid", rows=2, cols=2,
                                                                          partition=partition))
    for i, inlet in enumerate(cfg.inlets):
        inlet.pattern_row, inlet.pattern_col = divmod(i, 2)
    return cfg


def test_single_inlet_fills_the_section():
    part = partition_pattern(make_config(rates=(1.0,), layout="single"))
    assert part.kind == "single"
    assert np.all(part.evaluate(np.array([0.0, 1e-4, 3e-4]), np.array([0.0, 5e-5, 1.5e-4])) == 1)


@pytest.mark.parametrize("layout", ["layers_y", "layers_z"])
def test_unequal_layers_carry_their_flow_rates(layout):
    part = partition_pattern(make_config(rates=(1.0, 2.0, 3.0), layout=layout))
    assert part.max_fraction_error() < 1e-4
    assert np.allclose(part.target_fraction.ravel(), [1 / 6, 2 / 6, 3 / 6])


def test_every_array_inlet_keeps_its_own_flux():
    """The legacy array operator ignored individual rates; the partition must honour all of them."""
    part = partition_pattern(grid_config((1.0, 2.0, 3.0, 4.0)))
    assert part.kind == "staggered_rows"
    assert part.max_fraction_error() < 1e-4
    changed = partition_pattern(grid_config((9.0, 2.0, 3.0, 4.0)))
    assert not np.allclose(changed.row_edges[0], part.row_edges[0])


def test_separable_rates_admit_an_aligned_partition():
    part = partition_pattern(grid_config((1.0, 3.0, 2.0, 6.0)))
    assert part.kind == "aligned"
    assert part.max_fraction_error() < 1e-3


def test_forcing_an_infeasible_aligned_partition_fails_loudly():
    with pytest.raises(ValueError, match="aligned"):
        partition_pattern(grid_config((1.0, 2.0, 3.0, 4.0), partition="aligned"))


def test_viscous_layer_is_thicker_at_equal_flux():
    cfg = make_config(rates=(1.0, 1.0), layout="layers_z",
                      materials=[FluidProperties("A", 0.04), FluidProperties("B", 0.01)])
    coupled = partition_pattern(cfg)
    uncoupled = partition_pattern(cfg, viscosity_coupling=False)
    assert coupled.converged
    assert coupled.row_edges[0][1] > uncoupled.row_edges[0][1]
    assert coupled.max_fraction_error() < 1e-3


def test_legacy_style_inference_rejects_ambiguity():
    assert infer_pattern_from_points([(0, 0.5, 0.5)]) == "single"
    assert infer_pattern_from_points([(0, 0.2, 0.5), (0, 0.8, 0.5)]) == "layers_y"
    assert infer_pattern_from_points([(0, 0.5, 0.2), (0, 0.5, 0.8)]) == "layers_z"
    assert infer_pattern_from_points([(0, 0.5, 1), (1, 0.5, 1)]) == "serial_ports"
    grid = [(0, y, z) for y in (0.25, 0.75) for z in (0.25, 0.75)]
    assert infer_pattern_from_points(grid) == "grid"
    with pytest.raises(AmbiguousLayoutError):
        infer_pattern_from_points([(0, 0.2, 0.5), (1, 0.8, 0.5)])
    with pytest.raises(AmbiguousLayoutError):
        infer_pattern_from_points(grid[:3])


def test_arbitrary_stream_codes_are_preserved():
    cfg = make_config(rates=(1.0, 2.0), layout="layers_y")
    part = partition_pattern(cfg)
    assert sorted(np.unique(part.codes).tolist()) == sorted(cfg.stream_codes().values())
