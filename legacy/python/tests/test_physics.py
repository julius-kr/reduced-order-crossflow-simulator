import numpy as np

from crossflow_simulator.physics import (
    boundaries_from_flow_rates,
    parabolic_velocity_1d,
)


def test_parabolic_velocity_zero_at_walls():
    x = np.linspace(0.0, 1.0, 101)
    u = parabolic_velocity_1d(x)

    assert np.isclose(u[0], 0.0)
    assert np.isclose(u[-1], 0.0)
    assert u[50] == u.max()


def test_equal_flow_rates_give_unequal_layer_thicknesses():
    x = np.linspace(0.0, 1.0, 1000)

    boundaries = boundaries_from_flow_rates(x, [1, 1, 1])

    thicknesses = np.diff(boundaries)

    assert thicknesses[0] > thicknesses[1]
    assert thicknesses[2] > thicknesses[1]
    assert np.isclose(sum(thicknesses), 1.0)