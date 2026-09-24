import numpy as np
from crossflow_simulator.cross_section import CrossSection


def test_cross_section_shape():
    cs = CrossSection(ny=100, nz=80)
    assert cs.shape == (80, 100)


def test_cross_section_default_label():
    cs = CrossSection(ny=10, nz=10, default_label=3)
    assert np.all(cs.labels == 3)


def test_set_region():
    cs = CrossSection(ny=100, nz=100)
    cs.set_region(y_min=0.2, y_max=0.8, z_min=0.3, z_max=0.7, label=5)

    assert 5 in cs.unique_labels()
    assert 0 in cs.unique_labels()


def test_copy_is_independent():
    cs1 = CrossSection(ny=20, nz=20)
    cs1.fill(2)

    cs2 = cs1.copy()
    cs2.fill(7)

    assert np.all(cs1.labels == 2)
    assert np.all(cs2.labels == 7)