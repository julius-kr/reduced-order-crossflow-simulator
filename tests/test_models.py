"""Serial-inlet junctions and asymmetric partial-height pillars through the full pipeline.

These are numerical verification tests (conservation, invariance, symmetry,
reversibility, convergence). They are not validation against external CFD or
experiment, which requires reference data this repository does not contain.
"""
import numpy as np
import pytest

from crossflow_simulator import metrics
from crossflow_simulator.config import GridSpec, InletSpec, PillarSpec
from crossflow_simulator.models.domain import build_domain, plan_groups
from crossflow_simulator.models.stokes_map import equal_property_flow
from crossflow_simulator.simulator import simulate

from conftest import UL_MIN, UM, make_config

TRIANGLE = [(0.0, 0.0), (150 * UM, 0.0), (0.0, 120 * UM)]


def junction_config(ports, rates=None, viscosity=0.02, density=1000.0, first_rate=2.0):
    cfg = make_config(rates=(first_rate,), layout="single", grid=GridSpec(output_ny=96))
    for m in cfg.materials:
        m.viscosity, m.density = viscosity, density
    for i, (y, d) in enumerate(ports):
        rate = (rates or [2.0] * len(ports))[i]
        cfg.inlets.append(InletSpec(f"p{i + 1}", "B", "port", flow_rate=rate * UL_MIN, station=500 * UM,
                                    wall="top", position=y, diameter=d))
    return cfg


def run(cfg):
    result = simulate(cfg)
    assert result.ok, result.message
    return result


def region(result, codes):
    return np.isin(result.final.codes, list(codes))


# ----------------------------------------------------------------------------- serial inlet (work package D)

def test_junction_adds_the_port_flow():
    cfg = junction_config([(150 * UM, 150 * UM)], rates=[3.0])
    domain = build_domain(cfg, plan_groups(cfg)[0])
    flow = equal_property_flow(domain, 2 * UL_MIN, [3 * UL_MIN], 0.02)
    assert flow.solution.plane_flux(0) == pytest.approx(2 * UL_MIN)
    assert flow.solution.plane_flux(-1) == pytest.approx(5 * UL_MIN)


def test_junction_material_flux_fractions():
    result = run(junction_config([(150 * UM, 150 * UM)], rates=[1.0], first_rate=3.0))
    errors = result.final.diagnostics["flux_fraction_error"]
    assert max(abs(e) for e in errors.values()) < 0.005
    assert result.final.diagnostics["flux_fraction"][2] == pytest.approx(0.25, abs=0.005)


def test_port_diameter_changes_interface_curvature():
    narrow = run(junction_config([(150 * UM, 120 * UM)]))
    wide = run(junction_config([(150 * UM, 270 * UM)]))
    section = narrow.final
    lower = {1}
    h_narrow = metrics.interface_height(narrow.final.codes, section.z, lower)
    h_wide = metrics.interface_height(wide.final.codes, section.z, lower)
    k_narrow = metrics.signed_curvature(section.y, h_narrow, smoothing=20 * UM)
    k_wide = metrics.signed_curvature(section.y, h_wide, smoothing=20 * UM)
    assert np.sign(k_narrow) == np.sign(k_wide) and abs(k_narrow) > 1.3 * abs(k_wide)


def test_equal_summed_diameters_with_different_arrangements_differ():
    """Legacy used only sum(D)/W; physically, the arrangement matters."""
    one = run(junction_config([(150 * UM, 160 * UM)], rates=[2.0]))
    two = run(junction_config([(75 * UM, 80 * UM), (225 * UM, 80 * UM)], rates=[1.0, 1.0]))
    iou = metrics.iou(region(one, {2}).astype(int), region(two, {2, 3}).astype(int), 1)
    assert iou < 0.9


def test_common_viscosity_density_and_flow_scale_do_not_change_the_map():
    base = run(junction_config([(150 * UM, 150 * UM)]))
    viscous = run(junction_config([(150 * UM, 150 * UM)], viscosity=0.08))
    dense = run(junction_config([(150 * UM, 150 * UM)], density=1300.0))
    faster = run(junction_config([(150 * UM, 150 * UM)], rates=[4.0], first_rate=4.0))
    for other in (viscous, dense, faster):
        assert np.mean(other.final.codes != base.final.codes) < 1e-3
    assert viscous.diagnostics["groups"][0]["pressure_drop_Pa"] == pytest.approx(
        4 * base.diagnostics["groups"][0]["pressure_drop_Pa"], rel=1e-6)


def test_junction_reflection_equivariance():
    left = run(junction_config([(110 * UM, 120 * UM)]))
    right = run(junction_config([(190 * UM, 120 * UM)]))
    mirrored = right.final.codes[:, ::-1]
    assert metrics.iou(left.final.codes, mirrored, 2) > 0.95


# ----------------------------------------------------------------------------- partial pillars (work package E)

def pillar_config(pillars, swap_layers=False, grid=None):
    cfg = make_config(pillars=pillars, grid=grid or GridSpec(output_ny=96))
    if swap_layers:
        cfg.inlets[0].pattern_row, cfg.inlets[1].pattern_row = 1, 0
    return cfg


def test_asymmetric_pillar_deforms_the_interface():
    result = run(pillar_config([PillarSpec("tri", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM)]))
    inlet, final = result.sections[0].codes, result.final.codes
    assert metrics.iou(inlet, final, 1) < 0.97
    assert result.diagnostics["max_flux_fraction_error"] < 0.005


def test_lateral_reflection_equivariance():
    base = run(pillar_config([PillarSpec("tri", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM)]))
    mirror = run(pillar_config([PillarSpec("tri", TRIANGLE, height=90 * UM, x=300 * UM, y=260 * UM, mirror_y=True)]))
    assert metrics.iou(base.final.codes, mirror.final.codes[:, ::-1], 1) > 0.97


def test_top_bottom_reflection_equivariance():
    bottom = run(pillar_config([PillarSpec("tri", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM, mount="bottom")]))
    top = run(pillar_config([PillarSpec("tri", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM, mount="top")],
                            swap_layers=True))
    assert metrics.iou(bottom.final.codes, top.final.codes[::-1, :], 1) > 0.97


def test_stokes_reversibility_pillar_then_mirror_image_is_identity():
    """Creeping flow is reversible: a pillar followed by its fore-aft mirror image undoes the deformation."""
    tri = PillarSpec("tri", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM)
    mirror = PillarSpec("mirror", TRIANGLE, height=90 * UM, x=1450 * UM, y=40 * UM, rotation=180.0, mirror_y=True)
    result = run(pillar_config([tri, mirror]))
    assert len(result.sections) == 3  # separate groups: composition of two maps
    inlet, middle, final = (s.codes for s in result.sections)
    assert metrics.iou(inlet, middle, 1) < 0.97
    assert metrics.iou(inlet, final, 1) > 0.97


def test_fore_aft_symmetric_pillar_gives_no_net_deformation():
    isosceles = [(0.0, 0.0), (150 * UM, 0.0), (75 * UM, 120 * UM)]
    result = run(pillar_config([PillarSpec("iso", isosceles, height=90 * UM, x=300 * UM, y=40 * UM)]))
    assert metrics.iou(result.sections[0].codes, result.final.codes, 1) > 0.97


def test_taller_pillars_deform_more():
    def displacement(height):
        result = run(pillar_config([PillarSpec("tri", TRIANGLE, height=height, x=300 * UM, y=40 * UM)]))
        return float(np.mean(result.sections[0].codes != result.final.codes))

    assert displacement(100 * UM) > displacement(50 * UM) > 0.0


def test_nearby_pillars_are_grouped_and_distant_ones_compose_accurately():
    near = pillar_config([PillarSpec("a", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM),
                          PillarSpec("b", TRIANGLE, height=90 * UM, x=600 * UM, y=140 * UM)])
    assert len(plan_groups(near)) == 1
    far = [PillarSpec("a", TRIANGLE, height=90 * UM, x=300 * UM, y=40 * UM),
           PillarSpec("b", TRIANGLE, height=90 * UM, x=1300 * UM, y=140 * UM)]
    composed = run(pillar_config(far))
    coupled_cfg = pillar_config(far)
    coupled_cfg.model.interaction_distance = 1.0  # force one coupled domain
    coupled = run(coupled_cfg)
    assert len(composed.sections) == 3 and len(coupled.sections) == 2
    assert metrics.iou(composed.final.codes, coupled.final.codes, 1) > 0.97
