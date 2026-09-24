import json

import pytest

from crossflow_simulator.config import (
    ConfigError, FluidProperties, InitialPattern, InletSpec, OperatingConditions, PillarSpec, ProjectConfig,
    validate_config,
)
from crossflow_simulator.units import format_quantity, parse_quantity

from conftest import UL_MIN, UM, make_config


def errors(cfg):
    return [i for i in validate_config(cfg) if i.level == "error"]


def test_units_parse_and_format():
    assert parse_quantity("300 um", "length") == pytest.approx(300e-6)
    assert parse_quantity("10 uL/min", "flow_rate") == pytest.approx(10e-9 / 60)
    assert parse_quantity("20 cP", "viscosity") == pytest.approx(0.02)
    assert parse_quantity(1.5e-3, "length") == 1.5e-3
    assert format_quantity(300e-6, "um", "length") == "300 um"
    with pytest.raises(ValueError):
        parse_quantity("3 furlongs", "length")


def test_json_round_trip_is_exact(examples_dir):
    for path in examples_dir.glob("*.json"):
        if path.name == "index.json":
            continue
        cfg = ProjectConfig.load(path)
        again = ProjectConfig.from_dict(json.loads(cfg.to_json()))
        assert again == cfg
        assert again.input_hash() == cfg.input_hash()


def test_string_quantities_are_converted_to_si():
    data = make_config().to_dict()
    data["channel"] = {"width": "300 um", "height": "0.15 mm"}
    cfg = ProjectConfig.from_dict(data)
    assert cfg.channel.width == pytest.approx(300e-6)
    assert cfg.channel.height == pytest.approx(150e-6)


def test_unknown_fields_and_schema_are_rejected():
    data = make_config().to_dict()
    data["channel"]["depth_m"] = 1.0
    with pytest.raises(ConfigError):
        ProjectConfig.from_dict(data)
    data = make_config().to_dict()
    data["schema"] = "crossflow-project/99"
    with pytest.raises(ConfigError):
        ProjectConfig.from_dict(data)


def test_display_settings_do_not_change_input_hash():
    cfg = make_config()
    other = cfg.copy()
    other.display.length_unit = "mm"
    other.name = "renamed"
    assert cfg.input_hash() == other.input_hash()
    other.inlets[0].flow_rate *= 2
    assert cfg.input_hash() != other.input_hash()


def test_over_specified_boundary_conditions_are_rejected():
    cfg = make_config()
    cfg.inlets[0].pressure = 1000.0
    assert any("Over-specified" in e.message for e in errors(cfg))
    cfg = make_config()
    cfg.operating = OperatingConditions(drive="pressure")
    assert any("Over-specified" in e.message for e in errors(cfg))


def test_pattern_layout_consistency():
    cfg = make_config(layout="layers_z")
    cfg.inlets[1].pattern_row = 3
    assert errors(cfg)
    single = make_config(rates=(1.0,), layout="single")
    assert not errors(single)
    two_singles = make_config(rates=(1.0, 1.0), layout="single")
    assert errors(two_singles)


def test_ports_must_fit_the_wall():
    cfg = make_config(rates=(1.0,), layout="single")
    cfg.inlets.append(InletSpec("p", "B", "port", flow_rate=UL_MIN, station=500 * UM, wall="top", position=20 * UM,
                                diameter=100 * UM))
    assert any("extends beyond" in e.message for e in errors(cfg))


def test_pillar_geometry_validation():
    tri = [(0.0, 0.0), (100 * UM, 0.0), (0.0, 80 * UM)]
    good = make_config(pillars=[PillarSpec("p", tri, height=80 * UM, x=200 * UM, y=40 * UM)])
    assert not errors(good)
    degenerate = make_config(pillars=[PillarSpec("p", [(0, 0), (1e-4, 0), (2e-4, 0)], height=80 * UM, x=2e-4, y=5e-5)])
    assert any("zero" in e.message.lower() for e in errors(degenerate))
    bowtie = [(0, 0), (1e-4, 1e-4), (1e-4, 0), (0, 1e-4)]
    crossed = make_config(pillars=[PillarSpec("p", bowtie, height=80 * UM, x=2e-4, y=5e-5)])
    assert any("intersect" in e.message for e in errors(crossed))
    outside = make_config(pillars=[PillarSpec("p", tri, height=80 * UM, x=200 * UM, y=250 * UM)])
    assert any("side walls" in e.message for e in errors(outside))
    too_tall = make_config(pillars=[PillarSpec("p", tri, height=200 * UM, x=200 * UM, y=40 * UM)])
    assert any("taller" in e.message for e in errors(too_tall))


def test_overlapping_pillars_need_a_shared_group():
    square = [(0, 0), (1e-4, 0), (1e-4, 1e-4), (0, 1e-4)]
    a = PillarSpec("a", square, height=80 * UM, x=2e-4, y=5e-5)
    b = PillarSpec("b", square, height=80 * UM, x=2.5e-4, y=5e-5)
    assert any("touch or overlap" in e.message for e in errors(make_config(pillars=[a, b])))
    a.group = b.group = "composite"
    assert not errors(make_config(pillars=[a, b]))


def test_full_blockage_is_rejected():
    wall_to_wall = [(0, 0), (5e-5, 0), (5e-5, 300 * UM), (0, 300 * UM)]
    cfg = make_config(pillars=[PillarSpec("dam", wall_to_wall, height=150 * UM, x=2e-4, y=0.0)])
    assert any("close the channel" in e.message for e in errors(cfg))


def test_unknown_material_and_duplicate_ids():
    cfg = make_config()
    cfg.inlets[1].material = "Z"
    cfg.materials.append(FluidProperties("A"))
    messages = " ".join(e.message for e in errors(cfg))
    assert "Unknown material" in messages and "Duplicate material" in messages


def test_diffusion_is_rejected_until_implemented():
    cfg = make_config()
    cfg.operating.diffusion = True
    assert any("diffusion" in e.message.lower() for e in errors(cfg))


def test_stream_codes_are_stable_positions():
    cfg = make_config(rates=(1, 2, 3), layout="layers_y")
    assert cfg.stream_codes() == {"in1": 1, "in2": 2, "in3": 3}
    cfg.inlets[1].enabled = False
    assert cfg.stream_codes() == {"in1": 1, "in2": 2, "in3": 3}


def test_pillar_footprint_transform():
    local = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    p = PillarSpec("p", local, height=1.0, x=10.0, y=5.0, rotation=90.0)
    fp = p.footprint()
    assert sorted(map(tuple, fp.round(12).tolist())) == sorted([(10.0, 5.0), (10.0, 6.0), (9.0, 5.0)])
    m = PillarSpec("m", local, height=1.0, x=0.0, y=0.0, mirror_y=True).footprint()
    assert sorted(map(tuple, m.round(12).tolist())) == sorted([(0.0, 0.0), (1.0, 0.0), (0.0, -1.0)])


def test_pattern_inlets_require_positive_rates():
    cfg = make_config()
    cfg.inlets[0].flow_rate = 0.0
    assert errors(cfg)


def test_initial_pattern_grid_requires_full_coverage():
    cfg = make_config(rates=(1, 2, 3), layout="grid", pattern=InitialPattern(layout="grid", rows=2, cols=2))
    for i, inlet in enumerate(cfg.inlets):
        inlet.pattern_row, inlet.pattern_col = divmod(i, 2)
    assert any("exactly one inlet per cell" in e.message for e in errors(cfg))
