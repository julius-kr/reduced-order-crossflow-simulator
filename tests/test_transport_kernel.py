"""The optional compiled streamline kernel must reproduce the NumPy reference tracer."""
import numpy as np
import pytest

from crossflow_simulator import transport_kernel
from crossflow_simulator.config import ProjectConfig
from crossflow_simulator.models.domain import build_domain, plan_groups
from crossflow_simulator.models.stokes_map import equal_property_flow
from crossflow_simulator.transport import trace

pytestmark = pytest.mark.skipif(not transport_kernel.AVAILABLE, reason="numba not installed")


@pytest.mark.parametrize("name", ["right_triangle_bottom", "serial_three_inlets", "two_nearby_pillars"])
def test_kernel_matches_numpy_reference(examples_dir, name):
    cfg = ProjectConfig.load(examples_dir / f"{name}.json")
    domain = build_domain(cfg, plan_groups(cfg)[0])
    rates = [cfg.inlets[0].flow_rate] + [i.flow_rate for i in cfg.inlets if i.id in domain.port_ids]
    flow = equal_property_flow(domain, rates[0], rates[1:], 0.02)
    rng = np.random.default_rng(7)
    W, H = cfg.channel.width, cfg.channel.height
    seeds = np.column_stack([np.full(1500, domain.grid.xf[-1]), rng.uniform(0, W, 1500), rng.uniform(0, H, 1500)])
    reference = trace(flow.interpolator, seeds, domain.grid.xf[0], openings=domain.opening_shapes, use_kernel=False)
    compiled = trace(flow.interpolator, seeds, domain.grid.xf[0], openings=domain.opening_shapes, use_kernel=True)
    assert np.mean(reference.status == compiled.status) > 0.998
    same = (reference.status == compiled.status) & (reference.status <= 1)
    assert np.max(np.abs(reference.end[same] - compiled.end[same])) < 1e-3 * min(W, H)
    assert np.array_equal(reference.origin[same], compiled.origin[same])
