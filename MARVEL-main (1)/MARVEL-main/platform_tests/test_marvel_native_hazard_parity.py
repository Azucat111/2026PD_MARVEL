"""MARVEL-native hazard/Safety parity with the frozen Phase14-v9 worker.

Frozen ``MarvelGPPOTestWorker`` builds its hazard state from the
environment when ``enable_safety`` is set:

    hazards = HazardManager.project_hazards_from_environment(
        self.env, seed=hazard_seed, count=hazard_count)

with the defaults ``hazard_seed=0``, ``hazard_count=2``.  Two hazards are
drawn from the map cells reachable from the initial UAV cells, alternating
fire (growing) and collapse (static), and the ordinary-motion hazard guard
routes around ``radius + warning_margin`` (5 m).

Without that state the guard is a no-op, which is the first action
divergence of the 300-step run: at mission step 32 the target-search UAV
routes through (8,16) instead of the frozen-safe (4,16).

Frozen source: ``/home/nick/MARVEL`` @ ``e867117``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime

EXTENDED_SCENARIO = ROOT / "configs" / "scenarios" / "baseline_maps_test.yaml"
NATIVE_SCENARIO = ROOT / "configs" / "scenarios" / "marvel_native_test.yaml"

SEED = 12345
STARTS = [[-4.0, 0.0], [-8.0, 4.0], [4.0, -4.0], [0.0, 0.0]]

# Frozen hazards for maps_test/1.png, episode 0, these starts.
FROZEN_HAZARDS = (
    {"center": (14.0, 19.2), "radius": 3.0, "growth": 0.08,
     "max_radius": 8.0, "active_from": 20},
    {"center": (-13.6, 2.0), "radius": 4.0, "growth": 0.0,
     "max_radius": 4.0, "active_from": 50},
)

WARNING_MARGIN = 5.0


def _native_runtime():
    config = load_and_validate_scenario(NATIVE_SCENARIO)

    config["environment"]["geometry_mode"] = "marvel_native"
    config["environment"]["initial_headings"] = 270.0
    config["environment"]["episode_index"] = 0
    config["robots"] = [{
        "id_range": [0, 3],
        "type": "explorer",
        "team_id": 1,
        "config": {
            "fov": 120,
            "sensor_range": 10.0,
            "velocity": 1.0,
            "yaw_rate": 35,
            "initial_positions": [list(p) for p in STARTS],
        },
    }]
    config["scenario"]["random_seed"] = SEED

    np.random.seed(SEED)
    runtime = SimulationRuntime(config)
    runtime.reset()

    return runtime


def test_native_hazards_match_frozen_generation():
    """Two hazards, same geometry and activation as frozen."""

    runtime = _native_runtime()
    obstacles = runtime.obstacles.dynamic_obstacles

    assert len(obstacles) == len(FROZEN_HAZARDS)

    for obstacle, expected in zip(obstacles, FROZEN_HAZARDS):
        assert np.allclose(
            np.asarray(obstacle.position, dtype=float),
            np.asarray(expected["center"], dtype=float), atol=1e-9,
        )
        assert obstacle.initial_radius == pytest.approx(expected["radius"])
        assert obstacle.expansion_rate == pytest.approx(expected["growth"])
        assert obstacle.max_radius == pytest.approx(expected["max_radius"])
        assert int(obstacle.spawn_step) == expected["active_from"]


def test_native_hazard_clock_is_mission_steps():
    """Frozen activates on mission steps, not physics ticks."""

    runtime = _native_runtime()

    # The GPPO profile is what makes the two clocks differ.
    runtime.config["_gppo_profile"] = {"high_level_interval_steps": 10}

    from integrations.gppo.safety_layer import SharedHazardAdapter

    adapter = SharedHazardAdapter(runtime)
    fire = runtime.obstacles.dynamic_obstacles[0]

    assert adapter._hazard_step(10) == 1
    assert adapter._hazard_step(320) == 32

    # Mission step 19 is still inactive; mission step 20 activates.
    assert adapter._radius_at(fire, 19 * 10) is None
    assert adapter._radius_at(fire, 20 * 10) == pytest.approx(3.0)

    # Growth is per mission step: 3.0 + 10 * 0.08 at mission step 30.
    assert adapter._radius_at(fire, 30 * 10) == pytest.approx(3.8)


def test_step32_segment_leaves_the_warning_buffer():
    """The concrete step-32 case: (8,16) is blocked, (4,16) is not."""

    runtime = _native_runtime()
    runtime.config["_gppo_profile"] = {"high_level_interval_steps": 10}

    from integrations.gppo.safety_layer import SharedHazardAdapter

    adapter = SharedHazardAdapter(runtime)
    step = 32 * 10

    start = np.asarray([0.0, 24.0])
    unsafe = np.asarray([8.0, 16.0])
    safe = np.asarray([4.0, 16.0])

    assert adapter.segment_intersects_warning_buffer(
        start, unsafe, step=step
    ) is True
    assert adapter.segment_intersects_warning_buffer(
        start, safe, step=step
    ) is False


def test_extended_obstacles_keep_their_declared_schedule():
    """Extended mode reads obstacles on the physics clock, unchanged."""

    config = load_and_validate_scenario(EXTENDED_SCENARIO)

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    runtime.reset()

    from integrations.gppo.safety_layer import SharedHazardAdapter

    adapter = SharedHazardAdapter(runtime)

    # No native clock remap: the step is used verbatim.
    assert adapter._hazard_step(1234) == 1234
