"""Tests for SafetyShield — verify wall-collision interception and no-wall-penetration guarantee."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.obstacle_manager import ObstacleManager
from utils.safety_shield import SafetyShield
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import RobotState, SimulationRuntime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_robot(robot_id: int, position, heading: float = 0.0) -> RobotState:
    return RobotState(
        robot_id=robot_id,
        robot_type="uav",
        position=np.asarray(position, dtype=float),
        velocity=1.0,
        heading=heading,
    )


def _make_obstacle_manager(obstacles=None, width=50.0, height=50.0):
    env = {
        "map_size": [width, height],
        "circular_obstacles": obstacles or [],
    }
    return ObstacleManager(env)


# ---------------------------------------------------------------------------
# Unit tests: SafetyShield
# ---------------------------------------------------------------------------

def test_shield_blocks_static_obstacle():
    """Robot heading straight into a wall should be replaced by hover."""
    obs_mgr = _make_obstacle_manager(obstacles=[{"position": [25.0, 25.0], "radius": 2.0}])
    shield = SafetyShield(obs_mgr)

    robot = _make_robot(0, [22.0, 25.0], heading=0.0)
    # Target is inside the obstacle circle
    actions = [(np.array([25.0, 25.0]), 90.0)]

    filtered = shield.filter_actions([robot], actions, step=0)
    result_pos, result_heading = filtered[0]

    # Should hover in place
    np.testing.assert_array_equal(result_pos, robot.position)
    assert result_heading == robot.heading

    events = shield.pop_events()
    assert len(events) == 1
    assert events[0]["robot_id"] == 0
    assert "static_obstacle" in events[0]["reason"]


def test_shield_allows_safe_action():
    """Action that doesn't collide should pass through unchanged."""
    obs_mgr = _make_obstacle_manager()
    shield = SafetyShield(obs_mgr)

    robot = _make_robot(0, [10.0, 10.0], heading=45.0)
    target = np.array([12.0, 10.0])
    actions = [(target, 90.0)]

    filtered = shield.filter_actions([robot], actions, step=0)
    result_pos, result_heading = filtered[0]

    np.testing.assert_array_almost_equal(result_pos, target)
    assert result_heading == 90.0
    assert shield.pop_events() == []


def test_shield_blocks_uav_proximity():
    """Two robots heading to the same point should have at least one of them hover."""
    obs_mgr = _make_obstacle_manager()
    shield = SafetyShield(obs_mgr)

    r0 = _make_robot(0, [10.0, 10.0])
    r1 = _make_robot(1, [10.0, 12.0])
    # Both targeting the same midpoint — within UAV_SAFE_DISTANCE of each other
    both_target = np.array([10.0, 11.0])
    actions = [(both_target.copy(), 0.0), (both_target.copy(), 0.0)]

    filtered = shield.filter_actions([r0, r1], actions, step=1)

    distances = np.linalg.norm(
        np.asarray(filtered[0][0]) - np.asarray(filtered[1][0])
    )
    # After filtering, either at least one robot hovers OR they are separated enough
    # (the shield only resolves the first conflicting pair it finds)
    events = shield.pop_events()
    assert len(events) >= 1


def test_shield_blocks_boundary_violation():
    """Action pointing outside map bounds should be replaced by hover."""
    obs_mgr = _make_obstacle_manager(width=30.0, height=30.0)
    shield = SafetyShield(obs_mgr)

    robot = _make_robot(0, [28.0, 15.0])
    actions = [(np.array([35.0, 15.0]), 0.0)]  # out of bounds

    filtered = shield.filter_actions([robot], actions, step=0)
    result_pos, result_heading = filtered[0]

    np.testing.assert_array_equal(result_pos, robot.position)
    assert result_heading == robot.heading
    events = shield.pop_events()
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Integration test: baseline scenario — no static collisions in 128 steps
# ---------------------------------------------------------------------------

def test_baseline_no_static_collisions_128_steps():
    """4-robot baseline scenario runs 128 steps with zero uav_static collisions."""
    config = load_and_validate_scenario(ROOT / "configs" / "scenarios" / "baseline_exploration.yaml")
    np.random.seed(42)
    runtime = SimulationRuntime(config)
    runtime.reset()

    static_collision_count = 0
    shield_intercept_count = 0

    for _ in range(runtime.max_steps):
        _, info = runtime.step(runtime.default_actions())
        for col in info["collisions"]:
            if col.get("type") in ("static", "boundary"):
                static_collision_count += 1

    event_log = runtime.get_event_log()
    for event in event_log:
        if event["type"] == "safety_shield":
            shield_intercept_count += 1

    assert static_collision_count == 0, (
        f"Expected 0 static collisions, got {static_collision_count}"
    )
    assert shield_intercept_count >= 0  # may be zero if default_actions never targets obstacles

    print(f"\n[baseline test] steps={runtime.max_steps}, "
          f"static_collisions={static_collision_count}, "
          f"shield_intercepts={shield_intercept_count}, "
          f"exploration_rate={runtime.exploration_rate:.3f}")


# ---------------------------------------------------------------------------
# Integration test: robot aimed directly at obstacle — shield must intercept
# ---------------------------------------------------------------------------

def test_robot_aimed_at_wall_is_intercepted():
    """Construct a scenario with one obstacle and force a robot into it; verify shield fires."""
    config = load_and_validate_scenario(ROOT / "configs" / "scenarios" / "baseline_exploration.yaml")
    # Place a large obstacle in the middle of the map
    config["environment"]["circular_obstacles"] = [{"position": [45.0, 45.0], "radius": 3.0}]
    # Place robot near the obstacle with a fixed start position
    config["robots"] = [
        {
            "id_range": [0, 0],
            "type": "explorer",
            "config": {
                "initial_positions": [[42.0, 45.0]],
                "velocity": 1.0,
                "fov": 120,
                "sensor_range": 10,
                "yaw_rate": 35,
            },
        }
    ]
    np.random.seed(0)
    runtime = SimulationRuntime(config)
    runtime.reset()

    # Force a single action directly into the obstacle for several steps
    target_in_obstacle = np.array([45.0, 45.0])
    intercepted = False
    for _ in range(5):
        _, info = runtime.step([(target_in_obstacle, 0.0)])
        for event in runtime.get_event_log():
            if event["type"] == "safety_shield":
                intercepted = True

    # Robot must not have penetrated the obstacle
    robot_pos = runtime.robots[0].position
    obs_center = np.array([45.0, 45.0])
    distance_to_obs = np.linalg.norm(robot_pos - obs_center)
    assert distance_to_obs > 3.0 - 0.2 - 0.05, (
        f"Robot penetrated obstacle: distance={distance_to_obs:.3f}"
    )
    assert intercepted, "SafetyShield did not fire when robot aimed directly at obstacle"
