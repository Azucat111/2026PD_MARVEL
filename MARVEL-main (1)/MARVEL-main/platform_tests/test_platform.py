import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.dynamics_models import KinematicWithAccel
from utils.scenario_config import load_and_validate_scenario, robot_count
from utils.simulation_runtime import SimulationRuntime
from utils.task_scheduler import TaskScheduler


def test_scenario_loads():
    config = load_and_validate_scenario(ROOT / "configs" / "scenarios" / "urban_rescue_simple.yaml")
    assert robot_count(config) == 30


def test_runtime_step_and_dynamic_obstacle():
    config = load_and_validate_scenario(ROOT / "configs" / "scenarios" / "urban_rescue_simple.yaml")
    np.random.seed(7)
    runtime = SimulationRuntime(config)
    observations = runtime.reset()
    assert len(observations) == 30
    _, info = runtime.step(runtime.default_actions())
    assert "comm_topology" in info
    assert runtime.current_step == 1


def test_acceleration_model_reports_saturation():
    model = KinematicWithAccel({"v_min": 0.3, "v_max": 2.5, "a_max": 1.2, "omega_max": 45})
    _, feasibility = model.step(np.array([0.0, 0.0]), np.array([10.0, 0.0]), 0, 180, 0.3, 0.1)
    assert feasibility["turn_saturated"]


def test_runtime_scales_to_60_robots():
    config = load_and_validate_scenario(ROOT / "configs" / "scenarios" / "urban_rescue_simple.yaml")
    config["robots"] = [
        {"id_range": [1, 40], "type": "explorer", "config": {"initial_positions": "random_safe", "velocity": 1.0}},
        {"id_range": [41, 50], "type": "relay", "config": {"initial_positions": "random_safe", "velocity": 1.5}},
        {"id_range": [51, 60], "type": "rescue", "config": {"initial_positions": "random_safe", "velocity": 2.0}},
    ]
    np.random.seed(11)
    runtime = SimulationRuntime(config)
    runtime.reset()
    assert len(runtime.robots) == 60
    for _ in range(10):
        _, info = runtime.step(runtime.default_actions())
        assert info["comm_topology"].shape == (60, 60)


def test_connected_initialization_and_ratio():
    config = load_and_validate_scenario(ROOT / "configs" / "scenarios" / "urban_rescue_simple.yaml")
    config["communication"]["ensure_initial_connectivity"] = True
    np.random.seed(43)
    runtime = SimulationRuntime(config)
    runtime.reset()
    topology = runtime.comm.get_topology([robot.position for robot in runtime.robots])
    assert runtime.comm.connectivity_ratio(topology) == 1.0


def test_scheduler_overrides_active_target_search():
    config = load_and_validate_scenario(ROOT / "configs" / "scenarios" / "urban_rescue_simple.yaml")
    np.random.seed(42)
    runtime = SimulationRuntime(config)
    runtime.reset()
    runtime.tasks.tasks["T2_target_search"].status = "active"
    scheduler = TaskScheduler(runtime)
    actions = runtime.default_actions()
    updated = scheduler.apply(actions)
    target = config["tasks"][1]["params"]["targets"][0]
    assert any(np.allclose(action[0], [target["x"], target["y"]]) for action in updated)


def test_dynamic_obstacle_event_and_target_detection():
    config = load_and_validate_scenario(ROOT / "configs" / "scenarios" / "urban_rescue_simple.yaml")
    config["robots"] = [{
        "id_range": [0, 0],
        "type": "rescue",
        "team_id": 1,
        "config": {
            "fov": 120, "sensor_range": 10, "velocity": 1.0,
            "yaw_rate": 35, "initial_positions": [[30.0, 30.0]],
        },
    }]
    config["tasks"] = [{
        "task_id": "target",
        "type": "target_search",
        "priority": 1,
        "assigned_robot_types": ["rescue"],
        "params": {"targets": [{"x": 30, "y": 30, "radius": 2}]},
    }]
    config["dynamic_obstacles"] = [{
        "id": "test_fire", "type": "expanding_circle", "spawn_step": 0,
        "params": {"position": [120, 120], "initial_radius": 2,
                   "expansion_rate": 0.5, "max_radius": 5},
    }]
    np.random.seed(1)
    runtime = SimulationRuntime(config)
    runtime.reset()
    runtime.step([(np.array([30.0, 30.0]), 0.0)])
    events = runtime.get_event_log()
    assert any(event["type"] == "dynamic_obstacle_spawned" for event in events)
    assert any(event["type"] == "target_found" for event in events)
