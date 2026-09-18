import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.dynamics_models import KinematicWithAccel
from utils.scenario_config import load_and_validate_scenario, robot_count
from utils.simulation_runtime import SimulationRuntime


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
