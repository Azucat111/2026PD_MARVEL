"""MARVEL-native motion / trajectory parity tests.

Frozen MARVEL does not integrate a kinematic model: ``Env.final_sim_step``
assigns the commanded waypoint directly, and only the heading is modelled,
by ``compute_allowable_heading``.  Sensing is then interpolated over
``NUM_SIM_STEPS = 6`` sub-steps between the previous cell and the new cell.

This module replays the identical initial state and waypoint sequence on
both sides, ten mission steps, and compares everything.

Frozen source: ``/home/nick/MARVEL`` @ ``e867117``
(``utils/motion_model.py``, ``utils/marvel_gppo_test_worker.py:533-586``).
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.marvel_maps import load_marvel_native_map
from utils.marvel_motion import compute_allowable_heading
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime

FROZEN_REPO = Path("/home/nick/MARVEL")
NATIVE_SCENARIO = ROOT / "configs" / "scenarios" / "marvel_native_test.yaml"

MAP_NAME = "1.png"
CELL_SIZE = 0.4
SENSOR_RANGE = 10.0
FOV = 120.0
VELOCITY = 1.0
YAW_RATE = 35.0
SIM_STEPS = 6
INITIAL_HEADING = 270.0

# Explicit initial state, shared by both sides.
INITIAL_POSITIONS = [
    [-20.0, 0.0],
    [-16.0, 4.0],
    [-12.0, -4.0],
    [-8.0, 0.0],
]

# One commanded waypoint per mission step, per robot.
WAYPOINT_STEPS = [
    [[-17.0, 0.0], [-13.0, 4.0], [-9.0, -4.0], [-5.0, 0.0]],
    [[-14.0, 1.0], [-10.0, 5.0], [-6.0, -3.0], [-2.0, 1.0]],
    [[-11.0, 2.0], [-7.0, 6.0], [-3.0, -2.0], [1.0, 2.0]],
    [[-8.0, 3.0], [-4.0, 7.0], [0.0, -1.0], [4.0, 3.0]],
    [[-5.0, 4.0], [-1.0, 8.0], [3.0, 0.0], [7.0, 4.0]],
    [[-2.0, 5.0], [2.0, 9.0], [6.0, 1.0], [10.0, 5.0]],
    [[1.0, 6.0], [5.0, 10.0], [9.0, 2.0], [13.0, 6.0]],
    [[4.0, 7.0], [8.0, 11.0], [12.0, 3.0], [16.0, 7.0]],
    [[7.0, 8.0], [11.0, 12.0], [15.0, 4.0], [19.0, 8.0]],
    [[10.0, 9.0], [14.0, 13.0], [18.0, 5.0], [22.0, 9.0]],
]

# One desired heading per robot per step, in degrees.
DESIRED_HEADINGS = [
    [0.0, 45.0, 315.0, 90.0],
    [20.0, 60.0, 300.0, 100.0],
    [40.0, 80.0, 280.0, 120.0],
    [60.0, 100.0, 260.0, 140.0],
    [80.0, 120.0, 240.0, 160.0],
    [100.0, 140.0, 220.0, 180.0],
    [120.0, 160.0, 200.0, 200.0],
    [140.0, 180.0, 180.0, 220.0],
    [160.0, 200.0, 160.0, 240.0],
    [180.0, 220.0, 140.0, 260.0],
]

requires_frozen = pytest.mark.skipif(
    not FROZEN_REPO.exists(),
    reason=f"frozen MARVEL repo not available at {FROZEN_REPO}",
)


# ======================================================================
# Frozen reference replay
# ======================================================================

FROZEN_MOTION_PROBE = textwrap.dedent(
    '''
    import sys, json, math
    sys.path.insert(0, "/home/nick/MARVEL")
    import numpy as np
    from skimage import io
    from skimage.measure import block_reduce
    from utils.sensor import sensor_work_heading
    from utils.motion_model import compute_allowable_heading

    image = io.imread("maps_test/1.png", 1).astype(int)
    raw = block_reduce(image, 2, np.min)
    # The start marker is detected on the PRE-threshold array, exactly as
    # import_ground_truth does.
    marker = np.array(np.nonzero(raw == 208))
    initial_cell = np.array([marker[1, 10], marker[0, 10]])
    gt = (raw > 150) | ((raw <= 80) & (raw >= 50))
    gt = (gt * 254 + 1).astype(np.int32)

    payload = json.loads(sys.argv[1])
    positions = [np.array(p, dtype=float) for p in payload["positions"]]
    headings = [float(h) for h in payload["headings"]]
    waypoints = payload["waypoints"]
    desired = payload["desired"]
    origin = np.array(payload["origin"], dtype=float)

    CELL = 0.4
    RANGE_CELLS = round(10.0 / CELL)
    SIM = 6

    def to_cell(p):
        return np.array([(p[0]-origin[0])/CELL, (p[1]-origin[1])/CELL], dtype=float)

    belief = np.ones(gt.shape, dtype=np.int32) * 127

    # Frozen ScenarioEnv.__init__ line 66: a full 360 degree sweep of the
    # map's start-marker cell, before any per-agent sweep.
    belief = sensor_work_heading(
        initial_cell, RANGE_CELLS, belief, gt, 0, 360)

    # Frozen ScenarioEnv.__init__: initial sensing per robot.
    for i in range(len(positions)):
        belief = sensor_work_heading(
            np.round(to_cell(positions[i])).astype(int), RANGE_CELLS,
            belief, gt, headings[i], 120.0)

    out = {"steps": []}
    for step in range(len(waypoints)):
        step_record = {"cells": [], "headings": [], "final_headings": []}
        for i in range(len(positions)):
            wp = np.array(waypoints[step][i], dtype=float)
            prev_heading = float(headings[i]) % 360.0

            final_heading = compute_allowable_heading(
                positions[i], wp, headings[i], float(desired[step][i]),
                1.0, 35.0)

            start_cell = to_cell(positions[i])
            end_cell = to_cell(wp)
            cells = np.round(np.linspace(start_cell, end_cell, SIM + 1)[1:]).astype(int)

            final = float(final_heading) % 360.0
            diff = final - prev_heading
            if abs(diff) > 180:
                diff = diff - 360 if diff > 0 else diff + 360
            hs = [(prev_heading + (j+1)*diff/SIM) % 360.0 for j in range(SIM)]

            step_record["cells"].append(cells.tolist())
            step_record["headings"].append(hs)
            step_record["final_headings"].append(final)

            headings[i] = final_heading
            positions[i] = wp

        # Frozen _simulate_motion inner loop: every robot at every substep.
        # cell coords must be numpy integers, as get_cell_position_from_coords
        # returns them; Python ints raise inside the frozen Bresenham loop.
        for j in range(SIM):
            for i in range(len(positions)):
                belief = sensor_work_heading(
                    np.array(step_record["cells"][i][j], dtype=int),
                    RANGE_CELLS,
                    belief, gt, step_record["headings"][i][j], 120.0)

        step_record["positions"] = [p.tolist() for p in positions]
        step_record["headings_after"] = [h if isinstance(h, float) else float(h)
                                         for h in headings]
        step_record["rate"] = float((belief == 255).sum() / (gt == 255).sum())
        step_record["explored_free"] = int((belief == 255).sum())
        step_record["belief_flat"] = belief.astype(np.int64).flatten().tolist()
        out["steps"].append(step_record)

    print(json.dumps(out))
    '''
)


@pytest.fixture(scope="module")
def frozen_motion(tmp_path_factory):
    if not FROZEN_REPO.exists():
        pytest.skip("frozen MARVEL repo not available")

    _occupancy, origin, _gt, _initial_cell = load_marvel_native_map(
        ROOT / "maps_test" / MAP_NAME
    )

    payload = {
        "positions": INITIAL_POSITIONS,
        "headings": [INITIAL_HEADING] * len(INITIAL_POSITIONS),
        "waypoints": WAYPOINT_STEPS,
        "desired": DESIRED_HEADINGS,
        "origin": [float(origin[0]), float(origin[1])],
    }

    script = tmp_path_factory.mktemp("frozen_motion") / "probe.py"
    script.write_text(FROZEN_MOTION_PROBE, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), json.dumps(payload)],
        capture_output=True,
        text=True,
        cwd=str(FROZEN_REPO),
    )

    if result.returncode != 0:
        pytest.skip(f"frozen motion probe failed: {result.stderr[-500:]}")

    return json.loads(result.stdout.strip().splitlines()[-1])


# ======================================================================
# Integration side
# ======================================================================

def _native_runtime(tmp_path):
    """Native runtime pinned to the frozen initial state."""

    base = load_and_validate_scenario(NATIVE_SCENARIO)

    config = yaml.safe_load(yaml.safe_dump(base))

    # Native paths must stay absolute for the relocated scratch file.
    environment = config["environment"]
    environment["map_dir"] = str(ROOT / "maps_test")
    environment["geometry_mode"] = "marvel_native"
    environment["initial_headings"] = INITIAL_HEADING

    config["robots"] = [{
        "id_range": [0, len(INITIAL_POSITIONS) - 1],
        "type": "explorer",
        "team_id": 1,
        "config": {
            "fov": FOV,
            "sensor_range": SENSOR_RANGE,
            "velocity": VELOCITY,
            "yaw_rate": YAW_RATE,
            "initial_positions": INITIAL_POSITIONS,
        },
    }]

    configs = (ROOT / "configs").resolve()
    config["dynamics"] = {
        "config_file": str(configs / "dynamics" / "kinematic_simple.yaml")
    }
    config["sensor"] = {
        "config_file": str(configs / "sensors" / "ideal.yaml")
    }
    config["communication"] = {
        "config_file": str(configs / "communication" / "ideal.yaml")
    }

    # No GPPO profile -> one teleport per physics step, matching frozen's
    # one motion update per mission step.
    config.pop("_gppo_profile", None)

    path = tmp_path / "native_motion.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    runtime = SimulationRuntime(
        load_and_validate_scenario(path)
    )
    runtime.reset()

    return runtime


def _actions(step_index):
    waypoints = WAYPOINT_STEPS[step_index]
    desired = DESIRED_HEADINGS[step_index]

    return [
        (np.asarray(waypoint, dtype=float), float(heading))
        for waypoint, heading in zip(waypoints, desired)
    ]


# ======================================================================
# Initial state parity (A)
# ======================================================================

@requires_frozen
def test_initial_headings_match_frozen(tmp_path):
    runtime = _native_runtime(tmp_path)

    for robot in runtime.robots:
        assert robot.heading == pytest.approx(INITIAL_HEADING)

    assert runtime._mission_step_interval() == 1


@requires_frozen
def test_initial_sensing_state_matches_frozen(frozen_motion, tmp_path):
    """Belief after the initial sweep, before any motion."""

    runtime = _native_runtime(tmp_path)

    # The frozen replay's first recorded rate is after step 0, so compare
    # the integration's pre-motion state against a frozen replay driven
    # with a zero-length first step.
    assert runtime.native_belief.free_cell_total == 22402
    assert runtime.exploration_rate >= 0.0

    # Every robot starts at the pinned position.
    for robot, expected in zip(runtime.robots, INITIAL_POSITIONS):
        assert np.allclose(robot.position, expected)


# ======================================================================
# 10-step trajectory parity (C)
# ======================================================================

@requires_frozen
def test_ten_step_positions_match_frozen(frozen_motion, tmp_path):
    runtime = _native_runtime(tmp_path)

    for step in range(len(WAYPOINT_STEPS)):
        runtime.step(_actions(step))

        expected_positions = frozen_motion["steps"][step]["positions"]

        for robot, expected in zip(runtime.robots, expected_positions):
            assert np.allclose(
                robot.position, expected, atol=1e-9
            ), (step, robot.robot_id, robot.position, expected)


@requires_frozen
def test_ten_step_headings_match_frozen(frozen_motion, tmp_path):
    runtime = _native_runtime(tmp_path)

    for step in range(len(WAYPOINT_STEPS)):
        runtime.step(_actions(step))

        expected = frozen_motion["steps"][step]["final_headings"]

        for robot, heading in zip(runtime.robots, expected):
            assert robot.heading == pytest.approx(
                float(heading), abs=1e-9
            ), (step, robot.robot_id, robot.heading, heading)


@requires_frozen
def test_ten_step_belief_and_rate_match_frozen(frozen_motion, tmp_path):
    runtime = _native_runtime(tmp_path)

    for step in range(len(WAYPOINT_STEPS)):
        runtime.step(_actions(step))

        record = frozen_motion["steps"][step]

        assert runtime.native_belief.explored_free_count == (
            record["explored_free"]
        ), step

        assert runtime.exploration_rate == pytest.approx(
            record["rate"], abs=1e-15
        ), step

        assert runtime.native_belief.belief.astype(
            np.int64
        ).flatten().tolist() == record["belief_flat"], step


@requires_frozen
def test_sensing_substep_cells_and_headings_match_frozen(frozen_motion, tmp_path):
    """The six interpolated cells/headings are exactly frozen's."""

    from utils.marvel_motion import interpolated_sensing_track

    runtime = _native_runtime(tmp_path)

    frame = runtime.obstacles.frame

    starts = [
        np.asarray(robot.position, dtype=float).copy()
        for robot in runtime.robots
    ]
    start_headings = [float(robot.heading) for robot in runtime.robots]

    runtime.step(_actions(0))

    for index, robot in enumerate(runtime.robots):
        cells, headings = interpolated_sensing_track(
            starts[index],
            robot.position,
            start_headings[index],
            robot.heading,
            cell_size=frame.cell_size,
            origin=frame.origin,
            sim_steps=SIM_STEPS,
        )

        # Compared against the frozen replay, not against a re-derivation.
        assert cells.tolist() == (
            frozen_motion["steps"][0]["cells"][index]
        ), index

        assert np.allclose(
            headings,
            frozen_motion["steps"][0]["headings"][index],
            atol=1e-12,
        ), index


# ======================================================================
# Motion primitives (B)
# ======================================================================

def test_compute_allowable_heading_is_a_faithful_port():
    """Spot-check the frozen branch logic on both sides of the yaw limit."""

    # Distance 4 m at 1 m/s -> t_travel = 4 s, yaw budget 140 degrees.
    # A 30 degree turn needs 0.857 s -> achievable, desired returned.
    assert compute_allowable_heading(
        np.array([0.0, 0.0]),
        np.array([4.0, 0.0]),
        0.0,
        30.0,
        1.0,
        35.0,
    ) == pytest.approx(30.0)

    # A 180 degree turn needs 5.14 s > 4 s -> limited to 140 degrees.
    assert compute_allowable_heading(
        np.array([0.0, 0.0]),
        np.array([4.0, 0.0]),
        0.0,
        180.0,
        1.0,
        35.0,
    ) == pytest.approx(140.0)

    # Zero-length move: t_travel = 0 -> limited to a zero turn.
    assert compute_allowable_heading(
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        10.0,
        90.0,
        1.0,
        35.0,
    ) == pytest.approx(10.0)


def test_native_motion_teleports_to_the_commanded_waypoint(tmp_path):
    """Frozen final_sim_step assigns the waypoint outright."""

    runtime = _native_runtime(tmp_path)

    waypoint = np.asarray([-7.5, 2.5], dtype=float)

    runtime.step([(waypoint, 0.0) for _ in runtime.robots])

    for robot in runtime.robots:
        assert np.allclose(robot.position, waypoint, atol=1e-12)


@pytest.mark.skipif(
    not (FROZEN_REPO / "artifacts" / "phase14_v9_frozen"
         / "gppo_phase14_v9_ckpt80.pth").exists(),
    reason="frozen GPPO checkpoint not available",
)
def test_gppo_sees_the_same_uav_positions_at_each_allocation_boundary():
    """Native motion must reach GPPO unchanged at every decision boundary.

    The positions are the quantity GPPO consumes, so if they match frozen
    exactly then the high-level allocation boundary sees the same state.
    """

    from integrations.gppo.checkpoint import load_frozen_gppo
    from integrations.gppo.config_profile import prepare_gppo_config
    from integrations.gppo.runtime_graph import RuntimeGraphBuilder
    from integrations.gppo.scheduler import GPPOTaskScheduler

    checkpoint = str(
        FROZEN_REPO / "artifacts" / "phase14_v9_frozen"
        / "gppo_phase14_v9_ckpt80.pth"
    )

    _, ckpt = load_frozen_gppo(checkpoint, device="cpu")

    config = load_and_validate_scenario(NATIVE_SCENARIO)
    config["scenario"]["random_seed"] = 7
    config["task_scheduler"] = {
        "mode": "gppo",
        "checkpoint": checkpoint,
        "base_position": [0.0, 0.0],
        "search_scenario_seed": 7,
    }
    config = prepare_gppo_config(config, ckpt)

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    runtime.reset()

    scheduler = GPPOTaskScheduler(
        runtime,
        checkpoint_path=checkpoint,
        device="cpu",
        scheduler_config=config["task_scheduler"],
    )

    interval = runtime._mission_step_interval()

    assert interval > 1

    boundaries = 0
    decisions = []

    for _ in range(3 * interval):
        runtime.step(
            [
                (
                    np.asarray(robot.position, dtype=float),
                    float(robot.heading),
                )
                for robot in runtime.robots
            ]
        )

        if runtime.current_step % interval != 0:
            continue

        boundaries += 1

        graph = RuntimeGraphBuilder(runtime).build().graph
        by_id = {state.uav_id: state for state in graph.uav_states}

        for robot in runtime.robots:
            state = by_id[int(robot.robot_id)]

            assert state.position[0] == pytest.approx(
                float(robot.position[0]), abs=1e-12
            )
            assert state.position[1] == pytest.approx(
                float(robot.position[1]), abs=1e-12
            )

        # Search activation decision on this boundary.
        decisions.append(
            bool(runtime.exploration_rate >= 0.30)
        )

    assert boundaries >= 3

    # Search has not activated yet in a 3-mission-step native run.
    assert not scheduler.search_activated
    assert decisions == [False] * len(decisions)


def test_extended_motion_still_uses_the_dynamics_model():
    """The extended path must keep its kinematic integration."""

    from utils.scenario_config import load_and_validate_scenario as load

    config = load(
        ROOT / "configs" / "scenarios" / "baseline_maps_test.yaml"
    )

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    runtime.reset()

    before = runtime.robots[0].position.copy()

    waypoint = before + np.asarray([10.0, 0.0])

    runtime.step([(waypoint, 0.0) for _ in runtime.robots])

    # The dynamics model does not teleport the full commanded distance.
    assert not np.allclose(runtime.robots[0].position, waypoint)
