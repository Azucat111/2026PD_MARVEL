"""MARVEL-native parity environment tests.

Verifies the two explicit environment geometries:

* ``extended``      - the existing 500 m / 1 m team environment (unchanged);
* ``marvel_native`` - original MARVEL geometry: ``block_reduce(..., 2, np.min)``
  over a ``maps_test`` image, ``CELL_SIZE = 0.4``, and a non-zero belief
  origin ``-round(initial_cell * CELL_SIZE, 1)``.

The authoritative reference is ``/home/nick/MARVEL`` @ ``e867117``.  Anything
that needs the real frozen coordinator runs in a subprocess, because both
repos ship a top-level ``utils`` package.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.geometry import (
    GEOMETRY_MODE_EXTENDED,
    GEOMETRY_MODE_NATIVE,
    MARVEL_CELL_SIZE,
    GeometryFrame,
    resolve_geometry_mode,
)
from utils.marvel_maps import (
    FREE,
    OCCUPIED,
    import_marvel_ground_truth,
    load_marvel_native_map,
    marvel_belief_origin,
    resolve_marvel_map,
)
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime

FROZEN_REPO = Path("/home/nick/MARVEL")
FROZEN_CHECKPOINT = FROZEN_REPO / "artifacts" / "phase14_v9_frozen" / "gppo_phase14_v9_ckpt80.pth"

EXTENDED_SCENARIO = ROOT / "configs" / "scenarios" / "baseline_maps_test.yaml"
NATIVE_SCENARIO = ROOT / "configs" / "scenarios" / "marvel_native_test.yaml"

MAPS_TEST = ROOT / "maps_test"

requires_frozen = pytest.mark.skipif(
    not FROZEN_REPO.exists(),
    reason=f"frozen MARVEL repo not available at {FROZEN_REPO}",
)


# ======================================================================
# 1. Extended regression
# ======================================================================

def test_extended_scenario_geometry_is_unchanged():
    """The 500 m / 1 m environment keeps its exact dimensions."""

    config = load_and_validate_scenario(EXTENDED_SCENARIO)

    assert resolve_geometry_mode(config["environment"]) == (
        GEOMETRY_MODE_EXTENDED
    )

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    runtime.reset()

    frame = runtime.obstacles.frame

    assert frame.mode == GEOMETRY_MODE_EXTENDED
    assert frame.origin == (0.0, 0.0)
    assert frame.cell_size == 1.0
    assert runtime.obstacles.get_occupancy_grid().shape == (500, 500)
    assert frame.shape == (500, 500)
    assert frame.width_m == 500.0
    assert frame.height_m == 500.0
    assert runtime.obstacles.width == 500.0
    assert runtime.obstacles.height == 500.0
    assert runtime.free_cell_count == 250000


def test_extended_synthetic_obstacle_scenario_is_unchanged():
    config = load_and_validate_scenario(
        ROOT / "configs" / "scenarios" / "urban_rescue_simple.yaml"
    )

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    runtime.reset()

    frame = runtime.obstacles.frame

    # Declared 150 m extent backed by a 151-cell lattice, as before.
    assert frame.cell_size == 1.0
    assert frame.width_cells == 151
    assert frame.height_cells == 151
    assert frame.width_m == 150.0
    assert runtime.free_cell_count == 22500


def test_geometry_mode_is_explicit_and_validated(tmp_path):
    assert resolve_geometry_mode({}) == GEOMETRY_MODE_EXTENDED
    assert resolve_geometry_mode(
        {"geometry_mode": "extended"}
    ) == GEOMETRY_MODE_EXTENDED
    assert resolve_geometry_mode(
        {"geometry_mode": "marvel_native"}
    ) == GEOMETRY_MODE_NATIVE

    with pytest.raises(ValueError):
        resolve_geometry_mode({"geometry_mode": "native"})

    # A native scenario without map_dir is rejected outright.
    with pytest.raises(ValueError):
        load_and_validate_scenario(
            _write_scenario(
                tmp_path,
                {"geometry_mode": "marvel_native"},
            )
        )

    # An unknown mode is rejected at load time too.
    with pytest.raises(ValueError):
        load_and_validate_scenario(
            _write_scenario(
                tmp_path,
                {"geometry_mode": "native"},
                name="bad_mode",
            )
        )

    # A valid native scenario round-trips through the loader.
    loaded = load_and_validate_scenario(
        _write_scenario(
            tmp_path,
            {"geometry_mode": "marvel_native", "map_dir": str(MAPS_TEST)},
            name="good_native",
        )
    )

    assert resolve_geometry_mode(loaded["environment"]) == (
        GEOMETRY_MODE_NATIVE
    )


# ======================================================================
# 2. Native geometry
# ======================================================================

def test_native_map_geometry_matches_original_marvel():
    map_path = resolve_marvel_map(MAPS_TEST, 0)

    occupancy, origin, ground_truth = load_marvel_native_map(map_path)

    # raw 500x500 -> block_reduce(2, min) -> 250x250
    assert ground_truth.shape == (250, 250)
    assert occupancy.shape == (250, 250)
    assert set(np.unique(ground_truth)) <= {FREE, OCCUPIED}

    assert MARVEL_CELL_SIZE == 0.4

    frame = GeometryFrame.marvel_native(
        width_cells=occupancy.shape[1],
        height_cells=occupancy.shape[0],
        origin_x=origin[0],
        origin_y=origin[1],
    )

    assert frame.cell_size == 0.4
    assert frame.width_cells == 250
    assert frame.height_cells == 250
    assert frame.width_m == pytest.approx(100.0)
    assert frame.height_m == pytest.approx(100.0)


def test_native_runtime_uses_native_frame():
    config = load_and_validate_scenario(NATIVE_SCENARIO)

    assert resolve_geometry_mode(config["environment"]) == (
        GEOMETRY_MODE_NATIVE
    )

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    runtime.reset()

    frame = runtime.obstacles.frame

    assert frame.mode == GEOMETRY_MODE_NATIVE
    assert frame.cell_size == pytest.approx(MARVEL_CELL_SIZE)
    assert frame.shape == (250, 250)
    assert frame.width_m == pytest.approx(100.0, abs=0.1)
    assert frame.height_m == pytest.approx(100.0, abs=0.1)
    assert runtime.free_cell_count == 250 * 250

    # Initial UAVs live in the native world frame, not [0, 100].
    for robot in runtime.robots:
        assert frame.contains_world(robot.position), robot.position

    lower = frame.bounds_min
    assert lower[0] < 0 or lower[1] < 0


def test_native_sensor_cells_are_on_the_native_lattice():
    config = load_and_validate_scenario(NATIVE_SCENARIO)

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    runtime.reset()

    observations = runtime._get_observations()
    frame = runtime.obstacles.frame

    for observation in observations.values():
        cells = np.asarray(observation["visible_cells"], dtype=int)

        if cells.size == 0:
            continue

        cells = cells.reshape(-1, 2)

        assert cells[:, 0].max() < frame.width_cells
        assert cells[:, 1].max() < frame.height_cells
        assert cells.min() >= 0

        # Visible cells are within sensor range of the robot in metres.
        world = frame.cells_to_world(cells)
        robot_position = np.asarray(observation["position"], dtype=float)[:2]

        distances = np.linalg.norm(world - robot_position, axis=1)

        assert distances.max() <= 10.0 + 1e-9


# ======================================================================
# 3. World <-> grid round trip
# ======================================================================

def test_world_grid_round_trip_extended():
    frame = GeometryFrame.extended(
        width_cells=500, height_cells=500
    )

    for col, row in [(0, 0), (1, 7), (250, 499), (499, 0), (123, 456)]:
        world = frame.cell_to_world((col, row))
        assert frame.world_to_cell(world) == (col, row)
        assert frame.world_to_cell_floor(world) == (col, row)


def test_world_grid_round_trip_native_with_non_zero_origin():
    frame = GeometryFrame.marvel_native(
        width_cells=250,
        height_cells=250,
        origin_x=-69.2,
        origin_y=-60.0,
    )

    for col, row in [(0, 0), (1, 7), (125, 249), (249, 0), (61, 188)]:
        world = frame.cell_to_world((col, row))

        # Exact inverse: the lattice-point convention.
        assert frame.world_to_cell(world) == (col, row)

        # Areal lookup is defined on a point strictly inside the cell; a
        # lattice point sits exactly on a cell boundary, where either
        # neighbouring cell is defensible.
        inside = world + 0.5 * frame.cell_size
        assert frame.world_to_cell_floor(inside) == (col, row)

        # World coordinate is origin + index * cell_size.
        assert world[0] == pytest.approx(-69.2 + col * 0.4)
        assert world[1] == pytest.approx(-60.0 + row * 0.4)


def test_round_trip_through_runtime_frame():
    config = load_and_validate_scenario(NATIVE_SCENARIO)

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    runtime.reset()

    frame = runtime.obstacles.frame

    for col in range(0, frame.width_cells, 37):
        for row in range(0, frame.height_cells, 41):
            world = frame.cell_to_world((col, row))
            back = frame.world_to_cell(world)
            assert back == (col, row), (col, row, world, back)


def test_frame_bounds_and_containment():
    frame = GeometryFrame.marvel_native(
        width_cells=250,
        height_cells=250,
        origin_x=-69.2,
        origin_y=-60.0,
    )

    assert frame.bounds_min.tolist() == [-69.2, -60.0]
    assert frame.bounds_max[0] == pytest.approx(30.8)
    assert frame.bounds_max[1] == pytest.approx(40.0)

    assert frame.contains_cell(0, 0)
    assert frame.contains_cell(249, 249)
    assert not frame.contains_cell(250, 0)
    assert not frame.contains_cell(-1, 0)

    assert frame.contains_world((0.0, 0.0))
    assert not frame.contains_world((100.0, 0.0))

    # clip_cell clamps into range
    assert frame.clip_cell(-5, 999) == (0, 249)


def test_extended_frame_round_trip_is_legacy_identity():
    """On the extended lattice world metres ARE cell indices."""

    frame = GeometryFrame.extended(
        width_cells=500, height_cells=500
    )

    for value in (0.0, 1.0, 12.0, 250.0, 499.0):
        assert frame.world_to_cell((value, value)) == (
            int(value),
            int(value),
        )
        assert frame.cell_to_world((int(value), int(value)))[
            0
        ] == pytest.approx(value)


# ======================================================================
# 4. Upstream map parity
# ======================================================================

FROZEN_PROBE = textwrap.dedent(
    '''
    import sys, json
    sys.path.insert(0, "/home/nick/MARVEL")
    import numpy as np
    from utils.scenario_env import ScenarioEnv
    from utils.explore_search_coordinator import ExploreSearchCoordinator

    SCENARIO_SEED = int(sys.argv[1])
    SEED = int(sys.argv[2])
    EPISODE = int(sys.argv[3])

    env = ScenarioEnv(episode_index=EPISODE, fov=120.0, n_agents=4,
                      sensor_range=10.0, map_dir="maps_test",
                      scenario_seed=SCENARIO_SEED)

    coord = ExploreSearchCoordinator(
        env=env, robots=[None] * 4, task_manager=None, gppo_model=None,
        device="cpu", max_search_uavs=2, survivor_count=4, seed=SEED,
    )

    gt = np.asarray(env.ground_truth)
    out = {
        "cell_size": float(env.cell_size),
        "sensor_range": float(env.sensor_range),
        "ground_truth_shape": list(gt.shape),
        "ground_truth_free": int((gt == 255).sum()),
        "ground_truth_occupied": int((gt == 1).sum()),
        "ground_truth_checksum": int(gt.astype(np.int64).sum()),
        "ground_truth_rows": int(gt.shape[0]),
        "belief_origin": [float(env.belief_origin_x), float(env.belief_origin_y)],
        "map_name": env.map_name,
        "robot_locations": np.asarray(env.robot_locations, float).tolist(),
        "search_reach_radius": float(coord.search_reach_radius),
        "heat_target_radius": float(coord.heat_target_radius),
        "n_candidates": int(len(coord._free_world_coordinates())),
        "heat_points": [[float(h.position[0]), float(h.position[1])]
                        for h in coord.heat_points],
        "targets": [[int(t.heat_id), float(t.position[0]), float(t.position[1])]
                    for t in coord.targets],
    }
    print(json.dumps(out))
    '''
)


def _run_frozen_probe(tmp_path, scenario_seed=12345, seed=12345, episode=0):
    if not FROZEN_REPO.exists():
        pytest.skip("frozen MARVEL repo not available")

    script = tmp_path / "frozen_probe.py"
    script.write_text(FROZEN_PROBE, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), str(scenario_seed), str(seed), str(episode)],
        capture_output=True,
        text=True,
        cwd=str(FROZEN_REPO),
    )

    if result.returncode != 0:
        pytest.skip(f"frozen probe failed: {result.stderr[-400:]}")

    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def frozen_probe(tmp_path_factory):
    return _run_frozen_probe(tmp_path_factory.mktemp("frozen"))


@requires_frozen
def test_native_occupancy_matches_upstream_marvel(frozen_probe):
    """Our 0.4 m occupancy must equal the upstream Env's ground truth."""

    map_path = MAPS_TEST / frozen_probe["map_name"]

    occupancy, origin, ground_truth = load_marvel_native_map(map_path)

    assert list(ground_truth.shape) == frozen_probe["ground_truth_shape"]
    assert int((ground_truth == FREE).sum()) == frozen_probe["ground_truth_free"]
    assert int((ground_truth == OCCUPIED).sum()) == frozen_probe[
        "ground_truth_occupied"
    ]
    assert int(ground_truth.astype(np.int64).sum()) == frozen_probe[
        "ground_truth_checksum"
    ]

    # Occupancy is the inverse encoding of the same map.
    assert int((occupancy == 0).sum()) == frozen_probe["ground_truth_free"]

    assert origin[0] == pytest.approx(frozen_probe["belief_origin"][0])
    assert origin[1] == pytest.approx(frozen_probe["belief_origin"][1])

    assert frozen_probe["cell_size"] == pytest.approx(MARVEL_CELL_SIZE)
    assert frozen_probe["sensor_range"] == pytest.approx(10.0)


@requires_frozen
def test_native_belief_origin_formula(frozen_probe):
    ground_truth, initial_cell = import_marvel_ground_truth(
        MAPS_TEST / frozen_probe["map_name"]
    )

    origin = marvel_belief_origin(initial_cell)

    assert origin[0] == pytest.approx(frozen_probe["belief_origin"][0])
    assert origin[1] == pytest.approx(frozen_probe["belief_origin"][1])


# ======================================================================
# 5. Search generator parity
# ======================================================================

@requires_frozen
def test_search_generator_matches_frozen_coordinator(frozen_probe):
    """Same map, same starts, same seed -> identical scenario."""

    from integrations.gppo.search_scenario import (
        FrozenSearchRadii,
        build_search_scenario,
    )

    map_path = MAPS_TEST / frozen_probe["map_name"]

    occupancy, origin, _ = load_marvel_native_map(map_path)

    radii = FrozenSearchRadii.derive(
        sensor_range=frozen_probe["sensor_range"],
        cell_size=frozen_probe["cell_size"],
        search_service_steps=6,
    )

    assert radii.search_reach_radius == pytest.approx(
        frozen_probe["search_reach_radius"]
    )
    assert radii.heat_target_radius == pytest.approx(
        frozen_probe["heat_target_radius"]
    )

    heat_points, survivors, _ = build_search_scenario(
        occupancy_grid=occupancy,
        robot_locations=np.asarray(
            frozen_probe["robot_locations"], dtype=float
        ),
        seed=12345,
        survivor_count=4,
        radii=radii,
        origin=origin,
    )

    frozen_heat = np.asarray(frozen_probe["heat_points"], dtype=float)
    ours_heat = np.asarray(
        [point.position for point in heat_points], dtype=float
    )

    assert ours_heat.shape == frozen_heat.shape
    assert np.allclose(ours_heat, frozen_heat, atol=1e-9)

    frozen_targets = np.asarray(
        [entry[1:] for entry in frozen_probe["targets"]], dtype=float
    )
    ours_targets = np.asarray(
        [survivor.position for survivor in survivors], dtype=float
    )

    assert ours_targets.shape == frozen_targets.shape
    assert np.allclose(ours_targets, frozen_targets, atol=1e-9)

    # Heat ids follow the frozen ordering.
    assert [survivor.heat_id for survivor in survivors] == [
        int(entry[0]) for entry in frozen_probe["targets"]
    ]


@requires_frozen
def test_runtime_generated_scenario_matches_frozen_coordinator(frozen_probe):
    """The full runtime path reproduces the frozen coordinator output."""

    from integrations.gppo.search_scenario import (
        initialize_search_scenario,
    )

    config = load_and_validate_scenario(NATIVE_SCENARIO)

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    runtime.reset()

    # Drive the runtime to the exact frozen start positions.
    starts = np.asarray(frozen_probe["robot_locations"], dtype=float)

    assert len(runtime.robots) == len(starts)

    for robot, start in zip(runtime.robots, starts):
        robot.position = start.copy()

    truth = initialize_search_scenario(runtime, seed=12345)

    ours_heat = np.asarray(
        [point.position for point in truth.heat_points], dtype=float
    )
    frozen_heat = np.asarray(frozen_probe["heat_points"], dtype=float)

    assert np.allclose(ours_heat, frozen_heat, atol=1e-9)

    ours_targets = np.asarray(
        [survivor.position for survivor in truth.survivors], dtype=float
    )
    frozen_targets = np.asarray(
        [entry[1:] for entry in frozen_probe["targets"]], dtype=float
    )

    assert np.allclose(ours_targets, frozen_targets, atol=1e-9)

    # Connected candidate count matches too.
    assert int(
        (np.asarray(runtime.obstacles.get_occupancy_grid()) == 0).sum()
    ) == frozen_probe["n_candidates"]


@requires_frozen
def test_native_scenario_survivors_are_sensor_reachable(frozen_probe):
    """Every survivor stays inside heat_target_radius of its heat point."""

    from integrations.gppo.search_scenario import (
        initialize_search_scenario,
    )

    config = load_and_validate_scenario(NATIVE_SCENARIO)

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    runtime.reset()

    for robot, start in zip(
        runtime.robots,
        np.asarray(frozen_probe["robot_locations"], dtype=float),
    ):
        robot.position = start.copy()

    truth = initialize_search_scenario(runtime, seed=12345)

    by_id = {
        point.heat_id: np.asarray(point.position)
        for point in truth.heat_points
    }

    for survivor in truth.survivors:
        distance = float(
            np.linalg.norm(
                np.asarray(survivor.position) - by_id[survivor.heat_id]
            )
        )

        assert distance <= truth.radii.heat_target_radius + 1e-9


# ======================================================================
# 6. MARVEL low-level smoke (native geometry)
# ======================================================================

def _native_gppo_config(seed: int = 7):
    from integrations.gppo.config_profile import prepare_gppo_config

    config = load_and_validate_scenario(NATIVE_SCENARIO)

    config["task_scheduler"] = {
        "mode": "gppo",
        "checkpoint": str(FROZEN_CHECKPOINT),
        "base_position": [0.0, 0.0],
        "search_scenario_seed": int(seed),
    }

    config["scenario"]["random_seed"] = int(seed)

    from integrations.gppo.checkpoint import load_frozen_gppo

    _, checkpoint = load_frozen_gppo(str(FROZEN_CHECKPOINT), device="cpu")

    return prepare_gppo_config(config, checkpoint)


def test_native_marvel_low_level_smoke():
    """Real PolicyNet drives actions in native geometry."""

    from utils.policy_adapter import MARVELPolicyAdapter

    config = _native_gppo_config()

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    observations = runtime.reset()

    adapter = MARVELPolicyAdapter(runtime)
    adapter.setup()

    assert adapter._using_policy, "MARVEL PolicyNet must load to verify parity"

    # Belief map matches the native lattice: one belief cell per map cell.
    frame = runtime.obstacles.frame

    assert adapter._map_w == frame.width_cells
    assert adapter._map_h == frame.height_cells
    assert adapter._belief_origin == frame.origin

    base_actions = adapter._policy_actions(observations)
    default_actions = runtime.default_actions()

    assert len(base_actions) == len(runtime.robots)

    # PolicyNet actions, not the default fallback.
    assert any(
        not np.allclose(base[0], default[0])
        for base, default in zip(base_actions, default_actions)
    )

    for agent in adapter.agents:
        assert agent.utility is not None
        assert agent.node_coords is not None
        assert len(agent.node_coords) > 0

        # Node coordinates sit in the native world frame, not a stretched one.
        assert np.all(agent.node_coords >= frame.bounds_min - 50.0)
        assert np.all(agent.node_coords <= frame.bounds_max + 50.0)


def test_native_node_graph_has_native_scale():
    """NODE_PADDING_SIZE must not be breached by an accidental 5x stretch."""

    from utils.policy_adapter import MARVELPolicyAdapter

    config = _native_gppo_config()

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    observations = runtime.reset()

    adapter = MARVELPolicyAdapter(runtime)
    adapter.setup()
    adapter._policy_actions(observations)

    frame = runtime.obstacles.frame

    for agent in adapter.agents:
        coords = np.asarray(agent.node_coords, dtype=float)

        # Whole graph must fit the 100 m native world.
        assert coords[:, 0].max() - coords[:, 0].min() <= frame.width_m + 1e-6
        assert coords[:, 1].max() - coords[:, 1].min() <= frame.height_m + 1e-6


# ======================================================================
# 7. GPPO smoke (native geometry)
# ======================================================================

@requires_frozen
def test_native_gppo_smoke():
    from integrations.gppo.scheduler import GPPOTaskScheduler
    from integrations.gppo.search_scenario import (
        initialize_search_scenario,
    )

    config = _native_gppo_config()

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    runtime.reset()

    expectations = float(config["_gppo_profile"]["activation_threshold"])

    assert expectations == pytest.approx(0.30)
    assert config["_gppo_profile"]["geometry_mode"] == GEOMETRY_MODE_NATIVE
    assert config["_gppo_profile"]["search_cell_size"] == pytest.approx(
        MARVEL_CELL_SIZE
    )
    # Frozen source values: guaranteed hidden radius at 0.4 m cells.
    assert config["_gppo_profile"]["search_guaranteed_hidden_radius"] == (
        pytest.approx(6.1)
    )

    truth = initialize_search_scenario(runtime)

    scheduler = GPPOTaskScheduler(
        runtime,
        checkpoint_path=str(FROZEN_CHECKPOINT),
        device="cpu",
        scheduler_config=config["task_scheduler"],
    )
    scheduler.install_search_scenario(truth.public_heat_points())

    assert len(scheduler.heat_points) == 4
    assert not hasattr(scheduler, "target_to_heat_id")

    # Activate at the frozen threshold.
    runtime.tasks.tasks["T2_target_search"].status = "active"
    runtime.explored_cells = {
        (index % 250, index // 250)
        for index in range(int(0.35 * runtime.free_cell_count))
    }

    assert scheduler._maybe_activate_search()
    assert scheduler.search_activated

    stale = scheduler.tracker.positions(
        scheduler.profile.stale_age_fraction
    )
    scheduler._refresh_search(stale)

    assert len(scheduler.search_assignments) == 2

    # Hidden truth never reaches the scheduler.
    for heat_id in scheduler.search_assignments:
        assert heat_id in scheduler.heat_by_id

    assert not hasattr(scheduler, "targets")
    assert not hasattr(scheduler, "search_scenario")


@requires_frozen
def test_native_observed_graph_routing_uses_native_frame():
    """Search routing resolves one hop on the MARVEL graph in native metres."""

    from integrations.gppo.scheduler import GPPOTaskScheduler
    from integrations.gppo.search_scenario import (
        initialize_search_scenario,
    )
    from utils.policy_adapter import MARVELPolicyAdapter

    config = _native_gppo_config()

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    observations = runtime.reset()

    adapter = MARVELPolicyAdapter(runtime)
    adapter.setup()

    truth = runtime.search_scenario

    assert truth is not None

    scheduler = adapter.scheduler

    assert len(scheduler.heat_points) == 4

    runtime.tasks.tasks["T2_target_search"].status = "active"
    runtime.explored_cells = {
        (index % 250, index // 250)
        for index in range(int(0.35 * runtime.free_cell_count))
    }

    base_actions = adapter.get_actions(observations)

    assert scheduler.search_assignments

    frame = runtime.obstacles.frame
    robot_index = {
        int(robot.robot_id): index
        for index, robot in enumerate(runtime.robots)
    }

    # Any UAV commanded by Search must be routed onto a MARVEL graph node
    # inside the native world frame.
    for heat_id, uav_id in scheduler.search_assignments.items():
        agent = scheduler._marvel_agents.get(int(uav_id))

        if agent is None or agent.node_coords is None or len(agent.node_coords) == 0:
            continue

        index = robot_index[int(uav_id)]
        goal = np.asarray(base_actions[index][0], dtype=float)

        assert frame.contains_world(goal) or np.min(
            np.linalg.norm(
                np.asarray(agent.node_coords, dtype=float) - goal,
                axis=1,
            )
        ) < 1e-6


# ======================================================================
# 8. Extended mode smoke
# ======================================================================

def test_extended_runtime_still_runs():
    config = load_and_validate_scenario(EXTENDED_SCENARIO)

    np.random.seed(11)
    runtime = SimulationRuntime(config)
    runtime.reset()

    for _ in range(10):
        _, info = runtime.step(runtime.default_actions())

    assert runtime.current_step == 10
    assert "comm_topology" in info
    assert runtime.obstacles.frame.cell_size == 1.0


def test_extended_and_native_frames_do_not_collide():
    """Switching mode must not mutate the other mode's geometry."""

    extended = load_and_validate_scenario(EXTENDED_SCENARIO)
    native = load_and_validate_scenario(NATIVE_SCENARIO)

    np.random.seed(5)
    extended_runtime = SimulationRuntime(extended)
    extended_runtime.reset()

    np.random.seed(5)
    native_runtime = SimulationRuntime(native)
    native_runtime.reset()

    assert extended_runtime.obstacles.frame.cell_size == 1.0
    assert native_runtime.obstacles.frame.cell_size == pytest.approx(
        MARVEL_CELL_SIZE
    )
    assert extended_runtime.obstacles.frame.origin == (0.0, 0.0)
    assert native_runtime.obstacles.frame.origin != (0.0, 0.0)

    # Re-resolving the extended config still yields extended geometry.
    assert resolve_geometry_mode(extended["environment"]) == (
        GEOMETRY_MODE_EXTENDED
    )


# ======================================================================
# helpers
# ======================================================================

def _write_scenario(tmp_path, environment_overrides, name="tmp_scenario"):
    """Write a scratch scenario outside the repo.

    Module config paths are absolute so the scratch file can live in a
    temporary directory without leaking into ``configs/scenarios``.
    """

    import yaml

    configs = (ROOT / "configs").resolve()

    config = {
        "scenario": {
            "name": name,
            "duration": {"max_steps": 10, "step_dt": 0.1},
        },
        "environment": {
            "map_size": [100, 100],
            **environment_overrides,
        },
        "robots": [
            {
                "id_range": [0, 0],
                "type": "explorer",
                "config": {
                    "fov": 120,
                    "sensor_range": 10,
                    "velocity": 1.0,
                    "initial_positions": "random_safe",
                },
            }
        ],
        "tasks": [
            {
                "task_id": "T1",
                "type": "exploration",
                "priority": 3,
                "params": {"target_coverage": 0.8},
            }
        ],
        "dynamics": {
            "config_file": str(configs / "dynamics" / "kinematic_simple.yaml")
        },
        "sensor": {"config_file": str(configs / "sensors" / "ideal.yaml")},
        "communication": {
            "config_file": str(configs / "communication" / "ideal.yaml")
        },
    }

    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    return path
