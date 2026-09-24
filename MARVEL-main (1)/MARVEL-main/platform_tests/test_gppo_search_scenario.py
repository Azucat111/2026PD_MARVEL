"""Frozen Phase14-v9 Search scenario generator and privacy-boundary tests.

Source of truth for the generator semantics:
``/home/nick/MARVEL`` @ ``e867117``, ``utils/explore_search_coordinator.py``.

Coverage:
    A. generator determinism
    B. seed separation
    C. public/private leakage audit
    D. start-connected free space only
    E. line-of-sight filtering
    F. heat_target_radius containment
    G. detection -> heat_id completion bridge
    H. resource state (Relay helpers are globally unavailable to Search)
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integrations.gppo.event_sources import PublicHeatPoint
from integrations.gppo.search_scenario import (
    FROZEN_SENSOR_RANGE,
    HiddenSurvivor,
    FrozenSearchRadii,
    FrozenSearchScenarioGenerator,
    SearchScenarioTruth,
    build_search_scenario,
    initialize_search_scenario,
)
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime

FROZEN_CHECKPOINT = Path(
    "/home/nick/MARVEL/artifacts/phase14_v9_frozen/"
    "gppo_phase14_v9_ckpt80.pth"
)

# Frozen derivation from test_parameter: SENSOR_RANGE = 10, CELL_SIZE = 0.4.
FROZEN_SEARCH_REACH_RADIUS = 3.5
FROZEN_HEAT_TARGET_RADIUS = 4.0

SCENARIO_FILE = (
    ROOT / "configs" / "scenarios" / "baseline_maps_test.yaml"
)


# ======================================================================
# Helpers
# ======================================================================

def open_grid(rows: int = 24, cols: int = 24) -> np.ndarray:
    """Team runtime occupancy grid: 0 = free, 1 = occupied."""

    return np.zeros((rows, cols), dtype=np.uint8)


def split_grid(rows: int = 20, cols: int = 24) -> np.ndarray:
    """Two free components separated by an impassable band at cols 8..13."""

    grid = np.ones((rows, cols), dtype=np.uint8)
    grid[:, 0:8] = 0
    grid[:, 14:24] = 0
    return grid


def partial_wall_grid(
    rows: int = 24,
    cols: int = 24,
    wall_col: int = 10,
    wall_rows: tuple[int, int] = (5, 18),
) -> np.ndarray:
    """Open map with a wall that can be walked around.

    Both sides of the wall stay start-connected, so the LOS filter is the
    only thing that can reject a candidate across it.
    """

    grid = open_grid(rows, cols)
    grid[wall_rows[0]:wall_rows[1] + 1, wall_col] = 1
    return grid


ROBOTS = (
    (2.0, 2.0),
    (3.0, 2.0),
    (2.0, 3.0),
    (3.0, 3.0),
)


def generate(grid, seed: int, robots=ROBOTS, survivor_count: int = 4):
    return build_search_scenario(
        occupancy_grid=grid,
        robot_locations=robots,
        seed=seed,
        survivor_count=survivor_count,
    )


def heat_positions(heat_points):
    return [tuple(point.position) for point in heat_points]


def survivor_positions(survivors):
    return [tuple(survivor.position) for survivor in survivors]


# ======================================================================
# A. Determinism
# ======================================================================

def test_generator_is_deterministic_for_same_map_robots_seed():
    grid = partial_wall_grid()

    first = generate(grid, seed=1234)
    second = generate(grid, seed=1234)

    assert heat_positions(first[0]) == heat_positions(second[0])
    assert survivor_positions(first[1]) == survivor_positions(second[1])

    first_map = {
        survivor.target_index: survivor.heat_id
        for survivor in first[1]
    }
    second_map = {
        survivor.target_index: survivor.heat_id
        for survivor in second[1]
    }

    assert first_map == second_map


def test_determinism_does_not_depend_on_global_numpy_rng():
    grid = partial_wall_grid()

    np.random.seed(1)
    first = generate(grid, seed=99)

    np.random.seed(987654)
    np.random.get_state()
    second = generate(grid, seed=99)

    assert heat_positions(first[0]) == heat_positions(second[0])
    assert survivor_positions(first[1]) == survivor_positions(second[1])


# ======================================================================
# B. Seed separation
# ======================================================================

def test_different_seed_changes_the_scenario():
    grid = partial_wall_grid()

    baseline = generate(grid, seed=5)
    other = generate(grid, seed=6)

    assert heat_positions(baseline[0]) != heat_positions(other[0])


# ======================================================================
# C. Public/private leakage audit
# ======================================================================

def test_public_heat_point_has_no_hidden_target_field():
    names = {
        field.name
        for field in dataclasses.fields(PublicHeatPoint)
    }

    assert "target_index" not in names
    assert "target_position" not in names

    assert names == {
        "heat_id",
        "position",
        "confidence",
        "priority",
        "serviced",
    }


def test_public_projection_carries_no_association():
    grid = partial_wall_grid()

    heat_points, survivors, _ = generate(grid, seed=17)

    truth = SearchScenarioTruth(
        task_id="T2_target_search",
        heat_points=heat_points,
        survivors=survivors,
        radii=FrozenSearchRadii.derive(cell_size=1.0),
    )

    for point in truth.public_heat_points():
        for field in dataclasses.fields(point):
            value = getattr(point, field.name)

            if isinstance(value, (int, float)):
                continue

            assert not isinstance(value, tuple) or len(value) == 2

        # No public field exposes a target index.
        assert not hasattr(point, "target_index")

    # The association exists only on the private side.
    assert truth.heat_id_for_target(0) == 0
    assert truth.heat_id_for_target(3) == 3


def test_scheduler_module_has_no_hidden_association():
    """Static guard: the leak must not be reintroduced by a later edit."""

    import integrations.gppo.scheduler as scheduler_module

    source = Path(scheduler_module.__file__).read_text(
        encoding="utf-8"
    )

    assert "target_to_heat_id" not in source
    # The single mention of target_index is the boundary documentation.
    assert source.count("target_index") == 1


def test_generator_module_exposes_truth_only_privately():
    """The generator returns bare coordinates; the truth object is explicit."""

    import integrations.gppo.search_scenario as module

    source = Path(module.__file__).read_text(encoding="utf-8")

    # The heat-point projection is built from numbers only.
    assert "PublicHeatPoint(" in source
    assert "target_index=" not in source.split("class SearchScenarioTruth")[0]


# ======================================================================
# D. Start-connected free space
# ======================================================================

def test_heat_points_and_survivors_stay_in_start_connected_component():
    grid = split_grid()
    robots = [(2.0, 10.0), (3.0, 10.0)]

    heat_points, survivors, _ = generate(
        grid, seed=5, robots=robots
    )

    # Component A is cols 0..7; the wall band is cols 8..13.
    assert [point.heat_id for point in heat_points] == [0, 1, 2, 3]

    for point in heat_points:
        assert point.position[0] < 8.0, point

    for survivor in survivors:
        assert survivor.position[0] < 8.0, survivor


def test_start_connected_component_follows_the_initial_uav():
    """The same map yields the other component when the UAV starts there."""

    grid = split_grid()

    left, _, _ = generate(grid, seed=5, robots=[(2.0, 10.0)])
    right, _, _ = generate(grid, seed=5, robots=[(18.0, 10.0)])

    assert all(point.position[0] < 8.0 for point in left)
    assert all(point.position[0] >= 14.0 for point in right)


def test_generator_rejects_maps_without_start_connected_free_space():
    grid = np.ones((10, 10), dtype=np.uint8)
    grid[5, 5] = 1  # fully occupied

    with pytest.raises(ValueError):
        generate(grid, seed=1, robots=[(2.0, 2.0)])


# ======================================================================
# E. Line of sight
# ======================================================================

def test_line_of_sight_rejects_wall_blocked_pair():
    generator = FrozenSearchScenarioGenerator(
        occupancy_grid=partial_wall_grid(),
        robot_locations=[(2.0, 2.0)],
        seed=3,
    )

    heat = np.asarray((9.0, 10.0))

    # The wall is at col 10; the cell across it is free but occluded.
    assert generator.is_free_cell(11, 10)
    assert not generator._line_of_sight_free(heat, (11.0, 10.0))

    # Straight down the open side of the wall is fine.
    assert generator._line_of_sight_free(heat, (7.0, 10.0))


def test_survivors_never_cross_a_wall_from_their_heat_point():
    grid = partial_wall_grid()

    generator = FrozenSearchScenarioGenerator(
        occupancy_grid=grid,
        robot_locations=[(2.0, 2.0)],
        seed=1,
    )

    heat = np.asarray((9.0, 10.0))

    # Cell (11, 10) is free and only 2 m away, well inside
    # heat_target_radius = 4.0, so only the LOS filter can exclude it.
    assert np.linalg.norm(np.asarray((11.0, 10.0)) - heat) <= 4.0

    for seed in range(24):
        generator.seed = seed
        targets = generator._preset_targets([heat])
        (_, position) = targets[0]

        assert generator._line_of_sight_free(heat, position)
        assert position[0] <= 9.0, (seed, position)


def test_every_generated_survivor_has_line_of_sight_to_its_heat_point():
    grid = partial_wall_grid()

    for seed in range(8):
        generator = FrozenSearchScenarioGenerator(
            occupancy_grid=grid,
            robot_locations=[(2.0, 2.0)],
            seed=seed,
        )

        heat_points, targets = generator.generate()

        for heat_id, position in targets:
            assert generator._line_of_sight_free(
                heat_points[heat_id], position
            ), (seed, heat_id, position)


# ======================================================================
# F. Radii
# ======================================================================

def test_frozen_radii_match_source_formulas():
    radii = FrozenSearchRadii.derive(
        sensor_range=FROZEN_SENSOR_RANGE,
        cell_size=1.0,
        search_service_steps=6,
    )

    # max(10 * 0.35, cell_size)
    assert radii.search_reach_radius == FROZEN_SEARCH_REACH_RADIUS
    # max(10 - 3.5 - 1.0, 0)
    assert radii.guaranteed_hidden_radius == pytest.approx(5.5)
    # min(10 * 0.40, guaranteed)
    assert radii.heat_target_radius == FROZEN_HEAT_TARGET_RADIUS
    assert radii.search_service_steps == 6


def test_survivors_within_heat_target_radius_of_their_heat_point():
    grid = partial_wall_grid()
    radii = FrozenSearchRadii.derive(cell_size=1.0)

    for seed in range(8):
        heat_points, survivors, _ = generate(grid, seed=seed)
        by_id = {
            point.heat_id: np.asarray(point.position)
            for point in heat_points
        }

        for survivor in survivors:
            heat = by_id[survivor.heat_id]
            distance = float(
                np.linalg.norm(
                    np.asarray(survivor.position) - heat
                )
            )

            assert distance <= radii.heat_target_radius + 1e-9, (
                seed,
                survivor,
                distance,
            )


# ======================================================================
# Runtime / scheduler fixtures for G and H
# ======================================================================

def build_gppo_runtime(seed: int = 7):
    """Real runtime + frozen checkpoint scheduler, prepared and reset."""

    if not FROZEN_CHECKPOINT.exists():
        pytest.skip(
            f"frozen checkpoint not available: {FROZEN_CHECKPOINT}"
        )

    from integrations.gppo.checkpoint import load_frozen_gppo
    from integrations.gppo.config_profile import prepare_gppo_config
    from integrations.gppo.scheduler import GPPOTaskScheduler

    config = load_and_validate_scenario(SCENARIO_FILE)

    config["tasks"].append({
        "task_id": "T2_target_search",
        "type": "target_search",
        "priority": 5,
        "assigned_robot_types": ["explorer"],
        "activation": {
            "condition": "exploration_rate >= 0.3",
        },
        "params": {},
    })

    config["task_scheduler"] = {
        "mode": "gppo",
        "checkpoint": str(FROZEN_CHECKPOINT),
        "base_position": [10.0, 10.0],
        "search_scenario_seed": int(seed),
    }

    config["scenario"]["random_seed"] = int(seed)

    _, checkpoint = load_frozen_gppo(
        str(FROZEN_CHECKPOINT), device="cpu"
    )

    config = prepare_gppo_config(config, checkpoint)

    np.random.seed(seed)
    runtime = SimulationRuntime(config)
    runtime.reset()

    scheduler = GPPOTaskScheduler(
        runtime,
        checkpoint_path=str(FROZEN_CHECKPOINT),
        device="cpu",
        scheduler_config=config["task_scheduler"],
    )

    return runtime, scheduler


def activate_search(runtime, scheduler):
    """Drive the frozen activation path without running exploration."""

    task = runtime.tasks.tasks["T2_target_search"]
    task.status = "active"

    target = int(0.35 * runtime.free_cell_count)
    runtime.explored_cells = {
        (index % 500, index // 500)
        for index in range(target)
    }

    assert scheduler._maybe_activate_search()


def stale_positions(scheduler):
    return scheduler.tracker.positions(
        scheduler.profile.stale_age_fraction
    )


# ======================================================================
# G. Completion bridge
# ======================================================================

def test_runtime_resolves_detections_to_heat_ids():
    runtime, _ = build_gppo_runtime(seed=7)

    truth = initialize_search_scenario(runtime)

    assert runtime.search_scenario is truth
    assert runtime.completed_search_heat_ids() == set()

    task = runtime.tasks.tasks["T2_target_search"]

    # Sensor layer reports a hidden target index.
    task.found_targets.add(2)

    assert runtime.completed_search_heat_ids() == {
        truth.heat_id_for_target(2)
    }

    # The detector received grid-cell truth, not the public heat point.
    detector_targets = runtime.target_detector.hidden_targets[
        "T2_target_search"
    ]

    assert len(detector_targets) == truth.target_count
    assert all(
        set(entry) == {"x", "y"}
        for entry in detector_targets
    )


def test_reset_clears_previous_episode_scenario():
    runtime, _ = build_gppo_runtime(seed=7)

    first = initialize_search_scenario(runtime)
    runtime.tasks.tasks["T2_target_search"].found_targets.add(0)

    assert runtime.completed_search_heat_ids()

    runtime.reset()

    # No residue: neither the truth object nor the sensor carries the
    # finished episode, and stale detections cannot resolve.
    assert runtime.search_scenario is None
    assert runtime.completed_search_heat_ids() == set()
    assert all(
        entries == []
        for entries in runtime.target_detector.hidden_targets.values()
    )


def test_scenario_reproduces_across_resets_with_same_start_positions():
    runtime, _ = build_gppo_runtime(seed=7)

    first = initialize_search_scenario(runtime)

    # Initial UAV positions are drawn from the global numpy RNG by
    # SimulationRuntime._generate_initial_positions, so reproducing the
    # scenario across episodes means reproducing the episode seed.
    np.random.seed(7)
    runtime.reset()

    second = initialize_search_scenario(runtime)

    assert heat_positions(second.public_heat_points()) == (
        heat_positions(first.public_heat_points())
    )
    assert survivor_positions(second.survivors) == (
        survivor_positions(first.survivors)
    )
    assert {
        survivor.target_index: survivor.heat_id
        for survivor in second.survivors
    } == {
        survivor.target_index: survivor.heat_id
        for survivor in first.survivors
    }


def test_detection_frees_exactly_one_search_slot_and_refills():
    runtime, scheduler = build_gppo_runtime(seed=7)

    truth = initialize_search_scenario(runtime)
    scheduler.install_search_scenario(truth.public_heat_points())

    assert not hasattr(scheduler, "target_to_heat_id")

    activate_search(runtime, scheduler)

    stale = stale_positions(scheduler)
    scheduler._refresh_search(stale)

    assert scheduler.search_activated
    assert len(scheduler.search_assignments) == 2

    assigned_heat_ids = set(scheduler.search_assignments)

    # Every GPPO task node the allocator saw is a public heat point.
    public_positions = {
        tuple(point.position)
        for point in truth.public_heat_points()
    }

    assert public_positions

    # Detect the hidden survivor of one currently assigned heat point.
    target_index = next(
        survivor.target_index
        for survivor in truth.survivors
        if survivor.heat_id in assigned_heat_ids
    )
    expected_heat_id = truth.heat_id_for_target(target_index)

    runtime.tasks.tasks["T2_target_search"].found_targets.add(
        target_index
    )

    # The scheduler learns only the heat id.
    newly_serviced = scheduler._sync_search_completions()

    assert newly_serviced == [expected_heat_id]
    assert expected_heat_id in scheduler.serviced_heat_ids
    assert expected_heat_id not in scheduler.search_assignments

    # Exactly one Search slot freed -> immediate refill.
    before = set(scheduler.search_assignments)
    scheduler._refresh_search(stale_positions(scheduler))

    after = set(scheduler.search_assignments)

    assert expected_heat_id not in after
    assert len(after) == 2
    assert len(after - before) == 1
    assert after - before <= {
        point.heat_id
        for point in truth.public_heat_points()
    }


def test_dwell_completion_still_takes_six_consecutive_steps():
    """Frozen 6-step continuous dwell, unchanged by the scenario install."""

    runtime, scheduler = build_gppo_runtime(seed=7)

    truth = initialize_search_scenario(runtime)
    scheduler.install_search_scenario(truth.public_heat_points())

    activate_search(runtime, scheduler)
    scheduler._refresh_search(stale_positions(scheduler))

    heat_id, uav_id = next(iter(scheduler.search_assignments.items()))
    point = scheduler.heat_by_id[heat_id]

    robot = next(
        item
        for item in runtime.robots
        if int(item.robot_id) == int(uav_id)
    )

    # Teleport the assigned UAV onto its public heat point.
    robot.position = np.asarray(point.position, dtype=float)

    for step in range(scheduler.search_service_steps - 1):
        assert scheduler._update_search_service() == []
        assert heat_id not in scheduler.serviced_heat_ids, step

    events = scheduler._update_search_service()

    assert heat_id in scheduler.serviced_heat_ids
    assert [event["reason"] for event in events] == ["dwell"]
    assert events[0]["dwell_steps"] == 6
    assert events[0]["heat_id"] == heat_id


def test_leaving_the_reach_radius_resets_dwell():
    runtime, scheduler = build_gppo_runtime(seed=7)

    truth = initialize_search_scenario(runtime)
    scheduler.install_search_scenario(truth.public_heat_points())

    activate_search(runtime, scheduler)
    scheduler._refresh_search(stale_positions(scheduler))

    heat_id, uav_id = next(iter(scheduler.search_assignments.items()))
    point = scheduler.heat_by_id[heat_id]

    robot = next(
        item
        for item in runtime.robots
        if int(item.robot_id) == int(uav_id)
    )

    anchor = np.asarray(point.position, dtype=float)

    robot.position = anchor.copy()

    for _ in range(3):
        scheduler._update_search_service()

    assert scheduler.search_dwell_by_heat[heat_id] == 3

    # Step outside the frozen 3.5 m reach radius.
    robot.position = anchor + np.asarray(
        (scheduler.search_reach_radius + 1.0, 0.0)
    )

    scheduler._update_search_service()

    assert scheduler.search_dwell_by_heat[heat_id] == 0
    assert heat_id not in scheduler.serviced_heat_ids

    # Dwell rebuilt from zero still needs a full six steps.
    robot.position = anchor.copy()

    for _ in range(scheduler.search_service_steps - 1):
        scheduler._update_search_service()

    assert heat_id not in scheduler.serviced_heat_ids

    scheduler._update_search_service()

    assert heat_id in scheduler.serviced_heat_ids


def test_search_reach_radius_is_the_frozen_three_point_five():
    runtime, scheduler = build_gppo_runtime(seed=7)

    assert scheduler.search_reach_radius == pytest.approx(
        FROZEN_SEARCH_REACH_RADIUS
    )
    assert scheduler.search_service_steps == 6
    assert scheduler.max_search_uavs == 2
    assert scheduler.search_activation_threshold == pytest.approx(
        0.30
    )


# ======================================================================
# H. Resource state: assigned UAVs are globally unavailable
# ======================================================================

def test_search_allocation_excludes_active_relay_helpers():
    runtime, scheduler = build_gppo_runtime(seed=7)

    truth = initialize_search_scenario(runtime)
    scheduler.install_search_scenario(truth.public_heat_points())

    activate_search(runtime, scheduler)

    # A UAV is already executing Relay.
    relay_helper = int(runtime.robots[1].robot_id)
    scheduler.relay_assignments = {
        int(runtime.robots[3].robot_id): (
            relay_helper,
            np.asarray((20.0, 20.0), dtype=float),
        )
    }

    captured = []

    def capture(slots, **kwargs):
        captured.extend(slots)
        return []

    scheduler.allocator.allocate = capture

    scheduler._refresh_search(stale_positions(scheduler))

    assert captured

    for slot in captured:
        assert relay_helper in slot.forbidden_uav_ids, slot


def test_search_and_relay_never_share_a_uav_after_refresh():
    """End-to-end guard for the conflict apply() raises on."""

    runtime, scheduler = build_gppo_runtime(seed=7)

    truth = initialize_search_scenario(runtime)
    scheduler.install_search_scenario(truth.public_heat_points())

    activate_search(runtime, scheduler)

    relay_helper = int(runtime.robots[1].robot_id)
    scheduler.relay_assignments = {
        int(runtime.robots[3].robot_id): (
            relay_helper,
            np.asarray((20.0, 20.0), dtype=float),
        )
    }

    stale = stale_positions(scheduler)
    scheduler._refresh_search(stale)

    assert relay_helper not in set(
        scheduler.search_assignments.values()
    )

    actions = runtime.default_actions()

    # Must not raise "received both Search and Relay assignments".
    scheduler.apply(actions)


# ======================================================================
# I. Frozen constants regression
# ======================================================================

def test_profile_exposes_frozen_search_radii():
    from integrations.gppo.checkpoint import load_frozen_gppo
    from integrations.gppo.config_profile import prepare_gppo_config

    if not FROZEN_CHECKPOINT.exists():
        pytest.skip("frozen checkpoint not available")

    config = load_and_validate_scenario(SCENARIO_FILE)
    config["scenario"]["random_seed"] = 1

    _, checkpoint = load_frozen_gppo(
        str(FROZEN_CHECKPOINT), device="cpu"
    )

    config = prepare_gppo_config(config, checkpoint)
    profile = config["_gppo_profile"]

    assert profile["search_reach_radius"] == (
        FROZEN_SEARCH_REACH_RADIUS
    )
    assert profile["search_heat_target_radius"] == (
        FROZEN_HEAT_TARGET_RADIUS
    )
    assert profile["search_service_steps"] == 6
    assert profile["search_cell_size"] == 1.0
    assert profile["activation_threshold"] == 0.30


def test_public_heat_points_reach_the_scheduler_unchanged():
    runtime, scheduler = build_gppo_runtime(seed=7)

    truth = initialize_search_scenario(runtime)
    public = truth.public_heat_points()

    scheduler.install_search_scenario(public)

    assert scheduler.heat_points == public
    assert set(scheduler.heat_by_id) == {
        point.heat_id for point in public
    }
    assert scheduler.search_dwell_by_heat == {
        point.heat_id: 0 for point in public
    }


def test_search_activation_fails_closed_without_a_scenario():
    runtime, scheduler = build_gppo_runtime(seed=7)

    activate = runtime.tasks.tasks["T2_target_search"]
    activate.status = "active"

    target = int(0.35 * runtime.free_cell_count)
    runtime.explored_cells = {
        (index % 500, index // 500)
        for index in range(target)
    }

    assert scheduler.heat_points == []
    assert not scheduler._maybe_activate_search()
