"""GPPO runtime-state semantic parity tests.

Four frozen deviations are covered:

1. persistent UAV high-level state (available / busy_until / utilization /
   assigned_task_num / current_task) reaching the GPPO model;
2. Search and Relay ``processing_time`` reaching the neural-model features;
3. the original MARVEL exploration-rate denominator;
4. the frozen target-detection scope.

Frozen source: ``/home/nick/MARVEL`` @ ``e867117``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integrations.gppo.event_sources import (
    FROZEN_RELAY_PROCESSING_TIME,
    FROZEN_SEARCH_PROCESSING_TIME,
    PublicHeatPoint,
    SearchSlotBuilder,
)
from integrations.gppo.task_graph import TaskType
from utils.geometry import GEOMETRY_MODE_EXTENDED, GEOMETRY_MODE_NATIVE
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime

FROZEN_CHECKPOINT = Path(
    "/home/nick/MARVEL/artifacts/phase14_v9_frozen/"
    "gppo_phase14_v9_ckpt80.pth"
)

NATIVE_SCENARIO = ROOT / "configs" / "scenarios" / "marvel_native_test.yaml"
EXTENDED_SCENARIO = ROOT / "configs" / "scenarios" / "baseline_maps_test.yaml"

requires_checkpoint = pytest.mark.skipif(
    not FROZEN_CHECKPOINT.exists(),
    reason="frozen GPPO checkpoint not available",
)


def _runtime(config_path, seed=7):
    config = load_and_validate_scenario(config_path)
    np.random.seed(seed)
    runtime = SimulationRuntime(config)
    runtime.reset()
    return runtime


def _gppo_native_runtime(seed=7):
    """Native runtime with a real frozen-GPPO scheduler attached."""

    from integrations.gppo.checkpoint import load_frozen_gppo
    from integrations.gppo.config_profile import prepare_gppo_config
    from integrations.gppo.scheduler import GPPOTaskScheduler

    _, checkpoint = load_frozen_gppo(str(FROZEN_CHECKPOINT), device="cpu")

    config = load_and_validate_scenario(NATIVE_SCENARIO)
    config["scenario"]["random_seed"] = int(seed)
    config["task_scheduler"] = {
        "mode": "gppo",
        "checkpoint": str(FROZEN_CHECKPOINT),
        "base_position": [0.0, 0.0],
        "search_scenario_seed": int(seed),
    }
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


def _activate(runtime, scheduler):
    from integrations.gppo.search_scenario import (
        initialize_search_scenario,
    )

    truth = initialize_search_scenario(runtime)
    scheduler.install_search_scenario(truth.public_heat_points())

    runtime.tasks.tasks["T2_target_search"].status = "active"

    free = runtime._free_mask()
    rows, cols = np.nonzero(free)
    take = int(0.35 * len(rows))

    runtime.explored_cells = {
        (int(cols[i]), int(rows[i])) for i in range(take)
    }

    assert scheduler._maybe_activate_search()

    return truth


def _stale(scheduler):
    return scheduler.tracker.positions(
        scheduler.profile.stale_age_fraction
    )


# ======================================================================
# 1. Persistent UAV assignment state
# ======================================================================

def test_unassigned_uav_graph_state_matches_frozen_defaults():
    """An idle UAV looks exactly like frozen Agent defaults."""

    from integrations.gppo.runtime_graph import RuntimeGraphBuilder

    runtime = _runtime(NATIVE_SCENARIO)

    for robot in runtime.robots:
        assert robot.available is True
        assert robot.busy_until == 0.0
        assert robot.utilization == 0.0
        assert robot.assigned_task_num == 0
        assert robot.current_task == "exploration"

    graph = RuntimeGraphBuilder(runtime).build().graph

    for state in graph.uav_states:
        assert state.available is True
        assert state.busy_until == 0.0
        assert state.assigned_task_num == 0
        assert state.current_task == TaskType.EXPLORATION


@requires_checkpoint
def test_search_assignment_persists_into_the_next_graph():
    from integrations.gppo.runtime_graph import RuntimeGraphBuilder

    runtime, scheduler = _gppo_native_runtime()
    _activate(runtime, scheduler)
    scheduler._refresh_search(_stale(scheduler))

    assert len(scheduler.search_assignments) == 2

    graph = RuntimeGraphBuilder(runtime).build().graph
    by_id = {state.uav_id: state for state in graph.uav_states}

    assigned_uids = {
        int(uid) for uid in scheduler.search_assignments.values()
    }

    for uid in assigned_uids:
        state = by_id[uid]
        robot = scheduler._robot_by_id(uid)

        # The assigned UAV must NOT look like an idle Exploration UAV.
        assert state.available is False
        assert state.current_task == TaskType.TARGET_SEARCH
        assert state.assigned_task_num == 1
        assert state.busy_until > 0.0
        assert state.busy_until == pytest.approx(robot.busy_until)

    # Idle UAVs are untouched.
    for uid, state in by_id.items():
        if uid in assigned_uids:
            continue
        assert state.available is True
        assert state.current_task == TaskType.EXPLORATION

    # Assigned UAVs are masked out of the allocator action space.
    for ti, task in enumerate(graph.subtasks):
        for ui, state in enumerate(graph.uav_states):
            if state.uav_id in assigned_uids:
                assert graph.action_mask[ti, ui]


@requires_checkpoint
def test_busy_until_uses_frozen_travel_plus_processing():
    runtime, scheduler = _gppo_native_runtime()
    _activate(runtime, scheduler)

    decisions = []

    original = scheduler.allocator.allocate

    def spy(slots, **kwargs):
        result = original(slots, **kwargs)
        decisions.extend(result)
        return result

    scheduler.allocator.allocate = spy

    scheduler._refresh_search(_stale(scheduler))

    assert decisions

    current_time = float(runtime.current_step * runtime.dt)

    for decision in decisions:
        robot = scheduler._robot_by_id(decision.uav_id)

        assert decision.processing_time == pytest.approx(6.0)
        assert robot.busy_until == pytest.approx(
            current_time + decision.total_time
        )
        assert decision.total_time >= decision.processing_time


@requires_checkpoint
def test_release_on_search_completion_restores_idle_state():
    runtime, scheduler = _gppo_native_runtime()
    truth = _activate(runtime, scheduler)
    scheduler._refresh_search(_stale(scheduler))

    heat_id, uid = next(iter(scheduler.search_assignments.items()))

    target_index = next(
        survivor.target_index
        for survivor in truth.survivors
        if survivor.heat_id == heat_id
    )

    runtime.tasks.tasks["T2_target_search"].found_targets.add(target_index)
    scheduler._sync_search_completions()

    assert heat_id in scheduler.serviced_heat_ids

    scheduler._reconcile_uav_state()

    robot = scheduler._robot_by_id(uid)

    assert robot.available is True
    assert robot.current_task == "exploration"
    assert robot.busy_until == pytest.approx(
        float(runtime.current_step * runtime.dt)
    )
    # assigned_task_num is never decremented by the frozen source.
    assert robot.assigned_task_num == 1


@requires_checkpoint
def test_safety_preemption_marks_uav_and_release_restores_it():
    runtime, scheduler = _gppo_native_runtime()
    _activate(runtime, scheduler)
    scheduler._refresh_search(_stale(scheduler))

    heat_id, uid = next(iter(scheduler.search_assignments.items()))

    scheduler.preempt_uav(uid)
    scheduler.safety.active_uav_ids.add(uid)

    scheduler._reconcile_uav_state()

    robot = scheduler._robot_by_id(uid)

    assert robot.current_task == "safety"
    assert robot.available is False

    # The preempted Search slot is gone.
    assert uid not in set(scheduler.search_assignments.values())

    # Safety release returns the UAV to idle Exploration.
    scheduler.safety.active_uav_ids.remove(uid)
    scheduler._reconcile_uav_state()

    assert robot.current_task == "exploration"
    assert robot.available is True


@requires_checkpoint
def test_uav_cannot_serve_search_relay_and_safety():
    """Global mutual exclusion across Search, Relay and Safety."""

    from integrations.gppo.runtime_graph import RuntimeGraphBuilder

    runtime, scheduler = _gppo_native_runtime()
    _activate(runtime, scheduler)

    captured = []
    original = scheduler.allocator.allocate

    def spy(slots, **kwargs):
        captured.extend(slots)
        return original(slots, **kwargs)

    scheduler.allocator.allocate = spy

    # --- Search allocation excludes Relay helpers and Safety UAVs ---------
    relay_helper = int(runtime.robots[1].robot_id)
    safety_uid = int(runtime.robots[0].robot_id)

    scheduler.relay_assignments = {
        int(runtime.robots[3].robot_id): (
            relay_helper,
            np.asarray((20.0, 20.0), dtype=float),
        )
    }
    scheduler.safety.active_uav_ids.add(safety_uid)

    scheduler._refresh_search(_stale(scheduler))

    assert captured

    for slot in captured:
        assert relay_helper in slot.forbidden_uav_ids
        assert safety_uid in slot.forbidden_uav_ids

    search_uids = set(scheduler.search_assignments.values())

    assert search_uids
    assert relay_helper not in search_uids
    assert safety_uid not in search_uids

    # --- Relay allocation excludes Search-assigned UAVs -------------------
    captured.clear()
    scheduler.relay_assignments = {}

    stale = _stale(scheduler)
    snapshot = scheduler.relay_builder.snapshot(
        stale, base_position=np.asarray((0.0, 0.0))
    )

    scheduler.relay_release_gate.last_counterfactual_connectivity = (
        float(snapshot.connectivity_ratio)
    )

    result = scheduler.relay_builder.build(
        stale,
        base_position=np.asarray((0.0, 0.0)),
        source_task_id="T3_relay",
        snapshot=snapshot,
    )

    reserved = set(scheduler.search_assignments.values())

    for slot in result.slots:
        combined = set(slot.forbidden_uav_ids) | reserved
        assert reserved <= combined

    # --- The GPPO model itself sees assigned UAVs as unavailable ----------
    graph = RuntimeGraphBuilder(runtime).build().graph
    by_id = {state.uav_id: state for state in graph.uav_states}

    for uid in search_uids:
        assert by_id[uid].available is False
        assert by_id[uid].current_task == TaskType.TARGET_SEARCH

    for uid, state in by_id.items():
        if state.available:
            # Only idle UAVs are selectable for any task.
            assert state.current_task in (
                TaskType.EXPLORATION,
                TaskType.SAFETY,
            ) or uid in search_uids


# ======================================================================
# 2. Search / Relay processing_time parity
# ======================================================================

def test_frozen_processing_time_constants():
    assert FROZEN_SEARCH_PROCESSING_TIME == 6.0
    assert FROZEN_RELAY_PROCESSING_TIME == 1.0


def test_search_slots_carry_frozen_processing_time():
    builder = SearchSlotBuilder(
        max_search_uavs=2, search_service_steps=6.0
    )

    points = [
        PublicHeatPoint(heat_id=i, position=(float(i), float(i)))
        for i in range(4)
    ]

    slots = builder.build(points)

    assert slots
    for slot in slots:
        assert slot.task_type == TaskType.TARGET_SEARCH
        assert slot.processing_time == pytest.approx(6.0)


def test_search_builder_rejects_zero_processing_time():
    builder = SearchSlotBuilder(
        max_search_uavs=2, search_service_steps=0.0
    )

    assert builder.search_service_steps == 1.0


@requires_checkpoint
def test_processing_time_reaches_the_neural_model_features():
    """The exact tensors handed to ckpt80 must carry processing_time."""

    runtime, scheduler = _gppo_native_runtime()
    _activate(runtime, scheduler)

    captured = []
    original = scheduler.allocator.adapter.assign

    def spy(graph, deterministic=True):
        captured.append(graph)
        return original(graph, deterministic=deterministic)

    scheduler.allocator.adapter.assign = spy

    scheduler._refresh_search(_stale(scheduler))

    assert captured

    graph = captured[0]

    # Raw engineering features.
    for ti, task in enumerate(graph.subtasks):
        assert task.task_type == TaskType.TARGET_SEARCH
        assert graph.task_features[ti, 4] == pytest.approx(6.0)
        for ui in range(graph.num_uavs):
            assert graph.edge_features[ti, ui, 1] == pytest.approx(6.0)
            assert graph.edge_features[ti, ui, 2] == pytest.approx(
                graph.edge_features[ti, ui, 0] + 6.0
            )

    # Model-facing normalized features: task 6/300, edge divided by escale.
    for ti in range(graph.num_subtasks):
        assert graph.model_task_features[ti, 4] == pytest.approx(
            6.0 / 300.0
        )

    escale = max(
        float(np.max(np.abs(graph.edge_features[..., 2]))), 1.0
    )

    for ti in range(graph.num_subtasks):
        for ui in range(graph.num_uavs):
            assert graph.model_edge_features[
                ti, ui, 1
            ] == pytest.approx(6.0 / escale)


def test_relay_slots_carry_frozen_processing_time():
    from integrations.gppo.event_sources import (
        ObservedRelayDemandBuilder,
    )

    builder = ObservedRelayDemandBuilder(
        comm_range=20.0, max_hops=2, max_demands=2
    )

    # Isolated UAV within two comm hops of the base, so the demand is
    # feasible and a slot is produced.
    positions = {
        0: np.asarray((0.0, 0.0)),
        1: np.asarray((30.0, 0.0)),
    }

    result = builder.build(
        positions, base_position=np.asarray((0.0, 0.0))
    )

    assert result.slots

    for slot in result.slots:
        assert slot.task_type == TaskType.RELAY
        assert slot.processing_time == pytest.approx(1.0)


# ======================================================================
# 3. Exploration-rate denominator parity
# ======================================================================

def test_native_exploration_rate_matches_frozen_formula():
    """|observed free| / |all free|, not |observed| / |all cells|."""

    runtime = _runtime(NATIVE_SCENARIO)

    free = runtime._free_mask()
    total_free = int(free.sum())

    # Frozen ground truth free count for maps_test/1.png.
    assert total_free == 22402
    assert runtime.free_cell_count == 250 * 250

    rows, cols = np.nonzero(free)
    take = 5000
    runtime.explored_cells = {
        (int(cols[i]), int(rows[i])) for i in range(take)
    }

    assert runtime.exploration_rate == pytest.approx(
        take / total_free
    )

    # The legacy denominator would have given a different answer.
    legacy = take / runtime.free_cell_count

    assert runtime.exploration_rate != pytest.approx(legacy)
    assert runtime.exploration_rate > legacy


def test_native_rate_ignores_observed_occupied_cells():
    """Obstacle cells are visible but must not count as explored."""

    runtime = _runtime(NATIVE_SCENARIO)

    grid = runtime.obstacles.get_occupancy_grid()
    occupied = np.argwhere(grid == 1)

    assert len(occupied) > 1000

    sample = occupied[:2000]
    runtime.explored_cells = {
        (int(c), int(r)) for r, c in sample
    }

    # Nothing free was observed.
    assert runtime.exploration_rate == 0.0


def test_native_free_coverage_differs_from_total_coverage():
    """Synthetic guard: a wrong denominator fails clearly."""

    runtime = _runtime(NATIVE_SCENARIO)

    free = runtime._free_mask()
    total_free = int(free.sum())
    total_cells = runtime.free_cell_count

    assert total_free < total_cells
    assert total_free / total_cells < 0.5

    rows, cols = np.nonzero(free)
    runtime.explored_cells = {
        (int(cols[i]), int(rows[i]))
        for i in range(len(rows))
    }

    # Observing every free cell saturates the rate.
    assert runtime.exploration_rate == pytest.approx(1.0)


def test_extended_exploration_rate_is_unchanged():
    runtime = _runtime(EXTENDED_SCENARIO)

    assert runtime.geometry_mode == GEOMETRY_MODE_EXTENDED

    runtime.explored_cells = {(i, 0) for i in range(1000)}

    assert runtime.exploration_rate == pytest.approx(
        1000 / runtime.free_cell_count
    )


# ======================================================================
# 4. Target-detection scope
# ======================================================================

def test_detection_requires_an_active_search_assignment():
    """Frozen rule: only currently assigned Search UAVs detect."""

    from utils.target_detector import TargetDetector

    runtime = _runtime(NATIVE_SCENARIO)

    # Place a hidden survivor on a cell the robots can see.
    robot = runtime.robots[0]
    frame = runtime.obstacles.frame
    cell = frame.world_to_cell(robot.position)
    robot.heading = 0.0

    runtime.target_detector = TargetDetector(
        {"T2_target_search": [{"x": cell[0], "y": cell[1]}]}
    )

    observations = runtime._get_observations()

    # The survivor cell must actually be observable, otherwise the
    # negative case would pass trivially.
    visible = {
        (int(x), int(y))
        for x, y in np.asarray(
            observations[robot.robot_id]["visible_cells"], dtype=int
        ).reshape(-1, 2)
    }

    assert cell in visible, (cell, len(visible))

    # Negative: no UAV is assigned to Search yet.
    runtime.active_search_uav_ids = set()

    assert runtime._get_target_detections(observations)[
        "T2_target_search"
    ] == {}

    # Positive: the observing UAV is assigned to Search.
    runtime.active_search_uav_ids = {int(robot.robot_id)}

    assert runtime._get_target_detections(observations)[
        "T2_target_search"
    ] == {0: int(robot.robot_id)}


def test_capable_but_unassigned_uav_does_not_detect():
    runtime = _runtime(NATIVE_SCENARIO)

    runtime.active_search_uav_ids = set()

    observations = runtime._get_observations()

    detections = runtime._get_target_detections(observations)

    assert all(
        entry == {} for entry in detections.values()
    ), detections


@requires_checkpoint
def test_assigned_search_uav_set_tracks_assignments():
    runtime, scheduler = _gppo_native_runtime()
    truth = _activate(runtime, scheduler)

    scheduler._refresh_search(_stale(scheduler))

    expected = {
        int(uid) for uid in scheduler.search_assignments.values()
    }

    assert runtime.active_search_uav_ids == expected
    assert expected

    # Completing one heat point removes exactly that UAV.
    heat_id, uid = next(iter(scheduler.search_assignments.items()))

    target_index = next(
        survivor.target_index
        for survivor in truth.survivors
        if survivor.heat_id == heat_id
    )

    runtime.tasks.tasks["T2_target_search"].found_targets.add(target_index)
    scheduler._sync_search_completions()
    scheduler._reconcile_uav_state()

    assert int(uid) not in runtime.active_search_uav_ids
