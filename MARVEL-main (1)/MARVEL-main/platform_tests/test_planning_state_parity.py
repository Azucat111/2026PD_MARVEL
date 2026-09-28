"""Planning-state equivalence for the shared-graph cache.

`NodeManager.get_all_node_graph` used to rebuild the coordinate array, the
per-node features and the O(n_nodes^2) adjacency matrix for every UAV.  The
graph is now cached across the UAVs of one planning epoch, keyed on a version
counter that `update_graph` bumps.

These tests pin the two properties the cache must have:

* the cached path returns exactly what the original per-UAV reconstruction
  returned, for every field and every UAV; and
* the cache is invalidated by a graph mutation, so it can never serve a
  stale graph.

The reference implementation below is the pre-optimization body, kept
verbatim so the comparison is against the real old behaviour rather than a
restatement of the new one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime
from utils.policy_adapter import MARVELPolicyAdapter

SCENARIO = ROOT / "configs" / "scenarios" / "baseline_maps_test.yaml"

REQUIRED_FIELDS = (
    "node_coords", "utility", "guidepost", "occupancy", "adjacent_matrix",
    "current_index", "neighbor_indices", "highest_utility_angles",
    "frontier_distribution", "heading_visited", "path_coords",
)


def reference_get_all_node_graph(manager, robot_location, robot_locations):
    """Pre-optimization `NodeManager.get_all_node_graph`, verbatim."""

    all_node_coords = []
    for node in manager.nodes_dict.__iter__():
        all_node_coords.append(node.data.coords)
    all_node_coords = np.array(all_node_coords).reshape(-1, 2)
    utility = []
    frontiers_distribution = []
    highest_utility_angle = []
    heading_visited = []

    n_nodes = all_node_coords.shape[0]
    adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
    node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
    for i, coords in enumerate(all_node_coords):
        node = manager.nodes_dict.find((coords[0], coords[1])).data
        utility.append(node.utility)
        frontiers_distribution.append(node.frontiers_distribution)
        heading_visited.append(node.heading_visited)
        highest_utility_angle.append(node.highest_utility_angle)

        for neighbor in node.neighbor_list:
            index = np.argwhere(
                node_coords_to_check == neighbor[0] + neighbor[1] * 1j
            )
            if index or index == [[0]]:
                index = index[0][0]
                adjacent_matrix[i, index] = 0

    utility = np.array(utility)
    frontiers_distribution = np.array(frontiers_distribution)
    highest_utility_angle = np.array(highest_utility_angle)
    heading_visited = np.array(heading_visited)

    if n_nodes == 0:
        raise RuntimeError("Cannot build graph observation: node manager is empty")
    robot_in_graph = manager.nodes_dict.nearest_neighbors(
        np.asarray(robot_location, dtype=float).tolist(), 1
    )[0].data.coords
    planning_start = np.asarray(robot_in_graph, dtype=float)

    indices = np.argwhere(utility > 0).reshape(-1)
    utility_node_coords = all_node_coords[indices]
    dist_dict, prev_dict = manager.Dijkstra(planning_start)
    nearest_utility_coords = planning_start
    nearest_dist = 1e8
    for coords in utility_node_coords:
        dist = dist_dict[(coords[0], coords[1])]
        if 0 < dist < nearest_dist:
            nearest_dist = dist
            nearest_utility_coords = coords

    path_coords, dist = manager.a_star(planning_start, nearest_utility_coords)
    guidepost = np.zeros_like(utility)
    for coords in path_coords:
        index = np.argwhere(
            all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
            == coords[0] + coords[1] * 1j
        )[0]
        guidepost[index] = 1

    current_index = np.argwhere(
        node_coords_to_check == robot_in_graph[0] + robot_in_graph[1] * 1j
    )[0][0]
    neighbor_indices = np.argwhere(
        adjacent_matrix[current_index] == 0
    ).reshape(-1)

    occupancy = np.zeros((n_nodes, 1))
    for location in robot_locations:
        location_in_graph = manager.nodes_dict.nearest_neighbors(
            np.asarray(location, dtype=float).tolist(), 1
        )[0].data.coords
        index = np.argwhere(
            node_coords_to_check == location_in_graph[0] + location_in_graph[1] * 1j
        )[0][0]
        if index == current_index:
            occupancy[index] = -1
        else:
            occupancy[index] = 1

    return (
        all_node_coords, utility, guidepost, occupancy, adjacent_matrix,
        current_index, neighbor_indices, highest_utility_angle,
        frontiers_distribution, heading_visited, path_coords,
    )


def _extended_runtime(uavs: int, steps: int):
    config = load_and_validate_scenario(SCENARIO)
    config["robots"] = [{
        "id_range": [0, uavs - 1],
        "type": "explorer",
        "team_id": 1,
        "config": {
            "fov": 120, "sensor_range": 10.0, "velocity": 1.0,
            "yaw_rate": 35, "initial_positions": "random_safe",
        },
    }]
    config.setdefault("scenario", {})["random_seed"] = 12345

    np.random.seed(12345)
    runtime = SimulationRuntime(config)
    observations = runtime.reset()
    adapter = MARVELPolicyAdapter(runtime)
    adapter.setup()

    interval = max(1, int(runtime._mission_step_interval()))
    for _ in range(steps):
        actions = adapter.get_actions(observations)
        for _tick in range(interval):
            observations, _info = runtime.step(actions)

    return runtime, adapter


def _assert_identical(reference, candidate, label):
    assert len(reference) == len(candidate), label

    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        if isinstance(expected, np.ndarray):
            assert np.array_equal(expected, actual), (
                label, index, "arrays differ",
                expected.shape, np.asarray(actual).shape,
            )
            assert expected.dtype == np.asarray(actual).dtype, (
                label, index, "dtype differs",
                expected.dtype, np.asarray(actual).dtype,
            )
        else:
            assert expected == actual, (label, index, expected, actual)


@pytest.mark.parametrize("uavs", [4, 8])
def test_cached_graph_matches_reference(uavs):
    """Every field of every UAV equals the per-UAV reconstruction."""

    runtime, adapter = _extended_runtime(uavs, steps=4)
    manager = adapter._node_manager

    locations = [robot.position for robot in runtime.robots]

    for robot_index, agent in enumerate(adapter.agents):
        expected = reference_get_all_node_graph(
            manager, agent.location, locations
        )
        actual = manager.get_all_node_graph(agent.location, locations)

        _assert_identical(
            expected, actual, f"uavs={uavs} robot={robot_index}"
        )

    assert manager._shared_version == manager._version, (
        "cache served a graph version that does not match the live graph"
    )


@pytest.mark.parametrize("uavs", [4, 8])
def test_cache_invalidated_by_graph_update(uavs):
    """A graph mutation must invalidate the cached shared arrays."""

    runtime, adapter = _extended_runtime(uavs, steps=3)
    manager = adapter._node_manager
    locations = [robot.position for robot in runtime.robots]

    first = manager.get_all_node_graph(locations[0], locations)
    before_version = manager._version

    # Any update_graph call changes the graph and must bump the version.
    for agent, robot in zip(adapter.agents, runtime.robots):
        agent.update_graph(adapter._build_map_info(), robot.position.copy())

    assert manager._version > before_version, "update_graph did not bump"

    second = manager.get_all_node_graph(locations[0], locations)

    # Recompute the reference on the mutated graph; the cached path must
    # agree with it, i.e. it must not have served the pre-mutation arrays.
    expected = reference_get_all_node_graph(
        manager, adapter.agents[0].location, locations
    )
    _assert_identical(expected, second, f"uavs={uavs} after mutation")

    assert manager._shared_version == manager._version


def test_shared_structure_is_uav_independent():
    """Only the position-dependent fields may differ between UAVs."""

    runtime, adapter = _extended_runtime(4, steps=4)
    manager = adapter._node_manager
    locations = [robot.position for robot in runtime.robots]

    first = manager.get_all_node_graph(adapter.agents[0].location, locations)
    second = manager.get_all_node_graph(adapter.agents[-1].location, locations)

    # Shared: coords, utility, adjacency, per-node angle/frontier/visited.
    for index in (0, 1, 4, 7, 8, 9):
        assert np.array_equal(first[index], second[index]), index

    # UAV-specific: the path/guidepost and the current node.
    assert not np.array_equal(first[2], second[2])
    assert first[5] != second[5]


def _episode_trace(uavs: int, steps: int, use_reference: bool):
    """Full extended episode; optionally force the reference planner."""

    from utils.node_manager import NodeManager

    original = NodeManager.get_all_node_graph
    if use_reference:
        NodeManager.get_all_node_graph = (
            lambda self, loc, locs: reference_get_all_node_graph(self, loc, locs)
        )

    try:
        config = load_and_validate_scenario(SCENARIO)
        config["robots"] = [{
            "id_range": [0, uavs - 1],
            "type": "explorer",
            "team_id": 1,
            "config": {
                "fov": 120, "sensor_range": 10.0, "velocity": 1.0,
                "yaw_rate": 35, "initial_positions": "random_safe",
            },
        }]
        config.setdefault("scenario", {})["random_seed"] = 12345

        np.random.seed(12345)
        runtime = SimulationRuntime(config)
        observations = runtime.reset()
        adapter = MARVELPolicyAdapter(runtime)
        adapter.setup()

        interval = max(1, int(runtime._mission_step_interval()))
        trace = []

        for _ in range(steps):
            actions = adapter.get_actions(observations)

            trace.append({
                "positions": np.asarray(
                    [robot.position for robot in runtime.robots], dtype=float
                ).copy(),
                "headings": np.asarray(
                    [robot.heading for robot in runtime.robots], dtype=float
                ).copy(),
                "actions": np.asarray(
                    [np.asarray(a[0], dtype=float) for a in actions], dtype=float
                ).copy(),
            })

            for _tick in range(interval):
                observations, _info = runtime.step(actions)

        trace.append({
            "positions": np.asarray(
                [robot.position for robot in runtime.robots], dtype=float
            ).copy(),
            "headings": np.asarray(
                [robot.heading for robot in runtime.robots], dtype=float
            ).copy(),
            "explored": int(runtime.explored_cells.__len__()),
            "nodes": int(adapter._node_manager.nodes_dict.__len__()),
        })

        return trace
    finally:
        NodeManager.get_all_node_graph = original


@pytest.mark.parametrize("uavs", [4, 8])
def test_full_episode_identical_with_and_without_cache(uavs):
    """A whole episode must be bit-identical with the reference planner.

    This is the end-to-end equivalence check: same seeds, same scenario, the
    only difference being whether the shared graph arrays are reused.
    """

    cached = _episode_trace(uavs, 8, use_reference=False)
    reference = _episode_trace(uavs, 8, use_reference=True)

    assert len(cached) == len(reference)

    for step, (a, b) in enumerate(zip(cached, reference)):
        for key in a:
            if isinstance(a[key], np.ndarray):
                assert np.array_equal(a[key], b[key]), (
                    f"uavs={uavs} step={step} field={key}"
                )
            else:
                assert a[key] == b[key], (
                    f"uavs={uavs} step={step} field={key}"
                )


# ======================================================================
# I — shared-graph construction
# ======================================================================

def reference_shared_graph(manager):
    """Pre-optimization `_shared_graph`, verbatim.

    Re-found every node by coordinate and resolved every neighbour with a
    full `np.argwhere` scan of the coordinate array.
    """

    all_node_coords = []
    for node in manager.nodes_dict.__iter__():
        all_node_coords.append(node.data.coords)
    all_node_coords = np.array(all_node_coords).reshape(-1, 2)
    utility = []
    frontiers_distribution = []
    highest_utility_angle = []
    heading_visited = []

    n_nodes = all_node_coords.shape[0]
    adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
    node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
    for i, coords in enumerate(all_node_coords):
        node = manager.nodes_dict.find((coords[0], coords[1])).data
        utility.append(node.utility)
        frontiers_distribution.append(node.frontiers_distribution)
        heading_visited.append(node.heading_visited)
        highest_utility_angle.append(node.highest_utility_angle)

        for neighbor in node.neighbor_list:
            index = np.argwhere(
                node_coords_to_check == neighbor[0] + neighbor[1] * 1j
            )
            if index or index == [[0]]:
                index = index[0][0]
                adjacent_matrix[i, index] = 0

    return (
        all_node_coords,
        np.array(utility),
        np.array(frontiers_distribution),
        np.array(highest_utility_angle),
        np.array(heading_visited),
        adjacent_matrix,
        node_coords_to_check,
    )


@pytest.mark.parametrize("uavs", [4, 8, 16])
def test_shared_graph_matches_reference(uavs):
    """Every shared array identical, values and dtype."""

    runtime, adapter = _extended_runtime(uavs, steps=5)
    manager = adapter._node_manager

    expected = reference_shared_graph(manager)
    actual = manager._shared_graph()

    assert len(actual) == len(expected) == 7

    for index, (a, b) in enumerate(zip(expected, actual)):
        assert isinstance(b, np.ndarray), (index, type(b))
        assert a.shape == b.shape, (index, a.shape, b.shape)
        assert a.dtype == b.dtype, (index, a.dtype, b.dtype)
        assert np.array_equal(a, b), index

    # The adjacency must actually contain edges and non-edges.
    adjacency = actual[5]
    assert (adjacency == 0).any(), "no edges, case is vacuous"
    assert (adjacency == 1).any(), "no non-edges, case is vacuous"


@pytest.mark.parametrize("uavs", [4, 8])
def test_shared_graph_paired_episode_identical(uavs):
    """A whole episode is identical with the reference construction."""

    from utils.node_manager import NodeManager

    original = NodeManager._shared_graph

    def cached(uavs_arg, steps_arg, use_reference):
        if use_reference:
            NodeManager._shared_graph = reference_shared_graph
        else:
            NodeManager._shared_graph = original

        try:
            return _episode_trace(uavs_arg, steps_arg, use_reference=False)
        finally:
            NodeManager._shared_graph = original

    fast = cached(uavs, 6, False)
    slow = cached(uavs, 6, True)

    assert len(fast) == len(slow)

    for step, (a, b) in enumerate(zip(fast, slow)):
        for key in a:
            if isinstance(a[key], np.ndarray):
                assert np.array_equal(a[key], b[key]), (
                    f"uavs={uavs} step={step} field={key}"
                )
            else:
                assert a[key] == b[key], (
                    f"uavs={uavs} step={step} field={key}"
                )
