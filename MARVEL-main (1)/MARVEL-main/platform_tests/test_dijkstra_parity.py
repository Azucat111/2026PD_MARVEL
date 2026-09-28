"""Heap-frontier Dijkstra must reproduce the linear-scan Dijkstra exactly.

`NodeManager.Dijkstra` used to find the unvisited minimum by scanning the
whole remaining set on every iteration.  That is replaced here by a binary
heap with stale-entry skipping.  The distances feed
`get_all_node_graph`'s nearest-utility-node search, which selects the node
the a* guidepost is routed to, so any distance difference changes the
planner output.

The reference below is the pre-change body kept verbatim, so the comparison
is against the real old algorithm rather than a restatement of the new one.

Producer/consumer audit (2026-09-28):

* `NodeManager.Dijkstra` has exactly one production caller,
  `get_all_node_graph` (`utils/node_manager.py`).
* `dist_dict` is read only by that function's nearest-utility-node loop.
* `prev_dict` is assigned there and never read. `get_Dijkstra_path_and_dist`
  is its only other consumer and has no callers anywhere in `utils/` or
  `integrations/`.
* `utils/ground_truth_node_manager.py` carries its own independent
  `Dijkstra`; it is a different class and is not constructed by the adapter
  (`ground_truth_node_manager=None`), so it is unaffected either way.
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


def reference_dijkstra(self, start):
    """Pre-change `NodeManager.Dijkstra`, verbatim linear-scan frontier."""

    q = set()
    dist_dict = {}
    prev_dict = {}

    for node in self.nodes_dict.__iter__():
        coords = node.data.coords
        key = (coords[0], coords[1])
        dist_dict[key] = 1e8
        prev_dict[key] = None
        q.add(key)

    assert (start[0], start[1]) in dist_dict.keys()
    dist_dict[(start[0], start[1])] = 0

    while len(q) > 0:
        u = None
        for coords in q:
            if u is None:
                u = coords
            elif dist_dict[coords] < dist_dict[u]:
                u = coords

        q.remove(u)

        node = self.nodes_dict.find(u).data
        for neighbor_node_coords in node.neighbor_list:
            v = (neighbor_node_coords[0], neighbor_node_coords[1])
            if v in q:
                cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                        neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                cost = np.round(cost, 2)
                alt = dist_dict[u] + cost
                if alt < dist_dict[v]:
                    dist_dict[v] = alt
                    prev_dict[v] = u

    return dist_dict, prev_dict


def _extended_adapter(uavs: int, steps: int):
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


def _source_keys(manager, adapter):
    """A spread of sources: every UAV node plus degree extremes."""

    keys = []

    for agent in adapter.agents:
        node = manager.nodes_dict.nearest_neighbors(
            np.asarray(agent.location, dtype=float).tolist(), 1
        )[0].data.coords
        keys.append((node[0], node[1]))

    degrees = []
    for node in manager.nodes_dict.__iter__():
        coords = node.data.coords
        degrees.append(
            (len(node.data.neighbor_list), (coords[0], coords[1]))
        )

    degrees.sort()
    if degrees:
        keys.append(degrees[0][1])    # sparsest local region
        keys.append(degrees[-1][1])   # densest local region

    seen, unique = set(), []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)

    return unique


@pytest.mark.parametrize("uavs", [4, 8, 16])
def test_heap_dijkstra_matches_reference(uavs):
    """Identical keys, distances, reachability and predecessor sets."""

    from utils.node_manager import NodeManager

    runtime, adapter = _extended_adapter(uavs, steps=5)
    manager = adapter._node_manager

    assert manager.nodes_dict.__len__() > 0

    for source in _source_keys(manager, adapter):
        expected_dist, expected_prev = reference_dijkstra(manager, source)
        actual_dist, actual_prev = manager.Dijkstra(source)

        assert set(actual_dist) == set(expected_dist), (
            "dist_dict key set differs", source
        )

        reachable = {k for k, v in expected_dist.items() if v < 1e8}
        assert reachable, "no reachable nodes, case is vacuous"

        unreachable = {k for k, v in expected_dist.items() if v >= 1e8}
        assert unreachable, (
            "graph is fully connected, disconnected case untested"
        )

        for key, value in expected_dist.items():
            assert actual_dist[key] == value, (
                "distance differs", source, key, value, actual_dist[key]
            )

        # Unreachable nodes keep the sentinel and a None predecessor.
        for key in unreachable:
            assert actual_dist[key] == 1e8
            assert actual_prev[key] is None

        # Predecessor *identity* may differ among equal-cost paths, but the
        # shape must not: source has no predecessor, unreachable nodes have
        # none, and every other reachable node has one.
        for key in expected_prev:
            if key == source or expected_dist[key] >= 1e8:
                assert actual_prev[key] is None, (source, key)
            else:
                assert actual_prev[key] is not None, (source, key)


def test_prev_dict_is_not_read_by_the_planner():
    """The planner binds prev_dict but never reads it."""

    import ast
    import inspect
    import textwrap
    import utils.node_manager as module

    function = ast.parse(textwrap.dedent(
        inspect.getsource(module.NodeManager.get_all_node_graph)
    )).body[0]

    reads = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Name)
        and node.id == "prev_dict"
        and isinstance(node.ctx, ast.Load)
    ]
    assert not reads, f"prev_dict is read in the planner: {reads}"

    # Its only other consumer has no callers in the integration tree.
    tree = ROOT / "utils"
    uses = []
    for path in tree.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "get_Dijkstra_path_and_dist" in text:
            uses.append(path.name)
    assert uses == ["node_manager.py"], uses


@pytest.mark.parametrize("uavs", [4, 8])
def test_planner_output_identical(uavs):
    """Every Dijkstra-derived planner field is identical old vs new."""

    from utils.node_manager import NodeManager

    runtime, adapter = _extended_adapter(uavs, steps=4)
    manager = adapter._node_manager
    locations = [robot.position for robot in runtime.robots]

    for agent in adapter.agents:
        expected = reference_planner(manager, agent.location, locations)
        actual = manager.get_all_node_graph(agent.location, locations)

        for index, (a, b) in enumerate(zip(expected, actual)):
            if isinstance(a, np.ndarray):
                assert np.array_equal(a, b), (agent.id, index)
                assert a.dtype == np.asarray(b).dtype, (agent.id, index)
            else:
                assert a == b, (agent.id, index)


def reference_planner(manager, robot_location, robot_locations):
    """`get_all_node_graph` with the reference Dijkstra substituted."""

    original = manager.Dijkstra
    manager.Dijkstra = lambda start: reference_dijkstra(manager, start)
    try:
        return manager.get_all_node_graph(robot_location, robot_locations)
    finally:
        manager.Dijkstra = original


def _paired_episode(uavs: int, steps: int, use_reference: bool):
    """Run an extended episode, optionally with the old Dijkstra in place."""

    from utils.node_manager import NodeManager

    original = NodeManager.Dijkstra
    if use_reference:
        NodeManager.Dijkstra = (
            lambda self, start: reference_dijkstra(self, start)
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
                    [r.position for r in runtime.robots], dtype=float
                ).copy(),
                "headings": np.asarray(
                    [r.heading for r in runtime.robots], dtype=float
                ).copy(),
                "waypoints": np.asarray(
                    [np.asarray(a[0], dtype=float) for a in actions], dtype=float
                ).copy(),
                "commands": np.asarray(
                    [float(a[1]) for a in actions], dtype=float
                ).copy(),
                "nodes": int(adapter._node_manager.nodes_dict.__len__()),
            })

            for _tick in range(interval):
                observations, _info = runtime.step(actions)

        trace.append({
            "positions": np.asarray(
                [r.position for r in runtime.robots], dtype=float
            ).copy(),
            "headings": np.asarray(
                [r.heading for r in runtime.robots], dtype=float
            ).copy(),
            "nodes": int(adapter._node_manager.nodes_dict.__len__()),
            "explored": int(len(runtime.explored_cells)),
        })

        return trace
    finally:
        NodeManager.Dijkstra = original


@pytest.mark.parametrize("uavs,steps", [(4, 20), (8, 20), (16, 10)])
def test_paired_episodes_identical(uavs, steps):
    """Same seeds, old vs heap Dijkstra: every step must match exactly."""

    heap = _paired_episode(uavs, steps, use_reference=False)
    reference = _paired_episode(uavs, steps, use_reference=True)

    assert len(heap) == len(reference)

    for step, (a, b) in enumerate(zip(heap, reference)):
        for key in a:
            if isinstance(a[key], np.ndarray):
                assert np.array_equal(a[key], b[key]), (
                    f"uavs={uavs} step={step} field={key}"
                )
            else:
                assert a[key] == b[key], (
                    f"uavs={uavs} step={step} field={key}: "
                    f"{a[key]} vs {b[key]}"
                )

    final = heap[-1]
    assert final["nodes"] > 0 and final["explored"] > 0, "vacuous run"
