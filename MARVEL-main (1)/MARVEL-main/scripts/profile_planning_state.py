"""Decompose `NodeManager.get_all_node_graph` into its internal operations.

Read-only diagnostic: the sub-operations are re-run against the live node
graph exactly as the production function performs them, so the breakdown
reflects real node counts and real neighbour lists.

Usage:
    python scripts/profile_planning_state.py --uavs 4 16 30 60 --steps 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.scenario_config import load_and_validate_scenario  # noqa: E402
from utils.simulation_runtime import SimulationRuntime  # noqa: E402
from utils.policy_adapter import MARVELPolicyAdapter  # noqa: E402

SCENARIO = ROOT / "configs" / "scenarios" / "baseline_maps_test.yaml"


def build_config(uavs: int):
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
    return config


def decompose(manager, robot_location, robot_locations, repeats=1):
    """Time each internal operation of get_all_node_graph on a live graph."""

    nodes_dict = manager.nodes_dict
    out = {}

    def timed(name, fn):
        start = time.perf_counter()
        result = fn()
        out[name] = (time.perf_counter() - start) / repeats
        return result

    # --- 1. gather coordinates -------------------------------------------------
    def gather():
        coords = []
        for node in nodes_dict.__iter__():
            coords.append(node.data.coords)
        return np.array(coords).reshape(-1, 2)

    all_node_coords = timed("coords", gather)
    n = all_node_coords.shape[0]
    out["n_nodes"] = n

    # --- 2. adjacency allocation ----------------------------------------------
    adjacent_matrix = timed(
        "adjacency_alloc", lambda: np.ones((n, n)).astype(int)
    )

    node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j

    # --- 3. per-node feature gather + neighbour resolution --------------------
    def scan():
        utility, frontiers, angles, visited = [], [], [], []
        for i, coords in enumerate(all_node_coords):
            node = nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            frontiers.append(node.frontiers_distribution)
            visited.append(node.heading_visited)
            angles.append(node.highest_utility_angle)

            for neighbour in node.neighbor_list:
                idx = np.argwhere(
                    node_coords_to_check
                    == neighbour[0] + neighbour[1] * 1j
                )
                if idx or idx == [[0]]:
                    adjacent_matrix[i, idx[0][0]] = 0
        return (np.array(utility), np.array(frontiers),
                np.array(angles), np.array(visited))

    utility, frontiers, angles, visited = timed("feature_scan", scan)

    n_edges = int((adjacent_matrix == 0).sum())

    # --- 4. planning start -----------------------------------------------------
    def start_node():
        return np.asarray(
            nodes_dict.nearest_neighbors(
                np.asarray(robot_location, dtype=float).tolist(), 1
            )[0].data.coords
        )

    planning_start = timed("nearest_start", start_node)

    # --- 5. Dijkstra -----------------------------------------------------------
    dist_dict, prev_dict = timed(
        "dijkstra", lambda: manager.Dijkstra(planning_start)
    )

    # --- 6. nearest utility ----------------------------------------------------
    indices = np.argwhere(utility > 0).reshape(-1)
    utility_node_coords = all_node_coords[indices]

    def nearest_utility():
        best = planning_start
        best_dist = 1e8
        for coords in utility_node_coords:
            d = dist_dict[(coords[0], coords[1])]
            if 0 < d < best_dist:
                best_dist = d
                best = coords
        return best

    nearest = timed("nearest_utility", nearest_utility)

    # --- 7. A* -----------------------------------------------------------------
    path_coords, _ = timed(
        "astar", lambda: manager.a_star(planning_start, nearest)
    )

    # --- 8. guidepost ----------------------------------------------------------
    def guidepost():
        g = np.zeros_like(utility)
        complex_coords = (
            all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        )
        for coords in path_coords:
            idx = np.argwhere(
                complex_coords == coords[0] + coords[1] * 1j
            )[0]
            g[idx] = 1
        return g

    timed("guidepost", guidepost)
    out["path_len"] = len(path_coords)

    # --- 9. tail: current node, neighbours, occupancy --------------------------
    def tail():
        current_index = np.argwhere(
            node_coords_to_check
            == planning_start[0] + planning_start[1] * 1j
        )[0][0]
        neighbour_indices = np.argwhere(
            adjacent_matrix[current_index] == 0
        ).reshape(-1)
        occupancy = np.zeros((n, 1))
        for location in robot_locations:
            loc_in_graph = nodes_dict.nearest_neighbors(
                np.asarray(location, dtype=float).tolist(), 1
            )[0].data.coords
            index = np.argwhere(
                node_coords_to_check
                == loc_in_graph[0] + loc_in_graph[1] * 1j
            )[0][0]
            occupancy[index] = -1 if index == current_index else 1
        return occupancy

    timed("tail", tail)

    out["edges"] = n_edges
    out["total"] = sum(
        v for k, v in out.items()
        if k not in ("n_nodes", "edges", "path_len")
    )
    return out


def profile(uavs: int, steps: int):
    config = build_config(uavs)

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

    manager = adapter._node_manager
    locations = [robot.position for robot in runtime.robots]

    # Time one full get_all_node_graph, then decompose one call.
    start = time.perf_counter()
    manager.get_all_node_graph(locations[0], locations)
    whole = time.perf_counter() - start

    parts = decompose(manager, locations[0], locations, repeats=1)
    parts["uavs"] = uavs
    parts["whole_call_seconds"] = whole
    parts["calls_per_mission_step"] = len(runtime.robots)

    # Is the shared structure really identical across UAVs?
    c0, u0, _g0, _o0, a0, ci0, _n0, _ha0, _f0, _hv0, _p0 = (
        manager.get_all_node_graph(locations[0], locations)
    )
    c1, u1, _g1, _o1, a1, ci1, _n1, _ha1, _f1, _hv1, _p1 = (
        manager.get_all_node_graph(locations[-1], locations)
    )
    parts["shared_coords_identical"] = bool(np.array_equal(c0, c1))
    parts["shared_utility_identical"] = bool(np.array_equal(u0, u1))
    parts["shared_adjacency_identical"] = bool(np.array_equal(a0, a1))
    parts["current_index_differs"] = bool(ci0 != ci1)

    return parts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uavs", type=int, nargs="+", default=[4, 16, 30, 60])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / f"planning_state_profile_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for uavs in args.uavs:
        print(f"[profile] uavs={uavs} ...", flush=True)
        results.append(profile(uavs, args.steps))
        r = results[-1]
        print(
            f"[profile] uavs={uavs} nodes={r['n_nodes']} "
            f"whole={r['whole_call_seconds']:.4f}s total={r['total']:.4f}s",
            flush=True,
        )

    (out / "profile.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    keys = [k for k in results[0]
            if k not in ("uavs", "n_nodes", "edges", "path_len",
                         "calls_per_mission_step",
                         "shared_coords_identical", "shared_utility_identical",
                         "shared_adjacency_identical", "current_index_differs")]
    print()
    header = "operation".ljust(20) + "".join(
        f"{r['uavs']:>12d}" for r in results
    )
    print(header)
    for key in keys:
        line = key.ljust(20)
        for r in results:
            line += f"{r[key]:>12.4f}"
        print(line)
    print("n_nodes".ljust(20) + "".join(f"{r['n_nodes']:>12d}" for r in results))
    print("edges".ljust(20) + "".join(f"{r['edges']:>12d}" for r in results))
    print("calls/step".ljust(20) + "".join(
        f"{r['calls_per_mission_step']:>12d}" for r in results
    ))
    print()
    for r in results:
        print(
            f"uavs={r['uavs']:3d} shared coords/utility/adjacency identical "
            f"across UAVs: {r['shared_coords_identical']}/"
            f"{r['shared_utility_identical']}/"
            f"{r['shared_adjacency_identical']}, "
            f"current_index differs: {r['current_index_differs']}"
        )

    print(f"[profile] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
