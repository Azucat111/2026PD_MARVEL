"""PHASE 5 prototype: clearance-consistent extended graph (benchmark-only).

Compares the stock graph against a graph whose nodes and edges satisfy the
*same* physical footprint that SafetyShield and runtime.step enforce:

  * a node is kept only if ``ObstacleManager.check_collision(coords, 0.2)``
    is clean -- the identical primitive the shield calls;
  * an edge is kept only if the whole swept segment keeps that clearance,
    sampled at 0.05 m.

Nothing in the production tree is touched: the filter is a wrapper around
`NodeManager.update_graph` installed by this script.

Usage:
    python scripts/prototype_clearance_graph.py --uavs 4 8 16 30 60
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from audit_graph_clearance import (  # noqa: E402
    FOOTPRINT_RADIUS,
    SWEEP_STEP,
    build,
    segment_clear,
)
from utils import node_manager as node_manager_module  # noqa: E402
from utils.safety_shield import SafetyShield  # noqa: E402


# ----------------------------------------------------------------------
# the prototype filter
# ----------------------------------------------------------------------

def _coords_key(coords):
    return (round(float(coords[0]), 3), round(float(coords[1]), 3))


def install(manager, runtime):
    """Admit a node/edge only if the physical footprint fits.

    The filter runs at *creation* time, not on every `update_graph`: the
    occupancy grid is ground truth and therefore static, so a node or edge
    that fails once fails forever.  That keeps the cost proportional to the
    number of newly discovered nodes/edges instead of to the whole graph.
    """

    stats = {"removed_nodes": 0, "removed_edges": 0,
             "admitted_nodes": 0, "admitted_edges": 0,
             "prune_seconds": 0.0}

    original_add = manager.add_node_to_dict
    node_class = None

    def filtered_add(coords, local_frontiers, updating_map_info):
        nonlocal node_class
        start = time.perf_counter()
        try:
            node = original_add(coords, local_frontiers, updating_map_info)
            if node_class is None:
                node_class = type(node)
            if runtime.obstacles.check_collision(
                np.asarray(coords, dtype=float), radius=FOOTPRINT_RADIUS
            )[0]:
                # Detach: the caller still receives an object, but it is not
                # in the quadtree and is made inert so it never contributes
                # neighbours.
                try:
                    manager.nodes_dict.remove(list(coords))
                except Exception:
                    try:
                        manager.nodes_dict.remove(np.asarray(coords).tolist())
                    except Exception:
                        pass
                node.utility = 0
                node.need_update_neighbor = False
                node.neighbor_list = []
                node.observable_frontiers = set()
                node.frontiers_distribution = np.zeros(node.num_angles_bin)
                stats["removed_nodes"] += 1
                return node
            stats["admitted_nodes"] += 1
            return node
        finally:
            stats["prune_seconds"] += time.perf_counter() - start

    manager.add_node_to_dict = filtered_add

    from utils.node_manager import Node

    original_update_neighbours = Node.update_neighbor_nodes

    def filtered_update_neighbours(self, updating_map_info, nodes_dict):
        result = original_update_neighbours(self, updating_map_info, nodes_dict)

        coords = np.asarray(self.coords, dtype=float)
        kept = []
        dropped = []
        for entry in self.neighbor_list:
            target = np.asarray(entry, dtype=float)
            if segment_clear(runtime, coords, target,
                             radius=FOOTPRINT_RADIUS, step=SWEEP_STEP):
                kept.append(entry)
            else:
                dropped.append(entry)

        if dropped:
            self.neighbor_list[:] = kept
            for entry in dropped:
                neighbour = nodes_dict.find((entry[0], entry[1]))
                if neighbour is None:
                    continue
                reciprocal = neighbour.data.neighbor_list
                for index in range(len(reciprocal) - 1, -1, -1):
                    if [round(float(v), 3) for v in reciprocal[index]] == [
                        round(float(v), 3) for v in self.coords
                    ]:
                        reciprocal.pop(index)
                stats["removed_edges"] += 1

        stats["admitted_edges"] += len(kept)
        return result

    Node.update_neighbor_nodes = filtered_update_neighbours

    def restore():
        manager.add_node_to_dict = original_add
        Node.update_neighbor_nodes = original_update_neighbours

    return restore, stats


# ----------------------------------------------------------------------
# graph shape
# ----------------------------------------------------------------------

def graph_shape(runtime, adapter):
    manager = adapter._node_manager
    coords = []
    index = {}
    for node in manager.nodes_dict.__iter__():
        key = _coords_key(node.data.coords)
        if key in index:
            continue
        index[key] = len(coords)
        coords.append(np.asarray(node.data.coords, dtype=float))

    adjacency = [[] for _ in coords]
    seen = set()
    edges = 0
    for node in manager.nodes_dict.__iter__():
        key = _coords_key(node.data.coords)
        if key not in index:
            continue
        i = index[key]
        for entry in node.data.neighbor_list:
            other = _coords_key(entry)
            if other not in index:
                continue
            j = index[other]
            if i == j:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen:
                continue
            seen.add(pair)
            adjacency[i].append(j)
            adjacency[j].append(i)
            edges += 1

    n = len(coords)
    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for nxt in adjacency[current]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        components.append(size)
    components.sort(reverse=True)

    isolated = sum(1 for a in adjacency if not a)
    reachable = components[0] if components else 0

    return {
        "graph_nodes": n,
        "graph_edges": edges,
        "components": len(components),
        "largest_component": reachable,
        "reachable_fraction": reachable / n if n else 0.0,
        "isolated_nodes": isolated,
    }


# ----------------------------------------------------------------------
# arm
# ----------------------------------------------------------------------

def run_arm(uavs, steps, use_filter):
    runtime, adapter, _ = build(uavs, steps=0)

    stats = None
    restore = None
    if use_filter:
        restore, stats = install(adapter._node_manager, runtime)

    interval = max(1, int(runtime._mission_step_interval()))
    observations = runtime._get_observations()

    shield_counts: dict[str, int] = {}
    original_record = SafetyShield._record

    def counting(self, robot_id, step, reason):
        key = str(reason).split(":")[0]
        shield_counts[key] = shield_counts.get(key, 0) + 1
        return original_record(self, robot_id, step, reason)

    SafetyShield._record = counting

    parked_run = np.zeros(uavs, dtype=int)
    parked_max = np.zeros(uavs, dtype=int)
    grid_blocks = 0
    step_times = []

    try:
        for _ in range(steps):
            positions_before = np.asarray(
                [r.position for r in runtime.robots], dtype=float
            )
            start = time.perf_counter()
            actions = adapter.get_actions(observations)
            for _tick in range(interval):
                observations, info = runtime.step(actions)
                grid_blocks += sum(
                    1 for c in info["collisions"]
                    if c.get("type") == "grid_obstacle"
                )
            step_times.append(time.perf_counter() - start)

            moved = np.linalg.norm(
                np.asarray([r.position for r in runtime.robots], dtype=float)
                - positions_before, axis=1
            )
            parked_run = np.where(moved < 1e-9, parked_run + 1, 0)
            parked_max = np.maximum(parked_max, parked_run)
    finally:
        SafetyShield._record = original_record
        if restore is not None:
            restore()

    shape = graph_shape(runtime, adapter)
    return {
        "uavs": uavs,
        "filter": bool(use_filter),
        "steps": steps,
        "parked_uavs_half_window": int(np.count_nonzero(
            parked_max >= max(1, steps) // 2
        )),
        "shield_static_obstacle": shield_counts.get("static_obstacle", 0),
        "shield_uav_proximity": shield_counts.get("uav_proximity", 0),
        "grid_obstacle_blocks": grid_blocks,
        "exploration_rate": float(runtime.exploration_rate),
        "steady_mean_seconds": (
            sum(step_times) / len(step_times) if step_times else None
        ),
        "prune_stats": stats,
        **shape,
    }


def system_info():
    import psutil
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_mb": float(psutil.virtual_memory().total / (1024 * 1024)),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uavs", type=int, nargs="+", default=[4, 8, 16, 30, 60])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / f"extended_clearance_prototype_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "system_info.txt").write_text(
        json.dumps(system_info(), indent=2), encoding="utf-8"
    )

    rows = []
    for use_filter in (False, True):
        for uavs in args.uavs:
            label = "CLEARANCE" if use_filter else "STOCK"
            print(f"[proto] {label} uavs={uavs} steps={args.steps} ...",
                  flush=True)
            result = run_arm(uavs, args.steps, use_filter)
            rows.append(result)
            pruned = result["prune_stats"] or {}
            print(
                f"[proto]   parked={result['parked_uavs_half_window']}/{uavs} "
                f"nodes={result['graph_nodes']} "
                f"edges={result['graph_edges']} "
                f"components={result['components']} "
                f"isolated={result['isolated_nodes']} "
                f"reachable={result['reachable_fraction']:.3f} "
                f"shield_static={result['shield_static_obstacle']} "
                f"grid_blocks={result['grid_obstacle_blocks']} "
                f"rate={result['exploration_rate']:.4f} "
                f"pruned_nodes={pruned.get('removed_nodes', 0)} "
                f"pruned_edges={pruned.get('removed_edges', 0)}",
                flush=True,
            )

    with open(out / "prototype_comparison.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "uavs", "filter", "parked_half_window", "graph_nodes",
            "graph_edges", "components", "largest_component",
            "reachable_fraction", "isolated_nodes",
            "shield_static_obstacle", "shield_uav_proximity",
            "grid_obstacle_blocks", "exploration_rate", "steady_mean_s",
            "pruned_nodes", "pruned_edges",
        ])
        for r in rows:
            pruned = r["prune_stats"] or {}
            writer.writerow([
                r["uavs"], r["filter"], r["parked_uavs_half_window"],
                r["graph_nodes"], r["graph_edges"], r["components"],
                r["largest_component"], round(r["reachable_fraction"], 6),
                r["isolated_nodes"], r["shield_static_obstacle"],
                r["shield_uav_proximity"], r["grid_obstacle_blocks"],
                round(r["exploration_rate"], 6),
                round(r["steady_mean_seconds"], 4),
                pruned.get("removed_nodes", 0), pruned.get("removed_edges", 0),
            ])
    (out / "prototype_comparison.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    print(f"[proto] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
