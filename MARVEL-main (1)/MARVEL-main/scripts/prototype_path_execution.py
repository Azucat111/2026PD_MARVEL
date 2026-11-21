"""PHASE 6 prototype: correct the extended execution index space.

Root cause (proven in `scripts/audit_execution_path.py`): when the shared
graph exceeds ``NODE_PADDING_SIZE``, ``Agent.get_observation`` windows
``node_coords`` to the 360 nearest nodes and remaps ``current_edge`` into
*window* index space -- but ``Agent.decode_waypoint`` then indexes the
**full, unwindowed** ``self.node_coords`` with that window index.  The same
integer therefore means two different nodes, and the commanded waypoint
lands on an essentially arbitrary distant node instead of the selected
graph neighbour.

This prototype re-derives the identical window inside ``decode_waypoint``
and uses it for the coordinate lookup.  It touches nothing else:

* the PolicyNet forward is unchanged;
* ``action_index`` / ``waypoint_index`` are unchanged;
* the high-level selected neighbour is unchanged -- only the coordinate
  that neighbour maps to is corrected;
* no path planner, no new local planner, no policy change.

Benchmark-only: the patch is installed and removed by this script.

Usage:
    python scripts/prototype_path_execution.py --uavs 4 8 16 30 60
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

from benchmark_live_gppo import build  # noqa: E402
from parameter import NODE_PADDING_SIZE  # noqa: E402
from utils import agent as agent_module  # noqa: E402
from utils.safety_shield import SafetyShield  # noqa: E402

Agent = agent_module.Agent


# ----------------------------------------------------------------------
# the prototype patch
# ----------------------------------------------------------------------

def windowed_coords(agent):
    """The exact window `get_observation` builds, or None if not engaged."""

    full = np.asarray(agent.node_coords)
    if full.shape[0] <= NODE_PADDING_SIZE:
        return None
    current = int(agent.current_index)
    distances = np.linalg.norm(full - full[current], axis=1)
    keep = np.argsort(distances)[:NODE_PADDING_SIZE]
    if current not in keep:
        keep[-1] = current
    keep = np.unique(keep)
    return full[keep]


def patched_decode(agent, observation, logp, greedy=False, stats=None):
    _, _, _, _, current_edge, _, _, _, _ = observation

    if greedy:
        action_index = torch.argmax(logp, 1).long()
    else:
        action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)

    waypoint_index = action_index.item() // agent.num_heading_candidates
    next_node_index = current_edge[0, waypoint_index, 0].item()
    heading_index = agent.neighbor_best_indices[waypoint_index][
        action_index.item() % agent.num_heading_candidates
    ]

    window = windowed_coords(agent)
    if window is None:
        next_position = agent.node_coords[next_node_index]
    else:
        if stats is not None:
            full = np.asarray(agent.node_coords)
            stats["index_space_corrections"] += 1
            stats["distance_corrected_m"] += float(
                np.linalg.norm(full[next_node_index] - window[next_node_index])
            )
        next_position = window[next_node_index]

    return next_position, next_node_index, action_index, heading_index


def install(stats):
    original = Agent.decode_waypoint

    def replacement(self, observation, logp, greedy=False):
        return patched_decode(self, observation, logp, greedy=greedy, stats=stats)

    Agent.decode_waypoint = replacement

    def restore():
        Agent.decode_waypoint = original

    return restore


FOOTPRINT_RADIUS = 0.2


def install_node_clearance(runtime, stats):
    """PHASE 7's cheap M1 half: refuse nodes the footprint cannot occupy.

    One clearance test per *newly created* node -- no swept-edge sweep, no
    per-step rescan.  Same authoritative primitive the shield uses.
    """

    from utils.node_manager import NodeManager

    original_add = NodeManager.add_node_to_dict

    def filtered_add(self, coords, local_frontiers, updating_map_info):
        node = original_add(self, coords, local_frontiers, updating_map_info)
        if runtime.obstacles.check_collision(
            np.asarray(coords, dtype=float), radius=FOOTPRINT_RADIUS
        )[0]:
            try:
                self.nodes_dict.remove(list(coords))
            except Exception:
                pass
            node.utility = 0
            node.need_update_neighbor = False
            node.neighbor_list = []
            node.observable_frontiers = set()
            stats["nodes_refused_clearance"] += 1
            return node
        stats["nodes_admitted"] += 1
        return node

    NodeManager.add_node_to_dict = filtered_add

    def restore():
        NodeManager.add_node_to_dict = original_add

    return restore


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------

def graph_shape(adapter):
    manager = adapter._node_manager
    coords = []
    index = {}
    for node in manager.nodes_dict.__iter__():
        key = (round(float(node.data.coords[0]), 3),
               round(float(node.data.coords[1]), 3))
        if key in index:
            continue
        index[key] = len(coords)
        coords.append(np.asarray(node.data.coords, dtype=float))

    adjacency = [[] for _ in coords]
    seen = set()
    edges = 0
    for node in manager.nodes_dict.__iter__():
        key = (round(float(node.data.coords[0]), 3),
               round(float(node.data.coords[1]), 3))
        if key not in index:
            continue
        i = index[key]
        for entry in node.data.neighbor_list:
            other = (round(float(entry[0]), 3), round(float(entry[1]), 3))
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
    largest = 0
    components = 0
    for start in range(n):
        if visited[start]:
            continue
        components += 1
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
        largest = max(largest, size)

    return {
        "graph_nodes": n,
        "graph_edges": edges,
        "components": components,
        "reachable_fraction": largest / n if n else 0.0,
    }


def run_arm(uavs, steps, fixed, node_clearance=False):
    runtime, adapter, _ = build(uavs, steps=0)

    stats = {"index_space_corrections": 0, "distance_corrected_m": 0.0,
             "nodes_refused_clearance": 0, "nodes_admitted": 0}
    restore = install(stats) if fixed else None
    restore_nodes = (
        install_node_clearance(runtime, stats) if node_clearance else None
    )

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
    commanded = []
    step_times = []

    try:
        for _ in range(steps):
            positions_before = np.asarray(
                [r.position for r in runtime.robots], dtype=float
            )
            start = time.perf_counter()
            actions = adapter.get_actions(observations)
            for index, robot in enumerate(runtime.robots):
                commanded.append(float(np.linalg.norm(
                    np.asarray(actions[index][0], dtype=float) - robot.position
                )))
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
        if restore_nodes is not None:
            restore_nodes()

    return {
        "uavs": uavs,
        "fixed": bool(fixed),
        "steps": steps,
        "parked_uavs_half_window": int(np.count_nonzero(
            parked_max >= max(1, steps) // 2
        )),
        "shield_static_obstacle": shield_counts.get("static_obstacle", 0),
        "shield_uav_proximity": shield_counts.get("uav_proximity", 0),
        "grid_obstacle_blocks": grid_blocks,
        "exploration_rate": float(runtime.exploration_rate),
        "mean_commanded_distance_m": (
            sum(commanded) / len(commanded) if commanded else 0.0
        ),
        "steady_mean_seconds": (
            sum(step_times) / len(step_times) if step_times else None
        ),
        "node_clearance": bool(node_clearance),
        "nodes_refused_clearance": stats["nodes_refused_clearance"],
        "index_space_corrections": stats["index_space_corrections"],
        "mean_correction_m": (
            stats["distance_corrected_m"] / stats["index_space_corrections"]
            if stats["index_space_corrections"] else 0.0
        ),
        **graph_shape(adapter),
    }


def action_parity(uavs=8):
    """Prove the PolicyNet decision is untouched at an identical state."""

    runtime, adapter, _ = build(uavs, steps=2)

    def first_action_indices():
        agent = adapter.agents[0]
        observation = agent.get_observation(pad=adapter._observation_padding())
        with torch.no_grad():
            logits = agent.policy_net(*observation)
        return int(torch.argmax(logits, 1).item()), float(logits.max().item())

    before = first_action_indices()
    restore = install({"index_space_corrections": 0,
                       "distance_corrected_m": 0.0})
    try:
        after = first_action_indices()
    finally:
        restore()

    return {
        "action_index_before": before[0],
        "action_index_after": after[0],
        "logit_max_before": before[1],
        "logit_max_after": after[1],
        "identical": before[0] == after[0] and before[1] == after[1],
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
    parser.add_argument("--arms", nargs="+",
                        choices=("OLD", "PATH", "PATH+NODE"),
                        default=["OLD", "PATH", "PATH+NODE"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / f"extended_execution_path_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "system_info.txt").write_text(
        json.dumps(system_info(), indent=2), encoding="utf-8"
    )

    parity = action_parity()
    print(f"[path] policy parity: {parity}", flush=True)

    rows = []
    all_arms = [
        ("OLD", False, False),
        ("PATH", True, False),
        ("PATH+NODE", True, True),
    ]
    arms = [a for a in all_arms if a[0] in args.arms]
    for label, fixed, node_clearance in arms:
        for uavs in args.uavs:
            print(f"[path] {label} uavs={uavs} steps={args.steps} ...",
                  flush=True)
            result = run_arm(uavs, args.steps, fixed,
                             node_clearance=node_clearance)
            rows.append(result)
            print(
                f"[path]   parked={result['parked_uavs_half_window']}/{uavs} "
                f"shield_static={result['shield_static_obstacle']} "
                f"grid={result['grid_obstacle_blocks']} "
                f"cmd_dist={result['mean_commanded_distance_m']:.1f}m "
                f"corrections={result['index_space_corrections']} "
                f"mean_corr={result['mean_correction_m']:.1f}m "
                f"nodes={result['graph_nodes']} "
                f"reach={result['reachable_fraction']:.3f} "
                f"rate={result['exploration_rate']:.4f}",
                flush=True,
            )

    with open(out / "execution_comparison.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / "execution_comparison.json").write_text(
        json.dumps({"policy_parity": parity, "arms": rows}, indent=2,
                   default=str),
        encoding="utf-8",
    )
    print(f"[path] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
