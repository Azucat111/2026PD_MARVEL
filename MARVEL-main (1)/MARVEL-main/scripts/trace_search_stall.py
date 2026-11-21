"""PHASE 6 -- focused 60-UAV Search travel-stall trace.

Read-only with respect to production: every probe is a monkeypatch installed
by this script.  Instruments only the Search-assigned UAVs and records, per
mission step, why each one did or did not move:

  * commanded waypoint and resulting displacement
  * whether the waypoint came from the Search override or from the MARVEL
    fallback
  * `observed_next_hop` result (None means "graph expansion required")
  * per-UAV SafetyShield hover events, by reason
  * per-UAV dynamics collision skips
  * graph current node and observed-graph reachability of the heat point

Usage:
    python scripts/trace_search_stall.py --uavs 60 --steps 90
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
from integrations.gppo import observed_graph as observed_graph_module  # noqa: E402
from utils.safety_shield import SafetyShield  # noqa: E402


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


def run(uavs: int, steps: int, out: Path):
    runtime, adapter, activation = build(uavs, steps=0)
    scheduler = adapter.scheduler
    interval = max(1, int(runtime._mission_step_interval()))
    observations = runtime._get_observations()

    # Per-UAV shield events.
    shield_events: dict[int, list[str]] = {}
    original_record = SafetyShield._record

    def capture(self, robot_id, step, reason):
        shield_events.setdefault(int(robot_id), []).append(str(reason))
        return original_record(self, robot_id, step, reason)

    SafetyShield._record = capture

    # Record every observed_next_hop call so we can tell an override from a
    # MARVEL fallback.
    hop_log: dict[int, list] = {}
    original_hop = observed_graph_module.observed_next_hop

    def traced_hop(agent, goal):
        result = original_hop(agent, goal)
        hop_log.setdefault(int(agent.id), []).append(result)
        return result

    observed_graph_module.observed_next_hop = traced_hop

    # The scheduler imported the symbol directly, so patch its namespace too.
    import integrations.gppo.scheduler as scheduler_module
    had_scheduler_ref = hasattr(scheduler_module, "observed_next_hop")
    if had_scheduler_ref:
        scheduler_module.observed_next_hop = traced_hop

    rows = []
    try:
        for mission in range(1, steps + 1):
            for value in shield_events.values():
                value.clear()
            for value in hop_log.values():
                value.clear()

            positions_before = {
                int(r.robot_id): np.asarray(r.position, dtype=float).copy()
                for r in runtime.robots
            }
            actions = adapter.get_actions(observations)

            grid_by_uav: dict[int, int] = {}
            for _tick in range(interval):
                observations, info = runtime.step(actions)
                for event in info["collisions"]:
                    uid = event.get("robot_id")
                    if uid is not None:
                        grid_by_uav[int(uid)] = grid_by_uav.get(int(uid), 0) + 1

            for heat_id, uav_id in sorted(scheduler.search_assignments.items()):
                heat_id = int(heat_id)
                uav_id = int(uav_id)
                point = scheduler.heat_by_id.get(heat_id)
                robot = runtime.robots[uav_id]
                position = np.asarray(robot.position, dtype=float)
                moved = float(np.linalg.norm(position - positions_before[uav_id]))

                goal = np.asarray(actions[uav_id][0], dtype=float)
                anchor = np.asarray(point.position, dtype=float) if point else None
                distance = (
                    float(np.linalg.norm(position - anchor)) if anchor is not None
                    else None
                )

                hops = hop_log.get(uav_id, [])
                hop = hops[0] if hops else None
                if hop is None:
                    route_kind = "no_route" if hops else "not_called"
                else:
                    route_kind = "override"

                agent = getattr(scheduler, "_marvel_agents", {}).get(uav_id)
                current_node = None
                reach_goal = None
                reachable_count = None
                if agent is not None:
                    state = observed_graph_module._graph_state(agent)
                    if state is not None:
                        coords, adj, start = state
                        current_node = [
                            round(float(v), 2) for v in np.asarray(coords[start])[:2]
                        ]
                        seen = np.zeros(len(coords), dtype=bool)
                        from collections import deque
                        queue = deque([int(start)])
                        seen[int(start)] = True
                        while queue:
                            u = queue.popleft()
                            for raw in np.flatnonzero(adj[u] == 0):
                                v = int(raw)
                                if v != u and 0 <= v < len(coords) and not seen[v]:
                                    seen[v] = True
                                    queue.append(v)
                        reachable_count = int(seen.sum())
                        if anchor is not None:
                            reach = coords[seen]
                            reach_goal = round(float(
                                np.linalg.norm(reach - anchor[:2], axis=1).min()
                            ), 2)

                rows.append({
                    "mission_step": mission,
                    "heat_id": heat_id,
                    "uav_id": uav_id,
                    "x": round(float(position[0]), 3),
                    "y": round(float(position[1]), 3),
                    "distance_m": round(distance, 3) if distance is not None else None,
                    "goal_x": round(float(goal[0]), 3),
                    "goal_y": round(float(goal[1]), 3),
                    "goal_distance_m": round(float(np.linalg.norm(goal - position)), 3),
                    "moved_m": round(moved, 4),
                    "parked": int(moved < 1e-9),
                    "route_kind": route_kind,
                    "hop_x": round(float(hop[0][0]), 3) if hop else None,
                    "hop_y": round(float(hop[0][1]), 3) if hop else None,
                    "current_node_x": current_node[0] if current_node else None,
                    "current_node_y": current_node[1] if current_node else None,
                    "observed_reachable_nodes": reachable_count,
                    "observed_min_dist_to_heat_m": reach_goal,
                    "shield": ";".join(shield_events.get(uav_id, [])),
                    "grid_blocks": grid_by_uav.get(uav_id, 0),
                })
    finally:
        SafetyShield._record = original_record
        observed_graph_module.observed_next_hop = original_hop
        if had_scheduler_ref:
            scheduler_module.observed_next_hop = original_hop

    path = out / "stall_trace.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {"activation": activation, "rows": len(rows), "path": str(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uavs", type=int, default=60)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / f"extended_search_stall_trace_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "system_info.txt").write_text(
        json.dumps(system_info(), indent=2), encoding="utf-8"
    )

    print(f"[stall] uavs={args.uavs} steps={args.steps} ...", flush=True)
    started = time.perf_counter()
    result = run(args.uavs, args.steps, out)
    print(f"[stall] {json.dumps(result, default=str)[:200]}", flush=True)
    print(f"[stall] elapsed {time.perf_counter() - started:.1f}s", flush=True)
    print(f"[stall] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
