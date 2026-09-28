"""Old vs heap Dijkstra parity and timing on live extended graphs.

The pytest suite covers 4/8/16 UAVs; 30 and 60 are too slow to sit in the
regression suite, so they are verified here and recorded in the artifact.

Usage:
    python scripts/verify_dijkstra_parity.py --uavs 4 8 16 30 60 --steps 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "platform_tests"))

from utils.scenario_config import load_and_validate_scenario  # noqa: E402
from utils.simulation_runtime import SimulationRuntime  # noqa: E402
from utils.policy_adapter import MARVELPolicyAdapter  # noqa: E402
from platform_tests.test_dijkstra_parity import (  # noqa: E402
    reference_dijkstra,
    reference_planner,
)

SCENARIO = ROOT / "configs" / "scenarios" / "baseline_maps_test.yaml"
MAX_SOURCES = 8


def build(uavs: int, steps: int):
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


def sources(manager, adapter):
    keys = []
    for agent in adapter.agents[:MAX_SOURCES]:
        node = manager.nodes_dict.nearest_neighbors(
            np.asarray(agent.location, dtype=float).tolist(), 1
        )[0].data.coords
        keys.append((node[0], node[1]))

    degrees = sorted(
        (len(n.data.neighbor_list), (n.data.coords[0], n.data.coords[1]))
        for n in manager.nodes_dict.__iter__()
    )
    if degrees:
        keys.append(degrees[0][1])
        keys.append(degrees[-1][1])

    seen, unique = set(), []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def run(uavs: int, steps: int):
    runtime, adapter = build(uavs, steps)
    manager = adapter._node_manager
    locations = [robot.position for robot in runtime.robots]

    result = {
        "uavs": uavs,
        "nodes": int(manager.nodes_dict.__len__()),
        "edges": int(sum(
            len(n.data.neighbor_list) for n in manager.nodes_dict.__iter__()
        )),
        "sources_tested": 0,
        "distance_mismatches": 0,
        "key_set_mismatches": 0,
        "reachable_mismatches": 0,
        "unreachable_samples": 0,
        "prev_identity_differences": 0,
        "max_abs_distance_diff": 0.0,
        "reference_seconds_per_call": None,
        "heap_seconds_per_call": None,
        "planner_output_identical": None,
    }

    ref_total = heap_total = 0.0
    tested = 0

    for source in sources(manager, adapter):
        start = time.perf_counter()
        expected_dist, expected_prev = reference_dijkstra(manager, source)
        ref_total += time.perf_counter() - start

        start = time.perf_counter()
        actual_dist, actual_prev = manager.Dijkstra(source)
        heap_total += time.perf_counter() - start

        tested += 1

        if set(actual_dist) != set(expected_dist):
            result["key_set_mismatches"] += 1
            continue

        for key, value in expected_dist.items():
            got = actual_dist[key]
            if got != value:
                result["distance_mismatches"] += 1
                if value < 1e8 and got < 1e8:
                    result["max_abs_distance_diff"] = max(
                        result["max_abs_distance_diff"], abs(got - value)
                    )
            if (value < 1e8) != (got < 1e8):
                result["reachable_mismatches"] += 1

        for key in expected_prev:
            if key == source or expected_dist[key] >= 1e8:
                continue
            if expected_prev[key] != actual_prev[key]:
                result["prev_identity_differences"] += 1

        result["unreachable_samples"] += sum(
            1 for v in expected_dist.values() if v >= 1e8
        )

    result["sources_tested"] = tested
    if tested:
        result["reference_seconds_per_call"] = ref_total / tested
        result["heap_seconds_per_call"] = heap_total / tested
        result["dijkstra_speedup"] = (
            result["reference_seconds_per_call"]
            / result["heap_seconds_per_call"]
        )

    # Full planner output, old vs new, for the first few UAVs.
    identical = True
    for agent in adapter.agents[:4]:
        expected = reference_planner(manager, agent.location, locations)
        actual = manager.get_all_node_graph(agent.location, locations)
        for index, (a, b) in enumerate(zip(expected, actual)):
            if isinstance(a, np.ndarray):
                if not np.array_equal(a, b) or a.dtype != np.asarray(b).dtype:
                    identical = False
            elif a != b:
                identical = False
    result["planner_output_identical"] = bool(identical)

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uavs", type=int, nargs="+", default=[4, 8, 16, 30, 60])
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / f"dijkstra_parity_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for uavs in args.uavs:
        print(f"[dijkstra-parity] uavs={uavs} ...", flush=True)
        results.append(run(uavs, args.steps))
        r = results[-1]
        print(
            f"[dijkstra-parity] uavs={uavs} nodes={r['nodes']} "
            f"sources={r['sources_tested']} dist_mismatch={r['distance_mismatches']} "
            f"ref={r['reference_seconds_per_call']:.4f}s "
            f"heap={r['heap_seconds_per_call']:.4f}s "
            f"speedup={r.get('dijkstra_speedup', 0):.1f}x "
            f"planner_identical={r['planner_output_identical']}",
            flush=True,
        )

    (out / "dijkstra_parity.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    with open(out / "dijkstra_profile.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "uavs", "nodes", "edges", "sources_tested",
            "reference_s_per_call", "heap_s_per_call", "dijkstra_speedup",
            "distance_mismatches", "reachable_mismatches",
            "key_set_mismatches", "max_abs_distance_diff",
            "prev_identity_differences", "unreachable_samples",
            "planner_output_identical",
        ])
        for r in results:
            writer.writerow([
                r["uavs"], r["nodes"], r["edges"], r["sources_tested"],
                f"{r['reference_seconds_per_call']:.6f}",
                f"{r['heap_seconds_per_call']:.6f}",
                f"{r.get('dijkstra_speedup', 0):.2f}",
                r["distance_mismatches"], r["reachable_mismatches"],
                r["key_set_mismatches"], r["max_abs_distance_diff"],
                r["prev_identity_differences"], r["unreachable_samples"],
                r["planner_output_identical"],
            ])

    print(f"[dijkstra-parity] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
