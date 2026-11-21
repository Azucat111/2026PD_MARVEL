"""Extended graph-motion clearance consistency audit.

Measurement only: nothing in the production tree is modified.  Every probe
reuses the *authoritative* runtime primitive
(`ObstacleManager.check_collision`, the same one `SafetyShield` and
`runtime.step` call) so the audit compares like with like.

Modes
-----
``quantify``  PHASE 3: node/edge clearance statistics at 4/8/16/30/60 UAVs.
``examples``  PHASE 2: deterministic M1 and M2 counter-examples with the
              full geometry of each.

Usage
-----
    python scripts/audit_graph_clearance.py --mode quantify --uavs 4 8 16 30 60
    python scripts/audit_graph_clearance.py --mode examples
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from benchmark_live_gppo import (  # noqa: E402
    ACTIVATION_TARGET,
    CHECKPOINT,
    SEED,
    install_activation_state,
)
from utils.scenario_config import load_and_validate_scenario  # noqa: E402
from utils.simulation_runtime import SimulationRuntime  # noqa: E402
from utils.policy_adapter import MARVELPolicyAdapter  # noqa: E402
from utils.safety_shield import SafetyShield  # noqa: E402
from parameter import CELL_SIZE, NODE_RESOLUTION  # noqa: E402

SCENARIO = ROOT / "configs" / "scenarios" / "extended_live_gppo_full.yaml"

# The physical footprint used by SafetyShield and by runtime.step.
FOOTPRINT_RADIUS = 0.2
# Swept-segment sampling. 0.05 m is 4x finer than the 0.2 m footprint, so a
# segment cannot slip past an obstacle between samples.
SWEEP_STEP = 0.05


def build(uavs, *, steps=0, seed=SEED,
          activation_target=ACTIVATION_TARGET, policy=None):
    config = load_and_validate_scenario(SCENARIO)
    config["robots"] = [{
        "id_range": [0, uavs - 1], "type": "explorer", "team_id": 1,
        "config": {"fov": 120, "sensor_range": 10.0, "velocity": 1.0,
                   "yaw_rate": 35, "initial_positions": "random_safe"},
    }]
    config.setdefault("scenario", {})["random_seed"] = seed

    from integrations.gppo.checkpoint import load_frozen_gppo
    from integrations.gppo.config_profile import prepare_gppo_config

    config["task_scheduler"] = {
        "mode": "gppo", "checkpoint": str(CHECKPOINT),
        "base_position": [0.0, 0.0], "search_scenario_seed": seed,
    }
    _, checkpoint = load_frozen_gppo(str(CHECKPOINT), device="cpu")
    config = prepare_gppo_config(config, checkpoint)
    if policy:
        config["policy"] = dict(policy)

    np.random.seed(seed)
    runtime = SimulationRuntime(config)
    observations = runtime.reset()
    adapter = MARVELPolicyAdapter(runtime)
    adapter.setup()
    activation = None
    if activation_target is not None:
        activation = install_activation_state(
            runtime, adapter, target=activation_target
        )

    interval = max(1, int(runtime._mission_step_interval()))
    for _ in range(steps):
        actions = adapter.get_actions(observations)
        for _tick in range(interval):
            observations, _info = runtime.step(actions)

    return runtime, adapter, activation


# ----------------------------------------------------------------------
# clearance primitives (all reuse the runtime's own check_collision)
# ----------------------------------------------------------------------

def point_free(runtime, point, radius=0.0):
    return not runtime.obstacles.check_collision(
        np.asarray(point, dtype=float), radius=radius
    )[0]


def segment_clear(runtime, start, end, radius=FOOTPRINT_RADIUS,
                  step=SWEEP_STEP):
    """True when the whole swept segment keeps `radius` clearance."""

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    length = float(np.linalg.norm(end - start))
    samples = max(1, int(np.ceil(length / step)))

    for index in range(samples + 1):
        alpha = index / samples
        point = start + alpha * (end - start)
        if runtime.obstacles.check_collision(point, radius=radius)[0]:
            return False
    return True


def minimum_clearance(runtime, point, maximum=1.0, step=0.02):
    """Largest radius at which `point` is still collision-free."""

    radius = 0.0
    while radius <= maximum:
        if runtime.obstacles.check_collision(
            np.asarray(point, dtype=float), radius=radius
        )[0]:
            return round(radius - step, 4)
        radius += step
    return float(maximum)


def node_edges(adapter):
    """Canonical undirected edge set of the current graph."""

    edges = set()
    for node in adapter._node_manager.nodes_dict.__iter__():
        coords = tuple(np.round(np.asarray(node.data.coords, float), 3))
        for neighbour in node.data.neighbor_list:
            other = tuple(np.round(np.asarray(neighbour, float), 3))
            if coords == other:
                continue
            edges.add((coords, other) if coords < other else (other, coords))
    return edges


def graph_stats(adapter):
    node_coords = [
        np.asarray(node.data.coords, dtype=float)
        for node in adapter._node_manager.nodes_dict.__iter__()
    ]
    return node_coords, node_edges(adapter)


# ----------------------------------------------------------------------
# PHASE 3 -- quantify
# ----------------------------------------------------------------------

def quantify(uavs: int, steps: int):
    runtime, adapter, _ = build(uavs, steps=steps)
    coords, edges = graph_stats(adapter)

    # A: point-free in ground truth (radius 0)
    # B: footprint-valid at the physical radius
    point_free_nodes = [c for c in coords if point_free(runtime, c, 0.0)]
    footprint_ok = [c for c in coords if point_free(runtime, c, FOOTPRINT_RADIUS)]
    # C: admitted by the graph but the footprint does not fit
    insufficient = [c for c in coords
                    if point_free(runtime, c, 0.0)
                    and not point_free(runtime, c, FOOTPRINT_RADIUS)]
    inside = [c for c in coords if not point_free(runtime, c, 0.0)]

    # D: edges with both endpoints footprint-valid but an invalid swept path
    endpoint_ok = {tuple(np.round(c, 3)) for c in footprint_ok}
    edge_total = len(edges)
    edge_endpoints_ok = 0
    edge_swept_bad = 0
    for start, end in edges:
        if start not in endpoint_ok or end not in endpoint_ok:
            continue
        edge_endpoints_ok += 1
        if not segment_clear(runtime, np.asarray(start), np.asarray(end)):
            edge_swept_bad += 1

    # E: does the live planner actually select C nodes / D edges?
    interval = max(1, int(runtime._mission_step_interval()))
    observations = runtime._get_observations()
    selected = 0
    selected_bad = 0
    segment_bad = 0
    checks = 0
    parked_before = np.asarray(
        [r.position for r in runtime.robots], dtype=float
    )
    for _ in range(5):
        actions = adapter.get_actions(observations)
        for index, robot in enumerate(runtime.robots):
            waypoint = np.asarray(actions[index][0], dtype=float)
            checks += 1
            if not point_free(runtime, waypoint, FOOTPRINT_RADIUS):
                selected_bad += 1
            if not segment_clear(runtime, robot.position, waypoint):
                segment_bad += 1
            selected += 1
        for _tick in range(interval):
            observations, _info = runtime.step(actions)
    moved = np.linalg.norm(
        np.asarray([r.position for r in runtime.robots], dtype=float)
        - parked_before, axis=1
    )

    return {
        "uavs": uavs,
        "steps": steps,
        "nodes_total": len(coords),
        "nodes_point_free": len(point_free_nodes),
        "nodes_footprint_valid": len(footprint_ok),
        "nodes_inside_obstacle": len(inside),
        "nodes_insufficient_clearance": len(insufficient),
        "nodes_insufficient_clearance_fraction": (
            len(insufficient) / len(coords) if coords else 0.0
        ),
        "edges_total": edge_total,
        "edges_endpoints_valid": edge_endpoints_ok,
        "edges_swept_invalid": edge_swept_bad,
        "edges_swept_invalid_fraction": (
            edge_swept_bad / edge_endpoints_ok if edge_endpoints_ok else 0.0
        ),
        "selected_waypoints": selected,
        "selected_waypoints_invalid": selected_bad,
        "selected_waypoint_invalid_fraction": (
            selected_bad / selected if selected else 0.0
        ),
        "selected_segments_invalid": segment_bad,
        "selected_segment_invalid_fraction": (
            segment_bad / selected if selected else 0.0
        ),
        "uavs_stationary_last_step": int(np.count_nonzero(moved < 1e-9)),
    }


# ----------------------------------------------------------------------
# PHASE 2 -- deterministic examples
# ----------------------------------------------------------------------

def examples(steps: int = 25):
    runtime, adapter, _ = build(8, steps=steps)
    interval = max(1, int(runtime._mission_step_interval()))
    observations = runtime._get_observations()

    shield_events: list[tuple[int, str]] = []
    original_record = SafetyShield._record

    def capture(self, robot_id, step, reason):
        shield_events.append((int(robot_id), str(reason)))
        return original_record(self, robot_id, step, reason)

    SafetyShield._record = capture

    m1 = m2 = None
    try:
        for mission in range(1, steps + 6):
            actions = adapter.get_actions(observations)
            positions = np.asarray(
                [r.position for r in runtime.robots], dtype=float
            )
            shield_events.clear()
            collisions: dict[int, dict] = {}

            for _tick in range(interval):
                observations, info = runtime.step(actions)
                for event in info["collisions"]:
                    index = event.get("robot_id")
                    if index is not None and event.get("type") == "grid_obstacle":
                        collisions[int(index)] = event

            moved = np.linalg.norm(
                np.asarray([r.position for r in runtime.robots], dtype=float)
                - positions, axis=1
            )

            for index, robot in enumerate(runtime.robots):
                if moved[index] >= 1e-9:
                    continue
                target = np.asarray(actions[index][0], dtype=float)
                events = [reason for uid, reason in shield_events
                          if uid == index]
                record = {
                    "mission_step": mission,
                    "uav": index,
                    "start_position": [round(float(v), 3) for v in positions[index]],
                    "target_position": [round(float(v), 3) for v in target],
                    "target_distance": round(
                        float(np.linalg.norm(target - positions[index])), 3
                    ),
                    "target_cell": [int(target[0]), int(target[1])],
                    "target_cell_occupied": bool(
                        runtime.obstacles.check_collision(target, radius=0.0)[0]
                    ),
                    "target_minimum_clearance_m": minimum_clearance(
                        runtime, target
                    ),
                    "shield_events": events,
                    "shield_blocked": bool(events),
                    "runtime_step_collision": collisions.get(index),
                    "current_node": [
                        round(float(v), 3) for v in
                        np.asarray(adapter._snap_to_nearest_node(
                            positions[index]
                        ), dtype=float)
                    ],
                }

                # M1: graph-valid node whose clearance is below the footprint.
                if (
                    m1 is None
                    and record["shield_blocked"]
                    and record["target_cell_occupied"] is False
                    and record["target_minimum_clearance_m"] < FOOTPRINT_RADIUS
                ):
                    m1 = record
                # M2: nominally free target whose first motion segment collides.
                if (
                    m2 is None
                    and record["runtime_step_collision"] is not None
                    and record["target_cell_occupied"] is False
                    and not record["shield_blocked"]
                ):
                    record["first_collision_position"] = (
                        record["runtime_step_collision"].get("position")
                    )
                    m2 = record
            if m1 and m2:
                break
    finally:
        SafetyShield._record = original_record

    return {"radius": FOOTPRINT_RADIUS, "sweep_step": SWEEP_STEP,
            "M1": m1, "M2": m2}


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------

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


def contract():
    """The geometry contract, captured for the artifact."""

    return {
        "layer_1_node_admission": {
            "code": "utils/utils.py::get_updating_node_coords + "
                    "get_free_and_connected_map (called from "
                    "NodeManager.update_graph)",
            "occupancy_source": "the adapter's BELIEF map (visible_cells)",
            "resolution_m": CELL_SIZE,
            "node_lattice_m": NODE_RESOLUTION,
            "test": "belief cell == FREE and in the robot's 8-connected "
                    "free component",
            "point_or_footprint": "point (single belief cell)",
            "collision_radius_m": None,
            "obstacle_inflation": None,
            "endpoint_test": "yes (the cell itself)",
            "segment_test": "no",
            "boundary": "assert 0 <= cell < map shape",
        },
        "layer_2_edge_admission": {
            "code": "utils/node_manager.py::Node.update_neighbor_nodes -> "
                    "utils/utils.py::check_collision(start, end, map_info)",
            "occupancy_source": "the agent's local window of the BELIEF map",
            "resolution_m": CELL_SIZE,
            "test": "Bresenham walk; OCCUPIED or UNKNOWN => collision",
            "point_or_footprint": "point (walked centreline cells)",
            "collision_radius_m": None,
            "obstacle_inflation": None,
            "endpoint_test": "NO -- the loop breaks on the end cell before "
                             "testing it",
            "segment_test": "yes, but centreline only and only within the "
                            "local window",
            "boundary": "out-of-window cells end the walk untested "
                        "(returns no collision)",
        },
        "layer_3_physical_validity": {
            "code": "utils/obstacle_manager.py::check_collision "
                    "(SafetyShield.filter_actions and runtime.step)",
            "occupancy_source": "GROUND TRUTH (_file_grid)",
            "resolution_m": 1.0,
            "test": "5-point star: centre + (+-radius, 0) + (0, +-radius)",
            "point_or_footprint": "discrete 4-point footprint approximation",
            "collision_radius_m": FOOTPRINT_RADIUS,
            "obstacle_inflation": "implicit via the 5-point sample; not a "
                                  "true Minkowski disc",
            "endpoint_test": "yes (centre)",
            "segment_test": "NO -- only the post-step position is tested",
            "boundary": "outside frame bounds => boundary collision",
        },
        "why_they_disagree": [
            "different occupancy source: belief vs ground truth "
            "(belief is ~5% false-free even with occlusion on)",
            "different resolution: 0.4 m belief cell vs 1.0 m truth cell",
            "the graph applies NO clearance radius; physics demands 0.2 m "
            "in four directions",
            "_update_belief marks a whole 1 m runtime cell FREE, adding a "
            "~0.2 m free halo beyond the sensed cell",
            "the graph tests the centreline; physics tests only the "
            "endpoint -- so each is blind where the other is strict",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quantify", "examples"), required=True)
    parser.add_argument("--uavs", type=int, nargs="+", default=[4, 8, 16, 30, 60])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / f"extended_clearance_consistency_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "system_info.txt").write_text(
        json.dumps(system_info(), indent=2), encoding="utf-8"
    )
    (out / "geometry_contract.json").write_text(
        json.dumps(contract(), indent=2), encoding="utf-8"
    )

    if args.mode == "examples":
        payload = examples(steps=args.steps)
        (out / "clearance_examples.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        for key in ("M1", "M2"):
            entry = payload[key]
            print(f"[examples] {key}: "
                  f"{'found' if entry else 'NOT FOUND'}", flush=True)
            if entry:
                print(f"   {json.dumps(entry, default=str)}", flush=True)
        print(f"[examples] wrote {out}")
        return 0

    rows = []
    for uavs in args.uavs:
        print(f"[quantify] uavs={uavs} steps={args.steps} ...", flush=True)
        result = quantify(uavs, args.steps)
        rows.append(result)
        print(
            f"[quantify]   nodes={result['nodes_total']} "
            f"insufficient_clearance="
            f"{result['nodes_insufficient_clearance']} "
            f"({result['nodes_insufficient_clearance_fraction']:.2%}) "
            f"edges_swept_invalid={result['edges_swept_invalid']} "
            f"({result['edges_swept_invalid_fraction']:.2%}) "
            f"selected_invalid="
            f"{result['selected_waypoint_invalid_fraction']:.2%} "
            f"segment_invalid="
            f"{result['selected_segment_invalid_fraction']:.2%} "
            f"stationary={result['uavs_stationary_last_step']}/{uavs}",
            flush=True,
        )

    with open(out / "clearance_mismatch.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / "clearance_mismatch.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    print(f"[quantify] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
