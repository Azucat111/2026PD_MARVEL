"""Extended sensing / park-deadlock correctness audit.

Measurement only.  Nothing in the production tree is modified; the A/B is
driven by writing `sensor.params.los_occlusion` into the scenario dict
before the runtime is constructed, which is the same path
`prepare_gppo_config` itself uses.

Two things are measured:

1. `trace`  -- the causal chain of one deterministically parked UAV.
2. `ab`     -- `los_occlusion` false vs true, at 4/8/16/30/60 UAVs, on the
   live-GPPO controlled activation state.

Usage:
    python scripts/audit_sensing_correctness.py --mode trace
    python scripts/audit_sensing_correctness.py --mode ab --uavs 4 8 16 30 60
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
from types import SimpleNamespace

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
from parameter import CELL_SIZE, FREE  # noqa: E402

SCENARIO = ROOT / "configs" / "scenarios" / "extended_live_gppo_full.yaml"


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------

def build(uavs, *, los_occlusion, steps=0, seed=SEED,
          activation_target=ACTIVATION_TARGET):
    """Build the live-GPPO runtime with an explicit `los_occlusion`."""

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

    # prepare_gppo_config forces this True; the audit overrides it on the
    # same dict the runtime reads, so the real construction path is used.
    config.setdefault("sensor", {}).setdefault("params", {})[
        "los_occlusion"
    ] = bool(los_occlusion)

    config["policy"] = {"device": "cuda", "batch_inference": True}

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
# metrics
# ----------------------------------------------------------------------

def belief_truth_disagreement(runtime, adapter):
    """Fraction of belief-FREE cells that are truly occupied."""

    map_info = adapter._build_map_info()
    belief = np.asarray(map_info.map)
    source = np.asarray(runtime.obstacles._file_grid)

    height, width = belief.shape
    ys = (np.arange(height) + 0.5) * CELL_SIZE
    xs = (np.arange(width) + 0.5) * CELL_SIZE
    rows = np.clip((ys / 1.0).astype(int), 0, source.shape[0] - 1)
    cols = np.clip((xs / 1.0).astype(int), 0, source.shape[1] - 1)
    truth = source[np.ix_(rows, cols)] > 0

    free = belief == FREE
    total = int(free.sum())
    false_free = int((free & truth).sum())

    return {
        "belief_free_cells": total,
        "belief_false_free_cells": false_free,
        "belief_false_free_fraction": (
            false_free / total if total else 0.0
        ),
    }


def graph_obstacle_stats(runtime, adapter):
    """Graph nodes inside a true obstacle, and inside the shield's radius."""

    node_manager = adapter._node_manager
    inside = near = total = 0
    for node in node_manager.nodes_dict.__iter__():
        coords = np.asarray(node.data.coords, dtype=float)
        total += 1
        if runtime.obstacles.check_collision(coords, radius=0.0)[0]:
            inside += 1
        elif runtime.obstacles.check_collision(coords, radius=0.2)[0]:
            near += 1

    return {
        "graph_nodes": total,
        "nodes_inside_obstacle": inside,
        "nodes_inside_obstacle_fraction": inside / total if total else 0.0,
        "nodes_within_shield_radius": near,
    }


def run_arm(uavs, *, los_occlusion, steps, measure_shield=True):
    """One A/B arm: build, advance, and collect every audit metric."""

    runtime, adapter, activation = build(
        uavs, los_occlusion=los_occlusion, steps=0
    )

    interval = max(1, int(runtime._mission_step_interval()))
    observations = runtime._get_observations()

    shield_counts: dict[str, int] = {}
    original_record = SafetyShield._record

    def counting(self, robot_id, step, reason):
        key = str(reason).split(":")[0]
        shield_counts[key] = shield_counts.get(key, 0) + 1
        return original_record(self, robot_id, step, reason)

    if measure_shield:
        SafetyShield._record = counting

    parked_run = np.zeros(uavs, dtype=int)
    parked_max = np.zeros(uavs, dtype=int)
    invalid_targets = 0
    target_checks = 0
    collision_skips = 0
    step_times = []
    rate_curve = []

    try:
        for _ in range(steps):
            actions = adapter.get_actions(observations)

            for index, robot in enumerate(runtime.robots):
                wp = np.asarray(actions[index][0], dtype=float)
                target_checks += 1
                if runtime.obstacles.check_collision(wp, radius=0.2)[0]:
                    invalid_targets += 1

            positions_before = np.asarray(
                [r.position for r in runtime.robots], dtype=float
            )
            start = time.perf_counter()
            for _tick in range(interval):
                observations, info = runtime.step(actions)
                collision_skips += sum(
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
            rate_curve.append(float(runtime.exploration_rate))
    finally:
        if measure_shield:
            SafetyShield._record = original_record

    metrics = {
        "uavs": uavs,
        "los_occlusion": bool(los_occlusion),
        "steps": steps,
        "parked_uavs_half_window": int(np.count_nonzero(
            parked_max >= max(1, steps) // 2
        )),
        "parked_max_consecutive_steps": int(parked_max.max()) if uavs else 0,
        "invalid_targets": invalid_targets,
        "target_checks": target_checks,
        "invalid_target_fraction": (
            invalid_targets / target_checks if target_checks else 0.0
        ),
        "dynamics_collision_skips": collision_skips,
        "shield_blocks": dict(shield_counts),
        "shield_static_obstacle": int(shield_counts.get("static_obstacle", 0)),
        "exploration_rate_final": rate_curve[-1] if rate_curve else None,
        "exploration_rate_start": rate_curve[0] if rate_curve else None,
        "steady_mean_seconds": (
            statistics.fmean(step_times) if step_times else None
        ),
        "activation": activation,
        "graph": graph_obstacle_stats(runtime, adapter),
        "belief": belief_truth_disagreement(runtime, adapter),
        "rate_curve": rate_curve,
    }
    return runtime, adapter, metrics


# ----------------------------------------------------------------------
# PHASE 2 -- deterministic deadlock trace
# ----------------------------------------------------------------------

def deadlock_trace(los_occlusion: bool, uavs: int = 8, max_steps: int = 60):
    """Record one UAV from free motion into a permanent park."""

    runtime, adapter, _ = build(uavs, los_occlusion=los_occlusion, steps=0)
    interval = max(1, int(runtime._mission_step_interval()))
    observations = runtime._get_observations()

    shield_events: list[tuple[int, str]] = []
    original_record = SafetyShield._record

    def capture(self, robot_id, step, reason):
        shield_events.append((int(robot_id), str(reason)))
        return original_record(self, robot_id, step, reason)

    SafetyShield._record = capture

    plain = np.zeros(uavs, dtype=int)
    parked_max = np.zeros(uavs, dtype=int)
    history = []

    try:
        for mission in range(1, max_steps + 1):
            actions = adapter.get_actions(observations)
            positions_before = np.asarray(
                [r.position for r in runtime.robots], dtype=float
            )
            shield_events.clear()
            collision_position = [None] * uavs

            for _tick in range(interval):
                observations, info = runtime.step(actions)
                for event in info["collisions"]:
                    index = event.get("robot_id")
                    if index is not None and event.get("type") == "grid_obstacle":
                        collision_position[int(index)] = event.get("position")

            moved = np.linalg.norm(
                np.asarray([r.position for r in runtime.robots], dtype=float)
                - positions_before, axis=1
            )
            plain = np.where(moved < 1e-9, plain + 1, 0)
            parked_max = np.maximum(parked_max, plain)

            history.append({
                "mission_step": mission,
                "uavs": [
                    {
                        "uav": index,
                        "position": [round(float(v), 3) for v in positions_before[index]],
                        "waypoint": [
                            round(float(actions[index][0][0]), 3),
                            round(float(actions[index][0][1]), 3),
                        ],
                        "distance_to_waypoint": round(float(np.linalg.norm(
                            np.asarray(actions[index][0], dtype=float)
                            - positions_before[index]
                        )), 3),
                        "moved": round(float(moved[index]), 6),
                        "waypoint_cell_occupied": bool(
                            runtime.obstacles.check_collision(
                                np.asarray(actions[index][0], dtype=float),
                                radius=0.0,
                            )[0]
                        ),
                        "waypoint_blocked_at_shield_radius": bool(
                            runtime.obstacles.check_collision(
                                np.asarray(actions[index][0], dtype=float),
                                radius=0.2,
                            )[0]
                        ),
                        "shield_events": [
                            reason for uid, reason in shield_events
                            if uid == index
                        ],
                        "dynamics_collision_at": (
                            collision_position[index]
                        ),
                    }
                    for index in range(uavs)
                ],
            })
    finally:
        SafetyShield._record = original_record

    # Earliest UAV parked for at least three consecutive steps.
    subject = None
    for index in range(uavs):
        run = 0
        for row in history:
            if row["uavs"][index]["moved"] < 1e-9:
                run += 1
                if run >= 3:
                    subject = (index, row["mission_step"] - 2)
                    break
            else:
                run = 0
        if subject:
            break

    window = []
    if subject:
        index, start = subject
        window = [
            {"mission_step": row["mission_step"], **row["uavs"][index]}
            for row in history
            if start <= row["mission_step"] <= start + 3
        ]

    return {
        "los_occlusion": bool(los_occlusion),
        "uavs": uavs,
        "steps": max_steps,
        "parked_uavs_half_window": int(np.count_nonzero(
            parked_max >= max(1, max_steps) // 2
        )),
        "subject_uav": subject[0] if subject else None,
        "subject_first_parked_step": subject[1] if subject else None,
        "window": window,
    }


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------

def long_horizon(uavs: int, steps: int, los_occlusion: bool = True):
    """PHASE 4/5: sensing shape and task progress over a long episode."""

    runtime, adapter, _ = build(uavs, los_occlusion=los_occlusion, steps=0)
    scheduler = adapter.scheduler
    interval = max(1, int(runtime._mission_step_interval()))
    observations = runtime._get_observations()

    # Count sensor output size without changing how it is produced.
    sensor = runtime.sensor_by_type.get("explorer", runtime.sensor)
    original_sense = sensor.sense
    visible = {"cells": 0, "calls": 0}

    def counting_sense(robot_state, occupancy_grid, all_positions):
        result = original_sense(robot_state, occupancy_grid, all_positions)
        visible["cells"] += int(len(result["visible_cells"]))
        visible["calls"] += 1
        return result

    sensor.sense = counting_sense

    parked_run = np.zeros(uavs, dtype=int)
    parked_max = np.zeros(uavs, dtype=int)
    rows = []
    try:
        for mission in range(1, steps + 1):
            positions_before = np.asarray(
                [r.position for r in runtime.robots], dtype=float
            )
            visible["cells"] = 0
            visible["calls"] = 0

            actions = adapter.get_actions(observations)
            for _tick in range(interval):
                observations, _info = runtime.step(actions)

            moved = np.linalg.norm(
                np.asarray([r.position for r in runtime.robots], dtype=float)
                - positions_before, axis=1
            )
            parked_run = np.where(moved < 1e-9, parked_run + 1, 0)
            parked_max = np.maximum(parked_max, parked_run)

            rows.append({
                "mission_step": mission,
                "exploration_rate": float(runtime.exploration_rate),
                "graph_nodes": int(adapter._node_manager.nodes_dict.__len__()),
                "parked_now": int(np.count_nonzero(moved < 1e-9)),
                "visible_cells_per_call": (
                    visible["cells"] / visible["calls"]
                    if visible["calls"] else 0.0
                ),
                "sensing_calls": visible["calls"],
                "search_assignments": len(
                    getattr(scheduler, "search_assignments", {})
                ),
                "relay_assignments": len(
                    getattr(scheduler, "relay_assignments", {})
                ),
                "serviced": len(getattr(scheduler, "serviced_heat_ids", set())),
            })
    finally:
        sensor.sense = original_sense

    return {
        "uavs": uavs,
        "los_occlusion": bool(los_occlusion),
        "steps": steps,
        "parked_uavs_half_window": int(np.count_nonzero(
            parked_max >= max(1, steps) // 2
        )),
        "first_completion_step": next(
            (r["mission_step"] for r in rows if r["serviced"] > 0), None
        ),
        "first_relay_step": next(
            (r["mission_step"] for r in rows if r["relay_assignments"] > 0), None
        ),
        "final_rate": rows[-1]["exploration_rate"] if rows else None,
        "rows": rows,
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


def write_csv(path: Path, rows, keys):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for row in rows:
            writer.writerow([row.get(key) for key in keys])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("trace", "ab", "long"), required=True)
    parser.add_argument("--uavs", type=int, nargs="+", default=[4, 8, 16, 30, 60])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / f"extended_sensing_correctness_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "system_info.txt").write_text(
        json.dumps(system_info(), indent=2), encoding="utf-8"
    )

    if args.mode == "trace":
        payload = [
            deadlock_trace(True),
            deadlock_trace(False),
        ]
        (out / "deadlock_trace.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        for entry in payload:
            print(
                f"[trace] los_occlusion={entry['los_occlusion']} "
                f"subject=uav{entry['subject_uav']} "
                f"first_parked_step={entry['subject_first_parked_step']} "
                f"parked_half={entry['parked_uavs_half_window']}/{entry['uavs']}",
                flush=True,
            )
        print(f"[trace] wrote {out}")
        return 0

    if args.mode == "long":
        payload = []
        for uavs in args.uavs:
            for los_occlusion in (False, True):
                print(
                    f"[long] los_occlusion={los_occlusion} uavs={uavs} "
                    f"steps={args.steps} ...",
                    flush=True,
                )
                result = long_horizon(
                    uavs, args.steps, los_occlusion=los_occlusion
                )
                payload.append(result)
                print(
                    f"[long]   parked_half="
                    f"{result['parked_uavs_half_window']}/{uavs} "
                    f"rate={result['final_rate']:.4f} "
                    f"completion={result['first_completion_step']} "
                    f"relay={result['first_relay_step']} "
                    f"visible/call={result['rows'][-1]['visible_cells_per_call']:.1f} "
                    f"nodes={result['rows'][-1]['graph_nodes']}",
                    flush=True,
                )

        (out / "long_horizon.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        write_csv(
            out / "long_horizon.csv",
            [
                {
                    "uavs": r["uavs"],
                    "los_occlusion": r["los_occlusion"],
                    "steps": r["steps"],
                    "parked_uavs_half_window": r["parked_uavs_half_window"],
                    "final_rate": round(r["final_rate"], 6),
                    "first_completion_step": r["first_completion_step"],
                    "first_relay_step": r["first_relay_step"],
                    "final_nodes": r["rows"][-1]["graph_nodes"],
                    "final_parked_now": r["rows"][-1]["parked_now"],
                    "visible_cells_per_call": round(
                        r["rows"][-1]["visible_cells_per_call"], 3
                    ),
                    "final_search_assignments": r["rows"][-1]["search_assignments"],
                    "final_serviced": r["rows"][-1]["serviced"],
                }
                for r in payload
            ],
            ["uavs", "los_occlusion", "steps", "parked_uavs_half_window",
             "final_rate", "first_completion_step", "first_relay_step",
             "final_nodes", "final_parked_now", "visible_cells_per_call",
             "final_search_assignments", "final_serviced"],
        )
        print(f"[long] wrote {out}")
        return 0

    rows = []
    for los_occlusion in (False, True):
        for uavs in args.uavs:
            print(
                f"[ab] los_occlusion={los_occlusion} uavs={uavs} "
                f"steps={args.steps} ...",
                flush=True,
            )
            _runtime, _adapter, metrics = run_arm(
                uavs, los_occlusion=los_occlusion, steps=args.steps
            )
            rows.append(metrics)
            print(
                f"[ab]   parked={metrics['parked_uavs_half_window']}/{uavs} "
                f"invalid_targets={metrics['invalid_target_fraction']:.2%} "
                f"nodes_in_obstacle="
                f"{metrics['graph']['nodes_inside_obstacle_fraction']:.2%} "
                f"false_free={metrics['belief']['belief_false_free_fraction']:.2%} "
                f"dyn_skips={metrics['dynamics_collision_skips']} "
                f"shield_static={metrics['shield_static_obstacle']} "
                f"rate={metrics['exploration_rate_final']:.4f}",
                flush=True,
            )

    write_csv(
        out / "occlusion_false_vs_true.csv",
        [
            {
                "uavs": r["uavs"],
                "los_occlusion": r["los_occlusion"],
                "parked_half_window": r["parked_uavs_half_window"],
                "parked_max_consecutive": r["parked_max_consecutive_steps"],
                "invalid_target_fraction": round(r["invalid_target_fraction"], 6),
                "invalid_targets": r["invalid_targets"],
                "dynamics_collision_skips": r["dynamics_collision_skips"],
                "shield_static_obstacle": r["shield_static_obstacle"],
                "shield_uav_proximity": r["shield_blocks"].get("uav_proximity", 0),
                "graph_nodes": r["graph"]["graph_nodes"],
                "nodes_inside_obstacle": r["graph"]["nodes_inside_obstacle"],
                "nodes_inside_obstacle_fraction": round(
                    r["graph"]["nodes_inside_obstacle_fraction"], 6
                ),
                "nodes_within_shield_radius": r["graph"]["nodes_within_shield_radius"],
                "belief_free_cells": r["belief"]["belief_free_cells"],
                "belief_false_free_cells": r["belief"]["belief_false_free_cells"],
                "belief_false_free_fraction": round(
                    r["belief"]["belief_false_free_fraction"], 6
                ),
                "exploration_rate_start": round(r["exploration_rate_start"], 6),
                "exploration_rate_final": round(r["exploration_rate_final"], 6),
                "steady_mean_seconds": round(r["steady_mean_seconds"], 4),
            }
            for r in rows
        ],
        ["uavs", "los_occlusion", "parked_half_window", "parked_max_consecutive",
         "invalid_target_fraction", "invalid_targets", "dynamics_collision_skips",
         "shield_static_obstacle", "shield_uav_proximity", "graph_nodes",
         "nodes_inside_obstacle", "nodes_inside_obstacle_fraction",
         "nodes_within_shield_radius", "belief_free_cells",
         "belief_false_free_cells", "belief_false_free_fraction",
         "exploration_rate_start", "exploration_rate_final",
         "steady_mean_seconds"],
    )
    (out / "occlusion_ab.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    print(f"[ab] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
