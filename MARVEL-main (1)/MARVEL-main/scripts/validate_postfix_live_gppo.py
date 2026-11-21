"""POST-FIX live Search completion / refill validation.

Validation only: no production file is modified.  Runs the real integrated
system at e7246a8 -- extended geometry, occlusion-aware sensing, CUDA
batched MARVEL PolicyNet, the frozen GPPO checkpoint and GPPOTaskScheduler,
real Search, real Relay, real SafetyShield, frozen activation threshold and
frozen Search capacity -- on the controlled activation state.

Rows are appended to disk as the run proceeds so a long horizon that is
interrupted still leaves usable evidence.

Usage:
    python scripts/validate_postfix_live_gppo.py --uavs 8 --max-steps 450
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
    Timer,
    build,
    instrument,
)
from utils.safety_shield import SafetyShield  # noqa: E402

# Stop early once a completion, a refill, and this many further steps have
# all been observed -- or at --max-steps, whichever comes first.
POST_REFILL_WINDOW = 50
PARKING_CHECKPOINTS = (50, 120, 250, 450)


def run_scale(
    uavs: int,
    max_steps: int,
    out: Path,
    policy=None,
):
    runtime, adapter, activation = build(uavs, steps=0, policy=policy)
    scheduler = adapter.scheduler

    interval = max(1, int(runtime._mission_step_interval()))
    observations = runtime._get_observations()

    timer = Timer()
    restore = instrument(adapter, runtime, timer)

    allocation_steps = {"n": 0}
    allocator = scheduler.allocator
    original_allocate = allocator.allocate

    def tagging_allocate(slots, **kwargs):
        allocation_steps["n"] += 1
        return original_allocate(slots, **kwargs)

    allocator.allocate = tagging_allocate

    shield_counts: dict[str, int] = {}
    original_record = SafetyShield._record

    def counting(self, robot_id, step, reason):
        key = str(reason).split(":")[0]
        shield_counts[key] = shield_counts.get(key, 0) + 1
        return original_record(self, robot_id, step, reason)

    SafetyShield._record = counting

    search_rows: list[dict] = []
    event_rows: list[dict] = []
    refill_rows: list[dict] = []
    relay_rows: list[dict] = []
    step_rows: list[dict] = []
    parking_rows: list[dict] = []

    try:
        last_assignments: dict[int, int] = {}
        completed_at: dict[int, int] = {}
        serviced_seen: set[int] = set()
        service_start: dict[int, int] = {}
        arrival_step: dict[int, int] = {}
        first_completion = None
        first_refill = None
        refill_deadline = None
        first_relay = None
        relay_seen: set[int] = set()
        search_uavs_used: set[int] = set()
        travel: dict[int, float] = {}
        position_before: dict[int, np.ndarray] = {}

        for mission in range(1, max_steps + 1):
            before_calls = allocation_steps["n"]
            before_serviced = set(scheduler.serviced_heat_ids)

            positions_before = {
                int(r.robot_id): np.asarray(r.position, dtype=float).copy()
                for r in runtime.robots
            }

            start = time.perf_counter()
            actions = adapter.get_actions(observations)
            for _tick in range(interval):
                observations, info = runtime.step(actions)
            elapsed = time.perf_counter() - start

            grid_blocks = sum(
                1 for c in info["collisions"] if c.get("type") == "grid_obstacle"
            )

            moved = {}
            for r in runtime.robots:
                uid = int(r.robot_id)
                delta = float(np.linalg.norm(
                    np.asarray(r.position, dtype=float) - positions_before[uid]
                ))
                moved[uid] = delta
                if delta > 1e-9:
                    travel[uid] = travel.get(uid, 0.0) + delta

            allocated = allocation_steps["n"] - before_calls
            newly_serviced = sorted(
                int(x) for x in set(scheduler.serviced_heat_ids) - before_serviced
            )

            # --- Search per-UAV trace -------------------------------------
            for heat_id, uav_id in sorted(scheduler.search_assignments.items()):
                heat_id = int(heat_id)
                uav_id = int(uav_id)
                point = scheduler.heat_by_id.get(heat_id)
                if point is None:
                    continue
                distance = float(np.linalg.norm(
                    np.asarray(runtime.robots[uav_id].position, dtype=float)
                    - np.asarray(point.position, dtype=float)
                ))
                search_uavs_used.add(uav_id)

                if heat_id not in arrival_step and distance <= scheduler.search_reach_radius:
                    arrival_step[heat_id] = mission
                    event_rows.append({
                        "mission_step": mission, "event": "arrival",
                        "heat_id": heat_id, "uav_id": uav_id,
                        "distance": round(distance, 3),
                    })

                dwell = int(scheduler.search_dwell_by_heat.get(heat_id, 0))
                if dwell > 0 and heat_id not in service_start:
                    service_start[heat_id] = mission
                    event_rows.append({
                        "mission_step": mission, "event": "service_start",
                        "heat_id": heat_id, "uav_id": uav_id,
                        "distance": round(distance, 3),
                    })

                search_rows.append({
                    "mission_step": mission,
                    "heat_id": heat_id,
                    "uav_id": uav_id,
                    "distance_m": round(distance, 4),
                    "dwell": dwell,
                    "parked": int(moved.get(uav_id, 0.0) < 1e-9),
                    "task": str(getattr(runtime.robots[uav_id], "current_task", "")),
                    "x": round(float(runtime.robots[uav_id].position[0]), 4),
                    "y": round(float(runtime.robots[uav_id].position[1]), 4),
                })

            # --- allocation / refill events -------------------------------
            if allocated:
                new_assignments = {
                    int(k): int(v)
                    for k, v in scheduler.search_assignments.items()
                    if int(k) not in last_assignments
                }
                if first_completion is None:
                    if not event_rows or not any(
                        e["event"] == "initial_allocation" for e in event_rows
                    ):
                        event_rows.append({
                            "mission_step": mission, "event": "initial_allocation",
                            "heat_id": min(new_assignments) if new_assignments else -1,
                            "uav_id": new_assignments.get(
                                min(new_assignments), -1
                            ) if new_assignments else -1,
                            "distance": 0.0,
                        })
                else:
                    if first_refill is None:
                        first_refill = mission
                        refill_deadline = mission + POST_REFILL_WINDOW
                        event_rows.append({
                            "mission_step": mission, "event": "refill",
                            "heat_id": min(new_assignments) if new_assignments else -1,
                            "uav_id": -1, "distance": 0.0,
                        })
                    refill_rows.append({
                        "mission_step": mission,
                        "completed_heat": first_completion,
                        "new_assignments": json.dumps(new_assignments),
                        "search_assignments_after": json.dumps(
                            {int(k): int(v) for k, v in scheduler.search_assignments.items()}
                        ),
                        "serviced_total": len(scheduler.serviced_heat_ids),
                        "allocate_calls": allocated,
                    })

            for heat_id in newly_serviced:
                if heat_id in serviced_seen:
                    continue
                serviced_seen.add(heat_id)
                completed_at[heat_id] = mission
                if first_completion is None:
                    first_completion = mission
                event_rows.append({
                    "mission_step": mission, "event": "completion",
                    "heat_id": heat_id, "uav_id": -1, "distance": 0.0,
                })

            # --- Relay ----------------------------------------------------
            for target_uid, value in sorted(scheduler.relay_assignments.items()):
                target_uid = int(target_uid)
                helper = int(value[0]) if isinstance(value, (tuple, list)) else int(value)
                if target_uid in relay_seen:
                    continue
                relay_seen.add(target_uid)
                if first_relay is None:
                    first_relay = mission
                relay_rows.append({
                    "mission_step": mission, "event": "relay_assignment",
                    "target_uav": target_uid, "helper_uav": helper,
                    "serviced_total": len(scheduler.serviced_heat_ids),
                })
                event_rows.append({
                    "mission_step": mission, "event": "relay_assignment",
                    "heat_id": target_uid, "uav_id": helper, "distance": 0.0,
                })

            # --- step timing ---------------------------------------------
            if allocated:
                kind = "refill" if first_completion is not None else "allocation"
            else:
                kind = "steady"
            step_rows.append({
                "mission_step": mission,
                "kind": kind,
                "seconds": round(elapsed, 6),
                "grid_obstacle_blocks": grid_blocks,
                "allocate_calls": allocated,
                "parked_now": int(sum(1 for v in moved.values() if v < 1e-9)),
                "moving_now": int(sum(1 for v in moved.values() if v >= 1e-9)),
                "exploration_rate": float(runtime.exploration_rate),
                "active_search": len(scheduler.search_assignments),
                "active_relay": len(scheduler.relay_assignments),
                "serviced_total": len(scheduler.serviced_heat_ids),
            })

            if mission in PARKING_CHECKPOINTS:
                parking_rows.append({
                    "mission_step": mission,
                    "parked_now": step_rows[-1]["parked_now"],
                    "moving_now": step_rows[-1]["moving_now"],
                    "shield_static_obstacle": shield_counts.get("static_obstacle", 0),
                    "grid_obstacle_blocks": sum(
                        r["grid_obstacle_blocks"] for r in step_rows
                    ),
                    "exploration_rate": step_rows[-1]["exploration_rate"],
                })

            last_assignments = {
                int(k): int(v) for k, v in scheduler.search_assignments.items()
            }

            if refill_deadline is not None and mission >= refill_deadline:
                break
    finally:
        for obj, name, original in restore:
            setattr(obj, name, original)
        allocator.allocate = original_allocate
        SafetyShield._record = original_record

    n = max(1, len(step_rows))

    def summarize(kind):
        values = [r["seconds"] for r in step_rows if r["kind"] == kind]
        if not values:
            return {"n": 0, "mean": None, "p95": None}
        ordered = sorted(values)
        return {
            "n": len(values),
            "mean": statistics.fmean(values),
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        }

    import psutil
    result = {
        "uavs": uavs,
        "steps_run": len(step_rows),
        "activation": activation,
        "activated": bool(getattr(scheduler, "search_activated", False)),
        "activation_step": getattr(scheduler, "search_activation_step", None),
        "first_completion_step": first_completion,
        "first_refill_step": first_refill,
        "first_relay_step": first_relay,
        "arrivals": {int(k): int(v) for k, v in arrival_step.items()},
        "service_starts": {int(k): int(v) for k, v in service_start.items()},
        "completed": {int(k): int(v) for k, v in completed_at.items()},
        "search_completions": len(completed_at),
        "refills": len(refill_rows),
        "search_uavs_used": sorted(search_uavs_used),
        "mean_travel_m": (
            statistics.fmean(list(travel.values())) if travel else 0.0
        ),
        "shield_blocks": dict(shield_counts),
        "steady": summarize("steady"),
        "allocation": summarize("allocation"),
        "refill": summarize("refill"),
        "inclusive": {k: v / n for k, v in timer.inclusive.items()},
        "ram_mb": float(psutil.Process().memory_info().rss / (1024 * 1024)),
        "vram_peak_allocated_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            if torch.cuda.is_available() else None
        ),
        "vram_peak_reserved_mb": (
            float(torch.cuda.max_memory_reserved() / (1024 * 1024))
            if torch.cuda.is_available() else None
        ),
        "heat_points": len(getattr(scheduler, "heat_points", [])),
        "search_capacity": getattr(scheduler, "max_search_uavs", None),
        "search_service_steps": getattr(scheduler, "search_service_steps", None),
        "search_reach_radius": getattr(scheduler, "search_reach_radius", None),
        "activation_threshold": getattr(
            scheduler, "search_activation_threshold", None
        ),
    }

    # Persist immediately so an interrupted batch keeps its evidence.
    def append_csv(name, rows, keys):
        path = out / name
        exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not exists:
                writer.writerow(keys)
            for row in rows:
                writer.writerow([row.get(k) for k in keys])

    for row in search_rows:
        row["uavs"] = uavs
    for row in parking_rows:
        row["uavs"] = uavs
    for row in step_rows:
        row["uavs"] = uavs
    for row in event_rows:
        row["uavs"] = uavs
    for row in refill_rows:
        row["uavs"] = uavs
    for row in relay_rows:
        row["uavs"] = uavs

    append_csv("search_progress.csv", search_rows,
               ["uavs", "mission_step", "heat_id", "uav_id", "distance_m",
                "dwell", "parked", "task", "x", "y"])
    append_csv("search_events.csv", event_rows,
               ["uavs", "mission_step", "event", "heat_id", "uav_id",
                "distance"])
    append_csv("refill_events.csv", refill_rows,
               ["uavs", "mission_step", "completed_heat", "new_assignments",
                "search_assignments_after", "serviced_total", "allocate_calls"])
    append_csv("relay_events.csv", relay_rows,
               ["uavs", "mission_step", "event", "target_uav", "helper_uav",
                "serviced_total"])
    append_csv("parking_health.csv", parking_rows,
               ["uavs", "mission_step", "parked_now", "moving_now",
                "shield_static_obstacle", "grid_obstacle_blocks",
                "exploration_rate"])
    append_csv("timing.csv", step_rows,
               ["uavs", "mission_step", "kind", "seconds",
                "grid_obstacle_blocks", "allocate_calls", "parked_now",
                "moving_now", "exploration_rate", "active_search",
                "active_relay", "serviced_total"])

    return result


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
    parser.add_argument("--uavs", type=int, nargs="+", default=[8, 16, 30, 60])
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / f"extended_postfix_live_gppo_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "system_info.txt").write_text(
        json.dumps(system_info(), indent=2), encoding="utf-8"
    )

    results = []
    for uavs in args.uavs:
        print(f"[postfix] uavs={uavs} max_steps={args.max_steps} ...",
              flush=True)
        result = run_scale(uavs, args.max_steps, out,
                           policy={"device": "cuda", "batch_inference": True})
        results.append(result)
        print(
            f"[postfix]   steps={result['steps_run']} "
            f"completions={result['search_completions']} "
            f"refills={result['refills']} relay={result['first_relay_step']} "
            f"arrivals={sorted(result['arrivals'])} "
            f"steady={result['steady']['mean']} "

            f"shield={result['shield_blocks']}",
            flush=True,
        )
        (out / "summary.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )

    print(f"[postfix] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
