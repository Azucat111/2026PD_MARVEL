"""Full extended mission-step profile and GPPO allocator scaling.

Measurement only: every timer wraps existing calls by monkeypatching, no
production module is modified.

Parent/child timing records both inclusive and exclusive time, so nested
spans (for example PolicyNet preparation inside `update_planning_state`)
are attributed once.

Usage:
    python scripts/profile_full_pipeline.py --uavs 4 8 16 30 60 --steps 20
    python scripts/profile_full_pipeline.py --gppo-only
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from utils.scenario_config import load_and_validate_scenario  # noqa: E402
from utils.simulation_runtime import SimulationRuntime  # noqa: E402
from utils.policy_adapter import MARVELPolicyAdapter  # noqa: E402
from utils import agent as agent_module  # noqa: E402

HEURISTIC_SCENARIO = ROOT / "configs" / "scenarios" / "baseline_maps_test.yaml"
GPPO_SCENARIO = ROOT / "configs" / "scenarios" / "extended_gppo_scalability.yaml"
GPPO_CHECKPOINT = (
    Path("/home/nick/MARVEL") / "artifacts" / "phase14_v9_frozen"
    / "gppo_phase14_v9_ckpt80.pth"
)


class Timer:
    """Inclusive and exclusive wall time per named span."""

    def __init__(self):
        self.inclusive = defaultdict(float)
        self.exclusive = defaultdict(float)
        self.calls = defaultdict(int)
        self._stack = []

    def push(self, name):
        self._stack.append([name, time.perf_counter(), 0.0])

    def pop(self):
        name, start, child = self._stack.pop()
        elapsed = time.perf_counter() - start
        self.inclusive[name] += elapsed
        self.exclusive[name] += elapsed - child
        self.calls[name] += 1
        if self._stack:
            self._stack[-1][2] += elapsed

    def span(self, name):
        return _Span(self, name)


class _Span:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.timer.push(self.name)
        return self

    def __exit__(self, *exc):
        self.timer.pop()
        return False


def build(uavs: int, steps: int, gppo: bool, policy=None):
    scenario = GPPO_SCENARIO if gppo else HEURISTIC_SCENARIO
    config = load_and_validate_scenario(scenario)

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

    if gppo:
        from integrations.gppo.checkpoint import load_frozen_gppo
        from integrations.gppo.config_profile import prepare_gppo_config

        config["task_scheduler"] = {
            "mode": "gppo",
            "checkpoint": str(GPPO_CHECKPOINT),
            "base_position": [0.0, 0.0],
            "search_scenario_seed": 12345,
        }
        _, checkpoint = load_frozen_gppo(str(GPPO_CHECKPOINT), device="cpu")
        config = prepare_gppo_config(config, checkpoint)

    if policy:
        config["policy"] = dict(policy)

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


def instrument(adapter, runtime, timer: Timer):
    """Wrap the pipeline; every span is entered at most once per call."""

    restore = []

    def wrap(obj, name, span, owner=None):
        original = getattr(obj, name)

        def wrapped(*args, **kwargs):
            with timer.span(span):
                return original(*args, **kwargs)

        setattr(obj, name, wrapped)
        restore.append((obj, name, original))

    # --- MARVEL adapter stages -------------------------------------------
    wrap(adapter, "_update_belief", "policy.belief")
    wrap(adapter, "_build_map_info", "policy.map_info")

    wrap(agent_module.Agent, "update_graph", "policy.graph")
    wrap(agent_module.Agent, "update_planning_state", "policy.planning_state")

    # Inside planning state: shared graph vs the per-UAV remainder.
    manager = adapter._node_manager
    wrap(manager, "_shared_graph", "planning.shared_graph")

    # Inside observation: tensor construction vs heading candidates.
    wrap(agent_module.Agent, "compute_best_heading", "policy.heading_candidates")

    wrap(agent_module.Agent, "get_observation", "policy.observation")

    # PolicyNet: the forward pass is the module call itself.
    policy_net = adapter.policy_net
    original_call = type(policy_net).__call__
    forward_name = "policy.policynet_forward"
    original_forward = policy_net.forward

    def timed_forward(*args, **kwargs):
        with timer.span(forward_name):
            return original_forward(*args, **kwargs)

    policy_net.forward = timed_forward
    restore.append((policy_net, "forward", original_forward))

    wrap(agent_module.Agent, "select_next_waypoint", "policy.policynet")
    # Batched path decodes without re-entering select_next_waypoint.
    wrap(agent_module.Agent, "decode_waypoint", "policy.policynet_decode")
    # Batched path: packing, transfer and forward inside `_batched_actions`.
    wrap(adapter, "_batched_actions", "pipeline.batched_actions")

    # Scheduler (heuristic or GPPO) at its entry point.
    wrap(adapter.scheduler, "apply", "scheduler.apply")

    # --- runtime stages ---------------------------------------------------
    wrap(runtime, "_get_observations", "runtime.sensing")
    wrap(runtime, "_update_explored_cells", "runtime.sensing_explored")
    wrap(runtime.comm, "get_topology", "runtime.comm")
    wrap(runtime.tasks, "update", "runtime.tasks")
    wrap(runtime.obstacles, "check_collision", "runtime.collision")
    wrap(runtime.shield, "filter_actions", "runtime.shield")
    for key, model in runtime.dynamics_by_type.items():
        wrap(model, "step", "runtime.motion")

    return restore


def profile(uavs: int, steps: int, measure_steps: int, gppo: bool, policy=None):
    runtime, adapter = build(uavs, steps, gppo, policy=policy)

    interval = max(1, int(runtime._mission_step_interval()))
    observations = runtime._get_observations()

    timer = Timer()
    restore = instrument(adapter, runtime, timer)

    step_times = []
    try:
        for _ in range(measure_steps):
            start = time.perf_counter()

            with timer.span("pipeline.policy_total"):
                actions = adapter.get_actions(observations)

            with timer.span("runtime.step_total"):
                for _tick in range(interval):
                    observations, _info = runtime.step(actions)

            step_times.append(time.perf_counter() - start)
    finally:
        for obj, name, original in restore:
            setattr(obj, name, original)

    ordered = sorted(step_times)
    out = {
        "uavs": uavs,
        "gppo": gppo,
        "steps_measured": len(step_times),
        "interval": interval,
        "mean_step_seconds": statistics.fmean(step_times) if step_times else None,
        "p50_step_seconds": ordered[len(ordered) // 2] if ordered else None,
        "p95_step_seconds": (
            ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
            if ordered else None
        ),
        "inclusive": {},
        "exclusive": {},
        "calls_per_mission_step": {},
    }

    n = max(1, len(step_times))
    for name, value in timer.inclusive.items():
        out["inclusive"][name] = value / n
    for name, value in timer.exclusive.items():
        out["exclusive"][name] = value / n
    for name, value in timer.calls.items():
        out["calls_per_mission_step"][name] = value / n

    manager = adapter._node_manager
    out["graph_nodes"] = int(manager.nodes_dict.__len__())
    out["graph_edges"] = int(sum(
        len(node.data.neighbor_list) for node in manager.nodes_dict.__iter__()
    ))
    out["exploration_rate"] = float(getattr(runtime, "exploration_rate", 0.0))

    import psutil
    out["ram_peak_mb"] = float(
        psutil.Process().memory_info().rss / (1024 * 1024)
    )
    out["device"] = str(adapter.device)
    out["scheduler"] = type(adapter.scheduler).__name__
    out["batch_policy_inference"] = bool(adapter.batch_policy_inference)
    out["batched_path_active"] = bool(
        getattr(adapter, "_batched_inference_applies", lambda: False)()
    )

    return out


# ----------------------------------------------------------------------
# GPPO allocator scaling families
# ----------------------------------------------------------------------

def allocator_family(uavs: int, slots: int, repeats: int = 5):
    """Drive the real GPPO allocator with a controlled slot count.

    Family 1 fixes `slots` while UAV count varies; family 2 scales `slots`
    with UAV count.  The runtime state is real (real graph, real stale
    positions, real agent list); only the slot list is synthetic, so the
    measurement isolates graph/feature/inference cost.
    """

    from dataclasses import replace
    from integrations.gppo.event_allocator import GPPOEventSlot
    from integrations.gppo.task_graph import TaskType

    runtime, adapter = build(uavs, steps=6, gppo=True)
    allocator = adapter.scheduler.allocator

    # Real positions from the live graph: use the first N node coordinates
    # as slot anchors so the travel-time matrix is exercised.
    coords = [
        np.asarray(node.data.coords, dtype=float)
        for node in adapter._node_manager.nodes_dict.__iter__()
    ]
    if len(coords) < slots:
        step = max(1, len(coords) // max(1, slots))
        coords = (coords * step)[:max(slots, 1)]

    slots_list = []
    for i in range(slots):
        anchor = coords[i % len(coords)] if coords else np.zeros(2)
        slots_list.append(GPPOEventSlot(
            slot_id=100000 + i,
            task_type=TaskType.TARGET_SEARCH,
            priority=5.0,
            position=(float(anchor[0]), float(anchor[1])),
            processing_time=6.0,
            source_entity_id=i,
        ))

    # Count the sequential decisions by wrapping the adapter once.
    decisions = {"n": 0, "graphs": 0}
    original_assign = allocator.adapter.assign

    def counting_assign(graph, deterministic=True):
        decisions["graphs"] += 1
        result = original_assign(graph, deterministic=deterministic)
        if result is not None:
            decisions["n"] += 1
        return result

    allocator.adapter.assign = counting_assign

    timings = []
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            allocator.allocate(list(slots_list))
            timings.append(time.perf_counter() - start)
    finally:
        allocator.adapter.assign = original_assign

    graph = None
    try:
        base = adapter.scheduler.allocator
        from integrations.gppo.runtime_graph import RuntimeGraphBuilder

        graph = RuntimeGraphBuilder(runtime).build().graph
    except Exception:
        pass

    return {
        "uavs": uavs,
        "slots": slots,
        "allocate_seconds": min(timings) if timings else None,
        "sequential_graphs": decisions["graphs"],
        "sequential_decisions": decisions["n"],
        "n_uav_rows": int(graph.num_uavs) if graph is not None else None,
        "n_task_rows": slots,
        "flat_action_space": (
            int(graph.num_uavs * slots) if graph is not None else None
        ),
        "task_rows_plus_uav": slots,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uavs", type=int, nargs="+", default=[4, 8, 16, 30, 60])
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--gppo-only", action="store_true")
    parser.add_argument("--policy-device", type=str, default=None)
    parser.add_argument("--policy-batch", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / f"extended_gppo_scalability_baseline_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)

    payload = {"heuristic": [], "gppo_fixed": [], "gppo_scaled": []}

    if not args.gppo_only:
        for uavs in args.uavs:
            print(f"[pipeline] heuristic uavs={uavs} ...", flush=True)
            policy = None
            if args.policy_device:
                policy = {"device": args.policy_device,
                          "batch_inference": args.policy_batch}
            r = profile(uavs, args.steps, args.measure_steps, gppo=False,
                        policy=policy)
            payload["heuristic"].append(r)
            print(f"[pipeline] uavs={uavs} mean={r['mean_step_seconds']:.3f}s "
                  f"p95={r['p95_step_seconds']:.3f}s nodes={r['graph_nodes']}",
                  flush=True)

    # Family 1: fixed task count (1 slot) while UAV count varies.
    for uavs in args.uavs:
        print(f"[gppo] family1 uavs={uavs} slots=1 ...", flush=True)
        payload["gppo_fixed"].append(allocator_family(uavs, 1))
        print(f"[gppo] family1 uavs={uavs} "
              f"alloc={payload['gppo_fixed'][-1]['allocate_seconds']:.4f}s",
              flush=True)

    # Family 2: one slot per four UAVs, documented rule.
    for uavs in args.uavs:
        slots = max(1, uavs // 4)
        print(f"[gppo] family2 uavs={uavs} slots={slots} ...", flush=True)
        payload["gppo_scaled"].append(allocator_family(uavs, slots))
        print(f"[gppo] family2 uavs={uavs} slots={slots} "
              f"alloc={payload['gppo_scaled'][-1]['allocate_seconds']:.4f}s "
              f"decisions={payload['gppo_scaled'][-1]['sequential_decisions']}",
              flush=True)

    (out / "profile.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    def write_csv(name, rows, extra):
        if not rows:
            return
        keys = ["uavs"] + extra + [
            k for k in rows[0]
            if k not in ("uavs", "inclusive", "exclusive",
                         "calls_per_mission_step") and k not in extra
        ]
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for r in rows:
                w.writerow([r.get(k) for k in keys])

    write_csv("heuristic_scaling.csv", payload["heuristic"], [
        "mean_step_seconds", "p50_step_seconds", "p95_step_seconds",
        "graph_nodes", "graph_edges", "ram_peak_mb",
    ])
    write_csv("gppo_fixed_tasks.csv", payload["gppo_fixed"], [
        "slots", "allocate_seconds", "sequential_graphs",
        "sequential_decisions", "n_uav_rows", "flat_action_space",
    ])
    write_csv("gppo_scaled_tasks.csv", payload["gppo_scaled"], [
        "slots", "allocate_seconds", "sequential_graphs",
        "sequential_decisions", "n_uav_rows", "flat_action_space",
    ])

    print(f"[pipeline] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
