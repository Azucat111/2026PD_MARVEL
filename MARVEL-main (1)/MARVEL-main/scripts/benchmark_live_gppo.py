"""Live-GPPO activation-state benchmark.

CONTROLLED ACTIVATION-STATE BENCHMARK -- not a natural-from-reset deployment
run on the full 500 m sheet.

The extended 500 m scenario never reaches the frozen Search activation
threshold (`exploration_rate >= 0.30`) inside a feasible benchmark runtime,
so the end-to-end MARVEL + live Search + Relay + GPPO path had only ever
been measured with controlled synthetic slot lists.  This harness runs the
real path end to end by changing the *initial condition* (map extent) and
nothing else: the activation transition, the Search service timer, Search
capacity, Relay semantics, the GPPO policy and the MARVEL planner are all
the frozen ones.

Measurement only: the production tree is not modified.  Every counter and
timer is installed by monkeypatching in this file.

Modes
-----
``--mode trace``    per-mission-step lifecycle record (activation, initial
                    allocation, completion, refill, Relay) at each scale.
``--mode profile``  steady-state vs allocation-step timing split, plus the
                    MARVEL/GPPO breakdown and graph/action-space sizes.

Usage
-----
    python scripts/benchmark_live_gppo.py --mode trace --uavs 4 8 16 30 60
    python scripts/benchmark_live_gppo.py --mode profile --uavs 4 8 16 30 60
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from utils.scenario_config import load_and_validate_scenario  # noqa: E402
from utils.simulation_runtime import SimulationRuntime  # noqa: E402
from utils.policy_adapter import MARVELPolicyAdapter  # noqa: E402
from utils import agent as agent_module  # noqa: E402

SCENARIO = ROOT / "configs" / "scenarios" / "extended_live_gppo_full.yaml"
CHECKPOINT = (
    Path("/home/nick/MARVEL") / "artifacts" / "phase14_v9_frozen"
    / "gppo_phase14_v9_ckpt80.pth"
)
SEED = 12345

# The controlled initial condition is always seeded to just above the frozen
# threshold, so activation is reached by `maybe_activate` on the first
# high-level update rather than after a long (and, see the report, unstable)
# natural warm-up.
ACTIVATION_TARGET = 0.305
SWEEP_SPACING = 8.0
SWEEP_HEADINGS = (0.0, 90.0, 180.0, 270.0)


# ----------------------------------------------------------------------
# controlled activation state
# ----------------------------------------------------------------------

def install_activation_state(
    runtime,
    adapter,
    target: float = ACTIVATION_TARGET,
    margin: float = 0.005,
    spacing: float = SWEEP_SPACING,
    headings=SWEEP_HEADINGS,
):
    """Seed a controlled exploration state that has genuinely been observed.

    CONTROLLED ACTIVATION-STATE BENCHMARK.  The state below is not produced
    by flying the fleet; it is produced by running the runtime's *own*
    sensor model at real, collision-free lattice positions along a
    boustrophedon coverage path, from four headings, and unioning the
    ``visible_cells`` it returns.

    That matters for legitimacy: `explored_cells` is exactly the set the
    frozen sensing stack would have accumulated had the fleet flown that
    path, and the MARVEL belief is marked from the same cells, so the graph,
    the frontier distribution and every PolicyNet observation are built from
    a state the normal transition could have produced.  Nothing is faked at
    the scalar level -- `exploration_rate` is still
    ``len(explored_cells) / free_cell_count`` and Search still activates
    only through ``_maybe_activate_search``'s ``rate >= threshold`` test.

    Read the resulting numbers as a controlled activation-state benchmark,
    not as a natural-from-reset deployment run.
    """

    grid = runtime.obstacles.get_occupancy_grid()
    sensor = runtime.sensor_by_type.get("explorer", runtime.sensor)
    frame = runtime.obstacles.frame
    lower, upper = frame.bounds_min, frame.bounds_max

    needed = int(np.ceil((target + margin) * runtime.free_cell_count))

    cells: set[tuple[int, int]] = set()
    samples = 0
    headings = tuple(float(h) for h in headings)

    ys = np.arange(lower[1] + spacing * 0.5, upper[1], spacing)
    for row, y in enumerate(ys):
        xs = np.arange(lower[0] + spacing * 0.5, upper[0], spacing)
        if row % 2:
            xs = xs[::-1]
        for x in xs:
            point = np.array([float(x), float(y)])
            # Only sense from positions the fleet could actually occupy.
            if runtime.obstacles.check_collision(point, radius=0.5)[0]:
                continue
            for heading in headings:
                state = SimpleNamespace(
                    robot_id=0, position=point,
                    heading=heading, velocity=1.0,
                )
                seen = sensor.sense(state, grid, [point])
                cells.update(
                    (int(a), int(b)) for a, b in seen["visible_cells"]
                )
            samples += 1
            if len(cells) >= needed:
                break
        if len(cells) >= needed:
            break

    runtime.explored_cells = set(cells)
    adapter._update_belief(np.asarray(sorted(cells), dtype=float))

    return {
        "sweep_samples": samples,
        "sweep_spacing_m": float(spacing),
        "sweep_headings": list(headings),
        "explored_cells": len(cells),
        "free_cell_count": int(runtime.free_cell_count),
        "exploration_rate": float(runtime.exploration_rate),
        "target_rate": float(target),
    }


# ----------------------------------------------------------------------
# runtime construction
# ----------------------------------------------------------------------

def build(uavs: int, steps: int, policy=None, seed: int = SEED,
          activation_target: float | None = ACTIVATION_TARGET,
          scheduler_mode: str = "gppo"):
    """Build the controlled-activation runtime and advance `steps`."""

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
    config.setdefault("scenario", {})["random_seed"] = seed

    from integrations.gppo.checkpoint import load_frozen_gppo
    from integrations.gppo.config_profile import prepare_gppo_config

    config["task_scheduler"] = {
        "mode": scheduler_mode,
        "checkpoint": str(CHECKPOINT),
        "base_position": [0.0, 0.0],
        "search_scenario_seed": seed,
    }
    # Prepared for both modes so the only difference between the heuristic
    # and GPPO arms of the comparison is the scheduler itself.
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
# shared instrumentation
# ----------------------------------------------------------------------

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


def instrument(adapter, runtime, timer: Timer):
    """Wrap the live pipeline; every span is entered at most once per call."""

    restore = []

    def wrap(obj, name, span):
        # Spans that only exist on one scheduler are simply skipped, so the
        # heuristic arm can be profiled with the same harness.
        if not hasattr(obj, name):
            return
        original = getattr(obj, name)

        def wrapped(*args, **kwargs):
            with timer.span(span):
                return original(*args, **kwargs)

        setattr(obj, name, wrapped)
        restore.append((obj, name, original))

    # --- MARVEL adapter stages -------------------------------------------
    wrap(adapter, "_update_belief", "marvel.belief")
    wrap(adapter, "_build_map_info", "marvel.map_info")
    wrap(agent_module.Agent, "update_graph", "marvel.graph")
    wrap(agent_module.Agent, "update_planning_state", "marvel.planning_state")

    manager = adapter._node_manager
    wrap(manager, "_shared_graph", "marvel.planning_shared_graph")

    wrap(agent_module.Agent, "compute_best_heading", "marvel.heading_candidates")
    wrap(agent_module.Agent, "get_observation", "marvel.observation")
    wrap(agent_module.Agent, "select_next_waypoint", "marvel.policynet")
    wrap(agent_module.Agent, "decode_waypoint", "marvel.policynet_decode")
    wrap(adapter, "_batched_actions", "marvel.batched_actions")

    # The batched path calls the module directly, so the forward pass is
    # timed at the module boundary rather than through
    # `select_next_waypoint`.
    policy_net = adapter.policy_net
    original_forward = policy_net.forward

    def timed_forward(*args, **kwargs):
        with timer.span("marvel.policynet_forward"):
            return original_forward(*args, **kwargs)

    policy_net.forward = timed_forward
    restore.append((policy_net, "forward", original_forward))

    # --- GPPO scheduler ---------------------------------------------------
    wrap(adapter.scheduler, "apply", "gppo.scheduler_apply")
    wrap(adapter.scheduler, "_high_level_update", "gppo.high_level_update")
    wrap(adapter.scheduler, "_maybe_activate_search", "gppo.maybe_activate")
    wrap(adapter.scheduler, "_refresh_search", "gppo.refresh_search")
    wrap(adapter.scheduler, "_refresh_relay", "gppo.refresh_relay")

    allocator = getattr(adapter.scheduler, "allocator", None)
    if allocator is not None:
        wrap(allocator, "allocate", "gppo.allocate")
        for name in ("build_observed_graph", "observed_travel_time_matrix"):
            wrap(allocator, name, f"gppo.{name}")

    # --- runtime stages ---------------------------------------------------
    wrap(runtime, "_get_observations", "runtime.sensing")
    wrap(runtime, "_update_explored_cells", "runtime.sensing_explored")
    wrap(runtime.comm, "get_topology", "runtime.comm")
    wrap(runtime.tasks, "update", "runtime.tasks")
    wrap(runtime.obstacles, "check_collision", "runtime.collision")
    wrap(runtime.shield, "filter_actions", "runtime.shield")
    for _key, model in runtime.dynamics_by_type.items():
        wrap(model, "step", "runtime.motion")

    return restore


def graph_sizes(adapter):
    manager = adapter._node_manager
    nodes = int(manager.nodes_dict.__len__())
    edges = int(sum(
        len(node.data.neighbor_list) for node in manager.nodes_dict.__iter__()
    ))
    return nodes, edges


def action_space(adapter):
    """Flat GPPO action space at the current state, if it can be built."""

    try:
        from integrations.gppo.runtime_graph import RuntimeGraphBuilder
        graph = RuntimeGraphBuilder(adapter.runtime).build().graph
        return {
            "n_uav_rows": int(graph.num_uavs),
            "n_task_rows": int(graph.num_subtasks),
            "flat_action_space": int(graph.num_actions),
        }
    except Exception as exc:  # measurement-only probe
        return {"error": str(exc)}


# ----------------------------------------------------------------------
# mode: trace
# ----------------------------------------------------------------------

def _install_shield_counter():
    """Count SafetyShield blocks by reason, without touching production.

    `runtime.step` drains the shield's own event list, so the counts have to
    come from `_record` rather than from `pop_events`.
    """

    from utils.safety_shield import SafetyShield

    counts = defaultdict(int)
    original = SafetyShield._record

    def counting(self, robot_id, step, reason):
        counts[str(reason).split(":")[0]] += 1
        return original(self, robot_id, step, reason)

    SafetyShield._record = counting

    def restore():
        SafetyShield._record = original
        return dict(counts)

    return restore


def trace(uavs: int, steps: int, policy=None, seed: int = SEED):
    """Per-mission-step lifecycle record of the live GPPO path."""

    runtime, adapter, activation = build(
        uavs, steps=0, policy=policy, seed=seed
    )
    scheduler = adapter.scheduler

    interval = max(1, int(runtime._mission_step_interval()))
    observations = runtime._get_observations()

    # Allocation-call counter, so "an allocation step" is defined by the
    # real allocator running -- not by any inferred proxy.
    counts = {"allocate_calls": 0, "allocate_seconds": 0.0}
    original_allocate = scheduler.allocator.allocate

    def counting_allocate(slots, **kwargs):
        start = time.perf_counter()
        try:
            return original_allocate(slots, **kwargs)
        finally:
            counts["allocate_calls"] += 1
            counts["allocate_seconds"] += time.perf_counter() - start

    scheduler.allocator.allocate = counting_allocate

    pop_blocks = _install_shield_counter()
    parked_run = np.zeros(uavs, dtype=int)
    parked_max = np.zeros(uavs, dtype=int)

    rows = []
    try:
        for mission in range(1, steps + 1):
            before_serviced = set(scheduler.serviced_heat_ids)
            before_assign = dict(scheduler.search_assignments)
            before_relay = dict(scheduler.relay_assignments)
            before_calls = counts["allocate_calls"]

            positions_before = np.asarray(
                [r.position for r in runtime.robots], dtype=float
            )
            actions = adapter.get_actions(observations)
            for _tick in range(interval):
                observations, _info = runtime.step(actions)

            moved = np.linalg.norm(
                np.asarray([r.position for r in runtime.robots], dtype=float)
                - positions_before,
                axis=1,
            )
            parked_run = np.where(moved < 1e-9, parked_run + 1, 0)
            parked_max = np.maximum(parked_max, parked_run)

            allocated = counts["allocate_calls"] - before_calls
            newly_serviced = sorted(
                set(scheduler.serviced_heat_ids) - before_serviced
            )
            new_assign = {
                int(k): int(v) for k, v in scheduler.search_assignments.items()
                if int(k) not in before_assign
            }
            new_relay = {
                int(k): int(v[0]) for k, v in scheduler.relay_assignments.items()
                if int(k) not in before_relay
            }

            task_rows = sum(1 for r in runtime.robots if r.current_task == "target_search") \
                + sum(1 for r in runtime.robots if r.current_task == "relay")
            row = {
                "mission_step": mission,
                "exploration_rate": float(runtime.exploration_rate),
                "activated": bool(scheduler.search_activated),
                "activation_step": scheduler.search_activation_step,
                "search_assignments": dict(
                    (int(k), int(v))
                    for k, v in scheduler.search_assignments.items()
                ),
                "relay_assignments": dict(
                    (int(k), int(v[0]))
                    for k, v in scheduler.relay_assignments.items()
                ),
                "serviced": sorted(int(x) for x in scheduler.serviced_heat_ids),
                "newly_serviced": newly_serviced,
                "new_search": new_assign,
                "new_relay": new_relay,
                "allocate_calls": allocated,
                "allocate_seconds": counts["allocate_seconds"],
                "heat_points": len(scheduler.heat_points),
                "assigned_task_rows": task_rows,
                "parked_now": int(np.count_nonzero(moved < 1e-9)),
                "dwell": {
                    int(k): int(v)
                    for k, v in scheduler.search_dwell_by_heat.items()
                },
            }
            rows.append(row)
    finally:
        scheduler.allocator.allocate = original_allocate
        blocks = pop_blocks()

    return runtime, adapter, rows, activation, blocks, parked_max


# ----------------------------------------------------------------------
# mode: profile
# ----------------------------------------------------------------------

def profile(
    uavs: int,
    warmup: int,
    measure: int,
    policy=None,
    seed: int = SEED,
    scheduler_mode: str = "gppo",
    activation_target=ACTIVATION_TARGET,
):
    """Steady-state vs allocation-step timing split."""

    runtime, adapter, activation = build(
        uavs, steps=warmup, policy=policy, seed=seed,
        scheduler_mode=scheduler_mode, activation_target=activation_target,
    )
    scheduler = adapter.scheduler

    interval = max(1, int(runtime._mission_step_interval()))
    observations = runtime._get_observations()

    timer = Timer()
    restore = instrument(adapter, runtime, timer)

    # Tag every measured step with whether the allocator actually ran.
    allocation_steps = {"n": 0}
    allocator = getattr(scheduler, "allocator", None)
    original_allocate = allocator.allocate if allocator is not None else None

    def tagging_allocate(slots, **kwargs):
        allocation_steps["n"] += 1
        return original_allocate(slots, **kwargs)

    if allocator is not None:
        allocator.allocate = tagging_allocate

    steady_times, alloc_times = [], []
    step_rows = []
    parked_run = np.zeros(uavs, dtype=int)
    parked_max = np.zeros(uavs, dtype=int)
    try:
        for index in range(measure):
            before_calls = allocation_steps["n"]
            before = {k: v for k, v in timer.inclusive.items()}

            positions_before = np.asarray(
                [r.position for r in runtime.robots], dtype=float
            )
            start = time.perf_counter()
            with timer.span("pipeline.total"):
                actions = adapter.get_actions(observations)
                for _tick in range(interval):
                    observations, _info = runtime.step(actions)
            elapsed = time.perf_counter() - start

            moved = np.linalg.norm(
                np.asarray([r.position for r in runtime.robots], dtype=float)
                - positions_before,
                axis=1,
            )
            parked_run = np.where(moved < 1e-9, parked_run + 1, 0)
            parked_max = np.maximum(parked_max, parked_run)

            allocated = allocation_steps["n"] - before_calls
            kind = "allocation" if allocated else "steady"
            (alloc_times if allocated else steady_times).append(elapsed)

            step_rows.append({
                "index": index,
                "mission_step": warmup + index + 1,
                "kind": kind,
                "seconds": elapsed,
                "exploration_rate": float(runtime.exploration_rate),
                "active_search": len(getattr(scheduler, "search_assignments", {})),
                "active_relay": len(getattr(scheduler, "relay_assignments", {})),
                "serviced": len(getattr(scheduler, "serviced_heat_ids", set())),
                "allocate_calls": allocated,
                "parked_now": int(np.count_nonzero(moved < 1e-9)),
                "delta": {
                    k: timer.inclusive[k] - before.get(k, 0.0)
                    for k in timer.inclusive
                },
            })
    finally:
        for obj, name, original in restore:
            setattr(obj, name, original)
        if allocator is not None:
            allocator.allocate = original_allocate

    n = max(1, len(step_rows))
    nodes, edges = graph_sizes(adapter)

    import psutil
    out = {
        "uavs": uavs,
        "mode": "profile",
        "warmup_steps": warmup,
        "steps_measured": len(step_rows),
        "interval": interval,
        "activation": activation,
        "activated": bool(getattr(scheduler, "search_activated", False)),
        "activation_step": getattr(scheduler, "search_activation_step", None),
        "exploration_rate": float(runtime.exploration_rate),
        "parked_max_mission_steps": int(parked_max.max()) if uavs else 0,
        "parked_uavs_half_window": int(np.count_nonzero(
            parked_max >= max(1, len(step_rows)) // 2
        )),
        "steady_steps": len(steady_times),
        "allocation_steps": len(alloc_times),
        "steady_mean_seconds": (
            statistics.fmean(steady_times) if steady_times else None
        ),
        "steady_p95_seconds": (
            sorted(steady_times)[min(len(steady_times) - 1,
                                     int(0.95 * len(steady_times)))]
            if steady_times else None
        ),
        "allocation_mean_seconds": (
            statistics.fmean(alloc_times) if alloc_times else None
        ),
        "allocation_max_seconds": max(alloc_times) if alloc_times else None,
        "allocation_p95_seconds": (
            sorted(alloc_times)[min(len(alloc_times) - 1,
                                    int(0.95 * len(alloc_times)))]
            if alloc_times else None
        ),
        "inclusive": {k: v / n for k, v in timer.inclusive.items()},
        "exclusive": {k: v / n for k, v in timer.exclusive.items()},
        "calls_per_step": {k: v / n for k, v in timer.calls.items()},
        "graph_nodes": nodes,
        "graph_edges": edges,
        "ram_mb": float(psutil.Process().memory_info().rss / (1024 * 1024)),
        "device": str(adapter.device),
        "scheduler": type(adapter.scheduler).__name__,
        "batch_policy_inference": bool(adapter.batch_policy_inference),
        "search_capacity": getattr(scheduler, "max_search_uavs", None),
        "relay_capacity": getattr(scheduler, "max_relay_uavs", None),
        "activation_threshold": getattr(
            scheduler, "search_activation_threshold", None
        ),
        "search_reach_radius": getattr(scheduler, "search_reach_radius", None),
        "search_service_steps": getattr(scheduler, "search_service_steps", None),
        "heat_points": len(getattr(scheduler, "heat_points", [])),
        "action_space": action_space(adapter),
        "step_rows": step_rows,
    }

    if torch.cuda.is_available() and adapter.device.type == "cuda":
        out["vram_peak_allocated_mb"] = float(
            torch.cuda.max_memory_allocated() / (1024 * 1024)
        )
        out["vram_peak_reserved_mb"] = float(
            torch.cuda.max_memory_reserved() / (1024 * 1024)
        )

    return out


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------

def summarise_trace(rows, uavs, activation, blocks, parked_max, seconds):
    first_activation = next(
        (r["mission_step"] for r in rows if r["activated"]), None
    )
    first_alloc = next(
        (r["mission_step"] for r in rows if r["new_search"]), None
    )
    first_service = next(
        (r["mission_step"] for r in rows if r["newly_serviced"]), None
    )
    # A refill is a new Search assignment on a step *after* a completion.
    refill = None
    if first_service is not None:
        refill = next(
            (r["mission_step"] for r in rows
             if r["mission_step"] > first_service and r["new_search"]),
            None,
        )
    first_relay = next(
        (r["mission_step"] for r in rows if r["new_relay"]), None
    )
    return {
        "uavs": uavs,
        "mission_steps": len(rows),
        "seconds": seconds,
        "seconds_per_step": seconds / max(1, len(rows)),
        "seed_rate": activation["exploration_rate"] if activation else None,
        "activated": first_activation is not None,
        "activation_step": first_activation,
        "activation_rate": (
            next(r["exploration_rate"] for r in rows if r["activated"])
            if first_activation is not None else None
        ),
        "first_search_allocation_step": first_alloc,
        "first_completion_step": first_service,
        "first_refill_step": refill,
        "first_relay_step": first_relay,
        "final_rate": rows[-1]["exploration_rate"] if rows else None,
        "final_search_uavs": len(rows[-1]["search_assignments"]) if rows else 0,
        "final_relay_uavs": len(rows[-1]["relay_assignments"]) if rows else 0,
        "serviced_total": len(rows[-1]["serviced"]) if rows else 0,
        "allocation_steps": sum(1 for r in rows if r["allocate_calls"]),
        "heat_points": rows[-1]["heat_points"] if rows else 0,
        "parked_uavs_half_window": int(np.count_nonzero(
            parked_max >= max(1, len(rows)) // 2
        )),
        "parked_max_mission_steps": int(parked_max.max()) if uavs else 0,
        "shield_blocks": blocks,
        "shield_blocks_total": int(sum(blocks.values())),
    }


def write_csv(path: Path, rows, keys):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for row in rows:
            writer.writerow([row.get(k) for k in keys])


def scenario_manifest(results, policy):
    """Everything needed to reproduce the measured scenario."""

    config = load_and_validate_scenario(SCENARIO)
    first = results[0] if results else {}
    return {
        "scenario_path": str(SCENARIO.relative_to(ROOT)),
        "scenario_name": config.get("scenario", {}).get("name"),
        "geometry_mode": config.get("environment", {}).get("geometry_mode"),
        "map_file": config.get("environment", {}).get("map_file"),
        "map_size": config.get("environment", {}).get("map_size"),
        "map_resolution": config.get("environment", {}).get("map_resolution"),
        "seed": SEED,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_exists": CHECKPOINT.exists(),
        "policy": policy or {"device": "cpu", "batch_inference": False},
        "uav_counts": [r["uavs"] for r in results],
        "activation_threshold": first.get("activation_threshold"),
        "search_capacity_max_search_uavs": first.get("search_capacity"),
        "relay_capacity_max_relay_uavs": first.get("relay_capacity"),
        "search_service_steps": first.get("search_service_steps"),
        "search_reach_radius": (
            first.get("search_reach_radius")
        ),
        "heat_points": first.get("heat_points"),
        "frozen_threshold_note": (
            "extended mode divides explored_cells by obstacles.cell_count "
            "(all map cells), not by the free-cell count, so "
            "exploration_rate >= 0.30 corresponds to covering ~81% of the "
            "free area of maps_test/1.png."
        ),
        "controlled_activation_state": True,
        "measurement_only": (
            "No production module is modified. All counters and timers are "
            "installed by monkeypatching inside scripts/benchmark_live_gppo.py."
        ),
    }


def snapshot_manifest(results):
    """The controlled activation state used for every measured run."""

    return {
        "kind": "controlled-activation-state",
        "label": (
            "CONTROLLED ACTIVATION-STATE BENCHMARK -- not a "
            "natural-from-reset deployment run"
        ),
        "method": (
            "The runtime's own sensor model is evaluated at real, "
            "collision-free lattice positions along a boustrophedon coverage "
            "path, from four headings, and the returned visible_cells are "
            "unioned. runtime.explored_cells is set to that set and the "
            "MARVEL belief is marked from the same cells, so the graph, the "
            "frontier distribution and every PolicyNet observation are built "
            "from a state the normal transition could have produced."
        ),
        "not_done": [
            "exploration_rate is not assigned a scalar",
            "activation_threshold is unchanged (frozen checkpoint value)",
            "search_service_steps, Search capacity, Relay semantics and the "
            "GPPO policy are unchanged",
            "no synthetic GPPO slot list is injected: every Search row is "
            "produced by the real scheduler from a real heat point",
        ],
        "sweep_spacing_m": SWEEP_SPACING,
        "sweep_headings": list(SWEEP_HEADINGS),
        "target_rate": ACTIVATION_TARGET,
        "per_scale": [
            {
                "uavs": r["uavs"],
                "seed": r.get("activation"),
                "activation_reached_at_mission_step": r.get("activation_step"),
                "first_search_allocation_mission_step": next(
                    (row["mission_step"] for row in r.get("step_rows", [])
                     if row["kind"] == "allocation"),
                    None,
                ),
                "exploration_rate_at_measure": r.get("exploration_rate"),
            }
            for r in results
        ],
    }


def write_report(out: Path, results, mode_note: str):
    """Human-readable summary of the measured run."""

    lines = [
        "# Extended live-GPPO activation-state benchmark",
        "",
        "**CONTROLLED ACTIVATION-STATE BENCHMARK — not a natural-from-reset",
        "deployment run.** See `snapshot_manifest.json` for exactly how the",
        "initial exploration state is produced and what is deliberately left",
        "unchanged.",
        "",
        "Measurement only: no production module is modified. Every counter and",
        "timer is installed by monkeypatching inside",
        "`scripts/benchmark_live_gppo.py`.",
        "",
        mode_note,
        "",
        "## Scaling",
        "",
        "| UAVs | activated | steady mean s | steady p95 s | alloc mean s | "
        "alloc n | nodes | edges | parked (half window) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        lines.append(
            f"| {r['uavs']} | {r['activated']} | "
            f"{_fmt(r.get('steady_mean_seconds'))} | "
            f"{_fmt(r.get('steady_p95_seconds'))} | "
            f"{_fmt(r.get('allocation_mean_seconds'))} | "
            f"{r.get('allocation_steps')} | {r.get('graph_nodes')} | "
            f"{r.get('graph_edges')} | "
            f"{r.get('parked_uavs_half_window')}/{r['uavs']} |"
        )

    lines += [
        "",
        "## Per-step breakdown (mean, milliseconds)",
        "",
    ]
    spans = sorted({k for r in results for k in r["inclusive"]})
    header = "| span | " + " | ".join(str(r["uavs"]) for r in results) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(results))
    for span in spans:
        cells = []
        for r in results:
            value = r["inclusive"].get(span)
            cells.append("" if value is None else f"{value * 1000:.2f}")
        lines.append(f"| {span} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Action space",
        "",
        "| UAVs | uav rows | task rows | flat action space |",
        "|---:|---:|---:|---:|",
    ]
    for r in results:
        a = r.get("action_space", {})
        lines.append(
            f"| {r['uavs']} | {a.get('n_uav_rows')} | {a.get('n_task_rows')} "
            f"| {a.get('flat_action_space')} |"
        )

    lines += [
        "",
        "## Resource use",
        "",
        "| UAVs | RAM MB | VRAM alloc MB | VRAM reserved MB |",
        "|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['uavs']} | {r.get('ram_mb', float('nan')):.0f} | "
            f"{_fmt(r.get('vram_peak_allocated_mb'))} | "
            f"{_fmt(r.get('vram_peak_reserved_mb'))} |"
        )

    lines.append("")
    (out / "LIVE_GPPO_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _fmt(value, digits=4):
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def system_info():
    import psutil
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_mb": float(psutil.virtual_memory().total / (1024 * 1024)),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("trace", "profile"), required=True)
    parser.add_argument("--uavs", type=int, nargs="+",
                        default=[4, 8, 16, 30, 60])
    parser.add_argument("--steps", type=int, default=60,
                        help="trace mode: mission steps to record")
    parser.add_argument("--warmup", type=int, default=0,
                        help="profile mode: warmup mission steps")
    parser.add_argument("--measure", type=int, default=30,
                        help="profile mode: measured mission steps")
    parser.add_argument("--policy-device", type=str, default=None)
    parser.add_argument("--policy-batch", action="store_true")
    parser.add_argument(
        "--no-activation-seed", action="store_true",
        help="profile mode only: skip the controlled activation state and "
             "start from the scenario's own reset, to quantify how much of "
             "the step cost is state-dependent.",
    )
    parser.add_argument(
        "--scheduler", choices=("gppo", "heuristic"), default="gppo",
        help="profile mode only: run the same controlled activation state "
             "under the heuristic scheduler instead, to isolate the GPPO "
             "layer's contribution.",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / f"extended_live_gppo_{args.mode}_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)

    policy = None
    if args.policy_device:
        policy = {"device": args.policy_device,
                  "batch_inference": args.policy_batch}

    (out / "system_info.txt").write_text(
        json.dumps(system_info(), indent=2), encoding="utf-8"
    )

    if args.mode == "trace":
        summaries = []
        for uavs in args.uavs:
            print(f"[trace] uavs={uavs} steps={args.steps} ...", flush=True)
            started = time.perf_counter()
            runtime, adapter, rows, activation, blocks, parked_max = trace(
                uavs, args.steps, policy=policy
            )
            summary = summarise_trace(
                rows, uavs, activation, blocks, parked_max,
                time.perf_counter() - started,
            )
            summaries.append(summary)
            for row in rows:
                print(
                    f"  m{row['mission_step']:>3} rate={row['exploration_rate']:.4f} "
                    f"act={int(row['activated'])} "
                    f"search={row['search_assignments']} "
                    f"relay={row['relay_assignments']} "
                    f"serviced={row['serviced']} "
                    f"alloc={row['allocate_calls']} "
                    f"parked={row['parked_now']} "
                    f"rows={row['assigned_task_rows']}",
                    flush=True,
                )
            print(
                f"[trace] uavs={uavs} activated={summary['activated']} "
                f"at={summary['activation_step']} "
                f"alloc={summary['first_search_allocation_step']} "
                f"complete={summary['first_completion_step']} "
                f"refill={summary['first_refill_step']} "
                f"relay={summary['first_relay_step']} "
                f"parked_half={summary['parked_uavs_half_window']}/{uavs} "
                f"blocks={summary['shield_blocks']}",
                flush=True,
            )

        (out / "trace.json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )
        write_csv(
            out / "trace_summary.csv",
            summaries,
            ["uavs", "activated", "activation_step", "activation_rate",
             "seed_rate", "first_search_allocation_step",
             "first_completion_step", "first_refill_step", "first_relay_step",
             "final_rate", "final_search_uavs", "final_relay_uavs",
             "serviced_total", "allocation_steps", "heat_points",
             "parked_uavs_half_window", "parked_max_mission_steps",
             "shield_blocks_total", "seconds", "seconds_per_step"],
        )
        print(f"[trace] wrote {out}")
        return 0

    results = []
    for uavs in args.uavs:
        print(
            f"[profile] uavs={uavs} warmup={args.warmup} "
            f"measure={args.measure} ...",
            flush=True,
        )
        result = profile(
            uavs, args.warmup, args.measure, policy=policy,
            scheduler_mode=args.scheduler,
            activation_target=(
                None if args.no_activation_seed else ACTIVATION_TARGET
            ),
        )
        results.append(result)
        print(
            f"[profile] uavs={uavs} activated={result['activated']}@{result['activation_step']} "
            f"steady={result['steady_mean_seconds']} "
            f"alloc_mean={result['allocation_mean_seconds']} "
            f"n_steady={result['steady_steps']} n_alloc={result['allocation_steps']}",
            flush=True,
        )

    (out / "profile.json").write_text(
        json.dumps(
            [{k: v for k, v in r.items() if k != "step_rows"}
             for r in results],
            indent=2,
        ),
        encoding="utf-8",
    )

    write_csv(
        out / "scaling.csv",
        results,
        ["uavs", "activated", "activation_step", "exploration_rate",
         "steady_steps", "allocation_steps", "steady_mean_seconds",
         "steady_p95_seconds", "allocation_mean_seconds",
         "allocation_p95_seconds", "allocation_max_seconds",
         "graph_nodes", "graph_edges", "ram_mb",
         "vram_peak_allocated_mb", "vram_peak_reserved_mb",
         "search_capacity", "relay_capacity", "heat_points",
         "parked_uavs_half_window", "parked_max_mission_steps"],
    )

    # Per-step rows, tagged steady vs allocation.
    step_keys = [
        "uavs", "index", "mission_step", "kind", "seconds",
        "exploration_rate", "active_search", "active_relay", "serviced",
        "allocate_calls",
    ]
    with open(out / "steps.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(step_keys)
        for result in results:
            for row in result["step_rows"]:
                writer.writerow([result["uavs"]] + [row.get(k) for k in step_keys[1:]])

    # Named-span breakdown per UAV count.
    spans = sorted({k for r in results for k in r["inclusive"]})
    with open(out / "spans.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uavs", "span", "inclusive_s", "exclusive_s",
                         "calls_per_step"])
        for result in results:
            for span in spans:
                writer.writerow([
                    result["uavs"], span,
                    result["inclusive"].get(span),
                    result["exclusive"].get(span),
                    result["calls_per_step"].get(span),
                ])

    write_csv(
        out / "action_space.csv",
        [
            {
                "uavs": r["uavs"],
                **{k: r["action_space"].get(k) for k in
                   ("n_uav_rows", "n_task_rows", "flat_action_space")},
            }
            for r in results
        ],
        ["uavs", "n_uav_rows", "n_task_rows", "flat_action_space"],
    )

    # Steady-state-only summary (allocation steps are reported separately,
    # never averaged into this table).
    write_csv(
        out / "steady_state.csv",
        results,
        ["uavs", "steady_steps", "steady_mean_seconds", "steady_p95_seconds",
         "exploration_rate", "graph_nodes", "graph_edges", "ram_mb",
         "vram_peak_allocated_mb", "vram_peak_reserved_mb",
         "parked_uavs_half_window"],
    )

    # The allocation steps themselves, one row each.
    with open(out / "allocation_steps.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "uavs", "mission_step", "seconds", "active_search",
            "active_relay", "serviced", "allocate_calls",
            "parked_now", "exploration_rate",
        ])
        for result in results:
            for row in result["step_rows"]:
                if row["kind"] != "allocation":
                    continue
                writer.writerow([
                    result["uavs"], row["mission_step"],
                    f"{row['seconds']:.6f}", row["active_search"],
                    row["active_relay"], row["serviced"],
                    row["allocate_calls"], row["parked_now"],
                    f"{row['exploration_rate']:.6f}",
                ])

    # GPPO-only spans, so the high-level layer can be read on its own.
    with open(out / "gppo_profile.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uavs", "span", "inclusive_s", "exclusive_s",
                         "calls_per_step"])
        for result in results:
            for span in spans:
                if not span.startswith("gppo."):
                    continue
                writer.writerow([
                    result["uavs"], span,
                    result["inclusive"].get(span),
                    result["exclusive"].get(span),
                    result["calls_per_step"].get(span),
                ])

    (out / "scenario_manifest.json").write_text(
        json.dumps(scenario_manifest(results, policy),
                   indent=2, default=str),
        encoding="utf-8",
    )
    (out / "snapshot_manifest.json").write_text(
        json.dumps(snapshot_manifest(results), indent=2, default=str),
        encoding="utf-8",
    )

    write_report(
        out, results,
        "Profiled with `--mode profile`: mission steps are classified by "
        "whether the real GPPO allocator was invoked during the step, and "
        "steady-state and allocation steps are reported separately rather "
        "than averaged together.",
    )

    print(f"[profile] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
