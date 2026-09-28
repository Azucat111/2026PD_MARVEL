"""Extended-mode scalability benchmark.

Baseline measurement only: this script instruments the existing extended
pipeline by monkeypatching.  It does not modify any production module.

Usage:
    python scripts/benchmark_extended_scalability.py \
        --uavs 4 8 16 30 60 --steps 50 --budget 300
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
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

SCENARIO = ROOT / "configs" / "scenarios" / "baseline_maps_test.yaml"

# Timing buckets reported in timing_breakdown.csv.
PHASES = (
    "policy.belief",
    "policy.map_info",
    "policy.graph",
    "policy.planning_state",
    "policy.observation",
    "policy.policynet",
    "policy.post_selection",
    "policy.other",
    "scheduler.apply",
    "runtime.step_total",
    "runtime.motion",
    "runtime.collision",
    "runtime.shield",
    "runtime.sensing",
    "runtime.comm",
    "runtime.tasks",
    "runtime.other",
)


class PhaseTimer:
    """Accumulates wall time per phase; re-entrant calls are nested away."""

    def __init__(self):
        self.totals = defaultdict(float)
        self._depth = 0

    def add(self, phase: str, seconds: float) -> None:
        if self._depth == 0:
            self.totals[phase] += seconds


class Stopwatch:
    def __init__(self, timer: PhaseTimer, phase: str):
        self.timer = timer
        self.phase = phase

    def __enter__(self):
        self._start = time.perf_counter()
        self.timer._depth += 1
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self._start
        self.timer._depth -= 1
        if self.timer._depth == 0:
            self.timer.totals[self.phase] += elapsed
        return False


FROZEN_GPPO_CKPT = (
    Path("/home/nick/MARVEL") / "artifacts" / "phase14_v9_frozen"
    / "gppo_phase14_v9_ckpt80.pth"
)


def build_config(uavs: int, scheduler: str = "heuristic"):
    config = load_and_validate_scenario(SCENARIO)
    config["robots"] = [{
        "id_range": [0, uavs - 1],
        "type": "explorer",
        "team_id": 1,
        "config": {
            "fov": 120,
            "sensor_range": 10.0,
            "velocity": 1.0,
            "yaw_rate": 35,
            "initial_positions": "random_safe",
        },
    }]
    config.setdefault("scenario", {})["random_seed"] = 12345

    if scheduler == "gppo":
        # Extended geometry on the frozen GPPO protocol; exercise the task
        # graph and the allocation stages that the heuristic scheduler skips.
        from integrations.gppo.checkpoint import load_frozen_gppo
        from integrations.gppo.config_profile import prepare_gppo_config

        config["task_scheduler"] = {
            "mode": "gppo",
            "checkpoint": str(FROZEN_GPPO_CKPT),
            "base_position": [0.0, 0.0],
            "search_scenario_seed": 12345,
        }
        _, checkpoint = load_frozen_gppo(str(FROZEN_GPPO_CKPT), device="cpu")
        config = prepare_gppo_config(config, checkpoint)

    return config


def instrument(adapter, runtime, timer: PhaseTimer):
    """Wrap the pipeline stages with phase timers."""

    # Adapter stages inside `_policy_actions`.
    adapter_belief = adapter._update_belief
    adapter_map_info = adapter._build_map_info

    def timed_belief(cells):
        with Stopwatch(timer, "policy.belief"):
            return adapter_belief(cells)

    def timed_map_info():
        with Stopwatch(timer, "policy.map_info"):
            return adapter_map_info()

    adapter._update_belief = timed_belief
    adapter._build_map_info = timed_map_info

    # Per-agent stages.  The shared Agent class is patched once; instances
    # created before this point still resolve through the class.
    original_update_graph = agent_module.Agent.update_graph
    original_planning = agent_module.Agent.update_planning_state
    original_observe = agent_module.Agent.get_observation
    original_select = agent_module.Agent.select_next_waypoint

    def timed_update_graph(self, *args, **kwargs):
        with Stopwatch(timer, "policy.graph"):
            return original_update_graph(self, *args, **kwargs)

    def timed_planning(self, *args, **kwargs):
        with Stopwatch(timer, "policy.planning_state"):
            return original_planning(self, *args, **kwargs)

    def timed_observe(self, *args, **kwargs):
        with Stopwatch(timer, "policy.observation"):
            return original_observe(self, *args, **kwargs)

    def timed_select(self, *args, **kwargs):
        with Stopwatch(timer, "policy.policynet"):
            return original_select(self, *args, **kwargs)

    agent_module.Agent.update_graph = timed_update_graph
    agent_module.Agent.update_planning_state = timed_planning
    agent_module.Agent.get_observation = timed_observe
    agent_module.Agent.select_next_waypoint = timed_select

    # Scheduler.
    original_apply = adapter.scheduler.apply

    def timed_apply(actions):
        with Stopwatch(timer, "scheduler.apply"):
            return original_apply(actions)

    adapter.scheduler.apply = timed_apply

    # Runtime stages.
    original_step = runtime.step
    original_observations = runtime._get_observations
    original_explored = runtime._update_explored_cells
    original_topology = runtime.comm.get_topology
    original_tasks = runtime.tasks.update
    original_check_collision = runtime.obstacles.check_collision
    original_shield = runtime.shield.filter_actions
    original_dyn = {
        key: model.step for key, model in runtime.dynamics_by_type.items()
    }

    def timed_observations():
        with Stopwatch(timer, "runtime.sensing"):
            return original_observations()

    def timed_explored(observations):
        with Stopwatch(timer, "runtime.sensing"):
            return original_explored(observations)

    def timed_topology(positions):
        with Stopwatch(timer, "runtime.comm"):
            return original_topology(positions)

    def timed_tasks(*args, **kwargs):
        with Stopwatch(timer, "runtime.tasks"):
            return original_tasks(*args, **kwargs)

    def timed_check_collision(*args, **kwargs):
        with Stopwatch(timer, "runtime.collision"):
            return original_check_collision(*args, **kwargs)

    def timed_shield(*args, **kwargs):
        with Stopwatch(timer, "runtime.shield"):
            return original_shield(*args, **kwargs)

    runtime.obstacles.check_collision = timed_check_collision
    runtime.shield.filter_actions = timed_shield

    runtime._get_observations = timed_observations
    runtime._update_explored_cells = timed_explored
    runtime.comm.get_topology = timed_topology
    runtime.tasks.update = timed_tasks

    def timed_dynamics(key):
        model = runtime.dynamics_by_type[key]
        original = model.step

        def step(*args, **kwargs):
            with Stopwatch(timer, "runtime.motion"):
                return original(*args, **kwargs)

        model.step = step

    for key in list(runtime.dynamics_by_type):
        timed_dynamics(key)

    return {
        "step": original_step,
        "restore": lambda: _restore(
            runtime, agent_module, original_update_graph, original_planning,
            original_observe, original_select, original_apply,
            original_observations, original_explored, original_topology,
            original_tasks,
        ),
    }


def _restore(runtime, agent_module, ug, pl, ob, se, ap, obs, ex, top, tk):
    agent_module.Agent.update_graph = ug
    agent_module.Agent.update_planning_state = pl
    agent_module.Agent.get_observation = ob
    agent_module.Agent.select_next_waypoint = se
    runtime._get_observations = obs
    runtime._update_explored_cells = ex
    runtime.comm.get_topology = top
    runtime.tasks.update = tk


def graph_metrics(adapter):
    manager = getattr(adapter, "_node_manager", None)
    if manager is None:
        return 0, 0

    nodes = 0
    edges = 0

    try:
        for node in manager.nodes_dict.__iter__():
            nodes += 1
            edges += len(getattr(node.data, "neighbor_list", []))
    except Exception:
        return nodes, edges

    return nodes, edges


def run_one(uavs: int, steps: int, budget: float, scheduler: str = "heuristic"):
    import psutil

    process = psutil.Process()

    result = {
        "uavs": uavs,
        "steps_requested": steps,
        "steps_completed": 0,
        "scheduler": scheduler,
        "status": "ok",
        "error": None,
    }

    config = build_config(uavs, scheduler=scheduler)

    np.random.seed(12345)
    reset_start = time.perf_counter()
    runtime = SimulationRuntime(config)
    observations = runtime.reset()
    adapter = MARVELPolicyAdapter(runtime)
    adapter.setup()
    reset_seconds = time.perf_counter() - reset_start

    result["reset_seconds"] = reset_seconds
    result["using_policy"] = bool(adapter._using_policy)
    result["geometry_mode"] = str(runtime.geometry_mode)
    result["device"] = str(adapter.device)

    timer = PhaseTimer()
    handle = instrument(adapter, runtime, timer)
    step_fn = handle["step"]

    interval = max(1, int(runtime._mission_step_interval()))

    step_times = []
    budget_exceeded = False
    started = time.perf_counter()

    try:
        for _ in range(steps):
            mission_start = time.perf_counter()

            actions = adapter.get_actions(observations)

            for _tick in range(interval):
                tick_start = time.perf_counter()
                observations, _info = step_fn(actions)
                timer.totals["runtime.step_total"] += (
                    time.perf_counter() - tick_start
                )

            step_times.append(time.perf_counter() - mission_start)

            if time.perf_counter() - started > budget:
                budget_exceeded = True
                break
    except Exception as exc:  # noqa: BLE001
        result["status"] = "exception"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        handle["restore"]()

    result["steps_completed"] = len(step_times)
    if budget_exceeded and len(step_times) < steps:
        result["status"] = "truncated_budget"

    if step_times:
        ordered = sorted(step_times)
        result["mean_step_seconds"] = float(statistics.fmean(step_times))
        result["p50_step_seconds"] = float(ordered[len(ordered) // 2])
        result["p95_step_seconds"] = float(
            ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        )
        result["max_step_seconds"] = float(max(step_times))
    else:
        result["mean_step_seconds"] = None
        result["p50_step_seconds"] = None
        result["p95_step_seconds"] = None
        result["max_step_seconds"] = None

    nodes, edges = graph_metrics(adapter)
    result["graph_nodes"] = int(nodes)
    result["graph_edges"] = int(edges)
    result["node_padding_size"] = int(
        getattr(agent_module, "NODE_PADDING_SIZE", 0)
    )

    import parameter as marvel_parameter

    result["angles_bins"] = int(marvel_parameter.NUM_ANGLES_BIN)
    result["flat_action_space"] = int(
        result["node_padding_size"] * result["angles_bins"]
    )
    result["task_rows"] = int(len(getattr(runtime.tasks, "tasks", []) or []))

    result["ram_peak_mb"] = float(
        process.memory_info().rss / (1024 * 1024)
    )
    result["cuda_used"] = bool(runtime and torch.cuda.is_available() and str(adapter.device).startswith("cuda"))
    if result["cuda_used"]:
        result["vram_peak_mb"] = float(
            torch.cuda.max_memory_allocated() / (1024 * 1024)
        )
    else:
        result["vram_peak_mb"] = None

    result["exploration_rate"] = float(getattr(runtime, "exploration_rate", 0.0))
    result["valid"] = bool(
        np.isfinite(result["exploration_rate"])
        and 0.0 <= result["exploration_rate"] <= 1.0
    )

    per_step = {
        phase: (
            timer.totals.get(phase, 0.0) / max(1, len(step_times))
        )
        for phase in PHASES
    }
    per_tick_phases = (
        "runtime.motion",
        "runtime.collision",
        "runtime.shield",
        "runtime.sensing",
        "runtime.comm",
        "runtime.tasks",
    )
    ticks = max(1, len(step_times) * interval)
    per_step["runtime.step_unattributed"] = float(
        max(
            0.0,
            timer.totals.get("runtime.step_total", 0.0) / max(1, len(step_times))
            - sum(timer.totals.get(p, 0.0) for p in per_tick_phases) / max(1, len(step_times)),
        )
    )
    result["phase_seconds_per_step"] = per_step
    result["unaccounted_seconds_per_step"] = float(
        max(
            0.0,
            (result["mean_step_seconds"] or 0.0)
            - timer.totals.get("runtime.step_total", 0.0) / max(1, len(step_times))
            - timer.totals.get("scheduler.apply", 0.0) / max(1, len(step_times))
            - sum(
                timer.totals.get(p, 0.0) / max(1, len(step_times))
                for p in PHASES
                if p != "scheduler.apply"
            ),
        )
    )

    return result


def system_info() -> dict:
    import psutil

    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_mb": float(psutil.virtual_memory().total / (1024 * 1024)),
    }
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True,
        ).stdout.strip()
        info["git_branch"] = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        pass
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uavs", type=int, nargs="+", default=[4, 8, 16, 30, 60])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--budget", type=float, default=300.0,
                        help="wall-clock seconds per UAV count before truncating")
    parser.add_argument("--scheduler", choices=("heuristic", "gppo"),
                        default="heuristic")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / f"extended_scalability_baseline_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for uavs in args.uavs:
        print(f"[benchmark] uavs={uavs} ...", flush=True)
        try:
            result = run_one(uavs, args.steps, args.budget,
                             scheduler=args.scheduler)
        except Exception as exc:  # noqa: BLE001
            result = {
                "uavs": uavs, "steps_completed": 0, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print(
            f"[benchmark] uavs={uavs} status={result.get('status')} "
            f"steps={result.get('steps_completed')} "
            f"mean={result.get('mean_step_seconds')}",
            flush=True,
        )

    info = system_info()
    (out / "system_info.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in info.items()) + "\n",
        encoding="utf-8",
    )
    (out / "benchmark.json").write_text(
        json.dumps({"system": info, "results": results}, indent=2),
        encoding="utf-8",
    )

    with open(out / "benchmark.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "scheduler", "uavs", "steps_completed", "mean_step_s", "p50_step_s",
            "p95_step_s", "max_step_s", "reset_s", "ram_peak_mb",
            "vram_peak_mb", "graph_nodes", "graph_edges", "task_rows",
            "flat_action_space", "exploration_rate", "status",
        ])
        for r in results:
            writer.writerow([
                r.get("scheduler"), r.get("uavs"), r.get("steps_completed"),
                r.get("mean_step_seconds"), r.get("p50_step_seconds"),
                r.get("p95_step_seconds"), r.get("max_step_seconds"),
                r.get("reset_seconds"), r.get("ram_peak_mb"),
                r.get("vram_peak_mb"), r.get("graph_nodes"),
                r.get("graph_edges"), r.get("task_rows"),
                r.get("flat_action_space"), r.get("exploration_rate"),
                r.get("status"),
            ])

    with open(out / "timing_breakdown.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scheduler", "uavs", "phase", "seconds_per_step"])
        for r in results:
            for phase in PHASES:
                writer.writerow([
                    r.get("scheduler"), r.get("uavs"), phase,
                    (r.get("phase_seconds_per_step") or {}).get(phase, ""),
                ])
            writer.writerow([
                r.get("scheduler"), r.get("uavs"), "unaccounted",
                r.get("unaccounted_seconds_per_step", ""),
            ])

    print(f"[benchmark] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
