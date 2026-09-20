"""Run fixed-seed multi-scale policy/baseline evaluations."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.evaluator import Evaluator
from utils.policy_adapter import MARVELPolicyAdapter
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime


def _scale_config(base, scale: int):
    config = json.loads(json.dumps(base))
    if scale == 4:
        config["robots"] = [{
            "id_range": [0, 3],
            "type": "explorer",
            "team_id": 1,
            "config": {
                "fov": 120, "sensor_range": 10, "velocity": 1.0,
                "yaw_rate": 35, "initial_positions": "connected_safe",
            },
        }]
    elif scale == 30:
        # Preserve the scenario's 20/5/5 role split.
        config["robots"] = [
            {**config["robots"][0], "id_range": [1, 20]},
            {**config["robots"][1], "id_range": [21, 25]},
            {**config["robots"][2], "id_range": [26, 30]},
        ]
    elif scale == 60:
        config["robots"] = [
            {**config["robots"][0], "id_range": [1, 40]},
            {**config["robots"][1], "id_range": [41, 50]},
            {**config["robots"][2], "id_range": [51, 60]},
        ]
    else:
        raise ValueError(f"unsupported scale: {scale}")
    config["communication"]["ensure_initial_connectivity"] = True
    return config


def run_once(config, seed: int, steps: int, use_policy: bool, policy_interval: int):
    np.random.seed(seed)
    random.seed(seed)
    runtime = SimulationRuntime(config)
    observations = runtime.reset()
    adapter = None
    if use_policy:
        adapter = MARVELPolicyAdapter(runtime)
        adapter.setup()
    evaluator = Evaluator(config, ROOT / "logs" / "scale_eval_tmp")
    last_actions = runtime.default_actions()
    start = time.perf_counter()
    for step in range(steps):
        if adapter is not None:
            if step % max(1, policy_interval) == 0:
                last_actions = adapter.get_actions(observations)
            actions = last_actions
        else:
            actions = runtime.default_actions()
        observations, info = runtime.step(actions)
        evaluator.record_step(step, {}, info, runtime=runtime)
        if info.get("terminated") or info.get("truncated"):
            break
    report = evaluator.generate_report()
    report["elapsed_seconds"] = time.perf_counter() - start
    report["robots"] = len(runtime.robots)
    report["seed"] = seed
    report["policy"] = bool(use_policy)
    report["task_summary"] = runtime.tasks.summary()
    event_types: dict[str, int] = {}
    for event in runtime.get_event_log():
        event_types[event["type"]] = event_types.get(event["type"], 0) + 1
    report["event_counts"] = event_types
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="urban_rescue_simple.yaml")
    parser.add_argument("--scales", default="4,30,60")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--policy-interval", type=int, default=5)
    parser.add_argument("--use-marvel-policy", action="store_true")
    parser.add_argument("--output", default="logs/scale_evaluation.json")
    args = parser.parse_args()

    scenario = ROOT / "configs" / "scenarios" / args.scenario
    base = load_and_validate_scenario(scenario)
    reports = []
    for scale in [int(value) for value in args.scales.split(",")]:
        for seed in [int(value) for value in args.seeds.split(",")]:
            print(f"running scale={scale} seed={seed} policy={args.use_marvel_policy}", flush=True)
            reports.append(run_once(
                _scale_config(base, scale), seed, args.steps,
                args.use_marvel_policy, args.policy_interval))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
