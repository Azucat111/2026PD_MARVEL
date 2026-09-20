"""Run the headless scenario runtime and archive an auditable episode."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.evaluator import Evaluator
from utils.policy_adapter import MARVELPolicyAdapter
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime


def _make_log_dir(name: str) -> Path:
    path = ROOT / "logs" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    latest = ROOT / "logs" / "latest"
    if latest.exists() or latest.is_symlink():
        try:
            latest.unlink()
        except OSError:
            # Windows junctions can report as directories but are removed
            # through the directory-link command rather than rmtree.
            subprocess.run(["cmd.exe", "/c", "rmdir", str(latest)],
                           capture_output=True, check=False)
    if sys.platform == "win32":
        subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(latest), str(path)],
                       capture_output=True, check=False)
    else:
        latest.symlink_to(path, target_is_directory=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="MARVEL headless scenario runner")
    parser.add_argument("--scenario", required=True, help="scenario YAML file name or path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true",
                        help="仅校验并归档场景，不执行 episode")
    parser.add_argument("--use-marvel-policy", action="store_true",
                        help="使用 MARVEL PolicyNet 替代 default_actions()")
    parser.add_argument("--policy-interval", type=int, default=1,
                        help="MARVEL policy inference interval")
    args = parser.parse_args()

    scenario_path = Path(args.scenario)
    if not scenario_path.is_absolute():
        scenario_path = ROOT / "configs" / "scenarios" / scenario_path
    config = load_and_validate_scenario(scenario_path)
    np.random.seed(args.seed)
    random.seed(args.seed)
    config["scenario"]["random_seed"] = args.seed
    log_dir = _make_log_dir(config["scenario"]["name"])
    (log_dir / "scenario_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (log_dir / "scenario_config.yaml").write_text(
        Path(scenario_path).read_text(encoding="utf-8"), encoding="utf-8")
    (log_dir / "run_info.json").write_text(json.dumps({
        "scenario": config["scenario"]["name"],
        "seed": args.seed,
        "episodes": args.num_episodes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.prepare_only:
        print(f"prepared scenario={config['scenario']['name']} log_dir={log_dir}")
        return 0

    all_reports = []
    for episode in range(args.num_episodes):
        runtime = SimulationRuntime(config)
        observations = runtime.reset()

        policy_adapter = None
        if args.use_marvel_policy:
            policy_adapter = MARVELPolicyAdapter(runtime)
            policy_adapter.setup()
            policy_label = "marvel_policy" if policy_adapter._using_policy else "default_actions(fallback)"
        else:
            policy_label = "default_actions"

        evaluator = Evaluator(config, log_dir / f"episode_{episode:03d}")
        max_steps = args.max_steps or runtime.max_steps
        last_actions = runtime.default_actions()
        for step in range(max_steps):
            if policy_adapter is not None:
                if step % max(1, args.policy_interval) == 0:
                    last_actions = policy_adapter.get_actions(observations)
                actions = last_actions
            else:
                actions = runtime.default_actions()
            observations, info = runtime.step(actions)
            evaluator.record_step(step, {}, info, runtime=runtime)
            if info.get("terminated") or info.get("truncated"):
                break
        report = evaluator.generate_report()
        report["episode"] = episode
        report["task_summary"] = runtime.tasks.summary()
        all_reports.append(report)
        (log_dir / f"episode_{episode:03d}" / "events.json").write_text(
            json.dumps(runtime.get_event_log(), ensure_ascii=False, indent=2), encoding="utf-8")

    (log_dir / "episode_metrics.json").write_text(
        json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    (log_dir / "events.log").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in runtime.get_event_log()),
        encoding="utf-8")
    print(f"scenario={config['scenario']['name']} seed={args.seed} robots={len(runtime.robots)} policy={policy_label}")
    print(f"steps={max_steps} log_dir={log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
