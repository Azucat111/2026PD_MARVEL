"""Auditable metrics and report generation for simulation episodes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class Evaluator:
    def __init__(self, scenario_config: Dict[str, Any], log_dir: str | Path):
        self.config = scenario_config.get("evaluation", {})
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: list[Dict[str, Any]] = []
        self.total_distance = 0.0
        self.targets_found = 0
        self.collision_count = 0
        self.connected_steps = 0
        self.connectivity_steps = 0

    def record_step(self, step: int, observations: Dict, info: Dict[str, Any], runtime=None) -> None:
        events = info.get("task_events", [])
        self.targets_found += sum(event.get("type") == "target_found" for event in events)
        self.collision_count += len(info.get("collisions", []))
        if runtime is not None:
            self.total_distance = sum(robot.travel_distance for robot in runtime.robots)
            exploration_rate = runtime.exploration_rate
        else:
            exploration_rate = float(info.get("exploration_rate", 0.0))
        topology = info.get("comm_topology")
        connectivity = 1.0
        if topology is not None:
            connectivity, _ = _connectivity(topology)
            connectivity = float(info.get("connectivity_ratio", connectivity))
            self.connectivity_steps += 1
            self.connected_steps += connectivity
            connectivity = self.connected_steps / self.connectivity_steps
        self.metrics_history.append({
            "step": step,
            "exploration_rate": exploration_rate,
            "targets_found": self.targets_found,
            "connectivity_ratio": float(connectivity),
            "collision_count": self.collision_count,
            "total_distance": self.total_distance,
        })

    def generate_report(self) -> Dict[str, Any]:
        latest = self.metrics_history[-1] if self.metrics_history else {
            "step": 0, "exploration_rate": 0.0, "targets_found": 0,
            "connectivity_ratio": 0.0, "collision_count": 0, "total_distance": 0.0,
        }
        metric_values = {
            "exploration_rate": latest["exploration_rate"],
            "targets_found": latest["targets_found"],
            "makespan": latest["step"] + 1,
            "connectivity_ratio": latest["connectivity_ratio"],
            "collision_count": latest["collision_count"],
            "total_distance": latest["total_distance"],
        }
        final_metrics = {}
        for metric in self.config.get("metrics", []):
            name = metric["name"]
            value = metric_values.get(name, 0.0)
            target = metric.get("target")
            weight = float(metric.get("weight", 0.0))
            if name == "collision_count":
                score = max(0.0, 1.0 + float(value) * float(metric.get("penalty_per_collision", -1.0)))
            elif name in ("makespan", "total_distance") and target:
                score = max(0.0, float(target) / max(float(value), 1e-9))
            elif name == "targets_found" and metric.get("score_per_target") is not None:
                score = float(value) * float(metric["score_per_target"])
            elif target:
                score = float(value) / float(target)
            else:
                score = float(value)
            final_metrics[name] = {
                "value": value, "target": target, "weight": weight,
                "score": score, "weighted_score": score * weight,
            }
        report = {
            "metrics": final_metrics,
            "total_score": sum(item["weighted_score"] for item in final_metrics.values()),
            "steps": len(self.metrics_history),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (self.log_dir / "episode_metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.log_dir / "metrics_history.json").write_text(
            json.dumps(self.metrics_history, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


def _connectivity(topology) -> tuple[bool, int]:
    n = len(topology)
    if n == 0:
        return True, 0
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in range(n):
            if topology[node, neighbor] or topology[neighbor, node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return len(seen) == n, n - len(seen) + 1
