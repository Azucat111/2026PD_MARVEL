"""Minimal multi-task state machine for the rescue scenario."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

import numpy as np


@dataclass
class Task:
    task_id: str
    task_type: str
    priority: int
    params: Dict[str, Any]
    activation: Dict[str, Any] = field(default_factory=dict)
    assigned_robot_types: tuple[str, ...] = ()
    status: str = "pending"
    progress: float = 0.0
    found_targets: set[int] = field(default_factory=set)
    relay_ok: bool = False


class TaskManager:
    def __init__(self, task_configs: Iterable[Dict[str, Any]]):
        self.task_configs = list(task_configs)
        self.tasks: dict[str, Task] = {}
        self.reset()

    def reset(self) -> None:
        self.tasks = {}
        for config in self.task_configs:
            task = Task(
                task_id=config["task_id"],
                task_type=config["type"],
                priority=int(config.get("priority", 1)),
                params=config.get("params", {}),
                activation=config.get("activation", {}),
                assigned_robot_types=tuple(config.get("assigned_robot_types", [])),
            )
            self.tasks[task.task_id] = task

    def update(self, current_step: int, robot_states, observations,
               exploration_rate: float = 0.0, comm_connected: bool = True,
               connectivity_ratio: float | None = None,
               target_detections=None) -> list[Dict[str, Any]]:
        events = []
        for task in self.tasks.values():
            if task.status == "pending" and self._should_activate(task, current_step, exploration_rate):
                task.status = "active"
                events.append({"type": "task_activated", "task_id": task.task_id, "step": current_step})
            if task.status == "active":
                events.extend(self._update_active_task(
                    task, current_step, robot_states, exploration_rate,
                    comm_connected, connectivity_ratio,
                    target_detections))
        return events

    def _should_activate(self, task: Task, current_step: int, exploration_rate: float) -> bool:
        if not task.activation:
            return True
        step_gate = task.activation.get("step")
        if step_gate is not None and current_step < int(step_gate):
            return False
        condition = str(task.activation.get("condition", ""))
        if "exploration_rate" in condition and ">=" in condition:
            threshold = float(condition.split(">=", 1)[1].strip())
            return exploration_rate >= threshold
        return True

    def _update_active_task(self, task: Task, current_step: int, robot_states,
                            exploration_rate: float, comm_connected: bool,
                            connectivity_ratio: float | None = None,
                            target_detections=None) -> list[Dict[str, Any]]:
        events = []
        allowed_types = set(task.assigned_robot_types)
        if not allowed_types:
            allowed_types = set(task.activation.get("assigned_robot_types", []))
        eligible_robots = [robot for robot in robot_states
                           if not allowed_types or robot.robot_type in allowed_types]
        if task.task_type == "exploration":
            task.progress = exploration_rate
            if task.progress >= float(task.params.get("target_coverage", 1.0)):
                task.status = "complete"
                events.append({"type": "task_completed", "task_id": task.task_id, "step": current_step})
        elif task.task_type == "target_search":
            if target_detections is not None:
                detected = target_detections.get(
                    task.task_id,
                    {},
                )

                for idx, robot_id in sorted(
                    detected.items()
                ):
                    idx = int(idx)

                    if idx in task.found_targets:
                        continue

                    task.found_targets.add(idx)

                    events.append({
                        "type": "target_found",
                        "task_id": task.task_id,
                        "target_index": idx,
                        "robot_id": int(robot_id),
                        "step": current_step,
                        "detection_mode": "sensor_visible_cell",
                    })

                target_count = int(
                    task.params.get(
                        "target_count",
                        len(
                            task.params.get(
                                "targets", []
                            )
                        ),
                    )
                )

            else:
                # Legacy compatibility path for callers that do not
                # provide sensor-derived detections.
                targets = task.params.get("targets", [])

                for idx, target in enumerate(targets):
                    if idx in task.found_targets:
                        continue

                    target_pos = np.array(
                        [target["x"], target["y"]],
                        dtype=float,
                    )

                    radius = float(
                        target.get("radius", 5.0)
                    )

                    if any(
                        np.linalg.norm(
                            robot.position - target_pos
                        ) <= radius
                        for robot in eligible_robots
                    ):
                        task.found_targets.add(idx)

                        events.append({
                            "type": "target_found",
                            "task_id": task.task_id,
                            "target_index": idx,
                            "step": current_step,
                            "detection_mode": "legacy_radius",
                        })

                target_count = len(targets)

            task.progress = (
                len(task.found_targets)
                / max(target_count, 1)
            )

            if task.progress >= 1.0:
                task.status = "complete"

                events.append({
                    "type": "task_completed",
                    "task_id": task.task_id,
                    "step": current_step,
                })
        elif task.task_type == "relay":
            ratio = float(connectivity_ratio if connectivity_ratio is not None else comm_connected)
            task.progress = ratio
            target = float(task.params.get("min_connectivity", 0.9))
            relay_ok = task.progress >= target
            if relay_ok and not task.relay_ok:
                events.append({"type": "relay_connectivity_ok", "task_id": task.task_id,
                               "step": current_step, "connectivity_ratio": ratio})
            elif not relay_ok and task.relay_ok:
                events.append({"type": "relay_connectivity_lost", "task_id": task.task_id,
                               "step": current_step, "connectivity_ratio": ratio})
            task.relay_ok = relay_ok
        return events

    def summary(self) -> Dict[str, Any]:
        return {
            task_id: {
                "type": task.task_type,
                "status": task.status,
                "progress": task.progress,
                "found_targets": sorted(task.found_targets),
            }
            for task_id, task in self.tasks.items()
        }

    def all_complete(self) -> bool:
        required = [task for task in self.tasks.values() if task.task_type != "relay"]
        return bool(required) and all(task.status == "complete" for task in required)
