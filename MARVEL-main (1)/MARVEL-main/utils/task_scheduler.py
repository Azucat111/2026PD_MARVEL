"""Lightweight high-level task scheduler layered above MARVEL navigation."""

from __future__ import annotations

import numpy as np


class TaskScheduler:
    """Route task-specific robots while preserving MARVEL exploration actions."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.assignments: dict[int, tuple[str, int | None]] = {}

    def apply(self, actions):
        actions = list(actions)
        self.assignments = {}
        self._apply_target_search(actions)
        self._apply_relay(actions)
        return actions

    def _apply_target_search(self, actions) -> None:
        tasks = [task for task in self.runtime.tasks.tasks.values()
                 if task.task_type == "target_search" and task.status == "active"]
        if not tasks:
            return
        task = max(tasks, key=lambda item: item.priority)
        targets = task.params.get("targets", [])
        pending = [(idx, target) for idx, target in enumerate(targets)
                   if idx not in task.found_targets]
        if not pending:
            return
        allowed = set(task.assigned_robot_types)
        robots = [(idx, robot) for idx, robot in enumerate(self.runtime.robots)
                  if not allowed or robot.robot_type in allowed]
        for ordinal, (action_idx, robot) in enumerate(sorted(robots, key=lambda item: item[1].robot_id)):
            target_idx, target = pending[ordinal % len(pending)]
            waypoint = np.array([target["x"], target["y"]], dtype=float)
            heading = float(np.degrees(np.arctan2(
                waypoint[1] - robot.position[1],
                waypoint[0] - robot.position[0])) % 360.0)
            actions[action_idx] = (waypoint, heading)
            self.assignments[robot.robot_id] = (task.task_id, target_idx)

    def _apply_relay(self, actions) -> None:
        tasks = [task for task in self.runtime.tasks.tasks.values()
                 if task.task_type == "relay" and task.status == "active"]
        if not tasks:
            return
        relay_types = set().union(*(set(task.assigned_robot_types) for task in tasks))
        relays = [(idx, robot) for idx, robot in enumerate(self.runtime.robots)
                  if not relay_types or robot.robot_type in relay_types]
        peers = [robot for robot in self.runtime.robots
                 if robot.robot_type not in relay_types] or list(self.runtime.robots)
        comm_range = float(self.runtime.comm.comm_range)
        for action_idx, relay in relays:
            farthest = max(peers, key=lambda peer: np.linalg.norm(peer.position - relay.position))
            distance = np.linalg.norm(farthest.position - relay.position)
            waypoint = relay.position
            if distance > comm_range * 0.75:
                waypoint = relay.position + (farthest.position - relay.position) * 0.5
            heading = float(np.degrees(np.arctan2(
                farthest.position[1] - relay.position[1],
                farthest.position[0] - relay.position[0])) % 360.0)
            actions[action_idx] = (np.asarray(waypoint, dtype=float), heading)
            self.assignments[relay.robot_id] = ("relay", None)
