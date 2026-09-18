"""Safety shield: filters infeasible actions before dynamics execution."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .obstacle_manager import ObstacleManager


class SafetyShield:
    """Pre-filters robot actions to prevent wall collisions and UAV-UAV proximity violations."""

    UAV_SAFE_DISTANCE = 0.5  # metres — inter-robot safety margin
    COLLISION_RADIUS = 0.2   # metres — passed to ObstacleManager.check_collision

    def __init__(self, obstacle_manager: ObstacleManager) -> None:
        self.obstacles = obstacle_manager
        self._events: List[Dict[str, Any]] = []

    def filter_actions(
        self,
        robots: List[Any],
        actions: List[Tuple[np.ndarray, float]],
        step: int,
    ) -> List[Tuple[np.ndarray, float]]:
        """Return a new action list where infeasible actions are replaced by hover-in-place.

        Args:
            robots: list of RobotState objects (must have .position, .robot_id, .heading).
            actions: list of (target_position, target_heading) tuples, one per robot.
            step:    current simulation step (for event logging).

        Returns:
            Filtered action list of the same length.
        """
        # First pass: resolve each robot's intended position (or hover fallback).
        intended: List[np.ndarray] = []
        safe: List[bool] = []

        for robot, (target_position, _target_heading) in zip(robots, actions):
            target = np.asarray(target_position, dtype=float)
            collides, reason = self.obstacles.check_collision(target, radius=self.COLLISION_RADIUS)
            if collides:
                self._record(robot.robot_id, step, f"static_obstacle:{reason}")
                intended.append(robot.position.copy())
                safe.append(False)
            else:
                intended.append(target)
                safe.append(True)

        # Second pass: check UAV-UAV proximity among intended positions.
        for i, robot in enumerate(robots):
            if not safe[i]:
                continue  # already hovering
            for j, other in enumerate(robots):
                if i == j:
                    continue
                if np.linalg.norm(intended[i] - intended[j]) < self.UAV_SAFE_DISTANCE:
                    self._record(robot.robot_id, step, f"uav_proximity:robot_{other.robot_id}")
                    intended[i] = robot.position.copy()
                    safe[i] = False
                    break

        # Build filtered action list: hover keeps current heading.
        filtered: List[Tuple[np.ndarray, float]] = []
        for i, (robot, (_target_pos, target_heading)) in enumerate(zip(robots, actions)):
            if safe[i]:
                filtered.append((intended[i], target_heading))
            else:
                filtered.append((robot.position.copy(), robot.heading))

        return filtered

    def pop_events(self) -> List[Dict[str, Any]]:
        """Drain and return all shield intercept events recorded since last call."""
        events, self._events = self._events, []
        return events

    def _record(self, robot_id: int, step: int, reason: str) -> None:
        self._events.append({"robot_id": robot_id, "step": step, "reason": reason})
