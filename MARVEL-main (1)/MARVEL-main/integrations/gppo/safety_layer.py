from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .frozen_safety_controller import SafetyController


@dataclass(frozen=True)
class SharedHazardRisk:
    hazard_id: str
    kind: str
    center: np.ndarray
    radius: float
    clearance: float
    inside_no_fly_zone: bool
    within_warning_margin: bool


@dataclass
class SafetyRecord:
    uav_id: int
    hazard_id: str
    trigger_step: int
    preempted_assignment: tuple | None = None
    escaped_step: int | None = None


class SharedHazardAdapter:
    """Expose shared SimulationRuntime hazards with frozen Safety semantics."""

    def __init__(
        self,
        runtime,
        *,
        warning_margin: float = 5.0,
        safe_margin: float = 8.0,
    ):
        self.runtime = runtime
        self.warning_margin = float(warning_margin)
        self.safe_margin = float(safe_margin)

    @staticmethod
    def _radius_at(obstacle, step: int):
        if step < int(obstacle.spawn_step):
            return None

        elapsed = int(step) - int(obstacle.spawn_step)

        return min(
            float(obstacle.max_radius),
            float(obstacle.initial_radius)
            + elapsed * float(obstacle.expansion_rate),
        )

    def risk_at(
        self,
        point,
        *,
        step: int | None = None,
    ):
        if step is None:
            step = int(self.runtime.current_step)

        point = np.asarray(
            point,
            dtype=float,
        )

        best = None

        for obstacle in self.runtime.obstacles.dynamic_obstacles:
            radius = self._radius_at(
                obstacle,
                step,
            )

            if radius is None:
                continue

            center = np.asarray(
                obstacle.position,
                dtype=float,
            )

            clearance = (
                float(
                    np.linalg.norm(
                        point - center
                    )
                )
                - radius
            )

            risk = SharedHazardRisk(
                hazard_id=str(
                    obstacle.obstacle_id
                ),
                kind=str(
                    obstacle.obstacle_id
                ),
                center=center.copy(),
                radius=float(radius),
                clearance=float(clearance),
                inside_no_fly_zone=(
                    clearance <= 0.0
                ),
                within_warning_margin=(
                    clearance
                    <= self.warning_margin
                ),
            )

            if (
                best is None
                or risk.clearance
                < best.clearance
            ):
                best = risk

        return best

    def is_safe(
        self,
        point,
        *,
        step: int | None = None,
    ) -> bool:
        risk = self.risk_at(
            point,
            step=step,
        )

        return (
            risk is None
            or risk.clearance
            > self.safe_margin
        )


class FrozenSafetyLayer:
    """Frozen Phase-11/14 Safety state machine on shared runtime."""

    def __init__(
        self,
        runtime,
        *,
        warning_margin: float = 5.0,
        safe_margin: float = 8.0,
        stable_release_steps: int = 2,
        num_angle_bins: int = 36,
    ):
        self.runtime = runtime

        self.hazards = SharedHazardAdapter(
            runtime,
            warning_margin=warning_margin,
            safe_margin=safe_margin,
        )

        self.controller = SafetyController(
            num_angle_bins=num_angle_bins,
        )

        self.num_angle_bins = int(
            num_angle_bins
        )

        self.stable_release_steps = max(
            1,
            int(stable_release_steps),
        )

        self.active_uav_ids: set[int] = set()

        self.safe_counter: dict[int, int] = {}

        self.records: list[SafetyRecord] = []

        self.active_record_by_uav = {}

        self.trigger_count_by_uav = {}

        self._inside_prev = {
            int(robot.robot_id): False
            for robot in runtime.robots
        }

        self.safety_reward_sum = 0.0
        self.no_fly_violations = 0
        self.warning_margin_steps = 0
        self.successful_escapes = 0

        self.escape_waypoint_fallback_count = 0

        self._marvel_agents = {}

    def bind_marvel_agents(
        self,
        agents,
    ):
        self._marvel_agents = {
            int(agent.id): agent
            for agent in agents
        }

    def pre_motion_update(
        self,
        scheduler,
        step: int,
    ):
        triggered = []

        for robot in self.runtime.robots:
            uid = int(robot.robot_id)

            risk = self.hazards.risk_at(
                robot.position,
                step=step,
            )

            if (
                risk is None
                or risk.clearance
                > self.hazards.warning_margin
            ):
                continue

            if uid in self.active_uav_ids:
                continue

            previous = (
                scheduler.assignment_for_uav(
                    uid
                )
            )

            scheduler.preempt_uav(
                uid
            )

            self.active_uav_ids.add(uid)
            self.safe_counter[uid] = 0

            record = SafetyRecord(
                uav_id=uid,
                hazard_id=str(
                    risk.hazard_id
                ),
                trigger_step=int(step),
                preempted_assignment=previous,
            )

            self.records.append(record)

            self.active_record_by_uav[
                uid
            ] = record

            self.trigger_count_by_uav[
                uid
            ] = (
                self.trigger_count_by_uav.get(
                    uid,
                    0,
                )
                + 1
            )

            triggered.append(uid)

        return triggered

    def post_motion_update(
        self,
        scheduler,
        step: int,
    ):
        escaped = []

        for robot in self.runtime.robots:
            uid = int(robot.robot_id)

            risk = self.hazards.risk_at(
                robot.position,
                step=step,
            )

            inside = bool(
                risk is not None
                and risk.inside_no_fly_zone
            )

            previous_inside = bool(
                self._inside_prev.get(
                    uid,
                    False,
                )
            )

            # Exact frozen transition semantics.
            if inside and not previous_inside:
                self.safety_reward_sum -= 50.0
                self.no_fly_violations += 1

            elif (
                risk is not None
                and not inside
                and risk.within_warning_margin
            ):
                self.safety_reward_sum -= 10.0
                self.warning_margin_steps += 1

            self._inside_prev[uid] = inside

            if uid not in self.active_uav_ids:
                continue

            if self.hazards.is_safe(
                robot.position,
                step=step,
            ):
                self.safe_counter[uid] = (
                    self.safe_counter.get(
                        uid,
                        0,
                    )
                    + 1
                )
            else:
                self.safe_counter[uid] = 0

            if (
                self.safe_counter[uid]
                >= self.stable_release_steps
            ):
                self.active_uav_ids.remove(
                    uid
                )

                self.safe_counter.pop(
                    uid,
                    None,
                )

                self.safety_reward_sum += 2.0
                self.successful_escapes += 1

                record = (
                    self.active_record_by_uav.pop(
                        uid,
                        None,
                    )
                )

                if record is not None:
                    record.escaped_step = int(
                        step
                    )

                escaped.append(uid)

        return escaped

    def select_action(
        self,
        robot,
        *,
        step: int,
    ):
        uid = int(robot.robot_id)

        agent = self._marvel_agents.get(
            uid
        )

        if (
            agent is None
            or agent.node_coords is None
            or agent.neighbor_indices is None
        ):
            self.escape_waypoint_fallback_count += 1

            return (
                robot.position.copy(),
                float(robot.heading),
            )

        # Adapter presenting exactly the fields expected by
        # the frozen SafetyController.
        view = SimpleNamespace(
            id=uid,
            location=np.asarray(
                robot.position,
                dtype=float,
            ),
            node_coords=np.asarray(
                agent.node_coords,
                dtype=float,
            ),
            neighbor_indices=np.asarray(
                agent.neighbor_indices,
                dtype=int,
            ),
            current_index=(
                None
                if agent.current_index is None
                else int(agent.current_index)
            ),
        )

        # Frozen controller calls hazard_manager.risk_at(point)
        # without a step argument.
        original_step = int(
            self.runtime.current_step
        )

        waypoint, _node_index, heading_bin = (
            self.controller.select_next_waypoint(
                view,
                self.hazards,
            )
        )

        heading_deg = (
            float(heading_bin)
            * 360.0
            / float(self.num_angle_bins)
        )

        return (
            np.asarray(
                waypoint,
                dtype=float,
            ),
            heading_deg,
        )

    @property
    def retrigger_count(self):
        return int(
            sum(
                max(0, count - 1)
                for count in (
                    self.trigger_count_by_uav.values()
                )
            )
        )
