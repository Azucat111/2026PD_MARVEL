from __future__ import annotations

from dataclasses import replace

import numpy as np

from .event_allocator import GPPOEventAllocator
from .event_sources import (
    ObservedRelayDemandBuilder,
    PublicHeatPoint,
    SearchSlotBuilder,
)
from .protocol_profile import (
    GPPOControlClock,
    build_gppo_runtime_profile,
)
from .stale_state import StalePositionTracker
from .relay_release import FrozenRelayReleaseGate


class GPPOTaskScheduler:
    """High-level GPPO scheduler layered over MARVEL waypoint actions."""

    def __init__(
        self,
        runtime,
        checkpoint_path: str,
        *,
        device: str = "cpu",
        scheduler_config=None,
    ):
        self.runtime = runtime
        self.config = dict(scheduler_config or {})

        self.allocator = GPPOEventAllocator(
            runtime,
            checkpoint_path,
            device=device,
        )

        checkpoint = self.allocator.adapter.checkpoint

        self.profile = build_gppo_runtime_profile(
            runtime,
            checkpoint,
        )

        # Fail closed if runtime was not prepared before construction.
        if runtime.max_steps != self.profile.required_physics_steps:
            raise RuntimeError(
                "Runtime has not been prepared for the frozen GPPO "
                "mission horizon. Call prepare_gppo_config() before "
                "constructing SimulationRuntime."
            )

        if abs(
            float(runtime.comm.comm_range)
            - float(self.profile.comm_range)
        ) > 1e-9:
            raise RuntimeError(
                "Runtime communication range does not match "
                "the frozen GPPO protocol."
            )

        self.clock = GPPOControlClock(
            self.profile
        )

        self.tracker = StalePositionTracker()
        self.tracker.reset(runtime)
        self._last_observed_step = int(
            runtime.current_step
        )

        mixed = checkpoint["mixed_config"]

        self.max_search_uavs = int(
            mixed["max_search_uavs"]
        )
        self.max_relay_uavs = int(
            mixed["max_relay_uavs"]
        )

        self.search_builder = SearchSlotBuilder(
            max_search_uavs=self.max_search_uavs,
        )

        self.relay_builder = ObservedRelayDemandBuilder(
            comm_range=float(
                mixed["comm_range"]
            ),
            max_hops=int(
                mixed["comm_max_hops"]
            ),
            max_demands=self.max_relay_uavs,
        )

        self.relay_trigger_threshold = float(
            mixed["connectivity_threshold"]
        )

        self.relay_release_gate = FrozenRelayReleaseGate(
            release_threshold=float(
                mixed["relay_release_threshold"]
            ),
            # Frozen RelayCoordinator default.
            stable_release_steps=3,
        )

        self.heat_points = self._parse_heat_points(
            self.config.get(
                "public_heat_points",
                [],
            )
        )

        self.heat_by_id = {
            point.heat_id: point
            for point in self.heat_points
        }

        self.target_to_heat_id: dict[int, int] = {}

        for point in self.heat_points:
            if point.target_index is None:
                continue

            target_index = int(point.target_index)

            if target_index in self.target_to_heat_id:
                raise ValueError(
                    "Duplicate public heat mapping for "
                    f"target_index={target_index}"
                )

            self.target_to_heat_id[target_index] = int(
                point.heat_id
            )

        search_tasks = [
            task
            for task in self.runtime.tasks.tasks.values()
            if task.task_type == "target_search"
        ]

        if len(search_tasks) > 1:
            raise RuntimeError(
                "Current GPPO integration expects one "
                "target_search task."
            )

        if search_tasks:
            target_count = int(
                search_tasks[0].params.get(
                    "target_count",
                    0,
                )
            )

            expected = set(range(target_count))
            mapped = set(self.target_to_heat_id)

            missing = sorted(expected - mapped)
            extra = sorted(mapped - expected)

            if missing or extra:
                raise ValueError(
                    "Public heat-point mapping does not "
                    "match hidden target cardinality: "
                    f"missing={missing}, extra={extra}"
                )

        base = self.config.get("base_position")

        if base is None and self._has_task("relay"):
            raise ValueError(
                "GPPO relay integration requires an explicit "
                "task_scheduler.base_position."
            )

        self.base_position = (
            None
            if base is None
            else np.asarray(base, dtype=float)
        )

        self.serviced_heat_ids: set[int] = set()

        # heat_id -> uav_id
        self.search_assignments: dict[int, int] = {}

        # target_uav_id -> (helper_uav_id, anchor_position)
        self.relay_assignments: dict[
            int,
            tuple[int, np.ndarray],
        ] = {}

        # Compatibility with the existing TaskScheduler surface.
        self.assignments: dict[
            int,
            tuple[str, int | None],
        ] = {}

        self.last_event_assignments = []

    def _parse_heat_points(self, items):
        points = []

        for index, item in enumerate(items):
            heat_id = int(
                item.get("heat_id", index)
            )

            if "position" in item:
                position = item["position"]
            else:
                position = [
                    item["x"],
                    item["y"],
                ]

            points.append(
                PublicHeatPoint(
                    heat_id=heat_id,
                    position=(
                        float(position[0]),
                        float(position[1]),
                    ),
                    target_index=(
                        None
                        if item.get("target_index") is None
                        else int(item["target_index"])
                    ),
                    priority=float(
                        item.get("priority", 5.0)
                    ),
                    serviced=False,
                )
            )

        return points

    def _has_task(self, task_type: str) -> bool:
        return any(
            task.task_type == task_type
            for task in self.runtime.tasks.tasks.values()
        )

    def _active_task(self, task_type: str):
        tasks = [
            task
            for task in self.runtime.tasks.tasks.values()
            if task.task_type == task_type
            and task.status == "active"
        ]

        if not tasks:
            return None

        return max(
            tasks,
            key=lambda item: item.priority,
        )

    def mark_heat_serviced(
        self,
        heat_id: int,
    ) -> None:
        """Called later by the sensor-faithful Search controller."""
        heat_id = int(heat_id)

        self.serviced_heat_ids.add(heat_id)
        self.search_assignments.pop(
            heat_id,
            None,
        )

    def _observe_runtime(self) -> None:
        step = int(self.runtime.current_step)

        if step == self._last_observed_step:
            return

        self.tracker.observe(
            self.runtime
        )

        self._last_observed_step = step

    def _sync_search_completions(self) -> None:
        """Translate sensor-detected target IDs into serviced heat points.

        Target coordinates never enter this scheduler.
        """

        search_tasks = [
            task
            for task in self.runtime.tasks.tasks.values()
            if task.task_type == "target_search"
        ]

        for task in search_tasks:
            for target_index in task.found_targets:
                target_index = int(target_index)

                heat_id = self.target_to_heat_id.get(
                    target_index
                )

                if heat_id is None:
                    raise RuntimeError(
                        "Detected target has no public heat "
                        f"mapping: target_index={target_index}"
                    )

                self.mark_heat_serviced(
                    heat_id
                )

    def _refresh_search(
        self,
        stale_positions,
    ) -> None:
        task = self._active_task(
            "target_search"
        )

        if task is None:
            self.search_assignments.clear()
            return

        # Remove externally serviced heat points.
        for heat_id in list(
            self.search_assignments
        ):
            if heat_id in self.serviced_heat_ids:
                self.search_assignments.pop(
                    heat_id,
                    None,
                )

        free_slots = max(
            self.max_search_uavs
            - len(self.search_assignments),
            0,
        )

        if free_slots <= 0:
            return

        visible_points = [
            replace(
                point,
                serviced=(
                    point.heat_id
                    in self.serviced_heat_ids
                ),
            )
            for point in self.heat_points
        ]

        slots = self.search_builder.build(
            visible_points,
            active_heat_ids=(
                self.search_assignments.keys()
            ),
            source_task_id=task.task_id,
        )

        slots = slots[:free_slots]

        already_used = tuple(
            self.search_assignments.values()
        )

        if already_used:
            slots = [
                replace(
                    slot,
                    forbidden_uav_ids=tuple(
                        sorted(
                            set(
                                slot.forbidden_uav_ids
                            )
                            | set(already_used)
                        )
                    ),
                )
                for slot in slots
            ]

        decisions = self.allocator.allocate(
            slots,
            position_overrides=stale_positions,
        )

        self.last_event_assignments.extend(
            decisions
        )

        for decision in decisions:
            heat_id = int(
                decision.source_entity_id
            )

            self.search_assignments[
                heat_id
            ] = int(decision.uav_id)

    def _refresh_relay(
        self,
        stale_positions,
    ) -> None:
        task = self._active_task("relay")

        if (
            task is None
            or self.base_position is None
        ):
            self.relay_assignments.clear()
            self.relay_release_gate.reset_stability()
            return

        # --------------------------------------------------
        # Frozen semantics:
        #
        # Existing Relay event persists. Do not re-run GPPO
        # every second. First evaluate release counterfactually
        # with all helper UAVs removed.
        # --------------------------------------------------
        if self.relay_assignments:
            should_release = (
                self.relay_release_gate.observe(
                    assignments=self.relay_assignments,
                    positions=stale_positions,
                    relay_builder=self.relay_builder,
                    base_position=self.base_position,
                )
            )

            if should_release:
                self.relay_assignments.clear()

            # Frozen maybe_trigger() always returns here,
            # even on the step that release_all() fires.
            return

        # --------------------------------------------------
        # No active Relay event.
        # --------------------------------------------------
        self.relay_release_gate.reset_stability()

        snapshot = self.relay_builder.snapshot(
            stale_positions,
            base_position=self.base_position,
        )

        self.relay_release_gate.last_counterfactual_connectivity = (
            float(snapshot.connectivity_ratio)
        )

        # Frozen CommunicationManager.needs_relay().
        if (
            snapshot.connectivity_ratio
            >= self.relay_trigger_threshold
        ):
            return

        result = self.relay_builder.build(
            stale_positions,
            base_position=self.base_position,
            source_task_id=task.task_id,
            snapshot=snapshot,
        )

        # Search reservations remain unavailable to Relay if a
        # future scenario gives a UAV overlapping capabilities.
        reserved = set(
            self.search_assignments.values()
        )

        slots = []

        for slot in result.slots:
            forbidden = (
                set(slot.forbidden_uav_ids)
                | reserved
            )

            slots.append(
                replace(
                    slot,
                    forbidden_uav_ids=tuple(
                        sorted(forbidden)
                    ),
                )
            )

        if not slots:
            return

        decisions = self.allocator.allocate(
            slots,
            position_overrides=stale_positions,
        )

        self.last_event_assignments.extend(
            decisions
        )

        anchor_by_target = {
            int(slot.source_entity_id):
                np.asarray(
                    slot.position,
                    dtype=float,
                )
            for slot in slots
        }

        self.relay_assignments = {}

        for decision in decisions:
            target_uid = int(
                decision.source_entity_id
            )

            self.relay_assignments[
                target_uid
            ] = (
                int(decision.uav_id),
                anchor_by_target[target_uid],
            )

        # A newly triggered Relay event starts with a fresh
        # counterfactual stability window.
        self.relay_release_gate.reset_stability()

    def _high_level_update(self) -> None:
        stale = self.tracker.positions(
            self.profile.stale_age_fraction
        )

        self.last_event_assignments = []

        # Sensor detection from the preceding physics ticks is
        # consumed at this GPPO decision boundary.
        self._sync_search_completions()

        self._refresh_search(stale)
        self._refresh_relay(stale)

    @staticmethod
    def _heading(
        robot,
        goal,
    ) -> float:
        goal = np.asarray(
            goal,
            dtype=float,
        )

        delta = goal - robot.position

        if np.linalg.norm(delta) < 1e-12:
            return float(robot.heading)

        return float(
            np.degrees(
                np.arctan2(
                    delta[1],
                    delta[0],
                )
            )
            % 360.0
        )

    def apply(self, actions):
        actions = list(actions)

        self._observe_runtime()

        if self.clock.should_update(
            self.runtime.current_step
        ):
            self._high_level_update()

        robot_index = {
            int(robot.robot_id): index
            for index, robot
            in enumerate(self.runtime.robots)
        }

        self.assignments = {}

        used_uavs: set[int] = set()

        # Search override.
        for heat_id, uav_id in (
            self.search_assignments.items()
        ):
            if heat_id in self.serviced_heat_ids:
                continue

            point = self.heat_by_id.get(
                int(heat_id)
            )

            if point is None:
                continue

            index = robot_index.get(
                int(uav_id)
            )

            if index is None:
                continue

            robot = self.runtime.robots[index]

            goal = np.asarray(
                point.position,
                dtype=float,
            )

            actions[index] = (
                goal,
                self._heading(
                    robot,
                    goal,
                ),
            )

            used_uavs.add(int(uav_id))

            self.assignments[int(uav_id)] = (
                "target_search",
                int(heat_id),
            )

        # Relay override.
        for target_uid, (
            helper_uid,
            anchor,
        ) in self.relay_assignments.items():

            if helper_uid in used_uavs:
                raise RuntimeError(
                    f"UAV {helper_uid} received both "
                    "Search and Relay assignments"
                )

            index = robot_index.get(
                int(helper_uid)
            )

            if index is None:
                continue

            robot = self.runtime.robots[index]

            actions[index] = (
                anchor.copy(),
                self._heading(
                    robot,
                    anchor,
                ),
            )

            used_uavs.add(int(helper_uid))

            self.assignments[
                int(helper_uid)
            ] = (
                "relay",
                int(target_uid),
            )

        return actions
