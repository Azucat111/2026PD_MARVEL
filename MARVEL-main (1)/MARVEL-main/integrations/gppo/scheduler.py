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
from .safety_layer import FrozenSafetyLayer
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

        self.safety = FrozenSafetyLayer(
            runtime,
            warning_margin=float(
                self.config.get(
                    "safety_warning_margin",
                    5.0,
                )
            ),
            safe_margin=float(
                self.config.get(
                    "safety_safe_margin",
                    8.0,
                )
            ),
            stable_release_steps=int(
                self.config.get(
                    "safety_stable_release_steps",
                    2,
                )
            ),
        )

        # Optional hook used by SimulationRuntime after dynamics.
        runtime.high_level_scheduler = self

    def bind_marvel_agents(self, agents):
        self.safety.bind_marvel_agents(
            agents
        )

    def assignment_for_uav(
        self,
        uav_id: int,
    ):
        uid = int(uav_id)

        for heat_id, assigned_uid in (
            self.search_assignments.items()
        ):
            if int(assigned_uid) == uid:
                return (
                    "target_search",
                    int(heat_id),
                )

        for target_uid, (
            helper_uid,
            _anchor,
        ) in self.relay_assignments.items():
            if int(helper_uid) == uid:
                return (
                    "relay",
                    int(target_uid),
                )

        return None

    def preempt_uav(
        self,
        uav_id: int,
    ):
        uid = int(uav_id)

        previous = self.assignment_for_uav(
            uid
        )

        for heat_id in list(
            self.search_assignments
        ):
            if (
                int(
                    self.search_assignments[
                        heat_id
                    ]
                )
                == uid
            ):
                self.search_assignments.pop(
                    heat_id
                )

        relay_removed = False

        for target_uid in list(
            self.relay_assignments
        ):
            helper_uid, _anchor = (
                self.relay_assignments[
                    target_uid
                ]
            )

            if int(helper_uid) == uid:
                self.relay_assignments.pop(
                    target_uid
                )

                relay_removed = True

        if relay_removed:
            self.relay_release_gate.reset_stability()

        return previous

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

        blocked_uavs = (
            set(
                self.search_assignments.values()
            )
            | set(
                self.safety.active_uav_ids
            )
        )

        if blocked_uavs:
            slots = [
                replace(
                    slot,
                    forbidden_uav_ids=tuple(
                        sorted(
                            set(
                                slot.forbidden_uav_ids
                            )
                            | blocked_uavs
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
                | set(
                    self.safety.active_uav_ids
                )
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

    def _apply_hazard_traversability(
        self,
        actions,
        robot_index,
    ):
        """Frozen v9 ordinary-motion hazard guard.

        Explore/Search/Relay cannot enter the same warning
        buffer that triggers SAFETY. SAFETY escape motion is
        deliberately exempt.
        """

        guarded = list(actions)

        step = int(
            self.runtime.current_step
        )

        for robot in self.runtime.robots:
            uid = int(robot.robot_id)

            # SAFETY controller must remain free to escape.
            if uid in self.safety.active_uav_ids:
                continue

            index = robot_index.get(uid)

            if index is None:
                continue

            desired, desired_heading = (
                guarded[index]
            )

            start = np.asarray(
                robot.position,
                dtype=float,
            )[:2]

            desired = np.asarray(
                desired,
                dtype=float,
            )[:2]

            if not (
                self.safety.hazards
                .segment_intersects_warning_buffer(
                    start,
                    desired,
                    step=step,
                )
            ):
                continue

            alternatives = []

            agent = self.safety.marvel_agent(
                uid
            )

            if (
                agent is not None
                and agent.neighbor_indices is not None
                and agent.node_coords is not None
            ):
                node_coords = np.asarray(
                    agent.node_coords,
                    dtype=float,
                )

                for raw in np.asarray(
                    agent.neighbor_indices
                ).reshape(-1):

                    node_idx = int(raw)

                    if (
                        node_idx < 0
                        or node_idx
                        >= len(node_coords)
                    ):
                        continue

                    candidate = np.asarray(
                        node_coords[node_idx],
                        dtype=float,
                    )[:2]

                    if (
                        self.safety.hazards
                        .segment_intersects_warning_buffer(
                            start,
                            candidate,
                            step=step,
                        )
                    ):
                        continue

                    risk = (
                        self.safety.hazards
                        .risk_at(
                            candidate,
                            step=step,
                        )
                    )

                    if (
                        risk is not None
                        and risk.clearance
                        <=
                        self.safety.hazards.warning_margin
                    ):
                        continue

                    alternatives.append(
                        candidate
                    )

            if alternatives:
                final = min(
                    alternatives,
                    key=lambda candidate:
                        float(
                            np.linalg.norm(
                                candidate
                                - desired
                            )
                        ),
                )
            else:
                # Exact frozen fallback.
                final = start.copy()

            guarded[index] = (
                np.asarray(
                    final,
                    dtype=float,
                ),
                self._heading(
                    robot,
                    final,
                ),
            )

        return guarded

    def post_physics_step(self):
        """Called by SimulationRuntime after one physical 0.1 s step."""

        completed_physics_step = (
            int(self.runtime.current_step)
            + 1
        )

        # Frozen Safety post-motion semantics occur once
        # per 1-second mission step.
        if (
            completed_physics_step
            % self.profile.high_level_interval_steps
            != 0
        ):
            return []

        escaped = self.safety.post_motion_update(
            self,
            int(self.runtime.current_step),
        )

        return [
            {
                "type": "gppo_safety_escape",
                "robot_id": int(uid),
                "step": int(
                    self.runtime.current_step
                ),
            }
            for uid in escaped
        ]

    def apply(self, actions):
        actions = list(actions)

        if self.clock.should_update(
            self.runtime.current_step
        ):
            # Frozen CommunicationManager stores its position
            # history once per 1-second control step, not once
            # per 0.1-second physics tick.
            self.tracker.observe(
                self.runtime
            )
            self._last_observed_step = int(
                self.runtime.current_step
            )

            # Highest-priority frozen Safety override.
            # Preemption happens before Search/Relay allocation,
            # so orphaned tasks can be reassigned immediately.
            self.safety.pre_motion_update(
                self,
                int(self.runtime.current_step),
            )

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

        # Frozen v9 ordinary-motion traversability guard.
        # Search/Relay overrides are already present here.
        actions = self._apply_hazard_traversability(
            actions,
            robot_index,
        )

        # Highest-priority final override.
        for uid in sorted(
            self.safety.active_uav_ids
        ):
            index = robot_index.get(
                int(uid)
            )

            if index is None:
                continue

            robot = self.runtime.robots[
                index
            ]

            actions[index] = (
                self.safety.select_action(
                    robot,
                    step=int(
                        self.runtime.current_step
                    ),
                )
            )

            self.assignments[int(uid)] = (
                "safety",
                None,
            )

        return actions
