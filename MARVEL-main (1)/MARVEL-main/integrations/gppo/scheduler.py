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
from .observed_graph import observed_next_hop
from .search_recovery import ObservedSearchRecovery
from .task_graph import Subtask, TaskType
from .safety_layer import FrozenSafetyLayer
from .extended_safety_motion import ObservedMotionSafetyLayer
from .relay_release import FrozenRelayReleaseGate
from .mesh_relay import ObservedMeshRelayDemandBuilder, ObservedMeshRelayReleaseGate


# Frozen test_parameter.NUM_ANGLES_BIN.  The frozen worker quantizes the
# heading of the *final* commanded waypoint into this many bins before
# motion:
#
#   angle = degrees(atan2(dy, dx)) % 360
#   index = int(floor(angle / 360 * NUM_ANGLES_BIN)) % NUM_ANGLES_BIN
#   desired_heading = index * (360 / NUM_ANGLES_BIN)
FROZEN_NUM_ANGLES_BIN = 36

# Frozen Relay subtask as registered by `StudyTask`/`RelayCoordinator`:
# priority 2, one second of processing, and the relay anchor as its
# position.  Its identity only matters as the first graph row.
NATIVE_RELAY_SUBTASK_ID = 0
NATIVE_RELAY_PRIORITY = 2.0
NATIVE_RELAY_PROCESSING_TIME = 1.0


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

        candidate_options = {}
        if self.config.get('extended_checkpoint') is not None:
            candidate_options['extended_checkpoint'] = self.config['extended_checkpoint']
            candidate_options['extended_checkpoint_sha256'] = self.config.get('extended_checkpoint_sha256')
        elif self.config.get('extended_checkpoint_sha256') is not None:
            raise ValueError('Candidate hash requires explicit extended checkpoint')
        self.allocator = GPPOEventAllocator(runtime, checkpoint_path, device=device, **candidate_options)
        if getattr(self.allocator.adapter,'model_protocol',None)=='extended-gppo-public-opportunity-v1':
            from .opportunity_features import public_search_context
            self.allocator.adapter.public_search_context_provider=lambda:public_search_context(self)

        # Adapter that owns the frozen native post-selection; bound by
        # MARVELPolicyAdapter.setup() for marvel_native only.
        self._post_selection = None

        # Marvel-native only: frozen always hands the allocator the full
        # subtask registry, so the Search allocation must present the Relay
        # row plus every Search subtask registered so far, not just the slots
        # being allocated.  See `_native_search_context`.
        self._native_relay_row = None
        self._native_search_rows = {}

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

        self.search_activation_threshold = float(
            mixed["activation_threshold"]
        )

        frozen_profile = runtime.config.get(
            "_gppo_profile",
            {},
        )

        self.search_service_steps = int(
            frozen_profile.get(
                "search_service_steps",
                6,
            )
        )

        self.search_reach_radius = float(
            frozen_profile.get(
                "search_reach_radius",
                3.5,
            )
        )

        self.search_activated = False
        self.search_activation_step = None

        self.search_builder = SearchSlotBuilder(
            max_search_uavs=self.max_search_uavs,
            # Frozen Search subtasks are registered with
            # processing_time=float(search_service_steps).
            search_service_steps=float(self.search_service_steps),
        )

        extended_mesh = getattr(runtime,"geometry_mode",None)=="extended"
        relay_builder_type = ObservedMeshRelayDemandBuilder if extended_mesh else ObservedRelayDemandBuilder
        self.relay_builder = relay_builder_type(
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

        release_kwargs = {}
        if extended_mesh:
            release_kwargs = dict(min_connectivity=self.relay_trigger_threshold,
                fleet_uav_ids=tuple(int(robot.robot_id) for robot in runtime.robots))
        release_gate_type = ObservedMeshRelayReleaseGate if extended_mesh else FrozenRelayReleaseGate
        self.relay_release_gate = release_gate_type(
            release_threshold=float(
                mixed["relay_release_threshold"]
            ),
            # Frozen RelayCoordinator default.
            stable_release_steps=3,
            **release_kwargs,
        )

        # Public Search prior. Populated by install_search_scenario() after
        # SimulationRuntime.reset(), because the frozen generator needs the
        # initial UAV positions.  Empty until then, which fails closed.
        self.heat_points: list[PublicHeatPoint] = []

        self.heat_by_id: dict[int, PublicHeatPoint] = {}

        # Frozen coordinator stores dwell per Search subtask.
        # In this integration one heat point corresponds to one
        # persistent Search slot, so heat_id is the equivalent key.
        self.search_dwell_by_heat: dict[int, int] = {}
        self._search_scan_bearings: dict[int, float] = {}
        self._search_scan_phases: dict[int, int] = {}
        self._search_scan_owners: dict[int, int] = {}

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

        self._marvel_agents = {}
        self._search_recovery = self._make_search_recovery(runtime)
        self._search_viewpoints = None
        self._peripheral_search_targets = {}
        if self._search_recovery is not None:
            from .search_viewpoints import ObservedSearchViewpoints
            fov, sensor_range = runtime._sensor_params()
            self._search_viewpoints = ObservedSearchViewpoints(
                runtime.obstacles.frame, sensor_range=sensor_range, fov=fov,
                prior_radius=float(runtime.config.get('_gppo_profile', {}).get('search_guaranteed_hidden_radius', 5.5)))

        safety_type = FrozenSafetyLayer if self._search_recovery is None else ObservedMotionSafetyLayer
        safety_kwargs = {} if self._search_recovery is None else {"recovery": self._search_recovery}
        self.safety = safety_type(
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
            stable_release_steps=int(self.config.get("safety_stable_release_steps", 2)),
            **safety_kwargs,
        )

        if isinstance(self.relay_builder, ObservedMeshRelayDemandBuilder):
            self.relay_builder.anchor_candidates_provider = self._observed_relay_anchor_candidates
            self.relay_builder.motion_candidate_guard = self._observed_relay_motion_impossible

        # Optional hook used by SimulationRuntime after dynamics.
        runtime.high_level_scheduler = self

    def bind_marvel_agents(self, agents):
        self._marvel_agents = {
            int(agent.id): agent
            for agent in agents
        }

        self.safety.bind_marvel_agents(
            agents
        )

        self.allocator.bind_marvel_agents(
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

        recovery = getattr(self, "_search_recovery", None)
        if recovery is not None:
            recovery.reset_uid(uid)
        return previous

    def install_search_scenario(
        self,
        heat_points,
    ) -> None:
        """Install the PUBLIC projection of the frozen Search scenario.

        Must be called after ``SimulationRuntime.reset()`` and before any
        Search activation.  Only public heat points cross this boundary; the
        hidden survivors and the ``target_index -> heat_id`` association stay
        in the environment layer.
        """

        points = list(heat_points)

        self.heat_points = points

        self.heat_by_id = {
            int(point.heat_id): point
            for point in points
        }

        if len(self.heat_by_id) != len(points):
            raise ValueError(
                "Duplicate heat_id in Search scenario"
            )

        self.search_dwell_by_heat = {
            int(point.heat_id): 0
            for point in points
        }

        self._search_scan_bearings = {}
        self._search_scan_phases = {}
        self._search_scan_owners = {}
        recovery = getattr(self, "_search_recovery", None)
        if recovery is not None:
            recovery.reset()
        # A fresh scenario invalidates every previous Search assignment.
        self.search_assignments.clear()

        self.serviced_heat_ids.clear()

        self.search_activated = False
        self.search_activation_step = None

    # ------------------------------------------------------------------
    # Persistent UAV high-level state (frozen TaskManager bridge)
    # ------------------------------------------------------------------
    def _robot_by_id(self, uav_id: int):
        for robot in self.runtime.robots:
            if int(robot.robot_id) == int(uav_id):
                return robot

        return None

    def _apply_uav_state(
        self,
        uav_id: int,
        task_type: str,
        total_time: float,
    ) -> None:
        """Frozen ``TaskManager.apply_assignment`` robot-side effects.

        ``robot.busy_until = current_time + travel + processing`` and
        ``available = False`` until the assignment is released.
        """

        robot = self._robot_by_id(uav_id)

        if robot is None:
            return

        robot.available = False
        robot.busy_until = (
            float(self.runtime.current_step * self.runtime.dt)
            + float(total_time)
        )
        robot.assigned_task_num = int(
            getattr(robot, "assigned_task_num", 0)
        ) + 1
        robot.current_task = str(task_type)

    def _release_uav_state(self, uav_id: int) -> None:
        """Frozen ``TaskManager._release_robot`` robot-side effects."""

        robot = self._robot_by_id(uav_id)

        if robot is None:
            return

        robot.available = True
        robot.busy_until = float(
            self.runtime.current_step * self.runtime.dt
        )
        robot.current_task = "exploration"

    def _reconcile_uav_state(self) -> None:
        """Release UAVs that no longer hold any high-level assignment.

        Frozen releases on completion, cancellation and Safety preemption,
        which all end in ``_release_robot``.  Anything not currently owned
        by Search, Relay or Safety is an idle Exploration UAV.
        """

        owned: set[int] = set()

        for heat_id, uid in self.search_assignments.items():
            if int(heat_id) not in self.serviced_heat_ids:
                owned.add(int(uid))

        for helper_uid, _anchor in self.relay_assignments.values():
            owned.add(int(helper_uid))

        probe = getattr(self, '_relay_probe', None)
        if probe is not None:
            owned.add(int(probe['uid']))

        for robot in self.runtime.robots:
            uid = int(robot.robot_id)

            if uid in self.safety.active_uav_ids:
                # Frozen SafetyCoordinator: current_task=SAFETY,
                # available=False, busy_until=current_time.
                robot.current_task = "safety"
                robot.available = False
                robot.busy_until = float(
                    self.runtime.current_step * self.runtime.dt
                )
                continue

            if uid in owned:
                continue

            if not bool(getattr(robot, "available", True)):
                self._release_uav_state(uid)

            robot.current_task = "exploration"

        self._publish_search_uavs()

    def _publish_search_uavs(self) -> None:
        """Publish the frozen "currently assigned Search UAV" set.

        This is ``robot.current_task == TARGET_SEARCH`` in the frozen
        worker, and it is what gates sensor detection.
        """

        self.runtime.active_search_uav_ids = {
            int(uid)
            for heat_id, uid in self.search_assignments.items()
            if int(heat_id) not in self.serviced_heat_ids
        }

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

        self._publish_search_uavs()

    def _observe_runtime(self) -> None:
        step = int(self.runtime.current_step)

        if step == self._last_observed_step:
            return

        self.tracker.observe(
            self.runtime
        )

        self._last_observed_step = step

    def _maybe_activate_search(self) -> bool:
        """Frozen Search activation: explored_rate >= threshold only."""

        if self.search_activated:
            return False

        if not self._has_task(
            "target_search"
        ):
            return False

        # Fail closed: the frozen generator has not run, so there is no
        # public prior to allocate against.
        if not self.heat_points:
            return False

        if (
            float(self.runtime.exploration_rate)
            <
            self.search_activation_threshold
        ):
            return False

        self.search_activated = True

        self.search_activation_step = int(
            self.clock.mission_step(
                self.runtime.current_step
            )
        )

        return True

    def _sync_search_completions(self) -> list[int]:
        """Consume the abstract Search completion notification.

        The environment layer resolves "which hidden survivor was detected"
        into "which public heat point is complete" and hands back only heat
        ids.  Neither hidden coordinates nor the target-to-heat association
        enter this class.
        """

        resolver = getattr(
            self.runtime,
            "completed_search_heat_ids",
            None,
        )

        if resolver is None:
            return []

        newly_serviced = []

        for heat_id in sorted(
            int(value) for value in resolver()
        ):
            if heat_id in self.serviced_heat_ids:
                continue

            self.mark_heat_serviced(heat_id)

            newly_serviced.append(int(heat_id))

        return newly_serviced

    def _update_search_service(self):
        """Preserve native dwell; require public confirmation in extended scenarios."""

        events = []

        robot_by_id = {
            int(robot.robot_id): robot
            for robot in self.runtime.robots
        }

        for heat_id, uav_id in list(
            self.search_assignments.items()
        ):
            heat_id = int(heat_id)
            uav_id = int(uav_id)

            if heat_id in self.serviced_heat_ids:
                continue

            point = self.heat_by_id.get(
                heat_id
            )

            robot = robot_by_id.get(
                uav_id
            )

            if point is None or robot is None:
                continue

            distance = float(
                np.linalg.norm(
                    np.asarray(
                        robot.position,
                        dtype=float,
                    )[:2]
                    -
                    np.asarray(
                        point.position,
                        dtype=float,
                    )[:2]
                )
            )

            if (
                distance
                <=
                self.search_reach_radius
                + 1e-9
            ):
                self.search_dwell_by_heat[
                    heat_id
                ] = (
                    self.search_dwell_by_heat.get(
                        heat_id,
                        0,
                    )
                    + 1
                )
            else:
                # Frozen semantics require consecutive dwell.
                self.search_dwell_by_heat[
                    heat_id
                ] = 0

            if (
                self.search_dwell_by_heat[
                    heat_id
                ]
                >=
                self.search_service_steps
            ):
                dwell_steps = int(
                    self.search_dwell_by_heat[
                        heat_id
                    ]
                )

                confirmed = getattr(self.runtime, "completed_search_heat_ids", None)
                if (getattr(self, "_search_recovery", None) is not None
                        and (not callable(confirmed) or heat_id not in confirmed())):
                    if dwell_steps == self.search_service_steps:
                        events.append({"type": "gppo_search_service_unconfirmed",
                                       "heat_id": heat_id, "robot_id": uav_id,
                                       "dwell_steps": dwell_steps})
                    continue

                self.mark_heat_serviced(
                    heat_id
                )

                events.append({
                    "type":
                        "gppo_search_heat_serviced",
                    "heat_id":
                        heat_id,
                    "robot_id":
                        uav_id,
                    "reason":
                        "dwell",
                    "dwell_steps":
                        dwell_steps,
                    "step":
                        int(
                            self.runtime.current_step
                        ),
                })

        return events

    def _refresh_search(
        self,
        stale_positions,
    ) -> None:
        if not self.search_activated:
            self.search_assignments.clear()
            self._publish_search_uavs()
            return

        task = self._active_task(
            "target_search"
        )

        if task is None:
            self.search_assignments.clear()
            self._publish_search_uavs()
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

        extended_catalog = getattr(self.runtime, 'geometry_mode', None) == 'extended'
        slots = self.search_builder.build(
            visible_points,
            active_heat_ids=(
                self.search_assignments.keys()
            ),
            source_task_id=task.task_id,
            **({"include_all_pending": True} if extended_catalog else {}),
        )

        if not extended_catalog:
            slots = slots[:free_slots]

        # Frozen semantics: an assigned UAV is globally unavailable.
        # Active Relay helpers are already executing a task and must not be
        # re-allocated to Search, otherwise apply() would raise the
        # "received both Search and Relay assignments" conflict.
        relay_helper_uavs = {
            int(helper_uid)
            for helper_uid, _anchor
            in self.relay_assignments.values()
        }

        blocked_uavs = (
            set(
                self.search_assignments.values()
            )
            | relay_helper_uavs
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

        age_kwargs={}
        if self._search_recovery is not None:
            # StalePositionTracker interpolates consecutive CONTROL snapshots.
            age_kwargs['position_age_bound_seconds']=(
                max(0,self.runtime.current_step-self._last_observed_step)*self.runtime.dt
                +self.profile.stale_age_fraction*self.profile.high_level_interval_steps*self.runtime.dt)
        allocation_kwargs = {}
        if extended_catalog:
            # Candidate count is independent of actual concurrent capacity.
            # This guard only adds invalid actions; all existing hard masks
            # and Search/Relay/Safety reservations remain in force.
            allocation_kwargs['candidate_guard'] = lambda graph, assigned: np.full(
                graph.action_mask.shape, len(assigned) >= free_slots, dtype=bool)
        decisions = self.allocator.allocate(
            slots,
            position_overrides=stale_positions,
            context_subtasks=self._native_search_context(slots),
            **age_kwargs,
            **allocation_kwargs,
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

            self._apply_uav_state(
                decision.uav_id,
                "target_search",
                decision.total_time,
            )

        self._publish_search_uavs()

    def _refresh_relay(
        self,
        stale_positions,
    ) -> None:
        task = self._active_task("relay")

        if (
            task is None
            or self.base_position is None
        ):
            self._relay_probe = None
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
                self._relay_probe = None
                self.relay_assignments.clear()
            elif isinstance(self.relay_builder,ObservedMeshRelayDemandBuilder):
                # Arrival is an observed event. Consume its augmentation
                # opportunity before retargeting can make the helper in-flight.
                self._augment_arrived_relay(task,stale_positions)
                self.relay_builder.refresh_assignments(
                    self.relay_assignments,stale_positions,
                )
                self._refresh_observed_relay_probe(stale_positions)

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
            self._relay_probe = None
            return

        result = self.relay_builder.build(
            stale_positions,
            base_position=self.base_position,
            source_task_id=task.task_id,
            snapshot=snapshot,
        )

        self._allocate_relay_result(result,stale_positions)
        self._refresh_observed_relay_probe(stale_positions)

    def _refresh_observed_relay_probe(self, positions):
        """Bounded observed information gathering after required tasks complete."""
        runtime = getattr(self, 'runtime', None)
        if (getattr(runtime, 'geometry_mode', None) != 'extended'
                or getattr(self, '_search_recovery', None) is None or not runtime.tasks.all_complete()
                or self._active_task('relay') is None):
            self._relay_probe = None
            return
        from .mesh_relay import extended_relay_helper_budget
        budget = extended_relay_helper_budget(len(self.runtime.robots), self.relay_builder.max_demands)
        if (len(self.relay_assignments) >= budget
                or any(int(uid) not in positions or np.linalg.norm(np.asarray(positions[int(uid)])-anchor) > .5
                    for uid,anchor in self.relay_assignments.values())):
            self._relay_probe = None
            return
        snapshot = self.relay_builder.snapshot(positions)
        if snapshot.connectivity_ratio >= self.relay_trigger_threshold:
            self._relay_probe = None
            return
        probe = getattr(self, '_relay_probe', None)
        if probe is not None and int(probe['uid']) not in self.safety.active_uav_ids:
            return
        from .relay_frontier import ObservedRelayFrontier
        planner = getattr(self, '_relay_frontier', None)
        if planner is None:
            fov, sensor_range = self.runtime._sensor_params()
            planner = ObservedRelayFrontier(self._search_recovery,
                comm_range=self.relay_builder.comm_range, sensor_range=sensor_range, fov=fov)
            self._relay_frontier = planner
        hazards = getattr(self.safety, 'hazards', None)
        step = int(self.runtime.current_step)
        clear = None if hazards is None else lambda a,b: not hazards.segment_intersects_warning_buffer(a,b,step=step)
        reserved = set(self.search_assignments.values()) | set(self.safety.active_uav_ids)
        reserved.update(int(uid) for uid,_ in self.relay_assignments.values())
        choice = planner.select(positions, reserved=reserved, segment_filter=clear)
        self._relay_probe = choice
        if choice is not None:
            choice['path'] = [np.asarray(p,dtype=float).copy() for p in choice['path'][1:]]
            self._apply_uav_state(choice['uid'], 'relay', choice['known_path_m'])
            self.runtime._log_event('gppo_relay_observed_probe', dict(robot_id=int(choice['uid']),
                anchor=choice['anchor'].tolist(), potential_unknown_cells=choice['potential_unknown_cells'],
                known_path_m=choice['known_path_m'], forecast_lcc=choice['forecast_lcc'],
                potential_not_guaranteed=True))

    def _relay_probe_action(self, robot, choice):
        rec = self._search_recovery
        position = np.asarray(robot.position,dtype=float)
        while choice['path'] and np.linalg.norm(choice['path'][0]-position) <= .1:
            choice['path'].pop(0)
        if not choice['path']:
            self._relay_probe = None
            return position.copy(), float(robot.heading)
        target = choice['path'][0]
        hazards = getattr(self.safety, 'hazards', None)
        if (not rec.segment_safe(position,target)
                or (hazards is not None and hazards.segment_intersects_warning_buffer(
                    position,target,step=int(self.runtime.current_step)))):
            self._relay_probe = None
            return position.copy(), float(robot.heading)
        feedback = rec.failed.get(int(robot.robot_id),[])
        if any(np.linalg.norm(position-origin)<.5 and np.linalg.norm(target-old)<1e-6 for old,origin in feedback):
            self._relay_probe = None
            return position.copy(), float(robot.heading)
        heading = choice['heading'] if len(choice['path'])==1 else self._heading(robot,target)
        return target.copy(), float(heading)

    def _augment_arrived_relay(self,task,stale_positions):
        budget=int(self.relay_builder.max_demands)
        runtime=getattr(self,"runtime",None)
        if getattr(runtime,"geometry_mode",None)=="extended":
            from .mesh_relay import extended_relay_helper_budget
            budget=extended_relay_helper_budget(len(runtime.robots),budget)
        remaining=budget-len(self.relay_assignments)
        if remaining<=0:return
        if any(int(uid) not in stale_positions or np.linalg.norm(np.asarray(stale_positions[int(uid)])-anchor)>.5
            for uid,anchor in self.relay_assignments.values()):return
        snapshot=self.relay_builder.snapshot(stale_positions,base_position=self.base_position)
        if snapshot.connectivity_ratio>=self.relay_trigger_threshold:return
        result=self.relay_builder.build(stale_positions,base_position=self.base_position,
            source_task_id=task.task_id,snapshot=snapshot)
        self._allocate_relay_result(result,stale_positions,append=True,limit=remaining)

    def _allocate_relay_result(self,result,stale_positions,*,append=False,limit=None):
        # Search reservations remain unavailable to Relay if a
        # future scenario gives a UAV overlapping capabilities.
        reserved = set(
            self.search_assignments.values()
        ) | {int(uid) for uid,_ in self.relay_assignments.values()}

        probe = getattr(self, '_relay_probe', None)
        if probe is not None:
            reserved.add(int(probe['uid']))

        slots = []

        for slot in result.slots:
            if append and int(slot.source_entity_id) in self.relay_assignments:continue
            if limit is not None and len(slots)>=limit:break
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

        guard_kwargs = {}
        if isinstance(self.relay_builder,ObservedMeshRelayDemandBuilder):
            guard_kwargs["candidate_guard"] = self.relay_builder.candidate_mask
        decisions = self.allocator.allocate(
            slots,
            position_overrides=stale_positions,
            **guard_kwargs,
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

        if not append:
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

            self._apply_uav_state(
                decision.uav_id,
                "relay",
                decision.total_time,
            )

        # A newly triggered Relay event starts with a fresh
        # counterfactual stability window.
        self.relay_release_gate.reset_stability()

    def _release_proven_unreachable_search(self,stale_positions):
        released=[]
        if getattr(self.runtime,'geometry_mode',None)!='extended':return released
        from .observed_reachability import observed_search_impossibility
        age=(max(0,self.runtime.current_step-self._last_observed_step)*self.runtime.dt
            +self.profile.stale_age_fraction*self.profile.high_level_interval_steps*self.runtime.dt)
        robots={int(r.robot_id):r for r in self.runtime.robots}
        for heat,uid in list(self.search_assignments.items()):
            if uid not in stale_positions or uid not in robots or heat not in self.heat_by_id:continue
            reason=observed_search_impossibility(self.runtime,uid,stale_positions[uid],
                self.heat_by_id[heat].position,age)
            if reason is None:continue
            self.search_assignments.pop(heat)
            self.search_dwell_by_heat[heat]=0
            released.append((int(heat),int(uid)))
            log=getattr(self.runtime,'_log_event',None)
            if callable(log):log('gppo_search_assignment_infeasible',{'heat_id':int(heat),
                'robot_id':int(uid),**reason})
        return released

    def _release_unsupported_search_routes(self,stale_positions):
        if getattr(self.runtime,'geometry_mode',None)!='extended' or not self.search_assignments:return []
        from .runtime_graph import RuntimeGraphBuilder
        from .task_graph import TaskGraph
        from .route_support import observed_support_yields
        uavs=RuntimeGraphBuilder(self.runtime,position_overrides=stale_positions).build().graph.uav_states
        tasks=[Subtask(100000+int(heat),TaskType.TARGET_SEARCH,5,
            position=tuple(self.heat_by_id[heat].position),processing_time=float(self.search_service_steps))
            for heat in sorted(self.search_assignments) if heat in self.heat_by_id]
        matrix=self.allocator._travel_time_matrix(uavs,tasks,self._marvel_agents)
        graph=TaskGraph(uavs,tasks,current_time=self.runtime.current_step*self.runtime.dt,travel_time_matrix=matrix)
        age=(max(0,self.runtime.current_step-self._last_observed_step)*self.runtime.dt
            +self.profile.stale_age_fraction*self.profile.high_level_interval_steps*self.runtime.dt)
        self.allocator._apply_deadline_mask(graph,age)
        self.allocator._apply_observed_reachability_mask(graph,age)
        active={100000+int(heat):int(uid) for heat,uid in self.search_assignments.items()}
        yields=observed_support_yields(self.runtime,graph,self.allocator.last_travel_estimates,active)
        for reason in yields:
            heat=int(reason['task_id'])-100000;uid=int(reason['deferred_uav_id'])
            self.search_assignments.pop(heat,None);self.search_dwell_by_heat[heat]=0
            self.runtime._log_event('gppo_search_route_support_yield',{'heat_id':heat,'robot_id':uid,**reason})
        return yields

    def _high_level_update(self) -> None:
        stale = self.tracker.positions(
            self.profile.stale_age_fraction
        )

        self.last_event_assignments = []

        # Frozen maybe_activate() is evaluated once per
        # high-level mission step and has no time gate.
        self._maybe_activate_search()

        # Sensor detection from the preceding physics ticks is
        # consumed at this GPPO decision boundary.
        self._sync_search_completions()
        self._release_proven_unreachable_search(stale)
        self._release_unsupported_search_routes(stale)

        # Release anything that no longer holds an assignment, and publish
        # the currently assigned Search UAV set, before the allocation
        # graphs are built.
        self._reconcile_uav_state()

        self._refresh_search(stale)
        self._refresh_relay(stale)

        # Newly created assignments change UAV availability for the next
        # graph build.
        self._reconcile_uav_state()

    def _override_heading(self, robot, goal) -> float:
        """Heading commanded for a Search/Relay override.

        Frozen recomputes the heading bin from the final waypoint direction
        after collision resolution and hazard filtering, discarding the
        PolicyNet heading index:

            angle = degrees(atan2(dy, dx)) % 360
            index = int(floor(angle / 360 * NUM_ANGLES_BIN)) % NUM_ANGLES_BIN
            desired_heading = index * (360 / NUM_ANGLES_BIN)

        ``marvel_native`` reproduces that exactly; the continuous atan2
        angle is off by up to half a bin (e.g. 296.565 -> 290.0) and shifts
        every interpolated sensing heading downstream.  ``extended`` keeps
        its historical continuous heading.
        """

        goal = np.asarray(goal, dtype=float)

        delta = goal - np.asarray(robot.position, dtype=float)

        if float(np.linalg.norm(delta)) < 1e-12:
            return float(robot.heading)

        angle = float(
            np.degrees(np.arctan2(delta[1], delta[0])) % 360.0
        )

        frame = getattr(self.runtime.obstacles, "frame", None)

        if frame is None or not frame.is_native:
            return angle

        index = int(
            np.floor(angle / 360.0 * FROZEN_NUM_ANGLES_BIN)
        ) % FROZEN_NUM_ANGLES_BIN

        return float(index) * (360.0 / FROZEN_NUM_ANGLES_BIN)

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

        # Target-found short circuit is checked before dwell.
        detected_heat_ids = (
            self._sync_search_completions()
        )

        search_events = [
            {
                "type":
                    "gppo_search_heat_serviced",
                "heat_id":
                    int(heat_id),
                "reason":
                    "target_found",
                "step":
                    int(
                        self.runtime.current_step
                    ),
            }
            for heat_id in detected_heat_ids
        ]

        search_events.extend(
            self._update_search_service()
        )

        escaped = self.safety.post_motion_update(
            self,
            int(self.runtime.current_step),
        )

        safety_events = [
            {
                "type": "gppo_safety_escape",
                "robot_id": int(uid),
                "step": int(
                    self.runtime.current_step
                ),
            }
            for uid in escaped
        ]

        # Completion releases the UAV in the frozen coordinator, so the
        # published Search set must shrink before the next detection.
        self._reconcile_uav_state()

        return (
            search_events
            + safety_events
        )

    @staticmethod
    def _make_search_recovery(runtime):
        frame = runtime.obstacles.frame
        if frame.is_native:
            return None
        from utils.dynamics_models import KinematicSimple
        models = [runtime.dynamics, *runtime.dynamics_by_type.values()]
        if not all(isinstance(model, KinematicSimple) for model in models):
            raise ValueError(
                "Extended observed Search recovery requires KinematicSimple; "
                "other dynamics need their actual trajectory validation."
            )
        return ObservedSearchRecovery(frame)

    def _observed_relay_motion_impossible(self,uid,position,goal):
        from .observed_reachability import observed_motion_impossibility
        age=(max(0,self.runtime.current_step-getattr(self,'_last_observed_step',self.runtime.current_step))*self.runtime.dt
            +self.profile.stale_age_fraction*self.profile.high_level_interval_steps*self.runtime.dt)
        return observed_motion_impossibility(self.runtime,uid,position,goal,age) is not None

    def _observed_relay_anchor_candidates(self):
        rec=self._search_recovery
        if rec is None or not rec._authoritative_areal_grid:return []
        stamp=int(self.runtime.current_step)
        if getattr(self,'_relay_anchor_stamp',None)!=stamp:
            origin=np.asarray(rec.frame.origin,dtype=float)
            candidates=[]
            for (col,row),label in sorted(rec.cells.items()):
                if label!=0:continue
                point=origin+(np.array([col,row],dtype=float)+.5)*rec.frame.cell_size
                if rec.segment_safe(point,point):candidates.append(point)
            self._relay_anchor_points=candidates;self._relay_anchor_stamp=stamp
        return [point for point in self._relay_anchor_points
            if not self.safety.hazards.segment_intersects_warning_buffer(point,point,step=stamp)]

    def _relay_action(self,robot,anchor,agent,route,base_action):
        if self._search_recovery is None:
            if route is None:return base_action
            goal,_index=route
            return goal,self._override_heading(robot,goal)
        distance=float(np.linalg.norm(np.asarray(robot.position)-np.asarray(anchor)))
        if distance<=.1:return np.asarray(robot.position).copy(),float(robot.heading)
        if distance<=6.:route=(np.asarray(anchor,dtype=float),None)
        return self._search_action(robot,anchor,agent,route,base_action,
                                   allow_observed_frontier=True,allow_observed_grid_route=True)

    def before_physics_tick(self, actions):
        """Extended public warnings preempt motion at the physical clock."""
        if self._search_recovery is None:
            return actions
        self.safety.pre_motion_update(self, int(self.runtime.current_step))
        result=list(actions)
        for index,robot in enumerate(self.runtime.robots):
            if int(robot.robot_id) in self.safety.active_uav_ids:
                result[index]=self.safety.select_action(robot, step=int(self.runtime.current_step))
        return result

    def _peripheral_search_target(self, robot, anchor, agent):
        views = getattr(self, '_search_viewpoints', None)
        if views is None:
            return None
        uid = int(robot.robot_id)
        heat = next((int(h) for h, assigned in getattr(self, 'search_assignments', {}).items()
                     if int(assigned) == uid), None)
        targets = self._peripheral_search_targets
        if heat is None:
            targets.pop(uid, None)
            return None
        hazards = getattr(getattr(self, 'safety', None), 'hazards', None)
        step = int(self.runtime.current_step)
        risk = None if hazards is None else hazards.risk_at(anchor, step=step)
        # The public warning must exclude the WHOLE original service disk.
        # This changes execution goals only; Search completion still needs Find.
        if risk is None or risk.clearance + self.search_reach_radius > hazards.warning_margin:
            targets.pop(uid, None)
            return None
        segment_filter = lambda a, b: not hazards.segment_intersects_warning_buffer(a, b, step=step)
        cached = targets.get(uid)
        if cached is not None and cached[0] == heat and np.linalg.norm(robot.position-cached[1]) > .1:
            from .extended_travel import extended_observed_next_hop
            hop = extended_observed_next_hop(uid, robot.position, cached[1], agent,
                self._search_recovery, segment_filter=segment_filter)
            if hop is not None:
                return cached[1].copy()
        targets.pop(uid, None)
        choice = views.select(uid, heat, anchor, agent, self._search_recovery, segment_filter=segment_filter)
        if choice is None:
            return None
        targets[uid] = (heat, choice.position.copy())
        return choice.position.copy()

    def observe(self, observations):
        """Consume only sensor-visible physical labels and own motion feedback."""
        inference = getattr(getattr(self, 'allocator', None), 'adapter', None)
        candidate_observe = getattr(inference, 'observe', None)
        if callable(candidate_observe):
            control_index = int(self.runtime.current_step) // int(self.profile.high_level_interval_steps)
            if control_index != getattr(self, '_last_heading_observation_control_index', None):
                candidate_observe(observations)
                self._last_heading_observation_control_index = control_index
        if self._search_recovery is not None:
            self._search_recovery.observe(observations)
        views = getattr(self, '_search_viewpoints', None)
        if views is not None:
            views.observe(observations, self.search_assignments)

    def _search_action(self, robot, anchor, agent, route, base_action,
                       service_scan_heading=None, route_goal_radius=0.,
                       allow_observed_frontier=False,allow_observed_grid_route=False):
        peripheral_heat = None
        if self._search_recovery is not None and service_scan_heading is None:
            support = self._peripheral_search_target(robot, anchor, agent)
            if support is not None:
                peripheral_heat = np.asarray(anchor, dtype=float).copy()
                anchor = support
                route = None
                route_goal_radius = 0.
        if self._search_recovery is None:
            # Preserve frozen native routing and exploration heading exactly.
            if route is None:
                return base_action
            goal, _node_index = route
        else:
            hazards = getattr(getattr(self, "safety", None), "hazards", None)
            segment_filter = None if hazards is None else lambda a, b: not hazards.segment_intersects_warning_buffer(
                a, b, step=int(self.runtime.current_step))
            progress_target=None
            if service_scan_heading is None:
                fallback_preferred=None if route is None else route[0]
                route=None
                from .extended_travel import extended_observed_next_hop
                rec=self._search_recovery
                origin=np.asarray(robot.position,dtype=float)[:2]
                peers=rec.nearby_positions.get(int(robot.robot_id),[])
                def route_clear(a,b):
                    if segment_filter is not None and not segment_filter(a,b):return False
                    a=np.asarray(a,dtype=float)[:2];b=np.asarray(b,dtype=float)[:2];delta=b-a;length2=float(np.dot(delta,delta))
                    for peer in peers:
                        t=float(np.clip(np.dot(peer-a,delta)/length2,0.,1.)) if length2>1e-20 else 0.
                        if np.linalg.norm(a+t*delta-peer)<min(.5,float(np.linalg.norm(origin-peer)))-1e-9:return False
                    return True
                if np.linalg.norm(origin-np.asarray(anchor))<=6. and rec.segment_safe(origin,anchor) and route_clear(origin,anchor):
                    route=(np.asarray(anchor,dtype=float),None);progress_target=route[0]
                else:
                    details={}
                    certified=extended_observed_next_hop(robot.robot_id,origin,anchor,agent,rec,segment_filter=segment_filter,route_details=details,goal_radius=route_goal_radius)
                    if certified is None and allow_observed_grid_route:
                        from .relay_frontier import ObservedRelayFrontier
                        planner=getattr(self,'_relay_motion_routes',None)
                        if planner is None or planner.recovery is not rec:
                            planner=ObservedRelayFrontier(rec);self._relay_motion_routes=planner
                        path=planner.known_route(robot.robot_id,anchor,segment_filter=segment_filter)
                        if path is not None:
                            points=[np.asarray(p,dtype=float) for p in path[1:] if np.linalg.norm(np.asarray(p)-origin)>.1]
                            if points:
                                certified=(points[0],None)
                                details={'route_waypoints':[p.tolist() for p in points]}
                    if certified is None and allow_observed_frontier:
                        from .extended_travel import observed_frontier_next_hop
                        frontier = observed_frontier_next_hop(
                            robot.robot_id, origin, anchor, agent, rec, details,
                            segment_filter=segment_filter,
                        )
                        if frontier is not None:
                            certified, details = frontier
                    if certified is not None:
                        route=certified;progress_target=route[0]
                        # A peer can occupy a coarse lattice node while the
                        # continuous FREE corridor around it remains usable.
                        # Look ahead on the same observed path for progress;
                        # rec.select still certifies every actual move.
                        for point in details.get('route_waypoints',[]):
                            point=np.asarray(point,dtype=float)
                            if np.linalg.norm(point-origin)<=.1:continue
                            if all(np.linalg.norm(point-peer)>=.5-1e-9 for peer in peers):
                                progress_target=point;break
            goal = self._search_recovery.select(
                robot.robot_id, robot.position, anchor, agent,
                None if route is None else route[0], base_action[0],
                base_heading=base_action[1], segment_filter=segment_filter, progress_target=progress_target,
                prefer_base=(service_scan_heading is None and progress_target is None),
                fallback_preferred=(fallback_preferred if service_scan_heading is None else None),
                max_anchor_distance=(self.search_reach_radius
                                     if service_scan_heading is not None else None),
            )
        if self._search_recovery is not None and service_scan_heading is not None:
            if np.linalg.norm(np.asarray(goal)[:2] - np.asarray(robot.position)[:2]) <= .1:
                # A service pose may have only outward graph/base directions.
                # Retry the public heat direction through the same observed
                # prefix, failed-motion, peer and unchanged service-radius gate.
                goal = self._search_recovery.select(
                    robot.robot_id, robot.position, anchor, agent, anchor, base_action[0],
                    base_heading=base_action[1], max_anchor_distance=self.search_reach_radius,
                    segment_filter=segment_filter,
                )
            return (goal, float(service_scan_heading))
        heading_goal = goal
        if (self._search_recovery is not None
                and int(robot.robot_id) in self._search_recovery.sensing_uids):
            # KinematicSimple supports independent waypoint and desired heading.
            # A short legal movement lets FoV turn toward the public Search prior;
            # holding position would provide no yaw in this motion model.
            heading_goal = goal if progress_target is not None else anchor
        heading = self._override_heading(robot, heading_goal)
        if self._search_recovery is not None and progress_target is None:
            heading = self._search_recovery.prefix_headings.get(
                int(robot.robot_id), heading
            )
        if peripheral_heat is not None and progress_target is not None:
            heading = self._override_heading(robot, peripheral_heat)
        return (goal, heading)

    def _service_scan_heading(self, heat_id, robot, anchor):
        # Advance only after real movement has achieved the requested heading.
        for name in ("_search_scan_bearings", "_search_scan_phases", "_search_scan_owners"):
            if not hasattr(self, name):
                setattr(self, name, {})
        bearings = self._search_scan_bearings
        phases = self._search_scan_phases
        owners = self._search_scan_owners
        entering = (heat_id not in bearings
                    or owners.get(heat_id) != int(robot.robot_id)
                    or self.search_dwell_by_heat.get(heat_id, 0) == 0)
        if entering:
            bearings[heat_id] = self._override_heading(robot, anchor)
            phases[heat_id] = 0
            owners[heat_id] = int(robot.robot_id)
        sensors = getattr(self.runtime, "sensor_by_type", {})
        sensor = sensors.get(getattr(robot, "robot_type", None),
                             getattr(self.runtime, "sensor", None))
        fov = float(getattr(sensor, "fov", 120.))
        if not np.isfinite(fov) or fov <= 0:
            fov = 120.
        sectors = max(1, int(np.ceil(360. / min(360., fov))))
        desired = (bearings[heat_id] + 360. / sectors * phases[heat_id]) % 360.
        error = abs((float(robot.heading) - desired + 180.) % 360. - 180.)
        if not entering and error <= 1e-6:
            phases[heat_id] = (phases[heat_id] + 1) % sectors
            desired = (bearings[heat_id] + 360. / sectors * phases[heat_id]) % 360.
        return desired

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

        if self._search_recovery is not None:
            self._search_recovery.sync_assignments(self.search_assignments)

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

            # Frozen v9 Search routing:
            # one hop on the currently observed MARVEL graph.
            # If the best observed destination is the current
            # node, retain the original MARVEL exploration
            # action so the graph can expand.
            agent = self._marvel_agents.get(
                int(uav_id)
            )

            anchor = np.asarray(
                point.position,
                dtype=float,
            )

            distance_to_anchor = float(
                np.linalg.norm(
                    np.asarray(
                        robot.position,
                        dtype=float,
                    )[:2]
                    - anchor[:2]
                )
            )

            if (
                distance_to_anchor
                <=
                self.search_reach_radius
                + 1e-9
            ):
                if self._search_recovery is None:
                    # Frozen native hold_radius behavior.
                    goal = np.asarray(robot.position, dtype=float).copy()
                    actions[index] = (goal, self._override_heading(robot, anchor))
                else:
                    actions[index] = self._search_action(
                        robot, anchor, agent, None, actions[index],
                        service_scan_heading=self._service_scan_heading(
                            heat_id, robot, anchor),
                    )

            else:
                for name in ("_search_scan_bearings", "_search_scan_phases", "_search_scan_owners"):
                    getattr(self, name, {}).pop(heat_id, None)
                route = observed_next_hop(
                    agent,
                    point.position,
                )

                actions[index] = self._search_action(
                    robot, anchor, agent, route, actions[index],
                    route_goal_radius=self.search_reach_radius,
                    allow_observed_frontier=True,
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

            # Frozen v9 Relay routing uses the same observed
            # MARVEL graph principle.  No known progress =>
            # retain MARVEL exploration action.
            agent = self._marvel_agents.get(
                int(helper_uid)
            )

            route = observed_next_hop(
                agent,
                anchor,
            )

            actions[index] = self._relay_action(
                robot,anchor,agent,route,actions[index],
            )

            used_uavs.add(int(helper_uid))

            self.assignments[
                int(helper_uid)
            ] = (
                "relay",
                int(target_uid),
            )

        probe = getattr(self, '_relay_probe', None)
        if probe is not None:
            uid = int(probe['uid'])
            index = robot_index.get(uid)
            if index is not None and uid not in used_uavs and uid not in self.safety.active_uav_ids:
                actions[index] = self._relay_probe_action(self.runtime.robots[index], probe)
                if getattr(self, '_relay_probe', None) is not None:
                    used_uavs.add(uid)
                    self.assignments[uid] = ('relay', None)

        # Frozen v9 same-waypoint resolution.  `_select_actions` dispatches
        # the per-task waypoint before returning, so the frozen resolver
        # only ever sees waypoints that have already been overridden by
        # Search/Relay/Safety.  Resolving earlier would let a waypoint that
        # is about to be overridden displace the Exploration robot sharing
        # it.  Native only; extended mode keeps the raw PolicyNet output.
        post_selection = self._post_selection_adapter()

        if post_selection is not None:
            actions = post_selection.resolve_actions(actions)

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

        # Frozen recomputes the commanded heading bin from the final
        # waypoint once every override and the hazard guard have run.
        if post_selection is not None:
            actions = post_selection.recompute_headings(actions)

        return actions

    def _native_search_context(self, slots):
        """Frozen full subtask registry for the native Search allocation.

        `TaskManager.build_task_graph()` returns every registered subtask, so
        the frozen actor is always conditioned on the Relay row and on every
        Search row registered so far -- including the one it has just
        finished.  Handing it only the slots being allocated changes which
        UAV the same checkpoint selects.

        Rows keep their `assigned_uav_id` after completion, because frozen's
        `apply_assignment` writes the field and `complete_subtask` does not
        clear it; the assigned-UAV flag is one of the task features.
        """

        if not (
            self.runtime.obstacles.frame.is_native
        ):
            return ()

        if self._native_relay_row is None:
            self._native_relay_row = self._native_relay_subtask()

        for slot in slots:
            heat_id = (
                None
                if slot.source_entity_id is None
                else int(slot.source_entity_id)
            )

            if heat_id is None or heat_id in self._native_search_rows:
                continue

            self._native_search_rows[heat_id] = Subtask(
                subtask_id=int(slot.slot_id),
                task_type=TaskType.TARGET_SEARCH,
                priority=float(slot.priority),
                active=True,
                completed=False,
                position=slot.position,
                processing_time=float(slot.processing_time),
            )

        for heat_id, row in self._native_search_rows.items():
            completed = heat_id in self.serviced_heat_ids

            row.completed = bool(completed)
            row.active = not completed

            assigned = self.search_assignments.get(heat_id)

            if assigned is not None:
                row.assigned_uav_id = int(assigned)

        rows = []

        if self._native_relay_row is not None:
            rows.append(self._native_relay_row)

        # Registration order, matching the frozen subtask list.
        rows.extend(
            self._native_search_rows[heat_id]
            for heat_id in sorted(self._native_search_rows)
        )

        return tuple(rows)

    def _native_relay_subtask(self):
        """The Relay subtask row frozen registers before Search activates."""

        for _target_uid, (helper_uid, anchor) in (
            self.relay_assignments.items()
        ):
            return Subtask(
                subtask_id=NATIVE_RELAY_SUBTASK_ID,
                task_type=TaskType.RELAY,
                priority=NATIVE_RELAY_PRIORITY,
                active=True,
                completed=False,
                position=tuple(
                    float(value) for value in np.asarray(anchor, float)[:2]
                ),
                processing_time=NATIVE_RELAY_PROCESSING_TIME,
                assigned_uav_id=int(helper_uid),
            )

        return None

    def bind_post_selection(self, adapter):
        """Bind the adapter owning the frozen native post-selection."""

        self._post_selection = adapter

    def _post_selection_adapter(self):
        """Adapter owning frozen native post-selection, if it is deferred.

        Returns ``None`` for extended mode, and when the PolicyNet failed
        and the runtime fell back to default actions -- frozen's
        post-selection only exists on the real policy path.
        """

        adapter = self._post_selection

        if adapter is None or not getattr(
            adapter, "_defer_post_selection", False
        ):
            return None

        return adapter
