from __future__ import annotations

import numpy as np

from dataclasses import dataclass, replace
from typing import Iterable

from .adapter import GPPOInferenceAdapter
from .runtime_graph import RuntimeGraphBuilder
from .observed_graph import observed_travel_time_matrix
from .task_graph import Subtask, TaskGraph, TaskType


@dataclass(frozen=True)
class GPPOEventSlot:
    slot_id: int
    task_type: TaskType
    priority: float

    position: tuple[float, float] | None = None
    processing_time: float = 0.0

    source_task_id: str | None = None

    # Search: heat_id
    # Relay: target UAV id
    source_entity_id: int | None = None

    # Relay uses this to forbid target UAV -> own relay helper.
    forbidden_uav_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class GPPOEventAssignment:
    slot_id: int
    task_type: TaskType
    uav_id: int

    source_task_id: str | None
    source_entity_id: int | None

    flat_action: int
    value: float

    # Frozen TaskManager.apply_assignment quantities.
    processing_time: float = 0.0
    total_time: float = 0.0


class GPPOEventAllocator:
    """Sequential frozen-GPPO event allocation."""

    def __init__(
        self,
        runtime,
        checkpoint_path: str,
        device: str = "cpu",
        *,
        extended_checkpoint=None,
        extended_checkpoint_sha256=None,
    ):
        self.runtime = runtime
        if extended_checkpoint is None:
            if extended_checkpoint_sha256 is not None:
                raise ValueError('Candidate hash requires explicit extended checkpoint')
            self.adapter = GPPOInferenceAdapter(checkpoint_path, device=device)
        else:
            if getattr(runtime, 'geometry_mode', None) != 'extended':
                raise ValueError('Experimental candidate inference is extended-only')
            from .heading_adapter import ExtendedGPPOInferenceAdapter
            self.adapter = ExtendedGPPOInferenceAdapter(checkpoint_path, extended_checkpoint,
                device=device, expected_sha256=extended_checkpoint_sha256)

        self._marvel_agents = {}

    def bind_marvel_agents(
        self,
        agents,
    ):
        self._marvel_agents = {
            int(agent.id): agent
            for agent in agents
        }

    def _travel_time_matrix(self,uavs,subtasks,agents):
        if getattr(self.runtime,"geometry_mode",None)=="extended":
            from .extended_travel import extended_observed_travel_time_matrix
            controller=getattr(getattr(self.runtime,"high_level_scheduler",None),"_search_recovery",None)
            scheduler = getattr(self.runtime, "high_level_scheduler", None)
            hazards = getattr(getattr(scheduler, "safety", None), "hazards", None)
            step = int(getattr(self.runtime, "current_step", 0))
            segment_filter = None if hazards is None else lambda a, b: not hazards.segment_intersects_warning_buffer(a, b, step=step)
            self.last_travel_estimates={}
            return extended_observed_travel_time_matrix(
                uavs,subtasks,agents,prefix_controller=controller,
                diagnostics=self.last_travel_estimates,segment_filter=segment_filter)
        return observed_travel_time_matrix(uavs,subtasks,agents)

    def _apply_deadline_mask(self,graph,position_age_bound_seconds):
        self.last_deadline_masks=[]
        if (getattr(self.runtime,'geometry_mode',None)!='extended'
                or position_age_bound_seconds is None):return
        profile=self.runtime.config.get('_gppo_profile',{})
        if not all(k in profile for k in ('search_guaranteed_hidden_radius','search_cell_size')):return
        from .search_deadline import apply_search_deadline_bound,certified_constant_search_capability
        from .task_graph import TaskGraph
        from dataclasses import replace
        for ui,uav in enumerate(graph.uav_states):
            capability=certified_constant_search_capability(self.runtime,uav.uav_id)
            if capability is None:continue
            speed,sensor_range=capability
            # One-column view preserves all original row feasibility and uses
            # the actual constant-speed model's bound, not instantaneous state.
            column=TaskGraph([replace(uav,velocity=speed)],graph.subtasks)
            column.action_mask=graph.action_mask[:,ui:ui+1].copy()
            reasons=apply_search_deadline_bound(column,
                (self.runtime.max_steps-self.runtime.current_step)*self.runtime.dt,
                sensor_range,profile['search_guaranteed_hidden_radius'],
                profile['search_cell_size'],position_age_bound_seconds)
            graph.action_mask[:,ui]=column.action_mask[:,0]
            self.last_deadline_masks.extend(reasons)

    def _apply_observed_reachability_mask(self,graph,position_age_bound_seconds):
        self.last_observed_reachability_masks=[]
        from .observed_reachability import observed_search_impossibility
        for ti,task in enumerate(graph.subtasks):
            if task.task_type!=TaskType.TARGET_SEARCH or task.position is None:continue
            for ui,uav in enumerate(graph.uav_states):
                if graph.action_mask[ti,ui]:continue
                reason=observed_search_impossibility(self.runtime,uav.uav_id,uav.position,task.position,
                    position_age_bound_seconds)
                if reason is not None:
                    graph.action_mask[ti,ui]=True
                    self.last_observed_reachability_masks.append({'task_id':int(task.subtask_id),
                        'uav_id':int(uav.uav_id),**reason})

    def _apply_route_support_mask(self,graph):
        from .route_support import apply_observed_search_support
        self.last_route_support_masks=apply_observed_search_support(self.runtime,graph,
            getattr(self,'last_travel_estimates',{}))

    def allocate(
        self,
        slots: Iterable[GPPOEventSlot],
        *,
        deterministic: bool = True,
        position_overrides=None,
        context_subtasks: Iterable[Subtask] = (),
        position_age_bound_seconds=None,
        candidate_guard=None,
    ) -> list[GPPOEventAssignment]:
        """Assign the given slots.

        ``context_subtasks`` reproduces the frozen allocator input.  Frozen's
        ``ExploreSearchCoordinator._allocate_search_subtasks`` always builds
        the *full* task graph (``TaskManager.build_task_graph`` over every
        registered subtask), masks the rows that are not active Search
        subtasks, and rebuilds the graph after each assignment.  Rows that
        cannot be selected still reach the encoder, so presenting only the
        slots being allocated changes the policy's own choice.  An empty
        argument keeps the historical narrow behaviour.
        """

        slots = list(slots)

        if not slots:
            return []

        ids = [int(slot.slot_id) for slot in slots]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "GPPOEventSlot.slot_id values must be unique"
            )

        base_graph = RuntimeGraphBuilder(
            self.runtime,
            position_overrides=position_overrides,
        ).build().graph

        uavs = [
            replace(uav)
            for uav in base_graph.uav_states
        ]

        subtasks = [
            Subtask(
                subtask_id=int(slot.slot_id),
                task_type=TaskType(slot.task_type),
                priority=float(slot.priority),
                active=True,
                completed=False,
                position=slot.position,
                processing_time=float(slot.processing_time),
            )
            for slot in slots
        ]

        registry = [replace(row) for row in context_subtasks]

        if registry:
            # Frozen: every registered subtask is a graph row.  Only active,
            # uncompleted Search rows survive the mask; Relay rows and
            # already-finished Search rows stay in the graph as nodes.
            subtasks = registry

        active_ids = {
            int(task.subtask_id)
            for task in subtasks
            if task.task_type == TaskType.TARGET_SEARCH
            and task.active
            and not task.completed
        }

        slot_by_id = {
            int(slot.slot_id): slot
            for slot in slots
        }

        assignments: list[GPPOEventAssignment] = []

        current_time = float(
            self.runtime.current_step * self.runtime.dt
        )

        travel_time_matrix = (
            self._travel_time_matrix(
                uavs,
                subtasks,
                self._marvel_agents,
            )
        )

        for _ in range(len(subtasks)):
            if registry:
                # Frozen rebuilds the graph after every assignment, so the
                # row just assigned (and every UAV it took) is re-evaluated.
                travel_time_matrix = (
                    self._travel_time_matrix(
                        uavs,
                        subtasks,
                        self._marvel_agents,
                    )
                )

            graph = TaskGraph(
                uav_states=uavs,
                subtasks=subtasks,
                current_time=current_time,
                travel_time_matrix=travel_time_matrix,
            )

            # Event-specific hard masks.
            for ti, task in enumerate(graph.subtasks):
                slot = slot_by_id.get(int(task.subtask_id))

                if slot is None:
                    continue

                for forbidden_uid in slot.forbidden_uav_ids:
                    ui = graph.uav_id_to_index.get(
                        int(forbidden_uid)
                    )

                    if ui is not None:
                        graph.action_mask[ti, ui] = True

            if registry:
                for ti, task in enumerate(graph.subtasks):
                    if int(task.subtask_id) not in active_ids:
                        graph.action_mask[ti, :] = True

                if not (~graph.action_mask).any():
                    break

            self._apply_deadline_mask(graph,position_age_bound_seconds)
            self._apply_observed_reachability_mask(graph,position_age_bound_seconds)
            self._apply_route_support_mask(graph)

            if candidate_guard is not None:
                additional=np.asarray(candidate_guard(graph,assignments),dtype=bool)
                if additional.shape != graph.action_mask.shape:
                    raise ValueError("Candidate guard shape mismatch")
                graph.action_mask |= additional

            decision = self.adapter.assign(
                graph,
                deterministic=deterministic,
            )

            if decision is None:
                break

            subtask = subtasks[decision.task_index]
            uav = uavs[decision.uav_index]

            slot = slot_by_id.get(int(subtask.subtask_id))

            if slot is None:
                # Registry row that is not being allocated this cycle.
                break

            total_time = float(
                graph.edge_features[
                    decision.task_index,
                    decision.uav_index,
                    2,
                ]
            )

            assignments.append(
                GPPOEventAssignment(
                    slot_id=int(subtask.subtask_id),
                    task_type=TaskType(
                        subtask.task_type
                    ),
                    uav_id=int(uav.uav_id),
                    source_task_id=slot.source_task_id,
                    source_entity_id=slot.source_entity_id,
                    flat_action=int(
                        decision.flat_action
                    ),
                    value=float(decision.value),
                    processing_time=float(
                        subtask.processing_time
                    ),
                    total_time=total_time,
                )
            )

            # Frozen TaskManager.apply_assignment semantics.
            subtask.assigned_uav_id = int(
                uav.uav_id
            )

            uav.available = False
            uav.busy_until = float(current_time) + total_time
            uav.assigned_task_num += 1
            uav.current_task = TaskType(
                subtask.task_type
            )

        return assignments
