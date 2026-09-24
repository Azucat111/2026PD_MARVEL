from __future__ import annotations

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


class GPPOEventAllocator:
    """Sequential frozen-GPPO event allocation."""

    def __init__(
        self,
        runtime,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        self.runtime = runtime
        self.adapter = GPPOInferenceAdapter(
            checkpoint_path,
            device=device,
        )

        self._marvel_agents = {}

    def bind_marvel_agents(
        self,
        agents,
    ):
        self._marvel_agents = {
            int(agent.id): agent
            for agent in agents
        }

    def allocate(
        self,
        slots: Iterable[GPPOEventSlot],
        *,
        deterministic: bool = True,
        position_overrides=None,
    ) -> list[GPPOEventAssignment]:

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

        slot_by_id = {
            int(slot.slot_id): slot
            for slot in slots
        }

        assignments: list[GPPOEventAssignment] = []

        current_time = float(
            self.runtime.current_step * self.runtime.dt
        )

        travel_time_matrix = (
            observed_travel_time_matrix(
                uavs,
                subtasks,
                self._marvel_agents,
            )
        )

        for _ in range(len(subtasks)):
            graph = TaskGraph(
                uav_states=uavs,
                subtasks=subtasks,
                current_time=current_time,
                travel_time_matrix=travel_time_matrix,
            )

            # Event-specific hard masks.
            for ti, task in enumerate(graph.subtasks):
                slot = slot_by_id[int(task.subtask_id)]

                for forbidden_uid in slot.forbidden_uav_ids:
                    ui = graph.uav_id_to_index.get(
                        int(forbidden_uid)
                    )

                    if ui is not None:
                        graph.action_mask[ti, ui] = True

            decision = self.adapter.assign(
                graph,
                deterministic=deterministic,
            )

            if decision is None:
                break

            subtask = subtasks[decision.task_index]
            uav = uavs[decision.uav_index]

            slot = slot_by_id[
                int(subtask.subtask_id)
            ]

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
                )
            )

            # Frozen TaskManager.apply_assignment semantics.
            subtask.assigned_uav_id = int(
                uav.uav_id
            )

            uav.available = False
            uav.assigned_task_num += 1
            uav.current_task = TaskType(
                subtask.task_type
            )

        return assignments
