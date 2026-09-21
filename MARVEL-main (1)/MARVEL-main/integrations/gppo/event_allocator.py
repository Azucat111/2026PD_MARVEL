from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .adapter import GPPOInferenceAdapter
from .runtime_graph import RuntimeGraphBuilder
from .task_graph import Subtask, TaskGraph, TaskType


@dataclass(frozen=True)
class GPPOEventSlot:
    """One allocatable high-level event subtask.

    Search:
        one slot ~= one public heat-point search assignment

    Relay:
        one slot ~= one feasible relay demand
    """

    slot_id: int
    task_type: TaskType
    priority: float
    position: tuple[float, float] | None = None
    processing_time: float = 0.0
    source_task_id: str | None = None


@dataclass(frozen=True)
class GPPOEventAssignment:
    slot_id: int
    task_type: TaskType
    uav_id: int
    source_task_id: str | None
    flat_action: int
    value: float


class GPPOEventAllocator:
    """Sequential event allocator matching frozen Phase14 semantics.

    Each selected subtask becomes assigned and each selected UAV becomes
    unavailable before the next GPPO decision. The graph is rebuilt after
    every assignment.
    """

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

    def allocate(
        self,
        slots: Iterable[GPPOEventSlot],
        *,
        deterministic: bool = True,
    ) -> list[GPPOEventAssignment]:

        slots = list(slots)

        if not slots:
            return []

        ids = [slot.slot_id for slot in slots]
        if len(ids) != len(set(ids)):
            raise ValueError("GPPOEventSlot.slot_id values must be unique")

        # Reuse the already-tested runtime -> UAVState bridge.
        base_graph = RuntimeGraphBuilder(self.runtime).build().graph

        # Work on local copies. Shadow/planning allocation must not mutate
        # SimulationRuntime yet.
        uavs = [replace(uav) for uav in base_graph.uav_states]

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

        # Frozen Search/Relay coordinators make at most one assignment for
        # each active subtask in the event.
        for _ in range(len(subtasks)):

            graph = TaskGraph(
                uav_states=uavs,
                subtasks=subtasks,
                current_time=current_time,
            )

            decision = self.adapter.assign(
                graph,
                deterministic=deterministic,
            )

            if decision is None:
                break

            subtask = subtasks[decision.task_index]
            uav = uavs[decision.uav_index]

            slot = slot_by_id[int(subtask.subtask_id)]

            assignments.append(
                GPPOEventAssignment(
                    slot_id=int(subtask.subtask_id),
                    task_type=TaskType(subtask.task_type),
                    uav_id=int(uav.uav_id),
                    source_task_id=slot.source_task_id,
                    flat_action=int(decision.flat_action),
                    value=float(decision.value),
                )
            )

            # Match frozen TaskManager.apply_assignment semantics locally:
            #
            # 1. this subtask cannot be selected again
            # 2. this UAV cannot be selected again during the same event
            subtask.assigned_uav_id = int(uav.uav_id)

            uav.available = False
            uav.assigned_task_num += 1
            uav.current_task = TaskType(subtask.task_type)

        return assignments
