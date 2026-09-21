from __future__ import annotations

from dataclasses import dataclass

from .task_graph import Subtask, TaskGraph, TaskType, UAVState


RUNTIME_TASK_TYPE = {
    "exploration": TaskType.EXPLORATION,
    "target_search": TaskType.TARGET_SEARCH,
    "relay": TaskType.RELAY,
    "safety": TaskType.SAFETY,
}


@dataclass
class RuntimeTaskGraph:
    graph: TaskGraph
    subtask_to_runtime_id: dict[int, str]


class RuntimeGraphBuilder:
    """Convert the team's SimulationRuntime state into the frozen GPPO TaskGraph.

    This first integration version deliberately does not expose true target
    coordinates from TaskManager to GPPO. Target-search positions will later
    come from public heat points.
    """

    def __init__(self, runtime):
        self.runtime = runtime

    def build(self) -> RuntimeTaskGraph:
        uavs = self._build_uavs()
        subtasks, reverse = self._build_subtasks()

        graph = TaskGraph(
            uav_states=uavs,
            subtasks=subtasks,
            current_time=float(self.runtime.current_step * self.runtime.dt),
        )

        return RuntimeTaskGraph(
            graph=graph,
            subtask_to_runtime_id=reverse,
        )

    def _allowed_types_for(self, task_type: TaskType) -> set[str] | None:
        allowed: set[str] = set()
        found = False

        for task in self.runtime.tasks.tasks.values():
            mapped = RUNTIME_TASK_TYPE.get(task.task_type)
            if mapped != task_type:
                continue

            found = True
            allowed.update(task.assigned_robot_types)

        if not found:
            return None

        # Empty means unrestricted in the team's TaskManager semantics.
        return allowed if allowed else set()

    @staticmethod
    def _is_capable(
        robot_type: str,
        allowed: set[str] | None,
        *,
        default_if_missing: bool,
    ) -> bool:
        if allowed is None:
            return default_if_missing

        if not allowed:
            return True

        return robot_type in allowed

    def _build_uavs(self) -> list[UAVState]:
        explore_types = self._allowed_types_for(TaskType.EXPLORATION)
        search_types = self._allowed_types_for(TaskType.TARGET_SEARCH)
        relay_types = self._allowed_types_for(TaskType.RELAY)
        safety_types = self._allowed_types_for(TaskType.SAFETY)

        states: list[UAVState] = []

        for robot in self.runtime.robots:
            states.append(
                UAVState(
                    uav_id=int(robot.robot_id),
                    available=True,
                    busy_until=0.0,
                    utilization=0.0,
                    assigned_task_num=0,
                    alive=True,
                    can_explore=self._is_capable(
                        robot.robot_type,
                        explore_types,
                        default_if_missing=True,
                    ),
                    can_search=self._is_capable(
                        robot.robot_type,
                        search_types,
                        default_if_missing=True,
                    ),
                    can_relay=self._is_capable(
                        robot.robot_type,
                        relay_types,
                        default_if_missing=True,
                    ),
                    # The team's current TaskManager has no explicit SAFETY
                    # task branch. Keep capability available for the frozen
                    # four-task GPPO schema; hard preemption is wired later.
                    can_safety=self._is_capable(
                        robot.robot_type,
                        safety_types,
                        default_if_missing=True,
                    ),
                    position=(
                        float(robot.position[0]),
                        float(robot.position[1]),
                    ),
                    velocity=max(float(robot.velocity), 1e-6),
                    current_task=TaskType.EXPLORATION,
                )
            )

        return states

    def _build_subtasks(self) -> tuple[list[Subtask], dict[int, str]]:
        subtasks: list[Subtask] = []
        reverse: dict[int, str] = {}

        for subtask_id, task in enumerate(
            self.runtime.tasks.tasks.values()
        ):
            task_type = RUNTIME_TASK_TYPE.get(task.task_type)

            if task_type is None:
                continue

            # IMPORTANT:
            # Do not read task.params["targets"] here. In the team's current
            # implementation those are ground-truth target coordinates.
            # The GPPO integration must later use public heat-point priors.
            position = None

            subtasks.append(
                Subtask(
                    subtask_id=subtask_id,
                    task_type=task_type,
                    priority=float(task.priority),
                    active=(task.status == "active"),
                    completed=(task.status == "complete"),
                    position=position,
                    processing_time=0.0,
                )
            )

            reverse[subtask_id] = task.task_id

        return subtasks, reverse
