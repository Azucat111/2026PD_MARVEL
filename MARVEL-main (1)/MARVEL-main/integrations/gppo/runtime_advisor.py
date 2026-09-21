from __future__ import annotations

from dataclasses import dataclass

from .adapter import GPPOInferenceAdapter
from .runtime_graph import RuntimeGraphBuilder


@dataclass(frozen=True)
class GPPORuntimeDecision:
    step: int
    runtime_task_id: str
    task_type: str
    uav_id: int
    subtask_id: int
    flat_action: int
    value: float


class GPPORuntimeAdvisor:
    """Run frozen GPPO on SimulationRuntime without changing UAV actions."""

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

    def decide(self) -> GPPORuntimeDecision | None:
        built = RuntimeGraphBuilder(self.runtime).build()

        decision = self.adapter.assign(
            built.graph,
            deterministic=True,
        )

        if decision is None:
            return None

        task = built.graph.subtasks[decision.task_index]

        return GPPORuntimeDecision(
            step=int(self.runtime.current_step),
            runtime_task_id=built.subtask_to_runtime_id[
                decision.subtask_id
            ],
            task_type=task.task_type.name,
            uav_id=decision.uav_id,
            subtask_id=decision.subtask_id,
            flat_action=decision.flat_action,
            value=decision.value,
        )
