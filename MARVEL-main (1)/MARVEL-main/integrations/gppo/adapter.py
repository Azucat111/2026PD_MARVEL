from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import load_frozen_gppo
from .gppo_model import batch_single_task_graph
from .task_graph import TaskGraph


@dataclass(frozen=True)
class GPPOAssignment:
    flat_action: int
    task_index: int
    uav_index: int
    subtask_id: int
    uav_id: int
    value: float


class GPPOInferenceAdapter:
    """Thin inference-only wrapper around the frozen Phase14-v9 GPPO."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        self.model, self.checkpoint = load_frozen_gppo(
            checkpoint_path,
            device=str(self.device),
        )

    @torch.no_grad()
    def assign(
        self,
        graph: TaskGraph,
        deterministic: bool = True,
    ) -> GPPOAssignment | None:
        if graph.num_uavs == 0 or graph.num_subtasks == 0:
            return None

        batch = batch_single_task_graph(
            graph,
            device=self.device,
        )

        out = self.model(**batch)

        if not bool(out["has_valid_action"][0].item()):
            return None

        if deterministic:
            flat_action = int(
                torch.argmax(out["flat_logits"][0]).item()
            )
        else:
            distribution = torch.distributions.Categorical(
                logits=out["flat_logits"][0]
            )
            flat_action = int(distribution.sample().item())

        task_index = flat_action // graph.num_uavs
        uav_index = flat_action % graph.num_uavs

        task = graph.subtasks[task_index]
        uav = graph.uav_states[uav_index]

        return GPPOAssignment(
            flat_action=flat_action,
            task_index=task_index,
            uav_index=uav_index,
            subtask_id=int(task.subtask_id),
            uav_id=int(uav.uav_id),
            value=float(out["value"][0].item()),
        )
