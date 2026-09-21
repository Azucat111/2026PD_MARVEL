"""GPPO-style high-level UAV-task heterogeneous graph.

Phase 2 implements the graph representation only.  It intentionally contains no
policy network or PPO training code yet.

Core GPPO mapping:
    G = (T, U, L, E)
    T: subtask nodes
    U: UAV nodes
    L: directed precedence edges between subtasks
    E: UAV-subtask feasibility/capability edges

Project extensions are kept explicit in the feature names below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class TaskType(IntEnum):
    EXPLORATION = 0
    TARGET_SEARCH = 1
    RELAY = 2
    SAFETY = 3


TASK_TYPE_NAMES = {
    TaskType.EXPLORATION: "EXPLORATION",
    TaskType.TARGET_SEARCH: "TARGET_SEARCH",
    TaskType.RELAY: "RELAY",
    TaskType.SAFETY: "SAFETY",
}


UAV_FEATURE_NAMES = (
    # GPPO/dynamic-synchronization-facing state
    "available",
    "busy_until",
    "utilization",
    "assigned_task_num",
    "alive",
    # project capability extensions
    "can_explore",
    "can_search",
    "can_relay",
    "can_safety",
    # project spatial state used to derive edge travel time
    "x",
    "y",
    "velocity",
    # current high-level task, normalized to [0, 1]
    "current_task",
)

TASK_FEATURE_NAMES = (
    # GPPO-facing task state
    "priority",
    "active",
    "completed",
    "assigned_uav_count",
    "avg_processing_time",
    "remaining_predecessors",
    # explicit task-type encoding for the four project tasks
    "type_exploration",
    "type_target_search",
    "type_relay",
    "type_safety",
    # project spatial state
    "x",
    "y",
)

EDGE_FEATURE_NAMES = (
    # GPPO paper explicitly uses flight/processing-time information on edges
    "travel_time",
    "processing_time",
    "total_time",
)


@dataclass
class UAVState:
    uav_id: int

    available: bool = True
    busy_until: float = 0.0
    utilization: float = 0.0
    assigned_task_num: int = 0
    alive: bool = True

    can_explore: bool = True
    can_search: bool = True
    can_relay: bool = True
    can_safety: bool = True

    position: Tuple[float, float] = (0.0, 0.0)
    velocity: float = 1.0
    current_task: TaskType = TaskType.EXPLORATION

    def capability_for(self, task_type: TaskType) -> bool:
        task_type = TaskType(task_type)
        if task_type == TaskType.EXPLORATION:
            return bool(self.can_explore)
        if task_type == TaskType.TARGET_SEARCH:
            return bool(self.can_search)
        if task_type == TaskType.RELAY:
            return bool(self.can_relay)
        if task_type == TaskType.SAFETY:
            return bool(self.can_safety)
        return False


@dataclass
class Subtask:
    subtask_id: int
    task_type: TaskType
    priority: float

    active: bool = True
    completed: bool = False

    # Multiple predecessors are supported even though simple experiments may
    # only use one.  This directly supports directed precedence lines L.
    predecessors: Tuple[int, ...] = field(default_factory=tuple)

    position: Optional[Tuple[float, float]] = None
    processing_time: float = 0.0

    # One subtask is assigned once in this Phase-2 allocator scaffold.
    assigned_uav_id: Optional[int] = None

    def __post_init__(self):
        self.task_type = TaskType(self.task_type)
        self.predecessors = tuple(self.predecessors)

    @property
    def assigned_uav_count(self) -> int:
        return 0 if self.assigned_uav_id is None else 1


class TaskGraph:
    """Concrete, tensor-ready representation of G=(T,U,L,E).

    Matrix conventions:
        n_t = number of subtasks
        n_u = number of UAVs

        capability_matrix : [n_t, n_u] bool
            True when UAV k is capable of subtask i.

        edge_features : [n_t, n_u, EDGE_FEATURE_DIM] float32
            travel time, processing time, total time.

        precedence_matrix : [n_t, n_t] bool
            precedence_matrix[p, s] == True means p must precede s.

        action_mask : [n_t, n_u] bool
            True means the action (subtask i, UAV k) is INVALID.
            This convention matches MARVEL's existing masked-attention style.
    """

    def __init__(
        self,
        uav_states: Sequence[UAVState],
        subtasks: Sequence[Subtask],
        current_time: float = 0.0,
        travel_time_matrix: Optional[np.ndarray] = None,
    ):
        self.uav_states = list(uav_states)
        self.subtasks = list(subtasks)
        self.current_time = float(current_time)
        self.travel_time_matrix = None if travel_time_matrix is None else np.asarray(travel_time_matrix, dtype=np.float32)

        self.uav_id_to_index: Dict[int, int] = {
            u.uav_id: i for i, u in enumerate(self.uav_states)
        }
        self.subtask_id_to_index: Dict[int, int] = {
            t.subtask_id: i for i, t in enumerate(self.subtasks)
        }

        if len(self.uav_id_to_index) != len(self.uav_states):
            raise ValueError("Duplicate UAV IDs are not allowed.")
        if len(self.subtask_id_to_index) != len(self.subtasks):
            raise ValueError("Duplicate subtask IDs are not allowed.")

        if self.travel_time_matrix is not None and self.travel_time_matrix.shape != (len(self.subtasks), len(self.uav_states)):
            raise ValueError("travel_time_matrix must have shape [num_subtasks, num_uavs]")

        self.uav_features = self._build_uav_features()
        self.task_features = self._build_task_features()
        self.capability_matrix = self._build_capability_matrix()
        self.edge_features = self._build_edge_features()
        self.precedence_matrix = self._build_precedence_matrix()
        self.action_mask = self._build_action_mask()

        # Preserve raw engineering quantities for baselines/makespan while
        # presenting normalized features to the neural GPPO model.
        (
            self.model_uav_features,
            self.model_task_features,
            self.model_edge_features,
        ) = self._build_model_features()

        self.validate()

    @property
    def num_uavs(self) -> int:
        return len(self.uav_states)

    @property
    def num_subtasks(self) -> int:
        return len(self.subtasks)

    @property
    def num_actions(self) -> int:
        return self.num_subtasks * self.num_uavs

    @property
    def valid_action_pairs(self) -> List[Tuple[int, int]]:
        """Return valid pairs as (subtask_id, uav_id)."""
        pairs: List[Tuple[int, int]] = []
        for ti, task in enumerate(self.subtasks):
            for ui, uav in enumerate(self.uav_states):
                if not self.action_mask[ti, ui]:
                    pairs.append((task.subtask_id, uav.uav_id))
        return pairs

    def _build_uav_features(self) -> np.ndarray:
        feats = []
        denom_task = max(len(TaskType) - 1, 1)
        for u in self.uav_states:
            x, y = u.position
            feats.append(
                [
                    float(u.available),
                    float(u.busy_until),
                    float(u.utilization),
                    float(u.assigned_task_num),
                    float(u.alive),
                    float(u.can_explore),
                    float(u.can_search),
                    float(u.can_relay),
                    float(u.can_safety),
                    float(x),
                    float(y),
                    float(u.velocity),
                    float(int(TaskType(u.current_task)) / denom_task),
                ]
            )
        if not feats:
            return np.zeros((0, len(UAV_FEATURE_NAMES)), dtype=np.float32)
        return np.asarray(feats, dtype=np.float32)

    def _build_task_features(self) -> np.ndarray:
        completed_by_id = {t.subtask_id: bool(t.completed) for t in self.subtasks}

        feats = []
        for task in self.subtasks:
            if task.position is None:
                x, y = 0.0, 0.0
            else:
                x, y = task.position

            remaining_predecessors = sum(
                1 for pred in task.predecessors if not completed_by_id.get(pred, False)
            )

            one_hot = [0.0] * len(TaskType)
            one_hot[int(task.task_type)] = 1.0

            feats.append(
                [
                    float(task.priority),
                    float(task.active),
                    float(task.completed),
                    float(task.assigned_uav_count),
                    float(task.processing_time),
                    float(remaining_predecessors),
                    *one_hot,
                    float(x),
                    float(y),
                ]
            )

        if not feats:
            return np.zeros((0, len(TASK_FEATURE_NAMES)), dtype=np.float32)
        return np.asarray(feats, dtype=np.float32)

    def _build_capability_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.num_subtasks, self.num_uavs), dtype=bool)
        for ti, task in enumerate(self.subtasks):
            for ui, uav in enumerate(self.uav_states):
                matrix[ti, ui] = uav.capability_for(task.task_type)
        return matrix

    def _build_edge_features(self) -> np.ndarray:
        edge = np.zeros(
            (self.num_subtasks, self.num_uavs, len(EDGE_FEATURE_NAMES)),
            dtype=np.float32,
        )

        for ti, task in enumerate(self.subtasks):
            for ui, uav in enumerate(self.uav_states):
                if self.travel_time_matrix is not None:
                    travel_time = float(self.travel_time_matrix[ti, ui])
                elif task.position is None:
                    travel_time = 0.0
                else:
                    dx = float(task.position[0]) - float(uav.position[0])
                    dy = float(task.position[1]) - float(uav.position[1])
                    distance = float(np.hypot(dx, dy))
                    speed = max(float(uav.velocity), 1e-6)
                    travel_time = distance / speed

                processing_time = max(float(task.processing_time), 0.0)
                total_time = travel_time + processing_time
                edge[ti, ui] = [travel_time, processing_time, total_time]

        return edge

    def _build_model_features(self):
        """Normalize continuous GPPO inputs without changing baseline semantics.

        Raw ``uav_features/task_features/edge_features`` stay available for
        deterministic baselines and schedule calculations.  Only ``to_torch``
        exposes the normalized copies to the neural policy.
        """
        u = self.uav_features.astype(np.float32, copy=True)
        t = self.task_features.astype(np.float32, copy=True)
        e = self.edge_features.astype(np.float32, copy=True)

        # Shared translation/scale for world x,y keeps relative geometry while
        # avoiding map-size-dependent magnitudes.
        coords = []
        if len(u):
            coords.append(u[:, [9, 10]])
        if len(t):
            coords.append(t[:, [10, 11]])
        if coords:
            all_xy = np.concatenate(coords, axis=0)
            center = all_xy.mean(axis=0)
            span = max(float(np.max(np.ptp(all_xy, axis=0))), 1.0)
            if len(u):
                u[:, [9, 10]] = (u[:, [9, 10]] - center) / span
            if len(t):
                t[:, [10, 11]] = (t[:, [10, 11]] - center) / span

        # Time/count magnitudes.  300 steps is the project evaluation horizon.
        if len(u):
            u[:, 1] = np.clip((u[:, 1] - float(self.current_time)) / 300.0, -1.0, 1.0)
            u[:, 3] = u[:, 3] / max(float(self.num_subtasks), 1.0)
            vmax = max(float(np.max(np.abs(u[:, 11]))), 1.0)
            u[:, 11] = u[:, 11] / vmax

        if len(t):
            pmax = max(float(np.max(np.abs(t[:, 0]))), 1.0)
            t[:, 0] = t[:, 0] / pmax
            t[:, 4] = t[:, 4] / 300.0
            t[:, 5] = t[:, 5] / max(float(self.num_subtasks), 1.0)

        if e.size:
            escale = max(float(np.max(np.abs(e[..., 2]))), 1.0)
            e = e / escale

        return u, t, e

    def _build_precedence_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.num_subtasks, self.num_subtasks), dtype=bool)

        for successor_index, task in enumerate(self.subtasks):
            for predecessor_id in task.predecessors:
                if predecessor_id not in self.subtask_id_to_index:
                    raise ValueError(
                        f"Subtask {task.subtask_id} references missing predecessor "
                        f"{predecessor_id}."
                    )
                predecessor_index = self.subtask_id_to_index[predecessor_id]
                matrix[predecessor_index, successor_index] = True

        return matrix

    def _task_is_executable(self, task: Subtask) -> bool:
        if not task.active or task.completed:
            return False

        # Once assigned, the same subtask may not be selected again.
        if task.assigned_uav_id is not None:
            return False

        for predecessor_id in task.predecessors:
            predecessor = self.subtasks[self.subtask_id_to_index[predecessor_id]]
            if not predecessor.completed:
                return False

        return True

    def _uav_is_available(self, uav: UAVState) -> bool:
        return (
            bool(uav.alive)
            and bool(uav.available)
            and float(uav.busy_until) <= self.current_time
        )

    def _build_action_mask(self) -> np.ndarray:
        # True = invalid, matching MARVEL's mask convention.
        mask = np.ones((self.num_subtasks, self.num_uavs), dtype=bool)

        for ti, task in enumerate(self.subtasks):
            task_ok = self._task_is_executable(task)

            for ui, uav in enumerate(self.uav_states):
                uav_ok = self._uav_is_available(uav)
                capable = bool(self.capability_matrix[ti, ui])

                mask[ti, ui] = not (task_ok and uav_ok and capable)

        return mask

    def flatten_action(self, subtask_index: int, uav_index: int) -> int:
        if not (0 <= subtask_index < self.num_subtasks):
            raise IndexError("subtask_index out of range")
        if not (0 <= uav_index < self.num_uavs):
            raise IndexError("uav_index out of range")
        return subtask_index * self.num_uavs + uav_index

    def unflatten_action(self, action_index: int) -> Tuple[int, int]:
        if self.num_uavs == 0:
            raise ValueError("Cannot decode actions with zero UAVs.")
        if not (0 <= action_index < self.num_actions):
            raise IndexError("action_index out of range")
        return divmod(action_index, self.num_uavs)

    def action_ids(self, action_index: int) -> Tuple[int, int]:
        """Decode flat action to (subtask_id, uav_id)."""
        ti, ui = self.unflatten_action(action_index)
        return self.subtasks[ti].subtask_id, self.uav_states[ui].uav_id

    def flat_action_mask(self) -> np.ndarray:
        return self.action_mask.reshape(-1)

    def validate(self) -> None:
        expected_uav = (self.num_uavs, len(UAV_FEATURE_NAMES))
        expected_task = (self.num_subtasks, len(TASK_FEATURE_NAMES))
        expected_cap = (self.num_subtasks, self.num_uavs)
        expected_edge = (
            self.num_subtasks,
            self.num_uavs,
            len(EDGE_FEATURE_NAMES),
        )

        if self.uav_features.shape != expected_uav:
            raise ValueError(
                f"uav_features shape {self.uav_features.shape} != {expected_uav}"
            )
        if self.task_features.shape != expected_task:
            raise ValueError(
                f"task_features shape {self.task_features.shape} != {expected_task}"
            )
        if self.capability_matrix.shape != expected_cap:
            raise ValueError("capability_matrix has wrong shape")
        if self.edge_features.shape != expected_edge:
            raise ValueError("edge_features has wrong shape")
        if self.precedence_matrix.shape != (
            self.num_subtasks,
            self.num_subtasks,
        ):
            raise ValueError("precedence_matrix has wrong shape")
        if self.action_mask.shape != expected_cap:
            raise ValueError("action_mask has wrong shape")

        if not np.isfinite(self.uav_features).all():
            raise ValueError("uav_features contains non-finite values")
        if not np.isfinite(self.task_features).all():
            raise ValueError("task_features contains non-finite values")
        if not np.isfinite(self.edge_features).all():
            raise ValueError("edge_features contains non-finite values")
        if not np.isfinite(self.model_uav_features).all():
            raise ValueError("normalized UAV features contain non-finite values")
        if not np.isfinite(self.model_task_features).all():
            raise ValueError("normalized task features contain non-finite values")
        if not np.isfinite(self.model_edge_features).all():
            raise ValueError("normalized edge features contain non-finite values")

    def to_torch(self, device=None) -> Dict[str, "object"]:
        """Convert graph arrays to torch tensors for the future AHGNN/GPPO model."""
        import torch

        return {
            "uav_features": torch.as_tensor(
                self.model_uav_features, dtype=torch.float32, device=device
            ),
            "task_features": torch.as_tensor(
                self.model_task_features, dtype=torch.float32, device=device
            ),
            "capability_matrix": torch.as_tensor(
                self.capability_matrix, dtype=torch.bool, device=device
            ),
            "edge_features": torch.as_tensor(
                self.model_edge_features, dtype=torch.float32, device=device
            ),
            "precedence_matrix": torch.as_tensor(
                self.precedence_matrix, dtype=torch.bool, device=device
            ),
            "action_mask": torch.as_tensor(
                self.action_mask, dtype=torch.bool, device=device
            ),
        }
