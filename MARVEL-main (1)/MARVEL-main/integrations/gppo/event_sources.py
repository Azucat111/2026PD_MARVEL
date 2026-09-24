from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .event_allocator import GPPOEventSlot
from .task_graph import TaskType


# ======================================================================
# Search
# ======================================================================

@dataclass(frozen=True)
class PublicHeatPoint:
    """Public Search prior.

    Structurally free of hidden truth: there is no field that could carry a
    ``target_index`` or a survivor coordinate into GPPO, the task graph or
    Search routing.  The ``target_index -> heat_id`` association lives in
    the private environment layer (see ``search_scenario.py``).
    """

    heat_id: int
    position: tuple[float, float]
    confidence: float = 1.0
    priority: float = 5.0
    serviced: bool = False


class SearchSlotBuilder:
    def __init__(
        self,
        max_search_uavs: int = 2,
    ):
        self.max_search_uavs = int(max_search_uavs)

        if self.max_search_uavs <= 0:
            raise ValueError(
                "max_search_uavs must be positive"
            )

    def build(
        self,
        heat_points: Iterable[PublicHeatPoint],
        *,
        active_heat_ids: Iterable[int] = (),
        source_task_id: str = "T2_target_search",
    ) -> list[GPPOEventSlot]:

        active = {
            int(value)
            for value in active_heat_ids
        }

        pending = [
            point
            for point in heat_points
            if not point.serviced
            and int(point.heat_id) not in active
        ]

        pending.sort(
            key=lambda point: (
                -float(point.priority),
                int(point.heat_id),
            )
        )

        selected = pending[
            : self.max_search_uavs
        ]

        return [
            GPPOEventSlot(
                slot_id=100000 + int(point.heat_id),
                task_type=TaskType.TARGET_SEARCH,
                priority=float(point.priority),
                position=(
                    float(point.position[0]),
                    float(point.position[1]),
                ),
                source_task_id=source_task_id,
                source_entity_id=int(point.heat_id),
            )
            for point in selected
        ]


# ======================================================================
# Relay
# ======================================================================

@dataclass(frozen=True)
class RelayObservedSnapshot:
    connectivity_ratio: float

    connected_uav_ids: tuple[int, ...]
    isolated_uav_ids: tuple[int, ...]
    active_uav_ids: tuple[int, ...]

    # -1 means not connected to base within max_hops.
    hop_counts: dict[int, int]


@dataclass(frozen=True)
class RelayDemand:
    target_uav_id: int

    target_position: tuple[float, float]
    anchor_position: tuple[float, float]

    feasible_two_hop: bool
    base_distance: float
    current_hops: int


@dataclass(frozen=True)
class RelayDemandResult:
    snapshot: RelayObservedSnapshot
    demands: tuple[RelayDemand, ...]
    slots: tuple[GPPOEventSlot, ...]


class ObservedRelayDemandBuilder:
    """Frozen-v9 observed communication semantics."""

    def __init__(
        self,
        *,
        comm_range: float = 20.0,
        max_hops: int = 2,
        max_demands: int = 2,
    ):
        self.comm_range = float(comm_range)
        self.max_hops = int(max_hops)
        self.max_demands = int(max_demands)

        if self.comm_range <= 0:
            raise ValueError(
                "comm_range must be positive"
            )

        if self.max_hops <= 0:
            raise ValueError(
                "max_hops must be positive"
            )

    def snapshot(
        self,
        positions: dict[int, np.ndarray],
        *,
        base_position,
        exclude_uav_ids: Iterable[int] = (),
    ) -> RelayObservedSnapshot:

        base = np.asarray(
            base_position,
            dtype=float,
        )

        robot_ids = sorted(
            int(uid)
            for uid in positions
        )

        excluded = {
            int(uid)
            for uid in exclude_uav_ids
        }

        robot_positions = {
            uid: np.asarray(
                positions[uid],
                dtype=float,
            )
            for uid in robot_ids
        }

        # Robot nodes followed by the base node.
        points = [
            robot_positions[uid]
            for uid in robot_ids
        ]
        points.append(base)

        points = np.asarray(
            points,
            dtype=float,
        )

        n = len(robot_ids)
        base_index = n

        adjacency = np.zeros(
            (n + 1, n + 1),
            dtype=bool,
        )

        def enabled(index: int) -> bool:
            if index == base_index:
                return True

            uid = robot_ids[index]
            return uid not in excluded

        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if not enabled(i) or not enabled(j):
                    continue

                distance = float(
                    np.linalg.norm(
                        points[i] - points[j]
                    )
                )

                if (
                    distance
                    <= self.comm_range + 1e-9
                ):
                    adjacency[i, j] = True
                    adjacency[j, i] = True

        hop_by_index: dict[int, int] = {}

        queue = deque([
            (base_index, 0)
        ])

        visited = {base_index}

        while queue:
            node, hops = queue.popleft()

            if hops >= self.max_hops:
                continue

            for neighbor in np.flatnonzero(
                adjacency[node]
            ):
                neighbor = int(neighbor)

                if neighbor in visited:
                    continue

                visited.add(neighbor)

                new_hops = hops + 1

                if neighbor < n:
                    hop_by_index[
                        neighbor
                    ] = new_hops

                queue.append(
                    (neighbor, new_hops)
                )

        hop_counts: dict[int, int] = {}

        connected_ids = []
        isolated_ids = []
        active_ids = []

        for index, uid in enumerate(robot_ids):
            if uid in excluded:
                hop_counts[uid] = -1
                continue

            active_ids.append(uid)

            hop = int(
                hop_by_index.get(
                    index,
                    -1,
                )
            )

            hop_counts[uid] = hop

            if (
                hop >= 1
                and hop <= self.max_hops
            ):
                connected_ids.append(uid)
            else:
                isolated_ids.append(uid)

        active_count = len(active_ids)

        connectivity_ratio = (
            float(len(connected_ids))
            / float(active_count)
            if active_count > 0
            else 1.0
        )

        return RelayObservedSnapshot(
            connectivity_ratio=connectivity_ratio,
            connected_uav_ids=tuple(
                connected_ids
            ),
            isolated_uav_ids=tuple(
                isolated_ids
            ),
            active_uav_ids=tuple(
                active_ids
            ),
            hop_counts=hop_counts,
        )

    def bridge_anchor(
        self,
        target_uav_id: int,
        positions: dict[int, np.ndarray],
        *,
        base_position,
        snapshot: RelayObservedSnapshot | None = None,
    ) -> RelayDemand:

        uid = int(target_uav_id)

        if uid not in positions:
            raise KeyError(
                f"Unknown UAV id {uid}"
            )

        base = np.asarray(
            base_position,
            dtype=float,
        )

        target = np.asarray(
            positions[uid],
            dtype=float,
        )

        delta = target - base

        distance = float(
            np.linalg.norm(delta)
        )

        if distance <= 1e-9:
            anchor = base.copy()
            feasible = True

        elif (
            distance
            <= 2.0 * self.comm_range + 1e-9
        ):
            # Exact frozen v9 bridge rule.
            anchor = base + 0.5 * delta
            feasible = True

        else:
            # Frozen code still creates a diagnostic anchor,
            # but coordinator filters this demand before GPPO.
            anchor = (
                base
                + (
                    self.comm_range
                    / distance
                )
                * delta
            )

            feasible = False

        current_hops = -1

        if snapshot is not None:
            current_hops = int(
                snapshot.hop_counts.get(
                    uid,
                    -1,
                )
            )

        return RelayDemand(
            target_uav_id=uid,
            target_position=(
                float(target[0]),
                float(target[1]),
            ),
            anchor_position=(
                float(anchor[0]),
                float(anchor[1]),
            ),
            feasible_two_hop=bool(feasible),
            base_distance=distance,
            current_hops=current_hops,
        )

    def build(
        self,
        positions: dict[int, np.ndarray],
        *,
        base_position,
        source_task_id: str = "T3_relay",
        snapshot: RelayObservedSnapshot | None = None,
    ) -> RelayDemandResult:

        snap = snapshot or self.snapshot(
            positions,
            base_position=base_position,
        )

        demands = [
            self.bridge_anchor(
                uid,
                positions,
                base_position=base_position,
                snapshot=snap,
            )
            for uid in snap.isolated_uav_ids
        ]

        # Exact frozen ordering:
        # feasible first -> farther target first -> lower UAV id.
        demands.sort(
            key=lambda demand: (
                not demand.feasible_two_hop,
                -float(
                    demand.base_distance
                ),
                int(
                    demand.target_uav_id
                ),
            )
        )

        # Frozen CommunicationManager first truncates,
        # RelayCoordinator then filters infeasible demands.
        selected = demands[
            : self.max_demands
        ]

        feasible = [
            demand
            for demand in selected
            if demand.feasible_two_hop
        ]

        slots = [
            GPPOEventSlot(
                slot_id=(
                    200000
                    + int(
                        demand.target_uav_id
                    )
                ),
                task_type=TaskType.RELAY,
                priority=2.0,
                position=demand.anchor_position,
                source_task_id=source_task_id,
                source_entity_id=int(
                    demand.target_uav_id
                ),

                # Frozen v5+ self-relay prohibition.
                forbidden_uav_ids=(
                    int(
                        demand.target_uav_id
                    ),
                ),
            )
            for demand in feasible
        ]

        return RelayDemandResult(
            snapshot=snap,
            demands=tuple(selected),
            slots=tuple(slots),
        )
