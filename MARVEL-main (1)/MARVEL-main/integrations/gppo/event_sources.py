from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import inf
from typing import Iterable

import numpy as np

from .event_allocator import GPPOEventSlot
from .task_graph import TaskType


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PublicHeatPoint:
    heat_id: int
    position: tuple[float, float]
    priority: float = 5.0
    serviced: bool = False


class SearchSlotBuilder:
    """Build GPPO Search slots exclusively from public heat points.

    No hidden-target coordinate is accepted by this API.
    """

    def __init__(
        self,
        max_search_uavs: int = 2,
    ):
        self.max_search_uavs = int(
            max_search_uavs
        )

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
                source_entity_id=int(
                    point.heat_id
                ),
            )
            for point in selected
        ]


# ----------------------------------------------------------------------
# Relay
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RelayDemand:
    target_uav_id: int

    target_position: tuple[float, float]
    anchor_position: tuple[float, float]

    current_hops: int | None
    feasible_two_hop: bool


@dataclass(frozen=True)
class RelayDemandResult:
    demands: tuple[RelayDemand, ...]
    slots: tuple[GPPOEventSlot, ...]


class ObservedRelayDemandBuilder:
    """Build relay demands from observed/stale robot positions.

    The builder knows robot positions and an explicit base position.
    It never reads future/ground-truth trajectories.

    The midpoint anchor used here is a conservative one-helper bridge
    candidate for the <=2-hop protocol. Formal evaluation should
    replace this anchor rule with the exact frozen v9 generator once
    that source function is ported.
    """

    def __init__(
        self,
        *,
        comm_range: float = 20.0,
        max_hops: int = 2,
        max_demands: int = 2,
    ):
        self.comm_range = float(
            comm_range
        )
        self.max_hops = int(max_hops)
        self.max_demands = int(
            max_demands
        )

        if self.comm_range <= 0:
            raise ValueError(
                "comm_range must be positive"
            )

        if self.max_hops != 2:
            raise ValueError(
                "Current bridge builder implements "
                "the frozen <=2-hop protocol"
            )

    def build(
        self,
        positions: dict[int, np.ndarray],
        *,
        base_position,
        source_task_id: str = "T3_relay",
    ) -> RelayDemandResult:

        if not positions:
            return RelayDemandResult(
                demands=(),
                slots=(),
            )

        base = np.asarray(
            base_position,
            dtype=float,
        )

        robot_ids = sorted(
            int(uid)
            for uid in positions
        )

        robot_positions = {
            int(uid): np.asarray(
                positions[uid],
                dtype=float,
            )
            for uid in robot_ids
        }

        hops = self._hop_distances(
            robot_ids,
            robot_positions,
            base,
        )

        demands: list[RelayDemand] = []

        for uid in robot_ids:
            hop = hops.get(uid)

            if (
                hop is not None
                and hop <= self.max_hops
            ):
                continue

            target = robot_positions[uid]

            distance_to_base = float(
                np.linalg.norm(
                    target - base
                )
            )

            feasible = (
                distance_to_base
                <= self.comm_range
                * self.max_hops
                + 1e-9
            )

            # For <=2 hops, a helper must lie in both 20 m disks.
            # Their midpoint is a deterministic bridge candidate.
            anchor = (
                0.5 * base
                + 0.5 * target
            )

            demands.append(
                RelayDemand(
                    target_uav_id=uid,
                    target_position=(
                        float(target[0]),
                        float(target[1]),
                    ),
                    anchor_position=(
                        float(anchor[0]),
                        float(anchor[1]),
                    ),
                    current_hops=hop,
                    feasible_two_hop=feasible,
                )
            )

        # Disconnected first, then larger base distance.
        demands.sort(
            key=lambda demand: (
                0
                if demand.current_hops is None
                else 1,
                -float(
                    np.linalg.norm(
                        np.asarray(
                            demand.target_position
                        ) - base
                    )
                ),
                demand.target_uav_id,
            )
        )

        feasible_demands = [
            demand
            for demand in demands
            if demand.feasible_two_hop
        ][
            : self.max_demands
        ]

        slots = [
            GPPOEventSlot(
                slot_id=200000 + int(
                    demand.target_uav_id
                ),
                task_type=TaskType.RELAY,
                priority=2.0,
                position=demand.anchor_position,
                source_task_id=source_task_id,
                source_entity_id=int(
                    demand.target_uav_id
                ),
                forbidden_uav_ids=(
                    int(
                        demand.target_uav_id
                    ),
                ),
            )
            for demand in feasible_demands
        ]

        return RelayDemandResult(
            demands=tuple(demands),
            slots=tuple(slots),
        )

    def _hop_distances(
        self,
        robot_ids,
        positions,
        base,
    ) -> dict[int, int | None]:

        # Index 0 is the base station.
        nodes = [base] + [
            positions[uid]
            for uid in robot_ids
        ]

        points = np.asarray(
            nodes,
            dtype=float,
        )

        distances = np.linalg.norm(
            points[:, None, :]
            - points[None, :, :],
            axis=-1,
        )

        adjacency = (
            distances <= self.comm_range
        )

        np.fill_diagonal(
            adjacency,
            False,
        )

        hop_by_index = {
            0: 0,
        }

        queue = deque([0])

        while queue:
            node = queue.popleft()

            for neighbor in np.flatnonzero(
                adjacency[node]
            ):
                neighbor = int(neighbor)

                if neighbor in hop_by_index:
                    continue

                hop_by_index[neighbor] = (
                    hop_by_index[node] + 1
                )

                queue.append(neighbor)

        result: dict[int, int | None] = {}

        for index, uid in enumerate(
            robot_ids,
            start=1,
        ):
            result[uid] = (
                hop_by_index.get(index)
            )

        return result
