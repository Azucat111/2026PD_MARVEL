"""Low-level SAFETY escape controller.

The controller stays on MARVEL's current spatial graph and greedily chooses the
neighbor with the largest minimum clearance from active hazards.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _heading_bin(from_xy, to_xy, num_angle_bins: int) -> int:
    delta = np.asarray(to_xy, dtype=float) - np.asarray(from_xy, dtype=float)
    if float(np.linalg.norm(delta)) < 1e-9:
        return 0
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])) % 360.0)
    return int(np.floor(angle / 360.0 * num_angle_bins)) % num_angle_bins


class SafetyController:
    def __init__(self, num_angle_bins: int = 36):
        self.num_angle_bins = int(num_angle_bins)

    def _candidate_nodes(self, robot):
        if robot.neighbor_indices is None or robot.node_coords is None:
            return []

        out = []
        for raw_idx in np.asarray(robot.neighbor_indices).reshape(-1):
            idx = int(raw_idx)
            if 0 <= idx < len(robot.node_coords):
                out.append(
                    (idx, np.asarray(robot.node_coords[idx], dtype=float))
                )
        return out

    def _clearance(self, point, hazard_manager) -> float:
        risk = hazard_manager.risk_at(point)
        if risk is None:
            return float("inf")
        return float(risk.clearance)

    def select_next_waypoint(
        self,
        robot,
        hazard_manager,
    ) -> Tuple[np.ndarray, int, int]:
        current = np.asarray(robot.location, dtype=float)
        candidates = self._candidate_nodes(robot)

        if not candidates:
            idx = int(robot.current_index) if robot.current_index is not None else 0
            return current.copy(), idx, 0

        current_clearance = self._clearance(current, hazard_manager)

        ranked = sorted(
            candidates,
            key=lambda item: (
                self._clearance(item[1], hazard_manager),
                float(np.linalg.norm(item[1] - current)),
            ),
            reverse=True,
        )

        idx, coords = ranked[0]

        # Avoid staying in place when another node increases clearance.
        if (
            robot.current_index is not None
            and int(idx) == int(robot.current_index)
        ):
            for alt_idx, alt_coords in ranked[1:]:
                if (
                    int(alt_idx) != int(robot.current_index)
                    and self._clearance(alt_coords, hazard_manager)
                    >= current_clearance
                ):
                    idx, coords = alt_idx, alt_coords
                    break

        # Heading points away from the nearest active hazard.
        risk = hazard_manager.risk_at(current)
        if risk is None:
            heading_target = coords
        else:
            center = np.asarray(risk.center, dtype=float)
            away = current + (current - center)
            if float(np.linalg.norm(current - center)) < 1e-9:
                away = coords
            heading_target = away

        heading = _heading_bin(
            current,
            heading_target,
            self.num_angle_bins,
        )
        return np.asarray(coords, dtype=float).copy(), int(idx), int(heading)
