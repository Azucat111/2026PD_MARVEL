"""Sensor model interfaces for the new simulation runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


class SensorModel:
    def sense(self, robot_state, occupancy_grid: np.ndarray, all_robot_positions) -> Dict[str, Any]:
        raise NotImplementedError


class IdealSensor(SensorModel):
    """Simple FoV/range observation that is independent from MARVEL's policy graph."""

    def __init__(
        self,
        params: Dict[str, Any] | None = None,
        frame=None,
    ):
        params = params or {}
        self.fov = float(params.get("fov", 120.0))
        self.sensor_range = float(params.get("range", params.get("sensor_range", 10.0)))
        self.los_occlusion = bool(params.get("los_occlusion", False))
        # World <-> grid transform.  When absent, the historical implicit
        # frame (origin 0, 1 m cells) is assumed.
        self.frame = frame

    def sense(self, robot_state, occupancy_grid: np.ndarray, all_robot_positions) -> Dict[str, Any]:
        position = np.asarray(robot_state.position, dtype=float)
        return {
            "robot_id": robot_state.robot_id,
            "position": position.copy(),
            "heading": float(robot_state.heading),
            "velocity": float(robot_state.velocity),
            "visible_cells": self._visible_cells(position, float(robot_state.heading), occupancy_grid),
            "nearby_robots": self._nearby_robots(position, all_robot_positions),
        }

    def _resolve_frame(self, occupancy_grid: np.ndarray):
        """Return the active frame, defaulting to the historical one.

        The legacy default is origin ``(0, 0)`` with 1 m cells, which is
        exactly what callers that never set a frame relied on.
        """

        if self.frame is not None:
            return self.frame

        from .geometry import GeometryFrame

        return GeometryFrame.extended(
            width_cells=int(occupancy_grid.shape[1]),
            height_cells=int(occupancy_grid.shape[0]),
        )

    def _visible_cells(self, position: np.ndarray, heading: float, occupancy_grid: np.ndarray) -> np.ndarray:
        from .geometry import iter_cells_in_radius

        frame = self._resolve_frame(occupancy_grid)

        position = np.asarray(position, dtype=float)[:2]

        cells = []
        # Iterate candidate cells on the active lattice; all geometry below
        # is computed in world metres, never by reading a metre value as a
        # cell index.
        for col, row in iter_cells_in_radius(
            frame, position, self.sensor_range
        ):
            world = frame.cell_to_world((col, row))
            delta = np.asarray(world - position, dtype=float)
            distance = np.linalg.norm(delta)
            if distance > self.sensor_range or distance < 1e-9:
                continue
            angle = (np.degrees(np.arctan2(delta[1], delta[0])) - heading + 180.0) % 360.0 - 180.0
            if abs(angle) <= self.fov / 2:
                if (
                    self.los_occlusion
                    and not self._has_line_of_sight(
                        position,
                        (col, row),
                        occupancy_grid,
                    )
                ):
                    continue
                cells.append((col, row))
        return np.asarray(cells, dtype=int)

    def _has_line_of_sight(
        self,
        position: np.ndarray,
        target_cell,
        occupancy_grid: np.ndarray,
    ) -> bool:
        """Return False when an occupied cell blocks the sight line."""

        frame = self._resolve_frame(occupancy_grid)

        position = np.asarray(position, dtype=float)[:2]

        col0, row0 = frame.world_to_cell_float(position)

        col1 = int(target_cell[0])
        row1 = int(target_cell[1])

        d_col = float(col1) - col0
        d_row = float(row1) - row0

        # Half-cell sampling prevents thin occupied cells being skipped.
        # The sample count is derived from the cell-space distance so it
        # stays half-cell dense on a 0.4 m lattice too.
        samples = max(
            1,
            int(
                np.ceil(
                    2.0 * max(abs(d_col), abs(d_row))
                )
            ),
        )

        for i in range(1, samples):
            alpha = i / samples

            col = int(round(col0 + alpha * d_col))
            row = int(round(row0 + alpha * d_row))

            if not frame.contains_cell(col, row):
                return False

            # The target cell itself remains visible; an obstacle
            # strictly before it blocks the ray.
            if (col, row) == (col1, row1):
                break

            if int(occupancy_grid[row, col]) == 1:
                return False

        return True

    def _nearby_robots(self, position: np.ndarray, all_robot_positions) -> list[int]:
        nearby = []
        for item in all_robot_positions:
            if isinstance(item, tuple) and len(item) == 2:
                robot_id, other = item
            else:
                robot_id, other = len(nearby), item
            if np.linalg.norm(position - np.asarray(other, dtype=float)) <= self.sensor_range:
                nearby.append(int(robot_id))
        return nearby


def create_sensor_model(
    config_file: str | Path | Dict[str, Any],
    frame=None,
) -> SensorModel:
    config = config_file if isinstance(config_file, dict) else _read_yaml(config_file)
    model_type = config.get("sensor_type", "ideal")
    params = config.get("params", config)
    if model_type != "ideal":
        raise ValueError(f"Unsupported sensor model: {model_type}")
    return IdealSensor(params, frame=frame)


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
