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

    def __init__(self, params: Dict[str, Any] | None = None):
        params = params or {}
        self.fov = float(params.get("fov", 120.0))
        self.sensor_range = float(params.get("range", params.get("sensor_range", 10.0)))
        self.los_occlusion = bool(params.get("los_occlusion", False))

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

    def _visible_cells(self, position: np.ndarray, heading: float, occupancy_grid: np.ndarray) -> np.ndarray:
        y_max, x_max = occupancy_grid.shape
        xmin = max(0, int(np.floor(position[0] - self.sensor_range)))
        xmax = min(x_max - 1, int(np.ceil(position[0] + self.sensor_range)))
        ymin = max(0, int(np.floor(position[1] - self.sensor_range)))
        ymax = min(y_max - 1, int(np.ceil(position[1] + self.sensor_range)))
        cells = []
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                delta = np.array([x - position[0], y - position[1]], dtype=float)
                distance = np.linalg.norm(delta)
                if distance > self.sensor_range or distance < 1e-9:
                    continue
                angle = (np.degrees(np.arctan2(delta[1], delta[0])) - heading + 180.0) % 360.0 - 180.0
                if abs(angle) <= self.fov / 2:
                    if (
                        self.los_occlusion
                        and not self._has_line_of_sight(
                            position,
                            (x, y),
                            occupancy_grid,
                        )
                    ):
                        continue
                    cells.append((x, y))
        return np.asarray(cells, dtype=int)

    @staticmethod
    def _has_line_of_sight(
        position: np.ndarray,
        target_cell,
        occupancy_grid: np.ndarray,
    ) -> bool:
        """Return False when an occupied cell blocks the sight line."""

        x0 = float(position[0])
        y0 = float(position[1])

        x1 = int(target_cell[0])
        y1 = int(target_cell[1])

        dx = float(x1) - x0
        dy = float(y1) - y0

        # Half-cell sampling prevents thin occupied cells being skipped.
        samples = max(
            1,
            int(
                np.ceil(
                    2.0 * max(abs(dx), abs(dy))
                )
            ),
        )

        height, width = occupancy_grid.shape

        for i in range(1, samples):
            alpha = i / samples

            x = int(round(x0 + alpha * dx))
            y = int(round(y0 + alpha * dy))

            if not (
                0 <= x < width
                and 0 <= y < height
            ):
                return False

            # The target cell itself remains visible; an obstacle
            # strictly before it blocks the ray.
            if (x, y) == (x1, y1):
                break

            if int(occupancy_grid[y, x]) == 1:
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


def create_sensor_model(config_file: str | Path | Dict[str, Any]) -> SensorModel:
    config = config_file if isinstance(config_file, dict) else _read_yaml(config_file)
    model_type = config.get("sensor_type", "ideal")
    params = config.get("params", config)
    if model_type != "ideal":
        raise ValueError(f"Unsupported sensor model: {model_type}")
    return IdealSensor(params)


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
