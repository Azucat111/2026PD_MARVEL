"""Static and dynamic 2-D obstacle management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .geometry import (
    GEOMETRY_MODE_NATIVE,
    MARVEL_CELL_SIZE,
    GeometryFrame,
    resolve_geometry_mode,
)
from .marvel_maps import (
    load_marvel_native_map,
    resolve_marvel_map,
)


@dataclass
class ExpandingCircleObstacle:
    obstacle_id: str
    position: np.ndarray
    initial_radius: float
    expansion_rate: float
    max_radius: float
    spawn_step: int = 0
    radius: float = 0.0
    active: bool = False

    def reset(self) -> None:
        self.radius = self.initial_radius
        self.active = False

    def step(self, current_step: int, _dt: float) -> None:
        self.active = current_step >= self.spawn_step
        if self.active:
            self.radius = min(self.max_radius, self.initial_radius +
                              (current_step - self.spawn_step) * self.expansion_rate)


class ObstacleManager:
    def __init__(self, environment: Dict[str, Any] | None = None,
                 dynamic_obstacles: Iterable[Dict[str, Any]] | None = None):
        environment = environment or {}
        self.geometry_mode = resolve_geometry_mode(environment)
        self.cell_size = (
            MARVEL_CELL_SIZE
            if self.geometry_mode == GEOMETRY_MODE_NATIVE
            else 1.0
        )
        self._origin = (0.0, 0.0)
        self.width, self.height = self._map_size(environment)
        self.static_obstacles = [
            {"position": np.asarray(item["position"], dtype=float),
             "radius": float(item.get("radius", 1.0))}
            for item in environment.get("circular_obstacles", [])
        ]
        self.dynamic_obstacles = []
        for item in dynamic_obstacles or []:
            params = item.get("params", {})
            if item.get("type") == "expanding_circle":
                self.dynamic_obstacles.append(ExpandingCircleObstacle(
                    item["id"], np.asarray(params["position"], dtype=float),
                    float(params.get("initial_radius", 1.0)),
                    float(params.get("expansion_rate", 0.0)),
                    float(params.get("max_radius", params.get("initial_radius", 1.0))),
                    int(item.get("spawn_step", 0))))
        self._file_grid: Optional[np.ndarray] = None  # H×W uint8, 1=occupied 0=free
        self.reset()

    def load_from_file(self, path: str) -> None:
        """Load an occupancy grid from a .npy or .png file.

        .npy convention: OCCUPIED=1, FREE=255, UNKNOWN=127 (treated as free).
        .png convention: pixel < 128 → occupied, pixel >= 128 → free.

        Updates self.width and self.height to match the loaded map.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Map file not found: {path}")
        suffix = p.suffix.lower()
        if suffix == ".npy":
            raw = np.load(str(p))
            grid = np.where(raw == 1, 1, 0).astype(np.uint8)
        elif suffix in (".png", ".jpg", ".jpeg", ".bmp"):
            try:
                from PIL import Image
                img = Image.open(str(p)).convert("L")
                raw = np.array(img, dtype=np.uint8)
            except ImportError:
                import cv2  # type: ignore
                raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    raise IOError(f"Could not read image: {path}")
            grid = np.where(raw < 128, 1, 0).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported map file format: {suffix}")

        # grid is H×W; width=cols, height=rows
        self._file_grid = grid
        self.height = float(grid.shape[0]) * self.cell_size
        self.width = float(grid.shape[1]) * self.cell_size
        self._origin = (0.0, 0.0)

    def load_marvel_native(self, map_dir: str, episode_index: int = 0) -> None:
        """Load original MARVEL geometry: 0.4 m lattice, belief origin.

        The raw image is downsampled and thresholded exactly as upstream
        MARVEL does; ``width``/``height`` become the physical extent in
        metres (``cells * CELL_SIZE``), and the grid keeps the 0.4 m
        lattice with its non-zero belief origin.
        """

        map_path = resolve_marvel_map(
            map_dir, int(episode_index)
        )

        grid, origin, _ground_truth = load_marvel_native_map(
            map_path, cell_size=self.cell_size
        )

        self._file_grid = grid
        self._origin = (float(origin[0]), float(origin[1]))
        self.height = float(grid.shape[0]) * self.cell_size
        self.width = float(grid.shape[1]) * self.cell_size
        self.map_name = map_path.name

    @property
    def frame(self) -> GeometryFrame:
        """The world <-> grid transform for this environment."""

        rows, cols = self._grid_shape()

        if self.geometry_mode == GEOMETRY_MODE_NATIVE:
            return GeometryFrame.marvel_native(
                width_cells=cols,
                height_cells=rows,
                origin_x=self._origin[0],
                origin_y=self._origin[1],
                cell_size=self.cell_size,
                extent_x=self.width,
                extent_y=self.height,
            )

        return GeometryFrame.extended(
            width_cells=cols,
            height_cells=rows,
            cell_size=self.cell_size,
            extent_x=self.width,
            extent_y=self.height,
        )

    @property
    def has_occupancy_map(self) -> bool:
        """True when a real map file backs the grid (vs. synthetic circles)."""

        return self._file_grid is not None

    @property
    def cell_count(self) -> int:
        """Total cells in the occupancy lattice."""

        rows, cols = self._grid_shape()

        return int(rows) * int(cols)

    def _grid_shape(self) -> tuple[int, int]:
        """``(rows, cols)`` of the occupancy grid, without building it."""

        if self._file_grid is not None:
            return self._file_grid.shape

        rows = int(self.height / self.cell_size) + 1
        cols = int(self.width / self.cell_size) + 1

        return (rows, cols)

    @staticmethod
    def _map_size(environment: Dict[str, Any]) -> tuple[float, float]:
        size = environment.get("map_size", environment.get("size", [150, 150]))
        return float(size[0]), float(size[1])

    def reset(self) -> None:
        for obstacle in self.dynamic_obstacles:
            obstacle.reset()

    def step(self, current_step: int, dt: float) -> list[Dict[str, Any]]:
        events: list[Dict[str, Any]] = []
        for obstacle in self.dynamic_obstacles:
            was_active = obstacle.active
            previous_radius = obstacle.radius
            obstacle.step(current_step, dt)
            if obstacle.active and not was_active:
                events.append({"type": "dynamic_obstacle_spawned",
                               "obstacle_id": obstacle.obstacle_id,
                               "radius": obstacle.radius})
            elif obstacle.active and obstacle.radius != previous_radius:
                events.append({"type": "dynamic_obstacle_updated",
                               "obstacle_id": obstacle.obstacle_id,
                               "radius": obstacle.radius})
        return events

    def check_collision(self, position: np.ndarray, radius: float = 0.2) -> tuple[bool, str | None]:
        point = np.asarray(position, dtype=float)
        frame = self.frame
        # Physical bounds, expressed in the active frame so a non-zero
        # MARVEL belief origin is handled without ad-hoc shifts.
        if np.any(point < frame.bounds_min) or np.any(point > frame.bounds_max):
            return True, "boundary"
        if self._file_grid is not None:
            # Sample the grid at the agent's footprint (centre + radius offsets)
            for dx, dy in [(0, 0), (radius, 0), (-radius, 0), (0, radius), (0, -radius)]:
                col, row = frame.world_to_cell_floor(
                    (point[0] + dx, point[1] + dy)
                )
                col, row = frame.clip_cell(col, row)
                if self._file_grid[row, col] == 1:
                    return True, "grid_obstacle"
        for obstacle in self.static_obstacles:
            if np.linalg.norm(point - obstacle["position"]) <= radius + obstacle["radius"]:
                return True, "static"
        for obstacle in self.dynamic_obstacles:
            if obstacle.active and np.linalg.norm(point - obstacle.position) <= radius + obstacle.radius:
                return True, f"dynamic_{obstacle.obstacle_id}"
        return False, None

    def get_occupancy_grid(self, resolution: float | None = None) -> np.ndarray:
        if self._file_grid is not None:
            return self._file_grid.copy()
        if resolution is None:
            resolution = self.cell_size
        frame = self.frame
        grid = np.zeros(frame.shape, dtype=np.uint8)
        yy, xx = np.indices(grid.shape)
        points = np.stack((xx * resolution, yy * resolution), axis=-1)
        for obstacle in self.static_obstacles + [
            {"position": item.position, "radius": item.radius}
            for item in self.dynamic_obstacles if item.active
        ]:
            grid[np.linalg.norm(points - obstacle["position"], axis=-1) <= obstacle["radius"]] = 1
        return grid
