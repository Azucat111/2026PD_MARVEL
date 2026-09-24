"""Explicit world <-> grid coordinate frame for the simulation runtime.

The runtime historically assumed a single implicit frame: origin ``(0, 0)``
and a 1 metre grid cell.  That frame is the *extended* environment.  The
MARVEL-native parity environment runs the original MARVEL geometry instead:
a 0.4 m occupancy lattice over an origin that is generally **not** zero
(``belief_origin = -round(initial_cell * CELL_SIZE, 1)``).

Rather than stretch one representation to cover both, every subsystem that
converts between world metres and grid cells goes through
:class:`GeometryFrame`.  There is one transform, owned in one place.

Two lookup conventions live on the frame, and they are deliberately
different because they answer different questions:

``world_to_cell``
    Nearest lattice point, ``round((p - origin) / cell_size)``.  This is
    MARVEL's own convention (``utils/utils.get_cell_position_from_coords``)
    and the one the frozen Search scenario generator uses.  Use it when a
    coordinate denotes a *point on the lattice* (node coordinates, heat
    points, hidden survivors).

``world_to_cell_floor``
    Containing cell, ``floor((p - origin) / cell_size)``.  Use it when the
    grid is treated as *areal occupancy* (collision footprint probes,
    sensor ray sampling).

At the extended frame (origin 0, cell 1.0) both reduce to the legacy
behaviour, which keeps existing 500 m scenarios bit-identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


# Original MARVEL constants (parameter.py / test_parameter.py).
MARVEL_CELL_SIZE = 0.4
MARVEL_NODE_RESOLUTION = 4.0
MARVEL_SENSOR_RANGE = 10.0

GEOMETRY_MODE_EXTENDED = "extended"
GEOMETRY_MODE_NATIVE = "marvel_native"

GEOMETRY_MODES = (
    GEOMETRY_MODE_EXTENDED,
    GEOMETRY_MODE_NATIVE,
)


@dataclass(frozen=True)
class GeometryFrame:
    """World <-> grid transform for one environment geometry."""

    origin_x: float
    origin_y: float
    cell_size: float
    width_cells: int
    height_cells: int
    mode: str = GEOMETRY_MODE_EXTENDED

    # Physical extent in metres.  Normally ``cells * cell_size``, but the
    # synthetic-obstacle environment declares a 150 m map backed by a
    # 151-cell lattice, so the declared extent is carried explicitly
    # rather than re-derived.  ``None`` means "derive from the lattice".
    extent_x: float | None = None
    extent_y: float | None = None

    def __post_init__(self) -> None:
        if self.cell_size <= 0:
            raise ValueError(
                "cell_size must be positive"
            )

        if self.width_cells <= 0 or self.height_cells <= 0:
            raise ValueError(
                "grid must be non-empty"
            )

        if self.mode not in GEOMETRY_MODES:
            raise ValueError(
                f"unknown geometry mode: {self.mode}"
            )

    # ------------------------------------------------------------------
    # Extent
    # ------------------------------------------------------------------
    @property
    def origin(self) -> tuple[float, float]:
        return (float(self.origin_x), float(self.origin_y))

    @property
    def width_m(self) -> float:
        if self.extent_x is not None:
            return float(self.extent_x)

        return float(self.width_cells) * float(self.cell_size)

    @property
    def height_m(self) -> float:
        if self.extent_y is not None:
            return float(self.extent_y)

        return float(self.height_cells) * float(self.cell_size)

    @property
    def bounds_min(self) -> np.ndarray:
        return np.asarray(
            (self.origin_x, self.origin_y), dtype=float
        )

    @property
    def bounds_max(self) -> np.ndarray:
        return np.asarray(
            (
                self.origin_x + self.width_m,
                self.origin_y + self.height_m,
            ),
            dtype=float,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)``, matching numpy occupancy-grid shape."""

        return (int(self.height_cells), int(self.width_cells))

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def world_to_cell(self, position) -> tuple[int, int]:
        """Nearest lattice cell as ``(col, row)``."""

        point = np.asarray(position, dtype=float)

        col = int(
            round(
                (float(point[0]) - self.origin_x)
                / self.cell_size
            )
        )
        row = int(
            round(
                (float(point[1]) - self.origin_y)
                / self.cell_size
            )
        )

        return col, row

    def world_to_cell_floor(self, position) -> tuple[int, int]:
        """Containing cell as ``(col, row)``, for areal lookup."""

        point = np.asarray(position, dtype=float)

        col = int(
            np.floor(
                (float(point[0]) - self.origin_x)
                / self.cell_size
            )
        )
        row = int(
            np.floor(
                (float(point[1]) - self.origin_y)
                / self.cell_size
            )
        )

        return col, row

    def world_to_cell_float(self, position) -> tuple[float, float]:
        """Unrounded ``(col, row)`` for interpolation along a ray."""

        point = np.asarray(position, dtype=float)

        return (
            (float(point[0]) - self.origin_x) / self.cell_size,
            (float(point[1]) - self.origin_y) / self.cell_size,
        )

    def cell_to_world(self, cell) -> np.ndarray:
        """Lattice cell as ``(x, y)`` world metres."""

        col = float(cell[0])
        row = float(cell[1])

        return np.asarray(
            (
                self.origin_x + col * self.cell_size,
                self.origin_y + row * self.cell_size,
            ),
            dtype=float,
        )

    def cells_to_world(self, cells) -> np.ndarray:
        """Vectorised ``(N, 2)`` cell array to world metres."""

        array = np.asarray(cells, dtype=float).reshape(-1, 2)

        return np.stack(
            (
                self.origin_x
                + array[:, 0] * self.cell_size,
                self.origin_y
                + array[:, 1] * self.cell_size,
            ),
            axis=1,
        )

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------
    def contains_cell(self, col: int, row: int) -> bool:
        return (
            0 <= int(col) < self.width_cells
            and 0 <= int(row) < self.height_cells
        )

    def contains_world(self, position) -> bool:
        point = np.asarray(position, dtype=float)

        lower = self.bounds_min
        upper = self.bounds_max

        return bool(
            np.all(point >= lower - 1e-9)
            and np.all(point <= upper + 1e-9)
        )

    def clip_cell(self, col: int, row: int) -> tuple[int, int]:
        return (
            int(np.clip(col, 0, self.width_cells - 1)),
            int(np.clip(row, 0, self.height_cells - 1)),
        )

    def cell_center(self, cell) -> np.ndarray:
        """World position at the lattice point of *cell* (same as corner)."""

        return self.cell_to_world(cell)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def extended(
        cls,
        *,
        width_cells: int,
        height_cells: int,
        cell_size: float = 1.0,
        extent_x: float | None = None,
        extent_y: float | None = None,
    ) -> "GeometryFrame":
        """Team extended environment: origin (0, 0)."""

        return cls(
            origin_x=0.0,
            origin_y=0.0,
            cell_size=float(cell_size),
            width_cells=int(width_cells),
            height_cells=int(height_cells),
            mode=GEOMETRY_MODE_EXTENDED,
            extent_x=extent_x,
            extent_y=extent_y,
        )

    @classmethod
    def marvel_native(
        cls,
        *,
        width_cells: int,
        height_cells: int,
        origin_x: float,
        origin_y: float,
        cell_size: float = MARVEL_CELL_SIZE,
        extent_x: float | None = None,
        extent_y: float | None = None,
    ) -> "GeometryFrame":
        """Original MARVEL geometry with a non-zero belief origin."""

        return cls(
            origin_x=float(origin_x),
            origin_y=float(origin_y),
            cell_size=float(cell_size),
            width_cells=int(width_cells),
            height_cells=int(height_cells),
            mode=GEOMETRY_MODE_NATIVE,
            extent_x=extent_x,
            extent_y=extent_y,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def is_native(self) -> bool:
        return self.mode == GEOMETRY_MODE_NATIVE

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "origin": [
                float(self.origin_x),
                float(self.origin_y),
            ],
            "cell_size": float(self.cell_size),
            "width_cells": int(self.width_cells),
            "height_cells": int(self.height_cells),
            "extent_m": [
                float(self.width_m),
                float(self.height_m),
            ],
        }


def resolve_geometry_mode(environment: dict[str, Any]) -> str:
    """Read ``environment.geometry_mode`` explicitly.

    The mode is never inferred.  Omitting it means ``extended`` so that
    every pre-existing scenario keeps its current geometry.
    """

    environment = environment or {}

    mode = str(
        environment.get("geometry_mode", GEOMETRY_MODE_EXTENDED)
    ).strip().lower()

    if mode not in GEOMETRY_MODES:
        raise ValueError(
            "environment.geometry_mode must be one of "
            f"{GEOMETRY_MODES}, got {mode!r}"
        )

    return mode


def sample_free_cells(
    frame: GeometryFrame,
    grid,
    *,
    free_value: int = 0,
) -> np.ndarray:
    """Return the world coordinates of every free cell, row-major."""

    array = np.asarray(grid)

    rows, cols = np.nonzero(array == free_value)

    return np.stack(
        (
            frame.origin_x + cols * frame.cell_size,
            frame.origin_y + rows * frame.cell_size,
        ),
        axis=1,
    )


def iter_cells_in_radius(
    frame: GeometryFrame,
    position,
    radius: float,
) -> Iterable[tuple[int, int]]:
    """Yield candidate cell indices whose lattice point is within *radius*."""

    point = np.asarray(position, dtype=float)

    col_min = int(
        np.floor(
            (point[0] - radius - frame.origin_x) / frame.cell_size
        )
    )
    col_max = int(
        np.ceil(
            (point[0] + radius - frame.origin_x) / frame.cell_size
        )
    )
    row_min = int(
        np.floor(
            (point[1] - radius - frame.origin_y) / frame.cell_size
        )
    )
    row_max = int(
        np.ceil(
            (point[1] + radius - frame.origin_y) / frame.cell_size
        )
    )

    col_min = max(0, col_min)
    row_min = max(0, row_min)
    col_max = min(frame.width_cells - 1, col_max)
    row_max = min(frame.height_cells - 1, row_max)

    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            yield col, row
