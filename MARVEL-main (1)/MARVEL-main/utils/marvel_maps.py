"""Original MARVEL map pipeline.

Reproduces the upstream MARVEL map preparation exactly, so the integration
can run in the same physical geometry the official MARVEL checkpoint and
the frozen GPPO ckpt80 were trained in:

    map image (maps_test / maps_medium)
        -> skimage.io.imread(..., 1)
        -> block_reduce(image, 2, np.min)
        -> threshold to FREE = 255 / OCCUPIED = 1
        -> CELL_SIZE = 0.4 m
        -> belief_origin = -round(initial_cell * CELL_SIZE, 1)

Source of truth: ``/home/nick/MARVEL`` @ ``e867117``,
``utils/scenario_env.py`` (``ScenarioEnv.import_ground_truth``) and
``utils/env_test.py``.  The raw image pixels are *not* metres; only the
downsampled 0.4 m lattice is.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import io
from skimage.measure import block_reduce

from utils.geometry import MARVEL_CELL_SIZE


# Original MARVEL map encoding (parameter.py / test_parameter.py).
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# Start-marker pixel value in the upstream maps.
ROBOT_MARKER = 208

# Upstream downsample factor for the belief lattice.
DOWNSAMPLE_FACTOR = 2

MAP_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def list_marvel_maps(map_dir: str | Path) -> list[Path]:
    """Sorted map images in a MARVEL map directory."""

    root = Path(map_dir)

    if not root.exists():
        raise FileNotFoundError(
            f"MARVEL map directory not found: {root}"
        )

    maps = sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and path.suffix.lower() in MAP_SUFFIXES
    )

    if not maps:
        raise FileNotFoundError(
            f"no map images found under {root}"
        )

    return maps


def resolve_marvel_map(
    map_dir: str | Path,
    episode_index: int = 0,
) -> Path:
    """Pick a map the way the upstream env does (index modulo count)."""

    maps = list_marvel_maps(map_dir)

    return maps[int(episode_index) % len(maps)]


def import_marvel_ground_truth(
    map_path: str | Path,
    *,
    downsample: int = DOWNSAMPLE_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ground_truth, initial_cell)`` exactly as upstream MARVEL.

    ``ground_truth`` is a ``(H, W)`` int array holding ``FREE`` (255) or
    ``OCCUPIED`` (1).  ``initial_cell`` is the ``(col, row)`` start marker
    on that lattice.
    """

    image = io.imread(str(map_path), 1).astype(int)

    ground_truth = block_reduce(
        image, int(downsample), np.min
    )

    marker = np.array(np.nonzero(ground_truth == ROBOT_MARKER))

    if marker.shape[1] <= 10:
        raise ValueError(
            f"map {Path(map_path).name} does not contain enough "
            "MARVEL start-marker pixels"
        )

    initial_cell = np.array([marker[1, 10], marker[0, 10]])

    ground_truth = (ground_truth > 150) | (
        (ground_truth <= 80) & (ground_truth >= 50)
    )
    ground_truth = ground_truth * (FREE - OCCUPIED) + OCCUPIED

    return ground_truth, initial_cell


def marvel_belief_origin(
    initial_cell,
    *,
    cell_size: float = MARVEL_CELL_SIZE,
) -> tuple[float, float]:
    """``-round(initial_cell * CELL_SIZE, 1)`` per upstream ``Env``."""

    origin_x = -np.round(
        float(initial_cell[0]) * float(cell_size), 1
    )
    origin_y = -np.round(
        float(initial_cell[1]) * float(cell_size), 1
    )

    return float(origin_x), float(origin_y)


def marvel_occupancy(
    ground_truth,
) -> np.ndarray:
    """Convert the 255/1 MARVEL truth to the runtime's 0-free / 1-occupied."""

    array = np.asarray(ground_truth)

    return np.where(array == FREE, 0, 1).astype(np.uint8)


def load_marvel_native_map(
    map_path: str | Path,
    *,
    cell_size: float = MARVEL_CELL_SIZE,
    downsample: int = DOWNSAMPLE_FACTOR,
) -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
    """Load one map in original MARVEL geometry.

    Returns ``(occupancy, origin, ground_truth)`` where ``occupancy`` uses
    the runtime's 0-free / 1-occupied convention on the 0.4 m lattice and
    ``origin`` is the upstream belief origin in metres.
    """

    ground_truth, initial_cell = import_marvel_ground_truth(
        map_path, downsample=downsample
    )

    origin = marvel_belief_origin(
        initial_cell, cell_size=cell_size
    )

    return marvel_occupancy(ground_truth), origin, ground_truth


def free_lattice_coordinates(
    ground_truth,
    origin,
    *,
    cell_size: float = MARVEL_CELL_SIZE,
) -> np.ndarray:
    """World coordinates of every ``FREE`` cell, row-major."""

    array = np.asarray(ground_truth)

    rows, cols = np.nonzero(array == FREE)

    return np.stack(
        (
            float(origin[0]) + cols * float(cell_size),
            float(origin[1]) + rows * float(cell_size),
        ),
        axis=1,
    )
