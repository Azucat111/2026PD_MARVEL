"""Original MARVEL sensing, ported verbatim for the native parity mode.

Source of truth: ``/home/nick/MARVEL`` @ ``e867117``, ``utils/sensor.py``
and ``utils/env.py``.  The frozen belief/exploration state is produced by

    sensor_work_heading  ->  fov_sweep / calculate_fov_boundaries
                         ->  collision_check
                         ->  robot_belief

and consumed by ``Env.evaluate_exploration_rate``:

    explored_rate = sum(robot_belief == 255) / sum(ground_truth == 255)

Every function below is a line-for-line port; the only additions are type
hints and docstrings.  Do not "simplify" the Bresenham loop or the angular
sweep: the rounding and the branch order are what make the parity exact.

This module is used only by ``geometry_mode == 'marvel_native'``.  The
extended environment keeps its own ``IdealSensor`` path unchanged.
"""
from __future__ import annotations

import numpy as np


# Original MARVEL map / belief encoding.
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# Frozen sensor_work_heading angular increment, in degrees.
SENSOR_ANGLE_INCREMENT_DEG = 0.5

# Frozen collision_check: the ray ends after this many occupied cells.
MAX_COLLISION = 2

# Frozen test_parameter.NUM_SIM_STEPS: motion/sensing sub-steps per mission
# step in marvel_gppo_test_worker._simulate_motion.
FROZEN_NUM_SIM_STEPS = 6


def calculate_fov_boundaries(
    center_angle: float,
    fov: float,
) -> tuple[float, float]:
    """Frozen ``utils/sensor.py:calculate_fov_boundaries``."""

    half_fov = fov / 2

    start_angle = center_angle - half_fov
    end_angle = center_angle + half_fov

    start_angle = start_angle % 360
    end_angle = end_angle % 360

    return start_angle, end_angle


def fov_sweep(
    start_angle: float,
    end_angle: float,
    increment: float,
) -> np.ndarray:
    """Frozen ``utils/sensor.py:fov_sweep``; returns radians."""

    angles = []

    if start_angle < end_angle:
        angles = list(
            np.arange(start_angle, end_angle + increment, increment)
        )
    else:
        angles = list(np.arange(start_angle, 360, increment)) + list(
            np.arange(0, end_angle + increment, increment)
        )

    angles = [angle % 360 for angle in angles]

    angles_in_radians = np.radians(angles)

    return angles_in_radians


def collision_check(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    ground_truth: np.ndarray,
    robot_belief: np.ndarray,
) -> np.ndarray:
    """Frozen ``utils/sensor.py:collision_check``.

    Integer Bresenham walk from the robot cell to the ray endpoint, writing
    the *ground-truth* value of each visited cell into ``robot_belief``.

    Exact semantics that matter for parity:

    * ``x0, y0, x1, y1`` are rounded with ``np.round`` (half-to-even).
    * The robot's own cell is written on the first iteration when the
      guards allow it.
    * The endpoint cell is **never** written: the ``x == x1 and y == y1``
      guard breaks before the write.
    * At most **one** occupied cell is written per ray.  Reaching
      ``MAX_COLLISION`` occupied cells breaks; an occupied cell followed by
      a free cell also breaks, without writing the free cell.
    * Leaving the map terminates the walk silently.
    """

    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()

    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0
    max_collision = MAX_COLLISION

    while 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
        k = ground_truth.item(y, x)
        if k == 1 and collision_flag < max_collision:
            collision_flag += 1
            if collision_flag >= max_collision:
                break

        if k != 1 and collision_flag > 0:
            break

        if x == x1 and y == y1:
            break

        robot_belief.itemset((y, x), k)

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return robot_belief


def sensor_work_heading(
    robot_position,
    sensor_range: float,
    robot_belief: np.ndarray,
    ground_truth: np.ndarray,
    heading: float,
    fov: float,
) -> np.ndarray:
    """Frozen ``utils/sensor.py:sensor_work_heading``.

    ``robot_position`` is the robot's **cell** coordinate and
    ``sensor_range`` is expressed in **cells**.

    ``robot_position`` must be an **integer** cell, exactly as
    ``get_cell_position_from_coords`` returns it.  The frozen Bresenham
    loop keeps ``x``/``y`` as the integer seed it started from, which is
    what lets ``ground_truth.item(y, x)`` accept them; passing floats
    raises inside the frozen code.
    """

    sensor_angle_inc = SENSOR_ANGLE_INCREMENT_DEG
    x0 = robot_position[0]
    y0 = robot_position[1]
    start_angle, end_angle = calculate_fov_boundaries(heading, fov)
    sweep_angles = fov_sweep(start_angle, end_angle, sensor_angle_inc)

    for angle in sweep_angles:
        x1 = x0 + np.cos(angle) * sensor_range
        y1 = y0 + np.sin(angle) * sensor_range
        robot_belief = collision_check(
            x0, y0, x1, y1, ground_truth, robot_belief
        )

    return robot_belief


class MarvelNativeBelief:
    """Authoritative exploration state for ``geometry_mode='marvel_native'``.

    Owns the frozen ``robot_belief`` array and the frozen exploration-rate
    definition.  This is the single source of truth for native-mode
    exploration; the ordinary runtime ``explored_cells`` set is retained
    only for the extended environment.
    """

    def __init__(
        self,
        *,
        ground_truth: np.ndarray,
        cell_size: float,
        sensor_range: float,
        fov: float,
    ):
        if abs(float(cell_size)) < 1e-12:
            raise ValueError("cell_size must be positive")

        self.ground_truth = np.asarray(ground_truth)

        if self.ground_truth.ndim != 2:
            raise ValueError("ground_truth must be two-dimensional")

        self.cell_size = float(cell_size)
        self.sensor_range = float(sensor_range)
        self.fov = float(fov)

        # Frozen Env: np.ones(ground_truth_size) * UNKNOWN.
        self.belief = (
            np.ones(self.ground_truth.shape, dtype=np.int32) * UNKNOWN
        )

        self.free_total = int(
            np.sum(self.ground_truth == FREE)
        )

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------
    @property
    def sensor_range_cells(self) -> float:
        """Frozen ``Env.update_robot_belief``: round(range / cell_size)."""

        return round(self.sensor_range / self.cell_size)

    def observe(
        self,
        cell,
        heading: float,
    ) -> None:
        """One frozen ``Env.update_robot_belief(cell, heading)`` call.

        ``cell`` is converted exactly as ``get_cell_position_from_coords``
        does (``np.around(...).astype(int)``), because the frozen Bresenham
        loop requires integer seeds.
        """

        integer_cell = np.around(
            np.asarray(cell, dtype=float)
        ).astype(int)

        self.belief = sensor_work_heading(
            integer_cell,
            round(self.sensor_range / self.cell_size),
            self.belief,
            self.ground_truth,
            float(heading),
            self.fov,
        )

    # ------------------------------------------------------------------
    # Frozen exploration rate
    # ------------------------------------------------------------------
    @property
    def explored_free_count(self) -> int:
        return int(np.sum(self.belief == FREE))

    @property
    def free_cell_total(self) -> int:
        return self.free_total

    @property
    def explored_rate(self) -> float:
        """Frozen ``Env.evaluate_exploration_rate``."""

        return (
            np.sum(self.belief == FREE)
            / np.sum(self.ground_truth == FREE)
        )

    def free_mask(self) -> np.ndarray:
        return self.ground_truth == FREE
