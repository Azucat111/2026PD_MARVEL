"""Original MARVEL low-level motion semantics, for the native parity mode.

Frozen MARVEL does not integrate a kinematic model.  ``Env.final_sim_step``
assigns the commanded waypoint directly:

    def final_sim_step(self, next_waypoint, agent_id):
        self.evaluate_exploration_rate()
        self.robot_locations[agent_id] = next_waypoint

so the robot's location becomes exactly the selected graph node.  The only
continuously modelled quantity is the heading, produced by
``compute_allowable_heading`` from the distance travelled and the yaw rate.

Sensing is then interpolated across ``NUM_SIM_STEPS = 6`` sub-steps between
the previous cell and the new cell, with shortest-arc heading interpolation
(see ``marvel_sensing``).

Source of truth: ``/home/nick/MARVEL`` @ ``e867117``, ``utils/motion_model.py``
and ``utils/marvel_gppo_test_worker.py:533-586``.  Used only by
``geometry_mode = 'marvel_native'``; the extended environment keeps the team
dynamics model.
"""
from __future__ import annotations

import math

import numpy as np


# Frozen test_parameter constants.
FROZEN_VELOCITY = 1.0
FROZEN_YAW_RATE = 35.0  # degrees per second
FROZEN_NUM_ANGLES_BIN = 36


def normalize_angle(angle: float) -> float:
    """Frozen ``utils/sensor.py:normalize_angle``."""

    return angle % 360


def compute_allowable_heading(
    current_position,
    final_position,
    theta_current: float,
    theta_desired: float,
    v_current: float,
    omega_max: float,
) -> float:
    """Frozen ``utils/motion_model.py:compute_allowable_heading``.

    Returns the heading the robot can reach by the time it covers the
    commanded distance at ``v_current``, limited by ``omega_max``.
    """

    x_current, y_current = current_position
    x_final, y_final = final_position

    # Calculate target heading based on current and final positions
    theta_target = math.degrees(
        math.atan2(y_final - y_current, x_final - x_current)
    )
    theta_target = normalize_angle(theta_target)

    # Calculate the desired change in heading
    delta_theta_desired = (
        normalize_angle(theta_desired)
        - normalize_angle(theta_current)
    )

    # Normalize the desired change to the range [-180, 180]
    if delta_theta_desired > 180:
        delta_theta_desired -= 360
    elif delta_theta_desired < -180:
        delta_theta_desired += 360

    # Calculate time to achieve desired heading change
    t_desired_yaw = abs(delta_theta_desired) / omega_max

    # Calculate the distance to the final position
    distance_to_final = np.linalg.norm(
        np.asarray(final_position, dtype=float)
        - np.asarray(current_position, dtype=float)
    )

    # Calculate the time to reach the final position
    t_travel = distance_to_final / v_current

    # Check if the desired heading change is achievable within travel time
    if t_desired_yaw <= t_travel:
        return normalize_angle(theta_desired)

    # Calculate the achievable heading change within the max yaw rate and
    # travel time
    delta_theta_achievable = t_travel * omega_max
    theta_achievable = theta_current + math.copysign(
        delta_theta_achievable, delta_theta_desired
    )

    return normalize_angle(theta_achievable)


def interpolated_sensing_track(
    start_position,
    end_position,
    start_heading: float,
    final_heading: float,
    *,
    cell_size: float,
    origin: tuple[float, float],
    sim_steps: int,
) -> tuple[np.ndarray, list[float]]:
    """Frozen ``_simulate_motion`` cell/heading interpolation.

    ``cells = round(linspace(start_cell, end_cell, sim_steps + 1)[1:])``
    with the final entry equal to the commanded end cell, and headings
    interpolated along the shortest arc.
    """

    def to_cell(position):
        point = np.asarray(position, dtype=float)

        return np.asarray(
            (
                (point[0] - origin[0]) / cell_size,
                (point[1] - origin[1]) / cell_size,
            ),
            dtype=float,
        )

    start_cell = to_cell(start_position)
    end_cell = to_cell(end_position)

    cells = np.round(
        np.linspace(start_cell, end_cell, int(sim_steps) + 1)[1:]
    ).astype(int)

    previous = float(start_heading) % 360.0
    final = float(final_heading) % 360.0

    diff = final - previous

    if abs(diff) > 180:
        diff = diff - 360 if diff > 0 else diff + 360

    headings = [
        (previous + (j + 1) * diff / int(sim_steps)) % 360.0
        for j in range(int(sim_steps))
    ]

    return cells, headings
