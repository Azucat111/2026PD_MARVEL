from __future__ import annotations

import numpy as np


class StalePositionTracker:
    """Sub-step stale position approximation for GPPO decision state.

    age_fraction = 0:
        current physical position

    age_fraction = 1:
        previous physics-tick position

    For Phase14:
        50 ms / 100 ms = 0.5
        => midpoint between previous and current position.
    """

    def __init__(self):
        self.previous: dict[int, np.ndarray] = {}
        self.current: dict[int, np.ndarray] = {}

    def reset(self, runtime) -> None:
        self.current = {
            int(robot.robot_id): robot.position.copy()
            for robot in runtime.robots
        }
        self.previous = {
            uid: pos.copy()
            for uid, pos in self.current.items()
        }

    def observe(self, runtime) -> None:
        if not self.current:
            self.reset(runtime)
            return

        self.previous = {
            uid: pos.copy()
            for uid, pos in self.current.items()
        }

        self.current = {
            int(robot.robot_id): robot.position.copy()
            for robot in runtime.robots
        }

    def positions(
        self,
        age_fraction: float,
    ) -> dict[int, np.ndarray]:
        alpha = float(age_fraction)

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "age_fraction must be in [0, 1]"
            )

        result = {}

        for uid, current in self.current.items():
            previous = self.previous.get(uid, current)

            result[uid] = (
                (1.0 - alpha) * current
                + alpha * previous
            )

        return result
