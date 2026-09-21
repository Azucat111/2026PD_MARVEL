from __future__ import annotations

from typing import Any, Iterable

import numpy as np


class TargetDetector:
    """Sensor-faithful hidden-target detector.

    Hidden coordinates live here in the environment/detection layer.
    GPPO, Search routing, and TaskManager do not need target coordinates.
    """

    def __init__(
        self,
        hidden_targets: dict[str, list[dict[str, Any]]] | None = None,
    ):
        self.hidden_targets = hidden_targets or {}

    def detect(
        self,
        task_id: str,
        observations: dict[int, dict[str, Any]],
        eligible_robot_ids: Iterable[int],
    ) -> dict[int, int]:
        """Return target_index -> detecting_robot_id."""

        targets = self.hidden_targets.get(
            str(task_id),
            [],
        )

        eligible = {
            int(uid)
            for uid in eligible_robot_ids
        }

        visible_by_robot: dict[
            int,
            set[tuple[int, int]],
        ] = {}

        for uid in eligible:
            observation = observations.get(uid, {})

            visible = np.asarray(
                observation.get("visible_cells", []),
                dtype=int,
            )

            if visible.size == 0:
                visible_by_robot[uid] = set()
                continue

            visible = visible.reshape(-1, 2)

            visible_by_robot[uid] = {
                (int(x), int(y))
                for x, y in visible
            }

        detections: dict[int, int] = {}

        for target_index, target in enumerate(targets):
            # Target truth is converted into a sensor-grid cell only here.
            target_cell = (
                int(round(float(target["x"]))),
                int(round(float(target["y"]))),
            )

            for uid in sorted(eligible):
                if target_cell in visible_by_robot.get(
                    uid,
                    set(),
                ):
                    detections[int(target_index)] = uid
                    break

        return detections
