from __future__ import annotations

from copy import deepcopy
from typing import Any

from .search_scenario import search_radii_profile


def prepare_gppo_config(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Return an isolated config aligned with the frozen GPPO protocol."""

    cfg = deepcopy(config)
    frozen = checkpoint["mixed_config"]

    physics_step_ms = (
        float(cfg["scenario"]["duration"].get("step_dt", 0.1))
        * 1000.0
    )
    control_step_ms = float(frozen["control_step_ms"])

    ratio = control_step_ms / physics_step_ms
    interval = int(round(ratio))

    if interval <= 0:
        raise ValueError("Invalid GPPO/physics step ratio")

    if abs(interval * physics_step_ms - control_step_ms) > 1e-6:
        raise RuntimeError(
            "Frozen GPPO control period cannot be represented "
            "by the configured physics timestep"
        )

    # 300 GPPO mission steps × 10 physics ticks = 3000 physics ticks.
    mission_steps = int(frozen["max_episode_steps"])
    cfg["scenario"]["duration"]["max_steps"] = (
        mission_steps * interval
    )

    # Frozen communication range.
    cfg.setdefault("communication", {})
    cfg["communication"].setdefault("params", {})
    cfg["communication"]["params"]["comm_range"] = float(
        frozen["comm_range"]
    )

    # Hidden target truth is segregated from the public task layer.
    hidden_targets = {}

    # Align task activation with the frozen Phase14 protocol.
    for task in cfg.get("tasks", []):
        activation = task.setdefault(
            "activation",
            {},
        )
        task_type = task.get("type")

        if task_type == "target_search":
            # Frozen maybe_activate() has no independent time gate.
            activation.pop("step", None)
            activation["condition"] = (
                "exploration_rate >= "
                f"{float(frozen['activation_threshold']):.12g}"
            )

        elif "step" in activation:
            # Non-Search legacy gates remain in mission-time units.
            activation["step"] = (
                int(activation["step"]) * interval
            )

        params = task.get("params", {})

        if task_type == "target_search":
            truth = list(params.pop("targets", []))

            hidden_targets[
                str(task["task_id"])
            ] = truth

            params["target_count"] = len(truth)

        # Keep any task deadline in the same physical mission time.
        if "deadline" in params:
            params["deadline"] = (
                int(params["deadline"]) * interval
            )

        if task.get("type") == "relay":
            params["comm_range"] = float(
                frozen["comm_range"]
            )
            params["min_connectivity"] = float(
                frozen["connectivity_threshold"]
            )

    # Dynamic-obstacle timing is also currently step-index based.
    #
    # spawn_step: multiply by 10
    # expansion_rate: divide by 10
    #
    # This preserves the same obstacle radius as a function of
    # physical mission time.
    for obstacle in cfg.get("dynamic_obstacles", []):
        obstacle["spawn_step"] = (
            int(obstacle.get("spawn_step", 0))
            * interval
        )

        params = obstacle.get("params", {})
        if "expansion_rate" in params:
            params["expansion_rate"] = (
                float(params["expansion_rate"])
                / interval
            )

    cfg["_gppo_hidden_targets"] = hidden_targets

    cfg.setdefault("sensor", {})
    cfg["sensor"].setdefault("params", {})
    cfg["sensor"]["params"]["los_occlusion"] = True

    cfg.setdefault("_gppo_profile", {})
    cfg["_gppo_profile"].update({
        "physics_step_ms": physics_step_ms,
        "control_step_ms": control_step_ms,
        "high_level_interval_steps": interval,
        "mission_steps": mission_steps,
        "physics_steps": mission_steps * interval,
        "comm_range": float(frozen["comm_range"]),
        "comm_max_hops": int(frozen["comm_max_hops"]),
        "comm_delay_ms": float(frozen["comm_delay_ms"]),
        "stale_age_fraction": (
            float(frozen["comm_delay_ms"])
            / control_step_ms
        ),
        "activation_threshold": float(
            frozen["activation_threshold"]
        ),
        # Exact frozen Search defaults.  The frozen Phase14-v9 worker runs
        # with test_parameter.SENSOR_RANGE = 10 m and CELL_SIZE = 0.4 m.
        "search_service_steps": 6,
        "search_sensor_range": 10.0,
        # Resolution of the team runtime occupancy grid, which is the
        # lattice the generated Search scenario lives on.
        "search_cell_size": 1.0,
        "search_survivor_count": 4,
        **search_radii_profile(
            sensor_range=10.0,
            cell_size=1.0,
        ),
    })

    return cfg
