from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GPPORuntimeProfile:
    comm_range: float
    comm_max_hops: int
    comm_delay_ms: float

    control_step_ms: float
    max_episode_steps: int

    physics_step_ms: float
    high_level_interval_steps: int

    mission_duration_ms: float
    required_physics_steps: int

    effective_control_step_ms: float

    stale_age_fraction: float

    warnings: tuple[str, ...]


def build_gppo_runtime_profile(
    runtime,
    checkpoint: dict[str, Any],
) -> GPPORuntimeProfile:
    cfg = checkpoint["mixed_config"]

    comm_range = float(cfg["comm_range"])
    comm_max_hops = int(cfg["comm_max_hops"])
    comm_delay_ms = float(cfg["comm_delay_ms"])

    control_step_ms = float(cfg["control_step_ms"])
    max_episode_steps = int(cfg["max_episode_steps"])

    physics_step_ms = float(runtime.dt) * 1000.0

    ratio = control_step_ms / physics_step_ms
    interval = int(round(ratio))

    if interval <= 0:
        raise ValueError("Invalid GPPO high-level interval")

    effective_control_step_ms = (
        interval * physics_step_ms
    )

    if abs(
        effective_control_step_ms - control_step_ms
    ) > 1e-6:
        raise RuntimeError(
            "GPPO high-level control period cannot be represented "
            "by the current physics timestep: "
            f"control={control_step_ms} ms, "
            f"physics={physics_step_ms} ms"
        )

    mission_duration_ms = (
        max_episode_steps * control_step_ms
    )

    required_physics_steps = int(round(
        mission_duration_ms / physics_step_ms
    ))

    if required_physics_steps <= 0:
        raise ValueError("Invalid required physics horizon")

    # Phase14 represents the 50 ms weak-communication age as
    # a sub-step stale-state interpolation.
    stale_age_fraction = (
        comm_delay_ms / physics_step_ms
    )

    warnings: list[str] = []

    if stale_age_fraction > 1.0:
        warnings.append(
            "Communication state age exceeds one physics step; "
            "a multi-step history buffer is required."
        )

    if runtime.max_steps != required_physics_steps:
        warnings.append(
            "Runtime horizon differs from frozen GPPO protocol: "
            f"runtime={runtime.max_steps} physics steps, "
            f"required={required_physics_steps} physics steps."
        )

    return GPPORuntimeProfile(
        comm_range=comm_range,
        comm_max_hops=comm_max_hops,
        comm_delay_ms=comm_delay_ms,

        control_step_ms=control_step_ms,
        max_episode_steps=max_episode_steps,

        physics_step_ms=physics_step_ms,
        high_level_interval_steps=interval,

        mission_duration_ms=mission_duration_ms,
        required_physics_steps=required_physics_steps,

        effective_control_step_ms=effective_control_step_ms,

        stale_age_fraction=stale_age_fraction,

        warnings=tuple(warnings),
    )


def apply_gppo_runtime_profile(
    runtime,
    profile: GPPORuntimeProfile,
) -> None:
    """Apply frozen protocol fields representable by the shared runtime."""

    runtime.comm.comm_range = float(
        profile.comm_range
    )

    runtime.max_steps = int(
        profile.required_physics_steps
    )


class GPPOControlClock:
    """Map 100 ms physics ticks to frozen 1 s GPPO ticks."""

    def __init__(
        self,
        profile: GPPORuntimeProfile,
    ):
        self.interval = int(
            profile.high_level_interval_steps
        )

    def should_update(
        self,
        physics_step: int,
    ) -> bool:
        return int(physics_step) % self.interval == 0

    def mission_step(
        self,
        physics_step: int,
    ) -> int:
        return int(physics_step) // self.interval
