from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GPPORuntimeProfile:
    comm_range: float
    comm_max_hops: int
    comm_delay_ms: float
    control_step_ms: float

    physics_step_ms: float
    high_level_interval_steps: int
    effective_control_step_ms: float

    delay_exactly_representable: bool
    nearest_delay_steps: int
    nearest_delay_ms: float

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

    physics_step_ms = float(runtime.dt) * 1000.0

    ratio = control_step_ms / physics_step_ms
    interval = int(round(ratio))

    if interval <= 0:
        raise ValueError("Invalid GPPO high-level interval")

    effective_control_step_ms = (
        interval * physics_step_ms
    )

    if abs(effective_control_step_ms - control_step_ms) > 1e-6:
        raise RuntimeError(
            "GPPO control period is not representable by the "
            "current physics timestep: "
            f"requested={control_step_ms}ms "
            f"physics_dt={physics_step_ms}ms"
        )

    delay_ratio = comm_delay_ms / physics_step_ms
    nearest_delay_steps = max(0, int(round(delay_ratio)))
    nearest_delay_ms = nearest_delay_steps * physics_step_ms

    delay_exact = abs(
        nearest_delay_ms - comm_delay_ms
    ) <= 1e-6

    warnings: list[str] = []

    if not delay_exact:
        warnings.append(
            "checkpoint communication delay "
            f"{comm_delay_ms} ms cannot be represented exactly "
            f"with physics timestep {physics_step_ms} ms; "
            f"nearest integer-step delay is {nearest_delay_ms} ms"
        )

    return GPPORuntimeProfile(
        comm_range=comm_range,
        comm_max_hops=comm_max_hops,
        comm_delay_ms=comm_delay_ms,
        control_step_ms=control_step_ms,
        physics_step_ms=physics_step_ms,
        high_level_interval_steps=interval,
        effective_control_step_ms=effective_control_step_ms,
        delay_exactly_representable=delay_exact,
        nearest_delay_steps=nearest_delay_steps,
        nearest_delay_ms=nearest_delay_ms,
        warnings=tuple(warnings),
    )


def apply_gppo_runtime_profile(
    runtime,
    profile: GPPORuntimeProfile,
) -> None:
    """Apply only protocol fields that are exactly representable.

    Communication delay is intentionally left unchanged when the
    current physics timestep cannot represent the checkpoint value.
    """

    runtime.comm.comm_range = float(profile.comm_range)

    if profile.delay_exactly_representable:
        runtime.comm.delay_steps = int(
            profile.nearest_delay_steps
        )


class GPPOControlClock:
    """Gate GPPO inference to the frozen 1-second high-level period."""

    def __init__(
        self,
        profile: GPPORuntimeProfile,
    ):
        self.interval = int(
            profile.high_level_interval_steps
        )

    def should_update(self, physics_step: int) -> bool:
        return int(physics_step) % self.interval == 0
