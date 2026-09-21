from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProtocolAudit:
    ok: bool
    mismatches: tuple[str, ...]
    checkpoint: dict[str, Any]
    runtime: dict[str, Any]


def audit_runtime_protocol(
    runtime,
    checkpoint: dict[str, Any],
) -> ProtocolAudit:
    cfg = checkpoint["mixed_config"]

    ckpt_protocol = {
        "comm_range": float(cfg["comm_range"]),
        "comm_max_hops": int(cfg["comm_max_hops"]),
        "comm_delay_ms": float(cfg["comm_delay_ms"]),
        "control_step_ms": float(cfg["control_step_ms"]),
    }

    runtime_protocol = {
        "comm_range": float(runtime.comm.comm_range),
        "comm_delay_ms": (
            float(runtime.comm.delay_steps)
            * float(runtime.dt)
            * 1000.0
        ),
        "control_step_ms": float(runtime.dt) * 1000.0,

        # Current shared CommunicationModel does not yet expose
        # a <=N-hop protocol constraint.
        "comm_max_hops": None,
    }

    mismatches: list[str] = []

    if runtime_protocol["comm_range"] != ckpt_protocol["comm_range"]:
        mismatches.append(
            "comm_range: "
            f"runtime={runtime_protocol['comm_range']} "
            f"checkpoint={ckpt_protocol['comm_range']}"
        )

    if runtime_protocol["comm_delay_ms"] != ckpt_protocol["comm_delay_ms"]:
        mismatches.append(
            "comm_delay_ms: "
            f"runtime={runtime_protocol['comm_delay_ms']} "
            f"checkpoint={ckpt_protocol['comm_delay_ms']}"
        )

    if runtime_protocol["control_step_ms"] != ckpt_protocol["control_step_ms"]:
        mismatches.append(
            "control_step_ms: "
            f"runtime={runtime_protocol['control_step_ms']} "
            f"checkpoint={ckpt_protocol['control_step_ms']}"
        )

    if runtime_protocol["comm_max_hops"] != ckpt_protocol["comm_max_hops"]:
        mismatches.append(
            "comm_max_hops: "
            f"runtime={runtime_protocol['comm_max_hops']} "
            f"checkpoint={ckpt_protocol['comm_max_hops']}"
        )

    return ProtocolAudit(
        ok=not mismatches,
        mismatches=tuple(mismatches),
        checkpoint=ckpt_protocol,
        runtime=runtime_protocol,
    )
