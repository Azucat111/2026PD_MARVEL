from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .gppo_model import GPPOActorCritic


EXPECTED_PROTOCOL = "phase14-pretrain-v9"


def load_frozen_gppo(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[GPPOActorCritic, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"GPPO checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint dict, got {type(checkpoint).__name__}"
        )

    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model'")

    protocol = checkpoint.get("protocol_version")
    if protocol != EXPECTED_PROTOCOL:
        raise RuntimeError(
            f"Unexpected protocol_version={protocol!r}; "
            f"expected {EXPECTED_PROTOCOL!r}"
        )

    model = GPPOActorCritic().to(device)

    # Deliberately strict:
    # any model/checkpoint mismatch must fail immediately.
    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()

    return model, checkpoint
