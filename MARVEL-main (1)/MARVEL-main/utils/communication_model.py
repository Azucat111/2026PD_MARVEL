"""Communication range, connectivity and delayed message simulation."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable

import numpy as np


class CommunicationModel:
    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        params = config.get("params", config)
        self.comm_range = float(params.get("comm_range", params.get("range", 20.0)))
        self.delay_steps = int(params.get("delay_steps", 0))
        self.packet_loss = float(params.get("packet_loss", 0.0))
        self._queue: deque[tuple[int, int, Any]] = deque()

    def get_topology(self, positions: Iterable[np.ndarray]) -> np.ndarray:
        points = np.asarray(list(positions), dtype=float)
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        topology = (distances <= self.comm_range).astype(np.int8)
        np.fill_diagonal(topology, 0)
        return topology

    def check_connectivity(self, topology: np.ndarray) -> tuple[bool, int]:
        n = len(topology)
        seen = set()
        components = 0
        for start in range(n):
            if start in seen:
                continue
            components += 1
            stack = [start]
            seen.add(start)
            while stack:
                node = stack.pop()
                for neighbor in np.flatnonzero(topology[node] | topology[:, node]):
                    if int(neighbor) not in seen:
                        seen.add(int(neighbor))
                        stack.append(int(neighbor))
        return components <= 1, components

    def broadcast(self, sender: int, payload: Any, step: int,
                  recipients: Iterable[int] | None = None) -> None:
        for recipient in recipients or []:
            if recipient == sender or np.random.random() < self.packet_loss:
                continue
            self._queue.append((step + self.delay_steps, int(recipient), payload))

    def receive(self, recipient: int, step: int) -> list[Any]:
        ready, pending = [], deque()
        while self._queue:
            delivery_step, target, payload = self._queue.popleft()
            if delivery_step <= step and target == recipient:
                ready.append(payload)
            else:
                pending.append((delivery_step, target, payload))
        self._queue = pending
        return ready
