from __future__ import annotations

from collections import deque

import numpy as np


def _graph_state(agent):
    """Return validated MARVEL observed graph state."""

    if agent is None:
        return None

    node_coords = getattr(
        agent,
        "node_coords",
        None,
    )

    adjacency = getattr(
        agent,
        "adjacent_matrix",
        None,
    )

    current_index = getattr(
        agent,
        "current_index",
        None,
    )

    if (
        node_coords is None
        or adjacency is None
        or current_index is None
    ):
        return None

    coords = np.asarray(
        node_coords,
        dtype=float,
    )

    adj = np.asarray(adjacency)

    n = len(coords)
    start = int(current_index)

    if (
        n == 0
        or start < 0
        or start >= n
        or adj.shape != (n, n)
    ):
        return None

    return coords, adj, start


def observed_next_hop(
    agent,
    goal,
):
    """Frozen Search/Relay observed-graph next hop.

    MARVEL encodes valid graph edges as zeros in adjacent_matrix.
    Destination = reachable observed node nearest the public goal.
    Routing = deterministic shortest-hop BFS.

    None means graph expansion is required; caller must retain
    MARVEL's exploration action.
    """

    state = _graph_state(agent)

    if state is None:
        return None

    coords, adj, start = state

    goal = np.asarray(
        goal,
        dtype=float,
    )[:2]

    n = len(coords)

    prev = np.full(
        n,
        -1,
        dtype=int,
    )

    seen = np.zeros(
        n,
        dtype=bool,
    )

    q = deque([start])
    seen[start] = True

    while q:
        u = q.popleft()

        for raw_v in np.flatnonzero(
            adj[u] == 0
        ):
            v = int(raw_v)

            if (
                v == u
                or v < 0
                or v >= n
                or seen[v]
            ):
                continue

            seen[v] = True
            prev[v] = u
            q.append(v)

    reachable = np.flatnonzero(seen)

    if reachable.size == 0:
        return None

    distance_to_goal = np.linalg.norm(
        coords[reachable, :2]
        - goal[None, :],
        axis=1,
    )

    dest = int(
        reachable[
            int(
                np.argmin(
                    distance_to_goal
                )
            )
        ]
    )

    # Current node already is the best observed node.
    # Frozen worker falls back to MARVEL exploration.
    if dest == start:
        return None

    v = dest

    while (
        prev[v] != -1
        and prev[v] != start
    ):
        v = int(prev[v])

    if prev[v] == -1:
        return None

    return (
        np.asarray(
            coords[v],
            dtype=float,
        ).copy(),
        int(v),
    )


def _dijkstra_observed(
    agent,
):
    """Frozen obstacle-aware observed graph distances."""

    state = _graph_state(agent)

    if state is None:
        return None

    coords, adj, start = state
    n = len(coords)

    dist = np.full(
        n,
        np.inf,
        dtype=float,
    )

    used = np.zeros(
        n,
        dtype=bool,
    )

    dist[start] = 0.0

    for _ in range(n):
        masked = np.where(
            used,
            np.inf,
            dist,
        )

        cur = int(
            np.argmin(masked)
        )

        if not np.isfinite(
            masked[cur]
        ):
            break

        used[cur] = True

        for raw_v in np.flatnonzero(
            adj[cur] == 0
        ):
            v = int(raw_v)

            if (
                v == cur
                or v < 0
                or v >= n
            ):
                continue

            weight = float(
                np.linalg.norm(
                    coords[v, :2]
                    - coords[cur, :2]
                )
            )

            alternative = (
                dist[cur]
                + weight
            )

            if alternative < dist[v]:
                dist[v] = alternative

    return coords, dist


def observed_travel_time_matrix(
    uav_states,
    subtasks,
    agent_by_uid,
):
    """Frozen v9 GPPO edge travel-time semantics.

    known observed shortest path
      + Euclidean residual through unknown space
      -------------------------------------------
                    UAV speed

    Falls back to direct Euclidean distance only when the
    MARVEL observed graph is unavailable.
    """

    n_t = len(subtasks)
    n_u = len(uav_states)

    out = np.zeros(
        (n_t, n_u),
        dtype=np.float32,
    )

    for ui, uav in enumerate(
        uav_states
    ):
        uid = int(uav.uav_id)

        agent = agent_by_uid.get(uid)

        graph_result = (
            _dijkstra_observed(agent)
        )

        for ti, task in enumerate(
            subtasks
        ):
            if task.position is None:
                out[ti, ui] = 0.0
                continue

            goal = np.asarray(
                task.position,
                dtype=float,
            )[:2]

            speed = max(
                float(uav.velocity),
                1e-6,
            )

            if graph_result is not None:
                coords, path_dist = (
                    graph_result
                )

                finite = np.isfinite(
                    path_dist
                )

                if np.any(finite):
                    residual = np.linalg.norm(
                        coords[:, :2]
                        - goal[None, :],
                        axis=1,
                    )

                    total = (
                        path_dist
                        + residual
                    )

                    best = float(
                        np.min(
                            total[
                                np.isfinite(total)
                            ]
                        )
                    )

                    out[ti, ui] = (
                        best / speed
                    )

                    continue

            position = np.asarray(
                uav.position,
                dtype=float,
            )[:2]

            out[ti, ui] = float(
                np.linalg.norm(
                    position - goal
                )
                / speed
            )

    return out
