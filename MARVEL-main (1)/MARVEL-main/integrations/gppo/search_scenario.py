"""Frozen Phase14-v9 Search scenario generator.

Authoritative source
--------------------
``/home/nick/MARVEL`` branch ``phase14-v9-freeze-20260921`` @ ``e867117``,
file ``utils/explore_search_coordinator.py``.

``_free_world_coordinates`` / ``_line_of_sight_free`` /
``_preset_heat_points`` / ``_preset_targets`` are ported with the exact
frozen RNG streams, candidate ordering and fallbacks.  The only intended
difference is the occupancy representation:

======================  ====================  ================
layer                   free                  occupied
======================  ====================  ================
frozen ``ground_truth`` ``255``               ``1``
team ``occupancy_grid`` ``0``                 ``1``
======================  ====================  ================

Everything else - the 8-connected start-connected flood fill, the
row-major candidate ordering, the ``default_rng(seed)`` /
``default_rng(seed + 1_000_003)`` stream separation, the 12000-candidate
cap, farthest-point sampling, the LOS filter and the "prefer fresh
candidate" rule - is source-faithful.

Privacy
-------
``SearchScenarioTruth`` is environment / sensor-truth state.  It owns the
hidden survivor coordinates and the ``target_index -> heat_id``
association.  Callers project it with :meth:`SearchScenarioTruth.public_heat_points`
and may hand *only* that projection to GPPO, the task graph or Search
routing.

The generator needs initial UAV positions, so it can only run after
``SimulationRuntime.reset()``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .event_sources import PublicHeatPoint


# Frozen ``test_parameter`` constants used by the Phase14-v9 worker.
FROZEN_SENSOR_RANGE = 10.0
FROZEN_SEARCH_SERVICE_STEPS = 6
FROZEN_SURVIVOR_COUNT = 4

# Frozen ``ExploreSearchCoordinator`` acceptance window.
MIN_SURVIVOR_COUNT = 3
MAX_SURVIVOR_COUNT = 5

# Frozen ``rng = np.random.default_rng(self.seed + 1_000_003)``.
TARGET_SEED_OFFSET = 1_000_003

# Frozen candidate cap before farthest-point sampling.
MAX_CANDIDATES = 12000

# 8-connected neighbourhood, exactly as in the frozen flood fill.
NEIGHBOURHOOD = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


@dataclass(frozen=True)
class FrozenSearchRadii:
    """Derived Search geometry, mirroring the frozen constructor."""

    sensor_range: float
    cell_size: float
    search_service_steps: int
    search_reach_radius: float
    guaranteed_hidden_radius: float
    heat_target_radius: float

    @classmethod
    def derive(
        cls,
        *,
        sensor_range: float = FROZEN_SENSOR_RANGE,
        cell_size: float = 1.0,
        search_service_steps: int = FROZEN_SEARCH_SERVICE_STEPS,
    ) -> "FrozenSearchRadii":
        """Frozen ExploreSearchCoordinator radii derivation.

        ``sensor_range = max(float(getattr(env, "sensor_range", 1.0)), 1e-6)``
        ``cell_size   = max(float(getattr(env, "cell_size", 1.0)), 1e-6)``
        ``search_reach_radius     = max(sensor_range * 0.35, cell_size)``
        ``guaranteed_hidden_radius= max(sensor_range - search_reach_radius - cell_size, 0.0)``
        ``heat_target_radius      = min(sensor_range * 0.40, guaranteed_hidden_radius)``

        ``cell_size`` is the team runtime occupancy resolution, which is the
        grid the generated scenario actually lives on.
        """

        sensor_range = max(float(sensor_range), 1e-6)
        cell_size = max(float(cell_size), 1e-6)

        reach = max(sensor_range * 0.35, cell_size)
        guaranteed = max(sensor_range - reach - cell_size, 0.0)

        return cls(
            sensor_range=sensor_range,
            cell_size=cell_size,
            search_service_steps=max(1, int(search_service_steps)),
            search_reach_radius=float(reach),
            guaranteed_hidden_radius=float(guaranteed),
            heat_target_radius=float(
                min(sensor_range * 0.40, guaranteed)
            ),
        )


def search_radii_profile(
    *,
    sensor_range: float = FROZEN_SENSOR_RANGE,
    cell_size: float = 1.0,
    search_service_steps: int = FROZEN_SEARCH_SERVICE_STEPS,
) -> dict[str, float]:
    """Expose the derived frozen radii as GPPO profile entries.

    Keeps ``_gppo_profile`` and the generator on one implementation of the
    frozen formulas.
    """

    radii = FrozenSearchRadii.derive(
        sensor_range=sensor_range,
        cell_size=cell_size,
        search_service_steps=search_service_steps,
    )

    return {
        "search_reach_radius": radii.search_reach_radius,
        "search_guaranteed_hidden_radius": (
            radii.guaranteed_hidden_radius
        ),
        "search_heat_target_radius": radii.heat_target_radius,
    }


@dataclass(frozen=True)
class HiddenSurvivor:
    """PRIVATE ground truth. Never leaves the environment layer."""

    target_index: int
    heat_id: int
    position: tuple[float, float]


class FrozenSearchScenarioGenerator:
    """Source-faithful port of the frozen scenario generator.

    Operates on the team runtime occupancy grid:

    * ``occupancy_grid[row, col]`` with ``0 = free`` and ``1 = occupied``;
    * world ``x = col * cell_size``, world ``y = row * cell_size``.

    The ``col * cell_size`` corner convention is deliberate: the frozen
    generator uses the same convention, and the LOS / flood-fill round
    trips depend on it.
    """

    def __init__(
        self,
        *,
        occupancy_grid,
        robot_locations,
        seed: int,
        survivor_count: int = FROZEN_SURVIVOR_COUNT,
        radii: FrozenSearchRadii | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
    ):
        grid = np.asarray(occupancy_grid)

        if grid.ndim != 2:
            raise ValueError(
                "occupancy_grid must be two-dimensional"
            )

        if not (
            MIN_SURVIVOR_COUNT
            <= int(survivor_count)
            <= MAX_SURVIVOR_COUNT
        ):
            raise ValueError(
                "project TARGET_SEARCH protocol requires "
                "3-5 heat points/survivors"
            )

        self.occupancy_grid = grid

        # Frozen free predicate: ground_truth == 255.  Team runtime grid
        # encodes the same set as 0.
        self.free = grid == 0

        self.robot_locations = np.asarray(
            robot_locations, dtype=float
        ).reshape(-1, 2)

        self.seed = int(seed)
        self.survivor_count = int(survivor_count)
        self.radii = radii or FrozenSearchRadii.derive(
            cell_size=1.0
        )
        # World origin of the occupancy lattice.  The extended runtime uses
        # (0, 0); original MARVEL geometry uses a belief origin that is
        # generally non-zero.
        self.origin = (
            float(origin[0]),
            float(origin[1]),
        )

    # ------------------------------------------------------------------
    # Cell <-> world helpers
    # ------------------------------------------------------------------
    @property
    def cell_size(self) -> float:
        return float(self.radii.cell_size)

    def world_to_cell(
        self,
        position,
    ) -> tuple[int, int]:
        """Return ``(col, row)`` for a world position."""

        point = np.asarray(position, dtype=float)

        col = int(
            round(
                (float(point[0]) - self.origin[0])
                / self.cell_size
            )
        )
        row = int(
            round(
                (float(point[1]) - self.origin[1])
                / self.cell_size
            )
        )

        return col, row

    def cell_to_world(
        self,
        col: int,
        row: int,
    ) -> tuple[float, float]:
        return (
            self.origin[0] + float(col) * self.cell_size,
            self.origin[1] + float(row) * self.cell_size,
        )

    def is_free_cell(
        self,
        col: int,
        row: int,
    ) -> bool:
        height, width = self.free.shape

        if not (0 <= col < width and 0 <= row < height):
            return False

        return bool(self.free[row, col])

    # ------------------------------------------------------------------
    # Frozen: _free_world_coordinates
    # ------------------------------------------------------------------
    def _free_world_coordinates(self) -> np.ndarray:
        """Return free cells reachable from at least one initial UAV start.

        Exact frozen algorithm: seed the flood fill at every initial robot
        cell that lands on a free cell, expand 8-connected, then emit the
        visited cells in row-major order as world coordinates.

        Frozen uses ``ground_truth == 255``; this port uses the team
        runtime occupancy grid's free value (``0``).
        """

        free = self.free

        if not np.any(free):
            raise ValueError(
                "ground-truth map contains no free cells"
            )

        height, width = free.shape
        seen = np.zeros_like(free, dtype=bool)
        queue = deque()

        for loc in self.robot_locations:
            col, row = self.world_to_cell(loc)

            # Frozen silently skips start cells that are out of bounds or
            # occupied rather than failing the episode.
            if (
                0 <= col < width
                and 0 <= row < height
                and free[row, col]
                and not seen[row, col]
            ):
                seen[row, col] = True
                queue.append((row, col))

        # 8-connected flood fill matches MARVEL's diagonal-capable spatial
        # graph more closely than a 4-neighbour component test.
        while queue:
            row, col = queue.popleft()

            for d_row, d_col in NEIGHBOURHOOD:
                next_row = row + d_row
                next_col = col + d_col

                if (
                    0 <= next_col < width
                    and 0 <= next_row < height
                    and free[next_row, next_col]
                    and not seen[next_row, next_col]
                ):
                    seen[next_row, next_col] = True
                    queue.append((next_row, next_col))

        free_rows_cols = np.argwhere(seen)

        if free_rows_cols.size == 0:
            raise ValueError(
                "no start-connected free cells available for "
                "Search scenario"
            )

        # ``argwhere`` is row-major, so the candidate ordering is identical
        # to the frozen generator's.
        cols = free_rows_cols[:, 1].astype(float)
        rows = free_rows_cols[:, 0].astype(float)

        return np.stack(
            [
                self.origin[0] + cols * self.cell_size,
                self.origin[1] + rows * self.cell_size,
            ],
            axis=1,
        )

    # ------------------------------------------------------------------
    # Frozen: _line_of_sight_free
    # ------------------------------------------------------------------
    def _line_of_sight_free(
        self,
        start_xy,
        end_xy,
    ) -> bool:
        """Ground-truth scenario-generation LOS check; never exposed to policy.

        Frozen samples ``max(2, ceil(dist / (cell_size * 0.5)) + 1)`` points
        via ``np.linspace`` (both endpoints inclusive) and requires every
        sampled cell to be in bounds and free.
        """

        start = np.asarray(start_xy, dtype=float)[:2]
        end = np.asarray(end_xy, dtype=float)[:2]

        distance = float(np.linalg.norm(end - start))
        step = max(self.cell_size * 0.5, 1e-6)
        count = max(2, int(np.ceil(distance / step)) + 1)

        points = np.linspace(start, end, count)

        for point in points:
            col, row = self.world_to_cell(point)

            if not self.is_free_cell(col, row):
                return False

        return True

    # ------------------------------------------------------------------
    # Frozen: _preset_heat_points
    # ------------------------------------------------------------------
    def _preset_heat_points(self) -> np.ndarray:
        """Create 3-5 deterministic public heat points, independent of targets.

        Returns the chosen world coordinates in ``heat_id`` order.  The
        caller wraps them into the *public* projection; this method returns
        bare coordinates so that nothing target-shaped can leak.
        """

        coords = self._free_world_coordinates()
        rng = np.random.default_rng(self.seed)
        robot_locations = np.asarray(
            self.robot_locations, dtype=float
        )

        if robot_locations.size:
            distances = np.linalg.norm(
                coords[:, None, :] - robot_locations[None, :, :],
                axis=-1,
            )

            clearance = max(
                float(self.radii.sensor_range) * 1.25, 1.0
            )

            filtered = coords[
                np.min(distances, axis=1) > clearance
            ]

            # Only adopt the filtered set when it can still satisfy the
            # protocol; otherwise keep the unfiltered candidates.
            if len(filtered) >= self.survivor_count:
                coords = filtered

        # Bound the farthest-point calculation on very large maps without
        # changing determinism.
        if len(coords) > MAX_CANDIDATES:
            subset = rng.choice(
                len(coords),
                size=MAX_CANDIDATES,
                replace=False,
            )
            coords = coords[np.sort(subset)]

        first = int(rng.integers(0, len(coords)))
        chosen = [first]

        while len(chosen) < self.survivor_count:
            chosen_coords = coords[np.asarray(chosen, dtype=int)]

            distances = np.linalg.norm(
                coords[:, None, :] - chosen_coords[None, :, :],
                axis=-1,
            )

            score = np.min(distances, axis=1)
            score[np.asarray(chosen, dtype=int)] = -np.inf

            chosen.append(int(np.argmax(score)))

        return np.asarray(
            [coords[index] for index in chosen], dtype=float
        )

    # ------------------------------------------------------------------
    # Frozen: _preset_targets
    # ------------------------------------------------------------------
    def _preset_targets(
        self,
        heat_positions: Sequence[np.ndarray],
    ) -> list[tuple[int, tuple[float, float]]]:
        """Sample one hidden survivor near each already-public heat point.

        Returns ``(heat_id, world_position)`` pairs in heat-point order.
        """

        coords = self._free_world_coordinates()

        # Separate stream: changing hidden sampling cannot move public heat
        # points.
        rng = np.random.default_rng(
            self.seed + TARGET_SEED_OFFSET
        )

        used: set[int] = set()
        targets: list[tuple[int, tuple[float, float]]] = []

        for heat_id, heat_position in enumerate(heat_positions):
            heat_position = np.asarray(
                heat_position, dtype=float
            )

            distances = np.linalg.norm(
                coords - heat_position[None, :], axis=1
            )

            if self.radii.heat_target_radius > 0.0:
                candidate_idx = np.flatnonzero(
                    distances
                    <= self.radii.heat_target_radius + 1e-9
                )
            else:
                candidate_idx = np.asarray(
                    [int(np.argmin(distances))], dtype=int
                )

            if candidate_idx.size == 0:
                candidate_idx = np.asarray(
                    [int(np.argmin(distances))], dtype=int
                )

            # Keep hidden truth physically sensor-reachable from the public
            # heat point; Euclidean-near cells across a wall are rejected.
            visible_idx = [
                int(index)
                for index in candidate_idx
                if self._line_of_sight_free(
                    heat_position, coords[int(index)]
                )
            ]

            if visible_idx:
                candidate_idx = np.asarray(
                    visible_idx, dtype=int
                )
            else:
                # The nearest cell is the frozen fallback.  The heat point
                # itself is reachable free space, so this stays admissible.
                candidate_idx = np.asarray(
                    [int(np.argmin(distances))], dtype=int
                )

            fresh = [
                int(index)
                for index in candidate_idx
                if int(index) not in used
            ]

            pool = (
                fresh
                if fresh
                else [int(index) for index in candidate_idx]
            )

            index = int(
                pool[int(rng.integers(0, len(pool)))]
            )
            used.add(index)

            targets.append(
                (
                    int(heat_id),
                    self.cell_to_world(
                        *self.world_to_cell(coords[index])
                    ),
                )
            )

        return targets

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def generate(self) -> tuple[list[tuple[float, float]], list[tuple[int, tuple[float, float]]]]:
        """Return ``(heat_positions, targets)`` with heat points first.

        Heat points are produced strictly before hidden truth, mirroring the
        frozen constructor ordering.
        """

        heat_positions = self._preset_heat_points()

        return heat_positions, self._preset_targets(heat_positions)


class SearchScenarioTruth:
    """PRIVATE environment-layer Search scenario state.

    Owns the hidden survivor coordinates and the ``target_index -> heat_id``
    association.  The only public projection is
    :meth:`public_heat_points`; :meth:`completed_heat_ids` is the abstract
    completion notification the scheduler is allowed to consume.
    """

    def __init__(
        self,
        *,
        task_id: str,
        heat_points: Iterable[PublicHeatPoint],
        survivors: Iterable[HiddenSurvivor],
        radii: FrozenSearchRadii,
        origin: tuple[float, float] = (0.0, 0.0),
    ):
        self.task_id = str(task_id)
        self.radii = radii
        self.origin = (float(origin[0]), float(origin[1]))

        self._heat_points = tuple(heat_points)
        self._survivors = tuple(survivors)

        self._heat_id_by_target: dict[int, int] = {}

        for survivor in self._survivors:
            target_index = int(survivor.target_index)

            if target_index in self._heat_id_by_target:
                raise ValueError(
                    "Duplicate hidden target index "
                    f"{target_index}"
                )

            self._heat_id_by_target[target_index] = int(
                survivor.heat_id
            )

    @property
    def survivors(self) -> tuple[HiddenSurvivor, ...]:
        return self._survivors

    @property
    def heat_points(self) -> tuple[PublicHeatPoint, ...]:
        return self._heat_points

    @property
    def target_count(self) -> int:
        return len(self._survivors)

    def public_heat_points(self) -> list[PublicHeatPoint]:
        """Return the public projection handed to GPPO / the scheduler."""

        return list(self._heat_points)

    def detector_targets(self) -> list[dict[str, Any]]:
        """Encode hidden truth in ``TargetDetector``'s grid-cell format.

        This is the bridge into the sensor layer and is the only place a
        hidden survivor coordinate is converted.
        """

        return [
            {
                "x": int(
                    round(
                        (
                            float(survivor.position[0])
                            - self.origin[0]
                        )
                        / self.radii.cell_size
                    )
                ),
                "y": int(
                    round(
                        (
                            float(survivor.position[1])
                            - self.origin[1]
                        )
                        / self.radii.cell_size
                    )
                ),
            }
            for survivor in self._survivors
        ]

    def heat_id_for_target(
        self,
        target_index: int,
    ) -> int | None:
        """PRIVATE association lookup.  Environment layer only."""

        return self._heat_id_by_target.get(
            int(target_index)
        )

    def completed_heat_ids(
        self,
        target_indices: Iterable[int],
    ) -> set[int]:
        """Map detected target indices onto the completed heat ids.

        This is the abstract completion notification: the caller learns
        *which heat point finished*, never where the survivor was.
        """

        completed: set[int] = set()

        for target_index in target_indices:
            heat_id = self._heat_id_by_target.get(
                int(target_index)
            )

            if heat_id is None:
                raise ValueError(
                    "Detected hidden target has no heat-point "
                    f"association: target_index={target_index}"
                )

            completed.add(int(heat_id))

        return completed


def resolve_search_scenario_task_id(runtime) -> str:
    """Return the single ``target_search`` task id of the runtime."""

    search_tasks = [
        task
        for task in runtime.tasks.tasks.values()
        if task.task_type == "target_search"
    ]

    if not search_tasks:
        raise RuntimeError(
            "Frozen Search scenario requires a target_search task."
        )

    if len(search_tasks) > 1:
        raise RuntimeError(
            "Current GPPO integration expects one "
            "target_search task."
        )

    return str(search_tasks[0].task_id)


def resolve_search_scenario_seed(runtime) -> int:
    """Deterministic scenario seed, independent of the RNG global state."""

    scheduler_config = runtime.config.get(
        "task_scheduler", {}
    ) or {}

    if "search_scenario_seed" in scheduler_config:
        return int(scheduler_config["search_scenario_seed"])

    scenario_config = runtime.config.get("scenario", {}) or {}

    if "random_seed" in scenario_config:
        return int(scenario_config["random_seed"])

    return 0


def resolve_search_scenario_profile(runtime) -> dict[str, Any]:
    """Frozen Search constants, taken from the prepared GPPO profile."""

    profile = runtime.config.get(
        "_gppo_profile", {}
    ) or {}

    return {
        "sensor_range": float(
            profile.get(
                "search_sensor_range",
                FROZEN_SENSOR_RANGE,
            )
        ),
        "cell_size": float(
            profile.get("search_cell_size", 1.0)
        ),
        "search_service_steps": int(
            profile.get(
                "search_service_steps",
                FROZEN_SEARCH_SERVICE_STEPS,
            )
        ),
        "survivor_count": int(
            profile.get(
                "search_survivor_count",
                FROZEN_SURVIVOR_COUNT,
            )
        ),
    }


def initialize_search_scenario(
    runtime,
    *,
    seed: int | None = None,
    survivor_count: int | None = None,
) -> SearchScenarioTruth:
    """Generate and install the frozen Search scenario for one episode.

    Must run after ``SimulationRuntime.reset()``: the frozen generator seeds
    its flood fill from the initial UAV positions.

    Returns the PRIVATE truth.  The caller is expected to hand only
    ``truth.public_heat_points()`` to the high-level layer.
    """

    if not runtime.robots:
        raise RuntimeError(
            "Frozen Search scenario requires initial UAV positions; "
            "call SimulationRuntime.reset() first."
        )

    task_id = resolve_search_scenario_task_id(runtime)

    if seed is None:
        seed = resolve_search_scenario_seed(runtime)

    profile = resolve_search_scenario_profile(runtime)

    if survivor_count is None:
        survivor_count = profile["survivor_count"]

    grid = np.asarray(
        runtime.obstacles.get_occupancy_grid()
    )

    # The runtime occupancy grid is authoritative for geometry: the
    # extended environment is a 1 m lattice at origin (0, 0), the
    # MARVEL-native environment a 0.4 m lattice at the belief origin.
    frame = runtime.obstacles.frame

    origin = frame.origin

    radii = FrozenSearchRadii.derive(
        sensor_range=profile["sensor_range"],
        cell_size=frame.cell_size,
        search_service_steps=profile["search_service_steps"],
    )

    heat_points, survivors, radii = build_search_scenario(
        occupancy_grid=grid,
        robot_locations=[
            np.asarray(robot.position, dtype=float)
            for robot in runtime.robots
        ],
        seed=int(seed),
        survivor_count=int(survivor_count),
        radii=radii,
        origin=origin,
    )

    truth = SearchScenarioTruth(
        task_id=task_id,
        heat_points=heat_points,
        survivors=survivors,
        radii=radii,
        origin=origin,
    )

    runtime.install_search_scenario(truth)

    # The generated survivor count is authoritative for task progress.
    task = runtime.tasks.tasks[task_id]

    task.params["target_count"] = int(truth.target_count)

    return truth


def build_search_scenario(
    *,
    occupancy_grid,
    robot_locations,
    seed: int,
    survivor_count: int = FROZEN_SURVIVOR_COUNT,
    radii: FrozenSearchRadii | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[list[PublicHeatPoint], list[HiddenSurvivor], FrozenSearchRadii]:
    """Generate a frozen Phase14-v9 Search scenario.

    Heat points are generated and returned before any hidden survivor
    exists, so the public/private split is structural rather than
    conventional.
    """

    resolved = radii or FrozenSearchRadii.derive(cell_size=1.0)

    generator = FrozenSearchScenarioGenerator(
        occupancy_grid=occupancy_grid,
        robot_locations=robot_locations,
        seed=int(seed),
        survivor_count=int(survivor_count),
        radii=resolved,
        origin=origin,
    )

    heat_positions, targets = generator.generate()

    heat_points = [
        PublicHeatPoint(
            heat_id=index,
            position=(
                float(position[0]),
                float(position[1]),
            ),
            confidence=1.0,
        )
        for index, position in enumerate(heat_positions)
    ]

    survivors = [
        HiddenSurvivor(
            target_index=index,
            heat_id=int(heat_id),
            position=(
                float(position[0]),
                float(position[1]),
            ),
        )
        for index, (heat_id, position) in enumerate(targets)
    ]

    return heat_points, survivors, resolved
