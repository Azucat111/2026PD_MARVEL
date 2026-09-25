"""Unified headless simulation runtime for multi-task UAV scenarios."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .communication_model import CommunicationModel
from .dynamics_models import create_dynamics_model
from .geometry import GEOMETRY_MODE_NATIVE, resolve_geometry_mode
from .marvel_motion import (
    FROZEN_YAW_RATE,
    compute_allowable_heading,
    interpolated_sensing_track,
)
from .marvel_sensing import FROZEN_NUM_SIM_STEPS, MarvelNativeBelief
from .obstacle_manager import ObstacleManager
from .safety_shield import SafetyShield
from .sensor_models import create_sensor_model
from .task_manager import TaskManager
from .target_detector import TargetDetector


@dataclass
class RobotState:
    robot_id: int
    robot_type: str
    position: np.ndarray
    velocity: float
    heading: float
    angular_velocity: float = 0.0
    travel_distance: float = 0.0

    # High-level task state, mirroring the frozen MARVEL Agent attributes
    # (utils/agent.py:79-90) that the frozen TaskManager exposes to GPPO via
    # UAVState.  These persist across high-level decisions; they are NOT
    # rebuilt per graph.
    #
    #   available         False while the UAV holds a high-level assignment
    #   busy_until        current_time + travel + processing at assignment
    #   utilization       never updated by the frozen source; stays 0.0
    #   assigned_task_num incremented by every apply_assignment
    #   current_task      EXPLORATION between assignments
    available: bool = True
    busy_until: float = 0.0
    utilization: float = 0.0
    assigned_task_num: int = 0
    current_task: str = "exploration"

    # Frozen Agent.yaw_rate, used by compute_allowable_heading.
    yaw_rate: float = FROZEN_YAW_RATE


class SimulationRuntime:
    def __init__(self, scenario_config: Dict[str, Any]):
        self.config = scenario_config
        scenario = scenario_config["scenario"]
        self.max_steps = int(scenario["duration"]["max_steps"])
        self.dt = float(scenario["duration"].get("step_dt", 0.1))
        self.current_step = 0
        self.dynamics = create_dynamics_model(scenario_config["dynamics"]["params"])
        self.sensor = create_sensor_model(scenario_config["sensor"]["params"])
        self.dynamics_by_type = {}
        self.sensor_by_type = {}
        self.comm = CommunicationModel(scenario_config["communication"])
        environment = scenario_config["environment"]
        self.geometry_mode = resolve_geometry_mode(environment)
        self.obstacles = ObstacleManager(environment, scenario_config.get("dynamic_obstacles", []))
        self._load_environment_map(environment)
        self.tasks = TaskManager(scenario_config["tasks"])

        hidden_targets = scenario_config.get(
            "_gppo_hidden_targets"
        )

        # Backward-compatible truth extraction for ordinary configs.
        if hidden_targets is None:
            hidden_targets = {
                str(task["task_id"]):
                    list(
                        task.get(
                            "params", {}
                        ).get("targets", [])
                    )
                for task in scenario_config.get(
                    "tasks", []
                )
                if task.get("type") == "target_search"
            }

        self.target_detector = TargetDetector(
            hidden_targets
        )

        # Per-episode baseline for the detector, so reset() can drop any
        # generator-installed truth without losing config-declared targets.
        self._baseline_hidden_targets = {
            str(task_id): list(entries)
            for task_id, entries in (
                self.target_detector.hidden_targets.items()
            )
        }

        # PRIVATE Search scenario truth (hidden survivor coordinates plus
        # the target_index -> heat_id association), installed by the
        # environment-side scenario adapter after reset().  None until then,
        # and cleared on every reset so no episode leaks into the next.
        self.search_scenario = None

        # UAVs currently holding a Search assignment.  The frozen worker
        # gates detection on ``robot.current_task == TARGET_SEARCH``, so
        # this is the equivalent robot-level state published by the
        # high-level scheduler.  Empty when no GPPO scheduler is attached.
        self.active_search_uav_ids: set[int] = set()

        # The default sensor shares the environment frame.
        self.sensor.frame = self.obstacles.frame

        self.shield = SafetyShield(self.obstacles)
        self.robots: List[RobotState] = []
        self.events: list[Dict[str, Any]] = []
        self.explored_cells: set[tuple[int, int]] = set()
        self.free_cell_count = max(1, self._exploration_denominator())
        self._free_mask_cache: np.ndarray | None = None
        self._free_cell_total = 0

        # Authoritative exploration state for marvel_native: the frozen
        # MARVEL robot_belief array.  None in extended mode.
        self.native_belief = None
        self._mission_start_positions: list[np.ndarray] = []
        self._mission_start_headings: list[float] = []

    def _load_environment_map(self, environment: Dict[str, Any]) -> None:
        """Load the occupancy map in the configured geometry mode."""

        if self.geometry_mode == GEOMETRY_MODE_NATIVE:
            map_dir = environment.get("map_dir")

            if not map_dir:
                raise ValueError(
                    "environment.map_dir is required when "
                    "geometry_mode='marvel_native'"
                )

            self.obstacles.load_marvel_native(
                map_dir,
                int(environment.get("episode_index", 0)),
            )
            return

        map_file = environment.get("map_file")

        if map_file:
            self.obstacles.load_from_file(map_file)

    def _exploration_denominator(self) -> int:
        """Cell count for the exploration rate.

        Cell-based whenever a real map backs the grid.  The synthetic
        obstacle path keeps its historical metres-squared denominator so
        existing scenarios are unchanged.
        """

        if self.obstacles.has_occupancy_map:
            return self.obstacles.cell_count

        return int(
            self.obstacles.width * self.obstacles.height
        )

    def reset(self) -> Dict[int, Dict[str, Any]]:
        self.current_step = 0
        self.events = []
        self.explored_cells = set()
        self.obstacles.reset()
        self.tasks.reset()
        # Drop the previous episode's hidden truth. The scenario adapter
        # reinstalls it once the new initial UAV positions exist, and until
        # then no detection may resolve against the old episode.
        self.search_scenario = None
        self.active_search_uav_ids = set()
        self._free_mask_cache = None
        self._free_cell_total = 0
        self.native_belief = None
        self._mission_start_positions = []
        self._mission_start_headings = []
        self.target_detector.hidden_targets = {
            str(task_id): list(entries)
            for task_id, entries in (
                self._baseline_hidden_targets.items()
            )
        }
        self.robots = []
        ensure_connected = bool(self.config.get("communication", {}).get(
            "ensure_initial_connectivity", False))
        all_positions: list[np.ndarray] = []
        for group in self.config["robots"]:
            start, end = group["id_range"]
            positions = self._generate_initial_positions(
                end - start + 1,
                group.get("config", {}).get("initial_positions", "random_safe"),
                anchors=all_positions if ensure_connected else None,
            )
            all_positions.extend(positions)
            for offset, robot_id in enumerate(range(start, end + 1)):
                cfg = group.get("config", {})
                robot_type = group.get("type", "uav")
                self.dynamics_by_type[robot_type] = create_dynamics_model(
                    {**self.config["dynamics"]["params"], "params": {
                        **self.config["dynamics"]["params"].get("params", {}),
                        **{key: cfg[key] for key in ("velocity", "yaw_rate") if key in cfg},
                    }})
                self.sensor_by_type[robot_type] = create_sensor_model(
                    {**self.config["sensor"]["params"],
                     "fov": cfg.get("fov", self.config["sensor"]["params"].get("fov", 120.0)),
                     "range": cfg.get("sensor_range", self.config["sensor"]["params"].get("range", 10.0))},
                    frame=self.obstacles.frame)
                self.robots.append(RobotState(
                    robot_id=robot_id,
                    robot_type=robot_type,
                    position=positions[offset],
                    velocity=float(cfg.get("velocity", 0.0)),
                    heading=self._initial_heading(cfg),
                    yaw_rate=float(cfg.get("yaw_rate", FROZEN_YAW_RATE)),
                ))
        self._log_event("simulation_reset", {"num_robots": len(self.robots)})

        # Native mode: frozen belief map + initial sensing sweep.
        if self.geometry_mode == GEOMETRY_MODE_NATIVE:
            self._initialize_native_belief()

        self._capture_mission_start()

        observations = self._get_observations()
        self._update_explored_cells(observations)
        return observations

    def _initial_heading(self, robot_config: Dict[str, Any]) -> float:
        """Initial heading.

        Frozen ``ScenarioEnv`` uses ``np.full(n_agents, 270.0)`` unless
        ``randomize_headings`` is set, so native mode reproduces that
        exactly.  The extended environment keeps its random heading, which
        is also what keeps its RNG stream unchanged.
        """

        if self.geometry_mode != GEOMETRY_MODE_NATIVE:
            return float(np.random.uniform(0.0, 360.0))

        environment = self.config.get("environment", {}) or {}

        headings = environment.get("initial_headings")

        if headings is None:
            return 270.0

        return float(headings) % 360.0

    def _apply_native_motion(self, actions) -> None:
        """Frozen MARVEL motion: teleport to the commanded waypoint.

        ``Env.final_sim_step`` assigns the waypoint directly, so there is no
        kinematic integration.  The heading is the only modelled quantity,
        produced by ``compute_allowable_heading``.
        """

        for robot, (waypoint, heading) in zip(self.robots, actions):
            start = np.asarray(robot.position, dtype=float)
            end = np.asarray(waypoint, dtype=float)

            final_heading = compute_allowable_heading(
                start,
                end,
                float(robot.heading),
                float(heading),
                max(float(robot.velocity), 1e-6),
                max(float(robot.yaw_rate), 1e-6),
            )

            robot.heading = float(final_heading)
            robot.position = end.copy()
            robot.travel_distance += float(
                np.linalg.norm(end - start)
            )

    def _capture_mission_start(self) -> None:
        """Record positions/headings at the start of a mission step."""

        self._mission_start_positions = [
            np.asarray(robot.position, dtype=float).copy()
            for robot in self.robots
        ]
        self._mission_start_headings = [
            float(robot.heading) for robot in self.robots
        ]

    def step(self, actions: List[Tuple[np.ndarray, float]]) -> tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
        if len(actions) != len(self.robots):
            raise ValueError(f"Expected {len(self.robots)} actions, got {len(actions)}")
        for obstacle_event in self.obstacles.step(self.current_step, self.dt):
            event_type = obstacle_event.pop("type")
            self._log_event(event_type, obstacle_event)
        native_motion = self.geometry_mode == GEOMETRY_MODE_NATIVE

        if not native_motion:
            # Apply safety shield before passing actions to dynamics.
            actions = self.shield.filter_actions(self.robots, actions, self.current_step)
            for shield_event in self.shield.pop_events():
                self._log_event("safety_shield", shield_event)
        else:
            # Frozen MARVEL has no shield step in its motion path; the
            # GPPO layer applies the equivalent hazard guard instead.
            self.shield.pop_events()
        info = {
            "collisions": [],
            "comm_topology": None,
            "connectivity_ratio": 0.0,
            "task_events": [],
            "feasibility": [],
            "terminated": False,
            "truncated": False,
        }
        if native_motion:
            # Frozen MARVEL motion runs once per mission step: the robot is
            # assigned the commanded waypoint outright, and the heading is
            # the only modelled quantity.
            if (
                self.current_step + 1
            ) % self._mission_step_interval() == 0:
                self._apply_native_motion(actions)
        else:
            for robot, (target_position, target_heading) in zip(self.robots, actions):
                old_position = robot.position.copy()
                dynamics = self.dynamics_by_type.get(robot.robot_type, self.dynamics)
                new_state, feasibility = dynamics.step(
                    current_position=robot.position,
                    final_position=np.asarray(target_position, dtype=float),
                    theta_current=robot.heading,
                    theta_desired=float(target_heading),
                    v_current=robot.velocity,
                    dt=self.dt,
                )
                info["feasibility"].append({"robot_id": robot.robot_id, **feasibility})
                collision, collision_type = self.obstacles.check_collision(new_state["position"], radius=0.2)
                if collision:
                    event = {"robot_id": robot.robot_id, "type": collision_type, "step": self.current_step,
                             "position": np.asarray(new_state["position"]).round(3).tolist()}
                    info["collisions"].append(event)
                    self._log_event("collision", event)
                    continue
                robot.position = np.asarray(new_state["position"], dtype=float)
                robot.velocity = float(new_state["velocity"])
                robot.heading = float(new_state["heading"])
                robot.angular_velocity = float(new_state.get("angular_velocity", 0.0))
                robot.travel_distance += float(np.linalg.norm(robot.position - old_position))

        # Detect UAV-UAV overlap after all proposed states have been applied.
        for index, first in enumerate(self.robots):
            for second in self.robots[index + 1:]:
                if np.linalg.norm(first.position - second.position) < 0.4:
                    event = {"robot_ids": [first.robot_id, second.robot_id],
                             "type": "uav_uav", "step": self.current_step}
                    info["collisions"].append(event)
                    self._log_event("collision", event)

        # Frozen sensing runs once per mission step, not per physics tick.
        if (
            self.geometry_mode == GEOMETRY_MODE_NATIVE
            and (self.current_step + 1) % self._mission_step_interval() == 0
        ):
            self._update_native_belief()
            self._capture_mission_start()

        observations = self._get_observations()
        self._update_explored_cells(observations)
        topology = self.comm.get_topology([robot.position for robot in self.robots])
        info["comm_topology"] = topology
        connected, components = self.comm.check_connectivity(topology)
        connectivity_ratio = self.comm.connectivity_ratio(topology)
        info["connectivity_ratio"] = connectivity_ratio
        if not connected:
            self._log_event("disconnection", {
                "step": self.current_step,
                "components": components,
                "connectivity_ratio": connectivity_ratio,
            })
        target_detections = self._get_target_detections(
            observations
        )

        task_events = self.tasks.update(
            self.current_step,
            self.robots,
            observations,
            exploration_rate=self.exploration_rate,
            comm_connected=connected,
            connectivity_ratio=connectivity_ratio,
            target_detections=target_detections,
        )
        info["task_events"] = task_events
        for event in task_events:
            self._log_event(event["type"], event)
        scheduler = getattr(
            self,
            "high_level_scheduler",
            None,
        )

        if (
            scheduler is not None
            and hasattr(
                scheduler,
                "post_physics_step",
            )
        ):
            scheduler_events = (
                scheduler.post_physics_step()
                or []
            )

            info[
                "scheduler_events"
            ] = scheduler_events

            for event in scheduler_events:
                self._log_event(
                    event.get(
                        "type",
                        "scheduler_event",
                    ),
                    event,
                )

        self.current_step += 1
        info["terminated"] = self.tasks.all_complete()
        info["truncated"] = self.current_step >= self.max_steps and not info["terminated"]
        return observations, info

    @property
    def exploration_rate(self) -> float:
        """Exploration coverage.

        ``marvel_native`` uses the frozen MARVEL belief state, which is the
        authoritative exploration state in that mode:

            explored_rate = sum(robot_belief == 255)
                            / sum(ground_truth == 255)

        ``extended`` keeps the team runtime's own visible-cell count, which
        is a different sensing model by design.
        """

        if self.geometry_mode == GEOMETRY_MODE_NATIVE:
            if self.native_belief is None:
                return 0.0

            return float(self.native_belief.explored_rate)

        return min(1.0, len(self.explored_cells) / self.free_cell_count)

    def _free_mask(self) -> np.ndarray:
        """Cached free-cell mask of the occupancy lattice."""

        if self._free_mask_cache is None:
            self._free_mask_cache = (
                self.obstacles.get_occupancy_grid() == 0
            )
            self._free_cell_total = int(
                self._free_mask_cache.sum()
            )

        return self._free_mask_cache

    # ------------------------------------------------------------------
    # Frozen MARVEL sensing (marvel_native only)
    # ------------------------------------------------------------------
    def _mission_step_interval(self) -> int:
        """Physics ticks per high-level mission step."""

        profile = self.config.get("_gppo_profile", {}) or {}

        return max(
            1,
            int(profile.get("high_level_interval_steps", 1)),
        )

    def _sensor_params(self) -> tuple[float, float]:
        params = self.config.get("sensor", {}).get("params", {}) or {}

        fov = float(params.get("fov", 120.0))
        sensor_range = float(
            params.get("range", params.get("sensor_range", 10.0))
        )

        robots_cfg = self.config.get("robots", [])

        if robots_cfg:
            first = robots_cfg[0].get("config", {})
            fov = float(first.get("fov", fov))
            sensor_range = float(
                first.get("sensor_range", sensor_range)
            )

        return fov, sensor_range

    def _initialize_native_belief(self) -> None:
        """Create the frozen belief map and run the initial sensing sweep.

        Frozen ``ScenarioEnv.__init__`` senses once per robot from its
        initial cell and heading before the first mission step.
        """

        ground_truth = getattr(
            self.obstacles, "marvel_ground_truth", None
        )

        if ground_truth is None:
            raise ValueError(
                "marvel_native exploration sensing requires the original "
                "MARVEL 255/1 ground truth"
            )

        fov, sensor_range = self._sensor_params()

        self.native_belief = MarvelNativeBelief(
            ground_truth=ground_truth,
            cell_size=self.obstacles.frame.cell_size,
            sensor_range=sensor_range,
            fov=fov,
        )

        frame = self.obstacles.frame

        # Frozen ScenarioEnv.__init__ first sweeps the start-marker cell
        # with a full 360 degree field of view, before any per-agent sweep.
        initial_cell = getattr(
            self.obstacles, "marvel_initial_cell", None
        )

        if initial_cell is not None:
            self.native_belief.observe_with_fov(
                initial_cell, 0.0, 360.0
            )

        for robot in self.robots:
            self.native_belief.observe(
                frame.world_to_cell(robot.position),
                float(robot.heading) % 360.0,
            )

    def _update_native_belief(self) -> None:
        """Frozen 6-substep sensing sweep across one mission step.

        The frozen worker interpolates each robot's cell and heading across
        ``NUM_SIM_STEPS = 6`` sub-steps and calls
        ``Env.update_robot_belief`` for every robot at every sub-step.
        """

        if self.native_belief is None:
            return

        frame = self.obstacles.frame
        steps = max(1, int(FROZEN_NUM_SIM_STEPS))

        starts = self._mission_start_positions
        start_headings = self._mission_start_headings

        if len(starts) != len(self.robots):
            return

        # Per-substep cells and headings for every robot, matching the
        # frozen linspace/heading interpolation order.
        per_robot_cells = []
        per_robot_headings = []

        for index, robot in enumerate(self.robots):
            cells, headings = interpolated_sensing_track(
                starts[index],
                robot.position,
                start_headings[index],
                robot.heading,
                cell_size=frame.cell_size,
                origin=frame.origin,
                sim_steps=steps,
            )

            per_robot_cells.append(cells)
            per_robot_headings.append(headings)

        for sim_step in range(steps):
            for index in range(len(self.robots)):
                self.native_belief.observe(
                    per_robot_cells[index][sim_step],
                    per_robot_headings[index][sim_step],
                )

    def get_event_log(self) -> list[Dict[str, Any]]:
        return list(self.events)

    def default_actions(self) -> List[Tuple[np.ndarray, float]]:
        frame = self.obstacles.frame
        lower = frame.bounds_min
        upper = frame.bounds_max
        center = (lower + upper) / 2.0
        actions = []
        for idx, robot in enumerate(self.robots):
            angle = 2.0 * np.pi * idx / max(len(self.robots), 1)
            waypoint = robot.position + np.array([np.cos(angle), np.sin(angle)]) * 3.0
            waypoint = np.clip(waypoint, lower, upper)
            heading = np.degrees(np.arctan2(center[1] - robot.position[1], center[0] - robot.position[0])) % 360.0
            actions.append((waypoint, heading))
        return actions

    def _get_observations(self) -> Dict[int, Dict[str, Any]]:
        grid = self.obstacles.get_occupancy_grid()
        positions = [(robot.robot_id, robot.position) for robot in self.robots]
        return {
            robot.robot_id: self.sensor_by_type.get(robot.robot_type, self.sensor).sense(
                robot, grid, positions)
            for robot in self.robots
        }

    def _get_target_detections(
        self,
        observations: Dict[int, Dict[str, Any]],
    ) -> dict[str, dict[int, int]]:
        detections = {}

        for task in self.tasks.tasks.values():
            if task.task_type != "target_search":
                continue

            # Frozen worker rule: only UAVs whose current high-level task
            # is TARGET_SEARCH run detection.  Being capable of Search is
            # not sufficient, and an idle UAV never detects.
            eligible_ids = sorted(
                int(uid)
                for uid in self.active_search_uav_ids
                if uid in {
                    int(robot.robot_id)
                    for robot in self.robots
                }
            )

            detections[task.task_id] = (
                self.target_detector.detect(
                    task.task_id,
                    observations,
                    eligible_ids,
                )
            )

        return detections

    def completed_search_heat_ids(self) -> set[int]:
        """Abstract Search completion notification for the high-level layer.

        Private environment adapter: the sensor layer reports *detected
        target indices*; this resolves them through the private
        ``target_index -> heat_id`` association and returns only the
        completed public heat ids.

        GPPO and the scheduler consume this and never learn where a
        survivor was, or which target index maps to which heat point.
        """

        scenario = self.search_scenario

        if scenario is None:
            return set()

        task = self.tasks.tasks.get(scenario.task_id)

        if task is None:
            return set()

        return scenario.completed_heat_ids(
            task.found_targets
        )

    def install_search_scenario(
        self,
        scenario,
    ) -> None:
        """Install PRIVATE Search scenario truth and publish it to the sensor.

        Hidden survivor coordinates reach the detector here and nowhere
        else; the public heat-point projection is returned to the caller so
        the high-level layer can be fed separately.
        """

        self.search_scenario = scenario

        self.target_detector.set_hidden_targets(
            scenario.task_id,
            scenario.detector_targets(),
        )

    def _update_explored_cells(self, observations: Dict[int, Dict[str, Any]]) -> None:
        for observation in observations.values():
            for x, y in observation.get("visible_cells", []):
                self.explored_cells.add((int(x), int(y)))

    def _generate_initial_positions(self, count: int, mode,
                                    anchors: list[np.ndarray] | None = None) -> List[np.ndarray]:
        frame = self.obstacles.frame
        lower = frame.bounds_min
        upper = frame.bounds_max
        if isinstance(mode, list):
            return [np.asarray(item, dtype=float) for item in mode]
        if mode == "grid":
            cols = int(np.ceil(np.sqrt(count)))
            return [np.array([lower[0] + 8.0 + (idx % cols) * 4.0,
                              lower[1] + 8.0 + (idx // cols) * 4.0]) for idx in range(count)]
        positions = []
        attempts = 0
        while len(positions) < count and attempts < count * 500:
            attempts += 1
            point = np.array([
                np.random.uniform(lower[0] + 5.0, max(lower[0] + 6.0, upper[0] - 5.0)),
                np.random.uniform(lower[1] + 5.0, max(lower[1] + 6.0, upper[1] - 5.0)),
            ])
            collides, _ = self.obstacles.check_collision(point, radius=0.5)
            if collides or any(np.linalg.norm(point - existing) < 1.5
                               for existing in positions):
                continue
            if anchors is not None and (anchors or positions) and not any(
                    np.linalg.norm(point - existing) <= self.comm.comm_range
                    for existing in [*anchors, *positions]):
                continue
            positions.append(point)
        if anchors is not None and len(positions) != count:
            # Fill a connected chain deterministically if rejection sampling
            # cannot place the requested group within the bounded map.
            while len(positions) < count:
                base = np.asarray((anchors or positions)[-1], dtype=float)
                step = min(self.comm.comm_range * 0.6, 5.0)
                candidates = [
                    base + np.array([step, 0.0]),
                    base + np.array([0.0, step]),
                    base + np.array([-step, 0.0]),
                    base + np.array([0.0, -step]),
                ]
                candidate = next(
                    (np.clip(item, lower + 1.0, upper - 1.0)
                     for item in candidates
                     if not any(np.linalg.norm(item - existing) < 1.5
                                for existing in [*anchors, *positions])),
                    np.clip(base + np.array([step, step]), lower + 1.0, upper - 1.0),
                )
                positions.append(candidate)
        if len(positions) != count:
            raise RuntimeError(f"Failed to generate {count} safe start positions")
        return positions

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        self.events.append({"step": self.current_step, "timestamp": time.time(), "type": event_type, "data": data})
