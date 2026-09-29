"""MARVEL Policy Adapter — bridges SimulationRuntime observations to PolicyNet actions."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure MARVEL root is on sys.path so parameter.py and utils/* are importable.
_MARVEL_ROOT = Path(__file__).resolve().parent.parent
if str(_MARVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MARVEL_ROOT))

from parameter import (
    CELL_SIZE, NODE_RESOLUTION, FREE, OCCUPIED, UNKNOWN,
    SENSOR_RANGE, FOV, NUM_ANGLES_BIN, NUM_HEADING_CANDIDATES,
    NODE_INPUT_DIM, EMBEDDING_DIM, load_path,
)
from utils.model import PolicyNet
from utils.node_manager import NodeManager
from utils.agent import Agent
from utils.utils import MapInfo
from utils.task_scheduler import TaskScheduler

logger = logging.getLogger(__name__)


class MARVELPolicyAdapter:
    """Wraps MARVEL's PolicyNet for use inside SimulationRuntime.

    Usage::

        adapter = MARVELPolicyAdapter(runtime)
        adapter.setup()                          # call once after runtime.reset()
        actions = adapter.get_actions(obs)       # call each step instead of default_actions()
    """

    def __init__(self, runtime, device: str | None = None) -> None:
        self.runtime = runtime

        # Device selection is explicit and opt-in.  The scenario may declare
        #
        #   policy:
        #     device: cuda
        #     batch_inference: true
        #
        # but the default is CPU with batching off, and asking for CUDA
        # without a usable CUDA runtime raises rather than quietly
        # downgrading -- a silent fallback would change which arithmetic
        # produces the commanded waypoints.
        policy_cfg = runtime.config.get("policy", {}) or {}

        requested = (
            device if device is not None else policy_cfg.get("device", "cpu")
        )
        self.device = self._resolve_device(requested)
        self.verbose = bool(runtime.config.get("debug_policy", False))
        self._using_policy = False

        # Set by `_policy_actions`: True when native post-selection has been
        # deferred to the scheduler, which is where frozen applies it.
        self._defer_post_selection = False

        # True once a scheduler has accepted `bind_post_selection`, i.e. the
        # pipeline it owns can place the post-selection after the per-task
        # override.  The heuristic scheduler cannot, so the adapter falls
        # back to applying it itself.
        self._scheduler_owns_post_selection = False

        # One PolicyNet forward per step for all UAVs (extended mode).
        # Opt-in: it changes only the number of forward passes, and is a
        # wash on CPU, so it is off unless the scenario asks for it.
        self.batch_policy_inference = bool(
            policy_cfg.get("batch_inference", False)
        )

        # Shared belief map at MARVEL's CELL_SIZE resolution.
        self._frame = runtime.obstacles.frame
        width = float(runtime.obstacles.width)
        height = float(runtime.obstacles.height)
        if self._frame.is_native:
            # The parity runtime already sits on MARVEL's own lattice: one
            # belief cell per occupancy cell, anchored at the belief origin.
            # No artificial world-scale expansion.
            self._map_w = int(self._frame.width_cells)
            self._map_h = int(self._frame.height_cells)
            self._belief_origin = self._frame.origin
        else:
            self._map_w = int(np.ceil(width / CELL_SIZE)) + 1
            self._map_h = int(np.ceil(height / CELL_SIZE)) + 1
            self._belief_origin = (0.0, 0.0)
        self.belief_map: np.ndarray = np.ones((self._map_h, self._map_w), dtype=np.int32) * UNKNOWN

        # PolicyNet - 需要3个参数：node_dim, embedding_dim, num_angles_bin
        self.policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)
        self.policy_net.to(self.device)
        self._load_checkpoint()

        # Agents — populated in setup()
        self.agents: List[Agent] = []
        self._node_manager: Optional[NodeManager] = None
        scheduler_cfg = runtime.config.get(
            "task_scheduler", {}
        )
        scheduler_mode = str(
            scheduler_cfg.get("mode", "heuristic")
        ).lower()

        if scheduler_mode == "heuristic":
            self.scheduler = TaskScheduler(runtime)

        elif scheduler_mode == "gppo":
            from integrations.gppo import GPPOTaskScheduler

            checkpoint = scheduler_cfg.get(
                "checkpoint"
            )

            if not checkpoint:
                raise ValueError(
                    "task_scheduler.checkpoint is required "
                    "when mode='gppo'"
                )

            self.scheduler = GPPOTaskScheduler(
                runtime,
                checkpoint_path=str(checkpoint),
                device=str(self.device),
                scheduler_config=scheduler_cfg,
            )

        else:
            raise ValueError(
                f"Unknown task scheduler mode: "
                f"{scheduler_mode}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise agents from current runtime robot states.

        Must be called once after ``runtime.reset()``.
        """
        self.belief_map[:] = UNKNOWN

        robots_cfg = self.runtime.config.get("robots", [])
        fov = FOV
        sensor_range = SENSOR_RANGE
        if robots_cfg:
            first_cfg = robots_cfg[0].get("config", {})
            fov = float(first_cfg.get("fov", FOV))
            sensor_range = float(first_cfg.get("sensor_range", SENSOR_RANGE))

        self._node_manager = NodeManager(fov, sensor_range)

        self.agents = []
        for robot in self.runtime.robots:
            agent = Agent(
                id=robot.robot_id,
                policy_net=self.policy_net,
                fov=fov,
                heading=float(robot.heading) % 360.0,
                sensor_range=sensor_range,
                node_manager=self._node_manager,
                ground_truth_node_manager=None,
                device=self.device,
            )
            self.agents.append(agent)

        if hasattr(
            self.scheduler,
            "bind_marvel_agents",
        ):
            self.scheduler.bind_marvel_agents(
                self.agents
            )

        # Frozen applies the native post-selection after the per-task
        # override, which only the scheduler can order correctly.  Bind the
        # adapter so it can call back into the resolver/heading stages.
        if self._frame.is_native and hasattr(
            self.scheduler,
            "bind_post_selection",
        ):
            self.scheduler.bind_post_selection(self)
            self._scheduler_owns_post_selection = True

        self._initialize_search_scenario()

    def _initialize_search_scenario(self) -> None:
        """Install the frozen Search scenario once initial UAVs exist.

        Ordering is fixed by the frozen generator: it seeds its flood fill
        from the initial UAV positions, so it can only run after
        ``runtime.reset()``.  Only the PUBLIC heat-point projection reaches
        the scheduler; the hidden survivors and the target-to-heat
        association stay in the environment layer.
        """

        if not hasattr(
            self.scheduler,
            "install_search_scenario",
        ):
            return

        from integrations.gppo.search_scenario import (
            initialize_search_scenario,
        )

        truth = initialize_search_scenario(
            self.runtime
        )

        self.scheduler.install_search_scenario(
            truth.public_heat_points()
        )

    def get_actions(self, observations: Dict[int, Dict[str, Any]]) -> List[Tuple[np.ndarray, float]]:
        """Convert SimulationRuntime observations to a list of (waypoint, heading) actions."""
        if not self._using_policy or not self.agents:
            if self.verbose:
                print(
                    f"[PolicyAdapter] Fallback: "
                    f"_using_policy={self._using_policy}, "
                    f"agents={len(self.agents) if self.agents else 0}"
                )
            base_actions = self.runtime.default_actions()

        else:
            try:
                base_actions = self._policy_actions(
                    observations
                )

                if self.verbose:
                    print(
                        "[PolicyAdapter] Policy actions "
                        f"generated: {len(base_actions)} robots"
                    )

            except Exception as exc:
                if self.verbose:
                    print(
                        "[PolicyAdapter] Policy inference "
                        f"error: {exc}"
                    )

                logger.warning(
                    "Policy inference error (%s); "
                    "falling back to default_actions.",
                    exc,
                )

                base_actions = (
                    self.runtime.default_actions()
                )

        return self.scheduler.apply(
            base_actions
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _case_insensitive_child(parent: Path, name: str) -> Optional[Path]:
        """Return the child of *parent* matching *name* ignoring case."""

        if not parent.is_dir():
            return None

        wanted = name.lower()

        for entry in sorted(parent.iterdir()):
            if entry.name.lower() == wanted:
                return entry

        return None

    def _resolve_checkpoint_path(self) -> Optional[Path]:
        """Locate the MARVEL PolicyNet checkpoint.

        ``parameter.load_path`` is inherited verbatim from the frozen repo
        and spells the folder ``load_model/marvel``, while the shipped
        directory is ``load_model/MARVEL``.  That resolves on Windows but
        not on a case-sensitive filesystem.  Fall back to a
        case-insensitive lookup rather than editing the frozen-derived
        parameter file.
        """

        external_checkpoint = self.runtime.config.get(
            "marvel_checkpoint"
        )

        if external_checkpoint:
            return Path(
                external_checkpoint
            ).expanduser().resolve()

        expected = (
            _MARVEL_ROOT
            / load_path
            / "checkpoint.pth"
        )

        if expected.exists():
            return expected

        # Resolve "load_model/marvel" segment by segment, case-insensitively.
        resolved = _MARVEL_ROOT

        for segment in Path(load_path).parts:
            child = self._case_insensitive_child(
                resolved, segment
            )

            if child is None:
                return expected

            resolved = child

        checkpoint = self._case_insensitive_child(
            resolved, "checkpoint.pth"
        )

        return checkpoint if checkpoint is not None else expected

    def _load_checkpoint(self) -> None:
        checkpoint_path = self._resolve_checkpoint_path()

        if checkpoint_path is None or not checkpoint_path.exists():
            logger.warning(
                "Checkpoint not found at %s; will use default_actions.",
                checkpoint_path,
            )
            return
        try:
            ckpt = torch.load(str(checkpoint_path), map_location=self.device)
            state_dict = ckpt.get("policy_model") if isinstance(ckpt, dict) else ckpt
            if state_dict is None:
                raise ValueError("'policy_model' key missing in checkpoint")
            self.policy_net.load_state_dict(state_dict)
            self.policy_net.eval()
            self._using_policy = True
            episode = ckpt.get("episode", "?") if isinstance(ckpt, dict) else "?"
            logger.info("Loaded MARVEL policy from %s (episode %s)", checkpoint_path, episode)
        except Exception as exc:
            logger.warning("Failed to load checkpoint (%s); will use default_actions.", exc)

    def _update_belief(self, visible_cells: np.ndarray) -> None:
        """Mark a set of (x, y) visible cells (1 m resolution) as FREE in the belief map."""
        if len(visible_cells) == 0:
            return
        cells = np.asarray(visible_cells, dtype=float)
        # Convert the runtime lattice cell → world metres → MARVEL belief
        # cell at CELL_SIZE resolution.  On the extended lattice (1 m cells,
        # origin 0) this reduces to the historical `cell * (1/CELL_SIZE)`.
        # On the native lattice the two resolutions coincide, so a runtime
        # cell maps to exactly one belief cell.
        scale = 1.0 / CELL_SIZE  # ≈ 2.5 for CELL_SIZE=0.4
        world_lo = self._frame.cells_to_world(cells)
        world_hi = self._frame.cells_to_world(cells + 1.0)
        bx_lo = np.clip(((world_lo[:, 0] - self._belief_origin[0]) * scale).astype(int), 0, self._map_w - 1)
        by_lo = np.clip(((world_lo[:, 1] - self._belief_origin[1]) * scale).astype(int), 0, self._map_h - 1)
        bx_hi = np.clip(((world_hi[:, 0] - self._belief_origin[0]) * scale).astype(int), 0, self._map_w - 1)
        by_hi = np.clip(((world_hi[:, 1] - self._belief_origin[1]) * scale).astype(int), 0, self._map_h - 1)

        total_marked = 0
        for lx, ly, hx, hy in zip(bx_lo, by_lo, bx_hi, by_hi):
            self.belief_map[ly : hy + 1, lx : hx + 1] = FREE
            total_marked += (hy - ly + 1) * (hx - lx + 1)

        if total_marked > 0 and self.verbose:
            print(f"[PolicyAdapter] Updated belief: {len(visible_cells)} cells -> {total_marked} belief cells marked FREE")

    def _observation_padding(self) -> bool:
        """The `pad` argument for `Agent.get_observation`.

        Frozen Phase14-v9 action selection runs with ``pad=False``:

            observation = robot.get_observation(pad=False)

        (``utils/marvel_gppo_test_worker.py:240``; the original MARVEL
        evaluation worker at ``utils/test_worker.py:69`` does the same.)
        ``pad=True`` is the training-time shape
        (``multi_agent_worker.py:93``) and pads the node and edge axes to
        ``NODE_PADDING_SIZE``/``K_SIZE``, which changes the flat action
        space and therefore the logits the policy sees.

        ``marvel_native`` must reproduce the frozen reference exactly.
        The extended environment keeps the historical ``pad=True``.
        """

        return not self._frame.is_native

    def _belief_array(self) -> np.ndarray:
        """The belief map the MARVEL graph is built from.

        In ``marvel_native`` this MUST be the frozen ``robot_belief``
        array: frozen ``ScenarioEnv`` hands ``MapInfo(self.robot_belief,
        belief_origin_x, belief_origin_y, CELL_SIZE)`` to
        ``Agent.update_graph``.  The adapter's own ``belief_map`` is
        written from ``IdealSensor`` visible cells, which is a different
        observation model, so using it here would build a different node
        graph and therefore select different actions.

        ``extended`` keeps the adapter's accumulated map unchanged.
        """

        if self._frame.is_native:
            native = getattr(self.runtime, "native_belief", None)

            if native is not None:
                return native.belief.copy()

        return self.belief_map.copy()

    def _build_map_info(self) -> MapInfo:
        return MapInfo(
            self._belief_array(),
            float(self._belief_origin[0]),
            float(self._belief_origin[1]),
            CELL_SIZE,
        )

    def _snap_to_nearest_node(self, position: np.ndarray) -> np.ndarray:
        """Return the graph-node coordinate closest to *position*."""
        if self._node_manager is None or self._node_manager.nodes_dict.__len__() == 0:
            return position
        nearest = self._node_manager.nodes_dict.nearest_neighbors(position.tolist(), 1)
        if nearest:
            return nearest[0].data.coords
        return position

    def _policy_actions(self, observations: Dict[int, Dict[str, Any]]) -> List[Tuple[np.ndarray, float]]:
        # 1. Update shared belief map from every robot's visible cells.
        for obs in observations.values():
            self._update_belief(np.asarray(obs.get("visible_cells", []), dtype=float))

        map_info = self._build_map_info()
        all_positions = [robot.position for robot in self.runtime.robots]

        # 2. Update each agent's global map (they slice their updating_map from it).
        for agent in self.agents:
            agent.map_info = map_info

        # 3. Update each agent's graph (adds nodes around current position).
        for agent, robot in zip(self.agents, self.runtime.robots):
            agent.update_heading(float(robot.heading) % 360.0)
            try:
                if self.verbose:
                    print(f"[PolicyAdapter] Robot {robot.robot_id}: update_graph at pos={robot.position.round(2)}")
                agent.update_graph(map_info, robot.position.copy())
                if self.verbose:
                    print(f"[PolicyAdapter] Robot {robot.robot_id}: update_graph completed, nodes={len(self._node_manager.nodes_dict)}")
            except Exception as exc:
                if self.verbose:
                    print(f"[PolicyAdapter] Robot {robot.robot_id}: update_graph FAILED: {exc}")
                import traceback
                traceback.print_exc()
                logger.debug("update_graph failed for robot %d: %s", robot.robot_id, exc)

        # 3. Update planning state (needs all robot locations snapped to graph nodes).
        num_nodes = self._node_manager.nodes_dict.__len__()
        if self.verbose:
            print(f"[PolicyAdapter] Step 3: num_nodes={num_nodes}")
        if num_nodes == 0:
            if self.verbose:
                print(f"[PolicyAdapter] No nodes in graph, returning default actions")
            return self.runtime.default_actions()

        snapped = np.array([self._snap_to_nearest_node(p) for p in all_positions])
        for agent in self.agents:
            try:
                agent.update_planning_state(snapped)
            except Exception as exc:
                logger.debug("update_planning_state failed for agent %d: %s", agent.id, exc)

        # 4. Get observations and select waypoints.
        actions: List[Tuple[np.ndarray, float]] = []
        used_fallback = False
        default = self.runtime.default_actions()
        if self.verbose:
            print(f"[PolicyAdapter] Generating actions for {len(self.agents)} agents")

        # One forward pass for all UAVs.  Every per-UAV observation is built
        # exactly as the sequential path builds it -- including the per-agent
        # nearest-node window -- and the nine fixed-shape tensors are stacked
        # along the batch axis.  Any failure falls back to the sequential
        # loop below, so the batched path can never change behaviour, only
        # how many forward passes it costs.
        if self._batched_inference_applies():
            batched = self._batched_actions(default)
            if batched is not None:
                actions = batched
                used_fallback = False

        if not actions:
            for idx, (agent, robot) in enumerate(zip(self.agents, self.runtime.robots)):
                try:
                    obs = agent.get_observation(pad=self._observation_padding())
                    next_position, _, _, heading_index = agent.select_next_waypoint(obs, greedy=True)
                    heading_deg = float(heading_index) * (360.0 / NUM_ANGLES_BIN)
                    waypoint = np.asarray(next_position, dtype=float)

                    # 调试：检查waypoint是否合理
                    dist = np.linalg.norm(waypoint - robot.position)
                    if self.verbose:
                        print(f"[PolicyAdapter] Robot {robot.robot_id}: pos={robot.position.round(2)}, waypoint={waypoint.round(2)}, dist={dist:.2f}, heading={heading_deg:.1f}")

                    actions.append((waypoint, heading_deg))
                except Exception as exc:
                    if self.verbose:
                        print(f"[PolicyAdapter] Robot {robot.robot_id}: action selection FAILED: {exc}")
                    import traceback
                    traceback.print_exc()
                    logger.debug("Action selection failed for robot %d: %s", robot.robot_id, exc)
                    actions.append(default[idx])
                    used_fallback = True

        # 5. Frozen post-selection filtering.
        #
        # Frozen applies this *after* the per-task override, not before:
        # `_select_actions` dispatches Exploration/Search/Relay/Safety and
        # returns those waypoints, and only then does `run_episode` call
        # `_resolve_same_waypoint_collisions` / `_apply_hazard_traversability`
        # and recompute the heading bins.  Resolving duplicates against the
        # raw PolicyNet waypoints instead would displace an Exploration
        # robot that shares a raw waypoint with a robot whose waypoint is
        # about to be overridden by Search/Relay.
        #
        # The scheduler therefore owns the native post-selection, running it
        # once the overrides are in place.  Extended mode keeps the raw
        # PolicyNet output.  A policy failure means the frozen path did not
        # run either, so the scheduler is told to skip it.
        self._defer_post_selection = bool(
            self._frame.is_native
            and not used_fallback
            and self._scheduler_owns_post_selection
        )

        if (
            self._frame.is_native
            and not used_fallback
            and not self._defer_post_selection
        ):
            # Scheduler cannot order it after the override; keep the
            # historical eager behaviour rather than dropping it.
            actions = self._apply_frozen_post_selection(actions)

        return actions

    # ------------------------------------------------------------------
    # Frozen low-level post-selection semantics (marvel_native only)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(requested) -> torch.device:
        """Resolve the PolicyNet device, failing loudly on a bad request."""

        name = str(requested).strip().lower()

        if name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"policy.device={requested!r} was requested but CUDA is not "
                "available. Set policy.device to 'cpu', or run where a CUDA "
                "runtime is present."
            )

        return torch.device(name)

    def _batched_inference_applies(self) -> bool:
        """Whether one forward pass can serve every UAV this step.

        Extended mode only: `marvel_native` keeps the sequential path the
        frozen parity suites pin, so batching cannot perturb it.
        """

        return bool(
            self.batch_policy_inference
            and not self._frame.is_native
            and len(self.agents) > 1
        )

    def _batched_actions(self, default):
        """One PolicyNet forward for all agents; `None` means fall back.

        Each agent's observation is built by its own `get_observation`, so
        the per-agent nearest-node window, the remapped `current_edge` and
        the neighbour heading candidates are exactly what the sequential
        path produces.  Only the forward pass is shared: the nine
        observation tensors are stacked on the batch axis and each row is
        decoded against that agent's own observation.
        """

        pad = self._observation_padding()

        try:
            # Build every observation on CPU.  `get_observation` moves each
            # tensor to the agent's device as it builds it, so leaving the
            # agents on CUDA would do nine transfers per UAV; instead one
            # transfer per tensor happens below, once per mission step.
            devices = [agent.device for agent in self.agents]

            try:
                for agent in self.agents:
                    agent.device = torch.device("cpu")
                observations = [
                    agent.get_observation(pad=pad) for agent in self.agents
                ]
            finally:
                for agent, device in zip(self.agents, devices):
                    agent.device = device

            with torch.no_grad():
                batched = [
                    torch.cat([obs[index] for obs in observations], dim=0).to(
                        self.device
                    )
                    for index in range(len(observations[0]))
                ]
                logits = self.policy_net(*batched).to("cpu")

            actions = []

            for index, (agent, obs) in enumerate(
                zip(self.agents, observations)
            ):
                (
                    next_position,
                    _node,
                    _action,
                    heading_index,
                ) = agent.decode_waypoint(
                    obs, logits[index:index + 1], greedy=True
                )

                actions.append(
                    (
                        np.asarray(next_position, dtype=float),
                        float(heading_index) * (360.0 / NUM_ANGLES_BIN),
                    )
                )

            return actions
        except Exception as exc:
            logger.debug(
                "Batched policy inference failed (%s); falling back to "
                "sequential selection.",
                exc,
            )
            if self.device.type == "cuda":
                # A failed CUDA step must not poison the next one.
                torch.cuda.empty_cache()
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None

    def _apply_frozen_post_selection(self, actions):
        """Frozen worker ordering, reproduced exactly.

        ``marvel_gppo_test_worker.run_episode``:

            selected_locations, _, next_headings = self._select_actions()
            selected_locations = self._resolve_same_waypoint_collisions(...)
            selected_locations = self._apply_hazard_traversability(...)
            # recompute heading bins from the FINAL waypoint directions
            self._simulate_motion(selected_locations, next_headings)

        The PolicyNet's heading index is therefore overwritten by the
        heading bin nearest the resolved waypoint direction.

        This is the composite form, kept for direct use.  In the live
        pipeline the two stages are split -- see `resolve_actions` and
        `recompute_headings` -- because frozen runs the hazard guard
        between them.
        """

        return self.recompute_headings(self.resolve_actions(actions))

    def resolve_actions(self, actions):
        """Frozen `_resolve_same_waypoint_collisions` over actions."""

        waypoints = self._resolve_same_waypoint_collisions(
            [np.asarray(action[0], dtype=float) for action in actions]
        )

        return [
            (waypoint, heading)
            for waypoint, (_original, heading) in zip(
                waypoints, actions
            )
        ]

    def recompute_headings(self, actions):
        """Frozen heading-bin recomputation from the final waypoint.

        Frozen recomputes the bin from the final waypoint direction and
        keeps the previous index only when the displacement is ~zero.
        """

        resolved = []

        for robot, (waypoint, heading) in zip(
            self.runtime.robots, actions
        ):
            heading_index = int(
                round(float(heading) / (360.0 / NUM_ANGLES_BIN))
            )

            delta = np.asarray(waypoint, dtype=float) - np.asarray(
                robot.position, dtype=float
            )

            if float(np.linalg.norm(delta)) > 1e-9:
                angle = float(
                    np.degrees(
                        np.arctan2(delta[1], delta[0])
                    ) % 360.0
                )
                heading_index = int(
                    np.floor(angle / 360.0 * NUM_ANGLES_BIN)
                ) % NUM_ANGLES_BIN

            resolved.append(
                (
                    waypoint,
                    float(heading_index)
                    * (360.0 / NUM_ANGLES_BIN),
                )
            )

        return resolved

    def _resolve_same_waypoint_collisions(self, waypoints):
        """Frozen ``_resolve_same_waypoint_collisions``, ported verbatim.

        Duplicate destinations are resolved against the shared node graph:
        robots are processed closest-first to their own waypoint, the first
        claim wins, and a duplicate is moved to the first of the 25 nearest
        graph nodes whose exact coordinates are unclaimed.  If every
        candidate is taken the duplicate waypoint is left unchanged.
        """

        selected = np.asarray(waypoints, dtype=float).copy()

        order = np.argsort(
            [
                float(
                    np.linalg.norm(
                        selected[i]
                        - np.asarray(
                            self.runtime.robots[i].position,
                            dtype=float,
                        )
                    )
                )
                for i in range(len(selected))
            ]
        )

        occupied = set()

        for rid in order:
            loc = selected[rid]
            key = (float(loc[0]), float(loc[1]))

            if key not in occupied:
                occupied.add(key)
                continue

            node_manager = self.agents[int(rid)].node_manager

            nearby = (
                node_manager.nodes_dict.nearest_neighbors(
                    loc.tolist(),
                    25,
                )
            )

            for node in nearby:
                coords = np.asarray(
                    node.data.coords, dtype=float
                )
                candidate = (float(coords[0]), float(coords[1]))

                if candidate not in occupied:
                    selected[rid] = coords
                    occupied.add(candidate)
                    break

        return selected
