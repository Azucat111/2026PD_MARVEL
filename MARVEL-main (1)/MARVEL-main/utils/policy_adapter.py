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

    def __init__(self, runtime, device: str = "cpu") -> None:
        self.runtime = runtime
        self.device = torch.device(device)
        self.verbose = bool(runtime.config.get("debug_policy", False))
        self._using_policy = False

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

    def _build_map_info(self) -> MapInfo:
        return MapInfo(
            self.belief_map.copy(),
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
        default = self.runtime.default_actions()
        if self.verbose:
            print(f"[PolicyAdapter] Generating actions for {len(self.agents)} agents")
        for idx, (agent, robot) in enumerate(zip(self.agents, self.runtime.robots)):
            try:
                obs = agent.get_observation()
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

        return actions
