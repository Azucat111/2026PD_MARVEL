"""Regression tests for the accepted MARVEL-native policy-execution fixes.

Covers, in order:

A. ``marvel_native`` calls ``get_observation(pad=False)`` like the frozen
   Phase14-v9 worker (``marvel_gppo_test_worker.py:240``).
B. ``extended`` keeps its historical ``pad=True``.
C. ``edge_padding_mask`` marks ``current_in_edge``, not index 0.
D. Frozen post-selection heading recomputation, including the
   zero-displacement case that retains the policy heading index.
E. Same-waypoint collision resolver, compared against the real frozen
   implementation in a subprocess.
F. Real MARVEL PolicyNet drives both sides for 10 mission steps with no
   causal divergence.

Frozen source: ``/home/nick/MARVEL`` @ ``e867117``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.geometry import GEOMETRY_MODE_NATIVE
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime

FROZEN_REPO = Path("/home/nick/MARVEL")
NATIVE_SCENARIO = ROOT / "configs" / "scenarios" / "marvel_native_test.yaml"
EXTENDED_SCENARIO = ROOT / "configs" / "scenarios" / "baseline_maps_test.yaml"
CHECKPOINT = ROOT / "load_model" / "MARVEL" / "checkpoint.pth"

FROZEN_CHECKPOINT = (
    FROZEN_REPO / "artifacts" / "phase14_v9_frozen"
    / "gppo_phase14_v9_ckpt80.pth"
)

MAP_NAME = "1.png"
CELL_SIZE = 0.4
SENSOR_RANGE = 10.0
FOV = 120.0
NUM_ANGLES_BIN = 36
SEED = 12345
STEPS = 10

requires_frozen = pytest.mark.skipif(
    not FROZEN_REPO.exists(),
    reason=f"frozen MARVEL repo not available at {FROZEN_REPO}",
)
requires_policy = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason=f"MARVEL checkpoint not available at {CHECKPOINT}",
)


# ======================================================================
# Shared harness: native runtime pinned to the frozen scenario starts
# ======================================================================

FROZEN_STARTS_PROBE = textwrap.dedent(
    '''
    import sys, json
    sys.path.insert(0, "/home/nick/MARVEL")
    import numpy as np
    from utils.scenario_env import ScenarioEnv
    from parameter import FOV, SENSOR_RANGE
    env = ScenarioEnv(episode_index=0, fov=FOV, n_agents=4,
                      sensor_range=SENSOR_RANGE, map_dir="maps_test",
                      scenario_seed=int(sys.argv[1]))
    print(json.dumps({
        "starts": np.asarray(env.robot_locations, float).tolist(),
        "angles": [float(a) for a in env.angles],
    }))
    '''
)


@pytest.fixture(scope="module")
def frozen_starts(tmp_path_factory):
    if not FROZEN_REPO.exists():
        pytest.skip("frozen MARVEL repo not available")

    script = tmp_path_factory.mktemp("starts") / "probe.py"
    script.write_text(FROZEN_STARTS_PROBE, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), str(SEED)],
        capture_output=True, text=True, cwd=str(FROZEN_REPO),
    )

    if result.returncode != 0:
        pytest.skip(f"frozen starts probe failed: {result.stderr[-300:]}")

    return json.loads(result.stdout.strip().splitlines()[-1])


def _native_runtime(tmp_path, starts, headings=None):
    """Native runtime pinned to explicit starts, no GPPO profile."""

    base = load_and_validate_scenario(NATIVE_SCENARIO)
    config = yaml.safe_load(yaml.safe_dump(base))

    environment = config["environment"]
    environment["map_dir"] = str(ROOT / "maps_test")
    environment["geometry_mode"] = "marvel_native"
    environment["initial_headings"] = 270.0

    config["robots"] = [{
        "id_range": [0, len(starts) - 1],
        "type": "explorer",
        "team_id": 1,
        "config": {
            "fov": FOV,
            "sensor_range": SENSOR_RANGE,
            "velocity": 1.0,
            "yaw_rate": 35,
            "initial_positions": [
                [float(x), float(y)] for x, y in starts
            ],
        },
    }]

    configs = (ROOT / "configs").resolve()
    config["dynamics"] = {
        "config_file": str(configs / "dynamics" / "kinematic_simple.yaml")
    }
    config["sensor"] = {
        "config_file": str(configs / "sensors" / "ideal.yaml")
    }
    config["communication"] = {
        "config_file": str(configs / "communication" / "ideal.yaml")
    }
    config.pop("_gppo_profile", None)

    path = tmp_path / "native_policy.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    runtime = SimulationRuntime(load_and_validate_scenario(path))
    runtime.reset()

    return runtime


def _adapter(runtime):
    from utils.policy_adapter import MARVELPolicyAdapter

    adapter = MARVELPolicyAdapter(runtime)
    adapter.setup()

    return adapter


# ======================================================================
# A / B — observation padding
# ======================================================================

def test_native_mode_uses_pad_false(tmp_path):
    """marvel_native must reproduce the frozen worker's pad=False."""

    runtime = _native_runtime(tmp_path, [[-4.0, 0.0], [-8.0, 4.0]])
    adapter = _adapter(runtime)

    assert adapter._frame.mode == GEOMETRY_MODE_NATIVE
    assert adapter._observation_padding() is False


def test_extended_mode_keeps_its_padding():
    """The extended environment keeps the historical pad=True."""

    config = load_and_validate_scenario(EXTENDED_SCENARIO)

    np.random.seed(7)
    runtime = SimulationRuntime(config)
    runtime.reset()

    adapter = _adapter(runtime)

    assert adapter._frame.mode != GEOMETRY_MODE_NATIVE
    assert adapter._observation_padding() is True


# ======================================================================
# C — edge_padding_mask marks current_in_edge
# ======================================================================

@requires_policy
def test_edge_padding_mask_marks_current_in_edge(tmp_path, frozen_starts):
    """The marker must sit at argwhere(current_edge == current_index)[0][0].

    The historical bug compared the integer neighbour list against the
    already-tensorised ``current_index``, which broadcasts to (1, 1, k) and
    pinned the marker to index 0 for every state.
    """

    runtime = _native_runtime(tmp_path, frozen_starts["starts"])
    adapter = _adapter(runtime)

    assert adapter._using_policy

    observations = runtime._get_observations()
    adapter._policy_actions(observations)

    checked = 0

    for agent in adapter.agents:
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, \
            edge_padding_mask, _fd, _hv, _nbh = agent.get_observation(pad=False)

        index = int(current_index.item())
        neighbours = np.asarray(current_edge).reshape(-1)

        expected = int(np.argwhere(neighbours == index)[0][0])
        marker = np.asarray(edge_padding_mask).reshape(-1)

        assert int(np.argwhere(marker == 1)[0][0]) == expected, (
            agent.id, expected, marker.tolist(),
        )

        # The case that would have masked the bug: a non-zero position.
        if expected != 0:
            checked += 1

    assert checked > 0, "no agent exercised a non-zero current_in_edge"


# ======================================================================
# D — post-filter heading recomputation
# ======================================================================

def test_heading_recomputed_from_final_waypoint(tmp_path):
    """Frozen recomputes the bin from the resolved waypoint direction."""

    runtime = _native_runtime(tmp_path, [[-4.0, 0.0]])
    adapter = _adapter(runtime)

    robot = runtime.robots[0]

    # A waypoint due east of the robot, with a deliberately wrong policy
    # heading index (180 deg) that must be discarded.
    waypoint = np.asarray(
        [robot.position[0] + 4.0, robot.position[1]], dtype=float
    )

    actions = adapter._apply_frozen_post_selection(
        [(waypoint, 180.0)]
    )

    # atan2(0, 4) = 0 deg -> bin 0 -> 0 degrees.
    assert actions[0][0][0] == pytest.approx(waypoint[0])
    assert actions[0][1] == pytest.approx(0.0)

    # North is 90 degrees -> bin 9 -> 90 degrees.
    north = np.asarray(
        [robot.position[0], robot.position[1] + 4.0], dtype=float
    )
    actions = adapter._apply_frozen_post_selection([(north, 180.0)])

    assert actions[0][1] == pytest.approx(90.0)


def test_zero_displacement_retains_policy_heading(tmp_path):
    """Frozen's `norm(delta) > 1e-9` guard keeps the policy index."""

    runtime = _native_runtime(tmp_path, [[-4.0, 0.0]])
    adapter = _adapter(runtime)

    robot = runtime.robots[0]

    same = np.asarray(robot.position, dtype=float)

    actions = adapter._apply_frozen_post_selection([(same, 180.0)])

    assert actions[0][1] == pytest.approx(180.0)


# ======================================================================
# E — same-waypoint resolver vs the real frozen implementation
# ======================================================================

FROZEN_RESOLVER_PROBE = textwrap.dedent(
    '''
    import sys, json
    sys.path.insert(0, "/home/nick/MARVEL")
    import numpy as np
    from utils.scenario_env import ScenarioEnv
    from utils.node_manager import NodeManager
    from utils.agent import Agent
    from parameter import FOV, SENSOR_RANGE

    SEED = int(sys.argv[1])
    case = json.loads(sys.argv[2])

    env = ScenarioEnv(episode_index=0, fov=FOV, n_agents=4,
                      sensor_range=SENSOR_RANGE, map_dir="maps_test",
                      scenario_seed=SEED)
    nm = NodeManager(FOV, SENSOR_RANGE)
    agents = [Agent(i, None, FOV, float(env.angles[i]) % 360.0,
                    SENSOR_RANGE, nm, None, "cpu", False)
              for i in range(4)]

    class _W:
        """Minimal worker shim: only the resolver is exercised."""
        robot_list = agents
        n_agents = 4
        node_manager = nm

        @staticmethod
        def _resolve_same_waypoint_collisions(selected_locations):
            from utils.marvel_gppo_test_worker import MarvelGPPOTestWorker
            return MarvelGPPOTestWorker._resolve_same_waypoint_collisions(
                _W, selected_locations)

    n = int(case["n_nodes"])
    nodes = nm.nodes_dict
    # Seed the graph so nearest_neighbors has candidates.
    for i in range(n):
        nodes.add_node((float(case["seed_x"]) + i, float(case["seed_y"])))

    for i, coords in enumerate(case["nodes"]):
        nodes.add_node((float(coords[0]), float(coords[1])))

    for i in range(4):
        agents[i].location = np.asarray(case["locations"][i], dtype=float)

    try:
        out = _W._resolve_same_waypoint_collisions(case["waypoints"])
    except Exception as exc:  # pragma: no cover - reported to the test
        print(json.dumps({"error": repr(exc)}))
        raise SystemExit(0)

    print(json.dumps({"resolved": np.asarray(out, float).tolist()}))
    '''
)


RESOLVER_CASES = {
    "all_distinct": {
        "seed_x": 20.0, "seed_y": 20.0, "n_nodes": 30,
        "nodes": [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [12.0, 0.0]],
        "locations": [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [12.0, 0.0]],
        "waypoints": [[4.0, 0.0], [8.0, 0.0], [12.0, 0.0], [16.0, 0.0]],
    },
    "two_duplicates": {
        "seed_x": 20.0, "seed_y": 20.0, "n_nodes": 30,
        "nodes": [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [12.0, 0.0]],
        "locations": [[0.0, 0.0], [0.5, 0.0], [8.0, 0.0], [12.0, 0.0]],
        # robots 0 and 1 both claim [8, 0]; robot 0 is closer to it.
        "waypoints": [[8.0, 0.0], [8.0, 0.0], [12.0, 0.0], [16.0, 0.0]],
    },
    "multiple_duplicates": {
        "seed_x": 20.0, "seed_y": 20.0, "n_nodes": 30,
        "nodes": [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [12.0, 0.0]],
        "locations": [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]],
        # all four claim the same node.
        "waypoints": [[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [4.0, 0.0]],
    },
}


def _frozen_resolve(tmp_path, case):
    if not FROZEN_REPO.exists():
        pytest.skip("frozen MARVEL repo not available")

    script = tmp_path / "resolver_probe.py"
    script.write_text(FROZEN_RESOLVER_PROBE, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), str(SEED), json.dumps(case)],
        capture_output=True, text=True, cwd=str(FROZEN_REPO),
    )

    if result.returncode != 0:
        pytest.skip(f"frozen resolver probe failed: {result.stderr[-400:]}")

    payload = json.loads(result.stdout.strip().splitlines()[-1])

    if "error" in payload:
        pytest.skip(f"frozen resolver unavailable: {payload['error']}")

    return np.asarray(payload["resolved"], dtype=float)


@pytest.mark.parametrize("name", sorted(RESOLVER_CASES))
def test_same_waypoint_resolver_matches_frozen(tmp_path, name):
    """The port must reproduce the real frozen resolver exactly."""

    case = RESOLVER_CASES[name]

    frozen = _frozen_resolve(tmp_path, case)

    runtime = _native_runtime(tmp_path, case["locations"])
    adapter = _adapter(runtime)

    for robot, location in zip(runtime.robots, case["locations"]):
        robot.position = np.asarray(location, dtype=float)

    resolved = adapter._resolve_same_waypoint_collisions(
        [np.asarray(w, dtype=float) for w in case["waypoints"]]
    )

    assert np.allclose(
        np.asarray(resolved, dtype=float), frozen, atol=1e-9
    ), (name, np.asarray(resolved).tolist(), frozen.tolist())


def test_resolver_keeps_duplicate_when_no_alternative(tmp_path):
    """Frozen fallback: every candidate taken -> the duplicate is kept."""

    runtime = _native_runtime(tmp_path, [[0.0, 0.0], [4.0, 0.0]])
    adapter = _adapter(runtime)

    shared = np.asarray([4.0, 0.0], dtype=float)

    # Claim the node itself; with an empty graph the nearest-neighbour
    # search yields nothing, so the second duplicate must survive.
    resolved = adapter._resolve_same_waypoint_collisions(
        [shared.copy(), shared.copy()]
    )

    assert resolved.shape == (2, 2)
    assert np.allclose(resolved[0], shared, atol=1e-9)
    assert np.allclose(resolved[1], shared, atol=1e-9)


# ======================================================================
# F — real PolicyNet 10-step rollout
# ======================================================================

FROZEN_ROLLOUT_PROBE = textwrap.dedent(
    '''
    import sys, json
    sys.path.insert(0, "/home/nick/MARVEL")
    import numpy as np, torch
    from utils.scenario_env import ScenarioEnv
    from utils.agent import Agent
    from utils.node_manager import NodeManager
    from utils.model import PolicyNet
    from utils.sensor import sensor_work_heading
    from utils.motion_model import compute_allowable_heading
    from parameter import NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN, FOV, SENSOR_RANGE

    SEED, STEPS = int(sys.argv[1]), int(sys.argv[2])
    env = ScenarioEnv(episode_index=0, fov=FOV, n_agents=4, sensor_range=SENSOR_RANGE,
                      map_dir="maps_test", scenario_seed=SEED)
    nm = NodeManager(FOV, SENSOR_RANGE)
    net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN)
    net.load_state_dict(torch.load("load_model/MARVEL/checkpoint.pth",
                                   map_location="cpu")["policy_model"]); net.eval()
    agents = [Agent(i, net, FOV, float(env.angles[i]) % 360.0, SENSOR_RANGE,
                    nm, None, torch.device("cpu"), False) for i in range(4)]
    locs = np.asarray(env.robot_locations, float).copy()
    headings = [float(a) % 360.0 for a in env.angles]

    for a in agents: a.update_graph(env.belief_info, env.robot_locations[a.id].copy())
    for a in agents: a.update_planning_state(env.robot_locations)

    def cell(p):
        return np.round((np.asarray(p, float) - [env.belief_origin_x, env.belief_origin_y])
                        / env.cell_size).astype(int)

    out = {"start_positions": locs.tolist(), "start_headings": list(headings),
           "belief_free0": int((env.robot_belief == 255).sum()),
           "rate0": float((env.robot_belief == 255).sum() / (env.ground_truth == 255).sum()),
           "steps": []}

    for step in range(STEPS):
        env.evaluate_exploration_rate()
        rec = {"pre_positions": locs.tolist(), "pre_headings": list(headings),
               "belief_free": int((env.robot_belief == 255).sum()),
               "rate": float(env.explored_rate), "policy": [],
               "final_headings": [], "cells": [], "sub_headings": []}
        selected = []; nh = []
        for a in agents:
            o = a.get_observation(pad=False)
            wp, node, act, hidx = a.select_next_waypoint(o, greedy=True)
            rec["policy"].append({"node_index": int(node), "heading_index": int(hidx),
                                  "waypoint": [float(wp[0]), float(wp[1])]})
            selected.append(np.asarray(wp, float)); nh.append(int(hidx))
        sel = np.asarray(selected, float).copy()
        order = np.argsort([float(np.linalg.norm(sel[i] - locs[i])) for i in range(4)])
        occ = set()
        for rid in order:
            loc = sel[rid]; k = (float(loc[0]), float(loc[1]))
            if k not in occ: occ.add(k); continue
            for nd in nm.nodes_dict.nearest_neighbors(loc.tolist(), 25):
                c = np.asarray(nd.data.coords, float); cc = (float(c[0]), float(c[1]))
                if cc not in occ: sel[rid] = c; occ.add(cc); break
        for rid in range(4):
            d = sel[rid] - locs[rid]
            if float(np.linalg.norm(d)) > 1e-9:
                ang = float(np.degrees(np.arctan2(d[1], d[0])) % 360.0)
                nh[rid] = int(np.floor(ang / 360.0 * NUM_ANGLES_BIN)) % NUM_ANGLES_BIN
        rec["resolved_waypoints"] = [p.tolist() for p in sel]
        rec["recomputed_heading_index"] = list(nh)
        rec["desired_heading_deg"] = [h * (360.0 / NUM_ANGLES_BIN) for h in nh]
        for i in range(4):
            fh = compute_allowable_heading(locs[i], sel[i], headings[i],
                                           nh[i] * (360.0 / NUM_ANGLES_BIN), 1.0, 35.0)
            rec["final_headings"].append(float(fh) % 360.0)
            headings[i] = fh; locs[i] = sel[i]
            agents[i].update_heading(fh)
        rec["final_positions"] = locs.tolist()
        for i in range(4):
            sc = cell(rec["pre_positions"][i]); ec = cell(sel[i])
            cs = np.round(np.linspace(sc, ec, 7)[1:]).astype(int)
            prev = rec["pre_headings"][i]; fin = rec["final_headings"][i]
            dd = fin - prev
            if abs(dd) > 180: dd = dd - 360 if dd > 0 else dd + 360
            hs = [(prev + (j + 1) * dd / 6) % 360.0 for j in range(6)]
            rec["cells"].append(cs.tolist()); rec["sub_headings"].append(hs)
        for j in range(6):
            for i in range(4):
                env.robot_belief = sensor_work_heading(
                    np.array(rec["cells"][i][j], dtype=int),
                    round(SENSOR_RANGE / env.cell_size), env.robot_belief,
                    env.ground_truth, rec["sub_headings"][i][j], FOV)
        env.robot_locations = locs.copy()
        for i in range(4): agents[i].location = locs[i]
        for a in agents: a.update_graph(env.belief_info, env.robot_locations[a.id].copy())
        for a in agents: a.update_planning_state(env.robot_locations)
        env.evaluate_exploration_rate()
        rec["post_belief_free"] = int((env.robot_belief == 255).sum())
        rec["post_rate"] = float(env.explored_rate)
        out["steps"].append(rec)

    print(json.dumps(out))
    '''
)


@requires_frozen
@requires_policy
def test_real_policynet_ten_step_parity(tmp_path_factory, frozen_starts):
    """The real MARVEL PolicyNet drives both sides for 10 mission steps."""

    import utils.agent as agent_module

    tmp_path = tmp_path_factory.mktemp("rollout")

    script = tmp_path / "frozen_rollout.py"
    script.write_text(FROZEN_ROLLOUT_PROBE, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script), str(SEED), str(STEPS)],
        capture_output=True, text=True, cwd=str(FROZEN_REPO),
    )

    if result.returncode != 0:
        pytest.skip(f"frozen rollout probe failed: {result.stderr[-400:]}")

    frozen = json.loads(result.stdout.strip().splitlines()[-1])

    runtime = _native_runtime(tmp_path, frozen["start_positions"])
    adapter = _adapter(runtime)

    assert adapter._using_policy

    captured = []
    original = agent_module.Agent.select_next_waypoint

    def spy(self, observation, *args, **kwargs):
        waypoint, node, action, heading_index = original(
            self, observation, *args, **kwargs
        )
        captured.append({
            "node_index": int(node),
            "heading_index": int(heading_index),
            "waypoint": [float(waypoint[0]), float(waypoint[1])],
        })
        return waypoint, node, action, heading_index

    agent_module.Agent.select_next_waypoint = spy

    try:
        observations = runtime._get_observations()

        for step in range(STEPS):
            reference = frozen["steps"][step]

            # PRE-POLICY state.
            assert np.allclose(
                np.asarray([r.position for r in runtime.robots]),
                np.asarray(reference["pre_positions"]), atol=1e-9,
            ), f"step {step + 1} pre positions"

            assert np.allclose(
                [r.heading for r in runtime.robots],
                reference["pre_headings"], atol=1e-9,
            ), f"step {step + 1} pre headings"

            assert runtime.native_belief.explored_free_count == (
                reference["belief_free"]
            ), f"step {step + 1} belief"

            assert runtime.exploration_rate == pytest.approx(
                reference["rate"], abs=1e-15
            ), f"step {step + 1} rate"

            # POLICY.
            captured.clear()
            actions = adapter._policy_actions(observations)

            assert [c["node_index"] for c in captured] == [
                p["node_index"] for p in reference["policy"]
            ], f"step {step + 1} node index"

            assert [c["heading_index"] for c in captured] == [
                p["heading_index"] for p in reference["policy"]
            ], f"step {step + 1} raw policy heading index"

            assert np.allclose(
                [c["waypoint"] for c in captured],
                [p["waypoint"] for p in reference["policy"]], atol=1e-9,
            ), f"step {step + 1} raw waypoint"

            # POST-SELECTION.
            assert np.allclose(
                [a[0] for a in actions],
                reference["resolved_waypoints"], atol=1e-9,
            ), f"step {step + 1} resolved waypoint"

            assert [round(a[1] / (360.0 / NUM_ANGLES_BIN))
                    for a in actions] == (
                reference["recomputed_heading_index"]
            ), f"step {step + 1} recomputed heading index"

            # MOTION / SENSING.
            runtime.step(actions)

            assert np.allclose(
                np.asarray([r.position for r in runtime.robots]),
                np.asarray(reference["final_positions"]), atol=1e-9,
            ), f"step {step + 1} final positions"

            assert np.allclose(
                [r.heading for r in runtime.robots],
                reference["final_headings"], atol=1e-9,
            ), f"step {step + 1} final headings"

            assert runtime.native_belief.explored_free_count == (
                reference["post_belief_free"]
            ), f"step {step + 1} post belief"

            assert runtime.exploration_rate == pytest.approx(
                reference["post_rate"], abs=1e-15
            ), f"step {step + 1} post rate"

            observations = runtime._get_observations()
    finally:
        agent_module.Agent.select_next_waypoint = original
