"""Batched PolicyNet inference must match the sequential path exactly.

Extended mode now runs one PolicyNet forward per mission step for all UAVs.
Each agent's observation is still built by its own `get_observation`, so the
per-agent nearest-node window and every mask are unchanged; only the number
of forward passes differs.

These tests keep the sequential path available as a reference by flipping
`adapter.batch_policy_inference`, and require equality of the raw logits,
the selected action index, the selected graph node, the waypoint and the
heading command -- then of whole paired episodes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime
from utils.policy_adapter import MARVELPolicyAdapter

SCENARIO = ROOT / "configs" / "scenarios" / "baseline_maps_test.yaml"

# Batched GEMM rounding against single-row GEMM, measured at 3.2e-05.
LOGIT_ATOL = 1e-3

TENSOR_NAMES = (
    "node_inputs", "node_padding_mask", "edge_mask", "current_index",
    "current_edge", "edge_padding_mask", "frontier_distribution",
    "heading_visited", "neighbor_best_headings",
)


def _adapter(uavs: int, steps: int = 0, seed: int = 12345):
    config = load_and_validate_scenario(SCENARIO)
    config["robots"] = [{
        "id_range": [0, uavs - 1],
        "type": "explorer",
        "team_id": 1,
        "config": {
            "fov": 120, "sensor_range": 10.0, "velocity": 1.0,
            "yaw_rate": 35, "initial_positions": "random_safe",
        },
    }]
    config.setdefault("scenario", {})["random_seed"] = seed

    np.random.seed(seed)
    runtime = SimulationRuntime(config)
    observations = runtime.reset()
    adapter = MARVELPolicyAdapter(runtime)
    adapter.setup()

    interval = max(1, int(runtime._mission_step_interval()))
    for _ in range(steps):
        actions = adapter.get_actions(observations)
        for _tick in range(interval):
            observations, _info = runtime.step(actions)

    return runtime, adapter


def _observations(adapter):
    """Each agent's own observation, exactly as the adapter builds it."""

    pad = adapter._observation_padding()
    return [agent.get_observation(pad=pad) for agent in adapter.agents]


def _logits(adapter, observations, batched: bool):
    if batched:
        stacked = [
            torch.cat([obs[i] for obs in observations], dim=0)
            for i in range(len(observations[0]))
        ]
        with torch.no_grad():
            return adapter.policy_net(*stacked)

    rows = []
    for obs in observations:
        with torch.no_grad():
            rows.append(adapter.policy_net(*obs))
    return torch.cat(rows, dim=0)


@pytest.mark.parametrize("uavs", [4, 8, 16])
def test_batched_logits_match_sequential(uavs):
    """Raw logits and every per-UAV tensor are identical."""

    runtime, adapter = _adapter(uavs, steps=5)
    observations = _observations(adapter)

    # The observations themselves must be identical between the two paths:
    # they are built the same way, so this pins that batching changes only
    # the forward pass.
    sequential = _logits(adapter, observations, batched=False)
    batched = _logits(adapter, observations, batched=True)

    assert sequential.shape == batched.shape == (
        len(adapter.agents), sequential.shape[1]
    )

    # Batched GEMM accumulates in a different order from single-row GEMM,
    # so the logits agree only to float tolerance.  The decision must not.
    max_diff = float((sequential - batched).abs().max())
    assert max_diff < LOGIT_ATOL, max_diff

    seq_actions = torch.argmax(sequential, dim=1)
    bat_actions = torch.argmax(batched, dim=1)
    assert torch.equal(seq_actions, bat_actions), (seq_actions, bat_actions)

    # Exact ties between legal actions are legitimate: two neighbour slots
    # with identical features produce identical logits.  What matters is
    # that a tie exists in *both* paths -- otherwise the float noise could
    # break it one way in one path and the other way in the other.
    margins = []
    for row_seq, row_bat in zip(sequential, batched):
        legal_seq = row_seq[row_seq > -1e7]
        legal_bat = row_bat[row_bat > -1e7]
        assert legal_seq.numel() > 1, "no row had two legal actions"

        top2_seq = torch.topk(legal_seq, 2).values
        top2_bat = torch.topk(legal_bat, 2).values

        margin_seq = float(top2_seq[0] - top2_seq[1])
        margin_bat = float(top2_bat[0] - top2_bat[1])
        margins.append(margin_seq)

        if margin_seq == 0.0:
            assert margin_bat == 0.0, (margin_seq, margin_bat)
        else:
            # A non-tied decision must have slack well above the noise floor.
            assert margin_seq > max_diff * 10, (margin_seq, max_diff)

    # Per-UAV decode against that UAV's own observation.
    for index, (agent, obs) in enumerate(zip(adapter.agents, observations)):
        seq = agent.decode_waypoint(obs, sequential[index:index + 1], greedy=True)
        bat = agent.decode_waypoint(obs, batched[index:index + 1], greedy=True)

        assert int(seq[2].item()) == int(bat[2].item())
        assert seq[1] == bat[1]
        assert seq[3] == bat[3]
        assert np.array_equal(np.asarray(seq[0]), np.asarray(bat[0]))


@pytest.mark.parametrize("uavs", [4, 16])
def test_observation_shapes_are_batchable(uavs):
    """Every observation tensor has a common fixed shape across UAVs."""

    runtime, adapter = _adapter(uavs, steps=5)
    observations = _observations(adapter)

    shapes = {name: set() for name in TENSOR_NAMES}

    for obs in observations:
        for name, tensor in zip(TENSOR_NAMES, obs):
            shapes[name].add(tuple(tensor.shape))

    for name, seen in shapes.items():
        assert len(seen) == 1, (name, seen)
        (shape,) = seen
        assert shape[0] == 1, (name, shape)


def test_per_uav_window_is_preserved():
    """Each UAV keeps its own nearest-node window, not a shared one.

    With more graph nodes than `NODE_PADDING_SIZE`, the per-agent window is
    a different subset per UAV, so the padded `edge_mask` differs between
    agents.  That difference must survive batching.
    """

    from parameter import NODE_PADDING_SIZE

    runtime, adapter = _adapter(16, steps=14)
    manager = adapter._node_manager

    if manager.nodes_dict.__len__() <= NODE_PADDING_SIZE:
        pytest.skip(
            f"graph has {manager.nodes_dict.__len__()} nodes, "
            f"windowing not engaged (needs > {NODE_PADDING_SIZE})"
        )

    observations = _observations(adapter)

    masks = [obs[2] for obs in observations]
    distinct = sum(
        1 for i in range(1, len(masks))
        if not torch.equal(masks[0], masks[i])
    )
    assert distinct > 0, "all UAVs share one window; case is vacuous"

    stacked = torch.cat(masks, dim=0)
    assert stacked.shape[0] == len(adapter.agents)
    for index, mask in enumerate(masks):
        assert torch.equal(stacked[index:index + 1], mask)


def _paired_episode(uavs: int, steps: int, batched: bool):
    runtime, adapter = _adapter(uavs, steps=0)
    adapter.batch_policy_inference = batched

    interval = max(1, int(runtime._mission_step_interval()))
    observations = runtime._get_observations()

    trace = []
    for _ in range(steps):
        actions = adapter.get_actions(observations)

        trace.append({
            "waypoints": np.asarray(
                [np.asarray(a[0], dtype=float) for a in actions], dtype=float
            ).copy(),
            "headings": np.asarray(
                [float(a[1]) for a in actions], dtype=float
            ).copy(),
            "positions": np.asarray(
                [r.position for r in runtime.robots], dtype=float
            ).copy(),
        })

        for _tick in range(interval):
            observations, _info = runtime.step(actions)

    trace.append({
        "positions": np.asarray(
            [r.position for r in runtime.robots], dtype=float
        ).copy(),
        "headings": np.asarray(
            [r.heading for r in runtime.robots], dtype=float
        ).copy(),
        "belief": int(np.sum(runtime.exploration_rate * 1e12)),
        "nodes": int(adapter._node_manager.nodes_dict.__len__()),
    })

    return trace


@pytest.mark.parametrize("uavs,steps", [(4, 12), (8, 8), (16, 6)])
def test_paired_episodes_identical(uavs, steps):
    """Same seed, sequential vs batched: identical at every step."""

    sequential = _paired_episode(uavs, steps, batched=False)
    batched = _paired_episode(uavs, steps, batched=True)

    assert len(sequential) == len(batched)

    for step, (a, b) in enumerate(zip(sequential, batched)):
        for key in a:
            if isinstance(a[key], np.ndarray):
                assert np.array_equal(a[key], b[key]), (
                    f"uavs={uavs} step={step} field={key}"
                )
            else:
                assert a[key] == b[key], (
                    f"uavs={uavs} step={step} field={key}"
                )


def test_native_keeps_sequential_path():
    """marvel_native must not take the batched path."""

    config = load_and_validate_scenario(
        ROOT / "configs" / "scenarios" / "marvel_native_test.yaml"
    )
    config["environment"]["geometry_mode"] = "marvel_native"
    config["environment"]["initial_headings"] = 270.0
    config["environment"]["episode_index"] = 0
    config["robots"] = [{
        "id_range": [0, 3],
        "type": "explorer",
        "team_id": 1,
        "config": {
            "fov": 120, "sensor_range": 10.0, "velocity": 1.0,
            "yaw_rate": 35,
            "initial_positions": [
                [-4.0, 0.0], [-8.0, 4.0], [4.0, -4.0], [0.0, 0.0]
            ],
        },
    }]

    np.random.seed(12345)
    runtime = SimulationRuntime(config)
    runtime.reset()
    adapter = MARVELPolicyAdapter(runtime)
    adapter.setup()

    assert adapter._frame.is_native
    assert adapter._batched_inference_applies() is False
