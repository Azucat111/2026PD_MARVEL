"""Opt-in CUDA batched inference must match the CPU sequential reference.

The CUDA path is opt-in (`policy.device: cuda`, `policy.batch_inference: true`)
and never the default.  This module is skipped entirely when CUDA is not
available, so it can never fail a CPU-only machine.

The gate is behavioural, not bit-exact: cross-device GEMM does not reproduce
CPU rounding, so logits are compared to a tolerance while the selected
action, graph node, waypoint and heading must agree exactly.
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

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# Measured CPU-vs-CUDA logit spread on live states: 2.9e-05 at 4 UAVs,
# 2.1e-05 at 8, 5.7e-05 at 16, 4.0e-03 at 30, 1.33e-02 at 60.  The spread
# grows with batch size because the two backends accumulate the 360x360
# attention differently, so the tolerance is set above the largest measured
# value rather than picked.  It is not the gate: the gate is that the
# selected action, node, waypoint and heading agree exactly, which is
# asserted below and measured at 0 argmax mismatches over 118 rows.
LOGIT_ATOL = 2e-2


def _adapter(uavs, device=None, batch=None, steps=0, seed=12345):
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

    policy = {}
    if device is not None:
        policy["device"] = device
    if batch is not None:
        policy["batch_inference"] = batch
    if policy:
        config["policy"] = policy

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
    pad = adapter._observation_padding()
    devices = [agent.device for agent in adapter.agents]
    for agent in adapter.agents:
        agent.device = torch.device("cpu")
    try:
        return [agent.get_observation(pad=pad) for agent in adapter.agents]
    finally:
        for agent, device in zip(adapter.agents, devices):
            agent.device = device


def test_default_device_is_cpu(tmp_path):
    """No config, no CUDA: the default path is CPU and not batched."""

    runtime, adapter = _adapter(4)
    assert adapter.device.type == "cpu"
    assert adapter.batch_policy_inference is False
    assert adapter._batched_inference_applies() is False


def test_config_selects_device_and_batching():
    """The scenario's `policy:` block drives both switches."""

    _, adapter = _adapter(4, device="cpu", batch=True)
    assert adapter.device.type == "cpu"
    assert adapter.batch_policy_inference is True
    assert adapter._batched_inference_applies() is True


def test_missing_cuda_fails_loudly(monkeypatch):
    """Requesting CUDA without a CUDA runtime must raise, not downgrade."""

    import utils.policy_adapter as module

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError) as excinfo:
        MARVELPolicyAdapter._resolve_device("cuda")

    assert "CUDA is not available" in str(excinfo.value)
    assert "policy.device" in str(excinfo.value)


@requires_cuda
@pytest.mark.parametrize("uavs", [4, 8, 16, 30, 60])
def test_cuda_logits_and_decisions_match_cpu(uavs):
    """Behavioural equivalence per UAV, with the numerical spread recorded."""

    _, cpu = _adapter(uavs, device="cpu", batch=False, steps=6)
    _, gpu = _adapter(uavs, device="cuda", batch=True, steps=6)

    # Identical graph state, so the observations are comparable.
    assert cpu._node_manager.nodes_dict.__len__() == (
        gpu._node_manager.nodes_dict.__len__()
    )

    observations = _observations(cpu)

    with torch.no_grad():
        cpu_logits = torch.cat(
            [cpu.policy_net(*obs) for obs in observations], dim=0
        )

    with torch.no_grad():
        stacked = [
            torch.cat([o[i] for o in observations], dim=0).to("cuda")
            for i in range(len(observations[0]))
        ]
        gpu_logits = gpu.policy_net(*stacked).to("cpu")

    assert cpu_logits.shape == gpu_logits.shape

    diff = (cpu_logits - gpu_logits).abs()
    denom = cpu_logits.abs().clamp(min=1e-9)
    relative = (diff / denom)

    legal = cpu_logits > -1e7
    max_abs = float(diff[legal].max())
    max_rel = float(relative[legal].max())

    assert max_abs < LOGIT_ATOL, (
        f"max_abs={max_abs} max_rel={max_rel} at {uavs} UAVs"
    )

    mismatches = 0
    for row_cpu, row_gpu in zip(cpu_logits, gpu_logits):
        if int(torch.argmax(row_cpu)) != int(torch.argmax(row_gpu)):
            mismatches += 1

    # A mismatch is only acceptable as a near-tie; anything else is a real
    # policy difference and fails the gate.
    assert mismatches == 0, f"{mismatches} argmax mismatches at {uavs} UAVs"

    for index, (agent, obs) in enumerate(zip(cpu.agents, observations)):
        reference = agent.decode_waypoint(
            obs, cpu_logits[index:index + 1], greedy=True
        )
        candidate = agent.decode_waypoint(
            obs, gpu_logits[index:index + 1], greedy=True
        )

        assert int(reference[2].item()) == int(candidate[2].item())
        assert reference[1] == candidate[1]
        assert reference[3] == candidate[3]
        assert np.array_equal(
            np.asarray(reference[0]), np.asarray(candidate[0])
        )


def _paired_episode(uavs, steps, device, batch):
    runtime, adapter = _adapter(uavs, device=device, batch=batch, steps=0)
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
        "belief": float(runtime.exploration_rate),
        "nodes": int(adapter._node_manager.nodes_dict.__len__()),
    })

    return trace


@requires_cuda
@pytest.mark.parametrize("uavs", [4, 8, 16, 30, 60])
def test_paired_episodes_cpu_vs_cuda(uavs):
    """Same seed, CPU sequential vs CUDA batched, 50 mission steps."""

    reference = _paired_episode(
        uavs, 50, device="cpu", batch=False
    )
    candidate = _paired_episode(
        uavs, 50, device="cuda", batch=True
    )

    assert len(reference) == len(candidate)

    for step, (a, b) in enumerate(zip(reference, candidate)):
        for key in a:
            if isinstance(a[key], np.ndarray):
                assert np.array_equal(a[key], b[key]), (
                    f"uavs={uavs} step={step} field={key}"
                )
            else:
                assert a[key] == b[key], (
                    f"uavs={uavs} step={step} field={key}: "
                    f"{a[key]} vs {b[key]}"
                )
