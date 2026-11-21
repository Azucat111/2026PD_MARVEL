"""PolicyNet device benchmark: CPU sequential, CPU batched, CUDA batched.

Measurement only; the device is selected through the scenario's `policy:`
block, so nothing in production changes.

Usage:
    python scripts/benchmark_policy_devices.py --uavs 4 8 16 30 60
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from utils.scenario_config import load_and_validate_scenario  # noqa: E402
from utils.simulation_runtime import SimulationRuntime  # noqa: E402
from utils.policy_adapter import MARVELPolicyAdapter  # noqa: E402
from parameter import NUM_ANGLES_BIN  # noqa: E402

SCENARIO = ROOT / "configs" / "scenarios" / "baseline_maps_test.yaml"

MODES = (
    ("cpu_sequential", "cpu", False),
    ("cpu_batched", "cpu", True),
    ("cuda_batched", "cuda", True),
)


def build(uavs, device, batch, steps):
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
    config.setdefault("scenario", {})["random_seed"] = 12345
    config["policy"] = {"device": device, "batch_inference": batch}

    np.random.seed(12345)
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


def measure(uavs, device, batch, warmup, measured):
    import psutil

    process = psutil.Process()

    if device == "cuda" and not torch.cuda.is_available():
        return None

    try:
        runtime, adapter = build(uavs, device, batch, warmup)
    except RuntimeError as exc:
        return {"uavs": uavs, "device": device, "batch": batch,
                "status": f"error: {exc}"}

    interval = max(1, int(runtime._mission_step_interval()))
    observations = runtime._get_observations()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Split the PolicyNet cost into packing / transfer / forward / decode.
    pack = transfer_h2d = forward = transfer_d2h = decode = 0.0
    vram_series = []
    step_times = []

    original_batched = adapter._batched_actions

    def instrumented(default):
        nonlocal pack, transfer_h2d, forward, transfer_d2h, decode

        pad = adapter._observation_padding()
        devices = [agent.device for agent in adapter.agents]

        start = time.perf_counter()
        try:
            for agent in adapter.agents:
                agent.device = torch.device("cpu")
            observations_list = [
                agent.get_observation(pad=pad) for agent in adapter.agents
            ]
        finally:
            for agent, dev in zip(adapter.agents, devices):
                agent.device = dev
        pack += time.perf_counter() - start

        start = time.perf_counter()
        stacked = [
            torch.cat([o[i] for o in observations_list], dim=0).to(adapter.device)
            for i in range(len(observations_list[0]))
        ]
        if device == "cuda":
            torch.cuda.synchronize()
        transfer_h2d += time.perf_counter() - start

        start = time.perf_counter()
        with torch.no_grad():
            logits = adapter.policy_net(*stacked)
        if device == "cuda":
            torch.cuda.synchronize()
        forward += time.perf_counter() - start

        start = time.perf_counter()
        logits = logits.to("cpu")
        if device == "cuda":
            torch.cuda.synchronize()
        transfer_d2h += time.perf_counter() - start

        start = time.perf_counter()
        actions = []
        for index, (agent, obs) in enumerate(
            zip(adapter.agents, observations_list)
        ):
            next_position, _n, _a, heading_index = agent.decode_waypoint(
                obs, logits[index:index + 1], greedy=True
            )
            actions.append(
                (np.asarray(next_position, dtype=float),
                 float(heading_index) * (360.0 / NUM_ANGLES_BIN))
            )
        decode += time.perf_counter() - start
        return actions

    # Use the production path exactly, but time its parts.
    adapter._batched_actions = instrumented

    peak_reserved = []

    try:
        for _ in range(measured):
            start = time.perf_counter()
            actions = adapter.get_actions(observations)
            for _tick in range(interval):
                observations, _info = runtime.step(actions)
            step_times.append(time.perf_counter() - start)

            if device == "cuda":
                peak_reserved.append(
                    float(torch.cuda.memory_reserved() / (1024 * 1024))
                )
    finally:
        adapter._batched_actions = original_batched

    ordered = sorted(step_times)
    result = {
        "uavs": uavs,
        "device": device,
        "batch": batch,
        "status": "ok",
        "steps_measured": len(step_times),
        "mean_step_seconds": statistics.fmean(step_times),
        "p50_step_seconds": ordered[len(ordered) // 2],
        "p95_step_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "pack_ms": pack / max(1, len(step_times)) * 1000,
        "h2d_ms": transfer_h2d / max(1, len(step_times)) * 1000,
        "forward_ms": forward / max(1, len(step_times)) * 1000,
        "d2h_ms": transfer_d2h / max(1, len(step_times)) * 1000,
        "decode_ms": decode / max(1, len(step_times)) * 1000,
        "policynet_total_ms": (
            (pack + transfer_h2d + forward + transfer_d2h + decode)
            / max(1, len(step_times)) * 1000
        ),
        "ram_mb": float(process.memory_info().rss / (1024 * 1024)),
        "graph_nodes": int(adapter._node_manager.nodes_dict.__len__()),
    }

    if device == "cuda":
        result["vram_peak_allocated_mb"] = float(
            torch.cuda.max_memory_allocated() / (1024 * 1024)
        )
        result["vram_peak_reserved_mb"] = float(
            torch.cuda.max_memory_reserved() / (1024 * 1024)
        )
        result["vram_series_mb"] = peak_reserved
        result["vram_monotonic_growth_mb"] = (
            peak_reserved[-1] - peak_reserved[0] if peak_reserved else 0.0
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uavs", type=int, nargs="+", default=[4, 8, 16, 30, 60])
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--measured", type=int, default=20)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else (
        ROOT / "artifacts" / f"extended_cuda_policy_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for uavs in args.uavs:
        for name, device, batch in MODES:
            if device == "cuda" and not torch.cuda.is_available():
                continue
            print(f"[devices] uavs={uavs} {name} ...", flush=True)
            r = measure(uavs, device, batch, args.warmup, args.measured)
            if r is None:
                continue
            results.append(r)
            print(
                f"[devices] uavs={uavs} {name} status={r['status']} "
                f"step={r.get('mean_step_seconds')} pn={r.get('policynet_total_ms')}",
                flush=True,
            )

    (out / "profile.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    with open(out / "before_after.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "uavs", "mode", "device", "batch", "status", "mean_step_s",
            "p50_step_s", "p95_step_s", "policynet_total_ms", "pack_ms",
            "h2d_ms", "forward_ms", "d2h_ms", "decode_ms", "ram_mb",
            "vram_peak_allocated_mb", "vram_peak_reserved_mb", "graph_nodes",
        ])
        for r in results:
            w.writerow([
                r["uavs"], f"{r['device']}{'_batched' if r['batch'] else '_seq'}",
                r["device"], r["batch"], r["status"],
                f"{r.get('mean_step_seconds', float('nan')):.4f}",
                f"{r.get('p50_step_seconds', float('nan')):.4f}",
                f"{r.get('p95_step_seconds', float('nan')):.4f}",
                f"{r.get('policynet_total_ms', float('nan')):.2f}",
                f"{r.get('pack_ms', float('nan')):.2f}",
                f"{r.get('h2d_ms', float('nan')):.2f}",
                f"{r.get('forward_ms', float('nan')):.2f}",
                f"{r.get('d2h_ms', float('nan')):.2f}",
                f"{r.get('decode_ms', float('nan')):.2f}",
                f"{r.get('ram_mb', float('nan')):.0f}",
                f"{r.get('vram_peak_allocated_mb', '')}",
                f"{r.get('vram_peak_reserved_mb', '')}",
                r.get("graph_nodes"),
            ])

    with open(out / "cuda_memory.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "uavs", "vram_peak_allocated_mb", "vram_peak_reserved_mb",
            "vram_series_mb", "growth_mb_over_run",
        ])
        for r in results:
            if r["device"] != "cuda" or r["status"] != "ok":
                continue
            w.writerow([
                r["uavs"], r["vram_peak_allocated_mb"],
                r["vram_peak_reserved_mb"],
                ";".join(f"{v:.1f}" for v in r["vram_series_mb"]),
                f"{r['vram_monotonic_growth_mb']:.2f}",
            ])

    print(f"[devices] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
