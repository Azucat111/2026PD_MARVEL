"""Portable runner for the existing paired benchmark and whole-mission scorer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts")]

from gppo_benchmark_domains import load_domain
from gppo_joint_completion_metric import joint_completion_metric
from integrations.gppo.heading_adapter import ExtendedGPPOInferenceAdapter

import benchmark_fair_search as benchmark


def verified_asset(manifest, name):
    row = next(entry for entry in manifest if entry["name"] == name)
    path = PROJECT / row["cloud_path"]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != row["cloud_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch for {name}")
    return path, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--uavs", type=int, choices=(4, 30, 60), default=4)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--full-window", action="store_true")
    parser.add_argument("--wall-budget-seconds", type=float, default=120)
    args = parser.parse_args()
    if args.full_window and args.uavs not in (30, 60):
        parser.error("Full-window comparison requires a 30 or 60 UAV fleet.")

    manifest = json.loads((HERE / "checkpoint_manifest.json").read_text())
    base, _ = verified_asset(manifest, "Frozen native GPPO")
    checkpoint, checkpoint_sha = verified_asset(manifest, "November 11 guarded candidate")
    verified_asset(manifest, "MARVEL low-level actor")

    scenario = PROJECT / "configs/scenarios/extended_connected_gppo_cached_sensing_20261012.yaml"
    original_loader = benchmark.load_and_validate_scenario
    runtime_class = benchmark.SimulationRuntime
    adapter_class = benchmark.MARVELPolicyAdapter
    rows = []
    layouts = []

    class TracedRuntime(runtime_class):
        def __init__(self, config):
            config["task_scheduler"]["extended_checkpoint"] = str(checkpoint)
            config["task_scheduler"]["extended_checkpoint_sha256"] = checkpoint_sha
            super().__init__(config)
            assert self.dt == 0.1

        def step(self, actions):
            observations, info = super().step(actions)
            safety = self.high_level_scheduler.safety
            rows.append(dict(time=self.current_step*self.dt,
                             connectivity=float(info["connectivity_ratio"]),
                             nonrelay_complete=self.tasks.all_complete(),
                             safety_active=sorted(safety.active_uav_ids),
                             safety_records=len(safety.records),
                             escapes=safety.successful_escapes,
                             no_fly_violations=safety.no_fly_violations,
                             collisions=len(info["collisions"])))
            return observations, info

    class CheckedAdapter(adapter_class):
        def setup(self):
            with benchmark.torch.random.fork_rng(
                devices=[benchmark.torch.cuda.current_device()]
                if benchmark.torch.cuda.is_available() else []
            ):
                result = super().setup()
            allocator = self.scheduler.allocator.adapter
            assert isinstance(allocator, ExtendedGPPOInferenceAdapter)
            assert allocator.candidate_sha256 == checkpoint_sha
            assert not allocator.model.training
            hidden = self.runtime.target_detector.hidden_targets
            layouts.append(json.dumps(hidden, sort_keys=True, default=lambda v: v.tolist()))
            return result

    benchmark.CHECKPOINT = base
    benchmark.SimulationRuntime = TracedRuntime
    benchmark.MARVELPolicyAdapter = CheckedAdapter
    benchmark.load_and_validate_scenario = lambda file: load_domain(
        original_loader, file, scenario, args.dynamic, 0.2
    )
    args.out.mkdir(parents=True, exist_ok=False)
    mission_steps = 300 if args.full_window else 4
    results = {}
    for arm in ("rule", "gppo"):
        rows.clear()
        result = benchmark.run(args.out, arm, args.uavs, args.seed, mission_steps,
                               physical_coverage=True, stop_on_completion=False,
                               wall_budget_seconds=args.wall_budget_seconds)
        assert result["mission_steps"] == mission_steps and not result["policy_execution_failures"]
        summary = {"checkpoint_sha256": checkpoint_sha, "search": result["metric"]}
        if args.full_window:
            assert len(rows) == 3000
            summary["joint"] = joint_completion_metric(rows, 300)
        (args.out / arm / "physical_joint_trace.json").write_text(json.dumps(rows))
        results[arm] = summary
    assert layouts[0] == layouts[1]
    for name in ("initial.json", "config.json"):
        assert (args.out/"rule"/name).read_bytes() == (args.out/"gppo"/name).read_bytes()
    (args.out / "paired_cloud_result.json").write_text(json.dumps(results, indent=2))
    print("PAIRED_RUN_COMPLETE", "full-window" if args.full_window else "smoke")


if __name__ == "__main__":
    main()
