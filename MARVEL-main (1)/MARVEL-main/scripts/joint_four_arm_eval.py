"""Single paired-case H300 entry point for a separately frozen experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from verify_joint_atomic import digest, setup, write
from gppo_joint_completion_metric import joint_completion_metric

ARMS = dict(R0="legacy_rule", P0="legacy_gppo", R1="joint_rule", P1="joint_gppo")
HORIZON_SECONDS = 300
PHYSICAL_TICKS = 3000


def evaluate_arm(case, arm, *, p1_model=None):
    runtime, adapter, obs = setup(case, ARMS[arm], p1_model=p1_model)
    checkpoint_sha = adapter.scheduler.allocator.adapter.candidate_sha256
    scheduler = adapter.scheduler
    initial = digest(([asdict(r) for r in runtime.robots], scheduler.heat_points,
                      runtime.target_detector.hidden_targets))
    trace, commitments = [], []
    try:
        if runtime.max_steps != PHYSICAL_TICKS or runtime.dt != .1:
            raise ValueError("Unexpected physical horizon")
        for _ in range(PHYSICAL_TICKS // runtime._mission_step_interval()):
            commands = adapter.get_actions(obs)
            commitments.append([asdict(a) for a in scheduler.last_event_assignments])
            for _ in range(runtime._mission_step_interval()):
                obs, info = runtime.step(commands)
                safety = scheduler.safety
                trace.append(dict(time=runtime.current_step * runtime.dt,
                    connectivity=float(info["connectivity_ratio"]),
                    nonrelay_complete=runtime.tasks.all_complete(),
                    safety_active=sorted(safety.active_uav_ids),
                    safety_records=len(safety.records),
                    escapes=safety.successful_escapes,
                    no_fly_violations=safety.no_fly_violations,
                    collisions=len(info["collisions"])))
        score = joint_completion_metric(trace, HORIZON_SECONDS)
    except Exception as exc:
        return dict(arm=arm, initial_sha256=initial, status="truncated",
                    error=f"{type(exc).__name__}: {exc}", trace=trace, commitments=commitments,
                    checkpoint_sha256=checkpoint_sha)
    return dict(arm=arm, initial_sha256=initial, status="complete", trace=trace,
                commitments=commitments, score=score,
                checkpoint_sha256=checkpoint_sha, compatibility_policy=(arm == "P1" and p1_model is None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--uavs", required=True, type=int, choices=(30, 60))
    parser.add_argument("--domain", required=True, choices=("static", "dynamic"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--p1-checkpoint", type=Path)
    parser.add_argument("--p1-sha256")
    args = parser.parse_args()
    if (args.p1_checkpoint is None) != (args.p1_sha256 is None):
        parser.error("Provide the P1 checkpoint and its frozen SHA-256 together.")
    args.out.mkdir(parents=True, exist_ok=False)
    case = f"{args.uavs}_{args.domain}_{args.seed}"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = {}
    for arm in ARMS:
        model = (args.p1_checkpoint, args.p1_sha256) if arm == "P1" and args.p1_checkpoint else None
        result = evaluate_arm(case, arm, p1_model=model)
        write(args.out / f"{arm}.json", result)
        results[arm] = {key: val for key, val in result.items()
                        if key not in ("trace", "commitments")}
    paired = len({r["initial_sha256"] for r in results.values()}) == 1
    complete = all(r["status"] == "complete" for r in results.values())
    write(args.out / "paired.json", dict(case=case, initial_pairing=paired,
        all_windows_complete=complete, arms=results, acceptance_claim=False))
    return 0 if paired and complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
