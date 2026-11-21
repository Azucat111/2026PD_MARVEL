"""Single paired-case H300 entry point for a separately frozen experiment."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import logging
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from verify_joint_atomic import ROOT, digest, setup, write
from integrations.gppo.heading_adapter import ExtendedGPPOInferenceAdapter
from paired_eval import verified_asset
from gppo_joint_completion_metric import joint_completion_metric

ARMS = dict(R0="legacy_rule", P0="legacy_gppo", R1="joint_rule", P1="joint_gppo")
HORIZON_SECONDS = 300
PHYSICAL_TICKS = 3000


def physical_sample(runtime, info):
    safety = runtime.high_level_scheduler.safety
    return dict(time=runtime.current_step * runtime.dt,
                connectivity=float(info["connectivity_ratio"]),
                nonrelay_complete=runtime.tasks.all_complete(),
                safety_active=sorted(safety.active_uav_ids),
                safety_records=len(safety.records), escapes=safety.successful_escapes,
                no_fly_violations=safety.no_fly_violations, collisions=len(info["collisions"]))


@contextmanager
def require_marvel():
    class RejectFallback(logging.Handler):
        def emit(self, record):
            if any(marker in record.getMessage() for marker in (
                "Policy inference error", "update_graph failed", "update_planning_state failed",
                "Action selection failed", "batched inference failed",
            )):
                raise RuntimeError(record.getMessage())

    logger = logging.getLogger("utils.policy_adapter")
    handler = RejectFallback()
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def evaluate_arm(case, arm, *, p1_model=None, wall_seconds=3000, progress=None, compare_frozen=False):
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("A finite positive wall limit is required")
    start = time.monotonic()
    runtime, adapter, obs = setup(case, ARMS[arm], p1_model=p1_model)
    checkpoint_sha = adapter.scheduler.allocator.adapter.candidate_sha256
    scheduler = adapter.scheduler
    initial = digest(([asdict(r) for r in runtime.robots], scheduler.heat_points,
                      runtime.target_detector.hidden_targets))
    config = {k: v for k, v in runtime.config.items() if k != "task_scheduler"}
    config["task_scheduler"] = {k: v for k, v in runtime.config["task_scheduler"].items()
                                if k not in ("mode", "extended_checkpoint", "extended_checkpoint_sha256")}
    config_sha = digest(config)
    trace, commitments, offers = [], [], []
    reference = None
    if arm == "P1" and compare_frozen:
        assets = json.loads((ROOT / "handoff/cloud/checkpoint_manifest.json").read_text())
        base, _ = verified_asset(assets, "Frozen native GPPO")
        candidate, expected = verified_asset(assets, "November 11 guarded candidate")
        with torch.random.fork_rng():
            reference = ExtendedGPPOInferenceAdapter(base, candidate, expected_sha256=expected)
    inference = scheduler.allocator.adapter
    if arm in ("R1", "P1"):
        choose = scheduler._choose_joint

        def observe(graph):
            search, relay = [], []
            for tid, uid in graph.valid_action_pairs:
                role = graph.subtasks[graph.subtask_id_to_index[tid]].task_type
                (search if int(role) == 1 else relay).append(uid)
            choice = choose(graph)
            row = dict(step=runtime.current_step, search=len(search), relay=len(relay),
                       shared_uavs=len(set(search) & set(relay)))
            if reference is not None:
                # Diagnostic only. Reuse P1's exact input, with no impact on its pick.
                with torch.no_grad():
                    old = reference.model(**inference.last_batch)["flat_logits"].flatten()
                    valid = ~inference.last_batch["action_mask"].flatten()
                    frozen = int(old.masked_fill(~valid, -torch.inf).argmax())
                ti, ui = divmod(frozen, graph.num_uavs)
                row["frozen_greedy"] = dict(task_id=int(graph.subtasks[ti].subtask_id),
                                           uav_id=int(graph.uav_states[ui].uav_id),
                                           role=int(graph.subtasks[ti].task_type))
                row["trained_greedy"] = dict(task_id=choice.subtask_id, uav_id=choice.uav_id,
                                             role=int(graph.subtasks[choice.task_index].task_type))
            offers.append(row)
            return choice

        scheduler._choose_joint = observe
    error, score = None, None
    try:
        with require_marvel():
            if runtime.max_steps != PHYSICAL_TICKS or runtime.dt != .1:
                raise ValueError("Unexpected physical horizon")
            for _ in range(PHYSICAL_TICKS // runtime._mission_step_interval()):
                if time.monotonic() - start >= wall_seconds:
                    raise TimeoutError("Physical window exceeded its wall limit")
                commands = adapter.get_actions(obs)
                commitments.append([asdict(a) for a in scheduler.last_event_assignments])
                for _ in range(runtime._mission_step_interval()):
                    obs, info = runtime.step(commands)
                    trace.append(physical_sample(runtime, info))
                if progress and len(trace) % 100 == 0:
                    progress(dict(arm=arm, case=case, ticks=len(trace), seconds=time.monotonic()-start))
            if len(trace) != PHYSICAL_TICKS:
                raise ValueError("Incomplete physical window")
            score = joint_completion_metric(trace, HORIZON_SECONDS)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    result = dict(case=case, arm=arm, initial_sha256=initial, config_sha256=config_sha,
                  status="complete" if score else "truncated", trace=trace,
                  commitments=commitments, offers=offers, seconds=time.monotonic()-start,
                  checkpoint_sha256=checkpoint_sha, compatibility_policy=(arm == "P1" and p1_model is None))
    if score is not None:
        result["score"] = score
    else:
        result["error"] = error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--uavs", required=True, type=int, choices=(30, 60))
    parser.add_argument("--domain", required=True, choices=("static", "dynamic"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--p1-checkpoint", type=Path)
    parser.add_argument("--p1-sha256")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--wall-seconds", type=float, default=3000, help="Per-arm wall limit")
    parser.add_argument("--compare-frozen", action="store_true",
                        help="Log frozen 6D greedy choices on the actual P1 input graphs")
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error("Choose each arm once.")
    if (args.p1_checkpoint is None) != (args.p1_sha256 is None):
        parser.error("Provide the P1 checkpoint and its frozen SHA-256 together.")
    args.out.mkdir(parents=True, exist_ok=False)
    case = f"{args.uavs}_{args.domain}_{args.seed}"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = {}
    for arm in args.arms:
        model = (args.p1_checkpoint, args.p1_sha256) if arm == "P1" and args.p1_checkpoint else None
        result = evaluate_arm(case, arm, p1_model=model, wall_seconds=args.wall_seconds,
                              progress=lambda row: print(row, flush=True),
                              compare_frozen=args.compare_frozen)
        write(args.out / f"{arm}.json", result)
        results[arm] = {key: val for key, val in result.items()
                        if key not in ("trace", "commitments", "offers")}
    paired = (len(results) > 1 and
              len({(r["initial_sha256"], r["config_sha256"]) for r in results.values()}) == 1)
    complete = all(r["status"] == "complete" for r in results.values())
    write(args.out / "paired.json", dict(case=case, initial_pairing=paired,
        all_windows_complete=complete, arms=results, acceptance_claim=False))
    return 0 if complete and (paired or len(results) == 1) else 1


if __name__ == "__main__":
    raise SystemExit(main())
