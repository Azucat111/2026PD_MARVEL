"""Bounded real H300 runs for public-state action-return experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from joint_four_arm_eval import physical_sample, require_marvel
from joint_policy_causal_trace import Recorder, FROZEN_SHA, baseline_metadata
from gppo_joint_completion_metric import joint_completion_metric
from integrations.gppo.adapter import GPPOAssignment
from integrations.gppo.mesh_relay import ObservedMeshRelayDemandBuilder
from verify_joint_atomic import ROOT, SCENARIO, setup, write

HORIZON = 3000


def action_at(graph, flat, value):
    ti, ui = divmod(int(flat), graph.num_uavs)
    if graph.action_mask[ti, ui]:
        raise ValueError("Attempted override of an illegal action")
    slot, uid = graph.action_ids(flat)
    return GPPOAssignment(int(flat), ti, ui, int(slot), int(uid), float(value))


class ExperimentRecorder(Recorder):
    def __init__(self, scheduler, events, snapshots, *, policy, step=None, pair=None, ranker=None):
        super().__init__(scheduler, "J0", {"J0": scheduler.allocator.adapter.model}, events, snapshots)
        self.policy = policy
        self.override_step = step
        self.override_pair = pair
        self.ranker = ranker
        self.intervened = False

    def observe_choice(self, graph):
        normal = super().observe_choice(graph)
        record = self.event
        rows = {tuple((c["slot"], c["uav"])): c for c in record["candidates"]}
        prior = record["actors"]["J0"]
        indices = prior["flat_indices"]
        record["rows"] = []
        for flat, logit in zip(indices, prior["logits"], strict=True):
            ti, ui = divmod(flat, graph.num_uavs)
            pair = tuple(map(int, graph.action_ids(flat)))
            entry = rows[pair]
            record["rows"].append(dict(flat=flat, slot=pair[0], uav=pair[1],
                                      role=entry["role"], p=graph.uav_states[ui].position,
                                      anchor=entry["anchor"], waypoint=entry["waypoint"],
                                      total=float(graph.edge_features[ti, ui, 2]), logit=logit))
        record["range_m"] = self.scheduler.relay_builder.comm_range
        record["time_seconds"] = self.scheduler.runtime.current_step*self.scheduler.runtime.dt
        choice = normal
        if record["step"] == self.override_step:
            if self.policy == "pair":
                options = [r for r in record["rows"] if (r["slot"], r["uav"]) == self.override_pair]
                if len(options) != 1:
                    raise ValueError("Preregistered pair unavailable")
                flat = options[0]["flat"]
            elif self.policy in ("relay", "nearest_relay"):
                options = [r for r in record["rows"] if r["role"] == 2]
                if not options or not any(r["role"] == 1 for r in record["rows"]):
                    raise ValueError("Expected a real mixed graph at the intervention")
                key = (lambda r: (r["total"], r["flat"])) if self.policy == "nearest_relay" else (
                    lambda r: (-r["logit"], r["flat"]))
                flat = min(options, key=key)["flat"]
            else:
                raise ValueError("Unknown intervention")
            choice = action_at(graph, flat, normal.value)
            self.intervened = True
            record["mode"] = "single_intervention"
        if self.ranker is not None:
            flat, evidence = self.ranker.choose(record)
            choice = action_at(graph, flat, normal.value)
            record["ranker"] = evidence
            self.intervened |= choice.flat_action != normal.flat_action
        record["selected"] = [choice.subtask_id, choice.uav_id]
        return choice


def run(args):
    if not 0 < args.ticks <= HORIZON or args.ticks % 10:
        raise ValueError("Tick budget must align with the controller")
    if args.policy in ("pair", "relay", "nearest_relay") and args.step is None:
        raise ValueError("Single-action interventions require an explicit control step")
    if args.policy == "pair" and args.pair is None:
        raise ValueError("Pair intervention requires slot and UAV")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    runtime, adapter, obs = setup(args.case, "joint_gppo")
    scheduler = adapter.scheduler
    initial, config = baseline_metadata(runtime, scheduler)
    ranker = None
    checkpoint_sha = None
    if args.policy == "learned":
        from integrations.gppo.public_pair_preference import PublicPairPreference

        if args.checkpoint is None or args.sha256 is None:
            raise ValueError("A frozen learned checkpoint and its hash are required")
        ranker = PublicPairPreference.load(args.checkpoint, args.sha256)
        checkpoint_sha = args.sha256
    if scheduler.allocator.adapter.candidate_sha256 != FROZEN_SHA:
        raise ValueError("Incorrect J0 behavior weights")
    trace, commitments = [], []
    failure = None
    with (gzip.open(args.out/"events.jsonl.gz", "wt") as events,
          ZipFile(args.out/"public.zip", "w", compression=ZIP_DEFLATED, compresslevel=1) as snapshots):
        recorder = ExperimentRecorder(scheduler, events, snapshots, policy=args.policy,
                                      step=args.step, pair=tuple(args.pair) if args.pair else None, ranker=ranker)
        def monitor(builder, positions, **kwargs):
            return recorder.observe_demands(builder, positions, **kwargs)

        try:
            with require_marvel(), patch.object(ObservedMeshRelayDemandBuilder, "build", monitor):
                while runtime.current_step < args.ticks:
                    if time.monotonic() - start >= args.wall:
                        raise TimeoutError("Wall budget reached")
                    action = adapter.get_actions(obs)
                    recorder.finish_event(action)
                    commitments.append([asdict(row) for row in scheduler.last_event_assignments])
                    for _ in range(runtime._mission_step_interval()):
                        obs, info = runtime.step(action)
                        trace.append(physical_sample(runtime, info))
                    if runtime.current_step % 100 == 0:
                        print(json.dumps(dict(case=args.case, policy=args.policy, ticks=runtime.current_step,
                                              seconds=round(time.monotonic()-start, 2))), flush=True)
            if args.step is not None and not recorder.intervened:
                raise ValueError("Intervention was never applied")
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    score = joint_completion_metric(trace,300.) if len(trace)==HORIZON and failure is None else None
    write(args.out/"summary.json", dict(case=args.case, policy=args.policy, initial_sha256=initial,
          config_sha256=config, scenario_sha256=hashlib.sha256(SCENARIO.read_bytes()).hexdigest(),
          driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          base_sha256=FROZEN_SHA, checkpoint_sha256=checkpoint_sha, intervention_step=args.step,
          requested_pair=args.pair, score=score, status="complete" if score else ("truncated" if failure else "prefix"),
          intervened=recorder.intervened, error=failure, clock_ticks=len(trace), trace=trace, commitments=commitments,
          seconds=time.monotonic()-start))
    if failure:
        raise RuntimeError(failure)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--case", required=True)
    cli.add_argument("--policy", choices=["J0", "pair", "relay", "nearest_relay", "learned"], required=True)
    cli.add_argument("--step", type=int)
    cli.add_argument("--pair", nargs=2, type=int)
    cli.add_argument("--ticks", type=int, default=HORIZON)
    cli.add_argument("--wall", type=float, default=3000.)
    cli.add_argument("--checkpoint", type=Path)
    cli.add_argument("--sha256")
    cli.add_argument("--out", type=Path, required=True)
    run(cli.parse_args())


if __name__ == "__main__":
    main()
