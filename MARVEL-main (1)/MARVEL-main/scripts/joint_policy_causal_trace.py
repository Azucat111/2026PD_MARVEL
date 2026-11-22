"""Read-only Phase 3.5 telemetry for the frozen 30-static physical pairing."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from joint_four_arm_eval import physical_sample, require_marvel
from verify_joint_atomic import ROOT, SCENARIO, digest, setup, write
from gppo_joint_completion_metric import joint_completion_metric
from integrations.gppo.adapter import GPPOAssignment
from integrations.gppo.heading_adapter import ExtendedGPPOInferenceAdapter
from integrations.gppo.heading_features import heading_batch
from integrations.gppo.mesh_relay import ObservedMeshRelayDemandBuilder
from paired_eval import verified_asset

BASELINE = "3d070512dd7701b79d1dc2f6c4258aa6440b1988"
CASE = "30_static_1280021"
ARCHIVE = ROOT / "artifacts/joint_ppo_20261122"
TRAINED = ARCHIVE / "main-update/checkpoint.pth"
TRAINED_SHA = "d464796efcf866add38d3851cad4512bd46834ebdc6e30bb6a1f433511980c55"
FROZEN_SHA = "c419c175619fe1ea27bbf4da25ad43a7e4a8d6e74ec6bd6f7ec09d5cbc97c69d"
SWAP_STEP, SWAP_ACTION = 30, (100000, 14)


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj))


def line(stream, row):
    stream.write(json.dumps(row, default=json_default, allow_nan=False, separators=(",", ":")) + "\n")


def relay_assignments(scheduler):
    return {int(target): [int(uid), np.asarray(anchor).tolist()]
            for target, (uid, anchor) in scheduler.relay_assignments.items()}


def probe_state(scheduler):
    probe = scheduler._relay_probe
    return deepcopy(probe)


def pair_tuple(choice):
    return None if choice is None else [int(choice.subtask_id), int(choice.uav_id)]


def tensor_digest(batch):
    sha = hashlib.sha256()
    for key, tensor in sorted(batch.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        sha.update(json.dumps((key, str(array.dtype), array.shape)).encode())
        sha.update(array.tobytes())
    return sha.hexdigest()


@torch.no_grad()
def actor_audit(model, batch, graph):
    out = model(**batch)
    valid = ~batch["action_mask"].flatten()
    logits = out["flat_logits"].flatten()[valid]
    if not torch.isfinite(logits).all() or not len(logits):
        raise ValueError("Actor audit expects finite legal logits")
    indices = valid.nonzero().flatten().tolist()
    roles = [int(graph.subtasks[i//graph.num_uavs].task_type) for i in indices]
    relay = torch.tensor([role == 2 for role in roles])
    pick = indices[int(logits.argmax())]
    ti, ui = divmod(pick, graph.num_uavs)
    prob1 = logits.softmax(0)
    prob3 = (logits/3).softmax(0)
    return dict(pick=[int(graph.subtasks[ti].subtask_id), int(graph.uav_states[ui].uav_id)],
                flat_indices=indices, logits=logits.tolist(), prob_t1=prob1.tolist(), prob_t3=prob3.tolist(),
                relay_mass_t1=float(prob1[relay].sum()), relay_mass_t3=float(prob3[relay].sum()),
                value=float(out["value"].reshape(())))


class Recorder:
    def __init__(self, scheduler, arm, models, events, snapshots):
        self.scheduler = scheduler
        self.arm = arm
        self.models = models
        self.events = events
        self.snapshots = snapshots
        self.event = None
        self.last_event = None
        self.inference = scheduler.allocator.adapter
        self.enumerate = scheduler.joint_planner.enumerate
        self.allocate = scheduler._allocate_event
        self.choose = scheduler._choose_joint
        self.demand_build = ObservedMeshRelayDemandBuilder.build
        scheduler.joint_planner.enumerate = self.observe_plan
        scheduler._allocate_event = self.observe_allocation
        scheduler._choose_joint = self.observe_choice

    def observe_allocation(self, stale):
        s = self.scheduler
        self.event = dict(step=s.runtime.current_step, stale_xy=stale, demands=[],
                          search_before=dict(s.search_assignments), relay_before=relay_assignments(s),
                          probe_before=probe_state(s), safety_owners=sorted(s.safety.active_uav_ids),
                          unavailable=[int(r.robot_id) for r in s.runtime.robots if not r.available])
        self.allocate(stale)

    def observe_demands(self, builder, positions, **kwargs):
        result = self.demand_build(builder, positions, **kwargs)
        if self.event is not None:
            self.event["demands"].append(
                dict(connectivity=float(result.snapshot.connectivity_ratio),
                     connected=list(result.snapshot.connected_uav_ids),
                     isolated=list(result.snapshot.isolated_uav_ids),
                     demands=[asdict(demand) for demand in result.demands],
                     slots=[int(slot.slot_id) for slot in result.slots]))
        return result

    def observe_plan(self, snapshot):
        plan = self.enumerate(snapshot)
        graph = plan.graph
        self.event.update(snapshot_version=snapshot.version,
                          task_ids=[int(t.subtask_id) for t in graph.subtasks],
                          roles=[int(t.task_type) for t in graph.subtasks],
                          uav_ids=[int(u.uav_id) for u in graph.uav_states],
                          mask=graph.action_mask.tolist(),
                          candidates=[dict(slot=int(slot), uav=int(uid), role=int(edge.slot.task_type),
                                           anchor=list(edge.slot.position), waypoint=list(edge.waypoint),
                                           heading=float(edge.heading))
                                      for (slot, uid), edge in sorted(plan.candidates.items())],
                          rejected=[[int(slot), int(uid), list(reasons)]
                                    for (slot, uid), reasons in sorted(plan.rejected.items())],
                          release_gate=plan.relay_gate, retarget=plan.retarget,
                          refresh_probe=plan.refresh_probe)
        if plan.candidates or snapshot.step in (230, 240, 250):
            self.snapshots.writestr(f"{snapshot.step:04d}.json", snapshot.public_bytes)
        return plan

    def observe_choice(self, graph):
        choice = self.choose(graph)
        batch = (heading_batch(graph, self.inference.heading_observations, "cpu")
                 if self.arm == "R1" else self.inference.last_batch)
        self.event["input_sha256"] = tensor_digest(batch)
        self.event["actors"] = {label: actor_audit(model, batch, graph)
                                 for label, model in self.models.items()}
        self.event["normal_choice"] = pair_tuple(choice)
        policy = "P1" if self.arm == "P1" else "J0"
        if self.arm != "R1" and self.event["normal_choice"] != self.event["actors"][policy]["pick"]:
            raise ValueError("Paired logits disagree with the actual greedy selector")
        if self.arm == "J0_SWAP_S3" and self.event["step"] == SWAP_STEP:
            ti = graph.subtask_id_to_index[SWAP_ACTION[0]]
            ui = graph.uav_id_to_index[SWAP_ACTION[1]]
            if graph.action_mask[ti, ui]:
                raise ValueError("Preregistered replacement is masked")
            choice = GPPOAssignment(graph.flatten_action(ti, ui), ti, ui, *SWAP_ACTION, float(choice.value))
        self.event["selected"] = pair_tuple(choice)
        return choice

    def finish_event(self, commands):
        if self.event is None:
            return
        s = self.scheduler
        receipt = s.last_joint_receipt
        if receipt.step != self.event["step"] or receipt.version != self.event["snapshot_version"]:
            raise ValueError("Unmatched real atomic receipt")
        actual = None if receipt.assignment is None else asdict(receipt.assignment)
        selected = self.event.get("selected")
        if selected != (None if actual is None else [actual["slot_id"], actual["uav_id"]]):
            raise ValueError("Selected stable IDs did not commit")
        self.event.update(receipt=actual, search_after=dict(s.search_assignments),
                          relay_after=relay_assignments(s), probe_after=probe_state(s),
                          command_sha256=digest(commands))
        line(self.events, self.event)
        self.last_event = self.event["step"]
        self.event = None


def baseline_metadata(runtime, scheduler):
    initial = digest(([asdict(r) for r in runtime.robots], scheduler.heat_points,
                      runtime.target_detector.hidden_targets))
    config = {k: v for k, v in runtime.config.items() if k != "task_scheduler"}
    config["task_scheduler"] = {k: v for k, v in runtime.config["task_scheduler"].items()
                                if k not in ("mode", "extended_checkpoint", "extended_checkpoint_sha256")}
    return initial, digest(config)


def run(arm, output, ticks, wall_seconds):
    if not 0 < ticks <= 3000 or ticks % 10:
        raise ValueError("Choose a 1–300 second control-aligned prefix")
    if arm == "J0" and ticks != 3000:
        raise ValueError("J0 control must run the full H300")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((ROOT/"handoff/cloud/checkpoint_manifest.json").read_text())
    base, base_sha = verified_asset(manifest, "Frozen native GPPO")
    frozen, frozen_sha = verified_asset(manifest, "November 11 guarded candidate")
    _, marvel_sha = verified_asset(manifest, "MARVEL low-level actor")
    with TRAINED.open("rb") as handle:
        trained_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if frozen_sha != FROZEN_SHA or trained_sha != TRAINED_SHA:
        raise ValueError("Unexpected frozen treatment checkpoint identity")
    runtime, adapter, obs = setup(CASE, "joint_rule" if arm == "R1" else "joint_gppo",
                                  p1_model=(TRAINED, TRAINED_SHA) if arm == "P1" else None)
    scheduler = adapter.scheduler
    inference = scheduler.allocator.adapter
    initial, config = baseline_metadata(runtime, scheduler)
    with torch.random.fork_rng():
        if arm == "P1":
            j0 = ExtendedGPPOInferenceAdapter(base, frozen, expected_sha256=FROZEN_SHA).model
            p1 = inference.model
        else:
            j0 = inference.model
            p1 = ExtendedGPPOInferenceAdapter(base, TRAINED, expected_sha256=TRAINED_SHA).model
    trace, commitments = [], []
    with (gzip.open(output/"ticks.jsonl.gz", "wt") as tick_out,
          gzip.open(output/"events.jsonl.gz", "wt") as event_out,
          ZipFile(output/"public.zip", "w", compression=ZIP_DEFLATED, compresslevel=1) as snaps):
        recorder = Recorder(scheduler, arm, {"J0": j0, "P1": p1}, event_out, snaps)
        def watch_build(builder, positions, **kwargs):
            return recorder.observe_demands(builder, positions, **kwargs)

        with require_marvel(), patch.object(ObservedMeshRelayDemandBuilder, "build", watch_build):
            while runtime.current_step < ticks:
                if time.monotonic()-start > wall_seconds:
                    raise TimeoutError(f"Incomplete audit at tick {runtime.current_step}")
                command = adapter.get_actions(obs)
                recorder.finish_event(command)
                commitments.append([asdict(x) for x in scheduler.last_event_assignments])
                for _ in range(runtime._mission_step_interval()):
                    obs, info = runtime.step(command)
                    row = physical_sample(runtime, info)
                    trace.append(row)
                    line(tick_out, dict(**row, step=runtime.current_step, control_step=recorder.last_event,
                         observed=[(int(uid), o["position"], o["heading"], o["velocity"]) for uid, o in sorted(obs.items())],
                         comm_hex=np.packbits(info["comm_topology"]).tobytes().hex(),
                         search=dict(scheduler.search_assignments), relay=relay_assignments(scheduler),
                         probe=probe_state(scheduler), commands=command,
                         tasks={task.task_id: [task.status, len(task.found_targets), task.progress]
                                for task in runtime.tasks.tasks.values()}))
                if runtime.current_step % 100 == 0:
                    print(json.dumps(dict(arm=arm, ticks=runtime.current_step,
                                          seconds=round(time.monotonic()-start, 2))), flush=True)
    write(output/"summary.json", dict(case=CASE, arm=arm, initial_sha256=initial, config_sha256=config,
          frozen_sha256=frozen_sha, trained_sha256=TRAINED_SHA, native_sha256=base_sha, marvel_sha256=marvel_sha,
          checkpoint_sha256=inference.candidate_sha256,
          simulator_base=BASELINE, driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          scenario_sha256=hashlib.sha256(SCENARIO.read_bytes()).hexdigest(),
          source_head=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
          torch_version=str(torch.__version__), radio_range=runtime.comm.comm_range,
          seconds=time.monotonic()-start, ticks=ticks, trace=trace, commitments=commitments,
          score=joint_completion_metric(trace,300.) if ticks==3000 else None))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--arm", choices=["R1", "J0", "P1", "J0_SWAP_S3"], required=True)
    cli.add_argument("--ticks", type=int, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--wall-seconds", type=float, default=3000)
    args = cli.parse_args()
    run(args.arm, args.out, args.ticks, args.wall_seconds)


if __name__ == "__main__":
    main()
