"""Replay archived public states and exercise joint and legacy physical control."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "handoff/cloud")]

from integrations.gppo.adapter import GPPOAssignment
from integrations.gppo.checkpoint import load_frozen_gppo
from integrations.gppo.config_profile import prepare_gppo_config
from integrations.gppo.joint_atomic import JointCandidatePlanner, JointSnapshot
from integrations.gppo.joint_shadow import graph_record, unpack
from integrations.gppo.joint_shadow_recorder import JointShadowRecorder
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.rule_scheduling import choose_rule_assignment
from paired_eval import verified_asset
from gppo_benchmark_domains import load_domain, validate_physical_warning_rate
from utils.policy_adapter import MARVELPolicyAdapter
from utils.scenario_config import load_and_validate_scenario
from utils.simulation_runtime import SimulationRuntime

ARCHIVE = ROOT / "artifacts/joint_shadow_20261121"
SCENARIO = ROOT / "configs/scenarios/extended_connected_gppo_cached_sensing_20261012.yaml"
PARITY_CASE = "30_static_1280021"
WITNESSES = json.loads((ARCHIVE / "witnesses.json").read_text())


def digest(data):
    return hashlib.sha256(pickle.dumps(data, protocol=5)).hexdigest()


def write(path, contents):
    path.write_text(json.dumps(contents, indent=2, sort_keys=True, allow_nan=False) + "\n")


def setup(case, mode, *, p1_model=None):
    uavs, domain, seed = case.split("_")
    count, seed = int(uavs), int(seed)
    manifest = json.loads((ROOT / "handoff/cloud/checkpoint_manifest.json").read_text())
    base, _ = verified_asset(manifest, "Frozen native GPPO")
    candidate, sha = verified_asset(manifest, "November 11 guarded candidate")
    verified_asset(manifest, "MARVEL low-level actor")
    if p1_model is not None:
        candidate, sha = p1_model
        if mode != "joint_gppo":
            raise ValueError("A replacement high-level model applies only to P1")
    config = load_domain(load_and_validate_scenario, SCENARIO, SCENARIO, domain == "dynamic", .2)
    config["robots"] = [{"id_range": [0, count - 1], "type": "explorer", "team_id": 1,
                        "config": {"fov": 120, "sensor_range": 10., "velocity": 1.,
                                   "yaw_rate": 35, "initial_positions": "random_safe"}}]
    config["scenario"]["random_seed"] = seed
    config["task_scheduler"] = {"mode": mode if mode.startswith("joint_") else "gppo",
        "checkpoint": str(base), "base_position": [0., 0.], "search_scenario_seed": seed,
        "extended_checkpoint": str(candidate), "extended_checkpoint_sha256": sha}
    _, weights = load_frozen_gppo(str(base), device="cpu")
    config = prepare_gppo_config(config, weights)
    validate_physical_warning_rate(config)
    for task in config["tasks"]:
        if task["type"] == "exploration":
            task["params"]["coverage_basis"] = "observed_free"
    config["policy"] = {"device": "cpu", "batch_inference": True}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    runtime = SimulationRuntime(config)
    observations = runtime.reset()
    adapter = MARVELPolicyAdapter(runtime)
    with torch.random.fork_rng():
        adapter.setup()
    assert adapter._using_policy, "MARVEL model required"
    if mode == "legacy_rule":
        adapter.scheduler.allocator.adapter.assign = choose_rule_assignment
    return runtime, adapter, observations


def physical_row(rt, observations, info):
    return dict(step=rt.current_step, collisions=info["collisions"],
        observed={str(uid): {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in row.items() if k in ("position", "heading", "velocity", "motion_feedback")}
                  for uid, row in observations.items()},
        safety=sorted(rt.high_level_scheduler.safety.active_uav_ids),
        no_fly=rt.high_level_scheduler.safety.no_fly_violations)


def replay_public(witness):
    case, index = witness["case"], witness["event_index"]
    with gzip.open(ARCHIVE / case / f"public_state_{index:03d}.json.gz", "rt") as source:
        snapshot = JointSnapshot.from_public(unpack(json.load(source)))
    assert snapshot.version.rsplit(":", 1)[-1] == witness["public_state_sha256"]
    plan = JointCandidatePlanner().enumerate(snapshot)
    for role in ("search", "relay"):
        edge = plan.candidates[(witness[role]["slot_id"], witness["uav_id"])]
        assert np.allclose(edge.waypoint, witness[role]["prefix"]["waypoint"])
    return dict(case=case, step=snapshot.step, legal_count=len(plan.candidates),
                public_sha256=witness["public_state_sha256"], both_witnesses_legal=True)


def directed(witness, role, *, sample_relay=False):
    rt, adapter, obs = setup(witness["case"], "joint_gppo")
    scheduler = adapter.scheduler
    target_step = round(witness["time"] / rt.dt)
    real_alloc = scheduler._allocate_event
    boundary = {}

    def allocate(stale):
        if rt.current_step < target_step:
            return GPPOTaskScheduler._allocate_event(scheduler, stale)
        # Offline corroboration, outside the planner. Deterministic reset/replay
        # supplies the same physics without deserializing a runtime.
        snapshot = scheduler._snapshot(stale)
        boundary["snapshot"] = snapshot.version
        boundary["physical_sha256"] = digest([asdict(robot) for robot in rt.robots])
        boundary["task_sha256"] = digest(rt.tasks.tasks)
        boundary["hidden_layout_sha256"] = digest(rt.target_detector.hidden_targets)
        boundary["rng_sha256"] = digest((random.getstate(), np.random.get_state(),
                                         bytes(torch.random.get_rng_state().tolist())))
        # Reconstruct the exact archive digest using its original case label.
        public = unpack(json.loads(snapshot.public_bytes))
        public["scenario"] = witness["case"]
        boundary["archive_public_sha256"] = JointSnapshot.from_public(public).version.rsplit(":", 1)[-1]
        assert boundary["archive_public_sha256"] == witness["public_state_sha256"]
        return real_alloc(stale)

    def choose(graph):
        ti = graph.subtask_id_to_index[witness[role]["slot_id"]]
        ui = graph.uav_id_to_index[witness["uav_id"]]
        assert not graph.action_mask[ti, ui]
        boundary["joint_graph"] = graph_record(graph)
        boundary["rule_choice"] = asdict(choose_rule_assignment(graph))
        model_choice = scheduler.allocator.adapter.assign(graph)
        assert model_choice is not None and not graph.action_mask[model_choice.task_index, model_choice.uav_index]
        boundary["gppo_choice"] = asdict(model_choice)
        if sample_relay:
            model = scheduler.allocator.adapter
            with torch.no_grad():
                logits = model.model(**model.last_batch)["flat_logits"][0]
            probs = torch.softmax(logits, dim=0)
            role_ids = [int(t.task_type) for t in graph.subtasks]
            relay_mass = sum(float(probs[i]) for i in range(len(probs))
                             if role_ids[i // graph.num_uavs] == 2)
            assert relay_mass > 0
            # Pick a reproducible diagnostic seed that draws Relay from the
            # frozen model's actual distribution without changing any logits.
            generator = torch.Generator()
            for seed in range(4096):
                draw = int(torch.multinomial(probs, 1, generator=generator.manual_seed(seed)))
                if role_ids[draw // graph.num_uavs] == 2:
                    break
            else:
                raise RuntimeError("No Relay diagnostic draw within the bounded seed scan")
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                sampled = model.assign(graph, deterministic=False)
            assert sampled.flat_action == draw
            boundary["sampled_gppo_choice"] = asdict(sampled)
            boundary["diagnostic_policy_seed"] = seed
            boundary["relay_probability_mass"] = relay_mass
            return sampled
        return GPPOAssignment(graph.flatten_action(ti, ui), ti, ui,
                              witness[role]["slot_id"], witness["uav_id"], 0.)

    scheduler._allocate_event = allocate
    scheduler._choose_joint = choose
    while rt.current_step < target_step:
        commands = adapter.get_actions(obs)
        for _ in range(rt._mission_step_interval()):
            obs, info = rt.step(commands)
            assert not info["collisions"]
    commands = adapter.get_actions(obs)
    receipt = scheduler.last_joint_receipt
    uid = receipt.assignment.uav_id if sample_relay else witness["uav_id"]
    index = next(i for i, r in enumerate(rt.robots) if r.robot_id == uid)
    before = rt.robots[index].position.copy()
    if sample_relay:
        assert receipt.assignment.slot_id == boundary["sampled_gppo_choice"]["subtask_id"]
        assert receipt.assignment.task_type == 2
    else:
        assert receipt.assignment.uav_id == uid and receipt.assignment.slot_id == witness[role]["slot_id"]
    assert scheduler.assignments[uid][0] == ("target_search" if role == "search" else "relay")
    if not sample_relay:
        np.testing.assert_allclose(commands[index][0], witness[role]["prefix"]["waypoint"], atol=1e-9, rtol=0)
    obs, info = rt.step(commands)
    assert not info["collisions"] and scheduler.safety.no_fly_violations == 0
    after = rt.robots[index].position.copy()
    if sample_relay:
        assert np.linalg.norm(after - before) > 1e-9
    else:
        np.testing.assert_allclose(after, witness[role]["prefix"]["next_position"], atol=1e-9, rtol=0)
    trace = [physical_row(rt, obs, info)]
    for _ in range(2):
        obs, info = rt.step(commands)
        assert not info["collisions"] and scheduler.safety.no_fly_violations == 0
        trace.append(physical_row(rt, obs, info))
    boundary.update(case=witness["case"], role=role, uav_id=uid, receipt=asdict(receipt),
                    before=before.tolist(), after=after.tolist(), command=commands[index][0].tolist(),
                    physical=trace)
    return boundary


def smoke(case, mode, steps, shadow=False):
    rt, adapter, obs = setup(case, mode)
    scheduler = adapter.scheduler
    decisions, traces = [], []
    if mode.startswith("joint_"):
        choose = scheduler._choose_joint

        def observe(graph):
            decision = choose(graph)
            decisions.append(dict(graph=graph_record(graph), chosen=asdict(decision)))
            return decision

        scheduler._choose_joint = observe
    recorder = JointShadowRecorder(scheduler, scenario=case) if shadow else None
    receipts = []
    for _ in range(steps):
        commands = adapter.get_actions(obs)
        if mode.startswith("joint_"):
            receipts.append(asdict(scheduler.last_joint_receipt))
            scheduler._check_ownership()
        for _ in range(rt._mission_step_interval()):
            obs, info = rt.step(commands)
            traces.append(physical_row(rt, obs, info))
            assert not info["collisions"] and scheduler.safety.no_fly_violations == 0
    if recorder:
        recorder.close()
    return dict(case=case, mode=mode, shadow=shadow, traces=traces, receipts=receipts, decisions=decisions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", choices=("replay", "directed", "smoke", "legacy-parity", "sampled", "all"), default="all")
    parser.add_argument("--case", choices=[w["case"] for w in WITNESSES])
    parser.add_argument("--steps", default=3, choices=(1, 2, 3), type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    write(args.out / "runner_identity.json", dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        torch=torch.__version__, numpy=np.__version__, python=sys.version,
        suite=args.suite, case=args.case, steps=args.steps))
    # Legacy adapter logs several fallback paths rather than raising them.
    class FailOnFallback(logging.Handler):
        def emit(self, record):
            if any(reason in record.getMessage() for reason in (
                "Policy inference error", "update_graph failed", "update_planning_state failed",
                "Action selection failed", "will use default_actions",
            )):
                raise RuntimeError(record.getMessage())

    logger = logging.getLogger("utils.policy_adapter")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(FailOnFallback())
    for w in WITNESSES:
        case = w["case"]
        if args.case and args.case != case:
            continue
        if args.suite in ("all", "replay"):
            write(args.out / f"{case}_replay.json", replay_public(w))
            print(case, "archived candidates replayed", flush=True)
        if args.suite in ("all", "directed"):
            results = [directed(w, role) for role in ("search", "relay")]
            for key in ("snapshot", "physical_sha256", "task_sha256", "hidden_layout_sha256",
                        "rng_sha256", "archive_public_sha256", "joint_graph", "before"):
                assert results[0][key] == results[1][key], key
            assert not np.allclose(results[0]["after"], results[1]["after"], atol=1e-9, rtol=0)
            write(args.out / f"{case}_directed.json", results)
            print(case, "same-state Search and Relay moved differently", flush=True)
        if args.suite in ("all", "sampled") and case == PARITY_CASE:
            write(args.out / f"{case}_sampled_relay.json", directed(w, "relay", sample_relay=True))
            print(case, "sampled frozen GPPO Relay physically moved", flush=True)
        if args.suite in ("all", "smoke"):
            for mode in ("legacy_rule", "legacy_gppo", "joint_rule", "joint_gppo"):
                result = smoke(case, mode, args.steps)
                write(args.out / f"{case}_{mode}.json", result)
                print(case, mode, len(result["traces"]), "physical ticks", flush=True)
        if args.suite in ("all", "legacy-parity") and case == PARITY_CASE:
            plain = smoke(case, "legacy_gppo", args.steps)
            observed = smoke(case, "legacy_gppo", args.steps, shadow=True)
            assert plain["traces"] == observed["traces"]
            with gzip.open(ARCHIVE / case / "gppo/physical_observations.json.gz", "rt") as f:
                archived = json.load(f)
            assert archived[:len(plain["traces"])] == [
                {"step": r["step"], "observations": r["observed"]} for r in plain["traces"]]
            write(args.out / f"{case}_legacy_parity.json",
                  {"ticks": len(plain["traces"]), "phase1_and_shadow_off_on_exact": True})
            print(case, "legacy phase 1 and shadow on/off parity", flush=True)


if __name__ == "__main__":
    main()
