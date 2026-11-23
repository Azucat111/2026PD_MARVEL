"""Verify historical and freshly collected observations, then build training replay."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from joint_policy_causal_trace import FROZEN_SHA, tensor_digest
from verify_joint_atomic import ROOT, write
from integrations.gppo.event_rl_replay import (
    HORIZON_TICKS, load_replay, read_run, save_replay, transitions_from_events,
)
from integrations.gppo.joint_atomic import JointCandidatePlanner, JointSnapshot
from integrations.gppo.joint_shadow import unpack

P2 = ROOT / "artifacts/joint_autoresearch_20261122/runs"
CAUSAL = ROOT / "artifacts/joint_causal_20261122/runs"
PPO = ROOT / "artifacts/joint_ppo_20261122/training"


def replan(raw, version):
    state = unpack(json.loads(raw))
    snap = JointSnapshot.from_public(state, int(version.split(":")[-2]))
    if snap.version != version or snap.public_bytes != raw:
        raise ValueError("A serialized public snapshot cannot be restored exactly")
    return state, JointCandidatePlanner().enumerate(snap)


def restored_actions(plan, logits):
    graph = plan.graph
    result = []
    for flat, logit in zip(np.flatnonzero(~graph.action_mask.flatten()), logits, strict=True):
        ti, ui = divmod(int(flat), graph.num_uavs)
        slot, uid = graph.action_ids(int(flat))
        candidate = plan.candidates[(slot, uid)]
        result.append(dict(flat=int(flat), slot=int(slot), uav=int(uid), role=int(candidate.slot.task_type),
                           anchor=candidate.slot.position, waypoint=candidate.waypoint,
                           total=float(graph.edge_features[ti, ui, 2]), logit=float(logit)))
    return result


def import_live(directory):
    summary, events = read_run(directory)
    if "status" not in summary:
        if summary.get("ticks") != HORIZON_TICKS or not summary.get("score"):
            raise ValueError("Legacy episode does not prove a complete clock and score")
        summary["status"] = "complete"
    if any(row.get("selected") and "rows" not in row for row in events):
        with ZipFile(directory / "public.zip") as archive:
            for row in events:
                if row.get("selected"):
                    state, plan = replan(archive.read(f"{row['step']:04d}.json"), row["snapshot_version"])
                    if plan.graph.action_mask.tolist() != row["mask"]:
                        raise ValueError("Archived and replanned masks differ")
                    row["rows"] = restored_actions(plan, row["actors"]["J0"]["logits"])
                    row["range_m"] = state["relay_builder"]["comm_range"]
                    row["time_seconds"] = row["step"] * state["dt"]
    policy_path = directory / "learning_policy.json"
    if policy_path.is_file():
        policy = json.loads(policy_path.read_text())
        policy_id = policy["policy_id"]
        if policy.get("checkpoint_sha256"):
            policy_id += ":" + policy["checkpoint_sha256"]
    else:
        policy_id = summary.get("policy", "J0")
        if summary.get("checkpoint_sha256"):
            policy_id += ":" + summary["checkpoint_sha256"]
    return summary, events, policy_id


def import_ppo(directory):
    # Only manifest-verified, checked-in assets use weights_only deserialization.
    data = torch.load(directory / "rollout.pt", map_location="cpu", weights_only=True)
    summary = data["summary"]
    if summary["behavior_checkpoint_sha256"] != FROZEN_SHA or not summary["episode_terminated"]:
        raise ValueError("PPO archive does not have a matching frozen actor and full episode")
    trace = json.loads((directory / "trace.json").read_text())
    commitments = [[] for _ in range(HORIZON_TICKS // 10)]
    rows = []
    with ZipFile(directory / "public_observations.zip") as archive:
        for event in data["events"]:
            step = event["step"]
            state, plan = replan(archive.read(event["public_snapshot"]), event["snapshot_version"])
            graph = plan.graph
            mask = event["batch"]["action_mask"].reshape(graph.action_mask.shape).numpy()
            if (not np.array_equal(graph.action_mask, mask)
                    or event["task_ids"] != [t.subtask_id for t in graph.subtasks]
                    or event["uav_ids"] != [u.uav_id for u in graph.uav_states]):
                raise ValueError("PPO archive layout differs from the replanned graph")
            # Tempered log-probabilities recover logits up to a state-wide
            # additive constant, which cancels from every chosen feature.
            logp = event["old_log_probs"].double().numpy()
            chosen = event["selected_legal_index"]
            probabilities = np.exp(logp)
            if (not np.isclose(probabilities.sum(), 1., atol=1e-6, rtol=0.)
                    or not np.isclose(logp[chosen], event["old_logp"], atol=1e-6, rtol=0.)):
                raise ValueError("Archived Phase 3 behavior law disagrees with the sampled action")
            probabilities /= probabilities.sum()
            receipt = event["receipt"]["assignment"]
            if receipt["flat_action"] != np.flatnonzero(~mask.flatten())[chosen]:
                raise ValueError("Archived Phase 3 action does not match the physical receipt")
            commitments[step // 10] = [receipt]
            selected = [receipt["slot_id"], receipt["uav_id"]]
            positions = state["stale_positions"]
            rows.append(dict(step=step, snapshot_version=event["snapshot_version"], stale_xy=positions,
                 search_before=state["controls"]["search_assignments"],
                 relay_before=state["controls"]["relay_assignments"],
                 unavailable=[r["robot_id"] for r in state["robots"] if not r["available"]],
                 task_ids=event["task_ids"], uav_ids=event["uav_ids"], roles=event["roles"], mask=mask.tolist(),
                 rows=restored_actions(plan, logp * summary["temperature"]),
                 range_m=state["relay_builder"]["comm_range"], time_seconds=step*state["dt"],
                 selected=selected, receipt=receipt, input_sha256=tensor_digest(event["batch"]),
                 behavior_distribution=probabilities.tolist(),
                 executed_action_probability=float(probabilities[chosen])))
    return dict(summary, trace=trace, commitments=commitments), rows, "Phase3-J0-T3"


def fingerprint(summary):
    applied = [[(r["slot_id"], r["uav_id"], r["task_type"]) for r in batch] for batch in summary["commitments"]]
    blob = json.dumps([summary["case"], applied, summary["trace"]], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def build(run_paths, historical, output, audit_path, parent_replay=None):
    roots = []
    if historical:
        roots += [(p, import_live) for p in sorted(P2.iterdir()) if (p / "public.zip").is_file()]
        roots.append((CAUSAL / "J0", import_live))
        roots += [(p, import_ppo) for p in sorted(PPO.iterdir()) if (p / "rollout.pt").is_file()]
    roots += [(p, import_live) for p in run_paths]
    audit = dict(sources=[], coverage={}, groups={}, eligible_bootstrap_modes=["complete"])
    counts = Counter()
    episodes = []
    seen = {}
    if parent_replay is not None:
        episodes, audit = load_replay(parent_replay)
        audit["parent_replay_sha256"] = hashlib.sha256(parent_replay.read_bytes()).hexdigest()
        counts.update({k:v for k,v in audit["coverage"].items() if k != "unique_seed_groups"})
        seen.update((row["source_fingerprint"], row["directory"]) for row in audit["sources"] if row["used"])
    for directory, load in roots:
        summary, events, policy = load(directory)
        case = summary["case"]
        rows, prefix = transitions_from_events(summary, events, policy_id=policy, group=case)
        key = fingerprint(summary)
        status = summary["status"]
        source = dict(directory=directory.name, case=case, policy_id=policy, status=status,
                      transitions=len(rows), prefix_reward=prefix, score=summary.get("score"),
                      missing_behavior_probabilities=sum(r.behavior_probability is None for r in rows),
                      unavailable_rng_state=True, source_fingerprint=key)
        audit["sources"].append(source)
        if status != "complete" or key in seen or not rows:
            source.update(used=False, reason=("physical_duplicate" if key in seen else
                                              "no_policy_decisions" if not rows else "window_incomplete"),
                          duplicate_of=seen.get(key))
            continue
        source.update(used=True)
        seen[key] = directory.name
        episodes.append((directory.name, rows))
        counts["episodes"] += 1
        score = summary["score"]
        counts["successful_episodes" if score["joint_success"] else "failed_episodes"] += 1
        counts["unsafe_episodes"] += not score["whole_window_safe"]
        domain = case.rsplit("_", 1)[0]
        group = audit["groups"].setdefault(case, dict(episodes=0, transitions=0, domain=domain))
        group["episodes"] += 1
        group["transitions"] += len(rows)
        for row in rows:
            roles = {a[2] for a in row.state.candidates}
            role_set = "mixed" if len(roles) == 2 else "search_only" if roles == {1} else "relay_only"
            role = "search" if row.executed[2] == 1 else "relay"
            counts["transitions"] += 1
            counts["selected_"+role] += 1
            counts[role_set] += 1
            counts[f"{role_set}_selected_{role}"] += 1
            counts[domain] += 1
            counts["with_behavior_probability"] += row.behavior_probability is not None
    audit["coverage"] = dict(counts, unique_seed_groups=len(audit["groups"]))
    save_replay(output, episodes, audit)
    write(audit_path, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", type=Path, default=[])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--historical", action="store_true")
    source.add_argument("--parent-replay", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit = build(args.runs, args.historical, args.out, args.audit, parent_replay=args.parent_replay)
    print(json.dumps(audit["coverage"], sort_keys=True))


if __name__ == "__main__":
    main()
