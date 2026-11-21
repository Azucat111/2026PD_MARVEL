"""Audit stored decisions and compare two checkpoints on identical public inputs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

import torch

from joint_event_ppo import (action_distribution, configure_torch, verify_replay,
                             verify_rollout_files)
from verify_joint_atomic import setup, write
from integrations.gppo.joint_atomic import JointCandidatePlanner, JointSnapshot


def audit_masks(path, events):
    """Independently rebuild admission from each real public snapshot."""
    steps = []
    planner = JointCandidatePlanner()
    with ZipFile(path / "public_observations.zip") as archive:
        for row in events:
            raw = archive.read(row["public_snapshot"])
            version = row["snapshot_version"]
            if hashlib.sha256(raw).hexdigest() != version.rsplit(":", 1)[-1]:
                raise ValueError("Tampered public archive")
            snapshot = JointSnapshot(row["step"], int(version.split(":")[1]), version, raw)
            graph = planner.enumerate(snapshot).graph
            if (row["task_ids"] != [int(t.subtask_id) for t in graph.subtasks]
                    or row["uav_ids"] != [int(u.uav_id) for u in graph.uav_states]
                    or row["roles"] != [int(t.task_type) for t in graph.subtasks]):
                raise ValueError(f"Rebuilt stable IDs or roles differ at step {row['step']}")
            torch.testing.assert_close(torch.as_tensor(graph.action_mask), row["batch"]["action_mask"][0])
            steps.append(row["step"])
    return steps


@torch.no_grad()
def diagnose(behavior, candidate, temperature, events):
    rows = []
    for e in events:
        verify_replay(behavior, e, temperature)
        old, old_value, legal = action_distribution(behavior, e["batch"], temperature)
        new, new_value, _ = action_distribution(candidate, e["batch"], temperature)
        indices = legal.nonzero().flatten().tolist()
        n = len(e["uav_ids"])
        roles = [e["roles"][i//n] for i in indices]

        def pick(dist):
            flat = indices[int(dist.logits.argmax())]
            ti, ui = divmod(flat, n)
            return {"slot": e["task_ids"][ti], "uav": e["uav_ids"][ui], "role": e["roles"][ti]}

        relay = torch.tensor([r == 2 for r in roles])
        rows.append(dict(step=e["step"], duration_seconds=e["duration_seconds"],
                         role_sample=e["chosen_role"], advantage=e["advantage"], target=e["return_target"],
                         task_count=len(e["task_ids"]), legal_count=len(indices), mixed=len(set(roles)) > 1,
                         before=pick(old), after=pick(new), value_before=float(old_value), value_after=float(new_value),
                         relay_mass_before=float(old.probs[relay].sum()), relay_mass_after=float(new.probs[relay].sum())))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout", type=Path, nargs="+", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    configure_torch()
    results = []
    for path in args.rollout:
        episode = torch.load(path/"rollout.pt", weights_only=True)
        verify_rollout_files(path, episode)
        case = episode["summary"]["case"]
        _, parent, _ = setup(case, "joint_gppo")
        _, child, _ = setup(case, "joint_gppo", p1_model=(args.checkpoint, args.sha256))
        behavior = parent.scheduler.allocator.adapter
        if behavior.candidate_sha256 != episode["summary"]["behavior_checkpoint_sha256"]:
            raise ValueError("Diagnostic expects the frozen Phase 2 behavior weights")
        states = diagnose(behavior.model, child.scheduler.allocator.adapter.model,
                          episode["summary"]["temperature"], episode["events"])
        results.append(dict(case=case, independent_mask_rebuild_steps=audit_masks(path, episode["events"]),
                            greedy_actions_changed=sum(r["before"]!=r["after"] for r in states),
                            greedy_roles_changed=sum(r["before"]["role"]!=r["after"]["role"] for r in states),
                            states=states))
    write(args.out, dict(candidate_sha256=args.sha256, comparison="same stored public inputs, no policy rollout",
                         cases=results))


if __name__ == "__main__":
    main()
