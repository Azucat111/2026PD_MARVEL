"""Compare fitted and frozen choices on recorded legal graphs without stepping physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fit_joint_public_preference import read_events
from integrations.gppo.public_pair_preference import PublicPairPreference


def replay(policy, trajectories, *, include_unchanged=False):
    changes = []
    for trajectory in trajectories:
        for step, event in read_events(trajectory).items():
            if not event.get("selected"):
                continue
            flat, record = policy.choose(event)
            choice = next(row for row in event["rows"] if row["flat"] == flat)
            pair = [choice["slot"], choice["uav"]]
            if include_unchanged or pair != event["normal_choice"]:
                result = dict(trajectory=trajectory.name, step=step, j0=event["normal_choice"], ranked=pair,
                              behavior=event["selected"], raw_role=choice["role"],
                              rerank_score=record["logits"][record["legal_indices"].index(flat)])
                if include_unchanged:
                    prior = record["legal_indices"].index(record["prior_index"])
                    result.update(j0_action_support=record["supports"][prior],
                                  maximum_legal_support=max(record["supports"]),
                                  minimum_pair_support=policy.parameters["support"])
                changes.append(result)
    return changes


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--sha256", required=True)
    cli.add_argument("--trajectory", type=Path, nargs="+", required=True)
    cli.add_argument("--include-unchanged", action="store_true", help="Include support for every recorded decision")
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    policy = PublicPairPreference.load(args.checkpoint,args.sha256)
    with args.out.open("x") as handle:
        decisions = replay(policy,args.trajectory,include_unchanged=args.include_unchanged)
        handle.write(json.dumps(decisions,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
