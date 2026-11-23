"""Collect explicit behavior probabilities or deploy frozen value policies via MARVEL."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from pathlib import Path
import time

import numpy as np

from run_joint_autoresearch import ExperimentRecorder, action_at, run
from verify_joint_atomic import write
from integrations.gppo.event_rl import EventPolicy


def behavior_distribution(rows, greedy_flat, *, epsilon, mixed_epsilon, top):
    candidates = {role: [i for i, r in enumerate(rows) if r["role"] == role] for role in (1, 2)}
    groups = [group for group in candidates.values() if group]
    epsilon = mixed_epsilon if len(groups) == 2 else epsilon
    probabilities = np.zeros(len(rows))
    greedy = next(i for i, row in enumerate(rows) if row["flat"] == greedy_flat)
    probabilities[greedy] = 1. - epsilon
    for group in groups:
        # Restrict exploration by public rank within each role. Alternate fast
        # legal paths with high frozen-actor scores when there are ties.
        order = sorted(group, key=lambda i: (-rows[i]["logit"], rows[i]["total"], i))
        options = order[:top]
        quickest = min(group, key=lambda i: (rows[i]["total"], i))
        if quickest not in options:
            options[-1] = quickest
        probabilities[options] += epsilon / (len(options) * len(groups))
    return probabilities


class RLRecorder(ExperimentRecorder):
    def __init__(self, *args, agent=None, explore=None, generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = agent
        self.explore = explore
        self.generator = generator
        self.forked = False

    def observe_choice(self, graph):
        normal = super().observe_choice(graph)
        record = self.event
        rows = record["rows"]
        greedy = normal.flat_action
        if self.agent is not None:
            start = time.perf_counter()
            greedy, record["learned_scores"] = self.agent.choose(record)
            record["learned_inference_seconds"] = time.perf_counter() - start
        if self.explore is not None and (not self.explore["first_mixed_only"] or
                                         not self.forked and len({r["role"] for r in rows}) == 2):
            probabilities = behavior_distribution(rows, greedy, epsilon=self.explore["epsilon"],
                                                  mixed_epsilon=self.explore["mixed_epsilon"],
                                                  top=self.explore["top"])
            index = int(self.generator.choice(len(rows), p=probabilities))
            self.forked |= len({r["role"] for r in rows}) == 2
        else:
            index = next(i for i, row in enumerate(rows) if row["flat"] == greedy)
            probabilities = np.zeros(len(rows))
            probabilities[index] = 1.
        selected = action_at(graph, rows[index]["flat"], normal.value)
        record.update(selected=[selected.subtask_id, selected.uav_id],
                      behavior_distribution=probabilities.tolist(),
                      executed_action_probability=float(probabilities[index]))
        self.intervened |= selected.flat_action != normal.flat_action
        return selected


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--case", required=True)
    cli.add_argument("--policy", choices=("ddqn", "sac", "explore", "J0"), required=True)
    cli.add_argument("--checkpoint", type=Path)
    cli.add_argument("--sha256")
    cli.add_argument("--behavior-seed", type=int, default=8739)
    cli.add_argument("--epsilon", type=float, default=.18)
    cli.add_argument("--mixed-epsilon", type=float, default=.85)
    cli.add_argument("--top", type=int, default=6)
    cli.add_argument("--first-mixed-only", action="store_true")
    cli.add_argument("--collect-exploring", action="store_true")
    cli.add_argument("--wall", type=float, default=4500.)
    cli.add_argument("--ticks", type=int, default=3000)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.policy in ("ddqn", "sac"):
        if not args.checkpoint or not args.sha256:
            cli.error("A trained policy needs a checkpoint and verified SHA-256")
        agent = EventPolicy.load(args.checkpoint, args.sha256)
        if agent.settings.algo != args.policy:
            cli.error("Requested policy and checkpoint algorithm differ")
    else:
        agent = None
        if args.checkpoint or args.sha256:
            cli.error("An exploration or J0 run has no learning checkpoint")
    if min(args.epsilon, args.mixed_epsilon) < 0 or max(args.epsilon, args.mixed_epsilon) > 1 or args.top < 1:
        cli.error("Invalid behavior mixture")
    explore = (dict(epsilon=args.epsilon, mixed_epsilon=args.mixed_epsilon, top=args.top,
                    first_mixed_only=args.first_mixed_only)
               if args.policy == "explore" or args.collect_exploring else None)
    factory = partial(RLRecorder, agent=agent, explore=explore,
                      generator=np.random.default_rng(args.behavior_seed))
    args.step = args.pair = None
    try:
        run(args, recorder_factory=factory)
    finally:
        if (args.out / "summary.json").is_file():
            write(args.out / "learning_policy.json",
                  dict(policy_id=args.policy + ("-behavior" if explore and agent else ""), checkpoint_sha256=args.sha256,
                       exploration=explore, behavior_seed=args.behavior_seed if explore else None,
                       driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       ticks=args.ticks, wall_limit_seconds=args.wall))


if __name__ == "__main__":
    main()
