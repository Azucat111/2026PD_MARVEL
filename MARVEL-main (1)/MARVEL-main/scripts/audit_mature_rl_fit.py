"""Audit fitted decisions and observed continuation returns on verified replay."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.gppo.event_rl import CandidateNet, EventPolicy, TwinCritic, masked_probabilities, padded
from integrations.gppo.event_rl_replay import load_replay


@torch.no_grad()
def audit(path, episodes):
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    policy = EventPolicy.load(path, sha)
    data = torch.load(path, weights_only=True, map_location="cpu")
    settings = policy.settings
    net = CandidateNet if settings.algo == "ddqn" else TwinCritic
    critic = net(settings.features, settings.hidden).eval()
    critic.load_state_dict(data["critic"], strict=True)
    action_support = defaultdict(set)
    for _, rows in episodes:
        for row in rows:
            action_support[row.group, row.state.snapshot].add(row.executed)

    records = []
    for name, rows in episodes:
        future = 0.
        returns = []
        for row in reversed(rows):
            future = row.reward + settings.gamma_per_second ** row.dt * future
            returns.append(future)
        for row, observed in zip(rows,reversed(returns),strict=True):
            state = row.state
            values, legal = padded([state])
            q = critic(values)[0].min(-1).values
            logits = policy.net(values).flatten()
            chosen = int(logits.argmax())
            selected = state.candidates[chosen]
            roles = {a[2] for a in state.candidates}
            support = action_support[row.group, state.snapshot]
            relay = torch.tensor([a[2] == 2 for a in state.candidates], dtype=torch.bool)
            probability, _ = masked_probabilities(logits[None,:],legal) if settings.algo == "sac" else (None,None)
            records.append(dict(episode=name, case=row.group, step=state.step, observed_behavior_return=observed,
                                logged_q=float(q[row.action]), greedy_q=float(q[chosen]), logged=row.executed,
                                greedy=selected, supported_in_exact_replay_state=selected in support,
                                observed_distinct_actions=len(support), mixed=len(roles)==2,
                                relay_probability=None if probability is None else float(probability[0,relay].sum())))
    counts = Counter()
    for r in records:
        counts["total"] += 1
        counts["greedy_matches_logged"] += r["greedy"] == r["logged"]
        counts["greedy_without_same_state_execution"] += not r["supported_in_exact_replay_state"]
        counts["logged_positive_q"] += r["logged_q"] > 0
        counts["greedy_positive_q"] += r["greedy_q"] > 0
        if r["mixed"]:
            counts["mixed"] += 1
            counts["mixed_greedy_relay"] += r["greedy"][2] == 2
    return dict(algorithm=settings.algo, checkpoint_sha256=sha, train_replay_sha256=data["replay_sha256"],
                counts=dict(counts),
                mean_greedy_minus_logged_q=statistics.mean(r["greedy_q"]-r["logged_q"] for r in records),
                logged_behavior_return_mae=statistics.mean(abs(r["logged_q"]-r["observed_behavior_return"]) for r in records),
                rows=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    episodes, _ = load_replay(args.replay)
    data = audit(args.model,episodes)
    with args.out.open("x") as handle:
        json.dump(data,handle,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in data.items() if k!="rows"},sort_keys=True))


if __name__ == "__main__":
    main()
