"""Train and save an independently initialized event-level RL checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.gppo.event_rl import Settings, train
from integrations.gppo.event_rl_replay import load_replay


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", type=Path, required=True)
    cli.add_argument("--algo", choices=("ddqn", "sac"), required=True)
    cli.add_argument("--seed", type=int, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--updates", type=int, default=600)
    cli.add_argument("--n-step", type=int, default=1)
    cli.add_argument("--lr", type=float, default=.0007)
    cli.add_argument("--sync-every", type=int, default=50)
    cli.add_argument("--initial-alpha", type=float, default=.035)
    cli.add_argument("--cql-weight", type=float, default=0.)
    args = cli.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    episodes, audit = load_replay(args.replay)
    if not episodes:
        cli.error("Replay does not contain an eligible episode")
    size = len(episodes[0][1][0].state.features[0])
    settings = Settings(args.algo, size, args.seed, lr=args.lr, n_step=args.n_step,
                        sync_every=args.sync_every, initial_alpha=args.initial_alpha, cql_weight=args.cql_weight)
    source = hashlib.sha256(args.replay.read_bytes()).hexdigest()
    print(json.dumps(dict(algo=args.algo, seed=args.seed, coverage=audit["coverage"],
                          replay_sha256=source, planned_updates=args.updates), sort_keys=True), flush=True)
    results = train(episodes, audit, settings, args.updates, source,
                    lambda row: print(json.dumps(row, sort_keys=True), flush=True))
    with args.out.open("xb") as handle:
        torch.save(results, handle)
    print(json.dumps(dict(checkpoint=args.out.name, sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                          updates=results["updates"], replay_sha256=source)), flush=True)


if __name__ == "__main__":
    main()
