"""Bounded CLI for physical rollout and Joint PPO update."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from joint_event_ppo import collect_episode, configure_torch, update_rollouts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--sha256")
    command = p.add_subparsers(dest="command", required=True)
    collect = command.add_parser("collect")
    collect.add_argument("--case", required=True)
    collect.add_argument("--out", required=True, type=Path)
    collect.add_argument("--policy-seed", required=True, type=int)
    collect.add_argument("--temperature", type=float, default=3.)
    collect.add_argument("--wall-seconds", type=float, default=3000)
    collect.add_argument("--segment-ticks", type=int)
    update = command.add_parser("update")
    update.add_argument("--rollout", required=True, nargs="+", type=Path)
    update.add_argument("--out", required=True, type=Path)
    update.add_argument("--lr", default=2e-5, type=float)
    update.add_argument("--epochs", default=2, type=int)
    update.add_argument("--allow-segment", action="store_true")
    args = p.parse_args()
    if bool(args.checkpoint) != bool(args.sha256):
        p.error("Override checkpoint and exact SHA-256 together.")
    model = (args.checkpoint, args.sha256) if args.checkpoint else None
    configure_torch()
    if args.command == "collect":
        if not math.isfinite(args.wall_seconds) or args.wall_seconds <= 0:
            p.error("Provide a finite positive wall limit.")
        if args.segment_ticks is not None and not 1 <= args.segment_ticks < 3000:
            p.error("Segment threshold must be within the physical window.")
        summary = collect_episode(args.case, args.out, model=model, policy_seed=args.policy_seed,
                                  temperature=args.temperature, wall_seconds=args.wall_seconds,
                                  segment_ticks=args.segment_ticks)
        return 0 if summary["eligible"] else 1
    summary = update_rollouts(args.rollout, args.out, model=model, lr=args.lr, epochs=args.epochs,
                              allow_segment=args.allow_segment)
    return 0 if summary["changed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
