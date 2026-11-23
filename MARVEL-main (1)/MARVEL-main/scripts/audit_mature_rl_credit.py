"""Compare fitted action ranks with actual shared-prefix physical returns."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def expand_forks(pairs, benchmark, snapshots):
    result = list(pairs)
    for fork in benchmark["first_mixed_forks"]:
        if not fork["changed"] or fork["control_cost"] is None or fork["intervention_cost"] is None:
            continue
        case, step = fork["case"], fork["step"]
        result.append(dict(case=case, step=step, old_pair=fork["frozen"], new_pair=fork["explored"],
                           control_seconds=fork["control_cost"], treatment_seconds=fork["intervention_cost"],
                           intervention=f"collection-{case}", snapshot_version=snapshots[f"j0-{case}", step]))
    return result


def audit(replay, benchmark, labels, fits):
    snapshots = {(episode["name"], row["state"]["step"]): row["state"]["snapshot"]
                 for episode in replay["episodes"] for row in episode["transitions"]}
    comparisons = expand_forks(labels, benchmark, snapshots)
    rows = []
    for fit in fits:
        values = {}
        for row in fit["rows"]:
            snap = snapshots[row["episode"], row["step"]]
            key = row["case"], row["step"], snap, tuple(row["logged"][:2])
            previous = values.setdefault(key, row["logged_q"])
            if abs(previous - row["logged_q"]) > 1e-5:
                raise ValueError("Inconsistent Q on the same archived state and action")
        for sample in comparisons:
            prefix = sample["case"], sample["step"], sample["snapshot_version"]
            old,new = (values.get((*prefix,tuple(sample[k]))) for k in ("old_pair","new_pair"))
            measured = sample["control_seconds"] - sample["treatment_seconds"]
            score = None if old is None or new is None else (new-old) * replay["reward_scale_seconds"]
            rows.append(dict(algorithm=fit["algorithm"], case=sample["case"], step=sample["step"],
                             intervention=sample["intervention"], old_pair=sample["old_pair"],
                             new_pair=sample["new_pair"], measured_seconds_difference=measured,
                             paired_critic_difference_seconds_equivalent=score,
                             sign_matches=None if score is None else score*measured>0))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--paired-labels", type=Path, required=True)
    parser.add_argument("--fits", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.replay,"rt") as handle:
        replay=json.load(handle)
    data = audit(replay, json.loads(args.benchmark.read_text()),
                 json.loads(args.paired_labels.read_text())["physical_training_pairs"],
                 [json.loads(p.read_text()) for p in args.fits])
    with args.out.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
