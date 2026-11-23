"""Post-freeze public feature support audit against actual replay actions."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from build_mature_rl_replay import import_live
from verify_joint_atomic import write
from integrations.gppo.event_rl_replay import Decision, load_replay


def calibration(episodes):
    rows = [r for _, episode in episodes for r in episode]
    values = np.asarray([r.state.features[r.action] for r in rows])
    groups = np.asarray([r.group for r in rows])
    roles = np.asarray([r.executed[2] for r in rows])
    scale = np.maximum(values.std(axis=0), .1)

    def distance(x, mask):
        return float(np.sqrt(np.square((values[mask] - x) / scale).mean(axis=1).min()))

    thresholds = {}
    for role in sorted(set(roles)):
        leave_group = [distance(v, (groups != group) & (roles == role))
                       for v, group, r in zip(values, groups, roles, strict=True)
                       if r == role and ((groups != group) & (roles == role)).any()]
        if not leave_group:
            raise ValueError("A role has no independent seed-group support calibration")
        thresholds[int(role)] = float(np.quantile(leave_group, .95))
    return thresholds, lambda x, role: distance(x, roles == role)


def audit(paths, episodes):
    thresholds, distance = calibration(episodes)
    groups = {r.group for _, episode in episodes for r in episode}
    records = []
    counts = Counter()
    for path in paths:
        summary, events, _ = import_live(path)
        if summary["case"] in groups:
            raise ValueError("Support audit is intended for unseen validation groups")
        if summary["status"] != "complete":
            continue
        for event in events:
            if not event.get("selected"):
                continue
            state = Decision.from_event(event)
            pairs = {(slot,uid):i for i,(slot,uid,_) in enumerate(state.candidates)}
            picks = {}
            for name,key in (("actual","selected"),("same_state_j0","normal_choice")):
                pair = tuple(event[key])
                i = pairs[pair]
                role = state.candidates[i][2]
                value = distance(state.features[i], role)
                picks[name] = dict(pair=pair, role=role, distance=value,
                                   above_training_leave_group_p95=value>thresholds[role])
                counts[name] += picks[name]["above_training_leave_group_p95"]
            counts["decisions"] += 1
            records.append(dict(case=summary["case"],step=state.step,**picks))
    return dict(label="post_freeze_diagnostic", metric="RMS normalized same-role nearest executed feature",
                thresholds=thresholds, counts=dict(counts), rows=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows,_ = load_replay(args.replay)
    result = audit(args.runs,rows)
    write(args.out,result)
    print(result["counts"])


if __name__=="__main__":
    main()
