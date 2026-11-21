"""Recompute and pair completed development H300 runs without imputing truncations."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from gppo_joint_completion_metric import joint_completion_metric

ARMS = ("R0", "P0", "R1", "P1")


def arm_data(row):
    trace = row["trace"]
    score = None
    if row["status"] == "complete":
        if len(trace) != 3000:
            raise ValueError("Incomplete trace marked complete")
        score = joint_completion_metric(trace, 300)
        if row.get("score") != score:
            raise ValueError("Stored score does not match physical trace")
    elif "score" in row:
        raise ValueError("Truncated window must not have an H300 score")
    roles = Counter("search" if choice["task_type"] == 1 else "relay"
                    for batch in row["commitments"] for choice in batch)
    offers = row["offers"]
    mixed = [o for o in offers if o["search"] and o["relay"]]
    comparisons = [o for o in offers if "frozen_greedy" in o]
    final = trace[-1] if trace else {}
    return dict(status=row["status"], ticks=len(trace), score=score, error=row.get("error"),
                initial_sha256=row["initial_sha256"], config_sha256=row["config_sha256"],
                checkpoint_sha256=row["checkpoint_sha256"], compatibility_policy=row["compatibility_policy"],
                first_nonrelay_s=next((t["time"] for t in trace if t["nonrelay_complete"]), None),
                first_connected_s=next((t["time"] for t in trace if t["connectivity"] >= .9), None),
                connected_ticks=sum(t["connectivity"] >= .9 for t in trace),
                jointly_ready_ticks=sum(t["nonrelay_complete"] and t["connectivity"] >= .9
                                        and not t["safety_active"] and t["escapes"] == t["safety_records"]
                                        for t in trace),
                final_connectivity=final.get("connectivity"), final_nonrelay_complete=final.get("nonrelay_complete"),
                safety_triggers=final.get("safety_records"), final_active_safety=final.get("safety_active"),
                collisions=sum(t["collisions"] for t in trace),
                no_fly_violations=final.get("no_fly_violations"),
                actions=dict(roles), controller_events=len(row["commitments"]),
                joint_selection_events=len(offers),
                mixed_events=len(mixed), shared_uav_events=sum(o["shared_uavs"] > 0 for o in mixed),
                greedy_comparison_events=len(comparisons),
                trained_greedy_action_diffs=sum(o["trained_greedy"] != o["frozen_greedy"] for o in comparisons),
                trained_greedy_role_diffs=sum(o["trained_greedy"]["role"] != o["frozen_greedy"]["role"]
                                              for o in comparisons), seconds=row["seconds"])


def paired_summary(rows, cases, *, p1_sha256):
    records = {case: {} for case in cases}
    pairs = {key: [] for key in ARMS if key != "P1"}
    for row in rows:
        case, arm = row["case"], row["arm"]
        if arm not in ARMS or arm in records[case]:
            raise ValueError(f"Unknown or duplicate arm: {case}/{arm}")
        record = arm_data(row)
        if arm == "P1" and (record["compatibility_policy"] or record["checkpoint_sha256"] != p1_sha256):
            raise ValueError("P1 must run the same frozen trained checkpoint")
        records[case][arm] = record
    per_case = {}
    for case, available in records.items():
        if len({(r["initial_sha256"], r["config_sha256"]) for r in available.values()}) > 1:
            raise ValueError(f"Physical pairing mismatch: {case}")
        per_case[case] = {}
        p1 = available.get("P1", {}).get("score")
        for ref in pairs:
            baseline = available.get(ref, {}).get("score")
            if baseline is None or p1 is None:
                continue
            p, r = (x["secondary_joint_completion_seconds"] for x in (p1, baseline))
            pairs[ref].append(dict(case=case, p1=p, reference=r))
            per_case[case][ref] = 100*(1-p/r)
    pooled = {}
    for ref, matched in pairs.items():
        pooled[ref] = dict(count=len(matched), cases=[p["case"] for p in matched],
                           percent=None if not matched else
                           100*(1-sum(p["p1"] for p in matched)/sum(p["reference"] for p in matched)))
    return dict(case_arms={case: {arm: observed.get(arm, {"status": "not_run"})
                                 for arm in ARMS} for case, observed in records.items()},
                per_case_percent=per_case, pooled_comparable_pairs=pooled,
                independent_acceptance=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm-file", nargs="+", required=True, type=Path)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--p1-sha256", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    def load(path):
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as f:
                return json.load(f)
        return json.loads(path.read_text())

    protocol = load(args.protocol)
    summary = paired_summary([load(f) for f in args.arm_file], protocol["development_cases"],
                             p1_sha256=args.p1_sha256)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
