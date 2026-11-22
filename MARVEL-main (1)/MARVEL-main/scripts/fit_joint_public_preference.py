"""Fit a mask-preserving residual to genuine paired H300 action returns."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from gppo_joint_completion_metric import joint_completion_metric
from joint_policy_causal_trace import FROZEN_SHA
from integrations.gppo.public_pair_preference import event_features, make_checkpoint

VARIANTS = {
    "same_role": dict(mode="within_role", context_scale=1., action_scale=.85, support=.42),
    "all_roles": dict(mode="joint", context_scale=1., action_scale=.85, support=.42),
    "donor_aware": dict(mode="joint", context_scale=1., action_scale=.85, support=.42, opportunity=True),
    "donor_aware_calibrated": dict(mode="joint", context_scale=1., action_scale=.85, support=.42,
                                   opportunity=True, prior_scale=.15),
}


def read_summary(directory):
    path = directory/"summary.json"
    if path.is_file():
        return json.loads(path.read_text())
    with gzip.open(directory/"summary.json.gz","rt") as handle:
        return json.load(handle)


def read_events(directory):
    with gzip.open(directory/"events.jsonl.gz","rt") as handle:
        return {r["step"]: r for r in map(json.loads, handle)}


def applied(batch):
    return [(r["slot_id"], r["uav_id"], r["task_type"]) for r in batch]


def require_applied_choice(event, summary):
    receipt = event.get("receipt")
    step = event["step"]
    if (receipt is None or event["selected"] != [receipt["slot_id"], receipt["uav_id"]]
            or applied([receipt]) != applied(summary["commitments"][step//10])):
        raise ValueError("Counterfactual action does not match the real atomic commitment")


def validated_score(row):
    if (len(row["trace"]) != 3000 or len(row["commitments"]) != 300
            or joint_completion_metric(row["trace"],300) != row["score"]):
        raise ValueError("Expected a verified full H300 score")
    if (row.get("base_sha256") or row.get("frozen_sha256")) != FROZEN_SHA:
        raise ValueError("Counterfactual does not use the frozen J0 actor")
    return float(row["score"]["secondary_joint_completion_seconds"])


def contrast(control_dir, treatment_dir, *, control=None, control_events=None, opportunity=False):
    control = read_summary(control_dir) if control is None else control
    control_events = read_events(control_dir) if control_events is None else control_events
    treated = read_summary(treatment_dir)
    c0, c1 = validated_score(control), validated_score(treated)
    if any(control[k] != treated[k] for k in ("case", "initial_sha256", "config_sha256", "scenario_sha256")):
        raise ValueError("Counterfactual arms are not a physical pair")
    step = treated["intervention_step"]
    conditioned = treated["policy"] == "learned"
    if treated["policy"] not in ("pair", "relay", "nearest_relay", "opportunity_relay", "learned") or step is None or not treated["intervened"]:
        raise ValueError("No applied, single-action counterfactual")
    if conditioned and (control.get("policy") != "learned" or control.get("intervention_step") is not None
                        or not control.get("checkpoint_sha256")
                        or control["checkpoint_sha256"] != treated["checkpoint_sha256"]):
        raise ValueError("Conditioned counterfactual needs the same frozen continuation policy")
    # Before this selection, the two physics prefixes and every real commitment agree.
    if (control["trace"][:step] != treated["trace"][:step]
            or list(map(applied,control["commitments"][:step//10])) !=
            list(map(applied,treated["commitments"][:step//10]))):
        raise ValueError("Trajectory diverged before the chosen fork")
    events = read_events(treatment_dir)
    ref, observed = control_events[step], events[step]
    reference_key = "reference_choice" if conditioned else "normal_choice"
    mode = "conditioned_intervention" if conditioned else "single_intervention"
    if (any(ref[k] != observed[k] for k in ("snapshot_version", "input_sha256","mask"))
            or ref["selected"] != observed[reference_key] or ref["selected"] == observed["selected"]
            or [e["step"] for e in events.values() if e.get("mode")==mode] != [step]):
        raise ValueError("Counterfactual lacks an identical input or a changed committed choice")
    require_applied_choice(ref, control)
    require_applied_choice(observed, treated)
    rows = observed["rows"]
    by_pair = {(r["slot"],r["uav"]):i for i,r in enumerate(rows)}
    oi = by_pair[tuple(ref["selected"])]
    ni = by_pair[tuple(observed["selected"])]
    context, features = event_features(observed, opportunity=opportunity)
    row = dict(case=treated["case"], step=step, intervention=treatment_dir.name,
                control_seconds=c0, treatment_seconds=c1, treatment_safe=treated["score"]["whole_window_safe"],
                treatment_success=treated["score"]["joint_success"],
                context=context.tolist(), new_features=features[ni].tolist(), old_features=features[oi].tolist(),
                old_logit=rows[oi]["logit"], new_logit=rows[ni]["logit"],
                old_role=rows[oi]["role"], new_role=rows[ni]["role"],
                old_pair=ref["selected"], new_pair=observed["selected"],
                snapshot_version=observed["snapshot_version"], input_sha256=observed["input_sha256"])
    if conditioned:
        row["continuation_checkpoint_sha256"] = treated["checkpoint_sha256"]
    return row


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--control", type=Path, required=True)
    cli.add_argument("--treatment", type=Path, nargs="+", required=True)
    cli.add_argument("--extra-pair", type=Path, nargs=2, action="append", default=[],
                     metavar=("CONTROL", "TREATMENT"))
    cli.add_argument("--variant", choices=VARIANTS, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    variant = VARIANTS[args.variant]
    opportunity = variant.get("opportunity", False)
    base, events = read_summary(args.control), read_events(args.control)
    rows = [contrast(args.control, source, control=base, control_events=events, opportunity=opportunity)
            for source in args.treatment]
    rows.extend(contrast(control, treatment, opportunity=opportunity) for control, treatment in args.extra_pair)
    if args.variant=="same_role" and any(r["old_role"]!=r["new_role"] for r in rows):
        raise ValueError("The within-role model only accepts matched same-role evidence")
    data = make_checkpoint(rows, base_sha=FROZEN_SHA, **variant)
    blob = (json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("xb") as handle:
        handle.write(blob)
    fields = ("step", "intervention", "old_pair", "new_pair", "control_seconds", "treatment_seconds")
    log = dict(checkpoint=str(args.out), sha256=hashlib.sha256(blob).hexdigest(), variant=args.variant,
               pairs=[{key: row[key] for key in fields} for row in rows])
    print(json.dumps(log), flush=True)


if __name__=="__main__":
    main()
