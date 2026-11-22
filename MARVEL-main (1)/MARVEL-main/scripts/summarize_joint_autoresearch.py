"""Validate physical pairing and produce the research run ledger."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

from fit_joint_public_preference import applied, read_events, read_summary
from gppo_joint_completion_metric import joint_completion_metric
from summarize_joint_policy_causal import window_times
from verify_joint_atomic import ROOT, write

OLD = ROOT/"artifacts/joint_ppo_20261122/evaluation"
OLD_J0 = ROOT/"artifacts/joint_causal_20261122/runs/J0"
BASE_CASE = "30_static_1280021"


def full_score(row):
    score = row.get("score")
    if score is None:
        return None
    if (len(row["trace"]) != 3000 or len(row["commitments"]) != 300
            or joint_completion_metric(row["trace"],300.) != score):
        raise ValueError(f"Invalid H300 trace for {row['case']}")
    return float(score["secondary_joint_completion_seconds"])


def matching_identity(a,b):
    return all(a[k]==b[k] for k in ("case","initial_sha256","config_sha256"))


def physical_metrics(row):
    trace = row["trace"]
    times = window_times(trace)
    return dict(h300_seconds=full_score(row), radio_first=times["radio_first_09"],
                stable_start=times["final_joint_window_start"], tasks_done=times["nonrelay_done"],
                safe=None if row.get("score") is None else row["score"]["whole_window_safe"],
                success=None if row.get("score") is None else row["score"]["joint_success"],
                collisions=sum(r["collisions"] for r in trace),
                no_fly_violations=trace[-1]["no_fly_violations"] if trace else None)


def loaded_references(cases, rows, root):
    refs = {case: {} for case in cases}
    for case in cases:
        for arm in ("R0", "P0", "R1", "P1"):
            path = OLD/case/f"{arm}.json.gz"
            if path.is_file():
                with gzip.open(path,"rt") as handle:
                    refs[case][arm] = json.load(handle)
    if BASE_CASE in refs:
        refs[BASE_CASE]["J0"] = read_summary(OLD_J0)
    for row in rows.values():
        if row["policy"]=="J0":
            refs[row["case"]]["J0"] = row
    fresh = []
    for arm in ("R0", "R1", "P1"):
        for path in sorted(root.glob(f"*/{arm}.json*")):
            if path.suffix == ".gz":
                with gzip.open(path,"rt") as handle:
                    saved = json.load(handle)
            else:
                saved = json.loads(path.read_text())
            case = saved["case"]
            fresh.append((path.parent.name, saved))
            if case in refs and arm not in refs[case] and saved["status"] == "complete":
                refs[case][arm] = saved
    return refs, fresh


def action_metrics(events):
    result = []
    for e in events.values():
        if not e.get("selected"):
            continue
        selected = next(r for r in e["rows"] if [r["slot"],r["uav"]]==e["selected"])
        prior = next(r for r in e["rows"] if [r["slot"],r["uav"]]==e["normal_choice"])
        raw = e["actors"]["J0"]
        raw_probs = dict(zip(raw["flat_indices"],raw["prob_t1"],strict=True))
        score = e.get("ranker")
        probs = None if score is None else {r["flat"]: v for r,v in zip(e["rows"],score["probs"],strict=True)}
        result.append(dict(step=e["step"], pair=e["selected"], same_state_j0=e["normal_choice"],
                           role=selected["role"], j0_role=prior["role"],
                           j0_raw_logit_for_selected=selected["logit"], j0_raw_logit_for_j0=prior["logit"],
                           j0_probability_for_selected=raw_probs[selected["flat"]],
                           j0_probability_for_j0=raw_probs[prior["flat"]],
                           mixed=any(r["role"]==1 for r in e["rows"]) and any(r["role"]==2 for r in e["rows"]),
                           selected_combined_probability=None if probs is None else probs[selected["flat"]],
                           j0_choice_combined_probability=None if probs is None else probs[prior["flat"]],
                           snapshot_version=e["snapshot_version"], command_sha256=e["command_sha256"]))
    return result


def summarize(root, out_csv, out_json):
    dirs = {p.name:p for p in sorted(root.iterdir()) if p.is_dir() and
            any((p/name).is_file() for name in ("summary.json","summary.json.gz"))}
    rows = {key:read_summary(path) for key,path in dirs.items()}
    events_by_run = {key:read_events(path) for key,path in dirs.items()}
    control_events = {row["case"]:events_by_run[key] for key,row in rows.items() if row["policy"]=="J0"}
    control_events[BASE_CASE] = read_events(OLD_J0)
    refs, fresh = loaded_references({row["case"] for row in rows.values()},rows,root)
    records = []
    for label,row in rows.items():
        reference = {}
        for arm,base in refs[row["case"]].items():
            if not matching_identity(row,base):
                raise ValueError(f"Physical pairing mismatch for {label}/{arm}")
            reference[arm] = full_score(base)
        metrics = physical_metrics(row)
        live_events = events_by_run[label]
        actions = action_metrics(live_events)
        choices_changed = sum(a["pair"]!=a["same_state_j0"] for a in actions)
        roles_changed = sum(a["role"]!=a["j0_role"] for a in actions)
        base = refs[row["case"]].get("J0")
        actual_first_diff = None
        first_radio_trace_diff = None
        first_pose = None
        if base is not None:
            actual_first_diff = next((i*10 for i,(a,b) in enumerate(zip(row["commitments"],base["commitments"]))
                                      if applied(a)!=applied(b)),None)
            first_radio_trace_diff = next((i for i,(a,b) in enumerate(zip(row["trace"],base["trace"]),1)
                                           if a["connectivity"]!=b["connectivity"]),None)
            reference_events = control_events.get(row["case"], {})
            for step,e in live_events.items():
                if step not in reference_events:
                    continue
                old = reference_events[step]["stale_xy"]
                moved = {int(uid):math.dist(pos, old[uid]) for uid,pos in e["stale_xy"].items() if pos != old[uid]}
                if moved:
                    first_pose = dict(step=step, public_pose_displacements=moved)
                    break
        seconds = metrics["h300_seconds"]
        gain = {k: None if seconds is None or v is None else 100*(1-seconds/v) for k,v in reference.items()}
        j0_gain = gain.get("J0")
        verdict = ("control" if row["policy"]=="J0" else "failure" if metrics["success"] is False else
                   "prefix" if row["status"]=="prefix" else "unscored" if seconds is None else
                   "no_measured_J0" if j0_gain is None else
                   "improved_vs_J0" if j0_gain>1e-8 else
                   "regressed_vs_J0" if j0_gain<-1e-8 else "no_h300_gain_vs_J0")
        records.append(dict(run=label, case=row["case"], policy=row["policy"], status=row["status"],
                            ticks=len(row["trace"]), wall_seconds=row["seconds"], checkpoint_sha256=row["checkpoint_sha256"],
                            **metrics, choices=len(actions), mixed_events=sum(a["mixed"] for a in actions),
                            changed_choices=choices_changed, changed_roles=roles_changed,
                            first_committed_difference_tick=actual_first_diff, first_radio_difference_tick=first_radio_trace_diff,
                            first_public_pose_difference=first_pose,
                            comparisons=reference, gain_percent=gain, verdict=verdict, events=actions,
                            initial_sha256=row["initial_sha256"],config_sha256=row["config_sha256"], error=row["error"]))
    for label, row in fresh:
        records.append(dict(run=label, case=row["case"], policy=row["arm"], status=row["status"],
                            ticks=len(row["trace"]), wall_seconds=row["seconds"],
                            checkpoint_sha256=row["checkpoint_sha256"], **physical_metrics(row),
                            choices=sum(map(len,row["commitments"])), comparisons={}, gain_percent={},
                            verdict="control", error=row.get("error")))
    prototype = root/"p2-j-static-only-offline.json"
    if prototype.is_file():
        blob = (root/"models/p2-j-static-only.json").read_bytes()
        records.append(dict(run="p2-j-static-only", case=BASE_CASE, policy="all_roles", status="offline",
                            checkpoint_sha256=hashlib.sha256(blob).hexdigest(), comparisons={}, gain_percent={},
                            verdict="no_additional_changes_on_replayed_graphs", events=json.loads(prototype.read_text())))
    fields = ["run","case","policy","status","ticks","wall_seconds","h300_seconds","radio_first","stable_start","tasks_done",
              "safe","success","collisions","no_fly_violations","choices","mixed_events","changed_choices",
              "changed_roles","first_committed_difference_tick","first_radio_difference_tick",
              "J0_seconds","P1_seconds","R1_seconds","R0_seconds","P0_seconds","gain_vs_J0_pct","gain_vs_P1_pct",
              "gain_vs_R1_pct","gain_vs_R0_pct","checkpoint_sha256","verdict","error"]
    with out_csv.open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for row in records:
            flat = {f: row.get(f) for f in fields}
            for name in ("J0","P1","R1","R0","P0"):
                flat[f"{name}_seconds"] = row["comparisons"].get(name)
            for name in ("J0","P1","R1","R0"):
                flat[f"gain_vs_{name}_pct"] = row["gain_percent"].get(name)
            writer.writerow(flat)
    write(out_json, records)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path, required=True)
    cli.add_argument("--csv", type=Path, required=True)
    cli.add_argument("--json", type=Path, required=True)
    args = cli.parse_args()
    summarize(args.runs,args.csv,args.json)


if __name__=="__main__":
    main()
