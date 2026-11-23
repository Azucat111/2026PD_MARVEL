"""Recompute physical scores and summarize paired mature-RL benchmarks."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import statistics

from build_mature_rl_replay import import_live
from gppo_joint_completion_metric import joint_completion_metric
from summarize_joint_policy_causal import window_times
from verify_joint_atomic import ROOT, write

PHASE3 = ROOT / "artifacts/joint_ppo_20261122/evaluation"
P2_RUNS = ROOT / "artifacts/joint_autoresearch_20261122/runs"
P2_OFFICIAL = {
    "30_static_1280021": "p2-o-static",
    "30_dynamic_1280022": "p2-o-dynamic",
    "60_static_9164": "p2-60-static",
    "60_dynamic_1981204": "p2-60-dynamic",
}


def raw_summary(directory, filename):
    path = directory / f"{filename}.json"
    if path.is_file():
        return json.loads(path.read_text())
    path = path.with_suffix(".json.gz")
    if path.is_file():
        with gzip.open(path,"rt") as handle:
            return json.load(handle)
    return None


def verified_score(result):
    if not result:
        return None
    score = result.get("score")
    if not score:
        return None
    if (len(result["trace"]) != 3000 or len(result["commitments"]) != 300
            or joint_completion_metric(result["trace"],300.) != score):
        raise ValueError("Incomplete or mismatched scoring ledger")
    return score


def event_effect(directory, summary):
    if not (directory/"events.jsonl.gz").is_file():
        return {}
    _, events, _ = import_live(directory)
    counts = Counter()
    changed = []
    latencies = []
    bound = len(summary["trace"])
    for e in events:
        if not e.get("selected"):
            continue
        if e["step"] >= bound or [e["receipt"]] != summary["commitments"][e["step"]//10]:
            raise ValueError("Reported policy action lacks a real atomic receipt")
        counts["decisions"] += 1
        rows = { (row["slot"],row["uav"]): row for row in e["rows"] }
        choice, j0 = (rows[tuple(e[key])] for key in ("selected","normal_choice"))
        mixed = len({row["role"] for row in rows.values()}) > 1
        counts["mixed"] += mixed
        counts["search" if choice["role"] == 1 else "relay"] += 1
        counts["mixed_relay"] += bool(mixed and choice["role"] == 2)
        latency = e.get("learned_inference_seconds")
        if latency is not None:
            latencies.append(latency)
        if choice != j0:
            direction = f"{j0['role']}_to_{choice['role']}"
            counts[direction] += 1
            changed.append(dict(step=e["step"], candidate_count=len(rows), j0=e["normal_choice"], chosen=e["selected"],
                                role_change=direction, snapshot_version=e["snapshot_version"]))
    return dict(decisions=counts["decisions"], mixed=counts["mixed"], search=counts["search"], relay=counts["relay"],
                mixed_relay=counts["mixed_relay"], changed_count=len(changed),
                cross_role_count=counts["1_to_2"]+counts["2_to_1"], direction_counts=dict(counts),
                changed=changed, mean_inference_ms=None if not latencies else statistics.mean(latencies)*1000.)


def arm_from_folder(name):
    if name.startswith("validation-"):
        return name.removeprefix("validation-").rsplit("-", 1)[0]
    if name.startswith("dev-"):
        return name.rsplit("-", 1)[0].removeprefix("dev-")
    if name.startswith("smoke-"):
        return name.removeprefix("smoke-")
    return "J0" if name.startswith("j0-") else "explore"


def row_for(directory, summary, arm, *, inherited=False):
    score = verified_score(summary)
    times = window_times(summary["trace"])
    meta = directory/"learning_policy.json"
    if meta.is_file():
        model = json.loads(meta.read_text())
        sha = model["checkpoint_sha256"]
    else:
        sha = summary.get("checkpoint_sha256") or summary.get("base_sha256")
    row = dict(run=directory.name, case=summary["case"], arm=arm, status=summary["status"],
               init=summary["initial_sha256"], config=summary["config_sha256"], ticks=len(summary["trace"]),
               C=None if score is None else score["secondary_joint_completion_seconds"],
               safe=None if score is None else score["whole_window_safe"], success=None if score is None else score["joint_success"],
               search_done=times["nonrelay_done"], radio_first=times["radio_first_09"],
               stable_start=times["final_joint_window_start"], checkpoint_sha256=sha,
               wall_seconds=summary["seconds"], error=summary.get("error"),
               inherited=inherited, collisions=sum(r["collisions"] for r in summary["trace"]),
               no_fly=max((r["no_fly_violations"] for r in summary["trace"]),default=None))
    if not inherited:
        row.update(event_effect(directory, summary))
    return row


def pair_gain(records, cases, a, b):
    matched = []
    for case in cases:
        x,y = (records.get((case, arm)) for arm in (a,b))
        if x is None or y is None or x["C"] is None or y["C"] is None:
            continue
        if (x["init"],x["config"]) != (y["init"],y["config"]):
            raise ValueError(f"Initial or scenario configuration mismatch for {case}")
        matched.append((case,x["C"],y["C"]))
    new = sum(x[1] for x in matched)
    ref = sum(x[2] for x in matched)
    return dict(arm=a, reference=b, paired=len(matched), planned=len(cases), rows=matched,
                arm_seconds=new, reference_seconds=ref, gain_percent=None if not ref else 100.*(1.-new/ref))


def short_fork(case, path):
    baseline = path/f"j0-{case}"
    changed = path/f"explore-{case}"
    before, live = raw_summary(baseline,"summary"), raw_summary(changed,"summary")
    if before is None or live is None:
        return None
    if (before["initial_sha256"], before["config_sha256"]) != (live["initial_sha256"], live["config_sha256"]):
        raise ValueError("Same-seed exploration candidate did not pair")
    _, a, _ = import_live(baseline)
    _, b, _ = import_live(changed)
    base = {r["step"]:r for r in a if r.get("selected")}
    effect = next((r for r in b if r.get("selected") and r["selected"] != r["normal_choice"]),None)
    if effect is None:
        return dict(case=case, changed=False)
    step = effect["step"]
    old = base[step]
    if (before["trace"][:step] != live["trace"][:step] or
        before["commitments"][:step//10] != live["commitments"][:step//10] or
        any(old[k]!=effect[k] for k in ("snapshot_version","input_sha256","mask"))):
        raise ValueError("First mixed fork did not match the frozen trajectory prefix")
    return dict(case=case, changed=True, step=step, frozen=old["selected"], explored=effect["selected"],
                behavior_probability=effect["executed_action_probability"], matched_public_prefix=True,
                control_cost=verified_score(before)["secondary_joint_completion_seconds"] if before["score"] else None,
                intervention_cost=verified_score(live)["secondary_joint_completion_seconds"] if live["score"] else None)


def compile_rows(path):
    rows = []
    for directory in sorted(path.iterdir()):
        if not directory.is_dir():
            continue
        arm = arm_from_folder(directory.name)
        summary = raw_summary(directory,arm if arm in ("R0","R1") else "summary")
        if summary is not None:
            rows.append(row_for(directory,summary,arm))
    for case, p2 in P2_OFFICIAL.items():
        for arm in ("R0","R1","P1"):
            directory = PHASE3/case
            rows.append(row_for(directory,raw_summary(directory,arm),f"archive-{arm}",inherited=True))
        directory = P2_RUNS/p2
        rows.append(row_for(directory,raw_summary(directory,"summary"),"archive-P2-O",inherited=True))
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path, required=True)
    cli.add_argument("--csv", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    rows = compile_rows(args.runs)
    rows_by = {(r["case"],r["arm"]):r for r in rows}
    validation_cases = [f"{domain}_{24110801+i}" for i,domain in enumerate(("30_static","30_dynamic","60_static","60_dynamic"))]
    dev_arms = [f"expanded-{name}" for name in ("ddqn","sac")]
    archive_refs = ["archive-R0","archive-R1","archive-P1","archive-P2-O"]
    pairs = [pair_gain(rows_by,list(P2_OFFICIAL),a,b) for a in dev_arms for b in archive_refs]
    pairs.extend(pair_gain(rows_by,validation_cases,a,b) for a in ("ddqn","sac") for b in ("R0","R1","P2-O"))
    selection_cases = ["30_static_24110701","60_dynamic_24110704"]
    learning_curves = [pair_gain(rows_by,selection_cases,f"expanded-{a}",f"archive-{a}")
                       for a in ("ddqn","sac")]
    equal_budget = [pair_gain(rows_by,selection_cases,f"{stage}-ddqn",f"{stage}-sac")
                    for stage in ("archive","expanded")]
    samples = [row for row in rows if not row["inherited"]]
    fields = ["run","case","arm","status","ticks","C","safe","success","search_done","radio_first","stable_start",
              "checkpoint_sha256","wall_seconds","decisions","mixed","search","relay","mixed_relay","changed_count",
              "cross_role_count","mean_inference_ms","inherited","collisions","no_fly","error"]
    with args.csv.open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore",lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    forks = [short_fork(f"{domain}_{24110601+i}",args.runs)
             for i,domain in enumerate(("30_static","30_dynamic","60_static","60_dynamic"))]
    freeze_path = args.runs/"freeze.json"
    if not freeze_path.is_file():
        freeze_path = args.runs.parent/"freeze.json"
    freeze = json.loads(freeze_path.read_text()) if freeze_path.is_file() else None
    if freeze is not None:
        for row in samples:
            if row["run"].startswith("validation-") and row["arm"]==freeze["chosen"]:
                if row["case"] not in freeze["validation_cases"] or row["checkpoint_sha256"]!=freeze["checkpoint_sha256"]:
                    raise ValueError("Independent validation did not use the preregistered frozen checkpoint")
    result = dict(rows=rows, pairs=pairs, learning_curves=learning_curves, equal_budget=equal_budget,
                  freeze=freeze, first_mixed_forks=[f for f in forks if f is not None],
                  live_counts=dict(Counter(r["status"] for r in samples)))
    write(args.out,result)
    print(json.dumps(result["live_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
