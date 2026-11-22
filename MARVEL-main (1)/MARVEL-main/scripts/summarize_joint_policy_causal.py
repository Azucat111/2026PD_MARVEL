"""Validate the Phase 3.5 runs and align scoring, radio and decisions."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from gppo_joint_completion_metric import joint_completion_metric
from joint_event_ppo import verify_rollout_files
from joint_policy_causal_trace import ARCHIVE, BASELINE, CASE, FROZEN_SHA, TRAINED_SHA, SWAP_ACTION, SWAP_STEP
from verify_joint_atomic import write
from integrations.gppo.mesh_relay import mesh_components

H300 = 3000
PREFIX = 900
ARMS = ("R1", "J0", "P1", "J0_SWAP_S3")
TICK_FIELDS = ("observed", "commands", "comm_hex", "tasks", "search", "relay", "probe", "control_step")
EVENT_FIELDS = ("snapshot_version", "input_sha256", "task_ids", "uav_ids", "mask", "candidates",
                "actors", "demands", "stale_xy", "search_before", "relay_before", "search_after",
                "relay_after", "probe_before", "probe_after", "selected", "command_sha256")


def load_json(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as src:
            return json.load(src)
    return json.loads(path.read_text())


def load_lines(path):
    with gzip.open(path, "rt") as src:
        return [json.loads(line) for line in src]


def commitments(rows):
    """Physics-facing identities. Critic value is logged but does not enter motion."""
    return [[(a["slot_id"], a["uav_id"], a["task_type"]) for a in batch] for batch in rows]


def prefix_matches(run, archived):
    ticks = run["ticks"]
    if not 0 < ticks <= H300 or len(run["trace"]) != ticks or len(run["commitments"]) != ticks//10:
        raise ValueError("Incomplete replay ledger")
    if (archived["status"] != "complete" or len(archived["trace"]) != H300 or
            len(archived["commitments"]) != H300//10 or
            joint_completion_metric(archived["trace"], 300) != archived["score"]):
        raise ValueError("Invalid official H300 reference")
    for key in ("case", "initial_sha256", "config_sha256"):
        if run[key] != archived[key]:
            raise ValueError(f"Archived pairing mismatch in {key}")
    scoring = next((step for step, (a,b) in enumerate(zip(run["trace"], archived["trace"]), 1) if a != b), None)
    actual, reference = commitments(run["commitments"]), commitments(archived["commitments"])
    applied = next((i*10 for i,(a,b) in enumerate(zip(actual, reference)) if a != b), None)
    return dict(ticks=ticks, scoring_trace=scoring is None, stable_commitments=applied is None,
                first_scoring_mismatch_tick=scoring, first_commitment_mismatch_control=applied)


@dataclass
class Run:
    summary: dict
    ticks: dict
    events: dict

    @classmethod
    def read(cls, path, arm):
        summary_path = path/"summary.json"
        summary = load_json(summary_path if summary_path.exists() else path/"summary.json.gz")
        tick_rows = load_lines(path/"ticks.jsonl.gz")
        event_rows = load_lines(path/"events.jsonl.gz")
        ticks = {row["step"]: row for row in tick_rows}
        events = {row["step"]: row for row in event_rows}
        expected_ticks = H300 if arm == "J0" else PREFIX
        if (summary["case"] != CASE or summary["arm"] != arm or summary["ticks"] != expected_ticks or
                summary["simulator_base"] != BASELINE or summary["frozen_sha256"] != FROZEN_SHA or
                summary["trained_sha256"] != TRAINED_SHA or
                summary["checkpoint_sha256"] != (TRAINED_SHA if arm == "P1" else FROZEN_SHA) or
                len(ticks) != len(tick_rows) or list(ticks) != list(range(1, expected_ticks+1)) or
                len(events) != len(event_rows)):
            raise ValueError(f"Incorrect provenance or clock for {arm}")
        for step, expected in enumerate(summary["trace"], start=1):
            if {key: ticks[step][key] for key in expected} != expected:
                raise ValueError(f"Trace block disagrees with scoring ledger at {arm}/{step}")
        with ZipFile(path/"public.zip") as snapshots:
            for name in snapshots.namelist():
                step = int(Path(name).stem)
                if hashlib.sha256(snapshots.read(name)).hexdigest() != events[step]["snapshot_version"].split(":")[-1]:
                    raise ValueError(f"Invalid public snapshot at {arm}/{step}")
        for step, event in events.items():
            actual = event["receipt"]
            expected = [] if actual is None else [actual]
            if commitments([expected]) != commitments([summary["commitments"][step//10]]):
                raise ValueError(f"Atomic receipt disagrees with the case ledger at {arm}/{step}")
        return cls(summary, ticks, events)


def first_differences(left, right, fields):
    first = {field: None for field in fields}
    overlap_end = min(max(left, default=-1), max(right, default=-1))
    for step in sorted(set(left) | set(right)):
        if step > overlap_end:
            break
        for field in fields:
            if first[field] is None and left.get(step, {}).get(field) != right.get(step, {}).get(field):
                first[field] = step
    return first


def first_change(left, right, fields):
    return {field: {'step': step, 'time': round(step*.1, 3)}
            for field, step in first_differences(left, right, fields).items() if step is not None}


def first_policy_fork(left, right, ticks):
    """Treat only a different selection on the same pre-action input as a fork."""
    step = first_differences(left, right, ("selected",))["selected"]
    if step is None or step not in left or step not in right:
        return None
    graph = ("snapshot_version", "input_sha256", "mask", "candidates")
    if (any(left[step].get(key) != right[step].get(key) for key in graph)
            or not left[step].get("input_sha256")
            or any(diff["step"] <= step for diff in ticks.values())):
        return None
    return step


def window_times(trace):
    radio_first, radio_since, ready_since = None, None, None
    radio_stable, joint_stable, joint_end, tasks, radio_last_blocker = None, None, None, None, None
    for row in trace:
        t = row["time"]
        ready = row["nonrelay_complete"]
        if ready and tasks is None:
            tasks = t
        radio = row["connectivity"] >= .9
        if radio:
            if radio_first is None:
                radio_first = t
            if radio_since is None:
                radio_since = t
            if radio_stable is None and t-radio_since >= 3.-1e-8:
                radio_stable = radio_since
        else:
            radio_since = None
            if joint_end is None:
                radio_last_blocker = t
        safety = (not row["safety_active"] and row["escapes"] == row["safety_records"])
        if ready and radio and safety:
            if ready_since is None:
                ready_since = t
            if joint_stable is None and t-ready_since >= 3.-1e-8:
                joint_stable, joint_end = ready_since, t
        else:
            ready_since = None
    return dict(nonrelay_done=tasks, radio_first_09=radio_first, first_3s_radio_start=radio_stable,
                final_joint_window_start=joint_stable, first_joint_hold_end=joint_end,
                last_radio_blocker_before_completion=radio_last_blocker,
                wait_after_nonrelay=None if tasks is None or joint_stable is None else joint_stable-tasks)


def components(graph):
    remaining = set(range(len(graph)))
    groups = []
    while remaining:
        start = min(remaining)
        found, pending = {start}, [start]
        remaining.remove(start)
        for node in pending:
            adjacent = remaining.intersection(map(int, np.flatnonzero(graph[node])))
            pending.extend(sorted(adjacent))
            remaining.difference_update(adjacent)
            found.update(adjacent)
        groups.append(sorted(found))
    return sorted(groups, key=lambda group: (-len(group), group))


def decode_graph(row):
    n = len(row["observed"])
    packed = np.frombuffer(bytes.fromhex(row["comm_hex"]), dtype=np.uint8)
    return np.unpackbits(packed, count=n*n).reshape(n,n)


def threshold_merge(ticks, time):
    if time is None:
        return None
    step = round(time*10)
    if step-1 not in ticks:
        return None
    earlier, current = decode_graph(ticks[step-1]), decode_graph(ticks[step])
    before, after = components(earlier), components(current)
    labels = {uid: i for i, group in enumerate(before) for uid in group}
    positions = {row[0]: np.asarray(row[1]) for row in ticks[step]["observed"]}
    gained = []
    for i, j in zip(*np.where(np.triu((current != 0) & (earlier == 0), 1))):
        i, j = int(i), int(j)
        if labels[i] != labels[j]:
            graph = current.copy()
            graph[i,j] = graph[j,i] = 0
            gained.append(dict(uavs=[i,j], distance=float(np.linalg.norm(positions[i]-positions[j])),
                               largest_without_edge=len(components(graph)[0])))
    relay = ticks[step]["relay"]
    isolated = {}
    for helper, _ in relay.values():
        amended = current.copy()
        amended[helper, :] = amended[:, helper] = 0
        isolated[helper] = len(components(amended)[0])
    return dict(step=step, largest_before=len(before[0]), largest_after=len(after[0]),
                before=before, after=after, gained_intercomponent_edges=gained, relay=relay,
                largest_if_helper_isolated=isolated, probe=ticks[step]["probe"])


def helper_motion_until_radio(run, end_time):
    if end_time is None or round(end_time*10) not in run.ticks:
        return None
    first = next((e["step"] for e in run.events.values()
                  if e["receipt"] and e["receipt"]["task_type"] == 2), None)
    if first is None or first not in run.ticks:
        return None
    stop = round(end_time*10)
    motion = {}
    for uid in sorted({owner for owner, _ in run.ticks[stop]["relay"].values()}):
        poses = [next(row[1] for row in run.ticks[step]["observed"] if row[0] == uid)
                 for step in range(first, stop+1)]
        motion[uid] = sum(math.dist(a,b) for a,b in zip(poses, poses[1:]))
    return dict(first_relay_step=first, radio_step=stop, distance_m=motion)


def forecast_retarget(positions, helper, anchor, range_m):
    before = mesh_components(positions, range_m)
    after = mesh_components({**positions, helper: anchor}, range_m)
    preserved = all(any(set(old)-{helper} <= set(new) for new in after) for old in before)
    return dict(component_sizes=[len(group) for group in after], preserved_nonhelpers=preserved,
                helper_distance=math.dist(positions[helper], anchor))


def retarget_evidence(r1, j0):
    step = first_differences(r1.events, j0.events, ("relay_after",))["relay_after"]
    if step is None or step not in r1.events or step not in j0.events:
        return None
    left, right = r1.events[step], j0.events[step]
    for target, (uid, new) in right["relay_after"].items():
        if target not in right["relay_before"] or target not in left["relay_after"]:
            continue
        old = right["relay_before"][target][1]
        if new == old or left["relay_after"][target] != [uid, old]:
            continue
        records = {}
        for arm, run in (("R1", r1), ("J0", j0)):
            event = run.events[step]
            positions = {int(k): v for k,v in event["stale_xy"].items()}
            a, b = (decode_graph(run.ticks[k]) for k in (step-1, step))
            records[arm] = dict(
                stale_groups=mesh_components(positions, run.summary["radio_range"]),
                lost_physical_edges=np.argwhere(np.triu((a>0)&(b==0),1)).tolist(),
                retained_anchor=forecast_retarget(positions, uid, old, run.summary["radio_range"]),
                moved_anchor=forecast_retarget(positions, uid, new, run.summary["radio_range"]))
        return dict(step=step, target=int(target), helper=uid, old_anchor=old, new_anchor=new, arms=records)
    return None


def event_summary(event, range_m):
    selected = event.get("selected")
    chosen = next((c for c in event["candidates"] if [c["slot"], c["uav"]] == selected), None)
    actors = event.get("actors")
    logit_delta = tv_t1 = tv_t3 = None
    if actors:
        a, b = actors["J0"], actors["P1"]
        logit_delta = max(abs(x-y) for x,y in zip(a["logits"], b["logits"], strict=True))
        tv_t1 = .5*sum(abs(x-y) for x,y in zip(a["prob_t1"], b["prob_t1"], strict=True))
        tv_t3 = .5*sum(abs(x-y) for x,y in zip(a["prob_t3"], b["prob_t3"], strict=True))
    relay = event["relay_before"]
    stale = event["stale_xy"]
    distances = {int(owner): math.dist(stale[str(owner)], anchor) for owner, anchor in relay.values()}
    groups = mesh_components(stale, range_m)
    return dict(step=event["step"], seconds=event["step"]*.1, input_sha256=event.get("input_sha256"),
                selected=selected, chosen=chosen, receipt=event["receipt"],
                legal_counts=dict(Counter(str(c["role"]) for c in event["candidates"])),
                normal_choice=event.get("normal_choice"), relay_before=relay, relay_after=event["relay_after"],
                stale_relay_distance=distances, relay_demand_slots=[x["slots"] for x in event["demands"]],
                demand_public_connectivity=[x["connectivity"] for x in event["demands"]],
                observed_mesh_sizes=[len(group) for group in groups],
                probe_after=event["probe_after"], retarget=event["retarget"], release_gate=event["release_gate"],
                logit_delta_max=logit_delta, policy_tv_t1=tv_t1, policy_tv_t3=tv_t3,
                actor_picks=None if not actors else {k: v["pick"] for k,v in actors.items()},
                relay_mass_t1=None if not actors else {k: v["relay_mass_t1"] for k,v in actors.items()},
                relay_mass_t3=None if not actors else {k: v["relay_mass_t3"] for k,v in actors.items()})


def learning_data():
    main_audit = load_json(ARCHIVE/"main-audit.json")
    update = load_json(ARCHIVE/"main-update/update.json")
    if update["checkpoint_sha256"] != TRAINED_SHA or main_audit["candidate_sha256"] != TRAINED_SHA:
        raise ValueError("Training diagnostics do not bind to P1")
    audit = {(arm["case"], event["step"]): event for arm in main_audit["cases"] for event in arm["states"]}
    episodes, rows = [], []
    for path in sorted((ARCHIVE/"training").iterdir()):
        rollout = torch.load(path/"rollout.pt", weights_only=True)
        verify_rollout_files(path, rollout)
        s, events = rollout["summary"], rollout["events"]
        if s["behavior_checkpoint_sha256"] != FROZEN_SHA or s["status"] != "complete":
            raise ValueError("Unverified training batch")
        episodes.append(dict(case=s["case"], total_control=s["action_events"]+s["forced_events"],
                             mixed=s.get("mixed_events",0), samples=len(events),
                             duration=s["score"]["secondary_joint_completion_seconds"]))
        for e in events:
            detail = audit[(s["case"], e["step"])]
            ti = torch.where(~e["batch"]["action_mask"].flatten())[0] // len(e["uav_ids"])
            roles = [e["roles"][i] for i in ti.tolist()]
            old = e["old_log_probs"].double().exp()
            total = float(old.sum())
            if not math.isclose(total, 1., abs_tol=2e-6):
                raise ValueError("Invalid behavior distribution")
            rows.append(dict(case=s["case"], step=e["step"], next_step=e["next_step"],
                             duration_seconds=e["duration_seconds"], reward=e["reward"],
                             advantage=e["advantage"], return_target=e["return_target"], old_value=e["old_value"],
                             new_value=detail["value_after"], actual_role=e["chosen_role"],
                             actual_selected_logp=e["old_logp"],
                             search_legal=roles.count(1), relay_legal=roles.count(2),
                             relay_behavior_mass=float(old[torch.tensor([r==2 for r in roles])].sum()),
                             relay_p1_mass=detail["relay_mass_after"]))
    mixed = [e for e in rows if e["search_legal"] and e["relay_legal"]]
    return dict(episodes=episodes, events=rows, mixed=mixed, accepted=update["accepted_updates"],
                selected_role_counts=dict(Counter(e["actual_role"] for e in rows)),
                accepted_guard_rows=update["epochs"], controls=sum(e["total_control"] for e in episodes),
                path_search_probability_product=math.prod(1.-e["relay_behavior_mass"] for e in mixed))


def build_timeline(output, runs, official):
    columns = ["step", "t_s", "arm", "detail_available", "connectivity", "nonrelay_complete",
               "search_status", "search_targets_found", "safety_active", "active_search_assignments",
               "active_relay_anchors", "control_step", "relay_demand_slots", "relay_legal_pairs",
               "comm_hex", "observed_xy_heading_velocity", "control_commands", "probe"]
    with gzip.open(output, "wt", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for step in range(1, H300+1):
            for arm, run in runs.items():
                detail = run.ticks.get(step)
                trace = official[arm]["trace"] if arm in official else run.summary["trace"]
                if step > len(trace):
                    continue
                row = trace[step-1]
                event = run.events.get(detail["control_step"], {}) if detail else {}
                search_task = detail["tasks"].get("T2_target_search", []) if detail else []
                writer.writerow(dict(step=step, t_s=row["time"], arm=arm, detail_available=int(detail is not None),
                                     connectivity=row["connectivity"], nonrelay_complete=int(row["nonrelay_complete"]),
                                     search_status=search_task[0] if search_task else None,
                                     search_targets_found=search_task[1] if search_task else None,
                                     safety_active=json.dumps(row["safety_active"]),
                                     active_search_assignments=json.dumps(detail["search"]) if detail else None,
                                     active_relay_anchors=json.dumps(detail["relay"]) if detail else None,
                                     control_step=detail["control_step"] if detail else None,
                                     relay_demand_slots=json.dumps([x["slots"] for x in event["demands"]]) if event else None,
                                     relay_legal_pairs=sum(c["role"]==2 for c in event["candidates"]) if event else None,
                                     comm_hex=detail["comm_hex"] if detail else None,
                                     observed_xy_heading_velocity=json.dumps(detail["observed"]) if detail else None,
                                     control_commands=json.dumps(detail["commands"]) if detail else None,
                                     probe=json.dumps(detail["probe"]) if detail else None))


def summarize(root, out):
    runs = {arm: Run.read(root/arm, arm) for arm in ARMS}
    official = {arm: load_json(ARCHIVE/"evaluation"/CASE/f"{arm}.json.gz") for arm in ("R1", "P1")}
    for arm, expected in {"R1": FROZEN_SHA, "P1":TRAINED_SHA}.items():
        if official[arm]["checkpoint_sha256"] != expected or official[arm]["compatibility_policy"]:
            raise ValueError(f"{arm} official asset mismatch")
    checks = {arm: prefix_matches(runs[arm].summary, official[ref])
              for arm, ref in (("R1","R1"), ("J0","P1"), ("P1","P1"), ("J0_SWAP_S3","P1"))}
    shared = ("initial_sha256", "config_sha256", "radio_range", "scenario_sha256", "source_head",
              "driver_sha256", "native_sha256", "marvel_sha256", "torch_version")
    for key in shared:
        if len({run.summary[key] for run in runs.values()}) != 1:
            raise ValueError(f"Unpaired audit in {key}")
    if joint_completion_metric(runs["J0"].summary["trace"], 300.) != runs["J0"].summary["score"]:
        raise ValueError("J0 H300 scoring mismatch")
    pairs = {}
    for left, right in (("J0","P1"), ("R1","J0"), ("J0","J0_SWAP_S3"), ("R1","J0_SWAP_S3")):
        aa, bb = runs[left], runs[right]
        pairs[f"{left}/{right}"] = dict(ticks=first_change(aa.ticks, bb.ticks, TICK_FIELDS),
                                      events=first_change(aa.events, bb.events, EVENT_FIELDS))
    swap = runs["J0_SWAP_S3"].events[SWAP_STEP]
    if swap["selected"] != list(SWAP_ACTION) or swap["selected"] == swap["normal_choice"]:
        raise ValueError("Intervention did not change the earmarked real decision")
    times = {}
    for arm, run in runs.items():
        score_ref = official.get(arm, run.summary)
        times[arm] = window_times(score_ref["trace"])
        times[arm]["official_h300"] = score_ref["score"]
        times[arm]["search_done"] = next((row["time"] for row in run.ticks.values()
                                         if row["tasks"]["T2_target_search"][0] == "complete"), None)
        times[arm]["threshold_merge"] = threshold_merge(run.ticks, times[arm]["final_joint_window_start"])
        times[arm]["helper_motion"] = helper_motion_until_radio(run, times[arm]["final_joint_window_start"])
    # A same-state counterfactual is evaluated on every action-bearing J0 frame,
    # including any decisions beyond the prefix of the instrumented P1 replay.
    j0_rows = [e for e in runs["J0"].events.values() if "actors" in e]
    same_graph_changes = [e["step"] for e in j0_rows if e["actors"]["J0"]["pick"] != e["actors"]["P1"]["pick"]]
    archived_offers = {e["step"]: e for e in official["P1"]["offers"]}
    matched_offers = []
    for e in j0_rows:
        offer = archived_offers.get(e["step"], {})
        search = {c["uav"] for c in e["candidates"] if c["role"] == 1}
        relay = {c["uav"] for c in e["candidates"] if c["role"] == 2}
        if (offer.get("search") == sum(c["role"] == 1 for c in e["candidates"])
                and offer.get("relay") == sum(c["role"] == 2 for c in e["candidates"])
                and offer.get("shared_uavs") == len(search & relay)):
            matched_offers.append(e["step"])
    paired_j0_p1 = pairs["J0/P1"]
    replay_ok = all(checks[k]["scoring_trace"] and checks[k]["stable_commitments"] for k in ("R1", "P1"))
    observed_policy_change = first_policy_fork(runs["J0"].events, runs["P1"].events,
                                             paired_j0_p1["ticks"])
    full_score_same = all(checks["J0"][k] for k in ("scoring_trace","stable_commitments"))
    if (replay_ok and full_score_same and len(matched_offers) == len(j0_rows) == len(archived_offers)
            and not paired_j0_p1["ticks"] and not paired_j0_p1["events"] and not same_graph_changes):
        verdict = "POLICY_EFFECT_NOT_OBSERVED"
    elif replay_ok and observed_policy_change is not None:
        verdict = "POLICY_EFFECT_CONFIRMED"
    else:
        verdict = "INCONCLUSIVE"
    out.mkdir(exist_ok=True, parents=True)
    build_timeline(out/"aligned_ticks.csv.gz", runs, official)
    write(out/"learning_signal.json", learning_data())
    write(out/"action_differences.json", {
        arm:[event_summary(e, run.summary["radio_range"]) for e in run.events.values()
             if e.get("selected") is not None or 120<=e["step"]<=140 or 200<=e["step"]<=300]
        for arm,run in runs.items()})
    write(out/"comparison.json", dict(verdict=verdict, arms=times, prefix_validations=checks,
          first_differences=pairs, common_graph_events=len(j0_rows), same_graph_greedy_differences=same_graph_changes,
          matched_archived_p1_offer_steps=matched_offers,
          first_j0_p1_same_input_policy_fork=observed_policy_change,
          swap_against_r1_archive=prefix_matches(runs["J0_SWAP_S3"].summary, official["R1"]),
          swap_matches_r1_all_detailed_ticks=runs["J0_SWAP_S3"].ticks == runs["R1"].ticks,
          retarget_evidence=retarget_evidence(runs["R1"], runs["J0"]),
          detailed_ticks={k: run.summary["ticks"] for k,run in runs.items()},
          metadata={k: runs["J0"].summary[k] for k in shared}))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    summarize(args.runs, args.out)


if __name__ == "__main__":
    main()
