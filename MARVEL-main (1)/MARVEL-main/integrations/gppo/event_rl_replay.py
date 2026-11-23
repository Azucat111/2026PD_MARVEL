"""Replay records for the public, variable-size joint decision SMDP."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .public_pair_preference import event_features

SCHEMA = "joint-public-event-replay-v1"
FEATURES = "joint-public-event-candidates-v1"
DT = .1
HORIZON_TICKS = 3000
REWARD_SCALE_SECONDS = 30.


@dataclass(frozen=True)
class Decision:
    step: int
    snapshot: str
    public_input_sha256: str | None
    candidates: tuple
    features: tuple

    @classmethod
    def from_event(cls, event):
        mask = np.asarray(event["mask"], dtype=bool)
        task_ids, uav_ids, roles = (event[k] for k in ("task_ids", "uav_ids", "roles"))
        if mask.shape != (len(task_ids), len(uav_ids)) or len(roles) != len(task_ids):
            raise ValueError("Public graph shape does not match the mask")
        legal = np.flatnonzero(~mask.flatten())
        rows = event["rows"]
        if not len(legal) or list(legal) != [row["flat"] for row in rows]:
            raise ValueError("Recorded features do not cover exactly the legal mask")
        stable = []
        for flat, row in zip(legal, rows, strict=True):
            ti, ui = divmod(int(flat), len(uav_ids))
            action = (task_ids[ti], uav_ids[ui], roles[ti])
            if action != (row["slot"], row["uav"], row["role"]) or action[2] not in (1, 2):
                raise ValueError("Candidate ID disagrees with the graph")
            stable.append(action)
        context, action_features = event_features(event, opportunity=True)
        # Smooth physical-time and fleet-size scales. The old actor contributes
        # a public ranking feature, with no teacher targets or initialized weights.
        context[0] = math.log1p(event["time_seconds"] / 10.)
        context[5] = len(uav_ids) / 60.
        context[:5] = np.minimum(context[:5], 8.) / 2.
        logits = np.array([r["logit"] for r in rows], dtype=float)
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite public logits")
        logits = np.arcsinh((logits - logits.max()) / 8.)
        features = np.column_stack((np.tile(context, (len(rows), 1)), action_features, logits))
        return cls(event["step"], event["snapshot_version"], event.get("input_sha256"),
                   tuple(stable), tuple(tuple(map(float, row)) for row in features))

    @classmethod
    def unpack(cls, data):
        return cls(data["step"], data["snapshot"], data["public_input_sha256"],
                   tuple(map(tuple, data["candidates"])), tuple(map(tuple, data["features"])))


@dataclass(frozen=True)
class Transition:
    state: Decision
    action: int
    reward: float
    dt: float
    next_state: Decision | None
    done: bool
    truncated: bool
    scenario_id: str
    seed: int
    policy_id: str
    group: str
    behavior_probability: float | None
    behavior_distribution: tuple | None

    @property
    def executed(self):
        return self.state.candidates[self.action]


def joint_cost_trace(trace):
    """Return official success, first provisional success, and charged time.

    Provisional completion stops the running clock. A later whole-window
    failure restores the remainder as a terminal correction.
    """
    since = first = None
    safe = True
    for step, row in enumerate(trace, 1):
        t = row["time"]
        ratio = row["connectivity"]
        if not math.isclose(t, step * DT, abs_tol=1e-8) or not 0 <= ratio <= 1:
            raise ValueError("Replay requires the ordered physical clock")
        safe &= row["collisions"] == row["no_fly_violations"] == 0
        resolved = not row["safety_active"] and row["escapes"] == row["safety_records"]
        if row["nonrelay_complete"] and ratio >= .9 and resolved:
            if since is None:
                since = t
            if first is None and t - since >= 3. - 1e-8:
                first = t
        else:
            since = None
    if len(trace) != HORIZON_TICKS:
        return None, first, None
    success = safe and resolved and first is not None
    return success, first, first if success else HORIZON_TICKS * DT


def transitions_from_events(summary, events, *, policy_id, group):
    trace = summary["trace"]
    success, provisional, full_cost = joint_cost_trace(trace)
    terminal = summary["status"] == "complete"
    if terminal and (full_cost is None or summary["score"]["joint_success"] != success or
                     not math.isclose(full_cost, summary["score"]["secondary_joint_completion_seconds"])):
        raise ValueError("Replay cost disagrees with the official H300 score")
    decisions = []
    commitments = summary["commitments"]
    if terminal and len(commitments) != HORIZON_TICKS // 10:
        raise ValueError("Terminal replay requires the complete controller ledger")
    seen_steps = []
    case = summary["case"]
    for row in events:
        step = row["step"]
        if step % 10 or (seen_steps and step <= seen_steps[-1]):
            raise ValueError("Invalid joint controller event clock")
        seen_steps.append(step)
        if row.get("selected") is None:
            if row.get("receipt") is not None or (step // 10 < len(commitments) and commitments[step // 10]):
                raise ValueError("Missing recorded selection")
            continue
        receipt = row["receipt"]
        if receipt is None or step // 10 >= len(commitments) or commitments[step // 10] != [receipt]:
            raise ValueError("Recorded action does not match the atomic ledger")
        state = Decision.from_event(row)
        executed = (receipt["slot_id"], receipt["uav_id"], receipt["task_type"])
        if list(executed[:2]) != row["selected"]:
            raise ValueError("Selected and executed stable IDs differ")
        index = state.candidates.index(executed)
        behavior = row.get("behavior_distribution")
        probability = row.get("executed_action_probability")
        if behavior is not None:
            if (len(behavior) != len(state.candidates) or any(not 0 <= val <= 1 for val in behavior)
                    or not math.isclose(sum(behavior), 1., abs_tol=1e-7) or probability is None
                    or not math.isclose(behavior[index], probability, abs_tol=1e-7)):
                raise ValueError("Inconsistent recorded behavior distribution")
        if probability is not None and not 0 < probability <= 1:
            raise ValueError("Invalid observed action probability")
        decisions.append((state, index, probability, None if behavior is None else tuple(behavior)))
    actual_steps = [i * 10 for i, batch in enumerate(commitments) if batch]
    if [d[0].step for d in decisions] != actual_steps:
        raise ValueError("A recorded atomic commitment is missing from the event stream")
    if not decisions:
        return [], None
    end = len(trace)
    cost_cap = end * DT if provisional is None else provisional
    prefix = -min(cost_cap, decisions[0][0].step * DT) / REWARD_SCALE_SECONDS
    transitions = []
    for i, (state, action, probability, behavior) in enumerate(decisions):
        following = decisions[i + 1][0] if i + 1 < len(decisions) else None
        stop_step = following.step if following else end
        done = terminal and following is None
        elapsed = (stop_step - state.step) * DT
        if elapsed <= 0:
            raise ValueError("An action has no realized physical interval")
        spent = min(cost_cap, stop_step * DT) - min(cost_cap, state.step * DT)
        if done:
            spent += full_cost - cost_cap
        transitions.append(Transition(state, action, -spent / REWARD_SCALE_SECONDS, elapsed, following,
                                      done, not terminal and following is None, case, int(case.rsplit("_", 1)[1]),
                                      policy_id, group, probability, behavior))
    if terminal and not math.isclose(prefix + sum(t.reward for t in transitions),
                                     -full_cost / REWARD_SCALE_SECONDS, abs_tol=1e-8):
        raise ValueError("Replay and prefix rewards do not sum to official cost")
    return transitions, prefix


def read_run(directory):
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
    else:
        with gzip.open(directory / "summary.json.gz", "rt") as handle:
            summary = json.load(handle)
    with gzip.open(directory / "events.jsonl.gz", "rt") as handle:
        events = [json.loads(line) for line in handle]
    with ZipFile(directory / "public.zip") as archive:
        for event in events:
            if event.get("selected") is not None:
                raw = archive.read(f"{event['step']:04d}.json")
                if hashlib.sha256(raw).hexdigest() != event["snapshot_version"].rsplit(":", 1)[-1]:
                    raise ValueError("Public snapshot hash does not match the event")
    return summary, events


def save_replay(path: Path, episodes, audit):
    """Archive versioned replay with original episode boundaries and metadata."""
    payload = dict(schema=SCHEMA, features=FEATURES, reward_scale_seconds=REWARD_SCALE_SECONDS,
                   episodes=[dict(name=name, transitions=[asdict(t) for t in rows]) for name, rows in episodes],
                   audit=audit)
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    with path.open("xb") as handle:
        handle.write(gzip.compress(data, mtime=0))


def load_replay(path: Path):
    with gzip.open(path, "rt") as handle:
        data = json.load(handle)
    if (data["schema"], data["features"], data["reward_scale_seconds"]) != (SCHEMA, FEATURES, REWARD_SCALE_SECONDS):
        raise ValueError("Unsupported event replay version")
    episodes = []
    for episode in data["episodes"]:
        rows = []
        for row in episode["transitions"]:
            row["state"] = Decision.unpack(row["state"])
            if row["next_state"] is not None:
                row["next_state"] = Decision.unpack(row["next_state"])
            if row["behavior_distribution"] is not None:
                row["behavior_distribution"] = tuple(row["behavior_distribution"])
            rows.append(Transition(**row))
        episodes.append((episode["name"], rows))
    return episodes, data["audit"]
