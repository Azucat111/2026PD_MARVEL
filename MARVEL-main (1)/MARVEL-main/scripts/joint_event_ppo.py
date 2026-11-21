"""On-policy Joint GPPO collection and guarded updates using the real runtime."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from verify_joint_atomic import setup, write
from gppo_joint_completion_metric import joint_completion_metric
from gppo_rollout_batch import FrozenBehaviorBatch, model_state_fingerprint
from joint_four_arm_eval import HORIZON_SECONDS, PHYSICAL_TICKS, physical_sample, require_marvel
from integrations.gppo.adapter import GPPOAssignment
from integrations.gppo.heading_features import HEADING_PROTOCOL, heading_batch
from train_extended_gppo_batched import clipped_policy_loss

REWARD_PROTOCOL = "joint-h300-undiscounted-time-v1"
KL_LIMITS = dict(sampled_kl=.01, mean_exact_kl=.01, max_state_exact_kl=.05)


def configure_torch():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def action_distribution(model, batch, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid policy temperature")
    output = model(**batch)
    legal = ~batch["action_mask"].flatten()
    logits = output["flat_logits"].flatten()[legal] / temperature
    if not legal.any() or not torch.isfinite(logits).all() or not torch.isfinite(output["value"]).all():
        raise ValueError("Expected legal actions and finite actor/critic outputs")
    return torch.distributions.Categorical(logits=logits), output["value"].reshape(()), legal


class SegmentBoundary(Exception):
    """The next policy event supplies value without creating a new action."""


class LiveCollector:
    def __init__(self, scheduler, *, seed, temperature, archive, segment_ticks=None):
        self.scheduler = scheduler
        self.inference = scheduler.allocator.adapter
        self.generator = torch.Generator().manual_seed(seed)
        self.temperature = temperature
        self.archive = archive
        self.segment_ticks = segment_ticks
        self.events = []
        self.pending = None
        self.bootstrap = None
        scheduler._choose_joint = self.choose
        planner = scheduler.joint_planner
        original = planner.enumerate

        def inspect(snapshot):
            self.snapshot = snapshot
            return original(snapshot)

        planner.enumerate = inspect

    @torch.no_grad()
    def choose(self, graph):
        batch = heading_batch(graph, self.inference.heading_observations, self.inference.device)
        dist, value, legal = action_distribution(self.inference.model, batch, self.temperature)
        step = self.scheduler.runtime.current_step
        if self.segment_ticks is not None and step >= self.segment_ticks and self.events:
            self.bootstrap = dict(step=step, value=float(value), batch=batch,
                                  snapshot_version=self.snapshot.version)
            self.archive.writestr("bootstrap.json", self.snapshot.public_bytes)
            raise SegmentBoundary

        legal_indices = legal.nonzero().flatten()
        choice = int(torch.multinomial(dist.probs, 1, generator=self.generator))
        flat = int(legal_indices[choice])
        ti, ui = divmod(flat, graph.num_uavs)
        task = graph.subtasks[ti]
        uav = graph.uav_states[ui]
        snapshot_file = f"{step:04d}.json"
        self.archive.writestr(snapshot_file, self.snapshot.public_bytes)
        self.pending = dict(step=step, snapshot_version=self.snapshot.version, public_snapshot=snapshot_file,
                            task_ids=[int(t.subtask_id) for t in graph.subtasks],
                            roles=[int(t.task_type) for t in graph.subtasks],
                            uav_ids=[int(u.uav_id) for u in graph.uav_states],
                            batch={k: v.detach().cpu().clone() for k, v in batch.items()},
                            selected_flat=flat, selected_legal_index=choice,
                            chosen_role=int(task.task_type), old_logp=float(dist.logits[choice]),
                            old_log_probs=dist.logits.detach().cpu().clone(), old_value=float(value))
        return GPPOAssignment(flat, ti, ui, int(task.subtask_id), int(uav.uav_id), float(value))

    def confirm(self, receipt):
        if self.pending is None:
            if receipt.assignment is not None:
                raise RuntimeError("Assignment without a sampled policy event")
            return
        row = self.pending
        a = receipt.assignment
        ti, ui = divmod(row["selected_flat"], len(row["uav_ids"]))
        if (a is None or receipt.step != row["step"] or receipt.version != row["snapshot_version"]
                or (a.flat_action, a.slot_id, a.uav_id, int(a.task_type)) !=
                (row["selected_flat"], row["task_ids"][ti], row["uav_ids"][ui], row["roles"][ti])):
            raise RuntimeError("Sample/atomic receipt mismatch")
        row["receipt"] = asdict(receipt)
        row["receipt"]["assignment"]["task_type"] = int(a.task_type)
        self.events.append(row)
        self.pending = None


def assign_returns(events, *, end_step, dt, cost_seconds, bootstrap=0., terminal=True):
    """Physical-time cost ledger, gamma=lambda=1, optional real next-event bootstrap."""
    steps = [r["step"] for r in events] + [end_step]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Policy events must be ordered on the physical clock")
    if not 0 <= cost_seconds <= end_step * dt + 1e-8:
        raise ValueError("Time cost outside observed window")
    running = bootstrap
    for i in reversed(range(len(events))):
        a, b = steps[i:i+2]
        reward = -(min(cost_seconds, b * dt) - min(cost_seconds, a * dt)) / HORIZON_SECONDS
        running += reward
        events[i].update(next_step=b, duration_seconds=(b-a)*dt, reward=reward,
                         terminal=terminal and i == len(events)-1, truncated=not terminal and i == len(events)-1,
                         return_target=running, advantage=running-events[i]["old_value"])
    return -min(cost_seconds, steps[0]*dt)/HORIZON_SECONDS


def summarize_events(events):
    counts = Counter()
    for row in events:
        n = len(row["uav_ids"])
        legal = (~row["batch"]["action_mask"].flatten()).nonzero().flatten().tolist()
        search = {row["uav_ids"][i % n] for i in legal if row["roles"][i//n] == 1}
        relay = {row["uav_ids"][i % n] for i in legal if row["roles"][i//n] == 2}
        counts["mixed_events"] += bool(search and relay)
        counts["shared_uav_events"] += bool(search & relay)
        counts["search" if row["chosen_role"] == 1 else "relay"] += 1
    counts["role_switches"] = sum(a["chosen_role"] != b["chosen_role"] for a,b in zip(events,events[1:]))
    return dict(counts)


def collect_episode(case, path, *, model=None, policy_seed, temperature, wall_seconds, segment_ticks=None):
    start = time.monotonic()
    path.mkdir(parents=True, exist_ok=False)
    rt, adapter, obs = setup(case, "joint_gppo", p1_model=model)
    scheduler = adapter.scheduler
    inference = scheduler.allocator.adapter
    behavior = model_state_fingerprint(inference.model)
    marvel = model_state_fingerprint(adapter.policy_net)
    trace = []
    score = error = None
    status = "truncated"
    forced_events = 0
    with ZipFile(path / "public_observations.zip", "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
        collector = LiveCollector(scheduler, seed=policy_seed, temperature=temperature, archive=archive,
                                  segment_ticks=segment_ticks)
        try:
            with require_marvel():
                while rt.current_step < PHYSICAL_TICKS:
                    if time.monotonic()-start >= wall_seconds:
                        raise TimeoutError("Training rollout exceeded wall limit")
                    commands = adapter.get_actions(obs)
                    receipt = scheduler.last_joint_receipt
                    collector.confirm(receipt)
                    forced_events += receipt.assignment is None
                    for _ in range(rt._mission_step_interval()):
                        obs, info = rt.step(commands)
                        trace.append(physical_sample(rt, info))
                    if rt.current_step % 100 == 0:
                        print(f"{case} ticks={rt.current_step} events={len(collector.events)} "
                              f"elapsed={time.monotonic()-start:.1f}s", flush=True)
                if len(trace) != PHYSICAL_TICKS:
                    raise ValueError("Incomplete physical window")
                score = joint_completion_metric(trace, HORIZON_SECONDS)
                status = "complete"
        except SegmentBoundary:
            # Prefix diagnostic uses the same running cost. The full-window
            # terminal correction is not available at a bootstrapped boundary.
            if not any(r["nonrelay_complete"] for r in trace):
                status = "segment"
            else:
                error = "Prefix reached potential completion before the bootstrap boundary"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    events = collector.events
    prefix = None
    if status in ("complete", "segment") and events:
        prefix = assign_returns(events, end_step=rt.current_step, dt=rt.dt,
                                cost_seconds=score["secondary_joint_completion_seconds"] if score else rt.current_step*rt.dt,
                                terminal=status == "complete",
                                bootstrap=0. if score else collector.bootstrap["value"])
    if model_state_fingerprint(inference.model) != behavior or model_state_fingerprint(adapter.policy_net) != marvel:
        raise RuntimeError("A frozen rollout model changed")
    if collector.pending is not None:
        collector.pending["invalid_reason"] = error or "commit_not_confirmed"
    for row in events:
        row.update(case=case, episode_id=path.name, environment_seed=int(case.rsplit("_", 1)[-1]),
                   policy_seed=policy_seed)
    summary = dict(case=case, episode_id=path.name, status=status, error=error,
                   episode_terminated=status == "complete", episode_truncated=status != "complete",
                   eligible=status in ("complete", "segment") and bool(events),
                   behavior_fingerprint=behavior, behavior_checkpoint_sha256=inference.candidate_sha256,
                   frozen_marvel_fingerprint=marvel, policy_seed=policy_seed, temperature=temperature,
                   ticks=len(trace), score=score, seconds=time.monotonic()-start, action_events=len(events),
                   forced_events=forced_events, safety_triggered=len(scheduler.safety.records),
                   completion_tick=next((i+1 for i,r in enumerate(trace) if r["nonrelay_complete"]),None),
                   collisions=sum(r["collisions"] for r in trace), rejected=collector.pending is not None,
                   prefix_reward=prefix, reward_protocol=REWARD_PROTOCOL, **summarize_events(events))
    torch.save(dict(summary=summary, events=events, bootstrap=collector.bootstrap,
                    rejected=collector.pending), path / "rollout.pt")
    write(path / "trace.json", trace)
    write(path / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def verify_replay(model, row, temperature):
    dist, value, legal = action_distribution(model, row["batch"], temperature)
    indices = legal.nonzero().flatten()
    choice = row["selected_legal_index"]
    if int(indices[choice]) != row["selected_flat"]:
        raise ValueError("Stored legal mask or flat mapping differs")
    ti, ui = divmod(row["selected_flat"], len(row["uav_ids"]))
    receipt = row["receipt"]["assignment"]
    if (receipt["slot_id"], receipt["uav_id"], int(receipt["task_type"])) != (
            row["task_ids"][ti], row["uav_ids"][ui], row["roles"][ti]):
        raise ValueError("Stored stable action IDs differ")
    torch.testing.assert_close(dist.logits, row["old_log_probs"], atol=2e-6, rtol=0)
    if abs(float(dist.logits[choice])-row["old_logp"]) > 2e-6 or abs(float(value)-row["old_value"]) > 2e-6:
        raise ValueError("Behavior logp/value cannot be reproduced")


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(map(finite_tree, value.values()))
    if isinstance(value, (list, tuple)):
        return all(map(finite_tree, value))
    return not isinstance(value, float) or math.isfinite(value)


@torch.no_grad()
def kl_metrics(model, rows, temperature):
    sample, exact = [], []
    for row in rows:
        dist, _, _ = action_distribution(model, row["batch"], temperature)
        old = row["old_log_probs"].double()
        diff = dist.logits.double() - old
        d = diff[row["selected_legal_index"]]
        sample.append(float(torch.expm1(d)-d))
        exact.append(max(0., float((-old.exp()*diff).sum())))
    return dict(sampled_kl=sum(sample)/len(sample), mean_exact_kl=sum(exact)/len(exact),
                max_state_exact_kl=max(exact))


def guarded_epoch(model, optimizer, rows, temperature, *, limits=KL_LIMITS):
    # Same transactional boundary as the guarded Phase 1 implementation:
    # the post-update KL is always computed against the frozen collection policy.
    saved_model = deepcopy(model.state_dict())
    saved_adam = deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.
    metrics = {}
    try:
        for row in rows:
            dist, value, _ = action_distribution(model, row["batch"], temperature)
            new = dist.logits[row["selected_legal_index"]]
            actor = clipped_policy_loss(new, torch.as_tensor(row["old_logp"]),
                                        torch.as_tensor(row["advantage"]), .2)
            critic = .5*(value-row["return_target"]).square()
            loss = actor + .5*critic - .002*dist.entropy()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO objective")
            (loss/len(rows)).backward()
            loss_sum += float(loss.detach())/len(rows)
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        if not math.isfinite(grad):
            raise FloatingPointError("Nonfinite PPO gradient")
        optimizer.step()
        if not finite_tree(model.state_dict()) or not finite_tree(optimizer.state_dict()):
            raise FloatingPointError("Nonfinite model or Adam")
        metrics = kl_metrics(model, rows, temperature)
        if not all(math.isfinite(metrics[k]) and metrics[k] <= cap for k, cap in limits.items()):
            raise FloatingPointError("KL exceeded transaction bounds")
    except BaseException as exc:
        model.load_state_dict(saved_model)
        optimizer.load_state_dict(saved_adam)
        optimizer.zero_grad(set_to_none=True)
        if not isinstance(exc, FloatingPointError):
            raise
        return dict(accepted=False, reason=str(exc), **metrics)
    optimizer.zero_grad(set_to_none=True)
    return dict(accepted=True, loss=loss_sum, gradient_norm=grad, **metrics)

def verify_rollout_files(path, episode):
    summary = episode["summary"]
    events = episode["events"]
    trace = json.loads((path / "trace.json").read_text())
    if len(trace) != summary["ticks"]:
        raise ValueError("Physical trace length differs from the collection ledger")
    terminal = summary["status"] == "complete"
    bootstrap = episode["bootstrap"]
    if terminal:
        if len(trace) != PHYSICAL_TICKS or joint_completion_metric(trace, HORIZON_SECONDS) != summary["score"]:
            raise ValueError("H300 score differs from the physical trace")
        cost = summary["score"]["secondary_joint_completion_seconds"]
    else:
        if (summary["status"] != "segment" or bootstrap is None or bootstrap["step"] != len(trace)
                or summary["score"] is not None or any(t["nonrelay_complete"] for t in trace)):
            raise ValueError("Invalid bootstrapped prefix")
        cost = len(trace) * .1
    expected = [dict(e) for e in events]
    prefix = assign_returns(expected, end_step=len(trace), dt=.1, cost_seconds=cost,
                            bootstrap=0. if terminal else bootstrap["value"], terminal=terminal)
    if prefix != summary["prefix_reward"]:
        raise ValueError("Prefix reward differs")
    fields = ("reward", "return_target", "advantage", "next_step", "duration_seconds", "terminal", "truncated")
    if any(a[k] != b[k] for a, b in zip(events, expected) for k in fields):
        raise ValueError("Recorded event return is inconsistent")
    with ZipFile(path / "public_observations.zip") as archive:
        snapshots = [(e["public_snapshot"], e["snapshot_version"]) for e in events]
        if bootstrap is not None:
            snapshots.append(("bootstrap.json", bootstrap["snapshot_version"]))
        for name, version in snapshots:
            if hashlib.sha256(archive.read(name)).hexdigest() != version.rsplit(":", 1)[-1]:
                raise ValueError("Public snapshot version differs from recorded decision")


def update_rollouts(paths, output, *, model=None, lr=2e-5, epochs=2, allow_segment=False):
    if not 1 <= epochs <= 4 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid bounded PPO update parameters")
    data = [torch.load(p / "rollout.pt", weights_only=True, map_location="cpu") for p in paths]
    if len({(x["summary"]["case"], x["summary"]["policy_seed"]) for x in data}) != len(data):
        raise ValueError("Duplicate event episode supplied to one on-policy batch")
    _, adapter, _ = setup(data[0]["summary"]["case"], "joint_gppo", p1_model=model)
    inference = adapter.scheduler.allocator.adapter
    behavior = inference.model.eval()
    initial = model_state_fingerprint(behavior)
    ledger = FrozenBehaviorBatch(len(data))
    ledger.begin(behavior)
    temperatures = {row["summary"]["temperature"] for row in data}
    if len(temperatures) != 1:
        raise ValueError("Mixed behavior temperatures")
    temperature = temperatures.pop()
    for path, episode in zip(paths, data):
        s = episode["summary"]
        if not s["eligible"] or (s["status"] != "complete" and not allow_segment):
            raise ValueError(f"Episode is not eligible for this update: {s['case']}")
        if s["behavior_fingerprint"] != initial or s["behavior_checkpoint_sha256"] != inference.candidate_sha256:
            raise ValueError("Rollout behavior differs from the exact starting model")
        verify_rollout_files(path, episode)
        for event in episode["events"]:
            with torch.no_grad():
                verify_replay(behavior, event, temperature)
        if episode["bootstrap"] is not None:
            with torch.no_grad():
                _, value, _ = action_distribution(behavior, episode["bootstrap"]["batch"], temperature)
            if abs(float(value) - episode["bootstrap"]["value"]) > 2e-6:
                raise ValueError("Bootstrap value cannot be reproduced")
        ledger.add_episode(behavior, episode["events"], [r["reward"] for r in episode["events"]])

    optimizer = torch.optim.Adam(behavior.parameters(), lr=lr)
    reports = []
    for _ in range(epochs):
        report = guarded_epoch(behavior, optimizer, ledger.transitions, temperature)
        reports.append(report)
        if not report["accepted"]:
            break
    final = model_state_fingerprint(behavior)
    output.mkdir(parents=True, exist_ok=False)
    accepted = sum(r["accepted"] for r in reports)
    summary = dict(reward_protocol=REWARD_PROTOCOL, starting_checkpoint_sha256=inference.candidate_sha256,
                   before=initial, after=final, changed=initial!=final, epochs=reports,
                   samples=len(ledger.transitions), accepted_updates=accepted,
                   rejected_updates=len(reports)-accepted, optimizer_initialization="fresh Adam",
                   temperature=temperature, lr=lr, kl_limits=KL_LIMITS,
                   episodes=[x["summary"] for x in data])
    if final != initial and accepted:
        torch.save(dict(protocol_version=HEADING_PROTOCOL, reward_protocol=REWARD_PROTOCOL,
                        model=behavior.state_dict(), optimizer=optimizer.state_dict(),
                        initial_sha256=inference.candidate_sha256,
                        training_report=summary), output/"checkpoint.pth")
        summary["checkpoint_sha256"] = hashlib.sha256((output / "checkpoint.pth").read_bytes()).hexdigest()
    write(output/"update.json", summary)
    ledger.clear()
    print(json.dumps(summary), flush=True)
    return summary
