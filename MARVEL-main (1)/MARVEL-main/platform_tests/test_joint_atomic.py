from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from integrations.gppo.adapter import GPPOAssignment
from integrations.gppo.heading_adapter import ExtendedGPPOInferenceAdapter
from integrations.gppo.heading_features import HEADING_PROTOCOL
from integrations.gppo.joint_atomic import (
    JointCandidatePlanner, JointRejected, JointSnapshot, JointTaskScheduler,
)
from integrations.gppo.joint_shadow_recorder import rng_digest
from integrations.gppo.task_graph import TaskType
from integrations.rule_scheduling import choose_rule_assignment
from platform_tests.test_joint_shadow import live_fixture, state_fixture

ROOT = Path(__file__).resolve().parents[1]


def plan_state(state):
    return JointCandidatePlanner().enumerate(JointSnapshot.from_public(state))


def prepare_controller():
    scheduler, stale = live_fixture()
    scheduler.__class__ = JointTaskScheduler
    scheduler.joint_mode = "joint_rule"
    scheduler.joint_planner = JointCandidatePlanner()
    scheduler.tracker = SimpleNamespace(positions=lambda fraction: stale)
    scheduler._base_actions = list(state_fixture()["base_actions"].values())
    scheduler._joint_revision = 0
    scheduler.last_joint_receipt = None
    scheduler._joint_lock = RLock()
    scheduler._action_scope = None
    return scheduler, stale


def selected(plan, role=TaskType.RELAY):
    edge = next(edge for edge in plan.candidates.values() if edge.slot.task_type == role)
    ti = plan.graph.subtask_id_to_index[edge.slot.slot_id]
    ui = plan.graph.uav_id_to_index[edge.uav_id]
    return plan.propose(GPPOAssignment(plan.graph.flatten_action(ti, ui), ti, ui,
                                      edge.slot.slot_id, edge.uav_id, 0.))


def commit(scheduler, plan, proposal, stale, compile_commands=lambda: None):
    with ExitStack() as scope:
        scheduler._action_scope = scope
        try:
            scheduler._commit(plan, proposal, stale)
            compile_commands()
        finally:
            scheduler._action_scope = None


def test_mixed_rows_explicit_conflicts_stability_and_purity():
    state = state_fixture()
    snapshot = JointSnapshot.from_public(state)
    rng = rng_digest()
    a = JointCandidatePlanner().enumerate(snapshot)
    b = JointCandidatePlanner().enumerate(snapshot)
    assert JointSnapshot.from_public(state) == snapshot
    assert rng_digest() == rng
    assert list(a.candidates) == list(b.candidates)
    assert a.candidates == b.candidates and a.rejected == b.rejected
    np.testing.assert_array_equal(a.graph.action_mask, b.graph.action_mask)
    search, relay = (next(c for c in a.candidates.values()
                          if c.slot.task_type == role and c.uav_id == 2)
                     for role in (TaskType.TARGET_SEARCH, TaskType.RELAY))
    assert search.conflicts_with(relay)
    assert relay.conflicts_with(replace(relay, uav_id=77))
    same_anchor = replace(relay, slot=replace(relay.slot, slot_id=90), uav_id=77)
    assert relay.conflicts_with(same_anchor)


@pytest.mark.parametrize("disable", ["target_search", "relay", "both"])
def test_inactive_roles_are_not_candidates(disable):
    state = state_fixture()
    for task in state["tasks"]:
        if disable in (task["task_type"], "both"):
            task["status"] = "complete"
    plan = plan_state(state)
    roles = {edge.slot.task_type for edge in plan.candidates.values()}
    expected = {"target_search": {TaskType.RELAY}, "relay": {TaskType.TARGET_SEARCH}, "both": set()}
    assert roles == expected[disable]
    assert plan.graph.valid_action_pairs == list(plan.candidates)
    if not roles:
        assert plan.propose(None) is None


def test_single_candidate_and_mandatory_safety():
    state = state_fixture()
    state["controls"]["search_activated"] = False
    assert len(plan_state(state).candidates) == 1
    state["safety_ids"].add(2)
    assert not plan_state(state).candidates


def test_existing_search_capacity_unreachable_deadline_and_warning():
    state = state_fixture()
    state["controls"]["max_search_uavs"] = 1
    state["controls"]["search_assignments"] = {99: 0}
    assert all(edge.slot.task_type == TaskType.RELAY for edge in plan_state(state).candidates.values())
    state = state_fixture()
    state["max_steps"] = 11
    state["sensors"]["explorer"]["range"] = 1.
    assert "last_deadline_masks" in plan_state(state).rejected[(100000, 0)]
    state["max_steps"] = 3000
    for y in range(32):
        state["recovery"]["cells"][20, y] = 1
    assert "observed_closed_cut_outside_public_detection_support" in plan_state(state).rejected[(100000, 0)]
    state = state_fixture()
    state["hazards"] = [dict(obstacle_id="warning", position=np.array([10., 15.]), spawn_step=1,
                            initial_radius=0., expansion_rate=.02, max_radius=4.)]
    assert all(edge.uav_id != 2 for edge in plan_state(state).candidates.values())


@pytest.mark.parametrize("role", [TaskType.TARGET_SEARCH, TaskType.RELAY])
def test_commit_records_one_owner_and_rejects_duplicate(role):
    s, stale = prepare_controller()
    plan = s.joint_planner.enumerate(s._snapshot())
    proposal = selected(plan, role)
    commit(s, plan, proposal, stale)
    assert s.last_joint_receipt.assignment.uav_id == proposal.uav_id
    assert s.last_joint_receipt.assignment.task_type == role
    s._check_ownership()
    with pytest.raises(JointRejected, match="event_already_committed"):
        commit(s, plan, proposal, stale)
    s.runtime.current_step += 10
    s._last_observed_step += 10
    next_plan = s.joint_planner.enumerate(s._snapshot())
    assert all(edge.uav_id != proposal.uav_id for edge in next_plan.candidates.values())
    next_choice = choose_rule_assignment(next_plan.graph)
    commit(s, next_plan, next_plan.propose(next_choice), stale)
    assert s.last_joint_receipt.step == 20
    s._check_ownership()


@pytest.mark.parametrize("change", ["step", "uav", "task", "safety", "observation"])
def test_stale_live_state_rejects_before_writes(change):
    s, stale = prepare_controller()
    plan = s.joint_planner.enumerate(s._snapshot())
    proposal = selected(plan)
    if change == "step":
        s.runtime.current_step += 1
    elif change == "task":
        s.runtime.tasks.tasks["relay"].status = "complete"
    elif change == "safety":
        s.safety.active_uav_ids.add(proposal.uav_id)
    elif change == "observation":
        s._search_recovery.cells[0, 0] = 1
    else:
        s._robot_by_id(proposal.uav_id).available = False
    before = s._snapshot()
    with pytest.raises(JointRejected, match="stale_snapshot"):
        commit(s, plan, proposal, stale)
    assert s._snapshot() == before
    assert s.last_joint_receipt is None


def test_forged_anchor_and_inconsistent_action_mapping_rejected():
    s, stale = prepare_controller()
    plan = s.joint_planner.enumerate(s._snapshot())
    proposal = selected(plan)
    with pytest.raises(JointRejected, match="invalid_proposal"):
        commit(s, plan, replace(proposal, anchor=(0., 0.)), stale)
    choice = choose_rule_assignment(plan.graph)
    with pytest.raises(JointRejected, match="masked_or_inconsistent_action"):
        plan.propose(replace(choice, uav_id=-1))
    assert s.last_joint_receipt is None


@pytest.mark.parametrize("stage", ["helper_write", "command_generation", "probe"])
def test_partial_commit_failure_restores_tasks_anchors_uavs_and_receipt(monkeypatch, stage):
    s, stale = prepare_controller()
    plan = s.joint_planner.enumerate(s._snapshot())
    proposal = selected(plan)
    before = s._snapshot()
    apply = s._apply_uav_state

    def fail(*args):
        raise RuntimeError("injected")

    def fail_after_robot(*args):
        apply(*args)
        assert s.relay_assignments
        fail()

    if stage == "helper_write":
        monkeypatch.setattr(s, "_apply_uav_state", fail_after_robot)
    if stage == "probe":
        monkeypatch.setattr(s, "_refresh_observed_relay_probe", fail)
    with pytest.raises(RuntimeError, match="injected"):
        commit(s, plan, proposal, stale, fail if stage == "command_generation" else lambda: None)
    assert s._snapshot() == before
    assert s.last_joint_receipt is None and not s.last_event_assignments


def test_rule_and_frozen_gppo_consume_same_nonempty_mask():
    manifest = json.loads((ROOT / "handoff/cloud/checkpoint_manifest.json").read_text())
    base, candidate = (next(x for x in manifest if x["name"] == name)
                       for name in ("Frozen native GPPO", "November 11 guarded candidate"))
    adapter = ExtendedGPPOInferenceAdapter(ROOT / base["cloud_path"], ROOT / candidate["cloud_path"],
                                           expected_sha256=candidate["cloud_sha256"])
    assert adapter.model_protocol == HEADING_PROTOCOL
    plan = plan_state(state_fixture())
    adapter.observe({r["robot_id"]: {"heading": r["heading"]} for r in state_fixture()["robots"]})
    torch.set_num_threads(1)
    first = choose_rule_assignment(plan.graph)
    assert first == choose_rule_assignment(deepcopy(plan.graph))
    gppo = adapter.assign(plan.graph)
    assert plan.propose(first) and plan.propose(gppo)
    np.testing.assert_array_equal(adapter.last_batch["action_mask"][0], plan.graph.action_mask)
    assert adapter.last_batch["edge_features"].shape[-1] == 6
    plan.graph.action_mask[:] = True
    assert choose_rule_assignment(plan.graph) is None
    assert adapter.assign(plan.graph) is None


def test_changed_relay_anchor_invalidates_pending_proposal():
    state = state_fixture()
    # An intervening reservation changes the public resource version.
    base = plan_state(state)
    anchor = next(edge.slot.position for edge in base.candidates.values()
                  if edge.slot.task_type == TaskType.RELAY)
    proposal = selected(base)
    s, stale = prepare_controller()
    pending = s.joint_planner.enumerate(s._snapshot())
    s.relay_assignments[99] = (0, np.asarray(anchor))
    with pytest.raises(JointRejected, match="stale_snapshot"):
        commit(s, pending, proposal, stale)
    assert np.array_equal(s.relay_assignments[99][1], anchor)
    assert not s.last_event_assignments


def test_inactive_search_releases_only_its_commitment():
    s, stale = prepare_controller()
    s.search_assignments[0] = 0
    s._apply_uav_state(0, "target_search", 5)
    s.runtime.tasks.tasks["target_search"].status = "complete"
    plan = s.joint_planner.enumerate(s._snapshot())
    proposal = selected(plan)
    commit(s, plan, proposal, stale, s._reconcile_uav_state)
    assert not s.search_assignments
    assert s._robot_by_id(0).available
    assert set(s.relay_assignments) == {proposal.slot_id - 200000}


def test_finalizer_error_restores_adapter_motion_state(monkeypatch):
    s, stale = prepare_controller()
    base_actions = s._base_actions
    plan = s.joint_planner.enumerate(s._snapshot())
    proposal = selected(plan)
    evidence = deepcopy(s._search_recovery)
    motion_context = SimpleNamespace(_motion_roles={}, _observed_belief=SimpleNamespace(evidence=evidence))

    def compile_commands(self, actions):
        self._commit(plan, proposal, stale)
        return actions

    def finalize(commands):
        evidence.prefixes[proposal.uav_id] = [np.array([1., 2.])]
        motion_context._motion_roles[proposal.uav_id] = ("relay", None)
        raise RuntimeError("adapter command failure")

    from integrations.gppo.scheduler import GPPOTaskScheduler
    monkeypatch.setattr(GPPOTaskScheduler, "apply", compile_commands)
    with pytest.raises(RuntimeError, match="adapter command failure"):
        s.apply_with_finalizer(base_actions, finalize, motion_context=motion_context)
    assert not s.relay_assignments and s.last_joint_receipt is None
    assert not evidence.prefixes and not motion_context._motion_roles
