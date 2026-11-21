"""Single-action Search/Relay allocation for the extended runtime."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from threading import RLock

import numpy as np

from .event_allocator import GPPOEventAssignment, GPPOEventSlot
from .heading_features import HEADING_PROTOCOL
from .joint_shadow import (
    _collect_joint, _prefix, _reason_rows, capture_public_state, snapshot_bytes, unpack,
)
from .runtime_graph import RuntimeGraphBuilder
from .scheduler import GPPOTaskScheduler
from .task_graph import TaskGraph, TaskType
from integrations.rule_scheduling import choose_rule_assignment
from utils.safety_shield import SafetyShield

PROTOCOL = "extended-joint-single-action-v1"
GEOMETRY = "extended"
ANCHOR_SEPARATION = SafetyShield.UAV_SAFE_DISTANCE
_MOTION_FIELDS = ("prefixes", "prefix_headings", "anchors", "failed",
                  "assignment_tokens", "sensing_uids", "last_feedback_step")
_GATE_FIELDS = ("stable_counter", "release_count", "last_counterfactual_connectivity",
                "last_counterfactual_target_ratio")


@dataclass(frozen=True)
class JointSnapshot:
    step: int
    revision: int
    version: str
    public_bytes: bytes

    @classmethod
    def from_public(cls, state, revision=0):
        payload = snapshot_bytes(state)
        digest = hashlib.sha256(payload).hexdigest()
        return cls(state["step"], revision, f"{PROTOCOL}:{revision}:{digest}", payload)


@dataclass(frozen=True)
class Candidate:
    slot: GPPOEventSlot
    uav_id: int
    total_time: float
    waypoint: tuple[float, float]
    heading: float

    @property
    def key(self):
        return self.slot.slot_id, self.uav_id

    def conflicts_with(self, other):
        return (self.uav_id == other.uav_id
                or self.slot.slot_id == other.slot.slot_id
                or (self.slot.task_type == other.slot.task_type == TaskType.RELAY
                    and _anchor_conflict(self.slot.position, other.slot.position)))


@dataclass(frozen=True)
class Proposal:
    step: int
    version: str
    slot_id: int
    uav_id: int
    role: TaskType
    anchor: tuple[float, float]
    value: float = 0.0


@dataclass(frozen=True)
class Receipt:
    step: int
    version: str
    assignment: GPPOEventAssignment | None


@dataclass
class JointPlan:
    snapshot: JointSnapshot
    graph: TaskGraph
    candidates: dict[tuple[int, int], Candidate]
    rejected: dict[tuple[int, int], tuple[str, ...]]
    search_assignments: dict
    relay_assignments: dict
    relay_gate: dict
    relay_probe: dict | None
    retarget: bool
    refresh_probe: bool

    def propose(self, choice):
        """Translate the selector's graph indices to validated stable IDs."""
        if choice is None:
            if self.candidates:
                raise JointRejected("selector_abstained")
            return None
        try:
            index = self.graph.flatten_action(choice.task_index, choice.uav_index)
            key = self.graph.action_ids(index)
        except (IndexError, TypeError) as exc:
            raise JointRejected("invalid_action_index") from exc
        if (index != choice.flat_action or key != (choice.subtask_id, choice.uav_id)
                or key not in self.candidates or not math.isfinite(choice.value)):
            raise JointRejected("masked_or_inconsistent_action")
        edge = self.candidates[key]
        return Proposal(self.snapshot.step, self.snapshot.version, *key,
                        edge.slot.task_type, tuple(edge.slot.position), choice.value)


def _anchor_conflict(first, second):
    return np.linalg.norm(np.asarray(first) - second) < ANCHOR_SEPARATION


class JointCandidatePlanner:
    def enumerate(self, snapshot):
        """Pure enumeration. Reuse each role's allocator before forming mixed rows."""
        state = unpack(json.loads(snapshot.public_bytes))
        tasks, travel_rows = [], []
        uavs = None
        candidates, rejected = {}, {}

        def inspect(sandbox, graph, ti, slot):
            nonlocal uavs
            if uavs is not None and uavs != graph.uav_states:
                raise RuntimeError("Role graphs disagree on shared UAV state")
            uavs = graph.uav_states
            tasks.append(graph.subtasks[ti])
            travel_rows.append(graph.travel_time_matrix[ti])
            for ui, uav in enumerate(uavs):
                key = (slot.slot_id, uav.uav_id)
                if graph.action_mask[ti, ui]:
                    reasons = _reason_rows(sandbox.allocator, *key)
                    if reasons:
                        reason = tuple(row.get("reason", row["stage"]) for row in reasons)
                    elif not uav.capability_for(slot.task_type):
                        reason = ("role_capability",)
                    elif not uav.alive or not uav.available or uav.busy_until > graph.current_time:
                        reason = ("uav_unavailable",)
                    elif uav.uav_id in slot.forbidden_uav_ids:
                        reason = ("reserved_or_relay_endpoint",)
                    else:
                        reason = ("relay_admission_guard",)
                    rejected[key] = reason
                    continue
                # One shared resource check applies even if a role's slot builder
                # did not explicitly list the existing probe as forbidden.
                probe = state["relay_probe"]
                if uav.uav_id in state["safety_ids"] or (probe and probe["uid"] == uav.uav_id):
                    rejected[key] = ("safety_or_probe_owner",)
                    continue
                if slot.task_type == TaskType.RELAY and any(
                        _anchor_conflict(slot.position, anchor)
                        for _, anchor in sandbox.relay_assignments.values()):
                    rejected[key] = ("relay_anchor_reserved",)
                    continue
                certificate = _prefix(sandbox, slot, uav.uav_id, state)
                if not certificate["certified"] or not certificate.get("exact_executor_context"):
                    rejected[key] = (certificate["reason"], "requires_observed_first_tick")
                    continue
                target = tuple(certificate["waypoint"])
                own = state["recovery"]["observed_positions"][uav.uav_id]
                if sandbox.safety.hazards.segment_intersects_warning_buffer(
                        own, target, step=snapshot.step):
                    rejected[key] = ("waypoint_warning_buffer",)
                    continue
                total_time = float(graph.edge_features[ti, ui, 2])
                if not math.isfinite(total_time):
                    rejected[key] = ("nonfinite_time_estimate",)
                    continue
                candidates[key] = Candidate(slot, uav.uav_id, total_time, target,
                                            certificate["heading"])

        sandbox, _, _, maintenance = _collect_joint(
            state, defer_relay_maintenance=True, inspect_row=inspect)
        if uavs is None:
            uavs = RuntimeGraphBuilder(sandbox.runtime,
                position_overrides=state["stale_positions"]).build().graph.uav_states
        graph = TaskGraph(uavs, tasks, current_time=state["step"] * state["dt"],
                          travel_time_matrix=np.asarray(travel_rows) if tasks else None)
        for ti, task in enumerate(tasks):
            for ui, uav in enumerate(uavs):
                graph.action_mask[ti, ui] = (task.subtask_id, uav.uav_id) not in candidates
        return JointPlan(snapshot, graph, candidates, rejected, sandbox.search_assignments,
                         sandbox.relay_assignments,
                         {k: getattr(sandbox.relay_release_gate, k) for k in _GATE_FIELDS},
                         sandbox._relay_probe, maintenance["retarget"], maintenance["probe"])


class JointRejected(RuntimeError):
    """Stable rejection reason emitted before resource writes or after rollback."""


@contextmanager
def _rollback_on_error(scheduler, motion_context=None):
    """Save only the mutable surfaces touched between commitment and action return."""
    with ExitStack() as undo:
        def save(obj, names):
            for name in names:
                if not hasattr(obj, name):
                    undo.callback(lambda obj=obj, name=name: obj.__dict__.pop(name, None))
                    continue
                original = getattr(obj, name)
                value = deepcopy(original, {id(scheduler._search_recovery): scheduler._search_recovery})
                undo.callback(setattr, obj, name, value)

        save(scheduler, ("search_assignments", "relay_assignments", "_relay_probe",
                         "last_event_assignments", "assignments", "_joint_revision",
                         "last_joint_receipt", "_relay_frontier", "_relay_motion_routes",
                         "_relay_anchor_stamp", "_relay_anchor_points",
                         "_search_scan_bearings", "_search_scan_phases",
                         "_search_scan_owners", "_peripheral_search_targets"))
        save(scheduler.relay_release_gate, _GATE_FIELDS)
        save(scheduler.runtime, ("active_search_uav_ids",))
        if hasattr(scheduler.runtime, "events"):
            events = scheduler.runtime.events
            start = len(events)
            undo.callback(lambda: events.__delitem__(slice(start, None)))
        for robot in scheduler.runtime.robots:
            save(robot, ("available", "busy_until", "assigned_task_num", "current_task"))
        recovery = scheduler._search_recovery
        save(recovery, _MOTION_FIELDS)
        if motion_context is not None:
            save(motion_context, ("_motion_roles",))
            belief = motion_context._observed_belief
            if belief is not None and belief.evidence is not recovery:
                save(belief.evidence, _MOTION_FIELDS)
        safety = scheduler.safety
        save(safety, ("_sensing_pending", "_sensing_attempted", "_sensing_recovery_generation",
                      "_sensing_last_step", "_release_routes", "escape_waypoint_fallback_count"))
        yield
        undo.pop_all()


class JointTaskScheduler(GPPOTaskScheduler):
    """Feature-flagged controller sharing legacy sensing, motion and Safety."""
    def __init__(self, runtime, *args, scheduler_config=None, **kwargs):
        if runtime.geometry_mode != GEOMETRY:
            raise ValueError("Joint scheduling requires extended geometry")
        super().__init__(runtime, *args, scheduler_config=scheduler_config, **kwargs)
        self.joint_mode = self.config["mode"].lower()
        if self.joint_mode == "joint_gppo" and getattr(
                self.allocator.adapter, "model_protocol", None) != HEADING_PROTOCOL:
            raise ValueError(f"joint_gppo requires {HEADING_PROTOCOL} (6D edges)")
        self.joint_deterministic = self.config.get("joint_deterministic", True)
        if type(self.joint_deterministic) is not bool:
            raise ValueError("task_scheduler.joint_deterministic must be a boolean")
        self.joint_planner = JointCandidatePlanner()
        self._joint_revision = 0
        self.last_joint_receipt = None
        self._joint_lock = RLock()
        self._action_scope = self._base_actions = self._motion_context = None

    def _snapshot(self, stale=None):
        if stale is None:
            stale = self.tracker.positions(self.profile.stale_age_fraction)
        return JointSnapshot.from_public(capture_public_state(
            self, stale, scenario=PROTOCOL, base_actions=self._base_actions),
            self._joint_revision)

    def _choose_joint(self, graph):
        if self.joint_mode == "joint_rule":
            return choose_rule_assignment(graph)
        return self.allocator.adapter.assign(graph, deterministic=self.joint_deterministic)

    def _allocate_event(self, stale):
        if (self.last_joint_receipt is not None
                and self.last_joint_receipt.step == self.runtime.current_step):
            raise JointRejected("event_already_committed")
        plan = self.joint_planner.enumerate(self._snapshot(stale))
        choice = self._choose_joint(deepcopy(plan.graph)) if plan.candidates else None
        self._commit(plan, plan.propose(choice), stale)

    def _validate(self, plan, proposal):
        if self.last_joint_receipt is not None and self.last_joint_receipt.step == plan.snapshot.step:
            raise JointRejected("event_already_committed")
        if self._snapshot().version != plan.snapshot.version:
            raise JointRejected("stale_snapshot")
        if proposal is None:
            if plan.candidates:
                raise JointRejected("missing_action")
            return None
        if proposal.step != plan.snapshot.step or proposal.version != plan.snapshot.version:
            raise JointRejected("stale_proposal")
        edge = plan.candidates.get((proposal.slot_id, proposal.uav_id))
        if (edge is None or proposal.role != edge.slot.task_type
                or proposal.anchor != tuple(edge.slot.position) or not math.isfinite(proposal.value)):
            raise JointRejected("invalid_proposal")
        return edge

    def _commit(self, plan, proposal, stale):
        # The opaque plan is produced within this controller's synchronous call.
        # Validation hashes every planner input again, including live resources,
        # status, warning model, observation graph and the recorded base commands.
        edge = self._validate(plan, proposal)
        self._action_scope.enter_context(_rollback_on_error(
            self, getattr(self, "_motion_context", None)))
        self.search_assignments = dict(plan.search_assignments)
        self.relay_assignments = deepcopy(plan.relay_assignments)
        self._relay_probe = deepcopy(plan.relay_probe)
        for name, value in plan.relay_gate.items():
            setattr(self.relay_release_gate, name, value)

        assignment, locked_targets = None, ()
        if edge is not None:
            slot, uid = edge.slot, edge.uav_id
            ti = plan.graph.subtask_id_to_index[slot.slot_id]
            ui = plan.graph.uav_id_to_index[uid]
            assignment = GPPOEventAssignment(
                slot.slot_id, slot.task_type, uid, slot.source_task_id,
                slot.source_entity_id, plan.graph.flatten_action(ti, ui),
                proposal.value, slot.processing_time, edge.total_time)
            if slot.task_type == TaskType.TARGET_SEARCH:
                self.search_assignments[slot.source_entity_id] = uid
                role = "target_search"
            else:
                self.relay_assignments[slot.source_entity_id] = (uid, np.asarray(slot.position).copy())
                # The isolated certificate starts a new Relay route. Do not
                # reuse a still-valid route cached for a previous commitment.
                routes = getattr(self, "_relay_motion_routes", None)
                if routes is not None:
                    routes.routes.pop(uid, None)
                self.relay_release_gate.reset_stability()
                locked_targets = (slot.source_entity_id,)
                role = "relay"
            self._apply_uav_state(uid, role, edge.total_time)
            self.last_event_assignments.append(assignment)

        if plan.retarget:
            refreshed = dict(self.relay_assignments)
            self.relay_builder.refresh_assignments(refreshed, stale, fixed_targets=locked_targets)
            # Keep the certified new anchor and reject incompatible maintenance
            # moves as a group, preserving the builder's coupled radio forecast.
            if not any(_anchor_conflict(refreshed[target][1], anchor)
                       for target in locked_targets
                       for other, (_, anchor) in refreshed.items() if other != target):
                self.relay_assignments = refreshed
        if plan.refresh_probe:
            self._refresh_observed_relay_probe(stale)

        self._joint_revision += 1
        self.last_joint_receipt = Receipt(plan.snapshot.step, plan.snapshot.version, assignment)
        self._command_to_verify = edge
        self._check_ownership()

    def _check_ownership(self):
        owners = [(uid, "target_search") for uid in self.search_assignments.values()]
        owners.extend((uid, "relay") for uid, _ in self.relay_assignments.values())
        if self._relay_probe is not None:
            owners.append((self._relay_probe["uid"], "relay"))
        # Existing probes yield to mandatory Safety in the physical controller.
        if len({uid for uid, _ in owners}) != len(owners):
            raise JointRejected("duplicate_owner")
        for uid, role in owners:
            if uid in self.safety.active_uav_ids:
                if self._relay_probe is not None and uid == self._relay_probe["uid"]:
                    continue
                raise JointRejected("safety_owner_conflict")
            robot = self._robot_by_id(uid)
            if robot is None or robot.available or robot.current_task != role:
                raise JointRejected("resource_state_inconsistent")

    def apply(self, actions):
        return self.apply_with_finalizer(actions, lambda staged: staged)

    def apply_with_finalizer(self, actions, finalize, *, motion_context=None):
        with self._joint_lock, ExitStack() as transaction:
            if (self.last_joint_receipt is not None
                    and self.last_joint_receipt.step == self.runtime.current_step):
                raise JointRejected("event_already_committed")
            self._motion_context = motion_context
            self._base_actions = list(actions)
            self._action_scope = transaction
            self._command_to_verify = None
            try:
                result = finalize(super().apply(self._base_actions))
                edge = self._command_to_verify
                if edge is not None:
                    index = next(i for i, r in enumerate(self.runtime.robots)
                                 if r.robot_id == edge.uav_id)
                    waypoint, heading = result[index]
                    if (not np.allclose(waypoint, edge.waypoint, rtol=0, atol=1e-9)
                            or not math.isclose(heading, edge.heading, rel_tol=0, abs_tol=1e-9)):
                        raise JointRejected("command_differs_from_certificate")
                return result
            finally:
                self._action_scope = self._base_actions = self._command_to_verify = None
                self._motion_context = None
