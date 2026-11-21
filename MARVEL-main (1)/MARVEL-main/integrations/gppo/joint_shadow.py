"""Read-only, extended-only joint Search/Relay admission and first-step audit.

No production entry point imports this module. A public allowlist crosses the
boundary; all mutating scheduler/mask/motion methods run on disposable objects.
An admitted edge is NOT a path/completion/safety guarantee. The stronger prefix
certificate covers one physics tick, including a conservative peer speed ball.
"""
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace as NS

import numpy as np

from .event_allocator import GPPOEventAllocator
from .event_sources import PublicHeatPoint, SearchSlotBuilder
from .mesh_relay import ObservedMeshRelayDemandBuilder, ObservedMeshRelayReleaseGate
from .observed_graph import observed_next_hop
from .runtime_graph import RuntimeGraphBuilder
from .safety_layer import SharedHazardAdapter
from .scheduler import GPPOTaskScheduler
from .search_recovery import ObservedSearchRecovery
from .search_viewpoints import ObservedSearchViewpoints
from .task_graph import TaskType
from utils.dynamics_models import KinematicSimple
from utils.geometry import GeometryFrame
from utils.sensor_models import IdealSensor

SCHEMA = 'marvel-joint-shadow-public-v1'
ALLOWED = 'existing_scheduler'
ORDER = 'execution_supported_order_blocked'
CONTRACT = 'requires_execution_contract'
ILLEGAL = 'constraint_rejected'

# Every field originates in own sensor labels/feedback, never an obstacle map.
RECOVERY_FIELDS = (
    'cells', 'prefixes', 'prefix_headings', 'anchors', 'failed',
    'last_feedback_step', 'assignment_tokens', 'sensing_uids',
    'nearby_positions', 'observed_positions', '_authoritative_areal_grid',
)
SCHEDULER_FIELDS = (
    'max_search_uavs', 'max_relay_uavs', 'search_service_steps',
    'search_reach_radius', 'search_activated', 'search_activation_step',
    'search_activation_threshold', 'relay_trigger_threshold', '_last_observed_step',
    'heat_points', 'serviced_heat_ids', 'search_assignments', 'relay_assignments',
    'search_dwell_by_heat', '_search_scan_bearings', '_search_scan_phases',
    '_search_scan_owners', '_peripheral_search_targets',
)


def pack(value):
    """Lossless JSON for public arrays and typed keys; no pickle/class loading."""
    if isinstance(value, np.ndarray):
        return {'@array': value.tolist(), 'dtype': value.dtype.str,
                'shape': list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {'@dict': [[pack(k), pack(v)] for k, v in value.items()]}
    if isinstance(value, (set, frozenset)):
        return {'@frozenset' if isinstance(value, frozenset) else '@set':
                [pack(v) for v in sorted(value)]}
    if isinstance(value, tuple):
        return {'@tuple': [pack(v) for v in value]}
    if isinstance(value, list):
        return [pack(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Not a public JSON value: {type(value).__name__}')


def unpack(value):
    if isinstance(value, list):
        return [unpack(v) for v in value]
    if not isinstance(value, dict):
        return value
    if '@array' in value:
        return np.array(value['@array'], dtype=value['dtype']).reshape(value['shape'])
    if '@dict' in value:
        return {unpack(k): unpack(v) for k, v in value['@dict']}
    if '@set' in value:
        return set(unpack(v) for v in value['@set'])
    if '@frozenset' in value:
        return frozenset(unpack(v) for v in value['@frozenset'])
    if '@tuple' in value:
        return tuple(unpack(v) for v in value['@tuple'])
    raise ValueError('Unknown public snapshot tag')


def snapshot_bytes(state):
    return json.dumps(pack(state), sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def snapshot_digest(state):
    return hashlib.sha256(snapshot_bytes(state)).hexdigest()


def capture_public_state(scheduler, stale_positions, *, scenario):
    """Capture at _refresh_search entry, AFTER completion/release/reconcile.

    Do not deepcopy runtime/scheduler/agents: they link to hidden truth/model/RNG.
    Dynamics/sensor classes fail closed outside the audited extended contract.
    """
    rt = scheduler.runtime
    rec = scheduler._search_recovery
    if rt.geometry_mode != 'extended' or type(rec) is not ObservedSearchRecovery:
        raise ValueError('Only audited extended ObservedSearchRecovery is supported')
    robots, dynamics, sensors = [], {}, {}
    for r in rt.robots:
        uid, kind = int(r.robot_id), r.robot_type
        if uid not in stale_positions:
            raise ValueError('Missing public delayed fleet position')
        model = rt.dynamics_by_type.get(kind, rt.dynamics)
        sensor = rt.sensor_by_type.get(kind)
        if type(model) is not KinematicSimple or type(sensor) is not IdealSensor:
            raise ValueError('Uncertified dynamics/sensor contract')
        dynamics[kind] = dict(velocity=model.velocity, yaw_rate=model.yaw_rate)
        sensors[kind] = dict(range=sensor.sensor_range, fov=sensor.fov,
                             los_occlusion=sensor.los_occlusion)
        # Own pose comes only from an observation; never use robot.position.
        own = rec.observed_positions.get(uid)
        robots.append(dict(robot_id=uid, robot_type=kind,
            position=np.asarray(stale_positions[uid] if own is None else own).copy(),
            heading=float(r.heading), velocity=float(r.velocity),
            available=bool(getattr(r, 'available', True)),
            busy_until=float(getattr(r, 'busy_until', 0.)),
            utilization=float(getattr(r, 'utilization', 0.)),
            assigned_task_num=int(getattr(r, 'assigned_task_num', 0)),
            current_task=str(getattr(r, 'current_task', 'exploration'))))
    tasks = [dict(task_id=t.task_id, task_type=t.task_type, status=t.status,
                  priority=t.priority, assigned_robot_types=list(t.assigned_robot_types))
             for t in rt.tasks.tasks.values()]
    agents = {}
    for uid, a in scheduler._marvel_agents.items():
        agents[int(uid)] = {k: deepcopy(getattr(a, k, None)) for k in
                           ('node_coords', 'adjacent_matrix', 'current_index', 'neighbor_indices')}
    # SharedHazardAdapter already treats these declared warning circles as public.
    hazards = [{k: deepcopy(getattr(o, k)) for k in
                ('obstacle_id', 'position', 'spawn_step', 'initial_radius',
                 'expansion_rate', 'max_radius')}
               for o in rt.obstacles.dynamic_obstacles]
    controls = {k: deepcopy(getattr(scheduler, k, {} if k.startswith('_search_scan')
                                             or k == '_peripheral_search_targets' else None))
                for k in SCHEDULER_FIELDS}
    controls['heat_points'] = [asdict(p) for p in scheduler.heat_points]
    gate = scheduler.relay_release_gate
    views = scheduler._search_viewpoints
    state = dict(schema=SCHEMA, scenario=str(scenario), step=int(rt.current_step),
        dt=float(rt.dt), max_steps=int(rt.max_steps),
        profile=asdict(scheduler.profile),
        gppo_profile=deepcopy(rt.config.get('_gppo_profile', {})),
        robots=robots, tasks=tasks, dynamics=dynamics, sensors=sensors,
        frame=asdict(rec.frame),
        recovery={k: deepcopy(getattr(rec, k)) for k in RECOVERY_FIELDS},
        agents=agents, hazards=hazards,
        safety_ids=set(scheduler.safety.active_uav_ids),
        warning_margin=scheduler.safety.hazards.warning_margin,
        safe_margin=scheduler.safety.hazards.safe_margin,
        controls=controls, stale_positions=deepcopy(stale_positions),
        base_position=deepcopy(scheduler.base_position),
        relay_probe=deepcopy(getattr(scheduler, '_relay_probe', None)),
        relay_builder=dict(comm_range=scheduler.relay_builder.comm_range,
                           max_hops=scheduler.relay_builder.max_hops,
                           max_demands=scheduler.relay_builder.max_demands),
        release={k: deepcopy(getattr(gate, k)) for k in
                 ('release_threshold', 'stable_release_steps', 'stable_counter',
                  'release_count', 'last_counterfactual_connectivity',
                  'last_counterfactual_target_ratio', 'min_connectivity', 'fleet_uav_ids')},
        views=None if views is None else dict(sensor_range=views.sensor_range,
            fov=views.fov, prior_radius=views.prior_radius, max_candidates=views.max_candidates,
            checked_cells_by_heat=deepcopy(views.checked_cells_by_heat)))
    return state


def _sandbox(state):
    if state['schema'] != SCHEMA:
        raise ValueError('Snapshot schema mismatch')
    state = deepcopy(state)
    frame = GeometryFrame(**state['frame'])
    s = object.__new__(GPPOTaskScheduler)
    rt = NS(geometry_mode='extended', current_step=state['step'], dt=state['dt'],
            max_steps=state['max_steps'], config={'_gppo_profile': state['gppo_profile']},
            robots=[NS(**r) for r in state['robots']],
            obstacles=NS(frame=frame, dynamic_obstacles=[NS(**o) for o in state['hazards']]))
    rt.tasks = NS(tasks={t['task_id']: NS(**t) for t in state['tasks']})
    rt._log_event = lambda *args, **kwargs: None  # local diagnostics only
    rt.dynamics_by_type = {k: KinematicSimple(v) for k, v in state['dynamics'].items()}
    rt.dynamics = next(iter(rt.dynamics_by_type.values()))
    rt.sensor_by_type = {k: IdealSensor(v, frame=frame) for k, v in state['sensors'].items()}
    s.runtime = rt
    rt.high_level_scheduler = s
    s.profile = NS(**state['profile'])
    for k, v in state['controls'].items():
        setattr(s, k, v)
    s.heat_points = [PublicHeatPoint(**p) for p in s.heat_points]
    s.heat_by_id = {int(p.heat_id): p for p in s.heat_points}
    s.base_position = state['base_position']
    s._relay_probe = state['relay_probe']
    s._search_recovery = ObservedSearchRecovery(frame)
    for k, v in state['recovery'].items():
        setattr(s._search_recovery, k, v)
    s._marvel_agents = {uid: NS(**a) for uid, a in state['agents'].items()}
    s.safety = NS(active_uav_ids=state['safety_ids'], hazards=SharedHazardAdapter(rt,
                  warning_margin=state['warning_margin'], safe_margin=state['safe_margin']))
    s.search_builder = SearchSlotBuilder(s.max_search_uavs, s.search_service_steps)
    s.relay_builder = ObservedMeshRelayDemandBuilder(**state['relay_builder'])
    s.relay_builder.anchor_candidates_provider = s._observed_relay_anchor_candidates
    s.relay_builder.motion_candidate_guard = s._observed_relay_motion_impossible
    release = state['release']
    s.relay_release_gate = ObservedMeshRelayReleaseGate(
        min_connectivity=release['min_connectivity'], fleet_uav_ids=release['fleet_uav_ids'],
        release_threshold=release['release_threshold'], stable_release_steps=release['stable_release_steps'])
    for k, v in release.items():
        setattr(s.relay_release_gate, k, v)
    s._search_viewpoints = None
    if state['views'] is not None:
        v = state['views']
        s._search_viewpoints = ObservedSearchViewpoints(frame, **{
            k: v[k] for k in ('sensor_range', 'fov', 'prior_radius', 'max_candidates')})
        s._search_viewpoints.checked_cells_by_heat = v['checked_cells_by_heat']
    # Probe exploration is a separate rule action, not a Search/Relay assignment.
    # Existing probe reservations are retained. Do not create a new probe here.
    s._refresh_observed_relay_probe = lambda positions: None
    s.last_event_assignments = []
    a = object.__new__(GPPOEventAllocator)
    a.runtime, a._marvel_agents = rt, s._marvel_agents
    s.allocator = a
    return s


def graph_record(graph):
    return dict(time=graph.current_time,
        uav_ids=[int(u.uav_id) for u in graph.uav_states],
        task_ids=[int(t.subtask_id) for t in graph.subtasks],
        roles=[TaskType(t.task_type).name for t in graph.subtasks],
        mask=graph.action_mask.tolist(), edge_features=graph.edge_features.tolist(),
        uav_features=graph.uav_features.tolist(), task_features=graph.task_features.tolist(),
        precedence=graph.precedence_matrix.tolist())


def _reason_rows(allocator, tid, uid):
    reasons = []
    for field in ('last_deadline_masks', 'last_observed_reachability_masks', 'last_route_support_masks'):
        for row in getattr(allocator, field, []):
            if int(row.get('task_id', -1)) == tid and int(row.get('uav_id', row.get('deferred_uav_id', -1))) == uid:
                reasons.append(dict(stage=field, **deepcopy(row)))
    return reasons


def _prefix(s, slot, uid, state):
    """Dry-run existing motion interface; certify one step, not entire task."""
    # Motion selectors mutate recovery/viewpoint caches: isolate each candidate.
    s = copy(s)
    s._search_recovery = deepcopy(s._search_recovery)
    s._search_viewpoints = deepcopy(s._search_viewpoints)
    s._peripheral_search_targets = deepcopy(s._peripheral_search_targets)
    s.search_assignments = dict(s.search_assignments)
    s._search_scan_bearings = dict(s._search_scan_bearings)
    s._search_scan_phases = dict(s._search_scan_phases)
    s._search_scan_owners = dict(s._search_scan_owners)
    rec = s._search_recovery
    own = rec.observed_positions.get(uid)
    if own is None:
        return dict(certified=False, reason='missing_own_sensor_pose')
    r = copy(next(r for r in s.runtime.robots if r.robot_id == uid))
    r.position = np.asarray(own, dtype=float).copy()
    anchor = np.asarray(slot.position, dtype=float)
    agent = s._marvel_agents.get(uid)
    route = observed_next_hop(agent, anchor)
    base = (r.position.copy(), float(r.heading))
    if slot.task_type == TaskType.RELAY:
        target, heading = s._relay_action(r, anchor, agent, route, base)
    else:
        s.search_assignments[int(slot.source_entity_id)] = uid
        if np.linalg.norm(r.position-anchor) <= s.search_reach_radius + 1e-9:
            heading = s._service_scan_heading(int(slot.source_entity_id), r, anchor)
            target, heading = s._search_action(r, anchor, agent, None, base, service_scan_heading=heading)
        else:
            target, heading = s._search_action(r, anchor, agent, route, base,
                route_goal_radius=s.search_reach_radius, allow_observed_frontier=True)
    model = s.runtime.dynamics_by_type[r.robot_type]
    end = model.position_after(r.position, target, s.runtime.dt)
    moving = np.linalg.norm(end-r.position) > 1e-9
    maintaining = slot.task_type == TaskType.RELAY and np.linalg.norm(r.position-anchor) <= .1
    static_ok = rec.segment_safe(r.position, end)
    hz, step = s.safety.hazards, state['step']
    hazard_ok = (not hz.segment_intersects_warning_buffer(r.position, end, step=step)
                 and not hz.segment_intersects_no_fly(r.position, end, margin=.2, step=step+1))
    # Public delayed peers may have moved age*v and may move another dt*v.
    age = (max(0, step-s._last_observed_step)*s.runtime.dt
           + s.profile.stale_age_fraction*s.profile.high_level_interval_steps*s.runtime.dt)
    peer_ok = True
    delta = end-r.position
    length2 = float(np.dot(delta, delta))
    for other in s.runtime.robots:
        if other.robot_id == uid:
            continue
        peer = np.asarray(state['stale_positions'][other.robot_id])
        v = s.runtime.dynamics_by_type[other.robot_type].velocity
        t = float(np.clip(np.dot(peer-r.position, delta)/length2, 0., 1.)) if length2 > 1e-20 else 0.
        if np.linalg.norm(r.position+t*delta-peer) < .5+v*(age+s.runtime.dt)+1e-9:
            peer_ok = False
            break
    ok = bool((moving or maintaining) and static_ok and hazard_ok and peer_ok)
    return dict(certified=ok, reason='one_tick_public_certificate' if ok else 'prefix_not_certified',
        waypoint=np.asarray(target).tolist(), heading=float(heading), next_position=end.tolist(),
        moved=bool(moving), maintaining_relay=bool(maintaining),
        observed_five_probe_clear=bool(static_ok), public_hazard_clear=bool(hazard_ok),
        conservative_peer_ball_clear=bool(peer_ok), peer_age_seconds=float(age),
        horizon_seconds=s.runtime.dt, future_completion_guaranteed=False,
        base_action_assumption='own_pose_hold; no unobserved MARVEL fallback invented')


def enumerate_joint(state):
    """Both roles at one pre-reservation boundary; no inference, RNG or clock."""
    s = _sandbox(state)
    stale = state['stale_positions']
    graphs, rows = [], []
    original_allocate = s.allocator.allocate
    current = {}

    def inspect(graph, deterministic=True):
        graphs.append(graph_record(graph))
        for ti, task in enumerate(graph.subtasks):
            slot = current['slots'][int(task.subtask_id)]
            for ui, uav in enumerate(graph.uav_states):
                uid, tid = int(uav.uav_id), int(task.subtask_id)
                reasons = _reason_rows(s.allocator, tid, uid)
                if not uav.capability_for(task.task_type):
                    reasons.append(dict(stage='capability', reason='unsupported_role'))
                if not uav.alive or not uav.available or uav.busy_until > graph.current_time:
                    reasons.append(dict(stage='availability', reason='owned_or_busy'))
                if uid in slot.forbidden_uav_ids:
                    reasons.append(dict(stage='reservation_or_endpoint', reason='forbidden_uav_id'))
                masked = bool(graph.action_mask[ti, ui])
                if masked and not reasons:
                    reasons.append(dict(stage='candidate_guard', reason='relay_radio_or_observed_motion_guard'))
                legal = not masked
                certificate = _prefix(s, slot, uid, state) if legal else dict(certified=False, reason='admission_masked')
                rows.append(dict(slot_id=tid, uav_id=uid, role=TaskType(task.task_type).name,
                    source_task_id=slot.source_task_id, source_entity_id=slot.source_entity_id,
                    position=list(slot.position), admission_legal=legal,
                    legal_reasons=['trigger_open', 'capacity_open', 'all_existing_masks_pass'] if legal else [],
                    rejection_reasons=reasons,
                    immediately_executable=bool(legal and certificate['certified']), prefix=certificate,
                    category=ALLOWED if legal else ILLEGAL,
                    prerequisites=['one_uav_one_assignment', 'role_capacity', 'task_lifecycle',
                                   'Safety_preempts', 'per_tick_physics_and_Shield_authority'],
                    total_time_estimate=float(graph.edge_features[ti, ui, 2])))
        return None  # never select/reserve; examine first action of both roles

    def allocate(slots, **kwargs):
        slots = list(slots)
        current['slots'] = {int(slot.slot_id): slot for slot in slots}
        return original_allocate(slots, **kwargs)

    s.allocator.adapter = NS(assign=inspect)
    s.allocator.allocate = allocate
    s._refresh_search(stale)
    s._refresh_relay(stale)

    # Dormant catalog: explicit contract prerequisites, NEVER promoted to legal.
    # Diagnose release/arrival/capacity gates without clearing real reservations.
    diagnostics = []
    search_task = s._active_task('target_search')
    search_gate = []
    if not s.search_activated: search_gate.append('search_activation_not_reached')
    if search_task is None: search_gate.append('search_task_not_active')
    if len(s.search_assignments) >= s.max_search_uavs: search_gate.append('search_capacity_full')
    pending = s.search_builder.build([
        PublicHeatPoint(**{**asdict(p), 'serviced': p.heat_id in s.serviced_heat_ids})
        for p in s.heat_points], active_heat_ids=s.search_assignments,
        include_all_pending=True, source_task_id=None if search_task is None else search_task.task_id)
    if search_gate:
        diagnostics.extend(dict(role='TARGET_SEARCH', slot_id=p.slot_id,
            category=CONTRACT, admission_legal=False, immediately_executable=False,
            prerequisites=search_gate, position=list(p.position),
            reason='dormant_catalog_not_a_legal_action') for p in pending)
    relay_task = s._active_task('relay')
    original_owned = state['controls']['relay_assignments']
    snapshot = s.relay_builder.snapshot(stale)
    relay_gate = []
    if relay_task is None or s.base_position is None: relay_gate.append('relay_task_or_base_missing')
    if snapshot.connectivity_ratio >= s.relay_trigger_threshold: relay_gate.append('relay_trigger_not_reached')
    if original_owned and not any(row['role'] == 'RELAY' for row in rows):
        relay_gate.append('existing_relay_event_release_arrival_or_budget_gate')
    if relay_gate:
        result = s.relay_builder.build(stale, source_task_id=None if relay_task is None else relay_task.task_id)
        diagnostics.extend(dict(role='RELAY', slot_id=p.slot_id, category=CONTRACT,
            admission_legal=False, immediately_executable=False,
            prerequisites=relay_gate, position=list(p.position),
            reason='dormant_catalog_not_a_legal_action') for p in result.slots
            if p.source_entity_id not in original_owned)
    roles = {r['role'] for r in rows if r['immediately_executable']}
    legal_roles = {r['role'] for r in rows if r['admission_legal']}
    joint_uids = sorted({r['uav_id'] for r in rows if r['immediately_executable'] and r['role'] == 'RELAY'} &
                       {r['uav_id'] for r in rows if r['immediately_executable'] and r['role'] == 'TARGET_SEARCH'})
    return dict(schema=SCHEMA, step=state['step'], time=state['step']*state['dt'],
        uavs=len(state['robots']), scenario=state['scenario'], public_state_sha256=snapshot_digest(state),
        boundary='after_prerequisite_release_reconcile_before_search_refresh',
        potential_graphs=graphs, candidates=rows, dormant_catalog=diagnostics,
        rule_fixed=dict(role_order=['TARGET_SEARCH', 'RELAY'],
            search_capacity=s.max_search_uavs, relay_initial_capacity=s.max_relay_uavs,
            relay_trigger_threshold=s.relay_trigger_threshold,
            anchor_choice='existing_observed_mesh_builder',
            release='existing_counterfactual_stability_gate',
            safety='mandatory_preemption', probe='separate_rule_action_excluded'),
        metrics=dict(legal_candidates=sum(r['admission_legal'] for r in rows),
            executable_candidates=sum(r['immediately_executable'] for r in rows),
            potential_both_roles=len(legal_roles) == 2,
            executable_both_roles=len(roles) == 2,
            cross_role_shared_uavs=joint_uids,
            # Different role/task IDs define distinct high-level transitions.
            two_meaningful_executable_actions=len({(r['role'], r['slot_id']) for r in rows
                                                    if r['immediately_executable']}) >= 2,
            needs_contract_catalog=bool(diagnostics)),
        scope='one-step public evidence; no hidden truth, no performance claim')


def attach_actual(event, actual_graphs, assignments):
    """Compare same-boundary potential to observed original sequential calls."""
    visible = set()
    for g in actual_graphs:
        for ti, tid in enumerate(g['task_ids']):
            for ui, uid in enumerate(g['uav_ids']):
                if not g['mask'][ti][ui]:
                    visible.add((tid, uid))
    for row in event['candidates']:
        row['visible_in_actual_graph'] = (row['slot_id'], row['uav_id']) in visible
        if row['admission_legal'] and not row['visible_in_actual_graph']:
            row['category'] = ORDER
            row['actual_exclusion'] = 'sequential_choices_capacity_or_reservation; compare_actual_graphs'
    event['actual_graphs'] = actual_graphs
    event['actual_assignments'] = [asdict(a) for a in assignments]
    event['metrics']['order_excluded_candidates'] = sum(r['category'] == ORDER for r in event['candidates'])
    event['metrics']['order_excluded_executable_candidates'] = sum(
        r['category'] == ORDER and r['immediately_executable'] for r in event['candidates'])
    return event


def summarize(events):
    """Every primary count uses ALL recorded control boundaries as denominator."""
    result = dict(events=len(events), no_candidate_events=0, single_candidate_events=0,
        same_role_multiple_events=0, potential_cross_role_events=0,
        executable_cross_role_events=0, shared_uav_cross_role_events=0,
        order_excluded_events=0, order_excluded_executable_events=0,
        needs_contract_events=0, actual_mixed_graphs=0, actual_graphs=0)
    for e in events:
        m = e['metrics']; n = m['legal_candidates']
        result['no_candidate_events'] += n == 0
        result['single_candidate_events'] += n == 1
        result['same_role_multiple_events'] += n > 1 and not m['potential_both_roles']
        result['potential_cross_role_events'] += m['potential_both_roles']
        result['executable_cross_role_events'] += m['executable_both_roles']
        result['shared_uav_cross_role_events'] += bool(m['cross_role_shared_uavs'])
        result['order_excluded_events'] += m.get('order_excluded_candidates', 0) > 0
        result['order_excluded_executable_events'] += m.get('order_excluded_executable_candidates', 0) > 0
        result['needs_contract_events'] += m['needs_contract_catalog']
        for g in e.get('actual_graphs', []):
            roles = {role for role, mask in zip(g['roles'], g['mask']) if not all(mask)}
            result['actual_graphs'] += 1
            result['actual_mixed_graphs'] += len(roles) > 1
    result['denominator'] = 'events: all pre-search control boundaries, including no allocation'
    result['actual_graph_denominator'] = result['actual_graphs']
    return result
