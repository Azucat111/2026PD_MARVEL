"""Contract tests, including forbidden hidden reads and live allocator parity."""
from dataclasses import asdict
from types import SimpleNamespace as NS
from copy import deepcopy
import json
import pickle

import numpy as np
import pytest
import torch

from integrations.gppo.joint_shadow import (
    ALLOWED, CONTRACT, ILLEGAL, ORDER, _sandbox, attach_actual,
    capture_public_state, enumerate_joint, pack, unpack, snapshot_bytes, summarize,
)
from integrations.gppo.joint_shadow_recorder import (
    JointShadowRecorder, related_state_digest, rng_digest,
)
from integrations.gppo.protocol_profile import GPPORuntimeProfile
from integrations.rule_scheduling import choose_rule_assignment
from utils.geometry import GeometryFrame


def state_fixture():
    frame = GeometryFrame.extended(width_cells=64, height_cells=32)
    positions = {0: np.array([10., 10.]), 1: np.array([35., 10.]), 2: np.array([10., 15.])}
    profile = GPPORuntimeProfile(20., 2, 50., 1000., 300, 100., 10,
                                 300000., 3000, 1000., .05, ())
    coords = np.array([*positions.values(), [22.5, 10.5], [40., 15.]])
    agents = {uid: dict(node_coords=coords.copy(), adjacent_matrix=np.zeros((5, 5)),
                       current_index=uid, neighbor_indices=np.arange(5)) for uid in positions}
    return dict(schema='marvel-joint-shadow-public-v1', scenario='synthetic_contract_fixture',
        step=10, dt=.1, max_steps=3000, profile=asdict(profile),
        gppo_profile=dict(search_guaranteed_hidden_radius=5.5, search_cell_size=1.),
        robots=[dict(robot_id=uid, robot_type='explorer', position=p.copy(), heading=0.,
                     velocity=1., available=True, busy_until=0., utilization=0.,
                     assigned_task_num=0, current_task='exploration') for uid, p in positions.items()],
        tasks=[dict(task_id=kind, task_type=kind, status='active', priority=priority,
                    assigned_robot_types=['explorer']) for kind, priority in
               [('exploration', 3), ('target_search', 5), ('relay', 2)]],
        dynamics={'explorer': dict(velocity=1., yaw_rate=35.)},
        sensors={'explorer': dict(range=10., fov=120., los_occlusion=True)},
        frame=asdict(frame), recovery=dict(
            cells={(x, y): 0 for x in range(64) for y in range(32)},
            prefixes={}, prefix_headings={}, anchors={}, failed={}, last_feedback_step={},
            assignment_tokens={}, sensing_uids=set(), nearby_positions={uid: [] for uid in positions},
            observed_positions=deepcopy(positions), _authoritative_areal_grid=True),
        agents=agents, hazards=[], safety_ids=set(), warning_margin=5., safe_margin=8.,
        controls=dict(max_search_uavs=2, max_relay_uavs=2, search_service_steps=6,
            search_reach_radius=3.5, search_activated=True, search_activation_step=1,
            search_activation_threshold=.3, relay_trigger_threshold=.9, _last_observed_step=10,
            heat_points=[dict(heat_id=0, position=(40., 15.), confidence=1., priority=5., serviced=False)],
            serviced_heat_ids=set(), search_assignments={}, relay_assignments={}, search_dwell_by_heat={0: 0},
            _search_scan_bearings={}, _search_scan_phases={}, _search_scan_owners={}, _peripheral_search_targets={}),
        stale_positions=positions, base_position=np.zeros(2), relay_probe=None,
        relay_builder=dict(comm_range=20., max_hops=2, max_demands=2),
        release=dict(release_threshold=.95, stable_release_steps=3, stable_counter=0,
            release_count=0, last_counterfactual_connectivity=1., last_counterfactual_target_ratio=1.,
            min_connectivity=.9, fleet_uav_ids=frozenset(positions)), views=None)


def live_fixture():
    state = state_fixture()
    # frozenset is normalized to set in real capture; use exact public type here.
    state['release']['fleet_uav_ids'] = set(state['release']['fleet_uav_ids'])
    s = _sandbox(state)
    s.profile = GPPORuntimeProfile(**state['profile'])
    s.allocator.adapter = NS(assign=choose_rule_assignment)
    return s, state['stale_positions']


def test_repeated_enumeration_does_not_mutate_input_or_any_rng():
    state = state_fixture()
    state['release']['fleet_uav_ids'] = set(state['release']['fleet_uav_ids'])
    before, rng = snapshot_bytes(state), rng_digest()
    one, two = enumerate_joint(state), enumerate_joint(state)
    assert one == two
    assert snapshot_bytes(state) == before
    assert rng_digest() == rng
    assert one['metrics']['potential_both_roles']
    assert one['metrics']['executable_both_roles']
    assert one['metrics']['cross_role_shared_uavs'] == [2]


def test_exact_public_json_roundtrip_and_empty_array_shape():
    state = state_fixture()
    state['release']['fleet_uav_ids'] = set(state['release']['fleet_uav_ids'])
    replay = unpack(json.loads(json.dumps(pack(state))))
    assert snapshot_bytes(replay) == snapshot_bytes(state)
    assert enumerate_joint(replay) == enumerate_joint(state)
    assert unpack(pack(np.empty((0, 2)))).shape == (0, 2)


def test_safety_preemption_and_existing_owners_are_never_freed():
    state = state_fixture()
    state['release']['fleet_uav_ids'] = set(state['release']['fleet_uav_ids'])
    state['safety_ids'] = {2}
    for r in state['robots']:
        if r['robot_id'] == 2:
            r['available'] = False
            r['current_task'] = 'safety'
    event = enumerate_joint(state)
    assert not event['metrics']['executable_both_roles']
    assert all(not r['admission_legal'] for r in event['candidates'] if r['uav_id'] == 2)
    assert not state['robots'][2]['available']


def test_untriggered_search_and_capacity_are_not_legal_catalog_edges():
    state = state_fixture()
    state['release']['fleet_uav_ids'] = set(state['release']['fleet_uav_ids'])
    state['controls']['search_activated'] = False
    event = enumerate_joint(state)
    assert not any(r['role'] == 'TARGET_SEARCH' for r in event['candidates'])
    assert event['dormant_catalog']
    assert all(not r['admission_legal'] and r['category'] == CONTRACT for r in event['dormant_catalog'])
    state['controls']['search_activated'] = True
    # Keep builder's valid positive budget but occupy it with other heat IDs.
    state['controls']['max_search_uavs'] = 1
    state['controls']['search_assignments'] = {9: 0}
    event = enumerate_joint(state)
    assert not any(r['role'] == 'TARGET_SEARCH' for r in event['candidates'])
    assert any('search_capacity_full' in r['prerequisites'] for r in event['dormant_catalog'])


def test_deadline_and_observed_closed_cut_are_reused():
    state = state_fixture()
    state['release']['fleet_uav_ids'] = set(state['release']['fleet_uav_ids'])
    state['max_steps'] = 11
    state['sensors']['explorer']['range'] = 1.
    event = enumerate_joint(state)
    assert all(not r['admission_legal'] for r in event['candidates']
               if r['role'] == 'TARGET_SEARCH' and r['uav_id'] in (0, 2))
    assert any(any(x['stage'] == 'last_deadline_masks' for x in r['rejection_reasons']) for r in event['candidates'])
    state['max_steps'] = 3000
    # Observed solid wall: optimistic UNKNOWN connectivity still cannot cross it.
    for y in range(32):
        state['recovery']['cells'][20, y] = 1
    event = enumerate_joint(state)
    assert any(any(x['stage'] == 'last_observed_reachability_masks' for x in r['rejection_reasons'])
               for r in event['candidates'])


def test_radio_endpoint_prediction_is_not_a_motion_certificate():
    state = state_fixture()
    state['release']['fleet_uav_ids'] = set(state['release']['fleet_uav_ids'])
    state['recovery']['observed_positions'].pop(2)
    event = enumerate_joint(state)
    relay = [r for r in event['candidates'] if r['role'] == 'RELAY' and r['uav_id'] == 2]
    assert relay and relay[0]['admission_legal']
    assert not relay[0]['immediately_executable']
    assert relay[0]['prefix']['reason'] == 'missing_own_sensor_pose'


def test_observed_hazard_blocks_prefix_without_changing_dynamics():
    state = state_fixture()
    state['release']['fleet_uav_ids'] = set(state['release']['fleet_uav_ids'])
    state['hazards'] = [dict(obstacle_id='public', position=np.array([10., 15.]),
        spawn_step=1, initial_radius=0., expansion_rate=.02, max_radius=4.)]
    event = enumerate_joint(state)
    assert all(not r['immediately_executable'] for r in event['candidates'] if r['uav_id'] == 2)
    assert state['dynamics']['explorer']['velocity'] == 1.


def test_projection_never_reads_hidden_fields_and_live_caches_are_unchanged():
    s, stale = live_fixture()
    class Hidden:
        def __getattribute__(self, name):
            raise AssertionError(f'hidden read: {name}')
    s.runtime.target_detector = Hidden()
    s.runtime.search_scenario = Hidden()
    s.runtime.obstacles.get_occupancy_grid = Hidden()
    for t in s.runtime.tasks.tasks.values():
        t.params = Hidden()
    # robot.position must not be fetched, even as an eager dict.get default.
    for r in s.runtime.robots:
        del r.position
    before = related_state_digest(s, stale, 'trap')
    state = capture_public_state(s, stale, scenario='trap')
    enumerate_joint(state)
    assert related_state_digest(s, stale, 'trap') == before
    assert s._search_recovery._observed_reachability_gate is None
    assert not hasattr(s, '_relay_anchor_stamp')


def test_recorder_forwards_original_once_and_preserves_real_rule_actions():
    plain, stale = live_fixture()
    from dataclasses import replace
    plain.heat_points = [replace(plain.heat_points[0], position=(14., 18.))]
    plain.heat_by_id = {0: plain.heat_points[0]}
    plain._refresh_search(stale)
    plain._refresh_relay(stale)
    expected = list(plain.last_event_assignments)
    traced, stale = live_fixture()
    traced.heat_points = [replace(traced.heat_points[0], position=(14., 18.))]
    traced.heat_by_id = {0: traced.heat_points[0]}
    calls = []
    original = traced.allocator.adapter.assign
    def counted(graph, deterministic=True):
        calls.append(graph.valid_action_pairs)
        return original(graph, deterministic)
    traced.allocator.adapter.assign = counted
    recorder = JointShadowRecorder(traced, scenario='fixture')
    traced._refresh_search(stale)
    traced._refresh_relay(stale)
    recorder.close()
    assert traced.last_event_assignments == expected
    assert len(calls) == len(recorder.events[0]['actual_graphs'])
    assert traced.allocator.adapter.assign is counted
    assert '_refresh_search' not in traced.__dict__
    event = recorder.events[0]
    assert event['purity_verified'] and event['repeat_verified']
    assert event['metrics']['executable_both_roles']
    assert any(r['category'] == ORDER for r in event['candidates'])
    assert summarize([event])['events'] == 1
    assert summarize([event])['actual_mixed_graphs'] == 0


def test_unsupported_snapshot_contract_fails_closed():
    s, stale = live_fixture()
    s.runtime.geometry_mode = 'marvel_native'
    with pytest.raises(ValueError, match='Only audited extended'):
        capture_public_state(s, stale, scenario='invalid')
