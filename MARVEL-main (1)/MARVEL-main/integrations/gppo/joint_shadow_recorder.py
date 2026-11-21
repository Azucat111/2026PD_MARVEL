"""Opt-in instance instrumentation for joint_shadow; never selects a real action."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import random
import pickle

import numpy as np
import torch

from .joint_shadow import (capture_public_state, enumerate_joint,
                           attach_actual, graph_record, snapshot_digest)


def rng_digest():
    states = [random.getstate(), np.random.get_state(), torch.random.get_rng_state().numpy()]
    if torch.cuda.is_initialized():
        states.extend(s.cpu().numpy() for s in torch.cuda.get_rng_state_all())
    return hashlib.sha256(pickle.dumps(states)).hexdigest()


def related_state_digest(scheduler, stale, scenario):
    """Public state PLUS existing caches/counters, so cache writes cannot hide."""
    rec = scheduler._search_recovery
    gate = getattr(rec, '_observed_reachability_gate', None)
    extra = dict(recovery_cache=None if gate is None else gate.__dict__,
                 recovery_stamp=getattr(rec, '_observed_reachability_step', None),
                 anchors={k: v for k, v in scheduler.__dict__.items()
                          if k in ('_relay_anchor_stamp', '_relay_anchor_points')},
                 active_search=getattr(scheduler.runtime, 'active_search_uav_ids', None),
                 last_events=list(scheduler.last_event_assignments),
                 sensor_caches={k: (v._visible_cell_cache, v._visible_cache_hits, v._visible_cache_misses)
                                for k, v in scheduler.runtime.sensor_by_type.items()})
    payload = (snapshot_digest(capture_public_state(scheduler, stale, scenario=scenario)), extra)
    return hashlib.sha256(pickle.dumps(payload)).hexdigest()


class JointShadowRecorder:
    """Wrap only this scheduler instance. Restore every wrapper with close().

    _refresh_search entry is precisely after _high_level_update's reconcile.
    allocator adapter observation forwards each original assign exactly once.
    Calling enumerate_joint never invokes the original policy/allocator instance.
    """
    def __init__(self, scheduler, *, scenario, verify_repeat=True):
        self.scheduler, self.scenario = scheduler, scenario
        self.verify_repeat = verify_repeat
        self.events, self.states = [], []
        self._pending = None
        self._originals = {}
        self._install(scheduler, '_refresh_search', self._search)
        self._install(scheduler, '_refresh_relay', self._relay)
        self._install(scheduler.allocator, 'allocate', self._allocate)
        self._install(scheduler, 'apply', self._apply)

    def _install(self, obj, name, method):
        self._originals[name] = (obj, obj.__dict__.get(name), name in obj.__dict__, getattr(obj, name))
        setattr(obj, name, method)

    def close(self):
        for name, (obj, own, existed, _) in self._originals.items():
            if existed:
                setattr(obj, name, own)
            else:
                delattr(obj, name)

    def _search(self, stale):
        s = self.scheduler
        before, rng = related_state_digest(s, stale, self.scenario), rng_digest()
        state = capture_public_state(s, stale, scenario=self.scenario)
        event = enumerate_joint(state)
        if self.verify_repeat:
            assert event == enumerate_joint(state), 'Shadow enumeration is not repeatable'
        assert before == related_state_digest(s, stale, self.scenario), 'Shadow mutated live related state'
        assert rng == rng_digest(), 'Shadow consumed an RNG'
        event['purity_verified'] = True
        event['repeat_verified'] = self.verify_repeat
        self.states.append(state)
        self._pending = event
        event['actual_graphs'] = []
        return self._originals['_refresh_search'][3](stale)

    def _relay(self, stale):
        result = self._originals['_refresh_relay'][3](stale)
        if self._pending is not None:
            event = self._pending
            attach_actual(event, event['actual_graphs'], self.scheduler.last_event_assignments)
            event['executor_ownership_after'] = dict(
                search=dict(self.scheduler.search_assignments),
                relay={k: (int(uid), anchor.tolist()) for k, (uid, anchor)
                       in self.scheduler.relay_assignments.items()},
                safety=sorted(self.scheduler.safety.active_uav_ids))
            self.events.append(event)
            self._pending = None
        return result

    def _allocate(self, *args, **kwargs):
        adapter = self.scheduler.allocator.adapter
        original = adapter.assign
        own_assign = adapter.__dict__.get('assign')
        had_own_assign = 'assign' in adapter.__dict__
        def observe(graph, deterministic=True):
            record = graph_record(graph)
            choice = original(graph, deterministic=deterministic)
            record['chosen'] = None if choice is None else asdict(choice)
            if self._pending is not None:
                self._pending['actual_graphs'].append(record)
            return choice
        adapter.assign = observe
        try:
            return self._originals['allocate'][3](*args, **kwargs)
        finally:
            if had_own_assign:
                adapter.assign = own_assign
            else:
                delattr(adapter, 'assign')

    def _apply(self, actions):
        result = self._originals['apply'][3](actions)
        if self.events and self.events[-1]['step'] == self.scheduler.runtime.current_step:
            self.events[-1]['actual_scheduler_commands'] = [
                dict(uav_id=int(r.robot_id), waypoint=np.asarray(a[0]).tolist(), heading=float(a[1]))
                for r, a in zip(self.scheduler.runtime.robots, result)]
        return result
