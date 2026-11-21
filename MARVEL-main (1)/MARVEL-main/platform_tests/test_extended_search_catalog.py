"""A task catalog is not the number of concurrently executable Search UAVs."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from integrations.gppo.event_sources import SearchSlotBuilder,PublicHeatPoint
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.event_allocator import GPPOEventAllocator
from integrations.gppo.task_graph import UAVState
from integrations.rule_scheduling import choose_rule_assignment
import integrations.gppo.event_allocator as allocation_module

def points():
    return [PublicHeatPoint(0,(40.,0.)),PublicHeatPoint(1,(50.,0.)),PublicHeatPoint(2,(0.,0.)),PublicHeatPoint(3,(1.,0.))]

def test_default_builder_preserves_frozen_prefix():
    assert [s.source_entity_id for s in SearchSlotBuilder(2).build(points())]==[0,1]

def test_full_catalog_excludes_active_and_serviced_public_heats():
    selected=SearchSlotBuilder(2).build([PublicHeatPoint(i,(float(i),0.),serviced=i==1) for i in range(5)],
        active_heat_ids=[0],include_all_pending=True)
    assert [s.source_entity_id for s in selected]==[2,3,4]

@pytest.fixture
def scheduler(monkeypatch):
    s=GPPOTaskScheduler.__new__(GPPOTaskScheduler)
    s.runtime=NS(geometry_mode='extended',current_step=0,dt=.1)
    s.search_activated=True;s.max_search_uavs=2;s.search_assignments={};s.serviced_heat_ids=set()
    s.relay_assignments={};s.safety=NS(active_uav_ids=set());s.heat_points=points()
    s.search_builder=SearchSlotBuilder(2);s.last_event_assignments=[];s._search_recovery=None
    s._active_task=lambda kind:NS(task_id='T2_target_search')
    s._native_search_context=lambda slots:();s._apply_uav_state=lambda *a:None;s._publish_search_uavs=lambda:None
    states=[UAVState(i,position=(float(i),0.)) for i in range(4)]
    class GraphBuilder:
        def __init__(self,*a,**k):pass
        def build(self):return NS(graph=NS(uav_states=states))
    monkeypatch.setattr(allocation_module,'RuntimeGraphBuilder',GraphBuilder)
    a=GPPOEventAllocator.__new__(GPPOEventAllocator);a.runtime=s.runtime;a._marvel_agents={}
    a.adapter=NS(assign=choose_rule_assignment)
    a._travel_time_matrix=lambda uavs,tasks,agents:np.array([[np.linalg.norm(np.asarray(t.position)-np.asarray(u.position)) for u in uavs] for t in tasks],dtype=np.float32)
    a._apply_deadline_mask=lambda *a:None;a._apply_observed_reachability_mask=lambda *a:None;a._apply_route_support_mask=lambda *a:None
    s.allocator=a;s.catalog_calls=[];real_allocate=a.allocate
    def allocate(slots,**kwargs):
        s.catalog_calls.append([row.source_entity_id for row in slots]);return real_allocate(slots,**kwargs)
    a.allocate=allocate
    return s

def test_extended_rule_can_choose_cheapest_tasks_outside_prefix_without_exceeding_capacity(scheduler):
    scheduler._refresh_search({})
    assert scheduler.catalog_calls==[[0,1,2,3]]
    assert set(scheduler.search_assignments)=={2,3}
    assert len(scheduler.search_assignments)==2

def test_existing_search_relay_and_safety_reservations_and_remaining_capacity_are_kept(scheduler):
    scheduler.search_assignments={1:0};scheduler.relay_assignments={99:(1,np.array([2.,0.]))};scheduler.safety.active_uav_ids={2}
    scheduler._refresh_search({})
    assert scheduler.catalog_calls==[[0,2,3]]
    assert scheduler.search_assignments=={1:0,3:3}
    assert len(scheduler.search_assignments)==2

def test_full_capacity_does_not_allocate_or_reassign(scheduler):
    scheduler.search_assignments={0:0,1:1};scheduler._refresh_search({})
    assert scheduler.catalog_calls==[] and scheduler.search_assignments=={0:0,1:1}

def test_native_keeps_existing_prefix_and_one_free_slot(scheduler):
    scheduler.runtime.geometry_mode='marvel_native';scheduler.search_assignments={3:0}
    scheduler._refresh_search({})
    assert scheduler.catalog_calls==[[0]]
    assert set(scheduler.search_assignments)=={0,3}

def test_expanded_catalog_still_obeys_hard_unreachable_masks(scheduler):
    def mask(graph,age):
        index=graph.subtask_id_to_index.get(100002)
        if index is not None:graph.action_mask[index,:]=True
    scheduler.allocator._apply_observed_reachability_mask=mask
    scheduler._refresh_search({})
    assert set(scheduler.search_assignments)=={0,3}
    assert len(scheduler.search_assignments)==2
