from types import SimpleNamespace as NS
import numpy as np
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.mesh_relay import ObservedMeshRelayDemandBuilder

def fixture(arrived=True):
    sched=GPPOTaskScheduler.__new__(GPPOTaskScheduler)
    sched.base_position=np.array([0.,0.]);sched._active_task=lambda name:NS(task_id='T3_relay')
    sched.relay_assignments={2:(4,np.array([17.,0.]))};sched.search_assignments={}
    sched.relay_builder=ObservedMeshRelayDemandBuilder(comm_range=20,max_hops=2,max_demands=2)
    sched.relay_release_gate=NS(observe=lambda **kw:False,reset_stability=lambda:None)
    sched.relay_trigger_threshold=.9;sched.safety=NS(active_uav_ids=set())
    sched.last_event_assignments=[];sched._apply_uav_state=lambda *args:None
    calls=[]
    def allocate(slots,**kwargs):
        calls.append(slots)
        return [NS(source_entity_id=slots[0].source_entity_id,uav_id=0,total_time=1.)]
    sched.allocator=NS(allocate=allocate)
    public={i:np.array([float(x),0.]) for i,x in enumerate([0,4,30,34,17 if arrived else 8,65,69])}
    return sched,public,calls

def test_arrived_extended_bridge_can_add_second_existing_allocator_slot():
    sched,public,calls=fixture();sched._refresh_relay(public)
    assert len(calls)==1 and len(calls[0])==1
    assert 4 in calls[0][0].forbidden_uav_ids
    assert sched.relay_assignments[2][0]==4
    assert len(sched.relay_assignments)==2

def test_inflight_bridge_is_not_supplemented_from_a_hypothetical_arrival():
    sched,public,calls=fixture(arrived=False);sched._refresh_relay(public)
    assert not calls and len(sched.relay_assignments)==1
