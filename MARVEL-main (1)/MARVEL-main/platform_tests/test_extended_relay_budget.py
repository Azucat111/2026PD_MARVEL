"""Extended fleet capacity must not strand a legal residual Relay bridge."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from integrations.gppo.mesh_relay import ObservedMeshRelayDemandBuilder
from integrations.gppo.scheduler import GPPOTaskScheduler

@pytest.mark.parametrize('count,expected',[(4,2),(13,2),(30,3),(60,6)])
def test_extended_budget_scales_one_helper_per_ten_uavs(count,expected):
 from integrations.gppo.mesh_relay import extended_relay_helper_budget
 assert extended_relay_helper_budget(count,2)==expected

def test_arrived_two_helpers_allow_real_observed_third_bridge():
 s=GPPOTaskScheduler.__new__(GPPOTaskScheduler)
 s.base_position=np.array([0.,0.]);s.relay_assignments={2:(12,np.array([11.5,23.5])),12:(25,np.array([16.5,47.5]))};s.search_assignments={}
 s.relay_builder=ObservedMeshRelayDemandBuilder(comm_range=20,max_hops=2,max_demands=2)
 s.relay_builder.anchor_candidates_provider=lambda:[np.array([43.5,7.5])]
 s.runtime=NS(geometry_mode='extended',robots=[NS(robot_id=i) for i in range(30)])
 s.relay_trigger_threshold=.9;s.safety=NS(active_uav_ids=set());s.last_event_assignments=[];s._apply_uav_state=lambda *args:None
 # An already connected left backbone and isolated right pair; helpers are really at anchors.
 positions={0:np.array([33.,8.]),1:np.array([17.,10.]),2:np.array([14.,32.]),12:np.array([11.5,23.5]),25:np.array([16.5,47.5]),6:np.array([54.,8.]),29:np.array([67.,12.])}
 calls=[]
 def allocate(slots,**kwargs):
  calls.append(slots);return [NS(source_entity_id=slots[0].source_entity_id,uav_id=29,total_time=24.)]
 s.allocator=NS(allocate=allocate)
 s.relay_release_gate=NS(reset_stability=lambda:None)
 s._augment_arrived_relay(NS(task_id='T3_relay'),positions)
 assert len(calls)==1 and len(s.relay_assignments)==3
 assert {12,25}<=set(calls[0][0].forbidden_uav_ids)
 assert s.relay_assignments[6][0]==29
 assert s.relay_builder.snapshot({**positions,29:np.array([43.5,7.5])}).connectivity_ratio==1
