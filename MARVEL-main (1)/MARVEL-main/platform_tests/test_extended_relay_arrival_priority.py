from types import SimpleNamespace as NS
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))

from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.mesh_relay import ObservedMeshRelayDemandBuilder
from integrations.gppo.event_sources import ObservedRelayDemandBuilder

def test_arrival_is_considered_before_retarget_moves_anchor():
 s=GPPOTaskScheduler.__new__(GPPOTaskScheduler);s.base_position=np.zeros(2);s._active_task=lambda _:object();s.relay_assignments={26:(29,np.zeros(2))};s.relay_release_gate=NS(observe=lambda **kw:False);s.relay_builder=ObservedMeshRelayDemandBuilder();events=[]
 def refresh(assignments,positions):events.append('retarget');assignments[26]=(29,np.array([0.,59.]))
 s.relay_builder.refresh_assignments=refresh
 s._augment_arrived_relay=lambda *args:events.append('arrived' if np.linalg.norm(s.relay_assignments[26][1])<=.5 else 'inflight')
 s._refresh_observed_relay_probe=lambda *args:events.append('probe')
 GPPOTaskScheduler._refresh_relay(s,{29:np.zeros(2)})
 assert events==['arrived','retarget','probe']

def test_native_builder_branch_is_unchanged():
 s=GPPOTaskScheduler.__new__(GPPOTaskScheduler);s.base_position=np.zeros(2);s._active_task=lambda _:object();s.relay_assignments={1:(2,np.zeros(2))};s.relay_release_gate=NS(observe=lambda **kw:False);s.relay_builder=ObservedRelayDemandBuilder()
 s._augment_arrived_relay=lambda *args:(_ for _ in ()).throw(AssertionError('native changed'))
 GPPOTaskScheduler._refresh_relay(s,{2:np.zeros(2)})
 assert len(s.relay_assignments)==1

def test_existing_arrived_augmentation_and_reservations_remain():
 from platform_tests.test_extended_relay_augmentation import fixture
 s,public,calls=fixture();GPPOTaskScheduler._refresh_relay(s,public)
 assert len(calls)==1 and len(s.relay_assignments)==2
 assert 4 in calls[0][0].forbidden_uav_ids

def test_inflight_bridge_is_still_not_augmented():
 from platform_tests.test_extended_relay_augmentation import fixture
 s,public,calls=fixture(arrived=False);GPPOTaskScheduler._refresh_relay(s,public)
 assert not calls and len(s.relay_assignments)==1


