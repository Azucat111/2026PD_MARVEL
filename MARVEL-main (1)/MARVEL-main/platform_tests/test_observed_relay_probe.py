from types import SimpleNamespace
from utils.policy_adapter import MARVELPolicyAdapter as Adapter
from integrations.gppo.scheduler import GPPOTaskScheduler as Scheduler
import numpy as np
from integrations.gppo.relay_frontier import ObservedRelayFrontier
from integrations.gppo.search_recovery import ObservedSearchRecovery
from utils.geometry import GeometryFrame

def scene():
 rec=ObservedSearchRecovery(GeometryFrame.extended(width_cells=80,height_cells=80))
 # Two internally connected public components; known free probing lane between them.
 rec.cells={(c,r):0 for c in range(4,30) for r in range(4,12)}
 positions={0:np.array([5.5,8.5]),1:np.array([6.5,8.5]),2:np.array([45.5,8.5]),3:np.array([46.5,8.5])}
 rec.observed_positions={u:p.copy() for u,p in positions.items()}
 return rec,positions

def test_equal_lcc_probe_is_available_without_truth():
 rec,pos=scene();choice=ObservedRelayFrontier(rec).select(pos)
 assert choice is not None,'Missing observed information-gathering fallback'
 assert choice['forecast_lcc']>=2
 assert choice['potential_unknown_cells']>0
 assert all(rec.segment_safe(a,b) for a,b in zip(choice['path'],choice['path'][1:]))

def test_reserved_helpers_are_never_selected():
 rec,pos=scene();choice=ObservedRelayFrontier(rec).select(pos,reserved={0,1,2})
 assert choice is None or choice['uid']==3

def test_warnings_can_exclude_all_probe_segments():
 rec,pos=scene();assert ObservedRelayFrontier(rec).select(pos,segment_filter=lambda a,b:False) is None

def test_unknown_motion_is_never_certified():
 rec,pos=scene();rec.cells={};assert ObservedRelayFrontier(rec).select(pos) is None

def test_connected_mesh_does_not_probe():
 rec,pos=scene();pos[2]=np.array([9.5,8.5]);pos[3]=np.array([10.5,8.5]);rec.observed_positions=pos
 assert ObservedRelayFrontier(rec).select(pos) is None

def test_completed_stationkeeping_preserves_active_probe():
 robot=SimpleNamespace(robot_id=7,position=np.array([1.,1.]),heading=90.)
 rec=SimpleNamespace(reset_uid=lambda uid:None)
 adapter=Adapter.__new__(Adapter);adapter._observed_belief=SimpleNamespace(evidence=rec)
 adapter.scheduler=SimpleNamespace(_search_recovery=rec,search_assignments={},relay_assignments={},safety=SimpleNamespace(active_uav_ids=set()),_relay_probe={'uid':7})
 adapter.runtime=SimpleNamespace(tasks=SimpleNamespace(all_complete=lambda:True),robots=[robot])
 action=(np.array([2.,1.]),180.)
 result=adapter._stationkeep_completed_tasks([action])
 assert np.array_equal(result[0][0],action[0]),'Active observed Relay probe was canceled by completed-task hold'

def test_completed_idle_uavs_still_hold():
 robot=SimpleNamespace(robot_id=8,position=np.array([1.,1.]),heading=90.);rec=SimpleNamespace(reset_uid=lambda uid:None)
 adapter=Adapter.__new__(Adapter);adapter._observed_belief=SimpleNamespace(evidence=rec)
 adapter.scheduler=SimpleNamespace(_search_recovery=rec,search_assignments={},relay_assignments={},safety=SimpleNamespace(active_uav_ids=set()),_relay_probe={'uid':7})
 adapter.runtime=SimpleNamespace(tasks=SimpleNamespace(all_complete=lambda:True),robots=[robot])
 result=adapter._stationkeep_completed_tasks([(np.array([2.,1.]),180.)]);assert np.array_equal(result[0][0],robot.position)


def test_inactive_relay_cancels_probe():
 sched=Scheduler.__new__(Scheduler);sched._active_task=lambda kind:None;sched.base_position=np.zeros(2)
 sched._relay_probe={'uid':7};sched.relay_assignments={};sched.relay_release_gate=SimpleNamespace(reset_stability=lambda:None)
 sched._refresh_relay({});assert sched._relay_probe is None

def test_satisfied_radio_cancels_probe():
 sched=Scheduler.__new__(Scheduler);sched._active_task=lambda kind:SimpleNamespace(task_id='relay');sched.base_position=np.zeros(2)
 sched._relay_probe={'uid':7};sched.relay_assignments={};sched.relay_release_gate=SimpleNamespace(reset_stability=lambda:None)
 sched.relay_trigger_threshold=.9;sched.relay_builder=SimpleNamespace(snapshot=lambda *a,**k:SimpleNamespace(connectivity_ratio=1.))
 sched._refresh_relay({0:np.zeros(2)});assert sched._relay_probe is None

from integrations.gppo.mesh_relay import ObservedMeshRelayDemandBuilder

def probe_scheduler(arrived=True):
 rec,pos=scene();sched=Scheduler.__new__(Scheduler)
 robots=[SimpleNamespace(robot_id=u,position=p.copy(),heading=0.) for u,p in pos.items()]
 events=[];sched.runtime=SimpleNamespace(geometry_mode='extended',tasks=SimpleNamespace(all_complete=lambda:True),robots=robots,current_step=0,dt=.1,_sensor_params=lambda:(120.,10.),_log_event=lambda kind,data:events.append((kind,data)))
 sched._search_recovery=rec;sched._active_task=lambda kind:SimpleNamespace(task_id='relay');sched.search_assignments={}
 sched.relay_builder=ObservedMeshRelayDemandBuilder();sched.relay_trigger_threshold=.9;sched.safety=SimpleNamespace(active_uav_ids=set(),hazards=None)
 anchor=pos[3].copy() if arrived else pos[3]+np.array([3.,0.]);sched.relay_assignments={2:(3,anchor)}
 return sched,pos

def test_arrived_relay_can_probe_with_remaining_budget():
 sched,pos=probe_scheduler();sched._refresh_observed_relay_probe(pos)
 assert sched._relay_probe is not None,'Arrived incomplete bridge must permit bounded information gathering'
 assert sched._relay_probe['uid']!=3

def test_in_transit_relay_does_not_start_extra_probe():
 sched,pos=probe_scheduler(False);sched._refresh_observed_relay_probe(pos);assert sched._relay_probe is None
