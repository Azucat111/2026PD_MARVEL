import numpy as np
from integrations.gppo.mesh_relay import ObservedMeshRelayDemandBuilder,ObservedMeshRelayReleaseGate
from utils.communication_model import CommunicationModel

def positions(xs):return {i:np.array([float(x),0.]) for i,x in enumerate(xs)}
def builder():return ObservedMeshRelayDemandBuilder(comm_range=20,max_hops=2,max_demands=2)

def test_connected_chain_far_from_base_creates_no_false_relay():
 p=positions([30,45,60]);b=builder();s=b.snapshot(p,base_position=[0,0])
 assert s.connectivity_ratio==1
 assert not b.build(p,base_position=[0,0]).slots

def test_far_from_base_bridge_uses_existing_midpoint_primitive():
 p=positions([100,105,110,140,150]);b=builder();r=b.build(p,base_position=[0,0])
 assert r.snapshot.connectivity_ratio==.6
 assert len(r.slots)==1
 slot=r.slots[0]
 assert np.allclose(slot.position,[125,0])
 assert slot.source_entity_id==3
 assert {2,3}<=set(slot.forbidden_uav_ids)
 moved={**p,1:np.array(slot.position)}
 comm=CommunicationModel({'comm_range':20})
 assert comm.connectivity_ratio(comm.get_topology(moved.values()))==1

def test_release_uses_mesh_counterfactual_and_full_fleet_floor():
 b=builder();gate=ObservedMeshRelayReleaseGate(release_threshold=.95,stable_release_steps=3,min_connectivity=.9,fleet_uav_ids=tuple(range(30)))
 p=positions([45]*30);p[0]=np.array([100.,0.]);assignment={1:(0,np.array([45.,0.]))}
 assert not gate.observe(assignments=assignment,positions=p,relay_builder=b,base_position=[0,0])
 assert not gate.observe(assignments=assignment,positions=p,relay_builder=b,base_position=[0,0])
 assert gate.observe(assignments=assignment,positions=p,relay_builder=b,base_position=[0,0])
 p.pop(29)
 assert not gate.observe(assignments=assignment,positions=p,relay_builder=b,base_position=[0,0])
 assert gate.stable_counter==0

def test_small_fleet_release_cannot_ignore_isolated_helpers_in_denominator():
 b=builder();gate=ObservedMeshRelayReleaseGate(release_threshold=.95,stable_release_steps=1,min_connectivity=.9,fleet_uav_ids=(0,1,2))
 p=positions([100,45,60]);a={1:(0,np.array([50.,0.]))}
 assert not gate.observe(assignments=a,positions=p,relay_builder=b,base_position=[0,0])


def test_articulation_helper_is_masked_despite_larger_final_component():
 from integrations.gppo.task_graph import TaskGraph,UAVState,Subtask,TaskType
 p=positions([0,20,40,70,90]);b=builder()
 g=TaskGraph(uav_states=[UAVState(uid,position=tuple(xy)) for uid,xy in p.items()],
     subtasks=[Subtask(200003,TaskType.RELAY,2,active=True,position=(55.,0.))])
 mask=b.candidate_mask(g,[])
 assert mask[0,1]
 assert not mask[0,0]

def test_joint_mask_rejects_redundant_second_helper():
 from types import SimpleNamespace
 from integrations.gppo.task_graph import TaskGraph,UAVState,Subtask,TaskType
 p=positions([0,20,40,70,90]);b=builder()
 g=TaskGraph(uav_states=[UAVState(uid,position=tuple(xy)) for uid,xy in p.items()],
     subtasks=[Subtask(200003,TaskType.RELAY,2,active=True,position=(55.,0.)),
               Subtask(200004,TaskType.RELAY,2,active=True,position=(35.,0.))])
 assert b.candidate_mask(g,[SimpleNamespace(uav_id=0,slot_id=200003)])[1].all()

def test_public_component_ties_and_translation_do_not_depend_on_dict_order():
 p=positions([0,10,100,110]);b=builder()
 a=b.snapshot(p,base_position=[0,0])
 moved={uid:xy+np.array([1000.,500.]) for uid,xy in reversed(list(p.items()))}
 c=b.snapshot(moved,base_position=[0,0])
 assert a==c
 assert a.connected_uav_ids==(0,1)


def test_additional_guard_is_applied_in_real_event_allocator(monkeypatch):
 from types import SimpleNamespace
 import integrations.gppo.event_allocator as module
 from integrations.gppo.task_graph import TaskGraph,UAVState,TaskType
 from integrations.rule_scheduling import choose_rule_assignment
 base=TaskGraph(uav_states=[UAVState(0,position=(0,0)),UAVState(1,position=(1,0))],subtasks=[])
 monkeypatch.setattr(module,'RuntimeGraphBuilder',lambda *a,**kw:SimpleNamespace(build=lambda:SimpleNamespace(graph=base)))
 a=module.GPPOEventAllocator.__new__(module.GPPOEventAllocator)
 a.runtime=SimpleNamespace(current_step=0,dt=1);a._marvel_agents={}
 a.adapter=SimpleNamespace(assign=choose_rule_assignment)
 a._travel_time_matrix=lambda u,t,agents:np.zeros((len(t),len(u)))
 for name in ('_apply_deadline_mask','_apply_observed_reachability_mask','_apply_route_support_mask'):
  setattr(a,name,lambda *args:None)
 slot=module.GPPOEventSlot(slot_id=200003,task_type=TaskType.RELAY,priority=2,position=(55,0),processing_time=1,source_task_id='T3',source_entity_id=3)
 decisions=a.allocate([slot],candidate_guard=lambda graph,assigned:np.array([[True,False]]))
 assert len(decisions)==1 and decisions[0].uav_id==1

def test_active_mesh_bridge_tracks_moving_public_component():
 from types import SimpleNamespace
 from integrations.gppo.scheduler import GPPOTaskScheduler
 s=GPPOTaskScheduler.__new__(GPPOTaskScheduler)
 s.runtime=SimpleNamespace(geometry_mode='extended',robots=[SimpleNamespace(robot_id=i) for i in range(5)])
 s.base_position=np.array([0.,0.]);s._active_task=lambda role:SimpleNamespace(task_id='T3_relay')
 s.relay_builder=builder()
 s.relay_release_gate=ObservedMeshRelayReleaseGate(release_threshold=.95,stable_release_steps=3,min_connectivity=.9,fleet_uav_ids=range(5))
 # Isolate moving-anchor behavior; arrival augmentation has separate tests.
 s._augment_arrived_relay=lambda task,public:None
 s.relay_assignments={3:(1,np.array([125.,0.]))}
 p=positions([100,125,110,150,160])
 s._refresh_relay(p)
 assert s.relay_assignments[3][0]==1
 assert np.allclose(s.relay_assignments[3][1],[130,0])
 forecast={**p,1:s.relay_assignments[3][1]}
 assert s.relay_builder.snapshot(forecast).connectivity_ratio==1

def test_connected_mesh_bridge_retains_stability_window():
 b=builder();assignments={3:(1,np.array([125.,0.]))}
 b.refresh_assignments(assignments,positions([100,125,110,130,140]))
 assert np.allclose(assignments[3][1],[125,0])

def test_retarget_survives_target_component_becoming_largest():
 b=builder();a={3:(1,np.array([125.,0.]))};p=positions([100,125,110,150,160,170])
 b.refresh_assignments(a,p)
 assert np.allclose(a[3][1],[130,0])
 assert b.snapshot({**p,1:a[3][1]}).connectivity_ratio==1

def test_retarget_cannot_cut_existing_nonhelper_component():
 b=builder();a={3:(1,np.array([15.,0.]))};p=positions([0,15,0,-30,30])
 b.refresh_assignments(a,p)
 assert np.allclose(a[3][1],[15,0])

def test_joint_refresh_forecasts_unchanged_helpers_at_planned_anchors():
 b=builder();a={2:(0,np.array([25.,0.])),3:(1,np.array([67.5,0.]))}
 p=positions([60,30,95,55,80,45,15,95])
 b.refresh_assignments(a,p)
 assert np.allclose(a[2][1],[25,0])
 final={**p,**{uid:anchor for uid,anchor in a.values()}}
 assert b.snapshot(final).connectivity_ratio==1


def test_observed_free_radio_lens_replaces_unverified_midpoint():
 b=builder();b.anchor_candidates_provider=lambda:[np.array([20.,8.])]
 p={0:np.array([0.,0.]),1:np.array([5.,0.]),2:np.array([35.,0.]),3:np.array([36.,0.])}
 result=b.build(p,base_position=[0,0])
 assert len(result.slots)==1
 np.testing.assert_array_equal(result.slots[0].position,[20.,8.])

def test_no_observed_legal_anchor_does_not_create_blind_relay_slot():
 b=builder();b.anchor_candidates_provider=lambda:[]
 assert not b.build(positions([0,5,35,36]),base_position=[0,0]).slots

def test_anchor_checks_any_component_boundary_not_only_nearest_pair():
 b=builder();b.anchor_candidates_provider=lambda:[np.array([15.,20.])]
 p={0:np.array([0.,0.]),1:np.array([0.,10.]),2:np.array([30.,0.]),3:np.array([30.,10.])}
 result=b.build(p,base_position=[0,0])
 assert len(result.slots)==1
 np.testing.assert_array_equal(result.slots[0].position,[15.,20.])
 assert {1,3}<=set(result.slots[0].forbidden_uav_ids)

def test_refresh_uses_same_observed_legal_anchor_contract():
 b=builder();b.anchor_candidates_provider=lambda:[np.array([20.,8.])]
 p={0:np.array([0.,0.]),1:np.array([5.,0.]),2:np.array([35.,0.]),3:np.array([36.,0.]),4:np.array([2.,0.])}
 assignments={2:(4,np.array([20.,0.]))}
 b.refresh_assignments(assignments,p)
 np.testing.assert_array_equal(assignments[2][1],[20.,8.])


def test_observed_motion_impossibility_adds_mask_before_radio_forecast():
 from integrations.gppo.task_graph import TaskGraph,UAVState,Subtask,TaskType
 p=positions([0,20,40,70,90]);b=builder()
 g=TaskGraph(uav_states=[UAVState(uid,position=tuple(xy)) for uid,xy in p.items()],subtasks=[Subtask(200003,TaskType.RELAY,2,active=True,position=(55.,0.))])
 assert not b.candidate_mask(g,[])[0,0]
 b.motion_candidate_guard=lambda uid,position,goal:uid==0
 assert b.candidate_mask(g,[])[0,0]


def test_retarget_keeps_a_chain_through_smaller_component_when_largest_is_unreachable():
 p={0:np.array([80.,0.]),1:np.array([81.,0.]),2:np.array([82.,0.]),3:np.array([0.,0.]),4:np.array([1.,0.]),5:np.array([40.,0.]),6:np.array([0.,1.]),7:np.array([60.,0.])}
 b=builder();b.anchor_candidates_provider=lambda:[np.array([20.5,3.]),np.array([60.,0.])]
 assignments={3:(6,np.array([20.5,0.])),6:(7,np.array([60.,0.]))}
 before=b.snapshot(p).connectivity_ratio
 b.refresh_assignments(assignments,p)
 np.testing.assert_allclose(assignments[3][1],[20.5,3.])
 forecast=dict(p)
 for uid,anchor in assignments.values():forecast[uid]=anchor
 assert before<1 and b.snapshot(forecast).connectivity_ratio==1


def test_retarget_prefers_anchor_joining_three_components_over_pair_midpoint():
 b=ObservedMeshRelayDemandBuilder(comm_range=10,max_hops=2,max_demands=2)
 p={0:np.array([20.,16.]),1:np.array([20.,17.]),2:np.array([20.,18.]),3:np.array([20.,0.]),4:np.array([21.,0.]),5:np.array([0.,0.]),6:np.array([10.,2.]),7:np.array([0.,1.]),8:np.array([20.,8.])}
 b.anchor_candidates_provider=lambda:[np.array([5.,1.]),np.array([15.,1.]),np.array([10.,0.]),np.array([20.,8.])]
 assignments={6:(7,np.array([5.,1.])),7:(8,np.array([20.,8.]))}
 b.refresh_assignments(assignments,p)
 np.testing.assert_allclose(assignments[6][1],[10.,0.])
 forecast=dict(p)
 for uid,point in assignments.values():forecast[uid]=point
 assert b.snapshot(forecast).connectivity_ratio==1.

