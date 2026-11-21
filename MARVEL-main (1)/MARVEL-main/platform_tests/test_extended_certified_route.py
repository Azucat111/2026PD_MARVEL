from types import SimpleNamespace as NS
import numpy as np
from utils.geometry import GeometryFrame
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.observed_graph import observed_next_hop

def state():
 rec=ObservedSearchRecovery(GeometryFrame(-1,-1,1,10,10,'extended'))
 rec.cells={(x,y):0 for x in range(10) for y in range(10)};rec.cells[2,2]=1;rec.observed_positions[0]=np.array([0.,0.])
 agent=NS(node_coords=np.array([[0.,0.],[0.,4.],[4.,0.],[4.,4.]]),adjacent_matrix=np.zeros((4,4)),current_index=0,neighbor_indices=np.arange(4))
 return rec,agent

def test_execution_reuses_certified_route_instead_of_false_free_bfs_shortcut():
 from integrations.gppo.extended_travel import extended_observed_next_hop
 rec,agent=state();goal=np.array([4.,4.]);old,_=observed_next_hop(agent,goal)
 assert not rec.segment_safe([0.,0.],old)
 point,index=extended_observed_next_hop(0,[0.,0.],goal,agent,rec)
 np.testing.assert_array_equal(point,[0.,4.]);assert index==1 and rec.segment_safe([0.,0.],point)

def test_route_can_start_with_real_pose_attachment_away_from_goal():
 from integrations.gppo.extended_travel import extended_observed_next_hop
 rec,agent=state();rec.observed_positions[0]=np.array([.5,.5])
 point,index=extended_observed_next_hop(0,[.5,.5],[4.,4.],agent,rec)
 assert rec.segment_safe([.5,.5],point) and np.linalg.norm(point-[.5,.5])>.1

def test_missing_or_mismatched_own_sensor_cannot_issue_executable_route():
 from integrations.gppo.extended_travel import extended_observed_next_hop
 rec,agent=state();rec.observed_positions.clear()
 assert extended_observed_next_hop(0,[0,0],[4,4],agent,rec) is None
 rec.observed_positions[0]=np.array([1.,0.])
 assert extended_observed_next_hop(0,[0,0],[4,4],agent,rec) is None

def test_existing_segment_filter_applies_to_next_hop_path():
 from integrations.gppo.extended_travel import extended_observed_next_hop
 rec,agent=state()
 point,index=extended_observed_next_hop(0,[0,0],[4,4],agent,rec,segment_filter=lambda a,b: not np.array_equal(b,[0.,4.]) and not np.array_equal(a,[0.,4.]))
 np.testing.assert_array_equal(point,[4.,0.]);assert index==2

def test_peer_recovery_progress_follows_local_route_leg_not_remote_anchor():
 rec,agent=state();rec.cells[2,2]=0;rec.nearby_positions[0]=[np.array([2.,0.])]
 point=rec.select(0,[0,0],[0,8],agent,[4,0],[0,4],progress_target=[4,0])
 assert point[0]>.5 and abs(point[1])<.5 and rec.segment_safe([0,0],point)


def test_eight_metre_graph_edge_is_a_supported_target_without_teleportation():
 from integrations.gppo.extended_travel import extended_observed_next_hop
 from utils.dynamics_models import KinematicSimple
 rec,agent=state();rec.cells[2,2]=0
 agent.node_coords=np.array([[0.,0.],[0.,8.]]);agent.adjacent_matrix=np.zeros((2,2));agent.neighbor_indices=np.array([0,1])
 point,index=extended_observed_next_hop(0,[0,0],[0,8],agent,rec)
 np.testing.assert_array_equal(point,[0,8])
 proposal,_=KinematicSimple().step(np.array([0.,0.]),point,0.,90.,dt=.1)
 assert np.linalg.norm(proposal['position'])<=.1+1e-9


def test_unknown_residual_attachment_is_not_an_executable_goal_route():
 from integrations.gppo.extended_travel import extended_observed_next_hop,extended_observed_travel_time_matrix
 rec,agent=state();rec.cells[2,2]=0
 rec.observed_positions[0]=np.array([0.,1.])
 agent.node_coords=np.array([[0.,0.]]);agent.adjacent_matrix=np.zeros((1,1));agent.current_index=0;agent.neighbor_indices=np.array([0])
 task=NS(subtask_id=1,position=[0.,8.]);uav=NS(uav_id=0,position=[0.,1.],velocity=1.)
 assert extended_observed_travel_time_matrix([uav],[task],{0:agent},rec)[0,0]==9.
 details={}
 assert extended_observed_next_hop(0,[0.,1.],[0.,8.],agent,rec,route_details=details) is None
 assert details['route_goal_connected'] is False

def test_observed_goal_connector_keeps_backward_detour_executable():
 from integrations.gppo.extended_travel import extended_observed_next_hop
 rec,agent=state();rec.cells[2,2]=0;rec.observed_positions[0]=np.array([0.,1.])
 agent.node_coords=np.array([[0.,0.],[4.,0.],[4.,8.]])
 agent.adjacent_matrix=np.array([[0,0,1],[0,0,0],[1,0,0]])
 agent.current_index=0;agent.neighbor_indices=np.array([0])
 details={}
 hop=extended_observed_next_hop(0,[0.,1.],[4.,8.5],agent,rec,route_details=details)
 np.testing.assert_array_equal(hop[0],[0.,0.])
 assert details['route_goal_connected'] and rec.segment_safe([4.,8.],[4.,8.5])


def test_extended_scheduler_does_not_prefer_uncertified_upstream_hop():
 from integrations.gppo.scheduler import GPPOTaskScheduler
 rec,agent=state();rec.cells[2,2]=0;rec.observed_positions[0]=np.array([0.,1.])
 agent.node_coords=np.array([[0.,0.]]);agent.adjacent_matrix=np.zeros((1,1));agent.current_index=0;agent.neighbor_indices=np.array([0])
 captured={}
 def select(*args,**kwargs):
  captured['preferred']=args[4];return np.array([0.,1.25])
 rec.select=select
 sched=GPPOTaskScheduler.__new__(GPPOTaskScheduler);sched._search_recovery=rec;sched.runtime=NS(current_step=0,obstacles=NS(frame=rec.frame));sched.safety=None
 robot=NS(robot_id=0,position=np.array([0.,1.]),heading=90.)
 sched._search_action(robot,[0.,8.],agent,(np.array([0.,0.]),0),(robot.position,90.))
 assert captured['preferred'] is None


def test_uncertified_goal_recovery_prefers_legal_existing_exploration():
 rec,agent=state();rec.cells[2,2]=0
 target=rec.select(0,[0,0],[0,8],agent,None,[4,0],base_heading=0.,prefer_base=True,fallback_preferred=[0,4])
 np.testing.assert_array_equal(target,[4,0])
 assert rec.prefix_headings[0]==0.

def test_exploration_preference_does_not_bypass_collision_gate():
 rec,agent=state()
 target=rec.select(0,[0,0],[4,4],agent,None,[4,4],prefer_base=True)
 assert rec.segment_safe([0,0],target) and not np.array_equal(target,[4,4])


def test_exploration_prefix_precedes_goal_greedy_neighbor():
 rec,agent=state();rec.cells[2,2]=0;rec.cells[0,3]=1
 rec.observed_positions[0]=np.array([0.,1.])
 target=rec.select(0,[0,1],[8,1],agent,None,[0,8],base_heading=90.,prefer_base=True)
 assert target[0]==0. and 1.<target[1]<=2.
 assert rec.prefix_headings[0]==90. and rec.segment_safe([0,1],target)


def test_exploration_peer_avoidance_progresses_to_base_not_remote_anchor():
 rec,agent=state();rec.cells[2,2]=0
 rec.observed_positions[0]=np.array([4.,3.5]);rec.nearby_positions[0]=[np.array([4.,3.])]
 agent.node_coords=np.array([[4.,3.5],[4.,8.],[4.,0.]]);agent.neighbor_indices=np.arange(3)
 target=rec.select(0,[4,3.5],[4,8],agent,None,[4,0],base_heading=270.,prefer_base=True)
 assert not np.allclose(target,[4,3.5])
 # The first straight move must create lateral clearance at this boundary.
 assert abs(target[0]-4.)>.05
 assert rec.segment_safe([4,3.5],target)
 delta=target-np.array([4.,3.5]);t=np.clip(np.dot(np.array([0.,-.5]),delta)/np.dot(delta,delta),0.,1.)
 assert np.linalg.norm(np.array([4.,3.5])+t*delta-np.array([4.,3.]))>=.5-1e-9


def test_exploration_peer_recovery_completes_continuous_legal_moves():
 from utils.dynamics_models import KinematicSimple
 rec,agent=state();rec.cells[2,2]=0;rec.nearby_positions[0]=[np.array([4.,3.])]
 agent.node_coords=np.array([[4.,3.5],[4.,8.],[4.,0.]]);agent.neighbor_indices=np.arange(3)
 position=np.array([4.,3.5]);heading=270.;dynamics=KinematicSimple()
 for mission in range(40):
  rec.observed_positions[0]=position.copy()
  target=rec.select(0,position,[4,8],agent,None,[4,0],base_heading=270.,prefer_base=True)
  assert rec.segment_safe(position,target)
  for tick in range(10):
   proposal,_=dynamics.step(position,target,heading,rec.prefix_headings.get(0,270.),dt=.1)
   assert np.linalg.norm(proposal['position']-[4,3])>=.5-1e-9
   position=proposal['position'];heading=proposal['heading']
  if np.linalg.norm(position-[4,0])<=.1:break
 assert np.linalg.norm(position-[4,0])<=.1


def test_search_service_region_can_be_reached_when_heat_point_is_occupied():
 from integrations.gppo.extended_travel import extended_observed_next_hop
 rec,agent=state();rec.cells[2,2]=0;rec.cells[1,5]=1
 rec.observed_positions[0]=np.array([4.,0.])
 agent.node_coords=np.array([[4.,0.],[4.,4.]]);agent.adjacent_matrix=np.zeros((2,2));agent.neighbor_indices=np.arange(2)
 assert not rec.segment_safe([0,4],[0,4])
 assert extended_observed_next_hop(0,[4,0],[0,4],agent,rec) is None
 details={}
 point,index=extended_observed_next_hop(0,[4,0],[0,4],agent,rec,route_details=details,goal_radius=2.)
 assert rec.segment_safe([4,0],point) and np.linalg.norm(point-[0,4])<=2.+1e-9
 assert details['route_goal_radius_m']==2. and details['route_goal_connected']
