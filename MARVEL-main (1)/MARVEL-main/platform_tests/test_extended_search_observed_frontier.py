from pathlib import Path
import json,types,importlib.util
import numpy as np
from utils.geometry import GeometryFrame
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.extended_travel import extended_observed_next_hop
BASE=Path(__file__).resolve().parents[1]/'artifacts/extended_observed_frontier_probe_9165_20261011_attempt1'
import integrations.gppo.extended_travel as m
def state():
 f=json.loads((BASE/'captured_frame7.json').read_text());meta=f['observed_recovery']['frame'];uid=10
 rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']))
 rec.observe({int(k):v for k,v in f['selected_own_observations'].items()});rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']}
 g=f['observed_graphs_diagnostic_only']['10'];coords=np.asarray(g['node_coords']);adj=np.ones((len(coords),len(coords)))
 for a,b in g['zero_adjacency_edges']:adj[a,b]=0
 agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.asarray(g['neighbor_indices']))
 position=rec.observed_positions[uid].copy();goal=np.asarray(f['public_heat_positions']['2']);details={}
 assert extended_observed_next_hop(uid,position,goal,agent,rec,route_details=details,goal_radius=3.5) is None
 return uid,position,goal,agent,rec,details

def test_saved_failure_has_legal_known_prefix_without_goal_certificate():
 uid,p,g,a,r,d=state();hop,detail=m.observed_frontier_next_hop(uid,p,g,a,r,d)
 assert np.allclose(hop[0],[8,20]);assert r.segment_safe(p,hop[0])
 assert detail['observed_prefix_only'] and not detail['original_search_goal_connected'] and not detail['unknown_tail_executed']
 assert all(r.segment_safe(x,y) for x,y in zip([p]+detail['route_waypoints'],detail['route_waypoints']))
def test_stale_pose_cannot_issue_prefix():
 uid,p,g,a,r,d=state();r.observed_positions[uid]=p+np.array([.1,0]);assert m.observed_frontier_next_hop(uid,p,g,a,r,d) is None
def test_unknown_cells_are_not_executable():
 uid,p,g,a,r,d=state();r.cells={};assert m.observed_frontier_next_hop(uid,p,g,a,r,d) is None
def test_public_hazard_filter_cannot_be_bypassed():
 uid,p,g,a,r,d=state();assert m.observed_frontier_next_hop(uid,p,g,a,r,d,segment_filter=lambda start,end:False) is None
def test_no_progress_frontier_returns_exploration_fallback():
 uid,p,g,a,r,d=state();d['observed_destination']=p.tolist();assert m.observed_frontier_next_hop(uid,p,g,a,r,d) is None
def test_known_full_goal_route_is_not_replaced():
 uid,p,g,a,r,d=state();d['route_goal_connected']=True;assert m.observed_frontier_next_hop(uid,p,g,a,r,d) is None

import integrations.gppo.scheduler as module
def scheduler_state():
 uid,p,g,a,r,d=state()
 s=module.GPPOTaskScheduler.__new__(module.GPPOTaskScheduler)
 s._search_recovery=r;s.search_reach_radius=3.5
 s.runtime=types.SimpleNamespace(obstacles=types.SimpleNamespace(frame=r.frame),current_step=7)
 s._peripheral_search_target=lambda *args:None
 robot=types.SimpleNamespace(robot_id=uid,position=p,heading=148.495)
 return s,robot,g,a,r

def test_search_override_chooses_real_observed_prefix():
 s,robot,g,a,r=scheduler_state()
 goal,heading=s._search_action(robot,g,a,None,(np.array([16.,32.]),80.),route_goal_radius=3.5,allow_observed_frontier=True)
 assert r.segment_safe(robot.position,goal)
 assert np.linalg.norm(goal-g)<np.linalg.norm(robot.position-g)
 assert goal[1]<robot.position[1]
 assert abs((heading-265.637)%360)<1 or abs((heading-265.637)%360-360)<1

def test_relay_default_remains_existing_exploration():
 s,robot,g,a,r=scheduler_state()
 goal,heading=s._search_action(robot,g,a,None,(np.array([16.,32.]),80.),route_goal_radius=3.5)
 assert goal[1]>robot.position[1]

def test_native_returns_frozen_route_and_original_fallback():
 s,robot,g,a,r=scheduler_state();s._search_recovery=None
 base=(np.array([16.,32.]),80.)
 assert s._search_action(robot,g,a,None,base,allow_observed_frontier=True) is base
 waypoint=np.array([12.,20.]);goal,heading=s._search_action(robot,g,a,(waypoint,2),base,allow_observed_frontier=True)
 assert np.array_equal(goal,waypoint)

def test_unknown_map_retains_legal_existing_fallback():
 s,robot,g,a,r=scheduler_state();r.cells={}
 goal,heading=s._search_action(robot,g,a,None,(np.array([16.,32.]),80.),route_goal_radius=3.5,allow_observed_frontier=True)
 assert np.array_equal(goal,robot.position)
