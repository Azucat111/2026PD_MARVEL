from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from utils.simulation_runtime import SimulationRuntime
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.safety_layer import SharedHazardAdapter
from integrations.gppo.observed_graph import observed_next_hop
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('episode,uid',[(64,26),(112,29)])
def test_known_free_relay_exit_moves_with_missing_coarse_connector(episode,uid):
 # Break caught: falling back to a hold when observed FREE motion to the anchor exists.
 folder=ROOT/'artifacts/extended_gppo_joint_batch_20261012_attempt18'
 frame=next(x['frame'] for x in json.loads((folder/'readonly_static64_112_control_inputs.json').read_text()) if x['episode']==episode)
 runtime=SimulationRuntime(json.loads((folder/f'episode_{episode:03d}/gppo/config.json').read_text()));runtime.reset()
 for r in runtime.robots:
  p=frame['physical_poses_diagnostic_only'][str(r.robot_id)];r.position=np.asarray(p['position']);r.heading=p['heading'];r.travel_distance=0.
 runtime.current_step=frame['step'];robot=next(r for r in runtime.robots if r.robot_id==uid)
 rec=ObservedSearchRecovery(runtime.obstacles.frame);state=frame['observed_recovery'];rec.cells={(c,r):v for c,r,v in state['cells']};rec._authoritative_areal_grid=True
 rec.prefixes={int(k):np.asarray(v) for k,v in state['prefixes'].items()};rec.prefix_headings={int(k):v for k,v in state['prefix_headings'].items()};rec.anchors={int(k):np.asarray(v) for k,v in state['anchors'].items()};rec.observe({uid:frame['selected_own_observations'][str(uid)]})
 g=frame['observed_graphs_diagnostic_only'][str(uid)];coords=np.asarray(g['node_coords']);adj=np.ones((len(coords),len(coords)),dtype=np.int8)
 for a,b in g['zero_adjacency_edges']:adj[a,b]=0
 agent=SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.asarray(g['neighbor_indices']))
 scheduler=GPPOTaskScheduler.__new__(GPPOTaskScheduler);scheduler.runtime=runtime;scheduler._search_recovery=rec;scheduler._search_viewpoints=None;scheduler.safety=SimpleNamespace(hazards=SharedHazardAdapter(runtime))
 anchor=np.asarray(next(a['anchor'] for a in frame['public_relay_assignments'].values() if a['helper']==uid))
 action=scheduler._relay_action(robot,anchor,agent,observed_next_hop(agent,anchor),(robot.position.copy(),robot.heading))
 actions=[(r.position.copy(),r.heading) for r in runtime.robots];actions[next(i for i,r in enumerate(runtime.robots) if r.robot_id==uid)]=action
 old=robot.position.copy();_obs,info=runtime.step(actions)
 assert np.linalg.norm(robot.position-old)>.09, 'Relay held despite a known legal exit (verified separately with original runtime/Shield)'
 assert not info['collisions']

from integrations.gppo.relay_frontier import ObservedRelayFrontier
from utils.geometry import GeometryFrame

def simple_known_lane():
 rec=ObservedSearchRecovery(GeometryFrame.extended(width_cells=8,height_cells=8));rec.cells={(c,r):0 for c in range(1,7) for r in range(1,7)}
 for r in range(1,6):rec.cells[4,r]=1
 rec.observed_positions={0:np.array([2.5,2.5])};return rec

def test_known_route_detours_around_measured_wall():
 rec=simple_known_lane();route=ObservedRelayFrontier(rec).known_route(0,np.array([5.5,2.5]))
 assert route is not None
 assert np.allclose(route[-1],[5.5,2.5])
 assert any(point[1]>6. for point in route),'Measured wall requires going through the known top gap'

def test_unknown_top_gap_is_not_traversed():
 rec=simple_known_lane()
 for c in range(1,7):rec.cells.pop((c,6))
 assert ObservedRelayFrontier(rec).known_route(0,np.array([5.5,2.5])) is None

def test_occupied_anchor_peer_prevents_route():
 rec=simple_known_lane();rec.nearby_positions={0:[np.array([5.5,2.5])]}
 assert ObservedRelayFrontier(rec).known_route(0,np.array([5.5,2.5])) is None

def test_public_segment_filter_cannot_be_bypassed():
 rec=simple_known_lane()
 assert ObservedRelayFrontier(rec).known_route(0,np.array([5.5,2.5]),segment_filter=lambda a,b:False) is None

def test_cached_route_rechecks_new_occupied_goal():
 rec=simple_known_lane();planner=ObservedRelayFrontier(rec);assert planner.known_route(0,np.array([5.5,2.5])) is not None
 rec.cells[5,2]=1
 assert planner.known_route(0,np.array([5.5,2.5])) is None

def test_cached_route_rechecks_new_public_filter():
 rec=simple_known_lane();planner=ObservedRelayFrontier(rec);assert planner.known_route(0,np.array([5.5,2.5])) is not None
 assert planner.known_route(0,np.array([5.5,2.5]),segment_filter=lambda a,b:False) is None
