"""Read-only saved-state probe: known observed path only, no unknown tail execution."""
from pathlib import Path
import json,sys,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.simulation_runtime import SimulationRuntime
from utils.geometry import GeometryFrame
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.extended_travel import extended_observed_next_hop
BASE=ROOT/'artifacts/extended_observed_frontier_probe_9165_20261011_attempt1';PAIR=ROOT/'artifacts/extended_integrated_candidate_9165_30_static_20261011_attempt1/gppo'
f=json.loads((BASE/'captured_frame7.json').read_text());uid=10;goal=np.asarray(f['public_heat_positions']['2']);meta=f['observed_recovery']['frame']
rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']))
rec.observe({int(k):v for k,v in f['selected_own_observations'].items()});rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']}
rec.failed={int(k):[(np.asarray(t),np.asarray(p)) for t,p in entries] for k,entries in f['observed_recovery']['failed'].items()}
position=rec.observed_positions[uid];g=f['observed_graphs_diagnostic_only'][str(uid)];coords=np.asarray(g['node_coords']);adj=np.ones((len(coords),len(coords)))
for a,b in g['zero_adjacency_edges']:adj[a,b]=0
agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.asarray(g['neighbor_indices']))
detail={};original=extended_observed_next_hop(uid,position,goal,agent,rec,route_details=detail,goal_radius=3.5)
endpoint=detail.get('observed_destination');known_detail={};known=None
if endpoint is not None:known=extended_observed_next_hop(uid,position,endpoint,agent,rec,route_details=known_detail,goal_radius=0.)

sys.path.insert(0,str(BASE))
from test_scheduler_frontier_draft import scheduler_state
s,robot_stub,goal,agent,rec=scheduler_state()
cfg=json.loads((PAIR/'config.json').read_text());poses=f['physical_poses_diagnostic_only']
for group in cfg['robots']:
 lo,hi=group['id_range'];group['config']['initial_positions']=[poses[str(i)]['position'] for i in range(lo,hi+1)]
runtime=SimulationRuntime(cfg);runtime.reset();runtime.current_step=f['step'];runtime.obstacles.step(runtime.current_step,runtime.dt)
for robot in runtime.robots:robot.heading=poses[str(robot.robot_id)]['heading']
robot=next(r for r in runtime.robots if r.robot_id==uid);assert np.allclose(robot.position,position,rtol=0,atol=1e-9)
heading=float(np.degrees(np.arctan2(*(known[0]-position)[::-1]))%360.)
actions=[(r.position.copy(),r.heading) for r in runtime.robots];index=next(i for i,r in enumerate(runtime.robots) if r.robot_id==uid);actions[index]=(known[0],heading)

trace=[]
start=robot.position.copy()
for tick in range(130):
 s.runtime.current_step=7+tick//10
 robot_stub.position=robot.position.copy();robot_stub.heading=robot.heading
 rec.observed_positions[uid]=robot.position.copy()
 command=s._search_action(robot_stub,goal,agent,None,(np.array([16.,32.]),80.),route_goal_radius=3.5,allow_observed_frontier=True)
 # Measured own pose changes; observed cells remain the saved public map.
 actions=[(r.position.copy(),r.heading) for r in runtime.robots];actions[index]=command
 previous=robot.position.copy();_,info=runtime.step(actions)
 trace.append(dict(tick=tick,position=robot.position.tolist(),waypoint=np.asarray(command[0]).tolist(),heading=command[1],moved=float(np.linalg.norm(robot.position-previous)),collisions=info['collisions']))
 if np.linalg.norm(robot.position-np.array([8.,12.]))<.15:break
result=dict(start=start.tolist(),end=robot.position.tolist(),goal=goal.tolist(),distance_decrease=float(np.linalg.norm(start-goal)-np.linalg.norm(robot.position-goal)),trace=trace,other_uavs_held=True,truth_map_decision_input=False,physical_oracle_only=True,unknown_tail_executed=False,acceptance_claim=False)
with (BASE/'scheduler_prefix_physical_validation.json').open('x') as stream:json.dump(result,stream,indent=2)
assert all(not x['collisions'] for x in trace)
assert result['distance_decrease']>10
print(json.dumps({k:v for k,v in result.items() if k!='trace'}))
