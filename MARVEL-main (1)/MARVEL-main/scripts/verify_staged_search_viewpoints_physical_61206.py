"""Controlled continuous route fixture, never a production planner."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.simulation_runtime import SimulationRuntime
from utils.geometry import GeometryFrame
from utils.target_detector import TargetDetector
from integrations.gppo.search_recovery import ObservedSearchRecovery
from integrations.gppo.safety_layer import SharedHazardAdapter
BASE=ROOT/'artifacts/extended_dynamic_search_cycle_61206_20261010_attempt1';OUT=BASE/'staged_module_physical_motion';OUT.mkdir(exist_ok=False)
f=next(x for x in json.loads((BASE/'captured_frames.json').read_text()) if x['step']==400);oracle=json.loads((BASE/'peripheral_sensor_oracle.json').read_text());choice=oracle['sensor_visible_candidates'][0];cfg=json.loads((ROOT/'artifacts/extended_gppo_joint_batch_20261010_attempt8/episode_010/gppo/config.json').read_text());poses=f['physical_poses_diagnostic_only']
for group in cfg['robots']:
 lo,hi=group['id_range'];group['config']['initial_positions']=[poses[str(uid)]['position'] for uid in range(lo,hi+1)]
for warning in cfg['dynamic_obstacles']:
 warning['spawn_step']=0;warning['params']['initial_radius']=warning['params']['max_radius']
runtime=SimulationRuntime(cfg);runtime.reset();uid=28
for robot in runtime.robots:robot.heading=poses[str(robot.robot_id)]['heading']
robot=next(r for r in runtime.robots if r.robot_id==uid);index=runtime.robots.index(robot);meta=f['observed_recovery']['frame'];rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']));rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']};hazards=SharedHazardAdapter(runtime,warning_margin=5,safe_margin=8)
route=[np.asarray(p,dtype=float) for p in choice['route_waypoints']];goal=np.asarray(f['public_heat_positions']['2']);target=oracle['derived_hidden_target_oracle_only'];detector=TargetDetector({'oracle_only':[{'x':target[0],'y':target[1]}]});obs=runtime._get_observations();rows=[];collisions=[];shield_rejections=0;minimum_peer_separation=1e9;leg=0;arrived=False;detected=False
sources=list((ROOT/'utils').rglob('*.py'))+list((ROOT/'integrations').rglob('*.py'));hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
from utils.sensor_models import IdealSensor
sensor=IdealSensor(cfg['sensor']['params'],frame=rec.frame);checked=set(map(tuple,json.loads((BASE/'own_checked_free_cells_to_step400.json').read_text())));g=f['observed_graphs_diagnostic_only'][str(uid)];coords=np.asarray(g['node_coords']);adj=np.ones((len(coords),len(coords)))
for a,b in g['zero_adjacency_edges']:adj[a,b]=0
import types
from integrations.gppo.extended_travel import extended_observed_next_hop
agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=g['current_index'],neighbor_indices=np.asarray(g['neighbor_indices']))
import importlib.util
stage=ROOT/'artifacts/extended_public_peripheral_search_design_20261011_attempt1/search_viewpoints.py';spec=importlib.util.spec_from_file_location('integrations.gppo.search_viewpoints',stage);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);draft_views=module.ObservedSearchViewpoints(rec.frame,sensor_range=10,fov=120,prior_radius=5.5);draft_views.checked_cells_by_heat[2]=checked.copy()
def select_support(rec,agent,goal,checked,sensor,segment_filter):
 viewpoint=draft_views.select(uid,2,goal,agent,rec,segment_filter)
 if viewpoint is None:return None
 details={};hop=extended_observed_next_hop(uid,rec.observed_positions[uid],viewpoint.position,agent,rec,segment_filter=segment_filter,route_details=details)
 if hop is None:return None
 return (viewpoint.potential_new_cells/max(1,viewpoint.observed_route_length_m),viewpoint.position.tolist(),viewpoint.potential_new_cells,details['route_waypoints'])
route=[];leg=0;decisions=[];detected=False;no_support=False
for tick in range(1000):
 rec.observe(obs);draft_views.observe(obs,{2:uid})
 if detector.detect('oracle_only',obs,[uid]):detected=True;break
 while leg<len(route) and np.linalg.norm(robot.position-route[leg])<=1e-8:leg+=1
 if leg==len(route):
  filtered=lambda a,b:not hazards.segment_intersects_warning_buffer(a,b,step=runtime.current_step)
  agent.current_index=int(np.argmin(np.linalg.norm(agent.node_coords-robot.position,axis=1)));agent.neighbor_indices=np.flatnonzero(agent.adjacent_matrix[agent.current_index]==0)
  choice=select_support(rec,agent,goal,checked,sensor,filtered)
  if choice is None:no_support=True;break
  score,point,gain,waypoints=choice;route=[np.asarray(p,dtype=float) for p in waypoints];leg=0;decisions.append(dict(tick=tick,selected_public_support=point,unchecked_prior_gain=gain,public_score=score,route=waypoints))
  while leg<len(route) and np.linalg.norm(robot.position-route[leg])<=1e-8:leg+=1
  if leg==len(route):continue
 waypoint=route[leg];heading=float(np.degrees(np.arctan2(*(goal-robot.position)[::-1]))%360.)
 assert rec.segment_safe(robot.position,waypoint)
 assert not hazards.segment_intersects_warning_buffer(robot.position,waypoint,step=runtime.current_step)
 actions=[(r.position.copy(),r.heading) for r in runtime.robots];actions[index]=(waypoint,heading);obs,info=runtime.step(actions);collisions.extend(info['collisions']);shield_rejections+=int(obs[uid]['motion_feedback'].get('shield_rejected',False));minimum_peer_separation=min(minimum_peer_separation,min(np.linalg.norm(robot.position-other.position) for other in runtime.robots if other.robot_id!=uid));rows.append(dict(tick=tick+1,position=robot.position.tolist(),heading=float(robot.heading),leg=leg,waypoint=waypoint.tolist(),feedback=obs[uid]['motion_feedback']))
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
result=dict(seconds=len(rows)*runtime.dt,position=robot.position.tolist(),heading=float(robot.heading),target_detected_by_actual_sensor=detected,collisions=collisions,shield_rejections=shield_rejections,minimum_peer_separation_m=float(minimum_peer_separation),selected_supports=decisions,no_support=no_support,selector_hidden_target_input=False,selector_truth_map_input=False,unknown_visibility_is_potential_only=True,all_segments_observed_FREE=True,all_segments_public_warning_clear=True,physical_radius_m=.2,shield_enabled=True,source_stable=True,staged_module_sha256=hashlib.sha256(stage.read_bytes()).hexdigest(),scope='Independent offline bounded observed-only prototype using existing graph and Dijkstra, physical runtime and sensors; other UAVs held. Not integrated production behavior and not a full natural cooperative mission.',acceptance_claim=False,passed=detected and not collisions)
(OUT/'trajectory.json').write_text(json.dumps(rows));(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert result['passed']
