"""Captured Safety residual diagnosis; observed graph/public hazards decide, truth only checks motion."""
from pathlib import Path
import json,sys,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.simulation_runtime import SimulationRuntime
from utils.geometry import GeometryFrame
from integrations.gppo.search_recovery import ObservedSearchRecovery
import importlib.util
stage=ROOT/'artifacts/extended_integrated_candidate_design_20261011_attempt1'
spec=importlib.util.spec_from_file_location('integrations.gppo.extended_safety_motion',stage/'extended_safety_motion.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
from integrations.gppo.extended_safety_motion import ObservedMotionSafetyLayer
from integrations.gppo.extended_travel import extended_observed_travel_time_matrix
BASE=ROOT/'artifacts/extended_safety_residual_9164_20261011_attempt1';PAIR=ROOT/'artifacts/extended_catalog_candidate_9164_60_dynamic_20261010_attempt1/gppo';f=next(x for x in json.loads((BASE/'captured_frames_v2.json').read_text()) if x['step']==400);cfg=json.loads((PAIR/'config.json').read_text());poses=f['physical_poses_diagnostic_only']
for group in cfg['robots']:
 lo,hi=group['id_range'];group['config']['initial_positions']=[poses[str(uid)]['position'] for uid in range(lo,hi+1)]
runtime=SimulationRuntime(cfg);runtime.reset();runtime.current_step=400;runtime.obstacles.step(400,runtime.dt)
for robot in runtime.robots:robot.heading=poses[str(robot.robot_id)]['heading']
uid=0;robot=next(r for r in runtime.robots if r.robot_id==uid);meta=f['observed_recovery']['frame'];rec=ObservedSearchRecovery(GeometryFrame(*meta['origin'],meta['cell_size'],meta['width_cells'],meta['height_cells'],meta['mode']));rec.observe({int(k):v for k,v in f['selected_own_observations'].items()});rec.cells={(x,y):v for x,y,v in f['observed_recovery']['cells']};rec.observed_positions[uid]=robot.position.copy();rec.nearby_positions[uid]=[np.asarray(v['position']) for k,v in poses.items() if int(k)!=uid and np.linalg.norm(np.asarray(v['position'])-robot.position)<=10];rec.failed={int(k):[(np.asarray(t),np.asarray(p)) for t,p in entries] for k,entries in f['observed_recovery']['failed'].items()}
g=f['observed_graphs_diagnostic_only']['53'];coords=np.asarray(g['node_coords']);adj=np.ones((len(coords),len(coords)))
for a,b in g['zero_adjacency_edges']:adj[a,b]=0
agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=int(np.argmin(np.linalg.norm(coords-robot.position,axis=1))),neighbor_indices=np.flatnonzero(np.linalg.norm(coords-robot.position,axis=1)<=6));layer=ObservedMotionSafetyLayer(runtime,recovery=rec);layer._marvel_agents={uid:agent};old=layer.select_action(robot,step=400);risk=layer.hazards.risk_at(robot.position,step=400)
filtered=lambda a,b:not layer.hazards.segment_intersects_warning_buffer(a,b,step=400)
goals=[p for p in coords if np.linalg.norm(p-robot.position)<=20 and rec.segment_safe(p,p) and layer.hazards.is_safe(p,step=400)];details={};tasks=[types.SimpleNamespace(subtask_id=i,position=p) for i,p in enumerate(goals)]
if tasks:extended_observed_travel_time_matrix([types.SimpleNamespace(uav_id=uid,position=robot.position,velocity=1.)],tasks,{uid:agent},prefix_controller=rec,diagnostics=details,segment_filter=filtered,include_routes=True)
routes=[]
for i,p in enumerate(goals):
 row=details[(i,uid)]
 if not row['route_goal_connected'] or not row['route_first_segment_certified']:continue
 points=[robot.position]+[np.asarray(v) for v in row['route_waypoints']];cost=sum(float(np.linalg.norm(b-a)) for a,b in zip(points,points[1:]));routes.append(dict(goal=p.tolist(),known_route_m=cost,next_hop=row['route_next_waypoint'],route=row['route_waypoints']))
routes.sort(key=lambda r:(r['known_route_m'],r['goal']))


OUT=BASE/'controlled_staged_safety_action';OUT.mkdir(exist_ok=False)
from integrations.gppo.safety_layer import SafetyRecord
layer.active_uav_ids={uid};layer.safe_counter={uid:0};record=SafetyRecord(uid,'warning',10);layer.records=[record];layer.active_record_by_uav={uid:record}
obs=runtime._get_observations();rec.observe(obs);rec.cells.update({(x,y):v for x,y,v in f['observed_recovery']['cells']});rows=[];commands=[];collisions=[];rejections=0;minimum_peer=1e9
index=runtime.robots.index(robot)
def before_tick(actions):
 result=list(actions)
 if uid in layer.active_uav_ids:
  action=layer.select_action(robot,step=runtime.current_step);result[index]=action
  assert rec.segment_safe(robot.position,action[0]);assert not layer.hazards.segment_intersects_warning_buffer(robot.position,action[0],step=runtime.current_step)
  commands.append(dict(step=runtime.current_step,position=robot.position.tolist(),waypoint=action[0].tolist(),heading=float(action[1])))
 return result
runtime.high_level_scheduler=types.SimpleNamespace(before_physics_tick=before_tick)
for mission in range(10):
 actions=[(r.position.copy(),float(r.heading)) for r in runtime.robots]
 for tick in range(runtime._mission_step_interval()):
  obs,info=runtime.step(actions);rec.observe(obs);collisions.extend(info['collisions']);rejections+=int(obs[uid]['motion_feedback'].get('shield_rejected',False));minimum_peer=min(minimum_peer,min(np.linalg.norm(robot.position-r.position) for r in runtime.robots if r.robot_id!=uid));rows.append(dict(step=runtime.current_step,position=robot.position.tolist(),heading=float(robot.heading),clearance=float(layer.hazards.risk_at(robot.position,step=runtime.current_step).clearance)))
 layer.post_motion_update(None,runtime.current_step)
 if uid not in layer.active_uav_ids:break
summary=dict(seconds=len(rows)*runtime.dt,position=robot.position.tolist(),collisions=collisions,shield_rejections=rejections,minimum_peer_separation_m=float(minimum_peer),safety_released=uid not in layer.active_uav_ids,successful_escapes=layer.successful_escapes,escaped_step=record.escaped_step,no_fly_violations=layer.no_fly_violations,original_safe_margin=layer.hazards.safe_margin,stable_release_steps=layer.stable_release_steps,physical_radius_m=.2,selector_truth_map_input=False,scope='Controlled real sixty-UAV captured pose fixture, other59 hold, new own sensor/peer observations, original stable release state machine. Not exact historical own graph replay or natural full mission.',acceptance_claim=False,passed=uid not in layer.active_uav_ids and not collisions and not rejections)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'commands.json').write_text(json.dumps(commands,indent=2));(OUT/'trajectory.json').write_text(json.dumps(rows));print(json.dumps(summary));assert summary['passed']
