"""Captured Safety residual diagnosis; observed graph/public hazards decide, truth only checks motion."""
from pathlib import Path
import json,sys,types
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.simulation_runtime import SimulationRuntime
from utils.geometry import GeometryFrame
from integrations.gppo.search_recovery import ObservedSearchRecovery
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

OUT=BASE/'controlled_peer_aware_local_route';OUT.mkdir(exist_ok=False)

# Diagnostic bounded local graph uses only captured labels/public hazards/fresh sensed peers.
obs=runtime._get_observations();rec.observe(obs);rec.cells.update({(x,y):v for x,y,v in f['observed_recovery']['cells']})
peers=rec.nearby_positions.get(uid,[])
def peers_clear(a,b):
 d=b-a;s=float(np.dot(d,d))
 return all(np.linalg.norm(a+(float(np.clip(np.dot(p-a,d)/s,0,1)) if s>1e-20 else 0)*d-p)>=min(.5,np.linalg.norm(a-p))-1e-9 for p in peers)
def local_clear(a,b):return filtered(a,b) and peers_clear(a,b)
local=[robot.position+np.array([x,y])*.25 for x in range(-8,9) for y in range(-8,9) if x*x+y*y<=64]
local=[p for p in local if rec.segment_safe(p,p) and filtered(p,p) and peers_clear(p,p)]
coords=np.array(local);dist=np.linalg.norm(coords[:,None]-coords[None,:],axis=2);adj=np.where(dist<=.36,0,1);current=int(np.argmin(np.linalg.norm(coords-robot.position,axis=1)));agent=types.SimpleNamespace(node_coords=coords,adjacent_matrix=adj,current_index=current,neighbor_indices=np.flatnonzero(np.linalg.norm(coords-robot.position,axis=1)<=.36))
goals=[p for p in coords if layer.hazards.is_safe(p,step=400)];details={}
extended_observed_travel_time_matrix([types.SimpleNamespace(uav_id=uid,position=robot.position,velocity=1.)],[types.SimpleNamespace(subtask_id=i,position=p) for i,p in enumerate(goals)],{uid:agent},prefix_controller=rec,diagnostics=details,segment_filter=local_clear,include_routes=True)
routes=[]
for i,p in enumerate(goals):
 r=details[(i,uid)]
 if r['route_first_segment_certified'] and r['route_goal_connected']:
  pts=[robot.position]+[np.asarray(v) for v in r['route_waypoints']];cost=sum(float(np.linalg.norm(b-a)) for a,b in zip(pts,pts[1:]));routes.append(dict(goal=p.tolist(),known_route_m=cost,route=r['route_waypoints'],next_hop=r['route_next_waypoint']))
routes.sort(key=lambda r:(r['known_route_m'],r['goal']))

assert routes
route=[np.asarray(p) for p in routes[0]['route']];rows=[];commands=[];collisions=[];rejections=0;leg=0;obs=runtime._get_observations();rec.observe(obs);rec.cells.update({(x,y):v for x,y,v in f['observed_recovery']['cells']})
minimum_peer=1e9;start=robot.position.copy();safe_ticks=0
for mission in range(30):
 while leg<len(route) and np.linalg.norm(robot.position-route[leg])<=.1:leg+=1
 point=robot.position.copy() if leg==len(route) else route[leg]
 delta=point-robot.position;heading=float(robot.heading) if np.linalg.norm(delta)<1e-9 else float(np.degrees(np.arctan2(delta[1],delta[0]))%360)
 assert rec.segment_safe(robot.position,point)
 assert filtered(robot.position,point)
 actions=[(r.position.copy(),float(r.heading)) for r in runtime.robots];actions[runtime.robots.index(robot)]=(point,heading)
 commands.append(dict(step=runtime.current_step,position=robot.position.tolist(),waypoint=point.tolist(),heading=heading))
 for tick in range(runtime._mission_step_interval()):
  obs,info=runtime.step(actions);rec.observe(obs);collisions.extend(info['collisions']);rejections+=int(obs[uid]['motion_feedback'].get('shield_rejected',False));minimum_peer=min(minimum_peer,min(np.linalg.norm(robot.position-r.position) for r in runtime.robots if r.robot_id!=uid));safe=layer.hazards.is_safe(robot.position,step=runtime.current_step);safe_ticks=safe_ticks+1 if safe else 0;rows.append(dict(step=runtime.current_step,position=robot.position.tolist(),heading=float(robot.heading),clearance=float(layer.hazards.risk_at(robot.position,step=runtime.current_step).clearance),safe=bool(safe)))
 if safe_ticks>=20:break
summary=dict(start=start.tolist(),end=robot.position.tolist(),seconds=len(rows)*runtime.dt,safe_ticks=safe_ticks,collisions=collisions,shield_rejections=rejections,minimum_peer_separation=float(minimum_peer),physical_radius=.2,shared_observed_route=routes[0],bounded_search_radius_m=2.,resolution_m=.25,peer_aware_route=True,historical_own_graph_missing=True,fresh_sensor_peer_evidence=True,selector_truth_map_input=False,scope='Controlled snapshot route validation, other 59 hold; not exact missing historical graph reconstruction or full mission acceptance.',passed=safe_ticks>=20 and not collisions and not rejections)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'commands.json').write_text(json.dumps(commands,indent=2));(OUT/'trajectory.json').write_text(json.dumps(rows));print(json.dumps(summary));assert summary['passed']
