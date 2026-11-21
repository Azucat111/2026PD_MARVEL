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
result=dict(uid=uid,physical_step=400,origin=robot.position.tolist(),heading=float(robot.heading),public_radius=float(risk.radius),clearance=float(risk.clearance),warning_margin=layer.hazards.warning_margin,safe_margin=layer.hazards.safe_margin,inside_warning=risk.within_warning_margin,currently_released_safe=layer.hazards.is_safe(robot.position,step=400),old_reproduced_command=old[0].tolist(),old_reproduced_heading=float(old[1]),captured_command=f['actual_commands'][str(uid)],old_command_matches_capture=bool(np.allclose(old[0],f['actual_commands'][str(uid)]['waypoint'])),observed_safe_goals_within20m=len(goals),observed_full_warning_clear_routes=routes,hidden_or_truth_map_decision_input=False,capture_limitations=['Safety UAV 0 own graph and observation were not captured. Captured shared node graph from UAV 53 reattached using saved physical own pose; peer positions limited to 10m from saved fleet pose, not exact historical sensor record. Selection replay is diagnostic, not exact.'],continuous_path_and_peers_validation_pending=True,global_no_solution_claim=False,acceptance_claim=False)
with (BASE/'diagnosis.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(dict(origin=result['origin'],clearance=result['clearance'],old_command_matches_capture=result['old_command_matches_capture'],safe_routes=len(routes),best=routes[:2])))
